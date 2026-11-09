"""Measured CPU tuning in bounded time.

The tuner predicts first and benchmarks only what cannot be predicted:

1. **Gate.** The memory plan decides context, ubatch, weight repacking
   and KV type against the RAM free right now. A model that does not fit
   is never run.
2. **Predict.** Generation on a CPU is limited by memory bandwidth, so
   the speed is roughly ``efficiency x bandwidth / bytes read per token``.
3. **Calibrate.** One run with llama.cpp's defaults measures the real
   speed. It is the baseline the result must beat, and it sets the
   test sizes so each test takes a few seconds.
4. **Screen.** Thread counts x flash attention x repacking, all in one
   llama-bench process, one repetition each.
5. **Confirm.** The best third (successive halving), and anything
   within screening noise of the leader, runs again with three
   repetitions; untested thread counts next to the best are added.
6. **Prompt settings.** Prompt threads (``-tb``) come from the prompt
   tests; ubatch is checked when time allows.
7. **Drift and depth.** The defaults run again after the confirmation;
   the winner is compared with the average of both default runs, and a
   large difference between them is reported as a disturbance. Finally
   the winner runs once with a filled context.

A setting only replaces llama.cpp's default when it wins by more than
the measured noise; among settings within the noise, the one using less
memory and fewer threads is kept, leaving the PC responsive.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from statistics import median

from .benchmark import BenchRecord, SweepResult, run_sweep
from .compat import (
    NOT_FOUND_ADVICE, Capabilities, architecture_supported, detect_capabilities, update_llama_advice,
)
from .constraints import GateDecision, memory_budget, plan_memory
from .hardware import detect_hardware
from .memory import compute_model_for, kv_cache_bytes, repack_copy_bytes, tensor_sizes_of
from .model_inspector import inspect_model
from .heuristics import ABOVE_NORMAL
from .models import HardwareInfo, ModelInfo, SearchConfig
from .resources import cpu_load, measure_bandwidth, memory_status, power_status

# Verdict bands in generated tokens per second. A typical reader manages
# about 5 tokens/s (250 words a minute), so "Usable" starts a little
# below reading speed and "Fast" at about twice it.
FAST_TPS = 10.0
USABLE_TPS = 4.0

DEFAULT_EFFICIENCY = 0.8  # share of copy bandwidth reached; 0.66 to 0.98 measured
MIN_TESTABLE_TPS = 0.3  # below this even one probe takes too long
TG_NOISE_FLOOR = 0.04  # run-to-run spread seen on the test i3 was 2.6 to 6.8%
PP_NOISE_FLOOR = 0.06
MAX_NOISE = 0.10  # beyond this the PC is too busy to separate settings anyway
SCREEN_NOISE = 0.08  # single-repetition screening results this close to the leader are re-tested
MAX_FINALISTS = 6
DEFAULT_BUDGET_S = 600


def verdict(tps: float, fast: float = FAST_TPS, usable: float = USABLE_TPS) -> str:
    """Plain-language band for a generation speed: Fast, Usable or Slow."""
    if tps >= fast:
        return "Fast"
    if tps >= usable:
        return "Usable"
    return "Slow"


VERDICT_REASONS = {
    "Fast": "text appears faster than you can read it",
    "Usable": "text appears at about reading speed",
    "Slow": "text appears slower than you can read; a smaller model or quantization will feel better",
    "Won't fit": "the model does not fit in the memory free right now",
}


def predict_generation_tps(model: ModelInfo, bandwidth_gbs: float, efficiency: float = DEFAULT_EFFICIENCY,
                           depth: int = 0) -> float:
    """Predicted generation speed from memory bandwidth.

    Every generated token reads the active weights once (for an MoE model
    only the experts it uses), plus the whole KV cache filled so far
    (``depth`` tokens).
    """
    per_token = model.active_bytes_per_token or model.bytes_per_token or model.tensor_bytes
    if depth and model.kv_layout is not None:
        per_token += kv_cache_bytes(model.kv_layout, depth)
    if per_token <= 0 or bandwidth_gbs <= 0:
        return 0.0
    return efficiency * bandwidth_gbs * 1e9 / per_token


def thread_candidates(hw: HardwareInfo) -> list[int]:
    """Thread counts worth measuring on this CPU.

    Small CPUs (up to 8 threads) try every count from one below the
    performance-core count to every logical thread: on the test
    2-core/4-thread i3, 3 threads beat the default of 2 by 22%. Larger
    CPUs get a coarse spread from a quarter of the cores up, because
    bandwidth-bound generation often peaks well below the core count;
    :func:`refine_threads` fills in around the best one. Efficiency cores
    are left out; on multi-socket machines one socket's cores are tried.
    """
    perf = hw.performance_cores or hw.physical_cores or 1
    logical = hw.performance_threads if hw.efficiency_cores else hw.logical_cores
    logical = max(logical or perf, perf)
    if logical <= 8:
        low = max(2, perf - 1) if perf >= 2 else 1
        return list(range(min(low, logical), logical + 1))
    cands = {max(2, perf // 4), max(2, perf // 2), max(2, 3 * perf // 4), perf, logical}
    if hw.numa_nodes > 1:
        cands.add(max(2, perf // hw.numa_nodes))
    return sorted(c for c in cands if c <= logical)


def refine_threads(tested: list[int], best: int) -> list[int]:
    """Untested thread counts halfway between the best and its neighbours."""
    tested = sorted(set(tested))
    if best not in tested:
        return []
    i = tested.index(best)
    extra = []
    for j in (i - 1, i + 1):
        if 0 <= j < len(tested) and abs(tested[j] - best) > 1:
            mid = (tested[j] + best) // 2
            if mid not in tested:
                extra.append(mid)
    return extra


@dataclass(frozen=True)
class Candidate:
    """Settings that change between screened configurations."""

    threads: int
    flash_attn: bool | None = None
    repack: bool | None = None

    def describe(self) -> str:
        parts = [f"{self.threads} threads"]
        if self.flash_attn is not None:
            parts.append(f"flash attention {'on' if self.flash_attn else 'off'}")
        if self.repack is not None:
            parts.append(f"repacking {'on' if self.repack else 'off'}")
        return ", ".join(parts)


def _key(r: BenchRecord) -> Candidate:
    return Candidate(r.threads, r.flash_attn, r.repack)


@dataclass
class Speed:
    """Measured speed of one candidate."""

    tg: float = 0.0
    pp: float = 0.0
    tg_spread: float = 0.0
    pp_spread: float = 0.0


def collect(records: list[BenchRecord]) -> dict[Candidate, Speed]:
    """Group sweep records by candidate (generation and prompt tests)."""
    out: dict[Candidate, Speed] = {}
    for r in records:
        s = out.setdefault(_key(r), Speed())
        if r.kind == "tg":
            s.tg, s.tg_spread = r.median_ts, r.spread
        elif r.kind == "pp":
            s.pp, s.pp_spread = r.median_ts, r.spread
    return out


def pick(speeds: dict[Candidate, Speed], noise: float, metric: str = "tg") -> Candidate | None:
    """Fastest candidate, preferring less memory and fewer threads within the noise."""
    scored = {k: getattr(v, metric) for k, v in speeds.items() if getattr(v, metric) > 0}
    if not scored:
        return None
    top = max(scored.values())
    close = [k for k, v in scored.items() if v >= top * (1 - noise)]
    return min(close, key=lambda k: (k.repack is True, k.threads, -scored[k]))


@dataclass
class TuneReport:
    """Everything the tuner found, for display and storage."""

    status: str = "failed"  # tuned | prediction_only | wont_fit | unsupported | failed
    verdict: str = ""
    model: ModelInfo | None = None
    hw: HardwareInfo | None = None
    gate: GateDecision | None = None
    bandwidth_gbs: float = 0.0
    predicted_tps: float = 0.0
    efficiency: float = 0.0
    baseline: Candidate | None = None
    baseline_tps: float = 0.0
    baseline_pp: float = 0.0
    config: SearchConfig | None = None
    best_tps: float = 0.0
    best_pp: float = 0.0
    depth: int = 0
    depth_tps: float = 0.0
    predicted_depth_tps: float = 0.0
    drift: float = 0.0
    noise: float = TG_NOISE_FLOOR
    tests: int = 0
    elapsed: float = 0.0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def gain(self) -> float:
        """Generation speed-up over llama.cpp's defaults (0.1 = 10% faster)."""
        return self.best_tps / self.baseline_tps - 1 if self.baseline_tps else 0.0


class Tuner:
    """Measured CPU tuning for one model (see the module docstring)."""

    def __init__(
        self,
        model_path: str,
        *,
        ctx: int = 4096,
        budget_s: float = DEFAULT_BUDGET_S,
        caps: Capabilities | None = None,
        hw: HardwareInfo | None = None,
        say: Callable[[str], None] | None = None,
        sweep: Callable[..., SweepResult] | None = None,
        known_bandwidth: float = 0.0,
    ):
        self.model_path = model_path
        self.budget_s = budget_s
        self.say = say or (lambda msg: None)
        self.sweep = sweep or run_sweep
        self.known_bandwidth = known_bandwidth  # best earlier measurement on this PC
        self.caps = caps or detect_capabilities()
        self.hw = hw or detect_hardware(self.caps)
        self.model = inspect_model(model_path)
        if self.model.training_context:
            ctx = min(ctx, self.model.training_context)
        self.ctx = ctx
        self.report = TuneReport(model=self.model, hw=self.hw)
        self._start = 0.0

    # ── helpers ────────────────────────────────────────────────────────

    def _remaining(self) -> float:
        return self.budget_s - (time.monotonic() - self._start)

    def _run(self, base: SearchConfig, lists: dict[str, list] | None = None, **kw) -> SweepResult:
        kw.setdefault("registry", self.caps.bench)
        kw.setdefault("timeout", max(60.0, self._remaining() + 30))
        result = self.sweep(self.model_path, base, lists or {}, **kw)
        self.report.tests += len(result.records)
        if result.error:
            self.report.warnings.append(_explain_error(result, self.caps.build.build_number))
        return result

    @staticmethod
    def _sizes(tg: float) -> tuple[int, int]:
        """Generation tokens for screening and for confirming at a speed."""
        screen = int(min(64, max(8, round(tg * 3))))
        confirm = int(min(128, max(16, round(tg * 6))))
        return screen, confirm

    # ── stages ─────────────────────────────────────────────────────────

    def plan(self) -> GateDecision:
        """Fit the model into free memory (no benchmarks)."""
        mem = memory_status()
        gate = plan_memory(
            self.model, memory_budget(mem.available, mem.total), isa_tier=self.hw.isa_tier or "avx2",
            ctx=self.ctx, min_ctx=min(2048, self.ctx), ubatch=512,
            tensor_sizes=tensor_sizes_of(self.model_path),
            compute_model=compute_model_for(self.model_path, self.model),  # llama-bench: all outputs
        )
        self.report.gate = gate
        return gate

    def preflight(self) -> None:
        """Warn about conditions that make measurements unreliable."""
        power = power_status()
        if power.on_battery:
            self.report.warnings.append("The PC runs on battery: plug it in for full and stable speed.")
        if power.battery_saver:
            self.report.warnings.append("Battery saver is on: speeds will be lower than normal.")
        load = cpu_load(1.0)
        if load > 25:
            self.report.warnings.append(
                f"Other programs are using {load:.0f}% of the CPU: close them for more reliable results.")

    def run(self) -> TuneReport:
        """Run all stages within the time budget and return the report."""
        self._start = time.monotonic()
        rep = self.report
        try:
            self._run_stages(rep)
        finally:
            rep.elapsed = time.monotonic() - self._start
        return rep

    def _duel(self, base: SearchConfig, best: SearchConfig, n_gen: int,
              load_s: float) -> tuple[float, float, float] | None:
        """Alternate defaults and winner, one short process each.

        Returns ``(median default speed, gain, noise)``, or None when there
        is no time. The gain is the median of the paired ratios; the noise is
        twice the standard error of that median, from the median absolute
        deviation (one disturbed run does not decide), between 4% and 10%.
        Two more pairs are added when the gain is positive but unclear.
        """
        def one(cfg: SearchConfig) -> float:
            res = collect(self._run(cfg, n_gen=n_gen, repetitions=1).records)
            return next((v.tg for v in res.values() if v.tg), 0.0)

        tg_guess = max(self.report.baseline_tps, 0.1)
        pair_s = 2 * (load_s + 2 * n_gen / tg_guess + 0.5)
        if 3 * pair_s > 0.3 * self._remaining():
            return None
        self.say("Comparing the best settings with llama.cpp's defaults, taking turns...")
        d_runs: list[float] = []
        ratios: list[float] = []

        def judge() -> tuple[float, float]:
            mid = median(ratios)
            mad = median(abs(r - mid) for r in ratios)
            se = 1.25 * 1.4826 * mad / math.sqrt(len(ratios))
            return mid - 1, min(MAX_NOISE, max(TG_NOISE_FLOOR, 2 * se / mid))

        for n_pairs in (3, 2):
            for _ in range(n_pairs):
                d, w = one(base), one(best)
                if d:
                    d_runs.append(d)
                if d and w:
                    ratios.append(w / d)
            if not ratios:
                return None
            gain, noise = judge()
            if not 0 < gain <= noise or 2 * pair_s > 0.3 * self._remaining():
                break
        return median(d_runs), gain, noise

    @staticmethod
    def _choose_flash_attn(rep: TuneReport, config: SearchConfig, by_fa: dict[bool, float],
                           pool: dict[Candidate, Speed], screen: dict[Candidate, Speed], noise: float) -> None:
        """Set flash attention from its speed in an empty and in a long chat.

        Each setting scores the geometric mean of both speeds; a chat spends
        its time in between. The empty-chat speeds come from one source (the
        confirmation, else the screen) so both were measured alike. The
        setting switches only when its score wins beyond the noise.
        """
        threads = config.threads or rep.baseline.threads
        empty: dict[bool, float] = {}
        for src in (pool, screen):
            found = {k.flash_attn: s.tg for k, s in src.items()
                     if k.threads == threads and k.flash_attn in (False, True) and s.tg}
            if len(found) == 2:
                empty = found
                break
        score = {fa: math.sqrt(by_fa[fa] * empty[fa]) if empty else by_fa[fa] for fa in (False, True)}
        winner = max(score, key=score.get)
        loser = not winner
        if config.flash_attn == winner or score[winner] <= score[loser] * (1 + noise):
            return
        config.flash_attn = winner
        note = f"Flash attention {'on' if winner else 'off'}: {by_fa[winner] / by_fa[loser] - 1:.0%} faster after " \
               f"{rep.depth:,} tokens of chat"
        if empty:
            note += f", {empty[winner] / empty[loser] - 1:+.0%} in an empty one"
            # The reported speed is for the settings that are saved (Granite MoE: 47.8 -> about 38).
            rep.best_tps *= empty[winner] / empty[loser]
            rep.verdict = verdict(rep.best_tps)
        rep.notes = ["Apart from flash attention, llama.cpp's default settings are the fastest on this PC."
                     if n.startswith("llama.cpp's default settings are already") else n for n in rep.notes]
        rep.notes.append(note + ".")

    def _run_stages(self, rep: TuneReport) -> None:
        if architecture_supported(self.caps, self.model.architecture) is False:
            rep.status = "unsupported"
            rep.error = update_llama_advice(self.model.architecture, self.caps.build.build_number)
            return
        gate = rep.gate or self.plan()
        if not gate.fits:
            rep.status, rep.verdict = "wont_fit", "Won't fit"
            return
        self.preflight()

        self.say("Measuring memory speed...")
        perf = self.hw.performance_cores or self.hw.physical_cores or 1
        # A busy PC can only make the copy test slower: the best result ever
        # measured here is the real bandwidth (12.8 GB/s measured under 81%
        # CPU load gave an impossible 121% efficiency; the PC does 18.0).
        rep.bandwidth_gbs = max(measure_bandwidth(threads=min(perf, 16)), self.known_bandwidth)
        rep.predicted_tps = predict_generation_tps(self.model, rep.bandwidth_gbs)
        if rep.predicted_tps and rep.predicted_tps < MIN_TESTABLE_TPS:
            rep.status = "prediction_only"
            rep.verdict = verdict(rep.predicted_tps)
            rep.notes.append(f"Predicted {rep.predicted_tps:.2f} tokens/s: too slow to test within the time limit.")
            return

        # Every run at above-normal priority, as launch uses: the same speed
        # the user gets, and far steadier while other programs run (E3).
        prio = ABOVE_NORMAL if self.caps.bench is None or self.caps.bench.has("--prio") else None
        base = gate.apply(SearchConfig(priority=prio))
        base.ctx_size = None
        pred = rep.predicted_tps or 4.0
        _, n_confirm = self._sizes(pred)
        n_prompt = 64 if pred < USABLE_TPS else 128

        # 1. Baseline with llama.cpp's defaults: calibration and the bar to beat.
        self.say("Measuring llama.cpp's default settings...")
        first = self._run(base, n_prompt=n_prompt, n_gen=n_confirm, repetitions=2)
        speeds = collect(first.records)
        if not speeds or not any(s.tg for s in speeds.values()):
            if first.error == "unsupported_arch":
                rep.status = "unsupported"
                rep.error = update_llama_advice(first.detail, self.caps.build.build_number)
                rep.warnings.clear()
            else:
                rep.error = first.error or "failed"
            return
        rep.baseline, b = next((k, v) for k, v in speeds.items() if v.tg)
        rep.baseline_tps, rep.baseline_pp = b.tg, b.pp
        rep.noise = min(MAX_NOISE, max(TG_NOISE_FLOOR, 2 * b.tg_spread))
        if rep.predicted_tps:
            rep.efficiency = b.tg / (rep.predicted_tps / DEFAULT_EFFICIENCY)
        n_screen, _ = self._sizes(b.tg)  # confirm keeps the baseline's size, so the two compare
        pp = b.pp or b.tg * 5
        load_s = max(1.0, first.elapsed - 3 * (n_prompt / pp + n_confirm / b.tg))

        # 2. Screen: every candidate once, in one process.
        lists: dict[str, list] = {"threads": thread_candidates(self.hw)}
        if gate.cache_type == "f16":  # a quantized KV cache requires flash attention on
            lists["flash_attn"] = [False, True]
        if gate.repack and self.caps.bench is not None and self.caps.bench.has("--repack") \
                and repack_copy_bytes(self.model, self.hw.isa_tier or "avx2", tensor_sizes_of(self.model_path)):
            lists["repack"] = [False, True]

        def estimate(ls: dict[str, list], n_gen: int, n_pp: int, reps: int) -> float:
            n = math.prod(len(v) for v in ls.values())
            loads = len(ls.get("repack", [None]))
            return n * ((reps + 1) * n_pp / pp + reps * n_gen / b.tg + 0.5) + loads * load_s

        share = 0.4 * self._remaining()
        while estimate(lists, n_screen, n_prompt, 1) > share:
            if n_prompt > 32 or n_screen > 8:
                n_prompt, n_screen = max(32, n_prompt // 2), max(8, n_screen // 2)
            elif "repack" in lists:
                del lists["repack"]
            elif len(lists["threads"]) > 2:
                t = lists["threads"]
                far = max(t, key=lambda x: abs(x - rep.baseline.threads))
                lists["threads"] = [x for x in t if x != far]
            else:
                break
        n_cand = math.prod(len(v) for v in lists.values())
        self.say(f"Testing {n_cand} combinations of settings "
                 f"(about {_minutes(estimate(lists, n_screen, n_prompt, 1))})...")
        screen = collect(self._run(base, lists, n_prompt=n_prompt, n_gen=n_screen, repetitions=1).records)
        if not screen:
            rep.status, rep.verdict = "tuned", verdict(rep.baseline_tps)
            rep.config, rep.best_tps, rep.best_pp = base, rep.baseline_tps, rep.baseline_pp
            rep.notes.append("The settings test failed, so llama.cpp's defaults are kept.")
            return

        # 3. Confirm the best third and anything within screening noise of the
        #    leader, plus untested thread counts next to it. Candidates that
        #    share a repacking value run in one process (one model load).
        ranked = sorted((k for k in screen if screen[k].tg), key=lambda k: -screen[k].tg)
        leader = ranked[0]
        cut = max(2, math.ceil(len(ranked) / 3))
        finalists = [k for i, k in enumerate(ranked)
                     if i < cut or screen[k].tg >= screen[leader].tg * (1 - SCREEN_NOISE)][:MAX_FINALISTS]
        extra = refine_threads(lists["threads"], leader.threads)
        by_repack: dict[bool | None, list[Candidate]] = {}
        for k in finalists:
            by_repack.setdefault(k.repack, []).append(k)
        confirmed: dict[Candidate, Speed] = {}
        for rp, ks in sorted(by_repack.items(), key=lambda kv: kv[0] != leader.repack):
            if self._remaining() < 0.3 * self.budget_s and confirmed:
                break
            threads = sorted({k.threads for k in ks} | (set(extra) if rp == leader.repack else set()))
            fas = sorted({k.flash_attn for k in ks}, key=str)
            cfg = base.model_copy(update={"repack": rp if rp is not None else base.repack})
            self.say(f"Confirming {len(threads) * len(fas)} finalist(s) with 3 repetitions...")
            confirm_lists: dict[str, list] = {"threads": threads}
            if "flash_attn" in lists:
                confirm_lists["flash_attn"] = fas
            res = self._run(cfg, confirm_lists, n_gen=n_confirm, repetitions=3)
            confirmed.update(collect(res.records))
        pool = confirmed or screen

        spreads = [s.tg_spread for s in pool.values() if s.tg_spread] + [b.tg_spread]
        tg_noise = min(MAX_NOISE, max(TG_NOISE_FLOOR, 2 * median(spreads)))
        best = pick(pool, tg_noise) or leader
        best_speed = pool[best]
        best_cfg = base.model_copy(update={"threads": best.threads, "flash_attn": best.flash_attn})
        if best.repack is not None:
            best_cfg.repack = best.repack

        # Final check against the defaults. Alternating short runs (defaults,
        # winner, defaults, winner...) expose both to the same disturbances;
        # on the busy test PC sequential runs judged the same +8-21% gain
        # anywhere from -3% to +22%, alternating ones +8% to +21% (E2).
        duel = self._duel(base, best_cfg, n_confirm, load_s)
        if duel is not None:
            d_med, gain, tg_noise = duel
            rep.drift = d_med / rep.baseline_tps - 1
            rep.baseline_tps = d_med
            best_speed = Speed(tg=d_med * (1 + gain), pp=best_speed.pp)
            beats_defaults = gain > tg_noise
        else:  # no time for a duel: the defaults once more, averaged with the first run
            self.say("Re-checking llama.cpp's defaults...")
            last = collect(self._run(base, n_gen=n_confirm, repetitions=2).records)
            again = next((v for v in last.values() if v.tg), None)
            if again is not None:
                spreads.append(again.tg_spread)
                rep.drift = again.tg / rep.baseline_tps - 1
                rep.baseline_tps = (rep.baseline_tps + again.tg) / 2
            tg_noise = min(MAX_NOISE, max(TG_NOISE_FLOOR, 2 * median(spreads)))
            beats_defaults = best_speed.tg > rep.baseline_tps * (1 + tg_noise)
        rep.noise = tg_noise
        if abs(rep.drift) > max(2 * tg_noise, 0.08):
            rep.warnings.append(
                f"Speed changed by {rep.drift:+.0%} during the test (another program, or heat): "
                "run tune again when the PC is idle for more reliable results.")

        # Prompt threads: kept only when they read prompts faster beyond the noise.
        pp_pool = {k: v for k, v in screen.items() if (k.flash_attn, k.repack) == (best.flash_attn, best.repack)}
        tb = pick(pp_pool, PP_NOISE_FLOOR, "pp")
        best_pp = pp_pool.get(best, Speed()).pp
        threads_batch, tb_pp = None, 0.0
        if tb and tb.threads != best.threads and pp_pool[tb].pp > best_pp * (1 + PP_NOISE_FLOOR):
            threads_batch, tb_pp = tb.threads, pp_pool[tb].pp

        config = base.model_copy(update={"threads": best.threads, "threads_batch": threads_batch,
                                         "flash_attn": best.flash_attn})
        if best.repack is not None:
            config.repack = best.repack

        # 4. Multi-socket machines: spread memory over the sockets?
        if self.hw.numa_nodes > 1 and self._remaining() > 0.2 * self.budget_s:
            self.say("Testing NUMA memory placement...")
            numa = collect(self._run(config, n_gen=n_confirm, repetitions=3, numa="distribute").records)
            n_speed = next((v.tg for v in numa.values() if v.tg), 0.0)
            if n_speed > best_speed.tg * (1 + tg_noise):
                config.numa = "distribute"
                rep.notes.append(f"--numa distribute: {n_speed / best_speed.tg - 1:+.0%} on this multi-socket PC.")
                best_speed = Speed(tg=n_speed, pp=best_speed.pp)

        # Time for step 6 is set aside first: speed in a long chat matters more
        # than prompt batching, and it decides flash attention when screened.
        depth = min(gate.ctx // 2, 2048)
        one_depth_s = depth / (tb_pp or best_pp or pp) * 2 if depth >= 512 else 0.0
        compare_fa = "flash_attn" in lists
        depth_reserve = (2 if compare_fa else 1) * one_depth_s

        # 5. ubatch for prompt processing, if time allows.
        ubatches = [u for u in (128, 256, 512) if u <= gate.ubatch]
        prompt_speed = tb_pp or best_pp or pp
        n_pp = 512  # ubatch only matters for prompts longer than the smallest ubatch tested
        if len(ubatches) > 1 and                 3 * len(ubatches) * n_pp / prompt_speed < 0.25 * (self._remaining() - 2 * depth_reserve):
            self.say("Testing prompt batch sizes...")
            cfg = config.model_copy(update={"threads": threads_batch or best.threads})
            res = self._run(cfg, {"ubatch_size": ubatches}, n_prompt=n_pp, repetitions=2)
            by_ub = {r.ubatch: r.median_ts for r in res.records if r.kind == "pp"}
            if by_ub:
                top = max(by_ub.values())
                default = by_ub.get(gate.ubatch, 0.0)
                chosen = gate.ubatch if default >= top * (1 - PP_NOISE_FLOOR) else max(by_ub, key=by_ub.get)
                if chosen != gate.ubatch:
                    config.ubatch_size = chosen
                    rep.notes.append(f"ubatch {chosen}: prompts {top / default - 1:+.0%} faster."
                                     if default else f"ubatch {chosen}.")

        # Never slower than the defaults: a win must clear the noise.
        if not beats_defaults:
            rep.notes.append("llama.cpp's default settings are already the fastest on this PC "
                             "(differences were within measurement noise).")
            # ubatch was measured on its own (prompts only), so its win stays.
            config = base.model_copy(update={"ubatch_size": config.ubatch_size})
            best_speed = Speed(tg=rep.baseline_tps, pp=rep.baseline_pp)
            if threads_batch and threads_batch != rep.baseline.threads \
                    and tb_pp > rep.baseline_pp * (1 + PP_NOISE_FLOOR):
                config.threads_batch = threads_batch
            else:
                tb_pp = 0.0
        rep.config = config
        rep.best_tps = best_speed.tg
        rep.best_pp = tb_pp or best_speed.pp or best_pp
        rep.verdict = verdict(rep.best_tps)
        rep.status = "tuned"

        # 6. Speed with a filled context. When flash attention was screened it
        #    is compared here too: on a CPU it can tie in an empty context and
        #    still cost 30-40% once the chat grows (llama.cpp b12228, Llama 3.2
        #    1B: 9.3 vs 5.8 tokens/s after 2,048 tokens).
        if compare_fa and 2 * one_depth_s >= 0.5 * self._remaining():
            compare_fa = False  # too little time left: measure the chosen setting only
        if depth >= 512 and (2 if compare_fa else 1) * one_depth_s < 0.5 * self._remaining():
            self.say(f"Checking speed after {depth:,} tokens of conversation...")
            d_lists = {"flash_attn": [False, True]} if compare_fa else {}
            records = self._run(config, d_lists, n_gen=max(8, n_confirm // 2), repetitions=1, n_depth=depth).records
            by_fa = {r.flash_attn: r.median_ts for r in records if r.kind == "tg" and r.median_ts}
            rep.depth = depth
            if compare_fa and len(by_fa) == 2:
                self._choose_flash_attn(rep, config, by_fa, pool, screen, tg_noise)
            rep.depth_tps = by_fa.get(config.flash_attn) or max(by_fa.values(), default=0.0)
            if rep.efficiency:
                rep.predicted_depth_tps = predict_generation_tps(self.model, rep.bandwidth_gbs,
                                                                 rep.efficiency, depth)


def _minutes(seconds: float) -> str:
    if seconds < 90:
        return f"{max(10, round(seconds / 10) * 10):.0f} s"
    return f"{seconds / 60:.0f} min"


def _explain_error(result: SweepResult, build: int = 0) -> str:
    if result.error == "unsupported_arch":
        return update_llama_advice(result.detail, build)
    return {
        "memory_guard": "A test was stopped because free memory ran too low.",
        "memory_guard_external": "A test was stopped because another program used up free memory.",
        "timeout": "A test was stopped at the time limit; the results so far are used.",
        "not_found": NOT_FOUND_ADVICE,
    }.get(result.error, f"A llama-bench run failed ({result.error}).")
