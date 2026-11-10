"""Tests for the measured CPU tuner, on simulated machines."""

import itertools
import json
import struct

import pytest

import llama_autotune.tuner as tuner_mod
from llama_autotune.benchmark import BenchRecord, SweepResult, parse_records, render_lists
from llama_autotune.compat import Capabilities, parse_help
from llama_autotune.memory import ComputeModel
from llama_autotune.database import TunedSettings, connect, get_tuned, save_tuned
from llama_autotune.models import Backend, HardwareInfo, KVLayout, ModelInfo, SearchConfig
from llama_autotune.resources import MemoryStatus, PowerStatus, parse_core_records, topology_from_cores
from llama_autotune.tuner import (
    Candidate,
    Speed,
    Tuner,
    pick,
    predict_generation_tps,
    refine_threads,
    thread_candidates,
    verdict,
)

MIB = 1024 * 1024
FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def _hw(**kw) -> HardwareInfo:
    base = dict(cpu_name="Sim", physical_cores=2, logical_cores=4, performance_cores=2, ram_gb=8.0,
                backend=Backend.CPU, isa_tier="avx2")
    base.update(kw)
    return HardwareInfo(**base)


# ── thread candidates ────────────────────────────────────────────────


@pytest.mark.parametrize("hw, expected", [
    (_hw(), [2, 3, 4]),                                                   # i3-6100: 2C/4T
    (_hw(physical_cores=4, logical_cores=8, performance_cores=4), [3, 4, 5, 6, 7, 8]),
    (_hw(physical_cores=1, logical_cores=2, performance_cores=1), [1, 2]),
    (_hw(physical_cores=2, logical_cores=2, performance_cores=2), [2]),
    (_hw(physical_cores=12, logical_cores=20, performance_cores=8, efficiency_cores=4,
         performance_threads=16), [2, 4, 6, 8, 16]),                      # hybrid: E-cores left out
    (_hw(physical_cores=64, logical_cores=128, performance_cores=64), [16, 32, 48, 64, 128]),
    (_hw(physical_cores=24, logical_cores=48, performance_cores=24, numa_nodes=2), [6, 12, 18, 24, 48]),
])
def test_thread_candidates(hw, expected):
    assert thread_candidates(hw) == expected


def test_many_core_candidates_reach_well_below_the_core_count():
    assert min(thread_candidates(_hw(physical_cores=32, logical_cores=64, performance_cores=32))) == 8


def test_refine_threads_fills_gaps_next_to_the_best():
    assert refine_threads([16, 32, 48, 64, 128], 32) == [24, 40]
    assert refine_threads([2, 3, 4], 3) == []
    assert refine_threads([8, 16], 16) == [12]


# ── prediction and verdicts ──────────────────────────────────────────


def test_prediction_is_bandwidth_over_bytes_per_token():
    m = ModelInfo(bytes_per_token=1_000_000_000)
    assert predict_generation_tps(m, 20.0) == pytest.approx(16.0)
    assert predict_generation_tps(m, 20.0, efficiency=1.0) == pytest.approx(20.0)
    assert predict_generation_tps(ModelInfo(), 20.0) == 0.0


def test_prediction_slows_down_with_a_filled_context():
    m = ModelInfo(bytes_per_token=1_000_000_000, kv_layout=KVLayout(k_elems_full=65536, v_elems_full=65536))
    assert predict_generation_tps(m, 20.0, depth=4096) < predict_generation_tps(m, 20.0)


@pytest.mark.parametrize("tps, band", [(25, "Fast"), (10, "Fast"), (9.9, "Usable"), (4, "Usable"), (3.9, "Slow")])
def test_verdict_bands(tps, band):
    assert verdict(tps) == band


def test_pick_prefers_less_memory_and_fewer_threads_within_noise():
    speeds = {
        Candidate(4, False, True): Speed(tg=14.0),
        Candidate(3, False, True): Speed(tg=13.8),
        Candidate(3, False, False): Speed(tg=13.7),
        Candidate(2, False, False): Speed(tg=11.0),
    }
    assert pick(speeds, 0.04) == Candidate(3, False, False)
    assert pick(speeds, 0.0) == Candidate(4, False, True)
    assert pick({}, 0.04) is None


# ── llama-bench output and arguments ─────────────────────────────────

REAL_LINE = json.dumps({
    "build_number": 11347, "n_ubatch": 512, "n_threads": 3, "flash_attn": 0, "repack": True,
    "n_prompt": 0, "n_gen": 8, "n_depth": 0, "avg_ts": 12.906983, "stddev_ts": 0.5,
    "samples_ts": [12.4, 12.9, 13.4],
})


def test_parse_records_reads_jsonl_and_skips_noise():
    out = "load_backend: loaded CPU\n" + REAL_LINE + "\n{not json\n" + '{"other": 1}\n'
    (r,) = parse_records(out)
    assert (r.threads, r.flash_attn, r.repack, r.kind) == (3, False, True, "tg")
    assert r.median_ts == 12.9 and r.spread == pytest.approx(0.5 / 12.906983)


def test_parse_records_keeps_partial_output():
    partial = REAL_LINE + "\n" + REAL_LINE[:40]
    assert len(parse_records(partial)) == 1


def test_render_lists_joins_values_with_commas():
    args, skipped = render_lists({"threads": [2, 3, 4], "flash_attn": [False, True]}, None)
    assert args == ["-t", "2,3,4", "-fa", "off,on"] and skipped == []


def test_render_lists_drops_settings_the_build_lacks():
    reg = parse_help("  -t, --threads <n>   (default: 2)\n")
    args, skipped = render_lists({"threads": [2, 3], "repack": [False, True]}, reg)
    assert args == ["-t", "2,3"] and skipped == ["repack"]


# ── topology: processor groups ───────────────────────────────────────


def _core_record(eff_class: int, mask: int, group: int) -> bytes:
    body = struct.pack("<BB20sH", 0, eff_class, b"\0" * 20, 1) + struct.pack("<QH6s", mask, group, b"\0" * 6)
    return struct.pack("<II", 0, 8 + len(body)) + body


def test_threads_in_every_processor_group_are_counted():
    # 72 logical processors: Windows puts 32 cores in group 0 and 4 in group 1.
    buf = b"".join(_core_record(0, 0b11 << (2 * (i % 32)), i // 32) for i in range(36))
    topo = topology_from_cores(parse_core_records(buf))
    assert (topo.physical_cores, topo.logical_cores) == (36, 72)


def test_hybrid_topology_from_efficiency_classes():
    buf = b"".join(_core_record(1, 0b11, 0) for _ in range(6)) + b"".join(_core_record(0, 0b1, 0) for _ in range(8))
    topo = topology_from_cores(parse_core_records(buf))
    assert (topo.performance_cores, topo.efficiency_cores, topo.performance_logical) == (6, 8, 12)


# ── tuned settings storage ───────────────────────────────────────────


def test_newest_tuned_settings_win(tmp_path):
    conn = connect(str(tmp_path / "db.sqlite"))
    save_tuned(conn, "hw", "m", TunedSettings(config=SearchConfig(threads=2), generation_tps=10))
    save_tuned(conn, "hw", "m", TunedSettings(config=SearchConfig(threads=3, flash_attn=False), generation_tps=12))
    got = get_tuned(conn, "hw", "m")
    assert got.config.threads == 3 and got.config.flash_attn is False and got.generation_tps == 12
    assert get_tuned(conn, "other", "m") is None


def test_launch_uses_tuned_defaults_over_heuristic_guesses(monkeypatch):
    """A setting tune kept at llama.cpp's default must not fall back to the heuristic."""
    from llama_autotune import cli
    from llama_autotune.models import Backend, HardwareInfo, ModelInfo

    tuned = TunedSettings(config=SearchConfig(threads=4, flash_attn=None, repack=None), generation_tps=13)
    monkeypatch.setattr(cli, "get_tuned", lambda conn, hw, m: tuned)
    monkeypatch.setattr(cli, "hardware_fingerprint", lambda hw: "hw")
    heuristic = SearchConfig(threads=2, flash_attn=False, batch_size=512)
    out = cli._apply_tuned(heuristic, HardwareInfo(backend=Backend.CPU), ModelInfo(), 12228)
    assert out.threads == 4
    assert out.flash_attn is None          # measured at "auto" and won: not the guessed "off"
    assert out.batch_size == 512           # settings tune does not measure stay as they were


# ── the tuner end to end on a simulated machine ──────────────────────


class SimMachine:
    """llama-bench stand-in. ``tg(threads, fa, repack)`` sets the speed."""

    def __init__(self, tg, pp=lambda t, fa, rp, ub: 20.0 * t, default_threads=2, drift=1.0, numa_gain=1.0,
                 depth_factor=lambda fa: 0.75):
        self.tg, self.pp, self.default_threads = tg, pp, default_threads
        self.drift, self.numa_gain, self.depth_factor = drift, numa_gain, depth_factor
        self.calls = []

    def __call__(self, model_path, base, lists, **kw):
        self.calls.append((base.model_copy(), dict(lists), kw))
        base_runs = [c for c in self.calls if not c[1] and not kw.get("n_depth") and not kw.get("numa")]
        factor = self.drift if len(base_runs) > 1 and not lists and not kw.get("n_depth") else 1.0
        axes = {
            "threads": lists.get("threads", [base.threads or self.default_threads]),
            "flash_attn": lists.get("flash_attn", [base.flash_attn]),
            "repack": lists.get("repack", [True if base.repack is None else base.repack]),
            "ubatch_size": lists.get("ubatch_size", [base.ubatch_size or 512]),
        }
        records = []
        for t, fa, rp, ub in itertools.product(*axes.values()):
            if kw.get("n_prompt"):
                v = self.pp(t, fa, rp, ub)
                records.append(BenchRecord(t, fa, rp, ub, kw["n_prompt"], 0, 0, v, v * 0.01, [v]))
            if kw.get("n_gen"):
                v = self.tg(t, fa, rp) * factor * (self.numa_gain if kw.get("numa") else 1.0)
                if kw.get("n_depth"):
                    v *= self.depth_factor(fa)
                records.append(BenchRecord(t, fa, rp, ub, 0, kw["n_gen"], kw.get("n_depth", 0), v, v * 0.01, [v]))
        return SweepResult(records=records, elapsed=5.0)


I3_TG = {2: 11.5, 3: 13.9, 4: 13.7}


def _i3_speed(t, fa, rp):
    return I3_TG.get(t, 8.0) * (0.88 if fa in (True, None) else 1.0) * (1.0 if rp in (True, None) else 0.9)


@pytest.fixture
def sim(monkeypatch):
    model = ModelInfo(path="m.gguf", architecture="llama", n_layers=16, training_context=131072,
                      bytes_per_token=1300 * MIB, tensor_bytes=1300 * MIB, vocab_size=128256, n_embd=2048,
                      tensor_types={"Q4_K": 800 * MIB}, kv_layout=KVLayout(k_elems_full=8192, v_elems_full=8192),
                      fingerprint="fp")
    bench_help = (FIXTURES / "llama-bench-b11347-help.txt").read_text()
    caps = Capabilities(bench=parse_help(bench_help))
    monkeypatch.setattr(tuner_mod, "inspect_model", lambda p: model)
    monkeypatch.setattr(tuner_mod, "tensor_sizes_of", lambda p: None)
    monkeypatch.setattr(tuner_mod, "compute_model_for",
                        lambda p, m, server_build=0: ComputeModel(258 * MIB, 128256, 2048, 32))
    monkeypatch.setattr(tuner_mod, "memory_status", lambda: MemoryStatus(total=8 * 1024**3, available=6000 * MIB))
    monkeypatch.setattr(tuner_mod, "power_status", lambda: PowerStatus())
    monkeypatch.setattr(tuner_mod, "cpu_load", lambda s=1.0: 3.0)
    monkeypatch.setattr(tuner_mod, "measure_bandwidth", lambda **kw: 17.3)

    def make(machine, hw=None, budget=600):
        return Tuner("m.gguf", caps=caps, hw=hw or _hw(), sweep=machine, budget_s=budget)
    return make


def test_tuner_finds_the_faster_settings(sim):
    machine = SimMachine(_i3_speed)
    rep = sim(machine).run()
    assert rep.status == "tuned"
    assert rep.config.threads == 3 and rep.config.flash_attn is False and rep.config.repack is not False
    assert rep.best_tps == pytest.approx(13.9)
    assert rep.baseline.threads == 2 and rep.gain > 0.3
    assert rep.verdict == "Fast"
    assert rep.efficiency > 0 and rep.depth_tps == pytest.approx(13.9 * 0.75)
    screen = next(lists for _, lists, _ in machine.calls if "flash_attn" in lists)
    assert screen["threads"] == [2, 3, 4] and screen["repack"] == [False, True]


def test_defaults_are_kept_when_nothing_beats_them_beyond_noise(sim):
    rep = sim(SimMachine(lambda t, fa, rp: 10.0 + 0.01 * t)).run()
    assert rep.status == "tuned"
    assert rep.config.threads is None and rep.config.flash_attn is None
    assert any("already the fastest" in n for n in rep.notes)


def test_ubatch_win_survives_when_defaults_are_kept(sim):
    """Llama 1B on b12228: defaults won for writing, ubatch 256 read prompts 29% faster."""
    pp = lambda t, fa, rp, ub: 100.0 if ub == 256 else 78.0  # noqa: E731
    rep = sim(SimMachine(lambda t, fa, rp: 10.0 + 0.01 * t, pp=pp)).run()
    assert any("already the fastest" in n for n in rep.notes)
    assert rep.config.threads is None and rep.config.ubatch_size == 256
    assert any(n.startswith("ubatch 256") for n in rep.notes)


def test_flash_attention_is_decided_by_a_long_chat(sim):
    """b12228 on the i3: FA ties in an empty context but is 37% slower after 2,048 tokens."""
    machine = SimMachine(lambda t, fa, rp: 11.4 + 0.01 * t,
                         depth_factor=lambda fa: 0.51 if fa in (True, None) else 0.81)
    rep = sim(machine).run()
    assert rep.config.flash_attn is False and rep.config.threads is None   # defaults kept, FA off
    assert rep.depth_tps == pytest.approx(11.42 * 0.81, rel=0.01)
    assert any(n.startswith("Flash attention off") for n in rep.notes)
    depth_call = next(lists for _, lists, kw in machine.calls if kw.get("n_depth"))
    assert depth_call == {"flash_attn": [False, True]}


def test_flash_attention_stays_when_the_long_chat_agrees(sim):
    machine = SimMachine(_i3_speed, depth_factor=lambda fa: 0.75)
    rep = sim(machine).run()
    assert rep.config.flash_attn is False
    assert not any(n.startswith("Flash attention") for n in rep.notes)


def test_prompt_threads_are_chosen_separately(sim):
    rep = sim(SimMachine(_i3_speed, pp=lambda t, fa, rp, ub: 30.0 * t)).run()
    assert rep.config.threads == 3 and rep.config.threads_batch == 4


def test_repacking_is_not_tested_when_the_copy_does_not_fit(sim, monkeypatch):
    monkeypatch.setattr(tuner_mod, "memory_status", lambda: MemoryStatus(total=8 * 1024**3, available=2600 * MIB))
    machine = SimMachine(_i3_speed)
    tuner = sim(machine)
    assert tuner.plan().repack is False
    tuner.run()
    assert all(base.repack is False for base, _, _ in machine.calls)
    assert all("repack" not in lists for _, lists, _ in machine.calls)


def test_drift_between_baseline_runs_is_reported(sim):
    rep = sim(SimMachine(_i3_speed, drift=0.7)).run()
    assert rep.drift == pytest.approx(-0.3)
    assert any("changed by" in w for w in rep.warnings)


def test_model_that_does_not_fit_is_never_run(sim, monkeypatch):
    monkeypatch.setattr(tuner_mod, "memory_status", lambda: MemoryStatus(total=8 * 1024**3, available=900 * MIB))
    machine = SimMachine(_i3_speed)
    rep = sim(machine).run()
    assert rep.status == "wont_fit" and rep.verdict == "Won't fit" and machine.calls == []


def test_hopelessly_slow_model_gets_a_prediction_only(sim, monkeypatch):
    monkeypatch.setattr(tuner_mod, "measure_bandwidth", lambda **kw: 0.2)
    machine = SimMachine(_i3_speed)
    rep = sim(machine).run()
    assert rep.status == "prediction_only" and rep.verdict == "Slow" and machine.calls == []


def test_multi_socket_machine_tries_numa_distribute(sim):
    hw = _hw(physical_cores=24, logical_cores=48, performance_cores=24, numa_nodes=2)
    machine = SimMachine(lambda t, fa, rp: 30.0 - abs(t - 12), numa_gain=1.3)
    rep = sim(machine, hw=hw).run()
    assert any(kw.get("numa") == "distribute" for _, _, kw in machine.calls)
    assert rep.config.numa == "distribute"
    assert rep.config.threads == 12


def test_short_time_budget_trims_the_screen(sim):
    machine = SimMachine(_i3_speed)
    sim(machine, budget=20).run()
    screen = next(lists for _, lists, _ in machine.calls if "flash_attn" in lists)
    assert "repack" not in screen or len(screen["threads"]) < 3


def test_settings_close_to_the_leader_are_all_confirmed(sim):
    # Real i3 screen: 4 threads led, 3 threads within 3%; both must be re-tested.
    machine = SimMachine(lambda t, fa, rp: {2: 10.3, 3: 13.46, 4: 13.89}[t] * (1.0 if fa else 0.98))
    sim(machine).run()
    confirm = [lists for _, lists, kw in machine.calls if kw.get("repetitions") == 3 and "threads" in lists]
    assert any(3 in lists["threads"] and 4 in lists["threads"] for lists in confirm)


def test_winner_and_defaults_take_turns_in_the_final_check(sim):
    # The first default run was slowed by another program (-25%). The duel runs both under the
    # same conditions afterwards, so the real win (+37%) still shows and the drift is reported.
    machine = SimMachine(_i3_speed, drift=1.25)
    rep = sim(machine).run()
    assert rep.baseline_tps == pytest.approx(10.12 * 1.25)
    assert rep.config.threads == 3 and rep.gain == pytest.approx(13.9 / 10.12 - 1)
    assert any("changed by" in w for w in rep.warnings)
    duel = [lists for _, lists, kw in machine.calls if not lists and kw.get("repetitions") == 1
            and not kw.get("n_depth") and not kw.get("n_prompt")]
    assert len(duel) >= 6


def test_one_disturbed_duel_run_does_not_decide(sim):
    """E2: a program starting mid-duel slows one winner run by 35%; the median of pairs ignores it."""
    machine = SimMachine(_i3_speed)
    seen = {"n": 0}

    def disturbed(model_path, base, lists, **kw):
        res = machine(model_path, base, lists, **kw)
        if not lists and kw.get("repetitions") == 1 and base.threads == 3:
            seen["n"] += 1
            if seen["n"] == 2:
                for r in res.records:
                    r.avg_ts *= 0.65
                    r.samples_ts = [r.avg_ts]
        return res

    tuner = sim(machine)
    tuner.sweep = disturbed
    rep = tuner.run()
    assert rep.config.threads == 3 and rep.gain > 0.3


def test_flash_attention_stays_on_with_a_quantized_kv_cache(sim, monkeypatch):
    # Qwen3.5-4B on the test PC only fit with a q8_0 KV cache; "-fa off" made llama-bench abort.
    tuner = sim(SimMachine(_i3_speed))
    tuner.report.gate = tuner.plan()
    tuner.report.gate.cache_type = "q8_0"
    machine = tuner.sweep
    rep = tuner.run()
    assert rep.status == "tuned"
    assert all("flash_attn" not in lists for _, lists, _ in machine.calls)


# ── priority (experiment E3: above normal = +13% and steadier on a busy PC) ──

def test_tune_measures_at_above_normal_priority(sim):
    machine = SimMachine(_i3_speed)
    sim(machine).run()
    assert machine.calls and all(base.priority == 1 for base, _, _ in machine.calls)


def test_priority_is_rendered_for_both_tools():
    from llama_autotune.compat import render_args
    bench_help = (FIXTURES / "llama-bench-b11347-help.txt").read_text()
    reg = parse_help(bench_help)
    assert render_args(SearchConfig(priority=1), "bench", reg).args == ["--prio", "1"]
    assert render_args(SearchConfig(priority=1), "server", None).args == ["--prio", "1", "--prio-batch", "1"]


def test_launch_defaults_raise_priority_on_cpu_only():
    from llama_autotune.heuristics import apply_launch_defaults
    from llama_autotune.models import Backend, HardwareInfo, ModelInfo
    cpu = apply_launch_defaults(SearchConfig(), HardwareInfo(backend=Backend.CPU, ram_gb=8), ModelInfo())
    gpu = apply_launch_defaults(SearchConfig(), HardwareInfo(backend=Backend.CUDA, ram_gb=8), ModelInfo())
    assert cpu.priority == 1 and gpu.priority is None


def test_flash_attention_off_wins_a_long_chat_despite_a_small_empty_loss(sim):
    """Real b12228 run: FA on was 3% faster empty but 40% slower after 2,048 tokens; off must win."""
    machine = SimMachine(lambda t, fa, rp: _i3_speed(t, None, rp) * (1.03 if fa in (True, None) else 1.0),
                         depth_factor=lambda fa: 0.55 if fa in (True, None) else 0.85)
    rep = sim(machine).run()
    assert rep.config.flash_attn is False
    note = next(n for n in rep.notes if n.startswith("Flash attention off"))
    assert "-3% in an empty one" in note


def test_best_bandwidth_ever_measured_is_used(sim, monkeypatch, tmp_path):
    """A busy PC measured 12.8 GB/s and claimed 121% efficiency; the PC really does 18 GB/s."""
    from llama_autotune.database import best_bandwidth
    conn = connect(str(tmp_path / "db.sqlite"))
    for bw in (15.1, 18.0, 12.8):
        save_tuned(conn, "hw", f"m{bw}", TunedSettings(config=SearchConfig(), bandwidth_gbs=bw))
    assert best_bandwidth(conn, "hw") == 18.0 and best_bandwidth(conn, "other") == 0.0
    monkeypatch.setattr(tuner_mod, "measure_bandwidth", lambda **kw: 12.8)
    tuner = sim(SimMachine(_i3_speed))
    tuner.known_bandwidth = 18.0
    rep = tuner.run()
    assert rep.bandwidth_gbs == 18.0 and rep.efficiency < 1.0


def test_moe_prediction_reads_only_the_used_experts():
    """An MoE token reads shared weights plus `used` of `count` experts, not every expert."""
    from llama_autotune.tuner import predict_generation_tps
    dense = ModelInfo(bytes_per_token=1000 * MIB, active_bytes_per_token=1000 * MIB)
    # 200 MiB shared + 800 MiB of experts, 8 of 32 used -> 400 MiB read per token
    moe = ModelInfo(bytes_per_token=1000 * MIB, active_bytes_per_token=400 * MIB, is_moe=True)
    assert predict_generation_tps(moe, 18.0) == pytest.approx(2.5 * predict_generation_tps(dense, 18.0))
    assert predict_generation_tps(ModelInfo(bytes_per_token=1000 * MIB), 18.0) == \
        pytest.approx(predict_generation_tps(dense, 18.0))   # older records without the field


def test_reported_speed_matches_the_saved_flash_attention(sim):
    """Granite MoE: FA off was 20% slower empty, 120% faster after 2,048 tokens; report the saved setting."""
    machine = SimMachine(lambda t, fa, rp: 47.8 * (1.0 if fa in (True, None) else 0.8),
                         depth_factor=lambda fa: 0.30 if fa in (True, None) else 0.84)
    rep = sim(machine).run()
    assert rep.config.flash_attn is False
    assert rep.best_tps == pytest.approx(47.8 * 0.8, rel=0.02)
    assert not any(n.startswith("llama.cpp's default settings are already") for n in rep.notes)
    assert any(n.startswith("Apart from flash attention") for n in rep.notes)


# ── models: what has been tuned on this PC ──────────────────────────

def test_list_tuned_keeps_the_newest_result_per_model_fastest_first(tmp_path):
    from llama_autotune.database import list_tuned
    conn = connect(str(tmp_path / "db.sqlite"))
    save_tuned(conn, "hw", "a", TunedSettings(config=SearchConfig(), generation_tps=9.0, model_path="a.gguf"))
    save_tuned(conn, "hw", "a", TunedSettings(config=SearchConfig(threads=3), generation_tps=12.0,
                                              depth_tps=10.3, depth=2048, verdict="Fast", model_path="a.gguf"))
    save_tuned(conn, "hw", "b", TunedSettings(config=SearchConfig(), generation_tps=47.8, model_path="b.gguf"))
    save_tuned(conn, "other-pc", "c", TunedSettings(config=SearchConfig(), generation_tps=99.0))
    rows = list_tuned(conn, "hw")
    assert [r.model_path for r in rows] == ["b.gguf", "a.gguf"]
    assert rows[1].config.threads == 3 and rows[1].depth_tps == 10.3 and rows[1].verdict == "Fast"


def test_models_command_lists_everything_measured(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from llama_autotune import cli
    from llama_autotune.database import get_session
    from llama_autotune.models import HardwareInfo
    from llama_autotune.resources import MemoryStatus
    monkeypatch.setattr(cli, "detect_capabilities", lambda: Capabilities())
    monkeypatch.setattr(cli, "detect_hardware", lambda caps=None: HardwareInfo())
    monkeypatch.setattr(cli, "hardware_fingerprint", lambda hw: "hw")
    monkeypatch.setattr(cli, "memory_status", lambda: MemoryStatus(total=8 * 1024**3, available=3000 * MIB))
    conn = get_session()
    save_tuned(conn, "hw", "m", TunedSettings(config=SearchConfig(threads=3, flash_attn=False),
                                              generation_tps=12.9, baseline_tps=12.3, prompt_tps=95.4,
                                              depth_tps=10.3, depth=2048, verdict="Fast", llama_build=11347,
                                              model_path=str(tmp_path / "gone.gguf")))
    conn.close()
    out = CliRunner().invoke(cli.app, ["models"]).output
    assert "gone" in out and "Fast" in out and "12.9 (+5%)" in out and "95.4" in out
    assert "10.3 after 2,048 tokens of chat" in out and "3 threads, flash attention off" in out
    assert "file missing" in out
    assert "No models tuned" in CliRunner().invoke(cli.app, ["models", "--fits-now"]).output or \
        "None of them fits" in CliRunner().invoke(cli.app, ["models", "--fits-now"]).output


def test_screen_stopped_before_any_generation_result_keeps_the_defaults(sim):
    """The watchdog or the time limit can stop the screen after a prompt test only."""
    machine = SimMachine(_i3_speed)

    def cut_short(model_path, base, lists, **kw):
        result = machine(model_path, base, lists, **kw)
        if "threads" in lists and "flash_attn" in lists:   # the screen
            result.records = [r for r in result.records if r.kind == "pp"][:1]
            result.error = "memory_guard_external"
        return result

    rep = sim(cut_short).run()
    assert rep.status == "tuned" and rep.config.threads is None
    assert rep.best_tps == pytest.approx(rep.baseline_tps)
    assert any("defaults are kept" in n for n in rep.notes)


def test_prompt_threads_when_the_winner_was_added_after_the_screen(sim):
    """The best thread count can be one tried only in the confirmation: it has no prompt speed yet."""
    hybrid = _hw(physical_cores=12, logical_cores=20, performance_cores=8, efficiency_cores=4,
                 performance_threads=16)                        # screens 2, 4, 6, 8, 16 threads
    rep = sim(SimMachine(lambda t, fa, rp: 10.0 - abs(t - 5)), hw=hybrid).run()
    assert rep.status == "tuned" and rep.config.threads == 5
