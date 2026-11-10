"""Three-stage LLM inference parameter optimizer.

Stage A — Rule-based baseline: evaluate the heuristic initial config; if it
fails, try simple thread-count fallbacks.

Stage B — Local grid search: sweep over the primary parameters (threads,
batch_size, ubatch_size, n_gpu_layers) one at a time, keeping the best so far.

Stage C — Bayesian optimisation: use Optuna's TPE sampler to explore the full
search space around the best config found by the earlier stages.

Finally the winner and the baseline take turns once more; the winner is kept
only when it still leads by more than the measurement noise, because the best
of many single noisy measurements is partly luck.
"""

from __future__ import annotations

import logging
import math
import time
from statistics import mean
from typing import TYPE_CHECKING, Any

from .benchmark import run_benchmark
from .compat import detect_capabilities
from .constraints import GateDecision, detect_oom_in_output, is_plausible, memory_budget, plan_memory
from .hardware import detect_hardware
from .heuristics import generate_initial_config, to_cpu_config
from .memory import compute_model_for, tensor_sizes_of
from .model_inspector import inspect_model
from .resources import memory_status
from .models import (
    Backend,
    BenchmarkResult,
    HardwareInfo,
    ModelInfo,
    OptimizeObjective,
    SearchConfig,
)
from .search_space import ParamDef, config_from_params, get_search_space

if TYPE_CHECKING:  # optuna takes over half a second to import; only stage C needs it
    import optuna

logger = logging.getLogger(__name__)

#: A searched configuration replaces the baseline only when it wins by more than this.
CONFIRM_MARGIN = 0.04

class Optimizer:
    """Three-stage parameter optimizer for llama.cpp.

    Sequences a heuristic baseline (stage A), a local grid search (stage B),
    and a Bayesian optimisation pass (stage C) to find the best inference
    parameters for a given model and hardware combination.
    """

    def __init__(
        self,
        model_path: str,
        objective: OptimizeObjective = OptimizeObjective.BALANCED,
        n_trials_stage_b: int = 15,
        n_trials_stage_c: int = 40,
        llama_dir: str | None = None,
        slow: bool = False,
    ):
        """Initialise the optimizer.

        Args:
            model_path: Path to the GGUF model file.
            objective: Optimisation objective (e.g. balanced, max throughput).
            n_trials_stage_b: Maximum evaluations for the grid-search stage.
            n_trials_stage_c: Maximum evaluations for the Bayesian stage.
            llama_dir: Optional path to a custom llama.cpp directory. When set,
                the ``LLAMA_CPP_DIR`` environment variable is updated.
            slow: If True, fallback configs are tried even when the baseline is
                too slow for the host machine.
        """
        self.model_path = model_path
        self.objective = objective
        self.n_trials_stage_b = n_trials_stage_b
        self.n_trials_stage_c = n_trials_stage_c
        self.slow = slow

        if llama_dir:
            import os

            os.environ["LLAMA_CPP_DIR"] = llama_dir

        self.caps = detect_capabilities()
        self.hw: HardwareInfo = detect_hardware(self.caps)
        self.model: ModelInfo = inspect_model(model_path)
        self._initial_config: SearchConfig = generate_initial_config(self.hw, self.model)
        self.gate: GateDecision | None = None
        self._memory_check: dict[str, Any] = {}
        if self.hw.backend == Backend.CPU:
            mem = memory_status()
            ctx = self._initial_config.ctx_size or 4096
            self._memory_check = {
                "budget": memory_budget(mem.available, mem.total),
                "tensor_sizes": tensor_sizes_of(model_path),
                # llama-bench keeps logits for every token: the larger, safe figure.
                "compute_model": compute_model_for(model_path, self.model),
            }
            self.gate = plan_memory(
                self.model, isa_tier=self.hw.isa_tier or "avx2", ctx=ctx,
                ubatch=self._initial_config.ubatch_size or 512, **self._memory_check,
            )
            self._initial_config = self.gate.apply(self._initial_config)
        self._best_config: SearchConfig | None = None
        self._baseline_config: SearchConfig | None = None
        self._best_score: float = 0.0
        self._total_evals: int = 0
        self._cache: dict[str, BenchmarkResult] = {}

        self._speed_tier: str = "unknown"
        self._speed_estimate: float = 0.0
        self._n_prompt: int = 512
        self._n_gen: int = 128
        self._bench_reps: int = 3

    def _estimate_speed(self) -> None:
        """Run a minimal benchmark to determine hardware speed tier.

        Uses a small prompt (16 tokens) and short generation (8 tokens)
        with a single repetition and no warmup. The time limit allows for
        loading the model from a slow disk (15 s per GB on top of a minute),
        so a large file is not mistaken for a slow CPU. Based on the measured generation throughput the method sets
        ``_speed_tier``, ``_n_prompt``, ``_n_gen``, and ``_bench_reps``
        so that subsequent evaluations are scaled to the hardware.
        """
        cfg = self._initial_config.model_copy()
        result = run_benchmark(self.model_path, cfg,
                               repetitions=1, timeout=60 + int(15 * self.model.file_size_gb),
                               n_prompt=16, n_gen=8, no_warmup=True,
                               registry=self.caps.bench)
        if not result.success:
            self._speed_tier = "very_slow" if result.error == "timeout" else "failed"
            logger.warning(f"Speed probe failed ({result.error or 'no result'}) — tier: {self._speed_tier}")
            return

        tps = result.generation_tps
        if tps < 1:
            self._speed_tier = "very_slow"
        elif tps < 4:
            self._speed_tier = "slow"
            self._n_prompt, self._n_gen, self._bench_reps = 64, 32, 1
        elif tps < 15:
            self._speed_tier = "medium"
            self._n_prompt, self._n_gen, self._bench_reps = 256, 64, 2
        else:
            self._speed_tier = "fast"
        self._speed_estimate = tps
        logger.info(f"Speed tier: {self._speed_tier} (gen_tps={round(tps, 2)})")

    def run(self) -> SearchConfig:
        """Run all three optimisation stages and return the best config.

        Stages are executed sequentially: baseline (A), then local search (B),
        then Bayesian (C). If stage A fails completely and no fallback works,
        the initial heuristic config is returned.

        Returns:
            The best SearchConfig found, or the initial heuristic config if
            no successful evaluation was produced.
        """
        logger.info(f"Starting optimization — model={self.model_path} objective={self.objective.value} hw={self.hw.cpu_name}")

        if self.gate is not None and not self.gate.fits:
            logger.warning("Model does not fit in available memory — not benchmarking")
            return self._initial_config

        self._estimate_speed()
        if self._speed_tier in ("very_slow", "failed"):
            logger.warning("Hardware too slow for benchmarking — using heuristic" if self._speed_tier == "very_slow"
                           else "llama-bench could not run this model — using heuristic")
            self._n_prompt, self._n_gen, self._bench_reps = 64, 32, 1
            return self._initial_config

        self._stage_a_baseline()
        if self._best_config is not None:
            self._stage_b_local_search()

        if self._best_config is not None:
            self._stage_c_bayesian()
        self._confirm_winner()

        logger.info(f"Optimization complete — best_score={self._best_score} total_evals={self._total_evals}")
        return self._best_config or self._initial_config

    def _stage_a_baseline(self) -> None:
        """Evaluate the rule-based initial config.

        If the config succeeds its score is recorded.  On failure the method
        falls through to the fallback configs.
        """
        logger.info("Stage A: rule-based baseline")
        result = self._evaluate(self._initial_config)
        if result.success:
            self._best_config = self._baseline_config = self._initial_config
            self._best_score = self._score(result, self._initial_config)
            logger.info(f"Baseline score={self._best_score} gen_tps={result.generation_tps} prompt_tps={result.prompt_tps}")
        else:
            logger.warning("Baseline config failed, trying fallbacks")
            self._try_fallback_configs()

    def _try_fallback_configs(self) -> None:
        """Iterate generated fallback configs and use the first one that works.

        Once a fallback succeeds it becomes the new best config and iteration
        stops.
        """
        for cfg in self._generate_fallbacks():
            result = self._evaluate(cfg)
            if result.success:
                self._best_config = cfg
                self._best_score = self._score(result, cfg)
                logger.info(f"Fallback worked (score={self._best_score})")
                return

    def _generate_fallbacks(self) -> list[SearchConfig]:
        """Build a list of fallback configurations.

        Produces configs with varying thread counts (physical cores, logical
        cores, half physical cores), first for the initial heuristic config
        and then for a CPU-only variant.

        Returns:
            A list of fallback SearchConfig objects.
        """
        fallbacks = []
        threads_opts = sorted({
            self.hw.physical_cores,
            self.hw.logical_cores,
            max(1, self.hw.physical_cores // 2),
        }, reverse=True)
        for t in threads_opts:
            cfg = self._initial_config.model_copy()
            cfg.threads = t
            fallbacks.append(cfg)
        cpu = to_cpu_config(self._initial_config)
        for t in threads_opts:
            cfg = cpu.model_copy()
            cfg.threads = t
            fallbacks.append(cfg)
        return fallbacks

    def _stage_b_local_search(self) -> None:
        """Local grid search over primary parameters.

        Sweeps ``threads``, ``batch_size``, ``ubatch_size``, and
        ``n_gpu_layers`` one parameter at a time, evaluating up to
        ``n_trials_stage_b`` configs.  The best config is updated whenever a
        higher-scoring combination is found.
        """
        logger.info("Stage B: local grid search")
        space = get_search_space(self.hw, self.model, self.objective)
        primary_params = ["threads", "batch_size", "ubatch_size", "n_gpu_layers"]

        evals = 0
        for param_name in primary_params:
            if evals >= self.n_trials_stage_b:
                break
            if param_name not in space:
                continue
            param = space[param_name]
            values = self._grid_values(param, count=5)
            for val in values:
                if evals >= self.n_trials_stage_b:
                    break
                cfg = self._best_config.model_copy() if self._best_config else SearchConfig()
                setattr(cfg, param_name, val)
                if not is_plausible(cfg, self.model, self.hw, **self._memory_check):
                    continue
                result = self._evaluate(cfg)
                if result.success:
                    score = self._score(result, cfg)
                    if score > self._best_score:
                        self._best_config = cfg
                        self._best_score = score
                        logger.info(f"New best (Stage B) — {param_name}={val} score={score}")
                evals += 1

    def _stage_c_bayesian(self) -> None:
        """Bayesian optimisation over the full search space.

        Uses Optuna with a TPE sampler (seed 42) for reproducibility.
        Evaluates up to ``n_trials_stage_c`` configs and updates the best
        config if the study finds a superior score.
        """
        import optuna

        logger.info("Stage C: Bayesian optimization")
        space = get_search_space(self.hw, self.model, self.objective)

        def objective_fn(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {}
            for pname, pdef in space.items():
                if pdef.measured:
                    params[pname] = self._sample_param(trial, pname, pdef)
            cfg = config_from_params(params, self._best_config)
            # Pruning (rather than a sentinel score) keeps failed configs out
            # of best_value — a fixed sentinel would outrank legitimate
            # negative scores such as min_latency's -startup_time.
            if not is_plausible(cfg, self.model, self.hw, **self._memory_check):
                raise optuna.TrialPruned()
            result = self._evaluate(cfg)
            if not result.success:
                raise optuna.TrialPruned()
            return self._score(result, cfg)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective_fn, n_trials=self.n_trials_stage_c, show_progress_bar=False)

        try:
            best_value = study.best_value
        except ValueError:
            logger.info("Bayesian stage produced no successful trials")
            return

        if best_value > self._best_score:
            self._best_config = config_from_params(study.best_params, self._best_config)
            self._best_score = best_value
            logger.info(f"Bayesian improved (score={self._best_score})")

    def _confirm_winner(self) -> None:
        """Re-measure the winner against the baseline, taking turns.

        The search keeps the best of many single measurements, and the best
        of many noisy numbers is partly luck. Two alternating rounds expose
        both to the same disturbances; the winner stays only when its mean
        score still leads by more than :data:`CONFIRM_MARGIN`.
        """
        base, best = self._baseline_config, self._best_config
        if base is None or best is None or best == base:
            return
        logger.info("Confirming the best config against the baseline")
        scores: dict[str, list[float]] = {"base": [], "best": []}
        for _ in range(2):
            for name, cfg in (("base", base), ("best", best)):
                result = self.measure(cfg)
                if result.success:
                    scores[name].append(self._score(result, cfg))
        if not scores["base"] or not scores["best"]:
            return
        base_score, best_score = mean(scores["base"]), mean(scores["best"])
        if best_score - base_score > CONFIRM_MARGIN * abs(base_score):
            self._best_score = best_score
        else:
            logger.warning(f"The searched config did not hold up when re-measured "
                           f"({best_score:.2f} vs {base_score:.2f}): keeping the baseline")
            self._best_config, self._best_score = base, base_score

    def measure(self, config: SearchConfig) -> BenchmarkResult:
        """One fresh benchmark of ``config`` at this machine's test sizes (not cached)."""
        return run_benchmark(self.model_path, config, timeout=900,
                             n_prompt=self._n_prompt, n_gen=self._n_gen,
                             repetitions=self._bench_reps, registry=self.caps.bench)

    def _evaluate(self, config: SearchConfig) -> BenchmarkResult:
        """Run a benchmark for the given config, caching the result.

        Uses the speed-tiered prompt/gen token counts and repetitions
        set by :meth:`_estimate_speed`.

        Args:
            config: The configuration to evaluate.

        Returns:
            A BenchmarkResult with timing and memory information.
        """
        key = self._config_key(config)
        if key in self._cache:
            return self._cache[key]

        self._total_evals += 1
        for attempt in range(2):
            result = self.measure(config)
            if result.error != "memory_guard_external" or attempt:
                break
            logger.warning("Another program used RAM during the benchmark; retrying in 10 s")
            time.sleep(10)
        self._cache[key] = result

        if detect_oom_in_output(result.raw_output):
            result.success = False
            logger.warning(f"OOM detected — {key}")

        return result

    def _score(self, result: BenchmarkResult, config: SearchConfig) -> float:
        """Compute a scalar score (higher is better) for the objective.

        * ``max_generation_tps``: writing speed.
        * ``max_prompt_tps``: prompt reading speed.
        * ``balanced``: geometric mean of both, so neither is traded away.
        * ``min_latency``: minus the seconds until the first word appears for
          the test prompt (reading it, then one token).
        * ``max_efficiency``: writing speed per MiB of peak memory.
        * ``max_context``: writing speed. llama-bench cannot measure a context
          size, so the context is decided by the memory gate (``launch -c
          max``), not searched.
        """
        gen, prompt = result.generation_tps, result.prompt_tps
        if self.objective == OptimizeObjective.MAX_PROMPT_TPS:
            return prompt
        if self.objective == OptimizeObjective.MIN_LATENCY:
            if gen > 0 and prompt > 0:
                return -(self._n_prompt / prompt + 1 / gen)
            return -result.startup_time
        if self.objective == OptimizeObjective.MAX_EFFICIENCY:
            return gen / max(result.memory_usage, 1)
        if self.objective == OptimizeObjective.BALANCED and gen > 0 and prompt > 0:
            return math.sqrt(gen * prompt)
        return gen

    def _grid_values(self, param: ParamDef, count: int) -> list[Any]:
        """Up to ``count`` values spread evenly over the whole range of a parameter.

        Categorical parameters return their first ``count`` categories.
        Numeric values are snapped to the parameter's step, so the lowest and
        the highest value of the range are always tried.
        """
        if param.is_categorical and param.categories:
            return param.categories[:count]
        low, high = int(param.low), int(param.high)
        step = int(param.step or 1)
        n_steps = max(0, (high - low) // step)
        picks = min(count, n_steps + 1)
        if picks <= 1:
            return [low]
        return sorted({low + round(i * n_steps / (picks - 1)) * step for i in range(picks)})

    def _sample_param(
        self, trial: optuna.Trial, name: str, pdef: ParamDef
    ) -> Any:
        """Sample a parameter value for an Optuna trial.

        Delegates to the appropriate ``suggest_*`` method based on whether the
        parameter is categorical or integer-valued.

        Args:
            trial: The Optuna trial object.
            name: The parameter name.
            pdef: The parameter definition from the search space.

        Returns:
            A sampled value for the parameter.
        """
        if pdef.is_categorical and pdef.categories:
            return trial.suggest_categorical(name, pdef.categories)
        if pdef.step:
            return trial.suggest_int(name, pdef.low, pdef.high, step=int(pdef.step))
        return trial.suggest_int(name, pdef.low, pdef.high)

    def _config_key(self, config: SearchConfig) -> str:
        """Return a deterministic cache key for a config.

        Args:
            config: The search configuration.

        Returns:
            A JSON string uniquely representing the config.
        """
        return config.model_dump_json()

    @property
    def best_config(self) -> SearchConfig | None:
        """Return the best configuration found so far.

        Returns:
            The best SearchConfig, or None if no successful evaluation has
            been performed.
        """
        return self._best_config

    @property
    def best_score(self) -> float:
        """Return the score of the best configuration.

        Returns:
            The score associated with ``best_config`` (0.0 if none).
        """
        return self._best_score

    @property
    def total_evals(self) -> int:
        """Return the total number of benchmark evaluations performed.

        Returns:
            The cumulative evaluation count across all stages.
        """
        return self._total_evals
