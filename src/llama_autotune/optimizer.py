"""Three-stage LLM inference parameter optimizer.

Stage A — Rule-based baseline: evaluate the heuristic initial config; if it
fails, try simple thread-count fallbacks.

Stage B — Local grid search: sweep over the primary parameters (threads,
batch_size, ubatch_size, n_gpu_layers) one at a time, keeping the best so far.

Stage C — Bayesian optimisation: use Optuna's TPE sampler to explore the full
search space around the best config found by the earlier stages.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import optuna

from .benchmark import run_benchmark
from .constraints import detect_oom_in_output, is_plausible
from .hardware import detect_hardware
from .heuristics import generate_initial_config, to_cpu_config
from .model_inspector import inspect_model
from .models import (
    Backend,
    BenchmarkResult,
    HardwareInfo,
    ModelInfo,
    OptimizeObjective,
    SearchConfig,
)
from .search_space import ParamDef, config_from_params, get_search_space

logger = logging.getLogger(__name__)

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

        self.hw: HardwareInfo = detect_hardware()
        self.model: ModelInfo = inspect_model(model_path)
        self._initial_config: SearchConfig = generate_initial_config(self.hw, self.model)
        self._best_config: SearchConfig | None = None
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

        Uses a tiny prompt (64 tokens) and short generation (32 tokens)
        with a single repetition and a 20-second timeout.  Based on the
        measured generation throughput the method sets ``_speed_tier``,
        ``_n_prompt``, ``_n_gen``, and ``_bench_reps`` so that
        subsequent evaluations are scaled to the hardware.
        """
        cfg = SearchConfig()
        result = run_benchmark(self.model_path, cfg,
                               repetitions=1, timeout=20,
                               n_prompt=64, n_gen=32)
        if not result.success:
            self._speed_tier = "very_slow"
            logger.info("Speed probe failed — tier: very_slow")
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
        logger.info("Speed tier",
                    tier=self._speed_tier, gen_tps=round(tps, 2))

    def run(self) -> SearchConfig:
        """Run all three optimisation stages and return the best config.

        Stages are executed sequentially: baseline (A), then local search (B),
        then Bayesian (C). If stage A fails completely and no fallback works,
        the initial heuristic config is returned.

        Returns:
            The best SearchConfig found, or the initial heuristic config if
            no successful evaluation was produced.
        """
        logger.info(
            "Starting optimization",
            model=self.model_path,
            objective=self.objective.value,
            hw=self.hw.cpu_name,
        )

        self._estimate_speed()
        if self._speed_tier == "very_slow":
            logger.warning("Hardware too slow for benchmarking — using heuristic")
            return self._initial_config

        self._stage_a_baseline()
        if self._best_config is not None:
            self._stage_b_local_search()

        if self._best_config is not None:
            self._stage_c_bayesian()

        logger.info(
            "Optimization complete",
            best_score=self._best_score,
            total_evals=self._total_evals,
        )
        return self._best_config or self._initial_config

    def _stage_a_baseline(self) -> None:
        """Evaluate the rule-based initial config.

        If the config succeeds its score is recorded.  On failure the method
        falls through to the fallback configs.
        """
        logger.info("Stage A: rule-based baseline")
        result = self._evaluate(self._initial_config)
        if result.success:
            self._best_config = self._initial_config
            self._best_score = self._score(result)
            logger.info(
                "Baseline score",
                score=self._best_score,
                gen_tps=result.generation_tps,
                prompt_tps=result.prompt_tps,
            )
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
                self._best_score = self._score(result)
                logger.info("Fallback worked", score=self._best_score)
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
                if not is_plausible(cfg, self.model, self.hw):
                    continue
                result = self._evaluate(cfg)
                if result.success:
                    score = self._score(result)
                    if score > self._best_score:
                        self._best_config = cfg
                        self._best_score = score
                        logger.info(
                            "New best (Stage B)",
                            param=param_name,
                            val=val,
                            score=score,
                        )
                evals += 1

    def _stage_c_bayesian(self) -> None:
        """Bayesian optimisation over the full search space.

        Uses Optuna with a TPE sampler (seed 42) for reproducibility.
        Evaluates up to ``n_trials_stage_c`` configs and updates the best
        config if the study finds a superior score.
        """
        logger.info("Stage C: Bayesian optimization")
        space = get_search_space(self.hw, self.model, self.objective)

        def objective_fn(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {}
            for pname, pdef in space.items():
                params[pname] = self._sample_param(trial, pname, pdef)
            cfg = config_from_params(params, self._best_config)
            if not is_plausible(cfg, self.model, self.hw):
                return -1.0
            result = self._evaluate(cfg)
            if not result.success:
                return -1.0
            return self._score(result)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective_fn, n_trials=self.n_trials_stage_c, show_progress_bar=False)

        if study.best_value > self._best_score:
            best_params = study.best_params
            self._best_config = config_from_params(best_params, self._best_config)
            self._best_score = study.best_value
            logger.info("Bayesian improved", score=self._best_score)

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
        result = run_benchmark(self.model_path, config, timeout=900,
                               n_prompt=self._n_prompt, n_gen=self._n_gen,
                               repetitions=self._bench_reps)
        self._cache[key] = result

        if detect_oom_in_output(result.raw_output):
            result.success = False
            logger.warning("OOM detected", config=key)

        return result

    def _score(self, result: BenchmarkResult) -> float:
        """Compute a scalar score from a benchmark result based on the objective.

        Args:
            result: The benchmark result to score.

        Returns:
            A numerical score (higher is better).
        """
        if self.objective == OptimizeObjective.MAX_GENERATION_TPS:
            return result.generation_tps
        elif self.objective == OptimizeObjective.MAX_PROMPT_TPS:
            return result.prompt_tps
        elif self.objective == OptimizeObjective.MIN_LATENCY:
            return -result.startup_time
        elif self.objective == OptimizeObjective.MAX_EFFICIENCY:
            return result.generation_tps / max(result.memory_usage, 1)
        elif self.objective == OptimizeObjective.MAX_CONTEXT:
            return result.generation_tps
        else:
            return result.generation_tps

    def _grid_values(self, param: ParamDef, count: int) -> list[Any]:
        """Generate up to ``count`` evenly-spaced values for a parameter.

        For categorical parameters the first ``count`` categories are
        returned.  For numeric parameters with a step size the values are
        produced by repeated addition; otherwise a uniform step is computed.

        Args:
            param: The parameter definition.
            count: Maximum number of values to return.

        Returns:
            A list of parameter values to evaluate.
        """
        if param.is_categorical and param.categories:
            return param.categories[:count]
        if param.step:
            values = []
            v = param.low
            while v <= param.high and len(values) < count:
                values.append(int(v) if isinstance(param.low, int) else v)
                v += param.step
            return values
        step = max(1, (param.high - param.low) // (count - 1))
        return list(range(param.low, param.high + 1, step))

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
