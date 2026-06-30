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
    BenchmarkResult,
    HardwareInfo,
    ModelInfo,
    OptimizeObjective,
    SearchConfig,
)
from .search_space import ParamDef, config_from_params, get_search_space

logger = logging.getLogger(__name__)


class Optimizer:
    def __init__(
        self,
        model_path: str,
        objective: OptimizeObjective = OptimizeObjective.BALANCED,
        n_trials_stage_b: int = 15,
        n_trials_stage_c: int = 40,
        llama_dir: str | None = None,
    ):
        self.model_path = model_path
        self.objective = objective
        self.n_trials_stage_b = n_trials_stage_b
        self.n_trials_stage_c = n_trials_stage_c

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

    def run(self) -> SearchConfig:
        logger.info(
            "Starting optimization",
            model=self.model_path,
            objective=self.objective.value,
            hw=self.hw.cpu_name,
        )

        self._stage_a_baseline()
        if self._best_config is not None:
            self._stage_b_local_search()
        self._stage_c_bayesian()

        logger.info(
            "Optimization complete",
            best_score=self._best_score,
            total_evals=self._total_evals,
        )
        return self._best_config or self._initial_config

    def _stage_a_baseline(self) -> None:
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
        for cfg in self._generate_fallbacks():
            result = self._evaluate(cfg)
            if result.success:
                self._best_config = cfg
                self._best_score = self._score(result)
                logger.info("Fallback worked", score=self._best_score)
                return

    def _generate_fallbacks(self) -> list[SearchConfig]:
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
        key = self._config_key(config)
        if key in self._cache:
            return self._cache[key]

        self._total_evals += 1
        result = run_benchmark(self.model_path, config)
        self._cache[key] = result

        if detect_oom_in_output(result.raw_output):
            result.success = False
            logger.warning("OOM detected", config=key)

        return result

    def _score(self, result: BenchmarkResult) -> float:
        if self.objective == OptimizeObjective.MAX_GENERATION_TPS:
            return result.generation_tps
        elif self.objective == OptimizeObjective.MAX_PROMPT_TPS:
            return result.prompt_tps
        elif self.objective == OptimizeObjective.MIN_LATENCY:
            return result.startup_time
        elif self.objective == OptimizeObjective.MAX_EFFICIENCY:
            return result.generation_tps / max(result.memory_usage, 1)
        elif self.objective == OptimizeObjective.MAX_CONTEXT:
            return result.generation_tps
        else:
            return result.generation_tps

    def _grid_values(self, param: ParamDef, count: int) -> list[Any]:
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
        if pdef.is_categorical and pdef.categories:
            return trial.suggest_categorical(name, pdef.categories)
        if pdef.step:
            return trial.suggest_int(name, pdef.low, pdef.high, step=int(pdef.step))
        return trial.suggest_int(name, pdef.low, pdef.high)

    def _config_key(self, config: SearchConfig) -> str:
        return config.model_dump_json()

    @property
    def best_config(self) -> SearchConfig | None:
        return self._best_config

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def total_evals(self) -> int:
        return self._total_evals
