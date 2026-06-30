from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Backend, HardwareInfo, ModelInfo, OptimizeObjective, SearchConfig


@dataclass
class ParamDef:
    name: str
    low: int | float
    high: int | float
    step: int | float | None = None
    is_categorical: bool = False
    categories: list[Any] | None = None


def get_search_space(
    hw: HardwareInfo, model: ModelInfo, objective: OptimizeObjective
) -> dict[str, ParamDef]:
    space: dict[str, ParamDef] = {}

    if hw.backend == Backend.CPU:
        space.update(_cpu_space(hw, model, objective))
    else:
        space.update(_gpu_space(hw, model, objective))

    space.update(_memory_space(model, objective))
    space.update(_throughput_space(model, objective))

    return space


def _cpu_space(
    hw: HardwareInfo, model: ModelInfo, objective: OptimizeObjective
) -> dict[str, ParamDef]:
    space = {}
    min_threads = min(4, hw.physical_cores)
    step = 1 if hw.physical_cores <= 4 else 2
    space["threads"] = ParamDef("threads", min_threads, hw.physical_cores, step=step)
    return space


def _gpu_space(
    hw: HardwareInfo, model: ModelInfo, objective: OptimizeObjective
) -> dict[str, ParamDef]:
    space = {}
    max_layers = min(model.n_layers, 200) if model.n_layers > 0 else 200
    space["n_gpu_layers"] = ParamDef(
        "n_gpu_layers", 1, max_layers, step=max(1, max_layers // 4)
    )
    return space


def _memory_space(
    model: ModelInfo, objective: OptimizeObjective
) -> dict[str, ParamDef]:
    space = {}
    max_ctx = model.training_context if model.training_context > 0 else 8192
    if objective in (OptimizeObjective.MAX_CONTEXT,):
        space["ctx_size"] = ParamDef("ctx_size", 4096, min(max_ctx, 131072), step=4096)
    else:
        space["ctx_size"] = ParamDef("ctx_size", 1024, min(max_ctx, 32768), step=1024)
    return space


def _throughput_space(
    model: ModelInfo, objective: OptimizeObjective
) -> dict[str, ParamDef]:
    space = {}
    space["batch_size"] = ParamDef("batch_size", 128, 4096, step=128)
    space["ubatch_size"] = ParamDef("ubatch_size", 64, 1024, step=64)
    return space


def config_from_params(params: dict[str, Any], base: SearchConfig | None = None) -> SearchConfig:
    cfg = base.model_copy() if base is not None else SearchConfig()
    for key, value in params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
