"""Search-space definitions for hyper-parameter optimisation.

Defines the ``ParamDef`` data class and helper functions that build a
dictionary of tunable parameters (and their ranges) based on hardware,
model properties, and the optimisation objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Backend, HardwareInfo, ModelInfo, OptimizeObjective, SearchConfig


@dataclass
class ParamDef:
    """Description of a single tunable parameter.

    Attributes:
        name: Parameter name matching a ``SearchConfig`` field.
        low: Lower bound of the search range.
        high: Upper bound of the search range.
        step: Step size for continuous parameters. ``None`` when *is_categorical* is True.
        is_categorical: Whether the parameter uses discrete *categories* instead of a range.
        categories: Explicit list of values when *is_categorical* is True.
    """
    name: str
    low: int | float
    high: int | float
    step: int | float | None = None
    is_categorical: bool = False
    categories: list[Any] | None = None


def get_search_space(
    hw: HardwareInfo, model: ModelInfo, objective: OptimizeObjective
) -> dict[str, ParamDef]:
    """Build the full search space for the given hardware, model, and objective.

    Delegates to backend-specific helpers and always includes memory and
    throughput parameters.

    Args:
        hw: Detected hardware information.
        model: Model metadata.
        objective: Optimisation goal (e.g. latency, memory).

    Returns:
        Dictionary mapping parameter names to their ``ParamDef``.
    """
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
    """Build the CPU-specific portion of the search space.

    Currently only exposes the *threads* parameter.

    Args:
        hw: Hardware information used to derive thread bounds.
        model: Model metadata (currently unused but reserved).
        objective: Optimisation objective (currently unused but reserved).

    Returns:
        Dictionary with CPU-related ``ParamDef`` entries.
    """
    space = {}
    min_threads = min(4, hw.physical_cores)
    step = 1 if hw.physical_cores <= 4 else 2
    space["threads"] = ParamDef("threads", min_threads, hw.physical_cores, step=step)
    return space


def _gpu_space(
    hw: HardwareInfo, model: ModelInfo, objective: OptimizeObjective
) -> dict[str, ParamDef]:
    """Build the GPU-specific portion of the search space.

    Currently only exposes the *n_gpu_layers* parameter.

    Args:
        hw: Hardware information (currently unused but reserved).
        model: Model metadata used to derive layer bounds.
        objective: Optimisation objective (currently unused but reserved).

    Returns:
        Dictionary with GPU-related ``ParamDef`` entries.
    """
    space = {}
    max_layers = min(model.n_layers, 200) if model.n_layers > 0 else 200
    space["n_gpu_layers"] = ParamDef(
        "n_gpu_layers", 1, max_layers, step=max(1, max_layers // 4)
    )
    return space


def _memory_space(
    model: ModelInfo, objective: OptimizeObjective
) -> dict[str, ParamDef]:
    """Build the memory-related portion of the search space (context size).

    A larger range is used when the objective is ``MAX_CONTEXT``.

    Args:
        model: Model metadata used to cap context size.
        objective: Optimisation objective that influences the range.

    Returns:
        Dictionary with memory-related ``ParamDef`` entries.
    """
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
    """Build the throughput-related portion of the search space.

    Defines ranges for *batch_size* and *ubatch_size*.

    Args:
        model: Model metadata (currently unused but reserved).
        objective: Optimisation objective (currently unused but reserved).

    Returns:
        Dictionary with throughput-related ``ParamDef`` entries.
    """
    space = {}
    space["batch_size"] = ParamDef("batch_size", 128, 4096, step=128)
    space["ubatch_size"] = ParamDef("ubatch_size", 64, 1024, step=64)
    return space


def config_from_params(params: dict[str, Any], base: SearchConfig | None = None) -> SearchConfig:
    """Create a ``SearchConfig`` from a dictionary of parameter overrides.

    Starts from *base* (or a fresh ``SearchConfig``) and sets any attribute
    present in *params*.

    Args:
        params: Dictionary mapping attribute names to values.
        base: Optional base config to clone. If ``None`` a default config is used.

    Returns:
        A new ``SearchConfig`` with the supplied overrides applied.
    """
    cfg = base.model_copy() if base is not None else SearchConfig()
    for key, value in params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
