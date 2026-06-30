"""Multi-GPU tensor-split configuration generation.

Provides utilities that generate candidate ``SearchConfig`` instances with
various tensor-split ratios for multi-GPU setups.
"""

from __future__ import annotations

from .models import HardwareInfo, SearchConfig


def tune_tensor_split(hw: HardwareInfo) -> list[SearchConfig]:
    """Generate a list of candidate configs with different tensor-split ratios.

    Only produces configs when the system has more than one GPU.

    Args:
        hw: Hardware information containing the GPU count.

    Returns:
        List of ``SearchConfig`` instances, each with a distinct tensor split.
    """
    if hw.gpu_count <= 1:
        return []

    configs: list[SearchConfig] = []
    ratios = _generate_split_ratios(hw.gpu_count)

    for ratio in ratios:
        cfg = SearchConfig()
        cfg.split_mode = "layer"
        cfg.tensor_split = ratio
        configs.append(cfg)

    return configs


def _generate_split_ratios(gpu_count: int) -> list[str]:
    """Produce a list of heuristic tensor-split ratio strings.

    Ratios are expressed as comma-separated integers (e.g. ``"1,1"``
    for two GPUs).

    Args:
        gpu_count: Number of available GPUs.

    Returns:
        List of ratio strings ordered by expected quality.
    """
    ratios: list[str] = []

    if gpu_count == 2:
        ratios.extend(["1,1", "3,1", "1,3", "2,1", "1,2"])
    elif gpu_count == 3:
        ratios.extend(["1,1,1", "2,1,1", "1,2,1", "1,1,2", "3,1,1"])
    elif gpu_count >= 4:
        ratios.extend(["1,1,1,1", "2,1,1,1", "1,2,1,1", "3,1,1,1"])
    else:
        ratios.append(",".join(["1"] * gpu_count))

    return ratios


def apply_multi_gpu_config(config: SearchConfig, hw: HardwareInfo) -> SearchConfig:
    """Return a copy of *config* with multi-GPU split mode enabled if applicable.

    If the system has only one GPU the original config is returned unchanged.

    Args:
        config: The base configuration to clone.
        hw: Hardware information containing the GPU count.

    Returns:
        A new ``SearchConfig`` with ``split_mode`` set to ``"layer"`` when
        multiple GPUs are present.
    """
    if hw.gpu_count <= 1:
        return config
    cfg = config.model_copy()
    if cfg.split_mode is None:
        cfg.split_mode = "layer"
    return cfg
