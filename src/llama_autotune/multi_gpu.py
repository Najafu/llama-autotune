from __future__ import annotations

from .models import HardwareInfo, SearchConfig


def tune_tensor_split(hw: HardwareInfo) -> list[SearchConfig]:
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
    if hw.gpu_count <= 1:
        return config
    cfg = config.model_copy()
    if cfg.split_mode is None:
        cfg.split_mode = "layer"
    return cfg
