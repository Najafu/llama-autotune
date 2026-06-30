from __future__ import annotations

from .models import Backend, HardwareInfo, ModelInfo, SearchConfig


def generate_initial_config(hw: HardwareInfo, model: ModelInfo) -> SearchConfig:
    config = SearchConfig()

    _set_basics(config, model)
    _set_ctx(config, model)

    if hw.backend == Backend.CPU:
        _configure_cpu(config, hw)
    else:
        _configure_gpu(config, hw)

    return config


def _set_basics(config: SearchConfig, model: ModelInfo) -> None:
    config.flash_attn = True
    config.n_gpu_layers = 999
    config.batch_size = 2048
    config.ubatch_size = 512


def _set_ctx(config: SearchConfig, model: ModelInfo) -> None:
    ctx = model.training_context
    if ctx > 0:
        config.ctx_size = min(ctx, 4096)
    else:
        config.ctx_size = 4096


def _configure_cpu(config: SearchConfig, hw: HardwareInfo) -> None:
    config.threads = hw.physical_cores
    config.batch_size = min(config.batch_size or 2048, 512)
    config.ubatch_size = min(config.ubatch_size or 512, 256)
    config.n_gpu_layers = 0
    config.flash_attn = False


def _configure_gpu(config: SearchConfig, hw: HardwareInfo) -> None:
    config.threads = max(hw.physical_cores - 2, 4)
    if hw.gpu_count > 1:
        config.split_mode = "layer"


def apply_search_config(hw: HardwareInfo, model: ModelInfo, base: SearchConfig) -> SearchConfig:
    cfg = base.model_copy()
    cfg.threads = hw.physical_cores
    return cfg
