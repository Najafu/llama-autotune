"""Heuristic-based configuration generation for llama-autotune.

Provides functions that produce sane default configurations based on
detected hardware and model properties, without running benchmarks.
"""

from __future__ import annotations

from .models import Backend, HardwareInfo, ModelInfo, SearchConfig, SplitMode

#: llama.cpp --prio level used on CPUs (1 = above normal; measured in experiment E3).
ABOVE_NORMAL = 1


def generate_initial_config(hw: HardwareInfo, model: ModelInfo) -> SearchConfig:
    """Create a reasonable starting SearchConfig for the given hardware and model.

    Args:
        hw: Detected hardware information.
        model: Model metadata.

    Returns:
        A populated SearchConfig with conservative defaults.
    """
    config = SearchConfig()

    _set_basics(config, model)
    _set_ctx(config, model)

    if hw.backend == Backend.CPU:
        _configure_cpu(config, hw)
    else:
        _configure_gpu(config, hw)

    return config


def _set_basics(config: SearchConfig, model: ModelInfo) -> None:
    """Fill config with baseline parameter values.

    Args:
        config: The configuration object to mutate.
        model: Model metadata (currently unused but reserved).
    """
    config.flash_attn = True
    config.n_gpu_layers = 999
    config.batch_size = 2048
    config.ubatch_size = 512


def _set_ctx(config: SearchConfig, model: ModelInfo) -> None:
    """Set a sensible context size based on the model's training context.

    Args:
        config: The configuration object to mutate.
        model: Model metadata containing training_context.
    """
    ctx = model.training_context
    if ctx > 0:
        config.ctx_size = min(ctx, 4096)
    else:
        config.ctx_size = 4096


def _configure_cpu(config: SearchConfig, hw: HardwareInfo) -> None:
    """Tweak config values for CPU-only execution.

    On hybrid CPUs only the performance cores count: llama.cpp's own
    default includes efficiency cores on Windows, which can make
    generation several times slower.

    Args:
        config: The configuration object to mutate.
        hw: Hardware information used to set thread count.
    """
    config.threads = hw.performance_cores if hw.efficiency_cores else hw.physical_cores
    config.batch_size = min(config.batch_size or 2048, 512)
    config.ubatch_size = min(config.ubatch_size or 512, 256)
    config.n_gpu_layers = 0
    config.flash_attn = False


def _configure_gpu(config: SearchConfig, hw: HardwareInfo) -> None:
    """Tweak config values for GPU-accelerated execution.

    Args:
        config: The configuration object to mutate.
        hw: Hardware information used to set thread count and split mode.
    """
    config.threads = max(1, min(hw.logical_cores, hw.physical_cores))
    if hw.gpu_count > 1:
        config.split_mode = SplitMode.LAYER


def apply_launch_defaults(config: SearchConfig, hw: HardwareInfo, model: ModelInfo,
                          want_images: bool = False) -> SearchConfig:
    """Return a copy of ``config`` with safe llama-server defaults.

    llama-server's own defaults are tuned for servers, not small PCs:
    four parallel slots, an 8 GiB prompt cache in RAM, up to 32 context
    checkpoints, and the model's full training context. These choices
    keep a desktop session within its memory budget:

    * one slot (``-np 1``) for a single user,
    * the RAM prompt cache off on 8 GB machines and capped at 16 GB,
    * few context checkpoints for sliding-window and hybrid models,
    * no vision projector unless images are wanted,
    * above-normal priority on the CPU: +13% and far steadier speed while
      other programs keep half the CPU busy, with no gain from "high"
      (experiment E3 on the test i3).

    Args:
        config: Configuration to extend (its context should already be set).
        hw: Detected hardware.
        model: Inspected model.
        want_images: Keep the vision projector.

    Returns:
        A new :class:`SearchConfig`.
    """
    cfg = config.model_copy()
    if cfg.parallel is None:
        cfg.parallel = 1
    if cfg.cache_ram_mib is None and hw.ram_gb:
        if hw.ram_gb <= 8.5:
            cfg.cache_ram_mib = 0
        elif hw.ram_gb <= 16.5:
            cfg.cache_ram_mib = 1024
    layout = model.kv_layout
    has_window_state = layout is not None and (layout.swa_window or layout.recurrent_bytes)
    if cfg.ctx_checkpoints is None and has_window_state and hw.ram_gb <= 16.5:
        cfg.ctx_checkpoints = 2
    if not want_images and cfg.no_mmproj is None:
        cfg.no_mmproj = True
    if cfg.priority is None and hw.backend == Backend.CPU:
        cfg.priority = ABOVE_NORMAL
    return cfg


def to_cpu_config(config: SearchConfig) -> SearchConfig:
    """Return a copy of *config* with all GPU-related features disabled.

    Args:
        config: The source configuration to clone.

    Returns:
        A new SearchConfig with n_gpu_layers set to 0 and flash_attn disabled.
    """
    cfg = config.model_copy()
    cfg.n_gpu_layers = 0
    cfg.flash_attn = False
    return cfg
