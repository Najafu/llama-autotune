from __future__ import annotations

from .models import HardwareInfo, ModelInfo, SearchConfig


def estimate_vram(config: SearchConfig, model: ModelInfo, hw: HardwareInfo) -> float:
    model_size_gb = model.file_size_gb

    if model.quantization and "IQ" in model.quantization:
        overhead_factor = 1.1
    elif model.quantization and "Q4" in model.quantization:
        overhead_factor = 1.15
    elif model.quantization and "Q8" in model.quantization:
        overhead_factor = 1.2
    elif model.quantization and "F16" in model.quantization:
        overhead_factor = 1.3
    else:
        overhead_factor = 1.15

    kv_cache_gb = _estimate_kv_cache(config, model)
    estimated = model_size_gb * overhead_factor + kv_cache_gb

    return estimated


def _estimate_kv_cache(config: SearchConfig, model: ModelInfo) -> float:
    ctx = config.ctx_size or 4096
    n_layers = model.n_layers or 80
    n_heads = model.n_heads or 32
    kv_size = 2 * n_layers * n_heads * ctx * 2 * 2
    kv_cache_gb = kv_size / (1024**3)
    return kv_cache_gb


def is_oom(config: SearchConfig, model: ModelInfo, hw: HardwareInfo) -> bool:
    estimated = estimate_vram(config, model, hw)
    available = _available_vram(hw)
    return estimated > available


def _available_vram(hw: HardwareInfo) -> float:
    if hw.gpu_count > 0 and hw.vram_per_gpu:
        return sum(hw.vram_per_gpu) * 0.9
    return hw.ram_gb * 0.8


def is_plausible(config: SearchConfig, model: ModelInfo, hw: HardwareInfo) -> bool:
    if is_oom(config, model, hw):
        return False
    if config.threads is not None and config.threads > hw.logical_cores:
        return False
    return True


def detect_oom_in_output(output: str) -> bool:
    oom_patterns = [
        "out of memory",
        "oom",
        "cuda error",
        "cudamalloc",
        "cannot allocate",
        "failed to allocate",
    ]
    output_lower = output.lower()
    return any(p in output_lower for p in oom_patterns)
