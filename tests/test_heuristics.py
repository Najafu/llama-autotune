from llama_autotune.heuristics import generate_initial_config
from llama_autotune.models import Backend, HardwareInfo, ModelInfo


def _cpu_hw() -> HardwareInfo:
    return HardwareInfo(
        cpu_name="Test CPU",
        physical_cores=16,
        logical_cores=32,
        ram_gb=64,
        backend=Backend.CPU,
    )


def _gpu_hw() -> HardwareInfo:
    return HardwareInfo(
        cpu_name="Test GPU",
        physical_cores=16,
        logical_cores=32,
        ram_gb=64,
        gpu_count=1,
        gpu_vendor="nvidia",
        gpu_models=["RTX 4090"],
        vram_per_gpu=[24.0],
        backend=Backend.CUDA,
    )


def _model() -> ModelInfo:
    return ModelInfo(
        architecture="qwen3",
        parameters=30_000_000_000,
        quantization="Q4_K_M",
        n_layers=80,
        n_heads=32,
        training_context=131072,
        file_size_gb=3.2,
    )


def test_generate_cpu_config():
    cfg = generate_initial_config(_cpu_hw(), _model())
    assert cfg.threads == 16
    assert cfg.n_gpu_layers == 0
    assert cfg.flash_attn is False


def test_generate_gpu_config():
    cfg = generate_initial_config(_gpu_hw(), _model())
    assert cfg.flash_attn is True
    assert cfg.batch_size == 2048
    assert cfg.ubatch_size == 512
    assert cfg.ctx_size == 4096


def test_config_ctx_capped():
    model = _model()
    model.training_context = 1000000
    cfg = generate_initial_config(_gpu_hw(), model)
    assert cfg.ctx_size == 4096


def test_config_no_training_ctx():
    model = _model()
    model.training_context = 0
    cfg = generate_initial_config(_cpu_hw(), model)
    assert cfg.ctx_size == 4096


def test_hybrid_cpu_uses_performance_cores_only():
    hw = _cpu_hw()
    hw.physical_cores, hw.performance_cores, hw.efficiency_cores = 14, 6, 8
    assert generate_initial_config(hw, _model()).threads == 6


def test_launch_defaults_for_8gb_pc():
    from llama_autotune.heuristics import apply_launch_defaults
    from llama_autotune.models import KVLayout, SearchConfig

    hw = _cpu_hw()
    hw.ram_gb = 7.9
    model = _model()
    model.kv_layout = KVLayout(k_elems_full=1, v_elems_full=1, swa_window=512)
    cfg = apply_launch_defaults(SearchConfig(ctx_size=4096), hw, model)
    assert (cfg.parallel, cfg.cache_ram_mib, cfg.ctx_checkpoints, cfg.no_mmproj) == (1, 0, 2, True)


def test_launch_defaults_keep_explicit_values_and_images():
    from llama_autotune.heuristics import apply_launch_defaults
    from llama_autotune.models import SearchConfig

    cfg = apply_launch_defaults(SearchConfig(parallel=4), _cpu_hw(), _model(), want_images=True)
    assert cfg.parallel == 4
    assert cfg.no_mmproj is None
    assert cfg.cache_ram_mib is None          # 64 GB machine keeps llama.cpp's default
