from llama_autotune.models import Backend, HardwareInfo, ModelInfo, OptimizeObjective, SearchConfig
from llama_autotune.search_space import config_from_params, get_search_space


def _hw() -> HardwareInfo:
    return HardwareInfo(
        cpu_name="Test CPU",
        physical_cores=16,
        logical_cores=32,
        ram_gb=64,
        backend=Backend.CPU,
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


def test_get_cpu_space():
    space = get_search_space(_hw(), _model(), OptimizeObjective.BALANCED)
    assert "threads" in space
    assert "batch_size" in space
    assert "ctx_size" in space


def test_get_gpu_space():
    hw = _hw()
    hw.backend = Backend.CUDA
    hw.gpu_count = 1
    hw.gpu_vendor = "nvidia"
    space = get_search_space(hw, _model(), OptimizeObjective.BALANCED)
    assert "n_gpu_layers" in space
    assert "batch_size" in space


def test_config_from_params():
    params = {"threads": 8, "batch_size": 1024}
    cfg = config_from_params(params)
    assert cfg.threads == 8
    assert cfg.batch_size == 1024


def test_config_from_params_with_base():
    base = SearchConfig(threads=16, ctx_size=4096)
    params = {"batch_size": 2048}
    cfg = config_from_params(params, base)
    assert cfg.threads == 16
    assert cfg.batch_size == 2048
    assert cfg.ctx_size == 4096


def test_max_context_objective():
    hw = _hw()
    space = get_search_space(hw, _model(), OptimizeObjective.MAX_CONTEXT)
    ctx = space.get("ctx_size")
    assert ctx is not None
    assert ctx.high >= 4096
