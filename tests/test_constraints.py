from llama_autotune.constraints import (
    detect_oom_in_output,
    estimate_vram,
    is_oom,
    is_plausible,
)
from llama_autotune.models import HardwareInfo, ModelInfo, SearchConfig


def _hw() -> HardwareInfo:
    return HardwareInfo(
        cpu_name="Test",
        physical_cores=16,
        logical_cores=32,
        ram_gb=64,
        gpu_count=1,
        vram_per_gpu=[24.0],
    )


def _model() -> ModelInfo:
    return ModelInfo(
        architecture="test",
        parameters=7_000_000_000,
        quantization="Q4_K_M",
        n_layers=32,
        n_heads=32,
        file_size_gb=4.5,
    )


def test_estimate_vram():
    config = SearchConfig(ctx_size=4096)
    hw = _hw()
    model = _model()
    estimated = estimate_vram(config, model, hw)
    assert estimated > 0


def test_is_oom_false():
    config = SearchConfig(ctx_size=4096)
    hw = _hw()
    model = _model()
    assert not is_oom(config, model, hw)


def test_is_plausible_too_many_threads():
    hw = _hw()
    hw.logical_cores = 8
    config = SearchConfig(threads=16)
    assert not is_plausible(config, _model(), hw)


def test_is_plausible_valid():
    config = SearchConfig(threads=8, ctx_size=4096)
    assert is_plausible(config, _model(), _hw())


def test_detect_oom_in_output():
    assert detect_oom_in_output("CUDA error: out of memory")
    assert detect_oom_in_output("failed to allocate memory")
    assert detect_oom_in_output("OOM")
    assert not detect_oom_in_output("normal output")
    assert not detect_oom_in_output("")
