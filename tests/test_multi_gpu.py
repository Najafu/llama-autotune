from llama_autotune.models import HardwareInfo
from llama_autotune.multi_gpu import apply_multi_gpu_config, tune_tensor_split


def _hw_single() -> HardwareInfo:
    return HardwareInfo(gpu_count=1)


def _hw_dual() -> HardwareInfo:
    return HardwareInfo(gpu_count=2)


def _hw_quad() -> HardwareInfo:
    return HardwareInfo(gpu_count=4)


def test_single_gpu_returns_empty():
    configs = tune_tensor_split(_hw_single())
    assert configs == []


def test_dual_gpu_has_configs():
    configs = tune_tensor_split(_hw_dual())
    assert len(configs) > 0
    assert all(c.split_mode == "layer" for c in configs)
    assert all(c.tensor_split for c in configs)


def test_quad_gpu_configs():
    configs = tune_tensor_split(_hw_quad())
    assert len(configs) > 0


def test_apply_multi_gpu_single():
    from llama_autotune.models import SearchConfig
    cfg = SearchConfig()
    result = apply_multi_gpu_config(cfg, _hw_single())
    assert result.split_mode is None


def test_apply_multi_gpu_dual():
    from llama_autotune.models import SearchConfig
    cfg = SearchConfig()
    result = apply_multi_gpu_config(cfg, _hw_dual())
    assert result.split_mode == "layer"
