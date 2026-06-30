from llama_autotune.models import (
    Backend,
    BenchmarkResult,
    GpuVendor,
    HardwareInfo,
    LaunchProfile,
    ModelInfo,
    OptimizeObjective,
    SearchConfig,
    SplitMode,
)


def test_hardware_info_defaults():
    hw = HardwareInfo()
    assert hw.physical_cores == 0
    assert hw.backend == Backend.CPU


def test_hardware_info_full():
    hw = HardwareInfo(
        cpu_name="Test CPU",
        physical_cores=8,
        logical_cores=16,
        ram_gb=32.0,
        gpu_count=1,
        gpu_vendor=GpuVendor.NVIDIA,
        gpu_models=["RTX 4090"],
        vram_per_gpu=[24.0],
        backend=Backend.CUDA,
    )
    assert hw.cpu_name == "Test CPU"
    assert hw.gpu_vendor == GpuVendor.NVIDIA
    assert hw.backend == Backend.CUDA


def test_model_info():
    mi = ModelInfo(
        path="/models/test.gguf",
        architecture="qwen3",
        parameters=30_000_000_000,
        quantization="Q4_K_M",
        n_layers=80,
        n_heads=32,
        training_context=131072,
        is_moe=True,
        active_parameters=3_000_000_000,
        file_size_gb=3.2,
    )
    assert mi.is_moe
    assert mi.active_parameters < mi.parameters


def test_benchmark_result():
    br = BenchmarkResult(
        prompt_tps=5000.0,
        generation_tps=102.5,
        startup_time=2.3,
        memory_usage=8192.0,
        success=True,
    )
    assert br.success
    assert br.generation_tps == 102.5


def test_search_config_to_llama_args():
    cfg = SearchConfig(
        threads=12,
        batch_size=2048,
        ubatch_size=512,
        n_gpu_layers=999,
        flash_attn=True,
        ctx_size=4096,
    )
    args = cfg.to_llama_args()
    assert "-t" in args
    assert "12" in args
    assert "-b" in args
    assert "2048" in args
    assert "-fa" in args
    assert "1" in args


def test_search_config_empty():
    cfg = SearchConfig()
    assert cfg.to_llama_args() == []


def test_optimize_objective_values():
    assert OptimizeObjective("max_generation_tps")
    assert OptimizeObjective("max_prompt_tps")
    assert OptimizeObjective("balanced")


def test_split_mode_values():
    assert SplitMode.NONE.value == "none"
    assert SplitMode.LAYER.value == "layer"
    assert SplitMode.ROW.value == "row"
    assert SplitMode.TENSOR.value == "tensor"


def test_gpu_vendor():
    assert GpuVendor.NVIDIA.value == "nvidia"
    assert GpuVendor.UNKNOWN.value == "unknown"


def test_backend_enum():
    assert Backend.CPU.value == "cpu"
    assert Backend.CUDA.value == "cuda"


def test_launch_profile():
    lp = LaunchProfile(
        name="test_profile",
        args=["--flash-attn", "--gpu-layers", "all"],
        model_path="/models/test.gguf",
        hardware="RTX 4090",
        score=100.0,
    )
    assert lp.name == "test_profile"
    assert "--flash-attn" in lp.args
