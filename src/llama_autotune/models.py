"""Pydantic data models for the llama-autotune domain.

Defines all structured data types used throughout the codebase:
hardware profiles, model metadata, benchmark results, search
configurations, and launch profiles.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Backend(str, Enum):
    """Supported llama.cpp compute backends."""

    CUDA = "cuda"
    ROCM = "rocm"
    VULKAN = "vulkan"
    METAL = "metal"
    SYCL = "sycl"
    OPENCL = "opencl"
    CPU = "cpu"


class GpuVendor(str, Enum):
    """GPU manufacturer identifiers for hardware detection."""

    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


class OptimizeObjective(str, Enum):
    """Optimization objectives the search can target.

    Each objective tunes the search toward a different performance
    characteristic (generation speed, prompt processing, latency, etc.).
    """

    MAX_GENERATION_TPS = "max_generation_tps"
    MAX_PROMPT_TPS = "max_prompt_tps"
    MIN_LATENCY = "min_latency"
    MAX_CONTEXT = "max_context"
    MAX_EFFICIENCY = "max_efficiency"
    BALANCED = "balanced"


class SplitMode(str, Enum):
    """Multi-GPU split modes supported by llama.cpp."""

    NONE = "none"
    LAYER = "layer"
    ROW = "row"
    TENSOR = "tensor"


class HardwareInfo(BaseModel):
    """Detected hardware profile of the current machine.

    Populated by :func:`llama_autotune.hardware.detect_hardware`.
    """

    cpu_name: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    performance_cores: int = 0
    efficiency_cores: int = 0
    performance_threads: int = 0
    numa_nodes: int = 1
    ram_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpu_count: int = 0
    gpu_vendor: GpuVendor = GpuVendor.UNKNOWN
    gpu_models: list[str] = []
    vram_per_gpu: list[float] = []
    backend: Backend = Backend.CPU
    cpu_variant: str = ""
    isa_tier: str = ""


class KVLayout(BaseModel):
    """Per-token KV-cache geometry, summed over the layers that own a cache.

    Element counts are per token of context; multiply by the bytes per
    element of the chosen cache type. Sliding-window layers only hold
    ``swa_window`` (+ one ubatch) cells regardless of context size.
    """

    k_elems_full: int = 0
    v_elems_full: int = 0
    k_elems_swa: int = 0
    v_elems_swa: int = 0
    swa_window: int = 0
    recurrent_bytes: int = 0


class ModelInfo(BaseModel):
    """Metadata extracted from a GGUF model file.

    Populated by :func:`llama_autotune.model_inspector.inspect_model`.
    """

    path: str = ""
    architecture: str = ""
    parameters: int = 0
    quantization: str = ""
    quantization_declared: str = ""
    n_layers: int = 0
    n_heads: int = 0
    training_context: int = 0
    is_moe: bool = False
    active_parameters: int = 0
    file_size_gb: float = 0.0
    n_embd: int = 0
    vocab_size: int = 0
    tensor_bytes: int = 0
    bytes_per_token: int = 0
    tensor_types: dict[str, int] = {}
    kv_layout: KVLayout | None = None
    fingerprint: str = ""


class BenchmarkResult(BaseModel):
    """Result of a single llama-bench run."""

    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    startup_time: float = 0.0
    memory_usage: float = 0.0
    min_available_mb: float = 0.0
    success: bool = False
    error: str = ""
    raw_output: str = ""


class LoadMode(str, Enum):
    """Model loading modes of llama.cpp's ``--load-mode`` option."""

    AUTO = "auto"
    NONE = "none"
    MMAP = "mmap"
    MLOCK = "mlock"
    MMAP_MLOCK = "mmap+mlock"
    DIRECT_IO = "dio"


class SearchConfig(BaseModel):
    """A typed set of llama.cpp settings to benchmark or launch with.

    Every field is optional (None = leave llama.cpp's default). The
    config is an intent, not a command line: how each setting is spelled
    depends on the installed llama.cpp build, so argument lists are built
    by :mod:`llama_autotune.compat`. :meth:`to_bench_args` and
    :meth:`to_llama_args` render with the current llama.cpp spelling.
    """

    threads: int | None = None
    threads_batch: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    ctx_size: int | None = None
    n_gpu_layers: int | None = None
    flash_attn: bool | None = None
    tensor_split: str | None = None
    split_mode: SplitMode | None = None
    main_gpu: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    load_mode: LoadMode | None = None
    repack: bool | None = None
    numa: str | None = None
    cpu_affinity: str | None = None
    parallel: int | None = None
    no_kv_offload: bool | None = None
    n_cpu_moe: int | None = None
    cache_ram_mib: int | None = None
    ctx_checkpoints: int | None = None
    no_mmproj: bool | None = None
    spec_type: str | None = None
    priority: int | None = None  # --prio: 0 normal, 1 above normal, 2 high

    def to_bench_args(self) -> list[str]:
        """Render as ``llama-bench`` arguments using current llama.cpp flags.

        Returns:
            List of CLI arguments.
        """
        from .compat import render_args

        return render_args(self, "bench").args

    def to_llama_args(self) -> list[str]:
        """Render as ``llama-server`` arguments using current llama.cpp flags.

        Returns:
            List of CLI arguments.
        """
        from .compat import render_args

        return render_args(self, "server").args


class BenchmarkEntry(BaseModel):
    """A single benchmark record stored in the SQLite database."""

    hardware_id: str = ""
    model_id: str = ""
    config: SearchConfig = Field(default_factory=SearchConfig)
    result: BenchmarkResult = Field(default_factory=BenchmarkResult)
    objective: OptimizeObjective = OptimizeObjective.BALANCED
    timestamp: str = ""
    llama_build: int = 0
    model_path: str = ""


class LaunchProfile(BaseModel):
    """A reusable launch profile exported to or imported from JSON.

    ``llama_build`` records the llama.cpp build the profile was tuned on,
    so a profile can be retested after llama.cpp is upgraded.
    """

    name: str = ""
    args: list[str] = []
    model_path: str = ""
    hardware: str = ""
    created: str = ""
    score: float = 0.0
    llama_build: int = 0
