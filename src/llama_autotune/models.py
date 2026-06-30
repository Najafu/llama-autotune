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
    ram_gb: float = 0.0
    gpu_count: int = 0
    gpu_vendor: GpuVendor = GpuVendor.UNKNOWN
    gpu_models: list[str] = []
    vram_per_gpu: list[float] = []
    backend: Backend = Backend.CPU


class ModelInfo(BaseModel):
    """Metadata extracted from a GGUF model file.

    Populated by :func:`llama_autotune.model_inspector.inspect_model`.
    """

    path: str = ""
    architecture: str = ""
    parameters: int = 0
    quantization: str = ""
    n_layers: int = 0
    n_heads: int = 0
    training_context: int = 0
    is_moe: bool = False
    active_parameters: int = 0
    file_size_gb: float = 0.0


class BenchmarkResult(BaseModel):
    """Result of a single llama-bench run."""

    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    startup_time: float = 0.0
    memory_usage: float = 0.0
    success: bool = False
    raw_output: str = ""


class SearchConfig(BaseModel):
    """A full set of llama.cpp parameters to benchmark or launch with.

    Every field is optional (None = use llama.cpp's default). Use
    :meth:`to_bench_args` for llama-bench and :meth:`to_llama_args`
    for llama-server.
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
    mmap: bool | None = None
    mlock: bool | None = None
    numa: str | None = None
    cpu_affinity: str | None = None
    parallel: int | None = None
    no_kv_offload: bool | None = None

    def to_bench_args(self) -> list[str]:
        """Convert to command-line arguments for ``llama-bench``.

        Only includes flags that ``llama-bench`` actually accepts
        (omits context size, numa, mlock, etc.).

        Returns:
            List of CLI arguments.
        """
        args = []
        if self.threads is not None:
            args.extend(["-t", str(self.threads)])
        if self.threads_batch is not None:
            args.extend(["-tb", str(self.threads_batch)])
        if self.batch_size is not None:
            args.extend(["-b", str(self.batch_size)])
        if self.ubatch_size is not None:
            args.extend(["-ub", str(self.ubatch_size)])
        if self.n_gpu_layers is not None:
            args.extend(["-ngl", str(self.n_gpu_layers)])
        if self.flash_attn is not None:
            args.extend(["-fa", "1" if self.flash_attn else "0"])
        if self.tensor_split is not None:
            args.extend(["-ts", self.tensor_split])
        if self.split_mode is not None:
            args.extend(["-sm", self.split_mode.value])
        if self.main_gpu is not None:
            args.extend(["-mg", str(self.main_gpu)])
        if self.cache_type_k is not None:
            args.extend(["-ctk", self.cache_type_k])
        if self.cache_type_v is not None:
            args.extend(["-ctv", self.cache_type_v])
        if self.mmap is not None:
            args.extend(["-mmp", "1" if self.mmap else "0"])
        return args

    def to_llama_args(self) -> list[str]:
        """Convert to command-line arguments for ``llama-server``.

        Extends :meth:`to_bench_args` with server-specific flags
        like context size, numa, and parallel.

        Returns:
            List of CLI arguments.
        """
        args = self.to_bench_args()
        if self.ctx_size is not None:
            args.extend(["-c", str(self.ctx_size)])
        if self.mlock is not None:
            args.extend(["--mlock"])
        if self.numa is not None:
            args.extend(["--numa", self.numa])
        if self.parallel is not None:
            args.extend(["--parallel", str(self.parallel)])
        if self.no_kv_offload is not None:
            args.extend(["-nkvo", "1" if self.no_kv_offload else "0"])
        return args


class BenchmarkEntry(BaseModel):
    """A single benchmark record stored in the SQLite database."""

    hardware_id: str = ""
    model_id: str = ""
    config: SearchConfig = Field(default_factory=SearchConfig)
    result: BenchmarkResult = Field(default_factory=BenchmarkResult)
    objective: OptimizeObjective = OptimizeObjective.BALANCED
    timestamp: str = ""


class LaunchProfile(BaseModel):
    """A reusable launch profile exported to or imported from JSON."""

    name: str = ""
    args: list[str] = []
    model_path: str = ""
    hardware: str = ""
    created: str = ""
    score: float = 0.0
