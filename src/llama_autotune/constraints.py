"""Memory gate and configuration checks.

The gate plans against the RAM that is *available now* minus a safety
margin, never against total RAM: llama.cpp's own ``--fit`` compares with
total host memory, which let an 8 GB test PC fall to 6 MiB free with
default settings. When a configuration does not fit, savings are applied
in order of least harm before the gate gives up and says by how much.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .memory import MIB, MemoryEstimate, estimate_memory
from .models import Backend, HardwareInfo, ModelInfo, SearchConfig

GIB = 1024**3


def safety_margin(total_bytes: int) -> int:
    """Headroom kept free on top of what is already in use.

    Available RAM already excludes what Windows and open programs use, so
    the margin only covers growth: the larger of 512 MiB and 10% of total.
    """
    return max(512 * MIB, total_bytes // 10)


def memory_budget(available_bytes: int, total_bytes: int) -> int:
    """Bytes a model may use: available RAM minus the safety margin."""
    return max(0, available_bytes - safety_margin(total_bytes))


@dataclass
class GateDecision:
    """Outcome of fitting a model into the memory budget."""

    fits: bool
    ctx: int
    estimate: MemoryEstimate
    budget: int
    ubatch: int = 512
    repack: bool = True
    cache_type: str = "f16"
    savings: list[str] = field(default_factory=list)

    @property
    def shortfall(self) -> int:
        """Bytes missing when the model does not fit (0 when it fits)."""
        return max(0, self.estimate.total - self.budget)

    def apply(self, config: SearchConfig) -> SearchConfig:
        """Return a copy of ``config`` with the gate's choices applied."""
        cfg = config.model_copy()
        cfg.ctx_size = self.ctx
        cfg.ubatch_size = self.ubatch
        if cfg.batch_size is not None and cfg.batch_size < self.ubatch:
            cfg.batch_size = self.ubatch
        if not self.repack:
            cfg.repack = False
        if self.cache_type != "f16":
            cfg.cache_type_k = cfg.cache_type_v = self.cache_type
            cfg.flash_attn = True
        return cfg


def plan_memory(
    model: ModelInfo,
    budget: int,
    *,
    isa_tier: str = "avx2",
    ctx: int = 4096,
    min_ctx: int = 2048,
    ubatch: int = 512,
    allow_kv_quant: bool = True,
    tensor_sizes: dict[str, tuple[str, int]] | None = None,
    compute_override: int | None = None,
) -> GateDecision:
    """Find the least harmful settings that fit a model into ``budget``.

    Savings are tried in this order, stopping at the first that fits:

    1. smaller ubatch (256, then 128): no speed cost on CPU in testing,
    2. weight repacking off: saves the private copy, costs some speed,
    3. q8_0 KV cache: halves the cache, needs flash attention,
    4. shorter context, halving down to ``min_ctx``.

    Args:
        model: Inspected model.
        budget: Bytes the model may use (see :func:`memory_budget`).
        isa_tier: ISA tier of the CPU variant llama.cpp loads.
        ctx: Wanted context size.
        min_ctx: Smallest context worth offering.
        ubatch: Starting ubatch; only smaller values are tried as savings.
        allow_kv_quant: Permit a q8_0 KV cache (avoid for tool calling).
        tensor_sizes: Optional per-tensor sizes for exact repack accounting.
        compute_override: Compute-buffer bytes reported by llama.cpp at
            ubatch 512.

    Returns:
        The first fitting :class:`GateDecision`, or the smallest one with
        ``fits=False`` when nothing fits.
    """
    state = {"ctx": ctx, "ubatch": ubatch, "repack": True, "cache_type": "f16"}
    savings: list[str] = []

    def check() -> GateDecision:
        est = estimate_config_memory(model, state["ctx"], state["cache_type"], state["ubatch"],
                                     state["repack"], isa_tier, tensor_sizes, compute_override)
        return GateDecision(est.total <= budget, state["ctx"], est, budget,
                            state["ubatch"], state["repack"], state["cache_type"], list(savings))

    decision = check()
    steps = [("ubatch", u) for u in (256, 128) if u < ubatch] + [("repack", False)]
    if allow_kv_quant:
        steps.append(("cache_type", "q8_0"))
    for key, value in steps:
        if decision.fits:
            return decision
        before = decision.estimate.total
        state[key] = value
        decision = check()
        if decision.estimate.total < before:
            savings.append(_describe(key, value, before - decision.estimate.total))
            decision.savings = list(savings)
    while not decision.fits and state["ctx"] // 2 >= min_ctx:
        before = decision.estimate.total
        state["ctx"] //= 2
        decision = check()
        savings.append(f"context {state['ctx']} tokens (saves {round((before - decision.estimate.total) / MIB)} MiB)")
        decision.savings = list(savings)
    return decision


def estimate_config_memory(model: ModelInfo, ctx: int, cache_type: str, ubatch: int, repack: bool,
                           isa_tier: str, tensor_sizes: dict[str, tuple[str, int]] | None,
                           compute_override: int | None) -> MemoryEstimate:
    """The single memory estimate shared by the gate and the optimizer.

    ``compute_override`` is llama.cpp's compute buffer at ubatch 512 and is
    scaled to the actual ubatch.
    """
    return estimate_memory(
        model, ctx, type_k=cache_type, type_v=cache_type, ubatch=ubatch, repack=repack,
        isa_tier=isa_tier, tensor_sizes=tensor_sizes,
        compute_override=None if compute_override is None else compute_override * ubatch // 512,
    )


def _describe(key: str, value: object, saved: int) -> str:
    mib = round(saved / MIB)
    if key == "ubatch":
        return f"ubatch {value} (saves {mib} MiB)"
    if key == "repack":
        return f"weight repacking off (saves {mib} MiB)"
    return f"{value} KV cache with flash attention (saves {mib} MiB)"


# ── checks used by the optimizer ─────────────────────────────────────


def estimate_vram(config: SearchConfig, model: ModelInfo, hw: HardwareInfo) -> float:
    """Estimate the memory (GB) a configuration needs.

    Args:
        config: The configuration to evaluate.
        model: Model metadata.
        hw: Hardware information (ISA tier for repack accounting).

    Returns:
        Estimated memory in gigabytes.
    """
    return _estimate(config, model, hw).total / GIB


def _estimate(config: SearchConfig, model: ModelInfo, hw: HardwareInfo) -> MemoryEstimate:
    return estimate_memory(
        model, config.ctx_size or 4096,
        type_k=config.cache_type_k or "f16", type_v=config.cache_type_v or "f16",
        ubatch=config.ubatch_size or 512, repack=config.repack is not False,
        isa_tier=hw.isa_tier or "avx2",
    )


def _available_memory(hw: HardwareInfo) -> float:
    """Usable memory in GB: VRAM for GPU backends, else the RAM budget."""
    if hw.backend != Backend.CPU and hw.gpu_count > 0 and hw.vram_per_gpu:
        return sum(hw.vram_per_gpu) * 0.9
    total = int(hw.ram_gb * GIB)
    available = int(hw.ram_available_gb * GIB) if hw.ram_available_gb else int(total * 0.8)
    return memory_budget(available, total) / GIB


def is_oom(config: SearchConfig, model: ModelInfo, hw: HardwareInfo) -> bool:
    """Return True if a configuration would not fit in usable memory."""
    return estimate_vram(config, model, hw) > _available_memory(hw)


def is_plausible(config: SearchConfig, model: ModelInfo, hw: HardwareInfo, *,
                 budget: int | None = None, tensor_sizes: dict[str, tuple[str, int]] | None = None,
                 compute_override: int | None = None) -> bool:
    """Determine whether a configuration is worth benchmarking.

    Rejects configurations that do not fit in memory, request more threads
    than logical cores, use an ubatch larger than the batch, or quantize
    the V cache without flash attention. When ``budget`` is given, memory
    is checked with exactly the gate's estimate, so the optimizer never
    rejects a configuration the gate accepted.
    """
    if budget is not None:
        cache = config.cache_type_v or "f16"
        est = estimate_config_memory(model, config.ctx_size or 4096, cache, config.ubatch_size or 512,
                                     config.repack is not False, hw.isa_tier or "avx2",
                                     tensor_sizes, compute_override)
        if est.total > budget:
            return False
    elif is_oom(config, model, hw):
        return False
    if config.threads is not None and config.threads > hw.logical_cores:
        return False
    if config.batch_size and config.ubatch_size and config.ubatch_size > config.batch_size:
        return False
    if config.cache_type_v not in (None, "f16", "f32", "bf16") and config.flash_attn is False:
        return False
    return True


_OOM_PATTERNS = re.compile(
    r"out of memory|\boom\b|cudamalloc failed|cannot allocate|failed to allocate|"
    r"bad_alloc|unable to allocate|not enough memory",
    re.IGNORECASE,
)


def detect_oom_in_output(output: str) -> bool:
    """Return True if process output reports running out of memory."""
    return bool(output) and bool(_OOM_PATTERNS.search(output))
