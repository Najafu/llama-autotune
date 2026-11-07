"""Memory estimation for llama.cpp on CPU, from the GGUF file alone.

The estimate is built from parts measured against llama.cpp b11347:

* resident weights: the bytes read per token (embedding lookups excluded),
* the private copy made by weight repacking, which llama.cpp's own
  ``--fit`` estimate leaves out on CPU (up to 3.2 GB for an 8B model),
* the KV cache from the model's metadata (matched llama.cpp exactly on
  every tested model), plus fixed-size recurrent state,
* compute buffers, from ``llama-fit-params`` when available, otherwise a
  conservative estimate dominated by the logits buffer.
"""

from __future__ import annotations

import functools
import re
import subprocess
from dataclasses import dataclass

from .models import KVLayout, ModelInfo

MIB = 1024 * 1024

# Bytes per cached element for each --cache-type-k/v value.
KV_BYTES_PER_ELEMENT: dict[str, float] = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32,
    "iq4_nl": 18 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32,
}

# Tensor types that llama.cpp repacks into a private interleaved copy,
# by ISA tier (ggml-cpu/repack.cpp, ggml_repack_get_optimal_repack_type).
REPACKABLE_TYPES: dict[str, set[str]] = {
    "avx2": {"Q4_0", "Q4_K", "IQ4_NL", "MXFP4"},
    "avx512": {"Q4_0", "Q4_K", "IQ4_NL", "MXFP4", "Q2_K"},
    "arm": {"Q4_0", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "MXFP4", "Q8_0", "Q1_0"},
}

_NOT_REPACKED = ("token_embd.weight", "per_layer_token_embd.weight")


@dataclass
class MemoryEstimate:
    """Projected host memory use in bytes."""

    weights: int = 0
    repack_copy: int = 0
    kv_cache: int = 0
    recurrent: int = 0
    compute: int = 0

    @property
    def total(self) -> int:
        return self.weights + self.repack_copy + self.kv_cache + self.recurrent + self.compute

    @property
    def total_mib(self) -> int:
        return round(self.total / MIB)


def kv_cache_bytes(layout: KVLayout, ctx: int, type_k: str = "f16", type_v: str = "f16", ubatch: int = 512) -> int:
    """Size of the KV cache for a context length and cache types.

    Sliding-window layers hold ``window + ubatch`` cells, rounded up to
    256, never more than the context (llama.cpp's SWA cache sizing).
    """
    bk = KV_BYTES_PER_ELEMENT.get(type_k, 2.0)
    bv = KV_BYTES_PER_ELEMENT.get(type_v, 2.0)
    full = ctx * (layout.k_elems_full * bk + layout.v_elems_full * bv)
    swa = 0.0
    if layout.swa_window:
        cells = min(ctx, -(-(layout.swa_window + ubatch) // 256) * 256)
        swa = cells * (layout.k_elems_swa * bk + layout.v_elems_swa * bv)
    return int(full + swa)


def repack_copy_bytes(model: ModelInfo, isa_tier: str, tensor_sizes: dict[str, tuple[str, int]] | None = None) -> int:
    """Bytes of the private copy llama.cpp makes when repacking weights.

    Args:
        model: Inspected model; ``tensor_types`` gives bytes per type.
        isa_tier: ``"avx2"``, ``"avx512"`` or ``"arm"``. ``"avx"`` and
            ``"no-avx"`` have no repack kernels. Any other value (a
            self-built ``native`` library) is counted like AVX2, because
            overestimating memory is safe and underestimating is not.
        tensor_sizes: Optional ``name -> (type, bytes)`` map; when given,
            embedding tables (which are never repacked) are excluded.

    Returns:
        Estimated bytes; 0 when the ISA tier has no repack kernels.
    """
    if isa_tier in ("avx", "no-avx"):
        return 0
    types = REPACKABLE_TYPES.get(isa_tier, REPACKABLE_TYPES["avx2"])
    if tensor_sizes is not None:
        return sum(n for name, (t, n) in tensor_sizes.items() if t in types and name not in _NOT_REPACKED)
    return sum(n for t, n in model.tensor_types.items() if t in types)


def compute_buffer_bytes(model: ModelInfo, ubatch: int = 512) -> int:
    """Conservative compute-buffer estimate when llama.cpp cannot be asked.

    The largest part is the logits buffer (vocabulary x ubatch x 4 bytes);
    the margin covers activations. On the tested models this stayed above
    llama.cpp's own figure by 6 to 35%.
    """
    vocab = model.vocab_size or 32_000
    return int(vocab * ubatch * 4 * 1.15 + ubatch * max(model.n_embd, 1024) * 4 * 8 + 32 * MIB)


def estimate_memory(
    model: ModelInfo,
    ctx: int,
    *,
    type_k: str = "f16",
    type_v: str = "f16",
    ubatch: int = 512,
    repack: bool = True,
    isa_tier: str = "avx2",
    tensor_sizes: dict[str, tuple[str, int]] | None = None,
    compute_override: int | None = None,
) -> MemoryEstimate:
    """Project host memory for running a model on the CPU.

    Args:
        model: Inspected model.
        ctx: Context size in tokens.
        type_k: KV cache type for K.
        type_v: KV cache type for V.
        ubatch: Physical batch size.
        repack: Whether weight repacking is enabled (llama.cpp default).
        isa_tier: ISA tier of the CPU variant llama.cpp loads.
        tensor_sizes: Optional per-tensor ``(type, bytes)`` for exact
            repack accounting.
        compute_override: Compute-buffer bytes reported by llama.cpp.

    Returns:
        A :class:`MemoryEstimate`.
    """
    weights = model.bytes_per_token or model.tensor_bytes or int(model.file_size_gb * 1024**3)
    layout = model.kv_layout
    if layout is None or not (layout.k_elems_full or layout.k_elems_swa or layout.recurrent_bytes):
        heads = model.n_heads or 32
        head_dim = (model.n_embd // heads) if model.n_embd else 128
        layout = KVLayout(k_elems_full=(model.n_layers or 32) * heads * head_dim,
                          v_elems_full=(model.n_layers or 32) * heads * head_dim)
    return MemoryEstimate(
        weights=weights,
        repack_copy=repack_copy_bytes(model, isa_tier, tensor_sizes) if repack else 0,
        kv_cache=kv_cache_bytes(layout, ctx, type_k, type_v, ubatch),
        recurrent=layout.recurrent_bytes,
        compute=compute_override if compute_override is not None else compute_buffer_bytes(model, ubatch),
    )


def tensor_sizes_of(path: str) -> dict[str, tuple[str, int]]:
    """Map tensor name to ``(type, bytes)`` for exact repack accounting."""
    from .model_inspector import read_gguf

    return {t.name: (t.type_name, t.nbytes) for t in read_gguf(path).tensors}


_FIT_PRINT = re.compile(r"^\s*Host\s+(\d+)\s+(\d+)\s+(\d+)", re.MULTILINE)


def compute_buffer_for(model_path: str, ctx: int) -> int | None:
    """Compute-buffer bytes at ubatch 512 from the installed ``llama-fit-params``.

    llama.cpp is asked once per model file, at 4,096 tokens. Beyond that the
    attention mask grows by ``ubatch x 2`` bytes per context token, which
    matched llama.cpp's own figures for Qwen3.5 4B up to 262,144 tokens; on
    other models the buffer grows less, so the estimate stays on the safe
    side.

    Returns:
        Bytes, or None when the tool is not installed or fails, so callers
        fall back to :func:`compute_buffer_bytes`.
    """
    import os

    st = os.stat(model_path)
    base = _compute_at_4096(os.path.abspath(model_path), st.st_size, st.st_mtime_ns)
    if base is None:
        return None
    return base + max(0, ctx - 4096) * 512 * 2


@functools.lru_cache(maxsize=16)
def _compute_at_4096(model_path: str, size: int, mtime_ns: int) -> int | None:
    import os

    from .compat import find_llama_binary

    exe = find_llama_binary("llama-fit-params")
    if not os.path.isfile(exe):
        return None
    return llama_compute_buffer(exe, model_path, 4096)


def llama_compute_buffer(fit_params_exe: str, model_path: str, ctx: int, ubatch: int = 512, timeout: int = 120) -> int | None:
    """Ask ``llama-fit-params --fit-print`` for the host compute buffer.

    Only the compute column is used: on CPU the model column omits
    repacked weights, so the tool computes weights itself.

    Returns:
        Compute-buffer bytes, or None if the tool is missing or fails.
    """
    from .compat import _NO_WINDOW, clean_env

    try:
        proc = subprocess.run(
            [fit_params_exe, "-m", model_path, "-c", str(ctx), "-ub", str(ubatch),
             "--no-repack", "-fitp", "on"],
            capture_output=True, text=True, timeout=timeout, errors="replace",
            creationflags=_NO_WINDOW, env=clean_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = _FIT_PRINT.search(proc.stdout or "")
    return int(m.group(3)) * MIB if m else None
