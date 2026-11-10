"""Inspect GGUF model files without loading their weights.

Reads the header key-value store and the tensor table, then derives what
the tuner needs: architecture and size, the real quantization (from the
file type and the tensor types, never only the filename), the bytes the
model reads per generated token, and the per-token KV-cache geometry.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .models import KVLayout, ModelInfo

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR_FORMATS = {
    GGUF_TYPE_UINT8: "<B", GGUF_TYPE_INT8: "<b", GGUF_TYPE_UINT16: "<H",
    GGUF_TYPE_INT16: "<h", GGUF_TYPE_UINT32: "<I", GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f", GGUF_TYPE_BOOL: "<?", GGUF_TYPE_UINT64: "<Q",
    GGUF_TYPE_INT64: "<q", GGUF_TYPE_FLOAT64: "<d",
}

class GGUFError(ValueError):
    """The file starts like a GGUF but cannot be read: cut short or damaged."""


# No real string (chat template, token) or tensor comes near these; a damaged
# header can claim anything, and the claim must not be allocated.
_MAX_STRING_BYTES = 64 * 1024 * 1024
_MAX_DIMS = 8

# Arrays longer than this are skipped and only their length is kept
# (tokenizer vocabularies); per-layer arrays are far shorter.
_MAX_KEPT_ARRAY = 4096

# llama_ftype values written to general.file_type (llama.h).
_KNOWN_FILE_TYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE",
}

# ggml_type values used for individual tensors (ggml.h).
GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0", 43: "F8_E4M3",
}

_QUANT_PATTERNS = [
    "IQ4_XS", "IQ4_NL", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ1_S", "IQ1_M",
    "Q4_K_M", "Q4_K_S", "Q4_K_P", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_K",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "MXFP4",
    "Q8_0", "Q8_1", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "BF16", "F16", "F32",
]

# Tensors read one row at a time (embedding lookups), not in full per token.
_PER_LAYER_LOOKUP = "per_layer_token_embd.weight"
_TOKEN_EMBD = "token_embd.weight"
_OUTPUT = "output.weight"


@dataclass
class GGUFArray:
    """A long GGUF array that was skipped; only its length is kept."""

    elem_type: int
    length: int


@dataclass
class TensorInfo:
    """One entry of the GGUF tensor table."""

    name: str
    dims: list[int]
    ggml_type: int
    offset: int
    nbytes: int = 0

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAMES.get(self.ggml_type, f"type{self.ggml_type}")


@dataclass
class GGUFFile:
    """Header metadata and tensor table of a GGUF file."""

    path: str
    kv: dict[str, object] = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)
    file_size: int = 0


def read_gguf(path: str | Path) -> GGUFFile:
    """Read the metadata and tensor table of a GGUF file.

    Tensor byte sizes are derived from consecutive data offsets, so no
    per-type block-size table is needed and new quant types work too.
    Results are cached per file path, size and modification time; treat
    the returned object as read-only.

    Args:
        path: Path to the GGUF file.

    Returns:
        A :class:`GGUFFile`. ``kv`` and ``tensors`` are empty when the
        file does not start with the GGUF magic number.

    Raises:
        GGUFError: The file is a GGUF but is cut short or damaged (an
            unfinished download, usually).
    """
    st = os.stat(path)
    return _read_gguf_cached(os.path.abspath(str(path)), st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=32)
def _read_gguf_cached(path: str, size: int, mtime_ns: int) -> GGUFFile:
    try:
        return _read_gguf_file(path, size)
    except (struct.error, ValueError, EOFError, OverflowError, MemoryError) as e:
        if isinstance(e, GGUFError):
            raise
        raise GGUFError(f"{os.path.basename(path)} is not a complete GGUF file "
                        f"(damaged, or a download that did not finish)") from e


def _read_gguf_file(path: str, size: int) -> GGUFFile:
    gguf = GGUFFile(path=path, file_size=size)
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return gguf
        struct.unpack("<I", f.read(4))
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            key = _read_string(f)
            gguf.kv[key] = _read_value(f)
        for _ in range(n_tensors):
            name = _read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            if n_dims > _MAX_DIMS:
                raise GGUFError(f"tensor with {n_dims} dimensions")
            dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
            ggml_type, offset = struct.unpack("<IQ", f.read(12))
            gguf.tensors.append(TensorInfo(name, dims, ggml_type, offset))
        align = int(gguf.kv.get("general.alignment", 32) or 32)
        data_start = (f.tell() + align - 1) // align * align

    ordered = sorted(gguf.tensors, key=lambda t: t.offset)
    for i, t in enumerate(ordered):
        end = ordered[i + 1].offset if i + 1 < len(ordered) else size - data_start
        t.nbytes = max(0, end - t.offset)
    return gguf


def _read_gguf_header(path: str) -> dict[str, object]:
    """Read only the key-value store of a GGUF file (empty if not GGUF)."""
    return read_gguf(path).kv


def _read_string(f) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    if length > _MAX_STRING_BYTES:
        raise GGUFError(f"string of {length} bytes")
    data = f.read(length)
    if len(data) != length:
        raise GGUFError("file ends inside a string")
    return data.decode("utf-8", errors="replace")


def _read_value(f) -> object:
    val_type = struct.unpack("<I", f.read(4))[0]
    return _read_value_with_type(f, val_type)


def _read_value_with_type(f, val_type: int) -> object:
    if val_type in _SCALAR_FORMATS:
        fmt = _SCALAR_FORMATS[val_type]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if val_type == GGUF_TYPE_STRING:
        return _read_string(f)
    if val_type == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        if count > _MAX_KEPT_ARRAY:
            if elem_type in _SCALAR_FORMATS:
                f.seek(count * struct.calcsize(_SCALAR_FORMATS[elem_type]), 1)
            else:
                for _ in range(count):
                    _read_value_with_type(f, elem_type)
            return GGUFArray(elem_type, count)
        return [_read_value_with_type(f, elem_type) for _ in range(count)]
    raise ValueError(f"unknown GGUF value type {val_type}")


# ── metadata helpers ──────────────────────────────────────────────────


def _get_str(kv: dict, key: str) -> str:
    val = kv.get(key)
    if val is None or isinstance(val, (list, GGUFArray)):
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _get_int(kv: dict, key: str, default: int = 0) -> int:
    val = kv.get(key)
    if val is None or isinstance(val, (list, GGUFArray)):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _per_layer(kv: dict, key: str, n_layer: int, default: int) -> list[int]:
    """Return a per-layer integer list from a scalar or array metadata key."""
    val = kv.get(key)
    if isinstance(val, list) and val:
        vals = [int(v) for v in val]
        return (vals + [vals[-1]] * n_layer)[:n_layer]
    if val is None or isinstance(val, GGUFArray):
        return [default] * n_layer
    try:
        return [int(val)] * n_layer
    except (ValueError, TypeError):
        return [default] * n_layer


def _parse_size_label(label: str) -> list[int]:
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    out = []
    for part in label.split("-"):
        part = part.strip()
        m = re.fullmatch(r"(?:\d+x)?([\d.]+)([KMBT])", part)
        if m:
            out.append(int(float(m.group(1)) * multipliers[m.group(2)]))
    return out


def _resolve_param_count(kv: dict) -> int:
    """Resolve the parameter count from GGUF metadata.

    Uses ``general.parameter_count``, else the total from a
    ``general.size_label`` such as ``"7B"`` or ``"1B-7B"`` (active-total).

    Returns:
        The parameter count, or ``0`` if the metadata does not say.
    """
    count = _get_int(kv, "general.parameter_count", default=0)
    if count > 0:
        return count
    sizes = _parse_size_label(_get_str(kv, "general.size_label"))
    return sizes[-1] if sizes else 0


def _resolve_active_param_count(kv: dict, total_params: int, expert_count: int, expert_used: int) -> int:
    """Resolve the active parameter count of an MoE model from metadata.

    Prefers an ``"active-total"`` size label (``"1B-7B"`` gives 1B), else
    scales the total by the share of experts used per token.
    """
    label = _get_str(kv, "general.size_label")
    if "-" in label:
        sizes = _parse_size_label(label)
        if len(sizes) == 2:
            return sizes[0]
    if total_params > 0 and expert_count > 0:
        return total_params * expert_used // expert_count
    return 0


def _resolve_file_type(kv: dict, model_path: str = "", tensors: list[TensorInfo] | None = None) -> str:
    """Resolve the quantization label of a model.

    The tensor types are the ground truth: converted files can carry a
    stale ``general.file_type`` (a local Q6_K file declares Q4_K_S), and
    filenames are free text. The more specific metadata label is used
    only when it agrees with the most common transformer-weight type;
    otherwise that tensor type wins. Without tensors, the metadata, the
    filename and ``general.name`` are tried in that order.

    Returns:
        A label such as ``"Q4_K_M"``, or ``"unknown"``.
    """
    raw = _get_int(kv, "general.file_type", default=-1)
    declared = _KNOWN_FILE_TYPES.get(raw, "")
    dominant = _dominant_weight_type(tensors) if tensors else ""
    if dominant:
        return declared if declared.startswith(dominant) else dominant
    if declared:
        return declared
    sources = []
    if model_path:
        sources.append(os.path.splitext(os.path.basename(str(model_path)))[0])
    sources.append(_get_str(kv, "general.name"))
    for source in sources:
        upper = source.upper()
        for pat in _QUANT_PATTERNS:
            if pat in upper:
                return pat
    return str(raw) if raw >= 0 else "unknown"


def _dominant_weight_type(tensors: list[TensorInfo]) -> str:
    totals: dict[str, int] = {}
    for t in tensors:
        if t.name.startswith("blk.") and len(t.dims) >= 2:
            totals[t.type_name] = totals.get(t.type_name, 0) + t.nbytes
    return max(totals, key=totals.get) if totals else ""


def _resolve_block_count(kv: dict, arch: str) -> int:
    count = _get_int(kv, f"{arch}.block_count", default=0)
    if count == 0:
        count = _get_int(kv, "llama.block_count", default=0)
    return count


# ── derived quantities ───────────────────────────────────────────────


def bytes_read_per_token(gguf: GGUFFile) -> int:
    """Bytes of weights read to generate one token.

    Every tensor counts except embedding lookups, which are read one row
    at a time: the per-layer embedding table always, and the token
    embedding unless it doubles as the output matrix (tied embeddings).
    """
    names = {t.name for t in gguf.tensors}
    tied = _OUTPUT not in names
    total = 0
    for t in gguf.tensors:
        if t.name == _PER_LAYER_LOOKUP or (t.name == _TOKEN_EMBD and not tied):
            continue
        total += t.nbytes
    return total


def kv_layout(kv: dict, arch: str) -> KVLayout:
    """Per-token KV-cache geometry from model metadata.

    Handles grouped-query attention (``head_count_kv``, scalar or per
    layer), sliding-window layers (``sliding_window`` and its pattern),
    layers that share an earlier layer's cache (``shared_kv_layers``),
    hybrid models with periodic full attention
    (``full_attention_interval``), recurrent state (``ssm.*``) and
    compressed MLA caches (``kv_lora_rank``).
    """
    n_layer = _resolve_block_count(kv, arch)
    if n_layer <= 0:
        return KVLayout()
    a = f"{arch}.attention"
    n_embd = _get_int(kv, f"{arch}.embedding_length", 0)
    head = _per_layer(kv, f"{a}.head_count", n_layer, 0)
    head_kv = _per_layer(kv, f"{a}.head_count_kv", n_layer, head[0])
    default_dim = n_embd // head[0] if head[0] else 0
    k_len = _get_int(kv, f"{a}.key_length", default_dim)
    v_len = _get_int(kv, f"{a}.value_length", default_dim)
    k_len_swa = _get_int(kv, f"{a}.key_length_swa", k_len)
    v_len_swa = _get_int(kv, f"{a}.value_length_swa", v_len)

    lora = _get_int(kv, f"{a}.kv_lora_rank", 0)
    if lora:
        rope = _get_int(kv, f"{arch}.rope.dimension_count", 64)
        k_len, v_len, head_kv = lora + rope, lora, [1] * n_layer

    window = _get_int(kv, f"{a}.sliding_window", 0)
    pattern_raw = kv.get(f"{a}.sliding_window_pattern")
    if isinstance(pattern_raw, list):
        swa = [bool(x) for x in (pattern_raw + [False] * n_layer)[:n_layer]]
    elif isinstance(pattern_raw, int) and pattern_raw > 1 and window:
        swa = [(il % pattern_raw) < pattern_raw - 1 for il in range(n_layer)]
    else:
        swa = [False] * n_layer

    interval = _get_int(kv, f"{arch}.full_attention_interval", 0)
    has_attn = [h > 0 for h in head_kv]
    if interval > 0:
        has_attn = [(il + 1) % interval == 0 for il in range(n_layer)]

    n_own_kv = n_layer - _get_int(kv, f"{a}.shared_kv_layers", 0)

    layout = KVLayout(swa_window=window if any(swa) else 0)
    recurrent_layers = 0
    for il in range(n_layer):
        if not has_attn[il]:
            recurrent_layers += 1
            continue
        if il >= n_own_kv:
            continue
        if swa[il] and window:
            layout.k_elems_swa += head_kv[il] * k_len_swa
            layout.v_elems_swa += head_kv[il] * v_len_swa
        else:
            layout.k_elems_full += head_kv[il] * k_len
            layout.v_elems_full += head_kv[il] * v_len

    d_inner = _get_int(kv, f"{arch}.ssm.inner_size", 0)
    d_state = _get_int(kv, f"{arch}.ssm.state_size", 0)
    if recurrent_layers and d_inner and d_state:
        d_conv = _get_int(kv, f"{arch}.ssm.conv_kernel", 4)
        n_group = _get_int(kv, f"{arch}.ssm.group_count", 0)
        per_layer = 4 * (d_inner * d_state + (d_conv - 1) * (d_inner + 2 * n_group * d_state))
        layout.recurrent_bytes = recurrent_layers * per_layer
    l_cache = _get_int(kv, f"{arch}.shortconv.l_cache", 0)
    if recurrent_layers and l_cache > 1 and not layout.recurrent_bytes:
        # LFM2/LFM3 short-convolution layers keep (l_cache - 1) f32 columns of
        # n_embd each (0.61 MiB for LFM3 1.7B, as llama.cpp b12228 reports).
        layout.recurrent_bytes = recurrent_layers * 4 * (l_cache - 1) * n_embd
    return layout


def model_fingerprint(path: str | Path) -> str:
    """Stable id for a model file: hash of its size and first 4 MiB.

    Survives renames and moves, unlike a path, and is cheap to compute.
    """
    h = hashlib.sha256()
    h.update(str(os.path.getsize(path)).encode())
    with open(path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()[:16]


def _expert_elements(tensors: list[TensorInfo]) -> int:
    return sum(t.n_elements for t in tensors if "_exps" in t.name)


def inspect_model(path: str | Path) -> ModelInfo:
    """Inspect a GGUF model file and return its metadata.

    Args:
        path: Path to the GGUF model file.

    Returns:
        A populated :class:`ModelInfo`.

    Raises:
        FileNotFoundError: If the file does not exist.
        GGUFError: If the file is cut short or damaged.
    """
    path = str(path)
    gguf = read_gguf(path)
    kv = gguf.kv
    info = ModelInfo(path=path, file_size_gb=round(gguf.file_size / (1024**3), 2))
    arch = _get_str(kv, "general.architecture")
    info.architecture = arch
    info.quantization = _resolve_file_type(kv, path, gguf.tensors)
    info.quantization_declared = _KNOWN_FILE_TYPES.get(_get_int(kv, "general.file_type", -1), "")
    info.n_layers = _resolve_block_count(kv, arch)
    info.n_heads = _per_layer(kv, f"{arch}.attention.head_count", max(info.n_layers, 1), 0)[0]
    info.training_context = _get_int(kv, f"{arch}.context_length", default=0)
    info.n_embd = _get_int(kv, f"{arch}.embedding_length", default=0)

    tokens = kv.get("tokenizer.ggml.tokens")
    if isinstance(tokens, (GGUFArray, list)):
        info.vocab_size = tokens.length if isinstance(tokens, GGUFArray) else len(tokens)
    info.vocab_size = info.vocab_size or _get_int(kv, f"{arch}.vocab_size", 0)

    tensor_elems = sum(t.n_elements for t in gguf.tensors)
    info.parameters = _resolve_param_count(kv) or tensor_elems
    info.tensor_bytes = sum(t.nbytes for t in gguf.tensors)
    info.bytes_per_token = bytes_read_per_token(gguf)
    for t in gguf.tensors:
        info.tensor_types[t.type_name] = info.tensor_types.get(t.type_name, 0) + t.nbytes
    info.kv_layout = kv_layout(kv, arch) if arch else None
    info.fingerprint = model_fingerprint(path)

    expert_count = _get_int(kv, f"{arch}.expert_count", default=0)
    info.is_moe = expert_count > 1
    info.active_bytes_per_token = info.bytes_per_token
    if info.is_moe:
        used = _get_int(kv, f"{arch}.expert_used_count", default=0)
        expert_bytes = sum(t.nbytes for t in gguf.tensors if "_exps" in t.name)
        if expert_bytes and used:
            # One token reads the shared weights and only `used` of `expert_count` experts.
            info.active_bytes_per_token = (info.bytes_per_token - expert_bytes
                                           + expert_bytes * min(used, expert_count) // expert_count)
        expert_elems = _expert_elements(gguf.tensors)
        if expert_elems and tensor_elems:
            info.active_parameters = tensor_elems - expert_elems + expert_elems * used // expert_count
        else:
            info.active_parameters = _resolve_active_param_count(kv, info.parameters, expert_count, used)
    else:
        info.active_parameters = info.parameters
    return info
