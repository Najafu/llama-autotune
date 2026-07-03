"""Inspect GGUF model files and extract metadata.

Reads the header key-value store from GGUF-format model files and
resolves structured information such as architecture, quantization type,
parameter count, and layer count into a ModelInfo object.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

from .models import ModelInfo

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

_KNOWN_FILE_TYPES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_1_SOME_F16",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_K",
    10: "Q8_1",
    11: "Q2_K",
    12: "Q3_K",
    13: "Q4_K",
    14: "Q5_K",
    15: "Q6_K",
    16: "IQ1_S",
    17: "IQ1_M",
    18: "IQ2_XXS",
    19: "IQ2_XS",
    20: "IQ2_S",
    21: "IQ2_M",
    22: "IQ3_XXS",
    23: "IQ3_XS",
    24: "IQ3_S",
    25: "IQ3_M",
    26: "IQ4_XS",
    27: "IQ4_NL",
}

_QUANT_PATTERNS = [
    "IQ4_XS", "IQ4_NL", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ1_S", "IQ1_M",
    "Q4_K_M", "Q4_K_S", "Q4_K_P", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_K",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K",
    "Q8_0", "Q8_1", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "BF16", "F16", "F32",
]


def inspect_model(path: str | Path) -> ModelInfo:
    """Inspect a GGUF model file and return its metadata.

    Args:
        path: Path to the GGUF model file.

    Returns:
        A ModelInfo instance populated with file size, architecture,
        quantization, parameter count, layer count, head count,
        context length, and MoE details (if applicable).
    """
    path = str(path)
    info = ModelInfo()
    info.path = path
    info.file_size_gb = round(os.path.getsize(path) / (1024**3), 2)

    kv = _read_gguf_header(path)

    info.architecture = _get_str(kv, "general.architecture")
    info.quantization = _resolve_file_type(kv, path)
    info.parameters = _resolve_param_count(kv)
    info.n_layers = _resolve_block_count(kv, info.architecture)
    info.n_heads = _get_int(kv, f"{info.architecture}.attention.head_count", default=0)
    info.training_context = _get_int(kv, f"{info.architecture}.context_length", default=0)

    expert_count = _get_int(kv, f"{info.architecture}.expert_count", default=0)
    if expert_count == 0:
        for key, val in kv.items():
            if "expert_count" in key:
                try:
                    expert_count = int(val)
                except (ValueError, TypeError):
                    pass
                break
    info.is_moe = expert_count > 1

    if info.is_moe:
        active = _get_int(kv, f"{info.architecture}.expert_used_count", default=0)
        info.active_parameters = _resolve_active_param_count(kv, info.parameters, expert_count, active)
    else:
        info.active_parameters = info.parameters

    return info


def _resolve_file_type(kv: dict, model_path: str = "") -> str:
    """Resolve the quantization / file-type string from GGUF metadata.

    Prioritises (in order):
    1. ``general.name`` metadata field
    2. The model filename
    3. The integer ``general.file_type`` key mapped through
       ``_KNOWN_FILE_TYPES``
    4. The raw ``general.file_type`` value as a string

    Args:
        kv: The GGUF header key-value dictionary.
        model_path: The model file path, used for filename fallback.

    Returns:
        A string such as ``"Q4_0"``, ``"F16"``, or ``"unknown"``.
    """
    for source in (os.path.splitext(os.path.basename(str(model_path)))[0]
                   if model_path else "",
                   _get_str(kv, "general.name")):
        if not source:
            continue
        upper = source.upper()
        for pat in _QUANT_PATTERNS:
            if pat in upper:
                return pat
    raw = _get_int(kv, "general.file_type", default=-1)
    if raw in _KNOWN_FILE_TYPES:
        return _KNOWN_FILE_TYPES[raw]
    return str(raw) if raw >= 0 else "unknown"


def _resolve_param_count(kv: dict) -> int:
    """Resolve the parameter count from GGUF metadata.

    Reads ``general.parameter_count`` first.  If it is zero, attempts
    to parse a human-friendly label such as ``"7B"``, ``"70M"``, or
    ``"1B-7B"`` (active-total) from ``general.size_label``.

    For hyphenated labels like ``"1B-7B"`` the **total** (second)
    value is returned.

    Args:
        kv: The GGUF header key-value dictionary.

    Returns:
        The number of parameters as an integer, or ``0`` if unresolvable.
    """
    count = _get_int(kv, "general.parameter_count", default=0)
    if count > 0:
        return count
    label = _get_str(kv, "general.size_label")
    if label:
        multipliers = {"M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
        # Split on '-' to handle "1B-7B" format — try the last (total) part first
        parts = [p.strip() for p in label.split("-")]
        for part in reversed(parts):
            for suffix, mult in multipliers.items():
                if part.endswith(suffix):
                    try:
                        num = float(part[: -len(suffix)].strip())
                        return int(num * mult)
                    except ValueError:
                        pass
    return 0


def _resolve_active_param_count(kv: dict, total_params: int, expert_count: int, expert_used: int) -> int:
    """Resolve the active parameter count for an MoE model.

    Prefers ``general.size_label`` (e.g. ``"1B-7B"`` → 1B active),
    falling back to a proportional estimate from expert counts.

    Args:
        kv: The GGUF header key-value dictionary.
        total_params: The total parameter count.
        expert_count: Total number of experts.
        expert_used: Number of active experts per token.

    Returns:
        The estimated number of active parameters.
    """
    label = _get_str(kv, "general.size_label")
    if label and "-" in label:
        first = label.split("-", 1)[0].strip()
        multipliers = {"M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
        for suffix, mult in multipliers.items():
            if first.endswith(suffix):
                try:
                    return int(float(first[: -len(suffix)].strip()) * mult)
                except ValueError:
                    pass
    if total_params > 0 and expert_count > 0:
        return total_params * expert_used // expert_count
    return 0


def _resolve_block_count(kv: dict, arch: str) -> int:
    """Resolve the number of transformer blocks (layers) for an architecture.

    Checks ``{arch}.block_count`` first, with a fallback to the legacy
    ``llama.block_count`` key.

    Args:
        kv: The GGUF header key-value dictionary.
        arch: The model architecture name (e.g. ``"llama"``).

    Returns:
        The number of layers, or ``0`` if not found.
    """
    count = _get_int(kv, f"{arch}.block_count", default=0)
    if count == 0:
        count = _get_int(kv, "llama.block_count", default=0)
    return count


def _read_gguf_header(path: str) -> dict[str, object]:
    """Read the GGUF header key-value store from a file.

    Args:
        path: The filesystem path to the GGUF file.

    Returns:
        A dictionary mapping string keys to parsed values (strings,
        integers, floats, booleans, or lists thereof).  Returns an
        empty dict if the file does not start with a valid GGUF magic
        number.
    """
    kv: dict[str, object] = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return kv

        _ = struct.unpack("<I", f.read(4))[0]
        _ = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        for _ in range(kv_count):
            key = _read_string(f)
            val = _read_value(f)
            kv[key] = val

    return kv


def _read_string(f) -> str:
    """Read a GGUF string from a binary stream.

    A GGUF string is stored as an 8-byte little-endian length followed
    by that many UTF-8 bytes.

    Args:
        f: An open binary file handle positioned at the start of the
           string.

    Returns:
        The decoded string.
    """
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f) -> object:
    """Read a single GGUF key-value value from a binary stream.

    The first four bytes indicate the value type; the subsequent bytes
    are interpreted according to that type.

    Args:
        f: An open binary file handle positioned at the start of the
           value.

    Returns:
        The parsed Python object (``int``, ``float``, ``bool``,
        ``str``, ``list``, or ``None`` for unknown types).
    """
    val_type = struct.unpack("<I", f.read(4))[0]

    if val_type == GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif val_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif val_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif val_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif val_type == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif val_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif val_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif val_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack("<B", f.read(1))[0])
    elif val_type == GGUF_TYPE_STRING:
        return _read_string(f)
    elif val_type == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        arr = []
        for _ in range(count):
            arr.append(_read_value_with_type(f, elem_type))
        return arr
    elif val_type == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif val_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif val_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    else:
        return None


def _read_value_with_type(f, val_type: int) -> object:
    """Read a GGUF value whose type is already known.

    Unlike :func:`_read_value`, this function does **not** consume the
    type tag from the stream; the caller provides the type code.

    Args:
        f: An open binary file handle.
        val_type: One of the ``GGUF_TYPE_*`` constants.

    Returns:
        The parsed Python object, or ``None`` for unrecognised types.
    """
    if val_type == GGUF_TYPE_STRING:
        return _read_string(f)
    elif val_type == GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif val_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif val_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif val_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif val_type == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif val_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif val_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif val_type == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif val_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif val_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    elif val_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack("<B", f.read(1))[0])
    return None


def _get_str(kv: dict, key: str) -> str:
    """Safely retrieve a string value from the GGUF key-value store.

    Args:
        kv: The GGUF header key-value dictionary.
        key: The lookup key.

    Returns:
        The value as a string, or ``""`` if the key is missing or
        ``None``.
    """
    val = kv.get(key)
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _get_int(kv: dict, key: str, default: int = 0) -> int:
    """Safely retrieve an integer value from the GGUF key-value store.

    Args:
        kv: The GGUF header key-value dictionary.
        key: The lookup key.
        default: Value to return if the key is missing or cannot be
                 converted to ``int``.

    Returns:
        The value as an integer, or *default*.
    """
    val = kv.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default
