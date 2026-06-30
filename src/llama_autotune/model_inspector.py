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
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    10: "Q8_1",
    16: "IQ1_S",
    17: "IQ1_M",
    18: "IQ2_XXS",
    19: "IQ2_XS",
    20: "IQ2_S",
    21: "IQ2_M",
    22: "IQ2_S",
    23: "IQ3_XXS",
    24: "IQ3_XS",
    25: "IQ3_S",
    26: "IQ3_M",
    27: "IQ4_XS",
    28: "IQ4_NL",
    29: "IQ4_XS",
    30: "IQ4_XS",
    31: "IQ4_NL",
}


def inspect_model(path: str | Path) -> ModelInfo:
    path = str(path)
    info = ModelInfo()
    info.path = path
    info.file_size_gb = round(os.path.getsize(path) / (1024**3), 2)

    kv = _read_gguf_header(path)

    info.architecture = _get_str(kv, "general.architecture")
    info.quantization = _resolve_file_type(kv)
    info.parameters = _resolve_param_count(kv)
    info.n_layers = _resolve_block_count(kv, info.architecture)
    info.n_heads = _get_int(kv, f"{info.architecture}.attention.head_count", default=0)
    info.training_context = _get_int(kv, f"{info.architecture}.context_length", default=0)

    expert_count = _get_int(kv, f"{info.architecture}.expert_count", default=0)
    info.is_moe = expert_count > 1

    if info.is_moe:
        active = _get_int(kv, f"{info.architecture}.expert_used_count", default=0)
        if active > 0:
            info.active_parameters = info.parameters * active // expert_count
    else:
        info.active_parameters = info.parameters

    return info


def _resolve_file_type(kv: dict) -> str:
    raw = _get_int(kv, "general.file_type", default=-1)
    if raw in _KNOWN_FILE_TYPES:
        return _KNOWN_FILE_TYPES[raw]
    name = _get_str(kv, "general.name")
    if name:
        parts = name.split()
        for p in parts:
            if any(x in p for x in ["Q4", "Q5", "Q8", "IQ", "F16", "F32", "BF16"]):
                return p
    return str(raw) if raw >= 0 else "unknown"


def _resolve_param_count(kv: dict) -> int:
    count = _get_int(kv, "general.parameter_count", default=0)
    if count > 0:
        return count
    label = _get_str(kv, "general.size_label")
    if label:
        multipliers = {"M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
        for suffix, mult in multipliers.items():
            if suffix in label:
                try:
                    num_str = label.replace(suffix, "").strip()
                    num = float(num_str)
                    return int(num * mult)
                except ValueError:
                    pass
    return 0


def _resolve_block_count(kv: dict, arch: str) -> int:
    count = _get_int(kv, f"{arch}.block_count", default=0)
    if count == 0:
        count = _get_int(kv, "llama.block_count", default=0)
    return count


def _read_gguf_header(path: str) -> dict[str, object]:
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
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f) -> object:
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
    val = kv.get(key)
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _get_int(kv: dict, key: str, default: int = 0) -> int:
    val = kv.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default
