from __future__ import annotations

import os
from pathlib import Path

import gguf

from .models import ModelInfo


def inspect_model(path: str | Path) -> ModelInfo:
    path = str(path)
    reader = gguf.GGUFReader(path)

    info = ModelInfo()
    info.path = path
    info.file_size_gb = round(os.path.getsize(path) / (1024**3), 2)

    kv = reader.fields

    info.architecture = _get_str(kv, "general.architecture")
    info.quantization = _get_str(kv, "general.file_type")
    info.parameters = _get_int(kv, "general.parameter_count", default=0)
    info.n_layers = _get_int(kv, "llama.block_count", default=0) or _get_int(
        kv, f"{info.architecture}.block_count", default=0
    )
    info.n_heads = _get_int(kv, "llama.attention.head_count", default=0) or _get_int(
        kv, f"{info.architecture}.attention.head_count", default=0
    )

    ctx = _get_int(kv, "llama.context_length", default=0) or _get_int(
        kv, f"{info.architecture}.context_length", default=0
    )
    info.training_context = ctx

    expert_count = _get_int(kv, "llama.expert_count", default=0) or _get_int(
        kv, f"{info.architecture}.expert_count", default=0
    )
    info.is_moe = expert_count > 1

    if info.is_moe:
        active = _get_int(
            kv, "llama.expert_used_count", default=0
        ) or _get_int(kv, f"{info.architecture}.expert_used_count", default=0)
        if active > 0:
            info.active_parameters = info.parameters
            info.parameters = info.parameters * expert_count // active
    else:
        info.active_parameters = info.parameters

    return info


def _get_str(kv: dict, key: str) -> str:
    field = kv.get(key)
    if field is None:
        field = kv.get(key.encode())
    if field is None:
        return ""
    try:
        val = field.value
        if isinstance(val, list | tuple) and len(val) > 0:
            val = val[0]
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")
        return str(val)
    except Exception:
        return ""


def _get_int(kv: dict, key: str, default: int = 0) -> int:
    field = kv.get(key)
    if field is None:
        field = kv.get(key.encode())
    if field is None:
        return default
    try:
        val = field.value
        if isinstance(val, list | tuple) and len(val) > 0:
            val = val[0]
        return int(val)
    except Exception:
        return default
