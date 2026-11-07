"""Write tiny synthetic GGUF files for tests (header, tensor table, zeroed data)."""

from __future__ import annotations

import struct

_U32, _F32, _BOOL, _STR, _ARR, _U64 = 4, 6, 7, 8, 9, 10


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _value(v) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I?", _BOOL, v)
    if isinstance(v, int):
        return struct.pack("<II", _U32, v) if v < 2**32 else struct.pack("<IQ", _U64, v)
    if isinstance(v, float):
        return struct.pack("<If", _F32, v)
    if isinstance(v, str):
        return struct.pack("<I", _STR) + _string(v)
    if isinstance(v, list):
        if all(isinstance(x, bool) for x in v):
            return struct.pack("<IIQ", _ARR, _BOOL, len(v)) + b"".join(struct.pack("<?", x) for x in v)
        if all(isinstance(x, str) for x in v):
            return struct.pack("<IIQ", _ARR, _STR, len(v)) + b"".join(_string(x) for x in v)
        return struct.pack("<IIQ", _ARR, _U32, len(v)) + b"".join(struct.pack("<I", x) for x in v)
    raise TypeError(type(v))


def write_gguf(path, kv: dict, tensors: list[tuple[str, list[int], int, int]], align: int = 32) -> None:
    """Write a GGUF file.

    Args:
        path: Output path.
        kv: Metadata key-values (int, float, bool, str or lists of them).
        tensors: ``(name, dims, ggml_type, nbytes)`` per tensor.
        align: Data alignment.
    """
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, value in kv.items():
        head += _string(key) + _value(value)
    offset = 0
    offsets = []
    for _, _, _, nbytes in tensors:
        offsets.append(offset)
        offset += -(-nbytes // align) * align
    for (name, dims, ggml_type, _), off in zip(tensors, offsets):
        head += _string(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        head += struct.pack("<IQ", ggml_type, off)
    head += b"\0" * (-len(head) % align)
    with open(path, "wb") as f:
        f.write(head)
        for (_, _, _, nbytes), off in zip(tensors, offsets):
            f.write(b"\0" * (-(-nbytes // align) * align))
