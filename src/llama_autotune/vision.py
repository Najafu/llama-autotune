"""Vision files (multimodal projectors, ``mmproj*.gguf``) that belong to a model.

llama-server only loads a local vision file when it is named with ``--mmproj``.
The file that belongs to a model is found by its metadata, not its name: a
projector's ``clip.vision.projection_dim`` equals the embedding width of the
text model it feeds (in a folder holding Qwen3.5 2B, 4B and 9B projectors these
are 2,048, 2,560 and 4,096, one per model).
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

from .model_inspector import read_gguf
from .models import ModelInfo

MIB = 1024 * 1024


def _vision_width(path: Path) -> int:
    """The projector's output width, or 0 when the file is not a vision projector."""
    try:
        kv = read_gguf(str(path)).kv
    except (OSError, ValueError, EOFError, UnicodeDecodeError):  # unreadable or not GGUF
        return 0
    if kv.get("general.architecture") != "clip" or not kv.get("clip.has_vision_encoder", False):
        return 0
    return int(kv.get("clip.vision.projection_dim") or 0)


def find_vision_file(model_path: str, model: ModelInfo) -> Path | None:
    """The vision file in the model's folder whose width matches the model.

    Ties (two projectors of the same width, e.g. F16 and BF16 copies) go to
    the name closest to the model's file name.
    """
    folder = Path(model_path).resolve().parent
    if not model.n_embd or not folder.is_dir():
        return None
    matches = [p for p in sorted(folder.glob("*.gguf"))
               if p.resolve() != Path(model_path).resolve() and _vision_width(p) == model.n_embd]
    if not matches:
        return None
    stem = Path(model_path).stem.lower()
    return max(matches, key=lambda p: difflib.SequenceMatcher(None, stem, p.stem.lower()).ratio())


def is_vision_file(path: str) -> bool:
    """True for a GGUF vision projector."""
    return _vision_width(Path(path)) > 0


def vision_bytes(path: str | Path) -> int:
    """Memory a loaded vision file takes, including reading one image.

    Measured on llama.cpp b12228 with the Qwen3.5 2B projector (637 MiB file):
    +642 MiB after loading and +288 MiB more while reading a 1280x960 image,
    912 MiB in all (llama.cpp's own worst-case estimate: 861 MiB). The file
    size plus the larger of 256 MiB and half the file (955 MiB here) covers it.
    """
    size = os.path.getsize(path)
    return size + max(256 * MIB, size // 2)
