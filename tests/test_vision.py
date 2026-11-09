"""Vision files (mmproj): matching by width, memory, and images being optional in launch."""

from pathlib import Path

import pytest

from llama_autotune import cli
from llama_autotune.constraints import GateDecision
from llama_autotune.memory import MemoryEstimate
from llama_autotune.models import ModelInfo
from llama_autotune.vision import MIB, find_vision_file, vision_bytes


def _files(tmp_path: Path, widths: dict[str, int]) -> None:
    for name, w in widths.items():
        (tmp_path / name).write_bytes(b"x" * 1024)


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """Model + three projectors, like the Qwen3.5 2B/4B/9B folder on the test PC."""
    widths = {"model-4B.gguf": 0, "mmproj-2B-f16.gguf": 2048, "mmproj-4B-BF16.gguf": 2560,
              "mmproj-9B-BF16.gguf": 4096, "mmproj-4B-f16-copy.gguf": 2560}
    _files(tmp_path, widths)
    monkeypatch.setattr("llama_autotune.vision._vision_width", lambda p: widths.get(Path(p).name, 0))
    return tmp_path


def test_vision_file_is_matched_by_width_not_name(folder):
    found = find_vision_file(str(folder / "model-4B.gguf"), ModelInfo(n_embd=2560))
    assert found.name in ("mmproj-4B-BF16.gguf", "mmproj-4B-f16-copy.gguf")
    assert find_vision_file(str(folder / "model-4B.gguf"), ModelInfo(n_embd=4096)).name == "mmproj-9B-BF16.gguf"
    assert find_vision_file(str(folder / "model-4B.gguf"), ModelInfo(n_embd=1234)) is None
    assert find_vision_file(str(folder / "model-4B.gguf"), ModelInfo(n_embd=0)) is None


def test_vision_memory_covers_the_measurement(tmp_path):
    f = tmp_path / "mmproj.gguf"
    with open(f, "wb") as h:
        h.truncate(637 * MIB)                          # Qwen3.5 2B projector
    assert vision_bytes(f) >= 912 * MIB                # measured: +642 loading, +288 reading an image


def _plan_factory(budget_mib: int, per_1k_mib: int = 100, base_mib: int = 2000):
    """A stand-in for plan_memory: need = base + context share + vision."""
    def plan(ctx, min_ctx, vision=0):
        need = (base_mib + per_1k_mib * ctx // 1024) * MIB + vision
        return GateDecision(need <= budget_mib * MIB, ctx, MemoryEstimate(weights=need), budget_mib * MIB)
    return plan


def test_images_kept_when_they_fit_at_the_wanted_context(tmp_path):
    vf = tmp_path / "mmproj.gguf"
    with open(vf, "wb") as h:
        h.truncate(600 * MIB)
    gate, used = cli._plan_with_optional_vision(_plan_factory(4000), 32768, 8192, False, vf)
    assert gate.fits and used == vf and gate.ctx == 8192


def test_images_left_out_before_the_context_is_cut(tmp_path):
    vf = tmp_path / "mmproj.gguf"
    with open(vf, "wb") as h:
        h.truncate(600 * MIB)
    # 2000 + 800 (8K) = 2800 fits in 3000 MiB, but not with ~900 MiB of vision
    gate, used = cli._plan_with_optional_vision(_plan_factory(3000), 32768, 8192, False, vf)
    assert gate.fits and used is None and gate.ctx == 8192


def test_max_context_reports_both_and_keeps_images_when_they_fit(tmp_path, capsys):
    vf = tmp_path / "mmproj.gguf"
    with open(vf, "wb") as h:
        h.truncate(600 * MIB)
    gate, used = cli._plan_with_optional_vision(_plan_factory(4000), 32768, 32768, True, vf)
    assert used == vf and gate.fits
    out = capsys.readouterr().out
    assert "With images the largest context is" in out and "without images" in out


def test_context_is_read_back_for_the_app_hint():
    assert cli._ctx_of(["-t", "2", "-c", "20480", "--alias", "m"]) == 20480
    assert cli._ctx_of(["--ctx-size", "4096"]) == 4096
    assert cli._ctx_of(["-t", "2"]) == 0
