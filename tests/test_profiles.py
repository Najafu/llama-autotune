import json
import tempfile
from pathlib import Path

from llama_autotune.profiles import create_profile, export_profile, import_profile


def test_create_profile():
    prof = create_profile(
        name="test_profile",
        args=["--flash-attn", "--gpu-layers", "all"],
        model_path="/models/test.gguf",
        hardware="RTX 4090",
        score=95.5,
    )
    assert prof.name == "test_profile"
    assert prof.args == ["--flash-attn", "--gpu-layers", "all"]
    assert prof.score == 95.5
    assert prof.created != ""


def test_export_import_profile():
    prof = create_profile(
        name="roundtrip_test",
        args=["-t", "8", "-b", "2048"],
        model_path="/models/test.gguf",
        hardware="Test CPU",
        score=80.0,
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        temp_path = f.name
        export_profile(prof, temp_path)

    imported = import_profile(temp_path)
    assert imported.name == "roundtrip_test"
    assert imported.args == ["-t", "8", "-b", "2048"]
    assert imported.score == 80.0
    assert imported.model_path == "/models/test.gguf"

    Path(temp_path).unlink(missing_ok=True)
