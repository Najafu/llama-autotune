"""Tests for cli.py — command registration and error handling."""

from typer.testing import CliRunner

from llama_autotune.cli import app

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "llama-autotune" in result.output


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["inspect", "benchmark", "search", "launch", "export", "import"]:
        assert cmd in result.output


def test_import_command_is_named_import():
    """The command must be `import` (as documented), not `import-cmd`."""
    result = runner.invoke(app, ["--help"])
    assert "import-cmd" not in result.output
    result = runner.invoke(app, ["import", "--help"])
    assert result.exit_code == 0


def test_inspect_missing_model_exits_nonzero():
    result = runner.invoke(app, ["inspect", "no_such_model.gguf"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_import_missing_profile_exits_nonzero():
    result = runner.invoke(app, ["import", "no_such_profile.json"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_doctor_runs():
    result = runner.invoke(app, ["doctor", "--no-bandwidth"])
    assert result.exit_code == 0
    assert "llama.cpp" in result.output


def test_launch_dry_run_writes_safe_defaults():
    import os

    import pytest

    from llama_autotune.compat import detect_capabilities

    model = r"C:\sandbox\llama\models\lama\Llama-3.2-1B-Instruct-GPTQ-Q6_K.gguf"
    if not os.path.exists(model) or not detect_capabilities().build.build_number:
        pytest.skip("needs the local model and llama.cpp")
    result = runner.invoke(app, ["launch", model, "--dry-run", "-c", "2048"])
    if result.exit_code == 1 and "does not fit" in result.output:
        pytest.skip("not enough free RAM right now")
    assert result.exit_code == 0, result.output
    command = result.output.replace("\n", " ")
    assert "-np 1" in command and "--no-mmproj" in command
    assert "--mlock" not in command and "--no-mmap" not in command


def test_search_invalid_objective_exits_nonzero(tmp_path):
    fake_model = tmp_path / "model.gguf"
    fake_model.write_bytes(b"GGUF")
    result = runner.invoke(app, ["search", str(fake_model), "--objective", "bogus"])
    assert result.exit_code == 1
    assert "Invalid objective" in result.output
