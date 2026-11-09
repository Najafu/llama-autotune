"""Finding llama.cpp without environment variables: setup command and common folders."""

from pathlib import Path

from typer.testing import CliRunner

from llama_autotune import cli, compat
from llama_autotune.compat import binary_name, common_llama_dirs, find_llama_binary, saved_llama_dir

runner = CliRunner()


def _fake_llama(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for tool in ("llama-server", "llama-bench"):
        (folder / binary_name(tool)).write_bytes(b"")
    return folder


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(compat, "config_path", lambda: tmp_path / "config.json")
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    monkeypatch.setattr(compat.shutil, "which", lambda name: None)
    monkeypatch.setattr(compat, "common_llama_dirs", lambda: [])


def test_setup_saves_the_folder_and_finds_nested_binaries(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    inner = _fake_llama(tmp_path / "unzipped" / "llama-b12228-bin-win-cpu-x64")
    monkeypatch.setattr(cli, "detect_capabilities", lambda: compat.Capabilities())
    result = runner.invoke(cli.app, ["setup", str(tmp_path / "unzipped")])
    assert result.exit_code == 0, result.output
    assert Path(saved_llama_dir()) == inner.resolve()
    assert find_llama_binary("llama-server") == str((inner / binary_name("llama-server")).resolve())


def test_setup_refuses_a_folder_without_llama_cpp(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    (tmp_path / "empty").mkdir()
    result = runner.invoke(cli.app, ["setup", str(tmp_path / "empty")])
    assert result.exit_code == 1 and "No " in result.output
    assert saved_llama_dir() == ""


def test_environment_variable_wins_over_the_saved_folder(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    saved, env = _fake_llama(tmp_path / "saved"), _fake_llama(tmp_path / "env")
    compat.save_llama_dir(str(saved))
    monkeypatch.setenv("LLAMA_CPP_DIR", str(env))
    assert Path(find_llama_binary("llama-bench")).parent == env.resolve()


def test_newest_download_is_found_without_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(compat.Path, "home", lambda: tmp_path)
    _fake_llama(tmp_path / "Downloads" / "llama-b11347-bin-win-cpu-x64")
    newest = _fake_llama(tmp_path / "Downloads" / "llama-b12228-bin-win-cpu-x64")
    downloads = [d for d in common_llama_dirs() if "Downloads" in str(d)]
    assert downloads[0] == newest


def test_commands_explain_how_to_get_llama_cpp(monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(cli, "detect_capabilities",
                        lambda: compat.Capabilities(server_path="llama-server.exe", bench_path="llama-bench.exe"))
    result = runner.invoke(cli.app, ["launch", str(model)])
    assert result.exit_code == 1
    assert "llama.cpp was not found" in result.output and "llama-autotune setup" in result.output
