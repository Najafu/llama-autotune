"""Regression tests for the issues found in the review of the whole code base (after v0.5.0)."""

import json
import os
import struct

import pytest
from typer.testing import CliRunner

import llama_autotune.compat as compat
import llama_autotune.hardware as hardware
import llama_autotune.memory as memory
import llama_autotune.resources as resources
from llama_autotune import cli
from llama_autotune.compat import PROFILE_FLAGS, common_llama_dirs, foreign_profile_flags, parse_help
from llama_autotune.database import TunedSettings, adopt_hardware_id, connect, get_tuned, save_tuned
from llama_autotune.extra_args import ExtraArgsError, describe, is_secret, parse_extra_args
from llama_autotune.hardware import hardware_fingerprint, legacy_hardware_fingerprint
from llama_autotune.model_inspector import GGUFError, inspect_model
from llama_autotune.models import HardwareInfo, SearchConfig
from llama_autotune.tuner import Candidate, Speed, Tuner, TuneReport

runner = CliRunner()
MIB = 1024 * 1024


def _gguf_header(n_tensors: int = 0, n_kv: int = 0) -> bytes:
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", n_tensors) + struct.pack("<Q", n_kv)


# ── damaged model files ──────────────────────────────────────────────


def test_a_file_cut_short_is_a_gguf_error(tmp_path):
    path = tmp_path / "cut.gguf"
    path.write_bytes(_gguf_header(n_tensors=5, n_kv=5) + b"\x01\x02")
    with pytest.raises(GGUFError):
        inspect_model(path)


def test_a_header_claiming_a_huge_string_is_refused_not_allocated(tmp_path):
    path = tmp_path / "huge.gguf"
    path.write_bytes(_gguf_header(n_kv=1) + struct.pack("<Q", 1 << 60) + b"x" * 64)
    with pytest.raises(GGUFError):
        inspect_model(path)


def test_inspect_explains_a_damaged_file_in_plain_words(tmp_path):
    path = tmp_path / "cut.gguf"
    path.write_bytes(_gguf_header(n_kv=3))
    result = runner.invoke(cli.app, ["inspect", str(path)])
    assert result.exit_code == 1 and "Cannot read this model" in result.output
    assert "Traceback" not in result.output


# ── launch: the context the plan agreed to is the context that runs ──


@pytest.mark.parametrize("ctx", ["0", "100", "-5", "lots"])
def test_launch_refuses_a_context_the_plan_cannot_stand_behind(tmp_path, ctx):
    model = tmp_path / "m.gguf"
    model.write_bytes(_gguf_header())
    result = runner.invoke(cli.app, ["launch", str(model), "--dry-run", f"--ctx-size={ctx}"])
    assert result.exit_code == 1 and "Context must be" in result.output


# ── flash attention after the long-chat test ─────────────────────────


def _fa_case(baseline_fa, saved_fa, by_depth):
    rep = TuneReport(best_tps=10.0, depth=2048, baseline=Candidate(2, baseline_fa, None))
    cfg = SearchConfig(flash_attn=saved_fa)
    pool = {Candidate(2, False): Speed(tg=9.0), Candidate(2, True): Speed(tg=10.0)}
    Tuner._choose_flash_attn(rep, cfg, by_depth, pool, pool, 0.04)
    return rep, cfg


def test_speed_stands_when_auto_already_meant_the_winner():
    rep, cfg = _fa_case(None, None, {False: 5.0, True: 8.0})   # defaults (auto = on here) won, on wins at depth
    assert cfg.flash_attn is True                            # pinned
    assert rep.best_tps == pytest.approx(10.0)               # not multiplied by on/off again


def test_speed_is_rescaled_when_the_setting_really_changes():
    rep, cfg = _fa_case(None, None, {False: 8.0, True: 5.0})   # auto = on, off wins at depth
    assert cfg.flash_attn is False
    assert rep.best_tps == pytest.approx(9.0)


def test_reported_default_decides_what_auto_meant():
    rep, cfg = _fa_case(False, None, {False: 8.0, True: 5.0})  # llama-bench said the default ran with it off
    assert cfg.flash_attn is False and rep.best_tps == pytest.approx(10.0)


# ── hardware id ──────────────────────────────────────────────────────


def test_hardware_id_survives_an_os_update(monkeypatch):
    hw = HardwareInfo(cpu_name="Sim", ram_gb=8.0)
    monkeypatch.setattr(hardware.platform, "release", lambda: "6.8.0-45-generic")
    before, old = hardware_fingerprint(hw), legacy_hardware_fingerprint(hw)
    monkeypatch.setattr(hardware.platform, "release", lambda: "6.8.0-47-generic")
    assert hardware_fingerprint(hw) == before
    assert legacy_hardware_fingerprint(hw) != old


def test_results_stored_under_the_old_id_are_adopted(tmp_path):
    conn = connect(str(tmp_path / "db.sqlite"))
    save_tuned(conn, "old-id", "m", TunedSettings(config=SearchConfig(threads=3), generation_tps=12.0))
    adopt_hardware_id(conn, "old-id", "new-id")
    assert get_tuned(conn, "new-id", "m").config.threads == 3
    assert get_tuned(conn, "old-id", "m") is None


# ── profiles from elsewhere ──────────────────────────────────────────


def test_profile_options_llama_autotune_never_writes_are_named():
    args = ["-t", "3", "-c", "4096", "--host", "0.0.0.0", "--log-file", "x.txt", "--prio", "-1"]
    assert foreign_profile_flags(args) == ["--host", "--log-file"]


def test_every_spelling_of_an_allowed_option_passes():
    registry = parse_help("-t,    --threads N        number of threads\n-c,    --ctx-size N       context\n"
                          "--host HOST               address\n")
    assert foreign_profile_flags(["--threads", "3", "--ctx-size", "4096"], registry) == []
    assert foreign_profile_flags(["--host", "0.0.0.0"], registry) == ["--host"]


def test_profiles_written_by_launch_defaults_pass_the_check():
    from llama_autotune.compat import render_args
    from llama_autotune.heuristics import apply_launch_defaults
    from llama_autotune.models import Backend, ModelInfo

    cfg = apply_launch_defaults(SearchConfig(threads=3, threads_batch=4, ctx_size=4096, flash_attn=False,
                                             repack=False, ubatch_size=128, numa="distribute"),
                                HardwareInfo(ram_gb=8.0, backend=Backend.CPU), ModelInfo())
    args = render_args(cfg, "server").args
    assert foreign_profile_flags(args) == []
    assert "-np" in PROFILE_FLAGS


def test_import_says_so_when_a_file_is_not_a_profile(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text("[1, 2, 3]")
    result = runner.invoke(cli.app, ["import", str(bad)])
    assert result.exit_code == 1 and "not a llama-autotune profile" in result.output


# ── extra options ────────────────────────────────────────────────────


def test_a_second_model_cannot_arrive_as_an_extra_option():
    with pytest.raises(ExtraArgsError, match="already loads the model"):
        parse_extra_args(["-m", "other.gguf"], None)
    with pytest.raises(ExtraArgsError):
        parse_extra_args(["--hf-repo", "someone/model"], None)


def test_secret_values_are_masked():
    pairs = parse_extra_args(["--api-key", "hunter2", "--temp", "0.1"], None)
    assert is_secret("--api-key", None) and not is_secret("--temp", None)
    assert describe(pairs) == "--api-key *** --temp 0.1"


# ── where llama.cpp is looked for ────────────────────────────────────


def test_the_current_folder_is_not_searched_for_llama_cpp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert tmp_path.resolve() not in [d.resolve() for d in common_llama_dirs()]


# ── printing commands and names ──────────────────────────────────────


def test_commands_are_quoted_for_the_shell():
    spaced = cli._quote("C:\\my models\\m.gguf" if os.name == "nt" else "/my models/m.gguf")
    assert spaced[0] in "\"'" and spaced[-1] in "\"'"
    assert cli._quote("plain.gguf") == "plain.gguf"
    if os.name == "nt":
        assert cli._quote("a&b(1).gguf") == '"a&b(1).gguf"'


def test_models_survives_names_that_look_like_markup(monkeypatch, tmp_path):
    from llama_autotune.compat import Capabilities
    from llama_autotune.database import get_session
    from llama_autotune.resources import MemoryStatus

    monkeypatch.setattr(cli, "detect_capabilities", lambda: Capabilities())
    monkeypatch.setattr(cli, "detect_hardware", lambda caps=None: HardwareInfo())
    monkeypatch.setattr(cli, "hardware_fingerprint", lambda hw: "hw")
    monkeypatch.setattr(cli, "memory_status", lambda: MemoryStatus(total=8 * 1024**3, available=3000 * MIB))
    conn = get_session()
    save_tuned(conn, "hw", "m", TunedSettings(config=SearchConfig(), generation_tps=12.9, llama_build=0,
                                              model_path=str(tmp_path / "[old]" / "granite[bold].gguf")))
    conn.close()
    result = runner.invoke(cli.app, ["models"])
    assert result.exit_code == 0, result.output
    assert "granite[bold]" in result.output
    assert " b0" not in result.output           # an unknown build is left out, not shown as b0


# ── llama-fit-params answers are remembered between runs ─────────────


def test_compute_buffer_is_asked_once_per_file_and_tool(monkeypatch, tmp_path):
    model, tool = tmp_path / "m.gguf", tmp_path / "llama-fit-params"
    model.write_bytes(b"x" * 10)
    tool.write_bytes(b"tool")
    asked = []
    monkeypatch.setattr(compat, "find_llama_binary", lambda base: str(tool))
    monkeypatch.setattr(memory, "llama_compute_buffer", lambda exe, path, ctx: asked.append(path) or 258 * MIB)
    st = os.stat(model)
    assert memory._compute_at_4096(str(model), st.st_size, st.st_mtime_ns) == 258 * MIB
    memory._compute_at_4096.cache_clear()                      # a new process
    assert memory._compute_at_4096(str(model), st.st_size, st.st_mtime_ns) == 258 * MIB
    assert len(asked) == 1
    assert list(json.loads((tmp_path / "cache" / "compute-buffers.json").read_text()).values()) == [258 * MIB]
    tool.write_bytes(b"a newer tool")                          # llama.cpp was upgraded
    memory._compute_at_4096.cache_clear()
    memory._compute_at_4096(str(model), st.st_size, st.st_mtime_ns)
    assert len(asked) == 2


# ── the bandwidth test stays small when memory is short ──────────────


def test_bandwidth_buffers_take_at_most_a_quarter_of_free_ram(monkeypatch):
    class Mem:
        available = 256 * MIB

    sizes = []
    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: Mem)
    monkeypatch.setattr(resources, "_threaded_bandwidth",
                        lambda threads, size_mib, repeats: sizes.append((threads, size_mib)) or 0.0)
    assert resources.measure_bandwidth(threads=16, repeats=1) > 0
    assert all(threads * 2 * size <= 64 for threads, size in sizes)   # 64 MiB = a quarter of what is free
