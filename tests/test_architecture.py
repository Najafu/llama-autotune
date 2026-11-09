"""The installed llama.cpp may be older than a model's architecture."""

from pathlib import Path

from llama_autotune.compat import (
    Capabilities,
    architecture_supported,
    unknown_architecture,
    update_llama_advice,
)
from llama_autotune.tuner import Tuner


def _fake_build(folder: Path, names: list[str]) -> Capabilities:
    """A llama.cpp folder whose library holds the given architecture table."""
    exe = folder / "llama-bench.exe"
    exe.write_bytes(b"MZ not a real program")
    (folder / "llama.dll").write_bytes(b"junk\0" + b"\0".join(n.encode() for n in names) + b"\0more")
    return Capabilities(bench_path=str(exe), server_path=str(exe))


def test_known_and_unknown_architectures(tmp_path):
    caps = _fake_build(tmp_path, ["llama", "lfm2", "qwen35"])
    assert architecture_supported(caps, "lfm2") is True
    assert architecture_supported(caps, "qwen35") is True
    assert architecture_supported(caps, "lfm3") is False
    assert architecture_supported(caps, "lfm") is False  # whole names only, never a prefix


def test_unknown_when_no_table(tmp_path):
    exe = tmp_path / "llama-bench.exe"
    exe.write_bytes(b"MZ")
    caps = Capabilities(bench_path=str(exe), server_path=str(exe))
    assert architecture_supported(caps, "lfm3") is None
    assert architecture_supported(Capabilities(), "lfm3") is None
    assert architecture_supported(_fake_build(tmp_path, ["llama"]), "") is None


def test_load_error_is_recognised():
    err = "llama_model_load: error loading model: unknown model architecture: 'lfm3'\n"
    assert unknown_architecture(err) == "lfm3"
    assert unknown_architecture("failed to create context") == ""


def test_advice_names_build_and_release_page():
    text = update_llama_advice("lfm3", 11347)
    assert "b11347" in text and "'lfm3'" in text
    assert "github.com/ggml-org/llama.cpp/releases" in text


def test_tuner_stops_before_measuring(tmp_path, monkeypatch):
    from llama_autotune import tuner as tuner_mod
    from llama_autotune.models import ModelInfo

    caps = _fake_build(tmp_path, ["llama"])
    caps.build.build_number = 11347
    monkeypatch.setattr(tuner_mod, "inspect_model", lambda p: ModelInfo(architecture="lfm3"))

    def no_sweep(*a, **k):
        raise AssertionError("must not benchmark a model llama.cpp cannot load")

    t = Tuner("model.gguf", caps=caps, hw=object(), sweep=no_sweep)
    t.plan = lambda: (_ for _ in ()).throw(AssertionError("must not plan memory first"))
    rep = t.run()
    assert rep.status == "unsupported"
    assert "b11347" in rep.error
