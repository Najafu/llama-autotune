"""Tests for the GGUF reader on synthetic files written by gguf_writer."""

from gguf_writer import write_gguf

from llama_autotune.model_inspector import (
    bytes_read_per_token,
    inspect_model,
    model_fingerprint,
    read_gguf,
)

Q4_K, Q6_K, F16, F32 = 12, 14, 1, 0


def _kv(**extra):
    return {"general.architecture": "llama", "llama.block_count": 2, "llama.embedding_length": 64,
            "llama.attention.head_count": 4, "llama.attention.head_count_kv": 2,
            "llama.context_length": 4096, "tokenizer.ggml.tokens": ["a", "b", "c"], **extra}


def _tensors(tied: bool):
    t = [("token_embd.weight", [64, 3], F16, 384),
         ("blk.0.ffn_up.weight", [64, 64], Q4_K, 2304),
         ("blk.1.ffn_up.weight", [64, 64], Q4_K, 2304),
         ("blk.0.attn_q.weight", [64, 64], Q6_K, 3360),
         ("output_norm.weight", [64], F32, 256)]
    if not tied:
        t.append(("output.weight", [64, 3], Q6_K, 192))
    return t


def test_read_gguf_tensor_sizes_from_offsets(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, _kv(), _tensors(tied=True))
    g = read_gguf(p)
    sizes = {t.name: t.nbytes for t in g.tensors}
    assert sizes["blk.0.ffn_up.weight"] == 2304
    assert sizes["blk.0.attn_q.weight"] == 3360
    assert g.kv["llama.block_count"] == 2


def test_bytes_per_token_tied_vs_untied(tmp_path):
    tied, untied = tmp_path / "tied.gguf", tmp_path / "untied.gguf"
    write_gguf(tied, _kv(), _tensors(tied=True))
    write_gguf(untied, _kv(), _tensors(tied=False))
    g_tied, g_untied = read_gguf(tied), read_gguf(untied)
    # tied: the embedding is also the output matrix, so it is read every token
    assert bytes_read_per_token(g_tied) == sum(t.nbytes for t in g_tied.tensors)
    # untied: the embedding is only looked up by row
    embd = next(t.nbytes for t in g_untied.tensors if t.name == "token_embd.weight")
    assert bytes_read_per_token(g_untied) == sum(t.nbytes for t in g_untied.tensors) - embd


def test_quant_from_tensors_beats_stale_metadata(tmp_path):
    p = tmp_path / "Model-Q8_0.gguf"
    write_gguf(p, _kv(**{"general.file_type": 18}), _tensors(tied=True))   # declares Q6_K
    info = inspect_model(p)
    assert info.quantization == "Q4_K"            # most transformer-weight bytes are Q4_K
    assert info.quantization_declared == "Q6_K"


def test_quant_uses_specific_metadata_label_when_consistent(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, _kv(**{"general.file_type": 15}), _tensors(tied=True))   # Q4_K_M
    assert inspect_model(p).quantization == "Q4_K_M"


def test_inspect_model_fields(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, _kv(), _tensors(tied=True))
    info = inspect_model(p)
    assert info.architecture == "llama"
    assert info.vocab_size == 3
    assert info.n_layers == 2
    assert info.kv_layout.k_elems_full == 2 * 2 * 16      # 2 layers x 2 KV heads x head size 16
    assert len(info.fingerprint) == 16


def test_fingerprint_survives_rename(tmp_path):
    a = tmp_path / "a.gguf"
    write_gguf(a, _kv(), _tensors(tied=True))
    fp = model_fingerprint(a)
    b = tmp_path / "renamed.gguf"
    a.rename(b)
    assert model_fingerprint(b) == fp


def test_non_gguf_file_returns_empty(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOTGGUF" * 10)
    assert read_gguf(p).tensors == []
