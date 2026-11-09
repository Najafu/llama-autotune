"""Extra llama-server options given to launch (sampling settings and the like)."""

from pathlib import Path

import pytest

from llama_autotune.compat import FlagRegistry, FlagSpec
from llama_autotune.database import connect, get_model_args, save_model_args
from llama_autotune.extra_args import ExtraArgsError, apply_over, merge, parse_extra_args, to_args

REG = FlagRegistry([
    FlagSpec(["--temp", "--temperature"], True),
    FlagSpec(["--top-k"], True),
    FlagSpec(["--top-p"], True),
    FlagSpec(["--repeat-penalty"], True),
    FlagSpec(["-n", "--predict", "--n-predict"], True),
    FlagSpec(["-t", "--threads"], True),
    FlagSpec(["-fa", "--flash-attn"], True, ["on", "off", "auto"]),
    FlagSpec(["--repack", "-nr", "--no-repack"], False),
    FlagSpec(["--seed"], True),
])


def test_llama_cpp_spellings():
    assert parse_extra_args(["--temp", "0.1", "--top-k", "50", "--repeat-penalty", "1.1"], REG) == \
        [("--temp", "0.1"), ("--top-k", "50"), ("--repeat-penalty", "1.1")]


def test_model_card_names_are_translated():
    words = ["temperature=0.1", "top_k=50", "--repetition_penalty", "1.1", "--top-p=0.9", "max_new_tokens=256"]
    assert parse_extra_args(words, REG) == [("--temp", "0.1"), ("--top-k", "50"), ("--repeat-penalty", "1.1"),
                                            ("--top-p", "0.9"), ("-n", "256")]


def test_negative_numbers_are_values():
    assert parse_extra_args(["--seed", "-1"], REG) == [("--seed", "-1")]


def test_typo_gets_a_suggestion():
    with pytest.raises(ExtraArgsError, match="Did you mean --top-k"):
        parse_extra_args(["--topk", "50"], REG)


def test_missing_value_and_bad_choice():
    with pytest.raises(ExtraArgsError, match="needs a value"):
        parse_extra_args(["--temp"], REG)
    with pytest.raises(ExtraArgsError, match="accepts on, off, auto"):
        parse_extra_args(["-fa", "maybe"], REG)
    with pytest.raises(ExtraArgsError, match="Unexpected"):
        parse_extra_args(["0.1"], REG)


def test_user_options_replace_generated_ones():
    generated = ["-t", "3", "-fa", "off", "--repack", "-c", "4096"]
    user = parse_extra_args(["--threads", "2", "--no-repack", "--temp", "0.1"], REG)
    args, replaced = apply_over(generated, user, REG)
    assert args == ["-fa", "off", "-c", "4096", "--threads", "2", "--no-repack", "--temp", "0.1"]
    assert replaced == ["-t", "--repack"]


def test_a_setting_given_again_replaces_the_remembered_value():
    saved = [("--temp", "0.1"), ("--top-k", "50")]
    assert merge(saved, [("--temperature", "0.7")], REG) == [("--temperature", "0.7"), ("--top-k", "50")]
    assert to_args(merge(saved, [], REG)) == ["--temp", "0.1", "--top-k", "50"]


def test_options_are_remembered_per_model(tmp_path: Path):
    conn = connect(str(tmp_path / "db.sqlite"))
    save_model_args(conn, "model-a", [("--temp", "0.1"), ("--no-repack", None)], "a.gguf")
    assert get_model_args(conn, "model-a") == [("--temp", "0.1"), ("--no-repack", None)]
    assert get_model_args(conn, "model-b") == []
    save_model_args(conn, "model-a", [])
    assert get_model_args(conn, "model-a") == []
