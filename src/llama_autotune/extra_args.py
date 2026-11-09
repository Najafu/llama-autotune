"""Extra llama-server options given to ``launch``, such as sampling settings.

Users copy settings from model cards, which use Hugging Face or OpenAI names
(``temperature``, ``top_k``, ``repetition_penalty``). Both those names and
llama.cpp's own spellings are accepted, as ``--temp 0.1``, ``--temp=0.1`` or
``temperature=0.1``. Every option is checked against the installed build's
``--help`` before the server starts, so a typo gets a suggestion instead of
a server that exits.
"""

from __future__ import annotations

import difflib
import re

from .compat import FlagRegistry, FlagSpec

#: Familiar names (lower case, underscores) -> llama-server spelling.
FRIENDLY_NAMES: dict[str, str] = {
    "temperature": "--temp",
    "temp": "--temp",
    "top_k": "--top-k",
    "top_p": "--top-p",
    "min_p": "--min-p",
    "typical_p": "--typical",
    "repetition_penalty": "--repeat-penalty",
    "repeat_penalty": "--repeat-penalty",
    "presence_penalty": "--presence-penalty",
    "frequency_penalty": "--frequency-penalty",
    "max_tokens": "-n",
    "max_new_tokens": "-n",
    "seed": "--seed",
}

_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")


class ExtraArgsError(ValueError):
    """An extra option the installed llama-server would not accept."""


def _resolve(name: str, registry: FlagRegistry | None) -> str:
    key = name.lstrip("-").lower().replace("-", "_")
    if key in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[key]
    candidates = [name] if name.startswith("-") else []
    candidates.append("--" + key.replace("_", "-"))
    for c in candidates:
        if registry is None or registry.has(c):
            return c
    known = sorted({n for s in registry.specs for n in s.names} | {f"{k}=" for k in FRIENDLY_NAMES})
    close = difflib.get_close_matches(name if name.startswith("-") else "--" + name, known, n=1, cutoff=0.6)
    hint = f" Did you mean {close[0].rstrip('=')}?" if close else ""
    raise ExtraArgsError(f"This llama.cpp build has no option '{name}'.{hint}")


def parse_extra_args(tokens: list[str], registry: FlagRegistry | None) -> list[tuple[str, str | None]]:
    """Turn the words after the model into ``(flag, value)`` pairs.

    Args:
        tokens: Extra command-line words, e.g. ``["--temp", "0.1", "top_k=50"]``.
        registry: The installed llama-server's options (None: not checked).

    Returns:
        Pairs in llama-server spelling; the value is None for switches.

    Raises:
        ExtraArgsError: For an unknown option or a missing value.
    """
    pairs: list[tuple[str, str | None]] = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        if not word.startswith("-") and "=" not in word:
            raise ExtraArgsError(f"Unexpected '{word}': write options as --temp 0.1 or temperature=0.1.")
        name, value = word.split("=", 1) if "=" in word else (word, None)
        flag = _resolve(name, registry)
        spec = registry.get(flag) if registry else None
        takes_value = spec.takes_value if spec else value is not None or (
            i + 1 < len(tokens) and (not tokens[i + 1].startswith("-") or bool(_NUMBER.match(tokens[i + 1]))))
        if takes_value and value is None:
            if i + 1 >= len(tokens) or (tokens[i + 1].startswith("-") and not _NUMBER.match(tokens[i + 1])):
                raise ExtraArgsError(f"Option '{name}' needs a value, e.g. {flag} 0.5.")
            value = tokens[i + 1]
            i += 1
        if not takes_value and value is not None:
            raise ExtraArgsError(f"Option '{name}' is a switch and takes no value.")
        if spec and spec.choices and value is not None and value not in spec.choices:
            raise ExtraArgsError(f"Option '{flag}' accepts {', '.join(spec.choices)}, not '{value}'.")
        pairs.append((flag, value))
        i += 1
    return pairs


def setting_key(flag: str, registry: FlagRegistry | None) -> str:
    """One name per setting, whatever spelling was used (``-t`` and ``--threads``)."""
    spec = registry.get(flag) if registry else None
    return spec.names[0] if spec else flag


def merge(saved: list[tuple[str, str | None]], new: list[tuple[str, str | None]],
          registry: FlagRegistry | None) -> list[tuple[str, str | None]]:
    """Saved settings updated by new ones; a setting given again replaces its old value."""
    merged = {setting_key(f, registry): (f, v) for f, v in saved}
    for f, v in new:
        merged[setting_key(f, registry)] = (f, v)
    return list(merged.values())


def to_args(pairs: list[tuple[str, str | None]]) -> list[str]:
    """Flatten pairs into command-line words."""
    return [w for f, v in pairs for w in ((f,) if v is None else (f, v))]


def apply_over(args: list[str], pairs: list[tuple[str, str | None]],
               registry: FlagRegistry | None) -> tuple[list[str], list[str]]:
    """Append the user's options, removing generated options they replace.

    Returns:
        The final argument list and the generated flags that were replaced.
    """
    if not pairs:
        return list(args), []
    user = {setting_key(f, registry) for f, _ in pairs}
    out: list[str] = []
    replaced: list[str] = []
    i = 0
    while i < len(args):
        word = args[i]
        spec: FlagSpec | None = registry.get(word) if registry and word.startswith("-") else None
        is_user_setting = word.startswith("-") and setting_key(word, registry) in user
        if is_user_setting:
            replaced.append(word)
            takes = spec.takes_value if spec else (i + 1 < len(args) and not args[i + 1].startswith("-"))
            i += 2 if takes else 1
            continue
        out.append(word)
        i += 1
    return out + to_args(pairs), replaced


def describe(pairs: list[tuple[str, str | None]]) -> str:
    """Short plain text for the launch output."""
    return " ".join(to_args(pairs))
