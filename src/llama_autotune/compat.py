"""llama.cpp compatibility layer.

The only module that knows how llama.cpp spells its command-line flags.
It discovers what an installed build supports by parsing ``--version``,
``--help`` and ``--list-devices``, caches the result per binary, and
renders a typed :class:`~llama_autotune.models.SearchConfig` into the
arguments that build accepts. A setting the build cannot express is
skipped with a reason instead of being emitted, so an unknown or older
build never receives a flag it would reject.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .models import SearchConfig

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
_TOOLS = ("server", "bench")


# ── locating binaries ────────────────────────────────────────────────


def binary_name(base: str) -> str:
    """Return the platform-specific executable name for a llama.cpp tool.

    Args:
        base: Base tool name, e.g. ``"llama-bench"``.

    Returns:
        ``"<base>.exe"`` on Windows, ``base`` unchanged elsewhere.
    """
    return f"{base}.exe" if os.name == "nt" else base


def config_path() -> Path:
    """``~/.llama-autotune/config.json``: settings such as the llama.cpp folder."""
    return Path.home() / ".llama-autotune" / "config.json"


def cache_dir() -> Path:
    """``~/.llama-autotune/cache``: answers from llama.cpp that are slow to ask for again."""
    return Path.home() / ".llama-autotune" / "cache"


def saved_llama_dir() -> str:
    """The llama.cpp folder saved by ``llama-autotune setup``, or ``""``."""
    try:
        return str(json.loads(config_path().read_text(encoding="utf-8")).get("llama_cpp_dir", ""))
    except (OSError, ValueError, AttributeError):
        return ""


def save_llama_dir(folder: str) -> None:
    """Remember the llama.cpp folder for every later command."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data["llama_cpp_dir"] = str(Path(folder).resolve())
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _build_in_name(path: Path) -> int:
    m = re.search(r"-b(\d+)-", path.name)
    return int(m.group(1)) if m else 0


def common_llama_dirs() -> list[Path]:
    """Folders where beginners usually unzip llama.cpp, newest download first.

    The official Windows zips are named like ``llama-b12228-bin-win-cpu-x64``;
    when several sit in Downloads, the highest build number wins. The current
    folder is not searched: a program that happens to be named llama-bench in
    a folder someone else packed must not be run just because we stand in it.
    """
    home = Path.home()
    fixed = [Path("C:/llama"), Path("C:/llama.cpp"), Path("C:/llamacpp"),
             home / "llama", home / "llama.cpp", home / "llamacpp"] if os.name == "nt" else \
            [home / "llama.cpp", home / "llama.cpp" / "build" / "bin", Path("/opt/llama.cpp")]
    downloads = sorted((p for p in (home / "Downloads").glob("llama-*bin-*") if p.is_dir()),
                       key=_build_in_name, reverse=True) if (home / "Downloads").is_dir() else []
    return fixed + downloads


def find_llama_binary(base: str) -> str:
    """Locate a llama.cpp executable by base name.

    Looks in this order: the ``LLAMA_CPP_DIR`` environment variable, the
    folder saved by ``llama-autotune setup``, folders next to the package,
    the system ``PATH``, and finally common unzip locations (``C:\\llama``,
    Downloads).

    Args:
        base: Base tool name, e.g. ``"llama-bench"`` or ``"llama-server"``.

    Returns:
        Absolute path to the executable, or the bare platform-specific
        name as a last resort if none of the candidates exist.
    """
    exe = binary_name(base)
    candidates = []
    for folder in (os.environ.get("LLAMA_CPP_DIR"), saved_llama_dir()):
        if folder:
            candidates.append(os.path.join(folder, exe))
    here = os.path.dirname(__file__)
    candidates.extend([
        os.path.join(here, "..", "..", "..", exe),
        os.path.join(here, "..", "..", exe),
    ])
    for c in candidates:
        resolved = os.path.abspath(c)
        if os.path.isfile(resolved):
            return resolved
    on_path = shutil.which(exe)
    if on_path:
        return on_path
    for folder in common_llama_dirs():
        if (folder / exe).is_file():
            return str((folder / exe).resolve())
    return exe


def llama_found(path: str) -> bool:
    """True when a lookup above returned a real file, not the bare name."""
    return os.path.isfile(path)


NOT_FOUND_ADVICE = (
    "llama.cpp was not found. Download the newest release from https://github.com/ggml-org/llama.cpp/releases "
    "(for a PC without a GPU: the file ending in bin-win-cpu-x64.zip), unzip it, then run: "
    "llama-autotune setup <the folder you unzipped to>")


# ── parsing ───────────────────────────────────────────────────────────


@dataclass
class FlagSpec:
    """One option line of a ``--help`` listing.

    Attributes:
        names: Every spelling of the option, e.g. ``["-t", "--threads"]``.
        takes_value: Whether the option expects an argument.
        choices: Allowed values when the help lists them (``<0|1>``,
            ``[on|off]``, ``{none,layer}``), else an empty list.
    """

    names: list[str]
    takes_value: bool
    choices: list[str] = field(default_factory=list)


@dataclass
class FlagRegistry:
    """All options one llama.cpp binary accepts."""

    specs: list[FlagSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_name = {n: s for s in self.specs for n in s.names}

    def get(self, name: str) -> FlagSpec | None:
        """Return the spec for a flag spelling, or None if unsupported."""
        return self._by_name.get(name)

    def has(self, name: str) -> bool:
        """Return True if the binary accepts this flag spelling."""
        return name in self._by_name

    def to_json(self) -> list[dict[str, Any]]:
        return [s.__dict__ for s in self.specs]

    @classmethod
    def from_json(cls, data: list[dict[str, Any]]) -> FlagRegistry:
        return cls([FlagSpec(**d) for d in data])


_OPTION_LINE = re.compile(r"^\s{0,4}-{1,2}[A-Za-z]")
_FLAG_TOKEN = re.compile(r"^-{1,2}[A-Za-z][\w.+-]*,?$")
_VALUE_TOKEN = re.compile(r"^(<.*|\[.*|\{.*|[A-Z][\w,.\-+]*|\.\.\.|lo-hi|\S*[,|]\S*)$")
_CHOICES = re.compile(r"[<\[{]([^<>\[\]{}]+)[>\]}]")


def parse_help(text: str) -> FlagRegistry:
    """Parse a llama.cpp ``--help`` listing into a flag registry.

    Option lines start (after at most four spaces) with a flag; their
    descriptions are indented further and are ignored, so flags merely
    mentioned in a description are not registered.

    Args:
        text: The help output of ``llama-server`` or ``llama-bench``.

    Returns:
        The parsed :class:`FlagRegistry`.
    """
    specs: list[FlagSpec] = []
    for line in text.splitlines():
        if not _OPTION_LINE.match(line):
            continue
        names: list[str] = []
        value_tokens: list[str] = []
        for tok in line.split():
            if not value_tokens and _FLAG_TOKEN.match(tok):
                names.append(tok.rstrip(","))
            elif names and _VALUE_TOKEN.match(tok):
                value_tokens.append(tok)
            else:
                break
        if not names:
            continue
        value = " ".join(value_tokens)
        choices: list[str] = []
        m = _CHOICES.search(value)
        if m and ("|" in m.group(1) or "," in m.group(1)):
            choices = [c.strip() for c in re.split(r"[|,]", m.group(1)) if c.strip() and c.strip() != ".."]
        specs.append(FlagSpec(names=names, takes_value=bool(value), choices=choices))
    return FlagRegistry(specs)


@dataclass
class BuildInfo:
    """Version information printed by ``--version``."""

    version: str = ""
    build_number: int = 0
    commit: str = ""


_VERSION = re.compile(r"version:\s*(\S+)\s*\(build\s*(\d+),\s*commit\s*([0-9a-fA-F]+)\)")
_CPU_BACKEND = re.compile(r"loaded CPU backend from\s+(\S+)")
_DEVICE = re.compile(r"^\s+(\S+):\s+(.+?)\s+\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)")


def parse_version(text: str) -> BuildInfo:
    """Extract version, build number and commit from ``--version`` output."""
    m = _VERSION.search(text)
    if not m:
        return BuildInfo()
    return BuildInfo(version=m.group(1), build_number=int(m.group(2)), commit=m.group(3))


@dataclass
class Device:
    """A non-CPU device reported by ``--list-devices``."""

    name: str
    description: str
    total_mib: int
    free_mib: int


def parse_devices(text: str) -> list[Device]:
    """Parse ``--list-devices`` output; an empty list means CPU only."""
    devices = []
    for line in text.splitlines():
        m = _DEVICE.match(line)
        if m:
            devices.append(Device(m.group(1), m.group(2), int(m.group(3)), int(m.group(4))))
    return devices


_ISA_TIERS = {
    "x64": "no-avx", "sse42": "no-avx",
    "sandybridge": "avx", "ivybridge": "avx", "piledriver": "avx",
    "haswell": "avx2", "alderlake": "avx2",
    "skylakex": "avx512", "cannonlake": "avx512", "cascadelake": "avx512",
    "icelake": "avx512", "cooperlake": "avx512", "zen4": "avx512",
    "sapphirerapids": "avx512",
}


def parse_cpu_variant(text: str) -> tuple[str, str]:
    """Return the ggml CPU variant llama.cpp loaded and its ISA tier.

    The variant is read from the ``loaded CPU backend from ...`` log line,
    e.g. ``ggml-cpu-haswell.dll`` gives ``("haswell", "avx2")``.

    Returns:
        ``(variant, tier)``; both empty when the line is absent.
    """
    m = _CPU_BACKEND.search(text)
    if not m:
        return "", ""
    stem = PureWindowsPath(m.group(1)).stem
    stem = stem[3:] if stem.startswith("lib") else stem
    variant = stem.removeprefix("ggml-cpu").lstrip("-") or "native"
    if variant.startswith(("armv", "m1", "m2", "m3", "m4", "android")):
        return variant, "arm"
    return variant, _ISA_TIERS.get(variant, "unknown")


# ── capabilities of an installed build ───────────────────────────────


@dataclass
class Capabilities:
    """What one installed llama.cpp build supports."""

    build: BuildInfo = field(default_factory=BuildInfo)
    server: FlagRegistry | None = None
    bench: FlagRegistry | None = None
    devices: list[Device] = field(default_factory=list)
    cpu_variant: str = ""
    isa_tier: str = ""
    server_path: str = ""
    bench_path: str = ""

    def registry(self, tool: str) -> FlagRegistry | None:
        return self.server if tool == "server" else self.bench


def _run(cmd: list[str], timeout: int = 30) -> str:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            errors="replace", creationflags=_NO_WINDOW, env=clean_env(),
        )
        return (proc.stdout or "") + "\n" + (proc.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _cache_path(paths: list[str]) -> Path:
    key = hashlib.sha1()
    for p in paths:
        try:
            st = os.stat(p)
            key.update(f"{os.path.abspath(p)}|{st.st_size}|{st.st_mtime_ns}".encode())
        except OSError:
            key.update(p.encode())
    return cache_dir() / f"compat-{key.hexdigest()[:16]}.json"


def detect_capabilities(use_cache: bool = True) -> Capabilities:
    """Probe the installed llama.cpp build.

    Flag registries and build info are cached per binary (path, size and
    modification time), so a llama.cpp upgrade is detected automatically.
    Devices are probed every time because drivers and free VRAM change.

    Args:
        use_cache: Read and write the on-disk cache.

    Returns:
        The detected :class:`Capabilities`; fields stay empty when a
        binary is missing or a probe fails.
    """
    caps = Capabilities(
        server_path=find_llama_binary("llama-server"),
        bench_path=find_llama_binary("llama-bench"),
    )
    cache = _cache_path([caps.server_path, caps.bench_path])
    cached: dict[str, Any] = {}
    if use_cache and cache.is_file():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = {}

    try:
        caps.build = BuildInfo(**cached["build"])
        caps.server = FlagRegistry.from_json(cached["server"])
        caps.bench = FlagRegistry.from_json(cached["bench"])
        caps.cpu_variant, caps.isa_tier = cached.get("cpu_variant", ""), cached.get("isa_tier", "")
    except (KeyError, TypeError):
        cached = {}
    if not cached:
        version_text = _run([caps.bench_path, "--version"]) or _run([caps.server_path, "--version"])
        caps.build = parse_version(version_text)
        caps.cpu_variant, caps.isa_tier = parse_cpu_variant(version_text)
        server_help = _run([caps.server_path, "--help"])
        bench_help = _run([caps.bench_path, "--help"])
        caps.server = parse_help(server_help) if server_help.strip() else None
        caps.bench = parse_help(bench_help) if bench_help.strip() else None
        complete = caps.build.build_number and caps.server and caps.server.specs and caps.bench and caps.bench.specs
        if use_cache and complete:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({
                    "build": caps.build.__dict__,
                    "server": caps.server.to_json() if caps.server else None,
                    "bench": caps.bench.to_json() if caps.bench else None,
                    "cpu_variant": caps.cpu_variant,
                    "isa_tier": caps.isa_tier,
                }), encoding="utf-8")
            except OSError:
                pass

    caps.devices = parse_devices(_run([caps.bench_path, "--list-devices"]))
    return caps


# ── model architectures a build can load ─────────────────────────────

# llama.cpp keeps every architecture name it can load as a C string in its
# library (the LLM_ARCH_NAMES table), so a byte search answers "can this
# build load a <arch> model?" without loading anything. "llama" is in every
# build: when it is missing the file is not a llama.cpp library we understand.
_LIBRARY_NAMES = ("llama.dll", "libllama.so", "libllama.dylib")
_SENTINEL_ARCH = "llama"


def _library_candidates(caps: Capabilities) -> list[Path]:
    found: list[Path] = []
    for binary in (caps.bench_path, caps.server_path):
        if not binary or not os.path.isfile(binary):
            continue
        folder = Path(binary).resolve().parent
        for name in _LIBRARY_NAMES:
            for path in [folder / name, *sorted(folder.glob(name + ".*"))]:
                if path.is_file() and path not in found:
                    found.append(path)
        found.append(Path(binary).resolve())  # static builds carry the table in the executable
    return found


def architecture_supported(caps: Capabilities, arch: str) -> bool | None:
    """Tell whether the installed llama.cpp can load models of ``arch``.

    Args:
        caps: Detected capabilities (only the binary paths are used).
        arch: The model's ``general.architecture``, e.g. ``"lfm3"``.

    Returns:
        ``True`` or ``False`` when the build's architecture table was found,
        ``None`` when it could not be checked (then the first run tells).
    """
    if not arch:
        return None
    needle = b"\0" + arch.encode() + b"\0"
    sentinel = b"\0" + _SENTINEL_ARCH.encode() + b"\0"
    table_seen = False
    for path in _library_candidates(caps):
        try:  # mapped, not read: a GPU build's library runs to hundreds of megabytes
            with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
                if data.find(needle) != -1:
                    return True
                table_seen = table_seen or data.find(sentinel) != -1
        except (OSError, ValueError):  # unreadable, or empty (cannot be mapped)
            continue
    return False if table_seen else None


def unknown_architecture(text: str) -> str:
    """Return the architecture named in llama.cpp's load error, or ``""``."""
    m = re.search(r"unknown model architecture:\s*'([^']+)'", text)
    return m.group(1) if m else ""


def update_llama_advice(arch: str, build: int) -> str:
    """Plain-words advice when the installed llama.cpp is too old for a model."""
    which = f"Your llama.cpp (b{build})" if build else "Your llama.cpp"
    zip_hint = " (for a PC without a GPU: the file ending in bin-win-cpu-x64.zip)" if os.name == "nt" else ""
    return (f"{which} cannot load '{arch}' models: this model type is newer than your llama.cpp. "
            f"Download the newest release from https://github.com/ggml-org/llama.cpp/releases{zip_hint}, "
            f"unzip it into your llama.cpp folder, and run the command again.")


# ── rendering a config into arguments ────────────────────────────────


@dataclass
class RenderResult:
    """Arguments for one llama.cpp tool plus what could not be expressed."""

    args: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


_Emit = Callable[[Any, "FlagRegistry | None"], "list[str] | None"]


def _value(flag: str, transform: Callable[[Any], str] = str) -> _Emit:
    def emit(v, reg):
        spec = reg.get(flag) if reg else None
        if reg and (spec is None or not spec.takes_value):
            return None
        return [flag, transform(v)]
    return emit


def _bool_value(flag: str, true: str, false: str) -> _Emit:
    def emit(v, reg):
        spec = reg.get(flag) if reg else None
        if reg:
            if spec is None or not spec.takes_value:
                return None
            if spec.choices and true not in spec.choices:
                return None
        return [flag, true if v else false]
    return emit


def _switch(flag: str) -> _Emit:
    """A flag that is present for True and absent for False."""
    def emit(v, reg):
        spec = reg.get(flag) if reg else None
        if reg and (spec is None or spec.takes_value):
            return None
        return [flag] if v else []
    return emit


def _switch_pair(on: str, off: str) -> _Emit:
    def emit(v, reg):
        flag = on if v else off
        if reg and (not reg.has(flag) or reg.get(flag).takes_value):
            return None
        return [flag]
    return emit


def _legacy_load_server(v, reg):
    mode = getattr(v, "value", v)
    if mode in ("auto", "mmap"):
        return []
    wanted = {"none": "--no-mmap", "mlock": "--mlock", "mmap+mlock": "--mlock", "dio": "--direct-io"}.get(mode)
    if wanted and (reg is None or reg.has(wanted)):
        return [wanted]
    return None


def _legacy_load_bench(v, reg):
    mode = getattr(v, "value", v)
    if mode in ("none", "mmap") and (reg is None or reg.has("-mmp")):
        return ["-mmp", "0" if mode == "none" else "1"]
    if mode == "auto":
        return []
    return None


def _priority_server(v, reg):
    """``--prio`` for generation and ``--prio-batch`` for prompt threads."""
    if reg is not None and not reg.has("--prio"):
        return None
    args = ["--prio", str(v)]
    if reg is None or reg.has("--prio-batch"):
        args += ["--prio-batch", str(v)]
    return args


def _enum_value(v) -> str:
    return getattr(v, "value", str(v))


# For each setting, the spellings to try in order: current llama.cpp first,
# then older forms. The first one the build accepts wins.
_SPELLINGS: dict[str, dict[str, list[_Emit]]] = {
    "server": {
        "threads": [_value("-t")],
        "threads_batch": [_value("-tb")],
        "ctx_size": [_value("-c")],
        "batch_size": [_value("-b")],
        "ubatch_size": [_value("-ub")],
        "n_gpu_layers": [_value("-ngl")],
        "flash_attn": [_bool_value("-fa", "on", "off"), _bool_value("-fa", "1", "0"), _switch("-fa")],
        "tensor_split": [_value("-ts")],
        "split_mode": [_value("-sm", _enum_value)],
        "main_gpu": [_value("-mg")],
        "cache_type_k": [_value("-ctk")],
        "cache_type_v": [_value("-ctv")],
        "load_mode": [_value("-lm", _enum_value), _legacy_load_server],
        "repack": [_switch_pair("--repack", "--no-repack")],
        "numa": [_value("--numa")],
        "cpu_affinity": [_value("-C")],
        "parallel": [_value("-np")],
        "no_kv_offload": [_switch("-nkvo"), _bool_value("-nkvo", "1", "0")],
        "n_cpu_moe": [_value("-ncmoe")],
        "cache_ram_mib": [_value("-cram")],
        "ctx_checkpoints": [_value("-ctxcp")],
        "no_mmproj": [_switch("--no-mmproj")],
        "spec_type": [_value("--spec-type")],
        "priority": [_priority_server],
        "mmproj": [_value("--mmproj"), _value("-mm")],
    },
    "bench": {
        "threads": [_value("-t")],
        "batch_size": [_value("-b")],
        "ubatch_size": [_value("-ub")],
        "n_gpu_layers": [_value("-ngl")],
        "flash_attn": [_bool_value("-fa", "on", "off"), _bool_value("-fa", "1", "0")],
        "tensor_split": [_value("-ts", lambda v: str(v).replace(",", "/"))],
        "split_mode": [_value("-sm", _enum_value)],
        "main_gpu": [_value("-mg")],
        "cache_type_k": [_value("-ctk")],
        "cache_type_v": [_value("-ctv")],
        "load_mode": [_value("-lm", _enum_value), _legacy_load_bench],
        "priority": [_value("--prio")],
        "repack": [_bool_value("--repack", "1", "0")],
        "numa": [_value("--numa")],
        "cpu_affinity": [_value("-C")],
        "no_kv_offload": [_bool_value("-nkvo", "1", "0")],
        "n_cpu_moe": [_value("-ncmoe")],
    },
}

# Settings that have no meaning for a tool (not an incompatibility).
_NOT_APPLICABLE = {
    "bench": {"ctx_size", "parallel", "cache_ram_mib", "ctx_checkpoints", "no_mmproj", "spec_type", "mmproj",
              "threads_batch"},
    "server": set(),
}


def render_args(config: "SearchConfig", tool: str, registry: FlagRegistry | None = None) -> RenderResult:
    """Render a config as arguments for ``llama-server`` or ``llama-bench``.

    Args:
        config: The settings to express.
        tool: ``"server"`` or ``"bench"``.
        registry: Flags the installed build accepts. ``None`` renders the
            current llama.cpp spelling without checking a build.

    Returns:
        The arguments, plus a list of settings that were skipped and why.
    """
    if tool not in _TOOLS:
        raise ValueError(f"unknown tool {tool!r}")
    result = RenderResult()
    for name, value in config.model_dump(exclude_none=True).items():
        if name in _NOT_APPLICABLE[tool]:
            continue
        spellings = _SPELLINGS[tool].get(name)
        if not spellings:
            result.skipped.append(f"{name}: not supported by llama-{tool}")
            continue
        raw = getattr(config, name)
        for emit in spellings:
            args = emit(raw, registry)
            if args is not None:
                result.args.extend(args)
                break
        else:
            result.skipped.append(f"{name}={_enum_value(raw)}: this llama.cpp build has no matching flag")
    return result


# ── launch profiles from other people ────────────────────────────────

#: The llama-server options llama-autotune itself writes into a profile. A
#: profile file may come from anywhere; anything else in it (a network
#: address, a file to write, a model to download) is refused.
PROFILE_FLAGS = frozenset({
    "-t", "-tb", "-c", "-b", "-ub", "-ngl", "-fa", "-ts", "-sm", "-mg", "-ctk", "-ctv", "-lm", "--no-mmap",
    "--mlock", "--direct-io", "--repack", "--no-repack", "--numa", "-C", "-np", "-nkvo", "-ncmoe", "-cram",
    "-ctxcp", "--no-mmproj", "--spec-type", "--prio", "--prio-batch",
})
_FLAG_WORD = re.compile(r"^-{1,2}[A-Za-z]")


def foreign_profile_flags(args: list[str], registry: FlagRegistry | None = None) -> list[str]:
    """Options in a profile's arguments that llama-autotune would never write."""
    foreign = []
    for word in args:
        if not _FLAG_WORD.match(word):
            continue
        spec = registry.get(word) if registry else None
        if not PROFILE_FLAGS.intersection(spec.names if spec else [word]):
            foreign.append(word)
    return foreign


# ── environment hygiene ──────────────────────────────────────────────


def llama_env_overrides() -> dict[str, str]:
    """Return ``LLAMA_ARG_*`` environment variables that would change results."""
    return {k: v for k, v in os.environ.items() if k.upper().startswith("LLAMA_ARG_")}


def clean_env() -> dict[str, str]:
    """Return a copy of the environment without ``LLAMA_ARG_*`` variables."""
    return {k: v for k, v in os.environ.items() if not k.upper().startswith("LLAMA_ARG_")}


def config_ini_paths() -> list[Path]:
    """Return llama.cpp default-parameter files found on this machine."""
    found = []
    for var in ("APPDATA", "PROGRAMDATA"):
        base = os.environ.get(var)
        if base:
            p = Path(base) / "llama.cpp" / "config.ini"
            if p.is_file():
                found.append(p)
    xdg = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "llama.cpp" / "config.ini"
    if xdg.is_file():
        found.append(xdg)
    return found
