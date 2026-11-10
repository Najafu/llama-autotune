"""CLI interface for llama-autotune.

Commands: ``doctor`` (system and llama.cpp check), ``inspect`` (model and
memory plan), ``tune`` (measured CPU tuning), ``launch``, ``benchmark``,
``search``, ``export`` and ``import``. Memory is always planned against
the RAM available now, and llama.cpp arguments always come from the
compatibility layer.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import shlex
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .benchmark import run_benchmark
from .compat import (
    NOT_FOUND_ADVICE, Capabilities, architecture_supported, binary_name, clean_env, config_ini_paths,
    detect_capabilities, foreign_profile_flags, llama_env_overrides, llama_found, render_args, save_llama_dir,
    update_llama_advice,
)
from .constraints import GateDecision, estimate_config_memory, memory_budget, plan_memory, safety_margin
from .database import (
    TunedSettings, adopt_hardware_id, best_bandwidth, get_model_args, get_session, get_tuned, list_tuned,
    save_benchmark, save_model_args, save_tuned,
)
from .extra_args import ExtraArgsError, apply_over, describe, is_secret, merge, parse_extra_args, setting_key
from .hardware import detect_hardware, hardware_fingerprint, legacy_hardware_fingerprint
from .heuristics import apply_launch_defaults, generate_initial_config
from .memory import MIB, compute_model_for, estimate_memory, kv_cache_bytes, tensor_sizes_of
from .model_inspector import GGUFError, inspect_model
from .models import Backend, BenchmarkEntry, HardwareInfo, ModelInfo, OptimizeObjective, SearchConfig
from .optimizer import Optimizer
from .profiles import create_profile, export_profile, import_profile
from .resources import MemoryStatus, measure_bandwidth, memory_status, power_status
from .tuner import VERDICT_REASONS, Tuner, TuneReport, verdict
from .vision import find_vision_file, is_vision_file, vision_bytes

#: Smallest context worth starting a chat with. Also keeps out 0, which
#: llama-server reads as "the model's full context": the one size the memory
#: plan would never have agreed to.
MIN_CONTEXT = 256

EXAMPLES = (
    "[bold]Examples:[/bold]\n\n"
    "  llama-autotune doctor\n\n"
    "  llama-autotune inspect model.gguf\n\n"
    "  llama-autotune tune model.gguf\n\n"
    "  llama-autotune launch model.gguf\n\n"
    "  llama-autotune benchmark model.gguf -r 3\n\n"
    "  llama-autotune search model.gguf --objective max_generation_tps --profile best.json\n\n"
    "  llama-autotune import my_profile.json"
)

console = Console()
app = typer.Typer(
    name="llama-autotune",
    help="Benchmarking and optimization tool for llama.cpp",
    epilog=EXAMPLES,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
logger = logging.getLogger(__name__)
_FLAG = re.compile(r"^-{1,2}[A-Za-z]")


def _model_path(path: str) -> str:
    """The model file as an absolute path, or stop: it is stored and shown to the user later."""
    if not os.path.isfile(path):
        console.print(f"[red]Error: model file not found: {escape(path)}[/red]")
        raise typer.Exit(code=1)
    return os.path.abspath(path)


def _inspect(path: str) -> ModelInfo:
    """Read the model's metadata, or stop with plain words when the file is damaged."""
    try:
        return inspect_model(path)
    except (OSError, GGUFError) as e:
        console.print(f"[red]Cannot read this model: {escape(str(e))}.[/red] Download it again if it was cut short.")
        raise typer.Exit(code=1)


def _hardware_id(conn, hw: HardwareInfo) -> str:
    """This PC's id; results stored under the id of v0.5.0 and earlier move to it."""
    current = hardware_fingerprint(hw)
    adopt_hardware_id(conn, legacy_hardware_fingerprint(hw), current)
    return current


def _quote(word: str) -> str:
    """One command-line word, quoted so it can be pasted into the user's shell."""
    if os.name != "nt":
        return shlex.quote(word)
    if any(ch in word for ch in "&()^%!,;=<>|"):  # list2cmdline only quotes spaces
        return f'"{word}"'
    return subprocess.list2cmdline([word])


def _print_command(words: list[str], indent: str = "  ") -> None:
    """Print a command exactly: no markup, no line breaks inserted where the window ends."""
    console.print(indent + " ".join(_quote(w) for w in words), markup=False, highlight=False, soft_wrap=True)


def _require_llama(caps: Capabilities) -> None:
    """Stop with plain setup advice when llama.cpp cannot be found."""
    if not (llama_found(caps.server_path) and llama_found(caps.bench_path)):
        console.print(f"[red]{NOT_FOUND_ADVICE}[/red]")
        raise typer.Exit(code=1)


def _loadable(caps: Capabilities, model_info: ModelInfo) -> bool:
    """Print plain advice and return False when llama.cpp is too old for this model."""
    if architecture_supported(caps, model_info.architecture) is False:
        console.print(f"[red]{update_llama_advice(model_info.architecture, caps.build.build_number)}[/red]")
        return False
    return True


def _mib(n: int | float) -> str:
    return f"{round(n / MIB):,} MiB"


def _table(title: str | None = None) -> Table:
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    return table


def _hardware_table(hw: HardwareInfo, mem: MemoryStatus) -> Table:
    table = _table("Hardware")
    table.add_row("CPU", hw.cpu_name)
    cores = f"{hw.physical_cores} cores / {hw.logical_cores} threads"
    if hw.efficiency_cores:
        cores += f" ({hw.performance_cores} performance + {hw.efficiency_cores} efficiency)"
    if hw.numa_nodes > 1:
        cores += f", {hw.numa_nodes} NUMA nodes (multi-socket)"
    table.add_row("Cores", cores)
    if hw.cpu_variant:
        table.add_row("CPU instructions", f"{hw.isa_tier} (llama.cpp uses ggml-cpu-{hw.cpu_variant})")
    table.add_row("RAM", f"{hw.ram_gb} GB total, {_mib(mem.available)} free now")
    table.add_row("GPU", ", ".join(hw.gpu_models) or "none usable by llama.cpp")
    table.add_row("Backend", hw.backend.value)
    if hw.vram_per_gpu:
        table.add_row("VRAM", ", ".join(f"{v} GB" for v in hw.vram_per_gpu))
    return table


def _model_table(m: ModelInfo) -> Table:
    table = _table("Model")
    table.add_row("Path", escape(m.path))
    table.add_row("Architecture", m.architecture)
    table.add_row("Parameters", f"{m.parameters:,}")
    if m.is_moe:
        table.add_row("Active Params", f"{m.active_parameters:,}")
    quant = m.quantization
    if m.quantization_declared and not m.quantization_declared.startswith(m.quantization):
        quant += f" (file metadata says {m.quantization_declared}; tensors say {m.quantization})"
    table.add_row("Quantization", quant)
    table.add_row("Layers", str(m.n_layers))
    table.add_row("Heads", str(m.n_heads))
    table.add_row("Context Length", f"{m.training_context:,}")
    table.add_row("MoE", "Yes" if m.is_moe else "No")
    table.add_row("File Size", f"{m.file_size_gb} GB")
    table.add_row("Read per token", f"{(m.active_bytes_per_token or m.bytes_per_token) / 1e9:.2f} GB")
    if m.kv_layout is not None:
        table.add_row("KV cache", f"{_mib(kv_cache_bytes(m.kv_layout, 1024))} per 1,024 tokens (f16)")
    return table


def _memory_plan_table(model: ModelInfo, hw: HardwareInfo, mem: MemoryStatus, sizes: dict,
                       server_build: int = 0) -> Table:
    """What llama-server needs at each context with its defaults (flash attention auto)."""
    budget = memory_budget(mem.available, mem.total)
    table = Table(title=f"Memory plan (free now {_mib(mem.available)}, "
                        f"keeping {_mib(safety_margin(mem.total))} spare)", box=box.ROUNDED)
    table.add_column("Context", justify="right")
    table.add_column("Needs (defaults)", justify="right")
    table.add_column("Verdict")
    contexts = [c for c in (2048, 4096, 8192, 16384, 32768) if not model.training_context or c <= model.training_context]
    if model.training_context and model.training_context not in contexts:
        contexts.append(model.training_context)
    tier = hw.isa_tier or "avx2"
    cm = compute_model_for(model.path, model, server_build=server_build)
    for ctx in contexts:
        compute = cm.bytes(ctx) if cm else None
        default = estimate_memory(model, ctx, isa_tier=tier, tensor_sizes=sizes, compute_override=compute)
        if default.total <= budget:
            fit = "[green]Fits[/green]"
        else:
            plan = plan_memory(model, budget, isa_tier=tier, ctx=ctx, min_ctx=ctx, tensor_sizes=sizes,
                               compute_model=cm)
            if plan.fits:
                fit = f"[yellow]Fits with: {', '.join(s.split(' (')[0] for s in plan.savings)}[/yellow]"
            else:
                fit = f"[red]Won't fit (short by {_mib(plan.shortfall)})[/red]"
        label = f"{ctx:,}" + (" (full)" if ctx == model.training_context else "")
        table.add_row(label, _mib(default.total), fit)
    return table


@app.command(help="Tell llama-autotune where llama.cpp is: the folder you unzipped it to.")
def setup(
    folder: str = typer.Argument(..., help="Folder that contains llama-server and llama-bench"),
):
    """Save the llama.cpp folder so no environment variable is needed."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        console.print(f"[red]Folder not found: {escape(str(root))}[/red]")
        raise typer.Exit(code=1)
    server = binary_name("llama-server")
    found = root if (root / server).is_file() else next(
        (p.parent for p in sorted(root.glob(f"**/{server}"), key=lambda p: len(p.parts))), None)
    if found is None or not (found / binary_name("llama-bench")).is_file():
        console.print(f"[red]No {server} and {binary_name('llama-bench')} in {escape(str(root))} or its "
                      f"subfolders.[/red] "
                      "Unzip the llama.cpp release there first.")
        raise typer.Exit(code=1)
    save_llama_dir(str(found))
    env = os.environ.get("LLAMA_CPP_DIR")
    if env and Path(env).resolve() != found.resolve():
        console.print(f"[yellow]LLAMA_CPP_DIR is set to {escape(env)} and takes precedence. Remove it, or run "
                      f"setx LLAMA_CPP_DIR \"{escape(str(found))}\" and open a new terminal.[/yellow]")
    caps = detect_capabilities()
    build = f"b{caps.build.build_number}" if caps.build.build_number else "unknown build"
    console.print(f"[green]Saved.[/green] llama.cpp {build} in {escape(str(found))}. Next: llama-autotune doctor")


@app.command(help="Check this PC and the installed llama.cpp build.")
def doctor(
    bandwidth: bool = typer.Option(True, "--bandwidth/--no-bandwidth", help="Measure memory bandwidth (about 1 s)"),
):
    """Show what llama-autotune detects, with every source, for bug reports."""
    caps = detect_capabilities()
    hw = detect_hardware(caps)
    mem = memory_status()
    power = power_status()
    console.print(_hardware_table(hw, mem))

    sys_table = _table("System")
    sys_table.add_row("Power", "battery" if power.on_battery else ("mains (laptop)" if power.has_battery else "mains"))
    if power.battery_saver:
        sys_table.add_row("Battery saver", "[yellow]on: speeds will be lower[/yellow]")
    if mem.commit_limit:
        sys_table.add_row("Commit headroom", _mib(mem.commit_headroom))
    if bandwidth:
        threads = min(hw.performance_cores or hw.physical_cores or 1, 16)
        sys_table.add_row("Memory bandwidth", f"{measure_bandwidth(threads=threads)} GB/s")
    console.print(sys_table)

    llama = _table("llama.cpp")
    llama.add_row("llama-server", escape(caps.server_path))
    llama.add_row("llama-bench", escape(caps.bench_path))
    if caps.build.build_number:
        llama.add_row("Build", f"b{caps.build.build_number} ({caps.build.version}, commit {caps.build.commit})")
    else:
        llama.add_row("Build", "[red]not found[/red]")
    llama.add_row("Devices", ", ".join(f"{d.name} {d.description} ({d.total_mib} MiB)" for d in caps.devices)
                  or "CPU only")
    if caps.server:
        checks = {"--fit": "auto memory fitting", "-lm": "load modes", "--repack": "weight repacking",
                  "-cram": "prompt cache size", "--spec-type": "speculative decoding"}
        llama.add_row("Features", ", ".join(f"{name} {'yes' if caps.server.has(flag) else 'no'}"
                                            for flag, name in checks.items()))
    console.print(llama)
    if not llama_found(caps.server_path):
        console.print(f"[red]{NOT_FOUND_ADVICE}[/red]")

    warnings = []
    for key, value in llama_env_overrides().items():
        warnings.append(escape(f"{key}={value} is set and silently changes llama.cpp settings"))
    for path in config_ini_paths():
        warnings.append(escape(f"llama.cpp default-settings file found: {path}"))
    if power.on_battery:
        warnings.append("on battery: plug in before tuning for stable results")
    for w in warnings:
        console.print(f"[yellow]Warning:[/yellow] {w}")
    if not warnings:
        console.print("[green]No problems found.[/green]")


@app.command(help="Inspect hardware, model metadata and the memory plan.")
def inspect(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
):
    """Show hardware, model metadata and which context sizes fit in free RAM."""
    model = _model_path(model)
    caps = detect_capabilities()
    hw = detect_hardware(caps)
    mem = memory_status()
    model_info = _inspect(model)
    console.print(_hardware_table(hw, mem))
    console.print(_model_table(model_info))
    _loadable(caps, model_info)
    vision_file = find_vision_file(model, model_info)
    if vision_file:
        console.print(f"Vision file: {escape(vision_file.name)}. Images are optional (launch --images) and need "
                      f"{_mib(vision_bytes(vision_file))} on top of the plan below.")
    if hw.backend == Backend.CPU:
        console.print(_memory_plan_table(model_info, hw, mem, tensor_sizes_of(model), caps.build.build_number))


def _onoff(value: bool | None) -> str:
    return "auto" if value is None else ("on" if value else "off")


def _print_tune_report(rep: TuneReport, model: str) -> None:
    if rep.status == "wont_fit":
        return
    if rep.status == "prediction_only":
        console.print(f"[bold]Verdict: {rep.verdict}[/bold] (predicted, not measured): about "
                      f"{rep.predicted_tps:.2f} tokens per second; {VERDICT_REASONS[rep.verdict]}.")
        for note in rep.notes:
            console.print(f"  {note}")
        return
    if rep.status == "unsupported":
        console.print(f"[red]{rep.error}[/red]")
        return
    if rep.status != "tuned" or rep.config is None:
        console.print(f"[red]Tuning failed[/red] ({rep.error or 'no results'}).")
        for w in rep.warnings:
            console.print(f"[yellow]Warning:[/yellow] {w}")
        return

    colour = {"Fast": "green", "Usable": "yellow", "Slow": "red"}.get(rep.verdict, "white")
    console.print(f"\n[bold {colour}]Verdict: {rep.verdict}[/bold {colour}]: about {rep.best_tps:.1f} tokens per "
                  f"second; {VERDICT_REASONS[rep.verdict]}.")

    cfg, base = rep.config, rep.baseline
    table = Table(title="Settings", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("llama.cpp default")
    table.add_column("Tuned", style="green")
    table.add_row("Threads (writing)", str(base.threads), str(cfg.threads or base.threads))
    table.add_row("Threads (reading the prompt)", str(base.threads), str(cfg.threads_batch or cfg.threads or base.threads))
    table.add_row("Flash attention", _onoff(base.flash_attn), _onoff(cfg.flash_attn))
    if base.repack is not None or cfg.repack is not None:
        table.add_row("Weight repacking", _onoff(base.repack), _onoff(cfg.repack if cfg.repack is not None else base.repack))
    if cfg.ubatch_size:
        table.add_row("Prompt batch (ubatch)", "512", str(cfg.ubatch_size))
    if cfg.numa:
        table.add_row("NUMA", "off", cfg.numa)
    if cfg.priority:
        table.add_row("Priority", "normal", "above normal")
    console.print(table)

    speed = Table(title="Speed (tokens per second)", box=box.ROUNDED)
    speed.add_column("")
    speed.add_column("Default", justify="right")
    speed.add_column("Tuned", justify="right", style="green")
    speed.add_row("Writing (generation)", f"{rep.baseline_tps:.1f}", f"{rep.best_tps:.1f} ({rep.gain:+.0%})")
    if rep.baseline_pp and rep.best_pp:
        speed.add_row("Reading the prompt", f"{rep.baseline_pp:.1f}", f"{rep.best_pp:.1f}")
    if rep.depth_tps:
        speed.add_row(f"Writing after {rep.depth:,} tokens of chat", "", f"{rep.depth_tps:.1f}")
    console.print(speed)

    if rep.predicted_tps:
        console.print(f"[dim]Memory speed {rep.bandwidth_gbs} GB/s; predicted {rep.predicted_tps:.1f} tokens/s "
                      f"before measuring, this model reaches {rep.efficiency:.0%} of the bandwidth limit. "
                      f"{rep.tests} tests in {rep.elapsed / 60:.1f} min.[/dim]")
    for note in rep.notes:
        console.print(f"  {note}")
    for w in rep.warnings:
        console.print(f"[yellow]Warning:[/yellow] {w}")


@app.command(help="Find the fastest settings for a model on this PC by measuring (up to 10 minutes).")
def tune(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    ctx: int = typer.Option(4096, "-c", "--ctx-size", min=MIN_CONTEXT, help="Context size you want to chat with"),
    minutes: float = typer.Option(10.0, "--minutes", min=0.5, help="Time limit for the measurements"),
    output_profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Also save a launch profile file"),
):
    """Measure llama.cpp's defaults and better settings, then save the winner for ``launch``."""
    model = _model_path(model)
    caps = detect_capabilities()
    _require_llama(caps)
    hw = detect_hardware(caps)
    if hw.backend != Backend.CPU:
        console.print("[yellow]tune measures CPU settings. GPU tuning comes in a later version; "
                      "`launch` already lets llama.cpp fit the model to your GPU.[/yellow]")
        raise typer.Exit(code=1)
    _inspect(model)  # a damaged file stops here, in plain words
    with closing(get_session()) as conn:
        hardware_id = _hardware_id(conn, hw)
        known_bw = best_bandwidth(conn, hardware_id)
    tuner = Tuner(model, ctx=ctx, budget_s=minutes * 60, caps=caps, hw=hw, known_bandwidth=known_bw,
                  say=lambda msg: console.print(f"[dim]{msg}[/dim]"))
    if not _loadable(caps, tuner.model):
        raise typer.Exit(code=1)
    mem = memory_status()
    console.print(_hardware_table(hw, mem))
    gate = tuner.plan()
    _print_gate(gate, mem)
    if not gate.fits:
        raise typer.Exit(code=1)

    rep = tuner.run()
    _print_tune_report(rep, model)
    if rep.status != "tuned" or rep.config is None:
        raise typer.Exit(code=0 if rep.status == "prediction_only" else 1)

    tuned_cfg = rep.config.model_copy(update={"ctx_size": None})
    with closing(get_session()) as conn:
        save_tuned(conn, hardware_id, tuner.model.fingerprint, TunedSettings(
            config=tuned_cfg, llama_build=caps.build.build_number, generation_tps=rep.best_tps,
            prompt_tps=rep.best_pp, baseline_tps=rep.baseline_tps, ctx=gate.ctx, efficiency=rep.efficiency,
            bandwidth_gbs=rep.bandwidth_gbs, model_path=model,
            depth_tps=rep.depth_tps, depth=rep.depth, verdict=rep.verdict,
        ))

    if output_profile:
        launch_cfg = apply_launch_defaults(rep.config.model_copy(update={"ctx_size": gate.ctx}), hw, tuner.model)
        profile = create_profile(
            name=Path(output_profile).stem, args=render_args(launch_cfg, "server", caps.server).args,
            model_path=model, hardware=hw.cpu_name, score=rep.best_tps, llama_build=caps.build.build_number,
        )
        console.print(f"[green]Profile saved to {escape(export_profile(profile, output_profile))}[/green]")
    console.print("\n[green]Saved.[/green] Start chatting with these settings:")
    _print_command(["llama-autotune", "launch", model])


def _apply_tuned(config: SearchConfig, hw: HardwareInfo, model_info: ModelInfo, build: int) -> SearchConfig:
    """Overlay the settings saved by ``tune`` for this machine and model."""
    with closing(get_session()) as conn:
        tuned = get_tuned(conn, _hardware_id(conn, hw), model_info.fingerprint)
    if tuned is None:
        if hw.backend == Backend.CPU:
            console.print("[dim]Tip: run `llama-autotune tune` once to measure the fastest settings "
                          "for this model.[/dim]")
        return config
    # A setting tune left at None was measured at llama.cpp's own default and won,
    # so it replaces the heuristic guess too (e.g. flash attention "auto", not "off").
    keep = ("threads", "threads_batch", "flash_attn", "repack", "ubatch_size", "numa")
    update = {k: v for k, v in tuned.config.model_dump().items() if k in keep}
    when = tuned.timestamp[:10]
    console.print(f"Using settings measured by tune on {when} ({tuned.generation_tps:.1f} tokens/s).")
    if tuned.llama_build and build and tuned.llama_build != build:
        console.print(f"[yellow]They were measured on llama.cpp b{tuned.llama_build}; b{build} is installed "
                      "now. Run tune again to re-check.[/yellow]")
    return config.model_copy(update=update)


@app.command(help="Run a benchmark with optional custom parameters.")
def benchmark(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    threads: Optional[int] = typer.Option(None, "-t", "--threads", help="CPU threads"),
    batch_size: Optional[int] = typer.Option(None, "-b", "--batch-size", help="Batch size"),
    ubatch_size: Optional[int] = typer.Option(None, "-ub", "--ubatch-size", help="Ubatch size"),
    n_gpu_layers: Optional[int] = typer.Option(None, "-ngl", "--n-gpu-layers", help="GPU layers"),
    flash_attn: Optional[bool] = typer.Option(None, "--flash-attn/--no-flash-attn", help="Flash attention"),
    repack: Optional[bool] = typer.Option(None, "--repack/--no-repack", help="Weight repacking"),
    depth: int = typer.Option(0, "-d", "--depth", help="Tokens already in the context"),
    repetitions: int = typer.Option(3, "-r", "--repetitions", help="Benchmark repetitions"),
    output_file: Optional[str] = typer.Option(None, "-o", "--output", help="Save result to file"),
):
    """Run one benchmark, guarded by the RAM watchdog."""
    model = _model_path(model)
    caps = detect_capabilities()
    _require_llama(caps)
    hw = detect_hardware(caps)
    model_info = _inspect(model)
    if not _loadable(caps, model_info):
        raise typer.Exit(code=1)

    explicit = {"threads": threads, "batch_size": batch_size, "ubatch_size": ubatch_size,
                "n_gpu_layers": n_gpu_layers, "flash_attn": flash_attn, "repack": repack}
    if any(v is not None for v in explicit.values()):
        config = SearchConfig(**{k: v for k, v in explicit.items() if v is not None})
    else:
        config = generate_initial_config(hw, model_info)

    console.print("[bold]Running benchmark...[/bold]")
    result = run_benchmark(model, config, repetitions=repetitions, n_depth=depth, registry=caps.bench)

    table = Table(title="Benchmark Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Prompt TPS", f"{result.prompt_tps:.2f}")
    table.add_row("Generation TPS", f"{result.generation_tps:.2f}")
    table.add_row("Run Time", f"{result.startup_time:.2f}s")
    table.add_row("Peak Memory", f"{result.memory_usage:,.0f} MB")
    table.add_row("Lowest Free RAM", f"{result.min_available_mb:,.0f} MB")
    table.add_row("Success", "Yes" if result.success else f"No ({result.error})")
    console.print(table)
    if result.error.startswith("memory_guard"):
        console.print(escape((result.raw_output.splitlines() or [result.error])[0]), style="yellow")

    with closing(get_session()) as conn:
        save_benchmark(conn, BenchmarkEntry(
            hardware_id=_hardware_id(conn, hw), model_id=model_info.fingerprint, config=config, result=result,
            llama_build=caps.build.build_number, model_path=model))

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
        console.print(f"[green]Saved to {escape(output_file)}[/green]")
    if not result.success:
        raise typer.Exit(code=1)


@app.command(help="Advanced: search many settings with Optuna. Experimental on GPUs; on a CPU use tune.")
def search(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    objective: str = typer.Option("balanced", "--objective", "-O", help="Optimization objective"),
    trials_b: int = typer.Option(15, "--trials-b", help="Stage B trials"),
    trials_c: int = typer.Option(40, "--trials-c", help="Stage C trials"),
    output_profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Save best profile to file"),
):
    """Find the fastest configuration that fits in free memory."""
    model = _model_path(model)
    try:
        obj = OptimizeObjective(objective)
    except ValueError:
        valid = [o.value for o in OptimizeObjective]
        console.print(f"[red]Invalid objective. Choose from: {', '.join(valid)}[/red]")
        raise typer.Exit(code=1)

    if obj == OptimizeObjective.MAX_CONTEXT:
        console.print("[red]max_context cannot be searched: llama-bench does not measure a context size.[/red] "
                      "For the largest context that fits in free RAM, run: llama-autotune launch <model> -c max")
        raise typer.Exit(code=1)
    _inspect(model)  # a damaged file stops here, in plain words
    opt = Optimizer(model_path=model, objective=obj, n_trials_stage_b=trials_b, n_trials_stage_c=trials_c)
    _require_llama(opt.caps)
    if not _loadable(opt.caps, opt.model):
        raise typer.Exit(code=1)
    mem = memory_status()
    console.print(_hardware_table(opt.hw, mem))
    if opt.gate is not None:
        _print_gate(opt.gate, mem)
        if not opt.gate.fits:
            raise typer.Exit(code=1)

    console.print("[bold]Running optimization...[/bold]")
    best_config = opt.run()
    result = opt.measure(best_config)

    table = Table(title=f"Best Config ({opt.objective.value})", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Prompt TPS", f"{result.prompt_tps:.2f}" if result.success else "N/A")
    table.add_row("Generation TPS", f"{result.generation_tps:.2f}" if result.success else "N/A")
    table.add_row("Score", f"{opt.best_score:.2f}")
    table.add_row("Evaluations", str(opt.total_evals))
    for field, val in best_config.model_dump(exclude_none=True).items():
        table.add_row(field.replace("_", "-"), str(getattr(val, "value", val)))
    console.print(table)

    if output_profile:
        server_args = render_args(apply_launch_defaults(best_config, opt.hw, opt.model), "server", opt.caps.server)
        profile = create_profile(
            name=f"{opt.hw.cpu_name.replace(' ', '_')}_{opt.model.architecture}_{opt.objective.value}",
            args=server_args.args, model_path=model, hardware=opt.hw.cpu_name,
            score=opt.best_score, llama_build=opt.caps.build.build_number,
        )
        console.print(f"[green]Profile saved to {escape(export_profile(profile, output_profile))}[/green]")

    if opt.total_evals > 0:
        with closing(get_session()) as conn:
            save_benchmark(conn, BenchmarkEntry(
                hardware_id=_hardware_id(conn, opt.hw), model_id=opt.model.fingerprint,
                config=best_config, result=result, objective=obj,
                llama_build=opt.caps.build.build_number, model_path=model))


# Extra options that change memory use; the plan does not see them.
_MEMORY_FLAGS = {"-c", "-b", "-ub", "-ctk", "-ctv", "-fa", "--repack", "-np", "-cram", "-ngl", "--mlock"}


def _vision_for(model: str, model_info: ModelInfo, wanted: bool, explicit: str | None) -> Path | None:
    """The vision file to try: the one given, or the matching one next to the model."""
    if explicit:
        if not os.path.isfile(explicit) or not is_vision_file(explicit):
            console.print(f"[red]Not a vision file (mmproj GGUF): {escape(explicit)}[/red]")
            raise typer.Exit(code=1)
        return Path(explicit)
    found = find_vision_file(model, model_info)
    if wanted and found is None:
        console.print(f"[yellow]No vision file for this model in {escape(str(Path(model).resolve().parent))}: "
                      f"images stay off. "
                      f"Download the mmproj GGUF that belongs to this model, or name it with --mmproj.[/yellow]")
    elif not wanted and found is not None:
        console.print(f"[dim]Tip: {escape(found.name)} lets this model read images; add --images to use it.[/dim]")
    return found if wanted else None


def _plan_with_optional_vision(plan, limit: int, want: int, want_max: bool,
                               vision_file: Path | None) -> tuple[GateDecision, Path | None]:
    """Fit the model with images when possible; images are optional, context is not.

    With a fixed context, images are kept only when the model fits at that
    context with them. With ``-c max``, both largest contexts are reported.
    Otherwise the vision file is left out and the text chat starts.
    """
    def text_only() -> GateDecision:
        return _largest_fitting_context(plan, limit) if want_max else plan(ctx=want, min_ctx=min(2048, want))

    if vision_file is None:
        return text_only(), None
    vb = vision_bytes(vision_file)
    if want_max:
        with_v = _largest_fitting_context(functools.partial(plan, vision=vb), limit)
        without = text_only()
        if with_v.fits:
            if without.fits and without.ctx > with_v.ctx:
                console.print(f"With images the largest context is {with_v.ctx:,} tokens; without images it "
                              f"would be {without.ctx:,}. Leave out --images for the longer context.")
            return with_v, vision_file
        if without.fits:
            console.print(f"[yellow]Images left out: the vision file needs {_mib(vb)} that is not free. "
                          f"The text chat starts without it.[/yellow]")
        return without, None
    with_v = plan(ctx=want, min_ctx=want, vision=vb)
    if with_v.fits:
        return with_v, vision_file
    without = text_only()
    if without.fits:
        console.print(f"[yellow]Images left out: at {want:,} tokens of context the vision file ({_mib(vb)}) does "
                      f"not fit as well (short by {_mib(with_v.shortfall)}). The text chat starts without it; a "
                      f"smaller -c can make room for images.[/yellow]")
    return without, None


def _value_of(args: list[str], *flags: str) -> str:
    """The word after the first of ``flags`` in an argument list (``""`` when absent)."""
    for flag in flags:
        if flag in args and args.index(flag) + 1 < len(args):
            return args[args.index(flag) + 1]
    return ""


def _ctx_of(args: list[str]) -> int:
    """The context size in a llama-server argument list (0 when not set)."""
    value = _value_of(args, "-c", "--ctx-size")
    return int(value) if value.isdigit() else 0


def _load_profile(path: str, caps: Capabilities) -> list[str]:
    """The arguments of a profile file, after checking that they are safe to run.

    A profile can come from anyone. Only options llama-autotune itself writes
    are accepted, and only those the installed llama.cpp knows.
    """
    if not os.path.isfile(path):
        console.print(f"[red]Error: profile file not found: {escape(path)}[/red]")
        raise typer.Exit(code=1)
    try:
        prof = import_profile(path)
    except (OSError, ValueError) as e:
        console.print(f"[red]Cannot use this profile: {escape(str(e))}[/red]")
        raise typer.Exit(code=1)
    if prof.llama_build and caps.build.build_number and prof.llama_build != caps.build.build_number:
        console.print(f"[yellow]Profile was tuned on llama.cpp b{prof.llama_build}; "
                      f"installed is b{caps.build.build_number}. Consider running tune again.[/yellow]")
    foreign = foreign_profile_flags(prof.args, caps.server)
    if foreign:
        console.print(f"[red]This profile contains options llama-autotune never writes: "
                      f"{escape(' '.join(foreign))}.[/red] It was edited or comes from elsewhere. Remove them from "
                      f"the file, or give them yourself after the model name if you trust them.")
        raise typer.Exit(code=1)
    unknown = [a for a in prof.args if _FLAG.match(a) and caps.server and not caps.server.has(a)]
    if unknown:
        console.print(f"[red]This llama.cpp build does not accept: {escape(' '.join(unknown))}[/red]")
        raise typer.Exit(code=1)
    return list(prof.args)


def _gate_for_args(args: list[str], model: str, model_info: ModelInfo, hw: HardwareInfo, build: int,
                   mem: MemoryStatus) -> GateDecision:
    """Check a ready-made argument list (a profile) against the RAM free now.

    Nothing is changed to make it fit: a profile says exactly what to run.
    Without ``-c`` llama-server takes the model's full context, so that is
    what is counted.
    """
    ctx = _ctx_of(args) or model_info.training_context or 4096
    ubatch = int(v) if (v := _value_of(args, "-ub", "--ubatch-size")).isdigit() else 512
    cache = _value_of(args, "-ctv", "--cache-type-v") or "f16"
    repack = "--no-repack" not in args
    flash = True if cache not in ("f16", "f32", "bf16") else {"on": True, "off": False}.get(_value_of(args, "-fa"))
    cm = compute_model_for(model, model_info, build)
    budget = memory_budget(mem.available, mem.total)
    est = estimate_config_memory(model_info, ctx, cache, ubatch, repack, hw.isa_tier or "avx2",
                                 tensor_sizes_of(model), None,
                                 compute_bytes=cm.bytes(ctx, ubatch, flash) if cm else None)
    return GateDecision(est.total <= budget, ctx, est, budget, ubatch, repack, cache)


def _largest_fitting_context(plan, limit: int) -> GateDecision:
    """Binary search, in steps of 1,024 tokens, for the largest context the gate accepts."""
    best = plan(ctx=2048, min_ctx=2048)
    if not best.fits:
        return best
    lo, hi = 2, max(2, limit // 1024)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        gate = plan(ctx=mid * 1024, min_ctx=mid * 1024)
        if gate.fits and gate.ctx == mid * 1024:
            lo, best = mid, gate
        else:
            hi = mid - 1
    if limit % 1024 and limit > lo * 1024:
        gate = plan(ctx=limit, min_ctx=limit)
        if gate.fits:
            best = gate
    return best


def _print_gate(gate: GateDecision, mem: MemoryStatus) -> None:
    need, spare = _mib(gate.estimate.total), _mib(safety_margin(mem.total))
    if gate.fits:
        console.print(f"[green]Memory:[/green] needs {need} at {gate.ctx:,} tokens of context; "
                      f"{_mib(mem.available)} free, {spare} kept spare.")
        for s in gate.savings:
            console.print(f"  saving applied: {s}")
    else:
        console.print(f"[red]This model does not fit in the memory free right now.[/red] It needs {need} "
                      f"even at {gate.ctx:,} tokens; {_mib(mem.available)} is free and {spare} is kept spare "
                      f"so Windows stays responsive (short by {_mib(gate.shortfall)}).")
        console.print("  Close other programs to free memory, or choose a smaller model or quantization.")


@app.command(
    help="Start llama-server with a configuration that fits in free RAM. Any llama-server option can follow "
         "the model, e.g. --temp 0.1 --top-k 50 --repeat-penalty 1.1 (or temperature=0.1); "
         "they are remembered for this model.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def launch(
    extra: typer.Context,
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Path to profile JSON"),
    ctx: str = typer.Option("4096", "-c", "--ctx-size",
                            help="Wanted context size (reduced if it does not fit), or 'max' for the largest "
                                 "that fits in free RAM now"),
    images: bool = typer.Option(False, "--images",
                                help="Also read images, with the vision file (mmproj) found next to the model. "
                                     "Optional: left out when it does not fit, the text chat still starts"),
    mmproj: Optional[str] = typer.Option(None, "--mmproj", help="Vision file to use (implies --images)"),
    editing: bool = typer.Option(False, "--editing",
                                 help="Faster when rewriting or editing text you paste in (speculative decoding)"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", help="Server port"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command without starting the server"),
    forget_settings: bool = typer.Option(False, "--forget-settings",
                                         help="Forget the extra options remembered for this model"),
):
    """Plan memory, apply safe defaults, add the user's options and start llama-server."""
    model = _model_path(model)
    want_max = ctx.strip().lower() == "max"
    if not want_max and not (ctx.isdigit() and int(ctx) >= MIN_CONTEXT):
        console.print(f"[red]Context must be a number of tokens ({MIN_CONTEXT} or more) or 'max', "
                      f"not '{escape(ctx)}'.[/red]")
        raise typer.Exit(code=1)
    caps = detect_capabilities()
    _require_llama(caps)
    model_info = _inspect(model)
    if not _loadable(caps, model_info):
        raise typer.Exit(code=1)
    try:
        new_pairs = parse_extra_args(list(extra.args), caps.server)
    except ExtraArgsError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(code=1)
    with closing(get_session()) as conn:
        saved = [] if forget_settings else get_model_args(conn, model_info.fingerprint)
        pairs = merge(saved, new_pairs, caps.server)
        if forget_settings or new_pairs:
            # A password given once is used once: it is never written to the database.
            keep = [(f, v) for f, v in pairs if not is_secret(f, caps.server)]
            save_model_args(conn, model_info.fingerprint, keep, model)
    if forget_settings:
        console.print("Forgot the extra options remembered for this model.")
    if pairs:
        label = "Your options" if new_pairs else "Your remembered options"
        console.print(f"{label} for this model: {escape(describe(pairs, caps.server))}")
        if any(is_secret(f, caps.server) for f, _ in pairs):
            console.print("[dim]Secret values (such as --api-key) are not remembered; give them each time.[/dim]")
        unchecked = [f for f, _ in pairs if setting_key(f, caps.server) in _MEMORY_FLAGS]
        if unchecked:
            console.print(f"[yellow]Not covered by the memory plan: {' '.join(unchecked)}. If the PC slows "
                          f"down, run launch with --forget-settings.[/yellow]")
    hw = detect_hardware(caps)
    if profile:
        args = _load_profile(profile, caps)
        if hw.backend == Backend.CPU:
            mem = memory_status()
            gate = _gate_for_args(args, model, model_info, hw, caps.build.build_number, mem)
            _print_gate(gate, mem)
            if not gate.fits:
                raise typer.Exit(code=1)
    else:
        vision_file = _vision_for(model, model_info, images or bool(mmproj), mmproj)
        limit = model_info.training_context or 131072
        want = limit if want_max else min(int(ctx), limit)
        config = _apply_tuned(generate_initial_config(hw, model_info), hw, model_info,
                              caps.build.build_number)
        if hw.backend == Backend.CPU:
            mem = memory_status()
            plan = functools.partial(
                plan_memory, model_info, memory_budget(mem.available, mem.total),
                isa_tier=hw.isa_tier or "avx2", ubatch=config.ubatch_size or 512,
                tensor_sizes=tensor_sizes_of(model),
                compute_model=compute_model_for(model, model_info, caps.build.build_number),
                flash_attn=config.flash_attn)
            gate, vision_file = _plan_with_optional_vision(plan, limit, want, want_max, vision_file)
            if want_max and gate.fits:
                console.print(f"Largest context that fits now: {gate.ctx:,} tokens "
                              f"(the model supports {limit:,}).")
            _print_gate(gate, mem)
            if not gate.fits:
                raise typer.Exit(code=1)
            config = gate.apply(config)
        else:
            if want_max:
                console.print("[yellow]-c max plans system RAM only; on a GPU the model's full context is "
                              "requested and llama.cpp fits it to the GPU.[/yellow]")
            config.ctx_size = want
        config = apply_launch_defaults(config, hw, model_info, want_images=bool(vision_file))
        if vision_file:
            config.mmproj = str(vision_file)
            config.no_mmproj = None
            console.print(f"Images on, with {escape(vision_file.name)}. Reading an image is slow on a CPU: a "
                          f"1280x960 screenshot took 94 s and 1,221 tokens of context on the test PC.")
        if editing:
            if caps.server is not None and not caps.server.has("--spec-type"):
                console.print("[yellow]This llama.cpp build has no --spec-type; --editing is ignored.[/yellow]")
            else:
                config.spec_type = "ngram-mod"
        rendered = render_args(config, "server", caps.server)
        for s in rendered.skipped:
            console.print(f"[yellow]Skipped {escape(s)}[/yellow]")
        args = rendered.args

    # A short model name for apps (opencode's modelID and the like) instead of the full file path.
    if caps.server is None or caps.server.has("--alias"):
        args = [*args, "--alias", Path(model).stem]
    args, replaced = apply_over(args, pairs, caps.server)
    if replaced:
        console.print(f"[dim]Your options replace: {escape(' '.join(replaced))}[/dim]")
    cmd = [caps.server_path, "-m", model, "--host", host, "--port", str(port), *args]
    console.print("[bold]Starting llama-server:[/bold]" if not dry_run else "[bold]Command:[/bold]")
    shown = list(cmd)
    for i, word in enumerate(shown[:-1]):  # the command on screen never shows a password
        if is_secret(word, caps.server):
            shown[i + 1] = "***"
    _print_command(shown)
    if dry_run:
        return
    console.print(f"Open http://{host}:{port} in your browser to chat. Press Ctrl+C to stop.")
    console.print(f"For apps such as coding agents: base URL http://{host}:{port}/v1, model name "
                  f"{escape(Path(model).stem)}, context {_ctx_of(args) or 'default'} tokens.")
    try:
        proc = subprocess.run(cmd, env=clean_env())
    except FileNotFoundError:
        console.print(f"[red]Error: llama-server not found at: {escape(caps.server_path)}[/red]")
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        raise typer.Exit(code=0)
    raise typer.Exit(code=proc.returncode)


def _size_label(info: ModelInfo) -> str:
    """'4.0B, 2.6 GB' or '1.0B MoE (0.43B active), 0.8 GB'."""
    def billions(n: int) -> str:
        return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e9:.2f}B"
    size = f"{info.file_size_gb:.1f} GB"
    if info.is_moe and info.active_parameters:
        return f"{billions(info.parameters)} MoE ({billions(info.active_parameters)} active), {size}"
    return f"{billions(info.parameters)}, {size}" if info.parameters else size


def _settings_label(cfg: SearchConfig) -> str:
    parts = [f"{cfg.threads} threads" if cfg.threads else "default threads"]
    if cfg.threads_batch:
        parts.append(f"{cfg.threads_batch} for prompts")
    if cfg.flash_attn is not None:
        parts.append(f"flash attention {'on' if cfg.flash_attn else 'off'}")
    if cfg.repack is False:
        parts.append("repacking off")
    if cfg.ubatch_size and cfg.ubatch_size != 512:
        parts.append(f"ubatch {cfg.ubatch_size}")
    if cfg.numa:
        parts.append(f"numa {cfg.numa}")
    return ", ".join(parts)


def _fit_now(t: TunedSettings, info: ModelInfo, hw: HardwareInfo, budget: int, build: int) -> tuple[bool, str]:
    """Whether a tuned model can start in the RAM free now, and the words to show for it."""
    cfg, want = t.config, t.ctx or 4096
    gate = plan_memory(info, budget, isa_tier=hw.isa_tier or "avx2", ctx=want, min_ctx=min(2048, want),
                       ubatch=cfg.ubatch_size or 512, tensor_sizes=tensor_sizes_of(info.path),
                       compute_model=compute_model_for(info.path, info, build), flash_attn=cfg.flash_attn)
    if not gate.fits:
        return False, f"[red]no, {_mib(gate.shortfall)} short[/red]"
    if gate.ctx < want:
        return True, f"[yellow]yes, {gate.ctx:,} tokens[/yellow]"
    return True, "[green]yes[/green]"


def _print_tuned(number: int, t: TunedSettings, info: ModelInfo | None, fit_text: str, outdated: bool) -> None:
    path = t.model_path
    verdict_text = t.verdict or verdict(t.generation_tps)
    colour = {"Fast": "green", "Usable": "yellow", "Slow": "red"}.get(verdict_text, "white")
    gain = f" ({t.generation_tps / t.baseline_tps - 1:+.0%})" if t.baseline_tps else ""
    name = Path(path).stem if path else t.model_id[:12]
    console.print(f"[bold]{number}. {escape(name)}[/bold]  [{colour}]{verdict_text}[/{colour}]")
    if info:
        details = _size_label(info)
        if find_vision_file(path, info):
            details += ", can read images"
        console.print(f"   {details}")
    speeds = [f"Writing {t.generation_tps:.1f}{gain}"]
    if t.prompt_tps:
        speeds.append(f"reading prompts {t.prompt_tps:.1f}")
    speeds.append(f"{t.depth_tps:.1f} after {t.depth:,} tokens of chat" if t.depth_tps
                  else "long chat: run tune again to measure")
    console.print(f"   {'; '.join(speeds)}")
    console.print(f"   Settings: {_settings_label(t.config)}")
    measured = f"tuned {t.timestamp[:10]}" + (f" on llama.cpp b{t.llama_build}" if t.llama_build else "")
    console.print(f"   Fits now: {fit_text}; {measured}"
                  + (" [yellow](older than installed: run tune again)[/yellow]" if outdated else ""))
    console.print()


@app.command(help="List the models tuned on this PC with all their measured results, fastest first.")
def models(
    fits_now: bool = typer.Option(False, "--fits-now", help="Only models that fit in the RAM free right now"),
):
    """Everything tune has measured on this PC, and whether each model can start now."""
    caps = detect_capabilities()
    hw = detect_hardware(caps)
    with closing(get_session()) as conn:
        results = list_tuned(conn, _hardware_id(conn, hw))
    if not results:
        console.print("No models tuned on this PC yet. Run: llama-autotune tune model.gguf")
        return

    mem = memory_status()
    budget = memory_budget(mem.available, mem.total)
    build = caps.build.build_number
    rows: list[tuple[TunedSettings, ModelInfo | None, bool, str]] = []
    for t in results:
        info, fits, fit_text = None, False, "[dim]file missing[/dim]"
        if t.model_path and os.path.isfile(t.model_path):
            try:
                info = inspect_model(t.model_path)
                fits, fit_text = _fit_now(t, info, hw, budget, build)
            except (OSError, GGUFError):  # one damaged file must not hide the others
                info, fit_text = None, "[red]file unreadable[/red]"
        rows.append((t, info, fits, fit_text))
    shown = [r for r in rows if r[2] or not fits_now]

    counted = f"{len(shown)} of {len(rows)}" if fits_now else str(len(rows))
    console.print(f"[bold]Models tuned on this PC[/bold] ({counted}, fastest first; "
                  f"{_mib(mem.available)} of RAM free now). Speeds in tokens per second.\n")
    outdated = 0
    for number, (t, info, _, fit_text) in enumerate(shown, 1):
        old = bool(build and t.llama_build and t.llama_build < build)
        outdated += int(old)
        _print_tuned(number, t, info, fit_text, old)

    fitting = [(t, info) for t, info, fits, _ in rows if fits and info is not None]
    if fitting:
        fastest = max(fitting, key=lambda ti: ti[0].generation_tps)
        # Capability follows the total size; the active part of an MoE model only sets its speed.
        largest = max(fitting, key=lambda ti: ti[1].parameters or ti[1].active_parameters)
        console.print(f"Fastest that fits now: [bold]{escape(Path(fastest[0].model_path).stem)}[/bold] "
                      f"({fastest[0].generation_tps:.1f} tokens/s).")
        if largest is not fastest:
            console.print(f"Largest that fits now, usually the most capable: [bold]"
                          f"{escape(Path(largest[0].model_path).stem)}[/bold] "
                          f"({largest[0].generation_tps:.1f} tokens/s).")
        console.print("Start one with:")
        _print_command(["llama-autotune", "launch", os.path.abspath(fastest[0].model_path)])
    else:
        console.print("[yellow]None of them fits in the RAM free right now: close other programs and run "
                      "this again.[/yellow]")
    if outdated:
        console.print(f"[yellow]{outdated} result(s) were measured on an older llama.cpp; run tune again for "
                      f"current numbers.[/yellow]")


@app.command(help="Save the settings tune measured for a model as a launch profile (JSON).")
def export(
    profile: str = typer.Argument(..., help="File to write, e.g. my-model.json"),
    model: str = typer.Option(..., "--model", "-m", help="The tuned GGUF model file"),
):
    """Write the tuned settings of a model to a profile file for ``launch --profile``."""
    model = _model_path(model)
    caps = detect_capabilities()
    hw = detect_hardware(caps)
    model_info = _inspect(model)
    with closing(get_session()) as conn:
        tuned = get_tuned(conn, _hardware_id(conn, hw), model_info.fingerprint)
    if tuned is None:
        console.print("[red]No tuned settings for this model on this PC yet.[/red] Run first:")
        _print_command(["llama-autotune", "tune", model])
        raise typer.Exit(code=1)
    config = apply_launch_defaults(tuned.config.model_copy(update={"ctx_size": tuned.ctx or None}), hw, model_info)
    prof = create_profile(
        name=Path(profile).stem, args=render_args(config, "server", caps.server).args, model_path=model,
        hardware=hw.cpu_name, score=tuned.generation_tps, llama_build=tuned.llama_build,
    )
    console.print(f"[green]Profile exported to {escape(export_profile(prof, profile))}[/green]")


@app.command(name="import", help="Show a launch profile (JSON) before using it with launch --profile.")
def import_cmd(
    profile: str = typer.Argument(..., help="Path to profile JSON"),
):
    """Show a previously exported launch profile."""
    if not os.path.isfile(profile):
        console.print(f"[red]Error: profile file not found: {escape(profile)}[/red]")
        raise typer.Exit(code=1)
    try:
        prof = import_profile(profile)
    except (OSError, ValueError) as e:
        console.print(f"[red]Cannot read this profile: {escape(str(e))}[/red]")
        raise typer.Exit(code=1)
    table = Table(title=f"Profile: {escape(prof.name)}", box=box.ROUNDED)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", escape(prof.model_path))
    table.add_row("Hardware", escape(prof.hardware))
    table.add_row("Args", escape(" ".join(prof.args)))
    table.add_row("Score", f"{prof.score:.2f}")
    if prof.llama_build:
        table.add_row("llama.cpp build", f"b{prof.llama_build}")
    console.print(table)
    foreign = foreign_profile_flags(prof.args)
    if foreign:
        console.print(f"[yellow]launch will refuse this profile: it contains options llama-autotune never "
                      f"writes ({escape(' '.join(foreign))}).[/yellow]")


def _version_callback(value: bool) -> None:
    if value:
        from . import __version__

        console.print(f"llama-autotune {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True,
                                 help="Show version and exit"),
):
    """Configure global logging for the CLI."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


main = app
