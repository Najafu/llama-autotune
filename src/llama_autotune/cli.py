"""CLI interface for llama-autotune.

Commands: ``doctor`` (system and llama.cpp check), ``inspect`` (model and
memory plan), ``tune`` (measured CPU tuning), ``launch``, ``benchmark``,
``search``, ``export`` and ``import``. Memory is always planned against
the RAM available now, and llama.cpp arguments always come from the
compatibility layer.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from .benchmark import run_benchmark
from .compat import (
    Capabilities, architecture_supported, clean_env, config_ini_paths, detect_capabilities,
    llama_env_overrides, render_args, update_llama_advice,
)
from .constraints import GateDecision, memory_budget, plan_memory, safety_margin
from datetime import datetime, timezone

from .database import TunedSettings, best_bandwidth, get_session, get_tuned, save_benchmark, save_tuned
from .hardware import detect_hardware, hardware_fingerprint
from .heuristics import apply_launch_defaults, generate_initial_config
from .memory import MIB, compute_model_for, estimate_memory, kv_cache_bytes, tensor_sizes_of
from .model_inspector import inspect_model
from .models import Backend, BenchmarkEntry, HardwareInfo, ModelInfo, OptimizeObjective, SearchConfig
from .optimizer import Optimizer
from .profiles import create_profile, export_profile, import_profile
from .resources import MemoryStatus, measure_bandwidth, memory_status, power_status
from .tuner import VERDICT_REASONS, Tuner, TuneReport

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


def _require_model_file(path: str) -> None:
    if not os.path.isfile(path):
        console.print(f"[red]Error: model file not found: {path}[/red]")
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
    table.add_row("Path", m.path)
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
    table.add_row("Read per token", f"{m.bytes_per_token / 1e9:.2f} GB")
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
            verdict = "[green]Fits[/green]"
        else:
            plan = plan_memory(model, budget, isa_tier=tier, ctx=ctx, min_ctx=ctx, tensor_sizes=sizes,
                               compute_model=cm)
            if plan.fits:
                verdict = f"[yellow]Fits with: {', '.join(s.split(' (')[0] for s in plan.savings)}[/yellow]"
            else:
                verdict = f"[red]Won't fit (short by {_mib(plan.shortfall)})[/red]"
        label = f"{ctx:,}" + (" (full)" if ctx == model.training_context else "")
        table.add_row(label, _mib(default.total), verdict)
    return table


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
    llama.add_row("llama-server", caps.server_path)
    llama.add_row("llama-bench", caps.bench_path)
    if caps.build.build_number:
        llama.add_row("Build", f"b{caps.build.build_number} ({caps.build.version}, commit {caps.build.commit})")
    else:
        llama.add_row("Build", "[red]not found: set LLAMA_CPP_DIR to the folder with llama-server[/red]")
    llama.add_row("Devices", ", ".join(f"{d.name} {d.description} ({d.total_mib} MiB)" for d in caps.devices)
                  or "CPU only")
    if caps.server:
        checks = {"--fit": "auto memory fitting", "-lm": "load modes", "--repack": "weight repacking",
                  "-cram": "prompt cache size", "--spec-type": "speculative decoding"}
        llama.add_row("Features", ", ".join(f"{name} {'yes' if caps.server.has(flag) else 'no'}"
                                            for flag, name in checks.items()))
    console.print(llama)

    warnings = []
    for key, value in llama_env_overrides().items():
        warnings.append(f"{key}={value} is set and silently changes llama.cpp settings")
    for path in config_ini_paths():
        warnings.append(f"llama.cpp default-settings file found: {path}")
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
    _require_model_file(model)
    hw = detect_hardware()
    mem = memory_status()
    try:
        model_info = inspect_model(model)
    except (FileNotFoundError, PermissionError) as e:
        console.print(f"[red]Error reading model: {e}[/red]")
        raise typer.Exit(code=1)
    console.print(_hardware_table(hw, mem))
    console.print(_model_table(model_info))
    caps = detect_capabilities()
    _loadable(caps, model_info)
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
    ctx: int = typer.Option(4096, "-c", "--ctx-size", help="Context size you want to chat with"),
    minutes: float = typer.Option(10.0, "--minutes", help="Time limit for the measurements"),
    output_profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Also save a launch profile file"),
):
    """Measure llama.cpp's defaults and better settings, then save the winner for ``launch``."""
    _require_model_file(model)
    caps = detect_capabilities()
    hw = detect_hardware(caps)
    if hw.backend != Backend.CPU:
        console.print("[yellow]tune measures CPU settings. GPU tuning comes in a later version; "
                      "`launch` already lets llama.cpp fit the model to your GPU.[/yellow]")
        raise typer.Exit(code=1)
    conn = get_session()
    known_bw = best_bandwidth(conn, hardware_fingerprint(hw))
    conn.close()
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
    conn = get_session()
    save_tuned(conn, hardware_fingerprint(hw), tuner.model.fingerprint, TunedSettings(
        config=tuned_cfg, llama_build=caps.build.build_number, generation_tps=rep.best_tps,
        prompt_tps=rep.best_pp, baseline_tps=rep.baseline_tps, ctx=gate.ctx, efficiency=rep.efficiency,
        bandwidth_gbs=rep.bandwidth_gbs, model_path=model,
    ))
    conn.close()

    if output_profile:
        launch_cfg = apply_launch_defaults(rep.config.model_copy(update={"ctx_size": gate.ctx}), hw, tuner.model)
        profile = create_profile(
            name=Path(output_profile).stem, args=render_args(launch_cfg, "server", caps.server).args,
            model_path=model, hardware=hw.cpu_name, score=rep.best_tps, llama_build=caps.build.build_number,
        )
        console.print(f"[green]Profile saved to {export_profile(profile, output_profile)}[/green]")
    console.print(f"\n[green]Saved.[/green] Start chatting with these settings:  "
                  f"llama-autotune launch {subprocess.list2cmdline([model])}")


def _apply_tuned(config: SearchConfig, hw: HardwareInfo, model_info: ModelInfo, build: int) -> SearchConfig:
    """Overlay the settings saved by ``tune`` for this machine and model."""
    conn = get_session()
    try:
        tuned = get_tuned(conn, hardware_fingerprint(hw), model_info.fingerprint)
    finally:
        conn.close()
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
    _require_model_file(model)
    caps = detect_capabilities()
    hw = detect_hardware(caps)
    model_info = inspect_model(model)
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
        console.print(f"[yellow]{result.raw_output.splitlines()[0]}[/yellow]")

    entry = BenchmarkEntry(hardware_id=hardware_fingerprint(hw), model_id=model_info.fingerprint,
                           config=config, result=result, llama_build=caps.build.build_number,
                           model_path=model)
    conn = get_session()
    save_benchmark(conn, entry)
    conn.close()

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
        console.print(f"[green]Saved to {output_file}[/green]")
    if not result.success:
        raise typer.Exit(code=1)


@app.command(help="Advanced: search many settings with Optuna (for GPUs; on a CPU use tune).")
def search(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    objective: str = typer.Option("balanced", "--objective", "-O", help="Optimization objective"),
    trials_b: int = typer.Option(15, "--trials-b", help="Stage B trials"),
    trials_c: int = typer.Option(40, "--trials-c", help="Stage C trials"),
    output_profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Save best profile to file"),
):
    """Find the fastest configuration that fits in free memory."""
    _require_model_file(model)
    try:
        obj = OptimizeObjective(objective)
    except ValueError:
        valid = [o.value for o in OptimizeObjective]
        console.print(f"[red]Invalid objective. Choose from: {', '.join(valid)}[/red]")
        raise typer.Exit(code=1)

    opt = Optimizer(model_path=model, objective=obj, n_trials_stage_b=trials_b, n_trials_stage_c=trials_c)
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
    result = run_benchmark(model, best_config, repetitions=opt._bench_reps,
                           n_prompt=opt._n_prompt, n_gen=opt._n_gen, registry=opt.caps.bench)

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
        console.print(f"[green]Profile saved to {export_profile(profile, output_profile)}[/green]")

    if opt.total_evals > 0:
        conn = get_session()
        save_benchmark(conn, BenchmarkEntry(
            hardware_id=hardware_fingerprint(opt.hw), model_id=opt.model.fingerprint,
            config=best_config, result=result, objective=obj,
            llama_build=opt.caps.build.build_number, model_path=model))
        conn.close()


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


@app.command(help="Start llama-server with a configuration that fits in free RAM.")
def launch(
    model: str = typer.Argument(..., help="Path to GGUF model file"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Path to profile JSON"),
    ctx: int = typer.Option(4096, "-c", "--ctx-size", help="Wanted context size (reduced if it does not fit)"),
    images: bool = typer.Option(False, "--images", help="Keep the vision projector loaded"),
    editing: bool = typer.Option(False, "--editing",
                                 help="Faster when rewriting or editing text you paste in (speculative decoding)"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", help="Server port"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command without starting the server"),
):
    """Plan memory, apply safe defaults and start llama-server."""
    _require_model_file(model)
    caps = detect_capabilities()
    if not _loadable(caps, inspect_model(model)):
        raise typer.Exit(code=1)
    if profile:
        if not os.path.isfile(profile):
            console.print(f"[red]Error: profile file not found: {profile}[/red]")
            raise typer.Exit(code=1)
        prof = import_profile(profile)
        args = prof.args
        if prof.llama_build and caps.build.build_number and prof.llama_build != caps.build.build_number:
            console.print(f"[yellow]Profile was tuned on llama.cpp b{prof.llama_build}; "
                          f"installed is b{caps.build.build_number}. Consider re-running search.[/yellow]")
        unknown = [a for a in args if _FLAG.match(a) and caps.server and not caps.server.has(a)]
        if unknown:
            console.print(f"[red]This llama.cpp build does not accept: {' '.join(unknown)}[/red]")
            raise typer.Exit(code=1)
    else:
        hw = detect_hardware(caps)
        model_info = inspect_model(model)
        if model_info.training_context:
            ctx = min(ctx, model_info.training_context)
        config = _apply_tuned(generate_initial_config(hw, model_info), hw, model_info,
                              caps.build.build_number)
        if hw.backend == Backend.CPU:
            mem = memory_status()
            gate = plan_memory(model_info, memory_budget(mem.available, mem.total),
                               isa_tier=hw.isa_tier or "avx2", ctx=ctx, min_ctx=min(2048, ctx),
                               ubatch=config.ubatch_size or 512, tensor_sizes=tensor_sizes_of(model),
                               compute_model=compute_model_for(model, model_info, caps.build.build_number),
                               flash_attn=config.flash_attn)
            _print_gate(gate, mem)
            if not gate.fits:
                raise typer.Exit(code=1)
            config = gate.apply(config)
        else:
            config.ctx_size = ctx
        config = apply_launch_defaults(config, hw, model_info, want_images=images)
        if editing:
            if caps.server is not None and not caps.server.has("--spec-type"):
                console.print("[yellow]This llama.cpp build has no --spec-type; --editing is ignored.[/yellow]")
            else:
                config.spec_type = "ngram-mod"
        rendered = render_args(config, "server", caps.server)
        for s in rendered.skipped:
            console.print(f"[yellow]Skipped {s}[/yellow]")
        args = rendered.args

    cmd = [caps.server_path, "-m", model, "--host", host, "--port", str(port), *args]
    console.print("[bold]Starting llama-server:[/bold]" if not dry_run else "[bold]Command:[/bold]")
    console.print(f"  {subprocess.list2cmdline(cmd)}")
    if dry_run:
        return
    console.print(f"Open http://{host}:{port} in your browser to chat. Press Ctrl+C to stop.")
    try:
        proc = subprocess.run(cmd, env=clean_env())
    except FileNotFoundError:
        console.print(f"[red]Error: llama-server not found at: {caps.server_path}[/red]")
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        raise typer.Exit(code=0)
    raise typer.Exit(code=proc.returncode)


@app.command(help="Export a launch profile to JSON.")
def export(
    profile: str = typer.Argument(..., help="Profile name or path to save"),
    model: str = typer.Option("", "--model", "-m", help="Model path"),
    hardware: str = typer.Option("", "--hardware", help="Hardware description"),
    score: float = typer.Option(0.0, "--score", help="Profile score"),
):
    """Export a launch profile to JSON."""
    prof = create_profile(
        name=Path(profile).stem if Path(profile).suffix else profile,
        args=[], model_path=model, hardware=hardware, score=score,
    )
    console.print(f"[green]Profile exported to {export_profile(prof, profile)}[/green]")


@app.command(name="import", help="Import a launch profile from JSON.")
def import_cmd(
    profile: str = typer.Argument(..., help="Path to profile JSON"),
):
    """Show a previously exported launch profile."""
    if not os.path.isfile(profile):
        console.print(f"[red]Error: profile file not found: {profile}[/red]")
        raise typer.Exit(code=1)
    prof = import_profile(profile)
    table = Table(title=f"Profile: {prof.name}", box=box.ROUNDED)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", prof.model_path)
    table.add_row("Hardware", prof.hardware)
    table.add_row("Args", " ".join(prof.args))
    table.add_row("Score", f"{prof.score:.2f}")
    if prof.llama_build:
        table.add_row("llama.cpp build", f"b{prof.llama_build}")
    console.print(table)


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
