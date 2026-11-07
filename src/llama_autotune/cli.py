"""CLI interface for llama-autotune.

Commands: ``doctor`` (system and llama.cpp check), ``inspect`` (model and
memory plan), ``benchmark``, ``search``, ``launch``, ``export`` and
``import``. Memory is always planned against the RAM available now, and
llama.cpp arguments always come from the compatibility layer.
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
from .compat import clean_env, config_ini_paths, detect_capabilities, llama_env_overrides, render_args
from .constraints import GateDecision, memory_budget, plan_memory, safety_margin
from .database import get_session, save_benchmark
from .hardware import detect_hardware, hardware_fingerprint
from .heuristics import apply_launch_defaults, generate_initial_config
from .memory import MIB, compute_buffer_for, estimate_memory, kv_cache_bytes, tensor_sizes_of
from .model_inspector import inspect_model
from .models import Backend, BenchmarkEntry, HardwareInfo, ModelInfo, OptimizeObjective, SearchConfig
from .optimizer import Optimizer
from .profiles import create_profile, export_profile, import_profile
from .resources import MemoryStatus, measure_bandwidth, memory_status, power_status

EXAMPLES = (
    "[bold]Examples:[/bold]\n\n"
    "  llama-autotune doctor\n\n"
    "  llama-autotune inspect model.gguf\n\n"
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


def _memory_plan_table(model: ModelInfo, hw: HardwareInfo, mem: MemoryStatus, sizes: dict) -> Table:
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
    for ctx in contexts:
        compute = compute_buffer_for(model.path, ctx)
        default = estimate_memory(model, ctx, isa_tier=tier, tensor_sizes=sizes, compute_override=compute)
        if default.total <= budget:
            verdict = "[green]Fits[/green]"
        else:
            plan = plan_memory(model, budget, isa_tier=tier, ctx=ctx, min_ctx=ctx, tensor_sizes=sizes,
                               compute_override=compute)
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
        sys_table.add_row("Memory bandwidth", f"{measure_bandwidth()} GB/s")
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
    if hw.backend == Backend.CPU:
        console.print(_memory_plan_table(model_info, hw, mem, tensor_sizes_of(model)))


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


@app.command(help="Auto-optimize config via 3-stage search.")
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
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", help="Server port"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the command without starting the server"),
):
    """Plan memory, apply safe defaults and start llama-server."""
    _require_model_file(model)
    caps = detect_capabilities()
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
        config = generate_initial_config(hw, model_info)
        if hw.backend == Backend.CPU:
            mem = memory_status()
            gate = plan_memory(model_info, memory_budget(mem.available, mem.total),
                               isa_tier=hw.isa_tier or "avx2", ctx=ctx, min_ctx=min(2048, ctx),
                               ubatch=config.ubatch_size or 512, tensor_sizes=tensor_sizes_of(model),
                               compute_override=compute_buffer_for(model, ctx))
            _print_gate(gate, mem)
            if not gate.fits:
                raise typer.Exit(code=1)
            config = gate.apply(config)
        else:
            config.ctx_size = ctx
        config = apply_launch_defaults(config, hw, model_info, want_images=images)
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
