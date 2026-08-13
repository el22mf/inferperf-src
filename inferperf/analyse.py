#!/usr/bin/env python3
"""
analyse.py
- Analyse a workload and generate a baseline PMU profile.
- Produces a cached baseline used later by validate.py.
- Generates a baseline flamegraph and identifies bottlenecks.
- Displays findings and recommendations to user.
"""

import tomli_w
from pathlib import Path
from datetime import datetime
import sys
import subprocess

from inferperf.utils.pmu import run_pmu, EVENTS
from inferperf.utils.classify import classify_bottleneck
from inferperf.utils.workloads import load_config, select_workload, load_workload, normalize_workload_path
from inferperf.utils.flamegraph import generate_flamegraph
from inferperf.utils.recommendations import RECOMMENDATIONS_TABLE, MORE_RECOMMENDATIONS_TABLE

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def _sanitize_for_toml(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_toml(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_toml(v) for v in obj]
    if obj is None:
        return "unknown"
    return obj


def write_cache(data):
    """Write the analysis output to .inferperf_cache.toml"""
    cache_path = Path(".inferperf_cache.toml")
    safe = _sanitize_for_toml(data)
    with cache_path.open("wb") as f:
        tomli_w.dump(safe, f)


# ------------------ action bar for input ------------------
def analyse_action_bar():
    console.print(
        Panel(
            Text.from_markup(
                "[bold cyan]Next Actions[/bold cyan]\n\n"
                "[bold green][F][/bold green] View Flamegraph\n"
                "[bold yellow][T][/bold yellow] Toggle Recommendations (Short/Long)\n"
                "[bold blue][M][/bold blue] Show Metadata\n"
                "[bold magenta][B][/bold magenta] Next Bottleneck\n"
                "[bold cyan][R][/bold cyan] Reset\n"
                "[bold red][Q][/bold red] Quit"
            ),
            border_style="cyan"
        )
    )

    return Prompt.ask(
        "Choose an action",
        choices=["f", "t", "m", "b", "r","q"],
        default="r"
    )

def extract_warmup(args):
    warmup = 0
    clean = []

    i = 0
    while i < len(args):
        if args[i] == "--warmup":
            if i + 1 < len(args):
                try:
                    warmup = int(args[i+1])
                except ValueError:
                    warmup = 0
            i += 2
            continue
        clean.append(args[i])
        i += 1

    return warmup, clean


def get_warmup(args):
    if isinstance(args, list):
        if "--warmup" in args:
            idx = args.index("--warmup")
            if idx + 1 < len(args):
                try:
                    return int(args[idx + 1])
                except ValueError:
                    return 0
    return 0


# ------------------ rich renderers ------------------
def render_metrics(pmu):

    def colour(value, good=None, warn=None, decimals=2):
        rounded = float(f"{value:.{decimals}f}")
        text = f"{rounded:.{decimals}f}%"
        if good is None:
            return text
        if rounded <= good:
            return f"[green]{text}[/green]"
        elif warn is not None and rounded <= warn:
            return f"[yellow]{text}[/yellow]"
        else:
            return f"[red]{text}[/red]"

    def r_int(v): return f"{int(v)}"
    def r3(v): return f"{v:.3f}"

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    # Core Execution
    table.add_row("[bold white]Core Execution[/bold white]", "")
    table.add_row("cycles", r_int(pmu.get("cycles")))
    table.add_row("instructions", r_int(pmu.get("instructions")))
    table.add_row("ipc", r3(pmu.get("ipc")))

    # Tokens (only if present)
    if pmu.get("tokens") is not None:
        table.add_row("", "")
        table.add_row("[bold white]Per-Token Efficiency[/bold white]", "")
        table.add_row("tokens", r_int(pmu.get("tokens")))
        table.add_row("cycles_per_token", r_int(pmu.get("cycles_per_token")))
        table.add_row("instructions_per_token", r_int(pmu.get("instructions_per_token")))

    # Memory
    table.add_row("", "")
    table.add_row("[bold white]Memory Indicators[/bold white]", "")
    table.add_row("cache_misses", r_int(pmu.get("cache_misses")))
    table.add_row("cache_miss_rate", colour(pmu.get("cache_miss_rate") * 100, 10, 25))
    table.add_row("l1d_refill", r_int(pmu.get("l1d_refill")))
    table.add_row("l1d_rate", colour(pmu.get("l1d_rate") * 100, 5, 10))
    table.add_row("l2d_refill", r_int(pmu.get("l2d_refill")))
    table.add_row("l2d_rate", colour(pmu.get("l2d_rate") * 100, 2, 5))

    # Branch
    table.add_row("", "")
    table.add_row("[bold white]Branch Indicators[/bold white]", "")
    table.add_row("branch_misses", r_int(pmu.get("branch_misses")))
    table.add_row("branch_miss_rate", colour(pmu.get("branch_miss_rate") * 100, 1, 3))

    # TLB
    table.add_row("", "")
    table.add_row("[bold white]TLB Indicators[/bold white]", "")
    table.add_row("dtlb_misses", r_int(pmu.get("dtlb_misses")))
    table.add_row("itlb_misses", r_int(pmu.get("itlb_misses")))
    table.add_row("tlb_miss_rate", colour(pmu.get("tlb_miss_rate") * 100, 1, 3))

    console.print(
        Panel(
            table,
            title="[bold cyan]PMU Diagnostic Summary[/bold cyan]",
            border_style="cyan"
        )
    )


def render_bottleneck_panel(b, index=None):
    colour = "red" if b["confidence"] > 0.7 else "yellow"

    title = (
        f"Primary Bottleneck — {b['type']}"
        if index is None
        else f"Secondary Bottleneck #{index} — {b['type']}"
    )

    text = (
        f"[bold white]{b['type']}[/bold white]\n"
        f"Confidence: {b['confidence']:.2f}"
    )

    console.print(
        Panel(
            text,
            title=title,
            border_style=colour
        )
    )


def render_recommendations(recs):
    text = "\n".join(f"- {r}" for r in recs)
    console.print(Panel(text, title="Recommendations", border_style="green"))


def render_more_recommendations(recs):
    text = "\n".join(recs)
    console.print(Panel(text, title="More Recommendations", border_style="green"))


def render_metadata(meta):
    text = (
        f"Timestamp: {meta['timestamp']}\n"
        f"InferPerf version: {meta['inferperf_version']}\n"
        "\n"
        f"Device: {meta['device']}\n"
        f"CPU cores: {meta['cpu_cores']}\n"
        f"CPU freq (MHz): {meta['cpu_freq_mhz']}\n"
        f"RAM (GB): {meta['ram_gb']}\n"
        "\n"
        f"Threads: {meta['threads']}\n"
        f"Precision: {meta['precision']}\n"
        f"Inference engine: {meta['inference_engine']}\n"
        f"Python version: {meta['python_version']}\n"
        "\n"
        f"Workload file: {meta['workload_file']}\n"
        f"Workload args: {meta['workload_args']}\n"
        f"Input shape: {meta['input_shape']}\n"
        f"Batch size: {meta['batch_size']}\n"
        "\n"
        f"Perf events: {meta['perf_events']}\n"
        f"Perf command: {meta['perf_command']}"
    )

    console.print(Panel(text, title="Metadata", border_style="blue"))


def render_analysis(data):
    if not RICH_AVAILABLE:
        print("Rich is not installed. Install with: pip install rich")
        print(data)
        return

    console.print(Panel(f"[bold cyan]Workload:[/bold cyan] {data['workload']}", border_style="cyan"))

    render_metrics(data["pmu"])
    render_bottleneck_panel(data["bottleneck"])
    render_recommendations(data["recommendations"])


# ------------------ main block ------------------
def run_analyse(workload, args):
    # 1. Resolve workload path and load workload
    if workload is None:
        cfg_workload, cfg_args = load_config()
        if cfg_workload:
            workload_path = normalize_workload_path(cfg_workload)
            args = cfg_args
        else:
            workload_path, args = select_workload()
            workload_path = normalize_workload_path(workload_path)
    else:
        workload_path = normalize_workload_path(workload)

    workload = load_workload(workload_path)

    # Extract warmup from args if provided
    warmup, clean_args = extract_warmup(args)

    # If user did not provide --warmup, ask interactively
    if warmup == 0:
        warm = input("\nWarm-up runs [0–3]: ").strip()
        if warm not in ("0", "1", "2", "3"):
            warm = "1"
        warmup = int(warm)

    # Rebuild args for PMU run
    args = ["--warmup", str(warmup)]


    # 2. Run warmups (if chosen)

    if warmup > 0:
        console.print(f"[bold cyan]Running {warmup} warm-up pass(es)...[/bold cyan]")
        for i in range(warmup):
            console.print(f"[cyan]Warm-up run {i+1}/{warmup}[/cyan]")
            workload.run() 
        console.print("[green]Warm-up complete. Starting profiled run...[/green]\n")


    # 3. Run PMU collection
    pmu_result = run_pmu(workload_path, clean_args)
    pmu = pmu_result.get("pmu", {})

    # 4. Classification
    classification = classify_bottleneck(pmu)
    all_bottlenecks = classification.get("all_bottlenecks", [classification["bottleneck"]])

    # 5. Metadata
    metadata = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "inferperf_version": "1.0.0",

        "device": "Raspberry Pi 5 (Cortex-A72)",
        "cpu_cores": 4,
        "cpu_freq_mhz": 2400,
        "ram_gb": 8,

        "threads": getattr(args, "threads", 1),
        "precision": "fp32",
        "inference_engine": "ONNX Runtime",
        "python_version": sys.version.split()[0],

        "workload_file": workload_path,
        "workload_args": args,
        "input_shape": getattr(args, "input_shape", "unknown"),

        # temporary placeholders
        "batch_size": None,
        "sequence_length": None,
        "tokens": None,

        "perf_events": EVENTS,
        "perf_command": pmu_result.get("perf_command", "unknown"),
    }

    # Extract batch/sequence from workload module
    batch = getattr(workload, "BATCH_SIZE", None)
    seq = getattr(workload, "SEQUENCE_LENGTH", None)

    metadata["batch_size"] = batch
    metadata["sequence_length"] = seq

    tokens = None
    if isinstance(batch, int) and isinstance(seq, int):
        tokens = batch * seq

    metadata["tokens"] = tokens

    if tokens is not None:
        pmu["tokens"] = tokens
        pmu["cycles_per_token"] = pmu["cycles"] / tokens
        pmu["instructions_per_token"] = pmu["instructions"] / tokens
    else:
        pmu["tokens"] = None
        pmu["cycles_per_token"] = None
        pmu["instructions_per_token"] = None

    # 6. Assemble final output
    data = {
        "workload": workload_path,
        "args": args,
        "pmu": pmu,
        **classification,
        "all_bottlenecks": all_bottlenecks,
        "metadata": metadata,
    }

    data["workload_file"] = workload_path

    # 7. Generate baseline flamegraph
    svg_path = generate_flamegraph(workload_path, clean_args, "baseline_flamegraph.svg")

    # 8. Cache + render
    write_cache(data)
    render_analysis(data)

    current_bottleneck_index = 0
    show_long_recs = False

    while True:
        choice = analyse_action_bar()

        if choice == "f":
            subprocess.run(["xdg-open", svg_path])
            console.clear()


        elif choice == "t":
            console.clear()
            show_long_recs = not show_long_recs

            b = data["all_bottlenecks"][current_bottleneck_index]

            if current_bottleneck_index == 0:
                render_bottleneck_panel(b, None)
            else:
                render_bottleneck_panel(b, current_bottleneck_index)

            if show_long_recs:
                recs = MORE_RECOMMENDATIONS_TABLE[b["type"]]
                render_more_recommendations(recs)
            else:
                recs = RECOMMENDATIONS_TABLE[b["type"]]
                render_recommendations(recs)

            continue


        elif choice == "m":
            console.clear()
            render_metadata(data["metadata"])
            continue


        elif choice == "b":
            console.clear()
            current_bottleneck_index += 1

            if current_bottleneck_index >= len(data["all_bottlenecks"]):
                console.print("[bold red]No further bottlenecks detected.[/bold red]")
                current_bottleneck_index -= 1
                continue

            b = data["all_bottlenecks"][current_bottleneck_index]

            if current_bottleneck_index == 0:
                render_bottleneck_panel(b, None)
            else:
                render_bottleneck_panel(b, current_bottleneck_index)

            if show_long_recs:
                recs = MORE_RECOMMENDATIONS_TABLE[b["type"]]
                render_more_recommendations(recs)
            else:
                recs = RECOMMENDATIONS_TABLE[b["type"]]
                render_recommendations(recs)

            continue


        elif choice == "r":
            console.clear()
            current_bottleneck_index = 0
            b = data["all_bottlenecks"][0]

            console.print(Panel(f"[bold cyan]Workload:[/bold cyan] {data['workload']}", border_style="cyan"))
            render_metrics(data["pmu"])

            render_bottleneck_panel(b, None)

            if show_long_recs:
                recs = MORE_RECOMMENDATIONS_TABLE[b["type"]]
                render_more_recommendations(recs)
            else:
                recs = RECOMMENDATIONS_TABLE[b["type"]]
                render_recommendations(recs)

            continue


        elif choice == "q":
            break


