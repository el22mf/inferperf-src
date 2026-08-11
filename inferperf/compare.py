#!/usr/bin/env python3
"""
compare.py
- Compare two separate workloads side-by-side using fresh PMU runs.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False
    console = None

from inferperf.utils.pmu import run_pmu, EVENTS
from inferperf.utils.classify import classify_bottleneck
from inferperf.utils.workloads import load_workload, normalize_workload_path
from inferperf.utils.flamegraph import generate_flamegraph


# ------------------ workload selection menu ------------------

def list_workloads():
    folder = Path("inferperf/workloads")
    return [p for p in folder.glob("*.py") if p.name != "__init__.py"]


def workload_selection_menu():
    workloads = list_workloads()

    for i, w in enumerate(workloads, 1):
        print(f"{i}. {w.name}")

    print(f"{len(workloads)+1}. custom path\n")

    choice = input("Enter choice: ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(workloads):
        return str(workloads[int(choice)-1])

    return input("Enter path to workload: ").strip()


# ------------------ thresholds + semantic rules ------------------

INT_THRESHOLDS = {
    "cycles": 600_000_000,
    "instructions": 800_000_000,
    "dtlb_misses": 500_000,
}

PP_THRESHOLDS = {
    "ipc": 0.10,
    "cache_miss_rate": 0.05,
    "l1d_rate": 0.05,
    "l2d_rate": 0.05,
    "branch_miss_rate": 0.02,
}

SEMANTIC_RULES = {
    "ipc": "higher_is_good",
    "cycles": "higher_is_bad",
    "instructions": "neutral",
    "cache_miss_rate": "higher_is_bad",
    "l1d_rate": "higher_is_bad",
    "l2d_rate": "higher_is_bad",
    "dtlb_misses": "higher_is_bad",
    "branch_miss_rate": "higher_is_bad",
}


# ------------------ colour helpers ------------------

def colour_semantic(metric, delta, threshold):
    rule = SEMANTIC_RULES.get(metric, "neutral")

    if abs(delta) < threshold:
        return f"[white]{delta:+.2f}[/white]"

    if rule == "neutral":
        return f"[white]{delta:+.2f}[/white]"

    if rule == "higher_is_good":
        return f"[green]{delta:+.2f}[/green]" if delta > 0 else f"[red]{delta:+.2f}[/red]"

    if rule == "higher_is_bad":
        return f"[red]{delta:+.2f}[/red]" if delta > 0 else f"[green]{delta:+.2f}[/green]"

    return f"[white]{delta:+.2f}[/white]"


def colour_semantic_int(metric, delta, threshold):
    rule = SEMANTIC_RULES.get(metric, "neutral")

    if abs(delta) < threshold:
        return f"[white]{delta:+,}[/white]"

    if rule == "neutral":
        return f"[white]{delta:+,}[/white]"

    if rule == "higher_is_good":
        return f"[green]{delta:+,}[/green]" if delta > 0 else f"[red]{delta:+,}[/red]"

    if rule == "higher_is_bad":
        return f"[red]{delta:+,}[/red]" if delta > 0 else f"[green]{delta:+,}[/green]"

    return f"[white]{delta:+,}[/white]"


def format_pct(v):
    return f"{v*100:.2f}%"


# ------------------ side-by-side renderer ------------------

def render_side_by_side(A, B, metaA, metaB):
    keys = [
        ("cycles", "cycles", "Core Execution"),
        ("instructions", "instructions", None),
        ("ipc", "ipc", None),

        ("cache_misses", "cache_misses", "Memory Indicators"),
        ("cache_miss_rate", "cache_miss_rate", None),
        ("l1d_refill", "l1d_refill", None),
        ("l1d_rate", "l1d_rate", None),
        ("l2d_refill", "l2d_refill", None),
        ("l2d_rate", "l2d_rate", None),

        ("branch_misses", "branch_misses", "Branch Indicators"),
        ("branch_miss_rate", "branch_miss_rate", None),

        ("dtlb_misses", "dtlb_misses", "TLB Indicators"),
        ("itlb_misses", "itlb_misses", None),
        ("tlb_miss_rate", "tlb_miss_rate", None),
    ]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Workload A", style="grey70", no_wrap=True)
    table.add_column("Workload B", style="grey70", no_wrap=True)
    table.add_column("Delta", style="white", no_wrap=True)

    current_section = None

    for key, label, section in keys:
        if section and section != current_section:
            table.add_row(f"[bold white]{section}[/bold white]", "", "", "")
            current_section = section

        a = A.get(key, 0)
        b = B.get(key, 0)

        if key.endswith("_rate") or key == "ipc":
            a_str = format_pct(a)
            b_str = format_pct(b)
            delta_pp = (b - a) * 100
            threshold = PP_THRESHOLDS.get(key, 0.10)
            delta_str = f"[bold]{colour_semantic(key, delta_pp, threshold)}[/bold]"
        else:
            a_str = f"{a:,}"
            b_str = f"{b:,}"
            delta_int = b - a
            threshold = INT_THRESHOLDS.get(key, 500_000)
            delta_str = f"[bold]{colour_semantic_int(key, delta_int, threshold)}[/bold]"

        table.add_row(label, a_str, b_str, delta_str)

    console.print(Panel(table, title="Workload A vs Workload B", border_style="cyan"))

    # Bottleneck panels
    left = Panel(
        f"[bold white]{metaA['bottleneck']['type']}[/bold white]\nConfidence: {metaA['bottleneck']['confidence']:.2f}",
        title="Workload A Bottleneck",
        border_style="green"
    )
    right = Panel(
        f"[bold white]{metaB['bottleneck']['type']}[/bold white]\nConfidence: {metaB['bottleneck']['confidence']:.2f}",
        title="Workload B Bottleneck",
        border_style="yellow"
    )

    console.print(left)
    console.print(right)


# ------------------ metadata renderer ------------------

def render_metadata(metaA, metaB):
    def block(meta, title):
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
        console.print(Panel(text, title=title, border_style="blue"))

    block(metaA, "Metadata — Workload A")
    block(metaB, "Metadata — Workload B")


# ------------------ menu ------------------

def compare_action_bar():
    console.print(
        Panel(
            Text.from_markup(
                "[bold cyan]Next Actions[/bold cyan]\n\n"
                "[bold green][F][/bold green] View Flamegraph A\n"
                "[bold green][G][/bold green] View Flamegraph B\n"
                "[bold magenta][M][/bold magenta] Show Metadata\n"
                "[bold cyan][R][/bold cyan] Reset\n"
                "[bold red][Q][/bold red] Quit"
            ),
            border_style="cyan"
        )
    )

    return Prompt.ask(
        "Choose an action",
        choices=["f", "g", "m", "r", "q"],
        default="r"
    )


# ------------------ main ------------------

def run_compare():
    if not RICH_AVAILABLE:
        print("Rich is required for compare.py")
        return 1

    # 1. Workload selection menu
    print("\nSelect Workload A:\n")
    workloadA = normalize_workload_path(workload_selection_menu())

    print("\nSelect Workload B:\n")
    workloadB = normalize_workload_path(workload_selection_menu())

    # 2. Warmup if chosen
    warm = input("\nWarm-up runs [0–3]: ").strip()
    if warm not in ("0", "1", "2", "3"):
        warm = "1"
    warmup = int(warm)

    # 3. Load workloads
    wA = load_workload(workloadA)
    wB = load_workload(workloadB)

    # Warmup + Perf A
    if warmup > 0:
        console.print(f"[cyan]Running warmups for Workload A...[/cyan]")
        for i in range(warmup):
            console.print(f"[cyan]Warm-up run {i+1}/{warmup}[/cyan]")
            wA.run()

    console.print("[cyan]Running PMU for Workload A...[/cyan]")
    pmuA = run_pmu(workloadA, []).get("pmu", {})

    # Warmup + Perf B
    if warmup > 0:
        console.print(f"[cyan]Running warmups for Workload B...[/cyan]")
        for i in range(warmup):
            console.print(f"[cyan]Warm-up run {i+1}/{warmup}[/cyan]")
            wB.run()

    console.print("[cyan]Running PMU for Workload B...[/cyan]")
    pmuB = run_pmu(workloadB, []).get("pmu", {})


    # 4. Classification
    classA = classify_bottleneck(pmuA)
    classB = classify_bottleneck(pmuB)

    # 5. Metadata
    def build_meta(workload, pmu, bottleneck):
        return {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "inferperf_version": "1.0.0",
            "device": "Raspberry Pi 5 (Cortex-A72)",
            "cpu_cores": 4,
            "cpu_freq_mhz": 2400,
            "ram_gb": 8,
            "threads": 1,
            "precision": "fp32",
            "inference_engine": "ONNX Runtime",
            "python_version": sys.version.split()[0],
            "workload_file": workload,
            "workload_args": [],
            "input_shape": "unknown",
            "batch_size": "unknown",
            "perf_events": EVENTS,
            "perf_command": pmu.get("perf_command", "unknown"),
            "bottleneck": bottleneck["bottleneck"],
        }

    metaA = build_meta(workloadA, pmuA, classA)
    metaB = build_meta(workloadB, pmuB, classB)

    # 6. Flamegraphs
    svgA = generate_flamegraph(workloadA, [], "compare_A_flamegraph.svg")
    svgB = generate_flamegraph(workloadB, [], "compare_B_flamegraph.svg")

    # 7. Render interface
    console.clear()
    render_side_by_side(pmuA, pmuB, metaA, metaB)

    while True:
        choice = compare_action_bar()

        if choice == "f":
            console.clear()
            subprocess.run(["xdg-open", svgA])
            console.clear()
            render_side_by_side(pmuA, pmuB, metaA, metaB)
            continue

        elif choice == "g":
            console.clear()
            subprocess.run(["xdg-open", svgB])
            console.clear()
            render_side_by_side(pmuA, pmuB, metaA, metaB)
            continue

        elif choice == "m":
            console.clear()
            render_metadata(metaA, metaB)
            continue

        elif choice == "r":
            console.clear()
            render_side_by_side(pmuA, pmuB, metaA, metaB)
            continue

        elif choice == "q":
            break


if __name__ == "__main__":
    sys.exit(run_compare())
