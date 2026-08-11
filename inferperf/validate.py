#!/usr/bin/env python3
"""
validate.py
- Validate and compare a fresh profiled run against the cached baseline produced by analyse.py.
- Operates on the same workload recognised via cached path
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import tomllib
except Exception:
    tomllib = None

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
from inferperf.utils.flamegraph import generate_flamegraph
from inferperf.utils.workloads import load_workload, normalize_workload_path

CACHE_PATH = Path(".inferperf_cache.toml")

INT_THRESHOLDS = {
    "cycles": 600_000_000,
    "instructions": 800_000_000,
    "dtlb_misses": 500_000,
}

PP_THRESHOLDS = {
    "ipc": 0.10,               # 0.10 percentage points
    "cache_miss_rate": 0.05,   # 0.05 pp
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


def colour_delta(delta, threshold):
    if abs(delta) < threshold:
        return f"[white]{delta:+.2f}[/white]"
    if delta > 0:
        return f"[green]{delta:+.2f}[/green]"
    return f"[red]{delta:+.2f}[/red]"


def colour_delta_int(delta, threshold):
    if abs(delta) < threshold:
        return f"[white]{delta:+,}[/white]"
    if delta > 0:
        return f"[green]{delta:+,}[/green]"
    return f"[red]{delta:+,}[/red]"


def colour_semantic(metric, delta, threshold):
    """
    Apply semantic colouring:
    - Green = performance improvement
    - Red   = performance regression
    - White = insignificant change
    """
    rule = SEMANTIC_RULES.get(metric, "neutral")

    if abs(delta) < threshold:
        return f"[white]{delta:+.2f}[/white]"

    if rule == "neutral":
        return f"[white]{delta:+.2f}[/white]"

    if rule == "higher_is_good":
        return f"[green]{delta:+.2f}[/green]" if delta > 0 else f"[red]{delta:+.2f}[/red]"

    if rule == "higher_is_bad":
        return f"[red]{delta:+.2f}[/red]" if delta > 0 else f"[green]{delta:+.2f}[/green]"

    # fallback
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


def load_cache(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    if tomllib is None:
        raise RuntimeError("tomllib not available")
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data


def format_pct(v):
    return f"{v*100:.2f}%"


def render_side_by_side(baseline, current, baseline_meta, current_meta):
    keys = [
        # Core Execution
        ("cycles", "cycles", "Core Execution"),
        ("instructions", "instructions", None),
        ("ipc", "ipc", None),

        # Memory Indicators
        ("cache_misses", "cache_misses", "Memory Indicators"),
        ("cache_miss_rate", "cache_miss_rate", None),
        ("l1d_refill", "l1d_refill", None),
        ("l1d_rate", "l1d_rate", None),
        ("l2d_refill", "l2d_refill", None),
        ("l2d_rate", "l2d_rate", None),

        # Branch Indicators
        ("branch_misses", "branch_misses", "Branch Indicators"),
        ("branch_miss_rate", "branch_miss_rate", None),

        # TLB Indicators
        ("dtlb_misses", "dtlb_misses", "TLB Indicators"),
        ("itlb_misses", "itlb_misses", None),
        ("tlb_miss_rate", "tlb_miss_rate", None),
    ]


    if RICH_AVAILABLE:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Baseline", style="grey70", no_wrap=True)
        table.add_column("Current", style="grey70", no_wrap=True)
        table.add_column("Delta", style="white", no_wrap=True)


        current_section = None

        for key, label, section in keys:
            if section and section != current_section:
                # Add a section header row
                table.add_row(f"[bold white]{section}[/bold white]", "", "", "")
                current_section = section

            b = baseline.get(key, 0)
            c = current.get(key, 0)

            # percentage‑point metrics
            if key.endswith("_rate") or key == "ipc":
                b_str = format_pct(b)
                c_str = format_pct(c)
                delta_pp = (c - b) * 100
                threshold = PP_THRESHOLDS.get(key, 0.10)
                delta_str = f"[bold]{colour_semantic(key, delta_pp, threshold)}[/bold]"


            # integer metrics
            else:
                b_str = f"{b:,}"
                c_str = f"{c:,}"
                delta_int = c - b
                threshold = INT_THRESHOLDS.get(key, 500_000)
                delta_str = f"[bold]{colour_semantic_int(key, delta_int, threshold)}[/bold]"

            table.add_row(label, b_str, c_str, delta_str)


        console.print(Panel(table, title="Baseline vs Current (side-by-side)", border_style="cyan"))

        # Bottleneck panels
        b_bneck = baseline_meta.get("bottleneck", {})
        c_bneck = current_meta.get("bottleneck", {})

        left = Panel(f"[bold white]{b_bneck.get('type','unknown')}[/bold white]\nConfidence: {b_bneck.get('confidence',0.0):.2f}",
                     title="Baseline Bottleneck", border_style="green")
        right = Panel(f"[bold white]{c_bneck.get('type','unknown')}[/bold white]\nConfidence: {c_bneck.get('confidence',0.0):.2f}",
                      title="Current Bottleneck", border_style="yellow")

        console.print(left)
        console.print(right)

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

   
def validate_action_bar():
    console.print(
        Panel(
            Text.from_markup(
                "[bold cyan]Next Actions[/bold cyan]\n\n"
                "[bold green][F][/bold green] View Optimised Flamegraph\n"
                "[bold magenta][M][/bold magenta] Show Metadata\n"
                "[bold cyan][R][/bold cyan] Reset\n"
                "[bold red][Q][/bold red] Quit"
            ),
            border_style="cyan"
        )
    )

    return Prompt.ask(
        "Choose an action",
        choices=["f", "m", "r", "q"],
        default="r"
    )


def _get_warmup_from_args(args):
    if not isinstance(args, list):
        return 0
    if "--warmup" in args:
        try:
            idx = args.index("--warmup")
            if idx + 1 < len(args):
                return int(args[idx + 1])
        except Exception:
            return 0
    return 0


def _prompt_warmup(default=1):
    try:
        resp = input(f"Warm-up runs [0–3] (default {default}): ").strip()
        if resp == "":
            return default
        val = int(resp)
        if val < 0:
            return 0
        if val > 3:
            return 3
        return val
    except Exception:
        return default


def run_validate():
    # 1. Load cache
    try:
        cache = load_cache(CACHE_PATH)
    except FileNotFoundError:
        print(f"No cache found at {CACHE_PATH}. Run `inferperf analyse` first to create a baseline.")
        return 2
    except Exception as e:
        print(f"Failed to read cache: {e}")
        return 3

    baseline_workload = cache.get("workload")
    baseline_args = cache.get("args", [])
    baseline_pmu = cache.get("pmu", {})
    baseline_classification = {
        "bottleneck": cache.get("bottleneck", {}),
        "all_bottlenecks": cache.get("all_bottlenecks", [])
    }
    baseline_metadata = cache.get("metadata", {})

    # 2. Display baseline summary
    if RICH_AVAILABLE:
        console.print(Panel(f"[bold cyan]Cached baseline from analyse.py[/bold cyan]\nWorkload: {baseline_workload}\nTimestamp: {baseline_metadata.get('timestamp','unknown')}", border_style="green"))
    else:
        print("Cached baseline from analyse.py")
        print("Workload:", baseline_workload)
        print("Timestamp:", baseline_metadata.get("timestamp", "unknown"))

    # 3. Determine warmup strategy (prefer same as cached)
    baseline_warmup = _get_warmup_from_args(baseline_args)
    if baseline_warmup > 0:
        warmup = baseline_warmup
        if RICH_AVAILABLE:
            console.print(f"[cyan]Using warmup count from baseline: {warmup} run(s)[/cyan]")
        else:
            print(f"Using warmup count from baseline: {warmup} run(s)")
    else:
        warmup = _prompt_warmup(default=1)
        if RICH_AVAILABLE:
            console.print(f"[cyan]Using user-selected warmup: {warmup} run(s)[/cyan]")
        else:
            print(f"Using user-selected warmup: {warmup} run(s)")

    # If warmup > 0, load the workload and run warmups
    if warmup > 0:
        try:
            workload_path = normalize_workload_path(baseline_workload)
            workload = load_workload(workload_path)
        except Exception as e:
            print(f"Failed to load workload for warmup: {e}")
            return 4

        if RICH_AVAILABLE:
            console.print(f"[cyan]Running {warmup} warm-up pass(es)...[/cyan]")
        else:
            print(f"Running {warmup} warm-up pass(es)...")

        try:
            for i in range(warmup):
                if RICH_AVAILABLE:
                    console.print(f"[cyan]Warm-up run {i+1}/{warmup}[/cyan]")
                workload.run()
        except Exception as e:
            print(f"Warmup run failed: {e}")
            return 5

    # 4. Run fresh PMU collection for the same workload
    try:
        if RICH_AVAILABLE:
            console.print("[cyan]Running fresh PMU collection...[/cyan]")
        else:
            print("Running fresh PMU collection...")
        pmu_result = run_pmu(baseline_workload, baseline_args)
        current_pmu = pmu_result.get("pmu", {})
    except Exception as e:
        print(f"Failed to run PMU on workload {baseline_workload}: {e}")
        return 6

    # 5. Classify current run
    current_classification = classify_bottleneck(current_pmu)

    current_metadata = baseline_metadata.copy()
    current_metadata["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    current_metadata["bottleneck"] = current_classification.get("bottleneck", {})
    current_metadata["workload_file"] = baseline_workload


    # 6. Create flamegraph
    svg_path = generate_flamegraph(baseline_workload, baseline_args, "optimised_flamegraph.svg")

    # 7. Cache and render
    baseline = baseline_pmu
    current = current_pmu


    render_side_by_side(
        baseline,
        current,
        {
            "bottleneck": baseline_classification.get("bottleneck", {}),
            "timestamp": baseline_metadata.get("timestamp"),
            "workload_file": baseline_metadata.get("workload_file", baseline_workload),
        },
        current_metadata
    )

    while True:
        choice = validate_action_bar()

        if choice == "f":
            console.clear()
            subprocess.run(["xdg-open", svg_path])
            console.clear()
            continue

        elif choice == "m":
            console.clear()
            render_metadata(current_metadata)
            continue

        elif choice == "r":
            console.clear()
            render_side_by_side(
                    baseline,
                    current,
                    {
                        "bottleneck": baseline_classification.get("bottleneck", {}),
                        "timestamp": baseline_metadata.get("timestamp"),
                        "workload_file": baseline_metadata.get("workload_file", baseline_workload),
                    },
                    current_metadata
                )
            continue

        elif choice == "q":
            break


if __name__ == "__main__":
    sys.exit(run_validate())
