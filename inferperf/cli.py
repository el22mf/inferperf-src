#!/usr/bin/env python3

import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

VERSION = "1.0.0"


def print_rich_help():
    if not RICH_AVAILABLE:
        return

    # Title banner
    title = "[bold cyan]InferPerf - Lightweight PMU-Driven Performance Analysis[/bold cyan]"
    console.print(Panel(title, expand=False, border_style="cyan"))

    # Description panel
    console.print(
        Panel(
            "[white]InferPerf diagnoses performance bottlenecks in ML workloads using perf stat.\n"
            "It identifies stalls, memory pressure, branch issues, and more — then recommends optimisations.[/white]",
            border_style="blue",
        )
    )

    # Commands table
    cmd_table = Table(title="Commands", show_header=True, header_style="bold magenta")
    cmd_table.add_column("Command", style="cyan", no_wrap=True)
    cmd_table.add_column("Description", style="white")

    cmd_table.add_row(
        "analyse",
        "Run full performance analysis: collect PMU metrics, classify bottleneck, generate recommendations."
    )
    cmd_table.add_row(
        "validate",
        "Compare improved workload vs cached baseline. Shows before/after metrics."
    )
    cmd_table.add_row(
        "compare",
        "Analyse two separate workloads and compare results."
    )

    console.print(cmd_table)

    # Options table
    opt_table = Table(title="Options", show_header=True, header_style="bold magenta")
    opt_table.add_column("Option", style="cyan", no_wrap=True)
    opt_table.add_column("Description", style="white")

    opt_table.add_row("-h, --help", "Show this help message")
    opt_table.add_row("-v, --version", "Show version information")

    console.print(opt_table)


def main():
    if len(sys.argv) < 2:
        print_rich_help()
        return

    cmd = sys.argv[1]

    # Help
    if cmd in ("-h", "--help", "help"):
        print_rich_help()
        return

    # Version
    if cmd in ("-v", "--version"):
        print(f"InferPerf version {VERSION}")
        return

    # Commands
    if cmd == "analyse":
        from inferperf.analyse import run_analyse
        from inferperf.utils.workloads import parse_workload_args
        workload, args = parse_workload_args(sys.argv)
        run_analyse(workload, args)
        return


    if cmd == "validate":
        from inferperf.validate import run_validate
        run_validate()
        return

    if cmd == "compare":
        from inferperf.compare import run_compare
        run_compare()
        return

    # Unknown command
    print(f"Unknown command: {cmd}")
    print("Use 'inferperf --help' to see available commands.")


if __name__ == "__main__":
    main()
