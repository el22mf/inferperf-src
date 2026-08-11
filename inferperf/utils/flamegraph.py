import os
import subprocess
import shutil
from pathlib import Path

from inferperf.utils.pmu import build_perf_record_cmd

FLAMEGRAPH_DIR = Path(__file__).parent / "flamegraph_scripts/FlameGraph"
STACK_COLLAPSE = FLAMEGRAPH_DIR / "stackcollapse-perf.pl"
FLAMEGRAPH_PL = FLAMEGRAPH_DIR / "flamegraph.pl"
FLAMEGRAPH_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "flamegraphs"

def run_perf_record(workload: str, args: list[str], perf_data_path: str) -> str:
    print(f"[InferPerf] Running perf record for workload: {workload}")

    cmd = build_perf_record_cmd(workload, args)
    cmd = ["sudo"] + cmd

    subprocess.run(cmd, check=True)

    # Fix permissions so Python can read perf.data
    subprocess.run([
        "sudo", "chown",
        f"{os.getuid()}:{os.getgid()}",
        "perf.data"
    ], check=True)

    shutil.move("perf.data", perf_data_path)
    return perf_data_path

def collapse_stacks(perf_data_path: str, folded_path: str) -> str:
    print("[InferPerf] Collapsing perf stacks...")

    with open(folded_path, "w") as out:
        p1 = subprocess.Popen(["perf", "script", "-i", perf_data_path], stdout=subprocess.PIPE)
        p2 = subprocess.Popen([str(STACK_COLLAPSE)], stdin=p1.stdout, stdout=out)
        p1.stdout.close()
        p2.communicate()

    return folded_path


def render_flamegraph(folded_path: str, svg_path: str) -> str:
    print("[InferPerf] Rendering flamegraph...")

    with open(svg_path, "w") as out:
        subprocess.run([str(FLAMEGRAPH_PL), folded_path], stdout=out, check=True)

    return svg_path


def generate_flamegraph(workload: str, args: list[str], svg_path: str) -> str:
    tmp_perf = FLAMEGRAPH_OUTPUT_DIR / "inferperf_perf.data"
    tmp_folded = FLAMEGRAPH_OUTPUT_DIR / "inferperf_folded.txt"


    perf_data = run_perf_record(workload, args, tmp_perf)
    folded = collapse_stacks(perf_data, tmp_folded)

    svg_file = FLAMEGRAPH_OUTPUT_DIR / svg_path
    svg = render_flamegraph(folded, str(svg_file))
    print(f"[InferPerf] Flamegraph generated: {svg}")
    return svg
