#!/usr/bin/env python3
import subprocess
import shlex
import sys

# PMU events chosen to collect that don't break everything
EVENTS = [
    "cycles",
    "instructions",
    "cache-misses",
    "cache-references",
    "branch-misses",
    "dTLB-load-misses",
    "iTLB-load-misses",
    "l1d_cache_refill",
    "l2d_cache_refill",
]


def _build_perf_cmd(workload: str, args) -> list[str]:
    python_exe = sys.executable
    PERF_BIN = "/home/pi/linux/tools/perf/perf"

    return [
        "sudo", PERF_BIN, "stat",
        "-x", ",",
        "-e", ",".join(EVENTS),
        "--",
        python_exe, workload,
    ]

def build_perf_record_cmd(workload: str, args) -> list[str]:
    python_exe = sys.executable
    PERF_BIN = "/home/pi/linux/tools/perf/perf"

    return [
        "sudo", PERF_BIN, "record",
        "-F", "99",
        "-g",
        "--",
        python_exe, workload,
    ]

def _parse_perf_csv(stderr: str) -> dict:
    counters = {}

    for line in stderr.splitlines():
        line = line.strip()
        if not line:
            continue

        # Skip perf's summary lines
        if "seconds time elapsed" in line:
            continue

        # ARM perf often prints big spaces eg "99999      cycles"
        if "," not in line:
            parts = line.split()
            if len(parts) >= 2:
                value_str = parts[0]
                event = parts[-1]

                # Skip unsupported events
                if "<" in value_str:
                    continue

                try:
                    value = float(value_str.replace(",", ""))
                except ValueError:
                    continue

                counters[event] = value
            continue

        # CSV-style - value,unit,event
        parts = line.split(",")
        if len(parts) >= 3:
            value_str, unit, event = parts[0], parts[1], parts[2]

            if "<" in value_str:
                continue

            try:
                value = float(value_str)
            except ValueError:
                continue

            counters[event] = value

    return counters


def _derive_metrics(c: dict) -> dict:
    cycles = c.get("cycles", 1.0)
    instructions = c.get("instructions", 0.0)

    ipc = instructions / cycles if cycles else 0.0

    # Branch predictor quality
    branch_misses = c.get("branch-misses", 0.0)
    branch_miss_rate = branch_misses / instructions if instructions else 0.0

    # Cache information
    cache_misses = c.get("cache-misses", 0.0)
    cache_refs = c.get("cache-references", 0.0)
    cache_miss_rate = cache_misses / cache_refs if cache_refs else 0.0

    l1d_refill = c.get("l1d_cache_refill", 0.0)
    l2d_refill = c.get("l2d_cache_refill", 0.0)

    l1d_rate = l1d_refill / cycles if cycles else 0.0
    l2d_rate = l2d_refill / cycles if cycles else 0.0

    # TLB information
    dtlb_misses = c.get("dTLB-load-misses", 0.0)
    itlb_misses = c.get("iTLB-load-misses", 0.0)

    tlb_miss_rate = (
        (dtlb_misses + itlb_misses) / cycles
        if cycles else 0.0
    )


    return {
        "cycles": cycles,
        "instructions": instructions,
        "ipc": ipc,

        "cache_misses": cache_misses,
        "cache_miss_rate": cache_miss_rate,

        "l1d_refill": l1d_refill,
        "l2d_refill": l2d_refill,

        "l1d_rate": l1d_rate,
        "l2d_rate": l2d_rate,

        "branch_misses": branch_misses,
        "branch_miss_rate": branch_miss_rate,

        "dtlb_misses": dtlb_misses,
        "itlb_misses": itlb_misses,
        "tlb_miss_rate": tlb_miss_rate,
    }


def run_pmu(workload: str, args: list[str]) -> dict:
    cmd = _build_perf_cmd(workload, args)

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("PERF STAT FAILED:")
        print("Command:", " ".join(shlex.quote(x) for x in cmd))
        print("stderr:")
        print(e.stderr)

        return {
            "pmu": {},
            "error": e.stderr,
            "perf_command": " ".join(shlex.quote(x) for x in cmd),
        }


    # Combine stdout and stderr in case either is used
    combined = proc.stdout + "\n" + proc.stderr

    raw = _parse_perf_csv(combined)
    pmu = _derive_metrics(raw)

    return {
        "pmu": pmu,
        "perf_command": " ".join(shlex.quote(x) for x in cmd),
    }
