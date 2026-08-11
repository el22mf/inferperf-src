from pathlib import Path
import tomllib
import importlib.util

def load_config():
    cfg_path = Path("inferperf.toml")
    if not cfg_path.exists():
        return None, []

    with cfg_path.open("rb") as f:
        data = tomllib.load(f)

    workload = data.get("workload")
    args = data.get("args", [])
    return workload, args


def list_workloads():
    folder = Path("inferperf/workloads")
    return [p for p in folder.glob("*.py") if p.name != "__init__.py"]


def parse_workload_args(argv):
    if len(argv) < 3:
        return None, []
    workload = normalize_workload_path(argv[2])
    args = argv[3:]
    return workload, args


def select_workload():
    workloads = list_workloads()
    print("Select a workload:\n")
    for i, w in enumerate(workloads, 1):
        print(f"{i}. {w.name}")
    print(f"{len(workloads)+1}. custom path")

    choice = input("Enter choice: ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(workloads):
        return str(workloads[int(choice)-1]), []
    else:
        return input("Enter path to workload: ").strip(), []


def load_workload(path):
    spec = importlib.util.spec_from_file_location("inferperf_workload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "run"):
        raise RuntimeError(f"Workload {path} does not define a run() function")

    return module

def normalize_workload_path(workload):
    p = Path(workload)

    # Case 1: user passed a bare name like "mobilenetv2"
    if not p.suffix and "/" not in workload:
        candidate = Path("inferperf/workloads") / f"{workload}.py"
        if candidate.exists():
            return str(candidate)

    # Case 2: user passed proper name like "mobilenetv2.py"
    if p.suffix == ".py" and "/" not in workload:
        candidate = Path("inferperf/workloads") / workload
        if candidate.exists():
            return str(candidate)

    # Case 3: user passed a full path
    if p.exists():
        return str(p)

    raise FileNotFoundError(f"Workload file not found: {workload}")

