# InferPerf — PMU‑Driven AI Inference Profiling on Arm (IPC +20.53 uplift on Cortex‑A72)

InferPerf is a profiling toolkit that converts hardware-based performance counters into actionable optimisation guidance for AI workloads on Arm CPUs. It provides a reproducible **Analyse → Validate → Compare** workflow that helps engineers uncover microarchitectural inefficiencies and apply targeted fixes on Arm Cortex‑A devices.

---

## Reproducibility

| Software | Version |
|----------|---------|
| Python | **3.11.2** |
| ONNX Runtime | **1.18.0 (CPUExecutionProvider)** |
| Raspberry Pi OS | **2024‑05‑03 (64‑bit)** |
| Kernel | **6.6.20‑v8+** |
| perf | **6.6.20** |

| Hardware | Version |
|----------|---------|
| Platform | **Raspberry Pi 5 — Cortex‑A72 @ 2.4GHz, 8GB LPDDR4X** |
| SSD for Model I/O | **Fanxiang 128Gb SATA SSD** |

---

## Quick summary

InferPerf provides a structured way to understand why an AI workload behaves the way it does on Arm CPUs. Instead of treating perf output and flamegraphs as raw data, InferPerf turns them into a reproducible performance baseline, a bottleneck classification, and clear optimisation guidance. The included transformer encoder example simply demonstrates this workflow in action, showing how InferPerf can reveal inefficiencies and validate improvements such as the +20.53 IPC uplift seen later in this README.

---

## Quickstart

```bash
git clone <repo>
cd inferperf-src
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Baseline profile
inferperf analyse mobilenetv2 --warmup 3
```

---

## What InferPerf Does

InferPerf can be operated using three simple commands:

### Analyse
Profiles a workload with perf, collects PMU metrics, generates a flamegraph, classifies bottlenecks, and produces **recommendations** based on PMU behaviour.  Also writes a reproducible baseline cache used by `validate.py`.

```bash
inferperf analyse transformer_encoder --warmup 3
```

### Validate
Re-runs the previously analysed workload and compares fresh PMU data against the cached baseline, reporting deltas and bottleneck shifts.

```bash
inferperf analyse transformer_encoder --warmup 3
inferperf validate
```

### Compare
Profiles two workloads independently and presents a side‑by‑side microarchitectural comparison.

```bash
inferperf compare transformer_encoder transformer_encoder_optimised --warmup 3
```

---
## Optional interactive mode
InferPerf also includes an interactive mode for users who prefer not to specify workloads or warmup counts on the command line.
Running `inferperf analyse` or `inferperf compare` without arguments opens a guided selection menu.

<img width="600" height="160" alt="image" src="https://github.com/user-attachments/assets/99ba2ddf-34b5-4144-a1eb-15e2cfc53f53" />

*Example of guided selection menu prompt for the command `inferperf analyse`*

---

## Interface Overview

The `analyse.py` command produces a detailed diagnostic view combining PMU metrics, bottleneck classification, and optimisation recommendations.   The `validate.py` and `compare.py` commands produce similar outputs displaying PMU metrics across varying workloads.

<img width="452" height="561" alt="image" src="https://github.com/user-attachments/assets/97f99cdc-16da-4b12-90ea-d1e9d8c16c07" />

*Example output from `inferperf analyse` showing PMU metrics, bottleneck classification, recommendations, and available actions.*

---

## PMU Interpretation: Recommendations and Flamegraphs

InferPerf does more than collect PMU counters — it interprets them.  
The recommendations system is built on a set of semantic rules that map PMU behaviour to likely microarchitectural inefficiencies. For example:

- high `branch_miss_rate` → unstable control flow or non‑deterministic inputs  
- high `l1d_refill` / `l2d_refill` → poor locality or oversized working set  
- high `dtlb_misses` → irregular memory access patterns  
- low IPC with high instructions → Python overhead or fragmented execution  
- low IPC with high cycles → compute‑bound kernel

Each bottleneck classification produces both a short and long recommendation set tailored to the PMU symptoms. These recommendations highlight the most likely execution‑level fixes, helping engineers stabilise control flow, improve locality, and reduce Python or runtime overhead.

InferPerf also generates a flamegraph for every run. PMU counters tell you *what* is slow; flamegraphs show you *where* the time is spent. Together, they provide a complete picture: microarchitectural symptoms (PMU) and code‑level causes (flamegraph).

---

## Workloads and Models

InferPerf ships with two **demonstration workloads** to show how the toolkit detects bottlenecks, classifies behaviour, and validates improvements:

- `mobilenetv2.py` — MobileNetV2 inference  
- `transformer_encoder.py` — MiniLM‑L12 baseline (batch 4, seq 512)  
- `transformer_encoder_optimised.py` — Optimised transformer (batch 1, seq 128)

`mobilenetv2.py` exists as a validation of the tool, particularly during the generation of the `analyse.py` and `validate.py` scripts. `transformer_encoder_optimised.py`
is an optimised version of `transformer_encoder.py`, which was produced following the recommendations provided by the InferPerf toolkit. These two scripts
can be used to validate the `compare.py` script.

Included ONNX models:

- `mobilenetv2-7.onnx`  
- `minilm-l12.onnx`

---

## Demonstration: Transformer Encoder Optimisation
The following demonstration can be produced using the single commmand:
```bash
inferperf compare transformer_encoder transformer_encoder_optimised --warmup 3
```

<img width="600" height="390" alt="image" src="https://github.com/user-attachments/assets/f864a166-fea1-46f2-9d73-2674724008fd" />

*Output showing results of `compare,py` for the two demonstration workloads*

The optimisation in the demonstration workload falls into two areas:

- **Working‑set reduction**: 2048 → 128 tokens (16× fewer)
- **Execution‑level improvements**: +20.53 IPC uplift and large reductions in cache, TLB, and branch misses

| Category | What changed | Why it matters |
|----------|--------------|----------------|
| **Workload‑level** | batch 4→1, seq 512→128 (16× fewer tokens) | reduces total compute; explains part of cycles/instructions drop |
| **Execution‑level** | deterministic inputs, preallocation, persistent session, hot‑path minimisation | improves per‑token efficiency, IPC, locality, and branch predictor stability |


InferPerf identifies **execution-level** inefficiencies such as unstable branch behaviour, allocator churn, repeated runtime initialisation, and non-deterministic inputs. In the demonstration workload, these surfaced as recommendations like deterministic token IDs, deterministic attention masks, tensor preallocation, persistent ONNX Runtime sessions, cached input dictionaries, removal of randomness, and hot-path minimisation. These are examples of the kinds of fixes InferPerf guides engineers toward, and they are what drive the IPC uplift and the reductions in cache, TLB, and branch misses shown in the screenshot.


**Note:** The per‑token values shown in the screenshot naturally rise in the optimised workload because it processes 16× fewer tokens, meaning fixed ONNX Runtime overhead is divided by a much smaller token count. This behaviour is expected and not a regression; the meaningful efficiency indicator is IPC, which increases substantially in the optimised run.

---

## Design Decisions and Arm Focus

InferPerf is designed around the realities of Arm microarchitecture:

- **Working‑set reduction** improves L1/L2 residency and reduces TLB churn  
- **Preallocation** avoids allocator syscalls  
- **Persistent runtime** exposes steady‑state IPC  
- **Input regularisation** stabilises branch predictors  
- **Hot‑path minimisation** concentrates execution in optimised native code paths  

InferPerf is intended to complement Arm Performix, rather than compete with it. Performix focuses on operator‑level profiling - it tells you which parts of the model are slow. InferPerf looks at the CPU itself, showing you why the workload is struggling at the microarchitectural level. Used together, they give you the full picture: Performix explains the model’s behaviour, and InferPerf explains the processor’s.

--- 

## Developer Experience and Reuse

### Add a new workload
1. Store model in `models/<name.py`
2. Create `workloads/<name>.py`  
3. Run Analyse

### Artifacts included
- Deterministic workloads  
- ONNX models  
- Flamegraphs  
- Baseline cache schema

---

## Appendix — PMU Events and Cache Schema

InferPerf exposes a small, stable data model that other tools or workflows can build on.  
This appendix documents the **formal interface** of the toolkit: the PMU event set it guarantees to collect, and the structure of the baseline cache used for reproducible validation and comparison.

### PMU event set
InferPerf standardises on a minimal, Arm‑portable `perf` event set that is available across Cortex‑A devices.  The goal of this is to promote consistent behaviour across as many different platforms as possible.

`cycles, instructions, cache_misses, l1d_refill, l2d_refill, branch_misses, dtlb_misses, itlb_misses`

These events form the core of InferPerf’s bottleneck classifier (IPC, memory locality, branch stability, TLB behaviour).

### Baseline cache schema
InferPerf stores analysis results in a structured cache so future runs can be validated deterministically.  
This schema is intentionally simple, making it easy to inspect, diff, or integrate into external tooling.

- **workload_name** — identifier for the workload  
- **timestamp** — when the baseline was generated  
- **pmu_metrics** — raw PMU totals collected during Analyse  
- **flamegraph_path** — path to the generated flamegraph  
- **bottleneck_classification** — InferPerf’s interpretation of the PMU data  
- **workload_metadata** — deterministic metadata (batch size, sequence length, model name, etc.)

This appendix exists so users can see the **exact data contract** InferPerf provides — useful for reproducibility, auditing, and extending the toolkit.

---

# **License**
MIT License
