#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, Any

from inferperf.utils.recommendations import RECOMMENDATIONS_TABLE


def classify_bottleneck(pmu: Dict[str, float]) -> Dict[str, Any]:
    cycles = pmu.get("cycles", 1.0)
    ipc = pmu.get("ipc", 0.0)
    cache_miss_rate = pmu.get("cache_miss_rate", 0.0)
    branch_miss_rate = pmu.get("branch_miss_rate", 0.0)

    l1d_rate = pmu.get("l1d_rate", 0.0)
    l2d_rate = pmu.get("l2d_rate", 0.0)

    stall_fe = pmu.get("stall_frontend", 0.0)
    stall_be = pmu.get("stall_backend", 0.0)

    dtlb = pmu.get("dtlb_misses", 0.0)
    itlb = pmu.get("itlb_misses", 0.0)
    tlb_miss_rate = pmu.get("tlb_miss_rate", 0.0)

    # Stall scores
    fe_score = min(stall_fe / cycles, 1.0)
    be_score = min(stall_be / cycles, 1.0)

    # Memory pressure
    mem_score = min(cache_miss_rate / 0.30, 1.0)

    # L scores
    l1_score = min(l1d_rate * 50, 1.0)   # L1 refills are cheap so scale harder
    l2_score = min(l2d_rate * 20, 1.0)   # L2 refills are moderate

    # Branch pressure
    branch_score = min(branch_miss_rate / 0.05, 1.0)

    # ILP pressure
    ilp_score = 1.0 - min(ipc / 1.5, 1.0)

    # TLB pressure (very expensive →so scale hard)
    tlb_score = min(tlb_miss_rate * 20, 1.0)

    # Compute efficiency
    compute_score = max(
        0.0,
        1.0 - sum([
            fe_score, be_score, mem_score, branch_score,
            ilp_score, tlb_score, l1_score, l2_score,
        ]) / 9
    )


    scores = {
        "Frontend-bound": fe_score,
        "Backend-bound": be_score,
        "Memory-bound": mem_score,
        "Branch-bound": branch_score,
        "ILP-limited": ilp_score,
        "TLB-bound": tlb_score,
        "Compute-bound": compute_score,
        "L1-bound": l1_score,
        "L2-bound": l2_score,
    }

    bottleneck_type = max(scores, key=scores.get)
    confidence = round(scores[bottleneck_type], 3)

    all_bottlenecks = sorted(
        (
            {"type": k, "confidence": round(v, 3)}
            for k, v in scores.items()
        ),
        key=lambda x: x["confidence"],
        reverse=True
    )

    evidence = {
        "stall_frontend": stall_fe,
        "stall_backend": stall_be,
        "frontend_pressure": fe_score > 0.5,
        "backend_pressure": be_score > 0.5,
        "memory_pressure": mem_score > 0.5,
        "branch_pressure": branch_score > 0.5,
        "ilp_pressure": ilp_score > 0.5,
        "tlb_pressure": tlb_score > 0.5,

        "dtlb_misses": dtlb,
        "itlb_misses": itlb,
        "tlb_miss_rate": tlb_miss_rate,

        "ipc": ipc,
        "cache_miss_rate": cache_miss_rate,
        "branch_miss_rate": branch_miss_rate,

        "l1d_refill": pmu.get("l1d_refill", 0.0),
        "l2d_refill": pmu.get("l2d_refill", 0.0),
        "l1d_rate": l1d_rate,
        "l2d_rate": l2d_rate,


        "l1_pressure": l1_score > 0.5,
        "l2_pressure": l2_score > 0.5,

    }

    return {
        "bottleneck": {
            "type": bottleneck_type,
            "confidence": confidence,
            "evidence": evidence,
        },
        "recommendations": RECOMMENDATIONS_TABLE[bottleneck_type],
         "all_bottlenecks": all_bottlenecks, 
    }
