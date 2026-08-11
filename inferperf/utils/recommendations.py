RECOMMENDATIONS_TABLE = {
    "Frontend-bound": [
        "Reduce I-cache pressure; fuse kernels to reduce instruction footprint.",
        "Avoid Python-level branching; use vectorised operations.",
    ],
    "Backend-bound": [
        "Increase ILP via loop unrolling or kernel fusion.",
        "Use NEON/SIMD to widen execution and reduce backend stalls.",
    ],
    "Memory-bound": [
        "Improve cache locality (tiling, blocking, data layout).",
        "Reduce working set size or use fused kernels.",
    ],
    "Branch-bound": [
        "Reduce branch divergence or simplify control flow.",
        "Use branchless operations or vectorised conditionals.",
    ],
    "ILP-limited": [
        "Increase ILP via kernel fusion or loop unrolling.",
        "Use SIMD/NEON vectorisation to widen execution.",
    ],
    "TLB-bound": [
        "Reduce pointer chasing; use contiguous tensors.",
        "Reduce working set size or use static shapes.",
        "Fuse kernels to reduce page-table walks.",
    ],
    "L1-bound": [
        "Reduce tile size; L1 thrash detected.",
        "Improve data layout for spatial locality.",
    ],
    "L2-bound": [
        "Fuse kernels to reduce L2 pressure.",
        "Reduce working set size or use blocking.",
    ],
    "Compute-bound": [
        "Workload is compute-bound; consider algorithmic optimisation.",
        "Try FP16 or INT8 inference for higher throughput.",
    ],
}

MORE_RECOMMENDATIONS_TABLE = {
    "Frontend-bound": [
        "The CPU stalls before execution because the instruction stream is too fragmented — common in Python-heavy inference loops or graphs with many tiny ops. The frontend can't keep the instruction window full, so the backend sits idle.",
        "- Reduce I-cache pressure: fusing ops keeps the instruction footprint small enough for L1I, preventing fetch stalls that break operator fusion benefits and slow down kernel dispatch.",
        "- Avoid Python-level branching: unpredictable control flow forces the frontend to re-decode instructions; replacing Python conditionals with vectorised or fused paths keeps the decode stage saturated.",
    ],

    "Backend-bound": [
        "Instructions reach the backend, but they retire slowly due to dependency chains or saturated execution units. This is common in scalar-heavy kernels or models with sequential math.",
        "- Increase ILP: unrolling loops or fusing ops exposes independent instructions the backend can overlap, reducing stalls caused by serialized tensor operations.",
        "- Use NEON/SIMD: widening execution turns multiple scalar tensor ops into one vector op, reducing backend pressure and smoothing retire-rate fluctuations.",
    ],

    "Memory-bound": [
        "The CPU spends most of its time waiting on data — not computing. Large tensors, poor locality, or strided access patterns cause cache misses that dominate inference time.",
        "- Improve cache locality: tiling/blocking keeps active tensor regions in L1/L2, reducing DRAM trips that stall transformer attention blocks and convolution loops.",
        "- Reduce working set size: fusing ops or shrinking intermediate tensors helps them fit into cache, preventing refill storms that slow down matmul-heavy workloads.",
    ],

    "Branch-bound": [
        "Branch mispredicts flush the pipeline, forcing the CPU to restart work. This happens when control flow depends on data or Python-level logic.",
        "- Reduce branch divergence: simplifying conditionals gives the branch predictor stable patterns, reducing flushes that disrupt fused kernels or dynamic-shape inference.",
        "- Use branchless operations: vectorised conditionals avoid mispredicts entirely, keeping the pipeline full during activation functions or elementwise ops.",
    ],

    "ILP-limited": [
        "The pipeline has idle slots because the workload doesn't expose enough parallel work. This is common in operator graphs that do many small sequential steps.",
        "- Increase ILP: fusing ops or unrolling loops gives the scheduler multiple independent tensor ops to issue, reducing stalls caused by long dependency chains in matmuls or reductions.",
        "- Use SIMD/NEON: widening execution increases parallelism per instruction, filling pipeline slots that would otherwise sit empty during elementwise or reduction-heavy kernels.",
    ],

    "TLB-bound": [
        "The workload touches too many pages, causing TLB misses and expensive page-table walks — especially painful on ARM cores with small TLBs.",
        "- Use contiguous tensors: linear layouts reduce page churn and keep TLB entries hot, avoiding page-table walks that slow down attention blocks and embedding lookups.",
        "- Reduce working set size: fewer pages touched means fewer TLB misses, improving stability in models with large activation maps.",
        "- Fuse kernels: fewer intermediate tensors means fewer pages accessed, reducing TLB pressure during multi-stage ops.",
    ],

    "L1-bound": [
        "L1 is tiny, and if your tiles or tensors don't fit, you'll thrash it constantly — causing a cascade of misses into L2 and DRAM.",
        "- Reduce tile size: right-sizing tiles ensures they fit in L1, preventing thrash cycles that slow down convolution loops or batched matmuls.",
        "- Improve spatial locality: linear access patterns keep L1 lines hot, reducing refills during activation functions or fused elementwise ops.",
    ],

    "L2-bound": [
        "L2 is larger but still limited, and once you overflow it, you start hitting DRAM — which is slow and unpredictable.",
        "- Fuse kernels: reduces intermediate tensors that spill out of L2, smoothing memory access during multi-stage transformer blocks.",
        "- Use blocking: keeps working sets inside L2, reducing refill storms during large matmuls or attention score computations.",
    ],

    "Compute-bound": [
        "The ALUs/FMA units are saturated — the CPU is genuinely maxed out. Memory or branching optimisations won't help here.",
        "- Algorithmic optimisation: reducing FLOPs directly lowers pressure on compute units, which is the only meaningful lever when matmuls or convolutions dominate runtime.",
        "- Use FP16/INT8: lower precision increases throughput dramatically by reducing per-op cost and improving vector width utilisation in NEON-heavy inference paths.",
    ],
}
