# ADR-031 Phase C — step C0 diagnostic profile findings

**Date**: 2026-05-15
**HEAD**: post-`9e2ee851` (Phase C design analysis)
**Run**: tg2000 1-cycle with `HF2Q_PARALLEL_ENCODE=1` + `HF2Q_PARALLEL_PROFILE=1`
**Result**: bottleneck is downstream of CPU encoding — neither Direction C-α nor C-β will close the gap

## Per-token phase timings (10 samples shown; representative)

| Phase | µs | Annotation |
|---|---|---|
| channel construct | 0 | mpsc::channel() is negligible — C-β-1 (hoist) is NOT the lever |
| submit() | 1-2 | submit_to_global_worker is negligible |
| session_a.begin() (worker) | 1-2 | exec.begin() is negligible — C-β-2 (pre-allocate) is NOT the lever |
| encode_a (worker, ~13 layers) | 310-330 | well-balanced with chunk-B |
| encode_b (main, ~13 layers) | 280-320 | well-balanced with chunk-A |
| commit_a (worker) | 6-12 | tiny — C-α (enqueue API) saves at most ~10 µs/token = ~0.1% |
| recv_blocked (main waits worker) | 10-50 | parallelism IS realized — main waits only briefly |
| **total_helper** | **290-360** | total overhead from chunk-A+B parallel encode |

## Theoretical CPU savings

- Serial encode of layers 4..29 (~26 layers × 24 µs/layer): ~624 µs/token
- Parallel encode (max(encode_a, encode_b) + ~30 µs sync overhead): ~330 µs/token
- **CPU savings from parallelism: ~290 µs/token = ~2.7% theoretical speedup at 93 t/s**

## Measured wall-clock at tg2000 (Phase B v3 bench)

- PARALLEL=0: 92.93 ± 0.32 t/s
- PARALLEL=1: 92.43 ± 0.47 t/s
- **Δ −0.50 t/s = −0.54% — opposite sign from the theoretical 2.7% gain**

## Gap analysis

CPU profiling proves the CPU-side parallelism is realized as expected (encode_a ≈ encode_b, recv_blocked tiny). The **CPU is NOT the bottleneck**.

The measured wall-clock regression of −0.54% (vs theoretical +2.7%) means the parallel path loses ~3.2% of theoretical gain somewhere AFTER the CPU encode phase. Candidates:

1. **Metal GPU dispatch ordering**: with 2 CBs in flight, the GPU queue scheduler may add cross-CB latency that doesn't exist for 1-CB serial decode.
2. **Command-buffer submission overhead**: 2 commit() calls vs 1 commit() — Metal's submit path may have fixed per-CB cost (kernel argument upload, residency-set flush, etc.) that adds up.
3. **Memory-bus contention** (R-C1, HIGH): on Apple unified memory + shared cache, 2 concurrent encoders writing to disjoint kv_caches may stall on memory-bus. The GPU executes one CB at a time, but encoding-time memory pressure from the second thread could slow even the in-flight CB.
4. **Cache thrashing**: the worker thread's encoder state + main thread's encoder state both compete for L1/L2 caches.

## Why Phase C's planned levers don't help

**Direction C-α (`GraphSession::enqueue()`)**: saves at most commit_a time (~8 µs/token = 0.08%). Doesn't address GPU-side or memory-bus overhead.

**Direction C-β (CPU overhead reduction)**:
- C-β-1 (hoist mpsc): channel construct is already 0 µs. No savings possible.
- C-β-2 (pre-allocate session): begin() is already 1-2 µs. Negligible savings.
- C-β-3 (skip profile clone): profile is None in production path; clone is already cheap. Negligible savings.
- C-β-4 (memory-bus thread affinity): exploratory; potential mitigation for R-C1 but requires Apple Instruments + multi-day investigation.

## Honest verdict

**Phase C with the originally-planned levers will NOT deliver the ≥+2% gain needed for default-flip.** The architectural premise of Phase B (parallelize CPU encoding to overlap with serial GPU execution) doesn't pay off on hf2q-on-M5-Max because the CPU encoding is small relative to the GPU work, and the 2-CB overhead in some form (dispatch order, submission, memory) eats the theoretical savings.

This is exactly the R-C1 risk the Phase C design doc flagged as HIGH: "profiling may reveal the bottleneck is structural to having 2 concurrent encoders on Apple unified-memory, not amenable to either C-α or C-β."

## Recommendations

**Option 1 (recommended): close ADR-031 as Phase B terminal**
- Phase A landed (encode_one_layer extraction — a useful refactor regardless of parallel-encode)
- Phase B landed as opt-in scaffolding (default-OFF means production unaffected)
- Phase C abandoned: profiling shows the planned levers don't work; further investigation is multi-day and uncertain
- Update ADR-031 status: "Phase A + B landed; Phase C abandoned per C0 profiling"
- Remove HF2Q_PARALLEL_ENCODE env knob from the public API (or document as "experimental, no benefit measured")? — operator decision

**Option 2: deeper Phase C investigation via Apple Instruments**
- Run hf2q under Instruments Metal trace
- Compare PARALLEL=0 vs PARALLEL=1 GPU timeline
- Identify whether the 3.2% gap lives in GPU dispatch order, submission cost, or memory contention
- 1-2 days of exploratory work; may or may not yield a fix
- Even if root cause identified, fix may be in mlx-native (significant) or Apple-specific (no fix)

**Option 3: abandon Phase B itself**
- Revert merge `83c3ea6d`
- ADR-031 closes with only Phase A's refactor as durable work
- Removes the unsafe block + env knobs from main as "we tried, no win, cleaner to remove than carry"

My recommendation: **Option 1** (close Phase C as profiled-and-rejected; Phase B stays as default-OFF opt-in for future experimentation). Phase A's encode_one_layer refactor is valuable on its own.

## Phase C instrumentation legacy

The diagnostic code added in this iteration (`HF2Q_PARALLEL_PROFILE=1`) is kept in the codebase as a permanent tool. Future Phase C investigators (or anyone debugging parallel-encode behavior) can use it without re-implementing instrumentation.

## References

- `/opt/hf2q/docs/research/ADR-031-phase-C-design-analysis.md` — Phase C design that predicted this outcome as R-C1
- Phase B v3 bench: `/opt/hf2q/docs/research/ADR-031-phase-B-iter2-bench-results.md`
- Profile run log: `/tmp/parallel-profile-c0.log`
- Profile instrumentation: `src/serve/forward_mlx.rs` (HF2Q_PARALLEL_PROFILE gate)
