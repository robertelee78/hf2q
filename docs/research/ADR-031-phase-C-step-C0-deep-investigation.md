# ADR-031 Phase C — step C0 DEEP investigation (SPLIT_TIMING reveal)

**Date**: 2026-05-15
**HEAD**: post-`a6fdf252` (HF2Q_PARALLEL_PROFILE instrumentation)
**Method**: HF2Q_PARALLEL_ENCODE={0,1} × HF2Q_SPLIT_TIMING=1, single-cycle tg200 each (warm-up + ~28 samples per arm)
**Result**: the critical path is GPU (94% of decode wall), not CPU. No CPU-side lever can move the needle by more than ~1%.

## Body section comparison (representative samples, warmed up)

| Mode | CPU body encode | GPU body wait | Dispatches | Barriers |
|---|---|---|---|---|
| PARALLEL=0 | ~0.55 ± 0.06 ms | ~9.05 ± 0.18 ms | 866 | 420 |
| PARALLEL=1 | ~0.53 ± 0.04 ms | ~9.05 ± 0.18 ms | 866 | 420 |

**Three identical metrics: dispatch count (866), barrier count (420), GPU body wait (9.05ms).**  Phase B's parallel encode adds zero work to the GPU pipeline — Metal sees the same total work regardless of whether one or two CBs ferry it.

The only delta is CPU body encode: PARALLEL=1 is ~0.02 ms faster per token (~3.6% reduction in CPU body time).

## Wall-clock math

Decode token wall ≈ CPU body encode + GPU body wait + (head section, ~0.5-1 ms)

CPU body encode is ~0.55 ms = **~5% of decode wall**.
GPU body wait is ~9.05 ms = **~85% of decode wall**.
Head section is ~1 ms = **~10% of decode wall**.

**Halving CPU body encode (the theoretical maximum gain from CPU-parallel encoding) yields ~2.5% wall speedup at best.**  The actual measured CPU encode reduction is 3.6% of CPU encode time = 0.18% of wall time — **completely below the noise floor of the 5-cycle alt-pair bench (σ ≈ 0.5% per arm)**.

This explains Phase B v3's perf result: Δ +0.06 t/s at tg100 (where threshold gates parallel OFF), Δ −0.50 t/s at tg2000 (within σ).  The expected gain from CPU parallelism is fundamentally **at-or-below the measurement noise floor** on this hardware + model.

## Why neither C-α nor C-β nor R-C1 helps

The Phase C design doc considered three levers:

- **C-α (`GraphSession::enqueue()`)**: saves commit_a overhead (~8 µs/token = 0.08% of wall).  Architecturally cleaner but invisible at perf gate.
- **C-β (CPU overhead reduction)**: mpsc construct already 0 µs, session.begin already 1-2 µs, profile clone already cheap.  No measurable savings available.
- **R-C1 (memory-bus contention investigation)**: profiling shows the GPU body time is IDENTICAL between PARALLEL=0 and PARALLEL=1.  Memory-bus contention from concurrent CPU encoders does NOT manifest as longer GPU time.  This rules out memory-bus contention as the gap source.

All three Phase C levers are inert on hf2q-on-M5-Max because the critical path is **GPU body time at 9.05ms/token**, dominated by kernel execution + memory bandwidth on the GPU side.

## What WOULD move the needle (out of ADR-031 scope)

To reduce decode wall by ≥+2%:

1. **Reduce GPU dispatch count** (866 → fewer).  ADR-029's kernel-fusion work targets this.
2. **Reduce GPU barrier count** (420 → fewer).  Dual-buffer split + kernel reorganization help.
3. **Faster kernels** (per-dispatch GPU time reduction).  ADR-029's iter-160+ kernel-tuning lever.
4. **Smaller per-token memory traffic** (Q5_K + TQ-HB + flash_attn_vec_tq_hb already minimize this).

None of these are CPU-encode-parallelism territory.  They are ADR-029 (decode kernel optimization) territory.

## Honest verdict for ADR-031

**Phase B's design premise — that CPU encode parallelism overlapped with GPU work would save wall time — is architecturally wrong for hf2q-on-M5-Max.**  CPU encode is ~5% of wall; halving it saves ~2.5% theoretical maximum; the implementation overhead (2-CB submission, mpsc sync) eats most of that; the result is statistically indistinguishable from zero.

This is NOT a bug in Phase B's implementation.  The CFA workflow ran cleanly:
- Phase A: refactor landed (encode_one_layer extraction — valuable independently).
- Phase B: parallel-encode infrastructure landed (default-OFF, correctness preserved).
- Phase C: C0 profiling diagnosed that no CPU-side lever moves the needle.

**Recommendation**: close ADR-031.

## Recommended ADR-031 close-out

1. Update ADR-031 status to "Phase A + B landed; Phase C abandoned per C0 profiling — CPU is not the critical path on hf2q-on-M5-Max".
2. Add a "Lessons learned" section to ADR-031:
   - Profile before architecting (the original ADR-031 assumption that parallel-encode would help was based on iter-115's "GPU 95% body decode timing" memo — which actually SUPPORTED the conclusion that CPU is too small to matter, but was misread as "parallelize the 5% to save the 5%").
   - When the bottleneck is GPU at 85-95% of wall, no CPU-side parallelism can deliver >+5% gain regardless of how clever the CPU work is.
3. Keep HF2Q_PARALLEL_ENCODE env knob as opt-in experimental.  Document that no benefit has been measured on current production targets.
4. The Phase B infrastructure (worker registry, encode_parallel_layers_chunked, IIFE wrap, RAII semantics) is general-purpose and may be useful for future investigations.  Don't revert.
5. Phase C's planned `GraphSession::enqueue()` addition is NOT pursued for ADR-031, but could be added as a small mlx-native enhancement if a future investigation needs it.

## Future-friendly: what would change this conclusion

If hf2q ever runs on hardware where CPU encode is a larger fraction of wall time (e.g. very-fast GPU + slower CPU, or much-larger batch sizes that amortize GPU cost), Phase B's parallel-encode COULD pay off.  The infrastructure is ready.  But on M5 Max + gemma4-26B + single-token decode, the critical path lives elsewhere.

## Test artifacts

- `/tmp/parallel-profile-c0.log` — C0 worker/main per-phase µs timings
- `/tmp/parallel-split-timing.log` — SPLIT_TIMING PARALLEL=0 vs PARALLEL=1 comparison
- `src/serve/forward_mlx.rs` — HF2Q_PARALLEL_PROFILE instrumentation (permanent)
- `docs/research/ADR-031-phase-C-design-analysis.md` — predicted R-C1 as HIGH risk; C0 validated the prediction
- `docs/research/ADR-031-phase-C-step-C0-profile-findings.md` — initial C0 findings (before SPLIT_TIMING deep dive)
