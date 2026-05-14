# ADR-030 Phase 5 — Async parallel-encode design

**Date**: 2026-05-14 (iter-61)
**Status**: design artifact; no code change
**Pre-req**: Phases 1-4 + 6 (math) shipped on origin/main HEAD `c5bd41ca`

## Goal

Reach the ~1.3× speedup over the synchronous spec-decode baseline that
the Python DFlash MLX reference reports for `mx.async_eval` overlap.
This brings the projected hf2q Rust port speedup from ~1.31× (math
projection from Phase 1) to ~1.70×, comfortably clearing the mission
gate of ≥1.07× peer-FA parity.

## Where the overlap opportunity lives

A spec-decode round runs these GPU phases sequentially today:

```text
   [drafter forward]
        ↓ commit_and_wait at end of dispatch_dflash_model_forward
   [target verify forward_prefill_batched]
        ↓ commit_and_wait
   [per_position_argmax (lm_head + softcap + argmax loop)]
        ↓
   ...
```

**Data dependency analysis**:
- Drafter forward → uses `target_hidden_concat` from prior round's
  capture (data dep: PRIOR ROUND's target verify must complete first)
- Target verify → uses `verify_input = [last, draft_1, ..., draft_K]`
  where `drafts` come from this-round drafter forward
- Per-position argmax → uses target's captured pf_hidden + drafter's
  h_final

There IS no cross-phase parallelism WITHIN a round (data
dependencies force sequential).

**The real opportunity is CPU/GPU overlap**:
1. Drafter forward runs on GPU. Meanwhile CPU:
   - Builds verify_input Vec (microseconds — negligible)
   - Constructs the DFlashCaptureSession for next forward
2. Target verify runs on GPU. Meanwhile CPU:
   - Permutes captured hidden to drafter concat format (~270KB
     memcpy per round)
   - Uploads target_hidden_concat to GPU
3. Per-position argmax runs on GPU. Meanwhile CPU:
   - Builds RoundResult from final argmaxes
   - Sets up next round's state

Python's `mx.async_eval` essentially allows the lazy graph to be
SUBMITTED to GPU without waiting, so CPU can continue building the
next graph while GPU works.

**In hf2q's Rust + mlx-native:** the equivalent is using
`encoder.commit()` (non-blocking) instead of `commit_and_wait()`
between phases, with explicit `commit_and_wait()` only at
synchronization points where CPU truly needs GPU results.

## Current synchronization points

Tracing through `dispatch_dflash_generate`:

| Sync point | Reason | Necessary? |
|---|---|---|
| End of `forward_prefill_batched` (initial prompt) | Returns `first_token` u32 | YES — needed for output append |
| End of `dispatch_dflash_model_forward` (drafter) | Inside `dispatch_dflash_sdpa_cross_length` does CPU permute → must commit | YES (current implementation) — could be eliminated if SDPA were GPU-only |
| End of `per_position_argmax_from_hidden_opt` | Reads `argmax_index[0]` per position | YES — CPU reads result |
| End of `forward_prefill_batched` (verify) | Returns argmax, populates capture session | PARTIAL — argmax not used (we recompute via per_position); capture is GPU-resident, no CPU sync needed |
| `download_f32`/`as_slice<f32>` of captured | CPU permute to drafter concat | YES — CPU reads needed |
| Drafter h_final → `host_copy` | CPU prep for per_position_argmax | YES — CPU reads needed |

## Phase 5 implementation outline

### Quick win #1: Remove redundant sync after verify

The verify forward's returned `first_token` is unused (`_verify_first_token`).
The verify capture's `hidden_output` is read AFTER the forward
returns. Currently `forward_prefill_batched` does a final
`s.finish()` which blocks. If we could defer that block:
- CPU can immediately start building the next-round verify_input
- Or upload next-round's drafter h
- GPU continues finishing the verify forward in background

Issue: `forward_prefill_batched` is the 8,643-LOC monolith. Changing
its commit semantics requires careful review of all 4 production
callers (they expect `Result<u32>` returned synchronously).

Risk: medium-high. Defer until perf gate measurement shows it's
needed.

### Quick win #2: SDPA permute on GPU (eliminate CPU permute)

`dispatch_dflash_sdpa_cross_length` currently:
1. Commits + waits for Q
2. CPU downloads Q
3. CPU permutes seq-major → head-major
4. Uploads Q head-major
5. Runs SDPA
6. Downloads result
7. CPU permutes back
8. Uploads result

Replacement: pure-GPU permute via existing `mlx_native::ops::transpose`
or a GPU permute kernel. Removes 2 CPU↔GPU round-trips per layer per
round (5 layers × 2 = 10 round-trips/round eliminated).

Estimated win: ~30-50% of drafter forward time per round. Probably
~0.5-1ms per round. Cumulative over a generation: meaningful.

Risk: low-medium (replace 4 lines of CPU permute with a GPU dispatch;
existing dispatch_transpose is production-tested).

### Quick win #3: Batch per_position argmax sessions

`per_position_argmax_from_hidden_opt` runs `seq_len` sessions
sequentially:
```
for pos in 0..seq_len:
    begin session
    copy + norm + lm_head + softcap + argmax dispatches
    finish (waits for GPU)
    read argmax_index[0]
```

Could be batched into ONE session with `seq_len` instances of the
dispatch chain, then one finish, then read `seq_len` argmaxes from
a Vec-output buffer. Removes `seq_len - 1` sync points.

Risk: requires per-row argmax output buffer allocation; existing
single-shot argmax writes to `argmax_index[0]`. Modify to write to
`argmax_index[pos]` and allocate buffer of size `seq_len`. ~30 LOC.

### Cumulative impact

If all three quick wins land:
- Quick win #1: defers 1 sync per verify call (~1-2ms?)
- Quick win #2: eliminates 10 round-trips per round (~0.5-1ms)
- Quick win #3: collapses `seq_len` sync points to 1 (~0.5ms)
- **Total ~2-4ms/round saved** out of likely ~10-15ms total

At block_size=8, that's ~20-30% per-round speedup, matching the
Python reference's ~1.3× claim. **Net mission speedup**: ~1.31× ×
~1.3× = **~1.70×** over hf2q baseline → ~158 t/s vs hf2q's current
92.65 t/s = **1.59× peer-FA parity**, well above the 1.07× gate.

## Required scaffolding before Phase 5 implementation

1. **Coherence gate must be passing for sync version first** —
   no point optimizing buggy code. iter-58+iter-59 reviews surfaced
   2 coherence bugs; need GPU integration test on real models to
   confirm the multi-round wrapper produces correct output BEFORE
   any async work.

2. **Per-round timing instrumentation** — add `Instant::now()`
   bracketing to each Phase 5 candidate to measure exact wall-clock
   contribution. Without this, "quick win" estimates are speculation.

3. **Cargo bench harness** for the orchestrator path. Currently no
   per-round Criterion bench exists.

## Operator decision

Phase 5 is **performance optimization only** — does not gate
correctness. Given:
- Phase 1's measured Python baseline of 1.21× math + ~1.40× at K=7
- Rust port projection of ~1.31× (with cursor-mode KV)
- Mission gate at 1.07× (peer-FA parity)

There is **substantial margin** without Phase 5. If the post-Phase-4
perf gate clears 1.10× or higher on real gemma-4 hardware, Phase 5
becomes optional polish rather than required for mission completion.

**Recommendation**: defer Phase 5 implementation until the
end-to-end GPU integration test + initial perf measurement shows the
actual headroom on production hf2q + gemma-4-26B-A4B-it. Three
outcomes possible:
- Outcome A: ≥1.20× — Phase 5 deferred indefinitely (not needed)
- Outcome B: 1.07-1.19× — Phase 5 sized to the gap; quick win #2
  alone may suffice
- Outcome C: <1.07× — full Phase 5 + investigation of other levers

Implementation effort: ~100-200 LOC for quick wins #2 and #3; ~50
LOC for #1. Each is independently testable. The mantra-aligned path
is one quick win at a time with perf gates between.
