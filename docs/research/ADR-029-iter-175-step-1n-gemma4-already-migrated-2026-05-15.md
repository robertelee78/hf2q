# ADR-029 iter-175 Step 1n — Critical Chesterton's fence: gemma4 decode path ALREADY uses smart barriers

**Date**: 2026-05-15
**HEAD**: hf2q `c0f10d65`, mlx-native `7fd679f`
**Iteration**: 13 of /loop autonomous

## Summary

Per R2 from /ruflo-goals:research-synthesize, started H-D global barrier migration. **Audit reveals gemma4 production decode is ALREADY migrated** to smart conditional barriers (`session.barrier_between(&[reads], &[writes])` from `mlx-native/src/graph.rs:1494`). Hard-placed `enc.memory_barrier()` survives in 3 gemma4 sites — all are **debug-gated** (`b9_sequential`, `use_iter367_fusion` PROBE), NOT in production path. The 362 remaining unconditional barriers across hf2q live in qwen35/vision/bert/dflash paths — those aren't the iter-175 target.

**Implication**: the H-D 3.5pp ceiling identified at Step 1d is NOT capturable by migrating more gemma4 barriers, because gemma4 is already optimally migrated.

## Audit of barrier API usage

| File | `barrier_between` count | `memory_barrier()` count | Status |
|---|---:|---:|---|
| `src/serve/forward_mlx.rs` (gemma4 decode/prefill) | 62 | 3 | All 3 debug-gated; production ALREADY migrated |
| `src/serve/forward_prefill_batched.rs` | 55 | ? | Batched prefill path |
| `src/serve/forward_prefill.rs` | 30 | ? | Sequential prefill path |
| `src/inference/vision/vit_gpu.rs` | 0 | 81 | Vision path (different model class) |
| `src/inference/vision/vit_gpu_qwen3vl.rs` | 0 | 48 | Vision/qwen3vl |
| `src/inference/models/qwen35/gpu_delta_net.rs` | 0 | 48 | qwen35 delta net (iter37 deferred migration) |
| `src/inference/models/qwen35/gpu_full_attn.rs` | 0 | 40 | qwen35 attn |
| `src/inference/models/bert/bert_gpu.rs` | 0 | 27 | BERT |
| `src/inference/models/qwen35/gpu_ffn.rs` | 0 | 24 | qwen35 FFN |
| `src/inference/spec_decode/dflash/forward.rs` | 0 | 23 | DFlash spec-decode |
| `src/inference/models/nomic_bert/forward.rs` | 0 | 19 | Nomic BERT |
| `src/inference/models/qwen35/forward_gpu.rs` | 0 | 15 | qwen35 forward orchestrator |
| `src/calibrate/autograd_gpu_tape.rs` | 0 | 12 | Calibration |
| **Total** | **147** | **362** | |

## What the gemma4 path does today

`session.barrier_between(reads, writes)` at `mlx-native/src/graph.rs:1494`:

```rust
pub fn barrier_between(&mut self, reads: &[&MlxBuffer], writes: &[&MlxBuffer]) {
    // ... capture-mode handling ...
    let reason = self.tracker.conflicts_reason(reads, writes);
    if let Some((_kind, _new_ptr, _existing_ptr)) = reason {
        // ... emit memory_barrier + reset tracker ...
    }
    self.tracker.add(reads, writes);
}
```

This mirrors llama.cpp's `ggml_metal_op_concurrency_check` + `ggml_metal_op_concurrency_reset` pattern (peer's source: `ggml-metal-ops.cpp:147-225`). When the new dispatch's reads/writes don't conflict with the cumulative tracker state, the barrier is **elided** — exactly what peer does.

## Re-measured barrier ratios at HEAD

Fresh `HF2Q_DUMP_COUNTERS=1` decode (50 tokens at gemma4-APEX-Q5_K_M):

```
[MLX_COUNTERS] dispatches=42630 cmd_bufs=98 barriers=23667 decode_tokens=50
dispatches/decode_tok=852.60  barriers/decode_tok=473.34
```

- hf2q at HEAD: 852.6 dispatches/tok, 473.3 barriers/tok → **1.80 dispatches per concurrent group**
- peer (per iter-115 standing data + Step 1d re-measure): 1339 disp/tok, 844 bar/tok → **1.59 dispatches per concurrent group**

Peer's groups are SMALLER (more barriers per dispatch). To match peer's 1.59 ratio, hf2q would need ~536 barriers/tok = 63 MORE per token = ~2 extra barriers per layer.

## The H-D 3.5pp ceiling — re-interpreted

Step 1d's 4-arm bench showed:
- peer-concurrent / peer-serial = +11.9% benefit
- hf2q-concurrent / hf2q-serial = +8.4% benefit
- Gap: 3.5pp

This is NOT capturable by adding more barrier sites (each barrier is sub-µs Metal overhead; 63 extra barriers/tok × 5 ns = 315 ns/tok = negligible).

The 3.5pp must come from **dispatch structure differences**:
- Peer has 1339 dispatches/tok (53% more than hf2q's 875)
- Peer's individual kernels are smaller and run more efficiently on Apple GPU's scheduler
- hf2q's fused kernels are bigger but the schedule benefits less from concurrency

Per iter-105's "many small wins on Apple Metal" pattern, peer's strategy IS the high-throughput strategy. But unfusing hf2q kernels regresses (iter-1 H6 −2.8% for `fused_post_attn_triple_norm_f32`, iter-107 H76 −0.x% for split `fused_norm_add`). The granularity hf2q operates at is OPTIMAL for its current kernel set; further moves either direction regress.

## What this changes about the iter-175 mission

The synthesis report's R2 (start H-D global migration) needs revision:
- **H-D for gemma4 decode**: **ALREADY DONE** (production path uses 100% `barrier_between`)
- **H-D for qwen35**: still deferred (15 forward_gpu + 40 gpu_full_attn + 48 gpu_delta_net + 24 gpu_ffn = 127 unconditional barriers). Per iter37 plan ("iter38+ scope"). Not the ADR-029 target.
- **H-D for other models**: vision (129), bert (46), dflash (23), nomic_bert (19), calibrate (12). Each model would benefit independently, but none affect ADR-029's gemma4 target.

## Updated standing-context for ADR-029

The residual 6-8% peer-FA gap on gemma4-APEX-Q5_K_M at M5 Max is **STRUCTURAL FLOOR** for the current architecture given:
- ✓ Matvec kernels at 70-119% peak (Step 1h)
- ✓ Encoder fast-path equivalent to peer (Step 1b)
- ✓ Compile options equivalent (Step 1b runtime) + precompile lever closed (Step 1m, prefill +3-5%)
- ✓ Smart conditional barriers already in production (this Step 1n)
- ✗ Fusion-direction levers all regress on Apple Metal at gemma4 shape (iter-1 H6, iter-107 H76, iter-101)
- ✗ Unfusion-direction also regresses (iter-105 explored)

The gap is in HOW APPLE'S GPU SCHEDULES our specific dispatch structure vs peer's. This is not addressable at the framework/CPU level — would require a kernel-set rewrite to match peer's dispatch granularity (multi-week to multi-month engineering, no guarantee of success).

## Testable next hypotheses (H-H family)

**H-H1**: explicitly add `barrier_between` sites in gemma4 forward to FORCE smaller concurrent groups (target 1.59 ratio matching peer). If peer's smaller groups DO have Apple-GPU-scheduler advantage independent of dispatch granularity, this gives a free win. If not, it falsifies and shows the lever is in dispatch structure itself.

**H-H2**: unfuse ONE specific fused kernel in gemma4 (e.g. `fused_head_norm_rope_v2` back into separate norm + rope) and bench. Targets the dispatch-structure axis.

**H-H3**: profile a single token's worth of dispatches with Apple Instruments Metal trace (operator-runs) to attribute per-kernel-name GPU time and identify outliers.

## What to commit this iteration

1. The audit finding (gemma4-already-migrated) is a CRITICAL Chesterton's-fence finding that prevents future iterations from re-attempting unnecessary work.
2. Memory entry + ADR update.
3. Update R2 in the synthesis to reflect the new state.

## Cross-references

- `mlx-native/src/graph.rs:1494` `GraphSession::barrier_between` impl
- `mlx-native/src/mem_ranges.rs` MemRanges tracker
- Step 1d 4-arm bench: `docs/research/ADR-029-iter-175-step-1d-concurrency-lever-2026-05-15.md`
- Synthesis report: produced via `/ruflo-goals:research-synthesize` after Step 1l
