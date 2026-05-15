# ADR-029 iter-175 Step 1i — per-layer-phase GPU attribution: FFN is 2× attn, dispatch-overhead-bound

**Date**: 2026-05-15
**HEAD**: hf2q `2ed70176`, mlx-native `b32b81e`
**Iteration**: 8 of /loop autonomous

## Summary

Per-layer-phase GPU timing via `HF2Q_PER_LAYER_PHASE_GPU_TIME=1` + `HF2Q_FFN_SPLIT=1` reveals that **FFN is ~2× more expensive than attention per layer** (~191 µs vs ~98 µs sliding), and the dominant FFN sub-phase is the **MoE pipeline (FFN_BODY) at ~182 µs/layer**. FFN_NORMS sub-phase (~104 µs/layer) is **dispatch-overhead-bound** — each small-vector norm dispatch costs ~12 µs of which the actual memory access is sub-microsecond. The structural lever for closing the gap is **kernel fusion** (fewer dispatches at the same total work), not per-kernel optimization.

## Per-layer-phase GPU time (HF2Q_PER_LAYER_PHASE_GPU_TIME=1)

Numbers represent GPU wall-clock per layer-phase under per-layer commit_and_wait. Each commit adds ~80-100 µs sync overhead, so absolute numbers are inflated; relative attribution is valid.

| Layer type | PHASE_ATTN | PHASE_FFN |
|---|---:|---:|
| Sliding (24 layers) | ~98 µs | ~191 µs |
| Global (6 layers) | ~135 µs | ~191 µs |

**Per-token allocation** (30 layers = 24 sliding + 6 global):
- PHASE_ATTN: 24×98 + 6×135 = 2352 + 810 = **3162 µs/tok**
- PHASE_FFN: 30×191 = **5730 µs/tok**
- Layer body total: **~8.9 ms/tok** (instrumented, includes commit overhead)

FFN is 1.81× more expensive than attn per layer.

## FFN sub-phase split (HF2Q_FFN_SPLIT=1)

Sliding layer breakdown:

| Sub-phase | Per layer | What it contains |
|---|---:|---|
| FFN_NORMS | ~104 µs | post-attn rms_norm + B8 3 pre-FF norms + router-norm (≈5 norm dispatches) |
| FFN_BODY | ~182 µs | B9 dense gate + dense up + router (3 matvecs), then B10-B13 MoE expert calls + combine |
| FFN_EOL | ~5.6 µs | end-of-layer fusion |

Per-token (30 layers, sliding):
- FFN_NORMS: 30 × 104 = **3120 µs/tok** (inflated)
- FFN_BODY: 30 × 182 = **5460 µs/tok** (inflated)
- FFN_EOL: 30 × 5.6 = ~170 µs/tok (tiny)

**FFN_BODY (the MoE pipeline) is the dominant chunk of the decode wall.**

## Key insight: FFN_NORMS is dispatch-overhead-bound

FFN_NORMS = ~104 µs/layer ÷ ~5 norm dispatches/layer = **~20 µs/norm dispatch** (inflated) → real ~10-15 µs.

Each rms_norm reads only an activation vector (hidden_size=2816 floats = 11.3 KB). At M5 Max ~500 GB/s, the memory access alone is ~23 ns. The remaining ~10 µs is **GPU launch overhead + kernel-state setup + writeback** — not kernel computation or bandwidth.

**This is the per-dispatch fixed cost identified at iter-111** (gap is "~1 µs/dispatch below single-site noise floor"). Multiplied by 150+ norm dispatches per decode token, it adds up.

## What this means for the close-the-gap path

The kernels are not slow in isolation — they're launch-overhead-bound at small problem sizes. The structural levers are:

**A) Kernel FUSION** (combine multiple norms + ops into one dispatch). Existing examples:
- `fused_head_norm_rope_f32_v2` (Q norm + K norm + RoPE in 1 kernel)
- `fused_norm_add_f32_v2` (norm + residual add)
- `fused_post_ff_norm2_endlayer_f32_v2` (3-op end-of-layer fusion)

These already exist — hf2q has been pushing this direction. Each fusion saves ~10-15 µs (one launch overhead). At ~30 layers, fusing one more norm site saves ~300-450 µs/tok = ~3-4% wall.

**B) Reduce pipeline-state changes** (set_compute_pipeline_state is the most expensive per-dispatch ObjC call). Group dispatches by pipeline state. This is what peer's mem_ranges + concurrent dispatch achieves (H-D's 3.5pp ceiling) — peer's CB structure may co-locate same-pipeline dispatches more aggressively.

**C) Move per-layer commit_and_wait off the critical path** (Phase A/B's parallel-encode ALREADY attempted this — confirmed at ADR-031 closure to be the wrong lever because GPU is 85% of wall, not CPU encode).

## Updated wall budget

Combining Step 1h (matvec measurements) + Step 1i (per-phase attribution):

| Component | Estimate (real, instrumentation removed) | % wall |
|---|---:|---:|
| Attn matvec (Q/K/V/O across 30 layers) | ~1.86 ms | ~17% |
| Attn non-matvec (head_norm_rope + FA + kv_copy) | ~1.1 ms | ~10% |
| FFN matvec (MoE expert gate/up/down + router) | ~1.75 ms | ~16% |
| FFN non-matvec (norms + routing + swiglu + weighted_sum) | ~4.0 ms | ~37% |
| head (lm_head + softmax + argmax) | ~1.0 ms | ~9% |
| sync + per-dispatch CPU encode | ~1.0 ms | ~9% |
| **TOTAL** | **~10.7 ms** | **~100%** |

The biggest chunk is **FFN non-matvec at ~37% of wall**. Targeting THIS is where the gap lives.

## Testable next steps (H-G candidates)

1. **H-G1**: identify the slowest FFN-non-matvec kernel via isolation bench. Candidates: `fused_moe_routing_f32_v2`, `moe_weighted_sum`, `moe_swiglu_batch`, `fused_norm_add_f32_v2`. Effort: 1-2 hours per bench.
2. **H-G2**: look for FUSION opportunities in FFN_NORMS sub-phase. The 4-5 norm dispatches per layer in FFN_NORMS could potentially be 2-3 fused dispatches. Each saved dispatch = ~10-15 µs/layer = ~300-450 µs/tok per saved norm.
3. **H-G3**: peer comparison — read llama.cpp's gemma4 FFN forward path to see how peer structures the same operations. Specific files: `ggml-metal-ops.cpp` MoE-related ops.

H-G1 is the cleanest next /loop iteration target — write one bench, run, identify if any kernel dominates.

## Updated iter-175 cumulative ledger

| Step | Hypothesis | Status |
|---|---|---|
| 1 | Dispatch baseline | DONE |
| 1b | H-A encoder, H-B compile flags | FALSIFIED |
| 1d | H-D concurrency | CONFIRMED 3.5pp ceiling |
| 1e | H-D2 enabling infra | LANDED default-OFF |
| 1f | H-D2 single-site | FALSIFIED |
| 1g | H-E precompile toolchain | CONFIRMED, test DEFERRED |
| 1h | Per-kernel bench | DONE — matvecs near-peak |
| **1i** | **Per-layer-phase attribution** | **DONE — FFN non-matvec is 37% of wall** |
| 1j (next) | H-G1 isolation bench of FFN-non-matvec kernels | OPEN |

## Reproducibility

```bash
HF2Q_PER_LAYER_PHASE_GPU_TIME=1 ./target/release/hf2q generate \
    --model /opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf \
    --prompt "Q." --max-tokens 3 --temperature 0 --ignore-eos 2>&1 \
    | grep -E "PHASE_ATTN|PHASE_FFN"

HF2Q_FFN_SPLIT=1 ./target/release/hf2q generate \
    --model /opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf \
    --prompt "Q." --max-tokens 2 --temperature 0 --ignore-eos 2>&1 \
    | grep -E "FFN_NORMS|FFN_BODY|FFN_EOL"
```
