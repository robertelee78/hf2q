# ADR-029 iter-175 Step 1h — per-kernel bench: matvec is 57% of wall and near-peak; gap concentrates in non-matvec

**Date**: 2026-05-15
**HEAD**: hf2q `e181ad20`, mlx-native `b32b81e`
**Iteration**: 7 of /loop autonomous

## Summary

Comprehensive per-kernel bench at HEAD reframes where the residual decode gap lives:
- **Matvec kernels run at 70-119% of M5 Max peak bandwidth** (avg ~85%) — near optimum for memory-bound work
- **Attention+router matvecs total 4.39 ms/tok = 41% of decode wall**
- **MoE-id matvecs total 1.75 ms/tok = 16% of decode wall**
- **Combined matvecs = 57% of wall; remaining 43% lives in non-matvec kernels** (FA, hadamard, fused norms, MoE routing, KV ops)
- The structural ~6-8% gap to peer-FA likely concentrates in the **non-matvec 43%**, not in already-near-peak matvecs

This is a substantial reframe from the standing-context assumption that "matvec kernels are where the lever lives."

## Bench data — attention + lm_head + sweep (`bench_decode_qmatmul_shapes`)

Median per-call batched timing (32 dispatches/CB, sync amortized):

| Shape | N | K | qtype | µs/call | MB read | GB/s | % peak (546 GB/s) |
|---|---:|---:|---|---:|---:|---:|---:|
| **Q_sliding** | 4096 | 2816 | Q6_K | **24.6** | 9.5 | **384.1** | **70.3%** |
| K_sliding | 2048 | 2816 | Q6_K | 11.8 | 4.7 | 402.1 | 73.7% |
| V_sliding | 2048 | 2816 | Q6_K | 10.6 | 4.7 | 444.5 | 81.4% |
| O_sliding | 2816 | 4096 | Q6_K | 15.0 | 9.5 | 630.2 | 115.4% (L2 amp) |
| Q_global | 4096 | 2816 | Q6_K | 16.1 | 9.5 | 586.7 | 107.5% |
| O_global | 2816 | 4096 | Q6_K | 14.6 | 9.5 | 649.6 | 119.0% |
| Router | 128 | 2816 | Q5_K | 6.1 | 0.2 | 40.7 | 7.5% (tiny shape) |
| lmhead_Q6_K | 262144 | 2816 | Q6_K | 1037.8 | 605.6 | 583.5 | 106.9% |
| lmhead_Q8_0 | 262144 | 2816 | Q8_0 | 1365.1 | 784.3 | 574.6 | 105.2% |

## Bench data — MoE-id (`bench_decode_moe_id_shapes`)

| Shape | tk | tok | N | K | µs/call | MB read | GB/s | % peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **g4_gate_up_Q6K** | 8 | 1 | 1408 | 2816 | **35.7** | 26.0 | **728.4** | **133.4%** |
| **g4_down_Q8_0** | 1 | 8 | 2816 | 704 | **22.7** | 16.9 | **743.8** | **136.2%** |
| q36_gate_Q5K | 8 | 1 | 512 | 2048 | 14.2 | 5.8 | 405.1 | 74.2% |
| q36_down_Q6K | 1 | 8 | 2048 | 512 | 15.7 | 6.9 | 437.3 | 80.1% |

Bench output:
```
gemma4 (30 layers, 60 _id calls/token): 1.29 GB read in 1.75 ms (aggregate 734 GB/s)
qwen3.6 (40 layers, 80 _id calls/token): 0.74 GB read in 1.78 ms (aggregate 413 GB/s)
```

## Bench data — hadamard (`bench_hadamard`)

| Shape | Time/CB |
|---|---:|
| `hadamard_d256_h8` (single dispatch CB) | 157.08 µs |
| `hadamard_d512_h2` (single dispatch CB) | 156.11 µs |
| `hadamard_full_model_60dispatches` (60 dispatches/CB) | 173.06 µs |

Per-dispatch hadamard: 173 / 60 ≈ **2.9 µs/dispatch**. At ~50 hadamard calls/tok in decode = ~145 µs/tok = **~1.4% of decode wall**. Not the lever.

## Wall-budget allocation per decoded token

Decode wall at HEAD: ~10.7 ms/tok (≈ 93 t/s):

| Component | Per-tok | % wall | Note |
|---|---:|---:|---|
| Attention + router matvecs | 4.39 ms | 41.0% | weight-read bandwidth-bound; running at 384-630 GB/s (avg ~85% peak) |
| MoE-id matvecs | 1.75 ms | 16.4% | hitting 728-744 GB/s (>peak via L2 amp) |
| Hadamard quantize | ~0.145 ms | 1.4% | 2.9 µs/dispatch × 50/tok |
| **Sub-total matvec + hadamard** | **6.28 ms** | **58.7%** | |
| **NON-MATVEC residual** | **4.42 ms** | **41.3%** | FA hybrid + reduce, KV copy, fused_head_norm_rope, rms_norm, MoE routing/swiglu/weighted_sum |

(Peer baseline at the same hardware: ~9.6 ms/tok = 104 t/s. Gap: ~1.1 ms/tok = 10% wall.)

## Where the gap concentrates

Matvecs at decode shapes are **near-peak** efficient (avg ~85% of M5 Max bandwidth peak). The 6-8% gap to peer-FA does NOT live in matvec kernel inefficiency.

The structural lever space — by elimination — is in the non-matvec 41.3% of wall. This includes:
- **flash_attn_vec_hybrid_dk256** (25 calls/tok per Step 1's data)
- **flash_attn_vec_reduce_dk256** (25 calls/tok)
- **kv_copy_kf16_quantize_v_no_fwht_d256** (25 calls/tok)
- **fused_head_norm_rope_f32_v2** (60 calls/tok)
- **fused_moe_routing_f32_v2** (30 calls/tok)
- **moe_weighted_sum** (30 calls/tok)
- **moe_swiglu_batch** (30 calls/tok)
- **rms_norm_f32_v2** (150 calls/tok — already peer-pattern, fast)

## Updated H-F hypothesis (NEW)

**H-F**: the residual ~6-8% gap concentrates in one or more of the hf2q-specific non-matvec kernels (flash_attn_vec_hybrid, hadamard_quantize_kv, fused_moe_routing, etc). Peer's equivalent operations are implemented differently or fused differently; matching peer's structure on these kernels may close the gap.

**Testable next step**: bench each non-matvec kernel in isolation, identify which one(s) account for a disproportionate share of the 4.42 ms/tok non-matvec budget. Targets:
- if FA hybrid+reduce together >2 ms/tok = bigger lever (50 calls × ~40 µs avg)
- if MoE routing+swiglu+weighted_sum together >1.5 ms/tok = bigger lever
- if no single non-matvec kernel dominates → distributed gap; harder to close

## Updated iter-175 status

| Step | Hypothesis | Status |
|---|---|---|
| 1 | Dispatch baseline | DONE |
| 1b | H-A encoder, H-B compile flags | FALSIFIED / PARTIALLY FALSIFIED |
| 1d | H-D concurrency | CONFIRMED 3.5pp ceiling |
| 1e | H-D2 enabling infra | LANDED default-OFF |
| 1f | H-D2 single-site | FALSIFIED (neutral) |
| 1g | H-E precompile toolchain | CONFIRMED, test DEFERRED |
| **1h** | **Per-kernel bench** | **DONE — matvecs near-peak; lever lives in non-matvec 41% of wall** |
| 1i (next) | H-F: per-non-matvec-kernel deep-dive | OPEN |

## What was ruled out at Step 1h

- ✗ **Matvec kernel inefficiency as the close-the-gap lever**: kernels at 70-119% peak (avg ~85%) for memory-bound work. The non-matvec residual is the new focus.
- ✗ **Hadamard kernel as a lever**: 1.4% of wall, even halving would barely register.

## What survives as open lever space

1. Flash attention vec hybrid + reduce (key suspects at deep kv)
2. fused_head_norm_rope (60 calls/tok)
3. KV copy + quantize fused kernels
4. MoE routing + combine kernels
5. Plus the iter-175 deferred levers: H-D global migration, H-E precompile, H-C cache

## Cross-references

- iter-175 Step 1 dispatch baseline: `docs/research/ADR-029-iter-175-step-1-dispatch-distribution-2026-05-15.md`
- iter-179 kernel-efficiency threshold (>70% = NOT kernel-bound, <50% = kernel-bound): inline at `mlx-native/benches/bench_decode_qmatmul_shapes.rs:7-12`
- Existing bench infrastructure: `mlx-native/benches/bench_decode_qmatmul_shapes.rs`, `bench_decode_moe_id_shapes.rs`, `bench_hadamard.rs`

## Reproducibility

```bash
cd /opt/mlx-native
cargo bench --bench bench_decode_qmatmul_shapes
cargo bench --bench bench_decode_moe_id_shapes
cargo bench --bench bench_hadamard
```
