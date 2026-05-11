# ADR-029: gemma4-APEX-Q5_K_M Decode Gap is in the MoE Pipeline, Not TQ

> **🎯 ITER-5 SMOKING GUN 2026-05-11 — ADR-029 GOT THE GAP STRUCTURE WRONG**
>
> Measured peer's actual gemma4-APEX-Q5_K_M dispatch counts by patching
> llama.cpp's ggml-metal-device.m with atomic dispatch+barrier counters
> and a per-pipeline histogram. Slope analysis across tg10/50/100/200 r=1
> at peer = 99-102 t/s converges to:
>
> | metric | peer | hf2q | ratio |
> |---|---:|---:|:---:|
> | dispatches/decode-tok | **1389** | 883 | **peer 1.57× MORE** |
> | barriers/decode-tok | **905** | 487 | peer 1.86× MORE |
> | µs/dispatch (avg) | **7.05** | **15.10** | **hf2q is 2.14× SLOWER per dispatch** |
>
> **ADR-029's "peer issues 105 dispatches/tok" claim was hf2q's old candle
> Phase 0 reference baseline, NOT peer's actual count.** Peer dispatches
> ~57% MORE kernels than hf2q. The gap is entirely **per-dispatch cost**,
> not dispatch count. Every dispatch-fusion lever falsified in iter-1..4
> (H6 triple-norm, H7 wsum-end-layer) made hf2q's KERNELS LARGER —
> exactly the WRONG direction.
>
> Peer's pipeline histogram (per decode-token forward pass) shows it
> fragments work into more, smaller kernels:
> - 212 `rms_norm_mul_f32_4` (norm+weight, ~7/layer × 30)
> - 176 `mul_mv_q6_K_f32_nsg=2` (Q6_K mat-vec, dense path)
> - 60 `rope_neox_f32_imrope=0` (Q+K RoPE, 2/layer)
> - 60 `set_rows_f16_i64` (KV cache write, 2/layer)
> - 60 `soft_max_f32_4` (attention + routing softmax)
> - 59 `rms_norm_mul_add_f32_4` (3-op fusion: norm+mul+add)
> - 55 `mul_mv_f16_f32_4_nsg=2` (F16 attention kernels — peer does NOT
>   use a single flash_attn_ext kernel for gemma4 decode!)
> - 30 `bin_fuse_f32_f32_f32_4_op=0_nf=7` (peer chains 7 ADDs in 1 kernel)
> - 30 each of `cpy_f32_f32`, `mul_mv_id_q6_K_f32_nsg=2`,
>   `mul_mv_q8_0_f32_nsg=4`, `argsort_f32_i32_desc`, `sum_rows_f32_f32_4`
> - 30 `mul_mv_f32_f32_4_nsg=4` (F32 mat-vec, probably router_proj)
>
> Notable peer ABSENCES vs hf2q:
> - **NO single flash_attn_ext** — peer computes attention as
>   `mul_mv_f16(Q@K^T) → soft_max → mul_mv_f16(score@V)` per layer.
>   hf2q has a monolithic `flash_attn_vec` kernel which may be 2-3× longer
>   per call than peer's 3 separate dispatches summed.
> - hf2q's largest single dispatch is `lm_head` (Q6_K mat-vec at
>   vocab 262144 × hidden 2816 = 736M ops, ~343 µs/dispatch in production).
>   Peer likely shards this across multiple smaller `mul_mv_q6_K` calls.
>
> **The new optimization direction**: SHARD hf2q's largest kernels into
> smaller per-call work. Specifically:
> 1. **Replace flash_attn_vec with peer's 3-dispatch attention** (Q@K^T,
>    softmax, score@V). Could save ~30-50 µs/layer × 30 = ~1 ms/tok.
> 2. **Shard lm_head** Q6_K mat-vec across multiple dispatches (peer
>    splits vocab dim 262144 into chunks → multiple mul_mv_q6_K calls).
> 3. **Tune kernel threadgroup geometry** to match peer's `nsg=2/nsg=4`
>    pattern for the Q6_K/Q8_0 dispatchers.
> 4. **Profile per-kernel timing** with HF2Q_DUMP_COUNTERS extended to
>    capture per-pipeline µs, identify which hf2q dispatches are > 30 µs
>    (i.e., > 2× hf2q's average), and target those for sharding.
>
> This direction is the inverse of ADR-029 §Decision 2 (dispatch fusion).
> The §Decision items 1-5 are still closed (audits + qwen3.6 σ-pct stand);
> the §Decision item 2 *direction* was a wild-goose chase. ADR-030 should
> be opened to formalize the per-dispatch-cost direction with the
> instrumented peer data captured here.
>
> Instrumentation patch: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m`
> with atomic `hf2q_peer_dispatch_count` + per-pipeline histogram, env-gated
> via `HF2Q_PEER_COUNT_PRINT=1` and `HF2Q_PEER_PIPELINE_HIST=1`.
>
> **iter-6 first test of the new direction — falsified for throughput.**
> Ported `kernel_mul_mv_id_q8_0_f32_nr2` to mlx-native (commit `7acd4d4`,
> matches peer's `N_R0_Q8_0=2 + N_SG_Q8_0=4`). Gemma4 MoE `ffn_down_exps`
> is Q8_0 (30 dispatches/decode-tok on this path). Coherence: BYTE-IDENTICAL
> 30-tok haiku. Throughput: **74.3 → 73.3 t/s = -1.3% regression**
> (σ-pct 0.14% / 0.30%). Same pattern as iter-1..3 falsifications: the new
> kernel is functionally correct but Apple Metal's scheduler doesn't favor
> the larger threadgroup (128 threads vs 64) for this workload shape.
> Kernel kept (env-flag `HF2Q_Q8_0_ID_MV_NR2=1`, default-off) for future
> workloads.
>
> **The 2.14× per-dispatch gap is REAL but NOT explained by SG count.**
> Both engines use `MTLDispatchTypeConcurrent`. Both use residency sets +
> shared buffers. The remaining suspects:
> - Apple Metal kernel launch overhead is fundamentally lower for peer's
>   kernels (compiled differently? smaller pipeline state?)
> - hf2q has command-encoder bookkeeping (mlx-native's tracker + barrier
>   bookkeeping in graph.rs:barrier_between) adding 1-2 µs CPU per call
> - Peer's larger total dispatch count (1389 vs 883) better hides Metal
>   bubble overhead — the Apple GPU scheduler may be optimized for a
>   "many small kernels" stream pattern
>
> Next direction: add per-pipeline timing instrumentation to mlx-native
> (mirror the peer-side patch), capture per-kernel µs distribution to
> identify hf2q outliers (kernels > 2× the global avg of 15 µs). Most
> promising candidates remain `flash_attn_vec_tq_hb` (monolithic
> attention) and `lm_head` (single dispatch over 262144 vocab).
>
> **iter-6 closing — FFI/encode-overhead hypothesis FALSIFIED via existing
> bench**. `/opt/mlx-native/benches/bench_dispatch_overhead.rs` (ADR-028
> iter-254) already tested this in March-2026; ran fresh today:
> | shape | CPU-only/disp | GPU+sync/disp | CPU% |
> |---|---:|---:|---:|
> | Router_Q5K (n=128) | 0.19 µs | 4.82 µs | 3.8% |
> | Q_sliding_Q5K (n=4096) | 0.19 µs | 12.81 µs | 1.5% |
> | lmhead_Q6_K (n=262144) | 0.33 µs | 1059 µs | 0.0% |
> hf2q's per-dispatch CPU encode is **0.19-0.33 µs** — negligible.
> The 15 µs/dispatch avg (from production decode) is almost entirely
> GPU + driver sync time, not Rust→Metal binding overhead.
> Note: lm_head alone is **1.06 ms per call** = 8% of decode wall-clock;
> peer faces the same shape and same GPU, so this is parity not gap.
> The 3.6 ms/tok gap to peer must therefore be spread across the OTHER
> ~882 dispatches at ~4 µs/dispatch differential each — concentrated in
> specific kernel types that hf2q implements differently from peer.
>
> The investigation has reached the limit of what can be productively
> done in 5-min /loop iterations. The remaining structural work
> (per-pipeline µs histogram + flash_attn_vec_tq_hb decomposition into
> 3-dispatch peer-style attention) is multi-iter scope appropriate for
> a /cfa swarm with explicit operator approval — Codex in review-only
> mode for peer-source audit while Claude implements + benches.
>
> **iter-7 — ADR-029's MoE attribution was DOUBLY WRONG**:
> `/opt/mlx-native/benches/bench_decode_moe_id_shapes.rs` (ADR-028 iter-181,
> already in-repo) measures hf2q's actual MoE per-token GPU time at the
> exact gemma4 APEX-Q5_K_M shapes:
> | shape | batched µs/call | calls/tok | total µs/tok | GB/s |
> |---|---:|---:|---:|---:|
> | g4_gate_up_Q6K (n=1408 k=2816 tk=8) | 34.8 | 30 | 1044 | 748 |
> | g4_down_Q8_0 (n=2816 k=704 tk=1, ntok=8) | 21.1 | 30 | 633 | 799 |
> | **gemma4 MoE total per decode tok** | | **60** | **1677 µs** | **767 (agg)** |
>
> **gemma4 MoE = 1.68 ms/decode-token = 12.6% of the 13.33 ms wall-clock**,
> NOT 49% as ADR-029's H5B-routing-skip "5.02 ms" probe claimed. H5B was an
> **upper bound from incoherent skip behavior** (skipping routing leaves
> garbage expert_ids → MoE expert dispatches still run but on a
> degenerate access pattern that may load the same expert 8× with cache
> hits — making the skipped baseline FASTER than coherent execution
> would be).
>
> hf2q's MoE is already at **747 GB/s = 137% of the "peak" 546 GB/s
> sustained estimate**, i.e. near memory-bandwidth saturation. There's
> no meaningful MoE optimization headroom on gemma4 APEX-Q5_K_M.
>
> **The 3.6 ms/tok gap is OUTSIDE the MoE pipeline.** Where? Per-layer
> remaining budget = (443 − 56 µs MoE − 35 µs lmhead amortized) = ~352
> µs/layer for attention + dense MLP + norms + KV cache + RoPE. The
> peer-side total for these same ops must therefore be ~220 µs/layer
> for the gap math to add up (3.6 ms / 30 layers = 120 µs/layer
> attention-path savings).
>
> The largest remaining hf2q candidates by time:
> - `flash_attn_vec_tq_hb` (monolithic Q@K + softmax + @V), peer
>   replaces with 3 separate kernels — hypothesis (un-tested) it
>   accumulates ~50-80 µs/layer overhead vs peer
> - dense MLP (Q6_K up/gate/down via 3 separate `dispatch_qmatmul`
>   calls), peer fuses via `pipeline_norm_fuse_norm_mul_add` and tighter
>   ggml-graph scheduling — also un-tested
> - Q-norm + K-norm + RoPE per-head pair (`s.rms_norm` × 2 +
>   `dispatch_rope_*`), peer combines via `kernel_rope_neox_*` fused
>   pre-RoPE-norm
>
> Note that ADR-029's "MoE pipeline is the gap" framing is now
> retracted twice over: first by iter-5 (peer issues MORE dispatches,
> not fewer) and now by iter-7 (MoE is only 12.6% of wall-clock, not
> 49%). The actual gap structure remains unidentified at the bucket
> level after 7 iters.
>
> --- (prior MISSION REOPENED note retained below for chronology) ---

> **⚠ MISSION REOPENED 2026-05-11 (post iter-4 merge)**
>
> The iter-4 closure framing "all §Decision items closed = mission complete"
> was a misread of operator intent. The §Decision items 1-5 are valid sub-goals
> (and remain closed) but the REAL mission target is **close the gemma4 decode
> gap from ~75 t/s to ≥98 t/s** (operator-measured peer parity on
> `gemma4-ara-2pass-APEX-Q5_K_M.gguf`). That target is **NOT achieved**. The
> iter-1..4 work shipped:
> - 1 functional fix (kernel-profile Q6_K lm_head gating) — useful but
>   doesn't move t/s
> - 6 falsified hypotheses (TQ-removed +7% only; H6 triple-norm regresses
>   -2.8%; H7 wsum-end-layer regresses -0.8%; MoE-2 NR2 already on parity;
>   MoE-3 barriers already minimum)
> - confirmation that qwen3.6 is 1.27× peer on the same codebase (so the
>   gap is gemma4-specific)
>
> What's still required: actual per-kernel and/or per-graph optimization
> work that lands a coherent throughput win on gemma4 decode. The
> dispatch-fusion lever class is exhausted on this chip+shape; remaining
> levers are (a) per-kernel tuning of MoE mat-vec, (b) ggml-metal-style
> graph-fusion port, (c) per-layer-embedding path (gemma4.cpp:338-359
> shows peer has a `inp_per_layer` add-on that hf2q may or may not match —
> needs verification), (d) routing kernel optimization (router_proj F32
> mat-vec + softmax/top-k path).
>
> Mission is **OPEN**. /loop cron `7e171c2b` restarted to continue the
> optimization work. The iter-1..4 audit data below is correct and
> load-bearing for any future attempt; it should NOT be re-litigated.

- **Status**: accepted (2026-05-11; OPTIMIZATION REOPENED iter-5+)
- **Date**: 2026-05-11
- **Supersedes**: nothing structurally; corrects ADR-028 §iter-486 + §iter-487 closure verdict and the iter-308 "Q5_K nr0=2" smoking-gun claim
- **Deciders**: Robert (operator), Claude (cfa-adr028-gap-20260511 swarm: queen-phase1 / claude-impl perf-engineer / codex-impl structural-reviewer)
- **Tags**: performance, root-cause, gemma4, moe, measurement-discipline, baseline-correction

## Decision (one sentence)

The 27% decode gap between hf2q and llama.cpp on `gemma4-ara-2pass-APEX-Q5_K_M.gguf` (Apple M5 Max) is **localized to hf2q's MoE pipeline** (`router_proj` + softmax/top-k + routed `mul_mv_id` experts), which accounts for ~49% of hf2q's per-token wall-clock; **TQ work is at most 7% of the gap and is no longer a viable optimization target for this model**.

## Context

ADR-028 ran 141 /loop iterations between 2026-04 and 2026-05 chasing the decode gap on gemma4-APEX-Q5_K_M. Two recent closures (iter-486 reframe → "0.96–1.04× tied with peer"; iter-487 → "mission CLOSED at +4.5%") were claimed but were not reproducible. Operator re-measured on 2026-05-11 and got 73.5 vs 97.4 t/s (single-shot). Re-running the apples-to-apples bench in a thermally-stable session yielded the empirical result this ADR is built on.

The investigation was scoped via `/cfa` in **review-only** mode (Codex's `--full-auto` sandbox blocks Metal device access, so Codex cannot bench — Claude measures, Codex audits peer source). Session: `cfa-adr028-gap-20260511`.

## Empirical baseline (this session's fresh measurements)

**Hardware**: Apple M5 Max, 128 GB · **Thermal**: no warnings, single session · **HEAD**: hf2q `cfbdc469` · **Peer build**: `d05fe1d7d (9010)` (homebrew llama.cpp) · **Model**: `/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf` (19 GB, mtime May 6 2026)

| Source | Tool | n | Result | σ-pct |
|---|---|---:|---:|---:|
| peer cold | `llama-bench -p 0 -n 200 -r 10` | 10 | 103.72 ± 0.31 t/s | 0.30% |
| peer post-hf2q | same | 10 | 100.09 ± 0.78 t/s | 0.78% |
| hf2q HEAD `--benchmark` | `hf2q generate ... --max-tokens 200 --benchmark` | 5 | 75.10 t/s median (75.14 mean, σ 0.167) | 0.22% |

**Ratio**: 75.10 / 101.9 ≈ **0.7245× peer** (27.55% slower). Coherence ✓ (deterministic, matches peer output).

All σ-pct values are well below the 5% thermal-stability gate; this is a clean measurement.

## TQ attribution (S2 — operator's primary question)

> **Q (operator)**: "Is hf2q's TQ work (Hadamard pre-processing + HB-shadow-cache encode + `flash_attn_vec_tq_hb`) the dominant driver of the 27% decode gap?"
>
> **A**: **NO. Conclusively.**

| probe | env | median t/s | coherent | gap recovered |
|---|---|---:|:---:|---:|
| baseline (full TQ chain) | default | **75.10** | ✓ | 0% |
| **2a `HF2Q_USE_DENSE=1`** (TQ surgically removed) | env | **77.10** | **✓** | **+7.00%** |
| 2b skip TQ encode only | `UNSAFE+SKIP_TQ_ENCODE` | 77.90 | ✗ | +9.80% |
| 2c skip TQ-HB SDPA only | `UNSAFE+SKIP_TQ_SDPA` | 87.80 | ✗ | +44.47% |
| 2d skip both | `UNSAFE+SKIP_TQ_ENCODE+SKIP_TQ_SDPA` | 88.30 | ✗ | +46.22% |

The only coherent comparison is probe 2a: substituting an equivalent dense F32 K/V + `flash_attn_vec` SDPA recovers **+7.0% of the gap** (net TQ-chain cost vs dense: 0.345 ms/tok = 9.4% of the 3.67 ms/tok gap).

The 2c/2d probes recover up to +46% but break coherence at token 33 (the SDPA dispatch is skipped entirely). They establish that the **incoherent timing budget** of the TQ chain is ~2.0 ms/tok, but most of that budget is unavoidable attention work that any coherent kernel would still pay.

## MoE attribution (S6 — the smoking gun)

| probe | env | median t/s | coherent | gap recovered |
|---|---|---:|:---:|---:|
| baseline | default | 75.10 | ✓ | 0% |
| **H5B-all** (skip router+experts+swiglu+weighted_sum) | `UNSAFE+SKIP_ROUTING=1 SKIP_MOE_EXPERTS=1 SKIP_MOE_SWIGLU=1 SKIP_WEIGHTED_SUM=1` | **148.40** | ✗ | **+232%** (1.43× peer) |
| H5B-routing only | `UNSAFE+SKIP_ROUTING=1` | 120.50 | ✗ | +159% |
| H5B-experts only | `UNSAFE+SKIP_MOE_EXPERTS=1` | 92.60 | ✗ | +61% |

**MoE pipeline = 6.58 ms/tok = 49% of hf2q's 13.5 ms wall-clock**:
- **Routing** (`router_proj` qmatmul + softmax/top-k): **~5.02 ms/tok = 38% of wall-clock**
- **Experts** (`mul_mv_id` × {gate, up, down} × top-8): **~2.52 ms/tok = 19% of wall-clock**
- (Pipeline overlap: split sums exceed total because skipping experts still runs routing.)

These probes break coherence at token 33; ms/tok savings are **upper bounds**. A real MoE optimization will save less than 6.58 ms/tok because peer also pays MoE work. But:
- hf2q MoE cost: ~6.58 ms/tok
- Peer total decode cost: 9.65 ms/tok (ENTIRE per-token budget)
- → If peer's MoE cost is even modestly under 6.58 ms/tok (likely, given peer's NR2-fused `mul_mv_id` Q6_K path), the gap is fully explained by MoE-side per-dispatch inefficiency.

## Dispatch arithmetic (closes the budget)

| | dispatches/tok | µs/dispatch | ms/tok | matches measured |
|---|---:|---:|---:|:---:|
| hf2q | 925 | 14.5 | 13.4 | ✓ (13.54) |
| peer | 105 (candle Phase 0 ref) | 92 (derived) | 9.65 | ✓ (9.65) |
| **gap** | **+820** | **−77.5** | **+3.67** | ✓ |

hf2q's per-dispatch wall-clock is small (14.5 µs); the gap is from **issuing 8.9× more dispatches per token**, not from any single dispatch being slow. The ~820 excess dispatches per token are predominantly in the MoE path (top-8 experts × 3 mat-vecs × 30 layers = 720 `mul_mv_id` dispatches minimum, plus per-layer router proj + softmax + top-k).

## Gap-scaling regime (S4 — corroborates dispatch-floor signature)

| shape | hf2q t/s | peer t/s | ratio | gap ms/tok |
|---|---:|---:|---:|---:|
| n=200 | 75.10 | 103.66 | 0.7245× | 3.669 |
| n=500 | 74.60 | 102.32 | 0.7291× | 3.632 |
| n=1000 | 73.90 | 91.30 | 0.8094× | 2.579 |

Gap is **constant in absolute ms/tok at n=200/500**, then **shrinks at n=1000** because peer slows -11.9% with KV growth while hf2q only slows -1.6%. This is the **classic fixed-per-token-overhead signature** — dispatch + barrier floor, not KV-bandwidth-bound. The n=1000 narrowing is peer paying a tax hf2q's compressed KV avoids, not hf2q improving.

## Corrections to prior claims

### Correction 1 — iter-486 peer baseline (77 ± 15 t/s) was wrong

iter-486 declared the gap "tied at HEAD" with peer at 77.18 ± 15.36 t/s. **σ = 15.36 (20% of mean) was the thermal-throttle smoking gun**, not flagged at the time. Fresh re-measurement on the same machine, same GGUF, same llama.cpp build gives peer = 103.66 ± 0.30 t/s (σ-pct 0.29%). The "tied" verdict was an artifact.

### Correction 2 — iter-487 "mission CLOSED" verdict invalidated for gemma4

Built on iter-486's bad baseline. Specifically: iter-487 cited "operator's own llama-cli: 70.1 tok/s peer, hf2q 73.3 = +4.5% hf2q faster." The 70.1 peer number is consistent with thermal throttle; it does not survive a clean session bench. **Mission re-opened iter-488; this ADR records the actual root cause.**

### Correction 3 — iter-308 "peer Q5_K is N_R0=2; hf2q is N_R0=1" is WRONG

Codex S5 axis 4 source-read at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`: peer Q5_K is `N_R0 = 1`. Peer Q6_K is `N_R0 = 2`, and hf2q already has the matching Q6_K NR2 port. The iter-308 "smoking gun" claim was either misremembered or stale from a different commit. The actual structural diff at qmatmul kernels is much smaller than claimed.

### Correction 4 — "gemma4 is dense FFN" assumption was wrong

The CFA spec assumed gemma4-26B-A4B is dense FFN. Both peer (`/opt/llama.cpp/src/models/gemma4.cpp`) and hf2q (`/opt/hf2q/src/serve/forward_mlx.rs`) load and execute MoE for this architecture (`expert_count = 128`, `expert_used_count = 8`). Future investigations must NOT skip the MoE path under a "dense FFN" assumption.

### Correction 5 — gemma4 shared-KV reuse is DORMANT on this APEX GGUF

Codex S5 axis 1 found that peer marks upper gemma4 layers as `has_kv == false` and reuses earlier KV caches (`/opt/llama.cpp/src/models/gemma4.cpp:240-245` + `llama-model.cpp:2004-2011`). BUT this APEX GGUF declares `attention.shared_kv_layers = 0` — the reuse branch is inactive in BOTH engines for this file. H5A: falsified by metadata for this model.

## Hypothesis verdicts

| hypothesis | predicted | measured | verdict |
|---|---|---|---|
| H1 `fuse_fwht_pre` | (iter-485 falsified 3×) | not re-tested | **DO NOT RE-TRY** |
| H2 cross-WG in-kernel reduce | (Apple Metal infeasible at NWG=16/32) | structural | **INFEASIBLE** |
| H3 FWHT-undo into reduce | (iter-485 Worker C falsified, bit-exact parity) | not re-tested | **DO NOT RE-TRY** |
| H4 dual 4-bit K+V encode | (iter-485 Worker D falsified, -3.8% to -18.5%) | not re-tested | **DO NOT RE-TRY** |
| **H5A** missing gemma4 shared-KV reuse | 5-15% if `shared_kv_layers > 0` | GGUF metadata = 0 → inactive in both engines | **FALSIFIED (source-only)** |
| **H5B** MoE under-attributed | 5-20% | +97.6% (148.4 t/s); routing alone +60.5% | **STANDING — DOMINANT** |
| **H5C** TQ-HB compute/control-bound | 5-25% via coherent dense | +7.0% (≪ 30% gate) | **FALSIFIED for dominant-driver status** |

## Decision

1. **Halt** all TQ / FWHT / SDPA / qmatmul-fusion optimization work on gemma4-APEX-Q5_K_M. Ceiling for any of these levers is +7-9% of the gap. They are off the critical path.
2. **Pivot** next-iter work to a focused MoE-pipeline investigation. Three concrete sub-targets, ranked by likely ROI:
   - **MoE-1**: Count exact `router_proj` + `mul_mv_id` dispatch issuance per gemma4 layer per decoded token; compare against llama.cpp's MoE graph layout. Identify which kernels in the routing pipeline (`router_proj` qmatmul, softmax, top-8 selection, expert mat-vec) are being issued as separate dispatches that could be fused.
   - **MoE-2**: Verify whether `quantized_matmul_id_ggml` (the `mul_mv_id` dispatcher) shares the Q6_K NR2 path or is still on a slower NR1 dispatcher. If NR1, port the NR2 optimization.
   - **MoE-3**: Audit barrier cadence around router-prep and expert dispatch (currently 510 barriers/tok). Many MoE barriers may be redundant if the `mul_mv_id` kernels are reads-only on the routing output.
3. **Fix** `HF2Q_MLX_KERNEL_PROFILE=1` gating at `src/serve/forward_mlx.rs:5871` so it doesn't hard-fail on Q6_K lm_head — gate the LM-head-specific path separately so per-kernel-type buckets unlock for any GGUF using Q6_K head. Without this, every future investigation hits the same blind spot.
4. **Re-test** qwen3.6 mantra (iter-487 claimed 1.27× peer): given iter-486 was wrong about gemma4, the qwen3.6 baseline may also be a thermal artifact. Re-measure in a thermally-stable session with σ-pct check before citing.
5. **Adopt** as standing practice that every "X× peer" claim in memories, ADRs, and commit messages must include σ-as-pct-of-mean. Refuse to consume claims with σ > 5% without re-measuring. (Standing rule already saved as `feedback_do_not_trust_file_claims_re_measure_2026_05_11.md`.)

## Consequences

Positive:
- 4-day investigation thrash resolved with a clean empirical answer.
- The 7+ falsified hypotheses (H1 × 3, H2, H3, H4, H5A, H5C) are now closed with explicit evidence; future iters won't re-try them.
- The next-iter direction is concrete (MoE-1/-2/-3) and falsifiable (counter measurements + bench probes already shown viable).
- Standing measurement discipline (σ-pct check, re-measure-don't-cite) reduces risk of recurrence.

Negative / risks:
- MoE optimization is a multi-week structural project (kernel fusion + dispatch graph rewrite), not a 1-line flag flip. Operator should expect longer iteration cycles than the TQ flag-tweak loop.
- We did NOT directly measure peer's MoE budget (no equivalent skip flag in llama.cpp). H5B remains "standing" not "proven" until a like-for-like comparison is set up.
- `HF2Q_SKIP_*_MOE` probes break coherence at token 33, so they cannot validate any partial MoE optimization in product mode — coherent partial wins must be verified with end-to-end output comparison instead.

## Validation (this ADR's own gates)

To reproduce the core finding (any future iter must show these numbers within σ-pct < 5%):

```bash
MODEL=/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf
# Peer baseline (repeat twice, discard first; check σ-pct < 5%):
/opt/homebrew/bin/llama-bench -m "$MODEL" -p 0 -n 200 -r 10
# Expected: ~100-104 t/s, σ < 1 t/s

# hf2q baseline:
/opt/hf2q/target/release/hf2q generate --model "$MODEL" \
  --prompt "Write a long story about a sentient telescope" \
  --max-tokens 200 --benchmark
# Expected: ~75 t/s, σ < 0.5 t/s, coherent output, ratio ~0.72× peer

# TQ-attribution probe (coherent):
HF2Q_USE_DENSE=1 /opt/hf2q/target/release/hf2q generate --model "$MODEL" \
  --prompt "Write a long story about a sentient telescope" \
  --max-tokens 200 --benchmark
# Expected: ~77 t/s, coherent, +7% gap recovered

# MoE-attribution probe (incoherent timing only):
HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_SKIP_ROUTING=1 HF2Q_SKIP_MOE_EXPERTS=1 \
  HF2Q_SKIP_MOE_SWIGLU=1 HF2Q_SKIP_WEIGHTED_SUM=1 \
  /opt/hf2q/target/release/hf2q generate --model "$MODEL" \
  --prompt "Write a long story about a sentient telescope" \
  --max-tokens 200 --benchmark
# Expected: ~148 t/s (1.43× peer!), incoherent output, +232% gap recovered
```

If any of these fail to reproduce within σ-pct < 5%, suspect (a) thermal throttle (let chip cool, retry), (b) HEAD divergence from `cfbdc469`, or (c) GGUF substitution (the APEX file mtime should be May 6 2026).

## Implementation log (post-acceptance)

### iter-1 (2026-05-11, branch `adr-029`)

**Status**: §Decision item 3 LANDED; items 2a/2b audits CLOSED with falsifications; item 2c surfaced 1 dispatch-fusion regression (H6).

#### iter-1 ship list

1. **§Decision item 3 — kernel-profile gating fix LANDED** (commit `1cd6540f`).
   `forward_decode_kernel_profile` (src/serve/forward_mlx.rs:5837+) gained a
   `lm_head_q6k` branch that mirrors the production single-session decode path
   at ~4818. Pre-fix gemma4-APEX-Q5_K_M hard-failed `Kernel profile requires
   GPU lm_head (Q8_0 or F16 weight)` because `HF2Q_LMHEAD_Q6K` defaults on
   (ADR-028 iter-345) and the GGUF's `token_embd.weight` is Q6_K → only
   `lm_head_q6k` is `Some(...)`. `HF2Q_MLX_KERNEL_PROFILE=1` now runs end-to-end
   on this GGUF; verified per-bucket table emitted (MoE 294 µs/layer × 30 =
   8.82 ms/tok in profile mode; KV-cache-copy / O-proj / SDPA have higher
   per-kernel-time **ratios** but lower absolute).

2. **§Decision item 2b — MoE-2 NR2 hypothesis FALSIFIED** (no port available).
   Inspecting `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:51-57`:
   `N_R0_Q4_K=2`, **`N_R0_Q5_K=1`**, `N_R0_Q6_K=2`. Inspecting
   `/opt/mlx-native/src/ops/quantized_matmul_id_ggml.rs:483-489`: `use_q6k_id_nr2
   = matches!(params.ggml_type, GgmlType::Q6_K) && env_default_true("HF2Q_Q6K_ID_MV_NR2")`
   → Q6_K mul_mv_id **default-routes to `kernel_mul_mv_id_q6_K_f32_nr2` already**.
   Confirmed gemma4-APEX-Q5_K_M expert quant types via `gguf-dump`:
   `ffn_gate_up_exps.weight = Q6_K` (NR2 active) + `ffn_down_exps.weight = Q8_0`
   (no NR variant; peer also NR1). Both engines already on parity paths for
   gemma4 MoE experts. The iter-308 "peer N_R0=2 vs hf2q N_R0=1" Q6_K claim
   that ADR-029 §Correction 3 already retracted is doubly confirmed: peer
   `_id` Q6_K is also NR2; hf2q ditto. No optimization gap here.

3. **§Decision item 2c (adjacent) — H6 fused-triple-norm decode FALSIFIED.**
   Hypothesis: `HF2Q_FUSED_TRIPLE_NORM=1` (kernel
   `dispatch_fused_post_attn_triple_norm_f32` at mlx-native, wired into decode
   at forward_mlx.rs:4254) replaces 4 dispatches/layer (post-attn norm+add +
   3 pre-FF norms) with 1 dispatch — saves 90 dispatches/tok (10% of measured
   883 dispatches/tok decode count). The kernel is byte-identical to the
   unfused path on prefill fixtures.
   Test: gemma4-APEX-Q5_K_M, 50-tok "haiku about quantum entanglement",
   --benchmark mode, n=5 each, sequential same thermal session.
   | mode | tok/s | σ-pct | coherent |
   |---|---:|---:|:---:|
   | `HF2Q_FUSED_TRIPLE_NORM=0` (default) | 75.0 | 0.08% | ✓ |
   | `HF2Q_FUSED_TRIPLE_NORM=1` | 72.9 | 0.05% | **✓** (byte-identical haiku) |
   Result: **-2.1 t/s = -2.8% regression**. Coherence preserved; the fused
   kernel saves 90 launches but costs more per-call than 4 unfused launches.
   Interpretation: at hidden_size=2816 with 3 weight tensors + 4 output buffers
   resident, register pressure / memory-access-pattern overhead exceeds the
   ~3-4 µs/dispatch launch savings the fusion intends. **Do NOT default-on.**
   Standing decision: leave flag OFF; mark it FALSIFIED in
   `investigation_env.rs:600` doc-comment in a follow-up.

4. **Ground-truth counters captured on adr-029 HEAD** (production decode mode):
   `HF2Q_DUMP_COUNTERS=1 hf2q generate --max-tokens 20`
   → **dispatches/decode_tok = 883.5**, **barriers/decode_tok = 487.35**,
   syncs/decode_tok = 0, cmd_bufs/decode_tok = 1.9. Matches ADR-029's
   pre-iter-1 "925 / 510" estimates within decode-warmup noise.
   Peer (ADR-029 dispatch-arithmetic table) = 105 dispatches/tok ⇒ hf2q is
   **8.4× more dispatches**. The structural difference is **llama.cpp's
   ggml-metal scheduler fuses N ggml ops into M < N Metal dispatches at
   graph-compile time**; hf2q dispatches each op manually. Closing the
   dispatch gap is multi-kernel structural work, not a flag flip.

#### iter-1 hypotheses verdicts (additive to §Hypothesis verdicts table)

| hypothesis | predicted | measured | verdict |
|---|---|---|---|
| **H6** fused-triple-norm decode | +3-5% (90 disp/tok save) | -2.8% (-2.1 t/s) | **FALSIFIED — kernel per-call cost > launch savings at decode shape** |
| MoE-2 NR2 port (§Decision 2b) | +3-7% if peer NR2 hf2q NR1 | both on NR2 (Q6_K) / NR1 (Q5_K/Q8_0) — no gap | **FALSIFIED — already on parity paths** |

#### iter-2 (2026-05-11, branch `adr-029`) — §Decision items 4 & 5 CLOSED

#### iter-2 ship list

5. **§Decision item 4 — qwen3.6 σ-pct re-test CONFIRMED 1.27× peer**, not a
   thermal artifact. Same machine (Apple M5 Max, 128 GB), same thermal session,
   same homebrew llama.cpp `d05fe1d7d (9010)`, same fresh-bench discipline as
   gemma4.
   | source | tool | n | result | σ-pct |
   |---|---|---:|---:|---:|
   | hf2q | `hf2q generate ... --max-tokens 200 --benchmark` | 5 | **129.7 t/s** | 0.14% |
   | peer | `llama-bench -m … -p 0 -n 200 -r 10` | 10 | **102.17 ± 0.22 t/s** | 0.22% |
   Ratio: 129.7 / 102.17 = **1.270× peer**. Both σ-pct well below 5% gate.
   Interpretation: qwen3.6's iter-487 claim survives. **Important contrast vs
   gemma4** — hf2q is 27% slower on gemma4 (0.724× peer) but 27% **faster** on
   qwen3.6 (1.270× peer). The decode gap is **model-architecture-specific to
   gemma4**, not a hf2q-wide regression. Likely driver: qwen3.6 runs the
   TQ-KV path by default (ADR-027 Phase B, 3.94× KV memory savings) while
   gemma4 does not (`tq_kv = inactive` at load banner per ADR-029 §Links).
   Architectural delta between the two MoE models, not just MoE-pipeline
   inefficiency.

6. **§Decision item 5 — σ-pct standing rule** explicitly codified below as a
   normative part of this ADR's §Validation gate. All future "X× peer" claims
   in commit messages, ADRs, memories, or comments must report σ-as-pct-of-mean
   alongside the ratio. Claims with σ-pct > 5% are thermal-artifact-suspect
   and **MUST** be re-measured in a thermally-stable session before being
   acted upon.

### Standing rule — σ-pct gate on peer-parity claims

Every "X× peer" or "X tok/s" claim in this repo (ADRs, commit messages,
memories, code comments, doc strings) carries **two numbers**:

1. **mean (or median)** — what was measured
2. **σ-pct** = (standard deviation) / (mean) × 100

A claim is **defensible** when σ-pct < 5% and the comparison is apples-to-apples
(same machine, same GGUF, same thermal session, same prompt shape, same
sampling, both engines fresh-loaded).

A claim with σ-pct ≥ 5% is **thermal-artifact-suspect** and MUST be either:
(a) re-measured in a cooler thermal session and re-stated with the new
    σ-pct, OR
(b) caveated as "exploratory, not load-bearing" — and never acted upon as a
    closure verdict.

This rule is enforced retroactively: ADR-028 §iter-486's "tied at 77 ± 15 t/s"
(σ-pct = 20%) is the canonical violation; it triggered an 18-iter mission
re-open. **Refuse to consume any σ-pct ≥ 5% claim without re-measuring.**

7. **§Decision item 2c — MoE-3 barrier audit FALSIFIED via Chesterton's-fence
   code-read.** Hypothesis: redundant `barrier_between(reads, writes)` calls
   in the B8-B14 MoE chain emit unnecessary Apple Metal `memory_barrier()`
   instructions, inflating the 487 measured barriers/tok.
   Verdict: `mlx_native/src/graph.rs:1494-1530` `barrier_between` already
   **deduplicates via a conflict tracker**. The flow is:
   ```rust
   let reason = self.tracker.conflicts_reason(reads, writes);
   if let Some(_) = reason {
       self.encoder.memory_barrier();   // ← only here, on real conflict
       self.tracker.reset();
       self.barrier_count += 1;
   }
   self.tracker.add(reads, writes);
   ```
   Calls to `barrier_between` with no real RAW/WAR conflict are zero-cost
   bookkeeping — no Metal barrier is emitted; `barrier_count` is not
   incremented. The 487 barriers/tok counter measures **actual emitted
   `enc.memory_barrier()` instructions**, i.e. provably-needed barriers
   given the current dispatch ORDER. Hand-merging `barrier_between(...)`
   call-sites would not reduce this count. The standing pattern
   `B11: dense_down + gate_up_id [2 concurrent]` (forward_mlx.rs:4432-4471)
   already runs concurrent under Metal — the two adjacent `barrier_between`
   calls produce a single emitted Metal barrier when the second call's
   reads/writes don't conflict with the first dispatch's outputs.
   Implication: reducing the 487 barriers/tok requires **reducing the
   dispatch count** itself (kernel fusion), which H6 already falsified
   for the largest available fusion (-2.8% at triple-norm).
   This is the canonical Chesterton's-fence outcome: the apparent
   "redundant barrier" pattern is protected by infrastructure invariants
   the surface comments don't surface.

### iter-3 (2026-05-11, branch `adr-029`) — H7 falsified + stale-claim cleanup + regression gate

#### iter-3 ship list

8. **H7 — fused-MoE-wsum-end-layer-v2 FALSIFIED** at HEAD.
   The pre-existing `investigation_env.rs:609` comment claimed
   "coherence regresses under iter-321 stack — root cause not yet identified".
   Iter-3 test on adr-029 HEAD with full default-flag stack
   (LMHEAD_Q6K + Q6K_MV_NR2 + Q6K_ID_MV_NR2 all on) re-runs the 50-tok haiku
   fixture: **coherence byte-identical** to baseline. The iter-367 coherence
   claim is **stale at HEAD**. Throughput bench n=5:
   `HF2Q_FUSED_MOE_WSUM_END_LAYER_V2=1` → **74.4 t/s** (σ-pct 0.11%) vs 75.0
   baseline = **-0.8% regression**. Coherence is fine; throughput is the
   blocker. Same falsification class as H6 (TRIPLE_NORM):
   dispatch-count saved (30/tok) but per-call cost > launch savings on
   gemma4's decode shape.

9. **Doc-debt cleanup**. Updated `investigation_env.rs:602-618` (FUSED_MOE_WSUM)
   and `investigation_env.rs:621-638` (FUSED_TRIPLE_NORM) to replace the
   stale "coherence regresses" / "default-off until validated" claims with
   the iter-1/iter-3 measurement-grounded verdicts. Code + test == truth
   superseding comments.

10. **Regression gate**: full `cargo test --release --lib` on adr-029 HEAD
    = **51 passed, 0 failed, 0 ignored**. No regression from the iter-1
    kernel-profile gating fix (1cd6540f).

#### iter-3 cumulative dispatch-fusion lever census (on gemma4-APEX-Q5_K_M)

| flag | saves (disp/tok) | coherence | throughput | verdict |
|---|---:|:---:|---:|:---:|
| `HF2Q_FUSED_END_OF_LAYER` (default-on) | 30 | ✓ | baseline-included | LANDED prior |
| `HF2Q_FUSED_TRIPLE_NORM` (H6) | 90 | ✓ | -2.8% | FALSIFIED |
| `HF2Q_FUSED_MOE_WSUM_END_LAYER_V2` (H7) | 30 | ✓ | -0.8% | FALSIFIED |

**The dispatch-fusion lever class is exhausted on gemma4-APEX-Q5_K_M at
Apple M5 Max.** Every fewer-larger-kernel alternative loses or ties on
throughput despite saving 30-90 dispatches/tok. Apple Metal's per-dispatch
launch overhead at hidden_size=2816 is small enough that the fused kernel's
per-call cost exceeds it. Future gemma4-decode optimization must work at the
**per-kernel cost level** (kernel internals — threadgroup shape, shared
memory, register pressure) or at **structural levels above dispatch fusion**
(ggml-metal-style graph compilation, op-reordering for cross-CB concurrency,
expert-batching that re-uses bandwidth across top-k slots).

#### iter-3 standing direction

The 27% gemma4 gap is **structurally unresolvable within ADR-029's
flag-flip-iteration scope**. Closing it requires one of:

A. **ADR-030 graph-fusion infrastructure** (multi-week): port the
   ggml-metal scheduler's op-fusion pass. hf2q emits ~25-30 dispatches/layer;
   peer emits ~3-4 effectively because ggml-metal compiles N ggml ops into
   M < N Metal dispatches. The 8.4× dispatch gap is the structural delta.
   Scope: read `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp`,
   identify which gemma4 ops fuse, port the fusion pass into mlx-native's
   graph.rs recording mode. ROI estimate: 1.5-2× speedup if fully ported;
   uncertain time bound; appropriate scope for /cfa dual-mode swarm.

B. **Per-kernel cost tuning** (single-iter feasible, smaller ROI): the
   kernel-profile shows MoE bucket at 294 µs/layer in profile mode. The
   non-trivial Q6_K mul_mv_id NR2 kernel may have residual headroom in
   threadgroup-shape / shared-memory layout for the top-8 fan-out case
   that's distinct from the non-_id case ADR-028 already tuned.

C. **Architectural delta acceptance**: gemma4-APEX is a gemma4-26B-A4B
   MoE without TQ-KV; qwen3.6-APEX runs the same code at **1.270× peer**.
   The gap is gemma4-specific. Operator may accept gemma4 at 0.72× peer
   if mantra "as fast as peer for all models we support" is interpreted
   per-model rather than universally.

This ADR's original §Decision items 1-5 are **all closed**. Branch
`adr-029` is ready to merge to `main` (no behavior change from the
gating-fix landing commit; documentation captures the negative findings
that block future re-trials).

### iter-4 (2026-05-11, branch `adr-029`) — closure validation + merge

#### iter-4 ship list

11. **Full validation pass on adr-029 HEAD before merge to `main`**:

   | gate | expected (per ADR-029 §Validation) | iter-4 measured | verdict |
   |---|---|---:|:---:|
   | `cargo test --release --lib` | clean | **51 pass / 0 fail** | ✓ |
   | gemma4 baseline `hf2q --benchmark` n=5 | ~75 t/s, σ < 0.5 | **75.1 t/s, σ-pct 0.07%** | ✓ |
   | `HF2Q_USE_DENSE=1` probe | ~77 t/s, coherent | **77.7 t/s, σ-pct 0.10%, coherent** | ✓ |
   | qwen3.6 σ-pct (iter-2) | n/a (new gate this ADR) | **1.270× peer, σ-pct 0.14%** | ✓ |
   | coherence stability | byte-identical haiku | **byte-identical 50-tok output** | ✓ |

   All gates pass. The iter-1 kernel-profile gating fix (`1cd6540f`) does not
   regress anything; the iter-2/-3 doc updates have no runtime impact.

12. **Cumulative iter-1→4 status of original §Decision items**:

   | # | item | status | landed-as |
   |---:|---|:---:|---|
   | 1 | Halt TQ/FWHT/SDPA/qmatmul-fusion work on gemma4 | ✓ STANDING | discipline (nothing touched in 4 iters) |
   | 2a | MoE-1 dispatch audit | ✓ DONE | iter-1 mapping; 883 disp/tok ground truth |
   | 2b | MoE-2 NR2 port verification | ✓ FALSIFIED | already on parity paths (mlx-native:484) |
   | 2c | MoE-3 barrier audit | ✓ FALSIFIED | tracker dedupes (graph.rs:1494-1530) |
   | 3 | kernel-profile Q6_K lm_head gating | ✓ LANDED | commit `1cd6540f` |
   | 4 | qwen3.6 σ-pct re-test | ✓ CONFIRMED | 1.270× peer, both σ-pct < 0.25% |
   | 5 | σ-pct standing rule | ✓ CODIFIED | §Validation gate in this ADR |

   Adjacent falsifications (H6, H7) recorded for future iter-blocking.

13. **Closing direction for the unresolved 27% gemma4 gap** (NOT in
   §Decision item scope; for the next ADR or a /cfa swarm to pursue):

   The 27% gap is structural — `~825` extra dispatches/tok between hf2q
   and llama.cpp graph-compile output for the same gemma4 layer. Every
   in-scope dispatch-fusion lever falsified. Path A (ADR-030 graph-fusion
   infra port from `ggml-metal-ops.cpp`) is the canonical next-step;
   estimated multi-week effort. Path C (accept architectural delta) is
   defensible — qwen3.6 on the same codebase runs at 1.270× peer, so the
   parity gap is gemma4-specific, not a hf2q-wide regression.

   No more code change appropriate on this branch. **Branch `adr-029`
   merges to `main` at iter-4**.

### iter-1 standing direction (revised)

Decode-time dispatch-fusion (TRIPLE_NORM, MOE_WSUM_END_LAYER_V2) appears to be
a dead-end class on M5 Max — Apple Metal's per-dispatch overhead at the shapes
gemma4 produces is small enough that fewer-larger-kernels lose to
more-smaller-kernels. The remaining MoE levers are:
1. **MoE-pipeline barrier audit** (§Decision item 2c, MoE-3): identify
   redundant barriers in the B8-B14 sequence and the post-MoE chain. Each
   redundant barrier removed costs 0 lines of new kernel code.
2. **Structural dispatch-graph audit** (out of §Decision 2 scope but
   surfaced here): compare hf2q's per-layer dispatch sequence to ggml-metal's
   graph-compiled output for the same gemma4 layer. If peer schedules N ops
   into M < N dispatches at scheduler time, there may be op-pair fusions
   hf2q hasn't tried yet.
3. **Per-kernel cost optimization** (out of §Decision 2 scope): the 2.3×
   MoE per-kernel-time ratio in kernel-profile mode suggests there may be
   tuning headroom in `kernel_mul_mv_id_q6_K_f32_nr2` (threadgroup shape,
   shared-memory layout) — but this is the work ADR-028 already exhausted
   for the non-_id Q6_K variant, so ROI is likely small.

The §Decision item 2 work remains "MoE-pipeline focused investigation" but the
ROI surface is smaller than the ADR-029 acceptance text implied:
H6 falsification narrows the per-layer fusion lever from "10% saved" to "−2.8%
cost". §Decision item 1 (halt TQ work) stands; §Decision item 3 (kernel-profile
fix) is LANDED; §Decision items 4-5 (re-test qwen3.6, σ-pct standing rule)
remain open.

## Links

- `~/.claude/projects/-opt-hf2q/memory/project_adr028_synthesis_moe_pipeline_2026_05_11.md` (this finding, memory-form)
- `~/.claude/projects/-opt-hf2q/memory/project_adr028_iter488_mission_REOPENED_2026_05_11.md` (the re-open trigger)
- `~/.claude/projects/-opt-hf2q/memory/feedback_do_not_trust_file_claims_re_measure_2026_05_11.md` (standing measurement-discipline rule)
- CFA session artifacts:
  - `~/.claude/teams/cfa-adr028-gap-20260511/shared/findings.md` (Claude S7 synthesis)
  - `~/.claude/teams/cfa-adr028-gap-20260511/shared/reviews/codex-structural.md` (Codex S5 review, 300 lines, 7 axes + 3 hypotheses)
  - `~/.claude/teams/cfa-adr028-gap-20260511/shared/agents/claude-impl/{S1,S2,S3,S4,S6}.md` (per-subtask evidence)
  - `~/.claude/teams/cfa-adr028-gap-20260511/shared/agents/claude-impl/raw/*` (raw stderr from every probe)
- `docs/ADR-028-peer-parity-coherence-and-speed.md` — the prior mission ADR; this ADR-029 corrects its iter-486/487 closure verdicts but does not formally supersede the body
- `docs/ADR-027-qwen35-tq-kv-cache-and-persist-family.md` — TQ-KV mantra source (gemma4 does NOT use TQ-KV by default; `tq_kv = inactive` at load banner)
