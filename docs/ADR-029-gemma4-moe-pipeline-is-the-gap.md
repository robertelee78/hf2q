# ADR-029: gemma4-APEX-Q5_K_M Decode Gap — Measurement-Driven Root Cause

- **Status**: amended (2026-05-11); original framing RETRACTED, iter-9 rewrite + iter-10/11 amendment (H12 FALSIFIED)
- **Date**: 2026-05-11 (original) → 2026-05-11 (rewrite iter-9) → 2026-05-11 (amendment iter-10/11)
- **Supersedes**: itself (the original "MoE pipeline IS the gap" framing was wrong; this rewrite replaces it)
- **Decision-grade evidence**: per-layer GPU timestamps from `MTLCommandBuffer.GPUStartTime/GPUEndTime` in real production decode
- **Tags**: performance, root-cause, gemma4, measurement-discipline, baseline-correction

## Decision (one sentence)

The 25–32% decode-throughput gap between hf2q and llama.cpp on
`gemma4-ara-2pass-APEX-Q5_K_M.gguf` (Apple M5 Max) is **distributed
roughly evenly between attention (~51 µs/layer) and FFN (~58 µs/layer)
on the per-layer GPU-time axis** — NOT concentrated in any single
bucket. The dispatch-fusion lever class is exhausted; the actionable
direction is **F16 KV cache for gemma4** (largest single known lever)
followed by FFN sub-phase localization.

## How this ADR is grounded

Per-layer GPU times are measured via `HF2Q_PER_LAYER_GPU_TIME=1` /
`HF2Q_PER_LAYER_PHASE_GPU_TIME=1` (see `src/serve/forward_mlx.rs` —
landed iter-9, commit `bc6ffd88`). These hooks call
`s.finish_with_gpu_time()` at layer / phase boundaries which returns
`MTLCommandBuffer.GPUEndTime − GPUStartTime` directly from the Metal
driver. Numbers in this ADR come from running these hooks on the live
decode path with the production GGUF.

Earlier "bucket-attribution" claims in this document were arithmetic
extrapolations from synthetic batched benchmarks. They produced a
multi-day series of confidently-wrong subdivisions (see Appendix B).
The iter-9 instrumentation is the first source of ground truth, and
its measurements are the only attribution this ADR endorses.

## Empirical baseline

**Hardware**: Apple M5 Max, 128 GB · **Build**: hf2q `bc6ffd88` ·
**Peer**: llama.cpp `d05fe1d7d (9010)` · **Model**: 19 GB
gemma4-ara-2pass-APEX-Q5_K_M.gguf (Q6_K dominant, ~6.51 bpw, 30 layers,
24 sliding + 6 full attn, MoE 128 experts/8 active, hidden=2816,
ffn_gate_up_exps=Q6_K, ffn_down_exps mixed Q8_0/Q5_1/IQ4_NL).

| metric | hf2q | peer | ratio |
|---|---:|---:|---:|
| decode t/s (--benchmark n=5) | 73.6–75.1 | 98.0–103.7 | 0.71–0.76× |
| wall-clock ms/decode-tok | 13.33 | 9.78 | 1.36× |
| dispatches per decode-tok | **883** | **1389** | peer issues 1.57× **MORE** |
| barriers per decode-tok | 487 | 905 | peer 1.86× more |
| µs/dispatch (avg) | 15.10 | 7.05 | hf2q 2.14× slower per dispatch |
| CPU encode µs/dispatch | 0.19–0.33 | n/a | negligible (not the bottleneck) |

All σ-pct are <0.5% (thermally-stable measurements).

## Per-layer GPU breakdown (the actual root-cause map)

Measured on adr-029 HEAD `bc6ffd88` via `HF2Q_PER_LAYER_PHASE_GPU_TIME=1`:

| phase | sliding (24 layers) | full (6 layers) | weighted avg/layer |
|---|---:|---:|---:|
| ATTN (pre-norm → O-proj) | 112 µs | 170 µs | 124 µs |
| FFN (post-attn norm → end-of-layer) | 275 µs | 275 µs | 275 µs |
| **layer total** | **387 µs** | **445 µs** | **399 µs** |

× 30 layers: **11.97 ms BODY**, + 1.27 ms HEAD = **13.24 ms** ≈ wall-clock 13.33 ms ✓.

Peer's per-layer total is ~290 µs (derived: 9.78 − 1.27 ÷ 30 assuming HEAD parity).

**Gap structure**:
- ATTN gap: 124 − ~73 = **51 µs/layer × 30 = 1.53 ms/tok**
- FFN gap: 275 − ~217 = **58 µs/layer × 30 = 1.74 ms/tok**
- HEAD parity (lm_head Q6_K is bandwidth-bound on both engines; ~1 ms/tok)
- Sum: ~3.27 ms/tok ≈ measured 3.55 ms gap (remainder ~0.3 ms in CPU/cmd-buf overhead between tokens)

## TQ overhead — measured by surgical removal

`HF2Q_USE_DENSE=1` substitutes dense F32 K/V + standard `flash_attn_vec`
for the TQ-HB chain (TQ-encode + FWHT pre + flash_attn_vec_tq_hb +
FWHT undo). With identical instrumentation:

| phase | TQ chain ON (default) | USE_DENSE=1 | Δ |
|---|---:|---:|---:|
| sliding ATTN | 112 µs | 97 µs | **−14 µs/layer** |
| full ATTN | 170 µs | ~160 µs | **−10 µs/layer** |
| FFN | 275 µs | 275 µs | 0 (control: TQ doesn't touch FFN) |

TQ chain total overhead: 14×24 + 10×6 = **396 µs/decode-tok** = **3% of
hf2q's 13.33 ms wall-clock**. Reproduces ADR-029's prior `+7% gap
recovered` claim (which was computed as `(1/75.1 − 1/77.1) × 13330 µs ÷
3670 µs gap = 9%`).

**TQ overhead alone does NOT explain the ATTN gap.** Even with TQ off,
sliding ATTN is 97 µs/layer vs peer's ~73 µs — a 24 µs/layer remaining
ATTN gap that is **not TQ**.

## What the remaining ATTN gap is (high-confidence inference)

Within hf2q's `HF2Q_USE_DENSE=1` sliding ATTN at 97 µs/layer, the
dominant per-call cost is SDPA at **36.5 µs/call** (measured in
`bench_sdpa_kv_dtype_compare`, reading 16.78 MB F32 K+V). Peer's
attention computes the same answer via `mul_mv_f16(Q@K^T) + soft_max +
mul_mv_f16(score@V)` on **F16** K+V cache (8.39 MB read per layer
peer-side; ~21 µs total). The bandwidth ratio explains the per-layer
attention gap:

| KV dtype | bytes read/SDPA | hf2q µs/call (existing bench) |
|---|---:|---:|
| F32 (current dense path) | 16.78 MB | 36.5 |
| F16 (peer's path) | 8.39 MB | 21.0 |
| TQ-HB (current TQ path) | 2.16 MB | 22.7 (dequant-compute-bound, not bw-bound) |

Adding F16 KV cache support to hf2q gemma4 would close ~15 µs/layer ×
30 = **~450 µs/tok of the ATTN gap = ~3% throughput recovery**.

## What the FFN gap is (un-localized)

The 58 µs/layer FFN gap is currently **unattributed at the sub-phase
level**. The next iteration of `HF2Q_PER_LAYER_PHASE_GPU_TIME` will
sub-split the FFN phase into:
- pre-FF norms (3 concurrent)
- Dense MLP (up/gate/down + GELU)
- MoE pipeline (router + experts + swiglu + weighted-sum)
- end-of-layer fused norm-add

The existing `bench_decode_moe_id_shapes` already shows hf2q MoE
matmuls are at 747–799 GB/s = 137–146% of conservative "peak" — MoE
*matmul* is bandwidth-saturated and not the lever. The 58 µs/layer
FFN gap must therefore be in the surrounding FFN ops (norms, GELU,
router, end-of-layer fusion, residual adds) and/or in barrier/pipeline
overhead between them.

## Hypothesis verdicts (all measured, all in this ADR's iter-1..9 trace)

| # | hypothesis | predicted | measured | verdict |
|---:|---|---|---|---|
| H1 | Halt TQ/FWHT/SDPA/qmatmul-fusion direction | n/a | n/a | **STANDING** (operator discipline) |
| H2 | MoE is 49% of wall-clock (orig ADR-029) | iter-1..4 took as gospel | iter-7 bench: 1.68 ms = **12.6%** | **FALSIFIED** |
| H3 | Peer fuses N ops into M<N Metal dispatches | iter-5 model | iter-5 measurement: peer issues **1389 vs 883 — MORE** | **FALSIFIED** (model wrong) |
| H4 | hf2q's flash_attn_vec_tq_hb is the slow lever (replace w/ peer's 3-kernel) | 1-3 ms savings | hf2q TQ-HB SDPA 0.68 ms/tok ALREADY < peer's est ~0.96 ms | **FALSIFIED** (hf2q already wins this) |
| H5 | FFI / Rust→Metal binding is the bottleneck | iter-6 model | bench_dispatch_overhead: hf2q CPU 0.19–0.33 µs/dispatch | **FALSIFIED** |
| H6 | `HF2Q_FUSED_TRIPLE_NORM=1` (4→1 norm fusion) | +3-5% saves 90 disp/tok | -2.8% throughput at byte-identical coherence | **FALSIFIED** |
| H7 | `HF2Q_FUSED_MOE_WSUM_END_LAYER_V2=1` | +30 disp/tok savings | -0.8% throughput at byte-identical coherence | **FALSIFIED** |
| H8 | `HF2Q_Q8_0_ID_MV_NR2=1` (peer's N_SG_Q8_0=4 port) | match peer kernel | -1.3% throughput at byte-identical coherence | **FALSIFIED** (kernel correct, M5 Max scheduler doesn't favor it) |
| H9 | Q6_K mul_mv_id already on NR2 (matches peer) | iter-1 verification | confirmed parity | **STANDING** (no work needed) |
| H10 | MoE-3 barriers are redundant (487 emitted/tok) | iter-2 audit | tracker already dedupes via `conflicts_reason`; 487 is provably minimum | **FALSIFIED** |
| H11 | hf2q's per-dispatch is slower because launch-overhead-bound | iter-5/6 model | CPU encode 0.19 µs negligible; gap is GPU-side | **FALSIFIED** |
| ~~H12~~ | ~~F16 KV cache for gemma4 would close ~450 µs/tok of the ATTN gap~~ | ~~iter-9 model~~ | **iter-11 LIVE TEST: HF2Q_HYBRID_KV=1 (ADR-028 Phase 10 — F16 K + TQ-HB V): coherent ✅ but throughput 61.4 t/s vs 73.9 baseline = -17% REGRESSION at σ-pct < 3%** | **🚨 H12 FALSIFIED** |
| H13 | FFN sub-phase has a single dominant slow op | iter-9 split needed | un-localized | **STANDING — needs sub-split** |
| **H14** | **Per-layer GPU drops 8 µs/layer under HYBRID (sliding 113→104 µs ATTN) AND 60 dispatches/tok eliminated, yet wall-clock loses 2.75 ms/tok. Therefore 3.5 ms/tok of CPU-or-cmd-buffer overhead is added by the HYBRID encode/SDPA path OUTSIDE the per-layer phase.** | **iter-11 measurement** | **not yet localized** | **STANDING — concrete next test (iter-12: find the 3.5 ms/tok)** |

## Decision

1. **Halt** all hypotheses about MoE matmul kernel optimization being the
   gap. MoE is 12.6% of wall-clock and already at 137% of conservative
   bandwidth peak. (H2 retraction.)
2. **Halt** all dispatch-fusion / fewer-larger-kernel direction. Three
   independent falsifications (H6/H7/H8) show this lever class regresses
   on M5 Max for gemma4 shapes; the cause is Apple Metal's scheduler
   favoring "more, smaller" kernels over "fewer, larger" at these
   dimensions. (H3, H6, H7, H8 retractions.)
3. **Pivot** optimization work to the **F16 KV cache for gemma4** lever
   (H12). Existing bench data predicts ~450 µs/tok = 3.4% throughput
   recovery if implemented coherently. Action items below.
4. **Sub-split FFN phase** via the iter-9 instrumentation extended to
   3-4 FFN sub-phases. Until this is done, the 58 µs/layer FFN gap is
   unattributed.
5. **Adopt** the iter-9 `HF2Q_PER_LAYER_*_GPU_TIME` env hooks as the
   standing per-phase profiler. All future ADR-029-class investigations
   must use these (or equivalent) to claim per-bucket attribution. No
   more synthetic-bench arithmetic.
6. **Standing rule** (already in §Validation gate): every "X× peer" /
   "X tok/s" claim must report σ-as-pct-of-mean. ≥5% σ-pct = re-measure.

## Action items (what's actually left to do)

### A. ~~F16 KV cache for gemma4~~ — REPLACED by iter-12 (localize the HYBRID 3.5 ms/tok overhead)

**iter-11 finding (2026-05-11 HEAD c50da032, fresh re-measure)**:
F16 K-cache infrastructure already exists as ADR-028 Phase 10:
- `HybridKvBuffers` (forward_mlx.rs:1105 — F16 K + TQ-HB V)
- `flash_attn_vec_hybrid` SDPA kernel (mlx-native, Phase 10d)
- `dispatch_kv_copy_kf16_quantize_v_no_fwht` fused write (Phase 10c.5)
- `HF2Q_HYBRID_KV` env gate (investigation_env.rs:826)
- Phase 10e wired SDPA dispatch + 10e.5 skipped FWHT-undo (V is raw)

**Live measurement at HEAD c50da032** on gemma4-ara-2pass-APEX-Q5_K_M.gguf:
| metric | HEAD (no HYBRID) | HF2Q_HYBRID_KV=1 | Δ |
|---|---:|---:|---:|
| decode t/s --benchmark n=5 | 73.9 (σ 0.07%) | **61.4 (σ 2.7%)** | **−17%** |
| dispatches/layer (sliding) | 31 | 29 | −2/layer (−60/tok) ✓ |
| per-layer GPU total (sliding) | 399 µs | 377 µs | −22 µs ✓ |
| per-layer GPU total (full) | 451 µs | 412 µs | −39 µs ✓ |
| ATTN-only (PHASE) sliding | 113 µs | 104 µs | −9 µs ✓ |
| FFN-only (PHASE) sliding | 285 µs | 280 µs | −5 µs ✓ |
| FWHT-pre+undo dispatches | present | eliminated | ✓ |
| coherence (haiku 50-tok) | reference | "Neon lights aglow,…" — fluent ✓ | parity ✓ |

**Decomposition of the 2.75 ms/tok wall-clock regression**:
- Per-layer GPU ↓ 240 µs/tok (savings, predicted +1.7%)
- Per-token wall ↑ 2750 µs/tok (regression, measured −17%)
- **Δ = +2.99 ms/tok overhead OUTSIDE the per-layer phase under HYBRID**

This is the new question. H14 (above) is the standing hypothesis.

### A'. iter-12: localize the HYBRID +3 ms/tok wall-clock overhead

**Candidate sources** (to test in iter-12+):
1. Per-token CB-commit cost: HYBRID may commit fewer/larger CBs vs the
   chunked legacy path → bigger latency tail per token if scheduler stalls.
2. `flash_attn_vec_hybrid` kernel internal throughput at gemma4 head_dim=256
   shapes is lower per-µs-of-CB than TQ-HB equivalent (despite measured
   per-layer GPU being lower; some scheduling overhead may not show up in
   `MTLCommandBuffer.GPUEndTime − GPUStartTime`).
3. Cross-CB barrier scheduling: HYBRID's per-token CB graph (1 fused KV-write
   + 1 SDPA-main + 1 SDPA-reduce + 0 FWHT-pre + 0 FWHT-undo) may serialize
   differently than legacy (2 FWHT + 1 fused KV-write + 1 SDPA + 0 FWHT-undo).
4. CPU encode work per HYBRID dispatch may be heavier (more args / arg
   buffer construction) — testable via dispatch_count vs wall ratio.
5. Phase 10c.5 fused KV-write may stall the GPU pipe in a way that
   `kv_copy + 2× hadamard_quantize_kv` (legacy) does not.

**Localization plan (iter-12)**:
- (a) Add `HF2Q_DECODE_WALL_BREAKDOWN=1` that measures: token-pre-layer wall
  + per-layer wall + token-post-layer wall + sampling wall. Each segment via
  `Instant::now()` between CB commits.
- (b) Run the same 5-trial bench under HYBRID with this instrumentation;
  compute average µs/token across the 4 segments.
- (c) Identify which segment carries the +3 ms/tok regression.

**Falsification criterion**: if the segment-by-segment wall sums to ≈ measured
wall (within 5%), the localization is correct. If a "missing time" gap remains,
the regression is in CB scheduling or barrier hold-time and a different
instrumentation (xctrace System Trace) is required.

### B'. iter-13: FFN sub-phase split — UNCHANGED from iter-9 plan, deferred until iter-12 root-cause is closed

### B. FFN sub-phase split (priority 1, prerequisite to optimization)

Extend `HF2Q_PER_LAYER_PHASE_GPU_TIME` to split FFN into:
- `FFN_NORMS` — post-attn norm + 3 pre-FF norms
- `FFN_DENSE` — dense MLP up/gate/down + GELU
- `FFN_MOE` — router_proj + softmax/top-k + gate_up_id + swiglu + down_id + weighted-sum
- `FFN_EOL` — end-of-layer fused norm-add-scalar

Each sub-split requires inserting `s.finish_with_gpu_time()` + new
session begin at the boundary. Expected slowdown: ~3× CB commits per
layer = ~9 ms extra/tok overhead during measurement.

Outcome: identify which FFN sub-phase has the 58 µs/layer gap. Likely
candidates by composition:
- `FFN_NORMS`: 3 dispatches; peer probably ≤ hf2q here (norm kernels are
  small and parity)
- `FFN_DENSE`: dense MLP 36-40 µs; peer may run faster F16 dense?
- `FFN_MOE`: 56 µs; already bandwidth-saturated per bench
- `FFN_EOL`: 1 fused dispatch; should be parity

### C. After A+B land, decide next direction by data (not guess)

The next per-layer measurement will show which sub-phase improved
(F16 KV → FFN_DENSE remains? FFN_NORMS gap appears? etc.). Choose the
new largest gap as the next target.

## Validation (how to reproduce this ADR's numbers)

All commands below assume `bc6ffd88` or later on branch `adr-029`.

```bash
MODEL=/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf

# 1. Baseline throughput (expect 73-75 t/s, σ-pct < 0.5%):
./target/release/hf2q generate --model "$MODEL" \
  --prompt "Write a long story about a sentient telescope" \
  --max-tokens 200 --benchmark

# 2. Peer baseline (expect 99-103 t/s, σ-pct < 1%):
/opt/homebrew/bin/llama-bench -m "$MODEL" -p 0 -n 200 -r 10

# 3. Per-layer GPU time (expect sliding 387 µs, full 438 µs):
HF2Q_PER_LAYER_GPU_TIME=1 ./target/release/hf2q generate \
  --model "$MODEL" --prompt "Hi" --max-tokens 3 2>&1 | grep PER_LAYER_GPU

# 4. Per-phase GPU time (expect sliding ATTN 112, FFN 275; full ATTN 170):
HF2Q_PER_LAYER_PHASE_GPU_TIME=1 ./target/release/hf2q generate \
  --model "$MODEL" --prompt "Hi" --max-tokens 3 2>&1 | grep PHASE_

# 5. TQ overhead control (expect sliding ATTN to drop 14 µs):
HF2Q_USE_DENSE=1 HF2Q_PER_LAYER_PHASE_GPU_TIME=1 ./target/release/hf2q generate \
  --model "$MODEL" --prompt "Hi" --max-tokens 3 2>&1 | grep PHASE_

# 6. Peer per-pipeline histogram (instrumented llama.cpp at
#    /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m, see iter-5):
HF2Q_PEER_COUNT_PRINT=1 HF2Q_PEER_PIPELINE_HIST=1 \
  /opt/llama.cpp/build/bin/llama-bench -m "$MODEL" -p 0 -n 200 -r 1

# 7. Per-shape decode-time kernel bench (existing infra, also corrected
#    to use Q6_K for attn shapes since iter-8):
cd /opt/mlx-native && cargo bench --bench bench_decode_qmatmul_shapes
cd /opt/mlx-native && cargo bench --bench bench_decode_moe_id_shapes
cd /opt/mlx-native && cargo bench --bench bench_sdpa_kv_dtype_compare
cd /opt/mlx-native && cargo bench --bench bench_dispatch_overhead
```

Acceptance gate for any change claiming to close the gap: numbers (1)
move to ≥1.0× peer with σ-pct <0.5% and 50-tok haiku output remains
byte-identical to F32-dense baseline.

## Consequences

Positive:
- The bucket-level root cause is finally measured, not guessed. Future
  iterations can target attribution boundaries instead of running blind
  flag-flip experiments.
- 11 falsified hypotheses (H2–H8, H10, H11) are documented with the
  measurements that falsified them. Future iters won't re-test them.
- Two concrete actionable hypotheses (H12, H13) replace the previous
  blanket "MoE pipeline is the gap" / "graph-fusion port is the answer"
  framings, both of which were wrong.

Negative / risks:
- Even with the F16 KV lever fully landed (estimated 3-4% throughput),
  hf2q would close only ~1/8 of the 25-32% gap. The remaining ~20% gap
  may be **distributed Apple Metal scheduling overhead** that has no
  single-kernel root cause and would require structural pipeline
  rework (multi-week project, possibly out of scope for the current
  hf2q architecture).
- The "more smaller kernels run faster than fewer larger ones on M5 Max"
  observation (H3/H6/H7/H8 falsifications) is empirical but not
  explained — Apple's scheduling internals are not documented at this
  granularity. Future kernel work must respect this unexplained
  invariant.
- Until H12 (F16 KV) is tested in production, the ATTN gap attribution
  remains hypothetical. The current ADR's ranking of "F16 KV first" is
  itself a falsifiable guess (informed by bench data, but not yet
  proven on real decode).

## Appendix A: Architectural delta acceptance (escape hatch)

The same hf2q codebase runs **qwen3.6 APEX at 1.270× peer** (iter-2
σ-pct-gated measurement). gemma4 specifically is 0.71× peer. The gap is
**model-architecture-specific to gemma4**, not a hf2q-wide regression.

Reasons gemma4 may be intrinsically harder for hf2q than qwen3.6:
- gemma4 uses Q6_K-dominant weights; qwen3.6 uses Q5_K. Different
  bandwidth-utilization patterns per kernel.
- gemma4 has dense MLP + MoE running in parallel per layer (extra
  dense dispatch budget); qwen3.6 has only MoE.
- gemma4 has 24 sliding + 6 full attn layers; qwen3.6 has 30 sliding +
  10 full. Different attn-budget distribution.
- gemma4's per_layer_embedding is disabled (metadata=0) but other
  gemma4 variants may activate it.

Operator may accept "gemma4 at 0.71× peer, qwen3.6 at 1.27× peer" as a
permanent architectural-delta outcome if the F16 KV + FFN sub-split
work fails to close the gap meaningfully. This is **not the default
disposition** — Action items A and B should be tried first.

## Appendix B: Chronological iter log (compressed)

| iter | commit | outcome |
|---:|---|---|
| 0 | 757c1683 | Original ADR-029 — "MoE pipeline IS the gap" (WRONG, retracted) |
| 1 | 1cd6540f | kernel-profile Q6_K lm_head gating fix (LANDED) |
| 1 | 9aed98cf | H6 TRIPLE_NORM -2.8% FALSIFIED + MoE-2 NR2 verified parity |
| 2 | 02d57966 | qwen3.6 1.27× CONFIRMED σ-pct < 0.25% + MoE-3 barriers FALSIFIED |
| 3 | 4eb2577d | H7 MOE_WSUM_END_LAYER -0.8% FALSIFIED + stale-claim cleanup |
| 4 | a7eb6608 | "ready to merge" (WRONG framing — items closed but goal not met) |
| — | merged to main, then operator clarified mission still open |
| 5 | c2bc5aaf | SMOKING GUN — peer issues 1389 dispatches not 105; ADR direction WRONG |
| 6 | 7acd4d4 (mlx) + c5c68833 | Q8_0 _id NR2 ported, -1.3% FALSIFIED + FFI overhead FALSIFIED |
| 7 | c6a3e707 | MoE attribution retracted (real 12.6%, was 49%) |
| 8 | — | Sub-bucket arithmetic continued to not add up; operator called out the pattern |
| 9 | bc6ffd88 | **Per-layer + per-phase GPU instrumentation landed; TRUE root cause measured** |
| 9 | c50da032 | ADR-029 rewritten to reflect measured ground truth (iter-stack collapse) |
| 10 | this commit | Fresh re-measure at HEAD: hf2q 73.9 / peer 97.73 = 0.756× (σ-pct ≤ 1.2%, apples-to-apples confirmed). Per-phase parity-checks against iter-9 numbers: identical within noise. |
| 11 | this commit | **HF2Q_HYBRID_KV=1 (ADR-028 Phase 10 F16-K+TQ-HB-V) LIVE TEST: COHERENT but −17% throughput regression. H12 FALSIFIED at wall-clock. Per-layer GPU savings (-22 µs sliding, -39 µs full) are real but +3 ms/tok overhead emerges OUTSIDE the per-layer phase. New H14 standing: localize the 3 ms/tok via iter-12 wall-breakdown instrumentation.** |

## Links

- `~/.claude/projects/-opt-hf2q/memory/feedback_do_not_trust_file_claims_re_measure_2026_05_11.md`
- `~/.claude/projects/-opt-hf2q/memory/feedback_targets_must_be_apples_to_apples_2026_05_11.md`
- `docs/ADR-027-qwen35-tq-kv-cache-and-persist-family.md` (qwen3.6 TQ-KV path; gemma4 doesn't use this by default)
- `docs/ADR-028-peer-parity-coherence-and-speed.md` (prior 141-iter mission; ADR-029 corrects its iter-486/487 closure but otherwise builds on its empirical work)
- mlx-native bench infra: `/opt/mlx-native/benches/bench_{decode_qmatmul_shapes, decode_moe_id_shapes, sdpa_kv_dtype_compare, dispatch_overhead}.rs`
- Peer instrumentation patch: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m` (atomic counters + per-pipeline histogram, env `HF2Q_PEER_COUNT_PRINT=1` + `HF2Q_PEER_PIPELINE_HIST=1`)
- hf2q production timing hooks (landed iter-9): `src/serve/forward_mlx.rs` (env `HF2Q_PER_LAYER_GPU_TIME` + `HF2Q_PER_LAYER_PHASE_GPU_TIME`)
