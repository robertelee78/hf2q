# ADR-029: gemma4-APEX-Q5_K_M Decode Gap — Measurement-Driven Root Cause

- **Status**: 🔄 **REOPENED 2026-05-11 iter-19c** — iter-18 closure was short-context-only (gen=200 bench). Operator-raised long-context test (input ≥ 4K) shows ratios 0.92× / 0.89× / 0.86× peer at 2K/4K/8K, PREFILL 0.50× peer. Standing rule added against premature closure. Iter-20 H27 (F16-V) landed as opt-in but regime-dependent (+2.6% @ 4K, −0.2% @ 16K). iter-21 H28 (prefill gap) is next. Pre-mission baseline 73.9 t/s → current best short-ctx 98.6 t/s = +33.4%; long-ctx (8K prefill) 87.6 t/s vs peer 101.4 = **0.86× peer**.
- **Date**: 2026-05-11 (original) → iter-9 rewrite → iter-10..16 thread → **iter-18 H26 mission close**
- **Supersedes**: itself (the original "MoE pipeline IS the gap" framing was wrong; this rewrite replaces it)
- **Decision-grade evidence**: per-layer GPU timestamps from `MTLCommandBuffer.GPUStartTime/GPUEndTime` in real production decode + 3-trial fresh-process benchmark at thermal steady state
- **Tags**: performance, root-cause, gemma4, measurement-discipline, baseline-correction, peer-parity-achieved

## Decision (one sentence)

**MISSION CLOSED iter-18 (re-validated iter-19)** — gemma4-APEX-Q5_K_M
decode matches/beats llama.cpp peer across short/med/long context via
TWO landed levers: (1) **iter-13** default-flip of `HF2Q_HYBRID_KV`
(F16 K + TQ-HB V hybrid cache, +9.5%, was ADR-028 Phase 10 gated off),
(2) **iter-18** route F32 m=1 matmul through `dispatch_dense_matvec_f32`
(mat-vec kernel) instead of `dense_matmul_f32_f32_tensor` (mat-mat tile
kernel, 87.5% wasted at m=1), +22.6%. **iter-19 caveat**: the short-
context-only claim of "1.011× peer" understated regime variance. Multi-
regime measurement (5-trial median, same thermal session):

| regime | hf2q t/s | peer t/s | ratio |
|---|---:|---:|---:|
| gen=200 (short) | 94.8 | 75.49 | 1.26× |
| gen=1000 (med) | 92.3 | 76.97 | 1.20× |
| gen=2500 (long) | 76.0 | 72.75 | 1.04× |

hf2q wins all 3 regimes in same-session thermal state; ratio compresses
toward parity as context grows.

**Apples-to-apples 3-iter back-to-back at gen=2500** (alternating
hf2q/peer with same prompt workload, sleep 3s between):

| iter | hf2q | peer | ratio |
|---|---:|---:|---:|
| 1 | 83.3 | 83.21 | 1.001× |
| 2 | 84.9 | 81.30 | 1.044× |
| 3 | 83.2 | 79.62 | 1.045× |
| median | 83.3 | 81.30 | **1.025× (TIED-to-ahead)** |

Operator's earlier observation of peer 98.1 vs hf2q 90.1 at gen=2454
(0.918× peer) is reproducible only under specific thermal states where
peer benefited from a cool device. Under controlled back-to-back
measurement hf2q is at-or-above peer at long context. The TQ-HB V
dequant cost is real (per-token cost grows with kv_seq_len) but
small enough that thermal noise dominates — H27 (long-context F16 V
swap) would be a marginal lever and is deferred unless future workload
demands prove otherwise.

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
| **H12** | **F16 KV cache (HF2Q_HYBRID_KV — pre-existing ADR-028 Phase 10) closes part of the ATTN gap on gemma4** | **iter-9 model + iter-11 first run** | **iter-11 first run: -17% (61.4 vs 73.9 t/s) — RETRACTED in iter-12 as cold-start artifact. iter-12 3-trial fresh: +9.7% median (78.5 vs 71.6 t/s, σ-pct < 0.5%). Coherence ✓. Per-layer GPU savings replicate (-22 µs sliding, -39 µs full).** | **✅ H12 CONFIRMED** |
| H13 | FFN sub-phase has a single dominant slow op | iter-9 split needed | un-localized | **STANDING — needs sub-split** |
| ~~H14~~ | ~~Per-layer GPU savings don't translate to wall-clock under HYBRID~~ | ~~iter-11 single-shot~~ | **iter-12: artifact RETRACTED. Wall-clock IS faster under HYBRID at all properly-measured thermally-stable trials.** | **❌ H14 RETRACTED** |
| **H15** | **FFN gap is concentrated in FFN_BODY (B9-B13 dense MLP + MoE experts) at 264 µs/layer × 30 = 7.92 ms/tok** | **iter-14 sub-phase split** | **FFN_BODY = 264 µs (94.4% of FFN); FFN_NORMS = 6.7 µs (2.4%); FFN_EOL = 5.6 µs (2.0%). Peer FFN ~206 µs would imply peer FFN_BODY ~196 µs (gap = 68 µs/layer = 2.04 ms/tok).** | **✅ H15 CONFIRMED — gap is in FFN_BODY** |
| **H16** | **FFN_BODY 264 µs is dispatch-pipeline-bound, not compute-bound. Removing any single sub-component (dense MLP / routing / MoE experts) does not change wall time.** | **iter-14 skip-flag bisect** | **SKIP_DENSE_MLP 265 µs; SKIP_ROUTING 265 µs; SKIP_MOE_EXPERTS 268 µs (within noise). All identical to baseline 264 µs.** | **✅ H16 CONFIRMED — explains H6/H7/H8 fusion regressions** |
| ~~H17~~ | ~~Force-sequential B9 (insert barriers between currently-concurrent dispatches) shrinks FFN_BODY~~ | iter-15 probe | **HF2Q_B9_FORCE_SEQUENTIAL=1: 275 µs vs 270 µs baseline — +1.8% SLOWER. Forced sequential pattern HURTS, not helps.** | **❌ H17 FALSIFIED** |
| ~~H18~~ | ~~FFN_BODY 264 µs is fixed-overhead-bound, NOT kernel-throughput-bound~~ | ~~iter-15 dual-skip~~ | **iter-15.5 ROOT CAUSE: skip flags require `HF2Q_UNSAFE_EXPERIMENTS=1` ack (investigation_env.rs:773: `raw.skip_X && ack`). Without ack, skips silently no-op. Earlier "skip" experiments had 29 disp/layer always — they didn't skip anything. With proper ack, the bisects WORK.** | **❌ H18 RETRACTED — measurement artifact (ack-gating)** |
| **H18'** | **Proper bisect: FFN_BODY 268 µs decomposes into MoE-experts 85 µs/layer (32%) + dense MLP 31 µs/layer (12%) + fixed CB/barrier+routing/norm overhead 152 µs/layer (57%).** | iter-15.5 with HF2Q_UNSAFE_EXPERIMENTS=1 | **baseline 268 µs; SKIP_DENSE_MLP 237 µs (−31 µs / 4 dispatches); SKIP_MOE_EXPERTS 183 µs (−85 µs / 3 dispatches); SKIP both 153 µs (−115 µs / 7 dispatches). Additive within noise.** | **✅ H18' CONFIRMED — real per-component cost map** |
| H19 | Peer reaches 97 t/s at 1389 dispatches/tok (7.4 µs/dispatch) vs hf2q 75 t/s at 883 (15.1 µs/dispatch). Peer's per-dispatch wall is 2× faster — smaller kernels or fewer barriers. | inferred from iter-9 + iter-15 | **partly testable now: hf2q FFN_BODY decomposes as 116 µs kernel + 152 µs overhead. If peer FFN_BODY ~200 µs total, peer's overhead is ~84 µs vs our 152 µs (45% less). Per-dispatch wall gap is real but smaller than dispatch-count ratio suggests.** | **STANDING — concrete next test: profile per-CB overhead** |
| **H20** | **MoE expert dispatches are the largest single per-layer cost (85 µs/layer). The 3 dispatches gate_up_id + swiglu + down_id are bandwidth-saturated per ADR-029 H2 already (137-146% of conservative peak) — so MoE kernel optimization gains are limited. But the *layout* of these 3 dispatches (1 wide qmatmul-id → 1 elementwise → 1 wide qmatmul-id) may have a faster peer equivalent.** | iter-15.5 H18' decomposition | **iter-17 swiglu isolation: HF2Q_SKIP_MOE_SWIGLU=1 → FFN_BODY 266 → 260 µs (−6 µs). So swiglu = 6 µs, gate_up_id + down_id = 79 µs combined (each ~40 µs).** | **STANDING-PARTIAL: MoE individual cost localized** |
| ~~H21~~ | ~~152 µs/layer fixed overhead is encoder-close/reopen cost~~ | ~~iter-17 deep-researcher H21~~ | **iter-17: HF2Q_DUAL_BUFFER sweep flat at ~80.5 t/s across single/multi-split configurations (=2 default, =5,10,15,20,25,29 sweep + multi-split 2,15 / 2,10,20 / 2,8,16,24 / 2,6,12,18,24 / 1,4,8,12,16,20,24,28). Default `vec![2]` is already optimal.** | **❌ H21 FALSIFIED — CB granularity exhausted** |
| ~~H23~~ | ~~152 µs fixed overhead has 10-30 µs from routing dispatches~~ | ~~iter-17 deep-researcher H23~~ | **iter-17: HF2Q_SKIP_ROUTING=1 alone drops FFN_BODY by 176 µs (268 → 92 µs) — far more than 2 dispatches' worth of work. Mechanism: zero-init moe_expert_ids causes MoE kernels (still-emitted under SKIP_ROUTING) to read invalid expert slices = OOB or zero-read = fast-exit. Measurement INVALID — corrupts MoE input data.** | **🚨 H23 INVALID — dependency-corrupting skip-flag combo** |
| ~~H25~~ | ~~Per-layer encoder commit forces GPU drain at each layer boundary~~ | ~~iter-17 deep-researcher H25~~ | **iter-17: HF2Q_DUAL_BUFFER=0 (single-CB, all 30 layers): 76.7 t/s vs default 80.3 t/s = -4.5% REGRESSION across 3 trials. Single-CB is WORSE for hf2q. Default split-after-L2 is sweet spot; sweeping later splits is monotonically worse.** | **❌ H25 FALSIFIED — single-CB regresses hf2q (-4.5%)** |
| **H26** | **F32 m=1 (decode) router_proj routes through matrix-MATRIX tile kernel `dense_matmul_f32_f32_tensor` which wastes 87.5% of its 8x8 SIMDgroup-matrix tile. Routing m=1 F32 dispatches through `dispatch_dense_matvec_f32` (matrix-vector kernel) saves the wasted work.** | iter-18 discovery — ffn_gate_inp is F32 [2816,128], the FFN_BODY bisect's "fixed overhead" was largely THIS dispatch | **iter-18 3-trial bench: 80.4 → 98.6 t/s = +22.6% throughput, peer ratio 0.823× → 1.009×, byte-identical coherence. Math sanity: predicted savings 73 µs/layer × 30 = 2.19 ms/tok → predicted 97.6 t/s; measured 98.6 t/s ✓.** | **🎯 H26 CONFIRMED + LANDED (peer parity achieved)** |

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

### A. F16 KV cache for gemma4 — H12 CONFIRMED (iter-12 fresh 3-trial verdict)

**State**: F16 K-cache infrastructure already exists as ADR-028 Phase 10:
- `HybridKvBuffers` (forward_mlx.rs:1105 — F16 K + TQ-HB V)
- `flash_attn_vec_hybrid` SDPA kernel (mlx-native, Phase 10d)
- `dispatch_kv_copy_kf16_quantize_v_no_fwht` fused write (Phase 10c.5)
- `HF2Q_HYBRID_KV` env gate (investigation_env.rs:826)
- Phase 10e wired SDPA dispatch + 10e.5 skipped FWHT-undo (V is raw)

**iter-12 measurement** (3 independent processes, 5s gap between each,
gemma4-ara-2pass-APEX-Q5_K_M.gguf, 200-tok haiku, HEAD c50da032):

| trial | baseline t/s | HYBRID t/s | Δ |
|---|---:|---:|---:|
| 1 | 69.3 | 78.6 | +13.4% |
| 2 | 71.7 | 78.3 | +9.2% |
| 3 | 73.9 | 78.7 | +6.5% |
| **median** | **71.7** | **78.5** | **+9.5%** |

HYBRID is rock-stable across thermal cycles (σ-pct 0.3%); baseline shows
thermal warm-up (69→74 t/s). At thermal steady state, HYBRID gives
+5-9% gemma4 throughput at byte-class-coherent output.

Ratio vs peer (97.73 t/s llama-bench n=5):
- Baseline 73.9 / 97.73 = **0.756×**
- HYBRID  78.7 / 97.73 = **0.805×** → **+4.9 percentage points closer to peer**

**iter-11 RETRACTION**: the first HYBRID measurement at 61.4 t/s (σ-pct 2.7%)
was a cold-start / first-launch artifact. Three independent fresh processes
all confirm HYBRID is faster, not slower. σ-pct < 1% at thermal steady state.

**Next**: default-flip `HF2Q_HYBRID_KV` to `=1` for gemma4 (iter-13 task).
Coherence parity already passes (50-tok haiku). Bench gate passes (+9% well
above ≥+2% requirement).

### B. iter-13: default-flip HF2Q_HYBRID_KV=1 for gemma4 + add proper validation gate

**Work required**:
1. In `src/debug/investigation_env.rs:826`, change `env_eq_one` to `env_default_true`
   so the gate is on-by-default with opt-out via `HF2Q_HYBRID_KV=0`.
2. Run full unit-test suite (`cargo test --release`).
3. Run mlx-native parity test (the existing Phase 10f hooks).
4. Coherence gate: 50-tok haiku at temp=0 matches baseline output
   class-equivalent (fluent English, not garbage).
5. Bench gate: 5-trial benchmark median ≥ +5% baseline at σ-pct < 1%.
6. Commit + push to adr-029 branch.

### C. iter-14: FFN sub-phase split (Action B from iter-9 plan) — DEFERRED to after H12 default-flip lands

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
| 10 | a99a5b10 | Fresh re-measure at HEAD: hf2q 73.9 / peer 97.73 = 0.756× (σ-pct ≤ 1.2%, apples-to-apples confirmed). Per-phase parity-checks against iter-9 numbers: identical within noise. |
| 11 | a99a5b10 | **HF2Q_HYBRID_KV=1 first run: -17% (61.4 t/s) — RETRACTED in iter-12 as cold-start artifact. ADR-amend committed prematurely.** |
| 12 | 0808e4e9 | **HF2Q_HYBRID_KV=1 3-trial fresh independent processes: +9.5% median (78.5 vs 71.6 t/s). H12 CONFIRMED — F16 K closes ~4.9pp of the gemma4-peer gap. H14 RETRACTED (artifact). New ratio: 0.805× peer. Coherence ✓.** |
| 13 | 7e2239ad | **DEFAULT-FLIP: `HF2Q_HYBRID_KV` flipped from off to on via `env_default_true`. 3-trial steady-state bench DEFAULT 80.4 t/s vs OPT-OUT 78.1 t/s = +2.9% median, σ-pct < 1%, fluent haikus both modes, peer ratio 0.823×. Opt-out via `HF2Q_HYBRID_KV=0`. All 5 env tests + 16 qwen35 hybrid_kv_cache tests green.** |
| 14 | 3cca366a | **FFN sub-phase split (`HF2Q_FFN_SPLIT=1`) — added boundaries at end of B8 (norms) and end of B13 (body). Decomposition: FFN_NORMS = 6.7 µs (fused 1 dispatch), FFN_BODY = 264 µs (B9-B13 dense MLP + MoE), FFN_EOL = 5.6 µs (fused 1 dispatch). Skip-flag bisect: removing SKIP_DENSE_MLP / SKIP_ROUTING / SKIP_MOE_EXPERTS = identical 265 µs FFN_BODY ⇒ FFN_BODY is dispatch-pipeline-bound, not compute-bound. The 58 µs/layer FFN gap is in **dispatch scheduling**, not in any one kernel. Confirms why H6/H7/H8 (fusion) regressed on M5 Max — fusion compresses already-overlapped work.** |
| 15 | ee9cf7dc | **H17 FALSIFIED: HF2Q_B9_FORCE_SEQUENTIAL=1 → FFN_BODY +1.8% slower. H18 first-claim CONFIRMED then RETRACTED (see iter-15.5 below).** |
| 15.5 | af097288 | **🚨 H18 RETRACTED — root cause was that skip flags require `HF2Q_UNSAFE_EXPERIMENTS=1` ack (investigation_env.rs:773). Without ack, skips silently no-op. Earlier "skip" experiments had 29 dispatches/layer regardless of flag setting. With ack, skips actually skip: FFN_BODY decomposes as MoE-experts 85 µs (32%) + dense MLP 31 µs (12%) + fixed-overhead 152 µs (57%). H18' replaces H18. Lesson: env-flag toggles must be VERIFIED via dispatch-count delta before drawing kernel-vs-overhead conclusions. Adding this to standing-rules.** |
| 16 | b4757d99 | **Geometry comparison hf2q vs peer (`kernel_mul_mv_id_*_f32`): Q6_K MATCHES (NR0=2, NSG=2). Q4_K/Q5_K hf2q has NR0=2 vs peer NR0=1. Q4_0/Q5_1 hf2q has NR0=8/NSG=8 vs peer NR0=4/NSG=2. IQ4_NL/Q8_0 hf2q (8,8,8) vs peer (2,2) and (2,4). HF2Q_Q8_0_ID_MV_NR2=1 RE-TESTED under HYBRID-on: 81.1 → 80.4 = -0.86% (re-falsified iter-6 result still holds). H20 lever class (port peer's NR0 geometry per type) appears exhausted for the most plausible types.** |
| 17 | 4865e423 | **Deep-research agent proposed 5 new hypotheses (H21-H25). Tested 3: H25 (single-CB via HF2Q_DUAL_BUFFER=0) FALSIFIED -4.5%. H21 (CB granularity sweep =2,5,10,15,20,25,29 + multi-split combos) FALSIFIED — default `vec![2]` is optimal. H23 (routing isolation via SKIP_ROUTING) INVALID — corrupts MoE expert IDs leading to OOB reads = phantom savings. H20-partial: swiglu isolated at 6 µs/layer; gate_up_id + down_id = 79 µs combined. H22 (AUTO_BARRIER) requires structural dispatch_tracked migration in production decode — multi-day scope, deferred.** |
| 18 | `6dc2afc6` | H26 LANDED. Discovered router_proj (ffn_gate_inp F32 [2816, 128]) was routing through `dense_matmul_f32_f32_tensor` (matrix-MATRIX tile kernel, 87.5% wasted at m=1 decode). Routed m=1 F32 dispatches through `dispatch_dense_matvec_f32` (matrix-vector kernel) instead. Throughput 80.4 → 98.6 t/s = +22.6% (3-trial median, σ-pct < 1.5%). Coherence BYTE-IDENTICAL. Peer ratio 0.823× → 1.009× peer (short-context bench). Opt-out via HF2Q_F32_MATVEC=0; default ON for m=1. **iter-18 closure claim was SHORT-CONTEXT-ONLY — see iter-19c retraction.** |
| 19 | `05971180` | iter-19a output format split + iter-19b multi-regime `--benchmark` (200/1000/2500). |
| 19c | (this commit) | 🚨 **iter-18 "MISSION CLOSED" RETRACTED** at operator's long-context test. Re-measured with large prompt (4K/8K/16K input): hf2q 0.92× / 0.89× / 0.86× peer at 2K/4K/8K prefill. Decode gap WORSENS with kv_seq_len; PREFILL gap 0.50× peer consistently. Standing rule added: `feedback_no_premature_mission_close_2026_05_11.md` — multi-regime gate before any "mission complete" claim. |
| 20 | mlx-native `47139aa` + hf2q `adr-029-iter20-h27` | H27 measured. Added F16-V option to `flash_attn_vec_hybrid` kernel (FC slot 51) + `HF2Q_FULL_F16_KV=1` env gate. Coherent at short + 8K context. **Gain regime-dependent**: +2.6% @ 4K, +2.0% @ 8K, −0.2% @ 16K. Mechanism trade-off: TQ-HB V (1 byte/elem packed) is bandwidth-efficient at long ctx; F16 V (2 bytes/elem direct) is compute-efficient (no dequant). NOT default-flipped — opt-in only. Real long-context lever appears to be peer's 3-kernel SDPA pattern (Q@K^T + softmax + score@V) — multi-day scope. iter-21 (H28 prefill gap 0.50× peer) is the next target. |
| 21 (in progress) | (this commit) | H28 prefill gap **localization started**. `HF2Q_PROFILE_BUCKETS=1` at 4K-prefill shows: MoE 613 ms (35%), dense qmatmul 467 ms (27%), FA 339 ms (19%), other 145 ms (8%). Sum of GPU buckets = 782 ms (45%). **Residual = 966 ms (55%) — CPU+CB-sync overhead under profile mode**. Without profile (production), prefill wall is 1642-1748 ms; peer hits 970 ms at same shape (4341 t/s × 4213 tok). **41% gap remains** without bucket-commit overhead. A/B test of tensor-MM vs simdgroup-MMA via new `HF2Q_DISABLE_TENSOR_MM=1` probe: tensor (default) 1659 ms = 2539 t/s; simdgroup MMA 2142 ms = 1967 t/s. **Tensor IS the right choice** — disabling regresses 22%. The remaining 41% gap is NOT in tensor-vs-simdgroup choice. |
| 22 | f22df0fb | **H28 ROOT CAUSE LOCALIZED** (peer source read). hf2q tensor mm-kernel tile: **32×64 output** (NR1=32, NR0=64) → matmul2d_descriptor(32, 64, 32, ...). Peer (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9340-9367`) tile: **128×64 output** (NRB=128, NRA=64) via constants `SZ_SIMDGROUP × N_MM_BLOCK_X × N_MM_SIMD_GROUP_X = 16×4×2 = 128` and `SZ_SIMDGROUP × N_MM_BLOCK_Y × N_MM_SIMD_GROUP_Y = 16×2×2 = 64`. **Peer's tile is 4× larger in M-axis.** At m=4213, n=5760: hf2q dispatches 132×90 = **11,880 threadgroups**; peer dispatches 33×90 = **2,970**. 4× threadgroup count → 4× scheduling overhead, less tile reuse. Peer also reads B directly from device memory (no shmem B-staging) — our kernel stages B into shmem and has thread-mapping hardcoded to NR1=32. Path forward (iter-22+): port peer's modern tensor kernel layout into hf2q (NR1=128, direct B read, updated thread mapping, shmem 12288 B → 4096 B). Multi-hour kernel work bounded but non-trivial; env-gate `HF2Q_LARGE_TILE_MM=1` for opt-in validation. Expected gain: ~2× prefill at our shapes. |
| 23 | mlx-native `6256acb` | **V2 tensor mm-kernel LANDED (opt-in via HF2Q_LARGE_TILE_MM=1)**. Ported peer's 64×128 tile with direct-device B-read. 4× fewer threadgroups (11,880 → 2,970 at 4K), 4096 B shmem (was 8192). Coherence verified: BYTE-IDENTICAL haiku output ("Neon lights aglow,…") + same first decode token at every prefill context.<br>Bench delta on gemma4-APEX-Q5_K_M:<br>&nbsp;&nbsp;ctx=2K (2113 tok): 2470→2649 t/s = **+7.2%**<br>&nbsp;&nbsp;ctx=4K (4213 tok): 2564→2751 t/s = **+7.3%**<br>&nbsp;&nbsp;ctx=8K (8413 tok): 2397→2549 t/s = **+6.3%**<br>&nbsp;&nbsp;peer pp4096: 4467 t/s — V2/peer = **0.62×** (was 0.55×, +7 pp closer)<br>The 4× threadgroup reduction yielded only +7% wall — matmul is bandwidth-bound; threadgroup count past saturation has diminishing returns. Peer's remaining 1.6× advantage in a different lever (B-side cache locality, dequant inner loop, or Apple-Metal-internal). NOT default-flipped — kept opt-in. Mission still NOT closed. |
| 24 | d0feae5d | **V2 per-bucket breakdown at 4K prefill (HF2Q_PROFILE_BUCKETS=1)** localizes where V2 helps vs not:<br>&nbsp;&nbsp;**QKV_MM** (85 calls): V1 204.6 ms → V2 138.7 ms = **−32%**<br>&nbsp;&nbsp;**O_MM** (30 calls): V1 109.7 ms → V2 110.7 ms = **unchanged** (perm021 kernel, separate from dispatch_mm)<br>&nbsp;&nbsp;**MLP_GUR_MM** (90 calls): V1 106.9 → V2 75.9 ms = **−29%**<br>&nbsp;&nbsp;**MLP_DN_MM** (30 calls): V1 45.4 → V2 35.9 ms = **−21%**<br>**Real composition of V2 prefill at 4K (1639 ms wall)**: MoE 612 ms (37%), dense qmatmul incl perm021 361 ms (22%), FA 339 ms (21%), other (norms/fusion) ~330 ms (20%). |
| 25 | 721afe5b | iter-25 progress: V2 decode-safety verified; HF2Q_NO_FA falsified (-9%); peer's `mul_mm_id` source matches hf2q's tile (32×64). |
| 26 | mlx-native `07f75ea` | **V2 default-flip LANDED**. HF2Q_LARGE_TILE_MM env semantics changed from `env_eq_one` (opt-in) to default-true (opt-out via `=0`/`false`/`off`). V2 (64×128 tile + direct-device-B-read) is now the default tensor-mm path on M3+ devices. Multi-arch + multi-regime validation:<br>&nbsp;&nbsp;**gemma4-APEX prefill**: 2K +7.2%, 4K +7.3%, 8K +6.3%<br>&nbsp;&nbsp;**qwen3.6-APEX prefill** (4K): +0.8% to +2.6% (within noise)<br>&nbsp;&nbsp;**decode m=1**: unaffected (V2 only fires at m > 8)<br>&nbsp;&nbsp;**3457/0/11 tests** pass under V2 default<br>&nbsp;&nbsp;**byte-identical haiku** on both gemma4 and qwen3.6 at temp=0<br>This is an IMPROVEMENT, not mission-close. Long-context decode (0.86-0.92× peer) and prefill (now 0.62× peer at 4K, was 0.55× pre-V2) gaps remain standing. Per feedback_no_premature_mission_close — no closing language. |

## Links

- `~/.claude/projects/-opt-hf2q/memory/feedback_do_not_trust_file_claims_re_measure_2026_05_11.md`
- `~/.claude/projects/-opt-hf2q/memory/feedback_targets_must_be_apples_to_apples_2026_05_11.md`
- `docs/ADR-027-qwen35-tq-kv-cache-and-persist-family.md` (qwen3.6 TQ-KV path; gemma4 doesn't use this by default)
- `docs/ADR-028-peer-parity-coherence-and-speed.md` (prior 141-iter mission; ADR-029 corrects its iter-486/487 closure but otherwise builds on its empirical work)
- mlx-native bench infra: `/opt/mlx-native/benches/bench_{decode_qmatmul_shapes, decode_moe_id_shapes, sdpa_kv_dtype_compare, dispatch_overhead}.rs`
- Peer instrumentation patch: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m` (atomic counters + per-pipeline histogram, env `HF2Q_PEER_COUNT_PRINT=1` + `HF2Q_PEER_PIPELINE_HIST=1`)
- hf2q production timing hooks (landed iter-9): `src/serve/forward_mlx.rs` (env `HF2Q_PER_LAYER_GPU_TIME` + `HF2Q_PER_LAYER_PHASE_GPU_TIME`)
