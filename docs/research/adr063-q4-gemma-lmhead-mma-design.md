# ADR-063 Q4-GEMMA-LMHEAD-MMA: few-row MMA for the Gemma 4 lm head (Class B)

Status: design for review, revision 1 (2026-11-05). The operator opened this
story on 2026-11-05 under Q4-GEMMA-DECODE design §13 decision 2(b). No code
backs this document. Numbers come from retained receipts (paths given) or are
marked `[INFERENCE]`.

Abbreviations: `$D` = `~/.local/state/hf2q/v0130/q4-gemma-decode`;
`$V1` = `$D/v1-718e811-20261105T174241Z` (M0 cells, mlx-native `718e811`);
`$M0P` = `$D/m0p-20261105T053654Z` (the peer's backend-op perf at `e86f5fd1`);
`$V2` = `$D/v2/runs-1733a874-20261105T174142Z` (decode-workload histograms).

## 1. Summary

- **What.** The Gemma 4 lm head (Q6_K, 2816 → 262,144, tied to the token
  embedding, 605.6 MB) runs at decode widths 1..8. A few-row matrix-multiply
  kernel replaces it: the weight is dequantized once per lane for all columns,
  and 8×8 simdgroup tiles accumulate in F32. That is the peer's arrangement.
- **Why.** Today's exact multi-column kernel costs about 530 µs more for every
  extra column. At width 4 the peer's kernel takes 0.55× of our time (§2). The
  exact alternative (Q4-GEMMA-DECODE D1) was 1.6× slower and was dropped.
- **Class.** The MMA reorders each logit's K reduction, the way the peer does,
  so it is **Q1 Class B against stock**. Its output stays within rounding of
  today's logits.
  - The Class B gates in §6 are the program's standing nothing-broke checks
    for a reordered route: a decision envelope and a paired quality campaign
    against stock.
  - They do not claim that the route moves hf2q away from the peer.
- **Invariants that stay Class A.**
  - **Batched ≡ serial at every width**, width 1 included (ADR-040 S2/S3): the
    MMA route serves width 1 too.
  - **Column and width invariance:** each logit's bits depend only on its own
    column and the weight, never on the other columns or on the width.
- **Switch.** `HF2Q_GEMMA_LMHEAD_MMA`, `0` by default. It is read once at
  Gemma load, and any value other than `0`/`1` is a startup error. If the story
  is not promoted, all its code is removed (clean cutover).

## 2. Evidence

Lm head, cold weights (the 605.6 MB weight cannot stay cached), median µs:

| Width | hf2q stock `kernel_mul_mv_q6_K_f32_mN` ($V1) | The peer, `MUL_MAT` ($M0P) | Peer / stock |
|---:|---:|---:|---:|
| 1 | 1,088 | 1,062 | 0.98 |
| 2 | 1,434 | 1,235 | 0.86 |
| 3 | 1,748 | 1,202 | 0.69 |
| 4 | 2,152 | 1,181 | 0.55 |
| 8 | 4,306 | 1,208 | 0.28 |

- **Bandwidth.** At width 1 stock streams at 556 GB/s against a 547 GB/s
  read+write blit roofline (`$V1/cells.jsonl`). The width-1 head is
  bandwidth-bound; wider heads are bound by per-column dequantize-and-FMA
  work.
- **Peer kernel.** For Q6_K at two or more rows (`rows_min` 2), the peer uses
  `kernel_mul_mv_mma_*` (`P:ggml/src/ggml-metal/kernels/mul_mv_mma.metal`,
  source sha in Q4-GEMMA-DECODE design "Source identities"):
  - each lane dequantizes 16 weights once for all source rows;
  - tiles are `simdgroup_float8x8` with F32 activations, F32 dequantized
    weights and F32 accumulation;
  - there is no half-precision staging of activations.
- **Workload ($V2).** 471 decode steps: 403 at width 4, 66 at width 2, one
  each at widths 1 and 3.
- **Saving [INFERENCE].** (2,152 − 1,181) µs × 403 + about 200 µs × 67 ≈
  0.40 s of decode GPU per run. It scales down with fewer target steps
  (serving speculation) and up with wider verify rows.
- **The exact alternative failed.** Q4-GEMMA-DECODE D1 for Q6_K was 1.57–1.62×
  slower than stock at width 4 (`$V1/m0-adoption.json`). Q1 Class A forces the
  per-(row, column) operand form, so the per-column dequantize cannot be
  shared.

## 3. Contract

### 3.1 Class B against stock (numerics)

- **Reference.** Today's head:
  - `kernel_mul_mv_q6_K_f32_nr2` at width 1 (also the serial path's baked
    record);
  - the exact `mN` tiles at widths 2..8;
  - followed by the softcap.
- **Route.** The MMA head followed by the same softcap. The logits differ from
  the reference only by reassociating each logit's 2816-term reduction and its
  dequantize arithmetic.
- **Verdict.** The Q1 decision envelope (Q1 §3.2), τ = 0.25 nat over every
  logit row of the accepted greedy144 inputs (Gemma `842768`), with no subsets.
  - Element-wise error (max-abs, RMS, cosine, ULP histogram) is recorded,
    never gated.
  - Any `RouteSensitive` row is localized (here, by construction, the head) and
    recorded, with the near-tie margins of the flipped rows.

### 3.2 Class A invariants (bitwise, must hold)

1. **Determinism.** Two runs of the same input give the same bits.
2. **Column independence and width invariance.**
   - For every width m ∈ 1..8 and every tiling the dispatcher picks, logit
     (c, r) is u32-equal to that of the width-1 MMA dispatch on column c alone.
   - A NaN or ±∞ in a neighbouring column does not change it.
   - Permuting columns permutes the outputs.
3. **Batched ≡ serial.** Under `HF2Q_GEMMA_LMHEAD_MMA=1` the serial decode head
   (width 1) and the batched head (width N) both use the MMA route. By item 2
   they are then bitwise equal per slot, so ADR-040 S2/S3 and the slot-aware
   parity tests keep passing unchanged.
   - The kill switch `0` keeps today's routes everywhere.
   - Prefill's first-token head keeps its current route in both arms.
4. **Views and bounds.** Non-zero-offset views, row tails (synthetic N, e.g.
   7, 262,151), and nothing written outside the output view.

If item 2 cannot hold for a candidate tiling, that tiling is not allowed. The
peer's kernel varies its tile count with width (`nt1/2/4`, `rt1/2`). The port
must pin the per-element reduction order regardless of tile occupancy, or use
one tile shape for all widths.

### 3.3 Never

Half-precision activation staging (unless the envelope and quality gates are
re-run under a new revision). Changing the softcap or sampling. Width-dependent
reduction order. Any change to the stock routes when the switch is `0`.

## 4. Design

### 4.1 mlx-native

- **Kernel.** `kernel_mul_mv_mma_q6_K_f32_<geometry>`, a port of the peer's
  Q6_K dequantizer and the generic few-row MMA body:
  - F32 `simdgroup_float8x8` accumulators;
  - one K-step per 64-weight chunk;
  - output columns padded to the tile width with zeroed activations.

  The geometry (SIMD groups per threadgroup, rows per tile) is searched in the
  spike and frozen in a static table. The kernel lives in its own `.metal` file
  with the crate's default math flags; it has no stock reference to match bit
  for bit.
- **Entry point.**
  `quantized_matmul_ggml_q6k_mma(encoder, registry, device, input, weight, output, params)`
  - It admits Q6_K, m ∈ 1..8 and K % 256 == 0.
  - It is explicit only: no routing-policy field and no auto-route change, so
    no other caller or family can reach it.
  - It is exported from `lib.rs`. This is an additive minor release.
- **Tests** (`tests/gemma_lmhead_mma.rs`, both metallib paths): §3.2 items 1,
  2 and 4 at N = 262,144 and synthetic tails, m = 1..8, seeded valid blocks
  including zero and subnormal half scales, and activations with ±0,
  subnormals and large values.
  - Error-record test: the MMA output against stock across widths, written to
    a receipt; it never fails on magnitude.
- **Bench.** A `lmhead_mma` family in `benches/bench_gemma_decode_w4.rs`: stock
  against MMA at m = 1..8 on the same cold-rotation harness as M0.

### 4.2 hf2q

- **Switch.** `HF2Q_GEMMA_LMHEAD_MMA` (`0` default, `1`, anything else a
  startup error). It is resolved once at Gemma load, recorded in the banner
  and in `/hf2q/v1/runtime` `active_controls.gemma_lmhead_mma`, and declared in
  `gemma4.json` (variable `GEMMA_LMHEAD_MMA`, serve env `LMHEAD_MMA`,
  launcher forwarding) with `test_q5_calibration.py` tests.
- **Route.** With `1`:
  - the batched head (`encode_lm_head_into` and `lm_head_batched`) and the
    serial decode head (the `forward_mlx_shared` Q6_K width-1 path, including
    its baked record) call the MMA entry point;
  - only when the head weight is Q6_K with K = 2816; any other type or shape
    keeps stock.
- **Independence.** It is independent of `HF2Q_GEMMA_DECODE_KERNELS` (F2).
  Each switch has its own A/B, and the combined candidate is judged in the Q4
  closing campaign.

## 5. Spike and M0 (model-free; root GPU lane)

Spike go/no-go, both required, recorded before any hf2q integration:

- **Exactness:** §3.2 items 1, 2 and 4 pass on both metallib paths.
- **Speed** (precompiled, median, cold rotation): MMA ≤ 0.70 × stock at width
  4, and **no width 1..8 more than 2% slower** than stock. Width 1 matters
  because the MMA must serve the serial head.

A miss on either stops the story and records the numbers; the code is
removed.

## 6. Gates after integration (in order; any failure stops)

1. **Class A, hf2q** (release build, real artifact, `--exact --ignored
   --test-threads=1`, switch `1`):
   - the slot-aware n1/n4/n4-batched-body/n8/eviction parity tests (batched ≡
     serial);
   - a new `gemma_lmhead_mma_batched_equals_serial` (four ragged lanes, ≥ 256
     steps, logits u32 per slot against the serial MMA head).
2. **Q1 envelope.** `q1_gemma_actual_cohort_endpoint_trace` (greedy144, T2
   cohort) under `0` and `1`, then `q1_decision_policy.py envelope`. Expect
   `WithinDecisionEnvelope`. A `RouteSensitive` verdict is recorded with the
   per-row D, flipped-row margins and the element-wise record, and it requires
   gate 3 to pass before any promotion claim.
3. **Paired quality** (Q1 §3.5), against stock hf2q:
   - 24 paired instances (6 instance labels × 4 agents), one run per instance
     per arm;
   - post-#704 harness, cargo-1.99 sandbox proof, production sampling, no
     reruns;
   - `q1_decision_policy.py paired` must return `Pass`.
4. **Speed and fairness, same binary:**
   - M1: `slot_aware_n4_batched_body_throughput_probe` with `HF2Q_GPU_BUSY=1`,
     the lm head in its own command buffer (`HF2Q_FUSE_LMHEAD=0`); lm-head GPU
     per step ≤ 0.70 × the kill switch at N = 4, ≤ 1.01 at N = 1.
   - M2: four-agent Q5 A/B, ABBA, ≥ 4 runs per arm; median makespan ratio
     ≤ 0.98 with block-paired 95% CI upper < 1.00. The 0.98 bar is [INFERENCE]:
     about 0.4 s of a 14.9 s makespan, so this lever alone cannot clear 0.96.
   - Worst-slot TTFT p95 and SSE-gap max ratios ≤ 1.025.
5. **Combined candidate.** Q4 closing campaign (Q4-GEMMA-DECODE design §8.4)
   with every adopted Gemma lever: Win against the peer's MT configuration,
   paired quality Pass against v0.1.28 and the peer, fairness, and a fresh Q5.

Promotion flips the default to `1` only after gates 1–4. If not promoted, the
default stays `0` and §11-style clean cutover removes the code.

## 7. Order and cost

1. This design is reviewed and approved.
2. mlx-native spike: kernel, tests, bench. CPU, then about 15 min of
   model-free lane.
3. Spike go/no-go (§5).
4. hf2q integration (CPU) and gate 1, about 20 min of model lane.
5. Gate 2, about 10 min of model lane.
6. Gate 3: 48 four-agent runs, about 2–3 h of model lane.
7. Gate 4 (M1, M2).
8. Gate 5 with the combined candidate.

## 8. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Width-invariant tiling costs part of the peer's gain | Spike misses 0.70 | Geometry search; one tile shape for all widths if needed; stop on a miss |
| Width-1 MMA slower than the bandwidth-bound mat-vec | Serial and width-1 regression | §5 bars width 1 at ≤ 1.02; stop on a miss |
| Logit reassociation flips near-ties | `RouteSensitive`, quality noise | Gate 2 records margins; gate 3 decides |
| Quality campaign underpowered (n = 24) | `Inconclusive` | Stated power (Q1 §3.5 table); no reruns |
| Interaction with serving speculation (verify widths up to 16) | Admission at widths > 8 | Widths > 8 keep stock until measured; a later revision may extend |

## 9. Decisions requested

1. Open the story under this design: switch name, the Class B gates of §6 and
   the §5 stop rule.
2. M2 bar for this lever alone: ≤ 0.98 (proposed), with the 0.96 Win share
   judged on the combined candidate.
3. Whether the serial head's baked Q6_K record may be bypassed under `1`.
   It must be, for batched ≡ serial.
