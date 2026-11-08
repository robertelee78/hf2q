# ADR-063 Q4-GEMMA-LMHEAD-MMA: few-row MMA for the Gemma 4 lm head (Class B)

Status: **stopped at the spike (2026-11-05, §14)**; approved (operator
2026-11-05; review ACCEPT), revision 2 (2026-11-05). The operator opened this story on 2026-11-05 under
Q4-GEMMA-DECODE design §13 decision 2(b). Revision 2 applies ReviewLmHead's
REVISE findings 1–12, the root's rulings on decisions 2–4 (§12) and the
ACCEPT review's required edit and P3s (§13). No code backs this document.
Numbers come from retained receipts (paths given) or are marked `[INFERENCE]`.

Abbreviations:
- `$D` = `~/.local/state/hf2q/v0130/q4-gemma-decode`;
- `$V1` = `$D/v1-718e811-20261105T174241Z` (M0 cells, mlx-native `718e811`);
- `$M0P` = `$D/m0p-20261105T053654Z` (the peer's backend-op perf at `e86f5fd1`);
- `$V2` = `$D/v2/runs-1733a874-20261105T174142Z` (decode-workload histograms and two four-agent makespans);
- `P:` = the peer source at `e86f5fd1`.

## 1. Summary

- **What.** The Gemma 4 lm head is Q6_K, 2816 → 262,144, tied to the token
  embedding, 605.6 MB. At the serving decode heads it is replaced by a
  few-row matrix-multiply kernel:
  - the weight is dequantized once per lane for all columns;
  - 8×8 simdgroup tiles accumulate in F32;
  - this is the peer's arrangement.

  Prefill heads keep today's route.
- **Why.** The exact multi-column kernel costs about 530 µs more per extra
  column. At width 4 the peer's MMA takes 0.55× of our time (§2). The exact
  alternative (Q4-GEMMA-DECODE D1) was 1.6× slower and was dropped.
- **Class.** The MMA reorders each logit's K reduction, the way the peer does,
  so it is **Q1 Class B against stock**. That its output stays within rounding
  of today's logits is `[INFERENCE]` until the spike's error record (§5)
  exists.
  - The Class B gates (§6) are the program's standing checks that nothing broke
    for a reordered route.
  - They do not claim that the route moves hf2q away from the peer.
- **Class A invariants that remain:**
  - Batched ≡ serial at every width, including width 1 and widths above 8
    (ADR-040 S2/S3).
  - Column and width invariance.
  - The kill-switch arm (`0`) is bitwise today's serving.
- **Switch.** `HF2Q_GEMMA_LMHEAD_MMA`:
  - `0` is the default; `1` enables the route; any other value is a startup
    error. It is read once at Gemma load.
  - If the story is not promoted, its code is removed from both repositories
    (§11).

## 2. Evidence

Lm head, cold weights (the 605.6 MB weight cannot stay cached), median µs:

| Width | hf2q stock (`$V1`) | The peer (`$M0P`) | Peer kernel | Peer / stock |
|---:|---:|---:|---|---:|
| 1 | 1,088 | 1,062 | **mat-vec** `kernel_mul_mv_q6_K_f32` | 0.98 |
| 2 | 1,434 | 1,235 | MMA `nt4_rt1_nsg=8` | 0.86 |
| 3 | 1,748 | 1,202 | MMA | 0.69 |
| 4 | 2,152 | 1,181 | MMA | 0.55 |
| 8 | 4,306 | 1,208 | MMA | 0.28 |

Notes:
- **Stock bandwidth.** At width 1, stock streams the weight at 556 GB/s
  against a 547 GB/s read+write blit roofline (`$V1/cells.jsonl`).
- **The peer uses MMA only from 2 rows.** The peer sets Q6_K `rows_min` = 2
  (`P:ggml/src/ggml-metal/ggml-metal-common.cpp:120`, "the fewest src1 rows at
  which the few-row MMA kernels beat the mat-vec kernels"). Its width-1 row is
  therefore a mat-vec number, not an MMA number.
- **MMA cost is flat in width.** At n = 2..8 the MMA takes 1,181–1,235 µs, which
  is 1.085–1.135× stock width 1 (about 490–513 GB/s).
- **Prediction for width 1** `[INFERENCE]`: a port at the peer's efficiency
  runs width 1 at about 1,180 µs, roughly +90 µs over stock. §5 states the bar
  this must meet.
- **Peer numerics** (`P:…/kernels/mul_mv_mma.metal:504–606`, generic few-row
  body):
  - `simdgroup_float8x8` accumulators;
  - F32 activations and F32 dequantized weights;
  - no half-precision staging.
- **Workload** (`$V2`):
  - 471 decode steps: 403 at width 4, 66 at width 2, one each at widths 1
    and 3 (`{"1":1,"2":66,"3":1,"4":403}`);
  - the two four-agent makespans are 13.92 s and 13.95 s
    (`client/summary.json` `own_output_four_task_makespan_seconds`).
- **Saving** `[INFERENCE]` at the peer's times: (2,152 − 1,181) × 403 +
  (1,434 − 1,235) × 66 ≈ 0.405 s of decode GPU per run, about 2.9% of the
  13.9 s makespan. Fewer target steps (serving speculation) shrink it; wider
  verify rows raise it.
- **The exact alternative failed.** Q4-GEMMA-DECODE D1 for Q6_K was 1.57–1.62×
  slower than stock at width 4 (`$V1/m0-adoption.json`).

## 3. Contract

### 3.1 Class B against stock (numerics)

- **Reference:** today's head, then the softcap:
  - at width 1, `kernel_mul_mv_q6_K_f32_nr2` (the serial path's baked record);
  - at widths 2..8, the exact `mN` tiles;
  - at widths above 8, per-row width-1 calls.
- **Route:** the MMA head, then the same softcap. The logits differ from the
  reference only by reassociating each logit's 2816-term reduction and its
  dequantize arithmetic.
- **Verdict:** the Q1 decision envelope (Q1 §3.2), τ = 0.25 nat, on every
  logit row of the accepted greedy144 inputs (Gemma `842768`), with no subsets.
  - Element-wise error (max-abs, RMS, cosine, ULP histogram) is recorded and
    not gated at the route level.
  - **A `RouteSensitive` verdict stops the story.** The T2 cohort's decode
    feeds frozen tokens, the body is unchanged, and the head is the last op.
    So D measures only the head's F32 reassociation of a 2816-term dot
    product. Q1 §3.2 constructs D so that a near-tie flip has D ≤ τ, and the
    largest benign whole-model route D on record is 0.093 nat. A D > 0.25 here
    is therefore a kernel defect, to be localized, not near-tie noise.
  - **Coverage.** Of the 576 T2 rows, the 512 prefill rows go through the
    unchanged prefill head. Only the 64 decode rows (positions 128–143 × 4
    lanes, width 4) exercise the MMA. The kernel-level error record (§5)
    covers widths 1..16.

### 3.2 Class A invariants (bitwise, must hold)

1. **Determinism.** Repeated runs of the same input give the same bits.
2. **Column independence and width invariance.**
   - For every width m ∈ 1..16 and every tiling the dispatcher picks, logit
     (c, r) is u32-equal to the width-1 MMA dispatch on column c alone.
   - A NaN or ±∞ in a neighbouring column does not change it.
   - Permuting columns permutes the outputs.
   - Each element's reduction order is set by the K split across SIMD groups
     (NSG) and by which K chunks each SIMD group takes. Both are constants
     across widths. The column-tile count (the peer's `nt`) and row grouping
     (`rt`) may vary with width only if they leave that order unchanged.
   - Widths above 8 run in column groups of at most 8 (§4.1).
3. **Batched ≡ serial per slot** under `HF2Q_GEMMA_LMHEAD_MMA=1`, at every
   width. The serial and batched decode heads both use the MMA (§4.2), and modes
   that would compare any other head with them are refused, so ADR-040 S2/S3 and
   the slot-aware parity tests keep passing.
4. **Views and bounds.** Non-zero-offset views, row tails (synthetic N, for
   example 7 and 262,151), and no writes outside the output view.
5. **Kill-switch identity.** With `0`, serving is bitwise today's
   (`ExactTranscripts` against the pre-integration pin, §6 gate 1).

### 3.3 Never

- Half-precision activation staging, unless a later revision re-runs the
  envelope and quality gates under it.
- Changing the softcap or sampling.
- A width-dependent reduction order.
- Any change to shared dispatchers (`dispatch_qmatmul`,
  `dispatch_dense_rowident`) or to non-head Q6_K calls.

## 4. Design

### 4.1 mlx-native

- **Kernel.** `kernel_mul_mv_mma_q6_K_f32_<geometry>` ports the peer's Q6_K
  dequantizer and its generic few-row MMA body.
  - F32 `simdgroup_float8x8` accumulators, one K step per 64-weight chunk.
  - Output columns are padded to the tile width with zeroed activations.
  - NSG and the K-chunk assignment are fixed per shape, not per width.
  - Geometry is searched in the spike and frozen in a static table.
  - It lives in its own `.metal` file with the crate's default math flags.
    It has no stock kernel to match bit for bit; its contract is §3.2.
- **Entry point:**
  `quantized_matmul_ggml_q6k_mma(encoder, registry, device, input, weight, output, params)`.
  - Admits Q6_K, m ∈ 1..16 and K % 256 == 0. Widths above 8 are dispatched
    as column groups of at most 8, bitwise equal by §3.2 item 2.
  - Explicit only: no routing-policy field and no auto-route change, so no
    other caller or model family can reach it. Additive minor release.
- **Tests** (`tests/gemma_lmhead_mma.rs`, both metallib paths, N = 262,144 and
  synthetic tails, seeded valid blocks including zero and subnormal half
  scales, activations with ±0, subnormals and large values):
  1. **Functional correctness** against a CPU F64 dequantize-and-dot reference:
     |ŷ − y| ≤ 2·γ_K·Σ|w_i·x_i| + 2⁻²⁴·|y|, with γ_K = K·u / (1 − K·u),
     u = 2⁻²⁴, K = 2816, for every output at m = 1..16. The bound follows the
     pattern of the existing reordering-kernel tests
     (`adr_022_phase1_mv_ext_gpu_parity.rs`,
     `adr_034_task93_fused_gate_up_silu_q6_K_parity.rs`). It catches scale,
     sign and lane-map defects, which the invariance tests alone cannot.
  2. §3.2 items 1, 2 and 4 at m = 1..16.
  3. **Error record:** the MMA output against the stock head at m = 1..16,
     written to a receipt (max-abs, RMS, ULP histogram). It never fails on
     magnitude.
- **Bench.** A `lmhead_mma` family in `benches/bench_gemma_decode_w4.rs`: stock
  against MMA at m = 1..16, on the M0 cold-rotation harness.

### 4.2 hf2q

- **Switch.** `HF2Q_GEMMA_LMHEAD_MMA`: `0` by default, `1` to enable, anything
  else a startup error. It is resolved once at Gemma load and recorded:
  - in the banner;
  - in `/hf2q/v1/runtime` as `active_controls.gemma_lmhead_mma`;
  - in `gemma4.json` (variable `GEMMA_LMHEAD_MMA`, serve env `LMHEAD_MMA`, with
    launcher forwarding) and in `test_q5_calibration.py`.
- **Routing at the head call sites only.** The shared dispatchers stay
  byte-unchanged; with `0` every site below is exactly today's code.
  - **Predicate:** the weight is `resolved_lm_head()` (identity, not shape),
    with type Q6_K, K = 2816 and N = vocab.
  - **Under `1`, MMA at the serving decode heads only:**
    - the serial decode head (`forward_gpu.rs:994`);
    - the batched heads (`lm_head_batched`, `encode_lm_head_into`).
  - **Refused at startup under `1`:** every mode that would compare another
    head with these rows. That covers the DFlash, n-gram and EAGLE-3 tree-verify
    modes (`forward_gpu.rs:2216`, `:2700`, `io_heads.rs:141`, `:319`) and the
    kernel-profile modes. Extending the route to them needs a later revision.
  - **Prefill heads keep stock in both arms** (`forward_prefill.rs:3029`,
    `forward_prefill_batched.rs:5688`): the first token comes from the
    prefill head either way.
- **Independence.** `HF2Q_GEMMA_LMHEAD_MMA` is independent of
  `HF2Q_GEMMA_DECODE_KERNELS` (F2). Every A/B in §6 holds the other switch
  fixed and equal in both arms.

## 5. Spike and M0 (model-free; root GPU lane)

The spike is recorded before any hf2q integration. Go requires all three:

1. **Exactness and correctness:** the §4.1 tests 1–2 pass on both metallib
   paths.
2. **Projected saving**, from the spike's M0 cells weighted by the V2 step mix
   and including width 1:
   Σ_w steps_w · (stock_w − mma_w) ≥ **0.30 s per run**.
   - With the V2 mix this is about ≤ 0.62–0.65 × stock at width 4.
   - Basis: the M2 bar (§6 gate 4) on the 13.9 s V2 makespan.
3. **Width-1 bar** (decision 4 ruling), the same bar in the spike and at M1:
   - The width-1 head penalty (mma_1 − stock_1) must cost less than 0.5% of the
     n1-stage makespan. Per n1 decode step, that is
     **mma_1 − stock_1 ≤ 0.005 × T₁**, where T₁ is the n1-stage makespan
     divided by its decode steps. T₁ is therefore at least the GPU time of one
     width-1 decode step.
   - **Frozen value for the spike: 35 µs** (about 1.03 × stock width 1).
     - Basis `[INFERENCE]`: a width-1 Gemma decode step streams about 0.96 GB
       of dense weights, about 0.94 GB of routed experts (8 × top-8 at
       2.23 + about 1.7 MB per layer) and the 0.61 GB head, about 2.5 GB in
       all. At 547 GB/s that is about 4.6 ms, so T₁ ≳ 7 ms with attention and
       overheads.
   - **At M1** the bar becomes min(35 µs, 0.005 × measured T₁). T₁ is taken
     from the M1 N=1 kill-switch GPU per step, or from the B1 Gemma n1-stage
     receipt when available.
   - **Prediction:** a port at the peer's MMA efficiency (about 1,180 µs at
     width 1) misses this bar by about 55 µs. The spike passes only if the
     width-1 MMA reaches about 99% of roofline. Otherwise the story stops and
     records its numbers; there is no separate width-1 route, because
     batched ≡ serial requires the MMA at width 1 (§3.2 item 3).

A miss on any leg stops the story; the code is removed (§11).

## 6. Gates after integration (in order; any failure stops)

Every gate runs on one release binary with the cargo-1.99 four-agent proof and
the post-#704 harness, and records rustc in run-meta.

1. **Class A, serving level** (release build, real artifact, `--exact
   --ignored --test-threads=1`):
   - Slot-aware n1/n4/n4-batched-body/n8/eviction parity tests under `0` and
     under `1`.
   - A new `gemma_lmhead_mma_batched_equals_serial`: four ragged lanes,
     ≥ 256 steps, widths 4 → 3 → 1, plus widths 9..16 through the batched-head
     path (`lm_head_batched` at N = 9..16). Logits are compared u32 per slot
     against the serial MMA head.
   - Both `HF2Q_FUSE_LMHEAD` settings: production fuses the head, M1 runs
     with `=0`.
   - `0` against the pre-integration pin: `ExactTranscripts` over one
     four-agent run each.
   - Speculative verify modes are refused under `1` (§4.2). Combining this
     lever with Q4-GEMMA-SPEC-SERVE needs a later revision that routes the
     verify heads, and SPEC-SERVE gate A1 would then be re-run.
2. **Q1 envelope:** the T2 cohort trace under `0` and `1`, then `envelope`.
   It must be `WithinDecisionEnvelope`; `RouteSensitive` stops (§3.1).
3. **M1** (`slot_aware_n4_batched_body_throughput_probe`, `HF2Q_GPU_BUSY=1`,
   head in its own command buffer with `HF2Q_FUSE_LMHEAD=0`):
   - The §5 leg-2 projected-saving formula, Σ_w steps_w · (stock_w − mma_w)
     over the V2 step mix, evaluated on the in-situ lm-head GPU per step at
     N = 1, 2, 3, 4, must be ≥ 0.30 s per run.
   - At N = 1 the width-1 penalty is within the §5 leg-3 bar.
4. **M2:** four-agent Q5 A/B, ABBA, 4 runs per arm.
   - Extend once to 8 per arm if the CI half-width exceeds 2.5 points; no
     other extension.
   - Median makespan ratio `1`/`0` ≤ **0.98**, with block-paired 95% CI upper
     < 1.00. Baseline: the 13.9 s V2 makespan.
   - `ExactTranscripts` of arm `1` against itself, run to run, across the
     blocks.
   - Worst-slot TTFT p95 and SSE-gap max ratios ≤ 1.025.
5. **Paired quality** (Q1 §3.5) against the same binary at `0`:
   - `HF2Q_GEMMA_DECODE_KERNELS` is equal in both arms;
   - 24 paired instances (6 instance labels × 4 agents), one run per instance
     per arm, production sampling, no reruns;
   - `q1_decision_policy.py paired` = `Pass`.
6. **Combined candidate:** the Q4 closing campaign (Q4-GEMMA-DECODE design
   §8.4) with every adopted Gemma lever. It requires:
   - a Win against the peer's MT configuration;
   - paired quality Pass against v0.1.28 and against the peer;
   - fairness;
   - a fresh Q5.

Promotion flips the default to `1` only after gates 1–5 pass. Otherwise §11
applies.

## 7. Order and cost

| Step | Work | Lane | Estimate |
|---|---|---|---|
| 1 | Design review and approval | CPU | — |
| 2 | mlx-native kernel, tests, bench | CPU, then model-free GPU | about 1 day, about 15 min |
| 3 | Spike go/no-go (§5) | — | — |
| 4 | hf2q integration, gate 1 | CPU, model lane | about 30 min |
| 5 | Gate 2 | model lane | about 10 min |
| 6 | Gates 3–4 (M1, M2) | model lane | about 1 h |
| 7 | Gate 5 (48 four-agent runs) | model lane | about 2–3 h |
| 8 | Gate 6, with the combined candidate | root | — |

## 8. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Width-1 MMA slower than the bandwidth-bound mat-vec (predicted, §5) | Stop at the spike | Stated up front; §5 leg 3 decides |
| Hardware MMA arithmetic depends on a column's position inside the tile | §3.2 item 2 fails at every tiling | The spike tests it first; the story stops |
| Width-invariant tiling costs part of the peer's gain | §5 leg 2 misses | Geometry search; stop on a miss |
| A head defect shows up as D > τ | `RouteSensitive` | Stop and localize (§3.1) |
| Quality campaign underpowered (n = 24) | `Inconclusive` | Stated power (Q1 §3.5 table); no reruns |
| Verify widths 9..16 (Q4-GEMMA-SPEC-SERVE) | Batched ≡ serial at R = 12/16 | Column groups of at most 8 (§4.1); gate 1 widths 9..16 on the batched-head path; verify modes refused under `1` until a revision routes them |

## 9. Repository ownership

mlx-native owns the kernel, its tests and error record, the microbenchmark and
the release. hf2q owns the switch, head-site routing, the gates and the
campaigns, consuming an exact published mlx-native pin.

## 10. Q4 interaction

This lever's verdict is decided on the combined candidate in the Q4 closing
campaign (ADR-063 Q4-GEMMA-ATTN). Serving speculation (Q4-GEMMA-SPEC-SERVE)
removes target steps, which shrinks the saving, and widens verify rows, which
raises it per step.

## 11. Release and clean cutover

1. **mlx-native.**
   - A contract issue carries §3.
   - The PR carries the spike receipts on both paths (tests 1–3, the error
     record, M0 cells and the projected-saving computation).
   - After review: merge, publish the next minor release, verify it on
     crates.io.
2. **hf2q.**
   - An exact-pin commit (`=0.x.y`) from a clean registry build.
   - The integration commit with the switch at `0`, then gates 1–5.
3. **Promotion commit** (only after gates 1–5): default `1`, `0` kept as the
   kill switch, banner, case notes, `scripts/four_agent/README.md`, ADR-063
   receipts.
4. **Failure path** (any stop, at the spike or at a gate). Removal set:
   - **mlx-native:** `quantized_matmul_ggml_q6k_mma` and its export, the MMA
     kernel file and its registration, `tests/gemma_lmhead_mma.rs`, and the
     `lmhead_mma` bench family. If the release already shipped, a follow-up
     minor release.
   - **hf2q:** the switch, head-site routing, the startup refusals, the
     batched-versus-serial MMA test, the case variable and launcher
     forwarding, the env-inventory entry, and the exact pin moved forward to
     the follow-up mlx-native release that removes the kernel.
   - Then the slot-aware suite and `ExactTranscripts` against the
     pre-integration pin run again under the restored code. Receipts are
     recorded.

## 12. Decisions (root, on ReviewLmHead, 2026-11-05)

1. **Open the story:** accepted after findings 1–9.
2. **M2 bar for this lever alone: ≤ 0.98**, with the 0.96 Win share judged on
   the combined candidate. Accepted with the reviewer's conditions: the 13.9 s
   V2 makespan as baseline, the fixed 4 → 8 extension rule, and a spike go
   rule tied to M2 (§5 leg 2, ≥ 0.30 s including width 1).
3. **Bypass the serial head's baked Q6_K record under `1`:** accepted at the
   head call sites only. The shared dispatchers, including `dispatch_qmatmul`'s
   record branch, stay byte-unchanged under `0` and for every other caller.
4. **Width-1 bar:** the width-1 head penalty must cost less than 0.5% of the
   n1-stage makespan, the same bar in the spike and at M1: 35 µs frozen for
   the spike; min(35 µs, 0.005 × measured T₁) at M1.

## 13. Review record (ReviewLmHead, REVISE, 12 findings)

| Finding | Resolution |
|---|---|
| 1. Width-1 bar predicted to fail; the peer's width-1 row is mat-vec | §2 relabels the row and states the prediction; §5 leg 3 (decision 4) uses one bar for spike and M1 |
| 2. Gate 2 must stop on `RouteSensitive` | §3.1 and §6 gate 2 stop; risk row corrected |
| 3. No functional correctness test | §4.1 test 1: CPU F64 reference with a K-scaled bound, part of the spike's exactness leg |
| 4. Routing inside shared dispatchers | §4.2: head call sites only, `resolved_lm_head()` identity predicate, every decode-time head site listed, prefill heads stock, refusal for unroutable sites; §3.3 |
| 5. Widths > 8 broke batched ≡ serial | §3.2 item 2 and §4.1: column groups of at most 8, m = 1..16; gate 1 widths 9..16; SPEC-SERVE A1 re-run |
| 6. Spike bar admitted M2 misses | §5 leg 2: projected V2-weighted saving ≥ 0.30 s including width 1 |
| 7. Gate order and open-ended M2 | §6 order Class A → envelope → M1 → M2 → paired → combined; 4 → 8 extension rule; 13.9 s baseline cited |
| 8. Serving-level Class A missing | §6 gate 1: slot-aware suite at `0` and `1`, both `HF2Q_FUSE_LMHEAD` settings, `0` against the pre-integration pin; gate 4 `ExactTranscripts` of `1` against itself; gate 5 reference at `0` with `HF2Q_GEMMA_DECODE_KERNELS` equal |
| 9. Clean cutover under-specified | §11 release section and removal set in both repositories |
| 10. Wrong parameter named as setting the order | §3.2 item 2: NSG and K-chunk assignment fixed; tile-position risk added (§8) |
| 11. Envelope coverage overstated | §3.1: 64 of 576 rows exercise the MMA |
| 12. ADR consistency | ADR-063 header, Q4-GEMMA-ATTN predecessors and the decision log list this story; "within rounding" marked `[INFERENCE]` |

ACCEPT review (ReviewLmHead, revision 2): required edit applied in §4.2 (serving decode heads only; DFlash, n-gram, EAGLE-3 tree-verify and kernel-profile modes refused at startup under `1`; K = 2816 back in the predicate) and §6 gate 1 (widths 9..16 via `lm_head_batched`). P3s applied: M1 uses the §5 projected-saving formula on in-situ numbers (§6 gate 3); the failure path moves the pin forward to the follow-up release that removes the kernel (§11).

## 14. Outcome: spike NO-GO (2026-11-05)

Spike `57dd7c0` (mlx-native branch `feat/gemma-lmhead-mma` on main `f01d0ac`,
kept unpushed as evidence), measured under the root's GPU lane 20:09–20:18Z.
Receipts: `~/.local/state/hf2q/v0130/q4-gemma-lmhead-mma/spike-57dd7c0-20261105T195427Z/`
(`tests-path{1,0}.txt`, `path{1,0}/error_record.json`, `bench/cells.jsonl`,
`bench/identity.json`, `toolchain.txt`; macOS 26B5112a, metal 32023.922,
rustc 1.99.1).

| §5 leg | Bar | Result | |
|---|---|---|---|
| 1. Exactness and correctness | §4.1 tests 1–2 on both metallib paths | 7/7 on both paths: F64 K-scaled bound at m = 1..16 and tails, determinism, width and tile-position invariance, NaN/∞ neighbours, permutation, refusals | Pass |
| 2. Projected V2 saving incl. width 1 | ≥ 0.30 s per run | best 0.237 s (`nsg4_nt4`; `nsg8_nt4` 0.234 s) | **Fail** |
| 3. Width-1 penalty | ≤ 35 µs | best +103 µs (`nsg4_nt2`: 1,187 vs 1,084 µs); others +187 to +350 µs | **Fail** |

Cold-rotation medians (µs) for the stock head and the best MMA cells:

| Width | Stock | `nsg4_nt2` | `nsg4_nt4` | `nsg8_nt4` |
|---:|---:|---:|---:|---:|
| 1 | 1,084 | 1,187 | 1,280 | 1,271 |
| 2 | 1,208 | 1,331 | 1,290 | 1,319 |
| 4 | 1,980 | 1,432 | 1,378 | 1,380 |
| 8 | 4,054 | 1,519 | 1,443 | 1,434 |
| 16 | 4,644 | 2,642 | 2,498 | 2,351 |

- The port is exact in every Class A sense the design required (the
  hardware tile-position risk did not occur) and about 17% slower than the
  peer's MMA at width 4 (1,378 against 1,181 µs).
- Leg 3 is decisive: even at the peer's own MMA times the width-1 penalty is
  about +96 µs, above the 35 µs bar. Batched ≡ serial requires the MMA at
  width 1 (§3.2 item 3), so no geometry search can recover it under this
  contract.
- Per §5 the story stops. Nothing was merged or released; per §11 the code
  stays out of both repositories (the spike branch is retained unpushed as
  evidence). No hf2q integration was started.
