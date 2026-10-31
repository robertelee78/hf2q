# ADR-063 K1: small-row quantized decode products, disposition

Status: measured 2026-10-30/31 on an M5 Max with 128 GB.

- **DeepSeek-V4 Q8_0:** the exact multi-row kernel shows no end-to-end improvement. It is not published.
- **Qwen3.8 K-quant FFN:** the weight-shared route with existing kernels is a measured win. It landed in #480.

`E` is the ADR-063 campaign root `adr063-preflight-20261023`.

## 1. DeepSeek-V4: exact multi-row Q8_0 matvec

**Hypothesis.** Joint B-row decode (#458) runs each row-parallel product at m = B, 2 to 4 rows. A Q8_0 kernel that reuses each dequantized weight block across the B rows should beat the per-row matvec route while staying bitwise equal to it.

**Candidate.** The mlx-native branch `perf/q8-0-mv-multirow-exact` (e50e7e9) adds `kernel_mul_mv_q8_0_f32_nr2_r{2..8}`. It keeps the per-row accumulation order.
- The 8-wide DeepSeek-V4 joint decode exactness test passes on the official artifact with the kernel in place (`E/k1-spike-v1/exactness.log`).
- The kernel-level figures (bitwise parity at every width; 1.15–1.46× faster at M = 4; 217 suite summaries with 0 failures) come from the mlx-native commit message only. No receipts were saved, so treat them as unverified.
- The kernel was wired in through a local dependency override in an isolated spike worktree. The override was never committed, and ADR-063 rules it out as acceptance evidence.

**End-to-end measurement.** The workload is the DeepSeek-V4 four-agent C262144 coding task under the Q5 supervisor. The metric is the supervisor's client elapsed time (`client.elapsed_seconds`) for the four-agent run.
- Arms: J = joint decode, binary `ds4-joint-bin-fab4dc3b`; K = J plus the kernel, binary `k1-spike-v1`.
- Sessions: an earlier session, `E/k1-ab-v1`, stopped all 6 runs on memory guards before any timing, on both arms, so it holds no data. The two sessions that count ran JKKJJK and then JKKJJKKJ. All 14 runs were CalibrationPass with cleanup proven.
- Between the two sessions the case file and the client script changed. Both arms ran under each version, so the comparison between arms is not biased.

| Arm | n | Median elapsed |
|---|---:|---:|
| J (joint decode) | 7 | 106.0 s |
| K (joint + multi-row Q8_0) | 7 | 104.3 s |

The geometric-mean effect is +1.2% faster, 95% CI −1.8% to +4.4% (Welch on log values, p = 0.40). The independent reviewer recomputed it: +1.25% on client elapsed time, +1.31% on the F2 own-output makespan and +1.15% on tokens/s, robust to session adjustment and to bootstrap. The test resolves effects of about 4% or more. All 14 runs produced the same two final-source hashes.

**Disposition.** There is no detectable end-to-end improvement, so the kernel is **not published or pinned**, and the branch stays unmerged as evidence. [INFERENCE] Q8_0 matvec is a small share of the DeepSeek-V4 joint step; no stage attribution was kept.

## 2. Qwen3.8: K-quant products at the live verifier width

The live joint width is 4: padded `mul_mv_ext` mode is fixed at 4, so widths 5–8 never occur in serving. Every Qwen3.8 decode product was audited at m = 4 under the serving policy (`HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1`), using mlx-native 0.15.15 and the official GGUF header:

- **Already one weight pass per step:** DeltaNet `attn_qkv` (Q6_K/Q4_K), `attn_gate`, `ssm_alpha`/`ssm_beta`, `ssm_out`, attention q/k/v/o, `ffn_down` (Q6_K/Q4_K) and the output head (Q6_K). All of them go through `mul_mv_ext` with 4 rows per threadgroup.
- **One product per row:** FFN gate/up (Q4_K, 17408 × 5120, 64 layers). The fused gate/up/SiLU kernel launched one input row per `tgpig.y`, which re-read gate and up 4 times per step. That is 19.25 GB of the 35.06 GB a four-row step reads.

**Outcome (#480).** Gate and up now run in padded mode as weight-shared `mul_mv_ext` products followed by `silu_mul`, which brings a step down to 15.81 GB. Serial decode, prefill and MTP keep the fused kernel.

**Exactness.** This is Q1 Class B against stock. The real-prompt envelope is `WithinDecisionEnvelope`: max decision error 0.179 nat, top-1 agreement 128/128, 0 flips (`E/q1-qwen38-joint-envelope-20261031T*`). Cohort invariance passes on both FFN routes (`E/qwen-ffn-gates-v1`).

**Four-agent A/B against the kill switch** (ABBA × 4, same binary; `E/qwen-joint-ffn-ab-v1/analysis.json`):
- Makespan 91.4 → 87.8 s, −4.9%, block-paired 95% CI 2.2–7.6%.
- Per-request decode median 12.19 → 13.21 tok/s.
- The bandwidth model projected up to 2.2× faster joint steps. That did not hold.

**Why `mul_mv_ext` at width 4.** The October 28 primitive packet (`E/native-k1-existing-routes-actual-v1`, reviewed in `/tmp/adr063-k1-existing-route-independent-actual-v1.md`) compared the existing routes for Q4_K. Against the default route, `mul_mv_ext` ran −13.6% GPU median at M4 and −24.4% at M8, but +22% slower at M2. That is why padded mode packs to width 4.

The paired task-quality campaign against stock, and the makespan comparison against stock with #480 included, run on the final main binary and are recorded under S3/Q3.

## 3. Other families

- **Qwen3.6 (MoE):** joint decode is off by default. When enabled it runs in exact mode only, because `select_joint_route_mode` refuses MoE artifacts under `HF2Q_DECODE_MV_EXT=1` (`joint_decode_routes.rs:92-104`). It therefore uses the mvN, `Mv` or canonical Q5_K routes, never padded `mul_mv_ext`. The shared-weight FFN is never selected in exact mode, and the MoE FFN runs per row (`joint_decode.rs:448-470`). K1 keeps the existing exact routes; promotion is decided under S3/D1.
- **Gemma 4:** K1 did not audit Gemma decode. [INFERENCE from code] The default-on batched body sends quantized dense projections through `dispatch_qmatmul` at m = N (`batched_body.rs:469-475`). MoE gate/up and down are forced onto per-token `mv_id` to keep the batched body bitwise equal to serial decode (`batched_body.rs:2005-2070`). Sharing those weights across rows would break that contract and is out of K1's scope. Gemma's remaining four-agent gap is prefill, handled under S3.

## Evidence

- DeepSeek-V4: `E/k1-ab-v1` (aborted, no timing), `E/k1-ab-v2`, `E/k1-ab-v3`, `E/k1-ab-v3/analysis-pooled.json`, `E/k1-spike-v1/exactness.log`.
- Qwen3.8: `E/qwen-ffn-gates-v1`, `E/qwen-joint-ffn-ab-v1`, `E/q1-qwen38-joint-envelope-20261031T*`, and PR #480.
