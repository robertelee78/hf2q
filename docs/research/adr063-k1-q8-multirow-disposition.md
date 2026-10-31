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

The geometric-mean effect is +1.2% faster, 95% CI −1.8% to +4.4% (Welch on log values, p = 0.40). The independent reviewer recomputed it: +1.25% on F2 makespan, +1.15% on tokens/s, robust to session adjustment and to bootstrap. The test resolves effects of about 4% or more. All 14 runs produced the same two final-source hashes.

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

The paired task-quality campaign against stock runs on the final main binary and is recorded under S3/Q3.

## 3. Other families

- **Qwen3.6 (MoE):** joint decode is off by default. MoE expert products run per slot, and the dense projections follow the Qwen3.8 routes when joint decode is enabled. Whether to promote it is decided under S3/D1, not K1.
- **Gemma 4:** decode already uses the batched body at four slots (pre-existing). K1 found no per-row K-quant decode product to change. Its remaining four-agent gap is prefill, handled under S3.

## Evidence

- DeepSeek-V4: `E/k1-ab-v1` (aborted, no timing), `E/k1-ab-v2`, `E/k1-ab-v3`, `E/k1-ab-v3/analysis-pooled.json`, `E/k1-spike-v1/exactness.log`.
- Qwen3.8: `E/qwen-ffn-gates-v1`, `E/qwen-joint-ffn-ab-v1`, `E/q1-qwen38-joint-envelope-20261031T*`, and PR #480.
