# ADR-063 K1: exact multi-row Q8_0 matvec, disposition

Status: measured on 2026-10-30, M5 Max 128 GB. **No end-to-end improvement, so nothing is published.**

## Hypothesis

Joint B-row decode (#458, #466) runs each row-parallel product at m = B (2–4 rows).
K1's hypothesis: for the Q8_0 products in DeepSeek-V4's context path, a kernel that
reuses each dequantized weight block across the B rows should beat the existing
per-row matvec route. It should do so while staying bitwise equal to that route.

## Candidate

The candidate is an mlx-native branch, `perf/q8-0-mv-multirow-exact` (e50e7e9). It
adds a Q8_0 matvec that handles 2–8 rows and keeps the existing per-row
accumulation order.

- Exactness: bitwise parity tests against the per-row route pass for all widths.
  The full mlx-native suite also passes (217 summaries, 0 failures).
- Microbenchmark: 1.15–1.46× faster at M = 4 on the frozen DeepSeek-V4 shapes.
- Serving exactness: the 8-wide DeepSeek-V4 joint decode exactness test still
  passes on the official artifact with the kernel in place.

The candidate was integrated through a local dependency override in an isolated
spike worktree. The override was never committed. ADR-063 rules out such an
override as acceptance evidence.

## End-to-end measurement

The workload is the DeepSeek-V4 four-agent C262144 coding task under the Q5
supervisor. Both arms use the same model, case and harness:

- J: joint decode, binary `ds4-joint-bin-fab4dc3b`.
- K: joint decode plus the kernel, binary `k1-spike-v1`.

Arms were interleaved over two sessions (JKKJJK, then JKKJJKKJ). All 14 runs are
CalibrationPass with the protocol complete.

| Arm | n | Median makespan |
|---|---:|---:|
| J (joint decode) | 7 | 106.0 s |
| K (joint + multi-row Q8_0) | 7 | 104.3 s |

The effect on the geometric-mean makespan is +1.2% faster (95% CI −1.8% to
+4.4%). The first six runs alone gave +3.3% (CI −0.6% to +7.3%), and adding runs
moved the estimate toward zero.

## Disposition

The kernel is faster in isolation but does not move the four-agent makespan by a
detectable amount. Q8_0 matvec is a small share of the joint decode step once the
B rows share one forward. K1's acceptance requires F2 end-to-end improvement
before acceptance, so the kernel is **not published or pinned**. The mlx-native
branch stays unmerged as evidence.

The K-quant verifier widths K1 prioritizes (Q4_K/Q5_K/Q6_K at widths 4–8) are
served by the existing mvN / mul_mv_ext routes in the Qwen joint decode (#466).
Their exactness and end-to-end gains are recorded under S3, not here.

Evidence:

- `E/k1-ab-v2/` and `E/k1-ab-v3/`: receipts, plus the pooled analysis in
  `E/k1-ab-v3/analysis-pooled.json`.
- `E/k1-spike-v1/exactness.log`.

`E` is the ADR-063 campaign root `adr063-preflight-20261023`.
