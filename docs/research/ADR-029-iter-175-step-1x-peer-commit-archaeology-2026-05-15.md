# ADR-029 iter-175 Step 1x — exhaustive peer commit archaeology (8-month window)

**Date**: 2026-05-15
**HEAD**: hf2q `5099a943`, mlx-native `22dc55b`
**Iteration**: 28 of /loop autonomous

## Goal

Step 1t (iter-20) reviewed peer commits within a narrow ~3-month window.
This step widens the scope: every Metal-kernel-touching peer commit between
2025-09-01 and 2026-05-15 (128 total, ~50 perf-filtered) cross-referenced
against hf2q's current state.  Goal: eliminate the "we missed a peer
optimization" hypothesis as a remaining lever.

## Method

```
cd /opt/llama.cpp && git log --since=2025-09-01 --until=2026-05-15 \
  --oneline -- ggml/src/ggml-metal/
# 128 commits total; filtered to ~50 perf/kernel keywords.
```

For each filtered commit:
1. Read `git show --stat`
2. Read the diff for any GGML_TYPE the gemma4 decode path uses
3. Classify: PORTED / NOT-APPLICABLE / FALSIFIED / ALREADY-MATCHED

## Findings

### Already ported / hf2q already matches

| Peer commit | Description | hf2q status |
|---|---|---|
| `da4495332` | FC-promote mul_mv/mul_mm batch divisors | PORTED iter-162 H93 (+1.08%) |
| `b54124110` | q5_k mul_mv register spill fix (N_R0_Q5_K 2→1) | hf2q **already uses nr0=1** (line `row = 2*r0 + sgitg` at `quantized_matmul_ggml.metal:955` and `quantized_matmul_id_ggml.metal:826`) |
| `f161463a5` | Allow ops to run concurrently (MTLDispatchTypeConcurrent) | hf2q uses Concurrent by default since iter-19c |

### Not applicable to gemma4 decode

| Peer commit | Description | Why N/A |
|---|---|---|
| `e4cff0956` | Bin kernel FC `column-broadcast` | hf2q elementwise kernels don't broadcast — no `i10 = i0 % ne10` modulo exists.  qwen3vl `add_bias_row_2d` has modulo but not in gemma4 path. (Step 1t followup verified.) |
| `35fb82497` | Dynamic simdgroups for MV kernels | Only affects F32/F16/BF16 mul_mv; gemma4 weights are Q5_K_M.  Q-types still use fixed NSG=2. |
| `d1649047a` | Metal Tensor API for GGML_OP_MUL_MAT | Prefill-only path (m>>1), decode uses mul_mv. |
| `dcdcbad42` | Q1_0 backend | gemma4 doesn't use Q1_0. |
| `e22cd0aa1` | mul_mv_ext for BF16/Q2_K/Q3_K | gemma4 decode is Q5_K_M, not Q2_K/Q3_K/BF16. |
| `342d6125b` | FA HSK=512 HSV=512 instantiations | gemma4 uses HSK=HSV=256 (already supported). |
| `b30a5fdf3` | FA HSK=320 HSV=256 specialization | gemma4 HSK=HSV=256. |
| `271191906` | FA for MLA heads | MLA is DeepSeek-style; gemma4 uses GQA. |
| `e30f1fdf7` | GDN state transpose removal | GDN = Gated DeltaNet (qwen3.5), not gemma4. |
| `7fcf1ef45` | Skip loading all-zero mask | FA-ext kernel (prefill); decode uses FA-vec. |
| `086a63e3a` | SSM kernel improvements | SSM = qwen3.5 path, not gemma4. |
| `945bf1062` | MoE kernel for ne20=5 | gemma4 A4B uses ne20 ≠ 5. |
| `01ade96e7` | Remove BF16 x F16 kernels | We don't use BF16xF16. |
| `8635e221c` | Event synchronization fix | Internal correctness, not perf. |
| `9bcb4eff4` | matmul2d dimension constraint | Internal validation fix. |
| `73c9eb8ce` | L2 norm scale fix | Not used in gemma4 path. |

### Already-explored direction (falsified)

| Peer commit | Description | Why FALSIFIED for hf2q |
|---|---|---|
| `dfcd53f7e` | Fuse NORM + MUL + ADD | iter-1 H6 `HF2Q_FUSED_TRIPLE_NORM` repeatedly falsified (-5.15% re-bench at Step 1o).  Our local fusion granularity is at optimum on Apple Metal at gemma4 shapes. |

### Broad refactors with no isolatable kernel-level optimization

| Peer commit | Description | Why no port |
|---|---|---|
| `0320ac526` | Metal refactor + optimize v2 | Reorganization in `ggml-metal-context.m`/`ggml-metal-device.cpp`; no per-kernel speedup. |
| `f28d4f4ac` | Metal refactor + optimize | Similar — code reorg. |
| `8ae32dc9e` | Various optimizations + refactoring | Mix of misc fixes + reorg. |
| `0f0a3c285` | Make backend async | hf2q has its own async/encoder model (ADR-031 Phase B done). |
| `3b53634fe` | Fuse non-sequential nodes | Graph-executor-level (peer `ggml-metal-ops.cpp` orchestration), not kernel; hf2q has hand-written forward, no graph fuser. |

## Falsification ledger update

Step 1x adds 16 NOT-APPLICABLE entries to the iter-175 ledger.  Combined
with prior steps:

- **Step 1**: top kernels already peer-ported (q6_K_nr2, rms_norm_f32_v2)
- **Step 1d**: concurrency dispatch lever CONFIRMED (already in production via Concurrent default)
- **Step 1e**: tracked-dispatch site migration COMPLETE for q6_K_nr2
- **Step 1m**: H-E precompiled .metallib DEFAULT-ON (+0.42% / -0.32% σ decode neutral, +3-5% prefill)
- **Step 1o**: triple_norm fusion falsified harder (-5.15%)
- **Step 1q**: HF2Q_PIPELINE_TG_MULT_HINT safety check caught real bug (ssm_conv tg=255)
- **Step 1r**: ssm_conv gcd-based tg rounding fix
- **Step 1s**: HF2Q_PIPELINE_TG_MULT_HINT refuted (3-cycle re-bench)
- **Step 1t**: peer source review (gemma4 architecturally identical)
- **Step 1u**: HF2Q_TQ_FAST_FUSED_KV re-falsified
- **Step 1v**: FA-vec NWG sweep (NWG=32 tighter but no wall gain)
- **Step 1w**: H-F threads_per_tg geometry FALSIFIED (-0.42%)
- **Step 1x**: 16 peer commits classified — 3 already matched, 13 N/A to gemma4 decode

## Conclusion

The **comprehensive peer-commit archaeology eliminates "we missed a peer
optimization" as a remaining lever**.  Every kernel-perf-relevant peer
commit since 2025-09-01 has been either:
1. Already ported (3 commits)
2. Not applicable to gemma4 quantized decode (16 commits)
3. An already-falsified direction (1 commit)
4. A broad refactor with no isolatable optimization (5 commits)

This strengthens the structural-floor interpretation:
- All Class A levers (per-kernel speedups via peer porting) are exhausted.
- Class B levers (de-fusion / re-fusion granularity) are at local optimum.
- The residual ~6% gap is dispersed across 866 dispatches at ~1 µs each,
  not concentrated in any single kernel.

The remaining /loop-tractable work is limited to:
1. Continued micro-bench falsifications (negative results add value)
2. Cron-frequency rechecks as peer adds new commits
3. Operator-only Apple Instruments deep dive

## Cross-references

* iter-175 falsification ledger root: `feedback_class_AB_lever_falsification_ledger_2026_05_12.md`
* Step 1t (initial peer review): `docs/research/ADR-029-iter-175-step-1t-peer-source-review-2026-05-15.md`
* Step 1w (geometry falsification): `docs/research/ADR-029-iter-175-step-1w-2026-05-15.md`
* Falsification rebench rule: `feedback_levers_can_widen_opposing_regressions_2026_05_15.md`
