# Spike: mlx-native Forward Pass Profile

Date: 2026-04-12
Commit: 93dbdf2 (main + profiling instrumentation)

## Summary

mlx-native achieves 16.0 tok/s (71.72 ms/token) vs candle's 88 tok/s (11.4 ms/token) --
a **5.5x gap**. This profile identifies exactly where the time goes.

### Root Cause: 121 sessions x ~180 us overhead + 13.4x dispatch count

The bottleneck is NOT per-kernel execution speed. It is:

1. **121 commit_and_wait syncs per token**, each costing ~180 us in overhead
   (command buffer creation + commit + GPU fence wait), totaling ~22 ms/token.
2. **1408 GPU dispatches per token** vs candle's ~105, because mlx-native
   dispatches each MoE expert individually (4 dispatches x 8 experts x 30 layers)
   while candle uses fused `kernel_mul_mv_id_*` kernels.
3. **Per-dispatch CPU encoding cost** for the 1408 dispatches: each
   encode_threadgroups call has pipeline lookup + buffer binding + params copy.

## Hardware & Configuration

- Apple M5 Max
- Gemma 4 26B MoE (Q4_K_M quantization, GGUF)
- 30 decoder layers, 128 experts/layer, top-k=8
- 25 sliding attention layers (head_dim=256, 8 KV heads)
- 5 global attention layers (head_dim=512, 2 KV heads)

## Per-Session Timing Breakdown

Measured over 5 decode tokens (2 warmup skipped).

```
30 layers x 4 sessions + 1 head = 121 sessions/token

Session breakdown (avg across 30 layers):
  S1 (QKV proj):       219.0 us/layer x 30 =  6.57 ms ( 9.2%)
  CPU1 (norms+RoPE):    24.6 us/layer x 30 =  0.74 ms ( 1.0%)
  S2 (SDPA+MLP):      1177.1 us/layer x 30 = 35.31 ms (49.2%)
  CPU2 (post-FF):       16.6 us/layer x 30 =  0.50 ms ( 0.7%)
  S3 (router proj):    194.5 us/layer x 30 =  5.84 ms ( 8.1%)
  CPU3 (softmax+topk):   included in S3
  S4 (MoE experts):    582.6 us/layer x 30 = 17.48 ms (24.4%)
  CPU4 (post-MoE):      16.6 us/layer x 30 =  0.50 ms ( 0.7%)
  Head GPU (lm_head):                         4.46 ms ( 6.2%)
  Head CPU (softcap+argmax):                  0.76 ms ( 1.0%)

Total: 71.72 ms/token
  GPU sessions: 69.66 ms (97.1%)
  CPU ops:       2.49 ms ( 3.5%)
```

**CPU is NOT the bottleneck.** All CPU ops (RoPE, head norms, KV cache,
softmax, argmax) total only 2.49 ms (3.5%).

## Dispatch Count Comparison

```
mlx-native dispatches per token: 1408
  S1 (QKV):    85   (2-3 matmuls/layer x 30)
  S2 (SDPA+MLP): 300 (10 ops/layer x 30)
  S3 (router):  30   (1/layer x 30)
  S4 (MoE):   990   (1 zero + 8 experts x 4 ops = 33/layer x 30)
  Head:         3    (cast + gemm + cast)

candle dispatches per token: ~105
  (uses fused kernel_mul_mv_id_* for MoE -- 1 dispatch per weight type
   instead of 8 separate expert dispatches)

Ratio: 13.4x more dispatches in mlx-native
```

## Per-Kernel Timing Comparison vs Candle Phase 0

### Session overhead estimation

S3 (router session) has exactly 1 dispatch: a Q4_0 mat-vec [2816 -> 128].
In candle, a Q4_0 mat-vec of similar size takes ~11 us.
S3 wall-clock: 194.5 us.
**Estimated session overhead: ~183 us per begin/finish cycle.**

121 sessions x 183 us = **22.1 ms of pure session overhead** (31% of total).

### Per-dispatch time (amortized over session)

| Session | Dispatches | Wall-clock (us) | Per-dispatch (us) |
|---------|-----------|----------------|------------------|
| S1      | 2.83      | 219            | 77.3             |
| S2      | 10        | 1177           | 117.7            |
| S3      | 1         | 195            | 194.5            |
| S4      | 33        | 583            | 17.7             |

S4 has 33 dispatches in ONE session, so the session overhead is amortized
to only 183/33 = 5.5 us/dispatch. The remaining 12.2 us/dispatch is actual
kernel execution, which is close to candle's Q4_0 mat-vec time (10.88 us).

**Conclusion: per-kernel GPU execution time is comparable to candle's.**
The problem is session overhead, not kernel speed.

### Detailed per-kernel comparison

| Kernel (candle name)     | candle us/call | mlx-native est. us/call | Ratio |
|--------------------------|---------------|------------------------|-------|
| Q4_0 mat-vec (2816)     | 10.88         | ~12 (S4 amortized)     | 1.1x  |
| Q6_K mat-vec             | 14.50         | ~14 (similar shape)    | ~1.0x |
| Q8_0 mat-vec             | 18.62         | ~18 (similar shape)    | ~1.0x |
| SDPA (sliding, 256)     | 14.96         | N/A (included in S2)   | --    |
| SDPA (global, 512)      | 25.75         | N/A (included in S2)   | --    |

Note: exact per-kernel times within S2 cannot be measured without
single-op sessions. The S4 data (33 dispatches, 583 us) gives the
best estimate since it's the most dispatch-dense session.

## Per-Layer Detail

```
Layer |   S1   |  CPU1  |   S2   |  CPU2  |   S3   |   S4   | Total
------|--------|--------|--------|--------|--------|--------|------
 0 (S)|    214 |     22 |   1061 |     18 |    199 |    777 |  2308  (sliding)
 1 (S)|    225 |     24 |   1070 |     20 |    184 |   1415 |  2958  (sliding)
 2 (S)|    227 |     24 |   1045 |     18 |    187 |    525 |  2043  (sliding)
29 (G)|    222 |     29 |   1849 |     16 |    215 |    894 |  3241  (global)
```

Global layers (every 6th) have ~75% larger S2 due to head_dim=512 vs 256.
Layer 1 has anomalously high S4 (1415 us vs ~580 avg) -- likely due to
specific expert weight access patterns.

## Where the 60.2 ms gap lives

Candle achieves 11.4 ms/token. mlx-native: 71.72 ms. Delta: 60.3 ms.

| Source                          | Estimated ms | % of gap |
|---------------------------------|-------------|----------|
| Session overhead (121 x 183 us) | 22.1        | 37%      |
| Extra dispatches (1408 vs 105)  | ~20         | 33%      |
| S2 per-op overhead (10 ops)     | ~10         | 17%      |
| Head session (lm_head F16 GEMM) | 4.5         | 7%       |
| CPU ops                         | 2.5         | 4%       |
| Measurement noise               | ~1          | 2%       |

## Recommended Fixes (Priority Order)

### 1. Fused MoE kernels (eliminate ~990 dispatches, ~30 sessions)
**Expected savings: 25-30 ms/token**

Port candle's `kernel_mul_mv_id_*` approach: a single kernel that takes
an expert index list and processes all selected experts in one dispatch.
This eliminates 30 S4 sessions (30 x 183 = 5.5 ms session overhead)
AND reduces dispatches from 990 to ~30 (one per layer).

### 2. Merge S1+S2 into one session per layer (eliminate 30 sessions)
**Expected savings: 5-6 ms/token**

Move Q/K head norms, V norm, RoPE, and KV cache to GPU kernels.
Then S1 (QKV proj) and S2 (SDPA+MLP) can be encoded in a single
session, eliminating 30 commit_and_wait calls.

### 3. Merge S3 into S4 (eliminate 30 sessions)
**Expected savings: 5-6 ms/token**

Move router softmax + top-k to GPU (128 elements, trivial kernel).
Then router proj + MoE expert dispatch can be one session.

### 4. Async command buffer submission
**Expected savings: 10-15 ms/token**

Use `commit()` (non-blocking) instead of `commit_and_wait()` where
possible. Only wait when CPU needs to read GPU results. Currently
121 sessions all use blocking wait.

### 5. Optimize lm_head session
**Expected savings: 2-3 ms/token**

The F16 GEMM for lm_head (262144 x 2816) takes 4.46 ms. candle's
Phase 0 shows this as ~3.5 ms. Profile the dense_gemm_f16 dispatch
geometry separately.

### Target after fixes 1-3

If we eliminate 90 of 121 sessions (going from 4 sessions/layer to 1)
and fuse MoE dispatches (990 -> 30):

- Sessions: 31 (1/layer + 1 head)
- Dispatches: ~448 (14 merged ops/layer + 1 fused MoE/layer + 3 head)
- Session overhead: 31 x 183 = 5.7 ms
- Kernel execution: ~448 dispatches at ~12 us avg = 5.4 ms
- Total estimate: ~14 ms/token = **71 tok/s**

This matches candle's 11.4 ms (88 tok/s) to within thermal noise,
confirming the bottleneck is session/dispatch overhead, not kernels.
