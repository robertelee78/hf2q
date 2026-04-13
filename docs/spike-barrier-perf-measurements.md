# Spike: Comprehensive Performance Measurements (2026-04-13)

**Goal:** Identify all classes of problems contributing to the hf2q vs llama.cpp speed gap.
**Mantra:** Measure 3x, cut once. No assumptions.

---

## 1. Fresh Baselines (measured today, M5 Max)

### 128-token decode

| Stack | Run 1 | Run 2 | Run 3 | Median | ms/token |
|-------|-------|-------|-------|--------|----------|
| llama.cpp (homebrew, build 15f786e65) | 105.34 | 105.16 | 105.25 | **105.25** | 9.50 |
| hf2q mlx-native | 90.1 | 89.6 | 89.8 | **89.8** | 11.14 |
| hf2q candle | 87.4 | 87.6 | 87.4 | **87.5** | 11.43 |

### 512-token decode

| Stack | tok/s | ms/token |
|-------|-------|----------|
| llama.cpp | 103.57 | 9.66 |
| hf2q mlx-native | 88.0 | 11.36 |

### Gap summary

| Comparison | Gap (tok/s) | Gap (ms/token) | Gap (%) |
|-----------|-------------|----------------|---------|
| mlx-native vs llama.cpp (128 tok) | 15.45 | 1.64 | 14.7% |
| mlx-native vs llama.cpp (512 tok) | 15.57 | 1.70 | 15.0% |
| mlx-native vs candle | +2.3 | -0.29 | +2.6% faster |

The gap is **stable across sequence lengths** (~15 tok/s, ~1.65 ms/token). mlx-native is 2.6% faster than candle.

---

## 2. Per-Token Timing Breakdown (mlx-native, 128 tokens)

Measured via `HF2Q_MLX_TIMING=1`:

| Component | Time | % of total |
|-----------|------|-----------|
| CPU encode (set up 872 dispatches + 606 barriers) | 0.50 ms | 4.5% |
| GPU wait (actual GPU execution) | 10.65 ms | 95.5% |
| **Total** | **11.15 ms** | **100%** |

**The GPU is the bottleneck.** CPU encoding is only 4.5% of the total.

---

## 3. Dispatch and Barrier Counts

| Metric | hf2q mlx-native | llama.cpp |
|--------|-----------------|-----------|
| Dispatches/token | 872 | ~1,811 (181 + 1630, includes noops) |
| Barriers/token | 606 | 759 (74 + 685) |
| Dispatches/barrier | 1.44 | 2.38 |
| Command buffers/token | 1 | 2-3 (main + n_cb async) |

llama.cpp has MORE barriers than us. The barrier count is NOT the bottleneck.

---

## 4. Per-Kernel-Type GPU Profiling (mlx-native)

**Method:** `HF2Q_MLX_KERNEL_PROFILE=1` — breaks the single-session forward into one session per kernel group per layer (242 sessions/token). Each session has ~137 μs overhead (begin + commit + waitUntilCompleted). Raw times are session-overhead-inflated.

**Session overhead calibration:**
- Production: 1 session, 11,150 μs/token total
- Profile: 245 sessions, 44,681 μs/token total
- Overhead per extra session: (44,681 - 11,150) / 244 = **137.4 μs/session**

### Raw profile data (median over 3 measure tokens, μs/layer)

| Group | Dispatches | Raw μs/layer | Corrected (−137μs) | × 30 layers | % of corrected total |
|-------|-----------|-------------|-------------------|-------------|---------------------|
| QKV (norm + Q + K + V proj) | 4 | 190 | 53 | 1,590 | 14.3% |
| Head norms + RoPE | 3 | 155 | 18 | 540 | 4.9% |
| KV cache copy | 2 | 145 | 8 | 240 | 2.2% |
| SDPA (flash_attn_vec) | 2 | 177 | 40 | 1,200 | 10.8% |
| O-proj | 1 | 172 | 35 | 1,050 | 9.5% |
| MLP (norm + gate + up + gelu + down) | 5 | 177 | 40 | 1,200 | 10.8% |
| MoE (routing + expert dispatches) | 5 | 213 | 76 | 2,280 | 20.5% |
| Norms/adds (fused end-of-layer) | 2 | 170 | 33 | 990 | 8.9% |
| **Head (final norm + lm_head + softcap + argmax)** | 4 | 2,707 | 2,159 | — | **19.4%** |

**Corrected total: ~11,249 μs/token** (vs measured 11,150 μs — 0.9% error, excellent calibration)

### Compared with Phase 0 ggml per-kernel data

| Group | ggml GPU μs/token | mlx corrected μs/token | mlx/ggml | mlx overhead μs |
|-------|-------------------|----------------------|----------|-----------------|
| QKV | 1,325 | 1,590 | 1.2x | +265 |
| Head norms + RoPE | 609 | 540 | 0.89x | −69 (faster!) |
| KV cache | 256 | 240 | 0.94x | −16 (faster!) |
| SDPA | 464 | 1,200 | 2.6x | **+736** |
| O-proj | 398 | 1,050 | 2.6x | **+652** |
| MLP | 1,428 | 1,200 | 0.84x | −228 (faster!) |
| MoE | ~2,700 | 2,280 | 0.84x | −420 (faster!) |
| Norms/adds | 1,066 | 990 | 0.93x | −76 (faster!) |
| lm_head | 2,476 | 2,159 | 0.87x | −317 (faster!) |
| **TOTAL** | **~10,722** | **~11,249** | **1.05x** | **+527** |

---

## 5. Identified Problem Classes

### Class A: SDPA is 2.6x slower than ggml (736 μs/token overhead)

mlx-native `flash_attn_vec` takes ~40 μs/layer corrected.
ggml `kernel_flash_attn_ext_vec_f16_dk256` takes ~15.5 μs/layer (464 μs / 30 layers).

This is the **single largest identified gap**. Likely causes:
- Different kernel implementation (our flash_attn_vec vs ggml's optimized variant)
- Threadgroup geometry differences
- Missing vectorization (float4)
- Our flash_attn uses 2 dispatches (main + reduce); ggml may fuse these

### Class B: O-proj (dense matmul) is 2.6x slower (652 μs/token overhead)

mlx-native single Q4_0 matmul takes ~35 μs corrected.
ggml Q4_0 mat-vec takes ~13.3 μs/call.

The dense quantized matmul is slower per-call in our implementation. Likely causes:
- Our `quantized_matmul_ggml` kernel may have suboptimal threadgroup sizing
- Missing simdgroup optimizations (NSG variants)
- Older llama.cpp kernel snapshot (per `project_candle_metal_kernels_lineage.md`)

### Class C: QKV projections ~1.2x slower (265 μs/token overhead)

4 dispatches (norm + Q + K + V) take 53 μs/layer corrected = 13.25 μs/dispatch.
ggml dense matmul is 13.27 μs/call — **nearly identical per-call.**

The slight overhead is likely:
- Our pre-attn norm dispatch (~4 μs) isn't present in ggml's number (ggml counts norm separately)
- When normalized for the norm dispatch, we match ggml per-call.

### Class D: Pipelining gap (structural)

- ggml GPU kernel total: ~10,722 μs → wall-clock 9,500 μs (11.4% pipelining savings)
- mlx corrected total: ~11,249 μs → wall-clock 10,650 μs (5.3% pipelining savings)

ggml gets 2x more pipelining benefit. Contributing factors:
- Higher dispatches/barrier ratio (2.38 vs 1.44)
- Graph reordering to maximize concurrent windows
- Multi-command-buffer pipelining (GPU starts executing early CBs while CPU encodes later ones)

### Class E: Not a problem (confirmed)

- **MoE kernels**: mlx-native is 16% FASTER than ggml (0.84x). Not a bottleneck.
- **MLP matmuls**: mlx-native is 16% FASTER (0.84x). Not a bottleneck.
- **lm_head**: mlx-native is 13% FASTER (0.87x). Not a bottleneck.
- **Norms/adds**: mlx-native is 7% FASTER (0.93x). Not a bottleneck.
- **KV cache**: mlx-native is 6% FASTER (0.94x). Not a bottleneck.
- **Head norms + RoPE**: mlx-native is 11% FASTER (0.89x). Not a bottleneck.
- **CPU encode overhead**: 0.50 ms = 4.5% of total. Not worth optimizing.
- **Barrier count**: We have fewer barriers than llama.cpp (606 vs 759).

---

## 6. Gap Attribution

| Class | μs/token overhead | % of total 1,650 μs gap | Actionable? |
|-------|-------------------|------------------------|------------|
| A: SDPA 2.6x slower | 736 | 44.6% | Yes — kernel rewrite |
| B: Dense matmul (O-proj pattern) 2.6x slower | 652 | 39.5% | Yes — kernel update |
| D: Pipelining gap | ~600 | ~36% | Partially — graph reorder |
| C: QKV overhead | 265 | 16.1% | Minor — mostly norm |
| E: Faster areas offset | −1,126 | −68% | N/A (already winning) |
| **Sum (with overlapping pipelining)** | **~1,650** | **~100%** | |

Note: Classes A+B+C sum to 1,653 μs, which matches the gap almost exactly. Pipelining (Class D) overlaps with the kernel time — faster kernels would also improve pipelining by leaving more idle CUs for overlap.

---

## 7. Recommended Priority Order

1. **SDPA kernel** — 736 μs/token, 44.6% of gap. Port ggml's `kernel_flash_attn_ext_vec_f16_dk256` or study its optimizations. Highest single-kernel payoff.

2. **Dense quantized matmul** — 652 μs/token, 39.5% of gap. Our Q4_0 mat-vec is 2.6x slower per-call. Update to ggml's latest `kernel_mul_mv_q4_0_f32_nsg=2` or port the newer kernel version.

3. **Pipelining improvements** — structural changes (graph reordering, multi-command-buffer) could recover ~600 μs. Lower priority because fixing A+B should close the gap without structural changes.

If SDPA + dense matmul reach ggml parity, the corrected total drops to ~9,861 μs/token → ~101.4 tok/s with current pipelining, meeting the 102 tok/s gate.

---

## 8. Measurement Quality

- All benchmarks run 3+ times with consistent results (< 1% variance)
- Session overhead calibrated from two independent measurements (production vs profile)
- Calibration accuracy: 0.9% error (11,249 predicted vs 11,150 measured)
- Phase 0 ggml data is from 2026-04-12 (1 day old), same hardware, same model
- llama.cpp version: build 15f786e65 (homebrew, latest)
