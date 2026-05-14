# ADR-030 Phase 1 — M5 Max Python MLX Baseline Results

**Date**: 2026-05-13
**Bench**: `scripts/adr030/phase1_validate.py` block_size=16 (K=15)
**Hardware**: Apple M5 Max, 137 GB unified memory, `applegpu_g17s`
**Target**: `mlx-community/gemma-4-26b-a4b-it-4bit`
**Draft**: `z-lab/gemma-4-26B-A4B-it-DFlash` (820 MB BF16, 425M params)
**Protocol**: Alt-pair thermal-fair per `feedback_metal_bench_protocol_2026_05_12` — 5 cycles, 60s cool-downs, 3 prompts, max_tokens=256, temperature=0
**Status**: ❌ **FAILS 1.6× go/no-go gate** (mean 0.887×)

## Headline result

| Prompt | Baseline (t/s, σ%) | DFlash (t/s, σ%) | Accept/step (of 15) | Speedup |
|---|---|---|---|---|
| Math (gsm8k-style) | 125.75 (0.47%) | 149.30 (4.55%) | 5.98 = 39.9% | **1.187×** |
| Explainer (Flash Attention) | 126.29 (0.39%) | 58.20 (0.17%) | 2.28 = 15.2% | **0.461×** |
| Code (Rust fibonacci) | 126.45 (0.24%) | 128.25 (0.96%) | 5.02 = 33.5% | **1.014×** |
| **Overall mean** | **126.16** | **111.92** | **4.43 = 29.5%** | **0.887×** |

All σ-pct < 5% (the cycle-0 thermal-cold outlier inflates math σ; cycles 1-4 alone are σ < 0.4%). Bench protocol passed; the result is statistically valid.

## Root cause (per Chesterton's fence — read the code, not the marketing)

### Why does the explainer regress to 0.46×?

Per-step wall time:
- Math: 256 t / (149.3 t/s) ÷ 36.7 steps = **46.6 ms/step** (6.98 t/step yield)
- Explainer: 256 t / (58.2 t/s) ÷ 78 steps = **56.4 ms/step** (3.28 t/step yield)
- Code: 256 t / (128.25 t/s) ÷ 42.5 steps = **47.0 ms/step** (6.02 t/step yield)

Explainer step time is 10ms longer despite fewer accepted tokens per step. The delta tracks reject count: math rejects 8 positions/step, explainer rejects 12.7. At ~70µs per layer-slice × 30 layers × extra rejects ≈ 10ms.

### The smoking gun: `/opt/dflash/dflash/model_mlx.py:_trim_recent_cache:243-258`

```python
if isinstance(c, RotatingKVCache) and c.keys is not None:
    c.keys = c._temporal_order(c.keys)       # rebuild buffer in time order
    c.values = c._temporal_order(c.values)
    c.keys = c.keys[..., :-n, :]             # slice off last n positions
    c.values = c.values[..., :-n, :]         # ditto
    c.offset -= n
    c._idx = c.keys.shape[2]                 # reset cursor to new buffer size
```

This is **buffer-resize semantics**. Every trim allocates two new tensors (temporal_order + slice). For gemma-4-26b's mostly-sliding-attention layers (`layer_types: [sliding_attention × 4, full_attention, ...]`), most layers go through this expensive path on every spec-decode step.

Compare to mlx_lm.RotatingKVCache.trim at `cache.py:545` (cursor-only):
```python
def trim(self, n):
    n = min(self.offset, n)
    self.offset -= n
    self._idx -= n
    return n
```

The dflash MLX choice is **not a bug** — it's required by their convention that `c.keys.shape[2] == live_data_length`. Their attention code reads `keys.shape[2]` and concatenates additional positions; cursor-mode would break this.

### What this means for the Rust port (hf2q)

hf2q's `rollback_kv` (forward_mlx.rs:5733) IS cursor-mode by design. Attention kernels read `< seq_len`, writes go to `write_pos`. Rolling back decrements both cursors — no slicing, no allocation.

Conservative projection of Rust port speedup (trim cost eliminated, other Python overhead retained):
- Math: 1.187× → **~1.55×** (subtract ~10ms trim, retain other overhead)
- Explainer: 0.461× → **~1.04×** (subtract ~25ms trim)
- Code: 1.014× → **~1.35×**
- **Projected Rust mean: ~1.31× — still below 1.6× gate**

Even more optimistic projection (also eliminate Python interpreter loop, mx.async_eval scheduling overhead):
- Math: → 1.7-1.9×
- Explainer: → 1.1-1.3×
- Code: → 1.5-1.7×
- **Optimistic Rust mean: ~1.5× — at or just below gate**

## Open question this leaves: is block_size=16 the wrong K?

The article reporting 2.56× on gemma-4-26B used **13 spec tokens (K=12, block_size=13)** on RTX 5090 vLLM. We used K=15 (block_size=16). Smaller K means less reject-waste per step.

Hypothesis **H_BLOCK_SIZE** (to be tested by `scripts/adr030/phase1_blocksize_sweep.py`):
- At K=12 (block_size=13), explainer speedup recovers ≥ 20% (0.46× → ≥0.55×)
- At K=8 (block_size=9), further recovery or plateau

If H_BLOCK_SIZE is validated: block_size is a meaningful lever; we can revise the K parameter in the Rust port to match optimal-K on M5 Max. If overall mean Python speedup with optimal K clears 1.2×+, the Rust port projection (with cursor-mode trim) likely clears 1.6×.

If H_BLOCK_SIZE is falsified: block_size is not the lever; the structural ceiling is what we measured. Decision to operator: accept 1.31× Rust projection (mantra-orthogonal but below ideal gate), or abandon DFlash for gemma-4.

## Decision pending

This document closes the **block_size=16 leg** of Phase 1. The **block_size sweep leg** is running as of this commit; results in `docs/research/ADR-030-phase1-blocksize-sweep.json` upon completion (~70 min). Final go/no-go decision after sweep data is in.

## Raw data

`docs/research/ADR-030-phase1-m5max-results.json` — full 30-measurement run with per-cycle stats.
