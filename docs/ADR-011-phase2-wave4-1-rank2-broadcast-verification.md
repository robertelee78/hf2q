# ADR-011 Phase 2 — Wave 4.1: Rank-2 Broadcast Mask Verification

**Date:** 2026-04-17
**Wave:** 4.1 (rank2-broadcast-fix)
**Swarm:** swarm-1776516482254-ft5mwj

## Summary

Wave 4 identified a blocker: both the D=256 and D=512 flash-attention prefill
dispatchers hardcoded rank-4 `[B, H, qL, kL]` mask layout in their buffer-size
validation and stride calculations.  Wave 2D's `build_sdpa_mask_bf16` produces
a rank-2 `[qL, kL]` single-plane broadcast mask (llama.cpp convention).  The
rank-4 validation rejected the smaller rank-2 buffer, and the stride formulas
would have indexed past the single plane for any head > 0.

Wave 4.1 patches both dispatchers to accept rank-2 broadcast masks while
preserving the rank-4 per-head path for back-compat.

## API Shape Change

Both dispatchers now inspect `mask.shape().len()` at dispatch time:

```rust
let mask_is_rank2_broadcast = mask.is_some_and(|m| m.shape().len() == 2);
```

**Rank-2 path** (`shape = [qL, kL]`, Wave 2D output):
- Validates `mask` element count as `ql * kl`.
- Sets `m_batch_stride = 0`, `m_head_stride = 0`, `m_ql_stride = kl`.
- The Metal shader reuses the same plane for every (batch, head) pair via
  stride-0 addressing — no kernel changes required.

**Rank-4 path** (`shape = [B, H, qL, kL]` or any rank != 2, back-compat):
- Validates `mask` element count as `batch * h * ql * kl`.
- Sets the original per-head strides: `m_batch_stride = h*qL*kL`,
  `m_head_stride = qL*kL`, `m_ql_stride = kL`.

## Files Changed

| File | Change |
|------|--------|
| `/opt/mlx-native/src/ops/flash_attn_prefill.rs` | D=256 dispatcher: rank-2 branch in buffer validate + m_strides |
| `/opt/mlx-native/src/ops/flash_attn_prefill_d512.rs` | D=512 dispatcher: same (applied in `_with_nsg_and_blk`, the single implementation all variants delegate to) |
| `/opt/mlx-native/tests/test_flash_attn_prefill.rs` | 3 new Wave 4.1 tests in § 9 |
| `/opt/mlx-native/Cargo.toml` | Version already at 0.3.2 (bumped in an earlier wave) |

## Call Graph — Fix Location

```
dispatch_flash_attn_prefill_bf16_d256          →  delegates to:
dispatch_flash_attn_prefill_bf16_d256_with_blk    ← PATCHED HERE

dispatch_flash_attn_prefill_bf16_d512          →  delegates to:
dispatch_flash_attn_prefill_bf16_d512_with_blk →  delegates to:
dispatch_flash_attn_prefill_bf16_d512_with_nsg_and_blk  ← PATCHED HERE
```

The fix is applied at the narrowest point that covers all entry points.

## Test Results

```
42 tests — 42 passed, 0 failed (finished in 24.96s)
```

New Wave 4.1 tests:

| Test | Shape | Dispatcher | Result |
|------|-------|------------|--------|
| `test_mask_rank2_broadcast_d256_multihead` | batch=1, h=8, ql=kl=128, D=256, rank-2 causal+SWA mask | `d256` | PASS |
| `test_mask_rank2_broadcast_d512_multihead` | batch=1, h=8, ql=kl=128, D=512, rank-2 causal+SWA mask | `d512` | PASS |
| `test_mask_rank4_preserved_regression` | batch=1, h=4, ql=kl=128, D=256, rank-4 checkerboard mask | `d256` | PASS |

Tolerance: `atol=5e-3, rtol=2e-2` (same as all existing bf16 GPU tests).

All existing 39 tests continue to pass — no regressions.

## Version Note

`mlx-native` version is `0.3.2`.  `hf2q` accesses this via the
`.cargo/config.toml` path override (`patch.crates-io`), so the updated crate
is visible to hf2q immediately without a crates.io publish.

## No-Metal-Change Rationale

The Metal shader (`flash_attn_prefill.metal`) already computes the mask offset
as:

```metal
mask_ptr + batch_idx * m_batch_stride + head_idx * m_head_stride + q_pos * m_ql_stride + k_pos
```

Setting `m_batch_stride = 0` and `m_head_stride = 0` causes every (batch, head)
to address position `q_pos * kL + k_pos` within the single `[qL, kL]` plane —
correct broadcast behaviour with zero kernel changes.
