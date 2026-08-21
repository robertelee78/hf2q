# ADR-007 Path C / F-0 / iter-1 — F-0.1 oracle landed

**Date:** 2026-05-05
**Mantra discipline:** read kernel + shader before authoring oracle. Comments not load-bearing.

## What landed

mlx-native commit `52c87ff` on main:

- `src/turboquant.rs` — added `pub const CODEBOOK_HB_5BIT: [f32; 32]`,
  `CODEBOOK_HB_6BIT: [f32; 64]`, `CODEBOOK_HB_8BIT: [f32; 256]` mirroring
  `flash_attn_vec_tq_hb.metal:52-156` exactly. Added `hb_centroid` +
  `hb_nearest_centroid` helpers (encode-side reuse).
- `src/tq_oracle.rs` — new module, `flash_attn_vec_tq_hb_oracle` function
  + `TqHbOracleParams` struct. Pure F32 SDPA decode mirroring the kernel
  math byte-for-byte:
  - D=256: `value = codebook[byte] * (norm * inv_sqrt(DK))`
  - D=512: per-block norms, `value = codebook[byte] * (norm[block] / scale_factor_d512)`
  - mask: `logical_idx = (kv_pos - ring_start + cap) % cap; valid iff
    logical_idx ∈ [window_start_logical, kv_seq_len)`
  - GQA: `kv_head = q_head / heads_per_kv`
  - softmax: stable max-subtraction (offline, equivalent to kernel's online form)
  - all-masked → output zeros (mirrors kernel inv_S=0 path)
  - F32 throughout, deterministic serial reduction → bit-identical across runs

Mantra finding (logged for ADR closure): `softcap` is in `FlashAttnVecTqHbParams`
struct (kernel and oracle) but is **never read** by the kernel body
(`flash_attn_vec_tq_hb.metal:37` declares it; no reference in lines 215-487).
Contractual drift vs the dense `flash_attn_vec.metal` where softcap is also
declared but unimplemented. The oracle preserves the field for byte-compat with
the kernel ABI but does not apply softcap. Not a bug per se but worth fixing
when ADR-007 close gets cleaned up — either remove the field or implement.

## Tests (10 unit, all PASS in 0.00s release)

```
test tq_oracle::tests::codebooks_match_metal_shader_constants ... ok
test tq_oracle::tests::hb_centroid_unsupported_bits_returns_zero ... ok
test tq_oracle::tests::hb_centroid_lookup_matches_index ... ok
test tq_oracle::tests::oracle_all_masked_returns_zeros ... ok
test tq_oracle::tests::nearest_centroid_finds_closest ... ok
test tq_oracle::tests::oracle_is_bit_deterministic ... ok
test tq_oracle::tests::oracle_d512_per_block_norms ... ok
test tq_oracle::tests::oracle_gqa_routes_heads_to_correct_kv_head ... ok
test tq_oracle::tests::oracle_single_position_uniform_v_matches_manual ... ok
test tq_oracle::tests::oracle_sliding_window_masks_old_positions ... ok
```

The unit tests exercise:

- All three codebooks (5/6/8-bit) byte-identical to the Metal constants.
- 8-bit codebook symmetry (declared 3.41e-10 in the shader; verified <1e-5).
- Centroid lookup matches the kernel's `dequant_hb_single` indexing
  (`& 0x1F` for 5-bit, `& 0x3F` for 6-bit, full byte for 8-bit).
- No-panic guarantee for invalid bit-widths (returns 0, doesn't crash).
- Saturation behavior on values outside ±5.07.
- Hand-verified single-position attention output equals
  `V_dequant * 1.0` (softmax weight = 1 with single valid position).
- Bit-determinism: two runs with same inputs produce
  `assert_eq!(out_a[i].to_bits(), out_b[i].to_bits())`.
- Sliding-window mask actually masks (output differs from no-mask version
  by max_diff > 1e-3).
- All-masked produces all-zeros output.
- D=512 per-block dequant: block 0 uses `norm[0] / sf_d512`, block 1 uses
  `norm[1] / sf_d512`.
- GQA routing: heads 0..3 → kv_head 0; heads 4..7 → kv_head 1, with
  10× v_norm scale on kv_head 1 reflected in the output ratio.

## What F-0.1 is NOT (gaps logged for follow-up)

- No production-shape smoke test (Gemma 4 26B: 32 heads, 4 kv_heads,
  head_dim=256, kv_capacity=8192). Unit tests run at smaller shapes.
  Production-shape audit harness is F-0.2's deliverable; the oracle has
  no shape-specific assumptions, so no oracle change is expected.
- No comparison vs the actual GPU kernel yet. F-0.2 is the harness that
  feeds identical inputs through (a) GPU kernel, (b) CPU oracle,
  (c) dense F32 path, and computes per-layer NRMSE + max-abs-diff.
- No CPU-side HB encoder yet. The unit tests use a small encode helper
  (`encode_row_d256`) that mirrors what the
  `hadamard_quantize_kv_hb_d256` kernel produces. F-0.2 needs a
  full-fledged CPU encoder to feed identical inputs through both paths.

## Falsifier F-0 status

Not yet exercised — F-0 falsifier requires F-0.2 layer-by-layer audit
which depends on F-0.1 oracle (now landed). Status: **unblocked**.

## Iter-2 plan

- Move task #1 → completed.
- Move task #2 (F-0.2) → in_progress.
- Build divergence audit harness:
  1. CPU-side HB encoder mirroring `hadamard_quantize_kv_hb_d{256,512}.metal`.
  2. Test fixture: synthetic Gaussian K/V at production shape.
  3. Run identical inputs through (a) `flash_attn_vec_tq_hb` GPU kernel
     and (b) `flash_attn_vec_tq_hb_oracle` CPU. Compute NRMSE + max-abs-diff.
  4. Repeat at 5/6/8-bit; report per-bit-width divergence.
- F-0 falsifier check: any cell with `NRMSE > 0.15` → STOP and localize.
