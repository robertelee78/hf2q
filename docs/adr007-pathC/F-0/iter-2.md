# ADR-007 Path C / F-0 / iter-2 — CPU HB encoder mirror

**Date:** 2026-05-05
**Mantra discipline:** read GPU encoder shader before mirroring; never trust comments. Two F-0 findings logged below.

## What landed

mlx-native commit (TBD — about to commit) on main:

- `src/turboquant.rs` — new public-API additions:
  - `pub const TBQ_SIGNS_256: [u8; 32]` and `TBQ_SIGNS_512: [u8; 64]`
    lifted verbatim from `hadamard_quantize_kv_fast.metal:25-30,35-44`
    (AmesianX `cpy-utils.cuh` with documented sha256 sources).
  - `pub fn apply_d1_sign_mask_inplace` — bit-by-bit SRHT sign flip
    (LSB-first within byte; bit=1 → ×-1, bit=0 → ×+1).
  - `pub fn turboquant_hb_encode_d256(x, bits) -> (Vec<u8>, f32)` —
    byte-equivalent CPU mirror of
    `hadamard_quantize_kv_fast.metal::hadamard_quantize_kv_hb<256>`:
    1. D1 sign pre-multiplication via `TBQ_SIGNS_256`
    2. Normalized FWHT (`fwht_inplace`)
    3. L2 norm extraction
    4. Scale `(1/norm) * sqrt(d)` (lift to N(0,1)); norm≤1e-10 → scale=0
    5. Per-element nearest-centroid (5/6/8-bit codebook)

- 7 new unit tests (all PASS in 0.00s release):
  - `hb_encoder_d256_roundtrip_8bit_meets_gate_a`: synthetic Gaussian
    encode → kernel-formula decode → cosine ≥ 0.998, NRMSE ≤ 0.07.
    Clears strict Gate A (≥ 0.999 mean, close-section measured 0.9998).
  - `hb_encoder_d256_roundtrip_5bit_within_band`: 5-bit cosine ≥ 0.985.
    Confirms expected wider gap; matches Lloyd-Max 5-bit floor.
  - `hb_encoder_d256_is_deterministic`: bit-identical bytes + norm.
  - `hb_encoder_d256_zero_vector`: norm=0 path → packed bytes are
    centroid-closest-to-zero (idx 127 or 128 for 8-bit), decode → all zeros.
  - `hb_encoder_d256_validates_bits`: rejects 4 (not HB) and 7 (invalid).
  - `hb_encoder_d256_validates_size`: rejects head_dim ≠ 256.
  - `d1_sign_mask_is_self_inverse`: 2× application = identity.
  - `tbq_signs_first_32_bytes_match_512_prefix`: load-bearing identity
    of the two sign tables' shared prefix.

## F-0 findings (logged — to be resolved in subsequent iters)

### F-0 finding #1: D1 sign mask was missing from existing CPU codec

The pre-existing `turboquant_quantize` function in `turboquant.rs` (shipped
since iter-15ish) does NOT apply the D1 sign mask before FWHT. The GPU
encoder kernel `hadamard_quantize_kv_hb` DOES apply it (line 1b at
`hadamard_quantize_kv_fast.metal:572-583`). So the existing
`turboquant_quantize` was never byte-equivalent to the GPU HB path.

This is consistent with the close-section's iter-14 D1 SRHT landing
(`§1126 ✅ D1 SRHT sign tables (iter 14)`) on the GPU side — the CPU
side never followed, but no test ever cross-checked CPU vs GPU bytes,
so the drift was invisible.

The new `turboquant_hb_encode_d256` is the first CPU HB encoder that
matches the GPU math. The original `turboquant_quantize` (which only
supports 2/3/4-bit anyway) is left unchanged for back-compat with any
existing callers.

### F-0 finding #2: D=512 per-block norm computation may be buggy

Reading `hadamard_quantize_kv_fast.metal:599-609`:

```metal
} else {
    float blk0_sq = (lane < 16u) ? simd_sum(local_sq_sum) : 0.0f;
    float blk1_sq = (lane >= 16u) ? simd_sum(local_sq_sum) : 0.0f;
    blk0_sq = simd_broadcast(blk0_sq, 0u);
    blk1_sq = simd_broadcast(blk1_sq, 16u);
    norm0 = sqrt(blk0_sq / 256.0f);
    norm1 = sqrt(blk1_sq / 256.0f);
}
```

`simd_sum(x)` per Metal Shading Language Spec is uniform across the
simdgroup — every lane gets the SAME result, which is the full 32-lane
sum (= ||entire 512-vec||²). The conditional `(lane < 16u) ? ... : 0.0`
masks the assignment but doesn't change what `simd_sum` computes. Then
`simd_broadcast(blk0_sq, 0u)` reads lane 0's value (= ||full||²) and
broadcasts to all lanes; same for `blk1_sq` from lane 16 (also = ||full||²
because lane 16 also called the uniform `simd_sum`).

Result: `norm0 == norm1 == sqrt(||full||²/256) == ||full||/16` — they
are NOT per-block norms. This contradicts the comment block at
`hadamard_quantize_kv.rs:11-13` and AmesianX `cpy-utils.cuh:241-269`
intent.

**Hypothesis status:** unverified. Will be tested in iter-3 by writing
the same kernel call from Rust, dumping the norms array, and comparing
to a CPU computation that does both:
- Variant A: per-block norms (||elements 0..256||, ||elements 256..512||)
- Variant B: bug-mirror (both = ||full||/16)

Whichever the GPU produces tells us the truth.

If the bug is confirmed, this is a load-bearing finding for ADR-007:
the "D=512 per-256-block norm layout (iter 15)" close-section
achievement (§1130) may not actually be per-block in production. Affects
Gate C measurement on D=512 paths (Gemma 4 vision, Qwen3.5 GLM).

**Mitigation for iter-2 scope:** D=256 is the production Gemma 4 26B
text path. D=256 norm is correctly `||full||` (line 596). Iter-2's CPU
encoder for D=256 is unaffected by the suspected D=512 bug. The
iter-2 deliverable proceeds as planned.

## What iter-2 is NOT

- No GPU vs CPU encoder byte comparison yet. Roundtrip is via the
  CPU oracle decoder only. **A passing roundtrip proves the CPU
  encoder + CPU oracle are mutually consistent — it does NOT prove
  the CPU encoder bytes match what the GPU kernel writes.** That gate
  is iter-3's harness (Metal device init, dispatch the encoder kernel,
  read back norms + packed buffer, byte-diff vs CPU encoder).

- No D=512 encoder yet. Hypothesis F-0 finding #2 needs to be verified
  before authoring a D=512 mirror; otherwise iter-3 mirrors the wrong
  math.

- No production-shape full-attention divergence audit yet. That is the
  full F-0.2 deliverable (per ADR §F-0.2: 62 layers × 4 KV heads ×
  head_dim=256 × kv_capacity=8192). Iter-3 will start that harness.

## Iter-3 plan

1. **Add Metal-dependent integration test** at
   `mlx-native/tests/test_tq_encoder_byte_parity.rs` that:
   - Initializes `MlxDevice`.
   - Synthesizes Gaussian K-row of length 256 with deterministic seed.
   - Encodes via CPU `turboquant_hb_encode_d256` → (cpu_packed, cpu_norm).
   - Encodes via GPU `dispatch_hadamard_quantize_kv_hb` → (gpu_packed, gpu_norm).
   - Asserts byte-equality on packed buffer + |cpu_norm - gpu_norm| < 1e-4.
   - Falsifier: any byte mismatch → CPU encoder has drift; localize.
   - Repeat for 5/6/8-bit.

2. **Test F-0 finding #2 (D=512 norm bug)** with a single-vector test
   that reads both `norm0` and `norm1` from the GPU kernel and compares
   to:
   - `||first 256||` (correct per-block)
   - `||full||/16` (suspected bug)
   - Whichever matches, that's the production behavior.

3. **Production-shape SDPA divergence audit** — iter-4 territory; iter-3
   is encoder-side validation.
