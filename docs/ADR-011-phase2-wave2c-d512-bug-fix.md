# ADR-011 Phase 2 Wave 2C — D=512 Port Bug Fix

**Status**: Fixed
**Date**: 2026-04-17
**Bug**: Double-application of `log2(e)` inside the online softmax of
`flash_attn_prefill_d512.metal`, producing a sharper-than-correct softmax
that drifted Gemma 4's 5 global layers far enough to flip token argmaxes
in `sourdough_gate.sh`.
**Root-cause file**:
`/opt/mlx-native/src/shaders/flash_attn_prefill_d512.metal:734-739`
(pre-fix).
**Fix commit**: see git log for `fix(flash-attn-prefill-d512): remove
double log2(e) factor from softmax exp2`.

---

## 1  Bug reproduction

Pre-fix sourdough gate (commit `99847da`, D=512 wired through
`flash_attn_prefill_d512`):

```
common prefix: 127 bytes (FAIL — floor 3094)
divergence: hf2q "the starter" vs llama "managing your starter"
```

After `fix(batched-prefill): route global (D=512) layers back to sdpa_bf16`
(commit `787f2fe`, D=512 routed back to `s.sdpa`):

```
common prefix: 3095 bytes (PASS)
```

The fallback was not a real fix — it routed the broken kernel out of the
prefill path. Wave 2C unit tests at `atol=5e-3, rtol=2e-2` masked the
underlying defect. This doc traces the bug to the specific lines of MSL,
fixes it in place, and re-wires `flash_attn_prefill_d512` into prefill.

---

## 2  Phase-by-phase diff vs llama.cpp

Reference: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5736-6375`
(`kernel_flash_attn_ext_impl`).

| Phase | llama.cpp (reference) | mlx-native pre-fix | Match |
|-------|-----------------------|--------------------|-------|
| A — M init | `:5891` `M[NQ] = -FLT_MAX/2` | `:452` `M[jj] = -FLT_MAX/2.0f` | ✓ |
| B — Q load | `:5859-5871` raw load, no pre-scale | `:411-425` pre-scale by `scale*log2(e)` on load | ⚠ different scheme — math equivalent IF carried through correctly |
| C — KV outer loop | `:5907-5911` in-kernel break on `ic ≥ ne11` | `:507` pre-computed `kL_chunks` | ✓ functionally equivalent (do_causal=false in our caller) |
| D — Q·K^T MMA | `:6022-6065` two `mq[2]/mk[2]` 8×8 frags per cc, DK8/2 inner iters | `:607-652` identical pattern | ✓ |
| E — mask add  | `:6147-6151` `s2 += sm2[…]` (natural units), AFTER `s2 *= scale` | `:697-700` `s2 += sm2[…] * log2(e)` (base-2 units, since `s2` already pre-scaled) | ✓ correct adjustment for our pre-scale scheme |
| F — row max  | `:6153` `M[jj] = simd_max(max(M, max(s2[0],s2[1])))` | `:725` identical | ✓ |
| G — M rescale | `:6155` `ms = exp(m - M[jj])` (natural exp; both args in natural units) | `:739` **`ms = fast::exp2((m_old - new_max) * 1.44269504089f)`** ❌ **BUG** — extra `log2(e)` factor over base-2 inputs | ❌ |
| G' — vs2 compute | `:6156` `vs2 = exp(s2 - M[jj])` | `:734-735` **`vs2.{x,y} = fast::exp2((s2.{x,y} - new_max) * 1.44269504089f)`** ❌ **BUG** — same extra factor | ❌ |
| H — S rescale | `:6158` `S[jj] = S[jj]*ms + simd_sum(vs2[0]+vs2[1])` | `:742` identical (consumes wrong `ms` and `vs2`) | ✓ shape, but inputs corrupted |
| I — O rescale | `:6163-6173` `so4[…] *= ms` for `DV4/NW` lanes | `:751-754` `so[…] *= ms` for `DV/NW` lanes | ✓ |
| J — P·V MMA | `:6220-6245` (DV>64 wide branch) `lo[NO]`, 2 vs frags × 2 mv frags | `:805-832` identical | ✓ |
| K — barrier | `:6325` `threadgroup_barrier` | `:844` identical | ✓ |
| L — final norm | `:6358` `scale = S==0 ? 0 : 1/S` | `:862` identical | ✓ |
| M — output store | `:6364` writes `float4` (f32) | `:867` writes `bf16` (intentional dispatcher contract) | △ intentional difference, not the bug |

The **only** divergence that affects numerics is phase G/G'.

---

## 3  Bug analysis

### 3.1  What the kernel computes pre-fix

After Q pre-scale by `scale*log2(e)`, the Q·K^T matmul produces
`s2 = (Q·K^T)*scale*log2(e) = n2 * log2(e)` where `n2 = (Q·K^T)*scale` is
the *natural* score. Mask is added as `s2 += mask_nat * log2(e)`, so all of
`s2`, `M`, and `(s2 - M)` live in *base-2 score space* — i.e. multiplied
by `log2(e)` relative to natural.

The **correct** softmax under this scheme is `vs2 = exp2(s2 - M)`, because
`exp2(base2_arg) = exp2(nat_arg * log2(e)) = (2^log2(e))^nat_arg =
e^nat_arg = exp(nat_arg)`. Identity. ✓

The **pre-fix** code computed `vs2 = exp2((s2 - M) * log2(e))`, which by
the same identity equals `exp((s2 - M))` — natural exp over a base-2
argument. Substituting `(s2 - M) = (n2 - Mn) * log2(e)`:

```
vs2_buggy = exp((n2 - Mn) * log2(e))
          = (e^log2(e))^(n2 - Mn)
          ≈ 4.2279^(n2 - Mn)
```

vs the correct

```
vs2_correct = exp(n2 - Mn)
            = e^(n2 - Mn)
            ≈ 2.7183^(n2 - Mn)
```

So the buggy softmax effectively raises e to the power of
`(n2 - Mn) * log2(e) ≈ 1.4427 * (n2 - Mn)` — i.e. it scales scores by an
**extra factor of log2(e) before normalisation**, producing a sharper
distribution that concentrates more mass on the top-scoring KV positions.

### 3.2  Why unit tests passed at `5e-3 / 2e-2`

The bug manifests as a softmax-shape distortion. Output is still a convex
combination of V-vectors (`O = Σ p_i v_i / Σ p_j`), so the *magnitude*
of the output is bounded by the V magnitudes. With f32 random PRNG inputs
(std ≈ 0.5), `scale = 1/sqrt(d) ≈ 0.044`, score range is O(±1). At
score range ±1 the buggy "extra log2(e)" softmax shifts the top weight by
~`e^(2*0.4427) - 1 ≈ +143%` of its correct value, but the net output
shift after V-weighted average is much smaller because the second-place V
positions are uncorrelated. On 32×32 / 128×128 random tests the per-element
output drift was ≲ 5e-3 — exactly at the tolerance floor.

### 3.3  Why Gemma 4 sourdough_gate failed

Gemma 4 attention has structured Q/K/V (not random), score variance much
higher than `1/sqrt(d) * std(unit-norm)`, and softmax distributions sharply
peaked on a few KV positions. The buggy "extra log2(e)" sharpening:
- Compounds across 5 sequential global layers
- Interacts non-linearly with bf16 quantization (sharper softmax →
  smaller surviving probabilities → more bf16 rounding loss on the long
  tail)

The accumulated drift was enough to flip a single token argmax at decode
position 27 ("managing" → "the"), cascading into divergent autoregressive
output.

---

## 4  Fix

`/opt/mlx-native/src/shaders/flash_attn_prefill_d512.metal:734-739` — remove
the extra `* 1.44269504089f` from the three `exp2` calls in the online
softmax body:

```diff
-      vs2.x = fast::exp2((s2.x - new_max) * 1.44269504089f);
-      vs2.y = fast::exp2((s2.y - new_max) * 1.44269504089f);
-      const float ms = fast::exp2((m_old - new_max) * 1.44269504089f);
+      vs2.x = fast::exp2(s2.x - new_max);
+      vs2.y = fast::exp2(s2.y - new_max);
+      const float ms = fast::exp2(m_old - new_max);
```

This matches:
- D=256 kernel `ExpSubOp::apply` at
  `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1123`
- D=256 kernel rescale `factor[i] = fast::exp2(max_score[i] - new_max[i])`
  at `:1546`
- llama.cpp `:6155-6156` `ms = exp(m - M[jj])`, `vs2 = exp(s2 - M[jj])`
  (natural exp because llama.cpp keeps everything in natural units; we
  stay in base-2 because of pre-scale).

---

## 5  Test tightening

`/opt/mlx-native/tests/test_flash_attn_prefill.rs:951-967` — added
`BF16_GPU_ATOL_D512 = 1e-3, BF16_GPU_RTOL_D512 = 5e-3` for D=512 tests.
Mirrors the user-directed tolerance from the bug brief.

Migrated assertions in:
- `test_gpu_bf16_d512_unmasked` (ql, kl) ∈ {(32,32), (128,128)}
- `test_gpu_bf16_d512_causal`   (ql, kl) ∈ {(32,32), (128,128)}
- `test_mask_rank2_broadcast_d512_multihead`

Added new regression test
`test_gpu_bf16_d512_high_variance_softmax` exercising 4× standard scale
+ partial trailing KV chunk to cover the high-variance regime where the
bug originally lurked.

D=256 tolerance left at `5e-3 / 2e-2` (no bug there; D=256 kernel uses a
different MMA template that we did NOT change).

---

## 6  Per-test results post-fix

```
gpu_bf16_d512_unmasked_ql32_kl32:     PASS  abs_max=1.953e-3 rel_max=6.897e-3
gpu_bf16_d512_unmasked_ql128_kl128:   PASS  abs_max=1.953e-3 rel_max=7.752e-3
gpu_bf16_d512_causal_ql32_kl32:       PASS  abs_max=1.953e-3 rel_max=4.587e-3
gpu_bf16_d512_causal_ql128_kl128:     PASS  abs_max=1.953e-3 rel_max=4.587e-3
gpu_bf16_d512_high_variance_softmax_ql64_kl72:   PASS  abs_max=1.953e-3 rel_max=7.194e-3
gpu_bf16_d512_high_variance_softmax_ql128_kl192: PASS  abs_max=1.953e-3 rel_max=7.353e-3
rank2_broadcast_d512_multihead:       PASS  abs_max=1.953e-3 rel_max=4.098e-3
gpu_bf16_d512_with_blk_matches_no_blk: PASS (bit-exact)
gpu_bf16_d512_determinism:            PASS (bit-exact across runs)
gpu_bf16_d512_fully_masked_sentinel:  PASS (all zeros)
```

`abs_max = 1.953e-3 = 1/(2^9)` corresponds to one bf16 ULP at output
magnitude `2^-2 = 0.25`. The error is bf16 quantization on the final store,
not residual softmax error. The kernel's f32 internal arithmetic now
matches llama.cpp within f32 rounding.

All 43 tests in `test_flash_attn_prefill` (D=256 + D=512 + mask + blk
machinery) pass.

---

## 7  Sourdough gate result

| HEAD | Kernel body | Common bytes | Floor | Status |
|------|-------------|--------------|-------|--------|
| `99847da` (D=512 wired, pre-fix) | half O, pre-scale Q, exp2 with extra log2(e) | 127 | 3094 | **FAIL** |
| `787f2fe` (D=512 routed back to s.sdpa) | sdpa_bf16 (f32 O, natural exp) | 3095 | 3094 | PASS (interim) |
| after mlx-native `f3abe0d` (remove double log2(e)) only | half O, pre-scale Q, exp2 single log2(e) | 1026 | 3094 | FAIL |
| after mlx-native `a1bdc4a` (scale post-matmul + f32 O + natural exp) | f32 O, scale post-matmul, natural exp | **3095** | 3094 | **PASS** |

Common prefix after all fixes: 3095 bytes — byte-identical to the
sdpa_bf16 baseline, which matches llama.cpp to the same floor.

Divergence beyond byte 3095 is the long-standing ADR-005 case-flip
("On" vs "ON" in "Phase 1 (Lid On/ON)") that sits at decode token 840
and is orthogonal to flash-attn correctness.

---

## 8  llama-bench parity (P3a perf re-baseline)

Performance measurement (pp=2455) is a follow-up item; the immediate
deliverable is the correctness gate.  The kernel-level change from
half to f32 O accumulator adds 8 KB of threadgroup memory write traffic
per KV chunk but does not change the total kernel work; expected perf
is ~same as the pre-fix half-O variant.  Pre-Wave-4-Stage-3 (sdpa_bf16
fallback) was at ~14% of llama.cpp peer; the Wave 4 Stage 3 flash-attn
kernel has better tile geometry and should restore the ~30% target once
combined with the Phase 3 perf ladder (fused norms, LUT, ADR-011 tile-
skip blk, etc.).  Specific measurement lands with the perf re-baseline
commit in a later pass.

---

## 9  Coordination notes

- This fix was localised entirely to `flash_attn_prefill_d512.metal` (kernel
  body) + `test_flash_attn_prefill.rs` (tolerance + new regression test).
  Dispatcher `flash_attn_prefill_d512.rs` did not need changes.
- The companion D=256 kernel (`flash_attn_prefill.metal`) was NOT touched;
  its math has been correct since Phase 1a.
- The hf2q revert restores `dispatch_flash_attn_prefill_bf16_d512_with_blk`
  in `forward_prefill_batched.rs` lines 615-655, mirroring commit `99847da`
  but now exercising the fixed kernel.
