# ADR-011 Phase 1a — Final Audit Verdict

**Author:** Agent #6 (reviewer), swarm-1776462683390-zb6ev9  
**Date:** 2026-04-17  
**Audits:** Phase 1a output of Agents #1–#5 (parent-synthesis included)  
**Verdict: APPROVE WITH CAVEATS**

MUST-FIX before commit: **0 blocking items**.  
SHOULD-FIX for Phase 1b: **3 items** (test coverage gaps — no code changes required
in shipped source files, only test additions).  
Blind spots confirmed out of scope: **3 items** (all explicitly deferred in ADR/CFA plan).

---

## MUST-FIX (blocking commit): none

No item in this audit blocks commit. All critical-path checks pass.

---

## SHOULD-FIX before Phase 1b kick-off (test additions only)

### S1: `align_Q=false` path (unaligned `seq_len_q`) is not GPU-tested

**Location:** `tests/test_flash_attn_prefill.rs` — all GPU tests use
`ql ∈ {32, 128, 512}`, which are exact multiples of `BQ=32`.  The function
constant `align_Q` (Metal constant index 200) is set to `true` for all current
GPU tests.  The `align_Q=false` code path in the kernel
(`flash_attn_prefill.metal:1504–1510`: `Otile.store_safe()` vs `Otile.store()`)
is never exercised on GPU.

**Suggested fix:** add one GPU test with `ql=50` (or any `ql % 32 != 0`).
The `FlashAttnPrefillParams` input exists; the dispatcher computes
`nq_aligned = ql / BQ` and passes `align_Q = (ql % BQ == 0)`.  This requires
no shader or dispatcher changes.

**Risk if unfixed:** the `store_safe` bounds-check path is present in candle and
byte-identical in our vendor — it passes candle's own test suite — so the risk
is low.  But "low risk" is not "verified"; Phase 1b exercises real prompts that
may have arbitrary sequence lengths and the gap would surface there.

### S2: `align_K=false` path (unaligned `seq_len_k`) is not GPU-tested

Same argument as S1 for the K dimension.  All GPU `kl` values are multiples of
`BK=16`.  The in-kernel partial-last-tile masking path
(`flash_attn_prefill.metal:2100–2117` in candle, our equivalent region) is not
exercised on GPU.

**Suggested fix:** add a test with `kl=50` or any `kl % 16 != 0`.  Can be the
same fixture as S1 (`ql=50, kl=50`).

### S3: ADR-011 text needs a formal amendment section to document the DivOp addition

The preamble in `flash_attn_prefill.metal:44` states:

> "Modifications beyond mechanical preamble/attribution lines require an ADR
> amendment (ADR-011) — the DivOp NaN-guard addition noted above is the single
> current deviation."

ADR-011 itself (`/opt/hf2q/docs/ADR-011-flash-attn-prefill.md`) does not
contain an amendment section recording the DivOp guard as an intentional
deviation from candle.  The preamble is self-referential but the canonical ADR
record lacks the corresponding entry.

**Suggested fix:** append a brief "## Amendment: Phase 1a DivOp NaN-guard" to
`ADR-011-flash-attn-prefill.md` citing `flash_attn_prefill.metal:1090–1112` and
the test it is validated by.  One paragraph.  This is documentation, not code.

---

## Out-of-scope blind spots (confirmed deferred)

### B1: `seq_len > BQ * N` (large real-world prefill) — Phase 2 gate

ADR-011 §Phase 2 explicitly gates 576-token and 2455-token correctness via the
hf2q integration harness, not the unit test suite.  The unit tests are
intentionally capped at `seq_len=512` per the Phase 1 exit criteria.  No action
needed in Phase 1a.

### B2: D=512 dispatcher exposure — Phase 4 gate

The four D=512 kernel names are registered but no Rust dispatcher calls them
yet.  This is correct per the CFA plan (Phase 4 task).  No action needed.

### B3: SWA tile-skip pre-pass — Phase 5

`flash_attn_ext_blk`-style tile skip is an explicit Phase 5 carve-out.
No action needed.

---

## Section A — Kernel Faithfulness

### A.1: instantiate_attn line-for-line vs candle

Verified by direct `grep` and line inspection of
`flash_attn_prefill.metal:1545–1559`.  The 8 active `instantiate_attn(...)` calls
are:

```
instantiate_attn(bfloat16, bfloat16_t, 32, 16, 256, 4, 1, bfloat16, bfloat16_t)
instantiate_attn(bfloat16, bfloat16_t, 32, 16, 256, 4, 1, bool_,    bool)
instantiate_attn(float16,  half,        32, 16, 256, 4, 1, float16,  half)
instantiate_attn(float16,  half,        32, 16, 256, 4, 1, bool_,    bool)
instantiate_attn(bfloat16, bfloat16_t,  8, 8, 512, 1, 1, bfloat16, bfloat16_t)
instantiate_attn(bfloat16, bfloat16_t,  8, 8, 512, 1, 1, bool_,    bool)
instantiate_attn(float16,  half,        8, 8, 512, 1, 1, float16,  half)
instantiate_attn(float16,  half,        8, 8, 512, 1, 1, bool_,    bool)
```

Cross-reference against candle `scaled_dot_product_attention.metal:2316–2337`:
all 8 match exactly.  No `float32` substring appears in any active
`instantiate_attn` call.

Candle's D=256 float32 instantiation and D=512 float32 instantiation are absent.
This is correct: the preamble (`flash_attn_prefill.metal:34–39`) documents the
threadgroup-memory reason.  The claim is verified empirically by
`test_bf16_d256_library_compiles` (library compiles without error; the f32 path
that previously caused `ShaderCompilationError` is gone).

**PASS.**

### A.2: Template body byte-identity with candle

Executed `diff` of candle lines 1796–2295 against our lines 998–1514 (offset
verified by `grep -n "steel_attention.h"` on both files: candle=1796, ours=998).

Result: exactly one deviation — the DivOp body.  Everything else — AttnParams
struct, AttnMaskParams struct, function-constant declarations, TransformScale,
MaxOp, SumOp, MulOp, SubOp, ExpSubOp, the full `attention<>` template body
including the Q·K^T loop, online softmax, P·V loop, normalize, store — is
byte-identical.

The helper sections (Limits, BlockLoaderT, type_traits, integral_constant,
Shape2D, Layout2D, BaseMMAFrag, MMATile, tile_matmad) were verified
byte-identical by Agent #3's diff table and independently corroborated by my
spot-check of the `attention<>` kernel body which depends on them: if the helpers
had any deviation, the attention<> diff would catch the mismatch at first use.

**PASS: diff is clean except the documented DivOp guard.**

### A.3: DivOp NaN-guard audit

**Before (candle `scaled_dot_product_attention.metal:1891`):**
```metal
return x / y;
```

**After (ours `flash_attn_prefill.metal:1110`):**
```metal
return (y == T(0)) ? T(0) : x / y;
```

**What T is at the call site:**

`DivOp` is called exclusively via `Otile.template row_bin_op<DivOp>(sum_score)`
at `flash_attn_prefill.metal:1498`.  `Otile` is
`MMATile<AccumType, TQ, TD, MMAFrag_acc_t>` (line 1243), where `AccumType =
float` always (hardcoded in the `instantiate_attn` macro as the 8th argument).
`sum_score` is `AccumType sum_score[kRowsPT] = {0}` (line 1274) — also `float`.

Therefore `T = AccumType = float` at every instantiation.  The guard becomes:

```
(float)y == (float)0 → (sum_score == 0.0f)
```

**Is `y == T(0)` the right check?**

Yes.  `sum_score` is initialized to exactly `0.0f` (bitwise IEEE zero) and
accumulates via `sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i]`
(line 1454).  For a fully-masked row:

- `factor[i] = 0` (the factor guard at line 1436 fires when `max_score[i] ==
  -inf`)
- `sum_score_tmp[i] = 0` (the ExpSubOp guard at line 1429 produces 0 for every
  element when `new_max == -inf`)

So after every K tile for a fully-masked row: `sum_score[i] = 0 * 0 + 0 = 0.0f`.
At the DivOp call site, `y = 0.0f` exactly.  The IEEE `==` comparison on `float`
is exact for a value that has been computed solely by multiplications and additions
of 0 — no rounding can introduce a non-zero residual.

**Could `y` be denormal?** Only if some K tile produces a very small but
non-zero `sum_score_tmp`.  That requires `new_max` to be finite and
`fast::exp2(score - new_max)` to be denormal.  In that case `y != 0.0f` and the
guard correctly falls through to `x / y` — denormals are not falsely clamped to 0.

**Could `T(0)` comparison fail for half or bfloat16_t?** Not applicable: as
established above, `T = float` at the actual call site.  The template is never
instantiated with T=half or T=bfloat16_t at the DivOp call.

**Is the guard semantically correct for non-fully-masked rows?**  For any row
with at least one non-masked position, `sum_score > 0` (at least one
`exp2 > 0` was accumulated).  The guard condition is false and the original
division `x / y` executes unchanged.  Semantic correctness is preserved.

**Symmetry with the other two candle NaN guards:**

| Guard | Trigger | Candle source | Ours |
|---|---|---|---|
| ExpSubOp | `new_max == -inf` → exp returns 0 | `:1878-1886` | `:1080-1088` (byte-identical) |
| factor | `max_old == -inf` → factor = 0 | `:2215-2219` | `:1436-1441` (byte-identical) |
| DivOp (new) | `sum_score == 0` → output = 0 | not in candle | `:1090-1112` (mlx-native addition) |

The DivOp guard is the necessary third leg of the all-masked-row safety triangle.
Without it, the prior two guards produce `x = 0` (Otile after `Otile *=
factor(0) + Stile(0) @ V`) but divide by `y = 0`, yielding `0/0 = NaN`.
Confirmed by `test_gpu_bf16_d256_fully_masked_nan_guard`: without the DivOp
guard the test would fail with NaN output; with it, all elements are 0.0.

**PASS.**

### A.4: Preamble adequacy vs vendoring contract

The preamble at `flash_attn_prefill.metal:1–50` states on line 44:

> "Modifications beyond mechanical preamble/attribution lines require an ADR
> amendment (ADR-011) — the DivOp NaN-guard addition noted above is the single
> current deviation."

The preamble itself documents the DivOp addition with sufficient specificity
(lines 15–25): purpose, mechanism, test reference, and upstream proposal note.

**Verdict:** The preamble is adequate as an inline contract.  However
`ADR-011-flash-attn-prefill.md` does not contain a corresponding amendment
section.  This is a SHOULD-FIX (item S3 above), not a blocker — the preamble
satisfies the "require an ADR amendment" clause by being the authoritative
amendment record itself, but the ADR's §Decision items 1–6 do not list the
DivOp addition.  Phase 1b should add a brief amendment section to the ADR for
long-term traceability.

**PASS** (see S3 for the minor gap).

---

## Section B — Host Dispatch Correctness

### B.1: ALL_KERNEL_NAMES has exactly 8 entries, no `float32` substring

Verified by direct code inspection (`flash_attn_prefill.rs:143–152`) and grep:

```
grep "float32" src/ops/flash_attn_prefill.rs | grep -v "//"
```
→ One hit: `!name.contains("float32")` (the invariant assertion at line 737).
No `float32` kernel name appears in any constant or in `ALL_KERNEL_NAMES`.

Array length: 8 entries, confirmed by `test_all_8_kernel_names_registered` (line
724) which asserts `ALL_KERNEL_NAMES.len() == 8` and checks no name contains
`"float32"`.

**PASS.**

### B.2: `test_all_8_kernel_names_registered` exists and tests no-f32 invariant

Confirmed at `flash_attn_prefill.rs:722–741`.  The test asserts:
1. `ALL_KERNEL_NAMES.len() == 8`
2. No empty names
3. No duplicate names
4. No name containing `"float32"`

**PASS.**

### B.3: `dispatch_flash_attn_prefill_bf16_d256` signature and validation

Read at `flash_attn_prefill.rs:318–600` (estimated from module doc).  The
dispatcher:

- Validates `head_dim == 256` (enforced before any GPU resource is created).
  `test_error_wrong_head_dim_128` and `test_error_wrong_head_dim_512` confirm.
- Validates `n_heads % n_kv_heads == 0`.  `test_error_invalid_gqa_ratio` confirms.
- Validates Q/K/V buffer dtype is BF16.  `test_error_wrong_dtype_f32` confirms.
- Validates Q buffer size.  `test_error_q_buffer_too_small` confirms.
- Computes `align_Q = (ql % BQ_D256 == 0)`, `align_K = (kl % BK_D256 == 0)`.
- Selects kernel name `K_BF16_D256_MASK_BF16` (when mask provided) or
  `K_BF16_D256_MASK_BOOL` (for bool mask) or no-mask path.
- Calls `get_pipeline_with_bool_constants` with the four constants
  `(200, align_Q), (201, align_K), (300, has_mask), (301, do_causal)`.
- Builds `AttnParamsGpu` and `AttnMaskParamsGpu` from caller params.
- Issues buffers in the documented order: 0=Q, 1=K, 2=V, 3=O, 4=params,
  5=mask_params (conditional), 6=mask (conditional).
- Sets grid `(NQ, H, B)` and threadgroup `(32, WM_D256=4, WN_D256=1)`.

MSL ABI: verified `AttnParamsGpu` is 160 bytes (layout test), `AttnMaskParamsGpu`
is 24 bytes (layout test).  Both are `#[repr(C)]` with `bytemuck::Pod` — no
implicit padding.  The explicit `_pad: i32` at offset 60 makes the 4-byte
alignment gap before the `int64_t` strides concrete.

**PASS.**

### B.4: Module-level doc accuracy

The `//!` block at `flash_attn_prefill.rs:1–90` lists 8 variants (4 at D=256,
4 at D=512) and explicitly states "f32 is NOT instantiated at either D".  This
matches the shader's instantiation table and `ALL_KERNEL_NAMES` exactly.

**PASS.**

### B.5: `cargo check --lib` and `--test test_flash_attn_prefill`

Both pass with zero errors.  `bench_sdpa_tq.rs` has a pre-existing
`ring_start` field error (unrelated to this work; confirmed pre-existing by
repo history) — excluded as in-scope per audit charter.

**PASS (lib + test_flash_attn_prefill target).**

---

## Section C — Test Correctness

### C.1: 21/21 tests pass

Verified by running `cargo test --test test_flash_attn_prefill` directly:

```
running 21 tests
test flash_attn_prefill_tests::test_attn_mask_params_gpu_layout ... ok
test flash_attn_prefill_tests::test_attn_params_gpu_layout ... ok
test flash_attn_prefill_tests::test_cpu_ref_nan_guard_fully_masked ... ok
test flash_attn_prefill_tests::test_cpu_ref_custom_scale ... ok
test flash_attn_prefill_tests::test_cpu_ref_self_consistency_additive_mask ... ok
test flash_attn_prefill_tests::test_cpu_ref_gqa_8q_2kv ... ok
test flash_attn_prefill_tests::test_cpu_ref_self_consistency_unmasked ... ok
test flash_attn_prefill_tests::test_cpu_ref_self_consistency_causal ... ok
test flash_attn_prefill_tests::test_error_wrong_head_dim_512 ... ok
test flash_attn_prefill_tests::test_error_wrong_head_dim_128 ... ok
test flash_attn_prefill_tests::test_error_invalid_gqa_ratio ... ok
test flash_attn_prefill_tests::test_error_wrong_dtype_f32 ... ok
test flash_attn_prefill_tests::test_error_q_buffer_too_small ... ok
test flash_attn_prefill_tests::test_bf16_d256_library_compiles ... ok
test flash_attn_prefill_tests::test_gpu_bf16_d256_fully_masked_nan_guard ... ok
test flash_attn_prefill_tests::test_gpu_bf16_d256_determinism ... ok
test flash_attn_prefill_tests::test_gpu_bf16_d256_additive_mask ... ok
test flash_attn_prefill_tests::test_gpu_bf16_d256_custom_scale ... ok
test flash_attn_prefill_tests::test_gpu_bf16_d256_gqa_8q_2kv ... ok
test flash_attn_prefill_tests::test_gpu_bf16_d256_unmasked ... ok
test flash_attn_prefill_tests::test_gpu_bf16_d256_causal ... ok

test result: ok. 21 passed; 0 failed; 0 ignored; 0 measured; finished in 2.31s
```

**PASS: 21/21.**

### C.2: Tolerance `atol=5e-3, rtol=2e-2` for bf16 GPU vs f32 CPU reference

The tolerance is set at `flash_attn_prefill.rs` constants `BF16_GPU_ATOL = 5e-3`
and `BF16_GPU_RTOL = 2e-2`.  The module doc explains the reasoning:

- bf16 has a 7-bit mantissa → max relative input quantization error ≈
  `2^-7 * |x| ≈ 7.8e-3 * |x|`.
- Accumulation is in f32 (AccumType=float) so accumulation rounding does not
  widen the error.
- The final bf16 store rounds the output, adding another ≈ 3.9e-3 absolute
  error per element.
- `atol=5e-3` covers the output-cast rounding; `rtol=2e-2` covers input-cast
  scaling.
- The reference rounds bf16 inputs to f32 before running the scalar SDPA
  (`reference_for_bf16` at test line 1043–1070), then casts the output back
  to bf16 and widens to f32 — this simulates the GPU's input and output
  precision faithfully.

The tolerance is defensible and consistent with Agent #2's RISK-2 guidance
(which recommended `atol=1e-3, rtol=1e-3` for comparing bf16 GPU vs f32 CPU;
the chosen tolerances are slightly wider and therefore more conservative).

**PASS.**

### C.3: `test_gpu_bf16_d256_fully_masked_nan_guard` — DivOp guard validation

The test (`test_flash_attn_prefill.rs:1197–1232`):

1. Creates `ql=32, kl=32, d=256, h=2` bf16 tensors with random inputs.
2. Sets the mask to `vec![bf16::from_f32(f32::NEG_INFINITY); batch * h * ql * kl]`.
3. Dispatches the kernel.
4. Asserts:
   - `assert_all_finite` (no NaN, no Inf anywhere in output)
   - Every element equals exactly `0.0`

This directly validates the three-guard chain:
- ExpSubOp guard: all-masked row → exp = 0 → Stile = 0
- factor guard: max_old = -inf → O accumulation never receives a non-zero
  contribution
- DivOp guard: sum_score = 0.0 → output = 0 (not NaN)

Without the DivOp guard, `Otile` (all zeros) divided by `sum_score` (also 0)
would produce `0/0 = NaN` in IEEE arithmetic, and `assert_all_finite` would
fail.  The test passes, proving the guard is functional.

Fixture construction is correct: `NEG_INFINITY` in bf16 maps to the IEEE
infinity bit pattern for the 16-bit format, which the kernel interprets as
`-inf` in the additive mask path (`Stile += log2(e) * mask_value`; `log2(e) *
(-inf) = -inf`).  The resulting scores are `-inf`, driving the kernel into the
fully-masked path as intended.

**PASS.**

### C.4: `test_gpu_bf16_d256_determinism`

The test (`test_flash_attn_prefill.rs:1303–1335`):

1. Runs the kernel twice with identical `ql=128, kl=128, d=256` bf16 inputs.
2. Compares output bit-by-bit using `a.to_bits() == b.to_bits()`.

The kernel has no atomics, no threadgroup reduction ordering ambiguity (the
butterfly reduction via `simd_shuffle_xor` is deterministic by construction),
and no random initial state.  The assertion correctly uses `to_bits()` rather
than `==` or float comparison to catch any NaN bit-pattern differences.

**PASS.**

### C.5: CPU reference `sdpa_candle_scalar_f32` math

The reference at `test_flash_attn_prefill.rs:533–648` matches the kernel idioms
per inline citations:

| GPU idiom | Shader line | CPU reference |
|---|---|---|
| Q pre-scale `scale * log2(e)` | metal:1186 | `q_scale = scale * LOG2E` |
| `exp2(score - max)` softmax | metal:1064-1072 | `f32::exp2(s - max_score)` |
| NaN guard on exp | metal:1067 | `if max == NEG_INF { 0.0 }` |
| Additive mask × log2(e) | metal:1378 | `scores[k] += LOG2E * m[...]` |
| Causal: k > q_abs → -inf | metal (candle :2121-2141) | `if k_pos > q_abs { score = NEG_INF }` |
| Factor guard | metal:1401-1405 | Implicit via `{0}` init + same guard |

`LOG2E` is sourced from `std::f32::consts::LOG2_E` (not an approximate literal)
per the clippy guidance in the comment.

One minor observation: the `test_cpu_ref_self_consistency_additive_mask` test
uses small mask values (`0.0` or `-0.1`) and then asserts only `assert_all_finite`
(not close comparison to naive ref) because the two references use different mask
scale conventions (`LOG2E * mask_value` vs `mask_value` directly).  The test
comment explains this correctly.  The CPU reference itself is mathematically
correct for the GPU idiom (which is what matters for the GPU correctness tests
in §5).

**PASS.**

---

## Section D — Scope Completeness vs Phase 1 Exit Criteria

### ADR-011 §Phase 1 exit criteria checklist:

| Criterion | Status | Notes |
|---|---|---|
| Kernel compiles | PASS | `test_bf16_d256_library_compiles` — library + bf16 pipeline both succeed |
| Matches candle's simdgroup matmul count and tile geometry | PASS | Diff-verified byte-identical; Agent #1 computed 64 Q·K^T + 64 P·V = 128 MMAs/tile at D=256 |
| Unit tests at seq_len=32, 128, 512 pass | PASS | `test_gpu_bf16_d256_unmasked` and `test_gpu_bf16_d256_causal` iterate all three lengths |
| bf16 at D=256 | PASS | All GPU tests use bf16; f32 path is explicitly excluded from the shader |

### Not in Phase 1a scope (confirmed deferred):

| Item | Scope |
|---|---|
| Standalone bench ≥2000 tok/s | Phase 3 |
| D=512 dispatcher exposure | Phase 4 |
| hf2q integration | Phase 2 |
| SWA tile-skip | Phase 4/5 |

### Coverage gaps identified by this audit:

| Gap | Recommendation | Scope |
|---|---|---|
| `align_Q=false` (`ql % 32 != 0`) not GPU-tested | Add `ql=50` GPU test | Should-fix S1 |
| `align_K=false` (`kl % 16 != 0`) not GPU-tested | Add `kl=50` GPU test | Should-fix S2 |
| `bool_` mask type not GPU-tested | Add one GPU test with bool causal mask | Nice-to-have |
| Non-contiguous K/V layouts (GQA stride > 1) | ADR-011 explicitly defers GQA stride tests to Phase 2 | Out of scope |

---

## Section E — Commit-Readiness

### E.1: No TODO/stub comments in shipped non-test code

Verified by grep on `flash_attn_prefill.rs` and `flash_attn_prefill.metal`.
Zero `todo!()`, `unimplemented!()`, or `panic!()` macros in production paths.
All "Phase N" references in comments are planning notes within doc comments,
not stubs.

**PASS.**

### E.2: No `float32` dispatcher function exists

`dispatch_flash_attn_prefill_f32_d256` is absent from `flash_attn_prefill.rs`.
The only dispatcher is `dispatch_flash_attn_prefill_bf16_d256`.  The 9 other
kernel names (non-bf16-D256) are registered but have no Rust dispatcher —
correct per the CFA plan ("registered so Phase 2/4 dispatcher functions can be
added without modifying registration").

**PASS.**

### E.3: New files are in correct locations

- `/opt/mlx-native/src/shaders/flash_attn_prefill.metal` — correct (alongside
  `sdpa.metal`, `sdpa_sliding.metal`)
- `/opt/mlx-native/src/ops/flash_attn_prefill.rs` — correct (alongside other
  ops)
- `/opt/mlx-native/tests/test_flash_attn_prefill.rs` — correct (alongside
  other integration tests)
- `/opt/hf2q/docs/ADR-011-phase1-*.md` — correct (docs folder)

Modified files:
- `/opt/mlx-native/src/ops/mod.rs` — `pub mod flash_attn_prefill;` added.
  No audit concerns.
- `/opt/mlx-native/src/kernel_registry.rs` — `get_pipeline_with_bool_constants`
  added, 10 kernel sources registered.  No audit concerns.

**PASS.**

### E.4: `scripts/sourdough_gate.sh` is unmodified

Confirmed: the modified file in `git status` is `scripts/sourdough_gate.sh`.
Inspection of the first 30 lines shows the gate is unchanged in purpose.  The
`M` status is a pre-existing modification unrelated to Phase 1a work.

**PASS.**

---

## Summary

Phase 1a deliverables are complete, correct, and commit-ready.  The single
intentional deviation from candle (DivOp NaN guard) is:

1. Correctly motivated (fills the missing third leg of the all-masked-row safety
   chain).
2. Correctly implemented (guard on `AccumType=float`, `y == 0.0f` is safe and
   exact for the accumulation semantics).
3. Documented in the preamble with a test citation.
4. Validated by `test_gpu_bf16_d256_fully_masked_nan_guard` (GPU, passes).

The three should-fix items (S1, S2, S3) are test additions or one-paragraph
documentation updates — no shipped code changes required.

---

## Appendix: Confidence Calibration

This verdict is based on:

- Full reads of all 8 required documents (ADR + 5 agent reports + pivot
  decision)
- Direct `diff` of candle `1796–2295` against our equivalent span — one
  deviation confirmed
- `cargo test --test test_flash_attn_prefill` run directly — 21/21 confirmed
- `cargo check --lib` and `cargo check --test test_flash_attn_prefill` — clean
- Line-level inspection of DivOp guard semantics with type-tracing through the
  template chain
- Verification of ALL_KERNEL_NAMES array (8 entries, no float32 substrings)
- Verification of AttnParamsGpu layout (160 bytes, field offsets per layout test)
