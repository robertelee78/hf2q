# ADR-011 Phase 2 — Port spec: llama.cpp `-FLT_MAX/2` masked-position sentinel

Status: proposed (research complete, implementation pending)
Owner: research-sentinel (CFA swarm-1776516482254-ft5mwj)
Depends on: ADR-011 phase 1 (candle vs llama.cpp source decision), Agent #2's `cfa:agent-2:llamacpp-delta`

## Summary (one paragraph)

llama.cpp's non-vec flash-attention prefill kernel (`kernel_flash_attn_ext_impl`
at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5767`) eliminates the
entire NaN-guard class by **initialising the per-row running max `M` to
`-FLT_MAX/2` (a finite value, ~−1.7e38)** instead of `-infinity` (our current
choice via `Limits<AccumType>::min`, which for `float` resolves to
`-metal::numeric_limits<float>::infinity()` at
`/opt/mlx-native/src/shaders/flash_attn_prefill.metal:357-363`). Combined with
the fact that mask values still arrive at the GPU as `-INFINITY` (f32 on the
CPU side, cast to `half`-`-infinity` before dispatch), every softmax-rescale
expression (`exp(m - M)`, `exp(s2 - M)`) evaluates against a finite `M`, never
against `-inf - -inf`. The sum `S` can still reach exactly 0 for a fully-masked
row; llama.cpp protects the final normalisation with **one** guard
(`const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];` at
`ggml-metal.metal:6358`). That is the whole design. Our kernel's three guards
(`ExpSubOp` at `:1095-1103`, rescale-factor at `:1441-1444`, `DivOp` at
`:1105-1117`) collapse to one output-side guard identical in spirit to
llama.cpp's `:6358`. This ADR specifies the change precisely.

## 1. Exact sentinel values used by llama.cpp

### 1.1 CPU-side mask fill (the ONLY place where the f32 mask buffer is authored)

| Code path | File:Line | Value written for masked positions |
|---|---|---|
| no-cache attention (`llm_graph_input_attn_no_cache::set_input`) | `/opt/llama.cpp/src/llama-graph.cpp:421` | `std::fill(data, data + ggml_nelements(self_kq_mask), -INFINITY);` |
| no-cache SWA mask | `/opt/llama.cpp/src/llama-graph.cpp:436` | `std::fill(data, data + ggml_nelements(self_kq_mask_swa), -INFINITY);` |
| KV-cache `set_input_kq_mask_impl` (per-cell write after `skip:`) | `/opt/llama.cpp/src/llama-kv-cache.cpp:1572` | `data[idst + j] = -INFINITY;` |
| cross-attention mask init | `/opt/llama.cpp/src/llama-graph.cpp:557` | `float f = -INFINITY;` (later `data[j*n_tokens + i] = f;`) |

**llama.cpp writes raw `-INFINITY` (IEEE-754 f32 `0xFF800000`) into the mask
buffer from the CPU.** There is no CPU-side use of `-FLT_MAX/2` at all.

### 1.2 GPU-side value that actually arrives at the kernel (mask buffer dtype)

llama.cpp converts the f32 mask to f16 immediately before handing it to the
flash-attention op when the flash-attn path is enabled:

- `/opt/llama.cpp/src/llama-graph.cpp:1995`
  `inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;`
- Same cast at `:2077`, `:2178`, `:2338`, `:2398`, `:2576` for every
  attention-input builder.

f32 `-INFINITY` cast to f16 saturates to f16 `-INFINITY` (f16 has a real
`infinity` encoding). So the kernel sees **`half(-INFINITY)` at every masked
position from the main mask path**.

### 1.3 GPU-side M (running row-max) init — the design lever

**Non-vec (prefill) kernel**, at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5891`:

```cpp
float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };
```

**Vec (decode) kernel**, at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6725`:

```cpp
float M = -FLT_MAX/2;
```

**Sinks path** (one-extra-token-per-row bias used by Gemma sinks), at
`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6333` and `:6977`:

```cpp
const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;
```

`-FLT_MAX/2 ≈ −1.7014118 × 10³⁸` — a finite, very large negative number.
Chosen so that `M + M` does not overflow to `-inf` (a raw `-FLT_MAX - FLT_MAX`
saturates, but `-FLT_MAX/2 + -FLT_MAX/2 = -FLT_MAX`, still finite).

### 1.4 Secondary GPU-side sentinel for mask padding only

llama.cpp ALSO uses `-MAXHALF` (= `-HALF_MAX` ≈ `-65504`) at three
padding/boundary sites, because the shared-memory mask tile is half-precision
and `half(-FLT_MAX/2)` would saturate to `half(-inf)` anyway. These sites only
run when the KV tile is a partial remainder or an out-of-range row:

- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5649` —
  `mask_dst[i] = -MAXHALF;` in the kvpad mask writer
- `:5933` — same value when synthesising a mask for the kvpad path
  with no user mask
- `:5970` — `half2(-MAXHALF, -MAXHALF)` as the out-of-range fallback
  when `(iq1 + j) >= args.ne31`

These are **NOT** the primary mask value; they only appear in partial-tile
padding code paths. They coexist with `M = -FLT_MAX/2` without issue because
`-MAXHALF` is ALSO finite (it's f16-finite, and when promoted to f32 via
`s2_t(sm2[...])` at `:6149` it stays as the f32 value `-65504.0`).

### 1.5 Consistency (and why the two values differ)

`M` is f32 (kernel-local register). The mask tile is f16 threadgroup memory.
The two values are **not equal** (`-FLT_MAX/2` is ~10³⁸, `-MAXHALF` is
~6.5×10⁴), and they do not need to be. What matters is that BOTH are finite
and BOTH dominate any real QK^T score a non-masked position could produce
(real scores are at most ~O(√D), so |score| ≲ 20 after scaling). The arithmetic
that consumes them is:

- `s2 = QK[i,j] * scale + slope * mask[i,j]` (line 6149) — this is the
  addition of a near-zero score with either `0` (attended) or `-MAXHALF`
  (masked-by-f16-mask) or `-INFINITY` cast-to-f16 then promoted (`-INF`)
- `M = simd_max(max(M, max(s2[0], s2[1])))` (line 6153) — `simd_max`
  between `-FLT_MAX/2` and either the real score or `-INF` gives
  `-FLT_MAX/2` as a hard floor
- `exp(m - M)`, `exp(s2 - M)` at `:6155-6156` — `exp` of a finite
  negative is 0 or a tiny positive, never NaN

**Any place where `-INFINITY` IS still used despite the finite-M convention:**

1. CPU-side mask values (f32 `-INFINITY` at `llama-graph.cpp:421,436,1572`) —
   this is the ONLY place llama.cpp uses `-INFINITY`. After cast to f16 for
   flash-attn, it becomes f16-`-INFINITY`. When consumed inside the kernel at
   `s2 += s2_t(sm2[...])` (where `s2_t = float`), f16-`-inf` promotes to
   f32-`-inf`. So `s2` CAN be `-inf` mid-flight.
2. BUT: `simd_max(max(-FLT_MAX/2, max(-inf, -inf)))` = `-FLT_MAX/2`. The
   finite M-init **absorbs** the `-inf` values from the mask, so `M`
   NEVER becomes `-inf` across any K-tile.
3. `exp(-inf - (-FLT_MAX/2)) = exp(-inf) = 0` — exact zero in f32, no NaN
   (the `-inf - -inf` case is what produces NaN; llama.cpp never hits that
   case because M is never `-inf`).

This is the whole trick: **let `-inf` enter the score, but guarantee `M` is
finite so the `exp(score - M)` path never hits `exp(-inf - -inf) = exp(NaN)`.**

## 2. Why the sentinel approach eliminates each NaN guard

### 2.1 Guard #1 — `ExpSubOp` (our `flash_attn_prefill.metal:1095-1103`)

Our current code:
```cpp
METAL_FUNC static constexpr T apply(T x, T y) {
  return (y == -metal::numeric_limits<T>::infinity())
      ? T(0)
      : fast::exp2(x - y);
}
```

With `-FLT_MAX/2` sentinel for M init and `-FLT_MAX/2` effectively floor-capped
by the `simd_max` step (which we'd add to mirror the llama.cpp structure):

- The test `y == -infinity` is **never true** because M is initialised finite
  and `simd_max(-FLT_MAX/2, anything)` stays finite.
- With `y = -FLT_MAX/2` on the fully-masked-row iteration:
  `x - y = -inf - (-FLT_MAX/2) = -inf` (float subtraction of inf-from-finite
  stays inf). `fast::exp2(-inf) = 0` exactly (IEEE-754 guarantees
  `exp2(-inf) = +0`, no NaN). Same outcome as the guard, zero branch.
- For non-degenerate rows, `y` is a real finite score, and the behaviour is
  identical to the unguarded form.

**Verdict: always-false-with-sentinel; guard can be removed** (reverts to
`return fast::exp2(x - y);`).

**Verification from llama.cpp:** `ggml-metal.metal:6155-6156` does exactly this
with no guard:
```cpp
const float  ms  = exp(m  - M[jj]);
const float2 vs2 = exp(s2 - M[jj]);
```
`M[jj]` is finite (`-FLT_MAX/2` at init, or `simd_max` of finite floor and real
scores thereafter — never `-inf`), so these two lines are unguarded.

### 2.2 Guard #2 — rescale factor (our `flash_attn_prefill.metal:1441-1444`)

Our current code:
```cpp
factor[i] = (max_score[i] == -metal::numeric_limits<AccumType>::infinity())
    ? AccumType(0)
    : fast::exp2(max_score[i] - new_max[i]);
```

With `-FLT_MAX/2` sentinel:

- `max_score[i]` (= `m` in llama.cpp's notation) is finite on every iteration
  (`-FLT_MAX/2` initially, then `simd_max(finite, finite)` after).
- `new_max[i]` (= `M[jj]` after update) is also finite.
- `max_score[i] - new_max[i]` ∈ `[-FLT_MAX, 0]` — always finite.
  On the first K-tile iteration of a fully-masked row:
  `-FLT_MAX/2 - -FLT_MAX/2 = 0`, so `fast::exp2(0) = 1.0` (NOT 0).
  Then `sum_score = sum_score * 1.0 + 0 = 0` (the row-sum of exp2 of -inf
  values is 0), so sum stays 0 across the sweep. Otile is multiplied by 1.0
  every iteration: stays 0.
- On a non-degenerate row, `new_max ≥ max_score` so the factor is in `(0, 1]`,
  standard online-softmax semantics.

**Verdict: always-false-with-sentinel; guard can be removed** (reverts to
`factor[i] = fast::exp2(max_score[i] - new_max[i]);`).

**Verification from llama.cpp:** `ggml-metal.metal:6155` uses the guardless
form:
```cpp
const float  ms  = exp(m  - M[jj]);
```
which is Otile's rescale factor (`S[jj] = S[jj]*ms + ...` at `:6158`, and
`so4[j*PV4 + i] *= ms;` at `:6167-6168`, `:6171`).

### 2.3 Guard #3 — `DivOp` (our `flash_attn_prefill.metal:1105-1117`)

Our current code:
```cpp
return (y == T(0)) ? T(0) : x / y;
```

With `-FLT_MAX/2` sentinel the reasoning is **not** "sum is always > 0". The
sum CAN still reach exactly 0:

- On a fully-masked row, every score in the sweep is `-inf` (from the mask
  adding `-inf` to the real score). `fast::exp2(s - M) = fast::exp2(-inf -
  -FLT_MAX/2) = fast::exp2(-inf) = 0`. Sum of zeros is zero.
- On each K-tile iteration `sum_score[i] = sum_score[i] * factor[i] +
  sum_score_tmp[i] = 0 * 1.0 + 0 = 0`. Stays 0 for the whole sweep.

So the division IS still `0/0`. **This guard CANNOT be removed the same way
as Guards #1 and #2** — a bit-exact port of llama.cpp's design must keep
some equivalent.

Llama.cpp's equivalent is at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6358`:
```cpp
const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];
```
and then at `:6364` / `:6368`: `dst4[i] = (float4) so4[j*PV4 + i]*scale;` —
they compute the reciprocal with a zero-guard, then multiply O by it, rather
than dividing O by S directly. Same semantics; one guard survives.

**Verdict: still-needed but simplified** — keep the `DivOp` guard (or
equivalently fold it into the final store step as llama.cpp does: compute
`scale = S == 0 ? 0 : 1/S` once per row, then multiply O by scale).

## 3. Numerical implications (verified against llama.cpp)

### 3.1 `-FLT_MAX/2` arithmetic overview

- `-FLT_MAX/2 ≈ -1.7014118 × 10³⁸`, finite f32.
- `-FLT_MAX/2 - (-FLT_MAX/2) = 0.0` exactly. `fast::exp2(0) = 1.0`.
- `-FLT_MAX/2 + -FLT_MAX/2 = -FLT_MAX` (~`-3.4e38`), still finite. Does not
  overflow to `-inf` under IEEE-754 round-to-nearest f32 addition.
- No `-FLT_MAX/2` path flows into `exp2()` with a `NaN` argument, because
  both operands are finite and their difference is finite.

### 3.2 "What about the mask value itself?"

The masked positions arrive as `-inf` (half-cast from f32 `-INFINITY`). So
the **scores** `s2` can be `-inf` after mask addition. But `M` (the running
max) stays finite because `simd_max(finite, -inf)` returns the finite value.
Then `s2 - M = -inf - finite = -inf`. Then `fast::exp2(-inf) = 0` exactly.

This is bit-exact: the smallest f32 denormal has exponent `-149`, so any
`exp2(x)` for `x < -149` rounds to `+0.0`. `exp2(-inf)` is defined by IEEE-754
to be `+0.0`. Same result either way. **Bit-exact zero** for the softmax
numerator on masked positions.

### 3.3 Fully-masked-row output under llama.cpp's rules (code trace)

Trace `ggml-metal.metal:5888-6374` for a fully-masked row:

1. `S[jj] = 0.0f` (line 5888).
2. `M[jj] = -FLT_MAX/2` (line 5891).
3. First K-tile: `s2 = score + mask = finite + -inf = -inf` (line 6149).
4. `M[jj] = simd_max(max(-FLT_MAX/2, max(-inf, -inf))) = -FLT_MAX/2` (line 6153).
5. `ms = exp(-FLT_MAX/2 - -FLT_MAX/2) = exp(0) = 1` (line 6155).
6. `vs2 = exp(-inf - -FLT_MAX/2) = exp(-inf) = 0` (line 6156).
7. `S[jj] = S[jj]*1 + simd_sum(0 + 0) = 0` (line 6158).
8. `ss2[...] = vs2 = (0, 0)` (line 6161). P matrix row is all zeros.
9. `so4[...] *= 1` (line 6167 / 6171) — output accumulator unchanged (stays 0).
10. On each subsequent K-tile, steps 3-9 repeat identically with `S, M, so4`
    unchanged.
11. After the loop, `S[jj] == 0.0`.
12. `scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj] = 0.0f` (line 6358).
13. `dst4[i] = so4[...] * 0 = 0` (line 6364 / 6368).

**Fully-masked-row output: exact 0.0 in every component** — same as our current
kernel, just reached by different arithmetic and with only ONE guard.

### 3.4 Does llama.cpp handle the sinks path safely under this regime?

Yes. At `:6333-6338` the sinks update path does:
```cpp
const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;
M[jj] = simd_max(max(M[jj], s));
const float ms = exp(m - M[jj]);
const float vs = exp(s - M[jj]);
```
If no real K positions survived (`M[jj] == -FLT_MAX/2` still from init) AND
the sink is also `-FLT_MAX/2` for all non-first lanes, then after simd_max the
new `M[jj]` = max(−FLT_MAX/2, sinks[iq2]). sinks[iq2] is a finite bias (model
parameter), so `M[jj]` becomes finite too. Safe.

## 4. Change spec for our kernel

### 4.1 Kernel side — `/opt/mlx-native/src/shaders/flash_attn_prefill.metal`

1. **M-init**: change line `:1284` from
   ```cpp
   max_score[i] = Limits<AccumType>::min;
   ```
   to
   ```cpp
   max_score[i] = -FLT_MAX/2;   // llama.cpp convention: finite sentinel absorbs -inf from mask
   ```
   `Limits<AccumType>::min` here resolves to
   `-metal::numeric_limits<float>::infinity()` via the `instantiate_float_limit`
   template at `:353-368`. That is the root of every NaN-guard need; replacing
   it with `-FLT_MAX/2` kills the whole class.

2. **Guard #1 removal (ExpSubOp)**: revert `:1095-1103` to the guardless form:
   ```cpp
   struct ExpSubOp {
     template <typename T>
     METAL_FUNC static constexpr T apply(T x, T y) {
       return fast::exp2(x - y);
     }
   };
   ```

3. **Guard #2 removal (rescale factor)**: revert `:1439-1444` to the guardless
   loop:
   ```cpp
   STEEL_PRAGMA_UNROLL
   for (short i = 0; i < kRowsPT; ++i) {
     factor[i] = fast::exp2(max_score[i] - new_max[i]);
   }
   ```
   Delete the 3-line comment at `:1437-1438` (the "Guard: when max_score ==
   -inf (no valid K seen yet)" block) — no longer accurate.

4. **Guard #3 — KEEP but relocate**: the current `DivOp` at `:1105-1117` is the
   equivalent of llama.cpp's final-scale guard. Two acceptable forms:
   - **Option A (minimal change):** leave `DivOp` as-is. It is still correct
     under the `-FLT_MAX/2` regime: `sum_score` is bit-exact 0 on a fully
     masked row (see §3.3 step 11), and `DivOp` turns `0/0` into `0` at the
     final `Otile.template row_bin_op<DivOp>(sum_score)` at `:1503`.
   - **Option B (byte-identical to llama.cpp):** replace the `DivOp` + tile
     op at `:1503` with a pre-computed reciprocal scalar:
     ```cpp
     AccumType inv_sum[kRowsPT];
     STEEL_PRAGMA_UNROLL
     for (short i = 0; i < kRowsPT; ++i) {
       inv_sum[i] = sum_score[i] == AccumType(0) ? AccumType(0) : AccumType(1) / sum_score[i];
     }
     Otile.template row_bin_op<MulOp>(inv_sum);
     ```
     This mirrors `ggml-metal.metal:6358` + `:6364` exactly (reciprocal-with-
     zero-guard, then multiply).
   - **Recommended**: Option A. It is the smallest safe change and keeps the
     existing test working without modification. Option B is a follow-on
     micro-optimisation (one divide → one multiply in the hot loop).

5. **Rename the kernel preamble**: the comment block at `flash_attn_prefill.metal:32-46`
   ("Numerical guards (three of them)") must be rewritten to describe the
   `-FLT_MAX/2` sentinel and the single remaining guard at the output
   normalisation. Also update `:53-55` (the candle-derivation mention) —
   the design is now llama.cpp-derived at the numerical layer.

6. **Update cross-kernel references**: the causal-masking branch at `:1327,
   1348, 1371` fills the score tile with `-metal::numeric_limits<selem_t>::infinity()`
   at masked positions. This is fine — scores CAN be `-inf` (they already are
   when an external mask is `-inf`); M stays finite via the `simd_max`
   dynamics. **No change needed** at these lines. (We could in principle
   change them to `-FLT_MAX/2` for consistency, but (a) it's unnecessary —
   the math is the same — and (b) llama.cpp itself leaves `-INFINITY` in the
   mask buffer and only uses `-FLT_MAX/2` for M init, so our mirrored
   behaviour is faithful.)

### 4.2 Host side — `/opt/mlx-native/src/ops/flash_attn_prefill.rs`

**No change required.** The host-side dispatcher does not author mask values;
it accepts a caller-provided mask buffer (documented at `:384` as
"additive, log-scale: 0.0 = attend, -inf = mask out"). Callers continue to
pass `-inf` in the mask buffer, which is consistent with llama.cpp's CPU-side
convention (`-INFINITY`).

**Callers** of `dispatch_flash_attn_prefill_bf16_d256` and friends: verify
that every mask-authoring site uses `-inf` (= `f32::NEG_INFINITY` /
`half::neg_infinity()` / `bfloat16::NEG_INFINITY`), not `f32::MIN` or
`-FLT_MAX`. Grep survey needed:

```bash
grep -rn 'NEG_INFINITY\|MIN_POSITIVE\|f32::MIN\|MAX/2\|FLT_MAX' /opt/hf2q/src /opt/hf2q/crates /opt/mlx-native/src --include='*.rs'
```
This is a pre-port audit, not a mandated change — expected outcome is "only
NEG_INFINITY appears, matches llama.cpp".

**Documentation change at `flash_attn_prefill.rs:289-291`** (FlashAttnPrefillParams.do_causal doc
comment, citing the current kernel's neg-inf behaviour):
```rust
/// When true, positions where `row_pos < col_pos` receive a score of -inf
/// before softmax.  This can be combined with an external mask buffer.
```
This is still accurate under the new regime (scores CAN be `-inf`; the M init
is what changes, not the mask values). No edit needed.

### 4.3 Test side — `/opt/mlx-native/tests/test_flash_attn_prefill.rs`

**`test_cpu_ref_nan_guard_fully_masked`** at `:801-828`: the semantic intent —
"fully-masked input produces 0 output, not NaN" — remains valid. The CPU
reference's NaN-guard comments at `:793-799` must be updated to reflect the
new design (finite M init → no intermediate NaN → one DivOp-equivalent guard).
Test expectation (all outputs = 0.0) stays the same. **One-line test rename
recommended**: `test_cpu_ref_fully_masked_zero_output` — the term "NaN guard"
becomes misleading.

**`test_gpu_bf16_d256_fully_masked_nan_guard`** at `:1194-1229`: same comment
update needed at `:1184-1193` (the docstring describes "three guards"). Test
body (set entire mask to `-inf`; expect all-zero finite output) is unchanged.
Test expectation stays the same. **One-line test rename recommended**:
`test_gpu_bf16_d256_fully_masked_zero_output`.

**CPU reference (`sdpa_reference_f32`, `:606-635`)**: the max-score
init at `:609` is `f32::NEG_INFINITY` with a NaN-guard branch at `:622-625`.
Under the new regime this should be `f32::MIN / 2.0` (= −1.7014118 × 10³⁸,
equivalent to MSL's `-FLT_MAX/2`) and the branch removed:
```rust
let mut max_score = f32::MIN / 2.0; // llama.cpp convention (ggml-metal.metal:5891)
for &s in &scores {
    if s > max_score {
        max_score = s;
    }
}
// exp2(score - max), no branch — max is finite by construction
let exp_scores: Vec<f32> = scores.iter().map(|&s| f32::exp2(s - max_score)).collect();
```
Keep the final `safe_sum` guard at `:635` (mirror of Guard #3 / llama.cpp
`:6358`): `let safe_sum = if sum_exp == 0.0 { 1.0 } else { sum_exp };`
(or equivalently `let inv_sum = if sum_exp == 0.0 { 0.0 } else { 1.0 / sum_exp };`
and then `exp_scores[k] * inv_sum` in the V-weighted sum — bit-exact match to
llama.cpp's `:6358`+`:6364`).

Also update the `sdpa_naive_scalar_f32` reference at `:509-515`: same one-line
change (`max_s = f32::MIN / 2.0`; remove the `NEG_INFINITY` branch).

Numerical consequence: CPU reference output for non-masked inputs is BYTE-IDENTICAL
to the current NEG_INFINITY version, because:
- `max_score` in the NEG_INFINITY version becomes the real max after the first
  non-masked element; same as `f32::MIN / 2.0` overwritten by the first
  non-masked element.
- The tolerance budget at `:674, :788, :851` (CPU_SELF_CONSISTENCY_ATOL) is
  unchanged.

## 5. Chesterton's fence — why the guards were there

Candle commit `46928bcedb2751e7526112f847a0a88e2ee73d5d` ("Fix sliding window
full sdpa corner case (#3438)", author Eric Buehler, 2026-04-01) introduced
Guards #1 and #2 in candle-metal-kernels. The diff
(`/opt/candle/candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal`,
lines 1876-1886 and 2207-2219 in the post-commit file) is IDENTICAL — modulo
brace style — to our current `ExpSubOp` and rescale-factor guards. The
inline comment in candle's diff:
```
Guard: when y (row max) is -inf, all scores in the row are -inf
(entirely masked). Return 0 instead of exp2(-inf - (-inf)) = exp2(NaN).
```
matches our kernel preamble verbatim.

**Was this a real scenario or defensive programming?** Real scenario. The
commit title says "fix" and the PR number (#3438) is a bug fix, not a
prophylactic. In sliding-window attention (SWA), when the window is shorter
than the query offset, an entire tile can be fully out-of-window — every
score in the tile receives the mask's `-inf`, so `M` becomes `-inf` and every
subsequent op NaNs out. Gemma 4 and Gemma 2 both use SWA for 5 of every 6
layers. Agent #2's delta report (retrieved from
`cfa:agent-2:llamacpp-delta`) confirmed the scenario: "RISK-3: Candle NaN
guards … must be preserved. llama.cpp avoids NaN via -FLT_MAX/2 init, not
guards. Removing guards introduces latent NaN bug on all-masked tiles."

**Does llama.cpp's `-FLT_MAX/2` approach handle the same SWA scenario?** Yes,
completely. The SWA fully-masked-tile case reduces to the fully-masked-row
analysis in §3.3: scores are all `-inf` from the mask, M stays at
`-FLT_MAX/2`, ms=1, vs=0, S=0, Otile stays at its previous value (zero on the
first iteration, accumulated real contributions from prior non-masked tiles
otherwise), and the final `S == 0 ? 0 : 1/S` guard handles the
everything-masked-everywhere degenerate case.

**Guard #3 (our addition, not candle's)** was added because our **first**
fully-masked test uncovered that Guards #1 and #2 don't cover the
final-division NaN. It's a real fix, not defensive — but under the
`-FLT_MAX/2` regime it's the ONLY guard we need, and llama.cpp has it too
(`:6358`).

## 6. Side effects and risks

### 6.1 Numerical difference at non-degenerate inputs

**Bit-exact identical** for every non-degenerate input.
- In the current kernel, the first K-tile iteration overwrites the `-inf`
  M-init with the real row-max, which is some finite QK^T score. All subsequent
  arithmetic uses finite values.
- In the new kernel, the first K-tile iteration does `simd_max(-FLT_MAX/2,
  finite_score) = finite_score` (same as before), because any real score is
  >>> `-FLT_MAX/2`. Subsequent arithmetic is identical.
- The only code paths that differ are (a) the eliminated guard branches
  (compiler produces tighter code) and (b) the fully-masked-row path
  (previously Guard #1,2,3; now one output-side guard — same final output).

**No accuracy change to the non-degenerate path.** Verified by construction:
every real-score arithmetic operation produces the same f32 bits whether M
init was `-inf` or `-FLT_MAX/2`.

### 6.2 Near-`-inf` finite scores

"If our kernel currently has any places where a finite value flows INTO the
sentinel path (e.g. a near-`-infinity` but not exactly-`-infinity` value that
should be treated as masked), does `-FLT_MAX/2` still work?" — Yes.
- A "near `-inf`" finite mask value (e.g., `-1e30`) is treated as a very
  strong bias, not as "masked": real score + `-1e30` = `-1e30`; after
  softmax `exp2(-1e30 / finite)` is bit-exact 0 in f32 too. Same outcome.
- If any caller uses `f32::MIN` (-3.4e38) as a mask value, it becomes more
  negative than `-FLT_MAX/2`. In the new regime `simd_max(-FLT_MAX/2,
  -3.4e38) = -FLT_MAX/2`, exp(score - M) = exp(-3.4e38 - -1.7e38) =
  exp(-1.7e38) = 0 exactly. Same as before. No breakage.

### 6.3 Does MLX / candle-latest use `-infinity` or `-FLT_MAX/2`?

- **Candle HEAD** (`/opt/candle/candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal`,
  post-commit `46928bce`): uses `-infinity` + two guards. We are branching
  AHEAD of candle's current design by switching to llama.cpp's approach. This
  is intentional and documented here.
- **MLX upstream reference**: not checked out locally (confirmed by
  `find /opt -maxdepth 4 -path '*/metal/kernels/steel/attn*'` — only
  `/opt/candle`, `/opt/mlx-native`, `/opt/hf2q`). MLX's behaviour is not a
  direct input to this decision — we are porting llama.cpp's numerical
  convention, not MLX's. **Open question**: worth confirming MLX current
  behaviour in Phase 2 QA (if MLX has ALSO switched away from `-inf`, that's
  an independent confirmation signal; if not, it doesn't change our plan).

### 6.4 Risk: `-FLT_MAX/2` + `factor*factor*...` accumulation across K-tiles

Under the new regime, `factor = fast::exp2(max_score - new_max)` is always in
`[0, 1]` (because `max_score ≤ new_max`). Accumulated as `sum_score *= factor`
and `Otile *= factor`, no unbounded growth is possible. Bounded-below by 0
(sum stays non-negative; Otile's magnitude can decrease but never overflow).
**No new risk introduced.**

## 7. Port spec — actionable checklist

1. **[metal.shader]** Replace `max_score[i] = Limits<AccumType>::min;` at
   `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1284` with
   `max_score[i] = -FLT_MAX/2;`
   - Source: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5891`
     (non-vec prefill) and `:6725` (vec decode — both use the same value)

2. **[metal.shader]** Simplify `ExpSubOp` at
   `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1095-1103` to:
   ```cpp
   struct ExpSubOp {
     template <typename T>
     METAL_FUNC static constexpr T apply(T x, T y) {
       return fast::exp2(x - y);
     }
   };
   ```
   Delete the "Guard" comment at the start.
   - Source: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6156`
     (`const float2 vs2 = exp(s2 - M[jj]);` — unguarded)

3. **[metal.shader]** Simplify rescale factor loop at
   `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1439-1444` to:
   ```cpp
   STEEL_PRAGMA_UNROLL
   for (short i = 0; i < kRowsPT; ++i) {
     factor[i] = fast::exp2(max_score[i] - new_max[i]);
   }
   ```
   Delete the `Guard:` comment at `:1437-1438`.
   - Source: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6155`
     (`const float ms = exp(m - M[jj]);` — unguarded)

4. **[metal.shader]** Keep `DivOp` at
   `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1105-1117` as-is (Option A).
   Update the comment to reflect the new rationale: "mirrors llama.cpp's
   `S[jj] == 0.0 ? 0.0f : 1.0f/S[jj]` at ggml-metal.metal:6358".
   - Source: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6358`

5. **[metal.shader]** Rewrite the kernel preamble block at
   `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:32-46` to describe the
   `-FLT_MAX/2` sentinel approach and the single output-side guard.
   Update `:53-55` (candle-derivation reference) to clarify that the
   numerical design is llama.cpp's, not candle's.

6. **[cpu-ref.test]** In `/opt/mlx-native/tests/test_flash_attn_prefill.rs`,
   change `sdpa_reference_f32` at `:609` from
   `let mut max_score = f32::NEG_INFINITY;`
   to `let mut max_score = f32::MIN / 2.0;` and remove the branch at `:621-630`
   (replace with unconditional `f32::exp2(s - max_score)`).
   - Source: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5891`

7. **[cpu-ref.test]** Same change in `sdpa_naive_scalar_f32` at
   `/opt/mlx-native/tests/test_flash_attn_prefill.rs:509` — replace
   `f32::NEG_INFINITY` init with `f32::MIN / 2.0` and remove the branch at
   `:512` in favour of unconditional `(s - max_s).exp()`. The `safe_sum` guard
   at `:515` stays.
   - Source: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6358`

8. **[test.rename]** Rename
   `test_cpu_ref_nan_guard_fully_masked` → `test_cpu_ref_fully_masked_zero_output`
   and update its `:791-799` docstring to describe the finite-sentinel design
   with one output-side guard.

9. **[test.rename]** Rename
   `test_gpu_bf16_d256_fully_masked_nan_guard` → `test_gpu_bf16_d256_fully_masked_zero_output`
   and update its `:1183-1193` docstring similarly. Test body and expectations
   unchanged.

10. **[test.preserve]** Do NOT modify the test inputs (all-`-inf` mask) or the
    expected outputs (all 0.0). Those are the correct semantics in both
    regimes — llama.cpp produces the same all-zero output for a fully-masked
    row by the trace in §3.3.

11. **[docs]** In `/opt/hf2q/docs/ADR-011-flash-attn-prefill.md` (if it
    mentions the three-guard design, currently at comments around candle port
    rationale), add a pointer to this ADR and mark the guards discussion as
    superseded.

12. **[bench]** After the change lands, re-run the existing flash-attn-prefill
    benchmarks (Gemma 4 26B MoE DWQ, prefill tok/s). Expected: no regression,
    possibly a small speedup on global-attention layers where the eliminated
    branches were in the hot path. Expected prefill tok/s ~3300-3450 (matching
    llama.cpp's current measured peer reference, per
    `project_end_gate_reality_check` memory).

13. **[test.full-suite]** Before merging, run the full test suite including
    the existing `fully_masked_zero_output` test + all non-degenerate tests
    from §5-§7 of `test_flash_attn_prefill.rs`. Acceptance gate: all tests
    pass with bit-exact identical outputs for non-degenerate inputs and
    all-zero outputs for the fully-masked case.

## Cross-references

- Phase 1 delta report: `cfa:agent-2:llamacpp-delta` (RISK-3 flagged this
  exact switch opportunity) — retrieved and verified from source.
- Related ADR: `/opt/hf2q/docs/ADR-011-phase1-port-source-decision.md`
  (threadgroup-memory analysis; unaffected by this port).
- llama.cpp primary source: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
  lines `5767` (kernel start), `5891` (M init), `6155-6156` (unguarded exp),
  `6358` (output-side S==0 guard), `6725` (vec kernel same convention),
  `6333` and `:6977` (sinks same convention).
- llama.cpp mask authoring (CPU side): `/opt/llama.cpp/src/llama-graph.cpp:421,
  436, 557`; `/opt/llama.cpp/src/llama-kv-cache.cpp:1572`.
- Candle origin of guards: commit `46928bcedb2751e7526112f847a0a88e2ee73d5d`
  "Fix sliding window full sdpa corner case (#3438)" in `/opt/candle`.

## Open questions (document, don't guess)

- **Does MLX upstream (Apple's `metal/kernels/steel/attn`) use `-FLT_MAX/2` or
  `-inf`?** Not checked locally. Agent #2's delta report did not cover this.
  Non-blocking for the port (the llama.cpp convention is sufficient) but
  worth confirming as an independent cross-check.
- **Does llama.cpp's non-flash-attn (plain SDPA) path use the same convention?**
  `ggml-metal.metal:255` has `float max = -FLT_MAX;` (not `/2`) — different
  value. Plain SDPA is not in our port scope (we only port flash-attn-prefill
  and flash-attn-vec), so this divergence does not affect us. Documenting for
  completeness.
- **Option A vs Option B** for Guard #3 (`DivOp` in-place vs reciprocal scalar):
  leave as Option A for the initial port. Option B is a candidate
  micro-optimisation for a later phase (one divide → one multiply in the hot
  loop times `kRowsPT` rows).

## Confidence

High (0.93). Every non-degenerate-path claim is verified against llama.cpp
source lines directly; the fully-masked-row trace at §3.3 is a line-by-line
walk of ggml-metal.metal; the Chesterton's-fence origin is verified via
`git show 46928bce` in `/opt/candle`.
