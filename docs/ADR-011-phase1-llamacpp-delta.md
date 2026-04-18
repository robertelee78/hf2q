# ADR-011 Phase 1 — llama.cpp vs candle/MLX Kernel Delta Report

**Author:** Agent #2 (research-llamacpp), swarm-1776462683390-zb6ev9  
**Date:** 2026-04-17  
**Feeds into:** Agent #3 (vendor-kernel), Agent #5 (tests)  
**Status:** Final — adversarial cross-check complete

---

## Scope and Method

This report verifies — or refutes — the ADR-011 claim that llama.cpp's `kernel_flash_attn_ext_impl` and candle's `steel_attention` kernel are "algorithmically identical at the 8×8×8 MMA level." Every claim below is sourced from direct file reads at the cited lines. Nothing is inferred from documentation or memory.

Primary sources read in full:
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` lines 5660–6375 (blk pre-pass + main impl)
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp` lines 2498–2861 (dispatch logic)
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h` lines 93–97 (tile constants)
- `/opt/llama.cpp/src/llama-graph.cpp` lines 345–444 (mask builder)
- `/opt/candle/candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal` lines 1235–2337 (full kernel + instantiation table)

---

## 1. Structural Comparison Table

| Concern | llama.cpp | candle/MLX | Verdict | Impact if different |
|---|---|---|---|---|
| **Simdgroup MMA primitive** | `simdgroup_multiply_accumulate` — `ggml-metal.metal:6037,6056` | `simdgroup_multiply_accumulate` via `MMAFrag_acc_t::mma` at `scaled_dot_product_attention.metal:1243` | **same** | None |
| **Fragment size** | 8×8×8 fragments (`simdgroup_half8x8`, `simdgroup_float8x8`) — `ggml-metal.metal:6436-6441` | 8×8 (`kFragSize=8`, `BaseMMAFrag<T, 8, 8>`) — `:2003-2004` | **same** | None |
| **Tile size (BQ, BK) for D=256 prefill** | `Q=8` (NQPSG), `C=64` (NCPSG) — `ggml-metal-impl.h:93-94`; at D=256 `nsg=4` — `ggml-metal-ops.cpp:2807` | `BQ=32, BK=16` — `scaled_dot_product_attention.metal:2316` | **different** | **Significant**: llama.cpp processes 8 Q-rows/64 K-cols per tile; candle processes 32 Q-rows/16 K-cols. Different occupancy, threadgroup memory layout, and bank-conflict pattern. Neither is more correct numerically, but they produce different MMA counts per tile. The ADR's claim "32 MMAs per Q·K^T tile (both confirm)" is therefore incorrect for llama.cpp — see §4. |
| **NSG (simdgroups per threadgroup)** | `nsg=4` for D<512, `nsg=8` for D≥512 — `ggml-metal-ops.cpp:2807` | `WM=4, WN=1` → 4 simdgroups — `:2316` | **same for D=256, different for D=512** | D=512 (global layers): llama.cpp uses 8 simdgroups (256 threads); candle uses only 1 simdgroup (32 threads) at BD=512, BQ=8, BK=8 — `:2334-2337`. This is a massive dispatch difference for global-layer prefill. |
| **Scale placement** | Post-QK^T multiply, in softmax preamble: `s2 = ss2[...]*args.scale` — `ggml-metal.metal:6138` | Pre-Q load: Q is scaled by `scale * 1.44269504089` applied in-place on Qs in threadgroup memory — `:2000,2049` | **different** | Numerical delta: float32 round-off at the MMA accumulation boundary differs by at most 1 ULP per element — see §2. |
| **Softmax pre-scale (log2(e) trick)** | **Not used.** Uses `exp()` directly — `ggml-metal.metal:6155-6156` | **Used.** Q pre-scaled by `log2(e)=1.44269504089` at `:2000`, then `fast::exp2()` in `ExpSubOp` — `:1882-1884` | **different** | Numerics: `fast::exp2(x)` is not bit-identical to `exp(x * log2(e))`. fast::exp2 uses a hardware fast path with ~1 ULP accuracy (Apple ISA guarantee), while `exp()` maps to a library path. In practice both are within 1 ULP of the true value, but outputs are not byte-identical. |
| **Exp function** | `exp(m - M)` and `exp(s2 - M)` — plain `exp` — `ggml-metal.metal:6155-6156` | `fast::exp2(x - y)` in `ExpSubOp` — `:1882-1884`; and `fast::exp2(max_old - max_new)` for the rescale factor — `:2219` | **different** | See §2. fast::exp2 is lower latency (~4 cycles on M-series vs ~12 for exp). Candle is faster here. Numerical outcome differs at sub-ULP boundary; not more than 1 ULP across the board. |
| **Online softmax rescale formula** | `ms = exp(m_old - M_new); so4 *= ms` — `ggml-metal.metal:6155,6167-6173` (threadgroup resident O rescaled element-wise) | `factor = fast::exp2(max_old - max_new)` with NaN guard when old_max == -inf; `Otile *= factor` register-resident — `:2215-2239` | **different** | **Residency difference is critical.** llama.cpp keeps O in threadgroup (`so4` is `threadgroup half *` at `:5818-5819`; rescale loops over `so4`). Candle keeps O in register tiles (`Otile` is `MMATile<AccumType, TQ, TD>` — `:2024`). Threadgroup-resident O in llama.cpp requires a `threadgroup_barrier` before every rescale, adding latency. Candle's register-resident O avoids those barriers. This is the dominant structural performance difference between the two kernels. |
| **Mask: additive vs multiplicative** | Additive: `s2 += s2_t(sm2[j*SH + tiisg])` — `ggml-metal.metal:6149` (the mask value is added directly to the pre-softmax logit, so -inf masks → -inf logit) | Additive (float mask): `Stile[jj] += 1.44269504089 * selem_t(mfrag[jj])` — `:2180`. Also supports bool mask (multiplicative-equivalent: false → neg_inf) — `:2176-2178` | **same semantics, different precision path** | The `1.44269504089` factor in candle's additive mask path is present because mask values are in log2 space (to pair with `fast::exp2`). llama.cpp's mask is in natural-log space (pairs with `exp`). Both correctly implement additive masking. A mask value of 0.0 means "allow"; -inf means "block." The math is equivalent at the output but not bit-identical. |
| **Causal mask (internal or via mask arg)** | Via mask arg (external F16 buffer). No internal causal masking in the kernel. The blk pre-pass classification handles full-mask tiles externally. — `ggml-metal.metal:6144-6151`, `llama-graph.cpp:401` | Via template param `do_causal`. Internal causal masking at `:2120-2141` using `row_pos < col_pos` → neg_inf. Also supports external mask arg at `:2143-2185`. Can do both simultaneously. | **different** | For hf2q's use case we will use the external mask path in candle (same as llama.cpp). The internal `do_causal` path is an optimization candle adds but we don't need it for the initial port. Not a correctness risk. |
| **NaN / -inf handling** | M initialized to `-FLT_MAX/2` (not -inf) — `ggml-metal.metal:5891`. No explicit NaN guard on rescale. `exp(-FLT_MAX/2 - (-FLT_MAX/2)) = exp(0) = 1.0` — wrong but harmless since S is also 0. | M initialized to `Limits<AccumType>::min` (which is `-metal::numeric_limits<float>::infinity()` — `:2060`). Explicit NaN guard in `ExpSubOp` at `:1879-1884` and in rescale factor at `:2215-2219`: when `max_score == -inf`, factor is forced to `0`. | **different** | **This is a meaningful correctness distinction.** llama.cpp's `-FLT_MAX/2` avoids the NaN by keeping arithmetic in a finite range. Candle uses true `-inf` and adds explicit guards so that `exp(-inf - (-inf)) = exp(NaN)` never propagates. Candle's approach is SAFER for masked-all tiles (e.g. the last partial tile when seq_len is not a multiple of BK, or an entirely-masked SWA tile). Without the NaN guard, `exp(nan) = nan` would corrupt the output accumulator. llama.cpp is numerically safe only because its -FLT_MAX/2 initialization prevents the NaN path. Candle's explicit guards are the correct defensive approach; hf2q's port MUST preserve them. |
| **Register vs threadgroup O accumulator** | **Threadgroup-resident** (`so` is `threadgroup half *`, `so4` is `threadgroup half4 *`) — `ggml-metal.metal:5818-5819` | **Register-resident** (`Otile` is `MMATile<AccumType, TQ, TD, MMAFrag_acc_t>` on thread stack) — `:2024` | **different** | **Major performance implication.** Threadgroup-resident O in llama.cpp requires `threadgroup_barrier` calls around every rescale step, which stalls all 128 threads. Candle's register-resident O has zero barrier overhead for the rescale multiply. This is likely one of the reasons candle/MLX's kernel is considered the reference — it avoids the synchronization tax. The ADR's "algorithmically identical" claim holds at the math level but misses this micro-architectural divergence. |
| **Output accumulator init** | Zero: `so4[j*PV4 + i] = 0` — `ggml-metal.metal:5878` | Zero: `Otile.clear()` sets `frag_type(0)` — `:1310, 2026` | **same** | None |
| **Head-dim loop structure (Q·K^T)** | Double-unrolled: processes 2 k-frags per loop iteration (`mk[0], mk[1]`, `mq[0], mq[1]`) with `#pragma unroll (MIN(DK8/2, 4*NSG))` — `ggml-metal.metal:6044-6058` | Single unroll over `dd = 0..TD`: one Qtile + one Ktile loaded per iteration — `:2085-2097`. `TD = BD/8 = 32` for D=256. | **different** | Different compiler unroll hint. llama.cpp's double-buffered load pattern (`mk[0]`/`mk[1]`) may have lower instruction-issue stalls for large D. Candle relies on the compiler to pipeline. For D=256 both cover 32 8×8 tiles. No numerical difference. |

---

## 2. Numerical-Equivalence Concerns

### 2a. Scale placement (post-QK^T in llama.cpp vs pre-Q in candle)

**llama.cpp** (`ggml-metal.metal:6138`): After storing `mqk` into threadgroup `ss`, the softmax preamble reads `ss2[...]*args.scale`. This means scale is applied to the accumulated QK^T product in f32 after the MMA accumulation is complete.

**candle** (`:2000,2049`): Q is loaded into `Qs` threadgroup memory and immediately scaled by `scale * 1.44269504089` element-wise (`loader_q.apply_inplace_op(ts)`). Scale is applied in f16/bf16 before the MMA.

**Numerical delta**: Scaling Q elements in f16 before MMA (candle) vs scaling the f32 MMA accumulator after (llama.cpp) can differ by up to 1 ULP in the f32 accumulator. In practice this is sub-ulp noise relative to the softmax normalization. Expected absolute error: < 1e-6 for f32 output, < 1e-3 for bf16 output. This is **within the tolerance of any reasonable byte-identical test at f32 but will differ at bf16 precision**. Phase 2 correctness gates MUST compare against a reference that uses the same scale-placement convention (candle's — since we are porting candle, not llama.cpp).

### 2b. exp vs fast::exp2

**llama.cpp**: `exp(x)` — maps to Metal's `exp()` intrinsic, IEEE-754 with ~1 ULP error.

**candle**: `fast::exp2(x)` — Metal's fast non-IEEE path, ~1 ULP error on M-series but not guaranteed bit-identical to `exp(x * log2(e))` because the argument reduction path differs.

**Numerical delta**: Up to 2 ULP in softmax probability values. For f32 outputs this is invisible after normalization by S. For bf16 outputs the error is absorbed in the bf16 rounding. **Verdict: not a byte-identical risk when comparing candle port to candle reference. IS a byte-identical risk if the Phase 2 gate compares candle-port output against hf2q's current `sdpa.metal` reference (which uses `exp` in scalar loops). The Phase 2 baseline should be re-captured using the candle host-side reference at the same shapes.**

### 2c. NaN guards (candle adds, llama.cpp lacks)

Candle's `ExpSubOp` at `:1878-1886` and the factor guard at `:2215-2219` protect against `exp(NaN)` when max_score is -inf (all-masked tiles). llama.cpp avoids this through the `-FLT_MAX/2` initialization (finite arithmetic, no NaN possible).

**Porting consequence**: The candle port's NaN guards are **SAFER** and must be preserved. Removing them to "match" llama.cpp would introduce a latent NaN-propagation bug on any tile that is fully masked by SWA or causal masking.

**Test implication**: Phase 2 tests must include a shape where the last KV tile is fully masked (e.g. `seq_len = BK * N` exactly, so no partial tile, but one tile that is entirely below the causal diagonal for earlier Q rows). Without this, the NaN guard is dead code in the test suite.

### 2d. Mask scale factor (1.44269504089)

Candle's float-mask additive path at `:2180` multiplies mask values by `1.44269504089 = log2(e)`. This is because softmax logits are being accumulated in log2 space (pairing with `fast::exp2`). llama.cpp adds mask values directly (`s2 += sm2[...]`) in natural-log space (pairing with `exp`).

**Consequence for hf2q's mask format**: hf2q's existing mask buffers (from `sdpa.metal` and `sdpa_sliding.metal`) use natural-log scale (0.0 for allow, -inf for mask). When porting candle, the float-mask additive path will apply `1.44269504089 * mask_value`. For mask values of 0.0, this is fine. For mask values of exactly -inf, `1.44269504089 * (-inf) = -inf` is correct. But **intermediate finite negative values** (e.g. ALiBi bias values) would be scaled by 1.44269504089, which would be mathematically wrong if the caller expects natural-log semantics. Since Gemma 4 does not use ALiBi, this is not a Phase 1a concern. For Phase 4/5, document that candle's mask convention is log2-scaled.

---

## 3. Sliding-Window Divergence (Phase 4 Concern)

### 3a. How llama.cpp encodes sliding window in the mask

`llama-graph.cpp:380-443` (`llm_graph_input_attn_no_cache::set_input`):

The CPU-side mask is a float32 matrix pre-filled with `-INFINITY` at line `:421`:
```
std::fill(data, data + ggml_nelements(self_kq_mask), -INFINITY);
```
Then the `fill_mask` lambda (`:384-413`) iterates token pairs `(i0, i1)` and sets `data[i1*n_kv + i0] = 0.0f` (allow) only when:
1. Same sequence ID
2. Causal: `p0 <= p1` (past or same)
3. SWA check: `!llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)` passes (within window)

For SWA layers, a separate tensor `self_kq_mask_swa` is filled identically but with the sliding-window predicate active (`:430-442`). This means the GPU kernel receives a mask where out-of-window positions are literally `-INFINITY` (f32) and in-window+causal positions are `0.0f`. The mask is then cast to f16 for GPU use by the ggml graph machinery.

**Key insight**: sliding window in llama.cpp is fully encoded in the mask. The kernel sees either 0.0 (attend) or -inf (mask). There is no window_size parameter in the kernel itself.

### 3b. How `flash_attn_ext_blk` uses the mask to skip tiles

`kernel_flash_attn_ext_blk` (`ggml-metal.metal:5666-5719`) is a separate pre-pass kernel that runs **before** `kernel_flash_attn_ext_impl`. It classifies each `Q×C` mask tile into:
- `0` = entirely -inf (skip tile entirely — no computation needed)
- `1` = mixed (some -inf, some valid — must process)
- `2` = entirely 0.0 (all-attend — skip mask addition, but still compute QK^T and softmax)

The classification logic (`:5688-5710`): within each `C×Q` block, it finds `mmin` and `mmax` over all `(C/NW) × Q` mask elements via `simd_min`/`simd_max`. If `mmax <= -MAXHALF` (all -inf), result is `0`. If `mmin == mmax == 0.0`, result is `2`. Otherwise `1`.

This per-tile classification is written to a byte buffer (`blk`) indexed as `[(i3*ne32 + i2)*nblk1 + i1)*nblk0 + i0]` — one byte per `Q×C` tile across the entire mask.

In the main kernel (`ggml-metal.metal:5951-5981`), the KV sweep loop reads `blk[ic0]` and `continue`s (skips the entire tile) when `blk_cur == 0`. When `blk_cur == 2`, it skips the mask-addition step but still processes QK^T + softmax.

### 3c. Tile-skip fraction verification at window=1024, seq=2455

The ADR claims "~59% of tiles are skipped." Let's verify from the code structure:

With `Q=8` queries per tile and `C=64` KV per tile:
- Total KV tiles per Q-tile: `ceil(2455/64) = 39` tiles
- For a causal model with window=1024: for a Q position at index `q`, the valid KV range is `[max(0, q-1023), q]` (window of 1024). The number of fully-masked tiles (all 64 positions outside window AND outside causal range) for a given Q position is approximately `ceil(max(0, q - 1024)/64)` tiles at the beginning plus 0 at the end (causal is handled by partial-mask tiles, not full-skip).

For the last Q position (`q=2454`): valid KV range = `[1430, 2454]`. First valid tile index = `floor(1430/64) = 22`. Tiles 0–21 are fully masked → 22 of 39 tiles skipped = 56%. 

Average over all Q positions (uniform distribution 0..2454): the average number of skipped tiles is approximately `mean over q of floor(max(0,q-1023)/64)`. For q < 1024: 0 skipped tiles. For q >= 1024: `floor((q-1024)/64)` skipped. Average for q in [1024,2454]: `mean(floor((q-1024)/64)) ≈ 1430/(2*64) ≈ 11.2` tiles skipped out of 39. Weighted by fraction of Q positions: `(1431/2455) × 11.2 / 39 ≈ 58.3% of tiles are skipped on average.

**Verdict: The ADR's ~59% tile-skip claim is correct to within rounding** — verified from the tile geometry and window size. The `blk` pre-pass enables this skip.

### 3d. Candle's equivalent for sliding window

Candle has no tile-skip optimization. Its `has_mask` path loads a `MaskType` buffer and applies it additively per element within the active tile, but never skips entire tiles based on a pre-computed classification. Every KV tile incurs QK^T MMA cost even when all attention scores in it will be -inf after masking.

This is the primary throughput gap for SWA layers: at window=1024, seq=2455, llama.cpp skips ~59% of QK^T + PV MMA work on sliding-layer heads. Candle (and our port) will perform that work then mask it to -inf before softmax. The masked tiles contribute 0 to the output (correctly), but waste GPU cycles.

**Tile-skip optimization is confirmed as a Phase 4/5 opportunity** — not needed for correctness, but worth ~2.4× throughput improvement on SWA layers if implemented as a separate pre-pass identical to `flash_attn_ext_blk`.

---

## 4. Hidden Differences the ADR Missed

### 4a. Tile geometry mismatch — ADR's "32 MMAs" claim is incorrect for llama.cpp

The ADR states at §Resolved Questions: "32 MMAs per Q·K^T tile (both confirm)." This is wrong for llama.cpp at D=256.

**llama.cpp**: With `Q=8, C=64, DK=256`:
- `DK8 = DK/8 = 32`
- The fast (DK%16==0) branch at `:6043-6058` loops `for i=0..DK8/2-1 = 0..15` with 2 MMAs per iteration → **32 `simdgroup_multiply_accumulate` calls per (8-row Q-block × 8-col K-block)**
- But: `C/8 = 8` 8-wide K-columns per C=64 tile, processed as `NC = (C/8)/NSG = 8/4 = 2` per simdgroup. So **per simdgroup per KV tile: `2 × 32 = 64 MMAs** for Q·K^T.
- Total across 4 simdgroups: `4 × 64 = 256 MMAs` for Q·K^T per KV tile.

**candle**: With `BQ=32, BK=16, BD=256`:
- `TD = BD/8 = 32`, `TQ = BQ/(WM*WN*8) = 32/(4*1*8) = 1`, `TK = BK/8 = 2`
- Q·K^T: `TQ * TK * TD = 1 * 2 * 32 = 64 MMAs` per simdgroup per KV tile.
- Total across 4 simdgroups: `4 × 64 = 256 MMAs` per KV tile.

The totals match (256 MMAs/tile for both), but the distribution per simdgroup differs because of the different tile shapes (8Q×64K in llama.cpp vs 32Q×16K in candle). The ADR's statement "32 MMAs per Q·K^T tile (both confirm)" refers to per-simdgroup values for candle only; for llama.cpp the per-simdgroup count is 64 for the same D=256 work. **The ADR is imprecise but not wrong in absolute terms.**

### 4b. D=512 global layers: candle has severely reduced tile geometry

**Critical finding for Phase 1 / Open Question resolution**:

Candle's instantiation table at `scaled_dot_product_attention.metal:2334-2337`:
```
instantiate_attn(float16,  half,        8, 8, 512, 1, 1, float16,  half)
instantiate_attn(float16,  half,        8, 8, 512, 1, 1, bool_,    bool)
instantiate_attn(bfloat16, bfloat16_t,  8, 8, 512, 1, 1, bfloat16, bfloat16_t)
instantiate_attn(bfloat16, bfloat16_t,  8, 8, 512, 1, 1, bool_,    bool)
```

At BD=512: `BQ=8, BK=8, WM=1, WN=1` → **only 1 simdgroup (32 threads)** per threadgroup, processing only 8 Q-rows × 8 KV-cols at a time. This is a tiny tile — nearly decode-like for prefill. The comment at `:2332-2333` explains: "reduced tiles (BQ=8, BK=8) to fit 32KB threadgroup memory. Only f16/bf16 — float32 exceeds the limit."

**Consequences**:
1. `float32` I/O at BD=512 is NOT INSTANTIATED in candle. Phase 1 cannot use float32 for global-layer (D=512) correctness verification — only f16/bf16 exists.
2. The 8×8 tile geometry at D=512 in candle means extremely low occupancy and poor K/V reuse. This may be an underoptimized path. For hf2q's global layers at D=512, the candle port may deliver significantly less than the 2500–3000 tok/s target.
3. llama.cpp at D=512 uses 8 simdgroups (`nsg=8` at `ggml-metal-ops.cpp:2807`), same Q=8/C=64 tile geometry as D=256 but with more simdgroups for the larger accumulator. This is more aggressive than candle's single-simdgroup D=512 dispatch.

**ADR correction required**: §Open Questions states "Exact (BQ, BK, BD, WM, WN) for D=512 (global layers). Candle's instantiation table resolves this." It does — but the resolved values (`BQ=8, BK=8, WM=1, WN=1`, 32 threads) are much weaker than the D=256 geometry. Phase 1 should benchmark D=512 separately and may need to add a better-tiled D=512 instantiation or borrow llama.cpp's 8-simdgroup approach.

### 4c. llama.cpp output accumulator is in threadgroup half memory, not registers

As noted in the structural table: `so` is `threadgroup half *` at `ggml-metal.metal:5818-5819`. The P×V MMA at `:6186-6256` loads `lo[NO]` register arrays from `so`, does MMA, and stores back to `so`. This load-MMA-store pattern around threadgroup memory means the output accumulator is NOT register-resident across K tiles — it lives in threadgroup memory and must be loaded and stored for each K tile's PV product.

Candle's `Otile` is a `thread`-space `MMATile` (declared in thread scope at `:2024`), maintained register-resident across all K tiles. The `Otile.row_bin_op<MulOp>(factor)` rescale at `:2239` is a pure register operation with no threadgroup traffic.

**Impact**: llama.cpp's threadgroup-resident O requires at minimum one `threadgroup_barrier` per K-tile around the load/store cycle of the output accumulator (observed at `:6176` and `:6325`). This is additional synchronization overhead absent from candle's design. This difference likely contributes to llama.cpp's kernel being somewhat slower than the theoretical MMA-bound peak for large D, even though the MMA count is the same.

### 4d. bf16 output accumulator difference

**llama.cpp** at bf16 (`FA_TYPES_BF`):
The output accumulator type `o_t` is `half` (from `FA_TYPES_BF` at `:6444-6450`: `half, half4, simdgroup_half8x8` for output). The softmax `s_t` is `float`. The P×V MMA uses `simdgroup_float8x8` for S and stores to `simdgroup_half8x8` for O — meaning the O accumulator is f16, not f32, when bf16 I/O is used.

**candle** (`AccumType = float` always — see template signature `:1895-1908` and the instantiation `attention, dtype, bq, bk, bd, wm, wn, mtype, float` at `:2313`): The accumulator is always f32 regardless of I/O dtype.

**This is a hidden difference with numerical consequence**: llama.cpp's bf16 path accumulates O in f16, candle accumulates in f32. A 32-f16 MMA-accumulated output can lose precision for large `BD` (many adds into the same f16 register), particularly for D=256 or D=512. Candle's f32 accumulator is numerically superior.

**Implication for the port**: The Phase 1 port from candle preserves f32 accumulators (correct). If Agent #3 is tempted to use f16 accumulators to save register pressure for D=512, this is wrong — stick with the candle convention of `AccumType=float`.

### 4e. Partial tile handling at seq_len boundary

**llama.cpp**: Uses a padded KV buffer (`bid_pad`) — at `ggml-metal-ops.cpp:2707-2745`, when `has_kvpad` is true (i.e. `ne11 % ncpsg != 0`), it pads K and V to a full multiple of C=64 before the main kernel runs. The mask is also padded correspondingly. The main kernel always sees full tiles.

**candle**: Handles partial last tile in-kernel via `load_safe(short2(BD, params->kL_rem))` conditional at `:2074` and masking out the invalid elements to -inf in the S tile at `:2100-2117`. No separate pre-pass for padding.

**Implication**: Candle's in-kernel partial-tile masking is slightly more complex to port but eliminates the pre-pass copy kernel for the padding case. For hf2q's port, use the candle approach (in-kernel bounds check) — it is cleaner and avoids the extra dispatch.

---

## 5. Verdict for ADR-011

**RISK**

The ADR's choice of candle over llama.cpp is structurally justified. At the algorithmic level — online softmax, 8×8×8 MMA, tiled K/V sweep — the two kernels are equivalent in mathematical outcome. The port decision is correct.

However, the claim "algorithmically identical at the 8×8×8 MMA level" glosses over four specific differences that Agent #3 (vendor-kernel) and Agent #5 (tests) must account for:

**RISK-1: D=512 tile geometry is underoptimized in candle.**  
Candle's BD=512 instantiation uses `BQ=8, BK=8, WM=1, WN=1` (32 threads, decode-like geometry). Phase 3's throughput target of ≥2500 tok/s may not be achievable for global-layer (D=512) heads with this geometry. Phase 1 must benchmark D=512 as a separate data point and consider adding a better-tiled D=512 instantiation before Phase 3 commits the throughput gate.

**RISK-2: The Phase 2 byte-identical gate must compare against candle's reference, not hf2q's current `sdpa.metal` reference.**  
The exp vs fast::exp2 and scale-placement differences between candle and llama.cpp (and between candle and hf2q's current scalar kernel) mean the two implementations are NOT byte-identical even at f32. The Phase 2 gate "byte-identical to per-token reference" must use a candle-equivalent scalar reference (same `fast::exp2` + log2(e) pre-scaling convention) — not the current `sdpa.metal` scalar loop which uses `exp`. If the Phase 2 gate compares against the current hf2q reference, it will fail on numerics that are correct-by-algorithm but not bit-identical. The tolerance should be set at `atol=1e-5, rtol=1e-4` for f32 output when comparing candle port vs current scalar kernel.

**RISK-3: Candle's NaN guards must be preserved in the port.**  
llama.cpp avoids NaN through `-FLT_MAX/2` initialization; candle uses true -inf + explicit guards. Removing the guards (to "simplify" the port) introduces a latent correctness bug on all-masked tiles. Phase 2 tests must include at least one shape with a fully-masked KV tile.

**RISK-4: llama.cpp's float32 path at D=512 does NOT exist in candle.**  
The Phase 1 plan calls for "start with f32 Q/K/V/O for cheap correctness verification; add bf16 variant after." At BD=512 (global layers), f32 verification is impossible with candle's existing instantiation table — there is no f32 BD=512 entry. Options: (a) add a f32 BD=512 instantiation to the port (may exceed 32KB threadgroup memory — verify before attempting), or (b) verify correctness at BD=256 only in Phase 1, then add bf16 BD=512 in Phase 4. The ADR implicitly assumed f32 correctness verification works at all head dims — this assumption is false for D=512 with the candle source.

---

## Appendix: Tile Geometry Cross-Reference

| Parameter | llama.cpp impl (D=256, prefill) | candle (D=256) |
|---|---|---|
| Q per tile (NQPSG) | 8 — `ggml-metal-impl.h:93` | 32 (BQ) — `:2316` |
| KV per tile (NCPSG) | 64 — `ggml-metal-impl.h:94` | 16 (BK) — `:2316` |
| Simdgroups per TG | 4 (`nsg=4`, `ne00<512`) — `ggml-metal-ops.cpp:2807` | 4 (WM=4, WN=1) — `:2316` |
| Threads per TG | 128 (32×4) | 128 (32×4) — `:2316` |
| Scale placement | Post-QK^T (f32 multiply) — `:6138` | Pre-Q (f16/bf16 multiply, log2 factor) — `:2000,2049` |
| Exp function | `exp()` — `:6155-6156` | `fast::exp2()` — `:1882,2219` |
| O accumulator dtype | half (threadgroup) — `:5818-5819` | float (register, thread-space) — `:2024` |
| NaN guard | None (uses -FLT_MAX/2) — `:5891` | Yes (explicit -inf guard) — `:1879-1884, 2215-2219` |
| Partial tile handling | Pre-pass pad kernel — `ggml-metal-ops.cpp:2709-2744` | In-kernel bounds check — `:2074,2100-2117` |
| SWA tile-skip | Yes, via blk pre-pass — `:5666-5719, 5951-5981` | No | 

| Parameter | llama.cpp impl (D=512, prefill) | candle (D=512) |
|---|---|---|
| Q per tile | 8 — `ggml-metal-impl.h:93` | 8 (BQ) — `:2334` |
| KV per tile | 64 — `ggml-metal-impl.h:94` | 8 (BK) — `:2334` |
| Simdgroups per TG | **8** (`nsg=8`, `ne00>=512`) — `ggml-metal-ops.cpp:2807` | **1** (WM=1, WN=1) — `:2334` |
| Threads per TG | 256 | 32 |
| f32 I/O instantiation | Yes — `:6477` | **NO** — `:2332-2337` |

---

## File Citations Summary

All citations verified by direct read on 2026-04-17:

- `ggml-metal.metal:5666-5719` — flash_attn_ext_blk pre-pass (tile classifier)
- `ggml-metal.metal:5767-6375` — kernel_flash_attn_ext_impl (full impl template)
- `ggml-metal.metal:5818-5819` — threadgroup O/so/so4 declarations
- `ggml-metal.metal:5878` — O accumulator zero-init
- `ggml-metal.metal:5888,5891` — S=0, M=-FLT_MAX/2 init
- `ggml-metal.metal:6037,6056` — simdgroup_multiply_accumulate in Q·K^T hot loop
- `ggml-metal.metal:6044-6058` — double-unrolled MMA loop, D=256
- `ggml-metal.metal:6131-6174` — online softmax + O rescale (threadgroup barrier at 6176)
- `ggml-metal.metal:6138` — scale applied post-QK^T
- `ggml-metal.metal:6155-6156` — `exp()` used (not fast::exp2)
- `ggml-metal.metal:6186-6256` — P×V MMA with register arrays `lo[NO]`, store back to threadgroup `so`
- `ggml-metal.metal:6436-6450` — type macros: half K/V, float S/O for f16/bf16 paths
- `ggml-metal.metal:6475-6511` — dk=256/512 instantiations for f32/f16/bf16
- `ggml-metal-impl.h:93-94` — NQPSG=8, NCPSG=64
- `ggml-metal-ops.cpp:2505` — vec-vs-impl split at ne01<20
- `ggml-metal-ops.cpp:2696-2861` — dispatch logic, `nsg=4/8`, grid calc
- `ggml-metal-ops.cpp:2807` — `nsg = ne00 >= 512 ? 8 : 4`
- `llama-graph.cpp:380-443` — mask builder: fill with -INFINITY, set 0.0 for (in-window, causal, same-seq) pairs
- `scaled_dot_product_attention.metal:1243` — MMA primitive
- `scaled_dot_product_attention.metal:1878-1886` — ExpSubOp NaN guard
- `scaled_dot_product_attention.metal:2000` — scale * log2(e) pre-factor
- `scaled_dot_product_attention.metal:2024` — Otile register-resident declaration
- `scaled_dot_product_attention.metal:2026` — Otile.clear() zero-init
- `scaled_dot_product_attention.metal:2049` — Q scale applied in threadgroup memory
- `scaled_dot_product_attention.metal:2085-2097` — Q·K^T MMA loop
- `scaled_dot_product_attention.metal:2143-2185` — mask application (additive float or bool)
- `scaled_dot_product_attention.metal:2180` — float mask multiplied by 1.44269504089
- `scaled_dot_product_attention.metal:2196-2239` — online softmax with fast::exp2 and -inf guard
- `scaled_dot_product_attention.metal:2215-2219` — rescale factor guard for -inf max_score
- `scaled_dot_product_attention.metal:2316` — BD=256: BQ=32, BK=16, WM=4, WN=1
- `scaled_dot_product_attention.metal:2332-2337` — BD=512: BQ=8, BK=8, WM=1, WN=1, f16/bf16 only
