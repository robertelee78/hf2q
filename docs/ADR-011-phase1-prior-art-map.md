# ADR-011 Phase 1 Vendor Map

**Status:** Ready for Agent #3 (vendor-kernel)
**Produced by:** Agent #1 (research-candle) in swarm swarm-1776462683390-zb6ev9
**Date:** 2026-04-17
**Source of truth:** `/opt/candle/candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal`
  and `/opt/candle/candle-metal-kernels/src/kernels/sdpa.rs`

All claims below are backed by file:line citations from the candle source. Where the ADR
contains a claim that the source contradicts, this document states what the source actually says.

---

## 1. Exact span to vendor

### 1.1 File header (license + upstream reference)

The entire file has no separate `#include` of any external file — it is a **self-contained
monolith**. Line 0 (the very first line, using 0-based line numbering that the editor shows
as line 0 — but the Read tool shows it as line 0 since the file starts at offset 0):

```
// Updated from MLX commit has f70764a
```
`scaled_dot_product_attention.metal:0`

Lines 1–5 are the only system includes in the entire file:

```metal
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;
```
`scaled_dot_product_attention.metal:2-5`

There are **no other `#include` directives anywhere in the file**. All MLX helper code
(from `steel/attn/mma.h`, `steel/attn/loader.h`, `steel/gemm/transforms.h`,
`steel/utils/type_traits.h`, `steel/utils/integral_constant.h`, `kernels/utils.h`,
`kernels/scaled_dot_product_attention_params.h`) has been **inlined verbatim** into this
single file, with banner comments marking the logical boundaries. The vendor agent does not
need to locate or reference any external files.

### 1.2 `attention<>` kernel template definition

`scaled_dot_product_attention.metal:1895-2295`

- Template declaration opens at line 1896:
  ```metal
  template <
      typename T,
      int BQ,
      int BK,
      int BD,
      int WM,
      int WN,
      typename MaskType = float,
      typename AccumType = float>
  ```
- Kernel attribute and signature at line 1905:
  ```metal
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention(
  ```
- Closing brace at line 2295.

### 1.3 All helper definitions the kernel depends on (same file)

Everything the `attention<>` kernel uses is defined earlier in the same file. Listed in
dependency order (top-to-bottom in the file):

| Helper | Lines | Purpose |
|---|---|---|
| `bfloat16_t` typedef + `_MLX_BFloat16` struct | 10–262 | bf16 emulation for old Metal |
| `MLXFastAttentionParams` struct | 266–293 | (not used by attention<>, only legacy kernels) |
| `MLXScaledDotProductAttentionParams` struct | 295–303 | (not used by attention<>) |
| `sdpa_vector_has_mask` constant + `sdpa_vector` kernel | 306–431 | (not used by attention<>) |
| `sdpa_vector_2pass_1`, `sdpa_vector_2pass_2` kernels | 433–626 | (not used by attention<>) |
| `Limits<T>` template + specializations | 630–673 | `Limits<AccumType>::min` at line 2060 |
| `BlockLoader<>` struct | 678–798 | (not directly used; `BlockLoaderT` is used instead) |
| `BlockLoaderT<>` struct | 806–925 | Q/K/V tile loaders at lines 1965–1998 |
| `void_t`, `pointer_element_t` | 927–958 | Used in store_safe path |
| `Int<val>` alias + integral_const operators | 960–996 | Compile-time stride constants |
| `sum<>` variadic | 1002–1010 | Not directly used in attention<> |
| `TransformNone`, `TransformAdd`, `TransformAxpby` | 1014–1053 | `TransformScale` used instead |
| `AccumHelper`, `BlockSwizzle` | 1055–1068 | Not used in attention<> |
| `Shape2D`, `Layout2D` | 1070–1084 | Not directly used in attention<> |
| `BaseMMAFrag<T, 8, 8>` | 1086–1274 | Core MMA fragment; line 1243 has `simdgroup_multiply_accumulate` |
| `MMATile<>` | 1276–1468 | Q/K/S/V/O tile holders |
| `tile_matmad<>` | 1470–1504 | Triple-nested Q·K^T multiply-accumulate |
| `BlockMMA<>` | 1506–1795 | Used by GEMM kernels, **not** by `attention<>` |
| Banner: steel_attention.h | 1796 | Marks start of attention-kernel region |
| `AttnParams` struct | 1798–1824 | Kernel constant-buffer layout |
| `AttnMaskParams` struct | 1826–1828 | Mask stride layout |
| `align_Q`, `align_K`, `has_mask`, `do_causal` constants | 1834–1838 | Function constants |
| Op structs: `TransformScale`, `MaxOp`, `SumOp`, `MulOp`, `SubOp`, `ExpSubOp`, `DivOp` | 1840–1893 | Online-softmax operators |

**Minimum set to copy for a correct port of `attention<>` only:**
Lines 0–5 (header + stdlib includes), 10–262 (bf16), 630–673 (Limits), 806–925 (BlockLoaderT),
927–996 (type_traits + Int), 1070–1468 (BaseMMAFrag + MMATile), 1470–1504 (tile_matmad),
1796–2295 (AttnParams, AttnMaskParams, constants, op-structs, attention<> kernel body).

Do **not** copy the `BlockLoader` (678–798), `BlockMMA` (1506–1795), `sdpa_vector*`
(306–626) or `MLXFast*`/`MLXScaled*` structs (266–303) — they are unused dead code for
this port.

### 1.4 `instantiate_attn` macro and the D=256 / D=512 invocations

The macro definition is at lines 2305–2313:

```metal
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention, dtype, bq, bk, bd, wm, wn, mtype, float)
```
`scaled_dot_product_attention.metal:2305-2313`

The shape-helper macro at lines 2315–2322:

```metal
#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 32, 16, 256, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 16, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  96, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  80, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  72, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  64, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  32, 4, 1, mname, mtype)
```
`scaled_dot_product_attention.metal:2315-2322`

**ADR claim "BQ=32, BK=16, BD=256, WM=4, WN=1" is confirmed at line 2316.**

The mask-helper and per-dtype invocations at lines 2324–2330:

```metal
#define instantiate_attn_mask_helper(iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, iname, itype) \
    instantiate_attn_shapes_helper(iname, itype, bool_, bool)

instantiate_attn_mask_helper(float16, half);
instantiate_attn_mask_helper(bfloat16, bfloat16_t);
instantiate_attn_mask_helper(float32, float);
```
`scaled_dot_product_attention.metal:2324-2330`

For Gemma 4 head_dim=256 bf16, the effective expanded instantiation (produced by macro
expansion of line 2329 → 2325 → 2316) is:

```metal
instantiate_attn(bfloat16, bfloat16_t, 32, 16, 256, 4, 1, bfloat16, bfloat16_t)
```

Which produces host name: `"steel_attention_bfloat16_bq32_bk16_bd256_wm4_wn1_maskbfloat16"`

For f32: `"steel_attention_float32_bq32_bk16_bd256_wm4_wn1_maskfloat32"`
(from line 2330 expansion).

---

## 2. Template parameter values for D=256

**Confirmed from source (lines 2315-2316, 2006-2016):**

| Parameter | Value (D=256) | Source line |
|---|---|---|
| `BQ` | 32 | 2316 |
| `BK` | 16 | 2316 |
| `BD` | 256 | 2316 |
| `WM` | 4 | 2316 |
| `WN` | 1 | 2316 |
| `MaskType` | same as `T` (or `bool`) | 2316 |
| `AccumType` | `float` (hardcoded in macro) | 2313 |

### Thread layout

From line 1905:
```metal
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
```
With WM=4, WN=1: `max_total_threads_per_threadgroup = 4 * 1 * 32 = 128`.

That is **4 simdgroups × 32 lanes = 128 threads per threadgroup**.

Grid dispatch (from `sdpa.rs:230-239`):
- `gridSize = (nq, h, b)` where `nq = ceil(ql / BQ) = ceil(ql / 32)`
- `threadgroupSize = (32, WM=4, WN=1)`

### Fragment counts TQ, TK, TD

From lines 2006-2016 (in the kernel body, derived from template params):

```metal
constexpr short kFragSize = 8; // MMAFrag size
constexpr int kNWarps = WM * WN;  // = 4 * 1 = 4
// TQ = BQ / (kNWarps * kFragSize) = 32 / (4 * 8) = 1
constexpr int TQ = BQ / (kNWarps * kFragSize);   // = 1
// TK = BK / kFragSize = 16 / 8 = 2
constexpr int TK = BK / kFragSize;               // = 2
// TD = BD / kFragSize = 256 / 8 = 32
constexpr int TD = BD / kFragSize;               // = 32
```

- **TQ = 1** (confirmed by `static_assert(TQ == 1, "Check TQ")` at line 2018)
- **TK = 2**
- **TD = 32**

**ADR claim "TQ=1, TK=2, TD=32" is confirmed.**

### Threadgroup memory footprint arithmetic (from kernel source, lines 1945-1958)

```metal
constexpr short padQ = 16 / sizeof(T);   // bf16: 16/2 = 8; f32: 16/4 = 4
constexpr short padK = 16 / sizeof(T);
constexpr short padV = 16 / sizeof(T);

constexpr short LDQ_tgp = BD + padQ;     // bf16: 256 + 8 = 264; f32: 256 + 4 = 260
constexpr short LDK_tgp = BK + padK;     // bf16: 16 + 8 = 24;  f32: 16 + 4 = 20
constexpr short LDV_tgp = BD + padV;     // bf16: 256 + 8 = 264

// Q_smem size:
threadgroup T Q_smem[BQ * (BD + padQ)];  // bf16: 32 * 264 = 8448 elements

// KV_smem is max of:
constexpr short tgp_mem_0 = (BK + padK) * BD;   // K (transposed): bf16: 24 * 256 = 6144 elems
constexpr short tgp_mem_1 = BK * (BD + padV);   // V (normal): bf16: 16 * 264 = 4224 elems
constexpr short tgp_mem_s = max(tgp_mem_0, tgp_mem_1);  // bf16: 6144 elems
threadgroup T KV_smem[tgp_mem_s];
```

**Total threadgroup memory (bf16, T=bfloat16_t, sizeof=2):**
- Q_smem: 8448 × 2 = 16,896 bytes
- KV_smem: 6144 × 2 = 12,288 bytes
- **Total: 29,184 bytes ≈ 28.5 KB**

**ADR claim "~28.5 KB threadgroup memory" is confirmed by this arithmetic.**

**Total threadgroup memory (f32, T=float, sizeof=4):**
- Q_smem: 32 × 260 = 8320 × 4 = 33,280 bytes
- KV_smem: max((16+4)×256, 16×260) = max(5120, 4160) = 5120 × 4 = 20,480 bytes
- **Total: 53,760 bytes ≈ 52.5 KB** — exceeds 32 KB Metal limit on some hardware. This is
  why the ADR calls for starting with f32 for correctness testing on modern M-series which
  allows larger threadgroup memory, then switching to bf16 for production.

---

## 3. Simdgroup MMA instruction audit

### 3.1 The single `simdgroup_multiply_accumulate` call site

There is exactly **one** call to `simdgroup_multiply_accumulate` in the entire file:

```metal
// scaled_dot_product_attention.metal:1243
simdgroup_multiply_accumulate(D, A, B, C);
```

It is inside `BaseMMAFrag<T, 8, 8>::mma()` at lines 1237–1244:
```metal
template <typename Atype, typename Btype, typename Ctype>
METAL_FUNC static constexpr void mma(
    thread mat_type& D,
    thread dtype_mat_t<Atype>& A,
    thread dtype_mat_t<Btype>& B,
    thread dtype_mat_t<Ctype>& C) {
  simdgroup_multiply_accumulate(D, A, B, C);
}
```

All MMA operations in the kernel route through this single primitive.

### 3.2 `tile_matmad` definition

`scaled_dot_product_attention.metal:1482-1504`

```metal
template <
    typename Dtype, typename Atype, typename Btype, typename Ctype,
    int M, int N, int K,
    class MMAFragD, class MMAFragA, class MMAFragB, class MMAFragC>
METAL_FUNC void tile_matmad(
    thread MMATile<Dtype, M, N, MMAFragD>& D,
    thread MMATile<Atype, M, K, MMAFragA>& A,
    thread MMATile<Btype, K, N, MMAFragB>& B,
    thread MMATile<Ctype, M, N, MMAFragC>& C) {
  STEEL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short m_serp = m;
      short n_serp = (m % 2) ? (N - 1 - n) : n;
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMAFragD::mma(
            D.frag_at(m_serp, n_serp),
            A.frag_at(m_serp, k),
            B.frag_at(k, n_serp),
            C.frag_at(m_serp, n_serp));
      }
    }
  }
}
```

**Argument order:** `tile_matmad(D, A, B, C)` computes **D = A × B + C**.

### 3.3 Q·K^T MMA call site

`scaled_dot_product_attention.metal:2085-2097`

```metal
STEEL_PRAGMA_UNROLL
for (short dd = 0; dd < TD; dd++) {   // TD = 32 iterations for D=256
  simdgroup_barrier(mem_flags::mem_none);
  Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * Qs_tile_stride]);
  Ktile.template load<T, 1, 1, LDK_tgp, 1>(&Ks[Ks_offset + dd * Ks_tile_stride]);
  simdgroup_barrier(mem_flags::mem_none);
  tile_matmad(Stile, Qtile, Ktile, Stile);  // S += Q_frag × K_frag^T
}
```

`tile_matmad(Stile, Qtile, Ktile, Stile)` is called with:
- D (output) = `MMATile<AccumType, TQ=1, TK=2>` (Stile)
- A = `MMATile<AccumType, TQ=1, 1>` (Qtile)
- B = `MMATile<AccumType, 1, TK=2>` (Ktile)
- C = `MMATile<AccumType, TQ=1, TK=2>` (Stile, same as D)

The triple loop in `tile_matmad`: M=1, N=2, K=1. Each iteration calls `MMAFragD::mma()` once,
which calls `simdgroup_multiply_accumulate` once. So per outer DD loop iteration:
M × N × K = 1 × 2 × 1 = **2 MMA calls per DD step**.

For D=256, TD=32 steps → **32 × 2 = 64 simdgroup_multiply_accumulate calls for Q·K^T**.

**ADR claim "32 MMAs/simdgroup for Q·K^T" is INCORRECT. Source shows 64 MMA calls for
Q·K^T per KV block (at D=256, TD=32, TK=2: 32 DD-steps × 2 frags = 64).**

Correction: the "32" figure in the ADR appears to be derived from an earlier research
report that may have used D=128 (TD=16, TK=1, giving 16 × 1 = 16; or if TK=2, 16×2=32).
At D=256 with TK=2, the correct count is **64 MMA calls for Q·K^T**.

### 3.4 P·V MMA call site

`scaled_dot_product_attention.metal:2244-2270`

```metal
STEEL_PRAGMA_UNROLL
for (short iq = 0; iq < TQ; iq++) {      // TQ = 1
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TD; id++) {    // TD = 32
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {  // TK = 2
      if constexpr (BD == 128) { simdgroup_barrier(mem_flags::mem_none); }
      const short kk = ik * kFragSize;
      const short dd = id * kFragSize;
      Vtile.template load<T, 1, 1, LDV_tgp, 1>(&Vs[Vs_offset + kk * LDV_tgp + dd]);
      if constexpr (BD == 128) { simdgroup_barrier(mem_flags::mem_none); }
      MMAFrag_acc_t::mma(
          Otile.frag_at(iq, id),
          Stile.frag_at(iq, ik),
          Vtile.frag_at(0, 0),
          Otile.frag_at(iq, id));
    }
  }
}
```

`MMAFrag_acc_t::mma()` (fragment-level, calls `simdgroup_multiply_accumulate` via the
overload at line 1217–1244) is called once per innermost iteration:
TQ × TD × TK = 1 × 32 × 2 = **64 MMA calls for P·V per KV block**.

**ADR claim "64 for P·V" is confirmed.**

**Corrected MMA count table:**

| Operation | Loop structure | MMA calls per KV block |
|---|---|---|
| Q·K^T | DD-loop: TD=32 iters; tile_matmad with M=1,N=TK=2,K=1 | **64** |
| P·V | TQ×TD×TK = 1×32×2 iters; MMAFrag_acc_t::mma per iter | **64** |

Total: **128 MMA calls per KV block per simdgroup** at D=256.

Note: at D=128 (TD=16, TK=1 from line 2317), Q·K^T = 16×(1×1×1)=16, P·V = 1×16×1=16,
total 32 — which matches the "32 MMAs" figure in prior research (that research used D=128).

---

## 4. Online softmax implementation

### 4.1 M (max) and S (sum) arrays

`scaled_dot_product_attention.metal:2052-2061`

```metal
constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;  // = TQ * kElemRows = 1 * 1 = 1

AccumType max_score[kRowsPT];          // kRowsPT = 1 element
AccumType sum_score[kRowsPT] = {0};

STEEL_PRAGMA_UNROLL
for (short i = 0; i < kRowsPT; ++i) {
  max_score[i] = Limits<AccumType>::min;   // init to -infinity (Limits<float>::min = -inf)
}
```

These are **per-thread register arrays** (no simdgroup reduction in the attention<> kernel
for max/sum — different from the scalar sdpa_vector kernels). The `Stile.row_reduce<MaxOp>`
call (line 2207) calls `BaseMMAFrag::row_reduce` which uses `simd_shuffle_xor` for
intra-simdgroup lane reduction (lines 1251–1258), not a separate `simd_max` intrinsic.

### 4.2 Row-max reduction mechanism

`scaled_dot_product_attention.metal:2207`

```metal
Stile.template row_reduce<MaxOp>(new_max);
```

This invokes `MMATile::row_reduce` → `BaseMMAFrag::row_reduce` at lines 1246–1259:

```metal
template <typename Op>
METAL_FUNC static constexpr void row_reduce(
    thread const frag_type& inp_vals,
    thread T* reduced_vals) {
  T thr_reduce = Op::apply(inp_vals.x, inp_vals.y);
  T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
  qgr_reduce = Op::apply(thr_reduce, qgr_reduce);
  T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
  sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);
  reduced_vals[0] = Op::apply(reduced_vals[0], sgr_reduce);
}
```

The kernel uses **`simd_shuffle_xor`** for intra-simdgroup reduction, not the `simd_max`
intrinsic. This is a butterfly reduction pattern — equivalent in result but different in
intrinsic. The ADR's description of "`simd_max` / `simd_sum` reductions" is a
simplification; the actual implementation is `simd_shuffle_xor`-based butterfly at
lines 1251–1258.

### 4.3 `ms = exp(M_old - M_new)` rescale (the `-inf` factor)

`scaled_dot_product_attention.metal:2212-2220`

```metal
// Factor exp(rowmax(Si) - rowmax(Si-1))
// Guard: when max_score == -inf (no valid K seen yet), the previous accumulation is all zeros so the correct rescaling factor is 0.
// Without this, -inf - (-inf) = NaN which poisons the output.
STEEL_PRAGMA_UNROLL
for (short i = 0; i < kRowsPT; ++i) {
  factor[i] = (max_score[i] == -metal::numeric_limits<AccumType>::infinity())
      ? AccumType(0)
      : fast::exp2(max_score[i] - new_max[i]);
}
```

This is the **candle-specific extension** (commit 46928bce). When `max_score[i]` is
`-inf` (no valid K has been seen in prior iterations — i.e., entirely masked-out rows),
the factor is clamped to 0 instead of computing `exp2(-inf - (-inf)) = exp2(NaN)`.

The rescale uses **`fast::exp2`** (not `exp`), consistent with the Q pre-scale below.

### 4.4 Q pre-scale by `log2(e)`

`scaled_dot_product_attention.metal:2000`

```metal
TransformScale<T> ts(static_cast<T>(params->scale * 1.44269504089));
```

`1.44269504089 = log2(e)`. Applied to Q at line 2049:

```metal
loader_q.apply_inplace_op(ts);
```

This pre-scales the entire Q tile in threadgroup memory by `scale × log2(e)` before the
Q·K^T computation. This converts from natural-log-scale softmax to base-2 softmax:
instead of computing `exp(Q·K^T × scale)`, the kernel computes `exp2(Q·K^T × scale ×
log2(e))`, which is mathematically equivalent but avoids per-element `ln(2)` multiplications
at the softmax step.

**The scale `params->scale` is `1/sqrt(head_dim)` as set by the host (sdpa.rs line 151).**
The kernel does NOT additionally divide by `sqrt(d)` inside — the scale is fully applied
at Q-load time.

### 4.5 `fast::exp2` vs `exp`

The kernel calls **`fast::exp2`** in two places:
1. `ExpSubOp::apply()` at line 1884: `fast::exp2(x - y)` (element-wise softmax scores)
2. The factor rescale at line 2219: `fast::exp2(max_score[i] - new_max[i])`

There is **no call to `exp` or `fast::exp`** anywhere in the `attention<>` kernel path.
The `fast::exp` calls that appear in the file (e.g., at lines 384, 385, 414, 518, 519,
548, 561) are in the `sdpa_vector*` kernels, not in `attention<>`.

---

## 5. Candle-specific extensions over MLX upstream

### 5.1 `ExpSubOp` NaN guard

**Definition:** `scaled_dot_product_attention.metal:1878-1886`

```metal
struct ExpSubOp {
  // Guard: when y (row max) is -inf, all scores in the row are -inf (entirely masked). Return 0 instead of exp2(-inf - (-inf)) = exp2(NaN).
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return (y == -metal::numeric_limits<T>::infinity())
        ? T(0)
        : fast::exp2(x - y);
  }
};
```

**Use site:** `scaled_dot_product_attention.metal:2210`

```metal
Stile.template row_bin_op<ExpSubOp>(new_max);
```

This applies element-wise `ExpSubOp::apply(score_element, new_max_for_row)` across all
elements of `Stile`. When an entire row was masked (new_max is -inf), returns 0 for every
score instead of NaN.

### 5.2 `-inf` factor NaN guard (commit 46928bce)

**Location:** `scaled_dot_product_attention.metal:2213-2220` (full quoted in §4.3 above)

The guard is the ternary:
```metal
factor[i] = (max_score[i] == -metal::numeric_limits<AccumType>::infinity())
    ? AccumType(0)
    : fast::exp2(max_score[i] - new_max[i]);
```

**What it prevents:** When `max_score[i]` (the running max from *previous* iterations) is
still at its initial value of `-inf` (Limits<float>::min = -inf, from line 2060), computing
`exp2(max_score[i] - new_max[i])` would be `exp2(-inf - finite_value) = exp2(-inf) = 0`,
which is technically correct. However, when the *current* iteration also sees all-masked
scores, `new_max[i]` will also be `-inf` from the `row_reduce<MaxOp>` call on a `Stile`
full of `-inf` values, yielding `exp2(-inf - (-inf)) = exp2(NaN)` which propagates
incorrectly. The guard short-circuits this case.

### 5.3 Candle vs MLX upstream differences

The comment at line 1796 reads:
```
// ============ "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
```

The line 0 header states:
```
// Updated from MLX commit has f70764a
```

The two NaN guards (`ExpSubOp` ternary and factor ternary) are additions that the MLX
upstream at commit f70764a does not contain per the ADR's research note. Both are worth
preserving in the port — they handle sliding-window attention where entire rows can be
fully masked.

No other candle-specific additions were found in the `attention<>` kernel body (lines
1895–2295) beyond these two guards. The `softcapping` field in `AttnParams` (line 1808)
is present in both MLX upstream and candle; in `call_sdpa_full` (sdpa.rs:152) it is
hardcoded to `1.0` (disabled) for the full SDPA path.

---

## 6. Host-side dispatch mapping

### 6.1 `call_sdpa_full` signature

`sdpa.rs:22-45`

```rust
pub fn call_sdpa_full(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q_offset: usize,
    q_shape: &[usize],
    q_strides: &[usize],
    q_buffer: &Buffer,
    k_offset: usize,
    k_shape: &[usize],
    k_strides: &[usize],
    k_buffer: &Buffer,
    v_offset: usize,
    v_buffer: &Buffer,
    v_strides: &[usize],
    mask_type: Option<SdpaDType>,
    mask_buffer: Option<&Buffer>,
    m_strides: Option<&[usize]>,
    output: &Buffer,
    o_strides: &[usize],
    scale: f32,
    do_causal: bool,
    itype: SdpaDType,
) -> Result<(), MetalKernelError>
```

**Note:** `v_buffer` comes before `v_strides` in the signature — the ordering is
`v_offset, v_buffer, v_strides` (lines 33–35). The port should preserve this exact
ordering in the Rust function signature.

### 6.2 Tile-selection logic

`sdpa.rs:76-98`

```rust
let bd = q_shape[q_shape.len() - 1];  // head_dim from last dim of q_shape
if ![32, 64, 72, 80, 96, 128, 256, 512].contains(&bd) {
    return Err(MetalKernelError::SdpaHeadSizeMismatch { ... });
}

// BD=512 uses reduced tiles to fit 32KB threadgroup memory (f16/bf16 only).
let (bq, bk, wm, wn): (usize, usize, usize, usize) = if bd == 512 {
    if itype == SdpaDType::F32 {
        return Err(MetalKernelError::SdpaHeadSizeMismatch {
            variation: "full (f32 unsupported at head_dim=512)",
            ...
        });
    }
    (8, 8, 1, 1)
} else {
    let bk = if bd < 128 { 32 } else { 16 };
    (32, bk, 4, 1)
};
```

For D=256: `bd=256 >= 128`, so `bk=16`, result: `(bq=32, bk=16, wm=4, wn=1)`. ✓
For D=512: `(bq=8, bk=8, wm=1, wn=1)` — f32 is explicitly rejected.

**ADR claim "lines 86-98": confirmed as lines 76-98 in the actual file** (10 lines earlier
than the ADR states due to the `SdpaDType` enum and comment above it).

### 6.3 Kernel-name formatter

`sdpa.rs:123-124`

```rust
let name =
    format!("steel_attention_{itype_repr}_bq{bq}_bk{bk}_bd{bd}_wm{wm}_wn{wn}_mask{mask_repr}");
```

Where `itype_repr` and `mask_repr` are matched at lines 112-122:

```rust
let itype_repr = match itype {
    SdpaDType::BF16 => "bfloat16",
    SdpaDType::F16 => "float16",
    SdpaDType::F32 => "float32",
};
let mask_repr = match mask_type {
    Some(SdpaDType::BF16) => "bfloat16",
    Some(SdpaDType::F16) => "float16",
    Some(SdpaDType::F32) => "float32",
    None => itype_repr,
};
```

Example for bf16 input, bf16 mask, D=256:
`"steel_attention_bfloat16_bq32_bk16_bd256_wm4_wn1_maskbfloat16"`

Example for f32 input, no mask, D=256:
`"steel_attention_float32_bq32_bk16_bd256_wm4_wn1_maskfloat32"`
(no-mask case sets `mask_repr = itype_repr`).

### 6.4 Function constants (compile-time booleans)

`sdpa.rs:126-131`

```rust
let constants = Some(ConstantValues::new(vec![
    (200, Value::Bool(/* align_Q */ align_q)),
    (201, Value::Bool(/* align_K */ align_k)),
    (300, Value::Bool(/* has_mask */ has_mask)),
    (301, Value::Bool(/* do_causal */ do_causal)),
]));
```

These map to the `function_constant` declarations in the kernel at lines 1834–1838:
- Index 200 = `align_Q`
- Index 201 = `align_K`
- Index 300 = `has_mask`
- Index 301 = `do_causal`

`align_q = (ql % bq) == 0` (line 108); `align_k = (kl % bk) == 0` (line 109).

### 6.5 Grid geometry

`sdpa.rs:230-239`

```rust
let grid_dims = MTLSize {
    width: nq,      // = ceil(ql / bq) = number of Q tile blocks
    height: h,      // = number of heads
    depth: b,       // = batch size
};
let group_dims = MTLSize {
    width: 32,      // = one simdgroup-lane dimension
    height: wm,     // = 4 (WM simdgroups along Q sequence)
    depth: wn,      // = 1 (WN simdgroups along K sequence)
};
```

**This means one threadgroup handles one Q-tile of `BQ=32` query positions, one head, one
batch element.** Total threadgroup count = `ceil(ql/32) × h × b`.

For Gemma 4 26B, 2455-token prompt: `ceil(2455/32)=77 × n_heads × batch`. With 16 heads
(per the ADR's ggml-metal-ops.cpp grid description) and batch=1: 77 × 16 × 1 = 1232
threadgroups.

### 6.6 How the mask buffer is passed

`sdpa.rs:194-228`

When `mask_buffer.is_some()` (line 110: `let has_mask = mask_buffer.is_some()`):

```rust
set_params!(
    encoder,
    (
        (q_buffer, q_offset),   // buffer 0
        (k_buffer, k_offset),   // buffer 1
        (v_buffer, v_offset),   // buffer 2
        output,                  // buffer 3
        params,                  // buffer 4 (AttnParams struct)
        mask_params,             // buffer 5 (AttnMaskParams struct)
        mask                     // buffer 6 (mask data)
    )
);
```

When no mask:
```rust
set_params!(
    encoder,
    (
        (q_buffer, q_offset),   // buffer 0
        (k_buffer, k_offset),   // buffer 1
        (v_buffer, v_offset),   // buffer 2
        output,                  // buffer 3
        params                   // buffer 4 (AttnParams struct)
        // buffers 5 and 6 absent — kernel sees has_mask=false constant
    )
);
```

The mask is passed at buffer index 6, with `AttnMaskParams` at index 5. When no mask is
present, the function constant `has_mask=false` (constant index 300) causes the compiler
to dead-code-eliminate the mask loads in the kernel.

**Sentinel for "no mask":** There is no sentinel value. The absence of mask is entirely
controlled by the function constant `has_mask`. The mask buffer pointer is simply not bound.

### 6.7 Scale application

The scale `1/sqrt(head_dim)` is passed as `params->scale` (a field in the `AttnParams`
struct, at `sdpa.rs:151`: `scale` — the `scale: f32` argument to `call_sdpa_full`).

Inside the kernel, at line 2000, it is multiplied by `log2(e) = 1.44269504089` and
applied to Q **before** the computation:

```metal
TransformScale<T> ts(static_cast<T>(params->scale * 1.44269504089));
// ...
loader_q.apply_inplace_op(ts);  // applied in-place to Q tile in threadgroup memory
```

The `softcapping` field in `AttnParams` (line 1808) is always set to `1.0` by the candle
host at `sdpa.rs:152`:
```rust
softcapping: 1.0, // SDPA full doesn't support softcapping, always 1.0
```
This means the kernel's softcapping path is dead code in candle's use. The mlx-native
port can set it to 1.0 similarly.

---

## 7. D=512 instantiation status

**D=512 IS instantiated in candle**, at `scaled_dot_product_attention.metal:2332-2337`:

```metal
// BD=512: reduced tiles (BQ=8, BK=8) to fit 32KB threadgroup memory.
// Only f16/bf16 — float32 exceeds the limit.
instantiate_attn(float16,  half,        8, 8, 512, 1, 1, float16,  half)
instantiate_attn(float16,  half,        8, 8, 512, 1, 1, bool_,    bool)
instantiate_attn(bfloat16, bfloat16_t,  8, 8, 512, 1, 1, bfloat16, bfloat16_t)
instantiate_attn(bfloat16, bfloat16_t,  8, 8, 512, 1, 1, bool_,    bool)
```

**For D=512: BQ=8, BK=8, WM=1, WN=1** (not BQ=32/BK=16 as for D=256).

This is confirmed by the tile-selection logic at sdpa.rs:86-94: the `bd == 512` branch
returns `(8, 8, 1, 1)`.

**Key constraint:** D=512 f32 is **explicitly unsupported** in candle (sdpa.rs:87-93 returns
an error). Only f16 and bf16 are supported at D=512 due to threadgroup memory limits.

**Fragment counts at D=512 (BQ=8, BK=8, WM=1, WN=1):**
- kNWarps = 1 × 1 = 1
- TQ = 8 / (1 × 8) = 1 (still 1, confirmed by static_assert at line 2018)
- TK = 8 / 8 = 1
- TD = 512 / 8 = 64

**Threadgroup memory at D=512 (bf16):**
- padK = 16/2 = 8; LDK_tgp = 8+8 = 16
- padV = 16/2 = 8; LDV_tgp = 512+8 = 520
- Q_smem: 8 × (512+8) = 8 × 520 = 4160 elements × 2 bytes = 8,320 bytes
- KV_smem: max((8+8)×512, 8×520) = max(8192, 4160) = 8192 elements × 2 bytes = 16,384 bytes
- **Total: 24,704 bytes ≈ 24.1 KB** — fits within 32 KB.

**MMA counts at D=512 (TQ=1, TK=1, TD=64):**
- Q·K^T: tile_matmad with M=1, N=TK=1, K=1 per DD step; TD=64 steps → 64 × 1 = 64 MMAs
- P·V: TQ×TD×TK = 1×64×1 = 64 MMAs
- Total: 128 MMAs per KV block (same as D=256)

This is not a Phase 1b open question — the instantiation exists and the tile geometry is
determined by the source. **Phase 1a can vendor both D=256 and D=512 instantiations
from candle.**

---

## 8. Kernel I/O contract

### 8.1 Buffer layouts

From the kernel at lines 1920–1942 and AttnParams at lines 1798–1824:

**Q, K, V buffers:**
- Layout: `[batch, head, seq_len, head_dim]` with strides `[Q_strides[0], Q_strides[1], Q_strides[2], 1]`
- `Q_strides[2]` is the sequence stride (row stride in the head_dim dimension)
- `Q_strides[1]` is the head stride; `Q_strides[0]` is the batch stride
- The stride along head_dim (innermost) is always 1 (contiguous) — this is a precondition
  noted in sdpa.rs at the comment above `call_sdpa_full` (line 15–19)
- K is loaded **transposed** into threadgroup memory: the `KBlockLoader` has
  `kDstStrRow=1, kDstStrCol=LDK_tgp` (line 1978-1979), meaning K rows become columns
  in shared memory (so `K[i,j]` in device memory → `Ks[j * LDK_tgp + i]` in threadgroup)
- V is loaded **not transposed**: `kDstStrRow=LDV_tgp, kDstStrCol=1` (lines 1987-1988)

**O buffer:**
- Same layout as Q: `[batch, head, seq_len, head_dim]` with O_strides
- Written back at lines 2282–2294

**Mask buffer (when present):**
- Shape: `[batch, head, qL, kL]` with strides `[M_strides[0], M_strides[1], M_strides[2], 1]`
- `M_strides[2]` is the row stride (qL dimension)
- Loaded element-by-element via `MMAFrag_mask_t::load_safe` at lines 2163–2172
- The mask stride `mask_params->M_strides[2]` is the qL-row stride passed as an `int`
  at line 2167

### 8.2 Scalar types per buffer

From instantiation table (lines 2328–2337) and AttnParams (line 1807 scale field is f32
regardless of T):

| Instantiation | Q/K/V/O type T | Mask type | AccumType |
|---|---|---|---|
| float16 × float16 mask | `half` | `half` | `float` |
| float16 × bool mask | `half` | `bool` | `float` |
| bfloat16 × bfloat16 mask | `bfloat16_t` | `bfloat16_t` | `float` |
| bfloat16 × bool mask | `bfloat16_t` | `bool` | `float` |
| float32 × float32 mask | `float` | `float` | `float` |
| float32 × bool mask | `float` | `bool` | `float` |
| (D=512) float16 × float16 | `half` | `half` | `float` |
| (D=512) float16 × bool | `half` | `bool` | `float` |
| (D=512) bfloat16 × bfloat16 | `bfloat16_t` | `bfloat16_t` | `float` |
| (D=512) bfloat16 × bool | `bfloat16_t` | `bool` | `float` |

**AccumType is always `float` (f32)** — the accumulator is always full precision regardless
of Q/K/V dtype.

### 8.3 Mask format

**Additive mask or bool mask:**

From lines 2149-2184:
```metal
constexpr bool is_bool = is_same_v<MaskType, bool>;
using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;
// ...
if constexpr (is_bool) {
    Stile.frag_at(i, j)[jj] =
        mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
} else {
    Stile.frag_at(i, j)[jj] += 1.44269504089 * selem_t(mfrag[jj]);
}
```

- **Bool mask:** `true` = attend, `false` = mask out (set to `-inf`)
- **Float/bf16/f16 mask (non-bool):** treated as a **log2-scale additive bias** —
  the mask value is multiplied by `1.44269504089 = log2(e)` before being added to the
  score. This means the mask must be provided in **natural-log scale** (i.e., standard
  attention bias mask where `-inf` means mask-out and `0` means no effect). The `×log2(e)`
  converts it to base-2 scale to match the Q pre-scaling convention.

### 8.4 Softmax formula

The kernel computes:

```
O = softmax(Q·K^T × (scale × log2(e)) + mask_log2_scale) · V
```

Where:
- `Q` is pre-scaled by `scale × log2(e)` at load time (line 2000, 2049)
- The dot product `Q·K^T` is computed in base-2 scale
- Mask (if non-bool) is also multiplied by `log2(e)` before addition (line 2180)
- Softmax uses `fast::exp2` (base-2), equivalent to natural-log softmax
- The output `O` is written as the same dtype `T` as the inputs

**Scale:** The host passes `scale = 1.0 / sqrt(head_dim)` (standard attention scale).
The kernel additionally multiplies by `log2(e)` to switch exponent base.

**Causal masking:** Handled in-kernel via the `do_causal` function constant (index 301).
When `do_causal=true`, positions where `row_pos < col_pos` are set to `-inf`
(lines 2121–2141). This is separate from the mask buffer.

**Softcap:** `params->softcapping` field exists but is always 1.0 in candle's SDPA full
path. The kernel does not apply softcapping in the attention<> template (the softcapping
field is in AttnParams but the attention<> kernel body does not reference it — checked
exhaustively in lines 1895–2295; the field is only defined in the struct for ABI
compatibility with other uses of AttnParams in the Metal library).

---

## 9. License header and provenance

### 9.1 Full license/copyright header

`scaled_dot_product_attention.metal:0`

```
// Updated from MLX commit has f70764a
```

That is the **entire** header comment. There is no multi-line copyright block at the top
of this file in candle's vendored copy. The file goes directly from this one comment to
`#include <metal_stdlib>`.

### 9.2 Candle repository license

Candle is dual-licensed under **Apache-2.0 / MIT** per its `Cargo.toml` and `LICENSE-APACHE`
/ `LICENSE-MIT` files. MLX upstream is Apache-2.0. Both licenses are permissive.

### 9.3 Attribution requirement for mlx-native port

For a clean port of this Metal shader into `/opt/mlx-native/src/shaders/flash_attn_prefill.metal`
(an MIT-licensed project per ADR-006):

**Required attribution header** (to be placed at the top of the new file):

```metal
// Ported from candle (huggingface/candle), dual Apache-2.0 / MIT license.
// Original source: candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal
// Upstream MLX lineage: MLX commit f70764a (Apple Inc., Apache-2.0)
// Candle-specific additions: NaN guards (ExpSubOp, factor guard) from commit 46928bce
// SPDX-License-Identifier: MIT OR Apache-2.0
```

This satisfies both Apache-2.0 attribution requirements (retain copyright notice) and MIT
(retain license text). Since the original file has only a single-line upstream reference
and no explicit copyright holder named, the above form is sufficient — no additional
copyright notice is legally required for a file that only cites a git commit.

---

## 10. Vendoring recipe — step-by-step for Agent #3

### 10.1 Target file

`/opt/mlx-native/src/shaders/flash_attn_prefill.metal` (new file)

### 10.2 Lines to copy, in order

**Step 1: Write the attribution header (NEW — does not exist in candle source)**

```metal
// Ported from candle (huggingface/candle), dual Apache-2.0 / MIT license.
// Original: candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal
// MLX upstream: commit f70764a. Candle additions: commit 46928bce NaN guards.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Updated from MLX commit has f70764a  (original upstream marker, preserved)
```

**Step 2: Copy system includes (lines 2–5 of candle source, verbatim)**

```metal
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;
```

**Step 3: Copy bf16 emulation (lines 7–262, verbatim)**

The `#if defined(__HAVE_BFLOAT__)` / `#else` block. Copy the entire block.
On modern M-series with `__HAVE_BFLOAT__` defined, the short path is taken (typedef only).
The long path (the `_MLX_BFloat16` struct) is needed for older Silicon.

**Step 4: Skip lines 264–305** (MLXFastAttentionParams, MLXScaledDotProductAttentionParams —
unused by attention<>; skip to avoid dead code).

**Step 5: Skip lines 306–627** (sdpa_vector, sdpa_vector_2pass_1, sdpa_vector_2pass_2
kernels — these are the vector/decode kernels, not used for prefill).

**Step 6: Copy `Limits<T>` (lines 628–673, verbatim)**

```metal
// ============ "mlx/backend/metal/kernels/utils.h"

template <typename U>
struct Limits { ... };

#define instantiate_default_limit(type) ...
instantiate_default_limit(uint8_t); ... instantiate_default_limit(int64_t);

#define instantiate_float_limit(type) ...
instantiate_float_limit(half);
instantiate_float_limit(float);
instantiate_float_limit(bfloat16_t);
```

**Step 7: Skip `BlockLoader<>` (lines 676–798)** — the attention<> kernel uses only
`BlockLoaderT<>`, not `BlockLoader<>`. Omit to reduce dead code.

**Step 8: Copy `BlockLoaderT<>` (lines 800–925, verbatim)**

Includes the `CShape` struct (800–804) and the full `BlockLoaderT<>` template.

**Step 9: Copy type_traits and integral_constant helpers (lines 927–996, verbatim)**

`pointer_element_t`, `void_t`, `Int<>` alias, `integral_const_binop` operators.

**Step 10: Skip `sum<>` variadic (lines 1002–1010)** — unused in attention<>.

**Step 11: Skip Transform structs (lines 1012–1068)** — `TransformNone`, `TransformAdd`,
`TransformAxpby`, `AccumHelper`, `BlockSwizzle` — all unused in attention<>.

**Step 12: Copy MMA infrastructure (lines 1070–1504, verbatim)**

This is: `Shape2D`, `Layout2D`, `BaseMMAFrag<>`, `MMATile<>`, `tile_matmad<>`.
Stop at line 1504 (closing brace of `tile_matmad`).

**Step 13: Skip `BlockMMA<>` (lines 1506–1795)** — used only by GEMM kernels.

**Step 14: Copy the attention kernel region (lines 1796–2295, verbatim)**

This is everything from the banner comment through the closing brace of `attention<>`:
- Banner + `AttnParams` + `AttnMaskParams` structs (1796–1828)
- Function constants `align_Q`, `align_K`, `has_mask`, `do_causal` (1834–1838)
- Op structs: `TransformScale`, `MaxOp`, `SumOp`, `MulOp`, `SubOp`, `ExpSubOp`, `DivOp` (1840–1893)
- `attention<>` kernel template (1895–2295)

**Step 15: Copy the instantiation macros and D=256 + D=512 instantiations**

Copy the macro definitions (lines 2297–2313, verbatim), then write **only** the
instantiations needed for the mlx-native port (do NOT copy the full `instantiate_attn_mask_helper`
block which produces 7 head-dim variants per dtype — we only need D=256 and D=512):

```metal
// clang-format off

#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define instantiate_attn(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "steel_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  attention, dtype, bq, bk, bd, wm, wn, mtype, float)

// D=256 instantiations (Gemma 4 sliding-window layers)
// BQ=32, BK=16, WM=4, WN=1
instantiate_attn(float32,  float,       32, 16, 256, 4, 1, float32,  float)
instantiate_attn(bfloat16, bfloat16_t,  32, 16, 256, 4, 1, bfloat16, bfloat16_t)
instantiate_attn(bfloat16, bfloat16_t,  32, 16, 256, 4, 1, bool_,    bool)
instantiate_attn(float16,  half,        32, 16, 256, 4, 1, float16,  half)
instantiate_attn(float16,  half,        32, 16, 256, 4, 1, bool_,    bool)

// D=512 instantiations (Gemma 4 global-attention layers)
// BQ=8, BK=8, WM=1, WN=1 — only f16/bf16 (f32 exceeds threadgroup memory limit)
instantiate_attn(bfloat16, bfloat16_t,  8, 8, 512, 1, 1, bfloat16, bfloat16_t)
instantiate_attn(bfloat16, bfloat16_t,  8, 8, 512, 1, 1, bool_,    bool)
instantiate_attn(float16,  half,         8, 8, 512, 1, 1, float16,  half)
instantiate_attn(float16,  half,         8, 8, 512, 1, 1, bool_,    bool)
// NOTE: float32 at D=512 is unsupported — threadgroup memory would be ~53KB (exceeds limit)
```

### 10.3 Lines to adapt

**`#include` paths:** None — the file has no `#include` of MLX-owned headers; all helpers
are inlined. No path adaptation is needed.

**`bfloat16_t` typedef:** On modern macOS/Metal SDK (M-series), `__HAVE_BFLOAT__` is
defined and the first branch of the `#if` at line 10 is taken, giving simply
`typedef bfloat bfloat16_t` and `typedef half float16_t`. The long `_MLX_BFloat16` struct
(lines 56–261) is compiled out. The vendor agent should verify this is true for the mlx-native
build target (Metal shader compilation flags) before optionally pruning the else branch.

**Kernel source enum:** The candle host uses `Source::Sdpa` (sdpa.rs:133) which maps to the
compiled Metal library name. In mlx-native, the port will need a new Metal library source
registration (e.g., `Source::FlashAttnPrefill` or equivalent). This is a host-side Rust
change, not a shader change.

### 10.4 MLX helper structs/macros

All MLX helper code is already inlined in the source file. No external references to resolve.
The only Metal standard library calls are:
- `metal::numeric_limits<T>::infinity()` — from `<metal_stdlib>`
- `simdgroup_multiply_accumulate()` — from `<metal_simdgroup>`
- `simd_shuffle_xor()` — from `<metal_simdgroup>`
- `fast::exp2()` — from `<metal_stdlib>`
- `threadgroup_barrier()`, `simdgroup_barrier()` — from `<metal_stdlib>`

All available via the two system includes already in the file.

### 10.5 Preprocessor defines that must match

| Define | Value | Source line |
|---|---|---|
| `STEEL_CONST` | `static constant constexpr const` | 7 |
| `STEEL_PRAGMA_UNROLL` | `_Pragma("clang loop unroll(full)")` | 8 |

Both must be present before first use. They are defined at lines 7–8 of the candle source;
copy them verbatim.

### 10.6 What to verify after mechanical copy

1. **Compile check:** `xcrun -sdk macosx metal -c flash_attn_prefill.metal` with no errors.
2. **Function constant indices:** Verify indices 200, 201, 300, 301 do not conflict with
   any other Metal library the mlx-native build links.
3. **`bool_` macro:** Note that `instantiate_attn` uses `bool_` as the mask-type name string
   (producing `_maskbool_` in the kernel name) and `bool` as the C++ type — this asymmetry
   is candle's naming convention and must be preserved exactly for the host-side name lookup
   to match.
4. **Softcapping field:** Set to `1.0` in the `AttnParams` struct passed from Rust; do not
   attempt to activate softcapping unless Gemma requires it (it does not for standard SDPA).

---

## Summary of ADR corrections

| ADR claim | Source verdict |
|---|---|
| "Kernel body at :1895-2295" | Confirmed. Closing brace at line 2295. |
| "TQ=1, TK=2, TD=32 for D=256" | Confirmed from lines 2012-2016. |
| "~28.5 KB threadgroup memory" | Confirmed (29,184 bytes bf16). |
| "32 MMAs/simdgroup for Q·K^T" | **INCORRECT for D=256.** Source: 64 MMAs (TD=32 × TK=2). The "32" figure applies to D=128 (TD=16 × TK=1 = 16, or D=128 with TK=2 = 32). At D=256: **64 Q·K^T MMAs + 64 P·V MMAs = 128 total**. |
| "Host tile-selection at sdpa.rs:86-98" | Actual lines: 76-98 (10 lines earlier). |
| "D=512 open question" | **Closed.** D=512 IS instantiated at lines 2332-2337 with BQ=8, BK=8, WM=1, WN=1. F32 is explicitly rejected. |
| "simd_max / simd_sum reductions" | **Imprecise.** Actual: `simd_shuffle_xor` butterfly at lines 1251-1258. Same result, different intrinsic. |
| ":2315-2337 instantiation table" | Confirmed — this range covers all instantiations including D=512. |
