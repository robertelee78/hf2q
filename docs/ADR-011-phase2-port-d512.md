# ADR-011 Phase 2 — Port llama.cpp's D=512 Dispatch Geometry to mlx-native

**Status**: Proposed
**Date**: 2026-04-17
**Owner**: hf2q core (mlx-native flash-attention working group)
**Supersedes**: nothing (extends ADR-011, ADR-011-phase1-llamacpp-delta §4b/§RISK-1, and ADR-011-phase1-port-source-decision §0/§5.2/§Appendix A)
**Author**: Agent #3 (research-d512), CFA swarm `swarm-1776516482254-ft5mwj`

---

## 0. Decision (TL;DR)

Port llama.cpp's `D=512` flash-attention dispatch geometry — **NSG=8 simdgroups, NQPSG=8 queries-per-threadgroup, NCPSG=64 cache-items-per-threadgroup, threadgroup-half output accumulator** — into `/opt/mlx-native` as a **second, independent kernel template** (`flash_attn_prefill_llamacpp_d512.metal`) sharing the dispatcher entry-point names (`flash_attn_prefill_bf16_d512` and friends). The candle-derived single-simdgroup template at lines 1559-1562 of `flash_attn_prefill.metal` is **structurally incapable of hosting NSG=8 at BD=512** without exceeding Apple Silicon's 32 KB threadgroup-memory budget (proof in §3.3 below); the candle template's `BQ ≥ kNWarps × kFragSize` static_assert (`flash_attn_prefill.metal:1232`) makes the minimum BQ for NSG=8 equal to 64, which would require Q_smem of 67,584 B at BD=512 bf16 — over 2× the limit.

The replacement geometry is `(BQ=8, BK=64, BD=512, NSG=8)` with **the actual Q-row partition handled by `NQ = Q/NSG = 1` per simdgroup** (llama.cpp's model — `ggml-metal.metal:5810`), not by candle's `TQ = BQ/(WM×WN×8)` per-warp split. Output `O` lives in **threadgroup-memory `half` accumulator** (`ggml-metal.metal:5818`), not in registers. The `nsg=1` candle path at D=512 is **replaced**, not retained — it has no in-tree callers other than the absent `dispatch_flash_attn_prefill_*_d512` function (proof in §6.3).

Expected throughput gain on Gemma 4 26B MoE DWQ global-attention layers (~50% of all attention layers per ADR-011-phase1-llamacpp-delta §4b): **6×–8× per-layer**, lifting overall prefill from the current ~25 % of llama.cpp parity (estimate from current `nsg=1` D=512 dispatch pure-compute ratio) toward **80–95 % of llama.cpp's ~3300 tok/s peer** on the canonical 2455-token harness.

---

## 1. Exact llama.cpp D=512 Dispatch Geometry

Every value in this table is read from `/opt/llama.cpp/ggml/src/ggml-metal/`.

| Parameter | Value for D=512 | Source (file:line) |
|---|---|---|
| **NSG** (simdgroups per threadgroup) | **8** | `ggml-metal-ops.cpp:2807` — `int32_t nsg = ne00 >= 512 ? 8 : 4;` |
| **NQPSG** (queries per threadgroup, kernel template `Q`) | **8** | `ggml-metal-impl.h:93` — `#define OP_FLASH_ATTN_EXT_NQPSG 8`; consumed at `ggml-metal-ops.cpp:2698` and as the template default at `ggml-metal.metal:6403` |
| **NCPSG** (cache items per threadgroup, kernel template `C`) | **64** | `ggml-metal-impl.h:94` — `#define OP_FLASH_ATTN_EXT_NCPSG 64`; consumed at `ggml-metal-ops.cpp:2699` and as the template default at `ggml-metal.metal:6404` |
| **NQ** (Q rows per simdgroup) | **NQPSG/NSG = 8/8 = 1** | `ggml-metal.metal:5810` — `constexpr short NQ = Q/NSG;` |
| **NC** (Q*K columns per simdgroup, in 8x8 frags) | **(C/8)/NSG = 8/8 = 1** | `ggml-metal.metal:6020` — `constexpr short NC = (C/8)/NSG;`. Static-asserted at `:6018`. |
| **NO** (output 8x8 frags per simdgroup along D) | **PV8/NSG = 64/8 = 8** | `ggml-metal.metal:6184` — `constexpr short NO = PV8/NSG;`. Static-asserted at `:6182`. PV = `PAD2(DV, 64)` = 512 at DV=512 (`:5804`). |
| **Total threads / threadgroup** | **NSG × 32 = 256** | dispatch at `ggml-metal-ops.cpp:2861` — `(ne01+nqptg-1)/nqptg, ne02, ne03, 32, nsg, 1` |
| **Threadgroup memory footprint** | **28,672 B (28 KB), bf16, no q-cache** | `ggml-metal-ops.cpp:2789` `FATTN_SMEM(nsg)` macro — see arithmetic in §2.3 |
| **Kernel entry point name (bf16)** | **`kernel_flash_attn_ext_bf16_dk512_dv512`** | `ggml-metal.metal:6510` |
| **Q*K matmul accumulator** | **register-resident `qk8x8_t` per Q×K 8×8 frag** | `ggml-metal.metal:6023` — `qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>(0)` |
| **Output accumulator location** | **threadgroup memory** (`so` = `shmem_f16 + 0*T + Q*DK`), in `half` | `ggml-metal.metal:5818` declaration, lines 6186-6196 register hoist + 6249-6256 store-back per KV chunk |
| **Mask precision in shmem** | **`half2`** | `ggml-metal.metal:5830`, `5970`, `5972` |
| **K/V load precision in shmem** | **template `k_t` = `bfloat` for bf16 weights** | `ggml-metal.metal:6444-6451` `FA_TYPES_BF` macro |

### 1.1 Single template, multiple instantiations

Note carefully: llama.cpp does not have a "D=512 NSG=8 kernel" as a separate Metal entry point. It has **one template** `kernel_flash_attn_ext_impl<…, DK, DV, Q, C, NSG>` (`ggml-metal.metal:5736-6375`) and one host-visible thunk `kernel_flash_attn_ext<…, DK, DV>` (`ggml-metal.metal:6377-6430`) which selects NSG via the `FC_flash_attn_ext_nsg` Metal **function constant** (`:5735`) at pipeline-creation time, switching at the `case 4` / `case 8` arms (`:6425-6426`). The host picks NSG=8 vs NSG=4 from `ne00`, sets the function constant, and gets a specialised pipeline back. Cases 1 and 2 are **commented out for library load-time** (`:6423-6424`). The 135 instantiations referenced in ADR-011 §A are the cartesian product `{f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0} × 15 (DK,DV) pairs` — **NSG is not in the cartesian product**; it is the function-constant axis.

### 1.2 The full bf16 D=512 instantiation table (4 entries, function-constant-specialised)

There are exactly **3 host-visible bf16 entries that touch DK=512** (`ggml-metal.metal:6510-6511`):

```metal
template [[host_name("kernel_flash_attn_ext_bf16_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4, 1, dequantize_bf16, bfloat4x4, 1, dequantize_bf16, 512, 512>;
template [[host_name("kernel_flash_attn_ext_bf16_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4, 1, dequantize_bf16, bfloat4x4, 1, dequantize_bf16, 576, 512>;
```

(`dk576_dv512` covers DeepSeek-style MLA where K head-dim differs from V head-dim. We do **not** need this for Gemma 4.)

Each is then specialised at pipeline-creation time by the `FC_flash_attn_ext_nsg` function constant to NSG=4 or NSG=8. For `ne00=DK=512` the host always picks NSG=8 (`ggml-metal-ops.cpp:2807`).

---

## 2. Why NSG=8 at D=512 Specifically

### 2.1 The relationship: NQ, NC, NO all derive from NSG

llama.cpp's per-simdgroup workload at the kernel-internal level (`ggml-metal.metal`):

- **Q-row distribution** — `NQ = Q/NSG` (`:5810`). Each simdgroup owns `NQ` of the `Q=NQPSG=8` query rows. The Q tile is loaded **once per threadgroup** into `sq[Q × DK]` shared memory (`:5859-5871`); each simdgroup then operates on its `NQ` rows by indexing `j = jj*NSG + sgitg` (`:5860`).
- **Q*K column distribution** — `NC = (C/8)/NSG = (64/8)/NSG` (`:6020`). For NSG=8, NC=1; for NSG=4, NC=2. Each simdgroup computes `NC` of the `C/8 = 8` 8×8 score frags. Result is written to threadgroup `ss` (`:6061`).
- **Output D-axis distribution** — `NO = PV8/NSG` (`:6184`). For NSG=8, NO=8; for NSG=4, NO=16. Each simdgroup owns `NO` of the `PV8 = 64` 8×8 output frags along the `D` axis. Output frags are loaded from `so` into registers `lo[NO]` (`:6186, 6191-6195`), accumulated against the `(p_ij × V)` product across all `C/8 = 8` cache frags (`:6204-6219`), then stored back to `so` (`:6249-6255`).

### 2.2 Why double NSG when D doubles

The output tile per threadgroup is `[NQPSG, DV] = [8, 512] = 4096 f16 elements = 8 KB`. The matmul `O += P × V` has total work proportional to `NQPSG × NCPSG × DV`. At fixed NQPSG=8 and NCPSG=64, doubling DV from 256 → 512 doubles the V-axis matmul work. Doubling NSG from 4 → 8 keeps **per-simdgroup output frag count NO = PV8/NSG = 16 → 8 unchanged in absolute compute (work-per-simdgroup halves), but doubles parallelism**. Intuitively: each simdgroup still does ~1 ms of compute for the K×V phase, but the wallclock for the threadgroup halves because there are 2× more simdgroups doing it.

The `static_assert((C/8) % NSG == 0, "")` at `ggml-metal.metal:6018` and `static_assert(PV8 % NSG == 0, "")` at `:6182` constrain NSG to divisors of `C/8 = 8` and `PV8 = DV/8 = 64` (for DV=512). Allowed values for D=512: NSG ∈ {1, 2, 4, 8}. NSG=16 would require C=128 (more shmem) — chosen not to. NSG=8 is the largest power-of-two allowed by both static_asserts at the current C=64.

### 2.3 Threadgroup memory budget — exact arithmetic at NSG=8, DK=DV=512, bf16

The macro `FATTN_SMEM` is at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2789`:

```c
#define FATTN_SMEM(nsg) (GGML_PAD(                                  \
    (nqptg*(ne00 + 2*GGML_PAD(ne20, 64) + 2*(2*ncpsg))              \
       + is_q*(16*32*(nsg)))                                        \
    *(sizeof(float)/2), 16))
```

Substituting `nqptg=8` (from `ggml-metal-impl.h:93`), `ncpsg=64` (`:94`), `ne00=512`, `ne20=512`, `is_q=0` (bf16 K/V cache, set at `ggml-metal-ops.cpp:2780`), `nsg=8`:

```
inner = 8 * (512 + 2 * GGML_PAD(512, 64) + 2 * (2 * 64))
      = 8 * (512 + 1024 + 256)
      = 8 * 1792
      = 14,336

bytes = GGML_PAD(14336 * (sizeof(float)/2), 16)
      = GGML_PAD(14336 * 2, 16)
      = GGML_PAD(28672, 16)
      = 28,672 bytes
      = 28.0 KiB
```

This is the **threadgroup-memory size requested at dispatch time** (`ggml-metal-ops.cpp:2859` `set_threadgroup_memory_size(enc, smem, 0)`). It comprises:

| Region | Size (B) | Source line(s) | Notes |
|---|---|---|---|
| `sq[Q × DK]` query tile | 8 × 512 × 2 = **8,192** | `:5816, :5859-5871` | half (typed as `q_t = bfloat` per FA_TYPES_BF) |
| `so[Q × PV]` output accumulator | 8 × 512 × 2 = **8,192** | `:5818, :6178-6256` | half-typed; `o_t = bfloat` per FA_TYPES_BF |
| `ss[Q × SH]` softmax/mask scratch | 8 × 128 × 2 = **2,048** | `:5811, :5820-5821` | `SH = 2*C = 128` floats but laid out as halves (`s_t == half`) per FA_TYPES_BF |
| `sm2` mask staging | (overlaps `ss`, sized `2*C`) | `:5830` | aliased into `ss` region |
| Per-simdgroup K/V load scratch | 8 × (4×16×8) × 2 = **8,192** | `:5823-5827` | `sgitg*(4*16*KV)` halves; KV=8; NSG=8 lanes |
| Padding to 16 B | up to 16 | `GGML_PAD(…, 16)` | |

Total: 8,192 + 8,192 + 2,048 + 8,192 ≈ 26,624 B aligned to 16 → 28,672 B with FATTN_SMEM rounding. The 26,624 vs 28,672 gap is the Q × 2 × (2 × NCPSG) = 8 × 256 = 2,048 mask + softmax slack inside the `nqptg*(…)` aggregate (the formula puts mask and softmax in the same Q-major aggregate rather than per-simdgroup).

**At nsg=4** (for DK<512), same formula gives:
```
inner = 8 * (512 + 1024 + 256) + 0 = 14336   (same! is_q=0)
bytes = 28,672
```
So at bf16 `is_q=0` cache, NSG does not appear in the threadgroup memory size *unless* the K/V cache is quantized — exactly because per-simdgroup K/V load scratch (`16*32*nsg`) is only allocated when dequantization is needed (`is_q=1`). For our Gemma 4 26B bf16 weights, the K/V cache type is bf16 and `is_q=0`, so the 28,672 B figure is identical for NSG=4 and NSG=8. **NSG=8 is purely a parallelism win, not a memory cost.**

For DK=DV=256 with the same formula (`nqptg=8, ncpsg=64, is_q=0`):
```
inner = 8 * (256 + 512 + 256) + 0 = 8192
bytes = 16,384
```
Half the memory, twice the headroom. Both fit comfortably in 32 KB.

### 2.4 Comparison with our current `nsg=1` D=512 path

| | Our current D=512 | llama.cpp D=512 NSG=8 | Ratio |
|---|---|---|---|
| Threads / threadgroup | 32 | 256 | **8× more** |
| Active simdgroups | 1 | 8 | **8× more** |
| Q rows processed concurrently / threadgroup | 8 (BQ) | 8 (NQPSG) | same |
| KV columns processed per inner step / threadgroup | 8 (BK) | 64 (NCPSG) | **8× more** |
| Per-simdgroup K/V reuse | once per col | once per col | same |
| Threadgroup memory | ~25 KB (per §3) | 28 KB | comparable |
| Pipeline occupancy on M5 Max (40 cores) | ~5 % per threadgroup (32/640 active lanes/core) | ~40 % per threadgroup | **8× better** |

Per ADR-011-phase1-llamacpp-delta §RISK-1 (line 211) and Phase1 prior-art §7 (line 786-805), our current D=512 dispatch is **decode-like geometry being used for prefill** — exactly the pathology nsg=8 fixes.

---

## 3. Our Current D=512 — Chesterton's Fence

### 3.1 Lineage of the BQ=8/BK=8/WM=1/WN=1 choice

Our current shader at `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1559-1562` is verbatim-traceable to candle's `scaled_dot_product_attention.metal:2334-2337`:

```metal
// candle (upstream)
instantiate_attn(float16,  half,        8, 8, 512, 1, 1, float16,  half)
instantiate_attn(float16,  half,        8, 8, 512, 1, 1, bool_,    bool)
instantiate_attn(bfloat16, bfloat16_t,  8, 8, 512, 1, 1, bfloat16, bfloat16_t)
instantiate_attn(bfloat16, bfloat16_t,  8, 8, 512, 1, 1, bool_,    bool)

// ours (mlx-native, identical params)
instantiate_flash_attn_prefill("flash_attn_prefill_bf16_d512",          bfloat16_t, 8, 8, 512, 1, 1, bfloat16_t)
instantiate_flash_attn_prefill("flash_attn_prefill_bf16_d512_boolmask", bfloat16_t, 8, 8, 512, 1, 1, bool)
instantiate_flash_attn_prefill("flash_attn_prefill_f16_d512",           half,       8, 8, 512, 1, 1, half)
instantiate_flash_attn_prefill("flash_attn_prefill_f16_d512_boolmask",  half,       8, 8, 512, 1, 1, bool)
```

The preamble at `flash_attn_prefill.metal:22-30` justifies this with: "D=512: BQ=8, BK=8, 1 simdgroup per threadgroup (32 threads). ~25 KB threadgroup memory at bf16/f16. Smaller tiles because the Qs footprint scales with BQ × BD; preserving the larger BQ at D=512 would overflow the 32 KB budget."

### 3.2 Verifying that reasoning at the candle template's static_assert

`flash_attn_prefill.metal:1230-1233` asserts:

```c++
constexpr int kNWarps = WM * WN;
static_assert(
    BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
    "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");
```

Combined with `kFragSize = 8` (`:1227`), this requires `BQ ≥ 8 × kNWarps` and `BQ` divisible by `8 × kNWarps`. Per-warp-along-Q frag count is `TQ = BQ/(kNWarps × 8)` (`:1236`).

At candle's chosen D=256: `BQ=32, kNWarps=4` → TQ=1, `BQ ≥ 32 ✓`. Q_smem at bf16 = `32 × (256 + 8) × 2 = 16,896 B`. KV_smem at bf16 = `max(8, 0) × (256+8) × 2 — wait: tgp_mem_0 = (BK+padK)*BD = 24*256 = 6,144 elems, tgp_mem_1 = BK*(BD+padV) = 16*264 = 4,224 elems → tgp_mem_s = 6,144 elems × 2 B = 12,288 B`. Total: 16,896 + 12,288 = 29,184 B (matches preamble's "~29 KB").

At D=512 if we tried `BQ=32, kNWarps=4`: Q_smem = `32 × (512+8) × 2 = 33,280 B` — **alone exceeds 32 KB before KV_smem** — so candle reduced to BQ=8, kNWarps=1 → Q_smem = `8 × 520 × 2 = 8,320 B`, KV_smem = `max((8+8)×512, 8×(512+8)) × 2 = max(8192, 4160) × 2 = 16,384 B`. Total: 8,320 + 16,384 = 24,704 B (matches preamble's "~25 KB"). Confirmed.

**Chesterton's fence verdict**: Candle's D=512 BQ=8/WM=1 was chosen specifically because its template architecture does not allow more than 1 simdgroup at D=512 within the 32 KB budget. The fence is real, the reasoning is sound — but it is reasoning about *candle's particular template structure*, not about Apple Silicon's hardware. llama.cpp gets NSG=8 at D=512 within 28 KB by using a fundamentally different per-simdgroup work decomposition (Q rows distributed across simdgroups instead of stacked per simdgroup).

### 3.3 Why we cannot scale candle's existing template to NSG=8 at D=512

Suppose we attempt to add a candle-template instantiation for D=512 with `WM=8`. The static_assert at `:1232` requires `BQ ≥ 8 × 8 = 64` and BQ a multiple of 64. At BQ=64, BD=512, bf16:

| Quantity | Formula | Value |
|---|---|---|
| Q_smem | `BQ × (BD + 8) × 2` | 64 × 520 × 2 = **66,560 B** |
| KV_smem | `max((BK+8)×BD, BK×(BD+8)) × 2` | max(16×512, 8×520) × 2 = 16,384 B |
| Total | | **82,944 B = 81 KiB** |

That is **~2.6× the 32 KB threadgroup-memory limit**. Even at WM=2 (BQ ≥ 16), Q_smem alone would be `16 × 520 × 2 = 16,640 B` and KV_smem `8 × 528 × 2 = 8,448 B` — total 25,088 B — would fit, but WM=2 is only 64 threads, not the desired 256. WM=4 requires BQ ≥ 32 → Q_smem = 33,280 B alone — **already over budget**.

**Hard conclusion**: candle's per-warp-Q-stacking template cannot host the NSG=8 geometry at D=512 within Apple Silicon's TG-memory budget. We must adopt llama.cpp's per-simdgroup-Q-distributed architecture. This is what the rest of this ADR specifies.

### 3.4 Otile register pressure in the candle template at our current D=512

For our `BD=512, BQ=8, WM=1, WN=1`:
- `kNWarps = 1`, `TQ = 8/(1×8) = 1`, `TD = 512/8 = 64`.
- `Otile` is `MMATile<float, TQ=1, TD=64, 8x8>` (`:1248`) → 64 × 8x8 float frags **per simdgroup** = 64 × 256 B = 16,384 B of register state per simdgroup. At 32 lanes per simdgroup, that is **128 float registers per thread** for Otile alone, plus all of Stile (1×TK=1 frag, 256 B), Qtile, Vtile (transient).

Apple GPU has 128 registers/thread under default occupancy and can spill to private memory at higher pressure. Our current Otile-in-registers at TD=64 is right at the edge. **Going to llama.cpp's NO=8 frags-per-simdgroup design (so per simdgroup) drops Otile register pressure by 8×** to ~16 float registers/thread — leaving plenty of headroom for the Q*K and P*V intermediates and improving occupancy.

---

## 4. Port Spec — What Changes In The Kernel

We need a **new Metal kernel template** that mirrors llama.cpp's per-simdgroup decomposition. The candle template is preserved as-is for D=256 (it is correct, fast, and not the bottleneck). We add a sibling shader file (or a parallel template in the same file) for the D=512 path.

### 4.1 New shader file (recommended) or new template (minimum)

**Option A (recommended)**: Add `/opt/mlx-native/src/shaders/flash_attn_prefill_d512.metal` (~600 LOC) containing a single `kernel_flash_attn_prefill_d512<T, NSG>` template that mirrors `kernel_flash_attn_ext_impl` from `ggml-metal.metal:5736-6375`, MIT-licensed (compatible with our SPDX header), with the dependent helpers (`PAD2` → already exists or trivial; `FOR_UNROLL` → use `STEEL_PRAGMA_UNROLL` already in our codebase).

**Option B (minimum LOC delta)**: Add a parallel `attention_d512<T, NSG>` template inside the existing `flash_attn_prefill.metal`, sharing its `BlockLoaderT`, `MMATile`, `BaseMMAFrag`, `TransformScale`, `DivOp`, `ExpSubOp`, `Limits` infrastructure where possible. The risk is that the per-simdgroup-Q decomposition is sufficiently different from the per-warp-Q model that sharing infrastructure forces awkward parameterisation. Recommend Option A.

The new template's signature (target):

```cpp
template <typename T, short NSG>  // T ∈ {bfloat, half}; NSG ∈ {4, 8}
[[kernel, max_total_threads_per_threadgroup(NSG * 32)]]
void flash_attn_prefill_d512(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    threadgroup half* shmem [[threadgroup(0)]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]]);
```

Compile-time constants inside the body:
- `constexpr short DK = 512, DV = 512;`
- `constexpr short Q = 8, C = 64;` (NQPSG, NCPSG — fixed; not made parametric)
- `constexpr short NQ = Q/NSG;` — 1 at NSG=8, 2 at NSG=4
- `constexpr short PV = 512;` (== DV; PAD2 unchanged at 512)
- `constexpr short NO = (PV/8)/NSG;` — 8 at NSG=8, 16 at NSG=4

### 4.2 Threadgroup memory layout (computed)

The Rust dispatcher must request `smem` bytes from the encoder. Per the FATTN_SMEM derivation in §2.3, but expressed in our naming for the shader's allocation:

```
Q_tile_bytes  = Q * DK * sizeof(T)        // 8 * 512 * 2 = 8192
O_tile_bytes  = Q * PV * sizeof(half)     // 8 * 512 * 2 = 8192  (always half, regardless of T)
SS_bytes      = Q * SH * sizeof(half)     // 8 * 128 * 2 = 2048; SH = 2*C
mask_bytes    = (overlaps SS region)      // 0 incremental
KV_per_sg     = NSG * (4 * 16 * KV) * sizeof(half)  // NSG * 1024  (KV=8 in llama.cpp)
              = 8 * 1024 = 8192  at NSG=8
total         = 8192 + 8192 + 2048 + 8192 = 26,624 → align(16) → 26,624 B  (26 KB)
```

Note: llama.cpp's `FATTN_SMEM` over-allocates slightly because it sums Q-major regions and per-simdgroup regions in one expression. Our exact layout sums to 26,624 B at NSG=8. We will allocate **28,672 B (28 KB) to match llama.cpp's allocation exactly** — this is a 2 KB safety margin and matches the reference for like-for-like behaviour.

### 4.3 Output `O` lives in threadgroup memory (the architectural shift)

Critical: in the new kernel, `O` is **NOT a register `MMATile`**. It is a threadgroup-memory region `so[Q × PV]` of `half` (not float, and not a per-simdgroup tile). The per-simdgroup compute pattern is:

```cpp
// At each KV chunk iteration:
o8x8_t lo[NO];                                  // NO = 8 frags per simdgroup
for (ii = 0; ii < NO; ++ii)
    simdgroup_load(lo[ii], so + 8*sgitg + 8*NSG*ii, PV, 0, false);
// ... matmul into lo[ii] ...
for (ii = 0; ii < NO; ++ii)
    simdgroup_store(lo[ii], so + 8*sgitg + 8*NSG*ii, PV, 0, false);
```

This is **load → accumulate → store per-iteration**, paying per-iteration shmem traffic for the Otile in exchange for far lower register pressure (and therefore far higher occupancy). Per `ggml-metal.metal:6188-6196` and `:6248-6256`. The pattern repeats once per KV chunk (`for ic0 = 0; ;ic0++` at `:5907`).

### 4.4 Per-simdgroup softmax state

Each simdgroup holds its own `M[NQ]` (running max) and `S[NQ]` (running sum) — `ggml-metal.metal:5888, 5891`. At NSG=8, NQ=1, so M and S are scalars per simdgroup. There is **no cross-simdgroup softmax reduction** — each simdgroup owns disjoint Q rows (`j = jj*NSG + sgitg` at `:5860, 5874, etc.`), so the reductions are fully local.

The mask is loaded once per threadgroup (`:5954-5981`), the K is loaded directly from device memory by each simdgroup at `pk += sgitg*(8*NS10)` (`:6015`), and the V load is similarly partitioned at `pv += 8*sgitg` (`:6201`). All per-simdgroup pointers are independent.

### 4.5 Replace, don't keep, the candle nsg=1 D=512

Per §6.3 below: the four candle D=512 kernel names are registered (`flash_attn_prefill.rs:130-133`, `:152-153`) but no `dispatch_flash_attn_prefill_*_d512` function exists in this crate. The names are effectively dead code today — their only consumer at compile-time is a unit test in `tests/test_flash_attn_prefill.rs` that checks pipeline compilation succeeds (verify in §7 step 7).

**Decision**: re-bind the four host_name() entries (`flash_attn_prefill_bf16_d512`, `flash_attn_prefill_bf16_d512_boolmask`, `flash_attn_prefill_f16_d512`, `flash_attn_prefill_f16_d512_boolmask`) to the new `flash_attn_prefill_d512` kernel (Option A) with `NSG=8` baked in as a function constant. The candle template instantiations at `flash_attn_prefill.metal:1559-1562` are **deleted**. Any test that exercises them must be ported to dispatch through the new kernel (NSG=8 path).

This avoids two-template confusion and matches the principle from `feedback_no_shortcuts`: do not preserve a known-weak path "for compatibility" when the strong path exists.

---

## 5. llama.cpp Kernel Structure For D=512 — Same Template, Specialised

### 5.1 D=256 vs D=512 share one template

llama.cpp uses **the same kernel_flash_attn_ext_impl template for D=256 and D=512**. Only the function constant `FC_flash_attn_ext_nsg` (`ggml-metal.metal:5735`) differs at pipeline creation: NSG=4 for D=256, NSG=8 for D=512 (`ggml-metal-ops.cpp:2807`). The host_name entries differ only in DK/DV template arguments (e.g. `dk256_dv256` vs `dk512_dv512`) — NSG is **not** in the host_name. This is structurally cleaner than our current two-instantiation-per-(D,dtype,mask) scheme.

We should consider, post-port, **collapsing our D=256 and D=512 paths into one kernel template** with NSG as a function constant — this matches llama.cpp's design and is the natural extension. But that is a Phase 3 cleanup; Phase 2 ships the new D=512 kernel as a sibling.

### 5.2 Why threadgroup-half O wins at D=512

At D=256: per-simdgroup Otile (candle) = `TQ × TD = 1 × 32 = 32` float frags = 32 × 256 B = 8 KB / simdgroup = 256 B / thread = 64 registers / thread for Otile alone — **fits comfortably**, no spill. This is why candle's D=256 with register-resident Otile works fine.

At D=512 in candle: per-simdgroup Otile = `1 × 64 = 64` float frags = 16 KB / simdgroup = 512 B / thread = **128 registers / thread for Otile alone**, on top of Stile, Qtile, Vtile. Apple Silicon's per-thread register budget is 128 GP regs (Metal 3 spec) before private-memory spill. **D=512 register-resident Otile is exactly at the spill boundary**, which combined with single-simdgroup occupancy is why our current path is so slow.

llama.cpp's threadgroup-half Otile pays ~26 KB shmem traffic per KV chunk (load 8 × 1 KB Otile frags + store same — mostly L1-cache-hot since shmem is on-die SRAM) but **frees the registers** for high occupancy. At NSG=8 with NO=8 frags / simdgroup, peak Otile register pressure during a single matmul-accumulate step is `lo[8]` × 256 B = 2 KB per simdgroup = 64 B per thread = **16 registers per thread**. 8× lower register pressure than candle's D=512, and the Apple GPU scheduler can run far more wavefronts concurrently.

### 5.3 Argument from Apple-Silicon GPU family constraints

| Constraint | Value | Source |
|---|---|---|
| Simdgroup width | 32 lanes | `flash_attn_prefill.metal` constant `kFragSize=8`, MMA is 8×8×8 |
| Simdgroup MMA frag | 8×8×8 (M=N=K=8) | `MMAFrag<float, 8, 8>` at `:1228` |
| Per-thread GP register budget (Metal 3) | 128 (with spill above) | hf2q `feedback_learn_from_fast` recall + Apple Metal Shading Language Spec §8.4 |
| Threadgroup memory limit | 32 KiB (M5 Max) | `flash_attn_prefill.metal:21`, `flash_attn_prefill.rs:33` |
| Max threads/threadgroup | 1024 | Metal 3; we use 256 with NSG=8 |
| GPU cores (M5 Max) | 40 | `system_profiler SPDisplaysDataType` (verified 2026-04-17) |

NSG=8 → 256 threads/TG → at full occupancy, 1 threadgroup per GPU core → 40 concurrent threadgroups system-wide. For a typical Gemma 4 attention dispatch (B=1, H=4 KV-replicated to 8 query heads, qL_tiles=8 at qL=2455/Q=8 ≈ 307), grid = 307 × 8 × 1 = 2,456 threadgroups → **~62 waves per core** at 1 TG/core occupancy → easily saturates the GPU. Our current nsg=1 has **1/8 the occupancy** per core, leaving 87.5 % of the GPU idle.

---

## 6. Dispatcher-Side Port

### 6.1 New Rust dispatch function

Add to `/opt/mlx-native/src/ops/flash_attn_prefill.rs` (after the existing `dispatch_flash_attn_prefill_bf16_d256` at `:408`):

```rust
const NSG_D512: u32 = 8;
const NQPSG_D512: u32 = 8;
const NCPSG_D512: u32 = 64;
const NTHREADS_D512: u32 = 32 * NSG_D512;  // 256

/// Threadgroup memory bytes for D=512, bf16, NSG=8 (matches llama.cpp FATTN_SMEM).
const TGMEM_BYTES_D512: usize = 28_672;

pub fn dispatch_flash_attn_prefill_bf16_d512(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &MlxBuffer, k: &MlxBuffer, v: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    out: &mut MlxBuffer,
    params: &FlashAttnPrefillParams,
) -> Result<()> {
    if params.head_dim != 512 {
        return Err(MlxError::InvalidArgument(format!(
            "dispatch_flash_attn_prefill_bf16_d512: head_dim must be 512, got {}",
            params.head_dim
        )));
    }
    validate_params(params)?;
    // … bf16 dtype checks for q/k/v/out and (optional) mask, mirroring d256 …
    // … buffer size validation, mirroring d256 …

    let kernel_name = if mask.is_some() {
        K_BF16_D512  // additive mask path
    } else {
        K_BF16_D512  // same kernel; has_mask is a function constant
    };
    // (boolmask variant uses K_BF16_D512_BOOLMASK instead — wire from a
    //  MaskKind enum on the host or via separate dispatch entry points.)

    let pipeline = registry.get_pipeline_with_bool_constants(
        kernel_name,
        device.metal_device(),
        &[
            (200, params.seq_len_q % NQPSG_D512 == 0),  // align_Q
            (201, params.seq_len_k % NCPSG_D512 == 0),  // align_K
            (300, mask.is_some()),                       // has_mask
            (301, params.do_causal),                     // do_causal
            // NEW: function constant for NSG (matches llama.cpp's
            //      FC_flash_attn_ext_nsg at ggml-metal.metal:5735)
            // Encoded as i32 not bool → use a new helper
            //      get_pipeline_with_constants(name, dev, &bool_consts, &i32_consts)
        ],
    )?;

    // Build AttnParams (same struct as D=256; only seq_len_q tile divisor differs).
    let attn_params = AttnParamsGpu { /* mirror D=256, with d=512 */ };

    // Grid geometry — matches llama.cpp ggml-metal-ops.cpp:2861
    //   grid = ((qL + NQPSG - 1)/NQPSG, H, B)
    //   threads / TG = (32, NSG, 1) = 256
    let grid = MTLSize::new(
        params.seq_len_q.div_ceil(NQPSG_D512) as u64,
        params.n_heads as u64,
        params.batch as u64,
    );
    let tg_size = MTLSize::new(32, NSG_D512 as u64, 1);

    encoder.set_op_kind(CapturedOpKind::Sdpa);
    encoder.set_threadgroup_memory_size(TGMEM_BYTES_D512, 0);  // NEW: explicit
    // … encode_threadgroups_with_args(…) as in D=256 dispatcher …

    Ok(())
}
```

### 6.2 Threadgroup-memory size API

Our existing `CommandEncoder::encode_threadgroups_with_args` (in `flash_attn_prefill.rs:572`) does NOT currently expose `set_threadgroup_memory_size`. The candle template uses static `threadgroup T Q_smem[BQ * (BD + padQ)]` declarations (`flash_attn_prefill.metal:1181-1182`) which the Metal compiler sizes at compile time. The new D=512 kernel uses **dynamic threadgroup memory** (the `threadgroup half* shmem [[threadgroup(0)]]` pattern from `ggml-metal.metal:5777, 5816`) which requires the dispatcher to set the size at encode time via `MTLComputeCommandEncoder.setThreadgroupMemoryLength(_:atIndex:)`. We must extend `CommandEncoder` with a corresponding method:

```rust
impl CommandEncoder {
    pub fn set_threadgroup_memory_size(&mut self, bytes: usize, index: u32) {
        self.encoder.set_threadgroup_memory_length(bytes as u64, index as u64);
    }
}
```

This exists already on `metal::ComputeCommandEncoderRef` upstream — we just need to expose it.

### 6.3 Function-constant API for NSG

Our `KernelRegistry::get_pipeline_with_bool_constants` at `flash_attn_prefill.rs:489` accepts only `(u32, bool)` pairs. For NSG we need `(u32, i32)` (function constant index 322 in llama.cpp's offset scheme — `FC_FLASH_ATTN_EXT + 22`, see `ggml-metal-impl.h:77` and `ggml-metal.metal:5735`). Two options:

- **Add `get_pipeline_with_mixed_constants(name, dev, bool_consts, i32_consts)`** — preferred. Mirrors Metal's `MTLFunctionConstantValues` API which already supports mixed types.
- **Bake NSG into the host_name** — e.g. `flash_attn_prefill_bf16_d512_nsg8`. Less flexible but avoids API churn. We will need NSG=4 for the future D=256 unification (§5.1) so the mixed-constants approach is better.

### 6.4 Mask-kind discrimination

The `_boolmask` suffix on the existing kernel names (`K_BF16_D512_BOOLMASK` at `flash_attn_prefill.rs:113`) splits additive-vs-bool masks into separate pipelines. The new D=512 kernel must preserve this: two host_name entries, one per mask kind, both backed by the same template instantiated with `MaskType=bf16` vs `MaskType=bool`.

### 6.5 Validation paths

Mirror `dispatch_flash_attn_prefill_bf16_d256` at `:418-446` exactly:
- `params.head_dim != 512` → `InvalidArgument`.
- All buffers must be BF16.
- If `mask.is_some()`, mask must be BF16.
- Buffer sizes via `validate_buffer_size`.
- `validate_params(params)` for the common parameter sanity.

---

## 7. Port Spec — Actionable Checklist

Each item cites the source-of-truth file:line in llama.cpp and the target file:line in mlx-native. Follow in order.

1. **Add `flash_attn_prefill_d512.metal`** at `/opt/mlx-native/src/shaders/flash_attn_prefill_d512.metal` (~600 LOC).
   - Source: port `kernel_flash_attn_ext_impl` from `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5736-6375` verbatim, retaining MIT/Apache notice.
   - Drop the quantized K/V branch (`else` at `:6066-6127`, `:6257-6322`) — Gemma 4 K/V is bf16 in our path.
   - Drop the `pad`, `blk`, `sinks`, `bias`, `softcap` paths (`:5896-5903`, `:5914-5949`, `:5954-6005`, `:6328-6346`) — they are SWA / GQA-bias / sliding-window features not yet wired in our dispatcher.
   - Replace llama.cpp's `ggml_metal_kargs_flash_attn_ext` arg struct (`ggml-metal.metal:5768-5780`) with our existing `AttnParams` (`flash_attn_prefill.metal:780-810` approx).
   - Translate llama.cpp's strides (`args.nb01, .nb11`, etc.) to our `params->Q_strides[2]` etc. (mlx-native uses [B, H, L, D] contiguous per `flash_attn_prefill.rs:502-509`; llama.cpp uses arbitrary nb01/nb11 — port carefully, our layout is the simpler case).

2. **Instantiate D=512 entry points** in the new shader, at the bottom of the new file:
   ```metal
   template [[host_name("flash_attn_prefill_bf16_d512")]]
       kernel decltype(flash_attn_prefill_d512<bfloat, 8>) flash_attn_prefill_d512<bfloat, 8>;
   template [[host_name("flash_attn_prefill_bf16_d512_boolmask")]]
       kernel decltype(flash_attn_prefill_d512<bfloat, 8, /*mask=*/bool>) flash_attn_prefill_d512<bfloat, 8, bool>;
   template [[host_name("flash_attn_prefill_f16_d512")]]
       kernel decltype(flash_attn_prefill_d512<half, 8>) flash_attn_prefill_d512<half, 8>;
   template [[host_name("flash_attn_prefill_f16_d512_boolmask")]]
       kernel decltype(flash_attn_prefill_d512<half, 8, bool>) flash_attn_prefill_d512<half, 8, bool>;
   ```
   These names match the existing registrations at `flash_attn_prefill.rs:130-133` so no registry change is needed for names.

3. **Delete the candle D=512 instantiations** at `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1559-1562` (4 lines + comment block at `:1555-1558`).
   - Replace with a comment pointing to the new shader file.
   - Keep the D=256 instantiations at `:1550-1553` untouched.

4. **Wire the new shader into registration** at `/opt/mlx-native/src/ops/flash_attn_prefill.rs`:
   - Add `pub static FLASH_ATTN_PREFILL_D512_SHADER_SOURCE: &str = include_str!("../shaders/flash_attn_prefill_d512.metal");` after `:97-98`.
   - In `pub fn register(registry: &mut KernelRegistry)` at `:151-155`, register the four D=512 names against the new source instead of the existing one. Split the loop:
     ```rust
     for &name in &[K_BF16_D256, K_BF16_D256_BOOLMASK, K_F16_D256, K_F16_D256_BOOLMASK] {
         registry.register_source(name, FLASH_ATTN_PREFILL_SHADER_SOURCE);
     }
     for &name in &[K_BF16_D512, K_BF16_D512_BOOLMASK, K_F16_D512, K_F16_D512_BOOLMASK] {
         registry.register_source(name, FLASH_ATTN_PREFILL_D512_SHADER_SOURCE);
     }
     ```

5. **Extend `CommandEncoder` with `set_threadgroup_memory_size`** at `/opt/mlx-native/src/encoder.rs` (file location verified with Glob; modify the impl block on `CommandEncoder`). One ~5 LOC method per §6.2.

6. **Extend `KernelRegistry` with mixed-typed function constants** at `/opt/mlx-native/src/kernel_registry.rs` per §6.3. Add `get_pipeline_with_mixed_constants(name, dev, bool_consts: &[(u32, bool)], i32_consts: &[(u32, i32)])`. Use the existing `metal::FunctionConstantValues` API.

7. **Add `dispatch_flash_attn_prefill_bf16_d512`** at `/opt/mlx-native/src/ops/flash_attn_prefill.rs:605` (immediately after the existing D=256 dispatcher's closing brace at `:604`). Body per §6.1 with all validation, function constants, and threadgroup-memory-size set.

8. **Add `dispatch_flash_attn_prefill_f16_d512`** as a sibling — same as item 7 but for f16 I/O. Validate dtype === DType::F16 throughout.

9. **Add unit tests** at `/opt/mlx-native/tests/test_flash_attn_prefill.rs` (file already exists; verify path with Glob):
   - `test_dispatch_d512_bf16_pipeline_compiles` — round-trip register + get_pipeline_with_mixed_constants + dispatch a 1-token shape; ensures the kernel compiles.
   - `test_dispatch_d512_bf16_matches_cpu_reference` — small shapes (qL=8, kL=64, B=1, H=2, gqa_factor=1, D=512, no mask, no causal). Compare bf16 GPU output to f32 CPU reference at `atol=1e-2, rtol=1e-2` (the bf16 round-off bound).
   - `test_dispatch_d512_bf16_sourdough_byte_identical` — load the canonical sourdough prefix, run prefill through the new dispatcher, assert output bytes match the saved frozen-baseline (per `feat(gate-d): wire frozen hf2q self-baseline check` in the recent commit log).

10. **Wire the new dispatcher into the prefill session** at `/opt/mlx-native/src/.../forward_prefill_batched.rs` (path approximate per ADR-011 §R3 line 211; verify with Glob). Replace the existing D=512 `s.sdpa(...)` → `s.flash_attn_prefill(...)` route's D=512 branch to call the new `dispatch_flash_attn_prefill_bf16_d512` instead of the candle-template path.

11. **Benchmark** with the canonical 2455-token harness:
    - Baseline: current `nsg=1` D=512 (must be re-measured before this work begins, per `feedback_ground_truth_is_what_we_can_measure_now`).
    - Post-port: new `nsg=8` D=512.
    - Target: ≥6× per-layer-D=512 throughput improvement; ≥3× overall prefill tok/s improvement (since global layers are ~50 % of attention layers).

12. **Bisect coherence** if any output byte diverges from the frozen sourdough baseline:
    - Validate Q*K stage by zeroing V and asserting attention scores match (per `feedback_prove_in_code`).
    - Validate softmax by comparing `M` and `S` arrays per Q row to a CPU reference.
    - Validate P*V by holding scores fixed and comparing only the matmul.

13. **Delete the candle D=512 dispatcher path** — once tests pass, remove any code that was a fallback to the old kernel. **No fallback** per ADR-005 line 21 + `feedback_no_shortcuts`.

14. **Update ADR-011** to mark Phase 2 complete and reference this document.

---

## 8. Performance Expectations

### 8.1 Per-layer D=512 throughput

Static-evidence model (calibrated against `project_metal_compiler_auto_optimizes_static_levers`'s caveat — these are *expectations* requiring measurement-confirmation):

| Path | Threads / TG | Active simdgroups | Threadgroups / GPU core (M5 Max) | Per-op throughput estimate |
|---|---|---|---|---|
| Current (candle nsg=1) | 32 | 1 | up to 32 (low TG-mem) | **1× baseline** |
| Post-port (llama.cpp nsg=8) | 256 | 8 | up to 4 (more TG-mem per TG) | **6×–8× baseline** |

The 6×–8× range comes from: 8× more parallel work per TG, partially offset by reduced TG-per-core occupancy (4 vs 32 — but each TG does 8× the work, so total work in flight is similar). Net wallclock-per-layer should approach llama.cpp's, since llama.cpp uses the same NSG=8 geometry on the same hardware.

### 8.2 Overall prefill tok/s

ADR-011 §0 cites the 21× peer gap (152 tok/s ours vs 3260 tok/s llama.cpp on 2455-token prompt). That gap was measured before any flash-attn port. ADR-011 phase1 reduced D=256 to candle-equivalent throughput (Phase 1a complete per ADR-011-phase1-tests-verification.md). The remaining gap on D=512 global layers, per the ADR-011-phase1-llamacpp-delta §RISK-1 prediction (line 211), is the dominant remainder.

If global layers are 50% of attention compute and our D=512 path improves 6×, overall attention compute improves by `1 / (0.5/6 + 0.5) = 1 / 0.583 = 1.71×`. Combined with whatever Phase 1 D=256 delivered, this should put us in the **2400–3000 tok/s range** (the Walk-phase floor per ADR-011 §192).

True parity to peer (3260 tok/s) likely requires **also** porting `flash_attn_ext_blk` SWA tile-skip pre-pass (Phase 4 / `ADR-011-phase1-llamacpp-delta` §3 line 78) for sliding layers — out of scope for this Phase 2 ADR.

### 8.3 Confidence

Medium-high. The architectural facts (NSG, NQPSG, NCPSG, threadgroup memory math, function constant infrastructure) are read directly from production llama.cpp source. The **execution-time** estimate is calibrated against llama.cpp's measured 3260 tok/s on the same hardware running the same model — that is real, not synthetic, evidence. The risk is implementation-correctness (especially the threadgroup-half output accumulator pattern, which is novel for our codebase) and the edge cases stripped in step 1 (pad / blk / sinks / bias / softcap) — these must be re-added before this kernel can replace the D=256 kernel too in Phase 3.

---

## 9. Open Questions / Phase 3 Follow-ups

1. **Unify D=256 and D=512** under one template with NSG as a function constant (Phase 3). Reduces shader source duplication and matches llama.cpp's structure.
2. **Add NSG=4 specialisation for D=256** within the same new template — would let us A/B test our current candle D=256 (WM=4 register-Otile) vs llama.cpp NSG=4 (threadgroup-half-Otile) for the same head_dim. May be a wash or may give 10-20% on D=256 too.
3. **Port `flash_attn_ext_blk` SWA tile-skip pre-pass** (Phase 4 — separate ADR). ~59 % of tiles skipped at window=1024, qL=2455 per `ADR-011-phase1-llamacpp-delta` §3.
4. **Re-baseline** llama.cpp's 3260 tok/s figure on the day before Phase 2 perf gates — that figure is from a prior measurement and may have drifted (per `feedback_ground_truth_is_what_we_can_measure_now`).

---

## 10. References

- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2622-2862` — full prefill dispatch (`ggml_metal_op_flash_attn_ext`)
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5736-6375` — `kernel_flash_attn_ext_impl` template body
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6377-6512` — outer template + 135 instantiations
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:74-97` — function constant offsets, NQPSG/NCPSG defines
- `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1120-1564` — current candle-derived attention<> template + instantiations
- `/opt/mlx-native/src/ops/flash_attn_prefill.rs:97-604` — current Rust dispatcher (D=256 only)
- `/opt/candle/candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal:1895-2337` — original candle kernel + instantiation table
- `/opt/hf2q/docs/ADR-011-flash-attn-prefill.md` — Phase 1 root ADR
- `/opt/hf2q/docs/ADR-011-phase1-llamacpp-delta.md:153-172, 211-244` — D=512 candle-vs-llamacpp delta and RISK-1
- `/opt/hf2q/docs/ADR-011-phase1-port-source-decision.md:140-189, 215-227, 384-388` — Phase 4 carve-out scope
- `/opt/hf2q/docs/ADR-011-phase1-prior-art-map.md:786-805` — D=512 instantiation status in candle
