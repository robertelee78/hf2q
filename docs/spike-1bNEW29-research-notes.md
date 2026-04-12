# ADR-005 Phase 1b — 1bNEW.29 Pre-Spike Research Notes (Agent #3 / kernel-cite-research)

**Date:** 2026-04-11
**Role:** read-only citation map + surface-area estimate for a candidate 1bNEW.29 port of llama.cpp's `kernel_mul_mv_q4_0_f32` and `kernel_mul_mv_q6_K_f32` into `vendor/candle-metal-kernels/`.
**Status:** research-only. No `src/` or `vendor/` edits. No benchmarks run. Dispatch-count and per-threadgroup row-count claims are derived by static reading; they need runtime confirmation by the other CFA agents before any port work starts.
**Baseline:** hf2q HEAD `a377f76` (post-1bNEW.21 vendored candle-metal-kernels patch, `compute_per_buffer = 100`).
**llama.cpp tree:** `/opt/llama.cpp` HEAD as of 2026-04-11 on this machine (used as the reference target for the port).

---

## Executive summary

1. Candle's `kernel_mul_mv_q4_0_f32` and `kernel_mul_mv_q6_K_f32` are **derived from an older llama.cpp `ggml-metal.m` snapshot**, not from MLX. The MLX-derived code in the vendored kernels lives in `mlx_gemm.metal` (F16/BF16 gemm), not in `quantized.metal`. The helper functions `block_q_n_dot_y` in candle are byte-for-byte ports of the llama.cpp originals with the same ggml-style comments (`vendor/candle-metal-kernels/src/metal_src/quantized.metal:2265-2303`). Candle's lineage is therefore **"older-llama.cpp-derived, then frozen"**.
2. **NSG is not the lever.** Both candle and current llama.cpp hardcode `N_SIMDGROUP = 2` for Q4_0 (`quantized.metal:2307` / `ggml-metal-impl.h:15`) and `NSG = 2` for Q6_K (`quantized.metal:5215` via `row = 2*r0+sgitg` / `ggml-metal-impl.h:45`). llama.cpp uses Metal function constants (`FC_mul_mv_nsg`) only as a mechanism for compile-time specialization, not for runtime per-shape tuning — the **selection heuristic** at `ggml-metal-device.cpp:744-800` reads the `N_SG_Q*_K` macro constants unconditionally for every quant type and does not branch on `ne00/ne01` at all. The spike at ADR-005:902 framed NSG as "shape-aware tuning per `ggml_metal_op_mul_mat_id` heuristics"; the actual code is static per-type. **The NSG-sweep microbench proposed at ADR-005:898-916 will almost certainly falsify itself at nsg=2 for both dtypes** — because that *is* what llama.cpp picks, and the llama.cpp sources (numbers aside) do not contain a different value for these types on any shape.
3. **The real asymmetry between candle and llama.cpp for these kernels is elsewhere:**
   - **(a) Q6_K rows-per-simdgroup.** llama.cpp's Q6_K impl is templated on `nr0 = N_R0_Q6_K = 2` (`ggml-metal-impl.h:44`, `ggml-metal.metal:7924`) and the inner body loops over `nr0` rows per simdgroup. Candle's Q6_K impl is *not* templated; it hardcodes **exactly one row per simdgroup** via `const int row = 2*r0 + sgitg` at `quantized.metal:5215`. With nsg=2 in both, llama.cpp produces **4 rows per threadgroup**, candle produces **2 rows per threadgroup**. Candle therefore dispatches **2× more threadgroups for Q6_K** at identical ne01.
   - **(b) Argument-passing style.** Candle binds args per-scalar via 3 `set_buffer` + 18 `set_bytes` calls = **21 Metal API calls per Q-mul-mv dispatch** (`kernels/quantized.rs:146-169`). llama.cpp packs them into a `ggml_metal_kargs_mul_mv` POD struct (`ggml-metal-impl.h:439-459`) and issues 1 `set_bytes` + 3 `set_buffer` = **4 API calls per dispatch** (`ggml-metal-ops.cpp:2219-2223`). Spike 1bNEW.22 already documented this as "real CPU savings hidden behind GPU compute" (ADR-005:901). Not wall-clock-recoverable on its own given CPU is at 1.48 ms/token.
   - **(c) Inner-loop register structure.** llama.cpp's Q4_0 `mul_vec_q_n_f32_impl` (`ggml-metal.metal:3349-3434`) hoists per-row src0 pointers into an `ax[NR0]` array with `FOR_UNROLL` outside the block iteration, and splits `sumy` into two partial sums `sumy[0]/sumy[1]` (`ggml-metal.metal:3380-3416`). Candle's version (`quantized.metal:2312-2378`) recomputes `x+ib+row*nb` on each inner iteration and uses a single `sumy` accumulator. This is a pure micro-optimization win of unknown magnitude — needs measurement.
4. **Effective lever ranking for 1bNEW.29, highest-confidence first:**
   - **L1 — Q6_K NR0=2 row-loop port.** Asymmetry exists in the source, is easy to verify statically (above), and produces a 2× dispatch-count reduction for attention q_proj/k_proj/o_proj on Q6_K hf2q shapes. Direct dispatch savings (Q6_K subset only): ~6 dispatches/layer × 30 = ~180 dispatches/token eliminated → **~0.18 ms/token @ ~1 μs GPU launch** plus higher per-simdgroup arithmetic intensity. Estimated wall-clock gain: **+1.3 to +2.5 tok/s**, assuming the per-simdgroup shared-scale-factor effect (dh, sc reused across 2 rows) adds marginal bandwidth savings on top.
   - **L2 — Q4_0 `FOR_UNROLL` + `ax[NR0]` hoisting.** Purely a GPU compute-time improvement inside the MLP gate/up/down sites. Speculative magnitude; needs microbench to justify.
   - **L3 — Struct-arg passing (`ggml_metal_kargs_mul_mv`).** CPU-side win, hidden behind GPU (per 1bNEW.22 instrumentation spike), not wall-clock recoverable in isolation. Include only as a side-effect of the port, not as primary motivation.
5. **Surface-area verdict:** L1 alone is a **small port (~120-180 LOC)** confined to `quantized.metal` + `kernels/quantized.rs`. L1+L2+L3 is a **medium port (~350-500 LOC)** that rewrites the two kernels and their Rust dispatcher. A full "match llama.cpp byte-for-byte" maximum port across both kernel families is **~800-1100 LOC** and requires the function-constant infrastructure in candle-metal-kernels.

---

## A. Candle's implementation (current production path)

### A.1 Q4_0 — `kernel_mul_mv_q4_0_f32`

| Item | Citation |
|---|---|
| Metal kernel (host stub) | `vendor/candle-metal-kernels/src/metal_src/quantized.metal:2380-2404` |
| Templated impl | `vendor/candle-metal-kernels/src/metal_src/quantized.metal:2312-2378` (`mul_vec_q_n_f32_impl<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>`) |
| Template parameters | `N_DST = 4`, `N_SIMDGROUP = 2`, `N_SIMDWIDTH = 32` (`quantized.metal:2306-2307`, `:9`) |
| `block_q4_0` definition | `vendor/candle-metal-kernels/src/metal_src/quantized.metal:21-26` |
| `block_q_n_dot_y` for `block_q4_0` | `vendor/candle-metal-kernels/src/metal_src/quantized.metal:2231-2263` (inferred by proximity to the `block_q5_0` helper at 2266) |
| Rust dispatcher | `vendor/candle-metal-kernels/src/kernels/quantized.rs:25-176` (`call_quantized_matmul_mv_t`) |
| nth0/nth1/align selection | `vendor/candle-metal-kernels/src/kernels/quantized.rs:61-112` (Q4_0 → `nth0=8, nth1=8, align=8`) |
| Kernel-name lookup | `vendor/candle-metal-kernels/src/kernels/quantized.rs:123-139` |
| Threadgroup grid | `vendor/candle-metal-kernels/src/kernels/quantized.rs:113-122` — `width = divide(ne01, align) = ne01/8`, `height = ne11`, `depth = ne12*ne13` |
| Threads-per-threadgroup | `width = nth0 = 8`, `height = nth1 = 8`, `depth = 1` → **64 threads/tg = 2 simdgroups × 32 lanes**, consistent with `N_SIMDGROUP=2, N_SIMDWIDTH=32` |
| Rows per threadgroup | `N_SIMDGROUP × N_DST = 2 × 4 = 8 rows` |
| Argument binding | 3 `set_buffer` (rhs, (lhs, lhs_offset), (dst, dst_offset)) + 18 `set_bytes` (ne00, ne01, ne02, nb00, nb01, nb02, ne10, ne11, ne12, nb10, nb11, nb12, ne0, ne1, r2, r3) = **21 Metal API calls per dispatch** (`kernels/quantized.rs:146-169`, macro at `utils.rs:160-168`) |
| Resource-usage calls | `encoder.use_resource` ×3 at `kernels/quantized.rs:170-172` |
| Dispatch | `encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup)` at `kernels/quantized.rs:174` |
| Memory pattern | Each simdgroup handles half a block at a time via `tiisg/2` stride (`quantized.metal:2347-2353`); `yl[16]` local vector cache for src1; `float sumf[nr]` register accumulator; `simd_sum` reduction at finalize (`:2373`). Recomputes `x + ib + row*nb` per-row per-block iteration inside `block_q_n_dot_y(x+ib+row*nb, sumy, yl, il)` at `:2366`. |

### A.2 Q6_K — `kernel_mul_mv_q6_K_f32`

| Item | Citation |
|---|---|
| Metal kernel (host stub) | `vendor/candle-metal-kernels/src/metal_src/quantized.metal:5268-5294` |
| Impl function | `vendor/candle-metal-kernels/src/metal_src/quantized.metal:5186-5266` (`kernel_mul_mv_q6_K_f32_impl` — **non-templated**, no `nr` parameter) |
| `block_q6_K` definition | Inferred in same file; QK_K = 256 at `:18` |
| Rows per simdgroup | **1** — hardcoded via `const int row = 2 * r0 + sgitg` at `quantized.metal:5215`. No outer `for row` loop inside the impl. |
| Rows per threadgroup | `N_SIMDGROUP × 1 = 2` (with nsg=2 via `2*r0 + sgitg`) |
| nth0/nth1/align for Q6K | `vendor/candle-metal-kernels/src/kernels/quantized.rs:93-98` — `nth0=2, nth1=32, align=2` |
| Threadgroup grid | `width = ne01/2`, `height = ne11`, `depth = ne12*ne13` (`kernels/quantized.rs:113-122`). Note `align=2` ≠ rows-per-tg=2 is a coincidence — if Q6K got the llama.cpp NR0=2 treatment this would be `align=4`. |
| Memory pattern | Outer `for (int i = ix; i < nb; i += 2)` across super-blocks with stride 2 (`:5239`). Per iteration: unpack 4 sub-slots into `float4 sums` via bit-mask extraction (`:5250-5256`), accumulate into scalar `sumf` (`:5258`). Reduce via `simd_sum` (`:5262`). Single-row-per-simdgroup means `dh`/`sc`/`qh` loads from the weight tensor are **not reused across rows** — each simdgroup reads one row's worth of Q6_K blocks. |

### A.3 Templated parent and lineage

| Item | Citation |
|---|---|
| `mul_vec_q_n_f32_impl` template definition | `vendor/candle-metal-kernels/src/metal_src/quantized.metal:2312-2378` |
| Template signature | `template<typename block_q_type, int nr, int nsg, int nw> void mul_vec_q_n_f32_impl(...)` |
| Callers (Q4_0/Q4_1/Q5_0/Q5_1) | `:2403`, `:2429`, `:2455`, `:2481` — all block_size=32 quants share this template |
| MoE variants via `kernel_mul_mv_id_t` | `:7625-7628`, `:7633` (Q4_0/Q4_1/Q5_0/Q5_1/Q6_K via `kernel_mul_mv_id<mmv_fn<...>>`) |
| Origin / lineage | **Modified from older llama.cpp `ggml-metal.m`**. Evidence: (1) `block_q_n_dot_y` helpers at `:2231-2303` retain ggml-style comments about "function for calculating inner product between half a q4_0 block"; (2) file references ggml directly at `:215`, `:241`, `:1724`, `:1829`, `:1911`, `:1959`; (3) the `#define N_SIMDWIDTH 32` header block (`:9`) is a well-known ggml-metal idiom; (4) no MLX attribution or headers present — the only MLX-origin content in the vendored kernels is in `mlx_gemm.metal` and `mlx_sort.metal`. The spike's framing at ADR-005 docs as "MLX-derived QMatMul path" is **inaccurate for quantized types**; it's llama.cpp-derived and frozen at an older snapshot. **Confidence: high** (static textual evidence). |

### A.4 Per-dispatch argument passing (shared for all Q-mul-mv)

| Item | Citation |
|---|---|
| `set_params!` macro | `vendor/candle-metal-kernels/src/utils.rs:159-168` — expands to one `set_param` call per argument, each of which invokes `set_buffer` or `set_bytes` depending on type. |
| Q-mul-mv total calls | 3 buffers + 18 scalar ints/uints = 21 Metal encoder API calls per dispatch. |
| Comparison anchor | Confirmed by ADR-005:901 and `docs/spike-1bNEW22-instrumentation.md` (the "~21 individual set_buffer/set_bytes calls" claim at ADR-005:901). |

---

## B. llama.cpp's implementation (the porting target)

### B.1 Q4_0 — `kernel_mul_mv_q4_0_f32`

| Item | Citation |
|---|---|
| Metal kernel stub | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:3515-3525` |
| Templated impl | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:3348-3434` (`mul_vec_q_n_f32_impl`) |
| Template signature | `template<typename block_q_type, short NR0, typename args_t> void mul_vec_q_n_f32_impl(...)` — **NR0 is a template parameter** (candle's nr is also a template param but it's a fixed value `N_DST=4` pattern) |
| NSG from function constant | `const short NSG = FC_mul_mv_nsg;` at `ggml-metal.metal:3358`. `FC_mul_mv_nsg` declared at `ggml-metal.metal:3345` as `constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]]` with `FC_MUL_MV = 600` at `ggml-metal-impl.h:80`. |
| NR0 for Q4_0 | `#define N_R0_Q4_0 4` at `ggml-metal-impl.h:14`. **Same as candle's `N_DST=4`.** |
| NSG for Q4_0 | `#define N_SG_Q4_0 2` at `ggml-metal-impl.h:15`. **Same as candle's `N_SIMDGROUP=2`.** |
| Args struct | `ggml_metal_kargs_mul_mv` at `ggml-metal-impl.h:439-459` — 19 fields (ne00-nb13 ints/uint64 + nr0 + r2 + r3). |
| Host-side dispatch | `ggml_metal_op_mul_mat` at `ggml-metal-ops.cpp:2189-2234` (mul_mv branch) |
| NSG selection | `ggml_metal_library_get_pipeline_mul_mv` at `ggml-metal-device.cpp:702-879`. For Q4_0: `nsg = N_SG_Q4_0; nr0 = N_R0_Q4_0;` at `:744-748`. **No per-shape branch.** The selection reads the macro constants unconditionally. |
| Per-shape heuristic | **None for Q-types.** The only shape-dependent branches in `get_pipeline_mul_mv` are for `GGML_TYPE_F32/F16/BF16` at `:721-738` (`if ne00 < 32` / `nsg = min(4, (ne00+127)/128)`). All Q-types get static `N_SG_Q*` values. The ADR-005:902 claim about "per ggml_metal_op_mul_mat_id heuristics" is wrong — the code is per-dtype not per-shape. |
| Compile-time specialization | `ggml-metal-device.cpp:860-871` — the pipeline name is `kernel_mul_mv_q4_0_f32_nsg=2` and is compiled via `ggml_metal_library_compile_pipeline` with a `ggml_metal_cv_t` that sets FC_MUL_MV+0 to the static nsg value. So llama.cpp's mechanism *permits* nsg variation across quant types (Q8_0 uses `N_SG_Q8_0=4`, Q4_K uses `N_SG_Q4_K=2`), but at runtime for a given dtype the nsg is fixed. |
| Argument bind | `ggml-metal-ops.cpp:2219-2223`: 1 `set_pipeline` + 1 `set_bytes(args)` + 3 `set_buffer(src0/src1/dst)` + 1 `set_threadgroup_memory_size` = 6 encoder ops (4 data-binding calls). |
| Threadgroup grid dispatch | `ggml-metal-ops.cpp:2227-2234`. For quantized (non-F32/F16/BF16/Q8_0) types: `dispatch_threadgroups((ne01 + nr0*nsg - 1)/(nr0*nsg), (ne11 + nr1 - 1)/nr1, ne12*ne13, 32, nsg, 1)`. For Q4_0: `ne01/(4*2) = ne01/8` threadgroups — **matches candle exactly** (candle: `ne01/align = ne01/8`). |
| Memory pattern | Per-row src0 pointer array `ax[NR0]` hoisted outside the block loop via `FOR_UNROLL` (`ggml-metal.metal:3380-3385`). Split `sumy[2]` partial sums + `FOR_UNROLL` on the `i=0..8 i+=2` loop (`:3403-3413`). Row-dot-product loop is also `FOR_UNROLL`'d (`:3415-3417`). `yb += QK4_0 * 16` at `:3419` matches candle but without the commented-out NSG-aware stride that's in the code. |

### B.2 Q6_K — `kernel_mul_mv_q6_K_f32`

| Item | Citation |
|---|---|
| Metal kernel stub | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:8019-8030` |
| Templated impl | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:7924-8017` (`kernel_mul_mv_q6_K_f32_impl`) |
| Template signature | `template<int nr0, typename args_t> void kernel_mul_mv_q6_K_f32_impl(...)` — **nr0 templated**, unlike candle's non-templated impl. |
| `nr0` instantiation | `N_R0_Q6_K = 2` at `ggml-metal-impl.h:44` → `nr0 = 2` in the template call at `ggml-metal.metal:8029`. |
| NSG | `const short NSG = FC_mul_mv_nsg;` at `ggml-metal.metal:7934`; set via `N_SG_Q6_K = 2` at `ggml-metal-impl.h:45`. |
| First-row math | `const int first_row = (r0 * NSG + sgitg) * nr0;` at `ggml-metal.metal:7947`. With NSG=2, nr0=2: each threadgroup x-index handles 4 rows, each simdgroup handles 2. |
| Offset hoisting | `offset0 = first_row*args.nb01 + ...` at `:7952` computed once per threadgroup before the block loop. Candle re-applies this implicitly via `src0 + row*nb + offset0` at `quantized.metal:5222` but only once, not per-row-per-block. |
| Outer loop | `for (int i = ix; i < nb; i += 2)` at `:7973` — **identical outer structure to candle**. |
| Inner row loop | `for (short row = 0; row < nr0; ++row)` at `:7989` iterates over 2 rows per simdgroup. **This loop does not exist in candle.** |
| Per-row pointer advance | Lines `:8001-8005`: `q1 += args.nb01; q2 += args.nb01; qh += args.nb01; sc += args.nb01; dh += args.nb01/2;` advance to the next row's block offset after processing each row. Effectively: **the `yl[16]` cache of src1 is reused across the 2 rows**, amortizing the src1 vector load cost. This is the explicit bandwidth-optimization reason for NR0=2 on Q6_K. |
| `FOR_UNROLL` use | `:7992-7997` unrolls the 4-element bit-mask extraction loop; same unroll candle has without the macro. |
| Reduction | `simd_sum` + write per row inside `for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row)` at `:8011-8016`. Candle writes a single scalar at `:5263`. |
| Host dispatcher | Same `ggml_metal_op_mul_mat` function at `ggml-metal-ops.cpp:2189-2234`. For Q6_K: `nsg=2, nr0=2 → dispatch (ne01 + 3)/4 threadgroups on x-axis`, i.e. **`ne01/4` threadgroups** vs candle's `ne01/2`. |

### B.3 `ggml_metal_kargs_mul_mv` struct

| Item | Citation |
|---|---|
| Definition | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:439-459` |
| Fields | `ne00, ne01, ne02` (int32), `nb00, nb01, nb02, nb03` (uint64), `ne10, ne11, ne12` (int32), `nb10, nb11, nb12, nb13` (uint64), `ne0, ne1` (int32), `nr0` (int32), `r2, r3` (int16) — **19 scalar fields in total** |
| Packed size | Rough: 7×4 + 7×8 + 2×2 = 28+56+4 = **~88 bytes** (with natural alignment ~96 bytes) |
| Use | `ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0)` at `ggml-metal-ops.cpp:2220` — one `set_bytes` call for the whole struct at slot 0 of the encoder. |

### B.4 Per-shape NSG selection heuristic (full quote)

The actual selection code. This is what ADR-005:902 referred to as "ggml_metal_op_mul_mat_id heuristics". It is not in `ggml-metal-ops.cpp`, it is in `ggml-metal-device.cpp`:

> `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp:702-879` — function `ggml_metal_library_get_pipeline_mul_mv`. Lines `:721-738` handle F32/F16/BF16 with a real shape-dependent `nsg = std::min(4, (ne00 + 127) / 128)` heuristic. All quantized types follow the template at `:739-856`, each branch of which does `nsg = N_SG_<TYPE>; nr0 = N_R0_<TYPE>;` with **no reference to ne00, ne01, ne10, or ne11**. Q4_0 specifically at `:744-748`, Q6_K at `:796-800`. The suffix `snprintf(name, 256, "%s_nsg=%d", base, nsg)` at `:860` means the compiled pipeline is keyed by nsg but only one nsg value per dtype is ever produced.

**Restated plainly:** llama.cpp's function-constant infrastructure permits shape-tuning but llama.cpp does not currently do shape-tuning for Q-types. ADR-005:902 conflated "has function constants" with "uses function constants for shape tuning". Candle's hardcoded `N_SIMDGROUP = 2` is already correct; **no NSG-port can yield a GPU-time improvement for Q4_0 or Q6_K at hf2q's shapes**, because llama.cpp picks exactly the same NSG.

### B.5 llama.cpp per-dispatch API call count

| Call | Source |
|---|---|
| `ggml_metal_encoder_set_pipeline(enc, pipeline)` | `ggml-metal-ops.cpp:2219` |
| `ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0)` | `ggml-metal-ops.cpp:2220` |
| `ggml_metal_encoder_set_buffer(enc, src0, 1)` | `ggml-metal-ops.cpp:2221` |
| `ggml_metal_encoder_set_buffer(enc, src1, 2)` | `ggml-metal-ops.cpp:2222` |
| `ggml_metal_encoder_set_buffer(enc, dst, 3)` | `ggml-metal-ops.cpp:2223` |
| `ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0)` | `ggml-metal-ops.cpp:2225` |
| `ggml_metal_encoder_dispatch_threadgroups(...)` | `ggml-metal-ops.cpp:2231` or `:2233` |

**7 encoder ops per dispatch**, of which **4** are data-binding (1 `set_bytes` + 3 `set_buffer`). Compare to candle's 21 API calls for the same dispatch.

---

## C. Diff summary

### C.1 Q4_0: what llama.cpp does that candle does NOT

1. **Packs arguments into a POD struct.** `ggml_metal_kargs_mul_mv` is passed via a single `set_bytes` at `ggml-metal-ops.cpp:2220`. Candle issues 18 individual `set_bytes` calls. **Scope of difference:** CPU API-call count 21 → 4 per dispatch. **Known-to-be** a hidden-behind-GPU CPU win per 1bNEW.22 spike. **Not** a wall-clock recoverable win in isolation.
2. **Hoists src0 row pointers into `ax[NR0]` with `FOR_UNROLL`.** `ggml-metal.metal:3380-3385`. Candle does `x+ib+row*nb` inside the inner block loop at `quantized.metal:2366`. Effect: llama.cpp eliminates `row*nb` multiplication from the hot inner loop. **Speculative magnitude** — possibly a few % per matmul on Q4_0 throughput. Hard to attribute without a register-pressure analysis.
3. **Split `sumy[2]` + `FOR_UNROLL` inner unrolls.** `ggml-metal.metal:3403-3413`. Candle has a single `sumy` accumulator and a non-unrolled inner `for i+=2`. Effect: potentially lets the Metal compiler vectorize the `yl[]` fill + `sumy` accumulation into wider SIMD. **Speculative** — needs measurement.
4. **Templated `NR0` instead of `N_DST` define.** Candle's N_DST=4 is a `#define`; llama.cpp's NR0 is a template parameter. Not a behavioral difference for Q4_0 (both = 4), but matters for porting infrastructure (see C.3).

**Diff summary for Q4_0: candle and llama.cpp produce the same work grid (`ne01/8` threadgroups × 64 threads/tg, 8 rows/tg, NSG=2, NR0=4). The difference is purely in how the inner loop is laid out for the Metal compiler.** There is **no dispatch-count difference** for Q4_0 MLP gate/up/down between the two implementations.

### C.2 Q6_K: what llama.cpp does that candle does NOT

1. **Templated `NR0=2` row loop.** `ggml-metal.metal:7924-7989` templates on `nr0` and has an explicit `for (short row = 0; row < nr0; ++row)` inside the block iteration. Candle's `kernel_mul_mv_q6_K_f32_impl` is non-templated and hardcodes exactly 1 row per simdgroup via `const int row = 2*r0 + sgitg` at `quantized.metal:5215`. **Direct consequence**: llama.cpp dispatches `ne01/(nr0*nsg) = ne01/4` threadgroups; candle dispatches `ne01/2`. **Candle issues 2× more Q6_K threadgroups per matmul.**
2. **Amortizes src1 vector load across rows.** llama.cpp loads `yl[16]` once per outer `i` iteration (`ggml-metal.metal:7982-7987`) and uses it for `nr0=2` rows via the inner `for row` loop. Candle loads `yl[]` implicitly via `y[l+...]` dereferences that the compiler may or may not hoist. The llama.cpp layout is explicit and correct-by-construction; the candle layout relies on the compiler noticing the loop-invariant read pattern.
3. **Per-row pointer advancement.** `ggml-metal.metal:8001-8005`: `q1/q2/qh/sc/dh += nb01` shifts the row pointer by exactly one `block_q6_K` row stride per iteration. This is how llama.cpp gets NR0=2 working without recomputing from `first_row + row`. Candle's lack of the loop means this code has no candle-side analog.
4. **Reduction loop with bounds check.** `ggml-metal.metal:8011` writes 2 results per simdgroup with `first_row + row < args.ne0` bounds check. Candle writes 1 per simdgroup at `quantized.metal:5263` with `row` guaranteed < ne01 by the dispatch math.

**Diff summary for Q6_K: llama.cpp does 2× the computational work per simdgroup (via NR0=2 row loop), halving the dispatch count for the same ne01 and sharing the src1 vector load across 2 rows.** This is a **genuine dispatch-count and bandwidth-per-dispatch asymmetry**. **This is the only identified GPU-side lever.**

### C.3 Port options with LOC / wall-clock / risk estimates

All estimates are for the **vendored candle-metal-kernels/ directory only**. They assume the Rust dispatcher in `kernels/quantized.rs` is kept mostly intact to avoid rippling into candle-core's `QMatMul` abstraction.

#### Q4_0 port options

| Option | Description | Files touched | LOC delta | Wall-clock estimate | Risk |
|---|---|---|---|---|---|
| **Q4_0-min** | Do nothing. The NR0=4 and NSG=2 are already matched. There is no identified lever. | none | 0 | 0 tok/s | 0 |
| **Q4_0-mid** | Port the `FOR_UNROLL` + `ax[NR0]` + `sumy[2]` inner-loop restructuring from `ggml-metal.metal:3349-3434` into candle's `mul_vec_q_n_f32_impl`. This affects **all of Q4_0/Q4_1/Q5_0/Q5_1** since they share the template. | `quantized.metal` (one template fn body, ~70 lines) | +40 / -30 ≈ **~70 LOC churn** | Speculative +0 to +1 tok/s. Could be zero. Needs microbench to justify. | Low. Does not change the kernel interface or args. Sourdough gate tests the numerical output; FOR_UNROLL is semantics-preserving. |
| **Q4_0-max** | Q4_0-mid + switch to struct-arg passing (`ggml_metal_kargs_mul_mv` analog in candle). Requires introducing a struct layout in `quantized.rs` + changing the dispatcher call site. | `quantized.metal` + `kernels/quantized.rs` + possibly new file | +100 / -40 ≈ **~140 LOC** | 0 wall-clock (CPU win hidden per 1bNEW.22). | Low-medium. Metal struct alignment has to match exactly on Apple Silicon; any mismatch is a silent corruption. |

#### Q6_K port options

> **⚠️ FALSIFICATION BACKPOINTER (added 2026-04-11 PM):** The `Q6_K-min` row in this table — and its `+1.3 to +2.5 tok/s` wall-clock estimate — was **empirically falsified** by `docs/spike-1bNEW29C-q6k-nr0-microbench.md` (cfa swarm `swarm-1775951202282-uwlk55`, Agent C1). A byte-for-byte port of llama.cpp's `kernel_mul_mv_q6_K_f32_impl<nr0=2>` was built as a sibling kernel and timed head-to-head against the existing production Q6_K kernel on all 6 hf2q production shapes × 2 independent 4-run sweeps with 1000 iters per cell. Correctness sanity check passed perfectly (`max|Δ| = 0.000000e0` — bitwise identical output). Wall-clock result: **all 6 shapes within ±1.8% (Run 1) and ±0.4% (Run 2)**, sign flips between runs prove the deltas are M5 Max measurement noise, not signal. **The 2× threadgroup reduction trades off ~1:1 against doubled per-simdgroup work — net wall-clock zero on M5 Max.** The "Dependent on … needs runtime confirmation" caveat in this row triggered, and the answer was NO. Future investigators: do not re-derive this hypothesis from the static evidence below; the runtime payoff has been measured and is zero. See also `project_metal_compiler_auto_optimizes_static_levers.md` in user memory — this is the third consecutive static-evidence kernel hypothesis falsified on M5 Max in a single session.

| Option | Description | Files touched | LOC delta | Wall-clock estimate | Risk |
|---|---|---|---|---|---|
| **Q6_K-min** ❌ FALSIFIED | **Port the NR0=2 row loop only.** Rewrite `kernel_mul_mv_q6_K_f32_impl` at `quantized.metal:5186-5266` to match `ggml-metal.metal:7924-8017`. Change candle's dispatcher `align` for Q6K from 2 to 4 at `kernels/quantized.rs:96`. Keep everything else. | `quantized.metal` (`~80 lines replaced`) + `kernels/quantized.rs` (1-line align change) | **~120 LOC churn** (80 new + 80 deleted, net approximately equal) | **+1.3 to +2.5 tok/s** *[empirically falsified 2026-04-11; measured envelope = 0 tok/s — see `docs/spike-1bNEW29C-q6k-nr0-microbench.md`]* assuming (a) the 2× threadgroup reduction maps to ~1 μs/dispatch × ~6 dispatches/layer × 30 layers = 0.18 ms/token and (b) the src1-cache-sharing yields another 0.2-0.4 ms/token on Q6_K-bound attention time. Dependent on llama.cpp's NR0=2 being the reason for llama.cpp's higher observed Q6_K throughput — needs runtime confirmation. **[Confirmation answer: NO.]** | Low-medium. Metal threadgroup grid semantics change: number of threadgroups halves but each does 2× work. Sourdough gate must still pass. Correctness risk: the `first_row + row < args.ne0` bound check must be handled because `ne01` is not necessarily divisible by 4 (Gemma q_proj at 4096 is, k_proj at 2048 is, o_proj at 2816 is: all % 4 == 0, so the bound is **pragmatically a no-op for hf2q shapes**, but the guard must stay in the kernel for other shapes). |
| **Q6_K-mid** | Q6_K-min + also port the `FC_mul_mv_nsg` function constant mechanism into candle-metal-kernels so NSG can be specialized per-pipeline rather than via `#define`. Not strictly needed (NSG=2 is fine) but required if 1bNEW.29 wants to parameterize NR0 as a function constant too. | `quantized.metal` + `kernels/quantized.rs` + `kernels/mod.rs` + `source.rs`/`kernel.rs` (pipeline compile path) | **~250 LOC** | Same as Q6_K-min. The function-constant infrastructure is pure overhead unless it's used for shape-specialization on a later spike. | Medium. Touches the candle-metal-kernels pipeline-compilation code, which has pool + caching state. |
| **Q6_K-max** | Full port of llama.cpp's Q6_K kernel **plus** struct-arg passing **plus** function-constant NSG. | `quantized.metal` + `kernels/quantized.rs` + pipeline infra + new args struct | **~400 LOC** | +1.5 to +2.5 tok/s (same as min; max adds no further wall-clock). | Medium-high. All the risks of min + mid compound. |

#### Combined 1bNEW.29 port (recommended scope)

**Recommended scope for 1bNEW.29 if the other agents' measurements confirm Q6_K is on the hot path:**

- **Q4_0: nothing.** No identified lever.
- **Q6_K: Q6_K-min (NR0=2 row loop port + align change).**
- **Skip the struct-arg rewrite entirely.** It's a CPU win hidden behind GPU per 1bNEW.22's already-published data. Not a wall-clock recoverable lever.

**Total: ~120 LOC, ~1.3-2.5 tok/s, Low-medium risk.**

#### Full parity port (not recommended)

If a future spike decides to match llama.cpp byte-for-byte across both families:

- Q4_0-max + Q6_K-max + function-constant NSG mechanism + compile-pipeline caching for per-dtype NSG values.
- **Total: ~800-1100 LOC**, **~1.5-3 tok/s** wall-clock (dominated by Q6_K-min's contribution), **High** risk because it's a kernel-infrastructure rewrite.

---

## D. Surface-area estimate for 1bNEW.29

### D.1 Files touched (recommended Q6_K-min scope)

```
vendor/candle-metal-kernels/src/metal_src/quantized.metal       (~80 line replacement at :5186-5266)
vendor/candle-metal-kernels/src/kernels/quantized.rs            (1-line change at :96 — Q6K align 2→4)
```

No changes to `source.rs`, `kernel.rs`, `lib.rs`, or candle-core's consumers. The kernel **name** stays the same (`kernel_mul_mv_q6_K_f32`) so no dispatcher-side lookup change is needed.

### D.2 Total LOC delta estimate

| Scope | Added | Deleted | Net | Churn (added + deleted) |
|---|---|---|---|---|
| **Q6_K-min (recommended)** | ~85 | ~80 | +5 | ~165 |
| **Q4_0-mid + Q6_K-min** | ~155 | ~110 | +45 | ~265 |
| **Q4_0-max + Q6_K-max + FC infra** | ~650 | ~300 | +350 | ~950 |

**Recommended 1bNEW.29 budget: under 200 LOC total churn. Estimated wall-clock to implement: ~1 day (6-8 hours) including sourdough gate verification.**

### D.3 Test surface

candle-metal-kernels has **no existing unit tests for `call_quantized_matmul_mv_t`** (confirmed by Grep for `Quantized|quantized|Q4|Q6|q4_0|q6_K|mul_mv` in `vendor/candle-metal-kernels/src/tests.rs` — zero matches). The 59 vendored tests (per the 1bNEW.22 addendum) cover command-pool mechanics, not kernel numerical correctness.

**Tests that must keep passing:**
- 59/59 vendored candle-metal-kernels tests (`cargo test -p candle-metal-kernels`) — none touch `quantized.metal`, so they should be unaffected, but must be re-run to confirm.
- 307/307 hf2q tests (`cargo test`) — specifically the end-to-end tests that exercise Q6_K matmul paths.
- **Sourdough gate** (`scripts/sourdough_gate.sh`) — mandatory. Must produce ≥3094 byte common prefix vs llama.cpp on the DWQ GGUF per the spike 1bNEW.22 addendum correctness anchor.

**Additional unit tests recommended:**
- **New**: `vendor/candle-metal-kernels/src/tests.rs` — add a `quantized_matmul_mv_q6_K_smoke` test that builds a known Q6_K weight tensor (one or two rows worth of blocks with known values), runs it through `call_quantized_matmul_mv_t` with a fixed src1, and asserts the output matches a hand-computed reference within fp32 tolerance. This is the only way to catch the NR0=2 row-pointer arithmetic bug if it's wrong (because sourdough-gate failures on this would show as gibberish at token position > 0, which is much harder to bisect).
- **New**: at least one Gemma-shape Q6_K test (`[1, 2816] @ [4096, 2816]`) to exercise the exact dispatch grid hf2q uses.

### D.4 Correctness validation strategy (Walk discipline)

Sourdough gate is the mandatory primary correctness gate. It is a **system-level** numerical check, not a kernel-level one, so a correct NR0=2 port followed by a sourdough pass is strong evidence but not proof of kernel correctness — a compensating error elsewhere could still produce the right token sequence.

**Mandatory pre-merge sequence:**

1. **Unit-level smoke test** (new; see D.3): verify the ported Q6_K kernel produces the same output as the pre-port kernel on one or two synthetic block inputs before any full-model run.
2. **Canonical bench** (`./target/release/hf2q generate --model <gguf> --prompt-file tests/bench_prompt_128.txt --max-tokens 128 --temperature 0 --benchmark`) — must produce identical top-1 token sequence to pre-port HEAD for the first few tokens (byte-exact is not expected due to floating-point reassociation, but the decoded tokens should match via argmax).
3. **Sourdough gate** (`scripts/sourdough_gate.sh`) — must produce ≥3094 byte common prefix vs llama.cpp reference output. This is the non-negotiable gate per the 1bNEW.22 addendum.
4. **Fallback flag**: add a `--q6k-kernel=pre-1bNEW29` or similar fallback to the CLI so the old impl is still dispatchable for bisect (mirroring the `--moe-kernel=loop` / `--rope-kernel=loop` pattern documented in 1bNEW.22 spike Run SF at lines 119-153 of `docs/spike-1bNEW22-instrumentation.md`). This lets future spikes measure the marginal contribution of 1bNEW.29 the same way 1bNEW.1/4/6/17/20 are currently measurable.

### D.5 Risk register — top 3

1. **Bound check on non-multiple-of-4 ne01.** Gemma's hf2q Q6_K shapes are all multiples of 4 (4096, 8192, 2048, 1024, 2816), so the NR0=2 dispatch grid math works exactly. But if a future model has ne01 = e.g. 5120, the `(ne01 + 3)/4` dispatch rounds up and the kernel must handle `first_row + 1 >= args.ne0` correctly. **Mitigation**: port the exact `first_row + row < args.ne0` guard from `ggml-metal.metal:8011` verbatim into the candle port; add a non-multiple-of-4 shape to the new unit smoke test.
2. **src1 pointer aliasing across rows.** llama.cpp's row-loop reuses `yl[16]` across 2 rows but the `q1/q2/qh/sc/dh` pointers advance by `nb01` per row (`ggml-metal.metal:8001-8005`). If `nb01` is computed wrong in candle's dispatcher (candle currently passes `nb01 = 0` at `kernels/quantized.rs:44` because the layout is implicit-contiguous), the row-advance will read garbage. **Mitigation**: verify that candle's Q-mul-mv always sees contiguous src0 with `nb01 = QK_K-bytes × nb = row stride in bytes` (which llama.cpp computes from tensor metadata). **Open question**: is candle's `nb00=nb01=nb02=0` pass at `quantized.rs:43-45` a bug that works because the candle kernel ignores them, or an intentional elision? Needs a read-pass over `quantized.metal` to see whether nb01 is actually used in the Q6K body — candle's does **not** use nb01 (it indexes by `x + row*nb`), but llama.cpp's NR0=2 port **would** need nb01 to advance rows.
3. **Template instantiation for `kernel_mul_mv_id_q6_K_f32`.** candle's MoE path uses `kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl>>` at `quantized.metal:7633`. If the Q6K impl becomes templated on nr0, the MoE template instantiation has to be updated to pass the nr0 parameter. **Mitigation**: keep the old non-templated Q6K impl alongside a new `kernel_mul_mv_q6_K_f32_impl_nr0` template, or template the old one and specialize the MoE instantiation at 7633. Needs a read of `kernel_mul_mv_id_t` to confirm the signature doesn't fight the change.

### D.6 Open questions (not resolvable from static read)

1. **Is llama.cpp's actually-faster-at-hf2q-Q6K-shapes an empirical fact, or only a hypothesis?** The current research only established that llama.cpp dispatches 2× fewer Q6_K threadgroups. Whether that translates to a wall-clock win depends on M5 Max occupancy at hf2q's specific ne01 values. Q6_K at `[1, 2816] @ [4096, 2816]` dispatches 1024 tg (llama.cpp) vs 2048 tg (candle). If the M5 Max is already under-occupied at 1024 tg (likely — M5 Max has ~40 execution units), llama.cpp's halving of the tg count could actually **hurt** occupancy. **This is a runtime-measurement question** for Agent #1 or Agent #2 (nsg microbench / kernel-timing spike).
2. **Does `FOR_UNROLL` yield measurable wins?** Apple's Metal compiler may already unroll short `for` loops with `#pragma unroll` or automatically. The `FOR_UNROLL` macro in llama.cpp is likely a GNU/clang `#pragma` wrapper. **Needs microbench** against a synthetic candle kernel with and without the macro.
3. **Candle's `nb01 = 0` convention — intentional or bug?** `kernels/quantized.rs:43-45` sets `nb00 = nb01 = nb02 = 0` as explicit args to the kernel. Candle's Q6K impl at `quantized.metal:5186-5266` does not reference `nb01` — it uses `x + row * nb` where `nb = ne00/QK_K`. If the 1bNEW.29 port switches to llama.cpp's pattern of `q1 += args.nb01` per row, **candle's dispatcher must start passing a real nb01**. This is a **breaking call-site contract change** inside the vendored kernels. Needs verification that no other caller relies on the current `nb01 = 0` convention.
4. **Does the `kernel_mul_mv_id` MoE template path need the same port?** Per `quantized.metal:7633` and `ggml-metal.metal:10271`, the MoE-id path for Q6_K is the same impl. A Q6_K-min port that only touches the non-MoE impl would leave MoE unchanged. hf2q's MoE blocks are Q4_0 (matching `kernel_mul_mv_id_q4_0_f32` at `quantized.metal:7625`), so Q6_K MoE is not exercised today. **But** the template instantiation at `:7633` will refuse to compile if the Q6K impl signature changes. Needs a one-line fix to that instantiation.
5. **Is there an even cheaper alternative — just change candle's `align` from 2 to 4 at `kernels/quantized.rs:96` without changing the kernel?** **No**: halving the dispatch count without changing the kernel would leave rows 1 and 3 of every 4-row tile unwritten. The kernel body must change. Confirmed by static read of the `row = 2*r0 + sgitg` expression, which only covers 2 rows per tg.
6. **Does the 1bNEW.22 addendum's hypothesis ranking already preempt 1bNEW.29?** Per `docs/spike-1bNEW22-instrumentation.md:307-316`, the addendum ranks "per-kernel GPU compute time" as the most-likely bucket for the 2.37 ms gap, and identifies 1bNEW.29 as the only identified lever. The Q6_K-specific asymmetry found here is the first concrete evidence that per-kernel compute-time difference **exists in the source** for at least one of the two families. Q4_0 has no identified lever. **This means 1bNEW.29's realistic wall-clock envelope, under the diligent-discipline reading, is the Q6_K-only port: +1.3 to +2.5 tok/s.** It does not close the End gate to 107 tok/s on its own.
7. **Is Xcode Instruments → Metal System Trace the right orthogonal validator?** The 1bNEW.22 addendum (spike line 314) explicitly says "the next concrete next-spike action is per-kernel timing comparison via Xcode Instruments / ggml-metal verbose mode." That spike is a **prerequisite** to 1bNEW.29 under mantra discipline — you measure first, then port. Has Agent #1 or Agent #2 been asked to do that measurement? If not, 1bNEW.29 remains speculative regardless of how clean the port looks.

---

## Coordination notes

**Dependencies on other CFA agents:**
- **Agent #1 / Agent #2** (if doing nsg or per-kernel microbench) should be told that **NSG=2 is the correct value for both Q4_0 and Q6_K per llama.cpp's own source**, so the NSG sweep at ADR-005:898-916 is expected to show nsg=2 as optimal. A non-2 optimum would be a **novel** finding, not a matching one.
- **Agent #1 / Agent #2** should specifically measure whether llama.cpp's Q6_K per-kernel wall-clock at `[1, 2816] @ [4096, 2816]` (single matmul) is faster than candle's `call_quantized_matmul_mv_t` at the same shape. If parity, 1bNEW.29 is falsified and needs a new lever. If llama.cpp is faster by ≥10%, the Q6K-min port is justified.

**What the parent/synthesizer should compute from this data:**
- Estimated LOC delta for Q6_K-min: ~120-180 LOC.
- Estimated implementation time: ~1 day (6-8 hours).
- Estimated wall-clock gain: +1.3 to +2.5 tok/s (bounded by Q6_K's share of decode time).
- Correctness gate: sourdough + new unit smoke test.
- Confidence the port is justified: **conditional on Agent #1/#2 measurements**. The static evidence (NR0=2 asymmetry) is firm, but the runtime wall-clock impact is not measurable from reading alone.

---

## Return summary

- **summary**: Static citation map compiled for llama.cpp's `kernel_mul_mv_q4_0_f32` and `kernel_mul_mv_q6_K_f32` vs their candle-metal-kernels counterparts. **Key finding**: NSG values are identical (both =2) in both trees; the only real kernel-code asymmetry is **Q6_K's NR0 parameter — llama.cpp templates on nr0=2 and loops 2 rows per simdgroup, candle hardcodes 1 row per simdgroup**. This means candle dispatches **2× more Q6_K threadgroups** per matmul for hf2q's attention q/k/o_proj shapes. Q4_0 has no identified lever (both trees match exactly). Recommended 1bNEW.29 scope: **Q6_K NR0=2 row-loop port only, ~120 LOC, +1.3-2.5 tok/s estimated, conditional on runtime measurement agents confirming llama.cpp's Q6_K is actually faster at hf2q shapes on M5 Max**.
- **files_changed**: `docs/spike-1bNEW29-research-notes.md` (this file, new).
- **candle_kernel_origin**: **modified llama.cpp** (older ggml-metal.m snapshot, frozen; not MLX-derived). Evidence: ggml references in-file at `quantized.metal:215/241/1724/1829/1911/1959`, byte-for-byte identical `block_q_n_dot_y` helper structure.
- **llama_cpp_nsg_heuristic**: **static per-dtype**. `ggml-metal-device.cpp:702-879` sets `nsg = N_SG_Q*` from the `#define`s in `ggml-metal-impl.h:11-72` without any per-shape branching for Q-types. F32/F16/BF16 get `nsg = min(4, (ne00+127)/128)` at `:721-738`; all quantized types get their static macro value. Q4_0: N_SG=2. Q6_K: N_SG=2. (Note: ADR-005:902's claim that NSG selection is "per ggml_metal_op_mul_mat_id heuristics" conflates function-constant specialization infrastructure with runtime shape tuning. The infrastructure exists; the shape-tuning does not.)
- **min_port_loc_estimate**: **~120 LOC** (Q6_K-min alone: rewrite one impl function + change one align value).
- **max_port_loc_estimate**: **~950 LOC** (full parity across Q4_0 + Q6_K + function-constant NSG infra + struct-arg passing).
- **top_risks**:
  1. **`nb01 = 0` call-site contract**: candle currently passes nb01 as 0 at `kernels/quantized.rs:43-45` because the current Q6K kernel doesn't use it. The ported kernel would need a real nb01 to advance row pointers — this is a breaking change to the Q-mul-mv dispatcher contract inside vendored kernels. Needs audit of all callers (not just QMatMul).
  2. **MoE template instantiation at `quantized.metal:7633`**: the `kernel_mul_mv_id_q6_K_f32` template wraps `kernel_mul_mv_q6_K_f32_impl`. Any signature change to the impl must be mirrored at the MoE template instantiation line, or compilation fails. Low-code-cost, easy to miss.
  3. **Runtime wall-clock payoff unverified**: no static read can prove that halving Q6_K threadgroup count on M5 Max actually yields wall-clock savings. The 1024-tg → 512-tg drop for Gemma k_proj global may cross below the M5 Max occupancy sweet spot. This is a runtime-only question and must be answered by the measurement agents before any port begins.
- **open_questions**:
  1. Does M5 Max benefit from 2× fewer Q6_K threadgroups at hf2q shapes, or does it hurt occupancy? (runtime only)
  2. Is `FOR_UNROLL` in Metal a measurable win over the Metal compiler's default unrolling? (runtime only)
  3. Is candle's `nb00/nb01/nb02 = 0` pass at `kernels/quantized.rs:43-45` intentional or a latent bug? (code-archaeology — needs a pass over every Q-mul-mv caller)
  4. Would the port need to touch `kernel_mul_mv_id_q6_K_f32` too? (needs a read of `kernel_mul_mv_id_t` and `mmv_fn<>` wrapper)
  5. Is the NSG sweep microbench at ADR-005:898-916 still worth running, given the static finding that NSG is already at llama.cpp's value? (prioritization)
  6. Has the Xcode Instruments / Metal System Trace measurement called out as the "next concrete spike action" at `spike-1bNEW22-instrumentation.md:314` been run yet? If not, 1bNEW.29 is blocked on it.
  7. Why does llama.cpp hardcode `N_SG_Q8_0 = 4` (different from all other quants at 2)? Static read of `ggml-metal-impl.h:27` — is there something Q8_0-specific the ggml maintainers know about M-series that would also apply to Q4_0 if the port changed NSG?
- **confidence**: **0.82** on static findings (NR0=2 asymmetry, origin lineage, argument count, NSG static per-dtype, file:line citations). **0.45** on the wall-clock gain estimate (bounded above by the Q6_K share of decode time and below by zero, depending on M5 Max occupancy behavior). **0.65** on the recommended scope (Q6_K-min only) being the right call — conditional on Agent #1/#2's runtime data.
- **blockers**: none for the research deliverable itself. For the *port work that this research supports*, the blocker is **the runtime per-kernel timing measurement** (Xcode Instruments or equivalent on llama.cpp vs hf2q at Q6_K shapes). Without that measurement, 1bNEW.29 remains speculative even though the source-level asymmetry is now well-documented.
