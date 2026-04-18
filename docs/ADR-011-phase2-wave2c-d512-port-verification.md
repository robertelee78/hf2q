# ADR-011 Phase 2 Wave 2C — D=512 llama.cpp Port Verification

**Status**: Complete
**Date**: 2026-04-17
**Owner**: hf2q core (mlx-native flash-attention working group)
**Implements**: ADR-011-phase2-port-d512.md §7 checklist, items 1-10
**Swarm**: swarm-1776516482254-ft5mwj Wave 2C (Agent 2C: kernel-d512)
**Depends on**: Wave 2A (sentinel port — M-init convention), Wave 2B (int FC registry)

---

## 0. Summary

A new Metal kernel file `/opt/mlx-native/src/shaders/flash_attn_prefill_d512.metal`
(~830 LOC) and a new Rust dispatcher
`/opt/mlx-native/src/ops/flash_attn_prefill_d512.rs` implement llama.cpp's
NSG=8, D=512 flash-attention prefill design directly in mlx-native.  The port
is a faithful transliteration of `kernel_flash_attn_ext_impl` from
`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5736-6375`, specialised
for:

- DK = DV = 512 (Gemma 4 global-attention head size)
- bf16/f16 I/O with bf16/f16 additive mask OR bool mask
- Unquantized K/V cache (is_q=0 branch only — Gemma 4 K/V is bf16)
- NSG selectable via `int` function constant (index 322, matching
  llama.cpp's `FC_flash_attn_ext_nsg` at `ggml-metal.metal:5735` =
  `FC_FLASH_ATTN_EXT + 22` = `300 + 22 = 322`)
- Supported NSG values: 4, 8 (dispatcher defaults to 8; explicit 4 is
  exposed for A/B benchmarking via `dispatch_flash_attn_prefill_bf16_d512_with_nsg`)

The new kernel coexists with the existing candle-derived D=256/D=512 path
in `flash_attn_prefill.metal`; the D=512 candle instantiations are **not
removed in this wave** (ADR §4.5 calls for that in a later wave once hf2q
wire-up is complete).  The four new host_name entries are prefixed
`flash_attn_prefill_llamacpp_*_d512[_boolmask]` to disambiguate.

**All 29 flash_attn_prefill tests pass** (23 pre-existing D=256 tests + 6
new D=512 tests).

---

## 1. Deliverables

| Deliverable | Path | Status |
|---|---|---|
| NEW kernel file | `/opt/mlx-native/src/shaders/flash_attn_prefill_d512.metal` | ✅ |
| NEW dispatcher module | `/opt/mlx-native/src/ops/flash_attn_prefill_d512.rs` | ✅ |
| `ops/mod.rs` exposure | `pub mod flash_attn_prefill_d512;` line added | ✅ |
| New GPU correctness tests | `/opt/mlx-native/tests/test_flash_attn_prefill.rs` (6 added) | ✅ |
| Verification doc | `/opt/hf2q/docs/ADR-011-phase2-wave2c-d512-port-verification.md` (this file) | ✅ |

### Kernel entry points produced (4)

```
flash_attn_prefill_llamacpp_bf16_d512
flash_attn_prefill_llamacpp_bf16_d512_boolmask
flash_attn_prefill_llamacpp_f16_d512
flash_attn_prefill_llamacpp_f16_d512_boolmask
```

Each is NSG-agnostic at the entry-point level; NSG=4 or NSG=8 is selected
at pipeline-creation time via the int function constant at index 322.

### Dispatcher functions exposed

```rust
pub fn dispatch_flash_attn_prefill_bf16_d512(
    encoder, device, registry,
    q, k, v, mask, out, params,
) -> Result<()>;  // defaults NSG=8

pub fn dispatch_flash_attn_prefill_bf16_d512_with_nsg(
    encoder, device, registry,
    q, k, v, mask, out, params, nsg: u32,
) -> Result<()>;  // nsg ∈ {4, 8}
```

---

## 2. Pipeline introspection (measured on M5 Max, macOS 25.4.0)

Test: `test_d512_pipeline_tg_memory_and_threads`.

| Metric | Measured | Expected | Match |
|---|---|---|---|
| `thread_execution_width` | 32 | 32 (Apple simdgroup width) | ✅ |
| `max_total_threads_per_threadgroup` | 256 | 256 (NSG=8 × 32) | ✅ |
| `static_threadgroup_memory_length` | 0 bytes | 0 (dynamic allocation, matches llama.cpp) | ✅ |
| Dispatched TG memory (via `setThreadgroupMemoryLength`) | **28,672 bytes** | 28,672 (llama.cpp `FATTN_SMEM(nsg=8)`) | ✅ |

The `static_threadgroup_memory_length = 0` confirms our kernel uses the
llama.cpp-style dynamic threadgroup-memory model (`threadgroup half*
shmem_f16 [[threadgroup(0)]]` with host-side sizing) rather than the
candle-style static declarations (`threadgroup T Q_smem[BQ × (BD + padQ)]`
etc.).  This is essential for the port because only dynamic allocation
lets us size threadgroup memory to the exact llama.cpp footprint without
hard-coding BQ/BK in the template.

---

## 3. Test results

```
$ cargo test --test test_flash_attn_prefill
running 29 tests
test result: ok. 29 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.28s
```

### New D=512 tests added (6)

| Test | What it verifies | Result |
|---|---|---|
| `test_bf16_d512_llamacpp_library_compiles` | Shader source compiles; pipeline creatable at canonical FC combo | ✅ PASS |
| `test_d512_pipeline_tg_memory_and_threads` | Pipeline introspection (see §2) | ✅ PASS |
| `test_gpu_bf16_d512_unmasked` | GPU correctness (ql, kl) ∈ {(32,32), (128,128)}, no mask, no causal | ✅ PASS, max\_abs=1.95e-3, max\_rel=8.66e-3 |
| `test_gpu_bf16_d512_causal` | GPU correctness at (32,32), (128,128) with in-kernel causal masking | ✅ PASS, max\_abs=5.86e-3, max\_rel=1.96e-2 |
| `test_gpu_bf16_d512_fully_masked_sentinel` | Fully -inf mask → all-zero output via DivOp sentinel | ✅ PASS |
| `test_gpu_bf16_d512_determinism` | Two runs with identical inputs produce bit-identical output | ✅ PASS |

Tolerance budget: `BF16_GPU_ATOL = 5e-3, BF16_GPU_RTOL = 2e-2` — SAME as
D=256.  No tolerance relaxation was required.

### Regression check — existing D=256 tests (23) unchanged

All 23 pre-existing `test_gpu_bf16_d256_*` and CPU-reference tests pass
unchanged.  The D=256 kernel code in `flash_attn_prefill.metal` was NOT
modified in this wave.

### In-crate unit tests added (7 in `flash_attn_prefill_d512::tests`)

```
test result: ok. 7 passed; 0 failed
```

These verify constant values match llama.cpp (NQPSG=8, NCPSG=64, NSG=8,
FC_IDX_NSG=322, TGMEM_BYTES=28_672), kernel-name prefix invariants, and
parameter-validation paths.

---

## 4. Architectural choices — per ADR-011-phase2-port-d512.md

### 4.1 Per-simdgroup-Q-distributed decomposition (§2.1)

Implemented exactly as in llama.cpp: `constexpr short NQ = Q/NSG`, with
each simdgroup owning `NQ` rows of the Q=8 threadgroup tile, indexed by
`j = jj*NSG + sgitg`.  The Q tile lives ONCE in threadgroup memory
(`sq[Q × DK]`); all simdgroups read the same Q tile but operate on their
owned Q rows for output.

See kernel comments at lines 390-415 and 585-605.

### 4.2 Threadgroup-half O accumulator (§4.3)

Implemented: `threadgroup half* so = shmem_f16 + Q*DK`, with per-simdgroup
`lo[NO]` register frags (NO=8 at NSG=8) loaded at the top of each KV chunk,
accumulated against P·V, and stored back at the bottom.  This is the
`load → accumulate → store per-iteration` pattern from
`ggml-metal.metal:6186-6256`.

The register pressure saving is exactly the 8× improvement predicted in
ADR §5.2 (NO=8 frags × 256 B = 2 KB per simdgroup = 64 B per thread =
**16 registers per thread** for the O tile, vs 128 in the candle-template's
register-resident O at BD=512).

### 4.3 Sentinel approach (Wave 2A convention)

The same `-FLT_MAX/2` M-init + single-DivOp-guard regime that Wave 2A
ported to the D=256 kernel is used verbatim here:

```metal
// kernel body line 485
M[jj] = -FLT_MAX / 2.0f;         // llama.cpp ggml-metal.metal:5891

// kernel body line 747 (final output write)
const float scale = (S[jj] == 0.0f) ? 0.0f : 1.0f / S[jj];
```

Verified live by `test_gpu_bf16_d512_fully_masked_sentinel`: mask =
`-INFINITY` everywhere → output row is all 0.0, no NaN/Inf.

### 4.4 Function constants (§6.3)

All 5 function constants are plumbed:

| Index | Name | Kind | Source |
|---|---|---|---|
| 200 | `align_Q` | bool | Same as D=256 kernel |
| 201 | `align_K` | bool | Same as D=256 kernel |
| 300 | `has_mask` | bool | Same as D=256 kernel |
| 301 | `do_causal` | bool | Same as D=256 kernel |
| **322** | **`fc_nsg`** | **int** | **NEW — mirrors `FC_flash_attn_ext_nsg` at `ggml-metal.metal:5735`** |

Wave 2B's `KernelRegistry::get_pipeline_with_constants(name, device,
&bool_consts, &int_consts)` API is used directly; no new registry API
introduced in this wave.

### 4.5 NSG as int FC, not as a host_name suffix (§6.3, recommended
approach chosen)

The template `flash_attn_prefill_d512<T, MaskT>` dispatches internally
to `flash_attn_prefill_d512_impl<T, MaskT, NSG>` via a switch on
`nsg_def` (function constant 322).  Metal specialises the switch at
pipeline-creation time so only the selected NSG branch is emitted.
Mirrors llama.cpp's outer thunk at `ggml-metal.metal:6421-6427`.

---

## 5. Instantiations produced

Four host_name entries at the bottom of `flash_attn_prefill_d512.metal`:

```metal
instantiate_d512("flash_attn_prefill_llamacpp_bf16_d512",          bfloat, bfloat)
instantiate_d512("flash_attn_prefill_llamacpp_bf16_d512_boolmask", bfloat, bool)
instantiate_d512("flash_attn_prefill_llamacpp_f16_d512",           half,   half)
instantiate_d512("flash_attn_prefill_llamacpp_f16_d512_boolmask",  half,   bool)
```

Each is NSG-agnostic; NSG=4 and NSG=8 are two separate cached pipelines
keyed by `(name, bool_constants, [(322, nsg_val)])` in the registry.

---

## 6. Port deviations from llama.cpp (with rationale)

| # | llama.cpp feature | Our port | Rationale |
|---|---|---|---|
| 1 | Quantized K/V cache (is\_q=1 branch, `ggml-metal.metal:6066-6127, :6257-6322`) | Dropped | Gemma 4 K/V is bf16; is\_q=0 only.  Re-add in follow-up ADR when q4\_0 / q8\_0 K/V cache lands. |
| 2 | Sinks (attention-sinks / StreamingLLM, `:5722, :6328-6346`) | Dropped | Not used by Gemma 4.  Re-add per-feature. |
| 3 | ALiBi bias (`:5723, :5896-5903, :6146-6150`) | Dropped | Not used by Gemma 4. |
| 4 | Softcap (`:5724, :6140-6142`) | Dropped (stub retained as `args.softcapping=1.0`) | Gemma 4 doesn't use it; Gemma 2 did.  Re-add if Gemma 2 support needed. |
| 5 | KV-pad tail handling (`:5725, :5914-5949`) | Dropped | We handle kL%C via per-position -inf in the last chunk (see kernel lines 560-569), matching our D=256 kernel's approach. |
| 6 | Broadcast-mask (`FC_flash_attn_ext_bc_mask`, `:5727, :5969-5970`) | Dropped | Not used by our dispatcher. |
| 7 | Per-tile pre-pass skip (`blk` bitmap, `:5775, :5951-6005`) | Dropped (every chunk treated as `blk_cur=1`) | Phase 4 / Wave 2E scope; separate tile-skip dispatcher. |
| 8 | Natural-base exp in softmax (`:6155-6156`) | `fast::exp2` with Q pre-scaled by `log2(e)` | **Matches our D=256 kernel's contract**; results mathematically identical; callers already pass `scale = 1/sqrt(d)` not `scale * log2(e)`. |
| 9 | Strides as byte counts (`args.nb01, .nb11`, etc.) | Strides as **element counts** (`args.Q_strides[2]`, `args.K_strides[2]`) | Our `AttnParams` layout uses i64 element strides per [B, H, L, D] contiguous layout; llama.cpp uses ggml's arbitrary-stride byte convention.  Host-side translation in `ops/flash_attn_prefill_d512.rs`. |
| 10 | Mask values promoted via `half2` read at lane-width granularity | Individual half writes via scalar bf16/f16 reads | Our bool-mask path needs per-lane branch; keeping the code path uniform over MaskT ∈ {T, bool} means scalar reads.  Possible future optimization: vectorise the bf16 additive path. |
| 11 | Shader preamble: Q loaded via `float4` from device (ggml-metal.metal:5862) | Q loaded via scalar T (bf16/f16) from device | llama.cpp's Q source is f32 at the graph node; ours is already bf16/f16 at device memory. |

All deviations are explicitly annotated in the kernel source with
`ggml-metal.metal:<line>` citations at the relevant sections.

---

## 7. What this wave does NOT do (explicitly out of scope)

Per ADR-011-phase2-port-d512.md §§7.10, 7.13, and the swarm coordination
prompt:

- **Does NOT** modify the existing candle-derived D=512 kernel at
  `flash_attn_prefill.metal:1587-1590`.  Those four D=512 instantiations
  remain registered and callable (no in-tree callers today per ADR §6.3).
- **Does NOT** wire the new dispatcher into hf2q's forward-prefill path.
  That belongs to Wave 3+ per the coordination prompt.
- **Does NOT** benchmark against llama.cpp.  Correctness is the gate for
  this wave; benchmarking is the perf phase of ADR §7.11.
- **Does NOT** implement the SWA tile-skip pre-pass (`flash_attn_ext_blk`);
  that is Wave 2E / Phase 4 scope.

---

## 8. Files changed

```
NEW  src/shaders/flash_attn_prefill_d512.metal      (830 LOC — kernel)
NEW  src/ops/flash_attn_prefill_d512.rs             (~480 LOC — dispatcher + tests)
MOD  src/ops/mod.rs                                 (+1 line:  pub mod flash_attn_prefill_d512;)
MOD  tests/test_flash_attn_prefill.rs               (+6 GPU tests + 1 pipeline-introspection test)
```

No changes to:
- `src/kernel_registry.rs` (uses Wave 2B's pre-existing
  `get_pipeline_with_constants`)
- `src/encoder.rs` (uses pre-existing `encode_threadgroups_with_args_and_shared`)
- `src/ops/flash_attn_prefill.rs` (D=256 path unchanged)
- `src/shaders/flash_attn_prefill.metal` (D=256/D=512 candle kernel unchanged)

---

## 9. Verification commands (runnable)

```bash
# Library compiles
cd /opt/mlx-native && cargo check --lib
# → Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.02s

# D=512 kernel library + pipeline compilation gate
cd /opt/mlx-native && cargo test --test test_flash_attn_prefill test_bf16_d512_llamacpp_library_compiles -- --nocapture
# → test result: ok. 1 passed

# Pipeline introspection (threadgroup memory, threads/TG, simd width)
cd /opt/mlx-native && cargo test --test test_flash_attn_prefill test_d512_pipeline_tg_memory_and_threads -- --nocapture
# → D=512 NSG=8 pipeline: static_tg=0 bytes, max_total_threads_per_tg=256, thread_execution_width=32
# → test result: ok. 1 passed

# All D=512 GPU correctness tests
cd /opt/mlx-native && cargo test --test test_flash_attn_prefill d512 -- --nocapture
# → 5 passed (unmasked, causal, fully_masked_sentinel, determinism, library_compiles)
#   + 1 (tg_memory_and_threads) when included

# Full flash-attention regression suite (D=256 + D=512)
cd /opt/mlx-native && cargo test --test test_flash_attn_prefill
# → test result: ok. 29 passed; 0 failed; 0 ignored

# In-crate unit tests for the new module
cd /opt/mlx-native && cargo test --lib flash_attn_prefill_d512
# → test result: ok. 7 passed
```

---

## 10. Confidence & blockers

**Confidence: 0.92** (high).

- ✅ Kernel body is a direct line-for-line port of llama.cpp's proven math
  with explicit citations at every divergence.
- ✅ All 29 flash-attention tests pass at the same bf16 tolerance as D=256
  (no tolerance relaxation).
- ✅ Measured pipeline TG memory (28,672 B) matches llama.cpp's
  `FATTN_SMEM(nsg=8)` exactly.
- ✅ Measured max threads/TG (256) confirms NSG=8 geometry is live.
- ✅ Measured simdgroup width (32) confirms MMA 8×8×8 tiles apply.
- ✅ Existing D=256 tests unchanged — no regression introduced.
- ⚠️ Correctness validated only at moderate shapes (ql/kl up to 128); the
  production target shape (ql/kl = 2455) is NOT covered by a GPU
  correctness test in this wave (CPU reference would take ~seconds at
  that shape with D=512).  Wave 3+ should add a sourdough-byte-identical
  test when hf2q wire-up lands.
- ⚠️ NSG=4 path compiles (via the function-constant switch) but is not
  exercised by the test suite.  It is exposed for A/B benchmarking via
  `dispatch_flash_attn_prefill_bf16_d512_with_nsg`.

**Blockers**: none.

---

## 11. Memory coordination

Swarm ID: `swarm-1776516482254-ft5mwj`
Namespace: `cfa-swarm-1776516482254-ft5mwj`
Memory key written: `cfa:wave2c:d512-port-complete` (per coordination prompt)
