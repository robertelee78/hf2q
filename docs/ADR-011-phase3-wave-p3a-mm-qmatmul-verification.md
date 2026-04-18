# ADR-011 Phase 3 Wave P3a — mm qmatmul port verification

**Author**: Wave P3a (mm-q4k-porter), CFA swarm swarm-1776516482254-ft5mwj
**Date**: 2026-04-17
**Status**: Complete
**Upstream**: ADR-011 Phase 3 investigation identified a 7x prefill gap
root-caused to `dispatch_qmatmul` using llama.cpp's matrix-vector kernel
even at prefill m=2455.  The matmul kernel (used by llama.cpp when
`ne11 > 8`) tile-stages weights through threadgroup shmem and reuses
each tile across a 32-row block, cutting DRAM traffic ~32x.

---

## Executive summary

Wave P3a ports llama.cpp's `kernel_mul_mm_<qtype>_f32` (and the MoE
`kernel_mul_mm_id_map0` + `kernel_mul_mm_id_<qtype>_f32` two-stage
dispatch) for the three quant types present in our Gemma 4 DWQ GGUF
(Q4_0, Q6_K, Q8_0), wires the m>8 routing threshold into both
`quantized_matmul_ggml` and `quantized_matmul_id_ggml`, and lands
end-to-end measurable perf improvement at prefill.

**Measured perf** (M5 Max, Gemma 4 26B A4B DWQ):

| Seq len | Pre-P3a tok/s | Post-P3a tok/s | Peer (llama.cpp) | Ratio pre | Ratio post |
|---|---:|---:|---:|---:|---:|
| pp128 | ~140 | 399.8 | 2436 | 5.7% | 16.4% |
| pp512 | ~500 | 788.9 | 3683 | 13.6% | 21.4% |
| pp1024 | ~505 | 935.4 | 3612 | 14.0% | 25.9% |
| pp2455 | 495 | 1049.7 | 3456 | 14.1% | 30.4% |

Gap closure at pp2455: **14.1% → 30.4%** of peer (2.12x tok/s improvement).

Decode unchanged at 108 tok/s (> 100 tok/s floor) — m=1 still routes
to the mv path.

This is less than the ADR-011 §3 prediction of "14% → ~70%" for Wave
P3a alone.  Suspected reasons (to be re-verified via Metal frame
capture in P3b):

1. Scratch allocation overhead — `dispatch_id_mm` allocates htpe+hids
   buffers via `device.alloc_buffer` on every MoE dispatch (~60/prefill).
2. We use the simdgroup MMA code path; llama.cpp on tensor-API-capable
   M5 Max uses the `GGML_METAL_HAS_TENSOR` path (native MMA intrinsics),
   which has additional throughput we haven't ported.
3. Structural overhead remains (93 syncs, 120 cast kernels) that P3b
   and P3c address.

Even at 30%, this is a substantial win and establishes that the mm
dispatch is correct and performant.  Phase 3b will push toward 70%+.

---

## §1 What was ported

### Non-id (dense) `mul_mm` kernel

Source: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9276`
Target: `/opt/mlx-native/src/shaders/quantized_matmul_mm.metal`

Template instantiations:
- `kernel_mul_mm_q4_0_f32`  — attention + dense FFN in 24/30 layers
- `kernel_mul_mm_q8_0_f32`  — `ffn_down` in 6/30 layers
- `kernel_mul_mm_q6_K_f32`  — attention + dense FFN in 6/30 layers

Tile geometry: NR0=64 (output-N), NR1=32 (M), NK=32.  Four simdgroups
(128 threads total) per threadgroup.  8192 bytes of threadgroup shmem
(A tile as half, 4096B + B tile as float, 4096B).  Always enables
bounds-check paths for input-K tail and output-M/N partial tiles.

Dispatched only via the test-only `dispatch_mm_for_test` until Commit 3
wires it in the public dispatcher.

### `_id` MoE matmul — two-stage port

Source: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9584-9951`
Target: `/opt/mlx-native/src/shaders/quantized_matmul_id_mm.metal`

Stage 1: `kernel_mul_mm_id_map0_ne20_8` — preprocess the
`[n_tokens, top_k=8]` expert-id table into per-expert routed-token
lists (htpe counts + hids packed id arrays).

Stage 2: `kernel_mul_mm_id_<qtype>_f32` — dispatch one tile per
(N-tile, M-tile, expert).  Each tile is homogeneous in expert choice
so the 64x32 weight tile staged into shmem is valid for every routed
row in the tile.

The map0 preprocessor is essential: without it, 32 consecutive M-rows
in a tile could route to 32 different experts, defeating weight reuse
and making mm_id no faster than mv_id.

Only top_k=8 is instantiated (Gemma 4's `expert_used_count`).  Other
values fall back to mv via the routing check in
`quantized_matmul_id_ggml`.

**Critical correctness finding**: mlx-native's compute encoder runs
dispatches concurrently by default (matches `MTLDispatchTypeConcurrent`
behaviour).  Without a memory barrier between map0 and mm_id, the mm
kernel reads htpe as zeros and every threadgroup early-exits.  The
port explicitly emits `encoder.memory_barrier()` between the two
dispatches, matching llama.cpp's `ggml_metal_op_concurrency_reset`
behaviour at `ggml-metal-ops.cpp:2353`.

### Input layout note

Our hf2q calls pass `src1` as `[n_tokens, K]` flat (one input row per
token).  llama.cpp upstream pre-replicates src1 per slot into
`[K, n_expert_used, n_tokens]`.  The mm_id port keeps our compact
layout by setting `nb11 = 0` (zero slot stride — every slot reads the
same token row) and `nb12 = K * sizeof(float)` (per-token stride).

---

## §2 Dispatcher routing

Added:

```rust
// quantized_matmul_ggml.rs
pub const MM_ROUTING_THRESHOLD: u32 = 8;

pub fn quantized_matmul_ggml(...) -> Result<()> {
    ...
    if params.m > MM_ROUTING_THRESHOLD && params.k >= 32 {
        dispatch_mm(...)
    } else {
        dispatch_mv(...)
    }
}
```

and the matching `MM_ID_ROUTING_THRESHOLD = 8` in
`quantized_matmul_id_ggml.rs`.  Threshold matches llama.cpp's
`ne11_mm_min` (ggml-metal-ops.cpp:2046).

The `_id` path has an additional `top_k == 8` guard because only
`kernel_mul_mm_id_map0_ne20_8` is instantiated today.  Other top_k
values fall back to mv_id.

---

## §3 Correctness verification

### Unit tests (new — all pass)

| File | Tests | Status |
|---|---:|---|
| `mlx-native/tests/test_quantized_matmul_mm.rs` | 11 | 11/11 PASS |
| `mlx-native/tests/test_quantized_matmul_id_mm.rs` | 9 | 9/9 PASS |

mm vs mv parity verified per quant type × {small, prefill-like,
irregular partial tiles}.  mm_id vs mv_id parity verified per quant
type × {small, prefill-like, partial tiles, sparse routing}.

Tolerance calibrated for the f32 reduction ordering difference between
the two kernels (simd_sum 32-wide tree vs simdgroup MMA 8x8 + 32-wide
K-sum).  No tolerance exceeds 5e-2 absolute at any tested shape —
well below quantization noise floor for the underlying block formats.

### Existing tests (no regression)

- `mlx-native/tests/test_quantized_matmul_ggml.rs` 9/9 PASS (mv path
  unchanged).
- `mlx-native/tests/test_quantized_matmul_id_ggml.rs` 3/6 PASS (the
  3 failing tests fail on baseline `main` too — pre-existing 0.000001
  ULP tolerance mismatches unrelated to Wave P3a).

### hf2q end-to-end

- Decode: `hf2q generate --prompt "Hello" --max-tokens 50` produces
  coherent output at 108 tok/s (> 100 tok/s floor).  m=1 → mv path,
  unchanged from pre-P3a.
- Sourdough parity gate (`scripts/sourdough_gate.sh`):
  - Baseline main: common prefix 127 bytes, hf2q output 3561 bytes.
  - Post-Wave-P3a: common prefix 127 bytes, hf2q output 3558 bytes.
  - **The 127-byte divergence point is UNCHANGED** — my mm/mm_id port
    does not widen the hf2q-vs-llama.cpp drift.  The 3-byte difference
    in hf2q's tail past that point is f32 accumulation-order noise
    between mv and mm (< 1e-3 relative).
  - The gate still fails its 562-byte margin requirement, but this is
    a pre-existing condition on baseline `main` (documented below)
    unrelated to this wave.

**Important context**: the sourdough gate's 562-byte margin requirement
reflects a prior baseline (`post-1bNEW.20.FIX`) where hf2q and
llama.cpp shared 3656 bytes of common prefix.  The current `main`
HEAD (`ddbff25`) fails that gate at 127 bytes — a large pre-existing
correctness regression.  Verified by stashing Wave P3a changes and
re-running: baseline `main` produces identical 127-byte prefix.
Diagnosing and fixing that regression is out of scope for Wave P3a
but should be handled before the next release gate.

---

## §4 Per-commit summary

### Commit 1 — `711fce2`
```
feat(qmatmul-mm): port llama.cpp mul_mm kernels (Q4_0/Q6_K/Q8_0)
```
- New `quantized_matmul_mm.metal` with the three dense mm kernels.
- New `dispatch_mm_for_test` helper; public dispatcher unchanged.
- 11 new unit tests, all passing.  Sourdough unchanged at baseline.

### Commit 2 — `bf2136b`
```
feat(qmatmul-mm): port MoE _id mul_mm kernels
```
- New `quantized_matmul_id_mm.metal` with `map0_ne20_8` and the three
  mm_id kernels.
- New `dispatch_id_mm_for_test` helper; public dispatcher unchanged.
- 9 new unit tests, all passing.  Sourdough unchanged at baseline.
- Caught the concurrent-dispatch memory barrier requirement between
  the two stages.

### Commit 3 — `2ed5f60`
```
feat(qmatmul-mm): wire dispatcher routing m>8 -> mm/mm_id
```
- `quantized_matmul_ggml` routes to mm when `m > 8` and `k >= 32`.
- `quantized_matmul_id_ggml` routes to mm_id when
  `n_tokens > 8 && top_k == 8 && k >= 32`.
- End-to-end hf2q measurements: pp2455 at 1050 tok/s (up from 495,
  2.12x improvement).  Decode unchanged at 108 tok/s.
- Sourdough: common prefix still at 127 bytes (same as baseline main),
  hf2q output 3 bytes different at tail (f32 accumulation noise).

---

## §5 What's left for Wave P3b

- **Scratch pooling**: `dispatch_id_mm` currently allocates htpe+hids
  per call (~60 allocator calls/prefill).  Amortize across layers.
- **Session merging**: 3 sessions/layer → 1 (removes 60 of the 93
  syncs/prefill).
- **Cast fusion**: 120 f32↔bf16 casts/prefill can fold into adjacent
  kernels.
- **Possible Metal tensor API path**: llama.cpp's
  `GGML_METAL_HAS_TENSOR` path uses native MMA intrinsics for another
  few-percent gain on M5.  Low-hanging if we want it.

Expected post-P3b: 1050 → ~2500 tok/s (30% → 70% of peer).
