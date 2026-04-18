# ADR-011 Phase 2 — Wave 2E tile-skip pre-pass verification

**Author:** Agent 2E (tile-skip), CFA swarm `swarm-1776516482254-ft5mwj`
**Date:** 2026-04-17
**Status:** Implemented and verified. All bit-exact and classification gates pass.
**Port target:** llama.cpp `kernel_flash_attn_ext_blk` at
`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5666-5719` plus the
main-kernel consumption at `:5955-5963, :6145`.
**Port spec:** `/opt/hf2q/docs/ADR-011-phase2-port-tile-skip.md`.

---

## 0. TL;DR

Landed a Metal tile-classifier pre-pass and integrated it into both D=256
and D=512 flash_attn_prefill kernels. The pre-pass writes one byte per
`(qtile, ktile)` of the mask into a scratch device buffer; the main
kernels read that byte at the top of their KV-tile loop and:

- `byte == 0`: `continue` past the entire KV tile (K-load, Q·K^T,
  mask-add, V-load, P·V all skipped).
- `byte == 1`: normal mask-add + softmax path (unchanged from pre-Wave-2E).
- `byte == 2`: skip the mask-add; compute Q·K^T + softmax normally.

On Gemma 4's sliding prefill shape (qL=2455, window=1024, D=256) the
classifier reports **65.4% `skip`, 31.9% `all_attended`, 2.7% `mixed`** —
exceeding ADR-011 §3.3's predicted ~58.5% skip by ~7 percentage points.

All 39 flash_attn_prefill integration tests pass, including 5 new
Wave 2E tests. The critical `_with_blk_matches_no_blk` bit-exact gates
(both D=256 and D=512) confirm that enabling the pre-pass does not alter
output by a single bf16 bit.

---

## 1. Files changed

| File | Status | Purpose |
|------|--------|---------|
| `/opt/mlx-native/src/shaders/flash_attn_prefill_blk.metal` | NEW | Pre-pass classifier kernel (~150 lines incl. comments) |
| `/opt/mlx-native/src/ops/flash_attn_prefill_blk.rs`         | NEW | Host dispatcher + `BlkParams` + `alloc_blk_buffer` |
| `/opt/mlx-native/src/shaders/flash_attn_prefill.metal`      | MODIFIED | `has_blk` FC, buffer(7) binding, KV-loop skip branch, mask-add gating on `blk_cur != 2` |
| `/opt/mlx-native/src/shaders/flash_attn_prefill_d512.metal` | MODIFIED | Same treatment as D=256 |
| `/opt/mlx-native/src/ops/flash_attn_prefill.rs`             | MODIFIED | Added `dispatch_flash_attn_prefill_bf16_d256_with_blk`; existing dispatcher delegates with `blk=None` |
| `/opt/mlx-native/src/ops/flash_attn_prefill_d512.rs`        | MODIFIED | Added `dispatch_flash_attn_prefill_bf16_d512_with_blk` and `…_with_nsg_and_blk` |
| `/opt/mlx-native/src/ops/mod.rs`                            | MODIFIED | Expose the new `flash_attn_prefill_blk` module |
| `/opt/mlx-native/tests/test_flash_attn_prefill.rs`          | MODIFIED | Added 5 new tests + helper (`cpu_classify_blk`, `build_mask_and_blk`, `read_u8_buffer`) |

Grand total: 2 new files, 6 modified. Net addition: ~900 LoC (shader +
Rust + tests + comments).

---

## 2. Kernel name + function-constant map

### 2.1 Pre-pass kernel

- Entry point: `flash_attn_prefill_blk_bf16`
- Shader: `/opt/mlx-native/src/shaders/flash_attn_prefill_blk.metal`
- Function constants:
  - `400` (int) — `BQ_blk`: Q-rows per tile. `32` for D=256, `8` for D=512.
  - `401` (int) — `BK_blk`: K-cols per tile. `16` for D=256, `64` for D=512.
  - (No bool constants; all behaviour is driven by `BQ_blk` × `BK_blk`.)

### 2.2 Main-kernel addition (D=256 and D=512)

- New function constant `303` (bool) — `has_blk`.
- New buffer binding `7` (only bound when `has_blk == true`) — the classification byte buffer.
- Same index across both D=256 (`flash_attn_prefill.metal`) and D=512 (`flash_attn_prefill_d512.metal`) so the dispatcher helpers don't need to fork.

### 2.3 Blk byte encoding

| byte | Meaning | Main-kernel action |
|------|---------|--------------------|
| `0` | Fully masked (`mmax <= -1e30f`) | `continue` — skip the whole KV tile |
| `1` | Mixed | Normal mask-load + mask-add path (pre-Wave-2E behaviour) |
| `2` | All attended (`mmin == mmax == 0.0`) | Skip mask-add; compute Q·K^T + softmax normally |

Matches llama.cpp's three-way encoding at
`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5704-5710`. The
threshold `-1e30f` is a deliberate deviation from llama.cpp's `-MAXHALF`
(≈ -65504) because our bf16 mask carries a true `-INFINITY` (0xFF80)
sentinel, not the f16 `-MAXHALF` saturation. The conservative `-1e30`
threshold catches both representations plus any future finite "very
negative" sentinel without requiring exact-representation checks. See
port spec §5.2 Note 1.

---

## 3. Tile-shape choice rationale

The pre-pass's `(BQ, BK)` MUST match the main kernel's KV-tile loop
indexing. A mismatch would make `blk[qt][kt]` point at the wrong tile
and silently corrupt the classification.

| Kernel | BQ (Q-rows per tile) | BK (K-cols per tile) | Source |
|--------|----------------------|----------------------|--------|
| D=256 main (`flash_attn_prefill.metal`) | 32 | 16 | Existing tile geometry (pre-Wave-2E) |
| D=256 pre-pass | 32 | 16 | Matches D=256 main |
| D=512 main (`flash_attn_prefill_d512.metal`) | 8 (`NQPSG`) | 64 (`NCPSG`) | Existing llama.cpp-derived tile geometry |
| D=512 pre-pass | 8 | 64 | Matches D=512 main — outer `ic0` loop steps by C=64 per chunk |

**Note on D=512 pre-pass BK = 64, not 8**: the task description
specified `(BQ=8, BK=8)` for D=512, but reading the D=512 shader in
detail (`flash_attn_prefill_d512.metal:530-540`) confirms the outer KV
loop steps by `C = NCPSG = 64` per chunk, NOT by a hypothetical BK=8.
The `blk[qt][kt]` index must correspond to `(tgpig.x, ic0)`, and `ic0`
ranges over `[0, ceil(kL/64))`. Using BK=8 would make the pre-pass
produce `8×` too many tiles and the D=512 main kernel would read past
the correct blk row. BK=64 is the correct port.

The task description's BK=8 was an error from the original Agent #4 port
spec; the actual D=512 main kernel outer-loop step is 64. Using BK=64
matches the D=512 kernel's existing behaviour exactly. This is a
deviation from the task description, documented here for the downstream
record.

---

## 4. Dispatcher signature (design decision)

**Chose:** extend the existing dispatchers with an optional `blk:
Option<&MlxBuffer>` parameter via a new `_with_blk` function, with the
existing dispatcher delegating with `blk=None`. Rationale:

1. **Zero breaking-change**: all existing call sites work without
   modification. The existing dispatcher delegates to the new one with
   `blk=None`; the compiled pipeline with `has_blk=false` dead-codes
   every blk reference, so the compiled machine code is identical to
   pre-Wave-2E for callers that don't opt in.

2. **Single source of truth**: a single body handles all combinations
   of `(has_mask, has_blk)`. Avoids drift between twin implementations.

3. **Pipeline cache locality**: `has_blk` joins the existing
   `(align_Q, align_K, has_mask, do_causal)` bool-constant tuple. The
   cache key gains one `b0`/`b1` suffix — one extra pipeline per
   existing combination, compiled on first use.

### 4.1 Rust signatures

```rust
// D=256 — new blk-aware dispatcher
pub fn dispatch_flash_attn_prefill_bf16_d256_with_blk(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    blk: Option<&MlxBuffer>,   // <-- new
    out: &mut MlxBuffer,
    params: &FlashAttnPrefillParams,
) -> Result<()>;

// Pre-existing dispatcher — now delegates with blk=None.
pub fn dispatch_flash_attn_prefill_bf16_d256(…) -> Result<()>;

// D=512 equivalents (mirrors D=256).
pub fn dispatch_flash_attn_prefill_bf16_d512_with_blk(…, blk: Option<&MlxBuffer>, …) -> Result<()>;
pub fn dispatch_flash_attn_prefill_bf16_d512_with_nsg_and_blk(…) -> Result<()>;

// New pre-pass dispatcher
pub fn dispatch_flash_attn_prefill_blk(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    mask: &MlxBuffer,
    blk_out: &MlxBuffer,
    params: &BlkParams,
) -> Result<()>;

// Allocator helper
pub fn alloc_blk_buffer(device: &MlxDevice, params: &BlkParams) -> Result<MlxBuffer>;
pub fn blk_buffer_byte_len(params: &BlkParams) -> Result<usize>;
```

---

## 5. Correctness bug found and fixed during verification

### 5.1 The bug — block-loader advance missed on skip

The D=256 main kernel uses stateful `KBlockLoader` / `VBlockLoader`
objects that advance their internal `src` pointer via `.next()` at the
END of every KV-tile iteration. My initial implementation of the
`blk_cur == 0` skip branch used a bare `continue` — which skipped past
the `loader_k.next()` / `loader_v.next()` calls.

Consequence: the NEXT KV-tile iteration loaded the SAME K/V data as
the tile before the skip, scoring the wrong K/V pair. Outputs diverged
from the non-blk path at ~50% of all elements in the bit-exact test.

### 5.2 The fix

Advance both loaders BEFORE `continue`:

```metal
if (blk_cur == 0) {
  loader_k.next();
  loader_v.next();
  continue;
}
```

This restores strict invariance: the blk path reaches the same load
offsets on every iteration as the non-blk path, just with the
intermediate MMA work elided for `blk_cur == 0` tiles. Confirmed by
`test_gpu_bf16_d256_with_blk_matches_no_blk` passing with zero
bit-differences across 32 768 bf16 output elements.

### 5.3 Why the D=512 kernel is not affected

The D=512 main kernel recomputes its K/V device pointers from
`ic = ic0 * C` at the top of every iteration (no stateful loader) —
see `flash_attn_prefill_d512.metal:554, :759` (the
`const device T* pk = k_head + (ulong)ic * NS10` and similar V
pointer computations). `continue` is safe there as-is. The
`test_gpu_bf16_d512_with_blk_matches_no_blk` test passed on the first
run.

**Lesson for the record:** porting skip-branches into kernels that use
stateful iterators requires per-iteration invariants on iterator state
— llama.cpp's own `continue` at `ggml-metal.metal:5961` handles a
per-row `pm2[jj] += NW` advance inside the skip branch for exactly
this reason. Without the iterator-state advance, the skip breaks
correctness even though it appears to be a pure performance
optimisation.

---

## 6. Test results

All tests run on Apple M5 Max, macOS Sequoia, Metal 3.1, `cargo test`
in `--dev` profile. Command:

```bash
cd /opt/mlx-native && cargo test --test test_flash_attn_prefill
```

Result: **39 passed, 0 failed, 0 ignored** (24.95 s total).

### 6.1 New tests added (§7 of port spec)

| # | Test | Result | Notes |
|---|------|--------|-------|
| 1 | `test_blk_global_causal_d256` | PASS | seq=64, D=256 — verifies upper-triangle tiles=0, lower-triangle full tiles=2, diagonal tiles=1; cross-check against CPU classifier. |
| 2 | `test_blk_sliding_window_d256` | PASS | seq=256, window=64, D=256 — cross-check GPU vs CPU on every tile; verify all three classes appear. |
| 3 | `test_gpu_bf16_d256_with_blk_matches_no_blk` | PASS | **CRITICAL BIT-EXACT GATE** — 32 768 bf16 elements identical between `blk=None` and `blk=Some(built)`. |
| 4 | `test_gpu_bf16_d512_with_blk_matches_no_blk` | PASS | **CRITICAL BIT-EXACT GATE** — D=512 sibling, 32 768 bf16 elements identical. |
| 5 | `test_blk_gemma4_sliding_prefill` | PASS | Gemma 4 shape (seq=2455, window=1024, D=256) — measures tile-skip rate, cross-checks CPU, runs end-to-end flash_attn_prefill with blk. |

### 6.2 Regression check (34 pre-existing tests)

All pre-existing flash_attn_prefill integration tests continue to pass:

- Structural (params GPU layout): 2/2 PASS
- Library compilation: 2/2 PASS
- Error paths (head_dim, GQA, dtype, sizes): 5/5 PASS
- CPU reference self-consistency: 5/5 PASS
- GPU bf16 D=256 (unmasked, causal, additive mask, fully-masked NaN,
  GQA, custom scale, determinism, unaligned qL/kL): 9/9 PASS
- GPU bf16 D=512 (library, pipeline TG memory, unmasked, causal,
  fully-masked, determinism): 6/6 PASS
- Mask builder + integration: 5/5 PASS

No regressions. The `has_blk=false` pipeline compiles to the same
machine code as pre-Wave-2E (the compiler dead-codes all blk
references), and all dispatchers that don't opt into the blk path
continue to work unchanged.

### 6.3 Module-level unit tests

```bash
cd /opt/mlx-native && cargo test --lib flash_attn_prefill_blk
```

9 passed, 0 failed. Covers byte-size layout, FC-index stability,
buffer sizing for Gemma 4 tile counts (D=256 and D=512), zero-input
rejection, and registration.

---

## 7. Tile-skip rate measured (Gemma 4)

From `test_blk_gemma4_sliding_prefill` output:

```
ql=2455 kl=2455 window=1024 nq=77 nk=154 total=11858
classification — skip(0): 7756 (65.4%), mixed(1): 320 (2.7%), all_attended(2): 3782 (31.9%)
```

- **Skip rate: 65.4%** (ADR-011 §3.3 predicted ~58.5%; exceeded by ~7 pt).
- **All-attended rate: 31.9%** (bonus tiles that save the mask-add cost but still do the Q·K^T + softmax — this is the second-tier optimisation from byte=2).
- **Mixed rate: 2.7%** (the only tiles that pay the full mask-add + softmax path — concentrated along the SWA window boundary).

The measured 65.4% is higher than the ADR's 58.5% prediction because
the ADR accounted for the Q-tile granularity of skip decisions (tiles
overlapping the diagonal or window boundary) but not for the interior
full-skip tiles at the far upper-right (beyond the window across all Q
rows in a tile). The structural skip region is slightly larger than
the analytical prediction. Still strictly within bounds (skip + mixed
+ all_attended = 100.0%, accounted for below tolerance of floating
point).

---

## 8. Port deviations from llama.cpp

Documented deviations (all made consciously; none alter semantics):

1. **Sentinel threshold `-1e30f`** instead of llama.cpp's `-MAXHALF`
   (≈ -65504). Rationale: our bf16 masks carry true `-INFINITY`
   (0xFF80), not the f16 `-MAXHALF` saturation. The conservative wider
   threshold is correct for both representations and robust to future
   sentinel changes. See §2.3 and port spec §5.2 Note 1.
2. **bf16 mask dtype** instead of f16. Rationale: matches the Wave 2D
   mask builder and the rest of the mlx-native prefill pipeline. bf16
   has the same 8-bit exponent as f32 so `-inf` is exact.
3. **2D mask + single-plane blk layout** instead of llama.cpp's
   `[ne33, ne32, NQ, NK]` layout. Rationale: our mask builder produces
   a single `[qL, kL]` plane broadcast across batch and heads via
   `m_strides = [0, 0, kL]`; the blk mirrors that single-plane layout
   for consistency. Saves `8× n_heads` memory with no semantic change.
4. **Tile shape matches our main kernel**, not llama.cpp's. D=256 uses
   `(32, 16)` (vs llama.cpp's `(8, 64)`); D=512 uses `(8, 64)` (which
   happens to match llama.cpp's because the D=512 main kernel IS a
   llama.cpp port). Rationale: tile-shape consistency with the main
   kernel is REQUIRED for the `blk[qt][kt]` index to be correct.
5. **Loader-advance before skip** in the D=256 main kernel. Rationale:
   the mlx-native D=256 kernel uses stateful block loaders that
   advance at end-of-iteration; `continue` must manually advance them
   to preserve per-iteration invariants (see §5). llama.cpp's `continue`
   at `ggml-metal.metal:5961` is equivalent: it performs a per-row
   `pm2[jj] += NW` advance for the mask pointer, which serves the same
   invariant role.
6. **Chesterton's fence preserved**: the old inline `-INF` detection
   (`ggml-metal.metal:5983-6004`, `#if 0`) is NOT reintroduced. The
   pre-pass-kernel architecture — with barrier-free skip decisions —
   is the correct port per llama.cpp's own authorial comment
   ("obsoleted by pre-computing non-masked blocks").

---

## 9. Risk assessment

- **Bit-exact correctness**: VERIFIED at two shapes (D=256 seq=128,
  D=512 seq=64) with 32 768 bf16 elements each. The blk path does not
  alter a single output bit.
- **Classification correctness**: VERIFIED against a CPU reference on
  every tile of a global-causal 64×64 mask and a sliding-window
  256×256 mask, plus spot-checks on the Gemma 4 2455×2455 mask.
- **Pipeline compile time**: one additional `(align_Q, align_K,
  has_mask, do_causal, has_blk)` pipeline per pre-existing combo. Paid
  once at first-dispatch time, cached afterwards.
- **Pre-pass dispatch cost**: at Gemma 4 shape (77 × 154 = 11858
  threadgroups × 32 lanes each = 380k threads) the pre-pass dispatch
  is expected to be in the 10-50 μs range on M5 Max (per ADR §10.2).
  Empirical measurement is deferred to Wave 3 when the pipeline is
  integrated into hf2q prefill and wall-time can be measured against
  the llama.cpp baseline.

---

## 10. Downstream integration notes (for Wave 2F / Wave 3)

- **Wave 2F (bf16 f16 kernels)**: independent of Wave 2E. When Wave 2F
  lands, the blk pre-pass kernel is already bf16-mask-typed; no
  changes needed for the mask side.
- **Wave 3 (hf2q wire-up)**: must:
  - Register all three kernels (`flash_attn_prefill`,
    `flash_attn_prefill_mask`, `flash_attn_prefill_blk`) with the
    kernel registry.
  - Call `alloc_blk_buffer(device, &BlkParams {…})` once per prefill
    per layer-type (global/sliding), reusing the allocation across
    layers of the same type.
  - Sequence per layer:
    1. `build_sdpa_mask_bf16(device, registry, enc, &mask_params)` (once per layer type).
    2. `dispatch_flash_attn_prefill_blk(enc, device, registry, &mask, &blk, &blk_params)` (once per layer type).
    3. `dispatch_flash_attn_prefill_bf16_d256_with_blk(enc, …, Some(&mask), Some(&blk), …, &params)` per layer dispatch.
  - NO additional fence between the pre-pass and the main kernel —
    the `CommandEncoder`'s implicit ordering guarantees the pre-pass
    writes are visible to the main kernel read (same command queue,
    same encoder, or chained commits). `test_blk_gemma4_sliding_prefill`
    exercises this exact sequence and passes.

---

## 11. References

- **Spec:** `/opt/hf2q/docs/ADR-011-phase2-port-tile-skip.md`
  (Agent #4's port spec; 26-item checklist).
- **llama.cpp pre-pass kernel:**
  `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5666-5719`.
- **llama.cpp main-kernel consumption:**
  `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5955-5963` (skip),
  `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6145` (mask-add gate),
  `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5983-6004` (obsoleted inline — Chesterton's fence evidence).
- **Phase 1 sentinel convention:**
  `/opt/hf2q/docs/ADR-011-phase2-port-sentinel.md`.
- **Wave 2A main kernel:**
  `/opt/mlx-native/src/shaders/flash_attn_prefill.metal`.
- **Wave 2C D=512 kernel:**
  `/opt/mlx-native/src/shaders/flash_attn_prefill_d512.metal`.
- **Wave 2D mask builder:**
  `/opt/mlx-native/src/ops/flash_attn_prefill_mask.rs`,
  `/opt/mlx-native/src/shaders/flash_attn_prefill_mask.metal`.
