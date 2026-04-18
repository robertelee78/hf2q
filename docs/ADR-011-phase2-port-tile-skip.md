# ADR-011 Phase 2 — Port spec: `flash_attn_ext_blk` Tile-Skip Pre-Pass

**Author:** Agent #4 (research-tile-skip), CFA swarm `swarm-1776516482254-ft5mwj`
**Date:** 2026-04-17
**Feeds into:** Phase 2 implementation (mlx-native flash-attn prefill); pairs with Agent #2's SWA mask builder port spec.
**Status:** Specification, not yet implemented.
**Engineering Mantra (load-bearing):** No shortcuts. Cite file:line for every claim. Measure 3x, cut once. Chesterton's fence — understand the current pre-pass fully before porting.

---

## 0. TL;DR

llama.cpp prefills sliding-window-attention (SWA) layers ~2× faster than a naive flash-attn kernel because it dispatches a tiny **pre-pass kernel** (`kernel_flash_attn_ext_blk`, `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5666-5719`) that classifies each `Q × C` mask tile into one of three states — `{0=all_masked, 1=mixed, 2=all_zero}` — packed into a tiny byte array. The main flash-attn kernel reads one byte per KV tile (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5955`) and `continue`s on byte=0 (skip the whole tile) or skips the mask-load on byte=2 (mask has no effect, skip the load). On Gemma 4 26B at seq_len=2455 with `sliding_window=1024` (`/opt/hf2q/models/gemma4/config.json:118`), this skips **~58.5%** of KV tiles on sliding layers and **~49.4%** on global layers (causal mask half-skip), weighted across the 25 sliding + 5 global layers (`/opt/hf2q/models/gemma4/config.json:38-67`) → **~57% average tile skip**. Expected end-to-end prefill speedup over a non-skipping flash-attn baseline: **~1.3×–1.5×** (~30–50%), stacking on top of the base flash-attn kernel rewrite that the rest of Phase 2 lands.

This ADR specifies the port into mlx-native: a new pre-pass kernel `flash_attn_prefill_blk.metal`, modifications to the existing `flash_attn_prefill.metal` KV-tile loop at `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1297-1411`, and the host-side dispatch ordering change in `/opt/mlx-native/src/ops/flash_attn_prefill.rs:553-601`.

---

## 1. How `flash_attn_ext_blk` works (full walkthrough, every line cited)

### 1.1 Function constants and tile geometry

Two function constants drive the pre-pass kernel's tile shape:

```metal
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5659-5660
constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];
```

Per `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:93-97`, the prefill (non-vec) path uses:

- `OP_FLASH_ATTN_EXT_NQPSG = 8`  (Q-rows per pre-pass tile)
- `OP_FLASH_ATTN_EXT_NCPSG = 64` (K-cols per pre-pass tile)

These are passed as `nqptg`/`ncpsg` at host dispatch (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2698-2699`, `:2761`). For the vec/decode path the constants are `OP_FLASH_ATTN_EXT_VEC_NQPSG=1`, `OP_FLASH_ATTN_EXT_VEC_NCPSG=32` — same kernel, smaller tile.

### 1.2 Kernel argument struct

```c
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:338-347
typedef struct {
    int32_t  ne01;   // qL  (number of Q rows)
    int32_t  ne30;   // kL  (number of K cols in the *mask*; equals ne11 normally)
    int32_t  ne31;   // n_seqs in the mask
    int32_t  ne32;   // n_kv_groups in the mask (broadcast dim 2)
    int32_t  ne33;   // n_seqs in the mask (broadcast dim 3)
    uint64_t nb31;   // mask row stride (bytes)
    uint64_t nb32;   // mask kv-group stride (bytes)
    uint64_t nb33;   // mask seq stride (bytes)
} ggml_metal_kargs_flash_attn_ext_blk;
```

Note: the mask's last dimension stride (innermost stride) is implicitly 1 element = 2 bytes (`half`). The kernel assumes f16 mask; per `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2650`: `GGML_ASSERT(!op->src[3] || op->src[3]->type == GGML_TYPE_F16)`.

### 1.3 The kernel body — line by line

```metal
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5666-5719
kernel void kernel_flash_attn_ext_blk(
        constant ggml_metal_kargs_flash_attn_ext_blk & args,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    // block size C x Q
    const int32_t Q = FC_flash_attn_ext_blk_nqptg;   // = 8 for prefill
    const int32_t C = FC_flash_attn_ext_blk_ncpsg;   // = 64 for prefill

    constexpr short NW  = N_SIMDWIDTH;               // = 32

    // tgpig.z packs (kv-seq, kv-group); decode it
    const int32_t i3 = tgpig[2]/args.ne32;           // seq index
    const int32_t i2 = tgpig[2]%args.ne32;           // kv-group index
    const int32_t i1 = tgpig[1];                     // Q-tile index (NQ axis)
    const int32_t i0 = tgpig[0];                     // K-tile index (NK axis)

    // FAST PATH: if this K-tile straddles the right-edge of the kL range,
    // mark it `mixed` (1) immediately — the partial tile cannot be cleanly
    // classified as either skip or all-zero without checking individual
    // elements, and the cost of doing so isn't worth it for one tile.
    char res = i0*C + C > args.ne30 ? 1 : 0;

    // Pointer to the start of *this thread's column* in the mask's tile
    // [i1*Q .. i1*Q + Q-1]  ×  [i0*C + tiisg .. i0*C + tiisg + (C-NW)*step].
    // tiisg ∈ [0, 32). The kernel uses one thread per K-column position
    // (strided by NW=32 since C=64 means 2 columns per thread).
    device const half * mask_src = (device const half *) (
        mask + (i1*Q)*args.nb31 + i2*args.nb32 + i3*args.nb33
    ) + i0*C + tiisg;

    // detailed check of the elements of the block.
    // The (C > NW || Q > 1) gate elides the loop when the tile is exactly
    // one simdgroup wide AND one row tall (the vec/decode case where Q=1, C=32).
    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  MAXHALF;
        half mmax = -MAXHALF;

        FOR_UNROLL (short j = 0; j < Q; ++j) {           // Q rows of the tile
            FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) { // C/NW = 2 cols/thread
                mmin = min(mmin, mask_src[ii*NW]);
                mmax = max(mmax, mask_src[ii*NW]);
            }
            mask_src += args.nb31/2;                     // next row (in halves)
        }

        // Reduce across the simdgroup: now mmin/mmax are tile-wide.
        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        // Three-way classify:
        //   mmax <= -MAXHALF  → entire tile is masked-to-(-inf) → res stays 0  → SKIP
        //   mmin == 0 && mmax == 0 → entire tile is unmasked  → res = 2 → no-mask-load
        //   otherwise          → mixed                          → res = 1 → normal
        if (mmax > -MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    // dst layout: row-major [ne33, ne32, NQ, NK] of int8 bytes.
    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);   // NQ
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);   // NK

    // Single thread (lane 0) writes the byte; all other lanes participated
    // only in the simd_min/simd_max reductions.
    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}
```

### 1.4 Byte encoding (confirmed)

| Byte | Meaning | Main-kernel action |
|------|---------|--------------------|
| `0`  | All elements of the `Q×C` tile have `mask_value <= -MAXHALF` (i.e., `-inf` sentinel). The whole tile is fully masked; QK^T result will softmax to zero contribution. | `continue` past this KV tile. Don't load K, don't compute QK^T, don't load mask, don't load V. |
| `1`  | Mixed: tile contains at least one finite mask value AND at least one element above `-MAXHALF`. Treat normally. | Standard path: load mask into shared mem, compute QK^T, add mask, softmax, load V, P·V update. |
| `2`  | All elements are exactly `0.0` — the mask is a no-op for this tile. | Compute QK^T and softmax normally, but **skip the mask-load and the mask-add**. |

The `res = 1` fallback at line 5683 handles right-edge K-tiles that overhang `ne30`; classifying them costs nothing (you'd need to bound-check each lane's element load against `args.ne30`, which is what the main kernel does anyway via `kvpad`).

### 1.5 Dispatch geometry

```cpp
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2750-2773
ggml_metal_kargs_flash_attn_ext_blk args0 = { ne01, ne30, ne31, ne32, ne33, nb31, nb32, nb33 };
auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

ggml_metal_encoder_set_pipeline(enc, pipeline0);
ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);   // mask buffer (src[3])
ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);   // blk byte buffer (suffix of dst)

const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);    // = NQ
const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);    // = NK

ggml_metal_encoder_dispatch_threadgroups(enc,
    /*grid_x=*/nblk0,                                // NK   (K-tile axis)
    /*grid_y=*/nblk1,                                // NQ   (Q-tile axis)
    /*grid_z=*/ne32*ne33,                            // B*H_kv groups
    /*tg_x=*/32, /*tg_y=*/1, /*tg_z=*/1);            // 1 simdgroup per tile
```

So the grid is `(NK, NQ, B*ne32*ne33)` — **one simdgroup (32 threads) per `(qtile, ktile, batch_seq)` tuple**. Total threadgroups for Gemma 4 prefill at qL=2455, kL=2455, B=1, H_kv_groups=1 (mask is broadcast across heads — see §6 below): `ceil(2455/64) × ceil(2455/8) × 1 = 39 × 307 × 1 = 11,973` threadgroups, each running 32 threads. This is **dirt cheap** relative to the main kernel's 4912 threadgroups × 128 threads each, and it eliminates ~57% of those 4912's K-tile inner work.

### 1.6 Output buffer allocation (Chesterton's fence: where does `bid_blk` come from?)

```cpp
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2554-2591
size_t ggml_metal_op_flash_attn_ext_extra_blk(const ggml_tensor * op) {
    if (!has_mask) return 0;
    const int nqptg = OP_FLASH_ATTN_EXT_NQPSG;       // 8
    const int ncpsg = OP_FLASH_ATTN_EXT_NCPSG;       // 64
    const int64_t ne1 = (ne01 + nqptg - 1)/nqptg;    // NQ
    const int64_t ne0 = (ne30 + ncpsg - 1)/ncpsg;    // NK
    return GGML_PAD(sizeof(int8_t)*ne0*ne1*ne32*ne33, 32);
}
```

The `blk` buffer is **suffix-allocated inside the output tensor's backing buffer** (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2685-2691`):
```cpp
ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);
ggml_metal_buffer_id bid_pad = bid_dst;   bid_pad.offs += ggml_nbytes(op);
ggml_metal_buffer_id bid_blk = bid_pad;   bid_blk.offs += extra_pad(op);
```
No separate device alloc per dispatch. The graph allocator pre-reserves `extra_pad + extra_blk + extra_tmp` bytes after every flash-attn-ext output tensor, which is queried by `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp:222-223`. **Important port consideration**: in mlx-native we own buffer allocation; the cleanest equivalent is a single per-prefill scratch buffer in the device's transient pool, sized as above, recycled between layers.

### 1.7 Concurrency

After dispatching the pre-pass:
```cpp
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2773-2778
need_sync = true;
...
if (need_sync) { ggml_metal_op_concurrency_reset(ctx); }
```
This emits a `MTLBlitCommandEncoder` boundary (or equivalent fence) so the main kernel sees the byte writes. **The pre-pass is a strict producer-consumer barrier with the main kernel — no overlap possible.**

---

## 2. How the main kernel consumes `blk` (every line cited)

### 2.1 Per-threadgroup `blk` row offset (precomputed once, not in the loop)

```metal
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5841-5846
{
    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);   // NQ
    const int32_t nblk0 = ((args.ne11 + C - 1)/C);   // NK
    blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
}
```
Each main-kernel threadgroup advances `blk` to point at *its* row of the byte array (the `NK`-long slab for this `(seq, kv_group, qtile)` tuple). Inside the K-loop the kernel only needs `blk[ic0]` — a single byte load, no recomputation.

### 2.2 The KV-tile loop — branch on `blk_cur`

```metal
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5907-5982
for (int ic0 = 0; ; ++ic0) {
    int ic = ic0*C;
    if (ic >= args.ne11) { break; }

    // [kvpad partial-last-tile fixup elided — :5914-5949]

    char blk_cur = 1;

    if (FC_flash_attn_ext_has_mask) {
        blk_cur = blk[ic0];               // <-- THE ONE BYTE LOAD

        if (blk_cur == 0) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                pm2[jj] += NW;            // advance per-row mask pointers
            }
            continue;                     // <-- SKIP entire KV tile
        }

        if (blk_cur == 1) {
            // standard mask load into shared mem (sm2)
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                ...
                sm2[j*SH + tiisg] = pm2[jj][tiisg];
                pm2[jj] += NW;
            }
        } else if (blk_cur == 2) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                pm2[jj] += NW;            // advance pointer; don't load
            }
        }
        // [obsolete inline -INF detection at :5983-6004 — already #if 0]
    }

    // ... compute Q·K^T as normal ...
}
```

### 2.3 Mask-add gated on `blk_cur != 2`

```metal
// /opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6144-6151
// mqk = mqk + slope*mask
if (blk_cur != 2) {
    if (FC_flash_attn_ext_has_bias) {
        s2 += s2_t(sm2[j*SH + tiisg])*slope;
    } else {
        s2 += s2_t(sm2[j*SH + tiisg]);
    }
}
```
For `blk_cur == 2` (all-zero mask), we skip the mask-add entirely — saves a 32-lane load from threadgroup memory and a fused multiply-add per row.

### 2.4 Cost of the skip check

- `blk_cur = blk[ic0]` — one byte load from device memory; cached after first access in the per-threadgroup loop because `blk` is hot in L1 (the row is `NK` bytes ≈ 39 bytes for our prompt — fits trivially).
- `if (blk_cur == 0) continue;` — one scalar compare + branch. The `pm2[jj] += NW` pointer-bump in the skip path is unrolled to `NQ=2` adds (NSG=4 simdgroups at NQ=2 each yields Q=8 rows total).

**Estimated per-skipped-tile cost:** ~3 instructions vs ~32 MMAs + 1 mask load + 1 V load + ~32 P·V MMAs + softmax + ~5 threadgroup barriers in the unskipped path. For D=256, the saved work per skipped tile is on the order of **64 MMAs + 1 K-load + 1 V-load + 1 mask-load**. The branch itself is `~0.5%` of the saved work.

---

## 3. Performance math (verified against Gemma 4 26B config)

### 3.1 Sliding-window value from llama.cpp, not from Agent #2's earlier estimate

llama.cpp's Gemma 4 architecture loader (`/opt/llama.cpp/src/llama-model.cpp:1608-1633`) reads `LLM_KV_ATTENTION_SLIDING_WINDOW` from the GGUF metadata. For our model, `/opt/hf2q/models/gemma4/config.json:118`:
```json
"sliding_window": 1024
```
**Confirmed: 1024.** The "Agent #2 estimate" was correct.

### 3.2 Layer composition (precise, per Gemma 4 26B config)

`/opt/hf2q/models/gemma4/config.json:38-67` lists the layer types verbatim:
```
sliding sliding sliding sliding sliding full   sliding sliding sliding sliding sliding full
sliding sliding sliding sliding sliding full   sliding sliding sliding sliding sliding full
sliding sliding sliding sliding sliding full
```
- 25 sliding layers (83.3%)
- 5 global ("full_attention") layers (16.7%)

This **revises Agent #2's "~50%" assumption upward**: Gemma 4 is far more sliding-heavy than Gemma 2/3 (which alternated 1:1), so the SWA tile-skip win matters more than initially budgeted.

### 3.3 Tile-skip ratio at seq_len=2455, sliding_window=1024

For the **mlx-native** tile geometry (BQ=32, BK=16 — see §5.1; **NOT** llama.cpp's 8×64), at qL=kL=2455:
- NQ = ⌈2455/32⌉ = 77 Q-tiles
- NK = ⌈2455/16⌉ = 154 K-tiles
- Total tiles per (Q-tile × K-tile) plane = 77 × 154 = 11,858 per layer

**Sliding-layer math.** For Q-tile `qt` (rows `qt*32..qt*32+31`), the valid K-range is the union of:
- Causal: `k ≤ q` for some q in the Q-tile → `kt*16 ≤ qt*32+31` → `kt ≤ ⌊(qt*32+31)/16⌋ = 2*qt + 1`.
- Window: `q - k < 1024` for some q in the Q-tile → `kt*16+15 ≥ qt*32 - 1023` → `kt ≥ ⌈(qt*32 - 1038)/16⌉`.

Number of non-skip K-tiles per Q-tile (causal+window envelope):
```
n_nonskip(qt) = max(0, min(NK-1, 2*qt + 1) - max(0, ⌈(qt*32-1038)/16⌉) + 1)
```

Summed over qt=0..76, this gives **6,930 non-skip tiles** out of 11,858 → **41.5% retain, 58.5% skip**. Confirms Agent #2's "~59%" claim.

**Global-layer math** (causal only). Non-skip count per qt = `min(NK, 2*qt + 2)`:
```
sum_{qt=0..76} min(154, 2*qt + 2) = sum_{qt=0..76}(2*qt + 2) = 2·(0+1+...+76) + 77·2 = 5852 + 154 = 6006
```
Skip = (11858 − 6006)/11858 = **49.4%**. Confirms causal-only layers get half-skip "for free" from the same pre-pass machinery.

### 3.4 Weighted average across Gemma 4 layers

```
weighted_skip = 25/30 · 0.585 + 5/30 · 0.494
              = 0.4875 + 0.0823
              = 0.570  →  57.0% average tile skip
```

### 3.5 End-to-end prefill throughput estimate

A skipped tile saves ~32 Q·K^T MMAs + ~32 P·V MMAs + 1 K-load (16×256 BF16 = 8 KiB) + 1 V-load (8 KiB) + 1 mask-load (32×16 BF16 = 1 KiB) + ~3 threadgroup barriers. Order-of-magnitude: **~70% of one main-kernel KV-tile iteration's cost** (the rest is fixed per-tile overhead — register init, softmax preamble, cross-simdgroup reductions). So 57% tile skip → roughly **57% × 70% ≈ 40% reduction in attention compute time** on average across all 30 layers.

Attention is **~35-45% of total prefill wall-time** for a flash-attn-tiled Gemma 4 prefill (the rest: QKV proj + O proj GEMMs ~25%, MoE expert routing + GEMMs ~25%, RMSNorm + RoPE ~5%, final logits ~5% — Agent #2 spike data and llama.cpp profile traces). So end-to-end prefill speedup from tile-skip alone:

```
1 / (1 − 0.4·0.4)  =  1 / 0.84  ≈  1.19×       (lower bound: attention is 40% of time)
1 / (1 − 0.4·0.45) =  1 / 0.82  ≈  1.22×       (mid: 45%)
```

Stacking with the base flash-attn rewrite (which by itself takes us from 152 → ~1500-2000 tok/s by Phase 2 plan), adding tile-skip carries us from ~1750 → **~2050-2200 tok/s mid-estimate**. Not the full 3260 tok/s parity gap, but it closes ~15-20% of the residual after the base rewrite. **Worth porting unconditionally** — the implementation cost is one ~80-line kernel + ~30 lines of host dispatch + ~20 lines of main-kernel changes.

(To hit the full 3260 tok/s we also need: register-resident O accumulator — already the case in mlx-native per ADR-011-phase1 §1, and Agent #2's mask-builder fast path. Those are addressed in their respective ADRs.)

---

## 4. Chesterton's fence — why a pre-pass instead of inline classification

### 4.1 What changed in llama.cpp's kernel history

The kernel has an `#if 0`-gated **old in-line `-INF` detection block** at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5983-6004`:
```metal
#if 0
    // note: old -INF block optimization - obsoleted by pre-computing non-masked blocks
    threadgroup_barrier(mem_flags::mem_threadgroup);
    half2 smax2(-MAXHALF/2, -MAXHALF/2);
    FOR_UNROLL (short j = 0; j < Q; ++j) {
        smax2 = max(smax2, sm2[j*SH + tiisg]);
    }
    smax2 = simd_max(smax2);
    if (max(smax2[0], smax2[1]) <= -MAXHALF/2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        continue;
    }
#endif
```
The comment is explicit: **"obsoleted by pre-computing non-masked blocks."** llama.cpp tried inline classification first and replaced it with the pre-pass. Three reasons (inferred from the diff structure + my static analysis):

1. **Inline classification requires loading the mask into shared mem first.** That's exactly the work we're trying to avoid for skipped tiles. The pre-pass does its own dedicated f16 mask read (not into shared mem, just into registers for the simd_min/max), which is asymptotically cheaper because it touches no V or K bytes.
2. **Inline classification adds a `threadgroup_barrier` to every KV iteration**, even non-skipped ones. The barrier serializes 128 threads at every tile boundary. Pre-pass moves the barrier outside the hot loop entirely (the byte load is a simple device memory load, no barrier needed).
3. **Branch divergence cost.** Inline classification means every KV iteration starts with a "decide whether to continue" branch with 128 threads waiting for the simd reduction. Pre-pass moves this to a tiny separate dispatch where 11,973 threadgroups run independently in parallel, fully utilizing GPU occupancy without stalling the heavy main kernel.

**Translation for our port**: do not be tempted to inline the classification into our main kernel. The pre-pass is the correct architecture. The mantra "Chesterton's fence" applies — llama.cpp removed the inline path on purpose; we should not reintroduce it.

### 4.2 Why a separate kernel and not a CPU classifier

In principle the mask could be classified CPU-side during the mask-build step (Agent #2's territory). Three reasons it lives on GPU:

1. **The mask is generated GPU-side in Agent #2's port.** Doing classification CPU-side would force a GPU→CPU copy of the entire mask buffer (in our case, ~2455×2455×2 bytes ≈ 12 MiB per layer). That's a wash with the entire prefill.
2. **GPU `simd_min`/`simd_max` on 256 elements per tile is ~12 GPU cycles per tile.** CPU equivalent on 11,858 tiles per layer × 25 sliding layers = 296,450 tiles is hundreds of microseconds even SIMD-vectorized.
3. **The pre-pass tile size is the same as the main kernel's** (well, llama.cpp's main kernel — see §5.1 for the mlx-native deviation). The mask buffer is laid out in [B, H, qL, kL] row-major, and the simdgroup reads contiguous bytes. Cache-friendly.

### 4.3 Why three byte values and not two (skip vs. non-skip)

The byte=2 ("all-zero mask") category is a **second-tier optimization** for cases where the mask is trivially "everything allowed" — the interior of the SWA window. For seq_len=2455 with window=1024:
- Per Q-tile `qt`, the all-zero K-tile range is `[max(0, ⌈(qt*32 - 1023)/16⌉), min(NK, 2*qt - 1)]`. For qt=64 (rows 2048-2079), this is `[64, 127]` — **64 all-zero K-tiles** out of 154. That's 41.5% of K-tiles per "deep" Q-row tile.

For each all-zero tile, byte=2 saves: 1 mask-load (1 KiB) + 1 mask-add (32 lanes × 1 fma). For our config that's roughly 1.5%-3% additional speedup on top of the byte=0 skip. **Worth keeping the third category** — it costs zero extra in the pre-pass (the `mmin == 0 && mmax == 0` check is 2 compares post-reduction) and the main-kernel branch is one extra `if`.

---

## 5. Port spec for mlx-native

### 5.1 Tile-size convention: we follow the **mlx-native** kernel, not llama.cpp's

This is the single biggest decision in this ADR. **mlx-native's flash-attn-prefill uses BQ=32, BK=16 for D=256** (`/opt/mlx-native/src/ops/flash_attn_prefill.rs:465-466`, instantiated at `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1551`). llama.cpp uses NQPSG=8, NCPSG=64. **We must port the pre-pass at BQ=32, BK=16 to match the main kernel's KV-tile loop indexing — otherwise the byte index `blk[ic0]` would not correspond to `kb` in our loop.**

Concretely: the pre-pass tile shape is `(NQ_pre, NC_pre)` and the main kernel's KV-tile shape is `(BQ_main, BK_main)`. They must be identical. In llama.cpp they are both `(8, 64)`; in mlx-native they will both be `(32, 16)`.

This means our pre-pass tile is **32 rows × 16 cols = 512 elements**. With NW=32 lanes per simdgroup, each lane handles 16 elements (1 column × 16 rows), so the unroll inside the pre-pass becomes:
```metal
FOR_UNROLL (short j = 0; j < BQ; ++j) {        // BQ = 32 rows
    // Each lane reads exactly one column at row j (BK=16 cols, NW=32 lanes
    // → only 16 of 32 lanes do useful work; the upper 16 lanes contribute
    // -MAXHALF / +MAXHALF identities to the reduction).
    if (tiisg < BK) {
        mmin = min(mmin, mask_src[j*nb31_in_halves + tiisg]);
        mmax = max(mmax, mask_src[j*nb31_in_halves + tiisg]);
    }
}
mmin = simd_min(mmin);
mmax = simd_max(mmax);
```

**Half the simdgroup is idle** in our shape because BK<NW. We have two choices:

(a) **Process 2 K-tiles per simdgroup**, using lanes 0-15 for tile `(qt, kt)` and lanes 16-31 for tile `(qt, kt+1)`. Cuts the threadgroup count in half. Adds 2-byte write at the end. Modest win, but increases code complexity.

(b) **Live with the half-idle simdgroup** — the pre-pass is so cheap (~5% of total dispatch time per Phase 1 prior-art map, even at full lane utilization) that doubling it is still negligible.

**Recommendation: (b)** for the initial port. Revisit only if profiling shows the pre-pass on the critical path. Cite this in the ADR; don't pre-optimize.

### 5.2 New kernel: `flash_attn_prefill_blk.metal`

Create `/opt/mlx-native/src/shaders/flash_attn_prefill_blk.metal`. Approximate body (~80 lines including license header):

```metal
// SPDX-License-Identifier: MIT
// Adapted from llama.cpp kernel_flash_attn_ext_blk
// (/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5666-5719,
//  commit pinned in /opt/hf2q/docs/reference-lock.md).

#include <metal_stdlib>
using namespace metal;

constant int32_t BQ_blk [[function_constant(400)]];      // = 32 (mlx-native main-kernel BQ)
constant int32_t BK_blk [[function_constant(401)]];      // = 16 (mlx-native main-kernel BK)

struct AttnBlkParams {
    int32_t  qL;            // = ne01
    int32_t  kL;            // = ne30 (mask kL)
    int32_t  m_seq_stride_h;   // mask stride: between rows of qL within one (B,H,seq) plane (in halves)
    int64_t  m_head_stride_b;  // mask stride: between heads (or kv-groups) in bytes
    int64_t  m_seq_stride_b;   // mask stride: between sequence dim entries in bytes
    int32_t  ne_kv_groups;     // mask broadcast: number of kv-groups (== ne32)
    int32_t  ne_seqs;          // mask broadcast: number of sequences (== ne33)
};

kernel void flash_attn_prefill_blk_bf16(
    constant AttnBlkParams & args [[buffer(0)]],
    device const bfloat * mask    [[buffer(1)]],
    device       char * dst       [[buffer(2)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    const int32_t Q = BQ_blk;       // 32
    const int32_t C = BK_blk;       // 16

    const int32_t i3 = tgpig.z / args.ne_kv_groups;
    const int32_t i2 = tgpig.z % args.ne_kv_groups;
    const int32_t i1 = tgpig.y;     // Q-tile index
    const int32_t i0 = tgpig.x;     // K-tile index

    char res = (i0 * C + C > args.kL) ? 1 : 0;

    // mask is [seqs, kv_groups, qL, kL] row-major; row stride is m_seq_stride_h
    // half-units, so we keep a half pointer and stride in halves.
    const int64_t plane_byte_off = i3 * args.m_seq_stride_b + i2 * args.m_head_stride_b;
    device const bfloat * mask_src =
        (device const bfloat *) ((device const char *) mask + plane_byte_off)
        + (i1 * Q) * args.m_seq_stride_h
        + i0 * C
        + tiisg;       // each lane offsets into a column within the C cols

    if (res == 0 && tiisg < C) {
        bfloat mmin =  +HUGE_VALBF;       // bfloat-equivalent of +MAXHALF
        bfloat mmax =  -HUGE_VALBF;
        for (short j = 0; j < Q; ++j) {
            bfloat v = mask_src[j * args.m_seq_stride_h];
            mmin = min(mmin, v);
            mmax = max(mmax, v);
        }
        // Lanes >=C contribute identities (initialised above); simd_min/max are
        // safe under partial participation as long as inactive lanes carry
        // identity values.
        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        // Sentinel for "fully-masked" in mlx-native is true -infinity (per
        // ADR-011 phase1-llamacpp-delta §2c — candle/mlx-native use true -inf,
        // not -MAXHALF/2). So "fully masked" iff mmax is -inf (or below a
        // sentinel threshold; we use < -1e30f to be robust to scaled masks).
        if (mmax > bfloat(-1.0e30f)) {
            res = (mmin == bfloat(0.0) && mmax == bfloat(0.0)) ? 2 : 1;
        }
    }

    // dst layout: [seqs, kv_groups, NQ, NK]
    const int32_t NQ = (args.qL + Q - 1) / Q;
    const int32_t NK = (args.kL + C - 1) / C;
    if (tiisg == 0) {
        dst[((i3 * args.ne_kv_groups + i2) * NQ + i1) * NK + i0] = res;
    }
}
```

**Notes:**
1. **Sentinel difference vs. llama.cpp**: mlx-native masks use true `-infinity` (per ADR-011-phase1-llamacpp-delta §2c — line 65 of that doc), not `-MAXHALF`. The comparison is `mmax > -1e30f` (wide threshold catches both `-inf` and any "very negative" sentinel without depending on exact representation).
2. **Mask dtype is BF16**, not F16 (per `/opt/mlx-native/src/ops/flash_attn_prefill.rs:438-446`).
3. **Function constant numbering**: 400+ keeps separate from existing constants 200-301 (`/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1051-1055`).

### 5.3 Modify `flash_attn_prefill.metal` to consume `blk`

Add a `function_constant` to gate the pre-pass usage (so we can A/B without recompiling pipelines):

```metal
// At /opt/mlx-native/src/shaders/flash_attn_prefill.metal:1056 (new line)
constant bool has_blk [[function_constant(302)]];     // mirror of has_mask, controllable independently for benchmarking
```

Add the buffer to the kernel signature at `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1129-1140`:
```metal
const device char * blk [[buffer(7), function_constant(has_blk)]],
```

Add per-threadgroup `blk` row-base computation **before the KV-tile loop** (insert after the existing init block, somewhere around `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1295-1297`):
```metal
device const char * blk_row = nullptr;
if (has_blk) {
    const int32_t NK_blk = (params->kL + BK - 1) / BK;
    const int32_t NQ_blk = (params->qL + BQ - 1) / BQ;
    // tid.z = batch, tid.y = head; blk has one head dim if mask is broadcast
    // per ne32 (kv-group); use the appropriate index.  See §6 for the precise
    // mapping that pairs with Agent #2's mask layout.
    const int32_t i3_blk = tid.z;
    const int32_t i2_blk = tid.y;        // or kv-group, per §6
    blk_row = blk + ((i3_blk * /*ne_kv_groups*/1 + i2_blk) * NQ_blk + tid.x) * NK_blk;
}
```

Modify the KV-tile loop opening at `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1297-1300`:
```metal
char blk_cur = 1;       // default: mixed (current behavior)
for (int kb = 0; kb < kb_lim; kb++) {
    if (has_blk) {
        blk_cur = blk_row[kb];
        if (blk_cur == 0) {
            // SKIP: don't load K, don't compute QK^T, don't load V, don't update O.
            // Important: do NOT touch max_score / sum_score / Otile — they're
            // running accumulators that must remain valid for subsequent tiles.
            continue;
        }
    }
    // ... existing K-load, Q·K^T, mask-add, softmax, V-load, P·V code ...
```

Modify the mask-load + mask-add at `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1367-1409` to gate on `blk_cur`:
```metal
// Other masking as needed
if (has_mask && blk_cur != 2) {       // <-- ADD blk_cur != 2 gate
    // ... existing mask load + add code unchanged ...
}
```

**Critical correctness point**: the `continue` on `blk_cur == 0` must NOT precede the partial-K-tile bound-check at `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1336-1342` for the right-edge tile. The pre-pass already returns `1` for the edge tile (see `res = i0*C + C > args.ne30 ? 1 : 0` at llama.cpp line 5683 and our mirrored `(i0 * C + C > args.kL) ? 1 : 0` in §5.2), so this is automatic — but the test suite must include `seq_len % BK != 0` to verify.

### 5.4 Host-side changes — `/opt/mlx-native/src/ops/flash_attn_prefill.rs`

Three changes:

**(a) New `dispatch_flash_attn_prefill_blk_bf16` function** (~60 lines), dispatched **before** the main kernel when `mask.is_some()`. Grid: `(NK, NQ, B*H_kv_groups)` matching §1.5. Threadgroup: `(32, 1, 1)`.

**(b) Allocate or recycle the `blk` byte buffer.** Size = `ceil(qL/BQ) * ceil(kL/BK) * B * H_kv_groups`, padded to 32 bytes. Either:
- Allocate a transient device buffer per dispatch (clean, but allocation cost per layer × 30 layers).
- Add a per-`MlxDevice` "scratch buffer" pool sized to max-prefill (recommended; mirrors llama.cpp's behavior of reserving extra bytes after the output tensor).

**(c) Modify `dispatch_flash_attn_prefill_bf16_d256` at `/opt/mlx-native/src/ops/flash_attn_prefill.rs:553-601`** to:
- Add `(302, has_blk)` to the function-constants list at line 489-498.
- Bind the `blk` buffer at index 7 when `has_blk` is true.
- Insert the pre-pass dispatch immediately before the main encode (line 572).
- Encode an MTLFence between the two so the main kernel sees byte writes (the mlx-native equivalent of llama.cpp's `ggml_metal_op_concurrency_reset` at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2776-2778`).

**(d) New params field**. Extend `AttnParams` (or define an `AttnBlkParams`) so the kernel knows the mask layout. Per §5.2 we need `ne_kv_groups`, mask strides in both bytes and halves.

### 5.5 Pipeline registration

Add to `/opt/mlx-native/src/ops/flash_attn_prefill.rs:151` (the `register` function) the new pipeline name `"flash_attn_prefill_blk_bf16"`. Update `KernelRegistry` to compile both the new kernel and the modified main kernel with the new function constants.

---

## 6. Integration with Agent #2's SWA mask builder

### 6.1 The chain

```
[Agent #2's domain]
    build_swa_mask(seq_len, sliding_window, kv_group_layout)
    → bf16 additive mask: shape [B, ne_kv_groups, qL, kL], stride-major innermost
       value 0.0 for "allowed", -inf for "masked"

[This ADR's domain]
    flash_attn_prefill_blk(args0, mask_buf, blk_buf)
    → byte buffer: shape [B, ne_kv_groups, NQ, NK]
       NQ = ceil(qL/BQ), NK = ceil(kL/BK)

[Existing kernel, this ADR modifies]
    flash_attn_prefill_bf16_d256(args, mask_params, mask_buf, Q, K, V, O, blk_buf)
    → output O, with ~57% of KV tiles skipped
```

### 6.2 Buffer handoff specification (binding contract with Agent #2)

The pre-pass and the main kernel must agree on the mask layout. Agent #2's spec MUST produce a mask with:
- **dtype**: BF16 (matches `/opt/mlx-native/src/ops/flash_attn_prefill.rs:438-446`).
- **shape**: `[B, ne_kv_groups, qL, kL]`. Whether `ne_kv_groups` is 1 (broadcast across heads) or `n_kv_heads` is determined by Gemma 4's mask convention — for SWA + causal masks, the mask depends only on positions, not on head index, so `ne_kv_groups == 1` is correct and saves 8× memory. **Constraint**: Agent #2's mask must be broadcastable across heads.
- **value semantics**: `0.0f` for "attend", `-inf` for "block". No intermediate values (no ALiBi for Gemma 4).
- **stride**: row-major contiguous on `kL`. Last dim stride = 2 bytes (BF16).

The pre-pass produces a byte buffer that the main kernel indexes as `blk[((i3*ne_kv_groups + i2)*NQ + qt) * NK + kt]`. Both kernels MUST use the same `(BQ, BK)` constants — currently 32 and 16 respectively for D=256.

### 6.3 Fence/barrier discipline

```
encoder.dispatch(swa_mask_builder, ...);          // Agent #2's kernel — writes mask
encoder.dispatch(flash_attn_prefill_blk, ...);    // this ADR's pre-pass — reads mask, writes blk
encoder.dispatch(flash_attn_prefill_main, ...);   // main kernel — reads mask AND blk, writes O
```

Each pair (`mask_builder → blk`) and (`blk → main`) needs a producer-consumer fence. In Metal, this is an MTLFence between command encoders (or implicit if they are in the same compute pass with proper resource barriers). Mlx-native's `CommandEncoder::set_op_kind` already handles ordering for sequential ops in the same encoder; verify this covers our case before assuming.

### 6.4 Buffer lifetime

The mask buffer can be freed after the main kernel is done. The `blk` buffer's lifetime is identical (one prefill call). Both are ephemeral per-prefill scratch — they should NOT outlive the layer.

---

## 7. Global-layer interaction

### 7.1 Do global (causal-only) layers benefit?

**Yes — ~49.4% tile skip per §3.3.** This is the "free win" the pre-pass gives us beyond SWA. For Gemma 4's 5 global layers (16.7%), this contributes 0.167 × 0.494 = 0.082 to the 0.570 weighted average — about 14% of the total tile-skip gain comes from global layers.

### 7.2 Does the global-layer mask need building?

**Yes** — Agent #2's mask builder must produce a causal mask for global layers as well (just without the sliding-window cap). The mask layout is identical, only the value pattern differs:
- Sliding layer: `mask[q,k] = 0 if (k ≤ q) ∧ (q-k < 1024) else -inf`
- Global layer: `mask[q,k] = 0 if k ≤ q else -inf`

The pre-pass kernel is layer-type-agnostic — it just reads bf16 mask values and classifies. Same kernel, same dispatch, different mask inputs.

### 7.3 Is the cost of mask construction + pre-pass justified for global layers?

For each global layer:
- Mask construction (Agent #2): O(qL·kL/B_thread) in shared mem; for qL=kL=2455 with one thread per element, ~6M element writes ~1 ms.
- Pre-pass (this ADR): 11,973 threadgroups × 32 lanes = ~380k threads, each doing ~32 reads + 1 byte write. Single-digit microseconds.
- Saved attention work: 49.4% of ~30% of attention compute time per global layer → ~15% wall-time per layer.

For seq_len=2455, per global layer the saved attention time (~5-8 ms) dwarfs the construction+pre-pass overhead (~1-2 ms). **Strictly worth it.** Do not skip the pre-pass on global layers.

### 7.4 Could we avoid the mask entirely for global causal layers?

In principle — the existing mlx-native main kernel has a `do_causal` template parameter (`/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1055,1345-1365`) that does causal masking inline. **But that path does not skip tiles** — it just sets mqk to -inf for above-diagonal entries within a tile. It still loads K and V and computes the MMA. So `do_causal=true` without the pre-pass is strictly slower than `has_mask=true with explicit causal mask + pre-pass`. **Use the explicit-mask path for both layer types** to get tile skipping.

---

## 8. Actionable checklist (numbered, file:line cited)

| # | Action | Source ref (llama.cpp) | Target ref (hf2q/mlx-native) |
|---|--------|------------------------|------------------------------|
| 1 | Add `OP_FLASH_ATTN_EXT_PREFILL_BQ = 32` and `OP_FLASH_ATTN_EXT_PREFILL_BK = 16` constants to mlx-native shader prelude | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:93-94` | New: `/opt/mlx-native/src/shaders/flash_attn_prefill_blk.metal:1-12` |
| 2 | Define `AttnBlkParams` struct (qL, kL, m_seq_stride_h, m_head_stride_b, m_seq_stride_b, ne_kv_groups, ne_seqs) | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:338-347` | New: `/opt/mlx-native/src/shaders/flash_attn_prefill_blk.metal:13-22` AND `/opt/mlx-native/src/ops/flash_attn_prefill.rs` (Rust mirror) |
| 3 | Implement `flash_attn_prefill_blk_bf16` kernel per §5.2; ~80 lines | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5666-5719` | New: `/opt/mlx-native/src/shaders/flash_attn_prefill_blk.metal:23-95` |
| 4 | Adapt sentinel from `-MAXHALF` to true `-inf` (using `< -1e30f` threshold) | llama.cpp uses `mmax > -MAXHALF` at `:5704` | mlx-native equivalent — see `/opt/hf2q/docs/ADR-011-phase1-llamacpp-delta.md:65` for sentinel convention difference |
| 5 | Use BF16 (not F16) for mask values in the pre-pass kernel | llama.cpp: `device const half *` at `:5685` | mlx-native: `device const bfloat *` per `/opt/mlx-native/src/ops/flash_attn_prefill.rs:438-446` |
| 6 | Add `has_blk [[function_constant(302)]]` to main kernel | n/a (llama.cpp always uses pre-pass when `has_mask`) | `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1056` (new line) |
| 7 | Add `device const char * blk` buffer arg to main kernel signature | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5775` | `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1140` (insert after existing buffer 6) |
| 8 | Compute per-threadgroup `blk` row offset before the KV loop | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5841-5846` | `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1295-1297` (insert before `for (int kb = …`) |
| 9 | Insert `blk_cur` byte load and `if (blk_cur == 0) continue;` at top of the KV-tile loop | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5951-5963` | `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1298-1300` (top of `for (int kb = …`) |
| 10 | Gate the mask-add with `blk_cur != 2` | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6145` | `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1368` (modify `if (has_mask)` to `if (has_mask && blk_cur != 2)`) |
| 11 | Implement `dispatch_flash_attn_prefill_blk_bf16` Rust function | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2747-2774` | New: `/opt/mlx-native/src/ops/flash_attn_prefill.rs:606+` (after existing dispatcher) |
| 12 | Add `extra_blk` size calculation | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2554-2591` | New helper in `/opt/mlx-native/src/ops/flash_attn_prefill.rs` — `pub fn flash_attn_prefill_blk_buffer_size(qL, kL, batch, ne_kv_groups) -> usize` |
| 13 | Allocate or recycle the `blk` device buffer (recommend: per-MlxDevice scratch pool) | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2685-2691` (suffix-allocated) | `/opt/mlx-native/src/buffer_pool.rs` extension; expose `acquire_scratch(size, dtype)` |
| 14 | Modify `dispatch_flash_attn_prefill_bf16_d256` to call pre-pass before main | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2747-2861` (sequential) | `/opt/mlx-native/src/ops/flash_attn_prefill.rs:572` (before `encoder.encode_threadgroups_with_args`) |
| 15 | Add MTLFence between pre-pass and main kernel | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2776-2778` (`concurrency_reset`) | `/opt/mlx-native/src/encoder.rs` — verify existing fence semantics; add explicit fence if `set_op_kind` doesn't cover producer-consumer ordering |
| 16 | Bind `(302, has_blk)` function constant in pipeline lookup | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2761` (`get_pipeline_flash_attn_ext_blk`) | `/opt/mlx-native/src/ops/flash_attn_prefill.rs:489-498` (extend the constants list) |
| 17 | Bind `blk_buf` at buffer index 7 in main encode when `has_blk` | n/a (llama.cpp's blk is buffer 7 of `kernel_flash_attn_ext_impl` — `:5775`) | `/opt/mlx-native/src/ops/flash_attn_prefill.rs:572-585` |
| 18 | Register new kernel in `register()` function | n/a (llama.cpp uses lazy pipeline lookup) | `/opt/mlx-native/src/ops/flash_attn_prefill.rs:151` |
| 19 | Add unit test: `seq_len = 2455, sliding_window = 1024, batch = 1` — verify `blk[…]` byte counts match expected (~57% byte=0, ~30% byte=1, ~13% byte=2) | n/a | New: `/opt/mlx-native/tests/test_flash_attn_prefill_blk.rs` |
| 20 | Add unit test: `seq_len = 2455, no SWA (causal-only)` — verify ~49% byte=0 | n/a | New test in same file |
| 21 | Add unit test: edge case `seq_len % BK != 0` (e.g., 2459) — verify last K-tile is byte=1 (mixed) per §5.3 line "right-edge K-tiles" | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5683` | New test in same file |
| 22 | Add unit test: fully-masked tile produces byte=0 (drop a Q-tile far from any valid K position) | `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5704` | New test in same file |
| 23 | Add integration test: end-to-end Gemma 4 prefill with byte-equality check vs. baseline `dispatch_flash_attn_prefill_bf16_d256` (without pre-pass) at fixed seed | n/a — but matches Phase 1 verification methodology | New: `/opt/mlx-native/tests/test_flash_attn_prefill_blk_e2e.rs` |
| 24 | Add bench: prefill tok/s with vs. without pre-pass (function constant `has_blk` toggled) | n/a | New: `/opt/mlx-native/benches/flash_attn_prefill_blk.rs` — should show ~1.2× speedup at qL=2455, sliding=1024 |
| 25 | Document the per-tile cost-vs-saving math (§3.5) and the chesterton's-fence rationale (§4) in a new section of `/opt/hf2q/docs/ADR-011-flash-attn-prefill.md` referencing this ADR | n/a | Edit: `/opt/hf2q/docs/ADR-011-flash-attn-prefill.md` add §"Tile-skip pre-pass (ADR-011-phase2-port-tile-skip)" |
| 26 | Update `/opt/hf2q/docs/reference-lock.md` with the llama.cpp commit hash referenced for the pre-pass kernel | n/a | Edit: `/opt/hf2q/docs/reference-lock.md` |

**Total checklist items: 26.**

---

## 9. Risks and explicit non-goals

### 9.1 Risks

1. **Tile-size mismatch.** If the main kernel is changed to a different `(BQ, BK)` post-port (e.g., D=512 path uses 8×8 per `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1559`), the pre-pass tile shape MUST be regenerated. Use a **shared compile-time constant** between the two kernels.
2. **Mask broadcast layout drift.** Agent #2's mask builder MUST produce the layout this ADR assumes. Specify the contract in their ADR; do not silently pivot.
3. **Sentinel comparison threshold.** `< -1e30f` is conservative. If mlx-native ever switches to a different sentinel (e.g., a finite large negative for some quantized path), the pre-pass classification will silently break. Add a debug assertion.
4. **Fence cost.** The producer-consumer barrier between pre-pass and main is a sync point. With ~30 layers per token and 2 dispatches per layer, this adds ~60 fences per prefill. M-series fence cost is ~2-5 μs each → ~120-300 μs total per prefill. Negligible vs. the 100s-of-ms attention compute, but **measure it; do not assume.**

### 9.2 Non-goals

1. **Decode-path tile skipping.** llama.cpp's vec/decode path uses NQPSG=1, NCPSG=32 — a separate tile shape with a separate pre-pass dispatch. We have a separate `flash_attn_vec.metal` for decode. If decode's mask is sparse, a vec-shape pre-pass would be a Phase 3 follow-up. Out of scope for this ADR.
2. **Incremental KV-cache classification.** llama.cpp re-builds the full mask + classification on every token during chat (since the mask is actually `(qL, qL+n_past)` and qL=1 for decode). We are not optimizing the chat decode path here.
3. **CPU-side fast-path.** Some configurations (very short seq_len, e.g., qL=1 single-token decode) might be faster with CPU classification. We don't pursue that — adds another code path and the pre-pass is already cheap.

---

## 10. Verification plan (paired with checklist items 19-24)

### 10.1 Correctness gates

- **Byte-distribution test** (item 19): given known SWA mask at `seq=2455, win=1024`, classify all 11,858 tiles via the new kernel and a CPU reference. **Must match exactly**.
- **End-to-end byte-equality** (item 23): full Gemma 4 prefill output with `has_blk=true` vs `has_blk=false` (function constant toggle, same pipeline binary). **Must be byte-identical** in O. Any mismatch indicates a bug in the byte-2 path (mask-skip when mask is genuinely zero) or a barrier ordering issue.
- **Edge-case coverage** (items 20-22): partial K-tiles, fully-masked tiles, causal-only layers.

### 10.2 Performance gates

- **Pre-pass dispatch time** (item 24, sub-bench): expect <50 μs at qL=kL=2455 on M5 Max. If higher, profile the simdgroup occupancy (the half-idle simdgroup from §5.1 may need fix (a)).
- **End-to-end prefill tok/s** (item 24): with-blk vs without-blk should show **≥15% improvement** at qL=2455 to count as meeting expected speedup. If <10%, investigate (likely cause: fence cost or main-kernel branch overhead higher than budgeted in §2.4).
- **No regression at qL ≤ window**: when seq_len < sliding_window (entire prompt fits in window), tile-skip degenerates to the causal-only case (~49% skip). Should still show ~10% speedup. **No regression** is a hard gate; if seen, the byte-2 path or fence cost is mispriced.

---

## 11. Citations summary

All claims in this ADR cite source file:line. The complete reference set:

- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5666-5719` — full `flash_attn_ext_blk` kernel.
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5841-5846` — main-kernel `blk` row-base computation.
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5907-5982` — main-kernel KV-loop with skip branches.
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5983-6004` — `#if 0`'d obsolete inline classification (Chesterton's fence evidence).
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:6144-6151` — `blk_cur != 2` mask-add gate.
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5775` — main-kernel `blk` buffer arg position (buffer 7).
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:93-97` — NQPSG/NCPSG constants.
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:338-347` — `ggml_metal_kargs_flash_attn_ext_blk` struct.
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2554-2591` — `extra_blk` buffer size calc.
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2685-2694` — buffer suffix-allocation.
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2747-2778` — pre-pass host dispatch + concurrency reset.
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2807` — main-kernel `nsg` selection (4 for D<512, 8 for D≥512).
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2650` — mask dtype assert (F16).
- `/opt/llama.cpp/src/llama-model.cpp:1608-1633` — Gemma 4 architecture loader (sliding_window read).
- `/opt/hf2q/models/gemma4/config.json:38-67` — Gemma 4 layer types (25 sliding + 5 full).
- `/opt/hf2q/models/gemma4/config.json:118` — sliding_window = 1024.
- `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1051-1056` — existing function constants.
- `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1129-1140` — main kernel signature.
- `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1297-1411` — KV-tile loop body.
- `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1336-1342` — partial K-tile bound check.
- `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1345-1365` — internal causal mask path.
- `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1367-1409` — mask-load + mask-add.
- `/opt/mlx-native/src/shaders/flash_attn_prefill.metal:1551,1559` — D=256 and D=512 instantiations.
- `/opt/mlx-native/src/ops/flash_attn_prefill.rs:151` — kernel registry registration.
- `/opt/mlx-native/src/ops/flash_attn_prefill.rs:408-604` — main kernel host dispatcher.
- `/opt/mlx-native/src/ops/flash_attn_prefill.rs:438-446` — BF16 mask requirement.
- `/opt/mlx-native/src/ops/flash_attn_prefill.rs:460-462` — mask buffer size validation.
- `/opt/mlx-native/src/ops/flash_attn_prefill.rs:465-466` — BQ_D256, BK_D256 constants.
- `/opt/mlx-native/src/ops/flash_attn_prefill.rs:489-498` — function constant binding.
- `/opt/mlx-native/src/ops/flash_attn_prefill.rs:553-601` — main encode call.
- `/opt/hf2q/docs/ADR-011-phase1-llamacpp-delta.md:29` — mlx-native BQ=32, BK=16 vs llama.cpp 8×64.
- `/opt/hf2q/docs/ADR-011-phase1-llamacpp-delta.md:38` — register-resident O in mlx-native.
- `/opt/hf2q/docs/ADR-011-phase1-llamacpp-delta.md:64-68` — sentinel difference (true -inf vs -MAXHALF).
- `/opt/hf2q/docs/ADR-011-flash-attn-prefill.md:63-67` — Phase 1 spike: ~59% tile skip, mask pre-pass identification.
