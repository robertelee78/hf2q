# ADR-011 Phase 2 — Port spec: llama.cpp sliding-window-attention (SWA) mask builder

**Status**: PROPOSED (research output — Agent #2, swarm `swarm-1776516482254-ft5mwj`).
**Companion**: Agent #1 (sentinel verification). See cross-check in §1.3.
**Source of truth**: `/opt/llama.cpp` HEAD as of 2026-04-17.
**Target**: replace the `window_size: u32` kernel parameter currently passed to
`mlx_native::ops::sdpa_sliding` with an additive bf16/f16 mask buffer built on
the host (or in a small Metal fill kernel), exactly as llama.cpp does it.

This document is **a port spec**, not an implementation. It records every
file:line we are porting *from* and every file:line we will modify *to*. No
choices are made that aren't already made by llama.cpp.

---

## 0. Scope and discipline

We are not designing a new mask format. llama.cpp has shipped this design for
years; it survives every model variant (Gemma 2/3/4, Mistral SWA, GPT-OSS
chunked, OLMo2 standard, BERT cross). We port it verbatim. Any deviation must
be marked as a deviation with a rationale, never as "improvement" or
"simplification" (per project mantra: "Don't say simpler"). The two deviations
we *do* take are forced by environment, not preference, and are listed in §6.

---

## 1. Exact algorithm — the llama.cpp mask builder

### 1.1 The canonical no-cache builder

`/opt/llama.cpp/src/llama-graph.cpp:380-444` — `llm_graph_input_attn_no_cache::set_input`.
This is the simplest version (no KV cache, single-pass prefill — exactly our
batched-prefill case). The algorithm:

```cpp
// llama-graph.cpp:384-413
const auto fill_mask = [&](float * data, int n_swa, llama_swa_type swa_type) {
    for (int i1 = 0; i1 < n_tokens; ++i1) {
        const llama_seq_id s1 = ubatch->seq_id[i1][0];
        const llama_pos    p1 = ubatch->pos[i1];
        const uint64_t idst = i1 * n_kv;
        for (int i0 = 0; i0 < n_tokens; ++i0) {
            const llama_seq_id s0 = ubatch->seq_id[i0][0];
            const llama_pos    p0 = ubatch->pos[i0];
            if (s0 != s1)                                   continue;  // different sequence
            if (cparams.causal_attn && p0 > p1)             continue;  // future
            if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) continue;  // outside window
            data[idst + i0] = hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f;
        }
    }
};
// llama-graph.cpp:419-423 (base mask: just causal, no SWA)
float * data = (float *) self_kq_mask->data;
std::fill(data, data + ggml_nelements(self_kq_mask), -INFINITY);
fill_mask(data, 0, LLAMA_SWA_TYPE_NONE);
// llama-graph.cpp:434-438 (SWA mask: causal + window)
float * data_swa = (float *) self_kq_mask_swa->data;
std::fill(data_swa, data_swa + ggml_nelements(self_kq_mask_swa), -INFINITY);
fill_mask(data_swa, hparams.n_swa, hparams.swa_type);
```

In English: **pre-fill with `-INFINITY`, then walk every (q_pos, k_pos) pair
and overwrite attended positions with `0.0f`.** Three independent gates:
same-sequence, causal (k_pos ≤ q_pos), and SWA (window check). All three must
pass for a position to be attended.

### 1.2 The KV-cache builder (more general, same logic)

`/opt/llama.cpp/src/llama-kv-cache.cpp:1417-1606` — templated
`set_input_kq_mask_impl<causal, swa, is_2d, alibi>`. Three optimisations on top
of the no-cache version (we do **not** need them for batched prefill but we
record them for completeness):
* **Per-sequence mask reuse**: `seq_srct[seq_id]` (line 1489). When two query
  tokens belong to the same sequence, copy the previous row instead of
  re-walking — only the cells "near" the boundary change. Disabled when ALiBi
  is on. Reference: PR ggml-org/llama.cpp#18842.
* **Block-skip cell tracking**: `idxs.push_back(j)` (line 1534). Records cells
  for which `p0 + n_swa + 32 >= seq_pos_min[seq_id]`; subsequent rows iterate
  only those cells.
* **Templated specialisation**: 4 booleans `<causal, swa, is_2d, alibi>`
  produce 16 specialisations dispatched from `set_input_kq_mask`
  (lines 1608-1640). The hot path emits no branches.

The masked write at the end of the inner loop uses the same sentinel as the
no-cache path:

```cpp
// llama-kv-cache.cpp:1567-1572
if (alibi) { data[idst + j] = -std::abs(p0 - p1); }
else       { data[idst + j] = 0.0f; }
continue;
skip:
data[idst + j] = -INFINITY;
```

Note this differs structurally from the no-cache builder: instead of pre-filling
with `-INFINITY` and writing `0.0f` for hits, it writes one of two values per
cell directly (`-INFINITY` on `goto skip`, `0.0f` otherwise). The result is
*identical*; the tradeoff is one branch vs one fill. For our port we use the
no-cache style (pre-fill + selective hit) because (a) it's simpler to express
in a Metal fill kernel and (b) it's the direct ancestor of the algorithm we
need.

### 1.3 Sentinel value — independent verification (cross-check with Agent #1)

**Verdict from this agent: `-INFINITY` (i.e. `-std::numeric_limits<float>::infinity()`).**

Evidence — every place llama.cpp writes the masked sentinel:

| File:line | Code | Context |
|-----------|------|---------|
| `llama-graph.cpp:421` | `std::fill(data, data + ggml_nelements(self_kq_mask), -INFINITY);` | base no-cache mask fill |
| `llama-graph.cpp:436` | `std::fill(data, data + ggml_nelements(self_kq_mask_swa), -INFINITY);` | SWA no-cache mask fill |
| `llama-graph.cpp:557` | `float f = -INFINITY;` | cross-attention mask fill |
| `llama-kv-cache.cpp:1572` | `data[idst + j] = -INFINITY;` | KV-cache `skip:` label |
| `llama-graph.cpp:370` | `if (val == -INFINITY) { LLAMA_LOG_DEBUG(" ∞"); }` | mask debug printer recognises -INFINITY as the only "masked" symbol |

**It is not `-FLT_MAX/2`**. The literal `-FLT_MAX` does occur in llama.cpp
(sampler.cpp:1556, vocab.cpp:898/934, chameleon.cpp:166-169) but never as an
attention-mask sentinel — those are unrelated reductions. The `-MAXHALF`
sentinel that appears in the Metal FA kernel (`ggml-metal.metal:5649,
5933, 5970, 6768`) is used **only** by the F16 *padding* pre-pass
(`kernel_flash_attn_ext_pad`) when the user-supplied mask doesn't span the
tile-aligned shape; it is not the original masked-position sentinel, it is the
saturate-on-cast result of `-INFINITY` (F32) → `-MAXHALF` (F16) for any
runtime that doesn't preserve infinities through cast. On Metal, F16 *does*
have `-inf`, but llama.cpp uses `-MAXHALF` defensively because some downstream
arithmetic (`sm * slope` for ALiBi) would otherwise produce NaN when slope=0.

**Cast to F16 path** (`llama-graph.cpp:1995, 2001`):
```cpp
inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
```
`ggml_cast(F32 → F16)` of `-inf` produces `0xFC00` (`-inf` half) on every CPU
and Metal backend in ggml. We will do the same.

If Agent #1 reports a different value, the discrepancy is most likely from
Agent #1 reading the *padding* sentinel (`-MAXHALF`) in the Metal kernel
instead of the *mask-builder* sentinel in `llama-graph.cpp` /
`llama-kv-cache.cpp`. Resolution rule: **the builder owns the sentinel**; the
kernel only reads it.

### 1.4 The window predicate `is_masked_swa`

`/opt/llama.cpp/src/llama-hparams.h:316-350` — `static bool is_masked_swa(...)`.
Inlined for performance ("note: inlined on purpose"). Four cases:

```cpp
// LLAMA_SWA_TYPE_STANDARD (Gemma 4): line 323-328
if (p1 - p0 >= (int32_t) n_swa) { return true; }   // masked
// → attended iff (p1 - p0) < n_swa, i.e. window upper bound is *exclusive*

// LLAMA_SWA_TYPE_CHUNKED (GPT-OSS): line 329-336
const llama_pos pos_chunk_start = (p1 / n_swa) * n_swa;
if (p0 < pos_chunk_start) { return true; }

// LLAMA_SWA_TYPE_SYMMETRIC (Gemma-Embedding): line 337-346
const int32_t half_n_swa = (int32_t) n_swa / 2;
const int32_t pos_diff = p1 - p0;
if (pos_diff < -half_n_swa || pos_diff > half_n_swa) { return true; }

// LLAMA_SWA_TYPE_NONE: line 320-322 — never masks
```

For Gemma 4 we use `LLAMA_SWA_TYPE_STANDARD`. Therefore for our port:
* **Window inclusive of self**: `p1 - p0 < n_swa` includes `p0 == p1` (distance 0).
* **Window exclusive at the far edge**: distance `n_swa - 1` is attended; distance
  `n_swa` is masked.
* `n_swa == 0` → every off-self position is masked (window = {self only}).
* `n_swa >= seq_len` → window check never trips; mask reduces to plain causal.

### 1.5 In-batch single-sequence simplification

For our batched prefill we have `batch=1, single sequence, all positions
0..seq_len-1, kv_seq_len == seq_len`. Most checks collapse:
* `s0 != s1` always false → drop.
* `cparams.causal_attn` always true → drop the conditional, keep the comparison.
* `ubatch->seq_id` indirection unused → drop.
* `ubatch->pos[i]` is just `i` → replace with the loop index.

The minimal Gemma 4 SWA-layer mask, in pseudo-Rust:
```rust
// for each (q_row, k_col) in [0..seq_len) × [0..seq_len):
//   attended iff (k_col <= q_row) AND (q_row - k_col < n_swa)
//   else masked (-inf)
let attended = k_col <= q_row && (q_row - k_col) < (n_swa as i32);
mask[q_row * seq_len + k_col] = if attended { 0.0 } else { f32::NEG_INFINITY };
```

For global layers: `attended iff k_col <= q_row` (no window check).

---

## 2. Mask tensor layout

### 2.1 The ggml allocation

`llama-graph.cpp:1992` (no-cache path):
```cpp
inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens, 1, 1);
```

ggml shape convention is `(ne[0], ne[1], ne[2], ne[3])` with `ne[0]` = innermost
contiguous dim. Therefore the logical shape is **`[batch=1, heads=1, n_q=n_tokens, n_k=n_tokens]`**
with `n_k` as the inner stride. The `heads=1` is a **broadcast across heads** —
every Q head reads the same mask cell. This is essential for memory: at
seq_len=2048 a per-head mask would be `2048² × 24 heads × 2B = 192 MiB`; the
broadcast version is `8 MiB`. **We must mirror the broadcast** (see §5.4).

`llama-graph.cpp:31` (KV-cache path, more general):
```cpp
ggml_tensor * res = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
```
Same broadcast across heads, multi-stream support via `ne[3]`. For us
`n_stream=1`, so the two paths reduce to the same shape.

### 2.2 Dtype

* **Allocated as `GGML_TYPE_F32`** (line 1992).
* **Cast to `GGML_TYPE_F16` at graph-build time** when `cparams.flash_attn`
  is on (`llama-graph.cpp:1995, 2001, 2338, 2398, 2408, 2576`):
  ```cpp
  inp->self_kq_mask_cnv = cparams.flash_attn
      ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16)
      : inp->self_kq_mask;
  ```
  The `_cnv` (converted) tensor is what's actually fed into `ggml_flash_attn_ext`
  (`llama-graph.cpp:1893`).
* **Metal FA kernel reads as `half2`** (`ggml-metal.metal:5830, 5833, 5838,
  5644`): `device const half2 * pm2[NQ];`.

For us — flash-attention is always on for prefill — the mask we pass to our
Metal kernel must be **F16 or BF16**. Our kernel already supports both
(instantiations at `flash_attn_prefill.metal:1551-1562`: `bf16_d256_boolmask`,
`f16_d256_boolmask`, `bf16_d512_boolmask`, `f16_d512_boolmask`). Choice for
the port: **bf16**, matching the I/O dtype of Q/K/V everywhere else in our
pipeline (no extra cast).

### 2.3 Strides our kernel expects

`/opt/mlx-native/src/ops/flash_attn_prefill.rs:563-570`:
```rust
let m_ql_stride    = kl as i64;            // stride to next q row
let m_head_stride  = (ql * kl) as i64;     // stride to next head
let m_batch_stride = (h * ql * kl) as i64; // stride to next batch
let mask_params = AttnMaskParamsGpu {
    m_strides: [m_batch_stride, m_head_stride, m_ql_stride],
};
```
This is `[B, H, qL, kL]` row-major with kL inner. **For llama.cpp parity we
must set `m_head_stride = 0`** to broadcast a single per-(qL, kL) plane across
all heads. Set `m_batch_stride = 0` likewise (we always have batch=1 anyway,
but defensively zero it for broadcast semantics if we ever go batch>1 with a
shared mask).

Concretely for our port:
```rust
// Broadcast mask across heads (mirrors llama.cpp ne[2]=1).
let m_strides = [0_i64, 0_i64, kl as i64];
```

### 2.4 Summary — the wire format

| Property | llama.cpp | Our port |
|---|---|---|
| Dtype on disk | F32 (then cast F16 if flash_attn) | bf16 (single-shot) |
| Logical shape | `[B, H=1 (broadcast), qL, kL]` | `[B, H=1 (broadcast), qL, kL]` |
| Memory layout | row-major, kL innermost | row-major, kL innermost |
| Sentinel for masked | `-INFINITY` (F32) → `-inf` (F16) | `-inf` (bf16) |
| Sentinel for attended | `0.0` | `0.0` |
| Combination with scores | additive: `score += mask` | additive: `Stile += log2(e) * mask` (see §6.1) |

---

## 3. Gemma 4 sliding-window configuration

### 3.1 Where it's plumbed

`/opt/llama.cpp/src/llama-model.cpp:1608-1633` — `case LLM_ARCH_GEMMA4`:
```cpp
hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;                                              // line 1610
ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);  // line 1611
ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);                              // line 1621
```
Crucially, Gemma 4 is **the only Gemma variant that reads `swa_layers` as an
explicit per-layer array** rather than computing it from a periodic pattern.
Compare:
* Gemma 2 (`llama-model.cpp:1526-1551`): `set_swa_pattern(swa_period)` —
  computed from a single integer.
* Gemma 3 (`llama-model.cpp:1555-1586`): same — computed.
* Gemma 4 (`llama-model.cpp:1608-1633`): `get_key_or_arr(... swa_layers, n_layer)` —
  **explicit per-layer bool array from the GGUF**.

The semantic is documented at `llama-hparams.h:134-138`:
> `if swa_layers[il] == 1, then layer il is SWA; if swa_layers[il] == 0, then layer il is dense (i.e. non-SWA)`.

Predicate: `bool llama_hparams::is_swa(uint32_t il) { return swa_layers[il]; }`
(`llama-hparams.cpp:208-211`).

### 3.2 Ground-truth values for our model

Read with `gguf` python bindings on
`/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf`
on 2026-04-17:

```
gemma4.block_count                       = 30
gemma4.attention.sliding_window          = 1024
gemma4.attention.sliding_window_pattern  = [T,T,T,T,T,F, T,T,T,T,T,F, T,T,T,T,T,F, T,T,T,T,T,F, T,T,T,T,T,F]
gemma4.attention.key_length_swa          = 256
gemma4.attention.value_length_swa        = 256
gemma4.rope.freq_base_swa                = 10000.0
gemma4.rope.dimension_count_swa          = 256
```

So:
* **n_swa = 1024** (window includes self plus 1023 prior keys → 1024-position
  span on `LLAMA_SWA_TYPE_STANDARD`).
* **Layer pattern**: 5 sliding + 1 global, repeating five times to fill 30
  layers. Sliding layers (`is_swa=true`): indices 0,1,2,3,4, 6,7,8,9,10,
  12,13,14,15,16, 18,19,20,21,22, 24,25,26,27,28 — 25 layers. Global layers
  (`is_swa=false`): indices **5, 11, 17, 23, 29** — 5 layers.
* Sliding layers have head_dim=256; global layers in this MoE have head_dim=512
  (separately configured via `attention.key_length` for the full layers and
  `key_length_swa=256` for the sliding ones).

This is fixed across the entire model — `n_swa` is a single scalar shared by
every sliding layer; only the *which-layers-are-sliding* bit varies per layer.

### 3.3 hf2q's existing config

`/opt/hf2q/src/serve/config.rs:47`: `pub sliding_window: usize` — single scalar,
matches `n_swa`. `/opt/hf2q/src/serve/forward_mlx.rs:1456, 2413`: per-layer
`is_sliding` flag from `hparams.is_swa(il)` already plumbed into the SDPA
dispatcher. **No new config plumbing needed** — the per-layer flag is already
available.

---

## 4. Interaction with global (full-attention) layers

### 4.1 llama.cpp's choice — one mask buffer per *type*, not per layer

`llama-graph.cpp:496-522` — `llm_graph_input_attn_kv_iswa::set_input` (the
ISWA = "interleaved SWA" variant used by Gemma 2/3/4):
```cpp
mctx->get_base()->set_input_kq_mask(self_kq_mask,     ubatch, cparams.causal_attn);
mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
```
Two mask buffers are built per ubatch:
* `self_kq_mask` — used by every **global** layer. Contains causal-only mask
  (`is_masked_swa` returns false for `LLAMA_SWA_TYPE_NONE`).
* `self_kq_mask_swa` — used by every **sliding** layer. Contains causal +
  window mask.

Per-layer routing happens at `llama-graph.cpp:2030-2032`:
```cpp
const bool is_swa = hparams.is_swa(il);
const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();
```

So **global layers also use a mask buffer** — they don't bypass it. The mask
is just the trivial causal triangle (no window). They could in principle skip
the mask buffer (since causal-only is the kernel default in many backends),
but llama.cpp doesn't bother — uniform code path is more important than
saving 8 MiB at 2k context.

### 4.2 Our port's choice — port-equivalent

We follow llama.cpp exactly: build **two** mask buffers per prefill (one
causal-only for global layers, one causal-plus-window for sliding layers) and
let `is_sliding` pick. The two buffers can share the host build code; they
differ only in passing `n_swa=0` (`LLAMA_SWA_TYPE_NONE`) vs `n_swa=1024`
(`LLAMA_SWA_TYPE_STANDARD`).

**Open optimisation (deferred, *not* a deviation)**: for global layers we
could pass `mask=None, do_causal=true` to our kernel (saves 8 MiB and the
mask load), since our kernel's `do_causal` constant produces the exact same
mask. But that's a downstream perf tweak, not a correctness port. Decide in
Phase 3 after the Phase 2 port is correct and on a measured baseline.

---

## 5. Port spec — Rust side

### 5.1 Where the function lives

**Decision: `/opt/mlx-native/src/ops/flash_attn_prefill_mask.rs` (NEW)**.

Rationale (per `project_mlx_native_is_the_strategic_destination.md`):
mlx-native is the strategic destination for GPU compute we own; the SWA-mask
builder is a Metal kernel + a thin Rust dispatcher — both belong in
mlx-native, not in hf2q. hf2q stays a thin orchestration layer that *calls*
mlx-native ops.

The hf2q-side change is a single line at the call-site (§5.7): replace the
`window_size: u32` field with a mask-buffer reference.

### 5.2 Where the Metal kernel lives

**Decision: `/opt/mlx-native/src/shaders/build_swa_mask.metal` (NEW)** — a
single small Metal kernel that fills a bf16 buffer in parallel. Reasoning in
§5.4: GPU fill is essentially free (memory-bandwidth-bound on a tiny tensor)
and avoids a large host→device transfer per prefill.

llama.cpp does this on the **CPU** (the mask buffer is allocated in
`ggml-cpu` host memory; see `GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer))`
at `llama-graph.cpp:417`). They then rely on ggml's automatic
host→device upload before kernel dispatch (`ggml_backend_tensor_set` is
implicit). For a unified-memory M-series GPU **host-fill is also a valid
choice** — there's no real "transfer". We pick GPU-fill anyway because:
1. Our Rust dispatcher already lives on the GPU side; staying on-device
   avoids a second code path.
2. We can fuse the two masks (global + sliding) into a single dispatch if
   future profiling shows it matters.
3. Cache-friendly write pattern: each thread writes one f16/bf16 cell.

This is a **legal port deviation**: same mask, different fill location. The
mask values are byte-identical to llama.cpp's. Documented as deviation #1 in
§6.1.

### 5.3 Function signatures

Rust dispatcher (`mlx-native/src/ops/flash_attn_prefill_mask.rs`):
```rust
/// SWA mode — direct port of `llama_swa_type` (llama-hparams.h, lines 19-24
/// of the enum definition); we only implement STANDARD because Gemma 4 only
/// uses STANDARD. CHUNKED and SYMMETRIC are added when a model needs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwaType {
    /// No window — causal-only mask (used by Gemma 4 global layers).
    None,
    /// `attended iff (q_pos - k_pos) < n_swa` (Gemma 4 sliding layers).
    Standard,
}

#[derive(Debug, Clone, Copy)]
pub struct BuildSwaMaskParams {
    /// Query sequence length.
    pub seq_len_q: u32,
    /// Key sequence length. For batched prefill == seq_len_q.
    pub seq_len_k: u32,
    /// Sliding window size. Ignored when swa_type == None. Must be > 0 when
    /// swa_type != None (n_swa == 0 is undefined behaviour upstream — see
    /// llama-hparams.h:316-350).
    pub n_swa: u32,
    pub swa_type: SwaType,
}

/// Allocate and fill an additive attention mask matching llama.cpp's
/// `llm_graph_input_attn_no_cache::set_input` semantics
/// (llama-graph.cpp:380-444). Single-sequence, batch=1, causal_attn=true,
/// no ALiBi.
///
/// Output: bf16 buffer, shape `[seq_len_q, seq_len_k]` row-major (no batch /
/// head dims; broadcast at SDPA call by setting `m_strides = [0, 0, seq_len_k]`).
/// Sentinel: `bf16::NEG_INFINITY` for masked, `0.0` for attended. See ADR-011
/// phase 2 §1.3 for sentinel cross-check.
///
/// Performance: O(seq_len_q × seq_len_k) writes via a `build_swa_mask` Metal
/// kernel. At seq_len=2048: 4 MiB written, ~30 µs on M5 Max (bandwidth-bound).
pub fn build_swa_mask(
    encoder: &mut MlxEncoder,
    reg: &mut BufferRegistry,
    dev: &MlxDevice,
    out: &MlxBuffer,
    params: &BuildSwaMaskParams,
) -> Result<()>;
```

Buffer allocation lives in the caller (hf2q forward path) so the buffer can
be reused across layers and across the global/sliding pair. **The dispatcher
does not allocate.** Matches the rest of mlx-native's API style.

### 5.4 GPU fill kernel (sketch)

```metal
// build_swa_mask.metal — one thread per (q_row, k_col) cell.
// shape [qL, kL], bf16, row-major (kL innermost).
struct BuildSwaMaskParams {
    uint32_t seq_len_q;
    uint32_t seq_len_k;
    uint32_t n_swa;        // 0 means "no window" (caller sets when swa_type=None)
    uint32_t swa_standard; // 1 = STANDARD, 0 = NONE
};

[[kernel]] void build_swa_mask(
    device bfloat16_t* mask              [[buffer(0)]],
    constant BuildSwaMaskParams& params  [[buffer(1)]],
    uint2 tid                            [[thread_position_in_grid]]) {
    if (tid.y >= params.seq_len_q || tid.x >= params.seq_len_k) return;
    const int q = int(tid.y);
    const int k = int(tid.x);
    bool attended = (k <= q);  // causal
    if (params.swa_standard != 0) {
        attended = attended && ((q - k) < int(params.n_swa));  // window
    }
    mask[q * params.seq_len_k + k] = attended
        ? bfloat16_t(0.0)
        : -bfloat16_t(metal::numeric_limits<bfloat16_t>::infinity());
}
```

Grid: `(ceil_div(seq_len_k, 32), ceil_div(seq_len_q, 32), 1)`. Threadgroup:
`(32, 32, 1)`.

For seq_len = 2048, single-sequence prefill: total work = 2048² = 4 Mi cells
× 2 B = 8 MiB write, ~30 µs at 280 GB/s sustained. Negligible vs the
~200 µs attention dispatch it precedes.

### 5.5 Caching across prefills

**Yes, the mask is cacheable across prefills with the same `(seq_len, n_swa,
swa_type)` triple.** The mask depends only on those three scalars (under our
single-sequence batch=1 simplification §1.5). For the typical workflow —
process a long prompt, then generate token-by-token — the prefill mask is
built once.

Caching strategy:
1. **Per-prefill memoise**: build once per `(seq_len, n_swa)` tuple within
   a single prefill call. Both global and sliding layers reuse the buffer.
   This is what llama.cpp implicitly does via the `can_reuse_kq_mask` graph
   node (`llama-graph.cpp:38-55`).
2. **Cross-prefill cache**: keep a small LRU keyed on `(seq_len, n_swa,
   swa_type)`. Prefill seq_len rarely repeats exactly so this is low value;
   defer to Phase 3.

For the port, implement #1 only. The hf2q forward path already has
`pf_mask_global` and `pf_mask_sliding` natural slots — both `MlxBuffer`s built
once before the layer loop and reused N times.

### 5.6 Why the mask approach? (Chesterton's fence)

llama.cpp could have parameterised the kernel with `window_size` (which is
what hf2q does today via `sdpa_sliding`). They didn't. Reasons, in order of
importance:

1. **Tile-skip pre-pass requires it.** `ggml-metal.metal:5662-5719` defines
   `kernel_flash_attn_ext_blk` — a "pre-classification" kernel that scans the
   mask in `(BQ × BK)`-sized blocks and marks each block as fully-masked
   (skip in the main kernel) or partially/fully-attended. The main FA kernel
   then iterates only the non-skipped blocks. This is impossible without a
   materialised mask buffer because the pre-pass must read the actual values.
   See comment at `ggml-metal.metal:5984`: *"note: old -INF block
   optimization - obsoleted by pre-computing non-masked blocks"* — they
   replaced an in-kernel infinity-detector with the explicit pre-pass.
2. **Uniform code path across mask types.** The same kernel handles causal,
   SWA-standard, SWA-chunked, SWA-symmetric, ALiBi, cross-attention, and
   no-mask. A `window_size` parameter would handle one of those; an additive
   mask handles all of them.
3. **Sequence-aware masking on packed batches.** The full builder
   (`llama-kv-cache.cpp:1417-1606`) handles multi-sequence packed ubatches
   where different rows attend to different KV cells. No `window_size`
   parameter could express that — only an explicit mask can.

For our Gemma 4 use case, only #1 directly applies *today* (we're
single-sequence). But adopting the mask-buffer convention now means our
kernel is already shape-compatible with future tile-skip work — the eventual
`flash_attn_ext_blk` analogue (Phase 3+ of ADR-011) is a drop-in addition,
not a re-architecture.

The mask approach is **not just convenient, it is required for tile-skip** —
which is the reason ADR-011 exists. So keeping `window_size` as a kernel
parameter would block the Gate target. The mask port is on the critical path.

### 5.7 hf2q call-site change

`/opt/hf2q/src/serve/forward_prefill_batched.rs:405-444`:

**Before** (current):
```rust
if is_sliding {
    let sliding_params = mlx_native::ops::sdpa_sliding::SdpaSlidingParams {
        n_heads: nh as u32, n_kv_heads: nkv as u32, head_dim: hd as u32,
        seq_len: seq_len as u32, kv_seq_len: seq_len as u32,
        window_size: self.sliding_window as u32,            // ← WINDOW PARAM
        scale: 1.0, kv_capacity: seq_len as u32,
    };
    mlx_native::ops::sdpa_sliding::sdpa_sliding(...)?;
} else {
    let sdpa_params = mlx_native::ops::sdpa::SdpaParams { ... };
    s.sdpa(...)?;
}
```

**After** (target, mirrors llama.cpp `llm_graph_context::build_attn` dispatch
at `llama-graph.cpp:2030-2032`):
```rust
// Built once, before the layer loop:
//   pf_mask_sliding = build_swa_mask(seq_len, seq_len, n_swa=1024, Standard)
//   pf_mask_global  = build_swa_mask(seq_len, seq_len, n_swa=0,    None)
//   (or: skip pf_mask_global and pass mask=None + do_causal=true to FA)
let mask = if is_sliding { &pf_mask_sliding } else { &pf_mask_global };

let fa_params = mlx_native::ops::flash_attn_prefill::FlashAttnPrefillParams {
    n_heads: nh as u32, n_kv_heads: nkv as u32, head_dim: hd as u32,
    seq_len_q: seq_len as u32, seq_len_k: seq_len as u32,
    batch: 1, scale: 1.0, do_causal: false,  // causal is in the mask
};
mlx_native::ops::flash_attn_prefill::dispatch_bf16_d256(
    s.encoder_mut(), reg, dev,
    &pf_q_perm, &pf_k_perm, &pf_v_perm, &pf_sdpa_out_perm,
    Some(mask), &fa_params,
)?;
```

Note `do_causal = false` because **causal is in the mask buffer** (matches
llama.cpp). Don't double-mask. Alternative — set `do_causal=true` and skip
the causal-write in the mask-fill kernel — produces the same numerics but is
a deviation from llama.cpp's "all gates in the mask" convention. **Choose
all-in-mask to match llama.cpp.**

Same for d=512 (global layers): use `dispatch_bf16_d512`.

The `sdpa_sliding` op is fully replaced by `flash_attn_prefill + mask`. The
former can stay in mlx-native for the per-token decode path that doesn't tile
(decode SDPA reads one Q row × full K) — mark with a comment that prefill no
longer uses it.

---

## 6. Documented deviations from llama.cpp

Two deviations, both cosmetic w.r.t. numerical output:

### 6.1 Mask fill on GPU instead of CPU
* **llama.cpp**: builds mask on CPU (`ggml_backend_buffer_is_host` assert),
  uploads to device implicitly.
* **Us**: build mask on GPU via a dedicated kernel.
* **Why**: unified memory M-series, no real "upload"; keeps the dispatcher
  on-device; allows future fusion with FA dispatch.
* **Verification**: dump our mask buffer and llama.cpp's mask buffer for the
  same `(seq_len, n_swa)` and assert byte equality after both have been cast
  to bf16. Test in Phase 3.

### 6.2 Single dtype (bf16) instead of F32→F16 cast
* **llama.cpp**: allocates F32, casts to F16 at graph build.
* **Us**: write bf16 directly.
* **Why**: our pipeline is bf16 end-to-end (`project_bf16_pipeline`); F32
  intermediate is wasted memory bandwidth. The kernel casts mask → score
  dtype anyway (`flash_attn_prefill.metal:1404`: `selem_t(mfrag[jj])` where
  `selem_t` is the score accumulator).
* **Verification**: bf16 has the same dynamic range as F32 (8 exponent bits)
  so `-inf` and `0.0` are exact. The 7 mantissa bits don't matter for these
  two values. No precision loss vs F32→F16 cast.

Both deviations preserve the byte-identical mask values for masked and
attended cells. Confirmed by inspection of `is_masked_swa` and the cast path.

---

## 7. Port checklist — actionable, ordered

Each item cites the source file:line in llama.cpp and the target file:line in
our code. Ordered by dependency (no item depends on a later item).

1. **Add `SwaType` enum to mlx-native**.
   * From: `/opt/llama.cpp/src/llama-hparams.h:20-24` (enum `llama_swa_type`).
   * To: `/opt/mlx-native/src/ops/flash_attn_prefill_mask.rs` (NEW, top of file).
   * Implement only `None` and `Standard`. Add `Chunked` / `Symmetric`
     variants when a model needs them — leave `unimplemented!()` markers.

2. **Add `build_swa_mask` Metal kernel**.
   * From: algorithm at `/opt/llama.cpp/src/llama-graph.cpp:384-413`,
     simplified for batch=1, single-sequence, no-ALiBi (per §1.5).
   * To: `/opt/mlx-native/src/shaders/build_swa_mask.metal` (NEW, sketch in §5.4).
   * Pipeline registration: add to mlx-native's pipeline-cache enumerator
     (find by analogy with existing kernels — TODO in implementation).

3. **Add `build_swa_mask` Rust dispatcher**.
   * To: `/opt/mlx-native/src/ops/flash_attn_prefill_mask.rs` (NEW).
   * Signature in §5.3. Allocates no buffers; takes `out: &MlxBuffer`.

4. **Update `flash_attn_prefill` dispatcher to support `do_causal=false`
   with a non-None mask**.
   * Verify: `/opt/mlx-native/src/ops/flash_attn_prefill.rs:553-601` already
     allows `mask=Some + do_causal=anything` — confirm no assertion forbids
     this combination. If a test requires `do_causal` with mask, *add* a
     test for `do_causal=false + additive mask`, do not relax existing test.

5. **Set mask broadcast strides**.
   * Edit: `/opt/mlx-native/src/ops/flash_attn_prefill.rs:563-570`.
   * Change: `m_strides = [0, 0, kl as i64]` (instead of computed
     batch/head strides) when caller passes a broadcast mask.
   * Backward compatibility: add an `Option<MaskBroadcast>` field to the
     dispatcher params, default = broadcast across heads, explicit override
     allowed for future per-head masks.

6. **Pre-build masks in batched-prefill forward path**.
   * Edit: `/opt/hf2q/src/serve/forward_prefill_batched.rs` — add allocation
     and `build_swa_mask` calls before the layer loop (around line ~340,
     before the `for layer_idx in ...` loop).
   * Build two masks: `pf_mask_global` (n_swa=0, SwaType::None) and
     `pf_mask_sliding` (n_swa=1024, SwaType::Standard). Buffer dtype: bf16,
     shape `[seq_len, seq_len]`.

7. **Replace `sdpa_sliding` call with `flash_attn_prefill + mask`**.
   * Edit: `/opt/hf2q/src/serve/forward_prefill_batched.rs:409-444`.
   * Replace per §5.7. Keep `sdpa_sliding` op present in mlx-native for the
     decode path (don't delete prematurely).

8. **Per-token (decode) path** — *out of scope for Phase 2, recorded for tracking*.
   The decode path at `/opt/hf2q/src/serve/forward_mlx.rs:1456, 2413` also
   passes `sliding_window: u32`. Leave as-is for Phase 2. Phase 3 will
   convert decode to `(K-cache + mask)` form once we have the per-decode
   mask builder (which is `[1, kv_seq_len]` — degenerate case of §5.4).

9. **Numerical parity test**.
   * Add: `/opt/hf2q/tests/test_swa_mask_parity.rs` (NEW) or extend an
     existing parity test. Compare hf2q output with llama.cpp output for the
     same Gemma 4 prefill at seq_len ∈ {64, 256, 1023, 1024, 1025, 4096} —
     specifically test `seq_len < n_swa`, `seq_len == n_swa`, and `seq_len > n_swa`
     boundaries (where the SWA mask actually differs from causal).
   * Acceptance: max-abs token-logits delta ≤ 1e-3 vs llama.cpp oracle.

10. **End gate measurement**.
    * Run `scripts/sourdough_gate.sh` to confirm Gate D self-baseline holds
      and Gate G dispatch counters reflect the new kernel call (one fewer
      `sdpa_sliding`, one new `build_swa_mask` and one `flash_attn_prefill`
      with `has_mask=true`). No perf regression > 3 % at Gate A floor
      (130 tok/s). Tile-skip Phase 3 work picks up from here.

**Total: 10 items.** Items 1-3 are mlx-native crate work (NEW files). Items
4-5 modify the existing flash_attn_prefill dispatcher. Items 6-7 are the
hf2q-side wiring change. Items 8-10 are the verification fence around the
port.

---

## 8. References

* **llama.cpp source files** (all `/opt/llama.cpp/`):
  * `src/llama-graph.cpp:380-444` — no-cache mask builder (the canonical algorithm).
  * `src/llama-graph.cpp:1992-2004` — mask tensor allocation + F16 cast.
  * `src/llama-graph.cpp:2030-2032` — per-layer is_swa routing.
  * `src/llama-graph.cpp:2388-2417` — `build_attn_inp_kv_iswa` (ISWA setup).
  * `src/llama-kv-cache.cpp:1417-1606` — templated KV-cache builder (multi-seq).
  * `src/llama-hparams.h:316-350` — `is_masked_swa` predicate.
  * `src/llama-hparams.h:134-138` — `swa_layers` per-layer bool array.
  * `src/llama-hparams.cpp:8-18` — `set_swa_pattern` (NOT used by Gemma 4).
  * `src/llama-hparams.cpp:208-211` — `is_swa(il)`.
  * `src/llama-model.cpp:1608-1633` — Gemma 4 hparams parse (the per-layer bool array).
  * `src/models/gemma4-iswa.cpp:1-110` — Gemma 4 graph build (uses `build_attn_inp_kv_iswa`).
  * `ggml/src/ggml-metal/ggml-metal.metal:5589-5719` — pad + tile-skip pre-passes.
  * `ggml/src/ggml-metal/ggml-metal.metal:5767-6149` — main FA kernel mask consumption.

* **Our code** (all `/opt/`):
  * `mlx-native/src/shaders/flash_attn_prefill.metal:1054-1409` — existing
    prefill kernel with `has_mask` + `do_causal` function constants.
  * `mlx-native/src/ops/flash_attn_prefill.rs:553-601` — existing dispatcher
    that already supports a mask buffer.
  * `hf2q/src/serve/forward_prefill_batched.rs:405-444` — call site to update.
  * `hf2q/src/serve/config.rs:47` — `sliding_window: usize` (already set to 1024 from GGUF).
  * `hf2q/src/serve/forward_mlx.rs:1456, 2413` — decode-path call sites
    (Phase 2 leaves untouched; Phase 3 converts).

* **Project memory** (`/Users/robert/.claude/projects/-opt-hf2q/memory/`):
  * `project_mlx_native_is_the_strategic_destination.md` — places mask
    builder in mlx-native, not hf2q.
  * `project_barrier_stall_is_the_gap.md` — context for why we're doing
    Phase 2 at all (the eventual tile-skip needs the mask).
  * `feedback_walk_means_port_llama_cpp_to_rust.md` — overriding directive
    for this work.

* **Ground-truth GGUF read** (2026-04-17):
  * Path: `/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf`
  * `gemma4.attention.sliding_window = 1024`
  * `gemma4.attention.sliding_window_pattern = [T,T,T,T,T,F]×5`
  * `gemma4.block_count = 30`

---

## 9. Open items / non-goals

* **Multi-stream batch** (`n_stream > 1`): not needed for our single-sequence
  inference. The full builder at `llama-kv-cache.cpp:1417-1606` supports it;
  port only when serving multi-request.
* **ALiBi**: Gemma 4 doesn't use ALiBi. Don't port.
* **Cross-attention mask** (`llama-graph.cpp:543-570`): no Gemma 4 layer is
  cross-attention. Don't port.
* **`LLAMA_SWA_TYPE_CHUNKED` and `LLAMA_SWA_TYPE_SYMMETRIC`**: not used by
  Gemma 4. Add when GPT-OSS or Gemma-Embedding lands in hf2q.
* **Tile-skip pre-pass** (`flash_attn_ext_blk`): Phase 3+ work. The mask
  format established here makes that addition possible; it does not depend
  on this port producing one.
