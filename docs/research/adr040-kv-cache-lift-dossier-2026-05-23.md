# ADR-040 Phase A2 + A3 — KV-cache n_seqs lift grounding dossier

- **Date**: 2026-05-23
- **Author**: research agent (grounding pass for iter-2 implementation)
- **Inputs**:
  - `/opt/hf2q/docs/adr/ADR-040-continuous-batching-reopen.md`
  - `/opt/hf2q/src/serve/multi_seq_kv.rs` (Phase A iter-1 trait, 746 LOC)
  - `/opt/hf2q/src/inference/models/qwen35/kv_cache.rs` (5918 LOC)
  - `/opt/hf2q/src/inference/models/gemma4/kv_cache.rs` (466 LOC)
  - probes into `gpu_full_attn.rs`, `gpu_delta_net.rs`, `forward_gpu.rs`, `forward_prefill.rs`, `forward_prefill_batched.rs`, `engine.rs`, `kv_persist/*`, `spec_decode/{eagle3,dflash}/kv_cache.rs`, `/opt/mlx-native/src/ops/*`, `/opt/mlx-native/src/shaders/*.metal`
- **Mantra**: "Never guess; multi-week structural work is in scope; understand current fully (Chesterton's fence) before changing it."

---

## §1 Executive summary

1. **Qwen35 `HybridKvCache` carries `n_seqs` in the allocation shape AND in the persistor envelope, but the production forward path is hard-coded to `n_seqs=1` in at least 7 distinct sites** (linear-attn delta_net dispatch, capture-buffer rollback, conv-state extraction, doc comments declaring `n_seqs=1`, `current_len[0]` reads). ADR-040 §1.3's claim — "the structural shape supports >1 with no buffer-layout change" — is **partially true** (the F32 full-attn slot's buffer rank IS 4-D `[n_seqs, n_kv, max_seq, head_dim]`), **partially false** (the linear-attn DeltaNet capture buffer's layout was written assuming `n_seqs=1` and rejects `n_seqs>1` at runtime via an explicit guard at `kv_cache.rs:1567`).
2. **Gemma 4 has FOUR KV variants** (`MlxKvCache`, `HbKvBuffers`, `DenseKvBuffers`, `HybridKvBuffers`); the current production path is `HbKvBuffers` (TQ-active, 8-bit codebook default), with `HybridKvBuffers` opt-in via `HF2Q_HYBRID_KV` and `DenseKvBuffers` reachable on `HF2Q_USE_DENSE=1`. **None of the four carries `n_seqs` in its shape** — Gemma 4 is fully single-seq by construction.
3. **mlx-native kernels split into two camps on `n_seqs`**: the linear-attn family (`ssm_conv*`, `gated_delta_net*`) DOES carry `n_seqs` as a kernel parameter (`/opt/mlx-native/src/shaders/ssm_conv.metal:48`, `gated_delta_net_decode.metal:111`); the flash-attention family + the `kv_cache_copy_seq*` family DOES NOT (they accept a single `capacity + seq_pos_start + n_tokens` triple and implicitly write to a single sequence's slab). The ADR's claim "zero new Metal kernels needed" is **true for the flash-attn path** but **requires per-seq dispatch loops at the Rust layer** for both `kv_cache_copy_seq*` and `flash_attn_prefill_*` writes when `slot_count > 1`.
4. **The KV-persist subsystem (ADR-017) already serializes `n_seqs`** end-to-end (`qwen35_hybrid_persistor.rs:129`) and stores per-seq `current_len[]` — but the Gemma 4 spill descriptor (`KvSpillDescriptor` at `kv_spill_descriptor.rs:76`) does NOT carry `n_seqs`/`max_slots`. ADR-040 Phase A5 (per-slot OOM + budget) will require descriptor extension.
5. **Recommended iter-2 starting point: Qwen35 `HybridKvCache` full-attn slot only**, keep `linear_attn` and `mtp_slot` slot-count = 1 with a documented `ADR-040 Phase A2.1` defer marker. Confidence: **medium-high** — the structural lift is well-bounded but the linear-attn carve-out adds a non-obvious correctness boundary that iter-2 MUST pin with a test.

---

## §2 Per-question findings

### §2.1 Q1: Qwen35 `HybridKvCache` n_seqs reality check

**ADR claim** (§1.3): "Qwen35 `HybridKvCache` already carries `n_seqs` in buffer shape — production wiring uses `n_seqs=1` today; the structural shape supports >1 with no buffer-layout change."

**Verdict**: **Partially verified; one major falsification (linear-attn capture buffer); two minor concerns (delta_net dispatcher hard-codes `n_seqs=1`, `current_len[0]` reads in SDPA dispatch).**

#### 2.1.1 `n_seqs` declaration site

- `pub n_seqs: u32` declared at `src/inference/models/qwen35/kv_cache.rs:695`
- Constructor signature: `HybridKvCache::new(cfg, device, max_seq_len, n_seqs)` at `kv_cache.rs:1124`
- Constructor preflight: `n_seqs == 0` → `Err("HybridKvCache: n_seqs must be > 0")` at `kv_cache.rs:1163-1164`
- Constructor with options: `new_with_options(cfg, device, max_seq_len, n_seqs, tq_kv_active)` at `kv_cache.rs:1157`

#### 2.1.2 Production callsite range

Grep `HybridKvCache::new\b` across `/opt/hf2q/src` yields **40+ callsites**; every single one passes `n_seqs=1` as the 4th argument. Examples:
- `src/quantize/imatrix/forward.rs:703` — imatrix calibration
- `src/inference/models/qwen35/spec_decode.rs:222` — spec-decode verifier cache
- `src/inference/models/qwen35/activation_capture_real.rs:135` — calibration prompt
- `src/inference/models/qwen35/forward_gpu.rs:6504, 6533, 6534, ..., 7567` — 30+ test-only callsites all `n_seqs=1`
- `src/inference/models/qwen35/mtp_tests.rs:244`

**Conclusion**: production exercises `n_seqs=1` exclusively. No callsite tests `n_seqs>1`. The constructor accepts it but no integration test verifies behaviour at `n_seqs=2+`. This is the bug-class iter-2 MUST close before declaring "byte-equivalent at slot 0".

#### 2.1.3 Full-attn slot: 4-D shape, `n_seqs` outer

- `FullAttnKvSlot.k/v: Option<MlxBuffer>` with shape `[n_seqs, n_kv_heads, max_seq_len, head_dim]` F32 — explicitly declared at `kv_cache.rs:14-15` and allocated at `kv_cache.rs:2226-2236` (in `alloc_full_attn_slot`).
- `current_len: Vec<u32>` — per-seq write cursor, length = `n_seqs`. Allocated at `kv_cache.rs:2213, 2247`.
- **Verified**: the full-attn slot's GPU buffer rank IS 4-D with `n_seqs` outer; lifting `n_seqs` from 1 to N at the alloc site is a 0-LOC schema change (the alloc already multiplies by `n_seqs as usize`).

#### 2.1.4 Linear-attn slot: also 3-D/4-D with `n_seqs`, BUT capture buffer is 5-D and asserts `n_seqs=1`

- `LinearAttnStateSlot.conv_state: MlxBuffer` with shape `[conv_channels, K-1, n_seqs]` F32, allocated at `kv_cache.rs:2267-2268`.
- `LinearAttnStateSlot.recurrent: MlxBuffer` with shape `[D_k, D_v, num_v_heads, n_seqs]` F32, allocated at `kv_cache.rs:2284-2289`.
- **The capture buffer for spec-decode rollback is shape `[D_k, D_v, num_v_heads, n_tokens_max, n_seqs]`** at `kv_cache.rs:1476-1480` — `n_tokens_max` is OUTER of `n_seqs`, but `rollback_la_to()` at `kv_cache.rs:1567` **explicitly errors when `self.n_seqs > 1`**:

```rust
if self.n_seqs > 1 {
    return Err(anyhow!(
        "rollback_la_to: n_seqs > 1 not supported (capture buffer \
         layout assumes n_seqs=1; production Qwen 3.5/3.6 is \
         always n_seqs=1). ..."
    ));
}
```

  Per the inline ADR-034 comment, the per-token slice math interleaves incorrectly because `n_tokens_max` is the slowest-varying axis before `n_seqs` while the kernel writes assume sequence-major layout.

**This is the ADR-040 §1.3 falsification**: lifting `n_seqs > 1` is not free in the linear-attn path. Either (a) the capture buffer layout must be reshaped + the `gated_delta_net_decode_capture` kernel re-derived for `n_seqs > 1`, or (b) iter-2 ships full-attn-only and explicitly defers linear-attn multi-seq to a follow-up iter.

#### 2.1.5 Hard-coded `n_seqs = 1` in dispatch paths

`grep "n_seqs = 1u32\|n_seqs: 1," src/inference/models/qwen35/gpu_delta_net.rs` finds **15+ hard-codes** at production dispatch sites:

- `gpu_delta_net.rs:912` — `let n_seqs = 1u32;` in chunk-scan dispatch
- `gpu_delta_net.rs:1090` — `let n_seqs = 1u32; // hf2q forward path is single-seq.`
- `gpu_delta_net.rs:1556` — `let n_seqs = 1u32; // hf2q forward path is single-seq.`
- `gpu_delta_net.rs:1964, 2705, 3368` — additional autoreg dispatch sites
- `gpu_delta_net.rs:5091, 5498, 5514` — capture-variant dispatch (test-only)

Plus structural assertions inside `dispatch_ssm_conv` and `dispatch_gated_delta_net*` consume `n_seqs` as a kernel parameter (`ssm_conv.metal:48`, `gated_delta_net_decode.metal:111`) and route grid dispatch by `n_seqs` — so the **kernel surface already supports `n_seqs > 1`, but every call site in hf2q passes `1`**.

#### 2.1.6 SDPA / full-attn cur_len read uses `current_len[0]`

- `gpu_full_attn.rs:4212` — `let cur_len = slot.current_len[0] as usize;` in `apply_sdpa_with_kv_cache`
- Doc-comment at `gpu_full_attn.rs:4169-4176` describes the cache as `[1, n_kv_heads, max_seq_len, head_dim] F32 (SDPA-native layout, n_seqs=1 for single-sequence inference)`.

The dispatch reads slot 0 specifically. Lifting to N requires every SDPA dispatch to either (a) thread a `slot_id: u32` parameter through and read `current_len[slot_id]`, or (b) run the dispatch in a per-slot loop. Per ADR-040 §2.2 the slot-id threading is **Phase B iter-3** (forward_prefill.rs / forward_prefill_batched.rs).

#### 2.1.7 ADR-027 TQ-active mode

- `TqFullAttnKvBuffers` struct at `kv_cache.rs:2349-2362` carries shape `[n_seqs, n_kv_heads, max_seq_len, head_dim]` U8 for `k_packed`/`v_packed` and `[n_seqs, n_kv_heads, max_seq_len, norms_per_pos]` F32 for `k_norms`/`v_norms`.
- `alloc_tq_full_attn_buffers(cfg, device, max_seq_len, n_seqs)` at `kv_cache.rs:2393-2407` — accepts `n_seqs` parameter and multiplies allocation by it (`packed_elems` at `kv_cache.rs:2416`).
- **Verified**: TQ buffer has `n_seqs` in its shape. Lifting to N at the alloc site is structurally safe.
- **Caveat**: `flash_attn_vec_tq_hb` kernel reads "the inner three axes `[n_kv_heads, max_seq_len, head_dim]` per sequence; the n_seqs outer dimension is consumed at the call site" (per `kv_cache.rs:2342-2346`). I.e. the kernel itself reads one seq at a time; multi-seq requires a per-seq dispatch loop (same pattern as full-attn F32).

#### 2.1.8 MTP slot

- `mtp_slot: Option<FullAttnKvSlot>` at `kv_cache.rs:691`. Same `FullAttnKvSlot` type as regular full-attn → same 4-D shape with `n_seqs` outer.
- Allocated via the same `alloc_full_attn_slot(cfg, device, max_seq_len, n_seqs, tq_kv_active)` at `kv_cache.rs:1212-1213`.
- **Verified**: MTP slot inherits `n_seqs` naturally; no separate lift.

---

### §2.2 Q2: Gemma 4 KV cache variants and ADR-040 Phase A3 target

**Verdict**: 4 distinct structs, **none carry `n_seqs`**, current production = `HbKvBuffers` (TQ-active), Phase A3 lift requires adding a `n_seqs` axis to whichever variant production uses.

#### 2.2.1 Variant inventory (all at `src/inference/models/gemma4/kv_cache.rs`)

| Struct | Lines | K shape | V shape | Carries `n_seqs`? |
|---|---|---|---|---|
| `MlxKvCache` | 14-31 | `[num_kv_heads, capacity, head_dim/2]` U8 (4-bit nibble-packed) | same as K | No |
| `HbKvBuffers` | 84-100 | `[nkv_heads, capacity, head_dim]` U8 (byte-packed) | same | No |
| `DenseKvBuffers` | 103-130 | `[nkv, cap, hd]` (dtype F32 or F16 per `dtype` field) | same | No |
| `HybridKvBuffers` | 166-199 | `[nkv, cap, hd]` F16 | `[nkv, cap, hd]` U8 + `v_norms` F32 + optional BF16 xlen K/V | No |

**All four are 3-D**. There is no `n_seqs` axis anywhere in Gemma 4 KV storage. The struct field is `seq_len: usize` (singular) at `kv_cache.rs:30` (for `MlxKvCache`) — per-cache, not per-slot.

#### 2.2.2 Production routing

Per `forward_gpu.rs:407-465`:
- Default routing reads `INVESTIGATION_ENV.tq_codebook_bits` (default 8 per ADR-007) and `INVESTIGATION_ENV.hybrid_kv` (default false).
- `cb_bits >= 5 && hybrid_kv` → `HybridKvBuffers` (F16 K + TQ-HB V).
- `cb_bits >= 5 && !hybrid_kv` → `HbKvBuffers` (legacy TQ-HB on both K and V) — **this is the production default for Gemma 4 today**.
- `cb_bits == 4` → `MlxKvCache` (4-bit nibble-packed).
- `HF2Q_USE_DENSE=1` → `DenseKvBuffers` (F32/F16 dense).

For chat completions today: **`HbKvBuffers`** is the live variant per ADR-007 §1.2 (TQ-active 8-bit default).

#### 2.2.3 TQ-active path's per-cache variant

- Same `MlxKvCache` / `HbKvBuffers` / `HybridKvBuffers` structs — TQ-active is not a separate cache variant, it's a codec-bits selection on the same `HbKvBuffers` / `HybridKvBuffers` storage.

#### 2.2.4 LOC estimate per variant to add `n_seqs`

| Variant | Buffer-shape change | Per-seq cursor (`seq_len: Vec<u32>` of length `n_seqs`) | Write-routing | Total est. |
|---|---|---|---|---|
| `MlxKvCache` | 4 buffer reshapes at `model.rs:1282-1290` | replace `seq_len: usize, write_pos: usize` with `Vec<u32>`s | `trim()` at `kv_cache.rs:56` + 4 prod write sites | ~80 LOC |
| `HbKvBuffers` | 2 buffer reshapes + 2 norms reshapes at `forward_gpu.rs:443-459` (decode) + `forward_prefill.rs:843-882` + `forward_prefill_batched.rs:443-475` | NEW field `seq_len: Vec<u32>` | `dispatch_hadamard_quantize_kv_*` write site (per-seq loop) | ~120 LOC |
| `DenseKvBuffers` | 2 reshapes at `forward_prefill.rs:703`, `forward_prefill_batched.rs:367`, `engine.rs:3999` | NEW field | `dispatch_kv_cache_copy_seq_f32_dual` per-seq dispatch | ~150 LOC (more callsites) |
| `HybridKvBuffers` | F16 K + U8 V + F32 norms + optional BF16 xlen K/V — 4 reshapes | NEW field | `alloc_hybrid_kv_for_layer` at `kv_cache.rs:218-272` + dispatch sites | ~180 LOC (xlen-mode optional buffers complicate) |

**Phase A3 ITER target recommendation**: lift `HbKvBuffers` first (production default). Defer `MlxKvCache` (legacy 4-bit, off-path), `DenseKvBuffers` (dev/debug path), `HybridKvBuffers` (opt-in via `HF2Q_HYBRID_KV`). Total Phase A3 iter-1 LOC ≈ 120-150 (struct + allocator + per-seq cursor + 3-4 write sites).

---

### §2.3 Q3: mlx-native kernel surface for multi-seq

**ADR claim** (§2.3): "zero new Metal kernels needed for Phase A/B/C."

**Verdict**: **True for flash-attn family; requires per-seq Rust-layer dispatch loops for kv_cache_copy_seq family; already supported in linear-attn family but not exercised.**

#### 2.3.1 Linear-attn kernel family (DOES carry n_seqs)

- `dispatch_ssm_conv` at `/opt/mlx-native/src/ops/ssm_conv.rs` — `SsmConvParams` includes `n_seqs: u32`, kernel reads it at `ssm_conv.metal:48, 90, 142, 176`.
- `dispatch_gated_delta_net_decode` at `/opt/mlx-native/src/ops/gated_delta_net_decode.rs` — same; kernel grid dispatched by `(D_v/NSG, n_v_heads, n_seqs)` per `gated_delta_net_decode.metal:44`.
- `dispatch_gated_delta_net` — same `n_seqs` parameter, shape `[D_k, n_k_heads, n_tokens, n_seqs]` per `gated_delta_net.metal:35-39`.

These kernels **could already dispatch `n_seqs > 1`** if the Rust caller passed it. Today every caller passes `n_seqs = 1` (see §2.1.5).

#### 2.3.2 Flash-attn kernel family (DOES NOT carry n_seqs; per-seq Rust loop needed)

Signature of representative kernel:

```rust
pub fn dispatch_flash_attn_prefill_bf16_d256(
    encoder: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    q: &MlxBuffer,
    k: &MlxBuffer,
    v: &MlxBuffer,
    mask: Option<&MlxBuffer>,
    out: &MlxBuffer,
    params: &FlashAttnPrefillParams,
) -> Result<()>
```

(`/opt/mlx-native/src/ops/flash_attn_prefill.rs:487`)

No `n_seqs` or `slot_id` parameter. `FlashAttnPrefillParams` carries seq_len_q, seq_len_k, n_heads, head_dim, etc. **The kernel operates on a single sequence's Q/K/V buffer slice**.

Multi-seq dispatch options:
- **(a) Per-seq loop at Rust layer** — encode N flash-attn dispatches sequentially on the same encoder, each with a sliced Q/K/V buffer view (or sliced byte offsets). No kernel change. ADR's "zero new Metal kernels" claim holds. Cost: linear in N at the GPU command-encoder level.
- **(b) Add a `slot_stride` byte offset to `FlashAttnPrefillParams`** — caller's K/V buffer is the full slab `[n_seqs, n_kv, max_seq, head_dim]`; kernel reads slot 0 by default. New kernel constant (cheap, no kernel logic change). Requires touching `flash_attn_prefill_*.metal` to add the offset to load indices.
- **(c) Per-seq separate `MlxBuffer` per slot** — each slot has its own K and V buffer; per-seq dispatch is just per-buffer dispatch. No kernel change. Cost: N× the buffer-handle overhead.

**Recommendation**: option (a) for iter-2 — preserves "zero new kernels" claim and matches Rust-layer slot threading. Option (b) is the iter-2+1 optimisation if benchmark shows command-encoder overhead matters.

#### 2.3.3 KV-cache copy family (DOES NOT carry n_seqs)

Signature:

```rust
pub fn dispatch_kv_cache_copy_seq_f32_dual(
    encoder, registry, device,
    src_k, src_v,
    cache_k, cache_v,
    n_heads, head_dim,
    capacity, seq_pos_start, n_tokens, src_tok_offset,
) -> Result<()>
```

(`/opt/mlx-native/src/ops/kv_cache_copy.rs:542`)

`capacity` here is `max_seq_len` for the single sequence; the kernel writes `cache[h, seq_pos_start + t, d] = src[t, h, d]` style. **No `n_seqs` or slot offset**. Multi-seq write requires per-seq dispatch (option a) or kernel signature change to accept a slot byte-offset.

#### 2.3.4 Cited file:line references

- `/opt/mlx-native/src/shaders/ssm_conv.metal:48`: `const uint n_seqs = params[2];`
- `/opt/mlx-native/src/shaders/gated_delta_net_decode.metal:44`: `// grid (in tg): (D_v / NSG, n_v_heads, n_seqs)`
- `/opt/mlx-native/src/ops/flash_attn_prefill.rs:487`: no `n_seqs` in signature
- `/opt/mlx-native/src/ops/kv_cache_copy.rs:542`: no `n_seqs` in signature

---

### §2.4 Q4: ADR-017 KV-spill interaction

**Verdict**: Qwen35 hybrid persistor ALREADY threads `n_seqs`; Gemma4 dense spill descriptor does NOT and will need extension. Per-slot lifetime tracking is a Phase A5 concern (operator budget enforcement), NOT a Phase A2/A3 concern.

#### 2.4.1 What the spiller serializes today

`src/serve/kv_persist/families/qwen35_hybrid_persistor.rs:129` declares:

```rust
pub struct Qwen35HybridConfig {
    pub n_full_attn: u32,
    pub n_linear_attn: u32,
    pub has_mtp: bool,
    pub n_seqs: u32,             // <-- threaded end-to-end
    pub full_attn_shape: [u64; 4],   // [n_seqs, n_kv_heads, max_seq_len, head_dim]
    pub linear_conv_shape: [u64; 3], // [conv_channels, K-1, n_seqs]
    pub linear_recurrent_shape: [u64; 4],
    pub mtp_shape: [u64; 4],
    ...
}
```

- Wire format documented at `qwen35_hybrid_persistor.rs:35`: `[n_seqs: u32 LE]` in envelope header.
- `n_seqs` validated at deserialize against producer-time value at `qwen35_hybrid_persistor.rs:171-175`.
- `current_len: Vec<u32>` per slot serialized as `n_seqs` u32-LE values at `qwen35_hybrid_persistor.rs:521-528`.
- MTP slot `current_len` serialized same way at `qwen35_hybrid_persistor.rs:716-722`.

**Verified**: Qwen35 spiller is `n_seqs`-aware end-to-end. Iter-2 lift will round-trip naturally — but the spiller has NEVER been exercised with `n_seqs > 1` (all integration tests at `qwen35_hybrid_persistor.rs:1495+` use `n_seqs=1` or synthetic small values).

#### 2.4.2 Gemma 4 spill descriptor lacks n_seqs

`src/serve/api/kv_spill_descriptor.rs:76`:

```rust
pub struct KvSpillDescriptor {
    pub sliding_window: usize,
    pub max_decode_tokens: usize,
    pub num_layers: usize,
    pub layer_types: Vec<LayerType>,
    pub nkv_heads: Vec<usize>,
    pub head_dim: Vec<usize>,
    pub kv_dtype: KvDType,
    pub provenance: KvSpillProvenance,
}
```

No `n_seqs` or `max_slots`. Phase A3 lift must extend this struct (and `Gemma4DenseConfig` in `families/gemma4_dense.rs`) to thread per-seq metadata through the spiller hooks.

#### 2.4.3 Per-slot lifetime tracking

- Today the spiller assumes the whole cache is one logical unit. Per-slot spill-then-evict would require either:
  - (a) treating each slot as a separate per-(repo, quant, slot_id) namespace key — easiest, but the LcpRegistry's `(ModelFingerprint × prefix-hash)` keying scheme would need a 3rd axis.
  - (b) spilling the whole multi-slot slab and letting restore install all-N slots at once — simpler at iter-2; pessimistic on bandwidth.

**Recommendation**: iter-2 ships **(b)** — spill whole slab including unused slots. Iter-A5 (per-slot OOM + budget) revisits with per-slot lifetime via descriptor extension.

#### 2.4.4 Closest existing entry point

- Gemma4 dense KV-restore site: `src/serve/api/engine.rs:3999` (the `DenseKvBuffers` construction in `request_kv_restore` worker handler).
- This is the file:line that must change to alloc N slots when descriptor extends; today it allocs one slab per layer.

---

### §2.5 Q5: ADR-027 TQ-active KV interaction

**Verdict**: TQ buffer shape carries `n_seqs`; TqPackedSpillDescriptor does NOT. Multi-seq + TQ is **structurally compatible at iter-2** as long as we lift `HbKvBuffers` (Gemma) and `TqFullAttnKvBuffers` (Qwen35) in lockstep with their parent caches.

#### 2.5.1 TqPackedSpillDescriptor

`src/serve/api/tq_packed_descriptor.rs:43`:

```rust
pub struct TqPackedSpillDescriptor {
    pub bits_per_coord: TqBitsPerCoord,
    pub num_layers: usize,
    pub nkv_heads: Vec<u32>,
    pub head_dim: Vec<u32>,
    pub block_tokens: u32,
    pub flags: u32,
    pub scale: f64,
    pub provenance: KvSpillProvenance,
}
```

No `n_seqs`. Same observation as Gemma4 dense spill descriptor (§2.4.2). Phase A3 extension applies.

#### 2.5.2 Iter-2 strategy: ship non-TQ first?

**Not necessary** for Qwen35 — `TqFullAttnKvBuffers` already carries `n_seqs` in its shape (§2.1.7), the SDPA dispatch reads "one seq at a time" with `n_seqs outer dimension consumed at the call site" (`kv_cache.rs:2345-2346`). Per-seq loop at the dispatch site works for both F32 and TQ paths.

**Needed** for Gemma4 — `HbKvBuffers` and `HybridKvBuffers` are 3-D today; Phase A3 must add the `n_seqs` axis explicitly. This is a fork-in-the-road: either lift both F32 and TQ at iter-A3 (additive scope), or ship one then the other (additive iters).

**Recommendation**: A3 ships `HbKvBuffers` (TQ default) lift; non-default `DenseKvBuffers` and `MlxKvCache` (4-bit) stay single-seq for one iter with a documented `slot_count() = 1` clamp in their `MultiSeqKvCache` impls.

---

### §2.6 Q6: Spec-decode interaction

**Verdict**: Both `eagle3::DrafterKvCache` and `dflash::DFlashLayerKvCache` are **single-seq by construction (3-D `[num_kv_heads, capacity, head_dim]`)**. Per ADR-040 §6 Phase A iter-4 (research-quality) AND §5 AC-1 (not in Phase E1 gate), Phase A iter-2 + iter-3 do NOT need to lift them.

#### 2.6.1 Eagle3 drafter cache

`src/inference/spec_decode/eagle3/kv_cache.rs`:
- `DrafterKvCache.k_buf: MlxBuffer` shape `[num_kv_heads, capacity, head_dim]` F32 (per L48 + L17).
- No `n_seqs` anywhere in the file (grep result: 0 hits for `n_seqs` outside doc-comments).
- `append`, `rollback_to_accepted`, etc. — all single-seq.

#### 2.6.2 DFlash drafter cache

`src/inference/spec_decode/dflash/kv_cache.rs`:
- `DFlashLayerKvCache.keys/values: MlxBuffer` shape `[num_kv_heads, capacity, head_dim]` F32 (per L48).
- Module-level comment at L4-16 explicitly assumes single-seq via "block-diffusion scenarios" and asserts non-wrap.

#### 2.6.3 Recommendation

**Phase A iter-4 (drafter caches) should NOT be in scope for iter-2.** Per ADR-040 §4 question 5, spec-decode under continuous batching is a Phase E1 OPEN question; the default answer per §6's iter-4 entry is "research-quality only". Iter-2 and iter-3 ship Qwen35 + Gemma4 verifier-side multi-seq; the drafter cache continues to operate single-seq (current behaviour, single drafter K/V slot, no scheduler interaction).

---

### §2.7 Q7: Per-model impl shape proposal

**Trait surface** (per `src/serve/multi_seq_kv.rs:274`):

```rust
pub trait MultiSeqKvCache {
    fn layout(&self) -> MultiSeqLayout;
    fn slot_count(&self) -> u32;
    fn seq_len(&self, slot: SlotId) -> Result<u32, MultiSeqError>;
    fn append_for_seq(&mut self, slot: SlotId, n_tokens: u32) -> Result<(), MultiSeqError>;
    fn drop_seq(&mut self, slot: SlotId) -> Result<(), MultiSeqError>;
    fn fork_seq(&mut self, src: SlotId, dst: SlotId) -> Result<(), MultiSeqError>;
}
```

#### 2.7.1 Minimum real implementations vs NotImplemented

| Method | Qwen35 HybridKvCache | Gemma4 HbKvBuffers (per-layer) | LOC est. (per-model) |
|---|---|---|---|
| `layout()` | Always `SeparateSlots` | Always `SeparateSlots` | 2 LOC each |
| `slot_count()` | Return `self.n_seqs` | NEW field `n_seqs: u32` on container | 2 LOC + 1 field each |
| `seq_len(slot)` | `full_attn[0].current_len[slot.0]` (asserts homogeneous current_len across full_attn slots — TRUE in production, no spec-decode-aware split) | NEW per-cache `seq_lens: Vec<u32>` | 5-10 LOC |
| `append_for_seq(slot, n)` | Bump `full_attn[*].current_len[slot.0]` AND `mtp_slot.as_mut().map(\|s\| s.current_len[slot.0])` AND `linear_attn` slot-cursor (but the linear-attn slot has no logical "cursor" — recurrent state is updated in-kernel, not by an append cursor; iter-2 may treat linear-attn `append` as no-op + log) | NEW `seq_lens[slot.0] += n` + `write_pos` bookkeeping | 25-40 LOC |
| `drop_seq(slot)` | Zero `current_len[slot.0]` across full_attn + mtp + zero recurrent state for slot — but zeroing recurrent state requires a per-slot Metal kernel (the recurrent buffer is `[D_k, D_v, num_v_heads, n_seqs]` so per-slot zero is a contiguous slice for the outermost axis IFF the layout is "n_seqs outer" — VERIFY) | Reset `seq_lens[slot.0] = 0`, no buffer zero | 20-50 LOC (depends on need to zero recurrent state) |
| `fork_seq(src, dst)` | Copy full_attn K/V slab + MTP K/V slab from src to dst (per-slot byte-strided memcpy via `dispatch_kv_cache_copy_seq_*`) AND copy linear-attn conv_state + recurrent_state slot slabs | Copy K/V slab slot src → dst | 30-80 LOC |

#### 2.7.2 Is `fork_seq` used by any production callsite?

`grep "fork_seq" src/ --include="*.rs"` outside `multi_seq_kv.rs`: **0 callsites**.

Per ADR-040 §6 Phase B iter-6 ("Mixed prefill+decode `SchedulerStep::Mixed` handling") and Phase B iter-6's reference to "admitting a new request that shares a prefix with an in-flight slot — fork is cheaper than re-prefilling" (multi_seq_kv.rs:318-320), `fork_seq` is **NOT used by any current callsite**.

**Iter-2 recommendation**: implement `fork_seq` as `Err(MultiSeqError::SlotOom { ... unimplemented })` or a TODO-marker comment + `unimplemented!()`. **Wait** — mantra says no stubs. Cleaner option: implement `fork_seq` as a real per-buffer memcpy from the start (it's structurally identical to `dispatch_kv_cache_copy_seq_f32_dual` between two slot offsets on the same cache buffer). Estimated 30-50 LOC; not a major delta on the iter-2 critical path.

**Verdict**: ship `fork_seq` real impl in iter-2. The mantra prohibits stub fallback, and a working fork is one same-encoder kernel dispatch — not expensive.

#### 2.7.3 LOC estimate per model

| Model | Total LOC est. for iter-2 trait impl |
|---|---|
| Qwen35 `HybridKvCache` | ~150-250 LOC (full-attn + MTP + linear-attn carve-out) |
| Gemma4 `HbKvBuffers` | ~180-280 LOC (struct extension + alloc site + 3 write sites + trait impl) |

---

### §2.8 Q8: Testable hypotheses for Phase A2 (Qwen35)

The "Never guess" mantra means iter-2 must turn each lift into a falsifiable hypothesis BEFORE writing code.

#### H1 (Phase A2 / falsifiable in 1 hour)

> **H1**: Lifting `HybridKvCache::new(.., n_seqs=4)` (without changing any forward path) **does not panic** during allocation and produces buffers of exactly 4× the byte size of the `n_seqs=1` baseline (full-attn K/V + linear-attn recurrent).

**Falsification test**: write a unit test `hybrid_kv_cache_alloc_n_seqs_4_no_panic` that:
1. Constructs `HybridKvCache::new(&cfg, &device, 64, 4)` for a synthetic small `Qwen35Config`.
2. Asserts `cache.n_seqs == 4` and `cache.full_attn[0].k.as_ref().unwrap().byte_len() == 4 * baseline_byte_len`.
3. Calls `total_bytes()` and asserts 4× scaling within ±1% (linear-attn capture buffers may have non-linear scaling per `kv_cache.rs:1567+` guard).

**Why it's not guess-equivalent**: §2.1.2 shows zero `n_seqs > 1` callsites exist in the entire codebase. We are claiming the allocator works at `n_seqs=4` without proof. Without H1 passing, every later hypothesis is built on sand.

#### H2 (Phase A2 / falsifiable in 1 day)

> **H2**: With `n_seqs=4`, calling `forward_prefill_gpu(.., slot_offset=0)` (the existing single-seq path) produces **byte-identical** logits to `n_seqs=1` — i.e. the lift is invisible to slot-0 readers.

**Falsification test**: parity test `qwen35_forward_byte_identical_at_n_seqs_4_slot_0_vs_n_seqs_1` running the same prompt through both configurations and asserting `Linf(logits_4, logits_1) == 0.0` (NOT NRMSE — byte-identical).

**Why it falsifies the ADR claim**: §1.3's "no buffer-layout change" is testable. If logits drift even by ULP, the layout HAS changed (perhaps from stride misalignment in the 4-D buffer that wasn't there at 3-D-effective).

#### H3 (Phase A2 / falsifiable in 2-3 days)

> **H3**: Per-slot `append_for_seq` is O(1) per ADR-040 §5 AC-1, even with N=4. Specifically, `append_for_seq(SlotId(0), 1)` followed by `append_for_seq(SlotId(2), 1)` writes to two distinct cache regions and DOES NOT corrupt slot 1's contents.

**Falsification test**: integration test `hybrid_kv_per_slot_isolation_n_seqs_4` that:
1. Fills slot 1 with known data via direct buffer write.
2. Calls `append_for_seq(SlotId(0), 5)` and `append_for_seq(SlotId(2), 5)`.
3. Asserts slot 1's KV slab is unchanged byte-for-byte.

#### H4 (Phase A2 / falsifiable in 1 day)

> **H4**: `HybridKvCache.linear_attn[i].recurrent` shape `[D_k, D_v, num_v_heads, n_seqs]` has `n_seqs` as the **outermost** axis (slowest-varying), so per-slot drop is a contiguous byte-slice zero — no kernel needed.

**Falsification test**: read the `MlxBuffer` strides directly (or by allocating at `n_seqs=4` and checking element-count math). If `n_seqs` is NOT outermost (e.g. it's the innermost axis), then per-slot drop requires a strided-write kernel.

**Stakes**: if H4 is false, `drop_seq` LOC estimate balloons from 20-50 to 100-200 (per-arch strided drop kernel needed).

#### H5 (Phase A2 / falsifiable in 1 day)

> **H5**: The `rollback_la_to` guard at `kv_cache.rs:1567` is the ONLY hard-coded `n_seqs == 1` assertion in the linear-attn path. All other linear-attn dispatches at `gpu_delta_net.rs:912/1090/1556/...` are "soft" hard-codes that can be lifted by simply passing `cache.n_seqs` instead of `1u32`.

**Falsification test**: hypothesis-driven grep: enumerate every `n_seqs = 1u32` in `gpu_delta_net.rs`, classify each as either (a) "test fixture, kept at 1" or (b) "production dispatch, must lift". Iter-2 lifts category (b) and counts the deltas.

**Why this matters**: if H5 is false, the linear-attn carve-out is larger than 1 file:line and iter-2 may need to defer linear-attn entirely to iter-2b.

---

### §2.9 Q9: Testable hypotheses for Phase A3 (Gemma4)

#### H6 (Phase A3 / falsifiable in 1 hour)

> **H6**: Extending `HbKvBuffers` struct with a new field `seq_lens: Vec<u32>` and lifting `k_packed`/`v_packed` shape from 3-D `[nkv, cap, hd]` to 4-D `[n_seqs, nkv, cap, hd]` requires **zero `dispatch_hadamard_quantize_kv_*` kernel changes** because the existing kernels accept per-call `(cache_capacity, write_pos)` and treat the buffer as a flat slab keyed by those parameters.

**Falsification test**: write a synthetic `HbKvBuffers` at `n_seqs=2`, run `dispatch_hadamard_quantize_kv_hb` with `write_pos` adjusted to address slot 1's slab (via `cache_capacity * 1 * nkv + write_pos` byte arithmetic at the caller), assert byte-identical output to a standalone slot-1-only buffer.

#### H7 (Phase A3 / falsifiable in 1 day)

> **H7**: Gemma 4's sliding-window path (`is_sliding=true` ring buffer) is per-slot isolated: lifting to N=4 means N independent ring buffers, each wrapping independently. No cross-slot wrap interaction.

**Falsification test**: parity test that fills slot 0's sliding cache past `capacity` (forcing wrap), then checks slot 1's cache is empty. Validates that the ring-buffer wrap math at `MlxKvCache::trim` (`kv_cache.rs:56-71`) doesn't read across slot boundaries.

#### H8 (Phase A3 / falsifiable in 2 days)

> **H8**: The 3 distinct construction sites (`forward_prefill.rs:843-882`, `forward_prefill_batched.rs:443-475`, `forward_gpu.rs:443-459`) can be unified into a single `alloc_hb_kv_for_layer(dev, layer_idx, nkv, hd, cap, is_ring, n_seqs)` helper following the existing `alloc_hybrid_kv_for_layer` (`kv_cache.rs:218`) pattern, eliminating drift risk.

**Falsification test**: refactor + assert by-construction that all 3 sites produce identical buffer shapes for `n_seqs=1` (regression pin), then independently test `n_seqs=2/4` from the helper.

#### H9 (Phase A3 / falsifiable in 1 day)

> **H9**: The Gemma 4 sliding-window attention layers ALWAYS coexist with full-attention layers in the same model — every Gemma 4 layer is either `LayerType::Sliding` or `LayerType::FullAttention` (per `KvSpillDescriptor.layer_types`). Therefore the multi-seq lift must handle both per-layer (not all-sliding or all-full).

**Falsification test**: read `weights.layer_types` for a production Gemma 4 GGUF; assert mixed layer_types vector. If verified, iter-A3 MUST lift both branches simultaneously.

#### H10 (Phase A3 / falsifiable in 2 days)

> **H10**: `HF2Q_HYBRID_KV=1` (HybridKvBuffers path) is opt-in and not enabled in the production ADR-040 §3.4 default config (`max_slots=4`); therefore Phase A3 iter-2 can ship `HbKvBuffers` multi-seq WITHOUT lifting `HybridKvBuffers`, with a documented "iter-A3b lifts HybridKvBuffers" deferral.

**Falsification test**: check the production CI/test matrix for `HF2Q_HYBRID_KV=1` exposure. If any prod-equivalent test runs with the gate on, iter-A3 must lift both.

---

### §2.10 Q10: Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Linear-attn capture buffer (`kv_cache.rs:1567` guard) cannot be lifted in iter-2; spec-decode + continuous-batching combo blocked | **High** (the guard is real, the layout reason is documented) | **Medium** (spec-decode is research-quality per §5 AC-1; non-spec-decode multi-seq still ships) | Iter-2 ships full-attn lift only; mark `rollback_la_to` as ADR-040 Phase A2.5 deferred; add a regression test confirming non-spec-decode multi-seq path is unaffected by the guard |
| R2 | `current_len[0]` reads in `gpu_full_attn.rs` (apply_sdpa_with_kv_cache and similar) require parameter threading through to `forward_prefill.rs` / `forward_prefill_batched.rs`, which is Phase B iter-3 scope | Certain | High (without it, iter-2 cannot expose `slot_id > 0` to any real forward) | Iter-2's `append_for_seq` impl ONLY mutates per-cache state; the forward-path slot threading lands in Phase B iter-3 per ADR-040 §2.2. Document this dependency explicitly in iter-2's PR description so the per-family parity gate (iter-A6) doesn't fire prematurely |
| R3 | Gemma 4 has 4 KV variants and Phase A3 iter-1 only lifts one (`HbKvBuffers`); operators flipping `HF2Q_USE_DENSE=1` or `HF2Q_HYBRID_KV=1` mid-deployment hit a `LayoutNotSupported`-equivalent error | Medium (env flips are documented but used) | Medium (returns clear error vs silent corruption) | Iter-A3 ships explicit `slot_count() == 1` clamps in `DenseKvBuffers` / `MlxKvCache` / `HybridKvBuffers` impls with `Err(MultiSeqError::LayoutNotSupported { layout: SeparateSlots })` on `slot > 0` — same error shape, different cause |
| R4 | KV-spill (ADR-017) test surface has zero `n_seqs > 1` coverage; first multi-seq deployment hits an untested spill+restore path | Medium-High | Medium (worst case: cache corruption on restore) | Phase A6 (per-family parity gate) MUST include a `qwen35_hybrid_persistor_roundtrip_n_seqs_4` test. The persistor's wire format ALREADY supports `n_seqs` (`qwen35_hybrid_persistor.rs:171-175`) but has never serialized a value > 1. Add explicit envelope round-trip test at iter-2's PR. |
| R5 | `fork_seq` implementation requires same-encoder slot-to-slot memcpy via `dispatch_kv_cache_copy_seq_*` between buffer offsets of the same cache buffer; this is a NEW kernel pattern not exercised today | Medium | Low (correctness verifiable by unit test; no impact on FIFO contract) | Iter-2 ships `fork_seq` only after a unit test pins per-byte equivalence of (write to src → fork to dst → read from dst) vs (write to src and dst directly). If the kernel can't do same-buffer cross-region memcpy in one dispatch, fall back to a 2-step (download src to staging → upload to dst) at cost of one CPU round-trip per fork. |

---

## §3 Concrete hypothesis matrix

### Phase A2 (Qwen35 HybridKvCache)

| ID | Hypothesis | Test name (proposed) | Falsifies what claim | Cost to test |
|---|---|---|---|---|
| H1 | `n_seqs=4` alloc does not panic; bytes scale linearly | `hybrid_kv_alloc_n_seqs_4_byte_scale` | ADR §1.3 "structural shape supports >1 with no buffer-layout change" — half | 1 hour |
| H2 | At `n_seqs=4`, slot 0 logits are byte-identical to `n_seqs=1` | `qwen35_forward_byte_identical_at_n_seqs_4_slot_0_vs_n_seqs_1` | ADR §5 AC-1 "byte-equivalence at slot 0" | 1 day |
| H3 | Per-slot append is O(1) and isolated | `hybrid_kv_per_slot_isolation_n_seqs_4` | ADR §5 AC-1 "per-slot append + drop is O(1)" | 2-3 days |
| H4 | `n_seqs` is outermost axis in recurrent state → per-slot drop is contiguous-slice zero, no kernel | `linear_attn_recurrent_n_seqs_outermost_stride` | Trait `drop_seq` O(1) bound holds for linear-attn | 1 day |
| H5 | All `n_seqs = 1u32` in `gpu_delta_net.rs` are soft hard-codes (pass `cache.n_seqs` to lift) | `gpu_delta_net_n_seqs_classification` (manual probe, no kernel) | §2.1.5 enumeration is complete | 1 day |

### Phase A3 (Gemma4 HbKvBuffers)

| ID | Hypothesis | Test name (proposed) | Falsifies what claim | Cost to test |
|---|---|---|---|---|
| H6 | TQ kernels accept multi-seq via cache-byte-offset (no kernel change) | `hb_kv_buffers_dispatch_hadamard_quantize_multi_seq_via_offset` | ADR §2.3 "zero new Metal kernels needed" — for Gemma | 1 hour |
| H7 | Sliding-window ring is per-slot isolated | `mlx_kv_cache_sliding_per_slot_isolation` | Cross-slot wrap interaction does NOT occur | 1 day |
| H8 | 3 alloc sites can be unified into one helper | `hb_kv_alloc_helper_byte_equivalent_to_3_inlined_sites` | Drift risk reducible to zero by refactor | 2 days |
| H9 | Gemma 4 layer_types always mixed (sliding + full) | `gemma4_layer_types_always_mixed_in_production` | Phase A3 must lift both branches | 1 day |
| H10 | `HF2Q_HYBRID_KV=1` is opt-in, not exercised in default CI | `hybrid_kv_env_gate_off_in_prod_default` | Iter-A3 can defer HybridKvBuffers safely | 2 days |

---

## §4 Recommended sequencing for iter-2

### Iter-2a (Phase A2, Qwen35 first — recommended start)

**Goal**: lift `HybridKvCache` to N slots for FULL-ATTENTION only; defer linear-attn multi-seq to A2b.

**Steps** (in order):
1. **Run H1 unit test first** (1 hour) — verifies the structural claim before writing any production code. If H1 fails, the entire ADR §1.3 footing is wrong and operator must be notified.
2. **Implement `MultiSeqKvCache` for `HybridKvCache`** in `src/inference/models/qwen35/kv_cache.rs` (~150 LOC at the bottom of the file). Methods:
   - `layout()` → `SeparateSlots`
   - `slot_count()` → `self.n_seqs`
   - `seq_len(slot)` → bounds-check + return `self.full_attn[0].current_len[slot.0]` (assumes homogeneous current_len across full_attn slots — TRUE in production; assert at the boundary)
   - `append_for_seq(slot, n)` → bump `current_len[slot.0]` across ALL full_attn slots + MTP slot. NO linear-attn mutation (the `LinearAttnStateSlot` has no logical cursor).
   - `drop_seq(slot)` → zero `current_len[slot.0]` across all slots; do NOT zero recurrent state (deferred to A2b once linear-attn multi-seq lands; document this as a known-deferred behavior).
   - `fork_seq(src, dst)` → kernel-dispatch slab memcpy for full_attn K/V + MTP K/V; defer linear-attn fork.
3. **Run H2 + H3 + H4 tests** (~3-4 days total) to pin byte-equivalence and isolation.
4. **Wire `n_seqs` from operator surface**: Engine spawn-time reads `HF2Q_MAX_SLOTS` env (default 4 per ADR-040 §3.4), threads to `HybridKvCache::new`. Today's default behaviour preserved iff the trait's `slot_count() == 1` case is byte-equivalent to `n_seqs=1` legacy path.
5. **Land the persistor multi-seq round-trip test** (R4 mitigation): `qwen35_hybrid_persistor_roundtrip_n_seqs_4` exercises the existing `n_seqs`-aware wire format for the first time.
6. **CFA acceptance gate**: H1-H5 all PASS + existing 100+ qwen35 unit tests UNCHANGED + new persistor round-trip PASS.

**Estimated effort**: 3-5 days (consistent with ADR-040 §6 Phase A iter-2 estimate).

### Iter-2b (Phase A2 follow-up: linear-attn multi-seq)

**Goal**: lift `rollback_la_to` guard at `kv_cache.rs:1567`; reshape the per-token capture buffer; lift `gpu_delta_net.rs:912/1090/1556/...` hard-codes.

**Why deferred from iter-2a**: it's gated on the spec-decode + multi-seq combo (Phase E1 OPEN question 5), and lifting the capture-buffer layout requires either re-derivation of the `gated_delta_net_decode_capture` kernel OR a new dispatch wrapper. Both are >3 days of work each.

**Estimated effort**: 5-8 days.

### Iter-3 (Phase A3, Gemma4 HbKvBuffers)

**Goal**: extend `HbKvBuffers` struct with `n_seqs` axis + per-seq cursor; lift 3 alloc sites via H8's unified helper; implement `MultiSeqKvCache` for the per-layer KV cache aggregate (NOT per-buffer — likely a new `Vec<HbKvBuffers>` wrapper struct).

**Steps**:
1. Run H6 + H7 + H9 hypothesis tests first.
2. Refactor 3 alloc sites into `alloc_hb_kv_for_layer(dev, layer_idx, nkv, hd, cap, is_ring, n_seqs)` (per H8).
3. Implement trait on the wrapper.
4. Extend `KvSpillDescriptor` with `n_seqs` field; thread through `gemma4_dense` persistor.

**Estimated effort**: 3-5 days (consistent with ADR-040 §6 Phase A iter-3 estimate).

### Recommended overall sequencing

```
Iter-2a (Qwen35 full-attn) [3-5d]
  ├─→ Iter-3 (Gemma4 HbKvBuffers) [3-5d] — runs in parallel possible
  └─→ Iter-2b (Qwen35 linear-attn) [5-8d] — gated on iter-2a + iter-3 stability
        └─→ Iter-4 (Drafter caches) [5-8d] — research-quality, gated on Phase E1 OPEN question 5
```

**Do NOT** start iter-3 before iter-2a's H1-H5 all PASS. The hypotheses inform whether the per-model lift discipline is correct.

---

## §5 Closure note

The ADR's "existing footholds" framing is structurally correct but understates the linear-attn carve-out. The recommended iter-2 starts with **Qwen35 full-attn-only lift** (smallest blast radius, highest signal), pins the persistor round-trip (R4), and explicitly DEFERS linear-attn multi-seq + drafter caches with documented Phase A2b / Phase A4 markers.

Confidence in this dossier: **medium-high**. Every claim above has a `file:line` reference. The remaining unknowns are:
- Whether H4 (recurrent state outermost-axis stride) holds — needs direct probe of `MlxBuffer.strides()` at `n_seqs=4` alloc.
- Whether `dispatch_kv_cache_copy_seq_f32_dual` supports same-buffer cross-region writes for `fork_seq` (R5) — needs a kernel-level unit test.

Both are 1-hour probes that iter-2a's first day should run before any production code lands.
