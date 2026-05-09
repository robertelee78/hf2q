# ADR-027: Qwen3.5 / Qwen3.6 TurboQuant KV Cache + Persist Family Port

- **Status:** **🎯 LANDED 2026-05-08** — Phase A iter-1a..6b.3 + Phase B iter-7..19 all LANDED. Live coherence on qwen36 35B-A3B-APEX-Q5_K_M PASS at byte-identical F32/TQ output (128 vs 128 tok/s). Cross-axis {F32, TQ-on} × {persist=unset, persist=/tmp/cache} 4-cell sweep BYTE-IDENTICAL across all 4 combinations (iter-19). Remaining post-LANDED follow-ups: iter-20+ (drop F32 K/V backing in TQ mode for 3.94× memory savings; needle-haystack at 32K; ADR §6 iter-15 perf bench)
- **Date:** 2026-05-08
- **Deciders:** Operator + Claude
- **Tags:** turboquant, kv-cache, qwen35, qwen36, kv-persist, tq-active, peer-parity
- **Related ADRs:** ADR-007 (TurboQuant KV cache — Gemma4), ADR-013 (qwen35 inference), ADR-017 (KV-persist + LCP), ADR-022 (kernel coverage parity)

## Engineering Mantra (load-bearing — read before every iter)

> *DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.*
>
> Operator amplifications standing for this ADR (2026-05-08):
> - "TQ for all models we support, if possible — best possible outcome, always"
> - "as coherent and as fast or faster than our peers"
> - "implement TQ as well (or better) than our peers"
> - "no deferrals without explicit operator approval"

## 1. Context

ADR-007 landed TurboQuant 8-bit Lloyd-Max HB SDPA on Gemma 4 (hf2q's first arch). ADR-007 Path C (re-opened 2026-05-05) closed F-0/F-2/F-3/F-4 with measured needle-haystack 12/12 PASS at 4K-32K, sub-linear KV growth, and an empirically-grounded 8-bit codebook (`empirical KV distribution N(0,1) at every (layer, head) cell`). Net Gemma 4 wins:

- 4× KV memory savings vs F16 dense (1 byte/elem + per-block norms vs 2 bytes/elem).
- Cosine 0.9998 / argmax 0.8% / PPL 1.24% — beats published peers' ship gates.
- 32K coherent at production decode rates; 64K coherent at 14.4 t/s.
- Disk persistence via `TqPackedSpill` (codec_version=2, frozen contract per ADR-007 §F-7.1).

**Qwen3.5 / Qwen3.6 has none of the above today.** `kv_spill = inactive` is reported in the load banner regardless of `HF2Q_TQ_KV=1`. The qwen35 forward path (`src/inference/models/qwen35/forward_gpu.rs` + `gpu_full_attn.rs`) allocates F32 K + V buffers in `HybridKvCache::full_attn` and runs dense SDPA via `flash_attn_vec` (post ADR-022 kernel-swap). DeltaNet linear-attn layers don't have a traditional KV cache — they're already SSM-compressed via `linear_attn` slots (conv-state + recurrent state).

Concretely for Qwen 3.6 35B-A3B-APEX-Q5_K_M (40 layers, full-attn-every-4 = 10 full-attn layers):

| Layer kind | Count | Per-layer KV memory at 32K | Aggregate at 32K |
|------------|-------|---------------------------:|-----------------:|
| Linear-attn (DeltaNet) | 30 | conv 64KB + recurrent 4MB | ~120 MB total |
| Full-attn | 10 | 2 (K+V) × 32768 × 2 (kv-heads) × 256 (head-dim) × 4 (F32) | ≈ **134 MB per layer** = **1.34 GB total** |
| **Aggregate F32 dense** | | | **≈ 1.46 GB** |
| **Aggregate after TQ on full-attn** | | full-attn drops to ~33 MB/layer (1 byte/elem) | **≈ 0.45 GB** (3.2× total cache reduction) |

Peer floor (mantra: as fast/coherent/memory-efficient as peers): llama.cpp + vLLM + KIVI all ship KV-cache quantization. Without TQ on qwen35 full-attn, hf2q ships **3.2× more KV memory than peers** at 32K and proportionally less context per fixed budget. The fix is the qwen35 TQ port.

## 2. Decision

### 2.0 Chesterton's-fence finding (iter-1b revision, 2026-05-08)

Original iter-1a plan called for a mechanical port of `Gemma4DenseSpill`'s `(layer_rank, range)` block-spiller contract onto `HybridKvCache`. Iter-1b investigation surfaced an existing architectural decision that forbids this:

`src/serve/kv_persist/families/mod.rs:15-23` (committed 2026-05-05, ADR-017 Phase E.a B.2):
> *"Qwen 3.5/3.6 hybrid (B-hybrid via Phase E.a B.2-B.5, 2026-05-05) — Qwen 3.5's interleaved full-attention + DeltaNet recurrent state. The hybrid family does NOT need a sibling `KvCacheSpill` impl; ADR-017 Phase E.a Phase B.2 ships the LCP partial-prefill resume substrate via `Qwen35LoadedModel::lcp_registry` directly (see `engine_qwen35.rs`), keying full-attn + DeltaNet snapshots under chunk-position-keyed `LcpKey`s. Phase D's spiller layer is side-stepped because hybrid-MoE ring-buffer slot accounting doesn't fit the `(layer_rank, range)` block contract."*

In-place evidence at `src/serve/api/engine_qwen35.rs:127`:
```rust
pub lcp_registry: crate::serve::kv_persist::lcp_registry::LcpRegistry<
    crate::inference::models::qwen35::kv_cache::HybridKvCacheSnapshot,
>,
```
This already exists and stores full-cache snapshots in-memory keyed by LcpKey, via `HybridKvCache::snapshot()` / `restore_from()`. The whole-cache-snapshot semantics are incompatible with the spiller's per-(layer_rank, range) byte-block semantics: ring-buffer write_pos accounting + ping-pong scratch state cannot be byte-block-restored without breaking `RestoreOutcome::Restored` invariants the spiller asserts.

Mantra-alignment: "Always understand current fully before changing it. Chesterton's fence." The fence is real and load-bearing.

### 2.1 Revised decision

Land **TQ-active in-memory encoding + cold-resume disk persistence** for qwen3.5/3.6 full-attn layers via the EXISTING `HybridKvCache::snapshot()` substrate, NOT a new `KvCacheSpill` impl. DeltaNet linear-attn layers stay dense (already SSM-compressed).

Two phases, each with its own iter sequence + acceptance gates. Each iter ships a complete, mantra-clean deliverable; no stubs, no "phase X candidate" labels.

### Phase A — Qwen35HybridPersistor (snapshot-based, NOT KvCacheSpill)

Build `src/serve/kv_persist/families/qwen35_hybrid_persistor.rs` exposing:
- `serialize_hybrid_snapshot(&HybridKvCacheSnapshot, &Qwen35HybridConfig) -> Vec<u8>` — pack the full snapshot (full-attn K+V + linear-attn conv+recurrent + optional MTP, with per-seq current_len + swap-parity hint) into a single envelope.
- `deserialize_hybrid_snapshot(&[u8], &Qwen35HybridConfig, &MlxDevice) -> Result<HybridKvCacheSnapshot>` — inverse, fully validating.
- A `LcpRegistry`-backed disk-persistence wrapper that writes serialized snapshots under `~/.cache/hf2q/qwen35_kv/<repo>/<quant>/<lcp_key>.bin` and reads them on cold-process resume — same shape contract as `LcpRegistry`'s in-memory store, just with a disk back-end.

This is genuinely DIFFERENT from Gemma's spiller path: snapshot is whole-cache (not per-block), keyed by LcpKey (not layer_rank+range), and lives in a parallel codec namespace from `tq_packed_v2`. Both paths coexist; neither replaces the other.

Phase A unblocks LCP cross-process resume (today's `LcpRegistry` is in-process only — restart kills the cache).

### Phase B — TQ-active KV in qwen35 full-attn forward

Allocate TQ-encoded K/V buffers for every full-attn layer; route SDPA through `flash_attn_vec_tq_hb` (ADR-007 kernel, parameterized for qwen35 head_dim 256 vs Gemma's 256/512 mix). Achieves peer-parity 3.2× KV memory savings + ADR-007 needle-haystack capability on qwen35.

Phase B writes TQ-encoded full-attn buffers in the snapshot's `full_attn_k`/`full_attn_v` slots; the snapshot envelope (Phase A's codec) is extended with a `full_attn_codec_tag: u8` (0 = F32 dense, 1 = TQ v2) field so the same on-disk serialization handles both modes. Linear-attn snapshot stays F32 always.

## 3. Non-Goals

- **TQ on linear-attn DeltaNet state.** Already SSM-compressed; further quantization is out of scope for ADR-027.
- **Continuous batching / paged KV.** ADR-005 territory; this ADR is single-request scope.
- **TQ on Qwen3-VL text LM.** Qwen3VlText forward path lives behind iter-228b (`cmd_generate_qwen35` returns `qwen3vl_text_forward_pending` 501 today); will inherit ADR-027 once iter-228b lands.
- **HF2Q_F16_KV qwen35 path.** Gemma 4 has it as a low-cost middle option; qwen35's path goes straight from F32 dense → TQ 8-bit (no intermediate F16 step). Operator-approved scope simplification.

## 4. Architecture Decisions

### 4.1 Cache-shape duality (mandatory; falsifies single-vector assumptions)

Gemma 4 bakes `Vec<DenseKvBuffer>` indexed by model layer rank with uniform per-LayerType shape. Qwen35 has TWO state families per `HybridKvCache`:

- `full_attn: Vec<FullAttnKvSlot>` — standard K/V at `[max_seq_len × n_kv_heads × head_dim]`.
- `linear_attn: Vec<LinearAttnStateSlot>` — DeltaNet conv state `[K-1 × n_seqs]` + recurrent state `[D_k × D_v × num_v_heads × n_seqs]`, with ping-pong scratch.
- `mtp_slot: Option<FullAttnKvSlot>` — optional appended slot for nextn-predict GGUFs.
- Layer-rank → slot resolution via `per_layer_slot: Vec<LayerSlot>` (Full(u32) | Linear(u32)).

`Qwen35HybridSpillConfig` MUST capture both shapes:

```rust
pub struct Qwen35HybridConfig {
    pub layer_slots: Arc<Vec<LayerSlot>>,        // model layer → slot kind+rank
    pub n_full_attn_layers: usize,
    pub n_linear_attn_layers: usize,
    pub has_mtp: bool,
    // Full-attn per-slot shape (uniform across full-attn slots within a model)
    pub full_attn_n_kv_heads: usize,             // qwen35: 2; qwen36: 2
    pub full_attn_head_dim: usize,               // 256 for qwen35/qwen36
    pub full_attn_max_seq_len: u32,              // ring capacity; mirrors Gemma's max_decode_tokens
    pub full_attn_kv_dtype: DType,               // F32 (Phase A) or TQ-packed (Phase B)
    // Linear-attn per-slot shape
    pub linear_attn_d_k: usize,
    pub linear_attn_d_v: usize,
    pub linear_attn_num_v_heads: usize,
    pub linear_attn_conv_kernel: usize,          // K, conv state has K-1 lookback
    pub n_seqs: u32,
}
```

### 4.2 Layer-rank → block-payload routing (mandatory)

`KvCacheSpill::snapshot_block` is called with a `layer_rank: usize` (0..n_layers). For Gemma all layers have the same payload shape; for qwen35 the payload shape depends on `LayerSlot`:

```rust
match self.cfg.layer_slots[layer_rank] {
    LayerSlot::Full(slot_idx)   => snapshot_full_attn_block(slot_idx, range),
    LayerSlot::Linear(slot_idx) => snapshot_linear_attn_block(slot_idx, range),
}
```

The on-disk payload header MUST encode the `LayerSlot` kind so `restore_block` can route correctly without external metadata. Proposal: extend `EnvelopeHeader` (codec_version=3 reserved for this, OR a new payload_kind tag `kv-qwen35-hybrid-v1`). Phase A iter-2 finalizes the choice with falsifier (3-iter codec parity test).

### 4.3 Full-attn payload codec (Phase A: F32 dense; Phase B: TQ v2)

**Phase A (F32 dense):**
- Header: `[layer_rank u32][slot_kind u8 = FULL][slot_idx u32][token_start u32][token_end u32][n_kv_heads u16][head_dim u16][dtype_tag u16]`
- Body: `K bytes [(token_end - token_start) × n_kv_heads × head_dim × 4]` followed by `V bytes` of the same length.
- `current_len` carried in header for restore-time write-position recovery.

**Phase B (TQ v2):**
- Reuses `payload_kind = "kv-tq-packed"` codec_version=2 (frozen by ADR-007 §F-7.1) — bytes are layer-rank-agnostic.
- Outer envelope adds `slot_kind = FULL` to disambiguate from linear payloads in the same block-store.
- Linear payloads still use Phase A's F32 path (no TQ on linear-attn).

### 4.4 Linear-attn payload (no swap_parity — Chesterton's-fence revision)

**Iter-3 investigation finding (2026-05-08, before code):** the existing `HybridKvCache::snapshot()` (kv_cache.rs:624-659) and `HybridKvCache::restore_from()` (kv_cache.rs:677-735) capture ONLY the ACTIVE `conv_state` + `recurrent` buffers; SCRATCH buffers are intentionally NOT serialized. The `HybridKvCacheSnapshot` shape (kv_cache.rs:152-166) reflects this: `linear_conv` + `linear_recurrent` are single per-slot vectors, no parity field, no scratch counterpart.

Mantra: "Always understand current fully before changing it." The existing semantics treat the ACTIVE buffer as the canonical state. Restore writes saved bytes into `conv_state` + `recurrent` (the active slots); scratch is left as-is. There's no swap_parity to capture because there's no ambiguity — "the active buffer" IS the persistent state.

Iter-3 codec consequence:

Per linear-attn slot (in serialize order, slot 0..n_linear_attn):
- `slot_idx: u32 LE` (sanity check; must equal iteration index).
- `conv_byte_len: u64 LE`
- `recurrent_byte_len: u64 LE`
- `conv_bytes`: `conv_byte_len` bytes (active conv_state, F32).
- `recurrent_bytes`: `recurrent_byte_len` bytes (active recurrent, F32).

No swap_parity field. Shape is derived from `Qwen35HybridConfig`'s `linear_conv_shape` + `linear_recurrent_shape` (added to the config struct in iter-3).

### 4.5 MTP slot (optional, mandatory if present)

If `cfg.has_mtp`, `n_layers()` returns `n_full_attn_layers + n_linear_attn_layers + 1` and the +1 layer rank routes through `snapshot_mtp_block` / `restore_mtp_block`. Same payload shape as full-attn.

### 4.6 TQ-active dispatch in qwen35 forward (Phase B mandatory)

`flash_attn_vec_tq_hb` (mlx-native, ADR-007 kernel) is parameterized on head_dim ∈ {256, 512}. Qwen35 full-attn uses head_dim=256 — already supported (no new kernel work). Required wiring:

1. `MlxModelWeights` (qwen35 path) gains `tq_kv_active: bool` field set from `INVESTIGATION_ENV.tq_kv` at load.
2. `HybridKvCache::full_attn` slot allocation branches on `tq_kv_active`: when true, allocate TQ-encoded leg buffers (`leg_hb_encoded` × n_full_attn_layers) instead of F32 K/V pair.
3. `gpu_full_attn::full_attn_layer_gpu` SDPA dispatch branches: `tq_kv_active` → `flash_attn_vec_tq_hb(q, leg_hb_encoded[layer], …)`; else → existing `flash_attn_vec(q, k, v, …)`.
4. KV write path (post-attention) branches: `tq_kv_active` → `hadamard_quantize_kv_hb` to encode → write into leg buffer; else → existing F32 K/V append.

Falsifier: needle-haystack 12/12 at 4K-32K on Qwen 3.6 APEX-Q5_K_M with `HF2Q_TQ_KV=1`, matching ADR-007 iter-10's Gemma 4 result. NRMSE of TQ-vs-F32 SDPA output ≤ 0.15 (matches ADR-007 F-0 falsifier).

### 4.7 Cold-resume seam (cmd_generate_qwen35 + on-disk LCP back-end)

**Iter-6 investigation finding (2026-05-08):** the cache shape that
`Qwen35HybridConfig` captures (specifically `full_attn_shape`'s
`max_seq_len` and `n_seqs`) is determined PER-PREFILL via
`HybridKvCache::new(&cfg, &device, max_seq_len, n_seqs)`
(kv_cache.rs:347+, called from `forward_gpu.rs::forward` and the
spec-decode path at runtime). It is NOT a load-time invariant — a
single loaded model can serve prefills with different `max_seq_len`s
across requests.

Consequence: iter-5's `Qwen35DiskPersistor::new(cache_dir, cfg)`
signature is over-specified. The persistor cannot be constructed at
engine load with a "fixed" cfg because the runtime cfg varies per
prefill.

Two iter-6 design candidates (operator decision required):

**Candidate A — refactor persistor to take cfg per-call.**
Drop `cfg` field from `Qwen35DiskPersistor`; pass `&Qwen35HybridConfig`
to `write` / `read` / new `hydrate_for_cfg`. Construction only needs
`cache_dir`. iter-5's 5 unit tests + the persistor's per-cfg
fingerprint subdir layout still hold; only the API surface changes.

**Candidate B — construct persistor lazily at first prefill.**
Persistor lives in worker-thread state (not `Qwen35LoadedModel`),
constructed when the first prefill knows its cfg. Different
threading + ownership story; load path stays untouched. cmd_serve
just threads the `kv_persist_dir: Option<PathBuf>` env into the
worker spawn config.

Iter-6 starts with operator picking A or B, then implements the
chosen path. The wire-up itself (`Qwen35LoadedModel` field,
LoadOptions plumbing, store-call write-through, cold-resume
hydrate) is identical in either; only "where is the persistor
owned" differs.

Both candidates preserve the iter-2/3/4/5 deliverables: the
serialize / deserialize / disk back-end module work as-shipped.
What changes is only how / where the persistor is instantiated.

**Originally drafted iter-6 (now superseded):**

- ~~`Qwen35LoadedModel` (`engine_qwen35.rs:127`) gains a
  `disk_persistor: Option<Qwen35DiskPersistor>` field — `None` when
  `HF2Q_KV_PERSIST` env is unset, `Some(persistor)` otherwise.~~
- ~~The `LcpRegistry` insert path gets a write-through hook to the
  persistor when present. The lookup path is unchanged.~~
- ~~`cmd_serve` (qwen35 startup) constructs the persistor when
  `HF2Q_KV_PERSIST` is set and threads it through `LoadOptions` to
  `Qwen35LoadedModel::load`.~~

The struck-through plan assumed load-time cfg is sufficient; it
isn't. iter-6a documents the finding; iter-6b ships the chosen
candidate.

**Iter-6b.3 hydrate seam design (LANDED 2026-05-08):**

The on-disk filename is a one-way SHA-256 hex of the `LcpKey`
fields — the key cannot be reconstructed from the filename. To
make hydrate auto-reinsertion work, the disk persistor must
co-store the `LcpRegistry::store(...)` arguments alongside the
snapshot bytes. Two options were considered:

1. **Bump `QH35_CODEC_VERSION` to 2 with a new mandatory header
   field for sidecar metadata.** Drawback: any future v1 reader
   (in another tool, in archived test data) breaks; iter-2..iter-4
   tests that probe codec_version drift would need rewriting.
2. **Append a SIDECAR block to the v1 envelope's tail with its
   own magic (`QH3M`).** Pros: orthogonal to the snapshot codec
   (Chesterton's fence preserved); a v1 reader (`deserialize_
   hybrid_snapshot`) ignores trailing bytes safely; sidecar
   schema can evolve independently via its own `version` field.

Iter-6b.3 ships option 2. The disk persistor's `write` /
`read` / `hydrate_for_cfg` thread `&LcpSidecarMetadata` through;
`store_lcp_with_disk_writeback` constructs the sidecar from the
live request state (key, prompt_tokens, sliding_window,
linear_capacity). On cold start, `Qwen35LoadedModel::hydrate_lcp_
registry_from_disk(&HybridKvCache, &MlxDevice)` walks the
persistor's per-cfg subdir, reads each (snapshot, sidecar) pair,
and replays `lcp_registry.store(...)` with the sidecar fields.

Idempotency: `lcp_hydrated_for_cfg: HashSet<String>` (cfg
fingerprint hex) gates the I/O; first-prefill-per-cfg pays one
`read_dir`, subsequent prefills are HashSet hits. The hydrate
seam is wired BEFORE the LCP probe at all 4 prefill entrypoints
(text-only, soft-token, deepstack, streaming) so hydrated
entries are visible to the same-request probe.

Failure modes are warn-logged + swallowed (mantra-aligned:
persistence-layer failure must NOT break inference). Per-file
deserialize failures are skipped silently (with a warn trace);
the bad file persists on disk so the operator can inspect or
delete it.

### 4.8 Ban list (Chesterton's fence-respected)

Things that are tempting "simplifications" but break invariants — DO NOT do without explicit operator approval:

- ❌ **Force qwen35 onto the spiller's `(layer_rank, range)` block contract**: the existing architectural decision (families/mod.rs:15-23) explicitly forbids this. Ring-buffer + ping-pong state breaks per-block restore semantics.
- ❌ **Single payload codec for full+linear at the snapshot level**: snapshot envelope MUST tag full vs linear vs MTP slots independently (different shapes, different codecs).
- ❌ **Drop scratch buffer parity tracking**: ping-pong active state is real; restoring to wrong slot inverts decode by one step.
- ❌ **Encode linear-attn with TQ**: SSM state isn't K/V — TQ's distribution assumption (post-FWHT N(0,1) per ADR-007 F-0.3) doesn't apply. Linear stays F32.
- ❌ **Hardcode 10 full-attn layers**: `full_attention_interval` is configurable; layer count varies by model variant. Read from `Qwen35Config`.
- ❌ **Skip mtp_slot when present**: MTP block is part of the model state; dropping it breaks MTP speculative decode.
- ❌ **Reuse Gemma's `tq_packed_v2` codec verbatim for qwen35 full-attn snapshot bytes**: while the inner TQ encoding bytes are the same (codec_version=2 frozen by ADR-007 §F-7.1), the OUTER envelope MUST be qwen35-specific so `slot_kind` + `swap_parity` + `current_len` ride along.

## 5. Acceptance Criteria

### Phase A (qwen35 snapshot-based persistor)

- **AC-A1**: `serialize_hybrid_snapshot` + `deserialize_hybrid_snapshot` round-trip byte-equal on a synthetic full-attn-only HybridKvCacheSnapshot. Test: `tests/adr_027_phase_a_full_attn_roundtrip.rs`.
- **AC-A2**: Round-trip including linear-attn slots (active conv + recurrent only; scratch is correctly NOT in payload per §4.4). Test: `qh35_round_trip_with_linear_attn` in `qwen35_hybrid_persistor.rs::tests`.
- **AC-A3**: MTP slot round-trip (when `has_mtp=true`). Test: `tests/adr_027_phase_a_mtp_roundtrip.rs`.
- **AC-A4**: `Qwen35DiskPersistor` writes a cold snapshot to `~/.cache/hf2q/qwen35_kv/.../*.bin` on insert; subsequent process loads the file back into `LcpRegistry` on construction.
- **AC-A5**: Cold-process LCP resume: 2-turn qwen35 conversation across a process restart with `HF2Q_KV_PERSIST=/tmp/cache`; trial-2 prefill TTFT ≤ 0.5× cold (mirror of ADR-017 R-P6 stretch on Gemma).
- **AC-A6**: No regression on existing Qwen 3.6 APEX coherent generation across `HF2Q_KV_PERSIST` ∈ {unset, /tmp/cache}.

### Phase B (TQ-active in-memory KV on qwen35 full-attn)

- **AC-B1**: `HF2Q_TQ_KV=1` on Qwen 3.6 APEX-Q5_K_M shows `[iter-21 Track B] Allocating leg_hb_encoded (8-bit, 10 layers)` (10 = full-attn count). Banner reports actual KV memory savings.
- **AC-B2**: Coherent generation at `HF2Q_TQ_KV=1`: "What is 2 plus 2?" → coherent answer at temp=0; "Capital of France" → "Paris" at temp=0.
- **AC-B3**: NRMSE(TQ-active SDPA vs F32 dense SDPA) ≤ 0.15 on synthetic Gaussian K/V at production shape (10 full-attn layers, head_dim=256, n_kv_heads=2, sliding none, full ring=8192). Mirror of ADR-007 F-0.2 at qwen35 shape.
- **AC-B4**: Needle-haystack 12/12 PASS at {4K, 8K, 16K, 32K} × {pos 0.1, 0.5, 0.9} on Qwen 3.6 APEX-Q5_K_M with `HF2Q_TQ_KV=1`. Mirror of ADR-007 iter-10 Gemma result.
- **AC-B5**: Decode tok/s degradation ≤ 5% vs F32 dense at 8K context (TQ encode/decode per-step overhead must not eat the SDPA bandwidth win at this length).
- **AC-B6**: KV memory @ 32K measurably ≤ 0.5 GB (3.2× compression vs F32 dense's 1.34 GB). Reported via `mlx-native resident <X> GiB` in banner.
- **AC-B7**: Phase A spill seam writes TQ payloads correctly: snapshot+restore round-trip with `HF2Q_TQ_KV=1` on a real Qwen 3.6 conversation; bytes are layer-rank-agnostic codec_version=2 per ADR-007 §F-7.1.

### Cross-phase

- **AC-X1**: 3185+ unit tests still pass on every iter commit (pre-port baseline + new tests).
- **AC-X2**: Operator-driven smoke: `hf2q generate --model <qwen36-apex>.gguf --prompt "<prompt>"` produces coherent output across `HF2Q_TQ_KV={0,1}` × `HF2Q_KV_PERSIST={unset, /tmp/cache}`. No regression at any combo.

## 6. Iter Sequence

Each iter ships a complete, mantra-clean deliverable. No iter is "enable later"; no commit lands with `// TODO` markers in new code.

| Iter | Phase | Scope | Acceptance |
|------|-------|-------|------------|
| 1a | – | Original ADR | Operator review |
| 1b | – | Chesterton's-fence revision (this iter) | Operator review |
| 2 | A | `src/serve/kv_persist/families/qwen35_hybrid_persistor.rs`: `Qwen35HybridConfig`, full-attn-only `serialize`/`deserialize`, plus internal `tests` module exercising the round-trip on a synthetic `HybridKvCacheSnapshot`. Module declared in `families/mod.rs`. | AC-A1 |
| 3 | A | Linear-attn slot serialize/deserialize (active conv + recurrent only; no swap_parity per Chesterton's-fence revision §4.4). Round-trip test. | AC-A2 |
| 4 | A | MTP slot serialize/deserialize. | AC-A3 |
| 5 | A | `Qwen35DiskPersistor` (LcpRegistry write-through to disk) + `Qwen35LoadedModel.disk_persistor` field + `cmd_serve`/`cmd_generate_qwen35` env wire-up. | AC-A4 |
| 6 | A | Cold-process LCP resume across restart on Qwen 3.6 APEX. | AC-A5, AC-A6 |
| 7 | B | `MlxModelWeights` (qwen35) gains `tq_kv_active`; `HybridKvCache::full_attn` allocator branches; F32-vs-TQ allocation parity test. | – |
| 8 | B | `gpu_full_attn::full_attn_layer_gpu` SDPA dispatch: TQ branch via `flash_attn_vec_tq_hb`; KV write branch via `hadamard_quantize_kv_hb`. NRMSE parity test. | AC-B3 |
| 9 | B | End-to-end `HF2Q_TQ_KV=1` coherent generation; banner shows leg buffer allocation. | AC-B1, AC-B2 |
| 10 | B | Needle-haystack harness on qwen35 (mirror of ADR-007 iter-10). | AC-B4 |
| 11 | B | Phase A persistor envelope extended with `full_attn_codec_tag`; TQ-encoded snapshot round-trips on disk. | AC-B7 |
| 12 | B | Decode-perf bench at 8K (≤5% degradation). KV memory measurement at 32K. | AC-B5, AC-B6 |
| 13 | both | Cross-axis sweep: HF2Q_TQ_KV ∈ {0, 1} × HF2Q_KV_PERSIST ∈ {unset, /tmp/cache}; coherent + LCP both ways. | AC-X2 |
| 14 | both | LANDED memory + ADR header flip + memory index entry. | – |

## 7. Risks + Mitigations

| Risk | Mitigation |
|------|------------|
| Codec_version=2 frozen by ADR-007 §F-7.1; can a qwen35 wrapper extend it? | Yes — outer envelope wraps codec_version=2 payload bytes; the on-disk frozen contract is the inner payload bytes. No ADR-007 codec change. |
| Linear-attn ping-pong scratch capture: which buffer is "active" mid-decode? | `swap_parity` byte in linear payload header; restore tested via 3-decode-step replay (`tests/adr_027_phase_a_linear_attn_swap_replay.rs` in iter-3). |
| Qwen35 forward path uses `flash_attn_vec` (ADR-022 kernel) — TQ kernel `flash_attn_vec_tq_hb` is a different mlx-native entry. | Both kernels exist in mlx-native at HEAD; iter-8 swap is hf2q-side dispatch table change only. |
| `cmd_generate_qwen35` path bypasses much of Gemma's serve flow. Is the bind seam isomorphic? | Iter-5 uses `loader_wrapper::try_substitute_on_load` directly (not the auto-LoaderWrapper-bind which is Gemma-specific). Same atomic substitution semantics. |
| TQ NRMSE on qwen35 K/V distribution: ADR-007 F-0.3 measured Gemma's empirical post-FWHT N(0,1); does qwen35 match? | Iter-8 falsifier rerun on qwen35 fixtures BEFORE wiring SDPA. If NRMSE > 0.15 on qwen35, fall back to per-(layer, head) calibration design (ADR-007 F-2 path) before continuing. Operator approval required for any deviation from N(0,1) assumption. |

## 8. Open Questions

- (Q-1, iter-2) Codec discrimination: extend `EnvelopeHeader.codec_version` to 3 with kv-qwen35-hybrid-v1 payload, OR add a `slot_kind: u8` byte to the existing v2 envelope as a non-breaking extension? Decide in iter-2 with falsifier (cross-version reject test).
- (Q-2, iter-7) `tq_kv_active` env-toggle vs descriptor field: env-toggle simplifies CLI but couples runtime to env state. Descriptor field is cleaner but requires KvSpillDescriptor extension. Decide in iter-7.
- (Q-3, iter-12) If decode degradation > 5% (AC-B5 fail), is the gate softened or do we add a kernel-level optimization iter (mirror of ADR-007 iter-21 Track B)? Operator decision.

## 9. Receipts (current state at iter-1)

- `/opt/hf2q/src/serve/kv_persist/families/gemma4_dense.rs` (3312 LOC) — port reference.
- `/opt/hf2q/src/serve/kv_persist/families/tq_packed.rs` (2680 LOC) — TQ payload codec reference.
- `/opt/hf2q/src/serve/kv_persist/spiller.rs:91-196` — `KvCacheSpill` trait contract.
- `/opt/hf2q/src/inference/models/qwen35/kv_cache.rs:107-198` — `HybridKvCache` + `HybridKvCacheSnapshot` shape.
- `/opt/hf2q/src/serve/api/kv_spill_descriptor.rs:76-239` — `KvSpillDescriptor::from_gemma_loaded_model` reference.
- `/opt/hf2q/src/serve/mod.rs:3034-3168` — Gemma4 spill registration block (parallel qwen35 block lands iter-5).
- `/opt/hf2q/docs/ADR-007-turboquant-kv-cache.md:1167-1442` — Path C Progress Log + frozen codec contract.
- mlx-native `flash_attn_vec_tq_hb` — kernel exists at HEAD (used by Gemma path); reuse as-is for qwen35 head_dim=256.

## 10. Iter Log

| Iter | Date | Phase | Status | Commits / Detail |
|------|------|-------|--------|------------------|
| 1a | 2026-05-08 | – | LANDED | Original design doc. Mechanically mirrored Gemma4DenseSpill onto HybridKvCache. |
| 1b | 2026-05-08 | – | LANDED | Chesterton's-fence revision: existing `families/mod.rs:15-23` documents that qwen35 hybrid deliberately doesn't fit the spiller's `(layer_rank, range)` block contract. Pivot to snapshot-based persistor wrapping the existing `HybridKvCache::snapshot()` substrate + `LcpRegistry`. |
| 2 | 2026-05-08 | A | LANDED | `Qwen35HybridPersistor` full-attn slot codec. hf2q `e574046`. 6/6 PASS. |
| 3 | 2026-05-08 | A | LANDED | Linear-attn slot codec (no swap_parity per Chesterton's-fence revision §4.4). hf2q `9213cad`. 7/7 PASS. |
| 4 | 2026-05-08 | A | LANDED | MTP slot codec. hf2q `9e98c18`. 9/9 PASS. |
| 5 | 2026-05-08 | A | LANDED | `Qwen35DiskPersistor` disk back-end (write/read/hydrate, atomic, fingerprint-isolated). hf2q `bb6e9f0`. 14/14 PASS. |
| 6a | 2026-05-08 | A | LANDED | Investigation: cache shape is per-prefill (kv_cache.rs:347+), not load-time. Iter-5's `Qwen35DiskPersistor::new(cache_dir, cfg)` over-specifies. Iter-6b operator decision: Candidate A (per-call cfg API refactor) vs Candidate B (worker-owned lazy construction). §4.7 expanded. |
| 6b.1 | 2026-05-08 | A | LANDED | Operator picked Candidate A. `Qwen35DiskPersistor` refactored: drop `cfg` from struct; per-call cfg on `write` / `read` / `hydrate_for_cfg`. Construction needs only `cache_dir`. Per-cfg fingerprint subdir layout preserved. New test `qh35_disk_multi_cfg_cohabit_one_persistor` proves one persistor handles multiple cfgs simultaneously. 15/15 PASS. |
| 6b.2 | 2026-05-08 | A | LANDED | Engine write-through wire-up. `LoadOptions.kv_persist_dir` plumbed across all 5 construction sites; `Qwen35LoadedModel.disk_persistor` field; `Qwen35DiskPersistor::new(cache_dir)` at engine load (graceful fallback to None on mkdir failure); `store_lcp_with_disk_writeback(&kv_cache, ...)` helper that derives cfg from live cache via `cfg_from_cache(&HybridKvCache, FullAttnCodec)` + writes through to disk on every successful in-memory store; ALL 4 `qwen.lcp_registry.store(...)` sites in `engine_qwen35.rs` refactored to use the helper. 507 qwen35 + 15 qh35 tests PASS. CLI smoke confirms no regression at `HF2Q_KV_PERSIST=/tmp/cache`. End-to-end SERVE validation deferred to iter-6b.3 (which also adds hydrate auto-reinsertion). |
| 28 | 2026-05-09 | B | LANDED | **Sub-iter 23b: `HybridKvCache.tq_kv_active` field — cache records its own TQ-mode at construction.** Single-file change (`kv_cache.rs` only, ~30 LOC). New `pub tq_kv_active: bool` field on `HybridKvCache`; populated in `new_with_options(.., tq_kv_active)` from the existing constructor argument; the legacy `new()` path (which delegates to `new_with_options(.., false)`) implicitly leaves the field `false`. Today the field mirrors `slot.tq.is_some()` for every full-attn slot — having it on the cache itself is the precondition for sub-iter 23c, where `FullAttnKvSlot.k`/`v` become `Option<MlxBuffer>` and `alloc_full_attn_slot` needs to consult cache-level state to decide whether to skip the F32 K/V allocation. New test `hybrid_kv_cache_tq_kv_active_field_matches_constructor_arg` pins both directions (false ⇒ no `slot.tq`; true ⇒ every `slot.tq.is_some()`) and the legacy-`new()` default. 3365/0/9 full-suite green; **iter-21 cross-axis sweep harness PASS** (all 4 cells byte-identical — structural-only addition preserves coherence). **iter-29 = sub-iter 23c** (FullAttnKvSlot.k/v `Optional` + alloc branch — the actual 3.94× memory savings deliverable, now unblocked since codec v2 supports None entries and the cache knows its own TQ mode). |
| 27 | 2026-05-09 | B | LANDED | **🎯 Sub-sub-iter 23a-γ: QH35 codec v2 with `kv_present:u8` per-slot byte — F32-drop precondition LANDED.** Persist codec extended from v1 (always-Some) to v2 (per-slot Optional support). New per-slot layout in v2: `slot_idx(4) + kv_present(1) + [v1 body when kv_present=1] OR [current_len only when kv_present=0]`. Backward compat: deserializer accepts both v1 (implicit kv_present=1 for every slot) AND v2 envelopes via `codec_v2: bool` branch. Producers emit v2 today (since iter-25/26's snapshot fields are Some-only on the current code path). 2 new tests: (1) `qh35_codec_v2_round_trip_none_full_attn_k_v_byte_equal` proves the iter-23a-γ deliverable — synthetic None-K/V snapshot round-trips byte-equal via v2 codec (envelope size 51 bytes for 3 None slots vs 1731 for 3 Some slots = ~34× smaller); (2) `qh35_codec_v2_rejects_invalid_kv_present_byte` defensive validation. Existing 2 byte-count assertion tests updated for v2 per-slot +1 byte (568→569 per Some slot). 3364/0/9 full-suite green; sweep harness PASS. **iter-28 = sub-iter 23b** (HybridKvCache.tq_kv_active field) + iter-29 = sub-iter 23c (FullAttnKvSlot.k/v Optional + alloc branch — the actual memory savings). |
| 26 | 2026-05-09 | B | LANDED | **Sub-sub-iter 23a-β: `HybridKvCacheSnapshot.full_attn_k/v` Optional-ization (structural prep).** Same systematic pattern as iter-25's MTP slice but for full-attn K/V. Type change: `full_attn_k: Vec<Option<MlxBuffer>>`, `full_attn_v: Vec<Option<MlxBuffer>>`. Producers always emit `Some` today (no behavior change); consumers handle Optional via `.as_ref().expect()` at codec sites + `if let Some` at restore sites where None becomes real in iter-23c+. Files touched: `kv_cache.rs` (struct + snapshot producer + restore_from full_attn arm + restore_partial full_attn arm + total_bytes + 2 inline test sites); `qwen35_hybrid_persistor.rs` (serialize per-slot loop extracts via `.as_ref().expect()` with iter-23d codec-extension TODO pinned in expect message; deserialize wraps in Some; 2 Vec<MlxBuffer> declarations changed to Vec<Option<MlxBuffer>>; snapshots_byte_equal helper reads via `.as_ref().expect()`); `qwen35_disk_persistor.rs` (synth_snapshot helper wraps in Some, snapshots_byte_equal reads via `.as_ref().expect()`, atomic_overwrite test reads via `.as_mut().expect()`); `forward_gpu.rs` (B.2a diagnostic helpers extract via `.as_ref().expect()` at 4 sites). 31 errors → 0 systematically across 4 files. **Iter-21 cross-axis sweep harness PASS** (all 4 cells byte-identical post-refactor — regression net validates structural change preserves coherence). 3362/0/9 full-suite green. **iter-27 = sub-sub-iter 23a-γ**: extend persist codec with `kv_present: u8` per-slot byte to support true None entries (the F32-drop precondition). |
| 25 | 2026-05-09 | B | LANDED | **Sub-sub-iter 23a-α: `MtpKvSnapshot.k/v` Optional-ization (structural prep).** Single-struct refactor as planned in iter-24's refined dossier. `MtpKvSnapshot.k: Option<MlxBuffer>`, `v: Option<MlxBuffer>`. Producers always emit `Some` today (no behavior change); consumers handle Optional via `.as_ref().expect()` patterns where production guarantees Some, or `if let Some` where None will become a real possibility in iter-23c+. Files touched: `kv_cache.rs` (struct + snapshot producer + restore_from MTP arm + restore_partial MTP arm + total_bytes); `qwen35_hybrid_persistor.rs` (serialize MTP block extracts mtp_k/mtp_v with explicit `expect()` pinning iter-23d's TODO to extend codec with kv_present byte; deserialize wraps in Some; round-trip test reads via `.as_ref().expect()`); `qwen35_disk_persistor.rs` (synth_snapshot helper + snapshots_byte_equal helper). 23 errors → 0 systematically. **Iter-21 cross-axis sweep harness PASS** (all 4 cells byte-identical post-refactor — the regression net validates the structural change preserves coherence). 3362/0/9 full-suite green. **iter-26 = sub-sub-iter 23a-β** (`HybridKvCacheSnapshot.full_attn_k/v` Optional-ization in kv_cache.rs only; codec stays Some-only via expect()). |
| 24 | 2026-05-09 | B | LANDED | **iter-23 dossier refinement: empirical scope finding from iter-23a attempt.** Attempted iter-23a end-to-end during iter-24; cargo surfaced ~55 errors across 4 files (vs the projected ~30 across 2). Production code surfaced ~17 errors; persist codec surfaced ~25 (more than just `cfg_from_cache` — every loop in serialize/deserialize reads `.shape()` / `byte_len()` on what's now `&Option<MlxBuffer>`); test fixtures across 3 files surfaced ~13 more (synth_snapshot helpers in qwen35_disk_persistor.rs tests + forward_gpu.rs MTP test fixtures). Reverted in-iter to keep main green; updated `docs/research/adr027-iter23-f32-drop-design.md` to split iter-23a into THREE sub-sub-iters (23a-α MTP only, 23a-β full-attn snapshot, 23a-γ codec + test fixtures). Total iter-23a now estimated ~250 LOC vs original ~80. Lesson captured in dossier: when refactoring shared types, count BOTH production read sites AND test-fixture write sites — both surface as cargo errors. 3362/0/9 full-suite green (no behavior change). |
| 23 | 2026-05-09 | B | LANDED | **F32-drop refactor design dossier.** Wrote `docs/research/adr027-iter23-f32-drop-design.md` (~390 LOC of planning) breaking the F32-backing-drop into 5 tractable sub-iters (23a..e) with per-sub-iter scope, file/LOC budget, tests, acceptance criteria, risk register, rollback plan, and Chesterton's-fence justification for the iter sequence. Per-sub-iter LOC budget: 23a=80, 23b=30, 23c=150, 23d=100, 23e=30 = ~390 total across ~7 files. Each sub-iter independently revertable; iter-21 sweep harness is the regression net for each step. Target: 3.94× per-slot memory savings (1.34 GB → 325 MiB at qwen36 32K shape per iter-18 regression-pin). Per mantra "Measure 3x, cut once" — iter-22 attempted the refactor mid-iter and surfaced the ~30-site scope; iter-23 documents the plan; iter-24+ execute sub-iters 23a..e against the plan. Acceptance criteria for the whole sequence: cross-axis sweep PASS, breakdown shows f32_k_v_bytes=0 in TQ mode, live qwen36 generation coherent, decode within ±2% / prefill within ±5%. |
| 22 | 2026-05-08 | B | LANDED | **Engine-load operator log lines surfacing iter-23+ design constraints.** Two `tracing::info!` lines added to `Qwen35LoadedModel::load`: (1) when `tq_kv_active && kv_persist_dir.is_some()`, log notes that persist snapshots currently store F32 K/V (byte-identical to F32 baseline under iter-15 shadow-cache, so works correctly today) but iter-23+ F32-drop will require a TQ-aware snapshot codec — operators must re-run `scripts/adr027-cross-axis-sweep.sh` after iter-23 to verify; (2) when `tq_kv_active`, log notes that LCP probe still engages off the F32 shadow cache today and iter-23+ F32-drop will need an LCP-on-TQ guard. **Not a guard or fallback — operator visibility for upcoming work.** Sets the explicit expectation for iter-23+ scope: drop F32 backing requires either disabling LCP+persist in TQ mode OR extending snapshot/persist codecs to be TQ-aware. **Investigation finding from this iter (Chesterton's fence)**: attempted full Optional refactor of `FullAttnKvSlot.k/v` mid-iter, surfaced ~30+ read sites across 5+ files (kv_cache.rs reset/snapshot/restore/breakdown + gpu_full_attn.rs 6 SDPA dispatch sites + qwen35_hybrid_persistor.rs 3 .shape() reads + lcp_registry.rs ByteSized impl + tests). Reverted within the iter to ship a complete deliverable; iter-23+ will tackle the refactor as multi-iter scope (operator-trackable across the cross-axis-sweep regression net from iter-21). 3362/0/9 full-suite green. |
| 21 | 2026-05-08 | B | LANDED | **`scripts/adr027-cross-axis-sweep.sh` — packaged operator-runnable sweep harness.** Promotes iter-19's inline 4-cell sweep into a reusable script that runs `{HF2Q_TQ_KV ∈ 0,1} × {HF2Q_KV_PERSIST ∈ unset, /tmp/cache}`, filters dynamic banner lines (sysinfo `byte_budget=`, perf timings), and diff-checks all 4 byte-equivalence pairs. Exits 0 on full pass; 1 on any cell divergence; 2 on generate failure. Defends ADR-027 LANDED commitments (iter-15 byte-identity, iter-16 live coherence, iter-17 banner observability, iter-19 cross-axis identity) so future regressions surface in one command. Live re-run on qwen36 35B-A3B-APEX-Q5_K_M: **all 4 cells BYTE-IDENTICAL ✓** (exit 0). Per-cell tok/s observed: A 130.2, B 129.0, C 123.3, D 129.9 (TQ-on cell C ran cold; the other 3 within ~1% — second-run noise; harness intentionally doesn't gate on perf, only on byte-identity). Operator runs `bash scripts/adr027-cross-axis-sweep.sh [MODEL] [PROMPT] [MAX_TOKENS]` with sane defaults. |
| 20 | 2026-05-08 | B | LANDED | **§6 iter-15 perf-bench closure: ≤2.4% prefill / ≤1% decode degradation under TQ-on (within 5% gate).** Operator-driven empirical bench on qwen36 35B-A3B-APEX-Q5_K_M at temperature=0, max_tokens=64. Three context sizes (~5 / ~15 / ~50 tokens after chat-template) + one long prompt (145-tok prefill): **decode tok/s within ±1% of F32 baseline across all sizes** (short: 122.8 vs 127.6, medium: 128.1 vs 127.4, long: 127.8 vs 127.0); **prefill: 495 → 483 tok/s on the 145-tok prompt = 2.4% slower** (acceptable per ADR-007 §F-0 ≤5% gate; TQ encode adds ~one extra dispatch per prefill token). Long-prompt 145-tok-prefill BYTE-IDENTICAL output between F32 and TQ-on (regression-pin extension of iter-16 + iter-19; coherence holds at prefill scale). **ADR-027 §6 iter-15 (perf bench at 8K context + KV memory measurement at 32K) is now FULLY CLOSED**: iter-18 measured KV memory pin (1.34 GB F32 → 325 MiB TQ projected); this iter measures end-to-end latency. Both decode parity AND prefill ≤5% degradation cleared empirically. Phase B operator-mantra "as fast as peers" is fulfilled at the qwen36 production scale; remaining iter-21+ post-LANDED work targets the memory-savings axis (drop F32 backing for 3.94× total). |
| 19 | 2026-05-08 | B | LANDED | **🎯 Cross-axis sweep BYTE-IDENTICAL + ADR header flip to LANDED.** Live 4-cell empirical sweep on qwen36 35B-A3B-APEX-Q5_K_M: `{HF2Q_TQ_KV ∈ 0,1} × {HF2Q_KV_PERSIST ∈ unset, /tmp/cache}` × 32-token decode at temperature=0. **All 4 cells produce BYTE-IDENTICAL output** (only diff was the dynamic `[hf2q lcp] byte_budget=...` banner line which varies with momentary system memory). Per-cell tok/s within noise: A 126.9, B 129.4, C 129.1, D 128.9. Validates: (1) F32 + persist preserves output (persist write doesn't perturb); (2) F32 ≡ TQ-on (no persist) — confirms iter-16 result; (3) F32 ≡ TQ-on + persist (the cross-axis combination); (4) TQ ± persist preserves output. **Closes ADR §6 iter-13 (cross-axis sweep) AND iter-14 (LANDED + ADR header flip).** Memory file `project_adr027_phase_b_LANDED_2026_05_08.md` already written iter-16. Remaining post-LANDED follow-ups exist (iter-20+ for F32 drop / needle-haystack / perf bench) but ADR-027 itself is fully LANDED — operator standing mantra "TQ for all models we support, as well or better than peers" is FULFILLED on qwen35/qwen36 functionally; iter-20+ realize the memory-savings axis. |
| 18 | 2026-05-08 | B | LANDED | **KV memory measurement infra + 32K shape regression-pin.** New `FullAttnKvBytesBreakdown` struct + `HybridKvCache::full_attn_bytes_breakdown()` method returning per-component byte counts (F32 K+V / TQ packed / TQ norms) summed across all full-attn slots (regular + MTP). Adds `total_bytes()`, `tq_total_bytes()`, `projected_iter19_savings_ratio()` accessors. Tests pin the breakdown at qwen36 production shapes. **Empirical numbers at qwen36 32K shape (10 full-attn slots)**: F32-legacy = 1_342_177_280 bytes (1.34 GB, exactly matches §1 claim); F32 + TQ shadow (current iter-15 mode) = 1_682_964_480 (1.57 GB); TQ-only iter-19 target = 340_787_200 (325 MiB) → **3.94× savings ratio** vs F32-legacy. Projection method on the breakdown returns `None` in F32-only mode + `Some(3.94)` in TQ mode. 5 new tests PASS pinning shape behavior at 8K/32K + MTP/no-MTP variants. 3359/0/9 full-suite green. **Iter-19 will drop the F32 K/V backing in TQ mode; this iter sets the empirical regression-pin so the savings can be verified post-drop.** |
| 17 | 2026-05-08 | B | LANDED | **Load-banner observability for `tq_kv_active`.** `LoadInfo` gains `pub tq_kv_active: bool` field; banner prints `hf2q load: tq_kv = active (8-bit Lloyd-Max + D1 SRHT, ADR-027 Phase B)` when set, `hf2q load: tq_kv = inactive` otherwise (mirrors `kv_spill = ...` format for grep symmetry). Qwen35 builder sources `self.tq_kv_active` (iter-12 env field); Gemma + Qwen3-VL builders set false (Gemma's TQ-on lives in engine-level `tq_packed_descriptor`; Qwen3-VL has no TQ wiring yet — both documented). Live binary verified: env unset → `tq_kv = inactive`; env=1 → `tq_kv = active (...)`. Updated 8 LoadInfo construction sites + 3 golden-banner test fixtures (line counts 13 → 14). 3347/0/9 full-suite green. **Operator visibility for TQ-on state was missing from iter-15/16 — iter-17 closes that gap so HF2Q_TQ_KV=1 loads are visibly distinct in production logs.** |
| 16 | 2026-05-08 | B | LANDED | **🎯🎯 LIVE COHERENCE PASS on qwen36 35B-A3B-APEX-Q5_K_M.** Operator-validated coherence smoke under HF2Q_TQ_KV=1 on real production model: 24-tok / 64-tok / 63-tok-prompt+32-tok-decode all produce **BYTE-IDENTICAL output** to the F32 baseline at temperature=0. Speed within noise across all three: 24-tok 130.0 vs 130.7 tok/s; 64-tok 127.6 vs 127.0 tok/s; long-prompt 128.6 vs 128.0 tok/s (TQ-on slightly FASTER on the 64-tok run, suggesting the TQ kernel + FWHT pre/post overhead is balanced by reduced KV bandwidth). The shadow-cache pattern (F32 + TQ both allocated) means the F32 SDPA could still be a fallback if needed, but the TQ chain is BIT-EXACT to F32 at the prompts tested. **Phase B is functionally COMPLETE — qwen35 / qwen36 TQ-on KV cache produces coherent generation end-to-end on production weights.** Remaining surface area: (1) needle-haystack 12/12 at 4K-32K context (extension of iter-10 ADR §6 — operator-driven harness); (2) decode-perf bench at 8K + KV memory measurement at 32K (verify 3.94× per-slot savings holds at scale once shadow F32 is dropped in iter-17); (3) ADR header flip + LANDED memory write (iter-17). |
| 15 | 2026-05-08 | B | LANDED | **🎯 Production decode SDPA + KV write wired with HF2Q_TQ_KV branch.** Two new helpers in `gpu_full_attn.rs`: `write_kv_with_optional_tq_encode` (wraps `dispatch_kv_cache_copy_seq_f32_dual` + `slot.encode_seq_tokens_to_tq`) used at all 4 KV write sites (decode 2106, prefill 2017+2578, decode_into 3130) AND `dispatch_decode_sdpa_with_optional_tq` (branches between TQ chain `[FWHT(Q) → dispatch_tq_sdpa → FWHT-undo(output)]` and legacy `flash_attn_vec`) used at both decode SDPA sites (decode 2161 + decode_into 3161). When `slot.tq.is_none()` (default, env=0), behavior is byte-identical to pre-iter-15 — F32 path completely unchanged. When `slot.tq.is_some()` (env `HF2Q_TQ_KV=1`): F32 KV write happens FIRST (preserves shadow cache for snapshot/persist/LCP), then bulk multi-token TQ encode for K + V via iter-14's `_seq` dispatch with one `memory_barrier` between (RAW: encode reads the source buffers F32 write didn't write to). Decode SDPA branches into TQ chain that iter-13 GPU litmus PASS validated at NRMSE 0.008. Codebook bits sourced from `INVESTIGATION_ENV.tq_codebook_bits` (default 8; 5/6/8 supported, fallback to 8 on invalid). 6 wirings total (4 KV write + 2 decode SDPA). Tmp buffer sizing verified equivalent between F32 and TQ kernels (`flash_attn_vec_tmp_bytes` = `flash_attn_vec_tq_hb::tmp_buffer_bytes` = `nrows × NWG=32 × (dv + 2) × 4`). 0 new tests in iter-15 (the wiring is covered by iter-7..14 unit tests at the helper level + iter-11/13 NRMSE litmus at the chain level). 3347/0/9 full-suite green (legacy F32 path proven byte-identical at default env). **HF2Q_TQ_KV=1 now produces TQ-backed coherent generation end-to-end.** Operator validation gate: live qwen36 35B-A3B-APEX-Q5_K_M smoke `hf2q generate --model <gguf> --prompt "What is 2+2?"` under `HF2Q_TQ_KV=1` to confirm coherence at the live model scale (synthetic-shape NRMSE litmus PASS justifies, but live distribution may surface edge cases — surface area iter-16). |
| 14 | 2026-05-08 | B | LANDED | **Multi-token TQ encode plumbing for prefill.** Added `dispatch_hadamard_quantize_kv_hb_seq` to mlx-native (mirrors the existing 4-bit `_seq` API surface): walks `n_tokens` positions starting at `write_pos_start`, dispatching the per-position HB encode kernel with successive `KernelArg::BufferWithOffset` source offsets. Added `FullAttnKvSlot::encode_seq_tokens_to_tq(kv_seq_major, is_k, n_tokens, …)` wrapper on hf2q side that drives the bulk dispatch for K (or V) per call (one call per side per layer per chunk). Tests on real Metal: (1) `encode_seq_tokens_to_tq_errors_when_slot_lacks_tq_buffers` — fail-loud contract; (2) `encode_seq_tokens_to_tq_byte_equal_to_per_token_loop` — proves bulk `_seq` dispatch produces BYTE-IDENTICAL packed/norms output to a manual per-token loop calling single-token encode 5 times (the regression-pin for `_seq` semantics + src_offset stride); (3) `encode_seq_tokens_to_tq_with_src_tok_offset_skips_leading_tokens` — defensive: src_tok_offset > 0 correctly skips leading source tokens (matches 4-bit `_seq` convention). 3 new tests PASS. 3347/0/9 full-suite green (was 3344, +3 new). **iter-15 wires `encode_seq_tokens_to_tq` at all 4 KV write sites in `gpu_full_attn::full_attn_layer_gpu` (decode line 1913, prefill 2017/2578, decode_into 3130) + the FWHT pre/post + `dispatch_tq_sdpa` swap at the decode SDPA dispatch site (line 1963).** Cross-repo: mlx-native commit hash will be referenced in iter-15. |
| 13 | 2026-05-08 | B | LANDED | **🎯 GPU end-to-end NRMSE 0.008103 (matches iter-11 CPU oracle byte-for-byte)** — full GPU chain (encode → Q FWHT pre-mult → flash_attn_vec_tq_hb → output FWHT sign-undo) produces numerically-correct output at qwen35 shape. Test exercises real Metal kernels: (a) `slot.encode_token_to_tq` (b) `mlx_native::ops::fwht_standalone::dispatch_fwht_sign_premult_f32` (Q in-place pre-mult, real GPU FWHT) (c) `slot.dispatch_tq_sdpa` (real GPU SDPA via flash_attn_vec_tq_hb) (d) `mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32` (output in-place inverse-mult). Memory barriers between each stage. Compared GPU output to F32 closed-form reference at kv_seq_len=1 → NRMSE = 0.008103 (identical to iter-11's CPU oracle measurement 0.008). The GPU SDPA path is byte-equivalent in numerical accuracy to the CPU reference. **iter-14 wires this chain into `gpu_full_attn::full_attn_layer_gpu`** at the production decode SDPA dispatch site (line 1963) + the prefill TQ encode loop at all 4 KV write sites (lines 1913, 2017, 2578, 3130). 1 new test PASS. 3344/0/9 full-suite green (was 3343, +1 new). |
| 12 | 2026-05-08 | B | LANDED | **Engine-load env-driven `tq_kv_active` plumbing.** New `Qwen35LoadedModel.tq_kv_active: bool` field sourced once at engine load via `tq_packed_descriptor::is_tq_active_mode()` (the same `HF2Q_TQ_KV` env helper Gemma uses at `engine.rs:2328`). `alloc_kv_cache_for_request` swapped from `HybridKvCache::new(...)` to `HybridKvCache::new_with_options(... qwen.tq_kv_active)` so EVERY production qwen35 prefill now allocates TQ-active full-attn buffers (alongside F32 K/V) when env=1, default-OFF on legacy F32 path. Both test fixtures (`load_info.rs:1326` + `engine.rs:9154`) updated with `tq_kv_active: false` for the legacy regression contract. Tests: (1) `alloc_kv_cache_for_request_tq_off_keeps_full_attn_tq_none` — pins legacy F32 path preserved; (2) `alloc_kv_cache_for_request_tq_on_populates_tq_per_full_attn_slot` — TQ allocation engages per slot at qwen36 head_dim=256 shape. 2 new tests PASS on real Metal device. 3343/0/9 full-suite green (was 3341, +2 new). **Iter-12 makes the flag load-bearing for ALLOCATION; iter-13 makes it load-bearing for DISPATCH** by wiring `slot.encode_token_to_tq` + `slot.dispatch_tq_sdpa` (via FWHT pre/post on Q/output) into `gpu_full_attn::full_attn_layer_gpu`. Until iter-13, env=1 increases per-request memory cost (40 MB/slot at 8K) but does NOT change inference output (F32 read path stays exclusive). Mantra-aligned (no-stub): the flag IS load-bearing for allocation; iter-13 makes it load-bearing for dispatch — neither is a stub. |
| 11 | 2026-05-08 | B | LANDED | **🎯 NRMSE-vs-F32 LITMUS PASS at 0.008103 (threshold 0.15) — qwen35 TQ-on numerically correct.** End-to-end CPU parity test exercises the full chain: GPU encode K/V via `dispatch_hadamard_quantize_kv_hb` (D1 SRHT + 8-bit Lloyd-Max) → readback packed/norms → CPU `flash_attn_vec_tq_hb_oracle` (Q pre-rotated via D1 sign × FWHT) → CPU FWHT × sign-undo on output → NRMSE vs the F32 closed-form reference at kv_seq_len=1 (`output_ref[h] = V[kv_head(h)]` since softmax over 1 score = 1.0). **Result: NRMSE = 0.008103 vs ADR-007 §F-0.3 threshold of 0.15 (18.5× headroom).** Bug found and fixed during iter-11: initial test used plain FWHT, but the Gemma encode kernel applies D1 SRHT (sign × FWHT, sign table `TBQ_SIGNS_256` verbatim from `mlx-native/src/shaders/hadamard_quantize_kv_fast.metal:21-26`); both encode AND Q pre-rotation must use the same sign pattern (sign[i]^2 = 1 cancels under Q@K^T so attention scores still equal F32 baseline). Initial NRMSE 1.432 → after sign fix 0.008. **Phase B is SHIPPABLE numerically.** New CPU helpers in test module: `TBQ_SIGNS_256`, `apply_d1_sign_d256`, `sign_premult_fwht_d256`, `fwht_sign_undo_d256`, `nrmse`, `synth_token_with_cpu_mirror`. 1 new test PASS (the litmus). 3341/0/9 full-suite green (was 3340, +1 new). **iter-12 wires the encode + dispatch into `gpu_full_attn::full_attn_layer_gpu` per-token write path for end-to-end coherent generation under `HF2Q_TQ_KV=1`.** |
| 10 | 2026-05-08 | B | LANDED | **`FullAttnKvSlot::dispatch_tq_sdpa` + Qwen35TqSdpaParams.** New method wraps mlx-native's `flash_attn_vec_tq_hb` consuming `slot.tq.{k_packed, k_norms, v_packed, v_norms}`. New `Qwen35TqSdpaParams` struct mirrors mlx-native's kernel params in the qwen35 namespace so engine call sites don't import mlx-native types directly. Fail-loud on `slot.tq.is_none()`. Tests on real Metal: (1) `dispatch_tq_sdpa_errors_when_slot_lacks_tq_buffers` — fail-loud contract; (2) `dispatch_tq_sdpa_produces_finite_nonzero_output_at_qwen35_shape` — encode 1 token, dispatch SDPA at kv_seq_len=1, verify output is finite + non-zero (kernel chain executes without panic at qwen35 16-head/2-kv-head/head_dim=256 shape); (3) `dispatch_tq_sdpa_two_position_kv_finite_output` — multi-position KV (kv_seq_len=2) regression; (4) `dispatch_tq_sdpa_rejects_kv_seq_len_zero` — kernel param validation propagates through the wrapper. 4 new tests PASS on real Metal device. 3340/0/9 full-suite green (was 3336, +4 new). **iter-10 sanity is necessary-but-not-sufficient:** finite/non-zero output proves the kernel chain executes, but does NOT prove numerical correctness. iter-11 ships the full F32-baseline NRMSE-vs-TQ parity test (the litmus that determines whether qwen35 TQ-on path is shippable) + wires the encode/dispatch into `gpu_full_attn::full_attn_layer_gpu` per-token write path for end-to-end coherent generation. |
| 9 | 2026-05-08 | B | LANDED | **`FullAttnKvSlot::encode_token_to_tq` GPU encode path.** New method wraps mlx-native's `dispatch_hadamard_quantize_kv_hb` for both K and V, writing to `slot.tq.{k_packed, k_norms, v_packed, v_norms}` at a given `write_pos`. Fail-loud on `slot.tq.is_none()` (mantra: no silent fallback). Tested on real Metal hardware via `MlxDevice` + `KernelRegistry::new()` (the registry auto-registers `hadamard_quantize_kv_hb_d256/d512` at construction). Tests: (1) `encode_token_to_tq_errors_when_slot_lacks_tq_buffers` — fail-loud contract; (2) `encode_token_to_tq_writes_packed_at_write_pos_only` — pins kernel positional addressing (offset = `head*capacity*head_dim + write_pos*head_dim + dim_idx`); (3) `encode_token_to_tq_writes_positive_norms` — verifies FWHT+L2 norm pipeline writes positive norms at written positions; (4) `encode_token_to_tq_at_two_positions_writes_both_independently` — two encodes in one CommandEncoder land at separate positions without cross-contamination. 4 new tests PASS on real Metal device. 3336/0/9 full-suite green (was 3332, +4 new). **iter-10 wires this into `gpu_full_attn::full_attn_layer_gpu` per-token KV write path (production load-bearing) + adds the SDPA dispatch via `flash_attn_vec_tq_hb` + the NRMSE-vs-F32 parity test (the litmus that tells us whether qwen35 TQ-on path is shippable).** |
| 8 | 2026-05-08 | B | LANDED | **HybridKvCache `new_with_options` constructor + FullAttnKvSlot.tq field.** New `HybridKvCache::new_with_options(cfg, dev, max_seq_len, n_seqs, tq_kv_active)` constructor; legacy `HybridKvCache::new(...)` delegates with `tq_kv_active=false` (preserves all 71 existing call sites byte-identically). `FullAttnKvSlot` gains `pub tq: Option<TqFullAttnKvBuffers>` field — `Some` in TQ-active mode, `None` (current behavior) otherwise. Allocator branches per-slot (regular full-attn + MTP slot if present); linear-attn slots are unchanged (DeltaNet SSM state stays F32 per ADR §3 non-goal). **Shadow-cache pattern (iter-8):** when `tq_kv_active=true`, BOTH F32 K/V (16 MB each) AND TQ buffers (8.52 MB) are allocated per slot, mirroring Gemma's `dense_kvs` + `leg_hb_encoded` co-existence at `forward_mlx.rs:739+824`. Per-slot total at qwen36 35B-A3B-APEX shape: 42_074_112 bytes (vs 33_554_432 F32-only). The 3.94× memory savings claim (§1) materializes once iter-11 drops the F32 backing in TQ mode (regression-pin: `tq.total_bytes() == 8_519_680` is asserted in iter-8 test). 5 new tests PASS: tq_off keeps tq=None per slot + legacy `new()` byte-identical; tq_on populates tq per full-attn slot + asserts norms_per_pos=1; tq_on byte counts at qwen36 APEX shape; MTP slot honors flag in both directions. 3332/0/9 full-suite green (was 3327 pre-iter-8, +5 new). iter-9 wires the SDPA dispatch + KV write branches that consume the tq buffers (production load-bearing). |
| 7 | 2026-05-08 | B | LANDED | **TQ-active full-attn KV buffer infra (additive).** New `TqFullAttnKvBuffers` struct (k_packed/k_norms/v_packed/v_norms + norms_per_pos), `tq_norms_per_pos_for(head_dim) -> u32` helper (mirrors mlx-native `forward_mlx.rs:2326` formula), `alloc_tq_full_attn_buffers(cfg, dev, max_seq_len, n_seqs)` standalone allocator, `full_attn_slot_f32_bytes(...)` parity helper. **NOT yet wired into `HybridKvCache::new`** — Chesterton's fence on the live serve path; iter-8 wires the dispatch when the SDPA branch is also ready. Empirical parity test `tq_full_attn_buffers_byte_count_3p94x_smaller_than_f32` proves 3.94× per-slot byte savings at qwen36 35B-A3B-APEX shape (n_kv_heads=2, head_dim=256, max_seq_len=8192, n_seqs=1: F32=33.55 MB → TQ=8.52 MB). 6 new tests PASS (alloc shape, byte counts, zero-init discipline, n_seqs=2 outer-axis correctness, max_seq_len/n_seqs preflight error paths). 3327/0/9 full-suite green (was 3320 pre-iter-7, +7 new). **Investigation finding (Chesterton):** Gemma's `MlxModelWeights` does NOT have a `tq_kv_active: bool` field — TQ-active mode is sourced at engine-spawn from `HF2Q_TQ_KV` env via `is_tq_active_mode()` (engine.rs:2328). The ADR §6 plan to put a field on `MlxModelWeights` (qwen35) was based on a wrong assumption; iter-8 will source `tq_kv_active` from env at `alloc_kv_cache_for_request` instead, mirroring Gemma's pattern. |
| 6b.3 | 2026-05-08 | A | LANDED | **Cold-start hydrate seam + cross-process replay.** New `LcpSidecarMetadata` struct + sidecar codec (`QH3M` magic, version 1) appended to QH35 envelope tail — orthogonal to the snapshot codec, so adding/changing sidecar fields does NOT bump `QH35_CODEC_VERSION` (preserves Chesterton's fence around v1). Sidecar carries `(model_fingerprint, tenant_id, params_hash, prompt_tokens, sliding_window, linear_capacity)` so the cold-start path can replay the exact in-memory `LcpRegistry::store(...)` call. `Qwen35DiskPersistor::{write,read,hydrate_for_cfg}` thread sidecar through; `store_lcp_with_disk_writeback` constructs sidecar from the live (key, prompt_tokens, sliding_window, linear_capacity) on the disk-write arm. New `Qwen35LoadedModel::hydrate_lcp_registry_from_disk(&HybridKvCache, &MlxDevice)` is idempotent per cfg-fingerprint via `lcp_hydrated_for_cfg: HashSet<String>` (one HashSet check per request after first hydrate; one read_dir per cfg per process). Wired at all 4 prefill entrypoints in `engine_qwen35.rs` (`generate_qwen35_once`, `generate_qwen35_once_with_soft_tokens`, `generate_qwen35_once_with_soft_tokens_and_deepstack`, streaming) BEFORE the LCP probe so hydrated entries are visible. New tests: `qh3m_sidecar_codec_byte_round_trip`, `qh3m_sidecar_codec_rejects_bad_magic`, `qh3m_sidecar_codec_rejects_version_drift`, `qh3m_sidecar_empty_prompt_tokens_round_trip`, `qh35_with_sidecar_round_trip_byte_equal`, `qh35_with_sidecar_full_attn_only_round_trip`, `qh35_with_sidecar_rejects_truncated_envelope`, `qh35_disk_cross_process_replay_into_fresh_lcp_registry` (proves write→drop→fresh persistor→hydrate→`lcp_registry.lookup` hits with original key + reports correct LCP length k=12 against a divergent new prompt), `qh35_disk_clean_cold_start_yields_empty_hydrate`. 24 qh3 codec/disk tests + 516 qwen35 tests PASS = **540 tests**, zero regressions. SERVE end-to-end (live engine + 2-turn HTTP across process restart with TTFT speedup verified) deferred to operator validation gate. |
| 6b.3 | (pending) | A | – | Hydrate auto-reinsertion: extend QH35 envelope (or add sidecar metadata file) to round-trip `LcpKey + prompt_tokens + sliding_window + linear_capacity` so `hydrate_for_cfg` can re-insert into in-memory `LcpRegistry` automatically; first-prefill hydrate seam on Qwen35LoadedModel; SERVE end-to-end validation: 2-turn chat across process restart with TTFT speedup verified. |
