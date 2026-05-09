# ADR-027 iter-23: F32 KV Backing Drop — Multi-Iter Refactor Design

- **Date:** 2026-05-09
- **Status:** Planning (pre-execution)
- **Owner:** ADR-027 Phase B continuation
- **Regression net:** `scripts/adr027-cross-axis-sweep.sh` (iter-21)
- **Empirical pin:** iter-18 byte-breakdown tests at qwen36 32K shape

## Why this dossier exists

ADR-027 is LANDED (Phase A + Phase B iter-7..22). Cross-axis sweep BYTE-IDENTICAL (iter-19); live qwen36 35B-A3B-APEX-Q5_K_M produces coherent output under `HF2Q_TQ_KV=1` (iter-16); decode within ±1% / prefill within 2.4% (iter-20).

**The remaining gap**: under `HF2Q_TQ_KV=1`, the **shadow-cache pattern** allocates BOTH F32 K/V (16 MB/slot at 8K) AND TQ buffers (8.52 MB/slot) per full-attn slot. Net memory cost vs F32 baseline: **+25.4%** (1.34 GB → 1.57 GB at qwen36 32K shape). Peers (KIVI, vLLM TQ) ship **3-4× memory savings** vs F32; we currently ship 1.25× more memory in TQ mode.

iter-22's investigation surfaced the refactor scope: ~30+ read sites across 5 files. Attempting it in one iter half-completed the type change and hit cargo-error walls. Per mantra "Measure 3x, cut once" — this dossier breaks the work into iter-tractable pieces, each with clear scope, tests, and acceptance criteria validated against the iter-21 regression harness.

## Target state

When `HF2Q_TQ_KV=1`:
- `FullAttnKvSlot.k = None`, `FullAttnKvSlot.v = None`
- `FullAttnKvSlot.tq = Some(TqFullAttnKvBuffers)` (already iter-7..15 wiring)
- Per-slot total: **8.52 MB** (vs 33.55 MB F32 baseline at 8K) → **3.94× savings**
- Total at qwen36 32K (10 full-attn slots): **325 MiB** (vs 1.34 GB F32) → **3.94× savings** (matches the §1 ADR claim + iter-18 regression-pin projection)

Default (`HF2Q_TQ_KV=0` or unset):
- Behavior **byte-identical** to today's F32 baseline
- `FullAttnKvSlot.k = Some(F32 buf)`, `FullAttnKvSlot.v = Some(F32 buf)`, `FullAttnKvSlot.tq = None`
- Cross-axis sweep harness (iter-21) cell A/B coherence preserved

## Sub-iter sequence

The refactor splits into 5 sub-iters. Each ships a complete deliverable + passes the iter-21 sweep + adds at least one regression-pin test.

### iter-23a — `HybridKvCacheSnapshot` field Optional-ization (structural)

**Scope (1 file, ~80 LOC):**
- `kv_cache.rs::HybridKvCacheSnapshot` — change `full_attn_k: Vec<MlxBuffer>` → `Vec<Option<MlxBuffer>>` (same for `full_attn_v`).
- `MtpKvSnapshot.k/v` → `Option<MlxBuffer>`.
- `total_bytes()` and `byte_len()` impls handle Optional (sum only Some entries).
- `Debug` impl + persistor codec acknowledge Optional.

**Producers stay Some-only:** `HybridKvCache::snapshot` continues to push `Some(deep_copy_buffer(...))` for every slot. No allocation behavior change. F32 path identical.

**Consumers handle None:** `HybridKvCache::restore_from` and `restore_partial` skip slot K/V copy when source `Option` is None — but in iter-23a all sources are Some, so the None branches are unexercised.

**Persist codec (qwen35_hybrid_persistor.rs):** prepend a 1-byte `kv_present` flag per slot (1=Some, 0=None). codec_version stays 1 (extension is per-slot, not header-level). Reader rejects unknown flag values.

**Tests:**
- All existing 540+ tests pass (Some-only path byte-identical).
- New: `qh35_persist_codec_handles_none_full_attn_k_v_round_trip` (synthetic snapshot with `None` slots).
- `iter-21 sweep harness PASS` (cells A/B/C/D unchanged).

**Acceptance:** structural API ready for None producers; no behavior change.

### iter-23b — `HybridKvCache.tq_kv_active` field

**Scope (1 file, ~30 LOC):**
- `kv_cache.rs::HybridKvCache` gains `pub tq_kv_active: bool` field.
- `HybridKvCache::new_with_options` populates it from the parameter.
- `Qwen35LoadedModel::alloc_kv_cache_for_request` already passes `qwen.tq_kv_active`; nothing else changes (constructor wiring already plumbed in iter-12).

**Tests:**
- New: `hybrid_kv_cache_tq_kv_active_field_matches_constructor_arg` (both directions).
- All existing tests pass.
- Sweep harness PASS.

**Acceptance:** `HybridKvCache` knows its own TQ-active state; future iters branch on it.

### iter-23c — `FullAttnKvSlot.k/v` Optional-ization (the big one)

**Scope (5 files, ~150 LOC):**
- `kv_cache.rs`:
  - `FullAttnKvSlot.k: Option<MlxBuffer>`, `v: Option<MlxBuffer>`.
  - `alloc_full_attn_slot(cfg, dev, max_seq_len, n_seqs, tq_kv_active: bool)` — new param. Returns `k=Some(buf)/v=Some(buf)` when `tq_kv_active=false`; `k=None/v=None` when `true`.
  - `HybridKvCache::new_with_options` threads `tq_kv_active` to every `alloc_full_attn_slot` call (4 call sites: regular full-attn × N + MTP × 0/1).
  - `reset_all_buffers` — `if let Some(buf) = slot.k.as_mut()` guards.
  - `snapshot` — pushes `slot.k.as_ref().map(|buf| deep_copy_buffer(...))` (Optional-aware).
  - `restore_from` / `restore_partial` — skip K/V copy when destination is None OR source is None (mismatched modes are an error).
  - `full_attn_bytes_breakdown` — sum bytes only when slot.k/v is Some.

- `gpu_full_attn.rs` (6 sites):
  - `apply_sdpa_with_kv_cache` lines 113, 262, 2168, 2359, 2401, 3368: each call to `flash_attn_vec(... &slot.k, &slot.v ...)` becomes `flash_attn_vec(... slot.k.as_ref().expect("F32 mode required"), slot.v.as_ref().expect(...) ...)`. The `expect` fires only if `dispatch_decode_sdpa_with_optional_tq` (iter-15) routed wrong — that function gates on `slot.tq.is_some()`, so a panic here is a real bug, not a missing None handler.
  - Equivalent for `dispatch_kv_cache_copy_seq_f32_dual` calls — already guarded inside `write_kv_with_optional_tq_encode` (iter-15) by `slot.tq.is_some()`. No change needed; the helper just needs to use `slot.k.as_ref().expect(...)` patterns under its existing F32 branch.

- `qwen35_hybrid_persistor.rs::cfg_from_cache` (3 sites):
  - `slot.k.shape()` reads — branch on `slot.k.as_ref()`. When None, derive shape from `slot.tq` instead (TQ buffers carry the same outer shape minus the head_dim/4 quantize).
  - For iter-23c, since TQ-only mode now lacks F32, `cfg_from_cache` either reads from TQ buffers OR errors out with a clear "no F32 K/V — call cfg_from_tq_buffers instead" message. Iter-23d will add the TQ-side helper.

- `lcp_registry.rs::ByteSized` impl on `HybridKvCacheSnapshot`:
  - Already iterates `total_bytes()` which correctly sums over Optional Some slots (already updated in iter-23a). No change needed.

- Test fixtures (kv_cache.rs::tests, qwen35_disk_persistor.rs::tests, qwen35_hybrid_persistor.rs::tests):
  - `synth_full_attn_only_snapshot` etc. push `Some(buf)` entries.
  - New: `synth_tq_only_snapshot` builds `None`-K/V snapshots for iter-24 codec tests.

**Tests:**
- `full_attn_bytes_breakdown_tq_on_drops_f32_at_qwen36_32k` — empirically verifies the iter-18 regression-pin: at qwen36 32K shape with `tq_kv_active=true`, `f32_k_v_bytes == 0` and `total_bytes == 340_787_200` (3.94× savings). **This is the load-bearing memory-savings regression-pin.**
- `alloc_full_attn_slot_tq_active_returns_none_k_v` — pin allocator behavior.
- `hybrid_kv_cache_new_with_options_tq_off_byte_identical_to_legacy` — F32 path unchanged.
- All existing tests pass with appropriate `.as_ref()` adaptations.
- Sweep harness PASS — **critical**: cell C and D must still produce byte-identical output to F32 baseline. The TQ chain (iter-13 GPU litmus 0.008 NRMSE) is unaffected; the only risk is whether snapshot/restore semantics break when LCP probe fires off `None` K/V. iter-23a's None handling closes that risk.

**Acceptance:** memory savings live + sweep harness still passes byte-identity.

### iter-23d — TQ-aware persist codec

**Scope (1 file, ~100 LOC):**
- `qwen35_hybrid_persistor.rs::serialize_hybrid_snapshot` — write a per-slot `kv_present: u8` flag (1=F32 Some, 0=TQ None). When 0, skip K/V byte payload (just the current_len).
- `deserialize_hybrid_snapshot` — read the flag; on 0, leave slot.k/v as None.
- `cfg_from_cache` — branch on `slot.k.is_none()` to derive cfg from `slot.tq` shape instead.
- New: `cfg_from_tq_buffers(tq, codec, n_seqs)` helper for the TQ-only case.

**Tests:**
- `qh35_envelope_round_trip_tq_only_snapshot` — synthetic None-K/V snapshot round-trips byte-identical.
- `qh35_envelope_round_trip_mixed_some_none_snapshot` — defensive (shouldn't happen in production but the codec must handle it).
- Sweep harness cell B (F32 + persist) and D (TQ + persist) both still pass.

**Acceptance:** cross-axis sweep cell D produces byte-identical output even after F32 drop.

### iter-23e — LANDED memory file + ADR header note

**Scope (1 file, ~30 LOC):**
- `Users/robert/.claude/projects/-opt-hf2q/memory/project_adr027_phase_b_LANDED_2026_05_08.md` — append iter-23a..d closure section noting the F32-drop and the empirical 3.94× savings.
- `docs/ADR-027-qwen35-tq-kv-cache-and-persist-family.md`:
  - Bump iter log with iter-23a..d entries.
  - Update §1 KV memory table footnote: "iter-23 LANDED 2026-05-09 — empirically verified 3.94× per-slot savings at qwen36 32K shape (340 MiB vs 1.34 GB F32 baseline)."

**Tests:** None new. Doc-only.

**Acceptance:** ADR + memory reflect the realized memory savings.

## Risk register

| Risk | Mitigation |
|------|------------|
| iter-23c's `.as_ref().expect(...)` panics fire under unexpected paths | Iter-15's `dispatch_decode_sdpa_with_optional_tq` already gates F32 vs TQ on `slot.tq.is_some()`. The `expect` guards are belt-and-suspenders against future bugs that add new SDPA dispatch sites. |
| Snapshot/restore semantics break when `None` source meets `Some` destination (or vice versa) | iter-23a's restore handles all 4 (None, Some) × (None, Some) combinations explicitly with clear error messages. Test added per cell. |
| iter-21 sweep harness regresses on cell D (TQ + persist) | iter-23d explicitly covers this. Pre-flight: rerun harness BEFORE iter-23c lands to capture baseline; after iter-23d lands, harness must still pass. |
| LCP probe in TQ mode reads `None` K/V via snapshot | iter-23a's restore path handles None as "skip the F32 restore"; LCP semantics still work because the TQ buffers (which the SDPA reads) are correctly populated by encode_seq_tokens_to_tq. The fact that LCP's snapshot has empty K/V is fine — the TQ buffers carry the actual state. |
| Test fixtures across the codebase need `.as_ref()` updates | Mechanical refactor; cargo's exhaustiveness errors guide the migration. The iter-21 sweep is a runtime regression net; cargo build is a compile-time net. |

## Rollback plan

Each sub-iter is independently revertable:
- iter-23a: revert the `Vec<Option<MlxBuffer>>` change → field is `Vec<MlxBuffer>` again.
- iter-23b: revert the `tq_kv_active` field add.
- iter-23c: revert the `FullAttnKvSlot.k/v` Optional change → fields are `MlxBuffer` again.
- iter-23d: revert the codec extension → reader rejects the `kv_present` byte.

If iter-21 sweep fails after any sub-iter lands, revert that sub-iter and investigate. Don't pile on more changes on a broken regression net.

## Why split this way (Chesterton's fence on the iter sequence)

- **23a before 23b/c**: snapshot's `Vec<Option>` shape MUST exist before producers (HybridKvCache::snapshot in TQ mode) start emitting None. Otherwise the type system rejects.
- **23b before 23c**: `HybridKvCache.tq_kv_active` field is the source-of-truth for the alloc branch in 23c. Threading the flag through `new_with_options` is already done in iter-12; 23b just records it on the cache itself.
- **23c before 23d**: codec needs a None producer to round-trip-test against. Without 23c, the None branch of the codec is untested.
- **23d before 23e**: persist must work end-to-end before the doc declares LANDED.

## Estimated scope

| Sub-iter | Files | LOC | Tests added | Tests touched |
|----------|-------|-----|-------------|---------------|
| 23a | 2 (kv_cache.rs, qwen35_hybrid_persistor.rs) | ~80 | 1 | ~10 |
| 23b | 1 (kv_cache.rs) | ~30 | 1 | 0 |
| 23c | 5 (kv_cache.rs, gpu_full_attn.rs, qwen35_hybrid_persistor.rs, tests) | ~150 | 3 | ~15 |
| 23d | 1 (qwen35_hybrid_persistor.rs) | ~100 | 2 | 0 |
| 23e | 2 (memory, ADR) | ~30 | 0 | 0 |
| **Total** | **~7 files** | **~390** | **7** | **~25** |

Per the /loop's iter cadence (~10 min execution + commit per iter), this is 5 iters × ~10 min = ~1 hour of focused execution. Plus contingency for test failures + sweep harness re-runs.

## Acceptance criteria for the whole iter-23 sequence

1. ✅ Cross-axis sweep harness (iter-21) passes after every sub-iter — byte-identical output across all 4 cells.
2. ✅ `full_attn_bytes_breakdown` at qwen36 32K shape with `tq_kv_active=true` reports `f32_k_v_bytes=0`, `total_bytes=340_787_200` (3.94× savings).
3. ✅ Live qwen36 generation under `HF2Q_TQ_KV=1` produces coherent output (re-verify iter-16 result post-refactor).
4. ✅ `tq_kv_active` × `kv_persist` × `lcp_resume` cross-axis sweep all pass.
5. ✅ Decode tok/s within ±2% of F32 baseline (iter-20 perf gate); prefill within ±5%.
6. ✅ ADR-027 §10 iter log updated with iter-23a..e entries.
7. ✅ Memory file `project_adr027_phase_b_LANDED_2026_05_08.md` reflects realized 3.94× savings.

## Reference: investigation finding (iter-22)

iter-22 attempted the full Optional refactor mid-iter and surfaced these specific sites:

```
$ grep -rn "slot\.k\b\|slot\.v\b" src/inference/models/qwen35 --include="*.rs" \
    | grep -v "test\|//\|\.tq\." | wc -l
~30
```

Production read sites:
- gpu_full_attn.rs:113, 262, 2168, 2359, 2401, 3368 (6 SDPA dispatches)
- kv_cache.rs reset_all_buffers (4), snapshot (4), restore_from (4), restore_partial (4)

Persist codec:
- qwen35_hybrid_persistor.rs cfg_from_cache (3 .shape() reads)
- qwen35_hybrid_persistor.rs serialize/deserialize (2 byte read/write loops)

Test fixtures:
- ~8 sites that build synthetic FullAttnKvSlot directly

Total ~30+ sites — tractable per-sub-iter, intractable in one iter.
