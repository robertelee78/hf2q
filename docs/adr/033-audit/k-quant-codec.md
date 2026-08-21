# Audit: src/quantize/k_quant_codec.rs (P-1, ADR-033)

Pinned external references: llama.cpp @ `c779f6198` (`/opt/llama.cpp/ggml/src/ggml-quants.c`)

## Summary of structure

`k_quant_codec.rs` is a **pure dispatch shim** — no actual quant kernels live here. The
quant kernels (`quantize_row_qN_..._to_bytes`) live in `src/quantize/k_quant.rs` (K-family)
and `src/quantize/q_legacy.rs` (legacy 32-block formats); this file imports them and
chooses one per `(KQuantTarget, CalibrationData)` pair. Under ADR-033:

- `KQuantTarget` is **superseded by `GgmlType`** (Decision §1: the new IR carries
  `ggml_type: GgmlType` directly; metadata methods `bytes_per_block` /
  `elements_per_block` / `ggml_type()` / `bpw()` become methods on `GgmlType`).
- `CalibrationData::{None, Imatrix, ImatrixWithStats, DynamicQuant}` 4-variant routing is
  **superseded by the `Quantizer::quantize(src, n_per_row, imatrix: Option<&[f32]>)`
  signature** (Decision §2 — a single optional slice replaces the 4-variant enum + map
  lookup; `DynamicQuant` is gone because ADR-033 retires hf2q's homebrew DWQ entirely).
- `quantize_row_to_bytes` / `quantize_tensor_2d_to_bytes` are **superseded by per-type
  `impl Quantizer for QX` at `src/quantize/ggml_quants/<type>.rs`** — those impls are
  ports of `ggml_quants.c` and are called by the convert dispatcher directly. No
  intermediate `KQuantTarget` dispatch layer survives.
- `KQuantCodecError` (the dispatch-layer error type) is **superseded by
  `QuantizeError`** (`NoQuantizerForType` + per-impl errors per Decision §2).
- All known external callers (`k_quant_codec_quantizer.rs`, `variant_quantizer.rs`,
  `dwq_k_quantizer.rs`, `mixed.rs` and the test-only references) are themselves on the
  P6 delete-list per ADR memo `[[cfa-adr033-review-2026-05-17]]`. `main.rs:2127-2167`
  and `backends/gguf.rs:1328-4835` references all flow through
  `KQuantCodecQuantizer` / `METHOD_K_QUANT_CODEC_DIRECT` sentinel — those dispatch
  arms are rewritten in P1 (StandardPolicy wired direct to `Quantizer` trait), so the
  `KQuantTarget` enum + dispatch is gone end-to-end.

Net: every symbol in this file is DELETE; nothing here is a kernel that P0 needs to
port (kernels live in sibling files `k_quant.rs` / `q_legacy.rs`, audited separately).

## Per-symbol disposition

| Symbol (line range) | LOC | Disposition | Rationale (one line) |
|---|---|---|---|
| `enum KQuantTarget` (L54-94) | 41 | DELETE | Superseded by `GgmlType` (Decision §1); same wire values move to that enum. |
| `impl KQuantTarget` (L96-201) — block of 7 methods | 106 | (see rows below) | |
| `fn KQuantTarget::from_ggml_type` (L99-113) | 15 | DELETE | Replaced by `GgmlType::try_from(u32)` per Decision §1. |
| `fn KQuantTarget::ggml_type` (L116-129) | 14 | DELETE | Replaced by `From<GgmlType> for u32`. |
| `fn KQuantTarget::bytes_per_block` (L132-145) | 14 | DELETE | Replaced by `GgmlType::bytes_per_block` method (data from ggml-quants.c `static_assert`s — verbatim values move). |
| `fn KQuantTarget::elements_per_block` (L148-157) | 10 | DELETE | Replaced by `GgmlType::elements_per_block` method (256 for K, 32 for legacy — verbatim). |
| `fn KQuantTarget::supports_imatrix` (L161-163) | 3 | DELETE | Implicit in per-type `Quantizer` impl (impls that ignore `imatrix` arg are the "non-supporting" set); no API method needed. |
| `fn KQuantTarget::bpw` (L167-180) | 14 | DELETE | Display-only; if needed for `--help` / dry-run, becomes `GgmlType::bits_per_weight`. No production caller. |
| `fn KQuantTarget::all` (L187-200) | 14 | DELETE | Used only by codec-coverage tests in this file; replaced by `GgmlType` enum's exhaustive matches (compiler-checked). |
| `enum KQuantCodecError` (L204-236) | 33 | DELETE | Superseded by `QuantizeError` (Decision §2 — `NoQuantizerForType` + per-impl errors). `DwqAtRowQuantize` variant retires with hf2q-DWQ (P6). `ImatrixWeightsLengthMismatch` becomes a `QuantizeError` variant on the per-type `Quantizer::quantize` impls. |
| `fn lookup_imatrix_weights` (L241-253) | 13 | DELETE | The 4-variant `CalibrationData` enum is superseded by `Option<&[f32]>` (Decision §2); no lookup needed — convert dispatcher passes the slice directly. |
| `pub fn quantize_row_to_bytes` (L272-316) | 45 | DELETE | The match-on-`(target, weights)` dispatch is superseded by `Quantizer` trait dispatch — convert pipeline calls `<impl Quantizer>::quantize(row, n_per_row, imatrix)` on the chosen impl, no intermediate KQuantTarget routing. |
| `pub fn quantize_tensor_2d_to_bytes` (L338-378) | 41 | DELETE | The "row over 2D tensor" loop becomes a thin helper inside each `Quantizer` impl (or a default-method on the trait); no public `KQuantTarget`-keyed API survives. |
| **TEST BLOCK (`#[cfg(test)] mod tests`, L380-1452)** — 35 tests | 1073 | DELETE | All tests target `KQuantTarget` + `quantize_row_to_bytes` + `quantize_tensor_2d_to_bytes` + `CalibrationData` routing surface — every one of those is deleted above. Per-`GgmlType` byte-cmp tests against `tests/fixtures/ggml_quants/<type>_<n>.bin` (P0 acceptance criterion) replace the entire suite at higher fidelity (byte-cmp vs llama.cpp reference instead of intra-hf2q RMSE bounds + dispatch smoke). |
| ↳ `fn smooth_ramp` (L388-395) | 8 | DELETE | test helper. |
| ↳ `fn target_ggml_type_round_trip` (L399-418) | 20 | DELETE | Tests `KQuantTarget::from_ggml_type`/`ggml_type`; replaced by `GgmlType::try_from`/`u32::from` round-trip test. |
| ↳ `fn target_bytes_per_block` (L423-435) | 13 | DELETE | Tests `KQuantTarget` metadata; replaced by `GgmlType::bytes_per_block` test. |
| ↳ `fn target_elements_per_block` (L439-449) | 11 | DELETE | Tests `KQuantTarget` metadata; replaced equivalently on `GgmlType`. |
| ↳ `fn target_supports_imatrix` (L454-464) | 11 | DELETE | Tests removed method. |
| ↳ `fn target_bpw` (L468-478) | 11 | DELETE | Tests removed method. |
| ↳ `fn dispatch_none_q4_k` (L483-495) | 13 | DELETE | Tests removed dispatcher; per-type `impl Quantizer for Q4K` has byte-cmp coverage at P0. |
| ↳ `fn dispatch_none_q5_k` (L497-509) | 13 | DELETE | Same as above for Q5_K. |
| ↳ `fn dispatch_none_q6_k` (L511-526) | 16 | DELETE | Same as above for Q6_K. |
| ↳ `fn dispatch_imatrix_q4_k_uses_weighted_path` (L528-560) | 33 | DELETE | Tests `_imatrix` vs `_ref` dispatch divergence; replaced by P0 byte-cmp where the imatrix path is itself a fixture (no need to assert "differs from ref"). |
| ↳ `fn dispatch_imatrix_missing_tensor_falls_back_to_ref` (L562-588) | 27 | DELETE | Tests fallback behavior of removed lookup function; under ADR-033 caller passes `None` explicitly. |
| ↳ `fn dispatch_imatrix_with_stats_equiv` (L590-625) | 36 | DELETE | Tests `ImatrixWithStats` variant of removed `CalibrationData` enum. |
| ↳ `fn dispatch_dwq_rejected` (L627-649) | 23 | DELETE | Tests `DwqAtRowQuantize` error variant; hf2q-DWQ retires entirely in P6. |
| ↳ `fn dispatch_imatrix_length_mismatch` (L651-678) | 28 | DELETE | Tests removed error variant; replaced by per-impl `QuantizeError` length-check tests. |
| ↳ `fn dispatch_misaligned_input` (L680-699) | 20 | DELETE | Tests removed error variant; replaced by per-impl length-check tests. |
| ↳ `fn dispatch_none_q4_0` (L701-713) | 13 | DELETE | Tests removed legacy dispatch. |
| ↳ `fn dispatch_none_q4_1` (L715-727) | 13 | DELETE | Tests removed legacy dispatch. |
| ↳ `fn dispatch_none_q5_0` (L729-741) | 13 | DELETE | Tests removed legacy dispatch. |
| ↳ `fn dispatch_none_q5_1` (L743-755) | 13 | DELETE | Tests removed legacy dispatch. |
| ↳ `fn dispatch_none_q8_0` (L757-771) | 15 | DELETE | Tests removed legacy dispatch. |
| ↳ `fn dispatch_legacy_ignores_imatrix` (L773-799) | 27 | DELETE | Tests "imatrix silently ignored for legacy"; under ADR-033, legacy impls take `Option<&[f32]>` and ignore `Some(...)` if they don't use it — no separate test needed. |
| ↳ `fn dispatch_legacy_misaligned` (L801-819) | 19 | DELETE | Tests removed error variant. |
| ↳ `fn dispatch_dwq_rejected_for_legacy` (L821-841) | 21 | DELETE | Tests removed DWQ variant. |
| ↳ `fn dispatch_legacy_round_trip` (L843-897) | 55 | DELETE | RMSE-bound smoke; superseded by per-impl byte-cmp against llama.cpp fixture. |
| ↳ `fn synth_row` (L899-910) | 12 | DELETE | test helper. |
| ↳ `fn rmse_pair` (L912-919) | 8 | DELETE | test helper. |
| ↳ `fn decode_via_codec_format` (L921-970) | 50 | DELETE | test helper (routes through `k_quant::dequantize_*` for round-trip); P0 tests compare bytes directly, no dequantize-round-trip pattern. |
| ↳ `fn rmse_bound_for` (L972-988) | 17 | DELETE | RMSE-tolerance table for the dequantize-round-trip pattern; not used under byte-cmp gates. |
| ↳ `fn integration_round_trip_4096_all_formats` (L990-1024) | 35 | DELETE | RMSE-bound integration; superseded by P0 byte-cmp. |
| ↳ `fn integration_round_trip_16384_all_formats` (L1026-1053) | 28 | DELETE | Same. |
| ↳ `fn integration_bpw_matches_on_disk_size` (L1055-1082) | 28 | DELETE | Tests removed `bpw()` accessor; on-disk size matches `GgmlType::bytes_per_block` × block-count by construction in P0. |
| ↳ `fn integration_imatrix_changes_k_family_bytes` (L1084-1113) | 30 | DELETE | Tests imatrix divergence; replaced by P0 imatrix-path byte-cmp. |
| ↳ `fn tensor_2d_q4_k_round_trip` (L1115-1155) | 41 | DELETE | Tests removed 2D wrapper. |
| ↳ `fn tensor_2d_q8_0_round_trip` (L1157-1191) | 35 | DELETE | Same. |
| ↳ `fn tensor_2d_equivalent_to_per_row_loop` (L1193-1227) | 35 | DELETE | Tests removed 2D wrapper's loop-equivalence; under ADR-033 there's only the per-row impl, no separate 2D entry point to compare against. |
| ↳ `fn tensor_2d_rejects_data_length_mismatch` (L1229-1253) | 25 | DELETE | Tests removed error variant. |
| ↳ `fn tensor_2d_rejects_misaligned_row` (L1255-1277) | 23 | DELETE | Tests removed error variant. |
| ↳ `fn tensor_2d_imatrix_applies_to_every_row` (L1279-1321) | 43 | DELETE | Tests removed 2D imatrix-broadcast; under ADR-033 the per-row `Quantizer::quantize` is called per-row by the dispatcher with the same `imatrix: Option<&[f32]>` — broadcast is dispatcher behavior, tested by P0 fixtures. |
| ↳ `fn integration_resolution_ordering` (L1328-1361) | 34 | DELETE | RMSE-ordering check on removed `KQuantTarget`; if a similar invariant is wanted on `GgmlType`, it'd be a separate one-shot bench, not a unit test. |
| ↳ `fn target_all_round_trips_metadata` (L1374-1413) | 40 | DELETE | Tests removed `all()` enumerator. |
| ↳ `fn dispatch_all_targets_smoke` (L1427-1451) | 25 | DELETE | Tests removed dispatcher's exhaustiveness; under ADR-033 `Quantizer` trait dispatch is compile-time-exhaustive via the dispatcher's `match ggml_type` arm. |

**Totals:** DELETE: 51 fns (incl. 35 tests + 2 enums + 14 impl/free fns), 1452 LOC | MODIFY: 0 fns, 0 LOC | KEEP: 0 fns, 0 LOC

**Notes:**

1. **No P0 kernel-port targets in this file.** Every kernel referenced here lives in `src/quantize/k_quant.rs` (K-family ports of `quantize_row_q{2,3,4,5,6}_K_ref` / `_imatrix` at `/opt/llama.cpp/ggml/src/ggml-quants.c:{829,1167,1395,1582,1807}`) or `src/quantize/q_legacy.rs` (legacy ports of `quantize_row_q{4_0,4_1,5_0,5_1,8_0}_ref` at `:{71,108,145,189,234}`). The kernel-vs-utility audit for those files is separate (and is where P0's MODIFY rows for `q4_k.rs` / `q5_k.rs` / `q6_k.rs` / `q4_0.rs` etc. will land).
2. **Decision §1's `GgmlType` enum absorbs `KQuantTarget`'s metadata table.** When P0 writes `src/quantize/ggml_quants/mod.rs` defining `enum GgmlType` + its metadata methods, the verbatim values (84/110/144/176/210 for K-family `bytes_per_block`; 256/32 for `elements_per_block`; the 2.625..8.5 `bpw` table) should be copied across. Those values trace back to `/opt/llama.cpp/ggml/src/ggml-common.h` `static_assert`s — the file we're deleting is not the source of truth, just a Rust mirror of it.
3. **All external callers of this file are themselves DELETE per the broader ADR-033 plan.** `KQuantCodecQuantizer`, `VariantKQuantizer`, `DwqKQuantizer`, `MixedQuantizer`, `StaticQuantizer` are all on the P6 delete list per ADR memo `[[cfa-adr033-review-2026-05-17]]`; `main.rs:2127-2167` and `backends/gguf.rs:1328-4835` dispatch through `METHOD_K_QUANT_CODEC_DIRECT` sentinel which retires when the seek-back writer (P2) + StandardPolicy (P1) ship; `calibrator.rs:675` and `quality/mod.rs:1330-1463` references are inside test/quality-metric modules that compute via the soon-deleted dispatcher. None of this file's public symbols outlives P6.
4. **Caveat — uncertainty surfaced during audit:** I did not chase every call-graph terminal of `KQuantTarget` / `quantize_row_to_bytes` to its final consumer; I relied on the ADR memo classification that the 5 consumer modules (`k_quant_codec_quantizer`, `variant_quantizer`, `dwq_k_quantizer`, `mixed`, `static_quant`) plus the `main.rs`/`gguf.rs` dispatch arms are delete-listed. If P-1's audit of those files surfaces a use case for `KQuantTarget`'s metadata that doesn't fit cleanly on `GgmlType`, a single small `KEEP` could resurface here — but the API surface (especially the `CalibrationData`-keyed dispatch) is structurally incompatible with `Quantizer` trait dispatch, so the dispatcher fns won't survive in any form.
5. **No external pin needed for this file.** No symbol here is a port of any single ggml-quants.c function; the closest analog (`ggml_quantize_chunk` at `/opt/llama.cpp/ggml/src/ggml-quants.c`) is the public dispatch entry whose role is taken by the new `Quantizer` trait + per-type impls.
