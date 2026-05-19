# Audit: src/quantize/mod.rs (P-1, ADR-033)

Pinned external references: llama.cpp @ c779f6198 (`/opt/llama.cpp/ggml/src/ggml-quants.c`)

Scope note: this file is 6440 LOC total but only L1–L593 is production code (~10 fns). L596–L6440 is one `#[cfg(test)] mod tests` block containing ~55 unit tests written against the OLD `Quantizer` trait + `LayerQuantConfig` IR + the existing `VariantKQuantizer / KQuantCodecQuantizer / DwqKQuantizer / MixedBitQuantizer / StaticQuantizer` impls that ADR-033 P6 deletes. None of the test fns test kernels directly; they test the streaming-orchestration wedges and the policy/variant pair-matrices of the to-be-deleted impl classes. Every test fn here is DELETE: its production target is gone in P6.

| Symbol (line range) | LOC | Disposition | Rationale (one line) |
|---|---|---|---|
| `pub trait Quantizer { fn name }` (L61–L63) | 3 | MODIFY | rename + reshape to ADR-033 §2 trait `Quantizer::quantize(src: &[f32], n_per_row, imatrix) -> Result<Vec<u8>>`; lives at `src/quantize/mod.rs` (new home) or `src/quantize/trait.rs` per P0/P1 — same module, different signature |
| `Quantizer::requires_calibration` (L66) | 1 | DELETE | calibration-vs-not is now policy-side (`ApexPolicy` decides imatrix); trait no longer carries this flag |
| `Quantizer::quantize_tensor` (L69–L73) | 5 | MODIFY | replaced by `Quantizer::quantize(src: &[f32], n_per_row, imatrix)` per ADR-033 §2; the `&TensorRef + LayerQuantConfig` shape is superseded by the `QuantizedTensor { ggml_type, data }` IR |
| `pub fn validate_group_size` (L80–L93) | 14 | DELETE | group-size alignment is now decided inside the per-`GgmlType` `Quantizer::quantize` impl (each impl knows its own `n_per_row`); the misalignment-then-downshift policy lives in `StandardPolicy::tensor_type_fallback` per ADR-033 §3 |
| `pub fn quantize_via_streaming_borrowed` (L165–L195) | 31 | DELETE | orchestration wedge built around the OLD `&dyn Quantizer` + `&TensorMap` + `LazyTensorMap` shape; ADR-033 P1+P2 replace this with the policy-driven dispatcher that walks safetensors and feeds the seek-back writer (no `TensorMap`/`LazyTensorMap` intermediary) |
| `pub fn quantize_via_streaming_consuming_mut` (L226–L247) | 22 | DELETE | iter-83 zero-copy wedge for the OLD shape; same reason as `quantize_via_streaming_borrowed` |
| `pub fn quantize_via_streaming_consuming` (L279–L289) | 11 | DELETE | iter-47 wedge for the OLD shape; same reason as `quantize_via_streaming_borrowed` |
| `pub fn quantize_streaming` (L291–L371) | 81 | DELETE | streaming loop built on OLD `Quantizer::quantize_tensor` + `LayerQuantConfig` + `LazyTensorMap` + the `is_vision_tensor()` preserve flag inside the worker; P1/P2 replace with the policy-driven safetensors→writer pipeline (vision-F16 decided OUTSIDE policy at the dispatcher per ADR-033 Decision §3) |
| `pub fn quantize_streaming_parallel` (L415–L538) | 124 | DELETE | rayon-parallel variant of the same OLD-shape loop; same reason as `quantize_streaming`. ADR-033 P2's seek-back writer redesigns the parallelism model around the new IR |
| `pub fn quantize_model` (L541–L593) | 53 | DELETE | original `&TensorMap`-borrowing eager loop on the OLD shape; same reason as `quantize_streaming`. Called from `main.rs:1088` via `dispatch_phase3_quantize`, which itself disappears when P1/P2 land the new pipeline |
| `fn dummy_metadata` (L601–L639) test | 39 | DELETE | test helper for the OLD `quantize_model` / `StaticQuantizer` path |
| `fn make_tensor` (L640–L654) test | 15 | DELETE | test helper for the OLD `quantize_model` / `StaticQuantizer` path |
| `fn quantize_streaming_byte_identical_to_quantize_model` (L655–L736) | 82 | DELETE | tests OLD `quantize_streaming` vs `quantize_model` byte-identity; both production fns deleted |
| `fn quantize_streaming_parallel_byte_identical_to_serial` (L737–L834) | 98 | DELETE | tests OLD parallel-vs-serial byte-identity; both production fns deleted |
| `fn quantize_streaming_parallel_worker_clamp` (L835–L883) | 49 | DELETE | tests OLD `quantize_streaming_parallel` worker-cap arithmetic |
| `fn quantize_streaming_bf16_to_f16_path` (L884–L942) | 59 | DELETE | tests OLD streaming-path bf16→f16 prelude; new pipeline handles dtype at the safetensors reader / per-arch convert mapper (P0) |
| `fn test_vision_tensor_preserved` (L943–L993) | 51 | DELETE | tests OLD vision-preserve-inside-worker behaviour; vision-F16 is now an upstream dispatcher decision per ADR-033 §3 |
| `fn variant_streaming_byte_identical_to_eager` (L994–L1140) | 147 | DELETE | tests `VariantKQuantizer` through OLD streaming pipeline; `variant_quantizer.rs` is in P6 delete list |
| `fn variant_streaming_parallel_byte_identical_to_serial` (L1141–L1270) | 130 | DELETE | same as above, parallel variant |
| `fn variant_imatrix_lowers_importance_weighted_error` (L1271–L1454) | 184 | DELETE | tests `VariantKQuantizer` imatrix-improvement claim; the imatrix subsystem is reborn in Pi against `ApexPolicy::I*` tiers |
| `fn variant_streaming_q5km_q6k_round_trip_rmse_bounds` (L1455–L1674) | 220 | DELETE | RMSE-bound property test against deleted `VariantKQuantizer`; ADR-033 gate is byte-cmp not RMSE |
| `fn variant_streaming_q3km_policy_branches` (L1675–L1755) | 81 | DELETE | tests `KQuantVariant::Q3_K_M` policy branches in deleted `variant_quantizer` |
| `fn variant_streaming_q3kl_policy_branches` (L1756–L1828) | 73 | DELETE | as above for `Q3_K_L` |
| `fn variant_streaming_q4ks_policy_branches` (L1829–L1901) | 73 | DELETE | as above for `Q4_K_S` |
| `fn variant_streaming_q5ks_policy_all_base` (L1902–L1974) | 73 | DELETE | as above for `Q5_K_S` |
| `fn variant_streaming_q2k_policy_branches` (L1975–L2046) | 72 | DELETE | as above for `Q2_K` |
| `fn variant_streaming_q2ks_policy_branches` (L2047–L2125) | 79 | DELETE | as above for `Q2_K_S` |
| `fn variant_streaming_dwq28_policy_branches` (L2126–L2203) | 78 | DELETE | tests deleted `DwqKVariant::P28` policy; `dwq_k_quantizer.rs` + `calibrate/dwq.rs` deleted in P6 |
| `fn variant_streaming_q4km_round_trip_rmse_bound` (L2204–L2320) | 117 | DELETE | RMSE bound on deleted `VariantKQuantizer`; ADR-033 uses byte-cmp |
| `fn variant_imatrix_diverges_from_none_through_streaming` (L2321–L2438) | 118 | DELETE | imatrix divergence property on deleted `VariantKQuantizer` |
| `fn variant_imatrix_q3km_diverges_from_none_through_streaming` (L2439–L2561) | 123 | DELETE | as above for Q3_K_M |
| `fn variant_imatrix_q3km_lowers_importance_weighted_error` (L2562–L2701) | 140 | DELETE | as above for Q3_K_M error claim |
| `fn variant_imatrix_q2k_diverges_from_none_through_streaming` (L2702–L2814) | 113 | DELETE | as above for Q2_K |
| `fn variant_imatrix_q2k_lowers_importance_weighted_error` (L2815–L2965) | 151 | DELETE | as above |
| `fn variant_imatrix_q2ks_diverges_from_none_through_streaming` (L2966–L3080) | 115 | DELETE | as above for Q2_K_S |
| `fn variant_imatrix_q2ks_lowers_importance_weighted_error` (L3081–L3233) | 153 | DELETE | as above |
| `fn variant_streaming_q2k_vs_q2ks_cross_variant_divergence` (L3234–L3385) | 152 | DELETE | cross-variant property on deleted `VariantKQuantizer` |
| `fn variant_streaming_q3km_vs_q3kl_cross_variant_divergence` (L3386–L3532) | 147 | DELETE | as above |
| `fn variant_streaming_q4ks_vs_q4km_cross_variant_divergence` (L3533–L3681) | 149 | DELETE | as above |
| `fn variant_streaming_q5ks_vs_q5km_cross_variant_divergence` (L3682–L3833) | 152 | DELETE | as above |
| `fn variant_streaming_q3ks_vs_q3km_cross_variant_divergence` (L3834–L3986) | 153 | DELETE | as above |
| `fn variant_pair_matrix_base_target_consistency` (L3987–L4138) | 152 | DELETE | matrix consistency property on deleted `VariantKQuantizer` |
| `fn variant_calibrator_matrix_through_streaming` (L4139–L4312) | 174 | DELETE | calibrator-matrix property on deleted `VariantKQuantizer + Calibrator` |
| `fn variant_calibrator_parallelism_matrix_through_streaming` (L4313–L4477) | 165 | DELETE | parallel variant of above |
| `fn legacy_target_streaming_round_trip_rmse_bounds` (L4478–L4593) | 116 | DELETE | RMSE bound on legacy `q_legacy` targets through deleted streaming; q_legacy.rs deleted in P6 |
| `fn legacy_target_parallelism_byte_identity` (L4594–L4719) | 126 | DELETE | parallel byte-identity on deleted streaming |
| `fn legacy_target_non_256_multiple_emits_legacy_not_f16` (L4720–L4839) | 120 | DELETE | tests the legacy-fallback-not-F16 contract through deleted quantizer; ADR-033 §3 reimplements as `StandardPolicy::tensor_type_fallback` |
| `fn legacy_target_vision_tensor_emits_f16_passthrough` (L4840–L4948) | 109 | DELETE | tests inside-worker vision-F16 path; vision-F16 is now upstream of policy per ADR-033 |
| `fn dwq_sensitive_q8_0_layer_with_32_multiple_row_emits_q8_not_f16` (L4949–L5082) | 134 | DELETE | tests deleted `DwqKQuantizer` sensitivity path |
| `fn dwq_vision_tensor_on_sensitive_layer_emits_f16_passthrough` (L5083–L5197) | 115 | DELETE | as above |
| `fn vision_tensor_dual_predicate_path_boundary` (L5198–L5323) | 126 | DELETE | dual-predicate boundary on inside-worker vision-F16 |
| `fn vision_passthrough_paths_parallel_byte_identity` (L5324–L5434) | 111 | DELETE | parallel byte-identity on inside-worker vision-F16 |
| `fn streaming_output_validates_clean_against_gguf_backend` (L5435–L5548) | 114 | DELETE | tests old `gguf.rs:282-1259` two-pass writer; that slice is deleted in P6 |
| `fn iter3_end_to_end_flow_lazy_quantize_quality_validate` (L5549–L5655) | 107 | DELETE | end-to-end through deleted streaming + deleted writer slice |
| `fn quantize_via_streaming_consuming_byte_identical_to_quantize_model` (L5656–L5763) | 108 | DELETE | byte-identity between deleted wedge and deleted eager fn |
| `fn quantize_via_streaming_borrowed_byte_identical_to_quantize_model` (L5764–L5834) | 71 | DELETE | as above for borrowed wedge |
| `fn quantize_via_streaming_borrowed_byte_identical_under_imatrix_variant_kquantizer` (L5835–L5930) | 96 | DELETE | as above under deleted `VariantKQuantizer + imatrix` |
| `fn quantize_via_streaming_borrowed_byte_identical_under_dwq_k` (L5931–L6031) | 101 | DELETE | as above under deleted `DwqKQuantizer` |
| `fn iter3_lazy_pipeline_end_to_end_no_materialize_all` (L6032–L6158) | 127 | DELETE | end-to-end on deleted lazy pipeline; ADR-033 P2 redesigns the pipeline |
| `fn quantize_via_streaming_consuming_mut_byte_identical_to_quantize_model` (L6159–L6245) | 87 | DELETE | as above for consuming_mut wedge |
| `fn variant_menu_smoke_through_streaming` (L6246–L6326) | 81 | DELETE | menu-smoke on deleted streaming + deleted `VariantKQuantizer` |
| `fn variant_streaming_honors_iter3u_moe_classification` (L6327–L6439) | 113 | DELETE | iter3u MoE-classification test on deleted streaming; MoE classification is reborn in Pa via mudler-rule port |

**Totals:** DELETE: 65 fns, ~6422 LOC | MODIFY: 2 fns, 8 LOC | KEEP: 0 fns, 0 LOC

(MODIFY: the `Quantizer` trait declaration + the `quantize_tensor` method header are reshaped, not literally ported; they aren't kernels — they're trait scaffolding. No symbol in `mod.rs` is a per-`GgmlType` kernel; all ggml-quant kernels in hf2q live in sibling files `k_quant.rs` / `k_quant_codec.rs` / `q_legacy.rs` and are audited by sibling agents. The `requires_calibration` method on the trait is dropped.)

**Notes:**

- `mod.rs` is the orchestration / trait-definition layer. It owns no kernel code. P0's per-`GgmlType` ports at `src/quantize/ggml_quants/<type>.rs` come from sibling files (`k_quant.rs`, `k_quant_codec.rs`, `q_legacy.rs`), not from this file.
- External callers of `mod.rs` symbols at HEAD: `src/main.rs::dispatch_phase3_quantize` (uses `Quantizer`, `QuantizeError`, `quantize_via_streaming_consuming_mut`, `quantize_via_streaming_borrowed`, `quantize_model`); `src/quality/mod.rs` (uses `quantize_streaming`); `src/calibrate/dwq.rs` + `src/calibrate/apex.rs` (use `LayerQuantConfig`, `Quantizer`, `QuantizeError`). All five external sites are on doomed code paths: `main.rs::dispatch_phase3_quantize` is the OLD-trait dispatcher (replaced by P1's policy-driven pipeline), `quality/mod.rs`'s callers at L1330–L1465 use `KQuantCodecQuantizer` which is in the P6 delete list, `calibrate/dwq.rs` is explicitly deleted in P6, and `calibrate/apex.rs` is the homebrew apex shell superseded by `ApexPolicy`. So `LayerQuantConfig` truly has no surviving caller — DELETE is correct.
- `QuantizeError` (the enum at L29–L45) and its variants are not `fn`s so they're outside the audit scope per instructions. The new pipeline introduces typed `QuantizeError::NoQuantizerForType` (ADR-033 §6) and friends in the rewritten module; the present variants (`TensorQuantizeFailed`, `UnsupportedMethod`, `GroupSizeMismatch`, `IrError`) are tied to deleted callers and would naturally be replaced when the trait is reshaped.
- All test fns share their fate with their production target. None of the existing tests should be ported as-is to P0/P1; the ADR mandates byte-cmp against llama.cpp reference fixtures (P0 AC) and against `(convert_hf_to_gguf.py | llama-quantize)` end-to-end (P1 AC) — those are new test surfaces, not adaptations of the property/RMSE tests in this file.
- Edge case for synthesizer: if P0/P1 land before P6 and a transitional period needs `Quantizer` + `QuantizeError` to coexist with the new trait, treat the MODIFY rows as "replace in place; same path" rather than "port to ggml_quants/<type>.rs" (since they aren't kernel code).
