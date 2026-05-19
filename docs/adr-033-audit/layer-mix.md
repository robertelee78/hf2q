# Audit: src/quantize/layer_mix.rs (P-1, ADR-033)

Pinned external references: llama.cpp @ c779f6198 (`/opt/llama.cpp/src/llama-quant.cpp`)

File HEAD: 1304 LOC. Total fns (incl. tests + impl methods): 54.
- 5 `impl KQuantVariant` methods (L113–179) + 1 `impl Display` method (L181–187)
- 1 `impl TensorCategory::classify` (L236–303)
- 7 free fns (L317–591)
- 40 `#[test]` fns (L598–1303)

External callers (outside `src/quantize/`) — confirmed by grep:
- `src/main.rs:2226` — `KQuantVariant::Q4_K_M` literal inside the `ImatrixAdaptive` arm. That arm constructs `VariantKQuantizer`, which is on the ADR §P6 delete list, so the call site dies with the construct.
- `src/cli.rs:1264` — doc-comment only (no code).
- `src/backends/gguf.rs:1343–1344`, `src/serve/forward_mlx.rs:9807`, `src/calibrate/dynamic_quant.rs:26` — doc-comment only (no code).

All non-comment callers (`k_quant_codec_quantizer.rs`, `variant_quantizer.rs`, `dwq_k_quantizer.rs`, `mod.rs`) are **inside** `src/quantize/` and are themselves on the ADR §P6 delete list. So the "called from outside src/quantize/" KEEP criterion applies to **zero** functions in this file. `is_vision_tensor_pattern` is KEEP only because the ADR Decision §"Vision tensor pattern" names it the canonical gate.

| Symbol (line range) | LOC | Disposition | Rationale (one line) |
|---|---|---|---|
| `enum LayerMixError` (L61–66) | 6 | DELETE | error type for `KQuantVariant::parse` / `target_for_str`; both DELETE-bound; superseded by `QuantizeError`. |
| `enum KQuantVariant` (L78–111) | 34 | DELETE | superseded by `LlamaFtype` enum per ADR Decision §"LlamaFtype mapping"; `main.rs:2226` literal dies with `VariantKQuantizer` (P6 delete). |
| `impl KQuantVariant::parse` (L115–131) | 17 | DELETE | string parser for the deleted enum; CLI moves to `LlamaFtype` dispatch via `StandardPolicy`. |
| `impl KQuantVariant::base_target` (L135–143) | 9 | DELETE | maps to deleted `KQuantTarget`; `StandardPolicy::target_for` returns `GgmlType` directly. |
| `impl KQuantVariant::name` (L146–159) | 14 | DELETE | name strings for deleted enum; `LlamaFtype` provides its own. |
| `impl KQuantVariant::all` (L165–178) | 14 | DELETE | enumeration helper for deleted enum; test-only consumer. |
| `impl Display for KQuantVariant::fmt` (L184–186) | 3 | DELETE | format for deleted enum. |
| `enum TensorCategory` (L196–234) | 39 | MODIFY | port to `src/quantize/standard_policy.rs`; mirrors llama.cpp's `tensor_category` enum at `/opt/llama.cpp/src/llama-quant.cpp:99–108`. |
| `impl TensorCategory::classify` (L265–302) | 38 | MODIFY | port to `src/quantize/standard_policy.rs`; mirrors `tensor_get_category` at `/opt/llama.cpp/src/llama-quant.cpp:115–150`. |
| `fn should_skip_quantization` (L317–319) | 3 | DELETE | `ffn_gate_inp.weight` skip from `llama-quant.cpp:307`; absorbed into the convert dispatcher before policy dispatch (single line, not worth a fn). |
| `fn should_emit_f16_for_kquant` (L348–350) | 3 | DELETE | composite OR of vision-pattern + row-misalignment; under the ADR no-fallback rule the misalignment arm becomes a typed error (not silent F16), so the composite predicate disappears — dispatcher calls `is_vision_tensor_pattern` directly. |
| `fn is_vision_tensor_pattern` (L366–373) | 8 | KEEP | canonical vision-pattern gate per ADR Decision §"Vision tensor pattern"; move to `src/quantize/vision.rs`, same name; called by the convert dispatcher BEFORE `QuantPolicy::target_for`. |
| `fn is_kquant_row_misaligned` (L382–384) | 3 | DELETE | `row_len % 256 != 0` predicate; trivial; the misalignment policy lives inside each `Quantizer` impl under the no-fallback rule (typed error on second misalignment). |
| `fn use_more_bits` (L391–400) | 10 | MODIFY | port to `src/quantize/standard_policy.rs`; mirrors lambda at `/opt/llama.cpp/src/llama-quant.cpp:417–419`. |
| `fn target_for` (L427–539) | 113 | MODIFY | port to `StandardPolicy::target_for` in `src/quantize/standard_policy.rs`; partial mirror of `llama_tensor_get_type_impl` at `/opt/llama.cpp/src/llama-quant.cpp:411–657`. ADR P1 byte-cmp gate requires expanding this to full coverage (Falcon / MoE-8x / GQA-70B branches currently deferred). |
| `fn target_for_str` (L543–551) | 9 | DELETE | string-parse convenience for the deleted `KQuantVariant`; CLI uses `LlamaFtype` enum directly. |
| `fn kquant_misalignment_fallback` (L578–591) | 14 | DELETE | mirrors `tensor_type_fallback` at `/opt/llama.cpp/src/llama-quant.cpp:362–408` BUT silently downshifts; ADR §"shape_fallback contract" makes the equivalent path return `Err` (no silent demotion). The first-downshift behavior moves into `StandardPolicy::target_for`; the second-misalignment case becomes `QuantizeError`. |
| `#[cfg(test)] mod tests` — 40 `#[test]` fns (L597–1303) | 706 | mixed | Disposition follows the fn under test: tests of MOVE-targets (`target_for`, `use_more_bits`, `TensorCategory::classify`) port to `standard_policy.rs`; tests of `is_vision_tensor_pattern` (6 tests at L1035–1122) port to `vision.rs`; tests of DELETE-bound fns (`KQuantVariant::parse/all/name/base_target`, `target_for_str`, `kquant_misalignment_fallback`, `should_skip_quantization`, `is_kquant_row_misaligned`) DELETE with the fn. |

**Totals (non-test code; LOC counted from definition start to definition end inclusive):**
- DELETE: 12 fns/impl-methods + 1 enum + 1 error enum, 137 LOC
- MODIFY: 1 fn + 1 impl method + 1 enum, 190 LOC
- KEEP: 1 fn, 8 LOC

**Totals (incl. tests, 40 `#[test]` fns at 706 LOC):**
- DELETE: ~13 test fns (estimated; `KQuantVariant::parse/all/name/base_target` tests L598–642 + `target_for_str` test L817–830 + `kquant_misalignment_fallback` test L1124–1155 + `is_kquant_row_misaligned` tests L1056–1077 + `every_vision_pattern_variant_returns_true` L1079–1122 stays with vision) — fine-grained allocation is for the synthesizer.
- MODIFY: majority of test fns (28+ on `target_for` / `TensorCategory::classify` / `use_more_bits`) move with the impl to `standard_policy.rs`.
- KEEP: 6 vision-pattern tests move to `vision.rs`.

**Notes:**
- File has 1304 LOC total; ~55 LOC of module-level doc comments (L1–55) DELETE with the rest of the policy framing.
- The ADR P-1 acceptance criterion that delete-list LOC "sums correctly" requires per-file numbers; this file contributes **roughly 1107 LOC to DELETE, 190 to MODIFY-port, 8 to KEEP-port** (sum + use-statements + blank lines ≈ 1304).
- The `target_for` port to `StandardPolicy` is incomplete vs `llama_tensor_get_type_impl` at HEAD: current file explicitly defers Falcon, MoE-8x, GQA-70B, IQ-family branches (module doc L39–43). P1 byte-cmp gate against `(convert_hf_to_gguf.py | llama-quantize)` will FAIL on any matrix arch hitting those branches until they're added. Flag for synthesizer / P1 plan.
- `kquant_misalignment_fallback` removal is load-bearing for the ADR no-fallback rule (rolls in the §"shape_fallback contract" type-system check). Synthesizer should call out that the DELETE here is paired with adding the `Err` arm in `StandardPolicy::target_for`.
- `should_skip_quantization` (`ffn_gate_inp.weight`) is 1 line of substring check; the synthesizer should place it as a 1-line guard at the dispatcher entry, not a separate module.
- Per-file external-caller summary: ONLY `main.rs:2226` uses any non-comment symbol from this file (`KQuantVariant::Q4_K_M`), and that call site is itself on the P6 delete list.
