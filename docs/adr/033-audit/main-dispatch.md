# Audit: src/main.rs quant-dispatch regions (P-1, ADR-033)

Pinned external references: llama.cpp @ `c779f6198` (`/opt/llama.cpp/tools/quantize/quantize.cpp`, the canonical CLI parser whose semantics we collapse 5 dispatch arms into), mudler/apex-quant @ `63c5048b7dc9ff230f2397d7bc445ca28894b769` (the recipe `ApexPolicy` ports).

Audited file: `/opt/hf2q/src/main.rs` HEAD `ebecc21c` (3453 LOC total; only quant-relevant regions enumerated below; CLI bootstrap / serve / unrelated subcommands collapsed to one KEEP row).

ADR target shape: `Quantizer` trait (Decision §"Quantizer trait") + `QuantPolicy` impls `StandardPolicy` / `ApexPolicy` (Decision §"QuantPolicy trait") + `QuantizedTensor { ggml_type, data }` IR (Decision §"Per-tensor IR") + seek-back incremental writer (Decision §5) + Decision §6 CLI surface.

| Region (line range) | LOC | Disposition | Rationale (one line) |
|---|---|---|---|
| `fn main` / `fn run` / bootstrap, error printing, signal handlers, logging init (L1-196, L3163-3170, L3261-3300, L3365-3441) | ~390 | KEEP | CLI bootstrap, sidecar copy (Decision 15), `cmd_info` / `cmd_completions` / interrupt plumbing — orthogonal to quant pipeline. |
| `fn cmd_dwq_overlay_drift` (L197-307) | 111 | DELETE | Homebrew-DWQ debug subcommand; `--quant dwq` reserved per Decision §6; whole homebrew-DWQ surface deleted in P6 (ADR §"Explicitly NOT doing"). |
| `fn cmd_dwq_train` (L315-519) | 205 | DELETE | Standalone DWQ-train subcommand; same homebrew-DWQ delete-list as P6 (`calibrate::dwq_loop`, `DwqTrainingConfig`). |
| `fn cmd_dwq_train_full_model` (L521-724) | 204 | DELETE | Full-model-teacher DWQ-train; same homebrew-DWQ delete-list as P6. |
| `fn cmd_gguf_patch` (L726-746) | 21 | KEEP | Sidecar GGUF KV patcher; not part of convert/quant pipeline. |
| `fn cmd_smoke` (L748-784) | 37 | KEEP | Smoke-test subcommand; unrelated to the policy/trait shape. |
| `enum CaptureSpec` + doc (L786-802) | 17 | DELETE | Only consumed by `select_calibrator` and DWQ lazy-capture seam; calibrator concept replaced by `Quantizer` trait + `QuantPolicy::requires_imatrix()` (ADR Decision §"Quantizer trait"). |
| `fn select_calibrator` (L804-912) | 109 | DELETE | 17-variant Calibrator-dispatch (DWQ / Imatrix / None); collapses to `match name { standard => StandardPolicy, apex-* => ApexPolicy }` per Decision §6. Imatrix-capture wiring moves to Pi imatrix subsystem; Calibrator trait deleted in P6. |
| `fn build_calibration_corpus` (L914-955) | 42 | DELETE | Synthetic deterministic-token corpus for DwqCalibrator/ImatrixCalibrator; replaced by Pi's real corpus loader at `src/quantize/imatrix/` consuming `data/calibration/{cdv3,mudler_v1}.txt` (Decision §7). |
| `fn dwq_calibration_to_sensitive_ranges` (L957-1018) | 62 | DELETE | Bridges DwqCalibrator output → DwqConfig.sensitive_layers; both endpoints deleted with homebrew DWQ in P6. |
| `fn clone_tensor_map_to_lazy` (L1020-1038) | 19 | DELETE | Sole consumer is DWQ-on-qwen35 lazy-capture path; goes with DWQ delete. |
| `fn dispatch_phase3_quantize` (L1040-1092) | 53 | DELETE | Wraps `quantize_model` / `quantize_via_streaming_borrowed` / `quantize_via_streaming_consuming_mut` behind `HF2Q_STREAMING_PHASE3{,_MUT}` env vars; the seek-back single-pass writer (Decision §5, P2) replaces all three paths with one streaming write. Env-gated wedges retire. |
| `fn cmd_convert` signature + early pre-flight (L1094-1194) | 101 | MODIFY | Convert entry point survives; loses off-diagonal calibration/output-format selector (Decision 12 lock deleted with calibrator), keeps config-resolve / metadata-parse / dry-run / Ctrl-C cleanup. |
| Calibrator preview seam inside `cmd_convert` (L1195-1250) | 56 | DELETE | Diagnostic `select_calibrator(...).preview` call; goes with `select_calibrator`. |
| RuVector + hardware profiling + auto-resolution preamble (L1252-1308) | 57 | KEEP | Telemetry / intelligence; orthogonal to quant shape. |
| AutoResolver `match resolved.quant_method.as_str()` (L1310-1351) | 42 | MODIFY | Becomes the Decision §6 surface: `q4_0/q4_1/q5_0/q5_1/q4_k_s/q4_k_m/q5_k_s/q5_k_m/q6_k/q8_0/iq4_nl/f16/f32/bf16` + `apex-{quality,i-quality,balanced,i-balanced,compact,i-compact,mini,custom}`. Drops `imatrix-q*`, `dwq*`, `dynamic-quant-*`, `auto`-as-passthrough; `dwq` / unqualified `apex` / `tq*` arms become typed-error stubs per Decision §6 "Reserved" / "Out of v1 scope". |
| `--bits` × DWQ guard (L1386-1394) | 9 | DELETE | DWQ reserved per Decision §6; `--bits` semantics unchanged for StandardPolicy. |
| Preflight + dry-run + Ctrl-C handler + ProgressReporter (L1396-1900) | ~505 | KEEP | Preflight validation, dry-run plan emit, Ctrl-C cleanup, Phase-0/1/2 read-and-transform; orthogonal to quant trait shape. (Some local references to `config.quant` survive; signatures unchanged.) |
| DWQ-arch + lazy-capture build (L1918-1984) | 67 | DELETE | `dwq_arch_for_dispatch`, `needs_dwq_lazy_capture`, `capture_spec`, `select_calibrator(...)` invocation; whole DWQ+Imatrix-via-Calibrator path goes. Replaced by `policy.requires_imatrix()` gate that drives Pi imatrix runner. |
| Calibrator cache + calibrate() invocation (L1986-2070) | 85 | DELETE | DWQ sensitivity-cache hoist + `clone_tensor_map_to_lazy` + `calibrator.calibrate(...)`; superseded by Pi imatrix subsystem (in-memory `.imatrix.gguf` payload — Decision §7). |
| `drop(calibrator)` boundary (L2072-2091) | 20 | DELETE | Explicit drop of GPU+host weights; loses meaning when Calibrator trait is gone. (Equivalent peak-memory hand-off lives inside Pi imatrix runner.) |
| Phase 3 dispatch — K-quant codec arm (L2097-2218) | 122 | DELETE | `KQuantCodecQuantizer::new(...)` + `KQuantTarget::{Q2K,Q3K,Q4K,Q5K,Q6K}` mapping + `dispatch_phase3_quantize` — entire arm replaced by single `for tensor in tensors { writer.append(policy.quantizer_for(tensor)?.quantize(...)?) }` loop in P2 seek-back writer. |
| Phase 3 dispatch — `ImatrixAdaptive` arm (L2219-2258) | 40 | DELETE | `VariantKQuantizer::new(...)` + `KQuantVariant::Q4_K_M` + per-tensor `layer_mix::target_for`; superseded by `ApexPolicy::target_for` (Decision §"QuantPolicy trait"). The "Apex per-tensor optimal precision" comment confirms this WAS the proto-Apex path — `ApexPolicy` replaces it with the ported mudler recipe. |
| Phase 3 dispatch — DWQ arm (L2259-2468) | 210 | DELETE | `DwqKVariant::{P46,P48,P68,P28}` + `DwqKQuantizer::new(...)` + `MixedBitQuantizer` legacy path via `HF2Q_USE_LEGACY_DWQ_Q4_0=1`; `--quant dwq` reserved per Decision §6; all three quantizers (`DwqKQuantizer`, `MixedBitQuantizer`, `dwq_activation`) on P6 delete-list. |
| Phase 3 dispatch — `StaticQuantizer` arm (L2469-2510) | 42 | DELETE | `StaticQuantizer::new(quant_method_str)` for `Auto/F16/Bf16/Q2/Q4/Q8`; replaced by per-`GgmlType` `Quantizer` impls in `src/quantize/ggml_quants/{f16,bf16,q8_0,...}.rs` (ADR P0). |
| Phase 4/4.5/4.6 — bit_overrides, quality measure, write (L2512-2683) | 172 | MODIFY | Bit-override map computation (from auto_plan / `--sensitive-layers`) deleted-or-collapsed (per-tensor `_M` upgrades move inside `StandardPolicy::target_for`); `quality::measure_quality_streaming{,_lazy}` survives unchanged; `backend.write` becomes the new seek-back writer entry point (P2). |
| Phase 4.7-4.85 — sidecars, mmproj sha256 patch (L2685-2847) | 163 | KEEP | mmproj emission + SHA-256 sidecar patching; orthogonal to quant shape. |
| Phase 5+ — quality gate, RuVector store, JSON report, summary (L2849-3014) | 166 | KEEP | Quality gate + reporting; orthogonal. |
| `fn cmd_validate` (L3017-3160) | 144 | KEEP | Post-conversion validate subcommand; reads original + quantized, calls `quality::measure_quality`; unchanged by trait shape. |
| `fn check_interrupted` (L3163-3169) | 7 | KEEP | Signal-handler plumbing. |
| `fn print_dry_run_plan` (L3172-3237) | 66 | MODIFY | Prints `config.quant` + `bits` + `group_size`; signature unchanged, body trims `group_size` line if DWQ-specific (already KEEP-shaped for StandardPolicy/ApexPolicy). |
| `const SIDECAR_FILES` + `fn copy_sidecars` (L3245-3299) | 55 | KEEP | Decision 15 (carries forward unchanged from ADR-014). |
| `fn detect_quant_method_from_path` (L3302-3331) | 30 | MODIFY | Hard-coded scan-list for the 17-variant menu in output filenames; becomes Decision §6 list (`q4_0/.../iq4_nl/f16/bf16/apex-{tier}`), drops `imatrix-q*` / `dynamic-quant-*`. |
| `fn quantizer_default_bits` (L3339-3362) | 24 | MODIFY | Exhaustive match over the 17-variant menu; collapses to a one-line `ggml_type.bits()` call once `cli::QuantMethod` becomes the Decision §6 enum (the `_M` upgrades / imatrix-adaptive 4-bit-base notion goes away — bits is now a property of the resolved `GgmlType`, not the CLI variant). |
| `fn cmd_info` + `fn resolve_info_input` + `fn cmd_completions` + `fn extract_baseline_from_reasoning` (L3365-3453) | 89 | KEEP | Info / shell-completion / RuVector baseline extraction; orthogonal. |

**Totals:**
- **DELETE:** 17 regions, 1413 LOC (3× cmd_dwq_* subcommands 520 + calibrator/DWQ helpers 282 + cmd_convert quant-dispatch interior 826 + CaptureSpec 17 less double-counting — actual sum: 111+205+204+17+109+42+62+19+53+56+9+67+85+20+122+40+210+42 = 1473).
- **MODIFY:** 6 regions, 308 LOC (cmd_convert preamble 101 + AutoResolver map 42 + Phase 4 prelude 172 minus quality/sidecar carve-out → ~133 + dry-run-plan 66 + detect_quant_method 30 + quantizer_default_bits 24; pessimistic sum ~395).
- **KEEP:** 11 regions, ~1577 LOC (bootstrap 390 + cmd_gguf_patch 21 + cmd_smoke 37 + preflight/Phase-0/1/2 ~505 + mmproj 163 + Phase-5+ 166 + cmd_validate 144 + check_interrupted 7 + sidecars 55 + cmd_info-family 89).
- Sanity: 1473 + 395 + 1577 = 3445 ≈ 3453 LOC file size (small slack from blank lines / module-level imports counted once).

## Notes / follow-ups

1. **Decision §6 CLI surface absences in current code.** The Decision §6 surface includes `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `iq4_nl`, `f32`, `bf16` — none of these have an arm in `cmd_convert`'s Phase 3 match today. `StaticQuantizer` covers `f16/bf16/q2/q4/q8` only. The new `StandardPolicy` path needs `Quantizer` impls for the legacy block types (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/IQ4_NL) per ADR P0 — backing-out of the current main.rs is a strict subset of what P0 ports forward.
2. **Reserved / out-of-v1-scope CLI values that need typed-error stubs (Decision §6).** None of these exist in current `cli::QuantMethod`:
   - `dwq` (reserved for future ADR — real-DWQ Apple MLX port).
   - `apex` (unqualified — must spell tier explicitly per ADR-014 P8 D13).
   - `tq1_0`, `tq2_0` (BitNet ternary; out of v1 scope).
   - `apex-nano`, `apex-i-nano`, `apex-micro`, `apex-i-micro` (mudler experimental; v1 drops; reachable via `apex-custom`).
3. **Current `cli::QuantMethod` values that Decision §6 doesn't list.** These all become DELETE in P6 and need typed-error mapping during the transition window (so a stale shell history of `--quant dynamic-quant-4-6` produces "DWQ is reserved for a future ADR" rather than a panic):
   - `Auto` — Decision §6 has no `auto`; AutoResolver still emits a concrete `<name>` so the CLI surface drops the literal `auto` user-input. Auto-mode-as-implementation survives behind the scenes.
   - `Q2`, `Q4`, `Q8` (flat legacy without `_K` suffix; subsumed by `q4_0/q8_0`).
   - `Q2KS`, `Q2K`, `Q3KS`, `Q3KM`, `Q3KL` (out-of-v1 LlamaFtype holes 10/11/12/13; ADR §"LlamaFtype mapping" defers them).
   - `ImatrixQ2KS`, `ImatrixQ2K`, `ImatrixQ3KS`, `ImatrixQ3KM`, `ImatrixQ3KL`, `ImatrixQ4KS`, `ImatrixQ4KM`, `ImatrixQ5KS`, `ImatrixQ5KM`, `ImatrixQ6K`, `ImatrixAdaptive` — `imatrix-*` as a CLI prefix is gone; imatrix is `--imatrix <file> | --imatrix-corpus <name>` per Decision §6.
   - `DynamicQuant46`, `DynamicQuant48`, `DynamicQuant68`, `DynamicQuant28` — DWQ reserved.
4. **Vision-tensor F16 gate lives outside main.rs.** Per ADR Decision §"Vision tensor pattern", `crate::quantize::layer_mix::is_vision_tensor_pattern` is the only place vision-pattern membership is decided; the convert dispatcher calls it BEFORE `QuantPolicy::target_for`. main.rs has no inline vision-pattern check today (the check happens inside `quantize_model` / `KQuantCodecQuantizer`); the new dispatcher in P2's writer takes ownership.
5. **HF2Q_STREAMING_PHASE3 / HF2Q_STREAMING_PHASE3_MUT / HF2Q_USE_LEGACY_DWQ_Q4_0 env vars all retire** with the seek-back writer landing. The "MUT × Phase 4.5 incompatible" guard at L1149-1174 (26 LOC, inside the cmd_convert MODIFY row above) goes with them.
6. **Source-dtype handling (Decision §10 — FP8 auto-detect) is absent from main.rs today.** P0 adds `src/convert/source_dtype/fp8.rs`; main.rs's Phase 0 metadata parse stays as-is; the FP8 handling lives inside `convert::arch::<arch>::load_tensor` (not at the dispatch layer).
7. **The `(config.calibration, config.output_format)` off-diagonal selector at L1116-1138** (inside cmd_convert MODIFY row) — operator-controlled dev gate (`HF2Q_UNSAFE_EXPERIMENTS`) for "imatrix + safetensors-out" style crosses; this concept disappears with the calibrator surface. The new policy/format pair is `--quant <name> + -o <path>` only.
