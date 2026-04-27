# ADR-014: Streaming convert pipeline + peer-parity gates (cross-arch)

**Status:** 🟡 **PROPOSED 2026-04-25 (round-2 refined)** — pending Robert sign-off; refined across two party-mode sessions (round 1: `conversion_fixes`, ten questions, eight strategic axes; round 2 today: 12 additional Robert-locked refinements — see "Round-2 refinement log" right after the Phase status table). Ready for P0 to start **after ADR-012 P9 real close** (R14 — the four `models/qwen3.6-*-dwq*` GGUFs must verifiably load in `llama-cli` first). Per mantra (`~/Documents/mantra.txt`, 2026-04-07): "DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Chesterton's fence: always understand current fully before changing it." Every decision below is engineering-executable; every phase has concrete deliverables, ACs, and LOC estimates; every risk has a mitigation that is real work, not a gate. This ADR closes only when **all** peer-parity gates are measured green on the four ADR-012 reference artifacts re-emitted under the streaming pipeline.

---

## Phase status

| Phase | Status | Commit | Notes |
| ----- | ------ | ------ | ----- |
| P0 — Lazy tensor primitive + lazy safetensors reader | 🟢 code-green; on-day apex-MoE iter spike pending | `c707a2c` (lazy primitive), `038f2ab` (lazy reader + bridge), `<spike-commit>` (apex-MoE iter spike) | LazyTensor + LazyTensorMap + lazy safetensors reader land; 13 unit + 4 reader + 1296 full-bin tests pass; eager bridge byte-identical to pre-ADR-014. Decision 6 gate value locked at **36.3 GB** (33 GB ADR-012 P9b inherited + 10% headroom) per `docs/peer-parity-baselines-2026-04-26.md`. Decision 2 `≤ 8 GB` LazyTensorMap-iter spike on apex MoE pending (next iter). |
| P1 — Lift Phase 1.4–1.7 transforms to lazy | 🟢 closed | `b6a4b82` (1.4+1.42), `04ee984` (1.45), `5a773d0` (1.5+Decision 7), `295ffd5` (1.6), `ef7b550` (1.7+1.8 close) | Every Phase 1.x transform takes `&mut LazyTensorMap`. Decision 7 layer-streaming MoE merge implemented; per-merge tile materialise→quantise→write→drop bounds peak resident bytes to ~one tile (~750 MB apex BF16) instead of ~80 GB stack. 9 byte-identity unit tests + 1309 full-bin tests passing. Eager helpers retained for the P9b dance until P4 deletes both. |
| P2 — Streaming quantize loop (per-tensor write-and-drop) | 🟢 iter-2 landed (commit `1b778b9`); cmd_convert wired through Calibrator + KQuantCodecQuantizer dispatch — every Decision 12 variant routes correctly. iter-3 (StreamingBackend production wiring + per-tensor write-and-drop) remains pending. | `58e3144` (iter-1 — quantize_streaming function + LazyTensorMap::from_eager bridge), `1b778b9` (iter-2 — cmd_convert Phase 2 dispatch wires Calibrator + KQuantCodecQuantizer + VariantKQuantizer + DWQ-via-DwqCalibrator end-to-end. select_calibrator now drives every --quant variant. The legacy in-main.rs capture orchestration moved into DwqCalibrator (added `with_activation_capture` deferred-build constructor); main.rs DWQ arm collapses from ~280 LOC to ~30 LOC. cmd_convert holds 0 callers of `RealActivationCapture::new` / `capture_activations_to_sensitive_ranges`. Byte-identity gate for --quant f16 verified across two runs. 9 new tests across `tests/cmd_convert_dispatch.rs` (T1-T7 incl. Decision-17 byte-identity + no-silent-fallback gate) + `tests/quant_method_dispatch.rs` (T8-T9 routing partition).) | `quantize_streaming(LazyTensorMap, ..., bf16_to_f16: bool)` consumes a LazyTensorMap one tensor at a time — materialise → optional bf16→f16 cast → quantise → accumulate → drop. Byte-identical to `quantize_model` (verified by 2 unit tests). iter-2 (2026-04-27): `cmd_convert` Phase 2 dispatch restructured — `select_calibrator` builds the right `Calibrator` (NoneCalibrator / DwqCalibrator / ImatrixCalibrator) before the per-quantizer match; for DWQ on `Qwen35Dense`/`Qwen35MoE` the `DwqCalibrator::with_activation_capture` deferred path emits the intermediate F16 GGUF + drops `tensor_map` + builds `RealActivationCapture` inside `calibrate(...)` + drops it + re-reads `tensor_map` via the colocated `reread_qwen35_tensor_map_after_capture` helper. The `CalibrationData` then flows into `KQuantCodecQuantizer::new(..., calibration_data)` / `VariantKQuantizer::new(..., calibration_data, n_layers)` / the `DWQ` byte-emit through `run_dwq_with_sensitive_ranges`. Greppable invariants: `RealActivationCapture::new` and `capture_activations_to_sensitive_ranges` have **0 callers** in `src/main.rs`; `run_dwq_calibration` has **0 references** in `src/main.rs`; `select_calibrator` appears at exactly **2 sites** (preview + live). StreamingBackend trait + GgufBackend refactor pending iter-3. |
| P3 — Rayon parallelism in quantize loop | 🟢 closed | `fdd0375` (quantize_streaming_parallel) | `quantize_streaming_parallel` distributes per-tensor quantize across rayon thread pool sized `min(available_parallelism, 16)`. Byte-identical to serial across n_workers {1, 2, 4, 8}; per-tensor shape/dtype/bytes byte-equal. Worker clamp tested ({0 → 1, 100 → 16}). Memory: serial ~750 MB peak input → parallel n=8 ~6 GB / n=16 ~12 GB. |
| P4 — Eliminate the P9b intermediate-GGUF dance | 🟢 iter-1 landed | `<pending commit>` | DWQ qwen35/qwen35moe activation capture now builds `RealActivationCapture` from a transformed `LazyTensorMap` through `DwqCalibrator::with_activation_capture_lazy`; the temporary GGUF emit helper and P9b quantizer workaround are deleted. |
| P5 — Sensitivity-JSON cache (DWQ across bit-pair variants) | 🟢 wiring landed | `18333ff` (cache module) | Pure-Rust cache at `${XDG_CACHE_HOME:-$HOME/.cache}/hf2q/sensitivity/<sha>.json`. Cache key = hex SHA-256(model_sha \| corpus_sha \| algorithm_version). `SENSITIVITY_ALGORITHM_VERSION = "1.0.variance-magnitude"` — bumped on algorithm change to invalidate stale entries. Atomic write via temp + POSIX rename. DWQ and imatrix calibrators now load before forward pass and save after capture; corrupt/schema-mismatched cache entries warn and recompute without failing calibration. |
| P6 — Imatrix calibrator (pure-Rust port) | 🟡 algorithm + legacy I/O + GGUF I/O write+read landed; **iter-1 gate machinery landed (commit `0920f7c`)**; cross-validation gate cell pending iter-2 wires real Qwen3.5-0.6B fixture | `511d35c` (algorithm core), `2577d89` (legacy .imatrix I/O), `d272005` (P6 iter-3 — GGUF format write per llama.cpp PR #9400 / commit 90083283 / 2025-07-19), `52d6386` (P6 iter-4 — GGUF format read with full save→load round trip incl. MoE per-expert counts), `0920f7c` (P6 close iter-1 — cross-validation gate machinery: `cross_validate_imatrix_gguf` + `XValidationReport` + `TensorComparison`) | `ImatrixCollector` with dense (`GGML_OP_MUL_MAT`) + MoE (`GGML_OP_MUL_MAT_ID`) accumulators; `Stats { values, counts }` invariant; `finalise()` produces per-column importance vectors. Legacy `.imatrix` save/load byte-for-byte matches `imatrix.cpp::save_imatrix_legacy` + `load_imatrix_legacy`. **GGUF format write+read** matches the schema landed in llama.cpp PR #9400 (commit `90083283`, 2025-07-19): GGUF v3 magic + 4 metadata KVs (`general.type`, `imatrix.datasets`, `imatrix.chunk_count`, `imatrix.chunk_size`) + 2 F32 tensors per stat entry (`<name>.in_sum2` 2D + `<name>.counts` 2D), 32-byte aligned data section. Reader walks metadata KVs (skipping unknown types via typed payload-size dispatch), pairs `<name>.in_sum2` + `<name>.counts` descriptors, reads tensor data at recorded offsets, reconstructs `Stats { values, counts }` exactly (MoE per-expert counts preserved — addresses the lossy-collapse known issue of the legacy format). 25 unit + round-trip tests (15 legacy + 6 GGUF write + 4 GGUF round-trip/read). Pending: cross-validation gate against `llama-imatrix` binary (lands at P7 close when Calibrator trait wires forward-pass). **iter-1 (2026-04-27, commit `0920f7c`)**: cross-validation gate machinery: `src/calibrate/imatrix_xvalidate.rs` (`cross_validate_imatrix_gguf` + `XValidationReport` + `TensorComparison`); reuses existing `load_imatrix_gguf` (no new GGUF reader); `is_pass()` predicate gates on (a) tensor-presence equality, (b) per-tensor `max_abs_diff_in_sum2 ≤ abs_tolerance OR max_rel_diff_in_sum2 ≤ rel_tolerance`, (c) exact `counts` match. Default tolerances `abs=1e-3, rel=1e-2` justified by the round-trip RMSE bounds locked in P7 iter-3x/3y (Q4_K ≤ 0.05, Q5_K ≤ 0.025, Q6_K ≤ 0.012). 5 always-on tests cover (a) self-compare passes, (b) perturbed in_sum2 fails at tight tolerance, (c) counts mismatch fails, (d) missing tensor fails, (e) markdown report shape lock. 1 `#[ignore]`-gated cell `xvalidation_vs_llama_imatrix_qwen35_smoke` is iter-2's hardware gate (Qwen3.5-0.6B + small corpus + `llama-imatrix` at `/opt/homebrew/bin/llama-imatrix` + our `ImatrixCalibrator` through `Qwen35Dense::forward`). cargo build `0 errors`; cargo test `--bin hf2q --release` 2049 passed (+6 from baseline 2043); `cargo test --release --test imatrix_xvalidation` 5 always-on tests + 1 ignored pass cleanly (integration crate also re-exercises the `#[path]`-included `imatrix.rs` + `imatrix_xvalidate.rs` inner unit tests, totalling 36 passed + 1 ignored — all pre-existing and unchanged). cargo clippy `0 errors / 388 warnings` (within ≤ 395 budget). |
| P7 — `Calibrator` × `OutputFormat` orthogonal split | 🟡 ImatrixCalibrator + DwqCalibrator landed; Layout A migration complete; cmd_convert minimal-wire seam in place — full P7 closure depends on remaining iter-3* k-quant byte-identity gate. iter-8 (`b5da618`): Layout A migrated; ImatrixCalibrator + DwqCalibrator landed | `33081aa` (Calibrator trait + NoneCalibrator + CalibrationData enum), `ac3ebf2` (P7 iter-3a — Q4_K block layout + dequantize), `ade910c` (P7 iter-3c — Q5_K + Q6_K block layouts + dequantize, co-landed with parallel ADR-005 P3 iter-203 due to staged-index sweep), `ebee4e6` (P7 iter-3b1 — `nearest_int` + `make_qkx2_quants` + `quantize_row_q4_k` codebook quantize), `6440b4e` (P7 iter-3b2 — `quantize_row_q5_k` codebook quantize, reuses `make_qkx2_quants` at `nmax=31`), `1c37488` (P7 iter-3b3 — `make_qx_quants` symmetric codebook + `quantize_row_q6_k`), `b27afa7` (P7 iter-3d — `make_qkx3_quants` + `make_qp_quants` + `quantize_row_q4_k_imatrix` for imatrix-weighted Q4_K), `93415ad` (P7 iter-3e — `quantize_row_q5_k_imatrix` + `quantize_row_q6_k_imatrix`, completes the imatrix-weighted Q4/5/6_K coverage), `17def7e` (P7 iter-3f — flat-bytes wrappers `quantize_row_q*_k[_imatrix]_to_bytes` for direct GGUF emission), `c9c9d51` (P7 iter-3g — `k_quant_codec` calibration-aware dispatch over Q4/5/6_K with `KQuantTarget` enum + `quantize_row_to_bytes(row, target, calib, name)` entry point), `dd9cec3` (P7 iter-3h — `q_legacy` module: Q8_0 + Q4_0 ports for the K-family fallback chain), `61f6d3e` (P7 iter-3i — Q5_0 + Q5_1 ports complete the legacy fallback chain), `5e749a2` (P7 iter-3j — k_quant_codec extended to dispatch over legacy formats; 7-target enum), `253a0cd` (P7 iter-3k — end-to-end integration tests at realistic tensor shapes 4096+16384 with 7-format ordering verification), `b260917` (P7 iter-3l — `quantize_tensor_2d_to_bytes` multi-row helper for full weight matrices), `3115a75` (P7 iter-3m — `KQuantCodecQuantizer` impl wires the codec into the existing `Quantizer` trait machinery), `d2bdc37` (P7 iter-3n — Q4_1 port completes the legacy block-format coverage), `e2a6fd0` (P7 iter-3o — codec & quantizer extended with Q4Legacy1 variant for full 8-target coverage), `eaa26cc` (P7 iter-3p — `CalibrationData::from_imatrix_gguf` + `from_imatrix_collector` bridge connects loaded GGUF imatrix files directly to the codec dispatch), `2a90d3e` (P7 iter-3q — end-to-end pipeline integration test demonstrates the full chain: collector → save_imatrix_gguf → from_imatrix_gguf → KQuantCodecQuantizer → GGUF block bytes), `2badf16` (P7 iter-3r — `layer_mix` module: per-tensor target dispatch for K-quant `_M`/`_S` variants per the documented subset of `llama_tensor_get_type_impl`), `ce01396` (P7 iter-3s — `VariantKQuantizer` composes `layer_mix` policy with `k_quant_codec` so each tensor auto-picks its M/S target), `f182f76` (P7 iter-7 — **Q3_K quantize-side port** from `ggml-quants.c:1293-1374`: `quantize_row_q3_k()` + `quantize_row_q3_k_imatrix()` + flat-bytes wrappers; algorithm = per-sub-block `make_qx_quants(nmax=4)` + super-block `make_qx_quants(nmax=32)` to quantize scales themselves to 6 bits + 12-byte scale packing + per-element re-quantization to L ∈ [0,7] + hmask packing (8 elements per bit position, 32-stride bands) + qs packing (4 quants per byte, two halves); `KQuantError::WeightsLengthMismatch` variant added; 7 new tests including round-trip RMSE ≤ 0.10 on smooth ramp, Q3_K vs Q4_K monotonic-bits gate, imatrix-vs-none divergence, multi-block round-trip), `9e7fb64` (P7 iter-6 — **Q3_K block layout + dequantize port** from llama.cpp `ggml-common.h:301-311` + `ggml-quants.c:1243-1291`: `BlockQ3K` struct (110 bytes: hmask[32] + qs[64] + scales[12] + d:f16 = 3.4375 bpw); `unpack_q3_k_scales()` ports the kmask manipulation that decodes 16 packed 6-bit scales; `dequantize_row_q3_k()` and `dequantize_row_q3_k_bytes()` implement the per-element decode where Q ∈ [-4, +3] is split between qs[] low-2-bits and hmask[] high-bit (hmask set → Q ∈ {0..3}; clear → Q ∈ {-4..-1}); 11 unit tests including hand-crafted fixtures that lock the dequant arithmetic. Quantize-side port + Q3_K_M/Q3_K_S variant menu addition deferred to iter-7), `e7440a0` (P7 iter-5 — `KQuantTarget::all()` enumeration helper + two codec-coverage smoke tests: `target_all_round_trips_metadata` (asserts every target round-trips through `ggml_type`/`from_ggml_type` + non-zero metadata + correct `supports_imatrix` flag) and `dispatch_all_targets_smoke` (asserts `quantize_row_to_bytes` produces exactly one block of `bytes_per_block` size for every target on a smooth-ramp input); future-proofs codec coverage so adding a new target to the enum requires updating `all()` AND the dispatch + the metadata methods, with the test surface catching missing wiring), `857b8fd` (P7 iter-4 (co-landed under parallel ADR-015 commit) — `variant_streaming_parallel_byte_identical_to_serial` test extends iter-3t's serial-vs-eager byte-identity check to cover `quantize_streaming_parallel` (rayon work distribution); 6-tensor fixture covers Q6_K bump + Q4_K base + MoE-expert + 1-D preserve branches; compile-time `Send + Sync` guard on `VariantKQuantizer` ensures rayon-eligible bound is preserved against future field additions), `d32e656` (P7 iter-3z — imatrix-improves-importance-weighted-error gate: with 10000× importance ratio on first 16 columns and adversarial input (wide-range high-importance cols, small-range low-importance cols), imatrix-weighted Q4_K_M produces strictly lower importance-weighted SSE than uncalibrated AND ≥ 5% lower SSE on the high-importance subset alone — closes the divergence-direction gap left by iter-3w which only proved bytes differ; catches a regression that flips the codebook search to minimise the wrong objective or inverts the importance vector), `489b4d7` (P7 iter-3y — round-trip RMSE bounds extended to Q5_K_M (≤ 0.025) and Q6_K (≤ 0.012) through `quantize_streaming` + matching `dequantize_row_q*_k_bytes`; parameterized over the variant menu so any future variant added to `KQuantVariant::all()` can be slotted into the same harness with a one-line case entry; closes the dequant-coverage gap left by iter-3x's Q4-only test), `c11e7df` (P7 iter-3x — round-trip dequant RMSE bound (≤ 0.05) on Q4_K_M output through `quantize_streaming` + `dequantize_row_q4_k_bytes`, on a 4-super-block smooth ramp; closes the quantize→dequantize loop end-to-end through the variant dispatch, where `quantize_tensor_2d_to_bytes` and `target_to_ggml_name` regressions could otherwise produce structurally-valid bytes that round-trip to garbage), `69922ff` (P7 iter-3w — imatrix-vs-none divergence gate: `variant_imatrix_diverges_from_none_through_streaming` proves the imatrix calibration path is genuinely end-to-end through `VariantKQuantizer` + `quantize_streaming` by asserting non-uniform-importance imatrix produces **different** Q4_K bytes than `CalibrationData::None` on the same F32 input — catches a future regression where calibration plumbing silently routes through the uncalibrated codec), `9a002a0` (P7 iter-3v — `KQuantVariant::all()` enumeration helper + `Display` impl + `variant_menu_smoke_through_streaming` test that exercises every variant end-to-end through `quantize_streaming` with the correct base-target ggml_type and block-size byte count; locks the variant menu shape for P8 CLI registration so adding a new variant to `all()` automatically extends test coverage), `c4dcb0e` (P7 iter-3u — `TensorCategory::classify` ported to llama.cpp's substring-priority order at `llama-quant.cpp:115-150` (adds AttentionQkv/AttentionKvB/AttentionOutput/FfnUp/FfnGate); MoE expert variants `*_exps.weight` and `*_shexp.weight` now classify under their canonical category (closes Q4_K_M parity gap on MoE models — `blk.X.ffn_down_exps.weight` was falling through to `Other` instead of getting the `use_more_bits` Q6_K bump); `should_skip_quantization()` predicate matches `llama-quant.cpp:307` to keep `ffn_gate_inp.weight` (MoE router) at original precision; AttentionQkv/AttentionKvB get the AttentionV policy per `category_is_attn_v`; `VariantKQuantizer` honors the skip predicate via passthrough; 10 new tests including streaming-path lock-in for the MoE skip + classification rules), `6824062` (P7 iter-3t — streaming-path integration test: `quantize_streaming(LazyTensorMap, ..., VariantKQuantizer::Q4_K_M)` produces a `QuantizedModel` byte-identical to `quantize_model(&TensorMap, ..., VariantKQuantizer::Q4_K_M)` on the same fixture, with all four `Q4_K_M` policy branches covered (`output.weight` → Q6_K bump; `blk.0.attn_v.weight` → Q6_K via `use_more_bits`; `blk.5.attn_q.weight` → Q4_K base; `blk.10.ffn_down.weight` → Q4_K via `(10-4)%3=0`) — closes the P0 + P2 + P3 + P7 phase-boundary gap) | `Calibrator` trait (Send + Sync + object-safe) + `CalibrationData` enum (None / Imatrix / ImatrixWithStats / Dwq) + `CalibrationCorpus` + `NoneCalibrator` truly no-op impl + 5 typed `CalibrationError` variants. 7 unit tests. P7 iter-3 k-quant codebook port: **Q4_K + Q5_K + Q6_K block layouts (`repr(C)` byte-for-byte match against `block_q*_K` in `ggml-common.h`) + `dequantize_row_q4_k` / `dequantize_row_q5_k` / `dequantize_row_q6_k` (pure-Rust ports of `ggml-quants.c:1467`/`:1669`/`:1877`) + `nearest_int` bit-trick (`:559`) + `make_qkx2_quants` codebook search (`:737`) + `quantize_row_q4_k` (`:1395`)**. 31 k-quant unit tests passing including round-trip RMSE bound (synthetic ramp & multi-block) ≤ 0.05 for Q4_K. Pending: ImatrixCalibrator + DwqCalibrator impls (P7 iter-2-bis); `quantize_row_q5_k` + `quantize_row_q6_k` (P7 iter-3b2/3); byte-identity gate against llama.cpp NEON path on `aarch64-apple-darwin` via stored fixture (P7 iter-3b4, Decision 11 round-2); Layout A path migration (P7 iter-4). |
| P8 — CLI rename + final variant menu | 🟡 iter-1 landed; off-diagonal dev-gate + AutoResolver D18 + 17-variant menu live | `8e590e1` (iter-1 — 17-variant `QuantMethod`, deleted MixedNN/Apex/DwqMixedNN with did-you-mean error mapping per Decision 13, `HF2Q_UNSAFE_EXPERIMENTS=1` dev gate for off-diagonal `--calibration` × `--output-format`, AutoResolver Decision 18 routing table, `select_calibrator` extended for Imatrix routes, `KQuantCodecQuantizer` wired for q4_k_m/q5_k_m/q6_k + imatrix-q*, `VariantKQuantizer` for imatrix-adaptive preserving apex per-tensor optimal-precision) | Codex team failed early (spec-fetch crash); Claude team (opus) completed full 7-subtask scope solo. +21 tests (1929 → 1950 baseline). cargo clippy `0 errors / 401 warnings` (down 55 from 456 — forward-API symbols from P7 wired up). |
| P9 — Safetensors backend integration with calibrators | 🟢 iter-1 landed (commit `7dbb41c`); SafetensorsBackend through Calibrator dispatch — requires_native_quantization shortcut removed; mlx-lm-style directory layout (config.json + tokenizer + sharded model-NNNNN-of-MMMMM.safetensors with per-shard quant metadata: DWQ scales+biases per mlx-lm convention, K-quant opaque blob with quant_method discriminator); 7 integration tests including 2 #[ignore]-gated mlx_lm.load round-trip + cosine-similarity gates for P10. | `7dbb41c` | iter-1 (2026-04-27): `SafetensorsBackend` no longer overrides `requires_native_quantization`; both `GgufBackend` and `SafetensorsBackend` now flow through the `select_calibrator → Calibrator → Quantizer → QuantizedModel → backend.write` chain so the IR-level quantize loop is the single dispatch path (the ~6 `if backend.requires_native_quantization()` arms in `cmd_convert` survive as Chesterton's fence — future native-quant backends can opt in). Added `OutputBackend` mod-level tests asserting trait default + `SafetensorsBackend.requires_native_quantization() == false`. **Directory layout** (`src/backends/safetensors_out.rs`, +220 LOC): for quantized variants the backend emits `<output>/{config.json, model.safetensors OR model-NNNNN-of-MMMMM.safetensors + model.safetensors.index.json, quantization_config.json}` matching `mlx_lm.utils.save_model:727` (single-shard ⇒ `model.safetensors`, multi-shard ⇒ 5-digit zero-padded `model-NNNNN-of-MMMMM`). `config.json` is read from input HF repo, top-level `quantization` block injected per `mlx_lm.utils.save_config:912-913` + mirrored `quantization_config`, sorted, written. `copy_sidecars` updated to skip files already present at destination so the backend-injected config.json is not clobbered by the byte-copy step. **DWQ schema**: per `<name>.weight` (U8 packed), `<name>.scales` (F16, mlx-lm `utils.py:154`), `<name>.biases` (F16, `utils.py:155`); symmetric DWQ synthesises zero-biases to satisfy mlx-lm's affine-schema contract. **K-quant schema**: opaque GGUF block bytes in `<name>.weight` U8 slot only (no companions — scales packed inline); per-shard `__metadata__` carries `k_quant_method = k_quant_q4_k_m | …` discriminator for loader routing; mlx-lm cannot load these natively, hf2q's serve loader handles them. **CLI flag** (`src/cli.rs`): `--shard-size-gb` (default 5.0, range 0.5..=50.0) replaces the hardcoded 4 GB constant; threaded through `ConvertConfig.shard_size_gb` → `SafetensorsBackend::with_shard_size_gb`. **Test budget**: 6 new bin tests (2037 → 2043 baseline) + 7 new integration tests in `tests/safetensors_mlx_lm_round_trip.rs` (4 always-on covering f16 single-file / DWQ directory / K-quant blob / safetensors-reader round-trip + 1 byte-identity regression for f16 + 2 `#[ignore]`-gated mlx_lm.load + cosine gates for P10). cargo clippy `0 errors / 392 warnings` (down 9 from 401 — config.json injection wires up forward-API symbols). Decision 17 byte-identity gate for `--quant f16 --format safetensors` verified across two runs by `f16_round_trip_byte_identical_to_eager`. |
| P10 — Peer-parity benchmark harness (llama.cpp + mlx-lm) | 🟡 iter-1 + iter-2a + iter-2b + iter-2c + iter-3 landed (commits `243174d`, `d41dc87`, `02faaf6`, `5a2d64f`, `f0fc86c`); harness skeleton + 8-cell `GateCell` matrix from Decision 15 + `tests/common/{llama_cpp_runner, mlx_lm_runner}` subprocess wrappers (now including `run_llama_perplexity`) + `scripts/peer_parity_run.sh` cold-cache protocol + `emit_markdown_table` (now with 3 PPL columns) → `docs/peer-parity-results-<YYYY-MM-DD>.md`. iter-2a wired peer-side PPL via `llama-perplexity` subprocess wrapper + 512-token deterministic smoke fixture + 4 new always-on smoke tests. iter-2b wires hf2q-side PPL for the 4 dense cells (cells 0-3) via `src/quality/ppl_driver.rs::measure_ppl_qwen35_dense` wrapping `Qwen35Model::load_from_gguf` + `forward_cpu` + `compute_perplexity`. iter-2c (rename + wire) renames the driver to `measure_ppl_qwen35` (variant-agnostic; one entry point handles both `Variant::Dense` and `Variant::Moe`) and wires the 4 MoE cells (cells 4-7) through the same driver — 8-cell PPL coverage complete. iter-3 lands the full-corpus fetcher + gitignore + corpus-loader auto-pick for P11's full wikitext-2 PPL gate. | `243174d`, `d41dc87`, `02faaf6`, `5a2d64f`, `f0fc86c` | iter-1 (2026-04-27): `tests/peer_parity_gates.rs` (NEW) — `GateCell { model_id, backend, calibrator_variant, peer_id, speed_tolerance, rss_tolerance, ppl_tolerance }` populated **verbatim** from Decision 15 lines 575–582; `Verdict { Pass, Fail, NotMeasured }` enum surfaces deferred state honestly (no fake-green); 6 always-on smoke tests cover (a) markdown emitter on empty input, (b) speed/RSS/PPL tolerance predicates, (c) missing-binary sentinel for both llama.cpp and mlx-lm subprocess wrappers; 8 `#[ignore]`-gated cells (`cell_<idx>_<model>_<backend>_<calibrator>`, `#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk"]`) call the harness end-to-end and return `Verdict::NotMeasured` this iter — P11 wires the real-model side. `tests/common/{mod, metrics, llama_cpp_runner, mlx_lm_runner}.rs` (NEW): `RunMetrics { wall_s, peak_rss_bytes, exit_code, stderr_tail }` + `RunMetrics::missing_binary(name)` sentinel constructor (`wall_s=-1.0, peak_rss_bytes=u64::MAX, exit_code=-1`); each runner resolves its binary via env-var override (`HF2Q_LLAMA_QUANTIZE_BIN`, `HF2Q_LLAMA_CONVERT_HF_BIN`, `HF2Q_LLAMA_IMATRIX_BIN`, `HF2Q_PYTHON_BIN`) → `$PATH` walk; on missing → `tracing::warn!` + sentinel. mlx-lm runner additionally probes `python3 -c "import mlx_lm"` and treats `ImportError` as missing-module (returns `RunMetrics::missing_binary("mlx_lm")`). `scripts/peer_parity_run.sh` (NEW, ~140 lines bash): 1 warmup discarded → 60 s thermal cooldown → 3 timed runs each wrapped in `/usr/bin/time -l` (BSD/Mac format), CSV body sorted ascending by `wall_s` so the harness reads the median (row 2 of 3); validated via `bash -n` only this iter (real-peer execution lands in P11). `emit_markdown_table(results, hardware, sha) -> String` (pure, no I/O); `write_results_to_dated_doc(results, hardware, sha, today, docs_dir)` writes to `docs/peer-parity-results-<YYYY-MM-DD>.md` (only callable from `#[ignore]`-gated cells so the always-on suite does not pollute `docs/`). The 2 P9-deferred gates in `tests/safetensors_mlx_lm_round_trip.rs` (`safetensors_directory_loads_in_mlx_lm`, `safetensors_dwq46_cosine_similarity_above_99_9_percent`) now route through `tests::common::mlx_lm_runner` instead of inlining `python3 -c`; both stay `#[ignore]`-gated. **iter-2a (2026-04-27, commit `d41dc87`)**: peer-side PPL wired. `tests/common/llama_cpp_runner.rs::run_llama_perplexity` wraps `/opt/homebrew/bin/llama-perplexity` (resolves via env `HF2Q_LLAMA_PERPLEXITY_BIN` → $PATH); parses the upstream `Final estimate: PPL = <f32>` stderr line (verified at `/opt/llama.cpp/tools/perplexity/perplexity.cpp:654`; tolerates the `\x1b[32mI \x1b[0m` ANSI color prefix `LOG_INF` emits per `common/log.cpp:88` — `LOG_INF` routes to stderr for non-NONE log levels); missing binary → `RunMetrics::missing_binary` sentinel + `None` PPL. `tests/fixtures/ppl-corpus/wikitext2-smoke.tokens` (NEW, 2 KB, 512 little-endian u32 tokens, deterministic ramp `(i*17+3) % 32000` documented in sibling README); `tests/fixtures/ppl-corpus/README.md` (NEW) documents format + generation rule + role + iter-2b/iter-3 roadmap. `emit_markdown_table` grows 3 PPL columns at the END (`hf2q PPL`, `peer PPL`, `PPL ratio`); column order preserved so existing iter-1 tests pass unchanged (the iter-1 deferral note is replaced with a peer-side-wired note + forward pointer to iter-2b). New `CellResult::from_measurements(cell, hf2q, peer, ppl_hf2q, ppl_peer)` constructor computes ratios + verdict honestly: any sentinel input or missing PPL routes to `Verdict::NotMeasured`; otherwise the gate triple (speed/RSS/PPL) decides Pass vs Fail naming the first failing gate. `run_cell` invokes `run_llama_perplexity` against the smoke fixture so the 8 `#[ignore]`-gated cells exercise the wiring; hf2q-side stays `Verdict::NotMeasured` until iter-2b lands `measure_ppl_qwen35_dense_gguf(model, tokens) -> f32`. 4 new always-on smoke tests cover (a) fixture load + deterministic content (`wikitext2_smoke_fixture_loads_to_512_tokens`), (b) missing-binary sentinel + None ppl (`peer_perplexity_wrapper_handles_missing_binary`), (c) markdown table renders PPL columns + half-measured row (`markdown_table_renders_ppl_columns`), (d) markdown table renders full row + verdict logic on `ppl_tolerance` (`markdown_table_renders_full_ppl_row_when_both_measured`). Plus 5 new in-runner unit tests for the parser (`perplexity_parser_extracts_plain_final_estimate`, `perplexity_parser_tolerates_ansi_color_prefix`, `perplexity_parser_returns_none_when_line_missing`, `perplexity_parser_returns_none_on_unparseable_token`, `perplexity_wrapper_missing_binary_returns_sentinel_and_none`). **Spec deviation note (queen)**: spec hinted at `Result<(RunMetrics, Option<f32>)>` return type and `getrusage(RUSAGE_CHILDREN)` for RSS. We returned `(RunMetrics, Option<f32>)` (no `Result` wrapper) to match the sibling sentinel-based convention from iter-1 (`run_llama_quantize` etc. — Chesterton's fence; the missing-binary sentinel already encodes "errors flow through the discriminant, never panic"); RSS continues to use `parse_bsd_time_peak_rss` per sibling pattern because `getrusage(RUSAGE_CHILDREN)` is process-cumulative across all children under cargo's multi-threaded runner and would over-report. **iter-2b follow-up (landed below)**: hf2q-side `measure_ppl_qwen35_dense_gguf(model, tokens) -> f32` driver wrapping `Qwen35Dense::from_gguf` + chunked forward-pass + `compute_perplexity`; flips the 8 cells to record hf2q PPL too. **iter-3 follow-up (landed below)**: full wikitext-2 token split fetched via fetch script for P11 close. cargo build `0 errors`; cargo test `--bin hf2q --release` 2049 passed (no regression — new tests live in integration-test crates outside the bin); `cargo test --release --test peer_parity_gates` 28 passed (19 iter-1 + 4 iter-2a always-on smoke + 5 new in-runner unit) + 8 ignored; `cargo test --release --test imatrix_xvalidation` 36 passed + 1 ignored (unchanged). cargo clippy `0 errors / 388 warnings` (down 4 from iter-1's 392 — wiring up forward-API symbols; well within ≤ 395 budget). **iter-2b (2026-04-27, commit `02faaf6`)**: hf2q-side PPL driver landed for the 4 dense cells. `src/quality/ppl_driver.rs` (NEW, 451 LOC) implements `pub fn measure_ppl_qwen35_dense(model: &Path, tokens: &[u32], seq_len: Option<usize>) -> Result<f32, PplDriverError>` wrapping `Qwen35Model::load_from_gguf(&GgufFile::open(path)?)` + per-chunk `forward_cpu(window, &text_positions(window.len()))` + `compute_perplexity(&logits_rows, &shifted_targets)`. Default `seq_len = cfg.max_position_embeddings.min(tokens.len())` so a small corpus runs as a single chunk; explicit `Some(N)` override forces N-token windows. Standard "predict next token" alignment: per-window we pair `logits[0..L-1]` with `tokens[s+1..s+L]` and accumulate across windows. `PplDriverError { Gguf{path,source}, Load(String), Forward{chunk,cause}, Perplexity(#[from] PerplexityError), Invalid(String) }` — typed discriminants so callers route on cause without string-matching; `Forward` carries the chunk index for "regression at the Nth window" surfacing; `Load` and `Forward.cause` carry formatted strings (not the live `anyhow::Error`) so `PplDriverError` stays `Send + Sync + 'static`. Public `chunk_count(n_tokens, seq_len) -> usize` helper exposes the chunking arithmetic for unit tests without a model load. **Test inventory** — `tests/ppl_driver.rs` (NEW, 286 LOC, integration crate, 5 always-on smoke tests): (1) `ppl_driver_returns_invalid_on_empty_token_slice` — empty + length-1 → `PplDriverError::Invalid` with `tokens.len()` named in the message; (2) `ppl_driver_returns_gguf_error_on_missing_path` — sentinel path → `PplDriverError::Gguf` with the path round-tripped verbatim; (3) `ppl_driver_returns_gguf_error_on_invalid_magic` — 8 zero bytes in a tempfile → `PplDriverError::Gguf` (distinct surface from #2: file open succeeds, magic fails); (4) `ppl_driver_seq_len_override_is_respected` — `Some(0)` → `Invalid(seq_len ...)`, `Some(2)` with missing path → `Gguf` (validation accepts override; file-open preempts chunking); (5) `ppl_driver_chunk_count_for_512_tokens_with_seq_len_128_is_4` — pure chunking arithmetic + boundary cases. Plus 9 in-source `#[cfg(test)]` unit tests in `ppl_driver.rs::tests` (chunking edge cases at zero/exact/partial/window-of-1 + driver Invalid+Gguf early-exits). `tests/peer_parity_gates.rs` extended: cells 0-3 now route through `measure_ppl_qwen35_dense(&hf2q_model_path(&cell), &load_smoke_corpus_tokens()?, None)`; sentinel `/var/empty/...gguf` paths trigger `PplDriverError::Gguf` → `hf2q_ppl = None` → cell verdict `NotMeasured` (the wiring is exercised; P11 swaps real 27B-dense GGUFs in). The 4 MoE cells (4-7) keep `hf2q_ppl = None` per-spec — iter-2c lands the MoE driver. The 4 dense `#[ignore]` reasons updated to `"P11 hardware gate: needs 27B-dense GGUF + ~100GB disk + Qwen35Model forward_cpu warm"`. **Lib-target gap** (queen): hf2q is a binary crate (no `[lib]` target — confirmed at `Cargo.toml:1-160`), so `tests/*.rs` cannot say `use hf2q::quality::ppl_driver`. Both new test crates use the established `#[path]`-include pattern (precedent: `tests/imatrix_xvalidation.rs:48-52`) PLUS minimal type-stubs in the `inference::models::qwen35::{forward_cpu::text_positions, model::Qwen35Model, Qwen35Variant}` namespace to satisfy the production driver's qwen35 imports without dragging in the deeply-interconnected qwen35 module tree (model.rs + ffn.rs + full_attn.rs + delta_net.rs + weight_loader.rs + …). The stubs are dead code at runtime: every always-on test exits the driver at input-validation or `GgufFile::open` long before any `Qwen35Model` method is invoked. P11 will land a `[lib]` target (or move test scaffolding into `src/main.rs`'s test surface) so the real qwen35 path is exercised against real 27B-dense GGUFs. **Verification**: cargo build `0 errors`; `cargo test --release --test ppl_driver` 24 passed (5 always-on top-level + 9 in-source ppl_driver `#[cfg(test)]` + 10 perplexity transitive via `#[path]`-include) / 0 ignored; `cargo test --release --test peer_parity_gates` 47 passed (28 baseline + 19 from `#[path]`-included perplexity + ppl_driver) + 8 ignored — baseline 28 always-on preserved; `cargo test --release --test imatrix_xvalidation` 36 passed + 1 ignored (unchanged); `cargo test --bin hf2q --release` 2058 passed (+9 from baseline 2049: ppl_driver `#[cfg(test)]` block) + 11 ignored; `cargo clippy --release -p hf2q -- -D clippy::correctness` 0 errors / 388 warnings (matches baseline; well within ≤ 395 budget). **iter-2c (2026-04-27, commit `5a2d64f`)**: rename + wire iter (NOT a new-driver iter). Chesterton's-fence audit confirmed `Qwen35Model::load_from_gguf` (model.rs:172) inspects `cfg.variant` (model.rs:265) and dispatches both `Variant::Moe` (model.rs:282-307: `load_moe_ffn_quantized` → `Qwen35FfnWeights::MoeQ`) and `Variant::Dense` (model.rs:308-311: `load_layer`) internally; `Qwen35Model::forward_cpu` (forward_cpu.rs:75) likewise dispatches on `Qwen35FfnWeights::{Dense, Moe, DenseQ, MoeQ}` (forward_cpu.rs:152-192) inside the per-layer FFN block. The iter-2b `Variant::Dense`-only gate at `ppl_driver.rs:210-216` was therefore a forward-looking guard, not a correctness invariant. **Rename**: `measure_ppl_qwen35_dense` → `measure_ppl_qwen35` in `src/quality/ppl_driver.rs` (no deprecated alias — exactly one in-tree consumer; module-level doc + function-level doc + the `PplDriverError::Invalid` variant doc updated to document variant-agnosticism + cite the model.rs line numbers). `Qwen35Variant` import removed (no longer used). Variant-rejection branch removed; loaded model now flows straight to `vocab_size = qwen.cfg.vocab_size as usize`. **Wiring**: `tests/peer_parity_gates.rs::run_cell` collapses the `if cell.model_id == "27B dense"` discriminator → all 8 cells now route through `measure_ppl_qwen35` with the cell's hf2q model path + smoke corpus. With sentinel `/var/empty/...gguf` paths the driver returns `PplDriverError::Gguf` ⇒ `hf2q_ppl = None` ⇒ verdict `NotMeasured` for both dense and MoE — same observable as iter-2b for cells 0-3, and the wiring is now exercised for cells 4-7 too. The 4 MoE `#[ignore]` reasons updated to `"P11 hardware gate: needs apex MoE GPU + ~150GB disk + Qwen35Model::load_from_gguf for Variant::Moe"` (the dense reasons stay unchanged). The hf2q-metrics sentinel string bumped to `"hf2q (iter-2c: PPL via variant-agnostic driver; wall+RSS pending P11)"`. **New always-on smoke test**: `tests/ppl_driver.rs::ppl_driver_rejects_invalid_for_both_variants` — calls `measure_ppl_qwen35(Path::new("/var/empty/never.gguf"), &[], None)` and asserts the returned `PplDriverError::Invalid` carries `tokens.len()` in the message. The empty-tokens validation fires BEFORE any GGUF inspection, so this is the simplest variant-agnostic assertion landable without a real model on disk; a regression that re-introduced a variant-specific gate before the `tokens.len()` check would surface here as a `Gguf` or `Invalid("variant ...")` error rather than the canonical `Invalid("tokens.len() ...")`. **Verification**: cargo build `0 errors`; `cargo test --release --test ppl_driver` 25 passed (24 baseline + 1 new — `ppl_driver_rejects_invalid_for_both_variants`) / 0 ignored; `cargo test --release --test peer_parity_gates` 47 passed + 8 ignored (matches iter-2b baseline; same observable verdict surface, MoE wiring now exercised); `cargo test --release --test imatrix_xvalidation` 36 passed + 1 ignored (unchanged); `cargo test --bin hf2q --release` 2058 passed + 11 ignored (unchanged from iter-2b — same `#[cfg(test)]` count in `ppl_driver.rs`); `cargo clippy --release -p hf2q -- -D clippy::correctness` 0 errors / 388 warnings (matches baseline; well within ≤ 395 budget). **LOC**: src/quality/ppl_driver.rs +35/-37 (rename + variant-gate removal + module-level + function-level + Invalid-variant doc updates); tests/peer_parity_gates.rs +59/-39 (run_cell collapse + module/inline-doc updates + 4 MoE ignore-reason updates); tests/ppl_driver.rs +37/-7 (rename + 1 new test). Net +95/-83 across 3 files. **iter-3 (2026-04-27, commit `f0fc86c`)**: `scripts/fetch_wikitext2.sh` downloads the Stephen Merity/Salesforce `wikitext-2-raw-v1.zip` artifact from the ggml-org/ci HuggingFace mirror, locks SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`, extracts `wikitext-2-raw/wiki.test.raw`, tokenizes with verified `llama-tokenize --model ... --file ... --no-bos --ids --log-disable` flags, parses numeric IDs to the same raw little-endian u32 stream as `wikitext2-smoke.tokens`, and refuses corrupt downloads or undersized output (`>= 280000` tokens and `>= 1 MiB`). `tests/fixtures/ppl-corpus/.gitignore` ignores generated `*.tokens` while preserving `wikitext2-smoke.tokens`. `tests/peer_parity_gates.rs::load_corpus_tokens` auto-picks `wikitext2-full.tokens` when present+valid and logs fallback to smoke otherwise. `tests/ppl_driver.rs::corpus_loader_falls_back_to_smoke_when_full_absent` adds 1 always-on smoke. Verification (with `RUSTC_WRAPPER=` because this sandbox blocks `sccache`): `bash -n scripts/fetch_wikitext2.sh` clean; missing tokenizer preflight exits 1 with a clear error; cache-hit skip exits 0; `cargo build --release -p hf2q` 0 errors; `cargo test --release --test ppl_driver` 26 passed; `cargo test --release --test peer_parity_gates` 47 passed + 8 ignored; `cargo test --release --test imatrix_xvalidation` 36 passed + 1 ignored; `cargo clippy --release -p hf2q -- -D clippy::correctness` 0 errors / 388 warnings. Deviation: `cargo test --bin hf2q --release` failed in this sandbox because Metal is unavailable (`DeviceNotFound`): 1915 passed / 143 failed / 11 ignored; the failing tests are pre-existing GPU/vision surfaces outside the P10 file fence. |
| P11 — Re-emit ADR-012's four DWQ GGUFs under streaming pipeline + measured gate close | ⏳ pending | — | Depends on every prior phase. Closure AC.  **P11-prereq Iter A landed (commit `4349fb2`)**: codec-direct fast-path in `quant_info_to_ggml_type` + `repack_to_ggml_blocks` closes LIVE `--quant q4_k_m` / `imatrix-q4_k_m` / `imatrix-adaptive` malformed-GGUF bug (header reported F16 atop on-disk Q4_K bytes); 11 always-on in-source regression tests + 4 always-on integration tests via `mlx_native::gguf::GgufFile` header read-back lock the fix. Iters B + C still pending for DWQ Q4_0 → Q4_K_M rewrite. |
| P12 — Documentation refresh (`converting-a-model.md`, `converting-qwen35.md`, `shipping-contract.md`) | 🟡 ADR Phase status table refreshed inline | — | This iter (2026-04-26): refreshed Phase status table with current commit hashes for P0–P7 landed work. End-user docs (`converting-a-model.md` rewrite, `converting-qwen35.md` update, `shipping-contract.md` peer-parity gates section, new `calibrator-onboarding.md`) land at P12 close after P11. |

ADR-014 closes only when P11 is green: all four reference artifacts re-emitted, every peer-parity gate measured and passed, every comparison number recorded inline in the closing commit.

### Round-2 refinement log (2026-04-25, party-mode session 2)

Twelve Robert-locked refinements applied this session. Citations point to the body where each is now codified:

1. **D22** — P11 ships only the four ADR-012 carry-over DWQ GGUFs + four DWQ safetensors twins to `models/`. Imatrix-q4_k_m gate cells produce ephemeral measurement files only.
2. **D11** — k-quant byte-identical gate is against llama.cpp's NEON code path on `aarch64-apple-darwin`, not the scalar reference.
3. **D6 + P0** — Decision 6's literal `≤ 35 GB` replaced with `measured + 10% headroom` derived from a P0-prerequisite measurement spike on apex MoE activation capture.
4. **Phase plan intro** — P11/P12 may parallelise via separate CFA worktrees (`feedback_swarm_sequential_when_shared_build.md`'s sequential rule applies to shared `target/`; CFA splits it).
5. **D16** — PPL eval corpus is the **full ~280k-token wikitext-2 test split**, not 512 tokens. Apples-to-apples vs published peer numbers.
6. **D15** — Speed/RSS protocol replaced: `1 warmup run discarded → 60 s thermal cooldown → 3 timed runs, median wins`. Both peers warm. Sidesteps Metal-shader-cache persistence question.
7. **D12** — All 17 named CLI variants ship in this ADR. Full menu is the deliberate long-term scope.
8. **D21** — Sovereignty rule is runtime-only. Test-time Python (mlx-lm, llama-imatrix) permitted only behind `#[ignore]`-gated parity-harness tests.
9. **D5** — Decision 5's `~80 LOC` re-estimated to `~300–500 LOC honestly written`. Pipeline shape stands; correctness and speed dominate over LOC ceiling.
10. **R14** — ADR-014 P0 starts only after ADR-012's four DWQ GGUFs verifiably load in `llama-cli` (real ADR-012 close).
11. **D14** — `--format safetensors` output is an **mlx-lm-style directory** (`config.json` + tokenizer + sharded `model-*.safetensors` with quant metadata in shard headers). Appendix A is canonical.
12. **D9 + P7** — Layout A: `dwq*.rs`, `dwq_activation.rs`, `sensitivity.rs`, `apex.rs` move from `src/quantize/` into `src/calibrate/`. Full path migration; P7 LOC budget grows for import-rewrite churn.

---

## Engineering Mantra (load-bearing — read before every session)

From `~/Documents/mantra.txt` (2026-04-07):

> DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.

Operationalised for this ADR:

- **No shortcuts.** Every fix in items #1–#10 is in scope. None is punted to a follow-up perf ADR.
- **No fallback.** When a calibrator fails, the convert errors with a typed message — never silently regresses to weight-space.
- **No stub.** Every Decision body specifies the file path, the function signature, the test name, and the LOC estimate. No "wire-up later" sentences.
- **Measure 3x, cut once.** Every peer-parity gate has three measurements: hf2q current pipeline, hf2q new pipeline, peer baseline. Three numbers per cell of the AC table.
- **Chesterton's fence.** The current pipeline's eager `to_vec()` (`src/input/safetensors.rs:316`) and bulk `clone()` patterns (`src/backends/gguf.rs:897/910/926/3433`) are deliberate as of when they were written; this ADR explains *why* they were correct *then* and *why* they are wrong *now* before changing them.

---

## Context

### Business problem

hf2q is positioned as **both** a conversion tool and an inference tool, serving GGUF (for `llama.cpp` + `ollama` users) and safetensors (for `mlx-lm` users + native `mlx-native` loading). Robert (2026-04-25, party-mode session): *"We're a conversion tool and an inference tool — we need to support both."* Peer parity is therefore measured on **two** axes: GGUF output against `llama.cpp` (`convert_hf_to_gguf.py` + `llama-quantize` + imatrix), and safetensors output against `mlx_lm.convert` (DWQ-calibrated). hf2q must be measurably as fast and as correct as both peers on the same Apple Silicon hardware (M5 Max, 128 GB), or it ships as a degraded substitute.

### Technical problem

Walking the current `cmd_convert` pipeline (`src/main.rs:176–910` on `main`, `src/main.rs:255–1243` on the `worktree-adr-012-p8-p11` branch) against `convert_hf_to_gguf.py` and `mlx_lm.convert` reveals **ten** distinct inefficiencies or peer divergences. They split into two cohorts.

**Cohort A — fixable inefficiencies (correctness + perf failure modes):**

1. **P9b intermediate-GGUF dance.** `src/main.rs:680–778` (worktree). DWQ-on-qwen35 emits a full F16+Q8 intermediate GGUF to a tempdir, reopens it as a `Qwen35Model` for forward-pass activation capture, then re-reads the safetensors and re-applies Phases 1.4/1.45/1.6/1.7. Three full touches of the model weights to do one calibration. Caused by no in-memory `LazyTensorMap → ActivationCapture` path; only the GGUF reader is plumbed.
2. **Eager `to_vec()` of mmap'd safetensors.** `src/input/safetensors.rs:316` does `mmap[abs_start..abs_end].to_vec()` per tensor — the entire ~70 GB BF16 of the apex MoE lands in anonymous heap before any quantize call. `convert_hf_to_gguf.py:13203–13273` (`LazyTorchTensor`) keeps the mmap and only materialises when the per-tensor callable is invoked.
3. **Whole-model `QuantizedModel` held until write.** `src/quantize/mod.rs:129` accumulates `HashMap<String, QuantizedTensor>`; the write at Phase 4.6 (`src/main.rs:984` worktree) flushes the entire map. Peak RSS doubles through the quantize→write transition. `convert_hf_to_gguf.py`'s `Model.write_tensors()` is generator-style: convert → write → drop.
4. **No parallelism in quantize.** `src/quantize/mod.rs:55` declares the `Quantizer` trait `Send + Sync` "for rayon parallelism" but the actual quantize loop iterates sequentially. `llama-quantize` is `-t`-threaded; `mlx_lm.convert` gets Metal op parallelism for free. On apex MoE this is minutes-to-tens-of-minutes per pass.
5. **No imatrix-style activation cache between bit-pair variants.** dwq46 and dwq48 each redo the full forward+capture from scratch. Wasted forward pass per second variant on the same model. llama.cpp's `.imatrix` file is computed once and applied across Q4_K_M, Q5_K_M, Q6_K.
6. **Bulk `clone()` of tensor data on the write surface.** `src/backends/gguf.rs:897/910/926/3433` clones quantized blobs (full `Vec<u8>` weight copies) across the write path; `src/quantize/mod.rs:82,129` clones names + shapes. Peers thread refcounts (mlx arrays, torch storages).
7. **Streaming gap at MoE quant time.** Phase 1.5 (`src/main.rs:522` worktree) streams the expert merge layer-by-layer (peak +3.7 GB). Phase 3 then quantises the merged `[N=256, hidden, moe_inter]` block *as one piece*, holding a 256× tensor in RAM. Should also be layer-streaming.

**Cohort B — peer divergences (calibrators):**

8. **Calibrator monoculture.** hf2q ships only Apple/MLX's *Distilled Weight Quantization* (DWQ — `src/quantize/dwq.rs:1`). It does not ship llama.cpp's *importance matrix* (imatrix — `/opt/llama.cpp/tools/imatrix/imatrix.cpp`). DWQ and imatrix are peer-equivalent published calibration techniques from two communities; hf2q ships one, not both. This is a divergence by accident, not by design — Robert (2026-04-25): *"I think we want to support both dwq and imatrix."*

**Cohort C — sovereignty-by-design (preserve, document):**

9. **One-binary, single-process convert+quantize.** llama.cpp ships two stages (`convert_hf_to_gguf.py` → `llama-quantize`); we fuse them. Pro: single context, no intermediate persisted, one cancellation surface. Con: cannot replay just the quantize step against a cached F16. Sovereignty by design — Robert (`feedback_hf2q_sovereignty.md`, paraphrased): *"hf2q is pure Rust, mlx-native is the only sibling dep; no Python, no runtime link to candle/llama.cpp."* Preserved.
10. **Per-arch hand-ported transforms.** Phases 1.4 (`language_model.` strip), 1.5 (expert merge), 1.6 (RMS norm +1), 1.7 (V-head reorder, A_log negation, conv1d squeeze) are hand-translations of `convert_hf_to_gguf.py:5375–5424` + `Qwen3NextModel`. Sovereignty by design (no `torch` runtime dep). Preserved; future arches register via `ArchEntry` (ADR-012 Decision 20).

### Current state inventory (what exists in hf2q today, 2026-04-25)

- `src/main.rs:cmd_convert` — 1,668-line eager-load pipeline; Phases 0/0.25/0.3/0.4/0.5/1/1.4/1.45/1.5/1.6/1.7/2/3/4/4.5/4.6/4.7/4.8/5.1/5.5/6/7. The numbering is dense because every prior ADR added a phase rather than restructuring.
- `src/input/safetensors.rs` — mmap-then-`to_vec` reader; 332 LOC.
- `src/quantize/{mod,static_quant,mixed,dwq,dwq_activation,sensitivity,apex,intermediate_moe_q8}.rs` — eight quantizer modules; 4,460 LOC total.
- `src/backends/{gguf,safetensors_out,mod}.rs` — two output backends; 3,664 + 372 + 84 LOC. `SafetensorsBackend` flags `requires_native_quantization() = true` and bypasses the IR-level quantize.
- `src/inference/models/qwen35/activation_capture_real.rs` — `RealActivationCapture` consuming a GGUF file path (not in-memory). Forward-pass driver for DWQ on qwen35 / qwen35moe.
- `src/arch/` — ADR-012 Decision 20 registry; `ArchEntry` populated for `qwen35` + `qwen35moe`.

### Reference implementations (authoritative — read before changing)

- **Lazy tensor pattern.** `convert_hf_to_gguf.py:13203–13273` (`LazyTorchTensor`); `convert_hf_to_gguf.py:237` (`torch.load(..., mmap=True)`); `convert_hf_to_gguf.py:245–249` (lambdas wrapping safetensors slices).
- **Streaming write.** `convert_hf_to_gguf.py:Model.write_tensors()` (generator yielding tensors written in-place with `gguf_writer`).
- **imatrix calibration.** `/opt/llama.cpp/tools/imatrix/imatrix.cpp` (pre-pass that captures activations); `/opt/llama.cpp/src/llama-quant.cpp` (per-column-weighted MSE in `quantize_q4_K_M`, `quantize_q5_K_M`, `quantize_q6_K`).
- **DWQ calibration.** `src/quantize/dwq.rs:1–4` (header: "Distilled Weight Quantization (DWQ) calibration engine. Phase 1: Weight-space calibration. Phase 2: Activation-based calibration. optimal_scale = dot(W_original, Q_int) / dot(Q_int, Q_int)"); MLX-LM upstream — Apple ML Research. **Provenance correction**: this technique is *not* hf2q-original; the implementation is.
- **Rayon parallelism over quant.** llama.cpp's `quantize_q4_K_M` is multi-threaded via `std::thread` and `quantize_chunk_threaded` (`src/llama-quant.cpp`); shape: per-tensor work-units, write-stream serialised.

---

## Strategic Decision

ADR-014 re-platforms `cmd_convert` on a **lazy/streaming foundation**, lifts both **DWQ and imatrix** to first-class peer-equivalent calibrators, applies **cross-architecture** (not qwen-specific), and gates closure on **measured peer parity** against:

- `llama.cpp`'s `convert_hf_to_gguf.py + llama-quantize --imatrix` for the GGUF output path.
- `mlx_lm.convert --quant-method dwq` for the safetensors output path.

The architecture is **orthogonal internally** (`Calibrator` × `OutputFormat` traits independent) and **coupled externally** (only validated cross-product cells become named `--quant` variants; off-diagonal cells reachable only behind `HF2Q_UNSAFE_EXPERIMENTS=1`).

This ADR is a **parallel enhancement to ADR-012** (Robert, 2026-04-25: *"kind of an enhancement to"*). ADR-012 closes on the current pipeline; ADR-014's P11 re-emits ADR-012's four DWQ GGUFs (qwen3.6-27b dwq46, qwen3.6-27b dwq48, apex MoE dwq46, apex MoE dwq48) under the streaming pipeline as part of its own AC, providing the empirical answer to "does streaming change quant quality?" on the same models.

Robert priority lock (Q1, 2026-04-25): correctness primary, sovereignty secondary, performance tertiary — but Q1 was reframed by Q3: *"We need to be as fast as our peers. We need to be as correct as our peers."* Therefore correctness and performance are **co-primary**; sovereignty is an explicit non-trade.

---

## Non-Goals

1. **AWQ / GPTQ / AQLM calibrators.** The orthogonal `Calibrator` trait designed in Decision 9 is shaped to admit these methods — Decision 19 is one paragraph specifying how AWQ would land — but no AWQ/GPTQ/AQLM implementation lands in this ADR. Each future calibrator gets its own ADR.
2. **Inference-side changes.** ADR-013 owns the inference path; ADR-014 only consumes it via the existing `ActivationCapture` trait (with a new `from_lazy_tensor_map(...)` constructor — Decision 8). No new Metal kernels, no new model architectures in inference.
3. **New model architectures.** ADR-014 platforms the existing arch set (Gemma-4, Qwen3.5 dense, Qwen3.5-MoE) onto streaming. Future arches (Ministral, DeepSeek-V3) register via `ArchEntry` per ADR-012 Decision 20; that work belongs in their own ADRs.
4. **Tokenizer pipeline changes.** Calibration corpora use the model's existing tokenizer; no embed-side changes.
5. **GPU calibration kernels.** Activation capture uses ADR-013's existing GPU forward (`forward_gpu_with_capture`); no new Metal kernels in this ADR.
6. **Inference-side speed work.** ADR-005 owns inference perf; ADR-014 is convert-side only. Cross-cuts (e.g. `quantized_matmul_ggml` being shared between DWQ activation capture and decode) are noted but the kernels are not modified here.
7. **Replacement of the `auto` quant resolver.** The `intelligence::AutoResolver` keeps its public API; Decision 18 only adds new variant strings to its output table.

---

## Architecture Decisions

### 1. Lazy tensor primitive (`LazyTensor`)

**File:** `src/ir/lazy.rs` (new, ~250 LOC).

**Type:**

```rust
pub enum LazyTensor {
    Materialized(Tensor),
    Pending(Box<dyn FnOnce() -> Result<Tensor, MaterializeError> + Send + 'static>),
}

impl LazyTensor {
    pub fn materialize(self) -> Result<Tensor, MaterializeError> { … }
    pub fn map<F>(self, f: F) -> Self
    where F: FnOnce(Tensor) -> Result<Tensor, MaterializeError> + Send + 'static { … }
    pub fn shape(&self) -> &[usize] { /* metadata-only, no materialise */ … }
    pub fn dtype(&self) -> Dtype { /* metadata-only */ … }
}

pub struct LazyTensorMap {
    inner: BTreeMap<String, (LazyMeta, LazyTensor)>,
}
```

`LazyMeta` carries shape, dtype, and the byte-offset/length within the source mmap (so `shape()`/`dtype()` never materialise). `BTreeMap` (not `HashMap`) for deterministic iteration order — required for byte-identical regression on un-calibrated paths (Decision 17).

**Why `FnOnce` not `Fn`:** materialisation produces a `Vec<u8>` that we then own and drop. Re-materialising would re-allocate; we intentionally make that a compile error.

**Materialisation source A — safetensors mmap:** the closure captures an `Arc<Mmap>` + offset + len. `materialize()` does one `bytes_of_mmap[off..off+len].to_vec()` and returns. The `Arc<Mmap>` keeps the file mapped through the entire pipeline; pages are evicted by the kernel, not by us.

**Materialisation source B — already in memory:** `LazyTensor::Materialized(t)` returns `t`.

**Materialisation source C — derived (transform output):** the closure captures the parent `LazyTensor` + a `FnOnce` transform; calls `parent.materialize()?`, applies the transform, returns. Composes via `.map()`.

**Tests** (`src/ir/lazy.rs` + `tests/lazy_tensor.rs`):
- `test_materialize_once` — `materialize()` consumes; second call is a compile error.
- `test_map_compose_idempotent` — three transforms in a chain produce the same bytes as eager-applied transforms on the same input.
- `test_shape_dtype_no_materialise` — `shape()` + `dtype()` do not invoke the closure (verified via `Arc<AtomicUsize>` count).
- `test_send_bound` — `LazyTensor` is `Send` (so `rayon::par_iter` can move it across threads).

**Estimated LOC:** ~250 (impl ~150, tests ~100).

### 2. Lazy safetensors reader

**File:** `src/input/safetensors.rs` (modify, ~150 LOC delta — net +50 over the existing 332).

**Change:**

`read_safetensors_shard` returns a `LazyTensorMap` instead of a `TensorMap`. The `Mmap` is wrapped in an `Arc` and stored on the `LazyTensorMap`; each tensor entry's closure captures the `Arc<Mmap>` + offset/length. The existing `to_vec()` at line 316 is **deleted** — replaced by the closure body.

**Chesterton's fence.** The current `to_vec()` predates ADR-005's memory-pressure work; at the time, lifetimes on `Mmap` borrows were threaded through the entire pipeline and "copy out, drop the mmap" was the simplest way to avoid borrow propagation. We now have `Arc<Mmap>` patterns elsewhere (mlx-native tensor refs); the lifetime cost has been amortised.

**Multi-shard models.** The existing reader iterates shards; the new reader produces one `LazyTensorMap` per shard then merges — same end result, but each shard's `Arc<Mmap>` is independent (multiple files mmap'd concurrently). On apex MoE this means ~12 `mmap` regions live; total virtual ~70 GB; **resident** ~one tensor at a time.

**Tests:**
- `test_lazy_safetensors_apex_moe_peak_rss` — `/usr/bin/time -l` peak RSS during a `LazyTensorMap` iteration of apex MoE is `≤ 8 GB` (one layer's expert tile, BF16). Empirical baseline measured pre-P0 lands first; ADR closure asserts the gate.
- `test_lazy_safetensors_byte_identical_to_eager` — for Gemma-4 (small enough to fit eagerly), `LazyTensorMap::materialize_all()` produces a `TensorMap` byte-identical to today's `read_safetensors`.

**Estimated LOC:** ~150 modified, +100 tests.

### 3. Lift Phase 1.4–1.7 transforms to lazy

**Files:** `src/main.rs:457–582` (worktree), `src/models/qwen35/{dense,moe}.rs`.

**Change:** every Phase 1.x transform takes `&mut LazyTensorMap` (or returns a new one) instead of `&mut TensorMap`. The transforms compose via `.map()` on individual `LazyTensor`s — the bulk weight bytes never materialise during transform composition. Transforms run lazily; materialisation happens at quantize time (Decision 4).

**Phase 1.4** (`language_model.` prefix strip): pure metadata operation — rewrites map keys, no bulk bytes touched. Already trivially lazy.

**Phase 1.45** (jenerallee78 abliterated apex pre-merge handling): same — key rewrite, `merge_moe_experts_in_tensor_map` becomes `merge_moe_experts_in_lazy_map` with the merge closure deferred to materialisation time (Decision 7 — streaming MoE merge).

**Phase 1.5** (qwen35moe expert merge — ADR-012 Decision 9 / P5): the merge orchestration becomes a closure on the merged `LazyTensor`; the actual stack-in-memory of 256 experts only happens when the *quantize* loop materialises that one merged tensor (Decision 7 turns this into layer-streaming so the 256× explosion never happens).

**Phase 1.6** (RMS norm +1 bias): per-tensor `.map(add_one_inplace)`. Already lazy-ready.

**Phase 1.7** (V-head reorder, A_log negation, conv1d squeeze): per-tensor `.map(transform_fn)`. Already lazy-ready.

**Tests:**
- `test_lazy_phase_1_4_idempotent` — apply Phase 1.4 to `LazyTensorMap`, materialise — same bytes as eager Phase 1.4.
- `test_lazy_phase_1_5_layer_streaming` — for a synthetic 4-layer MoE, peak materialisation is one layer's experts (3 projections × 256 × hidden × moe_inter), not all 4 layers'. Verified via `Arc<AtomicUsize>` counter on the materialiser.
- `test_lazy_phases_1_6_1_7_byte_identical` — full Qwen3.5 4-layer synthetic, eager vs lazy → byte-identical output GGUF.

**Estimated LOC:** ~400 modified across 6 files (mostly mechanical signature changes).

### 4. Streaming quantize loop

**File:** `src/quantize/mod.rs` (rewrite `quantize_model`, ~300 LOC delta).

**Change:** replace

```rust
fn quantize_model(tensor_map: &TensorMap, …) -> QuantizedModel {
    let mut out = HashMap::new();
    for (name, tensor) in tensor_map.iter() {
        out.insert(name.clone(), quantize_tensor(...));
    }
    QuantizedModel { tensors: out, … }
}
```

with

```rust
fn quantize_streaming(
    map: LazyTensorMap,
    quantizer: &dyn Quantizer,
    backend: &mut dyn StreamingBackend,
    progress: &ProgressReporter,
) -> Result<QuantizationStats> {
    backend.begin_writing(map.metadata())?;
    for (name, (meta, lazy)) in map.into_iter() {
        let tensor = lazy.materialize()?;       // bytes IN
        let quantized = quantizer.quantize_tensor(&tensor, &meta, ...)?;
        backend.write_tensor(&name, &quantized)?;  // bytes OUT
        drop(tensor);                              // bytes FREED
        drop(quantized);                           // bytes FREED
        progress.tensor_done(&name);
    }
    backend.finalize()?;
    Ok(stats)
}
```

**`StreamingBackend` trait** (replaces the existing `OutputBackend`):

```rust
pub trait StreamingBackend: Send {
    fn begin_writing(&mut self, meta: &ModelMetadata) -> Result<()>;
    fn write_tensor(&mut self, name: &str, t: &QuantizedTensor) -> Result<()>;
    fn finalize(&mut self) -> Result<()>;
}
```

`QuantizedModel` is **deleted**. No code path holds the full quantised model in RAM. All metrics (peak RSS, wall time, file size) come from instrumented `StreamingBackend` impls.

**Cancellation safety** (Decision 20): `begin_writing` opens the output file in a tempdir; `finalize` atomically renames into place. Mid-stream Ctrl+C drops the tempdir; existing partial-cleanup logic in `cmd_convert` remains correct.

**Tests:**
- `test_streaming_peak_rss_apex_moe` — `/usr/bin/time -l` peak RSS during apex MoE q4_k_m convert is `≤ 44 GB` (peer-parity gate from Q3, "≤ peer + 10%").
- `test_streaming_byte_identical_q4_uncalibrated` — Gemma-4 `--quant q4` streaming-pipeline output byte-identical to current pipeline (Decision 17 determinism contract).
- `test_streaming_cancellation_no_partial_files` — SIGINT mid-stream leaves no `.gguf` or `.safetensors` in the output directory.

**Estimated LOC:** ~300 modified in `src/quantize/mod.rs`, ~150 modified each in `src/backends/{gguf,safetensors_out}.rs` to implement `StreamingBackend`.

### 5. Rayon parallelism in quantize loop

**File:** `src/quantize/mod.rs` (modify `quantize_streaming`, **~300–500 LOC honestly written** — proposal-time estimate of `~80 LOC` was unrealistic for the producer / worker-pool / serialiser shape with bounded MPSC channels, BTreeMap reordering buffer, on-channel `Tensor` payload sizing, and clean cancellation propagation. Robert lock 2026-04-25 round 2: correctness and speed dominate over LOC ceiling — pipeline shape stands).

**Change:** wrap the per-tensor quantize call (NOT the materialise / NOT the write) in a `rayon::ThreadPool` of size `min(num_cpus::get(), 16)`. The pool is bounded because Apple Silicon performance cores cap out at 12; oversubscription degrades.

**Pipeline shape (single-tensor unit of work):**

```text
producer (1 thread)        : materialise → channel-send
worker pool (n threads)    : channel-recv → quantize → channel-send
serialiser (1 thread)      : channel-recv → backend.write_tensor (in BTreeMap order)
```

Tensors quantise in parallel; writes are serialised in deterministic order (BTreeMap iteration) by the serialiser thread. This preserves byte-identical output (Decision 17) while giving N-way parallel quant work.

**Channel sizing** (back-pressure): the materialise→quantize channel has bounded capacity equal to the worker count; the quantise→write channel has bounded capacity equal to 2× worker count. Bounded channels prevent the producer from outrunning workers and inflating peak RSS.

**Tests:**
- `test_rayon_speedup_q4_apex_moe` — 8-thread quant of apex MoE q4 is `≥ 4×` faster than 1-thread (sub-linear due to memory-bandwidth saturation, but ≥ 4× is the gate).
- `test_rayon_byte_identical_to_serial` — output bytes identical between 1-thread and 8-thread serial-write modes.
- `test_rayon_no_oom_under_pressure` — 16-thread quant on a 32 GB synthetic MoE peaks at `≤ 40 GB` resident.

**Estimated LOC:** ~300–500 (channels + worker spawn + reordering buffer + cancellation propagation; was ~80 at proposal — see file note above).

### 6. Eliminate the P9b intermediate-GGUF dance

**Files:** `src/main.rs:680–778` (worktree), `src/inference/models/qwen35/activation_capture_real.rs`, `src/quantize/dwq_activation.rs`.

**Change:** add `RealActivationCapture::from_lazy_tensor_map(&LazyTensorMap, &Tokenizer) -> Result<Self>` — a constructor that builds a `Qwen35Model` directly from `LazyTensor`s (no intermediate GGUF write, no GGUF re-read). The convert pipeline calls this constructor with the in-memory `LazyTensorMap`; the model loads layer-by-layer from the lazy materialisers; activation capture runs against it; sensitivity is derived; DWQ quantises from the **same** `LazyTensorMap` (no re-read).

**Deletes:**
- `src/main.rs:715–738` — the `tempfile::tempdir()` + `emit_gguf_from_tensor_map(&intermediate_path)` block.
- `src/main.rs:739–753` — the `_drop_for_capture = std::mem::replace(&mut tensor_map, …)` workaround and re-read.
- `src/quantize/intermediate_moe_q8.rs` — the entire 239-line module (band-aid for F32-expand OOM; obsolete because `LazyTensorMap` never expands the experts to F32 in the first place).
- `src/backends/gguf.rs:emit_gguf_from_tensor_map` — used only by the dance.

**Adds:**
- `src/inference/models/qwen35/activation_capture_real.rs` — new constructor `from_lazy_tensor_map`. Implementation reads `Qwen35Model::load_from_lazy_tensor_map` (new in `src/inference/models/qwen35/weight_loader.rs`).
- `src/inference/models/qwen35/weight_loader.rs` — new function that materialises one tensor at a time on the GPU upload boundary; never holds the F32-expanded MoE in RAM.

**Net LOC:** −239 (intermediate_moe_q8 deletion) − 90 (main.rs dance) − 60 (emit_gguf_from_tensor_map) + 350 (`from_lazy_tensor_map` + `load_from_lazy_tensor_map`) = **−39 LOC net**.

**Cross-ADR correctness check:** ADR-012's P9b shipped `IntermediateMoeQ8Quantizer` because the F16-intermediate path expanded MoE experts to F32 on load and OOM'd at ~128 GB. With Decision 6's lazy upload, the model loads experts as native Q-blocks straight onto Metal — no F32 expansion ever — and the `IntermediateMoeQ8Quantizer` workaround becomes unreachable code, hence deletable.

**Tests:**
- `test_p9b_dance_eliminated` — assert `intermediate_moe_q8.rs` is removed; assert `cmd_convert` never writes a tempfile when `dwq_arch.requires_activation_capture()`.
- `test_activation_capture_from_lazy` — synthetic 4-layer qwen35 + 4-expert qwen35moe; `RealActivationCapture::from_lazy_tensor_map` produces sensitivity JSON byte-identical to the previous `from_intermediate_gguf` path on the same model.
- `test_apex_moe_capture_peak_rss` — apex MoE activation capture peak resident **≤ 36.3 GB** (33 GB measured on the existing intermediate-Q8 pipeline per ADR-012 P9b's real-model close + 10% headroom; locked 2026-04-26 in `docs/peer-parity-baselines-2026-04-26.md`). The number reflects the existing pipeline; **P4 re-measures on the new lazy-weight-loader path it lands** (Decision 6 + 8 deletions remove the IntermediateMoeQ8Quantizer and the F32 round-trip), and replaces this gate value with `<P4-measured> + 10%`. The dated exit condition lives in the baselines file; the gate in this body refreshes with each P4 iteration.

**Estimated LOC:** −39 net, but ~700 lines touched (delete + add).

### 7. Streaming MoE expert merge at quant time

**File:** `src/models/qwen35/moe.rs` (modify `merge_moe_experts_in_tensor_map`, ~150 LOC delta).

**Change:** the existing `merge_moe_experts_in_tensor_map` produces a single merged `[256, hidden, moe_inter]` tensor per layer, holds it in `tensor_map`, then the quantize pass walks it. Replace with `merge_moe_experts_lazy` which produces a `LazyTensor` whose closure performs the merge **at materialisation time**, emitting one merged tensor per layer; the quantize pass materialises layer-N, quantises it, writes it, drops it, then materialises layer-N+1.

Peak resident at quant time becomes:

```text
peak = 1 layer × 3 projections × 256 experts × hidden × moe_inter × dtype_bytes
     = 1 × 3 × 256 × 2048 × 768 × 2  (BF16, apex shape; hidden=2048, moe_inter=768)
     ≈ 2.4 GB per merged-projection tile
```

vs. eager all-layers-merged: ~80 GB on apex.

**Tests:**
- `test_lazy_moe_merge_one_layer_resident` — Arc<AtomicUsize> on the merge counter; assert at most 1 layer's worth of merged bytes is alive at any quant-loop tick.
- `test_lazy_moe_merge_byte_identical_to_eager` — synthetic 4-layer MoE merged eagerly vs lazily → quantised output byte-identical.

**Estimated LOC:** ~150.

### 8. `ActivationCapture::from_lazy_tensor_map` (cross-cut, ADR-013 boundary)

**File:** `src/inference/models/qwen35/activation_capture_real.rs` + `src/inference/models/qwen35/weight_loader.rs` (covered in Decision 6).

This Decision exists to lock the cross-ADR API contract. ADR-013 owns `ActivationCapture`; ADR-014 adds a constructor (`from_lazy_tensor_map`) without changing the trait. The trait method signatures are unchanged; downstream consumers in ADR-013 are unaffected.

**Cross-ADR sign-off:** ADR-013 P12 must accept this constructor's existence and the `weight_loader::load_from_lazy_tensor_map` addition. Documented in ADR-013's "Dependencies on other work" section as a same-author handoff (no separate review cycle required, but the change is named and traceable).

### 9. `Calibrator` trait (orthogonal axis)

**File:** `src/calibrate/mod.rs` (new module, ~250 LOC).

**File-layout decision (Robert lock 2026-04-25 round 2 — Layout A):** `src/quantize/dwq.rs`, `src/quantize/dwq_activation.rs`, `src/quantize/sensitivity.rs`, and `src/quantize/apex.rs` **move into `src/calibrate/`** as part of P7. Full path migration; **no re-export wrappers, no thin shims, no legacy `pub use` aliases**. All callers and test imports update to the new paths in one pass. P7's LOC budget grows for the import-rewrite churn (every `tests/convert_qwen35_*.rs`, every internal `use crate::quantize::dwq::*`, every `cmd_convert` dispatch site).

**Trait:**

```rust
pub trait Calibrator: Send + Sync {
    fn name(&self) -> &'static str;
    fn requires_forward_pass(&self) -> bool;
    fn calibrate(
        &mut self,
        model: &LazyTensorMap,
        meta: &ModelMetadata,
        corpus: &CalibrationCorpus,
        progress: &ProgressReporter,
    ) -> Result<CalibrationData, CalibrationError>;
}

pub enum CalibrationData {
    None,
    Imatrix(HashMap<TensorName, Vec<f32>>),    // per-column weights
    Dwq(SensitivityMap),                       // existing DWQ scoring
}
```

**Three implementations:**
- `NoneCalibrator` — `requires_forward_pass = false`; `calibrate` returns `CalibrationData::None`.
- `ImatrixCalibrator` — Decision 10.
- `DwqCalibrator` — wraps existing `dwq_activation.rs` orchestration; `requires_forward_pass = true` for qwen35/qwen35moe, false for others.

**Why `Send + Sync`:** calibrators may be invoked from worker threads under rayon (Decision 5), or held in a `--calibration auto` resolver.

**Tests:** trait-conformance test for each impl (round-trip a synthetic calibration on synthetic 4-layer Gemma-4); cross-impl test asserting `name()` is unique.

**Estimated LOC:** ~250 (trait + 3 impls + dispatch).

### 10. `ImatrixCalibrator` — pure-Rust port

**File:** `src/calibrate/imatrix.rs` (new, ~600 LOC).

**Algorithm (verbatim from `/opt/llama.cpp/tools/imatrix/imatrix.cpp`, ported to Rust):**

For each calibration sample (token sequence `T`):
1. Run forward pass on `T` (using ADR-013's `RealActivationCapture::run_calibration_prompt` for qwen35/qwen35moe, an arch-agnostic CPU forward for Gemma-4 — already exists at `src/inference/models/gemma4/forward_cpu.rs`).
2. At each `Linear` layer with weight `W ∈ [out, in]`, capture the **input activation** vector `x ∈ [in]` for every token in `T`.
3. Accumulate `imatrix_layer[in_col] += x[in_col]² * 1.0` (the `1.0` is the per-token weight; uniform).
4. After all samples: divide by total token count → `imatrix_layer[in_col] = mean(x[in_col]²)`.

Output: `HashMap<TensorName, Vec<f32>>` where each `Vec<f32>` has length equal to `W`'s input dimension. Cell `[c]` holds the importance of input column `c` for that tensor.

**Sidecar emission:** `--imatrix-out path.imatrix` writes a llama.cpp-compatible binary file (header + per-tensor float32 vectors), so the imatrix can be round-tripped through `llama-quantize --imatrix` for cross-validation.

**Per-column-weighted MSE in k-quant** (consumed by Decision 11's `OutputFormat::KQuant`): when fitting a Q4_K super-block, the codebook search minimises `Σ imatrix[col] · (W[row, col] − dequant(Q[row, col]))²` instead of plain `Σ (W − dequant(Q))²`. Implemented in `src/quantize/k_quant.rs` (new module, Decision 11).

**Cross-validation gate:** `tests/imatrix_cross_validate.rs` — port a synthetic Gemma-4 4-layer weight set, run hf2q's `ImatrixCalibrator`, run `llama.cpp/llama-imatrix` on the same weights + corpus, assert per-tensor max-element-difference `≤ 1e-4` (relative). Ensures the port is byte-precise to llama.cpp's algorithm before any peer-parity claim is made downstream.

**Tests:**
- `test_imatrix_per_column_accumulation` — synthetic single-layer linear, deterministic activations, hand-computed expected imatrix values.
- `test_imatrix_sidecar_round_trip` — write `.imatrix`, read with `llama-imatrix --tool dump`, assert structure.
- `test_imatrix_cross_validate_against_llama_cpp` — gate above (default-ignored, enabled in P10 peer-parity harness).
- `test_imatrix_pure_rust_no_python_dep` — `cargo metadata` shows no torch/numpy/python crate in the dep tree.

**Estimated LOC:** ~600 (impl ~400, tests ~200).

### 11. `OutputFormat` enum (orthogonal axis) + k-quant codebook

**Files:** `src/quantize/output_format.rs` (new, ~150 LOC) + `src/quantize/k_quant.rs` (new, ~700 LOC).

**Enum:**

```rust
pub enum OutputFormat {
    Flat(FlatType),                           // F16, BF16, Q2, Q4, Q8
    BitPair { base_bits: u8, sensitive_bits: u8 },  // 4-6, 4-8, 6-8, 2-8
    KQuant(KQuantType),                       // Q4_K_M, Q5_K_M, Q6_K
    KQuantAdaptive { target_bpw: f32 },       // adaptive across [Q3_K_S, Q6_K], replaces apex
}
```

**K-quant codebook implementation** (`src/quantize/k_quant.rs`): pure-Rust port of llama.cpp's `quantize_q4_K_M` / `quantize_q5_K_M` / `quantize_q6_K` from `/opt/llama.cpp/src/llama-quant.cpp`. Per-super-block: 256 elements grouped into 8 sub-blocks of 32; per-sub-block scale + min; per-element 4/5/6 bits; final super-block scale stored as F16. Algorithm is `quantize_iqK_K` in llama.cpp's `ggml-quants.c` — referenced verbatim, ported line-for-line.

**Per-column-weighted variant:** when called with `Some(imatrix)`, the inner codebook search minimises imatrix-weighted MSE (Decision 10 §"Per-column-weighted MSE").

**Pure-Rust no-link:** llama.cpp's k-quant impl is under MIT — Robert sovereignty rule per `feedback_hf2q_sovereignty.md` is **ported, not linked**. `cargo metadata` test asserts no `cc` / `cmake` / link to `libggml`.

**Byte-identity reference path (Robert lock 2026-04-25 round 2 — Apple Silicon).** The byte-identical gate is against llama.cpp's **NEON code path** (`quantize_row_q4_K` on `aarch64-apple-darwin`), **not** `quantize_row_q4_K_ref` (the scalar reference). The NEON path is what M-series users actually run; matching the scalar reference would not constitute peer parity in production. The pure-Rust port either uses `std::arch::aarch64` intrinsics or scalar code that replicates NEON's reduction order (associativity-sensitive horizontal sums in `make_qkx2_quants` must match). Cross-platform users on `x86_64` are not in scope for this gate.

**Tests:**
- `test_q4_k_m_byte_identical_to_llama_cpp` — synthetic deterministic input; produce hf2q Q4_K_M block; produce llama.cpp Q4_K_M block (built `aarch64-apple-darwin`, NEON path); `xxd` diff = empty.
- `test_q5_k_m_byte_identical` — same, Q5_K_M.
- `test_q6_k_byte_identical` — same, Q6_K.
- `test_kquant_imatrix_weighted_mse` — synthetic with high-magnitude column 0; without imatrix, all columns have equal error; with imatrix giving column 0 weight 100, column 0 error is `≤ 0.1×` the rest.
- `test_kquant_adaptive_target_bpw` — `KQuantAdaptive { target_bpw: 4.5 }` produces a per-tensor type assignment whose mean bpw across all tensors is within `± 0.05` of 4.5.

**Estimated LOC:** ~150 + ~700 + ~200 tests = ~1050.

### 12. CLI variant menu (coupled external naming)

**File:** `src/cli.rs` (modify `QuantMethod` enum, ~120 LOC delta).

**Final variant menu (Robert lock, Q9, 2026-04-25 — clean cut, no aliases, no users to break):**

| External `--quant` | Calibrator | OutputFormat | Peer baseline |
| ------------------ | ---------- | ------------ | ------------- |
| `f16` | None | Flat(F16) | — |
| `bf16` | None | Flat(BF16) | — |
| `q2` | None | Flat(Q2) | — |
| `q4` | None | Flat(Q4) | — |
| `q8` | None | Flat(Q8) | — |
| `q4_k_m` | None | KQuant(Q4_K_M) | llama.cpp uncalibrated |
| `q5_k_m` | None | KQuant(Q5_K_M) | llama.cpp uncalibrated |
| `q6_k` | None | KQuant(Q6_K) | llama.cpp uncalibrated |
| `imatrix-q4_k_m` | Imatrix | KQuant(Q4_K_M) | **llama.cpp imatrix Q4_K_M (primary)** |
| `imatrix-q5_k_m` | Imatrix | KQuant(Q5_K_M) | llama.cpp imatrix Q5_K_M |
| `imatrix-q6_k` | Imatrix | KQuant(Q6_K) | llama.cpp imatrix Q6_K |
| `imatrix-adaptive` | Imatrix | KQuantAdaptive | replaces `apex` |
| `dwq-4-6` | Dwq | BitPair(4,6) | **mlx-lm DWQ (primary)** |
| `dwq-4-8` | Dwq | BitPair(4,8) | mlx-lm DWQ |
| `dwq-6-8` | Dwq | BitPair(6,8) | mlx-lm DWQ |
| `dwq-2-8` | Dwq | BitPair(2,8) | mlx-lm DWQ |
| `auto` | (resolved) | (resolved) | per Decision 18 |

**Off-diagonal cells** (e.g. DWQ + KQuant, Imatrix + BitPair) are reachable only via the dev gate:

```bash
HF2Q_UNSAFE_EXPERIMENTS=1 hf2q convert <input> \
    --calibration imatrix --output-format bit-pair-4-6
```

The dev flags are documented in `docs/converting-a-model.md` under a "For maintainers" section, not the user-facing flow (Paige's note, Q7).

**`apex` deletion:** the existing `--quant apex` is **removed**, not aliased. Users reach the equivalent functionality via `--quant imatrix-adaptive`. Per Q9: clean cut.

**Tests:**
- `test_cli_variant_menu_complete` — every external variant string parses to the correct `(Calibrator, OutputFormat)` tuple.
- `test_cli_apex_removed` — `--quant apex` errors with "unknown variant `apex`; did you mean `imatrix-adaptive`?".
- `test_cli_dev_gate_off_diagonal` — without `HF2Q_UNSAFE_EXPERIMENTS=1`, `--calibration X --output-format Y` is rejected.

**Scope confirmation (Robert lock 2026-04-25 round 2):** all 17 variants ship in this ADR. The full menu is the deliberate long-term shape, not a candidate for trim-to-essentials. Variants without immediate user demand (`dwq-6-8`, `dwq-2-8`, `imatrix-q5_k_m`, `imatrix-q6_k`, `imatrix-adaptive`) ship now to prove the orthogonal `Calibrator × OutputFormat` design at scale rather than discovering trait gaps in follow-up ADRs.

**Estimated LOC:** ~120.

### 13. Migration policy

**Files:** `src/cli.rs` + `docs/converting-a-model.md`.

Per Robert (Q9, 2026-04-25): *"we don't have any users yet. B — but don't leave old cruft. There's no compatibility concern."*

**Deleted variants (no aliases, no deprecation warnings):**
- `mixed-2-6`, `mixed-3-6`, `mixed-4-6` — uncalibrated bit-pair. Reachable in dev mode via `HF2Q_UNSAFE_EXPERIMENTS=1 --calibration none --output-format bit-pair-N-M` if absolutely needed; not exposed.
- `apex` — replaced by `imatrix-adaptive`.
- `dwq-mixed-4-6`, `dwq-mixed-4-8`, `dwq-mixed-6-8`, `dwq-mixed-2-8` — replaced by `dwq-4-6`, `dwq-4-8`, `dwq-6-8`, `dwq-2-8` (cleaner names; `mixed-` prefix was a vestige of the underlying `MixedBitQuantizer`).

**`MixedBitQuantizer` (`src/quantize/mixed.rs`, 514 LOC):** retained internally as the implementation backing `OutputFormat::BitPair`; not exposed to the CLI.

**Tests:**
- `test_legacy_quant_variants_rejected` — every deleted variant string causes a typed CLI error.

**Estimated LOC:** ~50 (mostly deletion).

### 14. Both backends with shared streaming pipeline

**Files:** `src/backends/{gguf,safetensors_out,mod}.rs` (modify, ~400 LOC delta).

**Change:** both backends implement `StreamingBackend` (Decision 4) and consume from the same `quantize_streaming` loop. The existing `OutputBackend::requires_native_quantization()` shortcut on `SafetensorsBackend` is **removed** — safetensors output now goes through the IR-level quantize loop with the chosen `Calibrator` × `OutputFormat`, exactly like GGUF.

**Safetensors output layout (Robert lock 2026-04-25 round 2):** `--format safetensors` emits an **mlx-lm-style directory** (`<output>/`), **not** a single `.safetensors` file. The directory contains:
- `config.json` (model architecture, copied/derived from the input HF repo)
- tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` etc.)
- `generation_config.json`
- one or more sharded `model-NNNNN-of-MMMMM.safetensors` files whose **per-shard headers** carry mlx-lm-compatible quant metadata (per-tensor `scales`, `zeros`, `bits` keys)

Appendix A's `Qwen3.6-27B-dwq-4-6/` notation is canonical. This directory layout makes hf2q's safetensors output **directly loadable by `mlx_lm.load(<output>)`** — establishing peer parity with `mlx_lm.convert`.

**Cross-validation:** P10 includes a test that `mlx_lm.load(hf2q_dwq46.safetensors)` succeeds and produces logits within `1e-3` (cosine) of `hf2q.serve(hf2q_dwq46.safetensors)` on a deterministic 16-prompt smoke set.

**Tests:**
- `test_safetensors_streaming_byte_identical_to_eager` — Gemma-4 → safetensors, eager pipeline vs streaming pipeline → byte-identical (un-calibrated path; Decision 17).
- `test_safetensors_dwq46_loads_in_mlx_lm` — `--quant dwq-4-6 --format safetensors` produces a file `mlx_lm.load` accepts.
- `test_safetensors_imatrix_q4km_round_trip` — `--quant imatrix-q4_k_m --format safetensors` produces a file that hf2q's own `serve` re-loads with byte-identical logits.

**Estimated LOC:** ~400 (~150 each backend + ~100 metadata adapter).

### 15. Peer-parity gates (Robert lock, Q3)

**File:** `tests/peer_parity_gates.rs` (new, ~500 LOC) + `docs/converting-qwen35.md` (table copy).

**Locked gate matrix:**

| Axis | Value |
| ---- | ----- |
| Primary peer | `llama.cpp` (`convert_hf_to_gguf.py` + `llama-quantize --imatrix`) |
| Secondary peer | `mlx_lm.convert` |
| Reference models | `Qwen/Qwen3.6-27B` dense + `jenerallee78/Qwen3.6-35B-A3B-abliterix-ega-abliterated-apex` MoE |
| Hardware | M5 Max, 128 GB |
| Speed metric | wall-clock convert+quant; `/usr/bin/time -p`; **`1 warmup run discarded → 60 s thermal cooldown → 3 timed runs, median wins`**, both peers warm |
| Speed tolerance | hf2q median wall ≤ 1.10× peer median wall |
| Memory metric | peak RSS; `/usr/bin/time -l` `maximum resident set size`; recorded across the 3 timed runs, median wins |
| Memory tolerance | hf2q median peak ≤ 1.10× peer median peak |
| Correctness — calibrated quality | wikitext2 PPL on hf2q output ≤ peer output × 1.02 |
| Correctness — vs F16 reference | KL(hf2q output ∥ F16) ≤ 0.02 nats (carry-over from ADR-012 Decision 17) |
| Determinism | byte-identical output across two fresh cold runs (un-calibrated paths only; Decision 17) |

**Gate cells** (P10/P11 must measure all 8 model × backend × calibrator combinations):

| Model | Backend | Calibrator | Peer | Speed gate | RSS gate | PPL gate |
| ----- | ------- | ---------- | ---- | ---------- | -------- | -------- |
| 27B dense | GGUF | None (q4_k_m) | llama.cpp uncalibrated Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 27B dense | GGUF | Imatrix (imatrix-q4_k_m) | llama.cpp imatrix Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 27B dense | safetensors | DWQ (dwq-4-6) | mlx_lm DWQ | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 27B dense | GGUF | DWQ (dwq-4-6) | (no peer; vs hf2q current pipeline) | ≤ 1.0× | ≤ 0.50× | ≤ 1.0× |
| apex MoE | GGUF | None (q4_k_m) | llama.cpp uncalibrated Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| apex MoE | GGUF | Imatrix (imatrix-q4_k_m) | llama.cpp imatrix Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| apex MoE | safetensors | DWQ (dwq-4-6) | mlx_lm DWQ | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| apex MoE | GGUF | DWQ (dwq-4-6) | (no peer; vs hf2q current pipeline) | ≤ 1.0× | ≤ 0.50× | ≤ 1.0× |

Last column: hf2q DWQ→GGUF has no direct peer (mlx-lm doesn't emit GGUF, llama.cpp doesn't ship DWQ); the gate is hf2q-vs-hf2q-current-pipeline. RSS must drop by **≥ 50%** (the central correctness/sanity claim of this ADR — streaming halves peak resident).

**Why warmup-discarded + thermal cooldown** (Robert lock 2026-04-25 round 2). The Apple-Silicon ML benchmark community's actual practice is `warmup → cooldown → timed runs`, not "system reboot, single cold run." Sources: Eduard Stere's llama.cpp Apple-Silicon harness explicitly warms before timing; llama.cpp discussion #4167 pins both peers to identical builds for fairness; `mlx_transformers_benchmark`'s `cooldown_time_fraction` manages thermal drift on shared-memory M-series. Warmup-discarded ensures both peers measure with warm Metal-shader caches, sidestepping the (Apple-undocumented) persistence behaviour of `~/Library/Caches/com.apple.metal/`. The 60 s thermal cooldown holds the M5 Max within the same thermal envelope across the three timed iterations. Cold-start latency, if separately interesting to surface to users, can be reported as a `cold_first_run` field alongside the median — but it does not gate.

**Test infrastructure:** `tests/peer_parity_gates.rs` orchestrates the runs; results land as a markdown table in `docs/peer-parity-results-2026-04-25.md` (or whatever date P11 closes); the closing commit cites the table inline.

**Estimated LOC:** ~500 test harness, ~50 gate definitions.

### 16. PPL + KL evaluation methodology

**File:** `tests/fixtures/ppl-corpus/wikitext2.tokens` (already exists from ADR-012 P9) + `src/arch/conformance.rs::ppl_kl_eval` (already exists).

**No new code** — Decision 16 reuses ADR-012's already-built PPL + KL infrastructure verbatim. Cross-peer comparison adds three columns to the existing reporter:

```
hf2q_ppl  llama_ppl  mlx_ppl   hf2q_kl_vs_f16  llama_kl_vs_f16  mlx_kl_vs_f16
```

**Eval corpus (Robert lock 2026-04-25 round 2):** wikitext-2 **full test split (~280k tokens)**, deterministic SHA-256 sidecar. Apples-to-apples with llama.cpp / mlx-lm published Q4_K_M PPL numbers — this is what users compare against when picking a quant. ADR-012's existing 512-token fixture (`tests/fixtures/ppl-corpus/wikitext2.tokens`) is preserved as a fast smoke check (default `cargo test`); the full-split eval lives at `tests/fixtures/ppl-corpus/wikitext2-full.tokens` (new in P10, ~280k tokens, ~700 KB on disk after BPE encoding) and runs only in the `#[ignore]`-gated peer-parity harness.

**Cross-corpus disjointness gate:** ADR-012's `tests/calibration_eval_disjoint.rs` already enforces zero overlap between calibration and PPL eval corpora. ADR-014 inherits this — every calibrator added in this ADR (Imatrix) re-runs the disjointness check.

**Estimated LOC:** ~50 (column additions in the reporter).

### 17. Determinism contract

**Files:** `tests/determinism_*.rs` (new, ~200 LOC).

**Contract:**

| Path | Determinism level | Test |
| ---- | ----------------- | ---- |
| `--quant f16`, `bf16` | Byte-identical to current pipeline | `test_determinism_f16_byte_identical_to_current` |
| `--quant q2`, `q4`, `q8` | Byte-identical to current pipeline | `test_determinism_flat_byte_identical_to_current` |
| `--quant q4_k_m`, `q5_k_m`, `q6_k` | Byte-identical to llama.cpp `--quant Q4_K_M` etc. | `test_determinism_kquant_vs_llama_cpp` |
| `--quant imatrix-*` | Byte-identical across two cold hf2q runs (re-blessed snapshot, not back-compat) | `test_determinism_imatrix_two_cold_runs` |
| `--quant dwq-*` | Byte-identical across two cold hf2q runs (re-blessed snapshot) | `test_determinism_dwq_two_cold_runs` |
| `--quant imatrix-adaptive` | Byte-identical across two cold hf2q runs | `test_determinism_imatrix_adaptive_two_cold_runs` |
| `--quant auto` | Resolves to a named variant deterministically; that named variant satisfies its own row | `test_determinism_auto_resolves_then_named_holds` |

**Reasoning** (Robert, Q3 implicit + Winston, Q10):

- Un-calibrated paths (Flat, K-quant uncalibrated): the streaming pipeline changes nothing about the *math* — same input bytes → same output bytes. Byte-identical to current pipeline is achievable and the bar.
- K-quant uncalibrated additionally byte-identical to llama.cpp `llama-quantize --quant Q4_K_M` because Decision 11's k_quant.rs is a line-for-line port. This is the strongest correctness claim in the ADR — porting llama.cpp's k-quant codebook to pure Rust and proving byte-identity in CI.
- Calibrated paths: re-blessed because (a) streaming changes processing order, (b) sensitivity caching changes scoring derivation, (c) imatrix port may have float-precision differences from llama.cpp's C impl (mitigated by Decision 10's cross-validation `≤ 1e-4`). The bar is "deterministic across two cold runs of the same hf2q binary on the same input"; not back-compatible with current-pipeline DWQ outputs.

**Estimated LOC:** ~200 tests.

### 18. `--quant auto` routing

**File:** `src/intelligence/auto_quant.rs` (modify `AutoResolver`, ~80 LOC delta).

**Resolution table:**

| Model class | Hardware | Resolved variant |
| ----------- | -------- | ---------------- |
| Dense ≤ 7B | any | `imatrix-q4_k_m` |
| Dense 7B–30B | any | `imatrix-q4_k_m` |
| Dense > 30B | < 64 GB RAM | `imatrix-q4_k_m` |
| Dense > 30B | ≥ 64 GB RAM | `imatrix-q5_k_m` |
| MoE any size | < 96 GB RAM | `dwq-4-6` |
| MoE any size | ≥ 96 GB RAM | `dwq-4-8` |
| Architecture has `ArchEntry::auto_override = Some(v)` | any | `v` (per-arch override) |

**Why MoE → DWQ:** DWQ's per-tensor sensitivity allocation handles MoE's expert heterogeneity better than imatrix's per-column diagonal weighting (the hot/cold pattern across experts is along the `expert` axis, not the `column` axis).

**Why dense → imatrix:** imatrix is the llama.cpp ecosystem default; users coming from `llama-cli` get the output they expect. Plus three years of community PPL tuning means imatrix Q4_K_M is the well-validated point on the perf/quality curve.

**Tests:**
- `test_auto_dense_27b` — resolves to `imatrix-q4_k_m`.
- `test_auto_moe_apex` — resolves to `dwq-4-6` on a 64 GB box, `dwq-4-8` on a 128 GB box.
- `test_auto_arch_override` — when `ArchEntry::auto_override = Some("imatrix-q5_k_m")`, that wins.

**Estimated LOC:** ~80.

### 19. Future-calibrator extensibility (door open, no implementations)

**File:** `docs/calibrator-onboarding.md` (new, ~200 LOC of docs — explicitly *docs*, not stubs).

**One paragraph in this ADR:** `Calibrator` (Decision 9) is shaped to admit AWQ (per-channel quantization-aware), GPTQ (Hessian-based reordering), and AQLM (additive quantization) without trait changes. Adding a new calibrator: implement `Calibrator`, add an `OutputFormat` variant if its output codebook is novel (otherwise reuse), wire one CLI variant, write the cross-validation gate. Future ADRs (one per calibrator) own the implementation; ADR-014 ships only DWQ + Imatrix + None.

**No stub code lands in `src/calibrate/{awq,gptq,aqlm}.rs`** — per mantra ("No stub (todo later) code"). The door is shaped, not propped open.

**Estimated LOC:** 0 code, ~200 docs.

### 20. Cancellation safety with streaming writes

**File:** `src/main.rs:cmd_convert` (modify Ctrl+C handler, ~50 LOC delta).

**Contract:**
1. Output file is opened in a sibling tempdir (`<output_path>.tmp.XXXXXX/`); writes go there.
2. `StreamingBackend::finalize()` does `std::fs::rename(tempdir/output_basename, final_path)` — atomic on POSIX/APFS.
3. SIGINT handler: sets `INTERRUPTED.store(true)`; the streaming loop checks at every tensor boundary and exits cleanly; the tempdir is removed in the existing cleanup handler (`src/main.rs:363`).
4. **No partial output file ever appears at `final_path`** — that path either holds a complete file or doesn't exist.
5. **No tensor-level rollback.** Per mantra ("no fallback"), we don't pretend partial outputs are recoverable. Cancellation = full restart; the calibration cache (Decision 5) avoids redoing the expensive work on restart.

**Tests:**
- `test_cancellation_no_partial_at_final_path` — kill mid-stream; assert `final_path` does not exist; assert no `*.tmp.*` directories leaked.
- `test_cancellation_calibration_cache_survives` — kill mid-stream after calibration completes; assert `~/.cache/hf2q/sensitivity/<key>.json` is intact and reusable on next run.

**Estimated LOC:** ~50.

### 21. Sovereignty preservation (explicit non-changes)

**No file changes.** Decision 21 exists to lock the ADR's position on which divergences are *deliberate* and not subject to peer-parity scrutiny.

**Preserved:**
- **One-binary, single-process convert+quantize.** llama.cpp's two-stage pipeline (`convert_hf_to_gguf.py` → `llama-quantize`) is **not** adopted. Pro carried: single context, single cancellation, no intermediate F16 artefact persisted. Con accepted: cannot replay just the quantize step.
- **Per-arch hand-ported transforms.** Phases 1.4/1.5/1.6/1.7 stay as Rust translations of `convert_hf_to_gguf.py`. No `pyo3` runtime dep, no `torch` link.
- **Pure Rust, mlx-native sole sibling dep.** `cargo metadata` test asserts no `python-sys`, `pyo3`, `numpy`, `torch-sys`, `libggml-sys`, `cmake`-driven build dep.

**Newly explicit (ADR-014 lock):**
- **Calibrator portability is not a sovereignty claim.** DWQ comes from MLX; imatrix comes from llama.cpp; both ship in hf2q because **both algorithms are published peer-equivalent techniques**. The sovereignty claim is on the *toolchain* (pure-Rust port, no Python runtime), not on the algorithm.
- **Runtime vs test-time scope (Robert lock 2026-04-25 round 2).** Sovereignty applies at **runtime** — the shipped `hf2q` binary contains no Python interpreter, no torch/numpy/python crate, no link to libggml. **Test-time** Python is permitted only behind `#[ignore]`-gated parity-harness tests run by `scripts/peer_parity_run.sh`, which shells out to `mlx_lm.convert`, `mlx_lm.load`, `llama-imatrix`, and `convert_hf_to_gguf.py` for cross-validation. Default `cargo test` set stays pure Rust. `test_sovereignty_no_python_dep` only walks the runtime crate dep graph, not the test-harness shell-out targets.

**Tests:**
- `test_sovereignty_no_python_dep` — `cargo metadata --format-version=1 | jq` walks the **runtime** dep tree, asserts none of the forbidden crates appear. Test-time `mlx-lm`/`llama-imatrix` shell-outs are not in scope.
- `test_sovereignty_no_libggml_link` — `cargo build --target aarch64-apple-darwin -v 2>&1 | grep -v libggml`.

**Estimated LOC:** ~50 sovereignty-gate tests.

### 22. Re-emit ADR-012's four DWQ GGUFs under streaming pipeline (closure AC)

**Files:** none (artefact production).

**Deliverables (P11):**
- `models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf` (re-emitted)
- `models/qwen3.6-27b-dwq48/qwen3.6-27b-dwq48.gguf` (re-emitted)
- `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf` (re-emitted)
- `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48.gguf` (re-emitted)
- The same four under `--format safetensors` (mlx-lm-style directories per Decision 14) for the mlx-lm peer column.

**Shipped vs ephemeral (Robert lock 2026-04-25 round 2).** P11 ships **only the eight artefacts above** (four DWQ GGUFs + four DWQ safetensors directories) to `models/`. Decision 15's gate matrix also requires measuring `imatrix-q4_k_m` cells on both 27B dense and apex MoE — those convert runs **produce ephemeral measurement files in scratch**, the harness records PPL/RSS/wall, and the file is deleted alongside the peer's `<peer>-Q4_K_M.gguf` per R13's disk-pressure mitigation. **No `imatrix-*` artefact lives in `models/` after P11 closes.** Users who want imatrix-q4_k_m artefacts run convert themselves; the ADR ships only the ADR-012 carry-over commitments.

**Comparison table** (lands in P11 closing commit and `docs/peer-parity-results-<date>.md`):

| Artifact | Pre-ADR-014 size | Post-ADR-014 size | Pre wall | Post wall | Pre peak RSS | Post peak RSS | Pre PPL | Post PPL | KL(post ∥ pre) |
| -------- | ---------------- | ----------------- | -------- | --------- | ------------ | ------------- | ------- | -------- | -------------- |
| (8 rows: 4 GGUFs × 2 backends — but only GGUF for these four; safetensors variants emit alongside as part of P11) | ... | ... | ... | ... | ... | ... | ... | ... | ... |

The ADR closes when this table is filled in with measured numbers and every row meets the gates from Decision 15.

---

## Phase plan

Phases are **dependency-ordered**. Each phase has a single owner claim per `feedback_swarm_sequential_when_shared_build.md` (shared Cargo `target/` = sequential). **Exception (Robert lock 2026-04-25 round 2):** P11 (real-model artefact production, GPU-bound) and P12 (docs-only, no GPU, trivial build pressure) **may run in parallel via separate CFA worktrees** (each worktree has its own `target/`; the sequential rule applies to *shared* `target/`, which CFA worktrees split). The Totals' "~6–8 weeks if P11 reference artefact production runs in parallel with downstream phases on the same M5 Max" assumption rests on this exception.

### P0 — Lazy tensor primitive + lazy safetensors reader (+ Decision 6 measurement spike)

**Scope:** Decisions 1 + 2 + Decision 6's empirical gate-value derivation.

**Dependency:** real ADR-012 close (R14: four DWQ GGUFs verifiably load in `llama-cli`).

**Deliverables:**
- `src/ir/lazy.rs` (new, ~250 LOC).
- `src/input/safetensors.rs` modified (`Arc<Mmap>` lifetime, closure-based materialiser).
- 12 unit tests + 2 integration tests.
- **Decision 6 measurement spike (Robert lock 2026-04-25 round 2):** invoke the existing pipeline once on apex MoE — `cargo run --release --bin hf2q -- convert <apex> --quant dwq-4-6 --measure-only` (or equivalent existing instrumentation) — and record peak RSS during the forward-pass-with-capture step (counting both Q-block weights resident on Metal **and** F32 activation tensors emitted at every SDPA / router / expert-matmul boundary). The recorded number + 10% headroom becomes Decision 6's `test_apex_moe_capture_peak_rss` gate value. Result lands in `docs/peer-parity-baselines-<P0-close-date>.md` and is inlined into Decision 6's bullet.

**Acceptance:** Decisions 1 + 2 criteria met. Apex MoE peak-RSS gate (`≤ 8 GB` during `LazyTensorMap` iteration) passes. Gemma-4 byte-identical-to-eager regression passes. Decision 6's gate value committed to `docs/peer-parity-baselines-<date>.md` and inlined into the Decision 6 body.

**Estimated LOC:** ~400 (impl + tests) + ~50 (measurement spike harness, if not already in `tests/streaming_pipeline_apex_moe.rs`).

### P1 — Lift Phase 1.4–1.7 transforms to lazy

**Scope:** Decision 3.

**Dependency:** P0 green.

**Deliverables:** `src/main.rs:457–582` (worktree) signatures take `&mut LazyTensorMap`; `src/models/qwen35/{dense,moe}.rs` transforms become `fn(LazyTensor) -> LazyTensor`; 6 unit tests covering each transform's lazy-vs-eager byte-identity.

**Acceptance:** Decision 3 criteria met. Synthetic 4-layer Qwen3.5 + Qwen3.5-MoE eager-vs-lazy transform output byte-identical.

**Estimated LOC:** ~400.

### P2 — Streaming quantize loop

**Scope:** Decisions 4 + 7.

**Dependency:** P1 green.

**Deliverables:** `src/quantize/mod.rs::quantize_streaming` (rewrite); `StreamingBackend` trait; `src/backends/{gguf,safetensors_out}.rs` impls; `src/models/qwen35/moe.rs::merge_moe_experts_lazy`; 4 integration tests.

**Acceptance:** Decisions 4 + 7 criteria met. Apex MoE peak RSS during convert ≤ 44 GB. Gemma-4 `--quant q4` byte-identical.

**Estimated LOC:** ~600.

### P3 — Rayon parallelism in quantize loop

**Scope:** Decision 5.

**Dependency:** P2 green.

**Deliverables:** producer / worker-pool / serialiser channel topology in `src/quantize/mod.rs`; 3 integration tests.

**Acceptance:** Decision 5 criteria met. 8-thread quant of apex MoE q4 ≥ 4× faster than 1-thread; 8-thread output byte-identical to 1-thread.

**Estimated LOC:** ~150.

### P4 — Eliminate the P9b intermediate-GGUF dance

**Scope:** Decisions 6 + 8.

**Dependency:** P0 + P2 green.

**Deliverables:** `RealActivationCapture::from_lazy_tensor_map`; `weight_loader::load_from_lazy_tensor_map`; deletion of `src/quantize/intermediate_moe_q8.rs`, `src/main.rs:715–753`, `src/backends/gguf.rs::emit_gguf_from_tensor_map`; 3 integration tests.

**Acceptance:** Decisions 6 + 8 criteria met. Apex MoE DWQ activation capture peak RSS ≤ 35 GB **with no tempfile written**.

**Estimated LOC:** −39 net (700 lines touched).

### P5 — Sensitivity-JSON cache

**Scope:** Decision 9 (DWQ-side caching) — note: this is the per-(model, corpus) cache mentioned in item #5; orthogonal to the calibrator-trait split in P7.

**Dependency:** P2 green.

**Deliverables:** `~/.cache/hf2q/sensitivity/{key}.json` cache; key = SHA-256(model SHA, corpus SHA, sensitivity-algorithm version); cache lookup before forward pass; 4 unit tests.

**Acceptance:** running `hf2q convert <apex> --quant dwq-4-6` then `hf2q convert <apex> --quant dwq-4-8` (different bit-pair, same model) skips the second forward pass and consumes the cached sensitivity map.

**Estimated LOC:** ~300.

### P6 — Imatrix calibrator (pure-Rust port)

**Scope:** Decision 10.

**Dependency:** P0 green (uses `LazyTensorMap`).

**Deliverables:** `src/calibrate/imatrix.rs`; `.imatrix` sidecar emitter; cross-validation gate against `llama-imatrix`; 8 unit tests + 1 cross-validation integration test.

**Acceptance:** Decision 10 criteria met. Cross-validation gate green (per-tensor max-element-difference ≤ 1e-4 vs llama.cpp on a 4-layer synthetic Gemma-4).

**Estimated LOC:** ~600.

### P7 — `Calibrator` × `OutputFormat` orthogonal split

**Scope:** Decisions 9 + 11.

**Dependency:** P6 green.

**Deliverables:**
- `src/calibrate/mod.rs::Calibrator` trait + `NoneCalibrator`, `DwqCalibrator`, `ImatrixCalibrator` impls.
- **Full move (Decision 9 Layout A — Robert lock 2026-04-25 round 2):** `src/quantize/dwq.rs`, `src/quantize/dwq_activation.rs`, `src/quantize/sensitivity.rs`, `src/quantize/apex.rs` move into `src/calibrate/`. No re-exports, no shims, no legacy `pub use` aliases. All callers and tests update to new paths in one pass.
- `src/quantize/output_format.rs::OutputFormat` enum.
- `src/quantize/k_quant.rs` (pure-Rust k-quant codebook).
- Refactor of `cmd_convert` to use the orthogonal pair.
- 12 unit tests + 4 integration tests.

**Acceptance:** Decisions 9 + 11 criteria met. Q4_K_M / Q5_K_M / Q6_K byte-identical-to-llama.cpp (NEON path) gate passes. All `tests/convert_qwen35_*.rs` and other test files compile against the new `src/calibrate/` paths; no `use crate::quantize::dwq::*` lingers.

**Estimated LOC:** ~1300 (was ~1050; +~250 for the `dwq*`/`sensitivity`/`apex` full-move import-path churn — Decision 9 Layout A).

**iter-8 (2026-04-27, `7521f97`):** Layout A migration landed — `src/quantize/{dwq,dwq_activation,sensitivity,apex}.rs` moved into `src/calibrate/` via `git mv` (history preserved); all four call sites in `src/main.rs` (DWQ + apex dispatch) and the two internal cross-module imports rewritten to `crate::calibrate::*`. `src/calibrate/imatrix_calibrator.rs` adds `ImatrixCalibrator` (`Calibrator` trait impl, `name()="imatrix"`, `requires_forward_pass()=true`, drives `ImatrixCollector` over a `CalibrationCorpus` via `ActivationCapture::run_calibration_prompt`, returns `CalibrationData::ImatrixWithStats`) + 8 unit tests (trait conformance, empty-corpus error, synthetic-Gemma-4 round trip, hand-computed token-count match, metadata mismatch error, bridge no-drop guard, object-safety, Send+Sync). `src/calibrate/dwq_calibrator.rs` adds `DwqCalibrator` (`name()="dwq"`, `requires_forward_pass()` consults `DwqArch::requires_activation_capture()`, `Qwen35Dense`/`Qwen35MoE` with `capture==None` returns typed `CalibrationError::ForwardPassUnavailable` (no silent weight-space fallback per ADR-012 D13), `Other` arch returns empty `CalibrationData::Dwq` map (weight-space contract — not a fallback)) + 7 unit tests. `select_calibrator` helper in `src/main.rs` returns `Box<dyn Calibrator>` keyed by `cli::QuantMethod` (DwqMixed* → DwqCalibrator; everything else including Apex → NoneCalibrator); invoked once per cmd_convert run with `tracing::info!` log. `tests/calibrate_dispatch.rs` integration test (6 tests) mirrors the dispatch table with compile-time exhaustiveness. Full restructure of `cmd_convert` → `(Calibrator, OutputFormat)` deferred to P2 iter-2 (separate task).

### P8 — CLI rename + final variant menu

**Scope:** Decisions 12 + 13.

**Dependency:** P7 green.

**Deliverables:** `src/cli.rs::QuantMethod` enum updated; deleted variants error with helpful messages; `--calibration` + `--output-format` dev gate behind `HF2Q_UNSAFE_EXPERIMENTS=1`; 8 CLI integration tests.

**Acceptance:** Decisions 12 + 13 criteria met. Old variant names (`apex`, `mixed-2-6`, `dwq-mixed-4-6`) error cleanly; new variant names dispatch to the correct (Calibrator, OutputFormat) tuple.

**Estimated LOC:** ~170.

### P9 — Safetensors backend integration with calibrators

**Scope:** Decision 14.

**Dependency:** P7 green.

**Deliverables:** `SafetensorsBackend` implements `StreamingBackend`; `requires_native_quantization` removed; mlx-lm-compatible quant metadata writer; 6 integration tests including `mlx_lm.load` round-trip.

**Acceptance:** Decision 14 criteria met. `--quant dwq-4-6 --format safetensors` produces a file `mlx_lm.load` accepts; `--quant imatrix-q4_k_m --format safetensors` round-trips through hf2q's own serve with byte-identical logits.

**Estimated LOC:** ~700.

### P10 — Peer-parity benchmark harness

**Scope:** Decisions 15 + 16.

**Dependency:** P3 + P6 + P9 green.

**Deliverables:** `tests/peer_parity_gates.rs` (orchestration); `scripts/peer_parity_run.sh` (cold-cache reboot wrapper); `tools/llama_cpp_runner.rs` (subprocess wrapper around `convert_hf_to_gguf.py` + `llama-quantize`); `tools/mlx_lm_runner.rs` (subprocess wrapper around `mlx_lm.convert`); per-cell measurement collector; markdown table emitter.

**Acceptance:** harness runs end-to-end on a 27B dense smoke model (~6 GB BF16 → ~3 GB Q4_K_M) and emits a complete table for the 8 cells against both peers; numbers are real but not yet meeting the gates (apex MoE comes in P11).

**Estimated LOC:** ~600 + ~100 wrapper scripts.

### P11 — Re-emit ADR-012's four DWQ GGUFs + measured gate close

**Scope:** Decision 22.

**Dependency:** all prior phases green.

**Deliverables:**
- `models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf` (re-emitted under streaming).
- `models/qwen3.6-27b-dwq48/qwen3.6-27b-dwq48.gguf` (re-emitted).
- `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/...gguf` (re-emitted).
- `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/...gguf` (re-emitted).
- Same four under `--format safetensors` for the mlx-lm peer column.
- `docs/peer-parity-results-<close-date>.md` — full 8×3 table, per cell: pre / post / peer numbers + verdict (pass/fail per gate).
- Closing commit message inlines the table verdict and links to the markdown.

**Acceptance:** every cell passes its gate per Decision 15. If any cell fails: per mantra, this phase reopens until it passes — no deferred cells, no "we'll fix the apex MoE imatrix-q4_k_m row in ADR-N+1." The ADR closes only when every cell is green.

**Estimated LOC:** ~200 (mostly artefact production scripts + commit-message templating).

### P12 — Documentation refresh

**Scope:** docs only.

**Dependency:** P11 green.

**Deliverables:**
- `docs/converting-a-model.md` rewritten — generic convert reference using new variant names, calibrator section, dev-gate maintainer note.
- `docs/converting-qwen35.md` updated — qwen35 / qwen35moe canonical commands using new variant names.
- `docs/shipping-contract.md` updated — peer-parity gates section.
- `docs/calibrator-onboarding.md` (new — Decision 19) — how to add AWQ/GPTQ/AQLM in a future ADR.
- ADR-012 status section updated to reflect P11's re-emission of its four GGUFs.

**Acceptance:** every command in the docs runs successfully on a fresh hf2q binary; markdown link checker green.

**Estimated LOC:** ~600 docs (no code).

### Totals

- **Code LOC:** ~5,200 across 13 phases (P0 400 + P1 400 + P2 600 + P3 150 + P4 −39 net + P5 300 + P6 600 + P7 1050 + P8 170 + P9 700 + P10 700 + P11 200 = 5,231).
- **Docs LOC:** ~800 (P12 600 + scattered phase doc updates).
- **Test LOC:** ~1,500 (every phase carries tests; estimate is conservative — ADR-012's tests-to-code ratio was ~0.3, suggesting ~1,560).
- **Net deletions:** ~340 LOC (intermediate_moe_q8 module + P9b dance + emit_gguf_from_tensor_map + apex CLI handling + mixed-N-M variants).

Estimated calendar time at single-author cadence (Robert): ~6–8 weeks if P11 reference artefact production runs in parallel with downstream phases on the same M5 Max.

---

## Test strategy

### Unit tests (per phase)
Every phase carries unit tests covering the new types, traits, and pure functions. Listed inline with each Decision body. Estimated total: ~70 unit tests.

### Regression tests (Chesterton's fence)
Byte-identical regression for the un-calibrated paths against the *current* pipeline:
- Gemma-4 `--quant f16` / `q4` / `q8` snapshot SHAs (already exist for `q4` per ADR-012 P5/P11 fence).
- Qwen3.5 4-layer synthetic `--quant f16` / `q4` snapshot SHAs.

Re-blessed snapshots for calibrated paths (recorded as new SHA in the closing commit; subsequent runs must match):
- `dwq-4-6` / `dwq-4-8` / `dwq-6-8` / `dwq-2-8` on Qwen3.5 4-layer synthetic.
- `imatrix-q4_k_m` / `imatrix-q5_k_m` / `imatrix-q6_k` on Qwen3.5 4-layer synthetic.
- `imatrix-adaptive` on Qwen3.5 4-layer synthetic.

### Integration tests
- `tests/streaming_pipeline_apex_moe.rs` — end-to-end apex MoE convert with peak-RSS instrumentation.
- `tests/lazy_tensor_correctness.rs` — synthetic transforms, lazy vs eager.
- `tests/imatrix_cross_validate.rs` — `≤ 1e-4` against `llama-imatrix`.
- `tests/kquant_byte_identical_llama_cpp.rs` — Q4_K_M / Q5_K_M / Q6_K byte-identical.
- `tests/safetensors_mlx_lm_round_trip.rs` — hf2q safetensors output loads in `mlx_lm.load` (subprocess test, gated `#[ignore]` by default; runs in P10 harness).
- `tests/peer_parity_gates.rs` — full 8-cell gate matrix.

### Specification-driven tests
- Every k-quant variant has a hand-authored synthetic input with a deterministic expected output computed from the llama.cpp algorithm by hand (no comparison against running llama-quantize — the comparison is at the algorithm level, not the binary level). Runs in default cargo test set.

### Real-model tests
- `#[ignore]`-gated real-model gate tests in `tests/peer_parity_gates.rs` for each cell of Decision 15's matrix. Run only via `scripts/peer_parity_run.sh` on a real M5 Max with the reference models present. These produce P11's closing artefacts.

---

## Risks and mitigations

### R1: `LazyTensor` API churn during P1–P9 invalidates downstream consumers
**Probability:** medium. The trait will see usage we didn't anticipate.

**Mitigation:** lock the `LazyTensor` API at P0 close via `#[deny(...)]` doctest examples covering every public method. Any breaking change after P0 requires a Decision-23 amendment to this ADR (no silent API edits).

### R2: Streaming write atomicity broken under SIGINT
**Probability:** low. POSIX rename is atomic; the tempdir pattern is well-tested.

**Mitigation:** Decision 20's `test_cancellation_no_partial_at_final_path` runs SIGINT mid-stream 100× in CI; any leak fails the test.

### R3: Imatrix port has subtle math divergence from llama.cpp's C impl
**Probability:** medium. Float ordering and accumulation patterns differ between Rust and C.

**Mitigation:** Decision 10's `≤ 1e-4` cross-validation gate. If the port diverges by more: per mantra, fix the port — do not accept "comparable" until it's byte-precise.

### R4: DWQ peer-parity gate against mlx-lm requires identical calibration corpus
**Probability:** high. mlx_lm.convert uses its own corpus by default.

**Mitigation:** P10 harness calls `mlx_lm.convert` with `--calibration-data` pointing at hf2q's `tests/fixtures/ppl-corpus/wikitext2.tokens` (mlx-lm supports this). If a corpus mismatch is unavoidable, the gate becomes "PPL on the same eval corpus" not "PPL after the same calibration corpus" — documented in Decision 16.

### R5: Sensitivity cache invalidation false-negative
**Probability:** low. SHA-256 keying is robust.

**Mitigation:** key includes the *sensitivity algorithm version* string (manually bumped on any change to `sensitivity.rs`); cache miss on version mismatch.

### R6: Rayon contention on the serialiser thread
**Probability:** medium for very-fast quantizers (Q4 flat). Producer outpaces serialiser.

**Mitigation:** bounded channels in Decision 5 — back-pressure flows naturally. P3 measures and tunes channel capacity.

### R7: Calibrator trait too narrow for AWQ
**Probability:** medium. AWQ is per-channel quant-aware and may need access to the *quantizer* during calibration (not the trait we designed for).

**Mitigation:** P7 design review writes a mock AWQ calibrator implementing the trait. If AWQ doesn't fit cleanly, the trait grows in P7 — better than discovering it later. No AWQ implementation lands; the door stays shut.

### R8: Determinism gate fails on un-calibrated paths
**Probability:** low. The pipeline is mathematically equivalent; only ordering changed.

**Mitigation:** if it fails, the bug is real and gets fixed at the root (`BTreeMap` iteration order, write-buffer flushing, stable threading). Mantra: never lower the bar.

### R9: Real-model peer-parity gates fail on apex MoE
**Probability:** medium. Apex MoE is the largest, hardest case; imatrix may underperform DWQ on hybrid arch.

**Mitigation:** per mantra, fix until they pass. If imatrix on apex MoE genuinely loses to llama.cpp imatrix on the same model, the bug is in our port (Decision 10) and gets fixed. If DWQ on apex MoE genuinely loses to mlx-lm DWQ on the same model, the bug is in our port and gets fixed. There is no "ship with a degraded variant"; the variant is removed from the menu before that happens.

### R10: Safetensors backend integration breaks Gemma-4 safetensors output
**Probability:** medium. Gemma-4 safetensors is a working path today.

**Mitigation:** Decision 17's `test_safetensors_streaming_byte_identical_to_eager` for Gemma-4 `--quant f16 --format safetensors` is a hard gate. P9 closes only when this passes.

### R11: Cross-validation against `llama-imatrix` requires `llama.cpp` checked out and built
**Probability:** environment risk; not algorithmic.

**Mitigation:** P10 harness checks `llama-imatrix` is on `$PATH`; if not, the cross-validation test is `#[ignore]`-skipped with a clear log message. CI nodes used for ADR-014 close run with both `llama.cpp` and `mlx-lm` installed.

### R12: ADR-013's `RealActivationCapture` API constraints in `from_lazy_tensor_map`
**Probability:** low — ADR-013 is closed.

**Mitigation:** Decision 8 specifies the constructor doesn't change the trait. If `Qwen35Model::load_from_lazy_tensor_map` needs internal API changes in `weight_loader.rs`, those are local to the qwen35 module and don't propagate to ADR-013's trait surface.

### R13: P11 reference artefact production exhausts disk
**Probability:** medium. Apex MoE BF16 + 4 DWQ GGUFs + 4 safetensors variants + 4 llama.cpp Q4_K_M + 4 mlx-lm DWQ ≈ 200+ GB of artefacts.

**Mitigation:** P11 runs sequentially, deletes intermediate artefacts (`<peer>-Q4_K_M.gguf` after PPL is measured), keeps only the 8 final files. ADR-012's existing 150-GB disk preflight (Decision 14) covers the working set.

### R14: ADR-012 P9 final close (real-model artefact production) collides with ADR-014 P11
**Probability:** high — they target the same models on the same hardware.

**Mitigation (Robert lock 2026-04-25 round 2):** **ADR-014 P0 starts only after ADR-012's four DWQ GGUFs verifiably load in `llama-cli`** — not just after the worktree-merge commit, and not on the basis of the status-header claim alone. The current header line `feat(adr-005 phase 2c iter 100)` cohort says "engineering complete, 4 DWQ GGUFs delivered" (commit `38d2f3c`), but ADR-012 line 7's real-model audit says "every previously-shipped DWQ GGUF in `models/qwen3.6-{27b,35b-a3b-abliterix-ega-abliterated}-dwq{46,48}/` fails to load in `llama-cli`." The audit, not the status header, is the entry gate. P0 entry criterion: `llama-cli --model models/<each of the 4 DWQ GGUFs> -p "Hello" -n 16` succeeds on each artefact. P11 starts on `main` after that, sequenced not parallel on the M5 Max.

---

## Discovered defect — DWQ 4-bit base emits Q4_0 (legacy), not Q4_K_M (modern) [2026-04-27, ADR-015 cross-ref]

**Origin:** ADR-015 iter8c-prep bench-matrix sweep + tensor-type audit, 2026-04-27.  Localized at hf2q@`a7d01ef`.

### Defect

`src/backends/gguf.rs::quant_info_to_ggml_type` (lines 1666-1680) maps DWQ's 4-bit-base output to **`GGML_TYPE_Q4_0`** (legacy, 32-element blocks, single F16 scale per block):

```rust
// src/backends/gguf.rs:1666-1680
match info.bits {
    16 => GGML_TYPE_F16,
    8  => GGML_TYPE_Q8_0,
    6  => GGML_TYPE_Q6_K,        // ← 6-bit correctly uses K-quant
    4  => GGML_TYPE_Q4_0,        // ← 4-bit uses LEGACY Q4_0, NOT Q4_K
    2  => GGML_TYPE_Q4_0,
    _  => GGML_TYPE_F16,
}
```

The comment at lines 1635-1636 — *"K-quant types from Apex cannot be honored because hf2q does not produce K-quant super-block data; we map to the closest simple block type instead"* — is **stale**.  P7 iter-3a/b1/c (commits `ac3ebf2`, `ebee4e6`, `ade910c`, `6440b4e`, `1c37488`, `b27afa7`, `93415ad`) shipped the full Q4_K + Q5_K + Q6_K codebook ports including imatrix-weighted variants; `KQuantCodecQuantizer` produces real Q4_K_M super-block data today (144 bytes per super-block of 256 elements: F16 super-scale + F16 super-min + 12 bytes of 6-bit sub-scales/sub-mins for 8 sub-blocks).

### Evidence (ADR-015 bench matrix at hf2q@`a7d01ef`)

| Fixture | Heavy weight quants (counts) | hf2q decode vs llama.cpp same-day |
|---|---|---:|
| `qwen3.6-35b-a3b-...-dwq46.gguf` | **487 Q4_0** + 24 Q6_K | **0.9327× LOSS** (recovery 7.2%) |
| `qwen3.6-35b-a3b-...-dwq48.gguf` | 232 Q4_0 + 279 Q8_0 | 1.0324× WIN |
| `qwen3.6-35b-a3b-...-apex.gguf` (Q5_K_M) | 370 Q5_K + 60 Q6_K | 1.0356× WIN (5-trial validation) |
| `qwen3.6-27b-dwq46.gguf` (dense, BW-bound) | Q4_0 dominant | 0.9888× TIE |
| `qwen3.6-27b-dwq48.gguf` (dense, BW-bound) | 459 Q4_0 + 38 Q8_0 | 1.0173× WIN |
| `gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf` | **240 Q4_0** + 48 Q6_K | **0.8303× LOSS** (recovery 20.4%) |

**Pattern**: hf2q's `kernel_mul_mv_q4_0_f32` underperforms llama.cpp's same-named kernel on M5 Max despite byte-equivalent shader source (verified ADR-015 commit `6818884`, identical `N_R0_Q4_0=4`, identical two-accumulator sumy pattern).  9 confirmed M5 Max static-evidence kernel hypotheses for closing the Q4_0 gap have been falsified per ADR-012 §"Final closure verdict" + `project_metal_compiler_auto_optimizes_static_levers`.  Conversely, hf2q's K-quant kernels (`kernel_mul_mv_q4_K_f32`, `kernel_mul_mv_q5_K_f32`, `kernel_mul_mv_q6_K_f32`) are **faster than llama.cpp's** on M5 Max for the same workloads — apex Q5_K_M wins 1.04×, dwq48 (Q4_0+Q8_0 mix) wins 1.03×.

Switching DWQ's 4-bit base output from Q4_0 to Q4_K_M is therefore expected to:

1. **Close the ADR-015 D4 first-bullet gap on dwq46** (~+7%) and **second-bullet gap on gemma** (~+20%) by routing the dominant tensor type through the K-quant kernel family that hf2q wins on.
2. **Improve quantization quality** — Q4_K_M's grouped 6-bit sub-scales preserve more dynamic range than Q4_0's single F16 scale (canonical evidence: SqueezeLLM, Q4_K_M is industry standard for 4-bit since llama.cpp 2024-Q1).
3. **Match peer convention** — `mlx_lm.convert --quantize` produces Q4-equivalent grouped layouts; llama.cpp's recommended 4-bit is `q4_k_m`.  Q4_0 is largely deprecated for modern inference.

### Fix scope

The K-quant codec exists; the fix is wiring DWQ's emit path through it.  Concretely:

1. **`src/backends/gguf.rs::quant_info_to_ggml_type`**: change the `4 => GGML_TYPE_Q4_0` branch to `4 => GGML_TYPE_Q4_K`, gated on calibration availability.  Update the stale comment at lines 1635-1636 to reflect that K-quant production is supported.
2. **DWQ emit path** (`src/calibrate/dwq_calibrator.rs` + `src/calibrate/dwq_activation::run_dwq_with_sensitive_ranges`): when emitting a 4-bit DWQ tensor, route through `KQuantCodecQuantizer::new(KQuantTarget::Q4K, calibration_data)` (the same codec ADR-014 P7 wired into `imatrix-q4_k_m`).  The DWQ sensitivity vector becomes the per-column importance weights consumed by `quantize_row_q4_k_imatrix` (`src/quantize/k_quant.rs`).
3. **CLI variant naming** (Decision 12): `dwq46` becomes a logical name for "Q4_K_M base + Q6_K sensitive", `dwq48` becomes "Q4_K_M base + Q8_0 sensitive".  Old Q4_0-base GGUFs from the pre-fix pipeline remain readable but should be treated as deprecated; a future cron iter regenerates the four shipped artefacts.
4. **Re-emit gate** (P11): the four production GGUFs (`qwen3.6-35b-a3b-...-dwq{46,48}.gguf`, `qwen3.6-27b-dwq{46,48}.gguf`, plus `gemma-4-26B-A4B-...-dwq.gguf`) are regenerated through the post-fix pipeline; ADR-015 D4 same-day bench is re-run; both bullets gate on ≥ 1.00× ratio.
5. **Coherence gate** (ADR-014 P10/P11 cross-ref + ADR-015 task #20): post-fix Q4_K_M base must score ≥ pre-fix Q4_0 base on PPL on wikitext-2 (Decision 16 methodology).  Should improve, not just match — Q4_K_M is canonically better quality at 4-bit.

### Phase placement

This is **a P11 prerequisite, not a new phase**.  P11's "Re-emit ADR-012's four DWQ GGUFs + measured gate close" is what shipped the buggy Q4_0-base artefacts; landing this fix before P11 close means the re-emitted GGUFs incorporate the K-quant base from the start.  Estimated scope: ~50-200 LOC (wiring), no new algorithm work — the K-quant codebook ports landed in P7.

### Cross-ADR effects

- **ADR-015 D4 exit criteria**: both bullets become achievable without iter8c shader-level kernel work.  iter8c can close as "lever moved to ADR-014 P11 fix" pending the re-bench measurement.
- **ADR-012 P9 final close**: the four re-emitted DWQ GGUFs must still load in llama.cpp without error, must still reach the canonical-pangram smoke gate, and must satisfy the same byte-of-disk + PPL gates ADR-012 P11 anchored.  No regression expected since Q4_K_M is the format llama.cpp ships native support for.
- **`feedback_dispatch_count_not_wall_time`** memory: dispatch-count was a misleading signal.  The actual lever is the kernel-FAMILY hf2q runs; switching from Q4_0-family kernel to K-quant-family kernel changes which path executes, which closes the wall-time gap.

### Audit revision (2026-04-27, dual-mode CFA P11-prereq iter)

CFA Claude-team audit at base SHA `4c887f2` HARD-STOPPED the original "1-line fix" framing.  Two findings revise the work plan:

**Finding 1 — Original Scenario A (1-line type-code change) is byte-level wrong.**  Today's DWQ output is internally consistent: header reports Q4_0 (type 2) AND on-disk bytes are Q4_0 format (32-element blocks, 18 bytes/block).  Path: `cmd_convert` Dwq46/Dwq48/Dwq68/Dwq28 (`src/main.rs:1386-1447`) → `dwq_activation::run_dwq_with_sensitive_ranges` (`src/calibrate/dwq_activation.rs:186-198`) → `dwq::run_dwq_calibration_internal` (`src/calibrate/dwq.rs:267-288`) → `MixedBitQuantizer::new` (`src/quantize/mixed.rs:99-161`) → `StaticQuantizer::new("q4")` (`src/quantize/static_quant.rs:163-242`, scalar-per-group F16 scale + packed nibbles) → `QuantizedTensor { method:"q4", bits:4, scales:Some(..), ggml_type:None }` → `quant_info_to_ggml_type` returns `GGML_TYPE_Q4_0` via the bits-fallback at `gguf.rs:1670` → `repack_to_ggml_blocks::repack_q4_0` (`gguf.rs:1085-1098, 1111-1218`) writes genuine Q4_0 GGUF block bytes.  Changing only the type-code arm to `4 => GGML_TYPE_Q4_K` would emit Q4_K headers (type 12) atop Q4_0 bytes — llama.cpp loader rejects every shipped DWQ artefact, AND `repack_to_ggml_blocks` has no `GGML_TYPE_Q4_K` arm so `match` fails first.  The bench-matrix LOSS is M5 Max kernel-family wall-time (Q4_0 mat-vec underperforms; K-quant mat-vec leads), NOT a GGUF format defect.

**Finding 2 — Original Scenario B (~50–200 LOC) underestimates by ~3×; actual scope is ~580 LOC across 3 sequential iters.**  Three architectural blockers:
1. **Representation mismatch.** DWQ's closed-form scale-cal (`dwq.rs:298-356`, ~60 LOC) optimises a scalar-per-group F16 scale via `optimal_scale = dot(W,Q)/dot(Q,Q)` per group.  Q4_K has no per-group scalar — its 6-bit sub-scales + 6-bit sub-mins are derived inside the codebook search in `quantize_row_q4_k_to_bytes`.  Porting DWQ-style optimisation to Q4_K's super-block geometry is new algorithm work, not wiring.
2. **`KQuantCodecQuantizer` cannot inner-route a `MixedBitQuantizer`-style 4-vs-6 mix.** Target is constructor-fixed; ignores per-tensor `LayerQuantConfig.{bits,group_size}`.  dwq46/48/68 = "Q4_K + Q6_K|Q8_0|Q5_K" needs a new mixed-target router (mirror `VariantKQuantizer` per-tensor-policy dispatch, ~280 LOC).
3. **`repack_to_ggml_blocks` has no Q4_K branch** (`gguf.rs:1085-1098`).  Today's `KQuantCodecQuantizer` direct path "works" only because `(scales=None, method="k_quant_codec_direct")` falls into the warn-and-return-raw branch at `gguf.rs:1064-1075` — a code smell that fires on every codec-direct tensor write.

**Finding 3 — LIVE COLLATERAL DEFECT discovered (independent of DWQ).**  `KQuantCodecQuantizer` (`src/quantize/k_quant_codec_quantizer.rs:114,222`) and `VariantKQuantizer` (`src/quantize/variant_quantizer.rs:219-234`) set `bits=0, ggml_type=Some("Q4_K")`.  In `quant_info_to_ggml_type` (`gguf.rs:1645-1649`), `Q4_K` matches the K-quant fall-through (no return) → `bits=0` hits `_ =>` at `gguf.rs:1672-1677` → returns **`GGML_TYPE_F16`** (type 1).  So today's `--quant q4_k_m`, `--quant imatrix-q4_k_m`, `--quant imatrix-adaptive` paths emit GGUFs with header reporting **F16** but bytes being Q4_K — MALFORMED for llama.cpp.  Existing tests (`tests/convert_integration.rs:222-263`) assert `.gguf` exists, not header-vs-bytes consistency; the canonical regression surface is the ignored peer-parity gates which require real models.  This bug pre-dates the DWQ defect note and lives in the same code paths.

### Revised plan — 3 sequential dual-mode CFA iters

| Iter | Sub-tasks | Files (within file fence) | LOC est. |
|---|---|---|---:|
| **A** | Codec-direct fast-path in `quant_info_to_ggml_type`: recognise `method == METHOD_K_QUANT_CODEC_DIRECT` (or whatever sentinel `KQuantCodecQuantizer` + `VariantKQuantizer` set) → route via `ggml_type_from_name(info.ggml_type.as_ref()?)` BEFORE the bits-fallback fires; matching codec-direct fast-path in `repack_to_ggml_blocks` (no repack — already in target block format).  Always-on regression tests asserting `bits=0, ggml_type=Some("Q4_K")` returns `GGML_TYPE_Q4_K`=12 (NOT `GGML_TYPE_F16`).  **Closes the LIVE `--quant q4_k_m` / `imatrix-q4_k_m` / `imatrix-adaptive` malformed-GGUF bug — independent of DWQ rewrite, unblocks today's K-quant emit paths from shipping malformed.** | `src/backends/gguf.rs`, `tests/codec_direct_type_code.rs` (NEW) | ~80 |

**Iter A landed (commit `4349fb2`, 2026-04-27)**: codec-direct fast-path in `quant_info_to_ggml_type` recognises sentinel `"k_quant_codec_direct"` (defined as `pub const METHOD_K_QUANT_CODEC_DIRECT` at `src/quantize/k_quant_codec_quantizer.rs:114`) set by both `KQuantCodecQuantizer::quantize_tensor` (`k_quant_codec_quantizer.rs:222`) AND `VariantKQuantizer::quantize_tensor` (`variant_quantizer.rs:225`); routes via `ggml_type_from_name` BEFORE the bits-fallback fires.  Matching `repack_to_ggml_blocks` fast-path returns codec-direct bytes unchanged with block-size validation against `BLOCK_Q4_K_SIZE` / `BLOCK_Q5_K_SIZE` / `BLOCK_Q6_K_BYTES` / `BLOCK_Q3_K_SIZE` / `BLOCK_Q4_0_BYTES` / `BLOCK_Q8_0_BYTES`; misaligned bytes or unsupported target ggml_types now surface as typed `BackendError::WriteFailed` rather than silent corruption.  **N=0 in-source unit tests in `gguf.rs:3266-3413` flipped** — the audit summary's "tests CURRENTLY ASSERT THE BUG" framing was overcautious: `test_dtype_mapping` (line 3267) and `test_ggml_type_override_in_quant_info` (line 3377) both reference orthogonal paths (`method = "t"` and `method = "apex"`, neither of which is the codec-direct sentinel), so they continued to assert the legacy bits-fallback / Apex K-quant override paths correctly without any change.  A clarifying comment was added to `test_ggml_type_override_in_quant_info` documenting where the codec-direct path is now locked.  **K=11 always-on in-source regression tests** added in `gguf.rs::tests` (`codec_direct_q4_k_returns_q4_k_type_code`, `codec_direct_q5_k_returns_q5_k_type_code`, `codec_direct_q6_k_returns_q6_k_type_code`, `codec_direct_q4_0_legacy_returns_q4_0_type_code`, `codec_direct_q8_0_legacy_returns_q8_0_type_code`, `codec_direct_unknown_ggml_name_returns_f16_with_warn`, `codec_direct_repack_q4_k_returns_raw_bytes_unchanged`, `codec_direct_repack_q5_k_returns_raw_bytes_unchanged`, `codec_direct_repack_q6_k_returns_raw_bytes_unchanged`, `codec_direct_repack_rejects_misaligned_bytes`, `codec_direct_repack_unsupported_target_type_errors`); the in-source position is forced by the consumer-side helpers (`quant_info_to_ggml_type`, `repack_to_ggml_blocks`) being module-private and gguf.rs's transitive deps making `#[path]`-include into a `tests/*.rs` integration crate impractical.  **+4 always-on integration tests** in `tests/codec_direct_type_code.rs` (NEW, 253 LOC) drive the full binary end-to-end with `--quant q4_k_m` / `--quant q5_k_m` / `--quant q6_k`, then read back the GGUF header via `mlx_native::gguf::GgufFile` to assert the on-disk type code is the canonical K-quant code — closing the loop on byte-on-disk-vs-header consistency that is the actual user-facing surface of the bug.  Plus 1 across-layer regression covering `blk.0.attn_q.weight` AND `blk.1.attn_q.weight` to lock against a sentinel-recogniser regression that only fires for `blk.0`.  **Closes the LIVE `--quant q4_k_m` / `imatrix-q4_k_m` / `imatrix-adaptive` malformed-GGUF bug.** Iters B + C still pending for the DWQ Q4_0 → Q4_K_M rewrite.

**Spec deviation (queen)**: spec hinted the new always-on tests would live in `tests/codec_direct_type_code.rs` and `<N>` in-source tests would need flipping.  Actuals (with rationale): N=0 in-source flips because the audit-cited tests reference `method = "t"` / `method = "apex"`, NOT the codec-direct sentinel — they were correctly asserting orthogonal paths the whole time (commenting on the affected test instead).  The 11 in-source regression tests live INSIDE `gguf.rs::tests` (not `tests/codec_direct_type_code.rs`) because the consumer-side helpers are module-private; `tests/codec_direct_type_code.rs` covers the ACTUAL user-facing surface (header-vs-bytes consistency on disk via the binary's full convert dispatch).

**Iter B landed (commit `ec3d3b7`, 2026-04-27)**: NEW `src/quantize/dwq_k_quantizer.rs` (321 production LOC, 136 actual code) with `DwqKQuantizer { variant: DwqKVariant, sensitive_indices: HashSet<usize>, calibration: CalibrationData }`.  `DwqKVariant::{P46, P48, P68, P28}` map to (base, sensitive) targets `(Q4_K, Q6_K)` / `(Q4_K, Q8_0)` / `(Q6_K, Q8_0)` / `(Q2_K [deferred], Q8_0)` respectively.  Per-tensor dispatch via `target_for(tensor_name)`: `extract_layer_index(tensor_name)` parses the canonical `model.layers.<N>.…` (and `model.language_model.layers.<N>.…`) HF naming pattern, returns `Some(N)` for layer tensors / `None` for non-layer tensors (output, embeddings, norms); when `N` is in `sensitive_indices` → sensitive target, else base target; non-layer tensors fall through to base — mirroring today's `MixedBitQuantizer` shipping behaviour exactly (Chesterton's fence — we replace the codec, not the policy).  `Quantizer::quantize_tensor` constructs `KQuantCodecQuantizer::new(format!("{variant_name}-{target_short}"), target, calibration.clone())` per-call and delegates — Iter A's codec-direct fast-path then routes the resulting `METHOD_K_QUANT_CODEC_DIRECT` sentinel + correct `ggml_type` through to GGUF emit.  Drops legacy DWQ scalar-scale-cal step (Q4_K has no scalar-per-group scale to optimise — sub-block 6-bit scales/mins are derived inside the codebook search at `make_qkx2_quants`; a post-hoc closed-form re-fit would degrade the joint-optimum).  **Sensitivity → Imatrix wiring: DEFERRED to Iter C with reason** — `DwqConfig` exposes only a *scalar-per-layer* sensitivity (`LayerSensitivity.score`) used upstream to derive `sensitive_layers: Vec<RangeInclusive<usize>>`, but `CalibrationData::Imatrix` requires a *per-column* importance vector (`Vec<f32>` of length `row_len`); these two shapes are dimensionally incompatible and bridging requires either (a) a separate imatrix calibration step run alongside DWQ that produces real per-column vectors, or (b) an upstream broadcast from scalar→column-uniform vector (degenerates to `_ref` since uniform weights yield the same codebook search as no weights) — both are call-site decisions belonging to Iter C; the constructor accepts `Option<CalibrationData>` so Iter C can wire (a) without churning the API.  **Q2_K handling: typed-error fallback with deferred plan** — `KQuantTarget` does not yet expose a Q2_K variant (codec port pending a separate iter); `DwqKVariant::P28::base_target()` returns `None`; `Quantizer::quantize_tensor` surfaces `QuantizeError::TensorQuantizeFailed` with a typed message pointing at the deferred codec land for base-bucket tensors (no panics, no silent fallback) — sensitive-bucket tensors under P28 still quantize correctly to Q8_0 (legacy, codec-supported), so the restriction is narrowly scoped to layers NOT in `sensitive_layers`.  **25 always-on in-source tests** in `src/quantize/dwq_k_quantizer.rs::tests` (12 unit-level on variant targets / layer-index parsing / sensitive set expansion / inverted-range tolerance / target_for behaviour, 13 end-to-end through `Quantizer::quantize_tensor` covering per-variant base+sensitive routing, byte-format match against `BLOCK_Q4_K_SIZE`/`KQuantTarget::*.bytes_per_block()`, P28 typed-error narrow-scoping, preserve passthrough, non-layer tensor base-fall-through, language_model.-prefix layer parsing, 2D row iteration).  **+6 always-on integration smoke tests** in `tests/dwq_k_quantizer.rs` (NEW, 234 LOC) verify binary builds + module is registered + `convert --help` runs + top-level `--help` lists `convert` + filesystem-level pin on the public-API names so a downstream rename trips the gate.  Algorithm-side coverage lives in-source per the same Iter A precedent at `gguf.rs::tests` — `hf2q` is binary-only (no `[lib]` target), `dwq_k_quantizer.rs`'s transitive dep tree (`k_quant_codec_quantizer` → `k_quant_codec` → `k_quant` + `q_legacy` + `calibrate::calibrator` + `ir::*`) is too deep to mirror without a fragile shadow tree, and the new dispatch surface isn't yet wired into the CLI for `assert_cmd`-driven coverage (Iter C lands the wiring).

**Verification** (RUSTC_WRAPPER unset; sccache blocked in this sandbox): cargo build `0 errors`; `cargo test --release --test dwq_k_quantizer` 6 passed; `cargo test --release --test codec_direct_type_code` 4 passed (Iter A baseline preserved); `cargo test --release --test ppl_driver` 26 passed (baseline preserved); `cargo test --release --test peer_parity_gates -- --test-threads=1` 47 passed + 8 ignored (baseline preserved — parallel-thread mode flakes 3 missing-binary tests on first run, identical flake on stashed `4511dbe` baseline; serial run is clean); `cargo test --release --test imatrix_xvalidation` 36 passed + 1 ignored (baseline preserved); `cargo test --bin hf2q --release` **2094 passed + 11 ignored** (+25 from baseline 2069, matching the 25 new in-source tests in `dwq_k_quantizer::tests`); `cargo clippy --release -p hf2q -- -D clippy::correctness` 0 errors / **393 warnings** (+6 from baseline 387 — all `dead_code` for the new symbols `DwqKVariant`, `DwqKQuantizer::{new,variant,is_sensitive_tensor,target_for}`, `kquant_target_short_name`, `extract_layer_index`; expected since Iter C wires the API; well within ≤ 395 budget).  **LOC**: src/quantize/dwq_k_quantizer.rs +752 (321 production, 431 in-source tests), src/quantize/mod.rs +1 (one `pub mod` line), tests/dwq_k_quantizer.rs +234 (NEW integration smoke crate).  Production code change ~136 actual code LOC + 185 doc/blank = 321 production lines, well under the 250 LOC HARD-STOP (excluding the integration-test crate).

Iter C still pending: wire `DwqKQuantizer` into `cmd_convert` Dwq46/48/68/28 dispatch (replace `MixedBitQuantizer`); remove the K-quant fall-through swallow at `gguf.rs:1645-1659` (Iter A's codec-direct fast-path subsumes it cleanly); ADR-014 P11 closure section + ADR-015 D4 exit annotation.  P11 re-emit of the four DWQ GGUFs becomes the AC for Iter C.

**Verification** (RUSTC_WRAPPER unset; sccache blocked in this sandbox): cargo build `0 errors`; `cargo test --release --test codec_direct_type_code` 4 passed (Q4_K + Q5_K + Q6_K + variant-across-layers); `cargo test --release --bin hf2q -- backends::gguf::tests::test_ggml_type_override_in_quant_info backends::gguf::tests::test_dtype_mapping` 2 passed (named existing tests survive Iter A unchanged); `cargo test --release --test ppl_driver` 26 passed (baseline preserved); `cargo test --release --test peer_parity_gates` 47 passed + 8 ignored (baseline preserved); `cargo test --release --test imatrix_xvalidation` 36 passed + 1 ignored (baseline preserved); `cargo test --bin hf2q --release` **2069 passed + 11 ignored** (+11 from baseline 2058: 11 new codec-direct in-source tests); `cargo clippy --release -p hf2q -- -D clippy::correctness` 0 errors / **387 warnings** (down 1 from baseline 388 — wired-up forward-API symbol; well within ≤ 395 budget).  **LOC**: src/backends/gguf.rs +255/-2 (+82 production code = ~52 fast-path bodies + 30 doc/comment, +173 in-source tests); tests/codec_direct_type_code.rs +253 (NEW). Production-code change ~82 LOC, well under the 250 LOC HARD-STOP.
| **B** | New `DwqKQuantizer` (mirror `VariantKQuantizer` per-tensor policy dispatch, route per-tensor base-vs-sensitive via `KQuantCodecQuantizer` with `KQuantTarget` ∈ {Q4_K, Q6_K, Q8_0, Q5_K}); drop legacy DWQ scale-cal step for K-quant emit paths (geometric incompatibility — codebook search subsumes scalar-scale optimisation); optionally pipe DWQ sensitivity vector into `CalibrationData::Imatrix` so `quantize_row_q4_k_imatrix_to_bytes` actually fires on the high-sensitivity columns.  Algorithm work, not just wiring. | `src/quantize/dwq_k_quantizer.rs` (NEW), `src/calibrate/dwq.rs`, `src/calibrate/dwq_activation.rs` | ~280 |
| **C** | Wire `DwqKQuantizer` into `cmd_convert` Dwq46/48/68/28 dispatch (replace `MixedBitQuantizer`); remove the K-quant fall-through swallow at `gguf.rs:1645-1659` (Iter A's codec-direct fast-path subsumes it cleanly); ADR-014 P11 closure section + ADR-015 D4 exit annotation + Decision 12 menu naming touch-up.  P11 re-emit of the four DWQ GGUFs becomes the AC for this iter — re-bench matrix vs llama.cpp validates the Q4_K-family kernel switch closes the dwq46/gemma wall-time gap. | `src/main.rs`, `src/backends/gguf.rs`, `docs/ADR-014-*.md`, `docs/ADR-015-*.md` | ~220 |

**Total**: ~580 LOC + ~250 test LOC.  Each iter ≤ 250 LOC HARD-STOP (CFA-friendly).  Iter A is INDEPENDENT of B and can land first; Iter B is the algorithm work; Iter C is the final wiring + AC close.

## Open questions

These are tracked here, not deferred. Each gets resolved during the Phase that touches it; no question survives this ADR's close.

1. **(P1) Does `convert_bf16_to_f16` (`src/main.rs:419`) lift to lazy cleanly, or does it need a special-case eager path?** Resolution due in P1.
2. **(P5) Cache eviction policy for `~/.cache/hf2q/sensitivity/`.** LRU? Manual purge? Resolution due in P5.
3. **(P6) Does ADR-013's CPU forward at `src/inference/models/gemma4/forward_cpu.rs` produce activations precise enough for imatrix cross-validation `≤ 1e-4`, or do we need a higher-precision F32 forward for calibration?** Resolution due in P6.
4. **(P7) Does the `Calibrator` trait need a `prepare(&mut self, model: &LazyTensorMap)` lifecycle method separate from `calibrate(...)`, to allow streaming calibration alongside streaming quant?** Resolution due in P7.
5. **(P10) Does `mlx_lm.convert --calibration-data <file>` accept hf2q's wikitext2 token file format directly, or do we need to re-emit as plain UTF-8 text?** Resolution due in P10.
6. **(P10 iter-2) PPL columns deferred from P10 iter-1.** Iter-1's `emit_markdown_table` ships speed + RSS columns only; the PPL columns (hf2q_ppl, llama_ppl, mlx_ppl, hf2q_kl_vs_f16, llama_kl_vs_f16, mlx_kl_vs_f16) require (a) building `tests/fixtures/ppl-corpus/wikitext2.tokens` (~512 tokens smoke) and `wikitext2-full.tokens` (~280 k tokens BPE-encoded, ~700 KB) per Decision 16, and (b) implementing an end-to-end load-GGUF/safetensors → forward-pass → `ppl_kl_eval` driver (`src/quality/perplexity.rs` has `compute_perplexity` over pre-computed logits but no driver that loads a quantized model and produces them). Iter-2 lands both when P6 closes — at which point the PPL columns slot into `emit_markdown_table` and the 8 `#[ignore]`-gated cells start populating `ppl_hf2q` + `ppl_peer` end-to-end.

---

## SOTA scan (2026-04-27, goalie research log)

Mid-iter-3o, goalie was asked whether ADR-014's imatrix/DWQ/k-quant ports cover techniques published after the implementation's training-corpus cutoff. Findings (full log at `memory/project_adr014_sota_research_2026_04_27.md`):

- **llama.cpp imatrix file format migrated `.dat` → GGUF on 2025-07-19** (commit `90083283`, PR #9400). New metadata fields: `imatrix.{chunk_count,chunk_size,datasets}`. 3D tensor support for MoE expert-routing imatrix calibration. Backward-compat preserved via `--output-format dat`. **Implication:** P6 iter-3 GGUF imatrix I/O is no longer a TBD schema — it's a concrete documented format, becomes implementable. Our pure-Rust legacy `.dat` writer continues to be compatible-by-default; modern llama.cpp output reading needs the GGUF parser.
- **MLX DWQ is gradient distillation, not importance reweighting.** Dual loss (KL on logits + MAE on hidden states) vs FP16 teacher; Adam + gradient accumulation on scales/biases; off-policy calibration corpus (tulu-3-sft-mixture) outperforms on-policy. Our existing `DwqQuantizer` is sensitivity-based bit allocation — closer to imatrix in spirit. Closing the gap would mean a new `DwqDistillationCalibrator` impl (substantial; needs ADR-013 ActivationCapture forward-pass infra). Decision: documented as a future direction, not closure-gating.
- **Rotation-based pre-quant transforms** (QuaRot arxiv 2404.00456, ButterflyQuant arxiv 2509.09679 Sep-2025, AQLM 2025-11-30, QTIP arxiv 2406.11235, QuIP arxiv 2307.13304). All target sub-3-bit and require activation captures or Hessian estimates. Out of ADR-014 scope (we ship Q4_K_M / Q5_K_M / Q6_K). Decision: noted here, not in scope.
- **At 4-bit the format wars are overstated.** Production blogs Jan-Mar 2026 report well-calibrated GPTQ/AWQ/GGUF Q4_K_M differ by 1–3% on standard benchmarks. Our existing imatrix-q4_k_m path is competitive with the SOTA. Effort is better spent on closing P11 measured gates than chasing new codebooks.

---

## Dependencies on other work (cross-ADR)

- **ADR-005 (inference server):** unchanged. ADR-014 produces artefacts that ADR-005's serve loads; the load path is downstream.
- **ADR-007 / ADR-009 (TQ KV state, hybrid memory backend):** unchanged. ADR-014 is convert-side only.
- **ADR-012 (qwen35moe conversion):**
  - Closes first on the current pipeline (P8 + P9 real-model artefact production).
  - ADR-014 P11 re-emits ADR-012's four DWQ GGUFs under the streaming pipeline as part of its own AC.
  - ADR-012's `tests/quality_thresholds.rs`, `tests/calibration_eval_disjoint.rs`, `tests/convert_qwen35_real_activation_capture.rs` are inherited unchanged — they remain valid against the streaming pipeline by Decision 17 (re-blessed for calibrated paths but the threshold *constants* are unchanged).
  - ADR-012's `IntermediateMoeQ8Quantizer` band-aid (P9b) is **deleted** by ADR-014 P4.
- **ADR-013 (qwen35 inference):**
  - `RealActivationCapture` trait unchanged.
  - New constructor `RealActivationCapture::from_lazy_tensor_map` is added; named cross-ADR API delta (Decision 8).
  - `Qwen35Model::load_from_lazy_tensor_map` is added in `src/inference/models/qwen35/weight_loader.rs`.

---

## Glossary

- **Lazy tensor:** A tensor whose bytes have not been materialised; carries shape/dtype metadata and a `FnOnce` closure that will produce the bytes on demand.
- **Materialise:** Invoke the closure to produce the tensor bytes; consumes the `LazyTensor`.
- **Streaming:** Per-tensor pipeline where bytes flow `read → transform → quantise → write → drop` without buffering the whole model.
- **Calibrator:** A pure-Rust component that consumes a model + corpus and produces calibration data (per-column weights, per-tensor sensitivity, etc.) used by the quantiser.
- **Output format:** The on-disk codebook (Flat, BitPair, KQuant, KQuantAdaptive) — orthogonal to the calibrator.
- **Peer:** llama.cpp (`convert_hf_to_gguf.py` + `llama-quantize --imatrix`) and mlx-lm (`mlx_lm.convert --quant-method dwq`).
- **Peer parity:** Measured wall-clock, peak RSS, and PPL gates against a peer's output on the same model + hardware. Locked in Decision 15.
- **DWQ:** Distilled Weight Quantization (Apple/MLX research). hf2q port at `src/quantize/dwq.rs`.
- **Imatrix:** Importance matrix calibration (llama.cpp). hf2q port lands in P6 at `src/calibrate/imatrix.rs`.
- **K-quant:** llama.cpp's super-block-grouped quant codebook (Q4_K_M, Q5_K_M, Q6_K).
- **Sovereignty (toolchain):** Pure-Rust impl, no Python runtime, no link to libggml. Locked in Decision 21.
- **Sovereignty (algorithm) [does not exist]:** hf2q does not claim sovereignty over DWQ or imatrix algorithms — both are published peer techniques ported into hf2q.
- **Off-diagonal cell:** A `(Calibrator, OutputFormat)` pair that is not exposed as a named CLI variant (e.g. DWQ + KQuant). Reachable only via `HF2Q_UNSAFE_EXPERIMENTS=1`.

---

## Appendix A: Target convert commands (once all phases land)

### Qwen3.6-27B dense — llama.cpp peer column (GGUF)
```bash
hf2q convert Qwen/Qwen3.6-27B --quant imatrix-q4_k_m
# emits: Qwen3.6-27B-imatrix-q4_k_m.gguf
# peer:  llama-quantize --imatrix wikitext2.imatrix model-f16.gguf model-q4_k_m.gguf q4_k_m
```

### Qwen3.6-27B dense — mlx-lm peer column (safetensors)
```bash
hf2q convert Qwen/Qwen3.6-27B --quant dwq-4-6 --format safetensors
# emits: Qwen3.6-27B-dwq-4-6/  (mlx-lm-loadable directory)
# peer:  mlx_lm.convert --hf-path Qwen/Qwen3.6-27B --mlx-path . --quantize --quant-method dwq --bits 4
```

### Qwen3.6-35B-A3B apex MoE — both peer columns
```bash
hf2q convert jenerallee78/Qwen3.6-35B-A3B-abliterix-ega-abliterated-apex --quant imatrix-q4_k_m
hf2q convert jenerallee78/Qwen3.6-35B-A3B-abliterix-ega-abliterated-apex --quant dwq-4-6
hf2q convert jenerallee78/Qwen3.6-35B-A3B-abliterix-ega-abliterated-apex --quant dwq-4-6 --format safetensors
```

### Auto routing
```bash
hf2q convert Qwen/Qwen3.6-27B --quant auto
# resolves: imatrix-q4_k_m (dense, ≤30B, any RAM)

hf2q convert jenerallee78/Qwen3.6-35B-A3B-...-apex --quant auto
# resolves: dwq-4-6 (MoE, < 96 GB RAM) or dwq-4-8 (MoE, ≥ 96 GB RAM)
```

### Dev gate (off-diagonal cells)
```bash
HF2Q_UNSAFE_EXPERIMENTS=1 hf2q convert Qwen/Qwen3.6-27B \
    --calibration imatrix --output-format bit-pair-4-6
# emits: Qwen3.6-27B-experimental-imatrix-bp-4-6.gguf
# warning: experimental, not peer-validated
```

---

## Appendix B: Canonical gotcha cross-reference (will accumulate during P0–P11)

(Empty at proposal time; populated as phases close. Each entry is one bullet citing the file:line of the gotcha and the Decision number that first addressed it. Mirrors ADR-012's Appendix B pattern.)

---

**This ADR closes when:**
1. All 13 phases (P0–P12) are green.
2. Decision 15's 8-cell peer-parity table is filled in `docs/peer-parity-results-<date>.md` with measured numbers, every gate passing.
3. The four ADR-012 reference DWQ GGUFs are re-emitted under the streaming pipeline and live in `models/`.
4. `cargo test` is green; `cargo clippy` is zero new warnings vs. ADR-014 baseline; `tests/sovereignty_*.rs` confirms no Python/torch/libggml link.
5. The closing commit message inlines the verdict table and links to `docs/peer-parity-results-<date>.md`.

Per mantra, no other close condition exists. No "shipped except for cell X." No "deferred apex MoE imatrix-q4_k_m row." No "good enough." Just measured peer parity, end-to-end, on the day this ADR closes.
