# ADR-033 P-1 Audit Synthesis

Consolidated output of 7 parallel audit agents covering the 5 quantize/* files + main.rs dispatcher arms + backends/gguf.rs writer slice + the 5 ADR-delete-listed support files. Each per-file audit lives at `docs/adr-033-audit/<name>.md`.

Pins: hf2q HEAD `85bee70e`; llama.cpp HEAD `c779f619802c310798ca8c89695cec7dcfe38a99` (at `data/llama_cpp_pin.txt`); mudler/apex-quant `63c5048b7dc9ff230f2397d7bc445ca28894b769` (at `vendor/apex-quant`).

## Per-file disposition totals

| File | LOC | DELETE LOC | MODIFY LOC | KEEP LOC | Audit file |
|---|---|---|---|---|---|
| `src/quantize/mod.rs` | 6440 | ~6432 | 8 (trait reshape) | 0 | [quantize-mod.md](quantize-mod.md) |
| `src/quantize/k_quant.rs` | 5541 | 2474 (test mod) | 3067 (5 K-quant files + common helpers) | 0 | [k-quant.md](k-quant.md) |
| `src/quantize/k_quant_codec.rs` | 1452 | 1452 | 0 | 0 | [k-quant-codec.md](k-quant-codec.md) |
| `src/quantize/q_legacy.rs` | 2130 | 0 (file is mv+split+cfg-rehome) | ~1801 (6 legacy-quant files) | ~157 (dequant utils + QLegacyError) | [q-legacy.md](q-legacy.md) |
| `src/quantize/layer_mix.rs` | 1304 | ~1107 | ~190 (standard_policy.rs) | 8 (vision.rs) | [layer-mix.md](layer-mix.md) |
| **5-file subtotal** | **16,867** | **~11,465** | **~5,066** | **~165** | — |
| `src/backends/gguf.rs:282-1259` (writer slice) | ~977 | ~480 (9 regions; 4 zero-pad sites; size predictor; inline F16) | ~295 (8 regions; new seek-back writer) | ~286 (KV-pair encoding; tensor-name canonical) | [gguf-writer.md](gguf-writer.md) |
| **+ 5 ADR-delete-listed files** | 3,428 | 3,428 | 0 | 0 | [delete-listed.md](delete-listed.md) |
| `src/main.rs` dispatch arms (L1043-3453) | ~3445 | ~1473 (17 regions; 5 dispatch arms + 3 DWQ subcmds + 11 stale CLI variants) | ~395 (6 regions; cli::QuantMethod reshape + cmd_convert single-arm collapse) | ~1577 (11 regions; CLI bootstrap, serve, unrelated subcommands) | [main-dispatch.md](main-dispatch.md) |

**Confirmed delete-able 5-file subtotal: ~11,465 LOC** (out of 16,867 in those five files). The remaining ~5,231 LOC ports to the new shape (5,066 MODIFY + 165 KEEP).

**Grand totals across all P-1 audit scope (5 files + gguf writer slice + main dispatch + 5 ADR delete-listed files):**
- DELETE: ~16,846 LOC
- MODIFY: ~5,756 LOC (kernel ports + policy port + writer rewrite + CLI reshape)
- KEEP: ~2,028 LOC (CLI bootstrap, quality-test utils, dequant round-trip helpers, KV-pair encoding)

## Findings — ADR amendments needed

The audit surfaced 8 issues that require explicit changes to ADR-033 before P0 begins. These are not optional. Each is sourced from a per-file audit's verified analysis.

### A. P0 v1 ggml_quants set is 11 files, not 9. (k-quant.md flag 1)

ADR-033 §P0 §"What" lists `{q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q6_k, q8_0, iq4_nl}` (9 files). But Q3_K + Q2_K dequant is externally referenced by `src/quality/mod.rs:612` (codec-direct quality metrics) and `src/backends/gguf.rs:{1275, 1458, 2207, 2566, 2819, 3085}` (size estimator + writer dispatch). Dropping Q2_K/Q3_K would either break those call sites or require an immediate-follow-up rewrite. **Amendment:** §P0 v1 set becomes `{q2_k, q3_k, q4_0, q4_1, q5_0, q5_1, q4_k, q5_k, q6_k, q8_0, iq4_nl}` (11 files). LlamaFtype set adds `MostlyQ2_K = 10`, `MostlyQ3_K_S/M/L = 11/12/13` (the StandardPolicy CLI surface unchanged — these were already named in §6).

### B. `backends/gguf.rs` has TWO two-pass writers, not one. (gguf-writer.md core finding)

ADR-033 §P2 mentions "the two-pass writer at `backends/gguf.rs:282-1259`". The audit found TWO complete two-pass writers in that slice: `Backend::write` (L282-738) for the text GGUF and `write_mmproj_gguf` (L887-1189) for the mmproj GGUF. Deleting only the text writer leaves the bug-class half-alive in mmproj. **Amendment:** §P2 §"What" explicitly replaces BOTH writers; the new `src/backends/gguf/writer.rs` exposes a single seek-back implementation parametric on (text | mmproj) via the metadata builder, not two separate writers.

### C. 4 zero-pad fallback sites are the iter-99 bug-class targets. (gguf-writer.md)

Sites: `backends/gguf.rs:639-641, 659-661, 677-679, 1132-1134`. All are `if current_pos < target_pos { write zeros }`. These cause silent file inflation when the pass-1 size predictor over-predicts a tensor's bytes. **Amendment:** §P2 §"Acceptance criteria" adds: "no zero-pad write site exists in the new writer; this is enforced by `grep -n 'write.*zero\|fill.*zero' src/backends/gguf/writer.rs` returning no matches".

### D. 2 inline F16 fallback sites violate the no-fallback rule. (gguf-writer.md)

Sites: `backends/gguf.rs:496-502` (K-quant row-misalignment → F16), `:511-521` (block-32 misalignment → F16). These bypass `QuantPolicy::target_for`'s no-fallback contract by demoting silently in the writer. **Amendment:** §P2 §"What" explicitly states "any F16 demotion logic is moved to `QuantPolicy::target_for` as a typed `QuantizeError`; the writer never demotes". Pairs with §"shape_fallback contract" already in the ADR.

### E. 3 inline vision-pattern checks need consolidation + `is_audio_tensor_pattern`. (gguf-writer.md)

Sites: `backends/gguf.rs:322-333, 721-724, 905-909`. They duplicate `layer_mix.rs::is_vision_tensor_pattern` but with one EXTRA pattern not captured by the canonical fn: `audio_tower` substring. The audio-tensor-pattern fn doesn't exist today. **Amendment:** §"Vision tensor pattern" expands to add a sibling `is_audio_tensor_pattern(name: &str)` covering `audio_tower.` / `audio_model.` / `whisper.` (the model patterns we'd expect at this layer). Section title becomes "Vision/audio tensor patterns".

### F. q_legacy needs imatrix-aware variants ADDED in P0 (not ported). (q-legacy.md flag 7)

Current `src/quantize/q_legacy.rs` has zero `*_impl` (imatrix-aware) variants. llama.cpp's `quantize_row_q4_0_impl` (ggml-quants.c:2008) accepts a `quant_weights` arg and dispatches on null. ADR's `Quantizer::quantize(src, n_per_row, imatrix: Option<&[f32]>)` requires imatrix-aware variants for every legacy type. **Amendment:** §P0 §"What" notes that for `{q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl}` the P0 port is BOTH the no-imatrix path (port of `quantize_row_<T>_ref`) AND the imatrix path (port of `quantize_row_<T>_impl`). The K-family files already had `_imatrix` variants in hf2q's `k_quant.rs`, so this asymmetry only affects the 6 legacy files.

### G. `layer_mix.rs::target_for` is incomplete vs llama.cpp's `llama_tensor_get_type_impl`. (layer-mix.md flag)

The current `target_for` explicitly defers Falcon, MoE-8x, GQA-70B, and IQ-family branches per its module doc (L39-43). ADR-033's P1 byte-cmp gate against `(convert_hf_to_gguf.py | llama-quantize)` will FAIL on any matrix arch hitting those branches. **Amendment:** §P1 §"What" explicitly states that `StandardPolicy::target_for` is a COMPLETE port of `llama_tensor_get_type_impl` at the pinned SHA — the deferred branches are NOT optional. §P1 acceptance gate explicitly tests each arch the convert matrix names (gemma4-26B-A4B, qwen35moe-3.6, qwen3vl_text, gemma4-mmproj, bert/bge-large, nomic_bert, llama3-8B, minimax-m27).

### H. `src/calibrate/apex.rs` is an ADR orphan. (delete-listed.md note 2)

The file is an existing hf2q apex calibrator, not on the ADR §P6 delete list. The new `ApexPolicy` doesn't need a separate calibrator. **Amendment:** §P6 §"What" adds `src/calibrate/apex.rs` to the delete list (rationale: superseded by ApexPolicy + Pi imatrix subsystem). Open question for operator: is there functionality in this file that should survive in some form? If not, deletion is clean.

### I. NEON-order caveat for `make_qkx2_quants`. (k-quant.md flag 3)

The current `src/quantize/k_quant.rs` has a module doc (L9-18) noting NEON-vs-scalar argument-order divergence in `make_qkx2_quants`. P0's C harness for generating reference fixtures MUST be built `aarch64-apple-darwin` with NEON path enabled (or we must verify hf2q's port matches scalar reference). **Amendment:** §P0 §"Acceptance criteria" adds: "the C harness used to generate `tests/fixtures/ggml_quants/<type>_<n>.bin` is built `aarch64-apple-darwin` (NEON enabled) on macOS Apple Silicon; for portability, the same harness rebuilt `x86_64-pc-linux-gnu` (no NEON) on x86 Linux must produce byte-identical fixtures — if it doesn't, hf2q ports are matched against the NEON variant explicitly".

### J. `QLegacyError → KQuantCodecError::QLegacy` `#[from]` edge. (q-legacy.md flag 2)

`src/quantize/k_quant_codec.rs:212` uses `#[from]` to wrap QLegacyError. The new `QuantizeError` taxonomy in ADR-033 §"Quantizer trait" must absorb the three QLegacyError variants (`NotBlockAligned`, `OutputTooSmall`, `BufferTooSmall`). **No ADR amendment needed** — already implied by §"Quantizer trait" but worth flagging to P1 implementer.

### K. `cli::QuantMethod` enum is out of sync with Decision §6. (main-dispatch.md)

Current `src/main.rs` has 17 variants in `cli::QuantMethod`. Most DELETE under the new policy world: all 11 `Imatrix*` variants (replaced by `--imatrix` flag), all 4 `DynamicQuant*` variants (DWQ reserved), flat `Q2`/`Q4`/`Q8` (subsumed by `q4_0/q8_0`), `Q2KS`/`Q2K`/`Q3*` (LlamaFtype 10-13 — but per amendment A above, Q2_K and Q3_K_S/M/L ARE in v1 scope after all, so these stay). Decision §6 names CLI values that **don't exist in main.rs today**: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `iq4_nl`, `f32`, `bf16`. **Amendment:** §P1 §"What" explicitly states that `cli::QuantMethod` enum is REWRITTEN to match Decision §6's surface (StandardPolicy tier names + ApexPolicy tier names + reserved values with typed-error stubs). The current 17-variant enum dies wholesale; the new enum is a verbatim 1:1 of Decision §6.

### L. Vision-pattern gate currently lives inside `KQuantCodecQuantizer`, not the dispatcher. (main-dispatch.md)

ADR Decision §"Vision tensor pattern" says the dispatcher checks vision-pattern membership BEFORE calling the policy. But today, the gate is BURIED inside `KQuantCodecQuantizer`'s `quantize_tensor` impl. **No ADR amendment needed** — the move is implicit in the new shape. **Implementation note for P2:** the new seek-back writer's per-tensor loop must call `is_vision_tensor_pattern(name) || is_audio_tensor_pattern(name)` BEFORE invoking `policy.target_for()`. Pattern-match → emit F16 directly; else → policy → quantizer.

### M. Three `HF2Q_*` env vars retire with the seek-back writer. (main-dispatch.md)

Retired: `HF2Q_STREAMING_PHASE3`, `HF2Q_STREAMING_PHASE3_MUT`, `HF2Q_USE_LEGACY_DWQ_Q4_0`. The MUT × Phase-4.5 incompat guard at L1149-1174 also retires. **No ADR amendment, no migration code.** Per [[feedback-no-backwards-compat-2026-05-18]]: hf2q is pre-public-release, retired env vars get deleted from `parse_env`, callers compile-fail (caught at the same commit), grep + fix. No CHANGELOG appeasement; no deprecation message at startup.

## Findings — non-ADR (implementation notes)

These don't change the ADR but matter for the P0/P1/P2 implementation.

- **`mod.rs` is wholly orchestration.** Zero kernels. All P0 ports come from k_quant.rs + q_legacy.rs. The trait header + QuantizeError enum reshape happens in-place at `src/quantize/mod.rs` (rename optional — `src/quantize/quantizer.rs` would parallel `src/quantize/policy.rs`, but not required).
- **k_quant_codec.rs is pure dispatch shim.** No kernels here. The metadata table (84/110/144/176/210 bytes_per_block; 256/32 elements_per_block; bpw table) gets methods on `GgmlType` enum — values trace to `/opt/llama.cpp/ggml/src/ggml-common.h` `static_assert`s, not k_quant_codec.rs.
- **layer_mix.rs's `is_vision_tensor_pattern` (L366) is canonical.** Move to `src/quantize/vision.rs` per ADR. Sibling `is_audio_tensor_pattern` is NEW code (per finding E).
- **q_legacy's 6 BLOCK_*_SIZE constants** are externally imported by gguf.rs size estimator. P0 ports re-export them from `ggml_quants/<type>.rs`. The GGUF writer's byte-size estimator is orthogonal to P0; P2 rewrites it.
- **96 inline tests in k_quant.rs (2474 LOC, 45% of file)** are tied to deleted dispatchers. P0's byte-cmp fixtures against `ggml_quantize_chunk` reference output supersede them — DO NOT port these tests; write new ones against the `tests/fixtures/ggml_quants/*.bin` reference set.
- **35-test test block in k_quant_codec.rs (1073 LOC)** — same fate; superseded by P0 byte-cmp.
- **42% of q_legacy.rs (896 LOC) is test code** — split by type into `ggml_quants/<type>.rs#[cfg(test)]`. `adr022_iq4_nl_codebook_byte_equal_to_llama_cpp` test at L2009 is load-bearing parity gate; moves verbatim.
- **40 tests in layer_mix.rs** — 28+ on `target_for` move with the impl to `standard_policy.rs`; 6 vision-pattern tests move to `vision.rs`.

## Action plan — P-1 closeout

1. ✅ Vendor mudler @ 63c5048b (done, iter 1)
2. ✅ Record llama.cpp pin (done, iter 1)
3. ✅ Per-file audits (6/7 done, main-dispatch in flight)
4. **In progress:** synthesis + ADR amendment commit
5. **Pending:** main-dispatch agent completion + final audit-results table update
6. **Pending:** apply 8 amendments (A-H above) to ADR-033 §"Plan" + §"Audit results"

Once all 7 audit files exist + the ADR amendment commit lands, P-1 is exit-gated.
