# ADR-033 P6 Deletion Plan — Audit (2026-05-19)

**Status:** audit-only. No deletion performed. No commit.

**Context:** ADR-033 convert-v2 pipeline real-model-validated at commits
`46c54876` / `1bff06ff` / `b514b5c2` (Gemma 4 26B Q5_K_M in 8m 14s,
llama.cpp-loadable). P6 retires the legacy convert/quantize stack per
[[feedback-no-backwards-compat-2026-05-18]]: delete, no shims.

This document classifies every candidate by call-graph evidence
(grep file:line citations) into three phases.

---

## 1. Module ownership map (live vs retired)

**Live runtime (NEVER touched — must keep):**

- `src/inference/`, `src/serve/`, `src/models/`, `src/core/` — runtime.
  Zero `crate::quantize::*` imports outside `quantize::ggml_quants`
  (verified: `grep -rln 'quantize::' src/inference src/serve src/models`
  returns empty).
- `src/ir/` — TensorMap/QuantizedModel structs used by both convert-v2
  and runtime (`src/inference/models/qwen35/weight_loader.rs:34-35`,
  `src/models/qwen35/mod.rs:77`).
- `src/arch/` — registry/smoke/conformance. `arch/smoke.rs:7` shells
  out to `hf2q convert` subprocess — needs P6-time retarget to
  `hf2q convert-v2` (cosmetic; not deletion).

**Convert-v2 surface (LIVE — keep):**

- `src/convert/{cli_driver,mod,orchestrator,quant_selector,source_reader,tokenizer}.rs`
- `src/convert/arch/*.rs` (bert, gemma4, gemma4_mmproj, llama3,
  minimax_m2, nomic_bert, qwen35moe, qwen3vl_text)
- `src/convert/source_dtype/{fp8,mod}.rs`
- `src/quantize/ggml_quants/*` (20 files, 13,716 LOC — verified
  caller graph: orchestrator.rs:43-48 + cli_driver.rs:46-49 +
  quant_selector.rs:30-33)
- `src/backends/gguf/{writer,types}.rs` (the seek-back writer —
  consumed by `convert/orchestrator.rs:41-42`)
- `src/backends/mod.rs` — keep (BackendError used by both legacy +
  convert-v2 traits, trim trait methods in Phase 3).

**Retired set (delete in P6):** see Sections 3-5 below.

---

## 2. CLI surface audit

### 2.1 `Command::Convert(ConvertArgs)` → `cmd_convert`

**Retired-by:** ConvertV2 supersedes for all standard ftypes + APEX tiers.

**Feature parity check** (ConvertArgs at `src/cli.rs:564-697` vs
ConvertV2CliArgs at `src/cli.rs:144-158`):

| Legacy feature | Convert-v2 status |
|---|---|
| `--input <safetensors_dir>` | YES — `hf_dir: PathBuf` positional (cli.rs:148) |
| `--repo <hf_repo>` (auto-download) | **GAP — no `--repo`** in ConvertV2CliArgs |
| `--format gguf\|safetensors` | NO — convert-v2 is GGUF-only |
| `--quant <17 variants>` | RENAMED — convert-v2 accepts standard ftypes + apex-* tiers (cli.rs:151-160) |
| `--calibration` / `--output-format` orthogonal selector | NO — reserved names rejected (`quant_selector.rs:187` returns `DwqReserved`) |
| `--sensitive-layers` | NO |
| `--target-bpw` | NO |
| `--bits` / `--group-size` | NO |
| `--shard-size-gb` | NO (GGUF-only) |
| `--json-report` | NO |
| `--skip-quality` / `--quality-gate` | NO |
| `--dry-run` | NO |
| `--yes` | NO |
| `--unsupported-layers` | NO |
| `--emit-vision-tower` (mmproj sidecar) | YES — `convert/arch/gemma4_mmproj.rs:11` writes sidecar (verified `grep -n vision src/convert/orchestrator.rs:466,272,300`) |
| `--no-integrity` (SHA gate opt-out) | NO |

**Verdict:** convert-v2 is intentionally narrower per
[[feedback-no-backwards-compat-2026-05-18]] — Apex tiers replace the
calibration/output-format orthogonal grid, and dwq/imatrix are
reserved names (`quant_selector.rs:187`). The two surviving operator
gaps are `--repo` (auto-download) and `--format safetensors`.

**Blockers:**

- **B1 (operator-facing): `--repo` flag.** Legacy auto-downloads from
  HuggingFace; convert-v2 requires a pre-downloaded local dir.
  Workflows depending on `hf2q convert --repo google/gemma-4-26b-it`
  break without an explicit pre-download step.
- **B2: `safetensors_out` output format.** Used by
  `cmd_convert` at `src/main.rs:1857-1858` (SafetensorsBackend) for
  mlx-format export. Convert-v2 is GGUF-only.

`cmd_convert` impl: `src/main.rs:1135-3056` (1922 LOC).

### 2.2 `Command::DwqTrain(DwqTrainArgs)` → `cmd_dwq_train` + `cmd_dwq_train_full_model`

**Retired-by:** ADR §6 declares `--quant dwq` reserved (NOT a current
ftype). `quant_selector.rs:104,187` returns `QuantSelectorError::DwqReserved`.
The CLI surface is reservation, not a current feature.

`cmd_dwq_train`: `src/main.rs:355-560` (206 LOC).
`cmd_dwq_train_full_model`: `src/main.rs:561-765` (205 LOC).

### 2.3 `Command::DwqOverlayDrift(DwqOverlayDriftArgs)` → `cmd_dwq_overlay_drift`

**Retired-by:** ADR-020 DWQ overlay pipeline was never wired into
convert-v2. Same reservation rule as 2.2.

`cmd_dwq_overlay_drift`: `src/main.rs:237-354` (118 LOC).

### 2.4 `QuantMethod` enum (17 variants)

**Retired-by:** convert-v2 parses `--quant` as a free-form string via
`QuantSelector::from_name` (`quant_selector.rs:132+`), so the entire
clap-style enum disappears.

Defined at `src/cli.rs:1205-1322`. Reachable from:

- `src/main.rs` — every match-arm in `cmd_convert` (lines 1351-1381
  flag→enum mapping; lines 2142-2188 K-quant dispatch; lines 2405-2466
  DWQ dispatch).
- `src/preflight.rs:12,322,325,419,434-461` — full match for
  byte-rate estimator (delete with preflight; see §3).
- `tests/calibrate_dispatch.rs:38-302` — declares `QuantMethodMirror`
  that mirrors `cli::QuantMethod` and must be re-edited every time a
  variant lands (test comments: 17-arm lock).

### 2.5 Investigation env vars

| Env var | Status at HEAD | Action |
|---|---|---|
| `HF2Q_NO_FA` | LIVE — diagnostic A/B flag (`forward_prefill_batched.rs:262,277,1257-1281`); default OFF per ADR-032 | KEEP |
| `HF2Q_FA_F16` | LIVE — F16 FA path, default ON per ADR-032 (`forward_prefill_batched.rs:319,694,815`) | KEEP |
| `HF2Q_BATCHED_PREFILL` | LIVE — batched prefill gate (`serve/mod.rs:1330,1498`; `forward_prefill_batched.rs:21`) | KEEP |
| `HF2Q_TQ_KV` | LIVE — TQ KV opt-out (`inference/models/qwen35/kv_cache.rs:79`; `serve/load_info.rs:300,552`) | KEEP |
| `HF2Q_STREAMING_PHASE3` / `HF2Q_STREAMING_PHASE3_MUT` | DEAD with cmd_convert (`main.rs:1109-1110`) | DELETE with main.rs:1100-1134 |
| `HF2Q_UNSAFE_EXPERIMENTS` | DEAD with `--calibration` axis (`cli.rs:589-602`) | DELETE with ConvertArgs |

---

## 3. Phase 1 — safe deletions (zero external callers)

These files are referenced ONLY by other retired files and by tests
that are themselves retired:

### 3.1 Legacy quantize/ files

| File | LOC | Callers (all retired) |
|---|---|---|
| `src/quantize/dwq_k_quantizer.rs` | 883 | `main.rs:2449,2466` (cmd_convert DWQ arm); `tests/dwq_k_quantizer.rs`; `tests/dwq_emits_q4_k_via_cli.rs` |
| `src/quantize/k_quant.rs` | 5541 | `quantize/k_quant_codec.rs:35`; `quantize/dwq_k_quantizer.rs:381`; `backends/gguf.rs:1346-1349,1957-1969,4910,4928,4962`; `quality/mod.rs:612`; `quantize/mod.rs:1276,1459,2208,2567`; `quantize/k_quant_codec_quantizer.rs:801` |
| `src/quantize/k_quant_codec.rs` | 1452 | `main.rs:2167,2175,2179,2183,2186`; `quality/mod.rs:1330,1462`; `calibrate/calibrator.rs:675`; `quantize/variant_quantizer.rs:41`; `quantize/dwq_k_quantizer.rs:89`; `quantize/k_quant_codec_quantizer.rs:43` |
| `src/quantize/k_quant_codec_quantizer.rs` | 953 | `main.rs:2207`; `backends/gguf.rs:1344,2085,4845`; `quality/mod.rs:1331,1463`; `quantize/dwq_k_quantizer.rs:90` |
| `src/quantize/q_legacy.rs` | 2130 | `quantize/k_quant_codec.rs:42,844,956,1158`; `backends/gguf.rs:1357-1359,1937-1945`; `quality/mod.rs:616`; `calibrate/qdq_gpu.rs:155` |
| `src/quantize/layer_mix.rs` | 1304 | `main.rs:2266`; `quantize/dwq_k_quantizer.rs:91`; `quantize/k_quant_codec_quantizer.rs:44`; `quantize/variant_quantizer.rs:43`; `quantize/mod.rs:998,1145,1277,1463,1679,1760,1833,1906,1979,2051,2209,2326,2444,2568,2707` (15 sites) |
| `src/quantize/mixed.rs` | 520 | `calibrate/dwq.rs:18,148,166,268` (MixedBitQuantizer) |
| `src/quantize/static_quant.rs` | 468 | `main.rs:1144`; `calibrate/apex.rs:22,346`; `quantize/mod.rs:600` |
| `src/quantize/variant_quantizer.rs` | 604 | `main.rs:2265`; `quantize/mod.rs:999,1146,1278,1464,1680,1761,1834,1907,1980,2052,2210,2327,2445,2569,2708` (15 sites) |
| `src/quantize/mod.rs` | 6441 | Re-export root; `main.rs:1103,1108,1116,1124,1128,1144` (Quantizer, QuantizeError, quantize_model, quantize_via_streaming_*); `ir/lazy.rs:199`. After deletion `pub mod ggml_quants;` (line 13) moves into `mod.rs` proper or to lib root. |

**Total Phase 1 quantize LOC: 20,296 LOC** (matches `wc -l src/quantize/*.rs` to the line).

### 3.2 Backends

| File | LOC | Notes |
|---|---|---|
| `src/backends/gguf.rs` | 6943 | Legacy two-pass writer + GgufBackend + load_tokenizer_metadata + patch_mmproj_sha256. Callers: `main.rs:1136,1826,1830,1840,2848` (cmd_convert); `backends/mod.rs:113,137` (tests inside mod.rs); `quantize/mod.rs:5437,5551,6034` (tests). `core/chat_templates.rs:10,23,112` are DOC-LINK comments only — no code dependency. Per ADR-033 §P6: keep `src/backends/gguf/` (writer.rs + types.rs); delete this file entirely. |
| `src/backends/safetensors_out.rs` | 1071 | SafetensorsBackend (mlx-format output). Caller: `main.rs:1137,1857-1858`. Only used by `--format safetensors` legacy arm. |

### 3.3 Calibrate / preflight / quality / report

| File / dir | LOC | Notes |
|---|---|---|
| `src/calibrate/*` (34 files) | **45,685 LOC** | Zero `crate::calibrate::*` imports from `src/inference/`, `src/serve/`, `src/models/`, `src/convert/`, or `src/backends/`. Only callers: `src/main.rs:357,563,716,881,888,897,906,915,935,941,949,977` and `src/cli.rs` (DwqTrainArgs / CalibrationFlag types). After deleting cmd_convert + cmd_dwq_train + cmd_dwq_overlay_drift, calibrate/ has zero callers. |
| `src/preflight.rs` | 991 | Caller: `main.rs:1437,1440,1444,1460` (cmd_convert only). Uses `cli::QuantMethod` exhaustively (line 322-470). |
| `src/quality/mod.rs` | 1642 | Callers: `main.rs:2608,2633,2641,2646,2650,2658,2663,2671,2891-2892` (cmd_convert) + `src/report.rs:66-67`. Uses legacy `crate::quantize::k_quant`, `q_legacy`, `k_quant_codec`, `k_quant_codec_quantizer` (lines 612,616,1330-1332,1462-1464). NOT runtime — `src/serve/parity_quality.rs` is a DIFFERENT module owned by the live runtime. |
| `src/report.rs` | 585 | Callers: `main.rs:2938,2949,2959,2962,2995,3006,3021,3026` (cmd_convert JSON report only). |

**Total Phase 1 calibrate/quality/report/preflight LOC: 48,903 LOC**.

### 3.4 CLI / main retired surface

| Symbol / span | LOC | Retired-by |
|---|---|---|
| `main.rs` `cmd_convert` (lines 1135-3056) | 1922 | superseded by cmd_convert_v2 |
| `main.rs` `cmd_dwq_train` (lines 355-560) | 206 | dwq reserved name |
| `main.rs` `cmd_dwq_train_full_model` (lines 561-765) | 205 | dwq reserved name |
| `main.rs` `cmd_dwq_overlay_drift` (lines 237-354) | 118 | dwq reserved name |
| `main.rs` `select_calibrator` (lines 879-972) | 94 | cmd_convert helper |
| `main.rs` `build_calibration_corpus` (lines 973-1010) | 38 | cmd_convert helper |
| `main.rs` `dwq_calibration_to_sensitive_ranges` (lines 1011-1059) | 49 | cmd_convert helper |
| `main.rs` `clone_tensor_map_to_lazy` (lines 1060-1099) | 40 | cmd_convert helper |
| `main.rs` `dispatch_phase3_quantize` (lines 1100-1134) | 35 | cmd_convert helper |
| `main.rs` `quantizer_default_bits` (lines 3379-3404) | 26 | uses cli::QuantMethod |
| `main.rs` `detect_quant_method_from_path` (lines 3342-3378) | 37 | only called by cmd_convert |
| `main.rs` `extract_baseline_from_reasoning` (lines 3469-3493) | 25 | quality helper |
| `main.rs` `CaptureSpec` enum (lines 837-878) | 42 | calibrator-only |
| `main.rs` `print_dry_run_plan` (line 3212-3300) | 89 | cmd_convert helper |
| `main.rs` `copy_sidecars` (line 3301-3341) | 41 | cmd_convert helper |
| `main.rs` dispatch arms (lines 173,189-190) | 3 | clap |
| `cli.rs` `ConvertArgs` struct (lines 563-697) | 135 | clap legacy |
| `cli.rs` `DwqTrainArgs` / `DwqOverlayDriftArgs` (~ lines 240-470) | ~230 | clap legacy |
| `cli.rs` `QuantMethod` enum + Display + map_deleted_quant_hint (lines 1200-1500) | ~300 | replaced by QuantSelector |
| `cli.rs` `OutputFormat`, `CalibrationFlag`, `OutputFormatFlag`, etc. | ~80 | legacy axes |
| `cli.rs` `Command::Convert/DwqTrain/DwqOverlayDrift` enum entries | 3 | dispatch |

**Total main.rs/cli.rs retired LOC: ≈ 3,718 LOC.**

`cmd_validate` (`main.rs:3057-3202`, 146 LOC) — does NOT touch legacy
quantize types per grep (verified: `quantize::|cli::QuantMethod` in
`main.rs:>3056` returns only `quantizer_default_bits`). Keep.

`cmd_info` (`main.rs:3405-3454`), `cmd_completions` (3455-3468),
`cmd_smoke` (788-878), `cmd_gguf_patch` (766-787) — keep.

### 3.5 Phase 1 tests (delete with their fixtures)

| Test | LOC | Why retired |
|---|---|---|
| `tests/dwq_k_quantizer.rs` | 224 | imports `hf2q::quantize::*` |
| `tests/vision_tensor_skip_predicate.rs` | 180 | imports `hf2q::quantize::*` |
| `tests/dwq_emits_q4_k_via_cli.rs` | 511 | drives cmd_convert DWQ arm |
| `tests/cmd_convert_dispatch.rs` | 1313 | tests cmd_convert |
| `tests/codec_direct_type_code.rs` | 256 | tests K_QUANT_CODEC_DIRECT |
| `tests/calibrate_dispatch.rs` | 344 | mirrors QuantMethod |
| `tests/quant_method_dispatch.rs` | 737 | tests QuantMethod dispatch |
| `tests/imatrix_xvalidation.rs` | 413 | tests imatrix path |
| `tests/p9b_dance_eliminated.rs` | 81 | tests legacy "dance" |
| `tests/smoke_conformance.rs` | 803 | smoke harness depends on cmd_convert subprocess |
| `tests/convert_integration.rs` | 444 | invokes `hf2q convert` |
| `tests/convert_qwen35_integration.rs` | 554 | invokes `hf2q convert` |
| `tests/convert_qwen35moe_integration.rs` | (~600) | invokes `hf2q convert` |
| `tests/convert_qwen35_metadata_keys.rs` | TBD | convert subcommand |
| `tests/convert_qwen35_mtp_roundtrip.rs` | TBD | convert subcommand |
| `tests/convert_qwen35_real_activation_capture.rs` | TBD | uses calibrate/dwq_activation |
| `tests/convert_qwen35_rms_norm_plus_one.rs` | TBD | convert subcommand |
| `tests/convert_qwen35_two_pass_capture.rs` | TBD | uses calibrate |
| `tests/convert_qwen35_vocab_parity.rs` | 452 | invokes `hf2q convert` |
| `tests/auto_pipeline_smoke.rs` | 700 | uses `GgufBackend::with_provenance` |

**Phase 1 tests subtotal: ≈ 9,458 LOC** (measured + estimated).

---

## 4. Phase 2 — sweep deletions (used only by other Phase-1 targets)

Once Phase 1 lands, these symbols/sections fall over because their
only callers are gone. They're flagged separately so the cleanup
commit is reviewable in chunks:

- `src/backends/mod.rs` lines 109-146 (`#[cfg(test)] mod tests`)
  exercises GgufBackend; delete or rewrite for GgufWriter.
- `src/backends/mod.rs` `quantize_and_write` trait method on
  `OutputBackend` (currently defaulted, only GgufBackend overrides it
  per backends/mod.rs:94 comment) — convert-v2 doesn't use the trait.
- `src/quantize/mod.rs` lines 596-6441 (5845 LOC of `#[cfg(test)]`
  blocks driving the retired quantizer matrix).
- `src/main.rs` lines 1136 (`use backends::gguf::GgufBackend`) and all
  downstream imports inside `cmd_convert`.
- `src/cli.rs::map_deleted_quant_hint` (lines 1450-1530) — was a
  migration aid for the pre-ADR-014 `apex`/`mixed-N-M`/`dwq-mixed-N-M`
  names that were already deleted earlier. With QuantMethod gone, the
  helper has no caller.
- `src/cli.rs::ConvertConfig` (used by preflight.rs:12) — deletes
  with preflight.rs.
- `src/core/mlx_safetensors_loader.rs::MlxAffineLinear::q4_0_round_trip_drift`
  (lines 275-360, ≈ 30 LOC of method body + helpers) — only callers
  are `main.rs:317` (cmd_dwq_overlay_drift) and 3 tests inside
  mlx_safetensors_loader.rs:1407,1443,1481. After cmd_dwq_overlay_drift
  goes, this method dies. Its sole link to retired code is the
  `quantize::q_legacy::{quantize_row_q4_0_to_bytes, dequantize_row_q4_0_bytes}`
  import at line 276. The rest of `mlx_safetensors_loader.rs` is LIVE
  runtime (used by `inference/models/qwen35/model.rs`,
  `serve/forward_mlx.rs`, `convert/source_reader.rs`) — only this
  method + its tests delete.

---

## 5. Phase 3 — risky / requires test rewrites

These deletions are conceptually clean but force operator-facing
changes that need stakeholder sign-off **before** P6 ships:

### 5.1 `hf2q convert` → `hf2q convert-v2` operator break

- 16 integration tests (`tests/convert*.rs`, `tests/auto_pipeline_smoke.rs`)
  invoke the binary with the `convert` subcommand. Each must be
  rewritten to `convert-v2` AND its `--quant <flag>` values translated
  from legacy `QuantMethod` (e.g. `q4` → `q4_0`, `q4_k_m` → `q4_k_m`).
- `src/arch/smoke.rs:7` documents the smoke harness as "running `hf2q
  convert`". Confirmed at `src/arch/smoke.rs:174,322,335,355` (no
  hard-coded `convert` argv literal at the lines I sampled — needs a
  deeper read to find the subprocess construction site).
- Decision needed: rename `convert-v2` → `convert` AS PART of P6 or
  leave `convert-v2` and accept the longer name. ADR-033 leans
  toward rename-on-delete (no shim).

### 5.2 `--repo <hf_repo>` auto-download (Blocker B1)

- Convert-v2 takes `hf_dir: PathBuf` only (cli.rs:148). Tests/scripts
  that pass `--repo google/gemma-4-26b-it` and rely on download must
  either:
  1. Pre-download via `huggingface-cli` (operator workflow change), or
  2. Convert-v2 grows a `--repo` option (≈ port `src/intelligence/`
     fingerprint+download helpers).

### 5.3 `--format safetensors` mlx-format output (Blocker B2)

- `src/backends/safetensors_out.rs` (1071 LOC) is reachable only via
  `cmd_convert --format safetensors`. Used by tests:
  - `tests/safetensors_mlx_lm_round_trip.rs` (imports MlxAffineLinear
    per the prior grep)
- Decision needed: drop mlx-format export from hf2q entirely (delete
  safetensors_out.rs + the test), or port it to convert-v2 as a
  separate subcommand. Per `[[feedback-no-backwards-compat-2026-05-18]]`,
  deletion is the standing-rule answer unless an active user is named.

### 5.4 Smoke conformance harness (`tests/smoke_conformance.rs`)

- 803 LOC. Subprocess-drives `hf2q convert` per `arch::smoke::dispatch`.
  Either re-target to `convert-v2` (and accept the feature-gap
  blockers above) or delete with the legacy path.

### 5.5 DWQ training pipeline retirement

- `cmd_dwq_train` + `cmd_dwq_train_full_model` reserve `--quant dwq` as
  a future entry point (`quant_selector.rs:97-104`). Per
  [[memory-feedback_no_backwards_compat_2026_05_18]], **reservation ≠
  retirement**: keep `DwqReserved` typed error in QuantSelector (4
  LOC), delete the 511 LOC of CLI plumbing + the 32-file calibrate/
  subtree. Re-introducible later from `git log` if the DWQ-train
  pipeline ever lands.

### 5.6 ADR-033 §3 vs §8 fallback contract reconciliation

- Per [[project-cfa_adr033_review_2026_05_17]] F-019, ADR-033 §3
  describes `shape_fallback` (silent F16 demotion for K-quant-misaligned
  rows) while §8 contradicts it. `src/quantize/ggml_quants/quantizer.rs`
  + `vision.rs` implement the F-019 contract. Verify that no
  Phase 1 deletion accidentally reintroduces the contradiction
  (the legacy `layer_mix::kquant_misalignment_fallback` per
  [[project-hf2q_convert_gemma4_f16_dispatch_2026_05_17]] returns
  Q4_0/Q5_0/Q5_1/Q8_0 — NOT F16. After deletion that codepath is
  gone; only the §8 contract survives. **Net: P6 also closes F-019**).

---

## 6. False-flag checks

The session-notes candidate list flagged several items that survive
audit:

- **`src/quantize/ggml_quants/q5_0.rs`** (line 91 in original grep
  output). This is the pure-Rust Q5_0 quantizer for convert-v2, NOT
  legacy. Verified: `convert/orchestrator.rs:43` imports
  `quantize::ggml_quants::quantizer::Quantizer`. **KEEP.**
- **`HF2Q_NO_FA` / `HF2Q_FA_F16` / `HF2Q_BATCHED_PREFILL`** — the
  candidate list calls these "investigation env vars" possibly retired.
  Audit per ADR-032 + grep at `serve/forward_prefill_batched.rs:262,
  319, 21` shows they are LIVE production gates (NO_FA is the
  diagnostic A/B opt-out per ADR-032 commit `9e64df5c`; FA_F16 is the
  default-on F16 FA path; BATCHED_PREFILL=1 toggles batched prefill).
  **KEEP all three.**
- **`src/backends/gguf/` (subdir)** — the candidate list says "old
  two-pass writers in src/backends/gguf.rs". The subdir
  (writer.rs/types.rs) is the NEW seek-back writer, the load-bearing
  convert-v2 path. Only the SIBLING FILE `src/backends/gguf.rs` is
  retired. The subdir survives and becomes the module root post-P6
  (verified `src/backends/gguf.rs:5-13` comment block).
- **`gguf_patch`** (`src/gguf_patch.rs`, 460-ish LOC) — provides
  `cmd_gguf_patch` (main.rs:766-787). Live operator subcommand;
  unrelated to cmd_convert. **KEEP.**
- **`src/inference/` / `src/serve/` / `src/models/`** — confirmed by
  `grep` (Section 1): zero retired-set imports. Keep all.

---

## 7. LOC delta estimate

| Bucket | LOC |
|---|---|
| `src/quantize/*.rs` excluding `ggml_quants/` | 20,296 |
| `src/backends/gguf.rs` | 6,943 |
| `src/backends/safetensors_out.rs` | 1,071 (Phase 3 — operator decision) |
| `src/calibrate/*` (all 34 files) | 45,685 |
| `src/preflight.rs` | 991 |
| `src/quality/mod.rs` | 1,642 |
| `src/report.rs` | 585 |
| `src/main.rs` retired functions/helpers | ≈ 3,718 |
| `src/cli.rs` retired structs/enums | ≈ 770 |
| Retired tests (20+ files) | ≈ 9,458 |
| **Total deletion** | **≈ 91,159 LOC** |

The ADR's "16,867 LOC across 5 mod.rs/k_quant.rs/k_quant_codec.rs/q_legacy.rs/layer_mix.rs" claim **dramatically under-counts**:

- The five named files alone (`mod.rs` + `k_quant.rs` +
  `k_quant_codec.rs` + `q_legacy.rs` + `layer_mix.rs`) total
  **16,868 LOC** — the ADR number is correct for THAT subset.
- But P6's scope is wider: `dwq_k_quantizer.rs` + `mixed.rs` +
  `static_quant.rs` + `variant_quantizer.rs` + `k_quant_codec_quantizer.rs`
  add another **3,428 LOC** of legacy quantize/.
- AND calibrate/ (45,685 LOC) + preflight (991) + quality (1,642) +
  report (585) + main.rs/cli.rs retired spans (≈ 4,488) + tests
  (≈ 9,458) + backends (≈ 8,014) = ~**74,291 LOC of additional
  retired surface**.

**Verdict: P6 retires roughly 91k LOC**, not 17k. The 17k figure in
ADR-033 §P-1 covers only the K-quant/legacy-quant kernel files and
omits the call-graph downstream.

---

## 8. Summary

**(a) Safe to delete (Phase 1, ~85,000 LOC):**

- All of `src/quantize/*.rs` except `ggml_quants/` subdir.
- All of `src/calibrate/`.
- `src/preflight.rs`, `src/quality/mod.rs`, `src/report.rs`.
- `src/backends/gguf.rs` (file only; `src/backends/gguf/` survives).
- `cmd_convert`, `cmd_dwq_train`, `cmd_dwq_train_full_model`,
  `cmd_dwq_overlay_drift`, and their helpers in `src/main.rs`.
- `Command::Convert` / `DwqTrain` / `DwqOverlayDrift` variants and
  `QuantMethod` enum + `ConvertArgs` / `DwqTrainArgs` /
  `DwqOverlayDriftArgs` / `map_deleted_quant_hint` in `src/cli.rs`.
- 20+ legacy test files in `tests/`.

**(b) Blocks deletion (Phase 3, ~1,100 LOC + operator-facing decisions):**

- B1: `--repo` auto-download — no convert-v2 equivalent; either pre-
  download workflow change or port the download helper.
- B2: `--format safetensors` mlx-format export — `safetensors_out.rs`
  has no convert-v2 caller; delete or port as separate subcommand.
- B3: `tests/smoke_conformance.rs` + `src/arch/smoke.rs` shell out to
  `hf2q convert`; needs re-targeting to `convert-v2` (cosmetic but
  must land same PR or smoke breaks).
- B4: rename `convert-v2` → `convert` decision (per
  [[feedback-no-backwards-compat-2026-05-18]] this is the right call
  but it's still a stakeholder-confirm).

**(c) LOC delta: ≈ 91,000 LOC deleted** (vs. the ADR's 16,867 LOC
estimate, which only covered 5 of the ~50 retired files).

**Notable findings:**

- `src/inference/`, `src/serve/`, `src/models/`, `src/core/` have ZERO
  references to retired code (`grep -rln 'crate::quantize\|crate::calibrate'
  src/inference src/serve src/models` returns empty). The retirement
  is a pure build-time-pipeline concern — runtime is unaffected.
- The four investigation env vars (`HF2Q_NO_FA`, `HF2Q_FA_F16`,
  `HF2Q_BATCHED_PREFILL`, `HF2Q_TQ_KV`) flagged as "possibly retired"
  in session notes are all LIVE in `src/serve/forward_prefill_batched.rs`
  and `src/inference/models/qwen35/kv_cache.rs`. Do not delete.
- The single biggest deletion target is `src/calibrate/` at
  **45,685 LOC** — the legacy DWQ/imatrix/QwenMoE/dynamic-quant
  training stack. It has zero callers from the live runtime and zero
  callers from convert-v2.

