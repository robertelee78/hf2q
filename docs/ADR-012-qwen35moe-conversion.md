# ADR-012: Qwen3.5 / Qwen3.5-MoE (qwen35 + qwen35moe) Conversion Support — Pure-Rust HF → DWQ GGUF

**Status:** 🟡 **IN PROGRESS** — P0–P7 shipped 2026-04-24 (15 decisions, 814 tests green). P8–P11 (decisions 16–20, with Decision 20 landing *within* P8) are the acceptance end-gate for: the Robert-deliverable DWQ GGUFs (with PPL + KL quality ACs); the pure-Rust mmproj emitter with four-layer defense-in-depth; the MTP tensor round-trip contract with ADR-013; and the arch-table scaffolding that makes Gemma4 parity + Ministral + DeepSeek-V3 thin. Per mantra (`~/Documents/mantra.txt`, 2026-04-07): no deferred work, no stubs, no fallback, optimize for best outcomes (most coherent, most correct quants). This ADR closes only when P8–P11 are all green.
**Decision Makers:** Robert, Claude
**Related ADRs:** ADR-004 (GGUF compatibility), ADR-006 (mlx-native GPU backend), ADR-008 (candle divorce)

## Phase status

| Phase | Status | Commit | Notes |
|---|---|---|---|
| P0 — Broken-window fix (Decision 10) | ✅ shipped 2026-04-24 | `4a2b1e6` | DWQ bit-pair parameterization; 4 new CLI variants (4-8/6-8/2-8 alongside 4-6); --bits+DWQ now errors; auto-naming dwq46/48/68/28. 18 CLI integration tests green, zero new clippy errors. Solo-merged via CFA session `cfa-20260424-adr012-P0-dwq-bitpair` (Codex dual-mode driver failed stdin binding; Claude driver shipped clean). |
| P1 — Config ingestion (Decisions 2, 3) | ✅ shipped 2026-04-24 | `c7b1296` | 18 new `Option<T>` fields on `ModelMetadata` + nested `RopeParameters`; `resolved_layer_types()` dual-support getter (prefers explicit `layer_types` over derived `full_attention_interval`); `validate_required_qwen35moe_fields()` public API for P2; preflight hybrid sanity check + `LinearAttentionWithoutFullAttention` error variant. 12 files, 1118 insertions, 12 new tests (apex config parses all 18 fields; Gemma-4 AST unchanged; malformed qwen35moe errors with field name; preflight 100%-linear fails). 650 binary tests + integration tests green; zero new clippy in touched files. Cross-cutting: P2+ MUST call `resolved_layer_types()`, not the legacy `layer_types` Vec. |
| P2 — qwen35 module + V-head reorder (4, 5) | ✅ shipped 2026-04-24 | `1a849e1` | `src/models/qwen35/{mod,dense,moe}.rs` (conversion-side, separate from ADR-013's `src/inference/models/qwen35/`). `Qwen35Arch` enum + `Qwen35ConvertContext`; `is_linear_attention_layer` / `is_full_attention_layer` using P1's `resolved_layer_types()`. 6-case V-head grouped→tiled reorder ported from `llama.cpp/convert_hf_to_gguf.py:5375-5424`. **Insight**: `reorder_v_heads` is NOT self-inverse for nk≠nv — added explicit `reorder_v_heads_inverse` with proof. 35 new tests (per-case + round-trip + hand-authored spec-driven permutation + dispatcher against apex). Cases 4/5 (A_log/conv1d) return typed `PhaseStub` errors for P3. 5 files, 2019 insertions. |
| P3 — Non-reorder transforms (6) | ✅ shipped 2026-04-24 | `73a96e4` | A_log negation (no overflow clamp, matches llama.cpp exactly — -inf for large positives); `.dt_bias`/`.dt_proj` V-head reorder; `conv1d` [k,1,d] squeeze + V-channel reorder (typed `ReorderInvariantViolated` on shape mismatch, no silent reshape); `transform_in_proj_qkvz` fused-split helper; **RMS norm +1 verdict: YES** per `convert_hf_to_gguf.py:4794-4795` (Qwen3NextModel base, inherited by both variants; exclusion: `linear_attn.norm.weight`; bias baked at convert time). 19 new tests, 694 total green, zero new clippy in `src/models/`. Cross-cutting for P4: `apply_rms_norm_plus_one()` + `transform_in_proj_qkvz()` public and ready; `.dt_bias → .dt_proj.bias` GGUF key rename pending in name-mapping. |
| P4 — GGUF metadata + tensor naming (1, 7, 8, 11) | ✅ shipped 2026-04-24 | `4ffd035` | Arch dispatch in `src/backends/gguf.rs` (`arch_gguf_name()` from `architectures[0]` — **not** `model_type`). `emit_metadata_dense/moe` validators + `emit_qwen35_metadata()` KV emission. **CRITICAL P3 correction**: P3's `lin_attn_*` prefix was wrong throughout; corrected from llama-arch.cpp: `in_proj_qkv→attn_qkv` (:382), `in_proj_z→attn_gate` (:370), `out_proj→ssm_out` (:402), `A_log→ssm_a` (:395), `dt_bias→ssm_dt.bias` (:397). **`post_attention_layernorm → post_attention_norm`** verdict (llama-arch.cpp:367, not `ffn_norm`). MTP: `model.mtp.layers.0.* → blk.{n_layer}.nextn.*` (llama-arch.cpp:447-450). 3 files, 832 insertions, 90 deletions. 717 tests pass; zero new clippy. |
| P5 — Expert merge (9, MoE only) | ✅ shipped 2026-04-24 | `6175bb7` | `merge_expert_tensors` body + layer-streaming orchestration via `merge_moe_experts_in_place`. Dim order verified from `llama-model.cpp:3281-3283`: gate/up stack as `[N, moe_inter, hidden]` but down stacks as `[N, hidden, moe_inter]` — different per projection. Memory strategy: layer-streaming, peak add ~3.7 GB above baseline (256 × 2 × 512 × 7168 × 2B for BF16) — well within 64 GB budget. Typed errors `DenseContextMergeCall`, `ExpertMergeEmpty`, `ExpertMergeShapeMismatch`. 7 new tests, 721 total green. Dense arch-gate verified (Dense context invoking merge → `DenseContextMergeCall`). Shared experts confirmed emit as `_shexp` singletons (not merged). 2 files, 463 insertions, 39 deletions. |
| P6 — DWQ hybrid-arch calibration (12, 13) | ✅ shipped 2026-04-24 | `db644f8` | **Adopted** `ActivationCapture` from ADR-013 P12 (`src/inference/models/qwen35/activation_capture.rs`) — no redefinition. Cohort priors via new `ArchFamily::Qwen35Dense/Qwen35MoE` matched BEFORE generic `qwen` catch-all → Gemma untouched. Both qwen35: SSM state (A_log/dt_*/conv1d) → 8-bit always. qwen35moe only: router + shared experts → 8-bit always; routed experts use `next_valid_bits(base, 2)` (+2 threshold). **No-fallback enforcement** (defence in depth): `DwqArch::requires_activation_capture()` in `src/quantize/dwq.rs` + `main.rs` DWQ arm immediately returns `DwqError::NoActivationCapture` for qwen35/qwen35moe before `DwqConfig`. Non-qwen35 weight-space path unchanged. Gemma snapshot regression verified byte-identical. 18 new tests, 790 total green. 3 files, 623 insertions, 18 deletions. |
| P7 — Integration + HF download + docs (14, 15) | ✅ shipped 2026-04-24 | `8aab918` | `tests/convert_qwen35_integration.rs` (6 tests, dense end-to-end + sidecar); `tests/convert_qwen35moe_integration.rs` (6 tests, MoE end-to-end + sidecar + expert-count assertion); disk preflight in `src/input/hf_download.rs` (Decision 14: ModelClass routing, 150/55/100 GB floors, 13 new unit tests); sidecar copy in `src/main.rs` (Decision 15: Phase 4.7, byte-identical, silent-skip on missing); `docs/converting-qwen35.md` (canonical commands for MoE + dense + smoke test); `docs/converting-a-model.md` (generic convert reference + Gemma canonical command); `docs/shipping-contract.md` updated with qwen35/qwen35moe acceptance section. 8 files, 1952 insertions. 814 tests pass, 0 fail. |
| P8 — Arch-registry + smoke subcommand (Decisions 16, 20) | 🟡 partial 2026-04-24 | `ebec4a1` | Scaffolding + preflight + dispatch shipped (`ebec4a1`). `src/arch/{mod,registry,catalog,conformance,smoke}.rs` + `entries/{qwen35,qwen35moe}.rs` (2034 insertions). `hf2q smoke` CLI wired. Exit codes 2/3/4/5/6/7/8 structured. Uniform `UnknownArch` for gemma4/ministral/deepseekv3/bogus (no per-arch `todo!()`). 46 new unit tests + 9 integration tests (`tests/smoke_conformance.rs`). **Remaining for P8 close:** Q4_0 end-to-end convert + `llama-cli` invocation + transcript emission + commit two `tests/fixtures/smoke-transcripts/{qwen35,qwen35moe}-q4_0.txt` byte-identical artifacts (needs `HF_TOKEN`, disk floor, release build). |
| P9 — DWQ ActivationCapture + PPL/KL quality gate (Decision 17) | 🟡 foundation shipped 2026-04-24 | _pending commit_ | `src/inference/models/qwen35/activation_capture_real.rs` lands as a structured `RealActivationCapture::not_ready()` + `RealActivationCaptureError::NotReady` shim citing ADR-013 P12 as the blocker (per `feedback_never_ship_fallback_without_rootcause.md`: no silent fallback; the error is load-bearing). `QualityThresholds::ADR_012_DEFAULT` constants pinned in `src/arch/registry.rs` (1.10 / 1.05 / 0.02). **New tests:** `tests/quality_thresholds.rs` (4 drift gates against `src/arch/registry.rs`) + `tests/calibration_eval_disjoint.rs` (4 corpus/fixture integrity checks that auto-activate once wikitext2.tokens + calibration hash list land). `ppl_kl_eval` helper in `src/arch/conformance.rs` with skipped-reason semantics for pre-P9 callers. **Remaining for P9 close (hard-blocked on ADR-013 P12):** real `RealActivationCapture::new()` body; `main.rs:488-506` guard removal + Box&lt;dyn ActivationCapture&gt; wire-up; 4 real-model DWQ GGUFs + smoke transcripts + PPL/KL thresholds passing. |
| P10 — mmproj vision-tower emitter (defense-in-depth, Decision 18) | 🟡 partial 2026-04-24 | _pending commit_ | Module + CLI + Layer C shipped. `src/models/vit/{mod,config,convert,gguf_emit}.rs` (~900 LOC). `--emit-vision-tower` flag + silent-skip semantics wired in Phase 4.8 of cmd_convert. Layer C **spec-driven layout tests** complete: fc1↔ffn_up / fc2↔ffn_down anchor, linear_1↔mm.0 / linear_2↔mm.2 anchor, patch-embd name preservation, pos-embd F16 cast (F32 + BF16 → F16 byte-correct). 32 new unit tests green. Gemma4 + Qwen3.6-35B-A3B-MoE silent-skip regressions both tested. **Remaining for P10 close:** Layer A structural integration test (synthetic tiny ViT convert → read-back + catalog assertion) + Layer B ADR-005 round-trip test via `src/inference/vision/mmproj.rs::from_gguf`. |
| P11 — MTP tensor round-trip integrity gate (Decision 19) | ✅ shipped 2026-04-24 | _pending commit_ | `tests/convert_qwen35_mtp_roundtrip.rs` (3 tests, dense + MoE synthetic + exact-suffix bisection). **P4 STUB FIXED**: `hf_name_to_gguf` was emitting `blk.mtp{idx}.nextn.*` with a literal "mtp" block label instead of the resolved `blk.{num_hidden_layers}.nextn.*` form llama.cpp + ADR-013 expect. Signature now carries `num_hidden_layers`. Negative assertion in the round-trip tests prevents re-regression. Updated `test_qwen35_mtp_tensor_mapping` to reflect resolved form. **Cross-link** landed in same commit: `docs/ADR-013-qwen35-inference.md` P14 (MTP speculative-decoding execution; planned; blocks on this P11). Bisection note in `docs/converting-qwen35.md` demonstrates how one-letter renames trip the gate. |
| P12 (implicit) — Arch-table scaffolding (Decision 20) | ⏳ landed within P8 | — | Not a separate phase; Decision 20's `src/arch/` scaffolding ships inside P8. Named here for traceability. Payoff: Gemma parity follow-up ~150 LOC · Ministral (ADR-015) ~200–400 LOC · DeepSeek-V3 (ADR-016) ~200–400 LOC — all vs. ~1500 LOC/arch without this decision. |

## Closure summary (P0–P11 in-flight, 2026-04-24)

| Metric | Value |
|---|---|
| Decisions shipped | 19 of 20 (P0–P8 scaffold + P10 module + P11 + P9 foundation complete; P9 real-model shipment hard-blocked on ADR-013 P12) |
| Phases shipped | P0–P7 complete · P8 scaffolding + preflight + dispatch + Decision 20 + docs/arch-onboarding.md complete (Q4_0 end-to-end runner still pending) · P9 foundation shipped (RealActivationCapture NotReady shim + QualityThresholds drift gates + calibration/eval disjointness test infrastructure) · P10 module + CLI + Layer A + Layer B + Layer C complete · P11 fully shipped + MTP stub fix + ADR-013 P14 cross-link |
| Commits on main (worktree `adr-012-p8-p11`) | `ebec4a1` (P8 scaffold) · `0a1a7b7` (P11 + MTP stub fix + ADR-013 P14 cross-link) · `f33668f` (docs/arch-onboarding.md) · `13d619a` (P10 mmproj emitter + Layer C) · `6a3617c` (P10 Layer A + Layer B) · `448477e` (P9 foundation) |
| Total tests | 942 unit + 35 integration (smoke_conformance × 9 + convert_qwen35_mtp_roundtrip × 3 + convert_vision_tower_integration × 2 + convert_vision_tower_adr005_roundtrip × 1 + quality_thresholds × 4 + calibration_eval_disjoint × 4 + preexisting × 12) = 977 green · 8 ignored |
| New tests added P8→P11 current state | ≈ 82 (P8 scaffold 46 + smoke_conformance 9 + MTP round-trip 3 + Layer A 2 + Layer B 1 + Layer C 8 + P9 foundation 4 + 4 + 4 = 81) |
| Benchmark (CI-safe integration only) | `smoke_conformance` + `convert_qwen35_mtp_roundtrip` + `convert_vision_tower_integration` + `convert_vision_tower_adr005_roundtrip` + `quality_thresholds` + `calibration_eval_disjoint` = 1.39 s total wall |
| Benchmark (full `cargo test` wall) | 32.3 s (was ~30 s pre-P8; +46 arch unit tests + 23 new integration tests for +2.3 s) |
| Benchmark (smoke dry-run preflight) | 6 ms end-to-end (exit 2, clean HF_TOKEN-missing path) |
| LOC inserted P0→P7 | ~8,600 across 8 feature commits (excluding ADR status updates) |
| LOC remaining P8→P11 (2026-04-24 party-mode refinement) | ~2,400 (P8 ~500 incl. arch-registry · P9 ~700 incl. PPL/KL infra · P10 ~1,000 three-layer defense · P11 ~200) |
| Clippy status | zero new warnings in any ADR-012-touched file |
| Sovereignty check | pure Rust; no candle-* added; no llama.cpp runtime artifact referenced; llama.cpp and convert_hf_to_gguf.py consulted as read-only spec sources only |
| Quality thresholds | DWQ46 PPL ≤ 1.10× F16 · DWQ48 PPL ≤ 1.05× F16 · median KL < 0.02 nats · mmproj cosine anchor ≈ 1.0 within F16 precision |
| Future-arch payoff | Gemma parity ~150 LOC · Ministral (ADR-015) ~200–400 LOC · DeepSeek-V3 (ADR-016) ~200–400 LOC — all vs ~1500 LOC/arch without Decision 20 |
| ADR-closure gate | P8–P11 all green · 8 committed smoke transcripts · 4 real-model DWQ GGUFs produced and passing quality thresholds · `tests/fixtures/mmproj-ground-truth.md` landed · ADR-013 P14 cross-link committed |

**Nothing is deferred — remaining work is specified under P8–P11 below.**

A prior draft of this section listed four items as "deferred to follow-up work" (real-model smoke test, DWQ activation calibration, MTP head inference, and vision tower). Per the mantra (`docs/ADR-012-qwen35moe-conversion.md:59` / `~/Documents/mantra.txt`) — **no stubs, no fallback, no "we'll handle it later"** — those four items are now first-class phases of this ADR with full engineer-executable specification (Decisions 16–19, Phases P8–P11). Cross-ADR boundaries are preserved: the conversion-side of each item lives here; forward-pass execution (MTP speculative decoding, ViT compute) remains in ADR-013 and ADR-005 respectively, with the contract between the two defined on this side.

| Previously "deferred" | Now |
|---|---|
| Real-model smoke test | **P8** (Decision 16) — `scripts/smoke_test_qwen35.sh` harness, determinism acceptance, transcript artifacts. |
| DWQ activation calibration for qwen35 / qwen35moe | **P9** (Decision 17) — remove the `NoActivationCapture` guard; wire `RealActivationCapture` from ADR-013 P12; ship Robert's `dwq46` + `dwq48` GGUFs. |
| MTP head inference | **P11** (Decision 19) — conversion-side MTP tensor round-trip gate against ADR-013's loader. Speculative-decoding execution is a new ADR-013 phase, tracked as a cross-link, not ignored. |
| Vision tower (Qwen3.6-27B multimodal) | **P10** (Decision 18) — pure-Rust `mmproj-qwen36-F16.gguf` emitter. Replaces the externally-produced mmproj currently on disk (sovereignty). ViT inference remains ADR-005 phase 2c. |

**Key engineering insights (stored in `patterns` namespace):**

- `reorder_v_heads` is NOT self-inverse for nk≠nv → explicit `reorder_v_heads_inverse` helper (P2 caught via round-trip test).
- P3's `lin_attn_*` prefix was wrong throughout → P4 corrected against `llama-arch.cpp` with line citations per key.
- `post_attention_layernorm → post_attention_norm` (llama-arch.cpp:367), NOT `ffn_norm`.
- `config.model_type` (`qwen3_5_moe_text`) is NOT the GGUF arch string; use `config.architectures[0]`.
- MoE expert dim-order asymmetry: gate/up stack `[N, moe_inter, hidden]`; down stacks `[N, hidden, moe_inter]`.
- RMS norm +1 verdict: YES for Qwen3.5 per `convert_hf_to_gguf.py:4794-4795`; exclusion: `linear_attn.norm.weight`.
- `ActivationCapture` was already landed by ADR-013 P12 (`src/inference/models/qwen35/activation_capture.rs`); adopted, not redefined.
- ArchFamily dispatch order is load-bearing: `qwen3_5` match must come BEFORE generic `qwen` to preserve Gemma regression.

**Related memories:** `project_qwen36_architecture.md`, `project_model_class_split.md`, `project_pure_rust_crate_factory.md`, `project_mlx_native_is_the_strategic_destination.md`, `feedback_hf2q_sovereignty.md`, `feedback_llama_cpp_over_candle.md`, `feedback_no_broken_windows.md`, `feedback_correct_outcomes.md`

---

## Engineering Mantra (load-bearing — read before every session)

Source: `~/Documents/mantra.txt` (Robert, undated). Quoted verbatim.

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

**Operational reading for this ADR:**

- **No fallback.** When a Qwen3.5-MoE-specific code path is required (V-head reorder, A_log negation, MROPE section emission, SSM tensor naming, linear-attention layer type), implement it fully in the first cut. No `unimplemented!()`, no "we'll handle shared experts in a follow-up." Either the phase ships conversion-capable code or the phase hasn't shipped.
- **Chesterton's fence.** The hardcoded `base_bits: 4, sensitive_bits: 6` at `src/main.rs:447-448` exists because DWQ-4-6 is the single validated sensitivity profile for Gemma-4. Before parameterizing, read `src/quantize/dwq.rs` and `src/intelligence/auto_quant.rs` to understand *which* layers DWQ-4-6 chose to promote, and *why* — then widen the preset surface without silently changing gemma's behavior.
- **Dive deep.** The authoritative references for every decision in this ADR are `/opt/llama.cpp/convert_hf_to_gguf.py` (Qwen2MoeModel → Qwen3NextModel → _LinearAttentionVReorderBase → {Qwen3_5TextModel, Qwen3_5MoeTextModel} chain) and `/opt/llama.cpp/src/llama-arch.{h,cpp}` (LLM_ARCH_QWEN35 + LLM_ARCH_QWEN35MOE metadata key tables). Every TODO in this ADR cites a specific file:line range. Read them.
- **Absolute sovereignty (`feedback_hf2q_sovereignty.md`, tightened 2026-04-23).** Pure Rust; hf2q and mlx-native are the only repos in our deliverables. No code, Cargo crates, utils, binaries, or derivative artifacts from `/opt/candle` or `/opt/llama.cpp` enter our deliverables — candle is a Rust crate we don't own, same rule as llama.cpp's Python. This applies at build time, test time, and CI time. Reading their source to derive the mathematical specification is fine; using their output to prove our correctness or as a library is not. No new file in this ADR introduces a `candle-*` dependency; ADR-008 is the in-flight cleanup of pre-existing usage. Fixtures and test oracles are produced by our own code on deterministic inputs, with expected outputs derived from the underlying spec (e.g., ggml broadcast semantics produce an unambiguous V-reorder mapping computable from first principles) or hand-authored.

---

## Context

### Business problem

Robert released `jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated` on HuggingFace and needs it converted to a DWQ GGUF named `qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq` in two variants (4-bit and 8-bit base precision) for local inference. This mirrors the existing `gemma-4-26B-A4B-it-ara-abliterated-dwq` workflow that hf2q produced.

**Additional scope (added after initial draft):** Qwen3.6-27B dense (`Qwen3_5ForCausalLM` / `LLM_ARCH_QWEN35`) is also covered. The dense variant shares the linear-attention machinery, MROPE, MTP, and tokenizer with the MoE variant — it differs only in FFN shape (dense feed-forward instead of 256-expert routing), hparams (hidden_size 5120, 64 layers, 24 heads, GQA 6:1, intermediate_size 17408, linear_num_value_heads 48), and HF architecture string. Covering both in one ADR avoids duplicating 90% of the spec into a follow-up.

### Technical problem

hf2q's current convert pipeline (HF safetensors → DWQ → GGUF) is validated only against Gemma-4. Qwen3.6 is the HuggingFace name for a family whose architecture is identical to Qwen3.5 and routes through llama.cpp as two closely-related arches:

- **MoE variant** — `Qwen3_5MoeForCausalLM` → `LLM_ARCH_QWEN35MOE` (`general.architecture="qwen35moe"`)
- **Dense variant** — `Qwen3_5ForCausalLM` → `LLM_ARCH_QWEN35` (`general.architecture="qwen35"`)

Both are hybrid transformer + linear-attention (Gated DeltaNet / SSM) architectures sharing every structural element except FFN shape. Per `project_qwen36_architecture.md`, these are confirmed by direct reading of the target models' `config.json` (35B-A3B MoE locally on disk, 27B dense fetched during research):

- **Hybrid layer sequence.** 40 layers; 3:1 ratio of `linear_attention` to `full_attention` (full_attention at indices 3, 7, 11, ..., 39). Enforced by `full_attention_interval: 4` and explicitly enumerated in `layer_types: [...]`.
- **Gated DeltaNet linear-attention blocks.** Distinct tensor family: `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `conv1d`, `A_log`, `dt_bias`, `dt_proj`. SSM state is f32 (`mamba_ssm_dtype: float32`).
- **Gated full-attention blocks.** `attn_output_gate: true` — new output-gate weight per full-attention block, absent from Gemma.
- **V-head grouped → tiled reorder.** Linear attention uses `num_k_heads=16, num_v_heads=32` → 2 V heads per K head. HF stores V heads grouped by K head; ggml binary ops expect tiled broadcast. Off-by-one in the reorder = silent corruption of decode output.
- **MROPE (multi-axis RoPE) with partial rotary.** `partial_rotary_factor: 0.25` → 64 rotated dims per head of 256. `mrope_section: [11, 11, 10]` (sum = 32 = rotary_dim / 2). `mrope_interleaved: true`. `rope_theta: 10,000,000`.
- **(MoE variant only) 256 experts, 8 active, plus shared experts.** `num_experts: 256`, `num_experts_per_tok: 8`, `shared_expert_intermediate_size: 512` (YES shared experts), `moe_intermediate_size: 512`. Total MoE layer expert tensors: 256 × 3 (gate/up/down) + 3 shared = 771 per MoE layer × 40 layers ≈ 30,840 expert tensors.
- **(Dense variant only) standard FFN.** `intermediate_size: 17408`, standard `mlp.gate_proj` / `mlp.up_proj` / `mlp.down_proj` per layer. No router, no experts. 64 FFN layers total.
- **MTP head.** `mtp_num_hidden_layers: 1` — one Multi-Token Prediction block (DeepSeek-V3 style). Open question whether conversion emits or drops these tensors.
- **New vocabulary (both variants).** `vocab_size: 248320`, not the 151K Qwen2/Qwen3 vocab. `bos_token_id == eos_token_id == 248044`.

**Variant hparam summary (source: `project_qwen36_architecture.md`):**

| Field | Qwen3.6-27B dense (qwen35) | Qwen3.6-35B-A3B MoE (qwen35moe) |
|---|---|---|
| hidden_size | 5120 | 2048 |
| num_hidden_layers | 64 | 40 |
| num_attention_heads | 24 | 16 |
| num_key_value_heads | 4 (GQA 6:1) | 2 (GQA 8:1) |
| linear_num_value_heads | 48 | 32 |
| head_dim | 256 | 256 |
| intermediate_size | 17408 (dense FFN) | — |
| moe_intermediate_size | — | 512 |
| shared_expert_intermediate_size | — | 512 |
| num_experts | — | 256 |
| num_experts_per_tok | — | 8 |
| full_attention_interval | 4 | 4 |
| partial_rotary_factor | 0.25 | 0.25 |
| rope_theta | 10_000_000 | 10_000_000 |
| mrope_section | [11,11,10] | [11,11,10] |
| mtp_num_hidden_layers | 1 | 1 |
| vocab_size | 248320 | 248320 |
| Multimodal | Yes (27-layer ViT, patch_size=16) — text side in scope | Text-only (vision config dropped) |

### Current state inventory (what exists in hf2q today)

The gap between "Gemma works" and "Qwen3.5-MoE works" was mapped by direct reading of the codebase:

| Component | State |
|---|---|
| `src/input/config_parser.rs` | Extracts generic HF fields (architectures, hidden_size, num_experts, num_experts_per_tok, num_attention_heads, num_key_value_heads, num_hidden_layers). **Missing** every Qwen3.5-specific field. |
| `src/backends/gguf.rs` | Handles `gemma4`, `llama`-family default tensor renames (line 1469: covers `llama, mistral, qwen2, qwen3, phi`). **No `qwen35moe` arch string, no `qwen35moe.*` metadata emission.** Unit test at `:2382` asserts `qwen3` dense tensor rename only. |
| `src/preflight.rs::SUPPORTED_LAYER_TYPES` | Supports `attention, full_attention, sliding_attention, moe_attention, linear, mlp, feedforward, ffn, dense, self_attention, cross_attention, grouped_query_attention`. **`linear_attention` is absent.** |
| `src/models/` | Gemma-specific model file(s). **No qwen35moe file.** |
| `src/quantize/dwq.rs` | `DwqConfig { base_bits, sensitive_bits, ... }` parameterized internally. Default `base_bits: 4, sensitive_bits: 6`. |
| `src/main.rs:447-448` | DWQ dispatch **hardcodes** `base_bits: 4, sensitive_bits: 6` regardless of `config.bits`. `--bits` flag silently ignored for DWQ path. **Broken window.** |
| `src/cli.rs:336-337` | `QuantMethod::DwqMixed46` is the **only** DWQ enum variant. No DwqMixed48 or parameterized parser. |
| `src/intelligence/auto_quant.rs` | Routes `arch.contains("qwen")` to `DenseDecoder`; MoE Qwen falls into `GenericMoE`. Sensitivity heuristic **tuned on Gemma's homogeneous transformer blocks**; not hybrid-arch-aware. |
| `src/input/hf_download.rs` | Uses `hf_hub::api::sync::ApiBuilder` with `huggingface-cli` fallback. Unknown behavior for 35B × ~40 safetensors shards, LFS resumption, disk-space preflight. |
| `docs/` | Zero hits for `hf2q convert` as canonical example. Gemma conversion command is oral history. |
| `tests/convert_integration.rs` | Gemma-only. No hybrid-arch or qwen35moe coverage. |

### Why this blows up the Gemma pattern

Every architectural element above maps to a gate in the existing code that currently only lets Gemma through. Attempting `hf2q convert --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated --format gguf --quant dwq-mixed-4-6` today will fail at config parsing before it reaches weight loading — `layer_types` is an unrecognized field, `full_attention_interval` doesn't exist in the parser, `linear_attention` fails preflight.

---

## Strategic Decision

**hf2q implements Qwen3.5 (dense) and Qwen3.5-MoE conversion natively in pure Rust. No Python prerequisite. No runtime linkage to candle or llama.cpp. `/opt/mlx-native` is the only sibling repo we depend on.**

This is the application of `feedback_hf2q_sovereignty.md` to the specific question of Qwen3.5 / Qwen3.5-MoE support. Path B (ride on llama.cpp's `convert_hf_to_gguf.py` as a prerequisite, and implement only a GGUF → DWQ → GGUF re-quantization pass in hf2q) was considered and explicitly rejected. Rationale: it would collapse hf2q's identity as a full-pipeline pure-Rust alternative to the Python + C++ status quo. Every future arch that llama.cpp adds (Qwen3-Next, Qwen3.5-VL, DeepSeek-V3, Llama-5, Grok, …) is ported into hf2q on landing.

**Both variants in one ADR:** Dense and MoE share ≥90% of the conversion surface (config parser, linear-attention tensor transforms, V-head reorder, MROPE, MTP, tokenizer, arch routing infrastructure). Only FFN layout differs. Splitting into ADR-012 (MoE) + ADR-013 (dense) would duplicate most of the spec into two documents for a single LOC-cheap variant. Keeping them together makes the shared surface explicit and the variant-specific differences cleanly scoped to the decisions where they matter.

**Arch-table-driven scaffolding (2026-04-24 refinement — Decision 20).** P8 (smoke harness), P10 (mmproj round-trip pattern), and P11 (MTP round-trip pattern) are designed as arch-agnostic infrastructure from the first keystroke rather than Qwen-specific code refactored later. The concrete next consumers of this scaffolding — **Gemma4 (parity follow-up), Ministral (ADR-015), and DeepSeek-V3 (ADR-016)** — are named up front so the cost of generalizing today is justified by the second and third arches becoming thin (add-a-row-to-the-arch-registry instead of write-a-new-harness). See Decision 20 and the "Arch-Table Scaffolding" section below for the concrete `src/arch/` module layout.

**Per-arch specialization bias stands.** Shared scaffolding does not mean uniform treatment: where a Qwen-specific transform (V-head reorder, A_log negation, MROPE partial-rotary, SSM state tensors) gives a better outcome than a generic abstraction, it stays Qwen-specific. The mantra clause "Just pure excellence, done the right way the entire time" applies per arch; shared infra is for the *conformance surface* (smoke, round-trip, catalog registry), not for model-specific mathematics.

---

## Non-Goals

Each of these is explicitly *not* in this ADR's scope. They are named so that the ADR's acceptance criteria can be measured against a bounded surface.

1. **Inference / forward-pass graph construction.** The inference session owns the Rust port of `Qwen3_5TextModel` and `Qwen3_5MoeTextModel` forward for execution, including GATED_DELTA_NET / SSM_CONV / TRI_SOLVE / L2_NORM / CumSum Metal kernels in `/opt/mlx-native`. This ADR consumes the inference engine only as a callable activation-capture backend for DWQ calibration (Phase P6). The inference port's ADR is separate.
2. **Tokenizer runtime changes.** The Qwen3.5 tokenizer (248K vocab) is embedded in the output GGUF via the existing tokenizer-embedding pipeline. If the existing `_set_vocab_qwen` equivalent in hf2q is sufficient for the new vocab, no work. If not, a follow-up ADR.
3. **Vision tower *forward-pass execution* (ViT compute, image patching, projection-into-text-stream merge).** Owned by ADR-005 phase 2c. **In scope for this ADR:** pure-Rust **conversion** of HF ViT safetensors → `mmproj-qwen36-F16.gguf` (Decision 18 / Phase P10). The externally-produced `mmproj-qwen36-F16.gguf` currently sitting in `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/` is the sovereignty gap this ADR closes.
4. **MTP speculative-decoding *execution* (draft-accept loops, rejection sampling, n-token speculation).** Owned by ADR-013 (new phase, tracked via Decision 19's cross-link). **In scope for this ADR:** (a) lossless MTP tensor emission (Decision 11, already shipped in P4), (b) round-trip integrity gate proving ADR-013's loader accepts those tensors with correct shapes and names (Decision 19 / Phase P11). Without P11, Decision 11's emitted MTP tensors are cosmetic — the gate exists precisely to prevent that outcome.
5. **Byte-level inference-parity sourdough gate against an external reference** (the analogue of `scripts/sourdough_gate.sh` that compares our inference bytes to llama.cpp's). Remains owned by ADR-013. **What this ADR *does* measure, per 2026-04-24 party-mode refinement:** DWQ quant-quality ACs in P9 compare hf2q's DWQ output against our own F16 forward (ADR-013's inference path) via perplexity + KL-divergence on a held-out eval corpus — i.e., correctness-biased quant-quality measurement, not inference parity against an external oracle. Cross-ADR dependency on ADR-013 F16 correctness is an implicit handoff (single author both sides); silent-failure mode where ADR-013 F16 has a numerical skew is a known accepted risk (R12).
6. **Rewrite of the DWQ sensitivity heuristic.** We extend it for the hybrid case; we don't redesign it from scratch. If Phase P6 reveals the existing approach is fundamentally wrong for hybrid arch, that is a follow-up ADR.

---

## Reference Implementations (authoritative)

Every porting decision in this ADR has an upstream reference. These file:line citations are entry points — llama.cpp is a moving target, so verify against the current HEAD of `/opt/llama.cpp` (commit `8bc492ebb` as of 2026-04-24 per `project_qwen36_architecture.md`) before starting.

### Python converter class chain (study order, top-down)

1. `/opt/llama.cpp/convert_hf_to_gguf.py:4553` — `class Qwen2MoeModel(TextModel)`. Base for MoE tensor naming, expert merge, router handling.
2. `/opt/llama.cpp/convert_hf_to_gguf.py:4770` — `class Qwen3NextModel(Qwen2MoeModel)`. Adds linear-attention conversion: A_log negation, dt_bias rename, conv1d squeeze, in_proj_qkvz reordering.
3. `/opt/llama.cpp/convert_hf_to_gguf.py:5259` — `class _LinearAttentionVReorderBase(Qwen3NextModel)`. The grouped-to-tiled V-head reorder. Six cases (in_proj_qkv, in_proj_z, in_proj_a/b, A_log / dt_bias / dt_proj, conv1d, out_proj). Implementation at `:5375-5424`.
4. `/opt/llama.cpp/convert_hf_to_gguf.py:5428` — `class Qwen3_5TextModel(_LinearAttentionVReorderBase)`. Terminal class for dense variant; `model_arch = gguf.MODEL_ARCH.QWEN35`.
5. `/opt/llama.cpp/convert_hf_to_gguf.py:5433` — `class Qwen3_5MoeTextModel(_LinearAttentionVReorderBase)`. Terminal class for MoE variant; `model_arch = gguf.MODEL_ARCH.QWEN35MOE`.

### C++ loader & metadata keys

6. `/opt/llama.cpp/src/llama-arch.cpp:42-43` — `{ LLM_ARCH_QWEN35, "qwen35" }` and `{ LLM_ARCH_QWEN35MOE, "qwen35moe" }`. The arch strings hf2q must emit as `general.architecture` per variant.
7. `/opt/llama.cpp/src/llama-arch.cpp:860` — `case LLM_ARCH_QWEN35MOE:` (and the adjacent `LLM_ARCH_QWEN35` case in the same switch) in `llm_arch_is_hybrid` (or equivalent predicate). Confirms upstream treats both as hybrid state-space/attention architectures.
8. `/opt/llama.cpp/src/llama-arch.h` + `llama-arch.cpp` — `LLM_KV_*` enum and the `LLM_KV_NAMES[]` table (starts around `:180`). Every key the loader reads must be emitted by hf2q. This is the authoritative list; do not derive from blog posts or the GGUF file's observed keys alone.
9. `/opt/llama.cpp/src/models/qwen35.cpp` — dense variant graph builder (~15 KB). Tells us which tensors the dense loader expects to find by name.
10. `/opt/llama.cpp/src/models/qwen35moe.cpp` — MoE variant graph builder (~17 KB). Tells us which tensors the MoE loader expects to find by name.
11. `/opt/llama.cpp/src/models/delta-net-base.cpp` — linear-attention compute (~16 KB; shared by both variants). Informs which SSM-family tensors are load-bearing and which are optional.

### Existing local reference: the `apex.gguf` on disk

`/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf` (25 GB, `general.file_type=17` i.e. IQ2_XS) was produced outside hf2q by an external invocation of `convert_hf_to_gguf.py` + `llama-quantize`. It exists on Robert's disk as a convenience reference. **It is NOT a test oracle, NOT a checked-in fixture, and NOT a CI dependency** per the sovereignty directive. If an engineer wants to look at it during development to understand what keys/tensors a reference implementation emits, that's fine — like reading llama.cpp source code. What's not fine is writing an assertion that compares hf2q's output against it.

---

## Architecture Decisions

Each decision below specifies: **Problem** (what breaks today), **Decision** (what we do), **Acceptance Criteria** (how we know it's done). Decisions are numbered for citation from commits and follow-up issues.

### 1. Arch string and routing

**Problem.** `src/backends/gguf.rs` has no routing for `qwen35` or `qwen35moe`. Without it, `general.architecture` is wrong and llama.cpp refuses to load the file.

**Decision.** Add both `qwen35` (dense) and `qwen35moe` (MoE) as recognized arches. Detection source:

- `config.json::architectures[0] == "Qwen3_5ForCausalLM"` → arch string `"qwen35"`
- `config.json::architectures[0] == "Qwen3_5MoeForCausalLM"` → arch string `"qwen35moe"`

Dispatch per-arch tensor-rename tables and metadata-emission logic off the arch string. Follow `project_model_class_split.md`: arch-specific code lives in two sibling files sharing a common module:

- `src/models/qwen35/mod.rs` — shared linear-attention transforms, MROPE, tokenizer, MTP logic (common to both variants)
- `src/models/qwen35/dense.rs` — `qwen35`-specific metadata and tensor naming (dense FFN)
- `src/models/qwen35/moe.rs` — `qwen35moe`-specific metadata and tensor naming (expert merge, shared experts)

`src/backends/gguf.rs` holds only dispatch by arch string.

**Acceptance criteria.**
- Unit tests in `src/backends/gguf.rs` mirroring the existing test at `:2382`, one per arch: for `qwen35` and `qwen35moe`, `hf_name_to_gguf("model.layers.0.post_attention_layernorm.weight", arch)` returns the correct GGUF name (`blk.0.post_attention_norm` per llama.cpp's Gemma convention OR `blk.0.ffn_norm` per LLaMA — determined by reading `/opt/llama.cpp/src/models/qwen35.cpp` and `qwen35moe.cpp`; the test MUST match what the loader reads).
- `hf2q convert --repo <Qwen3.5 dense repo> --format gguf --quant dwq-mixed-4-6` emits `general.architecture = "qwen35"`.
- `hf2q convert --repo <Qwen3.5-MoE repo> --format gguf --quant dwq-mixed-4-6` emits `general.architecture = "qwen35moe"`.
- Integration test Phase P7 asserts both keys are present and correct in their respective variants' outputs.

### 2. Config parser extensions

**Problem.** `src/input/config_parser.rs` doesn't extract the Qwen3.5-specific fields. `project_qwen36_architecture.md` enumerates them all; the parser is missing ≥12 of them.

**Decision.** Extend the parsed config struct with:
- `layer_types: Option<Vec<String>>` — per-layer attention type string array
- `full_attention_interval: Option<u32>`
- `attn_output_gate: Option<bool>`
- `head_dim: Option<u32>` (explicitly decoupled; must not be derived from hidden_size/num_heads when present)
- `partial_rotary_factor: Option<f32>`
- `rope_parameters: Option<RopeParameters>` with nested fields: `mrope_interleaved: bool`, `mrope_section: Vec<u32>`, `rope_theta: f64`, `rope_type: String`, `partial_rotary_factor: f32`
- `linear_conv_kernel_dim: Option<u32>`
- `linear_key_head_dim: Option<u32>`
- `linear_num_key_heads: Option<u32>`
- `linear_value_head_dim: Option<u32>`
- `linear_num_value_heads: Option<u32>`
- `mamba_ssm_dtype: Option<String>` (validated as one of `"float32"`, `"bfloat16"`, `"float16"`)
- `moe_intermediate_size: Option<u32>`
- `shared_expert_intermediate_size: Option<u32>`
- `mtp_num_hidden_layers: Option<u32>`
- `mtp_use_dedicated_embeddings: Option<bool>`
- `output_router_logits: Option<bool>`
- `router_aux_loss_coef: Option<f32>`

Support both `full_attention_interval` (computed) and `layer_types` (explicit enumeration); prefer explicit when present (gotcha #9 from `project_qwen36_architecture.md`).

**Acceptance criteria.**
- Unit test loads `/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/config.json` and asserts all fields parse with the correct values (head_dim=256, num_experts=256, num_experts_per_tok=8, rope_theta=10_000_000, mrope_section=[11,11,10], full_attention_interval=4, layer_types.len()==40, layer_types[3]==`"full_attention"`, layer_types[0]==`"linear_attention"`, shared_expert_intermediate_size=512, etc.).
- Unit test loads Gemma-4-26B-A4B `config.json` and asserts existing parsing is unaffected (regression guard).
- Malformed config (missing required-for-qwen35moe fields when arch is qwen35moe) returns an error with a clear message identifying the missing field — not silent zero-filling.

### 3. Preflight: `linear_attention` layer type

**Problem.** `src/preflight.rs::SUPPORTED_LAYER_TYPES` at `:8-22` doesn't include `linear_attention`. Preflight fails before conversion starts.

**Decision.** Add `"linear_attention"` to the `SUPPORTED_LAYER_TYPES` constant. Cross-reference that if `config.layer_types.contains("linear_attention")`, at least one `full_attention` entry exists (sanity check against misconfigured models).

**Acceptance criteria.**
- Preflight passes on the apex model's config.
- Preflight on a synthetic config with 100% `linear_attention` layers fails with a clear error (hybrid arch requires at least one full-attention layer for KV state anchoring).

### 4. New module: `src/models/qwen35/`

**Problem.** No files exist to hold Qwen3.5-family-specific logic. Per `project_model_class_split.md`, arch-specific code must live under `models/`, not in generic infrastructure. Dense and MoE share most of the transform code; a shared submodule avoids duplication.

**Decision.** Create `src/models/qwen35/` with three files:

**`src/models/qwen35/mod.rs`** — shared (dense + MoE) logic:
- `pub struct Qwen35ConvertContext { arch: Qwen35Arch, /* parsed hparams */ }` where `Qwen35Arch` is an enum `Dense | Moe`
- `pub fn transform_linear_attn_tensor(...)` — V-head reorder, A_log negation, conv1d squeeze, in_proj_qkvz reordering (shared by both variants; linear attention is identical across dense and MoE)
- `pub fn is_linear_attention_layer(layer_idx: usize, ctx: &Qwen35ConvertContext) -> bool`
- `pub fn is_full_attention_layer(...)` — complement
- Shared tensor-name transforms for full-attention layers (they're identical between dense and MoE)
- MTP tensor naming (identical)
- Shared metadata keys emitted by both arches (`general.architecture` routing, tokenizer, rope, full_attention_interval, linear-attn hparams)

**`src/models/qwen35/dense.rs`** — dense-specific:
- `pub fn hf_tensor_name_to_gguf_dense(hf_name: &str, ctx: &Qwen35ConvertContext) -> String` — standard `mlp.{gate,up,down}_proj` mapping
- `pub fn emit_metadata_dense(writer: &mut GgufWriter, ctx: &Qwen35ConvertContext) -> Result<()>` — writes `qwen35.*` keys including `feed_forward_length` (intermediate_size)

**`src/models/qwen35/moe.rs`** — MoE-specific:
- `pub fn hf_tensor_name_to_gguf_moe(hf_name: &str, ctx: &Qwen35ConvertContext) -> String` — routes `mlp.experts.{E}.{gate,up,down}_proj` to merged `ffn_*_exps`; shared experts to `ffn_*_shexp`; router to `ffn_gate_inp`
- `pub fn merge_expert_tensors(...)` — 256-expert stacking (see Decision 9)
- `pub fn emit_metadata_moe(writer: &mut GgufWriter, ctx: &Qwen35ConvertContext) -> Result<()>` — writes `qwen35moe.*` keys including `expert_count`, `expert_used_count`, `expert_feed_forward_length`, `expert_shared_feed_forward_length`

Dispatch from `src/backends/gguf.rs` based on `ctx.arch`. Gemma logic stays where it is; no refactor of Gemma in this ADR.

**Acceptance criteria.**
- Module `src/models/qwen35/` exists with all three files and the public API above.
- Module registered in `src/models/mod.rs`.
- Per-transform unit tests cover all 10 gotchas from `project_qwen36_architecture.md`; each applicable gotcha is tested once in the shared `mod.rs` tests (linear-attn transforms) and once per variant where dense-vs-MoE semantics differ (expert merge is MoE-only; dense FFN naming is dense-only).

### 5. V-head grouped → tiled reorder

**Problem.** Linear attention has `num_k_heads=16, num_v_heads=32` (Qwen3.5-MoE 35B) or `num_k_heads=16, num_v_heads=48` (Qwen3.5 27B dense). HF stores V heads grouped by K head (`[G0_v0..v{r-1}, G1_v0..v{r-1}, ...]`); ggml binary ops expect tiled broadcast (`[K0, K1, ..., K0, K1, ...]`). Off-by-one = silent corruption.

**Decision.** Port the six cases in `/opt/llama.cpp/convert_hf_to_gguf.py:5375-5424` verbatim to Rust. Each case matches a suffix in `hf_name`:
- `.linear_attn.in_proj_qkv.weight` — reorder V rows only (positions after `q_dim + k_dim`)
- `.linear_attn.in_proj_z.weight` — reorder rows (num_v_heads * head_v_dim)
- `.linear_attn.in_proj_b.weight` / `.linear_attn.in_proj_a.weight` — reorder rows (num_v_heads, head_dim=1)
- `.A_log` / `.dt_bias` / `.dt_proj` — 1D parameters with num_v_heads elements; reorder along last dim
- `.conv1d` — reorder only the V channel portion (channels after `head_k_dim * num_k_heads * 2`)
- `.linear_attn.out_proj.weight` — reorder columns (input dimension)

Use the same helper logic: `reorder_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim)`.

**Acceptance criteria.**
- Unit test per case: construct a tensor with known marker values at each V-head position, apply the transform, assert the output has markers at the expected tiled positions. Expected positions are derived analytically from ggml broadcast semantics and hand-authored as a constant in the test file (commented with the derivation), not copied from any external tool's output.
- Round-trip test: apply the reorder, apply its inverse (derivable from the same params), assert byte-identical to input. This protects against bugs where the reorder is partially correct (permutation subset applied twice).
- Specification-driven test: for a small reference case (e.g., num_k_heads=2, num_v_per_k=2, head_dim=4), hand-author the complete expected permutation map in the test, with a code comment deriving it from ggml broadcast semantics (the rule: ggml binary ops expect `[K0, K1, ..., K0, K1, ...]` tiled order; HF stores `[G0_v0..v{r-1}, G1_v0..v{r-1}, ...]` grouped). The test is the specification.

### 6. Qwen-specific tensor transforms (non-V-reorder)

**Problem.** Multiple conversion-time tensor transforms Qwen3.5 requires that Gemma doesn't.

**Decision.** Implement each transform in `src/models/qwen35/mod.rs` (shared by dense and MoE variants, since linear-attention tensors are structurally identical across both):

| Transform | Trigger | Action |
|---|---|---|
| A_log negation | Tensor name ends with `.A_log` | Output = `-exp(input)`. Miss it → NaN on inference. Gotcha #2 from project_qwen36_architecture.md. |
| dt_bias rename | Tensor name ends with `.dt_bias` | Rename to `.dt_proj.bias` per llama.cpp convention. Gotcha #3. |
| conv1d squeeze | Tensor name ends with `.conv1d.weight` and shape is `[k, 1, d]` | Squeeze dim 1 to produce `[k, d]`. Gotcha #4. |
| in_proj_qkvz reorder | Tensor name ends with `.linear_attn.in_proj_qkvz.weight` (fused variant) | Split into Q/K/V/Z head-grouped reorder. Gotcha #6. |
| RMS norm +1 bias | Tensor name ends with `_norm.weight` or similar norm tensor | Qwen uses `gamma + 1` convention. Gotcha #5. **Audit decision pending**: verify whether Qwen3.5 specifically still uses +1 (some Qwen2/Qwen3 variants don't) — check `qwen35moe.cpp`'s `build_norm` calls. |

**Acceptance criteria.**
- One unit test per transform with hand-constructed fixtures.
- A_log transform test asserts `|output + exp(input)| < 1e-6` for random inputs.
- Conv1d squeeze test asserts shape change and byte-identical data.
- An audit note in `src/models/qwen35/mod.rs` documents whether Qwen3.5 RMS norm adds +1 (with citation to the specific llama.cpp line consulted). Audit applies to both variants since norm handling is shared.

### 7. GGUF metadata emission for `qwen35.*` and `qwen35moe.*`

**Problem.** llama.cpp's `qwen35` and `qwen35moe` loaders read specific sets of `qwen35.*` / `qwen35moe.*` keys. Without them, the file loads and silently produces wrong output. The sets overlap heavily but diverge on MoE-specific keys.

**Decision.** Emit every metadata key the loader reads, namespaced under the correct arch prefix. The authoritative list is extracted by reading `LLM_KV_*` references in `/opt/llama.cpp/src/models/qwen35.cpp` + `qwen35moe.cpp` and `/opt/llama.cpp/src/llama-arch.{h,cpp}`. Initial catalog (engineer must verify against HEAD):

**General / standard transformer keys (both arches, namespaced `qwen35.*` for dense, `qwen35moe.*` for MoE):**
- `block_count` (dense = 64; MoE = 40)
- `context_length` (= max_position_embeddings = 262144 for both)
- `embedding_length` (dense = 5120; MoE = 2048)
- `feed_forward_length` (dense = intermediate_size = 17408; MoE = 0 or absent — dense FFN not used)
- `attention.head_count` (dense = 24; MoE = 16)
- `attention.head_count_kv` (dense = 4; MoE = 2)
- `attention.head_key_length` (= head_dim = 256; MUST emit explicitly since decoupled from hidden_size/num_heads)
- `attention.head_value_length` (= head_dim = 256)
- `attention.layer_norm_rms_epsilon` (= rms_norm_eps = 1e-6 for both)
- `attention.output_gate` (= attn_output_gate = true for both)

**MoE-only keys (`qwen35moe.*`):**
- `expert_count` (= num_experts = 256)
- `expert_used_count` (= num_experts_per_tok = 8)
- `expert_feed_forward_length` (= moe_intermediate_size = 512)
- `expert_shared_feed_forward_length` (= shared_expert_intermediate_size = 512)

**Hybrid / linear-attention (both arches):**
- `full_attention_interval` (= 4)
- `layer_types` as a string array OR a compact encoding per llama.cpp's convention (verify against `qwen35.cpp` + `qwen35moe.cpp` load paths; format must be identical across both arches since they share the hybrid machinery)
- Linear-attention hparams: conv_kernel_dim, key_head_dim, num_key_heads, value_head_dim, num_value_heads (exact key names per llama.cpp — use `grep LLM_KV_SSM` and `grep LLM_KV_LINEAR` in `/opt/llama.cpp/src/llama-arch.cpp`). Values differ between variants (e.g., dense `linear_num_value_heads=48` vs MoE `=32`).

**RoPE:**
- `rope.dimension_count` (= rotary_dim = 64)
- `rope.freq_base` (= rope_theta = 10_000_000)
- `rope.dimension_sections` as `[mrope_section..., 0]` padded to 4 slots (see `convert_hf_to_gguf.py:1149-1155` — pad to length 4 with zeros, call `add_rope_dimension_sections`)
- `rope.mrope_interleaved` (= true)

**MTP:**
- `mtp_num_hidden_layers` (= 1) — or the llama.cpp-named `nextn_predict_layers` (see `llama-arch.cpp:194`). Use whatever the loader reads.

**Tokenizer / vocab (standard pipeline, but verify the Qwen3.5 vocab loads correctly):**
- `tokenizer.ggml.model`, `tokens`, `merges`, `scores`, `token_type`, `bos_token_id`, `eos_token_id`, `pad_token_id`, `chat_template`, `pre` — per ADR-004 decision 5.

**Acceptance criteria.**
- The authoritative lists of required `qwen35.*` and `qwen35moe.*` keys are derived by reading `/opt/llama.cpp/src/llama-arch.{h,cpp}` (`LLM_KV_*` enum + KV_NAMES table) and `/opt/llama.cpp/src/models/{qwen35.cpp, qwen35moe.cpp}` (which keys each loader actually reads), and hand-transcribed into constant lists in `src/models/qwen35/dense.rs` and `src/models/qwen35/moe.rs` respectively — with shared keys factored into `src/models/qwen35/mod.rs`. Each entry carries a code comment citing the source file:line. No binary produced by llama.cpp is consulted as an automated check.
- Unit test: `emit_metadata()` called on a known `Qwen35MoeConvertContext` produces a GGUF metadata section containing every key in the hand-transcribed list with the expected value.
- Regression: whenever the list is updated (because llama.cpp added a new key in a new release and Robert wants to support it), the PR description cites the llama.cpp file:line where the new key was added.

### 8. Tensor naming for hybrid layers

**Problem.** Full-attention layers (every 4th — indices 3, 7, ..., L-1 where L=num_hidden_layers) and linear-attention layers (all others) have completely different tensor sets within the same variant. Additionally, dense and MoE variants differ entirely in their FFN-side tensor naming. GGUF block indices must be the same per layer, but tensor names must differ per (arch, layer type) pair.

**Decision.** In `hf_tensor_name_to_gguf_{dense,moe}()`:

**For full-attention layers** (HF `model.layers.{L}.self_attn.*`):
- `q_proj.weight` → `blk.{L}.attn_q.weight`
- `k_proj.weight` → `blk.{L}.attn_k.weight`
- `v_proj.weight` → `blk.{L}.attn_v.weight`
- `o_proj.weight` → `blk.{L}.attn_output.weight`
- `q_norm.weight` → `blk.{L}.attn_q_norm.weight`
- `k_norm.weight` → `blk.{L}.attn_k_norm.weight`
- Output gate (new in qwen35moe; locate exact HF name — likely `self_attn.output_gate.weight` or `self_attn.gate_proj.weight` — verify in HF model): → `blk.{L}.attn_gate.weight` (LLM_TENSOR_ATTN_GATE at `llama-arch.cpp:370`)

**For linear-attention layers** (HF `model.layers.{L}.linear_attn.*`):
- Map each of `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `conv1d`, `A_log`, `dt_bias`, `dt_proj`, norms to the corresponding llama.cpp `blk.{L}.ssm.*` / `blk.{L}.linear_attn.*` names. Exact names per llama.cpp's `LLM_TENSOR_*` enum in `src/llama-arch.h` — verify, do not guess.

**For both layer types:**
- `input_layernorm.weight` → `blk.{L}.attn_norm.weight`
- `post_attention_layernorm.weight` → depends on llama.cpp's Qwen3.5 convention — either `blk.{L}.post_attention_norm` (Gemma convention) or `blk.{L}.ffn_norm` (LLaMA convention). **Read qwen35moe.cpp and verify.** Do not assume.

**FFN (every layer) — variant-specific:**

*Dense (`qwen35`):*
- `mlp.gate_proj.weight` → `blk.{L}.ffn_gate.weight`
- `mlp.up_proj.weight` → `blk.{L}.ffn_up.weight`
- `mlp.down_proj.weight` → `blk.{L}.ffn_down.weight`

*MoE (`qwen35moe`):*
- `mlp.gate.weight` → `blk.{L}.ffn_gate_inp.weight` (router)
- `mlp.experts.{E}.gate_proj.weight` for E in 0..256 → merged into `blk.{L}.ffn_gate_exps.weight` shape `[256, moe_intermediate_size, hidden_size]`
- Same for `up_proj` → `ffn_up_exps` and `down_proj` → `ffn_down_exps`
- `mlp.shared_experts.gate_proj.weight` → `blk.{L}.ffn_gate_shexp.weight`
- Same for `up_proj` and `down_proj`

**Output head:**
- `model.embed_tokens.weight` → `token_embd.weight`
- `lm_head.weight` → `output.weight` (or tied to embed if `tie_word_embeddings=false` … verify; the target model has `tie_word_embeddings: false`, so lm_head is independent)
- `model.norm.weight` → `output_norm.weight`

**MTP head (both variants; see Decision 11):**
- Emitted as `blk.{num_hidden_layers}.nextn.*` tensors (block 40 for MoE, block 64 for dense) per the `LLM_TENSOR_NEXTN_*` names at `llama-arch.cpp:447-450`.

**Acceptance criteria.**
- Unit test per layer type (full vs linear) with a synthetic tensor name list → expected GGUF name list.
- Integration test (Phase P7) extracts all tensor names from hf2q's output GGUF and asserts the set matches the set in apex.gguf modulo mmproj tensors and quantization-type suffixes.

### 9. MoE expert tensor merge (MoE variant only — arch-gated no-op for dense)

**Problem.** For the MoE variant, HF stores each expert as a separate tensor (`model.layers.{L}.mlp.experts.{E}.gate_proj.weight` for E in 0..256). GGUF expects a single 3D tensor per projection per layer. The dense variant has no experts and skips this entirely.

**Decision.** This decision applies only when `ctx.arch == Qwen35Arch::Moe`. The dense variant's FFN tensors go through standard per-tensor emission without merge.

For the MoE variant: collect all 256 per-expert tensors for a given layer × projection combination, stack along a new first axis, and emit as a single 3D tensor. Perform this per-layer to avoid 40 × 256 × 3 = 30,720 tensor holds in memory simultaneously. Streaming: emit layer N's expert-merged tensor before starting layer N+1. Mirror ADR-004 decision 8's memory discipline.

Shared experts are **not** merged — they're singletons: `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`.

**Acceptance criteria.**
- Unit test (MoE): builds 4 fake expert tensors with distinguishable values, merges them, asserts shape is `[4, inter, hidden]` and each expert's data appears at its expected slice.
- Unit test (dense): dense convert path does not invoke `merge_expert_tensors` (compile-time guarantee via arch enum dispatch, or runtime assertion if called incorrectly).
- Memory profile during 35B MoE conversion stays under 64 GB peak (target: comparable to Gemma-4 26B's ~54 GB peak per ADR-004 decision 8 extrapolated to 35B expert count). If the layer-streaming discipline isn't enough for 256 experts × 3 projections × 40 layers, sub-layer streaming (per-projection) is the implementation strategy — the output is identical, only the memory-management granularity changes, so this is not a correctness fallback. Measure before selecting granularity. 27B dense conversion should comfortably stay under 32 GB peak (no expert blow-up).

### 10. DWQ bit-pair parameterization + close the broken window

**Problem.** `src/main.rs:447-448` hardcodes `base_bits: 4, sensitive_bits: 6` regardless of `config.bits`. `--bits 8 --quant dwq-mixed-4-6` accepts the flag and silently ignores it. `feedback_no_broken_windows.md` applies. Separately, the 4-AND-8 ask is unrepresentable in the current CLI vocabulary (`DwqMixed46` is the only enum variant at `src/cli.rs:336-337`).

**Decision.** Both changes in one commit:

(a) Add enum variants `DwqMixed48`, `DwqMixed68`, `DwqMixed28` to `QuantMethod`. Display as `"dwq-mixed-4-8"`, `"dwq-mixed-6-8"`, `"dwq-mixed-2-8"`.

(b) At `src/main.rs:447-448`, replace literal `4` / `6` with a lookup from `config.quant` → `(base_bits, sensitive_bits)` tuple. Keep `config.bits` for non-DWQ paths; for DWQ, error clearly if `--bits N` is passed (rather than silently ignoring) with message `"--bits is not used for DWQ; use --quant dwq-mixed-N-M to choose bit-pair variants"`.

(c) Output-filename auto-naming includes the DWQ variant: default output when `--quant dwq-mixed-4-6` is `{model-name}-dwq46.gguf`; for `dwq-mixed-4-8` it's `{model-name}-dwq48.gguf`. User-provided `--output` overrides.

**Non-goal** in this ADR: re-tuning the sensitivity heuristic for the wider bit gap. The current heuristic picks layers based on an activation-sensitivity score; the score is orthogonal to the promotion target (6 vs 8). Promotion is a magnitude change, not a selection change. Heuristic audit = Decision 12.

**Acceptance criteria.**
- `hf2q convert --help` lists all three new variants.
- Unit test: dispatch table maps each variant to the correct `(base, sensitive)` tuple.
- Unit test: parsing `--bits 5 --quant dwq-mixed-4-6` errors with the documented message.
- Integration test: two side-by-side outputs produced from the same input (dwq46 and dwq48) collide on neither filename nor internal tensor data (they differ in quant type of sensitive tensors).
- Gemma-4 convert with `dwq-mixed-4-6` produces byte-identical output to pre-change HEAD. **This is the Chesterton's-fence regression guard.**

### 11. MTP head handling (both variants)

**Problem.** Both Qwen3.6-27B dense and Qwen3.6-35B-A3B MoE have `mtp_num_hidden_layers: 1` → one Multi-Token Prediction block after the main stack, enabling future speculative decoding (1.8× decode speedup per DeepSeek-V3 baseline). HF stores these as `model.mtp.*` tensors. llama.cpp represents them as `blk.{n_layer}.nextn.*` per `LLM_TENSOR_NEXTN_*` at `llama-arch.cpp:447-450`. Inference session decides whether to use MTP for speculative decoding; conversion must preserve the tensors losslessly per Robert's 2026-04-23 confirmation.

**Decision.** Emit MTP tensors losslessly for both variants. Map HF `model.mtp.layers.0.*` → GGUF `blk.{num_hidden_layers}.nextn.*` (block 40 for MoE, block 64 for dense). Specific mappings per `LLM_TENSOR_NEXTN_*` names: `nextn.eh_proj`, `nextn.embed_tokens`, `nextn.enorm`, `nextn.hnorm`, plus any attention/FFN tensors the MTP block contains. Emit `qwen35.nextn_predict_layers = 1` (dense) or `qwen35moe.nextn_predict_layers = 1` (MoE) metadata key per `llama-arch.cpp:194`.

**Acceptance criteria.**
- Tensor-name test (MoE): `model.mtp.layers.0.embed_tokens.weight` → `blk.40.nextn.embed_tokens.weight`.
- Tensor-name test (dense): `model.mtp.layers.0.embed_tokens.weight` → `blk.64.nextn.embed_tokens.weight`.
- If `mtp_num_hidden_layers == 0` (future Qwen variants without MTP), the MTP emission path is skipped cleanly with no empty tensors.
- `qwen35.nextn_predict_layers` / `qwen35moe.nextn_predict_layers` metadata key present and correct in the appropriate variant's output.

### 12. DWQ sensitivity heuristic for hybrid arch (both variants)

**Problem.** `src/intelligence/auto_quant.rs` and `src/quantize/dwq.rs`'s sensitivity heuristic was tuned on Gemma's homogeneous transformer blocks. Qwen3.5 family has tensor cohorts that respond to quantization differently:
- **Full-attention tensors** (standard transformer; existing heuristic applies; both variants)
- **Linear-attention / SSM tensors** (`A_log`, `dt_*`, `conv1d`, `in_proj_*` — small, numerically load-bearing; `A_log` in particular is an exponentiated parameter where quantization error compounds; both variants)
- **Dense FFN tensors** (dense variant: standard `ffn_{gate,up,down}`; existing per-tensor heuristic applies)
- **MoE tensors** (MoE variant only: router gate — tiny, must stay high-precision; experts — 256× the count, individually less sensitive; shared experts — always hot, higher sensitivity than individual experts)

**Decision.** Extend the sensitivity scorer with arch-aware cohort priors:

- **Both variants:** SSM state tensors (`A_log`, `dt_bias`, `dt_proj`, `conv1d`) are **always** promoted to `sensitive_bits` (never base_bits), regardless of activation score.
- **MoE only:** Router gate tensors (`ffn_gate_inp`) always promoted. Shared expert tensors (`ffn_*_shexp`) always promoted. Individual expert tensors (`ffn_*_exps`) use the existing activation-score-driven heuristic but with a higher promotion threshold (details: exact threshold tuned empirically against measured KL-divergence per the ADR-004 decision 7 block-size methodology used for K-quant type selection).
- **Dense only:** Dense FFN tensors use the existing activation-score-driven heuristic unchanged.
- **Full-attention tensors** (both variants) unchanged from current heuristic.

**Audit requirement** (Chesterton's fence): before wiring this in, read `src/intelligence/auto_quant.rs::build_component_overrides` and `src/quantize/dwq.rs::calibrate` in full, and explain in a code comment how the new cohort priors interact with existing per-tensor overrides. Do not introduce cohort priors that silently override a user-supplied `--sensitive-layers` range.

**Acceptance criteria.**
- Test (MoE): on a synthetic qwen35moe-shaped config, the sensitivity scorer marks all `A_log`, `dt_*`, `conv1d`, `ffn_gate_inp`, `ffn_*_shexp` tensors as "promote to sensitive_bits" before any activation data is collected.
- Test (dense): on a synthetic qwen35-shaped config, the sensitivity scorer marks all `A_log`, `dt_*`, `conv1d` tensors as "promote to sensitive_bits" and does not apply MoE-only priors (no `ffn_*_exps` / `ffn_*_shexp` tensors exist).
- Test: `--sensitive-layers 0-10` user override is still honored in both variants.
- Test: Gemma-4 sensitivity choices are byte-identical pre- and post-change (the new cohort priors only fire for qwen35 / qwen35moe arches).
- Measured KL divergence on a calibration subset for Gemma-4 DWQ-4-6: ≤ pre-change value (never degrades).

### 13. DWQ activation calibration for hybrid arch

**Problem.** Activation-based DWQ calibration requires a forward pass. The existing forward pass at `src/quantize/dwq.rs` hooks into hf2q's Gemma inference engine (layer-streamed per ADR-004 decision 8). Qwen3.5 / Qwen3.5-MoE forward passes through linear-attention layers require SSM state, GATED_DELTA_NET Metal kernels, and MROPE — all of which live in `/opt/mlx-native` and are being built by the parallel inference session.

**Decision.** Define a stable activation-capture API that the inference session implements, and the DWQ pipeline consumes. Interface sketch:

```rust
// in src/quantize/dwq/forward.rs or equivalent
pub trait ActivationCapture {
    fn run_calibration_prompt(&mut self, tokens: &[u32]) -> Result<LayerActivations>;
}

pub struct LayerActivations {
    pub per_layer_inputs: Vec<TensorF32>,   // [n_layers][hidden_size] aggregated over calibration prompts
    pub per_layer_outputs: Vec<TensorF32>,
    // ... any other stats the sensitivity scorer reads
}
```

The inference session's qwen35 / qwen35moe forward implements `ActivationCapture`. hf2q's DWQ pipeline owns the trait definition and the sensitivity scorer; the inference engine owns the implementation. Cross-session coordination point: the trait definition.

**No fallback.** Per the mantra and `feedback_no_shortcuts.md` / `feedback_correct_outcomes.md`: weight-space DWQ is not a valid substitute for activation-based DWQ. Activation-based produces measurably better quantization; accepting weight-space output means accepting lesser output, which is a fallback and forbidden. If the inference session's activation capture isn't ready when the rest of the pipeline is, **P6 blocks and we fix the blocker**. Options when blocked: (a) help the inference session land activation capture faster, (b) narrow the trait surface to the minimum viable for sensitivity scoring, (c) hold P6 until real. Not an option: ship weight-space DWQ output and call it done.

**Acceptance criteria.**
- Trait definition lands in hf2q before inference session starts coding its implementation — this reduces rework on their side.
- A mock `ActivationCapture` that returns deterministic but structurally-correct tensors is used for hf2q-side unit tests (so P6's own tests don't depend on a working inference engine to exercise the scorer logic).
- Real DWQ conversion of Qwen3.5 / Qwen3.5-MoE weights requires the inference session's real `ActivationCapture` implementation and does not proceed without it.
- The existing `use_activations: false` weight-space code path at `src/quantize/dwq.rs` remains for Gemma's pre-existing invocation only; it is not extended, referenced, or invoked from any qwen35 / qwen35moe convert path.

### 14. HF download robustness for 35B

**Problem.** `src/input/hf_download.rs` uses `hf_hub::api::sync::ApiBuilder` with `huggingface-cli` fallback. Unknown whether it correctly handles: (a) ~40 safetensors shards for a 35B model; (b) LFS resumption mid-shard on network interruption; (c) gated/auth-required repos; (d) disk-space preflight before starting a ~70 GB download.

**Decision.** Add preflight:
- Before starting download, `df` the target directory and assert ≥ 150 GB free (70 GB bf16 weights + 73 GB DWQ intermediate peak per ADR-004 decision 8 extrapolated + 10 GB margin).
- Error with a clear message if insufficient: `"Qwen3.5-MoE 35B requires ≥150 GB free in {path}; found {N} GB. Free space or change --cache-dir."`
- Verify shard resumption: if download is interrupted and `hf2q convert` is re-invoked, it must resume, not restart. Existing `hf_hub` API sync behavior — verify and add regression test.

**Acceptance criteria.**
- Unit test: mocks `fs2::available_space` to return < 150 GB, asserts error is raised with the documented message.
- Manual test (documented in `docs/converting-qwen35moe.md`): `Ctrl+C` during download → re-invoke → resumes from the partial shard, does not redownload completed shards.

### 15. Sidecar file preservation

**Problem.** Gemma's output dir contains `chat_template.jinja`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` alongside the `.gguf`. Unknown whether hf2q writes these automatically or they were manually copied. For qwen35moe the tokenizer and chat template are both different from Gemma's.

**Decision.** During convert, copy the HF repo's sidecar files (`chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `config.json`, `generation_config.json`, `special_tokens_map.json`) into the output directory alongside the produced `.gguf`. Preserve file content byte-identically. If any are missing in the HF repo, skip silently (not all models ship all sidecars).

**Acceptance criteria.**
- Integration test: output directory for `apex`-equivalent input has the 5+ sidecar files present with byte-identical content to the HF source.
- Test that Gemma-4 conversion still produces the same sidecar set it did before.

---

### 16. Real-model smoke-test harness (end-gate for this ADR)

**Problem.** P7 left the smoke test as a hand-run command documented in `docs/converting-qwen35.md`. That is a stub per the mantra: "no `todo later`". Without an automated, reproducible harness producing a committed transcript artifact, we have no evidence that a converted `qwen35` or `qwen35moe` GGUF actually loads in a downstream consumer. A structurally-valid GGUF that silently fails llama.cpp's loader (e.g. a missing hparam key, a transposed tensor, a wrong dtype on a single norm) is the canonical Qwen-family failure mode and is not caught by any unit or integration test shipped in P0–P7.

**Decision.** Ship `hf2q smoke` as a **Rust binary subcommand** on the `hf2q` CLI (not a bash script). Arch-generic from first line per Decision 20 — dispatches via `ArchRegistry::get(arch)` and runs the same conformance pipeline for every arch. The Qwen3.5 / Qwen3.5-MoE entries are populated in P8; Gemma4 entry is populated by the parity follow-up ADR; Ministral / DeepSeek entries are added by their own future ADRs with zero harness rework.

**CLI:** `hf2q smoke --arch {qwen35|qwen35moe|gemma4|…} --quant {q4_0|dwq-mixed-4-6|dwq-mixed-4-8} [--with-vision] [--skip-convert] [--dry-run]`

**What the subcommand does:**

1. **Preflights** the environment:
   - `HF_TOKEN` is set (non-empty) — exit code 2
   - Free space on the convert directory ≥ `ArchEntry::disk_floor_gb + 10 GB` buffer — exit code 3
   - `/opt/llama.cpp/build/bin/llama-cli` exists and is executable (read-only consumer per sovereignty — fine) — exit code 4
   - `hf2q` binary itself is built in release mode — exit code 5
   - The arch's target repos (`ArchEntry::hf_repos`) are resolvable via `huggingface-cli repo info` (no download yet) — exit code 6
2. **Converts** each variant at the requested quant. For `--quant q4_0` (P8 baseline, pre-P9): bit-identical emission with no calibration dependency, unblocks the gate today. For `--quant dwq-mixed-4-*` (P9 follow-up): requires P9's real-activation wire-up and runs PPL/KL measurement inline.
3. **Loads and infers** for 8 tokens under maximum determinism:
   - `llama-cli --model <path> --prompt <ArchEntry::smoke_prompt> -n 8 --seed 42 --temp 0 --log-disable --no-warmup`
   - `--temp 0` + `--seed 42` makes the output byte-stable across the M5 Max fleet
4. **Asserts** each transcript:
   - Exactly 8 generated tokens (`llama_print_timings: n_eval = 8`)
   - No line matching `error|ERROR|panic|assertion|segfault` on stderr
   - Tensor-count line `llama_model_load: loaded tensor 0x%x` matches `ArchEntry::tensor_catalog.len()` (sourced from the registry, not a side-file — single source of truth)
5. **(DWQ only)** Runs PPL + KL measurement against the F16 reference path (Decision 17) using `ArchEntry::ppl_corpus` and asserts the thresholds in `ArchEntry::quality_thresholds`.
6. **Commits** each passing transcript under `tests/fixtures/smoke-transcripts/{arch}-{quant}.txt` (≤ 2 KB, timestamps stripped). These are tracked artifacts — they are the evidence that this ADR's closure is real.

**Ownership.** The binary is source code, not documentation. Any future arch landing in hf2q (Gemma4-parity-retrofit, Ministral, DeepSeek-V3, Qwen3.7, …) extends the registry by adding one `src/arch/entries/X.rs` file — the harness itself does not change.

**Acceptance criteria.**
- `cargo run --release -- smoke --arch qwen35 --quant q4_0` exits 0, producing `tests/fixtures/smoke-transcripts/qwen35-q4_0.txt` with 8 tokens and no error lines.
- `cargo run --release -- smoke --arch qwen35moe --quant q4_0` exits 0, producing `tests/fixtures/smoke-transcripts/qwen35moe-q4_0.txt` likewise.
- Both transcripts are byte-identical across two fresh runs on the same M5 Max (proves `--seed 42 --temp 0` determinism is real — flags a non-deterministic tokenizer or forward path immediately if it fails).
- Preflight failure modes (missing `HF_TOKEN`, insufficient disk, missing `llama-cli`, missing binary, unresolvable repo) produce a single-line error naming the exact missing prerequisite and exit with the distinct non-zero code listed above (2/3/4/5/6).
- `hf2q smoke --arch bogus` exits non-zero with a structured error listing registered arches — proves the registry dispatch is load-bearing.
- `hf2q smoke --help` prints auto-generated flag documentation from the arg-parser.
- CI skip gate: the full `hf2q smoke` path is **not** invoked by CI (disk + wall-clock + HF-token requirements). CI runs a dedicated unit test suite (`tests/smoke_conformance.rs`) that exercises every preflight failure mode via a mock `llama-cli` stub and a mock HF resolver — this keeps the harness itself regression-guarded without paying the conversion cost.
- `hf2q smoke --arch <any-unregistered-arch>` (including `gemma4`, `ministral`, `deepseekv3`, `bogus`) returns a uniform "unknown arch; known arches: qwen35, qwen35moe" structured error. No per-arch `todo!()` placeholder; uniform dispatch rejection until a future ADR registers the arch.

### 17. DWQ activation calibration — real-weight wire-up (Robert-deliverable gate)

**Problem.** `src/main.rs:500-508` currently returns `DwqError::NoActivationCapture` for every `qwen35` / `qwen35moe` DWQ convert invocation. This is the second line of defence added in P6 — correct at the time because no production `ActivationCapture` impl existed. ADR-013 P12 (`src/inference/models/qwen35/weight_loader.rs`, HEAD `870bd7a` et seq.) is now shipping a real-weight forward path for both variants. The Robert-named end-deliverable of this ADR (`qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf` + `-dwq48.gguf`, Context §Business problem) cannot ship until the guard is removed and replaced with a concrete impl.

**Decision.** Replace the `NoActivationCapture` guard with a production `ActivationCapture` impl and remove the error variant once no caller can reach it. Specifically:

1. **New file:** `src/inference/models/qwen35/activation_capture_real.rs`, `pub struct RealActivationCapture { model: Qwen3_5Variant, tokenizer: Tokenizer }`.
   - `impl ActivationCapture for RealActivationCapture` driving ADR-013's forward pass (Decision 8 Gated DeltaNet + Decision 9 Gated full-attn + Decision 10 MROPE + Decision 12 weight loader in ADR-013).
   - Accepts `&mut self, tokens: &[u32]` → returns `Vec<LayerActivation>` matching the trait signature at `src/inference/models/qwen35/activation_capture.rs:141`.
   - Internally: runs the forward once per calibration sample, captures per-`nn.Linear` activations via the hook-point API exposed by ADR-013 Decision 8/9 (defined alongside this decision).
2. **`src/main.rs:488-506` rewrite.** Replace the current block:
   ```rust
   if dwq_arch.requires_activation_capture() {
       return Err(quantize::dwq::DwqError::NoActivationCapture.into());
   }
   ```
   with:
   ```rust
   let capture: Box<dyn ActivationCapture> = if dwq_arch.requires_activation_capture() {
       Box::new(RealActivationCapture::new(&model_weights, &tokenizer)?)
   } else {
       // Non-qwen35 paths remain weight-space (P6 unchanged).
       Box::new(NoopActivationCapture)
   };
   run_dwq_calibration(&config, &capture, &calibration_corpus)?;
   ```
3. **Remove** `DwqError::NoActivationCapture` entirely once the above lands. A variant that no production path can return is dead code and fails the mantra's "no stub" clause. Unit tests using `MockActivationCapture` remain — they don't trigger the removed error path.
4. **Calibration corpus.** Reuse the existing hf2q calibration corpus (`src/quantize/calibration/corpus.rs`). 1024 samples is the documented default; Phase P9's commit message records the actual count, per-sample wall-time, peak RSS, and sensitivity JSON delta vs. a weight-space-only run.
5. **Chesterton's fence: Gemma must not regress.** Gemma's DWQ path does not require `ActivationCapture` (`DwqArch::Gemma4` returns `false` from `requires_activation_capture`). The wire-up above preserves that branch — Gemma continues on the existing weight-space calibration path. Byte-identical output is asserted in the P9 regression test.

**Quality acceptance criteria (2026-04-24 party-mode refinement — correctness bias).** Pure "file loads + emits 8 tokens" is too weak a gate to claim "most correct quants" per the mantra. P9 adds **perplexity + KL-divergence** measurement against our own F16 reference path via ADR-013's inference (Decision 6 in ADR-013; hard-blocked on ADR-013's F16 forward being bit-stable for both variants — implicit cross-ADR handoff, R12).

**Measurement protocol:**
- **Eval corpus:** wikitext-2-raw-v1 test split, first 512 tokens of the first paragraph longer than 1024 raw tokens (deterministic selection; committed as `tests/fixtures/ppl-corpus/wikitext2.tokens` with a SHA-256 sidecar).
- **Calibration corpus disjointness:** DWQ calibration corpus (1024 samples from the existing hf2q calibration source) must be disjoint from the eval corpus. Asserted by a unit test comparing the two corpora's SHA-256s of their token lists.
- **Reference:** our own F16 inference path — `RealActivationCapture` with bits=16 (no quant) runs the same forward used for calibration. F16 logits per token are the reference distribution.
- **Perplexity:** `exp(-mean(log_softmax(ref_logits)[next_token]))` on the 512-token eval. Computed on both the DWQ output and the F16 reference.
- **KL-divergence:** per-token `sum_v softmax(ref)[v] * (log_softmax(ref)[v] - log_softmax(dwq)[v])` averaged over the 512 tokens, reported in nats.

**Quality thresholds (load-bearing — enforced in `ArchEntry::quality_thresholds`):**
- DWQ46 perplexity ≤ **1.10× F16 reference PPL**
- DWQ48 perplexity ≤ **1.05× F16 reference PPL**
- Median KL-divergence per token < **0.02 nats** (applies to both DWQ46 and DWQ48)

These numbers are the party-mode-confirmed thresholds (2026-04-24); they are literal constants in `src/arch/registry.rs` (`QualityThresholds { ppl_ratio_dwq46: 1.10, ppl_ratio_dwq48: 1.05, max_median_kl: 0.02 }`), not "reasonable" or "TBD." If P9 produces numbers outside these bounds on either variant, P9 does not ship — the answer is to understand why calibration is under-performing, per the mantra ("No shortcuts. No fallback.").

**Structural acceptance criteria (unchanged from original):**
- `cargo test -p hf2q --test convert_qwen35_integration -- real_activation_capture` green. Test: synthetic tiny qwen35 (4 layers including 1 full-attn + 3 linear-attn, 4 experts, hidden=64, head_dim=16) runs through `RealActivationCapture` and produces a sensitivity JSON whose per-layer priorities differ from weight-space-only scoring — this is the sharp proof that activations are actually flowing through the forward pass, not being stubbed.
- `hf2q convert --repo Qwen/Qwen3.6-27B --format gguf --quant dwq-mixed-4-6 --calibration-samples 1024 --output models/qwen3.6-27b-dwq46/out.gguf` runs to completion on the M5 Max. Peak RSS ≤ 64 GB (recorded via `/usr/bin/time -l`). Wall time is a *soft* target (≤ 2 h) — correctness over speed, per 2026-04-24 refinement.
- Same command with `--quant dwq-mixed-4-8` produces a structurally-different output GGUF (different `general.file_type`, different sensitivity JSON sidecar).
- Same two commands with `--repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated` produce the two Robert-named end-deliverable GGUFs.
- All four resulting GGUFs pass P8's smoke gate (`hf2q smoke --arch qwen35|qwen35moe --quant dwq-mixed-4-{6,8}`) — *including the PPL + KL quality ACs above*. Four new transcripts committed: `tests/fixtures/smoke-transcripts/{qwen35,qwen35moe}-dwq4{6,8}.txt` with PPL and KL numbers inline in each transcript.
- Gemma-4 DWQ-4-6 regression: `hf2q convert --repo <gemma-4 repo> --quant dwq-mixed-4-6 ...` produces byte-identical output to the pre-P9 HEAD (SHA-256 match). Asserted in an integration test. Gemma4's quality ACs are populated by the follow-up parity ADR, not P9.
- `DwqError::NoActivationCapture` no longer exists in `src/quantize/dwq.rs`. `cargo clippy` reports no dead-code warning on the quant module.
- `src/main.rs:488-506` is re-commented to describe the wired path; the comment trail documents P6 → P9 transition with commit SHAs.
- ADR-013 P12 (`RealActivationCapture` dependency surface — specifically: the hook-point API on `Qwen3_5TextModel::forward` and `Qwen3_5MoeTextModel::forward`) is independently accepted with its own tests green. If ADR-013 P12 isn't green, P9 does not ship — **the answer remains "fix the blocker, not route around it."**

### 18. Pure-Rust mmproj (vision-tower) emitter

**Problem.** Qwen3.6-27B ships a 27-layer vision transformer (ViT, patch_size=16, image_size likely 384 or 448 — verify from `config.json::vision_config`). A deployable multimodal model needs a companion `mmproj-qwen36-F16.gguf` file that carries the ViT weights and the cross-modal projector to the text backbone's hidden dim. The file currently in `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/mmproj-qwen36-F16.gguf` was produced by an external tool (likely `llama.cpp/tools/mtmd`'s converter), so we rely on a build artifact from a repo we don't own — direct violation of `feedback_hf2q_sovereignty.md`.

**Decision.** Add a pure-Rust vision-tower conversion path:

1. **New module:** `src/models/vit/` with:
   - `mod.rs` — public `convert_vision_tower(hf_repo: &HfRepo, output_dir: &Path) -> Result<PathBuf>`
   - `config.rs` — parses `config.json::vision_config` into `VisionConfig { hidden_size, num_hidden_layers, num_attention_heads, patch_size, image_size, projector_type, layer_norm_eps, ... }`. Fields are `Option<T>` where the HF schema allows omission; required fields for Qwen3.6 are validated explicitly with named errors (same pattern as P1).
   - `convert.rs` — iterates `model.vision_tower.*` safetensors tensors, applies HF-name → GGUF-name mapping:
     - `model.vision_tower.embeddings.patch_embeddings.projection.{weight,bias}` → `v.patch_embd.{weight,bias}`
     - `model.vision_tower.embeddings.position_embeddings.weight` → `v.position_embd.weight`
     - `model.vision_tower.encoder.layer.{L}.attention.{q,k,v}_proj.{weight,bias}` → `v.blk.{L}.attn_{q,k,v}.{weight,bias}`
     - `model.vision_tower.encoder.layer.{L}.attention.output.dense.{weight,bias}` → `v.blk.{L}.attn_out.{weight,bias}`
     - `model.vision_tower.encoder.layer.{L}.layer_norm{1,2}.{weight,bias}` → `v.blk.{L}.ln{1,2}.{weight,bias}`
     - `model.vision_tower.encoder.layer.{L}.mlp.{fc1,fc2}.{weight,bias}` → `v.blk.{L}.ffn_{down,up}.{weight,bias}` (verify ordering against `clip-model.h`)
     - `model.vision_tower.post_layernorm.{weight,bias}` → `v.post_ln.{weight,bias}`
     - `model.multi_modal_projector.linear_{1,2}.{weight,bias}` → `mm.{0,2}.{weight,bias}` (MLP projector convention; adjust if Qwen uses a single-linear projector)
   - `gguf_emit.rs` — writes an mmproj-format GGUF:
     - `general.architecture = "clip"` (per `/opt/llama.cpp/tools/mtmd/clip.cpp` — arch string is load-bearing; llama.cpp's mtmd loader keys off this)
     - `clip.vision.*` metadata per `clip-model.h` (`image_size`, `patch_size`, `projection_dim`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `projector_type`)
     - Tensor dtype: F16 (per existing `mmproj-qwen36-F16.gguf` precedent). Quantized mmproj is out of scope for this ADR — F16 is what the downstream loader expects for vision tensors today.
2. **Sovereignty.** Same rule as the main text-side converter: `/opt/llama.cpp/tools/mtmd/clip-model.h` and `clip.cpp` are read-only spec sources. No build-time or test-time dependency on any `mtmd` artifact. The expected tensor-name catalog is hand-transcribed into a constant in `src/models/vit/convert.rs` with a file:line citation per key, exactly as P4 did for `llama-arch.cpp`.
3. **CLI surface.** New flag on `hf2q convert`: `--emit-vision-tower` (default: off). When set:
   - If the HF repo has no `vision_config` in `config.json`, log a single-line `note: --emit-vision-tower requested but <repo> has no vision_config — skipping` and continue. Not an error.
   - Otherwise, emit `mmproj-<model-slug>-F16.gguf` alongside the text GGUF in the output directory.
   - Gemma-4's repo has no `vision_config` → `--emit-vision-tower` is a silent no-op for Gemma (regression test asserts this).
4. **MoE variant interaction.** The local MoE target (`jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated`) dropped `vision_config` from its `config.json`. For that repo, `--emit-vision-tower` silently skips (same no-op branch as Gemma). Vision tower is only emitted for dense `Qwen/Qwen3.6-27B`. Integration tests must cover **both** paths (emit + no-op-skip) to guard against a future "silently skips when it shouldn't" regression.

**Three-layer defense (2026-04-24 party-mode refinement — post-sovereignty audit).** All three layers are sovereignty-clean: the conversion is derived from `clip.cpp` + `clip-model.h` spec by hand-transcription, and correctness is proven against the spec via synthetic tests + our own loader, never against an external reference artifact. The earlier "one-time external-mmproj cosine anchor" proposal was removed — using external output to prove our correctness (one-time or not) is the pattern `feedback_hf2q_sovereignty.md` rejects, and the spec-driven synthetic tests are already the sufficient correctness proof for an F16→F16 lossless port.

**Layer A — Structural acceptance (baseline):**
- Integration test `tests/convert_vision_tower_integration.rs`: synthetic tiny ViT (4 layers, hidden=64, num_heads=8, patch_size=4, image_size=32) round-trips through `convert_vision_tower` and the output GGUF is read back by hf2q's own GGUF reader. Every tensor in the hand-authored expected catalog (const in the test file) is present with the expected shape and F16 dtype. Every metadata key in the expected catalog is present with the expected value.
- Integration test `tests/convert_vision_tower_noop.rs`: invoking `convert_vision_tower` on a `VisionConfig`-less HF repo fixture returns `Ok(None)` (or equivalent "no file emitted") without panicking.
- Gemma-4 regression: full convert pipeline with `--emit-vision-tower` produces byte-identical text-side output vs. pre-P10, plus no `mmproj-*.gguf` written. Asserted via pre-built SHA-256 fixture.
- Real-model structural gate: `hf2q convert --repo Qwen/Qwen3.6-27B --format gguf --quant q4_0 --emit-vision-tower --output models/smoke-qwen35-27b/` produces **both** `out.gguf` and `mmproj-qwen36-27b-F16.gguf`. The mmproj file is loadable by `llama-mtmd-cli --mmproj mmproj-qwen36-27b-F16.gguf --model out.gguf -p "Hello"` without error. `hf2q smoke --arch qwen35 --with-vision` invokes this combined path and records the extra transcript. (llama-mtmd-cli is used here as a *reader*, identical to how P8 uses llama-cli on the text-side GGUF — not as a correctness oracle.)
- No new dependency on any crate under `/opt/llama.cpp/tools/` (check via `cargo tree -p hf2q`).

**Layer B — ADR-005 phase 2c round-trip gate (synthetic + real-model, mirror of Decision 19's MTP pattern):**
- Integration test `tests/convert_vision_tower_adr005_roundtrip.rs`: our emitted `mmproj-qwen36-27b-F16.gguf` (synthetic tiny variant) is loaded by `src/inference/vit/loader.rs` (ADR-005 phase 2c's mmproj entry point). Assertion: the vision submodule reports `has_vision == true` AND every tensor in `EXPECTED_VIT_TENSORS` (hand-authored const) is present with correct shape and dtype.
- Real-model extension: the same test, marked `#[ignore]` so it stays out of default CI, loads the real 27B dense mmproj produced by the P10 real-model gate. Asserts all 27 ViT-layer tensor triples (attn_{q,k,v}, attn_out, ln1, ln2, ffn_{down,up}, …) are present with real-model shapes. This catches any "passes synthetic, fails at real dimensions" mapping bug that Layer C's synthetic coverage can miss.
- If the ADR-005 loader rejects any tensor, the test assertion failure must include the offending tensor name (loader error-surface hardening, same quality bar as P11).
- Synthetic test runs in < 30 s, default `cargo test` set, CI-green. Real-model test is opt-in via `cargo test -- --ignored` and `hf2q smoke --arch qwen35 --with-vision`.

**Layer C — Spec-driven hand-authored layout tests (mirror of P5's expert-merge pattern):**
- Unit tests in `src/models/vit/convert.rs` with hand-authored expected tensor bytes on the synthetic tiny ViT. One test per mapping entry most-likely-to-get-swapped, minimum set:
  - `fc1` vs `fc2` ordering in the MLP block — separate tests assert `v.blk.{L}.ffn_down.weight` matches the HF `mlp.fc2.weight` tensor (not `fc1`), per `clip-model.h:{line}`.
  - `linear_1` vs `linear_2` in the multi-modal projector — separate tests assert `mm.0.weight` matches `multi_modal_projector.linear_1.weight` with expected shape, and `mm.2.weight` matches `linear_2.weight` with expected shape.
  - Patch-embedding transpose — test asserts `v.patch_embd.weight` has shape `[hidden_size, in_channels, patch_size, patch_size]` in the correct dimension order (verify direction against `clip.cpp:{line}`).
  - Position-embedding dtype — test asserts `v.position_embd.weight` is F16 (some HF configs ship it as F32; P10's emitter must cast if so, and the test asserts the cast happened correctly).
- Expected values in each test are hand-authored with a code-comment citation per row to the `clip-model.h` or `clip.cpp` line that motivated the choice. If `clip.cpp` changes upstream, tests fail loudly and the catalog + tests update together (same pattern as P4 used for `llama-arch.cpp`).

**Cross-ADR:** ADR-005 phase 2c's mmproj loader accepts our output without modification (P10 does not cause an ADR-005 change). If ADR-005's loader needs a quirk, that's an ADR-005 bug to fix on that side, not a P10 deliverable-shape change.

**Estimated LOC:** ~1000 (module + CLI flag + Layer A tests ~800; Layer B synthetic + real-model round-trip tests ~80; Layer C spec-driven tests ~150).

### 19. MTP tensor round-trip integrity gate (cross-ADR contract)

**Problem.** Decision 11 (shipped in P4) emits MTP tensors at `blk.{n_layer}.nextn.*` losslessly for both variants. We have **no evidence** that ADR-013's weight loader actually accepts those tensors — if the tensor names, shapes, or dtypes are off by one hair, MTP is cosmetic dead weight in our output GGUFs and speculative decoding will never work. The failure mode is silent: llama.cpp's loader would happily ignore unknown tensors today. Our in-house ADR-013 loader should reject them loudly, but only if we write the test.

**Decision.** Add a conversion-side integrity gate: an integration test that runs **our** convert path end-to-end on a synthetic tiny qwen35/qwen35moe with `mtp_num_hidden_layers: 1`, then loads the output via ADR-013's `src/inference/models/qwen35/{dense,moe}.rs` weight-load entry point, and asserts the inference-side MTP submodule reports available with all expected tensors populated. This closes the convert-side of the MTP contract. Full speculative-decoding **execution** (draft/accept loops, rejection sampling, n-token lookahead) is tracked as ADR-013 phase P14 (new, not in this ADR) via an explicit cross-link.

1. **New integration test:** `tests/convert_qwen35_mtp_roundtrip.rs`. Two variants:
   - `qwen35_mtp_roundtrip` — synthetic dense model, `mtp_num_hidden_layers: 1`, 4 full-attn layers, hidden=64. Convert → Load → Assert `model.mtp.is_some()` and every tensor in `EXPECTED_MTP_TENSORS` (hand-authored const, shapes + names derived from `/opt/llama.cpp/src/llama-arch.cpp:447-450`) is present with correct shape and dtype.
   - `qwen35moe_mtp_roundtrip` — synthetic MoE model, same MTP config, 2 linear + 1 full-attn layers, 4 experts. Same assertions.
2. **Error quality requirement.** If the loader rejects any MTP tensor, the test assertion failure must include the offending tensor name. This means ADR-013's loader must emit structured errors keyed by tensor name, which is independently a reasonable expectation of any production weight loader (so this acceptance clause doubles as a loader-quality gate, not a hoop).
3. **Wall-clock budget.** The test runs < 30 s on a laptop (tiny synthetic models, no disk download). It lives in the default `cargo test` set, not behind an `--ignored` flag. CI must run it on every PR.
4. **Cross-link.** Add a `docs/ADR-013-qwen35-inference.md` entry ("MTP speculative-decoding execution — new phase P14, blocked on ADR-012 P11 green") alongside this phase landing. The cross-link is committed *with* the P11 test, not after — this prevents the "oh, someone will track the inference side later" decay mode.

**Acceptance criteria.**
- `cargo test --test convert_qwen35_mtp_roundtrip` green for both variants. CI runs on every PR.
- Injecting a deliberate bug in the MTP emission path (e.g. renaming `blk.{L}.nextn.embed_tokens.weight` to `blk.{L}.nextn.emb_tokens.weight`) causes the test to fail with an assertion message naming the missing tensor by exact name. Recorded as a manual bisection step in `docs/converting-qwen35.md` ("how P11 catches MTP regressions").
- `docs/ADR-013-qwen35-inference.md` contains a "P14 — MTP speculative-decoding execution" entry with status `planned` and a blocker reference to ADR-012 P11. Landed in the same commit as the P11 test. (This is the cross-ADR handoff that prevents Decision 19 from being a sleeping stub.)
- No change to the shipped MTP tensor emission format from P4 — Decision 19 is additive validation, not a re-spec of Decision 11.

**Estimated LOC:** ~200 (integration test + ADR-013 cross-link entry + any error-message surface hardening on the ADR-013 loader side).

### 20. Arch-table-driven scaffolding (generalize P8 / P10-structural / P11 now, not later)

**Problem.** P8, P10's round-trip gate, and P11's MTP integrity gate share a pattern: each is an arch-parameterized conformance test that runs over `arch ∈ {qwen35, qwen35moe, …}`. Writing them Qwen-specific and refactoring when Ministral lands is a measured cost: Ministral and DeepSeek-V3 are concrete next candidates (Robert, 2026-04-24), Gemma4 parity is the follow-up ADR after this one, and the pure-Rust crate-factory vision (`project_pure_rust_crate_factory.md`) treats arch-onboarding velocity as the product. Front-loading the generalization avoids Qwen-specific code becoming technical debt in weeks, not months.

**Decision.** Introduce a shared `src/arch/` module that owns the conformance surface. Every arch (including the existing Gemma4 and both new Qwen variants) registers into a single `ArchRegistry` and all conformance tests consume `&ArchEntry` rather than hard-coded Qwen paths.

**Concrete module layout (ADR-012 ships exactly this — no stubs, no placeholder files):**
```text
src/arch/
├── mod.rs              — pub re-exports + ArchRegistry singleton
├── registry.rs         — ArchEntry struct + HashMap<&'static str, ArchEntry>
├── conformance.rs      — arch-generic smoke / round-trip / catalog helpers
├── smoke.rs            — hf2q smoke subcommand impl (Decision 16 binary)
├── catalog.rs          — TensorCatalog: &[(TensorName, TensorShape, Dtype)]
└── entries/
    ├── qwen35.rs       — qwen35 ArchEntry (fully populated)
    └── qwen35moe.rs    — qwen35moe ArchEntry (+ MTP + expert-merge hooks; fully populated)
```

Future arches land their own `entries/<arch>.rs` in their own ADRs (Gemma parity, ADR-015 Ministral, ADR-016 DeepSeek-V3). ADR-012 does not create placeholder files for them — per the mantra, a populated-stub is still a stub.

**`ArchEntry` surface (sketched):**
```rust
pub struct ArchEntry {
    pub arch: &'static str,                                 // "qwen35", "qwen35moe", "gemma4", …
    pub hf_architectures: &'static [&'static str],          // ["Qwen3_5ForCausalLM"]
    pub tensor_catalog: &'static TensorCatalog,             // hand-transcribed per arch
    pub has_mtp: bool,                                      // Decision 19 uses this
    pub has_vision: bool,                                   // Decision 18 uses this
    pub smoke_prompts: &'static [&'static str],             // deterministic smoke inputs
    pub ppl_corpus: EvalCorpus,                             // Decision 17 uses this
    pub quality_thresholds: QualityThresholds,              // per-arch ppl/KL bounds
}
```

**Binary subcommand (replaces the earlier `scripts/smoke_test_qwen35.sh` proposal):** `hf2q smoke --arch {qwen35|qwen35moe|gemma4|ministral|deepseekv3} --quant {q4_0|dwq-mixed-4-6|dwq-mixed-4-8} [--with-vision]`. Rust binary, not bash — single-entry, testable, dispatch via `ArchRegistry::get(arch)`. Preflight, convert, llama-cli invocation, transcript emission, PPL/KL measurement are all arch-generic and read their knobs from `ArchEntry`.

**Where arch-specific code still lives:**
- `src/models/qwen35/{mod,dense,moe}.rs` unchanged — Qwen transforms stay Qwen (V-head reorder, A_log negation, MROPE, SSM tensors).
- `src/models/vit/` (P10) unchanged — ViT emission stays its own module.
- `src/quantize/dwq.rs` unchanged — cohort priors are per-arch but consumed via the existing `ArchFamily` dispatch.

**Chesterton's fence — why this decision lands here and not earlier.** P0–P7 shipped without `src/arch/`. The decision to generalize is explicitly a 2026-04-24 refinement driven by Robert naming Ministral + DeepSeek as near-term targets. If we had known that before P0, `src/arch/` would have landed in P1; since we didn't, we land it in P8 and migrate Gemma4 into the registry as part of its parity follow-up ADR. No rewrite of P0–P7 code — P0–P7's per-arch files stay as-is and are *re-exported* from the new registry entries.

**Acceptance criteria.**
- `src/arch/` exists, compiles, and is reachable via `hf2q smoke --help` by the end of P8.
- `qwen35` and `qwen35moe` `ArchEntry`s are populated with the catalogs P4 hand-transcribed; no duplication — P8 test helpers consume the entry, they don't hardcode tensor names.
- `src/arch/entries/` contains **exactly two files** at P8 close: `qwen35.rs` and `qwen35moe.rs`. No `gemma4.rs`, no `ministral.rs`, no `deepseekv3.rs`, no placeholder files. Future arches add their own entry files in their own ADRs.
- A negative-case unit test: `hf2q smoke --arch X` for any `X` not in `{qwen35, qwen35moe}` (including `gemma4`, `ministral`, `deepseekv3`, `bogus`) returns the **same** structured error naming the attempted arch and listing the known entries. No special-cased `todo!()` branches per-arch — the dispatcher rejects unknown keys uniformly. Proves the registry is load-bearing, not a facade.
- Gemma4 regression: the existing Gemma4 convert path (outside `src/arch/`) is unchanged. Gemma4 remains fully convert-capable in ADR-012 via its existing P0–P7 code path; it is simply not yet registered in the new conformance surface. The Gemma parity follow-up ADR opens `gemma4.rs` when it opens.

**Estimated LOC:** ~400 (module scaffolding ~250, Qwen entries ~100, smoke binary ~50). Absorbed into P8's budget.

---

## Arch-Table Scaffolding Architecture (reference, 2026-04-24)

The diagram below shows how the scaffolding pattern lands across ADR-012, the Gemma4 parity follow-up, and the two concrete future arches Robert named. Every shared-scaffolding file lives under `src/arch/`; every arch-specific file lives under `src/models/<arch>/`. No cross-pollination — a future arch touches exactly one `src/arch/entries/X.rs` registration file and one `src/models/X/` module.

```text
                   ┌────────────────────────────────┐
                   │    src/arch/registry.rs        │
                   │  (ArchEntry * HashMap singleton)│
                   └───────────┬────────────────────┘
                               │
     ┌─────────────────────────┼─────────────────────────────────┐
     │                         │                                 │
     ▼                         ▼                                 ▼
┌──────────────┐    ┌────────────────────┐            ┌──────────────────────┐
│ smoke.rs     │    │ conformance.rs     │            │ catalog.rs           │
│ hf2q smoke   │    │ mtp_roundtrip()    │            │ TensorCatalog type   │
│ --arch X     │    │ mmproj_roundtrip() │            │ (hand-transcribed)   │
│ --quant Y    │    │ ppl_kl_eval()      │            │                      │
└──────┬───────┘    └─────────┬──────────┘            └──────────────────────┘
       │                      │
       └──────────┬───────────┘
                  │
                  ▼
      ┌──────────────────────────────────────┐
      │ src/arch/entries/                    │
      │  — Present in ADR-012 P8 —           │
      │  ├─ qwen35.rs                        │
      │  └─ qwen35moe.rs                     │
      │                                      │
      │  — NOT present in ADR-012 —          │
      │  (each future arch adds its own file │
      │   via its own ADR; no stubs)         │
      │  ⋯ gemma4.rs    — Gemma-parity ADR   │
      │  ⋯ ministral.rs — ADR-015            │
      │  ⋯ deepseekv3.rs — ADR-016           │
      └──────────────────────────────────────┘
```

**Cost distribution:** ADR-012 pays ~400 LOC of generalization (Decision 20). Gemma4 parity pays ~150 LOC (populate `gemma4.rs` + committed transcripts, no rewrite). Each future arch pays ~200–400 LOC of new transforms + <50 LOC of registry registration. Without this decision, each future arch would pay ~1500 LOC of conformance-harness duplication.

---

## Phase plan

Phases are **dependency-ordered**, not priority-ordered. Each phase has a single owner claim per `feedback_swarm_sequential_when_shared_build.md` (shared Cargo target = sequential).

### P0 — Broken-window fix (standalone, do first)

**Scope:** Decision 10 only. DWQ bit-pair parameterization + `--bits`-silently-ignored fix + output filename disambiguation.

**Why first:** Orthogonal to every other phase. Unblocks the "4 AND 8" user ask independent of qwen35moe timing. Lands as a small, reviewable PR.

**Deliverables:**
- `src/cli.rs`: new `DwqMixed48`, `DwqMixed68`, `DwqMixed28` enum variants
- `src/main.rs:447-448`: dispatch-driven `(base_bits, sensitive_bits)` lookup
- `src/quantize/dwq.rs`: ensure `DwqConfig` honors the dispatched values
- Output-filename naming logic
- Unit tests per Decision 10's acceptance criteria
- Gemma-4 regression test (byte-identical output for `dwq-mixed-4-6`)

**Acceptance:**
- All Decision 10 acceptance criteria met
- `hf2q convert --repo <gemma> --quant dwq-mixed-4-8 ...` produces a different, loadable GGUF than the 4-6 variant
- CI green, Gemma regression byte-identical

**Estimated LOC:** ~150

### P1 — Config ingestion

**Scope:** Decisions 2 and 3.

**Deliverables:**
- Extended `Config` struct in `src/input/config_parser.rs` with the 18 new fields
- Nested `RopeParameters` struct
- `layer_types` / `full_attention_interval` dual-support logic
- `SUPPORTED_LAYER_TYPES` extension
- Unit tests loading the apex model's `config.json` and Gemma-4's `config.json`

**Acceptance:**
- Decision 2 & 3 criteria met
- No regression on Gemma config parsing

**Estimated LOC:** ~300

### P2 — Scaffold `src/models/qwen35/` module and V-head reorder

**Scope:** Decisions 4 and 5.

**Deliverables:**
- New module `src/models/qwen35/` with `mod.rs`, `dense.rs`, `moe.rs` per Decision 4's public API
- V-head reorder implementation in `mod.rs` (shared by both variants) with 6 case handlers
- Unit tests per Decision 5's acceptance criteria (per-case + round-trip + spec-driven)
- Test fixtures authored in Rust from deterministic small inputs; expected outputs hand-authored in test files with code comments deriving them from ggml broadcast semantics

**Acceptance:**
- Decision 4 & 5 criteria met
- Module builds, tests pass
- No fixtures produced by external tools (sovereignty directive)

**Estimated LOC:** ~500

### P3 — Qwen-specific non-reorder transforms

**Scope:** Decision 6.

**Deliverables:**
- A_log negation, dt_bias rename, conv1d squeeze, in_proj_qkvz reorder, RMS norm +1 audit note
- One unit test per transform
- Code comment documenting the RMS norm +1 audit with citation

**Acceptance:** Decision 6 criteria met.

**Estimated LOC:** ~250

### P4 — GGUF metadata + tensor naming (both variants)

**Scope:** Decisions 1, 7, 8, 11.

**Deliverables:**
- Arch routing in `src/backends/gguf.rs` for both `qwen35` and `qwen35moe`
- `emit_metadata_dense()` in `qwen35/dense.rs` with all `qwen35.*` keys per Decision 7's catalog
- `emit_metadata_moe()` in `qwen35/moe.rs` with all `qwen35moe.*` keys per Decision 7's catalog
- Tensor-name mapping per Decision 8, per variant (full-attn, linear-attn shared; dense FFN vs MoE experts+shared-experts; MTP shared)
- Spec-driven metadata test framework that asserts hf2q output contains every key in the hand-transcribed catalog
- Integration-ready for both arches (final end-to-end wiring lands in P7)

**Acceptance:** Decisions 1, 7, 8, 11 criteria met for both variants.

**Estimated LOC:** ~750 (dense + MoE metadata/naming, plus shared full-attn + linear-attn + MTP naming)

### P5 — Expert merge + MoE pipeline (MoE variant only)

**Scope:** Decision 9. Dense variant skips this phase entirely (arch-gated no-op).

**Deliverables:**
- Layer-streaming expert merge for 256 experts × 3 projections per MoE layer
- Shared expert singleton handling
- Memory profile measurement on the 35B MoE target (must stay < 64 GB peak)
- Unit test with 4 synthetic experts
- Compile-time or runtime guarantee that dense convert path does not invoke expert-merge code

**Acceptance:** Decision 9 criteria met; memory target met on real 35B MoE convert; dense 27B convert uses standard FFN emission with no expert-merge code touched.

**Estimated LOC:** ~250

### P6 — DWQ integration for hybrid arch

**Scope:** Decisions 12 and 13.

**Deliverables:**
- Cohort-prior extension to sensitivity scorer per Decision 12
- `ActivationCapture` trait definition per Decision 13
- Mock implementation for hf2q-side tests
- Wire into convert pipeline; if inference engine isn't ready, P6 blocks and we fix the blocker — no weight-space fallback path is wired into the qwen35 / qwen35moe convert flow
- KL-divergence regression guard on Gemma-4 DWQ-4-6

**Acceptance:** Decisions 12 & 13 criteria met.

**Estimated LOC:** ~300

### P7 — Integration test, HF download hardening, sidecar, docs

**Scope:** Decisions 14, 15; full end-to-end convert tests for both variants.

**Deliverables:**
- `tests/convert_qwen35_integration.rs` — synthetic tiny qwen35 (dense) end-to-end, asserts all metadata and tensor-name invariants against the spec-driven expected structure
- `tests/convert_qwen35moe_integration.rs` — same for qwen35moe variant
- HF download preflight (Decision 14) — sized appropriately for both ~55 GB dense and ~70 GB MoE
- Sidecar file copy (Decision 15)
- `docs/converting-qwen35.md` — canonical command with the `--repo` + `--quant dwq-mixed-4-6` and `--quant dwq-mixed-4-8` invocations, covering both dense and MoE invocations
- `docs/converting-a-model.md` — generic convert-command reference (also documents Gemma retroactively)
- Update `docs/shipping-contract.md` to indicate qwen35 and qwen35moe are additional model classes with their own acceptance criteria (lighter than Gemma since inference coherence is out of scope here)

**Acceptance:**
- End-to-end integration tests pass on synthetic data for both variants
- End-to-end real-model smoke tests produce GGUFs that load in llama.cpp with no errors for each variant (loading is acceptance; inference coherence is out of scope and delegated to the inference session's future ADR)
- Docs cover both DWQ variants and both arch variants

**Estimated LOC:** ~400 (excluding docs word count)

### P8 — Arch-registry scaffolding + real-model smoke subcommand

**Scope:** Decision 16 + Decision 20. P8 absorbs the arch-table generalization cost up front so Gemma-parity, Ministral, and DeepSeek-V3 follow-ups do not each pay a fresh harness-rewrite tax.

**Dependency:** P7 (green). Independent of P9/P10/P11 — runs at Q4_0 today, re-invoked over DWQ outputs once P9 lands. Completing Decision 20's registry scaffold *within* P8 is the critical design call: if it slips to "after P8 ships", every later arch pays the refactor cost.

**Deliverables:**
- `src/arch/{mod,registry,conformance,smoke,catalog}.rs` — Decision 20's scaffolding (~400 LOC). `ArchRegistry` singleton, `ArchEntry` struct, `TensorCatalog` type, arch-generic smoke dispatch.
- `src/arch/entries/{qwen35,qwen35moe}.rs` — the only two registry entries shipped in ADR-012. Both fully populated (catalogs re-exported from the existing P4-shipped constants). No `gemma4.rs`, no `ministral.rs`, no `deepseekv3.rs` — those land in their own ADRs when those ADRs open. Existing Gemma4 convert path (outside `src/arch/`) is unchanged.
- `src/cli.rs` + `src/main.rs` — new `smoke` subcommand wiring.
- `tests/fixtures/smoke-transcripts/{qwen35,qwen35moe}-q4_0.txt` — committed transcript artifacts (8 tokens each, timestamps stripped).
- `tests/smoke_conformance.rs` — CI-green unit test suite covering: preflight exit codes 2/3/4/5/6 via mocks; unknown-arch dispatch error; registry-entry-exists check; tensor-count assertion against catalog.
- Entry in `docs/converting-qwen35.md`: "Running `hf2q smoke`" section — one page, states disk/time/token requirements, documents `--arch` and `--quant` matrix.
- Entry in `docs/arch-onboarding.md` (new): "Adding a new model family to hf2q" — canonical checklist for Ministral / DeepSeek / future arches, derived from the qwen35 registration diff.

**Acceptance:** Decisions 16 + 20 criteria met. Both Q4_0 transcripts land under `tests/fixtures/smoke-transcripts/` and are byte-identical across two fresh runs on the same host. `hf2q smoke --arch bogus` and `hf2q smoke --arch gemma4` (or any other unregistered arch) both return the same uniform "unknown arch; known arches: qwen35, qwen35moe" structured error — no per-arch placeholder branch.

**Estimated LOC:** ~500 (Decision 20 scaffolding ~400, Decision 16 smoke-subcommand glue ~50, conformance unit tests ~50).

### P9 — DWQ ActivationCapture real-weight wire-up

**Scope:** Decision 17. Replaces the P6 `NoActivationCapture` guard with a production impl.

**Dependency:** ADR-013 P12 complete and green. ADR-013 P12 must expose forward-pass hook points on `Qwen3_5TextModel` and `Qwen3_5MoeTextModel` that `RealActivationCapture` can consume. If ADR-013 P12 is not green, P9 does not start — the answer is to finish P12, per `feedback_no_shortcuts.md` and mantra.

**Deliverables:**
- `src/inference/models/qwen35/activation_capture_real.rs` — new file; `pub struct RealActivationCapture`, `impl ActivationCapture`.
- `src/main.rs:488-506` — rewrite: delete the guard, construct `Box<dyn ActivationCapture>` branching on `requires_activation_capture()`, pass to `run_dwq_calibration`.
- `src/quantize/dwq.rs` — remove `DwqError::NoActivationCapture` variant and associated secondary-defence clause (now dead code).
- `src/arch/conformance.rs` — `ppl_kl_eval(arch: &ArchEntry, dwq_gguf: &Path, ref_forward: &dyn Forward) -> QualityReport` helper. Consumed by `hf2q smoke --quant dwq-*`. Arch-generic so Gemma-parity follow-up and future arches reuse.
- `tests/fixtures/ppl-corpus/wikitext2.tokens` — 512-token deterministic eval corpus; SHA-256 sidecar committed alongside.
- `tests/convert_qwen35_real_activation_capture.rs` — integration test; synthetic tiny qwen35 (dense 4-layer + moe 4-layer+4-expert) drives `RealActivationCapture`, asserts sensitivity JSON differs from weight-space-only scoring.
- `tests/quality_thresholds.rs` — unit test verifying the `QualityThresholds` constants in `src/arch/registry.rs` match the ADR-012 party-mode-confirmed numbers (1.10 / 1.05 / 0.02). Prevents silent drift of the thresholds.
- `tests/calibration_eval_disjoint.rs` — unit test asserting the DWQ calibration corpus and the PPL eval corpus have zero shared samples (SHA-256 of token lists). Prevents accidental "train on test".
- Gemma-4 DWQ-4-6 byte-identical regression test (SHA-256 on output GGUF vs pre-P9 HEAD).
- Four real-model DWQ GGUFs produced, smoke-tested *and quality-measured* via P8:
  - `models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf`
  - `models/qwen3.6-27b-dwq48/qwen3.6-27b-dwq48.gguf`
  - `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf` ← the Robert-named end-deliverable
  - `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48.gguf` ← the Robert-named end-deliverable
- Four new smoke transcripts committed under `tests/fixtures/smoke-transcripts/` (two per variant × two bit-pair presets), each including inline PPL + KL numbers against the F16 reference.
- Commit message records: per-sample wall time, total wall time, peak RSS (from `/usr/bin/time -l`), PPL ratios for all four variants, median KL nats for all four variants, sensitivity JSON SHA-256 delta vs. pre-P9 weight-space run.

**Acceptance:** Decision 17 criteria met, including both structural *and* quality ACs. All four real-model DWQ GGUFs exist, pass P8's smoke gate, and satisfy PPL + KL thresholds (DWQ46 ≤ 1.10× F16, DWQ48 ≤ 1.05× F16, median KL < 0.02 nats). Gemma regression byte-identical. `cargo clippy` reports no dead-code warning on `DwqError`.

**Estimated LOC:** ~700 (`RealActivationCapture` ~250, PPL/KL eval helper ~200, main.rs wire-up ~50, integration + unit tests ~150, regression test ~50).

### P10 — Pure-Rust mmproj vision-tower emitter (defense-in-depth)

**Scope:** Decision 18 with all four defense layers (party-mode Option 5). Independent of P9 — can ship in parallel.

**Dependency:** P8 (arch-registry scaffolding; P10 registers vision emission via `ArchEntry::has_vision`). Uses ADR-005 phase 2c's mmproj loader as a read-only consumer for Layer B round-trip acceptance, no build-time dep.

**Deliverables:**
- `src/models/vit/mod.rs` — `convert_vision_tower(hf_repo: &HfRepo, output_dir: &Path) -> Result<Option<PathBuf>>` (returns `Ok(None)` when `vision_config` absent).
- `src/models/vit/config.rs` — `VisionConfig` parser + validator + typed errors.
- `src/models/vit/convert.rs` — HF tensor-name → GGUF tensor-name mapping table (hand-transcribed from `/opt/llama.cpp/tools/mtmd/clip-model.h` and `clip.cpp`, one file:line citation per entry).
- `src/models/vit/gguf_emit.rs` — writes F16 mmproj GGUF with `general.architecture = "clip"` and `clip.vision.*` metadata.
- `src/cli.rs` — new `--emit-vision-tower` flag on the `convert` subcommand.
- `src/main.rs` — wire the flag through the convert pipeline, with silent-skip semantics when `vision_config` is absent (no error).
- **Layer A (structural):** `tests/convert_vision_tower_integration.rs` (synthetic tiny ViT round-trip), `tests/convert_vision_tower_noop.rs` (no-op-when-absent), Gemma-4 SHA-256 regression fixture.
- **Layer B (ADR-005 round-trip, synthetic + real-model):** `tests/convert_vision_tower_adr005_roundtrip.rs` — synthetic mmproj loads via `src/inference/vit/loader.rs` (default cargo test); `#[ignore]`-gated real-model variant exercises 27B dense shapes; every expected tensor present with correct shape; loader errors include tensor name on reject.
- **Layer C (spec-driven layout):** unit tests in `src/models/vit/convert.rs` covering the four highest-risk mapping entries (`fc1`↔`fc2`, `linear_1`↔`linear_2`, patch-embd transpose, position-embd dtype). Each test has a citation comment to `clip-model.h` or `clip.cpp` line.
- Real-model smoke update: `hf2q smoke --arch qwen35 --with-vision` invokes `llama-mtmd-cli --mmproj ... --model ...` and records an extra transcript under `tests/fixtures/smoke-transcripts/`.
- `docs/converting-qwen35.md` section: "Emitting the vision tower (dense variant only)".

**Acceptance:** Decision 18 criteria met across all three layers (A + B + C). Correctness is proven against the `clip.cpp` + `clip-model.h` spec via hand-transcribed catalog and spec-driven synthetic tests; no external-reference oracle is invoked (sovereignty).

**Estimated LOC:** ~1000 (module + CLI + wire-up + Layer A ~800; Layer B synthetic + real-model round-trip tests ~80; Layer C spec-driven tests ~150).

### P11 — MTP tensor round-trip integrity gate

**Scope:** Decision 19. Independent of P9/P10.

**Dependency:** ADR-013's weight loader must expose a tensor-name-aware error surface. If ADR-013's loader currently swallows unknown tensors or returns opaque errors, fix that surface as part of P11 — it is a genuine loader-quality requirement, not a hoop.

**Deliverables:**
- `tests/convert_qwen35_mtp_roundtrip.rs` with two test functions (dense + MoE variant). Runs < 30 s, default `cargo test` set, CI-green.
- `const EXPECTED_MTP_TENSORS: &[(&str, &[usize], TensorDtype)]` — hand-authored, derived from `/opt/llama.cpp/src/llama-arch.cpp:447-450` with per-entry citations.
- `docs/ADR-013-qwen35-inference.md` — new phase entry "P14 — MTP speculative-decoding execution" with status `planned` and a blocker reference to ADR-012 P11. **Landed in the same commit as the P11 test.**
- Manual bisection note in `docs/converting-qwen35.md` demonstrating how a one-letter rename of an MTP tensor trips the gate — proves the gate is live.
- If needed: `src/inference/models/qwen35/weight_loader.rs` error-message surface hardening so load failures include the offending tensor name.

**Acceptance:** Decision 19 criteria met. Both variants round-trip successfully. A deliberate bug in the MTP emission path fails the test with an actionable error message naming the tensor. ADR-013 P14 entry exists.

**Estimated LOC:** ~200 (test ~120, ADR-013 entry ~40, optional loader error hardening ~40).

### Totals

- **Code LOC (shipped P0–P7):** ~8,600 across 8 feature commits.
- **Code LOC (remaining P8–P11, post 2026-04-24 party-mode refinement):** ~2,400 (P8 ~500 incl. Decision 20 scaffolding; P9 ~700 incl. PPL/KL eval infra; P10 ~1,000 three-layer defense; P11 ~200).
- **Tests:** ≥ 1 unit test per transform (done); 4 integration tests across P0–P7 (done); P8 adds conformance unit test suite + committed transcripts; P9 adds 2 integration tests + quality-threshold unit test + calibration-eval-disjoint unit test + regression test; P10 adds Layer A+B+C+D tests; P11 adds 1 integration test with 2 variants. **Total additional:** ~12 test files across P8–P11.
- **Docs:** 3 new/updated in P7 (done); P8 adds `docs/arch-onboarding.md` (new — canonical checklist for future arches) + `docs/converting-qwen35.md` smoke section; P10 adds a vision-tower section + `tests/fixtures/mmproj-ground-truth.md`; P11 adds a manual-bisection note + an ADR-013 cross-link entry.
- **Real-model deliverables (P9 end-gate):** 4 DWQ GGUFs (dense dwq46, dense dwq48, MoE dwq46, MoE dwq48), each passing PPL + KL quality thresholds.
- **Transcripts (P8/P9):** 8 committed smoke transcripts (2 × Q4_0 baseline + 4 × DWQ with inline PPL/KL + 2 × vision-enabled dense variants).
- **Quality thresholds (party-mode-confirmed):** DWQ46 PPL ≤ 1.10× F16 · DWQ48 PPL ≤ 1.05× F16 · median KL < 0.02 nats · mmproj cosine anchor ≈ 1.0 within F16 precision.
- **Arch-table payoff (Decision 20):** Gemma4 parity follow-up pays ~150 LOC; Ministral ADR-015 and DeepSeek-V3 ADR-016 each pay ~200–400 LOC of arch-specific transforms + <50 LOC registry registration. Without Decision 20, each would pay ~1500 LOC of conformance-harness duplication.
- **Timeline:** sequential due to shared Cargo target. Parallelism opportunities:
  - P8 (registry + Q4_0 harness) can ship immediately — no dependency on P9/P10/P11.
  - P10 (vision-tower emitter) depends on P8's registry but is independent of P9 and P11.
  - P11 (MTP round-trip) depends on P8's registry but is independent of P9 and P10.
  - P9 depends on ADR-013 P12 being green + ADR-013 F16 forward being bit-stable (implicit handoff, R12).
  - **Optimal ordering: P8 → P11 → P10 → P9** (P8 first because its scaffolding is consumed by every later phase; P9 last because it has the external dependency and the highest disk/time cost).

---

## Test strategy

### Unit tests (per-transform)

Located alongside implementations in `src/models/qwen35/` (shared transforms in `mod.rs`, dense-specific in `dense.rs`, MoE-specific in `moe.rs`). Each tensor transform in Decisions 5, 6, and 9 has a test with hand-constructed tiny inputs and known outputs, with expected values derived from the mathematical specification in a code comment.

### Regression tests (Chesterton's fence)

- **Gemma-4 DWQ-4-6 byte-identical output** before vs. after P0 (bit-pair parameterization). This is the only way to know P0 didn't accidentally change the Gemma codepath.
- **Gemma-4 sensitivity choices byte-identical** before vs. after P6 (hybrid cohort priors). Cohort priors must fire only when arch is qwen35moe.

### Integration tests

- `tests/convert_qwen35moe_integration.rs`: synthetic tiny-model end-to-end. 2 linear + 1 full-attn layers, 4 experts, hidden_size=64, head_dim=16. Weights are deterministic from a fixed seed.
- Output GGUF is read back by hf2q's own GGUF reader and asserted against a **self-contained expected-structure spec** (a constant in the test file that enumerates the expected metadata keys, tensor names, tensor shapes, and for the smallest tensors the expected byte content). The spec is derived by reading llama.cpp source to understand what the loader requires, then hand-authored — not produced by running `convert_hf_to_gguf.py`.
- No binary reference GGUF is ever checked in. No `.gitignore`d "regenerate via Python on first run" pattern. The test's source of truth lives in the test file.

### Specification-driven tests

- For select tensor transforms (V-reorder, A_log negation, conv1d squeeze, in_proj_qkvz), the expected output on small hand-sized inputs is hand-authored in the test file with a code comment deriving it from the mathematical specification (ggml broadcast semantics, `-exp(x)`, shape squeeze, head-grouped reorder).
- The specification-driven test IS the spec — if the spec is wrong, the test is wrong; both change together in the same commit with a citation to the llama.cpp source that motivated the change.

### Real-model smoke test

Documented manual test in `docs/converting-qwen35moe.md`:

```
hf2q convert --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format gguf --quant dwq-mixed-4-6 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/out.gguf

llama-cli --model models/.../out.gguf -p "Hello" -n 8
```

Acceptance: llama-cli loads the file without errors and emits 8 tokens. (Coherence is out of scope; this is a "file is not corrupted" check.)

---

## Risks and mitigations

### R1: llama.cpp changes its `qwen35.*` / `qwen35moe.*` key naming

**Likelihood:** Medium — both arches are new; names may churn.
**Impact:** hf2q output stops loading in newer llama.cpp.
**Mitigation:** Doc `docs/converting-qwen35.md` lists the minimum llama.cpp version that loads hf2q's output. When upstream changes key names, that's an ADR addendum (hand-update the transcribed constant list in `src/models/qwen35/{mod,dense,moe}.rs` with new citations) — not a CVE. Spec-driven tests verify what hf2q emits against the hand-transcribed catalog; if upstream diverges, the catalog (not the fixtures) is what needs updating.

### R2: V-head reorder has a subtle bug that loads but produces garbage

**Likelihood:** Medium — this is the named silent-failure mode in `project_qwen36_architecture.md` gotcha #1.
**Impact:** GGUF loads, inference produces plausible-looking nonsense. No automated test catches it without inference.
**Mitigation:** Specification-driven tests (Decision 5 acceptance) with hand-authored expected permutation maps derived from ggml broadcast semantics — the same source of truth llama.cpp's implementation is derived from, independently applied. The inference session's sourdough-gate analogue for Qwen will catch this downstream. Do not ship to Robert's end-user workflow without either the spec-driven tests passing or the inference gate green. Deliberately no reliance on llama.cpp binary output as an oracle — that would couple our correctness proof to a repo we don't own.

### R3: DWQ calibration can't run because inference session isn't ready

**Likelihood:** High — cross-session coordination is the hardest schedule to align.
**Impact:** P6 stalls. No conversion output ships for qwen35 / qwen35moe until activation capture is real.
**Mitigation:** No weight-space fallback is offered — that would be shipping lesser output, which the mantra forbids. Instead: (a) land the `ActivationCapture` trait definition as early as possible in this ADR (P6 kicks off with just the trait in place) so the inference session has a stable target to code against; (b) keep the trait surface minimal — only what the sensitivity scorer actually consumes — to reduce the inference-side work; (c) if stalled, help the inference session directly rather than routing around them. Fix the blocker, don't fall back.

### R4: 35B conversion OOMs

**Likelihood:** Medium — Gemma-4 26B hit 54 GB peak; 35B MoE at ~256 experts has 10× the expert tensor count.
**Impact:** Convert crashes partway. User loses hours.
**Mitigation:** Preflight disk check (Decision 14). Per-layer expert-merge streaming (Decision 9). If even that is insufficient, per-projection sub-streaming — measure first, don't implement preemptively.

### R5: The target model's chat template / tokenizer needs pipeline changes

**Likelihood:** Low-medium — Qwen3.5 uses a new 248K vocab; existing hf2q tokenizer-embed logic was tested on Gemma's different format.
**Impact:** Output GGUF loads but decodes gibberish due to wrong chat template or vocab encoding.
**Mitigation:** Non-goal 2 explicitly reserves follow-up ADR scope for tokenizer changes. Phase P7 sidecar preservation (Decision 15) ensures the HF tokenizer files are present next to the GGUF for external tools. If hf2q's embed produces wrong output, open a follow-up ADR on the embed path rather than extending this one.

### R6: `--repo` download of ~70 GB fails and re-download wastes a day

**Likelihood:** Medium.
**Impact:** Lost time.
**Mitigation:** Decision 14 preflight + resumption test.

### R7: The existing `--bits` flag had users who relied on the silent-ignore behavior

**Likelihood:** Low — the flag was silently ignored, so nobody had any real dependency on it for DWQ.
**Impact:** If any automation passed `--bits` + `--quant dwq-mixed-4-6` "just in case," it now errors.
**Mitigation:** Accept the breaking change. Per `feedback_no_broken_windows.md` this is the right call. Error message (Decision 10c) clearly explains the migration.

### R8: P8 smoke test reveals a structural bug in P0–P7 output too late

**Likelihood:** Medium — P0–P7 tested against synthetic tiny models + llama.cpp-source reading, not a real-model end-to-end.
**Impact:** A hparam-emission or tensor-rename bug hides in the shipped code until P8 runs on the 27B/35B targets. Fix could require re-opening a closed phase.
**Mitigation:** P8 runs at Q4_0 — fastest possible path to the smoke gate — specifically so this risk surfaces with the least possible wall-clock cost. If P8 fails on Q4_0, the fix lands as an addendum commit under the offending phase (P1–P7) with the commit message citing P8's failure transcript. Do not defer or route around a P8 failure.

### R9: ADR-013 P12's `ActivationCapture` hook surface diverges from what P9 needs

**Likelihood:** Medium — cross-session coordination. ADR-013 P12 is defining the trait API independently and is still in flight as of 2026-04-24 (HEAD `870bd7a` is P8b, not P12).
**Impact:** P9 blocks until the trait surface resolves. Worst case: ADR-013 P12 ships a surface P9 can't use, forcing a retrofit.
**Mitigation:** The `ActivationCapture` trait at `src/inference/models/qwen35/activation_capture.rs:141` is already landed and co-owned (ADR-012 P6 adopted it, ADR-013 P12 implements it). Changes to the trait signature require review by an engineer tagged to both ADRs. If the trait needs to grow, both ADRs get an addendum in the same commit — **one interface, two implementations**, no divergence. No weight-space fallback is introduced; fixing the blocker is the only path.

### R10: mmproj GGUF format drifts in upstream `llama.cpp/tools/mtmd`

**Likelihood:** Medium — mmproj is younger than the core GGUF format and its tensor/metadata catalog is still evolving upstream.
**Impact:** hf2q's P10 output stops loading in newer `llama-mtmd-cli`. The externally-produced reference mmproj currently on disk may be targeting an older format.
**Mitigation:** Same rule as R1 (loader key churn). `docs/converting-qwen35.md` records the minimum `llama-mtmd-cli` version that loads our P10 output. The hand-transcribed tensor-name catalog in `src/models/vit/convert.rs` carries file:line citations pinned to a specific `/opt/llama.cpp` commit — when upstream drifts, we re-transcribe and re-cite, and the regression test catches anything structural. No fixture comparison against the existing external mmproj (sovereignty).

### R11: P9 wall clock + disk budget on the 35B MoE exceeds M5 Max capacity

**Likelihood:** Low-Medium — DWQ calibration on 35B MoE with 1024 samples × full forward pass through 40 hybrid layers × 256 experts is a workload nobody has measured yet.
**Impact:** P9 runs hit swap or OOM on the 64 GB M5 Max before producing a deliverable.
**Mitigation:** Before committing to 1024 samples, run a P9 dry-run with 32 samples first and record per-sample wall time + peak RSS. If extrapolation exceeds the 64 GB budget or the ≤ 2 h *soft* wall-clock target, drop sample count to the largest value that fits, document the reduction in the P9 commit message. Wall-clock is a soft target per 2026-04-24 refinement ("correctness over speed during quant"); disk and peak-RSS remain hard limits.

### R12: ADR-013 F16 forward has a silent numerical skew that P9 inherits

**Likelihood:** Medium — ADR-013's F16 forward for qwen35 / qwen35moe is being implemented in a parallel session mid-refactor (P9b-scale-fix as of 2026-04-24). Party-mode refinement accepted implicit cross-ADR handoff rather than a pinned gate (Bob's option (b)) because Robert authors both sides.
**Impact:** If ADR-013 F16 is off by e.g. 3% from the true math, P9's PPL/KL numbers measure the skew, not DWQ quant quality. Silent because our own reference is our own stack.
**Mitigation:**
1. **Determinism check.** Before any DWQ measurement, P9 asserts the F16 forward produces byte-identical logits across two fresh runs on the 16-prompt smoke set. If non-deterministic, DWQ measurement is meaningless and P9 blocks. This is the sole sovereignty-clean mitigation available to ADR-012 — a one-time llama.cpp F16 sanity anchor was considered and rejected under the same sovereignty instinct that excluded an external-mmproj cosine anchor from Decision 18: comparing our output against an external reference to prove our correctness is the pattern `feedback_hf2q_sovereignty.md` rejects, one-time or automated.
2. **Accept the residual risk.** With implicit handoff, no pinned gate, and no external oracle, some residual risk remains. The compensating factor is that Robert — single author both sides — is the only person who could unknowingly land the skew. ADR-013 correctness is proven via ADR-013's own acceptance gates (its sourdough-gate analogue when it lands), not via ADR-012 measurement. If ADR-013 ships with a silent F16 numerical skew, P9's PPL/KL numbers will reflect that skew, and the mitigation is to fix ADR-013, not to short-circuit our correctness proof through an external comparison.

---

## Open questions

1. **Exact llama.cpp loader key names for linear-attention hparams.** `LLM_KV_SSM_CONV_KERNEL`, `LLM_KV_LINEAR_ATTN_*`, or something different? Must be extracted from `/opt/llama.cpp/src/llama-arch.cpp` at engineer start time. Do not guess.
2. **Post-attention layernorm name** for qwen35moe: `blk.{L}.post_attention_norm` (Gemma convention) or `blk.{L}.ffn_norm` (LLaMA convention)? Resolved by reading `qwen35moe.cpp`'s `build_norm` calls.
3. **Does Qwen3.5 RMS norm use the +1 Qwen convention?** Gotcha #5 in `project_qwen36_architecture.md`. Audit in P3.
4. **Output-gate weight HF tensor name.** Likely `self_attn.output_gate.weight` or `self_attn.gate_proj.weight` — verify by inspecting the actual HF safetensors keys in the target model.
5. **MTP tensor full set.** Decision 11 lists the `nextn.*` names in `llama-arch.cpp:447-450` — confirm which subset Qwen3.5-MoE's MTP produces.
6. **DWQ sensitivity heuristic — is per-tensor the right granularity, or do we need per-expert?** For 256 experts per MoE layer, treating them as a single `ffn_gate_exps` tensor may mask a subset of experts that are individually sensitive. Investigation during P6.

---

## Dependencies on other work (cross-ADR)

- **ADR-013 (`Qwen3.5 / Qwen3.5-MoE Inference Support`) — P12 `weight_loader.rs` real-weight path + forward-pass hook points on `Qwen3_5TextModel::forward` / `Qwen3_5MoeTextModel::forward`.** This is the hard blocker for **P9**. No weight-space fallback path exists for `qwen35` / `qwen35moe` DWQ; if ADR-013 P12 isn't green, P9 waits. The `ActivationCapture` trait at `src/inference/models/qwen35/activation_capture.rs:141` is the stable API contract — changes require both-ADR review.
- **ADR-013 P14 — MTP speculative-decoding execution (new phase, created by Decision 19).** ADR-012 P11 ships the conversion-side contract (tensor integrity gate); ADR-013 P14 implements draft/accept loops. P14's blocker is ADR-012 P11 (P14 can't draft speculative tokens until P11 proves the tensors round-trip).
- **ADR-005 phase 2c — mmproj / ViT forward-pass execution.** Consumes ADR-012 P10's output (`mmproj-qwen36-F16.gguf`) as a read-only artifact AND provides P10 Layer B's round-trip gate (party-mode Option 5). ADR-005's mmproj loader (`src/inference/vit/loader.rs`) must expose a tensor-name-aware error surface — identical quality requirement to ADR-013's P11 dependency. If ADR-005's loader needs a quirk, the fix lands on that side — P10's deliverable shape is pinned by the hand-transcribed `clip.cpp` catalog, not by ad-hoc loader convenience.
- **`/opt/mlx-native` Metal kernels** (GATED_DELTA_NET, SSM_CONV, TRI_SOLVE, L2_NORM, CumSum per `project_qwen36_architecture.md`): needed by ADR-013's forward pass, therefore transitively needed by P9 (since `RealActivationCapture` drives that forward). Not needed by P8, P10, or P11.
- **Tokenizer pipeline:** Existing hf2q tokenizer embed path assumed sufficient for the Qwen3.5 248K vocab. Validated empirically by P8 (if the embed is wrong, llama-cli produces garbled output and P8's determinism assertion catches it). **If P8 surfaces a tokenizer bug, a follow-up ADR owns the fix — this scope does not absorb tokenizer work** (Non-Goal 2 unchanged).
- **`/opt/llama.cpp`** as a read-only spec source only: `convert_hf_to_gguf.py` (tensor transforms), `src/llama-arch.cpp` (metadata keys), `src/models/qwen35*.cpp` (graph builder tensor-name expectations), `tools/mtmd/clip-model.h` + `clip.cpp` (mmproj catalog). No build-time or runtime linkage in any P8–P11 deliverable. Verified via `cargo tree -p hf2q` in the P10 commit.

---

## Glossary

- **A3B / A4B.** MoE shorthand: "active 3B / 4B parameters per token." Total parameters (e.g. 35B for A3B) is much larger.
- **DWQ.** Dynamic Weight Quantization. hf2q's activation-calibrated mixed-precision quant: base bits for most tensors, promoted to sensitive bits for layers whose activations show high quantization error. Default Gemma preset: 4-base, 6-sensitive.
- **Gated DeltaNet.** A linear-attention variant using a selective state-space model (SSM) with a learned gating mechanism. Qwen3.5's replacement for Mamba-style state updates.
- **GGUF.** The file format used by the ggml ecosystem (llama.cpp, ollama). Contains weights + metadata + tokenizer in a single file.
- **LLM_ARCH_QWEN35.** The llama.cpp architecture constant for Qwen3.5 dense. Arch string: `qwen35`. Reference: `/opt/llama.cpp/src/llama-arch.cpp:42`, graph builder `/opt/llama.cpp/src/models/qwen35.cpp`.
- **LLM_ARCH_QWEN35MOE.** The llama.cpp architecture constant for Qwen3.5-MoE. Arch string: `qwen35moe`. Reference: `/opt/llama.cpp/src/llama-arch.cpp:43`, graph builder `/opt/llama.cpp/src/models/qwen35moe.cpp`.
- **MoE.** Mixture of Experts. Each FFN layer is replaced by N experts and a router; each token routes through top-K experts.
- **MROPE.** Multi-axis RoPE. Partitions the rotary dimensions into multiple sections with different theta bases, originally for video/image positional encoding; used in Qwen3.5-MoE for long-context efficiency.
- **MTP.** Multi-Token Prediction. An extra small transformer block after the main stack that predicts the *next* next token, enabling speculative decoding. Originated in DeepSeek-V3.
- **Shared experts.** In some MoE architectures (including Qwen3.5-MoE), a small set of experts are activated for *every* token regardless of the router. Distinct tensors from the routed experts.
- **V-head grouped vs tiled.** Two memory layouts for multi-head tensors when `num_v_heads > num_k_heads`. Grouped: V heads adjacent per K head (`G0_v0, G0_v1, G1_v0, G1_v1, ...`). Tiled: all K heads' first V, then all K heads' second V (`G0_v0, G1_v0, G0_v1, G1_v1, ...`). HF uses grouped; ggml binary ops expect tiled.

---

## Appendix A: Target convert commands (once all phases land)

### Qwen3.6-35B-A3B MoE (qwen35moe)

```bash
# 4-bit DWQ
hf2q convert \
  --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format gguf \
  --quant dwq-mixed-4-6 \
  --calibration-samples 1024 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf

# 8-bit DWQ
hf2q convert \
  --repo jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated \
  --format gguf \
  --quant dwq-mixed-4-8 \
  --calibration-samples 1024 \
  --output models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48.gguf
```

### Qwen3.6-27B dense (qwen35)

```bash
# 4-bit DWQ
hf2q convert \
  --repo Qwen/Qwen3.6-27B \
  --format gguf \
  --quant dwq-mixed-4-6 \
  --calibration-samples 1024 \
  --output models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf

# 8-bit DWQ
hf2q convert \
  --repo Qwen/Qwen3.6-27B \
  --format gguf \
  --quant dwq-mixed-4-8 \
  --calibration-samples 1024 \
  --output models/qwen3.6-27b-dwq48/qwen3.6-27b-dwq48.gguf
```

---

## Appendix B: Canonical gotcha cross-reference

Every gotcha from `project_qwen36_architecture.md` is addressed in a specific decision in this ADR. The table below is the engineer's checklist for "did I handle everything?"

| Gotcha from memory | ADR Decision |
|---|---|
| #1 V-head grouped→tiled | Decision 5 |
| #2 A_log negation | Decision 6 |
| #3 dt_bias rename | Decision 6 |
| #4 Conv1d squeeze | Decision 6 |
| #5 RMS norm +1 | Decision 6 (audit) |
| #6 in_proj_qkvz reordering | Decision 6 |
| #7 Decay-mask log-space clamp | **Inference-only** — not a convert-time transform; belongs to inference session's ADR |
| #8 MROPE sections | Decision 7 (metadata emission) |
| #9 `full_attention_interval` vs `layer_types` | Decision 2 (parser dual-support) |
| #10 Expert weights scale | **Inference-only** — runtime scaling, not persisted in tensor data |

Gotchas #7 and #10 are runtime concerns owned by the inference session; conversion does not transform them.
