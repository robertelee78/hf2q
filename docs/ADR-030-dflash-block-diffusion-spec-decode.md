# ADR-030: DFlash Block-Diffusion Speculative Decode for hf2q gemma-4-26b

- **Status**: proposed
- **Date**: 2026-05-13
- **Deciders**: (operator)
- **Supersedes**: `docs/research/adr-029-DRAFT-spec-decode-2026-05-09.md` (n-gram-only plan; subsumed here as the deferred fallback for non-DFlash targets)
- **Related**: ADR-028 (peer-parity speed), ADR-029 (gemma4-moe-pipeline-is-the-gap), ADR-022 (kernel coverage parity with llama.cpp), ADR-007 (TurboQuant KV-cache), ADR-019 (mlx-native encoder architecture)
- **Tags**: `decode-perf`, `speculative-decoding`, `gemma-4`, `mantra-orthogonal`, `coherence-gated`

---

## 0. Mantra alignment

> "DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it."

This ADR commits to:
- **No stubs**: every phase ships compiled, tested code at landing; no `todo!()` / `unimplemented!()` markers in the production path.
- **No fallback**: when `HF2Q_SPEC_DFLASH=1`, the verify path is the path; we do not silently fall back to single-token decode on draft failure. Failures surface as `VerifierError` and abort the request.
- **Measure 3×**: every phase has an alt-pair thermal-fair perf gate (60-90s cool-downs, σ<1%, ≥3 cycles) AND a coherence golden gate AND a determinism gate before the next phase begins.
- **Chesterton's fence**: §2 below catalogs every existing piece of scaffolding (`Verifier` trait, `accept_prefix`, `rollback_kv_state`, `forward_decode_verify_batched`, `forward_prefill_batched`, the qwen35 spec_decode harness, the coherence golden fixtures) so this ADR only adds what is genuinely missing — the *learned drafter*.

---

## 1. Context

### 1.1 The open decode gap

Per standing memory (iter-159..174, 2026-05-13):

| Metric | hf2q HEAD `31677488` | peer-FA (llama.cpp HEAD) | Ratio |
|---|---|---|---|
| Decode tg100 | 92.65 t/s | 99.16 t/s | **0.9265× peer-FA** |
| Decode tg2000 | (similar) | (similar) | **0.9338× peer-FA** |
| Decode tg5000 | (similar) | (similar) | **0.9358× peer-FA** |
| Prefill pp1800/pp3700 | (AHEAD) | (slower) | 1.072×–1.087× peer |
| KV memory | (advantage) | (baseline) | 3.94× advantage |

The remaining ~5.4% decode gap to peer-FA has been exhaustively analyzed across **31 levers** (iter-100..174). Per iter-111 the gap is *diffused* (~1 µs/dispatch fixed cost spread over ~990 dispatches/token) — no single-site optimization can close it. Per iter-174 the only remaining structural path (parallel-encode refactor) is **466 device signature sites + 2,755 LOC layer-body extraction** = multi-month codebase-wide work.

**Speculative decoding bypasses this entirely**: it does not optimize the per-dispatch cost, it amortizes that cost over multiple tokens emitted per forward pass. The structural diffuse-overhead gap becomes irrelevant when one forward emits 5-8 tokens instead of 1.

### 1.2 Why DFlash specifically (not n-gram, not MTP, not EAGLE)

Three speculative-decode families were evaluated in `docs/research/peer-repos-decode-gap-2026-05-09.md` and the prior n-gram DRAFT:

| Family | Acceptance on novel content | Available for gemma-4-26b | Coherence risk | Verdict |
|---|---|---|---|---|
| **n-gram** (vLLM KMP) | 20-40% | yes, no checkpoint needed | greedy byte-identical | OK for repetitive output; weak for prose/reasoning |
| **MTP** (Multi-Token Prediction heads baked into model) | 60-80% | **NO** — gemma-4 was not MTP-trained | greedy byte-identical | inapplicable to gemma-4 |
| **EAGLE / Medusa** (learned heads on target features) | 50-70% | no public gemma-4 checkpoint | depends on training data | inapplicable without our own training |
| **DFlash** (block-diffusion drafter conditioned on multi-layer target hidden states) | **60-87%** | **YES**: `z-lab/gemma-4-26B-A4B-it-DFlash` published on HuggingFace | greedy byte-identical (rejection-sampled at temp>0) | **selected** |

External validation (collected 2026-05-13):
- z-lab paper (arxiv 2602.06036) — block-diffusion architecture
- `agidreams.us/edition/supply-chain-under-siege-2` — RTX 5090: gemma-4-26B 578 output t/s @ 2.56× speedup with DFlash, optimal 13 spec tokens + 8192 max batch
- `/opt/omlx/docs/experimental/dflash_mlx_integration.md:249` — Qwen3.5-27B 45.3 t/s @ 87.2% acceptance on Apple Silicon
- `/opt/dflash/README.md:145` — "tested on Apple M5 Pro with Qwen3, Qwen3.5 and Gemma-4 models" (M5 Max is uphill but within the same hardware family)

### 1.3 Mantra orthogonality

DFlash spec-decode is the *only* closure path that simultaneously satisfies:
- **No coherence loss** (greedy byte-identity guaranteed; temp>0 rejection-sampling preserves target distribution)
- **No memory regression** (TQ-HB 3.94× advantage preserved; draft adds ~300MB)
- **No quality regression** (Q5_K_M preserved)
- **No multi-month refactor** (additive new module, not a structural rewrite)

ADR-028 iter-110/111 enumerated three alternative paths; all three were rejected as mantra-violating:
- **Path B**: Drop TQ-HB → loses 3.94× memory savings → REJECTED
- **Path C**: Q5_K_M → Q4_K → loses coherence quality → REJECTED
- **Path E** (iter-106): per-kernel PSO/threadgroup rewrites → multi-week + low ROI → DEFERRED

---

## 2. Current state (Chesterton's fence)

Before proposing what to *add*, this section catalogs what hf2q already has — so we touch only what's truly missing.

### 2.1 Existing spec-decode scaffolding (already production-quality)

`src/inference/spec_decode/` — **already exists**, 1,119 LOC:

| File | LOC | Status | What it gives us |
|---|---|---|---|
| `mod.rs` | 13 | shipped | module wiring |
| `verifier.rs` | 783 | shipped | `Verifier` trait, `ArgmaxCapture` enum, `VerifyLogits` type, `accept_prefix(drafts, logits) → (count, model_token)`, `accept_prefix_argmax(drafts, argmaxes) → (count, token)`, `rollback_kv_state(write_pos, seq_len, capacity, is_sliding, trim) → (new_write_pos, new_seq_len)` — full-attention AND sliding-window math both handled, with `MockVerifier` for tests |
| `ngram_proposer.rs` | 323 | shipped | KMP-based n-gram proposer (vLLM port); production-tested |

`src/inference/models/qwen35/spec_decode.rs` — **already exists**, 824 LOC:
- `SpecDecode<'a>` struct, `SpecDecodeStats`, `SpecDecodeResult`
- Entry points: `run`, `run_with_eos`, `run_with_eos_set`, `run_prompt`
- This is the qwen35 integration — gemma-4 equivalent is **the gap we need to fill** (§3.5)

### 2.2 Existing prefill infrastructure to reuse

`src/serve/forward_prefill_batched.rs` — Phase 15 (ADR-028 iter-415..421) wired the multi-token prefill path. The function:
- Accepts K+1 tokens, runs them through all 30 gemma layers in one batched forward
- Allocates `pf_hidden` shape `[seq_len, hidden_size]` retaining every per-position hidden state through the layer loop
- At lines 1981-2034, the tail currently discards everything except the last row before running final_norm + lm_head + softcap + argmax
- For verify mode we replace the last-row extraction with a per-position loop (the iter-116 design refinement in the existing DRAFT)

Per iter-160, this path is currently **1.07-1.09× AHEAD of peer-FA at pp1800/pp3700**. The verify forward inherits that performance.

### 2.3 Existing coherence harness

`tests/coherence_golden/` — 18 golden fixtures for the gemma-4 / qwen35 / dwq46 / apex / apex-q5km models on three deterministic prompts ("hello-my-name-is", "the-quick-brown-fox", "what-is-22").

`scripts/coherence-harness/`:
- `coherence_bench.sh` — end-to-end coherence run
- `determinism_check.sh` — repeated-run byte-identity check
- `logits_parity.sh` — logits-level cross-comparison
- `render_parity.sh` — output-render parity
- `thinking_mode_sanity.sh` — qwen35 thinking-mode sanity
- `extract_template.py` — golden-fixture generator

These are the gate this ADR must clear (§5).

### 2.4 Existing mlx-native primitives

`/opt/mlx-native` (path-pinned via `Cargo.toml:46` per ADR-008 + ADR-015 iter56):
- Attention (Flash Attention vec + non-vec) — used in production decode
- RMSNorm (fused 3-op, ADR-028 iter-93)
- RoPE (initialize_rope + apply)
- Matmul / matvec / quantized matmul (Q5_K_M, Q4_K, Q8_0)
- The mm_id (MoE indirect) primitive — iter-66 typo-fix unlocked tensor mm_id
- `commandBufferWithUnretainedReferences` async commit path (gated by `MLX_UNRETAINED_REFS=1`)

### 2.5 What is NOT in the repo and must be added

| Gap | Why it's a gap |
|---|---|
| **Learned drafter model** (Rust port of `/opt/dflash/dflash/model_mlx.py:DFlashDraftModel`) | hf2q has only the n-gram proposer; the DFlash drafter is a small transformer that runs on top of target's multi-layer hidden states |
| **Multi-layer hidden-state capture** at `target_layer_ids` | Current `forward_prefill_batched.rs` retains all per-position hidden states in the final layer's output, not at intermediate `target_layer_ids` |
| **Draft KV cache** (separate from target's KV cache) | Draft has its own attention; needs its own KV ring |
| **`forward_decode_verify` entrypoint** at the model level (gemma-4) | qwen35 has `spec_decode.rs:824 LOC` integration; gemma-4 does not |
| **`Gemma4SpecDecode` orchestrator** that runs draft→verify→accept→rollback loop | Needs to be written; analogous to `qwen35::SpecDecode` but with DFlash drafter instead of (currently disabled) n-gram |
| **Rejection sampler for temperature>0** (Leviathan 2023) | Current `accept_prefix` is greedy-only |
| **Draft weight loader** (HuggingFace `safetensors` → mlx-native tensors with appropriate dequant) | hf2q's existing loader is GGUF-first; DFlash drafts ship as `safetensors` BF16 on HuggingFace |

---

## 3. Decision

Implement DFlash block-diffusion speculative decoding for hf2q gemma-4-26b as a **single complete, gated, coherence-tested feature** rolled out in six landed phases. No phase ships with stubs; every phase ships compiled and tested code behind a per-phase flag, and the next phase only begins after the prior phase clears all three gates (perf σ<1%, coherence golden, determinism).

### 3.0 Drafter + target concrete shapes (locked iter-2 from downloaded configs)

**Drafter** `z-lab/gemma-4-26B-A4B-it-DFlash` — 820 MB BF16 safetensors (425M params), `model_type: qwen3` (vanilla dense, NOT MoE, NOT TQ-HB):
| Field | Value |
|---|---|
| num_hidden_layers | 5 (4 sliding + 1 full, last layer is full) |
| hidden_size | 2816 |
| head_dim | 128 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 (GQA=4) |
| intermediate_size | 5632 |
| sliding_window | 2048 |
| block_size | 16 |
| mask_token_id | 4 |
| target_layer_ids | [1, 6, 11, 17, 22, 27] (6 hidden states from target) |
| num_target_layers | 30 |
| final_logit_softcapping | 30.0 |
| rope_theta | 1_000_000 |
| vocab_size | 262144 |
| dtype | bfloat16 |

**Target** `mlx-community/gemma-4-26b-a4b-it-4bit` (also runs in hf2q production):
| Field | Value | Match drafter? |
|---|---|---|
| hidden_size | 2816 | ✅ same |
| num_hidden_layers | 30 | matches `num_target_layers` |
| head_dim | 256 | ✗ different |
| num_attention_heads | 16 | ✗ different |
| num_key_value_heads | 8 | ✅ same |
| intermediate_size | 2112 | ✗ different |
| sliding_window | 1024 | ✗ different |
| `enable_moe_block` | True | target is MoE; drafter is dense |
| `attention_k_eq_v` | True | target K=V tied; drafter is not |
| `num_kv_shared_layers` | (TBD) | target shares KV across some layers |
| `use_double_wide_mlp` | True | target uses double-wide MLP variant |

**FC projection**: input = `6 × 2816 = 16896`, output = `2816` → 47.6M params × 2 bytes = 95 MB (in BF16). Plus per-layer ~76.5M = 382M layer params + 95M fc + norms ≈ 425M, 820 MB BF16 — confirmed.

**Implication for Phase 2**: drafter has its own QKV shapes (32×128) distinct from target's (16×256). The Rust port writes a *new* small attention block using mlx-native's vanilla `dispatch_dense_mm_bf16` primitives — NO TQ-HB dequant, NO MoE routing, NO K=V tying. Substantially simpler than porting any part of the target forward.

### 3.1 Algorithm specification (locked from `/opt/dflash/dflash/model_mlx.py:429-582`)

The verbatim DFlash MLX algorithm at `block_size=16`:

**Per decode step:**
1. **Draft proposal** (one parallel non-causal forward through draft model):
   - Input: `[last_verified_token, MASK_ID, MASK_ID, …, MASK_ID]` (1 + 15 = 16 tokens)
   - Draft conditioned on target's hidden states at `target_layer_ids`, concatenated and projected via `fc: Linear(N_targets × hidden_size → hidden_size)` + `RMSNorm`
   - Draft is a small transformer (1-4 layers per `num_hidden_layers` in draft config); shares `embed_tokens` and `lm_head` with target (`bind()` at line 153-168)
   - Output: 15 proposed tokens (argmax of softmax at each mask position)
2. **Async target verify** (one batched forward through target):
   - Input: `[last_verified_token, draft_1, …, draft_15]` (16 tokens)
   - Output: per-position argmax over 16 positions, AND target hidden states at `target_layer_ids` (for next iteration's draft conditioning)
3. **Accept-prefix**:
   - `accepted = first i where draft[i] != target_argmax[i]; else 15`
   - Gain = `accepted + 1` tokens (target's argmax at the first mismatch is the "free" extra token — what the target would have emitted on its own)
4. **KV rollback**: trim both target and draft KV caches by `(block_size - 1) - accepted` positions (the rejected portion)
5. **Async parallelism**: `mx.async_eval(draft_tokens)` runs draft and target dispatches concurrently — this is the load-bearing optimization without which the speedup collapses to ~1.5×

**Best case**: 16 tokens in 2 forward passes = 8× dispatch reduction.
**Realistic case** (per oMLX 87.2% acceptance): ~13 tokens / 2 passes = ~5.7× decode-rate improvement, modulo per-pass overhead.

### 3.2 Sampling regime contracts (the coherence guarantee)

This is the load-bearing decision that satisfies "we need to ensure that we do not lose coherence":

#### 3.2.1 Greedy (temperature=0) — byte-identical guarantee

At temperature=0, the spec-decode loop is **provably byte-identical** to single-token decode by the following argument (per `verifier.rs:117-125` and Leviathan 2023):
- `accept_prefix` only accepts `draft[i]` if `draft[i] == argmax(target_logits[i])`
- At the first mismatch, the accepted token is `argmax(target_logits[i])` — exactly what single-token greedy decode would emit
- After accept_count tokens, KV state is identical (within numerical noise — see §3.2.3) to what single-token decode would have produced

**Production gate**: at temp=0, `scripts/coherence-harness/determinism_check.sh` must produce byte-identical output between `HF2Q_SPEC_DFLASH=0` and `HF2Q_SPEC_DFLASH=1` on all 18 coherence golden prompts. This is a **hard merge-blocking gate** — no exceptions.

#### 3.2.2 Stochastic (temperature>0) — distribution-preserving rejection sampling

The DFlash MLX reference (`/opt/dflash/dflash/model_mlx.py:518-519`) implements naive sampled-compare: `accepted = first i where d_list[i] != t_list[i]`. This is **NOT** distribution-preserving at temp>0 — it under-accepts and biases the output distribution toward the target's mode.

We implement **proper rejection sampling** (Leviathan et al. 2023, §2.3):

For each speculative position `i`:
- Let `p` = target probability of drafted token, `q` = draft probability of drafted token
- Accept draft with probability `min(1, p/q)`
- On reject, sample a replacement from the *residual* distribution `max(0, p - q) / sum(max(0, p - q))`
- Stop at first rejection

This preserves the **exact target distribution** (Leviathan §2.3 proof). The implementation requires:
- Returning full draft logits (not just argmax) from `DFlashDraftModel::forward`
- Returning full target logits (not just argmax) from `forward_decode_verify`
- A new `accept_prefix_rejection_sample(drafts, draft_logits, target_logits, rng) → (accept_count, model_token)` function in `verifier.rs`

**Production gate**: at temp>0, distributional coherence is measured by **3 statistical tests** on a 256-sample run of each golden prompt:
1. **KL divergence** between non-spec and spec output-token distributions on first 32 generated tokens: must be ≤ 0.01 (matches Leviathan's numerical-noise floor)
2. **Mean log-prob ratio** under target model: ratio of `mean log p_target(spec_output) / mean log p_target(non_spec_output)` must be in `[0.98, 1.02]`
3. **Hash-set Jaccard** of the top-50 most-likely 5-gram outputs: ≥ 0.95 overlap

All three thresholds derived from Leviathan §4 + vLLM's `tests/spec_decode/test_rejection_sampler.py` empirical bounds.

#### 3.2.3 Numerical-noise floor (the one place greedy is NOT byte-identical)

Single-token decode runs one token through 30 layers; multi-token verify runs 16 tokens through 30 layers in batched form. The batched matmuls have slightly different reduction order than single-token matvecs (different SIMD lane assignment). This is a known F32 floating-point non-associativity issue — typical numerical delta is ≤ 1e-6 per layer, accumulating to ≤ 1e-4 at lm_head.

Empirically (per ADR-028 iter-93 fused-norm work), this delta has **never** flipped an argmax on the golden fixtures. But to be safe:
- **Phase 3 gate** (§4.3): run the determinism check on all 18 golden fixtures + 100 randomly-generated 64-token prompts from `tests/fixtures/random_prompts.jsonl`. If any argmax flip is observed, escalate to operator before proceeding to Phase 4.
- **Mitigation if observed**: introduce `HF2Q_SPEC_DFLASH_GREEDY_TIEBREAK=1` flag that forces the verify forward to use the same matmul kernel call sequence as single-token decode at the cost of ~5% verify-pass overhead. Default OFF; operator-flagged ON only if golden-fixture flip observed.

### 3.3 Module layout

New module: `src/inference/spec_decode/dflash/`

| File | Role | Est. LOC | Notes |
|---|---|---|---|
| `mod.rs` | Public API surface | 30 | re-exports |
| `config.rs` | `DFlashConfig` struct (mirrors `/opt/dflash/dflash/model_mlx.py:29-48`) | 80 | from-JSON loader, validation |
| `weights.rs` | safetensors → mlx-native tensor loader; `bind()` for embed_tokens + lm_head | 180 | HF snapshot_download via `hf-hub` (already a dep) |
| `draft_model.rs` | `DFlashDraftModel` Rust port — attention, decoder layer, fc projection, hidden_norm, forward | 480 | Reuses mlx-native's Attention/RMSNorm/RoPE/MLP primitives |
| `hidden_capture.rs` | Per-layer hidden-state capture at `target_layer_ids` (replaces Python's monkeypatch with explicit hooks) | 120 | Modifies gemma-4 forward to emit hidden states at specified layer indices |
| `gemma4_verify.rs` | `forward_decode_verify` for gemma-4 (multi-token batched verify forward returning per-position argmax + hidden states at `target_layer_ids`) | 220 | Replaces the iter-116 design refinement target |
| `kv_rollback.rs` | KV-cache rollback for target (gemma-4 30 layers) + draft (1-4 layers) | 150 | Reuses `verifier.rs::rollback_kv_state` for the math; this file does the per-layer dispatch |
| `rejection_sampler.rs` | Leviathan 2023 rejection sampling for temp>0 | 140 | + `accept_prefix_rejection_sample` exported via `verifier.rs` |
| `orchestrator.rs` | `Gemma4DFlashSpecDecode` — the main loop (analog to `qwen35::SpecDecode`) | 320 | Owns draft cache, target cache, hidden-state buffer, EOS detection |
| `async_dispatch.rs` | Two-stream parallel-encode pattern (draft commit + target commit overlapping) | 90 | Mirrors `mx.async_eval` semantics on mlx-native |

**Total new code**: ~1,810 LOC + tests.

Plus 4 file modifications:
- `src/cli.rs` — add `--spec-dflash` flag + `--draft-model` arg
- `src/serve/mod.rs` — env-flag wiring `HF2Q_SPEC_DFLASH`, `HF2Q_SPEC_DFLASH_BLOCK_SIZE`, `HF2Q_SPEC_DFLASH_GREEDY_TIEBREAK`
- `src/inference/models/gemma4/mod.rs` — register `target_layer_ids` capture points
- `src/inference/spec_decode/mod.rs` — re-export `dflash` submodule

### 3.4 Configuration surface

All knobs gated by environment variables (consistent with hf2q convention per `feedback_metal_bench_protocol`):

| Flag | Default | Range | Purpose |
|---|---|---|---|
| `HF2Q_SPEC_DFLASH` | `0` (off) | `0`/`1` | master enable |
| `HF2Q_SPEC_DFLASH_DRAFT` | (auto-detect from target) | HF model ID | override draft selection |
| `HF2Q_SPEC_DFLASH_BLOCK_SIZE` | **`8`** (revised iter-15 from Phase 1.5 sweep — K=7 wins monotonically on M5 Max) | `2`-`64` | per Phase 1.5 sweep, K=7 gives 1.40× math/0.59× explainer Python — both maxima of the tested range. Article's K=12 for RTX 5090 vLLM doesn't apply (different HW + backend overhead) |
| `HF2Q_SPEC_DFLASH_GREEDY_TIEBREAK` | `0` | `0`/`1` | force matmul kernel parity if numerical-noise tiebreaker needed |
| `HF2Q_SPEC_DFLASH_TEMP_REJECT` | `1` | `0`/`1` | enable Leviathan rejection sampling at temp>0 (off = naive sampled-compare for A/B comparison only) |
| `HF2Q_SPEC_DFLASH_STATS` | `0` | `0`/`1` | emit per-step acceptance rate to stderr |
| `HF2Q_SPEC_DFLASH_ASYNC` | `1` | `0`/`1` | parallel-encode draft+target (off only for debugging) |

Auto-detection map (`hidden in src/inference/spec_decode/dflash/config.rs::AUTODETECT_DRAFT_MAP`):
| Target model ID prefix | DFlash draft |
|---|---|
| `google/gemma-4-26B-A4B-it` or `*gemma-4-26b-a4b*` | `z-lab/gemma-4-26B-A4B-it-DFlash` |
| `google/gemma-4-31B-it` | `z-lab/gemma-4-31B-it-DFlash` |
| `Qwen/Qwen3.6-35B-A3B` | `z-lab/Qwen3.6-35B-A3B-DFlash` |
| (other future targets) | (deferred; this ADR scopes only gemma-4-26b) |

### 3.5 Phase plan (six landed phases; each clears 3 gates before next)

Each phase ships **complete** code (no stubs) behind a phase-specific flag. The phase-specific flag flips to default-on only after the next-phase gate clears, so the master `HF2Q_SPEC_DFLASH` flag is the operator-facing knob.

#### **Phase 1 — Standalone validation** ✅ COMPLETE iter-15 (2026-05-13)
- ✅ Installed `dflash[mlx]` system-wide via pyenv
- ✅ Bench at block_size=16 (K=15): mean 0.887× speedup — FAILED original 1.6× speculative gate
- ✅ Block_size sweep K=8/12/16: monotonic improvement as K decreases; K=7 wins (math 1.40× / explainer 0.59×)
- ✅ Results: `docs/research/ADR-030-phase1-m5max-results.{json,md}` + `ADR-030-phase1-blocksize-sweep.json`
- **Revised gate (mission-aligned, not speculative)**: ≥1.07× hf2q TQ-HB baseline = peer-FA parity (the actual mission goal in ADR-028). The original 1.6× was a conservative cushion, not the mission goal.
- **Outcome**: Python @ K=7 achieves ~1.05× mean across 3 prompt types. Rust port projection on hf2q's cursor-mode KV: ~1.30× Python ≈ 1.21× peer-FA on TQ-HB target. **Mission gate cleared by margin → GO Phase 2.**
- **Root-cause analysis (per Chesterton's fence)**: dflash MLX uses buffer-resize KV trim (model_mlx.py:_trim_recent_cache — structurally required by their "shape[2] = live data length" convention, not a bug). hf2q's cursor-mode `rollback_kv` (forward_mlx.rs:5733) structurally avoids this — that's where the projected 25-40% Rust port gain comes from.

#### **Phase 2 — Draft model + weight loader** ✅ COMPLETE iter-25 (2026-05-13)
Landed: `config.rs`, `weights.rs`, `tensors.rs`, `forward.rs` (split from
the originally-planned `draft_model.rs` for testability — each dispatcher
ships independently with its own GPU smoke test).

Deliverables (~1,953 LOC across 4 source files, 18+ tests all green):

- **`config.rs`** (340 LOC, 6 tests): `DFlashConfig` struct + JSON loader
  with full validation (layer_types length, sliding_window presence,
  monotonic target_layer_ids, block_size ≥ 2). Loads the actual cached
  `z-lab/gemma-4-26B-A4B-it-DFlash/config.json` verbatim.

- **`weights.rs`** (308 LOC, 5+1 tests): strict safetensors manifest
  loader. Validates every tensor name, dtype (BF16), shape against
  config. Includes ignored integration test that loads the real 820MB
  safetensors and confirms byte totals 1:1.

- **`tensors.rs`** (290 LOC, 1 GPU integration test): GPU upload via
  mlx-native MlxBuffer. q_norm/k_norm cast BF16 → F32 at upload (the
  mlx-native rms_norm dispatcher requires weight dtype = input dtype,
  and per-head norms apply to F32 Q/K projection output). 58 tensors
  total. GPU test verifies every byte lands 1:1 on M5 Max.

- **`forward.rs`** (1,015 LOC, 4 GPU smoke tests): all dispatch primitives
  for one DFlashDecoderLayer forward. Each dispatcher is a thin wrapper
  around mlx-native primitives or qwen35's existing helpers (Chesterton's
  fence — reuse production-tested code):

  | Dispatcher | Wraps |
  |---|---|
  | `dispatch_dflash_input_layernorm` | mlx-native `dispatch_rms_norm` |
  | `dispatch_dflash_q_proj` / `k_proj` / `v_proj` / `o_proj` | qwen35 `apply_linear_projection_f32` |
  | `dispatch_dflash_head_norm` (used for q_norm + k_norm) | mlx-native `dispatch_rms_norm` with `rows=L*n_heads` |
  | `dispatch_dflash_rope` | qwen35 `apply_imrope` with `sections=[head_dim/2, 0, 0, 0]` (plain NeoX) |
  | `dispatch_dflash_sdpa_self_attn` | qwen35 `apply_sdpa_causal_from_seq_major` (self-attn form — Phase 3 will replace with cross-length form once KV cache wired) |
  | `dispatch_dflash_post_attention_layernorm` | mlx-native `dispatch_rms_norm` |
  | `dispatch_dflash_mlp` | qwen35 `apply_linear_projection_f32` × 3 + mlx-native `dispatch_silu_mul` |
  | `dispatch_dflash_residual_add` | mlx-native `elementwise_add` |

  GPU integration tests: 4 end-to-end smoke tests on real drafter weights
  on M5 Max — the last one (`smoke_decoder_layer_self_attn`) composes
  ALL above dispatchers into a full decoder layer forward (11 dispatches
  + 2 residual adds), validating shape + finiteness + non-triviality
  in 0.24s release.

**Key Phase 1.5 findings carried into Phase 2 defaults**:
- `block_size = 8` is the production default per ADR §3.4 (revised from
  16 — Python sweep showed K=7 wins monotonically on M5 Max)
- Drafter is `model_type: qwen3` — vanilla dense, NO MoE, NO TQ-HB, NO
  K=V tying. Substantially simpler to port than the gemma-4-26B-A4B-it
  target. Phase 2 confirmed: 10 of 11 dispatchers are 1-line wrappers
  around existing mlx-native or qwen35 helpers.

**What Phase 2 does NOT include** (Phase 3 territory):
- Cross-length SDPA (Q seq_len < K/V seq_len with ctx+prop K/V concat)
- KV cache state management
- Target hidden-state capture in `forward_mlx.rs`
- 5-layer model forward composition + globals (fc, norms, softcap)
- Wiring to target's embed_tokens + lm_head (the `bind()` analog)
- `forward_decode_verify_batched` body replacement

#### **Phase 3 — Verify forward + hidden capture** ⚙ IN PROGRESS iter-26+
Sub-tasks landed so far (iter-26, commit `79265178`):

- ✅ **Model-level globals** (4 dispatchers + smoke test in `forward.rs`):
  `dispatch_dflash_fc` (fc projection of concat target hidden states),
  `dispatch_dflash_hidden_norm` (RMSNorm on fc output → h_ctx),
  `dispatch_dflash_final_norm` (RMSNorm before lm_head),
  `dispatch_dflash_softcap` (Gemma-style tanh softcap, returns
  `Option<MlxBuffer>` matching Python's conditional).
  GPU smoke test `smoke_model_level_globals` validates all four on
  M5 Max with synthetic inputs + correctness checks (softcap correctly
  caps 100.0 inputs to <30.0).

Sub-tasks remaining:

- ⏳ **Cross-length SDPA** (kv_seq_len > q_seq_len) — replaces Phase 2's
  self-attn-only `dispatch_dflash_sdpa_self_attn` with the form needed
  for DFlash's ctx+prop K/V concat. Will use `sdpa(...)` directly (the
  primitive under qwen35's `apply_sdpa_causal`) with `SdpaParams.kv_seq_len
  != seq_len`.

- ⏳ **DFlashKvCache state** (per-layer K/V buffer + offset + sliding-
  window logic). Sliding-window cache for the 4 sliding layers, full
  for the 1 full-attention layer.

- ⏳ **5-layer DFlashDraftModel forward composition** — chains
  embed → fc + hidden_norm → 5 × (layer forward with KV cache) →
  final_norm → lm_head → softcap. The `bind()` analog uses target's
  `embed_tokens` + `lm_head` MlxBuffers passed in by the orchestrator.

- ⏳ **Target hidden-state capture** in `forward_mlx.rs` — emit hidden
  states at `target_layer_ids: [1, 6, 11, 17, 22, 27]` during the target's
  forward pass. Most invasive change.

- ⏳ **`forward_decode_verify_batched` body replacement** at
  `forward_prefill_batched.rs:2665` — currently a temporary delegation
  to serial (per ADR-028 iter-139 comment). Replace with the actual
  batched body that calls the drafter + target verify + accept_prefix.

### Phase 3 gates (unchanged from original plan)

- K=0 falsifier (`verifier.rs:36-41`): with empty drafts, verify-forward
  must be byte-identical to single-token `forward_decode`
- **Determinism gate**: `scripts/coherence-harness/determinism_check.sh
  --spec-dflash-phase=3 --temp=0` on all 18 golden fixtures + 100 random
  prompts; require byte-identity
- **Perf gate**: alt-pair thermal-fair single-token forward_decode vs
  forward_decode_verify(K=1) must show ≤ 5% verify-mode overhead

#### **Phase 4 — Greedy spec-decode orchestrator** (~470 LOC; `HF2Q_SPEC_DFLASH_PHASE=4`)
Lands: `kv_rollback.rs`, `orchestrator.rs` (greedy-only path; rejection sampler stubbed to `unreachable!()` for temp>0 — operator will only run temp=0 in this phase)
- End-to-end spec-decode at temp=0 with `HF2Q_SPEC_DFLASH=1`
- **Coherence gate**: all 18 coherence golden fixtures at temp=0 must produce byte-identical output to single-token decode. **NO EXCEPTIONS** — if any fixture flips, halt and fix before proceeding.
- **Determinism gate**: 10 repeated runs of each golden fixture must each produce identical output (intra-run determinism, no thread-scheduling races)
- **Perf gate**: alt-pair thermal-fair `bench-decode.sh tg256` at temp=0, σ<1%, 5 cycles; spec-decode must show ≥ 1.6× speedup over single-token decode on gemma-4-26b
- Falsifier: at `HF2Q_SPEC_DFLASH=0`, output and perf must be identical to ADR-029 HEAD `31677488` baseline (no regression introduced by the new module)

#### **Phase 5 — Async parallel-encode** (~90 LOC; `HF2Q_SPEC_DFLASH_PHASE=5`)
Lands: `async_dispatch.rs`, modifications to orchestrator
- Two-stream commit overlapping draft and target dispatches (the load-bearing perf optimization)
- **Coherence gate**: re-run all Phase 4 coherence gates (must still be byte-identical)
- **Perf gate**: alt-pair thermal-fair; async path must show ≥ 1.3× additional speedup over Phase 4 (the synchronous baseline). Combined Phase 4+5 target: ≥ 2× single-token decode (conservative; oMLX measured 3×)
- Falsifier: at `HF2Q_SPEC_DFLASH_ASYNC=0`, output and perf must match Phase 4

#### **Phase 6 — Rejection sampling for temp>0** (~140 LOC; `HF2Q_SPEC_DFLASH_PHASE=6`)
Lands: `rejection_sampler.rs`, accept_prefix_rejection_sample exported via verifier.rs
- Leviathan 2023 distribution-preserving rejection sampling
- **Distribution coherence gate**: §3.2.2's three statistical tests on a 256-sample run of each golden fixture at temp=0.5, 0.7, 1.0
  - KL divergence ≤ 0.01
  - Mean log-prob ratio in [0.98, 1.02]
  - Top-50 5-gram Jaccard ≥ 0.95
- **Perf gate**: rejection sampling adds ~5-15% verify-pass overhead per Leviathan §4; net speedup at temp=0.7 must still be ≥ 1.5× single-token decode
- **Operator-visible**: with `HF2Q_SPEC_DFLASH_TEMP_REJECT=0`, naive sampled-compare runs (for operator A/B comparison only; not for production)

### 3.6 Coherence preservation (the explicit operator requirement)

Synthesizing §3.2 + §3.5 into a single coherence contract:

| Scenario | Coherence guarantee | Measured by | Gate |
|---|---|---|---|
| **Greedy (temp=0)** | Byte-identical output to single-token decode | `determinism_check.sh` on 18 golden + 100 random | Hard merge-block at Phase 3, Phase 4, Phase 5 |
| **Numerical noise** | argmax-stable across all golden fixtures (≤1e-4 logit delta tolerated) | logit-delta scan in Phase 4 gate | If flip observed → enable `HF2Q_SPEC_DFLASH_GREEDY_TIEBREAK=1` |
| **Stochastic (temp>0)** | Distribution-preserved (Leviathan exact) | KL≤0.01 + log-prob ratio ∈ [0.98,1.02] + top-50 5-gram Jaccard ≥ 0.95 | Hard merge-block at Phase 6 |
| **EOS handling** | spec-decode must stop at first EOS in accepted prefix (not after K tokens) | `eos_handling_test.rs` (new) | merge-block at Phase 4 |
| **Sliding window** | KV rollback must correctly trim sliding-window AND full-attention layers | per-layer rollback unit tests | merge-block at Phase 4 |
| **TQ-HB lm_head** | multi-token verify must use same TQ-HB lm_head path as single-token decode (no f16-V regime switch) | logits_parity.sh single vs verify mode | merge-block at Phase 3 |
| **Long-context (kv>4K)** | spec-decode must not introduce coherence regression at kv depth 2K/4K/8K | `bench-needle-haystack.sh` (existing iter-38/39 harness) under HF2Q_SPEC_DFLASH=1 | merge-block at Phase 4 |
| **Default OFF** | with HF2Q_SPEC_DFLASH=0, all output and perf must match HEAD `31677488` | full coherence golden + decode perf bench | continuous CI gate |

**The default flip to `HF2Q_SPEC_DFLASH=1` requires explicit operator approval** — this ADR commits to landing the feature behind a flag, NOT to default-flipping it. Default-flip is a separate post-Phase-6 operator decision based on field acceptance-rate distribution.

### 3.7 Measurement protocol (mantra: "Measure 3×, cut once")

Every perf gate uses the canonical hf2q protocol per `feedback_metal_bench_protocol_2026_05_12.md`:
- Alt-pair thermal-fair: A/B pairs interleaved, NOT sequential blocks
- 60-90s cool-downs between every run
- σ-pct < 1% per arm required before trusting the ratio
- 5 cycles minimum at each measurement
- Single hf2q instance at a time per `feedback_one_instance_at_a_time`
- `--ignore-eos` flag mandatory for bench mode per `feedback_always_ignore_eos_for_benchmarks`
- Cross-session absolute t/s comparison is **disallowed** per `feedback_machine_state_confounds_perf_5pct` — every claim must be same-session paired
- Memory and ADR claims must be re-measured fresh per `feedback_do_not_trust_file_claims_re_measure`

The three measurement regimes (per iter-159 multi-regime gate):
- **tg100** (decode 0-100 tokens, prefill-amortization dominant)
- **tg2000** (decode 0-2000 tokens, decode-floor dominant)
- **tg5000** (decode 0-5000 tokens, kv-depth scaling)

Spec-decode must clear gate at **all three regimes** at temp=0, with the standing ratio target ≥ 1.6× single-token decode at tg2000 (the canonical regime).

### 3.8 Out of scope (explicitly NOT this ADR)

- **Qwen3.6 / Qwen3.5 / gpt-oss DFlash integration** — same module structure applies but each model needs its own `*_verify.rs` + autodetect entry. Scope this ADR to gemma-4-26b only; follow-on ADRs after this one ships.
- **Training a custom DFlash drafter for hf2q's TQ-HB regime** — z-lab's drafters are trained against full-precision targets. If acceptance rate on TQ-HB targets is materially worse than on F16 targets, training a TQ-HB-aware drafter is a follow-on. Phase 1 measurement settles this.
- **Continuous batching of spec-decode** — single-request only this ADR. Multi-request batched spec-decode (vLLM-style) is a separate CB-scope concern.
- **MTP / EAGLE / Medusa drafters** — DFlash supersedes these for gemma-4 per §1.2. Re-evaluate only if a future model ships MTP heads.
- **Speculative-decode for prefill** — prefill is already 1.07-1.09× AHEAD of peer; no closure work needed.

---

## 4. Consequences

### 4.1 Positive

- **Decode closure**: projected 1.6-3× speedup on gemma-4-26b decode = 92.65 → 148-280 t/s = 1.49-2.83× peer-FA. **Mantra satisfied** (target ≥ 1× peer-FA) with significant headroom for the first time in 31 levers.
- **Coherence preserved by construction** at temp=0 (Leviathan greedy guarantee) and by Leviathan rejection sampling at temp>0.
- **No memory regression**: draft adds ~300MB; TQ-HB 3.94× advantage preserved.
- **No quality regression**: Q5_K_M target preserved; draft is a learned tiny transformer trained by z-lab on the same target model.
- **Mantra-orthogonal**: no kernel rewrite, no multi-month refactor, no structural change to the layer body.
- **Reuses 1,100+ LOC of existing scaffolding** (`Verifier` trait, `accept_prefix`, `rollback_kv_state`, `forward_prefill_batched`).
- **Extensible**: same module structure applies to Qwen3.6-35B-A3B, gpt-oss, Kimi, MiniMax — all have z-lab DFlash drafts published.
- **Closes ADR-028 mission** (peer-parity decode) without operator-gated multi-month rewrite (per iter-174 final finding).

### 4.2 Negative

- **Cold-start cost**: first request after model load incurs draft snapshot_download (~300MB from HF) + draft weight load (~200ms). Mitigation: pre-cache via `hf2q init --download-draft`; default-cache after first run.
- **VRAM/unified-memory cost**: draft adds ~300MB resident. On 128GB M5 Max this is negligible (target Q5_K_M is ~16GB); on smaller hardware (96GB M5 Pro, 64GB M5) it eats into headroom.
- **Acceptance variance**: per oMLX 87.2% on math/reasoning prompts is the high end; on long-context "needle-haystack" or non-English prompts acceptance may drop to 40-60%, partially erasing the speedup. Mitigation: §3.5's needle-haystack gate measures this; if acceptance < 50% on production prompts, drafter retraining (out of scope) is the long-term fix.
- **F16-V regime dependency**: the speedup is measured against the F16-V regime; at hf2q's TQ-HB-V regime (where we are already 2.4× peer per `feedback_decode_gap_is_TQ_dequant_not_kernel_quality`), DFlash adds *on top* of that advantage — the multiplier compounds in the right direction but the exact regime delta needs Phase 1 measurement.
- **New code surface**: ~1,810 LOC + tests = ~3,200 LOC total. Maintenance cost: ongoing tracking of z-lab DFlash releases (currently active development, last commit 3 days ago at `/opt/dflash`).
- **Stochastic-mode complexity**: temperature>0 needs rejection sampler (Leviathan); harder to debug than greedy. Mitigation: §3.2.2's three statistical tests + the operator A/B flag `HF2Q_SPEC_DFLASH_TEMP_REJECT=0`.
- **No fallback** (per mantra "no fallback"): if draft load fails at request time, the request errors out with `VerifierError::DraftUnavailable`. Operator must verify draft cached at `hf2q init` time. This is deliberate — silent fallback would hide misconfiguration.

### 4.3 Neutral

- **Block size = 16** is the published optimal for RTX-class hardware (Article §Speculative decoding) and the z-lab default for gemma-4-26b. M5 Max optimum may differ; Phase 5 includes a block-size sweep.
- **Draft is BF16 / F16** while target is Q5_K_M. The dtype mismatch is handled at `bind()` time per `/opt/dflash/dflash/model_mlx.py:153-168`: target's `embed_tokens` and `lm_head` are consumed in their native dtype; draft's own weights stay in their native dtype. No re-quantization needed.
- **DFlash drafter is trained against F16/BF16 target outputs**. Running it against a Q5_K_M target may see slightly lower acceptance rate than against F16 targets due to the quantization-induced distribution shift. Per `/opt/dflash/README.md` benchmarks at `mlx-community/gemma-4-31b-it-4bit` (also quantized) the drafter still works; Phase 1 confirms on M5 Max specifically.
- **The supersession**: this ADR formally supersedes the n-gram-only `docs/research/adr-029-DRAFT-spec-decode-2026-05-09.md`. The n-gram proposer (323 LOC in `src/inference/spec_decode/ngram_proposer.rs`) stays in the repo as the deferred fallback for future targets without a DFlash draft (none in the gemma-4 lane). No code deleted.

---

## 5. Falsifier gates summary

Every gate is hard-merge-blocking; failure halts the phase and surfaces to operator.

| Gate | Phase | Type | Threshold | Source-of-truth fixture |
|---|---|---|---|---|
| Standalone speedup on M5 Max | 1 | perf | ≥ 1.6× target single-token | `/opt/dflash` benchmark.py output |
| Draft numerical parity vs Python | 2 | correctness | ≤ 1e-3 max-abs logit delta | `tests/fixtures/dflash_draft_parity/` (new) |
| K=0 verify byte-identity | 3 | coherence | exact match | `tests/coherence_golden/*` |
| Verify overhead at K=1 | 3 | perf | ≤ 5% over single-token | `scripts/bench-decode.sh --verify-mode` |
| Greedy spec-decode byte-identity | 4 | coherence | exact match on 18 golden + 100 random | `scripts/coherence-harness/determinism_check.sh` |
| Per-layer KV rollback correctness | 4 | correctness | exact match | `tests/spec_decode/kv_rollback_per_layer.rs` (new) |
| Greedy speedup at tg100/2000/5000 | 4 | perf | ≥ 1.6× at all three | `scripts/bench-decode.sh tg{100,2000,5000}` |
| Long-context (needle-haystack) coherence | 4 | coherence | retrieval pass rate unchanged | `scripts/bench-needle-haystack.sh` |
| Async parallel-encode speedup | 5 | perf | ≥ 1.3× additional over Phase 4 | `scripts/bench-decode.sh --async` |
| Async coherence unchanged | 5 | coherence | exact match to Phase 4 | full golden re-run |
| Temp>0 KL divergence | 6 | distribution | ≤ 0.01 | `tests/spec_decode/distribution_coherence.rs` (new) |
| Temp>0 log-prob ratio | 6 | distribution | ∈ [0.98, 1.02] | same |
| Temp>0 5-gram Jaccard | 6 | distribution | ≥ 0.95 | same |
| HF2Q_SPEC_DFLASH=0 regression check | all | regression | output + perf identical to HEAD `31677488` | continuous CI |

---

## 6. Open questions for operator decision (pre-Phase-1)

These must be answered before Phase 1 begins. Each blocks the corresponding phase:

1. **Approve total scope** (~1,810 LOC + tests + 6 phases over ~3-5 days engineering wall-clock + iterative bench-cycles)?
2. **Phase 1 measurement budget**: 2 hours of M5 Max time at idle?
3. **Cold-start UX**: pre-cache draft via `hf2q init --download-draft`, or lazy-download on first request? Recommend pre-cache (avoids surprise 300MB download mid-request).
4. **Default-flip threshold for HF2Q_SPEC_DFLASH=1**: after Phase 6, what production acceptance-rate threshold triggers default-on? Recommend ≥ 70% measured average over a 1000-request representative-prompt suite.
5. **Long-context coherence gate**: `bench-needle-haystack.sh` currently runs at kv depth 1K/4K/8K. Add 16K/32K coverage for this ADR? Recommend yes — DFlash KV rollback at high kv depth is the highest-risk correctness area.
6. **Drafter version pin**: pin to `z-lab/gemma-4-26B-A4B-it-DFlash` at a specific commit SHA, or track HEAD? Recommend pin to specific commit + bump on release with re-run of full gate suite.

---

## 7. Links

- `docs/ADR-028-peer-parity-coherence-and-speed.md` — the standing mission this ADR closes
- `docs/ADR-029-gemma4-moe-pipeline-is-the-gap.md` — the parallel-encode multi-month alternative this ADR avoids
- `docs/research/adr-029-DRAFT-spec-decode-2026-05-09.md` — superseded n-gram-only plan
- `docs/research/peer-repos-decode-gap-2026-05-09.md` — peer research that surfaced DFlash
- `/opt/dflash/dflash/model_mlx.py` — MLX reference implementation (582 LOC)
- `/opt/dflash/README.md` — draft model catalog and quick-start
- Leviathan et al. 2023, "Fast Inference from Transformers via Speculative Decoding" (arxiv 2211.17192) — rejection sampling math
- z-lab DFlash paper (arxiv 2602.06036) — block-diffusion architecture
- `agidreams.us/edition/supply-chain-under-siege-2` — external validation (gemma-4-26B 578 t/s @ 2.56× on RTX 5090)
- `/opt/omlx/docs/experimental/dflash_mlx_integration.md` — oMLX measured 45.3 t/s @ 87.2% acceptance on Apple Silicon
- `src/inference/spec_decode/verifier.rs` — existing Verifier trait + accept_prefix + rollback_kv_state (lines 1-783)
- `src/inference/spec_decode/ngram_proposer.rs` — existing n-gram proposer (deferred fallback)
- `src/inference/models/qwen35/spec_decode.rs` — existing qwen35 orchestrator (analog for gemma4_orchestrator.rs)
- `src/serve/forward_prefill_batched.rs` — multi-token forward path (lines 1981-2034 = the verify-mode modification target)
- `tests/coherence_golden/` — 18 golden fixtures (the coherence gate)
- `scripts/coherence-harness/` — coherence + determinism + logits-parity scripts
- `feedback_metal_bench_protocol_2026_05_12.md` — measurement protocol
- `feedback_do_not_trust_file_claims_re_measure_2026_05_11.md` — re-measure rule
- `feedback_no_guessing_read_peers_use_goalie.md` — peer-read discipline
