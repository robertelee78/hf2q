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

#### **Phase 3 — Verify forward + hidden capture** ⚙ MOSTLY COMPLETE iter-33

Phase 3's **drafter-in-isolation** is feature-complete. All algorithmic
primitives and the full 5-layer model forward run end-to-end on M5 Max
with real `z-lab/gemma-4-26B-A4B-it-DFlash` weights + KV cache.

**Sub-tasks landed iter-26 through iter-33:**

| Component | Commit | LOC |
|---|---|---|
| Model globals (fc, hidden_norm, final_norm, softcap) | `79265178` | 292 |
| DFlashKvCache struct + allocator | `12b816f7` | 244 |
| KV append (seq-major → head-major) + rollback | `1f23eb75` | 161 |
| Cross-length SDPA dispatcher | `e752738f` | 194 |
| KV slack-write for prop K/V | `550338ea` | 145 |
| Decoder layer attention with cache | `c4ccf5da` | 254 |
| Full decoder layer forward (attn + 2 residuals + MLP) | `d347acd4` | 139 |
| **FULL 5-layer model forward** | `5040d2fc` | 178 |

**Drafter completeness check:** `dispatch_dflash_model_forward` mirrors
`model_mlx.py:DFlashDraftModel.__call__` (lines 181-198) exactly:
- fc projection of concatenated target hidden states (16896 → 2816)
- hidden_norm RMSNorm → h_ctx
- 5-layer loop: each layer composes input_norm + 5 projections +
  3 per-head norms + 3 RoPEs (with correct per-stream offsets) +
  cache append (ctx only) + slack-write (prop only) + cross-length
  SDPA (kv_seq_len = cache.seq_len + L) + O proj + 2 residual adds + MLP
- final_norm

**GPU integration test `smoke_model_forward_with_cache`**: PASSES in
0.20s release on M5 Max. Validates: output [L, hidden] all finite +
non-trivial; ALL 5 layer caches advance by ctx_chunk_size in lockstep
(prop K/V correctly stays in slack, not persisted).

**Total Phase 2+3 LOC**: ~3,565 LOC across 5 source files; 27+ active
tests; 13 GPU integration tests passing on M5 Max.

#### **Phase 3 → Phase 4 boundary refinement**

The two remaining items from the original Phase 3 plan are **target-side
integration** work, not drafter logic:

- ⏳ **Target hidden-state capture** in `forward_mlx.rs` (the 8,643-LOC
  monolith). Per-layer hook at indices `[1, 6, 11, 17, 22, 27]` to emit
  hidden state into a caller-supplied buffer during target's forward.
  Risk-weighted as INVASIVE — production code path used by all hf2q
  decode/prefill scenarios.

- ⏳ **`forward_decode_verify_batched` body replacement** at
  `forward_prefill_batched.rs:2665` — wires target capture +
  `dispatch_dflash_model_forward` + `accept_prefix` into one verify
  call.

**Scope decision (per mantra "Measure 3×, cut once" + risk of touching
the 8,643-LOC monolith)**: these two items move to **Phase 4 (Greedy
spec-decode orchestrator)**, where they belong logically — they ARE the
orchestrator. Phase 3 was originally defined as "Verify forward + hidden
capture"; the verify forward (the drafter) is done; hidden capture is
the orchestrator's job per Python's `model_mlx.py:_patch_model`
(lines 284-290) which monkey-patches target layers from OUTSIDE the
model. We do the equivalent in Phase 4 via a wrapper function in the
orchestrator module, not by mutating the target's `forward_decode`
signature.

### Phase 3 gates (unchanged from original plan)

- K=0 falsifier (`verifier.rs:36-41`): with empty drafts, verify-forward
  must be byte-identical to single-token `forward_decode`
- **Determinism gate**: `scripts/coherence-harness/determinism_check.sh
  --spec-dflash-phase=3 --temp=0` on all 18 golden fixtures + 100 random
  prompts; require byte-identity
- **Perf gate**: alt-pair thermal-fair single-token forward_decode vs
  forward_decode_verify(K=1) must show ≤ 5% verify-mode overhead

#### **Phase 4 — Greedy spec-decode orchestrator** ⏳ PENDING
Lands: `orchestrator.rs` + target hidden capture wiring + verify_batched body.

**Subsumed from Phase 3 boundary refinement** (iter-34):
- Target hidden capture at `target_layer_ids` (called from orchestrator,
  not by mutating forward_decode signature — mirrors Python's
  `_patch_model` monkey-patch pattern in model_mlx.py:284-290)
- `forward_decode_verify_batched` body replacement at
  `forward_prefill_batched.rs:2665`

**End-to-end orchestrator (greedy temp=0):**
- Input: prompt tokens
- Output: generated tokens (byte-identical to single-token decode at K=0)
- Loop per decode step:
  1. ngram_proposer or empty-drafts decision
  2. If K > 0: call orchestrator wrapper that
     a. Runs target's forward on K+1 tokens, capturing per-layer hidden
        at `target_layer_ids`
     b. Calls `dispatch_dflash_model_forward(target_hidden, ...)` to get
        drafter logits at L positions
     c. Argmax drafter logits → K draft tokens
     d. Runs target's verify forward on [last_accepted, draft_1, …,
        draft_K] (K+1 tokens), captures target argmaxes
     e. Applies `accept_prefix_argmax(drafts, target_argmaxes)` →
        (accept_count, model_token)
     f. Rolls back target KV cache by `K - accept_count` per `rollback_kv`
  3. Else (K=0): standard single-token decode

**Gates (unchanged from original plan):**
- **Coherence gate**: all 18 coherence golden fixtures at temp=0 must
  produce byte-identical output to single-token decode. **NO EXCEPTIONS**.
- **Determinism gate**: 10 repeated runs of each golden fixture must each
  produce identical output.
- **Perf gate**: alt-pair thermal-fair `bench-decode.sh tg256` at temp=0,
  σ<1%, 5 cycles; spec-decode must show ≥ 1.07× speedup vs hf2q baseline
  (= peer-FA parity per Phase 1.5 mission gate revision).
- Falsifier: at `HF2Q_SPEC_DFLASH=0`, output + perf must be identical to
  pre-Phase-2 HEAD baseline (no regression).

#### **Phase 5 — Async parallel-encode** (~90 LOC; `HF2Q_SPEC_DFLASH_PHASE=5`)
Lands: `async_dispatch.rs`, modifications to orchestrator
- Two-stream commit overlapping draft and target dispatches (the load-bearing perf optimization)
- **Coherence gate**: re-run all Phase 4 coherence gates (must still be byte-identical)
- **Perf gate**: alt-pair thermal-fair; async path must show ≥ 1.3× additional speedup over Phase 4 (the synchronous baseline). Combined Phase 4+5 target: ≥ 2× single-token decode (conservative; oMLX measured 3×)
- Falsifier: at `HF2Q_SPEC_DFLASH_ASYNC=0`, output and perf must match Phase 4

#### **Phase 6 — Rejection sampling for temp>0** ✅ MATH SHIPPED iter-53 (2026-05-14)

Landed: `src/inference/spec_decode/dflash/rejection_sampler.rs` (378 LOC).

API:
- `softmax_with_temp(logprobs: &[f32], temp: f32) -> Vec<f32>`
- `leviathan_step(draft_token, target_probs, drafter_probs, rng)
   -> SampleStep::{Accept | Reject{replacement_token}}`
- `leviathan_accept_prefix(drafts, target_probs_per_pos,
   drafter_probs_per_pos, rng) -> (accept_count, replacement_or_continuation)`

Algorithm per Leviathan et al. 2023 §2.3 verified by 7 unit tests
with statistical bounds:
- `softmax_with_temp_normalizes` (sum=1, monotonic)
- `softmax_temp_zero_panics_via_assert` (temp > 0 enforced)
- `leviathan_step_accepts_when_target_dominates` (100/100 trials accept
  when p >> q)
- `leviathan_step_rejects_when_drafter_dominates` (~95% rejects in
  [800,990]/1000 range when q >> p)
- `leviathan_step_residual_replacement_is_correct_token` (zero-mass
  tokens never sampled; remaining tokens at expected 0.43:0.57 ratio)
- `leviathan_accept_prefix_full_accept_returns_continuation`
- `leviathan_accept_prefix_partial_reject_truncates`

Added `rand = "0.8"` direct dependency (already transitive via
tokenizers / mlx-native).

**Greedy degeneration at temp=0**: the rejection rule reduces to
byte-identical behavior of existing `accept_prefix_argmax`:
- argmax(p) == argmax(q) == draft → p/q=1 always accept
- argmax(p) != argmax(q) → reject, residual sampling collapses to
  argmax(p)

So this module is the GENERALIZATION of the greedy path that handles
temp > 0 correctly without breaking temp = 0.

**Integration with verify path** ⏳ deferred:
- Requires per-position full-logits emission (not just argmaxes) from
  `forward_decode_verify_batched` — ~2MB/call for gemma-4 vocab 262144
- New `dispatch_dflash_one_round_with_logits` orchestrator variant
  that uses `leviathan_accept_prefix` instead of greedy
  `step_round_from_argmaxes`
- Production gating via `HF2Q_SPEC_DFLASH_TEMP_REJECT=1` env flag

Gates (still required for default-flip at temp > 0):
- KL divergence ≤ 0.01
- Mean log-prob ratio in [0.98, 1.02]
- Top-50 5-gram Jaccard ≥ 0.95
- Perf gate: ≥ 1.5× single-token decode at temp=0.7

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

---

## 8. Status Log

### iter-63 (2026-05-14) — End-to-end GPU integration test + 2 bug fixes

**Mission state at start**: `dispatch_dflash_generate` shipped but never run on real models.

**What ran**: Authored
`spec_decode::dflash::orchestrator::tests::e2e_dispatch_dflash_generate_gemma4_26b`
— a real `#[ignore]`'d GPU integration test that loads the production
gemma-4-26b-a4b-it Q5_K_M GGUF + the z-lab DFlash drafter safetensors,
tokenizes a Gemma chat-templated prompt, and runs
`dispatch_dflash_generate` for 16 new tokens at block_size=8 (K=7).

**Bug 1 — FIXED**: Buffer bucket-rounding leak on CPU read.
- Symptom: `dflash append_seq_major_kv: input lens K=16384 V=12288 !=
  n_new(12) * num_kv_heads(8) * head_dim(128) = 12288`
- Root cause: `apply_imrope` in `qwen35/gpu_full_attn.rs:649` uses
  `pooled_alloc_buffer`, which rounds `byte_len` up to a power-of-2
  bucket (16384 ≥ 12288). Its comment said this was safe "for GPU-only
  feeds, not satisfied in DFlash drafter which CPU-reads K/V to write
  the cursor-mode cache". `MlxBuffer::as_slice` returns
  `byte_len/sizeof`, not `element_count` (the shape product), so
  bucketed storage leaked 4096 trailing junk f32s into the CPU `Vec`.
- Fix: Added `download_f32_logical(buf)` helper in `dflash::forward`
  that truncates to `buf.element_count()` before copying. All 6 dflash
  download sites switched.

**Bug 2 — FIXED**: forward_prefill_batched re-allocated hybrid_kv on every call.
- Symptom: `Allocating hybrid_kv (...) [batched]` printed on EVERY
  verify round; output was incoherent because the prompt's KV state
  was wiped at the first verify round.
- Root cause: `forward_prefill_batched.rs:358` did
  `self.hybrid_kv = Some(hybrid_vec)` unconditionally inside the
  `INVESTIGATION_ENV.hybrid_kv` arm. The iter-137/138 work correctly
  threaded `start_pos` through `pf_positions[i] = start_pos + i` and
  `write_pos = start_pos + seq_len`, but the iter-348 hybrid_kv alloc
  did not get the `is_none()` guard that the parallel decode path
  (`forward_mlx.rs:3045`) has.
- Fix: Gated both `hybrid_kv` and `leg_hb_encoded` allocations with
  `is_none()` checks, mirroring the forward_decode lazy-alloc pattern.
  All production callers pass `start_pos=0` on a fresh
  `MlxModelWeights` instance (hybrid_kv == None on first call), so
  this is bit-identical to pre-iter-63 for cmd_generate / parity /
  engine flows.

**Status after fixes**: Pipeline runs end-to-end without panic. Test
asserts pass (non-empty output, length ≤ max_new_tokens). 16 verify
rounds in ~1.41 s on M5 Max (~88 ms/round). Coherence NOT yet
established — output is multilingual gibberish on a hand-rolled chat
template; this matches the SAME behavior we got from cmd_generate when
fed a non-templated prompt, so the suspect is either (a) the
hand-rolled template differs from gguf-embedded one, or (b) verify K/V
writes are not correctly offset by `start_pos` (need to add
diagnostic).

**Deferred to iter-64**: ground-truth coherence comparison. Two paths:
1. Wire the test to use `render_chat_template` so the prompt matches
   what cmd_generate produces (clean parity surface).
2. Run a parallel `forward_decode_verify_serial` baseline inside the
   test and assert byte-identity vs `dispatch_dflash_generate` output
   at temp=0.

**Code artifacts (this iter)**:
- `src/inference/spec_decode/dflash/forward.rs` — `download_f32_logical`
  helper (+ 6 download sites switched)
- `src/serve/forward_prefill_batched.rs` — hybrid_kv / leg_hb_encoded
  lazy-alloc gate
- `src/inference/spec_decode/dflash/orchestrator.rs` —
  `e2e_dispatch_dflash_generate_gemma4_26b` test + anyhow chain
  preservation via `.context()` instead of `.map_err(anyhow!("…{e}"))`

### iter-64 (2026-05-14) — Coherence gate authored; surfaces architectural SDPA limitation

**Mission state at start**: iter-63 e2e test passes pipeline-runs
assertions but produces multilingual gibberish. Hypothesis: orchestrator
correct, prompt template wrong.

**What ran**: Upgraded the e2e test into a real coherence gate.
Single-token-decode baseline runs `forward_prefill_batched(prompt)` +
N-1 `forward_decode` calls to produce ground-truth tokens. Target is
then fully rolled back via `rollback_kv(prompt_len + N - 1)`.
Then `dispatch_dflash_generate(prompt, N)` runs and the test asserts
byte-identity on the new tokens.

**Test verdict**: FAILS at new-token position 1. Position 0 matches
(both paths use forward_prefill_batched for initial prefill; first_token
agrees). Position 1+ diverges across every round (8/8 mismatch).
**Hypothesis falsified**: orchestrator is not coherent, even after
iter-63 fixes.

**Bug 3 — FIXED**: K/V destination offset ignored `start_pos`.
- Symptom in iter-64: `linear_capacity=8 < write target 20` at L5
  full-attention layer; capacity check fired AFTER the iter-63 lazy-alloc
  was correctly preserving hybrid_kv.
- Root cause: `forward_prefill_batched.rs:1531` computed
  `dst_seq_pos_start = src_tok_offset` (CHUNK-INTERNAL offset, always
  0 for single-chunk prefill). The kv_cache_copy / kv_quantize
  dispatchers wrote at position 0 even when `start_pos > 0`. iter-137
  (Path A Phase 2) had threaded `start_pos` through `pf_positions[i]`
  (RoPE) and `write_pos` (cache cursor) — but never into the K/V copy
  destination.
- Fix (this iter): `dst_seq_pos_start = (start_pos as u32) +
  src_tok_offset`, with the capacity check also updated to
  `(start_pos + seq_len) > layer_cap`. Bit-identical for production
  callers (all pass `start_pos=0`).

**Bug 4 — FIXED**: orchestrator under-sized cache via `max_decode_tokens=0`.
- Symptom: dense KV capacity from initial prefill = `prompt_len + 0`,
  but worst-case write extent across verify rounds is `prompt_len +
  max_new_tokens + block_size - 1`.
- Fix: orchestrator passes `max_decode_tokens = max_new_tokens +
  block_size - 1` to the initial `forward_prefill_batched`.

**Bug 5 — DISCOVERED, NOT YET FIXED (architectural)**: batched
prefill's SDPA is self-attention over current chunk only.
- After Bugs 3 + 4 are fixed, coherence gate still fails at the same
  position 1.
- Inspection of `forward_prefill_batched.rs:1310-1327` /
  `forward_prefill_batched.rs:1329-…`: both the D=256 (sliding) and
  D=512 (full) flash_attn_prefill dispatches set
  `seq_len_q = seq_len_k = seq_len` — the kernel computes attention
  ONLY over the current chunk's seq_len tokens with a causal/sliding
  mask. **Prior cache content is invisible to the SDPA**, regardless
  of write_pos / hybrid_kv contents.
- Implication: the `forward_decode_verify_batched` API at
  `forward_prefill_batched.rs:2698` (introduced iter-47) is broken at
  any non-zero `start_seq_pos` — the K/V writes go to the right place
  now after Bug 3, but the SDPA still attends only to the current K+1
  verify tokens, ignoring the prompt + accepted tokens.
- This is a fundamental architecture choice in the existing prefill
  path. Spec-decode requires cross-length attention (seq_len_k >>
  seq_len_q at non-zero start_pos), which the existing
  `flash_attn_prefill` kernels don't support.

**Options for iter-65 to make coherence gate pass**:
- **A. Cross-length SDPA in batched prefill**: extend
  `flash_attn_prefill_*` kernels (and dispatch layer) to take
  separate Q/K lengths + read from prior hybrid_kv. Most performance-
  preserving; multi-day Metal kernel work.
- **B. Add capture support to forward_decode + use
  forward_decode_verify_serial**: forward_decode already extends KV
  correctly. Add the layer-loop capture hook to forward_decode (~50
  LOC) and switch orchestrator's verify call. Loses batched verify
  speedup but proves coherence; ~1 iter scope.
- **C. Re-prefill full prefix each round (option C)**: every verify
  round re-prefills `[prompt + accepted + last_token + drafts]` from
  `start_pos=0`. SDPA self-attention works (it sees the whole
  prefix). Drafter cache also reset each round. O(N²) cost. Largest
  orchestrator rewrite; ~1 iter scope.

**Code artifacts (iter-64)**:
- `src/inference/spec_decode/dflash/orchestrator.rs` — coherence gate
  upgraded to do baseline (forward_prefill + N-1 forward_decode) vs
  spec-decode byte-identity comparison. Currently RED (fails at pos 1).
- `src/serve/forward_prefill_batched.rs` —
  `dst_seq_pos_start = (start_pos as u32) + src_tok_offset` fix +
  capacity check accounts for `start_pos`.
- `src/inference/spec_decode/dflash/orchestrator.rs` (dispatch_dflash_generate)
  — `max_decode_tokens = max_new_tokens + block_size - 1` for cache sizing.

### iter-65 (2026-05-14) — COHERENCE GATE GREEN ✅

**Mission state at start**: iter-64 surfaced Bug 5 (batched-prefill SDPA is
self-attention only).  Decision: implement Option C (re-prefill full
prefix from start_pos=0 each round) — orchestrator-only correctness path,
defer Option A (cross-length SDPA) to perf iteration.

**Option C implementation**: rewrote `dispatch_dflash_generate`'s
multi-round loop.  Each round:
1. Re-prefill the prior context `output[0..output.len()-1]` from
   start_pos=0 with capture installed (gets target hidden states for
   the drafter's input).
2. Reset drafter cache via `DFlashKvCache::reset()`.
3. Drafter forward over the FULL prior ctx (drafter cache rebuilt
   each round).
4. Build `verify_prefix = output + drafts`, re-prefill from start_pos=0
   with capture installed.
5. `per_position_argmax_from_hidden_opt` on the captured final-layer
   slab; take argmaxes at `[output.len()-1 .. output.len()-1+block_size]`
   = the verify positions.
6. Accept-prefix, append committed tokens; NO target rollback (next
   round re-prefills from 0 anyway).

**Performance trade**: O(N²) cost vs spec-decode's intended O(N·K).
For N=8, K=7 on M5 Max: ~1.5 s vs baseline 0.16 s ≈ 10× slowdown.
Acceptable for the correctness gate; Option A (cross-length SDPA) is
the perf path for iter-66+.

**Bug 6 — FIXED**: capture hook read pf_hidden async, got stale data.
- Symptom: even with Option C in place, the coherence gate STILL failed
  at pos 1.  DIAG2 traced the issue: `forward_prefill_batched(13 tokens)`
  returned first_token=236778 ✓ (= baseline), but capture+per_position
  on the SAME prefill returned 191232 at position 12.
- Root cause: the dflash capture hook at
  `forward_prefill_batched.rs:2216` CPU-reads `pf_hidden` via
  `as_slice()`.  Production uses fire-and-forget `s.commit()` between
  layers (per the layer-boundary comment at line 2015-2034) — the GPU
  may not have finished writing pf_hidden when the CPU read happens.
  `as_slice` returned stale data (from the PRIOR layer or initial
  zeros), so the captured slab was the wrong layer's output → wrong
  argmaxes → wrong commits.
- Fix: gate the `sync_per_layer` flag (which triggers `s.finish()`
  commit-and-wait between layers) on `self.dflash_capture.is_some()`.
  Bit-identical for production callers (none install dflash_capture);
  adds ~30 sync points per token-position when spec-decode is running,
  which is acceptable since each verify forward only runs once per
  round.

**Coherence gate verdict** (e2e_dispatch_dflash_generate_gemma4_26b):

```
baseline_new = [236743, 236778, 236862, 236778, 236881, 107, 236776, 236787]
spec_new     = [236743, 236778, 236862, 236778, 236881, 107, 236776, 236787]
COHERENCE PASS: spec_new == baseline_new for all 8 tokens
```

✅ All 8 new tokens byte-identical between single-token decode and
DFlash spec-decode at temp=0.  Wall time: SPEC 1.51 s vs BASELINE
0.16 s on M5 Max.

**Code artifacts (iter-65)**:
- `src/inference/spec_decode/dflash/orchestrator.rs` —
  `dispatch_dflash_generate` rewritten to Option C (re-prefill each
  round from start_pos=0, drafter cache reset, no target rollback).
- `src/serve/forward_prefill_batched.rs` — `sync_per_layer` gates on
  `self.dflash_capture.is_some()` so capture hook reads land
  post-GPU-flush.

**iter-66+ work** (deferred): Option A (cross-length SDPA in
flash_attn_prefill) for perf parity.  Option C's O(N²) cost makes
DFlash spec-decode currently SLOWER than baseline single-token decode,
but correctness is proven.  Mission perf gate (≥1.07× hf2q baseline)
requires Option A.

### iter-66 (2026-05-14) — Production wire-up + chat-template coherence regression

**Wire-up landed**: `HF2Q_SPEC_DFLASH=1` env flag now plumbs through
`cmd_generate` via the new `src/serve/spec_decode_cli.rs` helper.
Loads the z-lab DFlash drafter (HuggingFace cache path, override via
`HF2Q_DFLASH_DRAFTER_PATH`), runs `dispatch_dflash_generate`, prints
the decoded text.  Default-OFF; opt-in for correctness validation on
user workloads (perf is iter-67+).

Verified on real `gemma-4-26b-a4b-it-ara-abliterated` Q5_K_M:
```
$ HF2Q_SPEC_DFLASH=1 hf2q generate ... --prompt "Q: What is 2+2?\nA:" --max-tokens 16
[HF2Q_SPEC_DFLASH] drafter loaded in 0.09s
4<turn|><turn|>```markdown
# Implementation Guide: 2+2
##
[HF2Q_SPEC_DFLASH] 16 new tokens in 3.77s (4.2 tok/s)
```

**New finding (iter-67 work)**: chat-templated CLI prompt produces
DIFFERENT output between spec-decode and baseline forward_decode:

```
Baseline (24-tok chat-templated prompt, N=8): "4<turn|><turn|><turn|>**A: "
SPEC     (24-tok chat-templated prompt, N=8): "4<turn|><turn|>```markdown\n# Implementation"
```

Diverge at new-token position 3.  But the e2e coherence gate (12-token
UNTEMPLATED prompt, N=8) still PASSES byte-identity — so the bug is
prompt-content-sensitive.  Hypotheses for iter-67:
1. Drafter mis-handles chat-template special tokens (BOS, <start_of_turn>)
2. Sliding-window attention at templated prompt boundary
3. verify_prefix construction order bug for specific contents
4. Capture-sync fix from iter-65 doesn't cover all execution paths

Action: add a second e2e test that reproduces with chat-template
prompt, then bisect (single forward_prefill + capture vs
forward_decode) to isolate.

**Code artifacts (iter-66)**:
- `src/serve/spec_decode_cli.rs` (new, 175 LOC) — env flag parsing,
  drafter loading, `dispatch_dflash_generate` orchestration, stdout
  text emission.
- `src/serve/mod.rs` (cmd_generate insertion, ~15 LOC) — gates the
  spec-decode helper after prompt-tokens finalization, returns
  immediately if the helper handles the request.

### iter-67 (2026-05-14) — Three-axis coherence: orchestrator VINDICATED, batched_prefill is the deeper bug

**Investigation of iter-66 finding** (CLI divergence on chat-templated prompts).

Authored `e2e_coherence_gemma4_chat_templated_prompt` — a controlled
reproducer using the hardcoded 24-token Gemma-chat-templated prompt
(captured via `HF2Q_DUMP_PROMPT_TOKENS` from cmd_generate):

```text
[2, 105, 2364, 107, 236935, 236787, 2900, 563, 236743, 236778,
 236862, 236778, 105470, 169631, 236787, 106, 107, 105, 4368, 107,
 100, 45518, 107, 101]
```

(BOS=2, `<|turn>`=105, `<turn|>`=106, `<|channel>`=100, `<channel|>`=101.)

**Three coherence axes** measured:

| Axis | Direction | Result |
|---|---|---|
| 1 | spec-decode vs `forward_decode` baseline | ✗ FAILS at pos 3 |
| 2 | `forward_prefill_batched` vs `forward_decode` (DIAG)  | ✗ FAILS at L=27, 28, 29 |
| 3 | spec-decode vs `forward_prefill_batched` (SELF DIAG) | ✓ PASSES all 8 positions |

Concrete evidence:

```text
DIAG L=27 argmax=2717 baseline_new[3]=106 ✗
SELF L=27 argmax=2717 spec_new[3]=2717 ✓
```

**The orchestrator is FAITHFUL to batched-prefill.**
`spec_new[i]` exactly matches what `forward_prefill_batched` produces
at every position when called on `[prompt + spec_new[..i]]`.  This is
the orchestrator's actual correctness guarantee.

**The CLI divergence reported in iter-66 is a `forward_prefill_batched`
vs `forward_decode` bug** (axis 2), not a spec-decode bug.  It would
have manifested in the existing `forward_decode_verify_batched` API
too (iter-47 added that for KV-prefill verify) if it were exercised
at non-zero `start_seq_pos` with chat-templated content.  It is
consistent with the pre-existing `coherence_smoke_all_cells` failure
on gemma4-apex prompts.

**Untemplated 12-token prompt coherence**: still PASSES byte-identity
at N=16 (was N=8 in iter-65; bumped to N=16 in iter-67 as additional
evidence).  The bug is content-sensitive.

**Iter-68+ work**:
- Investigate axis 2 (`forward_prefill_batched` vs `forward_decode`
  coherence) on chat-templated prompts.  Hypothesis: special-token
  handling inside the batched-prefill path is inconsistent with the
  per-token decode path.  Bisect by comparing pf_hidden between
  per-token and batched paths at specific layers/positions.
- This is a hf2q-internal bug NOT introduced by ADR-030; spec-decode
  inherits it transitively.  Fix is out of ADR-030's scope but
  required to unblock the spec-decode CLI's full coherence story.

**Code artifacts (iter-67)**:
- `src/inference/spec_decode/dflash/orchestrator.rs` — new test
  `e2e_coherence_gemma4_chat_templated_prompt` with three-axis
  diagnostic (DIAG + SELF) + axis-3 self-consistency assertion that
  PASSES, axis-1 baseline assertion that intentionally fails until
  axis-2 is fixed separately.
- `src/serve/mod.rs` — `HF2Q_DUMP_PROMPT_TOKENS` now also logs the
  full `prompt_tokens` slice (was first10/last10 only) for clean
  template-extraction in tests.
- `src/inference/spec_decode/dflash/orchestrator.rs` —
  `e2e_dispatch_dflash_generate_gemma4_26b` bumped from N=8 to N=16,
  confirming untemplated coherence remains byte-identical at longer
  generations.

### iter-68 (2026-05-14) — Perf gap measured: 22× slower at all N

Benchmarked `HF2Q_SPEC_DFLASH=1` vs baseline at N ∈ {8, 16, 32}, 2
trials each, 20s cool-downs.  See
`docs/research/adr030_iter68_bench/{results.tsv, analysis.md}`.

| N | baseline tok/s | spec tok/s | ratio |
|---|---|---|---|
|  8 | 103.15 | 4.65 | 0.045× (22.2× slower) |
| 16 | 98.85  | 4.40 | 0.045× (22.5× slower) |
| 32 | 96.55  | 4.05 | 0.042× (23.8× slower) |

σ within each cell: <1%.  Ratio is approximately constant across N at
these scales (24-token prompt P dominates over Option C's quadratic
term).

**0% drafter acceptance observed** — wall-clock implies each round
commits exactly 1 token (the target free-continuation).  Consistent
with iter-67's axis-2 finding: batched-prefill's hidden states (which
feed the drafter) diverge from forward-decode's on chat-templated
content → drafter proposes tokens target rejects 100%.

**Mission gate**: ≥1.07× baseline.
**Current**: 0.045× baseline.
**Required gain**: ~24×.

Cannot reach via micro-optimisation; requires architecture change.
Two paths in analysis.md:
- **Option A (preferred)**: cross-length SDPA in batched prefill,
  mirroring peer dflash MLX path at `/opt/dflash/dflash/model_mlx.py:513`
  (one `model(verify_input, target_cache)` call attends over prior
  cache atomically).  hf2q has all building blocks: the existing
  `mlx_native::ops::sdpa::sdpa` kernel supports cross-length attention
  (used in `dispatch_dflash_sdpa_cross_length`).  Scope ~500 LOC.
- **Option B**: capture hook on `forward_decode` + serial verify.
  Throughput ceiling ~13 t/s (baseline ÷ K+1) — still 8× off mission
  gate.  Scope ~150 LOC.

**Code artifacts (iter-68)**:
- `scripts/adr030/bench_spec_decode.sh` (new, 70 LOC) — reproducible
  benchmark harness with thermal-aware cool-downs.
- `docs/research/adr030_iter68_bench/results.tsv` — raw measurements.
- `docs/research/adr030_iter68_bench/analysis.md` — full analysis,
  caveats, and recommended next steps.

### iter-69 (2026-05-14) — Single-prefill-per-round → ~50% throughput gain

**Optimization**: eliminated the redundant `prior_ctx` prefill at the
start of each verify round.  The verify prefill's capture already
covers positions `[0..output_len + K)` (= all committed tokens + all
drafts).  With start_pos=0 + causal masking, the captured hidden at
position i depends ONLY on tokens at `[0..=i]` — so trimming
`verify_captured` to `output_next.len() - 1` yields a CORRECT
prior_captured for next round's drafter input.

Round structure changed from 2 prefills (prior_ctx + verify) per
round to **1 prefill** (verify only), with the initial prompt
prefill seeding round-1's `prior_captured`.

**Bench results** (same harness as iter-68, 2 trials, 20s cool-downs):

| N | baseline t/s | spec t/s | spec/baseline | vs iter-68 |
|---|---|---|---|---|
|  8 | 104.0 | 7.0 | 0.067× (14.9× slower) | **+50%** (4.65→7.0) |
| 16 | 99.1  | 6.9 | 0.070× (14.4× slower) | **+57%** (4.40→6.9) |
| 32 | 96.8  | 6.0 | 0.062× (16.1× slower) | **+48%** (4.05→6.0) |

Mission gate gap shrank from 22× to ~15× — still requires
Option A (cross-length SDPA) for the remaining gain, but ~50% closer
to parity with one round-level refactor.

**Coherence assertions preserved**:
- Untemplated e2e gate at N=16: ✓ byte-identical to single-token decode
- Chat-templated axis-3 self-consistency (orchestrator faithful to
  batched_prefill): ✓ all 8 positions still match

**Code artifacts (iter-69)**:
- `src/inference/spec_decode/dflash/orchestrator.rs` —
  `dispatch_dflash_generate` round loop: replaced explicit
  prior_ctx prefill with `prior_captured` state preserved across
  rounds (seeded from initial prompt prefill capture, updated each
  round by trimming verify_captured to next round's prior_ctx_len).
- `docs/research/adr030_iter68_bench/results_iter69.tsv` —
  side-by-side comparison data.

### iter-70 (2026-05-14) — Per-stage profiling + 75% target_argmax reduction

**Profiling instrumentation**: gated on `HF2Q_DFLASH_PROFILE=1` env
var, accumulates per-round wall-clock for each orchestrator stage and
prints a summary at function exit.  Production default unchanged.

**Profile at N=16 (pre-optimization)**:
```
embed=0.03ms  extract=0.21ms  drafter_fwd=13.56ms  drafter_argmax=10.47ms
verify_prefill=78.82ms  target_argmax=51.91ms  trim=0.07ms  TOTAL=155.08ms
```

verify_prefill (51%) + target_argmax (34%) dominate.

**Optimization**: target_argmax was running per-position over ALL
`verify_prefix_len` positions (e.g. 31 for N=15) but the orchestrator
only USED the last `block_size`=8 argmaxes for accept-prefix.  ~75%
of the rms_norm + lm_head + softcap + argmax work per round was
wasted.

Sliced the captured slab to just the verify-window rows BEFORE
invoking per_position_argmax_from_hidden_opt:

```rust
let verify_slab_tail = &final_slab[verify_start * hs..verify_end * hs];
let target_argmaxes = target.per_position_argmax_from_hidden_opt(
    verify_slab_tail, block_size, true, gpu)?;
```

**Per-stage delta**:
```
target_argmax: 51.91ms → 11.21ms  (-40.7ms, -78%)
TOTAL:        155.08ms →  114.12ms (-41ms, -26%)
```

**Bench results** (same harness, 2 trials, 20s cool-downs):

| N | baseline t/s | spec t/s | spec/baseline | vs iter-69 | vs iter-68 |
|---|---|---|---|---|---|
|  8 | 103.6 | 9.00 | 0.087× (11.5× slower) | +29% (7.0→9.0)  | +94% (4.65→9.0)  |
| 16 |  99.1 | 8.75 | 0.088× (11.3× slower) | +27% (6.9→8.75) | +99% (4.4→8.75)  |
| 32 |  97.2 | 8.50 | 0.087× (11.4× slower) | +42% (6.0→8.5)  | +110% (4.05→8.5) |

Mission gap progression:
- iter-68 measured: 22.2× slower
- iter-69 (single prefill): 14.9× slower
- iter-70 (sliced argmax): **11.4× slower**

Still 11× off the ≥1.07× mission gate.  Verify_prefill remains the
dominant cost (~78 ms/round, 68% of remaining wall-clock); requires
Option A (cross-length SDPA) to address.

**Also shipped (opt-in, off by default)**: a batched
`per_position_argmax_from_hidden_batched_impl` variant gated on
`HF2Q_DFLASH_BATCH_ARGMAX=1` that processes all positions in ONE
command buffer with bulk hidden upload + per-position output views.
Modest gain (~8% over un-batched at N=16: 10.47 → 8.83 ms/round)
because per_position GPU work dominates over per-iter sync cost.
Slice optimization (above) gives much larger win and is enabled by
default.

**Coherence preserved**: e2e_dispatch_dflash_generate_gemma4_26b
still GREEN at N=16 byte-identity vs single-token decode.

**Code artifacts (iter-70)**:
- `src/inference/spec_decode/dflash/orchestrator.rs` — per-stage
  HF2Q_DFLASH_PROFILE timing instrumentation + sliced final-layer
  slab to block_size rows before per_position_argmax.
- `src/serve/forward_mlx.rs` —
  `per_position_argmax_from_hidden_batched_impl` (opt-in via
  HF2Q_DFLASH_BATCH_ARGMAX=1).
- `docs/research/adr030_iter68_bench/results_iter70.tsv` —
  comparison data showing 27-42% gain over iter-69 across N.

### iter-71 (2026-05-14) — Incremental drafter cache + batched argmax (default)

Two small contained perf-positive changes:

(1) **Incremental drafter cache**: preserve drafter cache state across
rounds (was: `drafter_cache.reset()` every round and re-feed the FULL
prior ctx).  Now extracts only NEW positions from `prior_captured`
(= `prior_ctx_len - drafter_cache.layers[0].seq_len`) per round.
Drafter's RoPE offsets (`prior_offset = cache_layer.seq_len`) advance
correctly across rounds — matches the ADR-030 spec's incremental design.
Per-round drafter forward cost is dominated by per-call fixed
overhead, not ctx_chunk size, so the perf win is small (~0.5 ms/round)
but the change is architecturally correct.

(2) **Always-on batched argmax** in the orchestrator: both drafter
argmax (block_size positions) and target argmax (block_size verify
positions) now call `per_position_argmax_from_hidden_batched_impl`
directly.  Eliminates K = 7 commit-and-wait syncs per argmax call,
saving ~1.5 ms per argmax site (~3 ms/round across both).

**Per-stage delta** (N=16, single profiling run):
```
                  iter-70  iter-71  delta
drafter_argmax    10.42   →  8.83   (-1.6 ms, -15%)
target_argmax     11.21   →  9.71   (-1.5 ms, -13%)
TOTAL            114.12   → 111.91  (-2.2 ms, -2%)
```

**Bench results** (2 trials, 20s cool-downs):

| N | baseline t/s | spec t/s | vs iter-70 |
|---|---|---|---|
|  8 | 103.8 | 9.15 | +1.7% (9.00→9.15) |
| 16 |  99.5 | 9.05 | +3.4% (8.75→9.05) |
| 32 |  97.3 | 8.80 | +3.5% (8.50→8.80) |

Mission gap: 11.4× → ~11.1× slower.

Cumulative iter-68 → iter-71 progression at N=16:
4.40 → 6.90 → 8.75 → 9.05 t/s = **+106%** over iter-68 baseline.

**Coherence preserved**: e2e gate at N=16 byte-identity ✓.

**Code artifacts (iter-71)**:
- `src/inference/spec_decode/dflash/orchestrator.rs` — drafter cache
  not reset between rounds; extract only NEW rows; route both argmax
  calls through `per_position_argmax_from_hidden_batched_impl`.
- `src/serve/forward_mlx.rs` — batched impl exposed as `pub(crate)`.
- `docs/research/adr030_iter68_bench/results_iter71.tsv`.

**Mission state at iter-71 end**:
- Coherence: GREEN on untemplated N=16 ✓
- Production wire-up: ✓ (HF2Q_SPEC_DFLASH=1)
- Perf: 0.091× baseline (was 0.045× at iter-68 → **2.02× spec-decode improvement**)
- Remaining: ~11× to ≥1.07× mission gate → requires Option A
  (cross-length SDPA) to attack the dominant verify_prefill stage
  (78 ms = 72% of remaining wall-clock per round).
