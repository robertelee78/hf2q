# ADR-039 — HASS + Hydra speculative decoding (SOTA beyond EAGLE-3)

- **Status**: 📋 DRAFT — Design proposal 2026-05-23, awaits operator approval. Builds on ADR-037 Phase E9 (the "HASS/Hydra IN-SCOPE" commitment) + ADR-038 (Gemma 4 + Qwen 3.5 EAGLE-3 verifier infrastructure that this ADR's drafters plug into).
- **Date**: 2026-05-23
- **Supersedes**: nothing
- **Related**: ADR-037 (EAGLE-3 baseline), ADR-038 (Gemma 4 EAGLE-3 enablement), ADR-034 (spec-decode end-to-end)
- **Author note**: ADR-037 v1 chose EAGLE-3 over Medusa because *"better to do the right thing 1st"* per operator. This ADR continues that principle: HASS + Hydra are the current published SOTA for speculative decoding (Hu et al. 2024 + Ankner et al. 2024). EAGLE-3 was the foundation; HASS + Hydra are the destination.

> ## Mantra (verbatim from `~/Documents/mantra.txt`)
>
> *DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.*

---

## 1. Why (the problem)

### 1.1 Where EAGLE-3 plateaus

EAGLE-3 (ADR-037 Phase E1-E8) achieves typical acceptance length L≈1.5-2.5 tokens/step on dense LLMs at temp=0 (per RedHatAI's published benchmarks for `gemma-4-31B-it-speculator.eagle3`). This is the ADR-037 AC-4.10 bar.

Two structural limitations cap the gain:

1. **Single-head drafter** — EAGLE-3 predicts ONE next token per drafter forward pass. Tree expansion (Phase E4a's dynamic best-first) gets multiple candidates per depth by running the drafter multiple times. Latency: `depth × drafter_forward_cost`.

2. **Training-inference distribution mismatch** — EAGLE-3 drafter is trained to imitate target's next-token logits, BUT at inference time the drafter sees its OWN hidden state (not target's) at depth ≥ 1. This causes acceptance rate to decay with depth.

### 1.2 What HASS adds

**HASS (Harmonized Acceleration via Speculative Sampling) — Hu et al., arXiv:2408.15766 (2024)**: extends EAGLE-3 training with two losses:

- **Feature consistency loss** (`L_feat`): KL-divergence between drafter's depth-≥1 hidden state and what target's hidden state WOULD be if it processed the drafter's prediction. Closes the train-inference gap.
- **Token consistency loss** (`L_tok`): cross-entropy between drafter's depth-≥1 token distribution and target's distribution at that position. Sharpens depth ≥ 2 predictions.

Empirical (HASS paper Table 3, Llama-3-8B + Vicuna-7B): **L ≈ 3.0-4.5** (vs EAGLE-3's 1.5-2.5) at temp=0 — **1.5-2.0× extra speedup stacked on EAGLE-3**.

**Critical**: HASS is a TRAINING-side modification. Inference architecture is unchanged from EAGLE-3 — same drafter shape, same forward chain. **Means a HASS-trained drafter loads through hf2q's existing `Eagle3DrafterTensors` schema with zero verifier code changes.** Only the .safetensors file differs (different weight values from different training loss).

### 1.3 What Hydra adds

**Hydra: Sequential dependency in draft heads — Ankner et al., arXiv:2402.05109 (2024)**: replaces single-head drafter with N sequential heads, each conditioned on prior heads' outputs:

- Head 1 predicts position depth=1 from target hidden state
- Head 2 predicts depth=2 conditioned on head 1's predicted embedding + target hidden
- Head k predicts depth=k conditioned on heads 1..k-1

Versus Medusa-2 (parallel heads), Hydra's sequential conditioning achieves stronger depth-2+ predictions because each head sees the actual sampled tokens, not just hidden state.

Empirical (Hydra paper Table 2): **L ≈ 3.2-3.6** at temp=0 with 4 heads vs Medusa-2's 2.0-2.5 — **1.5× over Medusa-2, comparable to HASS on its own**.

### 1.4 Why both HASS + Hydra ("HASS-Hydra")

HASS and Hydra address ORTHOGONAL gaps:

- HASS fixes the train-inference distribution mismatch (loss-side).
- Hydra adds depth-aware sequential conditioning (architecture-side).

Combining them — train Hydra-architecture drafter with HASS consistency losses — should compound. No published paper has shipped this exact combo yet (as of January 2026); per the mantra *"better to do the right thing 1st"*, this ADR commits to HASS-Hydra as a single integrated technique rather than shipping HASS alone first.

Expected compound speedup: **L ≈ 4.5-5.5** if both improvements stack as published (open empirical question for hf2q's Q4_K_M serving regime).

---

## 2. Where (the integration surface)

### 2.1 Foundation reused from ADR-037 + ADR-038

| Component | Source | Reuse status |
|---|---|---|
| `Eagle3HiddenCollector` (multi-layer hidden capture) | ADR-037 E3a | ✅ unchanged |
| `Eagle3DrafterConfig` / `Eagle3Weights` (loader schema) | ADR-037 E3b | extended for Hydra (N-head variant) |
| `dispatch_eagle3_*` (14-stage drafter forward) | ADR-037 E4b | EAGLE-3 single-head; Hydra adds head-loop wrapper |
| `DrafterKvCache` (per-drafter K/V cache) | ADR-037 E5b | ✅ unchanged (per-head cache stacking) |
| `dispatch_qwen35_tree_verify_attention` / `dispatch_gemma4_tree_verify_attention` | ADR-037 E6 / ADR-038 G4-CFA-1 | ✅ unchanged (verifier-side) |
| `Eagle3Orchestrator` / `Gemma4Eagle3Orchestrator` | ADR-037 / ADR-038 | extended to dispatch `Hydra` drafter family |
| `forward_tree_verify_gpu` / `forward_tree_verify_gpu_with_cache` | ADR-037 / ADR-038 | ✅ unchanged (target-side runs same) |
| `core::tokenizer_adapter::tokenize_with_bos_eos_from_gguf` | ADR-038 G4-CFA-5e | ✅ unchanged |

**Key invariant**: HASS-Hydra reuses the entire verifier path. Only the drafter (training + inference) changes.

### 2.2 New components

| Component | Location (proposed) | LOC est. |
|---|---|---|
| `HydraDrafterConfig` (N-head architecture knobs) | `src/inference/spec_decode/hydra/config.rs` | ~200 |
| `HydraDrafterTensors` (N-head weight schema) | `src/inference/spec_decode/hydra/weights.rs` | ~300 |
| `dispatch_hydra_head_forward` (per-head 14-stage chain) | `src/inference/spec_decode/hydra/forward.rs` | ~500 |
| `HydraDrafter` impl `Drafter` trait (sequential N-head dispatch) | `src/inference/spec_decode/hydra/drafter_gpu.rs` | ~400 |
| `DrafterFamily { Eagle3, Hydra }` enum + dispatch in `Eagle3Orchestrator` | `src/inference/spec_decode/eagle3_orchestrator.rs` | ~80 |
| `DrafterFamily::Hydra` in `Gemma4Eagle3Orchestrator` | same | ~80 |
| `HF2Q_SPEC_HYDRA` env flag + serve wiring | `src/serve/spec_decode_cli.rs` | ~50 |
| HASS training scripts (Python sidecar OR Rust trainer) | scripts/training/hass_hydra/ (TBD §4) | ~1500-3000 |
| Unit tests (Hydra forward parity vs Eagle3 single-head + N-head sequential) | inline `#[cfg(test)] mod tests` per file | ~600 |

**Total Rust LOC**: ~2200 (inference side) + ~1500-3000 training (location TBD §4).

### 2.3 What does NOT change

- mlx-native kernels: zero new Metal kernels needed. Hydra's per-head forward reuses ALL existing dispatchers (`apply_linear_projection_f32`, `dispatch_eagle3_*`, `tree_attention`).
- Target model serving (Gemma 4 31B, Qwen 3.5/3.6 *) — unchanged.
- GGUF format — drafter weights are HF safetensors (separate file), not GGUF.
- CLI surface — `hf2q serve` and `hf2q generate` gain `HF2Q_SPEC_HYDRA=1` env flag analogous to existing `HF2Q_SPEC_EAGLE3=1`.

---

## 3. Architecture decisions

### 3.1 Hydra head topology

**Decision**: N=4 heads as the default (matches published Hydra config), with `num_heads` as a `HydraDrafterConfig` field for experimentation. Each head shares the same transformer body (Q/K/V/MLP) but has its own `lm_head` and an input projection that consumes prior heads' embeddings.

**Alternatives considered**:
- N=8 heads (matches HASS-paper tree depth) — rejected: latency cost exceeds acceptance gain past N=5 per Hydra paper §4.2.
- Single shared lm_head — rejected: published Hydra ablates this and shows per-head lm_head adds 0.3 to L.

### 3.2 Sequential vs parallel head dispatch

**Decision**: Sequential (Hydra-paper canonical). Head k waits for head k-1's argmax before forward.

**Alternatives considered**:
- Parallel heads (Medusa-2 style) — rejected: weaker depth-2+ predictions per §1.3.
- Batched-sequential (run heads as batch=N with causal mask) — DEFERRED to Phase 2 optimization once correctness is proven. Single-head-at-a-time is simpler and easier to verify.

### 3.3 KV cache strategy

**Decision**: Single `DrafterKvCache` shared across all N heads. Head k appends its predicted K/V to the same cache that head k+1 then reads. Matches `forward_tree_verify_gpu_with_cache` semantics from ADR-038 G4-CFA-5c.

**Why**: Each head's prediction conditions on the prior heads' tokens via attention, so all heads see the same K/V history. Separate per-head caches would require N× memory + duplicate the prior-context K/V writes.

### 3.4 Training: HASS losses + Hydra heads jointly

**Decision**: Train all N heads simultaneously with HASS feature consistency + token consistency losses. Loss = sum over heads k of (L_token_k + λ_feat × L_feat_k) where `λ_feat` is a tunable hyperparameter.

**Alternative considered**: Train heads sequentially (head 1 first, freeze, then head 2, ...) — rejected: published HASS-style joint training converges 2-3× faster per HASS §5.

### 3.5 Training infrastructure: Python sidecar vs Rust

**Decision DEFERRED to §4** — needs operator input. Three options:

| Option | Pros | Cons |
|---|---|---|
| **Python sidecar** (PyTorch / Transformers) | Fastest to implement; reuses HF training stack; Hydra paper authors' code is PyTorch | Adds Python toolchain dependency; not pure-Rust as preferred per memory |
| **Pure Rust trainer via `candle`** | Pure Rust; no Python dep; matches mantra | candle's training is less mature; multi-week extra implementation cost |
| **Pure Rust trainer in mlx-native** | Tightest integration with hf2q's serving stack | Multi-month extra cost; mlx-native lacks gradient ops today |

Recommendation pending §4 confirmation: **Option 1 (Python sidecar)** for fastest validation, with explicit ADR commitment to port to pure Rust once HASS-Hydra is empirically validated on Gemma 4 31B.

---

## 4. Open questions (operator decisions needed)

1. **Training infrastructure**: Python sidecar (fast) OR pure Rust via candle/mlx-native (slow but matches mantra)?
2. **Compute commitment**: HASS-Hydra training is ~2× EAGLE-3's H100 cost (joint N-head). Estimate ~2 weeks H100 for Gemma 4 31B + Qwen 3.6 27B-A3B + Qwen 3.5 35B-A3B combined. Acceptable?
3. **Phasing**: ship Hydra-only first (validates the architecture side) then add HASS losses, OR ship combined HASS-Hydra from the start? The latter is one fewer training run but doesn't validate where the gain comes from empirically.
4. **N (heads)**: default to 4 (Hydra paper) or experiment with 5-6 for higher acceptance lengths at the cost of latency?
5. **Drafter base model**: same drafter shape as EAGLE-3 (5376 hidden for Gemma 4 / 1024 for Qwen) or experiment with deeper drafter (literature suggests N-head Hydra benefits from slightly deeper drafter)?

---

## 5. Acceptance criteria (proposed — to be locked after §4 resolution)

### AC-1: Hydra inference parity vs EAGLE-3 at N=1 (foundation)

A Hydra drafter configured with `num_heads=1` MUST produce byte-identical token sequences to the equivalent EAGLE-3 drafter on the same prompt + same target model. Proves Hydra's single-head case is a strict superset of EAGLE-3.

### AC-2: Hydra N=4 acceptance-length floor

On Gemma 4 31B Q4_K_M + Qwen 3.6 27B-A3B Q4_K_M, untrained random-init Hydra drafter (N=4) MUST produce L ≥ EAGLE-3's random-init baseline on the same test prompt. Verifies that the additional heads don't break expansion at random-init (catches structural bugs early, before training cost).

### AC-3: HASS-Hydra trained-drafter acceptance length

On Gemma 4 31B Q4_K_M + Qwen 3.6 27B-A3B Q4_K_M at temp=0:
- L ≥ 3.0 (acceptance length, mean over 100 prompts of length 64+).
- Acceptance rate ≥ 60% at depth ≥ 2.
- Throughput improvement ≥ 1.5× over EAGLE-3 at same compute budget.

### AC-4: zero verifier-side regression

After Hydra dispatch lands, ALL existing 39/39 g4_cfa* + 31/31 qwen35_tree_verify tests MUST still PASS. Hydra is additive to the verifier infrastructure.

### AC-5: serve-time switching

`HF2Q_SPEC_HYDRA=1 HF2Q_HYDRA_DRAFTER_PATH=/path/to/hydra-drafter/` MUST cause `hf2q serve` to dispatch Hydra instead of EAGLE-3. Missing flags fall back to standard decode (matching existing `HF2Q_SPEC_EAGLE3` behavior at spec_decode_cli.rs:561).

### AC-6: HASS loss correctness (training-side)

The HASS feature consistency loss implementation MUST match Hu et al. 2024 §3.2 within numerical tolerance. Verifiable via a small fixture (10-token sequence, 1-layer toy drafter) comparing our loss vs a reference implementation (e.g., the HASS paper's released code if Python sidecar).

---

## 6. Sequencing (proposed)

| Phase | Scope | Estimated effort | Blocking |
|---|---|---|---|
| **H1 — Hydra config + weight schema** | `HydraDrafterConfig` + `HydraDrafterTensors` + manifest validation (extends ADR-037 E3b pattern) | 1-2 days | nothing |
| **H2 — Per-head forward dispatch** | `dispatch_hydra_head_forward` reusing existing eagle3_* primitives | 2-3 days | H1 |
| **H3 — Sequential N-head orchestration** | `HydraDrafter` impl Drafter trait + cache sharing | 2-3 days | H2 |
| **H4 — Eagle3Orchestrator dispatch** | `DrafterFamily` enum + Qwen35 + Gemma 4 orchestrator wiring | 1-2 days | H3 |
| **H5 — Serve wiring** | `HF2Q_SPEC_HYDRA` env flag + spec_decode_cli.rs integration | 1 day | H4 |
| **H6 — AC-1/AC-2 validation** | Hydra-N=1 byte-identity vs EAGLE-3 + N=4 random-init smoke | 1-2 days | H5 |
| **H7 — HASS training infrastructure** | Python sidecar OR Rust trainer (§4 decision) | 1-3 weeks | nothing (can parallelize with H1-H6) |
| **H8 — HASS-Hydra training run** | Multi-week H100 — Gemma 4 31B + Qwen 3.5/3.6 drafters | 1-2 weeks | H7 |
| **H9 — AC-3 empirical validation** | Real-prompt acceptance-length bench vs llama.cpp + ADR-037 EAGLE-3 baseline | 2-3 days | H8 + H6 |
| **H10 — Closure ADR** | Document empirical results + lock-in `HF2Q_SPEC_DRAFTER` flag for serve-time switching between EAGLE-3 / HASS-Hydra | 1 day | H9 |

**Critical path**: ~4-6 weeks elapsed assuming H1-H6 run serially in 2 weeks + H7 starts in parallel from day 1 + H8 runs after H7 + H9-10 close.

**Per "multi-week structural work always in scope" mantra**: this ADR's scope IS multi-week. Each phase ships incrementally; no big-bang merge.

---

## 7. Risks + mitigations

| Risk | Mitigation |
|---|---|
| HASS training doesn't converge for Gemma 4 31B specifically (paper trained on Llama-3 / Vicuna) | H7 validates HASS impl on Llama-3-8B first (smaller compute), then scales to Gemma 4 31B |
| Hydra N=4 latency overhead exceeds acceptance gain (i.e. negative speedup) | AC-3 explicitly gates on **throughput** improvement, not just L. Falls back to N=2 or EAGLE-3 if latency-dominated |
| Per-head lm_head adds N×vocab_size memory (Gemma 4: 4×262144×5376 BF16 = 11 GB) | DEFERRED to H9 if measured prohibitive — can share lm_head across heads at the cost of -0.3 L (Hydra paper §4.3 ablation) |
| Python training sidecar adds Python toolchain dep | Operator decision §4.1 |
| Drafter convergence sensitive to `λ_feat` hyperparameter | Standard ML hyperparameter sweep in H8; track validation acceptance length per epoch |

---

## 8. References

1. Hu et al., 2024. "HASS: Harmonized Speculative Sampling for Efficient and Reliable Speculative Decoding". arXiv:2408.15766.
2. Ankner et al., 2024. "Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding". arXiv:2402.05109.
3. Li et al., 2024. "EAGLE-3: Scaling up Inference Acceleration of Large Language Models". arXiv:2503.01840.
4. ADR-037 — EAGLE-3 with dynamic tree port (foundation).
5. ADR-038 — Gemma 4 monolith split + EAGLE-3 enablement (this depends on).
6. RedHatAI/gemma-4-31B-it-speculator.eagle3 (HF Hub, 2025) — first published EAGLE-3 drafter for Gemma 4 (baseline to beat).

---

## 9. Why this is the right next step (not a different SOTA)

Surveyed alternatives that compete with HASS + Hydra (as of January 2026):

| Technique | Reason rejected |
|---|---|
| Medusa-2 (parallel heads) | Hydra paper shows sequential beats parallel by 0.7-1.2 L |
| REST (retrieval-based) | Requires per-prompt corpus retrieval; latency dominated by k-NN search at serve time |
| Lookahead decoding (training-free n-gram) | Caps at L ≈ 1.5-2.0; doesn't beat trained EAGLE-3 |
| vLLM chunked-prefill speculation | Orthogonal — applies to prefill not decode; can stack with HASS-Hydra later |
| Hydra-α / Hydra-β published extensions | Marginal gains over base Hydra; defer to post-H10 |
| Long-N-gram drafting (PASS) | Less mature; not yet beating EAGLE-3 on standard benchmarks |

**Conclusion**: HASS-Hydra is the right next target because (a) both components are published, peer-reviewed, and empirically validated; (b) the combination is novel but additive (orthogonal axes); (c) integration with hf2q's existing EAGLE-3 infrastructure is clean (reuses verifier + tokenizer adapter + most of the drafter dispatch chain).
