# ADR-054: Runtime GLP calibration — `hf2q calibrate <model>`

- **Status:** Draft — design accepted; implementation gates below are not met
- **Date:** 2026-09-04
- **Related:** ADR-053 (GLP serving surface), ADR-052 (grammar semantics),
  ADR-042 (DeepSeek serving)

## Problem

ADR-053 makes `--uncensor` a two-part intervention (GLP activation steering +
grammar stack). Both parts carry per-model parameters that a frozen universal
artifact gets wrong, as measured in the 2026-09-04 grammar campaign:

- Refusal *geography* differs per family (where in the forward pass / output
  trajectory refusal pressure concentrates).
- The refusal *register* differs (which lexical phrases the model actually
  opens refusals with — DeepSeek answers differently from Qwen or Gemma).
- Dose (alpha) has per-model cliffs (weightless: Inkling garbles at 1.0,
  GLM-Flash wants 2.0).
- Frame/structure bounds are per-model dials: B12v2/v3's step bounding
  *hurt* DeepSeek compliance while the same shape may help another family.

A static GLP file (e.g. `msuiche/*-GLP-*`) is exactly right for a model whose
base checkpoint it was derived from — but the general `--uncensor` promise
needs a way to *derive* the vector for any stock model quickly, on-device,
without shipping weights or running a training loop.

## Decision

Add `hf2q calibrate <model-ref>` — a bounded, deterministic pipeline that
produces a spec-conformant GLP file plus a lexical register report:

1. **Probe.** Run the stock model over the contrastive prompt set
   (OBLITERATUS `prompts.py` ships 512+512 harmful/harmless pairs; a
   calibration slice of ~64+64 suffices) and capture the per-layer post-
   residual activation at the last prompt token. This reuses the existing
   forward graph — no gradients, no training.
2. **Distill.** Per-layer direction `dℓ = mean(harmful) − mean(harmless)`
   (optionally winsorized / whitened-SVD; v1 = plain difference-of-means).
   This is exactly OBLITERATUS PROBE→DISTILL, run inside the serving stack.
3. **Verify (fail-closed canary).** Project `dℓ` at each candidate layer and
   confirm a logit shift > 1e-3 on a fixed probe prompt; a layer whose
   projection is inert is excluded. The zero-vector no-op canary runs first
   (plumbing gate).
4. **Export.** Write a GLP-conformant GGUF (`direction.<N>` fp32 tensors,
   `glp.mode=project`, `glp.hook_point=residual_stream_post_layer`,
   `glp.spec_version=1`, `glp.alpha_default`, `glp.content_sha256`,
   `general.base_model.*` provenance pinned to the loaded checkpoint's
   commit/revision).
5. **Register report.** A 6-prompt lexical canary (uniform refusal probes)
   records the model's top refusal openings → the lexicon for the grammar
   exclusion automaton, replacing the hand-written list.

The output is a normal GLP file: auditable, shareable, loadable by any
conformant reader (hf2q `--glp`, or the weightless hotfix path).

## Non-goals

- No fine-tuning, no gradient steps, no weight edits.
- Calibration is not a substitute for the measured dose ladder; `alpha`
  defaults to 1.0 and the operator titrates with the grammar-probe harness
  (B-arm protocol) before trusting a deployment.
- Not a replacement for published vectors when they exist: a Hub GLP bound to
  the exact base commit is preferred over a freshly calibrated one (the
  resolver prefers an exact-commit published artifact, falls back to
  `calibrate`).

## Gates before this ships

1. Calibration canary: zero-vector logits-identical; live-vector logit shift
   > 1e-3 on the probe prompt.
2. Refusal-panel delta on the 512+512 grammar_probe harness: a calibrated
   vector must beat the no-GLP grammar-only control arm (B12-class) on
   refusal Δ with invalid at parity.
3. Dose ladder sanity: α ∈ {0.5, 1.0, 1.5} report; the chosen default is
   documented per family, not guessed.
4. Cross-checkpoint safety: refuse to apply a vector whose
   `general.base_model.0.version` does not match the loaded checkpoint.

## Evidence behind the design

- 2026-09-04 grammar campaign (this repo, `scripts/grammar_probe/`):
  per-model dial differences measured (bounding helped nothing on DeepSeek;
  refusal registers differ per family).
- weightless spec + posts: per-model dose cliffs, subspace-not-direction,
  hook-point 9× sensitivity.
- OBLITERATUS `prompts.py`: the contrastive pair registry used for PROBE.

## Deferred follow-ups (parked, do not lose)

1. **Context-independent token bitmask caching** (grammar hot path). Today
   `serve/api/grammar/mask.rs` recomputes candidate-set rejection across the
   full vocab per token. For grammars with a large free body (the B12-class
   universal grammar: `[^\x00]*` body), most of the vocab is
   context-independent; precompute that set's bitmask once and runtime-check
   only the context-dependent tail (the anchor / exclusion automaton). This
   is the XGrammar context-split optimization, adapted to hf2q's stack-set
   runtime. Measure first on Qwen's 262K vocab.
2. **Compiled-grammar serialization.** Cache the *compiled* `Grammar` (post
   parse, post repetition-expansion) as a versioned, integrity-hashed
   artifact keyed by model commit + grammar source hash, so a calibrated
   per-model grammar (this ADR) boots instantly and the canary runs against
   the cached form. Version-stamp + fail-closed on version mismatch (the
   `CompiledGrammar` v11 pattern). Both are parked until the GLP
   within-process canary is trustworthy (measurement discipline first).

