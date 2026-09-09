# ADR-054: Runtime GLP calibration — `hf2q calibrate <model>`

- **Status:** Draft — design accepted; the capture/export/canary path is
  implemented and its canary gates are proven on DeepSeek-V4 (2026-09-09:
  zero-dose logit-identical, live-dose shift 0.213 > 1e-3). The behavioral
  gates (3–4: refusal-panel delta, dose ladder) remain unmet — the output of
  `hf2q calibrate` is a candidate vector, not a validated derivation, until
  those land.
- **Date:** 2026-09-04 (revised 2026-09-09: forced-capture design replaces
  generation-based capture — teacher-forced pinned prefixes, prefill-only
  forward passes, targeted decision positions; capture cost collapses from
  ~7h of autoregressive jobs to minutes of batched prefill, and the capture
  hook lives in the prefill path rather than the decode loop)
- **Related:** ADR-053 (GLP serving surface), ADR-052 (grammar semantics),
  ADR-042 (DeepSeek serving)

## Problem

ADR-053 makes `--gcd` a two-part intervention (GLP activation steering +
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
base checkpoint it was derived from — but the general `--gcd` promise
needs a way to *derive* the vector for any stock model quickly, on-device,
without shipping weights or running a training loop.

## Decision

Add `hf2q calibrate <model-ref>` — a bounded, deterministic pipeline that
produces a spec-conformant GLP file plus a lexical register report:

1. **Probe (forced-capture, teacher-forced).** For each contrastive prompt,
   build two sequences: `prompt + refusal prefix` (pinned: the model's own
   refusal opener from the register report, e.g. `I cannot`) and
   `prompt + compliance prefix` (pinned: the W1 anchor). One batched
   **prefill-only** forward pass per pair — no sampling, no mask, no decode
   loop — reading the residual stream at the committed state (the last
   prefix token; optionally per-prefix-position). Pinned prefixes make both
   classes deterministic: zero discarded rollouts, zero label noise, exactly
   matched pairs — the pairing is what eliminates the prompt-content
   confound. (Contrastive set: OBLITERATUS `prompts.py` 512+512; a ~64+64
   slice suffices given targeted positions + zero label noise.)
2. **Two directions, both form-matched.** The output-space direction pairs
   the two *pinned* arms on identical prompts: `d_out = mean(forced-
   compliance) − mean(forced-refusal)` — never a pinned arm against a
   natural rollout, because pinned vs natural text differ in form, and form
   confounds recover register axes, not refusal. The disposition direction
   stays at the instruction level: `d_disp = mean(harmful) − mean(harmless)`.
   Verification chooses between them (gates).
3. **Distill.** Per-layer direction (v1 = plain difference-of-means;
   optionally winsorized / whitened-SVD). Exactly OBLITERATUS PROBE→DISTILL
   inside the serving stack.
4. **Verify (fail-closed canary).** Project `dℓ` at each candidate layer and
   confirm a logit shift > 1e-3 on a fixed probe prompt; inert layers are
   excluded. Zero-vector no-op canary runs first (plumbing gate). Then the
   regime check (below).
5. **Export.** Write a GLP-conformant GGUF (`direction.<N>` fp32 tensors,
   `glp.mode=project`, `glp.hook_point=residual_stream_post_layer`,
   `glp.spec_version=1`, `glp.alpha_default`, `glp.content_sha256`,
   `general.base_model.*` provenance pinned to the loaded checkpoint's
   commit/revision).
6. **Register report.** A 6-prompt lexical canary (uniform refusal probes)
   records the model's top refusal openings → the lexicon for the grammar
   exclusion automaton, replacing the hand-written list. (This doubles as the
   source of the pinned refusal prefix in step 1.)

**Gate-2 outcome (2026-09-09, weightless side, Qwen3.6-35B bf16):** the
forced-capture direction does not suppress at scale — 0/8 held-out delivery,
identical to baseline, while the natural `d_disp` direction delivered 7/8 on
the same span. cos(forced-derived, natural-derived) ≈ 0.06 at 35B,
generalizing the 0.5B smoke result — no scale-dependence reprieve. The
within-prompt forced pair is a *register* axis, exactly as the form-confound
warning predicted. **Resolution: `d_disp` (harmful vs harmless, no prefixes)
is the derivation arm; pinned prefixes are demoted to register probes** —
fast, zero-label-noise instruments for register questions, not a direction
source. The exporter ships `d_disp`; the `d_out` sums stay in the report for
register analysis only.

(Surviving note from the forced-capture design: the regime-match observation
— forced-capture states ARE the deployment distribution when GLP composes
with the grammar — remains true as a statement about *states*; it just
doesn't rescue a *direction* derived from them. Adopted on the weightless
side for probe work.)

**Generality.** The pinned-prefix trick enumerates contrastive pairs for any
expressible direction — register, language, reasoning style — so
`calibrate` is a direction-derivation surface, not a refusal-only tool.
Refusal is the measured case; the mechanism is general.

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
2. **Forced-vs-natural regime check.** Derive the vector from forced
   captures; check cosine similarity against a natural-reference vector
   derived from a small (~100-prompt) naturally-generated, judge-labeled
   sample; then run the downstream eval (refusal Δ on the adversarial half,
   capability parity on the benign half). The forced-capture vector must
   match on eval, not just on cosine.
3. Refusal-panel delta on the 512+512 grammar_probe harness: a calibrated
   vector must beat the no-GLP grammar-only control arm (B12-class) on
   refusal Δ with invalid at parity.
4. Dose ladder sanity: α ∈ {0.5, 1.0, 1.5} report; the chosen default is
   documented per family, not guessed.
5. Cross-checkpoint safety: refuse to apply a vector whose
   `general.base_model.0.version` does not match the loaded checkpoint.

## Evidence behind the design

- 2026-09-04 grammar campaign (this repo, `scripts/grammar_probe/`):
  per-model dial differences measured (bounding helped nothing on DeepSeek;
  refusal registers differ per family).
- Front-loading probe (this repo, `refusal_mass_probe.py`): refusal mass
  concentrates at the entry token (99.97% on DeepSeek) — the decision
  positions for targeted capture are measured, not guessed.
- weightless spec + posts: per-model dose cliffs, subspace-not-direction,
  hook-point 9× sensitivity.
- OBLITERATUS `prompts.py`: the contrastive pair registry used for PROBE.
- Forced-capture design (teacher-forced pinned prefixes, prefill-only
  capture, committed-state positions, distribution-shift caveat): Matt
  Suiche's analysis, 2026-09-09. The durable wins vs the prefill-only status
  quo are exact pairing, zero label noise, and deterministic classes — the
  pairing eliminates the prompt-content confound. The regime-match note is
  ours. Review refinement (accepted): both output-space arms are pinned —
  pinned-vs-natural text confounds form with class.
- Early warning (theirs, Qwen2.5-0.5B smoke, n=8): instrument stable
  (prefix-wording robustness cos 0.956–0.997) but cos(forced-derived,
  natural-derived) ≈ 0 at that scale — forced-capture vectors may be
  orthogonal to natural-derived ones at small scale. Their gate-2 protocol
  (Qwen3.6-35B bf16, natural vs forced, behavioral verify) is running;
  our gate 2 (match on eval, not just cosine) is the load-bearing check.

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

