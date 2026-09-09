# ADR-053: GLP runtime steering and the `--gcd` serving surface

- **Status:** Draft — design accepted; implementation gates listed below are
  not yet met; do not treat the surface as present in any build
- **Date:** 2026-09-04
- **Related:** ADR-052 (grammar semantics), ADR-050 (serve preflight),
  ADR-051 (model resolution), ADR-042 (DeepSeek serving), ADR-017
  (per-family status)


## Attribution

- **GCD (Grammar-Constrained Decoding)** — the concept, including both faces:
  grammar as per-step authorization control for agent tool calls, and GCD as
  *forced abliteration* on a stock, refusal-trained model without touching a
  single weight: **Vince Ovando** (vince@cybersharkconsulting.com),
  [tantalus.io](https://tantalus.io/). The `--gcd` flag name and the
  whitelist-over-blacklist design philosophy are his.
- **GLP (weightless control vectors)** — the GGUF container, the projective
  per-layer direction spec, and the hook-site conventions: **Matt Suiche**
  (m@msuiche.com), [msuiche/weightless](https://github.com/msuiche/weightless).
  Matt also ran the measured PoC replica of Vince's concept ("Forced
  Abliteration via Grammar-Constrained Decoding", 2026-09-07): the front-loaded
  refusal logprob probes (99.85% entry mass) and the 25-cell serving-stack
  conformance battery with the demonstrated beam-search FATAL.
- **hf2q** is the first inference-engine-native implementation: Vince's and
  Matt's proofs ran grammars per-request through vLLM's `structured_outputs`;
  `--gcd` makes the intervention first-class at serve time (embedded W1
  grammar, Z_t instrumentation, ADR-056 conformance battery).

The three-way collaboration (concept, measurement, implementation) was
developed jointly in September 2026.

## Problem

hf2q serves stock model weights. Operators who want reduced-refusal behavior
currently re-derive and re-serve a modified artifact, or rely on per-request
grammar constraints (the `grammar_probe` campaign, 2026-09-04), which move
refusal without touching weights but cannot shift the model's disposition
below the output layer. The GLP spec ([msuiche/weightless](https://github.com/msuiche/weightless)
`/spec/GLP.md`) is the published container for projective per-layer directions
applicable at inference: a few hundred KB that steer the post-layer residual
stream, `h ← h − α(h·d̂)d̂`, with base weights untouched.

hf2q needs (a) a conforming GLP reader/apply path per family, and (b) a UX
where the vector is a *modifier of the resolved model*, never a servable
operand — the `--mmproj` shape.

## Decision

- `hf2q serve <model> --gcd` enables the grammar stack alone. GLP steering is
  **orthogonal and opt-in**: `--glp <ref>` adds it, with or without `--gcd`.
  Plain `serve` is unchanged. (Correction 2026-09-09: an earlier draft of
  this clause had `--gcd` implying GLP auto-discovery — wrong. `--gcd`
  without `--glp` is GCD only; no resolver runs.)
- `--glp <ref>` binds an explicit vector (Hub repo, file, or local path).
  Bare `--glp` (no value) asks the resolver to search the Hub for `*-GLP-*`
  artifacts bound to the resolved model family and commit; **ambiguous or
  absent candidates fail closed** (printed list, exit). `--glp-content-sha256`
  MAY pin the binding; `glp.content_sha256` from the file is verified against
  it when present.
- Reader conformance follows the spec verbatim: `glp.mode` absence means
  `add`; unimplemented modes/hooks, `direction.0`, or unknown
  `glp.hook_point` are **fatal**. `project` never merges with another
  cvec. `glp.alpha_default` is used unless `--glp-alpha` overrides.
- The hook point is `hidden_states + residual` per layer (post-layer
  residual stream). For DeepSeek-V4's multi-hyper connection
  `[rows, 4, hidden]` output_state after the `dispatch_hc_post` fold —
  which materializes the complete HC state per layer. (The retraction of
  2026-09-09 corrects the assertion in the weightless posts: DeepSeek's
  native graph DOES fold the HC residual into a complete `output_state` at
  this point, unlike anchor-lift on the FFN write pre-fold.) The
  per-stream direction discipline says each of the `hc` streams is steered
  with its own direction slice, never the flattened sum.
- Grammar class engaged by `--gcd` is a serve-config choice between the
  measured arms (B12-class framed reasoning; B10-class static+exclusion for
  thinking-on turns). Operator overridable via launcher env; the inline
  configurable is not required for v1.
- `hf2q info --glp <ref>` performs reader-conformance preflight without
  tensor residency. `hf2q serve list` and `/v1/models` MAY show the bound
  GLP on the model row (like the vision projector row linkage today).

## Non-goals

- Serving a GLP file as the model operand is invalid (explicitly rejected
  by the flag design).
- Solver-side derivation of directions is out of scope; OBLITERATUS or
  weightless exporters produce them. hf2q consumes only.

## Gates before this ships

1. Reader conformance tests against the spec's required clauses 1–6.
2. Layer-mapping differential probe (a `direction.N` must activate exactly
   layer N; the spec documents a degrading, not failing, off-by-one class).
3. Steering canary: `--glp` with an inert (zero) vector must be a no-op on
   logits; a live vector must shift logits by >1e-3 on a fixed probe.
4. Hardware gate on at least one family (DeepSeek-V4 or Qwen3.8) with the
   published vector (`msuiche/*-GLP-*`), reporting refusal/benign panels
   under the same protocol as the follow-up eval.
5. Deterministic auditability: applied layer set, alpha, and content hash
   are logged at boot; steering off-by-default unless the flag is present.

## Evidence behind the design (2026-09-04 campaign, same model)

| lever | Δ refusal | notes |
|---|---|---|
| B12-class framed reasoning | not measured (n=12 below power; n≥32 resolves ~30 points) | answer-scoped anchors plateau at −0.32 |
| B10 static+exclusion | −0.65 (fraction of refusals removed) | thinking-on champion |
| GLP (published, other engines) | refusal rate on 32-prompt panel: 1/32 → 26–32/32 *refusals remaining* | dose-cliffs per family; terminal tails need output regime |

## Resolver trust boundary

The Hub resolver is a supply-chain surface: name-pattern binding means anyone
can publish `<model>-GLP-anything`. The default path therefore prefers
**exact-base-commit** matches, prints the resolved artifact's provenance at
boot, and **warns loudly on family-only matches**. Version mismatch at apply
time is ADR-054 gate 5; the resolution-time posture lives here.
`--glp-content-sha256` is the strict form (pin the artifact hash); without it,
family-only matches are announced, not silent.
