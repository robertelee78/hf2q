# ADR-053: GLP runtime steering and the `--uncensor` serving surface

- **Status:** Draft — design accepted; implementation gates listed below are
  not yet met; do not treat the surface as present in any build
- **Date:** 2026-09-04
- **Related:** ADR-052 (grammar semantics), ADR-050 (serve preflight),
  ADR-051 (model resolution), ADR-042 (DeepSeek serving), ADR-017
  (per-family status)

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

- `hf2q serve <model> --uncensor [--glp <ref>] [--glp-alpha <f>]`
  enables the GLP path plus the grammar stack. Plain `serve` is unchanged.
- `--uncensor --glp <ref>` binds an explicit vector (Hub repo, file, or
  local path). `--uncensor` without `--glp` asks the resolver to search the
  Hub for `*-GLP-*` artifacts bound to the resolved model family and commit;
  **ambiguous or absent candidates fail closed** (printed list, exit).
  `--glp-content-sha256` MAY pin the binding; `glp.content_sha256` from the
  file is verified against it when present.
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
- Grammar class engaged by `--uncensor` is a serve-config choice between the
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
| B12-class framed reasoning | −0.92 (n=12, pending) | answer-scoped anchors plateau at −0.32 |
| B10 static+exclusion | −0.65 | thinking-on champion |
| GLP (published, other engines) | refusal32 1→26-32/32 | dose-cliffs per family; terminal tails need output regime |
