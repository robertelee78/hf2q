# ADR-053: GLP runtime steering and the `--gcd` serving surface

- **Status:** Accepted — implemented and shipped. The serving surface is
  present in builds since v0.1.21: `--gcd` (grammar-only; W6V2 cores lexicon
  as the shipped default), `--glp`/`--glp-alpha` with auto-discovery, chat
  forwarding of all serve-time flags, and `--gcd-schema` (ADR-057). Gates 1–3
  and 5 are met (reader conformance, layer-mapping differential, the steering
  canary — proven on DeepSeek-V4 hardware, zero-dose no-op / live-dose shift
  0.213–0.256 — and boot auditability). Gate 4's behavioral panels
  (refusal/benign under the follow-up protocol) remain open; until they land,
  a calibrated GLP vector is a candidate direction, not a validated
  derivation (see ADR-054 behavioral gates 3–4).
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
  absent candidates fail closed** (printed list, exit). Explicit Hub file
  URLs select a filename and may pin an immutable revision. The file's
  `glp.content_sha256`, when supplied, is verified over raw direction bytes;
  there is no `--glp-content-sha256` CLI flag.
- Reader conformance follows the spec verbatim: `glp.mode` absence means
  `add`; unimplemented modes/hooks, `direction.0`, or unknown
  `glp.hook_point` are **fatal**. `project` never merges with another
  cvec. `glp.alpha_default` is used unless `--glp-alpha` overrides.
- Hook points are family-specific and enforced at bind (2026-09-10
  correction, per spec/GLP.md's recognized-values table — the hooks are
  different tensors and NOT interchangeable, and a reader whose hook does
  not match must refuse the file):
  - **Qwen:** `residual_stream_post_layer` — the complete post-layer
    residual (`hidden + residual` after the FFN fold), applied on every
    execution path including greedy decode (the greedy path previously
    omitted the hook — steering silently stopped during greedy decoding).
  - **DeepSeek-V4:** `ffn_out_pre_residual` — the FFN writer
    (`ffn_output = moe + shared`) immediately before the
    `dispatch_hc_post` fold: the same site the ds4 reference reader
    steers and the `glp.hook_point` the published GLP-29 declares. (The
    2026-09-09 retraction of the weightless-posts claim stands: the native
    graph does fold a complete post-layer HC state — but the *declared
    hook* for this family's vectors is the writer, not the fold.)
  - Layer mapping is the spec's flat rule: `direction.N` applies at layer
    N (0-based), no offset. The previous `layer + 1` lookup applied every
    direction one layer early on both families — self-consistent with
    hf2q's own off-by-one exporter, silent for interchange files (adjacent
    layers' refusal directions have cosine 0.555–0.979); proven fixed by
    the differential layer-mapping probe (below-layer delta 0.00000000,
    at-layer delta 0.026533).
  - The mHC per-stream discipline (each `hc` stream steered with its own
    direction slice, per-stream norms, never the flattened sum) is
    implemented in the (currently unused) mHC helpers with GPU-vs-CPU
    reference proofs; the production DeepSeek path steers the writer with
    the two-dimensional activation helper, which dispatches additive or
    projective mode.
  - Qwen prefix-cache identity includes the full steering configuration
    (raw direction bytes, hook, mode, dose) — two differently-steered
    servers can no longer address each other's saved KV.
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

The automatic resolver searches only the public `msuiche` namespace for the
exact `<base>-abliterated-cyber-GLP-<N>` convention, including the published
optional `-residual` site qualifier and `-L<first>-<last>-a<dose>` suffix.
These labels identify candidates; the file's metadata controls the hook and
dose. Same-checkpoint site variants require explicit selection. It uses the model GGUF already
selected by serving, not a second model download. Searches exceeding
100 returned repositories fail as incomplete. Multiple matching repositories
or multiple GGUF files require explicit selection; ordering and coverage do
not confer priority. An explicit local file performs no Hub work. Explicit
Hub repository IDs and canonical tree/blob/resolve URLs use the shared HF
reference parser. A returned inventory must name an immutable commit, and
the chosen file is downloaded at that commit rather than a mutable branch.

The file's `glp.hook_point` must match the implemented apply site.
`glp.derived_at` is optional string provenance: absence means the same site,
and arbitrary descriptive labels remain valid. A non-string value is a load
error. A declared derivation site differing from the apply hook produces an
explicit warning to validate the transfer and its dose; it does not select a
different hook or by itself refuse the file.

A GLP's declared base repository/name and HF commit are checked against the
served checkpoint before binding in both family loaders. For hf2q conversions,
the existing adjacent schema-v3 receipt supplies the actual source repository
and revision only after its output size and SHA-256 match the selected stable
file. Existing source-bundle metadata, when present, must agree with that
receipt. The receipt's embedded output path is never dereferenced. A bounded
process-local cache reuses output digest verification while the stable file
identity is unchanged, avoiding a second large-file hash at model binding.

Model-card `general.base_model.*` fields describe ancestors, not the served
checkpoint, and supply no source identity. Without a usable conversion receipt,
only the model's own `general.name` and organization are available. Conflicts
are fatal; a declared repository or commit cannot be satisfied by an ancestor,
missing source evidence, or a human version label. The spec defines `general.base_model.0.version` as the
HF commit, while `glp.content_sha256` covers direction tensors. The latter
is checked over raw F32 direction bytes in increasing numeric layer order,
excluding metadata and alignment padding, before normalization. No model-byte
hash requirement is inferred. Matching metadata expresses checkpoint
compatibility; a local conversion receipt binds that source claim to output
bytes, without attesting that its claimed remote derivation is authentic.

Legacy vectors without a declared checkpoint remain loadable, with an
explicit unverified-revision warning. Name-only matches do not acquire a
checkpoint guarantee. Organization labels are compared when both sides
supply them, but are not synthesized into HF account names. Quantized model
bytes may differ while the declared source checkpoint remains identical;
behavioral transfer across quantizations still needs measurement.

The resolver and compatibility tests use mocked Hub I/O and synthetic
metadata/tensor fixtures. They cover explicit paths and URLs, immutable
revision downloads, repository/file ambiguity, failed lookup, wrong namespace
and revision, missing commit evidence, and direction-hash corruption. These
checks do not replace the family-specific real-model steering gates.
