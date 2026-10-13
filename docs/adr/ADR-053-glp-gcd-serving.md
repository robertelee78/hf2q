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
- **Updated:** 2026-10-08 — `--glp`/`--glp-alpha` are now server-wide policy,
  resolved for every model the server loads, not only the `--model` given at
  startup. Before this, `hf2q chat --model X --glp V` (which starts serve with
  no startup model and activates X afterwards) silently served X unsteered.
  See "GLP applies to every load" below.
- **Updated:** 2026-10-13 — gemma4 joins the GLP family set with a
  `residual_stream_post_layer` apply arm on every gemma4 execution path
  (#299 reopened scope; the 2026-10-09 gemma Won't-do is superseded by the
  owner's take-back — Matt published a Gemma 4 vector). See "Gemma 4 joins
  the GLP family set" below.
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
  not match must refuse the file; superseded 2026-10-08 by the
  supported-site-set rule — see Dual hook sites below):
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

## GLP applies to every load (2026-10-08)

`hf2q serve --glp REF [--glp-alpha A]` is steering policy for the whole
server. The reference and dose live in the template every model load uses, and
`load_engine` resolves REF against the exact model file being loaded. That
covers the startup model, a model activated later (the path `hf2q chat
--model` uses), a model named by a request, and a reload after eviction. A
resolution is cached per reference and model, so a reload does not consult the
Hub again.

The load MUST fail when REF cannot be used for that model; it MUST NOT fall
back to unsteered serving:

- Only the `qwen35`, `qwen35moe`, `deepseek4`, and `gemma4` engines bind
  GLP (gemma4 since #299, 2026-10-13 — see "Gemma 4 joins the GLP family
  set" below). Loading any other architecture on a `--glp` server is
  refused with that reason, because its engine would otherwise ignore the
  vector.
- A reference that cannot be resolved or does not match the model's checkpoint
  fails that load. Over HTTP the public diagnostic names the server's GLP
  vector as the cause; the full reason, which can contain URLs and paths, stays
  in the server log.

With a startup `--model`, serve still checks REF before loading weights, so a
bad reference fails fast.

**Root cause of the defect this replaces.** GLP shipped (75b1f840) resolving
REF only inside serve's startup `--model` branch, while the template for later
loads carried `glp_path: None`. Chat forwarding (f6b95da4) assumed serve
applied the flag to whatever model chat activated, and its tests checked only
that `--glp` appeared on the child's command line. Found in the ADR-061 #237
hands-on pass: `/hf2q/v1/runtime` reported `glp_active: false` for a
chat-started server whose command line carried `--glp`.

## Dual hook sites (2026-10-08 — extension; landed: bind #295, Qwen writer-site apply #296, DeepSeek post-layer apply #297; gemma4 residual apply #299, 2026-10-13)

Operator goal (issue #276 remaining scope): every GLP family binds **both**
spec hook sites — `residual_stream_post_layer` and `ffn_out_pre_residual`.
The forcing case: the published DeepSeek-V4-Flash-0731 GLP-42 **residual**
vector is refused by hf2q on DeepSeek (bind accepts one family hook), and
ADR-054's null result says the calibrated DeepSeek vector needs exactly
that apply site.

- **Bind accepts the family's supported site set.** `BoundGlp::bind` takes
  `family_hooks: &[GlpHookPoint]` instead of one hook; a vector whose
  `glp.hook_point` is outside the set is refused with an error naming the
  family's supported sites. Both the Qwen and DeepSeek engines pass
  `[residual_stream_post_layer, ffn_out_pre_residual]`.
  `attn_out_pre_residual` stays refused globally (spec-recognized, no
  family implements).
- **Qwen writer-site apply** (`ffn_out_pre_residual`): steer `ffn_out`
  out-of-session (`apply_layer_gpu`, the same buffer lineage and queue
  ordering the working post-layer hook already uses) with
  `add_residual=None` at every fold site, then an explicit
  `residual_add_gpu` for every arm (all arms route through the explicit
  add when steering at the writer). Off by default: with no vector, every
  fold site passes `Some(residual)` exactly as today — byte-identical when
  unsteered.
- **DeepSeek post-layer apply** (`residual_stream_post_layer`): in the
  same per-layer FFN encode closure, after `dispatch_hc_post`, a barrier
  between the writer and the state, then `apply_layer_gpu_mhc_in_session`
  on `output_state` (`[rows, hc, hidden]`) — the validated, currently
  unused mHC helper. **Project mode only** at this site (no add-mHC kernel
  exists): bind refuses `(deepseek, residual, add)` with a named error
  rather than failing at runtime. Widths accepted at this site: `hidden`
  (one shared direction steering every stream — what `hf2q calibrate`
  exports) and `hc*hidden` (per-stream, the weightless mHC discipline);
  writer-site vectors stay `hidden`-only (owner decisions 2026-10-08).
- **Qwen writer-site verification**: no published writer-site vector
  exists (issue #276 matrix), so a local synthetic test vector — a
  throwaway operator script (not production code, not a committed runtime
  dependency) writing a conformant GLP GGUF bound to the local Qwen
  checkpoint's conversion-receipt identity, deterministic directions on a
  few layers, `hook_point=ffn_out_pre_residual` — drives the bind +
  dose-ladder (0 = identical, live = shifted) and confirms the shift lands
  via the writer site (the FFN write alone), not the residual path.
- **Unsupported-family consistency** (#271, already landed): `--glp` on a
  family outside `GLP_SERVING_ARCHITECTURES` fails every load with one
  message naming the supported families — hands-on verify only, fix if
  inconsistent.
- **Calibrate's export contract is now site-selectable and per-family
  (2026-10-12, #300 — supersedes the pre-design 'untouched' boundary).**
  `hf2q calibrate` accepts `--hook-point {writer|residual}` (the export's
  `glp.hook_point`) and `--capture-site` (default: the family's
  `default_derived_at` — deepseek4 keeps the residual_stream_post_layer
  capture, qwen defaults to its hook point — same-site capture/apply by
  default; a cross-site pair remains
  expressible and is exported with its honest `glp.derived_at`, per
  spec). Per-family defaults: deepseek4 keeps the GLP-29-declared pair
  (`ffn_out_pre_residual` / `residual_stream_post_layer`) until ADR-054's
  same-site ladder lands a measured winner; qwen35/qwen35moe default to
  `residual_stream_post_layer` same-site (the family's published site
  class). Calibrate covers exactly the serve-side GLP family set
  (qwen35, qwen35moe, deepseek4); gemma4 is refused consistently with
  `--glp` serving. The site options are plumbing, not effectiveness
  claims: behavioral validity remains ADR-054's gates (the null record
  and the ladder own that question).
  **2026-10-13 correction (#299):** the "exactly the serve-side family
  set" identity no longer holds — gemma4 now serves `--glp` but has no
  calibrate capture adapter, so calibrate refuses gemma4 at its own
  surface only, naming the implemented adapters (qwen35, qwen35moe,
  deepseek4) and the honest serve/calibrate split. A gemma4 capture
  adapter remains future work; until it exists, gemma4 consumes
  published or externally calibrated vectors only.

Verification is hands-on per owner direction (no new unit tests; one
worktree at a time into the shared `CARGO_TARGET_DIR`; one full-model
runtime at a time — the ~100 GiB DeepSeek artifact never co-resident with
a peer; release coordination checked before every heavy load): release
build; DeepSeek + the published GLP-42 residual vector binds and shifts a
fixed probe vs. the unsteered baseline; GLP-29 (FFN-writer) still binds
and steers on DeepSeek; Qwen + GLP-49 — the published residual-site vector
`msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49`, file
`Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf` (docs/qe/glp-gcd
2026-10-08) — still binds and steers; `--gcd --glp` combos on both
families. ADR-054 gains the follow-through: with the
residual-site apply existing, its null-result remedy (the calibrated
vector applied at its capture site) becomes measurable.
**gemma4 gates (#299, 2026-10-13 — the reopened scope):** gemma4 + the
published GLP-27 residual vector (below) binds on the google-converted
Gemma-4-26B-A4B-it GGUF and passes the dose ladder — dose 0 byte-exact
vs. the unsteered baseline, live dose (alpha 0.5) shifting a fixed probe
— plus `--gcd --glp` combos; a writer-site vector on gemma4 is refused by
name listing `residual_stream_post_layer`.

## Gemma 4 joins the GLP family set (2026-10-13 — #299)

The 2026-10-09 Won't-do record for this story ("no GLP file exists from
Matt for Gemma") is **superseded by the owner's take-back**: Matt
(msuiche) published a Gemma 4 vector —
`msuiche/gemma-4-26B-A4B-it-abliterated-GLP-27-L3-29-a0.5`
(`glp.hook_point=residual_stream_post_layer`, same-site `derived_at`,
project mode, `alpha_default` 0.5, rank-1 orthonormal per-layer
directions, layers 3–29 zero-based of 30, width 2816, base
`google/gemma-4-26B-A4B-it` @ `4d7ae498…`, held-out-vs-shuffled
validation ratio ≥ 5× per its metadata). The Qwen3-VL half of the
original story stays ruled out per the 2026-10-09 decision; the reopened
scope is gemma4 only.

- **Registration.** gemma4 joins `GLP_SERVING_ARCHITECTURES`
  (`src/inference/glp/discovery.rs`); the #271 server-wide gate passes
  gemma4 through, and the boot refusal's available-families list gains
  it automatically (the message is built from the same constant).
- **Bind site set.** The gemma4 engine passes
  `family_hooks = [residual_stream_post_layer]` — the dense residual
  contract. The writer site (`ffn_out_pre_residual`) is NOT implemented
  for gemma4 (no apply arm exists at the gemma FFN writer); a
  writer-site vector is refused at bind by name, listing residual as the
  supported site (the #295 contract). `attn_out_pre_residual` stays
  globally refused.
- **The apply arm — one hook site, four anchor points, every execution
  path.** The gemma4 post-layer residual stream is the buffer the fused
  end-of-layer (norm + residual add + per-layer scalar) writes; the hook
  is anchored immediately after that write on each of the four
  per-layer bodies:
  1. `encode_one_layer` (`src/inference/models/gemma4/gpu_full_attn.rs`)
     — the single-token decode body shared by the serial decode loop,
     the parallel-encode worker, `forward_decode_capture_hidden`, and
     the slot-aware decode entries;
  2. `encode_one_layer_batched`
     (`src/inference/models/gemma4/batched_body.rs`) — the `[N, hidden]`
     batched decode body (`forward_decode_body_batched` /
     `decode_batch_gemma4`);
  3. `forward_prefill_with_soft_tokens_resume`
     (`src/serve/forward_prefill.rs`) — the linear per-(token, layer)
     prefill body; every entry that routes through it is covered
     (forward_prefill, soft-token/vision prefill, LCP resume, the
     cold-admission tiny fallback, `forward_embed_last`, and the
     slot-aware/HB prefill delegations);
  4. `forward_prefill_batched_inner`
     (`src/serve/forward_prefill_batched.rs`) — the batched prefill
     body covering all `forward_prefill_batched*` entries including
     multi-seq and live-resume (the T-stream steers every seq's rows —
     the site is per-token) and the dflash verify forward (it routes
     through `forward_prefill_batched`).
  All four anchors sit AFTER the whole end-of-layer variant block (the
  fused iter-367/iter-208 kernels and the split decompositions all write
  the same buffer), are encoded in-session ahead of the session's
  commit, and dispatch `apply_layer_gpu_in_session` (the same dense
  in-session helper the DeepSeek arms use) behind an explicit
  read-modify-write barrier (`barrier_between(&[hidden], &[hidden])`) —
  the measured 2026-09-09 lesson: without the declared hazard the
  projection can race the end-of-layer write and deterministically
  no-op. The projection is row-wise, so the rows=1 per-(token, layer)
  linear-prefill arm and the rows=N arms are the same math.
- **Identity mechanism.** Unsteered (`glp: None`, the default): no
  branch, no barrier, no dispatch — byte-identical to the unsteered
  graph on every path. Dose-0 (`alpha == 0`): the dispatch is skipped
  entirely — bit-exact. Layers outside the vector's set (0–2 for
  GLP-27) are untouched. Spec layer mapping is the flat rule:
  `direction.N` applies at graph layer N.
- **Paths with no GLP surface (enumerated, with reasons).**
  `forward_decode_kernel_profile` is reachable only from the CLI
  generate diagnostic mode, which never binds a vector (the generate
  path constructs loads with no GLP reference); the tree-verify
  forwards (`forward_tree_verify_gpu*`) and
  `forward_decode_verify_serial` have production callers only through
  the dflash orchestrator, whose verify routes through
  `forward_prefill_batched` (covered) — their direct callers are tests;
  the parity/quality harness builds its own unsteered instance. None of
  these can serve a bound vector silently unsteered.
- **Cache identity.** gemma4's LCP identity
  (`gemma4_steering_and_graft_params_hash`) now includes the bound GLP
  through the SAME `glp_steering_params_hash` implementation as qwen35
  (raw direction bytes, hook, mode, dose): a steered server's saved KV
  is never addressed by an unsteered or differently-steered one. The
  unconfigured key value changes once across this change (the literal
  `glp=none` bytes are now hashed through the shared helper) — a
  one-time cold miss for in-process LCP entries, safe direction. Known
  limitation, unchanged from the qwen35 posture: the typed
  `--kv-persist` disk namespace is keyed by model fingerprint, not
  steering; a steered server's disk snapshots can be hydrated by a
  differently-configured server of the same artifact. That is a
  pre-existing family-wide posture, recorded here for the follow-up.
- **Calibrate.** gemma4 is refused at `hf2q calibrate` (no capture
  adapter — see the 2026-10-13 correction above), not at serve time.

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

The file's `glp.hook_point` must match the implemented apply site
(2026-10-08: a site in the family's supported site set — see Dual hook
sites below).
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
