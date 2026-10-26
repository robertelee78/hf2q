# Experimental Gemma full-input seven-wire post-oracle

This engine-free, post-unload helper is separate from the unchanged generic
strict cue-prefix oracle. It does not load weights or create a Metal device.
It qualifies only the experimental Gemma checkpoint mechanism for one actual
seven-request conversation. Four-slot qualification requires four independent
replays plus an independent review of the shared launch, coding results,
actual tool execution, source/build provenance and cleanup.

Turn zero must have a nonempty strict cue-less anchor S0 < P0, an exact token
prefix, and zero cached tokens. Every subsequent turn must be a text tool-result
input whose production `request_tag` accepts nonempty render(false) ==
render(true). The tag validates the actual input length and SHA256 of ordered
little-endian u32 token IDs. For each growing edge, the prior accepted anchor
must be an exact token prefix of the current prompt and strictly shorter.
Actual cached usage must be at least that entire prior anchor. Turn one can
therefore reuse S0; turn two onward must reuse the preceding full tool input P.
There is no constant slack and no P-minus-one allowance. Identical-input retry
is not a growing edge and is rejected by this suite.

The helper requires exact seven archived request/response/receipt triples,
request and response byte/hash joins, completed unary or SSE transport (one
finish and one DONE for SSE), and actual response usage equal to receipt usage.
It uses production schema, reasoning controls, HTTP tokenizer and renderer.
All turns retain the same tools/thinking settings and reject overflow rewriting.
For every full-input turn an archived production capture log must contain one
unambiguous matching prompt length, token digest and in-range slot. ANSI color
escapes are removed for field parsing. A capture message alone says publication
is pending: the matching completed response and independently verified source
route/build are also required. A log is evidence, not a cryptographic proof of
execution; independent launch/PID/binary/source review remains mandatory.

Use a clean, independently bound server build containing the experimental
stage, actual Hybrid/TQ8/no-graft profile, and explicit unsafe acknowledgment.
The approval document must contain
`experimental_gemma_full_input_checkpoint: true` alongside existing family
controls (`family: gemma4`, `codebook_bits: 8`, Hybrid, slots and context).
The replay process also requires both flags to exercise the actual production
planner; it must not infer eligibility by relaxing the generic oracle.

Config extends the strict Gemma helper's bound-file envelope with `capture_log`
and a `response` BoundFile in each of seven `turns`. BoundFile is an absolute
path plus `sha256`. Other fields remain `family`, `server_slots` (1 or 4),
`tokenizer`, `template`, `template_source`, `readiness`, `controller_receipt`,
`family_approval`, `server_build_receipt`, expected server commit/binary digest,
and an absolute new output path outside all Git ancestors. Files must be the
retained actual artifacts, not reconstructed requests or current-source
attestations substituted for the executed build.

Root-owned validation commands, not executed by the author:

```sh
cargo test --locked --bin hf2q gemma_full_input_v2_
HF2Q_GEMMA_FULL_INPUT_CHECKPOINT=1 HF2Q_UNSAFE_EXPERIMENTS=1 \
HF2Q_GEMMA_FULL_INPUT_POST_ORACLE_CONFIG=/absolute/private/config.json \
cargo test --locked --bin hf2q \
serve::api::handlers::gemma_full_input_post_oracle_tests::gemma4_full_input_seven_wire_post_oracle \
-- --exact --ignored --nocapture
```

The first filter selects five model-free controls: family/kwargs, clean build
identity, actual-used growing-prefix predicate, capture binding, and raw
response completion/usage. Negatives cover prior-full-input minus one on a
growing edge, partial/old anchor only, changed prefix, missing/wrong usage,
ineligible equality, exact retry misclassified as growth, missing/ambiguous or
wrong or duplicate capture identity/slot fields, unary error bodies or missing/wrong
choice indices, and incomplete/duplicate/wrong-model responses.
The ignored filter must execute exactly one replay. No actual seven-wire replay
has been performed by this author. Output status
`GemmaFullInputCheckpointSevenWireQualified` is specific to this mechanism;
`generic_strict_oracle_qualified`, `model_qualified`, and model inference by the
helper remain false. It proves neither speed nor physical fit nor whole Q5.
