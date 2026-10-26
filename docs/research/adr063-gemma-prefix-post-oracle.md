# Gemma4 exact seven-wire post-unload prefix oracle

Test-only source candidate; author has not run imports/tests/builds/models. Three
pure controls plus one ignored actual-config replay. No production code changes.
Use only after model unload; no engine, tensor, device or network is constructed.

The replay calls production Gemma render/tokenize/reasoning helpers twice per
actual request: P with generation cue, S without. It requires nonempty strict S
prefix P, exact actual prompt usage, coldcached0, and each continuation P starts
with previous S with cached>=previous S length. S==P is unproved/fails. No Qwen
preserve_thinking injection or DS8-tail slicing. Current Gemma omissions resolve
no extra template kwargs; original thinking defaults and actual reasoning_content
remain in the seven full wires. Changed history that removes prior reasoning
must fail, not become an allowance or forced preserve_thinking treatment.

Config deny-unknown-fields: family gemma4; server_slots1or4; hash-bound absolute
files tokenizer,template,readiness,controller_receipt,family_approval,
server_build_receipt; template_source GgufEmbedded; explicit expected_server_source_commit
40lowerhex and expected_server_binary_sha25664lowerhex; exactly seven turns each
{request,receipt}; new absolute private output. BoundFile is {path,sha256}.
Family approval must bind Gemma context262144/slots/Hybridtrue/graftfalse. Actual
runtime semantic tokenizer/template hashes must match. Merely supplying the
reviewed F1 export or canonical APEX label does not prove selected Googleartifact.
Root must bind the exact bf65 selected GGUF/launch and source equivalence through
independent evidence; helper build-receipt checks are configuration identity,
not self-attestation of the server binary. Controller must prove protocol,
positive final tests and cleanup, same processPID as readiness. N4 requires four
independent seven-wire replays/reviews. A loaded projector is not treated absent.

Root-controlled commands, after clean compile and all model unloads:

```sh
cargo test --locked --bin hf2q gemma_prefix_ -- --nocapture
HF2Q_GEMMA_PREFIX_POST_ORACLE_CONFIG="$PRIVATE_CONFIG" \
  cargo test --locked --bin hf2q gemma4_exact_prefix_post_oracle -- --ignored --nocapture
```

The exact ignored name is module-qualified in the test binary; when using
`--exact`, supply the listed full name (normally
`serve::api::handlers::gemma_prefix_post_oracle_tests::gemma4_exact_prefix_post_oracle`),
or remove `--exact` and require exactly one selected replay. No zero-filter pass.
Outputcreate_new0600 outside Git records actual config/evidence/source hashes,
row token hashes and observed cached counts. PassedExactGemmaSevenWireReplay is
not model quality, performance, physical KV proof or full Q5 acceptance. Preserve
failure raw artifacts; no tolerance/threshold retrofit or template fallback.


## Separate partial-failure analyzer

`gemma_prefix_post_oracle_tests::partial_failure::gemma4_two_wire_prefix_failure_analyzer`
is ignored and metadata-only, selected with `HF2Q_GEMMA_PREFIX_FAILURE_ANALYZER_CONFIG`.
It reuses the same closed Config/BoundFile inputs but requires exactly TWO wires
and an actually FailedOrHeld/cleanup-proved controller, not protocol success.
For this separate analyzer only, each turns[].receipt may bind a real JSON
receipt/response or raw SSE response when the client failed before publishing
its receipt; exactly one actual usage record is required. Raw DONE count is
reported rather than claimed as full SSE protocol qualification.

It runs the production renderer/tokenizer and reports P/S lengths, strictness,
S==P, LCP of prior S against current P, exact prior-prefix relationship and actual
cached/usage counts, plus bound readiness/source/binary and raw hashes.
The output is permanently AnalyzedGemmaPrefixFailure/NotQualified/seven_wire_pass=false.
It neither invokes accept_prefix nor weakens the existing three pure controls
or strict seven-wire replay. The failed real run remains failed. This analyzer
can decide whether a tool-response prompt with S==P still retains the prior
strict stable prefix, before investigating cold-batch routing independently.
No execution or real-model proof is asserted by adding this source.
