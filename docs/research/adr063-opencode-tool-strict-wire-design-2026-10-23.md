# OpenCode non-strict function-tool wire compatibility

Status: proposed prerequisite design, implementation approval pending. Work context: ADR-063 F1 (#356) real-client inventory finding. No production code changes, runtime qualification or performance claims are included in this design.

## Observed failure

At runtime source `c173c9a854ec0a15e0f70abc769a825e377846d9`, the unmodified OpenCode v2.1.1 agent requests for all four intended model profiles contain twelve function tools, each with `strict:false`. The offline renderer's shared `ChatCompletionRequest` parse fails with unknown field `strict`. `ToolFunction` in `src/serve/api/schema.rs` declares only name, description and parameters under `deny_unknown_fields`. The production `chat_completions` handler takes the same typed JSON extractor and maps a rejection to invalid-request output before model preparation. This establishes a source/schema incompatibility; a full-model HTTP run was not performed.

The synthetic coding fixture omitted the field and therefore did not cover this client request. Preserve both its successful token captures and the separate failed actual-client capture with their qualification limits. No request field may be stripped from recorded wire evidence to manufacture a pass.

## Proposed behavior

Accept absent, null and boolean false for `tools[].function.strict` as non-strict mode. Preserve the existing tool name, description and parameter schema unchanged. Normalize this redundant non-strict field out of the model-facing serialized function definition so explicit false/null produces exactly the same rendered prompt and token IDs as omission. Preserve the original raw request separately for provenance.

Boolean true continues to fail explicitly before model preparation, with a clear unsupported strict-tool-mode error. This change must not silently promise strict schema enforcement, silently discard a true constraint, enable grammar-constrained decoding, alter the operator's GCD policy, or relax rejection of arbitrary unknown fields. Invalid non-boolean values reject. This is compatible with the [official function-calling distinction](https://developers.openai.com/api/docs/guides/function-calling) between non-strict mode and strict schema enforcement; full strict-tool support is a separate contract.

## Minimal implementation

Keep the public `ToolFunction` fields and serialization shape unchanged. Replace only its derived deserializer with a small closed wire representation that additionally reads nullable `strict: Option<bool>`. Reject `Some(true)` with a specific error; map false/null/omitted into the existing three-field `ToolFunction`. Both outer Tool and inner wire function retain `deny_unknown_fields`. Do not add a new field to every existing Rust constructor or carry a false hint into templates as a new rendering input.

This is hf2q HTTP/schema policy. It needs no native API, kernel, dependency update or package publication by itself. No fallback process or external inference component is introduced.

## Required controls

1. Parse full chat requests for absent, false and null; assert the same normalized tool definitions and unchanged parameters/description. Retain a representative twelve-tool, text-only request shape without private paths or credentials.
2. Reject true with the precise unsupported-mode error, plus strings, numbers, objects, arrays and unknown inner/outer tool fields. Other request controls and GCD policy must retain their existing behavior.
3. Exercise the HTTP request-extraction/error boundary without a model: valid non-strict requests must pass JSON parsing, invalid strict requests must produce structured invalid-request output before engine work. An uninitialized-model response is distinct from successful generation and must be labeled accordingly.
4. In the reviewed offline capture helper, parse the unmodified captured client requests for all four profiles. Check false/null versus omitted prompt bytes and token IDs with each family's effective template and HTTP tokenizer. Capture receipts remain clean-source and exact-binary bound.
5. Run the relevant locked schema, handler and GCD contracts. The failure fixture must fail before the fix; a passing test that merely tests a substitute parser does not establish production acceptance.

Independent review must assess the immutable production delta and its controls. Real multi-turn tool execution, unary/SSE generation, prefix reuse and four-slot quality/performance remain mandatory under Q5/F2 before claiming serving or speed qualification. No model load is authorized by this design.

## Approval and delivery

ADR-063 approved F1 inventory/audit entry while requiring separate implementation designs. The integration owner has asked the operator to approve this newly discovered prerequisite change; execution waits for that answer. Keep this design linked to the F1 capability finding and record the chosen story/acceptance boundary before code edits. Do not relabel a patched stock executable as the published v0.1.27 baseline: any shared baseline compatibility adaptation needs an explicit, independently reviewed comparison disposition and source identity.
