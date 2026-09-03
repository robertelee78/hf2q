# ADR-052: Structured-output compatibility for every generative model

**Status:** Implemented
**Date:** 2026-09-03
**Updated:** 2026-09-03
**Implemented by:** PR #184, merge commit `719366328cfac4a7bb72c602802ccfd8b2de13c0`
**Supersedes:** the closed grammar-subset plan in ADR-005 Decision #6
**Related:** ADR-005, ADR-017, ADR-042, ADR-044, RFC 2119

## Context

hf2q already owns a native GBNF parser, serializer, runtime, vocabulary mask,
OpenAI `response_format` support, model-family tool-call grammars, lazy tool
activation, and grammar-aware cache identity. Those parts evolved in separate
paths:

- `src/serve/api/grammar/json_schema.rs` compiles response JSON Schemas.
- `src/serve/api/registry.rs` separately compiles Gemma, Qwen, and DeepSeek
  tool schemas onto their native wire formats.
- `src/serve/api/grammar/regex_gbnf.rs` enforces tool-schema regex patterns,
  while the response-schema compiler currently ignores `pattern`.
- `src/serve/api/grammar/{parser,serialize,sampler,mask}.rs` implement the
  shared grammar runtime.

The previous ADR-005 subset was intentionally sufficient for the callers that
existed in April 2026. The September 2026 product contract is broader:

1. r2c ReviewLens Stage 6 and CWE Stage 9 schemas are mandatory conformance
   cases, not compiler special cases.
2. hf2q must expose the structured-output modes expected by the current peer
   and vLLM clients.
3. every hf2q text-generating family must obey the same constraint semantics.
4. no supplied assertion may be silently weakened or ignored.

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** in
this ADR are to be interpreted as described by
[RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

## Source-bound baselines

- hf2q integration base: `cfcd79c1835e69bedc6d1d033c1219c5dea10562`.
- local peer research snapshot:
  `9cffdcc801582616250520966699cb5b25d28243`.
- hf2q's existing peer pin:
  `e15384a5cb092b080c2a01c0b9e3f8635079d6df`. That object is not present in
  the local non-shallow peer checkout, so research against the live
  snapshot MUST NOT be mislabeled as pin parity.
- vLLM and XGrammar behavior is bound to their official `main` sources read on
  2026-09-03: vLLM
  `443febe723f62381cda46a9d4f989b8e74a8a857` and XGrammar
  `71ab2256cf06be93e97c22fb5c0b2c6e09893be3`.
- Qwen3-VL native tool-call rendering is bound to the official
  `Qwen/Qwen3-VL-2B-Instruct` template at
  `89644892e4d85e24eaac8bacfd4f463576704203`: one JSON object containing
  `name` and object-valued `arguments`, surrounded by `<tool_call>` and
  `</tool_call>`. It MUST NOT be routed through Qwen 3.5/3.6's distinct
  `<function=...><parameter=...>` body.

The absent peer object was researched rather than silently substituted:
`git fetch origin e15384a5cb092b080c2a01c0b9e3f8635079d6df`
returns `upload-pack: not our ref`, and neither the local object database,
reflogs, nor unreachable-object scan contains it. This ADR therefore creates a
separate grammar-compatibility pin at the inspected local peer checkout commit
`9cffdcc801582616250520966699cb5b25d28243`; it does not change or falsely
validate the repository's broader quantization/parity pin.

## Hypothesis and spike result

### Original hypothesis

The uncommitted Stage 6/9 patch could be completed by adding several missing
keywords to `json_schema.rs`.

### Spike

The model-free baseline passed:

- grammar stack: 167 passed, 1 ignored;
- nested family tool grammars: 27 passed;
- response-format/cache precedence: 7 passed.

The upstream peer grammar parser/runtime binaries also passed. The audit
then found that the green baseline explicitly tests silent `pattern` widening,
loads only three of eight upstream grammar fixtures through absolute paths,
does not expose llama/vLLM structured-output fields, accepts left recursion,
allows empty token pieces around the grammar mask, and applies response-format
grammar while a seeded reasoning span is open.

### Reformulated hypothesis

The required product unit is a bounded structured-output subsystem, not a
larger one-pass JSON-Schema emitter. A request MUST normalize to one typed
constraint. JSON Schema MUST normalize to one shared semantic model. Each
wire-format emitter MAY render that model differently, but it MUST enforce the
same assertions or reject the request. Runtime activation and termination MUST
be explicit state transitions shared by unary and streaming generation.

The initial 8,192-rule, 262,144-element, and 16,384-active-stack resource
hypothesis also rejected required inputs. Measured compiler output is 66,226
rules / 930,013 elements for Stage 6 and 131,226 rules / 1,835,799 elements for
Stage 9. The supported twelve-key reverse-order boundary peaks at 32,768 active
stacks. The limits below are therefore reformulated with bounded headroom above
those measurements; public wire input remains capped independently.

The first complete locked-suite spike then falsified one request-ordering
hypothesis: compiling every public constraint only after model resolution let
an unresolvable model mask malformed schemas and conflicting structured-output
surfaces as `model_not_loaded`. The reformulated boundary performs
model-independent structured validation before model lookup and returns the
field-qualified HTTP 400 there. Only textual token-terminal resolution is
deferred to the selected model's authoritative tokenizer; it is still proved
before decoding begins.

## Decision

### 1. Supported request surfaces

`POST /v1/chat/completions` MUST preserve OpenAI `response_format`:

- `text`;
- `json_object`;
- `json_schema`.

For peer compatibility, `json_object` MAY carry its optional top-level
`schema` member. When present, hf2q MUST compile and enforce that schema; it
MUST NOT silently widen it to the generic JSON-object grammar.

It MUST also accept the current compatibility surfaces:

- the peer-style `grammar` and `json_schema`;
- the peer-style lazy grammar configuration where the runtime can preserve
  equivalent semantics;
- vLLM-style `structured_outputs.choice`;
- `structured_outputs.regex`;
- `structured_outputs.json` and `structured_outputs.json_object`;
- `structured_outputs.grammar`;
- `structured_outputs.structural_tag`;
- vLLM's `response_format.structural_tag` extension.

Exactly one user-output constraint MUST be active after normalization.
Malformed, empty, conflicting, or unknown structured-output constraints MUST
return HTTP 400 and MUST NOT become unconstrained generation.
Model-independent failures MUST be reported before model resolution so they
retain the exact offending request parameter. Vocabulary-dependent token
strings MUST instead be bound after model resolution and before decode.

Existing hf2q tool grammar has precedence when tool choice requires a native
tool-call wire format. That precedence MUST be resolved before compiling an
otherwise-unused response constraint. New compatibility fields MUST NOT be
silently ignored when their combination has no defined meaning.

### 2. Model-family coverage

The same normalized constraint and runtime MUST operate for every enabled text
decoder:

- Gemma;
- Qwen 3.5/3.6;
- DeepSeek 4;
- future registered text-generating families.

Standalone Qwen3-VL is not currently an enabled text-generating family: the
shipping contract and ADR-041 require `hf2q serve` to refuse it before model
load. Its staged Generate and GenerateWithSoftTokens implementation already
consumes the shared grammar runtime and has focused mask/terminal tests, but
that is not live-model proof. ADR-041 MUST NOT remove the startup guard or mark
Qwen3-VL serving implemented until unary response-schema, SSE, native tool,
tool-result continuation, and prefix-reuse grammar gates pass on an
authoritative artifact. This prevents the grammar feature from silently
creating a false claim that the currently disabled decoder can generate.

Family code MAY provide native reasoning and tool markers. It MUST NOT provide
a weaker JSON-Schema implementation. A family that cannot activate a requested
constraint safely MUST return a typed error rather than decode unconstrained.
Embedding-only models are outside this runtime because they emit no text.

### 3. JSON Schema semantic contract

hf2q targets the generator-relevant assertion vocabulary of JSON Schema Draft
2020-12. It does not claim to be a general instance validator.

The first implemented profile MUST support:

- boolean schemas and empty schemas;
- `type`, including nullable/type unions;
- `const` and scalar or container `enum` values;
- local JSON-Pointer `$ref`, `$defs`, and `definitions`, including repeated
  and recursive references within resource limits;
- `anyOf`;
- `oneOf` only when exactly-one semantics can be preserved or proven;
- mergeable `allOf` intersections;
- finite discriminator `if`/`then`/`else` branches needed by r2c;
- object `properties`, `required`, `additionalProperties` boolean or schema,
  `minProperties`, and `maxProperties`;
- array `items`, `prefixItems`, `minItems`, and `maxItems`;
- string `pattern`, `minLength`, `maxLength`, and the current vLLM/XGrammar
  format allowlist;
- integer and number bounds where exact grammar enforcement is possible.

Known unsupported assertions MUST fail with a JSON-pointer-qualified error.
This includes, until implemented exactly, `multipleOf`, `uniqueItems`,
`contains`, `minContains`, `maxContains`, general `not`, dependent schemas,
remote references, and unevaluated-vocabulary semantics. Annotations such as
`title`, `description`, `default`, and `examples` MAY be ignored because they
do not assert instance validity. Assertions MUST NOT be silently ignored.

hf2q-native JSON semantics remain authoritative:

- an empty schema accepts any JSON value;
- absent `additionalProperties` uses the JSON Schema default of open;
- object properties are not semantically ordered;
- the existing any-order grammar for up to eight required properties remains
  supported;
- schemas beyond an exact representable budget fail rather than switching to
  declaration-order-only semantics.

### 4. Grammar dialect contract

The peer GBNF character language MUST remain supported, including all
eight bundled upstream grammar fixtures. hf2q's underscore rule-name extension
MUST remain supported because its serializer and grammar combiner emit it.

The parser MUST add vocabulary-aware token terminals (`<[id]>`, `<token>`, and
negation), canonical serialization, and token-id-aware runtime masking. A token
string MUST resolve to exactly one vocabulary token. Undefined rules, missing
root rules for user grammars, direct or indirect left recursion, invalid token
ids, and unresolvable token strings MUST fail before decode.

XGrammar EBNF extensions and Lark input MUST be implemented only when their
syntax and semantics are proven by immutable upstream fixtures. The common
GBNF subset MUST NOT be described as full XGrammar or Lark compatibility.

The vLLM structural-tag surface MUST accept the pinned XGrammar format-node
vocabulary (`const_string`, `json_schema`, `grammar`, `regex`, `any_text`,
token formats, combinators, tags, and string/token dispatch). Its
`json_schema.any_order=true` mode MUST preserve XGrammar's deliberately
relaxed semantics at every object depth: declared-key/value validity and
entry counts remain enforced, but duplicate keys are permitted and required
key identity is not tracked. Ordinary hf2q response schemas MUST retain their
stricter unordered semantics.

XGrammar's non-JSON `JSONSchemaFormat.style` values and deprecated
`qwen_xml_parameter` node are model-specific argument serializers, not
standard grammar languages. They are outside this compatibility baseline and
MUST return a typed request error. hf2q's supported Qwen and DeepSeek4 native
tool calls instead MUST use the source-bound family emitters above; adding a
future MiniMax, GLM, Cohere, Kimi, or different DeepSeek wire requires its own
model-family contract and fixtures.

### 5. Runtime activation and termination

Output constraints MUST be request- and slot-local.

- Ordinary structured output activates eagerly at the first answer byte.
- When the prompt starts inside a native reasoning span, response constraints
  MUST remain suspended until the registered reasoning-close marker completes.
- Automatic tool grammar remains lazy until its native open marker.
- Required and named tool grammar remains eager unless it is waiting for a
  seeded reasoning close.
- Split-marker and marker-plus-body tokens MUST advance the runtime exactly
  once.

EOS, stop strings, cancellation, or token-budget exhaustion MUST NOT turn an
unaccepted constrained output into normal success. Empty or undecodable
non-terminal token pieces MUST NOT bypass the grammar. An all-invalid candidate
set MUST produce a generation error; it MUST NOT sample from all negative
infinity logits.

Until stop matching can validate a stripped suffix identically in every unary
and SSE family path, a non-empty `stop` value combined with an effective output
constraint MUST be rejected before generation. An empty stop array is inert
and MAY be accepted. This is an explicit fail-closed request limitation, not a
runtime grammar fallback.

Unary, SSE, speculative verification, cache replay, and multi-slot execution
MUST implement identical grammar-state transitions.

### 6. Resource limits

Untrusted constraints MUST be bounded before runtime. Initial limits are:

- raw grammar or serialized schema input: 1 MiB;
- compiler-generated rules: 262,144;
- total compiler-generated grammar elements after expansion: 4,194,304;
- schema traversal depth: 64;
- resolved local references: 1,024;
- object properties: 32 per object;
- unordered required properties: 12 per object (Stage 6 requires nine);
- choices/enumerants/structural-tag structures: 1,024;
- total choice/enumerant literal bytes: 1 MiB;
- repetition bound: 2,000;
- runtime active stacks: 32,768 (the measured peak for the supported twelve-key
  reverse-order boundary; the 16,384 hypothesis rejected it at its seventh
  object member);
- structural-tag formats: 256, nesting depth: 32;
- lazy-trigger buffer: 64 KiB.

Exceeding a limit MUST return a deterministic typed error. Limits MAY be
tightened from measured spike data, but MUST NOT be raised without adversarial
compile/runtime evidence.

### 7. Test and provenance contract

CI MUST be hermetic. It MUST NOT require a local peer checkout, `/tmp` diagnostics,
or a network connection.

The repository MUST contain attributed, revision-bound fixtures for:

- all eight current peer-bundled grammars;
- the peer parser/runtime positive and negative vectors, including tokens and
  left recursion;
- current vLLM/XGrammar request modes and accepted/rejected schema vocabulary;
- r2c ReviewLens Stage 6 and CWE Stage 9 schemas;
- every current hf2q family and native tool/reasoning wire format.

Each supported assertion MUST have at least one positive and one mutant case
that falsifies only that assertion. Each rejected assertion MUST have an HTTP
400 test. Model-free tests MUST cover request normalization through grammar
masking and terminal state. Realistic model gates MUST cover unary, SSE,
reasoning, tool calls, tool-result continuation, and unchanged-prefix reuse for
every available generative family before the work is merged.

Differential harnesses MAY execute exact external reference binaries during
validation, but neither production conversion nor serving may depend on them.

## Architecture

The implementation is sequenced by dependency:

1. `schema.rs` defines typed external constraint shapes.
2. A structured-output normalizer resolves precedence and conflicts.
3. A shared JSON-Schema semantic layer resolves references, validates
   assertions, applies intersections/conditionals, and enforces budgets.
4. Response JSON, family tool-wire, regex, choice, grammar, and structural-tag
   emitters consume the validated representation.
5. The GBNF parser validates recursion and resolves token terminals.
6. The sampler and mask consume character or token terminals and implement
   bounded trigger activation.
7. Engine terminal paths enforce accepted-state completion uniformly.
8. Cache identities include the normalized constraint and activation kind.

The integration worktree is the only writing lane. Independent audit and
reference lanes remain read-only.

### Token-terminal refinement evidence

The isolated `feat/grammar-token-terminals` lane, based on
`ecae7bdda931ea05000cebab33402660585ba034`, tested the grammar vocabulary
hypothesis against the local peer snapshot before implementation. The
result requires token ids to remain distinct from decoded byte text:

- `TOKEN = 8` and `TOKEN_NOT = 9` preserve the upstream element encoding;
- `<[id]>` and `!<[id]>` parse without a vocabulary, while textual terminals
  require a model-bound resolver and exactly one token id;
- binding a concrete tokenizer also rejects numeric ids absent from that
  vocabulary before generation;
- masking and advancement receive both token id and decoded bytes, so token
  terminals use identity while character terminals retain byte semantics;
- declared EOG ids bypass token terminals, survive only in an accepting state,
  and otherwise terminate fail-closed; empty non-EOG pieces are rejected.

The focused grammar suite and
`cargo check --locked --all-targets --all-features` are the proof boundary for
this lane. Request normalization, handler wiring, full-suite validation, and
real-model gates remain completion-gate work rather than inferred success.

## Current proof ledger

The implementation worktree has passed the model-free and build gates:

- the focused grammar suite passed 262 tests with one explicitly
  hardware/tokenizer-gated diagnostic ignored;
- the family registry suite passed 138 tests with two exact-tokenizer
  diagnostics ignored;
- the request-router suite passed 51 tests, including pre-model error
  attribution for malformed and conflicting structured-output requests;
- `cargo check --locked --all-targets --all-features`,
  `cargo test --locked`, and `cargo build --release --locked` passed;
- the nightly branch-instrumented full binary suite passed 5,020 tests with 55
  explicitly gated tests ignored. LLVM then crashed while merging the
  repository-wide raw profile set, after test completion; the valid focused
  profile remains the coverage authority.

The focused LLVM profile measures grammar code at 83.03% lines, 80.33%
functions, 85.67% regions/statements, and 66.60% branches. Agentic-QE consumes
those real values and reports 78.91% when averaging its four dimensions, below
its default 80% advisory threshold. Its separate quality command also fails
closed because it has no measured Cargo `testsPassing` evidence adapter. These
tool verdicts MUST NOT be relabeled as green and do not replace the locked Rust
gates above.

The realistic serving-family gate passed on the exact branch binary at
`776e4858a4ccf54619725f01ba223a900938b341` on an Apple M5 Max. Models were
loaded one at a time and shut down cleanly:

- Qwen3.5-4B Q4_K_M: the native-tool unary/SSE/recovery/continuation gate
  passed; repeated requests reused 392-450 prompt tokens. Stage 6 and Stage 9
  `response_format=json_schema` outputs passed live validation. An initial
  Stage 6 run exhausted the request's completion budget before reaching an
  accepting grammar state and returned an error rather than incomplete JSON;
  the reformulated 768-token run passed.
- Gemma4 Ara 2pass: the same native-tool gate passed with 171-216 cached
  prompt tokens; Stage 6 and Stage 9 passed live validation.
- DeepSeek-V4 Flash 0731 agentic Q2: the same native-tool gate passed with
  487-542 cached prompt tokens; Stage 6 and Stage 9 passed live validation.
  The 100.05 GiB artifact was loaded only after static capacity validation and
  with the launcher's explicit controlled-diagnostic memory override because
  the otherwise-idle host retained 23.83 GiB historical swap and 8.36 GiB in
  the compressor.

The native-tool evidence is reproducible with
`scripts/test_deepseek4_structured_tools.sh`. The r2c unary and response-format
SSE evidence is reproducible with `scripts/test_r2c_structured_outputs.sh`.
Standalone Qwen3-VL was not loaded: no authoritative GGUF is present and its
documented ADR-041 startup guard makes artifact preparation insufficient.

## Completion gates

This ADR moves to **Implemented** only when:

1. focused positive and mutant tests pass;
2. all compatibility fixtures are hermetic and revision-bound;
3. `cargo check --locked --all-targets --all-features` passes;
4. `cargo test --locked` passes;
5. `cargo build --release --locked` passes;
6. realistic unary/SSE/tool-result/prefix-reuse evidence passes for every
   enabled text-generating family, and disabled staged families retain an
   explicit activation gate requiring the same proof;
7. the exact feature commit is pushed, ordinary exact-head CI passes, and the
   pull request is merged to `main`.

## Consequences

The public surface grows, but every mode converges on one constraint lifecycle.
Some schemas accepted historically because hf2q ignored their assertions will
now return HTTP 400. That compatibility break is intentional: rejecting an
unsupported constraint is safer than claiming success while generating output
that violates it.

The supported grammar-language baseline is therefore the peer-compatible
GBNF plus vLLM choice, regex, JSON Schema, grammar, and structural-tag modes.
It is not a claim that every model-specific serializer or every validation-only
Draft 2020-12 vocabulary can be represented as a context-free generation
grammar. Unsupported assertions and serializers remain typed errors.
