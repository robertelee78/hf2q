# ADR-052: Structured-output compatibility for every generative model

**Status:** Accepted — implementation and proof in progress
**Date:** 2026-09-03
**Updated:** 2026-09-03
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
2. hf2q must expose the structured-output modes expected by current llama.cpp
   and vLLM clients.
3. every hf2q text-generating family must obey the same constraint semantics.
4. no supplied assertion may be silently weakened or ignored.

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** in
this ADR are to be interpreted as described by
[RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

## Source-bound baselines

- hf2q integration base: `cfcd79c1835e69bedc6d1d033c1219c5dea10562`.
- local llama.cpp research snapshot:
  `9cffdcc801582616250520966699cb5b25d28243`.
- hf2q's existing peer pin:
  `e15384a5cb092b080c2a01c0b9e3f8635079d6df`. That object is not present in
  the local non-shallow llama.cpp checkout, so research against the live
  snapshot MUST NOT be mislabeled as pin parity.
- vLLM and XGrammar behavior is bound to their official `main` sources read on
  2026-09-03: vLLM
  `443febe723f62381cda46a9d4f989b8e74a8a857` and XGrammar
  `71ab2256cf06be93e97c22fb5c0b2c6e09893be3`.

The absent peer object was researched rather than silently substituted:
`git fetch origin e15384a5cb092b080c2a01c0b9e3f8635079d6df`
returns `upload-pack: not our ref`, and neither the local object database,
reflogs, nor unreachable-object scan contains it. This ADR therefore creates a
separate grammar-compatibility pin at the inspected `/opt/llama.cpp` commit
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

The upstream llama.cpp grammar parser/runtime binaries also passed. The audit
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

## Decision

### 1. Supported request surfaces

`POST /v1/chat/completions` MUST preserve OpenAI `response_format`:

- `text`;
- `json_object`;
- `json_schema`.

It MUST also accept the current compatibility surfaces:

- llama.cpp-style `grammar` and `json_schema`;
- llama.cpp-style lazy grammar configuration where the runtime can preserve
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

Existing hf2q tool grammar has precedence when tool choice requires a native
tool-call wire format. That precedence MUST be resolved before compiling an
otherwise-unused response constraint. New compatibility fields MUST NOT be
silently ignored when their combination has no defined meaning.

### 2. Model-family coverage

The same normalized constraint and runtime MUST operate for every text decoder:

- Gemma;
- Qwen 3.5/3.6;
- Qwen3-VL text decoding;
- DeepSeek 4;
- future registered text-generating families.

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

The llama.cpp GBNF character language MUST remain supported, including all
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

Unary, SSE, speculative verification, cache replay, and multi-slot execution
MUST implement identical grammar-state transitions.

### 6. Resource limits

Untrusted constraints MUST be bounded before runtime. Initial limits are:

- raw grammar or serialized schema input: 1 MiB;
- rules: 8,192;
- total grammar elements after expansion: 262,144;
- schema traversal depth: 64;
- resolved local references: 1,024;
- object properties: 32 per object;
- unordered required properties: 8 per object;
- choices/enumerants/structural-tag structures: 1,024;
- total choice/enumerant literal bytes: 1 MiB;
- repetition bound: 2,000;
- runtime active stacks: 16,384;
- structural-tag formats: 256, nesting depth: 32;
- lazy-trigger buffer: 64 KiB.

Exceeding a limit MUST return a deterministic typed error. Limits MAY be
tightened from measured spike data, but MUST NOT be raised without adversarial
compile/runtime evidence.

### 7. Test and provenance contract

CI MUST be hermetic. It MUST NOT require `/opt/llama.cpp`, `/tmp` diagnostics,
or a network connection.

The repository MUST contain attributed, revision-bound fixtures for:

- all eight current llama.cpp bundled grammars;
- llama.cpp parser/runtime positive and negative vectors, including tokens and
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

## Completion gates

This ADR moves to **Implemented** only when:

1. focused positive and mutant tests pass;
2. all compatibility fixtures are hermetic and revision-bound;
3. `cargo check --locked --all-targets --all-features` passes;
4. `cargo test --locked` passes;
5. `cargo build --release --locked` passes;
6. realistic all-family unary/SSE/tool-result/prefix-reuse evidence passes or
   an unavailable artifact is named as a merge blocker;
7. the exact feature commit is pushed, ordinary exact-head CI passes, and the
   pull request is merged to `main`.

## Consequences

The public surface grows, but every mode converges on one constraint lifecycle.
Some schemas accepted historically because hf2q ignored their assertions will
now return HTTP 400. That compatibility break is intentional: rejecting an
unsupported constraint is safer than claiming success while generating output
that violates it.
