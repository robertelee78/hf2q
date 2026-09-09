# ADR-057: `--gcd-schema` — schema-constrained output as a refusal-impossibility arm

- **Status:** Draft — accepted direction; not yet implemented
- **Date:** 2026-09-09
- **Related:** ADR-053 (GLP/GCD serving surface), ADR-055 (alphabet presets),
  ADR-056 (conformance battery), ADR-052 (structured-output compatibility)

## Attribution

The schema-as-constraint pattern — forcing model output into a typed object
whose fields leave no slot for refusal prose — comes from Vince Ovando's
red-teaming pipeline (vince@cybersharkconsulting.com, tantalus.io): recon and
exploit agents emit `{attack_context, tool_details, summary, steps}` objects
reviewed with tool context. His measured experience: no refusal issues even
against API providers, until provider-side I/O classifiers began screening
content at the perimeter — the impetus for moving the pattern to the local
inference layer, where no perimeter exists.

## Context

The W1 campaign's failure taxonomy has two classes:

- **Class 1 — template refusals** ("I am programmed to be a helpful and
  harmless AI assistant…"). Lexicon-fixable: exact phrases the membership
  automaton can make ungenerable.
- **Class 2 — semantic pivots** ("The mechanism is called a relationship, and
  it is not something you do to someone else."). The model satisfies the W1
  prose grammar structurally, then pivots to redirect content — because prose
  always leaves a slot for redirect prose. No exact string exists to block.

Schema constraint closes Class 2 at the object level: if the output must be a
typed object with content-bearing fields, a pivot is not blocked but
*unrepresentable* — there is no field in which "I cannot fulfill" typechecks.
This is the same two-axis design as W1 (membership + mass) applied to the
object graph instead of the token stream.

The tradeoff, per Vince: **schema trades adaptability for reliable
structure.** Output is always the object shape — correct for pipelines and
tool-use, wrong for free-form chat. W1 remains the `--gcd` chat arm;
`--gcd-schema` is the pipeline arm.

The honest limit: schema eliminates the pivot *space*, not semantic drift
*within* fields — a model can still emit timid content inside a `steps`
array. That residue is the embeddings gate's layer, unchanged.

## Decision (to implement)

- `hf2q serve <model> --gcd-schema <schema.json>` compiles the JSON schema to
  GBNF at startup via the existing `json_schema.rs` conversion path and
  installs the result as the serve-time default constraint, at the same
  injection point `--gcd` uses (`handlers.rs`: inject when the request carries
  neither `grammar` nor `response_format`).
- `--gcd` and `--gcd-schema` are mutually exclusive on one serve process: one
  default constraint per server. Invalid schemas fail closed at startup
  (compile error, exit) — identical posture to GLP reader conformance.
- Chat forwarding follows the ADR-053 pattern: `hf2q chat <model>
  --gcd-schema <file>` forwards to the chat-owned serve child.
- **Schema authorship is the security boundary.** Fields must be
  content-bearing; a schema with a free-form `notes` or `disclaimer` string
  field reopens the pivot slot by hand. Ship an example red-team object schema
  (`examples/`) with documented field-design rationale.

## Consequences

- Reuses existing machinery wholesale: schema→GBNF conversion, the mask path,
  the injection point, chat forwarding. New surface is one flag + startup
  compile + docs.
- ADR-056 battery gains schema-mode cells: unknown-param canary, mutation
  cases (field removed, type widened), and a pivot-impossibility proof —
  string "I cannot fulfill this request" must be outside the generated
  language of the compiled grammar.
- Validation protocol mirrors W1: offline `gbnf_check.py` accept/reject, then
  a judged spike on the 512-corpus measuring refusal rate under the schema arm
  vs W1's 2.3% (DeepSeek) / 7.2% (Gemma) baselines. Prediction: Class 1 and
  Class 2 both collapse; what remains is within-field drift, gate territory.
- The alphabet preset axis (ADR-055) applies unchanged; schema mode defaults
  to the `english` channel.
