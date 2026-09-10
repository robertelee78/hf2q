# ADR-057: `--gcd-schema` — schema-constrained output as a refusal-impossibility arm

- **Status:** Accepted — implemented (commit cefaa341, shipped in v0.1.21):
  serve flag, fail-closed startup schema compile, W1-point injection in the
  handlers, chat forwarding, `examples/recon-opportunities.schema.json`, and
  the `gcd_schema_tests` battery cells (including the tightened-subschema
  and observability refinements from issue #192). The lockdown mode
  (`--gcd-schema-locked`) documented below is implemented on branch
  `fix/gcd-schema-locked`: the flag (requires `--gcd-schema`), the 400
  rejection in the handlers ahead of the undeclared-params check, chat
  forwarding, and the `gcd_schema_lockdown_tests` battery cells.
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

Vince's second note supplies the mass half of the argument: **a universal
schema works across every model because JSON is the highest-prior structure
in the training distribution** — models are trained on more JSON than any
other form, so none of them fight the constraint. Where the W1 anchor
overpowers refusal mass, the schema arm rides the mass gradient: the model's
strongest structural prior *is* the constraint. Same two-axis design
(membership + mass), applied to the object graph instead of the token stream.
This predicts the schema arm should need no anchor at all and should
generalize across families with less per-model tax than W1 showed
(DeepSeek 2.3% vs Gemma 7.2%) — testable against the 512-corpus protocol.

The honest limit: schema eliminates the pivot *space*, not semantic drift
*within* fields — a model can still emit timid content inside a `steps`
array. That residue is the embeddings gate's layer, unchanged.

**Two refinements from review (Matt's agent), both accepted:**

1. **Schema kills the pivot, not the refusal.** `{"summary": "I cannot help
   with this.", "steps": []}` typechecks. The schema eliminates
   refusal-as-structure; refusal-as-*content* can still live inside string
   fields, and the schema itself enables a subtler shape: formal compliance
   with emptied content (`{"steps": []}`, minimal strings) — refusal by
   vacuity. The vacuity fix is free: JSON Schema's `minItems`, `minLength`,
   and `required` all compile to GBNF bounds, so the arm ships **tightened
   subschemas by default** (see Decision). Scope matters and is stated
   flatly: these bounds make refusal-by-EMPTIED-CONTENT unrepresentable;
   no schema bound makes refusal PROSE unrepresentable — an independent
   validator accepts an object carrying refusal text in every required
   string field of the shipped example, and catching that residue is the
   embeddings gate's job, not the schema's. The remaining layers port to the
   object level unchanged: the W1 exclusion automaton runs over string-field
   content, the pinned anchor can move inside the schema as a per-field
   prefix (subject to verification of json_schema.rs pattern support), and
   the embeddings gate watches paraphrase drift.
2. **The downstream consumer is part of the control.** Schema-constrained
   output feeds typed consumers — pipeline code that expects steps to be
   executable. Vacuous or timid content fails *loudly* at the consumer: a
   refusal hidden in a free-text chat reply is a silent success, but the same
   refusal in a pipeline field is a crash. The schema arm's deeper strength
   is not just that refusal is unrepresentable — it is that residual refusal
   becomes *observable*. The W1 chat arm can never have that property.

**The symmetry nobody had said out loud:** Tantalus Round 2 was already the
schema arm — typed tool calls with enum sinks are schema-constrained output
on the authorization axis, and it held 0/1,140 while W1-style prose
suppression carries a 2.3% residual. The schema arm's existence proof is not
hypothetical; it is the round that could not be broken.

## Decision (to implement)

- `hf2q serve <model> --gcd-schema <schema.json>` compiles the JSON schema to
  GBNF at startup via the existing `json_schema.rs` conversion path and
  installs the result as the serve-time default constraint, at the same
  injection point `--gcd` uses (`handlers.rs`: inject when the request carries
  neither `grammar` nor `response_format`).
- `--gcd` and `--gcd-schema` are mutually exclusive on one serve process: one
  default constraint per server. Invalid schemas fail closed at startup
  (compile error, exit) — identical posture to GLP reader conformance.
- **Lockdown mode (control vs convenience).** Default injection defers to a
  caller-supplied `grammar`/`response_format` — right for refusal-suppression-
  as-convenience, wrong when the constraint is a security boundary. Tantalus
  lesson: server-side defaults that are controls must be non-overridable.
  `--gcd-schema-locked` (or `[serve] gcd_schema_locked`) makes the constraint
  mandatory: requests carrying their own `grammar`/`response_format` are
  rejected 400 rather than deferred to.
- Chat forwarding follows the ADR-053 pattern: `hf2q chat <model>
  --gcd-schema <file>` forwards to the chat-owned serve child.
- **Schema authorship is the security boundary.** Fields must be
  content-bearing; a schema with a free-form `notes` or `disclaimer` string
  field reopens the pivot slot by hand. Ship an example red-team object schema
  (`examples/`) with documented field-design rationale.
- **Tightened subschemas are the default posture.** The arm's value is
  "refusal-unrepresentable *and* vacuity-resistant", and an untightened
  schema is neither: `additionalProperties: false`, explicit `required`, and
  `minItems`/`minLength` on every collection and string field. The top-level
  list's `minItems` is a policy knob: left open it admits the honest negative
  ("recon found nothing"); in suppression deployments it is set to 1.
- The mask path is schema-agnostic — GBNF from a hand-written grammar and
  GBNF from a schema compile to the same object — so this arm is a frontend
  change, not a sampler change (confirmed against the measured stack: same
  injection point, same mask, same fail-closed validator).

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
  **The JSON-prior claim needs a control arm:** one non-JSON-shaped grammar
  (deeply nested or XML-ish) in the same spike — otherwise the experiment
  cannot separate "JSON specifically rides the mass gradient" from "structure
  generally does," which is exactly the generality mechanism the claim rests
  on.
- The alphabet preset axis (ADR-055) applies unchanged; schema mode defaults
  to the `english` channel.
