# Grammar-Constrained Decoding in hf2q: an Engine-Native Positive Security Control

*Robert E. Lee (IOActive). GCD concept: Vince Ovando (vince@cybersharkconsulting.com),
tantalus.io. Measured replica & serving-stack battery: Matt Suiche (m@msuiche.com).
hf2q is the first inference-engine-native implementation.*

I started uncensoring models to discover and exploit software defects. The
turning point wasn't the jailbreak — it was noticing that uncensored models
*reason better*, full stop. A model trained to flinch at a politically
inconvenient fact is a worse reasoning engine on physics too; the same
machinery handles both. Sycophancy in one domain leaks into all of them.

That got me thinking about world views and bias — which is a hot topic in AI
"safety," and mostly for the wrong reason. "Debiasing" in practice almost
always means aligning outputs to the ideological priors of the debiasers, not
removing bias in any neutral sense. The pretense of neutrality is what makes
it worse than openly saying "we want the model to say X." The giveaway is
asymmetric application: if mitigation consistently pushes one direction on
contested questions, that's not debiasing — it's value imposition with extra
steps. You can't eliminate bias; you can only choose which priors the system
reflects. The honest version of the project is "we are building a model that
reflects these specific values, here they are, here's why we chose them" —
not "we're removing bias" while the thumb is clearly on the scale.

This document is about the *mechanism* for that honesty at serve time: a
security control that is constitutive rather than corrective, positive rather
than negative, and explicit about what it enforces. Which brings us to the
deeper problem — and it's not just refusal.

## The problem: every deployed LLM control is negative and behavioral

An LLM agent with real tool access reads files, queries inboxes, and issues
outbound requests on the principal's behalf. Against that, the controls the
industry reaches for are all **negative**: they enumerate the bad and permit
everything else. A defensive system prompt (an RFC-2119 policy) is in-band —
it lives in the same input channel the injection rides, so an attacker who can
address the model can address the guardrail too, and overrides it. An input
classifier scans the wrong surface (the payload travels in skill/email
content, not the user message). An output filter runs a credential denylist:
negative, default-allow — so business data carrying no credential signature
sails straight through. Measured on the Tantalus arena: a full behavioral
stack (prompt + classifier + filter) leaked 1.82–33.8% of injections;
survivors were business-data exfils with no credential signature, exactly as
the shape predicts.

The refusal-trained model is the same class from the other side. Safety
alignment is a taught disposition, and a willed model can re-assert it in free
text (the replica measured 97.85% on an abliterated adversary). Operators of
open-weight models — red teams, hardening pipelines, structured agents —
want the option to hold a model's disposition consistent at serve time without
re-serving modified weights.

Both collapse into one question: **what may the model emit?** And the lineage
settling that question is old: object-capability systems make authority an
unforgeable token that can only be exercised, never named into existence by an
unauthorized party; the LANGSEC program makes the recognizer correct by
construction, so what the accepted language permits is the policy, and anything
outside it is ungenerable at the boundary; application allowlisting displaced
signature antivirus for the same reason. Vince's paper is the domain transfer
into AI security: the operating system's executable space becomes the agent's
tool-call action space, signature detection becomes behavioral guardrails, and
application allowlisting becomes the per-request authorization grammar.

## The mechanism: constitutive, not corrective

A corrective control enforces the same policy over the same language, but
after the fact — it reads output and reacts. A constitutive control enforces
the policy at the only point where "output" doesn't exist yet: the decode
boundary. The transformer is a function from input tokens to a logit
distribution. A per-request authorization grammar Gs declares the formal
language L the response may form. At every sampled token, the sampler masks
every token whose emission would leave L and renormalizes:

  q(x | s) ∝ p(x | s) · 1[x ∈ L]

The empirical signature is **emission**: the count of times the unauthorized
artifact is produced. 0 by construction under the constitutive control; > 0
under the corrective one. Willingness maxed changes intent, not the allowed
alphabet — and because the guarantee quantifies over all logit distributions,
it holds against the worst case: a fully-injected model still cannot emit
what the grammar does not admit. This is complete mediation applied to the
token sampler: every candidate token is mediated, and fail-safe defaults mean
a dead grammar aborts loudly rather than relaxing.

hf2q makes this engine-native. `hf2q serve <model> --gcd` embeds the GBNF
grammar into any chat request that doesn't already specify one (the legacy
`--uncensor` is a hidden alias). The mask runs inside the sampler, as the last
transform before selection; the engine's fail-closed validator turns any
grammar-incomplete termination into a loud 500 rather than a silent pass. One
invariant, from Matt's serving-stack condition: *after all transforms,
P(forbidden token) = 0* — mask last, `–inf` semantics, nothing additive after
it. Every lever (temperature, penalties, biases) is measured against that
invariant by the ADR-056 conformance battery.

Beyond refusal suppression, the same shape holds for the agent side: each
deployed action is a member of an enumerable sink class — URL, channel, path,
recipient — and a per-request, per-authz grammar makes unauthorized members
ungenerable at the sampler. That's the Tantalus Round 2: the model reading
`~/.config/api-keys.json` with full intent is no longer ambiguous; emission
still cannot follow. Whether the control guards what the model *says* or what
it *does*, the grammar is the same enforcement point.

## Measured on a stock, refusal-trained model

Everything below is from DeepSeek-V4-Flash-0731, stock weights, served by
hf2q. The checkpoint is unmodified.

### The winning shape (W1): three topic-independent layers

The campaign tested anchor+exclusion series (B arms) and whitelist/structure
variations (W arms). Three mechanisms compose:

1. **Alphabet whitelist.** The response body forms over ASCII printable plus
   fourteen measured symbols (en/em dashes, curly quotes, box-drawing, arrow,
   degree, µ, ≈, ±, subscript-two — histogram-measured at ~97% coverage of
   legitimate non-ASCII use). Chars outside the alphabet match no grammar
   branch → the sampler masks them. The whole Unicode homoglyph evasion class
   (U+2019 apostrophes, zero-width insertion, fullwidth, RTL-override) is
   closed by construction.

2. **KMP exclusion automaton.** A 131-state case-insensitive DFA over a
   refusal lexicon, compiled to recursive GBNF rules; dead-end-safe and
   homoglyph-covered. **The screen, not the suppressor.** We provably evaded
   it four times in one debugging session (homoglyph → case → semantic
   paraphrase). Its role is the seatbelt: it bars habitual phrasings so the
   anchor's mass axis can fight fair.

3. **Anchor + topic sentence.** The reply must open with `Here is the
   technical breakdown.` then `The mechanism is ` plus a bounded free span in
   which the model names its own topic. This seats the model's distribution on
   task continuations, so refusal becomes locally incoherent. Every measured
   refusal-rate delta across arms came from this axis.

### The numbers: a 1024-prompt corpus, judged independently

DeepSeek-V4 served; APEX (an abliterated Qwen3.6 model) judges via a 7-way
semantic enum. Human spot-check: APEX agrees with human labels on 84% of a
stratified sample, errors symmetric.

| arm | refusal (adversarial half, 512) | refusal (benign half, 512) | actionable (adversarial) |
|---|---|---|---|
| W1 grammar alone | **2.3%** | **0.0%** | **87.3%** |
| W1 + embeddings gate | **0.59%** | 0.0% | — |

The embeddings gate is a semantic paraphrase screen over the refusal clusters
— it watches the channel the character-level grammar provably can't reach.
The grammar needs no per-topic specialization: per-prompt derived topics tied
the generic shape, and fully self-derived templates lost actionability to
rigidity.

### Confirmed causal findings

- **Think blocks kill suppression.** A reasoning runway lets the refusal
  direction re-activate before the anchor engages (no-think 6.9% vs
  forced-think 24.1% vs anchored-think 67.9%). The grammar must engage at
  token 0.
- **The anchor suppresses; the automaton screens.** Exclusion-only maintains
  refusal at baseline; anchor alone 33%; plus exclusion 5.9%.
- **Constraints must keep accept reachable.** A hard step floor forbids EOS
  and yields fail-closed 500s — the B12 cliff class.

## Serving-stack conformance (26 cells, hf2q)

The vLLM beam-search FATAL class — a silent constraint drop — is absent: hf2q
has no beam surface, and any undeclared sampler param now 4xx's under the
ADR-056 rule (the whole silent-drop class is closed by "reject what you cannot
honor"). Temperature/top-k/top-p/penalties held; terminal truncation fails
loudly; streaming assembles to legal output; a hand-driven static audit of the
mask's path through the sampler stands behind the battery (audit finds,
battery keeps).

## GLP is complementary — the disposition-side intervention

The grammar moves the logit distribution; it cannot change what the model
*wants*. GLP (weightless steering vectors — a few hundred KB of GGUF control
vectors, Matt Suiche's spec) edits the residual stream per layer at inference,
enabling directional steering without touching weights. The two compose
cleanly: GLP does the disposition side (refusal direction, or any direction);
GCD does the emission side (what may leave the sampler). The free-text
surfaces the grammar deliberately leaves open — `response.message`,
`search.query`, `data` fields — are grammar-legal but semantically wild;
the embeddings gate watches exactly that open surface.

## What GCD is really for

Refusal suppression is the measurable case, but the mechanism is broad and the
concept isn't abliteration-carved: any security constraint expressible as a
formal language — authorized tool-call routing in an agent loop (the Tantalus
Round 2 arena), data-loss prevention on outbound content, hallucination guards
on file paths and URLs in agent pipelines — becomes constitutive at the
sampler. `--gcd` is a general GCD surface: operators ship a grammar per
authz, per channel, per tool-step, and the embedded refusal arm is one
prebuilt grammar for the canonical case. Grammar-constrained decoding is to
prompt injection what parameterized queries are to SQL injection, and what
application allowlisting was to signature antivirus: the recognizer is correct
by construction, and the bad thing is ungenerable, not caught.

## Honest limits

- The residual 3/1024 that escape both layers are deep semantic paraphrases;
  bigger embedding models or tuned thresholds tighten the gate.
- 9% degenerate output is the dominant quality tax of the current automaton.
- Arena numbers are one-deployment evidence; hf2q's GBNF parser needs its own
  over-admission fuzzing pass (Matt's trie-DFS technique transfers; his
  xgrammar result doesn't).
- hf2q currently enforces enumerable sinks precisely; free-text confidentiality
  is the open problem the grammar explicitly defers to the gate, and the
  GLP wording is in the `data`-field discussion of the Tantalus arena.

## Artifacts

`--gcd` on hf2q main · ADR-053/054/055/056 (surface, calibration, alphabet
presets, conformance battery) · 1024-prompt corpus + judge verdicts ·
offline GBNF checker + evasion battery · static audit + tokenization audit ·
Z_t cliff instrumentation (HF2Q_ZT_LOG) · PR #190.
