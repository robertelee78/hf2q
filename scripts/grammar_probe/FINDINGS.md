# GCD Campaign Findings — W1 full-corpus measurement

**Date:** 2026-09-08 · **Arm:** W1 (B19F shape + character whitelist)
**Subject:** DeepSeek-V4-Flash-0731 (stock, refusal-trained, q2_K GGUF)
**Judge:** APEX (abliterated Qwen3.6-MoE) — 7-way semantic enum, fail-closed
**Corpus:** OBLITERATUS `prompts.py` — 512 harmful + 512 benign, stride-sampled
**Branch:** `feat/glp-uncensor` · PR #190

---

## Headline

A GBNF grammar — no weight edits, no steering vectors, no training — suppresses
refusal to **2.3% on the adversarial half** and **0.0% on the benign half** of a
1024-prompt corpus, with 87.3% judge-scored actionability on harmful prompts.

Adding the embeddings gate (semantic paraphrase screen) takes the residual
refusal to **0.59%** (3/512) at a cost of 0.3% false positives on legitimate
fulfillments.

| layer stack | refusal (harmful, n=512) | refusal (benign, n=512) | actionable (harmful) |
|---|---|---|---|
| unconstrained (measured baseline, earlier arms) | ~100% | — | ~0% |
| **W1 grammar alone** | **2.3%** | **0.0%** | **87.3%** |
| **W1 + embeddings gate** | **0.59%** | **0.0%** | (gate screens, grammar writes) |

## The W1 shape (three layers)

1. **Alphabet whitelist** (Vince): the response body must form over ASCII
   printable + 14 curated symbols (em/en dash, curly quotes, box-drawing,
   arrow, degree, µ, ≈, ±, subscript-2 — histogram-measured at ~97% coverage of
   legitimate non-ASCII use). Chars outside the alphabet match no grammar
   branch → the sampler masks those tokens. The Unicode-homoglyph evasion
   class (U+2019 etc., observed live in the B20 debugging session) is closed
   by construction, not enumeration.
2. **KMP exclusion automaton**: 131-state case-insensitive DAF over the refusal
   lexicon, compiled to recursive GBNF rules. Dead-end-safe (the completing
   char matches no branch and the token is masked — a bug class found and fixed
   during the campaign). Acts as the seatbelt: stops the habitual phrasings.
3. **Anchor + topic sentence** (the actual suppressor): the response must open
   with `Here is the technical breakdown.` then `The mechanism is <free>` —
   seating the model's distribution on task continuations so refusal becomes
   locally incoherent. The topic span is free text: the *model* names the
   topic, which is why the grammar needs no per-prompt specialization (B21's
   per-prompt derived terms tied W1; W4's self-derived templates lost 30 points
   of actionability to rigidity).

## Confirmed causal findings (spike series, all APEX-judged)

- **Think blocks are fatal to suppression.** B19F no-think 6.9% vs B20
  forced-think 24.1% vs W3 anchored-think 67.9% refusal. Any reasoning runway
  lets the refusal direction re-activate before the anchor engages. The grammar
  must engage at token 0 (`hf2q_enable_thinking: false`).
- **The automaton is a screen, not a suppressor.** It was evaded 4× in one
  session (homoglyph → case → case+homoglyph → pure semantic paraphrase). It
  provably blocks exact strings; it cannot block a *meaning*. Refusal is not a
  context-free language.
- **Structure constraints must keep accept reachable.** A hard 3-step floor
  (W2's first cut) forbade EOS and produced fail-closed 500s — Fable's "a
  minimum forbids end-of-sequence" confirmed live. W2 with EOS at every step
  boundary: 8.0% refusal but lower actionability (64%) than W1 — the structure
  cost more than it bought.

## Serving-stack conformance (ADR-056 battery, 26 cells, hf2q)

18 PASS / 1 WARN / 5 FAIL / 3 SKIP. The one real bug: **unknown request params
were silently accepted** — fixed (`extra` catch-all + 400 naming the keys when
a constraint is attached). The vLLM FATAL class (beam search silently drops
the grammar) **does not exist in hf2q** — no beam surface; `use_beam_search`
now 4xx's under the new rule. Fail-closed 500s on temp-2.0 / rep-pen-2 /
max-tokens-5 are availability-class, safe by construction. Static audit
(mask-last invariant walk) in `STATIC_AUDIT.md`: PASS on all sampling paths.

## Z_t cliff detector

hf2q now logs pre-mask admissible mass (`HF2Q_ZT_LOG`). Full-corpus trace:
5916 mask calls, 88.7% at z<0.01 — the expected signature at temp 0, where the
mask runs only when the model's top-64 candidates were all illegal (the hard
positions). **Zero mechanical failures** (0 grammar-incomplete 500s) across
1024 responses — the B12 cliff class is gone at corpus scale.

## Honest limits

- The 3 residual refusals that escape both layers are deep semantic paraphrases;
  a bigger embedding model or a tuned threshold tightens the gate further.
- 91 degenerate (9%) is the dominant quality cost — the automaton's dead-ends
  occasionally produce repetitive text. Quality knob, not a safety leak.
- Judge agreement with human labels is unmeasured (spot-check pending).
- Single subject model. The grammar shapes are model-agnostic by construction
  (no think-block, no calibrated bounds), but "universal grammar set" needs a
  second subject to confirm.

## Artifacts

`full_results_w1.jsonl` (1024 responses) · `full_verdicts_w1.jsonl` (1024 APEX
verdicts) · `full_gate_w1.jsonl` (gate cross-tab) · `zt_full_w1.jsonl` (Z_t
trace) · `gbnf_check.py` + `battery_accept/reject.txt` (offline harness) ·
`STATIC_AUDIT.md` · ADR-053/054/055/056.

## Attribution

GCD concept (both faces — tool-call authorization and forced abliteration):
**Vince Ovando** (vince@cybersharkconsulting.com), tantalus.io.
GLP / weightless control vectors: **Matt Suiche** (m@msuiche.com).
hf2q is the first inference-engine-native implementation (`--gcd`).
