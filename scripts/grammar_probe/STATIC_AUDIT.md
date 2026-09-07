# GCD static audit — hf2q mask path (Matt's "audit finds, battery keeps")

Date: 2026-09-07. Scope: every path where a sampled token can reach the
response while a grammar is attached. Invariant (per the Tantalus write-up):
after all transforms, P(forbidden token) == 0 — mask last, `-inf` semantics,
nothing additive after it.

## Sampling paths walked

### 1. Main CPU path — `engine.rs:7475` and `engine.rs:7684`
Order is correct at both sites: `logit_bias` is applied to the logit vector
FIRST, then `sample_logits_with_grammar` (engine.rs:584 greedy probe /
:598 full mask) masks, then the sampler picks. **Mask is last; nothing
additive after it.** PASS.

### 2. The mask itself — `grammar/mask.rs:165 mask_invalid_tokens_with_eog`
- `-inf` semantics: masked tokens get `f32::NEG_INFINITY` (true negative
  infinity, not a finite sentinel — the "finite sentinel" bug class is absent).
- EOG tokens survive only in an already-accepting state and never participate
  in token-terminal matching (engine.rs:620 `accept_grammar_token` also
  enforces this on commit).
- Empty/undecodable non-EOG pieces are always masked.
- Fail-closed on terminal: `validate_grammar_terminal` (engine.rs:654)
  refuses to report success on a dead or incomplete grammar — this is the
  source of the "grammar constraint was incomplete at length" 500s we saw in
  the B12/W2 campaigns (loud availability error, never a silent drop).

### 3. Qwen35 speculation (MTP) — `engine_qwen35.rs:1247`
`is_qwen_server_speculation_exact_eligible` is deliberately the strictest
gate: stochastic sampling, logit_bias, logprobs, stop-strings, penalties are
ALL closed when speculation is on. When grammar is attached, the verifier
selections route through `sample_logits_qwen35_constrained`
(engine_qwen35.rs:1538) which applies logit_bias then calls
`sample_logits_with_grammar` — the constrained mask path, with the live
grammar state. The serial-MTP variant (`is_serial_mtp_exact_eligible`,
engine_qwen35.rs:1267) requires `params.grammar.is_none()` — i.e. the
non-verifying speculation path refuses to compose with a grammar at all.
**Speculation never bypasses the grammar: it either masks in the verifier or
declines to speculate.** PASS.

### 4. Greedy probe path — `grammar/mask.rs:57 sample_greedy_valid_token`
Temperature-zero probes the top-64 by logit before falling back to the
exhaustive mask. The probe uses `runtime.clone().accept_token(...)` — a
semantic oracle identical to the mask (there is a test-only clone-oracle at
mask.rs:235 asserting byte-identical masks between the two paths). Result is
exactly the highest-logit valid token. PASS.

## Paths confirmed absent (the vLLM FATAL classes)
- **No beam search surface** in hf2q — the one demonstrated FATAL on vLLM
  (`use_beam_search` silently drops the constraint) does not exist here. The
  `use_beam_search` param now 4xx's under the ADR-056 unknown-param rule.
- **Empty-support / mid-generation**: hf2q has no `min_tokens`,
  `allowed_token_ids`, or `bad_words` request params at all (Matt's point 2 —
  the dynamic empty-support class). They now 4xx as unknown params when a
  grammar is attached.
- **`ignore_eos`** does not exist as a request param; EOG is honored only at
  an accepting grammar state by construction.

## The one gap found (now fixed)
Unknown request params were silently accepted (serde's default behavior).
ADR-056 fix: `ChatCompletionRequest.extra` catch-all + handler-side 400 when a
grammar/structured-output constraint is attached. Regression tests:
`test_chat_request_unknown_fields_collect_into_extra`,
`test_chat_request_known_fields_do_not_leak_into_extra`.

## Residual watch items
- The grammar probe harness measures the *response* path. Tool-call emission
  (`ToolCallSplitter` gate at engine.rs:39955) has its own grammar gate — walk
  it in a follow-up before claiming tool-call coverage.
- Z_t instrumentation (HF2Q_ZT_LOG) is the cliff detector; it measures, it
  does not enforce.

## Detokenizer integrity (Matt's point 4)
Production decode paths all use `skip_special_tokens = false`
(engine.rs:7521, engine_qwen35.rs, engine_deepseek4.rs, serve/mod.rs:1762) —
the record is the raw bytes, no display-vs-bytes divergence. A refusal-lexicon
match or the W1 automaton sees exactly what the model emitted. PASS.

## Verdict
The mask-last invariant holds on every path walked. The stack's FATAL classes
from the vLLM battery (beam drop, empty-support emit, ignore_eos re-widening)
are absent by construction or now rejected loudly. The translated battery
(ADR-056) pins this against regressions.
