# Tokenization-seam audit: CLOSED (hf2q's mask is correct)

Matt's xgrammar audit verifies the char-to-token lift non-trivially. Our
trie-DFS audit asked: can any token sequence decode to a string that the
W1 automaton must exclude?

**Answer: hf2q's mask closes the seam by construction, verified empirically.**

## The theoretical seam (what the naive view missed)

Raw-token audit found 3 lexicon phrases reachable as *separate* tokens:
- `can't` = token `can` + token `'t`
- `cannot` = single token `␣cannot` (space-prefixed)
- `won't` = token `won` + token `'t`

A naive character-level automaton that only sees single tokens would pass
`can` (legal) and then `'t` (legal) — and "can't" appears in the decoded text
despite the lexicon. That's the seam.

## Why hf2q is NOT vulnerable

`GrammarRuntime::accept_token` feeds each token's **decoded bytes** through
the automaton **char-by-char**, and the automaton state (KMP partial-match
position) **persists across tokens**. The mask calls `accept_token` for each
candidate, so the automaton sees the uninterrupted decoded text stream.

Empirical verification (offline simulator + live grammar):

```
token 'can' passed (state now b18s18)     # partial match "can"
token ''t' BLOCKED at char 't'            # completing char = no branch -> masked
```

When the model tries `can` + `'t`, the second token's `'` char drives the
automaton to state `can'`, and `t` matches no branch → `-inf` mask → the
token is never sampled. The seam is closed at the byte level **because the
mask evaluates decoded text, not token-piece-ness.**

## What the audit actually confirmed

1. hf2q's mask evaluates **decoded text** through a character-level automaton
   whose state persists across token boundaries — the correct construction.
2. Both single-token (`␣cannot`) and cross-token (`can`+`'t`) forms are
   caught, because the automaton sees the decoded stream either way.
3. The earlier B20 evasions were NOT tokenization seams — they were the
   automaton's dead-end hole (completing chars fell through the catch-all),
   fixed in the W-series builds before the full-corpus run.

## Residual: the true seam classes that remain

- **Special-token & invalid-UTF-8 inside free-text fields** (Matt's two
  findings) — our whitelist alphabet closes both by construction; verified
  against the W1 grammar's charset.
- **Post-decode rewriting** — hf2q decodes with `skip_special_tokens=false`
  everywhere in production (STATIC_AUDIT.md). No display-vs-bytes divergence.

## Artifacts

- `trie_dfs_audit.py` (the raw-token DFS that found the 3 theoretical seams)
- the empirical mask-evaluation simulator (this file's rationale)
- `STATIC_AUDIT.md` (the mask-last invariant walk)
