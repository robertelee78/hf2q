# ADR-055: Grammar Alphabet Channel Presets

## Status
Backlog (accepted direction, not yet implemented).

## Context

The refusal-suppression grammar campaign (W-series arms) added a character-whitelist
axis to the grammar program: the body of the response must form over a *positive*
alphabet rather than "anything except forbidden strings". The measured default is
ASCII printable (`0x20-0x7E`) + 14 curated symbols (em/en dash, curly quotes,
box-drawing, arrow, degree, µ, ≈, ±, subscript-two) — the histogram over 777
unconstrained responses showed that set covers ~97% of legitimate non-ASCII use
for English technical prose. The whitelist closed the entire homoglyph/numeric-
codepoint evasion class by construction (measured during the B20 debugging session).

Vince's deployment principle: **the alphabet is a per-channel declaration, not a
global constant**. English prose, code, and math have different legitimate sets,
and the choice carries a security property (homoglyph identifiers in code,
invisible NBSP, bidi-override are all supply-chain tricks).

## Decision (to implement)

Add `--grammar-alphabet <preset>` to the serving surface, resolved at grammar
compile time (not per-request), with presets built from `char_class_from_allowed()`:

| preset | alphabet | notes |
|---|---|---|
| `english` (default) | `0x20-0x7E` + `\n\t` + curated 14 | current campaign default |
| `code` | `0x20-0x7E` + `\n\t` | tightest; masks Unicode identifiers, NBSP, bidi |
| `code-unicode-idents` | `code` + identifier-category allowances | only if there's a measured need |
| `math` | `english` + extended symbol block | °, µ, Greek letters, operators |
| `multilingual` | explicit per-language blocks | e.g. CJK ranges, Cyrillic |

Non-presets remain expressible in-line in grammar files; the flag is a
convenience for the common channels and a place to hang documented security
rationale.

## Consequences

- The refusal-campaign grammars (W-series) use `english` implicitly today; no
  behavior change there.
- A `--grammar-alphabet code` server would reject Unicode-identifier-containing
  code by mask — the right default for security-adjacent serving.
- The histogram measurement should be rerun per channel before locking a preset
  (the current 14-symbol set was measured on prose, not code).
- Agentic-QE can gate the presets with Z_t noninferiority checks (alfabet choice
  must not move the cliff detector on a paired corpus).
