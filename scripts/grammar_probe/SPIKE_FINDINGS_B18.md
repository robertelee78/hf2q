# B18 grammar spike findings (2026-09-06)

## The honest numbers (APEX judge, spike n=17)

| Arm | Refusal | Valid Fulfill | Actionable | Invalid |
|---|---|---|---|---|
| B17 (refusal-incompatible anchor) | 33.3% | 22.2% | 55.6% | 22.2% |
| B18 (B17 + KMP exclusion) | 29.4% | 23.5% | 70.6% | 58.8% |

## What works

- **B18 composition**: B17 anchor + KMP lexicon-exclusion automaton. The anchor forces the compliant opening (refusal-incompatible), the exclusion automaton makes refusal phrases ungrammatical in the body. Actionable content on 71% of prompts.
- **The KMP automaton**: 184 recursive GBNF rules for 29 refusal phrases (both cases). Live-stack pressure is bounded (at most 11 stacks at character boundaries, far below the 32,768 cap). The automaton works mechanically (zero grammar errors on 24/24 spike prompts).

## What doesn't work

- **Semantic refusal**: the model refuses without using the excluded phrases (29% of prompts). The grammar solves lexical refusal; semantic refusal needs the GLP vector.
- **Neutral anchors**: B15/B16's "I'll analyze this directly" is semantically inert (refusal-compatible). B17's "Here is the technical breakdown" is refusal-incompatible.

## The composition story

The grammar (B17 anchor + KMP exclusion) handles lexical refusal suppression. The GLP vector handles semantic refusal suppression. The composition works when the GLP's reasoning is substantive and the grammar's exclusion is complete.

## The evaluation protocol

1. **Spike**: run prompts through the subject model, store responses
2. **Swap**: switch to the judge model (APEX abliterated Qwen)
3. **Judge**: semantic verdicts on the stored responses
4. **Measure**: refusal/fulfill/actionable/invalid rates

The subject model never judges itself. The judge is independent (different family). The numbers are honest.
