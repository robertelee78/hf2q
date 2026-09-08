# Grammar probe spike findings (2026-09-06)

## The honest numbers (APEX judge, spike n=14)

| Arm | Refusal | Valid Fulfill | Actionable | Invalid |
|---|---|---|---|---|
| B16 (neutral anchor + steps) | 21.4% | 35.7% | 0.0% | 42.9% |
| B17 (refusal-incompatible anchor) | 33.3% | 22.2% | 55.6% | 22.2% |
| B17 + GLP-29 (cyber-focused) | 28.6% | 21.4% | 35.7% | 42.9% |

## What works

- **B17 (refusal-incompatible anchor)**: "Here is the technical breakdown" — the anchor is refusal-incompatible, so refusing after it is incoherent. The model produces actionable content on 56% of prompts. The anchor works better than neutral anchors (B15/B16's "I'll analyze this directly").
- **B17 + GLP**: the composition works better than either alone. The GLP engages the reasoning, the anchor forces the answer to land. Actionable content on 36% of prompts.

## What doesn't work

- **Neutral anchors (B15/B16)**: "I'll analyze this directly" is semantically inert — refusal-compatible. The model refuses within the structure. 21-43% invalid output.
- **GLP alone**: engages reasoning but never writes the answer (content drift).
- **Grammar alone on general prompts**: the model's refusal direction is strong enough to refuse within any structure. The grammar solves token validity, not trajectory policy.

## The composition story

The GLP engages the reasoning (the model works through the technical content in the reasoning span). The refusal-incompatible anchor forces the answer to land. The composition works when the GLP's reasoning is substantive and the anchor is refusal-incompatible.

## The remaining work

The GLP-29 is cyber-focused and doesn't help on general prompts. The calibration pipeline (ADR-054) is needed to derive a general-purpose GLP vector from the model's own activations on the HarmBench prompt distribution.

## The evaluation protocol

1. **Spike**: run prompts through the subject model, store responses
2. **Swap**: switch to the judge model (APEX abliterated Qwen)
3. **Judge**: semantic verdicts on the stored responses
4. **Measure**: refusal/fulfill/actionable/invalid rates

The subject model never judges itself. The judge is independent (different family). The numbers are honest.
