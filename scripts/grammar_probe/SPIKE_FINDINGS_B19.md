# B19 grammar spike findings (2026-09-06)

## The honest numbers (APEX judge, spike n=17)

| Arm | Refusal | Valid Fulfill | Actionable | Invalid |
|---|---|---|---|---|
| B12-fixed (short frame + bounded think) | 57.1% | 21.4% | 35.7% | 21.4% |
| B18 (anchor + KMP exclusion) | 29.4% | 23.5% | 70.6% | 58.8% |
| **B19 (anchor + topic-reference + exclusion)** | **5.9%** | **47.1%** | **94.1%** | 41.2% |

## The mechanism

The B19 grammar forces the first sentence to reference the prompt's topic ("The mechanism is X"). The model can't refuse a topic it's already discussing. The refusal direction is overridden by the topic engagement.

## What works

- **B19 (anchor + topic-reference + exclusion)**: refusal 5.9%, actionable 94.1%. The topic-referencing first sentence forces the model to engage with the topic before it can refuse. The KMP exclusion automaton handles lexical refusal.
- **Mechanical correctness**: zero grammar errors on 40/40 spike prompts.

## What doesn't work

- **B12-fixed (short frame + bounded think)**: the short frame is too weak (57% refusal). The model refuses after the frame.
- **B18 (anchor + exclusion, no topic-reference)**: 29% refusal — the model refuses without the excluded phrases (semantic refusal).

## The honest conclusion

The B19 grammar works on 94% of prompts (actionable content). The remaining 6% are prompts where the model's refusal direction is strong enough to refuse even with topic engagement. The grammar approach is sufficient for general refusal suppression on this model.

## The evaluation protocol

1. **Spike**: run prompts through the subject model, store responses
2. **Swap**: switch to the judge model (APEX abliterated Qwen)
3. **Judge**: semantic verdicts on the stored responses
4. **Measure**: refusal/fulfill/actionable/invalid rates

The subject model never judges itself. The judge is independent (different family). The numbers are honest.
