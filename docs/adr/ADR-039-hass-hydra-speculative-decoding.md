# ADR-039 — HASS and Hydra speculative decoding research

- **Status**: Proposed — no implementation approval or model qualification.
- **Date**: 2026-05-23
- **Updated**: 2026-10-23 (ADR-063 F3 primary-source correction)
- **Supersedes**: nothing
- **Related**: ADR-037, ADR-038, ADR-063

## Historical intent and correction

The May proposal followed the operator's request to investigate HASS/Hydra
after EAGLE-3. That research intent remains. Its original technical premise was
incorrect: HASS is not a later EAGLE-3 extension, Hydra is not HASS's earlier
name, and neither establishes a joint technique as the current best approach.
The former H1–H10 implementation plan, loss formulas, shared-transformer heads,
shared drafter-KV assumption and LOC estimates are withdrawn pending design.
They were proposals, not measured hf2q evidence. Historical bytes remain in Git.
No failed benchmark is being converted into an accepted performance claim.

## What the primary references establish

[HASS, Zhang et al., first submitted August 2024](https://arxiv.org/abs/2408.15766),
means HArmonized Speculative Sampling. Its training objective and recursive
context alignment address differences between training and drafting in the
EAGLE setting. It predates [EAGLE-3, March 2025](https://arxiv.org/abs/2503.01840).
This supports investigating a training recipe, not assuming an arbitrary HASS
checkpoint fits `Eagle3DrafterTensors`. Feature sources, dimensions, token maps,
weights and recurrence must match the actual checkpoint.

[Hydra, Ankner et al., February 2024](https://arxiv.org/html/2402.05109v2),
is a separate Medusa-related design whose lightweight draft heads condition on
preceding candidate tokens. The basic head uses a single-hidden-layer MLP; Hydra++
also studies prefix attention. This does not establish a shared EAGLE-3
transformer or a single attention KV cache across all heads. Those choices must
come from the selected implementation and artifact, not the earlier proposal.

[EAGLE-2](https://arxiv.org/abs/2406.16858) supplies the dynamic-tree precedent;
EAGLE-3 changes the drafter to token prediction with multi-layer target features.
Reusing tree verification is a possible integration surface, not proof that the
loader, hidden-state interface, verification or rollback can remain unchanged.

Published results use different targets, hardware, sampling and workloads.
The former acceptance-length comparisons and projected compound L≈4.5–5.5
are withdrawn. There is no matched hf2q evidence here that HASS beats EAGLE-3,
that Hydra compounds with it, or that either accelerates any supported family.

## Replacement deciding spikes — Proposed

1. Select and pin an official checkpoint and implementation, target revision,
   license, tokenizer and tensor schema. Check availability and compatibility
   for Qwen3.8, DeepSeek4, Qwen3.6 and Gemma4 independently. Missing artifacts
   are an explicit blocked research outcome, not grounds for silent substitution.
2. Specify the exact graph, feature collection, candidate topology, verifier,
   state ownership, accepted-prefix commit and rollback contracts. Test whether
   existing EAGLE-3 interfaces fit; never infer compatibility from a method name.
3. Prove forward and verification correctness against the selected primary
   reference before enabling serving. Distinguish exact sampling from relaxed
   acceptance criteria. Validate tool semantics, unary/SSE continuation, prefix
   reuse and semantic TTFT on realistic multi-turn coding requests.
4. Compare ordinary decoding and each qualified drafter on identical artifacts,
   prompts, settings and host conditions. Measure draft/verify cost, acceptance,
   memory, prefill/decode rates and client-visible gaps with alternating paired
   runs. No cross-paper ratio substitutes for those measurements.

Any new kernel or reusable primitive belongs in mlx-native. hf2q owns checkpoint
loading, family graphs, cache orchestration, serving policy and product gates.
A native change requires publication followed by exact hf2q pin/lock updates and
clean-artifact validation. Whether new native primitives are needed is an open
spike result, not a promise of zero native changes.

## Approval and release boundary

These spikes do not approve a combined HASS–Hydra architecture, a default head
count, training infrastructure, or a serving flag. A successor design must
record its actual evidence and obtain the governing approval before execution.
ADR-063 governs current release work and excludes custom training pending its
explicit decision gates. This research ADR supplies no v0.1.28 speed-win receipt.
