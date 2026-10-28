# Q1 Gemma T2 packed-cohort source adapter (prospective)

This extends the reviewed ordinary endpoint slice with one separate ignored
T2 filter. It is not T6 cancellation or a new serving/scheduling policy. No
model execution, numerical acceptance or F2 physical certificate is asserted.

The existing production `forward_prefill_batched_multi_seq` concatenates four
independent128-token inputs, installs their actual slot views and sequence
length/offset/start descriptor, and executes one512-row body. Its attention
uses the existing per-sequence isolation paths. The diagnostic head extension
now requires that exact descriptor to join a frozen four-sequence map: four
distinct slots, each start0/length128, contiguous offsets0/128/256/384. It maps
all512 actual hidden rows back to (agent,absolute token), rejects a lone512-row
sequence masquerading as four agents, and retains exactly the original four
last-row tokens/logit returns. Extra head rows are captured after their existing
checked finishes; they are not additional logical inputs or a performance run.

The test-only engine child reuses `Gemma4KvGuard`, the production prefill API,
and `commit_gemma4_slot_cursors`; cursor publication follows successful checked
prefill, not merely dimensions or submitted work. Decode uses the existing
production-default batched body with four actual token/slot/position arrays.
It respects `HF2Q_BATCHED_BODY=0` by refusing this axis rather than overriding
it. It preserves the existing fused-head eligibility or existing separate
`lm_head_batched` fallback. No flag is changed. It reuses the production cursor
validation/publication helpers, and checked body/head output is downloaded as
four actual vocabulary rows. Capture identifies these downloaded CPU Vec rows
as CPU parents, not invented GPU resource identities.

These are source-specific operator statements: prefill token-wise projections
and FFN consume the packed stream; attention preserves sequence boundaries;
normal prefill heads are per-sequence scalar heads; decode has an actual four
input row body and selected fused-or-batched head. This does not label every
inner kernel B4, infer native membership from a command-buffer union, or replace
the separately required F2 native witness/certification. Actual native kernel
selection and empirical execution remain unqualified until captured.

The same full144-per-agent logits and active Hybrid cache manifests at128/144
are required. Fixed input vectors, context262144, output8192, Hybrid dtype/
geometry, ownership and resource checks are unchanged. All four slots must be
fresh before prefill, every decode input position must match the real cursor,
and final cache capture requires144. The separate test filter refuses an
ordinary-axis packet and the ordinary filter refuses T2. T5/T6 remain missing
actual Gemma producers. All original endpoint/profile results remain unchanged.

New hosted controls exercise the actual packed-owner geometry parser and the
engine adapter's distinct-slot guard. Existing ordinary scope controls retain
multi-sequence mismatch and omitted-row refusals. Root must compile and execute
those controls before preparing a guarded actual Gemma T2 case. No CPU oracle
or new numeric tolerance is introduced; backend findings remain separate from
oracle completeness and independent task quality.
