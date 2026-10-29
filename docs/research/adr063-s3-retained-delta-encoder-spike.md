# Retained Delta encoder spike (unaccepted source)

Composed on af82f58a, including the corrected exact-family parity fixture,
with the V2 retained-Delta source reviewed independently. Native PR73 merged
as 55906d98 and published 0.15.12. The registry checksum, source VCS identity,
672 packed source files, tag and release assets were verified before this
exact Cargo pin. No local Cargo patch is used. Compilation, focused host
controls, real-model parity and measured serving wins remain pending.

## Narrow observable contract

Only the selected seq1 DeltaNet ops1-9 CB changes: checked shared graph submits
it asynchronously, moves the EXACT CommandEncoder (no rotation/raw escape)
into same-thread bounded ownership, and checks its Result after the EXISTING
healthy singleton attention-row-copy fence. The existing fence remains before
roots suffix truncation and ON24 pool reset. Older Error plus healthy anchor
fails the whole invocation; every retained entry is checked even after Error.
Unknown/error/panic paths quarantine the profile and retain pending encoders
and graph roots. No scalar replay/reset follows an uncertain submission.

Every other selected helper retains its old individual synchronous health
check. In particular embedding IDs remain synchronous, and full-attention
ops1-4/dumps/SDPA/LayerEncoder stages remain checked, preserving optional CPU
reads and their original lifetime boundaries. Shared FFN fence and final head
CPU-read completion remain unchanged. This is deliberately not a blanket
replacement of checked_active branches. Default scalar/teacher/prefill branches
are byte-unchanged. No scheduler, AUTO cost, sampler, math or kernel changes.

Native command_buffer_uses_unretained_references() is checked before
Invocation.begin/cache mutation and again at scope entry. Actual true refuses
with invocation Fresh; no environment approximation or observer-only invalid
flag substitutes for this authority. The caller's existing Fresh refusal path
can use scalar before shared mutation. Unsupported unretained mode is not
silently permitted merely because a CB is retained.

## Submission and ownership map

| Producer | CB/check | Owners through completion |
|---|---|---|
| packed embedding | unchanged checked helper | local IDs live through original check; model table/graph output retained |
| each selected linear-attention row | one actual seq1 ops1-9 CB deferred | pre-norm/projection/L2 outputs, scalar parameter buffers, qkv and Delta temporaries use actual pooled_alloc_buffer; pool in_use roots remain until row reset; weights and per-slot conv/recurrent owners live in model/cache; graph input/position/output retained in graph roots |
| existing row-copy CB | unchanged commit_and_wait Result THEN every retained Delta Result | graph attention/output and current-row roots survive both checks |
| full-attention stages/SDPA | unchanged original checked Results | existing roots/pooled buffers and direct owners retain old boundaries; no asynchronous conversion before optional dump/read |
| packed postattn RMS/DenseQ FFN | unchanged checked layer fence + empty/deferred check | graph dedicated input/residual/output and original pooled FFN scratch |
| final logits/nextn | unchanged checked/CPU-read producer | actual durable postRMS nextn rows preserve whole parent |

The selected Dense shape is64layers,48linear/16full,2..4owners. Therefore at
most48*4=192 deferred CBs are admitted over one invocation; at most ONE exists
at a row fence. The vector reserves one encoder slot before any enqueue.
Admission count is checked and encoder ownership is stored BEFORE commit.
No buffer-parent allocation is added and no pool reset is delayed beyond the
existing row fence; retained native encoder/CB/driver overhead is finite-count
but its physical bytes are Unknown, not zero. No fit or retirement credit.
Native quantized matmul dispatch uses the supplied encoder; the selected
seq1 branch performs no CPU downloads or internal pool reset between its
submission and return. The old per-row graph fence is the completion anchor;
queue order alone does not establish health.

## Failure and controls

Five authored host controls call actual mode/admission/check/release helpers:
unretained refusal; per-fence and total exact boundaries; check every entry
and keep first error; retain every owner on failed check; empty/healthy paths.
Two additional authority controls prove scalar checked_fixture never enters graph deferral and graph-only scope restores independently. No controls were run by the author. Existing invocation failed-stop and root
retirement controls remain. A failed fence skips status-release/reset and
scope Drop deliberately leaks uncertain encoder owners; run already leaks
uncertain graph roots and quarantines profile. No Drop-as-success or implicit
wait. Native observing callbacks may make reentry fail, never authorize reset.

Root must publish/verify/pin native0.15.12, compile/check focused controls,
review this bounded source, prove real target/cache/MTP/carry parity and
actual serving quality/prefix behavior, and measure matched performance before
promotion. First candidate and OFF control remain unqualified negative evidence.

## V2 authority correction

V1 incorrectly attached retention to CheckedScope, which scalar checked_fixture
also uses without graph row fences. V1 is preserved and remains blocked. V2
installs a distinct GraphDeferralScope only inside actual run. Delta defers
ONLY when both checked causal scope and graph deferral authority are active.
Scalar checked_fixture retains its original immediate checked commit. Bounds
remain one per fence/192 total; no increase masks the missing scalar fence.


## Published-native six-trial serving decision (2026-10-29 UTC)

Candidate `b9b0b6a6b36104fdf2ab1fcb4fcfb1055d05952b` used published
native 0.15.12, release binary SHA256
`c534de3fc928747ab23555f2a8dc3426b4b881624e266894f93d370170fea3cc`.
Matched scalar baseline `f34d33791855b49edce11c828b8d5174bc3cea4c`
used the same published dependency, binary SHA256
`f4fccc3d9d8dcb43fccaf2c7f6d7f49710377efab61db8269de4c227d913ca1b`.
Both used canonical four-agent coding inputs, C262144/N4/TQ8/AUTO/ON24,
temperature 0, repetition penalty 1.05, 8192 maximum output tokens,
and unchanged tool grammar/settings. Every run completed 28 HTTP requests,
24 correct tool calls and four correct coding tasks. All four Rust prefix
post-oracles passed for every run. Sampling streams are own-output continuations;
their differing output token counts are not token-matched throughput evidence.

The fixed crossover sequence and release-to-task-completion makespans were:

| Arm | Trial | Seconds |
| --- | --- | ---: |
| Candidate | C1 | 127.815465 |
| Scalar | A1 | 129.397703 |
| Scalar | A2 | 125.817797 |
| Candidate | C2 | 132.059172 |
| Candidate | C3 | 130.011151 |
| Scalar | A3 | 116.770954 |

Candidate median 130.011151 seconds versus scalar median 125.817797 seconds
is **3.332878% slower**. All six observations remain included. A separate
C3 preflight held for Fair thermal state before any model load; the fresh
replacement started Nominal under unchanged guards. Runtime Fair is permitted
by the existing policy. This is a negative engineering decision, not a complete
F2/Q3 confidence-qualified comparison or a whole S3 acceptance.

The independently audited width-two diagnostic separately passed full
output/nextn/cache bit parity on the actual selected inputs. It does not qualify
widths three/four, serving cancellation, actual command-buffer errors, or speed.
Final independent six-trial audit SHA256:
`9c6d69b22b7c60f2ddbb4b540a9ea1ea0f6887c751138c6a7bbf91c3edb82bc9`.
Durable decision: https://github.com/robertelee78/hf2q/issues/363#issuecomment-6486895390.

**Stop adoption of this candidate as a performance change.** Retain this isolated
branch as failed-experiment evidence. Do not retune AUTO cost thresholds to hide
the regression. The next test compares actual normal unchecked and retained
shared producers on identical cold-prepared state to identify the target/fence
bottleneck; that diagnostic is not a replacement serving acceptance test.
