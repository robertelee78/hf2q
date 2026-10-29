# Qwen3.8 partial shared-target serving: initial negative result

Status: unqualified experiment; keep implementation isolated from main.

The initial design batches post-attention RMS, DenseQ FFN and output-head work,
with singleton attention and checked completion at every causal helper boundary.
It preserves real AUTO/history/MTP continuations, request-owned semantics and
the existing ON24 scratch policy. Published mlx-native 0.15.11 is the exact
dependency in both arms.

## Source-bound observations

| Trial | Source | Sharing | Four-task makespan | Completion tokens |
|---|---|---|---:|---:|
| Initial baseline | `f6c4adce58e59558857bd2b190464428480fc3d3` | Original scalar | 115.544542 s | 3,518 |
| First candidate | `a17f7fa122a30a0f319f4f6091a5358f4276ad3e` | Enabled | 129.192972 s | 3,609 |
| Same candidate control | `a17f7fa122a30a0f319f4f6091a5358f4276ad3e` | Disabled | 126.623544 s | 3,554 |
| Original-source bracket | `f6c4adce58e59558857bd2b190464428480fc3d3` | Original scalar | 124.261773 s | 3,589 |

Each trial completed the unchanged canonical four-agent, seven-turn protocol.
The candidate passed 28 HTTP requests, 24 real tools, four correct patches,
20 sandboxed compile/test processes and all 24 exact full-prefix follow-ups.
Its 86 shared groups were all pairs; no triple or four-owner group was observed.
The maps retain singleton attention explicitly and do not constitute a full
native-dispatch F2 witness.

The candidate's observed makespan is 11.8123% above the initial baseline and
2.0292% above its same-source OFF control. These are individual own-output
observations, not qualified multiple-run medians. Final model-text differences
follow differing compiler paths/timing text, so the extra completion tokens
cannot be attributed to a numerical decoder difference from these runs.
The original-source bracketing control also passed independent raw-quality
and exact-prefix audits. Its 124.261773 s makespan reinforces the need for
matched repeated observations; it does not turn the candidate into a win.

The earlier checked-mode target microbenchmark improved its median by 3.3942%,
but that result did not carry into the canonical serving observation. Recorded
pair service had median 63.5905 ms per owner versus 36.3645 ms for singleton
service within the candidate. Shapes and positions differ; this is a lead,
not a matched kernel speed ratio. D/K allocation arithmetic was correct.

## Reformulated, unaccepted hypothesis

The partial graph adds checked singleton-row copies, checked layer boundaries
and helper waits absent from the normal asynchronous scalar route. A bounded
spike may retain the exact helper encoders after asynchronous submission, then
check each command buffer after the existing healthy row/layer fence and before
pool reuse. mlx-native 0.15.11 already exposes the necessary encoder lifetime
and Result-bearing completion operations. The spike additionally needs the
actual captured reference-mode accessor introduced by mlx-native PR73 for
0.15.12, rather than rereading an environment variable. No kernel change is
implied.

The spike must bound retained encoders and scratch, preserve every causal
status check, retain uncertain roots on failure, and reject unsupported
unretained-reference semantics before mutation. Existing row/layer fences and
ON24 limits remain. It requires raw active target/MTP/carry parity, real AUTO
and tool/prefix validation, cancel/terminal/failure controls, and matched
end-to-end wins before any promotion. A healthy final anchor alone is never
proof that an earlier command buffer succeeded.

Raw experiment receipts and independent audits are retained on ADR-063 S3
issue #363. No performance, physical-admission, full-F2 or S3 acceptance follows
from this record.
