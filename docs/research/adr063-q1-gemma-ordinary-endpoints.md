# Q1 Gemma ordinary full-row endpoint adapter (prospective)

This test-only slice supports T0/T1/T3/T4/T7/T8 with four distinct logical
agents in their real independent slots. It does not implement T2/T5/T6. Their
pre-load refusal is LocalizationIncomplete, not a claim that Gemma lacks those
capabilities. Fresh-process/repeat/resource supervision remains external.

The model/header/config/acquisition route is explicitly Gemma4, context262144,
output8192, four slots and Hybrid F16-K/U8-V with actual per-position norms.
Other Dense/Nibble/xlen/F16-V profiles fail rather than silently omitting a
live leg. The inactive HB scaffold is allocated by the production provisioner
but not read by the selected Hybrid mount. Its allocation remains part of the
external operating budget; it is not declared active KV state.

The ordinary batched prefill body retains all final hidden rows, but serving
normally executes the head only on its last row. An explicit diagnostic TLS
scope expands that SAME head loop to each actual hidden row. Every original
scalar head executes final norm, artifact head, softcap, argmax and checked
finish. The adapter captures logits after that finish. It preserves the
returned last-row token and refuses a multi-sequence head override: temporal
rows are not four physical agents. Scalar decode captures after its production
slot-aware return. No new body implementation, CPU head, or last-row surrogate
is used. Extra diagnostic head work changes cost and is not serving timing.

Each of four logical lanes captures all144 logits and complete active Hybrid
K, packed V and norm state at128/144, separately for every actual layer/head.
The config-derived expected manifest is independent of buffer bytes; runtime
checks exact slot/head/capacity axes, dtype, ring/global type and cursor. The
selected window1024 is larger than all144 rows, so all valid ring rows are
exactly [0,position); a smaller/wrapped configuration is refused. Packed U8 is
floating KV representation, not expert-ID discrete semantics. Native submission
IDs remain absent; a checked production return is not invented F2 membership.

T1/T3 first execute the real64 prefill and capture active64. T3 resumes using
the real suffix-aware slot API. T1 additionally calls the existing Hybrid slot
anchor snapshot and restore64, captures the restored full active64 separately
(segment+1000), and resumes through that SAME API. Sliding ring bytes are copied;
full-attention prefix rows remain append-only and cursors restore64. This is an
idle in-place production anchor resume, not a new-cache load or cancellation
rollback test. Exact before/restored comparison is still required externally;
this adapter does not proclaim equality. T0's All128 has no invented64 state.
T4 reverses independent slot order; T7/T8 are separate root-supervised cold
processes with the same frozen workload, not warm repeats in this test.

For the selected30-layer metadata (25 sliding Hkv8 D256, five global Hkv2 D512),
logits request 603,979,776 bytes for576 rows. Active cache requests are computed
before model/device loading, with an explicit positive packet quota at most
2GiB. T1 anchor requested payload is158,105,600 bytes, live for one lane at a
time; its allocator, model/cache/driver memory, temporary rows, host metadata,
and other owned I/O must additionally be bound by root's operating packet.
No bound here is physical admission or cache credit. Unknown native allocation,
margin, intermediate-stage and task-quality evidence remains separate.

Hosted controls exercise the actual pre-load axis guard, config-derived full
cache/restore coordinate plan, no-wrap refusal and actual TLS head scope
(multi-sequence/wrong-source/omitted-row negatives), without a device. Root must
compile and run those, review an immutable real-model packet and capture actual
endpoint receipts before any backend finding. No numerical tolerance is added;
Unspecified remains Inconclusive. This source slice alone does not close Q1.
