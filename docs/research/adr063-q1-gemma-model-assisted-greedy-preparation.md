# ADR063 Q1: separate model-assisted Gemma greedy144 preparation

This proposed test-only preparation preserves the earlier922bec workload and
all its diagnostic results. That forced tail is not a greedy continuation:
its input128 is496, while the earlier actual T0 row127 GPU argmax differs in
all four lanes. Nothing in this change rescales, reinterprets, or rescored
those traces.

The new ignored `gemma_greedy_prepare::q1_gemma_model_assisted_greedy144_prepare`
uses the selected bf65 Gemma26B-A4B artifact, ordinary speculation OFF,
Hybrid F16K/U8V, context262144/output allowance8192/four logical slots.
It loads one model and processes four lanes serially. Each exact old prefix128
gets one actual slot-aware prefill and16 actual scalar ordinary forwards.
Inputs are chosen ONLY from the real returned GPU argmax, never a CPU proposer.
The CPU scans each complete finite logits row only to verify that GPU choice.
A maximum shared by multiple IDs, or an EOS ID1/106 among the16 canonical
choices, makes the preparation ineligible. The source also checks the actual
loaded EOS set. No truncation, padding, resampling, retry, or favorable selection.

Exactly68 checked forwards and68 complete F32 logits rows127..143 are retained:
4*17*262144*4 =71,303,168B (68MiB). The dataset is prefix128 plus g0..g15;
g16 from row143 is carry metadata and never a145th model input. Store captures
all rows/page/full hashes; the actual-used chain gate checks every lane,
position, canonical input and final carry. One new proposed workload file
and its canonical LE-vector digest are emitted only after complete coverage.
A separate final review must stream all rows and independently validate every
argmax edge, prefix/hash, carry and opened-file/runtime identity before approving
this ONE new final-matrix workload. There is no T5 accepted/rejected transition
in preparation itself, no ordinary AUTO quality claim, and no numerical tolerance.

Acquisition is `WorkloadPreparationNotComparable`, bound to prefix-only identity,
actual executable/source/native/config/full-content receipt and all three actual
model FD consumptions/device/queue/nonce. Existing comparable-acquisition checks
still refuse that provenance. The existing preparation guard gains Gemma as an
explicit family, preserving old Qwen digests and refusals for unsupported families.
Ordinary endpoint capture is unchanged; only an explicitly installed preparation
scope selects the same ordinary last prefill head, rather than expanding128 heads.

Closed input requires the original workload file/hash, selected artifact identity,
independently reviewed composition, exact EOS/ambiguity policy and root execution
approval. Output is new/private/outside Git, files0600. Payload is exactly68MiB;
per-row CPU copy1MiB, Store and decision hashing have separate transient bounded
row copies (conservative4MiB CPU staging). Trace and preparation JSON each have
1MiB checks, proposed workload≤1MiB; Store's existing128MiB metadata hard writer
ceiling remains a conservative outer bound before those tighter checks. No whole
cache capture is required for this preparation; future matrix trajectories still
capture their full active logical cache roles. All full-context cache/model/CPU,
positive operating margins and prior I/O remain separately charged. This file
quota is not a process or physical-memory guarantee.

The disabled private guard recipe intentionally has no candidate/build/binary,
packet/composition/review/resource-policy bindings or final approval. Root must
bind/review an owned-process supervisor with existing memory/pressure/swap,
finite timeout, prior-I/O accounting, exact identity/cleanup guards, then execute
serially. Source-authored hosted controls exercise the actual selection/chain
validators and endpoint scope, not inference. No tests/build/model execution
has been performed by the author; all validation and dataset approval are pending.
