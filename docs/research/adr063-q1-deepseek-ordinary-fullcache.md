# Q1 DeepSeek ordinary full-active-cache endpoints (prospective)

This test-only slice extends the existing actual full144 target logits/HC-row
producer with complete native DeepSeek cache state at actual64 (T1/T3),128/144,
and a separately tagged restored64 for T1. Four distinct logical lanes own four
separate caches at context262144; output allowance8192 is explicit in the
acquisition recipe. This producer rejects T5 and owning T6 before model load.
Legacy deliberately rejected native commit-gate work remains a distinct source
diagnostic, not a real client cancellation or scheduler-reservation proof.

The expected manifest is built from actual bound Deepseek4Config and
Deepseek4CachePlan before model/device loading, including all endpoint and cache
bytes. Native requested four-cache bytes are reported separately; they are not
observed physical residency or memory credit. A positive <=2GiB host capture
limit must contain the entire config-derived payload. Root must additionally
bound weights, all four full-context arenas, scratch/queued work, snapshot,
metadata/owned I/O, observation and driver retirement in its operating packet.
No selected context/output/task is reduced to manufacture fit.

Cache roles are source-specific: BF16 physical circular window rows
min(position,128), BF16 completed compressed rows floor(position/ratio) for
ratios4/128, and BF16 indexer rows floor(position/4) only for ratio4. Ratios0/4/128
come from the actual per-layer config. Main recurrent KV/score F32 buffers have
[1,c*ratio,c*head_dim], c=2 for ratio4 else1; ratio4 also has indexer KV/score
state with index_head_dim. Ratio0 has none of these extra roles. Every optional
role's presence, full capacity shape and dtype is checked against the native
plan before active views are captured. Unknown/grafted/poisoned cache is refused.

Window and compressed buffers are views of a common attention parent. The
adapter captures the active views once, not the parent/capacity tail again.
At64, ratio128 has no completed compressed row; at128/144 it has one. Ratio4 has
16/32/36 compressed and indexer rows. Position144 wraps the128-row window: the
capture is explicitly physical ring order with absolute boundary metadata,
not a silently reordered chronological row array. Complete recurrent state is
captured, including source-initialized score -infinity sentinels. The comparator's
existing source-specific exact-score sentinel rule remains; NaN or other
nonfinite arithmetic is not newly accepted or smoothed.

## T1 same backing owner correction

The old generic diagnostic allocated a NEW cache then restored its compact
snapshot. DeepSeek snapshots intentionally omit append-only compressed/indexer
rows; a new owner would lack those valid prefix rows. No actual endpoint model
result is rescored. The corrected T1 uses the SAME cache backing, real snapshot
and native restore64, and explicitly claims no new-cache transfer. Window and
recurrent state are copied/restored; append-only rows survive in their original
owner. Native migration requires strictly larger capacity and is not substituted
as a same-C transfer. A typed Restored64 observation occurs after actual restore,
and complete before/restored64 fields are retained independently from forward
observations. The report is written before exact64 mismatch is returned as a
failure. Every pair must match coordinate, scalar, shape, bytes and full SHA.

T2 retains the existing cooperative row-packed FFN prefill and independent-row
decode graphs. Only those actual operator roles are described; a numeric cohort
width is not a whole-body/F2 physical witness. T4 reverses lane order; T7/T8
remain separately supervised fresh cold processes. No source math/kernel,
quantization policy, attention mask, route threshold or new tolerance changes.

## Genuine T6 still separate

Selected production recovery_position is prompt length minus8. A fresh128 prompt
has native anchor120, not64. A real completed prior72 request can promote64;
a later128 request's new120 is a pending candidate, not committed until request
success. Cancellation clears pending before restoring the committed pre-request
anchor, so it need not replace64 with120. But actual prior completion/generation
may also advance live cache/ledger to73, which affects same-prefix selection.
The real producer, exact input/canonical lineage, actual cancelled call axis,
reservation and pause semantics must therefore be proven before implementing
that explicitly different T6 variant. No hand-installed anchor or cursor is used
here. Missing capability stays Missing, not Unsupported. T5 likewise remains
artifact-specific missing proposal evidence; native routing-domain witness is a
separate published-dependency task.

Hosted controls cover actual config-derived ratio/state/wrapped-window plans,
zero-row and unsupported-ratio refusals, exact64 hash completeness and the
producer's pre-load axis/payload guards. They instantiate no model/device.
Root compilation, controls and independently guarded real-model/trace artifact
review remain required. Unspecified numerical policy and whole-Q1/task/physical
qualification remain explicitly unresolved.
