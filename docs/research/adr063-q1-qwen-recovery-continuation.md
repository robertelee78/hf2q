# Q1 Qwen recovery continuation diagnostic

This test-only adapter extends the actual owning-engine cancellation check. It
is not whole-Q1 acceptance, task quality, serving performance or model admission.
It compares **all80 endpoint rows64..143**, plus complete active cache states at
128/144. The engine's initial64-token path computes the Last head; it therefore
does not claim full144-row logit coverage.

Two separately supervised fresh processes use the same source/binary/artifact,
fixed four144-token vectors, capacity262144, TQ8, output allowance8192, and
explicit ordinary/speculation-OFF diagnostic policy. Serving defaults are
unchanged. The old64-boundary helper/filter remains available with unchanged
body; its default AUTO/MTP zero-manifest limitation still requires explicit
reviewed OFF before actual use.

- `ColdBaseline`: fresh serial physical slot0 per logical lane, target All128
  then16 single-token calls; record rows64..143 and active states128/144.
- `CancelledThenRecovered`: the real engine advances64, establishes its real
  stable checkpoint, advances another64, closes the actual response receiver
  after checked advance, and executes production recovery to64. Record the
  full active before/recovered64 states, then target All64 plus16 single calls.
  The before/recovered pair must compare Exact. These physical target calls
  process208 input rows per lane, including64 cancelled/unpublished rows;
  cold processes144. This overhead is disclosed, not a speed comparison.

Before each independent lane, source reset must leave BOTH conv/recurrent
halves semantic zero, checked directly over their complete logical extents.
There is no fake four-lane target graph: four logical lanes execute serially in
one physical slot. Unknown/error paths stop rather than reset/retry after an
uncertain GPU result. No inactive KV/MTP tail is captured; ordinary MTP cursor
must remain0 at required cache boundaries.

| Selected header | Tail records/payload | Each64 snapshot records/payload | Recovered total disk tensor payload |
|---|---:|---:|---:|
| Q38 |3136 /1609203712B|1408 /636092416B|2881388544B|
| Q36 |1440 /856074240B|560 /266117120B|1388308480B|

These come from the actual consumed cache allocation plan and all selected
logical elements. Per-store ceilings are2GiB tail and1GiB per snapshot; total
payload is bounded by4GiB, with separate128MiB serialized metadata per store.
Disk payload is not CPU/Metal admission. A cold All128 return can hold
127139840 host bytes plus its native GPU output, separately from cache/model
owners. Capture borrows cache views; comparison uses bounded paired pages.
Owned I/O/file cache and source model/snapshot/OLD+NEW phases require the
external execution packet. No general physical bound follows from these caps.

Root-controlled list/commands after independent source/resource review:

```sh
cargo test --locked --bin hf2q q1_ -- --list
cargo test --locked --bin hf2q q1_continuation_scope_covers_exact80_rows_without_first64_claim
cargo test --locked --bin hf2q q1_continuation_actual_plan_closes_disk_quota_without_inactive_mtp
cargo test --locked --bin hf2q q1_continuation_call_axis_cannot_relabel_cold_as_recovered
HF2Q_Q1_QWEN_CONTINUATION_PACKET=PRIVATE_PACKET TEST_BINARY --exact serve::api::engine::q1_qwen_cancel::continuation::q1_qwen_actual_recovery_continuation_trace --ignored --nocapture
```

The actual command requires the root's external process-group supervisor and
hash-bound acquisition/resource approval, not an unsupervised model launch.
Packet mode is exactly `ColdBaseline` or `CancelledThenRecovered`; the remaining
closed fields bind acquisition composition/review, model/output, immutable
identity, fixed Workload, configured_context and finite snapshot/tail limits.
Each process writes `continuation/trace.json`; recovery additionally writes
`before/trace.json` and `recovered/trace.json`. Require complete explicit marker,
all call/cursor/record counts and owned cleanup. Compare the two continuation
traces under a frozen source-qualified policy; absent policy produces explicit
numeric inconclusive, not learned tolerance or a false model pass.

Source: `src/serve/api/engine/q1_qwen_cancel/continuation.rs` and
`src/quality/adr063_invariance/qwen_continuation.rs`. This note records authored
scope only; no actual continuation/model result has been obtained.
