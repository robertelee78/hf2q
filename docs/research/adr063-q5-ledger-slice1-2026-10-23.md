# ADR-063 Q5: model-free ledger first slice

The operator [approved this first slice](https://github.com/robertelee78/hf2q/issues/371#issuecomment-6376203272).
It implements declared memory projections in `src/serve/memory_ledger/`.
It does not observe allocations or OS metrics, and is not wired into production
serving admission. Full [Q5 qualification](adr063-q5-memory-design-2026-10-23.md)
remains open; this merge cannot make Q5 Done or authorize a model run.

## Implemented contract

The closed inventory requires four slot owners, the separate DeepSeek boot
owner, each declared dispatch thread's pools, cache/logit/token and anchor roles,
growth, projector/vision/graft/drafter, persistence producers, service queues,
captured/queued/completed-retained resources and pending residency removal.
Every required category starts Unknown. Missing, duplicate, unexpected or
unknown entries reject. Disabling a feature does not silently infer zero.
Declared bounds require provenance and a formula revision; declared zero
requires a reason and configuration evidence.

Arithmetic uses checked byte counts. Shared Metal bytes are a subset of owned
bytes, charged once on the host axis. Host ownership plus competition plus at
least 12 GiB reserve must fit physical RAM; family ceilings and Metal working-set
policy apply independently. DeepSeek cannot omit or exceed its existing 116 GiB
ceiling. Explicit absence of an additional family ceiling differs from Unknown.
Artifact, workload and transition identities plus the complete expected shape
must match; a caller cannot pass a smaller shape as the same frozen packet.

The ownership simulator counts unique roots through aliases, views, pool and
command owners. Completion marks a retained owner; it does not release its root.
Only the last owning release removes the charge. Reservation and event batches
check intermediate peaks and publish atomically, so a later release cannot hide
an earlier breach. One transition permit serializes declared overlap. This is
pure `&mut` state simulation, not a production global allocator or GPU callback.

Receipts are serialization-only objects with fixed scope
`caller_declared_model_free_projection_only` and `runtime_qualified=false`.
The trace peak is checked against aggregate declared capacity; there is no
per-category root matching or external attestation. A caller can invent complete
Bound claims and obtain a declared toy projection. That never proves the claims,
metric coverage, actual ownership, or safety of a real configuration. Default
real-family packets remain Unknown and reject until later trusted adapters and
source-backed bounds supply evidence.

## Validation and remaining work

`cargo test --locked --bin hf2q serve::memory_ledger` passes 19 model-free
controls. They cover closed imports, boot and thread owners, anchor/persistence
overlap, unknown and missing evidence, frozen shape/identity changes, arithmetic
overflow and independent caps, retained parents after completion, four contending
reservation attempts, transient breach and atomic failure, and permanently
unqualified receipts. The documented DeepSeek lower bound independently exceeds
the observed recommendation policy in its control. These tests are blocking CI.

Independent q5-review-agent code review found no blocking defect within this
scope. No model loads, native or OS changes, hardware stress or performance tests
were performed for this slice. Native lifetime counters, complete source-bound
scratch/transition functions, calibrated complete competition observations,
external watchdog proof, actual runtime admission wiring and bounded real-model
qualification remain required. The known process-observation gap and DeepSeek
full-capacity policy rejection are not resolved by this library.
