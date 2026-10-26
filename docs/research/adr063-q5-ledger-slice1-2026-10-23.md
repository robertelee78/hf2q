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
bytes, charged once on the host axis. The current ADR063 dedicated-host policy
replaces historical12GiB reserve/116GiB DeepSeek ceiling. The updated declared
packet requires positive observation-age and in-flight-overshoot byte allowances,
positive maximum observation-age/stop-latency times and nonempty evidence
provenance. These numbers are caller claims, not calibrated observer evidence.
Owned+competition+reaction must fit physical RAM. Competition is a nonnegative
caller-declared aggregate; no cached-files-credit/refund field exists. Explicit
absence of a source-justified family ceiling differs from Unknown; no family
ceiling is invented from its name. MetalPolicy distinguishes explicitly selected
Enforced policy from RecommendationOnly (recorded, not enforced as hardware cap),
NoneWithInventory and Unknown. Unknown/missing provenance rejects.
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
Updated limit wire shape uses receipt schema `hf2q.memory-projection.v2`; old
`reserve`/`working_set` fields are rejected, not silently reinterpreted.
The trace peak is checked against aggregate declared capacity; there is no
per-category root matching or external attestation. A caller can invent complete
Bound claims and obtain a declared toy projection. That never proves the claims,
metric coverage, actual ownership, or safety of a real configuration. Default
real-family packets remain Unknown and reject until later trusted adapters and
source-backed bounds supply evidence.

## Validation and remaining work

Historical first-slice `cargo test --locked --bin hf2q serve::memory_ledger` passed 19 model-free
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

## Available-RAM declared-policy followup (2026-10-26)

Source candidate only; author has not built/run tests. Updated pure controls cover
positive reaction/provenance, checked competition underflow/overflow, optional
DeepSeek family inventory, explicit Metal recommendation versus selected policy,
and rejected generic cached-file refund. Root must run the focused filter and
regressions before acceptance. Old19-pass receipt remains historical. No runtime
adapter, OS trust, freshness verification, global allocator/native enforcement or
watchdog is added. A future trusted adapter must bind actual observation sources,
coverage and age, observed stop latency/allocation overshoot, matching operating
policy and family inventory before production admission; this module cannot
certify those inputs or authorize a model load.
