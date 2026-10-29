# Connected DeepSeek owner collector — implementation checkpoints

Research implementation only; ADR-063 Acceptance remains unchanged. Existing
FullCoalescedNoCopy requested envelope 120,783,134,084B plus positive operating
cells 14,092,861,440B = 134,875,995,524B (125.613059405GiB) stays rejected.
Collector metadata is additional, not deducted from an old positive cell.

Checkpoint A connects an explicit unsafe-acknowledged selector to the actual
DeepSeek loader, before Device/cache/weight construction. The selector is
validated at serve entry before phase observers and again at direct family
load entry. None invokes the existing constructor unchanged. Diagnostic header
parsing uses one readonly FD and its duplicate; exact FD metadata is compared
before/after parsing. No second weight mapping or observation-only model load.

The preload schema is deliberately narrower than the final selected-case HLD.
Planned and Unknown cells are descriptions only. There is no deserializable
permit, ActualBound assertion or positive admission implementation. Later
phase registries are unavailable and reject rather than reuse preload cells.
Caller-provided source/binary/policy hashes have syntax checked but remain
unverified; they cannot authorize a load. Native page observation remains
UnavailableUntilPublishedPin (future published native 0.15.14), never private
Mach code in HF or an ignored Cargo patch.

Pending checkpoints B–D: real residency output callbacks; actual owning-worker
ID/generation and bootstrap plus four sessions; all local/TLS/pending checked
Results; private-pool source domain partition or full conservative rejection;
tiny actual-constructor controls and bounded reaction proof. None is implemented
or accepted by this checkpoint. No largest-case, physical credit, memory fit,
serving correctness or whole Q5 qualification follows from source tests.

The separate delayed-weight-window proposal retains full requested roots,
active/next/retiring worst case, inactive PRESENT/uncertain full charge and the
global OS operating ledger. WIRED0 alone grants no credit; no elapsed-time
retirement guarantee or CachedFiles attribution is introduced here.
