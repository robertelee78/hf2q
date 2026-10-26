# Qwen owning-worker pool census — source candidate

Compilation and focused contracts pass against published registry mlx-native
0.15.5, pinned exactly in Cargo.toml and Cargo.lock. No local path patch is used.
Actual owning-worker serving validation remains pending; no model-memory or
performance qualification is claimed. Existing N4 benchmark/client freezes
are untouched.

`HF2Q_WORKER_POOL_CENSUS=1` opts into read-only diagnostic publication. Default OFF,
no unsafe acknowledgment. The centralized investigation env parser caches the
flag; allocation paths have no new telemetry hook, no extra GPU waits/resets,
and ordinary selected/legacy allocator policy is unchanged. Capture is once at
successful SlotAware Qwen request completion, not every decode token or HTTP poll.

The actual supervised model worker is registered by its existing supervisor
wrapper. Only that worker may read its TLS: default legacy DECODE_POOL or selected
GrowthLimited facade. Selected snapshot errors bypass checked/quarantine and
become unavailable diagnostics; they do not fail/replay/reset an inference.
HTTP measurement never reads native pools or TLS, starts an observer worker, or
waits for GPU. It clones the immutable last numeric record from the exact Engine's
shared supervisor. Another engine starts with an empty record and worker identity.

The two completion hooks follow Qwen SlotAware successful synchronous prefill/
decode/verifier paths returning CPU logits or predicted token. Qwen native output
head paths perform terminal commit_and_wait before CPU readback; finish(state)
only assembles CPU output/grammar. Seed completion follows completed prefill;
decode completion may consume an already verified pending token, whose verifier
has previously drained. No additional synchronization or NoFutureUse assertion
is introduced: this observes pool owners after the known synchronous work, not
global GPU quiescence or absence of retained future command/view owners. Cancelled,
failed, native-error and unhealthy cases never mint a fresh successful record.
Response-cache equality paths without a new state/forward leave the prior record
as last-known with its age; they do not invent a new drain.

Runtime measurement adds `worker_pool_census` under its existing per-loaded-engine
snapshot. Fields include status, last_observation, monotonic sequence, capture age,
worker thread, completed slot/generation, phase, per-pool name/mode, retained union,
free/issued/residency_hint labels and overlap. Sixty seconds is only a diagnostic
staleness label; it does not alter admission/scheduling. Cancellation/failure may
retain old numeric data explicitly as last observation, never current.

Native counts complete Metal object buffer.length capacities, deduplicated within
one pool. It does not measure view extents or physical pages. Labels overlap and
must NOT be added to retained_union. No cross-pool sum is published. Physical
residency, escaped/external owners and lifetime high-water are Unavailable, never
zero. KV/TQ/MTP/caches/anchors, GPU cache rings/hidden parents, IdMm, weight pool,
direct native allocations, other TLS/families and driver/allocator metadata are
explicitly excluded. Gemma/DeepSeek return unsupported for this first slice; no
fabricated whole-family or whole-process memory total follows.

Five focused pure controls exercise the actual record/projection seam: immutable
publication/sequence, stale+cancel/health unknowns, overflow, wrong-thread/new-engine
isolation/disabled state, and overlapping native labels without arithmetic sum.
At author handoff they were authored but unrun. Root's later tiny actual native/control packet should
also verify owning-worker selected/legacy capture and a different reader thread,
with no changes to pool counters/owners/GPU synchronization. No model needed for
that mechanism proof. Actual serving observation still needs independent source/
run evidence and does not establish Q5 admission, physical fit or any speed win.

Root validation now uses published registry mlx-native0.15.5 at e772b31a5baa09d07b5b8f110b75c1477f03cc25. The public645-member archive checksum is be6b17fcc030f03615f66873ba22e59aa29b5ae9ff30d6370cbdedf8649245e9;343 production/build source files match the release commit. No local Cargo patch is used. Locked focused census tests (5), supervisor tests (13), measurement tests (3), environment tests (43), and all-target/all-feature check passed. The first registry attempt encountered index propagation lag; the later normal registry resolution succeeded. A mutex temporary lifetime was fixed before the passing run. Actual owning-worker serving observation remains pending.
