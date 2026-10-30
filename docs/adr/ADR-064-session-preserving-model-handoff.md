# ADR-064: Session-preserving model handoff without repeated historical prefill

- **Status:** Proposed; research and ADR review authorized, production design and implementation not yet approved.
- **Date:** 2026-10-29
- **Decider:** Robert E. Lee (product owner)
- **Stable source ID:** ADR-064
- **Related:** ADR-005, ADR-008, ADR-013, ADR-017, ADR-027, ADR-040, ADR-042, ADR-050, ADR-059, ADR-062, ADR-063 C1.
- **Evidence:** [Research and bounded spikes](../research/model-handoff-feasibility-2026-10-29.md), hf2q baseline `7be222a5711dd9750b964e5555cdcfbece7dd2a6`.

## Context and product contract

The operator wants to continue one coding session in Qwen3.8, switch to
DeepSeek-V4 for part of the work, and switch back. Each incoming model receives
the complete chronological conversation, including work added by other models.
The main benefit is avoiding the minutes spent repeating historical prefill.
The complete switch target is under 30 seconds; under 10 seconds is the stretch
target. Both hf2q chat and external clients, initially OMP and OpenCode, matter.
Translators are an hf2q-managed capability, ready to use for qualified model
pairs, not files users must find, train or configure. Unsupported combinations
automatically replay authoritative context.

The session transcript MUST remain complete. The user has not approved a
quality discount for speed. A learned cache is approximate even when its source
text is retained; behavioral equivalence to native receiver prefill cannot be
promised by a tensor conversion or inferred from averaged benchmarks. The
qualification below determines whether a pair can be offered as a fast switch.
No protocol can detect every semantically degraded response before it happens.

**The core deliverable is first-visit and repeated model switching without
repeating historical prefill, not paper reproduction or checkpoint plumbing.**
Exact-return engineering is enabling work and a useful control, not a substitute
for this feature. A partial checkpoint-only result MUST be named as partial.

This ADR proposes a program and a candidate architecture. It does not claim a
trained translator exists, the latency target is met, or all research gates
will pass. No production serving behavior changes with this document.

## Decision

### Research basis and freedom to improve

The initiating research is [Swarm Telepathy: Prefill-Free KV-Cache Transfer
Across Families (SwaT), arXiv:2610.15095v1](https://arxiv.org/html/2610.15095v1).
Its learned, directional cache mapping between frozen models motivates this
proposal. [LatentPort](https://arxiv.org/html/2609.25053v1),
[CacheLink](https://arxiv.org/html/2610.15509v1) and
[CacheBridge](https://arxiv.org/html/2609.00891v1) inform the full-state and
behavioral validation candidates. The linked research report records their
boundaries and inspected implementations.

The operator explicitly permits improvements beyond the paper and adaptations
to hf2q. We MUST choose methods from measured correctness, quality, latency and
resource results, rather than reproduce SwaT's architecture as an end in itself.
Native checkpoint reuse, alternative transport heads, calibration, functional
training and architecture-specific execution remain candidates. No paper or
vendor claim waives the product's context-preservation requirement.

### D1. Preserve an authoritative session independent of model state

hf2q MUST reconcile a canonical chronological message ledger, preserving roles,
exact visible text/code, reasoning visibility as actually available, tool IDs,
tool names/arguments/results, content parts and tool definitions. Opaque or
provider-private reasoning not supplied to hf2q cannot be reconstructed; claim
coverage only over the context actually received. External client compaction,
history edits, tool-set changes and omitted content MUST be detected against
the actual request and handled by compatible recomputation or replay. Do not
silently synthesize missing history from a translated cache.

Let `R_m(L, C)` be model m's native rendering/tokenization of the actual
reconciled ledger L and settings C. The transfer target is the main inference
state produced by native prefill of `R_receiver(L, C_receiver)`, followed by the
same first-token boundary as its full-replay control. Transcript completeness
means retaining received ledger content; receiver-visible completeness means
matching that receiver's native rendering policy. Do not force non-native
reasoning retention merely to make prefixes match. A pair manifest MUST list
supported rendering divergences (role delimiters, token boundaries, prescribed
reasoning removal and tool-ID remapping). Arbitrary content edits, system/tool
changes, missing parts and unqualified divergences require a compatible earlier
native anchor or full replay. Editing source content near the start can
invalidate recurrent state throughout; it is not merely a RoPE adjustment.

Reuse eligibility MUST derive from verified content/prefix identity within a
principal-scoped namespace and complete D3 compatibility, never a token count.
Opaque session/branch IDs and monotonic revision hints improve intent, forks,
idempotency and telemetry but are not required for content-addressed native
reuse. A session ID alone never grants access to another principal's state.
Every turn MUST reconcile actual content; a deliberate fork gets distinct
lineage and the default switch continues the same chronology. Avoid interpreting
OpenAI `user` or an untrusted `prompt_cache_key` as authorization.

An optional hf2q session/control extension MAY carry IDs and revisions while
ordinary OpenAI unary/SSE requests remain valid. Legacy clients can use exact
native prefix restore across eviction without a custom extension. Translation
for those clients additionally requires a verified match between the actual
receiver request and a retained source ledger in the same principal scope;
ambiguity or missing history causes replay. No client modification may be
assumed without a versioned wire-capture proof.
Reconciliation MUST separately enumerate permitted client-side ledger rewrites
(for example bijective tool-ID changes) and preserve raw-request digests plus
the checked mapping. Receiver rendering has a different divergence manifest;
do not confuse a client rewrite with template formatting or silently delete
content. Unrecognized or ambiguous rewrites replay. H0 measures both boundaries.
The extension's exact wire schema is an H2 design deliverable, not frozen here.

### D2. Prefer independently native state; separate three execution paths

The planner MUST choose and expose one of:

1. **Native checkpoint plus suffix:** restore a compatible complete checkpoint
   previously computed by the requested model, then process its exact rendered
   suffix. This avoids repeating the old prefix but is not zero-prefill.
2. **Qualified full-state translation:** translate another model's state for a
   named direction and supported configuration; process any uncovered suffix
   and first-token boundary. No hidden replay may be called zero-prefill.
3. **Full replay:** recompute from the complete authoritative request when no
   compatible qualified state path exists. Report the reason and actual cost.

Returning to Qwen SHOULD use Qwen's independently retained native checkpoint
when its rendered prefix remains compatible. Place checkpoints before volatile
template regions and retain multiple compatible anchors through ADR-063 C1.
Measure the surviving rendered-prefix fraction instead of assuming stability.
Do not reverse-map an approximate DeepSeek state by default.
A full replay can establish a new exact checkpoint;
ordinary continuation from translated state cannot. Provenance MUST distinguish
native exact, translated, and derived-from-translated state through persistence,
branching, eviction, and restore. Translated lineage MUST NOT enter an exact
cache namespace or be relabeled exact after several ordinary tokens.
For initial product qualification translation inputs MUST be native-exact, with
approximation depth at most one. A derived-from-translated checkpoint can be
continued/restored only within a separately qualified same-model arm; it cannot
seed another translation. If no native source is available, prefer a compatible
native receiver anchor or replay. A resident model MAY rebuild exact state from
the ledger during explicitly budgeted idle work; charge that work separately
and never hide it in a zero-prefill claim. Native-state healing does not undo
mistakes in text already generated; later models inherit the real chronological
ledger, including those mistakes.

Use the existing ADR-063 C1 anchor work for exact component validity and
lineage. This ADR does not supersede it or authorize a second competing anchor
implementation. Existing persistence hooks are substrates to qualify, not
evidence the end-to-end handoff already works.

### D3. Bind every cache and translator to complete compatibility identity

Compatibility MUST cover immutable source and receiver artifact identities,
source HF revisions/conversion provenance where available, quantization and
cache codecs, model architecture/version, tokenizer and template bytes,
rendered prefix positions, inference configuration affecting forward state,
vision/projector identity when relevant, graft/steering identity, runtime state
ABI, session/branch lineage and state provenance. Geometry plus quant label is
insufficient. An opaque checksum is only integrity when verified against an
authenticated identity. Do not rehash a 100-GiB model on each switch: use a
verified preparation identity, retained descriptor and mutation invalidation.

Validate all metadata, payload bounds, required components and compatibility
before installation. Partial restore MUST never publish a runnable receiver.
Reset on partial mutation and use authoritative replay. Incompatible context
capacity MUST be reported without silently truncating or summarizing; an
unsupported model combination does not imply a requested context fits.

### D4. Translate full family state, not just attention tensors

For Qwen, the state contract MUST include full-attention K/V, recurrent GDN
matrices, convolution history, initialization/parity flags and logical cursors.
Draft/MTP and speculative state MUST either be reconstructed and qualified or
explicitly reset under a separately proven target-only path; a reset cannot
discard target-prefix information needed by the main graph. Packed cache
dequantization/requantization is part of measured execution and quality.

For DeepSeek, the contract MUST cover its circular window, append-only
compressed rows, indexer rows, compressor accumulators/score states, incomplete
compression groups, RoPE positions and visibility cursors. Its in-memory
recovery snapshot alone does not contain append-only history; eviction export
needs a complete image. Do not substitute a generic MLA or dense K/V loader.

Training candidates MAY combine token-indexed source features, pooled recurrent
state and receiver token embeddings with family-specific output heads. Learn
receiver representations and derive deterministic cursors/metadata rather than
learning arbitrary pointer values. SwaT-like attention and CacheLink-like
functional corrections are hypotheses, not selected winning architectures.
DeepSeek-to-Qwen and Qwen-to-DeepSeek MUST pass separately. Full-context export
of normally discarded intermediate features must be budgeted and charged; it
is not a free retrofit to already processed sessions.

Chat-template differences need a semantic message/content alignment with
explicit special-token and role handling. Raw character offsets in different
rendered prompts are insufficient by themselves. Preserve exact identifiers,
Unicode/code spans and tool pairings. Do not silently treat one model's private
reasoning tokens as another family's native control sequence.

DeepSeek-to-Qwen is expected to be harder than the reverse: the inspected
32K-state estimates are 263 MiB versus 1.67 GiB, with different retention
mechanisms. This is a hypothesis about information/representation asymmetry,
not a proof from byte counts. When receiver tokens are inputs, the mapper is
also an approximate receiver-prefill encoder conditioned on source state.
Require a token-only/no-source-state ablation to establish that source state
helps. A directional failure may reveal limited source information rather
than a broken installer.
If token-only encoding wins the same quality/latency gates, retain it as a
candidate prefill surrogate, not a failed experiment. Charge its full token
processing and call it neither source-state transfer nor zero prefix compute.

### D5. Make handoff an atomic resource-bounded lifecycle transaction

Extend the existing lifecycle rather than bypassing it:

1. Reconcile session/revision; resolve local target and qualified artifact;
   preflight disk, context, RAM and required operations before disturbing the sender.
2. Stop new admissions to exact victim generations, drain in-flight streams and
   GPU work with bounded cancellation, and freeze the acknowledged session revision.
3. Export/flush a complete source image and any returning-model native anchor;
   acknowledge durable writes before releasing the sole useful state.
4. Remove sender workers/weights and verify actual resource release. Metadata
   pool removal is not proof that outstanding Arcs or native buffers are freed.
5. Load/warm receiver and mapper within one admitted budget; validate and install
   pending state, process required suffix, then atomically publish the new generation.
6. On failure retain the canonical ledger and invalidate pending state. If the
   target cannot load after sender removal, attempt a separately admitted sender
   reload and acknowledged-image restore; otherwise leave an explicit recoverable
   no-model state. Record recovery time outside the successful-switch SLO.
   hf2q must not initiate tool reruns; external clients own tool execution and
   require revision/content idempotency to prevent retry-triggered duplicates.
   Exactly one generation may commit the next turn.

Snapshot sizes, transient staging, mapper weights/activations, receiver scratch,
retained anchors, disk/page-cache effects and concurrent slots MUST count in
admission. Never co-load full DeepSeek and another full-model runtime on this
128-GiB host to make a benchmark easier. Holding source state after weights are
unloaded still consumes RAM and requires its own budget. Disk artifacts have
bounded size/retention and user-visible cache management.

For each concrete execution schedule measure
`peak = max_over_stages(weights + retained_source_state + destination_state +
mapper_weights + mapper_activations + scratch + retained_anchors + staging)`.
Count only overlapping lifetimes but include owned and physically retained
backing, not merely logical views. A self-contained mapper executed between
weight unload/load and overlap with bounded receiver page-in are candidate
schedules; receiver embedding-table requirements may constrain them. H0 must
measure/map lifetimes before selecting one. Feature capture overhead is charged
to sender throughput as well as switch time.

SSE remains OpenAI compatible. A blank role event, activation-ready response,
or shader warmup does not satisfy useful resumption. Cancellation must not
publish partial state. Do not silently cancel another user's live generation
for a global model switch; reuse the existing explicit eviction/lifecycle
contract and expose bounded busy/draining outcomes to clients.
An ordinary external `model` change MUST NOT silently evict unrelated active
work. H2 defines authorized on-demand switching for a single-owner endpoint,
explicit confirmation/control for shared endpoints, queued-demand hysteresis,
fairness and bounded busy responses; no switch-on-every-request thrash. Use
protocol-valid keepalives where supported without counting them as semantic
output, and qualify actual client timeouts.

### D6. Own runtime execution and manage compatibility artifacts

Production execution MUST remain Rust/`mlx-native`; no Python, PyTorch, peer or
vendor runtime subprocess performs production conversion or inference.
Developer-only derivation/training MAY use an external reference stack, as a
separate offline workflow with pinned sources and reproducible export. This
does not authorize training inside normal serve or a production fallback.

hf2q owns session/model policy, family state graphs, cache compatibility,
translator orchestration and workload acceptance. mlx-native owns required
Metal primitives, buffer/fence semantics and primitive parity. Reuse published
operations when sufficient; publish any new native revision before pinning and
qualifying downstream. A local path patch never qualifies a shipped result.

Qualified directed translators MUST be in the normal hf2q model experience,
either bundled or automatically resolved through a checksum/signature-verified
release catalog. Pin source/receiver identities, schema/runtime ABI, dtype and
quantization coverage, supported lengths/features, derivation/license evidence,
and qualification receipt. Bound downloads and allocations; support offline
replay, eviction and rollback. Verify an artifact before evicting the sender.
Do not silently download large resources outside existing download policy.
The user need not manually select or train translators.

Treat translator bytes as trusted executable influence over model state:
verify the release signature and all bounds before installation as well as
before sender eviction. Specify signing-key custody, derivation provenance,
revocation and rollback in H7. The canonical transcript cannot expose every
effect of a malicious translator. User-installed artifacts require a verified
one-time immutable identity; mutations invalidate bindings.

Single-user loopback without auth is one explicitly documented owner principal;
do not claim isolation between local processes. LAN or multi-user operation
requires authenticated principal scopes, separate state namespaces and no
cross-principal prefix/translation sharing. Session hints confer no authority.
State files MUST use owner-only permissions, atomic writes and bounded retention;
cache deletion removes namespace access without promising physical secure erase.
The filesystem and retained-file mutation checks are part of the trust boundary.

## Research and acceptance sequence

All items below are proposed, not Ready. Stable work keys use
`robertelee78/hf2q:ADR-064:<ID>`. One AWA-managed, repository-linked GitHub
Project tracks this ADR and its H0–H8 stories against immutable source bindings.
Tracker setup does not approve production execution. No release version is assigned.

### H0. Establish the core feature's evidence-backed execution entry

Outcome: A measured go, bounded redesign or stop decision before costly training.

H0.0 first establishes an admissible DeepSeek host configuration: inventory a
coordinated quiescent baseline and actual Metal residency/working-set policy;
consider smaller qualified context or a separately identified artifact if
necessary. Do not kill unrelated work, require a reboot, silently change OS
limits, or substitute a model. If no schedule fits, stop/redirect the hardware
or explicitly reviewed target envelope before full-model work.

Acceptance: Measure native Qwen/DeepSeek full-replay switch stages at
16K/32K/64K and the operator's long coding workload, cold and warm filesystem
conditions, with exact artifacts and actual memory headroom. Verify DeepSeek
fits with the proposed state/mapper schedule before loading; current static
preflight lacks headroom and is a refusal to load now, not a permanent host
capacity verdict. Measure how much of today's minutes is prefill versus loading.
If loading alone exhausts the 30-second target or the schedule cannot fit,
redirect the implementation/envelope explicitly rather than claim mapper wins.
Predeclare a screen before measuring: p95 load-plus-reclaim above 20 seconds
on the declared installed-artifact condition leaves under ten seconds for all
other stages and requires a scheduling/loading redesign before costly training.
This engineering screen is not the final product SLO or proof of feasibility.

Run offline source-template prefix-stability probes and capture actual OMP /
OpenCode A-B-A requests; distinguish source checks from real client evidence.
Measure surviving native prefix fractions and feature-capture bytes/throughput.
A bounded random-weight mapper Metal timing/memory spike must precede accepting
a full-context transport schedule; random weights establish cost, not quality.
No full-model load until resource coordination and admission pass.

Before H4 data generation, write a rights determination for model weights,
activations/outputs, derivation/redistribution and training corpora; user sessions
are excluded absent explicit data-use authorization. Independently implement
unlicensed upstream methods unless permission is obtained. Freeze paired-state
generation and gradient-through-receiver compute/memory estimates, available
hardware/access, cost ceiling and a bit-faithful quant/compressor reference
strategy. Lack of an affordable available path redirects the method; it must
not turn the core feature into a paper-port project. Review H0 independently
before starting H4/H5. Cheap independent H1/H2 probes may proceed.

### H1. Freeze baseline and prove complete same-model capture/restore

Outcome: Reproducible native restore controls and complete state manifests.

Acceptance: Pin exact Qwen3.8 and DeepSeek artifacts, binary/source/native
versions, prompts, templates, settings, host and model-load conditions. Prove
same-model capture, eviction, reload and continuation on both families against
matched native control schedules, including partial compressor boundaries,
Qwen parity, tool rounds, packed codecs, corruption and cancellation. Treat
unfinished GPU/draft work explicitly. Record cold/full replay and warm return
timelines, semantic TTFT, cache/suffix counts and peak ownership. Reuse ADR-063
C1 proof and implementation where applicable. Bind images to the exact weight
artifact even where the old namespace hashes only geometry/quantization.
H0 admission precedes full-model runs. No cross-model quality claim.

### H2. Specify session protocol and preserve client history

Outcome: One revisioned chronological session survives model selection.

Acceptance: Remove hf2q chat's reset-on-model-change behavior through a scoped
reviewed implementation; preserve explicit `/new`. Freeze versioned wire
fixtures for hf2q chat, OMP and OpenCode with tools, results, reasoning,
system/tool changes, Unicode, branching, compaction and retry. Define discovery,
session/auth binding, activation, telemetry and legacy behavior. Test actual
client requests and prefix rendering. Never equate TUI history with sent context.
Acceptance also covers implicit-switch authorization, queue hysteresis/fairness,
busy outcomes, idempotency, keepalives and real client timeout behavior from D5.

### H3. Qualify exact returning-model resume

Outcome: Native checkpoint plus only compatible rendered suffix after A-B-A.

Acceptance: Independently native checkpoints remain available across eviction;
ledger edits invalidate descendants. Compare restore-plus-suffix to native
full replay with exact numerical controls and task oracles. Include deep
recovery images, disk faults, restart, two sessions sharing prefixes and actual
resource release. Measure template-aware checkpoint survival and reused-prefix
fraction at every return. Report replay costs honestly. Predecessors: H1 and
accepted ADR-063 C1 multi-anchor validity/lineage substrate. Native
content-addressed external-client restore does not wait for a custom H2 session
extension; hf2q chat history preservation and end-to-end client qualification do.

### H4. Establish a reproducible translator derivation pipeline

Outcome: Offline paired full-state data, functional training and export.

Acceptance: Start with small/random-weight hybrid and DeepSeek-V4-architecture
controls exercising compressor/indexer and partial groups before the full pair.
Separate clean native restores, random/zero, wrong-donor, no-context,
token-only/no-source mapper, KV-only, full-state, native receiver and replay arms. Pin
data licenses, deduplicate splits and hold out code/tool tasks and repository
identities. Use reconstruction plus receiver-behavior objectives, long suffix
rollouts and quantized-native validation. Freeze training hardware/capacity,
cost ceiling and stop rules before allocation; the small-paper GPU-day budget
does not estimate DeepSeek training. Verify export checksums, loader bounds and
framework-to-native operation parity. Missing weights/code/rights are explicit
prerequisites, not permission to claim a reproduction. Predecessors: H1 and
independently reviewed H0 rights, compute/access, cost and parity entry gates.

### H5. Decide each first-visit full-state transfer direction

Outcome: Evidence-backed implement, falsify or successor-design per direction.

Acceptance: Compare candidate families at 4K/16K/32K/64K and the actual long
session workload within native capacity, including cross-template boundaries
and first decoder-token semantics. Freeze scoring/sampling before held-out
evaluation. Test long tool outputs, exact code/paths/numbers, delayed constraints,
instruction hierarchy and independent task tests. Tensor shape/MSE, valid JSON,
or receiver-vs-sender gains cannot substitute for native receiver quality.
Cost must include extraction, alignment, codec/staging and all mapped components.
Predecessor: H4. No automatic experimental enablement from this gate.

### H6. Prove round trips and define the behavioral shipping bar

Outcome: No demonstrated context/tool regressions across realistic switching.

Acceptance: Run A-B-A, B-A-B, at least ten alternating handoffs, separate
exact-return and translated-return arms, and new-model first visits. Include
at least six tool-result rounds per coding task and multiple independent
sessions. Use greedy/fixed-seed matched controls for the exact-detail/tool
critical suite, requiring zero new failures relative to native replay; a native
failure is not a transport success. Use paired repeated sampling for task
outcomes and score independently of wording. Predeclare a
statistical noninferiority margin with the product owner before final quality
qualification; no quality discount is granted here. Report confidence intervals
and per-task regressions; underpowered results are inconclusive. A successful
qualification is not a universal no-information-loss proof or a per-request
semantic failure detector. Predecessors: H3/H5.
Predeclare native repeat/chunk/restore perturbations, measure native-vs-native
flip rates, and identify the stable critical items before candidate scoring.
Keep unstable items in reported task results; never exclude them post hoc to
manufacture success. Match native controls' deployed template settings.

### H7. Measure complete-switch UX and package qualified artifacts

Outcome: Qualified pairs work through all three clients with hf2q-managed assets.

Acceptance: Measure switch intent to first useful receiver output in one
monotonic end-to-end clock, including drain, export, flush, unload/reclaim,
target load/page faults, mapper load, mapping, install, suffix and first decode.
For client-instrumented tests start at the user's model-selection intent and end
at the first client-observed semantic content, reasoning or meaningful tool-call
delta. Separately report server request-arrival timings; never substitute those
for user-intent timings. Drain time counts. An invalid tool delta fails quality.
Report per-stage and total p50/p95/max from at least twenty balanced paired runs
per direction and workload; totals are measured per run, never sums of medians.
Separate first-use downloads, cold/warm filesystem cache and already resident
targets. Predeclare context and intervening-suffix length cells and weighting,
including A-B-A total and Qwen's added suffix cost. Report sample counts and
uncertainty; twenty observations give a preliminary p95, not a tail guarantee.
The initial target is p95 below 30 seconds for installed-artifact, quiescent
single-owner switches with explicitly labeled page-cache conditions; ten is
stretch. Cold activation remains a required reported stratum and cannot inherit
a warm-only claim. Count physical retained pages and pressure in residency,
including OS cache; a warm-cache speedup must not hide overlapping runtime weights.
Publish actual achieved envelope and hardware. Do not call a warm
tiny-model mapper benchmark the user's switch time. Verify signed/checksummed
catalog, update/rollback/offline/corrupt/missing/unsupported behavior and avoid
overlapping full DeepSeek runtimes. Predecessor: H6.

### H8. Review, publish and reconcile the qualified scope

Outcome: Reviewed, reproducible and explicitly bounded release support.

Acceptance: Opus reviews this ADR via `claude -p`; retain version/model, exact
input hashes, findings and dispositions. Review concrete implementation and
native primitive changes independently later. Run focused tests then applicable
locked regression/hardware checks, update governing ADRs and shipping contract,
publish required native revisions and translator bytes before hf2q pins them,
and verify clean-source builds and public package identities. Commit/push/merge
and release need operator authorization; do not interpret research permission
as release permission. Reconcile existing work plus new source-bound issues
without duplicating ADR-063 anchors. Predecessor: H7 for product release;
ADR review itself can occur now.

## Alternatives and consequences

- Full replay remains the fidelity baseline and universal compatible fallback,
  but does not solve the latency goal. Context overflow cannot use it silently.
- Native checkpoints plus suffix reduce return cost without learned transport,
  but first visits still replay and long intervening work can remain expensive.
- Shadow prefill of every future model duplicates compute and defeats single
  full-model residency; it is not the default proposal.
- Summary or lossy cache compaction is not approved to satisfy the requirement
  that context survive. Evaluate separately only with explicit product scope.
- Universal translators would reduce directed-pair maintenance but have no
  qualifying evidence here. Support a bounded pair matrix first, while keeping
  ordinary replay available for new models.
- Exact artifact binding creates a directed compatibility matrix multiplied by
  quantizations, fine-tunes and revisions. Qualify the two named vanilla artifacts
  first; user variants replay until separately qualified. Artifact matching needs
  a verified one-time digest, then retained identity/mutation validation; shape
  compatibility never implicitly transfers qualification. Maintenance cost and
  catalog retirement are part of the feature budget.
- Even if H5 fails for the target pair, H2/H3 may produce useful session
  preservation and faster native returns. They cannot be advertised as having
  solved prefill-free first-visit switching.

## Current disposition

Bounded source/CPU/specification spikes are recorded in the research report.
Full-model native restore, learned transport training, native client end-to-end runs,
round-trip task qualification and the latency target remain unproven. The
governing next decision is whether the reviewed program has an evidence-backed
execution entry; acceptance of the product ambition is not evidence of a
working transport.

Independent review: [Opus 5.5 findings, dispositions and follow-up verdict](../research/ADR-064-opus-review-2026-10-29.md).
