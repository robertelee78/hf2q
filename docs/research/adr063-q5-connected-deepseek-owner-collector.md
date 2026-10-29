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

Checkpoint B1 source wiring (not runtime coverage): actual residency buffers,
family alloc/host-input/persistent helper outputs and SubmissionChain push plus
EVERY existing checked encoder Result now reach the same optional owning-thread
sink. Default None does not call native backing getters. Active descriptors keep
root IDs and requested extents; distinct source-private domains may not share a
root without explicit new proof. No buffer/Metal clone changes lifetimes.
Submission metadata errors poison the sink but never abandon an already committed
encoder; existing all-Results drain finishes before the observation error surfaces.
Completion records remain retained, with no byte or physical credit.

The only production scope currently installable is the header rejection scope;
no runtime permit, worker scope or native page observer is available. B1 alone
therefore provides no actual-runtime census. SAME-FD weight interval joins, actual
worker/session mounting, full private arena visitors, pool union and native early
coverage, CPU envelope, sink serialization and tiny constructors are still pending.
Callback controls are authored only; root owns compilation and execution.

Checkpoint B2 adds private borrowed cache/anchor traversal (all eight/five fields)
and actual seed/bounded-prefill owning scopes. Scheduler lookup validates the
installed complete SlotHandle and obtains its true RequestId, including zero.
Before-mount/after-swap-back snapshots retain loaded/bootstrap plus all four
actual sessions; role observations do not assert allocator-domain membership.
Stale/overlapping callbacks poison; no native buffers are cloned. This is still
NOT complete worker coverage: decode/cohort/cancel/provision callbacks, mounted
snapshots, per-role current-vs-history manifest, same-FD actual tensor joins,
all arena/TLS residuals and final sink remain pending. No runtime permit exists.
The root map is an observed-history cross-check, not a current-live root census.

Checkpoint B3/C-source additions (unexecuted): phase-specific required-key sets
now exist for all six phases, but runtime packets remain categorically refused.
The native page-observer trait has only an explicit Unavailable implementation.
Real-load closure accepts the SAME parsed GgufFile; actual weight callbacks join
its tensor-data offsets and retained FD metadata to actual returned NoCopy spans
or independently owned F32/packed parents. No second map/load is introduced.

Large-case metadata uses constant-space cumulative u64 observations, not per-event
logs. Transient allocation observations are counts/sums, NOT unique creation/live
bytes. Only weight and role checkpoint roots enter the bounded16384-root census;
actual owning begin resets observation-only census and revisits retained caches,
without changing any native lifetime/counter/charged term. At most256 simultaneous
CB records are checked against exact actual chain length and ordered indices,
then replaced by cumulative all-Results counts. Completed Metal ownership remains
unchanged. These capacities need full source certification before runtime permit;
no physical/live census is inferred from observation history.

Metadata allocation failures in a forward are sticky and surface only AFTER the
existing checked boundary; earlier committed commands are not dropped by a new
allocation callback error. Existing verifier/chunked/cohort error closures all
call their own all-Results drain before dropping chains. Residency observations
can fail immediately because their actual map/CPU load path submits no forward
commands. Default None avoids native descriptor calls.

Tiny constructor source reuses the actual pre-existing three-ratio config:
five initialized caches plus eight snapshots,122distinct roots13,412,352B;
122*64KiB rounding plus1MiB metadata gives22,456,320B<32MiB named userdata.
The ignored actual control joins SAME collector, checks alias-parent dedup and
actual native root authority counts; dispatch/commit/sync counters must remain
unchanged. It does not construct a model/forward or selected-case permit.

Still pending B–D: actual device/epoch/native early authority integration;
worker transfer and complete decode/cohort/cancel/provision/mounted callbacks;
actual roles/spans per current checkpoint plus serialized final sink; complete
local-arena/TLS and native residual partition; CPU/source bounds; full error
fanout; selected-case capacity proof and external low-memory reaction control.
No source checkpoint authorizes any model run or whole-Q5/component acceptance.

B4 source adds all three ACTUAL owning TLS pool snapshots at the same worker
checkpoint. Each pool is bounded before native temporary census by<=16384
free+issued entries; source-private alloc/reset/clear-only closure bounds native
residency records by those entries (no register_existing/release). Retained,
free,issued,residency labels overlap and are not summed. Pool values remain
separate from cache/logit caller roots and native-outside-pool residual; this is
not a fabricated global root union or between-checkpoint lifetime peak.
Source allocation census is retained in the private author report; remaining
direct allocations are embedding's three buffers and cache/snapshot helpers,
which require callback joins before the complete closure is frozen.

B5 closes direct family constructor taps: embedding IDs/gathered/HC outputs,
initialized/overwrite cache allocation and both independent anchor-copy allocation
helpers now feed the same non-throwing sticky constructor sink. No test-only
allocation is classified as production. No native encoder, arithmetic, precision,
copy order, cache policy or API fallback changes. Native kernels' internal roots
remain outside this HF family callback scope and require native live coverage.

B6 corrects load-thread to worker continuity: actual model owns ONE optional
metadata-only handle (readonly source FD/counters, no MlxBuffer/Metal owner).
The same parsed header and handle form a private real-load carrier. Normal
constructor receives None. Selected SlotAware worker explicitly attaches that
handle once; attachments are exclusive and attached guards are non-Send.
Detached handles may move only with no pending source commands/owner transaction;
error/unwind/pending prevents transfer. No new collector is created on worker
entry and no global fallback exists. Metadata-only actual-helper controls cover
exclusive attachment, cross-thread transfer and pending-transfer refusal.
Runtime permit remains unimplemented/unavailable, so the carrier cannot yet be
obtained from a caller packet; this source continuation does not admit a model.
Actual cache/anchor visitor now checks every Present/Absent/dtype/shape/extent
against its source plan before recording that field. No payload read is added.
Weight root metadata retains SAME-FD registered+mapping file spans, not unsafe
mapping addresses; these are not physical-page observations or FD inference.

B7 forward-review closure: diagnostic inputs are opened readonly with
O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC then fstat-validated regular BEFORE reading;
normal None uses the original constructor/path open. The selector is read once
through centralized InvestigationEnv, unsafe acknowledgment and shipping matrix.
Planned evidence hashes are canonical lowercase; duplicate/excess cells refuse
while deserializing. Actual hosted controls cover default None callback, valid
96-byte SAMEFD GGUF-header refusal, FIFO/symlink inputs and closed parser. No
hosted controls or model/native tests were run by this author.

Owner finish refuses pending/unjoined commands and sticky metadata/reentry BEFORE
publication. Actual helper controls cover that sequence and20,000 aggregate
events without old per-case event capacity. The fixed checkpoint census excludes
streamed transient observations; persistent actual weight parents/file spans are
kept separately, with no lifetime/physical deduction. Phase-specific required
root-domain catalog distinguishes independently reviewed preload bounds, actual
owner/conservative native coverage, and fresh external OS evidence. Future Planned
cells never satisfy those requirements or mint a permit.

Worker source now joins seed, bounded/cooperative prefill, scalar/cohort decode
and scalar fallback, plus existing reset/cancellation helper paths. Native/host
failure after existing checks failstops the diagnostic; None preserves previous
error policy. Logical groups2–4 are not physical GPU row/batch claims. Cleanup
may see scheduler automatic max-token release before completion: only a prior
actual completed-owner witness plus exact empty slot and successor generation
can identify retained terminal cleanup; this grants NO forward authority and
refuses occupied/reused/gen+2 slots. All SourceIDs remain in the model's same
metadata handle, not a new collector/global fallback.

ROLLING REMAINING COVERAGE (not yet accepted/implemented): complete warmup and
provision lifecycle attachment; mounted swap snapshots and direct anchor promotion/
late-delivery recovery paths; current per-role/cursor manifest and final bounded
receipt; actual native early creation/live authority integration and exact
source/native/device/process UUID bindings; certified whole-domain partition or
full conservative rejection; source-frozen per-phase/callback/native capacity
proof; CPU/channel/persist and OS/reaction cells; external projected-breach/cleanup
proof. Root-domain registry names are requirements, not evidence of coverage.
The current packet identity hashes remain declared/unverified and only the
coalesced rejected preload schema is accepted syntactically. There is no positive
runtime permit or physical credit authority. Public native0.15.14 is pending;
provider remains explicitly Unavailable, no local Cargo patch or private HF Mach.

B8 rolling source closure (provider still Unavailable): actual provision now
attaches the SAME model metadata context on its constructor thread around the
existing staging/transfer body; None calls original body directly. Actual worker
warmup uses the attached SAME context, requires real quiescent pre-admission
state, rejects selected skip-warmup and observes before/after the existing checked
warmup. Provision/warmup lifecycle cannot overlap a request or close with pending
metadata results. No encoder/math/copy/wait changes. Actual source FD metadata is
checked again at every full model checkpoint. Warmup is a distinct required phase,
with bootstrap actor and NO invented request ID. Native errors remain in the
returned error chain alongside observation failure.

Independent B7 feedback is incorporated: per-operation OwnerScope now retains its
originating metadata handle, is non-Send, and finish verifies exact originating
handle == actual model handle == current attached worker before closing. Hosted
actual-helper cross-context/same-owner-keys negative is authored. These guards
retain only metadata/FD, never buffers. This does not create positive permits.

Remaining after B8: mounted swap/direct promotion/late-delivery observations;
current role/cursor/CPU manifest and bounded final receipt; actual early native
creation/live authority and public14 provider AFTER verified pin; exact process
UUID/source/native/binary/feature/workload/operator binding and independent review;
full per-phase source/capacity/partition/dirty-batch/CPU/OS/reaction proofs and
low-memory projected-breach child. Missing future domains remain Unknown/rejected.
Warmup skip and feature/co-activation intent must ultimately be validated in the
full preload packet BEFORE Device; current always-refused preload prevents a
model being loaded merely to discover these unresolved requirements.


## Consolidated source closure (unexecuted beyond B8)

The connected source now records bounded current borrowed-field manifests with
POSIX monotonic endpoints, exact actual cache positions/capacities/compression
ratios, actual device identity, and request/slot/generation or lifecycle identity.
Actual swap callbacks record explicitly partial mounted/unmounted pairs; an
unbalanced mount cannot publish owner completion. Late delivery/anchor promotion
and shutdown use the same originating context. The terminal record is emitted
once before model drop and explicitly does not certify native lifetime, physical
retirement, process UUID, or runtime admission. Full role snapshots and partial
swap snapshots are distinct. A fixed 16 MiB serialization quota fails closed.
No per-event Vec, buffer retention, new GPU waits, or replay is introduced.

The same-FD preload receipt now projects the actual strict config and pure cache
plan when available, otherwise a typed Unknown reason. It does not substitute
that plan for actual allocation or lower the retained 125.613 GiB envelope.
The callback accepting a private PreparedLoad remains unreachable: no positive
permit is minted. Native page provider stays Unavailable in this Cargo13 tree;
public14 verification elsewhere does not silently change this source pin.

Remaining admission obligations are real integration dependencies, not additional
observational helpers: a published/pinned provider plus actual early whole-native
requested/live authority; reviewed source-domain partition or conservative full
upper; exact source/binary/native/process/workload/operator joins; bounded CPU,
OS/reaction and projected-breach/cleanup evidence; independently reviewed selected
load/window policy. Current refusal is the connected safe outcome, not whole-Q5
acceptance. The author ran no tests, builds, model, header helper or GPU command
for this delta. Root must compile and run the authored actual-entry controls.

Bootstrap source distinction: pristine reuse transfers the original cache into
slot0 and leaves a C128 eager swap stub plus four normal caches total. The fallback
can retain five normal caches. This source does not assume the reuse predicate or
lower the five-cache conservative admission bound. A future tighter packet must
prove reuse and reject before fallback allocation. File-backed requested bytes,
file interval union and physical charge remain separate; no guessed deduplication.

## Published14 bounded provider source slice (unexecuted)

The published14 APIs now supply actual bounded NoCopy root page metadata in the
real SameFD weight callback, paired with an ephemeral readonly mmap from the
retained FD at the verified registered file offset. No raw descriptor address is
mapped or dereferenced. Both observations retain full requested charge. At most
four distinct roots are attempted, at most64 pages per query and250ms per native
query; this is explicitly partial coverage, never whole-file absence. Native
limits refuse stale/unknown/partial results. Owned roots, changedFD, overlap or
query errors retain Unknown/full charge. All unqueried roots remain fully charged.
The alias is dropped before returning; no persistent observer mapping is retained.
RootID/requestedbytes/fileinterval association comes from actual source callbacks.

Finite added work: at most8 native queries (2 seconds accepted query-time total;
OS calls are not interruptible and their total latency is NOT certified). At most
4*128*i32=2048 raw-disposition bytes, plus bounded record strings. One alias at a
time, at most64*65536=4MiB virtual file range (typical16KiB pages gives1MiB); its
PTE/driver/OS effects remain unqualified, with no physical credit. Source metadata
FD clones are transient. File interval union uses at most16384 actual parent
spans (<256KiB payload) and preserves FULL requested bytes separately; it does not
identify PFNs or deduplicate physical charge. Rounded spans outside EOF refuse
coverage rather than infer a shared physical tail.

Actual whole-native requested/live capture requires the real early public handle.
The connected context has a typed unavailable state by default; no process-global
getter, borrowed root census or pool sum substitutes for that authority. The
already finite ignored122root tiny control now installs the same requested/live
ceilings, binds that real handle and checks an actual snapshot without GPU commands.
Production cannot bind an invented early handle, so positive runtime admission is
still refused. The real strict-header projection now has a hosted positive using
the existing official0731 GGUF fixture and missing-field/layer-bound negatives.
Pure tests exercise the actual page-result gate and actual file-union overflow /
overlap handling. Author execution: none. No default loader/cache/math changes.

Remaining connected positive dependencies: reviewed early creation-history/live
limits and installation at the admitted pre-Device boundary; exact whole-domain
partition or conservative residual; certified fullresident CPU/channel/OS/reaction
upper and external process/source/binary/workload bindings; actual low-memory
reaction/cleanup qualification. A reclaim window remains a separately measured
memory mechanism, never an assumed latency win. Current125.613GiB rejection stays.
