# Isolated one-layer checked submission group

This is a source-only spike, not adoption, speed, Q1/Q5 completion or physical
admission. Branch cut from current mainD010; exact43-path main→c739dc03 candidate
patch is composed and separately hash-frozen before this delta. Published native
0.15.12 pin and original candidate fixture/input/workload remain unchanged. No
new native API/release/path patch or production-default activation. Only width2
is admitted; width3/4 refuses before graph mutation. Abstract width4 cap controls
are not real-model width4 qualification.

## Observable contract

Each selected Q38 layer opens one pool epoch. Its two actual owners each submit
one row-copy CB and, for linear layers, one Delta CB. The exact encoder enters a
pre-reserved bounded vector BEFORE one plain commit(). It is not recommitted by
retirement. All use the same actual graph MlxDevice/queue. Unretained references
and sync-redirecting MLX_PROFILE_CB are refused. FA helper/CPU-read/flag/head fences
are not deferred. FFN retains its existing checked commit_and_wait as the layer
anchor. Then wait_until_completed on EVERY retained actual CB obtains its own
healthy Result; these are real wait API calls after an existing anchor, not zero
API calls or inferred health. Existing native SYNC_COUNT does not count wait-only
calls. No timestamp property authorizes health or ownership retirement.

N2 cap:4pending+one anchor,224retained submissions max per64-layer invocation
(48linear*2+64copies*2). Abstract N4 cap8+anchor/448 is source-controlled but NOT
admitted. Epoch state checks selected-row order, exactly one Delta before copy
on linear layers, no Delta on full-attention layers, both copies before retirement,
and no inter-layer overlap. Partial entry/commit/anchor/status/timing error causes
sticky worker/cache/profile quarantine, no scalar replay or further graph work.
Check-all runs even on anchor Error; failed checks never clear the owner vector.
Scope Drop leaks uncertain pending owners rather than inventing release/refund.

## Pool lifetime proof

Keeping Metal CB references does not make a pool root unique. The unchanged pool
reset blindly makes issued roots free; CPU writes to parameters could then corrupt
in-flight commands. This spike moves ALL per-row resets/truncations to the existing
layer fence. The central checked_reset_decode_pool gate refuses ANY reset while
an epoch is open or an encoder pending, even before its first commit. Both public
reset wrappers route through it. The only production profile::reset caller remains
that central function; no per-buffer release/clear path exists in these helpers.
Allocations may use prior quiescent free roots or fresh roots, but newly issued
roots cannot reenter free during the layer. Selected helper params are freshly
issued pool roots; there is no cross-owner cached CPU-refilled parameter singleton.

Global roots retain actual helper/position/output wrappers across both rows. After
all checked Results, drop retained encoders, then truncate the helper suffix and
reset ONCE. Packed attention remains outside the suffix. The packed FFN's input,
residual and final hidden locals are direct device parents (N>=2), held across
suffix drop/reset and reattached to the durable roots. Returned hidden cannot be
recycled as a pooled gate buffer. Any preceding error keeps the suffix quarantined.
The existing final diagnostic fault hook now fires after all layer copy Results
are actually healthy, BEFORE retirement/reset; it remains injected failure rather
than a Metal error and runs no GPU work after the injected error.

## Source bounds, not physical credit

Prior source audit enumerates21 R1 Delta pooled roots/222044requested/320632bucket
bytes per owner. Shared FFN adds four pool roots: threeN*17408*4 scratch +4Bparams.
N2 encoded catalogue working group:46roots/861884requested/1427700bucket bytes.
If catalogue cannot exclude five F32->BF16 projection casts per owner, source upper
is56roots/464201404requested/540395764bucket bytes. Native/CPU/driver metadata is
not physical fit. ON24 bytes were already charged in prior prospective ledger;
actual historical root/bucket/attempt headroom still requires admission binding.
No live-cache/retired-I/O refunds. Full FA working sets/fences remain their existing
separate source cells. V2 introduces an explicit NEW measurement profile/schema: row-copy timing is
the exact retained wait-only check after the FFN anchor; its submission work
remains inside full target host timing. The old V1 contract is not reinterpreted. Stage-event context
is stored per actual entry, so earlier owners are not relabeled as the last row.

Authored UNRUN host controls cover actual epoch/cap/reset/truncation gates,
check-all on anchor/retained failure, owner retention/healthy retirement, default
checked-fixture no-group scope and retained-reference refusal. Original full-cache/
carry/subsequent-decision fixture remains byte-unchanged, including4GiB per-case cap.
All builds, actual model/parity/timing/admission and publication remain root-owned
and pending independent source review. No performance causality is inferred from
host versus summed chosen-CB durations; those are not GPU unions.


## V2 measurement interface repair after independent HOLD

V1's core lifetime proof had no additional source blocker, but its unchanged V1
timing protocol mislabeled post-anchor wait-only row checks and emitted a different
chronology. That source freeze/review remains retained as HOLD, not retroactively
accepted. This correction changes measurement metadata only; group lifetime and
all resource/CB caps remain unchanged.

The old target_timing.rs child, old Event fields/legacy Scope and old external V1
strict parser/results remain retained. NEW grouped_target_timing ignored filter
requires packet schema hf2q.s3.layer-group-target-timing.v2 and closed profile
qwen38-n2-one-layer-checked-group-v1. GroupedScope serializes a separate v2 event
payload, and new result/sample schemas bind that exact profile. A legacy active
timing scope is refused by this grouped graph, never silently relabeled V2. Old
binary/source receipts remain the only authority for historical V1 actual runs.
Parent activation must bind the NEW filter/profile/parser/source/binary; no new
controller/binary or activation is claimed by this source freeze.

Exact event order is FFN anchor first, then actual retained submission order:
LA Delta(owner3),copy3,Delta(owner0),copy0; FA copy3,copy0. An actual ordinal is the
ordered timing-record index, NOT a native CB ID. FFN host boundary is
commit_and_wait_after_encoding_including_queued_upstream; retained checks use
wait_until_completed_after_ffn_anchor. Every GPU duration is observed only after
its own healthy Result. None is measured/declared healthy at submit. Rust Instant
durations are named explicitly; no POSIX/GPU-clock join or GPU union is implied.
The full target host duration still includes submission costs omitted from wait-only
per-copy metadata. Unknown/unhooked stages and all scope qualification flags remain.

The NEW strict parser preserves original full1320records/sample,260persistent roots,
24fixed samples/12pairs/10timed perarm, complete cold preparations and all-full-output
parity/median/resource checks. It additionally rejects old schema/order, missing or
unknown profile, submit-as-health, old CPU boundary, reordered/duplicate ordinals,
fake GPU union and dropped metadata. Re-sealed negative controls invoke the SAME
new full parser; old V1 parser must reject the new schema. All new controls are
AUTHORED UNRUN. Serialized output remains capped64MiB, per-file16MiB, event1024;
no native/pool/physical cap increase. A separate grouped vector adds bounded
1024*sizeof(GroupedEvent) metadata while legacy vector capacity may remain allocated;
this fits the existing1GiB host allowance but is not an actual footprint claim.
