# DeepSeek actual cache/anchor ownership roles — diagnostic source candidate

Branch cut current main `63d0ba4c88db44dcf59675ee8a5c1c3e3fba1209`, which
contains the merged generic/Qwen helper. The seven-file family delta preserves
the independently reviewed source, with a test-name lint correction. Cargo remains exact
public13 registry dependency; checksum66d9f8bb0296ca7497ebe7834bc817b2a0326aa5755fdcc7660760ae2e442270.
No local patch, new native API/version, serving activation or model load.

## Family-owned source traversal

The cfg(test) adapter lives inside DeepSeek's cache module so it can read actual
private compact-anchor fields without exposing new production mutators. It reuses
only generic actual buffer backing_snapshot/RoleSnapshot, never Qwen semantics.
All eight LayerCache fields are explicitly observed: attention_kv parent, window
and compressed zero-copy views, indexer_kv, main KV/score recurrent state, indexer
KV/score recurrent state. Optional field None is Absent, Some is Present with actual
metadata; missing external ownership is Unknown. Full allocated parent/logical-view
capacity is reported regardless of cursor. Window/compressed views retain the SAME
native parent ID; consumers must not sum their logical extents as extra parents.
Ratios/capacity/position/poison/graft metadata are observed from the actual cache.

Deepseek4CacheSnapshot owns only compact window copies and four optional recurrent
state copies. It does NOT own append-only compressed/indexer KV: those remain on
the primary cache. All five snapshot fields are visited privately, preserving actual
independent copy root IDs. Same bytes are not aliases. Committed and pending anchor
presence is distinct, with neither automatically dropping/refunding the other.
Score-state -infinity initialization is a source sentinel, not a finite-float failure
or integer-ID interpretation; role observation itself does not touch payload.

A cfg(test) actual loaded-owner adapter also reads Deepseek4LoadedModel.cache,
live_logits, committed/pending anchors and every actual retained slot_session's
same four fields. Current loaded/bootstrap cache is included even when sessions
exist; no assumption that it equals a session, and no five-to-four root subtraction.
Session vector indices are actual indices, NOT fabricated scheduler slot handles,
generations or request identity. All optional presences come from actual fields.
Provided-owner traversal is shared with the tiny cache control; it proves only
provided owner scope, not global process completeness.

## Explicit external gaps

Deepseek4Cache and Snapshot do NOT own MTP/drafter caches, model weights/CPU staging,
TransientScratch pools, submission chains or local attention workspaces. No auxiliary
MTP zero-byte/unsupported inference follows from this type-level absence. These
scopes are Unknown, never zero. The selected DeepSeek serving owner is a verifier
bundle, but a separate producer/capability audit remains required for T5 semantics.

Actual forward-local owners reviewed: CompressedAttentionCoreArena's state_norm,
mixes/pre/post/comb, attention input/norm, Q_a/norm/Q/norm, KV/norm, unrotated/F32
attention/outputA/outputB/output_state, four param roots and optional grouped_output;
RatioFourIndexerArena's ten query/weight/compressor/score/topk roots;
SparseFlashDecodeArena's gatheredKV/mask/two invalid flags; prefill flash KV/mask/
block and sparse prefill KV/gatheredKV/mask/two flags. They are created by forward
locals, not retained on these cache structs. Some(grouped_output), indexer and
prefill/decode workspace traversal remains UNIMPLEMENTED/UNEXECUTED and Unknown.
No giant workspace is allocated to manufacture a fit claim. Such transient roots
need an actual in-scope observation or separate source-bound allocation ledger,
including all pending command owners. CPU vectors, disk persistence/IO and OS
partition remain outside native buffer metadata. Gemma's hybrid/cache variants are
not a small interchangeable follow-on and are not claimed covered by this delta.

## Controls and root validation

One pure actual production planner control verifies a legal tiny geometry with
three layers/ratios0,4,128, C256, window128, head512/index128; all other cache-relevant
DS dimensions retain their explicit family geometry. One fresh-process ignored
control allocates TWO independent initialized caches and TWO real compact snapshots
per cache. No model, forward or GPU commands; cursor commits are explicitly metadata
fixture setup, not an executed-token claim. Snapshot CPU copies read initialized
source bytes. Native dispatch/commit/sync counters must remain unchanged.

|Owner|Count|Roots each|Requested bytes each|
|---|---:|---:|---:|
|Primary tiny cache|2|10|1083392|
|Committed/pending compact anchors|4|9|999424|
|TOTAL||56|6164480|

Cache per-layer parent bytes are131072/294912/657408. Per cache all24field names
must match;15Present/9Absent,15buffer roles collapse to10roots via actual window/
compressed aliases. Every anchor has15field names/9Present independent roots.
All56root IDs must be disjoint except explicitly tested cache views/clones, with
exact requested-root union; alias logical bytes are NEVER summed. Both owners have
distinct cursors1/2 and exact full capacities. Source -infinity score sentinels,
optional presences, retained anchor ownership after primary drop, and unknown
external scopes are checked. No released page/physical credit is claimed.

PRE-DEVICE test-userdata envelope:6164480B payload +56*65536B pessimistic parent
rounding allowance +1MiB metadata <16MiB. This is not process/driver/physical peak.
Root must independently admit/control any actual run. Allocation authority bounds
56attempts/roots and exact6164480requested bytes, so unexpected roots/refills fail.

This supplies source-complete cache/compact-anchor/selected loaded-session BUFFER
field traversal, not full DeepSeek process ownership. Selected model snapshots, verified file-to-owned weight joins,
CPU/transient/OS disjoint ledger and full Q5 gates remain separate. No old result
is rescored, and no physical admission or Q5 completion is inferred.

Root validation on source `2c7fcab6068be1549f415653c8159e6eb0331fc0`:

- Locked optimized test compilation passed with debug assertions disabled.
- The pure planner control passed.
- The isolated actual two-cache/four-anchor control passed in a fresh process;
  all 56 roots and 6,164,480 requested bytes matched. The process group was reaped.
  The control asserted unchanged native dispatch/commit/sync counters.
- `cargo check --locked --all-targets --all-features` passed.

Evidence is retained locally under campaign `adr063-preflight-20261023`:
`hf-q5-deepseek-role-release-root-build-v2/build-receipt.json`,
`hf-q5-deepseek-role-actual-controls-root-v1/receipt.json`, and
`hf-q5-deepseek-role-check-all-v1.log`. Optimized test binary SHA256:
`9c67c380270dfaf41fa2124afa1893aab83e881e815c318dac8a05d74633074d`.
The initial compilation refused an uppercase test identifier; that failed build
is retained and the corrected build above is the validation authority.
No real model, serving behavior, performance, release credit or physical fit
was validated by these controls. Product release build and hosted CI are separate.
