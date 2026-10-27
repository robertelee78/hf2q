# ADR-063 C1: anchor identity, transactions and ownership

This contract implements the design scope of [C1](https://github.com/robertelee78/hf2q/issues/359)
and its [execution plan](adr063-c1-execution-plan.md). It specifies requirements
for later production integration; the test-only oracle is not a serving cache.
The existing production restore paths do not yet satisfy the atomic protocol
below. C2 and family integration must close those gaps before enabling a new
multi-depth store. No performance claim follows from these tests.

## Identity and coordinates

An anchor is compatible only when artifact digest, architecture/schema,
tokenizer, template, effective graph/cache configuration, conversation,
slot generation, lineage generation and graft generation/digest/length match.
Logical prefix count and digest are independently validated. Identical paths,
shapes or cursors do not establish compatibility. Defaults and quantization
representation belong in the effective configuration identity. Saved logits,
hidden state and speculative state also bind every setting affecting them.

Logical tokens exclude graft. Physical coordinates are family-specific:
checked logical + graft - unconsumed-tail arithmetic is permitted only where
the actual component defines that relationship. Every component validates its
own dtype, extent, capacity and cursor. Overflow, underflow, absent sidecars,
duplicate roles or mismatched representation refuse before mutation.

| Family | Required state and validity |
|---|---|
| Qwen3.8 / Qwen3.6 | FULL cursor plus still-valid live parent rows; per-slot conv/recurrent CPU bytes with actual sequence axes and ping-pong state; target/MTP state captured only after accept/reject settles. FULL KV is not duplicated in the current CPU anchor. |
| Gemma4 | FULL append state and SWA ring bytes/tags/cursor; Hybrid K, packed V, norms and optional xlen representations. Full-F16-V dummy norms are not TQ payloads. Preserve full-input token identity and distinct retry-P-minus-one semantics. |
| DeepSeek4 | Window plus compressor recurrent and score state; valid compressed/indexer rows and ratio-specific geometry. The persisted image also binds the rendered token ledger. Native eight-token recovery is explicit; existing graft/persistence refusal remains. |

A live dependency uses a physical parent identity and half-open byte interval.
Adapters translate the actual row/slot layout into those intervals. A cursor
remaining large enough after a divergent overwrite is insufficient evidence.

## Publication and restore protocol

The slot owns at most K committed anchors, one unpublished pending anchor and
one bounded construction transaction. K excludes pending. Proposed K=4 is a
policy fixture, not permission to allocate unbounded snapshots.

1. Reserve old committed owners, new capture owners and construction temporaries
   together before capture. Bind the request and settled family transaction.
2. After checked capture completion, publish only to pending. Pending cannot be
   selected for reuse, recovery or disk promotion.
3. Commit pending exactly once after successful request/cache/token-ledger
   publication. Cancellation and failed completion discard it. Eviction never
   refunds a still-live asynchronous reader.
4. Preflight every component, identity, physical dependency, intended copy and
   temporary allocation before changing any cursor or byte. No open speculative
   or capture transaction is compatible with restore. A validated ticket is
   bound to its store, epoch and anchor; any intervening mutation invalidates it.
5. Apply only the validated plan. Before any divergent write can execute,
   invalidate descendants and any ancestor/pending payload whose live rows can
   be overwritten. Cancellation cannot resurrect those anchors.
6. A preflight refusal preserves state and may take an explicit cold/reset path.
   A failure after possible mutation poisons the entire store: no shallower
   fallback. Unknown command retirement quarantines all affected owners and
   fails closed. Ordinary reader release cannot retire that quarantine.
7. Reset requires known retirement, no remaining readers, and an advanced slot,
   lineage or graft generation. Graft/session/model replacement invalidates the
   incompatible store before new writes.

The normative oracle uses distinct payload bytes and immutable restore tickets
to test this protocol. It does not perform GPU copies, establish command order,
implement an affinity policy, or establish production restore atomicity.

## Byte ownership and optional disk tier

Charge each unique physical parent extent once, including retained aliases.
Independent equal-valued copies are separate owners. Account for live cache,
K committed, pending, old/new construction coexistence, token ledgers, metadata,
logits, hidden state, serialization buffers and readers. A small view retaining
an 80 MiB parent costs 80 MiB. Saved hidden/logit payloads must use right-sized
owned storage; nominal view length is not a credit.

The oracle's component byte vectors and explicit owner extents exercise this
accounting rule. They are finite fixtures, not a complete production allocator
ledger: production adapters must enumerate all the above classes, including
metadata/container capacity. Existing family `total_bytes` methods cover only
their documented payload and must not be relabeled comprehensive totals.

At known idle, unique cache + anchor + other reservations + free grant must
equal the declared quota. Pending/construction must be empty. Quota release and
Arc lifetime are not OS resident-page or driver retirement measurements.

Disk promotion remains optional and disabled by this change. Only immutable
committed compatible payload may be written with version/checksum and atomic
temporary-file rename. Bound writer queues and encoding buffers; retain and
charge the source until its final reader completes. Hydration preflights all
typed identity/sidecars and old/new memory coexistence before mutation. Pending,
cancelled and poisoned payloads cannot hydrate.

## Regression scope and remaining integration

| Boundary | Normative regression | Actual source exercised | Remaining production proof |
|---|---|---|---|
| Identity / coordinates | Per-field incompatibility, prefix, checked overflow and stale/cross-store tickets | Family allocation/image geometry | Complete identity binding at selection/publication |
| All-component preflight | Last sidecar, dtype, role, capacity and missing component refusal leaves state unchanged | DS serializer rejects late absent state before allocator/read callbacks | Qwen/Gemma/DS restore preflight-all-then-mutate |
| A → B → C rewind/diverge | Old descendants rejected before writes; ancestor retained only while its bytes survive | Existing typed family state layouts | Actual submission/write-range lineage and recovery joins |
| Pending / cancellation | Before/during/after capture, duplicate completion, pending overlap | Actual payload types and CPU quota | Family terminal-success publication and cancellation worker joins |
| Partial failure | Whole-store poison; unknown retirement remains charged | No new GPU failure test claimed | Checked copy/drain failure handling in each worker |
| Isolation / graft | Four conversation identities and generation replacement | DS unsupported graft serialization remains rejected | Actual slot reassignment and graft write ordering |
| Ownership | K4 + pending + construction peak, aliases, right-sized negative, last reader | Gemma actual reservation Drop, Qwen actual CPU anchor payload | Full production owner manifest including tokens/logits/metadata |
| Typed components | Distinct component bytes; missing/invalid live-parent ranges | Qwen dense/MoE FULL/recurrent/MTP planners; Gemma ring/norm/xlen/F16-V; DS ratios 0/4/128 and image emitter | Real restore component parity and unchanged family quality gates |

The focused Rust filter is `anchor_c1`. The normative cases live under
`src/inference/anchor_contract`; actual layout tests are children of the existing
Qwen allocation-plan tests, Gemma anchor-budget module and DS image serializer
tests. They use bounded CPU payloads and no model/device allocation. Validation
results must be recorded against the exact final source and compiler receipt;
source authorship alone is not a PASS. Existing model failures remain failures.

Delivery still requires independent review, executed boundary/layout regressions,
merge and the original Q5 predecessor. Source work was explicitly authorized in
parallel on 2026-10-27; hardware gates and final acceptance were not waived.
