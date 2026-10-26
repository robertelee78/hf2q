# Qwen dense shared target forward: HLD and LLD proposal

Status: design for review, 2026-10-26. Source base
`98fdffa1817fc7c11c1fb9c5a68ab36ef4d2dd1f`. No runtime code, execution,
performance result, implementation approval, or Ready claim accompanies this note.
This proposal addresses issue #363 and [ADR-063](../adr/ADR-063-apple-silicon-four-agent-speed-coherence.md)
§S3: amortize target weight reads through real physical batching, preserve exact
ordinary fallback, and compare physical batch-four with independent and
speculative cohorts. F2 provides the source-bound baseline; Q1 correctness and Q5
memory remain execution gates. Designing is not implementation approval.

## Observed gap and hypothesis

`src/serve/api/engine.rs::decode_batch_qwen35` iterates scheduler handles and
invokes `Qwen35DecodeState::decode_tick` separately. In
`src/serve/api/engine_qwen35.rs::decode_tick`, pending speculative outputs,
history lookup and MTP are handled before the ordinary target branch. That branch
passes one last token, one SlotId and four repeated position coordinates to
`forward_gpu_greedy` or `forward_gpu_last_logits`. The coordinates are not four
independent rows. MTP verification rows are positions within one conversation;
they are not cross-slot batch rows.

Hypothesis: sharing dense projection weight reads for 2–4 simultaneously eligible
ordinary targets can reduce N4 target service time. Neither scheduler width nor
source inspection proves a speed win. AUTO may spend most of its time in history
or MTP; an ordinary-only mechanism could have little end-to-end benefit. Keep AUTO,
TQ8, context262144, all four requests, tool semantics and useful defaults intact.
The actual image-free run loaded a projector; account for its residency.

## HLD: one worker, explicit independent lanes

The existing owning worker constructs an immutable cohort of up to four live
`SlotHandle {slot_id,generation}` entries. Each lane carries token, absolute
multi-axis positions, current full-attention cursors, linear-state parity and
output destination. One model/device/queue/configuration generation is required.
Do not use token-sequence length as the independent-lane dimension.

First implementation candidate is dense Qwen ordinary one-token target work.
Pending-output lanes advance using their existing CPU path; history/MTP proposals
retain their existing target transaction. They cannot be coerced into ordinary
mode to fill a cohort. A one-lane or unsupported exact capability returns to the
existing route **before any mutation/submission**. No post-submit serial replay.
A future mixed speculative/ordinary cohort needs a separate lane/temporal mapping
and capture proof; it is outside this first candidate.

The graph shares embedding/body dense projections and output projection over
`[lanes,hidden]`. Between projections the proposed extracted helpers execute each lane's attention
and DeltaNet state transition against slot-indexed views, then pack outputs for
the next shared projection. These projection/state seams do not yet exist. This provides an incremental graph design without
pretending a contiguous multi-token prefill has independent-sequence semantics.
If extra pack/unpack dispatches erase amortization, stop or propose a measured
native strided/batched primitive. Never route these independent tokens through the
existing causal temporal scan as one sequence.

Sampling, grammar, thinking-forced tokens, repetition history, RNG, stop detection,
UTF-8/tool splitters and SSE remain per request. Batched greedy argmax is usable
only if its exact per-row tie semantics match the current greedy path; otherwise
return row logits and apply the existing independent policy. Avoid vocabulary-sized
CPU transfers for greedy requests merely to simplify the implementation.

## LLD: planning and execution boundaries

Proposed focused modules (names provisional):

- `src/serve/api/qwen35_shared_target.rs`: checked cohort planner and lane results;
  `engine.rs` retains scheduler ownership and reply routing.
- `engine_qwen35.rs`: split ordinary target preparation from existing token commit;
  reuse one commit routine for serial and shared paths, including forced thinking
  tokens, grammar advancement, EOS and sampling history. Preserve ordering/errors.
- `src/inference/models/qwen35/shared_decode.rs`: dense shared graph with explicit
  lane descriptors; `forward_gpu.rs` delegates shared projection assembly through
  existing helpers, rather than duplicating quantization or head policy.
- Existing `kv_cache.rs`, `gpu_delta_net.rs` and TQ attention helpers expose only
  the minimum checked slot views/transaction metadata required. A future native
  API is a separate native issue, publication, exact registry pin and parity gate.

`prepare_cohort` is pure: reject duplicate handles/slots, stale generations,
nondecode states, exhausted output budgets, incompatible cache/configuration,
unsupported graph modes and malformed checked extents. Capture exact row-to-handle
mapping and cancellation snapshot. Validate all lanes before taking ownership out
of slot storage. Cancellation before submit removes only that lane and replans;
it must not release another lane's reservation.

`encode_shared_target` retains every input, packed temporary, cache root and output
owner until a single known terminal drain. All fallible shape/capability checks
and reservation checks occur before writes. Existing graph annotations must include
physical RAW/WAR/WAW across packed projections and each slot's state views.
Do not reset scratch on drain alone while issued views have future use. Native55
quiescent and ordered-reuse contracts remain unchanged; no speculative reset or
new wait is introduced solely to make an unsafe reuse legal.

After successful drain, validate row count/identity and publish cache cursors/parity
and lane results in one worker-owned commit phase. A client cancelled after submit
still participates in drain; discard its response, run the current slot recovery
policy and never expose its row to a new generation. GPU failure/unknown completion
uses current worker fail-stop/quarantine; retain owners until established safe
cleanup. No healthy serial fallback after possible state writes.

### Required projection/state seams and pending metadata

`gpu_delta_net.rs::build_delta_net_layer_decode_into` is not a reusable state-only
API: it owns prenorm, qkv/z/gate projections, convolution/recurrent updates and
output processing. Required hf2q extraction separates (a) the same prenorm and
projection recipes over packed independent rows, (b) an encode-only per-lane state
helper consuming explicit preprojected qkv, z and gate outputs plus checked
slot-local conv/recurrent read/write views, and (c) unchanged output processing
and shared output projection. Exact tensor shapes, activation order and dtype
conversions must be lifted from that implementation, not rederived. No existing
helper is claimed to accept these inputs. Native operations remain native.

The existing `forward_gpu.rs` greedy assembly calls `swap_for_slot` during encode
(around7109), before terminal drain. The shared path must not call that mutating
publication method while encoding. A worker-owned `PendingCohortState` instead
holds each layer/lane's initial parity, effective parity after encoded steps,
expected cursor and next cursor. Subsequent encode stages resolve views through
this local pending descriptor. Global parity and cursor publication happens only
at checked terminal drain, once. Existing serial behavior is not changed by this
new path's refactor.

Full-attention needs an analogous extraction: shared q/k/v projections followed
by explicit per-lane position/RoPE, unchanged TQ encoding and slot-local append
writes, then per-lane attention and shared output projection. TQ code/scales and
all row/sidecar extents are validated before encoding; pending full-attention row
counts and cursors remain in the same descriptor. Physical writes occur before
logical publication: pending metadata is not a rollback of overwritten bytes.

Initial support is dense ordinary one-token decode with the production TQ8
configuration and exact supported graph helpers. Refuse pre-submit for MoE,
soft-token/deepstack/graft variants, diagnostic split/capture modes, temporal
verification/prefill or any helper layout not explicitly proved. This is a
capability refusal to the existing exact route, not a silent default change.

### Cache and rollback ownership

`kv_cache.rs::FullAttnKvSlot` owns optional F32 or TQ roots and a per-sequence
`current_len` vector. Preserve current TQ code/scales/layout and slot offsets;
sharing projection inputs does not share cache histories. Full-attention append
writes remain private to each slot's next cursor. Tail bytes are not logically
published before commit; they cannot justify reuse of poisoned state.

`LinearAttnStateSlot` has conv and recurrent ping-pong roots, per-slot
`pp_flipped`, plus optional recurrent/conv captures. `gpu_delta_net.rs` explicitly
narrows state through `slot_recurrent_region`, `slot_conv_state_region` and
`LaPingPongSlotView`; named current/scratch fields alone are insufficient. Preserve
per-slot parity and flip exactly once after successful target completion. Ordinary
one-token work does not require speculative captures merely because four lanes
exist. Inactive slots and existing captures must remain byte-identical.

Do not promise rollback by restoring cursors only: recurrent state and conv parity
also mutate. Preparation failures produce no writes. Post-submit device failure is
fail-stop, not a rollback opportunity. Semantic postprocessing failure after a
successful target follows existing slot recovery; cohort nonmembers must not be
rewound. Arithmetic-dependent cache/recurrent values require the Q1 numerical oracle;
bitwise ownership/index checks do not imply bitwise arithmetic equivalence.
If independent lane commit cannot preserve this policy, use staged next
state with explicit copy/owner cost or reject the design before implementation.

### Failure and once-only completion table

Before submission validate the immutable row/slot/configuration map, generation,
all extents and reservation identities. After any GPU state write, a row mapping,
slot identity, generation, cursor, parity or configuration integrity mismatch is
cohort/worker fail-stop and quarantine, even if device drain succeeded. Never
reset just the apparently offending lane, replay serially or recycle uncertain
owners. Unknown/failed drain has the same fail-stop disposition. Retain roots
through known-safe cleanup; absence of a completion proof is not quiescence.

A known lane cancelled after submission is different: mapping and state integrity
must first be established and the submitted cohort drained. Its response is then
discarded and the existing lane-local recovery applied. Known-lane sampling,
grammar or tool-postprocessing errors may use existing recovery only after this
same integrity proof; an error that prevents proving the proof is cohort fatal.
No other lane's state may be rewound as a convenience.

Prepare fallible host result data before publication where possible. Metadata
commit is an infallible checked application of the pending descriptor; an
unexpected failure after partial host publication poisons the worker rather than
returning it to service. A completion ledger keyed by full SlotHandle tracks
planned/submitted/drained/metadata-committed/replied/released states. Each host
commit, reply and scheduler reservation release occurs once, including error
fanout. It must not turn stale-generation cleanup into a new generation's release.
Failure injection covers before first write, after each layer/state-write stage,
failed/unknown drain, after drain before commit, each lane's host commit, and
cancel/stale-generation interleavings. Inactive and unaffected-lane continuation
oracles distinguish recoverable known-lane errors from fail-stop cases.

## Native capability and routing

Published0.15.4 and proposed0.15.5 census additions do not supply a Qwen cross-slot
graph. Reuse existing native dense quantized matmul dispatch where its actual
layout/dtype/codec contract matches. `mlx-native/src/ops/quantized_matmul_ggml.rs`
uses `plan_dense_auto_route` from `src/ggml_capability.rs`; MM threshold8 means
M4 is not automatically a matrix-matrix route. Existing Q5_K canonical routing,
Q4_K/Q6_K narrow multicolumn and mv_ext routes must be measured, not reimplemented.
The canonical Q38 launcher sets mvN0/mv_ext1; preserve these for the primary
comparison. A separate matched route experiment can decide alternatives. Generic
route changes may alter floating-point accumulation and require a numerical gate.

Do not assume existing temporal DeltaNet, TQ writer or attention kernels accept
heterogeneous slot positions. Initially invoke their checked per-slot path between
shared projections. Missing lane-strided kernels, batched greedy extraction or
weight-layout support belong exclusively in mlx-native. hf2q owns lane assembly,
state transactions and request policy. No kernel code or fallback engine in hf2q.

## Peak memory and observability

Before execution extend the existing disjoint Q5 packet with named shared-graph
roots and transitions. Persistent weights/cache/projector/MTP are shared existing
owners, not multiplied by lanes. Count packed hidden/projection buffers, per-lane
attention/DeltaNet temporaries, row logits or argmax, parameter buffers, any staged
next-state, preexisting serial arenas and graph-retained old owners that coexist.
Use actual allocator bucket extents, root identity/view dedup and bounded history;
logical rows alone are not a physical upper bound. Host lane descriptors, tokenizer,
grammar, sampling and returned outputs remain separate CPU cells. Driver/Metal
metadata and unavailable high-water are explicit unknowns/allowances, never zero.
Pool census snapshots are observational current-owner data, not admission credit.

Record actual target dimensions/codec/route counts and lane occupancy; target
forwards per committed token; history/MTP proposals, acceptance and cost disables;
GPU command time, CPU encode/wait time and per-lane queue delay. Use existing
instrumentation first, minimally instrument uncovered row mapping. Diagnostic
instrumentation arms do not substitute for uninstrumented timing.

## Deciding tests and measurements

1. Pure planner tests use noncontiguous slots, unequal positions/parity, duplicate
   and stale generations, cancellation before planning, unsupported modes and
   overflow. Assert exact mapping and no mutation on refusal; test actual planner.
2. Tiny model-free graph controls initialize all bytes with distinct signed lane
   data and inactive-slot canaries. Compare independent versus shared projections
   and full layer state transitions, permutations, widths1/2/3/4, unequal cursors,
   both ping-pong parities, TQ writes and capture noninterference. The independent
   reference is the existing production path, with a CPU oracle for small primitive
   cases. Intentionally swapped row/slot mappings must fail the same comparator.
3. Q1 numerical approval must predeclare dtype-specific tolerances from existing
   kernel parity gates before running. Require bitwise cache/index/cursor/parity
   equivalence where arithmetic is unchanged; report max/mean logit errors and
   greedy margin/tie changes where shared arithmetic differs. No tolerance retrofit.
   Fixed RNG sampling and grammar/tool outputs must pass the accepted quality rule.
4. After F2/Q5 and reviewed execution packet, compare serial versus shared target
   on the same artifact/configuration at N1 and N4, preserving AUTO. Include EOS,
   tool continuation, cancellation during submit, partial failures, inactive-slot
   next-output equality and prefix reuse post-oracle. A row-level result alone is
   insufficient for serving correctness.
5. Predeclare a fixed10 balanced paired timing blocks, identical workload and
   route settings; calculate the95% paired effect interval only after all10. The
   ADR precision target is half-width2.5percentage points: failure to reach it is
   inconclusive, not permission to stop/restart or extend opportunistically. A
   different fixed count or valid sequential-coverage protocol requires a new
   prospective packet before execution. Report successful-task goodput, makespan, semantic TTFT,
   cached tokens, prefill/decode rates and per-slot fairness. No source-derived5%
   claim. If ordinary-target share is small or packing outweighs savings, record
   inconclusive/negative disposition and retain serial production behavior.

## Other-family disposition and review decisions

Qwen3.6 shares the driver but MoE expert dispatch/IdMm ordering differs: do not
apply the dense graph until expert row routing, shape/capacity and numerical
parity have their own disposition. Gemma already uses `forward_decode_body_batched`;
DeepSeek uses `plan_deepseek4_decode_cohort`/`try_decode_batch_deepseek4_cohort`.
Profile their actual remaining unshared work rather than replacing those paths.
This dense ordinary candidate is only part of S3: all-family live profiling and
profitable heterogeneous prompt/verify/decode dispositions remain open. It
establishes no four-family win and leaves the separate DeepSeek mixed-cohort
isolation acceptance intact.

Before implementation review must decide: whether measured AUTO offers enough
ordinary lanes; exact supported dense graph modes; existing primitive versus native
extension disposition; numerical thresholds; cache commit/failure boundary; and a
source-closed peak packet. These are concrete decisions for this design, not a new
universal preflight framework or permission to change workload to obtain a win.
