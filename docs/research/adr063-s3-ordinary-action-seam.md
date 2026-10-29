# Qwen ordinary action preparation (source stage)

This narrowly prepares ADR-063 S3's real `decode_batch_qwen35` caller. It does
not activate a cohort, change a model graph or qualify S3. Default production
still calls the same scalar producer and existing semantic/cancellation tails.

The actual-used read-only classifier preserves decode dispatch precedence:
finished bound, verified pending output, history wrapper, MTP wrapper, ordinary
forward. It additionally identifies known nested ordinary leaves under AUTO:
when history's measured cost gate suppresses proposals, its temporary recursive
call can select coherent ordinary MTP warmup or plain ordinary. Looking only
for `history_lookup=None && mtp=None` would miss that real work. A live history
proposal decision remains `HistoryDecisionRequired`; the classifier does not
synchronize/propose, copy lookup history, consume RNG/forced tokens, or mutate
cost controllers, grammar, caches or response state. No hypothetical lookup
miss is presented as known eligibility.

`decode_tick` uses the same classifier for actual dispatch and ordinary greedy
versus full-logit selection. Installed-owner/generation and client-close checks
remain in the existing worker. A trace-level read-only owner preview also
refuses mismatched generation/physical slot or closed client, but `KnownLeaf`
is NOT GPU admission: a future collector must validate current cache cursor,
input/position, artifact/profile/geometry and fresh live ownership before any
shared submission. The preview never replaces scalar cancellation handling.

Seven hosted controls exercise the actual classifier/preview and real cost
controller/greedy gates: temperature0 plus repetition1.05 and tool grammar;
cold AUTO and actual cost warmup/disable; unresolved history; verified-output
and terminal precedence; coherent MTP tail versus greedy/full logits; closed
client and stale generation/slot; history-installed/no-MTP ordinary fallback.
They are authored source controls, not an executed model or scheduler proof.

## Exact minimum next hook after a profitable matched timing result

The next candidate must replace only the actual target forward inside
`Qwen35DecodeState::decode_tick_mtp_warmup` with a typed source-owned
`(full_logits, durable_post_output_RMSNorm_h_nextn)` result. The actual graph
must produce normalized rows using the existing head path; pre-norm residual
is not an acceptable substitute. Keep the current per-owner embedding,
`mtp.process_target_batch`, cursor checks, carry view,
`Qwen35SpecSemanticState::select_and_observe`, queued output commit, lookup
restoration and cost accounting in their existing tail. Plain ordinary retains
its current head and sampler path. This slice intentionally does not extract
that rollback/catchup tail or invent an unused GPU result carrier.

An actual-used lookup-decision continuation is still needed to expose ordinary
leaves selected AFTER a synchronized history miss; until then these remain
unknown, not speculative cohort members. A collector may first use the known
cold/cost-suppressed leaves, but must measure actual eligible widths in the
unchanged AUTO/ON24 canonical workload. The source-known first ordinary decision
per generation does not establish simultaneous width4 or end-to-end gain.

Production cohort activation remains contingent on the paired complete-decision
timing/correctness gate, real owner/cursor/profile admission, cancel/EOS/shared
failure tests, realistic four-agent multi-turn tool and exact-prefix proofs,
and F2/Q1/Q5 requirements. No speculation flag, sampling setting, native pin,
unsafe switch, pool policy, production GPU producer or acceptance criterion is
changed here. Native kernels remain in mlx-native. This preparation is a narrow
landing-path dependency, not an independently completed performance task.

## Typed coherent target-result seam (subsequent source slice)

The scalar warmup now prepares the prior target/MTP cursor pair and immutable
slot/step/input/position fingerprint before its SAME target forward. A typed
full-logits/post-output-RMSNorm row result then enters the SAME completion tail.
Preparation owns the old verifier-hidden reference and is consumed at finish.
The existing measured decision clock still starts after preparation and covers
forward, catchup, canonical decision and commit. No shared producer is selected.

Before any catchup/selection, the actual-used boundary validator refuses stale
state/input, absent MTP, wrong output lengths/F32 dtype, missing/double target
advance or prematurely advanced MTP. Target must be prior+1 while MTP remains
prior. Five hosted controls exercise this exact validator without GPU buffers.
The future shared caller must validate all cohort members first; scalar rollback
here does not authorize shared replay/recovery. All original catchup/carry/
semantic/sampler/cost/commit statements remain unchanged apart from their module
path. Active-history continuation and the owning-worker shared collector are
still the next required implementation, not claimed by this seam.

## Actual owning-worker candidate (default OFF; not qualified)

`HF2Q_QWEN_SHARED_ORDINARY=1` is a closed Category-3 request requiring
`HF2Q_UNSAFE_EXPERIMENTS=1`. It is parsed once in the existing investigation
environment registry. Unset/0 preserves scalar dispatch; invalid/non-Unicode/
unacknowledged intent refuses. Before weights load, selected Q38 dense
geometry/role layout and existing ON24 intent are required; fallible startup
requires SlotAware1–4. Infallible serial spawn refuses the experiment. The root
integration owns the final shipping/activation-matrix registration before merge.

The real `decode_batch_qwen35` collector forms only a contiguous prefix of its
already-selected decode handles; it never waits for four or moves a later RNG
owner before an unsupported earlier one. `admitted_request_id(handle)` reads
the actual scheduler's live exact-generation record, including legitimate ID0.
Worker slot/reply/generation checks and current cache cursors are checked again
before submission. Vision, active verifier capture, graft/GLP and actual
non-TQ8 modes use ordinary handling before shared mutation. Effective TQ bits
come from the SAME immutable dispatcher authority used by write/read/dequant,
combined with the actual cache's TQ-active state, not a hot environment read or
inference from U8 storage. Codebooks5/6 remain scalar.

History preparation now extracts the REAL existing synchronization/propose/
truncation/cost policy. A real nonempty proposal is handed directly to its
existing verifier body, without proposing twice. Empty/cost-suppressed history
owns its lookup continuation until the SAME leaf completes; the pre-existing
cost-disabled MTP-state drop is preserved, not forced for batching. It restores
or extends the exact lookup on success/error. Plain full-logit and coherent-MTP
ordinary actions are supported; GPU-greedy, genuine proposal/emit and unknown
semantics stay scalar. Prepared scalar fallbacks retain any already prepared
forced-token/lookup state rather than consuming it a second time.

During graph submission every reply remains in the original slot table. After
healthy target completion ALL member target/cursor/shape boundaries and the
exact per-operator maps are validated before ANY semantic/MTP tail. Tails and
existing emission/retained-prefix/release run in selected-owner order. A client
closing after shared submission is recovered only after its healthy transaction
tail; it cannot make the worker publish a target-only/MTP-stale prefix. An
uncertain graph/completion error quarantines retained normalized parents,
restores any locally taken reply into its original slot, and invokes the
existing worker fatal fanout with no origin handle to double-release. No reset,
scalar replay, later GPU work or stranded reply is permitted on that path.
Known graph descriptor refusal before operational begin remains prepared scalar.

Successful target-width counters distinguish prepared scalar targets(index1)
from actual shared target groups(index2–4); they are not task success counts.
Bounded per-target records retain actual request/slot/generation/position/token
rows and truthful singleton/shared operator populations. A cohort transaction
scope is NOT by itself an independently qualified native per-op B4 witness.
The strict F2 native observation join must still be proven or remain Unknown.

### Actual measured-service cost basis

The scalar facade retains its original `Instant` endpoints. Shared observations
retain measured complete target transaction D and width K; the allocation is
D/K plus ONLY that owner's originally timed leaf preparation and a fresh
measurement of its mandatory leaf tail. Prepared scalar work similarly measures
its actual width1 target plus its own segments. No virtual/backdated Instant,
other-owner waiting, hypothetical scalar counterfactual or requested latency is
used. Raw D,K,allocated target,own preparation,own tail and total service cost
are logged. These are amortized observed service costs, NOT request latency;
F2 makespan/TTFT remain actual unamortized wall times.

Both existing controllers observe the same completed leaf service value at the
same endpoint as before: plain after token selection, coherent after MTP
catchup/carry/canonical decision. Coherent prior-hidden preparation was outside
its old timer and remains excluded. Plain forced-token preparation was inside
its old timer and remains included. History's host lookup prelude/restoration
was outside the ordinary leaf observation and is not double charged. Existing
controller thresholds/algorithms, AUTO/lookup/MTP policies and sampler/grammar
are unchanged; their decisions can legitimately react to measured new service
cost. This cost basis requires independent review and the real AUTO gate.

The source composes with the independently authored shared graph (and root's
legitimate ID0 repair) plus the exact published1511 pin. It does not copy those
files or import the127-path Q1 research foundation. No source author ran a
build/test/model. Width2/3/4 raw target/MTP/carry/decision parity, fresh serial
model gates, unchanged canonical28-request serving/prefix proof and measured
end-to-end acceptance are mandatory before promotion. No S3/F2/Q1/Q5 or product
release acceptance is asserted by this candidate source.
