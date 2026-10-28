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
