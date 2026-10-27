# Gemma matched speed-study client profile (source stage)

Execution update, 2026-10-26: the fixed 20-cohort component study completed and
passed independent review. See the [recorded result](adr063-gemma-matched-study-result-2026-10-26.md)
for exact executed identities and scope. The source-stage discussion below
preserves the prospective design; its pending author-only validation statements
are historical. Current integration passed all 186 client controls, six neutral
exporter controls, the locked all-target/all-feature check, and the full binary
suite (5,425 passed, zero failed, 79 ignored).

Governing contracts: [ADR-063](../adr/ADR-063-apple-silicon-four-agent-speed-coherence.md)
and the experimental Gemma checkpoint section of [ADR-040](../adr/ADR-040-continuous-batching-reopen.md).

This benchmark-only profile does not start a server, run a study, or qualify a
speedup. Default client behavior and the canonical `Tools` implementation stay
unchanged. Explicit `--benchmark-profile gemma-matched-v1` and matching
`benchmark_profile` approval fields on coordinator and agent template are both
required. It refuses other families, peer engines and single-slot clients.

## Stable model-visible tool results from real work

The ordinary six real tools still execute in the same order. Reads return exact
actual file bytes; writes remain actual bounded, durable writes. `run_tests`
still invokes the production compiler/sandbox/process guard, fresh target,
real compiler-emitted executables and the actual Rust test harness. No command,
source, test, sampling, model output or quality assertion is replaced.

After successful real execution, the opt-in wrapper reads retained raw Cargo
JSON, process receipts and test stdout/stderr. It requires one successful build,
the exact expected test-target set, actual executable provenance/hash, every
process exit and cleanup, two named passed tests per artifact, and no test
stderr. Ports means one artifact/two tests; all means library plus integration
target/two artifacts/four tests. Malformed, nonzero, incomplete, foreign,
changed-artifact or cleanup evidence fails the conversation, not a semantic
pass. Compiler errors fail; warning counts remain in the returned result.

Only then does it return deterministic model-visible JSON: schema, requested
target, true pass outcome, exact source SHA, warning count and sorted actual
artifact/test names with statuses. Physical paths, compiler elapsed times,
PIDs, random cwd/evidence IDs and executable binary hashes are absent from this
semantic wire. Full raw compiler/test output, original model-result string,
artifact hashes and all process receipts remain retained. Nothing rewrites the
captured HTTP requests after execution. This is a new prospective tool-wire
benchmark profile, not a claim earlier canonical diagnostic requests were
identical across arms.

Existing `qwen_n4_live_smoke.identity(agent, run)` already excludes `run` from its
returned text; four stable agent identities remain distinct. Tool input paths
are already the fixed relative `src/ports.rs`/`tests/ports.rs`. Removing volatile
test-result details makes fresh physical fixture paths non-model-visible. The
profile always creates fresh exclusive owned fixtures/targets; it does not add
a destructive path-reuse or artifact-deletion mechanism. If a later controller
reuses canonical physical paths, it must archive prior raw evidence and verify
owned cleanup before reuse. No benchmark ID is added to model messages.

Exact token comparability remains mandatory: even with deterministic tools,
the model may generate different arguments/code/text under different cache
paths. Never canonicalize assistant output to make it match. A future reviewed
production-renderer exporter and bound evidence adapter must supply all28
actual input-token vectors for corresponding OFF/ON cohorts; the pure gate
compares every u32 token and rejects any mismatch, missing turn or hashes-only
substitute. The existing post-oracle hashes alone do not replace vector equality.

## Fixed checked run plan (not a hardware runner)

Candidate source18162bb702806ac6c083bf01ec80f4675207fda4 and clean binary
ff655efd5013a11ad608330f35b3622c2e448f019dcc4e94c8c483eb03cd09e1,
public native0.15.5, identical cached Google model. A=stage OFF (flag absent),
B=stage ON (flag1 plus unsafe acknowledgment1); OFF leaves both unset. All other family settings
and controls identical:4x262144,8192 output allowance, Hybrid/TQ8, rep1.05,
LCP8GiB/KV48GiB, text-only projectorfalse and actual six tools/seven turns per
agent. Each cohort starts a fresh sole server with identical warmup; no prior
KV between cohorts. No output/task/slot/context reduction or forced responses.

| Block | Fixed cohort indices | Arms |
|---|---|---|
|0|0,1,2,3|OFF,ON,ON,OFF|
|1|4,5,6,7|OFF,ON,ON,OFF|
|2|8,9,10,11|OFF,ON,ON,OFF|
|3|12,13,14,15|OFF,ON,ON,OFF|
|4|16,17,18,19|OFF,ON,ON,OFF|

All20 cohorts/ten per arm must pass actual coding, independent patch/tool
quality, raw response bindings, cleanup and paired exact tokens. No partial
quality denominator, failed-baseline ratio, favorable subset/retry replacement
or optional run. OFF needs honest strict-anchor reuse/coding correctness, not
an ON full-input capture assertion. ON additionally needs the full-input family
capture/replay proof. Never run an offline prefix helper while HF is loaded.

Use the same prospectively reviewed96GiB operating available-memory floor and
live guards in both arms. The extra1GiB conservative aggregate tools charge
brings explained operating packet to95.531978GiB; anchor1.5GiB is inside CPU3,
not extra. These remain operating assumptions, not formal physical peak proof.
Final observer/thermal policy, exact once-only load controller, clean candidate
binding and actual helper/evidence adapters require separate review before
execution; none is supplied or silently approved by this profile.

## Primary metric and reproducible pure calculation

SSE turns1,3,5 supply actual semanticTTFT (not empty-role or reasoning frames).
Unary turns2,4,6 have no such measurement; do not invent unary semanticTTFT.
Per-agent composite is the geometric mean of its three SSE times; N4 cohort
score is the geometric mean of four agent composites (equivalent to all12
positive finite observations). The coordinator records this score only after
existing full quality/cleanup aggregation succeeds. It explicitly leaves input
comparability pending and performance qualificationfalse.

Primary ratio: median(ten ON cohort scores)/median(ten OFF scores). CI uses
10000 fixed-seed6306 bootstrap samples of five paired ABBA blocks, preserving
two OFF/two ON replicates per selected block, percentile2.5/97.5. The cohort/
block is the unit, not120 independent TTFT observations. A primary observed win
requires ratio<1 and CI upper<1; the calculator still returns qualificationfalse
and awaits independently bound evidence. It reports a loss without selecting a
new primary. ADR-063 additionally requires CI half-width<=2.5 percentage points;
the calculator reports improvement point/CI/half-width, precision-target status
and a >=5% stock point flag. A fixed-plan precision miss is inconclusive, with
no ad hoc extension. The SSE composite is a COMPONENT latency study, not the
final N4 makespan/goodput gate against the latest best peer plus stock gap.
`peer_comparison_missing=true` and `campaign_primary_acceptance=false` remain
explicit even when component latency improves by >=5%. Secondary wall is the existing coordinator elapsed_seconds,
including client/barrier/tools/cleanup but excluding server startup; it is not
pure GPU/server compute time. Report individual turn/agent timings, cached
counts, outputs and wall regressions as well.

`gemma_matched_stats.PLAN`, `cohort_score`, `exact_vectors` and `analyze` implement
these exact pure boundaries. The later evidence adapter must bind raw source,
quality, production vectors and receipts before using the calculation; booleans
passed into pure math are not independent evidence. No new hardware runner or
qualification authority is introduced.

## Validation status and next bounded slice

Source stage only; author executed no imports, tests, builds, models or tools.
Focused synthetic parser/control fixtures are distinct from real capability
proof. Root can run the two focused test modules plus the retained canonical
client controls. Tests cover opt-in refusal, real-output provenance boundaries,
nonzero/missing/changed cleanup/artifact/test data, invocation-before-derivation,
raw preservation, fixed plan, exact vector mismatch, SSE-only metrics and fixed
block resampling. None proves actual tooling or serving by itself.

Before models: independently review this source; execute focused/retained pure
controls and a model-free actual `BenchmarkTools` compile/test control using a
trusted fixed valid patch; freeze the final study controller, production-token
exporter/evidence adapter, thermal/observer policy and exact sources. Then run
only the approved fixed campaign with every failure retained. Default activation,
all-four model wins, matched peer scoreboard and fullQ5 remain outstanding.

The next boundary is documented in [the study boundary design](adr063-gemma-matched-study-boundary.md). The neutral cfg(test) exporter is separate from the ON familyV2 prefix oracle; all offline work is scheduled only after all20 cohorts and verified unload. Its source identity checks intentionally require root composition with reviewed9b/181 before use.
