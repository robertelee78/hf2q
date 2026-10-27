# ADR063 Q5: final qualification matrix and remaining proof

Status: reviewable evidence draft, not Q5 acceptance. Source audit at hf2q
`a84e5c7bbfe93f0ffed4ebecfa3dc0ec844fa70b`, 2026-10-27. No models, builds,
probes or tests were executed for this audit. Issue #371's current acceptance
matches ADR063; its retained failures remain failures.

## Exact obligation

[ADR063 Q5](../adr/ADR-063-apple-silicon-four-agent-speed-coherence.md#q5-qe-prove-four-slot-memory-safety-on-m5-max-128-gb)
requires every **accepted configuration** to fit the operator host with measured
headroom and no deliberately induced OOM. Its acceptance requires:

1. A pre-load disjoint ledger covering model residency, four-slot KV/recurrent
   state, committed plus pending anchors, drafter state and peak in-flight scratch.
2. A frozen dedicated-host available-RAM policy, observed competing
   nonreclaimable use, observation/stop latency and maximum allocation overshoot;
   obey physical capacity and any source-justified configuration limits.
3. Sequential stock/experimental/eligible-peer execution, verified unload,
   listener/process and memory recovery; exactly one hf2q process, including
   metadata commands, and no DeepSeek co-residency.
4. Synthetic allocation/admission negative controls and bounded real four-slot
   stress at the largest declared tool-result/context cases, stopping before OOM.
5. Logical budgets, OS footprint/pressure, Metal residency, transient high-water,
   swap/compression deltas and cleanup receipts; no silent reduction of context,
   task or output budgets. Independent QE reviews ownership and arithmetic.

The approved operator update `8609fe3dbdc4ef69173f37313af4800604ca2b64`
explicitly supersedes the old fixed 12 GiB reservation / 116 GiB cap. Historical
private reviews and the October-23 ceiling inventory containing that old
inequality are not current admission authority. Older ADR042 H3 is historical
acceptance evidence, not permission to reintroduce an unexplained fixed Q5 cap.
Any genuinely source-justified remaining limit must be identified by scope;
KV/component limits and physical capacity still apply. This is not an approval
to ignore the current packet's prospectively frozen operating thresholds.

Q5 does **not** require a mathematical certificate of every macOS allocator,
page or driver object, a new allocator architecture, mandatory global-root
adoption, mandatory same-file overlap credit, or an optimization speed win.
It does require credible source-bounded known payload/coexistence and measured
operating margins. Known GiB arrays cannot be hidden in an opaque allowance;
positive driver/retirement/metadata uncertainty may be a declared conservative
operating margin. `Unavailable` measurements remain unavailable, never zero or
an observed physical bound. Root-completion, GPU completion, wrapper drop and
physical reclamation are different events.

## What “largest” and 8192 mean here

There is no literal `253952` or requirement to emit exactly 8192 tokens in Q5.
The currently declared profile is four independent contexts of 262144, with
8192 completion allowance. The maximum admitted input is family/source-specific, not simply C minus
output. Gemma's reviewed case uses253952. Qwen35 generation admission and
Qwen35PrefillState::begin require P + graft + max_tokens +64 <= cache capacity
(`engine.rs::validate_qwen35_generation_request`, `engine_qwen35.rs:4023–4037`).
Thus Qwen3.8/Qwen3.6 text-only ungrafted maximum is253888, preserving full
C262144 and output8192 plus the existing 64-token slack. DeepSeek must likewise
use its actual admission predicate; do not impose the Gemma number by analogy.
The exact artifact's tokenizer/renderer must produce the family's source-admitted count. Metadata-only
construction is not runtime stress; four merely configured slots are not four
executed large requests.

A successful natural early EOS with the full allowance retained does not
silently reduce the output budget. It also does not prove highest-index writes
or the complete 8192-token decode history. Q5 closure must cover that permitted
peak through source bounds and applicable allocation/growth/boundary controls,
plus the bounded real stress. Add an actual long-output case only where a
specific uncovered allocation/lifetime/high-water depends on executing it;
do not invent a universal exactly-8192-output requirement. Conversely, do not
assert full-cache/decode-boundary qualification from a short natural reply.

The tool-result maximum is the **frozen workload** maximum, not an arbitrary
new number. The canonical tools permit 64 KiB file/results and bounded complete
wires. Existing small parse-port results do not exercise that ceiling. Freeze
an actual largest result whose serialized next request still obeys the real
wire/body/context limits; escaping/metadata expansion is charged. Do not reduce
the matched tool budget or silently replace the real coding quality gate.

## Finite twelve-cell matrix

Each row is a configuration/evidence cell, not an order to rerun an identical
qualified receipt. A final source/binary/artifact/settings freeze must name the
stock control explicitly. Stage-OFF / pool-OFF receipts are component controls;
they are not automatically the official published crate or the final F1 stock
baseline. Broken historical stock must not become a favorable quality/speed
denominator. New memory-affecting modes, drafter, source or native revisions
require an affected-cell delta review, not relabeling old observations.

| Family | Arm | Existing reusable evidence | Exact unresolved qualification |
| --- | --- | --- | --- |
| Qwen3.8 | source-built stock/control | Fixed20 pool study OFF cohorts, normal four-agent real coding/tool/cleanup receipts; exact source/config retained | Final stock identity; largest-context and largest-tool-result stress plus closed source/telemetry joins, or independently justified coverage by an identical/dominating accepted configuration |
| Qwen3.8 | experiment | Pool ON coding cohorts; `qwen38-worker-census-20261026T141046Z` and `qwen38-n4-decode-profile-cb-20261026T233932Z` establish normal N4/tools, actual worker pool census and cleanup; new largest-admitted attempt150727 is IN PROGRESS, not PASS | Same largest cases; census excludes cache/direct/other TLS/physical high-water, so whole-process source union and missing telemetry still need joining; synchronous profile is diagnostic only |
| Qwen3.8 | eligible peer | Existing source/build/hash-bound d854 GGUF peer and explicit Qwen adapter; no full accepted memory stress identified | Freeze eligible reference source/mode and dynamic-library identities; full four-context load/readiness, largest context/tool result, same resource policy/recovery and independent budget review |
| Qwen3.6 | source-built stock/control | `qwen36-largest-admitted-prompt-n4-20261027T124629Z` independently PASSED all four P253888/full 8192-allowance requests and scoped unload recovery; older173831 coding E0282 remains FAILED | Largest-admitted input exposure is scoped PASS, not highest-index8192 writes, coding quality, tool-maximum or every-arm proof. Finish largest-tool-result, applicable decode/growth boundary/source joins and final stock identity; retain original124030 HTTP400 failure |
| Qwen3.6 | experiment | Shared Qwen source plans and pool mechanisms, model-specific source geometry; normal run above is partial failure | Freeze actual experimental configuration (may share control bytes if no memory delta), qualify its largest cases and source/telemetry union; do not substitute dense-Q38 dimensions |
| Qwen3.6 | eligible peer | d854 native qwen35moe source and binary; peer thinking-transport parity source prepared, current reference freshness recorded | Metadata-only reserve sizing exists, not runtime qualification. Freeze the exact eligible peer reasoning/template/mode, close its own source envelope, then four-slot same-artifact largest cases and recovery; no matched peer quality/runtime result exists yet |
| DeepSeek4 | source-built stock/control | Existing canonical cancellation/long-context receipts may cover their exact source/shape; compact image/cache plans are reusable source proof | Freeze current compatible stock identity; prior 125k growth/cancellation is not its largest source-admitted context case. Complete maximum-context/tool-result and source transition/CPU image/in-flight joins |
| DeepSeek4 | experiment | `deepseek-new-grammar-n4-20261026T222718Z` normal 28 requests/24 real tools/cleanup; output-head diagnostic separately verified | Largest cases and case/phase operating refinement: reviewed coarse source envelope126345754476B includes mapping/grouped inputs/current tiled history/direct-head/growth and the missing32KiB tile-map correction. Physical fit remains unqualified; no invented global cap or history-as-peak substitution |
| DeepSeek4 | eligible peer | Native GGUF peer source capability is a candidate; absence of a DeepSeek client adapter is not proof the engine is unavailable | Confirm actual artifact/loader/template applicability and eligible mode; build bounded peer stress/receipt adapter, then four-slot largest cases and same-policy recovery. Do not route through Qwen metadata or assume an absent drafter |
| Gemma4 | source-built stock/control | Fixed20 matched study OFF cohorts: normal N4 quality, tools, exact rendered input comparability and cleanup | Largest-context/tool-result stress and memory record. OFF has its original anchor semantics; no need to demand ON checkpoint proof of OFF |
| Gemma4 | experiment | Fixed20 ON coding cohorts and family seven-wire replay PASS; bounded component/CPU/real-GGUF controls | `gemma-largest-prompt-n4-20261027T005710Z` **FAILED** 3600s deadline at 239616/253952 in all four slots. Complete the largest case under a newly frozen prospective plan or implement a source-justified performance fix; preserve failed attempt, no rescore. Largest tool case still missing |
| Gemma4 | eligible peer | Native GGUF peer source candidate; no accepted largest four-slot receipt identified | Freeze actual eligible same-artifact source/template/mode and memory envelope, four full contexts and largest context/tool cases; eligibility must be established, not guessed from enum lists |

The d854 reference is not “latest best” merely because it is built. Read-only
GitHub origin resolution at 2026-10-27T01:37Z observed upstream
`7aca0d2ea8d92b56911233b822db01f9fbb1ae50`; no corresponding runtime result is
implied. F1 resolves/fixes reference candidates; Q5 can qualify a finite
source-derived envelope covering those candidates before F2 selection. It need
not first run a performance selection that itself waits for Q5. A later selected
mode outside that envelope reopens its affected memory cell.


## New scoped Qwen3.6 largest-input result

`qwen36-largest-admitted-prompt-n4-20261027T124629Z` independently passed its
corrected four-slot largest-admitted input: source
`2d7599907be92d17736bc205806907589e845306`, public mlx-native 0.15.5, binary SHA256
`111649ce1b0d635d0a1403f34004667e1ab61757ff203c68c513dcac912488c2`.
All four requests used exact actual-rendered P=253888, full 8192 allowed output,
and the existing 64-token admission slack. Two unary/two SSE returned HTTP200,
stop and exact CAPACITY_i; all four server slots reached253888. Completions
261/372/467/275 show natural early EOS, not full 8192 decode/highest writes.
Client exit0 after2674.748s; original77676 artifact entries independently hashed
with zero mismatches, plus149 post-unload recovery files. Owned groups were
reaped/empty, listener absent, scoped five-second operating recovery observed.

Minimum sampled operating Free+CachedFiles67.878GiB and peak raw footprint
37.258GiB stayed inside that packet's guards; these overlapping OS metrics do
not certify unique physical pages or Metal high-water. Largest tool-result,
coding E0282 recovery, seven-wire prefix and remaining arm obligations are open.
The earlier124030 P253952 immediateHTTP400 failure is preserved, not rescored.
Independent review basenames/digests are recorded in the
[operating ledger](adr063-q5-selected-profile-operating-ledgers.md).

Q38 `qwen38-largest-admitted-prompt-n4-20261027T150727Z` remains IN PROGRESS;
no final pass or cleanup is inferred. DeepSeek's finite coarse source envelope
is not yet an admitted operating packet. Two separate source-reviewed memory
spikes are UNEXECUTED: omit unused Q8/R>8 output-A input staging (local256MiB
atR2048/64MiB atR512), and shrink internal gather tile256→64 (admitted widest
bucket1GiB→256MiB). Neither is a speed/whole-pool/physical-fit claim or a changed
accepted mode; actual controls and real-model qualification remain pending.

## Reuse without overclaiming

Already reusable, with exact source/shape/pin bindings:

- Pure ledger schema/checked arithmetic/unknown rejection and ownership negative
  controls; no claim that pure ledger is wired runtime admission.
- Published native mapped-backing lifetime, root-authority/pool-history and
  bounded allocator controls, where the selected HF path actually uses them.
  Standalone controls do not become whole-process allocator coverage.
- Qwen cache/capture/FA/indexer checked plans; model-specific weight/load and
  CPU-anchor/source coexistence inventories; DS exact image sizing/bootstrap
  reuse; Gemma conservative weight/cache/scratch union and bounded anchor quota.
  Recheck changed callsites, conditional branches and selected defaults; do not
  mechanically sum overlapping historical packets.
- Observer/parser malformed-input and bounded retry controls; typed process
  telemetry capability; owned cleanup and sandbox controls; actual loaded-run
  samples; no need to rerun unrelated component controls merely to fill a table.
- Actual normal four-agent requests, tool/compiler results, source patches,
  logical settings, swap/compression, footprint and cleanup from the receipts
  named above. These support their exact achieved shapes, not unexecuted maxima.

The Gemma failed largest attempt is useful negative/runtime evidence: 6215
sampled global observations, all normal pressure, no sampled swap growth;
minimum operating available memory about47.93GiB; recorded peak raw footprint
about44.28GiB; verified owned cleanup. Its direct abort was the request deadline,
not a recorded memory-guard/OOM event. This is not a largest-context PASS and
cannot be extrapolated to the remaining14336 rows or accepted from a partial
progress log.

## Smallest remaining closure sequence

1. **One selected-profile ledger packet per family, linked to all three arms.**
   Assemble existing checked plans/source inventories into a reviewed disjoint
   peak table: weights/map versus copied payload, four caches/recurrent state,
   bootstrap, old+new growth/capture, committed/pending/constructing/rollback
   anchors, any enabled persistence/drafter, worker/TLS retained pools, direct
   scratch, command/graph roots, bounded CPU/request/result/log/I/O owners.
   Name aliases once, independent copies separately. Add explicit positive
   opaque/retirement/observation allowances; never turn unknown known arrays
   into zero. Source-frozen conservative envelopes are valid; extracting every
   constructor into another API is not mandatory. Root-history totals and live
   physical peaks are different and must not be equated.
2. **Close the existing operational evidence joins.** Freeze observed available
   RAM, competing nonreclaimable interpretation, reaction/overshoot margin,
   pressure/footprint/swap/compressor limits and recovery condition. Reuse
   existing sampler/cleanup controls; add only a missing deciding negative
   control for stale/error/breach-stop and its bounded reaction. A nominal
   unloaded sampler calibration alone is not measured loaded stop latency.
   Retain loaded raw timestamps and the largest uncancellable allocation/queued
   transaction overshoot bound. No adversarial whole-OS theorem is required.
3. **Make the telemetry scope complete and honest.** Record actual supported
   Metal allocation/residency observations and requested transient high-water
   or conservative peak where available. The worker census's retained union is
   not whole-model Metal residency; raw phys_footprint is not unique ownership.
   Existing `Unavailable` domains need a narrowly scoped supported observation
   or an explicit independent QE disposition supported by the source upper and
   margin. Merely writing `Unavailable` does not satisfy measured residency or
   turn G into a physical certificate. Do not demand a new allocator framework
   when an existing device/worker diagnostic and source bound supplies the data.
4. **Freeze the minimum missing real cases before execution.** Per accepted
   family/arm, one largest-context four-slot case and one largest-tool-result
   continuation case, or one combined case covering both with all real wire
   limits intact. Reuse exact already-passing shape-equivalent receipts; reuse
   cross-arm coverage only with an independently reviewed ownership/peak
   domination proof and separately verified lifecycle/source identity. Do not
   assume default-OFF dominates ON memory. Declare finite deadlines from the
   workload, recovery checks and no retries. The failed Gemma3600s packet cannot
   be extended in place; any longer prospective packet is a new attempt with
   its own independent rationale and preserved failure, not a favorable rescore.
5. **One evidence PR closes the matrix.** Record each cell PASS, FAIL or explicit
   ineligible-with-proof, precise source/pin/artifact/settings and raw receipts;
   independently audit the ledger/stress/recovery and merge the evidence or
   qualified implementation. An untested eligible peer is missing, not
   ineligible. The delivery boundary explicitly permits measured evidence;
   no new production admission architecture is mandatory solely for this PR.

Q5 cannot presently be marked Done: every-family largest-context/tool-result
coverage, the final disjoint selected-profile joins and scoped telemetry /
reaction evidence are not complete. Q5 also need not wait for coding speed
wins, a default promotion or peer-best timing proof: those remain their own
correctness/F2/Q3/Q4 gates. Any partial/conditional memory qualification must
name its accepted configuration and must not waive those gates.

Root reported a subsequent Qwen3.6 P253952 fixture received four immediate
HTTP400 capacity refusals (attempt ending124030); it did not prefill or prove
a memory failure. Its raw evidence remains a failed attempt. The separate
corrected124629 case now has a scoped independent PASS. Correcting that fixture to the existing P253888
admission ceiling is not lowering context/output or removing the64-token guard;
it is correcting an invalid harness input. Do not rescore the failed fixture.
