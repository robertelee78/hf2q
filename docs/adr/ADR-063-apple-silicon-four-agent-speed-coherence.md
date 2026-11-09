# ADR-063: Apple Silicon four-agent GGUF speed and coherence

- **Status:** Accepted program scope, F1/F3 execution entry, the separately reviewed F2/Q1/Q2/C1 source implementation entry and the 2026-11-03 measurement-only entry (see "Speed program, Q1 successors and release mapping"); other implementation designs require separate approval.
- **Date:** 2026-10-22
- **Updated:** 2026-11-05 (Q4-QWEN36 workload) — the operator's repair-turn workload (2026-11-03) is adopted for Qwen3.6: on fixed main (mlx-native 0.15.18 `rope_multi` partial-rotary pairing, #686 tool-call newline) the paired task campaign is hf2q 21/24 vs peer 20/24, `Pass`, so Qwen3.6 has a declared workload and its Q4 speed race runs on it (Q4-QWEN36). Earlier: 2026-11-05 (Q4-GEMMA-LMHEAD-MMA opened) — the operator opened a Class B few-row MMA lm-head story under Q4-GEMMA-DECODE design §13 decision 2(b) (`docs/research/adr063-q4-gemma-lmhead-mma-design.md`); it is a predecessor of the Gemma closing campaign. Earlier: 2026-11-05 (Q4-GEMMA-ATTN-EXACT outcome) — S0 parity passed on both metallib paths. The best exact configuration (G = 2, R = 1) has a chain ratio of 0.761 against the frozen 0.70 go bar, so the story as designed stops. The operator adopted the narrowed follow-up Q4-GEMMA-ATTN-EXACT-G2 (`docs/research/adr063-q4-gemma-attn-exact-design.md` §12–§13). Earlier: 2026-11-04 (Gemma 4 gap stories) — operator decision adding Q4-GEMMA-COLD, Q4-GEMMA-PREFILL-MOE, Q4-GEMMA-DECODE and Q4-GEMMA-SPEC-SERVE, with their designs approved (`docs/research/adr063-q4-gemma-{cold,prefill-moe,decode,spec-serve}-design.md`) and their recorded decisions; decode attention is lever A of Q4-GEMMA-DECODE (root decision); the four are predecessors of Q4-GEMMA-ATTN's closing combined candidate; the revised Gemma lever budget is an estimate of about 2.2–4.0 s against the 3.0 s a Win needs; the mlx-native `rope_multi` partial-rotary defect (fix mlx-native PR #86, merged as `dc236dc1`, release 0.15.18 not yet published): all Qwen GPU validation, the ADR-065 B1 baseline run on main and the Q1 spec b6 run wait until 0.15.18 is published and pinned `=0.15.18`, ADR-065's J1/P4b pin moves to 0.15.18, and Q1-QWEN38-RERUN's delivered result stands (see "Gemma 4 gap stories and the rope_multi pin"). Earlier 2026-11-04 (Splash peer) — for Qwen3.8 and Qwen3.6 a Q4 Win requires hf2q to finish the four-agent task at least 5% sooner than both the peer and Splash, each in its best valid configuration; Splash `031a25a6` joins as a second same-artifact peer of record on the hf2q GGUFs (see "Splash peer of record for the Qwen families"; design `docs/research/adr063-splash-peer-design.md`, decisions D1–D10). Earlier 2026-11-03 (design approvals) — operator approval of the designs for Q4-GEMMA-KERNEL/Q4-GEMMA-ATTN (`docs/research/adr063-gemma-tiled-attn-design.md`), Q4-TRACE (`docs/research/adr063-q4-trace-design.md`, with its three §7 answers), Q1-QWEN36-SPEC/Q1-GEMMA-SPEC (`docs/research/adr063-q1-spec-design.md`; the unlocalized-`RouteSensitive` exception is rejected, Gemma results must be localized and #665 includes the verifier-route localizer) and Q4-DS4-RACE with ADR-066 N-IQ2XXS (`docs/research/adr066-iq2xxs-subset-design.md`); Q4-QWEN36: the conversion audit found no defect, and the first third-party check is unsloth UD-Q4_K_M under the canonical setup on both engines (see "Design approvals and Qwen3.6 decisions"). Earlier 2026-11-03 (C2 sign-offs) — owner sign-off of C2 (#360) decisions D1–D5 (`docs/research/adr063-c2-design.md` §10): divergent turns are judged by the Q1 decision envelope on every teacher-forced row of each divergent turn, `RouteSensitive` blocks; append-only turns keep `ExactTranscripts` plus the bitwise same-route twin; "≤ 1.025" limits are 95% CI upper bounds at Q3's sequential precision stop; campaigns use that stop instead of fixed blocks; anchor identity omits C1's `conversation` field; cancellation discards the request's own pending anchor. Earlier 2026-11-03 (later) — Q4-DS4 result Fit (six of seven peer runs valid; best configuration provisional); Q4-DS4-RACE now consumes ADR-066 N-IQ2XXS (the IQ2_XXS subset as its own story) instead of N1/N3/N4; Q4-DS4-RACE decision D1: on the ds4 file the race is judged against the peer only, and the vs-stock makespan contrast, fairness band and stock quality gate come from a same-artifact candidate-vs-v0.1.28 record on hf2q's own DeepSeek-V4 artifact with the same candidate binary; Q4-DS4-RACE decision D2: ds4 runs on the same file in a declared same-artifact native-comparison campaign (design `docs/research/adr066-iq2xxs-subset-design.md`). Earlier 2026-11-03 — operator decisions for the v0.1.30 speed program (see "Speed program, Q1 successors and release mapping"): Q4 rewritten as the "clearly faster than the peer" umbrella, closed by the new per-family stories Q4-QWEN38, Q4-GEMMA with Q4-GEMMA-ATTN, Q4-DS4 with Q4-DS4-RACE, and Q4-QWEN36, with Q4-GEMMA-KERNEL (mlx-native) and Q4-TRACE; the measurement tracks Q4-GEMMA (profiling) and Q4-DS4 (fit) are split from their implementation tracks and their designs are operator-approved; Q4-DS4-RACE consumes the IQ2_XXS subset of ADR-066 N1/N3/N4 pulled ahead of ADR-066 B2; Q1 successors Q1-DS4, Q1-GEMMA-T2LOC, Q1-QWEN38-RERUN, Q1-QWEN36-SPEC and Q1-GEMMA-SPEC added; C2 amended (built regardless of measured demand, on ADR-065 P4a/P4b with J1 joining v0.1.30, frozen limits, delivery v0.1.30); remaining verification leftovers recorded as tracked successor scope, v0.1.31. 2026-11-01 — F2 closes on option B (per-family Q3 verdicts of record with controls, quality gate and fairness; see "F2 closure scope"). 2026-10-30 — Q5 rewritten as per-family calibrated admission with live out-of-memory stops; F2 and Q1 are no longer closure gates, so stories close on per-family, independently reviewed evidence; Q3/Q4 accept per-family peer wins where a peer arm exists; C2 moves to v0.1.29. Earlier 2026-10-27 parallel-source update unchanged.
- **Deciders:** Robert E. Lee (product owner)
- **Tags:** apple-silicon, m5-max, gguf, four-slot, speed, coherence, qe, v0.1.28, v0.1.30
- Stable source ID: ADR-063
- Audited baseline: hf2q `5d1ffe7f1c5cf4ba5ffca452ec4bde738231beef` / v0.1.27.
- Research cutoff: 2026-10-22; [research evidence and sources](../research/apple-silicon-speed-parity-2026-10-22.md).

## Context

### Intent and authority

The product goal is leadership in GGUF inference on Apple Silicon. The primary outcome is four concurrently active agentic coding slots on the operator M5 Max with 128 GB RAM, with coherence/tool correctness at least as good as peers and proven four-agent speed wins for each of Qwen3.8, DeepSeek-V4, Qwen3.6 and Gemma 4. Qwen3.8 and DeepSeek-V4 have first priority; Qwen3.6 and Gemma 4 follow. Current primary implementations and October releases determine the experiment shortlist; old ADRs supply constraints and failed evidence, not an automatic implementation queue.

This proposal consolidates the strongest ideas from ADR-010/011/016/019/034/037/039/040/042/044/049 and newer October work. It does not silently supersede their accepted safety contracts. The release target is v0.1.28; the prior milestone-6 producer scope moves to v0.1.29. The operator approved this program scope and the F1/F3 execution entry on 2026-10-23 after the recorded skeptic review. At that initial entry F1/F3 could become Ready after their approval was recorded; other stories required their own design approval and predecessor evidence. The later approved source-overlap entry below records the limited F2/Q1/Q2/C1 exception. Creating a project does not claim an experiment passed. Do not duplicate existing managed issues; source corrections must reconcile their existing bindings. A feature story is accepted only on its own observable contract; cross-family milestones do not establish universal coverage.

The primary comparison is four simultaneously runnable agent slots with at least six tool rounds each; one-slot controls and mixed prefill/decode populations 3+1, 2+2 and 1+3 are required. Stock current hf2q, experimental hf2q from that baseline, and latest peer are mandatory arms, frozen for each experiment. Same-artifact peers use hf2q-produced GGUFs. MLX/native-format comparisons are separate quality-qualified arms. The mandatory hardware is M5 Max with 128 GB RAM; other Apple generations are future separate strata. Linux/CUDA runtime support is outside scope. Ideas may originate anywhere but win only through owned Rust/Metal GGUF measurements on this host. Top-tier artifacts are vanilla Qwen3.8-27B dense and DeepSeek-V4. Qwen3.6-35B-A3B MoE and Gemma 4 are the second tier; prioritize the top-tier speed work, then prove speed wins for the second tier as well. Four slots do not mean four Macs. Quality gates and sampling transformations cannot be relaxed to manufacture a speed win.

## Decision

### Global acceptance and exclusions

The program MUST preserve Rust-owned conversion, quantization and inference, OpenAI unary/SSE and native tool semantics, exact supported cache/verification invariants, conversation isolation and operator resource ceilings. It MUST bind comparisons to source, registry dependencies, binaries, model/draft hashes, exact input/settings, hardware and raw receipts. No 90GiB peer/hf2q co-residency on the 128GiB host; consult release coordination before heavy work and stop every test server.

The program MUST NOT revive falsified router-only precision fixes, reimplement already shipped tiled/block-skip attention, presume HASS/Hydra stacked gains, introduce approximate recurrent or nonprefix reuse into exact serving, use Python/vendor production conversion, substitute prequantized targets, or equate one-slot headline throughput with four-agent success. ANE encoder offload, distributed Macs and custom training are excluded from this delivery; they may reopen only with a separate current evidence-backed design. The July BaseRT paper is historical context; October 22 BaseRT 0.4.0 and current peer kernels are the recency baseline.

Conditional work means its deciding experiment must record implement, falsify, or successor-design with evidence. It is never permission to land a stub or promise an unmeasured speedup. Experiment receipts and implementation candidates use merge delivery; Q4 alone closes the product release boundary; Q2 certifies the harness and Q3 independently proves the measured effects. A required runtime change needs a published native revision and clean registry-resolving proof. The final qualification depends on the actual winning subset, declared before candidate acceptance; it need not wait for unrelated or falsified hypotheses.


## Repository ownership and publication

This is one performance program spanning two owned repositories. Every Metal compute kernel and its Rust dispatcher belongs in `mlx-native` (`src/ops/`, `src/shaders/`); hf2q consumes that public API. `mlx-native` also owns buffer/resource safety, operation capability routing, graph/encoder submission mechanisms and primitive parity/microbenchmarks. hf2q owns conversion and quantization policy, model-family graphs, request scheduling, cache/anchor/cohort policy, tool semantics and model/workload acceptance. A family-shaped native primitive still needs an explicit reusable operation contract; model behavior does not move into the kernel library.

For every winning change that modifies `mlx-native`, record separate immutable native and hf2q commits and PRs. Complete native correctness/performance review and merge, publish a new native version through its release workflow, verify the crates.io archive checksum and downloaded-archive tests, then pin that exact registry version/checksum in hf2q `Cargo.toml`/`Cargo.lock` and its dependency-provenance verifier. Rebuild hf2q from a clean registry-resolving tree and qualify the combined configuration. A local path patch is spike evidence only.

Q4 MUST publish both updated packages when native changes are included: the required `mlx-native` crate/version/tag first, followed by the qualified hf2q crate of v0.1.30 (or the first release containing each family's accepted configuration) and signed/notarized standalone release artifacts. Verify public package/asset bytes, dependency provenance and clean installs through each repository's existing release gates. Native publication alone does not deliver the hf2q integration; hf2q publication against an unpublished native revision is forbidden. If the winning subset does not change native code, explicitly record the already-published native version and verify it rather than inventing a native release.

Keep this document's existing work keys as the program's acceptance ledger. For any native implementation, link a separately scoped `mlx-native` issue/PR with its primitive contract and tests to the owning story before implementation. Do not duplicate kernels or turn hf2q tickets into permission to edit native policy without a contract. Native implementation delivery is merged and published native code; the hf2q story's own boundary additionally requires pinned integration and its declared end-to-end evidence. One integration owner reconciles the exact combined configuration.

### Source map for the ownership split

Source inspected on 2026-10-23: hf2q baseline `5d1ffe7f1c5cf4ba5ffca452ec4bde738231beef`, mlx-native `f92eb02` (published version 0.15.1). Revalidate symbols at each candidate commit; this map is navigation, not performance proof.

| Concern | hf2q consumer/policy | mlx-native mechanism and evidence |
| --- | --- | --- |
| Dense and expert projections | `src/inference/models/qwen35/gpu_delta_net.rs`, `execution_dispatch/evidence.rs`, `src/inference/expert_dispatch.rs` | `src/ops/quantized_matmul_ggml.rs`, `quantized_matmul_id_ggml.rs`, matching `src/shaders/`; GGML routing and resolved-dispatch trace tests |
| Physical batching | `src/inference/models/gemma4/batched_body.rs`, family forward graphs and serve scheduler | Existing `quantized_matmul_ggml_broadcast_batched_mm` and perm021 broadcast APIs; do not assume shared-weight batching is absent |
| DeepSeek transactions | `src/inference/models/deepseek4/verifier_forward.rs`, family cache/sidecar state and engine cohort commit | `src/ops/deepseek_sparse_attention.rs`, shader and primitive parity/benchmark tests; encoder/fence contracts |
| Resource/command lifetime | Family-owned cache objects, graph construction and request admission | `src/device.rs`, `buffer.rs`, `encoder.rs`, `encoder_session.rs`, `graph.rs`; command-buffer/autorelease and session lifecycle tests |
| Publication | `.github/workflows/release.yml`, `scripts/verify_release_dependency_provenance.sh` | Native `.github/workflows/release.yml`: exact source/archive tests, registry byte verification and downloaded-crate tests |

## F. Evidence and current peers

### F1. Freeze October 2026 peer, target and hardware identities

Outcome: A reproducible manifest freezes current stock hf2q, experimental hf2q and latest peer, with separately labeled historical and Apple-specific comparators.

Requirement: The program MUST deliver a reproducible manifest freezes current stock hf2q, experimental hf2q and latest peer, with separately labeled historical and Apple-specific comparators.

Acceptance:

- Explain and resolve the three observed peer identities without silently changing the accepted product pin or modifying /opt/llama.cpp.

- Freeze source/build/binary hashes, target and draft artifact hashes, native registry pin, hardware/OS/Metal versions, templates and rendered tokens; date the research screening to 2026-10-22, but resolve the mandatory latest-peer runtime revision at campaign start rather than freezing it to the research cutoff.

- Include current peer and screen BaseRT 0.4.0, MLX 0.32.4/current main and Kapitsa for supported four-slot tool semantics; unsupported or unavailable arms are reported, never invented.

- Use identical hf2q-produced GGUFs for same-artifact lanes; native-format comparisons require separately labeled quantization and independent quality evidence.

- Resolve latest upstream peer at the beginning of each campaign and freeze it through that campaign; the historical pin is provenance, not a substitute for the latest-peer primary arm.

- Freeze official vanilla Qwen3.8-27B dense and DeepSeek-V4 as top-tier targets on the operator M5 Max with 128 GB RAM; Qwen3.6-35B-A3B MoE and Gemma 4 are second-tier targets with explicit artifact identities and regression coverage. Freeze exact HF revisions and hf2q conversion provenance, not an abliterated substitute.

- The default peer executable is absent and its build-info metadata is stale b9278/5306f4b3b; build an isolated current peer and verify its own --version plus binary hash. Source HEAD alone is not executable identity.

- The historical e15384a5 peer hash is unavailable locally and upstream; preserve historical receipts but propose a reachable validated replacement through the governing pin workflow.

- Declare peer matched-settings and peer-best-documented configurations. The mandatory latest-peer performance arm uses peer-best valid settings; speculation cannot be silently disabled to handicap it. Retain MTP tensors in the shared GGUF if any arm uses them and pin separate drafter weights.


Repository ownership: hf2q owns the campaign identity manifest; record both repository commits, the published mlx-native version/checksum, downstream hf2q binary/package identity and native release receipts. Do not alter kernels or release either package in this inventory task.

Validation: Validate an immutable comparison manifest against source, binary version output, hashes and capability probes; no model load before coordination and memory preflight.

Predecessors: none.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

### F2. Establish the four-agent coding quality and latency scoreboard

Outcome: Four genuinely concurrent agent conversations have reproducible quality, fairness and speed baselines.

Requirement: The program MUST deliver four genuinely concurrent agent conversations have reproducible quality, fairness and speed baselines.

Acceptance:

- Use one-slot diagnostic controls and four simultaneously runnable client slots with at least six tool rounds each, independent conversations, simultaneous cold starts, staggered arrivals, warm append turns, divergent edits, equality hits, cancellation/retry and request-order permutations.

- Freeze independent tool-name/argument and coding-task correctness oracles; unary/SSE, reasoning completion, tool-result continuation and isolation must pass. Peer wording need not match.

- Report semantic TTFT p50/p95/max per slot, maximum semantic SSE gap, four-agent makespan, successful-task goodput, new/cached tokens, prefill/decode rates, queue/restore/draft/verify time, peak RSS and thermal state.

- Use warmup and at least five balanced paired blocks, then a predeclared 95% confidence-interval half-width target of 2.5 percentage points for the primary effect, with a maximum of 20 blocks per primary comparison; unresolved results are inconclusive, never a win.

- Measure mixed prefill/decode populations 3+1, 2+2 and 1+3, plus pure four-prefill and four-decode controls; trace active/runnable slots and actual physical cohort membership. Overlapping queued HTTP requests alone do not prove concurrency.

  For prospective F2/Q2 phase controls, freeze `forward_demand_v2`: count a current admitted owner only when source-proven ready for forward work, or joined to its actual native submitted-outstanding forward obligation with eventual healthy checked completion. Report ready-now occupancy separately. Submit-to-checked-return is a host service bracket, not GPU Running time or utilization. HTTP overlap, unknown membership, emit-only readiness and repeated inflight heartbeats do not establish demand. Keep the first fixed two-second window, its end within thirty seconds, eighty-percent target coverage and new per-owner progress unchanged; do not rescore earlier ready-only cases under this policy. Physical membership remains the actual submitted owner set, including truthful singletons, rather than a union manufactured into B4.

- The mandatory primary arms are stock current hf2q, experiment from the same baseline, and latest peer, frozen per run; other engines and the historical peer are separately labeled secondary comparisons.

- Stage the matrix on the same M5 Max/128 GB host: vanilla Qwen3.8-27B dense, then DeepSeek-V4, then second-tier Qwen3.6 MoE and Gemma 4. Each needs N1 controls and N4 coding/tool proof. Use representative primary workloads first; expand only where needed to decide a candidate.

- Canonical transcript replay supplies identical rendered inputs for controlled timing and fairness; record the exact cache priming/reset policy and its cost, and never claim natural prefix reuse from a forced transcript. A separate own-output continuation lane scores real coding/tool success, natural reuse and task-level completion from the same initial tasks; its differing token streams are not token-matched timing evidence.

- Greedy is the internal exactness lane. Capture and freeze actual production-client sampling for the primary agentic quality lane; do not assume clients use temperature zero.

- Define scheduler-specific phase labels for Qwen and DeepSeek; stagger long tool-result prefills to hold 3+1, 2+2 and 1+3 populations over measured intervals and report time-weighted observed composition.

- Extend test_full_context_agent_slots.sh, bench_qwen36_agentic_cache.sh, qwen38_speculation_ab.sh and test_agentic_cache_lifecycle.sh where their contracts apply; add only the missing task oracles, phase tracing and independent three-arm receipts.

- Before every campaign read release coordination and reject host contention. Freeze thermal/cooldown protocol in the harness: at least 60 seconds between engines, Nominal thermal start and sustained Fair-or-better sampling; wait longer if needed. Never overlap full-model runtimes.


Repository ownership: hf2q owns clients, workload/task oracles, request/phase instrumentation and end-to-end receipts. Consume mlx-native dispatch/timing counters; add any missing family-neutral trace mechanism in a separately tested and published native change.

Validation: Run the checked-in certified harness sequentially against frozen engines with balanced crossover ordering; verify real runnable-slot traces, raw responses, result scoring, cleanup and receipt identities.

Predecessors: F1, Q5.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

### F3. Reconcile ADR execution status and invalidated scientific claims

Outcome: The governing ADRs describe current code and distinguish falsified hypotheses from remaining plans.

Requirement: The program MUST deliver the governing ADRs describe current code and distinguish falsified hypotheses from remaining plans.

Acceptance:

- Correct ADR-010 router-only work as invalidated by its own F64 evidence; reconcile ADR-011 integrated tiled/block-skip paths and ADR-019 implemented phases and current published native pin.

- Correct HASS chronology/attribution/EAGLE-2 evidence and Hydra architecture in ADR-037/039; remove unsupported compounded speed and checkpoint-compatibility assumptions.

- Update ADR-049 for logical/physical graft coordinates, DeepSeek persistence, family component validity and current source anchors; keep research decisions Proposed until approved.

- Add Updated stamps and concrete source citations; preserve historical failed experiment evidence and reconcile any existing managed source bindings rather than duplicate their work.


Repository ownership: hf2q owns the named ADR corrections and managed source reconciliation. Verify native implementation claims against mlx-native source and published version; do not implement kernels or silently rewrite native release history.

Validation: Run the ADR inspector, compare each corrected claim with named code/paper passages, and reconcile affected existing awa sources.

Predecessors: none.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

## C. Exact agentic state reuse

### C1. Specify atomic multi-anchor state and component validity

Outcome: A cross-family exact checkpoint contract handles lineage, mutable windows, grafts and publication without stale reuse.

Requirement: The program MUST deliver a cross-family exact checkpoint contract handles lineage, mutable windows, grafts and publication without stale reuse.

Acceptance:

- Distinguish logical token lengths, physical cursors and graft identity; validate FULL, SWA, recurrent and compressed/indexer sidecars according to family rules.

- Define committed versus pending state, preflight-all-then-mutate restore, descendant invalidation before divergent writes, hard reset after restore failure, and session/model/template/config identity.

- Account K committed plus one pending payload in reclaimable owned bytes; retained hidden rows cannot pin chunk-sized parent buffers. Define in-memory versus disk-promotion policy.

- Specify A-to-B-to-C rewind/diverge/old-C rejection, cancellation, graft replant, cross-conversation isolation and idle-byte conservation oracles before implementation.

- Cancellation never publishes pending anchors; prune descendants invalidated by executed writes and preserve only physically valid committed ancestors. Failed prefill/rollback cannot resurrect overwritten rows.


Repository ownership: hf2q owns anchor identity, lineage, component validity, publication and rollback policy. mlx-native owns any required buffer-view, copy, fence or allocation primitive contract; track native API gaps separately.

Validation: Review HLD/LLD and run boundary state-machine regressions with synthetic state and real family layout invariants.

Predecessors: F3. (C1 is a test-only contract with no hardware work; under the 2026-10-30 Q5 rewrite, Q5 gates only each family's hardware work, so Q5 is not a C1 predecessor.)

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

### C2. Implement and qualify Qwen multi-depth prompt reuse

Outcome: Qwen3.6 and Qwen3.8 reuse the deepest valid earlier turn after agent context edits.

Requirement: The program MUST deliver Qwen3.6 and Qwen3.8 reuse the deepest valid earlier turn after agent context edits. It is built regardless of measured divergent-turn demand on the canonical workload (operator decision, 2026-11-03); no demand measurement gates it.

Acceptance:

- Implement bounded slot-local anchors with pending publication, atomic restore, lineage pruning, saved-logit equality hits, right-sized payloads and owned-byte telemetry. Recurrent (DeltaNet/conv) state restore builds on the ADR-065 P4a/P4b GPU DeltaNet restore rather than a second restore path.

- An approved C2 HLD/LLD states how anchor-snapshot→state restore reuses the P4a primitive, whose contract covers capture→state only. If the primitive must be extended, the design links the mlx-native issue that states the extension's contract.

- Prove cold-versus-resumed exactness on both Qwen artifacts with graft absent/present/identity change, speculation off/auto, cancellation, failed restore and four concurrent clients.

- Test the cold (non-restore) divergent prefill that completes successfully and then refuses the older anchor whose rows it overwrote (C1 final review carry-over O1, success-path twin).

- No joint turn is refused `NotEligible` because of anchor capture/restore (a reason-tagged counter reads 0), and the joint-decode eligibility rate with anchors on is within the Q3 CI of anchors off on the same workload and binary.

- Frozen limits: divergent-turn worst-slot semantic TTFT, paired against cold on the same binary with the anchors kill switch, has a Q3 Williams-block (5–20 blocks) 95% CI that excludes 0 in the improving direction; append-only four-agent makespan and worst-slot TTFT p95/SSE-gap max ratios have 95% CI upper bounds ≤ 1.025 against anchors off under the Q3 sequential precision stop (C2 decision D2, owner, 2026-11-03; stricter than the Q3 regression band); process footprint with anchors on is ≤ 1.25× a fresh Q5 calibration.

- C2 campaigns use Q3's pre-registered sequential 2.5-point precision stop (`q3_stats.stop_decision`, 5–20 blocks), not a fixed block count; a fixed count is used only if the owner chooses one from a labelled pilot that is never pooled (C2 decision D3, owner, 2026-11-03).

- Fresh Q5 calibration for Qwen3.8 and Qwen3.6 with the declared anchors per slot at the declared workload precedes any C2 hardware run.

- Qwen3.6 is judged by cold-versus-resumed bitwise exactness against the same-route twin, `ExactTranscripts` against anchors off on the append-only workload, and, on divergent turns, the Class B decision envelope (Q1 numeric policy §3.2, D ≤ 0.25 nat) on every logit row of each divergent turn against teacher-forced anchors-off rows on the same frozen prefix; any `RouteSensitive` row blocks C2 until localized. Task pass rate is not used while no workload on which Qwen3.6 passes is declared (Q4-QWEN36); since 2026-11-05 the declared workload is the Qwen3.6 repair-turn workload, on which the paired task gate applies. Qwen3.8 is judged by the same rules and additionally passes the paired task-quality gate (C2 decision D1, owner, 2026-11-03: an anchors-on resume of a divergent turn is a Class B alternate route, so bitwise transcripts cannot be required there).

- The anchor identity omits C1's `conversation` field, so exact prefixes are shared across clients as today (C2 decision D4, owner, 2026-11-03; departs from C1's normative identity list).

- Cancellation discards the request's own pending anchor; the cancel-smoke retry TTFT and recovered-depth figures are attached to the C2 receipt (C2 decision D5, owner, 2026-11-03).

- Primary Qwen3.8 evidence uses the vanilla official 27B dense model; grafted/abliterated variants remain separate compatibility arms. Family parity obligations for future Gemma/DeepSeek implementations remain explicit successors, not a universal multi-anchor shipping claim.


Repository ownership: hf2q owns Qwen model/cache integration and anchor selection; consume published mlx-native operations. Any required state-copy or storage primitive is implemented and parity-tested in mlx-native before a pinned hf2q integration.

Validation: Run the four-slot divergent-edit state/lifecycle gates that C2 itself authors and runs (on the workload-variant mechanism), the paired TTFT and append-only campaigns above and allocation accounting. The F2 fixed-six successor campaign is not a C2 predecessor. An independent reviewer reproduces the exactness matrix and the paired TTFT effect from raw receipts.

Predecessors: C1, ADR-065 P4a (#526), ADR-065 P4b (#527; itself requires ADR-065 B1 #503 and J1 #512). Moved to v0.1.29 (operator decision, 2026-10-30), then to v0.1.30 (operator decision, 2026-11-03).

Priority: High. Core scope.

Delivery boundary: Qualified implementation merged to main for v0.1.30.

### C5. Remove measured warm-request frontend overhead exactly

Outcome: Warm four-agent requests spend less time in rendering/tokenization/lookup without changing token identities.

Requirement: The program MUST deliver warm four-agent requests spend less time in rendering/tokenization/lookup without changing token identities.

Acceptance:

- Measure primary and stable-prefix rendering/tokenization separately from affinity wait, restore, suffix prefill and first semantic output.

- Only implement a reuse or acceleration path if the profile shows material headroom; account for BPE merges across boundaries, special tokens, template changes and vision expansion.

- Compare exact tokens with the authoritative tokenizer across randomized boundary edits and full agent transcripts; a CUDA GraphTok result is not Metal speed evidence.

- Open an acceleration implementation only if frontend work exceeds 10% of measured warm semantic TTFT and an exact path has credible end-to-end headroom; otherwise close the deciding experiment with evidence.


Repository ownership: hf2q owns rendering/tokenization/lookup and request-level profiling. A demonstrated GPU primitive belongs in mlx-native; tokenizer and template semantics remain in hf2q.

Validation: Profile F2 warm turns, compare full-token oracle with proposed CPU/incremental/Metal path, and record pass/falsification and paired end-to-end timing.

Predecessors: none.

Priority: Medium. Conditional hypothesis; require deciding evidence before implementation.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

## S. Four-slot scheduling and shared forwards

### S1. Prove mixed cooperative prefill cannot corrupt decoders

Outcome: DeepSeek cooperative cohort commit and poison are isolated from simultaneously decoding nonmembers.

Requirement: The program MUST deliver deepSeek cooperative cohort commit and poison are isolated from simultaneously decoding nonmembers.

Acceptance:

- Enumerate every target cache, circular window, compressor/indexer sidecar, ledger and publisher mutation.

- Exercise success, late failure, poison and cancellation during a mixed step; nonmember state and subsequent outputs must equal independent execution.

- If isolation fails, retain the bypass and record a bounded successor scheduler design; removing the bypass alone is not a deliverable.

- This is a DeepSeek-specific bounded coherence experiment; follow-on mixed policy is a successor after the deciding result. MAX_COOPERATIVE_PREFILL_ROWS is 2048, defined at src/inference/models/deepseek4/verifier_forward.rs:26 and consumed at src/serve/api/engine.rs:9859; prior 4096-row OOM evidence prohibits speculative cap expansion.


Repository ownership: hf2q owns DeepSeek cohort membership, cache mutation transactions, poison/cancel handling and nonmember isolation. mlx-native owns any missing execution/fence primitive and its synchronization tests; a native mechanism does not define serving cohort policy.

Validation: Run adversarial transaction tests and a real-artifact cohort-prefill plus active-decode oracle.

Predecessors: Q5.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

### S3. Share target forwards across four active slots where profitable

Outcome: Four-slot serving amortizes target weight reads through real physical batching rather than independent batch-one loops.

Requirement: The program MUST deliver four-slot serving amortizes target weight reads through real physical batching rather than independent batch-one loops.

Acceptance:

- Profile live tensor row dimensions, dispatch counts and weight reads for Qwen dense/MoE, Gemma and DeepSeek under F2; distinguish existing batching from gaps.

- Implement missing profitable shared-forward paths with slot-indexed recurrent/KV state and heterogeneous prompt/verify/decode lengths; preserve ordinary fallback for unsupported exact semantics.

- Compare ordinary physical batch-four, independently scheduled requests and speculative row cohorts; require improved goodput/makespan with unchanged per-slot quality and fairness.

- Use ADR-040 historical N4 Gemma 0.75x-to-0.81x peer throughput and non-amortized attention as falsifiable leads, not current measurements. Confirm which batching/attention fixes already landed before creating another kernel.


Repository ownership: hf2q owns physical cohort construction, slot-indexed model graphs and heterogeneous phase scheduling. mlx-native owns batched matmul/attention kernels, dispatchers, shared-weight buffer layouts and primitive parity. Profile existing native broadcast batching before proposing new kernels; integrate only a published API.

Validation: Capture kernel/row traces and run four-slot independent-versus-shared target and end-to-end comparisons, including cancel/EOS and partial cohort failures.

Predecessors: Q5.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

## D. Current pretrained speculation

### D1. Qualify existing warm-turn speculation and close measured boundary gaps

Outcome: Existing warm-turn MTP/history behavior is proven at four slots and only demonstrated profitable boundary gaps are extended.

Requirement: The program MUST deliver existing warm-turn MTP/history behavior is proven at four slots and only demonstrated profitable boundary gaps are extended.

Acceptance:

- Current warm append with valid speculative metadata already works. PromptCacheHit only excludes full equality without a reusable speculative boundary; inspect src/serve/api/engine.rs:19240-19290 and src/serve/api/engine_qwen35.rs:4004-4234 before changing behavior.

- Pass full acceptance, partial rejection, zero acceptance, EOS, cancellation, graft restore and eight subsequent ordinary target decisions.

- Qualify exact represented API semantics at four slots and disable measured losing proposals; sampling support requires correct distributional acceptance, not equal-seed anecdotes.

- Bound grafts intentionally disable MTP because the draft arena lacks graft state; retain this fallback unless a separately approved exact graft-aware transaction proves correctness. No greenfield cached-speculation feature is assumed missing.


Repository ownership: hf2q owns warm-turn speculative state, proposal/acceptance policy and tool/client qualification. mlx-native owns draft/verify compute primitives and dispatch correctness; preserve native numerical contracts and published-pin sequencing.

Validation: Extend current exact speculation and cache lifecycle gates; compare F2 warm tool rounds off versus auto.

Predecessors: C1, Q5.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

### D2. Compare current pretrained DFlash2 and DSpark on exact targets

Outcome: A dated, reproducible experiment selects or rejects current pretrained draft architectures for four-agent Apple serving.

Requirement: The program MUST deliver a dated, reproducible experiment selects or rejects current pretrained draft architectures for four-agent Apple serving.

Acceptance:

- Compare native MTP/history, official DFlash2, September Red Hat DSpark and available current peer support; freeze target/draft metadata, tokenizer and weights.

- Include abliterated/grafted target compatibility, long contexts, nonrepetitive code, tool calls, memory and accepted emitted tokens per total draft/verify/recovery cost.

- Choose by four-slot task completion and quality, not acceptance length or single-slot headlines; document unavailable arms and negative results without forcing a winner.

- This is a deciding reference experiment, not a committed native port. Native drafter/tree work is successor scope only if actual four-slot quality, memory and net cost beat current MTP/history; primary vanilla Qwen3.8 is the compatible dense target.


Repository ownership: hf2q owns isolated reference qualification, target/draft compatibility and four-agent cost/quality decisions. No native port is promised; any winning successor separates hf2q model/policy integration from mlx-native primitive/kernel implementation.

Validation: Run isolated reference qualification and four-slot matched trials; validate independent sequential target decisions and model schema before proposing a native port.

Predecessors: Q5.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

**Result (2026-11-02, M5 Max 128 GB; `docs/research/adr063-d2-drafter-comparison.md`).**
No current pretrained external drafter qualifies as successor scope; D2 closes
negative for DFlash2 and DSpark on Qwen3.8 27B at four slots. Primary group
(vanilla target, 3 interleaved passes, lexicographic rule completion, final
tests, Q5 memory tier, median makespan), best first: P1 (peer MTP) 77.7 s, H0
81.4 s, H1 82.3 s, T2 (official DFlash2 on TensorFold) 90.7 s, P0 118.9 s, then
P3 (peer DSpark) and T0 (TensorFold copy-draft baseline), both arm-negative on a
task failure. The peer's DFlash2 (P2) corrupted tool-call output at four slots
and failed qualification. DSpark's one passing run on the primary workload
(qualification, 107.9 s) is behind H1, and it ranks above H1 only on the
abliterated target (group A), which the rule does not accept in place of the
primary rank. H1 does not beat H0 at N4, consistent with D1. P1, a
pre-registered arm, ranks above H1 in the primary group and in A, but Q5
stopped its long-context run with `Stopped: swap_growth` (swap +1.22 GB over
baseline). That memory veto means it is not successor scope as-is; it is the
strongest observation for successor MTP work. The TensorFold target tokenizer
differs from the base one hf2q serves (same vocab and merges; different
pre-tokenizer regex and offset/decoder options), so the T arms are near-exact
target matches. Long context (2 passes, about 40k-token prompts): the order is
P0 574.5 s, H0 581.0 s, H1 594.8 s, then T2 (one pass at 1034.5 s, one Q5
`Stopped: free_cached_floor`) and P1 (`Stopped: swap_growth`). Non-repetitive
code (2 task instances, 13 of 14 runs with data; N1-P0 was refused by the
quiet-host preflight): P1 78.1 s, T2 86.0 s, H1 89.6 s, then P3 108.8 s, T0,
H0 (one task failure) and P0. N does not veto P1, and T2's N gain cannot
override its primary rank, so neither DFlash2 nor DSpark is successor scope.

## K. Apple Silicon kernels and ownership

### K1. Tune October small-row quantized verifier and decode dispatch

Outcome: Actual four-slot and verifier shapes use profitable Apple GPU kernels without precision regressions.

Requirement: The program MUST deliver actual four-slot and verifier shapes use profitable Apple GPU kernels without precision regressions.

Acceptance:

- Prioritize actual GGUF Q4_K/Q5_K/Q6_K verifier widths 4-8 and ordinary physical batch-four, retaining widths 1/2 as controls; expand to 16/32/64 only when live cohort traces justify them. October MLX NVFP4/MXFP4/MXFP8 gains are design evidence, not K-quant results.

- Compare existing mvN/mul_mv_ext/TensorOps/split-K candidates with F32 partial accumulation where required; merged upstream bf16 gains are not proof for K-quants.

- Prove logit/routing and transaction invariants, then require F2 end-to-end improvement; selection must be hardware/shape-aware rather than one global override.

- Any mlx-native change must be published as a reproducible registry revision and pinned in Cargo.toml/Cargo.lock before hf2q acceptance. Account for this separate release critical path; a local path override is never acceptance evidence.


Repository ownership: mlx-native owns all kernel and dispatcher changes, hardware/shape capability routing, buffers and primitive parity/microbenchmarks. hf2q owns live-shape capture, public API integration, dependency pins and F2 end-to-end acceptance. Link a native implementation issue/PR; publish and verify the native crate before pinned hf2q qualification. No compute kernels are implemented in hf2q.

Validation: Run frozen shape microbenchmarks, independent sequential/logit oracles, four-slot row traces and paired F2 trials.

Predecessors: Q5.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

## Q. Coherence and final acceptance

### Q1. Separate internal numerical invariance from peer task quality

Outcome: A diagnostic oracle detects batching/reuse/state divergence without requiring universal peer text identity.

Requirement: The program MUST deliver a diagnostic oracle detects batching/reuse/state divergence without requiring universal peer text identity.

Acceptance:

- Compare internal cold/resume, batch-one/batch-four, chunk size, request order and sequential/speculative paths; locate first norm/QKV/RoPE/attention/residual/MoE divergence with frozen intermediates.

- Record routing margins, expert IDs, logit error and task outputs; no routing smoothing, tie-threshold change or approximate recurrent replay may masquerade as an exactness fix.

- Use independent correctness fixtures for peer coherence; sampled quality margins and trial counts are frozen before candidate runs.


Repository ownership: hf2q owns family state/logit/routing and task-quality oracles. mlx-native owns operation-level numerical parity, batched-versus-sequential primitive tests and low-level trace support. A passing kernel test alone does not establish model/task coherence.

Validation: Run source-bound intermediate and transaction comparisons with deliberate stale-state/incorrect-routing negative controls plus the F2 task oracle.

Predecessors: none.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

### Q2. QE-certify the comparison harness with adversarial controls

Outcome: The experiment harness rejects false quality or speed wins and measures actual slot progress.

Requirement: The program MUST deliver the experiment harness rejects false quality or speed wins and measures actual slot progress.

Acceptance:

- Fail deliberately serialized four-client service, role-only SSE, missing semantic output, wrong tool arguments, stale lineage, cross-slot leakage, silent truncation and altered output budgets.

- Use injected known delays/speedups and corrupted receipts to prove timer direction, metric units, cache accounting, binary/model identity, result scoring and cleanup; no implementation-mirroring tests.

- Prove active slot/phase counts and physical cohort membership rather than merely overlapping HTTP lifetimes; distinguish one-slot controls, mixed 3+1/2+2/1+3 populations and pure four-prefill/four-decode phases.

- Check critical-path and failure-path coverage with Agentic-QE or equivalent independent evidence. A MetaHarness audit may assess harness quality, but scores cannot substitute for real artifact tests.


Repository ownership: Independent QE qualifies hf2q campaign clients, scoring, concurrency and receipt validation. Native instrumentation and its failure controls remain mlx-native API/test work, consumed at an exact published revision.

Validation: Run deterministic fake-server/receipt negative controls, then verify instrumentation against actual server traces and fail-closed artifact identity checks.

Predecessors: none.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

### Q3. Independently prove gains over stock hf2q and latest peer

Outcome: A QE-owned three-arm comparison quantifies the experimental gain and the remaining peer gap without quality loss.

Requirement: The program MUST deliver a QE-owned three-arm comparison quantifies the experimental gain and the remaining peer gap without quality loss.

Acceptance:

- Freeze three primary arms at run start: current unmodified hf2q, experimental hf2q based on that exact baseline, and latest upstream peer; retain immutable source/binary identities for all runs.

- Use balanced interleaved three-arm crossover blocks, warmup and stock controls; target a predeclared 95% effect-interval half-width <=2.5 percentage points, minimum five and maximum twenty blocks. An unresolved interval means inconclusive, not faster.

- Report experiment-versus-stock and experiment-versus-latest-peer effects separately for one-slot controls and primary four-slot agentic workloads, including mixed 3+1/2+2/1+3 and pure-phase controls.

- Cluster uncertainty by independent run/conversation, not individual tokens; report quality/task completion, makespan, successful-task goodput, per-slot p95/max semantic TTFT and SSE gaps, memory and thermal validity.

- An independent reviewer must reproduce from checked-in commands and raw receipts; faster incorrect tasks, serial scheduling, mismatched settings, changed artifacts or cherry-picked trials fail.

- A separate named QE session/host from the candidate implementer executes Q3 and records its identity; freeze the certified harness commit and metric definitions before candidate implementation/evaluation.

- Use latest peer in its best valid configuration as the mandatory peer arm for each family that has a runnable peer on its declared workload, and retain a separately labeled matched-settings diagnostic. A family without a runnable peer arm is reported as unavailable with evidence. Peer quality failures are recorded; a missing, failed or silently disqualified peer cannot yield a speed victory by default.


Repository ownership: Independent QE executes source-bound hf2q-versus-stock-versus-peer campaigns, recording both native and hf2q identities. Native microbenchmarks are diagnostic evidence, not substitutes for all four model-level speed wins.

Validation: Independent QE executes the certified harness against frozen stock/experiment/latest-peer binaries and publishes paired effect sizes, confidence intervals, rejected trials and verdict.

Predecessors: Q2, Q5.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

### Q4. Accept and release a faster coherent four-agent Apple configuration

Outcome: The program understands why hf2q is slower than the peer on the same Mac and becomes clearly faster: per family, the published hf2q finishes the canonical four-agent coding task at least 5% sooner than the peer in its best valid configuration, with quality gates passing. For Qwen3.8 and Qwen3.6 it must also finish at least 5% sooner than Splash in its best valid configuration (see "Splash peer of record for the Qwen families").

Requirement: The program MUST explain, per family, where hf2q loses time against the peer on the M5 Max 128 GB and MUST deliver, for each of Qwen3.8, DeepSeek-V4, Qwen3.6 and Gemma 4, a configuration whose Q4 verdict is Win, with the declared quality, fairness and memory gates intact. Q4 is the umbrella story; it closes through the per-family stories Q4-QWEN38, Q4-GEMMA with Q4-GEMMA-ATTN, Q4-DS4 with Q4-DS4-RACE, and Q4-QWEN36.

Acceptance:

- Q4 win = `scripts/four_agent/q3_stats.py` `family_verdict` == Win: the hf2q/peer four-agent makespan ratio, peer in its best valid configuration, has a 95% CI upper bound ≤ 0.95 (`PEER_MARGIN`), under the Q3 Williams-block design (5–20 blocks). For Qwen3.8 and Qwen3.6, the hf2q/Splash ratio, Splash in its best valid configuration, must meet the same bound in the same campaign. The F2-tier "Win" label, which is against v0.1.28 stock only, is not a Q4 win.

- The paired task-quality gates on the exact candidate binary pass against v0.1.28 and against the peer (against the native ds4 arm where the same-GGUF peer is unavailable), and for Qwen3.8 and Qwen3.6 also against Splash, and all deterministic task/tool/isolation gates pass.

- Fairness: worst-slot TTFT p95 and SSE-gap max ratios against v0.1.28 stock stay within the stock fairness band 1.025 (`NEUTRAL_RATIO`). Vs-peer TTFT ratios are disclosed, not gated; SSE gaps against the peer are shown as not comparable.

- DeepSeek-V4 exception (Q4-DS4-RACE decision D1, operator, 2026-11-03): no released hf2q opens ds4's file, so in Q4-DS4-RACE the vs-stock makespan gain, its precision, this fairness band and the v0.1.28 quality gate are taken from a same-artifact candidate-vs-v0.1.28 campaign on hf2q's own DeepSeek-V4 artifact run with the same candidate binary, bound into the ds4-file record campaign by path and sha256; the ds4-file campaign itself measures only the peer contrast.

- Each family's accepted configuration passes a fresh Q5 calibration on the release candidate.

- One-slot (n1) regressions against v0.1.28 or the peer are disclosed with the release, not gating. Single-slot or microkernel speed alone cannot satisfy this story.

- Each family story records a per-stage attribution of the hf2q-versus-peer gap measured on the same run, so the explanation of the gap is evidence, not inference.

- Give separate speed/quality/fairness verdicts for vanilla Qwen3.8-27B dense, DeepSeek-V4, vanilla Qwen3.6 MoE and Gemma 4 on M5 Max/128 GB. Each family's verdict is Win, Loss, Unavailable (no runnable peer) or Failed (hf2q task failure), with evidence; only Win counts; nothing is pooled or dropped. Latest peer must complete the declared workload for a valid performance verdict; peer task failures are explicit quality findings, never grounds for claiming a win against nothing. Native-format peer competition is reported separately from the same-GGUF peer verdict.

- Name the release candidate commit at closure. Per family, re-run the Q3 n4 (+n1) campaign on that candidate, or show `ExactTranscripts` equivalence to the measured binary. Record the exact winning candidate work keys and immutable commits as Q4 predecessors in the source/extraction/awa graph and obtain independent verdicts for their combined configuration; a prose-only winning subset cannot bypass candidate gates. Excluded/falsified experiments need a recorded disposition, not forced implementation.

- The Q1 successors (Q1-DS4, Q1-GEMMA-T2LOC, Q1-QWEN38-RERUN, Q1-QWEN36-SPEC, Q1-GEMMA-SPEC) do not gate Q4. DeepSeek-V4 quality for Q4 is the paired task-quality gate plus the S3 bitwise joint == serial check.

- Q4 cannot close until every family is Win or the operator records a re-scope of that family in this ADR. A family's Win is measured in its closing story: Q4-QWEN38, Q4-GEMMA-ATTN (after the Q4-GEMMA profile), Q4-DS4-RACE (after the Q4-DS4 fit result) and Q4-QWEN36. A family that cannot reach Win keeps its verdict of record (Loss, Unavailable or Failed) with evidence; a Q4-DS4 no-fit or a Q4-QWEN36 BelowFloor result goes to the operator as `needs_decision`.

Repository ownership: The integration owner coordinates two release chains: merge/publish/verify mlx-native updates first, pin and clean-build hf2q, then qualify and publish the hf2q crate plus signed standalone artifacts. Record both package versions/checksums, immutable source commits, native PRs and all four independent model verdicts.

Validation: Independent review of the combined candidate and raw receipts, followed by exact-artifact release verification on the target Apple host.

Predecessors: F3, Q2, Q3, Q5, Q4-QWEN38, Q4-GEMMA, Q4-GEMMA-ATTN, Q4-DS4, Q4-DS4-RACE, Q4-QWEN36.

Priority: High. Core scope.

Delivery boundary: Release v0.1.30, or the first published hf2q release containing each family's independently accepted configuration.

### Q5. Run four-slot configurations on the M5 Max 128 GB without exhausting memory

Outcome: Every accepted configuration, per model family, runs its full four-agent workload on the dedicated laptop without running out of memory, and without reserving memory that nothing uses.

Requirement: The program MUST admit each family's four-slot configuration from measured behavior on this host and MUST stop any run before it exhausts memory.

Acceptance:

- The host is dedicated to this program and its single active runtime. Admission charges the measured host baseline at admission time, plus hf2q's calibrated peak for the same artifact, source, workload, context and slot count. It reserves nothing for other agents and no fixed worst-case cells.

- Each family (Qwen3.8 27B, Qwen3.6 35B-A3B, Gemma4 26B-A4B, DeepSeek-V4 Flash) is calibrated by one supervised run of its largest declared four-agent workload. A calibration passes when pressure never reaches critical, swap grows by less than 512 MiB and Free plus cached memory stays at or above the effective floor. The family is recalibrated when its artifact, source, context or slot count changes.

- The effective floor is at least 2 GiB. It is raised to cover the fastest measured post-readiness drop in Free plus cached memory multiplied by the sampling interval plus the measured stop latency. On 2026-10-30 that required 1.0 GiB for DeepSeek-V4, 5.3 GiB for Gemma4 and 13.2 GiB for Qwen3.8, the last from first-request allocation in the 1.5 s after readiness.

- Every run has live stops: critical memory pressure, swap growth beyond 512 MiB, Free plus cached memory below the effective floor, or process growth beyond a mandatory cap of 1.25 times the calibrated peak footprint. A calibration run without a prior receipt uses the effective floor as its cap. macOS warning pressure is recorded but is not a stop. It was present for most of each DeepSeek-V4 run, with at least 4 GiB still reclaimable. A stop is a preserved failed case, not a retry trigger.

- Exactly one hf2q or peer runtime runs at a time, including metadata commands. Listeners, processes and memory recovery are verified between loads. Idle user applications may be quit when a measurement needs the headroom; system services are kept.

- A configuration may not fit by silently reducing matched context, task size or output budget. Any reduction is an explicit operator decision.

- Record the host baseline, calibrated peak, pressure, swap and compression deltas, minimum Free plus cached memory and cleanup for each family.


Repository ownership: hf2q owns the admission policy, calibration and live stops. mlx-native owns allocation and residency lifetime and the low-level counters the stops read.

Validation: An independent reviewer checks each family's calibration receipt and stop behavior, including one negative control that proves a stop fires before memory is exhausted.

Predecessors: F1.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

### Q4-QWEN38. Diagnose and close the Qwen3.8 gap to the peer

Outcome: hf2q finishes the canonical Qwen3.8-27B four-agent coding task at least 5% sooner than the peer in its best valid configuration (with its own MTP head) and than Splash in its best valid configuration, and the measured cause of today's gap is recorded.

Requirement: The program MUST attribute the Qwen3.8 hf2q-versus-peer gap by stage and MUST deliver a Qwen3.8 configuration whose Q4 verdict is Win.

Acceptance:

- Starting point (Q3 v11, hf2q `38485a23`): the MTP peer finishes 6.0% sooner (95% CI 4.2–7.9%), emitting 1.88× the completion tokens at a 2.00× per-output-token rate; hf2q's worst-slot p95 TTFT is 2.61× the peer's.

- Per-stage attribution of the gap on the same run, using the ADR-065 B1 (#503) baseline and fairness freeze and its E1–E8 (#504–#511) deciding measurements (stage time split, complete-round cost, counterfactual acceptance, makespan composition and eligibility), with the Q4-TRACE joint-decode witness. The distortion of each attribution is stated.

- The 2.6× worst-slot TTFT gap is decomposed into queueing, admission, prefill and first-token stages, and each stage is closed or disclosed with evidence.

- Speculation at four slots is decided from measurement (dense Qwen speculation defaults to off at four slots since #493; the peer uses its MTP head): the chosen setting is the one in the measured winning configuration, and its exactness follows the ADR-065 numerical and transaction contract.

- The family verdict meets the Q4 win definition and every Q4 quality, fairness (stock band 1.025) and fresh-Q5 gate on the release candidate.

Repository ownership: hf2q owns attribution wiring, scheduling, speculation policy and the campaign. Every Metal kernel, dispatcher and primitive parity test belongs in mlx-native and is consumed at a published, exactly pinned revision.

Validation: Independent review of the per-stage attribution and of the Q3-design hf2q-versus-peer campaign from raw receipts.

Predecessors: ADR-065 E8 (#511, G0 gate decision), Q4-TRACE, Q5, the Splash Qwen3.8 fit and best-configuration record. Consumes ADR-065 P4a (#526), P4b (#527) and any ADR-065 build story admitted after the E8 gate.

Priority: High. Core scope (first-tier family).

Delivery boundary: Release v0.1.30, or the first published hf2q release containing the accepted Qwen3.8 configuration.

### Q4-GEMMA-KERNEL. Provide a precise tiled warm attention primitive in mlx-native

Outcome: mlx-native provides a tiled warm attention primitive with 32-bit accumulation that Gemma 4 warm resumes can use without the `TiledF16` decision flips.

Requirement: mlx-native MUST provide, under a stated primitive contract, a tiled warm attention kernel with F32 accumulation and F32 output, published and pinnable before hf2q integrates it.

Acceptance:

- A linked mlx-native issue states the primitive contract before implementation: input layouts (F16 K cache, dequantized TQ-HB V, Q rows), sliding and global masks, scale and softcap, supported head dimensions, F32 accumulation and F32 output, and the tolerance against an F64 reference attention (design §5.1).

- Design approved by the operator, 2026-11-03: `docs/research/adr063-gemma-tiled-attn-design.md` §3–§7 (F16 Q and K with F32 scores, softmax, V operand, accumulator and stored O; one dispatcher pair, d256 sliding and d512 global; existing F16/BF16 entry points stay bit-identical; tolerances frozen in §5.1 against an F64 reference). It extends the #552 patch rather than adopting it.

- The draft F32-accumulate/F32-output patch from the closed PR #552, kept on branch `exp/gemma-tiled-precise` (`docs/research/adr063-gemma-tiled-precise-mlx-native.patch`, with the diagnosis in `docs/research/adr063-gemma-tiled-precise.md`), is the starting point; the branch is retained until the kernel lands or is falsified.

- Primitive parity tests against an F64 reference attention (design §5.1) and against the vec-hybrid route pass at Gemma 4 warm-resume shapes, and a microbenchmark compares it with `TiledF16` and `VecHybrid` at those shapes.

- The kernel is merged, released, verified on crates.io and available for an exact hf2q pin.

Repository ownership: mlx-native owns the kernel, its dispatcher, contract, parity tests, microbenchmarks and release. hf2q consumes it only through Q4-GEMMA-ATTN.

Validation: Independent review of the contract, the parity and microbenchmark receipts and the published release.

Predecessors: none.

Priority: High. Core scope (second-tier family).

Delivery boundary: Published mlx-native release containing the primitive, pinned by hf2q for v0.1.30.

### Q4-GEMMA. Profile the Gemma 4 gap to the peer

Outcome: The Gemma 4 26B-A4B gap to the peer in its best valid configuration is attributed by stage on the same run.

Requirement: The program MUST establish the peer's best valid Gemma 4 configuration and MUST profile hf2q and that peer by stage on the same run.

Acceptance:

- Starting point (Q3 v9, hf2q `5d421a2c`): the peer finishes 13.4% sooner (CI 10.6–16.2%, 20-block cap), measured against the peer with speculation off (`--spec-type none`); the stock SSE-gap fairness gate fails at 1.047× (CI 1.036–1.058).

- Establish the peer's best valid Gemma 4 configuration (its speculation options), as D2 did for Qwen3.8, before any Q4 campaign. The gap may widen.

- A per-stage profile of hf2q and the peer on the same run attributes the gap (attention, MoE, dispatch/encode, scheduling, SSE cadence) with stated distortion, including the share attributable to warm attention and to the stock SSE-gap regression.

- The profile names the stages Q4-GEMMA-ATTN and any further Gemma change must close; it makes no speed claim by itself.

Repository ownership: hf2q owns profiling wiring, the peer configuration search and the profile record.

Validation: Independent review of the peer best-configuration evidence and the per-stage profile from raw receipts.

Predecessors: Q5 (#371).

Priority: High. Core scope (second-tier family).

Delivery boundary: Measured evidence merged to main for v0.1.30.

### Q4-GEMMA-ATTN. Integrate the precise tiled warm attention and close the Gemma 4 gap

Outcome: hf2q finishes the canonical Gemma 4 26B-A4B four-agent coding task at least 5% sooner than the peer in its best valid configuration, with the stock fairness gate passing.

Requirement: The program MUST integrate the Q4-GEMMA-KERNEL primitive in hf2q under the Q1 exactness/envelope and quality gates, and MUST deliver a Gemma 4 configuration whose Q4 verdict is Win.

Acceptance:

- A written design, approved before implementation, states the route selection, kill switch and how the integration addresses the stages the Q4-GEMMA profile attributes. Approved by the operator, 2026-11-03: `docs/research/adr063-gemma-tiled-attn-design.md` §8–§9 (`HF2Q_GEMMA_WARM_TILED_ATTENTION=precise` selects `TiledF32`, `VecHybrid` stays default until every gate passes, `0` is the kill switch; the closing campaign is a combined candidate with the profile-named changes, each under its own design).

- #544 set warm tiled attention default-off after `TiledF16` flipped a near-tie decision against `VecHybrid`; tiled-off costs Gemma about 7.8–9.2% (v9 review), and v7 with tiled on was still a Loss (1.058), so the kernel alone cannot be assumed to flip the verdict. Any further change the profile requires is scoped with its own design.

- Outcome 2026-11-05: the Q4-GEMMA-KERNEL route (`TiledF32`) failed E1 and is falsified and removed from hf2q; see "Q4-GEMMA-ATTN outcome: TiledF32 falsified" below. The warm-attention lever moves to the bitwise-exact successor story.

- In hf2q the Q4-GEMMA-KERNEL route stays within the Q1 decision envelope and exactness classes (including the 0.10-nat near-tie policy) at the canonical divergence coordinate and passes the paired task-quality gates before it can become a default.

- The stock SSE-gap max ratio returns within 1.025, and the family verdict against the peer's best valid configuration (from Q4-GEMMA) meets the Q4 win definition and every Q4 gate on the release candidate.

Repository ownership: hf2q owns route selection, integration and the campaign, consuming Q4-GEMMA-KERNEL at an exact published mlx-native pin.

Validation: Independent review of the design, the Q1-envelope receipts, the quality gates and the Q3-design hf2q-versus-peer campaign.

Predecessors: Q4-GEMMA-KERNEL, Q4-GEMMA (#660, profiling result), Q5 (#371). The closing campaign (the combined candidate) also has as predecessors Q4-GEMMA-COLD, Q4-GEMMA-PREFILL-MOE, Q4-GEMMA-DECODE, Q4-GEMMA-SPEC-SERVE (operator decision 2026-11-04) Q4-GEMMA-ATTN-EXACT with its narrowed follow-up Q4-GEMMA-ATTN-EXACT-G2, and Q4-GEMMA-LMHEAD-MMA (operator decisions 2026-11-05): each delivers its qualified change, or its recorded falsification or stop, before that campaign runs.

Priority: High. Core scope (second-tier family).

Delivery boundary: Release v0.1.30, or the first published hf2q release containing the accepted Gemma 4 configuration.

### Q4-GEMMA-COLD. Prefill the simultaneous cold Gemma 4 first turns in one exact joint transaction

Outcome: The four simultaneous cold first prompts of the Gemma 4 26B-A4B four-agent task are prefilled in one joint transaction that is bitwise equal to per-slot prefill, so no agent waits for the other agents' serial prefills on turn 0.

Requirement: The program MUST admit every eligible cohort (design §3.1: all slots idle, two or more text lanes, each cold with an exact pre-generation boundary of at least 33 rows, at most 4,096 prompt rows in total, no graft, no diagnostic route override) of cold Gemma 4 requests into one joint prefill transaction that is Q1 Class A against the per-slot route, publishing each lane's boundary anchor, cursor and first token exactly as per-slot admission does. Every other group keeps its current route. It MUST measure the turn-0 effect on the four-agent task.

Acceptance:

- A written design, approved before implementation: `docs/research/adr063-q4-gemma-cold-design.md`. It covers the cohort rule (§3.1), per-sequence cold attention (§4.3), cue replay through the batched decode body (§4.4), failure semantics (§4.5) and kill switch `HF2Q_GEMMA_JOINT_COLD_PREFILL=0` (default on once the bitwise gates pass). Approved by the operator, 2026-11-04, with decisions D1–D4 as recommended (design §12).

- Starting point (Q4-GEMMA #660, L1, 7 blocks, hf2q `7f295b38`): the four turn-0 requests coalesce but are admitted per slot in series. Because each carries a stable boundary, the stable batch refuses them as uncached and the cold batch refuses boundaries. The result is a server-queue staircase of 1/285/560/834 ms (lane medians), 275 ms of prefill per lane and 1.19 s of "decode" on the first-served lane. The turn-0 span is +496 ms against the peer's MT, 18% of the 2.73 s makespan gap.

- Class A: on the official artifact, every cohort lane's first-token logits, cursors, all 30 layers' slot KV through the prompt, boundary anchor bytes and 16 following decode rows equal the per-slot route byte for byte. This covers the turn-0 prompts and the §6.2 synthetic shapes, including the 32-row per-slot cue and mixed {7, 40} cues, and is deterministic over 20 repeats. The iter-G(a) cold batch moved to the same per-sequence attention (lane floor 33 rows) also matches per slot bitwise. For the installed long-prompt cold prefill, only the cursor-0 chunk forward is claimed exact. Any differing byte blocks; no Class B fallback ships without a separate operator decision and the Q1 §3.5 campaign.

- `ExactTranscripts` on the four-agent workload, kill switch on against off.

- Speed, paired four-agent replay (same binary, kill switch off against on, 7 paired blocks, Q5 `CalibrationPass` per run): turn-0 span median reduction ≥ 0.20 s per agent with a 95% CI excluding 0; turn-0 server queue median ≤ 30 ms; turn-0 decode median ≤ 420 ms; cohort size 4 in at least 6 of 7 runs. Makespan is reported, not gated; the makespan verdict belongs to the combined candidate. The transaction itself takes ≤ 0.80× the per-slot sum on the turn-0 prompts. The no-boundary cold batch on the new attention route takes ≤ 1.05× its previous transaction time.

- The stock fairness band 1.025 (worst-slot TTFT p95, SSE-gap max) and the Q5 memory tier hold in the Q4-GEMMA-ATTN combined candidate, which carries this change; the Q4 Win is measured there, not here.

Repository ownership: hf2q owns the cohort admission, the per-sequence cold attention route, the cue replay, the tests and the campaign. It uses existing mlx-native kernels at the current pin; no mlx-native change.

Validation: Independent review of the design, the bitwise and determinism gate receipts, the Q1 T0/T2 `ExactEqual` witness on the candidate build, the `ExactTranscripts` record and the paired four-agent replay from raw receipts.

Predecessors: Q4-GEMMA (#660, profiling result), Q5 (#371).

Priority: High. Core scope (second-tier family).

Delivery boundary: Qualified implementation merged to main for v0.1.30 and included in the Q4-GEMMA-ATTN combined candidate.

### Q4-GEMMA-PREFILL-MOE. Reduce the Gemma 4 warm-prefill MoE cost per row

Outcome: Gemma 4 26B-A4B warm prefill spends measurably less GPU time per row in the routed experts, bitwise identical to the previous route, and the saving enters the combined Gemma candidate.

Requirement: The program MUST reduce the GPU time per row of the warm-prefill routed-expert stage (34% of warm-prefill GPU time in the Q4-GEMMA profile) without changing any output byte. It MUST first test, with a model-free microbenchmark, the hypothesis that expert matrix-multiply launch geometry is a material part of that stage. It MUST record the measured saving, or the falsification, against the profiled MoE share.

Acceptance:

- Starting point (Q4-GEMMA #660, hf2q `7f295b38`, native 0.15.16): warm-prefill MoE and layer tail are 34% of warm-prefill GPU time (1.09–1.11 s per run serialized; attention 48%, dense 11%, pre-attention 7.5%). About 225 M expert-matmul threadgroups are launched per four-agent run, of which an estimated 3.1 M have work (routing proxy; design §2.4).

- A written design, approved before implementation: `docs/research/adr063-q4-gemma-prefill-moe-design.md`. It covers the Gemma prefill MoE down call on the slotted expert entry point (P1, hf2q); a compact per-expert tile schedule with schedule reuse (P2, P2b, mlx-native); and a half-tile skip only where it is bitwise (P2c). The `mm_id`/`mv_id` threshold (32), dense thresholds, tensor-API selection and storage types are unchanged. Approved by the operator, 2026-11-04, with decisions D1–D7 as recommended (design §12).

- Every shipped change is Q1 Class A against the previous route. The mlx-native parity matrix (design §5.1) is bitwise on both metallib paths, and so are the hf2q real-artifact gates (joint warm == per-slot on every warm attention route, engine cohort == serial tool turns, per-slot parity, the route A/B layer-byte test, and the greedy144 Q1 endpoint `ExactEqual`). `ExactTranscripts` holds over every A/B run pair. A non-bitwise variant does not ship under this story.

- Speed, with one binary and the previous route as the A arm: at least 6 ABBA pairs matched by warm-transaction structure (design §7.1). The per-run sum of unprofiled joint-warm forward walls improves by at least 0.05 s (95% CI upper bound < 0). The profiled MoE category falls by at least half the measured microbenchmark saving, with the other categories within ±3%. The four-agent makespan does not regress (CI upper bound ≤ +0.05 s). The worst-slot TTFT p95 and SSE-gap max ratios stay within 1.025.

- Stop path (design §6, §7.4): if the frozen microbenchmark prediction for P1 + P2 + P2b is below 0.05 s per run, or the warm-prefill gate fails, P1 still merges as a bitwise change with its switch removed, P2/P2b/P2c are not built, and the story closes with the recorded falsification.

- The real-artifact gates run under the design's invocation rules (`--exact`, env gates set, `1 passed` with no skip line). The shared-kernel regression gates for DeepSeek-V4 and Qwen MoE pass at the native pin. Every run passes Q5 `CalibrationPass`.

Repository ownership: mlx-native owns the compact schedule, the schedule-reuse entry point, the half-tile skip, the parity tests, the microbenchmark and the release. hf2q owns the slotted down call, route switches, integration, the real-artifact gates and the A/B campaign, consuming mlx-native at an exact published pin.

Validation: Independent review of the design, the parity and microbenchmark receipts, the Class A real-artifact receipts and the A/B campaign from raw receipts.

Predecessors: Q4-GEMMA (#660, profile result PR #687, merged on main as `2dfe71ba`), Q5 (#371). Consumed by Q4-GEMMA-ATTN's combined candidate, which lists this story among its closing-campaign predecessors.

Priority: High. Core scope (second-tier family).

Delivery boundary: P1 merged to main; and either the mlx-native release containing P2 pinned and integrated by hf2q (P3), or, on the stop path, the recorded falsification. For v0.1.30, or the first published hf2q release containing the accepted Gemma 4 configuration.

### Q4-GEMMA-DECODE. Make Gemma 4 four-slot decode kernels exact and faster

Outcome: Gemma 4 26B-A4B joint decode at four slots spends less GPU time per step in its dense projections, routed experts and attention. Every output stays bitwise equal to today's decode.

Requirement: mlx-native MUST provide exact row-tiled dense, expert-grouped and GQA-grouped attention decode kernels under a stated contract (Q1 Class A against the stock kernels). The program MUST integrate them into the Gemma batched decode body behind a kill switch, and MUST measure the per-step GPU and makespan saving against that switch.

Acceptance:

- Starting point (Q4-GEMMA #660, binary `7f295b38`, mlx-native 0.15.16):
  - The decode body splits as follows: routed MoE 40% (gate/up 21.8%, down 15.3%), dense projections and FFN 37.5%, attention 17.8%, norms 4.5%.
  - In decode-only windows, hf2q is GPU-busy 2.0–2.5 s per run longer than the peer's MT configuration. Part of that gap is MT's speculation (2.34 tokens per verify step, i.e. fewer target steps), not kernel speed; #660 sized decode kernels at 0.3–0.7 s.
  - At width 4, the Q4_K/Q6_K dense projections move their weights at 155–184 GB/s. Per element-column they already match or exceed the expert kernels (1.0–1.19 T against 0.83 T per second), so the exact lever is capped near 2×.
  - Dense `ffn_down` (Q5_0/Q8_0) streams its weight once per row.
  - Both engines stream each (token, expert) pair separately, although the four rows share about 35% of their expert selections (Q1 witness proxy).
  - Decode attention loads and decodes each KV row once per query head: 2× on sliding layers, 8× on global layers.

- Design approved before implementation: `docs/research/adr063-q4-gemma-decode-design.md`, revision 2: §3 (exactness contract), §4 (S0 spike; levers D1, D2, E1, A1/A2; optional F1/F2; policy and refusal rules), §7 (M0 thresholds), §8 (gates). Approved by the operator, 2026-11-04, with the design §13 decisions 1–5.

- A linked mlx-native issue states the §3 contract before implementation. The kernels are:
  - row-tiled Q4_K/Q6_K `mN`, with the lm head decided independently;
  - Q5_0/Q8_0 `mN`;
  - expert-grouped `mv_id` for Q4_K gate_up and Q5_0/Q8_0 down, admitting up to 16 tokens × top-8;
  - GQA-grouped and mixed-bucket vec-hybrid decode attention.

  New per-lever routing-policy fields default off and are never resolved from the environment. Existing entry points stay bit-identical. Each new kernel is compiled with its reference kernel's flags.

- Exactness, verified in order:
  - A Q4_K R0 = 2 parity spike runs before the microbenchmark.
  - Primitive parity is u32-bitwise against the stock kernels at every Gemma shape, synthetic row tails, m = 1..8, all expert patterns (all-distinct, all-shared, empirical, invalid id), attention KV lengths across every bucket edge, and both metallib paths, with the toolchain recorded.
  - M0 uses cold-weight rotation and a host roofline. It reports weight GB/s and element-columns per second, uses the real-workload routing and KV-length distributions, and includes a peer per-kernel reference. Each lever is adopted only at its frozen threshold.

- hf2q switch and Class A gates:
  - `HF2Q_GEMMA_DECODE_KERNELS` (`0` is the kill switch) captures the environment routing policy at load. It refuses unless the reference flags hold.
  - A stock-equivalence test compares logits, tokens, KV, and expert ids and weights bitwise against `0`.
  - The slot-aware n1/n4/n8 and eviction parity tests pass.
  - The two arms are `ExactEqual` on the Q1 T2 cohort trace.

- Speed:
  - In situ, decode-step GPU at width 4 is ≤ 0.90 × the kill switch, with no width-1 regression and the lm head reported separately.
  - Four-agent Q5 A/B, same binary (ABBA, at least 4 runs per arm): median makespan ratio ≤ 0.96, block-paired 95% CI upper below 1.00.
  - `ExactTranscripts` holds across the A/B.
  - Worst-slot TTFT p95 and SSE-gap max ratios are ≤ 1.025.
  - The 0.96 closes this story's share of the Gemma Win only if the other Gemma stories reach their high estimates; otherwise about 0.93 is needed. Any shortfall is re-budgeted across the Gemma stories (operator preference, 2026-11-04).

- The default flips only after every gate passes. The Gemma verdict is then decided on the combined candidate in the Q4 closing campaign: Win against the peer's MT configuration, paired quality Pass against v0.1.28 and against the peer, stock fairness 1.025, and a fresh Q5.

Repository ownership: mlx-native owns the kernels, dispatchers, policy fields, contract, parity tests, microbenchmarks and release. hf2q owns route selection, the switch and refusal rules, the overlap diagnostic, integration, the Class A gates and the campaigns, consuming an exact published mlx-native pin.

Validation: Independent review of the contract, of the spike, parity and microbenchmark receipts (including the peer reference), of the Class A receipts, and of the A/B from raw receipts.

Predecessors: Q4-GEMMA (#660, profiling result), Q5 (#371). It is independent of Q4-GEMMA-KERNEL and Q4-GEMMA-ATTN. Its grouped expert and attention kernels serve Q4-GEMMA-SPEC-SERVE's verify rows.

Priority: High. Core scope (second-tier family).

Delivery boundary: A published mlx-native release pinned exactly by hf2q, plus the integration merged for v0.1.30 (or the first published hf2q release containing the accepted Gemma 4 configuration).

### Q4-GEMMA-SPEC-SERVE. Serve Gemma 4 with its assistant (MTP) drafter at four slots

Outcome: hf2q's Gemma 4 26B-A4B serving worker speculates with the Gemma 4 assistant drafter at four slots, as the peer's best configuration does, and finishes the canonical four-agent coding task measurably sooner than the same binary without speculation, with transcripts unchanged.

Requirement: The program MUST convert and load `google/gemma-4-26B-A4B-it-assistant@6e5aaaf4` as a paired drafter and MUST provide a joint four-slot draft/verify transaction in the Gemma serving worker whose verified rows, commits and rollbacks are Class A exact against the serial decode route.

Acceptance:

- A written design, approved before implementation, states the verify route, rollback, switch and kill switch: `docs/research/adr063-q4-gemma-spec-serve-design.md`. In it: the assistant comes from managed download pinned by sha256, with `--gemma-assistant` as the override; `HF2Q_GEMMA_SPECULATION=mtp` enables speculation; it is off by default until every gate passes, then on when an assistant is resolved; `0` is the kill switch; only greedy requests speculate; the depth K ∈ {1, 2, 3} is measured with K = 1 first. Approved by the operator, 2026-11-04, with the design §13 recorded decisions.
- The drafter loads only through a drafter contract paired to a `gemma4` target, behind a fail-closed pairing guard. `gemma4-assistant` stays out of the primary runtime contracts. The 26B-A4B checkpoint's head is the dense tied embedding (`use_ordered_embeddings: false`); ordered-embedding (centroid) assistants are refused, never approximated. The drafter's step-0 input is the target head's retained post-final-norm operand; the fused head stays on.
- The round-cost curve is measured on the verify route before any drafter runtime, with a frozen drafter-cost bound, a frozen acceptance model fitted to the peer, and a time-weighted width histogram. Gate G0: the projected decode saving at the chosen depth must be ≥ 0.40 s per run (the lower end of the Q4-GEMMA booked 0.4–0.5 s). G0 is re-checked, and the depth fixed, on hf2q's measured acceptance. A miss ends the story as a recorded falsification, and the windowed verify is removed unless a zero-cost drafter reuses it.
- Class A gates pass:
  - verified rows are bitwise equal to teacher-forced serial decode rows in every cohort;
  - ordinary rows are invariant;
  - after a round with *a* accepted drafts, the state readable at the cursor (full layers below it, the sliding ring's window, cursors) equals serial decoding of a+1 tokens, and so do the next eight decodes;
  - the per-token bookkeeping (valid tokens, cursor commit, scheduler advance, emit failure, finish and cancellation after restore) equals the ordinary ticks';
  - the drafter pass leaves the target state unchanged;
  - serving transcripts with speculation on and off are `ExactTranscripts` in every A/B block.
- Same-binary A/B under the ADR-065 two-arm rules (12 ABBA blocks): verdict Win, median makespan reduction ≥ 0.30 s and ≥ 60% of the M0 projection, decode stage −0.35 s or more, and paired TTFT/SSE fairness within 1.025 against the speculation-off arm. A fresh Q5 calibration of the target plus assistant artifact set passes.
- Every native primitive (the byte-exact ring-row restore with a source row offset) is published in mlx-native and consumed at an exact pin. The Q8_0 assistant artifact is published and pinned before promotion.

Repository ownership: hf2q owns the converter, drafter contract, serving transaction, policy, telemetry and campaigns. mlx-native owns the ring-row restore primitive, its parity tests and release.

Validation: Independent review of the design, the Class A receipts, the M0 round-cost curve and G0 decision, the acceptance replay and the A/B receipts.

Predecessors: Q4-GEMMA (#660, profiling result), Q5 (#371). Q4-GEMMA-ATTN's closing campaign consumes the accepted configuration.

Priority: High. Core scope (second-tier family).

Delivery boundary: The promotion commit merged to main for v0.1.30 and included in the Q4-GEMMA-ATTN combined candidate, or the recorded G0 falsification.

### Q4-GEMMA-ATTN-EXACT. Make Gemma 4 warm-prefill attention faster without changing any output bit

Outcome: Gemma 4 warm-prefill attention runs a batched kernel whose output is u32-identical to the vec-hybrid warm route, so the joint warm prefill is faster with no envelope or quality campaign.

Requirement: mlx-native MUST provide the batched warm-attention kernel under the exactness contract of `docs/research/adr063-q4-gemma-attn-exact-design.md` §3. Its R = 1 case is Q4-GEMMA-DECODE lever A1's `flash_attn_vec_hybrid_batched_gqa`; multi-row blocks are `flash_attn_vec_hybrid_batched_rows`. hf2q MUST route Gemma 4 warm-prefill attention through it only where it is proven u32-equal to `VecHybrid` (Class A, Q1 §3.1).

Acceptance:

- Design approved by the operator, 2026-11-05: `docs/research/adr063-q4-gemma-attn-exact-design.md`, revision 3 with ReviewGemmaExact's three P3 fixes (independent review ACCEPT).
- The S0 feasibility spike runs first (design §5), measuring R = 1 before any multi-row block.
  - S0-a: u32 parity of O, the NWG partials and the reduced output on both metallib paths over the full matrix.
  - S0-b: speed, with a weighted chain ratio ≤ 0.70 against the vec chain.
  - The kill criteria of design §9 are frozen. A failed S0 ends the story as a recorded falsification, with no Class B fallback.
- Gates X1–X6 of design §7 pass bitwise, each showing the new kernel ran:
  - X1 primitive parity;
  - X2 model route;
  - X3 joint == per-slot;
  - X4 envelope harness with `d_nat` = 0 on every row;
  - X5 Q1 `ExactEqual`;
  - X6 Q5 `CalibrationPass` with `ExactTranscripts` across arms.
- M0 (design §9): the median per-run joint-warm saving is ≥ 0.25 s over matched transactions, and makespan does not regress by the paired ABBA rule (≥ 5 runs per arm).
- The kernel is published in an mlx-native release, verified on crates.io and pinned exactly by hf2q (design §10). The native contract issue is shared with Q4-GEMMA-DECODE lever A1.

Repository ownership: mlx-native owns the kernel, contract, parity tests and release; the shared primitive file is owned by Q4-GEMMA-DECODE's implementer, and this story adds its own warm-prefill tests and bench. hf2q owns the route, the kill switch `HF2Q_GEMMA_WARM_ATTENTION_EXACT`, the gates and the A/B.

Validation: Independent review of the S0 receipt, the native release, the X1–X6 receipts and the M0 A/B.

Predecessors: Q5 (#371). For its R = 1 case it also depends on Q4-GEMMA-DECODE's native lever A1 (shared contract and release).

Priority: High. Core scope (second-tier family).

Delivery boundary: A published mlx-native release pinned exactly by hf2q, plus the promotion commit merged to main for v0.1.30, or the recorded S0/M0 falsification.

Result (2026-11-05, design §12):
- S0-a passed on both metallib paths for R = 1 and for every multi-row (G, R).
- S0-b: the best exact configuration (G = 2, R = 1) has a chain ratio of 0.761 precompiled and 0.764 runtime (sliding 0.737, global 0.817) against the frozen 0.70.
- The story as designed stops and the multi-row kernel is dropped.
- The operator adopted the narrowed follow-up Q4-GEMMA-ATTN-EXACT-G2.

### Q4-GEMMA-ATTN-EXACT-G2. Route Gemma 4 warm-prefill attention through the bitwise G = 2 grouped kernel

Outcome: Gemma 4 warm-prefill attention runs mlx-native `flash_attn_vec_hybrid_batched_gqa` with G = 2 heads per threadgroup. Its output is u32-identical to the vec-hybrid warm route, and the four-agent makespan improves measurably.

Requirement: hf2q MUST add the bitwise warm kernel `VecHybridGqa`, specified in `docs/research/adr063-q4-gemma-attn-exact-design.md` §13:
- The kernel is the A1 `flash_attn_vec_hybrid_batched_gqa`, with G capped at 2, R = 1 and the chunk's stock bucket on every row (no per-slot buckets).
- It sits in the warm-prefill `VecHybrid` route, behind the reachable selector and kill switch `HF2Q_GEMMA_WARM_ATTENTION_EXACT`.
- Every `GemmaLiveKernel` dispatch point MUST stay exhaustive.
- Installed `Tiled` lanes MUST stay stock.

Acceptance:

- Design approved by the operator, 2026-11-05: `docs/research/adr063-q4-gemma-attn-exact-design.md` §13, revision 2 (ReviewGemmaExact ACCEPT).
- Before integration, the capped-A1 S0-b receipt runs on both metallib paths (design §13.0). It reports per-kind ratios and cells more than 2% slower, and it freezes the per-(kind, bucket) stock-fallback table (ratio > 1.02 keeps stock). The expected saving is restated from that receipt. If capped A1's chain ratio is more than 0.02 above the rows kernel's 0.761, the story returns to the operator.
  - Result (2026-11-05, design §13.0): X1 and T-N8 pass on both paths. Capped A1's chain ratio is 0.791 (saving 299.6 ms of chain per run) with an empty fallback table. In a same-run head-to-head the rows kernel was only 1.2% faster (0.762 against 0.771), so capped A1 is adopted.
- Gates X1–X6 of design §13.3 pass bitwise, each with dispatch-count proof that the kernel ran:
  - X1: the mlx-native warm suite and T-N8, both metallib paths;
  - X2: model route;
  - X3: joint == per-slot;
  - X4: envelope harness, `d_nat` = 0 on every row;
  - X5: Q1 `ExactEqual`;
  - X6: Q5 `CalibrationPass` with `ExactTranscripts` across arms.
- No timed run sets `MLX_DISP_BUCKET`.
- Class A holds everywhere. Any non-bitwise result keeps the switch off; the route is never relaxed to an envelope.
- Speed: in a paired ABBA A/B of the switch (≥ 5 runs per arm), the per-pair makespan difference has a 95% CI that excludes 0 in the saving direction. There is no fixed size bar. The matched-transaction joint-warm saving is reported. Dispatch-count proofs count warm-prefill dispatches only (counters reset before the warm forward and read before decode). Not promoted means clean-cutover removal of the variant and its switch.
- The pinned mlx-native release contains #89 (0.15.20), verified on crates.io and pinned exactly by hf2q.

Repository ownership: mlx-native owns the kernel, its contract, the warm parity suite `tests/gemma_warm_attn_exact_s0.rs` (X1, landed with #89) and the release. The primitive file is owned by Q4-GEMMA-DECODE's implementer. hf2q owns the route, the selector and kill switch, the gates and the A/B.

Validation: Independent review of the X1–X6 receipts and the paired A/B.

Predecessors: Q4-GEMMA-ATTN-EXACT (S0 result), Q4-GEMMA-DECODE's native release containing #89 (mlx-native 0.15.20), and Q5 (#371). X6 also waits for the four-agent harness normalisation (2026-11-05).

Priority: High. Core scope (second-tier family).

Delivery boundary: hf2q pinned to the release containing #89, with the promotion commit merged to main for v0.1.30, or the recorded failure of the exactness or speed rule.

### Q4-GEMMA-LMHEAD-MMA. Run the Gemma 4 lm head through a few-row MMA kernel (Class B)

Outcome: The Gemma 4 lm head (Q6_K, 2816 → 262,144) at decode widths 2..16 costs about weight-streaming time instead of growing per column. Batched decode stays bitwise equal to serial decode at every width, and the kill-switch arm is bitwise today's serving.

Requirement: mlx-native MUST provide a few-row MMA Q6_K head kernel whose output for each column is bitwise independent of the other columns and of the width. The program MUST route the serving decode heads (serial and batched) through it behind a kill switch, at the head call sites only, refusing speculative-verify and kernel-profile modes under the switch, and MUST pass the Q1 Class B gates against stock before promotion.

Acceptance:

- Starting point (Q4-GEMMA-DECODE M0, mlx-native `718e811`, cold weights): stock head 1,088 / 2,152 / 4,306 µs at widths 1 / 4 / 8. The peer: 1,062 µs at width 1 (its mat-vec; MMA starts at two rows), MMA 1,181 / 1,208 µs at widths 4 / 8 (M0-P, `e86f5fd1`). The exact alternative (Q4-GEMMA-DECODE D1) was 1.6× slower and was dropped. The V2 workload has 403 of 471 decode steps at width 4 and a 13.9 s four-agent makespan: about 0.40 s of decode GPU per run at the peer's MMA times [INFERENCE].

- Design approved before implementation: `docs/research/adr063-q4-gemma-lmhead-mma-design.md` revision 2 (§3 contract, §5 spike rule, §6 gates, §11 release and removal set, §12 decisions). The story was opened by the operator on 2026-11-05 under Q4-GEMMA-DECODE design §13 decision 2(b). The MMA reorders each logit's accumulation the way the peer does; that its output stays within rounding of today's is [INFERENCE] until the spike's error record. The Class B gates are the program's standing nothing-broke checks for a reordered route, not a claim about distance from the peer.

- Kernel tests, both metallib paths, toolchain recorded: functional correctness against a CPU F64 reference with a K-scaled bound; determinism; column independence and width invariance (m = 1..16, column groups of at most 8, every tiling, NaN/∞ neighbours, permutation); views and row tails; an element-wise error record against stock (recorded, not gated).

- Spike rule, model-free, before integration: the V2-weighted projected saving including width 1 is ≥ 0.30 s per run, and the width-1 penalty is ≤ 35 µs (the width-1 head penalty must cost under 0.5% of the n1-stage makespan; at M1 the bar is min(35 µs, 0.005 × measured n1 time per decode step)). The width-1 leg is predicted to fail at the peer's MMA efficiency. A miss stops the story and the code is removed.

- Gates in order, any failure stops:
  - Class A at serving level: slot-aware n1/n4/n8/eviction parity under `0` and `1`, a ≥ 256-step batched-versus-serial MMA test including widths 9..16, both `HF2Q_FUSE_LMHEAD` settings, and `ExactTranscripts` of `0` against the pre-integration pin.
  - Q1 decision envelope on the greedy144 T2 cohort trace (τ = 0.25 nat, every row; 64 of its 576 rows exercise the head). `RouteSensitive` stops the story as a kernel defect.
  - M1: the spike's projected-saving formula on in-situ lm-head GPU per step is ≥ 0.30 s per run; width-1 penalty within the spike bar at N = 1.
  - M2: four-agent Q5 A/B, ABBA, 4 runs per arm (extended once to 8 if the CI half-width exceeds 2.5 points): median makespan ratio ≤ 0.98 against the 13.9 s baseline with block-paired 95% CI upper below 1.00; `ExactTranscripts` of arm `1` run to run; worst-slot TTFT p95 and SSE-gap max ratios ≤ 1.025.
  - Paired quality against the same binary at `0` (equal `HF2Q_GEMMA_DECODE_KERNELS`): 24 instances, `q1_decision_policy.py paired` = `Pass`, no reruns.

- Switch `HF2Q_GEMMA_LMHEAD_MMA` (`0` default kill switch, `1`, anything else a startup error), independent of `HF2Q_GEMMA_DECODE_KERNELS`; shared dispatchers byte-unchanged; prefill heads stock in both arms. The default flips only after every gate passes; the Gemma verdict is then decided on the combined candidate in the Q4 closing campaign. On any stop the removal set of design §11 applies in both repositories.

Repository ownership: mlx-native owns the kernel, its invariance tests, the error record, the microbenchmark and the release. hf2q owns the switch, the serial and batched head routing, the Class A and Class B gates and the campaigns, consuming an exact published mlx-native pin.

Validation: Independent review of the design, the spike receipts (invariance, error record, M0 cells), the envelope and paired-quality receipts, and the A/B from raw receipts.

Predecessors: Q4-GEMMA-DECODE (M0 and M0-P results), Q1 (numeric policy and paired-quality tooling), Q5 (#371), the four-agent harness normalisation and the cargo-1.99 sandbox proof (2026-11-05).

Priority: High. Core scope (second-tier family).

Delivery boundary: A published mlx-native release pinned exactly by hf2q, plus the integration and promotion merged for v0.1.30, or the recorded failure of the spike rule or a gate.

Outcome (2026-11-05): **stopped at the spike.** Exactness passed on both metallib paths, but both speed legs missed: the projected V2 saving was 0.237 s (bar ≥ 0.30 s) and the width-1 penalty was +103 µs (bar ≤ 35 µs). This is the recorded failure of the spike rule; see design §14.

### Q4-DS4. Run the peer on the ds4 DeepSeek-V4 file and determine fit

Outcome: Whether the same-GGUF peer can run ds4's DeepSeek-V4 file on the 128 GB host is established, and the peer's per-stage behavior on it is recorded.

Requirement: The program MUST establish whether the peer fits on this host with ds4's DeepSeek-V4 file and, if it fits, MUST record a per-stage attribution against hf2q.

Acceptance:

- Starting point (Q3 v7): Unavailable; the same-GGUF peer runs out of memory on hf2q's DeepSeek-V4 artifact. On different artifacts hf2q `38485a23` finishes 5.2% sooner than ds4 (never a Win).

- Run the peer alone on `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (86,720,111,488 bytes, sha256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`, re-hashed locally 2026-11-02; it sits in the ds4 peer directory `peers/ds4-c97964b/gguf/` under the hf2q data directory, not the `/opt/ds4/gguf/` path in `docs/research/adr063-q3-campaign.md`): `deepseek4` architecture, the same 1,328 tensors, IQ2_XXS/Q2_K experts. Record fit or no-fit with Q5 live-stop receipts, one runtime at a time.

- Fit = the peer, in its best valid configuration, completes the largest declared four-agent workload (4 slots × 262,144 context, matched settings) with Q5 `CalibrationPass`. Any context or slot reduction is an operator decision (Q5).

- A no-fit result closes this story, keeps DeepSeek-V4 Unavailable with that evidence and returns the family to the operator as `needs_decision`; Q4-DS4-RACE does not start.

- If it fits: a per-stage profile of the peer on that file (prefill, decode, MoE/attention, dispatch, scheduling, SSE cadence) set against hf2q's per-stage profile on its DeepSeek-V4 artifact at the same workload, with stated distortion and the artifact difference disclosed. The same-file attribution follows in Q4-DS4-RACE.

- The file is a benchmark comparison artifact, not a substitute for hf2q conversion.

Repository ownership: hf2q owns the peer run, Q5 receipts, profiles and the fit record.

Validation: Independent review of the fit receipt and the per-stage profiles.

Predecessors: Q5 (#371).

Priority: High. Core scope (first-tier family).

Delivery boundary: Measured evidence merged to main for v0.1.30.

### Q4-DS4-RACE. Support the ds4 file in hf2q and race hf2q, the peer and ds4 on it

Outcome: hf2q finishes the four-agent task on ds4's DeepSeek-V4 file at least 5% sooner than the peer in its best valid configuration.

Requirement: The program MUST support the IQ2_XXS ds4 file in hf2q by consuming ADR-066 N-IQ2XXS and MUST compare hf2q, the peer and ds4 on that one file.

Acceptance:

- A written design, approved before implementation, states the loader, tensor mapping and quality checks for this file. Approved by the operator, 2026-11-03: `docs/research/adr066-iq2xxs-subset-design.md` (loader changes L1–L9, no hf2q kernel; quality checks §6; race plan §7 under Q4-DS4-RACE decisions D1/D2; harness changes §8).

- Starting point (Q4-DS4, #661): Fit. Six of seven peer runs were valid (f1 was `ClientFailed` and ran no model requests). By the declared makespan rule the best valid configuration is load mode none with ubatch 1024 (median 65.2 s). It is provisional and within noise of auto/1024, so before the verdict of record this story re-runs none/1024, auto/1024 and none/512 on this file (evidence `/Users/robert/.local/state/hf2q/v0130/q4-ds4/`).

- Consume the IQ2_XXS storage type, CPU dequantization, grids and expert kernels of ADR-066 N-IQ2XXS, which gains parity at DeepSeek-V4-Flash expert shapes (256 experts, top-6) and proves the Q2_K expert kernels at the down shape; no duplicate kernels. ADR-066 N1/N3/N4 build on N-IQ2XXS and are not predecessors. Add the hf2q loader for this file and the quality checks (tensor-load parity, real-weight kernel parity, joint == serial bitwise on this file, paired task-quality gate).

- Fresh Q5 calibration of hf2q on this file. The verdict of record runs hf2q (experiment) and the peer on this file under the Q3 design, with no stock arm, bound to a same-artifact candidate-vs-v0.1.28 campaign on hf2q's own artifact for the vs-stock contrast, fairness band and stock quality gate (Q4-DS4-RACE decision D1, operator, 2026-11-03; Q4 DeepSeek-V4 exception). ds4 runs on the same file in a separately labeled native-comparison campaign that declares the shared artifact (Q4-DS4-RACE decision D2, operator, 2026-11-03); it never enters the verdict.

- A per-stage attribution of the hf2q-versus-peer gap on the same file and run (prefill, decode, MoE/attention, dispatch, scheduling, SSE cadence), with stated distortion.

- The family verdict meets the Q4 win definition and every Q4 gate, with the vs-stock terms supplied by the bound same-artifact record. The paired task-quality gates are against the peer on this file and against v0.1.28 on hf2q's artifact.

- A Q4-DS4 no-fit would have stopped this story before any hf2q change; only the mlx-native N-IQ2XXS work is ADR-066 scope independent of this story.

Repository ownership: mlx-native owns the IQ2_XXS type, grids, kernels and their parity tests through ADR-066 N-IQ2XXS; hf2q owns the loader for this file, quality checks, the harness changes, Q5 calibration and campaigns.

Validation: Independent review of the design, the parity and quality receipts, the bound same-artifact stock record, the ds4-file campaign of record and the same-file native comparison.

Predecessors: Q4-DS4 (#661, fit result), ADR-066 N-IQ2XXS (pulled ahead of ADR-066 B2, operator decision 2026-11-03), Q5 (#371).

Priority: High. Core scope (first-tier family).

Delivery boundary: Release v0.1.30, or the first published hf2q release containing the accepted DeepSeek-V4 configuration.

### Q4-QWEN36. Investigate the Qwen3.6 task failure, then choose its speed workload

Outcome: The cause of the Qwen3.6 task failure is understood against Qwen's recommended agentic setup, and Qwen3.6 has a declared speed workload on which a Q4 verdict can be measured.

Requirement: The program MUST investigate the Qwen3.6 failure before choosing a speed workload and MUST then deliver a Qwen3.6 configuration whose Q4 verdict is Win on that workload.

Acceptance:

- Starting point (Q3 v10): Failed; the experiment and v0.1.28 both fail the canonical task (E0282, no repair turn), diagnosed as coming from the model weights (`docs/research/adr063-qwen36-task-failure-diagnosis.md`).

- The check used below is the 24-instance paired task campaign (Q1 numeric policy §3.5, `q1_decision_policy.py paired`) as in `adr063-qwen36-task-failure-diagnosis.md`, reporting instances passed per arm and the floor verdict.

- Diff the harness against Qwen's recommended agentic setup, with sources: thinking mode, whether past reasoning is resent, system prompt and tool-call format. Re-run the check with the recommended setup on the peer and on hf2q.

- Download third-party vanilla Q4_K_M GGUFs (ggml-org, unsloth) as diagnostic data only, not as a conversion substitute or a release artifact, and run the check on the peer and on hf2q with them. Operator decision, 2026-11-03: the first third-party check is unsloth UD-Q4_K_M under the canonical setup on both engines.

- Conversion audit (2026-11-03): no defect found in hf2q's Qwen3.6 artifact against the BF16 source and an independent converter (evidence `~/.local/state/hf2q/v0130/q4-qwen36/conversion-audit/report.txt`).

- If Qwen3.6 stays BelowFloor under every setup tested, Q4-QWEN36 returns to the operator as `needs_decision` with that evidence.

- Qwen3.6 race workload (operator decision, 2026-11-03; result recorded 2026-11-05). Qwen3.6 is raced on the canonical four-agent coding task with one repair cycle (`--repair-turns 1`). After a failed `run_tests`, an agent gets exactly one more `write_file` and one more `run_tests` on the same target; a second failure fails the agent. Every other step of the six-step workflow, the prompts, tools, template, sampling and thinking budgets are unchanged. The race runs on the hf2q-converted GGUF (`815a22ca…`), identically on hf2q and the peer (`q5_cases/qwen36-repair1.json`, `peer-qwen36-repair1.json`); the approval binds the grant, and the Q2 run audit replays it.
  - Rationale: under the strict workflow Qwen3.6 is BelowFloor on every engine, precision and conversion tested (hf2q 5/24 and peer 4/24 on our GGUF, Q8_0 3/24, unsloth UD-Q4_K_M 1/24 on both engines). The failure is one first-write idiom, `map_err(|e| e.to_string())` (E0282), which grows more frequent with precision (Q4 22 → Q8 34 → BF16 41 of 48 replays). Failing agents usually write a type-correct fix on the next turn, which the strict workflow refuses; a repair cycle measures the engines on a task Qwen3.6 can complete without changing the model, artifact or sampling.
  - Scope: Qwen3.6 only (`q5_case.REPAIR_FAMILIES`). Its results are not comparable to the other families' canonical task (pass rates, up to 9 requests per agent instead of 7, makespans). Qwen3.8, Gemma 4 and DeepSeek-V4 keep the strict workflow.
  - Defect found and fixed on this workload: the first repair-turn campaign was an hf2q `Loss` (14/24 vs peer 23/24). The same-prefix fork replay and a layer-by-layer comparison against the peer and the BF16 source localized it to mlx-native `rope_multi`, which paired partial-rotary RoPE dimensions `(p, p + head_dim/2)` instead of `(p, p + rope_dim/2)` on every Qwen3.5-family full-attention layer; fixed in mlx-native 0.15.18. The Qwen tool grammar also dropped the template's newline after `<tool_call>`; fixed in #686. After both fixes hf2q matches the peer within KV-precision noise on the 63 copy/fix decision points (`docs/research/adr063-qwen36-recommended-setup.md`).
  - Result: the 24-instance paired task campaign on fixed main (`4bfb0436`, rustc 1.99.1) is hf2q 21/24 vs peer 20/24, paired verdict `Pass` (one-sided p 0.81). The floor is cleared, so this is the declared Qwen3.6 workload of C2's quality rule, and the Q4 speed race runs on it. The Splash Qwen3.6 arm (D9) runs it too: #715 extends the repair grant to Splash for Qwen3.6 (`q5_cases/splash-qwen36-repair1.json`).

- Record the workload decision with this evidence; changing the canonical workload is an operator decision recorded in this ADR. On that workload, a per-stage attribution of the hf2q-versus-peer gap on the same run (prefill, decode, MoE/attention, dispatch, scheduling, SSE cadence) is recorded with stated distortion, and the family verdict meets the Q4 win definition (against the peer and against Splash) and every Q4 gate. Splash runs the same declared Qwen3.6 workload, the repair-turn workload (Splash decision D9; harness support in #715).

Repository ownership: hf2q owns the harness diff, templates/tool format, the checks and the campaign.

Validation: Independent review of the setup diff, both checks and the subsequent Q3-design campaign.

Predecessors: Q5, the Splash Qwen3.6 fit and best-configuration record.

Priority: High. Core scope (second-tier family).

Delivery boundary: Release v0.1.30, or the first published hf2q release containing the accepted Qwen3.6 configuration.

### Q4-TRACE. Add the Qwen joint-decode forward witness

Outcome: Every Qwen joint ordinary decode forward is attributable to its owning slots in a trace, for profiling now and for the later Q2 v3 proof.

Requirement: The program MUST deliver a Qwen joint-decode forward witness/trace hook that maps each joint ordinary decode forward to its owners.

Acceptance:

- `scripts/four_agent/f2_native_trace.py` reports `unmapped_operator_attempts == 0` for every `joint_ordinary_decode` forward of a full traced four-agent Qwen3.8 run (the Q2 closure QE §6 found 43,008 per run, as `forward_witness_unmapped` records with reason `NoActualInputManifest`).

- The hook is off by default and fails closed; with it on, transcripts are `ExactTranscripts` against it off, and its timing overhead is measured and stated.

- Its output is consumable by Q4-QWEN38 attribution; the Q2 v3 counting rule that consumes it stays successor scope (v0.1.31).

- Design approved by the operator, 2026-11-03: `docs/research/adr063-q4-trace-design.md`, with its three §7 answers: new report keys only under the new policy; padding declared as owner-less physical rows; transaction binding by record order with the epoch freshness and consistency check for v0.1.30, an epoch field deferred to Q2 v3.

Repository ownership: mlx-native owns low-level trace support; hf2q owns the witness wiring, records and validation.

Validation: Independent review of a traced run with `unmapped_operator_attempts == 0` and the trace-on/off exactness control.

Predecessors: Q5.

Priority: High. Core scope.

Delivery boundary: Qualified implementation merged to main for v0.1.30.

Result (2026-11-06, M5 Max 128 GB, PR #677; evidence `~/.local/state/hf2q/v0130/q4-trace/`): **Exact; acceptance met.** Binary H `f33f9fa1` (sha256 `ecff6960…`) against its merge-base P `1336f7e0` (`f546d149…`), both built with rustc 1.99.1, on the official Qwen3.8-27B artifact and the n4 case `q5_cases/qwen38.json`.
- **Witness coverage.** In the traced n4 run T1, every one of the 1,671 joint ordinary decode forwards is witnessed and none is unwitnessed. `unmapped_operator_attempts` is 0. The width mix is 1,472 four-lane, 74 three-lane, 75 two-lane and 50 one-lane steps; the one-lane steps come from the #717 solo route. The parent's traced run T0 recorded 80,352 unmapped attempts.
- **Exactness.** G1 passes: on a fixed schedule, every width's output is bitwise the same with the hook armed as unarmed. Transcripts are `ExactTranscripts` (28 turn positions) in all five pairs: T1 O1, T0 O0, the determinism control O1 O1′, T1 T0 and O1 O0. The classification under the design's pre-registered rule, as amended before any run with the O1′ determinism control, is `Exact`.
- **Q2 isolation.** Q2 v2 window outputs on T1 are identical with and without the joint records. All four frozen windows are `Unqualified` in both shapes, so the synthetic isolation control P5 is the test that would catch a leak.
- **Overhead (trace on).** Per joint epoch the hook adds a median +0.64 ms at four lanes (bootstrap 95% CI 0.34–0.86 ms) on an epoch of about 86 ms, and +1.2 to +2.1 ms at one to three lanes. It cuts about 45 trace records per joint epoch.
- **Operator waiver (2026-11-06): no speed tests.** The off-default O1 vs O0 makespan comparison of design §5.3 is dropped. The untraced runs serve only the exactness classification.
- **Validation.** The traced run was reviewed independently (Review677, ACCEPT on acceptance item 5).
- **Attribution.** The per-phase and per-pipeline attribution of T1 for Q4-QWEN38 and the gap program is recorded as `analysis/attribution-T1.json` (`q4_trace_attribution.py`). The phase trace carries no per-kernel or per-command GPU time.

### Q4-TRACE-GPU-TIMING. Per-command GPU timestamps in the diagnostic phase trace

Outcome: A traced four-agent run attributes GPU time to each native command, so each joint-decode layer chain, each prefill command and the bubbles between them have a measured GPU duration. The Q4-TRACE attribution could only split a joint step into host encode (about 7 ms) and an unattributed completion tail (about 79 ms).

Requirement: The program MUST record mlx-native's per-command GPU timing stream in the diagnostic phase trace behind an explicit opt-in. It MUST validate the stream fail closed, and MUST attribute GPU time per joint chain and per phase without assuming a clock mapping or a command layout.

Acceptance:

- With `native_observer.gpu_timestamps` absent, configs, headers, traces and reports are byte-identical to today. The flag is accepted only for a diagnostic case, and an explicit `false` is refused.
- Timing events travel in the same ordered queue as native events. The reducer checks their streaming order: after a status-4 drain and before `Dropped`, exactly one per command. It also checks a closed schema with reason-to-cause rules, and clock-free duration bounds (per command against its host submit-to-drain bracket, per epoch against the host span).
- Chains are mapped to layers from a published `joint_chain_plan` derived from the code's chain plan. An epoch whose command count or dispatch signature does not match is refused for per-chain attribution, never mapped silently.
- G2 passes: on a fixed schedule, with the stream installed, joint forwards are bitwise equal armed vs unarmed. On one new binary, a traced n4 run with the flag on is `ExactTranscripts` against the same binary with it off. The timing-event and trace-byte deltas are stated.
- `q4_trace_attribution.py` reports per-chain GPU time by logical width, GPU bubbles, and prefill GPU time. It is handed to the Qwen op-parity lead. Per-kernel time stays out of scope.

Design: `docs/research/adr063-q4-trace-gpu-timing-design.md` (revision 3). Main records the design approval on the story issue on Review677's ACCEPT, per Main's instruction of 2026-11-06. If operator approval is required, the approval event names the operator.

Repository ownership: mlx-native already owns the timing stream (0.16.0, no change). hf2q owns the opt-in, sink, records, reducer and attribution.

Validation: Independent review of the traced run (T1g) and of the G2 result. Correctness runs only, per the operator's 2026-11-06 no-speed-test directive.

Predecessors: Q4-TRACE.

Priority: High. It serves the gap program's attribution (Q4-QWEN38 and the Qwen op-parity work).

Delivery boundary: Qualified implementation merged to main for v0.1.30.

Result (2026-11-06, M5 Max 128 GB, PR #732; evidence `~/.local/state/hf2q/v0130/q4-trace-gpu-timing/`): **acceptance met.** Binary `433e2ca9` (sha256 `7173b547…`), built with rustc 1.99.1, on the official Qwen3.8-27B artifact and the n4 case. Later commits on the PR change only docs.
- **Neutrality.** G2 passes: with the stream installed, joint forwards at all four widths are bitwise equal armed vs unarmed, with 18 `Available` timings per armed forward. G1 also passes on this binary. T1n (flag off) and T1g (flag on), run back to back on the same binary, are `ExactTranscripts` (28 turn positions).
- **Coverage and checks.** T1g reduces: 41,052 commands are all `Available` (0 `Unknown`). Every epoch passes the clock-free bounds (1,726 epochs), with 0 commit-order inversions. The Q4-TRACE witness acceptance holds in both runs (1670/1670).
- **Cost.** +41,052 native events (8 M limit) and +15.6 MB of trace (6 GiB server budget). The design estimated about 10.7 MB; the records are larger than estimated.
- **Attribution.** Every joint epoch matched the published plan (64 layers, `chain_n` 4, native embedding: 18 commands), so `chain_map_refused` is 0.
  - At 4 lanes, a step's GPU is busy 87.9 ms of an 89.1 ms median epoch, with bubbles of about 0.5 ms.
  - Each of the 16 four-layer chains (3 DeltaNet + 1 full attention) takes 5.14–5.29 ms, and the output head 3.6 ms.
  - Width costs little: chains take 4.7–5.1 ms at 1 lane.
  - Per-kernel time remains unobservable.
  - Handed to the Qwen op-parity lead (`attribution-handoff.md`).
- **Validation.** Review677 approved #732; there were no speed tests, per the operator's 2026-11-06 directive.

### Q1-DS4. Capture DeepSeek-V4 Q1 endpoints on this 128 GB Mac (#549)

Outcome: The deferred DeepSeek-V4 Q1 numerical endpoints run on the M5 Max 128 GB under a measured, reviewed resource policy.

Requirement: The program MUST replace the borrowed serving-census preload floor with a measured Q1 test-run footprint and an independently reviewed new resource policy, then capture the DeepSeek-V4 Q1 endpoints.

Acceptance:

- The borrowed 119.18 GiB preload floor (`$E/q1-closure-v1/README.md` §4.2) is replaced by a measured Q1 test-run footprint and a new resource policy that an independent reviewer accepts before the endpoint runs. If the measured footprint does not fit, the result is recorded as not runnable with evidence.

- The footprint measurement run uses Q5 live stops, with the effective floor as its cap.

- The new policy reconciles the Q1 supervisor's normal-pressure-at-every-sample rule with Q5, where warning pressure is not a stop; every DeepSeek serving run here reached `warning` (README §4.2).

- Run on the served artifact `DeepSeek-V4-Flash-0731-hf2q-deepseek4-agentic-q2.gguf` (sha256 `50df7e3dfe06321613c78075df4111b14985a0db31087992d7438f2895a3aad6`) with a Q1 test binary built from main at or after `ea6caf46` (native 0.15.16); the composition is authored immediately before the family block.

- `exact` T1 vs T3 is `ExactOnCommon` with restored-64 extras only and the receipt `full_active64_exact_restore {"Ok":n>0}`; `exact` T4 vs T0 and T7 vs T0 are `ExactEqual`.

- `envelope` T0 vs T3 is classified; a `RouteSensitive` result is localized with `deepseek_localizer`.

- Routing margins are Missing-with-reason (no published native DeepSeek selection witness); task outputs reuse `ExactTranscripts` plus the paired 20-vs-20 Pass.

- An independent reviewer accepts the DeepSeek-V4 verdict.

Repository ownership: hf2q owns the Q1 adapters, supervisor, resource policy and verdict; mlx-native owns operation-level parity and trace support.

Validation: Independent review of the footprint measurement, resource policy and endpoint receipts.

Predecessors: Q1, Q5.

Priority: Medium. Successor scope; does not gate Q4.

Delivery boundary: Measured evidence merged to main for v0.1.30.

### Q1-GEMMA-T2LOC. Localize the Gemma packed T2 divergence by stage (#652)

Outcome: The first stage at which the Gemma packed (T2) route diverges from All128 is localized for all four lanes.

Requirement: The program MUST deliver a localization record for the Gemma packed-versus-All128 divergence.

Acceptance:

- Prefill capture maps the packed layout (512 rows at `lane*128`) through `gemma_endpoint::packed_owners` to per-agent coordinates.

- New `#[cfg(test)]` stage hooks in the batched decode body (`batched_body.rs`) cover the decode stages; the existing single-row hooks are not reused as a substitute.

- All128 reference stores exist for agents 0–3 on the same build (build of record b3 `91b0c382`; stores `gemma-t0-b3`, `gemma-t2-b3`).

- The Q1 endpoint supervisor change for four-lane localizer cases passes an admission review, with a composition/acquisition review and a four-lane resource policy.

- Observer-effect control: the localizer's outputs at the captured cache and logit coordinates are `ExactEqual` to `gemma-t2-b3` (packed) and `gemma-t0-b3` (All128).

- Packed versus All128 is a Class B route, so no fix is implied; the repeated-token prefix in rows 29–127 of greedy144 is disclosed with the magnitude.

Repository ownership: hf2q owns the hooks, supervisor and localization record; mlx-native owns low-level trace support.

Validation: Independent review of the supervisor admission, the observer-effect control and the localization record.

Predecessors: Q1, Q5.

Priority: Medium. Successor scope; does not gate Q4.

Delivery boundary: Measured evidence merged to main for v0.1.30.

Result (2026-11-04): Localized. The packed route first diverges from All128 at layer 0, prefill `Attention` (the SDPA output), in all four lanes and on all 128 prompt rows. Layer-0 prefill Q/K/V and RoPE outputs are exact. At the observed sliding layers 0–1, batched decode is bit-exact given equal inputs (decode layer 0 is exact), and decode layer 1 first differs at `Attention`, which reads the prefill-written KV cache. Full-attention layers were not observed in decode. Every observer-effect control is `ExactEqual` (5,616 records each): against `gemma-t0-b3`, against `gemma-t2-b3`, and against a same-build T0. The routing bank reproduces the b3 T0/T2 routing decision byte for byte. The repeated-token prefix (rows 29–127) diverges at the same stage with a slightly lower magnitude. Class B, so no fix is implied. Build of record: the PR #675 merge commit `b0dac052` instead of b3 `91b0c382` (root decision, 2026-11-03; bridged to b3 by the observer-effect control). Evidence: `docs/research/adr063-q1-gemma-t2-localization.md`.

### Q1-QWEN38-RERUN. Re-run Qwen3.8 Q1 T1/T3 on a build after #561

Outcome: The Qwen3.8 restore-versus-cold Q1 evidence is current for the restore cursor fix.

Requirement: The program MUST re-run Qwen3.8 T1/T3 on a Q1 test binary built after #561.

Acceptance:

- Runs use a Q1 test binary from main at or after `ea6caf46` (the retained build `1d200821` predates #561, `9fc7b31b`) and before ADR-065 P4b (#527) or J1 (#512) merges (P4b changes solo K3 rollback). Otherwise pin a commit that precedes both and record that relation.

- `exact` T1 vs T3 is `ExactOnCommon` with restored-64 extras only.

- Amended 2026-11-04: the delivered result (restore exactness on `d817dffa`/`5504f4ee`, mlx-native 0.15.16) stands as invariance evidence, because state-restore exactness does not depend on the `rope_multi` numerics (see "Gemma 4 gap stories and the rope_multi pin").

Repository ownership: hf2q owns the run and verdict.

Validation: Independent review of the endpoint receipts.

Predecessors: Q1, Q5.

Priority: Medium. Successor scope; does not gate Q4.

Delivery boundary: Measured evidence merged to main for v0.1.30.

### Q1-QWEN36-SPEC. Execute Qwen3.6 Q1 T5/T6 speculative endpoints

Outcome: The Qwen3.6 speculative and owning-worker recovery endpoints have actual execution evidence.

Requirement: The program MUST execute Qwen3.6 T5/T6 with the existing shared Qwen adapters (`QwenCoherentSpeculation`, `QwenOwningWorkerRecovery`).

Acceptance:

- Before the run, name the store T6 is compared against and the expected `common` and `left_only` counts (T6 captures rows 64..143, not all 144 logits).

- Routing margins are Missing-with-reason (no Qwen3.6 routing witness); a verifier-route `RouteSensitive` result is reported, not called localized, unless a verifier-route localizer is added.

- Design approved by the operator, 2026-11-03: `docs/research/adr063-q1-spec-design.md` Part 1 and Part 3.

- Runs use a Q1 test binary from main at or after `ea6caf46` and before ADR-065 P4b (#527) or J1 (#512) merges (P4b changes solo K3 rollback). Otherwise pin a commit that precedes both and record that relation.

- Amended 2026-11-04: the b6 run uses a build pinned to mlx-native `=0.15.18` (the `rope_multi` partial-rotary fix) and waits until 0.15.18 is published and pinned (see "Gemma 4 gap stories and the rope_multi pin").

Repository ownership: hf2q owns the run and verdict.

Validation: Independent review of the endpoint receipts.

Predecessors: Q1, Q5.

Priority: Medium. Successor scope; does not gate Q4.

Delivery boundary: Measured evidence merged to main for v0.1.30.

### Q1-GEMMA-SPEC. Add the Gemma T5 speculative comparison

Outcome: Gemma n-gram speculative rows have a Q1 numeric comparison against ordinary rows.

Requirement: The program MUST design and build a Gemma T5 actual-execution adapter and run its comparison.

Acceptance:

- The adapter design is approved before implementation (the matrix currently maps Gemma T5 to `Missing`). Approved by the operator, 2026-11-03: `docs/research/adr063-q1-spec-design.md` Part 2 and Part 3.

- A `RouteSensitive` Gemma T5 result MUST be localized; the operator rejected reporting it unlocalized (2026-11-03). This story builds the verifier-route localizer of the design's §2.6, reusing Q1-GEMMA-T2LOC's stage capture and localization pieces, and runs it for branch N (committed rows `RouteSensitive`) and branch P (refusal at row 127).

- Before any run, the classification rule is frozen: Class A or Class B for n-gram committed versus ordinary rows, the τ and the record set.

- The supervisor change is shared with Q1-GEMMA-T2LOC: one change, one admission review.

- The comparison runs and is classified under the frozen rule; the receipt records the verdict.

Repository ownership: hf2q owns the adapter, supervisor case and verdict.

Validation: Independent review of the design, the frozen rule and the receipts.

Predecessors: Q1, Q5, Q1-GEMMA-T2LOC (its supervisor change and admission review).

Priority: Medium. Successor scope; does not gate Q4.

Delivery boundary: Measured evidence merged to main for v0.1.30.

## Coverage and sequencing

F1 precedes F2; F2 is the scoreboard that renders every measured lane, but since 2026-10-30 it no longer gates other stories. F3 precedes C1 because the checkpoint contract must reflect current graft and persistence behavior. A family's Q5 calibration precedes that family's hardware experiments and retained-state expansion; families do not wait on each other. Qwen implementation C2 follows C1 and ADR-065 P4a/P4b and is delivered in v0.1.30 (see "Speed program, Q1 successors and release mapping (operator decisions, 2026-11-03)"). Gemma/DeepSeek multi-anchor implementation, mixed-policy production work, queue-aware admission, new native block drafters/lattice trees, NAX attention and scratch-wait changes are successor candidates after the primary deciding measurements, not promised v0.1.28 features. S1 and D2 record deciding evidence before successor designs. Q1's numerical-invariance evidence informs shared target batching, speculation and numerical kernels, which must still prove exactness against their own serial reference. Different edits to a shared file require merge coordination, not invented product dependencies.

All acceptance, exclusions, failed-path requirements and the winning-subset release rule above are material source obligations and must be represented in the awa extraction coverage. Numerical or API-semantic failures stop promotion. A negative experiment is valuable measured evidence, not a shipped feature; no performance estimate in the research report is an acceptance receipt.

## Execution entry (approved 2026-10-23)

The operator authorized merging the corrected planning PR and making F1 (#356) and F3 (#358) Ready. The operator subsequently prioritized Qwen3.8 and DeepSeek-V4, with Qwen3.6 and Gemma 4 the next tier, and required proven speed wins for all four. Separate per-artifact M5 Max/128 GB results and independent QE remain required. The recorded skeptic review supplies the technical review of the program; this entry does not approve a later runtime design.

- F1 starts by reading coordination, inventorying source/build/binary identities and existing model/conversion receipts without model loads, then designs a versioned immutable comparison manifest and fail-closed validator. Resolve current peer source at campaign start in an isolated checkout. Preserve the product pin until its governing replacement review. Missing artifacts, conversion provenance or capabilities are explicit unresolved inputs, never invented evidence. Deliver the validated manifest, capability dispositions and reproducible commands through a reviewed merge.
- F3 starts with the named ADR/source/paper audit. Produce a claim-to-evidence table for ADR-010/011/019/037/039/049, preserve failed experiments, correct only supported claims, run the ADR inspector and reconcile existing source-bound issue keys. Deliver reviewed documentation and verified tracking through a merge; do not change runtime behavior under F3.
- After F1 delivery, Q5 is the next design/qualification gate; F2 waits for both F1 and Q5. F3 can proceed independently of F1. Q5 needs an independently reviewed memory ledger and bounded stress plan before model execution. No runtime candidate begins before the certified harness and applicable state/numerical contracts are frozen.
- Each execution session claims exactly one Ready story through an awa attempt before editing, uses its own topic worktree, and reads release coordination before heavy host work. Candidate implementers and independent QE have separate identities and receipts. Only the release driver dispatches shared model gates.

### Dedicated-host resource policy (operator update, 2026-10-25)

The operator confirmed that this laptop is dedicated to the program and authorized using all RAM available to the system, with one hf2q process at a time. This updates Q5's resource policy; it does not accept a model configuration, waive the disjoint-ledger or bounded stress proof, reduce matched context/output/task budgets, or establish any speed win. A chosen reaction margin must be tied to observed stop latency and allocation overshoot rather than retained as an unexplained fixed reservation.

### Parallel source implementation (operator update, 2026-10-27)

After separate design review, the operator explicitly authorized implementing source work in parallel for F2 (#357), Q1 (#367), Q2 (#368) and C1 (#359). Their reviewed execution plans are linked below. These source attempts may execute while Q5 remains unfinished. F2 source work may overlap Q5, C1 source work may overlap Q5, and Q1/Q2 source work may overlap F2; these execution-entry exceptions do not remove the original predecessors for hardware qualification, final acceptance or delivery. F1/F3 remain satisfied predecessors. The dependency declarations in each story above retain those final obligations.

Each attempt has a separate topic worktree and writer. Shared-file integration and all heavy builds, tests and model/helper execution are scheduled by one integration owner. Only one hf2q instance may run at a time, including test/metadata helpers, and a peer model must be unloaded before loading hf2q or vice versa. Source implementation is not permission to activate an unqualified runtime candidate or expand retained-state memory without the applicable Q5, harness and correctness gates. A native API change must be independently validated, published and exactly pinned before its hf2q consumer is reproducibly delivered.

The approved source designs are [F2](../research/adr063-f2-execution-plan.md), [Q1](../research/adr063-q1-execution-plan.md), [Q2](../research/adr063-q2-execution-plan.md) and [C1](../research/adr063-c1-execution-plan.md). Approval and parallel source authorship establish no quality, memory-admission, performance or task-completion result. Candidate authors and independent QE remain distinct. The existing four-family win requirement and all failed-path, confidence and release gates remain unchanged.

### Measured memory admission, per family (operator update, 2026-10-30)

The operator restated that the laptop is dedicated to this program and that
its work is the only significant memory user. Runs must not exhaust memory,
and must not be blocked by reserves that nothing uses. Q5 above was
rewritten accordingly. Admission now comes from each family's calibrated run
plus live stops, not from a worst-case charged ledger.

The worst-case ledger could not admit even the configuration already
exercised. For the DeepSeek-V4 Flash q2 artifact (107.4 GB), the reviewed
worst-case operating case charged 149,367,089,308 B against 137,438,953,472 B
of physical memory. The guarded attempt with the provenance-bound `538e9ef3`
binary refused before Device on 2026-10-30. The same artifact had run the
four-agent C262144 coding workload six times on 2026-10-29. Every sample
reported normal pressure, swap stayed flat at 0.020 GB, and Free plus cached
memory never fell below 5.11 GB
(`ds-current-port-balanced-series-analysis-v2/metrics.json`).

Q5 gates each family separately. Once a family's calibration passes, that
family's hardware work in F2, C1, S1, S3, D1, D2, K1 and Q3 may proceed; no
family waits on another family's calibration. DeepSeek-V4 needs no
exception, because its existing six-run evidence already meets the
calibration conditions; a recalibration is still needed when its source
changes. The worst-case constructor at `538e9ef3` (draft #453) remains a
diagnostic, not the admission gate. This update establishes no speed win or
task completion.

### Per-family closure and release scope (operator update, 2026-10-30)

The operator decided three scope changes after the v0.1.28 package was
published:

- **Per-family evidence closes stories.** F2 and Q1 remain program stories but
  are no longer closure gates for other stories. A story closes on per-family
  hardware evidence that an independent reviewer accepts: an A/B against stock
  hf2q on the same artifact, workload and binary family, the story's exactness
  or isolation tests, a passing task-quality result and a passing Q5
  calibration for that family. F2 renders those results on its scoreboard.
  The full fixed-six, three-arm F2 campaign needs about 72 runs plus peer
  adapters that do not exist for DeepSeek-V4 or Gemma 4; it would otherwise
  have gated every story.
- **Q3/Q4 accept per-family peer wins.** Each family receives a measured win,
  a measured loss, unavailable (no runnable peer arm on its declared workload)
  or failed (hf2q task failure) verdict with evidence. Only measured wins count
  as wins; nothing is pooled or dropped. Q4 ships in the first hf2q release
  that contains the accepted configuration.
- **C2 moves to v0.1.29.** The four-agent workload has no divergent-turn
  demand (24 of 24 follow-ups hit the exact prefix, about 76% cached), so
  multi-depth Qwen reuse would show no measurable gain, while it requires
  atomic restore of recurrent state that does not exist yet.

Evidence at the time of the decision: DeepSeek-V4 joint decode (#458)
reduced four-agent median makespan from 141.7 s to 105.0 s, bitwise equal to
serial decode. Qwen3.6 fails its coding task on both engines because of a
non-compiling idiom in the model weights, not hf2q numerics (BelowFloor
campaign `quality-qwen36-peer-exp0b95-20261030T174020Z`; diagnosis in
`docs/research/adr063-qwen36-task-failure-diagnosis.md`; hf2q fixes in #461).
DeepSeek-V4 and Gemma 4 have no peer adapter. This update
establishes no speed win by itself.

### Native-engine peer arms (operator decision 2026-10-31)

The operator approved native-engine peers as separately labeled benchmark arms:
antirez ds4, TensorFold and oMLX. Each runs its own native artifact, not the
hf2q GGUF. This section records the ds4 arm (`engine_kind ds4`,
`scripts/four_agent/native_peer_ds4.py`,
`q5_cases/native-ds4-deepseek4.json`), the TensorFold arm (`engine_kind
tensorfold`, below) and the oMLX arm (`engine_kind omlx`, below). Every
consumer resolves a native kind through one registry,
`scripts/four_agent/native_peers.py`.

- **The DeepSeek-V4 same-GGUF peer stays Unavailable.** Both runs
  `$E/peer-arms-v1/peer-deepseek4-20261030T165847Z` and
  `peer-deepseek4-mmap-20261030T165948Z` loaded the hf2q artifact and then hit
  Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory` on their first requests.
  The Q3/Q4 same-GGUF verdict for DeepSeek-V4 remains `Unavailable` with that
  evidence. A native arm never substitutes for it.
- **The ds4 arm is a labeled different-artifact comparison.** ds4 c97964b
  serves its ds4f-q2 Flash 0731 GGUF: about 81 GiB, IQ2_XXS gate/up, Q2_K down
  and Q8 attention projections, shared experts and output. ds4 rejects the
  hf2q artifact's Q3_K routed experts (`ds4.c:4994`). Reports name the arm
  `ds4 c97964b (native Flash-0731 q2 artifact)`. As required by Q4, this
  native-format competition is reported separately from the same-GGUF verdict:
  `verdict.json` `native_peer`, F2 baseline `ds4_native`.
- **The verdict wording is task makespan only.** The claim takes the form
  "<hf2q label> finishes the same coding task X% sooner (or later) than ds4
  c97964b (native Flash-0731 q2 artifact)", with the paired 95% interval. It
  always comes with each arm's token volumes (median prompt and completion
  tokens per run), per-output-token rate (median completion tokens per
  makespan second) and the worst-slot p95 TTFT ratio. The weights differ, so
  it is never a same-weights engine-speed, per-token-efficiency or
  model-quality claim.
- **The quality gate applies.** The experiment arm keeps every Q1/Q3 task and
  quality gate. ds4 must complete the same canonical four-agent task:
  - ds4 task failures are recorded;
  - an unavailable ds4 arm is recorded;
  - ds4 server-side tool-call recovery is recorded per run from `--trace`.
    ds4 can repair an unterminated tool call, or append a model-visible tool
    error and decode again inside one request, so the client sees only the
    second attempt (`ds4_server.c:12627-12639, 14313-14385, 14448-14520`). The
    run stays valid;
  - while any of these holds, or while either arm has a failed run, no
    makespan statement is published (`statement_withheld`).
  - Native comparisons never count as F2 wins. The scoreboard renders them in
    their own "Native-engine comparisons" section, together with their
    companion figures.
- **The Think Max control is explicit.** The canonical DeepSeek-V4 request uses
  `reasoning_effort=max`.
  - ds4 renders Think Max only when the per-session `--ctx` is at least
    393216, and silently falls back to normal thinking otherwise
    (`ds4.c:424-427, 61096-61100`).
  - The case therefore runs four 393216-token sessions (`NATIVE_DS4_CTX`), and
    the adapter refuses any lower context. The workload stays 262144 tokens per
    slot, enforced on every turn's usage.
  - The added KV memory is recorded from ds4's startup line. An estimate from
    ds4's own sizing formulas (`ds4.c:40421-40530`): about 16.5 GiB of context
    buffers with a shared prefill workspace, or 25.5 GiB without one, versus
    about 11.5 GiB at 262144. That is on top of about 81 GiB of weights. The
    root's first calibration establishes whether this fits.
  - Prompt structure is not identical. ds4 renders the tool schemas before the
    client system text (`ds4_server.c:3114-3133`), while the 0731 template
    appends them after it (`deepseek-v4-flash-0731.jinja:42-46, 54-60`). ds4's
    Think Max prefix is the template's `high` text, not its `max` text. The
    weaker wording likely shortens ds4's reasoning, which biases makespan in
    ds4's favour. Byte-identical prompts are therefore `Unavailable`.
    Per-request think mode, greedy sampling, the rendered prefix and the token
    counts are proven from ds4's `--trace`, bound to each client request by the
    sha256 of its body.
  - ds4 cohorts are not Q2-certified (`q2_certify.py` refuses them). Q3/Q1
    audit every ds4 Pass with the trace audit instead.

**TensorFold arm (operator approval 2026-10-31).** TensorFold 1157ea5b
(version 1.2.0) serves Qwen3.8 27B from its own MLX 4-bit checkpoint,
`TensorFold/Qwen3.8-27B-MLX-4bit@22d8d538`. Two cases are recorded:
without a drafter (`q5_cases/native-tensorfold-qwen38.json`) and with the
stock DFlash2 drafter `z-lab/Qwen3.8-27B-DFlash2@50307d4c`
(`native-tensorfold-qwen38-dflash2.json`). The two argv differ only by
`--drafter DIR --drafter-bits 4`.

- **It never produces the same-GGUF verdict.** Qwen3.8's same-GGUF peer is
  runnable, so TensorFold runs only in a separate Q3 campaign declared with
  `--native-comparison`. Its `verdict.json` and F2 rows are labeled
  "native-comparison campaign; not the verdict of record". `native_peer`
  carries the task-makespan-only statement, under the same rules as ds4: never
  a win, withheld on failures or a missing audit. F2 baselines are
  `tensorfold_native` and `tensorfold_dflash2_native`.
- **Request parity is explicit.** The adapter sends greedy sampling,
  thinking on, and `reasoning_effort: "xhigh"`. That is the effort hf2q
  renders: its Qwen3.8 template has the same bytes as TensorFold's
  `chat_template.jinja`, and it defaults to xhigh. The adapter also sends hf2q's
  2048/512 thinking ceilings as `thinking_budget`, and leaves
  `parallel_tool_calls` unsent. With `false`, TensorFold would silently drop
  extra calls; unsent, the client refuses them. TensorFold ignores unknown
  request fields, so the adapter enforces its own closed envelope.
- **These differences are recorded, not hidden:**
  - no repetition penalty (hf2q uses 1.05);
  - a different forced thinking close;
  - no tool grammar;
  - an unlogged closing of tool calls cut by an end token;
  - a re-saved `tokenizer.json`. Per-turn TensorFold token ids are archived
    for the hf2q token oracle.
- **Identity is bound end to end.** The client binds:
  - the binary sha256 and the pinned commit and version;
  - the snapshot revisions, through the HF cache layout and
    `/v1/benchmark/manifest`;
  - `config_sha256` and `tokenizer_sha256`, recomputed locally with
    TensorFold's framing;
  - the exact server argv;
  - `/health` (4 lanes) and `/v1/models` (262144 context);
  - the startup lines for streams, context, drafter state, reserve and
    prompt cache;
  - every turn's rendered prompt, via `/tokenize` (count equals
    `usage.prompt_tokens`).

  The post-run audit binds each reply to TensorFold's own `done` line
  (thinking, effort, tokens, finish, draft counters).
- **Memory may not fit.** Four 262144-token bf16 slots plus weights and the
  margin come to about 96–106 GiB against Metal's working set. `--parallel 4`
  refuses rather than shrinking. The first calibration decides whether the arm
  runs at full context.
- **D2 (#365) uses both cases as T0 and T2.** The two arms are compared in this
  order: four-slot task completion, then quality, then memory (Q5 verdict,
  startup reserve, `/health` memory, slot waits), then net makespan.
  Accepted-per-drafted and tokens-per-round counts explain the result but never
  decide it.
- **D2's long contexts use `--workload long-context-v1`.** The canonical
  coding client reaches only about 2.8k prompt tokens per turn. The variant
  keeps the canonical task, prompts, tools, workflow and grading, and appends a
  pinned 150 KB read-only Rust module (`fixtures/long-context-v1.rs`) to
  `tests/ports.rs`. Turns 2–6 then carry about 40k prompt tokens (estimated as
  wire bytes / 4). Approvals and summaries bind the variant id and the fixture
  sha256. Q3 plans and the D2 command refuse runs on different variants. T0
  and T2 are derived with `coding_long_context.py derive-case`; see
  `scripts/four_agent/README.md`. The fixture is templated, repetitive code, so
  it covers long contexts only. It is not evidence for the nonrepetitive-code
  item.

**oMLX arm (operator decision 2026-10-31).** oMLX 6b413c90 (version 0.7.3rc1,
MLX/mlx-lm continuous batching) serves three families from mlx-community 4-bit
checkpoints: Qwen3.8 27B (`Qwen3.8-27B-4bit@10c35caa`), Qwen3.6 35B-A3B
(`Qwen3.6-35B-A3B-4bit@38740b84`) and Gemma 4 26B-A4B
(`gemma-4-26b-a4b-it-4bit@0d77464e`), with cases
`q5_cases/native-omlx-{qwen38,qwen36,gemma4}.json`. DeepSeek-V4 is out of
scope because no 4-bit conversion fits. The registry's family scope became a
tuple (`FAMILIES`, `Q3_FAMILIES`) so one arm can serve several families.

- **It never produces the same-GGUF verdict.** All three families have a
  runnable same-GGUF peer, so oMLX runs only in a separate `--native-comparison`
  campaign. Labels, withholding and the F2 placement are TensorFold's: F2
  baselines `omlx_<family>_native`, never a win. Any oMLX tool-parser warning
  or INFO repair line in a run also withholds the statement; repairs oMLX does
  not log are disclosed with every statement (`recovery_attribution`). The Qwen3.6 re-test with a different
  tokenizer and sampler stack uses `quality_campaign.py --reference-kind
  omlx-native`; oMLX cannot be the subject arm.
- **Request parity is explicit.** oMLX silently drops unknown request fields,
  so the adapter enforces a closed envelope. It sends greedy sampling, hf2q's
  repetition penalty 1.05 over a 64-token window, and thinking as hf2q renders
  it. For Qwen that is `enable_thinking`/`preserve_thinking` true, no explicit
  effort, and `thinking_budget` 2048/512. For Gemma it is `enable_thinking`
  false. `parallel_tool_calls` is not sent: oMLX has no such field, and the
  client refuses a second call. A generated `model_settings.json` pins the same
  values server-side. Its sha256 is in the approval, and it must be unchanged
  at closing.
- **These differences are recorded, not hidden:**
  - MLX affine 4-bit weights instead of hf2q's Q4_K_M GGUF;
  - the mlx-community tokenizer, and a template whose identity with hf2q's
    reviewed template is recorded per run;
  - oMLX's tool-schema normalization (sorted keys, injected empty
    descriptions, no tool grammar);
  - a different forced thinking close;
  - unproven seeding of the repetition window with the prompt tail.
- **Identity is bound end to end.** The client binds:
  - the detached clean checkout and its version file;
  - the venv's mlx, mlx-lm and mlx-vlm pins, from dist-info (a pip-freeze
    equivalent hash);
  - the snapshot revision, `config.json` and `chat_template.jinja`;
  - the launch record (pid, exact argv, environment);
  - `/api/status` (version, every custom kernel available, no request before
    the barrier) and `/v1/models/status` (served path, text engine, 262144
    context);
  - the re-saved `settings.json` (hot-only prefix cache, no SSD tier);
  - every turn's rendered prompt via `/v1/tokenize`.

  At closing the server's session counters must equal the client receipts.
  oMLX logs no response id, so the post-run audit binds the multiset of
  `Chat completion:` lines (transport, tokens, finish) to the client turns.
- **Memory is paged.** KV grows with real tokens: up to about 64 GiB
  (Qwen3.8), 20 GiB (Qwen3.6) or 20 GiB (Gemma 4) at four full 262144-token
  slots, on top of 15–20 GB of weights. Raising `iogpu.wired_limit_mb` is the
  root's decision.

This decision establishes no speed win and changes no same-GGUF verdict.

### Splash peer of record for the Qwen families (operator decisions 2026-11-03/11-04)

For Qwen3.8 27B and Qwen3.6 35B-A3B, a v0.1.30 Win requires hf2q to finish the
canonical four-agent coding task at least 5% sooner than **both** the peer and
Splash, each in its best valid configuration. Splash (`incoai/splash`, source
build `031a25a6`, engine kind `splash`) is therefore a second same-artifact
peer of record for these two families. Splash serves neither Gemma 4 nor
DeepSeek-V4 (`install/families.py`), so those families keep one peer. The
design is `docs/research/adr063-splash-peer-design.md`: design approved
(operator, 2026-11-04), with decisions D1–D10 (operator, 2026-11-04, as
amended by its independent review).

- **Same artifact.** Splash serves the hf2q GGUF of the family's hf2q arm, the
  same path and sha256, through its local-file mode. Its `model.json` digest
  must equal the hf2q artifact sha256. Splash's GGUF screening and engine
  `model-check` accept both hf2q Qwen GGUFs and render from their embedded
  templates, which are byte-identical to hf2q's reviewed ones.
- **Pinned drafter.** Splash decodes with its DFlash2 drafter, a separately
  pinned artifact (F1):
  - Qwen3.8: `incoai/Qwen3.8-27B-DFlash2@015e7956`, byte-identical to
    `z-lab/Qwen3.8-27B-DFlash2@50307d4c`;
  - Qwen3.6: `incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b69`.

  Both are bound by digest. Splash serves text only, with four lanes and a
  262,144-token context.
- **Best valid configuration.** Splash has no switch that turns speculation
  off, so a no-speculation Splash arm is Unavailable.
  - Selection: the declared makespan rule, from a one-factor sweep of
    Splash's documented knobs. The knobs are KV and state formats, the Neural
    Engine prefill split (Qwen3.8), verify rows (Qwen3.6), decode share and
    the SSD cache tier.
  - The choice is confirmed before the campaign of record.
  - Validity comes from Splash's own runs (D10). Every run must:
    - pass Q5;
    - pass the run audits;
    - pass the Q1/Q3 task and tool gates.

    Splash's task pass fraction must also be at least the Q1 numeric-policy
    floor (0.75).
  - Engine-documented lossy settings are valid under that rule. These are
    int8 KV (the default), bf16 state if a sweep point selects it (fp32 is the
    default), and the Neural Engine W8A8 FFN split (the Qwen3.8 default).
  - The paired hf2q-versus-Splash quality gate is a Q4 gate on the candidate.
    It never disqualifies a Splash configuration.
  - `--max-memory` stays automatic unless the Q5 floor trips at a sweep point
    or in the campaign. A cap is then part of the recorded configuration (D6).
  - A matched-settings diagnostic is measured in the sweep only and labeled.
    It is not a campaign arm. Its settings are bf16 KV, fp32 state, the Neural
    Engine split off (Qwen3.8), every verify row (Qwen3.6) and no SSD tier.
- **Q3 campaign.** Each of these families runs one five-arm Williams
  campaign: stock, experiment, peer, peer_matched and splash (D1). It reports
  experiment-versus-Splash beside experiment-versus-peer, in the n4 primary
  and the n1 control. Splash does not run the 3+1/2+2/1+3 mixed controls
  (D7). Splash cohorts are audited as the peer's are: the dry audit plus
  `q2_run_audit` with the arm's audit module. Like the peer's, they are not
  Q2-certified.
- **Q4 win.** For these families `family_verdict` is Win only when both of
  these ratios have 95% CI upper bounds ≤ 0.95, each against that engine's
  best valid configuration:
  - hf2q/peer four-agent makespan;
  - hf2q/Splash four-agent makespan.

  Every other Q4 gate must also hold.
  - A Qwen campaign without a Splash arm is `Unavailable`, never a Win against
    the peer alone.
  - A failed Splash arm is `Failed`.
  - An experiment slower than Splash (ratio CI above 1) is `Loss`.

  These are checked in the order given in the design (§6.6). Splash task
  failures are also recorded as quality findings.
- **Quality gate.** The paired task-quality gate on the exact candidate
  binary also runs against Splash, bound to the Splash readiness identity.
- **Fairness.** Vs-Splash worst-slot TTFT ratios are disclosed and not gated.
  SSE gaps against Splash are not comparable, because one emission can carry
  several speculative tokens.
- **Request parity.** The client sends:
  - greedy sampling;
  - Qwen thinking on with `preserve_thinking`;
  - hf2q's 2048/512 thinking ceilings as `thinking_token_budget`, closed by
    hf2q's forced text `"\nI need to answer now.</think>"` (D4);
  - `parallel_tool_calls: false` (D5), which turns on Splash's tool grammar.

  Every prompt is re-rendered by Splash's `/apply-template` and counted by
  `/tokenize` (count equal to the turn usage). Its bytes are compared after
  the run with hf2q's own rendering of the same canonical request, by the hf2q
  exact-prefix post-oracle in a Splash mode. A byte mismatch makes the run
  Invalid; until that check runs, parity is `PendingPostOracle`.
- **Recorded differences.**
  - No repetition penalty is sent (D3). Splash's penalty counts the whole
    prompt and output, unlike hf2q's 1.05 over a 64-token window.
  - Splash greedy output is not run-to-run exact on the MoE target under its
    default verify rows.
  - Splash's numerics include int8 KV and the Neural Engine split where the
    chosen configuration uses them.
- **Qwen3.6 workload.** Splash runs the declared Qwen3.6 workload, the
  repair-turn workload (one repair turn, recorded under Q4-QWEN36), as for the
  hf2q and peer Qwen3.6 arms (D9). Repair-turn support for Splash landed in
  #715 (`q5_cases/splash-qwen36-repair1.json`). Qwen3.8 keeps the canonical
  protocol.
- **Identity.** These are frozen with the campaign identity:
  - the commit and a clean tree;
  - the sha256s of `build/splash`, `build/splash.metallib` and
    `build/host-step.metallib`, the `build_id` and the build toolchain
    versions (Xcode, Metal, clang, SDK, macOS);
  - the venv freeze;
  - the drafter revision and digests;
  - the GGUF sha256, through the same-model rule;
  - the configuration;
  - the server argv template.

  Per run, readiness binds:
  - that run's assembly path;
  - `launch.json`, whose sha256 goes into that run's approval;
  - the argv derived from the template;
  - every `model.json` field except paths and file times;
  - the Neural Engine calibration state before and after the run.

  These per-run items are never frozen across runs: an hf2q run that
  hard-links the GGUF changes its ctime, and with it Splash's assembly path.

  Splash's model assembly symlinks the GGUF and the drafter and never writes
  or hard-links them (D8).

This decision establishes no speed win.

Research update (2026-11-09): the frozen Qwen3.8 Splash sweep produced no
valid configuration. All 16 runs (eight one-factor points, two runs each)
failed the canonical task gate; no run reached `CalibrationPass`, a passing
audit and the prompt-parity post-oracle. This remains a `Failed` Splash arm
under the decision above and establishes no Q4 verdict.

Follow-up parity correction (2026-11-09): a replay exposed one request adapter
difference. Splash rendered each tool's semantic default `"strict": false`
while hf2q omitted it. The corrected adapter omits only that false default on
the Splash wire and restores it for the canonical request. All 20 completed
prompts in a corrected Q5 replay passed the byte post-oracle. The corrected
Q5 still failed the task gate: two agents produced and tested the correct
`std::num::ParseIntError` edit and two retained `std::fmt::Error`, below the
0.75 task floor. This closes request rendering as the remaining cause in that
run; it neither validates nor relabels the frozen sweep.

The raw decode lead is nevertheless measured and source-attributed. Across
S0's 17 completed responses before its task failure, Splash accepted 61.48%
of 4,746 DFlash2 proposals and produced 5.329 completion tokens per cycle at
a median 60.05 tok/s per request. The pinned implementation combines a
five-layer, seven-proposal DFlash2 drafter and coherent rank-256 path selector;
confidence/cost-priced 1--8-row verification; three command stages that
overlap CPU grammar simulation with target forward; and decode-debt
scheduling. hf2q's ordinary four-lane target step is weight-streaming-bound
at a traced median 89.08 ms, while its first production joint-MTP pilot can
only draft one token per lane at width four. The source and receipt analysis,
including validity limits and the required successor architecture, is
[the Qwen3.8 Splash gap record](../research/adr063-qwen38-splash-gap.md).

### F2 closure scope: option B (operator decision 2026-10-31/11-01)

The operator approved option B of the F2 intent review
(`$E/reviews/adr063-f2-intent-v1.md` §5 B) on 2026-10-31/11-01 as the closure
scope of F2 (#357). F2 closes when the scoreboard shows, for each family, the
items below. Each option-B item is kept, changed or waived as stated:

| Option B item | Disposition |
|---|---|
| 1. An independently reviewed Q3 `n4` verdict | Kept: the Q3 verdict of record (Win, Loss, Unavailable or Failed) from a reviewed `verdict.json`. |
| 2. A one-slot control stage | Kept: the record's completed `n1` control is reported as a control of that record. |
| 3. Task quality bound to the experiment binary | Kept and widened: paired quality campaigns run on the exact Q3 experiment binary against v0.1.28 and against the peer (against the native ds4 arm for DeepSeek-V4, whose same-GGUF peer is unavailable). Each reference is bound to the campaign's arm identity: the v0.1.28 binary, the peer binary, or one native readiness identity of that engine. Final sources still pending patch review also block a Win. |
| 4. The Q2 run audit's co-progress width of 4 for every passing run | Kept: every counted run passes the v3 width audit (the Q3 Pass rule). A width failure makes the run `Invalid`, which stands. |
| 5. Worst-slot TTFT and SSE-gap against stock and peer, with a declared fairness ceiling against the peer | Changed: worst-slot TTFT p95 and SSE-gap max ratios are shown against stock and peer. The stock band (×1.025) gates a Win. Against the peer they are reported only, with no ceiling, because the operator worded the peer comparison as task makespan only. SSE gaps against the peer are shown as not comparable (different event granularity). An SSE gap the verdict cannot estimate is shown as not measurable and does not gate, as in the Q3 rule. |
| 6. 3+1, 2+2 and 1+3 controls (descriptive) | Kept, for every family: completed mixed stages are controls of the record. Stages that halted or did not run are disclosed as unmeasured. |
| One statistics rule | The Q3 design statistics (5–20 Williams blocks, sequential 2.5-point precision stop). The sequential rule is the Q3 rule pre-registered on #369 (`docs/research/adr063-q3-campaign.md` "Design", fixed before any campaign data), not a new choice made for F2. The fixed-six rule moves to successor scope. |
| A board that shows a Q3 Win as qualified | Changed: the "Q3 verdicts of record" tier is F2's qualified display. The fixed-six "Qualified wins" section is labeled successor scope. |

Justification of the sequential rule. The independent Q3 reviews re-derived
every stopping decision from the raw runs under it and accepted the verdicts
of record: `$E/reviews/adr063-q3-v7-qwen38-verdict-v1.md`,
`adr063-q3-v7-deepseek4-verdict-v1.md`, `adr063-q3-v7-gemma4-verdict-v1.md`
and `adr063-q3-v9-gemma4-verdict-v1.md`. A stage that reaches 20 blocks
without precision is inconclusive on that contrast and is disclosed as such.

Native-engine rows stay separate. They render in their own section and never
stand in for the same-GGUF verdict. A campaign with a native peer arm
contributes only its native rows.

The scoreboard renders this as the "Q3 verdicts of record" tier above the
Descriptive results (`scripts/four_agent/f2_q3_tier.py`,
`f2_q3_view.py`; how to read it is in `docs/research/adr063-f2-scoreboard.md`).
"Win" appears there only against v0.1.28 stock (release `9a0b79c7`; the stock
arm is v0.1.28, not v0.1.27), and only when all of these hold:
- the makespan interval against stock excludes a tie, with half-width ≤ 2.5 points;
- neither measurable stock fairness effect regresses beyond the neutral band;
- the primary stage reached its stopping rule, with valid thermal and memory receipts;
- no final source awaits patch review;
- the quality gate passed against both references.

Moved to successor scope: the fixed-six three-arm campaign, canonical-transcript
replay timing lanes, the cancellation/retry, divergent-edit and request-order
permutation scenarios, and captured production-client sampling. The F2
acceptance bullets that name them are the successor's acceptance, not this
closure's; `f2_scoreboard_cli.py` still renders any fixed-six result produced.
This decision establishes no speed win by itself.

### F2 scoreboard of record (2026-11-02)

The "Q3 verdicts of record" tier of `docs/research/adr063-f2-scoreboard.md`
now carries one reviewed record per family, rendered with `--verify` from
`docs/research/adr063-f2-scoreboard-rows.json`. Every quality gate of record
is `Pass` (all six `quality-gate-v1` manifests and patch reviews; Qwen3.6
declares none). The
reviews' scoreboard sentences and disclosures are in the tier's "Family
notes". Per family:

- **Qwen3.8 27B (Q3 v11, hf2q 38485a23): Loss against the peer.** v11
  (pre-registered on #369, comment 6583318349; review
  `$E/reviews/adr063-q3-v11-qwen38-verdict-v1.md`) ran the peer in its best
  valid configuration, with its own MTP head. That peer finishes the same
  coding task 6.0% sooner than hf2q (95% CI 4.2–7.9%; 79.7 vs 84.3 s; 5
  blocks), emitting 1.88× the completion tokens at a 2.00× per-output-token
  rate; hf2q's worst-slot p95 TTFT is 2.61× its. hf2q finishes 25.3% sooner
  than v0.1.28 (CI 23.7–26.8%), which is not a Q3 Win because the family
  verdict is Loss vs the peer. Against the peer with speculation off (matched
  settings) hf2q finishes 29.8% sooner. n1: hf2q is 20.3% sooner than the MTP
  peer, and 10.4% slower than v0.1.28 (CI 7.7–13.1%) at an unchanged
  per-output-token rate (0.985×), entirely from a longer final summary turn.
  The peer quality gate (run on the speculation-off peer) carries over under
  the pre-registered exactness contingency (not triggered, 41/41 under the
  registered tool, which compared only turn 0 for peer runs; turns 0–5 are
  content-identical across all MTP and speculation-off runs). v11 supersedes
  the v7 record (26.4% sooner than stock, 31.5% sooner than the
  speculation-off peer), which stays published beside it as matched-settings
  evidence, as do the v8 3+1 (25.9%) and 2+2 (23.0%) controls of v7; 1+3 is
  unmeasured after a width-only halt at the peer warmup.
- **DeepSeek-V4 Flash (Q3 v7, hf2q 38485a23): Unavailable.** The same-GGUF
  peer runs out of memory (prior evidence). Against stock, hf2q finishes 24.7%
  sooner (CI 23.8–25.6%, 5 blocks), shown in the tier as a Win against
  v0.1.28, with byte-identical final sources in both
  arms; the SSE fairness check is not measurable, and n1 is neutral. On
  different artifacts, hf2q 38485a23 finishes 5.2% sooner than ds4 c97964b
  (CI 3.0–7.4%; task makespan only, with the disclosed parity gaps; never a
  Win). The ds4 artifact sha256 is per HF LFS metadata and not yet locally
  re-hashed. Every mixed stage is unmeasured (structural stock-warmup halts).
  The DeepSeek-V4 Q1 numerical endpoint captures are deferred to v0.1.29
  (#549; #367 comment 6566342659; their 119.18 GiB preload floor exceeds the
  host's 109.34 GiB), so DeepSeek-V4 quality in v0.1.28 is the paired
  task-quality gate against v0.1.28 and against ds4, plus the S3 bitwise
  joint == serial check.
- **Qwen3.6 35B-A3B (Q3 v10, hf2q 38485a23): Failed** (Q3 rule 1). The n4
  experiment warmup failed its task (agent-2, E0282, no repair turn allowed);
  0 paired blocks, so no speed effect. Stock v0.1.28 failed the same way; the
  peer was not run.
- **Gemma 4 26B-A4B (Q3 v9, hf2q 5d421a2c): Loss.** hf2q finishes 37.0%
  sooner than v0.1.28 (CI 35.6–38.4%), but it is not a Win: the stock SSE-gap
  fairness gate fails (1.047×, CI 1.036–1.058). The peer finishes 13.4% sooner
  (CI 10.6–16.2%; 20-block cap, vs-peer precision not met). Every mixed stage
  is unmeasured (structural stock-warmup halts in v7, v8 and v9).

Two verdicts on the board are recomputed copies, because the reviews cite the
frozen verdict files by hash (`748f6507…` for the Qwen3.8 controls,
`a1caf7d4…` for native ds4), so those files stay unchanged. Each copy is under `$E/f2-assembly/`, beside a byte-identical
copy of its `campaign.json`. `deepseek4-native-ds4/verdict.json` carries the
native statement labelled hf2q 38485a23 (the frozen v8 file says "main HEAD").
`qwen38-controls/verdict.json` binds both Qwen3.8 quality-gate manifests and
the v7 patch review.

The root session ran every Q3 campaign (`qe_identity` `root-resume-20261030`)
rather than a separate QE session or host. Q3's independence requirement is
met by substitution: independent reviewers who did not implement the
candidates reproduced every verdict from the raw receipts
(`docs/research/adr063-q3-campaign.md`, "Independent QE and reproduction";
`$E/reviews/adr063-q3-closure-v1.md`, finding 5).

### Speed program, Q1 successors and release mapping (operator decisions, 2026-11-03)

The operator decided the following after the Q3 verdicts of record (no peer
Win in any family):

- **Releases.** v0.1.29 is a patch release: current main plus the Gemma 4
  U+FFFD tool-argument fix (#553, a serving defect outside this ADR's
  stories). v0.1.30 is the speed program: Q4 and its family stories, C2,
  Q4-TRACE and the Q1 successors. v0.1.31 carries ADR-064 and the tracked
  successor scope listed under "Successor watchlist and tracked successor
  scope".
- **Q4 reframed.** Do not measure harder; understand why hf2q is slower than
  the peer on the same hardware and improve. All four families are targets,
  and the bar is "clearly faster": the Q4 win definition in Q4 (peer margin
  0.95), with the quality gates still required. Q4 becomes the umbrella
  closed by Q4-QWEN38, Q4-GEMMA with Q4-GEMMA-ATTN, Q4-DS4 with
  Q4-DS4-RACE, and Q4-QWEN36.
- **Gemma 4.** A precise tiled attention kernel (32-bit accumulation, in
  mlx-native) and a per-stage profile of hf2q against the peer proceed in
  parallel (Q4-GEMMA-KERNEL, Q4-GEMMA); the hf2q integration follows both
  (Q4-GEMMA-ATTN).
- **DeepSeek-V4.** Run the peer alone on ds4's file (Q4-DS4); if it fits,
  add IQ2_XXS support and compare hf2q, the peer and ds4 on that file
  (Q4-DS4-RACE). The IQ2_XXS subset of ADR-066 N1 (#580), N3 (#582) and
  N4 (#583) is pulled forward ahead of ADR-066 B2; DeepSeek-V4 consumes it,
  with no duplicate kernels. (Amended later on 2026-11-03: the subset is its
  own ADR-066 story, N-IQ2XXS, and is Q4-DS4-RACE's predecessor in place of
  N1/N3/N4.)
- **Qwen3.6.** Investigate the failure before choosing a speed workload
  (Q4-QWEN36).
- **ADR-065.** Its baseline freeze and deciding measurements, B1 (#503) and
  E1–E8 (#504–#511), and P4a/P4b (#526/#527) GPU DeltaNet restore join the
  speed program; J1 (#512) joins v0.1.30 as P4b's predecessor. Its other
  build stories join after the E8 gate decision. C2 builds on P4a/P4b.
- **Q1 successors.** All run on this Mac. #549 becomes Q1-DS4 with a
  measured footprint replacing the borrowed floor; #652 becomes
  Q1-GEMMA-T2LOC as planned; #653 is split into Q1-QWEN38-RERUN,
  Q1-QWEN36-SPEC and Q1-GEMMA-SPEC.
- **C2** is built regardless of measured demand, on ADR-065 P4a/P4b, with
  joint-decode eligibility protected; its delivery moves to v0.1.30,
  superseding the 2026-10-30 v0.1.29 placement.
- **v0.1.28 leftovers.** Only what the speed work needs is v0.1.30 scope
  (Q4-TRACE, the Qwen joint-decode trace hook). The remaining verification
  leftovers are tracked successor scope, v0.1.31 (operator-confirmed).

Sequencing derived from the planning reviews (not separate operator
decisions): Q4-QWEN38 follows the ADR-065 E8 gate; ADR-065 consumes Q1
Class A/B unchanged and P4b/J1 change the Qwen solo K3 rollback, so
Q1-QWEN38-RERUN and Q1-QWEN36-SPEC run before P4b or J1 merges or pin a
commit that precedes both; Q1-GEMMA-T2LOC's supervisor change precedes
Q1-GEMMA-SPEC; the Gemma/DeepSeek multi-anchor parity and the DeepSeek
selection witness in the v0.1.31 list come from the planning reviews.

Execution entry. The measurement-only work may start once its listed
predecessors are met: Q1-QWEN38-RERUN, Q1-QWEN36-SPEC, Q1-DS4,
Q4-DS4, the Q4-QWEN36 investigation and Q4-GEMMA. Each implementation
story (C2, Q4-TRACE, Q4-GEMMA-KERNEL, Q4-GEMMA-ATTN, Q4-DS4-RACE, the
Q1-GEMMA-T2LOC hooks and supervisor change, the Q1-GEMMA-SPEC adapter)
needs a recorded design approval before implementation.

Design approval of the measurement tracks (operator, 2026-11-03): the
operator approved the designs of Q4-QWEN36 (#662), Q4-DS4 (#661, the fit
determination), Q4-GEMMA (#660, the profiling track), Q1-QWEN38-RERUN
(#664), Q1-DS4 (#549), Q1-GEMMA-T2LOC (#652) and ADR-065 B1/E1–E8
(#503–#511). Q4-DS4 and Q4-GEMMA were split from their implementation
tracks (Q4-DS4-RACE, Q4-GEMMA-ATTN) so that per-issue tracker blockers do
not hold the approved measurement work.

Later decisions (operator, 2026-11-03), after Q4-DS4 reported Fit (six of
seven peer runs valid, f1 `ClientFailed`; provisional best valid
configuration load mode none / ubatch 1024, makespan median 65.2 s, within
noise of auto/1024, so Q4-DS4-RACE re-runs none/1024, auto/1024 and
none/512; `/Users/robert/.local/state/hf2q/v0130/q4-ds4/`)
and the Q4-DS4-RACE design review
(`docs/research/adr066-iq2xxs-subset-design.md`):

- **Q4-DS4-RACE decision D1, stock arm on ds4's file.** No released hf2q opens the file, and
  comparing v0.1.28 on a different artifact would mix an artifact change
  with the engine change (AGENTS.md: compare identical artifacts). The
  ds4-file campaign of record therefore runs without a stock arm and is
  judged against the peer. The vs-stock makespan gain, its precision, the
  fairness band and the v0.1.28 quality gate come from a same-artifact
  candidate-vs-v0.1.28 campaign on hf2q's own DeepSeek-V4 artifact with the
  same candidate binary, bound by path and sha256. This needs a no-stock-arm
  campaign mode and matching `required_references`/`family_verdict`
  changes in the harness.
- **Q4-DS4-RACE decision D2, ds4 on the same file.** ds4 c97964b runs on the same file only in a
  `--native-comparison` campaign whose native case declares the shared
  artifact. Both sha256 values must be present and equal, the declaration
  is part of the frozen campaign identity, `--peer-unavailable-evidence` is
  refused in that mode (the same-file peer runs), and labels say "same ds4
  file" instead of "different artifact". The statement stays task makespan
  only, because ds4's rendered prompts differ.

C2 sign-offs (owner, 2026-11-03), from `docs/research/adr063-c2-design.md`
§10; they amend C2's acceptance text:

- **C2 decision D1, divergent-turn gate.** On divergent turns anchors-on resumes from a
  state built by a different transaction history, a Class B alternate
  route. Every logit row of each divergent turn is checked against
  teacher-forced anchors-off rows and must be `WithinDecisionEnvelope`; any
  `RouteSensitive` row blocks until localized. Append-only turns keep
  `ExactTranscripts`, and bitwise exactness is proven by the same-route twin
  on both artifacts.
- **C2 decision D2, "≤ 1.025".** Read as the paired-ratio 95% CI upper bound at Q3's
  sequential precision stop.
- **C2 decision D3, block count.** Q3's sequential stop, not fixed blocks.
- **C2 decision D4, anchor identity.** Omits C1's `conversation` field.
- **C2 decision D5, cancellation.** Discards the request's own pending anchor.

Design approvals and Qwen3.6 decisions (operator, 2026-11-03):

- **Q4-GEMMA-KERNEL (#659) and Q4-GEMMA-ATTN (#668).** Design
  `docs/research/adr063-gemma-tiled-attn-design.md` approved. The kernel
  alone cannot produce a Gemma Win; Q4-GEMMA-ATTN closes only through a
  combined candidate.
- **Q4-TRACE (#663).** Design `docs/research/adr063-q4-trace-design.md`
  approved with the three §7 answers as recommended by its review.
- **Q1-QWEN36-SPEC (#653) and Q1-GEMMA-SPEC (#665).** Design
  `docs/research/adr063-q1-spec-design.md` approved. The proposed exception
  that would report a Gemma `RouteSensitive` result without localizing it is
  rejected: Gemma `RouteSensitive` results must be localized, and #665's
  scope includes the verifier-route localizer (§2.6), landing after
  Q1-GEMMA-T2LOC's supervisor change.
- **Q4-DS4-RACE (#667) and ADR-066 N-IQ2XXS.** Design
  `docs/research/adr066-iq2xxs-subset-design.md` approved.
- **Qwen3.6 (#662).** The conversion audit found no defect
  (`~/.local/state/hf2q/v0130/q4-qwen36/conversion-audit/report.txt`). The
  first third-party check is unsloth UD-Q4_K_M under the canonical setup on
  both engines.

This amendment establishes no speed win, quality result or memory admission.

### Gemma 4 gap stories and the rope_multi pin (operator decisions, 2026-11-04)

The Q4-GEMMA profile result (#660,
`docs/research/adr063-q4-gemma-profile-result.md` §7) found that
Q4-GEMMA-ATTN alone cannot produce a Gemma Win and that no v0.1.30 story
covered the other levers. The operator decided on 2026-11-04 to add four
v0.1.30 stories, each a predecessor of Q4-GEMMA-ATTN's closing combined
candidate: Q4-GEMMA-COLD, Q4-GEMMA-PREFILL-MOE, Q4-GEMMA-DECODE and
Q4-GEMMA-SPEC-SERVE.

Design approvals (operator, 2026-11-04), with the accepted decisions as
recorded in each design:

- **Q4-GEMMA-COLD.** `docs/research/adr063-q4-gemma-cold-design.md`
  approved with §12 D1–D4 as recommended. D1: Class A only; a failed bitwise
  gate stops the story for localization, and no Class B joint cold route
  ships without a separate decision and the Q1 §3.5 campaign. D2: clean
  cutover of every cold multi-sequence caller to the per-sequence cold
  attention (iter-G(a) lane floor 33 rows; the installed long-prompt prefill
  is exact only for its cursor-0 chunk; G-S3 shows no no-boundary
  regression; graft-bound callers go per slot). D3: default on, kill switch
  `HF2Q_GEMMA_JOINT_COLD_PREFILL=0`. D4: Q4-GEMMA-ATTN predecessor and a
  dated #660 §7 note. If D1 stops the story, the combined candidate proceeds
  without it (kill switch default off or code reverted), the root restates
  the Win arithmetic here before the closing campaign, and the story goes to
  the operator as `needs_decision`. On the revised basis below, the levers
  without Q4-GEMMA-COLD come to about 2.0–3.7 s (2.24–4.05 s less the cold
  turn's 0.20–0.35 s) `[INFERENCE]`; the design's 1.9–2.9 s used the #660
  basis.
- **Q4-GEMMA-PREFILL-MOE.**
  `docs/research/adr063-q4-gemma-prefill-moe-design.md` approved with §12
  D1–D7 as recommended. D1: Class A changes only. D2: one story in three
  PRs (P1 hf2q, P2 mlx-native with a linked contract issue, P3 pin and
  integration). D3: `HF2Q_GEMMA_PREFILL_MOE_DOWN` is deleted at promotion or
  in the stop-path PR; `HF2Q_MM_ID_SCHEDULE=grid` stays one release as a
  native diagnostic. D4: stop path if `S_pred(P1) + S_pred(P2) < 0.05 s` per
  run. D5: the story's own acceptance is a category saving with no makespan
  regression; the Win is measured on the combined candidate. D6: DeepSeek-V4
  and Qwen MoE bitwise gates plus one `ExactTranscripts` smoke each at the
  pin. D7: P2 ships with Q4-GEMMA-KERNEL's mlx-native release or alone,
  whichever is ready first, sequenced so receipts and digests regenerate
  once per pin.
- **Q4-GEMMA-DECODE.** `docs/research/adr063-q4-gemma-decode-design.md`
  approved with §13 decisions 1–5: Class A-only scope (the peer's MMA,
  `mul_mv_ext` and K-split routes excluded; the S0 spike runs before M0); if
  M2 misses 0.96, prefer (a) accepting the measured saving and re-budgeting
  the Win across the other Gemma stories, then (b) a Class B few-row MMA
  spike only if M0-P shows a large enough per-kernel peer advantage, with
  (c) a re-scope of the Gemma Win last; the M0-P peer reference build runs
  before adoption is frozen; decode attention is added as lever A; the
  `HF2Q_GEMMA_DECODE_KERNELS` switch and promotion rule follow §4.8.
- **Root decision: decode attention is lever A of Q4-GEMMA-DECODE**
  (A1/A2, Class A, bitwise against today's vec-hybrid route, with adopt/drop
  thresholds and the peer's decode attention in M0-P). The #660 plan left it
  unowned.
- **Q4-GEMMA-SPEC-SERVE.**
  `docs/research/adr063-q4-gemma-spec-serve-design.md` approved with its §13
  decisions. D1: verify route R-A only; on a G0 miss the windowed verify,
  rank-phase code and walk are deleted unless a zero-cost drafter reuses
  them. D2: Q8_0 drafter artifact by default, BF16 only if M1's
  accepted-per-drafted drop exceeds 0.02. D3: managed download with a
  sha256 pin, `--gemma-assistant` as the override. D4: the peer's MT arm is
  re-qualified with hf2q's Q8_0 assistant GGUF as a tie-break arm, and the
  faster becomes the peer arm of record for the closing campaign. D5: after
  promotion the default is `mtp` when an assistant is resolved, `0` the kill
  switch. D6: G0 threshold 0.40 s, re-checked at M1. D7: `mtp-decode-only`
  is the one pre-declared extra arm if S1's TTFT fairness fails. Depth K = 1
  is measured first and is the working value; one cost controller per
  decoding width, recreated at each admission; only greedy requests
  speculate.

**Revised Gemma lever budget (estimate).** A Win needs about 3.0 s off the
14.87 s makespan (0.95 × the peer's MT 12.48 s). From the designs'
re-estimates, low/high: precise warm attention 1.2/1.4 s (#668); cold turn
0.20/0.35 s (down from 0.4–0.5 s); prefill MoE 0.08/0.28 s; serving
speculation 0.40 s (G0) / 0.50 s; decode kernels including attention 0.36 s
(lowest adoptable, f = 0.78) / 1.52 s (all levers at threshold, f = 0.90;
decode design §5.1). The levers together come to about 2.2–4.0 s
(2.24–4.05 s) against the 3.0 s needed `[INFERENCE]`. Interactions
(speculation removes target steps and shrinks the decode levers) pull the
high end down. A Win is not assured; any shortfall is re-budgeted across
the Gemma stories (Q4-GEMMA-DECODE decision 2), and only the combined
candidate gives a valid total.

**mlx-native `rope_multi` partial-rotary defect (found 2026-11-04).** The
`rope_multi` kernel was defective for partial rotary dimensions. The fix is
mlx-native PR #86, merged as `dc236dc1`, to be released as 0.15.18 (not yet
published at this amendment). Operator principle: don't test something
known defective. All Qwen GPU validation, the ADR-065 B1 (#503) baseline run
on main and the Q1-QWEN36-SPEC b6 run wait until 0.15.18 is published and
hf2q pins `=0.15.18`; B1 then baselines the fixed main. ADR-065's J1/P4b pin
moves to `=0.15.18` as well. Q1-QWEN38-RERUN's delivered result (restore
exactness on `d817dffa`/`5504f4ee`, 0.15.16) stands as invariance evidence,
because state-restore exactness does not depend on the `rope_multi`
numerics.

### Q4-GEMMA-LMHEAD-MMA opened (operator decision, 2026-11-05)

Q4-GEMMA-DECODE M0 dropped every exact dense lever (D1, D2, F1), standalone
E1 and decode lever A, and adopted F2 only. M0-P showed one large per-kernel
peer lead: the Q6_K lm head at width 4 runs 1,181 µs in the peer's few-row
MMA against 2,152 µs for the exact stock kernel, cold. Under Q4-GEMMA-DECODE
design §13 decision 2(b) the operator opened Q4-GEMMA-LMHEAD-MMA (story
section above; design `docs/research/adr063-q4-gemma-lmhead-mma-design.md`).
Framing (operator): the MMA reorders the accumulation the way the peer does
and its output stays within rounding of today's; the Class B gates are the
program's standing nothing-broke checks, not a claim that the route moves
away from the peer. Root rulings on review: M2 ≤ 0.98 for this lever alone
(the 0.96 Win share is judged on the combined candidate), the serial head's
baked record bypassed at head call sites only, and a width-1 bar of < 0.5% of
the n1-stage makespan used in both the spike and M1.

**Outcome (2026-11-05): stopped at the spike.** The MMA port passed every
exactness leg on both metallib paths but missed the §5 speed legs: projected
V2 saving 0.237 s against ≥ 0.30 s, and a width-1 penalty of +103 µs against
≤ 35 µs (design §14; receipts
`~/.local/state/hf2q/v0130/q4-gemma-lmhead-mma/spike-57dd7c0-20261105T195427Z/`).
Nothing was merged or released; the story closes as Won't do.

### Host toolchain (operator decision, 2026-11-05)

rustc 1.99.1 from 2026-11-05 07:14Z for builds and the four-agent fixture
compile; earlier receipts used 1.94.0; no cross-run transcript comparison
mixes toolchains.

### Four-agent workload normalization (operator decision, 2026-11-05)

The canonical four-agent workload did not send the same prompts in every
run. Its `run_tests` result carried the raw cargo and libtest output: the
fixture's absolute path (run directory and agent id) and the wall-clock
durations (`Finished … in 0.15s`, `finished in 0.00s`). These are a few
digit tokens in an otherwise identical transcript. Q4-GEMMA-COLD G-S2 run 011
failed on them alone. Its turn-4 prompt matched the other ten runs in length
(1,247 tokens) but not in content, and all four Gemma lanes read
`src/ports.py` (one `src/ports.rypts`) instead of `src/ports.rs`. A
deterministic replay reproduced 011 on its binary and on `main` `0d0179b7`
for every warm composition (3 joint + 1 per-slot, 4 joint, serial) and for a
cold full prefill. Per-token logprobs were bitwise equal across the warm
compositions, and run 010's prompt chose `rs` by 13.8–15.4 nats everywhere.
The cause was the input, not prefill route numerics or state corruption.

Decision (operator, 2026-11-05): from the ADR-065 B1 (#503) baseline run on,
the model sees a run-invariant `run_tests` result. `qwen38_n1_tools.py`
replaces the owned tool root with `/owned` and the cargo working directory
with `/cargo-cwd`, sets every cargo and libtest duration to `0.00s` and drops
the OS thread id from failing-test panic lines (Rust ≥ 1.91). The
tool receipt keeps the verbatim output as `raw_output`, and error codes and
the libtest result marker are unchanged. `read_file` returns fixture bytes
and `write_file` a content digest, so neither changes. Results recorded
before the change stay as recorded and are not re-scored. A transcript
comparison across runs (ExactTranscripts, G-S2, T-J1-8, TRACE T/O, the DS4
non-regression run, B1) is valid only between runs on the same side of the
change, so those runs use a harness at or after it.

### Q4-GEMMA-ATTN outcome: TiledF32 falsified (operator ruling, 2026-11-05)

The precise tiled warm route (`TiledF32`, `HF2Q_GEMMA_WARM_TILED_ATTENTION=precise`)
was integrated on PR #698 at mlx-native `=0.15.18` and gated on the M5 Max
(head `bb2e50ea`, evidence `~/.local/state/hf2q/v0130/q4-gemma-attn/gates-bb2e50ea/`):

- G1 Metal fixtures (declared in-place reads, A7 stage/global layout): pass on
  both metallib paths.
- A1 joint == per-slot and A2 determinism (both paths): bitwise pass.
- Localization: layer-0 `attn_rel` nmax 4.95e-7 against the frozen 2e-5
  (`TiledF16` 3.63e-4). The kernel met its precision target.
- E1 (all 952 suffix rows, against `VecHybrid`): 72 near-tie-policy failures
  (top-1 flips at a vec margin ≥ 0.10 nat), 121 flips, 352 rows with D > 0.25,
  max D 4.42 nat (`TiledF16`: 86 / 135 / 383 / 4.06). These come from the
  rows logged before the run stopped on an A3 harness bug, fixed in
  PR #698; A4 (the vec-hybrid control) passed. Downstream the
  layer-0 difference grows to 6.3e-4 at layer 1 and 0.249 at layer 7, and the
  final logits differ by up to 2.00 (`TiledF16` 4.07).

Ruling: per design §8.5 the route is falsified for default use. hf2q removes
the `precise` value and the `TiledF32` variant in a clean cutover (PR #698);
the precise kernels stay in mlx-native 0.15.18. The design records the
outcome in its §12.

**Revised Gemma lever budget without it (estimate).** Cold turn 0.20–0.35 s,
prefill MoE 0.08–0.28 s, decode kernels including attention 0.36–1.52 s,
serving speculation 0 (parked): about 0.64–2.15 s against the 3.0 s a Win
needs `[INFERENCE]`. A Win is not reachable on these levers alone.

**New story (operator, 2026-11-05).** A warm-prefill attention kernel that is
u32-bitwise equal to the vec-hybrid warm route (Class A), so it needs no
envelope or quality campaign: Q4-GEMMA-ATTN-EXACT (story section above;
design `docs/research/adr063-q4-gemma-attn-exact-design.md`, approved
2026-11-05, S0 feasibility spike first). Q4-GEMMA-ATTN keeps owning the
Gemma closing campaign.

## Consequences

### Positive

The program concentrates on same-GGUF wins for four real coding agents on M5 Max/128 GB. Quality, memory safety and peer-best performance become independent, auditable gates. Invalidated experiments and already-shipped work stop consuming the implementation queue.

### Negative

Four artifact lanes and sequential runtime isolation make real-model evidence expensive. Kernel wins can require a separately published mlx-native revision. Statistical uncertainty may yield an inconclusive outcome instead of a release superiority claim.

### Neutral

The program scope, F1/F3 execution entry and separately reviewed F2/Q1/Q2/C1 source implementation entry are Accepted. Other implementation designs require their own approval; acceptance of these entries is not acceptance of implementation or performance claims. Historical pin/source drift is repaired by F1/F3 through existing authority, never silently by benchmark configuration.

### Native command diagnostic metadata capacity

The legacy F2/Q2 trace contract keeps its256MiB combined serialized server+client ceiling. One explicit closed opt-in, `metadata_budget_profile="native_command_capture_v2"`, may select a finite2GiB combined serialized ceiling ONLY when `native_observer` is present and `diagnostic_case=true`. Absence retains legacy behavior; unknown/null profile or a profile without that diagnostic observer refuses before output creation. The exact profile is included in the trace header and must match the independently frozen artifact expectation. This is not a general trace-budget override, physical-RAM authority, production-default activation or primary performance approval.

Reason: actual diagnostic `f2-qwen38-native-trace-operating90-20261028T010639Z` exhausted116,391,936 server JSONL bytes (111MiB after the native carve-out); it retained116,391,761 bytes/187,748 records without a final native status/footer. Strict validation also found unchecked asynchronous owner Drop and scope loss on command rotation before quota. These earlier lifecycle failures and missing final counters remain failures/Unknown; the new profile never rescales, rescues or qualifies that historical attempt. Native fixes require independent proof and exact published dependency pinning.

The exception enlarges serialized capacity only: minimum client128MiB, native event1,000,000, live native metadata16MiB and bookkeeping1MiB caps remain fixed. Native reservations remain carved from the declared server amount. With client128MiB and the maximum native reservation, total2GiB allows1903MiB server JSONL. This finite operating choice is not a measured wholecase event upper bound or a guarantee of success. Runtime packets must separately charge extra allowed I/O/file-cache exposure (up to1.75GiB above legacy) and actual parser/runtime ownership; no whole physical-memory equivalence is implied.

The later public-native0.15.8 full Qwen3.8 coding capture reached the independent1,000,000 native-event ceiling: its complete footer reports2,524,480 dropped events, invalid capture, pending0/limit1024 and unknown scopes0. The HF serialized writer did not exhaust its quota (702,914,119B/1,230,478records). No earlier retained structural fault was found by the independent streaming audit, but missing native tail events still make the WHOLE trace FAILED. These measured counters are capacity evidence, not a numerical/performance result or permission to rescore that capture.

A NEW closed opt-in, `metadata_budget_profile="native_command_capture_v3"`, permits at most8GiB combined serialized server+client bytes and16,000,000 total historical native events ONLY for an explicit native diagnostic observer. Legacy absence remains256MiB/1,000,000events and v2 remains2GiB/1,000,000events, even after the native dependency permits a higher total. v3 requires the published native0.15.9-or-later primitive, exact registry pin and clean-source verification before a consuming result can qualify. Header and independently frozen expectation MUST bind the exact profile; unknown/null profiles, missing/non-diagnostic observers, one-past byte/event limits and arithmetic overflow refuse before output creation.

The new total-event limit is a finite scalar counter ceiling, not live-memory authority or preallocation. Live native metadata16MiB, HF native bookkeeping1MiB, pending owners1024, capture ownership4096, minimum client128MiB and checked completion/zero-loss/final-footer rules remain unchanged. Caller-selected event and serialized quotas must be explicit and finite; a permitted maximum is not a measured wholecase upper bound or guarantee of success. No sampling, omitted tail, compression, narrower coding workload or favourable fixed-window search may substitute for the complete declared diagnostic.

A v3 runtime packet MUST separately charge its additional allowed serialized I/O/file-cache exposure (at most7.75GiB above legacy), actual producer ownership and offline parser ownership. Offline full-list parsing is allowed ONLY after the full-model runtime is proved unloaded with owned groups/listeners empty, under a separately reviewed prospective process RSS/footprint, available-memory, normal-pressure and deadline guard; no fixed parser fit is inferred from file size. Failure of any quota/guard remains Failed/Inconclusive, with raw evidence retained. These operating charges prove no whole physical-memory admission or observer-OFF primary/speed equivalence.


Both actual Rust config validation and strict Python artifact validation must reject legacy256MiB+1, v2 2GiB+1, v3 8GiB+1, unsupported or unbound profile, missing/non-diagnostic native observer and client under128MiB. Native live metadata above16MiB refuses in every profile; native events above1,000,000 refuse for legacy/v2 and above16,000,000 refuse for v3. No compression, omission, sampling, trimming, wrapping, favourable window, changed workload/context/output or rescore is permitted. Every quota/identity/lifecycle/unknown/drop failure still invalidates the comparison; complete native trace, Q2 certification and independent quality/performance acceptance remain open gates.

## Successor watchlist and tracked successor scope

Production mixed-prefill policy after S1; queue-aware locality after traces; a native DFlash2/DSpark port after D2; lattice trees at a global four-slot row budget; NAX attention; scratch-lifetime synchronization removal; output/LSE shared-segment attention. Each needs its own measured deciding result, memory budget and approved design. ANE conversion/offload, approximate recurrent/nonprefix replay and custom-training-first plans remain excluded.

Tracked successor scope, v0.1.31 (operator-confirmed 2026-11-03; the last two items are carried from the planning reviews; not v0.1.30 speed-program scope):

- **Q2 v3:** the `queued_forward_demand_v3` population counting rule with fresh traced n1/n4/3+1/2+2/1+3 cohorts (consuming the Q4-TRACE witness), the F1 multi-owner-epoch rule with a two-owner control before any DeepSeek-V4 or Gemma window is scored, the first positive hardware `q2_certify` certificate and the hardware one-slot/four-client width negative control.
- **Q3/F2:** mixed stages against the best-valid (MTP) peer, the F2 fixed-six three-arm campaign with its cancellation/retry, divergent-edit and request-order permutation scenarios, and a separate QE identity.
- **Gemma/DeepSeek multi-anchor parity:** their multi-depth anchor implementations, with the C1 carry-overs O2–O4 (Gemma production quota sizing, Gemma window position, DeepSeek window from the plan).
- **DeepSeek selection witness:** a published native DeepSeek selection witness and its hf2q wiring, which would turn Q1-DS4's routing margins from Missing into measured.

## Links

- [Research and primary sources](../research/apple-silicon-speed-parity-2026-10-22.md)
- [Claude critique and dispositions](../research/apple-silicon-speed-parity-2026-10-22-critic.md)
- [ADR-049: agentic state reuse](ADR-049-agentic-state-reuse-and-mixed-cooperative-prefill.md)
- [ADR-040: batching](ADR-040-continuous-batching-reopen.md)
- [ADR-042: DeepSeek-V4](ADR-042-deepseek-v4-flash-rust-native.md)
- [ADR-044: Qwen3.8](ADR-044-qwen38-native.md)
- [ADR-059: graft coordinates](ADR-059-kv-grafts.md)
- [ADR-062: serving parity and persistence](ADR-062-serve-family-parity.md)
- [ADR-019: encoder ownership history](ADR-019-mlx-native-encoder-architecture.md)

This ADR consolidates research and prospective work; it does not supersede accepted family-specific safety contracts. F3 reconciles historical ADR drift in the same work that updates those sources and their trackers.

### Prospective queued-forward demand and bounded quantum diagnostic

The full native0.15.9 trace `061436Z` is lifecycle/capacity-valid but its FIRST
fixed2s window is Unqualified: all4 actual input-ready owners existed, but
serialized720-row transactions serviced only slots0/1 during that window.
Unknown admission snapshots did not fabricate readiness. This failed result
MUST remain unchanged; a narrow input-RMS interval is not whole-forward GPU
service or utilization.

A separate explicit `forward_demand_policy` may select ONLY
`population_policy="queued_forward_demand_v3"` with
`qwen_prefill_chunk_tokens=128`, ONLY with a native diagnostic observer. Its
actual Qwen owning-worker acknowledgement MUST precede admission. Header,
config digest and independent expectation MUST bind the exact pair. Legacy
absence, `ready_now_v1`, `forward_demand_v2`, production2048/4096 quanta and
all finite v3 metadata ceilings remain unchanged.

This new policy distinguishes actual input-ready queued/selected-forward
DEMAND from executor availability: waiting slots do not own another slot's
transaction, and unavailable executor remains runnable=false. Output-only,
terminal, cancelled, stale and Unknown states MUST NOT be credited. Demand
is never GPU Running, physical B4, full-body membership or progress. Keep the
first fixed2s window/end by30s, >=80% target coverage, actual new per-owner
progress/native witness and eventual healthy exact checked completion. A new
128-quantum experiment MUST retain the full canonical4-client28HTTP/24-tool
workload, context/output/sampling and all failure outcomes; no favourable
window/search/rescore or quota expansion. Source controls do not establish
chunking numerical parity, model quality, timing success or primary acceptance.
