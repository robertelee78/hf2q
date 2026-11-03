# ADR-063: Apple Silicon four-agent GGUF speed and coherence

- **Status:** Accepted program scope, F1/F3 execution entry, the separately reviewed F2/Q1/Q2/C1 source implementation entry and the 2026-11-03 measurement-only entry (see "Speed program, Q1 successors and release mapping"); other implementation designs require separate approval.
- **Date:** 2026-10-22
- **Updated:** 2026-11-03 — operator decisions for the v0.1.30 speed program (see "Speed program, Q1 successors and release mapping"): Q4 rewritten as the "clearly faster than the peer" umbrella, closed by the new per-family stories Q4-QWEN38, Q4-GEMMA, Q4-DS4 and Q4-QWEN36, with Q4-GEMMA-KERNEL (mlx-native) and Q4-TRACE; Q4-DS4 consumes the IQ2_XXS subset of ADR-066 N1/N3/N4 pulled ahead of ADR-066 B2; Q1 successors Q1-DS4, Q1-GEMMA-T2LOC, Q1-QWEN38-RERUN, Q1-QWEN36-SPEC and Q1-GEMMA-SPEC added; C2 amended (built regardless of measured demand, on ADR-065 P4a/P4b with J1 joining v0.1.30, frozen limits, delivery v0.1.30); remaining verification leftovers recorded as tracked successor scope, v0.1.31. 2026-11-01 — F2 closes on option B (per-family Q3 verdicts of record with controls, quality gate and fairness; see "F2 closure scope"). 2026-10-30 — Q5 rewritten as per-family calibrated admission with live out-of-memory stops; F2 and Q1 are no longer closure gates, so stories close on per-family, independently reviewed evidence; Q3/Q4 accept per-family peer wins where a peer arm exists; C2 moves to v0.1.29. Earlier 2026-10-27 parallel-source update unchanged.
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

- Frozen limits: divergent-turn worst-slot semantic TTFT, paired against cold on the same binary with the anchors kill switch, has a Q3 Williams-block (5–20 blocks) 95% CI that excludes 0 in the improving direction; append-only four-agent makespan and worst-slot TTFT p95/SSE-gap max ratios are ≤ 1.025 against anchors off; process footprint with anchors on is ≤ 1.25× a fresh Q5 calibration.

- Fresh Q5 calibration for Qwen3.8 and Qwen3.6 with the declared anchors per slot at the declared workload precedes any C2 hardware run.

- Qwen3.6 is judged by cold-versus-resumed bitwise exactness and `ExactTranscripts` against anchors off, not by task pass rate, while no workload on which Qwen3.6 passes is declared (Q4-QWEN36). Qwen3.8 additionally passes the paired task-quality gate.

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

Outcome: The program understands why hf2q is slower than the peer on the same Mac and becomes clearly faster: per family, the published hf2q finishes the canonical four-agent coding task at least 5% sooner than the peer in its best valid configuration, with quality gates passing.

Requirement: The program MUST explain, per family, where hf2q loses time against the peer on the M5 Max 128 GB and MUST deliver, for each of Qwen3.8, DeepSeek-V4, Qwen3.6 and Gemma 4, a configuration whose Q4 verdict is Win, with the declared quality, fairness and memory gates intact. Q4 is the umbrella story; it closes through the per-family stories Q4-QWEN38, Q4-GEMMA, Q4-DS4 and Q4-QWEN36.

Acceptance:

- Q4 win = `scripts/four_agent/q3_stats.py` `family_verdict` == Win: the hf2q/peer four-agent makespan ratio, peer in its best valid configuration, has a 95% CI upper bound ≤ 0.95 (`PEER_MARGIN`), under the Q3 Williams-block design (5–20 blocks). The F2-tier "Win" label, which is against v0.1.28 stock only, is not a Q4 win.

- The paired task-quality gates on the exact candidate binary pass against v0.1.28 and against the peer (against the native ds4 arm where the same-GGUF peer is unavailable), and all deterministic task/tool/isolation gates pass.

- Fairness: worst-slot TTFT p95 and SSE-gap max ratios against v0.1.28 stock stay within the stock fairness band 1.025 (`NEUTRAL_RATIO`). Vs-peer TTFT ratios are disclosed, not gated; SSE gaps against the peer are shown as not comparable.

- Each family's accepted configuration passes a fresh Q5 calibration on the release candidate.

- One-slot (n1) regressions against v0.1.28 or the peer are disclosed with the release, not gating. Single-slot or microkernel speed alone cannot satisfy this story.

- Each family story records a per-stage attribution of the hf2q-versus-peer gap measured on the same run, so the explanation of the gap is evidence, not inference.

- Give separate speed/quality/fairness verdicts for vanilla Qwen3.8-27B dense, DeepSeek-V4, vanilla Qwen3.6 MoE and Gemma 4 on M5 Max/128 GB. Each family's verdict is Win, Loss, Unavailable (no runnable peer) or Failed (hf2q task failure), with evidence; only Win counts; nothing is pooled or dropped. Latest peer must complete the declared workload for a valid performance verdict; peer task failures are explicit quality findings, never grounds for claiming a win against nothing. Native-format peer competition is reported separately from the same-GGUF peer verdict.

- Name the release candidate commit at closure. Per family, re-run the Q3 n4 (+n1) campaign on that candidate, or show `ExactTranscripts` equivalence to the measured binary. Record the exact winning candidate work keys and immutable commits as Q4 predecessors in the source/extraction/awa graph and obtain independent verdicts for their combined configuration; a prose-only winning subset cannot bypass candidate gates. Excluded/falsified experiments need a recorded disposition, not forced implementation.

- The Q1 successors (Q1-DS4, Q1-GEMMA-T2LOC, Q1-QWEN38-RERUN, Q1-QWEN36-SPEC, Q1-GEMMA-SPEC) do not gate Q4. DeepSeek-V4 quality for Q4 is the paired task-quality gate plus the S3 bitwise joint == serial check.

- Q4 cannot close until every family story is Win or the operator records a re-scope of that family in this ADR. A family that cannot reach Win keeps its verdict of record (Loss, Unavailable or Failed) with evidence.

Repository ownership: The integration owner coordinates two release chains: merge/publish/verify mlx-native updates first, pin and clean-build hf2q, then qualify and publish the hf2q crate plus signed standalone artifacts. Record both package versions/checksums, immutable source commits, native PRs and all four independent model verdicts.

Validation: Independent review of the combined candidate and raw receipts, followed by exact-artifact release verification on the target Apple host.

Predecessors: F3, Q2, Q3, Q5, Q4-QWEN38, Q4-GEMMA, Q4-DS4, Q4-QWEN36.

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

Outcome: hf2q finishes the canonical Qwen3.8-27B four-agent coding task at least 5% sooner than the peer in its best valid configuration (with its own MTP head), and the measured cause of today's gap is recorded.

Requirement: The program MUST attribute the Qwen3.8 hf2q-versus-peer gap by stage and MUST deliver a Qwen3.8 configuration whose Q4 verdict is Win.

Acceptance:

- Starting point (Q3 v11, hf2q `38485a23`): the MTP peer finishes 6.0% sooner (95% CI 4.2–7.9%), emitting 1.88× the completion tokens at a 2.00× per-output-token rate; hf2q's worst-slot p95 TTFT is 2.61× the peer's.

- Per-stage attribution of the gap on the same run, using the ADR-065 B1 (#503) baseline and fairness freeze and its E1–E8 (#504–#511) deciding measurements (stage time split, complete-round cost, counterfactual acceptance, makespan composition and eligibility), with the Q4-TRACE joint-decode witness. The distortion of each attribution is stated.

- The 2.6× worst-slot TTFT gap is decomposed into queueing, admission, prefill and first-token stages, and each stage is closed or disclosed with evidence.

- Speculation at four slots is decided from measurement (dense Qwen speculation defaults to off at four slots since #493; the peer uses its MTP head): the chosen setting is the one in the measured winning configuration, and its exactness follows the ADR-065 numerical and transaction contract.

- The family verdict meets the Q4 win definition and every Q4 quality, fairness (stock band 1.025) and fresh-Q5 gate on the release candidate.

Repository ownership: hf2q owns attribution wiring, scheduling, speculation policy and the campaign. Every Metal kernel, dispatcher and primitive parity test belongs in mlx-native and is consumed at a published, exactly pinned revision.

Validation: Independent review of the per-stage attribution and of the Q3-design hf2q-versus-peer campaign from raw receipts.

Predecessors: ADR-065 E8 (#511, G0 gate decision), Q4-TRACE, Q5. Consumes ADR-065 P4a (#526), P4b (#527) and any ADR-065 build story admitted after the E8 gate.

Priority: High. Core scope (first-tier family).

Delivery boundary: Release v0.1.30, or the first published hf2q release containing the accepted Qwen3.8 configuration.

### Q4-GEMMA-KERNEL. Provide a precise tiled warm attention primitive in mlx-native

Outcome: mlx-native provides a tiled warm attention primitive with 32-bit accumulation that Gemma 4 warm resumes can use without the `TiledF16` decision flips.

Requirement: mlx-native MUST provide, under a stated primitive contract, a tiled warm attention kernel with F32 accumulation and F32 output, published and pinnable before hf2q integrates it.

Acceptance:

- A linked mlx-native issue states the primitive contract before implementation: input layouts (F16 K cache, dequantized TQ-HB V, Q rows), sliding and global masks, scale and softcap, supported head dimensions, F32 accumulation and F32 output, and the tolerance against an F32 reference attention.

- The draft F32-accumulate/F32-output patch from the closed PR #552, kept on branch `exp/gemma-tiled-precise` (`docs/research/adr063-gemma-tiled-precise-mlx-native.patch`, with the diagnosis in `docs/research/adr063-gemma-tiled-precise.md`), is the starting point; the branch is retained until the kernel lands or is falsified.

- Primitive parity tests against an F32 reference attention and against the vec-hybrid route pass at Gemma 4 warm-resume shapes, and a microbenchmark compares it with `TiledF16` and `VecHybrid` at those shapes.

- The kernel is merged, released, verified on crates.io and available for an exact hf2q pin.

Repository ownership: mlx-native owns the kernel, its dispatcher, contract, parity tests, microbenchmarks and release. hf2q consumes it only through Q4-GEMMA.

Validation: Independent review of the contract, the parity and microbenchmark receipts and the published release.

Predecessors: none.

Priority: High. Core scope (second-tier family).

Delivery boundary: Published mlx-native release containing the primitive, pinned by hf2q for v0.1.30.

### Q4-GEMMA. Profile the Gemma 4 gap and integrate the precise tiled warm attention

Outcome: hf2q finishes the canonical Gemma 4 26B-A4B four-agent coding task at least 5% sooner than the peer in its best valid configuration, with the stock fairness gate passing.

Requirement: The program MUST, in parallel, (a) profile hf2q and the peer by stage on the same run and (b) integrate the Q4-GEMMA-KERNEL primitive in hf2q, and MUST deliver a Gemma 4 configuration whose Q4 verdict is Win.

Acceptance:

- Starting point (Q3 v9, hf2q `5d421a2c`): the peer finishes 13.4% sooner (CI 10.6–16.2%, 20-block cap), measured against the peer with speculation off (`--spec-type none`); the stock SSE-gap fairness gate fails at 1.047× (CI 1.036–1.058).

- Before the Q4 campaign, establish the peer's best valid Gemma 4 configuration (its speculation options), as D2 did for Qwen3.8. The gap may widen.

- Track (a): a per-stage profile of hf2q and the peer on the same run attributes the gap (attention, MoE, dispatch/encode, scheduling, SSE cadence) with stated distortion.

- Track (b): #544 set warm tiled attention default-off after `TiledF16` flipped a near-tie decision against `VecHybrid`; tiled-off costs Gemma about 7.8–9.2% (v9 review), and v7 with tiled on was still a Loss (1.058), so the kernel alone cannot be assumed to flip the verdict. In hf2q the Q4-GEMMA-KERNEL route stays within the Q1 decision envelope and exactness classes (including the 0.10-nat near-tie policy) at the canonical divergence coordinate and passes the paired task-quality gates before it can become a default.

- The stock SSE-gap max ratio returns within 1.025, and the family verdict meets the Q4 win definition and every Q4 gate on the release candidate.

Repository ownership: hf2q owns route selection, integration, profiling and the campaign, consuming Q4-GEMMA-KERNEL at an exact published mlx-native pin.

Validation: Independent review of the profile, the Q1-envelope receipts, the quality gates and the Q3-design hf2q-versus-peer campaign.

Predecessors: Q5, Q4-GEMMA-KERNEL (track (b) only; track (a) may start on Q5 alone).

Priority: High. Core scope (second-tier family).

Delivery boundary: Release v0.1.30, or the first published hf2q release containing the accepted Gemma 4 configuration.

### Q4-DS4. Run the peer on the ds4 DeepSeek-V4 file and compete on it

Outcome: Whether the same-GGUF peer can run ds4's DeepSeek-V4 file on the 128 GB host is established; if it can, hf2q finishes the four-agent task on that file at least 5% sooner than the peer.

Requirement: The program MUST establish whether the peer fits on this host with ds4's DeepSeek-V4 file and, if it fits, MUST support that file in hf2q and compare hf2q, the peer and ds4 on it.

Acceptance:

- Starting point (Q3 v7): Unavailable; the same-GGUF peer runs out of memory on hf2q's DeepSeek-V4 artifact. On different artifacts hf2q `38485a23` finishes 5.2% sooner than ds4 (never a Win).

- Step 1: run the peer alone on `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (86,720,111,488 bytes, sha256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`, re-hashed locally 2026-11-02; it sits in the ds4 peer directory `peers/ds4-c97964b/gguf/` under the hf2q data directory, not the `/opt/ds4/gguf/` path in `docs/research/adr063-q3-campaign.md`): `deepseek4` architecture, the same 1,328 tensors, IQ2_XXS/Q2_K experts. Record fit or no-fit with Q5 live-stop receipts, one runtime at a time.

- Fit = the peer, in its best valid configuration, completes the largest declared four-agent workload (4 slots × 262,144 context, matched settings) with Q5 `CalibrationPass`. Any context or slot reduction is an operator decision (Q5).

- A no-fit result closes step 1, keeps DeepSeek-V4 Unavailable with that evidence and returns Q4-DS4 to the operator as `needs_decision`; step 2 does not start.

- Step 2, only if it fits: consume the IQ2_XXS storage type, CPU dequantization and expert kernels of ADR-066 N1 (#580), N3 (#582) and N4 (#583), whose IQ2_XXS subset gains parity at DeepSeek-V4-Flash expert shapes (256 experts, top-6); no duplicate kernels. Add the hf2q loader for this file and the quality checks (tensor-load parity, paired task-quality gate). Then run hf2q, the peer and ds4 on that one file under the Q3 design.

- A per-stage attribution of the hf2q-versus-peer gap on the same run (prefill, decode, MoE/attention, dispatch, scheduling, SSE cadence), with stated distortion.

- The file is a benchmark comparison artifact, not a substitute for hf2q conversion. The family verdict meets the Q4 win definition and every Q4 gate.

Repository ownership: mlx-native owns the IQ2_XXS kernels and their parity tests through ADR-066 N1/N3/N4; hf2q owns the loader for this file, quality checks, Q5 calibration and campaign.

Validation: Independent review of the fit receipt and, after step 2, of the parity, quality and three-arm campaign receipts.

Predecessors: Q5; for step 2, the IQ2_XXS subset of ADR-066 N1 (#580), N3 (#582) and N4 (#583), pulled ahead of ADR-066 B2 (operator decision, 2026-11-03).

Priority: High. Core scope (first-tier family).

Delivery boundary: Release v0.1.30, or the first published hf2q release containing the accepted DeepSeek-V4 configuration.

### Q4-QWEN36. Investigate the Qwen3.6 task failure, then choose its speed workload

Outcome: The cause of the Qwen3.6 task failure is understood against Qwen's recommended agentic setup, and Qwen3.6 has a declared speed workload on which a Q4 verdict can be measured.

Requirement: The program MUST investigate the Qwen3.6 failure before choosing a speed workload and MUST then deliver a Qwen3.6 configuration whose Q4 verdict is Win on that workload.

Acceptance:

- Starting point (Q3 v10): Failed; the experiment and v0.1.28 both fail the canonical task (E0282, no repair turn), diagnosed as coming from the model weights (`docs/research/adr063-qwen36-task-failure-diagnosis.md`).

- The check used below is the 24-instance paired task campaign (Q1 numeric policy §3.5, `q1_decision_policy.py paired`) as in `adr063-qwen36-task-failure-diagnosis.md`, reporting instances passed per arm and the floor verdict.

- Diff the harness against Qwen's recommended agentic setup, with sources: thinking mode, whether past reasoning is resent, system prompt and tool-call format. Re-run the check with the recommended setup on the peer and on hf2q.

- Download third-party vanilla Q4_K_M GGUFs (ggml-org, unsloth) as diagnostic data only, not as a conversion substitute or a release artifact, and run the check on the peer and on hf2q with them.

- If Qwen3.6 stays BelowFloor under every setup tested, Q4-QWEN36 returns to the operator as `needs_decision` with that evidence.

- Record the workload decision with this evidence; changing the canonical workload is an operator decision recorded in this ADR. On that workload, a per-stage attribution of the hf2q-versus-peer gap on the same run (prefill, decode, MoE/attention, dispatch, scheduling, SSE cadence) is recorded with stated distortion, and the family verdict meets the Q4 win definition and every Q4 gate.

Repository ownership: hf2q owns the harness diff, templates/tool format, the checks and the campaign.

Validation: Independent review of the setup diff, both checks and the subsequent Q3-design campaign.

Predecessors: Q5.

Priority: High. Core scope (second-tier family).

Delivery boundary: Release v0.1.30, or the first published hf2q release containing the accepted Qwen3.6 configuration.

### Q4-TRACE. Add the Qwen joint-decode forward witness

Outcome: Every Qwen joint ordinary decode forward is attributable to its owning slots in a trace, for profiling now and for the later Q2 v3 proof.

Requirement: The program MUST deliver a Qwen joint-decode forward witness/trace hook that maps each joint ordinary decode forward to its owners.

Acceptance:

- `scripts/four_agent/f2_native_trace.py` reports `unmapped_operator_attempts == 0` for every `joint_ordinary_decode` forward of a full traced four-agent Qwen3.8 run (the Q2 closure QE §6 found 43,008 per run, as `forward_witness_unmapped` records with reason `NoActualInputManifest`).

- The hook is off by default and fails closed; with it on, transcripts are `ExactTranscripts` against it off, and its timing overhead is measured and stated.

- Its output is consumable by Q4-QWEN38 attribution; the Q2 v3 counting rule that consumes it stays successor scope (v0.1.31).

Repository ownership: mlx-native owns low-level trace support; hf2q owns the witness wiring, records and validation.

Validation: Independent review of a traced run with `unmapped_operator_attempts == 0` and the trace-on/off exactness control.

Predecessors: Q5.

Priority: High. Core scope.

Delivery boundary: Qualified implementation merged to main for v0.1.30.

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

### Q1-QWEN38-RERUN. Re-run Qwen3.8 Q1 T1/T3 on a build after #561

Outcome: The Qwen3.8 restore-versus-cold Q1 evidence is current for the restore cursor fix.

Requirement: The program MUST re-run Qwen3.8 T1/T3 on a Q1 test binary built after #561.

Acceptance:

- Runs use a Q1 test binary from main at or after `ea6caf46` (the retained build `1d200821` predates #561, `9fc7b31b`) and before ADR-065 P4b (#527) or J1 (#512) merges (P4b changes solo K3 rollback). Otherwise pin a commit that precedes both and record that relation.

- `exact` T1 vs T3 is `ExactOnCommon` with restored-64 extras only.

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

- Runs use a Q1 test binary from main at or after `ea6caf46` and before ADR-065 P4b (#527) or J1 (#512) merges (P4b changes solo K3 rollback). Otherwise pin a commit that precedes both and record that relation.

Repository ownership: hf2q owns the run and verdict.

Validation: Independent review of the endpoint receipts.

Predecessors: Q1, Q5.

Priority: Medium. Successor scope; does not gate Q4.

Delivery boundary: Measured evidence merged to main for v0.1.30.

### Q1-GEMMA-SPEC. Add the Gemma T5 speculative comparison

Outcome: Gemma n-gram speculative rows have a Q1 numeric comparison against ordinary rows.

Requirement: The program MUST design and build a Gemma T5 actual-execution adapter and run its comparison.

Acceptance:

- The adapter design is approved before implementation (the matrix currently maps Gemma T5 to `Missing`).

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
  closed by Q4-QWEN38, Q4-GEMMA, Q4-DS4 and Q4-QWEN36.
- **Gemma 4.** A precise tiled attention kernel (32-bit accumulation, in
  mlx-native) and a per-stage profile of hf2q against the peer proceed in
  parallel (Q4-GEMMA-KERNEL, Q4-GEMMA).
- **DeepSeek-V4.** Run the peer alone on ds4's file; if it fits, add IQ2_XXS
  support and compare hf2q, the peer and ds4 on that file (Q4-DS4). The
  IQ2_XXS subset of ADR-066 N1 (#580), N3 (#582) and N4 (#583) is pulled
  forward ahead of ADR-066 B2; DeepSeek-V4 consumes it, with no duplicate
  kernels.
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
predecessors are met: Q1-QWEN38-RERUN (ready now), Q1-QWEN36-SPEC, Q1-DS4,
Q4-DS4 step 1, the Q4-QWEN36 investigation and the profiling track of
Q4-GEMMA. Each implementation story or step (C2, Q4-TRACE,
Q4-GEMMA-KERNEL, Q4-GEMMA track (b), Q4-DS4 step 2, the Q1-GEMMA-T2LOC
hooks and supervisor change, the Q1-GEMMA-SPEC adapter) needs a recorded
design approval before implementation.

This amendment establishes no speed win, quality result or memory admission.

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
