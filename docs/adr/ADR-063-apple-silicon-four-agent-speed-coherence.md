# ADR-063: Apple Silicon four-agent GGUF speed and coherence

- **Status:** Accepted program scope, F1/F3 execution entry and the separately reviewed F2/Q1/Q2/C1 source implementation entry; other implementation designs require separate approval.
- **Date:** 2026-10-22
- **Updated:** 2026-10-27 — separately reviewed F2/Q1/Q2/C1 designs may be implemented in parallel while Q5 continues; original hardware qualification, acceptance and delivery predecessors remain. The dedicated-host resource policy and all-four speed/quality requirements remain unchanged.
- **Deciders:** Robert E. Lee (product owner)
- **Tags:** apple-silicon, m5-max, gguf, four-slot, speed, coherence, qe, v0.1.28
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

Conditional work means its deciding experiment must record implement, falsify, or successor-design with evidence. It is never permission to land a stub or promise an unmeasured speedup. Experiment receipts and implementation candidates use merge delivery; Q4 alone closes the v0.1.28 product release boundary; Q2 certifies the harness and Q3 independently proves the measured effects. A required runtime change needs a published native revision and clean registry-resolving proof. The final qualification depends on the actual winning subset, declared before candidate acceptance; it need not wait for unrelated or falsified hypotheses.


## Repository ownership and publication

This is one performance program spanning two owned repositories. Every Metal compute kernel and its Rust dispatcher belongs in `mlx-native` (`src/ops/`, `src/shaders/`); hf2q consumes that public API. `mlx-native` also owns buffer/resource safety, operation capability routing, graph/encoder submission mechanisms and primitive parity/microbenchmarks. hf2q owns conversion and quantization policy, model-family graphs, request scheduling, cache/anchor/cohort policy, tool semantics and model/workload acceptance. A family-shaped native primitive still needs an explicit reusable operation contract; model behavior does not move into the kernel library.

For every winning change that modifies `mlx-native`, record separate immutable native and hf2q commits and PRs. Complete native correctness/performance review and merge, publish a new native version through its release workflow, verify the crates.io archive checksum and downloaded-archive tests, then pin that exact registry version/checksum in hf2q `Cargo.toml`/`Cargo.lock` and its dependency-provenance verifier. Rebuild hf2q from a clean registry-resolving tree and qualify the combined configuration. A local path patch is spike evidence only.

Q4 MUST publish both updated packages when native changes are included: the required `mlx-native` crate/version/tag first, followed by the qualified hf2q v0.1.28 crate and signed/notarized standalone release artifacts. Verify public package/asset bytes, dependency provenance and clean installs through each repository's existing release gates. Native publication alone does not deliver the hf2q integration; hf2q publication against an unpublished native revision is forbidden. If the winning subset does not change native code, explicitly record the already-published native version and verify it rather than inventing a native release.

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

Predecessors: F3, Q5.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

### C2. Implement and qualify Qwen multi-depth prompt reuse

Outcome: Qwen3.6 and Qwen3.8 reuse the deepest valid earlier turn after agent context edits.

Requirement: The program MUST deliver Qwen3.6 and Qwen3.8 reuse the deepest valid earlier turn after agent context edits.

Acceptance:

- Implement bounded slot-local anchors with pending publication, atomic restore, lineage pruning, saved-logit equality hits, right-sized payloads and owned-byte telemetry.

- Prove cold-versus-resumed exactness on both Qwen artifacts with graft absent/present/identity change, speculation off/auto, cancellation, failed restore and four concurrent clients.

- Divergent-turn semantic TTFT improves over cold while append-only tasks, memory bounds and ongoing decoder latency stay within frozen acceptance limits.

- Primary Qwen3.8 evidence uses the vanilla official 27B dense model; grafted/abliterated variants remain separate compatibility arms. Family parity obligations for future Gemma/DeepSeek implementations remain explicit successors, not a universal multi-anchor shipping claim.


Repository ownership: hf2q owns Qwen model/cache integration and anchor selection; consume published mlx-native operations. Any required state-copy or storage primitive is implemented and parity-tested in mlx-native before a pinned hf2q integration.

Validation: Run four-slot divergent-edit state/lifecycle gates and paired F2 scoreboard with allocation accounting.

Predecessors: C1, F2.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

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

Predecessors: F2.

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

Predecessors: F2, Q5.

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

Predecessors: F2, Q1, Q5.

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

Predecessors: C1, F2, Q1, Q5.

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

Predecessors: F2, Q1, Q5.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

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

Predecessors: F2, Q1, Q5.

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

Predecessors: F2.

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

Predecessors: F2.

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

- Use latest peer in its best valid configuration as the mandatory peer arm and retain a separately labeled matched-settings diagnostic. Peer quality failures are recorded; a missing, failed or silently disqualified peer cannot yield a speed victory by default.


Repository ownership: Independent QE executes source-bound hf2q-versus-stock-versus-peer campaigns, recording both native and hf2q identities. Native microbenchmarks are diagnostic evidence, not substitutes for all four model-level speed wins.

Validation: Independent QE executes the certified harness against frozen stock/experiment/latest-peer binaries and publishes paired effect sizes, confidence intervals, rejected trials and verdict.

Predecessors: F2, Q1, Q2, Q5.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

### Q4. Accept and release a faster coherent four-agent Apple configuration

Outcome: The published hf2q configuration meets peer coherence and demonstrates a proven four-agent speed win for each of the four required model lanes.

Requirement: The program MUST deliver peer coherence and a proven four-agent speed win for each of Qwen3.8, DeepSeek-V4, Qwen3.6 and Gemma 4, with the declared fairness and memory gates intact.

Acceptance:

- All deterministic task/tool/isolation gates pass and predeclared stochastic noninferiority holds against eligible current peers.

- For each required artifact (Qwen3.8, DeepSeek-V4, Qwen3.6 and Gemma 4), require at least 5% lower four-agent makespan or 5% higher successful-task goodput than the latest peer in its best valid configuration, with paired uncertainty excluding a tie. Report both stock improvement and peer gap. Competitive/parity or inconclusive results are intermediate evidence and cannot satisfy final release acceptance.

- No accepted result worsens frozen p95 TTFT or maximum semantic-gap fairness ceilings; single-slot or microkernel speed alone cannot satisfy this story.

- Choose only winning qualified branches, re-run exact combined configuration gates, publish required native pins, and retain commands/receipts for exact release artifacts. Excluded/falsified experiments need a recorded disposition, not forced implementation.

- Record separate stock-to-experiment and experiment-to-latest-peer verdicts; report native-format peer competition separately from the mandatory same-GGUF peer verdict.

- Latest peer must complete the declared workload for a valid performance verdict; peer task failures are explicit quality findings, never grounds for dropping the comparator and claiming a win against nothing.

- Give separate passing speed/quality/fairness verdicts for vanilla Qwen3.8-27B dense, DeepSeek-V4, vanilla Qwen3.6 MoE and Gemma 4 on M5 Max/128 GB, in that priority order. Require all four wins; do not pool results or use a favorable model to hide another lane failure. Bind every claim to exact artifacts and workloads rather than universal untested superiority.

- Before final candidate acceptance, record the exact winning candidate work keys and immutable commits, add them as required Q4 predecessors in the source/extraction/awa graph, and obtain independent verdicts for their combined configuration. A prose-only winning subset cannot bypass candidate gates.

Repository ownership: The integration owner coordinates two release chains: merge/publish/verify mlx-native updates first, pin and clean-build hf2q, then qualify and publish the hf2q crate plus signed standalone artifacts. Record both package versions/checksums, immutable source commits, native PRs and all four independent model verdicts.

Validation: Independent review of the combined candidate and raw F2 receipts, followed by exact-artifact release verification on the target Apple host.

Predecessors: F3, Q1, Q2, Q3, Q5.

Priority: High. Core scope.

Delivery boundary: Published v0.1.28 release containing the independently accepted configuration.

### Q5. QE-prove four-slot memory safety on M5 Max 128 GB

Outcome: Every accepted configuration fits the operator host with measured system headroom and no out-of-memory experiment.

Requirement: The program MUST deliver every accepted configuration fits the operator host with measured system headroom and no out-of-memory experiment.

Acceptance:

- Before loading, admit model residency, four-slot KV/recurrent state, anchor committed+pending bytes, drafter state and peak in-flight scratch against one disjoint unified-memory ledger; avoid double-counting shared CPU/Metal pages.

- Use the dedicated host's available system RAM as authorized by the operator, replacing the earlier fixed 12 GiB reservation and 116 GiB cap. Subtract observed competing nonreclaimable use and explicitly budget observation/stop latency plus maximum in-flight allocation overshoot. Freeze the available-RAM policy, measured reaction margin and any source-justified family limit before loading and obey those limits; apply the same resource policy to comparison arms and stop on a projected breach. Physical capacity remains the hard upper bound, never a promise of availability. A reclaimable-file-cache working estimate is not unique physical-ownership evidence and MUST NOT silently refund a live allocation.

- Run stock, experimental hf2q and peer sequentially, unloading and verifying listeners/process/memory recovery before each load. Never co-reside DeepSeek runtimes or add a drafter/cohort that exceeds the measured budget. Keep exactly one hf2q process at a time, including metadata commands; those commands must finish before the server starts.

- QE uses synthetic allocation/admission negative controls and bounded real four-slot stress with the largest declared tool-result/context cases; stop on memory pressure, abnormal footprint growth or projected breach before OOM.

- Record logical budgets, OS footprint/pressure, Metal residency, transient high-water, swap/compression deltas and cleanup. Reject a candidate that fits only by silently reducing matched context, task size or output budget.


Repository ownership: hf2q owns family/request admission policy and a disjoint whole-process memory ledger. mlx-native owns allocation/view/residency/scratch lifetime guarantees and low-level counters. Review both sides so shared unified-memory pages are not double-counted; any native fix requires separate parity and publication evidence.

Validation: Independent QE reviews byte ownership and admission arithmetic, runs fail-closed budget controls, then conducts bounded M5 Max real-artifact memory stress without deliberately provoking OOM.

Predecessors: F1.

Priority: High. Core scope.

Delivery boundary: Measured evidence or qualified implementation merged to main for v0.1.28.

## Coverage and sequencing

F1 precedes F2; F2 is the source-bound baseline for every measured lane. F3 precedes C1 because the checkpoint contract must reflect current graft and persistence behavior. Q5 memory admission precedes risky model experiments and retained-state expansion. Qwen implementation C2 follows C1 and F2. Gemma/DeepSeek multi-anchor implementation, mixed-policy production work, queue-aware admission, new native block drafters/lattice trees, NAX attention and scratch-wait changes are successor candidates after the primary deciding measurements, not promised v0.1.28 features. S1 and D2 record deciding evidence before successor designs. Q1 is a correctness dependency for shared target batching, speculation and numerical kernels. Different edits to a shared file require merge coordination, not invented product dependencies.

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


Both actual Rust config validation and strict Python artifact validation must reject legacy256MiB+1, new2GiB+1, unsupported or unbound profile, missing/non-diagnostic native observer, client under128MiB and raised native live/event limits. No compression, omission, sampling, trimming, wrapping, favourable window, changed workload/context/output or rescore is permitted. Every quota/identity/lifecycle/unknown/drop failure still invalidates the comparison; complete native trace, Q2 certification and independent quality/performance acceptance remain open gates.

## Successor watchlist (not committed v0.1.28 implementation scope)

Gemma and DeepSeek multi-depth anchor implementations; production mixed-prefill policy after S1; queue-aware locality after traces; a native DFlash2/DSpark port after D2; lattice trees at a global four-slot row budget; NAX attention; scratch-lifetime synchronization removal; output/LSE shared-segment attention. Each needs its own measured deciding result, memory budget and approved design. ANE conversion/offload, approximate recurrent/nonprefix replay and custom-training-first plans remain excluded.

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
