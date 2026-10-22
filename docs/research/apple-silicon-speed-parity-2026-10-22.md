# Apple Silicon four-agent speed and coherence: research, 22 October 2026

Research cutoff: 2026-10-22. Audited hf2q: `5d1ffe7f1c5cf4ba5ffca452ec4bde738231beef` (v0.1.27). This is a source and primary-literature investigation, not a benchmark receipt. No model was loaded, no hardware performance was measured, and no new speedup is claimed here. The companion [ADR-063](../adr/ADR-063-apple-silicon-four-agent-speed-coherence.md) owns proposed work and acceptance intent; this report preserves the reasoning and evidence.

## Outcome and scope

The operator's primary workload is **four concurrently active agentic coding slots on M5 Max with 128 GB RAM**. The primary vanilla GGUF artifacts are Qwen3.6-35B-A3B MoE, Qwen3.8-27B dense and DeepSeek-V4. Linux/CUDA runtime support is outside scope: ideas from elsewhere must be translated into owned Rust/Metal wins on this host. The desired outcome is coherence and tool correctness at least as good as peers, with speed at least competitive and preferably better. A single-request tokens/second win is insufficient. Primary experiments have three frozen arms: current unmodified hf2q, experimental hf2q from that exact baseline, and latest upstream llama.cpp resolved at campaign start. Historical and native-format competitors are secondary labeled arms. The key outcomes are time to first semantic output, four-agent task completion time, successful tasks per unit time, p95/worst-slot latency, no starvation, and correctness after tool results and conversation edits.

There is no defensible universal October-2026 winner from the evidence reviewed. Kernel shapes, quantization, context, recurrent state, concurrency, and sampling can reverse rankings. We should select a measured combination of exact cache reuse, actual batched execution, verifier kernels, and proposal policy. Old ADRs are useful failure histories, not a current research shortlist.

## 1. Baselines must be rebuilt before prioritizing implementation

Three reference identities currently disagree:

| Surface | Observed identity |
|---|---|
| hf2q repository peer pin | `data/llama_cpp_pin.txt`: `e15384a5cb092b080c2a01c0b9e3f8635079d6df` |
| Local peer checkout | `/opt/llama.cpp`: `970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc`, 22 October |
| DeepSeek server parity harness default | `scripts/benchmark_deepseek4_server_parity.sh:25`: `15586e2d7165570fb3aa7c26e0d442e289ef69de` |
| Native compute checkout | `/opt/mlx-native`: `f92eb020c4d3f821700e648fbce15d6cf75c2cc6`, version 0.15.1 |
| hf2q native dependency | `Cargo.toml:174`: published `mlx-native = "=0.15.1"` |

The repository-pinned peer object was absent locally and GitHub's commit API returned HTTP 422 (no such commit), consistent with ADR-052 and the grammar fixture provenance. The default peer executable build/bin/llama-server is absent; build/common/build-info.cpp records old b9278 / 5306f4b3b metadata. Therefore neither source HEAD nor leftover build metadata establishes a current runnable peer. Do not alter that checkout or silently substitute the newest engine for historical acceptance evidence. Retain the unreachable hash only as historical provenance, propose a reachable validated replacement through the existing pin workflow, and establish an isolated rebuilt latest-peer lane, with binary hashes and build flags. Inspecting the native checkout does not prove registry-archive equivalence.

For same-format comparisons, use the same **hf2q-produced GGUF**, tokenizer, rendered prompt tokens, model configuration, sampling transformations, context, slot count, memory allowance, and output budget. Keep both a matched-settings diagnostic and a latest-peer-best-documented performance configuration, including supported speculation and complete MTP tensor retention in the shared GGUF. A peer with speculation silently disabled is not the definitive competitor. Alternative native formats can inform product competition, but must be labeled separately and pass quality equivalence; MLX affine 4-bit and GGUF Q4_K_M are not the same artifact.

Current Apple peers also include MLX and emerging specialized engines. Important dated primary evidence:

- [BaseRT v0.4.0](https://github.com/basecompute/baseRT/releases/tag/v0.4.0), **22 October**, tag commit `24badc8dd173e4111b64996a718d0e056405ea62`, supersedes July-paper performance assumptions. Its GGUF K/IQ support makes it a same-artifact secondary candidate, not automatically a native-format-only arm. Its [v0.3.0 serving release](https://github.com/basecompute/baseRT/releases/tag/v0.3.0), **8 October**, introduces shared-forward continuous batching and prefix reuse. These are author claims to reproduce, not hf2q baseline results.
- [Kapitsa benchmark documentation](https://github.com/nace-ai/kapitsa/blob/c490f8159622bb2d356b21c8d670d668997cbfb4/docs/benchmarks.md) includes October 17–18 Apple runs, runtime settings, contention flags and workload-dependent outcomes. Freeze code and result files before comparing.
- [MLX source](https://github.com/ml-explore/mlx/commit/c6abf0dbb4fc141fad739b32ae486c06728e49e5), **22 October**, adds quantized small-row matrix kernels. Latest stable and current main must be separate reference configurations.
- [Apple's 15 October MLX report](https://machinelearning.apple.com/research/mlx-distributed-inference) reinforces hardware/context sensitivity. Its M5 Ultra and multi-Mac results do not establish speed on the current M5 Max. Four agent slots are not four physical Macs; distributed inference is outside this program.

## 2. What the ADRs actually establish

| ADR | Declared standing and actual evidence | Disposition |
|---|---|---|
| 049 | Proposed, updated 22 August; single-anchor storage and mixed-phase cooperative bypass still exist | Highest immediate agentic latency opportunity; update for grafts before executing |
| 037 | In progress, 22 May; EAGLE-3 infrastructure exists, E2 training/E7 paired qualification TODO | Reuse useful verifier infrastructure; do not presume custom training is current best investment |
| 039 | Draft, 23 May; scientific premise and architecture claims contain errors | Correct the source; no implementation queue based on asserted stacked gains |
| 010 | Parity deferred, speed work shipped; later F64 reconciliation invalidates router-only fix | Numerical diagnostics, not a new router projection kernel |
| 011 | Proposed header and Phase-2-next text contradict current tiled/block-mask dispatch | Reconcile status; do not reimplement tiled attention or tile skipping |
| 019 | Proposed overall, early fusion phases implemented; chunk scratch lifetime wait remains | Profile current lifetime/synchronization tail; no greenfield EncoderSession rewrite |
| 016 | Proposed-Accepted, 26 April; no current hf2q CoreML integration | Lower-value for four text agents; conversion policy unresolved, exclude from main delivery |
| 036 | Shipped, 19 May | Conversion parallelization is not an open speed opportunity |
| 040/042/044 | Accepted implementations and substantial exact-artifact evidence | Preserve their serving contracts; historical measured gains do not establish today's four-slot parity |

### Critical corrections to the initial shortlist

ADR-010:160–173 explicitly falsifies the dedicated router-matmul hypothesis. Given its own inputs, hf2q agrees with an F64 reconstruction to `1.25e-7 rel_rms`; upstream inputs already differ by `1.66e-4`, enough to change a near-tie MoE route. The wider pre-MoE chain, not a less precise router projection, is the numerical investigation surface. Altering routing thresholds or smoothing changes model semantics.

ADR-011's integrated D256/D512 f16/bf16, resumed prefill and block-mask routes are visible in `src/serve/forward_prefill_batched.rs:4211,4243,4476,4625,4659`. The proposed porting work is stale. NAX attention remains a different, potentially current opportunity.

ADR-019 phases 1, 2 and 3a have implementation/measurement history. Its original CPU-overhead explanation was contradicted by traces. The current retained wait at `src/inference/models/qwen35/gpu_delta_net.rs:2254–2273` is an ownership fence; removing it without retaining scratch storage can corrupt state.

## 3. Exact state reuse: still current, but modernize the contract

Current Qwen/Gemma workers retain one prompt anchor per slot (`src/serve/api/engine.rs:17416–17528,18367` for Qwen; `:11756` for Gemma). DeepSeek has one pending and one committed turn anchor (`src/serve/api/engine_deepseek4.rs:1109`). Older conversation edits can therefore discard useful prefix work. Four-agent tool workloads amplify repeated prefills and contention.

Start with bounded slot-local multi-anchor state, not a general distributed radix cache. Required properties are committed/pending publication, invalidation of deeper descendants before divergent writes, complete owned-byte accounting, right-sized hidden-state ownership, and preflight-all-then-mutate restore. A→B→C, rewind A, write X, request old C must never restore stale C. Cancellation must discard pending captures and preserve only physically valid ancestors; it cannot resurrect a committed descendant whose KV backing was overwritten. Current Qwen `restore_slot_anchor` at `src/inference/models/qwen35/kv_cache.rs:1625` still interleaves mutation and later fallible validation.

ADR-059 introduces a critical new coordinate distinction: capture uses `graft_len + end` physically (`src/serve/api/engine_qwen35.rs:4335–4340`), while token-ledger and speculative lengths are logical. New anchors must bind graft digest/configuration and both coordinate domains. The old ADR-049 blanket equality of all cursors with prompt token count is unsafe to copy. Preserve the graft-region checks at `src/serve/api/engine.rs:19161–19195`. Sliding windows, recurrent state, vision expansions and DeepSeek compressor/indexer sidecars require separate family validity rules.

Current DeepSeek already owns disk persistence (`src/serve/api/engine_deepseek4.rs:554–560,1123–1127,1158,1325–1330`; ADR-062:207–250). Multi-anchor work must preserve it and avoid synchronously persisting every capture. ADR-049's old restart-tier description needs correction.

The [SGLang August 11 unified cache implementation](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache) provides current component-voting validity and compressed/indexer sidecars. Borrow those contracts, not datacenter hit-rate/throughput claims or a general shared-tree implementation before ownership economics are established.

Recent exact-cache research remains relevant:

- [FreeToken](https://arxiv.org/html/2608.16157v1), **August**, supports agentic semantic-boundary checkpointing. Its PCIe weight-transfer economics do not transfer to unified memory. Source commit `546c5fe2e4468e2f3f8a574c64021a47016c35f7` shows ownership/publication and bounded-history handling. hf2q must justify its own default depth and byte budget.
- [Sparse Prefix Caching](https://arxiv.org/html/2605.05219v1), inspected abstract submission 17 April (no reliance on an unverified later manuscript date), motivates optimizing placement from observed overlap-depth distributions. Use telemetry before replacing the simple bounded policy.
- [Marconi](https://arxiv.org/html/2411.19379v2) supplies older foundational hybrid-cache admission reasoning; it is not itself the October implementation proposal.
- [SuffixReplay](https://arxiv.org/html/2609.33477v1), **September**, reconstructs approximate recurrent state. It does not prove exact reuse and is excluded from production scope without separate quality authority.
- [Request Order Matters](https://arxiv.org/abs/2610.05833), **October**, reinforces the need to test permutations and nonprefix history dependence. No selective nonprefix reuse should be introduced merely to improve hit rate.
- [Vesta](https://arxiv.org/abs/2610.14686), **October**, makes queue-aware locality a useful hypothesis: waiting briefly for a warm active slot might beat immediate cold execution. Distributed results do not predict a four-slot optimum; measure queue delay plus total completion time.
- [GraphTok](https://arxiv.org/abs/2610.13325), **October**, identifies tokenization in cached latency. Its CUDA implementation is not a Metal result. First profile hf2q rendering/tokenization/lookup/queue cost; exact CPU/incremental reuse may be sufficient.

The local current peer provides corroborating implementation evidence: `server-context.cpp:2668` bounded checkpoints; `:2721–2725` captures target, draft and speculation together; `:2770–2805` persists/validates checkpoint trailers; `:4123–4174` restores then invalidates descendants. Its SWA-range TODO around `:2718` means it is a comparator, not unquestionable correctness authority. hf2q's saved-logit equality hit can avoid the peer's one-token replay.

## 4. Four-slot scheduling and physical batching

`src/serve/api/engine.rs:8097` limits prefill when a decode is runnable, and `:9945` sends bounded windows to serial prefill. Thus existing DeepSeek cooperative prefill bypasses the primary mixed-load case. Preserve ADR-049's coherence spike before changing policy: prove commit and poison affect only cohort members, including compressed-state/indexer sidecars and token ledgers. A GPU synchronization event does not establish cache isolation.

Only after that passes, test mixed prefill/decode populations 3+1, 2+2 and 1+3 at per-prefill-lane 128/256-row windows, with the existing aggregate cap, decode-visit gaps and semantic SSE gaps. Four-prefill waves are pure-phase controls; one-slot tests diagnose mechanisms. Queueing/fairness and actual completed work matter more than bulk prefill throughput. [Sarathi-Serve](https://arxiv.org/abs/2403.02310) is background for bounded mixed scheduling; its datacenter gains are not Apple acceptance evidence.

Distinguish four slots that can run independently from **one shared target forward over their rows**. ADR-044 explicitly leaves true physical multi-slot speculation batching open. A four-slot-first project must profile kernel batch dimensions and weight rereads, and qualify cross-slot target-row batching where profitable. BaseRT's October shared-forward implementation makes this a current competitive concern. Do not assume that a wider tree is profitable merely because a single request accepts more tokens.

## 5. October speculative decoding shortlist

The best current hypothesis is pretrained parallel block proposals plus efficient exact verification and adaptive row budgets. It is not a commitment to one fashionable drafter.

**Existing native MTP/history.** ADR-044:299–357 records exact single-slot gains and a historical matched-peer gap, with verification consuming roughly 58–61 ms of a 68–75 ms round. These numbers identify a historical bottleneck, not today's performance. Current warm append speculation already works when a valid target-hidden/speculative boundary exists (`src/serve/api/engine.rs:19240–19290`; `src/serve/api/engine_qwen35.rs:4004–4234,5122`). The PromptCacheHit exclusion covers full-prompt equality without reusable speculative metadata, not every cache hit. Bound grafts deliberately disable MTP because the separate draft arena lacks graft state. The remaining task is four-slot qualification and measured gap closure, not a greenfield cached-speculation implementation.

**DFlash 2.** The [18 August official release](https://inco.ai/blog/dflash2/), [code](https://github.com/z-lab/dflash) and [Qwen3.8-27B checkpoint](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) offer a pretrained candidate with Apple/peer support. The local peer's `src/models/dflash.cpp:31–34,152–176,325+` implements selector/convolution metadata. Old hf2q DFlash experiments in ADR-034 were slower and sometimes corrupted long continuations; those failures constrain a new port but do not falsify the new model.

**DSpark.** [Red Hat's 21 September Qwen3.8 checkpoint](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) is a newer candidate than the weights in August comparisons. [mlx-dspark](https://github.com/ARahim3/mlx-dspark/blob/d6042f38f0aa7dd3e03a2a406111d725056d45a3/CHANGELOG.md) gives Apple implementation and paired shape-sensitive evidence. Its roughly 2B-parameter drafter has real memory/latency cost. Target compatibility with abliterated or grafted hf2q models must be measured, not inferred from the official target name.

**KFlash lattice trees.** [October Kapitsa design](https://github.com/nace-ai/kapitsa/blob/c490f8159622bb2d356b21c8d670d668997cbfb4/docs/kflash.md) reuses a parallel drafter's candidate lattice instead of paying repeated autoregressive draft forwards. It carries branch-specific recurrent state and compacts only accepted paths. Its concurrency gain was measured at three requests, not four. Its own reported long-agent regression and near-tie caveats prevent selecting it on code/prose headlines. Compare chain/tree at fixed total target rows and four live slots, with measured per-emitted-token cost and parent-state exactness.

**Why HASS/Hydra is not the primary plan.** [HASS](https://arxiv.org/html/2408.15766v3) is Zhang et al., submitted 2024, evaluating EAGLE-2; it cannot demonstrate gains stacked on later [EAGLE-3](https://arxiv.org/html/2503.01840v3). [Hydra](https://arxiv.org/html/2402.05109v2) uses lightweight conditioned heads, unlike ADR-039's repeated full transformer-head interpretation. ADR-037's parallel-drafter description and ADR-039's promised combined speedup/checkpoint compatibility lack evidence. Reuse proven infrastructure; consider custom training only after current pretrained candidates fail for a diagnosed, trainable reason.

[D²SD](https://arxiv.org/abs/2606.04446) and [PRESTO](https://arxiv.org/abs/2607.22634) inform diffusion-tree research, but did not establish a target-compatible four-agent Apple implementation better than the running candidates above. They are watchlist sources, not immediate implementation obligations.

## 6. October Metal opportunities

The current native crate already has TensorOps dense/expert matrix kernels. Generic adoption of Metal 4 matmul is dated work. Remaining candidates are dispatch at the actual small row counts, attention QK/PV acceleration, scratch lifetime, and selected epilogues.

- [MLX thin-M GEMM #4654](https://github.com/ml-explore/mlx/pull/4654), merged **9 October**, targets small matrix rows. [F32 split-K partials #4641](https://github.com/ml-explore/mlx/pull/4641), merged **8 October**, demonstrates that accumulation precision belongs beside throughput in the dispatch contract.
- [MLX small-row quantized kernels #4712](https://github.com/ml-explore/mlx/pull/4712), merged **22 October**, specifically targets 6–16 rows with large N/K in NVFP4/MXFP4/MXFP8 formats. Its gains do not establish an ordinary four-row or GGUF K-quant win; shared verification can reach those larger row counts. This is immediately relevant to grouped target verification; port eligibility/layout ideas only after profiling GGUF K-quants and owned kernel numerics.
- [MLX issue #4695, opened 14 October](https://github.com/ml-explore/mlx/issues/4695), reports regressions on some few-head/small-query shapes; it is an author-reported open proposal, not a merged universal optimization. Sweep widths 1/2/4/8/16, heterogeneous four-slot cohorts, short cached suffixes and long prompts; avoid one global route.
- Current peer `ggml-metal/kernels/fa_tensor.metal:120–125` uses TensorOps QK/PV. Native tiled attention still uses simdgroup MMA. An owned NAX attention spike is justified, with D256 first and D512 only after benefit, preserving block skipping, resumed offsets, sliding masks and final-layer bf16 overflow protection.
- ADR-019's chunk scratch lifetime remains an explicit fence. First retain every temporary through completion, then remove one measured blocking edge. Validate logit-tuple determinism, long raw-worker runs and heap slope; order-only events do not extend allocation lifetime.
- Fused expert gate/up/activation is conditional on profiling actual routed-token distributions. Preserve family activation semantics. Old large-K split-K tuning is lower priority than the October small-row regimes.

[Apple's M5 optimization discussion](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) and [GPU profiling guidance](https://developer.apple.com/videos/play/tech-talks/111432/) justify hardware-specific evaluation, not a universal speed multiplier. BaseRT's newer release is a better current comparator than repeating its July paper headline.

## 7. Quality and speed acceptance must be separate

Cross-engine coherence is task correctness, not universal byte identity between numerically different engines. Internal target optimizations nevertheless need strict cold/resumed, sequential/verified and slot-isolation invariants wherever semantics promise them. [Batch-invariant inference analysis](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) is useful diagnostic background; its CUDA costs do not transfer automatically.

The primary scoreboard extends existing test_full_context_agent_slots.sh, bench_qwen36_agentic_cache.sh, qwen38_speculation_ab.sh and test_agentic_cache_lifecycle.sh rather than rewriting their established gates. It needs four independent agent conversations, at least six tool rounds each, and deterministic checked tasks: correct tool name/arguments, executable patch or code result, tool-result continuation, completion rather than silent reasoning, valid unary and SSE output, no cross-conversation state. Include simultaneous cold starts, staggered arrivals, warm append-only turns, divergent edits, retries/cancellation and prompt equality. Test permutation/order changes. Grafts, constrained generation and sampling must be represented explicitly, not assumed safe from ordinary greedy tests.

Quality gates should freeze task fixtures and score both peers and hf2q against independent expected results. For stochastic tasks predeclare sample size and a paired noninferiority margin; no threshold may be chosen after seeing results. Faster wrong answers, truncated answers, altered tool provenance or reduced output budgets fail.

Performance receipts must contain source/dependency/binary/model/drafter identities, exact inputs/settings, new and cached token counts, prefill/decode rates, first semantic output, per-slot p50/p95/max TTFT, worst semantic SSE gap, full makespan, successful-task goodput, draft/verify/recovery costs, peak memory and thermal state. Use balanced three-arm crossover blocks with warmup, a minimum of five blocks and a predeclared 95% effect-interval half-width target of 2.5 percentage points, maximum twenty blocks; unresolved effects are inconclusive. Independent QE must be a named session distinct from the implementer, using a certified frozen harness. Latest llama.cpp cannot be silently removed after a quality failure and leave a victory against no peer. A proposed winner must improve four-agent end-to-end time, not only a microkernel or single-slot decode rate. No 90GiB DeepSeek peer and hf2q co-residency on the 128GiB host.

## 8. Exclusions and research limits

ANE ViT/BERT offload was investigated, but is peripheral to four text agents and ADR-016's Python/CoreML conversion prescription conflicts with today's product boundary. Approximate recurrent-state reconstruction, routing smoothing, prequantized artifact substitution, distributed Macs, untrained-Hydra acceptance rules, custom-training-first plans and universal headline speed claims are outside the recommended delivery.

This investigation used source inspection, primary papers, current code/release metadata and screened web results. Academic prompt examples triggered some scanner heuristics; none were treated as instructions. Live Ruflo unified recall failed with `policy-ledger-anchor-missing`; secondary Ruflo searches returned no relevant entries and release/status retrieval was absent. No integrity anchor was changed. No research record was stored pending acceptance. Hardware, model/checkpoint compatibility, current four-slot baseline and statistical quality are unresolved execution questions, explicitly represented in the proposal.

The proposal includes dedicated independent QE tasks to certify the harness with adversarial negative controls and prove stock-to-experiment and experiment-to-latest-peer gains. A final release verdict requires those receipts; a benchmark score or a successful process alone is insufficient.

## 9. Skeptic findings and final prioritization

The requested `claude -p` review returned REVISE after checking code and primary source dates. The full review and dispositions are in [the critic record](apple-silicon-speed-parity-2026-10-22-critic.md). It caught stale/missing executable identity, peer-configuration ambiguity, multi-round transcript comparability, comparator exclusion, overly broad delivery scope and nominal QE independence. Those are now explicit ADR/task obligations. The operator subsequently required vanilla Qwen3.8 and DeepSeek-V4 as primary models; they remain in the staged matrix despite the critic's narrower Qwen3.6-only suggestion.

Historical four-slot evidence matters: ADR-040:410–434 measured Gemma batched-body aggregate around 150 versus peer 200 tok/s, later about 0.81x after attention work. KV formats differed, so this is not current matched-KV proof. Its investigation localized non-amortized attention and documented failed barrier/dense-row hypotheses. Existing batched attention already landed. S3 must reprofile those mechanisms on current source rather than repeat a closed experiment. Qwen's historical sublinear four-slot scaling is likewise a lead, not a current receipt.

The v0.1.28 queue is narrowed to current three-arm/quality foundations, coordinate-safe exact Qwen reuse, existing warm-speculation qualification, actual four-slot shared-forward and GGUF K-quant verifier dispatch, deciding DeepSeek/pretrained-drafter experiments, and independent QE. Gemma/DeepSeek multi-anchor implementation, mixed-policy rollout, queue-aware admission, lattice trees/new native drafters, NAX attention and scratch-wait removal remain documented successors, not promised release features. They reopen only from measured bottlenecks and available memory.

Canonical replay controls identical-input speed/fairness; own-output continuation separately measures natural multi-turn task success and cache realism. Priming/reset cost and differing token histories must be explicit. Greedy proves represented exactness; actual production-client sampling supplies the realistic quality lane. Fixed release qualification is per model, never pooled into universal superiority.

The memory gate uses one nonoverlapping ownership ledger for shared CPU/Metal memory and a ceiling no higher than physical memory minus at least 12 GiB of system headroom, further reduced by other resident use and existing family caps. Stock, experiment and peer run sequentially. Drafter, four-slot KV/recurrent state, pending anchors and peak scratch must fit before loading. Bounded QE stress and admission negative controls must reject pressure/breaches before OOM; lowering matched context or output budgets is not a winning configuration.

Evidence grades: current source/release identities and existing code paths are High confidence; external author benchmark results are Medium confidence for their own configurations and Low for transferred hf2q performance; projected four-slot speed gains are unproven hypotheses. No hardware performance or new quality verdict was produced in this research session.
