# Claude skeptic pass: Apple Silicon speed/coherence program

- Date: 2026-10-22
- Command: `claude -p` with read-only Read/Grep/Glob/WebSearch/WebFetch tools.
- Verdict: **REVISE**. This is a planning critique, not hardware acceptance.
- Reviewed source: pre-ADR research report and proposal; superseded by revised ADR-063.

## Independent review

Editorial note (2026-10-23): peer names in prose are normalized to NOTICE terminology; exact URLs and source paths are preserved. The review findings and historical verdicts are unchanged.

## Verdict: REVISE (document-level fixes, no new research needed)

The research report is well sourced. Every external date and identity I could check matched: BaseRT v0.4.0 (22 Oct, commit 24badc8), v0.3.0 (8 Oct, shared-forward batching plus prefix reuse), MLX PRs #4641/#4654/#4712 (8, 9, 22 Oct), MLX v0.32.4 (14 Oct), the Apple RDMA post (15 Oct), the SGLang unified radix cache post (11 Aug), DFlash 2 (18 Aug), Red Hat DSpark (21 Sep), and all eight arXiv IDs with the stated months. The hf2q and peer line citations resolve, with one exactness defect noted below. The plan's weaknesses are structural: the three mandatory arms are not defined tightly enough to make the Q3/Q4 verdict fair, the workload design does not say how multi-round agent transcripts stay comparable across engines, and the scope is not prioritized enough to finish.

### BLOCKER

- **B1. The local "latest peer" is not built.** R:18 and P:37 treat `/opt/llama.cpp` at 970b51ce (tag b11827, fetched 22 Oct) as the latest peer. The checked-out source is current, but `build/common/build-info.cpp` records the compiled binary as b9278 at 5306f4b3b, roughly five months old. Every script defaulting to `/opt/llama.cpp/build/bin/llama-server` would run a May-era peer under an October label. F1 must record the binary's own `--version` output and hash, not the source checkout, and must rebuild per campaign.
- **B2. The peer arm's configuration is undefined, which is a fairness loophole in both directions.** P:16 and P:70 say "matched settings," but hf2q's launcher enables MTP speculation, repetition penalty 1.05, thinking budgets and TQ-active KV by default (`scripts/serve_qwen36_opencode.sh:64,192-199`), while the latest peer defaults `--spec-type none` and supports `draft-mtp`, `draft-dflash`, `draft-eagle3` and `draft-dspark` (`/opt/llama.cpp/common/arg.cpp:357-416`, server README line 276). A "matched" run with speculation off in the peer handicaps the peer. A run with hf2q MTP on against a GGUF that dropped MTP tensors makes the peer unable to match. The plan must declare two peer configurations: matched-settings and peer-best-documented, and the "faster than peer" claim in Q4 counts only against peer-best. It must also require the same-artifact GGUF to retain MTP tensors if either arm uses them.
- **B3. Multi-round transcript policy is missing, so makespan and cache behavior are not comparable across engines.** P:64 requires six tool rounds per conversation but never says whether round N+1 continues from each engine's own generated assistant turn or from a canonical one. Own-output continuation makes token counts and prefix-reuse differ per engine, so makespan is not paired. Canonical replay keeps inputs matched but defeats own-output prefix reuse, which is the C-lane payoff. The harness needs an explicit policy per metric. A workable split: canonical replay for speed and fairness metrics, own-output continuation as a separate quality and cache-realism lane.
- **B4. Q4's comparator rule can award a win by default.** P:581 compares against "the best quality-qualified matched peer." If the mandatory peer arm fails a quality oracle on some tasks, it drops out and hf2q can win against nothing. Require that a verdict is reported only when the peer arm completed the same task set, and that peer quality failures are themselves recorded as a finding, not as disqualification.
- **B5. Scope cannot finish as written.** F2 and Q3 multiply arms (3), populations (1-slot, 4-slot, 3+1, 2+2, 1+3, 4+0, 0+4), families (Qwen3.6, Qwen3.8, Gemma, DeepSeek), five paired trials, and warmup. DeepSeek alone is a 100 GiB model that must run one runtime at a time with cooldowns (`scripts/benchmark_deepseek4_server_parity.sh:35-37`). No family or model is named primary. Name Qwen3.6-35B-A3B on the production launcher as the primary artifact for every mandatory verdict, and make Qwen3.8, Gemma and DeepSeek secondary lanes with reduced population sets.

### HIGH

- **H1. The report omits the only existing four-slot peer evidence.** ADR-040 lines 410-432 measured hf2q at 0.75× then 0.81× of `llama-server -np 4` aggregate decode on the same Gemma GGUF, localized to non-amortized per-slot attention, and ADR-040-history §6.1.57 recorded Qwen3.6 SlotAware per-slot decode dropping to about half at N=4 with aggregate scaling near N^0.55. R:46 dismisses ADR-040 as "historical measured gains" without quoting this. That evidence is the baseline for S3 and should drive its hypothesis: four-slot decode attention and MoE row amortization, not a generic "profile everything" story.
- **H2. BaseRT is a same-artifact candidate, not only a native-format arm.** R:25 and P:43 frame GGUF versus MLX affine as different artifacts. BaseRT v0.4.0's own notes list Q4_K, Q5_K, Q6_K plus new K/IQ quant reading on Mac. BaseRT can load the hf2q-produced GGUF and should be screened as a same-GGUF secondary arm, which strengthens the peer-competition claim. Kapitsa also imports Qwen GGUF, though it is Qwen and Nemotron only and its own benchmarks run `-np 1`.
- **H3. The mixed-phase design is DeepSeek-specific but the acceptance is written as universal.** The 128/256-row lane windows and the cooperative-prefill bypass at `src/serve/api/engine.rs:8074-8097` exist only on the DeepSeek path. S2 is honest about this, but F2's requirement at P:72 to "measure 3+1, 2+2 and 1+3" for all families needs a mechanism definition for Qwen and Gemma, where prefill chunking and decode interleave differently. Also, a 3+1 population is transient. The harness must specify how long the composition is held, for example by staggered arrival offsets with multi-thousand-token tool results, and must report time-weighted phase composition, not just the intended label.
- **H4. Research report citations omit the directory.** R:58, R:60, R:62, R:82 and R:92 cite `engine.rs`, `engine_deepseek4.rs`, `engine_qwen35.rs`, `kv_cache.rs` and `qwen35_speculation.rs` without `src/serve/api/` or `src/inference/models/qwen35/`. The lines are correct at the real paths. Fix before this becomes an ADR anchor; ADR-049 shows how stale anchors rot.
- **H5. The repository peer pin is unreachable upstream, not only locally.** R:23 says the object is absent locally. GitHub also returns 404 for e15384a5, and ADR-052:45 and `scripts/fixtures/grammars/peer/PROVENANCE.md:12` already record this. F1's "explain and resolve" at P:37 should state the conclusion: the pin cannot be rebuilt by anyone and must be replaced with a reachable tag (docs/adr/033 recorded build b9223) while historical receipts keep the old hash as provenance only.
- **H6. Statistical design is too thin for a 5% claim.** P:70 and P:554 fix five alternating paired trials, and P:581 demands a 5% makespan or goodput improvement with uncertainty excluding a tie. Agentic four-slot makespans on a thermally variable M5 Max will have run-to-run spread that five pairs rarely resolve at 5%. Predeclare a precision rule (confidence-interval half-width target) and allow extension, as P:554 hints, but make it the primary rule rather than a fallback. Cluster by conversation and run, as P:558 already says.
- **H7. Hidden critical path through mlx-native publication.** K1, K2, K3 and most of S3 are kernel work in a separate crate pinned at `mlx-native = "=0.15.1"` (`Cargo.toml:174`). P:24 requires a published revision before acceptance. The plan should say that every kernel candidate costs a crate release and a Cargo.lock bump, and that a worktree path-patch is never a receipt. This affects sequencing more than any F-story.
- **H8. QE independence is nominal unless the executor is named.** Q2 and Q3 say "independent QE" but the same agent could write the harness, run it, and certify it. State that Q3 is executed by a different session or host from the one that implemented the candidate, with the harness commit frozen before the candidate exists.

### MEDIUM

- **M1. Uncited claim.** R:110 asserts MLX October attention tuning regressed some few-head/small-query shapes. I could not find a PR, issue or release note for this. Cite it or mark it as an unverified hypothesis.
- **M2. Unverified detail.** R:71 claims a Sparse Prefix Caching rendered manuscript dated 24 August; arXiv lists only v1 from 17 April. R:96's "roughly 2B-parameter drafter" matches the Hugging Face metadata but the card does not state it. Both are minor; downgrade wording.
- **M3. Existing harnesses are not referenced.** `scripts/test_full_context_agent_slots.sh` already runs four concurrent agents through the shared OpenCode contract with cold, cached and tool-result stages and TTFT ceilings; `scripts/bench_qwen36_agentic_cache.sh` already pairs hf2q against llama-server on three-turn exactness; `scripts/qwen38_speculation_ab.sh` already proves OFF/AUTO byte equality. F2 reads as greenfield. Say what is extended and what is new, otherwise Q2's "no implementation-mirroring tests" rule will be violated by a rewrite that copies these.
- **M4. The 128/256-row figure and the aggregate cap need a source line.** P:289 preserves "the aggregate cap" without naming `MAX_COOPERATIVE_PREFILL_ROWS` (`src/serve/api/engine.rs:9859`) or the ADR-042 OOM evidence at 4,096 rows that ADR-049:31 cites.
- **M5. Thermal and host-contention protocol is asserted, not specified.** P:20 and R:125 mention thermal state and the release-coordination key. Require a fixed cooldown and a thermal-pressure reading between arms, and a check of `hf2q/release/status` before each campaign, with the campaign refusing to start under contention.
- **M6. Sampling realism.** OpenCode and similar clients do not run at temperature 0. P:508 freezes "sampled quality margins," but no story says which temperature the primary verdict uses. Pick greedy for the exactness lanes and the client's actual default for the Q3 quality lane, and say so.

### Disproven versus uncertain

- **Disproven in the documents (keep as corrections):** ADR-010's router-only precision fix (ADR-010:160-170 F64 reconciliation). ADR-011's "Phase 2 next" text (integrated D256 FA with block masks at `src/serve/forward_prefill_batched.rs:4211,4243`). ADR-039's HASS attribution ("Hu et al."; arXiv 2408.15766 is Zhang et al. against EAGLE-2). ADR-034's DFlash v1 port being slower and corrupting long continuations (ADR-034:135-138).
- **Uncertain hypotheses (keep as experiments):** DFlash2 versus DSpark versus native MTP at four slots (mlx-dspark shows them within noise at 8-bit and shape-sensitive at 4-bit). KFlash lattice gains (measured at three requests, with its own agent-turn regressions). MLX #4712 small-row kernels (rows 4 and 5 unchanged by its own data, so no batch-four gain is implied). Queue-aware locality (Vesta is a distributed result). The chunk-scan fence removal (K3).

### Oracles: adequate for internal exactness, not for the program's claims

Same-artifact oracles exist and are sound at one tool round: byte-identical slot_aware n1/n4 gates, cold-versus-cached three-turn exactness, OFF/AUTO speculation equality, DeepSeek sequential parity with prompt-cache credit rejection. Independent references exist only at kernel level (F64 and CPU references). Missing for this program: lineage oracles for rewind and divergence (the A→B→C case at R:60), mixed-phase composition tracing, multi-round task scoring by executed code rather than sentinel strings, and any cross-engine quality scoring that is not byte comparison. Q1 and Q2 describe these correctly; they do not yet exist, and nothing in the plan may be accepted before they do.

### Keep, drop, defer

- **Keep:** F1, F3, C1, C2, S1, S3 reframed around ADR-040 §M4 evidence, D1, D2, Q1, Q2, Q3, Q4.
- **Drop from the delivery:** D4 lattice trees and C6 queue-aware admission until S3 and C2 produce four-slot traces. K1 should be narrowed to GGUF K-quant verifier widths 4 to 8, since the October MLX kernels do not cover them.
- **Defer:** C3 Gemma, C4 DeepSeek, K2 NAX attention and K3 behind the primary Qwen3.6 verdict. C5 tokenization only if the F2 profile shows it above a stated fraction of warm TTFT.

### Mandatory changes before the ADR and awa project

1. Record and rebuild the peer binary identity; replace the unreachable pin with a reachable tag (B1, H5).
2. Define matched and peer-best configurations, including speculation and MTP tensor retention in the shared GGUF (B2).
3. Define the transcript continuation policy per metric (B3).
4. Fix the Q4 comparator rule so the peer arm cannot drop out (B4).
5. Name Qwen3.6-35B-A3B on the production launcher as the primary artifact and cut the population matrix for secondary families (B5).
6. Quote the ADR-040 four-slot baseline and reframe S3 around it (H1).
7. Add BaseRT as a same-GGUF screened arm (H2).
8. Specify how mixed phases are held and reported for non-DeepSeek families (H3).
9. Correct the source paths in the research report (H4).
10. Replace five fixed trials with a predeclared precision rule, and name the independent Q3 executor (H6, H8).

Sources consulted: [BaseRT releases](https://github.com/basecompute/baseRT/releases), [BaseRT v0.4.0](https://github.com/basecompute/baseRT/releases/tag/v0.4.0), [MLX #4712](https://github.com/ml-explore/mlx/pull/4712), [MLX #4654](https://github.com/ml-explore/mlx/pull/4654), [MLX #4641](https://github.com/ml-explore/mlx/pull/4641), [MLX releases](https://github.com/ml-explore/mlx/releases), [peer releases](https://github.com/ggml-org/llama.cpp/releases), [Kapitsa kflash.md](https://github.com/nace-ai/kapitsa/blob/c490f8159622bb2d356b21c8d670d668997cbfb4/docs/kflash.md), [Kapitsa benchmarks.md](https://github.com/nace-ai/kapitsa/blob/c490f8159622bb2d356b21c8d670d668997cbfb4/docs/benchmarks.md), [Kapitsa README](https://github.com/nace-ai/kapitsa), [Apple MLX RDMA post](https://machinelearning.apple.com/research/mlx-distributed-inference), [SGLang unified radix cache](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache), [DFlash 2 blog](https://inco.ai/blog/dflash2/), [Red Hat DSpark card](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark), [mlx-dspark changelog](https://github.com/ARahim3/mlx-dspark/blob/d6042f38f0aa7dd3e03a2a406111d725056d45a3/CHANGELOG.md), [HASS](https://arxiv.org/abs/2408.15766), [Vesta](https://arxiv.org/abs/2610.14686), [Request Order Matters](https://arxiv.org/abs/2610.05833), [GraphTok](https://arxiv.org/abs/2610.13325), [SuffixReplay](https://arxiv.org/abs/2609.33477), [FreeToken](https://arxiv.org/abs/2608.16157), [Sparse Prefix Caching](https://arxiv.org/abs/2605.05219), [D²SD](https://arxiv.org/abs/2606.04446), [PRESTO](https://arxiv.org/abs/2607.22634).

## Author disposition

| Finding | Disposition in ADR-063 |
|---|---|
| B1/H5 source/binary/pin identity | F1 requires rebuilt executable identity, reachable replacement proposal and historical provenance. Author verification found the default executable absent, with stale build metadata; the review's implication of a runnable old binary is not independently confirmed. |
| B2 configuration fairness | F1/Q3 use peer-best for the mandatory peer verdict, plus matched-settings diagnostic and retained MTP tensors where used. |
| B3 transcript policy | F2 separates canonical replay/explicit priming from natural own-output task/cache quality. |
| B4 peer exclusion | Q3/Q4 forbid a default win when peer comparison is missing or failed. |
| B5 scope | Release narrowed; secondary techniques become successor hypotheses. Operator explicitly retained vanilla Qwen3.8 dense and DeepSeek-V4 as primary artifacts, so their proofs are staged rather than removed. |
| H1 four-slot history | Research section 9 and S3 name historical attention/amortization evidence and already-landed fixes. |
| H2 same-artifact competitors | BaseRT GGUF screening is explicit. |
| H3 phase composition | F2 requires family-specific time-weighted observed populations and controlled staggered tool waves. |
| H4 source paths | Report anchors now include repository-relative directories. |
| H6 precision | F2/Q3 specify interval precision, bounded sequential sampling and inconclusive verdicts. |
| H7 native publication | K1/global contract explicitly include native crate publication, registry pin and clean-source proof. |
| H8 QE independence | Q3 requires a named separate executor and frozen certified harness. |
| M1/M2 citation details | MLX issue4695 cited as an open author report; uncertain Sparse Prefix manuscript date removed. |
| M3/M4 existing harnesses and row cap | Existing harness extensions and DeepSeek2048-row/OOM fence named. |
| M5/M6 host and sampling | Contention preflight, fixed cooldown/thermal rules and real client sampling required. |

Additional parent source verification corrected a flaw not caught in the critique: warm cached speculation already works with valid metadata. D1 now qualifies actual remaining boundaries rather than assuming the whole feature is absent. Q5 adds the operator's explicit M5 Max/128 GB, no-OOM acceptance contract. No critique recommendation is treated as a benchmark result or approval to change runtime behavior.

## Focused follow-up review

**Verdict: APPROVE FOR PROPOSED PLANNING.** No BLOCKER remains. This approves ADR-063 as a Proposed ADR and awa extraction only. It is not runtime design approval or performance acceptance.

**Prior mandatory flaws, checked against the revised ADR**

- **B1/H5 peer binary identity.** F1 now states the default peer executable is absent, build-info metadata is stale at b9278/5306f4b3b, source HEAD is not executable identity, and a per-campaign isolated build must record its own version output and hash. The unreachable e15384a5 pin is routed through the governing pin workflow. Addressed.
- **B2 peer configuration.** F1, Q3 and Q4 declare peer-best as the mandatory arm, matched-settings as a labeled diagnostic, forbid silently disabling speculation, and require MTP tensor retention in the shared GGUF. Addressed.
- **B3 transcript policy.** F2 separates canonical replay for token-matched timing from an own-output continuation lane for task success and natural reuse, and forbids claiming natural reuse from forced transcripts. Addressed.
- **B4 comparator drop-out.** Q3 and Q4 require the peer to complete the declared workload and record peer failures as findings. Addressed.
- **B5 scope.** The operator overrode the critique's "Qwen3.6 only" cut. The ADR keeps vanilla Qwen3.6 MoE, vanilla Qwen3.8-27B dense and DeepSeek-V4 as primaries with staged N1 plus N4 proof per artifact and separate Q4 verdicts. Gemma is secondary. Consistent with operator direction. Scope cost is acknowledged under Consequences.
- **H3 phases.** F2 requires family-specific phase labels, staggered tool-result waves to hold 3+1, 2+2 and 1+3, and time-weighted observed composition. Addressed.
- **H6 statistics.** F2 and Q3 predeclare a 2.5 point half-width target, five to twenty blocks, inconclusive otherwise. Addressed.
- **H7 native publication.** K1 and the global contract require a published registry revision pinned in the lockfile. Addressed.
- **H8 QE independence.** Q3 names a separate executor and freezes the harness commit before candidate implementation. Q5 adds the independent no-OOM contract on M5 Max 128 GB. Addressed.
- **M3 existing harnesses.** All four scripts named in F2 exist under scripts/. Addressed.

**Author's additional correction verified against source.** D1's claim that warm cached speculation already works holds. The code at `src/serve/api/engine.rs:19240` keeps retained speculative metadata when the graft is absent and cursors validate, classifies PromptCacheHit only when cached tokens equal the full prompt with no speculative candidate, and downgrades to ordinary decode under a bound graft. The `src/serve/api/engine_qwen35.rs:4004` boundary checks also match. D1 correctly scopes to remaining boundaries.

**Dependency graph.** F1 → Q5 → F2 → {Q1, Q2, C5, S1}; F3 → C1 → C2 (also F2); D1 needs C1, F2, Q1, Q5; D2, S3, K1 need F2, Q1, Q5; Q3 needs F2, Q1, Q2, Q5; Q4 needs F3, Q1, Q2, Q3, Q5. No cycles. Excluded successors are enumerated in one watchlist and are consistent with the per-story successor language in C2, S1, D1, D2 and S3.

**Remaining HIGH findings**

- **H-A. S1 source anchor is wrong.** S1 says the row cap is in `src/serve/api/engine.rs`. The constant is defined in `src/inference/models/deepseek4/verifier_forward.rs:26` and only consumed at engine.rs:9859. The value 2048 is correct. Fix the anchor before extraction so the ADR does not inherit the rot the critique's H4 warned about.
- **H-B. Q4 gating on candidate stories is prose only.** Q4's predecessor list omits C2, S3, D1, D2 and K1. The "winning subset declared before candidate acceptance" rule covers this, but an extraction that reads only predecessor fields will treat Q4 as independent of every candidate. The extraction must carry the winning-subset rule as an explicit obligation on Q4, or Q4 must name the candidate stories as conditional predecessors.

Neither finding changes the verdict. Both are document and extraction fixes, not research.

Final disposition: H-A source anchor corrected to verifier_forward.rs:26; H-B is an explicit Q4 acceptance obligation requiring the selected work keys/commits to become graph predecessors before candidate acceptance. Planning approval remains limited to Proposed ADR/extraction. Automatic environment notices were omitted from this technical record.
