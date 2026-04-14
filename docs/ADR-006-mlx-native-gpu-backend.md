# ADR-006: mlx-native as hf2q's GPU Compute Backend (migrate from candle)

**Status:** Accepted (Phase 0 complete 2026-04-12; verdict: framework-overhead-dominated; see `docs/spike-1bNEW30-per-kernel-attribution.md`). Phase 4 scope revised 2026-04-13: kernel-level changes (4a F16 KV, 4b Q8_0 NSG) implemented; gap re-attributed to barrier stall cost. Phase 4e revised 2026-04-13: proper computation graph architecture (graph IR + fusion + reorder + dual command buffer encoding) — porting llama.cpp's full Metal execution pipeline.
**Date:** 2026-04-11 (Proposed) → 2026-04-12 (Accepted) → 2026-04-13 (Phase 4 revised: computation graph architecture)
**Decision Makers:** Robert, Claude
**Supersedes (implicitly):** the original "use candle" decision from hf2q's early Crawl phase, never written as an ADR
**Related ADRs:** ADR-005 (Inference Server, Phase 1b speed gap context), ADR-004 (GGUF compatibility — mlx-native must preserve all of it)

---

## Engineering Mantra (load-bearing — read before every session)

Source: `~/Documents/mantra.txt` (Robert, undated). Quoted verbatim. This is the discipline this ADR — and every spike, every commit, every decision under it — must be executed against. It supersedes any tactical convenience that conflicts with it.

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

**Operational reading for this ADR specifically** (how the mantra constrains every phase of the candle → mlx-native migration plan):

- **DO NOT BE LAZY / no short cuts** — Phase 0 (per-kernel timing diagnosis) is mandatory before Phase 4 (Build Phase) starts. Skipping Phase 0 to "just port the framework patterns and hope" would be exactly the lazy pattern the 1bNEW.22 sticky encoder ($3-hour refuted patch) cost demonstrated. Phase 0 is cheap insurance against multi-week refuted patches.
- **Plenty of time** — the 4-7 week migration estimate is the right size, not "too long". Compressing it via shortcuts would violate this directly. If Phase 0 reveals the work is bigger than 4-7 weeks, the timeline expands; the bar does not move.
- **Never make assumptions** — every borrowed kernel in Phase 3 (Borrow Phase) gets a bitwise correctness validation against the candle source. Every framework pattern in Phase 4 (Build Phase) gets a microbench validation before integration. Every per-op cutover in Phase 5 (Integration) gets a sourdough gate run. No "looks right, ship it".
- **Dive deep / use search as needed** — Phase 1's Chesterton's fence work on candle, mlx-native, coreml-native, and ggml-metal is non-negotiable scaffolding for the migration. Phase 2 reads coreml-native's full PRD as the maturation template, not a summary.
- **Measure 3x, cut once** — Phase 0 measures both stacks (3+ runs each). Phase 3 validates each borrowed op against multiple references where possible (candle source AND mlx-native's existing implementation, picking the more correct one). Phase 4 validates each framework pattern against an isolated microbench BEFORE integrating. Phase 6 runs a 5-run canonical bench median, not a single-shot.
- **No fallback / no stub** — Phase 5's per-op cutover deletes the candle paths after each op is validated on mlx-native. There is no "leave it as a fallback in case mlx-native breaks". The migration ends with a clean stack, not a dual-backend morass. The Option D (hybrid: candle + mlx-native behind a feature flag forever) was rejected for exactly this reason.
- **Pure excellence, done right** — every Phase 3 borrow gets license attribution + source `file:line` citation in a header comment. Every Phase 4 port gets ggml-metal `file:line` citation. Every commit follows the project's commit + push cadence (`feedback_commit_push_cadence.md`). The "v0.2.0 prepared for crates.io" target for mlx-native (Phase 2) is the bar, not a stretch goal — it's what coreml-native already executed once and the template we're following.
- **Chesterton's fence** — Phase 1 understands candle's hot path before deprecating it. Phase 2 understands mlx-native's existing kernels before borrowing on top of them. Phase 3 validates mlx-native's existing implementations against candle's borrows before deciding which to keep. Phase 4 understands ggml-metal's framework patterns before porting them. Every step understands what currently works before changing it.

**Falsified hypothesis register** (cumulative across the 2026-04-11 session that motivated this ADR — each is evidence the mantra discipline works, and each is a reason this ADR proposes measure-first Phase 0 rather than jumping straight to Phase 4 patch work):

1. CPU dispatch overhead is the bottleneck (1bNEW.22 instrumentation spike)
2. Pool-tuning has multi-tok/s headroom (1bNEW.21 sweep landed +0.7 tok/s, not multi-tok/s)
3. Single-command-buffer-per-forward is faster (empirical: 5000 dispatches/buffer regressed 85.0 → 79.4)
4. Encoder creation is the bottleneck (1bNEW.22 sticky encoder: 168k saved encoder creations × ~50 ns each = 0.10% improvement, well below noise; ~3 hours of refuted patch work)
5. Per-buffer wait semantics is the dominant lever (dissolved by 1bNEW.20 via a different mechanism)
6. Per-shape NSG selection has wall-clock headroom on hf2q's M5 Max shapes (1bNEW.22-v2 NSG sweep + Agent #3 static prediction converged on NULL across 8 shapes × 4 NSG values × 4 runs)
7. hf2q dispatch count (2104) is high vs llama.cpp (Agent #2: ggml graph has 2652 nodes per forward — llama.cpp does MORE work and is still faster)
8. llama.cpp peer measures 107 tok/s today on this hardware (Agent #2 5-run re-measurement: 102.01 median; End gate re-baselined per `feedback_ground_truth_is_what_we_can_measure_now.md`)
9. Q6_K NR0=2 row-loop port has wall-clock envelope on M5 Max (Agent C1: bitwise-correct port, `max|Δ|=0.000000e0`, two independent 4-run sweeps converged on ±0.4% noise — the 2× threadgroup reduction trades off ~1:1 against doubled per-simdgroup work)
10. The 17 tok/s gap lives in candle's individual kernel implementations (Phase 0 FALSIFIED: per-call GPU times show candle's kernels are comparable or faster than ggml's in 7/9 matched families — the gap was framework dispatch patterns)
11. Barrier count is the bottleneck on mlx-native (2026-04-13 barrier profiling FALSIFIED: hf2q has 606 barriers/token, llama.cpp has 759 — more barriers yet faster. All 606 barriers are true RAW dependencies; none removable. Buffer aliasing ruled out via exhaustive audit — all activation buffers are physically distinct `MTLBuffer` allocations. See `docs/spike-barrier-alias-audit.md`)
12. Dispatches-per-barrier ratio explains the speed gap (2026-04-13 PARTIALLY FALSIFIED: ratio is 1.44 vs 2.38, but per-kernel profiling showed the gap is kernel execution speed, not pipelining. MoE/MLP/lm_head/norms are all faster than ggml. Only SDPA and dense matmul are slower — due to F32 KV cache and Q8_0 NSG=2)
13. Framework overhead is still the dominant gap after mlx-native migration (2026-04-13 FALSIFIED: mlx-native's single-session concurrent dispatch closed the framework gap to +2.6% vs candle. Remaining 15.5 tok/s is kernel-implementation-dominated — F32 KV cache and Q8_0 threadgroup width)
14. SDPA and dense matmul kernel speed are 2.6x slower than ggml (2026-04-13 FALSIFIED by no-barriers experiment: removing all 606 barriers drops gpu_wait from 10.55ms to 5.88ms, matching llama.cpp's compute floor. The per-kernel profiling session overhead inflated the apparent kernel gap. The real gap is 4.67ms of barrier stall cost, not kernel speed.)
15. Q8_0 NSG=2→4 will close the dense matmul gap (2026-04-13 FALSIFIED: model is 80% Q4_0, 16% Q6_K, 4% Q8_0. NSG=4 change correctness-verified but performance-neutral.)
16. ConflictTracker barrier elision will reduce barrier count (2026-04-13 FALSIFIED: all 606 barriers are true RAW dependencies with physically distinct buffers. The computation graph is a strict chain — no barriers can be elided without changing dispatch ORDER, not just dispatch TIMING.)
17. V-proj reorder is safe because buffer ranges are disjoint (2026-04-13 PARTIALLY FALSIFIED: V-proj after Q-norm showed 6% speedup (119 tok/s) but produced all-pad output. The dependency analysis says the reorder is safe — Metal barriers are full memory fences — but output is wrong. Hidden dependency not yet identified. Investigation required before any reorder work proceeds.)

Each one is a hypothesis that *sounded* right in static analysis and would have produced multi-day patch refutation cycles without the measure-first discipline. The mantra is the load-bearing reason hf2q is at 89.8 tok/s *coherent* rather than at some hypothetical "faster but broken" state. **This ADR's 6-phase plan is structured around the same discipline: measure (Phase 0) before deciding the Phase 4 scope, validate (Phase 3 bitwise) before integrating (Phase 5), and measure again (Phase 6) before declaring Walk done.**

**Cross-reference:** the same mantra section appears verbatim in [ADR-005](ADR-005-inference-server.md). Both ADRs should remain in sync if the mantra source file is updated.

---

## Problem Statement

hf2q's Phase 1b End gate (per ADR-005:162, re-baselined 2026-04-11) requires decode speed `≥102 tok/s` on M5 Max, Gemma 4 26B MoE, Q4_K_M, with byte-identical greedy generation vs llama.cpp at T=0. Coherence is met; speed is not.

**Baseline progression:**
- 2026-04-11 (pre-ADR-006): 84.9 tok/s on candle backend. Gap: 17 tok/s.
- 2026-04-13 (post-mlx-native migration): 89.8 tok/s on mlx-native backend (candle: 87.5). Gap: 15.5 tok/s.
- 2026-04-13 (post-Phase 4a F16 KV + 4b Q8_0 NSG): **91.0 tok/s**. Gap: **15.0 tok/s** to llama.cpp's 106.0 tok/s.

The mlx-native migration closed 2.3 tok/s (framework overhead). F16 KV cache closed 1.2 tok/s (bandwidth). Q8_0 NSG=4 was neutral (only 4% of tensors). The remaining **15.0 tok/s is entirely barrier stall cost** — raw GPU compute matches llama.cpp at ~5.9ms, but our barriers cost 4.67ms vs their ~3.1ms. See §Phase 4 Revision below.

In a single 2026-04-11 session, three consecutive static-evidence-driven kernel-port hypotheses were empirically falsified on M5 Max:

1. **1bNEW.22 sticky compute encoder** — 168k saved encoder creations × ~50 ns each = 0.10% improvement (below noise). Patch correct, payoff zero.
2. **1bNEW.22-v2 NSG sweep** (cfa swarm `swarm-1775949388026-7eii34`) — 8 production dispatch shapes × 4 NSG values × 4 independent runs. Largest reproducible margin 1.08%, well below jitter floor. Static prediction (Agent #3) and runtime measurement (Agent #1) converged on NULL.
3. **1bNEW.29-C Q6_K NR0=2 row-loop port** (cfa swarm `swarm-1775951202282-uwlk55`, Agent C1) — bitwise-identical kernel port (`max|Δ| = 0.000000e0`) timed across 6 production shapes × 2 independent 4-run sweeps. Wall-clock delta within ±1.8% / ±0.4%, sign flips between runs prove pure noise.

The base rate of "static-evidence kernel optimization → measurable speedup on M5 Max" in this codebase is now **0/3**. The diagnosis (recorded as `project_metal_compiler_auto_optimizes_static_levers.md` in user memory): Apple Silicon's Metal compiler auto-applies the optimizations that llama.cpp's hand-tuned ggml-metal kernels make explicit. Where llama.cpp evolved hand-unrolled / per-shape-tuned variants because *some* hardware genuinely benefits, M5 Max's compute units already operate near-optimal at the candle (older llama.cpp snapshot) baseline for hf2q's specific dispatch shapes.

This means **the remaining 17 tok/s does not live in candle's individual kernels.** Where it does live is unknown without per-kernel GPU timing measurement on both stacks (Phase 0 below). Two diagnoses are plausible:

- **Framework-overhead diagnosis** (suggested by 1bNEW.20's +26.83 tok/s win, which was attributed to "freeing the windowed-drain path from pool-wide `flush_and_wait` serialization" — a framework win disguised as a kernel port). Under this diagnosis, the gap is in the per-dispatch overhead candle's general-purpose ML framework imposes on top of correct kernels.
- **Distributed-kernel-implementation diagnosis** (untested) — perhaps several individual kernels (Q4_0 ext variant, fused `rms_norm_mul_*`, FA `dk512_dv512`) are each individually slower in candle than in ggml-metal, and the aggregate accounts for the gap.

Both diagnoses converge on the same strategic answer for **where future optimization work should live**, even though they imply different tactical work.

Independent of the diagnosis, an ownership constraint surfaced (recorded as `project_mlx_native_is_the_strategic_destination.md` in user memory): **mlx-native is owned by Robert; candle is not.** Every candle optimization hf2q has landed becomes a vendor patch maintained against upstream forever (precedents: 1bNEW.20.FIX vendored `candle-nn`, 1bNEW.21 vendored `candle-metal-kernels`). Every mlx-native optimization is full-control, no upstream coordination, no rebase pain. For non-trivial future GPU compute work, mlx-native is the strategically correct destination *regardless of where the current bottleneck is*.

`/opt/coreml-native` (Robert's other Rust-Apple crate, currently at v0.2.0 prepared for crates.io with full sprint cycles) exists as the **existence proof** that mlx-native can reach that maturity. mlx-native is currently at 29 commits with a 1-line README, no docs, no design doc, no `_bmad/` artifacts, no examples beyond benchmarks, no integration tests, no CI, missing unit tests for the largest op (`quantized_matmul`, 1403 LOC). The maturation gap is concrete, bounded, and has been executed once already on a sister crate.

This ADR proposes to migrate hf2q's GPU compute backend from candle to mlx-native, with mlx-native to be matured along the coreml-native trajectory as part of the migration.

---

## Decision Drivers

In rough priority order:

1. **Walk discipline (per ADR-005:162 + `feedback_walk_means_port_llama_cpp_to_rust.md` + user reaffirmation 2026-04-11):** Walk is binary on both axes — coherence AND speed must match llama.cpp. We are NOT done at 84.9 tok/s. Whatever framework we use must let us close the remaining 17 tok/s without shortcuts.
2. **Repository ownership.** Future optimization work belongs in a repo we own. candle is upstream; mlx-native is ours. Vendor patches are a maintenance debt, not a strategy.
3. **Right abstraction layer.** hf2q's product is "control the inference forward pass in Rust." mlx-native sits at exactly that layer. coreml-native sits one layer up (CoreML's runtime owns the forward pass). candle sits one layer broader (general-purpose ML framework where inference is one of many use cases).
4. **Pure Rust, no C++ dependencies** (`feedback_gpu_everything.md`, `project_pure_rust_crate_factory.md`). mlx-native is pure Rust + objc2. candle is pure Rust + objc2. coreml-native is pure Rust + objc2-core-ml. All three meet this driver; none of them require it as a tiebreaker. But candle's general-purpose-ness adds dependencies (CUDA support, multi-backend abstractions) that don't pull their weight for hf2q's specific product.
5. **Maintainable over time without compounding debt.** Every hf2q vendor patch into candle is one more thing to rebase on every candle release. Migrating to mlx-native eliminates that debt class entirely.
6. **Crate-factory alignment** (`project_pure_rust_crate_factory.md`). hf2q is intended to be a workspace publishing reusable Rust crates. mlx-native could become the published crate that other Rust inference projects depend on, the same way coreml-native already is. candle cannot serve that role — it's not ours to publish.
7. **Coherence preservation is non-negotiable** (`project_crawl_walk_run_mental_model.md` — coherence > speed always). The migration must preserve the byte-identical 16-token greedy gen vs llama.cpp at T=0 that the current candle path achieves. Sourdough gate (`scripts/sourdough_gate.sh`) at ≥3094 byte common prefix must pass at every commit.
8. **Mantra-alignment** (`feedback_mantra.md`). No shortcuts, no stubs, no fallbacks, measure 3x cut once, Chesterton's fence on what currently works before changing it.

---

## Considered Options

### Option A — Stay on candle, continue vendor-patching

Continue using candle as hf2q's GPU compute backend. Close the remaining speed gap via additional vendor patches to `candle-metal-kernels`, `candle-nn`, and possibly `candle-core`. Examples: graph scheduler patch, command buffer cadence patch, kernel fusion patches, MTLCounterSampleBuffer instrumentation patch.

**Pros:**
- Zero migration cost; we keep what works
- candle is mature, well-tested by external users
- Smaller LOC delta per individual change than rewriting the framework
- Coherence is already met on this path

**Cons:**
- Every patch is a permanent maintenance burden against upstream candle
- We've already accumulated two vendor patches (`candle-nn` for SDPA byte-offset, `candle-metal-kernels` for `compute_per_buffer`); the rate is monotonically growing
- Framework-level optimization in candle is fundamentally a fight against candle's design (general-purpose ML, not inference-specific)
- candle's command pool architecture (`flush_and_wait` everywhere on sync) is structural; changing it means rewriting candle's dispatch path, which upstream is unlikely to accept
- Doesn't satisfy the ownership driver
- Doesn't satisfy the crate-factory driver
- Compounds debt over Phase 2/3/4/5 where more optimizations are guaranteed

### Option B — Migrate to mlx-native (this ADR's proposal)

Mature mlx-native to coreml-native's v0.2.0 trajectory, port useful work from candle with attribution, port framework patterns from llama.cpp's ggml-metal that candle doesn't have, then swap hf2q's forward pass from candle to mlx-native.

**Pros:**
- Satisfies ownership driver (mlx-native is Robert's repo)
- Satisfies abstraction-layer driver (mlx-native is at exactly the inference-control layer hf2q needs)
- Satisfies crate-factory driver (mlx-native becomes a publishable crate, matching coreml-native's trajectory)
- Eliminates the candle vendor-patch maintenance debt class entirely after cutover
- Lets us match ggml-metal's framework patterns directly without fighting candle's design
- coreml-native existence proof shows the maturation work is bounded and has been done once already
- License-clean borrow path from candle (Apache-2.0 → Apache-2.0 with attribution)
- Forward-compatible: future optimization work has a clean home

**Cons:**
- Significant up-front investment (4-7 weeks)
- mlx-native is currently 29 commits with 1-line README; substantial maturation gap
- mlx-native has missing test coverage on its largest op (`quantized_matmul`, 1403 LOC)
- Recent mlx-native commits are bug fixes, suggesting active issues at integration time
- Multi-day "neither stack is fully production-ready" window during migration
- Speed payoff is unproven until Phase 0 diagnosis completes
- If Phase 0 reveals the gap is somewhere unexpected, the Phase 4 plan must be revised

### Option C — Use coreml-native as hf2q's backend

Convert Gemma 4 26B MoE to CoreML format (`.mlmodelc`), use coreml-native to load and run it via CoreML's runtime, with ANE acceleration.

**Pros:**
- coreml-native is already mature (v0.2.0, prepared for crates.io)
- ANE (Apple Neural Engine) acceleration is unique to this path — neither candle nor mlx-native can use it
- Much smaller code surface to maintain (CoreML's runtime owns the forward pass)
- Apple's runtime is highly optimized for inference workloads on Apple Silicon

**Cons:**
- Wrong abstraction layer for hf2q's product — we lose control of the forward pass entirely
- Conversion path from GGUF/safetensors to CoreML for Gemma 4 26B MoE is unverified; CoreML's standard converters (`coremltools`) may not support the MoE architecture
- ANE acceleration is workload-dependent and unproven for MoE — MoE routing is dispatch-heavy and unlikely to be ANE-friendly; CoreML's planner may fall back to GPU/CPU for the routing path
- Black-box performance — if CoreML's runtime is slow at this model, we have no levers to optimize it
- Doesn't satisfy the "control the forward pass in Rust" product driver
- Locks hf2q to Apple's ecosystem (CoreML), reducing portability if future hf2q targets non-Apple platforms

### Option D — Hybrid: keep candle, add mlx-native for specific ops behind a feature flag

Use mlx-native for specific kernels (quantized matmul, SDPA) behind a feature flag, keep candle for everything else. No full migration.

**Pros:**
- Lower migration risk per step
- Can validate mlx-native one op at a time
- Incremental wins are visible earlier
- Reversible per op

**Cons:**
- Permanent dual maintenance — we maintain candle integration AND mlx-native integration AND the bridge between them, forever
- The framework-overhead win (if it exists) requires owning the *whole* dispatch path, not just individual kernels — running mlx-native kernels through candle's command pool reintroduces the very overhead we're trying to escape
- Doesn't satisfy ownership driver (we still depend on candle long-term)
- Doesn't satisfy crate-factory driver (hf2q can't publish a clean mlx-native-based stack if half the path is candle)
- Looks cheaper than full migration but produces the worst-of-both-worlds maintenance shape

---

## Decision Outcome

**Chosen option: B — Migrate to mlx-native, with the 6-phase plan below.**

**Rationale (one sentence each):**

- Option A (stay on candle) is rejected because the maintenance debt compounds and the framework-level work the speed gap requires is structurally at odds with candle's general-purpose ML design.
- Option B is chosen because it satisfies the ownership, abstraction-layer, and crate-factory drivers simultaneously while preserving the option to close the speed gap via either kernel work or framework work as the Phase 0 diagnosis directs.
- Option C (coreml-native as backend) is rejected as hf2q's *primary* path because it's at the wrong abstraction layer for hf2q's product, but is preserved as a parallel future option for users who want to load pre-compiled CoreML models with ANE acceleration. coreml-native is not in competition with mlx-native; they serve different use cases.
- Option D (hybrid behind a feature flag) is rejected because it produces the worst-of-both-worlds maintenance shape and structurally cannot capture the framework-level wins this ADR is motivated by.

**Status: Proposed (not Accepted).** This ADR moves to Accepted only after Phase 0 (diagnosis) completes. The destination commitment (mlx-native) is independent of Phase 0's outcome — even if Phase 0 reveals the gap is in kernel implementations rather than framework patterns, mlx-native is still the right repo to do that kernel work in. But the *plan section* (specifically Phase 4's scope) depends on Phase 0's findings, and the ADR should not be Accepted until Phase 4 has a measured target rather than a hypothetical one.

---

## The Plan (PRD section)

Six phases with explicit gates. Each phase ends with a deliverable that authorizes the next phase. **No phase starts without the prior phase's gate passing.** No shortcuts, no stubs, no "TODO later" placeholders.

### Phase 0 — Diagnosis

**Goal:** Definitively answer "where does the 17 tok/s actually live?" via direct measurement.

**Why this phase exists:** Three static-evidence kernel hypotheses have been empirically falsified in this session. Continuing to guess at where the gap lives would repeat the lazy pattern. The cost of measurement is one-time; the cost of a wrong guess is multi-day patch work that produces zero speedup.

**Method:**
1. Plumb `MTLCounterSampleBuffer` instrumentation into `vendor/candle-metal-kernels/src/metal/encoder.rs` (vendor patch in hf2q). Capture per-dispatch GPU start/end timestamps. Aggregate per kernel type. ~1 day.
2. Plumb equivalent instrumentation into `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp` (separate vendor patch in /opt/llama.cpp; this is a one-shot diagnostic patch, not landed in mainline). ~1 day.
3. Run hf2q canonical bench (`tests/bench_prompt_128.txt`, T=0, 128 tokens, 5 runs) with the instrumented build. Run llama.cpp on byte-identical rendered prompt (per ADR-005:998 flag set) with the instrumented ggml-metal. Extract per-kernel μs/call median.
4. Build a per-kernel attribution table: candle's kernel X takes A μs/call × N calls/token = A·N μs/token; llama.cpp's equivalent takes B μs/call × M calls/token = B·M μs/token; difference D = (A·N − B·M) μs/token. Sum across all kernel types should account for the wall-clock gap (10.23 ms/token candle-side, ~9.8 ms/token llama.cpp-side per Agent #2's measurement).

**Deliverable:** `docs/spike-1bNEW30-per-kernel-attribution.md` containing:
- Per-kernel attribution table (candle vs ggml-metal at hf2q's exact dispatch shapes)
- Top 3-5 contributors to the gap with ms/token each
- Verdict: framework-overhead-dominated, distributed-kernel-implementation-dominated, or mixed
- Implication for Phase 4 scope (which specific patterns to port)

**Gate to next phase:** Per-kernel attribution table accounts for ≥90% of the 17 tok/s wall-clock gap. We can name the top contributors with measured ms/token each. Hand-waving is no longer acceptable.

**Estimated:** 2-3 days.

**Risks:** MTLCounterSampleBuffer instrumentation may have observer-effect overhead that distorts measurements at fine granularity. Mitigation: also run uninstrumented bench at the same time and confirm wall-clock unchanged (within noise).

### Phase 1 — Architectural Commitment (this ADR)

**Goal:** Lock in mlx-native as hf2q's long-term GPU compute backend, with explicit user sign-off.

**Why this phase exists:** Strategic decisions need written records that survive context loss. The destination commitment is independent of Phase 0's outcome; documenting it before Phase 0 lets the plan section be revised after Phase 0 without re-litigating the destination.

**Method:**
1. This ADR is drafted, status Proposed.
2. ADR-005 gets cross-references at three sites: (a) the Speed gate checkbox at line ~874 ("Migration strategy: see ADR-006"), (b) the Resolved Questions section ("GPU compute backend: see ADR-006 (Proposed 2026-04-11)"), (c) the Next Walk session block ("Phase 0 diagnosis is the entry point per ADR-006").
3. User reviews and signs off on the destination commitment (Option B chosen).
4. Status remains Proposed pending Phase 0; flips to Accepted after Phase 0 confirms the plan section.

**Deliverable:** This ADR file (`docs/ADR-006-mlx-native-gpu-backend.md`) plus the three cross-references in ADR-005.

**Gate to next phase:** ADR drafted, user has signed off on the destination commitment, ADR-005 cross-references landed.

**Estimated:** 1 day (this turn covers most of it).

### Phase 2 — mlx-native Maturation PRD

**Goal:** Define what "mlx-native ready for hf2q production" means by studying coreml-native's already-completed sprint history as the template.

**Why this phase exists:** Without an explicit target state, the maturation work risks being open-ended or shortcut. coreml-native has already executed this trajectory once and provides a ready-made checklist of "what done looks like for a Rust-Apple compute crate".

**Method:**
1. Read `/opt/coreml-native/_bmad-output/prd-coreml-crate.md` and `/opt/coreml-native/_bmad-output/epics-and-stories.md` end-to-end.
2. Read `/opt/coreml-native/CHANGELOG.md` to understand the per-version cadence.
3. Read `/opt/coreml-native/.github/` for the CI workflow shape.
4. Inventory what coreml-native has at v0.2.0: full README (8KB), examples, integration tests, CHANGELOG, CONTRIBUTING, SECURITY, license dual, crate metadata for crates.io, ndarray feature flag, async APIs, batch prediction, model lifecycle management, stateful prediction, device enumeration.
5. Write `mlx-native/_bmad-output/prd-mlx-native-v0.2.md` modeled on the coreml-native PRD but scoped to mlx-native's purpose (Metal compute, not CoreML wrapper). Include:
   - Op coverage audit: every op hf2q's `Gemma4Model::forward` uses, mapped to existing mlx-native ops and gaps
   - Test gap analysis: every load-bearing op should have a unit test that compares output bitwise (or ε ≤ 1e-5) against a known-correct reference
   - Integration design: how does hf2q's `Gemma4Model::forward` talk to mlx-native? Direct `MlxBuffer` + `CommandEncoder`, or a thin wrapper crate?
   - Migration order: which ops swap first (likely the highest-confidence ones — quantized matmul, SDPA), which last
   - Sprint plan with story-level breakdown matching coreml-native's `Story 1.1`-style commit cadence
6. Write a corresponding `mlx-native/_bmad-output/epics-and-stories.md` listing the concrete work items.

**Deliverable:** `mlx-native/_bmad-output/prd-mlx-native-v0.2.md` and `mlx-native/_bmad-output/epics-and-stories.md`. User signs off on the PRD before any code lands in mlx-native.

**Gate to next phase:** PRD signed off; we know what we're building toward; the test gap analysis identifies every load-bearing op that lacks coverage.

**Estimated:** 1-2 days.

### Phase 3 — Borrow Phase (port useful work from candle to mlx-native)

**Goal:** Bring mlx-native to feature parity with what hf2q needs from candle, with full attribution and bitwise correctness validation against the candle source.

**Why this phase exists:** candle has been thoroughly battle-tested for hf2q's specific Gemma 4 26B MoE workload over months of Phase 1b work. Re-deriving every kernel from llama.cpp source would be wasteful and risk re-introducing bugs that have already been fixed in candle. Borrowing with attribution is faster, safer, license-clean (Apache-2.0 → Apache-2.0), and respects the work already done.

**Attribution discipline (mantra-aligned, NON-NEGOTIABLE):** every borrowed file in mlx-native gets a header comment of the form:

```
// Portions of this file are derived from candle-metal-kernels v0.10.2
// (https://github.com/huggingface/candle), Apache-2.0 licensed.
// Source: candle-metal-kernels/src/metal_src/quantized.metal:5186-5294 (kernel_mul_mv_q6_K_f32_impl)
// Modifications: ported to mlx-native's CommandEncoder dispatch path; argument-passing
// adapted to mlx-native's ABI; threadgroup geometry preserved.
//
// Copyright the candle Authors. See LICENSE-APACHE-candle in this directory.
```

A `LICENSE-APACHE-candle` file goes in `/opt/mlx-native/` containing the verbatim Apache-2.0 license text from candle's repo plus the upstream NOTICE if any. No silent copies. Every borrowed file is grep-able by "derived from candle-metal-kernels".

**Sub-phases (each is a separate commit, each gates on bitwise correctness vs the source):**

| # | Borrow target | Source in candle | Validation method |
|---|---|---|---|
| 3a | Quantized matmul kernels: Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, Q8_0 (the `kernel_mul_mv_q*_f32` family) | `candle-metal-kernels/src/metal_src/quantized.metal:2280-6100` | Unit test: dispatch identical input through candle's path and mlx-native's port; assert `max\|Δ\| = 0.000000e0` (bitwise) on each quantization type at hf2q's production shapes |
| 3b | SDPA full kernel `bd=512` BF16 (Gemma 4 global attention head_dim=512) | `candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal:2332-2400` plus the byte-offset fix from `vendor/candle-nn/src/ops.rs::Sdpa::forward` (1bNEW.20.FIX) | Unit test: dispatch the same Q/K/V through both paths; assert `max\|Δ\| ≤ 1e-5`. Specifically test the non-zero `start_offset` case that 1bNEW.20.FIX surfaced. |
| 3c | RoPE neox + rope_freqs | candle's RoPE path + the rope_freqs port from 1bNEW.18 (`src/serve/rope_kernel.rs`) | Unit test: 6 cases mirroring 1bNEW.18's Phase A test suite (decode full-rotary sliding, decode partial-rotary global, prefill partial-rotary global, decode at seqlen_offset=42, prefill partial-rotary generic, GPT-J interleaved sanity). All pass at ε=1e-5. |
| 3d | RmsNorm fused F=1/F=2/F=3 (the 1bNEW.4 ports) | `candle-metal-kernels` runtime-compiled RmsNorm fuse modes | Unit test: 7 cases × float/float4 × F=1/2/3 × 5 shapes, all passing at ε=1e-5 with max\|Δ\| ≤ 2.384e-7 (1 ULP). |
| 3e | F16 lm_head matmul path (1bNEW.17 MLX gemm) | `candle-core/src/metal_backend/mod.rs:1685-1709` `call_mlx_gemm` | Unit test: 3 cases at ε=1e-3 (wider because reduction order deliberately changes), argmax preserved across n_tokens=1 (decode) and n_tokens=8 (prefill). |
| 3f | KV cache in-place append (1bNEW.20) | `candle-core/src/tensor_cat.rs:246` `slice_set` semantics | Unit test: 5 cases at strict `max\|Δ\| = 0.000e0` (sliding decode, global decode, prefill, sliding truncation with grow_if_needed, decode stride correctness). |
| 3g | Embedding, softmax, sampling, transpose, elementwise, gelu, softcap | candle's standard op implementations | Unit tests at ε ≤ 1e-5. |
| 3h | MoE dispatch (`kernel_mul_mv_id_*` family from 1bNEW.1) | `candle-metal-kernels/src/metal_src/quantized.metal:7600-7700` | Unit test: dispatch a 4-token × 8-expert × 2-top-k MoE forward; assert bitwise match. |

Each sub-phase ends with a commit that includes:
- The borrowed source files with attribution headers
- A unit test in `/opt/mlx-native/tests/test_<op>.rs` validating bitwise correctness against a reference
- An entry in `/opt/mlx-native/CHANGELOG.md` describing the borrow and its source

**Deliverable:** mlx-native at op-coverage parity for hf2q's Gemma 4 26B MoE forward pass, with every load-bearing op having a passing unit test that proves bitwise (or ε ≤ 1e-5) correctness against its candle counterpart. mlx-native's README is expanded from 1 line to coreml-native-quality (8 KB+). mlx-native's `_bmad-output/` matches coreml-native's pattern.

**Gate to next phase:** Every load-bearing op in mlx-native produces output that matches its candle counterpart on the same input at the required tolerance. Test coverage gaps from Phase 2 are closed. mlx-native's maturation gap to coreml-native v0.2.0 is closed for the inference-relevant axes (op coverage, test coverage, README, examples).

**Estimated:** 1-2 weeks.

**Risks:**
- Candle's kernel ABI may not map cleanly to mlx-native's `CommandEncoder` ABI; some borrow ports may need argument-passing adaptation. Mitigation: do the ABI work as part of the port; document the adapter pattern.
- mlx-native's existing kernel implementations (e.g., its own Q6_K) may diverge from candle's borrowed version. Mitigation: validate every borrow against BOTH the candle source AND mlx-native's existing implementation; if they diverge, pick the one with the clearest correctness story and document why.

### Phase 4 — Build Phase (kernel-level parity with llama.cpp)

**Goal:** Close the remaining 15.5 tok/s gap by bringing two specific kernel implementations to ggml-metal parity.

**Why this phase exists:** The mlx-native migration (Phases 3+5, now complete) brought framework-level parity — mlx-native's single-session concurrent dispatch eliminated candle's per-dispatch encoder overhead. The remaining gap is **kernel-implementation-dominated**, not framework-overhead-dominated. This was confirmed by comprehensive profiling on 2026-04-13 (see `docs/spike-barrier-perf-measurements.md` and `docs/spike-barrier-alias-audit.md`).

**What the profiling proved:**
- **Not the bottleneck:** Barriers (606/token vs llama.cpp's 759 — we have fewer), buffer aliasing (ruled out — all activation buffers are physically distinct `MTLBuffer` allocations), CPU encode overhead (0.50ms = 4.5% of total), barrier count, pipelining structure.
- **Already faster than ggml:** MoE (0.84x), MLP (0.84x), lm_head (0.87x), norms (0.89-0.93x), KV cache copy (0.94x).
- **The gap (two kernels):**

| Kernel | mlx-native | ggml | Root cause | μs/token overhead |
|--------|-----------|------|-----------|-------------------|
| SDPA (`flash_attn_vec`) | 40 μs/layer | 15.5 μs/layer | **F32 KV cache** (ggml uses F16 — 2x bandwidth), fixed NSG=1 (ggml scales to NSG=4) | 736 (44.6% of gap) |
| Dense quantized matmul (Q8_0 layers) | NSG=2 (64 threads/tg) | NSG=4 (128 threads/tg) | **Half the threadgroup width** for Q8_0 quant type | 652 (39.5% of gap) |

**Phase 4 scope (concrete, measured, ordered by leverage):**

#### 4a — F16 KV cache (highest leverage)

Convert KV cache from F32 to F16. This halves memory bandwidth for SDPA's K/V reads — the dominant cost in a bandwidth-bound decode-path attention kernel at batch=1.

**Changes:**
1. `forward_mlx.rs:820-826`: KV cache allocation `DType::F32` → `DType::F16`, halve `cache_bytes`
2. `kv_cache_copy.metal`: new F32→F16 cast-and-store variant (or template the existing kernel)
3. `flash_attn_vec.metal`: change K/V buffer types from `device const float *` to `device const half *`, add `float4(half4(...))` at load time. The kernel already uses `half` for Q in shared memory — the infrastructure exists.
4. `kv_cache_copy.rs` dispatch: adjust byte sizes for F16 output
5. `flash_attn_vec.rs` dispatch: no change needed (buffer bindings are type-agnostic)

**Expected payoff:** ~736 μs/token. If this alone reaches ggml SDPA parity, predicted tok/s: ~96.5.
**Side benefit:** KV cache memory halved (from 10.7 GB at max capacity to 5.35 GB).
**Coherence risk:** F16 has 3 decimal digits of precision vs F32's 7. The sourdough gate must pass after this change. If it fails, the cast point may need to be after the norm (which reduces the dynamic range).
**Estimated:** 1-2 days.

#### 4b — Q8_0 threadgroup width NSG=2→4

Change `N_SIMDGROUP` from 2 to 4 for Q8_0 matmul kernel. llama.cpp uses `N_SG_Q8_0=4` (128 threads per threadgroup) vs our 2 (64 threads). This directly doubles the threadgroup width for every Q8_0 layer in the mixed Q4/Q8 model.

**Changes:**
1. `quantized_matmul_ggml.metal`: add Q8_0-specific `N_SIMDGROUP_Q8 = 4` or template the kernel
2. `quantized_matmul_ggml.rs` dispatch: change `nth0 × nth1` for Q8_0 from `(8, 8)` to `(8, 16)` or `(32, 4)` (must equal 128 threads = 4 simdgroups)
3. Verify threadgroup memory allocation scales correctly with wider dispatch

**Expected payoff:** substantial portion of 652 μs/token on Q8_0 layers. Exact amount depends on what fraction of dense matmuls in this model use Q8_0 vs Q4_0.
**Estimated:** 0.5-1 day.

#### 4c — Dynamic NSG scaling for flash_attn_vec

After 4a lands, measure the remaining SDPA gap. If significant at longer sequences, add dynamic NSG scaling (1→4 based on KV seq len) matching llama.cpp's `while (2*nwg*nsg*ncpsg < ne11 && nsg < 4) nsg *= 2` logic.

**Changes:**
1. `flash_attn_vec.metal`: template for NSG=1/2/4 (or use function constants)
2. `flash_attn_vec.rs` dispatch: select NSG based on `kv_seq_len` at dispatch time
3. Shared memory allocation must scale with NSG

**Expected payoff:** matters primarily at longer sequences (512+ tokens). May be unnecessary if 4a closes the gap.
**Estimated:** 1 day.

#### 4d — Incremental: loop unrolling and function constants (if needed)

After 4a-4c, if a gap remains:
- Add `FOR_UNROLL` equivalent (`#pragma unroll`) to inner loops in quantized matmul kernels
- Convert runtime constants (`N_SIMDGROUP`, `N_DST`) to Metal function constants for PSO-level optimization
- Port Q4_0 row-pointer pre-computation pattern from llama.cpp's `mul_vec_q_n_f32_impl`

**Expected payoff:** incremental (estimated 1-3% each). Only pursue if 4a+4b don't reach 102 tok/s.
**Estimated:** 1-2 days if needed.

**Attribution discipline:** same as Phase 3. Every ported pattern from ggml-metal gets a header comment with `file:line` source citation and the LICENSE notice.

**Deliverable:** mlx-native commits implementing each sub-phase. Each commit includes a 3-run benchmark showing tok/s progression toward 102. Updated `docs/spike-barrier-perf-measurements.md` with post-fix measurements.

#### 4a — F16 KV cache — DONE (89.8→91.0 tok/s, +1.3%)

Implemented and verified. KV cache allocation F32→F16, `kv_cache_copy_batch_f32_to_f16` kernel, `flash_attn_vec` templated for `KV_T=half`. Byte-identical output for 3094 bytes. KV memory halved.

#### 4b — Q8_0 NSG=4 — DONE (neutral — model is 80% Q4_0)

Implemented and verified correct. Only 12/300 quantized tensors are Q8_0 (4%). No measurable impact on this model.

#### 4a/4b post-mortem: gap re-attributed to barrier stalls

The initial profiling (per-kernel session overhead inflated SDPA/matmul apparent gaps) led to the wrong conclusion. A no-barriers experiment proved:

| Mode | gpu_wait | Raw compute |
|------|----------|-------------|
| Normal (606 barriers) | 10.55ms | 5.88ms compute + 4.67ms barrier stalls |
| No barriers (broken output) | 5.88ms | 5.88ms compute |
| llama.cpp | ~9.0ms | ~5.9ms compute + ~3.1ms barrier stalls |

**The kernels are at parity.** The 15 tok/s gap is 4.67ms vs 3.1ms barrier stall cost. Each of our 606 barriers costs 7.7μs; llama.cpp's 759 barriers cost 4.1μs each. llama.cpp pays less per barrier because its graph reordering puts more dispatches between barriers (2.38 vs our 1.44).

**Infrastructure implemented:**
- `ConflictTracker` in `mlx-native/src/graph.rs` — automatic buffer-range-based barrier elision matching llama.cpp's `ggml_mem_ranges` system
- All `s.barrier()` calls in `forward_mlx.rs` converted to `s.barrier_between(reads, writes)` — correctly emits 606/606 barriers (all true RAW conflicts confirmed)

#### 4e — Computation graph with fusion, reorder, and dual-buffer encoding

**Goal:** Close the remaining barrier stall gap (4.67ms → ≤3.1ms) by porting llama.cpp's full Metal execution architecture: graph IR → fusion pass → reorder pass → dual command buffer encoding. Target: 93.6 → ≥102 tok/s.

**Measured baselines (2026-04-13, clean GPU, 3-run median each):**
- hf2q: **93.6 tok/s** (post router/MLP interleave), encode=0.53ms, gpu_wait=10.52ms, 486 barriers, ratio≈1.73
- llama.cpp: **103.5 tok/s**, 9.66 ms/token, ~1811 nodes, 759 barriers, ratio=2.39

**Concurrent group histogram (measured):**
- hf2q: 456 × size-1 (75%), 65 × size-2 (11%), 85 × size-3 (14%), 0 × size-4+
- llama.cpp: ~40% singletons, ~50% size-3+, groups up to size-4

---

##### Deep analysis of llama.cpp's Metal execution architecture

Source study completed 2026-04-13. Every claim below is cited to a specific file and line range in `/opt/llama.cpp/ggml/src/ggml-metal/`.

**Execution pipeline (4 stages):**

```
graph_build → graph_optimize → encode → GPU execute
   (ggml)     (ggml-metal-      (ggml-metal-    (Metal
               common.cpp)       context.m)      hardware)
```

**Stage 1: Graph construction** — ggml builds `ggml_cgraph` during the forward pass. Each tensor operation is a node (`ggml_tensor *`) with `src[0..GGML_MAX_SRC]` input pointers and an implicit output (the tensor itself). The graph is a DAG — dependencies are structural through pointer relationships, not explicit edges.

**Stage 2: Graph optimization** (`ggml_graph_optimize` at `ggml-metal-common.cpp:375-473`) — runs once per forward pass BEFORE any Metal encoding. Two sub-passes:

*Sub-pass 2a — Fusion* (`ggml-metal-common.cpp:386-432`):
- Scans consecutive nodes for fusable sequences starting with `ADD`, `NORM`, or `RMS_NORM`.
- Uses `ggml_can_fuse()` (`ggml-impl.h:693-706`) which validates: all nodes same shape, all intermediate nodes have exactly 1 consumer, each node is a src of the next.
- Fused sequences are packed into `node_info.fused` — the fused nodes travel as an atomic unit through the reorder pass. This prevents reorder from breaking fusion groups.
- Concrete fused kernel pipelines (`ggml-metal-device.cpp:1592-1633`):
  - `kernel_rms_norm_f32` (n_fuse=1, unfused baseline)
  - `kernel_rms_norm_mul_f32` (n_fuse=2: RMS_NORM + MUL in one dispatch)
  - `kernel_rms_norm_mul_add_f32` (n_fuse=3: RMS_NORM + MUL + ADD in one dispatch)
  - Same for NORM: `kernel_norm_f32`, `kernel_norm_mul_f32`, `kernel_norm_mul_add_f32`
  - Binary ops: `kernel_bin_fuse_f32_f32_f32` with function constants for op type, fuse count, row-broadcast, column-broadcast (`ggml-metal-device.cpp:1467-1511`)
  - Each fusion eliminates 1-2 dispatches AND 1-2 barriers between them.

*Sub-pass 2b — Reorder* (`ggml_metal_graph_optimize_reorder` at `ggml-metal-common.cpp:209-373`):
- Greedy algorithm with 64-node lookahead (`N_FORWARD = 64`).
- Maintains two memory range sets:
  - `mrs0`: ranges of the current concurrent group (nodes that will execute between the same pair of barriers)
  - `mrs1`: ranges of skipped-over nodes (nodes between the current position and the reorder candidate)
- When node `i0` conflicts with `mrs0` (would need a barrier), the algorithm looks forward up to 64 nodes for any node `i1` that satisfies BOTH:
  - `h_check(mrs0, node1)` — concurrent with the existing concurrent set
  - `h_check(mrs1, node1)` — concurrent with all unprocessed nodes between `i0` and `i1`
- Reorderable nodes are pulled forward into the current concurrent set. Marked `used[i1] = true` so they're skipped when the main loop reaches them.
- Safety whitelist (`h_safe` at `ggml-metal-common.cpp:259-291`): only MUL_MAT, MUL_MAT_ID, ROPE, NORM, RMS_NORM, L2_NORM, SUM_ROWS, CLAMP, MUL, ADD, SUB, DIV, GLU, SCALE, UNARY, GET_ROWS, SET_ROWS, SET, CPY, CONT, REPEAT. NOT SDPA, NOT SOFTMAX — those are treated as barriers to reordering.
- Memory range conflict check (`ggml_mem_ranges_check` at `ggml-metal-common.cpp:124-153`): ranges identified by `(buffer_id, p0, p1, type)`. Two SRC (read) ranges in the same buffer: OK. Any SRC overlapping a DST, or any DST overlapping anything: CONFLICT. Uses `view_src` to resolve tensor views to their base allocation. This is functionally identical to our `ConflictTracker::conflicts()`.
- The output is a permutation `Vec<int>` of node indices. The original `gf->nodes[]` array is rewritten in the new order (`ggml-metal-common.cpp:461-472`), with fused sub-nodes unpacked inline.

**Stage 3: Encoding** (`ggml_metal_graph_compute` at `ggml-metal-context.m:441-644`):
- Default `n_cb = 1` (set at `ggml-metal.cpp:602`), meaning 2 command buffers total.
- `n_nodes_0 = MAX(64, 0.1 * gf->n_nodes)` ≈ 181 nodes for our graph size. These are encoded by the main thread into `cmd_bufs[n_cb]` and committed immediately via `[cmd_buf enqueue]` + `[cmd_buf commit]` (`ggml-metal-context.m:531-545`).
- The remaining `n_nodes_1` ≈ 1630 nodes are encoded into `cmd_bufs[0]` by a separate thread via `dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async)` (`ggml-metal-context.m:572`).
- The GPU starts executing cmd_buf[n_cb] while the async thread is still encoding cmd_buf[0]. This overlaps ~0.5ms of CPU encode time with GPU execution.
- Each command buffer uses `MTLDispatchTypeConcurrent` encoder (`ggml-metal-device.m:850-851`).
- During encoding, `ggml_metal_op_concurrency_check` (`ggml-metal-ops.cpp:227`) runs the SAME buffer-range conflict check as the reorder pass. If conflict → `ggml_metal_op_concurrency_reset` emits `memoryBarrierWithScope:MTLBarrierScopeBuffers` and resets ranges. If no conflict → dispatch runs concurrently with the previous dispatches in the group.
- Because the graph was pre-reordered in Stage 2, nodes arrive at the encoder in an order that maximizes concurrent groups. The encoder's conflict check confirms what the reorder already arranged.

**Stage 4: GPU execution** — Metal's hardware scheduler executes dispatches within a concurrent group in parallel (subject to resource availability). Barriers are full memory fences that serialize across groups.

---

##### Why imperative reorder failed (V-proj experiment)

Two clean attempts to reorder V-proj after Q-norm in our imperative forward pass produced all-pad output, despite the buffer ranges being disjoint. Adding a forced barrier before the reordered dispatch fixed correctness.

The root cause is now understood: llama.cpp's reorder happens at the **graph data structure level** — it changes the order nodes appear in `gf->nodes[]`. The encoder then walks this array linearly, encoding each node sequentially. The encoder's `ggml_metal_op_concurrency_check` decides per-node whether to emit a barrier. The Metal concurrent encoder sees a single linear stream of dispatches interspersed with barriers.

Our imperative approach changes the **Rust source code order** of dispatch calls within a concurrent encoder. This is semantically different — Metal's concurrent dispatch scheduler may process imperative dispatches with different timing/ordering guarantees than graph-scheduled dispatches. The graph-then-encode pattern is the only one llama.cpp has validated at scale.

**Falsified:**
- 17. V-proj reorder with disjoint buffers is safe under imperative concurrent dispatch (2026-04-13 FALSIFIED: 2 clean implementations produce all-pad output; forced barrier fixes it)
- 18. Imperative dispatch reordering is equivalent to graph-level reorder + sequential encoding (2026-04-13 FALSIFIED by the same evidence — they produce different behavior on Apple Silicon's Metal concurrent encoder)

---

##### Architecture for mlx-native computation graph

Port llama.cpp's full execution architecture. No shortcuts, no half-measures.

**4e.1 — Graph IR** (`mlx-native/src/graph.rs`, new `ComputeGraph` type)

```rust
/// A recorded kernel dispatch — the graph node.
pub struct GraphNode {
    /// Which pipeline state object to bind.
    pipeline: metal::ComputePipelineState,
    /// Kernel argument bindings: (slot_index, buffer_or_bytes).
    bindings: SmallVec<[(u64, BindingKind); 8]>,
    /// Threadgroup dispatch dimensions.
    threads_per_grid: MTLSize,
    threads_per_threadgroup: MTLSize,
    /// Optional threadgroup memory size.
    threadgroup_memory: Option<(u64, u64)>,
    /// Read buffer ranges: (contents_ptr, byte_len) — for conflict detection.
    reads: SmallVec<[(usize, usize); 4]>,
    /// Write buffer ranges: (contents_ptr, byte_len) — for conflict detection.
    writes: SmallVec<[(usize, usize); 2]>,
    /// Fused sub-nodes (travel as atomic unit through reorder).
    fused: SmallVec<[usize; 2]>,
    /// Op tag for the safety whitelist (can this node be reordered?).
    op_kind: OpKind,
}

pub enum OpKind {
    MatMul, MatMulId, Norm, Rope, Elementwise, Copy, Gather,
    Sdpa,       // NOT reorderable
    Softmax,    // NOT reorderable
    Other,      // NOT reorderable
}
```

The forward pass calls `graph.record_*()` methods instead of dispatching directly. Each method mirrors the existing `GraphSession` op methods but pushes a `GraphNode` onto a `Vec<GraphNode>` instead of encoding into a Metal command encoder.

`ConflictTracker` is reused as-is — the conflict detection logic is identical to llama.cpp's `ggml_mem_ranges_check`. It's used at both reorder time (to validate reorder candidates) and encode time (to decide barrier placement).

**4e.2 — Fusion pass** (runs after forward pass finishes recording)

Scan the recorded `Vec<GraphNode>` for fusable sequences:

- `RMS_NORM + MUL` → replace 2 nodes with 1 node dispatching `kernel_rms_norm_mul_f32`
- `RMS_NORM + MUL + ADD` → replace 3 nodes with 1 node dispatching `kernel_rms_norm_mul_add_f32`

Fusion rules (matching `ggml_can_fuse_ext` at `ggml-impl.h:663-689`):
- All nodes must be same shape
- All intermediate nodes must have exactly 1 consumer (no other node reads their output)
- Each node must be a src of the next node
- No node can have the OUTPUT flag set

Consumer-count tracking: during graph recording, maintain a `HashMap<usize, u32>` mapping each buffer's `contents_ptr` to the number of nodes that read from it. A node has 1 consumer if its write buffer appears in exactly 1 subsequent node's read set.

New fused Metal kernels needed in mlx-native:
- `kernel_rms_norm_mul_f32` / `kernel_rms_norm_mul_f32_4` (4-wide vectorized variant for ne00 % 4 == 0)
- `kernel_rms_norm_mul_add_f32` / `kernel_rms_norm_mul_add_f32_4`

These can be ported directly from llama.cpp's `ggml-metal.metal` with attribution (Apache-2.0). The kernel takes the norm input, the scale weights (MUL src[1]), and optionally the bias weights (ADD src[1]), producing the fused output in one pass over shared memory. This eliminates the intermediate buffer allocation AND the barrier between norm and mul.

Each fusion for Gemma4's forward pass saves 1-2 dispatches and 1-2 barriers per layer × 30 layers = 30-60 fewer barriers per token.

**4e.3 — Reorder pass** (runs after fusion, before encoding)

Port `ggml_metal_graph_optimize_reorder` (`ggml-metal-common.cpp:209-373`) to Rust:

```rust
fn reorder(nodes: &[GraphNode]) -> Vec<usize> {
    let n = nodes.len();
    let mut result = Vec::with_capacity(n);
    let mut used = vec![false; n];
    let mut mrs0 = ConflictTracker::new(); // current concurrent set
    let mut mrs1 = ConflictTracker::new(); // skipped-over nodes

    for i0 in 0..n {
        if used[i0] { continue; }
        let node0 = &nodes[i0];

        if !node0.is_empty() && mrs0.conflicts_node(node0) {
            mrs1.reset();
            mrs1.add_node(node0);

            // 64-node lookahead
            for i1 in (i0 + 1)..min(i0 + 64, n) {
                if used[i1] { continue; }
                let node1 = &nodes[i1];
                if !node1.op_kind.is_reorderable() { break; }

                let is_empty = node1.is_empty();
                if (is_empty || !mrs0.conflicts_node(node1))
                    && !mrs1.conflicts_node(node1)
                {
                    mrs0.add_node(node1);
                    result.push(i1);
                    used[i1] = true;
                } else {
                    mrs1.add_node(node1);
                }
            }
            mrs0.reset(); // emit barrier boundary
        }

        mrs0.add_node(node0);
        result.push(i0);
    }
    result
}
```

The reorderable-op whitelist (matching `h_safe` at `ggml-metal-common.cpp:259-291`): MatMul, MatMulId, Rope, Norm, Elementwise (add/mul/sub/div/scale/unary/gelu), Copy, Gather, SumRows, Clamp. NOT Sdpa, NOT Softmax.

**4e.4 — Dual command buffer encoding** (runs after reorder)

Port `ggml_metal_graph_compute` (`ggml-metal-context.m:441-644`) to Rust:

```rust
fn encode_and_execute(device: &MlxDevice, nodes: &[GraphNode], order: &[usize]) {
    let n = order.len();
    let n0 = max(64, n / 10);  // ~10% for immediate commit

    // Command buffer 0: first n0 nodes, committed immediately
    let mut enc0 = device.command_encoder()?;
    let mut tracker0 = ConflictTracker::new();
    for &idx in &order[..n0] {
        if tracker0.conflicts_node(&nodes[idx]) {
            enc0.memory_barrier();
            tracker0.reset();
        }
        nodes[idx].encode_into(&mut enc0);
        tracker0.add_node(&nodes[idx]);
    }
    enc0.commit(); // GPU starts executing immediately

    // Command buffer 1: remaining nodes, encoded while GPU runs cmd_buf_0
    let mut enc1 = device.command_encoder()?;
    let mut tracker1 = ConflictTracker::new();
    for &idx in &order[n0..] {
        if tracker1.conflicts_node(&nodes[idx]) {
            enc1.memory_barrier();
            tracker1.reset();
        }
        nodes[idx].encode_into(&mut enc1);
        tracker1.add_node(&nodes[idx]);
    }
    enc1.commit_and_wait()?; // wait for both buffers to complete
}
```

The key insight: `enc0.commit()` (without wait) submits to the GPU immediately. `enc1` is then encoded on the CPU while `enc0` executes on the GPU. The 0.53ms CPU encode time now overlaps with GPU execution instead of serializing.

**4e.5 — Integration into GraphSession**

The existing `GraphSession` API changes from "encode directly" to "record then optimize then encode":

```rust
impl GraphSession {
    // Existing op methods change internally: push GraphNode instead of dispatch
    pub fn rms_norm(...) { self.graph.record_rms_norm(...); }
    pub fn quantized_matmul(...) { self.graph.record_quantized_matmul(...); }
    // etc.

    // New: optimize and execute the recorded graph
    pub fn finish(self) -> Result<()> {
        let mut graph = self.graph;
        graph.fuse();           // 4e.2
        let order = graph.reorder(); // 4e.3
        graph.encode_dual_buffer(&self.device, &order)?; // 4e.4
        Ok(())
    }
}
```

The forward pass in `forward_mlx.rs` requires NO CHANGES — it calls the same `s.rms_norm()`, `s.quantized_matmul()`, `s.barrier_between()` methods. The difference is purely internal: those methods now record instead of dispatching. `barrier_between()` becomes a no-op at record time (barriers are computed from the optimized graph at encode time).

---

##### Execution plan

| Sub-phase | Deliverable | Estimated |
|-----------|-------------|-----------|
| 4e.1 | `GraphNode`, `ComputeGraph`, `OpKind` types in `graph.rs`. `record_*()` methods. Forward pass unchanged. | 2 days |
| 4e.2 | Fusion pass + `kernel_rms_norm_mul_f32` + `kernel_rms_norm_mul_add_f32` Metal shaders (ported from llama.cpp with attribution). Unit tests proving fused output matches sequential. | 2 days |
| 4e.3 | Reorder pass (Rust port of `ggml_metal_graph_optimize_reorder`). Debug output matching llama.cpp's `GGML_REORDER_DUMP` format for validation. | 1-2 days |
| 4e.4 | Dual command buffer encoding. Benchmark showing CPU/GPU overlap. | 1 day |
| 4e.5 | Integration: `GraphSession::finish()` runs fuse→reorder→encode. Forward pass unchanged. Full benchmark + sourdough gate. | 1-2 days |

**Total estimated: 7-9 days.**

**Validation at each sub-phase:**
- 4e.1: Record-then-encode produces identical output to direct-encode (byte-identical sourdough gate)
- 4e.2: Fused kernels match unfused sequential output at ε ≤ 1e-5; dispatch count drops measurably
- 4e.3: Reorder changes group size histogram toward llama.cpp's distribution; sourdough gate passes
- 4e.4: `encode_ns` drops measurably (CPU/GPU overlap)
- 4e.5: 3-run median tok/s ≥ 102 with sourdough gate passing

**Gate to Phase 5:** 3-run median tok/s ≥ 102 with sourdough gate ≥ 3094 byte common prefix.

### Phase 5 — Integration into hf2q

**Goal:** Swap hf2q's `Gemma4Model::forward` from candle to mlx-native, op-by-op behind a feature flag, with sourdough gate validation at every step.

**Why this phase exists:** A big-bang cutover risks an unbounded debugging window with no working baseline. Per-op gradual swap is reversible, gives a measurable signal at every step, and respects the mantra discipline (no shortcuts, no "we'll figure it out").

**Method:**
1. Add a new Cargo feature in `hf2q/Cargo.toml`: `mlx-native-backend = ["dep:mlx-native"]`. Default OFF; explicitly opt-in via `--features mlx-native-backend`. Adds `mlx-native = { path = "/opt/mlx-native" }` as an optional dependency.
2. For each op category in Phase 3's sub-phase order (3a quantized matmul → 3b SDPA → 3c RoPE → 3d RmsNorm → 3e lm_head → 3f KV cache → 3g misc → 3h MoE), add a per-op CLI flag: `--matmul-backend candle|mlx-native`, `--sdpa-backend candle|mlx-native`, etc. Default to `candle` for every op.
3. For each op category, in sequence:
   - Wire the mlx-native path into `src/serve/gemma4.rs` behind the per-op flag
   - Run sourdough gate with `--matmul-backend mlx-native` (and so on for each op as we go)
   - Run canonical 5-run bench, record tok/s
   - If sourdough gate fails: stop, debug, fix in mlx-native, re-validate. Do NOT proceed to the next op while any prior op's mlx-native path is broken.
   - If sourdough passes and tok/s ≥ candle baseline: commit, move to next op
   - If sourdough passes but tok/s < candle baseline: investigate why before proceeding (the framework-overhead win should manifest at this step; if it doesn't, Phase 0's diagnosis was wrong)
4. After every op is migrated and validated individually, flip the defaults to `mlx-native` for all ops. Sourdough gate runs full sweep. Canonical bench runs full sweep.
5. After defaults flipped: drop the `candle` paths from `src/serve/gemma4.rs` (no fallback per `feedback_no_shortcuts.md`). Drop the candle dependency from `hf2q/Cargo.toml`. Drop the `vendor/candle-nn/` and `vendor/candle-metal-kernels/` vendor patches (they're no longer needed).

**Deliverable:** hf2q runs end-to-end on mlx-native for the canonical bench. Sourdough gate passes at every commit. Every op flag's mlx-native path is validated. Final commit removes candle dependency entirely.

**Gate to next phase:** Sourdough gate continuously green across the full per-op cutover. Canonical bench tok/s ≥ 84.9 (the post-1bNEW.22 baseline) at every commit. The full mlx-native path is the default.

**Estimated:** 1 week.

**Risks:**
- An op's mlx-native path may be subtly wrong in a way the unit test missed but the sourdough gate catches. Mitigation: every per-op cutover is its own commit with its own sourdough gate run; we never advance with a known regression.
- The order of op cutovers may interact (e.g., RoPE precision change cascades through the rest of the forward). Mitigation: Phase 3's tolerances are tight enough (bitwise where possible, ε ≤ 1e-5 elsewhere) that cascade effects should be bounded; if not, slow down.

### Phase 6 — Final Measurement and Walk Closure

**Goal:** Validate that hf2q + mlx-native ≥ 102 tok/s on canonical bench, with sourdough gate passing AND byte-identical 16-token greedy gen vs llama.cpp at T=0.

**Method:**
1. 5-run canonical bench (`tests/bench_prompt_128.txt`, T=0, 128 tokens) on the post-Phase-5 hf2q HEAD. Record median tok/s.
2. Sourdough gate (`scripts/sourdough_gate.sh`) full run. Record common-byte-prefix vs llama.cpp.
3. Byte-identical 16-token greedy gen check vs llama.cpp at T=0 on the canonical prompt (Agent #2's coherence baseline from `docs/spike-1bNEW29-llamacpp-timings.md`).
4. Update ADR-005:836 (the speed gate checkbox) to MET, citing this commit.
5. Update ADR-006 status to Accepted (from Proposed).
6. Update the cumulative falsified register with whichever hypotheses were closed by the migration.

**Deliverable:** ADR-005 Walk-speed End gate marked MET. ADR-006 status flipped to Accepted. Cumulative falsified register updated.

**Gate to "done":** Median canonical bench tok/s ≥ 102. Sourdough gate common-byte-prefix ≥ 3094 (the existing floor). Byte-identical 16-token greedy gen vs llama.cpp at T=0 preserved. Walk is complete on both axes.

**If Phase 6 fails** (speed < 102 OR coherence regression): identify the remaining gap, decide whether it's a Phase 4 (more framework patterns) or Phase 3 (more kernel borrows / different reference) issue, do the additional work, return to Phase 6. **No giving up at < 102.** Walk = match peer, period.

**Estimated:** 1-2 days.

---

## Validation

The ADR is **Validated** when:
1. Phase 0 is complete and the per-kernel attribution table accounts for ≥90% of the wall-clock gap with measured numbers.
2. Phase 5 is complete and hf2q is running on mlx-native end-to-end with sourdough gate passing.
3. Phase 6 is complete and canonical bench tok/s ≥ 102 with byte-identical greedy gen.

Status flips from **Proposed → Accepted** at the end of Phase 0 (when the diagnosis confirms the migration plan's scope is achievable). Status flips from **Accepted → Implemented** at the end of Phase 6.

If any phase produces evidence that refutes this ADR (e.g., Phase 0 shows the gap is somewhere we cannot fix in either candle or mlx-native, OR Phase 3 shows mlx-native cannot reach correctness parity with candle at hf2q's tolerances, OR Phase 4 shows the framework patterns produce no measurable speedup), the ADR is **revisited honestly** rather than pushed through. The mantra (`feedback_mantra.md`) takes precedence over the plan: if the plan turns out to be wrong, we re-plan, we don't ratchet.

---

## Consequences

### Positive

- **Ownership.** All future hf2q GPU compute optimization work happens in a repo Robert owns. Vendor-patch maintenance debt against candle is eliminated post-cutover.
- **Strategic alignment with the "Pure Rust crate factory" vision.** mlx-native becomes a published crate that other Rust inference projects can depend on, joining coreml-native in Robert's published-crate portfolio.
- **Right abstraction layer for hf2q's product.** mlx-native is at exactly the inference-control layer hf2q's product needs. No more fighting against candle's general-purpose ML design.
- **Walk closure.** Phase 6 closes the speed gate (≥102 tok/s) and the coherence gate (byte-identical 16-token gen) simultaneously, completing Phase 1b on both axes.
- **mlx-native maturation as a side effect.** mlx-native goes from 29-commit WIP to v0.2.0-prepared-for-crates.io quality. README, examples, integration tests, CI, CHANGELOG, sprint history — all on the coreml-native trajectory.
- **Forward-compatibility with Phase 2/3/4/5.** The HTTP server (Phase 2), vision (Phase 3), auto pipeline (Phase 4), and multi-model (Phase 5) work all benefit from mlx-native being the backend rather than candle. Each future phase has a clean home for its compute work.
- **License-clean borrow path.** Apache-2.0 (candle) → Apache-2.0 (mlx-native) with attribution preserves the option to use any candle code we need without future legal concerns.
- **Coherence preservation.** Per-op cutover with sourdough gate at every commit means the migration never breaks coherence — Walk-correctness stays met throughout.

### Negative

- **4-7 weeks of focused work** before Phase 6 closes Walk. This is the largest single Walk investment in Phase 1b's history.
- **mlx-native maturation cost is real.** The README, docs, test coverage, integration tests, CI, CHANGELOG work is not optional under the mantra discipline; it's the difference between "WIP that hf2q happens to use" and "v0.2.0 publishable crate that hf2q depends on".
- **Vendor patches to candle continue until Phase 5 cutover.** The existing `candle-nn` (1bNEW.20.FIX) and `candle-metal-kernels` (1bNEW.21) vendor patches stay live throughout the migration. They're deleted at the end of Phase 5.
- **Integration risk during Phase 5's per-op cutover.** Each op swap is a potential bug surface. Sourdough gate at every commit mitigates but doesn't eliminate this.
- **mlx-native's existing kernels may have latent bugs** that surface during Phase 3's borrow-and-validate work. The recent commit history (Q6 matmul fix, RoPE freq fix, MoE per-expert scale fix) shows active fixing — Phase 3 will likely surface more.
- **Speed payoff is unproven until Phase 0 completes.** If Phase 0 reveals the gap is somewhere unexpected (e.g., a Metal driver behavior we can't control), the speed payoff may be smaller than projected. Mitigation: Phase 0 comes first specifically to surface this risk early.

### Neutral

- **coreml-native stays alive as a separate parallel backend option.** Not in competition with mlx-native; serves a different use case (load pre-compiled `.mlmodelc` with ANE acceleration). hf2q could someday support both via `--backend mlx-native|coreml-native` flags. Out of scope for this ADR.
- **candle remains a dependency of /opt/hf2q's quantization pipeline** (not the inference path) for as long as the quantization code uses it. Phase 5 only removes candle from the inference path. A separate future ADR could address quantization.
- **mlx-native's name** ("MLX-native" — referencing Apple's MLX framework) is a historical artifact from when the crate was started before candle was discovered. The name does NOT mean mlx-native depends on Apple's MLX framework; it's pure Rust + Metal. Considering whether to rename is out of scope for this ADR.

---

## Open Questions / Risks

1. **Does Phase 0's diagnosis confirm the framework-overhead hypothesis or refute it?** Affects Phase 4 scope substantially. The destination commitment (mlx-native) is independent; the work at Phase 4 is not.

2. **Can mlx-native's existing kernels (especially `quantized_matmul.rs` at 1403 LOC with no unit test) be made bitwise-correct vs candle's at hf2q's shapes?** Phase 3a is the test. If it fails, Phase 3a becomes "rewrite mlx-native's quantized_matmul from candle's source" rather than "validate the existing one", which extends Phase 3 by 2-4 days.

3. **Is the `nb01 = 0` call-site contract issue Agent #3 documented at `vendor/candle-metal-kernels/src/kernels/quantized.rs:43-45` going to bite during Phase 3?** It's a borrow-time concern. mlx-native's host dispatcher will need to pass non-zero row strides for any borrowed kernel that uses them.

4. **Will mlx-native's `MlxBufferPool` (power-of-two bucketing) handle hf2q's KV cache growth pattern correctly?** Phase 3f is the test. If the pool fragments under sliding-window KV growth, Phase 3 needs an additional sub-phase for memory allocator work.

5. **Does coreml-native's PRD template actually map to mlx-native's purpose?** Phase 2 will surface this. coreml-native is a wrapper crate; mlx-native is a compute library. The maturation axes (README, tests, CI, examples) are universal; the API design axes are not.

6. **Is the 4-7 week estimate realistic?** Phase estimates compound; the longest tail is Phase 4 at 1-2 weeks because it's gated on Phase 0's findings. Realistic worst case is closer to 8-10 weeks if Phase 0 reveals a surprising gap distribution.

7. **What if Phase 6 fails and we cannot reach 102 tok/s even on mlx-native?** The mantra requires not lowering the bar; the loop is to identify the remaining gap and do additional Phase 3/4 work. But there is a finite scenario where the M5 Max hardware itself caps below 102 due to thermal or driver constraints we cannot control. In that scenario, the End gate gets re-baselined a second time (per the precedent of `project_end_gate_reality_check.md`) with explicit user authorization, not silently.

---

## Cross-references to other ADRs

- **ADR-005 (Inference Server):** Phase 1b speed gap is the immediate motivation for this ADR. Cross-references go in three places in ADR-005:
  - Line ~162 (Walk Replan End gate): "GPU compute backend migration: see ADR-006"
  - Line ~874 (Acceptance Criteria speed checkbox): "Migration strategy: see ADR-006"
  - Resolved Questions section: "GPU compute backend choice: see ADR-006 (Proposed 2026-04-11)"
  - Next Walk session block: "Phase 0 diagnosis is the entry point per ADR-006"
- **ADR-004 (GGUF Compatibility):** mlx-native must preserve all of ADR-004's correctness work. Specifically: Q4_0/Q8_0 block repacking, dimension ordering, architecture-aware tensor name mapping, V=K duplicate tensors for full-attention layers, K-quant block size fallback. No regressions.

---

## References

### Project memory files (user-private; load-bearing context)
- `project_mlx_native_is_the_strategic_destination.md` — ownership rationale
- `project_crawl_walk_run_mental_model.md` — Walk = port llama.cpp to Rust, coherence > speed
- `feedback_walk_means_port_llama_cpp_to_rust.md` — Walk includes the framework, not just the kernels
- `feedback_ground_truth_is_what_we_can_measure_now.md` — peer references must be re-measured on the day
- `project_metal_compiler_auto_optimizes_static_levers.md` — three-falsified-hypothesis pattern
- `project_candle_metal_kernels_lineage.md` — candle's quantized kernels are an older llama.cpp snapshot
- `feedback_swarm_sequential_when_shared_build.md` — sequencing rule for build-touching workers
- `project_pure_rust_crate_factory.md` — hf2q workspace publishes reusable crates
- `feedback_mantra.md` — the engineering mantra
- `feedback_no_shortcuts.md` / `feedback_correct_outcomes.md` — never accept partial outcomes
- `feedback_prove_in_code.md` — never assume from memory/docs; read the code and test to verify

### Spike reports from the 2026-04-13 session (Phase 4 revision)
- `docs/spike-barrier-perf-measurements.md` — comprehensive per-kernel profiling on mlx-native: gap attribution to F16 KV + Q8_0 NSG
- `docs/spike-barrier-alias-audit.md` — exhaustive buffer alias audit proving all 606 barriers are necessary and no cross-layer aliasing exists

### Spike reports from the 2026-04-11 session
- `docs/spike-1bNEW22-instrumentation.md` — sticky encoder hypothesis falsified; CPU/GPU framing corrected
- `docs/spike-1bNEW29-pre-microbench-results.md` — synthesis of three-worker pre-microbench session
- `docs/spike-1bNEW29-nsg-sweep-data.md` — NSG sweep falsification (Agent #1)
- `docs/spike-1bNEW29-llamacpp-timings.md` — llama.cpp re-measurement at 102.01 tok/s (Agent #2)
- `docs/spike-1bNEW29-research-notes.md` — candle-vs-llama.cpp citation map + falsification backpointer (Agent #3)
- `docs/spike-1bNEW29C-q6k-nr0-microbench.md` — Q6_K NR0=2 falsification (Agent C1)

### External references
- `/opt/mlx-native/` — the destination repo
- `/opt/coreml-native/` — the maturation template (v0.2.0 prepared for crates.io)
- `/opt/coreml-native/_bmad-output/prd-coreml-crate.md` — Phase 2's source template
- `/opt/llama.cpp/ggml/src/ggml-metal/` — the framework patterns to port in Phase 4
- `vendor/candle-nn/` — the 1bNEW.20.FIX vendor patch (deleted at end of Phase 5)
- `vendor/candle-metal-kernels/` — the 1bNEW.21 vendor patch + the source for Phase 3 borrows (deleted at end of Phase 5)

---

## Sign-off

| Role | Name | Date | Status |
|---|---|---|---|
| Author | Claude | 2026-04-11 | Drafted |
| Decision maker | Robert | 2026-04-12 | Accepted — Phase 0 verdict: framework-overhead-dominated |
| Phase 4 revision 1 | Claude | 2026-04-13 | Scope: kernel-implementation (F16 KV + Q8_0 NSG) |
| Phase 4a/4b | Claude | 2026-04-13 | DONE — F16 KV (+1.3%), Q8_0 NSG (neutral). 91.0 tok/s. |
| Phase 4 revision 2 | Claude | 2026-04-13 | Gap re-attributed: barrier stalls (4.67ms, 44% of GPU time). Dispatch reorder is the lever. |
| Phase 4 revision 3 | Claude | 2026-04-13 | V-proj imperative reorder FALSIFIED (2 clean attempts → all-pad). Root cause: Metal concurrent encoder doesn't guarantee imperative reorder safety even with disjoint buffers. Need graph-then-encode (Option A) or dual command buffer (Option B). Re-measured: hf2q 90.1 tok/s, llama.cpp 103.5 tok/s, gap=13.4 tok/s. |
| Phase 4e revision 4 | Claude | 2026-04-13 | Deep study of llama.cpp's full Metal execution architecture. Replaced Options A/B/C with proper graph IR + fusion + reorder + dual command buffer — the same architecture llama.cpp uses, not a shortcut. 5 sub-phases, 7-9 days estimated. |
| Phase 4e.1-4e.4 | Claude | 2026-04-13 | DONE — mlx-native commit 904d186. Graph IR (capture mode), fusion pass (RmsNorm+MUL→fused kernel), reorder pass (64-node lookahead), dual command buffer encoding. 1,815 lines across 11 files. Queen-reviewed each phase; barrier recomputation bug caught and fixed in 4e.3. |
| Phase 4e.5 | Claude | 2026-04-13 | DONE — hf2q commit 6754603. Integration via `HF2Q_GRAPH_OPT=1 --backend mlx-native`. Correctness verified (identical output tokens). Metal command buffer pool exhaustion bug found and fixed (mlx-native d82a13c). |
| Phase 4e results | Claude | 2026-04-14 | **Honest result: graph optimization provides NO measurable speedup.** 3-run median: baseline 92.2 tok/s, graph-opt 91.8 tok/s (within noise). fusions=0 (model already uses hand-fused kernels), reordered=60/872 (7%), dual-buffer net zero (0.3ms saved by overlap, 0.3ms added by encode overhead). |
| Phase 4e falsified | Claude | 2026-04-14 | 19. Graph-level fusion+reorder+dual-buffer will close the barrier stall gap (FALSIFIED: 0.0 tok/s improvement. The forward pass already has hand-fused kernels and the 7% reorderable nodes don't change the barrier cost profile. The remaining 13 tok/s gap to llama.cpp (91.8 vs 104.9) is NOT in dispatch ordering.) |
| Phase 4e partial | Claude | 2026-04-13 | Router/MLP interleaving: 120 barriers eliminated (606→486), 90.1→93.6 tok/s (+3.9%). Sourdough gate PASS (3095≥3094). New `router_norm_out` buffer avoids WAW on `norm_out`. |
| Status | **Accepted** | Phase 4e complete | **Gap: 13.1 tok/s to llama.cpp (91.8 vs 104.9). Graph optimization infrastructure built but speedup is zero. The remaining gap lives somewhere other than dispatch ordering — needs fresh Phase 0-style investigation.** |
