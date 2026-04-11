# ADR-005 Phase 1b — 1bNEW.29 Pre-Microbench Synthesis

**Date:** 2026-04-11 (PM session)
**Runner:** Claude (CFA parent synthesizer for swarm `swarm-1775949388026-7eii34`)
**Scope:** Synthesize the three pre-spike measurement reports for 1bNEW.29 into a GO/NO-GO decision and a recommended next concrete Walk action. Inputs: NSG sweep microbench (Agent #1), llama.cpp per-kernel timing extraction (Agent #2), kernel citation map (Agent #3).
**Baseline binary:** main HEAD `fb65fd7` / `a377f76` (post-1bNEW.21 + post-1bNEW.22 sticky-encoder revert).
**Hardware:** Apple M5 Max, 128 GB unified memory, `applegpu_g17s`.

---

## Executive summary — three measurements, three meaningful findings

1. **NSG hypothesis is empirically dead.** Both static read (Agent #3) and 1000-iter × 4-run runtime measurement (Agent #1) converge on `N_SIMDGROUP = 2` being optimal for every Q4_0 and Q6_K production dispatch shape on M5 Max. Largest reproducible margin = 1.08% (well below per-shape jitter floor). The 1bNEW.22-v2 "port to per-shape NSG selection" item has a measured envelope of **0 tok/s** and joins sticky-encoder in the empirically-falsified register. Coherence gate not triggered; production binary untouched.

2. **The End gate target value (107 tok/s) is suspect.** Agent #2's fresh 5-run llama.cpp re-measurement on the same M5 Max, same DWQ GGUF, same canonical bench prompt, same `llama-completion` flag set lands at **102.01 tok/s median** — 5 tok/s below the 107 figure the prior spike (`docs/spike-1bNEW22-instrumentation.md`) cited as the peer reference. Sources of the 107 number are unverified; possibilities include a different llama.cpp build, a different thermal state, a different hardware configuration, or an aspirational rather than measured value. **The actual gap is closer to ~17 tok/s (84.9 → 102), not 21.6 tok/s (85.4 → 107).** Under strict Walk discipline (Walk = match peer, not exceed peer), 107 cannot be the End gate if peer's measured speed today is 102.

3. **Coherence is solid; the question is purely speed.** Agent #2 confirmed **byte-identical 16-token greedy generation** between hf2q and llama.cpp at T=0 on the canonical prompt: `The evolution of computing—from mechanical calculators to modern microprocessors—is not merely`. This is strictly stronger evidence than the prior top-1-token-match End gate. Walk-correctness is met; remaining work is purely speed.

**Combined effect on roadmap:** the 1bNEW.22-29 dispatch-count-reduction roadmap from `docs/spike-1bNEW22-instrumentation.md:228-246` is now substantially weakened. Two of its sub-hypotheses are falsified (encoder creation, NSG selection), and a third was independently weakened by Agent #2's discovery that **llama.cpp's ggml graph runs more nodes (~2652) than hf2q's dispatches (2104)**, not fewer — i.e., dispatch-count reduction isn't the lever at all. The remaining live hypothesis is **per-kernel implementation efficiency** (specific candle vs llama.cpp kernel-set deltas) — but per-kernel μs/call measurement is not extractable in a 90-min shell spike on either side.

---

## Per-worker outcomes

### Agent #1 — NSG sweep microbench (`docs/spike-1bNEW29-nsg-sweep-data.md`)

**Verdict: hypothesis EMPIRICALLY FALSIFIED.**

- 8 production dispatch shapes × 4 NSG values {1, 2, 4, 8} × 4 independent full sweeps (3 × 1000-iter + 1 × 2000-iter per cell, per-call commit + waitUntilCompleted on M5 Max).
- Best alt-NSG margin per shape (4-run median basis): all between 0.997× (slower) and 1.011× (faster). Largest = 1.08% on MLP down_proj at nsg=1, vs that shape's ~1.3% inter-run jitter floor — closer to noise than to a real signal.
- Single 21% outlier on `k_proj sliding @ nsg=8` run 1 failed to reproduce on runs 2/3/4; 4-run median shows nsg=8 ~1% **slower** than nsg=2 at that cell.
- **Coherence gate: N/A** (no NSG≠2 produced ≥5% speedup; production binary untouched, sourdough not invoked).
- Methodology deviation from task spec was load-bearing: the spec said "patch `#define N_SIMDGROUP` and rebuild", but Q6_K bakes NSG=2 structurally into `row = 2*r0 + sgitg` at `quantized.metal:5215` and doesn't reference the `#define` at all. Agent #1 correctly added per-NSG kernel variants instead, producing honest data on all 6 Q6_K shapes that the simple `#define` flip would have missed entirely.
- Vendor worktree clean post-revert: `git diff vendor/` empty.
- Cost: ~17 minutes of agent runtime, well under the 90-min budget.

### Agent #2 — llama.cpp per-kernel timings (`docs/spike-1bNEW29-llamacpp-timings.md`)

**Verdict: 1bNEW.22 dispatch-count hypothesis further weakened; per-kernel implementation hypothesis indirectly strengthened by static analysis but not directly measured.**

- **hf2q HEAD baseline confirmed**: 84.9 tok/s clean-vendor, 85.8 as-found — both corroborate the 1bNEW.22 spike's 85.4 median within noise.
- **llama.cpp re-measured at 102.01 tok/s median** (5 cold-mmap runs, byte-identical rendered prompt, same M5 Max, same GGUF, same flag set per ADR-005:998). The 1bNEW.22 spike's 107 figure does not reproduce on this hardware on this date.
- **Coherence baseline**: byte-identical 16-token generation between both tools at T=0 — strictly stronger than top-1 token match.
- **Direct per-kernel timing was unmeasurable in 90 min** on either side. Reasoning: candle-metal-kernels has no built-in per-region timing primitive that doesn't inflate via pool-wide drains (the 1bNEW.22 forced-sync flaw); ggml-metal only supports Xcode `MTLCaptureManager` GUI traces via `GGML_METAL_CAPTURE_COMPUTE`, no shell-extractable per-kernel μs/call. Plumbing `MTLCounterSampleBuffer` instrumentation in both stacks is a ~1-day-each effort. Agent #2 documented the gap honestly rather than fabricating numbers.
- **Static substitution**: extracted llama.cpp's compiled-pipeline set from its `-v` verbose logs and identified **three concrete candle-vs-llama.cpp kernel-set deltas** that have no candle equivalent and plausibly account for ~0.85 ms of the gap:
  - **Q4_0 2-row extended variant** `kernel_mul_mv_ext_q4_0_f32_r1_2_nxpsg=16` — used for MLP gate/up/down on Gemma 4 in current llama.cpp; no candle analogue.
  - **Fused `rms_norm_mul_f32_4` / `rms_norm_mul_add_f32_4`** — combines RmsNorm + scalar mul + optional add in one kernel; partially overlaps with hf2q's existing 1bNEW.4 fused F=2/F=3 RmsNorm but with a different fusion boundary.
  - **Flash Attention `kernel_flash_attn_ext_f16_dk512_dv512_nsg=8`** — global Gemma 4 attention layers compile-time-specialized at head_dim=512; **candle has no equivalent for this head_dim/dtype combo at all**. This is a separately scoped Walk-KERNEL-PORT item not previously in the 1bNEW.22-29 roadmap.
- **Critical sub-hypothesis falsifier**: llama.cpp's ggml graph has **2652 nodes per Gemma 4 26B MoE forward** vs hf2q's **2104 dispatches**. llama.cpp runs MORE nodes, not fewer. The "cut hf2q's dispatch count from 2104 to ~1000 to match llama.cpp" framing at `docs/spike-1bNEW22-instrumentation.md:23` is **wrong** — the 1000 figure was an estimate; the actual measurement contradicts it.

### Agent #3 — Kernel citation map (`docs/spike-1bNEW29-research-notes.md`)

**Verdict: NSG is not the lever (predicted before Agent #1 ran). NR0 is the lever for Q6_K. Two prior spike-doc framings are wrong.**

- **NSG=2 is structural in both trees**: candle and llama.cpp both use `N_SG_Q4_0 = N_SG_Q6_K = 2` (citation: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:11-72` for llama.cpp; `quantized.metal:2307` and `:5215` for candle). Agent #3 predicted Agent #1's null result from static read.
- **Q6_K NR0 asymmetry IS the lever**: llama.cpp templates `nr0 = 2` (2 rows per simdgroup; citation: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:7924-8017` + `ggml-metal-impl.h:44`); candle hardcodes 1 row per simdgroup at `quantized.metal:5215`. **Candle dispatches 2× more Q6_K threadgroups per matmul than llama.cpp.** Net wall-clock effect on M5 Max is occupancy-dependent and requires runtime measurement before any port lands — gain bounded above by Q6_K's share of decode time, bounded below by zero (could even hurt occupancy at the halved tg count on M5 Max's compute-unit budget).
- **Q4_0 has no identified asymmetry** at the threadgroup-geometry level: both trees use `NR0=4, NSG=2` with identical work grids. The Q4_0 lever Agent #2 found is at a *different* layer (the `kernel_mul_mv_ext_*_r1_2_nxpsg=16` extended variant, which is a separate kernel family).
- **Prior framing correction #1**: candle's quantized kernels are **a modified older snapshot of llama.cpp's ggml-metal.m**, not MLX-derived as `docs/spike-1bNEW22-instrumentation.md:309-310` and `docs/spike-post-walk-results.md` both assumed. Evidence: ggml-style in-file references at `quantized.metal:215`, `:241`, `:1724`, `:1829`, `:1911`, `:1959`; byte-identical `block_q_n_dot_y` helper comments; no MLX attribution in this file (MLX-derived content lives in `mlx_gemm.metal`/`mlx_sort.metal`, not `quantized.metal`). Speed gaps to current llama.cpp are best framed as **snapshot drift between two evolving ports**, not novel kernel R&D.
- **Prior framing correction #2**: ADR-005:902 conflated "llama.cpp has function-constant infrastructure (`FC_mul_mv_nsg`)" with "llama.cpp uses function constants for shape-dependent NSG tuning". The actual code at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp:702-879` reads `nsg = N_SG_<TYPE>` unconditionally for all quantized types — no `ne00/ne01` branching. The FC infrastructure exists for cross-hardware specialization, not for per-shape tuning.
- **Min-port estimate**: ~120 LOC for Q6_K-only NR0=2 row-loop port, projected +1.3-2.5 tok/s, **conditional on runtime measurement** confirming M5 Max benefits from halved tg count.
- **Max-port estimate**: ~950 LOC for Q4_0 `FOR_UNROLL`+`ax[NR0]` rewrite + Q6_K NR0=2 + struct-arg passing via `ggml_metal_kargs_mul_mv` analog + function-constant pipeline-compile infrastructure.
- **Top risks for any port**: (1) the `nb01 = 0` call-site contract at `quantized.rs:43-45` would have to change (audit all callers); (2) MoE template instantiation at `quantized.metal:7633` must be updated in lockstep with any signature change; (3) **wall-clock payoff is unverified** — only static evidence exists, runtime measurement is the gating decision.

---

## Updated empirically-falsified register (cumulative across 1bNEW.22 + 1bNEW.29 spikes)

| # | Hypothesis | Falsified by | Cost of falsification |
|---|---|---|---|
| 1 | CPU dispatch overhead is the bottleneck | 1bNEW.22 instrumentation spike (forward_total = 1.48 ms = 12.6% of wall-clock) | ~1 day instrumentation |
| 2 | Pool-tuning has multi-tok/s headroom | 1bNEW.21 sweep (optimum=100 already shipped, larger regresses) | ~2 hours sweep |
| 3 | Single-command-buffer-per-forward is faster | Empirical (5000 dispatches/buffer regressed 85.0 → 79.4) | ~30 min |
| 4 | Encoder creation is the bottleneck | 1bNEW.22 sticky-encoder patch (98.98% reuse, ZERO speedup) | ~3 hours patch + revert |
| 5 | Per-buffer wait semantics (RUN-1 from re-spike) | Dissolved by 1bNEW.20 via different mechanism | N/A |
| 6 | **NEW: Per-shape NSG selection has wall-clock headroom on hf2q's M5 Max shapes** | **Agent #1 NSG sweep + Agent #3 static read** | **~17 min agent + ~10 min static** |
| 7 | **NEW: hf2q's dispatch count (2104) is HIGH compared to llama.cpp (~1000)** | **Agent #2 measured llama.cpp at 2652 nodes — actually higher than hf2q** | **~10 min via -v logs** |
| 8 | **NEW: llama.cpp peer measures 107 tok/s today on this hardware** | **Agent #2 5-run re-measurement = 102.01 tok/s** | **~10 min** |

**Live hypotheses remaining (none fully measured yet):**

| Hypothesis | Status | Next required measurement |
|---|---|---|
| Q6_K NR0=2 row-loop port saves wall-clock on hf2q | Static evidence only (Agent #3) | Per-kernel μs/call on M5 Max via Xcode Instruments OR MTLCounterSampleBuffer plumbing |
| Q4_0 `kernel_mul_mv_ext_*_r1_2_nxpsg=16` extended variant is faster than candle's plain `kernel_mul_mv_q4_0_f32` | Static evidence only (Agent #2) | Standalone Metal port + per-call sync microbench (Q2 blessed path at ADR-005:1000) |
| Fused `rms_norm_mul_*` boundary saves wall-clock vs hf2q's existing F=2/F=3 fusion | Static evidence only (Agent #2) | Static diff vs 1bNEW.4's existing kernel |
| Flash Attention `dk512_dv512_nsg=8` port closes a chunk of the gap | Static evidence only (Agent #2) | Per-kernel timing on global Gemma 4 attention layers |
| End gate value 107 reflects today's reality on this hardware | **Refuted** by Agent #2's 102.01 measurement | None — needs ADR re-baselining decision |

---

## GO/NO-GO on 1bNEW.29 (originally framed as "port llama.cpp's hand-tuned Q-kernels for the dense projection sites")

**Verdict: NO-GO on the originally framed item. CONDITIONAL re-scope below.**

- The originally framed item assumed a single-axis "port llama.cpp's `kernel_mul_mv_q4_0_f32` and `kernel_mul_mv_q6_K_f32`" worked because one specific tuning lever (NSG) was different. **Both halves of that framing are now wrong:** (a) NSG is identical between the trees at the relevant dispatches, (b) the kernels are in fact divergent llama.cpp snapshots, not "MLX-derived vs hand-tuned" as assumed.
- The actual identified levers are **finer-grained**: Q6_K NR0=2 rewrite (Agent #3), Q4_0 `_ext_*_r1_2` extended variant adoption (Agent #2), fused `rms_norm_mul_*` boundary refinement (Agent #2), and a new Flash Attention `dk512_dv512` port (Agent #2). Each is its own item with its own envelope, risk, and runtime-measurement prerequisite.
- **None of the four levers above can be GO without per-kernel μs/call measurement on M5 Max.** The 1bNEW.22 falsification taught that "static evidence + LOC estimate" is not sufficient justification for vendor surgery — the encoder-creation hypothesis had stronger static evidence than any of these and still produced zero speedup.

---

## Recommended next concrete Walk action (decision needed from user)

The data converges on one of the following four paths, in order of cheapest-to-act:

### Option A — Re-baseline the End gate to 102 tok/s, mark Walk-speed as MET-or-very-close

**Rationale:** Walk discipline = match peer. Peer (llama.cpp) measures 102.01 today on this exact hardware. hf2q is at 84.9. The gap to peer is 17.1 tok/s, not 21.6. Under strict Walk discipline, the End gate should be the peer's measured speed on the same day, not a historical figure.

**Cost:** ~30 min — update ADR-005:836 (`≥107 tok/s` → `≥102 tok/s on this hardware on this date, re-measured 2026-04-11 by cfa swarm swarm-1775949388026-7eii34`); document the source-of-107 investigation; reset End-gate progress framing.

**Risk:** If 107 is recoverable (e.g., a faster llama.cpp build exists), the re-baseline understates the bar. Mitigation: document the re-baseline reasoning explicitly so it's reversible if a 107-tok/s llama.cpp build is found.

**This option does NOT close the remaining 17 tok/s gap.** It re-frames it. The remaining gap still requires Option B/C/D.

### Option B — Plumb per-kernel timing in both stacks, then decide

**Rationale:** Both Agent #2 and Agent #3 say the same thing — without per-kernel μs/call data, all kernel-port decisions are speculation. The decisive measurement is per-kernel time on M5 Max for the specific shapes hf2q dispatches.

**Cost:** ~1 day each side. candle side: vendor-patch `MTLCounterSampleBuffer` into `vendor/candle-metal-kernels/src/metal/encoder.rs` with timestamp pre/post each dispatch. llama.cpp side: either patch `ggml-metal-ops.cpp` similarly OR run via Xcode Instruments → Metal System Trace (GUI-only, manual, but no plumbing required).

**Risk:** If the per-kernel data shows candle's kernels are *not* materially slower than llama.cpp's at hf2q's shapes (which is plausible given Agent #2's 5-tok/s discrepancy on the llama.cpp side itself), the entire kernel-port roadmap dissolves and we're left with only Option A.

**Reward:** Resolves four live hypotheses simultaneously. The single highest-information-density measurement available.

### Option C — Land the cheapest concrete Walk port without runtime measurement and gate on coherence + speed delta

**Rationale:** The Q6_K NR0=2 row-loop port (Agent #3 min-port estimate ~120 LOC) is small enough that the cost of building + measuring + reverting is comparable to Option B's plumbing cost, while producing a definitive speedup-or-not signal directly. "Just try it" is sometimes cheaper than instrumentation.

**Cost:** ~1 day implementation + ~2 hours validation (sourdough gate + 5-run bench + revert if no improvement). Walk-faithful port: Agent #3 has the citation trail for the byte-for-byte llama.cpp source.

**Risk:** Repeats the 1bNEW.22 mantra violation if there's no microbench first. Mitigation: write a 50-line standalone microbench (`call_quantized_matmul_mv_t` vs ported NR0=2 kernel) FIRST, ~30 min, decide GO based on its result. If the microbench shows ≥10% per-call improvement, then build the full port. This restores the mantra discipline.

**Reward:** Direct empirical answer to the highest-confidence remaining lever.

### Option D — Accept that Walk has no further identified +N tok/s lever and pivot to Run

**Rationale:** Under strict Walk discipline, every remaining hypothesis is speculative, every previously-measured candidate has been falsified, and the End gate value itself is suspect. The discipline-honest move is to declare Walk done at the current state (84.9 tok/s, byte-identical 16-token generation, all correctness gates met) and open Run scope where novel optimization beyond peer is allowed.

**Cost:** ~1 hour — update ADR-005, declare Walk complete, file the Run-scope queue with the four currently-unmeasured levers as the Run-1 candidate set.

**Risk:** Closes Walk on a potentially-recoverable speed gap. Mitigation: include a "re-open trigger" — if any of the four levers above gets per-kernel measurement evidence that justifies it, Walk can re-open for that specific item.

**Reward:** Honest framing. Stops grinding on hypothesis after hypothesis without measurement infrastructure.

---

## Recommended sequencing (synthesizer's pick, not binding)

**A → C-with-microbench-first → revisit B if C lands a win.**

1. **Option A first (~30 min)**: re-baseline the End gate to whatever llama.cpp actually measures on the day, document the 107-vs-102 investigation, reset the framing. This is honest accounting independent of any other decision.
2. **Option C with the mantra discipline restored (~1.5 days total)**: write a 50-line standalone microbench for the Q6_K NR0=2 hypothesis, decide GO based on its result, then build + validate the full port if GO. Lowest-cost path to definitive evidence on the highest-confidence remaining lever.
3. **Option B if C lands a win (~2 days)**: if Q6_K NR0=2 produces measurable speedup, plumb per-kernel timing infrastructure to systematically evaluate the other three levers (Q4_0 ext variant, fused rms_norm_mul boundary, FA dk512 port).
4. **Option D if B yields nothing**: accept Walk complete, open Run.

This sequence respects the mantra discipline (every step gates on cheap measurement first), respects the coherence-over-speed priority (every port runs sourdough gate), and produces incremental value at each step regardless of where the chain terminates.

**The user's call.** The synthesizer recommends but does not decide — re-baselining the End gate (Option A) is a project-level decision that needs explicit user sign-off.

---

## Honest contamination disclosure

The three workers ran in parallel during this swarm, and the user correctly flagged that Agent #1 (vendor patches + cargo build) and Agent #2 (cargo build of hf2q which compiles the same vendored crate) were not actually independent at the build-system level — they share `target/`, the vendored source files, and build artifacts. **Agent #2 explicitly observed Agent #1's mid-sweep 308-line NSG-variant patch in `quantized.metal` and reported it appearing and disappearing during its own measurement runs.**

This is a real interference, not a theoretical one. Two mitigations protected the data:

1. **Agent #1's 4-run median methodology absorbed the contamination.** The single dramatic outlier on run 1 (`k_proj sliding @ nsg=8` at 1.21× over nsg=2) is exactly the kind of artifact that build-state interference would produce. Runs 2/3/4 did not reproduce it; the 4-run median dismissed it as a transient. **The negative-result verdict survives the contamination because it depended on absence-of-signal across multiple runs, not on a specific cell value.**
2. **Agent #2's hf2q baseline was measured twice** — once with as-found vendor state (85.8 tok/s) and once after `git checkout HEAD -- vendor/` to clean it (84.9 tok/s). Both numbers corroborate the 1bNEW.22 spike's 85.4 median within noise.

**The contamination did not bias either verdict.** That is luck-plus-redundancy, not good design. The lesson is documented in user feedback memory: "swarm workers that share a build dir are not parallel-safe even if their source-file claims are disjoint". Future spikes should run build-touching workers sequentially.

---

## Citation trail

- `docs/spike-1bNEW29-nsg-sweep-data.md` — Agent #1 NSG sweep (333 lines, raw data, methodology, per-shape verdict)
- `docs/spike-1bNEW29-llamacpp-timings.md` — Agent #2 llama.cpp re-measurement + kernel-set delta inventory + 102-not-107 finding (473 lines)
- `docs/spike-1bNEW29-research-notes.md` — Agent #3 candle-vs-llama.cpp citation map + NR0 finding + lineage correction (312 lines)
- `docs/spike-1bNEW22-instrumentation.md` — Prior spike that proposed the 1bNEW.22-29 roadmap (now half-falsified by this synthesis)
- `vendor/candle-metal-kernels/examples/metal_benchmarks.rs` — NSG sweep harness (added by Agent #1, currently gitignore-shadowed; see followup #1)
- `swarm-1775949388026-7eii34` — cfa swarm coordination ID

---

## Followups for the user (decisions or one-line ack needed)

1. **End gate re-baseline (Option A above)** — accept, reject, or modify? Synthesizer recommendation: accept, document the source-of-107 investigation, mark the reversibility condition.
2. **Next concrete Walk action** — A / B / C / D from the recommendation section above? Synthesizer recommendation: A → C-with-microbench-first.
3. **Microbench harness commit decision** — Agent #1's harness at `vendor/candle-metal-kernels/examples/metal_benchmarks.rs` is gitignore-shadowed by the global `examples/` rule at `.gitignore:20`. Synthesizer is committing it with a `!vendor/candle-metal-kernels/examples/` exception so it's available for future re-runs (different hardware, different shapes). The 308-line NSG variant kernels are NOT being committed (they were dead code added only for the sweep; the production kernel binary stays at HEAD).
4. **Source-of-107 investigation** — should the synthesizer or a follow-up swarm dig into git log on llama.cpp / look for an older binary that hits 107? Or accept that 107 was a measurement artifact and re-baseline?

---

## Worktree status

```
git status (post-synthesis, pre-commit):
?? docs/spike-1bNEW29-llamacpp-timings.md
?? docs/spike-1bNEW29-nsg-sweep-data.md
?? docs/spike-1bNEW29-research-notes.md
?? docs/spike-1bNEW29-pre-microbench-results.md  ← this file
?? vendor/candle-metal-kernels/examples/metal_benchmarks.rs  ← gitignore-shadowed, see followup #3
   docs/ADR-005-inference-server.md (about to be edited with Next Walk session update)
   .gitignore (about to add !vendor/candle-metal-kernels/examples/ exception)
```

`git diff src/` empty. `git diff vendor/` empty (both worker temporary patches reverted by Agents #1 and #2).
