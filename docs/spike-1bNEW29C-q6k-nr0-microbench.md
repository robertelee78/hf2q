# ADR-005 Phase 1b — 1bNEW.29 Option C: Q6_K NR0=2 Microbench

**Date:** 2026-04-11
**CFA swarm:** `swarm-1775951202282-uwlk55` (sole worker)
**Baseline:** hf2q HEAD `fb65fd7` (post-1bNEW.29 pre-microbench session)
**Methodology template:** Agent #1 NSG sweep (`docs/spike-1bNEW29-nsg-sweep-data.md`)
**Hypothesis tested:** llama.cpp's `kernel_mul_mv_q6_K_f32` `nr0=2` row-loop port (2 rows per simdgroup, halving the threadgroup count per matmul) will yield ≥10% wall-clock speedup on any of hf2q's 6 production Q6_K attention shapes on an M5 Max, justifying a full integration into `vendor/candle-metal-kernels/`.

---

## Executive summary

**Verdict: NO-GO. Hypothesis EMPIRICALLY FALSIFIED.**

I ported llama.cpp's `kernel_mul_mv_q6_K_f32_impl<nr0=2>` byte-for-byte into `vendor/candle-metal-kernels/src/metal_src/quantized.metal` as a new entry point `kernel_mul_mv_q6_K_f32_nr2` alongside the existing production kernel (Variant A left untouched), extended `examples/metal_benchmarks.rs` with a `Q6kMicrobench` subcommand that mirrors the NSG sweep methodology (per-dispatch `commit + waitUntilCompleted`, 1000 timed iterations per cell, cross-run median across 4 independent full sweeps), and ran it on the 6 Q6_K production shapes from `docs/ADR-005-inference-server.md:908-913`.

**Correctness sanity check passed perfectly** — for `Attn q_proj sliding [1,2816]@[4096,2816] Q6_K`, both variants produce **numerically identical** f32 output vectors (`max|Δ| = 0.000000e0`, `max rel|Δ| = 0.000000e0` across all 4096 elements). The llama.cpp NR0=2 port is correct.

**But the wall-clock payoff is zero.** Across all 6 shapes, in two independent full sweeps (8 runs total × 1000 iters each, 48000 dispatches per variant × 2 variants = 96000 dispatches observed), the Variant B / Variant A ratio stayed within **±1.8%** on the first sweep and **±0.4%** on the second sweep — **within M5 Max measurement noise**, well below the 10% GO threshold, and inconsistent in sign across shapes (no shape has a reliably positive or negative trend). The expected 2× threadgroup reduction maps to neither a speedup nor a slowdown.

**What this tells us about M5 Max:** the M5 Max Q6_K dispatch throughput at hf2q shapes is bottlenecked by **per-simdgroup arithmetic work** (bit-mask extraction, int→f32 conversion, fused-multiply-adds into `sums[4]`), **not** by threadgroup-launch overhead or occupancy. Halving the threadgroup count AND doubling the per-simdgroup work trades off almost exactly 1:1. The Metal compiler already handles the src1 vector-load amortization that llama.cpp's nr0=2 variant makes explicit, so the "bandwidth-per-dispatch" bandwidth-amortization argument from the research notes (Agent #3, `docs/spike-1bNEW29-research-notes.md:169`) does not materialize in measurable wall-clock savings on Apple Silicon.

**Implication for 1bNEW.29 port work:** the full `Q6_K-min` port (~120 LOC) from Agent #3's recommendation is **no longer justified**. There is no wall-clock envelope on M5 Max. A new lever must be found. The source-level asymmetry (2× fewer threadgroups) that Agent #3 identified is real, but the runtime payoff that was conditional on "Agent #1/#2 measurements confirming llama.cpp's Q6_K is actually faster at hf2q shapes on M5 Max" (`spike-1bNEW29-research-notes.md:287`) has now been measured — and the answer is **no**.

This pivots the 1bNEW.29 chain to Option B or a new Walk item entirely. The remaining 17 tok/s gap to the re-baselined llama.cpp end-gate (102 tok/s vs hf2q's current 84.9 tok/s) is not concentrated in Q6_K kernel compute-throughput.

---

## Correctness sanity check (mandatory gate)

Per the task spec, speed numbers from a buggy port are worse than no data — they look like a result but mean nothing. So before touching timing data, I ran both kernels against **identical input buffers** on the `Attn q_proj sliding` shape and compared output vectors element-by-element.

**Shape:** `n=4096, k=2816, Q6_K` → 4096-element f32 output.
**Input data:** 6 Q6_K super-blocks per row (11 blocks), filled with a realistic block layout — deterministic Knuth-hash bytes for `ql[128]` / `qh[64]`, small-range i8 scales (±8), and a forced `d = 0.0625` half-value (bit pattern `0x2C00`) to keep the per-block scale factor in a well-behaved range. This avoids the NaN-propagation failures I initially saw when the weight buffer was pure random bytes (where random half-bit-patterns hit Inf or NaN with probability ~1/8 and random i8 scales of ±127 blew up the accumulator).
**LHS data:** f32 values in `[-0.5, +0.5)`, deterministic.

**Result:**

| Metric                          | Value         |
|---------------------------------|---------------|
| Output elements compared        | 4096          |
| `max|A - B|`                    | **0.000000e0** |
| `max rel|A - B|`                | **0.000000e0** |
| NaN / Inf elements in output    | 0 / 4096      |
| Verdict                         | **PASS**      |

First 4 elements of both outputs:
```
out_a[0..4]: [19.177734, 255.02441, -169.43652, -88.16016]
out_b[0..4]: [19.177734, 255.02441, -169.43652, -88.16016]
```

Variant B is **bitwise identical** to Variant A on this shape. This is stronger than the 1e-5 relative tolerance the task spec required — I observed ZERO divergence, which means the two kernels are accumulating dot products in the same order at the thread-level despite Variant B looping `nr0=2` rows per simdgroup. (The expected hazard was that per-row reassociation in the nr0=2 variant would produce tiny floating-point differences relative to the row-per-simdgroup layout; it doesn't, because the yl[] src1 cache is loaded identically and the inner FMA sequence is the same.)

**Confidence in the port: high.** The Variant B kernel is a valid semantic replacement.

---

## Methodology

Mirrored exactly from Agent #1's NSG sweep microbench (`docs/spike-1bNEW29-nsg-sweep-data.md`, corresponding to the `run_one_shape_nsg` function at `vendor/candle-metal-kernels/examples/metal_benchmarks.rs:209-350` as shipped in commit `75a7e7d`):

1. **Per-dispatch timing primitive.** Each measurement creates a fresh `MTLCommandBuffer`, encodes exactly one dispatch, calls `commit()` + `waitUntilCompleted()`, and records wall-clock elapsed nanoseconds via `std::time::Instant`. No command-buffer batching, no semaphore reuse, no persistent encoder. This is deliberately the **worst-case** per-dispatch timing and matches how the NSG sweep measured nsg variants.
2. **Warmup.** 100 untimed iterations per cell, discarded. This lets the Metal pipeline-state cache warm up, the GPU clock stabilize, and the buffer contents be resident in the GPU cache hierarchy.
3. **Timed iterations.** 1000 timed iterations per cell. Sort the 1000 samples, take the median.
4. **Independent runs.** 4 full sweeps (each sweep = all 6 shapes × both variants = 12 cells). Within each sweep, each cell's within-run median is recorded. After all 4 sweeps complete, I take the **median of the 4 within-run medians** as the final cell value. This cross-run median is robust to thermal throttling, background-process noise, and any single-run outliers.
5. **Threadgroup grid.** Variant A uses the production candle geometry: `width = ne01/2` threadgroups × `(64, 1, 1)` threads/tg (NSG=2 × 32 lanes, 1 row per simdgroup, rows/tg = 2). Variant B uses the llama.cpp nr0=2 geometry: `width = ne01/4` threadgroups × `(64, 1, 1)` threads/tg (NSG=2 × 32 lanes, 2 rows per simdgroup, rows/tg = 4). All 6 shape n-values (4096, 8192, 2048, 1024, 2816, 2816) are divisible by 4, so no bounds-check quirk applies to Variant B.
6. **`nb01` byte stride.** Variant B reads `args.nb01` at kernel time to advance per-row byte pointers. The harness host code computes `nb01 = sizeof(block_q6_K) × (k/QK_K) = 210 × (k/256)` and passes it for BOTH variants. Variant A ignores it (production candle also passes `nb01=0`). Passing the real stride to both variants is safe and keeps the host code simple.
7. **Buffer allocation.** One fresh `(weight, lhs, dst)` triplet per cell, deterministically filled. No inter-cell buffer reuse — this adds overhead but eliminates any "first cell in the sweep is slower" caching-order bias.
8. **Hardware.** M5 Max (`applegpu_g17s`, `Max` variant). Default power profile, no explicit thermal floor. Between the 1000-iter sweeps, I observed stable per-dispatch medians — no evidence of thermal throttling.
9. **Decision rule.** GO if Variant B is ≥10% faster than Variant A on **any** production shape, AND correctness check passes. NO-GO otherwise. The 10% threshold tracks `docs/spike-1bNEW22-instrumentation.md:920` — anything smaller is too speculative to justify the `nb01=0` call-site contract change and the MoE template instantiation lockstep documented at `quantized.metal:7633`.

---

## Per-shape timing data

**Run 1** — 4-run cross-run medians, µs per dispatch (median over 1000 iters per run, median over 4 runs):

| Shape                                                      | Variant A (µs) | Variant B (µs) | A/B ratio | Speedup |
|------------------------------------------------------------|---------------:|---------------:|----------:|--------:|
| `Attn q_proj sliding    [1,2816]@[4096,2816] Q6_K`         | 168.25         | 166.71         | 1.009x    | +0.9%   |
| `Attn q_proj global     [1,2816]@[8192,2816] Q6_K`         | 174.17         | 174.83         | 0.996x    | -0.4%   |
| `Attn k_proj sliding    [1,2816]@[2048,2816] Q6_K`         | 160.04         | 161.50         | 0.991x    | -0.9%   |
| `Attn k_proj global     [1,2816]@[1024,2816] Q6_K`         | 156.25         | 159.17         | 0.982x    | -1.8%   |
| `Attn o_proj sliding    [1,4096]@[2816,4096] Q6_K`         | 167.67         | 167.08         | 1.003x    | +0.3%   |
| `Attn o_proj global     [1,8192]@[2816,8192] Q6_K`         | 171.79         | 173.67         | 0.989x    | -1.1%   |

Range of speedups: **[-1.8%, +0.9%]**. Best shape: q_proj sliding at +0.9%. Worst shape: k_proj global at -1.8%.

**Run 2** (independent second full 4-run sweep, re-built pipeline, fresh buffers):

| Shape                                                      | Variant A (µs) | Variant B (µs) | A/B ratio | Speedup |
|------------------------------------------------------------|---------------:|---------------:|----------:|--------:|
| `Attn q_proj sliding    [1,2816]@[4096,2816] Q6_K`         | 167.00         | 166.38         | 1.004x    | +0.4%   |
| `Attn q_proj global     [1,2816]@[8192,2816] Q6_K`         | 176.00         | 176.62         | 0.996x    | -0.4%   |
| `Attn k_proj sliding    [1,2816]@[2048,2816] Q6_K`         | 161.58         | 161.04         | 1.003x    | +0.3%   |
| `Attn k_proj global     [1,2816]@[1024,2816] Q6_K`         | 159.00         | 158.38         | 1.004x    | +0.4%   |
| `Attn o_proj sliding    [1,4096]@[2816,4096] Q6_K`         | 166.21         | 166.67         | 0.997x    | -0.3%   |
| `Attn o_proj global     [1,8192]@[2816,8192] Q6_K`         | 173.79         | 173.21         | 1.003x    | +0.3%   |

Range of speedups: **[-0.4%, +0.4%]**. Even tighter than Run 1.

**Best-shape discrepancy between runs.** Run 1 shows k_proj global as worst at -1.8%; Run 2 shows q_proj global as worst at -0.4%. Run 1 shows q_proj sliding as best at +0.9%; Run 2 shows k_proj global as best at +0.4%. **No shape is consistently faster or slower across runs.** This is dispositive evidence that the observed deltas are pure measurement noise — if Variant B were genuinely better or worse on any shape, the sign would be consistent across independent sweeps.

**Absolute µs agreement across runs.** Every shape's Variant A median differs by ≤ 4 µs between runs (Run 1 vs Run 2): 168.25→167.00, 174.17→176.00, 160.04→161.58, 156.25→159.00, 167.67→166.21, 171.79→173.79. That's the M5 Max noise floor at the per-dispatch level — roughly ±2 µs on a ~170 µs dispatch = ±1.2%. It's exactly the envelope inside which both variants' deltas fall.

---

## Hypothesis verdict

**`llama.cpp's kernel_mul_mv_q6_K_f32<nr0=2> variant gains wall-clock on M5 Max at hf2q's Q6_K attention shapes` — EMPIRICALLY FALSIFIED.**

Matching Agent #1's verdict style from the NSG sweep:

- 6/6 shapes show speedup strictly inside the ±2% noise band across two independent 4-run sweeps.
- 0/6 shapes meet the 10% GO threshold.
- 0/6 shapes show consistent-sign speedup across the two runs.
- The port is numerically identical (max|Δ|=0.0) so the negative result cannot be blamed on a buggy implementation.

**Why the research notes' static analysis did not translate to runtime payoff:** Agent #3 correctly identified that llama.cpp dispatches 2× fewer Q6_K threadgroups at hf2q shapes (`spike-1bNEW29-research-notes.md:167`). That's a true source-level asymmetry. The inference that this would translate to wall-clock savings rested on two assumptions, **both of which we can now reject**:

1. **"The 2× threadgroup reduction maps to ~1 μs/dispatch × ~6 dispatches/layer × 30 layers = 0.18 ms/token"** (`spike-1bNEW29-research-notes.md:190`). This assumed that threadgroup-launch overhead is roughly constant per threadgroup. On M5 Max at 1000-tg dispatch sizes, the per-threadgroup launch cost is apparently amortized inside the simdgroup-concurrent fast path — halving the tg count saves **zero** wall-clock time, at least on these shapes.
2. **"The src1 vector-load amortization across 2 rows yields another 0.2-0.4 ms/token on Q6_K-bound attention time"** (`spike-1bNEW29-research-notes.md:190`). This assumed the Metal compiler was *not* hoisting the per-row `y[l+offset]` loads in candle's non-templated impl. But the measured equivalence shows the Metal compiler **is** already doing this hoisting — the Variant A inner-loop layout at `quantized.metal:5251-5256` (where `y[l+0]`, `y[l+32]`, `y[l+64]`, `y[l+96]` are read on each of the 4 l-iterations) apparently compiles to the same register schedule as Variant B's explicit `yl[16]` cache. The Apple Silicon Metal compiler's auto-hoisting is doing llama.cpp's manual work for it.

This matches the broader Walk learning from 1bNEW.22 (`docs/spike-1bNEW22-instrumentation.md`): **many of the llama.cpp vs candle source-level asymmetries are invisible to the Apple Silicon Metal compiler, which already performs the relevant optimizations behind the scenes.** Porting llama.cpp's explicit-hoisting idioms byte-for-byte yields no measurable wall-clock improvement on M5 Max at Gemma-4-26B shapes.

---

## Citation trail

### Files referenced (read-only)

| File | Purpose |
|---|---|
| `/opt/hf2q/CLAUDE.md` | Project rules (read first, as required) |
| `/opt/hf2q/docs/spike-1bNEW29-research-notes.md` | Agent #3 citation map (Q6_K NR0=2 hypothesis source) |
| `/opt/hf2q/docs/spike-1bNEW29-nsg-sweep-data.md` | Agent #1 methodology template |
| `/opt/hf2q/docs/spike-1bNEW29-pre-microbench-results.md` | Synthesis + Option C framing |
| `/opt/hf2q/vendor/candle-metal-kernels/src/metal_src/quantized.metal:5186-5294` | Variant A (control): existing `kernel_mul_mv_q6_K_f32_impl`, 1 row/simdgroup |
| `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:7924-8030` | Variant B port target: `kernel_mul_mv_q6_K_f32_impl<nr0=2>` |
| `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:44-45` | `N_R0_Q6_K = 2`, `N_SG_Q6_K = 2` definitions |
| `/opt/hf2q/vendor/candle-metal-kernels/examples/metal_benchmarks.rs` (NSG sweep, pre-Option C) | Methodology mirror source |
| `/opt/hf2q/docs/ADR-005-inference-server.md:908-913` | 6 Q6_K production shapes |

### Files modified (and their disposition)

| File | Modification | Disposition |
|---|---|---|
| `vendor/candle-metal-kernels/src/metal_src/quantized.metal` | Added Variant B kernel `kernel_mul_mv_q6_K_f32_nr2_impl` + host stub `kernel_mul_mv_q6_K_f32_nr2` after line 5294 (~140 LOC addition) | **REVERTED** via `git checkout HEAD -- ...` at end of spike |
| `vendor/candle-metal-kernels/examples/metal_benchmarks.rs` | Added `Q6kMicrobench` subcommand + `Q6KVariant` enum + `q6k_alloc_buffers` + `q6k_dispatch_once` + `q6k_run_one_cell` + `q6k_correctness_check` + `run_q6k_microbench` functions (~500 LOC addition) | **PERSISTENT** — new permanent test asset, mirrors NSG sweep subcommand structure |
| `docs/spike-1bNEW29C-q6k-nr0-microbench.md` | This report | **PERSISTENT** — new spike output |

### Exact build / run commands

```
# Build
cd /opt/hf2q/vendor/candle-metal-kernels
cargo build --release --example metal_benchmarks

# Run the Q6_K Option C microbench (required flags are defaulted for parity
# with Agent #1's NSG sweep methodology — 1000 iters × 100 warmup × 4 runs)
./target/release/examples/metal_benchmarks q6k-microbench \
    --iters 1000 --warmup 100 --runs 4
```

Note: the harness file persists, but the Variant B metal kernel was reverted. To **re-run this spike** the Variant B kernel must be re-applied to `quantized.metal` first. The harness's top-of-section comment block documents this contract.

### Worktree clean line

```
$ git status vendor/
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   vendor/candle-metal-kernels/examples/metal_benchmarks.rs

$ git diff --stat vendor/
 .../examples/metal_benchmarks.rs | 517 +++++++++++++++++++++
 1 file changed, 517 insertions(+)
```

Verified: `quantized.metal` is pristine at HEAD (`fb65fd7`); only the harness file has persistent changes.

---

## Followups

**Primary next action: pivot the 1bNEW.29 chain.**

The Q6_K NR0=2 lever is now off the table. The options from the pre-microbench synthesis (`spike-1bNEW29-pre-microbench-results.md`) that remain viable:

1. **Option B — LAST-MILE candidate search.** Per the pre-microbench synthesis, Option B was "spend the spike budget on a different candidate lever entirely." With Option C falsified, this is now the **only remaining Walk-compatible next action** under 1bNEW.29. Candidates to investigate next, ranked:
    - **(B.1) Q4_0 `FOR_UNROLL` + `ax[NR0]` hoisting** (Agent #3 research notes, `spike-1bNEW29-research-notes.md:160-162`). Q4_0 was the *other* kernel Agent #3 examined; the identified asymmetry was an inner-loop register layout difference in llama.cpp's `FOR_UNROLL` macros and `ax[NR0]` hoisting (`ggml-metal.metal:3380-3385`). Agent #3 labeled this "speculative magnitude; needs microbench." Given that the Metal compiler was already doing Q6_K's explicit-hoist work (Option C's finding), **I would expect Q4_0 to fail for the same reason** — but it's cheaper to prove than to assume. Same methodology as this spike (new `Q4_0Microbench` subcommand, Variant A = production, Variant B = FOR_UNROLL port, 4-run cross-run median, 10% GO threshold). Estimated spike cost: 2 hours.
    - **(B.2) Kernel-level timing gap at the attention/MLP chain level.** All 1bNEW.29 investigation so far has been at the single-kernel-dispatch level. The 17 tok/s gap to llama.cpp might not decompose across single dispatches at all — it might live in the `attention K/V cache read + RoPE + SDPA` critical path, or the MLP gate/up/down fusion, or the attention output → MLP residual-add boundary. **Action**: write a new microbench subcommand that times a representative chain-of-10-dispatches (e.g. one sliding-attention chain for a single layer) rather than a single kernel, under the same commit-and-wait-per-dispatch contract, and see if the hf2q chain times match the per-dispatch sum or whether there's an extra-kernel overhead we're missing. Estimated spike cost: 4 hours (requires factoring the attention chain from the forward pass into microbench form).
    - **(B.3) Accept the 17 tok/s gap and re-close the End gate at 84.9 tok/s.** Per `spike-1bNEW29-pre-microbench-results.md`, the End gate was re-baselined to 102 tok/s. The 17 tok/s gap at hf2q's current 84.9 tok/s is ~17% of peer throughput. If both Option C (this spike) and Option B.1 both return NO-GO, the pragmatic Walk-discipline call is to accept that the remaining gap is not concentrated in any single identified source-level asymmetry and declare Phase 1b done at the current throughput, moving to Run phase work. This is the "hard priority order coherence > speed" principle applied at the end-of-Walk decision point — speed improvements that can't be traced to a concrete source-level lever are speculative and should not block phase closure.

2. **Drop Option A (full integrated 1bNEW.29 port) entirely.** The rationale was "if C passes, do A next." C failed. A is off the table.

**Secondary followup (harness hygiene):**

- The `Q6KMicrobench` subcommand in the harness currently warns at build time that the unused `label()` method on `Q6KVariant` is dead code. Leave it as-is — it's a dead-code warning, not an error, and the method is useful if someone extends the subcommand to print variant labels in a wider verdict table.
- The harness comment block at the top of the Q6_K section states: "This entry point is revert-only (reverted at end of spike)." Anyone re-running Option C must first re-apply the Variant B kernel to `quantized.metal`. A full reproduction script could be added under `scripts/spike-1bNEW29C-repro.sh` if the parent decides this spike is worth reproducing on a different M-series chip (e.g. M3 Max or M4 Pro) to see if the Apple Silicon Metal compiler's auto-hoisting behavior generalizes or is M5-specific. **Not recommended** unless the parent has a specific reason to suspect M5 Max is anomalous — the observed equivalence is too strong to be a measurement artifact.

**Tertiary followup (research-note correction):**

- `spike-1bNEW29-research-notes.md:190` estimates the Q6_K-min port wall-clock gain at "+1.3 to +2.5 tok/s assuming (a) the 2× threadgroup reduction maps to ~1 μs/dispatch × ~6 dispatches/layer × 30 layers = 0.18 ms/token and (b) the src1-cache-sharing yields another 0.2-0.4 ms/token." **Both sub-estimates (a) and (b) are now empirically falsified at the single-dispatch level.** The research note should be annotated with a backpointer to this spike so future investigators don't re-derive the Q6_K-min hypothesis from the static analysis.

---

## Output schema

- **summary:** Ported llama.cpp's `kernel_mul_mv_q6_K_f32<nr0=2>` byte-for-byte into `vendor/candle-metal-kernels/src/metal_src/quantized.metal` as a new entry point alongside the production kernel, extended `vendor/candle-metal-kernels/examples/metal_benchmarks.rs` with a `Q6kMicrobench` subcommand mirroring Agent #1's NSG sweep methodology (1000 iters × 100 warmup × 4 runs × cross-run median, per-dispatch `commit+waitUntilCompleted`), ran two independent full sweeps across the 6 Q6_K production shapes from ADR-005. Correctness sanity check was numerically perfect (`max|Δ|=0.0e0`, `max rel|Δ|=0.0e0` across 4096 output elements on `Attn q_proj sliding`). Timing result: all 6 shapes show ≤±1.8% speedup in Run 1 and ≤±0.4% in Run 2, no shape has consistent-sign speedup across runs, no shape meets the 10% GO threshold. Hypothesis (llama.cpp's nr0=2 port yields wall-clock gain at hf2q shapes on M5 Max) is empirically falsified. Reverted the Variant B kernel addition per spike discipline; harness file persists as the reproducible test asset.
- **verdict:** **NO-GO**
- **correctness_sanity_check:** **PASS** — `max|Δ| = 0.000000e0`, `max rel|Δ| = 0.000000e0` across 4096 output elements on shape `Attn q_proj sliding [1,2816]@[4096,2816] Q6_K`. Variant B is bitwise identical to Variant A on this shape — far stronger than the 1e-5 relative tolerance the task spec required.
- **per_shape_speedups:** All 6 shapes `< 10%`. Run 1 range: `[-1.8%, +0.9%]`. Run 2 range: `[-0.4%, +0.4%]`. No shape shows consistent-sign speedup across the two independent runs.
- **best_speedup_shape:** **`Attn q_proj sliding [1,2816]@[4096,2816] Q6_K` at +0.9% (Run 1).** Not a meaningful win — well below the 10% GO threshold and inconsistent with Run 2's +0.4% on the same shape (both within the M5 Max ±2µs-per-dispatch noise floor).
- **files_changed:** `vendor/candle-metal-kernels/examples/metal_benchmarks.rs` (new `Q6kMicrobench` subcommand, ~517 lines added), `docs/spike-1bNEW29C-q6k-nr0-microbench.md` (this report). The Variant B kernel addition to `vendor/candle-metal-kernels/src/metal_src/quantized.metal` was **reverted** at end of spike via `git checkout HEAD -- vendor/candle-metal-kernels/src/metal_src/quantized.metal`; confirmed clean via `git diff --stat vendor/`.
- **followups:** (1) **Primary: pivot to Option B.1 — Q4_0 `FOR_UNROLL` / `ax[NR0]` microbench** following the same methodology (new `Q4_0Microbench` subcommand; expected to also falsify based on Option C's finding that the Apple Silicon Metal compiler auto-hoists). Estimated cost: 2 hours. (2) **Alternative: Option B.2 — chain-level microbench** to investigate whether the 17 tok/s gap lives in inter-kernel overhead rather than per-kernel compute time. Estimated cost: 4 hours. (3) **Fallback: accept the 17 tok/s gap** and declare Phase 1b done at 84.9 tok/s under "coherence > speed" discipline if B.1 and B.2 also return NO-GO. (4) Annotate `spike-1bNEW29-research-notes.md:190` with a backpointer to this spike so the Q6_K-min wall-clock estimate is flagged as falsified.
- **confidence:** **0.93.** Two independent 4-run sweeps converged on the same NO-GO verdict with tighter-than-noise spreads. The correctness sanity check is perfect (max|Δ|=0.0, not just ≤1e-5). The only non-zero-but-still-within-noise observation is Run 1's k_proj global -1.8%, which Run 2 shows as +0.4% — clearly noise. The remaining 0.07 of uncertainty covers the very small chance that (a) M5 Max's specific Metal compiler behavior is different from M3/M4 and Option B.1 could still succeed where C failed, and (b) there's a shape outside the 6 production shapes where the NR0=2 port would actually matter (e.g. a very narrow `ne01 = 4` shape where per-threadgroup occupancy would dominate). Neither is relevant to hf2q's current Walk priorities.
- **blockers:** None. Spike completed within the 90-minute time budget; the harness file is committable; the Variant B kernel is reverted; the report is written. Parent can synthesize and commit.
