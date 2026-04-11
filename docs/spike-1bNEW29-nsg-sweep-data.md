# ADR-005 Phase 1b — 1bNEW.29 NSG Sweep Microbenchmark (Pre-Spike Measurement)

**Date:** 2026-04-11
**Runner:** Claude (CFA agent #1, nsg-microbench), as the mandatory pre-spike microbench for 1bNEW.29
**Scope:** Falsify or confirm the static-analysis hypothesis from `docs/spike-1bNEW22-instrumentation.md` (lines 898–916) that candle-metal-kernels' hardcoded `N_SIMDGROUP = 2` (`vendor/candle-metal-kernels/src/metal_src/quantized.metal:2307`) is **not** optimal at the exact dispatch shapes hf2q runs in production for the Gemma 4 26B MoE forward, and that a per-shape NSG selection (matching llama.cpp's `FC_mul_mv_nsg` function-constant pattern) would give measurable wall-clock speedup.
**Discipline:** This spike is the "measure first, patch second" gate that the 1bNEW.22 sticky-encoder falsification (`docs/spike-1bNEW22-instrumentation.md:893`) made mandatory. Cost of a refuted patch on 1bNEW.22: ~3 hours of wasted vendor surgery. Cost of this microbench: ~30 minutes from start of the agent turn to verdict.
**Hardware:** Apple M5 Max, `applegpu_g17s`, `MetalDeviceType::Max`, 128 GB unified memory (per the 1bNEW.22 spike's hardware line and confirmed by the microbench's own `device.architecture_name()` print).
**Baseline binary:** main HEAD `fb65fd7` (post-1bNEW.22 sticky-encoder revert; canonical decode 85.4 tok/s on the DWQ Gemma 4 26B GGUF).
**Worktree discipline:** vendored Metal source patch reverted before this report was written. `git diff vendor/` returns empty (verified at end of report).

---

## Executive summary — hypothesis empirically FALSIFIED

The 1bNEW.22 spike's static-comparison reading of llama.cpp vs candle Q-matmul kernels noted that llama.cpp compiles per-NSG specialized variants via Metal function constants (`FC_mul_mv_nsg ∈ {1, 2, 4, 8}`) and selects per dispatch shape via heuristics in `ggml_metal_op_mul_mat_id`, while candle hardcodes `N_SIMDGROUP = 2` for Q4_0 (and bakes the equivalent constant directly into the Q6_K row formula via `row = 2*r0 + sgitg`). The hypothesis was: **for some of hf2q's actual production dispatch shapes, NSG ≠ 2 will give a measurable wall-clock speedup, justifying a port to per-shape NSG selection in candle.**

Direct measurement at all 8 production dispatch shapes, across 4 NSG values, with 1000 timed iterations per cell (and one 2000-iteration confirmation run), shows:

**Across 4 independent runs of the full 32-cell sweep, no (shape, NSG≠2) pair produces a reproducible speedup of even 5%, let alone the ≥5% threshold the task instructions set as the coherence-gate trigger. The largest "winner" margin that reproduced across runs was ~2% — well within per-call jitter (the median per-run µs/call shifts by ~1–3% between runs even at constant NSG).**

The single dramatic outlier from run 1 (`Attn k_proj sliding @ nsg=8` showing 1.21× over nsg=2) **failed to reproduce** on three subsequent runs of the identical configuration. It was a one-shot scheduling/thermal transient on that cell, not a genuine kernel-implementation effect.

**Verdict:** candle's hardcoded `N_SIMDGROUP = 2` is at (or indistinguishable from) the optimum for every Q4_0 and Q6_K dispatch shape hf2q runs in production on M5 Max. The 1bNEW.29 work item as originally framed in the spike — "port to per-shape NSG selection" — has a **measured envelope of ~0 tok/s** and is therefore not justified under Walk discipline. The hypothesis joins the 1bNEW.22 sticky-encoder pattern in the empirically-falsified register.

This is exactly the outcome the spike methodology is built to surface cheaply: 30 minutes to falsify, instead of multi-hour vendor surgery to land a no-op port.

**Coherence gate result:** N/A. The task instructions make the sourdough-gate run mandatory only when "any (shape, NSG) pair where NSG ≠ 2 produces a ≥5% speedup". Zero such pairs exist in the data. No production rebuild was performed; the production path remains untouched on disk and in git.

---

## Methodology

### Measurement primitive

Per-dispatch end-to-end timing via:

1. Build a fresh `MTLCommandBuffer` from a single shared `CommandQueue`.
2. Encode one compute command: `set_compute_pipeline_state` → `set_buffer` (×3) → `set_bytes` (×16 scalar args matching the existing `kernel_mul_mv_q*_f32` ABI byte-for-byte) → `dispatch_thread_groups`.
3. `end_encoding`.
4. **Start timer.**
5. `commit()`.
6. `wait_until_completed()`.
7. **Stop timer.** Record elapsed nanoseconds.

The timer brackets the commit + GPU execution + completion handler — i.e., exactly the wall-clock cost the Rust caller sees per dispatch. Encoder build time is **excluded** from the measurement (it sits before the timer start) so we are measuring kernel execution latency, not Rust-side encode overhead.

This is intentional. The 1bNEW.22 instrumentation already established that CPU encode cost on candle is at the ~0.7 µs/dispatch floor; the question 1bNEW.29 is asking is **whether the GPU-side kernel implementation (specifically NSG threadgroup geometry) can be cheaper at hf2q's exact shapes**, and that question requires a per-dispatch GPU timing primitive. `commit + wait_until_completed` per call gives us exactly that.

The known downside of per-call sync is that it forfeits CPU/GPU pipelining and inflates absolute µs/call vs the production decode path (where 100 dispatches per command buffer are pipelined with `compute_per_buffer = 100`, see 1bNEW.21). That is fine: we are measuring **relative differences between NSG values for the same shape**, not absolute throughput. Inflation is shape-and-NSG-uniform and cancels out in the comparison.

### Iteration plan

* **Warmup:** 100 iterations per (shape, NSG) cell, untimed. Sufficient for the kernel binary to be JIT-loaded into the Metal driver, the threadgroup memory layout to be cached, and the M5 Max compute units to ramp to a stable thermal state.
* **Timed:** 1000 iterations per cell. Reported number is the median (sample[500]). Mean is computed but not reported because the per-call distribution has enough tail to make mean less meaningful than median for cell-to-cell comparison.
* **Confirmation run:** one 2000-iteration sweep (`/tmp/nsg_sweep_run4_2k.txt`) and three 1000-iteration sweeps (`/tmp/nsg_sweep_run{1,2,3}.txt`) — four independent full sweeps total — to gauge run-to-run reproducibility of any putative winner.

### Shapes (from `docs/ADR-005-inference-server.md:906–913`)

| Shape label                      | qtype | n (output rows) | k (input cols) |
|----------------------------------|------:|----------------:|---------------:|
| MLP gate_proj                    |  Q4_0 |            2112 |           2816 |
| MLP down_proj                    |  Q4_0 |            2816 |           2112 |
| Attn q_proj sliding              |  Q6_K |            4096 |           2816 |
| Attn q_proj global               |  Q6_K |            8192 |           2816 |
| Attn k_proj sliding              |  Q6_K |            2048 |           2816 |
| Attn k_proj global               |  Q6_K |            1024 |           2816 |
| Attn o_proj sliding              |  Q6_K |            2816 |           4096 |
| Attn o_proj global               |  Q6_K |            2816 |           8192 |

The activation is always shape `[1, k]` (one decode token), matching the production decode path.

All shapes have `n` divisible by `N_DST × nsg` for every NSG ∈ {1, 2, 4, 8} (Q4_0: N_DST=4 → align ∈ {4,8,16,32}, all divide 2112 and 2816; Q6_K: N_DST=1 → align = nsg ∈ {1,2,4,8}, all divide 4096/8192/2048/1024/2816). No shape was skipped.

### NSG sweep mechanics — DEVIATION FROM TASK SPEC

**Task instruction (verbatim):** "vendor-patch `quantized.metal:2307` to set `#define N_SIMDGROUP <value>`, rebuild kernels only with `cargo build -p candle-metal-kernels`, run the microbench, capture data."

**Reality discovered by reading the kernel:** `#define N_SIMDGROUP 2` at line 2307 only governs the four `kernel_mul_mv_qN_{0,1}_f32` Q4-family kernels via the templated `mul_vec_q_n_f32_impl<block_q*, N_DST, N_SIMDGROUP, N_SIMDWIDTH>` instantiations at lines 2403/2429/2455/2481. **The Q6_K kernel (`kernel_mul_mv_q6_K_f32_impl` at line 5186) does NOT use `N_SIMDGROUP` at all.** The NSG=2 assumption is baked structurally into the row-fanout formula `const int row = 2 * r0 + sgitg;` at line 5215, with no `nsg` parameter or template hook anywhere in the function. Flipping the `#define` would do nothing for any of the six Q6_K shapes in the sweep — and six of the eight production shapes are Q6_K.

Additionally, even for the Q4_0 case where the `#define` is live, flipping it alone is unsafe: the host-side launch geometry in `vendor/candle-metal-kernels/src/kernels/quantized.rs:61–122` is hardcoded to launch a 64-thread (= 2 simdgroups) threadgroup with `align = 8 = N_DST × 2`, and a different NSG requires a matching `(threads_per_threadgroup, threadgroups_per_grid)` change. The host-side file is **not in the spike's allowed-edit list** ("The only legal edits during the spike are `vendor/candle-metal-kernels/{examples/, src/metal_src/quantized.metal}`"), so the simple `#define`-flip-and-rebuild approach was ruled out.

**Adapted approach.** Per the task instruction "If you discover the kernel uses NSG via a code path that ISN'T `#define N_SIMDGROUP` at line 2307 (e.g., per-shape branching, function constant), grep for it, document what you find, and adapt", the sweep was implemented by adding **eight new kernel entry points** to `quantized.metal`:

```text
kernel_mul_mv_q4_0_f32_nsg{1,2,4,8}
kernel_mul_mv_q6_K_f32_nsg{1,2,4,8}
```

Each Q4_0 variant is a thin wrapper that calls the existing `mul_vec_q_n_f32_impl<block_q4_0, N_DST, NSG, N_SIMDWIDTH>` template with the requested NSG. Each Q6_K variant calls a new `kernel_mul_mv_q6_K_f32_impl_nsg<NSG>` function — a template-parameterized clone of the original Q6_K impl with two changes:

1. `const int row = nsg * r0 + sgitg;` (was `2 * r0 + sgitg`)
2. `if (row >= ne01) return;` guard (the original impl had no OOB guard because nsg=2 and the host enforces `align=2`; the parameterized version needs the guard for general nsg).

The original `kernel_mul_mv_q4_0_f32` and `kernel_mul_mv_q6_K_f32` entry points (and the production `#define N_SIMDGROUP 2`) are **byte-identical to HEAD throughout the spike** — production correctness cannot regress because the production code path is never modified.

The microbench (`vendor/candle-metal-kernels/examples/metal_benchmarks.rs::run_nsg_microbench`) bypasses `call_quantized_matmul_mv_t` entirely and dispatches the variant kernels manually via the candle metal-kernels public API (`Kernels::load_pipeline(Source::Quantized, "kernel_mul_mv_q*_f32_nsgN")` → `ComputeCommandEncoder::set_compute_pipeline_state` → manual `set_buffer`/`set_bytes`/`dispatch_thread_groups`), with the launch geometry computed per NSG:

* Q4_0: `threadgroups_per_grid = (n / (4*nsg), 1, 1)`, `threads_per_threadgroup = (32*nsg, 1, 1)`
* Q6_K: `threadgroups_per_grid = (n / nsg, 1, 1)`, `threads_per_threadgroup = (32*nsg, 1, 1)`

Total threads per dispatch is identical across NSG values (`32 × nsg × n / (N_DST × nsg) = 32n / N_DST` for both qtypes), as is the total work — only the threadgroup-vs-grid partition differs, which is exactly what we want to measure.

This adaptation is faithful to the task's intent (sweep NSG at the exact production shapes) while respecting both the no-`src/`-edits rule and the no-`vendor/.../kernels/quantized.rs`-edits rule. It also has the side benefit of **never touching the production kernel binary in any of the 4 sweeps**, eliminating coherence risk by construction during the measurement phase.

### Buffer/data setup

* Weight buffer: `n × (k / block_k) × block_bytes` raw bytes, populated with a Knuth-multiplicative low-entropy pattern (`((i * 2654435761) & 0xFF)`) so the kernel does real arithmetic on real bytes (zero-fill can be optimized by aggressive compilers in some Metal driver versions).
* LHS (activation) buffer: `1 × k × sizeof(f32)`, populated with `(i & 0xFF) * 1e-3` (small magnitudes, finite, non-zero).
* DST buffer: `1 × n × sizeof(f32)`, allocated but not pre-filled (kernel writes every output element).
* All buffers use `RESOURCE_OPTIONS = MTLResourceOptions::StorageModeShared` matching the production crate.

The numerical results are not validated against a reference (this is a pure latency microbench, not a correctness microbench). Bit-exactness of NSG variants vs the production kernel was confirmed indirectly: the Q4_0 nsg=2 and Q6_K nsg=2 variants share the SAME instantiation as the production kernels (`mul_vec_q_n_f32_impl<block_q4_0, 4, 2, 32>` and `kernel_mul_mv_q6_K_f32_impl_nsg<2>` is byte-equivalent to the production `kernel_mul_mv_q6_K_f32_impl` with `row = 2*r0 + sgitg`), so the reported nsg=2 µs/call is comparable to (and bounded by) what production would show under per-call sync.

---

## Data — full sweep (4 independent runs)

### Run 1 (1000 iters/cell, warmup 100; raw at `/tmp/nsg_sweep_run1.txt`)

| shape                                       | type |  nsg=1 µs |  nsg=2 µs |  nsg=4 µs |  nsg=8 µs |
|---------------------------------------------|-----:|----------:|----------:|----------:|----------:|
| MLP gate_proj `[1,2816]@[2112,2816]`        | Q4_0 |    166.62 |    155.25 |    155.00 |    156.79 |
| MLP down_proj `[1,2112]@[2816,2112]`        | Q4_0 |    157.25 |    156.38 |    157.12 |    158.29 |
| Attn q_proj sliding `[1,2816]@[4096,2816]`  | Q6_K |    167.33 |    166.25 |    167.38 |    166.33 |
| Attn q_proj global  `[1,2816]@[8192,2816]`  | Q6_K |    173.92 |    173.67 |    173.83 |    175.12 |
| Attn k_proj sliding `[1,2816]@[2048,2816]`  | Q6_K |    159.67 | **188.17** |   199.71 |    155.71 |
| Attn k_proj global  `[1,2816]@[1024,2816]`  | Q6_K |    142.96 |    145.67 |    144.83 |    142.33 |
| Attn o_proj sliding `[1,4096]@[2816,4096]`  | Q6_K |    155.46 |    156.38 |    153.67 |    155.42 |
| Attn o_proj global  `[1,8192]@[2816,8192]`  | Q6_K |    168.50 |    164.62 |    167.25 |    168.75 |

The bolded `188.17` is the run-1 nsg=2 outlier on `k_proj sliding` that briefly looked like a 1.21× speedup for nsg=8. Spoiler: it does not reproduce.

### Run 2 (1000 iters/cell; raw at `/tmp/nsg_sweep_run2.txt`)

| shape                                       | type |  nsg=1 µs |  nsg=2 µs |  nsg=4 µs |  nsg=8 µs |
|---------------------------------------------|-----:|----------:|----------:|----------:|----------:|
| MLP gate_proj                               | Q4_0 |    163.88 |    154.29 |    155.62 |    155.08 |
| MLP down_proj                               | Q4_0 |    155.29 |    157.79 |    157.79 |    157.21 |
| Attn q_proj sliding                         | Q6_K |    166.21 |    168.00 |    167.00 |    165.04 |
| Attn q_proj global                          | Q6_K |    176.42 |    173.62 |    173.33 |    174.21 |
| Attn k_proj sliding                         | Q6_K |    158.25 |    157.92 |    158.29 |    160.62 |
| Attn k_proj global                          | Q6_K |    158.00 |    155.00 |    156.46 |    158.75 |
| Attn o_proj sliding                         | Q6_K |    166.96 |    168.38 |    164.62 |    165.21 |
| Attn o_proj global                          | Q6_K |    171.17 |    175.33 |    172.04 |    174.33 |

### Run 3 (1000 iters/cell; raw at `/tmp/nsg_sweep_run3.txt`)

| shape                                       | type |  nsg=1 µs |  nsg=2 µs |  nsg=4 µs |  nsg=8 µs |
|---------------------------------------------|-----:|----------:|----------:|----------:|----------:|
| MLP gate_proj                               | Q4_0 |    156.50 |    154.83 |    155.29 |    156.21 |
| MLP down_proj                               | Q4_0 |    155.75 |    157.29 |    156.96 |    159.08 |
| Attn q_proj sliding                         | Q6_K |    165.79 |    166.08 |    167.29 |    165.88 |
| Attn q_proj global                          | Q6_K |    175.54 |    174.21 |    174.83 |    174.75 |
| Attn k_proj sliding                         | Q6_K |    161.25 |    158.83 |    159.46 |    160.83 |
| Attn k_proj global                          | Q6_K |    154.96 |    157.00 |    155.71 |    158.75 |
| Attn o_proj sliding                         | Q6_K |    163.88 |    165.04 |    164.04 |    165.25 |
| Attn o_proj global                          | Q6_K |    171.79 |    171.79 |    170.62 |    171.38 |

### Run 4 (2000 iters/cell, warmup 200; raw at `/tmp/nsg_sweep_run4_2k.txt`)

| shape                                       | type |  nsg=1 µs |  nsg=2 µs |  nsg=4 µs |  nsg=8 µs |
|---------------------------------------------|-----:|----------:|----------:|----------:|----------:|
| MLP gate_proj                               | Q4_0 |    159.96 |    156.08 |    157.38 |    155.71 |
| MLP down_proj                               | Q4_0 |    155.96 |    158.38 |    159.04 |    158.54 |
| Attn q_proj sliding                         | Q6_K |    165.04 |    168.58 |    167.88 |    170.92 |
| Attn q_proj global                          | Q6_K |    177.50 |    173.62 |    176.42 |    174.83 |
| Attn k_proj sliding                         | Q6_K |    159.17 |    158.38 |    157.96 |    159.88 |
| Attn k_proj global                          | Q6_K |    154.54 |    155.25 |    155.42 |    155.42 |
| Attn o_proj sliding                         | Q6_K |    165.38 |    165.21 |    165.00 |    162.79 |
| Attn o_proj global                          | Q6_K |    173.88 |    172.38 |    170.58 |    174.33 |

### Per-cell jitter floor

Holding NSG = 2 constant and tracking each shape's median across the 4 runs gives the **inter-run noise floor** that any "winner" must clear to be a real signal:

| shape (NSG=2 only)              | run1 | run2 | run3 | run4 | min | max | spread |
|---------------------------------|-----:|-----:|-----:|-----:|----:|----:|-------:|
| MLP gate_proj                   | 155.25 | 154.29 | 154.83 | 156.08 | 154.29 | 156.08 | **1.16%** |
| MLP down_proj                   | 156.38 | 157.79 | 157.29 | 158.38 | 156.38 | 158.38 | **1.28%** |
| q_proj sliding                  | 166.25 | 168.00 | 166.08 | 168.58 | 166.08 | 168.58 | **1.51%** |
| q_proj global                   | 173.67 | 173.62 | 174.21 | 173.62 | 173.62 | 174.21 | **0.34%** |
| k_proj sliding                  | **188.17** | 157.92 | 158.83 | 158.38 | 157.92 | 188.17 | **19.16%** ← outlier |
| k_proj sliding (excl. run 1)    |  —   | 157.92 | 158.83 | 158.38 |  —   |  —  | **0.58%** |
| k_proj global                   | 145.67 | 155.00 | 157.00 | 155.25 | 145.67 | 157.00 | **7.78%** |
| o_proj sliding                  | 156.38 | 168.38 | 165.04 | 165.21 | 156.38 | 168.38 | **7.67%** |
| o_proj global                   | 164.62 | 175.33 | 171.79 | 172.38 | 164.62 | 175.33 | **6.51%** |

The jitter floor is **0.3% to 7.8% per shape** for the same NSG=2 across runs, with the `k_proj sliding @ run 1` cell as a 19% transient outlier. **Any cross-NSG "speedup" of <8% on a single run is not even potentially a signal — it's lost in the inter-run jitter** for that shape.

---

## Per-shape verdict

For each shape, taking the **median across all 4 runs** for each NSG and computing the speedup of the best NSG vs nsg=2:

| shape                | type | nsg=1 (4-run med) | nsg=2 (4-run med) | nsg=4 (4-run med) | nsg=8 (4-run med) | best nsg | speedup vs nsg=2 |
|----------------------|-----:|------------------:|------------------:|------------------:|------------------:|---------:|-----------------:|
| MLP gate_proj        | Q4_0 |            161.92 |            155.04 |            155.45 |            155.96 |        2 |          1.0000× |
| MLP down_proj        | Q4_0 |            155.86 |            157.54 |            157.45 |            158.41 |        1 |          1.0108× |
| q_proj sliding       | Q6_K |            166.00 |            167.12 |            167.33 |            166.11 |        1 |          1.0068× |
| q_proj global        | Q6_K |            175.98 |            173.64 |            174.33 |            174.79 |        2 |          1.0000× |
| k_proj sliding¹      | Q6_K |            159.42 |            158.61 |            158.88 |            160.25 |        2 |          1.0000× |
| k_proj global        | Q6_K |            154.75 |            155.12 |            155.56 |            157.08 |        1 |          1.0024× |
| o_proj sliding       | Q6_K |            164.63 |            165.12 |            164.33 |            164.00 |        8 |          1.0069× |
| o_proj global        | Q6_K |            171.48 |            172.08 |            170.60 |            172.86 |        4 |          1.0087× |

¹ k_proj sliding 4-run median **including** the run-1 nsg=2 outlier of 188.17 µs. Excluding the run-1 row (3-run median of runs 2/3/4): nsg1=159.17, nsg2=158.38, nsg4=158.29, nsg8=160.62 — still a 4-way tie, with nsg=4 nominally winning by 0.05% (158.38 / 158.29 = 1.0006×). The transient outlier does not change the verdict.

**Per-shape verdicts:**

* **MLP gate_proj** — nsg=2 wins by ≥0.3% over the next nearest. **Winner: nsg=2 (production).**
* **MLP down_proj** — nsg=1 nominally fastest by 1.08%, well below the per-shape jitter floor (~1.3%). **Winner: nsg=2 (no reproducible improvement).**
* **q_proj sliding** — nsg=1 nominally fastest by 0.68%, below jitter (~1.5%). **Winner: nsg=2.**
* **q_proj global** — nsg=2 wins by 0.4% over nsg=4. **Winner: nsg=2.**
* **k_proj sliding** — nsg=2 wins (including outlier); 4-way tie ±0.06% (excluding outlier). **Winner: nsg=2.**
* **k_proj global** — nsg=1 nominally fastest by 0.24%, well below jitter (~7.8%). **Winner: nsg=2.**
* **o_proj sliding** — nsg=8 nominally fastest by 0.69%, below jitter (~7.7%). **Winner: nsg=2.**
* **o_proj global** — nsg=4 nominally fastest by 0.87%, well below jitter (~6.5%). **Winner: nsg=2.**

**No shape has a non-default winner that survives the per-shape inter-run jitter floor.** The largest "winner" margin (4-run median basis) is **1.08%** on MLP down_proj at nsg=1, vs that shape's ~1.3% jitter floor — i.e., the alleged winner is closer to noise than to a real signal. Every other shape is even tighter. The only candidates that beat nsg=2 by more than the jitter floor on a single run are the run-1 outliers, which the methodology specifically anticipated and ruled out via the multi-run reproducibility check.

---

## Coherence column

The task specifies: "for any (shape, NSG) pair where NSG ≠ 2 produces a ≥5% speedup, ALSO rebuild full hf2q ... and run `bash scripts/sourdough_gate.sh`."

**Number of (shape, NSG≠2) pairs with reproducible ≥5% speedup: 0.**

| shape                | best alt NSG | 4-run median speedup vs nsg=2 | ≥5%? | sourdough required? | sourdough result |
|----------------------|-------------:|------------------------------:|-----:|--------------------:|-----------------:|
| MLP gate_proj        | 4            | 0.997× (slower)               |   no |                  no |              N/A |
| MLP down_proj        | 1            | 1.011×                        |   no |                  no |              N/A |
| q_proj sliding       | 1            | 1.007×                        |   no |                  no |              N/A |
| q_proj global        | 4            | 0.996× (slower)               |   no |                  no |              N/A |
| k_proj sliding       | 4            | 0.998× (slower)               |   no |                  no |              N/A |
| k_proj global        | 1            | 1.002×                        |   no |                  no |              N/A |
| o_proj sliding       | 8            | 1.007×                        |   no |                  no |              N/A |
| o_proj global        | 4            | 1.009×                        |   no |                  no |              N/A |

The full-hf2q rebuild + sourdough gate run was therefore not executed. The decision is consistent with the task instructions, the spike's measure-first discipline, and the load-bearing constraint that "Speed numbers are NOT a result without coherence" — there are no speed numbers to validate.

For completeness, the 1bNEW.22 baseline of 85.4 tok/s + sourdough common-prefix 3095 bytes (≥3094 floor) recorded at HEAD `fb65fd7` in `docs/spike-1bNEW22-instrumentation.md:885` was NOT touched by this spike. The production kernel binary on disk is byte-identical to that HEAD throughout (verified by `git diff vendor/` returning empty after the revert).

---

## Hypothesis verdict

**Hypothesis (from `docs/spike-1bNEW22-instrumentation.md:898–916`):** candle's hardcoded `N_SIMDGROUP = 2` for Q4_0 (and the structural NSG=2 baked into Q6_K via `row = 2*r0 + sgitg`) is suboptimal for one or more of hf2q's production dispatch shapes on M5 Max, and porting to per-shape NSG selection (matching llama.cpp's `FC_mul_mv_nsg` function-constant pattern) would yield a measurable wall-clock speedup of "0 to 10+ tok/s" depending on the microbench result.

**Verdict:** **EMPIRICALLY FALSIFIED.**

* All 8 production dispatch shapes show NSG=2 within the per-shape jitter floor of the best alternative across 4 independent runs (3 × 1000 iters + 1 × 2000 iters per cell).
* The largest "winner" margin (4-run median basis, vs nsg=2) is **1.08%** (`MLP down_proj` at nsg=1), vs that shape's ~1.3% inter-run jitter floor — i.e., the alleged winner is closer to noise than to a real signal. Every other shape's winner margin is even tighter.
* The only ≥5% margin observed in any single run was the run-1 `k_proj sliding @ nsg=8` outlier at 1.21×; **this did not reproduce in any of the three subsequent runs**. The 4-run median at the `k_proj sliding @ nsg=8` cell comes in at 160.25 µs vs nsg=2 at 158.61 µs — i.e., nsg=8 is actually ~1% **slower** than nsg=2 once the transient is averaged out.

**Projected envelope of a port to per-shape NSG selection: 0 tok/s on hf2q decode at M5 Max.** The 1bNEW.29 work item as originally framed should not be landed.

**Why the static comparison with llama.cpp was misleading:** llama.cpp uses Metal function constants for NSG specialization across its full target hardware matrix (M1 Pro through M5 Ultra, and more), where the optimum NSG genuinely varies with compute-unit count, L1 cache behavior, and the n/k aspect ratio of the dispatch shape. On M5 Max specifically, at hf2q's specific Gemma 4 26B MoE shapes (which all have small n in the 1024-8192 range and a fixed activation k of 2112-8192 with batch=1 for decode), the cost model is dominated by **per-dispatch GPU launch latency** (the ~150-175 µs floor that every cell hits regardless of NSG, which the 1bNEW.22 instrumentation already attributed to per-kernel-launch overhead at ~1 µs × 2104 dispatches/token). The threadgroup-vs-grid partitioning of the work has approximately zero effect on latency once the per-launch floor dominates and total work is small enough to fit comfortably within M5 Max's compute-unit budget without saturation effects. **Different target hardware or different dispatch shapes (e.g., the much larger projection matrices in a 70B dense model, or the prefill path with m ≫ 1) might give a different verdict — but for hf2q's exact post-1bNEW.22 production shapes on M5 Max, NSG=2 is at the optimum.**

**Implication for the End-gate-closing strategy:** the 21.6 tok/s gap to llama.cpp's 107 tok/s on this hardware/model **is not closeable via NSG tuning**. The 1bNEW.22 spike's fallback line item — "the per-kernel GPU profiling becomes mandatory" via Xcode Instruments → Metal System Trace, or via per-kernel `MTLCounterSampleBuffer` instrumentation in both candle and ggml-metal — is now the next concrete pre-spike candidate. (Out of scope for this microbench; flagged as a follow-up to the parent.)

---

## Citation trail

* **Patch site for static-analysis hypothesis:** `vendor/candle-metal-kernels/src/metal_src/quantized.metal:2307` — `#define N_SIMDGROUP 2`. Confirmed on disk at HEAD `fb65fd7` by `Read` tool. Used by Q4_0/Q4_1/Q5_0/Q5_1 templates at `quantized.metal:2403`, `:2429`, `:2455`, `:2481`. **Not** used by Q6_K (which structurally bakes nsg=2 into `row = 2*r0 + sgitg` at `quantized.metal:5215`). **Not** used by Q2/3/4/5 K-quant kernels in the way the static analysis assumed — most of them use `(r0 * N_SIMDGROUP + sgitg) * N_DST` row formulas at `quantized.metal:4596`, `:5321`, `:5450`, `:5589`, `:5721`, `:5853`, `:5986`, `:6076`, but these are unused in hf2q's Gemma 4 26B production path.

* **Host-side launch geometry:** `vendor/candle-metal-kernels/src/kernels/quantized.rs:61–122` — `nth0/nth1/align` table per qtype. Q4_0 = `(8, 8, 8)` → 64 threads/threadgroup = 2 simdgroups. Q6_K = `(2, 32, 2)` → 64 threads/threadgroup = 2 simdgroups. Both consistent with NSG=2. **This file was not modified during the spike.**

* **NSG variant kernels added (and reverted):** appended to `vendor/candle-metal-kernels/src/metal_src/quantized.metal` after the existing `kernel_mul_mv_q4_0_f32` (line 2404) and after `kernel_mul_mv_q6_K_f32` (line 5294). 8 new kernel entry points + 1 new templated impl function. Total addition: 308 lines. Reverted via `git checkout HEAD -- vendor/candle-metal-kernels/src/metal_src/quantized.metal` at end of measurement phase.

* **Microbench harness added (NOT reverted, intended to land per task instructions):** `vendor/candle-metal-kernels/examples/metal_benchmarks.rs` — added `Task::NsgMicrobench { iters, warmup }` clap subcommand and the supporting `run_nsg_microbench` / `run_one_shape_nsg` / `nsg_microbench_shapes` functions. The example file currently sits under a gitignore-shadowed path (`.gitignore:20` ignores `examples/` globally); see "Followups" section.

* **Build commands:**
  ```bash
  cd /opt/hf2q/vendor/candle-metal-kernels
  cargo build --example metal_benchmarks --release
  ```
  Build was clean on every NSG variant; no Metal compile errors. The Metal source compiles at runtime via `Kernels::load_pipeline` so kernel-level errors would only surface at first-dispatch — none observed across 4 sweeps × 8 shapes × 4 NSG values × 1100-2200 iterations each.

* **Bench commands:**
  ```bash
  ./target/release/examples/metal_benchmarks nsg-microbench --iters 1000 --warmup 100  # runs 1, 2, 3
  ./target/release/examples/metal_benchmarks nsg-microbench --iters 2000 --warmup 200  # run 4
  ```

* **Raw outputs:**
  * `/tmp/nsg_sweep_run1.txt`
  * `/tmp/nsg_sweep_run2.txt`
  * `/tmp/nsg_sweep_run3.txt`
  * `/tmp/nsg_sweep_run4_2k.txt`

* **Reference docs read at the start of the spike:**
  * `docs/ADR-005-inference-server.md:880–938` — Phase 1b Next Walk session entry, sourdough baseline, 1bNEW.22-v2 NSG sweep specification at lines 898–916.
  * `docs/spike-1bNEW22-instrumentation.md` (full file, read in chunks) — operational context, falsified hypothesis register, mandatory pre-spike microbench discipline.
  * `docs/CLAUDE.md` (project rules — file org, security, swarm discipline, no-`src/` rules during spikes).
  * `vendor/candle-metal-kernels/examples/metal_benchmarks.rs` (existing harness pattern).
  * `vendor/candle-metal-kernels/src/metal_src/quantized.metal:2280-2480, 5180-5294` (Q4_0 and Q6_K impls and their template/structural NSG dependencies).
  * `vendor/candle-metal-kernels/src/kernels/quantized.rs:1-176` (host-side launch glue, the `(nth0, nth1, align)` table per qtype, the `set_params!` ABI for the kernel scalar args).
  * `vendor/candle-metal-kernels/src/metal/{device,encoder,buffer,command_buffer,commands}.rs` (the lower-level dispatch primitives the microbench uses).
  * `scripts/sourdough_gate.sh` (verified the coherence-gate path was understood, even though no run was triggered).

* **Sanity check that production was not touched:** post-revert, `cargo build -p candle-metal-kernels` builds clean from the same HEAD source, confirming the revert is complete and no transitive dependencies were affected.

---

## Followups for the parent

1. **Example file needs un-gitignoring before commit.** `.gitignore:20` ignores `examples/` globally, which shadows `vendor/candle-metal-kernels/examples/` — the harness file is on disk and runnable but invisible to git. Editing `.gitignore` was NOT in the spike's allowed-edit list, so I did not touch it. The parent should add a `!vendor/candle-metal-kernels/examples/` exception (or a more targeted rule) before committing the example as the permanent test asset the task spec calls for. The current on-disk example contents are at `vendor/candle-metal-kernels/examples/metal_benchmarks.rs` and the patch is functionally complete.

2. **The metal kernel patch is reverted.** If the parent wants the NSG variant kernels to land alongside the example (so the test asset is actually runnable post-commit), the parent will need to re-apply the 308-line addition to `vendor/candle-metal-kernels/src/metal_src/quantized.metal`. The diff is reproducible by reading the "NSG sweep variants" comment-banner blocks added in this spike — the contents are documented in this report's "NSG sweep mechanics — DEVIATION" section and the kernel naming scheme (`kernel_mul_mv_q4_0_f32_nsg{1,2,4,8}` and `kernel_mul_mv_q6_K_f32_nsg{1,2,4,8}`). Alternatively, the parent can decline to land the metal patch and treat the harness as a "code preserved for the followup spike that wants to run NSG sweeps on different hardware" — that's a valid call given the negative result.

3. **Next concrete pre-spike candidate.** With NSG falsified, the `docs/spike-1bNEW22-instrumentation.md:920–921` "Alternative if NSG sweep yields nothing" register is now active. The candidates listed there are: (a) Xcode Instruments → Metal System Trace (manual GUI session), (b) `MTLCounterSampleBuffer` instrumentation in both candle and ggml-metal for per-kernel GPU timing (~1 day plumbing in each), (c) accept that the gap to 107 tok/s exceeds Walk-cost-justification and revisit the End-gate definition. None of these are in scope for this spike but the parent will need to choose between them as the next item.

4. **Model path correction in the task spec.** The task instructions cite `/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf` as the GGUF path for the (unused) sourdough gate. The actual path on this machine is `/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf` (the GGUF lives one directory deeper). Noted here for the next agent that may need to invoke the gate.

5. **The spike's "deviation from task spec" was load-bearing.** The task instructed `#define N_SIMDGROUP` flips at line 2307 + `cargo build -p candle-metal-kernels`. That approach would have produced wrong output at every NSG ≠ 2 (because the host-side launch geometry in `kernels/quantized.rs` is hardcoded for nsg=2), and would have produced **zero coverage at all** for the six Q6_K shapes (which don't reference `N_SIMDGROUP`). The task pre-anticipated this with the "If you discover the kernel uses NSG via a code path that ISN'T `#define N_SIMDGROUP` ... grep for it, document what you find, and adapt" clause — flagging here that the adaptation was not optional, it was required for the spike to produce honest data at all.

---

## Worktree clean

```text
$ git diff vendor/
(no output)
```

The vendored `quantized.metal` has been reverted to HEAD `fb65fd7`. The example file edits remain on disk at `vendor/candle-metal-kernels/examples/metal_benchmarks.rs` but are gitignore-shadowed (see Followup 1) and therefore do not appear in `git diff`. No `src/` files were modified at any point during this spike.

`cargo build -p candle-metal-kernels` clean post-revert (verified at end of measurement phase, output at `target/debug/libcandle_metal_kernels.rlib`).

`cargo build --example metal_benchmarks --release` is **not** clean post-revert (the example file references the now-deleted `kernel_mul_mv_q*_f32_nsg*` kernel names by string literal, but these are resolved at runtime via `Kernels::load_pipeline`, so the build still succeeds — the runtime dispatch would fail with a "function not found" error if the example were re-run after the revert without re-applying the metal patch). This is the intended state: the harness is preserved for re-use, the production kernel binary is byte-identical to HEAD, and the parent's commit operation can choose whether to land the metal patch alongside or not.
