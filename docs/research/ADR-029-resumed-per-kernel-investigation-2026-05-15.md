# ADR-029 resumed — per-kernel investigation plan (post-ADR-031 close)

**Date**: 2026-05-15
**HEAD**: post-`9e88af76` (ADR-031 closure)
**Trigger**: ADR-031 closed; gap is per-kernel GPU time, not CPU work. Operator approved options 2+3: use Phase B's instrumentation toolkit + investigate via Apple Instruments to identify slow kernels.

## What we know (from standing memory + C0 deep findings)

- hf2q tg100: 95.86 ± 0.34 t/s
- llama.cpp peer-FA tg100: ~104.77 t/s
- Gap: ~8.5% (0.918× peer-FA ratio at tg100; tighter at deeper context per multi-regime data)
- hf2q dispatches/token: **866** (fewer than peer's 1339)
- hf2q barriers/token: **420** (fewer than peer's 844)
- Per-dispatch gap: ~1 µs/dispatch slower × 866 dispatches = ~866 µs/token = ~8% of wall
- The gap is **diffused, not concentrated** — per `project_adr029_iter111_constant_ratio_2026_05_12`: "ratio CONSTANT 0.92× at tg100/2000/5000... per-token delta ~0.85 ms = ~30 µs/layer = ~1 µs/dispatch — below single-site noise floor."

## What we don't yet know

- **Which specific Metal kernel(s)** account for the largest share of the per-dispatch gap?
- Are some kernels at-parity with peer (or even faster) and others much slower, OR is it uniformly slower across all kernels?
- If concentrated in a few kernels: port them from llama.cpp.
- If diffused across all kernels: harder problem — may need general kernel-tuning effort.

## Investigation plan

### Step 1 (this iteration / next iteration) — kernel-level GPU timing

Use mlx-native's per-kernel timing infrastructure (the pipeline_dispatch_buckets at `encoder.rs:342` and barrier_total_ns at `encoder.rs:387`).  Extend or use as-is to count and time individual kernels per token.

Need:
- Per-kernel-name dispatch count per token
- Per-kernel-name cumulative GPU time per token
- Compare to llama.cpp's equivalent (e.g. via `llama-bench` -v or Metal Capture)

mlx-native API: `mlx_native::pipeline_dispatch_buckets()` returns `Vec<(String, u64)>` — kernel-name → dispatch count.  Need to extend with per-kernel cumulative GPU time, or use Apple Instruments to get it.

### Step 2 — side-by-side with llama.cpp

For the top-5 hottest hf2q kernels (by dispatch count × cumulative GPU time):
1. Identify the corresponding llama.cpp kernel (same operation, e.g. `quantized_matmul_simd_bf16` vs `kernel_mul_mm_q4_K_f32`).
2. Compare the Metal Shading Language source side-by-side.
3. Look for: argument-buffer layout differences, threadgroup-size differences, SIMD-width differences, memory access pattern differences, loop unrolling, fused-vs-separate ops.

ADR-029's iter-162 (H93 FC-promote port) is the existence proof: porting llama.cpp's per-kernel pattern can yield +1.26% multi-regime decode wins.  Other kernels may have similar opportunities.

### Step 3 — port + measure

For each candidate slow kernel:
1. Port the llama.cpp pattern (preserving correctness).
2. Run gates (sourdough byte-identity, coherence_smoke).
3. Bench alt-pair (PARALLEL_ENCODE OFF since unrelated; just kernel speed).
4. If gain ≥ 0.5% and within ±0.3 t/s noise floor on tg100/tg2000/tg5000 → land.
5. Continue with next kernel.

This is the lever ledger ADR-029 has been running. New port adds a row.

### Step 4 — Apple Instruments side investigation

In parallel with steps 1-3, run hf2q under Apple Instruments Metal trace:
- "Metal System Trace" template
- Capture: GPU encode timing, kernel execution, memory-bus utilization
- Compare PARALLEL=0 hf2q tg100 vs llama.cpp tg100 traces

This is a GUI tool — requires manual launch in Xcode → Instruments → Metal.  Results: per-kernel breakdown with high-resolution GPU timer.  Cannot be scripted from /loop iterations; operator-runs.

Findings inform Step 1/2 prioritization (which kernels to deep-dive first).

### Step 5 — close-out

When the cumulative gap-closing effort puts hf2q tg100 within ±2% of peer-FA on multi-regime bench, the mission is "complete enough" for production.  Continue per-kernel work as ADR-029 iterations.

## Tools available

- `HF2Q_SPLIT_TIMING=1` — CPU/GPU body+head split (already used in C0 deep finding)
- `HF2Q_PARALLEL_PROFILE=1` — per-phase µs inside encode_parallel_layers_chunked (added in ADR-031 C0)
- `HF2Q_PER_LAYER_DISP=1` — per-layer dispatch count printout
- `mlx_native::pipeline_dispatch_buckets()` — per-kernel-name dispatch counts (existing API)
- `MTLCaptureManager` — Metal Capture trace (set `MLX_METAL_CAPTURE=<path>` + `METAL_CAPTURE_ENABLED=1`)
- Apple Instruments Metal System Trace template (manual / GUI)

## Risks

- **R1 (MEDIUM)**: gap may be truly diffused (1 µs/dispatch across all 866) with no concentrated lever.  In that case, no single kernel port closes meaningful share — would need fleet-wide kernel-tuning effort over weeks.  Mitigation: profile first; if Step 2 shows concentration in 2-3 kernels, proceed; if not, scope-back.
- **R2 (LOW)**: porting llama.cpp kernels may introduce hf2q-specific correctness issues (different KV layout, different MoE structure).  Mitigation: sourdough byte-identity gate per port + coherence_smoke.
- **R3 (LOW)**: Apple Instruments traces are operator-runnable; if operator isn't available to capture them, this branch of investigation pauses.

## References

- `/opt/hf2q/docs/research/ADR-031-phase-C-step-C0-deep-investigation.md` — establishes GPU=85% of wall, CPU=5% of wall
- `project_adr029_iter174_FINAL_session_outcome_2026_05_13.md` — 31-lever ledger baseline; mission resumes from here
- `project_adr029_iter162_h93_WIN_2026_05_13.md` — H93 FC-promote port reference (template for new kernel ports)
- `project_adr029_iter111_constant_ratio_2026_05_12.md` — gap is diffused 1 µs/dispatch
- `project_adr029_iter115_gpu95_body_decode_timing_2026_05_12.md` — early CPU/GPU split (the data ADR-031's premise misread)
- llama.cpp ggml-metal kernels: `/opt/llama.cpp/ggml/src/ggml-metal/*.metal`
- hf2q mlx-native kernels: `/opt/mlx-native/src/shaders/*.metal`
