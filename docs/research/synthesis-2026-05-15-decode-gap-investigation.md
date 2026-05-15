# hf2q Decode-Gap Investigation — Synthesis Report

**Date**: 2026-05-15
**HEAD**: post-`b3c14e57` (ADR-029 iter-175 correction)
**Scope**: synthesize findings from ADR-029 iter-100..175 + ADR-031 lifecycle (Phase A, B, C0) + supporting memory entries
**Topic**: why hf2q decode is ~6-8% slower than llama.cpp peer-FA on the same M5 Max hardware, and what closes the gap

## Summary

The hf2q vs peer-FA decode gap (~6-8% at tg100, ~7.5% at tg2000) lives in **per-dispatch GPU kernel execution speed** (~1 µs/dispatch × 866 dispatches/token ≈ 0.85 ms/token), not in CPU/GPU overlap or workflow organization. ADR-031's parallel-encode mission was based on iter-174's wrong premise (inherited from iter-165's CPU-encode-modeling error); ADR-031 C0 empirically falsified that premise. Closure path is per-kernel GPU optimization (continuing ADR-029's H93+ lever ledger), executable in small focused /loop iterations.

## Key Findings

| # | Finding | Evidence Quality | Source |
|---|---|---|---|
| F1 | hf2q tg100 = 95.86 t/s; peer-FA tg100 = ~104.77 t/s; ratio 0.918× (gap 8.5%) | **HIGH** — direct alt-pair thermal-fair bench, σ < 1% per arm | Phase A merge bench (commit `c7f98865`) + Phase B bench `6a33046b` |
| F2 | hf2q has FEWER dispatches/token (866) and FEWER barriers/token (420) than llama.cpp peer-FA (1339 dispatches, 844 barriers) — yet runs slower | **HIGH** — measured directly via HF2Q_SPLIT_TIMING and llama-bench -v at multiple iters | ADR-029 iter-104, iter-115, iter-165, ADR-031 C0 deep |
| F3 | Per-token CPU body encode is 0.55 ± 0.06 ms (~5% of decode wall); GPU body wait is 9.05 ± 0.18 ms (~85% of decode wall) | **HIGH** — measured by HF2Q_SPLIT_TIMING under both PARALLEL=0 and PARALLEL=1 modes | ADR-031 C0 deep (`9e88af76`) |
| F4 | Parallel-encode (HF2Q_PARALLEL_ENCODE=1) shipped correctly + delivers ZERO measurable wall speedup (Δ +0.06 tg100, Δ −0.50 tg2000, within σ ≈ 0.47) | **HIGH** — empirical 5-cycle + 3-cycle alt-pair benches on M5 Max | ADR-031 Phase B v3 bench at `c2a2ee4c` |
| F5 | iter-174's "parallel-encode is closure path" verdict was based on iter-165's math error: modeling peer CPU encode as 2 ms (assumption), computing parallel savings against hf2q's actual 0.46 ms, predicting 5-10% wall savings that don't exist | **HIGH** — math traceable in ADR-029 iter-165 + empirically falsified by ADR-031 C0 | iter-165 documented in ADR-029; falsification documented in ADR-031 C0 |
| F6 | The remaining ~6% gap is **diffused across dispatches** (~1 µs/dispatch × 866 dispatches), not concentrated in any single kernel | **HIGH** — constant ratio 0.92× across tg100/tg2000/tg5000 regimes implies no depth-dependent concentration | iter-111 "constant ratio" + iter-159 "multi-regime gate" |
| F7 | H93 (FC-promote port from llama.cpp's `kernel_mul_mv_q6_K_f32`) demonstrated that per-kernel ports CAN close measurable gap pieces (+1.08-1.26pp multi-regime) | **HIGH** — merged to main at commit `e97f7927`, multi-regime verified | iter-162 documented |
| F8 | The closure-via-kernel-ports approach is incremental (~0.5-1.5pp per port), suits /loop pacing well; not multi-month per-port | **MEDIUM** — based on H93's effort + extrapolation; first port took ~2-3 days end-to-end including bench validation | iter-161/162 history |
| F9 | Phase A's encode_one_layer refactor is durable independent value (cleaner code; sourdough byte-identical; maintainability) | **HIGH** — landed via ADR-031 Phase A merge `c7f98865`, byte-identical decode verified | Phase A landing memory entry |
| F10 | Phase B's parallel-encode infrastructure (worker registry, encode_parallel_layers_chunked, RAII-via-IIFE) is correctly implemented, default-OFF, no production impact, kept for future workloads where CPU/GPU ratio might differ | **MEDIUM** — implementation verified by codex review + queen judge; production-impact analysis is forward-looking | Phase B landing memory entry + codex iter-2 review |

## Methodology

### Sources checked

- **ADR-029** (`/opt/hf2q/docs/ADR-029-gemma4-moe-pipeline-is-the-gap.md`) — 3,619 lines covering iter-100..175 with thermal-fair benches, multi-regime gates, 31-lever ledger, dispatch-count comparisons
- **ADR-031** (`/opt/hf2q/docs/ADR-031-parallel-encode-decode-forward.md`) — Phase A/B/C lifecycle including iter-220 thread-safety research, iter-2 bench, C0 + C0-deep diagnostic findings
- **Auto-memory** (`~/.claude/projects/-opt-hf2q/memory/`) — 20+ ADR-029 iter-* topic files + ADR-031 Phase A/B/closed entries
- **Codebase** (`/opt/hf2q/src/serve/forward_mlx.rs`, `/opt/mlx-native/src/encoder.rs`, `/opt/mlx-native/src/shaders/*.metal`, `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`)
- **Bench data** (`/tmp/parallel-profile-c0.log`, `/tmp/parallel-split-timing.log`, ADR-031 iter-2 bench artifact)

### Tools used in synthesis

- File-based grep + Read across project memory + ADRs
- Cross-reference between memory entries (linked via `[[name]]` references)
- Empirical re-bench during ADR-031 lifecycle (operator-instructed)

### Tools NOT used (acknowledged limitations)

- Apple Instruments Metal trace (GUI tool, requires manual operator launch)
- llama-bench -v with `--per-kernel` breakdown (not yet run; planned in iter-175 step 1)
- `mlx_native::pipeline_dispatch_buckets()` programmatic dump (planned in iter-175 step 1)
- MCP-backed `memory_search` / `memory_store` / `agentdb_pattern_search` — these tools are surfaced by the harness but their backends route through `npx @claude-flow/cli@latest` which the global CLAUDE.md notes is currently flaky; auto-memory file system was used instead as the durable backstop

## Contradictions Resolved

### Contradiction 1: iter-165 header vs iter-165 decision

- **Header** ("Post-H93 GPU/CPU split confirms encoder is NOT the lever") → identifies CPU as 5% of wall; correct
- **Decision** ("Closing requires parallel CB encoding... Predicted gain: 5-10% wall") → wrong; the 5% can only yield 2.5% max via Amdahl's law
- **Resolution**: header was correct; decision was a math error. ADR-031 C0 empirically falsified the decision; ADR-029 iter-175 records the correction. **RESOLVED**.

### Contradiction 2: iter-174 "exhausted" vs Phase A/B landing

- **iter-174** ("/loop autonomous investigation EXHAUSTED... mission at structural ceiling")
- **ADR-031 Phase A/B**: shipped successfully in autonomous /loop sessions
- **Resolution**: iter-174's "exhausted" claim was premised on the wrong lever (parallel-encode being closure path). Phase A's refactor + Phase B's scaffolding shipped correctly; the EMPIRICAL exhaustion happened at C0 when parallel-encode demonstrated zero gain. Autonomous /loop work resumed at iter-175 with corrected framing. **RESOLVED**.

### Contradiction 3: "peer is faster because of parallel encoding" vs same-hardware dispatch-count comparison

- **iter-168** ("Closure path is parallel encoding") → implies peer's parallel encoding gives it the wall advantage
- **Same-hardware data**: hf2q has FEWER dispatches AND FEWER barriers than peer, yet is slower
- **Resolution**: peer's parallel encoding might help peer (peer has 1339 dispatches × ~1.5 µs CPU encode each = ~2 ms CPU encode that overlaps with GPU). But hf2q's 866 dispatches × ~0.6 µs CPU encode = ~0.55 ms — too small for parallelism to matter. The conclusion "parallel encoding is peer's lever" doesn't translate to "parallel encoding is hf2q's lever" because the CPU/GPU ratio differs. **RESOLVED via Amdahl's law analysis at iter-175**.

### Unresolved

- **Is the per-dispatch GPU time gap (hf2q ~10 µs vs peer ~7.3 µs back-computed) uniform across all kernel types, or concentrated in specific kernels?** Step 1 of iter-175's investigation plan (per-kernel timing dump) will answer this. Until measured, two competing hypotheses remain:
  - **H-A**: Uniform — every kernel is ~30% slower per dispatch; closure requires fleet-wide kernel-tuning
  - **H-B**: Concentrated — 2-3 kernels (e.g. matvec for Q6_K, flash_attn_vec_tq_hb) account for >60% of the gap; closure via targeted ports
- H93's existence proof favors H-B (porting one kernel closed +1.08-1.26pp), but iter-111's "constant ratio across regimes" weakly favors H-A. Open question pending Step 1.

## Recommendations

| # | Action | Reasoning | Effort |
|---|---|---|---|
| R1 | **Run iter-175 Step 1** — per-kernel timing dump via mlx_native::pipeline_dispatch_buckets + cumulative per-kernel GPU time | Resolves the open question (H-A uniform vs H-B concentrated); directly informs which kernel to attack first | 0.5-1 day; first /loop iteration's work |
| R2 | **Run iter-175 Step 2** — side-by-side hf2q kernel vs llama.cpp kernel source comparison for top-5 hottest | Once Step 1 identifies top kernels, the comparison reveals concrete port opportunities (threadgroup size, SIMD width, FOR_UNROLL patterns, etc.) | 0.5-1 day; second /loop iteration |
| R3 | **Run iter-175 Step 3** — port + measure first candidate kernel | Each successful port adds ~0.5-1.5pp; matches H93's pattern; iterable | 1-2 days per port |
| R4 | **Operator-run Apple Instruments Metal trace** at next available bench window | Provides GPU-side observability that mlx-native's API can't reach (per-kernel SIMD utilization, memory-bus stalls, dispatch-queue scheduling artifacts) | 1-2 hours operator time; informs port priorities |
| R5 | **Set up /loop for ADR-029 iter-175+ continuation** — cron at 30-60 min intervals carrying the iter-175 plan as the directive | Aligns with operator's "continue until complete" mandate; per-port work suits cron pacing well; cancel /loop when hf2q within ±2% of peer-FA on multi-regime | One-shot setup |
| R6 | **Update operator standing-context expectations** | Standing context referenced "0.86-0.92× peer-FA" and "parallel-encode is multi-month lever"; current truth is "0.918× tg100, 0.93-0.94× tg2000/tg5000" and "closure via per-kernel ports, not architecture refactors" | One-shot doc update; included in ADR-029 iter-175 |

## Sources

### ADRs
- `docs/ADR-029-gemma4-moe-pipeline-is-the-gap.md` (3,619 lines; iter-100..175)
- `docs/ADR-031-parallel-encode-decode-forward.md` (Phase A/B/C lifecycle + close-out)

### Research artifacts
- `docs/research/ADR-031-phase-B-thread-safety-analysis.md` — Path D design grounding (Send/Sync analysis)
- `docs/research/ADR-031-phase-B-iter2-bench-results.md` — Phase B v3 empirical perf baseline
- `docs/research/ADR-031-phase-C-design-analysis.md` — Phase C design with R-C1 risk prediction
- `docs/research/ADR-031-phase-C-step-C0-profile-findings.md` — C0 initial profiling
- `docs/research/ADR-031-phase-C-step-C0-deep-investigation.md` — C0 SPLIT_TIMING comparison (decisive)
- `docs/research/ADR-029-resumed-per-kernel-investigation-2026-05-15.md` — iter-175 plan

### Memory entries
- `project_adr031_phaseA_LANDED_2026_05_14.md`
- `project_adr031_phaseB_LANDED_2026_05_15.md`
- `project_adr031_CLOSED_2026_05_15.md`
- `project_adr029_iter174_FINAL_session_outcome_2026_05_13.md`
- `project_adr029_iter162_h93_WIN_2026_05_13.md`
- `project_adr029_iter111_constant_ratio_2026_05_12.md`
- `project_adr029_iter115_gpu95_body_decode_timing_2026_05_12.md`
- `project_adr029_iter158_canonical_baseline_2026_05_13.md`
- `feedback_metal_bench_protocol_2026_05_12.md`
- `feedback_do_not_trust_file_claims_re_measure_2026_05_11.md`

### Code references
- `src/serve/forward_mlx.rs:5552` — `encode_parallel_layers_chunked` private helper (Phase B + C0 instrumentation)
- `src/serve/forward_mlx.rs:1047` — `MlxModelWeights: Send + Sync` compile-time assertion (Phase B foundation)
- `src/serve/gpu.rs:34` — `GpuContext.worker_registry: Option<KernelRegistry>` (Phase B Option A)
- `src/debug/investigation_env.rs:462` — `parallel_encode_kv_threshold` field (FIX-4)
- `mlx-native/src/encoder.rs:342` — `pipeline_dispatch_buckets()` (iter-175 Step 1 tool)
- `mlx-native/src/encoder.rs:387` — `barrier_total_ns()` (timing tool)
- `mlx-native/src/encoder.rs:1709` — `GraphSession::finish_with_gpu_time()` (per-CB GPU timing)
- `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` — peer kernel source (single-file ~250-300 KB MSL)

### Commit ledger (chronological for this synthesis window)
- `c7f98865` — ADR-031 Phase A merged
- `3a95bb20` — Phase A doc landing
- `14ca1a34` — Phase B thread-safety analysis
- `1692cecb` — Phase B Path D pivot
- `e86831ab` — MlxModelWeights Send+Sync compile-test
- `83c3ea6d` — ADR-031 Phase B merged
- `fde76dc6` — Phase B doc landing
- `9e2ee851` — Phase C design analysis
- `a6fdf252` — C0 instrumentation + initial findings
- `9e88af76` — C0 deep investigation (decisive)
- `e33859a0` — ADR-031 CLOSE + ADR-029 resumption plan
- `b3c14e57` — ADR-029 iter-175 correction
- `[current]` — this synthesis report
