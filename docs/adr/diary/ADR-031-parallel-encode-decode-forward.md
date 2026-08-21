# ADR-031: Parallel-encode the gemma4 decode forward path

- **Status**: **CLOSED 2026-05-15** — Phase A landed (`c7f98865`, durable refactor); Phase B landed as INFRASTRUCTURE-ONLY (`83c3ea6d`, default-OFF, no measurable benefit on hf2q-on-M5-Max); **Phase C abandoned per C0 deep investigation (`9e88af76`) — parallel-encode hypothesis empirically disproven: CPU encode is ~5% of decode wall, GPU body is ~85%, so no CPU-side lever can move the gap by more than ~1%. Real gap-closing levers live in ADR-029 (per-kernel GPU optimization).**

## Close-out 2026-05-15

**ADR-031's premise was wrong.** The original hypothesis — that parallel-encoding the gemma4 decode forward path would close the ~6% gap to llama.cpp peer-FA — assumed CPU encoding was on the critical path.  C0 diagnostic profiling (`a6fdf252` HF2Q_PARALLEL_PROFILE instrumentation + `9e88af76` SPLIT_TIMING comparison) empirically disproved that assumption:

- **CPU body encode**: ~0.55 ms/token = **~5% of decode wall**
- **GPU body wait**: ~9.05 ms/token = **~85% of decode wall**
- Halving CPU encode (theoretical maximum from CPU-parallelism) saves ≤ 2.5% of wall, at noise floor.
- Phase B's 2-CB overhead eats most of that. Net measured: Δ +0.06 t/s tg100, Δ −0.50 t/s tg2000 — statistically indistinguishable from zero.

**Where the gap actually lives**: per-kernel GPU execution speed.  Comparing hf2q vs llama.cpp on the SAME hardware shows llama.cpp has MORE dispatches (1339 vs our 866) and MORE barriers (844 vs 420) yet runs FASTER.  That means per-dispatch GPU time is ~1 µs slower in hf2q × 866 dispatches/token = ~866 µs/token = ~8% of wall = the observed gap.  The lever is per-kernel work (argument-buffer setup, threadgroup sizes, SIMD-width, PSO tuning, memory access patterns), not CPU work organization.

### What remains durable from ADR-031

- **Phase A** (commit `c7f98865`): `encode_one_layer` method extraction from inline forward_decode body.  Pure refactor; zero behavior change; sourdough byte-identical; tg100 neutral.  This is a code-quality win independent of parallel-encode — keeps the layer body as a maintainable unit.
- **Phase B** (commit `83c3ea6d`): default-OFF opt-in `HF2Q_PARALLEL_ENCODE=1` scaffolding.  Worker registry, encode_parallel_layers_chunked private helper, RAII-via-IIFE registry restoration, INV-3..7 + R-B9 invariants.  Kept for future experimentation if a different workload (batch size, hardware) ever makes CPU-encode parallelism worthwhile.
- **Diagnostic instrumentation** (commit `a6fdf252`): `HF2Q_PARALLEL_PROFILE=1` env that prints per-phase µs timings inside `encode_parallel_layers_chunked`.  Permanent diagnostic tool.
- **Research artifacts** at `docs/research/ADR-031-*`: thread-safety analysis, iter-2 bench results, Phase C design + step-C0 + step-C0-deep findings.  Documented closure.

### Lessons learned

1. **Profile before architecting**.  ADR-031's premise was based on iter-174's verdict ("closure requires parallel-encode refactor").  C0 profiling would have shown CPU is 5% of wall in the first hour, saving the 2 days of CFA work that produced Phase B's correct-but-irrelevant infrastructure.
2. **`HF2Q_SPLIT_TIMING=1` is the right tool to ask "is CPU or GPU the bottleneck?"** before deciding on any CPU-side optimization.  The data was already collected in `project_adr029_iter115_gpu95_body_decode_timing_2026_05_12` ("GPU 95% body decode timing") — but the implication ("don't optimize CPU; it's only 5% of wall") was misread by the iter-174 follow-up as "parallelize the CPU to overlap with GPU."
3. **Halving the smaller fraction caps gain at that fraction × halving = (5% × 50%) = 2.5%.**  Even perfect parallelism can't beat that, and 2-CB overhead eats most of it.  This is generalizable: CPU/GPU split timings + Amdahl's law dictate the ceiling BEFORE any architectural work begins.

### Pointer to the real lever

Active mission shifts back to ADR-029 (per-kernel GPU optimization).  Documented baseline:
- `project_adr029_iter174_FINAL_session_outcome_2026_05_13.md` — 31-lever ledger, decode 0.94× peer-FA at HEAD `31677488`
- `project_adr029_iter162_h93_WIN_2026_05_13.md` — H93 FC-promote port from llama.cpp's quantized_matmul_ggml.metal: +1.26% multi-regime decode win
- `project_adr029_iter111_constant_ratio_2026_05_12.md` — gap is ~1 µs/dispatch over 866 dispatches; diffused, not concentrated

Next /loop iteration's focus: identify which specific Metal kernel(s) contribute the largest share of the ~866 µs/token gap, port the slow one(s) from llama.cpp.


- **Date**: 2026-05-14
- **Deciders**: operator (max3@agidreams.us)
- **Tags**: performance, decode, metal, gemma4, peer-parity

## Phase B landing — 2026-05-15 (merge `83c3ea6d`) — INFRASTRUCTURE ONLY, default OFF

Phase B v3 shipped the parallel-encode worker-thread infrastructure that
Phase C will activate.  **Phase B itself does not deliver a measurable
speedup** — bench at HEAD `c2a2ee4c` (pre-merge) showed:

| Regime | MAIN (e86831ab) | PARALLEL=0 (B9) | PARALLEL=1 (B9) | Δ vs PARALLEL=0 |
|---|---|---|---|---|
| tg100 (5-cycle alt-pair) | 95.86 ± 0.34 | 95.52 ± 1.11 | **95.58 ± 0.62** | **+0.06 t/s** (within ±0.3 gate ✓) |
| tg2000 (3-cycle alt-pair) | 92.63 ± 0.47 | 92.93 ± 0.32 | **92.43 ± 0.47** | **−0.50 t/s** (just outside ±0.3, but Δ ≈ σ → statistically within noise) |

At tg100 the FIX-4 kv-depth threshold (`HF2Q_PARALLEL_ENCODE_KV_THRESHOLD`
default 512) routes the call back through the serial path, so PARALLEL=1
is effectively PARALLEL=0 (Δ +0.06 t/s, well within noise).  At tg2000
the parallel path engages but hits the **R-B7 commit-serialization
ceiling**: CPU encoding overlaps across the worker + main threads, but
the Metal command-buffer commits are serialized (worker commits CB-A
before mpsc::send; main commits CB-B after recv) to enforce
chunk-A-before-chunk-B GPU execution order without adding a new
`mlx_native::GraphSession::enqueue()` API.  This caps the realistic gain
ceiling at small-positive in v1.

**Why merge with no demonstrated gain**: Phase B v3 is operator-approved
as infrastructure scaffolding for Phase C.  Default-OFF means production
tg100 + tg2000 are unaffected.  Phase C (separate future CFA session)
will add `GraphSession::enqueue()` to overlap commits, then re-bench to
find the depth where parallel actually wins, then default-flip if
≥+2% gain with coherence preserved.  Operator decision recorded:
**default-flip is GATED on Phase C delivering measurable benefit.**

### Commits merged (10 substeps, preserved via `--no-ff`)

- B0 `3add7a8a` — GpuContext.worker_registry: Option<KernelRegistry>; lazy alloc at HF2Q_PARALLEL_ENCODE=1
- B1 `0d8b28b8` — InvestigationEnv.parallel_encode_enabled() + HF2Q_PER_LAYER_DISP interaction guard
- B2 `1fba9fea` — encode_parallel_layers_chunked Path-D unsafe-lifetime-forge (initial)
- B3 `91883b51` — forward_decode wiring + LayerCtx derive Copy
- B4 `0701ae35` — sourdough byte-identity PARALLEL=1 3/3 PASS
- B5 `51893164` — coherence_smoke 2/2 PARALLEL=1 PASS
- B6 `7468cdaf` — tg100 bench: default neutral, parallel −1.17% (v1 regression that triggered iter-2)
- B7 `e954742d` — FIX-1 (INV-4 unconditional recv) + FIX-2 (INV-3 single unsafe) + FIX-3 (R-B9 mpsc carries reg) + FIX-4 (kv-depth threshold 512)
- B7-gates `ce19198d` — post-fix sourdough 6/6 + coherence 2/2 PASS
- B8 `6a33046b` — perf re-bench: tg100 Δ +0.06, tg2000 Δ −0.50
- B9 `c2a2ee4c` — R-B9 RAII: IIFE wrap + tuple-return signature so put_worker_registry runs unconditionally on every exit path

### Architecture summary

- `MlxModelWeights: Send + Sync` compile-tested at `e86831ab` (Phase B foundation)
- `GpuContext.worker_registry: Option<KernelRegistry>` — second registry pre-warmed at GpuContext::new IF HF2Q_PARALLEL_ENCODE=1 is set in the env
- `encode_parallel_layers_chunked(...) -> (Option<KernelRegistry>, Result<()>)` — private helper containing ONE unsafe `std::mem::transmute` block (three forges of &'a → &'static for self/ctx/exec); mandatory `done_rx.recv()` AFTER worker submit guarantees the forged lifetimes are bounded; tuple return ensures the worker's KernelRegistry is always recoverable through all post-submit error paths
- `forward_decode`'s inner gpu.split() block wrapped in IIFE so `?` returns from the closure, NOT from forward_decode — the post-IIFE `put_worker_registry` runs unconditionally (R-B9 RAII)
- HF2Q_PARALLEL_ENCODE_KV_THRESHOLD (default 512): below threshold, parallel path is skipped — tg100 falls back to serial

### Invariants (all verified at merge)

- INV-1 byte-identity default: sourdough 6/6 PASS, byte-identical to HEAD `e86831ab`
- INV-2 byte-identity parallel: sourdough 3/3 PASS with HF2Q_PARALLEL_ENCODE=1
- INV-3 single unsafe block: 1 unsafe block in encode_parallel_layers_chunked, 0 elsewhere
- INV-4 mandatory recv: done_rx.recv() runs on every return path of the helper (codex iter2 verified)
- INV-5 spawn-and-wait atomicity: spawn + recv co-located in single private helper
- INV-6 interaction guard: parallel_encode_enabled() returns false when HF2Q_PER_LAYER_DISP=1, warn-once
- INV-7 commit order: worker session_a.commit() BEFORE done_tx.send; main session_b.commit() AFTER done_rx.recv
- R-B9 panic-leak: closed for all post-submit-success paths via tuple-return + RAII-IIFE; documented submit-failure edge case remains (Phase C addresses via Arc<Mutex<KernelRegistry>>)

### Carry-forward to Phase C

1. `mlx_native::GraphSession::enqueue()` API addition (the lever for actual speedup)
2. tg5000+ deep-kv-context benches to find regime where parallel wins
3. HF2Q_PARALLEL_ENCODE_KV_THRESHOLD tuning
4. HF2Q_SPEC_NGRAM + PARALLEL stack interaction measurement (R-B10)
5. R-B9 submit-failure registry leak (Arc<Mutex<KernelRegistry>>)
6. Default-flip decision (gated on ≥+2% measured gain + coherence preserved)

## Phase A landing — 2026-05-14 (merge `c7f98865`)

Mechanical layer-body extraction shipped via CFA review-only workflow (queen
Phase 1 spec → claude-impl 5 substeps A1-A5 → codex review → queen Phase 3
judgment 88/100 approve → operator gate green on tg100 alt-pair).

**Commits merged** (preserved via `git merge --no-ff`):
- A1 `38e6aa57` — `LayerCtx` +4 fields, `kv_info` usize type fix
- A2 `818cea1d` — `encode_one_layer` stub signature fix + `LayerCtx` pre-loop construct
- A3 `c748cd46` — parallel copy of 2,278 LOC body into `encode_one_layer`
- A4 `f02495c8` — switch forward_decode loop to `self.encode_one_layer(...)`
- A5 `526e6f70` — cleanup placeholder comments, drop 4 unused LayerCtx fields

**Final diff**: 2 files (`src/serve/forward_mlx.rs`, `src/serve/layer_ctx.rs`),
+676 / -674 lines net via parallel-copy-then-strip strategy.

**Scope correction** (vs the planning numbers above): the actual extracted
body was lines 3513-5792 = **2,280 LOC**, not the "3279-6268 = 2,989 LOC"
this ADR's Context section originally claimed.  Lines 3279-3288 are a
separate KV bookkeeping loop and 3289-3512 are session-begin pre-work —
both stayed in `forward_decode`.  Queen caught this during Phase 1
planning and the spec scoped against the corrected range.

**Gates** (M5 Max, alt-pair thermal-fair, σ<1% per arm, 75s cool-downs):

| Gate | Result |
|---|---|
| `cargo build --release` | clean |
| `coherence_smoke` | 2/2 PASS in 27.40s |
| `sourdough_gate` × 3 | PASS 3/3 (2045 byte common prefix ≥ 179 floor) |
| `clippy --release -D warnings` | no new lints vs `9d08bcd6` baseline |
| Chesterton's fence | 0 `&mut self` references inside `encode_one_layer` body |
| `tg100` Phase A warmed (4 runs) | 96.23 ± 0.15 t/s (σ 0.16%) |
| `tg100` MAIN baseline (3 runs) | 96.10 ± 0.14 t/s (σ 0.15%) |
| peer-FA tg100 (4 runs) | 104.77 ± 0.6 t/s |
| Phase A / peer-FA ratio | **0.918×** vs MAIN 0.917× — refactor neutral |

The refactor is byte-identical at decode output (sourdough) AND at perf
(tg100 within ±0.15 t/s of MAIN; identical peer ratio).

**Spec deviations accepted by queen** (final-report.json):
1. `LayerCtx` ended at 13 fields instead of planned 17.  4 fields
   (`input_token`, `num_layers`, `num_attention_heads`, `eps`) dropped in A5
   because the body reads them via `&self.X` directly — iter-388 invariant
   is `&mut self == 0`, not `&self == 0`.  Eliminated a clippy lint.
2. `encode_one_layer(exec: &'sess GraphExecutor, reg: &mut KernelRegistry)`
   instead of the spec's planned `gpu: &mut GpuContext`.  The outer block in
   `forward_decode` already has `exec`/`reg` from `gpu.split()`; passing
   pre-split refs avoids the R5 double-borrow risk the spec itself flagged.
3. `std::mem::replace(session, exec.begin()?).{finish|finish_with_gpu_time|commit}()`
   at 14 sites — reorders NEW_BEGIN before OLD_FINISH vs the original
   `s.finish(); s = exec.begin()?` pattern.  Empirically validated as safe
   by sourdough 3/3 byte-identity PASS at the only hot-path site (L5468
   dual_buffer commit); 13 debug-only sites theoretically safe because
   `exec.begin()` allocates a disjoint `MTLCommandBuffer` descriptor without
   submitting GPU work.

**Phase B is now unblocked** — `encode_one_layer` is a free-standing method
that can be invoked from a worker thread via the existing
`EncoderWorker`+`encoder_worker_singleton` infrastructure (iter-382).

## Context

ADR-029's investigation thread (iter-100→174) localized hf2q's residual
**~6% decode gap** vs peer-FA (llama.cpp with `-fa 1`) to a single
architectural difference: peer encodes Metal command buffers in
**parallel** with prior dispatch execution via GCD `dispatch_apply` at
`n_cb=1..2`, while hf2q encodes serially on the CPU thread.  iter-174's
final memo concluded closure required a "multi-month codebase-wide
refactor (466 device signature sites + 2,755 LOC layer body
extraction)" and ended autonomous investigation.

Fresh measurement at HEAD `d96c369d` confirms the gap is stable:
- hf2q tg100: 95.4 t/s
- peer-FA tg100: 101.95 t/s
- Ratio: **0.94×** (multi-regime: tg100/tg2000/tg5000 all ≥ 0.92×)

We DO already win on other axes (prefill 1.07-1.09× ahead, KV memory
3.94× advantage, quant-V regime 2.41× ahead per iter-112), so the
overall story is "peer-class on most axes, 6% behind on one."  Operator
ask: close this remaining gap.

The iter-220 deep-research artifact
(`docs/research/ADR-030-iter-220-parallel-encode-research.md`)
revisited the iter-174 verdict and found the foundation work is
**80% complete** from the iter-380→398 thread:

- `&mut self → &self` cascade through the entire layer body (lines
  3279-6268 of `src/serve/forward_mlx.rs`) — verified 0 `&mut self`
  references inside the body (iter-388 commit `5ed6684a`).
- `LayerCtx` struct placeholder defined (iter-390).
- `encode_one_layer` stub declared at `forward_mlx.rs:3190` (iter-391).
- `mlx_native::encoder_worker::EncoderWorker` spawn-and-submit thread
  working and tested (iter-382 / iter-400).
- `src/serve/encoder_worker_singleton.rs` process-level singleton in
  place (iter-382 commit `5c3b97ea`).
- `dual_buffer_split=3` already default-on (iter-373 commit `799bf026`),
  capturing **+4.7%** of the encode-hiding opportunity (94.3 → 98.7
  tok/s in iter-373's measurement).

What is NOT done: the mechanical body extraction + the second-stage
parallelism (split remaining 90% across n_cb=2 encoder threads, peer's
`dispatch_apply` equivalent).  This is the 20% remaining work and is
the scope of this ADR.

Apple Metal's threading model permits this:
- `MTLCommandQueue` is thread-safe for encoders across DIFFERENT
  command buffers.
- `MTLCommandBuffer.enqueue()` declares execution order WITHOUT
  waiting on encoding completion.
- Multiple threads can encode separate buffers in parallel; GPU
  executes them serially in enqueue order.
Source: [MTLCommandBuffer Apple docs](https://developer.apple.com/documentation/metal/mtlcommandbuffer),
[Command Organization Programming Guide](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Cmd-Submiss/Cmd-Submiss.html).

## Decision

Land the parallel-encode refactor in three reviewable phases, each
gated by byte-identity against the prior HEAD's decode output and the
existing coherence-smoke + sourdough gates.

### Phase A — Mechanical layer-body extraction (estimated 2-3 days)
- Catalog every `let` binding in `forward_decode`'s pre-loop section
  (`forward_mlx.rs:3204-3278`) that the body reads.
- Extend `LayerCtx` to cover all of them (expected 25-40 fields total
  vs the 13 placeholders).
- Move the 2,989-LOC layer body (lines 3279-6268) into
  `encode_one_layer(&self, layer_idx, ctx, session, gpu, profile,
  total_dispatches)`.  Replace each inline `let X` with `ctx.X`.
- Replace inline body in `forward_decode` with `for layer_idx {
  self.encode_one_layer(layer_idx, &ctx, &mut session, gpu, ...) }`.
- **Gate per substep**: byte-identical decode output on
  `scripts/sourdough_gate.sh` + same tg100 t/s ± noise floor.
- Zero behavior change.  Pure refactor.

### Phase B — Parallel encoding via EncoderWorker (estimated 1-2 days)
- After `dual_buffer_split=3` commits first 3 layers' work, split the
  remaining 27 layers (4..29) into TWO chunks of ~13-14 layers each.
- Pre-create two fresh `GraphSession`s, each with its own command
  buffer, enqueued on the shared `MTLCommandQueue` in execution order
  (chunk A buffer before chunk B buffer) BEFORE encoding starts.
- Submit chunk A encoding to `global_encoder_worker()` via the
  existing `submit<F: FnOnce + Send + 'static>(f)` API.
- Encode chunk B inline on the main thread.
- Wait for worker completion via mpsc channel.
- Gated behind `HF2Q_PARALLEL_ENCODE=1` (default OFF) for safe rollout.

### Phase C — Measure, tune, default-flip decision (estimated 1-2 days)
- Per-layer encode-cost profile to find the optimal split point (may
  not be 50/50 if dispatch counts vary across layer types).
- Validate tg100 + tg2000 + tg5000 regimes — no regression at any.
- Coherence-smoke green on all production fixtures.
- If gain ≥ +2% with coherence preserved and no regression: default-
  flip `HF2Q_PARALLEL_ENCODE=1` to ON via the env_default_true pattern
  (mirrors iter-149's NWG=32 default-flip).

## Consequences

### Positive
- Closes ~1/3 of the remaining 6% decode gap (95.4 → ~97-98 t/s),
  bringing hf2q to **~0.96-0.98× peer-FA** at decode.
- Combined with existing prefill 1.07-1.09× ahead + KV memory 3.94×
  advantage + quant-V 2.41× ahead, decode parity story becomes "0.96+×
  on all 3 regimes, KV-memory dominant, ahead on prefill and quant-V."
- Encoder thread is reused across decode tokens; no per-token spawn
  cost (the iter-388-pre falsified ad-hoc thread-spawn attempt).
- Sets up the infrastructure for further parallelism work if a future
  kernel-level gain requires per-layer concurrency.

### Negative
- Phase A is mechanically risky (lots of variable moves; one dropped
  binding silently breaks decode).  Mitigated by per-substep byte-
  identity gating.
- Profile counters (`mlx_native::dispatch_count()`) are global state;
  parallel encoding would race them.  Mitigation: gate parallel mode
  OFF when `HF2Q_PER_LAYER_DISP=1` is set.
- Does NOT close the full 6% gap.  Remaining ~2-4% would require
  separate kernel-level optimization work.  Honest expectation
  management for the operator.
- Adds a new env knob (`HF2Q_PARALLEL_ENCODE`) to the contract.

### Neutral
- Phase A is invisible to users (behavior-preserving refactor).
- Phase B/C are opt-in and reversible.
- Default-flip decision in Phase C requires explicit operator approval.

## Links

- ADR-029: peer-parity investigation history (parent context, 174-iter
  ledger establishing the 6% decode gap and its localization).
- ADR-030: spec-decode mission (DFlash + ngram).  Different scope
  (alternative decode strategies, not regular decode path
  optimization).  This ADR can run independently.
- `docs/research/ADR-030-iter-220-parallel-encode-research.md` —
  research artifact establishing scope/risk grounded against current
  code state.
- Peer reference: `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:438-560`
  (the `n_cb` + `dispatch_apply` + `commandBufferWithUnretainedReferences`
  pattern that we mirror).
- `src/serve/forward_mlx.rs:282-288` — current `dual_buffer_split=3`
  comment + measurement.
- `src/serve/forward_mlx.rs:3190-3201` — `encode_one_layer` stub.
- `src/serve/layer_ctx.rs` — `LayerCtx` placeholder.
- `mlx-native/src/encoder_worker.rs` — `EncoderWorker` impl.
- [MTLCommandBuffer Apple docs](https://developer.apple.com/documentation/metal/mtlcommandbuffer).
- [commandBufferWithUnretainedReferences](https://developer.apple.com/documentation/metal/mtlcommandqueue/1508684-commandbufferwithunretainedrefer).
