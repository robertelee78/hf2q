# ADR-031 Phase C — design analysis (post-Phase-B landing)

**Date**: 2026-05-15 (post-Phase-B merge `83c3ea6d`)
**Status**: research artifact; informs Phase C CFA scope
**Pre-req**: Phase B v3 shipped at HEAD `fde76dc6` (infrastructure-only, no measured speedup)

## Executive summary

Phase B v3 demonstrated the parallel-encode worker thread infrastructure works (byte-identical sourdough, coherence preserved, all invariants verified) but did NOT deliver a measurable speedup: tg100 PARALLEL=1 vs PARALLEL=0 was Δ +0.06 (within noise) and tg2000 was Δ −0.50 (within σ).

Conventional wisdom assumed the missing piece was `mlx_native::GraphSession::enqueue()` to overlap Metal command-buffer commits — the R-B7 commit-serialization ceiling. This analysis shows that assumption may be incomplete: Phase B v1's pre-FIX-4 measurement at tg100 showed −1.17% regression driven by **CPU overhead** (worker submit, session allocation, mpsc roundtrip, 2-CB memory bus contention) — none of which are commit-order issues.

Two viable Phase C directions emerge:

**Direction C-α (enqueue API)**: add `GraphSession::enqueue()` to overlap commits. Architecturally clean, peer-aligned (llama.cpp uses this pattern), and small in scope (~5 LOC mlx-native + ~50 LOC hf2q). **But may not move the perf needle** because Metal's commit() is sub-millisecond async — the savings from parallel-commit are tiny if commits are already fast.

**Direction C-β (CPU overhead reduction)**: profile where the per-token CPU overhead actually lives, then attack the dominant cost. Likely candidates: per-token mpsc channel allocation, per-token GraphSession::begin(), profile clone, memory-bus contention from running 2 concurrent encoders. **More likely to move the needle**, but requires diagnostic profiling first.

Recommendation: Phase C v1 starts with **Direction C-β profiling** to determine where overhead actually lives, then chooses between C-α, C-β-specific-fix, or a combination based on measured findings.

## Phase B v3 final measurements (baseline for Phase C)

| Regime | Arm | Mean ± σ | Δ vs PARALLEL=0 |
|---|---|---|---|
| tg100 | MAIN (e86831ab) | 95.86 ± 0.34 | — |
| tg100 | PARALLEL=0 (B9) | 95.52 ± 1.11 | — |
| tg100 | PARALLEL=1 (B9) | 95.58 ± 0.62 | +0.06 (kv-threshold gates parallel OFF) |
| tg2000 | MAIN | 92.63 ± 0.47 | — |
| tg2000 | PARALLEL=0 | 92.93 ± 0.32 | — |
| tg2000 | PARALLEL=1 | 92.43 ± 0.47 | **−0.50** (parallel engages; within σ) |

The tg2000 −0.50 t/s regression is the perf gap Phase C must close (and ideally invert into a gain ≥+2%).

## Direction C-α — `GraphSession::enqueue()` API

### Pattern (llama.cpp ggml-metal-context.m:512-542)

```objc
id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
[cmd_buf enqueue];   // reserve queue position
// ... encode into cmd_buf ...
[cmd_buf commit];    // commits whenever encoding is done
```

The key insight: `enqueue()` reserves queue position BEFORE encoding completes. Multiple CBs can be enqueued upfront in the desired GPU execution order, then encoded in parallel by different threads, then committed independently. Metal queue executes in enqueue-order regardless of commit-order.

### mlx-native implementation surface

- `metal-rs` (0.29 + 0.33): `CommandBuffer::enqueue(&self)` already exposed at `commandbuffer.rs:76`. ✓
- `mlx_native::CommandEncoder` (encoder.rs:700-765): owns `cmd_buf: CommandBuffer` privately. Need a public `enqueue(&self)` method that delegates.
- `mlx_native::GraphSession<'a>` (graph.rs:1029): owns `encoder: CommandEncoder`. Need a public `enqueue(&self)` method.

API addition (~5 LOC mlx-native):

```rust
impl CommandEncoder {
    /// Reserve this command buffer's position on the Metal queue without
    /// committing.  Subsequent commit() places this CB at the reserved
    /// position regardless of when other CBs commit.  Used in parallel
    /// encoding where multiple threads encode separate CBs that must
    /// execute in a specific GPU order.  Idempotent.
    #[inline]
    pub fn enqueue(&self) {
        self.cmd_buf.enqueue();
    }
}

impl<'a> GraphSession<'a> {
    /// Reserve this session's command buffer position on the Metal queue.
    /// See CommandEncoder::enqueue documentation.
    #[inline]
    pub fn enqueue(&self) {
        self.encoder.enqueue();
    }
}
```

### hf2q `encode_parallel_layers_chunked` integration

Currently the helper enforces GPU order via commit-time serialization: worker commits session_a BEFORE mpsc::send; main commits session_b AFTER mpsc::recv. With enqueue(), this becomes:

```rust
// Inside the worker closure:
let mut session_a = exec_static.begin()?;
session_a.enqueue();             // reserve queue position #1
enqueued_tx.send(()).ok();       // signal main that position is locked
// ... encode chunk_a ...
session_a.commit();              // commits at our own pace; queue order preserved
done_tx.send(result);

// In main thread (after submit_to_global_worker succeeds):
enqueued_rx.recv();              // wait for worker to enqueue
session_b.enqueue();             // reserve queue position #2
// ... encode chunk_b ...
session_b.commit();              // commits independently
// ... done_rx.recv() ...
```

Estimated diff: ~50 LOC hf2q + ~5 LOC mlx-native. New env var not needed; just structural reshape of the existing helper.

### Predicted perf impact

Metal commit() is **sub-millisecond async** (returns immediately, GPU runs queued work). The savings from parallel-commit vs serial-commit are tiny (~10-100 µs per commit pair). At gemma4 tg2000 (~22 seconds total decode for 2000 tokens), 100 µs saved per token × 2000 tokens = 200 ms = ~1% speedup — close to noise floor.

**Honest expected gain: marginal, possibly within noise.** Direction C-α is architecturally cleaner but may not move the perf gate to ≥+2%.

## Direction C-β — CPU overhead reduction

Phase B v1 (pre-FIX-4, raw PARALLEL=1) measured −1.17% at tg100. After FIX-4 kv-threshold gates parallel OFF at tg100, the regression there is hidden but the underlying overhead is still present at tg2000.

Claude-impl's iter-1 self-diagnosis attributed overhead to:
1. Fresh mpsc channel per token (`std::sync::mpsc::channel::<WorkerResult>()` at every forward_decode)
2. Worker GraphSession::begin() per token
3. `profile_main.clone()` per token (when profiling is active)
4. Memory-bus contention from 2 concurrent encoders writing to disjoint kv_caches

### C-β-1: hoist mpsc to per-process

Per-token `mpsc::channel()` allocates queue infrastructure + ~64 bytes for the Sender/Receiver pair. At ~95 t/s × 1 channel/token = ~95 channels/s = ~6 KB/s. Not a CPU hotspot in absolute terms, but adds GC-pressure.

Fix: hoist channel to `EncoderWorker` itself or `GpuContext` — reuse across tokens. Bounded mpsc with capacity 1 avoids unbounded growth. ~30 LOC.

### C-β-2: pre-allocate session_a

Per-token `exec_static.begin()` allocates a fresh CommandBuffer + CommandEncoder. Allocation is fast but not free (~10-50 µs per CB on M5 Max).

Fix: pre-warm session_a at GpuContext::new (similar to worker_registry pre-warm). Each forward_decode resets the session via `session_a.reset()`. New API needed in mlx-native: `GraphSession::reset(&mut self)` that clears the encoder state without re-allocating.

Trickier than it sounds because session_a is created inside the worker closure (from exec_static). Pre-warming requires the session to live BETWEEN tokens — extending lifetime beyond the closure.

Estimated diff: medium (~100-200 LOC). Requires lifetime restructuring.

### C-β-3: skip profile.clone()

When profiling is OFF (typical production), `profile_main.clone()` clones `None` — cheap. When ON, it clones the Option<TokenProfile> which contains per-layer Vecs — could be 30 layers × few-hundred bytes = ~10 KB per token.

Fix: skip clone when profile is None; use Option<&mut TokenProfile> with shared atomic accumulator otherwise. ~20 LOC.

Likely small impact (profile is usually OFF).

### C-β-4: instrument memory-bus contention

The 2 concurrent encoders both write to MlxBuffer-backed Metal buffers. On M5 Max (unified memory + shared cache), concurrent writes to disjoint kv_caches may stall on memory-bus.

Fix: measure first via Apple's Instruments (Metal counter `metal::FrameTimeAvg` and shared-memory `MTLCounterSamplingPoint`). If contention is real, possible mitigations include:
- Run worker thread at lower priority (less competition for memory bus)
- Pin worker to efficiency cores (Apple Silicon E-cores have separate memory paths)
- Stagger memory writes via CommandBuffer ordering

This is exploratory — could be a non-issue or a major bottleneck. Requires diagnostic profiling before committing to a fix.

## Phase C scope recommendation

### Step C0 (research): diagnostic profiling

Before implementing either direction, profile Phase B v3's per-token overhead breakdown:

1. Run hf2q with HF2Q_PARALLEL_ENCODE=1 at tg2000 under Apple Instruments.
2. Measure CPU time spent in: mpsc::channel construct, exec.begin(), profile.clone(), worker thread sleep/wake, main thread blocked on recv(), commit() calls.
3. Identify the dominant cost.

This is 1-2 hours of bench work. The result chooses Phase C's primary lever.

### Step C1 (implement primary lever)

Based on C0 findings:
- If commit-order is dominant → implement Direction C-α (enqueue API)
- If mpsc/session/clone is dominant → implement Direction C-β
- If memory-bus contention is dominant → measure thread affinity options (Apple-specific)

### Step C2 (re-bench)

Same 3-arm alt-pair protocol as Phase B (MAIN / PARALLEL=0 / PARALLEL=1) at tg100, tg2000, and tg5000 (deep context where parallel-encode WIN regime is most likely).

### Step C3 (default-flip decision)

If PARALLEL=1 shows ≥+2% gain over PARALLEL=0 with coherence preserved across all regimes → default-flip to ON.
If gain is <2% but ≥+0% non-regression → keep default-OFF, document as opt-in for specific workloads.
If still regression → revert Phase B (or further iterate).

### Step C4 (close remaining items)

- R-B9 submit-failure leak via `Arc<Mutex<KernelRegistry>>` — eliminates the rare leak path
- HF2Q_SPEC_NGRAM + PARALLEL stack interaction measurement (R-B10)
- HF2Q_PARALLEL_ENCODE_KV_THRESHOLD tuning if a sweet spot emerges

## Estimated effort

- C0 profiling: 0.5 day
- C1 implementation (whichever direction): 1-2 days
- C2 re-bench: 0.5 day (gates + alt-pair benches)
- C3 default-flip decision: operator-gated (~0 dev time, just judgment call)
- C4 hardening: 0.5-1 day

Total Phase C: **2.5 - 4 days of focused work** (matches Phase A and Phase B durations).

## Risks

- **R-C1 (HIGH)**: profiling may reveal the bottleneck is structural to having 2 concurrent encoders on Apple unified-memory, not amenable to either C-α or C-β. If so, Phase C may not deliver gain regardless of which lever is pulled, and the conclusion is "Phase B's design is unsuitable for hf2q-on-M5-Max specifically."
- **R-C2 (MEDIUM)**: enqueue() pattern may interact unexpectedly with mlx-native's residency-set + dispatch-counter machinery (the existing `cmd_buf_count`, `barrier_count`, etc. globals). Need to verify dispatch counts are still correct under enqueue.
- **R-C3 (LOW)**: GraphSession::reset() API addition (for C-β-2) requires consensus with mlx-native maintainers (us); may have implications for graph_opt recorded mode.
- **R-C4 (LOW)**: changing the helper's commit-ordering semantics is a behavior-equivalent refactor but tightly coupled to R-B7 commit-ordering invariant — re-verify with sourdough byte-identity gate.

## References

- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:512-542` — peer enqueue pattern
- `~/.cargo/registry/src/index.crates.io-*/metal-{0.29,0.33}/src/commandbuffer.rs:76` — `pub fn enqueue(&self)` in metal-rs
- `/opt/mlx-native/src/encoder.rs:700-765` — current CommandEncoder construction (already uses unretained-refs)
- `/opt/mlx-native/src/graph.rs:1029` — GraphSession definition
- `/opt/hf2q/docs/research/ADR-031-phase-B-thread-safety-analysis.md` — Phase B design grounding
- `/opt/hf2q/docs/research/ADR-031-phase-B-iter2-bench-results.md` — Phase B empirical baseline
- Phase B merge commit `83c3ea6d` — Phase C builds on this
- Memory: `project_adr031_phaseB_LANDED_2026_05_15.md` — Phase B context
- `feedback_metal_bench_protocol_2026_05_12` — bench methodology
