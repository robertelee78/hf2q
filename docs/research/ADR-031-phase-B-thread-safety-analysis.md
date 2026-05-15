# ADR-031 Phase B — Send/Sync thread-safety analysis (post-Phase-A)

**Date**: 2026-05-14 (post-Phase-A merge)
**HEAD analyzed**: `3a95bb20` (origin/main, ADR-031 Phase A landing doc update on top of merge `c7f98865`)
**Trigger**: iter-220 research deferred Q1 ("Can both `GraphSession`s share the same `KernelRegistry` and device, or do we need per-thread registries?") — Phase B needs the answer before implementation.

## Executive summary

Phase B as scoped in ADR-031 (submit chunk-A encoding to `global_encoder_worker()`, encode chunk-B on main thread, wait on mpsc) hits a **`'static` closure constraint** that iter-220 didn't anticipate.  The persistent encoder worker's `submit<F: FnOnce + Send + 'static>(f)` API forbids borrowed references to the per-token stack frame.  iter-220's prior "per-token thread spawn" falsification (`-43 tok/s` at forward_mlx.rs:4592-4595) rules out `std::thread::scope` as a workaround.

Realistic Phase B becomes a 2-3 day refactor (not 1-2 days) because the call signature `encode_one_layer(&self, ctx, session, exec, reg, ...)` needs to either Arc-wrap or owned-clone every captured reference.

## Send/Sync inventory (verified by reading mlx-native at HEAD)

| Type | Send | Sync | Verified by |
|---|---|---|---|
| `MlxDevice { device: Device, queue: CommandQueue, residency_set: Option<ResidencySet> }` | ✓ | ✓ | `static_assertions_send_sync!(MlxDevice)` at `device.rs` |
| `GraphExecutor { device: MlxDevice }` | ✓ | ✓ | auto-derived (only field is `MlxDevice`) |
| `KernelRegistry { cache: HashMap<String, ComputePipelineState>, sources: HashMap<String, &'static str> }` | ✓ | ✓ | auto-derived (`metal::ComputePipelineState` is Send+Sync; HashMap is Send+Sync when both K and V are) |
| `GraphSession<'a> { encoder: CommandEncoder, device: &'a MlxDevice, ... }` | ✓ | — | auto-derived; CommandEncoder is `Send` only (encoder.rs:671 `unsafe impl Send`) |
| `CommandEncoder` | ✓ | — | explicit `unsafe impl Send` |
| `EncoderSession` | ✓ | — | explicit `unsafe impl Send` |
| `ObjcResidencySet` | ✓ | ✓ | explicit `unsafe impl Send + Sync` |
| `MlxBuffer` | ✓ | ✓ | `static_assertions_send_sync!(MlxBuffer)` at buffer.rs:80 |

The bottom line: every relevant orchestration type IS `Send`.  `GraphSession` is not `Sync` because its `CommandEncoder` field is `Send`-only.  This is the right pattern — each encoder is owned by exactly one thread at a time.

## The persistent-worker `'static` constraint

`global_encoder_worker()` returns a singleton whose `submit` signature is:

```rust
pub fn submit_to_global_worker<F>(f: F) -> Result<(), &'static str>
where F: FnOnce() + Send + 'static
```

The `'static` bound means F cannot borrow from any non-`'static` scope.  In a per-token context like:

```rust
pub fn forward_decode<'sess>(&self, ...) -> Result<...> {
    let mut s = exec.begin()?;          // GraphSession<'sess>
    let ctx = LayerCtx { ... };          // borrows pre-loop locals
    for layer_idx in 0..num_layers {
        self.encode_one_layer(layer_idx, &ctx, &mut s, exec, &mut reg, ...)?;
    }
}
```

…the closure captured by `submit(move || ...)` cannot reference `&ctx`, `&self`, `&mut s`, `exec`, or `&mut reg`, because all of them are tied to `forward_decode`'s stack frame and the worker thread outlives any single call.

## Serve-layer ownership pattern (verified at HEAD)

`mlx_w: MlxModelWeights` is held by direct ownership (not Arc) at every call
site touched:

| Site | Pattern |
|---|---|
| `src/serve/mod.rs:1422` | `mlx_w.forward_decode(next_token, pos, &mut ctx, &mut p)?` — direct `&mut` |
| `src/serve/mod.rs:1582, 5244, 5432` | same direct ownership |
| `src/serve/api/engine.rs:4107, 4784, 7291` | `loaded.weights.forward_decode(...)` via LoadedModel container |
| `src/serve/parity_quality.rs:649` | direct |
| `src/inference/spec_decode/dflash/orchestrator.rs:1354, 1583` | direct |
| `src/inference/spec_decode/ngram_orchestrator.rs:229` | direct |

No Arc-wrap exists today.  Path A (Arc-everywhere) would touch ~6 call
sites × 5+ methods = ~30 mechanical edits.  Not insurmountable but a
large diff.

## Three paths forward

### Path A — Arc-wrap everything (recommended)

Make `encode_one_layer` callable from `'static` closures by adapting the call:

- `self` → take `Arc<Self>` instead of `&self`.  forward_decode receives `&self` but Phase B-aware paths construct an `Arc::clone(self_arc)` once at process boot or once at forward_decode entry.
- `ctx` → snapshot `LayerCtx`'s slice fields into `Arc<[T]>` (or `Arc<Vec<T>>`) before the parallel split.  Cheap — most fields are ≤32 bytes.
- `session` → each parallel chunk OWNS its own `GraphSession` (created via `exec.begin()`), so no shared reference at all.
- `exec` → wrap once in `Arc<GraphExecutor>` (GraphExecutor is Send + Sync, cheap to share).
- `reg` → wrap once in `Arc<KernelRegistry>` IF kernel cache is read-only after warmup, else `Arc<Mutex<KernelRegistry>>`.  See "KernelRegistry mutation analysis" below.

Trade-off: introduces an Arc-clone-per-token at the parallel-split boundary.  Cost is nanoseconds (atomic incr + decr), invisible at 95 t/s.

### Path B — Owned-clone of LayerCtx fields

LayerCtx's slice references can become owned `Vec<T>` (per-token clone).  `&self` becomes a snapshot struct of self's relevant constants (eps, num_layers, hidden_size, num_attention_heads, etc.) cloned per token.

Pro: no Arc overhead.  Con: per-token allocation cost may eat the parallel speedup.

### Path C — Refactor encode_one_layer to take owned/Arc args directly

Same as A but make it the only API — no fallback to `&self`-borrowing.  Cleanest long-term shape; biggest immediate diff.

**Recommendation revised after serve-layer audit**: Path A is still the
*cleanest long-term shape*, but Path D below (Design Y) is the
*pragmatic v1 choice* for Phase B.

### Path D — Surgical unsafe lifetime-forge with manual wait

The persistent worker's `'static` constraint can be bypassed by
forging a `&'a T` into a `&'static T` AT THE SUBMIT BOUNDARY ONLY,
under the strict precondition that the worker mpsc::recv() completes
BEFORE forward_decode returns and the `'a` lifetime ends.  This is
exactly what `crossbeam::thread::scope` does internally; we replicate
the pattern manually around the persistent global worker.

```rust
// In forward_decode, when HF2Q_PARALLEL_ENCODE=1:
let me: &MlxModelWeights = self;
let ctx_ref: &LayerCtx<'_> = &ctx;
let exec_ref: &mlx_native::GraphExecutor = exec;
let reg_mutex: &Mutex<mlx_native::KernelRegistry> = &reg_mutex;

// SAFETY: this transmute is sound IFF we wait on `done_rx.recv()`
// BEFORE returning from forward_decode (which is enforced below).
// All forged references outlive the worker's use of them.
let me_static: &'static MlxModelWeights = unsafe { std::mem::transmute(me) };
let ctx_static: &'static LayerCtx<'_> = unsafe { std::mem::transmute(ctx_ref) };
// ... etc

let (done_tx, done_rx) = std::sync::mpsc::channel();
submit_to_global_worker(move || {
    let mut session_a = exec_static.begin().expect("begin chunk A");
    for layer_idx in CHUNK_A {
        me_static.encode_one_layer(
            layer_idx, ctx_static, &mut session_a,
            exec_static, &reg_mutex_static, &mut profile_a,
            &mut per_layer_disp_a, &mut total_disp_a,
        ).expect("encode chunk A");
    }
    session_a.commit();
    done_tx.send(()).ok();
}).expect("submit");

// Main thread encodes chunk B:
for layer_idx in CHUNK_B {
    self.encode_one_layer(layer_idx, &ctx, &mut s, exec, &mut reg, ...)?;
}

// MANDATORY: wait before returning, or UB.
done_rx.recv().expect("worker mpsc died");

// Now safe to fall out of forward_decode — the forged 'static refs
// were only valid during the worker's execution, which is now done.
```

Trade-off:
- ✓ Diff is ~50 LOC in forward_mlx.rs ONLY — no serve-layer refactor.
- ✓ Per-token cost: zero Arc-clone overhead.
- ✓ Encapsulates the unsafe block behind the HF2Q_PARALLEL_ENCODE=1 gate.
- ✗ One block of unsafe code requires a clear safety comment + invariant
  documentation.
- ✗ Future maintainer who removes the `done_rx.recv()` introduces UB.
  Mitigation: surround the worker spawn + wait in a private helper
  function that makes the spawn-and-wait atomic at the API level.

This is the recommended path for Phase B v1.  Path A remains the
right shape for a future cleanup.

## KernelRegistry mutation analysis

The `&mut KernelRegistry` parameter in `encode_one_layer` and all `ops::*::dispatch_*` calls suggests mutation, but inspection of `kernel_registry.rs:36-75` shows the only field mutated is `cache: HashMap<String, ComputePipelineState>`.  Insertion is lazy — first reference to a kernel triggers compilation and cache insert.  After the first decode token (which exercises every kernel), the cache is effectively frozen.

For Phase B:
- **Option 1**: Pre-warm at model load.  Loop over every known kernel name and call its lazy-init path once.  After warmup, switch to `&KernelRegistry` reads only (would require touching all `ops::*::dispatch_*` signatures — large surface).
- **Option 2**: `Arc<Mutex<KernelRegistry>>`.  Lock acquired briefly per dispatch.  Contention is ~50-100 ops per layer × 2 threads = ~100 lock-and-release per layer = negligible at µs scale.
- **Option 3**: Switch cache to `RwLock<HashMap>` or even `dashmap::DashMap` for finer-grained concurrency.

**Recommendation**: Option 2 for Phase B v1.  Simplest, correct, perf-acceptable.  Option 1 is a separate optimization if measurement shows lock contention dominates.

## Per-token shape proposal

```rust
// Once at forward_decode entry (or higher):
let self_arc: Arc<Self> = self.clone_arc();   // requires Self: Clone OR pre-construct
let exec_arc: Arc<GraphExecutor> = exec.clone_arc();
let reg_arc: Arc<Mutex<KernelRegistry>> = reg.clone_arc();
let ctx_arc: Arc<LayerCtxOwned> = Arc::new(ctx.to_owned());

// Per-token, after dual_buffer_split=3 commits first 3 layers:
let (mid_lo, mid_hi) = split_layers_4_to_n(num_layers);    // e.g., [4..15] and [15..n-1]
let (tx, rx) = std::sync::mpsc::channel();

// Chunk-A on worker (closures are now 'static via Arc-everything):
{
    let self_arc = Arc::clone(&self_arc);
    let exec_arc = Arc::clone(&exec_arc);
    let reg_arc = Arc::clone(&reg_arc);
    let ctx_arc = Arc::clone(&ctx_arc);
    submit_to_global_worker(move || {
        let mut session_a = exec_arc.begin().expect("begin chunk A");
        for layer_idx in mid_lo {
            self_arc.encode_one_layer_arc(
                layer_idx, &ctx_arc, &mut session_a, &exec_arc, &reg_arc, ...
            ).expect("encode chunk A");
        }
        session_a.commit();
        tx.send(()).ok();
    })?;
}

// Chunk-B on main thread (parallel encode):
let mut session_b = exec_arc.begin()?;
for layer_idx in mid_hi {
    self_arc.encode_one_layer_arc(
        layer_idx, &ctx_arc, &mut session_b, &exec_arc, &reg_arc, ...
    )?;
}
session_b.commit();

// Wait for worker:
rx.recv().expect("worker died");

// Continue with post-layer-loop work (final norm, lm_head, argmax)…
```

`encode_one_layer_arc` is a new sibling of `encode_one_layer` that takes Arc params (or simply re-uses the same method with adapted args).

## Risks specific to Phase B

| Risk | Severity | Mitigation |
|---|---|---|
| `Self` is not `Clone` — can't `Arc::clone(self)` cheaply | HIGH | Verify model type is Send+Sync (likely Arc-able at higher level since model is shared across HTTP requests already) |
| GPU command-buffer enqueue order vs encode order | MEDIUM | Pre-enqueue both buffers BEFORE encoding (per llama.cpp pattern + ADR-031 Phase B point 2) |
| ~~MlxBuffer Send/Sync unverified~~ — VERIFIED | resolved | confirmed Send + Sync at buffer.rs:80 |
| KernelRegistry Mutex contention | LOW | Op-count × thread-count is small; lock is brief HashMap lookup |
| dual_buffer_split=3 already commits first 3 layers — does Phase B interact safely? | MEDIUM | Layers 0-2 run serially BEFORE parallel split; layers 3+ are the parallel target |
| Per-layer dispatch counters (`mlx_native::dispatch_count()`) are global, races under parallel encode | LOW | Spec already mentions gating off HF2Q_PER_LAYER_DISP; needs implementation |

## KernelRegistry sharing strategy refinement

Two viable approaches for the parallel encode:

**Option A (recommended)**: separate `KernelRegistry` per worker thread.
The cache state duplicates across threads (a few hundred kernels ×
ComputePipelineState pointers = negligible memory).  Each thread owns
its own `&mut KernelRegistry`.  Zero Mutex contention.  Pre-warming
both registries at model load ensures no lazy-compile in decode hot
path.  Adds ~5 ms one-time cost at startup; pays zero cost per token.

**Option B**: `Arc<Mutex<KernelRegistry>>` shared.  Brief locks per
dispatch.  At ~50 dispatches/layer × 30 layers × 95 t/s = ~143k locks/s
across both threads.  Empirically: a stdlib Mutex acquisition is
~25-50 ns uncontended.  Total: 143k × 50 ns = 7 ms/s = 0.7% overhead.
Acceptable.

**For Phase B v1**: use Option A (separate registries).  Simplest, no
contention, opt-in cost paid once at startup behind the
HF2Q_PARALLEL_ENCODE=1 gate.

## Recommended next steps

1. ~~Verify `MlxBuffer` Send/Sync by reading buffer.rs.~~ DONE — confirmed Send + Sync via static_assertions at buffer.rs:80.
2. ~~Verify model `Self` type in serve module.~~ DONE — `MlxModelWeights` is held by direct ownership at every call site; not Arc'd today.  Refactoring to Arc would require ~30 mechanical site changes (Path A).  Path D (surgical unsafe lifetime-forge) avoids this.
3. **Verify Send + Sync of `MlxKvCache` and `MlxActivationBuffers`** (the remaining big MlxModelWeights fields) — likely fine since they're built from MlxBuffer (Send+Sync), but unverified at HEAD.  TODO for next iteration.
4. **Spawn /cfa for Phase B implementation** with Path D (surgical) + Option A (separate KernelRegistry per thread) directive.
5. **Phase B substep plan** (revised):
   - B0: Verify MlxKvCache + MlxActivationBuffers Send+Sync; add static_assertions to MlxModelWeights itself.
   - B1: Pre-warm both `KernelRegistry`s (main + worker) at model-load time to eliminate lazy-compile in decode hot path.
   - B2: Add private helper `encode_parallel_layers_chunked(&self, layer_range_a, layer_range_b, ...) -> Result<()>` to forward_mlx.rs that encapsulates the unsafe transmute + submit + main-thread encode + mpsc::recv() pattern as a single atomic API.  The unsafe block lives in this helper ONLY.
   - B3: In forward_decode, after `dual_buffer_splits` commits layer 3, check `HF2Q_PARALLEL_ENCODE=1`; if set, call `encode_parallel_layers_chunked` for layers 4..n-1.  Otherwise fall through to serial loop (status quo).
   - B4: Verify the worker thread holds its own `KernelRegistry` instance (not shared).
   - B5: Gate parallel mode OFF when HF2Q_PER_LAYER_DISP=1 (global counter races) — env-check at forward_decode entry.
   - B6: Build clean + coherence_smoke + sourdough + tg100 alt-pair gates (per Phase A protocol).
   - B7: Default-OFF + opt-in env flag.  Phase C (Phase B perf tune + default-flip decision) is separate work.

## References

- `/opt/hf2q/docs/research/ADR-030-iter-220-parallel-encode-research.md` (predecessor research; superseded on Q1 by this artifact)
- `/opt/hf2q/docs/ADR-031-parallel-encode-decode-forward.md` (parent ADR, Phase A landed at `c7f98865`)
- `/opt/mlx-native/src/encoder_worker.rs` (`submit<F: FnOnce + Send + 'static>(f)` API)
- `/opt/mlx-native/src/graph.rs:1029` (GraphSession definition)
- `/opt/mlx-native/src/kernel_registry.rs:36` (KernelRegistry definition)
- `/opt/mlx-native/src/device.rs:32` (MlxDevice + static_assertions_send_sync)
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:438+550` (peer parallel-encode pattern)
- `feedback_metal_bench_protocol_2026_05_12` (bench methodology for Phase B perf gate)
