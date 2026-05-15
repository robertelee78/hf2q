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

**Recommendation**: Path A.  Minimum surface change, preserves the &self call site for serial decode, opt-in Arc construction only when HF2Q_PARALLEL_ENCODE=1.

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

## Recommended next steps

1. ~~Verify `MlxBuffer` Send/Sync by reading buffer.rs.~~ DONE — confirmed Send + Sync via static_assertions at buffer.rs:80.
2. **Verify model `Self` type** in serve module — confirm it's already `Arc`-shared at the HTTP layer (gemma4_decode_model or whatever the actual type is).
3. **Wrap up this analysis as Phase B research artifact** (this file).
4. **Spawn /cfa for Phase B implementation** with explicit Path A plan + Arc-wrap directive.
5. **Phase B substep plan** (preliminary):
   - B0: Verify Self thread-safety + add MlxBuffer Send/Sync if needed
   - B1: Add `encode_one_layer_arc(Arc<Self>, ..., Arc<Mutex<KernelRegistry>>)` sibling method
   - B2: Add layer-chunking helper + Arc-construction at forward_decode entry behind HF2Q_PARALLEL_ENCODE=1
   - B3: Wire parallel split for layers 4..n-1 (or chosen range)
   - B4: Per-token mpsc completion signaling
   - B5: Per-layer profile interference handling (gate parallel mode OFF when HF2Q_PER_LAYER_DISP=1)
   - B6: Build clean + coherence_smoke + sourdough + tg100 alt-pair gates (per Phase A protocol)
   - B7: Default-OFF + opt-in env flag

## References

- `/opt/hf2q/docs/research/ADR-030-iter-220-parallel-encode-research.md` (predecessor research; superseded on Q1 by this artifact)
- `/opt/hf2q/docs/ADR-031-parallel-encode-decode-forward.md` (parent ADR, Phase A landed at `c7f98865`)
- `/opt/mlx-native/src/encoder_worker.rs` (`submit<F: FnOnce + Send + 'static>(f)` API)
- `/opt/mlx-native/src/graph.rs:1029` (GraphSession definition)
- `/opt/mlx-native/src/kernel_registry.rs:36` (KernelRegistry definition)
- `/opt/mlx-native/src/device.rs:32` (MlxDevice + static_assertions_send_sync)
- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:438+550` (peer parallel-encode pattern)
- `feedback_metal_bench_protocol_2026_05_12` (bench methodology for Phase B perf gate)
