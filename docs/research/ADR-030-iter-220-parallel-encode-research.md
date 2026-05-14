# ADR-030 iter-220 — Deep research: parallel-encode refactor scope

**Date**: 2026-05-14 (iter-220)
**Status**: research artifact; informs Phase A/B/C plan
**Pre-req**: ADR-030 iter-216 ngram Plan B shipped (HEAD `9c48064d`)
**Trigger**: operator question "should/can we do what peer is doing to close the gap?"

## Executive summary

The "multi-month codebase-wide refactor" estimate cited in iter-174 is
stale.  Foundation work landed in iter-380→398 reduces the remaining
work to ~4-5 focused days, NOT multi-month.  Realistic gain is +2-5%
on top of current HEAD's already-realized +4.7% from `dual_buffer_split=3`
— total decode parity improves from 0.94× peer-FA to ~0.96-0.98×, not
all the way to 1.00×.  Risk is low-medium and mitigatable.

## Key findings (evidence-graded)

### F1 [HIGH] — Metal threading allows the pattern.
- `MTLCommandQueue` is thread-safe for encoders across DIFFERENT buffers.
- `MTLCommandBuffer.enqueue()` declares execution order WITHOUT waiting
  for encoding to complete.
- Multiple threads can encode separate command buffers in parallel as
  long as each thread owns its own encoder.
- Sources: [MTLCommandBuffer Apple docs](https://developer.apple.com/documentation/metal/mtlcommandbuffer),
  [Command Organization Programming Guide](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Cmd-Submiss/Cmd-Submiss.html)

### F2 [HIGH] — Peer's exact pattern (llama.cpp ggml-metal-context.m:438-560).
```
n_main = MAX(64, 0.1 * gf->n_nodes)        // main thread does first 10%
n_cb = 1 or 2 (optimal on M1 Pro / M2 Ultra)
// 1. Main thread creates cmd_bufs[n_cb], enqueues, encodes first n_main nodes
// 2. Creates cmd_bufs[0..n_cb), enqueues each in execution order
// 3. dispatch_apply(n_cb, d_queue, encode_async)
//    — GCD runs encode_async in parallel n_cb times,
//      each iteration fills cmd_bufs[cb_idx] with its slice of remaining nodes
// 4. GPU executes buffers serially in enqueue order
```
Key insight: **enqueue-before-encode** decouples execution order from
encoding completion order.

### F3 [HIGH] — hf2q already does step 1 (synchronous early-commit).
At `src/serve/forward_mlx.rs:282-288`:
> "Default: split after layer 3 (~10% of dispatches committed early
> so GPU starts while CPU encodes the remaining 90%). Measured +4.4
> tok/s (94.3→98.7) with zero correctness impact (sourdough gate PASS)."

Commit 799bf026 (ADR-028 iter-373) shipped this default-flip.  This is
llama.cpp's `n_main` equivalent.

### F4 [HIGH] — The missing piece is parallel encoding of the remaining 90%.
- `mlx_native::encoder_worker::EncoderWorker` (104 LOC, complete, tested):
  persistent thread, `submit<F: FnOnce + Send + 'static>(f)`, completion
  via mpsc.  Submissions run in order.
- `src/serve/encoder_worker_singleton.rs`: process-level singleton.
- `src/serve/layer_ctx.rs`: `LayerCtx` struct (13 fields, placeholder).
- `src/serve/forward_mlx.rs:3190`: `encode_one_layer` STUB (returns Ok(())).

Foundation = 80% done; extraction + parallelism wiring = the remaining 20%.

### F5 [HIGH] — `&mut self → &self` cascade is COMPLETE in the layer body.
- Verified: `awk 'NR>=3279 && NR<=6268 && /&mut self/' src/serve/forward_mlx.rs | wc -l` → 0.
- iter-388 commit 5ed6684a finished the refactor.
- This is the BIGGEST risk-mitigator: typical Rust extraction refactors
  fail on borrow checker; ours won't.

### F6 [MEDIUM] — Realistic speedup estimate.
Prior measurements (conflicting):
- iter-115 (commit dbfd0be6): body encode 0.45 ms (5%) + GPU 8.7 ms
  (95%).  Total decode ≈ 9.15 ms/token at 109 tok/s peak.  **Hide-all-
  encode ceiling: +5%**.
- iter-373 (commit 799bf026): `dual_buffer_split=3` → +4.7% measured
  (94.3 → 98.7 tok/s).  This SUBSETS the encode-hide opportunity.
- iter-397 (commit beb49e5a): multi-thread realistic estimate +2.2%.
- iter-174 final memo: claimed 5.4% closure achievable.

**Synthesis**: iter-373 already captured the easy win.  Adding n_cb=2
on top should be incremental, NOT additive — diminishing returns.
Realistic total improvement from current HEAD: +2-5%.
- Best case: 95.4 → 100 tok/s = 0.98× peer-FA (closing 4 of the 6 gap).
- Likely case: 95.4 → 97 tok/s = 0.95× peer-FA (closing 2 of the 6 gap).
- Worst case: no measurable gain due to thread sync overhead.

### F7 [LOW] — Alternative patterns don't apply.
- omlx (Apple's MLX swift) uses framework-level `async_eval` — different
  abstraction (auto-batches dispatches).  Not portable to our hand-
  rolled Metal forward path.
- candle's metal backend: synchronous, no parallel-encode pattern.

### F8 [HIGH] — Risk surface analysis.
- **Memory ordering**: layer N writes to `kv_caches[N]`; each layer has
  its own slot.  No write-after-read or write-after-write conflict
  ACROSS layers.  Within a layer, dispatches respect Metal's
  `set_pending_buffer_ranges` ordering.  **Safe**.
- **Command buffer reuse**: must allocate fresh buffer per encoding
  chunk via `[queue commandBuffer]`.  llama.cpp uses
  `commandBufferWithUnretainedReferences`; we use the same.  hf2q
  retains buffers via `Arc<MlxBuffer>` so the unretained-refs path is
  safe.
- **Profile interference**: `mlx_native::dispatch_count()` is a global
  counter.  Per-layer-disp profiling (`HF2Q_PER_LAYER_DISP=1`) would
  read non-deterministic values under parallel encoding.  Mitigation:
  gate parallel-mode OFF when profiling is enabled, OR thread-local
  counters.
- **Coherence**: byte-identity preserved because GPU execution order is
  identical (enqueue order is preserved); only the CPU encoding
  schedule changes.  Sourdough gate + coherence_smoke verify this.

## Phased plan (revised: 4-5 focused days, NOT multi-month)

### Phase A — Mechanical extraction (~2-3 days, zero behavior change)
1. Walk `forward_decode` pre-loop section (lines 3204-3278): catalog
   every `let` binding the body (3279-6268) reads.  Expected: 25-40
   bindings beyond the 13 already in `LayerCtx`.
2. Add missing fields to `LayerCtx`; populate it in pre-loop.
3. Move 2,989 LOC layer body into `encode_one_layer(&self, layer_idx,
   ctx, session, gpu, profile, total_dispatches)`.  Replace inline
   `let X` with `ctx.X` references.
4. Replace inline body with `for layer_idx { self.encode_one_layer(...) }`.
5. **Gate**: byte-identity on `scripts/sourdough_gate.sh` AND same
   tg100 t/s on `llama-bench`-style fresh measurement.

Risks: dropping a variable.  Mitigation: byte-identity gate fails loud.

### Phase B — Parallelize remaining 90% via EncoderWorker (~1-2 days)
1. After `dual_buffer_split` commits first 3 layers, split layers 4..29
   (27 layers) into TWO chunks of 13-14 each.
2. Pre-create two fresh `GraphSession`s, each with its own command
   buffer, enqueued in order on the same queue.
3. Submit Chunk A (layers 4..16) to `global_encoder_worker()`.
4. Encode Chunk B (layers 17..29) on the main thread inline.
5. Wait on worker completion via mpsc channel.
6. Behind env flag `HF2Q_PARALLEL_ENCODE=1` (default OFF).

Risks: thread sync overhead may eat the gain; concurrent access to
shared buffers (mitigated by per-layer ownership of `kv_caches[layer_idx]`).

### Phase C — Measure, tune, default-flip decision (~1-2 days)
1. Per-layer encode-cost profile to find optimal split point (50/50 may
   not be optimal; ratio depends on per-layer dispatch count).
2. Validate tg100 + tg2000 + tg5000 regimes — no regression at any.
3. Coherence_smoke green.
4. If gain ≥ +2% with coherence preserved: default-flip
   `HF2Q_PARALLEL_ENCODE=1` to ON.

## Open questions

- **Q1**: Can both `GraphSession`s share the same `KernelRegistry` and
  device, or do we need per-thread registries?  Need to inspect
  `GraphSession::new` and `MlxBuffer` thread-safety.
- **Q2**: Are dispatches that write to `activations.hidden` actually
  per-layer-isolated, or does layer N+1 read layer N's output?  If
  there's cross-layer hidden-state R/W, the chunks can't run truly
  parallel — they'd need to be sequential within a chunk, parallel
  across chunks.  Initial answer: layers DO depend on prior layer's
  hidden output, so chunks must be sequential within and parallel
  across — exactly llama.cpp's pattern (each chunk encodes a CONSECUTIVE
  range of layers).
- **Q3**: Profile interference — easy mitigation (gate parallel mode
  OFF when profiling enabled).  Confirm no other global state reads
  that depend on dispatch order.

## Recommended next step

Start **Phase A — mechanical extraction**.  Each substep commits
independently, gated by byte-identity.  Operator can review after each
commit.  If any substep regresses, revert that single commit and stop.

## References

- llama.cpp ggml-metal-context.m:438-560 — parallel-encode pattern
- src/serve/forward_mlx.rs:282-288 — current dual_buffer_split comment
- src/serve/forward_mlx.rs:3190-3201 — encode_one_layer stub
- src/serve/layer_ctx.rs — LayerCtx placeholder
- mlx-native/src/encoder_worker.rs — EncoderWorker complete
- docs/research/ADR-030-phase5-async-design.md — earlier spec-decode-
  specific design (different scope: CPU/GPU overlap WITHIN spec-decode
  round, not regular decode path)
- [MTLCommandBuffer Apple docs](https://developer.apple.com/documentation/metal/mtlcommandbuffer)
- [Command Organization Programming Guide](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Cmd-Submiss/Cmd-Submiss.html)
- [commandBufferWithUnretainedReferences](https://developer.apple.com/documentation/metal/mtlcommandqueue/1508684-commandbufferwithunretainedrefer)
