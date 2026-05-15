# ADR-029 iter-175 Step 1ah — gap LOCALIZED to CPU encode time

**Date**: 2026-05-15
**HEAD**: hf2q `d321e542`, mlx-native `2898e02`
**Iteration**: 39 of /loop autonomous

## Background

Step 1ag proved the per-kernel speed is NOT the bottleneck (hf2q is FASTER
than peer for both top FFN kernels).  Step 1ah localizes where the 6.35%
wall gap actually lives by measuring CPU encode time per token.

## Method

Use the existing `HF2Q_SPLIT_TIMING=1` instrumentation in
`src/serve/forward_mlx.rs:6272-6280` which splits per-token timing into
CPU encode and GPU execution components:

```
HF2Q_SPLIT_TIMING=1 ./target/release/hf2q generate \
  --model gemma4-ara-2pass-APEX-Q5_K_M.gguf \
  --prompt "Q." --max-tokens 10 --ignore-eos
```

## Result

Steady-state (tokens 2-9, post-cold-start):

```
[SPLIT] BODY: encode=0.62ms gpu=8.80ms dispatches=866 barriers=420
[SPLIT] BODY: encode=0.60ms gpu=8.65ms dispatches=866 barriers=420
[SPLIT] BODY: encode=0.58ms gpu=8.62ms dispatches=866 barriers=420
[SPLIT] BODY: encode=0.56ms gpu=8.65ms dispatches=866 barriers=420
[SPLIT] BODY: encode=0.57ms gpu=8.61ms dispatches=866 barriers=420
[SPLIT] BODY: encode=0.58ms gpu=8.61ms dispatches=866 barriers=420
[SPLIT] BODY: encode=0.59ms gpu=9.14ms dispatches=866 barriers=420
[SPLIT] BODY: encode=0.59ms gpu=8.61ms dispatches=866 barriers=420

Generation: 102.8 t/s
```

Per-token breakdown:
- **CPU encode body**: ~0.59 ms (6.0% of wall)
- **GPU body**: ~8.70 ms (89.4% of wall)
- **HEAD + misc**: ~0.44 ms (~4.5% of wall, derived from 9.73 ms wall - 9.29 ms body)
- **Total wall**: ~9.73 ms (matches Generation 102.8 t/s)

## Where the 6.35% gap lives

| Component | hf2q | peer (derived) | delta |
|---|---|---|---|
| GPU body | ~8.70 ms | ~8.70 ms | 0 (kernels equal-or-faster per Step 1ae/1ag) |
| CPU encode | ~0.59 ms | ~0.30 ms | **-0.29 ms (peer faster)** |
| HEAD + misc | ~0.44 ms | ~0.13 ms | -0.31 ms (peer faster) |
| **Total wall** | **9.73 ms** | **9.13 ms** | **-0.60 ms (peer +6.4% wall)** |

Peer's per-dispatch CPU encode is significantly cheaper:
- hf2q: 0.59 ms / 866 dispatches = **~0.68 µs/dispatch**
- peer: ~0.30 ms / 1339 dispatches = **~0.22 µs/dispatch**

Peer's per-dispatch cost is approximately **3× lower** than hf2q's.

(Note: peer's per-dispatch number is estimated by attributing ~0.6 ms
wall gap to encode + head proportionally.  Direct peer instrumentation
would be more accurate but requires modifying peer or running Apple
Instruments — both operator-territory.)

## Why peer's encode is faster

Hypotheses (testable in future iter):
1. **Slimmer dispatcher**: peer uses a minimal ggml dispatcher with fewer
   abstraction layers; hf2q's `dispatch_tracked_*` (Step 1e) adds work.
2. **Better struct layout** for kargs: peer uses tightly packed structs
   with prebaked offsets; hf2q recomputes some on each encode.
3. **Less Rust overhead** — `set_buffer` + `set_bytes` calls go through
   Rust → C++ ffi; peer's C++ avoids the boundary cost.
4. **Different buffer slot binding API** — peer caches `kBuffer*` pointers
   in `argument-buffer-mode` or similar; hf2q calls `set_buffer` per slot
   per dispatch.

## Implication for ADR-029 closure

The remaining 6.35% decode wall gap is **dispersed across CPU-side cost**:
- ~3% from per-dispatch encode (~0.30 ms)
- ~3% from head + misc

To close requires:
1. **Encode-side optimization** — reduce per-dispatch CPU cost.  Multi-day
   work in `mlx-native`'s `encoder.rs` + `kernel_registry.rs` + dispatcher.
2. **HEAD optimization** — fuse final norm + lm_head + softcap into fewer
   dispatches, batch argmax.
3. **Operator-only**: Apple Instruments timeline trace would isolate
   the exact CPU encode bottleneck (RNG, GC, ffi crossings, etc.).

Per-kernel work is DONE (Step 1ag confirmed).

## Standing-context update

For iter-176+ planning:
- Decode wall gap is **localized to CPU side**, not GPU side
- Steady-state hf2q body: 0.59 ms encode + 8.70 ms GPU + 0.44 ms head = 9.73 ms wall
- Per-dispatch encode cost: ~0.68 µs (target: ~0.22 µs to match peer)
- 3× per-dispatch encode reduction would close ~3% of the 6.35% gap

## Cross-references

* Step 1ag (kernel speed confirmed not the bottleneck): `docs/research/ADR-029-iter-175-step-1ag-down-exps-q8_0-corrected-shape-2026-05-15.md`
* Step 1ae (Q6_K kernel +3.89%): `docs/research/ADR-029-iter-175-step-1ae-peer-vs-hf2q-id-kernel-2026-05-15.md`
* Step 1y per-layer-phase profile: `docs/research/ADR-029-iter-175-step-1y-phase-profile-at-HEAD-2026-05-15.md`
* Step 1aa canonical wall ratio 0.9365×: `docs/research/ADR-029-iter-175-step-1aa-peer-ratio-at-HEAD-2026-05-15.md`
* Instrumentation source: `src/serve/forward_mlx.rs:6272-6280` (HF2Q_SPLIT_TIMING=1)
