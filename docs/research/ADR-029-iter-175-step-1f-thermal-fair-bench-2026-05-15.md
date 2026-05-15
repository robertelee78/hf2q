# ADR-029 iter-175 Step 1f — H-D2 thermal-fair bench: single-site migration is NEUTRAL

**Date**: 2026-05-15
**HEAD**: hf2q `bfda6d8f`, mlx-native `b32b81e`
**Iteration**: 5 of /loop autonomous

## Summary

The single-site `dispatch_tracked` migration committed at iter-175 Step 1e is **NEUTRAL at both tg100 and tg2000** in thermal-fair alt-pair benches. The H-D 3.5pp ceiling is **not capturable via single-site migration**; surrounding hand-placed barriers already drain the encoder state, so the tracker rarely sees an in-flight write to conflict against. Closing the gap requires GLOBAL migration + hand-placed-barrier removal (multi-day refactor), not partial opt-in.

## tg100 — 2-cycle alt-pair, 60s cool-downs

| Arm | C1 t/s | C2 t/s | Mean | Range |
|---|---:|---:|---:|---:|
| A (default, `HF2Q_AUTO_BARRIER=0`) | 95.4 | 94.8 | **95.10** | 0.6 |
| B (`HF2Q_AUTO_BARRIER=1`) | 94.4 | 95.1 | **94.75** | 0.7 |

Delta: **−0.37%** (within bench noise: per-arm range 0.6-0.7 t/s).

## tg2000 — 1-cycle alt-pair, 75s cool-down

| Arm | t/s |
|---|---:|
| A (default) | **93.0** |
| B (`HF2Q_AUTO_BARRIER=1`) | **92.7** |

Delta: **−0.32%** (single-cycle, larger uncertainty but consistent with tg100 direction).

## Why neutral, not the predicted +0.5-1pp

The Step 1e migration switched ONE dispatch site (`kernel_mul_mv_q6_K_f32_nr2` and other quantized matvecs in the `dispatch_mv` else-branch, 19.91% of all decode dispatches) from `encode_threadgroups_with_args` to `dispatch_tracked_threadgroups_with_args`.

For the auto-tracker to ELIDE a hand-placed barrier (= net gain), it would need to detect that a recorded prior dispatch's write range does not conflict with the current dispatch's reads. But:

1. **Hand-placed barriers around the migrated site STILL FIRE**. When they fire, they reset the mem_ranges tracker (the Step 1e change to `memory_barrier()` ensures this). After a hand-placed barrier, the tracker starts fresh.
2. **Inside a sequence of matvec dispatches, the tracker accumulates writes**. But consecutive matvecs have DIFFERENT output buffers (each matmul writes to a fresh slot). No conflict → no auto-barrier added → no net change vs the unmigrated state.
3. **The dispatch site sees the SAME number of barriers as before** — hand-placed ones from the surrounding code, plus zero new auto-barriers.

In short: **the auto-tracker on a single site is a no-op + tiny overhead** because the surrounding code already handles all the real conflicts via hand-placed barriers.

## What this falsifies vs preserves

**Falsified**: H-D2 as a single-site lever. Opt-in migration of one hot site doesn't close any measurable share of the 3.5pp H-D ceiling.

**Preserved**: H-D itself (concurrency strategy lever) — Step 1d's 4-arm matrix bench remains valid. peer-concurrent vs peer-serial = +11.9%, hf2q-concurrent vs hf2q-serial = +8.4%. The 3.5pp gap is real; it's just not unlockable via partial migration.

## What it would take to capture the H-D ceiling

The only way to capture H-D's 3.5pp benefit is **global migration**:
1. Migrate ALL ~400+ hand-placed `enc.memory_barrier()` call sites to use `dispatch_tracked_*` instead
2. REMOVE the hand-placed barriers (let the tracker decide)
3. Verify byte-identity + coherence everywhere
4. Bench to confirm the auto-tracker's barrier-elision rate is materially higher than the hand-placed strategy's

This is **multi-day to multi-week effort** — outside /loop scope. Comparable in size to ADR-031's parallel-encode refactor (which produced 0% wall benefit after similar effort).

## Decision: keep the Step 1e change as default-OFF infrastructure

The Step 1e commit (`mlx-native b32b81e`) is:
- Small (19 LOC net)
- Default-OFF (`HF2Q_AUTO_BARRIER=0` makes it behaviorally identical to the prior code)
- Correctness-tested (298/298 mlx-native unit tests, 2/2 hf2q coherence_smoke, byte-identical first decode token)
- Bench-tested as neutral (no regression at default)

Keeping it preserves the **migration infrastructure** for future global-migration work, identically to how ADR-031 Phase B was kept as scaffolding (operator decision: "if we found benefit, why not enable? — answer: didn't find benefit, so it stays OFF").

If/when a future operator decides the multi-day global migration is worth attempting, this commit + the Step 1d 4-arm bench provide the foundation.

## Updated iter-175 hypothesis ledger

| Hypothesis | Status | Source |
|---|---|---|
| H-A: per-dispatch encoder overhead | **FALSIFIED** (Step 1b) | encoder.rs side-by-side; CPU <1% of wall |
| H-B: Metal compile-options divergence | **PARTIALLY FALSIFIED** (Step 1b) | Both repos default MTLCompileOptions |
| H-C: cache/memory layout | **DEFERRED** (operator-runs Instruments) | Not /loop-suitable |
| H-D: concurrency dispatch strategy | **CONFIRMED but not partial-capturable** (Step 1d + 1f) | 3.5pp ceiling; single-site = neutral |
| H-D2: partial migration captures share of H-D | **FALSIFIED** (Step 1f) | tg100 −0.37%, tg2000 −0.32% within noise |
| H-E: precompiled .metallib vs runtime compile | **OPEN** | Single-shader test, 0.5-1 day, next-iter candidate |

## Next iteration candidates (ordered by /loop-tractability)

1. **H-E (precompiled metallib)**: take one hot shader (`quantized_matmul_ggml.metal`), compile via `xcrun metal -O3 -c shader.metal -o shader.air && xcrun metallib shader.air -o shader.metallib`, load via `device.new_library_with_url()` in a test path, bench. 0.5-1 day.

2. **iter-175 close-out**: if H-E is also neutral, close iter-175 with the verdict "structural parity reached; remaining ~6-8% gap to peer is the floor on M5 Max + current hardware/Apple SDK + current architecture". Update ADR-029 with closure entry.

3. **(NOT recommended)**: continue H-D2 by extending migration to more sites. Without removing hand-placed barriers AROUND each newly-migrated site, the result will stay neutral. Until we have an end-to-end plan for global migration, individual sites add code complexity without benefit.

## Reproducibility

```bash
MODEL=/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf

# tg100 cycle 1
./target/release/hf2q generate --model "$MODEL" --prompt "Q." --max-tokens 100 \
    --temperature 0 --ignore-eos
sleep 60
HF2Q_AUTO_BARRIER=1 ./target/release/hf2q generate --model "$MODEL" --prompt "Q." \
    --max-tokens 100 --temperature 0 --ignore-eos
sleep 60

# tg100 cycle 2 — repeat above two arms

# tg2000 cycle 1
./target/release/hf2q generate --model "$MODEL" --prompt "Q." --max-tokens 2000 \
    --temperature 0 --ignore-eos
sleep 75
HF2Q_AUTO_BARRIER=1 ./target/release/hf2q generate --model "$MODEL" --prompt "Q." \
    --max-tokens 2000 --temperature 0 --ignore-eos
```
