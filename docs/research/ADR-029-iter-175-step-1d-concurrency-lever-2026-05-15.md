# ADR-029 iter-175 Step 1d — H-D investigation: concurrency dispatch strategy IS a lever

**Date**: 2026-05-15
**HEAD**: hf2q `35cdb6a2`
**Iteration**: 3 of /loop autonomous
**Hardware**: M5 Max, same thermal session

## Summary

**H-D (stage-boundary serialization / CB structure) is CONFIRMED as a real lever** via a 4-arm matrix bench. Both hf2q and llama.cpp use `MTLDispatchTypeConcurrent` with auto-barrier tracking, but hf2q's hand-placed barrier strategy extracts only ~70% of the concurrency benefit that peer's `mem_ranges`-based auto-tracker extracts. The remaining ~3.5pp of concurrency throughput is a realistic close-the-gap target. Migration path already exists: `mlx_native::CommandEncoder::dispatch_tracked*` API + `HF2Q_AUTO_BARRIER=1` env, but is currently infrastructure-only (no production call sites).

## 4-arm matrix bench

Same-session, M5 Max, `gemma4-ara-2pass-APEX-Q5_K_M.gguf`, tg100, single-rep with 60-90s cool-downs between arms:

| Variant | tg100 t/s | Concurrency benefit |
|---|---:|---:|
| **peer-FA concurrent (default)** | **103.79** | +11.9% over serial |
| peer-FA serial (`GGML_METAL_CONCURRENCY_DISABLE=1`) | 92.77 | baseline (no concurrency) |
| **hf2q HEAD (concurrent + hand-barriers)** | **92.7** | +8.4% over serial |
| hf2q `HF2Q_FORCE_SERIAL_DISPATCH=1` | 85.5 | baseline (no concurrency) |

Cross-comparisons:
- hf2q-concurrent / peer-concurrent = 92.7 / 103.79 = **0.893×** (10.7% gap at concurrent)
- hf2q-serial / peer-serial = 85.5 / 92.77 = **0.922×** (7.8% gap at serial)
- peer concurrency multiplier: **1.119×**
- hf2q concurrency multiplier: **1.084×**

**Key delta**: peer extracts 1.119/1.084 = **1.032× more concurrency benefit** than hf2q. Stated differently, hf2q is leaving ~3.5pp of concurrency benefit on the table compared to peer.

## Why

Both repos use `MTLDispatchTypeConcurrent` by default. The difference is in **barrier placement strategy**:

| Repo | Strategy | Source |
|---|---|---|
| **peer (llama.cpp)** | **Auto-track via `mem_ranges`**: at every op, check read/write ranges against prior writes; if conflict → insert barrier + reset ranges; otherwise add to running set and let GPU run concurrent | `ggml-metal-ops.cpp:147-225` (called at every `ggml_metal_op_encode_impl` invocation) |
| **hf2q** | **Hand-placed**: explicit `enc.memory_barrier()` calls inserted at specific points in `forward_decode` and equivalents | `forward_mlx.rs` decode path; also see `encoder.rs:940-960` `memory_barrier` method |

Per iter-115 counts: peer has 1339 disp + 844 barriers/tok (0.63 barriers/disp); hf2q has 866 disp + 420 barriers/tok (0.49 barriers/disp). Peer is more aggressive with barriers AND more concurrent in net throughput. The 3.5pp gap suggests hf2q's hand-placed barriers either miss some dependency boundaries (forcing Metal to auto-stall) or over-serialize at safer points than necessary.

## Existing migration infrastructure (Chesterton's fence — read before changing)

`mlx-native` already has the **full peer-equivalent infrastructure**, intentionally built then left uncalled in production:

| Component | Path | Purpose |
|---|---|---|
| `HF2Q_AUTO_BARRIER` env gate | `mlx-native/src/encoder.rs:439-462` | Cached at first read; default OFF |
| `MemRanges` tracker | `mlx-native/src/mem_ranges.rs` | Identical algorithm to peer's `ggml_mem_ranges_*` |
| `dispatch_tracked_*` API family | `mlx-native/src/encoder.rs:1307-1497` (5 variants) | Auto-barrier-aware versions of `encode_*` |
| Counters | `mlx-native/src/encoder.rs:471-479` | `AUTO_BARRIER_COUNT` + `AUTO_BARRIER_CONCURRENT` for measuring elision rate |

Per ADR-015 iter37's comment at `encoder.rs:1286-1290`:

```
// No production callsite migrates in iter37 — this is the API
// surface the qwen35 forward path will adopt incrementally in
// iter38+.  Today, every call to `dispatch_tracked` from a
// production code path lives behind an explicit caller decision
// to opt in.
```

**The migration was planned but never executed.** `dispatch_tracked` has zero production call sites in either `/opt/hf2q/src` or `/opt/mlx-native/src` (verified via grep). Enabling `HF2Q_AUTO_BARRIER=1` alone is a no-op because no production code calls `dispatch_tracked`.

## H-D Verdict

**CONFIRMED**: concurrency strategy IS a real lever; peer extracts 3.5pp more concurrency benefit. The lever is **barrier placement**, not framework defaults.

## Testable next step (H-D2)

**H-D2**: Migrate the hottest hf2q dispatch site (`encode_threadgroups_with_args_and_shared` for `kernel_mul_mv_q6_K_f32_nr2`, 17425 dispatches/100-tok = 174/tok = 19.91% of all dispatches) to use `dispatch_tracked_threadgroups_with_args_and_shared`. Enable `HF2Q_AUTO_BARRIER=1`. Bench tg100 alt-pair vs HEAD.

**Predictions**:
- Best case: q6_K_nr2 site contributes ~3.5pp × (19.91% of dispatches) ≈ ~0.7pp improvement (if barrier-elision rate at this site matches the average). Below noise floor for tg100 but real.
- Expected: 0-1pp improvement; will need to migrate multiple sites to see cumulative effect
- Worst case: site's read/write ranges are over-conservative in the tracker → MORE barriers than hand-placed version → mild regression at this site

**Effort**: 0.5-1 day per dispatch site (need to identify read/write buffer ranges for each dispatch and pass them to `dispatch_tracked_*` instead of `encode_*`). Full migration of 20+ hot sites: multi-day to multi-week.

## What this DOES NOT explain

The 10.7% peer-FA gap is composed of:
- ~3.5pp = concurrency-strategy gap (H-D)
- ~7pp residual = unknown (H-C cache, H-E precompile, or per-kernel intrinsic time differences)

The H-D lever is necessary but not sufficient. Even matching peer's concurrency efficiency leaves ~7pp on the table from other sources.

## Cross-references

- iter-175 Step 1 (dispatch baseline): `docs/research/ADR-029-iter-175-step-1-dispatch-distribution-2026-05-15.md`
- iter-175 Step 1b (H-A/H-B): `docs/research/ADR-029-iter-175-step-1b-encoder-and-compile-options-2026-05-15.md`
- iter-115 dispatches+barriers counts (the data this step refines): memory `project_adr029_iter115_gpu95_body_decode_timing_2026_05_12`
- ADR-015 iter37 (mem_ranges + dispatch_tracked infrastructure that needs production wiring): `mlx-native/src/encoder.rs:1268-1290`
- Peer's mem_ranges algorithm (the target pattern): `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:147-225`

## Reproducibility

Bench command sequence:
```bash
# Arm 1: peer concurrent (default)
llama-bench -m <gguf> -p 0 -n 100 -r 1 -fa 1
sleep 60  # cool-down

# Arm 2: peer serial
GGML_METAL_CONCURRENCY_DISABLE=1 llama-bench -m <gguf> -p 0 -n 100 -r 1 -fa 1
sleep 60

# Arm 3: hf2q concurrent (default)
./target/release/hf2q generate --model <gguf> --prompt "Q." --max-tokens 100 \
    --temperature 0 --ignore-eos
sleep 60

# Arm 4: hf2q serial
HF2Q_FORCE_SERIAL_DISPATCH=1 ./target/release/hf2q generate --model <gguf> \
    --prompt "Q." --max-tokens 100 --temperature 0 --ignore-eos
```
