# ADR-029 iter-175 Step 1aq — `lto = "thin"` FALSIFIED at HEAD

**Date**: 2026-05-15
**HEAD**: hf2q `691b3fc2`, mlx-native `6edc809`
**Iteration**: 47 of /loop autonomous

## Hypothesis

Step 1ak attributed ~3.1% wall to dispersed cross-crate function-call overhead
in the 5-layer dispatch chain `hf2q::forward_decode → encode_one_layer →
attn_norm_qkv_rope → mlx_native::dispatch_id_mv → CommandEncoder::dispatch_tracked_*`.

The Rust compiler cannot inline across crate boundaries without LTO.  Adding
`lto = "thin"` to hf2q's `[profile.release]` should enable cross-crate
inlining and recover some of that 3.1%.

## Method

1. Added 4 lines to `hf2q/Cargo.toml`:
   ```toml
   [profile.release]
   lto = "thin"
   ```
2. Rebuilt: `time cargo build --release --bin hf2q` → **39.4s** (acceptable).
3. Sanity check: 5-token decode looked good (encode 0.46-0.50 ms, Generation 116.9 t/s in burst).
4. Ran 3-cycle thermal-fair alt-pair, gemma4-APEX-Q5_K_M tg200, M5 Max.

## Result

```
[Step 1aq] 3-cycle bench with LTO=thin enabled
  cycle 0: 95.2 tok/s
  cycle 1: 95.0 tok/s
  cycle 2: 94.9 tok/s

hf2q : mean 95.03 t/s  σ 0.12 (0.13%)  samples: [95.2, 95.0, 94.9]
Step 1z baseline    : 95.30 ± 0.21 t/s
Step 1ao (6 caches) : 95.47 t/s (+0.17% vs 1z)

delta vs Step 1z    : -0.28%
delta vs Step 1ao   : -0.46%
```

**Verdict: FALSIFIED — small but consistent regression.**

## Why LTO didn't help (and may hurt)

The initial 5-token burst showed Generation 116.9 t/s which is +14% — but
this was a **thermal/burst artifact**, not a real LTO benefit.  Steady-state
3-cycle bench reveals the true effect: LTO=thin is slightly worse.

Possible explanations:
1. **ICache pressure**: more aggressive inlining grows the hot-loop's
   instruction footprint and forces ICache misses.
2. **Different optimizer decisions**: cross-crate inlining changes what
   the optimizer "sees" and may make worse inlining decisions for
   our shape than the per-crate optimizer makes.
3. **Cross-session variance**: my 3-cycle was 95.03 ± 0.12 vs Step 1z's
   95.30 ± 0.21.  Δ -0.28% is 2-4× σ, likely real but in the noise.

## Action

Reverted the Cargo.toml change.  hf2q HEAD `691b3fc2` retains default
release profile (no LTO).

## What this rules in / out

**Rules out**: "LTO=thin will help close the dispersed nested-call overhead."
The naive Cargo-level LTO toggle does NOT produce a measurable wall improvement
and may slightly regress.

**Rules in** for further investigation (if pursued):
- Targeted `#[inline]` hints on specific hot dispatch methods in mlx-native.
  These work without LTO via Rust's #[inline] attribute and avoid the global
  cost of LTO.
- `lto = "fat"` could be tried (slower compile, more aggressive — may differ
  from thin's result), but the iter-175 lever inventory suggests the
  remaining 3.1% may be irreducible without deeper restructuring.

## Cross-references

* Step 1ak (orchestration localization): `docs/research/ADR-029-iter-175-step-1ak-orchestration-localization-2026-05-15.md`
* Step 1ao baseline (95.47 t/s with 6 env caches): commit `mlx-native/2e381b0`
* Mantra: 'Measure 3x, cut once' — initial burst showed +14%, 3-cycle
  showed -0.28%.  Mantra prevented committing on a false signal.
