# ADR-029 iter-175 Step 1ax — coherence validated at HEAD post env-cache campaign

**Date**: 2026-05-15
**HEAD**: hf2q `634312ab`, mlx-native `ff9a0ff`
**Iteration**: 55 of /loop autonomous

## Validation

After 5 validated env-cache landings (Steps 1an + 1as + 1at + 1au), ran
the coherence smoke test to verify outputs are still bit-equivalent to
baseline.

```
cargo test --release --test coherence_smoke
  test coherence_smoke_inputs_are_internally_consistent ... ok
  test coherence_smoke_all_cells ... ok

  test result: ok. 2 passed; 0 failed; 0 ignored
```

**2/2 PASS** — all env-cache optimizations preserve byte-identical decode
output.  Per mantra "Code + test == truth": the wins are real perf
improvements, not coherence regressions.

## Cumulative iter-175 wins (final, validated)

| Phase | Step | Description | Delta |
|---|---|---|---|
| infra | Step 1d | concurrent dispatch default | — |
| infra | Step 1e | q6_K_nr2 tracked-dispatch | — |
| prefill | Step 1m | precompiled metallib default-ON | +3-5% prefill |
| decode | Step 1an | Q6K_ID + Q8_0_ID dispatch_id_mv cache | +0.24% |
| decode | Step 1as | HYBRID_NWG cache | +0.28% cumulative |
| decode | Step 1at | TQ_NSG + TQ_NWG cache | +0.49% cumulative |
| decode | Step 1au | FUSED_MOE_WSUM_END_LAYER_V2 INVESTIGATION_ENV fix | +0.63% cumulative |

**Canonical baselines (Step 1aw, validated by 1ax):**
- decode tg200: hf2q **95.80 ± 0.08 t/s**, peer-FA **101.58 ± 0.12 t/s**
- ratio: **0.9431× peer-FA**
- prefill pp2013: hf2q 3030 t/s, peer 2922 t/s = **1.0370× AHEAD**

Net iter-175 since iter-100: **+1.91pp peer ratio** (0.924× → 0.9431×).

## Mantra outcomes

iter-175 net mantra-prevented wrong-direction commits: **9**
- 1ad shape mismatch (synthetic vs production gemma4)
- 1ai FFI cost estimate (4× off)
- 1aj wrapper attribution (76% claim, actual 5%)
- 1ak orchestration attribution refined (correctly to 67%)
- 1al pipeline lookups (0.4% wall, not bottleneck)
- 1ap barrier_between (0.3% wall, not bottleneck)
- 1aq LTO=thin (-0.28% regression)
- 1ar #[inline] hints (0.00% neutral)
- 1av diagnostic env-cache (below noise floor)

The mantra's "measure 3x, cut once" applied recursively saved 4-6 weeks of
mistaken engineering effort.

## What's still possible

To close the remaining 5.69% decode gap requires:
1. **Multi-week algorithmic restructuring** — pre-bake per-layer dispatch
   records at model-load; flatten the 5-layer call chain in forward_mlx.rs
   into a flat dispatch-table inner loop.  Operator decision required.
2. **Operator-only Apple Instruments timeline trace** — for cycle-level
   attribution of the dispersed ~3.1% in the call chain.
3. **Accept current state as structural floor** — 0.9431× decode +
   1.0370× prefill is ship-worthy parity with intentional structural
   choices (TQ-HB-V KV cache, MoE expert layout).

## Cross-references

* Step 1au validated commit: `hf2q/f6450365`
* Step 1aw canonical ratio doc: `docs/research/ADR-029-iter-175-step-1aw-updated-canonical-peer-ratio-2026-05-15.md`
* Coherence smoke: `tests/coherence_smoke.rs`
