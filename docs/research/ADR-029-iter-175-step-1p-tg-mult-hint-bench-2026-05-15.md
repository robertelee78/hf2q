# ADR-029 iter-175 Step 1p — HF2Q_PIPELINE_TG_MULT_HINT bench: +2.08% tg100, neutral tg2000

**Date**: 2026-05-15
**HEAD**: hf2q `6e320ddf`, mlx-native `7fd679f`
**Iteration**: 16 of /loop autonomous

## Summary

Tested ADR-028 iter-376's `HF2Q_PIPELINE_TG_MULT_HINT=1` flag at current HEAD (it was originally added as opt-in default-OFF — never benched at gemma4-APEX-Q5_K_M). The flag sets `threadGroupSizeIsMultipleOfThreadExecutionWidth(true)` on every pipeline descriptor, letting the Metal compiler skip bounds checks and use more aggressive codegen.

Result: **+2.08% decode at tg100, neutral at tg2000**. Coherence_smoke 2/2 PASS. Bench is reproducible at current HEAD.

## Bench data

### tg100 — 2-cycle alt-pair, 60s cool-downs

| Cycle | Arm A (default) | Arm B (HINT=1) |
|---|---:|---:|
| C1 | 92.9 t/s | 95.9 t/s |
| C2 | 94.7 t/s | 95.6 t/s |
| **Mean** | **93.80** | **95.75** |

**Delta: +2.08%** (Arm B is tighter: range 0.3 vs Arm A range 1.8 — likely thermal warm-up affecting A).

### tg2000 — 1-cycle alt-pair, 75s cool-down

| Arm | t/s |
|---|---:|
| A (default) | 92.3 |
| B (HINT=1) | 92.1 |

**Delta: −0.2%** (within bench noise; single cycle, tg2000 has thermal-accumulation effects over the 21.7 s run).

### Correctness

- `coherence_smoke` under `HF2Q_PIPELINE_TG_MULT_HINT=1`: **2/2 PASS** (same as default)

## Why the regime split

`threadGroupSizeIsMultipleOfThreadExecutionWidth(true)` is a HINT to the Metal compiler that lets it elide threadgroup-size bounds checks in the PSO. The benefit accrues every dispatch (small per-call speedup). At tg100, the per-token cost is dominated by dispatches; the per-call speedup compounds across ~850 dispatches/tok → measurable gain. At tg2000, thermal accumulation over the longer run reduces overall throughput regardless of the hint; the per-dispatch speedup is masked.

Hypothesis to confirm in a future bench: tg100 short-runs benefit because they don't accumulate thermal; deeper-kv long-runs don't.

## Why not default-flip yet

The Apple Metal spec requires every dispatched threadgroup to be a multiple of `threadExecutionWidth` (32 on Apple silicon) when this hint is set; otherwise behavior is UNDEFINED. The kernel-registry comment at `mlx-native/src/kernel_registry.rs:1184-1186` claims: "Our hot kernels use tg_size ∈ {32, 64, 256, 1024} (all multiples of 32)."

**Verified**:
- gemma4 production decode + prefill via coherence_smoke (2/2 PASS)

**NOT yet verified** (default-flip safety blocker):
- qwen35 hot path
- qwen3vl vision encoder
- BERT
- nomic_bert
- spec_decode (DFlash)
- calibration paths

Per `feedback_apex_focus_not_dev_ggufs_2026_05_10`: qwen3.6 is a production APEX target. If qwen35 path uses non-multiple-of-32 threadgroups for any dispatch, default-flipping this flag would corrupt qwen3.6 output silently. Coherence test for qwen3.6 would need to pass before default-flipping.

## Safer alternatives

1. **Per-kernel opt-in**: only set the hint when the kernel-registry has verified the kernel's dispatch site uses a multiple-of-32 threadgroup. Requires a static-analysis pass or per-kernel metadata table.

2. **Runtime safety check**: in `CommandEncoder::dispatch_thread_groups` (and similar), assert that `threadgroup_size.x * threadgroup_size.y * threadgroup_size.z` is a multiple of 32 when the hint is set. Converts UB to a panic.

3. **Default-flip gemma4-only**: per `feedback_apex_focus` gemma4 is the ADR-029 target; flipping just for that path captures the +2.08% without risk to other models. Requires plumbing the env-flag through to per-model path or per-pipeline.

Option 3 is cheapest. Option 2 is most robust.

## Recommendation

**Operator decision**: should we default-flip this for gemma4-only (~+2% tg100 free), or invest in option 2 (runtime safety check) before universal default-flip?

The bench data is solid; the gating question is risk tolerance vs reward (+2% tg100 decode).

## Cross-references

- `mlx-native/src/kernel_registry.rs:1184-1192` (and :1349-1352) — hint application site
- ADR-028 iter-376 — original opt-in landing
- `feedback_apex_focus_not_dev_ggufs_2026_05_10` — qwen3.6 production target
