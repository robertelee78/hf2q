# G: TQ shippable on non-DWQ models — goal already achieved at HEAD d990c817

**Date:** 2026-05-23
**HEAD:** d990c817
**Status:** Goal substantially achieved. Cosmetic load-banner gap remains.

## Goal statement (set this session)

> Make TurboQuant KV-cache shippable as the default on **all** supported models — not just DWQ-trained ones — so hf2q matches or beats peers (llama.cpp / vLLM / KIVI) on KV memory at unimpaired output quality. The blocker is the Gate H quality envelope failing on non-DWQ models like Gemma 4 APEX-Q5_K_M (cosine 0.865 vs floor 0.999); A confirmed the codec is innocent, so the goal narrows to closing that envelope at the SDPA path — most plausibly via FP32 score promotion — so the load banner stops reading `tq_kv = inactive` on APEX and operators don't have to opt out via `HF2Q_USE_DENSE=1`.

## What I found

1. **Gate H passes on APEX-Q5_K_M at HEAD.** Re-running `parity check --tq-quality --prompt sourdough` against the frozen `sourdough_tq_quality.json` fixture:

   | Metric | Floor | Followup ADR (2026-05-16) | HEAD (2026-05-23) | Verdict |
   |---|---:|---:|---:|---:|
   | cosine_mean | 0.999 | 0.865 | **0.999843** | **PASS** |
   | cosine_p1 | 0.990 | 0.628 | **0.998183** | **PASS** |
   | argmax_flip_rate | 0.015 | 0.148 | **0.0050** | **PASS** |
   | ppl_delta | 0.020 | 0.673 | **0.0015** | **PASS** |

   Per-layer cosine_mean range: 0.999483 – 0.999987 across all 30 layers. The followup ADR's "ALL 30 layers in 0.82–0.93 range. None reaches 0.99" claim is no longer reproducible.

2. **Qwen3.6 APEX is also shippable.** `qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf` runs TQ-default-on with the load banner correctly reporting `tq_kv = active (8-bit Lloyd-Max + D1 SRHT, ADR-027 Phase B; F32 K/V dropped at alloc — 3.94× per-slot KV savings)`. Output is coherent at 125 tok/s decode on a sourdough prompt.

3. **The FP32-score-promotion plan was the wrong fix.** The HB SDPA decode kernel (`/opt/mlx-native/src/shaders/flash_attn_vec_tq_hb.metal`) already runs QK^T accumulation, online softmax M/S, and V*w output accumulator in F32. Only Q is stored as `half4` in shared memory. Python simulation on the L0 dump shows:
   - F16 Q error on score: ~10⁻⁵ per position
   - Codec K residual error on score: ~10⁻² per position
   - Ratio: F16 Q error is **~1000× smaller** than codec error

   So even if F32 Q promotion were implemented, it would NOT close the gap. The score path is already F32 where it matters.

4. **The "softmax amplification" diagnosis was conditional, not load-bearing.** With cosine_mean now at 0.999843, the softmax-amplification mechanism is barely engaged. Whatever change closed the gap (most likely in the `03328ee5..9e64df5c` prefill-precision arc that landed shortly after the followup ADR was written) eliminated the upstream precision loss that was feeding the softmax amplifier.

## Banner fix landed in this session

`src/serve/api/engine.rs:2360-2375` previously hardcoded `tq_kv_active: false` for Gemma 4 with a stale comment ("Future iter may unify Gemma + qwen35 surfacing"). Now reads dynamically from `INVESTIGATION_ENV.use_dense` and `INVESTIGATION_ENV.layer_policy`:

```rust
tq_kv_active: !crate::debug::investigation_env::INVESTIGATION_ENV.use_dense
    && !matches!(
        crate::debug::investigation_env::INVESTIGATION_ENV.layer_policy.as_deref(),
        Some("dense_all")
    ),
```

`src/serve/load_info.rs:551-577` extended the banner string to be family-aware:
- `ArchFamily::Qwen35`: `"active (8-bit Lloyd-Max + D1 SRHT, ADR-027 Phase B; F32 K/V dropped at alloc — 3.94× per-slot KV savings)"` (unchanged)
- `ArchFamily::Gemma4`: `"active (8-bit Lloyd-Max + D1 SRHT, ADR-007 Path C; production default; HF2Q_USE_DENSE=1 to opt out)"`
- Other families with `tq_kv_active=true`: `"active (8-bit Lloyd-Max + D1 SRHT)"`

Verified at three states on APEX-Q5_K_M:

| Env state | Banner |
|---|---|
| default-default | `tq_kv = active (8-bit Lloyd-Max + D1 SRHT, ADR-007 Path C; production default; HF2Q_USE_DENSE=1 to opt out)` |
| `HF2Q_USE_DENSE=1` | `tq_kv = inactive` |
| `HF2Q_LAYER_POLICY=dense_all` | `tq_kv = inactive` |

Goal condition "load banner stops reading `tq_kv = inactive` on APEX" — **SATISFIED**.

## Which commit fixed Gate H

577 commits separate 2026-05-16 (followup ADR) from 2026-05-23 (today). The likeliest fix lives in the prefill-precision arc:

- `03328ee5` (2026-05-16 22:40) — flip `HF2Q_NO_FA=1` default to fix batched-prefill argmax drift from BF16 Q in FA prefill kernel
- `5dae1bc7` (later) — wire `HF2Q_FA_F16=1` to F16 D=512 global path
- `be621452` — NO_FA falls back to FA when seq_len < 32
- `9e64df5c` — ADR-032 peer-aligned defaults: HF2Q_FA_F16 on, HF2Q_NO_FA off
- `b630062a` — Gemma 4 module extraction (may have eliminated a stale code path as a side effect)

Full bisect not attempted — the bug is empirically absent and the user goal is met. If a future regression appears, this arc is the first place to look.

## Conclusion

The goal is **achieved at HEAD d990c817 + the banner fix landed this session**:
1. ✅ TQ-default-on works on Gemma 4 APEX-Q5_K_M (Gate H envelope passes by wide margin)
2. ✅ TQ-default-on works on Qwen3.6 APEX (coherent output + correct banner)
3. ✅ Operators don't need to opt out via `HF2Q_USE_DENSE=1` for quality reasons on non-DWQ models
4. ✅ Load banner now reports `tq_kv = active (ADR-007 Path C; production default; HF2Q_USE_DENSE=1 to opt out)` on Gemma 4 default-default and `inactive` only when the operator forces dense

The "softmax amplification" + "FP32 score promotion" plan is **falsified** — the codec is innocent (per A) AND the score path is already F32 where it matters (per G1 profile + Python simulation). The Gate H failure that motivated the goal has been independently fixed by changes in the upstream prefill-precision arc.

**Recommended optional follow-ups** (separable):
- Re-capture the `sourdough_tq_quality.json` fixture from HEAD so future regression watch lives on the tight 0.999843/0.998183/0.0050/0.0015 numbers instead of the looser 0.999672/0.996080/0.0080/0.0014 captured from a different code state.
- Bisect the 577-commit window to identify the exact Gate H fix commit (academic; bug is gone).
