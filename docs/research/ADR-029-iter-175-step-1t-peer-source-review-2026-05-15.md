# ADR-029 iter-175 Step 1t — peer source review: gemma4 + recent Metal commits

**Date**: 2026-05-15
**HEAD**: hf2q `bbeba9a7`, mlx-native `e7a6b33`
**Iteration**: 20 of /loop autonomous

## Summary

Read `/opt/llama.cpp/src/models/gemma4.cpp` (457 LOC) + recent peer commits to `ggml/src/ggml-metal/` to surface any unported optimizations.

**Finding: peer's gemma4 forward path is architecturally IDENTICAL to ours.** No obvious missing big lever. Found one minor candidate: FC-bake of the bin kernel `column-broadcast` flag (peer commit `e4cff0956`), which saves a per-thread modulo when src/dst have the same broadcast-axis shape. Small ALU win.

## Peer's gemma4 per-layer pipeline (lines 200-260 of gemma4.cpp)

Per-layer attention:
1. `build_norm(cur, attn_norm, RMS_NORM)` — pre-attn norm
2. Q proj → reshape → q_norm → RoPE
3. K proj → reshape → k_norm → RoPE
4. V proj → reshape → **`ggml_rms_norm(Vcur, eps)` (NO weight scale)**
5. `build_attn(...)` (Flash Attention)
6. `build_norm(cur, attn_post_norm, RMS_NORM)` — post-attn norm
7. `ggml_add(cur, inpL)` — residual

Per-layer MoE (gemma4 has MoE for some layers):
1. `build_norm(attn_out, ffn_norm)` — MLP norm
2. `build_ffn(cur_mlp, up, gate, down)` — shared expert FFN with GELU
3. `build_norm(cur_mlp, ffn_post_norm_1)`
4. `build_norm(attn_out, ffn_pre_norm_2)` — MoE norm
5. Custom MoE logits: `ggml_rms_norm(attn_out) → ggml_scale(1/sqrt(n_embd)) → ggml_mul(gate_inp_s) → matmul(gate_inp)`
6. `build_moe_ffn(...)` — MoE expert FFN with GELU
7. `build_norm(cur_moe, ffn_post_norm_2)`

Compared to hf2q's `forward_mlx.rs`: same operations, same order, same gemma4-specific patterns (V-norm with no scale, MoE-router-on-attn_out, dual post-FF norms). hf2q fuses more ops (head_norm_rope_v2, norm_add_v2, post_ff_norm2_endlayer_v2) — iter-1 H6 / iter-105 / iter-107 / iter-175 1o re-bench all confirm hf2q's fusion granularity is at local optimum on Apple Metal at gemma4 shapes.

## Recent peer commits (filtered to potentially-applicable optimizations)

| Commit | Description | Applicable to hf2q? |
|---|---|---|
| `da4495332` | metal: FC-promote mul_mv/mul_mm batch divisors | **DONE** — ported as iter-162 H93 (+1.08% multi-regime) |
| `d1649047a` | metal: separate Metal Tensor API matmul2d path | NO — only affects MUL_MAT (prefill m>>1); we use mul_mm_tensor_v2 (already separate) for prefill |
| `e4cff0956` | metal: avoid divisions in bin kernel via FC `column-broadcast` | **CANDIDATE** — small ALU win on elementwise; see below |
| `a71b56613` | metal: avoid modulus in bin kernel when not broadcasting | predecessor to e4cff0956 |
| `9f5f0e689` | Gemma4_26B_A4B_NVFP4 model support | NO — quant format, not relevant to Q5_K_M |
| `342d6125b` | FA HSK=512 HSV=512 instantiation | NO — gemma4 uses HSK=HSV=256 (already supported) |
| `e22cd0aa1` | mul_mv_ext to BF16, Q2_K, Q3_K | NO — gemma4 is Q5_K_M, not Q2/Q3 |
| `8635e221c` | Metal event synchronization fix | NO — internal correctness |

## Candidate: bin kernel FC-CB port (small ALU win)

Peer commit `e4cff0956`:
- Adds new FC `FC_bin_cb` at function_constant slot `FC_BIN + 3`
- Switches `i10 = i0 % args.ne10` to `i10 = FC_CB ? i0 % args.ne10 : i0`
- When `args.ne0 == args.ne10` (no broadcasting), the modulo becomes constexpr false → compiler folds the branch to just `i10 = i0` (one instruction saved per element)

**For hf2q applicability**: our `elementwise_add` and similar bin kernels live in `mlx-native/src/shaders/elementwise.metal`. Need to check if they already use this pattern.

```bash
# In a future iter:
grep -n "ne10\|i10\|broadcast" /opt/mlx-native/src/shaders/elementwise.metal
```

If they already FC-bake or don't broadcast → port is no-op. If they do per-element modulo on every dispatch → port would save 1 ALU/element on the non-broadcasting hot path.

Expected gain: **<0.5% wall** (elementwise ops are <2% of decode wall per Step 1h). Not iter-175 closure but small consistent win.

## What this review DID NOT find

- ❌ No big algorithmic difference between hf2q and peer at the architecture level on gemma4
- ❌ No recent peer commit that we missed and could deliver multi-percent gain
- ❌ No magic optimization that closes the residual 6-8% gap

## Implication for iter-175 closure

The peer-source review CONFIRMS the structural-floor interpretation. Peer's gemma4 implementation is architecturally equivalent to ours. The ~6-8% peer-FA decode gap on M5 Max is from:
- Apple GPU scheduler quirks at our specific kernel granularity (per iter-1 H6 / iter-105 / iter-175 1o falsifications of fusion-direction levers)
- Per-dispatch GPU compile/scheduler differences not visible at the source level

The remaining /loop-tractable work:
1. Port `e4cff0956` bin-kernel FC-CB (if applicable; <0.5% win)
2. R3: apply H-E + smart-barrier pattern to non-gemma4 paths (R4 from synthesis)
3. Operator-only Apple Instruments investigation

## Cross-references

- Peer gemma4 source: `/opt/llama.cpp/src/models/gemma4.cpp`
- Peer recent commits filtered: `cd /opt/llama.cpp && git log --since=2026-03-01 -- ggml/src/ggml-metal/`
- iter-1 H6 fusion falsification (re-confirmed Step 1o): `src/debug/investigation_env.rs:665-686`
- iter-175 full state: `docs/research/ADR-029-iter-175-FULL-STATE-2026-05-15.md`
