# ADR-063 Stage 2 follow-up: a precise tiled warm attention for Gemma 4

Status: implemented behind `HF2Q_GEMMA_WARM_TILED_ATTENTION=precise`
(experimental, default unchanged); hardware gates not yet run. A full-precision
variant needs the mlx-native patch in
`docs/research/adr063-gemma-tiled-precise-mlx-native.patch`.

## Question

`TiledF16` (tiled flash attention for warm resumes) changed a decision against
`VecHybrid`: at the canonical coding-task token (agent-3, turn 2, after
`if port == 0`) the top-2 log-probs moved from −0.149 / −1.978 (vec) to
−0.368 / −1.179 (tiled), and repetition penalty 1.05 flipped the choice
(`gemma-divergence-v1/solo-vs-concurrent-by-tiled.json`,
`bisect-summary.json`). The question: is there a semantic mismatch (window,
mask, scale, softcap) or only rounding, and can a tiled variant match
vec-hybrid decisions (0.10-nat near-tie policy) at tiled speed?

mlx-native citations are to 0.15.15 (`/opt/mlx-native`, byte-identical to the
pinned crate for every file cited here). hf2q citations are to this worktree.

## The two routes, per chunk

Shared by both routes (`src/serve/gemma_live_attention.rs`): Q/K/V arrive as
F32 normed, RoPE'd rows. The suffix K is written to the F16 cache and V is
TQ-HB-quantized (FWHT-rotated, byte codes and per-position F32 norms) exactly
as decode stores it (`write_kv_rows`, :490-:555). Sliding layers stage
`min(start, cap - 1)` chronological history rows plus the chunk
(`linearize_rows` + `write_kv_rows`, :665-:699). Both routes then attend the same K bytes and V
codes, and the layer ends with the same F32 FWHT sign undo (:412-:421). The
O projection reads the F32 attention rows (`forward_prefill_batched.rs:3978-3990`).

- `VecHybrid` (:720-:776): `flash_attn_vec_hybrid_batched`, one threadgroup
  per (query, head). Sliding chunks use `mask_type = 3` with per-row stage
  positions; global chunks use `mask_type = 1` with the graft-shifted RoPE
  positions (`forward_prefill_batched.rs:1554-1556`), `scale = 1.0`,
  `softcap = 0.0`.
- `TiledF16` (`encode_tiled_chunk`, :807-:1035): compact K to dense F16
  (:861-:869), dequantize V to F16 (`dispatch_tq_dequantize_hb_kv_seq_f16`,
  :871-:886), permute Q F32 → F16 (:904-:914). Sliding: build the F16 SWA mask
  (`window = sliding_window`, causal, `q_abs_offset = history`, :917-:928),
  classify tiles (`blk`), `dispatch_flash_attn_prefill_f16_d256_with_blk`
  (:956-:981, `scale = 1.0` on the default route). Global:
  `dispatch_flash_attn_prefill_f16_d512_resume` (`scale = 1.0`, causal,
  `q_offset_in_k = graft + start`, :988-:1009). Then O F16 → F32 cast
  (:1012-:1021) and permute back (:1023-:1033).

## Rounding and semantic map

| Quantity | VecHybrid | TiledF16 sliding (d256) | TiledF16 global (d512) |
|---|---|---|---|
| Q | F32 → `half4` once in threadgroup memory (`flash_attn_vec_hybrid.metal:884-885`) | F32 → F16 once (`elementwise.metal:553`), then **multiplied in F16 by `half(scale·log2e)`** (`flash_attn_prefill.metal:1250`, applied at :1299): a second rounding per element, and the factor itself rounds 1.4426950 → 1.4423828 | F32 → F16 once, stored unscaled (`flash_attn_prefill_d512.metal:442-447`) |
| K | F16 cache, `half4` → `float4` (:1003-1004) | same F16 bytes, compacted (copy) | same F16 bytes, compacted (copy) |
| Scores Q·K | F32 dot, `fma(mqk, scale, mask)` (:1007, :1011) | F32 MMA accumulators (`flash_attn_prefill.metal:1270-1274`, :1431), base-2 domain | F32 MMA (`flash_attn_prefill_d512.metal:668`), `× scale` in F32 after Q·K (:788) |
| Softmax | F32 online, natural `exp` (:1016-1028) | F32 online, `fast::exp2` (:1575) | F32 online, natural `exp` (:833-838) |
| Mask | additive −65504 for out-of-window / beyond `pos + 1` (:917-920, :934-947) | F16 additive `−inf`/0, ×log2e (:1533); `blk` skip/no-op tiles (`flash_attn_prefill_blk.metal:252-258`) | in-kernel causal, `q_abs = qL_off + row` (:807-812) |
| P (weights) | F32 | F32 (Stile) | F32 (`ss`, :844, :940-942) |
| V | dequantized in F32 per use: `centroid · norm / √256` (d256) or `norm / sf` (d512) (:1069-1095) | **dequantized to F16** (`tq_dequantize_kv.metal:425`), F16 → F32 at the MMA | **dequantized to F16**, F16 at the MMA (`flash_attn_prefill_d512.metal:945`) |
| O accumulator | F32 threadgroup (:892-896, :1099-1101); split-K reduce in F32 | F32 registers (Otile) | F32 threadgroup (`so`, :904) |
| O store | F32 (:1172-1176) | **F16** (`flash_attn_prefill.metal:1647/1649`), cast back to F32 | **F16** (`flash_attn_prefill_d512.metal:1054`), cast back to F32 |
| Reduction order | per-query, 32-key chunks, NSG/NWG split | 32×16 tiles | 8×64 tiles |

Downstream of both routes, the O projection's tensor-MM stages its F32 input
as `half` (`quantized_matmul_mm_tensor.metal:16-21`, :394-395), so the F32 O of
vec-hybrid is rounded to F16 there too. The tiled O is therefore rounded twice:
once in the FWHT-rotated domain by the kernel, then again after the F32 sign
undo by the projection. `[INFERENCE]` from the kernel source: the Gemma live
O projection takes this tensor-MM path for these row counts.

Semantic checks, all equal on both routes:
- Sliding window: vec keeps keys with `pos + 1 − W ≤ k ≤ pos` (stage
  coordinates); the tiled mask keeps `k ≤ q_abs` and `q_abs − k < W`
  (`flash_attn_prefill_mask.metal:139-141`) with `q_abs = history + row`.
  Same W keys, same `sliding_window`.
- Ring/stage: both read the same chronological stage; the tiled compaction
  copies rows `[0, kv_seq_len)` of it.
- Global causal offset: `q_offset_in_k = graft + start` equals the vec
  `positions[row] = graft + start + i` (`forward_prefill_batched.rs:1554-1556`).
- Scale and softcap: Gemma 4 has no attention logit soft cap (only the final
  logit cap); both routes pass `softcap = 0` / disabled and `scale = 1.0`.
  The d512 kernel applies `scale` in F32 with natural `exp`, consistently with
  its host contract.
- GQA: `kv_head = head / (nh / nkv)` in both.
- V dequantization: identical formula and identical codebooks (the 5-, 6- and
  8-bit tables in `tq_dequantize_kv.metal` and `flash_attn_vec_hybrid.metal`
  compare equal).

No semantic mismatch was found.

## Diagnosis

Confidence: high that the route difference is numeric, not semantic (every
mask/window/offset/scale/softcap item above checks equal by source reading).
Medium `[INFERENCE]` on the ranking below; it is unmeasured.

1. Sliding layers (25 of 30): the d256 kernel pre-scales Q in F16. The factor
   `half(1.0 · log2 e)` is 1.4423828, 2.16e-4 below log2 e, so every sliding
   softmax runs on `0.999784 · q·k`: a systematic temperature of 1.000216 on
   every row, head and sliding layer, plus a second per-element rounding of Q.
   The attention output moves by about `δ · Cov_p(s, v)` with δ = 2.16e-4,
   comparable to or larger than one F16 rounding of O, and it does not
   average out across layers. vec-hybrid has neither. (The d512 kernel's own
   comments, `flash_attn_prefill_d512.metal:442-450`, record the same Q
   pre-scale as a measured drift source on Gemma global layers; that is why
   d512 scales after Q·K.)
2. O stored as F16 (both tiled kernels): a second, independent F16 rounding
   before the projection's own.
3. V dequantized to F16: 2^-12 relative, averaged by the softmax weights; the
   smallest of the three.
4. Reduction order: F32, negligible.

The stored localization (`adr063-gemma-four-agent-reuse.md`, Stage 2 table:
layer-0 attention relative Δ 3.6e-4, residual Δ ×40 at layer 1 while attention
Δ stays 0.014) fits a small kernel-level perturbation amplified by near-tied
MoE routing. No variant that is not bitwise equal to vec-hybrid can rule
routing flips out; the target is to shrink the perturbation enough that
decisions match at the 0.10-nat policy.

Secondary finding, outside this route: the cold prefill uses the same d256
kernel (`forward_prefill_batched.rs:3272-3460`). Single-sequence cold prefill
runs it in F16 (the same 1.000216 temperature); the cold multi-sequence prefill
runs it in BF16, where `bf16(log2 e)` is 1.4453125 (+0.18%, temperature
0.9982). Both routes resume from those histories, so this does not explain the
tiled-vs-vec difference; the mlx-native patch below would let the cold path
drop it too.

## Implemented: `HF2Q_GEMMA_WARM_TILED_ATTENTION=precise`

`GemmaLiveKernel::TiledF16Precise` (`src/inference/models/gemma4/model.rs`)
is the tiled route with the item-1 fix, using only kernels mlx-native 0.15.15
already ships:
- sliding chunks: `scalar_mul_f32(Q, log2 e)` into the O32 scratch (unused
  until the O cast), then the usual F32 → F16 permute, so Q is rounded to F16
  once; the d256 dispatch passes `scale = ln 2`
  (`SLIDING_PRESCALED_SCALE`), for which `half(ln 2 · log2 e)` is exactly 1.0
  (`LN_2 * LOG2_E == 1.0` in F32), making the in-kernel Q scale the identity
  and the softmax `exp2(log2 e · q·k)`;
- global chunks, chunking, ring staging, the 32-row vec-hybrid threshold,
  `LiveRoute` (single slot == joint cohort) and the FWHT undo are those of
  `TiledF16`.

Cost: one Q-sized F32 elementwise pass and one barrier per sliding chunk.
Defaults are unchanged; `1`/`on`/`true` and unset still select `TiledF16`,
and `0` selects `VecHybrid`.

Files: `src/serve/gemma_live_attention.rs` (switch, route, Q prescale, unit
tests including the exact-factor test), `src/inference/models/gemma4/model.rs`
(variant, `is_tiled`), `src/serve/api/engine_gemma_warm_tiled_tests.rs`
(envelope and localization per tiled arm, 0.10-nat policy),
`src/serve/api/engine_gemma_warm_batch_tests.rs` (joint == per-slot on the
precise route), `scripts/serve_gemma4_opencode.sh` and the Q5 `gemma4` case
(`WARM_TILED_ATTENTION=precise`), operator docs.

## Full precision needs mlx-native

Items 2 and 3 have no hf2q-side fix: every tiled entry point in 0.15.15 stores
O in its I/O type, and there is no F32 instantiation (the d256 Q tile alone
would fill 32 KB of threadgroup memory). The proposed patch
(`adr063-gemma-tiled-precise-mlx-native.patch`, applies to 0.15.15 with
`patch -p1`; not compiled here):
- `flash_attn_prefill.metal`: template parameters `OutT` (default `T`) and
  `ScaleScores` (default `false`); with `ScaleScores` the Q tile is not
  scaled and the F32 scores are multiplied by `scale · log2 e` right after
  Q·K^T; O stores as `OutT`. New entry point `flash_attn_prefill_f16_d256_o32`
  (F16 Q/K/V/mask, F32 O, scaled scores). Existing entry points are unchanged
  (defaults reproduce the current code).
- `flash_attn_prefill_d512.metal`: template parameter `OutT`; new entry point
  `flash_attn_prefill_d512_f16_o32` (F32 O store). Existing entry points pass
  `OutT = T`.
- `dispatch_flash_attn_prefill_f16_d256_with_blk` and
  `dispatch_flash_attn_prefill_f16_d512_resume` accept an F32 `out` and then
  select the `_o32` entry point; registration lists and name tests updated.

hf2q wiring once that ships (mlx-native 0.15.16 or later): on
`TiledF16Precise`, drop the Q prescale (pass the F16 Q and `scale = 1.0`; the
kernel scales in F32), pass the O32 scratch as `out` to both kernels, and
delete the O16 scratch and the F16 → F32 cast. That removes the double O
rounding and saves one dispatch per chunk; only the F16 V (item 3) and the
reduction order remain different from vec-hybrid. V stays F16 because the d256
V tile shares the 32 KB threadgroup budget; the d512 kernel reads V from device
memory and could take F32 V later if the gates show V matters.

## Gates and commands (root)

Build and model-free checks:

```sh
cd /opt/hf2q-worktrees/exp-gemma-tiled-precise
cargo build --locked --release --bin hf2q
cargo test --locked --bin hf2q gemma_live_attention
python3 -B -m unittest discover -s scripts/four_agent -p 'test_q5_calibration.py'
bash scripts/audit_hf2q_env_surface.sh | grep -E 'HF2Q_GEMMA_(WARM_TILED_ATTENTION|TILED_LIVE)'
```

Real-artifact gates, one process each:

```sh
GGUF=/abs/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf
for t in \
  serve::api::engine::gemma_warm_tiled_tests::gemma_warm_tiled_precise_attention_decision_envelope \
  serve::api::engine::gemma_warm_tiled_tests::gemma_warm_tiled_attention_decision_envelope \
  serve::api::engine::gemma_warm_tiled_tests::gemma_warm_tiled_precise_attention_layer_localization \
  serve::api::engine::gemma_warm_tiled_tests::gemma_warm_tiled_attention_layer_localization \
  serve::api::engine::gemma_warm_batch_tests::gemma_joint_warm_prefill_matches_per_slot_bitwise_tiled_precise; do
  HF2Q_GEMMA4_GGUF="$GGUF" cargo test --locked --release --bin hf2q "$t" -- --exact --ignored --nocapture --test-threads=1
done
```

Read:
- `GEMMA_WARM_TILED_ENVELOPE_LANE kernel=…` per lane: `max_abs_logprob_diff`
  (whole vocabulary), `max_abs_logprob_diff_top2` (the vec-hybrid top-2),
  `top1_agree`, both log-prob pairs and `margin_vec_nat`. The summary
  `GEMMA_WARM_TILED_ENVELOPE kernel=…` fails when top-1 differs and the
  vec-hybrid margin is at least 0.10 nat. Run both arms: the `TiledF16` line
  is the baseline the precise arm must improve on.
- `GEMMA_WARM_TILED_LAYER kernel=…`: layer 0 sees identical inputs, so its
  `attn_rel` is the pure kernel difference (3.6e-4 measured for `TiledF16`).
  The precise arm should lower it; what remains is the F16 V/O floor.
- `GEMMA_JOINT_WARM_BITWISE_COMPLETE kernel=TiledF16Precise`.

Replay A/B of the divergent decision: start the server once per arm with the
same text-model hard link the gemma-divergence-v1 run served, replay the same
agent-0/agent-3 turn-2 requests with the harness that produced
`gemma-divergence-v1`, and compare the decision token's top-2 log-probs and
chosen token with the vec arm (agent-3: " {" −0.149 / " ||" −1.978; agent-0:
" {" −0.058 / " ||" −2.877):

```sh
E=/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023
for ARM in 0 precise 1; do
  MODEL=$E/gemma-divergence-v1/text-model/gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf \
  MMPROJ=$E/gemma-divergence-v1/text-only-no-external-projector.gguf \
  HF2Q_BIN=/opt/hf2q-worktrees/exp-gemma-tiled-precise/target/release/hf2q \
  LCP_CAPACITY=8g KV_CACHE_BUDGET_BYTES=51539607552 REP_PENALTY=1.05 MAX_SLOTS=4 \
  HOST=127.0.0.1 PORT=18244 WARM_TILED_ATTENTION=$ARM \
    bash scripts/serve_gemma4_opencode.sh 2> "$E/gemma-divergence-v1/server-precise-ab-$ARM.stderr"
  # replay the agent-0 / agent-3 turn-2 requests (solo and the 4-way concurrent trial), then stop the server
done
```

Each `server-precise-ab-$ARM.stderr` must log `Gemma warm live attention
kernel kernel=VecHybrid|TiledF16Precise|TiledF16`, and warm banners
`live=tiled-f16-precise (vec-hybrid below 32 rows)` on the precise arm.

Speed and task A/B (Q5 `gemma4` case, ABAB with a thermal gap, at least 3 per
arm; the precise arm against the vec-hybrid arm):

```sh
HF2Q_Q5_GEMMA_WARM_TILED_ATTENTION=precise python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" --output "$Q5_ROOT/q5-gemma4-warm-tiled-precise-$(date -u +%Y%m%dT%H%M%SZ)"
HF2Q_Q5_GEMMA_WARM_TILED_ATTENTION=0 python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
  --negative-control-receipt "$NC_RECEIPT" --output "$Q5_ROOT/q5-gemma4-warm-tiled-off-$(date -u +%Y%m%dT%H%M%SZ)"
```

Re-enable rule: make `precise` the warm default only if both precise
envelope gates hold (no top-1 flip at a vec-hybrid margin ≥ 0.10 nat), the
precise joint bitwise and determinism gates pass, the replay keeps the
vec-hybrid decision at the divergence token, and the Q5 makespan keeps the
tiled saving (A −7.8% against vec-hybrid in the Stage 2 A/B).
