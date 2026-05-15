# ADR-029 iter-175 Step 1af — shape correction: Steps 1ad/1ae used synthetic shape, not gemma4 production

**Date**: 2026-05-15
**HEAD**: hf2q `b5744493`, mlx-native `4905f09`
**Iteration**: 37 of /loop autonomous

## The error

Steps 1ad and 1ae assumed gemma4 MoE down_exps shape was **K=8192, N=2816,
Q6_K**.  Per the actual gemma4-26B-A4B-APEX-Q5_K_M model file inspected via
the (just-extended) `dump_gguf_types --all`:

```
blk.0.attn_k.weight          type=Q6_K  shape=[2048, 2816]
blk.0.attn_norm.weight       type=F32   shape=[2816]
blk.0.attn_output.weight     type=Q6_K  shape=[2816, 4096]
blk.0.attn_q.weight          type=Q6_K  shape=[4096, 2816]
blk.0.attn_v.weight          type=Q6_K  shape=[2048, 2816]
blk.0.ffn_down.weight        type=Q8_0  shape=[2816, 2112]
blk.0.ffn_down_exps.weight   type=Q8_0  shape=[128, 2816, 704]
blk.0.ffn_gate.weight        type=Q6_K  shape=[2112, 2816]
blk.0.ffn_gate_inp.weight    type=F32   shape=[128, 2816]
blk.0.ffn_gate_up_exps.weight type=Q6_K shape=[128, 1408, 2816]
blk.0.ffn_norm.weight        type=F32   shape=[2816]
```

**Actual MoE shapes**:
- `ffn_gate_up_exps`: **Q6_K**, per-expert [N=1408, K=2816]  ← gate + up concatenated
- `ffn_down_exps`: **Q8_0** (NOT Q6_K), per-expert [N=2816, K=704]

Plus a DENSE FFN side that runs alongside MoE:
- `ffn_gate`: Q6_K [N=2112, K=2816]
- `ffn_down`: Q8_0 [N=2816, K=2112]
- (no separate `ffn_up`; likely gate-up combined elsewhere)

ATTN tensors (all Q6_K):
- attn_q [4096, 2816]: 16 heads × 256 head_dim, hidden=2816
- attn_k [2048, 2816]: 8 kv-heads × 256, GQA factor 2
- attn_v [2048, 2816]: same as K
- attn_output [2816, 4096]: O projection

## What Step 1ad/1ae findings actually mean

Steps 1ad and 1ae used K=8192, N=2816 — a **synthetic shape** that doesn't
match any actual gemma4 tensor.  The qualitative findings still hold AT THAT
SHAPE:

* **Step 1ad** (_id vs non-_id, synthetic K=8192/N=2816): _id is -36% per-row,
  better GPU fill from 8× more TGs.  Still valid as a per-shape datum showing
  amortization is beneficial when single-matvec is under-occupied.
* **Step 1ae** (hf2q _id vs peer _id, synthetic K=8192/N=2816): hf2q wins
  3.89%.  Still valid as evidence that hf2q's Q6_K matvec algorithm is
  competitive with peer's AT THAT SHAPE.

But neither directly answers "what is the per-dispatch time of the actual
gemma4 down_exps kernel?" — because down_exps is Q8_0, not Q6_K, and
K=704 not 8192.

## The correct bench plan

Three actual hot kernels to bench at correct shapes:

| Kernel | Type | K | N | top_k | Dispatches/tok |
|---|---|---|---|---|---|
| `kernel_mul_mv_id_q6_K_f32_nr2` (gate_up_exps) | Q6_K | 2816 | 1408 | 8 | 1/layer × 30 = 30 |
| `kernel_mul_mv_id_q8_0_f32` (down_exps) | Q8_0 | 704 | 2816 | 8 | 1/layer × 30 = 30 |
| `kernel_mul_mv_q6_K_f32_v2` (Q/K/V/O proj) | Q6_K | 2816 | 2048-4096 | — | 4/layer × 30 = 120 |

Plus norms (rms_norm_f32_v2) at hidden=2816.

## What I'm doing about it

This iteration:
- Documenting the shape error honestly (this Step 1af artifact).
- Extending `dump_gguf_types` with `--all` flag so future investigators can
  inspect any gguf's shapes.
- Planning to re-bench at correct shapes in a future iter (next /loop fire).

This is the mantra in action: "Never make assumptions. Always dive deep
and ensure you know the problem you're solving."  I assumed a plausible
shape; the actual shape is different; I'm correcting.

## What this DOES NOT do

- Does NOT invalidate Step 1aa's canonical wall ratio 0.9365× (that was
  end-to-end production measurement, no shape assumption).
- Does NOT invalidate Step 1z's canonical baseline 95.30 t/s.
- Does NOT invalidate Step 1ac's prefill 1.0370× (also end-to-end).
- Does NOT invalidate any of Steps 1-1aa's lever falsifications (each
  tested a specific env var or code change at production wall — no
  synthetic kernel shapes).

What IS provisional now:
- Step 1ad's specific number (-36% per-row) — qualitatively still
  "amortization is good at this kind of shape" but the magnitude is for
  a non-gemma4 shape.
- Step 1ae's specific number (hf2q +3.89% vs peer) — qualitatively still
  "hf2q Q6_K matvec is competitive" but at a different shape than gemma4
  actually uses.

## Cross-references

* Step 1ad: `docs/research/ADR-029-iter-175-step-1ad-id-kernel-perdispatch-2026-05-15.md`
* Step 1ae: `docs/research/ADR-029-iter-175-step-1ae-peer-vs-hf2q-id-kernel-2026-05-15.md`
* Step 1ab (similar honest correction): `docs/research/ADR-029-iter-175-step-1ab-prefill-methodology-2026-05-15.md`
* Mantra: `~/Documents/mantra.txt`
