# B: Dense non-determinism — cannot reproduce at HEAD d990c817

**Date:** 2026-05-23
**HEAD:** d990c817
**EXP-3 baseline:** 2026-05-16 at HEAD d29da003 (577 commits behind today)
**Model:** `models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf`

## EXP-3's finding (2026-05-16)

`docs/EXP-3-tq-default-on-end-to-end-2026-05-16.md` reported dense-mode non-determinism on `temp=0` greedy across 3 runs:
- `short_hello`: cp(run1, run2) = 8 / 16-20 b (~50% divergence)
- `sourdough`: cp(run1, run2) = 66 / 3747 b (~1.8% divergence)
- `sliding_wrap`: cp(run1, run2) = 199 / 2299 b (~9% divergence)

TQ-mode was deterministic on short prompts but also non-deterministic on `sliding_wrap` (cp=2089/2417). The hypothesized suspects were `HF2Q_BATCHED_PREFILL=1`, rayon CPU-side reductions, GPU atomics, and sliding-window FP order.

## Repro at HEAD d990c817

Used the exact EXP-3 invocation (`parity capture --prompt all` in a single process, `HF2Q_USE_DENSE=1`, `HF2Q_BATCHED_PREFILL=1` inherited from env):

```
Dense, 3 runs, --prompt all:
  short_hello     lens=[16, 16, 16]       cp(1,2)=16   cp(1,3)=16   cp(2,3)=16   PASS
  sourdough       lens=[3782, 3782, 3782] cp(1,2)=3782 cp(1,3)=3782 cp(2,3)=3782 PASS
  sliding_wrap    lens=[2380, 2380, 2380] cp(1,2)=2380 cp(1,3)=2380 cp(2,3)=2380 PASS

TQ, 3 runs, --prompt all:
  short_hello     lens=[16, 16, 16]       cp(1,2)=16   cp(1,3)=16   cp(2,3)=16   PASS
  sourdough       lens=[3773, 3773, 3773] cp(1,2)=3773 cp(1,3)=3773 cp(2,3)=3773 PASS
  sliding_wrap    lens=[2378, 2378, 2378] cp(1,2)=2378 cp(1,3)=2378 cp(2,3)=2378 PASS
```

**Both dense and TQ are byte-deterministic across 3 runs on all 3 prompts.** The EXP-3 non-determinism is fixed.

## TQ vs Dense at HEAD d990c817

Still diverge at byte level (this is expected — they're different SDPA paths) but the divergence point shifted:

| prompt | EXP-3 cp(TQ,Dense) | HEAD cp(TQ,Dense) | change |
|---|---:|---:|---|
| short_hello | 8 | 16 (==len) | now identical |
| sourdough | 66 | 577 | ~9× later divergence |
| sliding_wrap | 199 | 309 | ~1.5× later divergence |

## Which commit fixed it?

577 commits separate EXP-3 from today. The single highest-suspicion candidate was `03328ee5` (2026-05-16 22:40, same day as EXP-3) — "fix(prefill): flip HF2Q_NO_FA default ON to fix batched-prefill argmax drift". Root cause was BF16-Q in FA prefill kernel for D=256, with 0.39% per-operand error accumulating across 256-element dot products × 30 layers into argmax flips.

Tested the hypothesis by forcing `HF2Q_NO_FA=0 HF2Q_FA_F16=0` (the pre-fix state) — dense remained byte-identical across 3 runs. So `03328ee5` alone is not load-bearing for run-to-run determinism — that commit fixed model-vs-llama argmax DRIFT (cumulative error pushing outputs over decision boundaries), not run-to-run variance.

A full bisect would identify the exact fix but is not load-bearing for the user's question — the bug is gone.

## Most plausible candidates from the commit-range scan

Suspect-shaped commits in the 577-commit window that touch the relevant code paths:

```
03328ee5 fix(prefill): flip HF2Q_NO_FA default ON to fix batched-prefill argmax drift
be621452 fix(prefill): NO_FA default falls back to FA when seq_len < 32
5dae1bc7 feat(prefill): wire HF2Q_FA_F16=1 to F16 D=512 global path (Bug A fix)
9e64df5c fix(coherence): ADR-032 peer-aligned defaults — HF2Q_FA_F16 on, HF2Q_NO_FA off
95546adf fix(decode): restore FWHT on hybrid_kv V quantization — Phase 10e.5 was unsafe
b630062a feat(adr-038 step 3): rename forward_mlx.rs → gemma4/ tree (atomic; Path A)
```

The `b630062a` rename (which restructured the entire Gemma 4 forward path from a single `src/serve/forward_mlx.rs` file into `src/inference/models/gemma4/` tree) is a likely catch-all — any latent non-determinism in the old monolith may have been eliminated as a side effect of the cleaner extraction.

## Verdict

The dense-mode non-determinism observed in EXP-3 is **no longer reproducible** at HEAD d990c817. The fix appears to be a side effect of one of the prefill correctness fixes (likely in the `03328ee5..9e64df5c` arc) or the `b630062a` Gemma 4 module extraction.

**No action needed.** The followup observation from EXP-3 — "newly-deterministic TQ output that's also byte-identical to the frozen `_hf2q.txt` reference" — still holds today. The shipping recommendation is unchanged: TQ-8-bit as default with `HF2Q_USE_DENSE=1` as the byte-exact opt-out.

## Caveats

- 3 runs each is a low-N test. The EXP-3 nondeterminism was visible at 3 runs because the divergence was large (50% on short_hello). A subtle remaining non-determinism could surface only at higher N.
- Did not test other models, only `gemma4-ara-2pass-APEX-Q5_K_M.gguf`. The original EXP-3 was on this exact model.
- Did not test temperature > 0 sampling. EXP-3 was greedy-only; that's what's reproduced here.
- A full bisect would identify the exact fix commit. Not done in this session — the bug being absent at HEAD is sufficient.
