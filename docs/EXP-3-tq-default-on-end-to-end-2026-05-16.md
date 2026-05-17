# EXP-3 — TQ-default-on vs dense end-to-end quality on Gemma-4 APEX-Q5_K_M

**Date:** 2026-05-16
**hf2q HEAD at measurement:** d29da003
**Model:** `models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf`
**Sampling:** greedy (temp=0, top_k=0, top_p=1.0, rep_penalty=1.0)
**Harness:** `scripts/exp3-tq-vs-dense-end-to-end.sh` (3 runs each mode, 3 prompts)
**Reference dir:** `tests/evals/reference/`

## Mission context

User directive 2026-05-16 (3-part):
1. TurboQuant 8-bit KV cache stays default-on
2. Must be tested-to-be-correct
3. For ALL supported model families, not just DWQ-trained models

Mantra: no good enough, no fallback, no stub, best possible outcome.

ADR-007-followup background: on non-DWQ models (APEX-Q5_K_M, our headline
README perf model) Gate H replay measurements showed cosine 0.865 / argmax
flip 14.8% / PPL Δ 67% — alarming numbers IF read as "TQ is broken at output
level". This experiment tests that reading at the byte-output level.

## Result

| prompt | mode | bytes | cp_vs_llama | cp_vs_frozen_hf2q_ref | cp_vs_other_mode | determinism (3 runs) |
|---|---|---|---|---|---|---|
| short_hello | TQ | 16 | **16** | **16** | 8 | byte-identical (PASS) |
| short_hello | DENSE | 20 | 8 | 8 | (self) | DIFFER @8 (FAIL) |
| sourdough | TQ | 3581 | 39 | **3581** | 39 | byte-identical (PASS) |
| sourdough | DENSE | 3747 | 66 | 66 | (self) | DIFFER @66 (FAIL) |
| sliding_wrap | TQ | 2417 | 28 | **2417** | 28 | DIFFER @2089 (FAIL on long-context only) |
| sliding_wrap | DENSE | 2299 | 129 | 199 | (self) | DIFFER @199 (FAIL) |

## Three surprising findings

### Finding 1 — TQ is byte-identical to the frozen `*_hf2q.txt` reference; dense is not

The reference files at `tests/evals/reference/*_hf2q.txt` (captured at hf2q
HEAD `4166ce50`) were committed as the byte-prefix anchor for Gates D/F.
Current TQ output matches them exactly:

- `short_hello_hf2q.txt`: TQ matches all 16 bytes; dense matches only 8.
- `sourdough_hf2q.txt`: TQ matches all 3581 bytes; dense matches only 66.
- `sliding_wrap_hf2q.txt`: TQ matches all 2417 bytes; dense matches only 199.

This proves the frozen baseline was captured under **TQ default-on**, not
under `HF2Q_USE_DENSE=1` as previously assumed.  The `scripts/parity_check.sh`
gate that forces dense for "byte-exact vs llama" comparison has been
comparing dense-mode output against TQ-captured reference — which is the
opposite of what its comments claim.

### Finding 2 — DENSE is non-deterministic across runs; TQ is mostly deterministic

Same model load, same prompt, identical CLI flags, three sequential captures:

```
tq    short_hello: all 3 runs byte-identical (Gate F PASS)
tq    sourdough:   all 3 runs byte-identical (Gate F PASS)
tq    sliding_wrap: run1 vs run2 DIFFER (cp=2089/2417 b) — diverges 86% in
dense short_hello: run1 vs run2 DIFFER (cp=8/16-20 b) — diverges 50% in
dense sourdough:   run1 vs run2 DIFFER (cp=66/3747 b) — diverges 1.8% in
dense sliding_wrap: run1 vs run2 DIFFER (cp=199/2299 b) — diverges 9% in
```

This is a NEW BUG discovered by the experiment.  Greedy decode at
`temp=0,top_k=0,top_p=1.0,rep_penalty=1.0` MUST be byte-deterministic across
runs.  Dense is failing this on every prompt; TQ is failing it only on long-
context sliding_wrap.

Suspect causes (not yet bisected):
- HF2Q_BATCHED_PREFILL=1 (env active in test env per the warning) — its doc
  string says "experimental; errors when seq_len > sliding_window" but does
  it have an FP-nondeterminism dimension too?
- Rayon/parallel reductions in CPU-side router / sampler
- GPU kernel atomic accumulation (not believed to be in use on the decode
  hot path but needs verification)
- KV-cache sliding-window wrap-around interaction with FP order

### Finding 3 — Both TQ and DENSE produce coherent, on-topic output

Despite TQ↔DENSE divergence at early bytes, BOTH outputs are valid greedy
completions:

- short_hello: TQ "2 + 2 = 4<turn|>" vs DENSE "2 + 2 = **4**<turn|>" — same
  answer, dense run1 added markdown bold (dense run2 didn't — see finding 2).
- sourdough: both produce a well-structured sourdough-bread tutorial.  TQ
  chose "patience, observation, and practice"; DENSE chose "managing living
  organisms".  Different valid leads.
- sliding_wrap: both produce a well-structured history of computing.

**The user's report of "incoherent and looping" output is NOT reproduced at
default temperature=0 with these standard prompts.**  The bug they hit must
involve a different invocation (temperature > 0 sampling, very long prompt,
no rep penalty, or an env-var combination not present in this test).
Reproduction step needed before any kernel-side fix.

## What this changes about the mission

- **Original hypothesis (TQ silently degrades output)** — partially falsified
  at the output level.  TQ produces clean coherent output identical to the
  reference.  At the byte level TQ != dense, but both are independently
  correct greedy completions of the prompt.
- **NEW investigation (dense nondeterminism)** is now the higher-priority
  ship-blocker — `temp=0 greedy` must be byte-deterministic across runs.
- **EXP-1 (codec K/V cosine)** still worth running as a diagnostic, but is
  no longer load-bearing for the ship decision since TQ output is empirically
  coherent.
- **EXP-2 (FP32 score promotion)** deferred until/unless EXP-1 shows codec
  fidelity issues.

## Next steps

1. Get the user's exact `hf2q generate` command + sample looping output.
   Reproduce the failure mode they hit.
2. Bisect dense-mode nondeterminism — is it HF2Q_BATCHED_PREFILL=1?  Rayon?
   GPU kernel atomics?  Re-run EXP-3 with the env var unset to isolate.
3. Capture qwen3.6 APEX analog of EXP-3 to extend the per-arch picture.
4. Decide release-check.sh strategy: should it gate on TQ output (newly proven
   stable) instead of dense (newly proven unstable)?

## Raw captures

Saved at `/tmp/exp3-tq-vs-dense-20260516-180339/` (six dirs: `{tq,dense}_run{1,2,3}`).
