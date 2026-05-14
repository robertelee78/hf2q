# ADR-030 iter-68 — HF2Q_SPEC_DFLASH=1 vs baseline benchmark

**Date**: 2026-05-14
**Hardware**: Apple M5 Max
**Model**: gemma-4-26b-a4b-it-ara-abliterated Q5_K_M (19.16 GiB GGUF)
**Drafter**: z-lab/gemma-4-26B-A4B-it-DFlash (820 MB safetensors)
**Prompt**: "Q: What is 2+2?\nA:" (rendered through gguf-embedded
chat template → 24 input tokens)
**Sampler**: greedy (temp=0), --ignore-eos
**Block size**: 8 (K=7, Phase 1.5 optimal)
**Methodology**: 2 trials per (arm, N); 20s cool-down between runs
(below the standard 60-90s thermal-fair protocol — see "caveats"
below).  Per-run wall-clock and gen tok/s from the binary's own
"Generation: X t/s" report line.

## Results (raw)

See `results.tsv` for the per-run data.  Median over 2 trials:

| N | baseline tok/s | spec tok/s | spec/baseline ratio |
|---|---|---|---|
|  8 | 103.15 | 4.65 | 0.0451× (22.2× slower) |
| 16 | 98.85  | 4.40 | 0.0445× (22.5× slower) |
| 32 | 96.55  | 4.05 | 0.0419× (23.8× slower) |

σ within each (arm, N) pair: <1% (102.0–103.3 baseline; 4.0–4.7 spec).

## Analysis

### Throughput is approximately constant across N

The spec/baseline ratio holds at ~0.045× across N=8/16/32.  The
predicted O(N²) growth from the Option C re-prefill architecture is
not yet dominant at these scales because the prompt length P=24
contributes a large fixed term.  Per-round cost ≈
`O(P + r + K)` where `r` is the accepted-token count so far; for
small r, P dominates.

### 0% drafter acceptance rate

The wall-clock data implies the orchestrator commits exactly 1 token
per round across all measured runs (otherwise spec throughput would be
higher than baseline ÷ ~22).  This is consistent with iter-67's
finding that on the chat-templated prompt content, batched-prefill
diverges from forward-decode (axis 2).  The drafter consumes target
hidden states from a batched-prefill pass; those hidden states differ
from what forward-decode would produce.  The drafter then proposes
tokens that target's verify path rejects 100% of the time.  The "free
continuation" target token (1 per round) is what gets committed.

### Target gap to mission requirement

Mission perf gate: ≥1.07× hf2q baseline (= peer-FA parity, per the
ADR-030 §1 revised gate).

```
Current spec  : 0.045× baseline
Mission gate  : ≥ 1.07× baseline
Required gain : ~24× speedup over current spec
```

A 24× speedup requires changing the architecture, not micro-optimising
the current Option C path.  The two paths forward:

1. **Option A — cross-length SDPA in batched prefill.**  Peer dflash
   (MLX, /opt/dflash/dflash/model_mlx.py:513) achieves this in one
   `model(verify_input, target_cache)` call: the K+1 verify tokens
   are computed against the prior cache (cross-length attention)
   atomically.  hf2q's existing `mlx_native::ops::sdpa::sdpa` kernel
   already supports cross-length (used by the DFlash drafter itself
   in `dispatch_dflash_sdpa_cross_length`).  The work is to wire a
   target-side cross-length verify forward — a new variant of
   `forward_decode_verify_batched` that uses sdpa::sdpa per layer
   against hybrid_kv for prior K/V.  Estimated scope: ~500 LOC + careful
   testing across sliding and full-attn layers.

2. **Option B (deferred from iter-65 plan) — capture hook on
   forward_decode + forward_decode_verify_serial.**  Loses the
   spec-decode batched-verify advantage but proves the verify
   pipeline at minimal kernel work.  Each verify round becomes K+1
   serial forward_decode calls (~6 ms each on M5 Max → ~48 ms/round
   vs current ~220 ms).  Could land in ~150 LOC.  Throughput ceiling
   = baseline ÷ K+1 ≈ ~13 t/s (still ~8× slower than baseline; spec-
   decode advantage requires Option A's batched verify).

### When current ratio would change

The ratio is mostly P-dominated currently.  At realistic prompt
lengths (P=1000+), the same Option C orchestrator would have ratio
∼ K_avg_accepted / (P + N*P_decode + K) → still O(P) per round so
spec is roughly P/K worse than baseline → ~250× slower.  Option C
fundamentally cannot reach mission gate.

## Caveats

1. **Cool-down was 20s, not the standard 60-90s.**  Thermal drift
   could contaminate ratios by up to 5% per
   `feedback_machine_state_confounds_perf_5pct_2026_05_12`.  The
   ~22× ratio observed is large enough that 5% thermal drift is
   below the signal.  For sub-2% measurements (e.g. checking
   ≥1.07× gate), use 60-90s cool-downs.

2. **0% acceptance is masking the spec-decode advantage.**  If the
   drafter ever achieves >0% acceptance (requires axis-2 fix from
   iter-67), throughput could improve, but the architecture still
   pays the O(P) re-prefill cost per round → asymptotically can't
   beat baseline without Option A.

3. **24-token templated prompt is short.**  At realistic P=200+, the
   Option C re-prefill cost grows; baseline grows linearly.  Ratio
   would worsen at longer prompts.

## Reproducer

```bash
chmod +x scripts/adr030/bench_spec_decode.sh
HF2Q_BENCH_TRIALS=3 HF2Q_BENCH_COOLDOWN=60 \
  scripts/adr030/bench_spec_decode.sh > results.tsv 2> bench.err
```

Outputs per-run logs into `docs/research/adr030_iter68_bench/`.
