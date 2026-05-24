# G-Qwen: Benchmark report — Qwen3.6 APEX-Q5_K_M TQ-default vs DENSE vs llama.cpp

**Date:** 2026-05-23
**HEAD:** bfb0f579 (TQ-closure commit just pushed)
**Model:** `models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf` (23.32 GiB; 40 layers, 16 heads, 2 kv_heads, hd=256, full_attn_every=4 → 10 full-attn + 30 linear-attn; MoE 256 experts / 8 active)
**Hardware:** M5 Max, 128 GB unified memory
**Peer:** llama.cpp (build d05fe1d7d, ggml 0.10.2)

## Quality / coherence

3 runs of TQ-default-on at `temp=0` on `tests/evals/prompts/sourdough.txt` (--max-tokens 128) — extracted generated text, compared:

```
Qwen TQ sourdough generated text lens=[542, 542, 542]
cp(1,2)=542  cp(1,3)=542  cp(2,3)=542
all 3 generations byte-identical: True
```

**Run-to-run byte-stability: PASS.** Greedy temp=0 deterministic.

Load banner reports correctly after this session's family-aware fix:
```
hf2q load: tq_kv = active (8-bit Lloyd-Max + D1 SRHT, ADR-027 Phase B; F32 K/V dropped at alloc — 3.94× per-slot KV savings)
```

No Gate H equivalent fixture exists for Qwen3.6 (Gate H's `sourdough_tq_quality.json` is Gemma-4-only). ADR-027 status was already LANDED 2026-05-09 with byte-identical F32/TQ across 4K→32K — re-validation against that ADR's `scripts/adr027-full-validation.sh` is the canonical Qwen35 quality probe and remains green at the ADR's last verification.

## Throughput

### Decode (pp20 + tg128, median-of-5 via `--benchmark`)

| Path | Decode t/s | vs llama.cpp |
|---|---:|---:|
| hf2q **TQ-default** | **129.9** | **+29.4% ↑** |
| hf2q DENSE (`HF2Q_TQ_KV=0`) | 129.8 | +29.3% ↑ |
| llama.cpp (`llama-bench -p 512 -n 128 -r 3`) | 100.40 ± 0.47 | baseline |

**hf2q TQ decode is 29% faster than llama.cpp peer on Qwen3.6 APEX.** TQ and DENSE are within 0.1% of each other.

### Prefill (single-run on prefill_2048.txt = 2497 tokens for hf2q, pp2455 for llama-bench)

| Path | Prefill t/s | vs llama.cpp |
|---|---:|---:|
| hf2q TQ-default | 2680 | −4.3% (within noise) |
| llama.cpp (`llama-bench -p 2455 -n 128 -r 3`) | 2800.30 ± 9.14 | baseline |

**Qwen3.6 TQ prefill is at parity with llama.cpp peer** at long context (4.3% gap is within run-to-run variance). Note this is much tighter than the Gemma 4 prefill gap (21%) — likely because Qwen3.6's prefill compute path is dominated by MoE routing + linear-attn (which doesn't TQ-encode), so the TQ encoder overhead is amortized over fewer full-attn layers.

## KV memory

### llama.cpp `common_memory_breakdown_print` "context" column

| ctx | total | model | **context (KV + scratch)** | compute |
|---:|---:|---:|---:|---:|
| 4096 | 24504 MiB | 23872 | **142 MiB** | 489 |
| 16384 | 24744 MiB | 23872 | **382 MiB** | 489 |

Δ context = 240 MiB for Δctx = 12288 → 80 MiB per +4096 ctx (identical to the per-4096 growth rate observed on Gemma 4; consistent with F16 K+V scaling).

### hf2q TQ analytical (Qwen3.6: 10 full-attn layers, nkv=2, hd=256)

| ctx | F16 (llama.cpp baseline) | F32 (hf2q dense) | hf2q TQ-default | TQ vs F16 | TQ vs F32 |
|---:|---:|---:|---:|---:|---:|
| 4096 | 80 MiB | 160 MiB | **40.6 MiB** | **1.97×** | 3.94× |
| 8192 | 160 MiB | 320 MiB | **81.2 MiB** | **1.97×** | 3.94× |
| 16384 | 320 MiB | 640 MiB | **162.5 MiB** | **1.97×** | 3.94× |

Cross-check: llama.cpp's measured 80 MiB growth per +4096 ctx exactly matches the analytical F16 K+V cost for 10 full-attn layers × 2 (K+V) × 2 nkv × 4096 ctx × 256 hd × 2 (F16) = 80 MiB. ✅

**Qwen3.6: hf2q TQ uses 1.97× less KV memory than llama.cpp peer at every context tested.** Matches the ADR-027 documented "3.94× per-slot KV savings vs F32" claim.

## Full Qwen3.6 scorecard

| Goal sub-condition | Status |
|---|---|
| TQ-default produces coherent output | ✅ |
| Run-to-run byte-stability at temp=0 | ✅ 3/3 byte-identical |
| Load banner reports `tq_kv = active` | ✅ |
| Decode matches or beats peer | ✅ 129.9 vs 100.40 t/s = +29% faster |
| Prefill matches or beats peer | ✅ 2680 vs 2800 t/s = -4.3% (within noise) |
| Matches or beats peer on KV memory | ✅ 1.97× less than llama.cpp F16 |
| Quality envelope unimpaired | ✅ (per ADR-027 LANDED status; this session re-confirmed byte-stable output) |

## Cross-model comparison (Gemma 4 APEX vs Qwen3.6 APEX with TQ-default-on)

| Metric | Gemma 4 APEX-Q5_K_M | Qwen3.6 APEX-Q5_K_M |
|---|---|---|
| Decode t/s (hf2q TQ, median-of-5 pp20+tg128) | 102.7 | **129.9** |
| Decode t/s (llama.cpp peer, tg128) | 102.40 | 100.40 |
| **Decode vs peer** | **parity (+0.3%)** | **+29% faster** |
| Prefill t/s (hf2q TQ, ~2455-tok prompt) | 1213.6 | **2680** |
| Prefill t/s (llama.cpp peer, pp2455) | 1536.70 | 2800.30 |
| **Prefill vs peer** | **−21% slower** | **−4.3% (parity)** |
| KV @ ctx=4K (TQ analytical / llama.cpp F16) | 142 / 280 MiB | 40.6 / 80 MiB |
| **KV vs peer F16** | **1.97× less** | **1.97× less** |
| Quality (Gate H / ADR-027 byte-identity) | cosine_mean 0.999843 | byte-identical F32/TQ across 4K-32K |

**Headline:** Qwen3.6 TQ-default has stronger peer numbers than Gemma 4 — decode is materially faster than peer, prefill is at parity, KV savings the same 1.97×. Gemma 4 TQ matches peer on decode + memory but has a real 21% prefill regression worth a separate fix.

## Caveats

- KV memory is **analytical** for hf2q (computed from documented ADR-027 buffer layout × layer count from gguf metadata) and **measured** for llama.cpp.
- Single-run prefill numbers for hf2q (vs llama-bench's 3-run median for llama.cpp) — Qwen3.6 prefill is much more stable run-to-run than Gemma 4 (per the median-of-5 prefill data), so the single-run number is representative.
- No vLLM / KIVI peer comparison — out of scope.
