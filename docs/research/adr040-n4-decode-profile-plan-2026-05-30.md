# ADR-040 N=4+ Per-Slot Decode Profiling — Investigation Plan

- **Date**: 2026-05-30
- **Goal**: identify root cause of per-slot decode throughput regression as N grows past 1 on hf2q SerialFifo + InflightBatched at Qwen3.6-35B-A3B Q4_0
- **Status**: 📋 PLAN — measurement work deferred; this document specifies hypotheses + the data needed to falsify each
- **Related**: ADR-040 §6.1.56 (initial N=4 bench), §6.1.57 (N=1,4,8 bench expansion + this plan reference)

---

## 1. Empirical baseline (from §6.1.57 bench data)

Qwen3.6-35B-A3B Q4_0 MoE on M5 Max, 40 max_tokens per stream, 3-rep medians:

| N | policy | agg tok/s | per-slot tok/s | per-slot drop vs N=1 | scaling efficiency |
|---|---|---|---|---|---|
| 1 | fifo_serial | 28.9 | 28.9 | (baseline) | 1.00× |
| 4 | fifo_serial | 55.1 | 21.2 | −26.6% | 0.48× (target=1.00×) |
| 8 | fifo_serial | 63.7 | 13.5 | −53.3% | 0.28× (target=1.00×) |

**Sublinear scaling shape**: aggregate growth is `~N^0.55` rather than the ideal `~N^1.0`. Per-slot regression is **monotonic and accelerating** between N=4 → N=8 (−26.6% vs −53.3%; the second 4 slots cost more per-slot than the first 4).

This is the **dossier-predicted inflection point** from §6.1.53 (vLLM, P-EAGLE, TensorRT-LLM all flag net-regression at N>4-8 for memory-bandwidth-bound MoE workloads).

## 2. Root cause hypotheses (testable)

Each hypothesis names a kernel/subsystem with the predicted profiler signature that would confirm it.

### H1 — Memory bandwidth saturation on Q4_K dequant (highest prior)

**Claim**: At N>1, multiple decode requests interleave dequantization of the same expert blocks. The M5 Max has ~400 GB/s LPDDR bandwidth; Qwen3.6-A3B's 256-expert top-8 routing means each decode step dequants ~8 × `intermediate_size × hidden / 2` bytes per layer × 60 layers. At N=4 this is ~4× the dequant volume with no shared dequant cache → bandwidth-bound.

**Falsifier**: Metal frame capture showing `mm_id_q4_k` kernel runtime grows linearly with N (4× at N=4 vs N=1) while compute occupancy stays low (<30%). Confirms bandwidth-bound.

**Counter-evidence**: if `mm_id_q4_k` runtime is constant across N (kernel-batched), but per-frame Metal command-buffer dispatch overhead grows linearly, hypothesis is REJECTED.

### H2 — Per-request Metal command-buffer dispatch overhead

**Claim**: Each request submits its own command buffer per decode step. At N=4, that's 4 separate `MTLCommandBuffer::commit` calls per decode step rather than one batched commit. Overhead per commit is ~50-200 µs on Apple Silicon; at 20 t/s × 4 streams that's ~16-64 ms/s overhead = ~10-40% of decode budget.

**Falsifier**: Instruments `Metal System Trace` showing total `commit` time at N=4 ~ 4× N=1 commit time. Combined with H1 falsifier, would say "both contribute".

**Counter-evidence**: if commit time is amortized via background queue submission (the mlx-native session/encoder pattern), this contributes <5% and is REJECTED.

### H3 — KV-cache rolling-window cache miss for distinct sequences

**Claim**: Each decode step reads KV-cache for ALL slots' prefill regions; at N=4 with distinct prompts, the working-set KV-cache pages are 4× larger and exceed L2 (~32 MB on M5 Max). Cache-miss penalty per attention step grows with N.

**Falsifier**: Bench at N=4 with **same prompt** for all 4 streams (shared prefix). If per-slot rate improves to >25 t/s (within 13% of N=1), KV cache-miss is confirmed root cause. Bench with distinct prompts (current behavior) is the baseline.

**Counter-evidence**: identical prompts give identical per-slot rate as distinct prompts → KV layout is not the bottleneck.

### H4 — MoE router serialization at top_k=8

**Claim**: Qwen3.6-A3B's expert router does per-token expert selection. At N=4, 4 routers fire per layer; if the dispatcher serializes them rather than fusing, that's 4× router overhead per decode step.

**Falsifier**: Bench at the same N=4 with a **dense** model (Qwen3.5-12B or Gemma 4 dense) — if dense per-slot drop is ≪ 27%, MoE-specific router serialization is implicated.

**Counter-evidence**: dense model shows similar per-slot drop → router is not the bottleneck.

### H5 — Scheduler-side request-queue head-of-line blocking

**Claim**: SerialFifo's mpsc + worker thread does one request at a time but pipelines via queue. If queue dequeue + tokenizer encode is on critical path, per-slot rate drops as queue lengthens.

**Falsifier**: Bench at N=4 with `--scheduler inflight_batched` and compare per-slot rate. If SlotAware gives ≥10% better per-slot (despite worse aggregate — already measured as 96.6%), HOL blocking is confirmed.

**Counter-evidence**: SlotAware per-slot rate is the same as SerialFifo per-slot rate (current data: 20.4 vs 21.2 — within 4%, no real difference) → request queue is NOT the bottleneck. **PARTIALLY MEASURED — current data rejects this hypothesis.**

## 3. Prioritization

| H | Prior | Falsifier cost | Decision-information value |
|---|---|---|---|
| H1 (bandwidth) | high | medium (Metal frame capture) | high (would justify dequant kernel fusion work) |
| H2 (commit overhead) | medium | low (Instruments trace) | medium (could justify batched-commit refactor — ~500 LOC) |
| H3 (KV cache miss) | medium | low (same-prompt bench, 1 hour) | high (could justify KV layout redesign for shared prefix) |
| H4 (MoE router) | medium | medium (dense-model bench, needs Gemma 4 GGUF) | high (informs MoE vs dense routing decision) |
| H5 (HOL blocking) | low | none — current data rejects | (rejected) |

**Recommended order**: H3 → H1 → H4 → H2. H3 is cheapest, H1 has highest prior, H4 informs strategic direction.

## 4. Bench harness extensions needed

The existing `cb_throughput_n_1_2_4_8_fifo_vs_inflight` bench is single-prompt + distinct-stream. Profile-specific extensions:

1. **Same-prompt variant** for H3: replace `for i in 0..N { spawn(distinct prompt) }` with `for _ in 0..N { spawn(same prompt) }`.
2. **Dense-model variant** for H4: parameterize the GGUF path; run against Gemma 4 31B Q4_0 or Qwen3.5-12B Q4_0.
3. **Per-frame Metal profile capture**: env-gate `HF2Q_METAL_PROFILE=1` that opens an Instruments trace via `xcrun xctrace record --template "Metal System Trace"`.

Each is ~50-200 LOC. Total profile-harness extension: ~500 LOC.

## 5. What this plan does NOT include

- **Optimization work** — root-cause identification only. Per-hypothesis fixes scale 500-5000 LOC each (kernel fusion, batched commit, KV layout redesign) and belong in separate ADRs.
- **Multi-arch profiling** — Qwen3.6-A3B Q4_0 is the production-default; Gemma 4 31B + Qwen3.5-12B come in only as falsifiers for H4.
- **TTFT vs throughput tradeoff** — this plan profiles **per-slot decode rate**. TTFT regression at N>1 is a separate (orthogonal) concern.

## 6. Success criterion

A subsequent profile-iter publishes:
- Confirmed root-cause hypothesis (or "all four contribute by X%, Y%, Z%, W%")
- Profiler artifact (Metal frame capture, Instruments trace, or `xctrace` JSON) attached
- Estimated optimization headroom: "fixing root cause N closes K% of the per-slot gap"
- Per-ADR scoping decision: which root cause(s) warrant a new ADR

## 7. References

- ADR-040 §6.1.53 — dossier (3-source published inflection-point evidence)
- ADR-040 §6.1.56 — first N=1,4 bench (single-rep medians)
- ADR-040 §6.1.57 — N=1,4,8 bench (3-rep medians; basis for this plan)
- `/tmp/n148_bench.log` — raw bench output from 2026-05-30 run
- `tests/continuous_batching_throughput.rs` — bench harness to be extended
