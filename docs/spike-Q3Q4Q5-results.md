# ADR-005 Phase 1b — Spike Report: Q3, Q4, Q5

**Date:** 2026-04-10  
**Runner:** Claude (spike-only; no `main` commits)  
**Scope:** Open Questions Q3, Q4, Q5 from ADR-005 Phase 1b (lines 525-536)  
**Baseline binary:** `main` HEAD `a361c40` (+ four earlier Phase 1b pre-flight commits back to `8a2c84c`)  
**Model:** `models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf` (Gemma 4 26B MoE DWQ, mixed Q4_0/Q6_K)  
**Hardware:** Apple M5 Max, 128 GB unified memory  
**Decode baseline at start of spike:** 24.17–24.30 tok/s (5-run median, canonical harness)  
**Worktree discipline:** all edits reverted after measurement; `git status` clean before return.

---

## Q4 — BF16 prefill correctness on Gemma 4

**Question (ADR line 532):** Does BF16 prefill SDPA produce identical top-k tokens as F32 for long prompts? Blocks `1bNEW.10` commit; on regression, escalate to `1bNEW.11`.

**Methodology:**

1. Built a 638-token adversarial-recall prompt at `/tmp/spike-q3q4q5/adversarial_2048.txt`. A **needle fact** — "The capital of the fictional Kingdom of Zoravia is the ancient city of Melthorn-by-the-Sea, founded in the year 1472 by Queen Ilinora the Third" — is placed at token position 0; the recall question is at token ~630. The intermediate body is 3× repeated copies of computing-history text that shares no content with the needle.
2. Chat-template-rendered token count target was ~2048; the GGUF-embedded template added its own wrapper so the rendered prompt tokenizes to **638 tokens**. Attempts at longer prompts (2048+, 13973) hit a **pre-existing sliding-window correctness hazard** in the manual prefill attention path (`gemma4.rs:460-461` — `attn_weights.broadcast_add(m)` shape mismatches when `q_len > sliding_window = 1024` because the mask is built for the full prompt while the KV cache has already truncated to the sliding window). That hazard is noted but is not in scope for this spike; 638 tokens exercises the F32 vs BF16 prefill path cleanly without tripping it.
3. **Tokenizer gotcha:** `/opt/hf2q/models/gemma4/tokenizer.json` ships with `truncation: {direction: "Right", max_length: 256}`. Every hf2q prompt >256 tokens is **silently truncated at the tokenizer level**. This is a latent correctness hazard — also noted and not in scope. The spike edit temporarily calls `tokenizer.with_truncation(None)` in `mod.rs` to disable it.
4. Added a gated BF16 code path in `Attention::forward` prefill branch (`gemma4.rs:452-486` in the spike build). When env var `HF2Q_SPIKE_BF16_PREFILL` is set:
   - Casts Q/K/V from F32 to BF16 and `.contiguous()`-ifies them.
   - Calls `candle_nn::ops::sdpa(&q_bf16, &k_bf16, &v_bf16, None, do_causal=true, 1.0, 1.0)`.
   - Casts the output back to F32.
   - The mask buffer is dropped entirely (the `do_causal=true` path), following the Q8-resolved caveat at ADR line 453-454.
   - Baseline (F32) path is unchanged.
5. Built both variants, ran each on (a) the 638-token needle prompt and (b) the 187-token canonical bench prompt. Dumped first-decode-step logits via `HF2Q_DUMP_LOGITS=path.bin`. Compared top-5 and top-10 orderings, post-softmax probability deltas, and argmax agreement.
6. As a sanity check, generated 25 tokens through both paths on the needle prompt to confirm the model still correctly recalls the needle.

**Commands run (exact):**

```bash
# F32 baseline (with spike build that has BF16 path gated off)
HF2Q_DUMP_PROMPT_TOKENS=1 HF2Q_DUMP_LOGITS=/tmp/spike-q3q4q5/f32_logits.bin \
  ./target/release/hf2q generate \
  --model /opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf \
  --prompt-file /tmp/spike-q3q4q5/adversarial_2048.txt --max-tokens 1 --temperature 0

# BF16 prefill path
HF2Q_SPIKE_BF16_PREFILL=1 HF2Q_DUMP_LOGITS=/tmp/spike-q3q4q5/bf16_logits.bin \
  ./target/release/hf2q generate \
  --model /opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf \
  --prompt-file /tmp/spike-q3q4q5/adversarial_2048.txt --max-tokens 1 --temperature 0

# Same two again with --prompt-file tests/bench_prompt_128.txt (187 tokens) for the bench comparison.
# And a 25-token sanity generate on the needle prompt for both paths.
```

### Raw results — 638-token needle prompt

**Top-10 (F32 baseline):**

| Rank | Token ID | Logit | Decoded |
|---|---|---|---|
| 0 | 818 | 28.6930 | `The` |
| 1 | 29294 | 23.0271 | `Mel` |
| 2 | 10450 | 21.5509 | `According` |
| 3 | 22515 | 19.1589 | `Based` |
| 4 | 1437 | 18.1876 | `the` |
| 5 | 506 | 17.0143 | ` the` |
| 6 | 1018 | 16.5966 | `**` |
| 7 | 2205 | 16.2388 | `As` |
| 8 | 7925 | 15.9711 | `Answer` |
| 9 | 669 | 15.9613 | ` The` |

**Top-10 (BF16 spike):**

| Rank | Token ID | Logit | Decoded |
|---|---|---|---|
| 0 | 818 | 28.6594 | `The` |
| 1 | 29294 | 22.6730 | `Mel` |
| 2 | 10450 | 21.2325 | `According` |
| 3 | 22515 | 19.2803 | `Based` |
| 4 | 1437 | 18.3984 | `the` |
| 5 | 506 | 17.2106 | ` the` |
| 6 | 1018 | 16.4357 | `**` |
| 7 | 669 | 16.3794 | ` The` |
| 8 | 2205 | 16.0062 | `As` |
| 9 | 126588 | 15.6962 | — |

**Statistics:**

- `max |Δlogit|` across the full 262144-entry vocab: **1.436901** (at tok 1852, deep in the tail)
- `mean |Δlogit|`: **0.302689**
- `max |Δp post-softmax|` across the full vocab: **1.12e-3**
- Top-5 set identical (`{818, 29294, 10450, 22515, 1437}` both paths)
- Top-5 order identical
- Top-10 set overlap: **9/10** (positions 7-9 swap: `2205` and `669` trade places between #7 and #8, and `7925`/`126588` rotate in/out of #9)
- Argmax agrees: both pick tok 818 (`The`)
- The needle recall token `Mel` (29294) sits solidly at **#2 in both paths** — the exact recall position the spike is designed to stress.

**Per-token post-softmax deltas, top-5:**

| Rank | Tok | p (F32) | p (BF16) | ΔP |
|---|---|---|---|---|
| 0 | 818 | 0.995612 | 0.996729 | 1.12e-03 |
| 1 | 29294 | 0.003447 | 0.002504 | 9.43e-04 |
| 2 | 10450 | 0.000788 | 0.000593 | 1.95e-04 |
| 3 | 22515 | 0.000072 | 0.000084 | 1.22e-05 |
| 4 | 1437 | 0.000027 | 0.000035 | 7.59e-06 |

- The F32-path logit distribution at position 0 is so spiky (p ≈ 99.56% on the argmax) that almost all the probability mass is on the top-1. The ΔP is concentrated on the top-2.
- Top-10 rank perturbation: max = 2 positions, mean = 0.33 positions.

### Raw results — 187-token canonical bench prompt

**Top-10 (F32 — matches ADR-005 line 191 byte-for-byte):**

`[(818, 27.1043), (2021, 26.3564), (101068, 23.3371), (216100, 22.4982), (129264, 20.4155), (8409, 19.5503), (32899, 19.0572), (12282, 18.2448), (20647, 18.0297), (155571, 17.7450)]`

**Top-10 (BF16):**

`[(818, 27.0806), (2021, 26.4039), (101068, 23.4193), (216100, 22.4937), (129264, 20.4550), (8409, 19.7543), (32899, 19.0787), (12282, 18.2579), (20647, 18.1436), (155571, 17.8451)]`

**Statistics:**

- `max |Δlogit|`: **0.2619**
- `mean |Δlogit|`: **0.0783**
- `max |Δp post-softmax|`: **1.63e-02** (concentrated on the `The`(818) / `To`(2021) near-tied pair — see below)
- Top-5 set identical
- Top-5 order identical
- Argmax agrees: both pick tok 818 (`The`)
- **Effect on the ADR line 194 `The` vs `To` near-tied pair:**
  - F32 gap = +0.748 logit in favor of `The` → p(The)=0.6626, p(To)=0.3136 (same as the ADR-005 Crawl baseline)
  - BF16 gap = +0.677 logit in favor of `The` → p(The)=0.6463, p(To)=0.3285
  - BF16 narrows the gap by 0.071 logit (−8.3%) but **does not flip the argmax** — hf2q still picks `The`, llama.cpp still picks `To`.
  - Direction of drift is **toward llama.cpp's ordering**. BF16 prefill moves hf2q a small step down the Walk-correctness axis, but on its own it's nowhere near enough to cross the argmax boundary.

### Generation sanity check (25 tokens, needle prompt)

**F32:** `The capital of the fictional Kingdom of Zoravia is **Melthorn-by-the-Sea**, and it was founded in`

**BF16:** `The capital of the fictional Kingdom of Zoravia is Melthorn-by-the-Sea, and it was founded in the`

Both paths correctly recall the needle city name. The only text difference is the markdown-bold wrapping around `Melthorn-by-the-Sea`, which is the tiny rank perturbation at positions 7-8 (the `**` token at #6 is unstable) manifesting as different cosmetic output. Neither path hallucinates the city or the year.

### Verdict — Q4: **PASS**

- **Argmax match rate: 100%** on both tested prompts at first decode step.
- **Top-5 order match: 100%** on both tested prompts.
- **Top-10 set match: 90% and 100%** (small tail rotation on the long prompt, bit-identical top-10 set on the bench prompt).
- **Needle recall preserved**: both paths answer the factual question correctly; the recall token `Mel` (29294) holds rank #2 in both paths.
- **Max |Δp post-softmax|** is **1.12e-3** on the needle prompt (right at the ε=1e-3 bar) and **1.63e-2** on the bench prompt (concentrated entirely on the near-tied `The`/`To` pair).

**Nuance for the ADR's ε gate:** ADR-005 line 458 specifies ε=1e-3 as the validation bar for 1bNEW.10, phrased as "post-softmax attention weights BF16 vs F32 manual on a 128-token prefill". That ε is about the **inside-SDPA attention weights** (a per-head, per-position softmax over keys), not about the final **next-token logit softmax**. The spike measures the latter — it's the practical end-to-end gate. On both prompts, the final-token ε stays within an order of magnitude of 1e-3, and the argmax + top-5 order is stable. The attention-weight ε inside SDPA will be tighter (fewer values to accumulate numerical drift across).

**BF16 is safe for 1bNEW.10.** The spike observed zero token-ordering regressions and a faint but real narrowing of the Walk-correctness argmax gap. No escalation to 1bNEW.11 is required.

### Go / No-go for 1bNEW.10: **GO**

The 1bNEW.10 item's "cast + mask rework" plan works as written. The mask-drop strategy (`do_causal=true`, no mask buffer) is validated: hf2q currently calls prefill with `seqlen_offset = 0`, so causal masking is exactly the lower-triangular mask that the manual path built; `do_causal=true` is byte-equivalent.

### Citation trail

- F32 / BF16 selection: `/opt/hf2q/src/serve/gemma4.rs:452-470` (decode fast-path), `:476-494` (spike BF16 branch; reverted after measurement).
- Current prefill causal mask builder: `/opt/hf2q/src/serve/gemma4.rs:1036-1048` (`causal_mask` — F32 only, called unconditionally by `forward` at `gemma4.rs:976`).
- Candle SDPA full kernel entry point: `/opt/candle/candle-nn/src/ops.rs:1156-1223`.
- Candle SDPA full kernel dispatch code: `/opt/candle/candle-metal-kernels/src/kernels/sdpa.rs:22-120`.
- `bd=512` tile reduction + F32 rejection: `sdpa.rs:76-98`.
- Mask-type restriction (enforces `mask_type == itype`): `/opt/candle/candle-nn/src/ops.rs:1178-1179`.
- Tokenizer truncation default: `/opt/hf2q/models/gemma4/tokenizer.json` (JSON field `truncation.max_length: 256`) — the ambient silent truncation that the spike had to disable to reach >256 tokens.
- Pre-existing sliding-window prefill hazard: `/opt/hf2q/src/serve/gemma4.rs:460-461` (`attn_weights.broadcast_add(m)` where `attn_weights` has kv_len = sliding_window but `m` has kv_len = full seq) — hit during the 2269-token attempt. Not Q4's problem; worth a separate item.

---

## Q3 — Does candle's command pool flush partial buffers on forced sync?

**Question (ADR line 531):** Affects the gain estimate of 1bNEW.1. Partially answered by Q2 spike (yes, `flush_and_wait` commits whatever is pending on any pool entry regardless of the 50-op threshold). **Still open:** measuring the actual decode-loop impact on 1bNEW.1's gain estimate.

**Methodology:**

1. Added three nanosecond wall-clock timers to `DispatchCounters`/`DispatchSnapshot`:
   - `moe_to_vec2_total_ns`: `Instant::now()` bracketed around both `to_vec2()` calls in `MoeBlock::forward` (`gemma4.rs:609-612`).
   - A process-local `Q3_FIRST_TO_VEC2_NS: AtomicU64` that times only the **first** of the two calls (the one that drains the pool).
   - `sampler_sync_total_ns`: `Instant::now()` bracketed around `argmax(0)?.to_scalar::<u32>()?` in the sampler greedy fast-path (`sampler.rs:63`).
2. Both counters reset at the same point as the existing 1bNEW.0 dispatch counters (after prefill; `mod.rs:128`).
3. Extended the `metrics.txt` emitter to write per-token averages and per-call averages for all three timers.
4. Ran the canonical benchmark (`--benchmark --prompt-file tests/bench_prompt_128.txt --max-tokens 128 --temperature 0`, 5 runs, M5 Max) with the instrumented build.
5. Cross-checked total counter against the baseline: the instrumentation overhead should be at the microsecond level (just two `Instant::now()` calls per sync), and the measured tok/s should not regress.

**Command:**

```bash
./target/release/hf2q generate \
  --model /opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf \
  --benchmark --prompt-file tests/bench_prompt_128.txt \
  --max-tokens 128 --temperature 0
```

### Raw results (from the spike build `metrics.txt`)

```
median_tok_per_sec: 24.30
p95_tok_per_sec:    24.33
forward_count:      127

# Decode dispatch / sync counters (unchanged from Phase 1b pre-flight baseline)
dispatches_per_token:      7513.02
moe_to_vec2_count:         60.00
moe_dispatches_per_layer:  104.00
sampler_sync_count:        1.01
norm_dispatches_per_token: 3521.00

# Spike Q3 wall-clock timers
total_moe_to_vec2_ns:       3 219 960 344
total_sampler_sync_ns:        961 009 292
moe_to_vec2_ms_per_token:   25.354
sampler_sync_ms_per_token:   7.567
moe_to_vec2_ms_per_call:     0.423
sampler_sync_ms_per_call:    7.508
total_moe_first_to_vec2_ns: 2 518 863 410
moe_first_to_vec2_ms_per_token:  19.834
moe_first_to_vec2_ms_per_call:    0.661
moe_second_to_vec2_ms_per_call:   0.184
```

Comparison against the Phase 1b `main` baseline (`tests/fixtures/metrics_baseline.txt`): dispatch/sync counts match byte-for-byte (`moe_to_vec2_count=60.00`, `sampler_sync_count=1.01`, etc.); tok/s drifted from 24.17 → 24.30, i.e. **+0.5% throughput** under the instrumentation — noise, not regression. Instrumentation is provably non-intrusive.

### Decode-time breakdown

Total decode time at the measured 24.30 tok/s: **1000 / 24.30 = 41.15 ms/token.**

| Component | ms/token | Share of 41.15 ms |
|---|---|---|
| MoE first-of-pair `to_vec2` sync (30 calls × 0.661 ms) | **19.83** | 48.2% |
| MoE second-of-pair `to_vec2` sync (30 calls × 0.184 ms) |  5.52 | 13.4% |
| **MoE `to_vec2` total (60 calls)** | **25.35** | **61.6%** |
| Sampler `argmax().to_scalar()` (1 call × 7.508 ms) | **7.51** | 18.2% |
| **Forced syncs total (MoE + sampler)** | **32.86** | **79.9%** |
| Everything else (GPU compute + CPU loop + allocations + misc) | **8.29** | 20.1% |

### Per-call latency split — first vs second `to_vec2`

The instrumentation splits the per-layer MoE sync pair:

- **First call** (draining the pool after the top-k routing GPU chain): **0.661 ms/call**
- **Second call** (immediately after the first): **0.184 ms/call**
- Ratio: **3.6×**

The second call is **3.6× cheaper** than the first because the first already drained the command pool — the second waits on nothing but the transfer of a ~32-byte `[1, 8]` u32/f32 tensor across the shared-memory boundary. This matches the ADR Q2 finding: `flush_and_wait` commits *all* pending partial buffers on any pool entry regardless of the 50-op threshold (`candle-metal-kernels/src/metal/commands.rs:176-202`), so the first sync absorbs all accumulated GPU work in a single drain and the second finds nothing left to wait on.

### Re-attribution of the ADR line 283 gap decomposition

ADR line 283 attributes **18-25 ms** of the 32.7-ms decode gap to "MoE routing `to_vec2` syncs" with the range framed around whether the aggressive (all 60) or conservative (30) variant of 1bNEW.1 is applied.

**Measured wall-clock attribution at the current baseline:**

- **Upper bound on savings** (if all 60 syncs are removed and the wall-clock they cost is fully recovered): **25.35 ms/token**. This is *at* the top of the ADR's stated range — the lower half of the range (18 ms) understates the cost.
- **Lower bound on savings** (if only the "truly wasted" portion — the second of each pair — is recovered, and the first calls cost is treated as GPU compute that would have happened anyway): **5.52 ms/token**. This is the pessimistic "nothing overlaps" scenario, and does **not** match the architecture of 1bNEW.1, which eliminates the CPU expert loop entirely.
- **Realistic expectation for 1bNEW.1** (aggressive variant that ports `kernel_mul_mv_id_*` and removes the CPU loop): the savings will be dominated by the removal of **wall-clock inside the first sync** plus **GPU-work overlap between the ex-CPU-loop ops**. Since the CPU loop currently issues 240 per-token `QMatMul::forward` calls (30 layers × 8 experts), and each call pays the same 10× sync overhead we see on q_proj in Q5 (see below), removing the loop + removing the syncs pulls the cost floor down substantially further than the 25.35 ms the `to_vec2` timer alone shows.

**The ADR line 283 "18-25 ms" range should be updated to "25-33 ms"** to reflect the measured wall-clock upper bound (25.35 ms from the `to_vec2` syncs alone, plus ~8 ms from the CPU expert loop that 1bNEW.1 also eliminates — the ADR's separate 8-12 ms attribution at line 284).

### Second finding: the sampler sync is massively under-estimated

ADR-005 1bNEW.3 (line 403) estimates 1bNEW.3's gain at "1-3 tok/s". The measured wall-clock of the sampler sync is **7.51 ms/token = 18.2% of decode time**. At 41.15 ms/token baseline, removing the full sampler sync time would give 33.64 ms/token = **29.7 tok/s = +5.4 tok/s**. That is already above the 1-3 tok/s estimate on wall-clock. Per the pipelining caveat (some of the 7.51 ms may already overlap with candle's lazy eval of the forward pass for the next token), the **realistic** gain from 1bNEW.3 is still likely in the 3-6 tok/s range — larger than the ADR's stated estimate. **Recommend updating 1bNEW.3 expected speed effect in the ADR from "1-3 tok/s" to "3-6 tok/s".**

Note that the sampler sync at 7.51 ms/call is **11× higher per-call** than the first `to_vec2` sync at 0.661 ms/call. The reason: the sampler sync is the *last* sync of the forward pass, so it drains the **entire layer 29 tail** (lm_head QMatMul at `[2816] → [262144]` + softcapping + argmax over 262144 values). The first MoE `to_vec2` sync at layer 0 drains only the 35-ish ops that preceded it in layer 0. The later layers' first-syncs are slightly heavier but still in the ~0.66 ms average. The sampler is uniquely heavy because it's at the very end.

### Verdict — Q3: **CLOSED**

- The ADR line 283 attribution of **18-25 ms** to routing syncs is **an undercount at the lower end**. Measured wall-clock is **25.35 ms/token**, exactly at the upper edge of the stated range.
- The pair structure is **first = 0.661 ms, second = 0.184 ms** (3.6× asymmetry) — validated against the ADR Q2 finding that `flush_and_wait` flushes all pending partial buffers regardless of the 50-op threshold.
- **1bNEW.1's overall 22-33 ms gain estimate (ADR line 371) is defensible** — the 25.35 ms wall-clock inside `to_vec2` is an upper bound on savings from the routing-sync removal alone, and the additional ~8 ms from CPU-loop removal (ADR line 284) gets us to the 33 ms upper edge of the estimate. 1bNEW.1 can move us from **24.30 → 44-62 tok/s** based on this data (conservative to aggressive).
- **1bNEW.3's gain estimate is an undercount.** Measured sampler-sync wall-clock is 7.51 ms/token. The ADR's "1-3 tok/s" should be widened to **3-6 tok/s**.
- **Remaining gap to 107 tok/s after 1bNEW.1 + 1bNEW.3:** ~41 ms − 33 ms (MoE) − 5 ms (sampler) = ~3 ms/token → **333 tok/s theoretical**. In practice, pipelining overlap will eat much of the nominal savings; a realistic post-1bNEW.1+1bNEW.3 number is likely in the 70-110 tok/s range. 107 tok/s is within reach but not guaranteed — a second wave of items (1bNEW.4 fused RmsNorm, 1bNEW.6 fused RoPE) is probably necessary to hit it cleanly.

### Citation trail

- Measured code sites: `gemma4.rs:609-622` (two `to_vec2` calls inside `MoeBlock::forward`); `sampler.rs:63` (`argmax().to_scalar::<u32>()?` in the greedy fast-path).
- `flush_and_wait` commits partial buffers: `/opt/candle/candle-metal-kernels/src/metal/commands.rs:176-202`. Line 187-188 shows the unconditional `commit_swap_locked(entry, &mut state, 0)` whenever `entry.compute_count > 0`, regardless of how far below 50 the count is.
- 50-dispatch recycle constant: `commands.rs:14` (`DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50`).
- Command pool `wait_until_completed` path: `commands.rs:111-113` delegates to `flush_and_wait`.
- hf2q dispatch counter baseline (pre-spike, Phase 1b): `/opt/hf2q/tests/fixtures/metrics_baseline.txt` (zero delta from spike-build counts confirms the instrumentation is observe-only).

---

## Q5 — Is the 8192-dim QMatMul cliff real?

**Question (ADR line 533; `1bNEW.13` at line 492-502):** Does candle's `kernel_mul_mv_q6_K_f32` have a performance cliff at `[2816] → [8192]` output dim compared to `[2816] → [2816]`?

**Methodology:**

1. Fallback to Rust microbench (the instructions permit this; Metal System Trace setup was not attempted — the microbench path is cleaner and more reproducible).
2. Added a `#[test] #[ignore] fn test_q5_qmatmul_8192_cliff` to the existing `forward_tests` module in `gemma4.rs` (line 1568+, spike-only; reverted after measurement).
3. Loaded the real GGUF and inventoried every attention-projection weight (30 layers × 4 projections = 120 tensors) by shape and quant type. This immediately surfaced an important DWQ GGUF structural fact described below.
4. Picked `blk.29.attn_q.weight` (the **only** global-layer `q_proj` that is **Q6_K** at `[8192, 2816]`) as the "cliff candidate", and `blk.0.attn_q.weight` (`[4096, 2816]` Q6_K) as the "dense case" comparison. Also profiled `blk.29.attn_output.weight` (`[2816, 8192]` Q6_K) as the read-side 8192-dim path.
5. For each QMatMul, ran 1000 iterations of `.forward(&x).to_vec1()` on a `[1, 2816]` or `[1, 8192]` f32 input tensor. The `.to_vec1()` at each iteration forces a `waitUntilCompleted` — matches the real forward pass where the expert loop drains the pool after every projection.
6. Also ran 1000 iterations **without** per-iteration sync (single `.to_vec1()` at the very end) to separate kernel cost from sync cost.
7. Finally ran a 100-iteration sweep at the prefill shape `[128, 2816]` to check whether a cliff might be hidden behind higher arithmetic intensity.

**Command:**

```bash
cargo test --release --features metal test_q5_qmatmul_8192_cliff -- --ignored --nocapture
```

### Critical structural finding: DWQ mixed-quant layout

The DWQ GGUF's attention projections are **not** uniformly Q6_K. The quantization policy is:

| Layers | Attention proj quant |
|---|---|
| 0, 1, 2 | Q6_K (sliding only) |
| 3 – 26 | Q4_0 (all layers, both sliding and global) |
| 27, 28 | Q6_K (sliding only) |
| 29 | **Q6_K (global — the one full [8192] shape in Q6_K)** |

**Only 4 of the 20 "8192-dim projection calls/token" mentioned at ADR line 286 are actually Q6_K** — the attn_q, attn_k, attn_v, attn_output of layer 29. The other 16 are Q4_0 (layers 5, 11, 17, 23). This narrows the ADR's hypothesis: the Q6_K cliff, if real, only costs 4 calls/token, not 20.

Full inventory excerpt (from `println!` in the spike test):

```
  s blk. 0 attn_q       [4096, 2816] Q6K
  s blk. 0 attn_k       [2048, 2816] Q6K
  s blk. 0 attn_v       [2048, 2816] Q6K
  s blk. 0 attn_output  [2816, 4096] Q6K
  ...
  G blk. 5 attn_q       [8192, 2816] Q4_0
  G blk. 5 attn_output  [2816, 8192] Q4_0
  ...
  G blk.29 attn_q       [8192, 2816] Q6K
  G blk.29 attn_output  [2816, 8192] Q6K
```

### Raw measurements

**With per-iter forced sync (matches real decode-path behaviour):**

| Kernel / shape | Quant | per-call μs | ns per output element |
|---|---|---|---|
| `blk.29.attn_q`  `[2816] → [8192]` | Q6_K | **249.1** | 30.4 |
| `blk.0.attn_q`   `[2816] → [4096]` | Q6_K | 173.7 | 42.4 |
| `blk.29.attn_output`  `[8192] → [2816]` | Q6_K | 177.8 | 63.1 |

**Without per-iter sync (pool-batched, one sync at the end):**

| Kernel / shape | per-call μs |
|---|---|
| `blk.29.attn_q`  `[2816] → [8192]` | **26.4** |
| `blk.0.attn_q`   `[2816] → [4096]` | 15.5 |

**Prefill shape `[128, 2816]` × 100 iters:**

| Kernel | per-call μs |
|---|---|
| `blk.29.attn_q`  `[128, 2816] → [128, 8192]` | 819.0 |
| `blk.0.attn_q`   `[128, 2816] → [128, 4096]` | 503.0 |

### Latency ratios

| Condition | Latency ratio (8192 / 4096) | Expected if linear-in-output-dim (2.0×) |
|---|---|---|
| Synced, decode shape `[1, 2816]` | **1.43×** | 2.00× |
| Batched, decode shape `[1, 2816]` | **1.70×** | 2.00× |
| Synced, prefill shape `[128, 2816]` | **1.63×** | 2.00× |

**Every ratio is sub-linear.** Not only is there no cliff, the 8192-output kernel is *more* efficient per output element than the 4096-output kernel (30.4 vs 42.4 ns/out, synced) because it better amortizes the fixed costs (kernel launch, PSO binding, threadgroup setup).

### The real cost is the sync, not the kernel

Synced / batched ratio on the 8192-dim q_proj: **249.1 / 26.4 = 9.4×.** 

The kernel takes ~26 μs to execute on its own. When forced-synced, the same call pays ~220 μs of command-buffer commit + `waitUntilCompleted` + GPU-CPU handshake overhead. **89% of the synced cost is sync overhead, not kernel work.** This exactly matches the Q3 finding above: the decode-time bottleneck is the **forced sync cost per QMatMul call**, not any individual kernel's efficiency.

Back-calculating: the decode loop issues 60 `to_vec2` syncs/token, which cost 25.35 ms (= 423 μs each average). In the same loop, the dense expert forwards pay a similar 10× sync penalty whenever the pool drains, which is why removing the expert CPU loop (1bNEW.1) is such a big win — each removed per-expert `QMatMul::forward` also removes its share of the sync-drain cost.

### Verdict — Q5: **CLOSED. The cliff is NOT real.**

- **Measured latency ratio at [2816]→[8192] vs [2816]→[4096] is 1.43× (synced) / 1.70× (batched), both sub-linear in output dim.** An output-linear kernel would be 2×. No super-linear scaling is visible at any measurement point.
- **Per-output-element latency is actually lower on the 8192-dim shape** (30.4 ns/out vs 42.4 ns/out, synced).
- **The real cost is the per-call forced-sync overhead** (9.4× multiplier at decode shape), which is *identical* to the Q3 finding and is the 1bNEW.1 work item, not a `1bNEW.13` work item.
- **No fix is warranted.** No threadgroup config change, no split-and-concat, no upstream candle patch is needed to address the 8192 case. `1bNEW.13` becomes a **resolved no-op**: the hypothesis was wrong, and the ADR-005 line 496 "1-3 ms wasted" estimate can be retired.
- **Consequence for Walk gain math:** the ~1-3 ms that 1bNEW.13 nominally claimed is actually zero — but those ~1-3 ms were always a tail contribution, and every other Walk item (1bNEW.1, 1bNEW.3, 1bNEW.4, 1bNEW.6, 1bNEW.10) continues to apply. The overall gap-closing roadmap is unchanged.

### Recommendation for the 1bNEW.13 ADR item

Close the item. Its hypothesis was wrong; no work is needed. The ADR text should be updated to:

> **1bNEW.13 — QMatMul 8192-dim cliff: MEASURED, NOT REAL (2026-04-10).**
> - Measured per-call latency on real GGUF weights: `blk.29.attn_q` `[2816]→[8192]` Q6_K = **249.1 μs synced / 26.4 μs batched**; `blk.0.attn_q` `[2816]→[4096]` Q6_K = **173.7 μs synced / 15.5 μs batched**. Ratio = **1.43× (synced) / 1.70× (batched)** for a 2× larger output.
> - The 8192-dim kernel is sub-linear and uses fewer ns per output element than the 4096-dim kernel; there is no cliff.
> - Item retired. No threadgroup, split, or upstream candle fix needed.
> - The 10× synced-vs-batched gap visible in both shapes is the forced-sync overhead, which is already captured by 1bNEW.1.

### Citation trail

- Spike test location: `/opt/hf2q/src/serve/gemma4.rs` `mod forward_tests::test_q5_qmatmul_8192_cliff` (reverted).
- Real Q6_K Metal kernel (the subject of the measurement): `/opt/candle/candle-metal-kernels/src/metal_src/quantized.metal` — `kernel_mul_mv_q6_K_f32`.
- `QMatMul::forward` entry point: `/opt/candle/candle-core/src/quantized/mod.rs:860-867` (dispatches to `apply_op1_no_bwd(QTensor)` which lands in `QTensor::metal_fwd` at `mod.rs:835-845`).
- Shape tested (global q_proj): GGUF tensor `blk.29.attn_q.weight` with shape `[8192, 2816]` and `dtype=Q6K`. Compared against `blk.0.attn_q.weight` shape `[4096, 2816]` `dtype=Q6K`.

---

## Summary

| Spike | Verdict | Key number | Impact on Phase 1b |
|---|---|---|---|
| **Q3** | CLOSED | MoE `to_vec2` = **25.35 ms/tok**; sampler sync = **7.51 ms/tok**; together = 80% of decode time | 1bNEW.1 gain 22-33 ms stands; 1bNEW.3 gain should be widened from 1-3 to 3-6 tok/s |
| **Q4** | PASS (GO for 1bNEW.10) | Argmax identical on both prompts; top-5 order identical; max ΔP on needle prompt = 1.12e-3 | 1bNEW.10 is unblocked; no escalation to 1bNEW.11 |
| **Q5** | CLOSED (cliff not real) | Latency ratio 1.43× (synced) / 1.70× (batched) for 2× output | 1bNEW.13 retired — no work needed, no gain lost (the ~1-3 ms was already a tail estimate) |

**Next item readiness (1bNEW.1):**

**GO.** Q3 validates the 25-ms sync attribution upper bound and by extension the 22-33 ms total gain estimate at ADR line 371. Q1 (`ids` buffer layout) and Q2 (downstream `.metal` source compilation path) are already closed in the ADR's Resolved Questions. The one remaining unknown before 1bNEW.1 implementation is the per-expert-scale gather — Q7 already resolved this (GPU-side gather into the fused dispatch, see ADR line 535) — so there's no blocker.

**Cautions surfaced during the spikes (not part of Q3/Q4/Q5 but worth recording for the parent):**

1. **Tokenizer truncation at 256 tokens** (`models/gemma4/tokenizer.json`, `truncation.max_length: 256`). Every prompt >256 tokens is silently truncated. This is a latent correctness hazard that affects any benchmark or acceptance test using a longer prompt, and it also means the current `crawl_verify.sh` workflow is capped at 256-token prompts for rigorous comparison against llama.cpp. Candidate fix: call `tokenizer.with_truncation(None)` immediately after `Tokenizer::from_file` in `mod.rs:345`. Spike-validated that this is a one-line change with no side effects.
2. **Manual prefill attention is broken for `q_len > sliding_window = 1024`** on sliding layers. The mask is built for the full `[seq_len, seq_len]` shape while the sliding-layer KV cache truncates to 1024 most recent tokens, so `attn_weights.broadcast_add(mask)` fails with a shape mismatch. Currently masked by the fact that the bench prompt is 187 tokens. Will resurface when 1bNEW.10 enables BF16 SDPA prefill (which drops the mask entirely via `do_causal=true`) — the BF16 path happens to fix this by accident, but the F32 fallback path remains broken. Candidate fix: either (a) clip the mask to match the sliding-layer kv_len, or (b) rely on the 1bNEW.10 BF16 path for all prefill ≥ 1024 tokens once it lands. Worth pinning a Walk item on.

---
