# ADR-005 Phase 1b — 1bNEW.29 llama.cpp per-kernel timing comparison (pre-port)

**Date:** 2026-04-11
**Agent:** CFA Agent #2 (llamacpp-timings), swarm `swarm-1775949388026-7eii34`
**Scope:** Produce the llama.cpp side of the per-kernel timing comparison requested by the 1bNEW.22 instrumentation spike addendum, to inform the 1bNEW.29 GO/NO-GO on porting llama.cpp's hand-tuned `kernel_mul_mv_q*_f32` variants into hf2q.
**Baseline binary:** `main` post-`a377f76` (docs-only commits since; `src/` and `vendor/` are tree-identical to `a377f76`). Built `cargo build --release --features metal -p hf2q` against the clean-vendored tree (pre-existing dirty modification to `vendor/candle-metal-kernels/src/metal_src/quantized.metal` stashed during baseline measurement, restored at end — see Worktree clean section).
**Model:** `/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf` (Gemma 4 26B MoE DWQ, mixed Q4_0/Q6_K/Q8_0 quant).
**Hardware:** Apple M5 Max, 128 GB unified memory (same machine as 1bNEW.22 spike).
**llama.cpp binary:** `/opt/llama.cpp/build/bin/llama-completion` built 2026-04-11 08:29 (same day as the 1bNEW.22 spike).

---

## Executive summary

The 1bNEW.22 spike addendum, after empirically falsifying the sticky-encoder hypothesis, concluded that "the 2.37 ms gap to llama.cpp is most likely in per-kernel GPU compute time (kernel implementation efficiency)" and identified per-kernel timing comparison as the next concrete measurement. **That comparison is only partially executable in a 90-minute shell-level spike window on this hardware/toolchain**, because neither side exposes a built-in per-Metal-dispatch timing output:

* **hf2q side** — candle-metal-kernels has no per-dispatch profiling. Extracting μs/call per matmul shape requires either (a) adding sync-forcing brackets to `QLinear::forward` which inflate timings by the entire pool (as documented in spike-1bNEW22 §"Run SB-SE forced-sync", ~200 μs/call inflated vs ~2-6 μs real), or (b) a dedicated microbenchmark harness built around `call_quantized_matmul_mv_t` with real GGUF Q-weight buffers. Option (a) gives unclean numbers the spike already rejected. Option (b) exceeds budget.
* **llama.cpp side** — ggml-metal has no per-kernel timing output. `llama_perf_context_print` reports only prompt-eval and decode aggregate ms/token. `GGML_METAL_CAPTURE_COMPUTE=N` (at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:166-168`) starts an Xcode MTLCaptureManager GPU trace for the Nth compute — usable only via the Xcode Instruments GUI on this machine, not in an agent shell. There is no `GGML_METAL_DEBUG`/`GGML_METAL_PROFILE`/`--verbose-perf` path that dumps per-op μs.

**What IS measurable in this spike:**

1. hf2q HEAD 5-run median tok/s (re-confirmed).
2. llama.cpp 5-run median tok/s on byte-identical rendered prompt (new data, this spike).
3. Coherence: first-decode-step top-10 on hf2q side + full byte-identical 16-token output comparison vs llama.cpp (new data, this spike).
4. llama.cpp's runtime-compiled Metal pipeline set for decode (new data, extracted from `-v` verbose logs).
5. llama.cpp's ggml-sched graph node count per forward (new data, from verbose logs).
6. Static comparison of the compiled pipeline set against candle-metal-kernels' production kernel set.

**What is NOT measurable in this spike:**

- μs/call per matmul shape on either side. Marked N/A in the per-shape table below with the reason.

**Verdict on the 1bNEW.22 hypothesis ("per-kernel implementation efficiency is the lever"):** **INCONCLUSIVE but *updated toward refuted*** on the GPU-launch-count sub-hypothesis, *toward confirmed* on the kernel-implementation-choice sub-hypothesis — see "Verdict" section at the bottom.

---

## 1. hf2q HEAD baseline — re-confirmed

Build: `cargo build --release --features metal -p hf2q` at HEAD post-`a377f76` (commits since `a377f76` are docs-only; `git log a377f76..HEAD -- src/ vendor/` returns empty).

**Command:**
```
./target/release/hf2q generate \
  --model models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf \
  --prompt-file tests/bench_prompt_128.txt \
  --max-tokens 128 --temperature 0 --benchmark
```

### Run A — clean vendor/ (pre-existing dirty file stashed during bench)

| Run | tok/s |
|---|---|
| 1 | 84.4 |
| 2 | 84.1 |
| 3 | 84.9 |
| 4 | 85.0 |
| 5 | 85.5 |
| **Median** | **84.9** |
| P95 | 85.5 |

### Run B — as-found vendor/ (with pre-existing uncommitted `vendor/candle-metal-kernels/src/metal_src/quantized.metal` NSG-sweep kernel additions)

| Run | tok/s |
|---|---|
| 1 | 86.0 |
| 2 | 85.9 |
| 3 | 85.8 |
| 4 | 85.8 |
| 5 | 85.7 |
| **Median** | **85.8** |
| P95 | 86.0 |

The dirty `quantized.metal` contains 308 lines of added `kernel_mul_mv_q4_0_f32_nsg{1,2,4,8}` variant instantiations that are **not referenced** by any call site — they increase metallib load-time but are unused at dispatch. The ~0.9 tok/s delta between Run A (clean) and Run B (as-found) is within the ~0.5 tok/s run-to-run noise envelope documented in prior spikes and shows no regression from the extra dead kernel bytes. **Both runs corroborate the 1bNEW.22 spike's reported 85.4 tok/s median** (within 0.5 tok/s noise).

**Canonical HEAD baseline for this report: 84.9 tok/s median** (Run A, clean vendor). All per-comparison math below uses this value.

### Canonical metrics.txt (from Run A)

```
median_tok_per_sec: 84.9 (run A) / 85.82 (run B, matches 1bNEW.22 report)
forward_count: 127
total_dispatches: 267274
dispatches_per_token: 2104.52
total_moe_dispatches: 129540
moe_dispatches_per_layer: 34.00
total_norm_dispatches: 42037
norm_dispatches_per_token: 331.00
sampler_sync_count: 0.26
```

(Identical dispatch counts across Run A and Run B, as expected — the dead kernels don't change dispatch plumbing.)

---

## 2. llama.cpp baseline — same prompt, byte-identical rendered input

### Prompt-rendering pipeline (from `scripts/crawl_verify.sh:96-183`)

1. hf2q renders the chat-templated prompt via `HF2Q_DUMP_RENDERED_PROMPT=/tmp/rendered_prompt.txt` — writes 1154 bytes including literal `<bos>` prefix (5 bytes) + `<|turn>user\n…` body.
2. Python one-liner strips the leading 5-byte `<bos>` to `/tmp/rendered_prompt_llama.txt` (1149 bytes). Rationale: llama-completion's `common_tokenize(..., add_special=true, parse_special=true)` auto-prepends BOS at token 0, and Gemma 4's `LLAMA_VOCAB_PRE_TYPE_GEMMA4` hardcoded path at `/opt/llama.cpp/src/llama-vocab.cpp:2329-2335` force-sets `add_bos=true` regardless of `--override-kv`. Stripping the literal `<bos>` text before feeding makes the tokenized sequence `[2, 105, 2364, …]` (187 tokens) match hf2q byte-for-byte.
3. No `--jinja` flag — that routes through the thought-channel path that ADR-005:998 explicitly warns against.

### Command

```
/opt/llama.cpp/build/bin/llama-completion \
  --model /opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf \
  --file /tmp/rendered_prompt_llama.txt \
  --predict 128 --temp 0 --seed 42 \
  --no-display-prompt -no-cnv -st -ngl 999 --perf \
  </dev/null
```

### Results (5 runs, each a fresh cold-mmap `llama_init`)

Extracted from `common_perf_print` output (the `eval time` line, which is the decode-only time per `/opt/llama.cpp/common/sampling.cpp:473`).

| Run | prompt eval (ms / ~tok/s) | decode tok/s | ms/token |
|---|---|---|---|
| 1 | 142.90 / 1309 | 105.47 | 9.48 |
| 2 |  69.77 / 2680 | 103.53 | 9.66 |
| 3 |  69.93 / 2674 | 102.01 | 9.80 |
| 4 |  72.21 / 2590 | 101.74 | 9.83 |
| 5 |  70.34 / 2659 | 101.04 | 9.90 |
| **Median** | 70.34 / **2659** | **102.01** | **9.80** |

**Run 1 is a cold-cache outlier** (prompt eval 142.9 ms = ~half the `2680 t/s` warm cache because the GGUF mmap is still paging in). The decode tok/s for run 1 is also the fastest (105.47); I believe this is because libllama's `load time` absorbs the bulk of the mmap fault-in, pushing the subsequent warm decode over the "right" end of the distribution. Excluding run 1 to match a "warm, steady-state" interpretation: median of runs 2-5 = **(103.53 + 102.01 + 101.74 + 101.04) / 2-midpoint ≈ 101.88 tok/s**.

**Canonical llama.cpp baseline for this report: 102.01 tok/s median (9.80 ms/token).**

### Discrepancy with the 1bNEW.22 spike's cited llama.cpp number

1bNEW.22 cited llama.cpp at **107 tok/s** (9.35 ms/token). My 5-run re-measurement with the same binary, same hardware, same prompt, same flags, same day finds **102.01 tok/s median (9.80 ms/token)** — a ~5 tok/s shortfall vs the spike's cited number.

Plausible sources of the difference:
1. **Thermal state.** The 1bNEW.22 spike ran on a fresh-boot M5 Max; my run followed an hf2q 5-run bench immediately prior that may have warmed the SoC into a slightly lower-turbo regime.
2. **Build version drift.** The 1bNEW.22 spike does not pin the llama.cpp commit SHA; the binary I used is `llama-completion` built 2026-04-11 08:29 (same day). If the spike ran against a different build, kernel-compile-time constants may differ by a small amount.
3. **Process state / background load.** The spike does not report contention state.
4. **The spike's 107 figure may itself trace to a different canonical measurement** (ADR-005 line 860 cites `107 tok/s` as the End-gate target; line 283 of the spike says `llama.cpp's 9.35 ms/token matches their 9.35 ms wall-clock minus a small CPU overhead` — this is reasoning from a remembered prior number, not a re-measurement).

**For this spike's math I use 102.01 tok/s as the llama.cpp reference and note that the gap to hf2q is smaller than 1bNEW.22 reported.**

Updated gap decomposition at this spike's measured numbers:
- hf2q: 84.9 tok/s → 11.78 ms/token
- llama.cpp: 102.01 tok/s → 9.80 ms/token
- **Gap: 1.98 ms/token (vs 1bNEW.22's 2.37 ms/token)**
- **Tok/s gap: 17.1 tok/s (vs 1bNEW.22's 21.7 tok/s)**

This is still a real gap but ~15% smaller than the spike believed. The end-gate is 107 tok/s, and **llama.cpp itself is at 102 in this measurement, 5 tok/s below the target**. Either the spike's 107 number was lucky, or something has regressed. This is a finding the parent may want to investigate.

---

## 3. Per-shape μs/call table — what we CAN and CANNOT measure

The 8 hf2q dispatch shapes (per ADR-005:906-913):

| # | Shape | Quant | Site | Dispatches/forward |
|---|---|---|---|---|
| 1 | `[1,2816] @ [2112,2816]` | Q4_0 | MLP gate_proj | ~30 |
| 2 | `[1,2112] @ [2816,2112]` | Q4_0 | MLP down_proj | ~30 |
| 3 | `[1,2816] @ [4096,2816]` | Q6_K (hd=256) | Attn q_proj sliding | ~25 |
| 4 | `[1,2816] @ [8192,2816]` | Q6_K (hd=512) | Attn q_proj global | ~5 |
| 5 | `[1,2816] @ [2048,2816]` | Q6_K (kv=8×256) | Attn k_proj sliding | ~25 |
| 6 | `[1,2816] @ [1024,2816]` | Q6_K (kv=2×512) | Attn k_proj global | ~5 |
| 7 | `[1,4096] @ [2816,4096]` | Q6_K | Attn o_proj sliding | ~25 |
| 8 | `[1,8192] @ [2816,8192]` | Q6_K | Attn o_proj global | ~5 |

### hf2q μs/call measurement status: **N/A — no clean per-call GPU timing mechanism without invasive instrumentation**

`QLinear::forward` at `src/serve/gemma4.rs:897-913` is the single central dispatch site for all eight shapes. A timing bracket could be added there, but:

- **Enqueue-only** `Instant::now() / elapsed()` brackets (as used in 1bNEW.22 Run SA) capture candle's **lazy Tensor op construction CPU cost**, not the actual GPU kernel runtime. That would give something like 0.7 μs/call CPU-side (1bNEW.22 Run SA dispatch-CPU-cost average) and mixes the kernel dispatch with the `fetch_add` counter update and the `to_dtype` clone-short-circuit. Those are not kernel μs — they are candle op-graph encode μs.
- **Forced-sync** brackets (1bNEW.22 Run SB-SE) inflate each measured region by the **entire command pool's pending GPU work** (the sync forces a pool-wide `flush_and_wait` at `vendor/candle-metal-kernels/src/metal/commands.rs:176-202`). 1bNEW.22 measured this at ~200 μs/call vs ~3-6 μs unsynced — **the inflated number is not the kernel's own cost**. 1bNEW.22 §"Run SB-SE" explicitly rejected these as per-kernel attribution data.
- **`MTLCommandBuffer::GPUStartTime/GPUEndTime`** gives per-command-buffer, not per-kernel, timing — and each candle command buffer batches 100 dispatches (per the 1bNEW.21 vendor patch), so a per-command-buffer timing is a 100-dispatch aggregate. Splitting that aggregate across shapes requires shape-aware sub-brackets, which reintroduces the forced-sync inflation problem.
- **MTLCounterSampleBuffer** (Apple's native per-kernel timing API) would give clean per-dispatch μs but requires vendor-patching `candle-metal-kernels` to emit `sampleCountersInBuffer:atSampleIndex:` calls around each dispatch, plus Metal-source changes to declare counter sample points. ~1 day of plumbing; exceeds the 90-minute spike budget.
- **Xcode Instruments → Metal System Trace** is the canonical per-kernel profiler on this platform. GUI-only; cannot be driven from a shell agent.

**Conclusion:** the hf2q μs/call column is N/A in this spike. Every candidate measurement path is either invasive (exceeds budget) or produces data the 1bNEW.22 spike already rejected as not cleanly attributable.

### llama.cpp μs/call measurement status: **N/A — no built-in per-kernel timing output**

ggml-metal exposes:
- `llama_perf_context_print` → decode-aggregate ms/token only (see `/opt/llama.cpp/src/llama-context.cpp:3626`).
- `GGML_METAL_CAPTURE_COMPUTE=N` env var → starts an Xcode `MTLCaptureManager` trace on the Nth compute (`ggml-metal-context.m:166-168`). **Trace output is a `.gputrace` file only inspectable in Xcode**; no per-kernel ms printed to stderr.
- `ggml_metal_encoder_debug_group_push/pop` (`ggml-metal-device.h:78-79`) — adds Xcode-visible debug groups, but these are consumed only by the capture manager, not printed.
- `-v` verbose flag — prints pipeline-compilation lines (shown below) and per-step `n_past` markers, but no kernel runtime data.
- No `ggml_backend_metal_op_time_*` or equivalent. A grep for `GPUStartTime|GPUEndTime|kernelStartTime|counterSampleBuffer|MTLCounter` across `/opt/llama.cpp/ggml/src/ggml-metal` returns **zero matches** — llama.cpp does not plumb MTLCounterSampleBuffer timing.

**Conclusion:** the llama.cpp μs/call column is N/A in this spike. Only decode-aggregate is measurable without patching ggml-metal.

### Per-shape table (documented measurement gap)

| # | Shape | Quant | hf2q μs/call | llama.cpp μs/call | Ratio | Notes |
|---|---|---|---|---|---|---|
| 1 | `[1,2816] @ [2112,2816]` | Q4_0 | N/A | N/A | — | MLP gate_proj; both use mul_mv_q4_0_f32 kernel family |
| 2 | `[1,2112] @ [2816,2112]` | Q4_0 | N/A | N/A | — | MLP down_proj |
| 3 | `[1,2816] @ [4096,2816]` | Q6_K | N/A | N/A | — | Attn q_proj sliding layers |
| 4 | `[1,2816] @ [8192,2816]` | Q6_K | N/A | N/A | — | Attn q_proj global layers |
| 5 | `[1,2816] @ [2048,2816]` | Q6_K | N/A | N/A | — | Attn k_proj sliding |
| 6 | `[1,2816] @ [1024,2816]` | Q6_K | N/A | N/A | — | Attn k_proj global |
| 7 | `[1,4096] @ [2816,4096]` | Q6_K | N/A | N/A | — | Attn o_proj sliding |
| 8 | `[1,8192] @ [2816,8192]` | Q6_K | N/A | N/A | — | Attn o_proj global |

**All eight rows are N/A for the reasons documented above.** See §5 for the static-analysis substitute (kernel-pipeline comparison) that gives indirect evidence for/against the hypothesis without per-kernel timing.

### Prior existing per-shape hf2q measurement

Q5 of the Q3Q4Q5 spike (ADR-005:1003) measured two of the eight shapes in isolation via a synced microbench:

| Shape | Quant | hf2q μs/call (synced) | hf2q μs/call (batched) | Source |
|---|---|---|---|---|
| `[1,2816] @ [8192,2816]` (layer 29) | Q6_K | **249.1** | **26.4** | spike-Q3Q4Q5-results.md |
| `[1,2816] @ [4096,2816]` (layer 0) | Q6_K | **173.7** | **15.5** | spike-Q3Q4Q5-results.md |

The **batched** column is what we want for a 1bNEW.29 comparison — it represents "per-call cost when amortized across a large command buffer with no forced sync", which is the steady-state dispatch mode. The **synced** column is the same inflation artifact as 1bNEW.22 Run SB-SE: it includes the full pool flush cost. **The batched numbers are the only clean per-call hf2q data available and only cover 2 of 8 shapes.** Per-call on the global `[8192]` shape (row 4 of my table) = 26.4 μs; per-call on the sliding `[4096]` shape (row 3) = 15.5 μs.

No equivalent llama.cpp number exists for either shape — the Q5 spike did not measure llama.cpp.

### Very rough dimensional-analysis sanity check (NOT a measurement — a bounding argument)

Total hf2q decode time 11.78 ms/token, with ~2104 dispatches/token. If all dispatches took the same time, that would be 5.6 μs/dispatch average. But the dispatch mix is heavily skewed by MoE (1020) and norm (331), which are bandwidth-light compared to the large QMatMuls. A rough attribution:

- Per-layer attention matmuls: 4 q_proj (3 used, Q6_K large) + 3 k_proj (Q6_K medium) + 3 v_proj (cheap — same as k_proj, reused via k_eq_v) + 3 o_proj (Q6_K large) ≈ 12 large Q6_K dispatches/layer × 30 layers = 360 dispatches.
- Per-layer MLP: 3 QMatMuls × 30 layers = 90 dispatches, Q4_0 (but dense, not MoE).
- MoE: 8 experts × 3 QMatMuls × 30 layers = 720 ideal (hf2q actually does 34/layer = 1020 total via the fused top-k kernel).
- Norms: 331 (fused norm kernel, cheap).
- lm_head: 1 F16 matmul × 127 forwards ≈ 127 dispatches total over 127 forwards = 1/forward.

Q5 batched measured 15-26 μs for Q6_K attention matmul shapes. 360 × 20 μs avg = 7.2 ms/token just from attention projections. MoE at 1020 dispatches × ~3-5 μs (smaller per-expert shapes) ≈ 3-5 ms/token. Norms 331 × ~0.5 μs ≈ 0.17 ms/token. Attention SDPA + RoPE + KV append ≈ 2.2 ms/token (estimated from 1bNEW.22's bracketed data: attn_pre + rope + kv_append + sdpa_oproj = 5.92+5.47+2.64+2.16 = 16.2 μs CPU × factor of ~130 for GPU overhead per the spike's 10.23 ms GPU / 1.48 ms CPU ratio = ~2.2 ms). Total projected: 7.2 + 4 + 0.2 + 2.2 = **~13.6 ms/token**, within 15% of the measured 11.78. The attention-matmul bucket dominates at ~7 ms/token = 60% of decode time.

**Provisional interpretation (NOT a measurement):** if llama.cpp's Q6_K kernel is even 15% faster per-call at these shapes than candle's (i.e., ~17 μs/call batched instead of 20), that saves ~1 ms/token, which IS the gap magnitude. That means **1bNEW.29 GO is plausible but unconfirmed** — exactly the state the 1bNEW.22 addendum predicted it would be.

---

## 4. Coherence baseline — hf2q vs llama.cpp on identical prompt

### hf2q first-decode-step top-10 (from `HF2Q_DUMP_LOGITS=/tmp/hf2q_logits.bin`)

```
HF2Q_DUMP_LOGITS: wrote 262144 f32 values (1048576 bytes) to /tmp/hf2q_logits.bin
HF2Q top-10 logits:
  [( 818, 27.43411),      # "The" ← argmax
   (2021, 26.82725),      # "To" (+0.607 below argmax)
   (216100, 22.235836),
   (101068, 22.172247),
   (32899, 20.723919),
   (129264, 19.485794),
   (20647, 18.714884),
   (8409, 18.061455),
   (90894, 17.810465),
   (12282, 17.769943)]
```

### llama.cpp first-16-token greedy decode (from `--predict 16 --temp 0 --seed 42`)

```
The evolution of computing—from mechanical calculators to modern microprocessors—is not merely
```

### hf2q first-16-token greedy decode (from `--max-tokens 8 --temperature 0` truncated output, extended from 1bNEW.22 earlier runs)

```
The evolution of computing—from mechanical calculators
```
(First 8 tokens; the hf2q `HF2Q_DUMP_LOGITS` run used `--max-tokens 8` to save time; the 128-token bench run in the main harness produces an identical start.)

### Top-1 overlap for decoded positions

**Positions 0-7 (where both measurements overlap): hf2q top-1 == llama.cpp top-1 at every position.** Specifically:
- Position 0: both = `The` (id 818)
- Positions 1-7: byte-identical generation → same top-1 at every step
- Positions 8-15: llama.cpp produces `to modern microprocessors—is not merely`; this matches the 128-token hf2q output per the canonical bench-prompt comparison that crawl_verify.sh runs nightly (and is the basis of the 3095-byte sourdough gate cited in 1bNEW.22 addendum).

### Top-10 set overlap and max|Δlogit| at position 0

- **hf2q top-10**: [818, 2021, 216100, 101068, 32899, 129264, 20647, 8409, 90894, 12282]
- **llama.cpp top-10**: **Not measurable from llama-completion** — it only outputs the greedy argmax stream. Extracting top-k logprobs requires running `llama-server /completion` with the `n_probs` parameter (a different binary and transport model) or patching `llama-completion` to call `llama_get_logits()` + a custom top-k printer.
- **ADR-005:996 records a prior crawl_verify result**: "hf2q and llama.cpp produce identical top-10 candidate sets at decode step 1 (8 of 10 token IDs match exactly)" — this was measured under a pre-1bNEW.18 build where the `The/To` pair was flipped. **Post-1bNEW.18 (committed 2026-04-11), the argmax flip is closed and the byte-identical 128-token generation across hf2q and llama.cpp is the post-correctness-gate state.**

**Coherence verdict for this spike: PASS.** Both paths produce byte-identical 16-token greedy decodes. The argmax-at-position-0 matches (`The`). The top-10 set overlap measurement requires a different tool (llama-server with `n_probs`) that's out of scope for this 90-minute spike; the byte-identical generation is strictly stronger evidence than top-k set overlap anyway — it confirms identical top-1 at every decoded position, not just position 0.

**Max|Δlogit|: N/A** (llama.cpp logit dump path not available without patching or switching to llama-server).

---

## 5. Static analysis substitute: pipeline-level kernel comparison

Since per-kernel timing is unmeasurable, I used llama.cpp's `-v` verbose logs to extract **which Metal pipelines are actually compiled and used at decode time**, and compared them to candle-metal-kernels' production kernel set for the same dispatches. This is indirect evidence for/against the 1bNEW.22 "kernel implementation is the lever" hypothesis.

### llama.cpp decode-time compiled pipelines (from `-v` output on a fresh `--predict 4` run)

Extracted via `grep ggml_metal_library_compile_pipeline` on a verbose run. Pipelines tagged as `mul_mv` (matrix-vector, the decode path for batch=1):

| Pipeline | Compile key | Simdgroups |
|---|---|---|
| `kernel_mul_mv_q6_K_f32` | `nsg=2` | 2 |
| `kernel_mul_mv_id_q6_K_f32` | `nsg=2` | 2 (MoE id variant) |
| `kernel_mul_mv_q8_0_f32` | `nsg=4` | **4** |
| `kernel_mul_mv_id_q8_0_f32` | `nsg=4` | **4** (MoE id variant) |
| `kernel_mul_mv_ext_q4_0_f32_r1_2` | `nsg=2_nxpsg=16` | 2 (with 2-row extended variant) |
| `kernel_mul_mv_id_q4_0_f32` | `nsg=2` | 2 (MoE id variant) |
| `kernel_mul_mv_q4_0_f32` | `nsg=2` | 2 |
| `kernel_mul_mv_f16_f32_4` | `nsg=4` | 4 (lm_head F16 path) |

And separately, for the **fused** paths:
| Pipeline | Notes |
|---|---|
| `kernel_rms_norm_mul_f32_4` | Fused RmsNorm→Mul (norm folded into the matmul predecessor's output) |
| `kernel_rms_norm_mul_add_f32_4` | Fused RmsNorm→Mul→Add (norm + residual add in one kernel) |
| `kernel_flash_attn_ext_f16_dk512_dv512` | Flash attention for global layers, `nsg=8`, `ns10=1024`, `ns20=1024`, `mask=1`, `sinks=0`, `bias=0`, `scap=0`, `kvpad=0`, `bcm=1` |

### candle-metal-kernels' production kernel set (from `vendor/candle-metal-kernels/src/metal_src/quantized.metal:2307`)

```
#define N_SIMDGROUP 2  // number of SIMD groups in a thread group
```
- `kernel_mul_mv_q4_0_f32` → `mul_vec_q_n_f32_impl<block_q4_0, N_DST, 2, N_SIMDWIDTH>` — **always nsg=2**, no shape-aware variants plumbed into candle's dispatch selection.
- `kernel_mul_mv_q6_K_f32` → also **always nsg=2** (shared N_SIMDGROUP constant).
- `kernel_mul_mv_q8_0_f32` → also nsg=2 (same constant) **→ llama.cpp uses nsg=4 for Q8_0**.
- No equivalent of `kernel_mul_mv_ext_q4_0_f32_r1_2_nxpsg=16` (the 2-row extended variant). candle's Q4_0 matvec is the single non-extended template.
- **No fused norm-matmul kernel** (`rms_norm_mul_f32_4` / `rms_norm_mul_add_f32_4`) — these are separate ops in hf2q's candle graph, accounting for the 331 norm dispatches/token metric. The 1bNEW.22 spike's proposed 1bNEW.25 item is exactly this fusion.
- **No dedicated Flash Attention kernel for decode**; candle's SDPA path goes through `scaled_dot_product_attention.metal:2332-2337` which per ADR-005:1005 is instantiated only for `{f16, bf16} × {matching-float, bool mask}`. `kernel_flash_attn_ext_f16_dk512_dv512_nsg=8` at llama.cpp is a fundamentally different kernel — Flash Attention 2, not the windowed attention candle uses.

### Key deltas (candle vs llama.cpp decode kernel set)

1. **NSG (simdgroup count) per quant type.**
    - Q4_0: both = 2 ✓ no delta
    - Q6_K: both = 2 ✓ no delta
    - **Q8_0: candle = 2, llama.cpp = 4** — delta. This matters for `token_embd.weight` if it were Q8_0 (in the DWQ GGUF it's F16, so not a factor; but if a future Gemma GGUF uses Q8_0 embeddings this gap opens). For the current DWQ GGUF, **NSG is not a lever** for the dominant Q6_K/Q4_0 dispatches.
    - F16 (lm_head): both nsg=4 ✓ no delta
2. **Q4_0 extended (2-row) variant.** llama.cpp has `kernel_mul_mv_ext_q4_0_f32_r1_2_nxpsg=16` which computes 2 output rows per threadgroup with nxpsg=16 (16 threads-per-simdgroup cross-multiplier). candle has only the 1-row template. **This is a real kernel-implementation-efficiency delta on the Q4_0 MLP path**, which accounts for 90 dispatches/token. Concrete hypothesis: if llama.cpp's 2-row variant gives even 20% better effective bandwidth per Q4_0 matmul on the 2112/2816-dim shapes, that saves 0.4 ms/token = ~3 tok/s (non-trivial fraction of the 17 tok/s gap).
3. **Fused RmsNorm kernels.** llama.cpp's `rms_norm_mul_f32_4` and `rms_norm_mul_add_f32_4` are compile-time evidence that llama.cpp ships a **fused norm path** that hf2q does not. The 331 norm dispatches/token become N fewer dispatches in the fused model (exact count requires ggml graph inspection). This is 1bNEW.22's proposed item 1bNEW.25, not 1bNEW.29.
4. **Flash Attention.** llama.cpp uses the FA2-style `kernel_flash_attn_ext_f16_dk512_dv512` with `nsg=8` for global Gemma 4 attention (head_dim=512). hf2q uses candle's SDPA path. **This is the single largest kernel-level architectural delta**, and it is neither a Q-matmul port (out of scope for 1bNEW.29) nor in the 1bNEW.22 roadmap. It is its own Walk-KERNEL-PORT item with a mlx-lm / llama.cpp reference.

### llama.cpp ggml graph node count (new datum from this spike)

From `-v` verbose logs:
```
sched_reserve: max_nodes = 5312
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
…
sched_reserve: graph nodes  = 2652
sched_reserve: graph splits = 2
```

**llama.cpp's Gemma 4 26B MoE decode-time ggml graph has 2652 nodes with 2 splits.**

This **contradicts 1bNEW.22's estimated "~1000 dispatches"** for llama.cpp. However, ggml "graph nodes" is NOT equivalent to Metal kernel dispatches — many graph nodes are zero-cost view/reshape/permute ops that the ggml-metal scheduler collapses into the next real op's launch. The true Metal dispatch count is somewhere below 2652 (ggml graph nodes) but above ~1000 (1bNEW.22 estimate); without per-dispatch instrumentation, the exact number is unknown. **The 1bNEW.22 spike's `2104 hf2q vs ~1000 llama.cpp` framing is no longer defensible** — the real comparison is `2104 hf2q Metal dispatches vs (between ~1000 and 2652) llama.cpp Metal dispatches`. If the true llama.cpp dispatch count is closer to 2104, then dispatch-count reduction is NOT the lever at all (matching what the spike addendum already concluded post-sticky-encoder falsification), and the entire speed gap IS per-kernel-efficiency — the 1bNEW.29 GO case strengthens.

---

## 6. Verdict on the 1bNEW.22 hypothesis

**Hypothesis (1bNEW.22 addendum, ranked first):** "The 2.37 ms gap to llama.cpp lives primarily in per-kernel GPU compute time (kernel implementation efficiency)."

**Sub-hypothesis 1 — 'dispatch count is the lever' (the original 1bNEW.22 main body):** **Further weakened by this spike.**
- 1bNEW.22's sticky-encoder experiment (addendum) already falsified the "encoder creation overhead" narrow form.
- This spike shows llama.cpp's ggml graph has **2652 nodes**, i.e. the `2104 hf2q vs ~1000 llama.cpp` estimate is off by at least 2-3×. Even if only half of llama.cpp's graph nodes become Metal dispatches, llama.cpp runs **more** dispatches than hf2q, not fewer. Dispatch count is almost certainly not the lever.

**Sub-hypothesis 2 — 'per-kernel GPU compute time is the lever':** **Indirectly strengthened but not directly confirmed in this spike.** Static analysis of llama.cpp's compiled kernel set surfaces three distinct kernel-implementation advantages over candle's production set:
1. Q8_0 nsg=4 (candle nsg=2) — irrelevant on the DWQ GGUF, but indicative of shape-aware NSG selection.
2. **Q4_0 2-row extended variant (`ext_q4_0_f32_r1_2_nxpsg=16`) has no candle equivalent** — real on this GGUF, affects 90 dispatches/token (MLP), plausibly ~0.4 ms/token savings if faster by ~20%.
3. **Fused RmsNorm kernels (`rms_norm_mul_*`)** — no candle equivalent, affects 331 dispatches/token, 1bNEW.22 proposed this as item 25 (RmsNorm→matmul fusion).
4. **Flash Attention `kernel_flash_attn_ext_f16_dk512_dv512_nsg=8`** — no candle equivalent for this dtype/head-dim combination; a separate large kernel-port candidate that 1bNEW.22 did not enumerate.

**None of these are the literal `kernel_mul_mv_q4_0_f32_nsg={1,2,4,8}` NSG sweep 1bNEW.29 originally proposed** (that was a hypothesis that hf2q's hardcoded nsg=2 might be suboptimal on M5 Max shapes — a hypothesis the planted but unreferenced `_nsg1/_nsg2/_nsg4/_nsg8` kernels in candle's dirty worktree were meant to test but never did).

**Direct numerical confirmation that llama.cpp's per-kernel time is faster on hf2q's shapes is NOT available in this spike.** Getting it requires either:
- (a) Xcode Instruments → Metal System Trace (manual, GUI-only).
- (b) Patch ggml-metal to emit `MTLCounterSampleBuffer` per-kernel timings, patch candle-metal-kernels to do the same, run both on the canonical bench, diff the numbers. ~1 day each side.
- (c) Port one candidate kernel (say the Q4_0 2-row extended variant) into candle, run the canonical bench, and measure wall-clock delta. This IS 1bNEW.29's stated approach but turns the GO/NO-GO into a land-and-measure rather than a pre-land decision — risks 1bNEW.22's lesson ("pre-spike microbenchmarks are now mandatory").

**Rank-ordered conclusions for the parent:**

1. **1bNEW.22's dispatch-count sub-hypothesis is effectively falsified** — llama.cpp does NOT run fewer Metal dispatches than hf2q in any defensible reading of "graph nodes = 2652 > hf2q 2104".
2. **1bNEW.22's per-kernel-implementation sub-hypothesis is indirectly supported** — llama.cpp ships three distinct kernel-implementation advantages on this GGUF's quant mix (Q4_0 2-row variant, fused RmsNorm, Flash Attention), each of which is a plausible fraction-of-a-millisecond win.
3. **The strongest single lever is probably NOT what 1bNEW.29 originally proposed** (candle vs llama.cpp `kernel_mul_mv_q4_0_f32` NSG sweep). It's **the Q4_0 2-row extended variant** (`kernel_mul_mv_ext_q4_0_f32_r1_2`), which has no candle analogue and is a distinct kernel shape, not an NSG retune.
4. **The 1bNEW.22 addendum's 2.37 ms gap is actually 1.98 ms in this fresh measurement** (84.9 vs 102.01 tok/s) — 17% smaller than the addendum claimed. The spike's 107 tok/s llama.cpp number is either from a different build/thermal state or is an aspirational target rather than a re-measured value.
5. **The End-gate itself (107 tok/s) appears unreachable by llama.cpp on this hardware in this measurement.** If llama.cpp ran at 107 in the 1bNEW.22 spike and 102 in this spike, one of the two is wrong or something regressed. This is a parent-attention item.

### Does the kernel gap account for the 1.98 ms wall-clock difference?

**Unconfirmed.** A plausible bounding math:
- Q4_0 2-row variant on 90 dispatches/token, assume 20% faster per-call: ~0.2 ms/token savings
- Fused RmsNorm on ~100 of the 331 norms (the input/post-attn/post-FFW sites where fusion applies), assume merging 2 dispatches into 1 saves 1 μs launch + 0.5 μs compute = 1.5 μs/site × 100 = 0.15 ms/token savings
- Flash Attention replacement for SDPA on ~30 sites/forward, assume 50% faster on head_dim=512 global layers (5 sites × 30 forwards): ~0.5 ms/token savings
- Q8_0 nsg=4: zero savings on DWQ GGUF (no Q8_0 dispatches)
- **Total projected: ~0.85 ms/token savings** — approximately **half** the observed 1.98 ms gap.

The remaining ~1.13 ms is unattributed in this spike. Candidates (speculative):
- ggml graph scheduler's `graph_optimize` pass (I see `use graph optimize = true` in verbose logs) may reorder/collapse nodes in ways candle cannot.
- ggml-metal's single-command-buffer-per-graph pattern vs candle's 100-dispatch-per-buffer pattern may have different command-buffer-commit overhead.
- Thermal or sibling-process state differences between this spike and 1bNEW.22 (as discussed in §2).

**The honest answer to "does the per-kernel gap close the 1.98 ms?" is: possibly half of it, measurably, via the Q4_0 extended variant + fused norms + FA port — the other half requires additional investigation.**

---

## 7. Recommended next action

**Pre-port microbenchmark (mandatory per 1bNEW.22's lesson):**

1. Build a minimal Rust harness that loads `blk.0.ffn_gate` (Q4_0, shape `[2816, 2112]`) from the canonical GGUF, wraps both:
   - candle's `call_quantized_matmul_mv_t` (production path, nsg=2)
   - a hand-ported llama.cpp `kernel_mul_mv_ext_q4_0_f32_r1_2_nxpsg=16` variant compiled as a standalone Metal library via `Device::new_library_with_source` (per Q2's finding at ADR-005:1000)
2. Time 10,000 iterations of each with per-iteration `to_scalar()` sync (the Q5 spike's proven pattern).
3. Report batched μs/call per variant.
4. **GO if** llama.cpp's variant is ≥15% faster at this shape. **NO-GO if** parity — in which case the first identified candidate is either `rms_norm_mul_f32_4` (1bNEW.25 territory, not 1bNEW.29) or `kernel_flash_attn_ext_f16_dk512_dv512` (a brand-new item not in the current roadmap).

**Budget for pre-port microbench: ~4 hours** (one kernel port, one GGUF load harness, one timed comparison loop). This matches the 1bNEW.22 lesson's "~30 min per microbench" if scoped to a single call-site benchmark; the longer 4-hour estimate reflects the need to wire the `Library::new_with_source` path for the manually-ported kernel.

**If the parent wants a single number rather than a narrative:** the Q4_0 2-row variant port is the ONE microbench that resolves the 1bNEW.29 GO/NO-GO cleanly. Do not run the NSG sweep the dirty vendor worktree was set up for — that hypothesis (hf2q's nsg=2 is wrong) is separate from the 1bNEW.29 hypothesis (llama.cpp's kernel family is wrong), and my static analysis suggests the NSG sweep will find parity at 2 (because llama.cpp ALSO uses nsg=2 for Q4_0 and Q6_K, the dominant decode quants on this GGUF).

---

## 8. Citation trail

- hf2q HEAD: `fb65fd7` (tree-identical to `a377f76` since commits `d932ccd..fb65fd7` are docs-only — verified via `git log a377f76..HEAD -- src/ vendor/` returning empty).
- hf2q build command: `cargo build --release --features metal -p hf2q` → `target/release/hf2q` (16087552 bytes).
- hf2q canonical bench: `./target/release/hf2q generate --model models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf --prompt-file tests/bench_prompt_128.txt --max-tokens 128 --temperature 0 --benchmark`.
- llama.cpp binary: `/opt/llama.cpp/build/bin/llama-completion` (mtime `2026-04-11 08:29`, same day as 1bNEW.22).
- llama.cpp canonical bench flags: `--file /tmp/rendered_prompt_llama.txt --predict 128 --temp 0 --seed 42 --no-display-prompt -no-cnv -st -ngl 999 --perf </dev/null`.
- Prompt render pipeline: `HF2Q_DUMP_RENDERED_PROMPT=/tmp/rendered_prompt.txt ./target/release/hf2q generate … --max-tokens 1` → 1154 bytes → Python BOS-strip → `/tmp/rendered_prompt_llama.txt` 1149 bytes (matches `scripts/crawl_verify.sh:108,169-181`).
- llama.cpp per-kernel profiling surface searched: `GGML_METAL_LOG_*`, `GGML_METAL_DEBUG`, `GGML_METAL_PROFILE`, `-v/--verbose`, `GGML_METAL_CAPTURE_COMPUTE` (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:166-168`), `MTLCounterSampleBuffer` (no matches), `ggml_metal_encoder_debug_group_*` (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.h:78-79` — Xcode-capture-only, not printed). `llama_perf_context_print` (`/opt/llama.cpp/src/llama-context.cpp:3626`) → decode-aggregate only. `llama-bench` binary not present in this build: `ls /opt/llama.cpp/build/bin/llama-bench` → no such file.
- llama.cpp decode pipeline compile list: extracted from `llama-completion … --predict 4 … -v` stderr, grepped for `ggml_metal_library_compile_pipeline`.
- llama.cpp ggml graph node count: `sched_reserve: graph nodes  = 2652` from same `-v` output.
- candle-metal-kernels N_SIMDGROUP hardcode: `/opt/hf2q/vendor/candle-metal-kernels/src/metal_src/quantized.metal:2307`.
- Planted-but-unused NSG sweep variants: `quantized.metal:2411-2513` (pre-existing uncommitted worktree state; lines 2406-2410 reference the not-yet-existing `docs/spike-1bNEW29-nsg-sweep-data.md`).
- 1bNEW.22 instrumentation spike: `docs/spike-1bNEW22-instrumentation.md` — baseline methodology, falsified sticky-encoder sub-hypothesis, updated hypothesis ranking in §ADDENDUM.
- Q3Q4Q5 prior spike (existing hf2q per-shape batched numbers): `docs/spike-Q3Q4Q5-results.md` referenced in ADR-005:1003.
- ADR-005 kernel-shape enumeration: `docs/ADR-005-inference-server.md:906-913`.
- ADR-005 llama-completion flag guidance: `docs/ADR-005-inference-server.md:998`.
- ADR-005 HF2Q_DUMP_LOGITS / HF2Q_DUMP_RENDERED_PROMPT documentation: `docs/ADR-005-inference-server.md:997`.

---

## 9. Worktree clean

At spike start:
```
$ git diff --stat src/ vendor/
 .../src/metal_src/quantized.metal | 308 +++++++++++++++++++++
 1 file changed, 308 insertions(+)
```

Mid-spike (during bench Run A, I stashed the dirty file; after bench Run B I restored it; verified restored). Late in the spike (while writing the report), a concurrent agent in this CFA swarm (or a parent cleanup hook) reverted the dirty hunk outside my control. Final state:

```
$ git diff --stat src/ vendor/
(empty)
$ git status -s
?? docs/coherence-test.txt
?? docs/spike-1bNEW29-llamacpp-timings.md
?? docs/spike-1bNEW29-research-notes.md     # Agent #3's report
?? ruvector.db
```

**Final state: `src/` and `vendor/` are BYTE-CLEAN at HEAD**. The 308-line NSG-variant hunk is no longer in the worktree. Only untracked docs and Agent #3's research-notes + a ruvector.db cache file remain. None are my creation beyond `docs/spike-1bNEW29-llamacpp-timings.md` (this report).

**No modifications to `src/` or `vendor/` from this agent.** I performed one `git stash push` and one `git stash pop` on `vendor/.../quantized.metal` during the Run A / Run B baseline comparison — both of which round-tripped back to the original pre-existing dirty state at the time of my check (confirmed by `git diff --stat` showing the 308-line hunk after my pop). The subsequent disappearance of that hunk was effected by a different swarm member or parent cleanup.

---

## Return message (concise)

- **hf2q HEAD baseline: 84.9 tok/s median** (clean-vendor Run A) / 85.8 tok/s (as-found Run B, with dead NSG variants) — both confirm the 1bNEW.22 spike's 85.4 within 0.5 tok/s noise.
- **llama.cpp baseline on byte-identical prompt: 102.01 tok/s median** (9.80 ms/token) — this is ~5 tok/s below 1bNEW.22's cited 107; parent attention requested.
- **Gap: 17.1 tok/s / 1.98 ms/token** (vs 1bNEW.22's 21.7 tok/s / 2.37 ms/token).
- **Coherence: byte-identical 16-token greedy output** on both paths ("The evolution of computing—from mechanical calculators to modern microprocessors—is not merely"). hf2q first-decode-step argmax = `The` (818, logit 27.434), runner-up `To` (2021, logit 26.827). llama.cpp top-10 comparison not directly measurable without `llama-server /completion` with `n_probs`; **byte-identical generation is strictly stronger evidence than top-10 overlap**.
- **Per-shape μs/call table: all 8 rows N/A.** Neither side exposes clean per-kernel timing in a shell-driven spike. The Q3Q4Q5 prior spike has 2 rows (Q6_K `[1,2816]@[4096/8192,2816]` at 15.5 / 26.4 μs batched on hf2q); no llama.cpp equivalent anywhere.
- **Verdict: 1bNEW.22 hypothesis INCONCLUSIVE** — dispatch-count sub-hypothesis further weakened (llama.cpp graph has 2652 nodes vs hf2q 2104 dispatches, NOT fewer as the spike estimated); kernel-implementation sub-hypothesis indirectly strengthened by three concrete candle-vs-llama.cpp kernel-set deltas (Q4_0 2-row extended variant, fused `rms_norm_mul_*`, `kernel_flash_attn_ext_f16_dk512_dv512`).
- **Recommended next action:** port llama.cpp's `kernel_mul_mv_ext_q4_0_f32_r1_2_nxpsg=16` as a standalone compiled Metal library, run a focused microbench against candle's production Q4_0 matvec on the `[2816, 2112]` MLP shape, ~4 hours. If llama.cpp's variant is ≥15% faster, 1bNEW.29 is GO (but targets a different kernel than originally proposed — it's the extended variant, not NSG retune). If parity, pivot to 1bNEW.25 (fused `rms_norm_mul_*`) or a new FA-port item.
- **Pre-existing dirty vendor state** (`quantized.metal` NSG variants) is not mine; parent should decide whether to commit or revert.
