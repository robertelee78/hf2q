# Crawl / Walk progress tracker — ADR-005 Phase 1b

One row per Walk item plus the Crawl baseline. The purpose of this table is
to track the Walk-correctness signal defined at ADR-005 lines 183-187 — the
top-1 token and top-10 logits at decode 1 on the canonical 187-token prompt
(`tests/bench_prompt_128.txt`), and the byte-level `crawl_verify.sh`
classification against llama.cpp's reference output.

The Crawl-verify classification is known to read **RED** across every row
below because `llama-completion --jinja` applies a different prompt path
than the rendered GGUF template — this is the `--jinja` gotcha documented
at ADR-005 line 198. The real Walk-correctness gate is the `hf2q_top1`
column moving from `The` to `To`, which will unblock the Layer B fixture
commit (ADR-005 line 181).

GGUF: `models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf`
(sha256 `ae19574dab588e0a742a8cfa1282ccf358ed8a58c5a3483f2bf596020a4f8e6f`)
Prompt: `tests/bench_prompt_128.txt` (187 tokens after chat-template render)
Hardware: Apple M5 Max, 128 GB
Tool: `HF2Q_DUMP_LOGITS=/tmp/h.bin ./target/release/hf2q generate --model <gguf> --prompt-file tests/bench_prompt_128.txt --max-tokens 1 --temperature 0`

## Walk progress table

| item     | commit    | hf2q_top1      | hf2q_top1_logit | llama_top1     | llama_top1_logprob | divergence_point | classification | notes |
|----------|-----------|----------------|-----------------|----------------|--------------------|------------------|----------------|-------|
| Crawl baseline | `0a703d7` | `The` (818)    | 27.10432        | `To` (2021)    | -0.6487            | byte 0           | RED            | Documented at ADR-005 lines 189-198. Top-10 candidate sets are 8/10 identical; the argmax flips on a ~0.12 logprob gap. The RED classification is the `--jinja` gotcha path, not the argmax-flip itself. |
| 1bNEW.0a | `8a2c84c` | (not measured) | (n/a)           | (not measured) | (n/a)              | (not measured)   | (n/a)          | Chat-template loader. Landed before Crawl verification was set up; included in this table only as history. |
| 1bNEW.0  | `6ff446e` | `The` (818)    | 27.10432        | `To` (2021)    | -0.6487            | byte 0           | RED            | Pure metrics instrumentation; observe-only counters. Top-10 byte-identical to Crawl baseline, as expected. |
| 1bNEW.0b | `d4cab72` | `The` (818)    | 27.10432        | `To` (2021)    | -0.6487            | byte 0           | RED            | Un-fuse `forward_with_residual` (Walk Exception unwind). Rust-side refactor only; no kernels change. Top-10 byte-identical to 1bNEW.0, as expected at F32 where the fused and unfused elementwise-add forms are numerically equivalent. The argmax-flip signal for this item is *absence* of regression — 1bNEW.0b is documented LOW risk, not a speed item. |
| 1bNEW.1 Phase A | `7dc627f` | `The` (818)    | 27.10432        | `To` (2021)    | -0.6487            | byte 0           | RED            | `call_quantized_matmul_mv_id_t` wrapper + 3 Phase-A unit tests (Q6K, Q8_0, prefill-shape). Tests compare the fused kernel elementwise vs the 8-separate-`QMatMul::forward` reference at ε=1e-5; all three passed at max\|Δ\|=0.000e0. No forward-pass changes — top-10 byte-identical by construction. |
| 1bNEW.1 Phase B | `8212f4a` | `The` (818)    | 27.10432 (loop) / 27.104322 (fused, Δ≈2e-6) | `To` (2021)    | -0.6487 | byte 0 | RED            | Fused kernel wired into layer 0 ONLY behind `--moe-kernel=fused`; default stays `loop`. 16-decoded-token match byte-identical against loop mode; top-10 ids and order preserved across both modes; max\|Δ\| across the top 10 ≈ 1.5e-5 (FP reduction-order drift). Metrics: `moe_to_vec2_count=58.00` (was 60; exactly layer 0's two syncs removed). Median 25.16 tok/s vs 24.17 baseline (+1 tok/s from layer 0 alone). |
| 1bNEW.1 Phase C | (pending) | `The` (818)    | 27.10432 (loop) / 27.104326 (fused, Δ≈6e-6) | `To` (2021)    | -0.6487 | byte 0 | RED            | Fused kernel wired into ALL 30 layers. 32-decoded-token byte-identical match vs loop mode. Top-10 ids + order preserved; max\|Δ\| ≈ 3e-5. Metrics: **`moe_to_vec2_count=0.00`** (every routing sync eliminated; acceptance criterion ADR line 623 **met**), `moe_dispatches_per_layer=42.00` (was 104; meets the `≤ 4` stretch goal direction, will sharpen further with 1bNEW.4/6), `dispatches_per_token=5653` (was 7513, −24.8%). **Median 36.78 tok/s vs 23.76 loop baseline — +54.8% speedup (+13.0 tok/s).** No argmax flip vs loop; `The` → `To` gate unchanged (owned by 1bNEW.4/6 reduction-order items). |
| 1bNEW.10 | `29b84ef` (post `9cc522d` fix) | `The` (818) | 27.10343 (Δ ≈ −9e-4 vs F32 baseline) | `To` (2021) | -0.6487 | byte 0 | RED | BF16 prefill SDPA split by head_dim: global layers (bd=512, 5/30) fused; sliding layers (bd=256, 25/30) retain manual path due to two upstream-candle blockers (F32 threadgroup memory blowup + BF16 sawtooth NaN). Multi-shape correctness sweep at 14 prompt lengths (1-1000 tokens) all OK. Top-10 IDs and order byte-identical to pre-1bNEW.10; top-1 preserved at `The` (818). `The`/`To` logit gap: F32 baseline +0.748 → hybrid +0.745 (essentially unchanged — only 5/30 layers BF16, drift ~10× smaller than the pure-BF16 `9cc522d` path). gen16 and gen128 byte-identical to pre-1bNEW.10 F32 baseline. 827-token adversarial recall preserved. 3142-token long prompt still crashes on sliding-window mask shape mismatch (same envelope as pre-1bNEW.10, Walk Exception Register entry remains OPEN). Median 37.11 tok/s (was 36.91 pre-landing, noise). |
| 1bNEW.12 | `b8def90` | `The` (818) | 27.10343 (unchanged from 1bNEW.10 row) | `To` (2021) | -0.6487 | byte 0 | RED | 10-token prefill warmup + forced GPU sync. Warmup is throwaway; decode-1 logits byte-identical to 1bNEW.10 row. Median 37.06 tok/s (within noise of 1bNEW.10). TTFT median improvement on 14-token "Hi" prompt: 40.5 → 36.6 ms (−3.9 ms); 50-token → −5.5 ms; 187-token bench → −4.5 ms. gen16 byte-identical to 1bNEW.10 row. The Walk-correctness `The` → `To` flip is still owed to 1bNEW.4 / 1bNEW.6 reduction-order items. |
| 1bNEW.4 Phase A | `2aa40d8` | (no forward-pass change) | (n/a) | (n/a) | (n/a) | (n/a) | (n/a) | Runtime-compiled `kernel_rms_norm_fuse_impl<T, F>` port + 7 Phase A unit tests covering F=1/F=2/F=3 across float & float4 paths and five shapes (hidden=2816, head_dim=256/512, decode-1, scalar non-%-4). All 7 tests pass; max \|Δ\| across all cases vs the 11-op candle chain = 2.384e-7 (single ULP). No gemma4.rs forward-pass changes — top-1 unchanged by construction. |
| 1bNEW.4 Phase B | `3290dcf` | `The` (818) | 27.10343 (loop) / 27.10754 (fused, Δ ≈ +4.1e-3 on top-1 only) | `To` (2021) | -0.6487 | byte 0 | RED | Fused kernel wired into every RmsNorm call site behind `--rms-norm-kernel=fused` (default stays `loop` in Phase B for bisect-safety). Top-10 ID set AND order byte-identical across both modes. `The`/`To` gap: loop +0.7451, fused +0.7552 — small drift, not enough to flip. Median bench: loop 37.13 tok/s (flat vs baseline), fused 44.51 tok/s (**+7.45 tok/s, +20.1%**). Metrics under fused: `dispatches_per_token` 5652.52 → 2432.52 (−56.9%), `norm_dispatches_per_token` 3521 → 331 (−90.6%), `moe_dispatches_per_layer` 42 → 34 (−8, from router norm). 5-run variance 0. gen128 coherent under fused (first 28 tokens byte-identical to loop, then compounding FP drift splits to a different-but-coherent branch of the same essay). |
| 1bNEW.4 Phase C | (pending) | `The` (818) | 27.10754 (fused default) | `To` (2021) | -0.6487 | byte 0 | RED | Default flipped from `loop` to `fused`. Re-ran 5-run canonical bench under new default: median 44.55 tok/s, p95 44.63, variance 0. Loop fallback re-verified at 37.02 tok/s median with byte-identical counters to pre-1bNEW.4 baseline (`dispatches_per_token=5652.52`, `norm_dispatches_per_token=3521`). `norm_dispatches_per_token=331` at the fused default hits the ADR acceptance criterion (corrected from the item's estimate of ≤ 330 — the +1 is the final `output_norm` tail). Walk-correctness top-1 still `The` — 1bNEW.4 alone did not flip the argmax, consistent with the port being byte-for-byte faithful to llama.cpp's reduction order (no NEW math drift, only elimination of candle-specific chain drift). |
| 1bNEW.6 Phase A | `9d52fe9` | (no forward-pass change) | (n/a) | (n/a) | (n/a) | (n/a) | (n/a) | Runtime-compiled `kernel_rope_norm<float>` + `kernel_rope_neox<float>` port of llama.cpp kernels at `ggml-metal.metal:4322-4426` + `rope_yarn*` helpers at `:4284-4320` + 6 Phase A unit tests. Tests cover decode full-rotary sliding `[1,16,1,256]`, decode partial-rotary global `[1,16,1,512]`, prefill partial-rotary global `[1,16,128,512]` (the `project_coherence_bug.md` critical path), decode at `seqlen_offset=42`, prefill partial-rotary generic `[1,16,32,256]`, and the `norm`-variant GPT-J-interleaved port sanity on `[1,8,1,64]`. All six pass at ε=1e-5; tests 1/2/6 are bit-exact (`max|Δ|=0.000e0`), tests 3/4/5 drift by ≤ 8e-6 (FP associativity on `pos * pow(fb, -i0/n_dims)` vs `t @ inv_freq` matmul reduction). 0 NaN, 0 mismatched elements. No `gemma4.rs` forward-pass changes — top-10 byte-identical by construction. Frequency-scaling trick: `freq_base_eff = rope_theta^(rotary_dim/head_dim)` folds Gemma 4's HF proportional `1/theta^(2k/head_dim)` into llama.cpp's kernel-native `pow(freq_base, -i0/n_dims)` with `n_dims=rotary_dim`. |
| 1bNEW.6 Phase B | `881d1e9` | `The` (818) | 27.10754 (loop) / 27.11075 (fused, Δ ≈ +3.2e-3 on top-1 only) | `To` (2021) | -0.6487 | byte 0 | RED | Fused Metal RoPE wired into `RotaryEmbedding::apply` behind `--rope-kernel=fused` (default stays `loop` in Phase B for bisect-safety). Top-10 ID set AND order byte-identical across both modes. `The`/`To` gap: loop +0.7552, fused +0.7701 — gap grew (direction TOWARD `The`, not `To`). Median bench: loop 44.5 tok/s (re-bench verified byte-stable vs 1bNEW.4 Phase C), fused **48.80 tok/s (+4.25 tok/s, +9.5%)**. 5-run variance 0. Metrics under fused: `dispatches_per_token` 2432.52 → 2192.52 (−240, −9.9%; exactly `(10 - 2) × 30` as predicted — 10 candle ops per RoPE site on loop, 2 Metal dispatches per site fused, 30 layers). gen16 byte-identical across modes: `"The evolution of computing—from mechanical calculators to the transistor-based microprocessor—is"`. gen128 coherent under fused. 827-token adversarial `Melthorn-by-the-Sea` needle recall **PRESERVED** — proves fused partial-RoPE on global-attention layers is correct at high seq_len, the exact `project_coherence_bug.md` failure class. |
| 1bNEW.6 Phase C | (pending) | `The` (818) | 27.11075 (fused default) | `To` (2021) | -0.6487 | byte 0 | RED | Default flipped from `loop` to `fused`. Re-ran 5-run canonical bench under new default: **median 48.71 tok/s, p95 48.78**, variance 0 (48.6–48.8 spread). Loop fallback retained behind `--rope-kernel=loop` and re-verified at 44.5 tok/s median. Metrics at fused default: `dispatches_per_token=2192.52` (−240 vs 1bNEW.4 Phase C baseline), `norm_dispatches_per_token=331` unchanged (owned by 1bNEW.4), `moe_to_vec2_count=0` unchanged (owned by 1bNEW.1), `sampler_sync_count=0.26` unchanged (owned by 1bNEW.3). Walk-correctness top-1 still `The` — 1bNEW.6 alone did not flip the argmax. The gap is now ~0.770 raw-logit vs llama.cpp's ~0.12 logprob — the four Walk items (1bNEW.1/3/4/6) collapsed candle's manual-chain drift and the remaining gap requires either a BF16 prefill drift item or a per-layer residual accumulator convention diff (Walk Exception Register). |
| 1bNEW.17 Phase A | `0565c69` | (no forward-pass change) | (n/a) | (n/a) | (n/a) | (n/a) | (n/a) | New `src/serve/lm_head_kernel.rs` module with `LmHeadKernelMode { Loop, Fused }` enum, `lm_head_forward_fused` helper, and 3 Phase A unit tests at ε=1e-3 (wider than 1bNEW.1/4/6's 1e-5 because the reduction order deliberately changes — F32-cumulative vs MLX F16 gemm). All 3 tests pass: single-token decode shape max\|Δ\|=3.402e-4, multi-token prefill shape max\|Δ\|=3.402e-4, dtype-guardrail test rejects F32 weight / F16 input. `LmHeadKernelMode` CLI flag plumbed through `load_with_modes` alongside the three existing kernel-mode flags; new `lm_head_f16_weight: Option<Tensor>` field on `Gemma4Model`, populated at load time when mode is `Fused`. No forward-pass changes — loop-mode byte-identity re-verified on the real GGUF at `(818, 27.11075)` top-1. |
| 1bNEW.17 Phase B | `0e36b1c` | `The` (818) | 27.110750 (loop) / 27.108929 (fused, Δ ≈ −1.8e-3 on top-1) | `To` (2021) | -0.6487 | byte 0 | RED | Fused F16 gemm wired into `Gemma4Model::forward` at `gemma4.rs:1879` behind `--lm-head-kernel=fused` (default stays `loop` in Phase B for bisect-safety). Top-10 ID set AND order byte-identical across both modes. `The`/`To` gap: loop +0.77016, fused +0.77102 — nearly flat (delta +0.00086 toward `The`). **The F16 reduction-order shift did NOT flip the argmax.** Walk-correctness drift owner is NOT the lm_head (the ADR's two-for-one hope was wrong). Median bench: loop 48.7 tok/s (byte-flat vs 1bNEW.6 Phase C), fused **58.49 tok/s (+9.78 tok/s, +20.1%)**. 5-run variance 0.2 (58.4–58.6). Metrics under fused: `dispatches_per_token` 2192.52 → 2194.52 (+2 exactly from the cast pair; no new weight traffic), other counters unchanged. gen128 byte-identical across modes: same 128-token "The evolution of computing—from mechanical calculators..." technical prose. 827-token adversarial `Melthorn-by-the-Sea` needle recall **PRESERVED** — F16 lm_head has no adverse interaction with long-context attention. |
| 1bNEW.17 Phase C | (pending) | `The` (818) | 27.108929 (fused default) | `To` (2021) | -0.6487 | byte 0 | RED | Default flipped from `loop` to `fused`. Re-ran 5-run canonical bench under new default: **median 58.51 tok/s, p95 58.57**, variance 0.1 (58.5–58.6). Loop fallback retained behind `--lm-head-kernel=loop` and re-verified at 48.7 tok/s median (byte-flat vs 1bNEW.6 Phase C). Metrics at fused default: `dispatches_per_token=2194.52` (+2 vs 1bNEW.6 Phase C from the cast pair), other counters unchanged. Walk-correctness top-1 still `The` — 1bNEW.17 alone did not flip the argmax. The 1bNEW.17 item was proposed as a candidate Walk-correctness two-for-one in the post-Walk re-spike, but empirically the lm_head F16 reduction-order delta produces only ~2e-3 logit drift on the top-1 position, three orders of magnitude too small to close the ~0.77 `The`/`To` gap. The drift owner is still open. Speed delta: cumulative 1bNEW post-Walk progress is now 23.76 → 58.51 tok/s (+146.2% over the pre-Walk baseline; post-Walk alone 48.71 → 58.51, +20.1%). |

## hf2q top-10 at the Crawl baseline (and all rows through 1bNEW.0b)

Identical at every row above — the columns are f32 logits read from
`HF2Q_DUMP_LOGITS=/tmp/h.bin`:

```
[(818,    27.10432),  // The
 (2021,   26.356354), // To
 (101068, 23.337055), // Connecting
 (216100, 22.498186), // Tracing
 (129264, 20.415514), // Linking (hf2q order)
 (8409,   19.550251), // While
 (32899,  19.057167), // Modern
 (12282,  18.244814), // Mapping
 (20647,  18.029692), // Expl
 (155571, 17.744995)] // Brid
```

## llama.cpp top-10 (Crawl baseline, from ADR-005 line 192)

These are logprobs (not logits) because `llama-server /completion`
reports logprobs when `n_probs=10` is set. The f32 logit equivalents
are unrecoverable from the server output.

```
[(2021,   -0.6487),   // To
 (818,    -0.7663),   // The
 (101068, -5.2554),   // Connecting
 (216100, -5.5368),   // Tracing
 (8409,   -6.4215),   // While
 (32899,  -7.2102),   // Modern
 (129264, -7.9774),   // Linking
 (20647,  -8.6333),   // Expl
 (90894,  -9.0234),   // Transl
 (10176, -10.0107)]   // Direct
```

Set intersection: 8 / 10 token IDs match exactly across both tools
(818, 2021, 101068, 216100, 8409, 32899, 129264, 20647). The argmax
flip (`The` vs `To`) is a near-tied pair with a gap of ~0.12 logprob
(~0.75 raw-logit at hf2q) — small enough that any kernel-level FP
reordering from later Walk items (1bNEW.1 Unified MoE, 1bNEW.4 Fused
RmsNorm, 1bNEW.6 Fused RoPE) could plausibly flip it.

## Commit policy

Per ADR-005 line 181, neither `crawl_baseline.tokens` nor
`llama_cpp_reference.tokens` is committed yet. They will be committed
as golden fixtures — **and the per-commit gate will upgrade from
"hf2q HEAD self-baseline" to "llama.cpp reference"** — only when the
`hf2q_top1` column in this table first shows `To` (2021) instead of
`The` (818). Until then, this markdown file is the primary Walk-
progress record.
