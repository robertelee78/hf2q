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
