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
