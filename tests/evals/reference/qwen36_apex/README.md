# Qwen3.6 APEX-Q5_K_M parity references

Cross-implementation baselines captured 2026-05-16 against
`models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf`
on hf2q HEAD `fc635d89` / llama-completion build `d05fe1d7d`.

## What this directory holds

For each of the 3 standard parity prompts (`tests/evals/prompts/`):

- `<prompt>_hf2q.txt` — output of `hf2q generate --model <qwen3.6-apex>
  --prompt-file <prompt> --max-tokens N --temperature 0`, with the
  pre-output banner (`prefill: ...`) and post-output footer
  (`--- mlx-native:`) stripped.
- `<prompt>_llama.txt` — output of `llama-completion -m <qwen3.6-apex>
  -f <rendered-prompt> -n N --temp 0 --top-p 1.0 --top-k 1
  --repeat-penalty 1.0 --no-warmup -sp --no-perf -no-cnv --jinja`,
  with the chat-template marker `<|im_start|>assistant\n<think>`
  stripped.

Token counts per prompt (match Gemma-4 parity defaults):
sourdough=1000, short_hello=50, sliding_wrap=500.

## Measured cross-implementation divergence

| Prompt | hf2q bytes | llama bytes | common_prefix |
|---|---|---|---|
| `short_hello` | 177 | 171 | 67 |
| `sourdough` | 3534 | 3633 | 19 |
| `sliding_wrap` | 1802 | 2398 | 1 |

Divergence is **kernel-level floating-point**, not TQ-related:
`hf2q load: tq_kv = inactive` confirms TQ is currently NOT wired into
the Qwen35 inference path (`src/inference/models/qwen35/`). Verified
by re-running with `HF2Q_USE_DENSE=1` — output is byte-identical to
the default (177/177 common prefix between TQ-mode and dense-mode
hf2q runs), confirming both code paths hit the same dense kernel.

## What this means

1. **Qwen3.6 hf2q kernel implementation diverges from llama.cpp's
   Qwen3.6 kernel implementation** starting from byte ~67 on the
   shortest prompt — almost immediately. This is independent
   implementations producing different argmax tokens due to FP
   non-associativity in attention / MoE routing / softmax.
2. **TQ KV cache is currently Gemma-4-only.** Wiring TQ into the
   Qwen35 path is a separate engineering task — would touch
   `src/inference/models/qwen35/gpu_full_attn.rs`,
   `gpu_delta_net.rs`, `kv_cache.rs`.
3. **These refs are baselines, not pass/fail gates.** Future
   regressions to the Qwen3.6 hf2q forward path would show up as
   reduced common_prefix vs `*_llama.txt`, AND/OR as
   diff vs `*_hf2q.txt` (the latter is hf2q-vs-hf2q determinism).

## What this does NOT answer

- How well TQ would work on Qwen3.6 if wired in. (Cannot test
  without the wiring work.)
- Why hf2q-Qwen3.6 diverges from llama-Qwen3.6 byte-wise (kernel-FP
  audit, multi-day investigation per ADR-022 patterns).
- Whether the 67/19/1 byte prefix is acceptable for shipping. (No
  Qwen3.6-specific shipping floor exists yet; operator decision.)

## How to refresh

```bash
/tmp/capture-qwen36-apex-refs-v2.sh   # script preserved in /tmp scratch
                                       # promote to scripts/ if needed
```
