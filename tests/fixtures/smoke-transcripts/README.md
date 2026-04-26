# Smoke transcripts (ADR-012 Decision 16)

Each `<arch>-<quant>.txt` file is a `llama-cli -n 8 --seed 42 --temp 0
--no-warmup` transcript captured by `hf2q smoke --arch <X> --quant <Y>` on
hardware.  Transcripts are committed only after a human has run them; CI does
not regenerate them.

## Why these are tracked artifacts

These are the **evidence** that ADR-012 closure is real.  Per Decision 16:

* Exactly 8 generated tokens (`llama_print_timings: n_eval = 8`)
* No `error|ERROR|panic|assertion|segfault` lines
* Tensor-count sanity: at least the registered catalog's pattern count

If the same `hf2q smoke` invocation produces a different transcript on the
same M5 Max, the determinism gate (`--seed 42 --temp 0`) is broken — the
transcript files exist precisely to surface that regression on the day a
fresh build hits the smoke gate.

## File-naming contract

```
{arch}-{quant}.txt           # text-side smoke transcript
{arch}-{quant}-vision.txt    # paired mmproj smoke (qwen35 dense only)
```

## Adding a new arch

A new arch landing in its own ADR (Gemma4 parity, ADR-015 Ministral, ADR-016
DeepSeek-V3) appends transcripts here under its own filename — no rewrite of
this README, no harness changes per Decision 20.

## Current contents

| File | Status | Notes |
|---|---|---|
| `qwen35-q4_0.txt` | pending real-model run | requires `Qwen/Qwen3.6-27B` + ≥ 65 GB free |
| `qwen35moe-q4_0.txt` | pending real-model run | requires `jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated` + ≥ 160 GB free |

P9 will add four DWQ transcripts (`{arch}-dwq4{6,8}.txt`) with inline PPL
and KL numbers per Decision 17.  P10 will add `qwen35-q4_0-vision.txt`.
