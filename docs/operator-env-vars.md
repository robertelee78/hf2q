# hf2q Operator Environment Variables

Default behavior on a supported model class (e.g., Gemma-4 26B DWQ GGUF):

- **Coherence:** matches the F16 lm_head reference on the locked gates
  (sourdough byte-identical vs llama.cpp, sliding_wrap byte-identical
  vs the F16-hf2q reference).
- **Throughput:** ~98% of llama.cpp decode on the same model and prompt.
- No flags required — the defaults are the ones you want.

The env vars below are escape hatches and experimental toggles. In
normal operation none of them need to be set.

---

## lm_head path

The decoder's lm_head is the single biggest memory-bandwidth consumer
at batch=1. hf2q auto-selects between an F16 dense mat-vec and a Q8_0
quantized mat-vec based on the loaded weights' size, and when Q8 is
used, a CPU-side exact rerank recovers the F16 trajectory.

| Var | Default | Values | Effect |
|---|---|---|---|
| `HF2Q_LMHEAD_Q8` | auto | `1`, `0` | `1` forces Q8 (requires `hidden_size % 32 == 0`); `0` forces F16 (the escape hatch). Unset = auto: Q8 when F16 weight > 256 MB AND hidden_size % 32 == 0. |
| `HF2Q_LMHEAD_RERANK` | on when Q8 | `0` | `0` disables the exact-F32 rerank of top candidates, leaving raw Q8 argmax. **Unsafe** — Q8's ~5e-3 logit noise envelope occasionally flips a near-tiebreak (observed as rare mid-decode `<pad>` emission). Only set for speed-vs-correctness benchmarking. |
| `HF2Q_LMHEAD_COMPARE` | off | `1` | Keeps both F16 and Q8 buffers resident so a future A/B diagnostic can compare logits at every step. Not wired into the live decode path today. |

**Why the default is Q8+rerank on large-vocab models:** Q8 alone
recovers ~12% decode throughput vs F16 by halving the lm_head weight
traffic (1.47 GB → 784 MB on Gemma-4 26B). The rerank adds ~0.4% back
of overhead and preserves F16 output byte-for-byte on the locked gates.
A dormant GPU top-K kernel exists (mlx-native `top_k_f32`); it is
intentionally unused because for vocab=262144 the CPU threshold scan
costs ~40 μs/token while the single-threadgroup GPU kernel costs
~5 ms/token on phase-2 serial extraction. If a parallel-phase-2 redesign
lands, the GPU path can be wired in without changing the Rust-side
rerank logic.

---

## Prefill path

| Var | Default | Values | Effect |
|---|---|---|---|
| `HF2Q_BATCHED_PREFILL` | off | `1` | Use the experimental batched prefill (`forward_prefill_batched`) instead of per-token prefill. Retained for parity diagnostics; per-token is the production path. |
| `HF2Q_F16_KV` | off | `1` | Allocate the dense KV cache as F16 instead of F32. Experimental — the current F16 path has a separate bug worse than F32; per ADR-009 the default F32 path is preferred. |

## Dense KV / decode layout

`dense_kv_capacity` is sized per-layer at prefill time. Sliding layers
use a ring buffer capped at `sliding_window` (1024 on Gemma-4); global
layers use a linear buffer of `seq_len + max_decode_tokens`. No env var
controls this — it's a correctness property, not a tunable.

## Diagnostic dumps

These are for investigation work only. Output goes to `HF2Q_DUMP_DIR`
(defaults to `/tmp`).

| Var | Values | Effect |
|---|---|---|
| `HF2Q_PREFILL_DUMP` | `"L,T"` | Dump the full Q/K/V norm chain at (layer L, token T) during per-token prefill. |
| `HF2Q_BATCHED_DUMP` | `"L,T"` | Same as above but for batched prefill: dumps pf_q_normed_row, pf_k_normed_row, dense KV cache slice, etc. |
| `HF2Q_BATCHED_LAYER_SCAN` | `T` | Dump pf_hidden row T at the start of EVERY layer (used for cross-layer drift bisection). |
| `HF2Q_DUMP_LAYERS` | `<seq_pos>` | Enable decode-time hidden-state dumps at a given position. |
| `HF2Q_DUMP_BOUNDARY` | `<seq_pos>` | Dump pre-lm_head hidden + logits + top-10 argmax for a specific decode position. |
| `HF2Q_DUMP_ALL_CACHE` | `1` | When dumping, include the full cached K,V tensors (not just current-layer). |
| `HF2Q_DUMP_NORM_WEIGHT` | `<layer>` | One-shot dump of `input_layernorm.weight` as hf2q sees it (used to verify against GGUF). |

## Perf diagnostics

| Var | Values | Effect |
|---|---|---|
| `HF2Q_MLX_TIMING` | `1` | Log per-token encode/gpu_wait times, dispatch+barrier counts. |
| `HF2Q_SPLIT_TIMING` | `1` | Insert an extra commit-and-wait between body and head to measure them separately (~50 μs overhead). |
| `HF2Q_MLX_KERNEL_PROFILE` | `1` | Per-op kernel profile mode (runs one commit per op — heavy overhead, useful for relative attribution only). |
| `HF2Q_DUAL_BUFFER` | `3` | Split the decode forward into two command buffers after layer N (0 = disabled). Default is 3, which overlaps buf0's early layers with buf1 encoding on the CPU. |
| `HF2Q_GRAPH_OPT` | off | `1` | Use `begin_recorded` + `finish_optimized` for the decode session. Fusion/reorder pass runs; currently yields no measurable win because the big candidates are already expressed as fused kernels and the reorder pass aborts on unannotated dispatches. |

---

## Status

Parity investigation (ADR-010): `deferred`. Batched long-sequence parity
against llama.cpp remains open at the ~752-byte sliding_wrap level.
This is a numerical MoE top-K threshold sensitivity in L6 — not a
fixable single-kernel mismatch. Closing it would require pervasive
pre-MoE kernel alignment (option 1 in the ADR). Not pursuing in the
current phase.

Speed line: `shipping`. Default decode matches llama.cpp's coherence
on the locked gates at ~98% of its throughput. Q8+rerank is the
production lm_head strategy; F16 remains available via
`HF2Q_LMHEAD_Q8=0`.
