# hf2q Operator Environment Variables

Default behavior on a supported model class (e.g., Gemma-4 26B GGUF,
Qwen 3.5/3.6 GGUF):

- **Coherence:** matches the locked reference trajectories on the qualified
  family gates. Gemma executes the artifact's declared embedding and output
  matrix representation directly; it does not silently change the target
  head at load time.
- **Throughput** (M5 Max, APEX-Q5_K_M, post-ADR-032):
  - Gemma 4 26B-A4B-it: 1.51× AHEAD of llama.cpp `-fa 1` at tg2000,
    1.67× AHEAD at tg200.
  - Qwen 3.6 35B-A3B: 1.47× AHEAD of llama.cpp `-fa 1` at tg1500,
    1.31× AHEAD at tg200 — TQ-V active by default.
- No flags required — the defaults are the ones you want.

The env vars below are escape hatches and experimental toggles. In normal
operation none of them need to be set. Ordinary serving controls—context,
scheduler, slots, shared KV budget, repetition penalty, and thinking
defaults—are typed CLI/config fields, not hidden environment variables. See
[ADR-050](adr/ADR-050-operator-context-and-static-info.md) and the
[complete source inventory](env-var-inventory.md).

---

## Gemma matrix storage

Gemma embeds tokens and projects logits from the exact matrix encoding stored
in the GGUF. The loader checks that embedding gather and projection kernels
both support that encoding before mapping model storage. A tied output head
shares the embedding allocation; an explicit `output.weight` retains its own
declared encoding. There is no head-format environment override because
silently dequantizing or re-quantizing a served artifact would change its
model semantics. Unsupported encodings fail closed before allocation.

---

## Prefill path

| Var | Default | Values | Effect |
|---|---|---|---|
| `HF2Q_BATCHED_PREFILL` | on | `0`/`false`/`off` | Batched prefill (`forward_prefill_batched`) — the production path since ADR-028 iter-344 (default-flipped from per-token, which was 14-45× slower than peer at pp512–pp4096).  Coherence intact at every tested length up to pp3813 (4× sliding_window). Opt-out via `0`/`false`/`off` reverts to per-token prefill for parity diagnostics only. |
| `HF2Q_CROSS_SLOT_ADMIT` | off globally; `1` in the canonical Gemma and Qwen launchers | `0`/`1` | Allows a SlotAware worker to aggregate an already-runnable compatible FIFO prefix without skipping or waiting. Gemma applies one shared 4,096-row cap across all lanes. Qwen admits two to four cold text lanes only at an identical stable boundary of 16–128 rows per lane and publishes every lane atomically; `0` is the matched serial-control and incident-isolation path. This launcher-owned production policy is queued for typed internalization by ADR-050; it is not acceptable ordinary shell UX. |
| `HF2Q_ADMIT_COALESCE_US` | `25000` in the canonical Gemma and Qwen launchers | integer microseconds, `0..=100000` | Maximum idle-worker first-cohort collection window used only with cross-slot admission. A larger value may improve initial batching at the cost of admission latency; zero disables collection without disabling other cross-slot routes. Invalid direct-runtime values warn and disable collection. It does not relax FIFO, transaction-row, context, or KV-budget checks. |
| `HF2Q_F16_KV` | off | `1` | Allocate the dense KV cache as F16 instead of F32. Experimental — the current F16 path has a separate bug worse than F32; per ADR-009 the default F32 path is preferred. |
| `HF2Q_NO_FA` | off | `1`/`true`/`on` | Diagnostic A/B knob.  When set, routes the global D=512 attention path through F32 tensor-mm instead of flash-attention.  Forced off at `seq_len < 32` (the dense-matmul kernel requires K ≥ 32).  Per ADR-032 the FA path is the production default — peer-aligned with llama.cpp's `kernel_flash_attn_ext_*_dk512_dv512`.  This flag exists for bisection work against the tensor-mm reference, not for production use. |
| `HF2Q_FA_F16` | on | `0`/`false`/`off` | F16 (`half`, 10-bit mantissa) Q/K/V in flash-attention shared memory.  Matches llama.cpp's default `FA_TYPES` template specialisation for F16 KV cache (the standard production path).  Per ADR-032 this is the peer-aligned default — Q-shmem precision is the binding constraint on argmax stability at D=512 global layers (BF16's 7-bit mantissa accumulates ~9% relative error over a 512-element dot product, flipping argmax on narrow-margin greedy decode).  Opt-out reverts to BF16 (`bfloat`, 7-bit mantissa) shmem — peer's `FA_TYPES_BF` specialisation, only used in llama.cpp when KV cache is explicitly BF16.  Provided for diagnostic A/B against the BF16 instantiation; not for production. |

## Qwen reasoning and decode

The three request-omission defaults are no longer environment variables:

| CLI | `config.toml` | Built-in | Effect |
|---|---|---|---|
| `--default-repetition-penalty` | `[serve] repetition_penalty` | `1.0` | Server default only when the request omits `repetition_penalty`; explicit requests win. |
| `--default-thinking-token-budget` | `[serve] thinking_token_budget` | unset | Qwen reasoning ceiling when the request omits it; `0` disables the configured default. |
| `--default-tool-thinking-token-budget` | `[serve] tool_thinking_token_budget` | unset | First tool-continuation/required-tool ceiling; `0` disables the continuation override. |

CLI wins over setup config, which wins over the built-in. `hf2q setup` writes
the qualified agentic profile (`1.05`, `2048`, `512`) when the operator chooses
long agent/tool serving. The former `HF2Q_DEFAULT_*` readers and process bridge
were removed because they were both poor UX and initialization-order unsafe.

The remaining Qwen rows are technical routing escape hatches. The Qwen GGUF
loader resolves them once into immutable model-owned routing state; it does not
mutate process environment. ADR-050 still tracks replacement of their shell UX
with typed operator configuration.

| Var | Default | Values | Effect |
|---|---|---|---|
| `HF2Q_QWEN_SPECULATION` | `auto` | `off`, `auto` | Controls the live SlotAware Qwen server path. Default `auto` since 2026-08-21 (previously `off` outside the canonical Qwen3.8 launcher): `auto` measures ordinary decode first, tries exact request-history lookup (6-12-token match, up to three draft tokens), then fixed-K3 native MTP when available; each proposer disables itself for the generation after two consecutive four-round windows are not better than equivalent ordinary output. Stochastic sampling, logprobs, stop strings, logit bias, frequency/presence/min-p, parallel tool calls, and unsupported tool policy stay on ordinary target decode. Invalid values warn and fail safe to `off`. Explicit `off` remains the escape hatch. |
| `HF2Q_DECODE_MVN` | `1` | `0`, `1` | Controls exact-tree Q4_K/Q6_K multi-column matvec routing. All Qwen artifacts use the shared native default; model labels do not alter it. The multi-row kernels preserve the scalar accumulator and reduction order, and explicit operator values are resolved into immutable per-model routing without mutating process environment. |
| `HF2Q_DECODE_MV_EXT` | `0` | `0`, `1` | Experimental weight-amortized multi-column matvec. K-quants route only at widths 4-8; legacy Q4_0/Q8_0 retain widths 2-8. Its different reduction tree is not byte-exact: a Qwen3.8 repeated-verifier gate changed a target decision at completion token 206 despite an earlier four-position pass. Enabling it therefore invalidates exact speculative-decoding authority; production launchers leave it disabled. |
| `HF2Q_QWEN_GQA_Q2` | `auto` | `auto`, `off`/`0`/`false`, `on`/`1`/`true` | Qwen3.8 TQ-HB decode shares each KV-head load/dequantization across two query heads when the exact D=256/GQA/no-mask geometry is supported. `auto` selects it at KV length ≥8,192; `off` is the production escape hatch; `on` forces the candidate only where its hard geometry checks pass. Invalid values fail safe to `off`. Release requires the exact-output, thermally supervised short/long receipt in the shipping contract. |

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

Speed line: `shipping`. Default decode matches the locked reference coherence
on the locked gates at 1.31–1.67× of its throughput across Gemma 4
and Qwen 3.6 APEX-Q5_K_M (see "Default behavior" at top for the historical
per-regime measurements). Gemma's current output path uses the stored artifact
representation directly.
