# hf2q Operator Environment Variables

Default behavior on a supported model class (e.g., Gemma-4 26B GGUF,
Qwen 3.5/3.6 GGUF):

- **Coherence:** matches the F16 lm_head reference on the locked gates
  (Qwen sourdough byte-identical vs llama.cpp; Gemma 300-Format greedy
  enumeration coherent through 94+ items; chemistry+thinking probe
  first-token byte-identical vs llama.cpp `--reasoning on`).
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
| `HF2Q_BATCHED_PREFILL` | on | `0`/`false`/`off` | Batched prefill (`forward_prefill_batched`) — the production path since ADR-028 iter-344 (default-flipped from per-token, which was 14-45× slower than peer at pp512–pp4096).  Coherence intact at every tested length up to pp3813 (4× sliding_window). Opt-out via `0`/`false`/`off` reverts to per-token prefill for parity diagnostics only. |
| `HF2Q_CROSS_SLOT_ADMIT` | off globally; `1` in the canonical Gemma launcher | `1` | Allows the Gemma SlotAware admission loop to coalesce one contiguous FIFO class into a multi-sequence prefill. The engine applies one shared 4,096-row transaction cap across all lanes; it does not multiply the cap by slot count. This launcher-owned production policy is queued for typed internalization by ADR-050; it is not acceptable ordinary shell UX. |
| `HF2Q_ADMIT_COALESCE_US` | `25000` in the canonical Gemma launcher | positive integer microseconds | Maximum first-cohort collection window used only with cross-slot admission. A larger value may improve initial batching at the cost of admission latency; it does not relax FIFO, transaction-row, context, or KV-budget checks. This is part of the same typed-admission follow-up, not a recommended user knob. |
| `HF2Q_F16_KV` | off | `1` | Allocate the dense KV cache as F16 instead of F32. Experimental — the current F16 path has a separate bug worse than F32; per ADR-009 the default F32 path is preferred. |
| `HF2Q_NO_FA` | off | `1`/`true`/`on` | Diagnostic A/B knob.  When set, routes the global D=512 attention path through F32 tensor-mm instead of flash-attention.  Forced off at `seq_len < 32` (the dense-matmul kernel requires K ≥ 32).  Per ADR-032 the FA path is the production default — peer-aligned with llama.cpp's `kernel_flash_attn_ext_*_dk512_dv512`.  This flag exists for bisection work against the tensor-mm reference, not for production use. |
| `HF2Q_FA_F16` | on | `0`/`false`/`off` | F16 (`half`, 10-bit mantissa) Q/K/V in flash-attention shared memory.  Matches llama.cpp's default `FA_TYPES` template specialisation for F16 KV cache (the standard production path).  Per ADR-032 this is the peer-aligned default — Q-shmem precision is the binding constraint on argmax stability at D=512 global layers (BF16's 7-bit mantissa accumulates ~9% relative error over a 512-element dot product, flipping argmax on narrow-margin greedy decode).  Opt-out reverts to BF16 (`bfloat`, 7-bit mantissa) shmem — peer's `FA_TYPES_BF` specialisation, only used in llama.cpp when KV cache is explicitly BF16.  Provided for diagnostic A/B against the BF16 instantiation; not for production. |

## Qwen reasoning and decode

The three request-omission defaults are no longer environment variables, and
their `[serve]` config keys are retired (ADR-062 D1: parse warns and ignores;
setup no longer writes them):

| CLI | Family built-in (ADR-062 D1) | Effect |
|---|---|---|
| `--default-repetition-penalty` | Qwen/Gemma 4 `1.05`, DeepSeek-V4 `1.0` | Server default only when the request omits `repetition_penalty`; explicit requests win. |
| `--default-thinking-token-budget` | Qwen `2048`, others unset | Thinking ceiling when the request omits it; `0` disables the configured default. |
| `--default-tool-thinking-token-budget` | Qwen/DeepSeek-V4 `512`, others unset | Tool-continuation/required-tool ceiling, the same at every tool-chain depth; `0` disables the continuation override. |

CLI wins over the loaded family's built-in. The former `HF2Q_DEFAULT_*`
readers and process bridge were removed because they were both poor UX and
initialization-order unsafe.

The remaining Qwen rows are technical routing escape hatches. ADR-050's
inventory marks speculation and process-global decode routing for typed backend
policy: their current environment implementation is not the final multi-model
architecture.

| Var | Default | Values | Effect |
|---|---|---|---|
| `HF2Q_QWEN_SPECULATION` | `auto` | `off`, `auto` | Controls the live SlotAware Qwen server path. Default `auto` since 2026-08-21 (previously `off` outside the canonical Qwen3.8 launcher): `auto` measures ordinary decode first, tries exact request-history lookup (6-12-token match, up to three draft tokens), then fixed-K3 native MTP when available; each proposer disables itself for the generation after two consecutive four-round windows are not better than equivalent ordinary output. Stochastic sampling, logprobs, stop strings, logit bias, frequency/presence/min-p, parallel tool calls, and unsupported tool policy stay on ordinary target decode. Invalid values warn and fail safe to `off`. Explicit `off` remains the escape hatch. |
| `HF2Q_DECODE_MVN` | `1`; `0` applied automatically when loading a Qwen3.8-identified model | `0`, `1` | Controls exact-tree Q4_K/Q6_K multi-column matvec routing. The Qwen3.8-qualified route disables it because the K=3 verifier is qualified on the weight-amortized width-four route. Previously only `serve_qwen38_opencode.sh` applied this; `hf2q serve` now applies it at engine load for Qwen3.8-identified models. |
| `HF2Q_DECODE_MV_EXT` | `0`; `1` applied automatically when loading a Qwen3.8-identified model | `0`, `1` | Enables weight-amortized multi-column matvec. K-quants route only at widths 4-8; legacy Q4_0/Q8_0 retain widths 2-8. Enabled after the real-model four-position decision, hybrid-cache cursor, and eight-step continuation gate. Unlike the byte-exact default-on mvN route, `mul_mv_ext` is not bit-exact, so the default stays Qwen3.8-scoped — no other family carries the qualifying receipt. |
| `HF2Q_QWEN_GQA_Q2` | `auto` | `auto`, `off`/`0`/`false`, `on`/`1`/`true` | Qwen3.8 TQ-HB decode shares each KV-head load/dequantization across two query heads when the exact D=256/GQA/no-mask geometry is supported. `auto` selects it at KV length ≥8,192; `off` is the production escape hatch; `on` forces the candidate only where its hard geometry checks pass. Invalid values fail safe to `off`. Release requires the exact-output, thermally supervised short/long receipt in the shipping contract. |

## Qwen joint decode

With two or more Qwen agents decoding at once, eligible slots can advance through one shared target forward per scheduler turn instead of one forward per slot. Joint rows run ordinary target decode: MTP drafting and history proposals are suppressed for them (slots with MTP state still run their MTP catch-up, so a later solo turn resumes K3). With fewer than two eligible slots the turn is serial, including MTP K3 and history speculation.

Under the default exact routing policy each joint row is bitwise identical to its serial target forward and independent of which slots share its step; per-request output can still differ from stock because joint rows do not speculate. Under the Qwen3.8 production policy (`HF2Q_DECODE_MV_EXT=1`) dense artifacts use the padded `mul_mv_ext` mode: every step packs exactly 4 rows (joint width is capped at 4), so a row's packed width never depends on how many tenants share the step. That mode is an ADR-063 Q1 Class B route against stock (decision envelope, not bitwise; `docs/research/adr063-q1-numeric-policy.md`).

Joint decode is on by default for dense artifacts (Qwen3.8) and off by default for MoE artifacts (Qwen3.6), whose four-agent A/B is incomplete. MoE artifacts with `HF2Q_DECODE_MV_EXT=1`, block storage outside Q4_0/Q8_0/Q4_K/Q5_K/Q6_K, float FFN storage, GLP steering, layer dumps, `HF2Q_LEGACY_PER_LAYER_CB=1` and `MLX_UNRETAINED_REFS=1` keep serial decode. The decision and its reason are logged once at worker start; the routing policy is re-checked every turn. See [ADR-063 S3](research/adr063-s3-qwen-joint-decode.md).

| Var | Default | Values | Effect |
|---|---|---|---|
| `HF2Q_QWEN_JOINT_DECODE` | on for dense (Qwen3.8), off for MoE (Qwen3.6) | `0`/`off`/`false`, `1`/`on`/`true` | Escape hatch and A/B switch. `0` restores serial per-slot decode; `1` opts MoE artifacts in. Read once at worker start. Invalid values warn and use the per-family default. |
| `HF2Q_QWEN_JOINT_DECODE_MAX_ROWS` | `4` | `2`–`8` | Widest joint step; further eligible slots stay serial that turn. The padded `mul_mv_ext` mode caps at 4 regardless. Widths above 4 have no serving measurement. Invalid values warn and use 4. |

## DeepSeek-V4 decode

With two or more agents decoding at once, DeepSeek-V4 advances every eligible slot through one joint forward per token step. Each row is bitwise identical to serial decode. Joint decode engages only for artifacts whose batched products all use proven-exact storage types: Q8_0, Q2_K or Q3_K; float weights only where they are split per row; grouped output-A only as Q8_0. Other artifacts stay serial, and the reason is logged at worker start. See [ADR-063 S3](research/adr063-s3-ds4-joint-decode.md).

| Var | Default | Values | Effect |
|---|---|---|---|
| `HF2Q_DEEPSEEK_JOINT_DECODE` | on | `0`/`off`/`false`, `1`/`on`/`true` | Escape hatch. `0` restores serial per-slot decode, including the exact B=4 cohort. Read once at worker start. It is also forced serial by `MLX_UNRETAINED_REFS`, DeepSeek layer dumps or stage profiling, and `HF2Q_DECODE_MV_EXT=1`. Loading a Qwen3.8 model sets `HF2Q_DECODE_MV_EXT=1` for the whole process; DeepSeek turns then decode serially. |
| `HF2Q_DEEPSEEK_JOINT_DECODE_MAX_ROWS` | `4` | `2`–`8` | Widest joint step. Widths above 4 are covered only by the official-artifact bitwise test and have no serving measurement; use them only for experiments. Invalid values warn and use 4. |

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
on the locked gates at 1.31–1.67× of its throughput across Gemma 4
and Qwen 3.6 APEX-Q5_K_M (see "Default behavior" at top for per-regime
numbers).  Q8+rerank is the production lm_head strategy; F16 remains
available via `HF2Q_LMHEAD_Q8=0`.

### Experimental Qwen scratch-pool creation profile

`HF2Q_QWEN_POOL_GROWTH_PROFILE=adr063-24g-v1` requires
`HF2Q_UNSAFE_EXPERIMENTS=1`. Unset leaves the legacy path unchanged. Unknown
values or a missing acknowledgment refuse requested startup; there is no
uncapped fallback. This Category-3 profile supports the audited native
DenseQ/MoeQ Qwen SlotAware text/tool route with one worker and one generation,
1–4 slots. Unsupported vision/embedding/diagnostic request paths explicitly
error under this profile; canonical projector loading is unchanged.

The fixed enforcing policy is 24GiB aggregate successful bucket creation,
16,384 roots, 32,768 constructor attempts, rank8 and batch1, with 35 exact
power-of-two buckets1..16GiB. Free reuse consumes no new creation credit;
reset, clear, errors and teardown never refund successful creation history.
Failure is sticky and stops selected replay/reuse; recovery requires a new
process, not automatic retries or a larger cap. These are operating ceilings,
not measured requirements or a whole-memory admission decision. Direct arenas,
weights/caches, native convenience scratch, IdMm, MTP, CPU/driver metadata and
external competition remain separate. A later smaller/larger named policy is
a new reviewed packet; the program does not resize this one automatically.
Real-model execution remains separately gated on a complete memory packet.

### Gemma full-input checkpoint (default on)

`HF2Q_GEMMA_FULL_INPUT_CHECKPOINT` controls the Gemma 4 full-input checkpoint.
No unsafe acknowledgment is needed.
- **Unset (default):** on for a Gemma Hybrid/TQ8 SlotAware worker with 1–4
  slots whose provisioned geometry holds two CPU anchor generations per slot,
  at the longest prompt ledger, inside the 1536 MiB requested-anchor quota.
  Otherwise it stays off, and startup does not refuse.
- **`0`:** kill switch. Tool-result turns reuse only the strict cue-less
  anchor.
- **`1`:** forces it on. Startup refuses if it cannot activate.
- **Any other value:** refuses startup.

When active, a native tool-result turn whose cue-less render equals the prompt
captures the slot's sliding rings at the completed input. There is no extra
forward or sample. The next turn restores the capture and prefills only the
appended suffix. The quota covers all Hybrid snapshot constructors on that
worker, including projected token-ledger and metadata bytes; payloads drop
before the refund. Each worker has its own quota. Any refused capture keeps
the previous anchor and serves the request unchanged. `0` also skips the
slot-route load warmup, so the kill-switch arm is stock serving. Allocator/OS overhead and other CPU owners are
outside this domain, so this is not a physical admission cap. Evidence:
`docs/research/adr063-gemma-four-agent-reuse.md`.

### Gemma joint warm prefill (default on)

`HF2Q_GEMMA_JOINT_WARM_PREFILL` controls ADR-063 Stage 1 joint warm prefill
on the Gemma 4 SlotAware worker. It is read once when the worker starts.
- **Unset, `1`, `on` or `true` (default):** when every slot is idle, a
  contiguous run of warm text requests in one admission quantum (tool-result
  turns resuming a full-input checkpoint, or any request whose slot holds a
  reusable prefix) is admitted without a yield between requests and
  prefilled as one transaction. Projections, norms, FFN/MoE and the output
  head run once over every suffix row; attention runs per slot with the
  single-slot kernel. Each request's tokens, logprobs and cache bytes are
  bitwise equal to the per-slot prefill. A lane joins only if its suffix is
  longer than 32 tokens and the cohort stays within 4,096 rows; other
  requests prefill per slot as before.
- **`0`, `off` or `false`:** kill switch; every warm request prefills per
  slot.
- **Any other value:** logs a warning and keeps the default.

It is also off when `HF2Q_CROSS_SLOT_ADMIT` is off, `HF2Q_PREFILL_SLOT_BATCHED=0`
or `HF2Q_MM_ID_ROUTING_THRESHOLD` is set, because those change the per-slot
route the cohort must equal. The worker logs `Gemma joint warm prefill worker
decision` at start, and each cohort logs `Gemma4 installed prefills advanced in
one aggregate-bounded transaction` with `transaction="joint warm"`. Evidence:
`docs/research/adr063-gemma-four-agent-reuse.md`.

### Gemma joint cue replay (default on)

`HF2Q_GEMMA_JOINT_CUE` controls how the Gemma 4 SlotAware worker replays the
short generation cue after a request's stable prompt boundary. It is read once
when the worker starts.
- **Unset, `1`, `on` or `true` (default):** when one admission quantum holds
  several stable-split requests that cannot form a stable batch (cold turns,
  or cues shorter than 32 tokens), each is still prefilled to its stable
  boundary and checkpointed per slot in arrival order. Their cues are then
  replayed together: one batched decode-body step per cue position for all
  lanes, and the batched output head on each lane's last cue token. Each
  request's tokens, logprobs and cache bytes are bitwise equal to the
  per-slot cue replay (one single-slot decode forward per cue token).
- **`0`, `off` or `false`:** kill switch; every cue replays per slot.
- **Any other value:** logs a warning and keeps the default.

It is also off when `HF2Q_CROSS_SLOT_ADMIT` is off, `HF2Q_BATCHED_BODY=0` or
hybrid KV is off, and a cohort replays per slot under any non-default GGML
routing policy (for example `HF2Q_DECODE_MV_EXT=1`, which loading a Qwen3.8
model sets for the whole process). The worker logs `Gemma joint cue replay
worker decision` at start, and each cohort logs `Gemma4 joint cue replay`.
Evidence: `docs/research/adr063-gemma-four-agent-reuse.md`.
