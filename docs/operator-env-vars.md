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
| `HF2Q_QWEN_SPECULATION` | unset: `off` on SlotAware workers that selected joint decode and have `max_slots` ≥ 4 (dense Qwen3.8 and MoE Qwen3.6); `auto` otherwise (SerialFifo, fewer than 4 slots, joint decode off or unavailable) | `off`, `auto` | Controls the live SlotAware Qwen server path. `auto` observes ordinary decode first, tries exact request-history lookup (6-12-token match, up to three draft tokens), then fixed-K3 native MTP when available; each proposer disables itself for the generation after two consecutive four-round windows that are not better than equivalent ordinary output, judged as `HF2Q_QWEN_SPECULATION_ROUTE` selects. Stochastic sampling, logprobs, stop strings, logit bias, frequency/presence/min-p, parallel tool calls, and unsupported tool policy stay on ordinary target decode. The unset default resolves once at worker start from the slot count and joint-decode decision, logged as `Qwen speculation worker policy`. Dense (Qwen3.8): at four slots with joint decode, `auto` was slower than `off` (own-output makespan ratio 1.020, 95% CI 1.0095–1.0308, 8 adjacent ABBA pairs, every pair slower, `ExactTranscripts`; receipts `$E/d1-speculation-ext-v2` with `E=~/.local/state/hf2q/campaigns/adr063-preflight-20261023`; `docs/research/adr063-d1-speculation-qualification.md` §7). MoE (Qwen3.6): off since 2026-11-06 for repeatable four-agent serving (operator ruling; supported by #719, where per-slot serial K3 at N4 cost about 25% against mostly-joint M1 rows; `docs/research/adr063-qwen-spec-repeatable.md`). Explicit values always win: `auto` opts back in, `off` is the escape hatch. Invalid values warn and fail safe to `off`. |
| `HF2Q_QWEN_SPECULATION_ROUTE` | `repeatable` | `repeatable`, `adaptive` | How speculating SlotAware lanes are routed and gated. `repeatable`: each tick's route (commit verified output, history round, K3 round, or ordinary decision) is decided from the request's own state only; a lane that speculates never joins a joint step, and the proposer gates compare acceptance with fixed round costs (MTP 1.26, history 1.49 ordinary decisions, from the D2 receipts' controller counters) instead of wall time, so identical requests take identical routes. `adaptive`: the phase-1 behaviour, where a speculation-capable lane joins joint steps as an M1 row when it has company and the gates compare measured wall time; kept for A/B and as an escape hatch. Read once at engine load and logged with the worker policy. Invalid values warn and use `repeatable`. `/metrics` proposer timing stays wall-clock under both. |
| `HF2Q_QWEN_LA_GPU_RESTORE` | unset: GPU restore for dense Qwen3.8; host restore for Qwen3.6 MoE | `0`, `1` | SlotAware Qwen speculative partial-reject rollback. `1` restores DeltaNet recurrent and conv state directly from the capture buffers in one GPU command buffer; `0` forces the previous host copy. The GPU route is bitwise equal to the host route. Read once when the SlotAware cache is provisioned; invalid values fail startup. Qwen3.6 remains host-default; explicit `1` is still an operator override. |
| `HF2Q_QWEN_ANCHOR_DEPTH` | `1` | integer `1`–`8` | Maximum committed slot-local prompt anchors. `1` is the C2 kill switch and current shipped default; pending construction is additional. `4` is the multi-depth qualification shape. The worker selects the deepest committed exact prefix, publishes captures only after successful decode completion, and prunes descendants before rewinds. Read once at SlotAware startup; invalid values fail startup. |
| `HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX` | unset | positive decimal bytes | Optional hard quota for committed plus pending anchor-owned image, logits, speculative hidden row, tokens, and metadata in one slot. A capture that would exceed the quota is skipped and never published. Unset relies on Q5 admission. Read once at SlotAware startup; invalid values fail startup. |
| `HF2Q_DECODE_MVN` | `1`; `0` for a Qwen3.8-identified model unless set explicitly | `0`, `1` | Controls exact-tree Q4_K/Q6_K multi-column matvec routing. The Qwen3.8-qualified route disables it because the K=3 verifier is qualified on the weight-amortized width-four route. That default is resolved per model at load from the model's identity and frozen into its kernel registries; the process environment is never written, so other models in the same process keep `1` whatever the load order. An explicit value is an operator override and applies to every model in the process. |
| `HF2Q_DECODE_MV_EXT` | `0`; `1` for a Qwen3.8-identified model unless set explicitly | `0`, `1` | Enables weight-amortized multi-column matvec. K-quants route only at widths 4-8; legacy Q4_0/Q8_0 retain widths 2-8. Enabled after the real-model four-position decision, hybrid-cache cursor, and eight-step continuation gate. Unlike the byte-exact default-on mvN route, `mul_mv_ext` is not bit-exact, so the default stays Qwen3.8-scoped — no other family carries the qualifying receipt. The Qwen3.8 default is resolved per model at load and frozen into its kernel registries; an explicit value is an operator override and applies to every model in the process. |
| `HF2Q_QWEN_ATTN_ROWS` | on | `0`/`off`/`false`, `1`/`on`/`true` | Qwen TQ full-attention decode. On: joint decode and the greedy single-command-buffer serial layer run every row's K/V encode, Q rotation, attention and output gate in one pass (no per-row barriers or row copies), and `HF2Q_QWEN_GQA_Q2=auto` uses the measured Qwen3.6 (16/2 heads) crossovers. `0` restores the per-row chain and the single 8,192 crossover. Both give the same bits. Read once. Invalid values warn and use on. |
| `HF2Q_QWEN_GQA_Q2` | `auto` | `auto`, `off`/`0`/`false`, `on`/`1`/`true` | Qwen3.8 TQ-HB decode shares each KV-head load/dequantization across two query heads when the exact D=256/GQA/no-mask geometry is supported. `auto` selects it at KV length ≥8,192 (Qwen3.6's 16/2-head geometry with `HF2Q_QWEN_ATTN_ROWS` on: ≥4,096 for one row, ≥1,024 when several rows share the pass); `off` is the production escape hatch; `on` forces the candidate only where its hard geometry checks pass. Invalid values fail safe to `off`. Release requires the exact-output, thermally supervised short/long receipt in the shipping contract. |

## DFlash2

DFlash2 remains opt-in and requires an identity-matched target/drafter pair.
Unknown revisions, source identities, tensor manifests and family adapters
fail before serving.

| Var | Default | Values | Effect |
|---|---|---|---|
| `HF2Q_DFLASH2` | off | `1` | Provisions the pinned DFlash2 checkpoint selected by the loaded target. Both checkpoint path variables below are required. |
| `HF2Q_DFLASH2_CONFIG` | unset | path | Exact checkpoint configuration. Its SHA-256, schema and target binding must match a compiled descriptor. |
| `HF2Q_DFLASH2_WEIGHTS` | unset | path | Exact safetensors checkpoint. Its SHA-256, byte length and tensor manifest must match the same descriptor. |
| `HF2Q_DFLASH2_DRAFT_AF4_G64` | off | `1` | Uses source-bound group-64 affine four-bit draft projections (36 bytes per 64 weights) instead of BF16. Qwen3.8 and Qwen3.6 Q5 have passed; the route remains opt-in until every activated family is qualified. |
| `HF2Q_DFLASH2_GPU_TARGET_TOKENS` | on for the pinned dense Qwen3.8 descriptor; off otherwise | `0`/`off`/`false`, `1`/`on`/`true` | Keeps selected proposals on the GPU through target embedding, removing the draft-to-target host wait and token packing. `0` restores staged host handoff. Explicit `1` on Qwen MoE or Gemma fails checkpoint load: their dynamic verification-row and target-state transaction is not yet qualified. Invalid values fail load. |

## Qwen joint decode

With two or more Qwen agents decoding at once, eligible slots can advance through one shared target forward per scheduler turn instead of one forward per slot. Joint rows run ordinary target decode: MTP drafting and history proposals are suppressed for them (slots with MTP state still run their MTP catch-up, so a later solo turn resumes K3). In exact mode, a turn with fewer than two eligible slots is serial, including MTP K3 and history speculation, and slots beyond the joint width stay serial that turn. In the padded mode (below), every eligible slot runs in consecutive joint steps of at most the joint width. That includes a lone slot whose next tick is ordinary target decode, so an ordinary request's arithmetic does not depend on which tenants share its turns (#717). A lone slot with MTP or history speculation state keeps its serial speculative round.

Under the default exact routing policy each joint row is bitwise identical to its serial target forward and independent of which slots share its step; per-request output can still differ from stock because joint rows do not speculate. Under the Qwen3.8 production policy (the per-model Qwen3.8 default, or an explicit `HF2Q_DECODE_MV_EXT=1`) dense artifacts use the padded `mul_mv_ext` mode: every step packs exactly 4 rows (joint width is capped at 4), so a row's packed width never depends on how many tenants share the step. In that mode the dense FFN gate and up products are also shared across the packed rows (each weight read once per step) unless `HF2Q_QWEN_JOINT_FFN_SHARED=0`. That mode is an ADR-063 Q1 Class B route against stock (decision envelope, not bitwise; `docs/research/adr063-q1-numeric-policy.md`).

Joint decode is on by default for dense (Qwen3.8) and MoE (Qwen3.6) artifacts. MoE artifacts whose routing policy enables `mul_mv_ext`, block storage outside Q4_0/Q8_0/Q4_K/Q5_K/Q6_K, float FFN storage, GLP steering, layer dumps, `HF2Q_LEGACY_PER_LAYER_CB=1` and `MLX_UNRETAINED_REFS=1` keep serial decode. The decision and its reason are logged once at worker start, with the model's routing policy. That policy is frozen at load and cannot change mid-run; `HF2Q_LEGACY_PER_LAYER_CB` and `MLX_UNRETAINED_REFS` are re-checked every turn. See [ADR-063 S3](research/adr063-s3-qwen-joint-decode.md).

| Var | Default | Values | Effect |
|---|---|---|---|
| `HF2Q_QWEN_JOINT_DECODE` | on (Qwen3.8 and Qwen3.6) | `0`/`off`/`false`, `1`/`on`/`true` | Escape hatch and A/B switch. `0` restores serial per-slot decode. Read once at worker start. Invalid values warn and use the default (on). |
| `HF2Q_QWEN_JOINT_DECODE_MAX_ROWS` | `4` | `2`–`8` | Widest joint step. In exact mode, or with `HF2Q_QWEN_JOINT_SOLO=0`, further eligible slots stay serial that turn. In the padded mode they run in further joint steps. The padded `mul_mv_ext` mode caps at 4 regardless. Widths above 4 have no serving measurement. Invalid values warn and use 4. |
| `HF2Q_QWEN_JOINT_FFN_SHARED` | on | `0`/`off`/`false`, `1`/`on`/`true` | Padded mode only. `0` restores the serial builder's per-row fused gate/up/SiLU kernel, which reads the gate and up weights once per packed row (four times per step). Read once at worker start. Invalid values warn and use on. |
| `HF2Q_QWEN_JOINT_SOLO` | on | `0`/`off`/`false`, `1`/`on`/`true` | Padded mode on workers with two or more slots only. On: every eligible slot joins a joint step, and a lone ordinary slot runs a one-lane padded step instead of the serial forward, so N4 output with speculation off no longer depends on request timing. `0` restores the serial forward for a lone slot and for slots beyond the joint width (A/B and escape hatch). Read once at worker start. Invalid values warn and use on. |
| `HF2Q_REPACKED_GGUF_M32` | off | `1` | Qwen3.8 dense DFlash2 target verification only. Exact value `1` prepares eligible Q4_K/Q6_K dense FFN gate/up/down planes once and dispatches the active 32-row joint transaction directly from BF16 activation rows. Ordinary prefill, serial decode, vision-language, MoE and other-family products retain their established kernels; ineligible shapes retain the stock GGUF route. The exact Qwen3.8 four-agent campaign peaked at 70.17 GB process footprint against 59.40 GB for the stock route, so admission must include the prepared buffers. Read once per process. |
| `HF2Q_REPACKED_GGUF_M32_PROJ` | off | `1` | Qwen DFlash2 target verification only, and only with `HF2Q_REPACKED_GGUF_M32=1`. Exact value `1` also prepares the eligible Q4_K/Q6_K DeltaNet (qkv, gate, output, and alpha with beta as one fused product) and full-attention (fused Q/gate, K, V, output) projections on first use and dispatches them in the active 32-row joint transaction from one BF16 copy of the shared activation. Output differs numerically from the stock projections. On the exact Qwen3.8 four-agent campaign the process footprint peaked at 69.95 GiB against 65.34 GiB without it. Read once per process. |
| `HF2Q_REPACKED_GGUF_M32_HEAD` | off | `1` | Qwen DFlash2 target verification only, and only with `HF2Q_REPACKED_GGUF_M32=1`. Exact value `1` prepares an eligible Q4_K/Q6_K output head on first use and runs the 32-row joint step's logits through it. Logits differ numerically from the stock head, so the greedy token can differ at near-ties. It shortened the four-lane target stage by 3.3% on Qwen3.8 without a resolved change in client makespan, and added about 0.94 GiB of footprint. Read once per process. |
| `HF2Q_QWEN_MOE_BATCHED` | off | `0`/`off`/`false`, `1`/`on`/`true` | Qwen3.6 MoE artifacts. `1` loads the router (`ffn_gate_inp`, F32), shared-expert gate logit (F32) and shared expert (Q4_K or Q6_K) in their GGUF storage instead of BF16 copies, and every MoE decode layer then runs one batched route over the step's rows (F32 row-exact router mat-vec, SIMD-group top-k over all rows, one `mul_mv_id` each for expert gate/up/down over rows × top-k pairs, one weighted reduce). A row's output is bitwise the same solo (serial decode, MTP drafting) and in a joint step of 2–8 rows, but differs from the default BF16 route (Class B: the router and shared expert are no longer BF16-rounded). Prefill reads the same native weights. Layers with other storage types keep the default route; DWQ affine expert overlays refuse to load. Read once at model load; the worker-start `Qwen joint decode configuration` line logs `moe_ffn=batched_native`. Invalid values warn and use off. |
| `HF2Q_QWEN_MOE_STAGED_M8` | off | `0`/`off`/`false`, `1`/`on`/`true` | Qwen3.6 Q4_K/Q6_K MoE artifacts. Requires `HF2Q_QWEN_MOE_BATCHED=1`. `1` prepares quantized expert gate/up/down planes at model upload; separate grouped staged-M8 F32 gate and up projections reuse one compact route schedule and gathered input, followed by the established F32 SwiGLU and an F32-input prepared down projection. Unsupported top-k, shapes, quantization, affine overlays or imatrix collection fail startup. This is a quality-qualified opt-in, not the release default; unset or `0` keeps the established native expert matvecs. |

## DeepSeek-V4 decode

With two or more agents decoding at once, DeepSeek-V4 advances every eligible slot through one joint forward per token step. Each row is bitwise identical to serial decode. Joint decode engages only for artifacts whose batched products all use proven-exact storage types: Q8_0, Q2_K or Q3_K; float weights only where they are split per row; grouped output-A only as Q8_0. Other artifacts stay serial, and the reason is logged at worker start. See [ADR-063 S3](research/adr063-s3-ds4-joint-decode.md).

| Var | Default | Values | Effect |
|---|---|---|---|
| `HF2Q_DEEPSEEK_JOINT_DECODE` | on | `0`/`off`/`false`, `1`/`on`/`true` | Escape hatch. `0` restores serial per-slot decode, including the exact B=4 cohort. Read once at worker start. It is also forced serial by `MLX_UNRETAINED_REFS`, DeepSeek layer dumps or stage profiling, and an explicit `HF2Q_DECODE_MV_EXT=1`. The routing policy is frozen into the DeepSeek-V4 model's registries at load, so loading a Qwen3.8 model in the same process does not affect it. |
| `HF2Q_DEEPSEEK_JOINT_DECODE_MAX_ROWS` | `4` | `2`–`8` | Widest joint step. Widths above 4 are covered only by the official-artifact bitwise test and have no serving measurement; use them only for experiments. Invalid values warn and use 4. |

## Dense KV / decode layout

`dense_kv_capacity` is sized per-layer at prefill time. Sliding layers
use a ring buffer capped at `sliding_window` (1024 on Gemma-4); global
layers use a linear buffer of `seq_len + max_decode_tokens`. No env var
controls this — it's a correctness property, not a tunable.

## LCP resume budget (not an environment variable)

The in-memory LCP resume registry budget is the typed
`hf2q serve --lcp-resume-budget <SIZE>` flag or `[serve] lcp_resume_budget`
config key (CLI wins over config). Omitted, it is a constant 8GiB on every
host. Serve resolves it once at startup, logs
`[hf2q lcp] byte_budget=<bytes> (source=default|config|cli)`, and passes it to
every Qwen and Gemma engine load, including model swaps. Gemma's per-request
pre-copy snapshot check uses the same value. Free memory is never consulted,
so which prompts get snapshotted and which entries are evicted does not depend
on the host or on load order. Zero and malformed values fail startup. The
former `HF2Q_KV_LCP_RESUME_CAPACITY` variable (and its `available_memory() ×
5%` default) is removed and no longer read (#730); replace `=8g` with
`--lcp-resume-budget 8GiB`, and a legacy entry count `N` with `N × 300MiB`
(for example `=8` → `--lcp-resume-budget 2400MiB`).

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
  single-slot route. Each request's tokens, logprobs and cache bytes are
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

### Gemma joint cold prefill (default on)

`HF2Q_GEMMA_JOINT_COLD_PREFILL` controls ADR-063 Q4-GEMMA-COLD joint cold
prefill on the Gemma 4 SlotAware worker. It is read once when the worker
starts.
- **Unset, `1`, `on` or `true` (default):** when every slot is idle, two or
  more cold text requests in one admission quantum, each with an exact
  pre-generation boundary of at least 33 tokens (the four-agent first turn),
  run as one transaction: one joint prefill of every boundary, each lane's
  boundary checkpoint in arrival order, then one joint replay of the short
  generation cues. Attention runs per request with the single-slot route.
  Each request's tokens, logprobs, checkpoint and cache bytes are bitwise
  equal to the per-slot admission, and no request waits for the others'
  serial prefills. Groups with a cached lane, a vision input or more than
  4,096 rows keep the per-slot route.
- **`0`, `off` or `false`:** kill switch; every cold request is admitted per
  slot.
- **Any other value:** logs a warning and keeps the default.

It is also off when cross-slot admission or hybrid KV is off, or under
`HF2Q_BATCHED_BODY=0`, `HF2Q_PREFILL_SLOT_BATCHED=0`,
`HF2Q_MM_ID_ROUTING_THRESHOLD`, `HF2Q_DFLASH_XLEN_SDPA=1`, `HF2Q_FA_F16=0`,
`HF2Q_NO_FA` or `HF2Q_GLOBAL_FA`, because those change or bypass the per-slot
route the cohort must equal. The worker logs `Gemma joint cold prefill worker
decision` at start, and each cohort logs `Gemma4 cold stable cohort prefilled
in one joint transaction` with its phase timings. Evidence:
`docs/research/adr063-q4-gemma-cold-design.md`.

### Gemma tiled warm attention (default off)

`HF2Q_GEMMA_WARM_TILED_ATTENTION` selects the attention kernel of Gemma 4
warm prefills (ADR-063 Stage 2): every single-slot prefill that resumes a
cached prefix, and every lane of a joint warm cohort. It is read once when
the model loads.
- **Unset, `0`, `off` or `false` (default):** warm prefills use the
  vec-hybrid kernel (one decode-style pass per query over the whole history).
- **`1`, `on` or `true`:** opt in to tiled F16 flash attention over the
  slot's K and dequantized V range, one sequence per dispatch (1,024-query
  sliding chunks, 4,096-query global chunks). Suffixes shorter than 32 tokens
  keep vec-hybrid attention. A joint cohort stays bitwise equal to its
  per-slot prefills, but the route changes warm-prefill decisions against
  the default: on the canonical four-agent task it moved a follow-up turn's
  top-2 log-probabilities from −0.149/−1.978 to −0.368/−1.179 and changed the
  written code (`$E/gemma-divergence-v1`). Default off since the ADR-063
  quality gate (2026-11-01).
- **Any other value:** logs a warning and keeps the default.

The switch also covers installed long-prompt lanes (prompts over 4,096
tokens), which run the single-slot warm route (#725). The model load logs `Gemma warm live
attention kernel`, each live prefill banner prints `live=`, and
`/hf2q/v1/runtime` publishes the resolved kernel (`VecHybrid` or `TiledF16`)
as `measurement_snapshot.active_controls.gemma_warm_attention_kernel`.
Evidence: `docs/research/adr063-gemma-four-agent-reuse.md`. A precise F32
tiled route was evaluated and falsified (ADR-063 Q4-GEMMA-ATTN,
`docs/research/adr063-gemma-tiled-attn-design.md` §12).

### Gemma decode dense kernel (default `stock`)

`HF2Q_GEMMA_DECODE_DENSE` selects the kernel of the dense GGUF matmuls of
Gemma 4 decode (attention q/k/v/o, dense FFN gate/up/down and the lm head) in
the solo decode, the `[N,hidden]` batched body, the batched and fused lm head,
the speculative verify heads and tree-verify layers, and the prefill
first-token head, so every logit row comes from the same per-row kernel (ADR-063
Q4-GEMMA-DECODE op parity). It is read once per process, at model load.
- **Unset or `stock` (default):** the existing routes.
- **`row_exact`:** the row-exact few-row mat-vec (`row_exact_matvec`) for
  Q4_K or Q6_K weights with `K % 256 == 0` (Q4_K also needs a 16-byte-aligned
  weight start) and Q5_0 or Q8_0 weights with `K % 32 == 0`.
- **`mma`:** the column-invariant few-row MMA mat-vec (the peer's
  `mul_mv_mma` technique) for Q4_K, Q6_K or Q5_0 weights with `K % 64 == 0`
  (and the type's block size) and Q8_0 weights with `K % 32 == 0`. Numerics
  class B against the default routes.
- **Any other value:** the model load fails.

Under `row_exact` and `mma` each row's result is bitwise the same however many
rows share the dispatch, so a slot's logits do not depend on the decode width
or path; products wider than 8 rows run in chunks of at most 8 rows. Whether a
weight takes the kernel depends on the weight alone, never on the width; every
other weight (F32 router, F16/BF16, other types) and the other prefill
matmuls keep their routes. `/hf2q/v1/runtime` publishes the resolved kernel (`Stock`,
`RowExact` or `ColumnInvariantMma`) as
`measurement_snapshot.active_controls.gemma_decode_dense_kernel`; Gemma Q1
endpoint and localizer packets declare it as `decode_dense_kernel`.
The Gemma launcher forwards `DECODE_DENSE=stock|row_exact|mma`; the Q5 case
maps `HF2Q_Q5_GEMMA_DECODE_DENSE` to that launcher variable. Leaving it unset
keeps the stock route.

### Native Q5_0 tensor matmul (default off)

`HF2Q_Q5_0_TENSOR_MM` is read by mlx-native (0.18.0 and later), not by hf2q.
Only the exact value `1` routes Q5_0 dense and expert matrix multiplications
through the tensor pipeline; unset or any other value keeps the simdgroup
route. It exists for the Gemma combined-candidate quality campaign; the
tensor route is not yet qualified as a default. The first Gemma 4 model load
of a process logs one line `Gemma4 native routing: q5_0_tensor_mm=<true|false>
mm_id_schedule=<grid|compact>`, and
`/hf2q/v1/runtime` publishes the resolved value as
`measurement_snapshot.active_controls.q5_0_tensor_mm` (`true`/`false`) beside
`mm_id_schedule`. The Gemma launcher forwards it from `Q5_0_TENSOR_MM=1`
and the Q5 case from `HF2Q_Q5_Q5_0_TENSOR_MM=1`.
