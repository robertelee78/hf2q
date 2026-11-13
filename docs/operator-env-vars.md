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

DFlash2 is the default route of every Qwen target that a pinned drafter
serves (ADR-067 D6). The server decides once per model load, from the GGUF
header, before any weight is read. A default `hf2q serve` needs no variable
and no manual download.

**Policy (`HF2Q_DFLASH2`).** Unset is *auto*: on a SlotAware worker of one to
four slots, DFlash2 is selected for a target in either of two classes.

- *Pinned pair*: the target was produced by hf2q and its
  `hf2q.source_sha256` equals the `target_source_sha256` of a pinned Qwen
  descriptor of the same architecture.
- *Same architecture and shapes*: the target has the architecture string,
  hidden size, layer count, vocabulary size, tokenizer identity (SHA-256 over
  the GGUF tokenizer pre-type, tokens and merges), attention and DeltaNet
  head counts and dimensions, FFN width, MoE expert count, routed experts per
  token and expert widths, and embedding and output-head tensor shapes of a
  pinned descriptor's target. This admits fine-tunes and other quantizations
  of the same base, with or without hf2q provenance (operator decision
  2026-11-12). The drafter only proposes tokens; the target decides every
  emitted token, so a drafter that fits a fine-tune less well costs
  acceptance, not correctness. There is no acceptance-based drop-back.

One static boundary narrows auto for MoE targets: auto qualifies a
`qwen35moe` target only when its batched MoE verification route
(`HF2Q_QWEN_MOE_BATCHED`) is admitted for the artifact's storage. Without it
every row of a 32-row verify step runs the per-row expert path, which is not
a qualified fast path (measured on the Q5_K launcher default, below). This is
decided from the GGUF header before the load, like everything else here: the
server does not watch acceptance, output or timing, and never changes route
after the load.

Anything else (other shapes, another family, more slots, another scheduler)
keeps the prior route and the audit line says why. Auto never fails startup:
when the drafter cannot be obtained or does not bind to the target, or the
load fails with DFlash2 selected, the server logs the reason and loads the
prior route. `0` is the kill switch: no drafter, no profile, the route the
server ran before DFlash2 was a default. `1` requires DFlash2: a target in
neither class, or a drafter that cannot be obtained, fails startup
(schedulers and families DFlash2 never served keep ignoring it, with a
warning). `1` also serves the drafter on a MoE target without its batched
route, with a WARN line that the route is unqualified and was measured
slower.

**Drafter.** In order: `HF2Q_DFLASH2_CONFIG` and `HF2Q_DFLASH2_WEIGHTS` when
both name a file (under auto a stale or incomplete pair is reported at WARN
and ignored, under `HF2Q_DFLASH2=1` it fails startup, under `HF2Q_DFLASH2=0`
it is not read; the launchers check the pair only when DFlash2 is required); the Hugging Face cache snapshot of the descriptor's repository
at its pinned revision (`HF_HUB_CACHE`, `HF_HOME/hub`,
`XDG_CACHE_HOME/huggingface/hub`, `~/.cache/huggingface/hub`); a download of
`config.json` and `model.safetensors` at that revision into that cache, with
progress logged, before the target's weights are loaded. `HF_HUB_OFFLINE=1`
skips the download. A failed download under auto logs one WARN line naming
the repository, revision and reason. The checkpoint identity checks are
exactly as strict for either class: the configuration's SHA-256 and schema,
the weights' byte length and SHA-256 and the tensor manifest must match the
descriptor, and unknown revisions and family adapters are rejected. The
checkpoint is bound to the target by the target adapter's own shape check,
on the GGUF header before the load and on the loaded model at provisioning;
a pinned pair is additionally bound by its source identity.

| Descriptor | Repository at revision | Weights |
|---|---|---:|
| `qwen38-27b-dflash2-50307d4c` | `z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | 3,848,817,896 bytes |
| `qwen36-35b-a3b-dflash2-51ef7b69` | `incoai/Qwen3.6-35B-A3B-DFlash2` at `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` | 1,052,513,456 bytes |

**Route profile.** Each pinned Qwen descriptor has one profile: the value of
every fast-route switch in its measured combined configuration (one table,
`src/inference/spec_decode/dflash2/route_profile.rs`). Each switch is
resolved once per model load: a non-blank environment value wins, in either
direction; otherwise the profile's value when DFlash2 is selected for that
model and the row is not withheld (below); otherwise the switch's own
default. A model for which DFlash2 is not selected therefore runs exactly the
kernels it ran before, also when it shares a process with one for which it
is, and `HF2Q_DFLASH2=0` turns the whole bundle off.

| Switch | Qwen3.8 profile | Qwen3.6 profile | Requires | Prepared planes |
|---|---|---|---|---|
| `HF2Q_DFLASH2_DRAFT_AF4_G64` | `1` | `1` | | none |
| `HF2Q_DFLASH2_DRAFT_STAGED` | `1` | `1` | `HF2Q_DFLASH2_DRAFT_AF4_G64` | output head |
| `HF2Q_DFLASH2_GPU_TARGET_TOKENS` | unset (descriptor default: on) | `1` | | none |
| `HF2Q_DFLASH2_CONTEXT_FIRST` | `1` | `1` | | none |
| `HF2Q_DFLASH2_DRAFT_PREFETCH` | unset | unset | | none |
| `HF2Q_DFLASH2_GPU_DECISIONS` | unset | unset | | none |
| `HF2Q_REPACKED_GGUF_M32` | `1` | `1` | | dense FFN |
| `HF2Q_REPACKED_GGUF_M32_PROJ` | `1` | `1` | `HF2Q_REPACKED_GGUF_M32` | DeltaNet and attention projections |
| `HF2Q_REPACKED_GGUF_M32_HEAD` | `1` | `1` | `HF2Q_REPACKED_GGUF_M32` | output head |
| `HF2Q_REPACKED_GGUF_ANY_WIDTH` | `1` | `1` | `HF2Q_REPACKED_GGUF_M32` | none |
| `HF2Q_REPACKED_GGUF_SHARED_HEAD` | `1` | `1` | | none |
| `HF2Q_QWEN_VERIFY_ATTN_TILE` | `1` | `1` | | none |
| `HF2Q_QWEN_VERIFY_MIXER_FUSED` | `1` | `1` | | none |
| `HF2Q_QWEN_VERIFY_REPLAY_REUSE` | `1` | `1` | | none |
| `HF2Q_QWEN_PREFILL_PREPARED` | `1` | `1` | | dense FFN, projections |
| `HF2Q_QWEN_PREFILL_FOLD_CUE` | `1` | `1` | `HF2Q_QWEN_PREFILL_PREPARED` | none |
| `HF2Q_QWEN_MOE_BATCHED` | unset | `1` | | none |
| `HF2Q_QWEN_MOE_STAGED_M8` | unset | `1` | `HF2Q_QWEN_MOE_BATCHED` | routed expert stacks |
| `HF2Q_QWEN_MOE_LIVE_TILES` | unset | `1` | `HF2Q_QWEN_MOE_STAGED_M8` | shared expert |

`HF2Q_DFLASH2_DRAFT_PREFETCH` and `HF2Q_DFLASH2_GPU_DECISIONS` are rows without a
profile value: they are resolved per model like every other row and stay off
until the combined configuration with them is qualified.

The profile is applied switch by switch. A row's profile value is *withheld*
(the switch keeps its own default, and the drafter serves with the rows that
are left) when:

- the switch it requires did not resolve on, whether withheld or set to `0`
  by hand;
- its kernels would refuse the artifact: the batched MoE route on a DWQ
  overlay, a router or shared-expert gate logit that is not F32, or a shared
  expert that is not Q4_K/Q6_K with gate and up stored alike; the staged MoE
  route on routed expert stacks that are not 64-aligned Q4_K/Q6_K; the
  prepared prefill beside the chunk-scan investigation route;
- the weights its planes cover exist and none is Q4_K or Q6_K (a weight kind
  stored partly in those types keeps its row: the prepared kernels admit
  weight by weight and the rest keep their stock kernels);
- this host cannot run its kernels (for example a Metal runtime without the
  tensor API the prepared kernels are written over);
- the memory rule below dropped it.

For example a dense target quantized to Q5_K runs the drafter with the
AF4G64 and staged draft, the tiled verify attention, the fused mixer and the
replay reuse, and keeps its stock FFN and projection kernels. Nothing here
fails startup, under auto or `HF2Q_DFLASH2=1`. A switch
set by hand is never withheld: on storage it does not admit it still refuses
the load as it always has, and on a host that cannot run its kernels the
server refuses to start, naming the switch, before the model is loaded.
"Set on" is what the switch's own reader takes as on, here and everywhere a
rule is settled before the reader runs (the planes a route keeps, the row
another row requires): exactly `1` for the `HF2Q_DFLASH2_*` and
`HF2Q_REPACKED_GGUF_*` rows and the two `HF2Q_QWEN_VERIFY_*` rows that take
no other value, and `1`, `on` or `true` for `HF2Q_DFLASH2_GPU_TARGET_TOKENS`,
`HF2Q_QWEN_PREFILL_*` and `HF2Q_QWEN_MOE_*` (the grammars of
`src/inference/route_host.rs`). `HF2Q_REPACKED_GGUF_M32=on` therefore selects
no kernel, prepares no plane and refuses no load.

A MoE target qualifies under auto only when its batched MoE verification
route is admitted, and admitted means what the loader will do with the
artifact's storage, not what the switch says: the loader keeps a layer on the
per-row route when its storage is not the batched route's, whoever turned
`HF2Q_QWEN_MOE_BATCHED` on. Setting the switch by hand on such an artifact
does not select the drafter under auto. `HF2Q_DFLASH2=1` is the override: it
serves the drafter there with a WARN that the route is unqualified, and the
staged expert rows stay off because they need the batched layer.

**Memory rule.** Prepared planes are a second copy of the weights they cover,
and the drafter is resident beside the target. Before selecting, the server
computes

```
present  = memory available now - charged engines - model file bytes - KV budget - 8 GiB
budget   = physical memory      - charged engines - model file bytes - KV budget - 8 GiB
headroom = the smaller of present and budget
planes   = sum over the covered Q4_K/Q6_K weights of the plane sets of every
           switch that resolved on:
           Q4_K: 16 bytes per 32 weights + 16 bytes per 256 weights
           Q6_K: 24 bytes per 32 weights + 20 bytes per 256 weights
scratch  = what the kernels of those switches retain between steps
```

where the KV budget is the configured `--kv-cache-budget` (zero when none is
configured), the drafter is its file size, *charged engines* is what every
engine of this process has committed (its model file, KV budget, drafter,
planes and scratch, allocated or not), and *memory available now* is what
the host reports free, inactive or purgeable at the decision.

What is available now is still owed to everything that has been promised
memory and has not taken it: the incoming model's file and KV budget, and
whatever the engines already loaded have not allocated of their commitments
(a KV cache not filled yet, planes prepared on first use). The server knows
what each engine was promised, not how much of it the engine has allocated,
so it subtracts the whole of every commitment; the part an engine has
already allocated is then counted twice, which errs on the safe side. What
is available also leaves out, by itself, everything anybody holds now:
memory of an engine that was just torn down and that the kernel has not
reclaimed yet, and every other program. On an idle host the two bounds are
equal; on a host where other programs hold memory the present bound is the
smaller and the default may keep fewer routes. With a 48 GiB KV budget the
whole profile needs about 90 GiB available for the Qwen3.8 gate artifact
and about 97 GiB for the Qwen3.6 one. The audit line gives both bounds and
what the host had available. Another process that grows after the decision
is covered by neither; the 8 GiB reserve is the only margin for it.

The decision and the reservation of what it decided are one transaction, so
two loads running at once, through one pool or through separate ones, are
never both decided against the same total. A reservation belongs to whoever
owns the model's buffers: the loading thread during the load, then the
engine's worker thread. It is released by the worker's teardown and by
nothing else, in this order: the worker drops the model; it drops the GPU
weight cache it built for it (uploaded copies and prepared planes); the
thread terminates, which destroys every other thread-local cache it had;
every command queue the model's threads encoded on is settled (an empty
command buffer is committed and waited for, which commits the removals the
dropped buffers staged in the Metal residency set and waits for the commands
committed earlier on that queue); then the reservation goes. A request's own
queue is settled under the engine's supervision as the last GPU step of the
request, before its result or `Done` is reported: a settle that fails is
that request's error, and one that never returns fails the engine like any
other GPU call that does. Evicting an engine or dropping its
last handle releases nothing: an engine whose worker is still tearing down,
or is wedged inside Metal, is still charged, and a teardown step that fails
leaves it charged until the process restarts (one ERROR line, `engine memory
stays reserved until the process restarts`).

New loads are serialized against teardowns: a load that arrives while an
engine is being torn down waits for the teardown to end before it is
decided (at most 30 s; after that it is decided with the engine still
charged). The kernel returns the pages of released buffers on its own, some
time after the process has let go of them (measured on the Qwen3.6 gate
artifact: 15 to 28 GiB of a 40 GiB engine were still charged to the process
after the last step above and were gone 0.1 to 1 s later). The teardown
gives it up to 10 s, watching the process footprint, so that a waiting load
reads the available memory afterwards; that is a courtesy and no claim
rests on it. What keeps a load from being given memory that is not back is
the present bound.

A load that fails after its worker started stops the worker and waits for
that teardown before the load is retried on the prior route, and the retry
reserves its own commitment. The loading thread drops the GPU cache it
primed and settles its queues before a reservation it still holds is
released.

The planes of the model's own prepared routes are resident whatever the
policy says: `HF2Q_DFLASH2=0 HF2Q_QWEN_PREFILL_PREPARED=1` prepares and keeps
the dense FFN and projection planes, and that engine is committed, published
and charged for them like any other. Only the staged draft's head needs a
drafter. The hot-swap pool charges a load the same figure without the KV
budget (model file, drafter, planes, scratch) instead of the GGUF file size,
before the loader is called. A plane set is prepared once however many switches read
it, except the output head: the staged draft (at four slots) and the
prepared target head each keep a copy of their own when
`HF2Q_REPACKED_GGUF_SHARED_HEAD=0` (1,052,876,800 bytes each on Qwen3.8).
No current route retains scratch; one that does gets its figure in
`route_scratch_bytes` (`src/serve/dflash2_policy.rs`), which sees the
resolved routes as a whole because routes may share a scratch pool.

If `drafter + planes + scratch <= headroom` every admitted row applies.
Otherwise every plane-carrying row is withheld and DFlash2 is kept when the
drafter and what is left fit; if not even the drafter fits, auto keeps the
prior route. The audit line reports the headroom, the present bound, the
available memory, the charged engines, the drafter, the plane and the
scratch bytes. A plane-carrying switch set by
hand is never dropped, and its planes count.

Measured, exact four-agent Q5 on the gate artifacts, peak process
footprint (the KV caches and decode scratch are in both columns):

| Target | Prior route | Default profile | Difference | Rule charges (drafter + planes) |
|---|---:|---:|---:|---:|
| Qwen3.8 gate artifact | 57.87 GB | 76.73 GB | 18.86 GB | 3.85 + 16.09 = 19.94 GB |
| Qwen3.6 gate artifact | 35.18 GB | 57.22 GB | 22.04 GB | 1.05 + 21.37 = 22.42 GB |

By component, Qwen3.6 gate artifact, physical footprint after load and
worker provisioning with no request served (GiB): weights as the prior route
holds them 21.6; with the drafter 22.3; with the batched MoE route 21.6 (the
router and shared expert stay in GGUF storage instead of BF16 copies); with
the staged expert planes 39.8 (one copy: 18.7 GiB by layout); with the live
tiles' shared-expert planes 39.9. The projection and head planes (0.75 and
0.39 GiB) are prepared on first use. On Qwen3.8 the planes are 9.83 GiB
(dense FFN), 4.17 GiB (projections) and 0.98 GiB (head), all prepared on
first use.

The prepared planes are a second representation of weights the GGUF blocks
already hold, and both stay resident: prefill outside the prepared scope,
MTP and every non-prepared path still read the blocks. A single copy of
every weight would cost about the prior route's footprint plus the drafter
(Qwen3.8 about 61.7 GB, Qwen3.6 about 36.2 GB at the Q5 peak); that requires
every consumer of a weight to read its planes and is not implemented. The
plane-dropped configuration of either family has no exact Q5 run of its own.


**Audit.** Each load logs one INFO line, `DFlash2 policy resolved`, with
`policy` (`auto`, `off`, `required`), `selected`, `reason` (when selected:
`pinned pair` or `same architecture and shapes as <descriptor id>`),
`descriptor`, `target_source` (the target's own `hf2q.source_sha256`, or
`external`), `drafter_source` (`explicit`, `cache`, `downloaded`, `none`),
`profile` (`full`, `partial`, `none`), `withheld` (`NAME: reason` for every
withheld row, or `none`), the memory figures, and `switches`: every table
switch as `NAME=value(source)` with source `env`, `profile`, `default` or
`withheld`. The same decision is published by `/hf2q/v1/runtime` as
`measurement_snapshot.active_controls.dflash2` (`policy`, `selected`,
`reason`, `descriptor`, `qualification` = `pinned_pair` | `same_shapes`,
`target_source`, `drafter_source`, `profile`, `switches.<NAME>` =
`{value, source, withheld}`, `memory`), so a gate receipt that records the
runtime document binds the route that ran, not the environment that asked.
It is `null` for an engine that was not loaded by `hf2q serve`. The memory
rule, the host probe and the artifact's storage can each leave a route of
the profile off without failing startup, so the default a gate measures is
only the qualified default when `profile` is `full`: the release-gate Q5
cases require it (`HF2Q_Q5_DFLASH2_PROFILE=full` is a readiness check on
that field, beside the policy, descriptor and qualification checks).

| Var | Default | Values | Effect |
|---|---|---|---|
| `HF2Q_DFLASH2` | auto | unset, `0`/`off`/`false`, `1`/`on`/`true` | The policy above. Unset selects DFlash2 for a pinned pair or a target with a pinned target's architecture and shapes, and never fails startup; `0` is the kill switch; `1` requires DFlash2. Any other value fails startup. |
| `HF2Q_DFLASH2_CONFIG` | unset | path | Explicit checkpoint configuration, used only together with `HF2Q_DFLASH2_WEIGHTS`; unset, the pinned snapshot in the Hugging Face cache is used or downloaded. Its SHA-256, schema and target binding must match a compiled descriptor. |
| `HF2Q_DFLASH2_WEIGHTS` | unset | path | Explicit safetensors checkpoint, used only together with `HF2Q_DFLASH2_CONFIG`. Its SHA-256, byte length and tensor manifest must match the same descriptor. |
| `HF2Q_DFLASH2_DRAFT_AF4_G64` | off; `1` in the route profile of a selected DFlash2 target | `1` | Uses source-bound group-64 affine four-bit draft projections (36 bytes per 64 weights) instead of BF16. Qwen3.8 and Qwen3.6 Q5 have passed; the route remains opt-in until every activated family is qualified. |
| `HF2Q_DFLASH2_DRAFT_STAGED` | off; `1` in the route profile of a selected DFlash2 target | `0`, `1` | Requires `HF2Q_DFLASH2_DRAFT_AF4_G64=1`. `1` runs the draft command on Apple10-class kernels: staged AF4G64 projections over BF16 rows with gate and up in one pass, attention over the K/V rings in place, and, at four lanes, the vocabulary head on the prepared target tile (one extra repacked copy of a Q4_K/Q6_K head: 0.98 GiB for Qwen3.8 and 0.39 GiB for Qwen3.6 by plane size). Draft proposals may differ from the default route within the kernels' documented numeric bounds; the target still decides every emitted token. On Qwen MoE targets left at their default `HF2Q_DFLASH2_GPU_TARGET_TOKENS` (unset), verify rows are chosen from the draft's confidence and the verify tile width follows them, so a different draft also changes the target's logit bits and can change the generated text; that mode is not yet qualified with this switch. A checkpoint whose shapes or row count the staged tile does not take, or any other value, fails load. |
| `HF2Q_DFLASH2_GPU_TARGET_TOKENS` | on for the pinned dense Qwen3.8 descriptor; `1` in the Qwen3.6 route profile; off otherwise | `0`/`off`/`false`, `1`/`on`/`true` | Keeps selected proposals on the GPU through target embedding, removing the draft-to-target host wait and token packing. `0` restores staged host handoff. Explicit `1` is admitted on both Qwen adapters; the Qwen3.6 route profile sets it, because the staged draft is not qualified in the waited MoE mode. Explicit `1` on a Gemma descriptor fails checkpoint load: its dynamic verification-row and target-state transaction is not yet qualified. A blank value is unset; other invalid values fail load. |
| `HF2Q_DFLASH2_CONTEXT_FIRST` | off; `1` in the route profile of a selected DFlash2 target | `1` | Qwen DFlash2 steps that settle acceptance on the host. Exact value `1` submits the draft-context commit before the DeltaNet replay is encoded, so the GPU works on the context commit while the host encodes the replay and the two commands overlap; the step still waits for both. Output and cache state are bitwise unchanged. A replay command that fails after the submission clears the draft rings of the step's lanes (they re-fill on the next prefill); if they cannot be cleared the worker stops. Logged at worker start as `dflash2_context_first`. |
| `HF2Q_DFLASH2_DRAFT_PREFETCH` | off (a route-profile row with no profile value yet) | `1` | Qwen DFlash2 steps that settle acceptance on the host. Exact value `1` commits the next step's draft command right behind the draft-context commit and the DeltaNet replay, before the step waits for either, so the GPU drafts while the host waits, settles and emits tokens. The next step uses that draft only when its inputs are exactly the prefetched ones; otherwise the command is waited for and dropped and the step drafts again. Output, rings and cache state are bitwise unchanged. Supersedes `HF2Q_DFLASH2_CONTEXT_FIRST` on the steps it applies to. A step that fails after its submissions drops the draft and clears its lanes' draft rings (they re-fill on the next prefill). Logged at worker start as `dflash2_draft_prefetch`. |
| `HF2Q_DFLASH2_GPU_DECISIONS` | off (a route-profile row with no profile value yet) | `1`, `check` | Qwen DFlash2 lanes that sample on the host at temperature zero (repetition penalty, thinking budget, lazy tool grammar; no logit bias, top-k, top-p, min-p, seed or logprobs). `1` decides each verify row from candidates the output-head command computes (the repetition-window logits and the best logit outside the window) instead of copying and scanning 32 rows of logits on the host; rows under an active grammar and rows with a non-finite logit use the host sampler over their own logits. Emitted text is unchanged. `check` runs both routes, uses the host sampler's result and logs any disagreement at error level (diagnostic; slower than off). Logged at worker start as `dflash2_decisions`. |
| `HF2Q_QWEN_VERIFY_REPLAY_REUSE` | off; `1` in the route profile of a selected DFlash2 target | `1` | Qwen verify steps (DFlash2 and joint MTP). Exact value `1` keeps the DeltaNet replay storage between verify steps instead of allocating and zero-filling it for every step (188 MB per four-lane step on Qwen3.8-27B, 91 MB on Qwen3.6-35B-A3B), which otherwise leaves the GPU idle between the draft command and the target's first command. Output and cache state are bitwise unchanged. The worker holds the largest storage a step has used for its lifetime; that memory is not counted by memory admission or any budget, so leave headroom for it. Logged at worker start as `verify_replay_reuse`. |

### Host capability of the fast routes

Some fast-route kernels exist only on the Metal tensor API and have no
portable twin. That API is absent when the runtime Metal compiler has no
`<metal_tensor>` header (macOS before 26) or when the GPU is outside the
Metal 4 family (virtual machines, including GitHub's hosted macOS runners).
`src/inference/route_host.rs` asks mlx-native once per process which kernel
families the GPU runs and holds the table of what each switch needs:

| Switch | On a host without the Metal tensor API |
|---|---|
| `HF2Q_DFLASH2_DRAFT_STAGED` | cannot run (staged tiles, ring attention, prepared head) |
| `HF2Q_REPACKED_GGUF_M32`, `_M32_PROJ`, `_M32_HEAD`, `_ANY_WIDTH`, `_SHARED_HEAD` | cannot run (prepared GGUF matmul) |
| `HF2Q_QWEN_PREFILL_PREPARED` | cannot run (prepared GGUF matmul) |
| `HF2Q_QWEN_MOE_STAGED_M8` | cannot run (prepared expert planes) |
| `HF2Q_QWEN_MOE_LIVE_TILES` | cannot run (prepared expert planes, live tiles) |
| `HF2Q_QWEN_VERIFY_ATTN_TILE` | runs: every layer keeps the per-row chain (logged once) |
| `HF2Q_QWEN_PREFILL_FOLD_CUE` | runs: two forwards, as without a routed prepared prefill |
| `HF2Q_QWEN_VERIFY_MIXER_FUSED` | runs: the fused core is portable; its wave follows the prepared routes |
| `HF2Q_DFLASH2_DRAFT_AF4_G64`, `HF2Q_DFLASH2_GPU_TARGET_TOKENS`, `HF2Q_DFLASH2_GPU_DECISIONS`, `HF2Q_DFLASH2_CONTEXT_FIRST`, `HF2Q_DFLASH2_DRAFT_PREFETCH`, `HF2Q_QWEN_VERIFY_REPLAY_REUSE`, `HF2Q_QWEN_MOE_BATCHED` | run: portable kernels or host-side only |

A switch from the "cannot run" rows that the operator sets fails the model
load, before any weight is uploaded, with the switch, the kernel family and
the device. "Sets" follows each switch's own reader: exactly `1` for the
`HF2Q_REPACKED_GGUF_*` switches and `HF2Q_DFLASH2_DRAFT_STAGED`, and
`1`/`on`/`true` for `HF2Q_QWEN_PREFILL_PREPARED`, `HF2Q_QWEN_MOE_STAGED_M8`
and `HF2Q_QWEN_MOE_LIVE_TILES`; a value the reader treats as off is not
refused.

```text
HF2Q_REPACKED_GGUF_M32 is on, but the prepared GGUF matmul (repacked_gguf_mm) cannot run on this host: the Metal tensor API is unavailable (a runtime compiler without <metal_tensor>, or a GPU outside the Metal 4 family) (Apple Paravirtual device). Unset HF2Q_REPACKED_GGUF_M32: the stock route serves this host.
```

A load that selects none of those switches never fails on this check. If the
capability probe itself fails for a reason other than a missing tensor API (a
shader or toolchain regression on a Metal 4 device), a load that selects one
of the switches fails and names it; a load that selects none logs the failure
once and serves on the stock route.

What happens when such a switch is on without the operator having set it (a
route profile) depends on the route:

- The prepared GGUF matmul routes (`HF2Q_REPACKED_GGUF_*`,
  `HF2Q_QWEN_PREFILL_PREPARED`) ask the host at every admission, so on a
  host that cannot run them each product keeps its stock kernel.
- `HF2Q_QWEN_MOE_STAGED_M8`, `HF2Q_QWEN_MOE_LIVE_TILES` and
  `HF2Q_DFLASH2_DRAFT_STAGED` have no per-product admission: selected on
  such a host they fail the load with the typed "unavailable on this host"
  error. Whatever selects them asks `route_host` first
  (`HostRoutes::switch`) and leaves them off with the reason.

`MLX_NATIVE_DISABLE_METAL_TENSOR=1` makes the same kernel families
unavailable on any host, and the reason then names the variable instead of
the tensor API. It is not a faithful host without the API: mlx-native's
quantized and expert matmuls, the Gemma tensor flash attention and the tiled
verify attention keep their tensor kernels under it.

Outside the switches, two stock kernels had the same dependency and now take
a portable route on such a host. Both are Class B against the tensor kernel
they replace, not bitwise:

- The head-major O-projection of Gemma's batched prefill (Q4_0, Q5_0, Q8_0
  and Q6_K weights) permutes the activation and runs the ordinary matmul,
  the route Q4_K O-projections take on every host. At nine or more rows the
  two routes agree to about 2e-7 relative RMS; at eight or fewer rows the
  ordinary matmul is the matvec family, so a row's bits differ between an
  eight-row and a nine-row lane, as they already do for Q4_K.
- F16 weights at two or more rows, and an F16 DFlash2 drafter head at any
  row count, run mlx-native's portable F16 matmul (within
  `2^-11 * sum(|w * x|)` of the F64 product, deterministic, row-independent;
  about 4.5e-4 relative RMS from the tensor kernel).

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
| `HF2Q_REPACKED_GGUF_M32` | off; `1` in the route profile of a selected DFlash2 target | `1` | Qwen3.8 dense DFlash2 target verification only. Exact value `1` prepares eligible Q4_K/Q6_K dense FFN gate/up/down planes once and dispatches the active 32-row joint transaction directly from BF16 activation rows. Ordinary prefill, serial decode, vision-language, MoE and other-family products retain their established kernels; ineligible shapes retain the stock GGUF route. The exact Qwen3.8 four-agent campaign peaked at 70.17 GB process footprint against 59.40 GB for the stock route, so admission must include the prepared buffers. Resolved once per model load. |
| `HF2Q_REPACKED_GGUF_M32_PROJ` | off; `1` in the route profile of a selected DFlash2 target | `1` | Qwen DFlash2 target verification only, and only with `HF2Q_REPACKED_GGUF_M32=1`. Exact value `1` also prepares the eligible Q4_K/Q6_K DeltaNet (qkv, gate, output, and alpha with beta as one fused product) and full-attention (fused Q/gate, K, V, output) projections on first use and dispatches them in the active 32-row joint transaction from one BF16 copy of the shared activation. Output differs numerically from the stock projections. On the exact Qwen3.8 four-agent campaign the process footprint peaked at 69.95 GiB against 65.34 GiB without it. Resolved once per model load. |
| `HF2Q_REPACKED_GGUF_M32_HEAD` | off; `1` in the route profile of a selected DFlash2 target | `1` | Qwen DFlash2 target verification only, and only with `HF2Q_REPACKED_GGUF_M32=1`. Exact value `1` prepares an eligible Q4_K/Q6_K output head on first use and runs the 32-row joint step's logits through it. Logits differ numerically from the stock head, so the greedy token can differ at near-ties. It shortened the four-lane target stage by 3.3% on Qwen3.8 without a resolved change in client makespan, and added about 0.94 GiB of footprint. Resolved once per model load. |
| `HF2Q_REPACKED_GGUF_ANY_WIDTH` | off; `1` in the route profile of a selected DFlash2 target | `1` | Qwen DFlash2 target verification only, and only with `HF2Q_REPACKED_GGUF_M32=1` (and `_PROJ` / `_HEAD` for the projections and the head). Exact value `1` admits the prepared dense FFN, DeltaNet and full-attention projections and output head at every tile width of a verify step instead of only at 32 rows, so a Qwen3.6 verify tile of 4, 8 or 16 rows no longer falls back to the stock kernels and a verified row's projection bits no longer depend on the tile width or the rows sharing its step. Plain joint decode and joint MTP verify steps keep their routes, so an exact-mode joint row stays bitwise its serial one-token forward. Output differs numerically from the stock kernels at tile widths below 32 (Class B): a token verified by DFlash2 and the same token decoded serially or in a plain joint step come from different routes at every tile width, as they already do at 32 rows, so a lane that drops out of DFlash2 changes route. On the exact Qwen3.6 four-agent run with every other candidate switch and the prepared head on (same binary): one-lane cycle 17.1 / 17.4 / 18.6 ms against 22.6 / 22.1 ms, two-lane 22.0 / 23.0 / 23.0 ms against 36.8 / 34.5 ms, three- and four-lane cycles unchanged within run noise; a registry build of the final change read 17.8-18.8 and 23.2-24.3 ms in four runs. All runs passed. Of eight runs with the switch, six gave one transcript and two another that differs at one token of one agent's last turn; without it every pair of runs differed in every agent. Dense Qwen3.8 DFlash2 always verifies at 32 rows and is unaffected. Needs mlx-native 0.21.8. Resolved once per model load. |
| `HF2Q_REPACKED_GGUF_SHARED_HEAD` | off; `1` in the route profile of a selected DFlash2 target | `1` | Qwen targets with a Q4_K/Q6_K native output head. Exact value `1` prepares the head once, owned by the model, when the first consumer that reads it asks, and gives the same planes to a staged DFlash2 draft command of 32 draft rows (`HF2Q_DFLASH2_DRAFT_STAGED=1`, four lanes) and to the prepared target head (`HF2Q_REPACKED_GGUF_M32_HEAD=1`) instead of one prepared copy each (0.98 GiB on Qwen3.8, 0.39 GiB on Qwen3.6 by plane size). With neither consumer nothing is prepared. Logits are bitwise unchanged. On the exact Qwen3.8 four-agent run the prepared target head with the shared planes peaked at 75.80 / 75.69 GB process footprint against 75.81 / 75.79 GB with the stock target head (76.75 GB with the prepared target head and no sharing), and shortened the four-lane cycle to 71.0 / 70.7 ms from 72.5 / 72.4 ms. Resolved once per model load. |
| `HF2Q_QWEN_PREFILL_PREPARED` | off; `1` in the route profile of a selected DFlash2 target | `0`/`off`/`false`, `1`/`on`/`true` | Qwen bounded prompt prefill (SlotAware). `1` runs the eligible Q4_K/Q6_K dense FFN (gate, up, down), DeltaNet (qkv, gate, output, and alpha with beta as one fused product) and full-attention (fused Q/gate, K, V, output) products of every prompt-prefill chunk of two or more rows through the prepared row-tile kernel: one BF16 copy of the activation, 32-row tiles with a partial last tile, where the stock matmul pays a whole 128-row tile for every started one. It uses the planes of `HF2Q_REPACKED_GGUF_M32` / `HF2Q_REPACKED_GGUF_M32_PROJ` and prepares them itself if those are off (their footprint then applies; not measured alone). MTP catch-up, verify windows and decode keep their kernels. Output differs numerically from the stock prefill (Class B), so the greedy text can differ. On the exact Qwen3.8 four-agent workload with both `M32` switches on (two runs each) total prefill time went from 18.0–19.3 s to 13.6–14.3 s and client makespan from 54.2–55.4 s to 50.1–51.4 s, with a different greedy transcript (5,361 against 5,237 completion tokens) and unchanged peak footprint; a 31-row chunk went from 0.23 s to 0.07 s. Requires mlx-native 0.21.8: with 0.21.7 the prepared kernel accepts 32 rows only and a prefill under this switch fails. Resolved once per model load; invalid values warn and keep the default. |
| `HF2Q_QWEN_PREFILL_FOLD_CUE` | off; `1` in the route profile of a selected DFlash2 target | `0`/`off`/`false`, `1`/`on`/`true` | Qwen bounded prompt prefill (SlotAware); requires `HF2Q_QWEN_PREFILL_PREPARED=1` (the server refuses to start otherwise, as on an invalid value). `1` runs the last content chunk and the generation cue behind it as one forward that captures the stable-boundary checkpoint inside it, instead of a content forward, a checkpoint snapshot and a cue forward. The checkpoint, the logits, the DFlash2 capture and the slot state are bit-identical to the two-forward path, and the Qwen3.8 four-agent transcript is unchanged. A request keeps its two forwards, without an error, when any of these holds: a vision or grafted request; MTP catch-up; a content chunk of fewer than 2 rows or a cue outside 2 to 32 rows; a content chunk of at most 32 rows that the cue would carry past 32; an active linear-attention state capture; a GLP vector bound (`hf2q serve --glp`), dump-bisect or the dense-Q split profile enabled; a checkpoint over its quota or whose storage cannot be allocated; a model with a product off the prepared kernel. A folded checkpoint counts in `hf2q_qwen_anchor_snapshot_milliseconds_count`; its time there is only the host time of assembling it, because the state is captured inside the forward. Exact Qwen3.8 four-agent run pair on an undisturbed host, 24 of 28 turns folded: logged prefill 12.18 s to 11.43 s and makespan 37.71 s to 36.61 s; per turn (median, content chunk plus cue chunk against the folded chunk) 218 to 191 ms at 75–118 content rows, 300 to 293 ms at 157, 400 to 367 ms at 200 and 498 to 470 ms at 266, of the 55 ms a cue chunk costs; the folded turns' checkpoint snapshots (about 10 ms each, outside the logged time) are no longer taken. Qwen3.6: 4–9 ms per folded turn model-free (three runs), per-turn medians in the four-agent workload 2–16 ms lower in four of the five folded size classes and 6 ms higher in one; run totals do not resolve it because every run emits a different transcript. |
| `HF2Q_QWEN_PREFILL_JOINT` | off | `0`/`off`/`false`, `1`/`on`/`true` | Qwen bounded prompt prefill on a multi-slot SlotAware worker; requires `HF2Q_QWEN_PREFILL_PREPARED=1` (the worker refuses to start otherwise, as on an invalid value). `1` runs the pending prefill chunks of slots that wait at the same time as one target forward instead of one forward per slot. Each slot keeps its own chunk boundaries, KV rows, DeltaNet state, checkpoint and DFlash2 lane; they are bit-identical to that slot's own prefill (`prefill_joint_tests`, `joint_capture_tests`, `moe_prefill_rows_tests`, and the real-artifact gate on both Qwen models), so the switch changes when a request is served, never what it computes. Cold slots whose prompts start with the same tokens (a shared system prompt and tool block) have those rows computed once in that forward; each slot receives its own copy and nothing is shared after the forward. Policy, fixed per model kind and logged at worker start as `Qwen joint prompt prefill policy`: (1) which slots join the scheduled one: the waiting slots in FIFO order that the rule below admits and that fit the row ceiling, up to the first one the rule or the ceiling turns away; a waiting slot the forward cannot serve at all (vision input, a graft, MTP catch-up, a one-row segment, a client that has gone) is left out and slots behind it may still join, and because joined slots yield their turn it is scheduled first on the next step. The rule: on a MoE model (Qwen3.6) every waiting slot, because a forward of more than 32 rows streams every expert stack once whatever its row count and the joint forward runs the MoE FFN once over all rows; on a dense-FFN model (Qwen3.8) only a slot whose chunk has at most 32 rows and a first prompt that starts with the scheduled slot's first prompt (at least 32 tokens), because a dense forward costs its rows and a mid-size chunk that joined would only make the first slot wait for the second. (2) Row ceiling of one transaction: the worker's per-transaction ceiling (2,048 rows on a multi-slot worker) while no slot decodes, 512 rows while any slot decodes, a quarter of what one slot's first prompt may run in a transaction. At equal rows a joint forward is not cheaper than one slot's: on Qwen3.6 four lanes of 2,048 rows together took 716-760 ms against 590-645 ms for one 2,048-row lane (attention and short segments are built per lane); on Qwen3.8 the two are equal within 3%. (3) While nothing else runs on the worker and a slot is free, the first prompt of a conversation waits up to 25 ms for other first prompts, once per cohort, so agents started together prefill in one forward; no other request is ever delayed. (4) A scheduler step with decoding and prefilling slots runs the prefill transaction before the decode cycle (off: the cycle first). A decoding slot's cycles and the transactions alternate either way, so its largest gap between tokens is one transaction plus one cycle in both orders; the waiting request gets its first token one decode cycle earlier. A slot that cannot join (vision, graft, MTP catch-up, a one-row segment) runs its own step; a failed joint forward fails its slots as their own failed forwards would. On a MoE model no joint forward is taken while `HF2Q_MM_ID_ROUTING_THRESHOLD` is set (any value): the forward runs the MoE FFN once over all segments of more than 32 rows, which is exact only at mlx-native's stock route boundary; every slot then keeps its own forwards. |
| `HF2Q_QWEN_PREFILL_WAIT_LAYERS` | `1` | `1`..`8` | Qwen text prompt prefill (chunks and joint forwards). By default the layer loop of a forward of more than 32 rows waits for the GPU after every layer, which bounds how long a Metal fault goes unobserved, and under `HF2Q_ENCODER_SESSION=1` every stage of a layer ends in a shared-event fence; the GPU idles about 0.33 ms per layer between those command buffers. With a value above 1, under `HF2Q_ENCODER_SESSION=1` a layer is one command buffer where its stages allow it (a DeltaNet layer's projection stage, recurrence stage and FFN, instead of three; an attention layer's output stage and FFN), each layer's command buffer is committed as soon as it is encoded and without the fence (the command queue's commit order sequences it, as for the command buffers of a forward without the encoder session), and the host waits once per that many layers, so the GPU runs a layer while the host encodes the next. A forward of at most 32 rows is submitted the same way at its `HF2Q_FFN_TERMINAL_K_BATCH` cadence. Without `HF2Q_ENCODER_SESSION=1` the stages of a layer stay separate command buffers, committed as at the default, and the switch only moves the host wait to every N layers. At every value of this switch a multi-row forward keeps the command buffers it commits without a wait (the embedding, every stage command buffer of a layer in both session modes, the joint forward's gather and join, and at a value above 1 whole layers) until its next host wait, which waits for each one and reads each one's status (command buffers of one queue may overlap, so the last one's status says nothing about the others); before it returns either result it settles what is left, a failed command buffer is the forward's error even when the host saw none, and a forward that fails waits for everything it committed before the error leaves it, so its slots can be reset and reused at once. Memory: the worker's scratch pool takes its buffers back only at a host wait, keeps them in exact power-of-two buckets and never shrinks on its own, so layers that do not wait accumulate their scratch. While a forward that skips host waits or a joint forward (`HF2Q_QWEN_PREFILL_JOINT`) runs, and until the pool reset after it, three rules hold for every model geometry, slot count and sequence of forwards. (a) Hard cap, checked before every pooled allocation: the pool never holds more than 3 GiB; a request that could take it over is served by a plain device buffer that is released at the next host wait. (b) A pool reset releases the pool's free buffers when they exceed 2 GiB, and so does the start of such a forward when earlier forwards left the pool larger, before its first capped allocation. (c) A layer that ends with more than 1 GiB of scratch out waits. This is a threshold, not a cap: the layer that crosses it has already allocated, so between two host waits up to 1 GiB plus that one layer's scratch is out; large forwards therefore wait more often than every N layers. Released buffers are allocated again when needed, which costs time, never memory. The scratch one layer requests is derived from the prepared route's request sites for the model's geometry, the forward's rows and its slots, bucket rounding and the per-slot copies of a shared prefix included (`prefill_layer_scratch_bound_bytes` in `prefill_prepared.rs`: 2.31 GiB for four slots and 2,048 forward rows on Qwen3.8, 1.38 GiB on Qwen3.6; 1.52 GiB and 0.72 GiB for a single-slot 4,096-row chunk); every layer is checked against it and a layer over it is reported. What memory admission charges, absolute and not per switch: for `HF2Q_QWEN_PREFILL_JOINT` the 2 GiB of (b), one layer, and the buffers a joint forward allocates for its own life beside the pool, derived in `joint_arena_bound_bytes` (per slot and segment the attention projection and prefill arenas and positions, per slot the dequantized history of a resumed segment over at most the cache's rows and the gathered rows behind a shared prefix, on a MoE model the forward-wide FFN arena and one per short segment, the DeltaNet run scratch), together `joint_scratch_bound_bytes`; for this switch above `1` the 2 GiB of (b), the 1 GiB of (c) and one layer (`wait_layers_scratch_bound_bytes`; a forward that skips host waits allocates no per-forward buffer a waiting forward does not); with both on, the second figure with the joint shape plus the joint arenas covers both. The worker logs them at start (`Qwen prompt prefill scratch bounds`). The derivation covers the prepared route (`HF2Q_QWEN_PREFILL_PREPARED` with every projection of the layer on the prepared kernel), and a forward skips host waits only there: a value above `1` without `HF2Q_QWEN_PREFILL_PREPARED` stops the worker at start, and a forward that a route switch takes off the prepared route (a bound GLP vector, a dump, the dense-Q split profile) keeps its waits. With the growth-limited pool investigation profile, or with `HF2Q_DENSE_Q_ARENA_RESET=0` (no scratch reset at a layer's host wait), the switch keeps the wait after every layer; with that override the cue fold and the joint forward are not taken either. Measured before the caps existed, one forward sequence up to four lanes and 2,900 rows at `8` against `1`: 4.37 against 0.98 GiB in the pool on Qwen3.8, 1.34 against 0.47 GiB on Qwen3.6. A layer whose output the host copies (a DFlash2 target layer) still waits, as does the last layer; forwards with vision input, a steering vector, a recovery capture or a diagnostic dump keep their waits. Dispatches, their order and their operands do not change, so no bit does (`waiting_every_few_layers_leaves_the_bits_of_waiting_every_layer`, `a_joint_prefill_that_waits_every_few_layers_keeps_each_lanes_bits`, both also under the encoder session). Measured on the real artifacts at `8`, model-free of the server: 6-8 ms less per Qwen3.6 forward (49.0 to 41.4 ms at 33 rows, 82.1 to 73.9 ms at 160) and 12-14 ms less per Qwen3.8 forward (116.7 to 104.1 ms at 33 rows, 265.1 to 252.9 ms at 160). An invalid value stops the worker at start. |
| `HF2Q_QWEN_PREFILL_KEEP_FRESH` | off | `0`/`off`/`false`, `1`/`on`/`true` | Qwen SlotAware workers. On Apple GPUs a buffer that no command buffer has used for about a second costs time on its next use, in proportion to its size. The decode routes run on prepared copies of some weights (the MoE experts' tile planes, the shared output head planes) while the prompt prefill reads the GGUF blocks of the same weights, so a stretch of decode cycles leaves those blocks unused and the next prompt prefill pays for all of them. `1` makes the worker issue one small command buffer at most every 250 ms between decode cycles: a four-byte read of each such buffer (per MoE layer the routed experts' gate, up and down stacks and, with live tiles, the shared expert's three weights; the GGUF output head) into private scratch, committed without a wait. It changes no model state and no output bit; a failure is logged once and ends the keep-alive without failing a request. Measured on the real artifacts: a resumed 157-row Qwen3.6 chunk after 1.5 s of decode cycles takes 190-193 ms without the switch and 85-97 ms with it (80 ms with no pause); Qwen3.8 has one such buffer (the output head) and gains nothing. An invalid value stops the worker at start. |
| `HF2Q_QWEN_VERIFY_ATTN_TILE` | off; `1` in the route profile of a selected DFlash2 target | `1` | Qwen DFlash2 target verification only. Exact value `1` runs the full attention of every verify step as a tiled pass over the unchanged TQ-HB cache: one batched K and one V encode for all rows (the serial bytes), one query rotation, one split dispatch that computes a lane's whole window and a KV head's query heads together with Metal tensor matmuls, one reduce and one fused un-rotation + gate, instead of the serial chain once per row. Cache bytes and cursors are the serial chain's; the attention output differs in bits (Class B) and does not depend on the cohort. Needs a TQ-only cache with 8-bit codes, head dim 256, six or eight query heads per KV head and the Metal tensor API; otherwise the step keeps the per-row chain. Joint decode, MTP verify and prefill are unchanged. Resolved once per model load. |
| `HF2Q_QWEN_VERIFY_MIXER_FUSED` | off; `1` in the route profile of a selected DFlash2 target | `1` | Qwen DFlash2 target verification only. Exact value `1` changes how each target layer's mixer is issued, not what it computes (Class A: rows, target captures, DeltaNet state and the rows retained for the replay commit are bitwise the stock route's; padding rows of the DeltaNet gated output are zero. The equality is qualified by test on the Apple M5 Max; on any other GPU family mlx-native `tests/test_gdn_window_fused.rs` must pass first). With `HF2Q_REPACKED_GGUF_M32_PROJ=1` the prepared input products of a layer (DeltaNet alpha/beta, qkv and z gate; attention Q/gate, K and V) run as one wave wherever they are prepared (32-row steps; every tile width, so Qwen3.6 steps of 4, 8 and 16 rows too, with `HF2Q_REPACKED_GGUF_ANY_WIDTH=1`; the in-situ numbers below predate that switch and were not repeated with it), one kernel pass each with no reduce pass or barrier between them, and when every product of a layer is prepared its pre-mixer RMS norm writes their BF16 rows without a cast dispatch. The DeltaNet core of every lane window (convolution, Q/K norms, query scale, gates, recurrence, gated norm, retained rows) runs as one dispatch that writes the BF16 rows of the output projection: 7 dispatches in five serialized stages per DeltaNet layer at four lanes instead of 159 in 14. The core needs DeltaNet head dimension 128, two to nine convolution taps, windows of at most eight rows and a device whose thread and threadgroup-memory limits admit its 256-thread pipeline; a layer that does not qualify keeps the stock chain, and the first use and the first refusal (with its reason) are logged at info level. Exact four-agent runs with every other combined-candidate switch on (two per arm and model on each of two binaries, all passing): Qwen3.8 four-lane cycle 75.1/75.6 to 70.6/70.9 ms and, on a slower host, 92.6/92.3 to 86.0/88.0 ms (about 6% both times), 9,063 to 1,703 dispatches per cycle, identical text in both arms; Qwen3.6 33.3/33.9 to 31.2/31.7 ms and 36.6/36.4 to 35.6/34.7 ms, 6,316 to 1,716 dispatches. No paired campaign has been run. Needs unreleased mlx-native kernels. Decode steps and MTP verify never take it. Resolved once per model load. |
| `HF2Q_QWEN_MOE_BATCHED` | off; `1` in the Qwen3.6 route profile of a selected DFlash2 target | `0`/`off`/`false`, `1`/`on`/`true` | Qwen3.6 MoE artifacts. `1` loads the router (`ffn_gate_inp`, F32), shared-expert gate logit (F32) and shared expert (Q4_K or Q6_K) in their GGUF storage instead of BF16 copies, and every MoE decode layer then runs one batched route over the step's rows (F32 row-exact router mat-vec, SIMD-group top-k over all rows, one `mul_mv_id` each for expert gate/up/down over rows × top-k pairs, one weighted reduce). A row's output is bitwise the same solo (serial decode, MTP drafting) and in a joint step of 2–8 rows, but differs from the default BF16 route (Class B: the router and shared expert are no longer BF16-rounded). Prefill reads the same native weights. Layers with other storage types keep the default route; DWQ affine expert overlays refuse to load. Read once at model load; the worker-start `Qwen joint decode configuration` line logs `moe_ffn=batched_native`. Invalid values warn and use off. |
| `HF2Q_QWEN_MOE_STAGED_M8` | off; `1` in the Qwen3.6 route profile of a selected DFlash2 target | `0`/`off`/`false`, `1`/`on`/`true` | Qwen3.6 Q4_K/Q6_K MoE artifacts. Requires `HF2Q_QWEN_MOE_BATCHED=1`. `1` prepares quantized expert gate/up/down planes at model upload; separate grouped staged-M8 F32 gate and up projections reuse one compact route schedule and gathered input, followed by the established F32 SwiGLU and an F32-input prepared down projection. Unsupported top-k, shapes, quantization, affine overlays or imatrix collection fail startup. This is a quality-qualified opt-in, not the release default; unset or `0` keeps the established native expert matvecs. |
| `HF2Q_QWEN_MOE_LIVE_TILES` | off; `1` in the Qwen3.6 route profile of a selected DFlash2 target | `0`/`off`/`false`, `1`/`on`/`true` | Qwen3.6 Q4_K/Q6_K MoE artifacts. Requires `HF2Q_QWEN_MOE_STAGED_M8=1` (and so `HF2Q_QWEN_MOE_BATCHED=1`). The three switches are resolved together at model load: an invalid value, or this switch without staged M8, fails startup whatever the other switches say, and the worker logs the resolved route as `moe_experts=matvec|staged_m8|live_tiles` in `Qwen joint decode configuration`. `1` also prepares the shared expert's planes at model upload and runs each MoE layer as one schedule (which also applies a DFlash2 verify-row mask), one gather, one gate/up pass that leaves a BF16 SwiGLU in the schedule's rows, and one down pass over those rows; the routed experts' tiles of at most eight routes and the shared expert's tiles (one per block of eight rows) share those dispatches. The passes launch only the live tiles, so rows a DFlash2 verify budget masks cost no routed-expert work. Router and shared-gate logits run on the single-row kernel, every full block of eight rows in one dispatch. A layer whose routed and shared planes differ in format or expert width, or with more than 256 experts or routes a step, runs the same arithmetic stage by stage (two schedules, two dispatches per pass) with the same bits; model upload logs `Qwen MoE live-tile layers` once with `fused_layers` and `stage_layers`, the number of layers on each form. A row's result does not depend on the other rows or lanes of its step and is bitwise the row decoded alone under this switch, except where an expert product underflows FP32: the one- and two-route register path and the matmul tile may round such a product differently (observed on results of magnitude at most 4e-31). Expert arithmetic differs from the staged-M8 route (half-precision staged weights, one BF16 rounding of the SwiGLU) and the shared expert moves from FP32 to that arithmetic. Experimental opt-in, not quality-qualified; unset or `0` keeps the staged-M8 route. |

## DeepSeek-V4 decode

With two or more agents decoding at once, DeepSeek-V4 advances every eligible slot through one joint forward per token step. Each row is bitwise identical to serial decode. Joint decode engages only for artifacts whose batched products all use proven-exact storage types: Q8_0, Q2_K or Q3_K; float weights only where they are split per row; grouped output-A only as Q8_0. Other artifacts stay serial, and the reason is logged at worker start. See [ADR-063 S3](research/adr063-s3-ds4-joint-decode.md).

| Var | Default | Values | Effect |
|---|---|---|---|
| `HF2Q_DEEPSEEK_JOINT_DECODE` | on | `0`/`off`/`false`, `1`/`on`/`true` | Escape hatch. `0` restores serial per-slot decode, including the exact B=4 cohort; it does not turn off rows steps over token-wise prompt work, which run on the same graph at one slot too and have their own kill switch (`HF2Q_DEEPSEEK_TAIL_ROWS_STEP=0`, below). Read once at worker start. It is also forced serial by `MLX_UNRETAINED_REFS`, DeepSeek layer dumps or stage profiling, and an explicit `HF2Q_DECODE_MV_EXT=1`. The routing policy is frozen into the DeepSeek-V4 model's registries at load, so loading a Qwen3.8 model in the same process does not affect it. |
| `HF2Q_DEEPSEEK_JOINT_DECODE_MAX_ROWS` | `4` | `2`–`8` | Widest joint step. Widths above 4 are covered only by the official-artifact bitwise test and have no serving measurement; use them only for experiments. Invalid values warn and use 4. |
| `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC` | on | `0`, `1` | Kill switch of the row-exact few-row products (default since v0.1.30). On: F16 products at one row and independent F16 rows, Q8_0 products at up to 8 rows and Q8_0 grouped output-A run one mlx-native `row_exact_matvec_stock` dispatch whose every row is bitwise the stock one-row result. A product the kernel does not reproduce (other storage types; Q8_0 under `HF2Q_Q8_0_MV_NR2=0` or at 2–8 rows under `HF2Q_DECODE_MV_EXT=1`) keeps its stock route. `0` restores one dispatch per row. Any other value refuses the model load. |
| `HF2Q_DEEPSEEK_TAIL_ROWS_STEP` | on | `0`, `1` | Kill switch of rows steps over token-wise prompt work (default since v0.1.30): the 8-token recovery tail and cached suffixes below the 33-token matrix floor run as rows steps of up to 8 tokens whose every row and cache byte equal the one-token steps. While joint decode is unavailable for the artifact or process the work runs one-token steps and the load logs why. `0` restores one-token steps. Any other value refuses the model load. |
| `HF2Q_DEEPSEEK_BATCHED_ATTN_CORES` | on | `0`, `1` | Kill switch of the batched joint attention cores (default since v0.1.30): per layer, one rows dispatch per op covers every joint row, each row bitwise equal to its per-row core. A layer with a row at position 0 keeps the per-row cores. `0` restores one row at a time. Any other value refuses the model load. |

The three routes above are the configuration of the DeepSeek-V4 release gate (ADR-063 Q4-DS4-RACE, campaign `q4-deepseek4-ds4file-release-gate-v4`) and change dispatch counts only: every output bit equals the v0.1.29 routes, on hf2q's own artifact and on the ds4 file. They are read once at model load, which logs one audit line, `DeepSeek-V4 default routes resolved`, with each value, its source (`default` or `env`), the tokens per token-wise step and whether the joint routes are available. The line has log target `hf2q::audit`, which `hf2q serve` shows without `-v`; the load banner prints the same values as `routes = …`, and `/hf2q/v1/runtime` publishes them as `measurement_snapshot.active_controls.default_routes` (`deepseek_row_exact_matvec`: `RowExact`/`Stock`, `deepseek_tail_rows_step`: `On`/`Off`, `deepseek_batched_attn_cores`: `Batched`/`PerRow`, each with its `source`). No other model family reads them.

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

### Gemma 4 release-gate routes (default on)

Since v0.1.30 the four routes of the Gemma 4 release gate are the defaults
of a Gemma 4 model (ADR-063 Q4-GEMMA-ATTN, campaign
`q4-gemma-release-gate-max40-v3`): decode attention head split NE 4,
`row_exact` decode dense kernel, tensor-API warm attention and the Q5_0
tensor matmul. Nothing has to be set to get them; each switch below is a
kill switch. No other model family reads them or changes a bit.

Every Gemma 4 model load logs one audit line, `Gemma 4 default routes
resolved`, with each resolved value and its source: `default`, `env` (the
variable is set), or `fallback` (unset, and the default route is unavailable
on the device). The line has log target `hf2q::audit`, which `hf2q serve`
shows without `-v`; the load banner prints the same values as `routes = …`.
`/hf2q/v1/runtime` publishes the four values under
`measurement_snapshot.active_controls` and, with their sources, as
`measurement_snapshot.active_controls.default_routes`. To serve exactly the
v0.1.29 route set, set all four kill switches:

```
HF2Q_GEMMA_DECODE_ATTN_NE=0 HF2Q_GEMMA_DECODE_DENSE=0 \
HF2Q_GEMMA_WARM_TILED_ATTENTION=0 HF2Q_Q5_0_TENSOR_MM=0
```

All four accept the same off spellings: `0`, `off`, `false` or `stock`
(case-insensitive). A set value that is not one of a switch's values
refuses the Gemma 4 model load, so a mistyped kill switch never serves the
route it was meant to turn off. Unset (or empty) is the only way to get a
default.

The defaults never refuse a model: a layer, weight or device a route does
not admit keeps the stock route (details per switch).

Retained Gemma Q1 packets (`decode_dense_kernel`, `decode_attn_ne`,
`warm_live_kernel` and `q5_0_tensor_mm` absent) mean the kill-switch routes.
`scripts/four_agent/q1_endpoint_supervisor.py` refuses ambient `HF2Q_*` and
sets the child's four switches from the packet alone, so a retained packet
replays under the kill switches without any variable, and a default-route
run declares `TensorF16`, `RowExact`, `Ne4` and `true` in its packet.

#### Gemma decode attention head split (default NE 4)

`HF2Q_GEMMA_DECODE_ATTN_NE` selects the decode attention kernel of d256
(sliding) layers on every decode path (batched cohort dispatch, mixed-bucket
per-slot fallback and single-slot decode). It is read once when the model
loads.
- **Unset (default) or `4`:** mlx-native's NE 4 head-split kernels. They
  share their per-row math, so a slot's decode is bitwise the same on each
  path. Numerics class B against the stock kernels.
- **`0`, `off`, `false` or `stock`:** kill switch; the stock vec-hybrid
  kernels (`1`, NE 1, is the same route).
- **`2`:** the NE 2 head split.
- **Any other value:** the model load fails.

Layers whose head dimension is not 256 (the d512 global layers) keep the
stock kernels under every value. `/hf2q/v1/runtime` publishes `Stock`, `Ne2`
or `Ne4` as `measurement_snapshot.active_controls.gemma_decode_attn_ne`. The
Gemma launcher forwards `DECODE_ATTN_NE=0|2|4`.

#### Gemma warm attention (default tensor)

`HF2Q_GEMMA_WARM_TILED_ATTENTION` selects the attention kernel of Gemma 4
warm prefills (ADR-063 Stage 2): every single-slot prefill that resumes a
cached prefix, and every lane of a joint warm cohort. It is read once when
the model loads.
- **Unset (default) or `tensor`:** mlx-native's tensor-API flash attention
  over the slot's compacted K and dequantized V range, one sequence per
  dispatch (1,024-query sliding chunks, 4,096-query global chunks). Suffixes
  shorter than 32 tokens keep vec-hybrid attention. A joint cohort stays
  bitwise equal to its per-slot prefills. When the switch is unset and the
  device cannot build the tensor pipelines (no Metal tensor API), the load
  logs `Gemma warm live attention default is unavailable; using the stock
  kernel` and serves vec-hybrid; an explicit `tensor` on such a device
  refuses the model load with the reason.
- **`0`, `off`, `false` or `stock`:** kill switch; warm prefills use the
  vec-hybrid kernel (one decode-style pass per query over the whole
  history).
- **`1`, `on` or `true`:** opt in to tiled F16 flash attention with the same
  chunking. This route is not a default: on the canonical four-agent task it
  moved a follow-up turn's top-2 log-probabilities from −0.149/−1.978 to
  −0.368/−1.179 and changed the written code (`$E/gemma-divergence-v1`,
  ADR-063 quality gate, 2026-11-01).
- **Any other value:** the model load fails.

Transient memory of the tensor route. Beside the cache itself the route
needs scratch that lives for one prefill forward:
- *Sliding layers:* a stage of at most one window of history plus the
  chunk, so its size follows the suffix rows, not the history.
- *Global layers:* dense F16 K and V copies of the sequence's whole history
  (`4 · kv_heads · head_dim` bytes per key: 4 KiB per key on 26B-A4B) and an
  additive mask of `2 · rows · keys` bytes with its tile-classification
  buffer (one byte per tile of 16 queries × 64 keys at head dim 512, at most
  32 KiB). hf2q builds the copies once per
  sequence and layer, runs the sequence's queries in sub-chunks whose mask
  stays under 64 MiB, and waits for the GPU before it would hold more than
  512 MiB of submitted global-layer scratch.

Envelope of the global-layer scratch, for any number of lanes and layers:
the larger of 512 MiB and one sequence's copies of one layer, plus one mask
(at most 64 MiB) and its tile-classification buffer (at most 32 KiB). Every
buffer of the global path is accounted at its exact size
(`tensor_global_envelope_bytes`); the table rounds to MiB.

| Keys (history plus suffix) | One sequence's copies | Envelope |
|---:|---:|---:|
| up to 131,072 | up to 512 MiB | 576 MiB |
| 200,000 | 781 MiB | 845 MiB |
| 262,144 | 1,024 MiB | 1,088 MiB |

The bound needs sessions that can be finished between sub-chunks. The
development flag `HF2Q_GRAPH_OPT_PREFILL=1` records each prefill layer's
session, which cannot be finished mid-layer, so under it the unset default
serves vec-hybrid (logged at load, source `fallback`) and an explicit
`tensor` refuses the model load.

The vec-hybrid kill switch needs none of it. The splits and waits change no
output bit (the kernel's results do not depend on where a chunk starts).
They do not act on the release-gate workload (histories under 2,000
tokens): a sequence splits only above 8,192 keys at a 4,096-row suffix, and
the first wait appears once one forward's global-layer scratch passes
512 MiB (for example four lanes at about 8,000 tokens each). The ignored
real-artifact gate `gemma_warm_tensor_scratch_is_bounded_and_bitwise_at_depth`
(`HF2Q_GEMMA_WARM_DEPTH_TOKENS`, `_ROWS`, `_LANES`, `_ARMS`) checks the
envelope and bitwise equality with the unbounded route at a chosen depth and
prints wall time and peak transient Metal allocation per arm.

The switch also covers installed long-prompt lanes (prompts over 4,096
tokens), which run the single-slot warm route (#725). The model load logs `Gemma warm live
attention kernel`, each live prefill banner prints `live=`, and
`/hf2q/v1/runtime` publishes the resolved kernel (`TensorF16`, `VecHybrid`
or `TiledF16`) as
`measurement_snapshot.active_controls.gemma_warm_attention_kernel`.
Evidence: `docs/research/adr063-gemma-four-agent-reuse.md`,
`docs/research/adr063-q4-gemma-attn-tensor-design.md`. A precise F32
tiled route was evaluated and falsified (ADR-063 Q4-GEMMA-ATTN,
`docs/research/adr063-gemma-tiled-attn-design.md` §12).

#### Gemma decode dense kernel (default `row_exact`)

`HF2Q_GEMMA_DECODE_DENSE` selects the kernel of the dense GGUF matmuls of
Gemma 4 decode (attention q/k/v/o, dense FFN gate/up/down and the lm head) in
the solo decode, the `[N,hidden]` batched body, the batched and fused lm head,
the speculative verify heads and tree-verify layers, and the prefill
first-token head, so every logit row comes from the same per-row kernel (ADR-063
Q4-GEMMA-DECODE op parity). It is read once per process, at model load.
- **Unset (default) or `row_exact`:** the row-exact few-row mat-vec
  (`row_exact_matvec`) for Q4_K or Q6_K weights with `K % 256 == 0` (Q4_K
  also needs a 16-byte-aligned weight start) and Q5_0 or Q8_0 weights with
  `K % 32 == 0`.
- **`stock`, `0`, `off` or `false`:** kill switch; the previous routes.
- **`mma`:** the column-invariant few-row MMA mat-vec (the peer's
  `mul_mv_mma` technique) for Q4_K, Q6_K or Q5_0 weights with `K % 64 == 0`
  (and the type's block size) and Q8_0 weights with `K % 32 == 0`. Numerics
  class B against the default routes.
- **Any other value:** the model load fails.

Under `row_exact` and `mma` each row's result is bitwise the same however many
rows share the dispatch, so a slot's logits do not depend on the decode width
or path; products wider than 8 rows run in chunks of at most 8 rows. Whether a
weight takes the kernel depends on the weight alone, never on the width; every
other weight (F32 router, F16/BF16, DWQ affine overlays, other quantization
types, other K) and the other prefill matmuls keep their stock routes, so no
artifact is refused. `/hf2q/v1/runtime` publishes the resolved kernel (`Stock`,
`RowExact` or `ColumnInvariantMma`) as
`measurement_snapshot.active_controls.gemma_decode_dense_kernel`; Gemma Q1
endpoint and localizer packets declare it as `decode_dense_kernel` (a packet
without the field means `Stock`, the kill-switch route).
The Gemma launcher forwards `DECODE_DENSE=stock|row_exact|mma`; the Q5 case
maps `HF2Q_Q5_GEMMA_DECODE_DENSE` to that launcher variable. The Q1 endpoint
supervisor sets the switch from the packet's `decode_dense_kernel`.

#### Q5_0 tensor matmul (default on for Gemma 4)

`HF2Q_Q5_0_TENSOR_MM` selects whether Q5_0 dense and expert matrix
multiplications use mlx-native's tensor pipeline. mlx-native's own default
is off and stays off: hf2q sets the routing-policy field `q5_0_tensor_mm`
in the policy a Gemma 4 model freezes into its own kernel registries, so a
Qwen, DeepSeek-V4 or embedding model in any process keeps the simdgroup
route unless the variable is `1`.
- **Unset (default):** on for a Gemma 4 model, off for every other family.
- **`1`:** on for every family (mlx-native's reading, as before).
- **`0`, `off`, `false` or `stock`:** kill switch; the simdgroup route.
- **Any other value:** a Gemma 4 model load fails. Other families do not
  read the variable in hf2q; mlx-native treats every value except `1` as
  off.

The tensor pipeline still depends on mlx-native's device probe; a device
without it serves the simdgroup route, as for the other tensor-routed
types. The first Gemma 4 model load of a process logs one line `Gemma4
native routing: q5_0_tensor_mm=<true|false> mm_id_schedule=<grid|compact>`,
and `/hf2q/v1/runtime` publishes the value the loaded model froze as
`measurement_snapshot.active_controls.q5_0_tensor_mm` (`true`/`false`) beside
`mm_id_schedule`. The Gemma launcher forwards `Q5_0_TENSOR_MM=0|1` and the
Q5 case `HF2Q_Q5_Q5_0_TENSOR_MM`.
