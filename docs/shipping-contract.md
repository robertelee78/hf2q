# hf2q Shipping Contract

> Terminology: "the peer" = llama.cpp, the pinned upstream GGUF engine (see NOTICE, data/llama_cpp_pin.txt).

Current published release: `v0.1.13`.

This document defines the public hf2q product surface and the **next-release
candidate** where explicitly marked. It also defines
the policy each environment variable is classified under. Per-variable
effects live in `docs/operator-env-vars.md`; this document sits one level above
and defines *what is supported*.

**Anything not listed in Category 1, its explicitly named appendices, or
Categories 2–3 below may be removed or relocated without notice** — it is not
part of the supported surface. Support is per operation; conversion support
does not imply native generation or serving support.

---

## Category 1 — Production contract and next-release candidate

What the default release binary does with **no environment variables set**.
The exact model-family surface is explicit below; no family inherits another
family's graph, cache, or scheduler contract by approximation.

### Supported family and command matrix

| Family / emitted GGUF architecture | `hf2q convert` | Native runtime surface | Boundary |
|---|---|---|---|
| Gemma 4 (`gemma4`, including source-matched projectors) | Supported | CLI generation and OpenAI-compatible chat, SSE, tools, embeddings, and qualified vision serving | Uses the Gemma graph and cache contract only. |
| Qwen3.5 / Qwen3.6 (`qwen35`, `qwen35moe`) | Supported | Text CLI generation plus OpenAI-compatible chat, SSE, tools, embeddings, retained-prefix reuse, and source-matched paired vision | Uses the shared Qwen35 autoregressive graph. Multimodal requests must pass the Qwen soft-token, DeepStack, 3D-position, and projector-binding checks. |
| Qwen3.8-27B (`qwen35`) | Supported; an ordinary conversion of a multimodal source automatically publishes the bound text GGUF and F16 projector pair | The Qwen35 surface above, including the qualified SlotAware paired-vision, exact-speculation, and long-context decode paths | `hf2q generate` is text-only; paired vision uses `hf2q serve --mmproj` through `scripts/serve_qwen38_opencode.sh`. The text and projector provenance/digests must match. |
| Legacy Qwen 3 MoE (`qwen3moe`) | Supported | None | Conversion-only. It is not silently routed through the `qwen35moe` runtime. |
| Standalone Qwen3-VL dense (`qwen3vl` / `qwen3_vl`) | Supported | None | CLI generation and server startup fail closed before loading weights pending the ADR-041 engine seam. Qwen3-VL MoE conversion/runtime is unsupported. This is distinct from the qualified Qwen3.8 text/projector pair. |
| DeepSeek-V4 (`deepseek4`) | Supported | CLI generation and OpenAI-compatible chat, SSE, tools, embeddings, and retained-prefix reuse | Uses the DeepSeek-V4 graph and compressed-cache contract only. |
| BERT / Nomic-BERT (`bert`, `nomic-bert`) | Supported | OpenAI-compatible `/v1/embeddings` when loaded with `--embedding-model` | Embeddings-only; no chat generation. |
| Llama 3 / MiniMax M2.7 (`llama`, `minimax_m2`) | Supported | None | Conversion-only; no native generation or serving graph. |

Variants and operations absent from this matrix are unsupported and must fail
closed rather than entering an approximately compatible loader, template,
cache, or forward graph.

- Batched `forward_prefill_batched` (default-on since ADR-028
  iter-344; per-token `forward_prefill` was 14-45× slower than peer).
  Opt out to per-token via `HF2Q_BATCHED_PREFILL=0` for parity
  diagnostics — see Category 2.
- KV representation is family-specific. The canonical Qwen launcher uses
  TQ-packed 8-bit K/V without an F32 shadow; `HF2Q_TQ_KV=0` is the explicit
  conventional-F32 diagnostic. DeepSeek and Gemma retain their own qualified
  cache contracts rather than inheriting Qwen's representation.
- Default decode (single-buffer or dual-buffer internal tuning; not
  user-configurable).
- **Artifact-native Gemma matrix storage.** The loader admits the declared
  GGUF encoding before Metal allocation, maps the exact stored embedding,
  output-head, dense-projection, and expert bytes, and routes those bytes
  through a matching native kernel. A tied output head reuses the embedding
  allocation. Production load does not manufacture an F16, F32, or re-quantized
  shadow; an unsupported encoding fails before model allocation. Explicit
  affine overlay formats remain a separate, declared representation. The real
  A→B→A gate requires Gemma's live artifact mapping to disappear on eviction,
  reappear only for a fresh A generation, preserve exact A replay, and remain
  within the endpoint-based no-double-residency and reload memory bounds.
- **Artifact-native Qwen3.5-family matrix storage.** Before Metal allocation,
  the loader admits every embedding, head, dense, MoE, and MTP matrix codec and
  all serving widths. One model-scoped GGUF mapping owns exact rank-2 matrices,
  rank-3 expert stacks, and the single schema-declared rank-1 shared-expert
  gate exposed as a zero-copy logical row. No other implicit squeeze is
  admitted; only elementwise/state tensors may materialize F32.
  Router and shared-expert matrices are not decoded and uploaded as BF16, and
  tied/shared heads are aliases rather than duplicate allocations. Runtime
  unique-view count and bytes must exactly equal the independent preflight
  matrix receipt with zero anonymous matrix bytes. Model swap applies the same
  mapping-disappearance, fresh-generation, exact-replay, bounded-peak, and
  reclaim contract described above; copied storage is not an accepted Qwen
  matrix fallback.
- **Public by 0.1.6; strengthened through the 0.1.8 release:**
  Qwen3.5/Qwen3.6 and Qwen3.8 generation and OpenAI-compatible
  serving use the shared autoregressive `qwen35`/`qwen35moe` graph by default.
  Slot-aware Qwen prefill
  is bounded and scheduler-yielding; no `HF2Q_QWEN36_AUTOREG` activation is
  required. This default contract is the plain-text unary/SSE chat surface,
  including native tools, reasoning, grammar, and retained-prefix
  continuations. SlotAware multimodal work retains and validates soft-token,
  DeepStack, and 3D-position state and advances prefill in bounded transactions;
  the exact source-matched Qwen3.8 text/projector pair also passed the
  first-image-after-text cache-reuse gate. Unbound projectors and unsupported
  request geometries fail closed. The separate chunk-scan prefill experiment
  remains Category 3.
- **Public by 0.1.6; strengthened in the 0.1.7 release:** long
  plain-text Gemma SlotAware prefill advances in
  at most 4,096-token transactions, split at the stable-prefix boundary. The
  transaction publishes all configured per-layer cache cursors together.
  Compatible installed prefill states may share those 4,096 aggregate rows;
  the bound never multiplies by the number of slots. Long soft-token work
  remains fail-closed until a resumable graph is proven.
- **Public by 0.1.6; strengthened in the 0.1.7 release:** DeepSeek
  meaningful cached suffixes use the same
  atomic resumable verifier transactions as cold prefill. Lopsided cold waves
  use an interactive budget of up to eight decode tokens between prefill slices
  capped at two native windows. When a filling cold cohort still has another
  cold request queued, cold-wave unary decode is deferred through its draining
  phase while any cold prefill remains and full 2,048-token transactions
  resume; unary output could not be delivered before that barrier. Streaming
  and warm decode remain responsive. Without a runnable visible decoder, the
  full prefill transaction is also restored.
  Outside a cold-cohort barrier, staggered warm work may occupy any free
  physical slot. Two to four already-runnable compatible warm matrix suffixes
  may share one layer-local FFN/MoE transaction while attention and cache writes
  remain sequence-local; aggregate rows never exceed 2,048, and no request
  waits or is skipped to form a cohort. Cancellation restores only a valid, position-consistent
  pre-request turn anchor; poisoned or inconsistent state resets fully.
  The 0.1.7 release also pairs large automatic MoE gate/up
  projections through the routing-schedule primitive introduced in
  `mlx-native 0.10.10` and retained by the pinned `mlx-native 0.12.1`.
  Decode-sized and forced diagnostic routes remain independent;
  native microbenchmarks do not replace the exact packed hf2q hardware gates.
- A typed fatal Metal command-buffer/watchdog/ignored-submission error, or an
  independently observed transaction deadline that never returns, fails the
  affected Qwen, Gemma, or DeepSeek worker closed. Every owned reply
  terminates once; saturated SSE consumers cannot block fatal fanout; no cache
  reset or later GPU submission is permitted. `/health` remains process
  liveness while `/readyz` and new generation fail closed. OS process
  supervision, not an in-process slot reset, owns device recovery.

### Required gates before merging

Every Gemma change that could affect the forward pass or lm_head must pass
`scripts/release-check.sh`:

| Gate | Floor |
|---|---|
| `short_hello` exact-byte match vs locked peer reference | ≥ 29 bytes |
| `sourdough` common-byte-prefix with the peer | ≥ 3094 bytes |
| `sliding_wrap` common-byte-prefix with locked hf2q reference | ≥ 700 bytes |
| Decode perf sanity on the sourdough prompt | ≥ 95 tok/s |

Before a Qwen SlotAware serving change merges, it must additionally pass the
applicable gates below from a clean packed artifact that resolves the
published, checksum-pinned `mlx-native` dependency. The resulting evidence is
authority for those model/serving bytes; an unchanged distribution-only
descendant does not rerun it merely to publish the CLI:

| Gate | Contract |
|---|---|
| Hosted model-free gates | Bounded 2,048-token plan, decode-first `Mixed`, cold round-robin, fatal fanout, readiness, request-boundary tests, stable fixture bytes, and the receipt-parser negative matrix pass in CI. |
| Apple-Silicon artifact gates | Cross-layer/MTP cursor-ledger coherence and transaction-boundary cancellation pass against the packed candidate; these require the native cache/model path and are not inferred from hosted scheduler tests. |
| Exact overlap | The deterministic 552-token SSE lane is enqueued immediately before the 87,972-token/347-tool lane; the short lane makes semantic progress while the long lane completes exactly 42×2,048 + 1,956 prompt tokens. |
| Disconnect | Dropping the long SSE is observed at a transaction boundary, releases the same physical slot once, and a following request succeeds. |
| Agentic four-slot gate | Required/automatic tools, unary/SSE, tool-result continuation, exact arguments, and retained-prefix reuse pass for four independent slots. Qwen uses the canonical prompt-visible `/opt/hf2q/Cargo.toml` path, a direct-tool system instruction, and an unambiguous completed-tool-result envelope; their SHA-256 identities are receipt-bound so an ephemeral package path or prompt rewrite cannot silently change the workload. |
| Qwen3.8 exact speculation | `scripts/qwen38_speculation_ab.sh` requires the expected `MODEL_SHA256` and runs fresh one-slot servers in fixed OFF/AUTO/AUTO/OFF order from the same binary and artifact. Each arm executes three deterministic code prompts and three repeat-heavy prompts; all 24 `choices[0]` values must be byte-identical, AUTO must record accepted target-verified proposals, and each workload's six-sample per-mode median throughput must improve by at least 5%. The exact artifact must also pass required-tool unary, tool-result continuation with nonzero retained-prefix reuse, valid SSE plus one `[DONE]`, cancellation recovery, and a four-slot exact-output wave. The four-slot wave is a correctness gate only until a true width-N body/head establishes aggregate throughput; scalar slot interleaving must not be marketed as batched speed. |
| Qwen3.8 short/long decode | The same packed binary and model run a fixed OFF/AUTO/AUTO/OFF comparison. Every fresh server first emits 512 greedy tokens below the 8,192-token selector crossover, then emits 512 greedy tokens at 100K–120K prompt tokens. Short and long output bytes must match across all arms; AUTO's short mean may regress by at most 2%; AUTO must improve long mean decode throughput by at least 15%; and each short and long arm's two trials must remain within 5%. AUTO must reduce mean long-request curl wall time. Every request binds curl time and shell phase time within two seconds of the response's total timer, requires its own unary SlotAware decode-complete event from the same decode clock, and runs inside the release gate's continuous fair-or-better thermal envelope. The short log snapshot must prove AUTO stayed on the scalar path below crossover; the long AUTO log must prove Q2 selection. A benchmark-only summary is diagnostic evidence, not release authority. |
| Native lifetime/fatal recovery | Exact-artifact hardware waves keep command-buffer and CFString populations bounded and reject every timeout or ignored-submission signature. Packed model-free fail-stop and supervisor tests inject the fatal return/dead-worker state, prove no post-fatal submission, preserve `/health` as process liveness, and require `/readyz` plus new generation to fail closed. The hardware gate does not intentionally poison Metal. |

The shared cross-family changes additionally require:

The hardware binary is compiled only from the `.crate` unpacked into a fresh
temporary directory outside the source checkout. That build uses a fresh,
checkout-disjoint `CARGO_HOME` and target directory, clears Rust toolchain,
compiler, documentation, flags, wrapper, target, and profile override
variables, and rejects Cargo config anywhere in the packed root's ancestry.
Its dependency receipt binds the packed `Cargo.lock` and raw `cargo metadata`
bytes, including the exact `mlx-native 0.12.1` crates.io source and checksum.
The protected release workflow rehashes and revalidates those downloaded raw
files, then requires its newly packed `Cargo.lock` to be byte-identical before
publishing.

The protected cross-family gate content-hashes a model artifact once for each
stable file identity. It records the exact path, digest, device/inode, size,
modification time, and change time in the self-hosted runner's persistent tool
cache. Later release attempts, child gates, and the terminal manifest boundary
reject a changed or replaced file using that receipt without rereading the
complete GGUF. Standalone harnesses without a parent receipt continue to verify
the full model digest themselves.

The workflow replaces its step shell with a process-group supervisor before
starting the hardware gate. Cancellation terminates that scoped group, so an
in-flight compiler, model runtime, and power helpers cannot survive a canceled
job as orphan processes.

Calibrated macOS gates compile the checked-in Foundation thermal-state helper
once before any model load and reuse that executable for every observation.
They must not launch the Swift interpreter/compiler inside the sampling loop.
The helper source, compiler, executable result, and returned state are
validated fail-closed; cleanup may remove only the exact private probe path.
This implementation rule preserves the existing two-second measurement
cadence, five-second maximum gap, and eight-second settle-gap limit. It does
not authorize a wider thermal envelope or a product-latency SLO change.

DeepSeek calibrated phases also sample the host process table at every thermal
observation. The release gate's isolated process group is the ownership
boundary: its sealed server, shell harnesses, and prebuilt test binary are
allowed, while any compiler (`cargo` or `rustc`), any `llama-cli` or
`llama-server` runtime, and every `hf2q` or `hf2q-*` process outside that group
invalidates the measurement. Contention resets the trailing 60-second settle
window and fails an active measurement; the guard never signals foreign work.
Separate settle and measurement logs are content-hashed into the
offline-verifiable receipt and must match the thermal sample timestamps and
phases exactly. This is an evidence-integrity gate, not a load-average
threshold or a latency rebaseline.

| Family | Candidate artifact gate |
|---|---|
| Gemma 4 | Fresh-versus-reused bounded output parity at the 4,096 boundary and the non-aligned 8,193-token tail; aggregate cross-slot and installed-state transaction rows remain <=4,096 at both four and eight configured slots; short-SSE/long-prefill overlap; transaction cancellation; existing agentic/cache gate; bounded native object populations. The two four-slot calibrated waves retain the default latency limits, run before the destructive 175K/120K soak, and each require a trailing 60 seconds of Nominal state plus fail-closed two-second sampling through the complete cold/cached/tool-result sequence. The experimental eight-slot correctness/aggregate-cap wave is not a latency SLO, but its 40-second TTFT, 60-second whole-response, and 30-second tool-result functional ceilings are accepted only after the already-loaded eight-slot process receives its own trailing 60-second Nominal settle and continuous full-wave thermal receipt binding all eight cold requests. The transaction cap is not accepted until this passes. |
| Qwen3.5 / Qwen3.6 / Qwen3.8 | Bounded 2,048-token SlotAware prefill, exact short-SSE/long-prefill overlap, cancellation recovery, four-slot agentic/tool/cache semantics, and native lifetime checks pass. Qwen3.8 additionally binds the exact speculation and short/long decode gates above. A multimodal candidate also requires a source-matched text/projector receipt, GPU vision execution, correct first-image semantics, nonzero retained-prefix reuse for an image following a text anchor, and a healthy `/readyz` result. |
| DeepSeek-V4 | Cached suffix spanning at least three native transactions with a live decode peer; middle-transaction cancellation and recovery; lopsided cold SSE progress with terminal parking; the four-agent cold/cached/tool gate twice using the immutable `full-context-agentic-v2` prompt contract and its `2c894c9e…b4ef` repository context, exactly 6,684 insertion-ordered prompt tokens per agent, explicit rejection of the 6,685-token legacy key-sorted rendering, zero cold reuse, and the literal 60-second cold bounds. The contract binds all request/render/token hashes, the historical 8,912-byte tool result, the exact 6,676-token recovery anchor, and the 2,798-token continuation suffix. The ceiling remains 9.2 seconds below the current thermally valid matched peer median. Each calibrated wave starts only after at least 60 seconds of Nominal, process-contention-free samples at five-second cadence with no hf2q/peer model runtime loaded, then remains under fail-closed two-second thermal and host-process sampling until all four atomic cold receipts exist. Four cold prefills may run in one bounded cohort; terminal cold unary lanes publish together, and only a warm 1–8-token recovery suffix may align four compatible decode cursors before cached work. Large tool-result suffixes remain interleavable. The same live caches must finish cached unary/SSE, automatic tool choice, and tool-result continuation under the unchanged 15/15/15/35-second bounds. Before those waves, a prebuilt exact-artifact test binary launched from a minimal clean-environment whitelist must pass B=2/3/4 non-aligned warm-prefix cooperative state/logit/subsequent-token parity and its alternating five-pair N=4 speed benchmark, plus the exact four-lane decode proof across at least 130 steps and ratio-four/ratio-128 boundaries. The sustained cooperative-prefill and decode microbenchmarks each still require their own 60-second Nominal, contention-free settle and a Nominal, contention-free first measurement, but may reach Fair under continuous two-second telemetry; Serious or Critical thermal samples or forbidden host work fail either gate. The decode proof must show bit-identical per-lane state, logits, cache, and recurrent data; 92-to-23 command-buffer and four-to-one synchronization topology; and a positive alternating-order median. Release independently rehashes the raw timing, test, thermal measurement/settle, and host-contention measurement/settle files, recomputes medians and speedups, and replays both validator families. Each wave's rehashed server log must contain positive post-publication warm-prefill transactions and exact warm B=4 decode selections; cold server completion and client publication must remain cohort-synchronized. The thermal receipt binds the four cold-receipt names and hashes; semantic/tool parity and retained-prefix counts remain unchanged. |
| Gemma 4 + Qwen35 family + DeepSeek-V4 | The generic fail-stop ownership test covers origin, installed, buffered, and pre-close-permitted replies; synthetic dead workers keep `/health` live while `/readyz` and new generation fail with 503. |

---

## Category 2 — Supported operator knobs

User-facing escape hatches. Stable in the contractual sense: we will
not remove or silently change them without an ADR.

| Var | Values | Purpose |
|---|---|---|
| `HF2Q_DEFAULT_THINKING_TOKEN_BUDGET` | non-negative integer, unset | Operator default for Qwen reasoning when a request omits `thinking_token_budget`; the qualified agentic profile uses 2,048 and is persisted into `config.toml` by `hf2q setup` (applied by `hf2q serve` only when this variable is absent). The handler still reserves answer capacity. `0` disables the default. Explicit request budgets take precedence. |
| `HF2Q_DEFAULT_TOOL_THINKING_TOKEN_BUDGET` | non-negative integer, unset | Operator ceiling for the first Qwen tool-result continuation; the qualified agentic profile uses 512 (persisted into `config.toml` by `hf2q setup`, applied only when this variable is absent) and deeper cycles reduce to a 256-token floor. `0` disables this override. |
| `HF2Q_QWEN_SPECULATION` | `off`, `auto` | Live Qwen SlotAware speculation policy. The qwen35 server engine defaults to `auto` when the variable is unset (since 2026-08-21; previously the default was off outside the canonical Qwen3.8 launcher). Auto preserves the target sampler/grammar state, requires coherent request-owned cache metadata, and cost-gates history lookup and fixed-K3 MTP independently. Unsupported semantics and runtime failures fail closed to ordinary decode or invalidate the affected slot; invalid values warn and resolve to off. Explicit `off` remains the escape hatch. |
| `HF2Q_DECODE_MVN` | `0`, `1` | Exact-tree Q4_K/Q6_K multi-column matvec routing. The global default is `1`; loading a Qwen3.8-identified model applies `0` at engine load (previously only via the canonical Qwen3.8 launcher) because its K=3 verifier is qualified on the weight-amortized width-four route. Explicit values always win. |
| `HF2Q_DECODE_MV_EXT` | `0`, `1` | Weight-amortized multi-column matvec routing. The global default is `0`; loading a Qwen3.8-identified model applies `1` at engine load (previously only via the canonical Qwen3.8 launcher). K-quants route only at widths 4–8; legacy Q4_0/Q8_0 route at widths 2–8. Unlike the byte-exact default-on mvN route, `mul_mv_ext` is not bit-exact, so the default remains Qwen3.8-scoped. |
| `HF2Q_QWEN_GQA_Q2` | `auto`/unset, `off`/`0`/`false`, `on`/`1`/`true` | Qwen3.8 long-context TQ-HB selector. Auto uses the bit-exact Q2 cooperative kernel only at KV length ≥8,192 and only for its hard D=256/GQA/no-mask geometry. Off is the supported escape hatch. On cannot bypass geometry checks. Invalid values fail safe to off. |
| `HF2Q_BATCHED_PREFILL` | `0`/`false`/`off`, unset | Opt out of the default batched prefill path (Category 1) back to per-token `forward_prefill`. For parity diagnostics only — per-token is 14-45× slower than peer. Default-on since ADR-028 iter-344; decoupled from the `HF2Q_UNSAFE_EXPERIMENTS` ack at that iter. The remaining `sliding_wrap` long-sequence byte-parity gap is the operator-signed deferral (2026-04-16; see ADR-010), a coherence deferral — not a runtime error. |
| `HF2Q_STREAMING_PHASE3` | `1`, unset | ADR-014 P7 iter-3 production wire-up. Routes all 4 Phase 3 quantize dispatch arms (K-quant codec direct / ImatrixAdaptive / StaticQuantizer / DwqK) and Phase 4.5 quality measurement through the streaming `LazyTensorMap` pipeline (`quantize_via_streaming_borrowed` + `measure_quality_streaming_lazy`). Output is byte-identical to the eager path — every wired arm has a per-arm byte-identity gate. Currently a TEST INTEGRATION channel, not a memory win (wedge clones bytes ~2× peak briefly); actual memory savings land when iter-3 wholesale surgery removes the upstream `materialize_all()` bridge. Default OFF; default behavior unchanged. |

---

## Category 3 — Benchmarking-only (user-triggerable but unsafe)

Documented knobs for controlled measurement. Activating them requires
an explicit acknowledgment: `HF2Q_UNSAFE_EXPERIMENTS=1`.

| Var | Unsafe-ack | Purpose |
|---|---|---|
| `HF2Q_CHUNK_SCAN_PREFILL=1` | **required** | Wave 5b iter 5 opt-in: route Qwen3.6 prefills at `seq_len > 64` through the mlx-native chunk-parallel delta-rule pipeline (`mlx_native::ops::chunk_gated_delta_rule::dispatch_chunk_gated_delta_rule_fwd`). This is a performance experiment distinct from the production autoregressive path. Decode parity ±5% (AC 5468) and walk-bar parity at pp4096+ (W-5b.3) are required before this experimental kernel can become Category 1. |

---

## Category 4 — Investigation-only (not part of product surface)

Internal scaffolding. Not listed in `docs/operator-env-vars.md` as
operator-facing; loaded through `src/debug/investigation_env.rs`
(centralized), not read ad-hoc in hot paths.

ADR-046's fixed-profile `source-teacher` characterization operator and closed
`source-teacher-acceptance-verify` command are hidden validation surfaces, not
public conversion, serving, or default behavior. Their minimal authenticated
calibration, exact-teacher, source-precision, and base-text-cache chain compiles
in release under ADR-048's reachability rule. The one-time AcceptanceHoldout
execution and comparison-minting routes were removed after their exact receipts
passed and were checked in; the retained verifier performs no Metal model load
and accepts no caller-provided receipt. Copied-execution evidence and trace
capture, exact Dynamic-frontier generation, selector/autoquant activation,
compatibility writers, and replay remain test-only or unavailable.

**Ack-required (known to risk correctness or runtime reliability):**

| Var | Notes |
|---|---|
| `HF2Q_F16_KV` | Known-worse KV cache representation; separate bug vs F32 path. |
| `HF2Q_SKIP_TQ_ENCODE` | Bisection scaffolding; produces garbage output. |
| `HF2Q_SKIP_TQ_SDPA` | Bisection scaffolding; produces garbage output. |
| `HF2Q_TEST_QWEN_POST_ADMISSION_PREFILL_FAILURE_MAX_TOKENS` | Positive integer selecting one Qwen SlotAware request for the ADR-049 hardware gate. After that request completes a non-empty GPU prefill slice, a one-shot request failure is injected before scheduler publication so the real reset/AnchorStore invalidation lifecycle can be proven. Requires `HF2Q_UNSAFE_EXPERIMENTS=1`; never active in ordinary serving. |

**Warn-on-activation, no ack (ineffective but safe):**

| Var | Notes |
|---|---|
| `HF2Q_GRAPH_OPT` | No measured win; reorder aborts on unannotated dispatches. |
| `HF2Q_DUAL_BUFFER` | Internal perf tuning; default (3) is part of category 1. |

**Silent / read-only diagnostics (no warning, no ack):**

| Var | Notes |
|---|---|
| `HF2Q_PREFILL_DUMP`, `HF2Q_BATCHED_DUMP`, `HF2Q_BATCHED_LAYER_SCAN`, `HF2Q_DUMP_LAYERS`, `HF2Q_DUMP_BOUNDARY`, `HF2Q_DUMP_ALL_CACHE`, `HF2Q_DUMP_LAYER_DETAIL`, `HF2Q_DUMP_NORM_WEIGHT`, `HF2Q_DUMP_DIR` | Hidden-state / cache dumps; output-only, cannot affect decode. |
| `HF2Q_DUMP_RENDERED_PROMPT`, `HF2Q_DUMP_PROMPT_TOKENS` | Prompt-path diagnostics. |
| `HF2Q_MLX_TIMING`, `HF2Q_SPLIT_TIMING`, `HF2Q_MLX_KERNEL_PROFILE`, `HF2Q_MLX_PROFILE` | Timing / kernel-attribution diagnostics. |

---

## Classification rule

A toggle requires the `HF2Q_UNSAFE_EXPERIMENTS=1` acknowledgment when
it is **known to risk correctness or runtime reliability** — not
merely because it is experimental or inert. Toggles that are
ineffective-but-safe get a startup warning, not a gate. Toggles that
are read-only diagnostics get neither.

When a new toggle is introduced, classify it by this rule and register
it in `src/debug/investigation_env.rs` (for category 4) or update this
document (for categories 2–3).

---

---

## Category 1 appendix — Qwen conversion acceptance (ADR-012 foundation)

ADR-012 originally accepted `qwen35` (dense 27B) and `qwen35moe` (MoE
35B) as convert-only classes. That historical conversion contract remains
authoritative for emitted artifacts. ADR-013, ADR-027, and ADR-040 now own the
shipped inference, cache, and SlotAware serving contracts; Qwen is no longer
convert-only.

### Acceptance gates for a converted GGUF

| Gate | Criterion |
|---|---|
| Structural validity | File begins with magic `GGUF`, version 3, tensor_count > 0, kv_count > 0 |
| Metadata completeness | Every key in the ADR-012 Decision 7 catalog is present |
| Tensor naming | Every tensor name matches the ADR-012 Decision 8 naming spec |
| Peer load | `llama-cli --model out.gguf -p "Hello" -n 8` exits 0 |
| Sidecar set | `tokenizer.json`, `tokenizer_config.json`, `config.json`, `generation_config.json`, `special_tokens_map.json` (and `chat_template.jinja` when present) are byte-identical copies alongside the GGUF |
| MTP tensors (when `mtp_num_hidden_layers > 0`) | Round-trip integrity gate at `tests/convert_qwen35_mtp_roundtrip.rs` (Decision 19); 4 tensors land at `blk.{num_hidden_layers}.nextn.{enorm,hnorm,embed_tokens,eh_proj}.weight` |
| mmproj (when `--emit-vision-tower` and `vision_config` present) | Pure-Rust emitter at `src/models/vit/`; produces `mmproj-<slug>-F16.gguf` per Decision 18 with three layers of structural / round-trip / spec-driven test coverage |
| Smoke harness | `hf2q smoke --arch <qwen35\|qwen35moe> --quant q4_0` exits 0 with byte-identical transcripts across two fresh runs (Decision 16) |

### DWQ activation-based quantization for qwen35/qwen35moe

**Shipped 2026-04-25** under ADR-012 P9 + P9b (formerly listed as
"out-of-scope" pending ADR-013 P12). The convert pipeline now runs
the full two-pass activation calibration end-to-end:

  1. Emit intermediate F16 GGUF from the in-memory tensor_map
     (`backends::gguf::emit_gguf_from_tensor_map`, P9b.1).
  2. Construct `RealActivationCapture::new(intermediate_gguf, tokenizer)`
     which loads via the ADR-013 `Qwen35Model::load_from_gguf` path
     (P9b.3b).
  3. Run `quantize::dwq_activation::run_dwq_activation_calibration`
     which generates calibration tokens, runs the CPU forward pass
     through the loaded model, computes per-layer sensitivity, and
     produces a derived `MixedBitQuantizer` configured with
     activation-driven sensitive layers (P9b.3a).
  4. Final GGUF is emitted at the user-specified output path. The
     intermediate is dropped via `tempfile::TempDir` RAII (P9b.5).

No weight-space fallback for these architectures (Decision 13).

Real-model artifact production for the four end-deliverable GGUFs
(qwen35/qwen35moe × dwq46/dwq48) is gated only on environment
(HF_TOKEN + ~150 GB disk + Metal-validated peer build).

### Out-of-scope for ADR-012

- Inference coherence (sourdough gate, sliding-window parity) — ADR-013.
- MTP head **inference** (speculative decoding) — ADR-013 P14. ADR-012 P11
  ships the conversion-side tensor round-trip integrity gate; runtime
  draft/accept loops are owned by ADR-013.
- ViT compute path for the converted mmproj — ADR-005 phase 2c. ADR-012 P10
  ships the GGUF emitter; forward-pass execution is ADR-005's deliverable.

### CI integration tests

`tests/convert_qwen35_integration.rs` and
`tests/convert_qwen35moe_integration.rs` run the full convert pipeline on
synthetic tiny models (4 layers, hidden=64, 4 experts) to validate structural
correctness and sidecar behavior without downloading real model weights.

---

## Category 1 appendix — Peer-parity gates (ADR-014 P10)

ADR-014 P10 lands the **8-cell peer-parity benchmark harness**
(`tests/peer_parity_gates.rs`) that compares hf2q's streaming convert
pipeline against `llama.cpp` and `mlx-lm` across the matrix locked in
ADR-014 Decision 15. Closure of ADR-014 (and final P12 doc-refresh)
gates on every cell measuring green.

### The 8-cell matrix

`GateCell { model_id, backend, calibrator_variant, peer_id,
speed_tolerance, rss_tolerance, ppl_tolerance }`, populated **verbatim**
from Decision 15 lines 575–582. The `gate_cells_match_decision_15_verbatim`
smoke test wedges this against a duplicate literal table to catch silent
edits.

| # | Model | Backend | Calibrator | Peer | Speed gate | RSS gate | PPL gate |
|---|---|---|---|---|---|---|---|
| 1 | 27B dense | GGUF | None (`q4_k_m`) | llama.cpp uncalibrated Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 2 | 27B dense | GGUF | Imatrix (`imatrix-q4_k_m`) | llama.cpp imatrix Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 3 | 27B dense | safetensors | DWQ (`dwq-4-6`) | mlx-lm DWQ | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 4 | 27B dense | GGUF | DWQ (`dwq-4-6`) | (no peer; vs hf2q current pipeline) | ≤ 1.0× | ≤ 0.50× | ≤ 1.0× |
| 5 | apex MoE | GGUF | None (`q4_k_m`) | llama.cpp uncalibrated Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 6 | apex MoE | GGUF | Imatrix (`imatrix-q4_k_m`) | llama.cpp imatrix Q4_K_M | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 7 | apex MoE | safetensors | DWQ (`dwq-4-6`) | mlx-lm DWQ | ≤ 1.10× | ≤ 1.10× | ≤ 1.02× |
| 8 | apex MoE | GGUF | DWQ (`dwq-4-6`) | (no peer; vs hf2q current pipeline) | ≤ 1.0× | ≤ 0.50× | ≤ 1.0× |

**Tolerance triple semantics:**

- **Speed**: `hf2q_wall ≤ tolerance × peer_wall`. The 1.10× headroom on
  rows 1–3 and 5–7 is the documented "no-regression-vs-peer" budget for
  ADR-014's streaming pipeline.
- **RSS**: `hf2q_peak_rss ≤ tolerance × peer_peak_rss`. Rows 4 and 8
  encode the central correctness/sanity claim of the ADR — streaming
  halves peak resident vs the pre-streaming pipeline (≤ 0.50×).
- **PPL**: `hf2q_ppl ≤ tolerance × peer_ppl`. Wikitext-2 perplexity at
  1.02× headroom on the cross-peer rows; rows 4 and 8 require strict
  PPL parity (≤ 1.0×) against the hf2q current pipeline.

### Verdict surface

```rust
pub enum Verdict {
    Pass,
    Fail { reason: String },
    NotMeasured { reason: String },
}
```

`NotMeasured` is the canonical outcome for cells that cannot run yet
(no real model on disk, peer binary missing, hf2q-side driver pending).
**Distinct from `Pass` and `Fail` so the markdown table surfaces the
deferred state honestly** — no fake-green, no fake-red. The
`reason` field carries the disqualifier so the table reader sees *why*
the cell was deferred.

### Markdown table emitter

`emit_markdown_table(results, hardware_fingerprint, sha) -> String` is a
**pure function** (no I/O) that produces the full markdown document for
a slice of `CellResult`s. Header columns (14 total):

```
Model | Backend | Calibrator | Peer | hf2q wall (s) | peer wall (s) |
speed ratio | hf2q RSS (B) | peer RSS (B) | RSS ratio | Verdict |
hf2q PPL | peer PPL | PPL ratio
```

PPL cells render `f32` to 4 decimal places; un-measured PPLs render as
the em-dash `—` so the deferred state is visually distinct from a real
`0.0000`.

`write_results_to_dated_doc(results, hardware, sha, today, docs_dir)`
writes the table to `docs/peer-parity-results-<YYYY-MM-DD>.md`. The
write is only callable from `#[ignore]`-gated cells so the always-on
test suite does not pollute `docs/`.

### Smoke-vs-full corpus auto-pick (P10 iter-3)

The PPL corpus loader (`load_corpus_tokens` in
`tests/peer_parity_gates.rs`) **auto-picks** between two corpora:

| Corpus | File | Size | When used |
|---|---|---|---|
| **Smoke** | `tests/fixtures/ppl-corpus/wikitext2-smoke.tokens` | 2 KB (512 little-endian u32 tokens, deterministic ramp `(i*17+3) % 32000`) | Committed to the repo; used when the full corpus is missing or fails validation. Default for every `cargo test` run. |
| **Full** | `tests/fixtures/ppl-corpus/wikitext2-full.tokens` | ~700 KB+ on disk (≥ 280 000 tokens) | Generated by `scripts/fetch_wikitext2.sh` (Stephen Merity / Salesforce wikitext-2 raw v1; SHA-256 locked). Gitignored. Used by P11 for parity-grade PPL. |

The fetcher refuses corrupt downloads (SHA mismatch) and undersized
output (`< 280 000` tokens or `< 1 MiB`). The loader logs the selected
corpus to stderr so a CI run that silently falls back to smoke is
diagnosable from the build log alone.

### P11 hardware gate

All 8 cells are `#[ignore]`-gated as of P10 with reasons like:

```
#[ignore = "P11 hardware gate: needs apex MoE GPU + ~150GB disk + Qwen35Model::load_from_gguf for Variant::Moe"]
```

P11 swaps the sentinel `/var/empty/...gguf` paths in
`hf2q_model_path(&cell)` for real model artefacts staged on disk and
runs the harness via `scripts/peer_parity_run.sh` (1 warmup discarded
→ 60 s thermal cooldown → 3 timed runs each wrapped in
`/usr/bin/time -l`; the harness reads the median of 3).

### Calibrator cross-validation gate (P6 close iter-1)

Independent of the speed/RSS/PPL parity matrix, ADR-014 P6 ships a
**byte-equivalent (with documented float tolerance) cross-validation
gate** for the imatrix calibrator:
`src/calibrate/imatrix_xvalidate.rs::cross_validate_imatrix_gguf`.

Given two GGUF v3 imatrix files (the schema landed by the peer's
PR #9400 / commit `90083283` / 2025-07-19), the comparator:

1. Loads both via `ImatrixCollector::load_imatrix_gguf` — the same
   reader the runtime quantize loop uses (load-bearing: a regression
   in the loader breaks both production and the gate, not just one).
2. Diffs the tensor-name set; tensors present in only one side surface
   as `tensors_in_a_only` / `tensors_in_b_only`.
3. For every shared tensor, computes element-wise
   `max(abs(a - b))` and `max(abs(a - b) / max(abs(a), abs(b), 1e-12))`
   over `in_sum2` (the importance vector).
4. Asserts `counts` arrays are byte-equal (counts are exact integer
   token counts; no float-precision leeway).
5. Returns an `XValidationReport` with `is_pass()` predicate.

**Tolerance defaults:** `abs_tolerance = 1e-3`, `rel_tolerance = 1e-2`.
Justification (P7 iter-3x/3y dequant round-trip RMSE bounds): Q4_K
≤ 0.05, Q5_K ≤ 0.025, Q6_K ≤ 0.012 — the gate would still catch a
Q6_K-level regression even if the imatrix port introduced noise at the
Q4_K-precision level. Callers wanting tighter or looser tolerances
pass them explicitly.

The `#[ignore]`-gated cell `xvalidation_vs_llama_imatrix_qwen35_smoke`
(at `tests/imatrix_xvalidation.rs`) wires this comparator against the
external `llama-imatrix` binary on a Qwen3.5-0.6B fixture; that cell
is the pre-P11 close gate proving hf2q's pure-Rust port produces
per-tensor `in_sum2` + `counts` numerically equivalent to the peer's
C++ implementation.

---

## Known out-of-scope

These are deliberately not part of any category:

- Byte-identical batched-prefill parity with the peer at the ~752-byte
  `sliding_wrap` level (see `docs/adr/ADR-010-exact-batched-kernel-parity.md`;
  deferred).
- Standalone Qwen3-VL generation and serving, pending the ADR-041 engine seam.
  Dense conversion is supported, but server startup and CLI generation fail
  closed before weights load; the Qwen3-VL MoE variant is also unsupported.
- Qwen multimodal artifacts or request geometries that have not passed the
  source-pair binding and family-specific soft-token/DeepStack/3D-position
  validation gates. The accepted Qwen3.8 pair does not qualify arbitrary
  Qwen projectors or standalone Qwen3-VL.
- In-process recovery after a fatal Metal command-buffer/watchdog/ignored-
  submission failure or an expired non-returning transaction. The worker and
  HTTP surfaces fail closed, but an OS supervisor must recreate the
  process/device generation.

---

## References

- `docs/operator-env-vars.md` — per-variable effects and defaults.
- `docs/adr/ADR-004-gguf-compatibility.md` — source-bound Qwen3.8 automatic
  pair and first-image cache acceptance evidence.
- `docs/adr/ADR-009-reference-parity-and-coherence-recovery.md` — the
  historical F32-KV and per-token prefill baselines; current family defaults
  are defined above and in the family-specific ADRs.
- `docs/adr/ADR-010-exact-batched-kernel-parity.md` — why batched-prefill
  is now the default and why its `sliding_wrap` byte-parity is deferred.
- `docs/adr/diary/ADR-028-peer-parity-coherence-and-speed.md` — iter-344
  default-flip of batched prefill and ack-decoupling.
- `docs/adr/diary/ADR-012-qwen35moe-conversion.md` — qwen35/qwen35moe convert spec.
- `docs/adr/diary/ADR-013-qwen35-inference.md` — Qwen35 inference graph.
- `docs/adr/ADR-027-qwen35-tq-kv-cache-and-persist-family.md` — Qwen cache
  and persisted-family contract.
- `docs/adr/ADR-040-continuous-batching-reopen.md` — Qwen SlotAware bounded
  text/multimodal prefill and cross-family fatal-device ownership.
- `docs/adr/ADR-041-qwen3vl-text-lm-engine-seam.md` — standalone Qwen3-VL
  runtime blocker and fail-closed boundary.
- `docs/adr/diary/ADR-014-streaming-convert-pipeline.md` — streaming pipeline +
  Decision-15 peer-parity gate matrix (the source of truth for the
  8-cell table above).
- `docs/converting-qwen35.md` — canonical convert commands for Qwen3.5/3.6.
- `docs/converting-a-model.md` — generic convert reference including Gemma.
- `docs/calibrator-onboarding.md` — developer guide for adding new
  Calibrator implementations (Imatrix, DWQ, future ones).
- `scripts/release-check.sh` — the reproducible gate runner.
- `scripts/peer_parity_run.sh` — cold-cache protocol for the peer-parity
  harness (1 warmup discarded → 60 s thermal cooldown → 3 timed runs;
  median read).
- `scripts/fetch_wikitext2.sh` — full wikitext-2 corpus fetcher for the
  PPL gate (smoke fixture is committed; full corpus is gitignored).
