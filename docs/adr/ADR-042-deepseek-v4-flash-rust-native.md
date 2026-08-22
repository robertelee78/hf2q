# ADR-042: DeepSeek-V4-Flash-0731 — Rust-native source conversion and MLX inference

> Terminology: "the peer" = llama.cpp, the pinned upstream GGUF engine (see NOTICE, data/llama_cpp_pin.txt).

- **Status:** Accepted for hf2q 0.1.2; full-context four-agent serving and
  cache growth revalidated 2026-08-08
- **Updated:** 2026-08-22 — the four-agent workload is bound to an immutable
  insertion-ordered prompt contract and historical tool-result payload;
  `mlx-native =0.10.12` fixes non-aligned D512 tail loads; warm compatible
  suffixes cooperate through FFN/MoE; and an exact four-lane decode transaction
  plus synchronized cold handoff restores every unchanged product latency
  bound. Calibrated gates compile their Foundation thermal probe once and then
  reuse it for every sample, preserving the strict cadence contract without a
  Swift compiler launch in the measurement loop. Earlier real-model
  structured-tool, prefix-reuse, scheduling, and thermal evidence remains as
  recorded below. DeepSeek single-tool completion now also honors the existing
  non-parallel default end to end and stops when its constrained grammar is
  accepted instead of evaluating an unused final token. The 2026-08-20 hf2q
  candidate requires exact-SHA CI and
  protected packed-artifact hardware receipts before publication. The B=4
  decode proof now holds the already-loaded producer at its acknowledged
  readiness barrier until setup telemetry proves a continuous 30-second
  nominal thermal tail, with a 240-second runner deadline inside the producer's
  300-second fail-closed acknowledgement timeout.
- **Owner:** hf2q integration lane
- **Source model:** `deepseek-ai/DeepSeek-V4-Flash-0731`
- **Pinned source revision:** `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- **Reference implementation:** `/opt/llama.cpp` at
  `3653e6d6d547ec763317d9ecd0ace334a7e21359` (build 10326)
- **Target host:** Apple M5 Max, 40-core GPU, 128 GiB unified memory

## Decision

hf2q downloads the **official Hugging Face checkpoint** and converts it to the
owned `deepseek4-agentic-q2` mixed profile. Expert gate/up tensors use Q2_K,
expert and shared down tensors use Q3_K, and the token/output, HC, indexer, and
compressed-attention context paths use Q8_0. Conversion, quantization, loading,
and inference are implemented in-process in Rust with owned Metal kernels in
`mlx-native`.

The product may invoke `hf`, `wget`, or `curl` solely to fetch official source
repository files. It must not invoke Python, the peer, or another converter,
quantizer, or inference runtime. A separate developer-only parity harness may
invoke the pinned peer build as an oracle.

Prebuilt quantized weights are not an input, fallback, cache seed, or release
artifact. Their published sizes may be used only as non-authoritative capacity
evidence.

That conversion/release boundary does not restrict the server to hf2q-produced
files. `hf2q serve` does not require an hf2q provenance receipt and may load an
externally produced or downloaded GGUF when its architecture, tensor catalog,
quantization types, tokenizer, and template are explicitly supported. Unknown
or incompatible model-family layouts still fail closed; producer identity by
itself is never a serving rejection reason.

The accepted runtime executes prompts in bounded 2,048-token matrix
transactions, 16 times wider than the 128-token physical attention
window. Attention reads each transaction's compact KV source while only the
final physical window is published to the circular cache. A suffix shorter than
33 tokens retains exact token-wise extension; longer suffixes use another
matrix transaction so an agentic follow-up does not degrade into full
decode-style replay. The 2K bound is measured: 4K prefill exceeded the Metal
command-buffer working-set limit beside the 100.05 GiB artifact, while 2K
preserved matrix prefill without OOM. The OpenAI server context is configured
with `HF2Q_DEEPSEEK_MAX_SEQ_LEN` and capped by checkpoint metadata. The separate
DSpark draft artifact remains follow-up work and is never silently represented
as part of the base GGUF.

Agentic serving is stateful. A growing rendered transcript reuses the exact
live native KV/recurrent prefix and evaluates only its suffix. Because the
official thinking encoder removes old hidden reasoning on later turns, hf2q
also retains one native recovery checkpoint eight tokens before the last prompt
boundary. A canonicalized transcript restores that checkpoint and replays only
the rewritten tail. Divergent edits or context compaction reset safely. None of
these paths invokes an external converter, runtime, model, or process.

The server supports OpenAI-compatible unary and SSE chat completions,
cancellation, sampling, stop/EOS, reasoning deltas, official DeepSeek DSML,
required/automatic tool choice, structured tool-call responses, and multiple
invocations in one DSML block. Slot-aware serving retains one independent
full-logical-context session per agent while sharing weights and one physical
KV byte budget. DeepSeek-V4 embeddings and multimodal inputs remain explicitly
rejected rather than routed to another model family.

For SSE, `time_to_first_token_ms` measures the first API-visible reasoning,
content, or structured tool-call event. Prefill completion alone is not TTFT:
DSML text is intentionally withheld until a complete tool block can be parsed,
so the metric includes that silent decode interval.

Serving stays quiet at the default warning log filter. With `-v`, the native
worker reports a content-free request identifier, cache action, prompt/cached/
suffix counts, throttled prefill and decode progress, first semantic stream
event, and terminal timing or error. The canonical DeepSeek OpenCode launcher
enables `-v` so a long cold prefill or buffered DSML tool call is visibly
progressing without exposing prompt text, decoded text, or tool arguments.
Prefill progress reports both the request-wide average and the latest interval
rate; the latter exposes context-position cost instead of making every later
sample look like an unexplained cumulative slowdown. API prefill throughput
counts only uncached tokens, so an exact cache hit reports zero prefill work
rather than dividing the entire prompt by a near-zero cache lookup duration.

Completed prefill scratch has a request-bounded lifetime. Prefill and decode
use separate `mlx-native` buffer arenas; once prompt logits and transactional
KV state are committed, hf2q clears the prefill arena before decode. Successful
requests may retain at most 256 MiB of decode scratch, while cancellation or an
error clears both arenas. Logical KV/recurrent cache state, recovery snapshots,
weights, and prompt logits are not allocated from either transient arena.

## Memory-pressure incident and correction (2026-08-06)

The earlier memory-safety acceptance statement was falsified by a real OpenCode
session. At `2026-08-06 10:06:41.847` macOS logged `triggering no paging space
action` followed by `killing largest compressed process hf2q [74158] 108840
MB`. The shell's unqualified `zsh: killed` was therefore a kernel SIGKILL, not
an hf2q inference error, an operator interrupt, or an agent-issued signal.

Source inspection localized the retained footprint to the transient Metal
arena rather than the logical DeepSeek cache. `MlxBufferPool::reset()` moves
in-use buffers into power-of-two free buckets for reuse; it does not release
those buckets. DeepSeek previously shared one thread-local arena between
matrix prefill and decode, so successively larger context-dependent score,
top-k, attention, and activation workspaces from the cold prompt remained
resident through every later cached agent turn. Small 15-1,030-token suffixes
therefore reused KV correctly but could not return the multi-gigabyte cold-
prefill high-water to macOS.

The corrected release build produced the following exact evidence on the same
M5 Max and 107,431,343,104-byte artifact:

| Gate | Result |
|---|---|
| Startup warmup | Released 106,832,652 transient bytes before readiness. |
| 32,777-token cold prompt | 90.393 s, 362.60 tok/s request average; interval rate declined from 526.92 tok/s at 4K to 269.58 tok/s at 32K as ratio-four history grew; released 3,990,399,216 prefill bytes before decode. |
| 98,009-token continuation | Reused 32,769 tokens, evaluated 65,240 in 292.049 s (223.39 tok/s), released 4,275,579,120 prefill bytes, and survived. |
| Exact 98,009-token repeat | Reported all 98,009 tokens cached and returned `OK` in 0.798 s, proving scratch release did not discard logical cache state. |
| Process/host memory | 107 GB peak, 102 GB after requests, 94% system free, and no increase in swapouts during the 98K run. |
| Agentic semantics after the long run | `scripts/test_deepseek4_agentic.sh` passed required/automatic tools, unary/SSE, exact Rust source arguments, tool-result continuation, and 6,252/6,260 prefix reuse; cached TTFT was 259 ms. |

The context-position throughput slope is not itself an arena leak. Ratio-four
index selection must score an expanding compressed history, and the pinned
peer graph also scans the valid lightning-indexer history. The new interval
metric makes that workload visible. A later matched reference and optimized
native run are recorded below and supersede the earlier H4 qualification.
The reviewed phase-adaptive DSpark bundle is a CUDA/vLLM speculative-decoding
patch; the accepted base artifact explicitly excludes DSpark tensors, so that
work cannot correct this prefill-arena lifetime defect and remains a separate
future optimization.

## Long-context prefill root cause and correction (2026-08-06)

Stage timing falsified the hypothesis that model arithmetic or the lightning
indexer was the primary remaining gap. The sparse D=512 adapter represented
each original token as a Metal batch with `qL=1` and 64 heads. The inherited
flash tile is physically eight query rows wide, so seven of eight query rows
were idle for every head and token. The accepted `mlx-native` dispatcher views
the unchanged contiguous `[tokens, 64, 512]` storage as
`[tokens, 8, 8, 512]`: eight physical heads become the eight query rows, eight
logical heads cover all 64 physical heads, and no transpose or copy is added.
The mask remains token-specific and broadcasts across those rows; learned
attention sinks remain physical-head-specific.

A strict A/B test with 16 query tokens, 64 distinct sinks, and a mixed mask is
bit-exact against the former one-head-per-tile path. Six existing sparse-
attention parity cases also pass. The production-shape benchmark measured the
following packed-attention speedups:

| Queries | Former tile | Heads-as-rows tile | Speedup |
|---:|---:|---:|---:|
| 64 | 4.991 ms | 0.827 ms | 6.03x |
| 128 | 9.708 ms | 1.460 ms | 6.65x |
| 256 | 19.212 ms | 2.643 ms | 7.27x |
| 512 | 38.087 ms | 5.037 ms | 7.56x |
| 1,024 | 75.910 ms | 9.760 ms | 7.78x |

The speedup changed the dense/sparse crossover. Dense compressed attention was
about tied at 2K raw tokens but reached 0.98 s per transaction at 4K, while
the packed sparse path was about 0.69 s. hf2q therefore selects sparse prefill
at 1,024 compressed entries rather than the former 6,144-entry threshold. A
Q=1/one-simdgroup variant, an F16 path with conversion kernels, and temporary
index-overlap instrumentation were rejected after their spikes and are absent
from the production diff.

The other memory copy was architectural, not model policy. `mlx-native` now
owns read-only GGUF mmap-backed Metal resources, typed tensor views, nested
view offsets, and strict mapped-versus-owned kernel parity. hf2q chooses that
mechanism for immutable DeepSeek raw matrices and reports the logical split;
`HF2Q_DEEPSEEK_MMAP_WEIGHTS=0` is the diagnostic rollback. On this artifact,
107,422,652,416 weight bytes were file-backed, 7,518,556 bytes remained
anonymous, and two Metal resources covered the file because of the device's
per-buffer limit. Weight residency setup fell from about 40 seconds to
0.85-0.98 seconds; the complete post-reboot server load, including the
remaining model setup, was 10.79 seconds. After the 120K gate, process RSS was
2.0 GiB and the server shut down normally.

## Logical-view snapshot correction (2026-08-06)

The first agentic gate against the published `mlx-native` 0.10.2 release
falsified cache compatibility with
`WindowKv cache snapshot shape or dtype does not match`. Version 0.10.2
correctly made `MlxBuffer::as_slice` and `as_mut_slice` honor a view's logical
`data_byte_len` and `byte_offset`. DeepSeek's snapshot helper still allocated
and compared `byte_len`, which is the parent Metal allocation length. A
128-row `window_kv` view over a context-linear `attention_kv` buffer therefore
tried to snapshot the entire backing allocation while copying only the logical
view.

Snapshots, restores, and their resident-byte accounting now use
`data_byte_len`. The focused regression asserts the compact snapshot equals
the sum of the cache plan's logical snapshot buffers, restores overwritten
window state and position, and preserves non-aliasing. All nine DeepSeek cache
tests pass. The post-correction real gate processed a 6,262-token cold prompt
at 505.81 tok/s, reused 6,254 tokens on the next turn, reduced cached TTFT to
234.6 ms, decoded at about 32.3 tok/s, and passed required/automatic tools,
source-code arguments, tool-result continuation, unary, and SSE assertions.

## Context and spike result

The official repository is approximately 166.9 GB (48 safetensor shards) and
contains about 304.18 billion logical parameters. About 284.34B belong to the
target model and 19.85B to the attached DSpark namespaces. A four-bit artifact is
larger than host memory before scales, higher-precision tensors, KV state,
scratch space, Metal allocations, and macOS. The existing hf2q `Q2_K_S` policy
is therefore the first feasible target; the reference artifact size is about
98.6 GB decimal (91.8 GiB).

The checkpoint is not an ordinary BF16 model:

- dense weights use FP8 E4M3 with E8M0/UE8M0 block scales;
- expert weights pack two E2M1 FP4 values in each I8 byte with E8M0 scales;
- the first three MoE layers use integer token-hash routing tables;
- 43 layers combine 256 routed experts (top 6) and one shared expert;
- the graph uses compressed sparse attention, a learned indexer, attention
  sinks, four-stream hyperconnections, clamped SwiGLU, and YaRN;
- the repository includes one attached DSpark next-token prediction stage made
  of three `mtp.{0,1,2}` blocks (4,705 checkpoint tensors).

At spike time, the converter rejected the source dtypes and architecture, expert
fusion created a multi-gigabyte F32 aggregate, and the runtime had no
DeepSeek-V4 graph or cache. The accepted implementation now converts the pinned
source, executes every verifier layer and the vocabulary head, performs batched
prefill, maintains the compressed cache transactionally, and generates coherent
greedy text entirely through the owned Rust/Metal path.

## Hypotheses and falsifiers

### H1 — exact source decoding

For every official source block, Rust E4M3 + E8M0 and packed E2M1 + E8M0
decoding produces the same F32 bit pattern as the pinned reference formulas.

**Falsifier:** any exhaustive E2M1 codebook mismatch, E8M0 exponent mismatch,
block-scale indexing mismatch, or accepted malformed dtype/shape.

### H2 — bounded streaming conversion

One tensor or bounded row group can be decoded and requantized without a full
model copy or an 8 GiB fused-expert F32 allocation.

**Falsifier:** conversion private RSS exceeds 20 GiB, conversion requires all
experts resident, an interrupted run replaces the requested output with a
partial tensor stream, or source/output identity is not bound to a receipt.

### H3 — the agentic mixed profile fits and remains coherent

The produced main-model artifact is at most 108 GB decimal (100.6 GiB) and loads
with enough headroom for Metal state and a 131,072-token live cache plus recovery
checkpoint on the 128 GiB host.

**Falsifier:** peak process footprint exceeds 116 GiB at the acceptance context,
macOS enters critical memory pressure, swap grows materially during steady-state
decode, or deterministic prompts become incoherent relative to the reference.

### H4 — owned low-bit kernels meet or exceed the peer

The Rust/Metal Q2_K path matches a scalar Rust decoder within the declared
numeric tolerance, and the complete runtime reaches at least 1.00x the pinned
peer decode and prompt-processing rates under the same artifact, prompt,
context, threading, and sampling settings.

**Falsifier:** decoded blocks differ, token decisions diverge outside documented
floating-point ties, or three-run medians miss either throughput floor after
profiling and measured optimization.

### H5 — architecture state is exact

Compressed attention, index selection, hash routing, hyperconnections, and cache
mutation reproduce the pinned reference on small deterministic fixtures and on
incremental-vs-one-shot real-model prompts.

**Falsifier:** layer checkpoints exceed tolerance, cache rewind/fork changes
tokens, slot interleaving perturbs a peer, or unknown architecture metadata can
fall through to another model family.

## Architecture contract

### Source ingest

The source reader gains typed views for packed I8/U8, integer routing tensors,
E8M0 scales, and FP8 payloads. Dtype, rank, logical shape, scale shape, block
size, and source revision are validated before allocation. Missing or ambiguous
scale siblings are typed errors.

The converter decodes and requantizes bounded rows directly into the incremental
GGUF writer. Expert fusion must preserve canonical expert-major layout while
streaming; it must not build a complete F32 fused projection. The writer uses a
same-directory temporary, finalizes and syncs the complete GGUF, and only then
atomically replaces the requested output. An interruption before promotion
therefore preserves the previous complete artifact. If receipt promotion fails
after artifact promotion, the new GGUF remains complete but hf2q removes any
stale sidecar, returns an error, and does not treat the result as provenance-
complete. Restart deterministically regenerates from the pinned source; this
implementation does not claim per-tensor resume.
Before artifact promotion, hf2q prepares and durably syncs a sidecar receipt that
binds source revision and file hashes, converter commit, quant selector, final
output size and checksum, excluded DSpark count, and peak streaming bounds. The
GGUF and sidecar are each promoted with a same-filesystem atomic rename. Artifact
acceptance independently verifies the recorded output hash; the two renames are
not claimed as one filesystem transaction.

### GGUF identity

The architecture string is `deepseek4`. Metadata and tensor names follow the
pinned peer registry, including:

- q/o low-rank projection parameters and output groups;
- compressor ratios and rotary base;
- indexer head count, key length, and top-k;
- hyperconnection stream count, Sinkhorn iterations, and epsilon;
- routed/shared expert counts, top-k, scoring, scaling, and normalization;
- hash-layer count, sliding window, attention sinks, and YaRN settings.

MTP/DSpark tensors are never silently dropped. The main model may be emitted and
accepted before the optional draft artifact only when the receipt explicitly
marks DSpark as a separate pending artifact and the parity benchmark disables
draft decoding on both runtimes.

### Runtime ownership

hf2q owns model dispatch, tokenization/chat encoding, generation, cache policy,
sampling, tool-call framing, and server behavior. `mlx-native` owns buffer
management plus dense, low-bit, routing, hyperconnection, compressor, indexer,
and sparse-attention Metal operations. No runtime branch may default an unknown
architecture to Gemma or Qwen.

The implementation sequence is:

1. exact router and clamped SwiGLU;
2. hyperconnection/Sinkhorn;
3. learned compression and indexer;
4. sparse top-k attention with sinks and its dedicated cache;
5. full block/model integration;
6. DSpark draft integration after base-model parity.

Dense D512 attention is allowed only as a small-fixture correctness oracle. It is
not the production path.

The official `encoding/` implementation, not an older DeepSeek chat template,
defines prompt behavior. The Rust encoder must pin BOS/EOS, user/assistant,
thinking, DSML, and DSpark-noise token IDs from the source manifest and reproduce
0731 system/chat/thinking/tool-call framing. A crafted 0731 template from the
pinned reference may be ported as an owned asset, but it is not executed by or
loaded from the peer at runtime.

### Agentic serving and cache contract

The SerialFifo DeepSeek worker owns one live session. The slot-aware worker owns
one such session per configured agent and executes them through the same model
surface. Each records the exact rendered token sequence corresponding to the
native cache position and selects the longest safe prefix for each request:

1. reuse the live cache when the new transcript extends it exactly;
2. restore the prompt-tail recovery checkpoint when canonical reasoning removal
   changes the recent suffix;
3. reset and replay when neither token prefix matches.

A short suffix of at most 32 tokens always keeps a valid prefix. For a larger
suffix, a prefix shorter than the 128-token native window is not treated as a
useful cache hit: the worker resets and uses full matrix prefill. This prevents
a shared BOS/role prefix from forcing hundreds of tokens through the slower
incremental path while retaining fast normal tool-result and follow-up turns.

Cache mutation is transactional and partial-token failure poisons the live
state until reset or checkpoint restoration. Decode is capped before the fixed
context boundary, so an oversized `max_tokens` request terminates with
`finish_reason: "length"` instead of writing out of bounds.

For a context capacity `C` divisible by 128, one native cache allocation is
`17,842,176 + 6,880*C` bytes. The canonical OpenCode launcher now requests
`C=524,288`: a fully grown live allocation is 3,624,943,616 bytes. Serving
capacity and physical cache capacity are distinct: the canonical launcher
advertises 512K but initially allocates 131,072 tokens and grows in 131K steps
only when a request needs more space. Ordinary 100K-class OpenCode sessions do
not pay for unused half-million-token KV. Recovery snapshots copy only the
mutable 128-token circular windows and recurrent compressor state; compressed
and indexer history is append-only and becomes invisible when the logical
position rolls back. The compact snapshot is therefore context-independent
(17,842,176 bytes for the official 43-layer schedule) instead of duplicating
the entire live cache. At the trained one-million-token limit a fully grown
live allocation is 7,232,045,056 bytes. Operators must still leave sufficient
unified-memory headroom for Metal scratch.

The original growth implementation dropped the old allocation and cleared the
live token ledger, logits, and recovery anchor before allocating the larger
cache. That made every capacity change a full transcript replay. The corrected
design must allocate a strictly prefix-compatible destination, copy the live
window/compressed/indexer/recurrent state, rebind only the matching compact
recovery snapshot, and swap only after every copy succeeds. Allocation or copy
failure leaves the old cache and all serving ledgers usable. Matrix prefill
remains at the measured 2,048-token transaction while physical capacity is
131K and drops to 1,024 tokens after growth unless an operator supplies an
explicit benchmark override.

Tool definitions are rendered by the owned 0731 encoder. Grammar-constrained
generation emits one official `<｜DSML｜tool_calls>` envelope containing one or
more invokes, validates required parameters, and converts the completed block
to OpenAI `tool_calls` objects for both unary JSON and SSE. Raw DSML framing is
not exposed to API clients. String parameters admit ordinary source-code angle
brackets. An earlier `[^<\\]` grammar rule forced calls containing Rust generic
or lifetime syntax such as `fmt::Formatter<'_>` to close immediately before
the `<`, yielding valid but semantically truncated tool arguments.

Non-string DSML parameters are constrained by their full recursive JSON
schema, not merely by generic JSON syntax. Nested required fields, scalar
types, arrays, objects, enums, and closed-object boundaries therefore apply
before a token can be emitted. This is required for client tools whose sole
top-level parameter is an array of structured objects: syntactically valid
JSON such as a required string field set to `null` must be rejected by the
grammar rather than discovered after generation by the client.

## Acceptance gates

### Converter gate

- A synthetic official-layout checkpoint converts to `Q2_K_S` in-process.
- Positive tests cover E4M3, all E2M1 nibbles, E8M0 edges, integer routing,
  canonical metadata, tensor naming, expert ordering, and GGUF round-trip.
- Negative tests cover missing/malformed scales, wrong dtypes and ranks,
  incomplete expert groups, invalid route tables, truncation, and interrupted
  temporary-output preservation.
- The converter never spawns a converter/runtime process.
- Existing Gemma 4 and Qwen 3.5 conversion regressions remain green.

### Official conversion gate

- The downloader records the exact official revision and every downloaded file
  hash before conversion.
- Output is reproducible from that manifest and carries an immutable receipt.
- Peak private RSS, wall time, output bytes, and per-type byte totals are saved.
- GGUF metadata, tensor names/shapes/types, and sampled dequantized payloads are
  compared with the pinned reference. Whole-payload byte identity is the target;
  any exception requires a tensor-local numeric proof and an ADR amendment.

### Inference gate

- Tiny exact fixtures pass CPU-reference-vs-Metal checks for every new primitive.
- Real-model greedy runs pass repeated determinism, one-shot-vs-incremental,
  cache rewind, cache fork, context-boundary, and interleaved-slot coherence.
- A fixed prompt corpus covers plain chat, thinking, reasoning-effort controls,
  tool calls/DSML, long context, Unicode, and stop/EOS behavior.
- Token/logit parity is measured against the same source-bound peer build.
- Three-run median prompt/decode throughput and peak memory meet H3/H4.
- Existing public API and model-family regression suites remain green.

## Benchmark discipline

The parity harness is outside product code. It records exact hf2q, mlx-native,
peer, source-model, and artifact commits/hashes; prompt bytes; context and
batch sizes; sampling parameters; cache state; thermals; memory pressure; and
all raw timing samples. Warm and cold-cache results are not mixed. Optimization
is allowed only after a profile identifies the limiting operation, and every
optimization reruns coherence before its speed result is accepted.

## Delivery and publication

Work occurs in isolated writer worktrees. The integration owner alone reconciles
shared manifests and lockfiles. Focused tests precede regression, coherence,
memory, and benchmark gates. Local commits and remote publication are separate
actions; pushing requires explicit authorization. No artifact is described as
ready until its exact-source receipt and all applicable acceptance gates are
green.

## Agentic acceptance revalidation (2026-08-06)

The following source-bound measurements validate the mixed profile and runtime
changes. The measured candidate is
`/opt/hf2q/artifacts/DeepSeek-V4-Flash-0731-agentic-q2.gguf`:

| Evidence | Candidate result |
|---|---|
| Source identity | Official revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`; 73-file bundle SHA-256 `a8544e6469f8f392e72f953e9a2b4ee33a23c50a859f47dd354d37ab0093993d` |
| Artifact identity | 107,431,343,104 bytes (100.05 GiB); SHA-256 `914e9de7f6bad70d795179fedf68ac4336880c93c3a8c7c09ad019f0b28f6bc4` |
| Native dependency | Exact crates.io pin `mlx-native =0.10.3`, released from main commit `9f2210c8f03484a9c1122ef6a2bcf3be4226f29a` with implementation head `43d4b43b2cce1e9289561f45b9f6433437714028`; registry checksum and independently downloaded crate SHA-256 `1db3a6e739a199c7e9a7820a49718d549f6b03a77241f461cffb3ad085cb833d` |
| Quant plan | 1,328 verifier tensors: 172 Q2_K, 86 Q3_K, 532 Q8_0, 535 F32, and 3 I32; 4,705 DSpark tensors explicitly excluded from the base artifact |
| Conversion bound | Rust-native row-aligned streaming; maximum working vector bound 4,798,873,600 bytes; no external converter or inference process |
| Arithmetic coherence | Greedy `What is 2+2? Reply with only number.` returned exactly `4` |
| Tool semantics | The curl/OpenAI-compatible harness made required and automatic choice both select `read_file` with exactly `/opt/hf2q/Cargo.toml`; unary and SSE returned valid OpenAI `tool_calls` and `finish_reason: "tool_calls"`. The source-argument regression also returned `fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result` byte-for-byte in one `emit_source` call (4 s response), proving the formerly truncated `<` syntax through the real model. A real OpenCode two-turn run issued five tool calls on its first turn and two on its second, changed the requested source, and passed the named oracle/regression checks in the same session. |
| Tool-result continuation | With tools enabled in `auto` mode, the model consumed the Cargo result and returned the requested sentinel without another call. The real OpenCode continuation consumed prior tool results, issued the next required calls, and reused the same live session. |
| Prefix reuse | The current checked-in gate used a 6,262-token prompt and reused 6,254 tokens on repeated, automatic, SSE, and post-tool turns. Cold TTFT was 9.564 s and cached TTFT was 227 ms; every required/automatic tool, source-argument, tool-result, unary, and SSE assertion passed. |
| Canonical launcher | `scripts/serve_deepseek4_opencode.sh` passed the curl agentic gate, the real OpenCode coding run, the four-agent full-context gate, and the 120K cache-growth gate. It advertises 524,288 tokens for every slot and demand-allocates 131K initially. Memory/port preflight refuses an unsafe 100 GiB load before model mapping. The 131K-to-262K boundary is proven; a near-512K physical allocation remains unproven on this 128 GiB host. |
| Ordinary agentic prompt | On the same approximately 5.9K-token README coding prompt, hf2q warm prefill was about 518 tok/s and median decode was 32.1 tok/s; peer build 10293 reported 399.4 prompt tok/s and 31.7 generation tok/s. |
| 120K cold prompt | `scripts/test_deepseek4_long_context_cache.sh` produced the exact required tool call for 119,808 prompt tokens in 321.034 s TTFT (373.194 tok/s); decode was 23.869 tok/s. The same source-bound prompt and artifact under peer build 10298 processed 119,807 tokens in 749.015 s (159.953 tok/s) and decoded at 19.565 tok/s. hf2q was 2.33x the reference prompt rate and 1.22x its decode rate for these source-bound runs. |
| 120K continuation cache | Appending the real tool result produced a 119,907-token request that reused 119,800 tokens (99.91%), evaluated only a 107-token suffix, returned its first semantic event in 1.113 s, and emitted the exact requested sentinel. |
| 98K OpenCode-scale revalidation | A fresh 97,127-token required-tool request completed cold prefill in 424.522 s (228.79 tok/s). Its 97,214-token tool-result turn restored the compact recovery anchor, reused 97,119 tokens (99.90%), evaluated a 95-token suffix in 1.378 s TTFT, and completed in 2 s. |
| Before/after control | The earlier identical-class hf2q run required 594.575 s for 119,808 tokens (201.502 tok/s). The completion candidate reduced cold-prefill time by 46.0% and increased its token rate by 85.2%. At the shorter exact 26,024-token gate, the threshold retune improved 388.846 tok/s to 497.277 tok/s while retaining the exact tool call; the cached continuation reused 26,016 tokens and reached 927 ms TTFT. |
| Output parity | Both runtimes returned the exact requested comma-separated sequence. The peer also returned the exact required `read_file` path on the long repository prompt. |
| Memory safety | A 4,096-row prefill command buffer produced Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory`; the accepted 2,048-row transaction completed the 119.8K gate. A later OpenCode session falsified the steady-state claim when macOS killed hf2q at 108,840 MB after the shared transient arena retained cold-prefill buckets. The split-arena build releases prefill scratch before decode; the new file-backed build released 4,933,917,968 transient bytes after the 120K prefill, remained alive through its cached continuation, reported 2.0 GiB RSS, and shut down cleanly. Eagerly allocating the full 524,288-token cache beside this artifact still OOMs, so demand growth remains required. Only one 100 GiB-class runtime was resident during every comparison. |

### Completion-audit performance ledger (accepted candidate, 2026-08-06)

The tighter completion gate uses the reproduced 107,431,343,168-byte artifact
with SHA-256
`936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d` and
peer commit `15586e2d7165570fb3aa7c26e0d442e289ef69de`. On the first clean
cooled three-trial matched run, peer medians were 674.458 prompt tok/s and
31.955 decode tok/s; hf2q medians were 620.321 prompt tok/s and 33.984 decode
tok/s. Every deterministic reasoning/answer transcript matched exactly. Decode
passed; prefill reached 91.97% and remained an acceptance failure.

The accepted lower-level candidate already generalizes batched GGML MM/MV,
explicit token-major input strides, slotted expert-down output, and pooled
scratch. The current completion candidate removes the remaining dense D=512
query/output permutations: the `mlx-native` flash primitive accepts native
token-major `[B,Q,H,D]` Q/O strides while K/V keep their existing layout. The
16x512 and unaligned 13x517 tests are BF16 bit-exact against the former
head-major path. On the isolated first-cool exact prompt, throughput rose from
651.391 to 658.294 tok/s, GPU time fell by 71 ms, and transient scratch fell by
approximately 403 MiB. This is a measured 1.06% improvement, not yet H4
acceptance; a second identical request measured 624.892 tok/s and preserved the
repeat-run drift under audit.

Exact layer timing localized that drift to GPU execution rather than CPU graph
encoding. On two identical 4,987-token requests, chunk GPU intervals changed
from 2,598.870/2,923.858/1,490.944 ms to
2,605.862/3,068.827/1,603.322 ms. Source-bound crossover accounting corrects
the original interpretation: chunk 1 has about 512 valid compressed entries
and uses dense flash; chunks 2 and 3 reach at least 1,024 entries and use
gathered sparse flash. The gathered chunks grew by 5.0% and 7.5%, while the
dense chunk was flat. Encoding and commit residuals remained small. This ruled
out dense-path residency as the whole explanation, but the later Metal System
Trace below supersedes the inference that gathered GPU arithmetic is the
primary wall-clock deficit.

The sparse adapter nevertheless had a separately measured gather cost. A
packed BF16x4 gather changed the production-tile adapter from 1.449 to 0.816 ms
and the complete sparse path from 4.106 to 3.450 ms at Q=256. Across
Q=64/128/256/512/1,024, adapter time fell 43.7% at the production tile and
complete sparse-path time fell about 16%. That isolated win did not survive a
quiet same-binary full-model A/B. Packed-gather trials measured
654.980/616.345/622.019 prompt tok/s; the fresh scalar arm measured
644.114/621.784/624.185 prompt tok/s. All six transcripts were exact. The
packed median, 622.019 tok/s, was slightly below the scalar median, 624.185
tok/s, so the shader, dispatch seam, and tests were removed from the landing
diff. An earlier 112.186 tok/s scalar sample was excluded because an unrelated
`/opt/repo-to-cve` build and XProtect activity contaminated the host.

Follow-up hypotheses measured and rejected or removed from the landing diff:

- Gathering only the ratio-128 compressed-attention rows before the dense
  projection was tested behind a fail-closed diagnostic toggle. Same-binary
  ABBA trials produced 608.998 prompt tok/s for gather versus 616.924 for the
  existing dense path. Gather was 1.28% slower and has been removed.
- CPU-side `encode_layer()` work across all 43 layers and three chunks consumed
  about 211 ms of an 8.146 s prefill, an approximately 2.6% upper bound even if
  perfect overlap removed all of it. Layer pipelining alone therefore cannot
  explain or close the measured 8.03% reference gap.
- One all-layer command buffer page-faulted; naive asynchronous scratch reuse
  page-faulted; duplicated full scratch arenas exhausted memory. A corrected
  host/GPU lifetime split proved safe graph submission. It did not remove the
  separate repeat-run residency decay, but the later heartbeat RCA cleared
  that confounder. The unsafe arena variants were removed; the dependency-
  reordered, retained-reference path is accepted at depth four below.
- Retaining at most 5 GiB of prefill scratch between requests did not remove
  the slowdown. Two exact trials measured 614.184 and 593.865 prompt tok/s
  while retaining 3,938,691,698 bytes and leaving only 13% free memory. The
  diagnostic was rejected and removed because it reduced safety margin without
  improving throughput.
- Lowering the ratio-four gathered-sparse crossover from 1,024 to 512
  compressed entries preserved the exact 1-through-64 transcript, but the
  first clean 4,987-token trial measured only 593.882 prompt tok/s and 33.856
  decode tok/s. This was below the predeclared 648 prompt-tok/s immediate
  rejection floor. The threshold and its diagnostic logging were restored to
  the accepted default after the single falsification run.
- A Q2_K/top-six tensor-API gate/up fusion reused the routed map and input tile
  while retaining separate F32 outputs for DeepSeek's exact asymmetric-clamp
  activation. Full-tile and alignment-tail tests were bit-identical to two
  production tensor `mm_id` projections. At a production-width synthetic M5
  shape (256 tokens, 32 resident experts, N=2,048, K=4,096), however, the
  current two-pass median was 2.339 ms and the fused median was 3.078 ms
  (0.760x). Two live cooperative accumulators reduced occupancy more than
  shared staging saved. The kernel, dispatcher, and tests were removed.
- Reducing the production packed D512 flash kernel from eight simdgroups to
  four preserved BF16 bit identity against the existing one-head-per-tile
  reference. It was nevertheless about 2x slower across Q=64 through Q=1,024;
  at Q=256 and a 640-entry attention width, NSG=4 measured 5.512 ms versus
  2.669 ms for NSG=8. The explicit-NSG spike was removed and the peer-derived
  NSG=8 geometry remains authoritative.
- Keeping every 2,048-token transaction on dense masked flash matched the
  exact 1-through-64 transcript with zero cached tokens, but reached only
  615.146 prompt tok/s and 33.865 decode tok/s. That missed the predeclared
  624.185 prompt-tok/s scalar floor and the 674.458 peer median. The
  all-dense route was rejected; one final mixed-route spike isolates chunk 2
  before closing this routing hypothesis.
- Routing chunk 2 through dense flash while leaving chunk 3 gathered produced
  three exact, zero-cache-credit trials at 625.484/625.724/625.436 prompt
  tok/s and 33.855/33.886/33.898 decode tok/s. The 625.484 median was only
  0.21% above the clean 624.185 scalar median and remained 7.26% below the
  674.458 peer prompt median. That noise-sized short-shape result does
  not justify changing long-context routing, so the 1,024-entry crossover was
  restored and the routing hypothesis is closed.
- A packed F16 heads-as-rows variant was bit-identical to the existing F16
  one-head-per-tile reference, but it did not outperform packed BF16. At
  Q=256 and width 640 they measured 2.667 and 2.664 ms; across Q=64 through
  Q=1,024 the F16/BF16 speed ratio stayed between 0.98x and 1.01x. The F16
  dispatcher extension and benchmark wiring were removed rather than add a
  conversion and model-quality risk without a speed result.
- Doubling the gathered sparse query tile from 256 to 512 rows preserved the
  exact transcript and produced a 658.512 prompt-tok/s first long request, but
  the two cooled follow-ups fell to 623.076 and 623.007 tok/s. The 623.076
  median missed the 630 tok/s acceptance floor and did not justify another
  160 MiB of transient gathered KV. The 256-row tile was restored.
- Tightening mlx-native's dependency ranges from the entire parent Metal
  allocation to each slice's exact logical byte window passed new disjoint-
  sibling and overlapping-sibling tracker tests, plus the existing D512
  bit-identity gate. It did not improve the exact full-model workload. The
  old-range arm measured 647.074/643.362/588.671 prompt tok/s; the exact-range
  arm measured 653.990/632.829/607.175 tok/s. All six transcripts were exact,
  both arms reported 385 GPU synchronizations and 302,850 dispatches per
  trial, and the exact-range median was 1.64% lower. Arm-order sustained-load
  drift was larger than the candidate delta, so this cannot support a speed
  claim. The range spike was removed from the landing diff; any future
  correctness cleanup must be justified independently of DeepSeek H4.
- Raising the matrix transaction from 16 to 20 physical windows changed the
  exact 4,987-token prompt from 2,048/2,048/883 rows to 2,560/2,427. This
  safely reduced GPU synchronizations from 385 to 342 and dispatches from
  302,850 to 300,076, while preserving all six exact transcripts. It did not
  improve sustained throughput: the 2,048-row arm measured
  653.593/645.593/613.829 prompt tok/s, while the 2,560-row arm measured
  659.968/602.231/570.344. The wider first run was followed by a much steeper
  sustained-load collapse, and its 602.231 median was 6.72% below the
  2,048-row median. The default remains 2,048; fewer command buffers are not a
  win when the larger Metal workload drives worse steady-state behavior.
- Source verification closed the proposed activation-quant bypass. DeepSeek's
  official `inference/model.py` at verified revision `2b2bebc` explicitly
  applies block-64 FP8 simulation to every main non-RoPE KV row to match QAT,
  and applies normalized Hadamard rotation plus block-32 FP4 simulation to the
  lightning-indexer query and compressed KV. The peer's pinned DeepSeek-V4
  graph visibly applies the Hadamard transforms but has no equivalent runtime
  fake-quant operation. hf2q must retain the official arithmetic; removing it
  merely to win a benchmark would violate the coherence-first contract.
- The prompt cache planner stated that a matrix transaction retains only its
  final physical 128-row window, but the encoder passed all 2,048 rows to one
  parallel circular-copy dispatch. Sixteen source rows consequently targeted
  each cache slot without an ordering guarantee. The accepted correction adds
  explicit `window_source_start` and `window_write_count` bounds and copies
  only the newest non-overlapping suffix. A 2,048-row hosted test proves the
  exact 1,920-row skip and 128-row write. On the real 4,987-token oracle, the
  pre-fix arm measured 657.377/641.241/592.672 prompt tok/s and the bounded
  arm measured 657.211/629.829/600.777; all six transcripts were exact. This
  lands as a cache-correctness fix, not as a throughput claim.
- A follow-up Metal spike fused KV tail RoPE, required MXFP8 simulation, and
  the bounded cache write. A 137-row wrapped-cache test was BF16 bit-identical
  to the three-operation chain, and the production path reduced reported GPU
  dispatches from 302,850 to 290,982. It nevertheless made sustained prefill
  worse: the bounded control measured 658.091/640.947/607.926 prompt tok/s,
  while fusion measured 655.515/603.950/575.113. The fused median regressed
  5.77%; decode improved only from 33.928 to 34.167 tok/s at the median. The
  kernel, dispatcher, and model wiring were removed. This is further evidence
  that reducing dispatch count by making each M5 workload denser can worsen
  sustained prefill.

#### Metal System Trace and fresh-buffer initialization RCA

A sanitized Metal System Trace of the exact 4,987-token request changed the
working hypothesis from slower GPU arithmetic to poorer host/submission
continuity. During prefill, the pinned peer occupied 6.982789 seconds of a
7.496751-second GPU submission span: 0.513962 seconds idle and 93.14% union
utilization. hf2q occupied only 6.869377 seconds of a 7.757780-second span:
0.888403 seconds idle and 88.55% utilization. hf2q therefore completed about
113.4 ms less GPU work but accumulated about 374.4 ms more idle time, producing
the observed approximately 261.0 ms wall-clock loss. Decode showed the
opposite shape: the peer occupied 4.033765 of 5.113851 seconds (78.88%), while
hf2q occupied 3.637267 of 3.822293 seconds (95.16%). The checked-in
`aggregate_decode_mst.py` reports clipped phase windows, union busy time,
overlap, idle time, utilization, and gap distributions; its units are GPU
submissions, not an invented kernel-dispatch count.

The trace also established an operator-security rule. `xctrace` can attach the
target process environment to raw trace metadata, including inherited
credentials. Trace-launched servers now receive a minimal `env -i`
environment, the trace trial must be last so the model unloads before trace
packaging, and retained evidence contains only the four required GPU-schema
XML exports after a credential-pattern scan. Raw trace bundles and raw TOC
dumps are not retained or printed.

Per-layer timing separated host encoding from GPU waits. Across the three
prefill chunks, all 129 layer encodes consumed 217.609 ms in total; the first
chunk's layer 0 accounted for 132.221 ms of first-shape setup. GPU waits were
2,618.319, 2,937.359, and 1,506.513 ms, while all pool resets together were
under 0.23 ms. The synchronous 64-row startup warmup does not establish the
2,048-row, multi-gigabyte transient high-water. Existing deeper command-buffer
pipeline experiments nevertheless remain rejected because they failed
sustained-load or memory-safety gates; this trace does not revive them without
a different lifetime design.

The sanitized encoder list makes that submission difference structural rather
than speculative. hf2q used 129 compute encoders for prefill, exactly three
2,048/2,048/891-row chunks times 43 layers. The pinned peer used six compute
encoders over the same prompt, two per chunk, before its output work began.
The runtimes therefore do not merely choose different kernel shapes: hf2q
forces 43 CPU/GPU rendezvous points per chunk while the peer lifetime-plans a
whole chunk graph and partitions it into two command buffers.

Hoisting hf2q's two reusable full-state buffers cannot close that gap. A
temporary allocation-timing build measured the six chunk-local allocations at
5.654/5.603, 7.612/6.224, and 2.886/3.116 ms, about 31.1 ms total in a
7.991-second exact-output prefill. The instrumentation was removed after the
measurement. Reusing maximum-sized state buffers across chunks therefore has
only an approximately 0.4% upper bound and is not the next implementation
target.

The accepted submission design differs from the rejected naive pipelines.
DeepSeek's transient arena mixes GPU-produced scratch with CPU-written RMS
parameters, positions, frequencies, attention indices, validity flags, and MoE
token IDs. Resetting and reissuing that mixed arena while a command buffer is
pending can overwrite inputs before the GPU reads them; retaining a complete
arena per in-flight layer reproduces the prior multi-gigabyte OOM. The safe
graph path gives CPU-written inputs a submission-lifetime arena, reuses only
hostile-fill-proven GPU scratch between layers, preserves retained buffer
references, reorders the dependency graph before commit, and inserts explicit
layer barriers inside one ordered command buffer.

The first depth-two gate proved the lifetime mechanism but rejected shallow
grouping as a performance result. Three exact, zero-cache-credit trials
measured 669.898/631.328/633.317 prompt tok/s and
33.895/33.892/33.867 decode tok/s. The 633.317 prompt median is only 0.07%
above the accepted 632.861 baseline, while GPU synchronizations fell from 385
to 322 and dispatches stayed at 302,850. The server released 3,936,241,088
bytes of prefill scratch, remained alive with no throttled pages, and shut down
cleanly. Therefore reducing one rendezvous per adjacent layer pair is not by
itself the missing speedup. Deeper groups remain a separate falsification of
whether the observed idle is amortized only at chunk scale; they must reuse
the same bounded scratch design and stop immediately on any command-buffer or
memory fault.

Depth four preserved exact output and bounded memory while reducing total
synchronizations to 289. A later same-candidate consecutive set measured
677.684/638.907/640.949 prompt tok/s and a 33.900 decode median. The repeat-run
stall later proved to be Metal residency preparation rather than grouping, so
that decayed median cannot reject the safe graph path. Depth 43 reduced
synchronizations further to 259 but its exact first trial reached 677.373
prompt tok/s and 33.890 decode tok/s, no improvement over depth four's exact
677.096 safety trial. Four layers per command buffer is therefore the accepted
default: it removes 96 layer rendezvous from the three-chunk prompt without the
lifetime and command-buffer risk of an all-layer submission. The graph reorder
and retained-reference checks fail closed; incompatible layer-timing or dump
diagnostics fall back to one layer.

Two-request stage profiling then separated arithmetic from residency idle.
With the profiler enabled, total prefill wall time grew by about 456 ms between
the identical requests. Summed GPU intervals were nevertheless stable:
gate/up changed -0.238 ms, expert/shared down -0.691 ms, ratio-four sparse
attention -0.600 ms, ratio-four cache/indexer +0.659 ms, and the remaining
named buckets changed by roughly one millisecond or less. This instrumentation
adds many synchronization points and is not a throughput benchmark, but it
falsifies a thermal slowdown inside a specific MoE, attention, or indexer
kernel as the explanation for the residual wall-time drift.

Source comparison exposed a residency-preparation difference below hf2q's
model graph. The pinned peer commits each Metal residency set, immediately
calls `requestResidency`, and refreshes active sets from a five-millisecond
keep-alive loop. `mlx-native` commits pending membership and attaches its set
to the command queue but has no `requestResidency` call. Apple's API contract
says that `requestResidency` asks Metal to perform preparatory residency work
and should ideally run after the set commit, well before a consuming command
buffer. This is an optimization gap, not an inference-correctness defect.

The first gated implementation requested residency synchronously before every
bounded prompt chunk. A quiet same-binary A/B measured the untreated control at
668.235/630.826/625.513 prompt tok/s and the candidate at
669.034/630.801/632.584 prompt tok/s. All six transcripts were exact, but the
0.28% median change was noise-sized; per-chunk preparation was removed.

Lower-level timing then isolated the repeat-run loss. On the first long
request, the first 4,096-token chunk spent 2,676.927 ms wall time and
2,591.656 ms GPU time, leaving 85.271 ms outside GPU execution. On the next
request, the same chunk spent 3,092.863 ms wall time and 2,573.018 ms GPU time,
leaving 519.845 ms outside GPU execution even though the GPU work was faster.
Inside the first reordered layer group, command-buffer wall time grew from
225.642 to 706.790 ms while GPU time changed from 225.327 to 233.018 ms: the
pre-GPU residual grew from 0.315 to 473.772 ms. The same request made three
fresh pool allocations totaling 3,918,088,192 bytes in 2.301 ms, and retaining
the complete 3.96 GiB scratch high-water still left a 426.565 ms first-group
residual. Disabling residency sets also preserved the collapse. These spikes
falsify Rust allocation, scratch lifetime, kernel arithmetic, and residency-set
membership as the cause; Metal was preparing the inactive weight resources at
the first consuming command buffer.

The pinned peer supplies the missing lifecycle contract. Its Metal backend
refreshes active residency sets every five milliseconds and keeps that work
alive for three minutes after graph execution. The accepted `mlx-native`
boundary now owns the same family-neutral policy: each live residency set has
a weak-owned background heartbeat, every command-buffer commit refreshes a
180-second counter, and the heartbeat serializes `requestResidency` with
membership mutation. The thread exits after the final residency-set owner is
dropped. `MLX_NATIVE_RESIDENCY_KEEP_ALIVE_SECONDS=0` disables the policy for a
controlled diagnostic; invalid or absent values use 180 seconds. No
DeepSeek-specific runtime call or product fallback is involved.

The two-request spike returned exact transcripts at 677.673 and 675.923 prompt
tok/s. The second request's first-group pre-GPU residual fell from 473.772 to
2.487 ms while its 4.59 GiB transient prefill scratch was still released before
decode. Focused residency and graph tests passed before the full gate.

Fresh-allocation profiling then found 77 pooled allocations requesting
3,635,345,664 bytes and spending 182.287 ms in CPU allocation/clearing. An
eight-thread clearing spike preserved exact output but measured
654.977/622.899/623.142 prompt tok/s versus the zeroed control's
660.326/623.350/624.618; its median was 0.24% slower and the spike was removed.
A diagnostic that skipped fresh clearing measured
669.223/633.288/632.911 prompt tok/s, a 1.39% median gain, establishing a real
but bounded opportunity.

Coherence was proved before making that behavior explicit. The zero-filled
control and a hostile `0xA5` fresh-fill candidate each dumped every prefill
chunk and decode step for artifact
`936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d`.
All 11,954 files and the final transcript were byte-identical. This matches the
pool's existing contract: reused buffers are not cleared between arena cycles,
so DeepSeek's transient graph already must fully produce every consumed byte.

The accepted boundary is therefore an explicit `mlx-native`
`MlxBufferPool::alloc_uninitialized` operation. Default `alloc` and
`alloc_batch` keep zero-on-fresh behavior; only hf2q's DeepSeek-V4 prefill and
decode transient arenas opt in. Qwen, Gemma, persistent allocations, and
inactive DeepSeek allocation paths are unchanged until they have equivalent
whole-graph producer-coverage proof. Focused allocator tests and all 94 hosted
DeepSeek tests passed. The real release build returned the exact transcript in
three zero-cache trials at 670.921/632.703/632.861 prompt tok/s and
33.888/33.889/33.891 decode tok/s. The 632.861 prompt median is 1.32% above the
matched 624.618 zero-filled control; decode is unchanged. The diagnostic
environment switches were removed. This gain is accepted, but it does not
close H4 because the pinned peer prefill median remains higher.

The earlier attribution to ordinary sustained-load GPU DVFS is rejected by the
GPU-interval and first-group measurements above. The final clean-registry
paired gate used the same artifact, 4,987-token prompt, greedy
temperature-zero/seed-42 settings, exact transcript oracle, zero prompt-cache
credit, and 60-second gaps before and between three trials. The pinned peer
commit `15586e2d7165570fb3aa7c26e0d442e289ef69de` measured
673.497/672.744/674.711 prompt tok/s and
31.810/31.855/31.821 decode tok/s. The hf2q build pinned to the verified
`mlx-native` 0.10.3 registry archive measured
674.026/674.785/676.812 prompt tok/s and
34.054/33.885/33.958 decode tok/s. Every transcript was exact; hf2q therefore
passed this cold-prefix gate at 1.0019x peer prefill and 1.0672x decode.
The complete source-bound receipt is retained under
`hf2q-deepseek-parity.XXXXXX.1PXRWahgxc`; it records artifact, binary, hf2q
patch, the clean mlx-native source state, and implementation hashes.

The 2026-08-07 current-reference refresh used the same reproduced artifact,
hf2q runtime `03e378e9862e6d9add0d08ea68c1d6c449357364`, clean `mlx-native`
head `eb1b031876a0d5aa3b16803a54e78aa5de7d2e62`, and the peer at
`3653e6d6d547ec763317d9ecd0ace334a7e21359`. All six transcripts were
again exact with zero cache credit. The peer measured
666.655/674.345/669.544 prompt tok/s and 31.404/31.732/31.814 decode tok/s;
hf2q measured 661.982/669.469/674.258 prompt tok/s and
33.993/33.499/33.880 decode tok/s. The medians were therefore 669.544 versus
669.469 prompt tok/s and 31.732 versus 33.880 decode tok/s. Decode remained
1.0677x faster, but the strict prompt gate failed at 0.999888x even though the
0.0112% difference is noise-sized. The raw evidence is
`hf2q-deepseek-parity.XXXXXX.ue8E69ocNH`; this refresh reopens the prompt
margin rather than weakening the `>= 1.00x` rule.

The first reformulated hypothesis attributed the monotonic hf2q request times
(7.5334/7.4492/7.3963 seconds) to production-shape work left uncovered by the
64-token startup warmup. An initial direct 4,096-row implementation was
stopped before loading either runtime when source review showed that it
bypassed the normal chunker and would recreate the already rejected 4K Metal
OOM transaction. The corrected spike used 4,096 varied valid token IDs through
the ordinary prompt chunker, producing two bounded 2,048-row transactions and
resetting cache/scratch before readiness. Its exact three-trial hf2q results
were 661.703/671.261/624.270 prompt tok/s and
33.687/33.659/30.504 decode tok/s, while the matched peer arm stayed flat
at a 669.954 prompt-tok/s median. The warmup neither improved the first trial
nor preserved sustained performance, so the entire code/test spike was
removed. Moving another full sparse/indexer workload into startup is not an
accepted performance technique.

The next spike kept the same three Metal transactions for the 4,987-token
parity prompt but changed their shapes from 2,048/2,048/891 to
1,664/1,664/1,659 by selecting 13 sparse windows. Against the same current
peer source, the exact zero-cache hf2q trials measured
671.015/672.071/671.476 prompt tok/s; the peer measured
673.491/670.910/666.989. The 671.476 versus 670.910 medians passed at
1.00084x, and hf2q decode remained 1.0653x faster. The raw evidence is
`hf2q-deepseek-parity.XXXXXX.cBi5FosgDZ`.

Thirteen windows are not a global replacement for the established 16-window
transaction. The required 119,821-token cold agentic prompt remained coherent
and produced the exact tool call, but fell to 336.226 prompt tok/s and
356.371-second TTFT. Its 119,916-token continuation correctly reused 119,813
tokens and reached the semantic response in 1.164 seconds, proving that cache
semantics were intact rather than explaining the cold regression. Therefore
the implementation balances only an uncached prompt
strictly between two and three default transactions. Cached suffixes,
boundary-sized prompts, long prompts, and grown-cache requests retain their
previous measured policies.

The final bounded-adaptive gate used the same reproduced artifact, current
peer `3653e6d6d547ec763317d9ecd0ace334a7e21359`, clean mlx-native
`eb1b031876a0d5aa3b16803a54e78aa5de7d2e62`, and the exact hf2q candidate
binary SHA-256
`222251a89a3535a92e6ba7c847fb1e395a5d617e78668c4c6f2449baf6ffae69`.
The peer measured 670.226/670.948/670.117 prompt tok/s and
31.658/31.547/31.534 decode tok/s. hf2q measured
672.913/678.760/677.283 prompt tok/s and
33.647/33.982/33.985 decode tok/s. Every transcript was exact with zero cache
credit. The 677.283 versus 670.226 prompt medians pass at 1.0105x; the 33.982
versus 31.547 decode medians pass at 1.0772x. The source-bound evidence is
`hf2q-deepseek-parity.XXXXXX.oeSqpKMbBX`.

The exact candidate then passed the complete agentic gate: required and
automatic tools, unary and SSE encoding, source-shaped arguments, and a real
tool-result continuation. It reused 6,250 of 6,258 prompt tokens and reduced
cached TTFT from 9.660 seconds cold to 228 ms. The no-cooldown 119,821-token
correctness run on the same server produced the exact required tool at
332.240 prompt tok/s; its 119,916-token continuation reused 119,813 tokens and
reached TTFT in 1.132 seconds. The lower cold rate than the earlier isolated
373.194 tok/s observation is recorded as sustained-run variance, not claimed
as a speedup. It remains over the historical 159.953-217.5 tok/s peer
long-prompt observations, while the current strict matched performance claim
is limited to the cooled three-trial gate above.

The canonical OpenCode launcher defaults to that same schema-v2 reproduced
artifact, not the earlier schema-v1 mixed artifact or the rejected plain
Q2_K_S artifact. `MODEL` remains an explicit operator override because serving
supports compatible external GGUFs independent of producer identity. This
keeps the turnkey path bound to the exact source receipt and artifact hash used
by the accepted gates without turning provenance into a runtime restriction.

The parity harness now cools between measured trials as well as before and
between runtime arms. External source reviews, including Kimi and Claude,
supply testable hypotheses only; source inspection plus exact hf2q
measurements decide whether a change lands. Packed gather, scratch retention,
and the 512-entry sparse crossover are failed experiments and are not present
in the landing code. The accepted mlx-native changes were published as 0.10.3,
hf2q is pinned to its verified registry checksum, and locked check, release
build, and full hosted-safe tests passed after removing the local path patch.

The performance result comes from two measured defaults. Decode groups two
verifier layers per Metal command buffer; one layer reached 29.49 tok/s, two
reached 33.10, four plateaued at 33.05, and eight regressed to 32.80. The Q8_0
matvec uses the peer's geometry (`N_SG=4`, `N_R0=2`); enabling it raised
the accepted path to 35.55 tok/s with byte-identical Q8 parity tests. The Q3_K
expert-down choice was retained after an exact production-shape spike measured
201 us for six decode rows versus 351 us for Q2_K. These results falsify both
"more command-buffer grouping is always faster" and "Q2 expert downs would
close decode parity."

The checked-in gate is `scripts/test_deepseek4_agentic.sh`. It intentionally
uses a unique first turn, keeps the requested manifest out of the prompt,
checks cold and cached timing, exercises required and automatic tool choice,
reconstructs SSE arguments, requires one terminal `[DONE]`, appends a real tool
result, and verifies that the continuation reuses the unchanged prefix. A
short-output smoke test is not an acceptance substitute.

The near-boundary companion is
`scripts/test_deepseek4_long_context_cache.sh`. It calibrates with the server's
actual tokenizer, constructs a 116K-125K prompt, requires a valid tool call,
then appends its tool result with a 16,384-token generation reservation. That
reservation deliberately crosses the initial 131,072-token physical capacity;
the gate fails unless almost the entire prefix remains cached with bounded
continuation TTFT. This is the release guard against both the original
every-turn replay and destructive demand growth.

### Cache-growth boundary revalidation (2026-08-08)

An operator OpenCode transcript exposed a release-gate hole in main at
`1349ec6f`: the first 131,072-to-262,144 physical-cache growth logged
`cache="grow-reset"`, credited zero cached tokens, and replayed the complete
129,015-token transcript. The old implementation discarded the source cache
and its serving ledgers before installing the larger allocation. A later
same-capacity reset in the same operator session had no growth event and is
not attributed to this defect without prefix-divergence evidence.

The corrected candidate was built from that base with release-binary SHA-256
`af3b571c1215497773d0f0f12a9baeca80c15921d8f3335ba50431e2e449031f` and
served the exact accepted schema-v2 artifact on the M5 Max. Its isolated
boundary receipt is
`hf2q-ds4-growth.XXXXXX.GEgGT5Y6n5` under the host temporary directory.

| Phase | Prompt | Cached | TTFT | Result |
|---|---:|---:|---:|---|
| calibration | 27,320 | 0 | — | exact required tool; 580.747 prompt tok/s |
| cold long prompt | 119,856 | 0 | 359.402 s | exact required tool; 333.487 prompt tok/s |
| forced-growth continuation | 119,943 | 119,848 | 1.321 s | exact terminal sentinel; no extra tool call |

The continuation reserved 16,384 generation tokens, forcing a 262,144-token
physical allocation. hf2q migrated cache position 119,909 in 101.865 ms from
a 919,617,536-byte source to a 1,821,392,896-byte destination, then evaluated
only the 95-token suffix. Swap moved from 3,520.06 MiB before the run to
3,504.00 MiB after growth. The server shut down cleanly after the receipt.

The same candidate then passed `scripts/test_deepseek4_agentic.sh` in a fresh
server process. Required and automatic tools, unary and SSE responses,
source-shaped arguments, and tool-result continuation were exact. The cached
turn reused 6,333 of 6,341 prompt tokens and reduced TTFT from 9.796 seconds
to 227 ms. That receipt is `hf2q-ds4-agentic.XXXXXX.zDLrp8abZX` under the host
temporary directory.

Verbose reset diagnostics now report only token counts, cache positions,
common-prefix lengths, poison state, growth state, and matrix-prefill policy.
They do not log prompt text, decoded content, tool names, or arguments. This
distinguishes an incompatible transcript rewrite from a future cache defect
without exposing operator data.

### Full-context multi-agent revalidation (2026-08-08)

`scripts/test_full_context_agent_slots.sh` passed four concurrent DeepSeek
agent conversations through the canonical slot-aware launcher. Each slot
advertised the complete 524,288-token serving context; no context or KV stride
was divided by four. All four conversations passed required and automatic
tool choice, unary and SSE tool-call encoding, tool-result continuation, and
source-shaped arguments. Two powered gates passed. Cached turns reused at
least 6,677/6,685 tokens with maximum cached TTFT 268.68 ms, cached unary/SSE
turns completed in 6-13 seconds, and every four-agent tool-result turn
completed within 20-32 seconds. Exact server-side cold-cohort makespans were
53.86 and 52.32 seconds (53.09-second median); client-observed semantic walls
were 52-55 seconds.

The matched peer server completed its corresponding cold four-request
wave in about 54.1 seconds on the same artifact and host with `--kv-unified`,
`--parallel 4`, and 131,072 logical tokens per slot. Its 524,288-token unified
allocation did not fit beside the 100 GiB artifact on this 128 GiB host. hf2q
therefore met the matched wall-clock bar while providing 524,288 logical
tokens to every slot through demand-grown physical admission.

The 0.1.6 candidate scheduling policy uses resumable cold prefill at atomic
cache+ledger commit boundaries. At most two active cold prefills alternate
complete matrix transactions through the one shared prefill scratch arena. A
decode-ready member advances with an eight-token quantum between bounded
prefill transactions in the lopsided interactive case. When a filling cohort
still has another cold request queued, cold-wave unary decoders instead defer
through `Draining` while any cold prefill remains and bulk prefill resumes;
unary output could not be delivered before the barrier. Streaming and warm
decoders remain responsive. Once the cohort drains,
longest-prefix continuations run before unrelated cold work can evict a
retained agent cache. A paired long-row graph exceeded Metal memory, and a
batched-output-head-only spike did not improve the cold tail; neither failed
experiment is present in the landing code.

### Interactive mixed-work correction (2026-08-09 measured candidate)

A real foreground OpenCode run on released 0.1.4 admitted a 523-token decoder
beside a 107,045-token/347-tool cold prefill. The long lane committed 2,048
tokens every 6.27–6.98 seconds. The short lane advanced exactly one token after
each commit and its reported decode rate fell from 0.313 to 0.181 tok/s. This
falsifies the weaker assumption that any bounded semantic progress is usable
interactive progress: the scheduler was live, but the experience was not.

The replacement preserves the proven bulk-prefill plan and changes only a
genuinely mixed quantum:

- when a filling cohort still has another cold request queued, defer cold-wave
  unary decode through `Draining` while any cold prefill remains and restore
  the plan's normal 16-window/2,048-token transaction; streaming and warm
  decode remain visible and keep the interactive budget;
- while at least one `Decode` owner is runnable, cap the next matrix prefill
  transaction at two native 128-token windows and run the configured decode
  quantum, clamped to at most eight tokens, except for the saturated case;
- when every decode owner is absent or `ParkedCompletion`, remove the mixed
  cap and resume the plan's normal 16-window/2,048-token transaction;
- rebalance a capped slice that would leave an illegal 9–32-token matrix tail,
  leaving either at least 33 matrix tokens or at most the eight-token recovery
  tail;
- retain terminal parking and the cold-cohort cache barrier unchanged.

The exact release-candidate hardware gate passed on AC power with binary
SHA-256 `cd8867820898eb33beb5523894084ed5af5a8cdbba92c4aaa8ca4bbb48150784`,
model SHA-256
`936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d`,
and public 347-tool fixture SHA-256
`6671a0c89b8d4935caa4b87bee08361c5b8727ec557e9edb05947ad90c94c13d`.
The cold prompt rendered to 94,576 tokens. Its first genuinely mixed prefill
transaction was 256 tokens/two native windows, while the short lane generated
49 tokens in that first reporting window at about 9.16 tok/s and completed its
128-token response before the cold prefill ended. The long lane completed at
303.29 prompt tok/s, emitted the exact `fixture_tool_346` call with semantic
arguments `{ "path": "src/serve/api/engine.rs" }`, and left `/readyz` at 200.
The power-log delta was zero. The atomic summary SHA-256 is
`6f93283e07f65952bbd314cc01b791e439875ed0ca7a8a72b4553378ae9c177c`.

This receipt proves the user-visible overlap correction on the exact candidate
binary. The original 2,048-token plan remains selected whenever no Decode
owner is runnable, as pinned by the pure scheduler tests; public release
authority still requires the immutable packed artifact and exact-main CI.

The slot-aware long-cache falsifier first exposed an affinity defect: a
tool-result continuation matched the native recovery anchor but not the raw
generated-token tail, so the scheduler selected a fresh slot and growth
reported `migrated_tokens=0`. Recovery-anchor-aware affinity corrected the
selection. The final boundary request grew one retained slot from 131,072 to
262,144 capacity, migrated 119,762 live tokens, credited 119,692/119,778
cached tokens (99.92%), evaluated only an 86-token suffix, and returned in
1.132 seconds. That final correctness run occurred on battery power; its cold
prefill rate is not used for a performance claim.

### Cached-suffix overlap and cancellation revalidation (2026-08-09)

A released `0.1.5` OpenCode continuation exposed a short-tail planner defect
after reusing 107,066 tokens. The rendered request added 24 tokens, split at
the recovery-anchor boundary into 16 and 8 tokens. DeepSeek's verifier
correctly declines matrix append below its 33-token minimum, but the resumable
serve adapter accepted incremental replay only for the final eight recovery
tokens and rejected the preceding 16-token segment as an empty chunk.
Resumable prefill now treats every nonempty sub-33-token segment as incremental
replay and captures the recovery anchor when that replay reaches its boundary.
The exact `107066 + 16 + 8` shape is pinned by a model-free regression; release
authority still requires the retained-prefix OpenCode continuation on the
packed artifact.

The first focused overlap run exposed a scheduler-policy defect that the
co-admitted four-agent gate could not see. `Idle` meant that no cold-cohort
barrier was active, but admission reopened only when every physical slot was
empty. A staggered cached continuation therefore waited behind an already
decoding peer even though another slot was free. The corrected reconciliation
opens `Idle` admission whenever a physical slot is free; full cohorts remain
closed and the measured two-active-cold-prefill bound is unchanged.

The cancellation half exposed a separate cache-lifetime defect. A bounded
cached suffix observed client closure correctly, but its ordinary reset also
discarded the valid pre-request recovery anchor. Cancellation now restores an
unpoisoned anchor only when its snapshot position exactly agrees with the
anchor token ledger. A poisoned, missing, empty, or inconsistent anchor still
takes the conservative full-reset path. No partially extended request cursor,
logits, or scheduler accounting is published.

`scripts/test_deepseek4_cached_suffix.sh` binds both properties in one fresh
four-slot process. The focused M5 Max run used release binary SHA-256
`ee576b75e86623dd5887224450d37dcf6c9bad5d5f5f955338267ff6b9124076` and
produced the following receipt:

- the cached tool-result suffix reused 7,174 tokens and completed three native
  prefill transactions;
- its decoding SSE peer recorded two decode-progress events between the first
  and last suffix transactions, grew from 1,097 to 137,841 bytes, and emitted
  exactly one terminal `[DONE]`;
- cancellation disconnected exactly after transaction three, admitted no
  later transaction during settle or stability windows, incremented the
  client-cancellation counter exactly once, and emitted no `[DONE]`;
- the post-cancellation control request reused the same 7,174-token anchor;
  readiness remained true and the request-log delta contained zero fatal
  worker signatures.

The atomic summary is under the host temporary receipt root
`hf2q-deepseek-cached-suffix-rollback.XXXXXX.Iim31evVsJ`. This is focused
hardware evidence for the two repaired invariants, not final release
authority: a clean packed-artifact rerun and the unchanged full four-agent
quality/performance gate remain required.

### Four-agent workload identity correction (2026-08-10 candidate)

Release run `31443407887` passed DeepSeek interactive overlap, terminal
parking, three-transaction cached suffixes, three cancellation/rollback
positions, and the generic 116K-token lifecycle, then failed the first
four-agent cold wave at 80–85 seconds. That result was not comparable to the
55-second calibration: `scripts/test_deepseek4_agentic.sh` embedded the entire
mutable `README.md`, which had grown from 21,204 to 29,882 bytes. The rendered
DeepSeek prompt therefore grew from 6,685 to 8,573 tokens. Scheduler policy had
also changed since the earlier 52–54-second evidence, so the unmatched failure
does not prove or clear a scheduler regression.

The gate now reads the exact 21,204-byte calibration context from
`scripts/fixtures/deepseek4-agentic-repo-context.txt`, SHA-256
`2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef`.
At that revision, the release wrapper verified that artifact before model
startup and preserved the prompt-visible calibration path
`/opt/hf2q-worktrees/full-context-slots/Cargo.toml` from exact commit
`863ea423`; the simulated tool result was still read from the packed
candidate's `Cargo.toml`. This was workload identity, not a runtime dependency
on that local path. Every cold agent then rendered exactly 6,685 tokens.
Producer, aggregate, and publication checks all
bind the fixture ID, digest, byte/character counts, exact prompt count, zero
cold reuse, semantic/tool assertions, and the then-literal 55,000 ms cold
bounds. A
model-free negative matrix rejects missing, mistyped, stale, off-by-one, or
over-limit receipts.

At that stage the 55-second bound was intentionally unchanged. A fresh
exact-packed M5 Max
rerun of the frozen request twice is the discriminator: success restores
matched release authority; failure leaves a current-scheduler performance
blocker that must be optimized or re-baselined against a same-input
peer. This section records a candidate correction, not a passing hardware
claim.

### Thermal validity of calibrated four-agent waves (2026-08-11 candidate)

Exact-main run `31468849847` passed the 94,576-token interactive overlap,
cached-suffix cancellation matrix, and 119,855-token generic lifecycle before
the first frozen 6,685-token wave reported 61–65 seconds and failed the literal
55-second limit. The workload and scheduler shape were correct: all four
requests were cold, generated the same 61-token tool result, used the expected
two-prefill/mixed-decode/terminal-parking sequence, and emitted no fatal Metal
signature.

macOS unified logging made the measurement invalid as performance evidence.
Thermal pressure was level 2/Heavy from `00:45:25.617` through `00:51:34.616`;
the timed wave began only after the preceding lifecycle completed at about
`00:49:44`. Nominal returned at `00:51:54`, immediately after work stopped.
Against the cool-host control traces, the first pair fell from roughly 315–322
prompt tok/s to 256–261 tok/s and the mixed pair fell from about 232 to 197
tok/s; the separate published-0.10.6 decode control fell from roughly
17.2–17.5 to 15.1 tok/s. AC power and caffeinate remained valid, so AC presence
alone was insufficient evidence of an unthrottled host.

The corrected wrapper runs both calibrated waves before the long functional
workloads. Before each fresh server starts, with no hf2q/peer model runtime
loaded, the host
must report `ProcessInfo.thermalState == nominal` at five-second cadence for at
least 60 seconds. Thermal state is then sampled every two seconds throughout
the wave; any observed non-Nominal state,
malformed read, monitor failure, or telemetry gap invalidates the run. Each
envelope records the settle and measurement sample counts and SHA-256 digests,
and publication independently rehashes and validates the measurement log.
At that stage the 55-second bound remained unchanged. Two thermally valid
exact-packed passes were
still required before release authority is restored.

### Saturated four-cold bulk-prefill correction (2026-08-11 candidate)

Exact-main run `31477280331` exercised source
`c92d0b251bd49e43f9a1a70c41985b4ba45ae8fd`, crate SHA-256
`726966f381637fdc7eb63a123fd63b42836237ecdac8ef4fa2f08ee800829662`,
and packed binary SHA-256
`1a796156a073f69d85a33b36f5de6d367d541994e03ae914be2327048bea616b`.
The host was continuously Nominal for the required 60-second settle and the
whole measured wave. The run therefore supplies valid performance evidence,
and it failed the unchanged bound: two clients completed at 53.351 seconds,
while one client failed at 56.662 seconds.

Server clocks isolate the failure. The first two 6,685-token cold prefills
completed in 22.180 and 22.751 seconds at 301.39 and 293.84 prompt tok/s. The
second pair was then admitted alongside both decode-ready first-pair lanes and
took 30.382 and 30.434 seconds at about 220 prompt tok/s. Their cohort
endpoints were therefore about 56.444 and 56.941 seconds before small HTTP
overhead. The first pair's early 61-token responses could not be published:
they became terminal while the second pair was still prefilling and were
parked behind the cold-cohort barrier. The work delayed the second pair
without improving any user-visible completion. Cache reuse itself was healthy
before fail-fast cleanup: the next two requests reused 6,677/6,685 tokens and
finished their eight-token suffix prefills in 0.271 and 0.864 seconds.

The first draining-only correction then reached exact `main` as
`e9887cceeee9c69543e5b69193434db6602f4c9d`. Its protected packed-artifact run
`31484992493` used crate SHA-256
`b68d8f946c051c641d04da17505cf6c8d214e3f75a027eb64a9df8abb4082a44`.
All four individual cold responses met the unchanged 55-second limit at
54.476, 54.311, 54.642, and 53.377 seconds, and every cached, automatic, and
tool-result continuation reused exactly 6,677/6,685 tokens. The monotonic
cohort wall was nevertheless 55.250 seconds. The workflow itself failed closed
first when its thermal monitor observed a `fair` sample, before receipt
validation; the preserved temporary receipt was also 250 milliseconds over the
unchanged limit and would have failed that validator independently. The server
trace exposed the remaining transition: after request 1 finished prefill,
request 3 was admitted while request 4 was still queued, but the cohort was
still `Filling`; request 1 therefore consumed an eight-token decode quantum and
imposed the two-window prefill cap before the cohort entered `Draining`.
Request 2 then consumed one decode token at the same boundary. That unary work
was still not deliverable before all four cold prefills crossed the cohort
barrier.

The correction distinguishes this saturated state from the lopsided
interactive case. When the filling cohort still has another cold request
queued, cold-wave unary decode handles remain installed but are omitted from
GPU decode while any cold prefill remains; that deferral continues through
`Draining`, including the `1 prefill + 3 decoders` tail. Streaming and warm
decode handles remain runnable and preserve the measured eight-token
decode/two-window prefill budget. At zero active cold prefills, every deferred
decode resumes. Model-free scheduler tests pin those transitions. This is
source-level candidate evidence only; the exact packed four-agent wave must
pass twice and the 94,576-token interactive overlap must remain green before
the correction is accepted for publication.

### Exact-main failure and matched-peer discriminator (2026-08-11)

The refined barrier reached exact `main` as
`d930c3982fb326c7b11697faea2c0379520e536f`. Its first protected packed run,
`31492478455`, completed all four cold/cached/automatic/SSE/tool-result
conversations correctly: every cold request rendered 6,685 tokens with zero
reuse, every continuation reused 6,677 tokens, the maximum cold semantic wall
was 54.933 seconds, and the monotonic cohort wall was 55.499 seconds. The final
thermal sample changed to `fair`, so that otherwise-correct receipt was
invalidated rather than accepted.

Unchanged exact rerun `31495503525` stayed continuously Nominal and therefore
is valid performance evidence. It failed the literal bound when one client
reported 56.271 seconds. Server timing showed that the refined scheduler did
what it claimed: cold-unary decode remained deferred until all four prefills
finished, then the four 61-token responses decoded together. There was no
cache, ownership, Metal, or tool-semantics failure. The remaining floor was
roughly 48.6 seconds of cohort prefill plus 7.5 seconds of four-way decode.
One favorable rerun cannot erase that exact failure, and the 55-second limit
must not be relaxed without a source-bound peer or a measured optimization.

`scripts/test_deepseek4_peer_cold_wave.sh` and
`scripts/run_deepseek4_matched_peer.sh` now make that peer comparison
reproducible. They use peer build 10326 (`3653e6d6d`), binary SHA-256
`90bdf03673f7ee61d65d579a4e0be64a914edac1ccb23e74871040bc30d13543`,
the exact model SHA-256
`936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d`,
`--ctx-size 131072 --parallel 4 --kv-unified --flash-attn on`, and disabled
prompt caching. The four request JSON files are generated by the same frozen
builder used by hf2q. The pinned peer renders those identical bytes as 6,695
tokens rather than hf2q's 6,685; the peer-specific count is bound explicitly.

The first peer probe returned four exact zero-cache `read_file` calls in
72.037–72.423 seconds, but its final thermal sample was `fair`, so the timing
is diagnostic only. A second attempt loaded the peer, sent no requests, and
waited for a three-minute trailing Nominal window. All 176 five-second samples
over 904 seconds were `fair`; the harness timed out and unloaded the model
without running a measured wave. This falsifies treating the historical,
unreceipted approximately 54.1-second peer number as current thermal authority,
but it does not authorize a rebaseline from thermally invalid data. The release
remains blocked. The next accepted experiment is the family-neutral paired
MoE gate/up schedule for large prefill; decode-sized work remains a separate
measurement.

### Family-neutral paired MoE schedule (2026-08-11 candidate)

That experiment is now implemented without adopting the rejected literal
gate/up arithmetic megakernel. Published `mlx-native 0.10.8` comes from exact
main `0733dfbb280b3ceb8a8526489f1aebcdf454ee60`; the crates.io archive,
GitHub release asset, and local Cargo-1.96 seal all have SHA-256
`f2257d5afd2b0e246049e79b4857b0edc1554e93c81d7d2d2d1bcd64c73e22a9`.
Its family-neutral pair primitive builds one expert-ID routing schedule and
encodes two existing quantized projections with distinct weights and outputs.

hf2q selects that primitive only for rows above the native routing threshold,
automatic routing, available scratch, and no
`HF2Q_MM_ID_ROUTING_THRESHOLD` diagnostic override. Small/decode work,
forced matvec, slotted matrix routing, and calls without scratch preserve the
independent gate/up path. The paired call reuses the first routing scratch for
the pair; the routed down projection may reuse it only after both outputs have
been encoded. The existing graph barriers cover the input, both weights,
sanitized IDs, and both outputs.

The exact packed native benchmark passed on AC power under Nominal thermal
state and remained positive across eight covered shapes: Qwen Q5 pairs measured
1.0396–1.0657x and DeepSeek Q2 pairs 1.0102–1.0332x against the independent
schedule. Those are primitive measurements, not hf2q serving claims. The
published dependency, selector regression, broad model-free DeepSeek suite,
and locked all-target/all-feature check establish reproducibility and routing
safety only. Acceptance still requires two thermally valid exact-packed
6,685-token four-agent waves, unchanged 6,677-token retained-prefix reuse and
tool semantics, plus the 94,576-token overlap/lifecycle gate. Decode-sized
fusion remains a later, separately measured experiment.

### Pure-decode cohort quantum correction (2026-08-12 candidate)

Exact-main packed-artifact run `31660507971` exercised merge commit
`6ba436442a3fb86fe2e732913ebf237cb0d7fe0f` under continuous AC power and a
Nominal thermal settle/measurement window. Model identity, the 6,685-token
four-agent fixture, tool semantics, and 6,677-token recovery anchors were
correct, but the unchanged release gate failed closed when one cold semantic
response took 55.438 seconds against the 55.000-second limit. Server timing
put the first pair at 55.306 and 55.470 seconds and the delayed second pair at
cohort endpoints in the same range. This was a scheduling-performance miss,
not a cache or tool-semantic failure.

A same-binary, same-model, same-fixture discriminator changed only
`HF2Q_DEEPSEEK_SLOT_DECODE_QUANTUM` from 8 to 16. All four cold, cached,
automatic-tool, SSE, and tool-result conversations passed: cold semantic
responses were 50.520–51.691 seconds, the monotonic cohort wall was 51.841
seconds, and every continuation reused 6,677/6,685 tokens. The corresponding
server totals were 50.477, 50.865, 30.127, and 30.288 seconds. Thermal samples
remained Nominal throughout the cold receipts.

The candidate initially changed the default *pure-decode* slot quantum to 16
to amortize session swaps and scheduler publication after prefill drains.
`deepseek4_mixed_work_budget` still clamps any genuinely mixed prefill/decode
turn to eight tokens and two native prefill windows, so the 94,576-token
interactive-overlap contract is unchanged. The environment override remains
bounded to 1–64. This focused discriminator is causal evidence, not release
authority: two exact packed thermal waves and the complete cross-family
lifecycle gate must still pass before publication.

The exact-main packed gate at source
`2929a00145c6a59484fe54f0be8a88162ab5127c` then falsified 16 as a stable
default. All thermal samples were Nominal and three cold receipts completed in
53.745–54.601 seconds, but the fourth took 55.024 seconds and correctly failed
the unchanged 55.000-second limit by 24 milliseconds. Server completion times
were 53.620, 54.044, 32.009, and 32.189 seconds; cached work had already begun
with 6,677/6,685 tokens reused. This remained a cohort-tail scheduling miss,
not a cache, tool-semantic, power, or thermal failure.

Two fresh-server same-binary discriminators changed only
`HF2Q_DEEPSEEK_SLOT_DECODE_QUANTUM` from 16 to 32. Both completed the entire
four-agent cold, cached, automatic-tool, SSE, and tool-result sequence under
continuous Nominal thermal samples. Wave one cold semantic responses were
49.827–52.452 seconds; wave two was 48.859–51.485 seconds. Every continuation
reused 6,677/6,685 tokens, and every tool-result turn passed. The default is
therefore 32 for pure decode while `Mixed` remains clamped to eight. These two
discriminator passes provide repeatable causal margin, but they are not a
substitute for the exact packed two-wave and cross-family release gate.

The exact-main packed gate then falsified 32 as a stable default. Run
`31667493067` built crate SHA-256
`ded71757e823bcc6a24830332918b191959bb2bcbae2657a35fde285feda19f5`
from source `eec6900d501f656ae8bb3862ad4c31d0d1a1c8fe`, verified all three
canonical model digests, and measured the DeepSeek cold cohort under a
continuous Nominal thermal window. Three cold receipts completed in
52.497–54.435 seconds, but the fourth took 55.411 seconds and correctly failed
the unchanged 55.000-second limit. Every request used the exact 6,685-token
fixture with zero cold reuse; 6,677-token recovery anchors were captured and
cached requests had already begun successfully. This remained a pure-decode
cohort-tail miss, not a cache, tool, power, thermal, or artifact-identity
failure.

Two fresh-server discriminators then changed only
`HF2Q_DEEPSEEK_SLOT_DECODE_QUANTUM` from 32 to 64 on that exact packed binary
(SHA-256
`a2bd042ece84a5fb54059a7089d47a0bef9087599b3c62525f59aaa8b1db9187`).
Both completed the full cold, cached, automatic-tool, SSE, and tool-result
sequence under independent 60-second Nominal settles and continuous Nominal
measurement samples. Wave one cold semantic responses were 45.812–51.415
seconds with a 51.641-second cohort wall; wave two was 45.725–51.337 seconds
with a 51.503-second cohort wall. All eight cached, automatic-tool, and
continuation turns reused 6,677/6,685 tokens. The default is therefore 64 for
pure decode while `Mixed` remains clamped to eight. These repeated
same-artifact discriminators establish causal margin, but the new source still
requires its own exact packed two-wave and cross-family release gate before
publication.

The resulting exact-main source `a58932a830834c6dfa19e94a29dfb6ad956160d5`
passed hosted CI run `31670800480`. Its protected packed M5 run
`31670801462` verified all three canonical model digests, the exact 87,972-token
tool fixture, continuous AC power, and an entirely Nominal settle/measurement
window. Three DeepSeek cold receipts passed at 49.428, 51.522, and 53.554
seconds; the fourth completed correctly at 55.585 seconds and failed the then
literal 55-second limit by 585 milliseconds. All prompts were exactly 6,685
tokens with zero cold reuse, every 6,677-token recovery anchor was captured,
and cached work had begun successfully. With a 64-token quantum each 61-token
response already finishes in one pure-decode visit, so another quantum increase
cannot remove the remaining serial four-slot decode floor.

That exact failure triggered the checked-in same-input peer discriminator
rather than a favorable hf2q rerun. The pinned peer build 10326 (binary
SHA-256 `90bdf03673f7ee61d65d579a4e0be64a914edac1ccb23e74871040bc30d13543`)
ran alone against the same model and request bytes with prompt caching disabled.
After separate 180-second loaded-idle Nominal settles, both continuously
Nominal four-agent waves returned the exact zero-cache `read_file` calls. Their
cohort walls were 68.438 and 69.944 seconds (69.191-second median); manifest
SHA-256 is `d31164f1eef641b6db98d38f504e02b2da26ff5a80f3cc50f3f0e8a69d3f8052`.
The peer renders 6,695 prompt tokens, an explicitly receipted then-current
tokenizer delta of ten tokens from hf2q's 6,685.

The old unreceipted approximately 54.1-second peer number is therefore
superseded for current thermal authority. The protected hf2q cold ceiling is
60 seconds: 9.2 seconds below the current peer median and 4.4 seconds above the
valid exact-main tail. This is a source-bound rebaseline, not a waiver. Both
new exact-packed hf2q waves must still pass that literal ceiling together with
unchanged cache, tool, overlap, and cross-family gates before publication.

The local acceptance setup exposed an independent launcher false positive:
the remote-inference OpenCode process reported 10.5 GiB RSS, while macOS
`footprint` measured 1.7 GiB physical use and 9.5 GiB reclaimable mappings.
System memory remained 93% free with 2.4 GiB compressor use and 3.2 GiB swap.
The canonical launcher now refines only processes whose RSS crosses the 8 GiB
ceiling and falls back to that conservative RSS value if `footprint` is absent,
fails, exits during inspection, or returns malformed output. This changes no
model/cache budget and does not authorize co-resident inference runtimes.

### Cold-cohort thermal evidence boundary (2026-08-11 candidate)

Local exact-packed probes of hf2q source `db2d7750d0e2a5b2b364fe2324cca9334cdcf652`
used packed binary SHA-256
`3a4202c26b66c8dae1a4a2e8b9f6364b7906792cd0708f5b21005d7857ce4093`.
The paired large-prefill route engaged in every fresh process. Four diagnostic
waves published all four atomic cold receipts with zero reuse and maximum cold
semantic walls of 52.875, 52.848, 51.132, and 51.193 seconds. Their first
non-Nominal samples arrived only at 80, 79, 84, and 77 seconds respectively,
after the calibrated cold cohort had finished. The fourth attempt began after
304 uninterrupted seconds of unloaded-host Nominal samples, so residual heat
and the remote-inference coding client were not the cause. The combined
cold/cached/SSE/tool sequence itself eventually moved macOS to `fair`.

Those runs are diagnostic, not acceptance receipts: the then-current wrapper
defined the thermal envelope as the complete 24-request functional sequence
and correctly invalidated every non-Nominal run. They exposed an evidence
boundary that was broader than the comparative claim. The paired primitive is
selected only by large prefill; later cached, SSE, and tool-result turns prove
cache and semantic correctness under independent upper bounds rather than the
then-current 55-second cold comparison.

The candidate wrapper therefore preserves the exact workload and request
ordering but ends calibrated thermal measurement only after all four nonempty
`agent-*.cold.json` receipts exist. It does not pause agents: cached work may
still overlap the cold tail exactly as before. Any non-Nominal sample, probe
failure, telemetry gap, producer exit, timeout, missing receipt, or excess
receipt before that boundary fails closed. The thermal receipt records and
rehashes the four cold-receipt filenames and SHA-256 digests. The same live
server and KV sessions must then finish cached unary/SSE, automatic tool
choice, and tool-result continuation under the unchanged latency, reuse, and
semantic gates. Two thermally valid waves from the newly sealed exact package,
plus the 94,576-token overlap/lifecycle gate, remain required before this
optimization is accepted.

### Busy-affinity admission progress correction (2026-08-10 candidate)

The cross-family lifecycle gate reproduced the operator's long-session shape
with a 116,776-token DeepSeek prompt. The active request correctly reused
116,725 tokens, and an exact retry correctly selected the active slot as its
strictly strongest affinity. The worker then stopped making scheduler
progress: `Idle` cohort reconciliation treated any nonempty pending queue as a
reason to rerun admission while another physical slot was free. Because a
Busy-affinity retry is deliberately not admissible until its owner releases
the retained cache, the next admission pass selected nothing and immediately
reran again before `scheduler.step()`. A two-second process sample placed
essentially the entire slot-aware worker in repeated
`deepseek4_request_affinity` scans of the 116K-token prompt. Decode,
disconnect observation, checkpoint rollback, and shutdown were all starved.

Reconciliation now distinguishes a nonempty queue from a queue containing
runnable work. `Cold`, `Cached`, and `Control` requests may request another
admission pass; a Busy-only queue must fall through to exactly one scheduler
step. This preserves the strict active-prefix wait, lets the owner observe
cancellation and restore its request-local anchor, and makes the retry
runnable from that idle retained session on the next loop. Model-free tests
pin Busy-only no-rerun and the unchanged runnable behavior of the other three
affinity classes.

The rebuilt candidate then passed the exact long-context lifecycle after an
AC-power preflight, with binary SHA-256
`da970f10a3866048dfb1d2ce9f727e71c2aa31402374223265be3170cf1744bf`.
The base request rendered to 123,085 tokens and emitted the required
`lifecycle_probe` tool call. Its seed continuation reused 123,077 of 123,186
prompt tokens. An active 123,244-token stream then reused 123,193 tokens and
was cancelled without a terminal `[DONE]`; the queued exact retry immediately
reused 123,178 tokens rather than beginning a second cold prefill. A separate
50-token conversation reused zero tokens and returned `ISOLATION_OK`, binding
conversation isolation. The server subsequently completed its KV drain and
worker join cleanly. The atomic summary is
`/var/tmp/hf2q-cache-lifecycle.Casl3W/summary.json`, SHA-256
`67d18dec8594838ed1d40a55d4a524a8bdcd6fbe5ec1c48ecdceb99047ef2f56`.

This is exact-binary M5 Max correctness evidence for the repaired failure
path. Because the lifecycle harness does not continuously bind power state,
performance authority still requires a guarded AC-only rerun. Immutable
release authority additionally requires the clean committed artifact,
exact-SHA CI, packed-artifact validation, and the corresponding Qwen/Gemma
lifecycle gates.

### Client-order prompt serialization correction (2026-08-16 candidate)

An exact 6,673-token required-tool request exposed a serving-only prompt
divergence. The OpenAI request and the published DeepSeek encoder retain JSON
object insertion order, but hf2q's typed request path passed the tool schema
through a `serde_json::Map` configured to sort keys. The native encoder then
received `description, name, parameters` and sorted parameter keys instead of
the client's `name, description, parameters` and `type, properties, required,
additionalProperties` order. The altered prompt was 6,674 tokens and produced
the correct `read_file` operation only after 136 greedy tokens, exceeding the
checked 128-token agentic budget.

The source and quantized weights were not defective. The exact hf2q artifact
(SHA-256 `936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d`,
107,431,343,168 bytes) produced the correct operation under an independent
reference program, and hf2q's CLI produced it when given the exact published
prompt. Embedded and source tokenizers also emitted identical token IDs for
that prompt. Replaying hf2q's sorted prompt reproduced the 136-token path,
isolating the difference before model execution.

hf2q now enables `serde_json` insertion-order preservation for client-facing
chat data. A model-free regression deserializes the OpenAI request and requires
the rendered DeepSeek tool schema to retain the client's exact object order;
its `deepseek4` name places it under the existing blocking hosted-safe and
packed-artifact test filters. The supplied 22,976-byte published prompt
(SHA-256 `5f4b1444b317a5f27c6a45a4ea0d91790c648e78a43ed3cb2a8cc1ae4944a81b`)
also matches hf2q's render exactly, and the embedded/source tokenizer check
passes on the official artifact.

Release binary SHA-256
`9a10a31798f9f97a4f5662d7408060b2b3c34d55b59ab61ede72ea4a398109db`
(33,772,816 bytes) then passed the cold required call within 96 completion
tokens: 10.387 seconds of prefill, 13.284 seconds total, and the exact
`read_file` path. Automatic tool choice reused 6,665 tokens and completed in
3.128 seconds. SSE reused the same boundary, emitted one structured call plus
terminal `[DONE]`, and completed in 3.117 seconds. The tool-result continuation
returned the exact sentinel in 16 tokens while reusing 6,665 tokens, completing
in 6.108 seconds. This is local exact-artifact candidate evidence. Clean
immutable source, exact-SHA CI, and the protected packed-artifact hardware gate
remain the publication authority.

### Nested DSML schema repair (2026-08-16 candidate)

A stock client `question` call exposed a distinct nested-schema failure: the
model emitted a syntactically damaged questions array whose required `header`
was `null`, then repeated an ineffective repair intention. The close-time DSML
parser correctly rejected the damaged body, but generation had constrained a
non-string top-level parameter only to generic JSON. It therefore could not
prevent well-formed JSON with a schema-invalid nested value.

The DSML compiler now applies the same recursive schema rules used for nested
objects and arrays throughout the agentic grammar. Focused tests reject the
observed trailing-corrupt body, reject a well-formed questions array with
`header:null`, accept the same body with a string header, and reject invalid
token candidates using the exact 129,280-entry DeepSeek tokenizer.

The release binary then ran the exact 100.05 GiB Q2 artifact on an Apple M5
Max. A forced `question` request emitted one OpenAI tool call with the string
header `Video Type`, a string question, and three complete label/description
option objects. Its repeat reused 459 of 467 prompt tokens. The tool-result
continuation reused the same boundary, acknowledged the selected video type,
and stopped without another question. The SSE variant emitted 53 JSON chunks,
one `tool_calls` finish, one terminal `[DONE]`, and no null header or parser
error. This proves the repaired production failure path; clean immutable
source and the protected exact-artifact gate remain release authority.

### Prefill and structured-JSON surface correction (2026-08-16 candidate)

The nested-schema compiler was necessary but did not close the operator
failure. An exact 458-token stock-client-shaped `question` request still
produced a meaningless string at temperature zero. This was two independent
inference defects rather than a client, conversion, quantization, or transport
failure.

First, the dense prefix-attention kernel produced materially different first
logits from the scalar attention contract. On the exact request its three
highest token IDs were `271`, EOS, and `6328`; the gathered implementation
ranked `671` (`The`), `43` (`I`), and `128822` (`</think>`), matching the
coherent trajectory from the same artifact in a matched external runtime. The
dense path also failed a separate approximately 6K-token agentic prompt. It is
therefore retained only as a diagnostic oracle: every nonempty production
prefill and cached suffix now uses gathered attention. The 458-token gathered
run measured approximately 368--375 prompt tokens/s versus approximately 360
on the incorrect dense path, so the correctness repair did not trade away
prompt throughput.

Second, the recursive JSON grammar accepted only whitespace-free separators,
while the trained template surface places one ASCII space after commas and
colons. At the first nested `header` value, raw logits ranked token `582` (space
plus opening quote) at `37.73235`; the compact-only grammar masked it and forced
token `3305` (`\".`) at `12.709398`. The model was being compelled away from a
high-confidence coherent string. Native `tojson` rendering now preserves the
canonical spaced surface, and recursive grammars accept both canonical spacing
and compact JSON. A boundary regression proves the space-prefixed opening quote
remains sampleable, while null required strings and malformed nested bodies
remain rejected.

The canonical launcher also defaults `HF2Q_DEFAULT_REPETITION_PENALTY` to
`1.0`. Its former hidden `1.05` value distorted constrained strings and did not
prevent client-side action loops. Operators may still opt into a measured
non-default value; hf2q does not silently change a stock client's sampling
request.

The exact 107,431,343,168-byte artifact (SHA-256
`936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d`) then
passed `scripts/test_deepseek4_structured_tools.sh` at temperature `0.55`:
three required and three automatic `question` calls, three required and three
automatic `todowrite` calls, two-prior-null recovery for both tools, SSE, and a
tool-result continuation all emitted meaningful schema-valid strings. Repeated
requests reused 395--450 prompt tokens. The existing full agentic gate also
passed required/automatic tools, unary/SSE, tool-result continuation, exact
source arguments, and 6,666-token prefix reuse.

Finally, unmodified OpenCode 1.18.18 ran in an isolated configuration against
the candidate server. Its direct compatibility turn executed a valid two-item
`todowrite`, executed the requested shell command, and stopped; subsequent
steps reused 7,438 and 7,660 prompt tokens. The stronger checked-in coding gate
completed five tools on its first Rust repair turn and two on its continuation,
preserved one session, changed the source correctly, passed the immutable
behavior oracle, executed the named regression, passed tests after both turns,
and deleted the isolated session. This is local exact-artifact candidate
evidence; clean immutable source, exact-SHA CI, and protected packed-artifact
hardware replay remain publication authority.

### OpenCode loop, sampling, and recovery hardening (2026-08-17 candidate)

The observed repeated-call incident had two independent layers. OpenCode
1.18.18's native doom-loop detector reads only tool parts belonging to the
current assistant message. A tool result starts another assistant message, so
three identical calls spread across normal agent continuations are invisible;
interleaved reasoning or text also masks its same-message tail check. The open
upstream patch that counts matching calls anywhere in compacted history is too
broad for hf2q operations because legitimate nonconsecutive inspections would
become false positives.

The immediate machine mitigation therefore lives in Agentic Kit's managed
OpenCode lifecycle-plugin template rather than this Rust server. It tracks only
the trailing completed call within one user turn, canonicalizes recursively
reordered argument keys, and requires the tool name, arguments, and output to
repeat three times before aborting a fourth identical attempt. A changed call,
changed output, new user message, or different session resets or isolates the
streak; compaction and assistant continuations preserve it. Six focused plugin
tests cover the helper and deployed-hook abort path. This prevents the known
loop without claiming to replace an eventual tested upstream core repair.

On the hf2q boundary, DeepSeek already understood
`chat_template_kwargs.reasoning_effort`; generic OpenAI-compatible clients sent
the same setting at the request top level, where it was ignored. The request
schema now accepts `reasoning_effort` directly and merges it into the native
template context for DeepSeek only. Valid values remain `low`, `high`, and
`max`; an explicitly supplied compatibility kwarg retains its documented
precedence. Eight focused API/template tests pass, including top-level `max`
and compatibility precedence.

The canonical OpenCode profile and isolated coding gate now declare
`temperature=0.55`, `top_p=0.95`, interleaved `reasoning_content`, and a `max`
variant mapped to `reasoningEffort=max`. The structured-tool harness accepts
these parameters explicitly and records them in its result. The official model
card's code-agent comparison profile (`temperature=1`, `top_p=0.95`) remains a
required matched comparison rather than being silently substituted for the
locally proven lower-temperature starting point.

`scripts/test_deepseek4_reasoning_recovery.sh` adds an opt-in sanitized
incident-class replay. It calibrates an assistant-`reasoning_content` history
to 168K--178K actual server tokens, requires a cold structured tool call, then
checks a tool-result continuation and a following recovery turn for near-total
prefix reuse and valid reasoning/tool semantics. The script hashes its fixture,
request, and normalized recovery result and labels itself explicitly as a
reconstruction, not a byte-exact replay of private historical content. The
recovery turn has its own 512-token default because the incident-shaped history
causes the model to reconcile the archived "never repeat" rule before obeying
the explicit new recovery request. The first 128-token attempt failed closed;
an observational automatic-choice replay showed the correct DSML call beginning
at the 256-token boundary, and the 512-token required-choice replay completed
the call in 290 tokens. This is a fixture-budget correction, not a relaxation
of the fail-closed parser.

The rebuilt release binary produced the following local hardware receipts on
the M5 Max and exact 100.05 GiB agentic Q2 artifact:

- Identical seed `424242` produced byte-identical unary/unary and unary/SSE
  choices (SHA-256
  `49f33e6b0a4cf5273d7e3958b53ef06a1c156110fd148969d795b3153caa1073`),
  while seed `7` diverged
  (`f096b3353d1f7f2099a23fd3783a67e79d6e48a44b38c02f3eb5bfcf978a8ec5`).
  Four successful requests advanced the completed, prompt, and decode counters
  by exactly 4, 488, and 384. Top-level `reasoning_effort=max` was consumed;
  `bogus` failed with HTTP 400.
- Both matched structured-tool profiles passed every required/automatic,
  repeated-null recovery, SSE, continuation, and prefix-reuse assertion:
  local OpenCode `temperature=0.55, top_p=0.95` and the model-card comparison
  `temperature=1.0, top_p=0.95`, both with reasoning effort `max`.
- Stock OpenCode 1.18.18 passed the isolated coding gate in one continued
  session: five tools on the repair turn, two on the continuation, exact source
  mutation, immutable behavior oracle, named regression, and tests after both
  turns. The isolated session was deleted.
- The sanitized recovery fixture calibrated to 169,864 cold prompt tokens and
  completed in 537.558 s with zero cached tokens. Its tool-result continuation
  reused 169,856 of 169,962 prompt tokens in 4.034 s. The corrected recovery
  turn reused 170,045 of 170,053 prompt tokens and emitted the exact
  `inspect_file` call in 13.493 s. Fixture, cold request, and normalized final
  response SHA-256 values are respectively
  `71ce1f1c4a8dd83bd8a5e04f8dc8b28532703890512d537407bc4c2c1e8e12ef`,
  `469186318228ed79e4d31c25b6320b27c1de5dbfdcb427806d146e3bd5f2529a`,
  and `a495f462660d9970086dec328e1b2d80312b66a7dd23d15541110f37c3ac510f`.

These receipts close the local hardening candidate. They do not replace clean
immutable source, exact-SHA CI, or packed-artifact publication authority.

### Insertion-ordered release-fixture recalibration (2026-08-19)

Exact-main cache-lifecycle run `32282861299` exposed a stale test contract, not
an inference or thermal failure. Commit `d4874792` had correctly enabled
`serde_json` insertion-order preservation so the DeepSeek chat template sees
the client's tool schema in its original wire order. The immutable 21,204-byte
repository fixture, prompt-visible path, model artifact, and request semantics
were unchanged, but the corrected serialization renders to 6,684 tokens rather
than the historical key-sorted 6,685.

Four independent requests in the failed run all reported exactly 6,684 prompt
tokens with zero cold reuse. Three completed the required `read_file` tool call
before the first agent rejected the stale 6,685 assertion and the harness
cleaned up the fourth. Prefill completed for all four under entirely Nominal
thermal telemetry. The release fixture, receipt parser, publication verifier,
and current operator documentation therefore bind 6,684 tokens; older 6,685
measurements remain below as historical evidence for the superseded
serialization. No latency SLO or semantic acceptance predicate changed.

The protected gate now also generates the exact first-wave request and renders
and tokenizes it directly from the release GGUF before loading the 100 GiB
verifier. A future template/schema/tokenizer drift therefore fails during the
cheap preflight instead of after model startup and a partial four-agent wave.

### Immutable agentic prompt provenance v2 (2026-08-19 candidate)

Protected run `31854922028` passed with four 6,685-token prompts before
`d4874792` enabled `serde_json` insertion-order preservation. Runs
`32175556046` and `32272660422` then failed deterministically because all four
native requests rendered 6,684 tokens while the harness still required 6,685.
Exact-main run `32282861299` independently repeated that result for all four
requests; its first two cold completions also crossed the unchanged 60-second
acceptance ceiling, so it is diagnostic RCA evidence rather than a performance
receipt.
Their thermal samples remained entirely Nominal; the outer thermal failure was
only fail-closed fallout from missing accepted cold receipts. The isolated
semantic change and byte-identical context, builder, and template inputs bind
the one-token delta to JSON key order, not to model execution or thermals.

`full-context-agentic-v2` makes that ordering part of the checked-in workload.
The accepted policy is client JSON insertion order: tool function keys are
`name, description, parameters`, with parameter keys `type, properties,
required, additionalProperties`. Its exact prompt is 6,684 tokens. A complete
recursive lexicographic-key replay is separately rendered and must remain the
explicitly rejected 6,685-token legacy case. The contract binds all four
22,955-byte request bodies, rendered-prompt hashes, and little-endian token-ID
hashes:

| Agent | Request SHA-256 | Rendered prompt SHA-256 | Token IDs SHA-256 |
|---:|---|---|---|
| 1 | `f70f24bb875e0d99a8f1f6e3e15be3c8c69f55e09f9e0cc251a98dd24bf11f5e` | `ff031a247908832feefec530813161aee4debd2f22c20641f8afd5d2c6bdb9c2` | `daaada048f48d613f7e98181eae6c3849253147fed0181c3831a4cfba3de9a86` |
| 2 | `6b6892f56dc256ebc4388d39be905fa6dcf7c1533a3c7b3c0f7e5315d8301693` | `2ed766812723f87adaf8633e1c0c265e6c3a029c643d39e774a16dc485f6b851` | `566df8f92d7eb2076898a5163ace03224c2fe7c562dc962fedacb08755d5b0eb` |
| 3 | `0d7f2b983b24cdc1e7c634ba958d8ceba7dff8beb57f91e2b4b0bc1ccabafb0e` | `934db33f83e64e434ea7136dbb87429803f89e42ebbf84ed7842dda903187a71` | `367da8c22c59e326fac37d1851dbfd728c8ba06daf8312559d42fe7579ea5411` |
| 4 | `bec7e1161537279a7b85c9b7e28fd5546bdc048df650c1f53e2ed21b96dfd9d4` | `cb1a369a0dc99054b290841618071ad2e485e4a7adf8cd74b0b0dc38157cbd98` | `3356019bef7e6cd323db78ec64264feba1e7daa50fc9a5a1bf398172164d8f88` |

The tool result no longer reads the mutable, Cargo-normalized manifest from the
packed release directory. It is the exact `Cargo.toml` bytes requested by the
historical prompt-visible path at commit
`863ea423a4ec4a4e46fc4bcce41ef2f439214a83`: 8,912 bytes, 8,892 characters,
SHA-256
`10d0410c76313d1783e491e17760a0946704e35c1566a93363dcb009f396bbbd`.
The 43-byte success prefix plus that file is an 8,955-byte payload with SHA-256
`34826d9e2ced0f41f4d57bf873ac6c1d8c294955893ae5f4d0815457e28b3c3c`.
The contract requires the exact 6,676-token recovery anchor and 2,798-token
uncached continuation suffix across unary, automatic-tool, SSE, and
tool-result paths. This closes an unbound path/content drift where `$PWD` in a
packed crate named the right file but supplied changing normalized bytes.

Hosted negative tests now reject any changed policy, count, source digest,
request mapping, rendered/token hash, tool payload, agent identity, duplicate,
or swapped receipt. The protected gate additionally derives all counts and
hashes from the same contract, replays both serializers with the embedded GGUF
tokenizer, binds a four-agent prompt-provenance receipt into both waves, and
revalidates it after artifact download. This section remains candidate-only
until exact-SHA CI and both guarded four-agent waves pass. It does not waive or
widen the existing 60-second cold, 15-second cached/automatic/SSE, or 35-second
tool-result ceilings.

### Non-aligned D512 tail-load correction (2026-08-19)

The production gathered DeepSeek attention path exposed an `mlx-native` D512
kernel defect for key lengths not divisible by 64. QK and V simdgroup loads
read complete tiles before the validity mask was applied, so the final partial
tile could read beyond the logical allocation. The structural repair in
`mlx-native =0.10.12` populates partial matrices lane by lane, retains valid
values, and writes zero for invalid rows before matrix arithmetic. Tests cover
every modulo-64 tail for both four- and eight-simdgroup variants, guard-region
independence, and CPU-reference parity. Release tag and main commit are
`338392704d39786ae5f7a8145ac5e5f3fc087c81`; the crates.io and GitHub source
archive SHA-256 is
`a5791c12542c6232888c6d4391be62aad305974e92f61f70d2cddd7f21997355`.

A competing hypothesis that Metal silently truncates the sparse gather's very
large one-dimensional grid at 256 batches was falsified on the M5 Max. The
exact production gather body overwrote poisoned full buffers correctly at
255, 256, and 257 batches on repeated runs, and an independent coverage kernel
showed no missing threadgroups. No speculative grid reshape is part of the
accepted repair.

### Cooperative warm-suffix FFN/MoE prefill (2026-08-19 candidate)

The four-agent scheduler previously executed every warm matrix suffix as a
complete sequence-serial verifier pass. The candidate keeps attention and all
KV/compressor/indexer writes sequence-local, but packs the already available
lanes after each attention layer and executes the row-local FFN/MoE once over
the aggregate rows. It uses shared model weights as one ordinary matrix with
`M = sum(sequence rows)`; `mlx-native`'s batched-quantized-matmul API represents
independent batched weights and is not the correct primitive for this case.

The serving boundary is deliberately narrow:

- only already-installed warm cached matrix suffixes are eligible;
- cold prefill, mixed prefill/decode, incremental tails, recovery-anchor
  capture, final head publication, and decode retain the serial path;
- the worker never waits to form a cohort, never skips an older incompatible
  prefiller, and tries only the contiguous current FIFO prefix;
- the serial round-robin cursor advances only when the serial fallback really
  executes, so successful cohorts cannot skip a later serial lane;
- two, three, and four lanes receive at most 1,024, 640, and 512 rows each,
  keeping aggregate work at or below the existing 2,048-row transaction bound;
- all lanes have the same start/cursor, token count, recovery boundary, and
  reply class, otherwise the oldest request runs serially.

Each lane encodes its attention, packs the resulting rows, and drains the
command buffer before the combined FFN. All GPU work drains before one shared
supervisor gate. Cache commits validate every lane first, reject/poison all on
a stale peer or gate error, and then publish every opaque cursor ticket
infallibly. Serving pre-reserves token ledgers and validates every cache,
ledger, and request cursor before publishing any of them. Client cancellation
does not reject the shared gate: a lane that closes during submitted work is
made cache/ledger-consistent and recovered alone, while healthy peers keep
their committed suffix. Fatal cleanup owns every removed reply.

The protected benchmark compiles its release test binary before the Nominal
settle, launches it from a minimal `env -i` whitelist with no hf2q, MLX, Metal,
profiling, or scheduling override, and fixes the alternating pair count at
five. CI, packed-artifact, release-check, and publication builds also reject a
set `MLX_NATIVE_SKIP_METALLIB` before Cargo runs, because that build-script
override would irreversibly replace the embedded Metal library before the
clean runtime environment exists. Its completion trace is emitted only after
cache, token-ledger, request-state, and scheduler publication. The receipt
binds and independently rehashes the raw timing arrays, test log, measurement
log, and settle log; publication recomputes both medians and speedup, replays
the thermal timestamp/gap/phase validators, and rehashes and recounts each
four-agent wave's production completion traces.

Model-free tests cover the 2/3/4-lane width plan, exclusions, FIFO observation,
pre-submit oversize failure, four-cache success, supervisor rejection, stale
peer rejection, and prevalidated state publication. A full 43-layer spike on
the exact 107,431,343,168-byte artifact produced bit-identical two-lane states
and changed serial/cooperative 512-row timings from 3,384.780/2,649.979 ms on
the first measured run to a three-run median 2,718.067/2,277.239 ms, or
1.1936x. Those timings motivated production integration but are not release
authority. The landing test now requires nonzero warm prefixes, exact state and
logit equality, identical subsequent-token behavior for B=2/3/4 including the
2,048-row bound, and at least five alternating-order timing pairs with a faster
cooperative median and recorded process peak RSS. Protected agentic waves must
still pass every unchanged semantic, cache, cancellation, thermal, and latency
bound before this candidate is accepted.

Two protected exact-artifact attempts then falsified the original requirement
that this sustained microbenchmark remain Nominal for its entire lifetime. Run
`32319235539` passed exact parity and the performance test at `1.2296x` before
the host reached Fair near the end. After removing unrelated host work, run
`32327823594` on exact source
`4d14af2cf1a7a76b61996d4428966d253788deca` again passed every B=2/3/4 state,
logit, and subsequent-token comparison. Its five alternating pairs measured
serial milliseconds `[4915.009292,5293.279791,5318.741042,5301.899667,
5333.909916]` and cooperative milliseconds `[4136.835417,4133.733167,
4107.403,4271.480375,4175.0555]`: medians `5301.899667` versus `4136.835417`,
or `1.2816317625816729x`. The 144-second test logged 50 consecutive
Nominal samples before reaching Fair and recorded no Serious or Critical
sample. The old monitor then stopped at Fair, leaving a 23-second telemetry gap
before the terminal Fair sample, so that artifact cannot be promoted into a
revised-contract receipt. Thus the algorithm and product contract passed while
the all-Nominal measurement policy rejected the sustained load it was intended
to measure; a fresh protected run must prove continuous Fair-or-better sampling.

The reformulated thermal contract is narrow and matches the already accepted
Qwen3.8 sustained long-decode gate. Cooperative prefill still requires a
60-second uninterrupted Nominal settle with no model runtime and a Nominal
first measurement. It is then sampled every two seconds and may reach Fair;
any Serious or Critical sample, telemetry gap over five seconds, test failure,
or loss of exact parity remains fail-closed. The receipt records and the
independent verifier recomputes Nominal/Fair/over-limit counts and checks the
first and last phase labels. This changes neither the benchmark's positive
speedup requirement nor any DeepSeek cold, cached, SSE, automatic-tool, or
tool-result latency ceiling. Those product waves retain their separately
calibrated Nominal-only contracts. The exact B=4 decode gate remains exact and
requires a positive median, but later evidence below moved its sustained
measurement to the same Nominal-start, Fair-or-better contract.

### Four-agent cold handoff and exact warm B=4 decode (2026-08-19 candidate)

The product failure was scheduler latency, not a corrupt cache, stalled GPU, or
weak model artifact. In protected run `32299105258`, the unchanged 6,684-token
four-agent workload produced the correct `read_file` calls, but the first cold
pair spent about 26 seconds in prefill and completed around 63–64 seconds. The
second pair completed around 39–40 seconds. Two active cold prefills therefore
serialized the cohort enough for one correct response to miss the literal
60-second product ceiling. Expanding cold admission without controlling the
handoff then exposed a second failure: early lanes published immediately and
their cached continuations waited behind the remaining cold wave.

The accepted candidate keeps four distinct caches and independent arithmetic;
it does not concatenate requests into one semantic sequence. Its contract is:

- admit all four cold prefills and retain the measured 16-window adaptive
  matrix schedule for this 6,684-token prompt;
- keep cold unary decode on the 64-token bulk quantum, but keep warm and
  streaming fallback decode at no more than eight tokens;
- when a cold unary lane becomes terminal while another cold lane is runnable,
  retain its final scheduler tick and publish the cold cohort together;
- replay only a warm 1–8-token recovery suffix before admitting decode, which
  aligns four compatible cursors without blocking a large tool-result suffix;
- use the four-lane transaction only for exactly four warm unary lanes with
  identical cache plans and positions, installed live logits, and aligned token
  ledgers; every other shape uses the established serial path;
- pre-reserve all four token ledgers, prevalidate all four cache commits, drain
  one retained Metal command-buffer chain, pass one supervisor gate, and only
  then publish every cache cursor and next-logit row. Any submitted failure
  poisons the affected cohort rather than partially publishing it.

The smallest arithmetic spike used the exact 107,431,343,168-byte official
artifact, four deliberately permuted lanes, distinct supplied tokens, a
148-token prefix, and 132 subsequent steps. It crossed both ratio-four and
ratio-128 recurrent/cache boundaries and compared every F32 state bit, logit
bit, cache byte, cursor, and recurrent state against four serial executions.
All comparisons were exact. The transaction reduced the decode body from 92
command buffers and four synchronizations to 23 command buffers and one
synchronization. The isolated alternating benchmark measured a serial median
of 273.361 ms and B=4 median of 230.208 ms, or 1.1875x. A later noisier run was
still positive at 1.0732x; the protected exact-artifact gate therefore requires
a positive median rather than claiming the best sample as a universal gain.

Protected run `32332231049` on exact source
`5eb47c7851c448314f12826a414d599d98d409b7` then falsified the decode gate's
remaining all-Nominal assumption. The 208.74-second exact-artifact test passed
all 132 state/logit/cache/recurrent parity steps and preserved the 92-to-23
command-buffer and four-to-one synchronization topology. Ten alternating pairs
measured serial and cohort medians of `512.3692495` and `490.7976045` ms, or
`1.0439522214497687x`. After a separate 60-second Nominal settle, its
measurement recorded 54 Nominal samples before Fair. The old monitor stopped
at that first Fair sample, leaving 63 seconds before the terminal Fair sample;
it recorded no Serious or Critical sample but cannot be promoted into a
continuous revised-contract receipt.

The decode gate therefore now uses the same narrow sustained-load policy as
cooperative prefill and Qwen3.8 long decode: its own uninterrupted 60-second
Nominal settle, a Nominal first measurement, continuous two-second sampling
through Nominal or Fair, and fail-closed rejection of Serious, Critical, a gap
over five seconds, a test failure, parity drift, topology drift, or a
non-positive median. Its receipt and independent verifier recompute all state
counts and phase boundaries. This changes no agentic product-wave SLO or
thermal calibration.

Protected run `32336641261` then proved the arithmetic while exposing a
measurement-path defect. Cooperative prefill passed exact B=2/3/4 parity and
measured `5267.199` versus `4199.051666` ms, a `1.2543782308393248x`
speedup. The four-lane decode proof passed all 132 exact steps, retained the
92-to-23 command-buffer and four-to-one synchronization topology, and measured
`438.8917085` versus `344.2250415` ms, a `1.2750138879713084x` speedup. No
Serious or Critical state occurred. The workflow nevertheless rejected the
decode receipt because one sample interval was eight seconds, above the
unchanged five-second maximum.

The RCA was the telemetry implementation, not the model or Metal work. Every
sample launched `swift -e`, starting the Swift interpreter/compiler inside the
hot loop. Idle probes took about 111–127 ms, and the exact run recorded one
multi-second scheduling/compiler outlier under concurrent load. The accepted
correction compiles the checked-in Foundation helper once before any model
load, validates its first result, then reuses that executable. Twenty unloaded
samples measured 7.6–9.5 ms end to end; ten samples under 16 busy CPU workers
retained 2–3 second timestamp spacing with zero gaps. The helper remains
fail-closed on a missing compiler/source, compilation failure, non-executable
output, probe failure, or malformed state. Its exact private executable is
removed at process cleanup. The two-second target, five-second maximum gap,
60-second Nominal settle, and Serious/Critical rejection are unchanged.
Because the independently verified decode-cohort summary now binds the helper
source, compiler, compiler version, and executable digests, that summary uses
schema version 2; the underlying Rust benchmark receipt remains schema version
1 and is compared after removing only the summary's version and envelope
fields.

Exact-main hardware run `32344447013` proved that correction under the original
failure load: cooperative prefill retained exact B=2/3/4 parity and measured
`4757.599792` versus `3794.979875` ms (`1.2536561322344297x`) with 65
measurement samples and zero gaps. Four-lane decode retained all 132 exact
steps, the 92-to-23 command-buffer and four-to-one synchronization topology,
and measured `510.182875` versus `347.0996665` ms
(`1.4698454773652168x`) with 95 samples, 28 Fair samples, no Serious or
Critical sample, and zero gaps.

#### Predeclared `mlx-native` 0.11 B=4 conditioning experiment (2026-08-21)

The first protected `mlx-native` 0.11.0 remeasurement did not reproduce that
positive timing result. Run `32536030587` retained bit-exact arithmetic and the
92-to-23 command-buffer/four-to-one synchronization topology, but measured
serial and B=4 medians of `217.870` and `218.040` ms. Earlier raw receipts also
showed a large alternating-order signature: repeating one topology was fast,
while the first serial transaction after B=4 work and the first B=4 transaction
after serial work inherited different deferred buffer/residency costs. Because
the production scheduler executes up to the 64-token decode quantum on one B=4
cohort rather than alternating serial and cohort topology every token, the
current alternating microbenchmark may be measuring topology switches rather
than steady-state product work. This is a hypothesis, not an accepted fix.

Before another protected run, the experiment is fixed as follows:

- retain the 148-token prefix plus 132 exact serial-versus-B=4 steps, then add
  one state/logit/cache/recurrent exactness comparison at the actual 6,676-token
  benchmark anchor;
- use four serial and four cohort caches at logical capacity 131,072, preserve
  independent anchor snapshots for all eight, record weight/live-cache/snapshot
  resident bytes, and separate setup from timing with the existing 45-second
  loaded-idle interval;
- retain the old unconditioned alternating series only as a diagnostic. Its
  historical even/odd order signature is recorded but cannot accept or reject
  the experiment;
- for each measured arm, restore its anchor, execute and drain one untimed
  transaction of that same topology, restore the same anchor again, then time
  the identical token transaction. Record both prime and measured durations;
- execute 20 paired trials in alternating serial-then-cohort/cohort-then-serial
  order. Accept the conditioning hypothesis only if the overall conditioned
  median and both order-stratified paired-delta medians favor B=4. A ratio or
  paired delta equal to zero fails; no latency or topology threshold is widened;
- derive expected command-buffer counts from the artifact's declared layer
  count, retain nonzero dispatch/barrier and exact synchronization checks for
  every prime and measurement, and publish raw timing/topology evidence before
  asserting failure;
- the original protected-run environment gate required the target host's
  calibrated normal macOS memory-pressure signal for every sample and zero
  increase in the cumulative swapout counter. That preregistration was tested
  before any timing data was produced and was falsified by protected run 2:
  the 107,431,343,168-byte mmap artifact drove free memory from 89 percent to
  8--9 percent during load and the kernel signal changed from normal (`1`) to
  warning (`2`), while `Swapouts` remained exactly `47,238,488`, throttled
  pages remained zero, the test had not reached its loaded-idle marker, and no
  timing receipt existed. Repeating or incrementally widening that setup gate
  would not test the product's structurally memory-constrained operating point.

The memory policy is therefore amended, before observing any conditioned B=4
timings, from `darwin25-normal-no-swapout-v1` to
`darwin25-phase-bound-no-vm-churn-v2`. The benchmark acceptance criteria,
pairing, order strata, exactness checks, topology checks, 45-second loaded
settle, thermal envelope, and host-contention envelope do not change. Only the
environment observation is corrected:

- the Rust test emits fsynced, run-UUID/PID/sequence-bound process-start,
  loaded-settle-start, measurement-ready, and measurement-complete markers.
  The runner acknowledges readiness only after setup telemetry is complete;
  the Rust producer captures the VM baseline after that acknowledgement and
  captures the terminal VM state before emitting measurement-complete. Thus
  receipt and telemetry file I/O is outside the admitted timing window;
- setup may report memory-pressure level `1` (normal) or `2` (warning), because
  warning is structural for this artifact on the 128 GiB target. Level `4`
  (critical), any other value, throttled pages, swapout growth during setup,
  thermal state above Fair, foreign heavy work, or a changed boot epoch rejects
  the run. Free percentage remains diagnostic and has no threshold;
- over the exact Rust-captured measurement-ready to measurement-complete
  window, cumulative system `pageins`, `pageouts`, `swapins`, `swapouts`,
  `compressions`, `decompressions`, and `purges`, plus the test process's
  `ri_pageins`, must each have delta zero. This closes the clean mmap
  eviction/refault and swap read-back holes that a swapout-only policy misses.
  Counter monotonicity, unchanged boot epoch, unchanged page size, and boundary
  pressure in `{1,2}` are also required. Reactivations, wired pages, compressor
  occupancy, free percentage, and the sampled normal/warning distribution are
  recorded as diagnostics, not converted into invented tolerances;
- the receipt binds the effective mmap residency shape (`file_backed_bytes`,
  `anonymous_bytes`, and `mapped_segment_count`), requires file-backed weight
  bytes to remain the majority, and labels the result as a within-run paired
  comparison. Warning-admitted absolute latencies are not treated as
  interchangeable with earlier all-normal runs without a separate calibration
  experiment;
- the runner executes each thermal/contention/VM probe once before arming and
  buffers later samples in shell memory until measurement-complete. This warms
  the probe executables and removes telemetry log writes from the exact window.
  Probe execution still occurs during the window so a probe or any unrelated
  host activity that causes a real page-in fails the exact-zero rule; that is
  an intentional invalid-run signal, not an allowance to widen the threshold;
- protected run 4 reached `measurement-ready` with 114,929,848,668 tracked
  resident bytes on a quiet host, but macOS had moved from Nominal to Fair
  during the loaded settle. The runner rejected the run before acknowledgement
  or any timed trial. That falsified the runner's single-sample readiness
  implementation: the Rust producer already reserved a 300-second
  acknowledgement barrier for a loaded cooldown, but the runner sampled once
  and aborted. The corrected protocol keeps the model and production-capacity
  caches resident, requires an uninterrupted 30-second Nominal suffix in the
  setup log, and acknowledges only after a second Nominal boundary sample.
  The runner gives that cooldown 240 seconds from first marker observation,
  leaving roughly 57 seconds after observation lag for fail-closed cleanup
  before the producer's existing timeout. Fair remains valid during
  setup and measurement, but measurement must still start Nominal; no thermal
  or performance acceptance threshold is widened;
- protected run 5 proved the corrected cooldown and reached all 20 conditioned
  pairs with exact recurrent state/logits, exact residency shape, and exact
  topology. Conditioned B=4 measured 113.371 ms versus 117.409 ms serial
  (`1.0356x`), with both alternating order strata positive (`1.0380x` and
  `1.0320x`). The result is not accepted because its committed v2 policy failed:
  producer `ri_pageins`, system swap-ins, and system swap-outs were zero,
  pressure stayed Normal, and throttled pages stayed zero, but the exact Rust
  endpoints recorded 415 system page-ins, 25 page-outs, 8,864,476
  compressions, 8,869,594 decompressions, and 5,967 purges. The earlier
  68-second interpretation was wrong: that marker span included about 50
  seconds parked at the acknowledgement barrier. The enclosing sampled
  measurement interval was 18 seconds. At the 16 KiB page size, the balanced
  compressor traffic still represents about 145 GB, but its rate must be
  compared with a control rather than treated as ambient by assertion;
- the same run already contains that control. After `measurement-ready`, the
  loaded producer remained parked while the identical two-second wrapper probes
  continued. The unambiguous post-marker suffix from epoch seconds 1,787,384,657
  through 1,787,384,706 lasted 49 seconds and recorded 10,496 compressions,
  7,965 decompressions, 1,930 page-ins, 34 page-outs, and zero swapout growth
  under Normal pressure. The sampled measurement interval recorded 8,864,476
  compressions and 8,870,902 decompressions in 18 seconds. That is about 214
  versus 492,471 compression pages per second, a roughly 2,300-fold separation
  with process, artifact, probes, and time adjacency held fixed. System page-in
  rate was higher while idle, not while measuring. The wrapper and in-process
  compression deltas were identical; wrapper decompressions exceeded the exact
  endpoint by only 1,308 and correctly enclosed every exact global counter.
  This falsifies the probes and steady ambient activity as explanations for the
  bulk compressor cycle and identifies it as workload-correlated;
- a preregistered no-probe control was prepared before this existing control was
  recognized. Its first attempt failed closed during loaded setup when a foreign
  Cargo build started after host reservation; it never acknowledged readiness,
  opened the exact VM window, or produced an admissible result. The adjacent
  idle interval supersedes that spike because it holds the suspected probes
  constant on both sides and varies the workload directly. The failed spike is
  evidence, not landing code;
- v3 replaces the falsified global-zero rule with
  `darwin25-phase-bound-process-residency-v3`. It retains counter monotonicity,
  unchanged boot epoch/page size, boundary pressure in `{1,2}`, zero throttled
  pages, zero system swap-in and swap-out deltas, and zero process-scoped
  `ri_pageins` as hard conditions. Host-global page-ins, page-outs,
  compressions, decompressions, purges, and reactivations remain mandatory raw
  diagnostics whose endpoints and deltas are independently recomputed; they are
  not optional and receive no invented threshold. The runner's quiet-process,
  AC-power, Nominal-start, Fair-or-better measurement, and continuously Nominal
  30-second loaded tail rules do not change;
- every v3 summary permanently includes the exact post-ready/pre-ACK suffix of
  loaded setup memory telemetry as a hash-bound, non-gating idle control. The
  verifier reconstructs that suffix from the setup log, rejects a detached or
  missing selection, and requires the sampled measurement counters to enclose
  the in-process exact endpoints. Old v2 receipts fail the v3 schema/policy
  contract. Run 5 remains a v2 failure and justifies the amendment only; a fresh
  protected run from a committed v3 producer and verifier is required for
  acceptance;
- the 2026-08-22 v3 proof campaign preserved two invalid attempts rather than
  weakening the gate. Attempt 1 failed during loaded setup when the concurrently
  developed Qwen lane started Cargo and Rustc after DeepSeek had reserved the
  host. Attempt 2 failed on 15,980 pages of setup-phase swap-out growth while
  macOS was still reclaiming state from the first model load. An idle spike then
  showed pressure Normal, 90--91% free memory, and both swap directions flat for
  four consecutive samples over 40 seconds before the next attempt. Neither
  invalid attempt opened an accepted measurement receipt;
- the fresh committed v3 producer and verifier at source
  `006174ab87742e2fc1a457a9e7a8b04b826e5de4` then passed against the
  107,431,343,168-byte artifact with SHA-256
  `936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d`.
  Run UUID `7cf8081f-544b-47ca-b38a-984efc85cb87` completed all 20
  conditioned pairs with exact recurrent state/logits/cache and exact residency
  shape. Conditioned B=4 measured 115.419 ms versus 119.441 ms serial
  (`1.0348x`), with both alternating order strata positive (`1.0368x` and
  `1.0324x`). The exact 19-second VM window recorded zero swap-ins, zero
  swap-outs, zero process `ri_pageins`, zero throttled pages, and pressure
  Normal-to-Warning; all ten wrapper samples were uncontended and thermally
  Nominal. The mandatory diagnostic deltas were 8,488 page-ins, 141 page-outs,
  8,180,313 compressions, 8,262,262 decompressions, 10,120 purges, and 438,803
  reactivations. The non-gating loaded-idle control remained attached and
  hash-bound. The raw receipt SHA-256 is
  `b265f06c3653af5659b7a2fe173bb2393ab0eae5371574c84caa73d2ecfd66f3`;
  the verified schema-v6 summary SHA-256 is
  `4cd5c4743ae4971abee0173dda4934ec5ba85eec6265eda99217de414f39e424`;
- the verifier independently recomputes marker order and the at-least-45-second
  loaded-settle span, matches marker objects to the raw receipt, replays every
  gated-zero/epoch/residency rule from raw values, hashes all telemetry, and
  rejects missing, duplicate, truncated, reordered, wrong-run, or wrong-process
  evidence.

The passing fresh v3 experiment accepts the conditioned protocol and replaces
the switch-contaminated performance verdict. The positive-median requirement
was not waived. No result from a contended, thermally invalid, critically
memory-pressured, swapping, process-refaulting, or unreceipted run is
admissible.

Run `32344447013` then exposed a receipt-verifier defect rather than a hardware
defect.
The verifier required the literal substring
`official_artifact_b4_decode_body_is_exact_and_measured ... ok`, but Rust's
`--nocapture` output inserted the benchmark diagnostics between the test name
and libtest's later standalone `ok`. The synthetic contract had modeled the
unrealistic one-line shape. The corrected contract separately requires the
exact named test invocation, exactly one libtest result line, and an anchored
result proving one pass and zero failures. Its fixture now mirrors the real
interleaved log and rejects a missing test name, a failed final result, or
concatenated contradictory results. Every receipt-contract fixture now emits
the producer's schema-2 summary before testing its named failure condition, so
thermal negatives cannot pass merely by failing an earlier schema check. No
parity, topology, performance, thermal-state, sampling-cadence, or gap threshold
changes. A new exact-main hardware run remains required before release
authority is restored.

Several plausible alternatives were measured and rejected, and none remains
in the landing diff:

- a two-lane decode transaction was bit-exact over 132 steps but regressed from
  72.072 to 76.233 ms (0.9454x);
- 4,096-row and paired cold cooperative-prefill attempts either exceeded the
  M5 Max memory envelope or regressed the exact workload;
- prefill windows 14, 12, and 8, and a four-active window-14 schedule, still
  missed the cold bound (61.370, 60.979, 67.304, and 61.789 seconds);
- four-active/window-16 without synchronized handoff passed cold but missed
  cached latency at 19.856 and 17.199 seconds; terminal parking fixed cold at
  59.808 seconds but cached work still took 17.202 and 17.422 seconds;
- a global eight-token decode quantum, split warm/cold quantum without cursor
  alignment, and widened B=4 eligibility either missed cold or never formed an
  eligible cohort;
- cursor alignment without synchronized cold publication made server decode
  fast but left the first clients waiting 6.77 seconds at the cohort barrier,
  producing 19.268- and 18.454-second cached responses.

The reformulated hypothesis—four cold lanes, synchronized cold publication,
short-suffix cursor alignment, then exact warm B=4 decode—passed the complete
unchanged product contract on AC power with Nominal thermal state. Evidence is
under local artifact
`hf2q-b4-cohort-handoff-full.XXXXXX.kwNCHNYoIe`; its source-bound server log is
`/var/tmp/hf2q-b4-cohort-handoff-server.log`. The four cold responses were at
most 59.097 seconds, cohort wall was 59.489 seconds, and maximum cold TTFT was
52.413 seconds. Cached unary responses were 12.486–12.577 seconds, automatic
tool calls 12.520–12.525 seconds, SSE 12.619–12.923 seconds, and tool-result
continuations 22.178–30.575 seconds. Every required/automatic tool, exact
argument, SSE terminal, continuation, source-syntax, and retained-prefix check
passed; cached requests retained 6,676 of 6,684 prompt tokens. The log records
all four cold terminal lanes parking and releasing together, followed by
positive exact B=4 selection for cached and automatic-tool work.

The proof reuses the already-recorded immutable model identity
`936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d`;
it does not reread the 107 GB artifact merely to repeat that digest. Prompt
contract SHA-256 is
`c06dc12dd7b20cdbc7a17de87b09fbbe523ce898ce7c7fb7b31b2f502dfb1a9f`,
prompt-provenance SHA-256 is
`22c5c482515435db1521b9fe49c5d1c0a12007d1dbf4f0ff217a03e9e9e3d2aa`,
and repository-context SHA-256 is
`2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef`.
Publication still requires the packed exact-SHA binary to reproduce the B=4
parity/positive-median gate and both continuously monitored four-agent waves.
No 60/15/35-second ceiling is widened by this decision.

### Accepted single-tool terminal forward (2026-08-20 candidate)

Exact-main protected workflow `32350542331` reproduced a narrow release
regression after the four-agent handoff work above. All four 6,684-token cold
requests returned the exact required `read_file` call, but client response
times were 61.245–61.255 seconds and therefore failed the unchanged 60-second
product ceiling. The server spent 47.716–53.448 seconds in cold prefill and
then held the completed calls for synchronized publication; there was no cache
miss, GPU stall, malformed tool payload, or widened prompt.

The terminal path had two inconsistent contracts. Tool grammar compilation
already treated an omitted `parallel_tool_calls` field as `false`, matching
ADR-005's accepted single-call default, while `SamplingParams` treated the
same omission as `true`. After the single constrained DeepSeek tool body became
accepted, unary, slot-aware, and SSE decode also evaluated one more complete
model forward even though no later token could consume its logits. Stop
strings and the maximum-token boundary already avoid that final cache
mutation. The correction uses one effective default in grammar compilation and
sampling, preserves explicit `true`, and treats an accepted non-parallel
DeepSeek `ToolCallBodyAuto` or `ToolCallBodyRequired` runtime as terminal. It
never truncates an unaccepted, dead, trigger-waiting, response-format, or
parallel-call grammar.

The smallest tests pin both halves: omitted and explicit-false parallel calls
resolve identically while explicit-true remains enabled; and an accepted
single-tool grammar is terminal only under the constrained non-parallel
conditions above. The realistic product gate then proves that response parsing
still yields one exact tool call rather than accepting merely valid JSON.

Several adjacent hypotheses were measured before accepting this change and do
not remain in the implementation:

- ragged exact B=4 arithmetic was bit-exact, but scheduler variants with
  32-, 16-, and 8-token cold quanta still measured maximum cold response times
  of about 60.78, 60.57, and 63.22 seconds;
- exact B=3 arithmetic reduced 69 command buffers and three synchronizations
  to 23 and one, but its eight-pair median was only 1.0416x, below the
  predeclared 1.05x acceptance threshold and too small to own the failure;
- cross-slot cache forking was rejected by a tokenizer-only spike: the four
  6,684-token prompts share only 319 leading tokens. Their large common text is
  a suffix after the agent identity diverges and is therefore not a reusable
  transformer prefix.

After a continuously Nominal 60-second unloaded settle, the release binary
built from this candidate passed the unchanged complete four-agent contract.
Maximum cold semantic response was 59.360 seconds, cohort wall was 59.736
seconds, and maximum cold TTFT was 49.597 seconds. Cached turns retained 6,676
of 6,684 prompt tokens; maximum cached TTFT was 0.707 seconds, cached unary was
12.631 seconds, automatic tool calling was 12.854 seconds, SSE was 13.030
seconds, and tool-result continuation was 30.933 seconds. Every required and
automatic tool name, exact argument, SSE terminal, source-syntax check, and
tool-result continuation passed. The evidence directory is
`hf2q-terminal-forward-nominal.XXXXXX.AndTpexqvZ`, and the source-bound server
log is `/var/tmp/hf2q-terminal-forward-server.log`. This candidate evidence
reuses the previously recorded immutable model digest; it does not reread the
107 GB model merely to repeat that identity. Packed exact-SHA release gates
remain required before publication.

### Bounded required-tool reasoning and timer-free warm alignment (2026-08-20 candidate)

The accepted-terminal-forward correction above removed one provably unused
model evaluation, but it did not bound the model's reasoning before a required
tool body. On the exact single-tool release workload, DeepSeek can remain in
the forced-open reasoning phase until the 128-token completion ceiling even
though the required DSML grammar already determines the only legal next
action. At current decode rates, that uncontrolled phase consumes the narrow
margin below the unchanged 60-second cold-response ceiling. This is generated
work, not a cache miss or a stalled GPU.

The candidate gives the canonical DeepSeek launcher an eight-token operator
default only for the narrow path where a safe transition is mechanically
provable: DeepSeek reasoning is forced open, tool choice is required or names
the tool, exactly one tool is declared, parallel calls and logprobs are off,
and the caller did not supply the Qwen-only public thinking-budget extension.
Zero disables the operator default. Auto tool choice, multi-tool catalogs,
parallel calls, response formats, and other model families keep their existing
behavior. At the limit the slot emits the tokenizer-derived bare `</think>`
sequence, advances the suspended required-tool grammar with those exact bytes,
and then resumes normal constrained sampling. It never truncates a tool JSON
body or treats a partially accepted call as complete.

The same product trace exposed a separate warm-suffix scheduling defect.
Tool-result requests are submitted close together but not atomically. Three
warm matrix lanes can therefore commit one transaction before the fourth is
admitted, leaving the fourth exactly five native 128-token windows behind.
The ordinary cooperative selector then sees unequal cursors and falls back to
serial work for the rest of the suffix. The timer-free correction selects the
lowest compatible warm lane, uses the existing serial planner to prove that a
bounded catch-up lands exactly on the leading cursor, advances only that
lane, and then lets the normal widest-cohort selector resume. Cold work,
recovery tails, unequal recovery anchors, non-window-aligned deltas, and the
planner's recovery shrink band remain ineligible. No timer, sleep, request
concatenation, or cross-sequence cache sharing is introduced.

A continuously Nominal diagnostic run under
`/var/tmp/hf2q-deepseek-alignment-budget8-quiet.X2zx3h` passed all four
cold/cached/automatic/SSE/tool-result/source-tool conversations. All four cold
semantic responses completed in 59.690 seconds; cached unary was 9.297
seconds, automatic tool calls were 12.999--13.103 seconds, SSE tool calls were
9.288--9.669 seconds, tool-result continuations were 27.062--30.173 seconds,
and source-tool calls were 11.601--11.861 seconds. Every warm replay retained
6,676 of 6,684 prompt tokens. The server recorded all 16 eight-token budget
resolutions and forced closes, one exact `6,676 -> 7,316` catch-up with
`window_cap=5`, and a following four-lane cooperative transaction at cursor
7,316. The protected wave now rejects a run unless it observes those budget
and scheduler transitions rather than inferring them from a final HTTP pass.

A later canonical-launcher diagnostic proved that the launcher selected the
eight-token default, but is deliberately rejected as performance evidence.
An unrelated conversion had left roughly 10 GiB of swap in use and another
foreign `hf2q` process appeared at the measurement tail; cold response was
65.742 seconds. The failure does not authorize a wider latency limit. Release
measurement must begin only after the existing 60-second thermal settle and a
new process-group-scoped contention check prove that no foreign compiler or
model runtime is active. The model identity continues to use the cached,
unchanged-file verification receipt; these reruns do not reread the 107 GB
artifact merely to restate its SHA-256.

Model-free validation after the exact planner hardening passed 134 DeepSeek
tests with zero failures and eight hardware-only tests ignored. Publication
still requires a clean packed exact-main binary to pass both guarded product
waves plus the independent cooperative-prefill and decode-cohort gates. No
60/15/35-second product ceiling is widened by this candidate.

## Historical agentic revalidation (superseded, 2026-08-05)

This section records the rejected 89.65 GiB Q2_K_S artifact and the defects that
motivated the mixed profile. It is not the current acceptance result.

The earlier short-prompt acceptance did not represent OpenCode. A source-bound
revalidation used a real repository prompt, OpenAI tool definitions, a required
tool call, a tool-result continuation, SSE, and a growing live prefix. The
following results supersede any readiness or H4-pass statement in the
historical ledger below:

| Check | Revalidated result |
|---|---|
| Reproducible dependency | The former `mlx-native` revision `7cc3d30` rejects the second 4K matrix transaction with `incremental calls require seq_len=1`. hf2q now pins the immutable crates.io release `mlx-native =0.9.6`, published from `7b05016b1bc2b4cce06bb0c4336abf8bded1c394`. Its registry checksum and downloaded crate SHA-256 are both `7a91af027a38c1cf606f00a39da90ba2353843b155529b4cd4d00c6d29f7b015`. |
| Dependency parity tests | From the clean release source: 320 library tests passed; five compressor, three sparse-mask, and two F16/BF16 D512 skip-map-with-sinks tests passed; the optimized bit-width experiment passed; and `cargo package --locked` verified the 525-file crate. An external `/opt/llama.cpp` peer benchmark was corrected to remain explicit/ignored rather than making default crate tests depend on a machine-local checkout. |
| Readiness warmup | The server now exercises matrix prefill and the vocabulary head before binding readiness, then resets logical cache state. Load-to-ready increased from about 20.2 s to 25.3 s, moving first-use pipeline work out of the first request. |
| Realistic cold tool request | 8,957 prompt tokens, 25.558 s TTFT, 350.46 prompt tok/s, 29.505 s total, `cached_tokens=0`. This is responsive rather than hung, but not yet an agentic acceptance pass. |
| Exact-prefix reuse | Repeating the request reused 8,949 of 8,957 tokens and reduced prefill to 1.138 s. A 9,056-token tool-result continuation reused the same 8,949-token prefix and prefetched only its suffix in 2.234 s. |
| OpenAI/SSE protocol | Unary and SSE both emitted structured `tool_calls`, `finish_reason: "tool_calls"`, and a terminal `[DONE]`; no raw DSML leaked. |
| Checked-in agentic gate | `scripts/test_deepseek4_agentic.sh` uses a unique run ID and deliberately withholds the requested manifest until the tool result. It requires a zero-cache first request, near-complete reuse on repeats and tool results, bounded wall-clock semantic response latency, reconstructed valid SSE tool-call JSON, matching call identity/arguments, `finish_reason: "tool_calls"`, usage, and exactly one terminal `[DONE]`. The 128-token completion budget prevents a valid but slightly longer DSML call from being mistaken for a grammar failure. The gate remains fail-closed on the Q2 tool-semantic defect below. |
| Tool semantics at Q2 | Failed. Required-tool runs requested paths including `/home/robertl/agent/coding/1`, `/opt/hf2q/README.md`, and, after removing the initially embedded manifest from the fixture, `/cargo/1.0.0.1/daniel/hf2q/cargo.toml` instead of the explicit `/opt/hf2q/Cargo.toml`. One separate cold SSE run chose the correct path, demonstrating inconsistent artifact quality rather than a stable agentic pass. Automatic tool choice also failed to produce a reliable native call. |
| Long-prompt throughput | A controlled 18,526-token candidate improved from 95.57 s / 193.85 tok/s at 512-row chunks to 70.52 s / 262.70 tok/s at 4K chunks. The matched peer reference was 361.89 tok/s, so hf2q reached about 72.6% and failed the H4 85% prompt-processing floor. |
| Decode throughput | A warmed 6.2K agent prompt decoded at about 16.35 tok/s versus about 38.58 tok/s for the matched peer run. H4 decode parity is therefore reopened. |
| Quality reference | The peer on the same Q2 artifact also emitted malformed/non-actionable tool content in automatic and required modes. That implicates the aggressive quantization, but does not relax hf2q's agentic correctness gate. |

DeepSeek-V4-Flash Q2 is therefore an experimental serving target. “Ready” now
means the process and Metal pipelines are initialized; it does not mean the
artifact has passed the agentic quality or peer parity gates. Release
acceptance requires a realistic multi-turn coding fixture to produce the right
tool call, consume its result, reuse the unchanged prefix, stream a timely first
semantic token, and meet both H4 throughput floors.

## Evidence ledger

Rows below are retained as historical implementation evidence. Where they
conflict with the agentic revalidation table, the revalidation table governs
current readiness.

| Evidence | Result |
|---|---|
| Hardware/storage audit | 128 GiB M5 Max; about 1.8 TiB free |
| Official repository metadata | 48 shards; about 166.9 GB; 304.18B params |
| Pinned `hf download --dry-run` | 74 official files; 166.9 GB; no payload fetched |
| Official source download | 73 receipt-bound files, including all 48 shards; 166,898,659,555 bytes |
| Exact source bundle | `a8544e6469f8f392e72f953e9a2b4ee33a23c50a859f47dd354d37ab0093993d` |
| hf2q Q2_K_S converter | Rust in-process; FP8/FP4/E8M0 ingest, bounded expert fusion, atomic provenance receipt |
| mlx-native Q2_K loader/matvec | Dense + expert-ID Metal paths; exact routing, activation, sparse-attention, compressor, indexer, HC, and tail-RoPE primitives |
| Q2_K decode microbench | 55.81 us / 98.63 GB/s at M=1, N=K=4096 (integration rerun) |
| Pinned peer conversion and graph | Present at reference commit |
| Pinned peer oracle binaries | Rebuilt locally as version 10276 (`6ea215d17`) |
| Synthetic converter proof | Positive/negative dtype, shape, scale, fusion, receipt, and round-trip suites green |
| Official full conversion | Passed from pinned source with hf2q converter commit `a8e00a24c1ac043182761e9df3347853b2d74d41` |
| Official output | 96,265,459,008 bytes (89.65 GiB), SHA-256 `0318b99b4ece1222d8cf4d93a705458d339907910af5af3a175bc3989dcb01a1` |
| Conversion telemetry | 1,823.71 s wall; 120,490,524,672-byte max RSS including file mappings; 5,196,469,928-byte peak footprint; zero process swaps |
| Bounded working vectors | Receipt maximum 4,670,627,840 bytes; 529,530,880 F32 elements in the largest row-aligned chunk |
| DSpark boundary | 4,705 source tensors explicitly excluded from the base GGUF and receipt-marked for a separate draft artifact |
| Official GGUF catalog | Strict metadata and all 1,328 verifier tensors validated exactly |
| Native primitive + serving regression | 71 DeepSeek-focused tests passed; 0 failed; three real-artifact hardware tests ignored by default |
| Pinned compute dependency | `mlx-native` commit `7cc3d308c37161e6602c9218ad3a14b5f86d7d4a`; pushed, clean-checkout build verified, and pinned exactly in `Cargo.lock` |
| Clean dependency regression | 27 mlx-native tests passed across indexer, MoE routing, Q2_K, sparse-prefill-mask, and dense-GEMM suites |
| Official activation simulation | Owned Metal E4M3/E8M0 main-KV and Hadamard+E2M1/E8M0 indexer paths match exact BF16 CPU references; 3/3 focused tests passed |
| Compressed cache ownership | Fixed-offset contiguous raw/compressed KV views plus exact main/indexer F32 recurrent states; compact rollback snapshots copy the overwritten circular windows and recurrent state without aliasing, while append-only compressed/indexer rows are hidden by restored logical position; the 1M-token live-cache plan is 7,232,045,056 bytes and the official compact snapshot is context-independent at 17,842,176 bytes |
| Official native residency | 96,265,327,964 resident weight bytes; 128-token cache admitted; zero process swaps |
| Official uncompressed-prefix proof | Layers 0-1 Q2_K attention plus exact hash-routed/shared MoE passed transactionally in 28.36 s cold; 99,047,478,968-byte peak footprint |
| Official compressed-prefix proof | Layers 0-3 passed through ratio-4 and ratio-128 attention plus hash/learned MoE in 25.25 s after load/build; 68,298,014,720-byte max RSS; zero swaps |
| Official full-verifier proof | All 43 layers passed for one token in 25.52 s after load/build; 67,200,876,544-byte max RSS; zero swaps; finite nonzero `[1, 4, 4096]` state and transactional cache publication |
| Partial-token failure safety | Any failure after verifier execution begins poisons the request cache; reset atomically clears recurrent state, cache visibility, and poison before replay |
| Official vocabulary-logits proof | Owned HC collapse, final RMSNorm, and Q6_K output projection passed with the 43-layer verifier in 27.81 s; 62,708,498,432-byte max RSS; zero swaps; finite nonzero `[1, 129280]` logits |
| Official tokenizer parity | GGUF-driven Rust GPT-2 BPE emitted the same IDs as the pinned source `tokenizer.json` for the fixed Unicode, numeric, punctuation, and 0731 prompt-atom corpus |
| Coherent real-model inference | Rust-native `hf2q generate` rendered the 0731 chat prompt, crossed the position-3 ratio-four compression boundary, selected the greedy token on Metal, and decoded `Hello` for the fixed six-token prompt |
| One-token reference parity | Pinned peer `llama-simple` on the exact same GGUF and rendered six-token prompt also greedily decoded `Hello`; no product path invoked the oracle |
| Native inference telemetry | 20.80 s load; 8.15 s incremental six-token prefill (0.736 tok/s); 30.38 s total; 67,364,585,472-byte max RSS; zero swaps |
| Reference inference telemetry | Pinned peer raw-completion oracle processed the same six tokens in 73.89 ms (81.20 tok/s) after load; not yet an H4 comparison because hf2q currently submits one-token graphs while the reference batches all six |
| Official arithmetic coherence | Prompt `What is 2+2? Answer with only the number.` rendered to 17 source-parity token IDs; greedy output reasoned to `2+2 = 4` and terminated with final answer `4` |
| Extended-context execution | Fresh prompts matrix-prefill up to 1,024 rows beyond the 128-row physical window, then extend exactly; cached suffixes retain exact token-wise extension; server default 131,072 tokens with a metadata-enforced 1,048,576-token ceiling |
| Matrix-bound artifact proof | A fresh 1,036-token request crossed the 1,024-row matrix bound, completed its exact incremental tail coherently, and reached 356.03 prompt tok/s / 2.910 s TTFT on the warm shape |
| Matched fresh-prefix prefill benchmark | Identical 306-token greedy request, warm shape, `cached_tokens=0`, three trials: committed baseline 5.4951/5.4957/5.4905 s (median 55.69 tok/s); matrix-prefill candidate 1.0004/1.0067/1.0035 s (median 304.94 tok/s), a 5.48x TTFT speedup |
| Trivial-prefix policy proof | A one-token recovery-anchor hit followed by a 305-token suffix resets to `cached_tokens=0`; two real-artifact trials prefetched in 1.0034 and 0.9572 s instead of taking the incremental path |
| Artifact-backed tests | Exact 1,328-tensor catalog, all 43 verifier layers plus finite vocabulary logits, greedy token selection, and embedded tokenizer parity passed against the 96.3 GB artifact |
| Final native decode sample | 56 generated tokens in 1.24 s (45.308 tok/s) on the official arithmetic prompt after the clean pinned-dependency release build |
| Matched five-run native benchmark | 45.1, 45.1, 37.7, 45.2, and 45.1 tok/s; median 45.1 tok/s, p95 45.2 tok/s; 63 verifier evaluations for 64 generated tokens; warm-prefill median 74.6 tok/s |
| Matched peer reference | 41.58 tok/s on the same artifact and benchmark contract; hf2q median is 1.085x (+8.5%) |
| Performance parity | Passed: native median exceeds the H4 0.90x decode floor and the pinned peer reference while retaining coherent greedy output |
| Native OpenAI server load | Real 96.3 GB artifact loaded in 19.92 s as `DeepSeek-V4-Flash-0731-Q2_K_S`; `/v1/models` reports `deepseek4`, Q2_K, 256 experts/6 active, native MLX, and the configured 4,096-token validation context |
| Exact growing-turn cache | Final real-artifact unary check reused 16 of 27 prompt tokens; only the 11-token suffix ran, with 0.276 s TTFT; an earlier 37-of-48-token coding turn prefetched its suffix in 0.287 s |
| Canonical reasoning recovery | Thinking-mode follow-up with old `reasoning_content` removed restored the native recovery checkpoint (`cached_tokens=4` of 21 on the deliberately tiny prompt) rather than replaying the whole prefix |
| Required tool call | Real model returned OpenAI `read_file` with arguments `{"path":"/tmp/example.txt"}` and `finish_reason: "tool_calls"`; no raw DSML leaked |
| Constrained tool performance | Greedy candidate-ranked grammar enforcement improved the same 51-token tool call from 2.9 to 25.4 tok/s while preserving the structured call |
| SSE tool + cache | Streaming response emitted role, indexed `tool_calls`, `finish_reason: "tool_calls"`, usage, and `[DONE]`; it reused 290 of the 298 rendered tool-prompt tokens |
| Unsupported surfaces | DeepSeek-V4 embeddings and multimodal injections fail explicitly; no alternate family/runtime fallback is selected. Slot-aware text generation is supported with independent full-context sessions and shared physical KV admission. |
| Full binary regression | 3,814 passed, 0 failed, and 44 ignored in 173.97 s after the matrix-prefill/cache-policy change |
