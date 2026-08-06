# ADR-042: DeepSeek-V4-Flash-0731 — Rust-native source conversion and MLX inference

- **Status:** Accepted for hf2q 0.1.1
- **Updated:** 2026-08-06 — file-backed GGUF weights removed the anonymous
  100 GiB copy; packing eight physical heads into each D=512 sparse-attention
  tile raised the exact 120K cold-prefill gate to 331.53 tok/s. The required
  tool call remained exact, the next turn reused 99.91% of its prefix with
  1.289 s TTFT, and the process completed without the former memory-pressure
  kill. The final `mlx-native` 0.10.2 view-semantics revalidation reused 6,254
  of 6,262 prompt tokens with 235 ms cached TTFT
- **Owner:** hf2q integration lane
- **Source model:** `deepseek-ai/DeepSeek-V4-Flash-0731`
- **Pinned source revision:** `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- **Reference implementation:** `/opt/llama.cpp` at
  `15586e2d7165570fb3aa7c26e0d442e289ef69de` (build 10298)
- **Target host:** Apple M5 Max, 40-core GPU, 128 GiB unified memory

## Decision

hf2q downloads the **official Hugging Face checkpoint** and converts it to the
owned `deepseek4-agentic-q2` mixed profile. Expert gate/up tensors use Q2_K,
expert and shared down tensors use Q3_K, and the token/output, HC, indexer, and
compressed-attention context paths use Q8_0. Conversion, quantization, loading,
and inference are implemented in-process in Rust with owned Metal kernels in
`mlx-native`.

The product may invoke `hf`, `wget`, or `curl` solely to fetch official source
repository files. It must not invoke Python, llama.cpp, or another converter,
quantizer, or inference runtime. A separate developer-only parity harness may
invoke the pinned llama.cpp build as an oracle.

Prebuilt quantized weights are not an input, fallback, cache seed, or release
artifact. Their published sizes may be used only as non-authoritative capacity
evidence.

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
invocations in one DSML block. DeepSeek-V4 embeddings, multimodal inputs, and
the slot-aware concurrent scheduler are explicitly rejected rather than routed
to another model family.

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
llama.cpp graph also scans the valid lightning-indexer history. The new interval
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

### H4 — owned low-bit kernels meet or exceed llama.cpp

The Rust/Metal Q2_K path matches a scalar Rust decoder within the declared
numeric tolerance, and the complete runtime reaches at least 1.00x the pinned
llama.cpp decode and prompt-processing rates under the same artifact, prompt,
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
pinned llama.cpp registry, including:

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
loaded from llama.cpp at runtime.

### Agentic serving and cache contract

The DeepSeek worker owns a single serialized live session. It records the exact
rendered token sequence corresponding to the native cache position and selects
the longest safe prefix for each request:

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
unified-memory headroom for Metal scratch. The previously validated 131,072-
token gate remains the performance baseline until a larger real prompt crosses
the first growth boundary. Matrix prefill remains at the measured 2,048-token
transaction while physical capacity is 131K and drops to 1,024 tokens after
growth unless an operator supplies an explicit benchmark override.

Tool definitions are rendered by the owned 0731 encoder. Grammar-constrained
generation emits one official `<｜DSML｜tool_calls>` envelope containing one or
more invokes, validates required parameters, and converts the completed block
to OpenAI `tool_calls` objects for both unary JSON and SSE. Raw DSML framing is
not exposed to API clients. String parameters admit ordinary source-code angle
brackets. An earlier `[^<\\]` grammar rule forced calls containing Rust generic
or lifetime syntax such as `fmt::Formatter<'_>` to close immediately before
the `<`, yielding valid but semantically truncated tool arguments.

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
- Token/logit parity is measured against the same source-bound llama.cpp build.
- Three-run median prompt/decode throughput and peak memory meet H3/H4.
- Existing public API and model-family regression suites remain green.

## Benchmark discipline

The parity harness is outside product code. It records exact hf2q, mlx-native,
llama.cpp, source-model, and artifact commits/hashes; prompt bytes; context and
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
| Native dependency | Exact crates.io pin `mlx-native =0.10.2`, published from implementation commit `9b496c475f2200eb968bafc861dba6f65dade0e6`; registry checksum and downloaded crate SHA-256 `e1d02b2f8f401bf6afe100144884b5d76c2c1154ee7c9a79299856f4aee0506e` |
| Quant plan | 1,328 verifier tensors: 172 Q2_K, 86 Q3_K, 532 Q8_0, 535 F32, and 3 I32; 4,705 DSpark tensors explicitly excluded from the base artifact |
| Conversion bound | Rust-native row-aligned streaming; maximum working vector bound 4,798,873,600 bytes; no external converter or inference process |
| Arithmetic coherence | Greedy `What is 2+2? Reply with only number.` returned exactly `4` |
| Tool semantics | The curl/OpenAI-compatible harness made required and automatic choice both select `read_file` with exactly `/opt/hf2q/Cargo.toml`; unary and SSE returned valid OpenAI `tool_calls` and `finish_reason: "tool_calls"`. The source-argument regression also returned `fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result` byte-for-byte in one `emit_source` call (4 s response), proving the formerly truncated `<` syntax through the real model. A real OpenCode two-turn run issued five tool calls on its first turn and two on its second, changed the requested source, and passed the named oracle/regression checks in the same session. |
| Tool-result continuation | With tools enabled in `auto` mode, the model consumed the Cargo result and returned the requested sentinel without another call. The real OpenCode continuation consumed prior tool results, issued the next required calls, and reused the same live session. |
| Prefix reuse | The current checked-in gate used a 6,262-token prompt and reused 6,254 tokens on repeated, automatic, SSE, and post-tool turns. Cold TTFT was 12.380 s and cached TTFT was 235 ms; every required/automatic tool, source-argument, tool-result, unary, and SSE assertion passed. |
| Canonical launcher | `scripts/serve_deepseek4_opencode.sh` passed the curl agentic gate, the real OpenCode coding run, and the 120K cache gate. It advertises 524,288 tokens and demand-allocates 131K initially. Memory/port preflight now refuses an unsafe 100 GiB load before model mapping. A prompt crossing the initial 131K allocation and a near-512K physical allocation remain unproven on this 128 GiB host. |
| Ordinary agentic prompt | On the same approximately 5.9K-token README coding prompt, hf2q warm prefill was about 518 tok/s and median decode was 32.1 tok/s; llama.cpp build 10293 reported 399.4 prompt tok/s and 31.7 generation tok/s. |
| 120K cold prompt | `scripts/test_deepseek4_long_context_cache.sh` produced the exact required tool call for 119,808 prompt tokens in 361.384 s TTFT (331.525 tok/s); decode was 21.742 tok/s. The same source-bound prompt and artifact under llama.cpp build 10298 processed 119,807 tokens in 749.015 s (159.953 tok/s) and decoded at 19.565 tok/s. hf2q was 2.07x the reference prompt rate and 1.11x its decode rate for this run. |
| 120K continuation cache | Appending the real tool result produced a 119,907-token request that reused 119,800 tokens (99.91%), evaluated only a 107-token suffix, returned its first semantic event in 1.289 s, and emitted the exact requested sentinel. |
| 98K OpenCode-scale revalidation | A fresh 97,127-token required-tool request completed cold prefill in 424.522 s (228.79 tok/s). Its 97,214-token tool-result turn restored the compact recovery anchor, reused 97,119 tokens (99.90%), evaluated a 95-token suffix in 1.378 s TTFT, and completed in 2 s. |
| Before/after control | The earlier identical-class hf2q run required 594.575 s for 119,808 tokens (201.502 tok/s). The accepted path reduced cold-prefill time by 39.2% and increased its token rate by 64.5%. At the shorter exact 26,024-token gate, the threshold retune improved 388.846 tok/s to 497.277 tok/s while retaining the exact tool call; the cached continuation reused 26,016 tokens and reached 927 ms TTFT. |
| Output parity | Both runtimes returned the exact requested comma-separated sequence. llama.cpp also returned the exact required `read_file` path on the long repository prompt. |
| Memory safety | A 4,096-row prefill command buffer produced Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory`; the accepted 2,048-row transaction completed the 119.8K gate. A later OpenCode session falsified the steady-state claim when macOS killed hf2q at 108,840 MB after the shared transient arena retained cold-prefill buckets. The split-arena build releases prefill scratch before decode; the new file-backed build released 4,933,917,968 transient bytes after the 120K prefill, remained alive through its cached continuation, reported 2.0 GiB RSS, and shut down cleanly. Eagerly allocating the full 524,288-token cache beside this artifact still OOMs, so demand growth remains required. Only one 100 GiB-class runtime was resident during every comparison. |

The performance result comes from two measured defaults. Decode groups two
verifier layers per Metal command buffer; one layer reached 29.49 tok/s, two
reached 33.10, four plateaued at 33.05, and eight regressed to 32.80. The Q8_0
matvec uses llama.cpp's peer geometry (`N_SG=4`, `N_R0=2`); enabling it raised
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
then appends its tool result and fails unless almost the entire prefix is
reported as cached with bounded continuation TTFT. This is the release guard
against the original agentic failure mode: recomputing the whole conversation
on every turn.

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
| Long-prompt throughput | A controlled 18,526-token candidate improved from 95.57 s / 193.85 tok/s at 512-row chunks to 70.52 s / 262.70 tok/s at 4K chunks. The matched llama.cpp reference was 361.89 tok/s, so hf2q reached about 72.6% and failed the H4 85% prompt-processing floor. |
| Decode throughput | A warmed 6.2K agent prompt decoded at about 16.35 tok/s versus about 38.58 tok/s for the matched llama.cpp run. H4 decode parity is therefore reopened. |
| Quality reference | llama.cpp on the same Q2 artifact also emitted malformed/non-actionable tool content in automatic and required modes. That implicates the aggressive quantization, but does not relax hf2q's agentic correctness gate. |

DeepSeek-V4-Flash Q2 is therefore an experimental serving target. “Ready” now
means the process and Metal pipelines are initialized; it does not mean the
artifact has passed the agentic quality or llama.cpp parity gates. Release
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
| Pinned llama.cpp conversion and graph | Present at reference commit |
| Pinned llama.cpp oracle binaries | Rebuilt locally as version 10276 (`6ea215d17`) |
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
| One-token reference parity | Pinned llama.cpp `llama-simple` on the exact same GGUF and rendered six-token prompt also greedily decoded `Hello`; no product path invoked the oracle |
| Native inference telemetry | 20.80 s load; 8.15 s incremental six-token prefill (0.736 tok/s); 30.38 s total; 67,364,585,472-byte max RSS; zero swaps |
| Reference inference telemetry | Pinned llama.cpp raw-completion oracle processed the same six tokens in 73.89 ms (81.20 tok/s) after load; not yet an H4 comparison because hf2q currently submits one-token graphs while the reference batches all six |
| Official arithmetic coherence | Prompt `What is 2+2? Answer with only the number.` rendered to 17 source-parity token IDs; greedy output reasoned to `2+2 = 4` and terminated with final answer `4` |
| Extended-context execution | Fresh prompts matrix-prefill up to 1,024 rows beyond the 128-row physical window, then extend exactly; cached suffixes retain exact token-wise extension; server default 131,072 tokens with a metadata-enforced 1,048,576-token ceiling |
| Matrix-bound artifact proof | A fresh 1,036-token request crossed the 1,024-row matrix bound, completed its exact incremental tail coherently, and reached 356.03 prompt tok/s / 2.910 s TTFT on the warm shape |
| Matched fresh-prefix prefill benchmark | Identical 306-token greedy request, warm shape, `cached_tokens=0`, three trials: committed baseline 5.4951/5.4957/5.4905 s (median 55.69 tok/s); matrix-prefill candidate 1.0004/1.0067/1.0035 s (median 304.94 tok/s), a 5.48x TTFT speedup |
| Trivial-prefix policy proof | A one-token recovery-anchor hit followed by a 305-token suffix resets to `cached_tokens=0`; two real-artifact trials prefetched in 1.0034 and 0.9572 s instead of taking the incremental path |
| Artifact-backed tests | Exact 1,328-tensor catalog, all 43 verifier layers plus finite vocabulary logits, greedy token selection, and embedded tokenizer parity passed against the 96.3 GB artifact |
| Final native decode sample | 56 generated tokens in 1.24 s (45.308 tok/s) on the official arithmetic prompt after the clean pinned-dependency release build |
| Matched five-run native benchmark | 45.1, 45.1, 37.7, 45.2, and 45.1 tok/s; median 45.1 tok/s, p95 45.2 tok/s; 63 verifier evaluations for 64 generated tokens; warm-prefill median 74.6 tok/s |
| Matched llama.cpp reference | 41.58 tok/s on the same artifact and benchmark contract; hf2q median is 1.085x (+8.5%) |
| Performance parity | Passed: native median exceeds the H4 0.90x decode floor and the pinned llama.cpp reference while retaining coherent greedy output |
| Native OpenAI server load | Real 96.3 GB artifact loaded in 19.92 s as `DeepSeek-V4-Flash-0731-Q2_K_S`; `/v1/models` reports `deepseek4`, Q2_K, 256 experts/6 active, native MLX, and the configured 4,096-token validation context |
| Exact growing-turn cache | Final real-artifact unary check reused 16 of 27 prompt tokens; only the 11-token suffix ran, with 0.276 s TTFT; an earlier 37-of-48-token coding turn prefetched its suffix in 0.287 s |
| Canonical reasoning recovery | Thinking-mode follow-up with old `reasoning_content` removed restored the native recovery checkpoint (`cached_tokens=4` of 21 on the deliberately tiny prompt) rather than replaying the whole prefix |
| Required tool call | Real model returned OpenAI `read_file` with arguments `{"path":"/tmp/example.txt"}` and `finish_reason: "tool_calls"`; no raw DSML leaked |
| Constrained tool performance | Greedy candidate-ranked grammar enforcement improved the same 51-token tool call from 2.9 to 25.4 tok/s while preserving the structured call |
| SSE tool + cache | Streaming response emitted role, indexed `tool_calls`, `finish_reason: "tool_calls"`, usage, and `[DONE]`; it reused 290 of the 298 rendered tool-prompt tokens |
| Unsupported surfaces | DeepSeek-V4 embeddings, multimodal injections, and slot-aware scheduling fail explicitly; no alternate family/runtime fallback is selected |
| Full binary regression | 3,814 passed, 0 failed, and 44 ignored in 173.97 s after the matrix-prefill/cache-policy change |
