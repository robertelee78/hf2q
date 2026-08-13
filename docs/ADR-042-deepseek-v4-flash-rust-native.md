# ADR-042: DeepSeek-V4-Flash-0731 — Rust-native source conversion and MLX inference

- **Status:** Accepted for hf2q 0.1.2; full-context four-agent serving and
  cache growth revalidated 2026-08-08
- **Updated:** 2026-08-11 — cross-family lifecycle review added active-prefix
  affinity, request-local rollback, and the DeepSeek Busy-only admission
  no-spin correction. The four-agent performance workload is now an immutable,
  SHA-bound 6,685-token fixture after mutable README growth invalidated one
  release run. One exact-packed wave exposed and corrected a thermal-order
  defect in that release gate; the next thermally valid wave isolated a
  saturated-cold scheduling regression. The bounded mixed-work replacement,
  saturated-cold barrier, paired-prefill schedule, and cold-cohort thermal
  evidence boundary remain release candidates pending immutable
  packed-artifact hardware proof.
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
repository files. It must not invoke Python, llama.cpp, or another converter,
quantizer, or inference runtime. A separate developer-only parity harness may
invoke the pinned llama.cpp build as an oracle.

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
| Native dependency | Exact crates.io pin `mlx-native =0.10.3`, released from main commit `9f2210c8f03484a9c1122ef6a2bcf3be4226f29a` with implementation head `43d4b43b2cce1e9289561f45b9f6433437714028`; registry checksum and independently downloaded crate SHA-256 `1db3a6e739a199c7e9a7820a49718d549f6b03a77241f461cffb3ad085cb833d` |
| Quant plan | 1,328 verifier tensors: 172 Q2_K, 86 Q3_K, 532 Q8_0, 535 F32, and 3 I32; 4,705 DSpark tensors explicitly excluded from the base artifact |
| Conversion bound | Rust-native row-aligned streaming; maximum working vector bound 4,798,873,600 bytes; no external converter or inference process |
| Arithmetic coherence | Greedy `What is 2+2? Reply with only number.` returned exactly `4` |
| Tool semantics | The curl/OpenAI-compatible harness made required and automatic choice both select `read_file` with exactly `/opt/hf2q/Cargo.toml`; unary and SSE returned valid OpenAI `tool_calls` and `finish_reason: "tool_calls"`. The source-argument regression also returned `fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result` byte-for-byte in one `emit_source` call (4 s response), proving the formerly truncated `<` syntax through the real model. A real OpenCode two-turn run issued five tool calls on its first turn and two on its second, changed the requested source, and passed the named oracle/regression checks in the same session. |
| Tool-result continuation | With tools enabled in `auto` mode, the model consumed the Cargo result and returned the requested sentinel without another call. The real OpenCode continuation consumed prior tool results, issued the next required calls, and reused the same live session. |
| Prefix reuse | The current checked-in gate used a 6,262-token prompt and reused 6,254 tokens on repeated, automatic, SSE, and post-tool turns. Cold TTFT was 9.564 s and cached TTFT was 227 ms; every required/automatic tool, source-argument, tool-result, unary, and SSE assertion passed. |
| Canonical launcher | `scripts/serve_deepseek4_opencode.sh` passed the curl agentic gate, the real OpenCode coding run, the four-agent full-context gate, and the 120K cache-growth gate. It advertises 524,288 tokens for every slot and demand-allocates 131K initially. Memory/port preflight refuses an unsafe 100 GiB load before model mapping. The 131K-to-262K boundary is proven; a near-512K physical allocation remains unproven on this 128 GiB host. |
| Ordinary agentic prompt | On the same approximately 5.9K-token README coding prompt, hf2q warm prefill was about 518 tok/s and median decode was 32.1 tok/s; llama.cpp build 10293 reported 399.4 prompt tok/s and 31.7 generation tok/s. |
| 120K cold prompt | `scripts/test_deepseek4_long_context_cache.sh` produced the exact required tool call for 119,808 prompt tokens in 321.034 s TTFT (373.194 tok/s); decode was 23.869 tok/s. The same source-bound prompt and artifact under llama.cpp build 10298 processed 119,807 tokens in 749.015 s (159.953 tok/s) and decoded at 19.565 tok/s. hf2q was 2.33x the reference prompt rate and 1.22x its decode rate for these source-bound runs. |
| 120K continuation cache | Appending the real tool result produced a 119,907-token request that reused 119,800 tokens (99.91%), evaluated only a 107-token suffix, returned its first semantic event in 1.113 s, and emitted the exact requested sentinel. |
| 98K OpenCode-scale revalidation | A fresh 97,127-token required-tool request completed cold prefill in 424.522 s (228.79 tok/s). Its 97,214-token tool-result turn restored the compact recovery anchor, reused 97,119 tokens (99.90%), evaluated a 95-token suffix in 1.378 s TTFT, and completed in 2 s. |
| Before/after control | The earlier identical-class hf2q run required 594.575 s for 119,808 tokens (201.502 tok/s). The completion candidate reduced cold-prefill time by 46.0% and increased its token rate by 85.2%. At the shorter exact 26,024-token gate, the threshold retune improved 388.846 tok/s to 497.277 tok/s while retaining the exact tool call; the cached continuation reused 26,016 tokens and reached 927 ms TTFT. |
| Output parity | Both runtimes returned the exact requested comma-separated sequence. llama.cpp also returned the exact required `read_file` path on the long repository prompt. |
| Memory safety | A 4,096-row prefill command buffer produced Metal `kIOGPUCommandBufferCallbackErrorOutOfMemory`; the accepted 2,048-row transaction completed the 119.8K gate. A later OpenCode session falsified the steady-state claim when macOS killed hf2q at 108,840 MB after the shared transient arena retained cold-prefill buckets. The split-arena build releases prefill scratch before decode; the new file-backed build released 4,933,917,968 transient bytes after the 120K prefill, remained alive through its cached continuation, reported 2.0 GiB RSS, and shut down cleanly. Eagerly allocating the full 524,288-token cache beside this artifact still OOMs, so demand growth remains required. Only one 100 GiB-class runtime was resident during every comparison. |

### Completion-audit performance ledger (accepted candidate, 2026-08-06)

The tighter completion gate uses the reproduced 107,431,343,168-byte artifact
with SHA-256
`936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d` and
llama.cpp commit `15586e2d7165570fb3aa7c26e0d442e289ef69de`. On the first clean
cooled three-trial matched run, llama.cpp medians were 674.458 prompt tok/s and
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
  2.669 ms for NSG=8. The explicit-NSG spike was removed and the llama-derived
  NSG=8 geometry remains authoritative.
- Keeping every 2,048-token transaction on dense masked flash matched the
  exact 1-through-64 transcript with zero cached tokens, but reached only
  615.146 prompt tok/s and 33.865 decode tok/s. That missed the predeclared
  624.185 prompt-tok/s scalar floor and the 674.458 llama.cpp median. The
  all-dense route was rejected; one final mixed-route spike isolates chunk 2
  before closing this routing hypothesis.
- Routing chunk 2 through dense flash while leaving chunk 3 gathered produced
  three exact, zero-cache-credit trials at 625.484/625.724/625.436 prompt
  tok/s and 33.855/33.886/33.898 decode tok/s. The 625.484 median was only
  0.21% above the clean 624.185 scalar median and remained 7.26% below the
  674.458 llama.cpp prompt median. That noise-sized short-shape result does
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
  lightning-indexer query and compressed KV. llama.cpp's pinned DeepSeek-V4
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
continuity. During prefill, pinned llama.cpp occupied 6.982789 seconds of a
7.496751-second GPU submission span: 0.513962 seconds idle and 93.14% union
utilization. hf2q occupied only 6.869377 seconds of a 7.757780-second span:
0.888403 seconds idle and 88.55% utilization. hf2q therefore completed about
113.4 ms less GPU work but accumulated about 374.4 ms more idle time, producing
the observed approximately 261.0 ms wall-clock loss. Decode showed the
opposite shape: llama.cpp occupied 4.033765 of 5.113851 seconds (78.88%), while
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
2,048/2,048/891-row chunks times 43 layers. Pinned llama.cpp used six compute
encoders over the same prompt, two per chunk, before its output work began.
The runtimes therefore do not merely choose different kernel shapes: hf2q
forces 43 CPU/GPU rendezvous points per chunk while llama.cpp lifetime-plans a
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
model graph. Pinned llama.cpp commits each Metal residency set, immediately
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

Pinned llama.cpp supplies the missing lifecycle contract. Its Metal backend
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
close H4 because the pinned llama.cpp prefill median remains higher.

The earlier attribution to ordinary sustained-load GPU DVFS is rejected by the
GPU-interval and first-group measurements above. The final clean-registry
paired gate used the same artifact, 4,987-token prompt, greedy
temperature-zero/seed-42 settings, exact transcript oracle, zero prompt-cache
credit, and 60-second gaps before and between three trials. Pinned llama.cpp
commit `15586e2d7165570fb3aa7c26e0d442e289ef69de` measured
673.497/672.744/674.711 prompt tok/s and
31.810/31.855/31.821 decode tok/s. The hf2q build pinned to the verified
`mlx-native` 0.10.3 registry archive measured
674.026/674.785/676.812 prompt tok/s and
34.054/33.885/33.958 decode tok/s. Every transcript was exact; hf2q therefore
passed this cold-prefix gate at 1.0019x llama.cpp prefill and 1.0672x decode.
The complete source-bound receipt is retained under
`hf2q-deepseek-parity.XXXXXX.1PXRWahgxc`; it records artifact, binary, hf2q
patch, the clean mlx-native source state, and implementation hashes.

The 2026-08-07 current-reference refresh used the same reproduced artifact,
hf2q runtime `03e378e9862e6d9add0d08ea68c1d6c449357364`, clean `mlx-native`
head `eb1b031876a0d5aa3b16803a54e78aa5de7d2e62`, and llama.cpp
`3653e6d6d547ec763317d9ecd0ace334a7e21359`. All six transcripts were
again exact with zero cache credit. llama.cpp measured
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
33.687/33.659/30.504 decode tok/s, while the matched llama.cpp arm stayed flat
at a 669.954 prompt-tok/s median. The warmup neither improved the first trial
nor preserved sustained performance, so the entire code/test spike was
removed. Moving another full sparse/indexer workload into startup is not an
accepted performance technique.

The next spike kept the same three Metal transactions for the 4,987-token
parity prompt but changed their shapes from 2,048/2,048/891 to
1,664/1,664/1,659 by selecting 13 sparse windows. Against the same current
llama.cpp source, the exact zero-cache hf2q trials measured
671.015/672.071/671.476 prompt tok/s; llama.cpp measured
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
llama.cpp `3653e6d6d547ec763317d9ecd0ace334a7e21359`, clean mlx-native
`eb1b031876a0d5aa3b16803a54e78aa5de7d2e62`, and the exact hf2q candidate
binary SHA-256
`222251a89a3535a92e6ba7c847fb1e395a5d617e78668c4c6f2449baf6ffae69`.
llama.cpp measured 670.226/670.948/670.117 prompt tok/s and
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
as a speedup. It remains over the historical 159.953-217.5 tok/s llama.cpp
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

The matched llama.cpp server completed its corresponding cold four-request
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
The release wrapper verifies that artifact before model startup and preserves
the prompt-visible calibration path
`/opt/hf2q-worktrees/full-context-slots/Cargo.toml` from exact commit
`863ea423`; the simulated tool result itself is still read from the packed
candidate's `Cargo.toml`. This is workload identity, not a runtime dependency
on that local path. Every cold agent must render exactly 6,685 tokens.
Producer, aggregate, and publication checks all
bind the fixture ID, digest, byte/character counts, exact prompt count, zero
cold reuse, semantic/tool assertions, and literal 55,000 ms cold bounds. A
model-free negative matrix rejects missing, mistyped, stale, off-by-one, or
over-limit receipts.

The 55-second bound is intentionally unchanged. A fresh exact-packed M5 Max
rerun of the frozen request twice is the discriminator: success restores
matched release authority; failure leaves a current-scheduler performance
blocker that must be optimized or re-baselined against a same-input llama.cpp
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
workloads. Before each fresh server starts, with no hf2q/llama model runtime
loaded, the host
must report `ProcessInfo.thermalState == nominal` at five-second cadence for at
least 60 seconds. Thermal state is then sampled every two seconds throughout
the wave; any observed non-Nominal state,
malformed read, monitor failure, or telemetry gap invalidates the run. Each
envelope records the settle and measurement sample counts and SHA-256 digests,
and publication independently rehashes and validates the measurement log.
The 55-second bound is unchanged. Two thermally valid exact-packed passes are
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
reproducible. They use llama.cpp build 10326 (`3653e6d6d`), binary SHA-256
`90bdf03673f7ee61d65d579a4e0be64a914edac1ccb23e74871040bc30d13543`,
the exact model SHA-256
`936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d`,
`--ctx-size 131072 --parallel 4 --kv-unified --flash-attn on`, and disabled
prompt caching. The four request JSON files are generated by the same frozen
builder used by hf2q. Pinned llama.cpp renders those identical bytes as 6,695
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
55-second cold comparison.

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
| Unsupported surfaces | DeepSeek-V4 embeddings and multimodal injections fail explicitly; no alternate family/runtime fallback is selected. Slot-aware text generation is supported with independent full-context sessions and shared physical KV admission. |
| Full binary regression | 3,814 passed, 0 failed, and 44 ignored in 173.97 s after the matrix-prefill/cache-policy change |
