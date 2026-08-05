# ADR-042: DeepSeek-V4-Flash-0731 — Rust-native source conversion and MLX inference

- **Status:** Conversion accepted; agentic serving and performance gates reopened
  after long-context OpenCode revalidation (2026-08-05)
- **Updated:** 2026-08-05 — published the required native append kernels as
  `mlx-native 0.9.6` and strengthened the agentic serving gate
- **Owner:** hf2q integration lane
- **Source model:** `deepseek-ai/DeepSeek-V4-Flash-0731`
- **Pinned source revision:** `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- **Reference implementation:** `/opt/llama.cpp` at
  `360e1349f0009c5ad99d21e3c4546b707addc68a`
- **Target host:** Apple M5 Max, 40-core GPU, 128 GiB unified memory

## Decision

hf2q will download the **official Hugging Face checkpoint** and convert it to a
`Q2_K_S` GGUF itself. Conversion, quantization, loading, and inference are
implemented in-process in Rust (with owned Metal kernels in `mlx-native`).

The product may invoke `hf`, `wget`, or `curl` solely to fetch official source
repository files. It must not invoke Python, llama.cpp, or another converter,
quantizer, or inference runtime. A separate developer-only parity harness may
invoke the pinned llama.cpp build as an oracle.

Prebuilt quantized weights are not an input, fallback, cache seed, or release
artifact. Their published sizes may be used only as non-authoritative capacity
evidence.

The revalidated runtime candidate executes prompts in bounded 4,096-token
matrix transactions, 32 times wider than the 128-token physical attention
window. Attention reads each transaction's compact KV source while only the
final physical window is published to the circular cache. A suffix shorter than
33 tokens retains exact token-wise extension; longer suffixes use another
matrix transaction so an agentic follow-up does not degrade into full
decode-style replay. This requires `mlx-native =0.9.6`, published from source
commit `7b05016b1bc2b4cce06bb0c4336abf8bded1c394`; it must not be enabled against
an older dependency that only accepts one-token nonzero-position appends. The
OpenAI server context may be configured with `HF2Q_DEEPSEEK_MAX_SEQ_LEN` and is
capped by checkpoint metadata. The separate DSpark draft artifact remains
follow-up work and is never silently represented as part of the base GGUF.

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
experts resident, resume can publish a partial tensor, or source/output identity
is not bound to a receipt.

### H3 — Q2_K_S fits and remains coherent

The produced main-model artifact is at most 100 GB decimal and loads with enough
headroom for Metal state and useful KV capacity on the 128 GiB host.

**Falsifier:** peak process footprint exceeds 116 GiB at the acceptance context,
macOS enters critical memory pressure, swap grows materially during steady-state
decode, or deterministic prompts become incoherent relative to the reference.

### H4 — owned low-bit kernels reach parity

The Rust/Metal Q2_K path matches a scalar Rust decoder within the declared
numeric tolerance and reaches at least 0.90x the pinned llama.cpp decode rate
and 0.85x its prompt-processing rate under the same model, prompt, context,
threading, and sampling settings.

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
streaming; it must not build a complete F32 fused projection. Checkpoints bind
the source revision, shard hashes, converter commit, tensor name, output offset,
length, quant type, and payload checksum. Resume revalidates all of them.

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
`17,842,176 + 6,880*C` bytes. The default `C=131,072` allocation is
919,617,536 bytes (about 877 MiB); the recovery snapshot has the same fixed
capacity, so admission accounts for 13,760 bytes per context token and about
1.71 GiB across both allocations. At the trained one-million-token limit, each
allocation is 7,232,045,056 bytes. Operators must choose a serving context that
leaves sufficient unified-memory headroom for both cache copies and scratch.

Tool definitions are rendered by the owned 0731 encoder. Grammar-constrained
generation emits one official `<｜DSML｜tool_calls>` envelope containing one or
more invokes, validates required parameters, and converts the completed block
to OpenAI `tool_calls` objects for both unary JSON and SSE. Raw DSML framing is
not exposed to API clients.

## Acceptance gates

### Converter gate

- A synthetic official-layout checkpoint converts to `Q2_K_S` in-process.
- Positive tests cover E4M3, all E2M1 nibbles, E8M0 edges, integer routing,
  canonical metadata, tensor naming, expert ordering, and GGUF round-trip.
- Negative tests cover missing/malformed scales, wrong dtypes and ranks,
  incomplete expert groups, invalid route tables, truncation, and corrupt resume.
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

## Agentic revalidation (2026-08-05)

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
| Compressed cache ownership | Fixed-offset contiguous raw/compressed KV views plus exact main/indexer F32 recurrent states; snapshots copy all attention/indexer/recurrent state without aliasing; 1M-token resident plan 7,232,045,056 bytes per live cache or checkpoint |
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
