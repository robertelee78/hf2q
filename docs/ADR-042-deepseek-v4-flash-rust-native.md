# ADR-042: DeepSeek-V4-Flash-0731 — Rust-native source conversion and MLX inference

- **Status:** Accepted for implementation (2026-08-04); evidence gates open
- **Owner:** hf2q integration lane
- **Source model:** `deepseek-ai/DeepSeek-V4-Flash-0731`
- **Pinned source revision:** `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- **Reference implementation:** `/opt/llama.cpp` at
  `6ea215d171fd31df943bf1ac8227129f2b963160`
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
- the repository includes one attached DSpark next-token prediction layer.

The existing converter rejects the source dtypes and architecture. Its current
expert fusion creates a multi-gigabyte F32 aggregate, and the runtime has no
DeepSeek-V4 graph or cache. `mlx-native` can parse only four-bit-and-higher GGUF
types, so it cannot load the selected target until Q2_K support lands.

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
memory, and benchmark gates. Commits may be merged and pushed because the user
authorized the full delivery workflow, but no artifact is described as ready
until its exact-source receipt and all applicable acceptance gates are green.

## Current evidence ledger

| Evidence | Result |
|---|---|
| Hardware/storage audit | 128 GiB M5 Max; about 1.8 TiB free |
| Official repository metadata | 48 shards; about 166.9 GB; 304.18B params |
| Existing hf2q Q2_K_S encoder | Present; source-format ingest absent |
| Existing mlx-native Q2_K loader/kernel | Absent |
| Existing DeepSeek-V4 converter/runtime | Absent |
| Pinned llama.cpp conversion and graph | Present at reference commit |
| Pinned llama.cpp oracle binaries | Rebuilt locally as version 10276 (`6ea215d17`) |
| Synthetic converter proof | Pending implementation |
| Official full conversion | Pending converter gate |
| Coherent real-model inference | Pending full conversion/runtime |
| Performance parity | Pending coherent inference |
