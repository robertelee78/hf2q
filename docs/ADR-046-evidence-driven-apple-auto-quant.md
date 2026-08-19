# ADR-046: Evidence-driven Apple-Silicon auto quantization

- Status: Accepted; measured-selector foundation implemented, artifact
  generation and CLI activation remain gated by the phases below
- Date: 2026-08-18
- Owners: hf2q product pipeline; mlx-native model-agnostic execution primitives
- Supersedes: ADR-020's proposed DWQ architecture and performance claims

## Context

hf2q has three surfaces that have been called "auto" but do not form one
production-optimal quantization system:

1. `src/intelligence/auto_quant.rs` estimates a bit policy from parameter
   count and nominal memory bandwidth. It cannot know the actual Metal kernel,
   execution shape, output quality, or measured token rate. It emits names such
   as `imatrix-q4_k_m` and `dynamic-quant-4-6`; the adjacent heuristic resolver
   can emit bare `apex`. `QuantSelector::from_name` does not accept those names.
2. `src/serve/quant_select.rs` chooses from a static memory-fit table.
3. `hf2q convert --quant` requires a concrete selector. `dwq` is reserved and
   fails with `QuantSelectorError::DwqReserved`.

Two prerequisite seams are also open:

- `src/quality/` contains KL, perplexity, cosine, and regression code but is
  not declared by the production crate. Its integration tests use path-included
  stubs, real quantized Qwen FFN forward is unsupported there, and the DWQ
  smoke route in `src/arch/smoke.rs` returns `Skipped`. Source-text tests pin
  thresholds but do not execute a real statistical quality gate.
- The converter accepts or can internally select GGUF tensor encodings that
  `mlx-native 0.10.11` does not read. Confirmed examples are explicit Q4_1 and
  Q5_0, plus the Q4_K shape fallback to Q5_0. A production auto system must
  prove converter-output-to-runtime servability before considering quality or
  speed.

The production baseline is the native GGUF conversion and `mlx-native`
runtime. For Qwen3.8-27B, ADR-044 records a 16.81 GB native Q4_K_M artifact and
a matched seven-run M=1 decode median of 29.19 tok/s on an Apple M5 Max. It
also records that four independent scalar forwards lose to a width-four peer.
Those results demonstrate why model bytes or a single M=1 number cannot be the
whole objective.

The old ADR-020 correctly distinguished dynamic mixed precision from genuine
MLX DWQ and identified that learned affine parameters cannot survive a lossy
MLX-affine-to-GGUF-Q4_K translation. Its proposed producer was subsequently
removed by ADR-033 P6, however, and its remaining overlay consumer is not a
complete production format:

- `src/core/mlx_safetensors_loader.rs` supports only 4- and 8-bit input and
  expands packed codes to bytes and floating values to F32;
- `src/serve/forward_mlx_shared.rs` repacks those codes and dispatches only
  4-bit/group-32 affine weights through the overlay route;
- the overlay is read as one complete file and covers only selected dense
  linear slots;
- there is no current Rust DWQ trainer or full-model MLX-affine converter.

The exact `mlx-native = 0.10.11` dependency has affine packed-weight kernels
and QDQ affine primitives, but the existence of a primitive does not prove
that hf2q's loader, graph routing, prompt QMM, token QMV, width-N, or model
family is complete or fast.

Serving transformations are part of the candidate too. For example, the
Qwen35 GPU path currently uploads `output.weight` as Q4_0 from its F32-loaded
form even when conversion promoted the tensor to Q6_K. That load-time
requantization may be a valid speed trade, but it changes the executed weights
and therefore requires the same source-quality evidence as conversion.

The motivating source model may be the upstream checkpoint or any exact
weight-modified checkpoint: fine-tuned, merged, pruned, abliterated, or
otherwise transformed. The conversion architecture must not infer or
special-case that history. Its source teacher is the exact supplied tensor
bundle.

The [PocketAiHub Qwen3.8 repository](https://huggingface.co/PocketAiHub/Qwen3.8-27B-Abliterated-MLX)
does not demonstrate DWQ. Its card labels the experimental 2-bit candidate
AWQ/group-32, the 4/6/8-bit candidates MLX affine/group-64, and the vision
tower BF16. The published artifacts combine the refusal-direction projection
with those quantizations and pass that project's limited behavior screen, but
the card does not publish complete build code/order, a DWQ training receipt,
or exact behavior-preservation proof. Its own M5 Max measurements also show
the 4-bit candidate decoding faster than its 2-bit candidate, reinforcing that
lower bit width is not an inference-performance proof.

## Terminology and separations

hf2q treats the following as different architectural axes:

| Term | Role | Runtime implication |
| --- | --- | --- |
| MLX affine | Encoding/dequantization algebra: packed integer codes plus per-group scale and bias, reconstructed as `q * scale + bias` | Fast only when the exact bit width, group size, shape, dtype, and Metal kernel are efficient |
| GGUF K-quant | Different block encoding with format-specific scale/min packing | Requires its own native kernels; it cannot losslessly store learned MLX-affine scale/bias values |
| RTN | Calibration-free code/scale initialization | Produces an encoding; does not define execution speed |
| imatrix | Data-driven tensor/block importance input to a quantization policy | Changes allocation or error weighting, not the kernel ABI by itself |
| dynamic mixed precision | Calibration-driven policy that assigns different precision to different tensors | Speed follows the resulting encodings and routing |
| AWQ | Activation-aware rescaling/clipping before quantization | Quality method; speed requires a hardware-efficient output encoding and kernel |
| GPTQ | Layer-wise error-minimizing post-training quantization | Quality method; speed requires a hardware-efficient output encoding and kernel |
| DWQ | Distillation that trains non-quantized parameters of affine-quantized modules, especially scales and biases, against a source teacher | Executes as ordinary affine quantization after training; DWQ is not a distinct inference kernel |

Apple's mlx-lm learned-quantization guide describes dynamic quantization, AWQ,
GPTQ, and DWQ as calibration methods and explicitly permits them to be
cascaded. Its current DWQ implementation freezes integer codes, trains affine
scales and biases with KL distillation, and can store teacher top-k targets.
The algorithm label therefore cannot be used as an inference speed predictor.

Primary references:

- [MLX learned quantization guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md)
- [MLX DWQ implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/quant/dwq.py)
- [MLX dynamic quantization implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/quant/dynamic_quant.py)
- [MLX affine quantization API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
- [AWQ paper](https://arxiv.org/abs/2306.00978)
- [GPTQ paper](https://arxiv.org/abs/2210.17323)

## Decision

### 1. Optimize a serving candidate, not a bit width

The unit selected by production auto quantization is a `ServingCandidate`:

```text
exact source identity
+ calibration algorithm and corpus/teacher identity
+ per-tensor precision policy
+ stored weight encoding
+ native loader and kernel/routing profile
+ KV/cache/speculation/concurrency configuration
= one candidate artifact and server configuration
```

Conversion time is recorded but is secondary. The product objective is fastest
inference among candidates that satisfy every correctness and quality gate.

### 2. The source contract is transformation-agnostic

Every candidate is bound to the exact source revision, `config.json` hash, and
canonical tensor-bundle hash. Teacher logits and behavior baselines come from
that exact source, never from a checkpoint inferred from its model name.

The selector contains no `abliterated`, `fine_tuned`, or similar branch. An
owner may require named behavioral regression suites, but these are opaque,
generic gates. The same mechanism can preserve refusal changes, code behavior,
tool behavior, a domain fine-tune, or any other intentional source property.

Quantization error can reintroduce a component that a weight transform removed.
For a projected source weight `W'` and quantization error `E`, the executed
weight is `W' + E`; even if `W'd = 0`, generally `(W' + E)d != 0`. This is
ordinary approximation drift. It is not proof that DWQ intrinsically reverses
the transformation. DWQ should use the exact modified source as teacher, and
the generic source-logit plus behavioral gates decide whether the result is
acceptable.

This supersedes any historical rule that required a stock/upstream teacher.
Substituting a different checkpoint as teacher is source drift and is rejected,
regardless of whether the supplied checkpoint is vanilla or weight-modified.

### 3. Quality gates precede performance ranking

A candidate is ineligible unless all applicable evidence passes:

1. source identity, converter completion, tensor catalog, artifact integrity,
   exact-runtime loading, and required kernel-contract execution;
2. tokenizer/template identity, teacher-logit KL, top-1 agreement, activation
   cosine similarity, and perplexity-ratio thresholds;
3. exact agentic tool name, schema, arguments, tool-result continuation, and
   unary/SSE semantics;
4. context retrieval, cache-prefix reuse, and cold/cached continuation parity;
5. multimodal grounding and image/cache isolation when the family supports it;
6. every owner- or family-required behavioral regression suite.

Thresholds and corpora are versioned inputs to the evidence contract. Passing
a phrase-based refusal screen, producing valid JSON with wrong arguments, or
matching only a few sampled completions is not sufficient evidence.

### 4. Performance evidence is execution-regime specific

There is no universal fastest quant. Each workload profile declares required
service levels and one primary ranking regime from:

- text prompt prefill (QMM);
- single-request token decode (M=1 QMV);
- width-N or concurrent decode;
- long-context decode including KV/cache costs;
- multimodal/projector prefill when applicable.

For every required regime the receipt records exact prompt/settings, warmup,
run count, tokens per run, median semantic TTFT, median token rate, peak MLX
memory, and output-quality result. All required service levels must pass.
Eligible candidates are ranked by the profile's primary measured token rate,
then semantic TTFT, artifact size, and a deterministic id tie-breaker.

This lexicographic contract prevents a weighted average from hiding a hard
failure. Separate profiles may legitimately choose different artifacts.

### 5. Evidence is exact and non-transferable by default

Candidate and selection receipts are keyed by at least:

- source config and tensor-bundle hashes;
- complete calibration corpus/teacher and precision-policy hashes;
- artifact hash;
- hf2q revision and exact `mlx-native` version;
- hardware identity, memory class, and OS build;
- tokenizer, chat template, prompt suite, sampling settings, context and batch;
- kernel/routing, KV/cache, speculation, and concurrency configuration.

An identity mismatch rejects cached evidence. Similar-model or theoretical
bandwidth estimates may generate an experiment plan but never authorize a
production winner. If no exact candidate receipt passes, auto selection returns
"no eligible candidate" and requests measurement; it does not guess.

### 6. Artifact format and Metal kernels are co-designed

Current GGUF K-quant candidates remain the control and first candidate family.
Genuine DWQ must preserve its trained affine scales and biases in a native
MLX-affine artifact; hf2q must not translate them through GGUF Q4_K.

Before full-model affine candidates are eligible, the two-repository stack must
provide:

- hf2q sharded, bounded-memory artifact writing/manifest validation and
  mlx-native bounded-memory safetensors-to-Metal loading;
- packed codes with no whole-model unpack/repack cycle;
- scale/bias dtype and metadata compatible with the exact native kernels;
- explicit support for each bit width/group size/family/shape;
- prompt QMM, token QMV, routed-expert, width-N, and multimodal graph routes as
  applicable;
- fail-closed validation of tensor catalog, shard index, policy manifest, and
  kernel capability.

Group size and bit width are candidate dimensions, not constants. A smaller
group may improve approximation while increasing scale/bias traffic and
slowing Metal execution. A two-bit artifact may be smaller and still be slower
than four-bit for prompt or batched shapes.

### 7. Learned quantization is staged on one affine ABI

After the full-model affine baseline is correct, candidate producers land in
measured order:

1. affine RTN control;
2. dynamic mixed-precision allocation;
3. AWQ and GPTQ candidates;
4. DWQ scale/bias distillation, optionally initialized from a preceding method.

All production implementation remains Rust plus `mlx-native`. External tools
may be reference oracles in benchmark harnesses only. Each algorithm consumes
the exact source and emits the same manifest/receipt schema, so the selector is
agnostic to how the codes/scales were produced.

For 27B-class DWQ on 128 GiB unified memory, the implementation must stream
teacher targets to sharded storage, release teacher state before student
training where possible, bound calibration batches, and prove peak memory.
The default corpus cannot be assumed behavior-complete; owner/family regression
suites supplement source-logit distillation.

### 8. Existing heuristic auto remains planning-only

`src/intelligence/auto_quant.rs` and static memory-fit logic may propose a
candidate set or reject artifacts that cannot fit. Their output is not an
"optimal" result and must not be presented as measured authority. Production
`--quant auto` remains unavailable until the CLI drives candidate generation,
quality evaluation, matched Apple-Silicon benchmarking, and the selector in
`src/intelligence/measured_auto_quant.rs`.

### 9. hf2q and mlx-native have an explicit ownership seam

The two repositories are one product stack but do not duplicate responsibility.

hf2q owns:

- Hugging Face source identity, safetensors ingestion, architecture and tensor
  mapping;
- calibration corpus handling, RTN/imatrix/dynamic/AWQ/GPTQ/DWQ algorithms,
  per-tensor policy, and bounded-memory artifact production;
- the model-family graph, tensor-to-operation routing, candidate manifests,
  source-quality/performance evidence, auto selection, and serving behavior.

mlx-native owns reusable, model-agnostic execution machinery:

- packed affine tensor/buffer types and low-level safetensors-to-Metal loading;
- quantize/dequantize and scale/bias forward/backward primitives used by native
  calibration;
- Metal QMV, QMM, expert, embedding, and related kernels, dispatch constraints,
  buffer/dtype behavior, and kernel-level correctness/performance tests;
- a machine-readable capability surface describing supported bit widths, group
  sizes, dtypes, shapes, and execution regimes.

The `mlx-native 0.10.11` release contains a generic `QuantizedWeight` and
sharded safetensors loader, packed affine 4/6/8-bit execution, optimized
4/8-bit QMV routes, affine QMM variants, packed 4/6-bit embedding gather, and
QDQ affine scale/bias gradient primitives. It also publishes the serde-backed
`packed_affine_capability` contract for dense, expert-offset, expert-ID, and
embedding operations, including exact bit, group, shape, bias, dtype, regime,
kernel-route, fallback, and rejection data. hf2q must consume and extend those
generic facilities instead of growing a second private affine runtime around
the legacy overlay.

Those facilities are not yet one complete fast ABI. The generic packed-weight
route has row-wise SIMD for supported 4/8-bit decode layouts, a scalar 6-bit
fallback, and packed 4/6-bit embedding gather, but no specialized packed
prompt-QMM/width-N route. The older `qmm_affine` family has a separate
unpacked-U8/F32 contract plus a narrow packed 4-bit variant. The published
capability response proves executability and exposes fallback routing; it does
not turn a fallback into a performance claim. Upstream MLX supporting a bit
width or group size likewise does not make that tuple fast—or even executable—
in the pinned Rust runtime. Phase C must converge these paths around one packed
artifact contract and report fallbacks rather than hiding them.

Model-family decisions do not move into mlx-native, and Metal kernels do not
move into hf2q. hf2q asks the pinned runtime capability surface whether every
required tensor and regime is executable, then fails closed when it is not.
The artifact manifest and capability response form a versioned ABI between the
repositories.

When a candidate needs new native support, work lands in this order:

1. implement and benchmark the model-agnostic primitive in a dedicated
   mlx-native branch/worktree;
2. publish the tested mlx-native revision;
3. pin that published revision in hf2q's `Cargo.toml` and `Cargo.lock`;
4. implement the hf2q producer/graph route and run exact-artifact quality and
   serving gates.

An ignored local Cargo patch may accelerate a spike but cannot be acceptance
evidence or part of a landed result.

## Implementation sequence

### Phase A — evidence and selector foundation (implemented)

- Add typed source, execution, algorithm, encoding, quality, performance, and
  workload receipt contracts.
- Require exact-artifact converter, tensor-catalog, runtime-load, and kernel
  servability proof before quality or performance ranking.
- Reject stale source/runtime/hardware evidence, incomplete regimes, failed
  behavior gates, malformed digests/encodings, invalid measurements, duplicate
  ids, and over-budget or missing memory evidence with typed diagnostics.
- Bind quality metrics to an exact evaluation-suite hash, bind kernel routing
  to a capability-profile hash, and emit a self-describing selection receipt.
- Select only among eligible candidates using measured profile performance.
- Unit-test that a faster but behavior-drifting candidate loses and that
  vanilla or modified source weights follow the same exact-hash rule.

This phase changes no conversion format and makes no new speed claim.

### Phase A.1 — close current Gate-0/Gate-1 seams

- Pin and consume `mlx-native 0.10.11`'s published capability surface from
  hf2q as a machine-readable converter/runtime tensor-type contract. Keep a
  candidate ineligible unless every required tensor and regime is executable.
- Reject or implement Q4_1, Q5_0, BF16, MXFP4, and every internal shape
  fallback consistently; test every accepted selector through exact-runtime
  GGUF reopen and the regimes it advertises.
- Replace the disconnected statistical-quality stubs with a receipt-producing
  source-vs-candidate logit harness. A missing real-model fixture must be a
  recorded non-pass, never a green quality result.
- Make load-time requantization and dense-expansion choices visible in the
  candidate manifest rather than implicit runtime behavior.

### Phase B — current GGUF measured-candidate orchestrator

- Generate reproducible candidates from the existing Q3/Q4/Q5/Q6/Q8 and APEX
  policies supported by each explicit model family.
- Add a checked-in benchmark/quality harness that emits Phase A receipts.
- Establish Qwen3.8 Q4_K_M as the control and measure alternative candidates
  on the exact ADR-044 workloads, including width-N and long context.
- Store validated receipts in project evidence memory only after exact-artifact
  tests; persist portable receipts beside local artifacts, never in Git.

### Phase C — production MLX-affine artifact and runtime

- In hf2q, replace the legacy overlay with a manifest-driven full-model graph
  route and produce indexed, sharded artifacts without whole-model residency.
- Extend the generic mlx-native packed-weight loader/capability API where a
  required regime is absent, and eliminate packed-code expansion/repacking in
  every required execution route.
- Support and benchmark the exact 4/6/8-bit group-size matrix exposed by
  mlx-native; implement and publish native kernel work where a required shape
  or regime is absent, then pin that release in hf2q.
- Prove full Qwen3.8 conversion, load, deterministic generation, tools, cache,
  vision where applicable, memory bound, and all workload regimes.

Two-bit is not implied by this phase. It requires an explicit storage/kernel
capability and must independently beat eligible candidates.

### Phase D — calibration producers

- Port dynamic sensitivity measurement and allocation with exact corpus hashes.
- Add AWQ and GPTQ only behind the common candidate/evidence contract.
- Add native DWQ with teacher-target sharding, trainable affine scale/bias
  parameters, optimizer state bounds, checkpoints, and source-teacher gates.
- Compare algorithms at identical encoding, group size, artifact/runtime,
  corpus, and workload wherever the question is algorithmic quality.

### Phase E — production `--quant auto`

- Add an operator profile and budget surface.
- Generate or reuse only exact receipts, show candidate rejections, and require
  no undocumented fallback.
- Activate auto conversion only after an end-to-end real-model run proves the
  complete loop from Hugging Face safetensors through native serving.

## Acceptance evidence required for later phases

Model-free gates:

- receipt schema round-trip and deterministic selection;
- all identity, quality, missing-regime, NaN/inf, evidence-depth, and memory
  failures reject with typed diagnostics;
- candidate registries reject unsupported family/encoding/kernel tuples;
- sharded affine format round-trips packed codes, scales, biases, tensor names,
  shapes, and policy hashes without expansion-dependent behavior.

Real-model Apple-Silicon gates:

- exact official or owner-supplied source bundle and exact artifact hashes;
- matched unquantized-source teacher logits and documented thresholds;
- realistic coding/tool/context/cache/multimodal suites as applicable;
- at least five measured runs after warmup, with medians and every profile
  regime reported;
- matched prompts, tokens, sampling, process residency, server configuration,
  and hardware across candidates;
- no quality regression or structural failure hidden by a speed result;
- cleanup proves only one large reference/runtime instance is resident.

## Consequences

"Optimal" now means fastest measured inference for an explicit Apple-Silicon
workload profile after hard coherence gates, not smallest artifact, highest
nominal compression, or best theoretical bandwidth. DWQ becomes a quality
optimization that can make a lower-precision affine candidate eligible; it is
not a speed feature by itself.

The design applies unchanged to vanilla and weight-modified checkpoints. An
intentional source behavior is preserved because the exact source is the
teacher and its required suites are gates, not because hf2q recognizes the
transformation's name.

Full production auto quantization is not claimed by Phase A. The remaining
work is measurable engineering—artifact ownership, kernel routing,
calibration, and real-model evidence—not an unresolved conceptual dependency.
