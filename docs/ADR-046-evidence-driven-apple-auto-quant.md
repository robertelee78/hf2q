# ADR-046: Evidence-driven Apple-Silicon auto quantization

- Status: Accepted; measured-selector foundation implemented, artifact
  generation and CLI activation remain gated by the phases below
- Date: 2026-08-18
- Updated: 2026-08-20 — source-to-stored evidence and the bounded
  loaded/executed/encoded producer use exact published
  `mlx-native = 0.10.16`. The backend-independent exact-teacher target storage
  and model-free allocation binding remain structural and cannot invoke the
  proposer without authenticated source-precision completion, sensitivity,
  measurement, and quality evidence
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
- At the time this ADR was accepted, the converter accepted or could internally
  select GGUF tensor encodings that the then-pinned `mlx-native 0.10.11` did
  not read. Confirmed examples were explicit Q4_1 and
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

The then-exact `mlx-native = 0.10.11` dependency had affine packed-weight kernels
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

### Dynamic 3.0 evidence reviewed on 2026-08-19

Unsloth's [Dynamic 3.0 GGUF documentation](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)
provides useful independent evidence for the shape of this decision. Its public
description reports a model-specific mixed-precision PTQ process, a larger
multi-domain imatrix corpus refined for agentic coding, chat, and multilingual
inputs, and held-out evaluation (including long documents) with both KL
divergence and 32 tokens of free-running greedy generation over 300 prompts. It
also explicitly warns that instruct calibration must use the model's chat
template and that a Wikipedia-like calibration set can make evaluation on
Wikipedia misleading.

The transferable conclusions are:

- raw corpus bytes are not a complete calibration identity; the exact native
  template rendering and token stream are inputs to the quantizer;
- calibration and evaluation inputs must be independently hashed and checked
  for leakage;
- one-step top-1 agreement can miss trajectory drift, while perplexity can hide
  offsetting token flips, so fixed-horizon greedy trajectory and KL gates are
  required in addition to perplexity;
- the useful allocation unit is per tensor or tensor group, and policies are
  model-specific rather than universal bit-width recipes;
- formats chosen partly for Apple/ARM execution must still be benchmarked on
  hf2q's exact Metal kernels and workloads.

The page does not disclose a complete selection/search algorithm, the exact
per-tensor policy, or enough build material to reproduce every published
candidate from source. Although the page says its imatrix is available, the
exact [Qwen3.8-27B-GGUF repository revision
`27af057e`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/27af057ecb382ddfea5d12837360a8980560e3ed)
reviewed here did not list an imatrix or importance-matrix file. Its
`Divergence-300 @32` prose describes the workload but does not publish a
precise scalar formula. hf2q therefore does not claim to implement or
reproduce Dynamic 3.0. It adopts the evidence lessons and defines its own
deterministic receipt metric: for a suite-bound set of prompts, the source and
candidate each generate exactly N greedy tokens with early stopping disabled;
the receipt stores the exact-match prompt count and total common-prefix token
count, and the selector derives both rates from integers.

The pinned
[`Qwen3.8-27B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/27af057ecb382ddfea5d12837360a8980560e3ed/Qwen3.8-27B-UD-Q4_K_XL.gguf)
at repository revision
`27af057ecb382ddfea5d12837360a8980560e3ed` is nevertheless a useful output
oracle. The 17,559,178,144-byte artifact has LFS SHA-256
`3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` and its
866 tensors use nine storage types: 360 F32, 110 Q8_0, 56 Q6_K, 191 Q5_K, 69
Q4_K, 3 Q3_K, 70 IQ4_XS, 6 IQ4_NL, and 1 IQ3_S. Assignments are
tensor-specific and non-monotonic: embeddings, output, attention-value, and
FFN tensors do not simply follow one global bit tier. The histogram was derived
by opening the exact remote GGUF header with a local GGUF metadata reader
and counting tensor type codes; it is a manual research observation, not a
checked-in hf2q validation receipt. It proves that the published policy is
substantially more expressive than hf2q's current two-level heuristics. Public
material does not independently prove that those exact assignments are useful
or optimal on mlx-native; for example, a policy containing 70 IQ4_XS tensors
is ineligible until the pinned runtime proves the required QMV, QMM, width-N,
and family routes for those exact shapes without hidden fallback.

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
   cosine similarity, fixed-horizon greedy trajectory agreement, and
   perplexity-ratio thresholds;
3. exact agentic tool name, schema, arguments, tool-result continuation, and
   unary/SSE semantics;
4. context retrieval, cache-prefix reuse, and cold/cached continuation parity;
5. multimodal grounding and image/cache isolation when the family supports it;
6. every owner- or family-required behavioral regression suite.

Thresholds and corpora are versioned inputs to the evidence contract. Passing
a phrase-based refusal screen, producing valid JSON with wrong arguments, or
matching only a few sampled completions is not sufficient evidence.

For every calibrated candidate, receipt schema v2 requires and cross-checks
identifiers for the raw corpus, the exact template-rendered UTF-8 stream, the
canonical token-id stream, and a calibration manifest. The planned canonical
calibration manifest binds source identity, structured examples and
licenses/splits, seed/order/context/chunking, collector version and accumulation
order/dtype, tensor and expert coverage/counts, and the final imatrix payload.
The planned per-tensor precision-policy manifest binds tensor name, shape,
role/layer, candidate and selected codec/group size, bytes/effective BPW,
sensitivity/error evidence, required runtime route, and the reason for any
promotion or protection. For MLX affine storage the encoding's bit width and
group size are defaults; all heterogeneous overrides are authoritative in that
manifest. The recipe records an ordered calibration pipeline so cascades such
as dynamic allocation, AWQ, and DWQ are not collapsed into one algorithm label.

Quality evidence independently requires and cross-checks identifiers for an
evaluation manifest, deduplication policy, overlap receipt, KL receipt, and
per-prompt greedy-trajectory receipt. When required by the selection profile,
nonzero calibration/evaluation overlap is ineligible. KL is reported as mean,
p95, and maximum with prompt and token counts; a single average cannot hide a
catastrophic tail.

Schema v2 currently validates the shape, exact identifier agreement, metric
arithmetic, thresholds, and evidence depth. Canonical manifest types, content
rehashing, overlap recomputation, and propagation into the conversion receipt
remain Phase A.1/Phase D work. Until those producers and verifiers land, a
syntactically valid digest is an asserted identity, not independent proof of
the referenced content.

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

### 7. Learned quantization methods share one candidate boundary

After the full-model affine baseline is correct, the active implementation
order is:

1. affine RTN control;
2. dynamic mixed-precision allocation;
3. AWQ and GPTQ candidates only when separately measured evidence justifies
   them.

Per the owner decision on 2026-08-19, DWQ is not part of the current
implementation program, dependency set, release claim, or acceptance
requirement. Dynamic allocation, ordinary PTQ materialization, held-out repair,
and Apple selection must be complete without it. The common candidate boundary
may accommodate a future separately authorized DWQ experiment, but no DWQ
trainer or scale/bias-distillation work is implied by this ADR phase.

All production implementation remains Rust plus `mlx-native`. External tools
may be reference oracles in benchmark harnesses only. Each algorithm consumes
the exact source and emits the same manifest/receipt schema, so the selector is
agnostic to how the codes/scales were produced.

If DWQ is authorized later for a 27B-class model on 128 GiB unified memory, its
own amendment must bind teacher-target storage, teacher/student residency,
calibration bounds, optimizer state, peak memory, and owner/family behavior
suites before implementation begins.

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
- calibration corpus handling and authorized RTN/imatrix/dynamic/AWQ/GPTQ
  candidate producers,
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

The `mlx-native 0.10.11` release originally established a generic `QuantizedWeight` and
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
- Receipt schema v2 requires separate rendered-text and token-stream
  identifiers, integer fixed-horizon greedy trajectory evidence,
  distribution-aware KL evidence, exact manifest identifiers, and rejects
  asserted dataset overlap when the profile requires that gate. Content
  rehashing and overlap recomputation remain explicit follow-up work.

This phase changes no conversion format and makes no new speed claim.

### Phase A.1 — close current Gate-0/Gate-1 seams

- Pin and consume `mlx-native 0.10.16`'s published capability and resolved
  dispatch-trace surface from
  hf2q as a machine-readable converter/runtime tensor-type contract. Its
  `GgufFile::from_file`, exact host raw/F32 tensor reads, unified routing
  policy, and GGML capability schema are the current source-to-stored/runtime
  seam. Keep a candidate ineligible unless every required tensor and regime is
  executable.
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

- Port dynamic sensitivity measurement and allocation with exact raw-corpus and
  rendered-token-stream hashes. The default instruct corpus must be
  multi-domain, template-rendered, versioned, and large enough to cover the
  declared agentic/chat/multilingual/long-context profile; the current generic
  `cdv3` corpus remains a control, not an assumed optimal default.
- Search per-tensor or tensor-group precision policies against a held-out suite
  using marginal quality gain and the measured Apple kernel cost of the exact
  tensor shape/regime. Validate the final policy on a second untouched suite;
  do not tune and report on the same prompts.
- Preserve per-expert MoE activation statistics and counts through policy
  allocation; averaging experts into one vector is not eligible evidence.
  Calibration-required policies fail closed on missing tensors or insufficient
  expert coverage rather than warning and continuing.
- Remove the current Q8_0 proxy from source calibration where the family loader
  cannot consume F16/BF16 expert weights, or bind that proxy transformation as
  a distinct candidate and prove it against the exact source. An unrecorded
  proxy is not teacher evidence.
- Add AWQ and GPTQ only behind the common candidate/evidence contract.
- Do not add DWQ in this program. A future DWQ candidate requires explicit
  owner authorization plus a separate exact-teacher, memory, and behavior
  evidence amendment; production Dynamic autoquant cannot depend on it.
- Compare algorithms at identical encoding, group size, artifact/runtime,
  corpus, and workload wherever the question is algorithmic quality.
- Report KL, one-step top-1, the schema-v2 fixed-horizon greedy trajectory
  metrics, and required behavioral suites. Perplexity remains supplementary and
  cannot by itself make a candidate eligible.

#### 2026-08-19 — bounded Dynamic allocation proposer substrate

The first Phase-D implementation slice is deliberately narrower than a
calibrator or production `--quant auto`. It establishes the fail-closed search
and evidence boundary needed before either can be truthful:

- the allocation problem must match its supplied canonical source-catalog
  identity and count, with every supplied tensor occurring exactly once,
  including atomic tied/fused groups and packed or separately stored experts.
  Source ingestion must independently bind that catalog to the checkpoint
  tensor bundle before production use;
- each option binds the exact source, calibration/sensitivity definition,
  Apple execution identity, workload profile, runtime capability profile, and
  sufficient dense or per-expert activation coverage;
- storage and execution are separate. Allocation schema v4 embeds canonical
  source -> converted -> stored -> loaded -> executed physical manifests and
  lets tensor options reference verifier-derived lineage slices only. Payload,
  terminal tensors, source closure, and runtime-operation bindings are derived
  from those manifests rather than repeated in caller-authored shadow fields;
- fused operations such as gate+up carry one operation receipt and contribute
  their measured cost once, while still covering every member tensor;
- proposal search is a bounded exact multi-choice Pareto dynamic program over
  variable-option payload bytes, fixed-point local loss, and every required
  Apple workload regime. Exact dominance and exact metric-equivalence collapse
  are the only reductions. One deterministic tensor assignment represents each
  equal proxy-metric vector, with collapsed equivalents counted for later
  diversity repair. Exceeding the configured
  live-state bound returns `FrontierLimitExceeded`; it never silently becomes
  greedy or truncates the frontier;
- every selected option retains the physical-manifest and lineage-slice
  identities plus complete route, measurement, and sensitivity evidence in the
  policy hash. A verifier independently regenerates the frontier from the
  allocation problem and rejects mutation.

This closes only the **proposal substrate**. By itself it does not produce
structured calibration evidence, imatrices, KL gradients, quantized artifacts,
repair a policy against full-model validation, or authorize a serving
candidate.
Unsloth Dynamic 3 motivates model-specific heterogeneous allocation and
disjoint calibration/evaluation; its public material does not disclose a
reproducible selector that hf2q can import. hf2q therefore keeps the exact
owner-supplied checkpoint—including an abliterated or otherwise modified
checkpoint—as teacher, generates its own source-bound evidence, and leaves
final eligibility to held-out quality/behavior gates plus matched whole-model
Apple measurements.

#### 2026-08-19 — structured calibration and coverage producer substrate

The next Phase-D slice freezes the inputs that later sensitivity, repair, and
acceptance producers are allowed to consume. It still does not claim to
measure Dynamic sensitivity:

- structured examples preserve message roles, native tool definitions and
  results, thinking mode, template arguments, provenance, license, domain, and
  exact example order. Conversion and calibration share one fail-closed chat
  template resolver with the authoritative priority sidecar ->
  `tokenizer_config.json` -> family fallback. The v1 GGUF/runtime metadata
  surface represents one template string, so Hugging Face named template maps
  are rejected identically in both paths instead of selecting an implicit
  `default` or `tool_use` template;
- every split is rendered through the same production chat renderer used by
  serving and tokenized from the same bytes used to compute the exact source
  tokenizer/template bundle. Receipts retain enough ephemeral material to
  independently rerender and recompute every raw, rendered, token, and stream
  digest. Nested JSON insertion order is deliberately evidence-significant:
  production templates expose that order to the model, so this contract does
  not relabel recursively sorted JSON as semantically equivalent;
- calibration, policy-repair validation, and final acceptance holdout are
  hash-bound split identities checked at runtime. Partition verification
  rejects overlap by upstream source-record identity, content-only message/tool
  payload, rendered text, or fixed-width token window and rejects any source,
  template, tokenizer, renderer, token-bound, or window-bound mismatch;
- the source tensor inventory is constructed only by reading every tensor from
  an opaque `VerifiedSourceManifest` snapshot and hashing its actual bytes,
  name, source-order shape, dtype, and size. Partitioning then binds complete
  atomic unit membership and packed-expert topology alongside explicit fixed,
  protected, or excluded source dispositions. Execution codec is intentionally
  absent here; the later source -> stored -> loaded -> executed manifest owns
  that claim;
- coverage contracts consume a validated partition plus an explicit structural
  collector topology. Caller-supplied observation records must match every
  declared operation id, graph path, tensor mapping, dense row floor, and
  per-expert row floor exactly. This prevents substitutions within the declared
  topology, but D1 does not authenticate activation arrays or prove that the
  declared topology is the family graph. A family-owned Qwen collector plus an
  opaque materialization verifier remains mandatory before these records are
  accepted as measured attention or DeltaNet coverage;
- allocation schema v3 originally recorded both the three-way dataset
  partition and the full tensor partition. The solver still treats these as
  opaque identities;
  structural binding therefore requires
  `validate_structural_dynamic_allocation_bindings`, which
  regenerates the dataset partition and coverage receipt, validates the source
  tensor partition, and cross-checks every child hash. SHA-shaped substitutions
  are rejected. The returned opaque structural type has no allocation
  entrypoint; authenticated family/runtime evidence remains a separate gate.

This is a **model-free producer substrate**, not a completed calibration run.
It does not yet contain the real Qwen3.8 variable-unit/tap catalog. The exact
Qwen3.8 source-precision teacher, full differentiable QDQ graph, sensitivity
receipts, materialized mixed policies, repair loop, untouched acceptance
results, typed execution manifest, and matched Apple measurements remain
mandatory later gates. In particular, the current Qwen inference loader's
hidden stored-to-Q4_0 conversions must be made explicit before Apple cost can
guide allocation.

#### 2026-08-19 — physical execution-lineage schema v4 substrate

The next model-free slice replaces schema v3's flat per-source transform
claims with a structurally validated physical execution graph. This is the
minimum truthful representation needed before Apple runtime costs can guide a
Dynamic policy. Its first family scope is Qwen3.8 dense autoregressive text;
MoE expert stack/split layouts remain unadmitted until equally typed transforms
and exact runtime evidence exist:

- each candidate embeds one canonical manifest for an atomic allocation unit.
  Stage-tagged nodes record source, converted, stored, loaded, and executed
  representations; named transform ports record the physical DAG; stable
  logical operations have separate typed prefill/decode regime bindings;
- source tensors are rebound to the authenticated D1 inventory by exact name,
  source-order shape, dtype, byte length, and byte hash. Every represented
  candidate source must be a D1 variable tensor with the matching disposition.
  Fixed, protected, and excluded physical lineage plus their base artifact
  bytes/costs remain a D2b materializer responsibility and are not silently
  included in this frontier;
- every option contains one lineage-slice reference per source member. The
  verifier derives the complete stored and executed node sets, exact source
  closure, runtime operation bindings, capability-binding bundle, and variable
  payload bytes from the embedded manifest. Every stored/executed node must be
  covered, every manifest is used by exactly one candidate option, and fused
  multi-source storage is counted once by physical node identity;
- physical sizes are checked from a single canonical
  `row-major-outermost-first-v1` shape order. GGML block sizes are computed per
  row, artifact byte regions must be in-bounds and non-overlapping, and float
  GGML wire tensors remain representable alongside quantized blocks. Qwen
  DeltaNet conv1d rank changes use an exact singleton-axis `Squeeze` transform;
  the generic architecture-bake tag cannot hide a shape change;
- Qwen's fused-Q load path is represented as stored dequantization, explicit
  q/gate splitting, and two distinct `Qwen35LoadQ4Amax7V1` requantizations.
  The amax/7 transform is deliberately not mislabeled as canonical converter
  Q4_0. Direct packed FFN block loads remain byte-identical edges;
- the graph is acyclic and stage-monotonic, runtime source closure and terminal
  consumption are exact, and workload regimes are typed. Every catalog entry
  must share the exact routing policy, graph configuration, capability schema,
  capability profile, hardware profile, and execution scope declared by the
  allocation problem. Each regime cost names the exact physical binding ids
  and hashes their full binding bundle; caller-authored route, shape, or
  invocation-count shadows are not accepted. Hardware and measurement hashes
  remain structural identities rather than authenticated benchmark evidence.
  Malformed transform arity, roles, geometry, codec, or layout fail closed;
- schema v4 rejects MLX-affine execution and any DWQ overlay. This work contains
  no DWQ training, artifact, runtime, or acceptance path.

The result is named `ValidatedTensorExecutionManifest`, not `Verified`, on
purpose. D2a validates structure, canonical hashes, and cross-object bindings;
it does not yet authenticate converter payload bytes, loader materializations,
Metal uploads, or mlx-native request/decision semantics. The capability
envelope is only opaque canonical JSON plus a digest in D2a; it makes no typed
mlx-native ABI claim, regardless of the version string recorded by a fixture.

D2b and D2c must instrument the real hf2q converter and Qwen execution path,
rehash actual source/GGUF/loaded/executed bytes, and recompute typed
mlx-native capability decisions under the exact routing policy. Even after
the bounded D2c producer below, schema v4 cannot authorize a Dynamic cost,
candidate artifact, or production `--quant auto` choice without completed
runtime, measurement, quality, and materialization receipts.

#### 2026-08-19 — dense-Qwen source-to-stored evidence producer

The first D2b implementation slice authenticates only the source-to-stored
tensor segment of that larger contract. It is deliberately not a loader,
runtime-cost, sensitivity, or mixed-policy result:

- an opt-in dense-Qwen conversion path consumes the opaque D1 source inventory
  and partition, reopens authenticated owned safetensors bytes, executes the
  production mapper and typed bake operations, and streams one tensor at a
  time through the production writer. Ordinary conversion does not pay the
  evidence hashing or retention cost;
- the writer records exact post-bake F32, mandatory quantizer F16-roundtrip,
  and packed payload hashes. The finalized temporary GGUF is parsed through
  one already-open file identity; its tensor directory, exact payload region,
  host-decoded logical F32 values, and whole-container identity are checked
  before artifact and sidecar promotion. Source payload vectors and quantizer
  workspaces are dropped per tensor;
- persisted replay does not trust the self-hashed JSON to name a codec or
  payload. It reopens the authenticated source, reruns the standard Q4_K_M or
  Q8_0 policy plus the same production F16/quantizer encoder, and compares the
  resulting bytes to the promoted GGUF. Replay requires the exact hf2q commit;
  selector relabeling, commit substitution, sidecar mutation, and GGUF
  mutation fail closed;
- Q4_K_M is correctly treated as a heterogeneous policy, not as a claim that
  every tensor is Q4_K. Each tensor receipt records the actual policy-selected
  wire type. MTP tensors may be stored only when D1 classifies them fixed or
  protected; vision/excluded source tensors remain explicit mapper drops;
- the GGUF-wide hash is a container identity used to bind tensor offsets and
  payloads. D2b v1 explicitly does **not** authenticate derivation of
  non-tensor GGUF metadata such as model-card, tokenizer, generation, or chat
  template fields. Serving admission must validate those inputs separately;
  this receipt cannot be cited as whole-GGUF metadata provenance;
- the source-to-stored receipt admits only standard Q4_K_M and Q8_0 with no imatrix,
  calibration, learned affine state, or DWQ overlay. It does not splice a
  mixed candidate artifact, connect receipts to schema-v4 allocator options,
  or authorize any Apple performance comparison. Loaded/executed observations
  are owned separately by the subsequent D2c producer and do not retroactively
  widen this receipt's metadata or runtime claims.

#### 2026-08-20 — bounded dense-Qwen loaded/executed/encoded producer

The D2c producer joins the verified stored catalog to one deliberately bounded
dense-Qwen text execution. It closes byte and host-command-encoding lineage;
it does **not** complete or time Metal work and therefore does not authorize a
cost or a Dynamic proposal:

- persisted source-to-stored replay retains one already-open GGUF identity.
  The copied Qwen loader consumes that parser, checks the authenticated source
  configuration before and after load, rejects legacy synthetic vocabulary
  extension, and rehashes the same file identity after the load closure. The
  opaque candidate exposes neither the raw model nor mutable weight buffers;
- every stored tensor loaded by the production path is observed with its exact
  source name, D1 disposition, shape, codec, physical byte hash, decoded F32
  hash, and materialization count. Repeated shared-head loads are accepted only
  when every physical field is identical. Fixed/protected MTP weights remain
  explicit authenticated loaded tensors but are marked non-executed by the
  base autoregressive profile; variable MTP is rejected;
- cache construction rehashes the actual CPU/GPU values. It proves retained
  embedding and norm values, byte-identical direct packed dense-FFN blocks,
  DeltaNet's two conv transposes and five amax/7 projection packs, the fused-Q
  parent split into role-distinct q/gate branches followed by two production
  amax/7 Q4_0 packs, and the output-head pack. Ordinary sources have exactly
  one executed terminal; fused Q has exactly the two declared terminals;
- one canonical execution configuration resolves the full typed
  `GgmlRoutingPolicy` once. The evidence scope passes that exact policy to
  explicit-policy production entrypoints and fixes its graph choices: dense
  prefill keeps separate gate/up projections, decode may use the supported
  fused pair, fused QKVG and diagnostic split/chunk routes are disabled, and
  the standard vector paths cannot be changed by later environment mutation;
- an evidence session is bounded to one prompt and one single-token decode.
  Each weight projection yields a typed mlx-native 0.10.16 resolved-dispatch
  trace with exact operation id and executed-node ids. The verifier recomputes
  the capability decision, checks request dimensions/codec/byte minima,
  routing policy, device, resolved dispatch count, both workload coverages,
  and exact one- or two-weight topology. Duplicate operation/workload evidence,
  a missing regime, or a node substitution fails closed;
- the Apple model fixture uses the production converter, copied loader, cache,
  M=9 prompt, and M=1 decode for both Q8_0 and heterogeneous Q4_K_M. It contains
  one DeltaNet layer and one full-attention layer and exercises the typed
  Delta projections, conv roundtrip, fused-Q fanout, separate/fused FFNs, and
  output head. This is a small exact-path falsifier, not the required official
  Qwen3.8 quality or performance gate.

`GgmlResolvedDispatchTrace` proves host-side command encoding only. It does not
prove command-buffer submission or completion, numerical correctness, latency,
energy, peak memory, or a cross-process hardware identity. The generic
hardware profile still lacks the Metal registry/OS/runtime binding required by
a performance receipt. Non-tensor GGUF metadata derivation, full official
Qwen3.8 coverage, persisted D2c replay, schema-v4 manifest materialization,
mixed-artifact writing, exact-teacher sensitivity, matched Apple measurement,
and production `--quant auto` remain later gates. The allocator has no public
entrypoint from structural or D2c evidence. No DWQ overlay, learned affine
state, training, or candidate is admitted.

Only after completed runtime and quality/measurement authority is joined to
this physical path may D3 produce exact-teacher Dynamic sensitivity and
materialize candidate policies. DWQ remains outside the authorized program.

#### 2026-08-20 — D3a bounded exact-teacher target substrate

The first D3 slice freezes input and target bytes without claiming that an
authoritative source teacher exists yet:

- a structured Calibration corpus may be admitted only from one opened,
  bounded JSON artifact whose exact bytes, SHA-256, dataset id, revision,
  declared license, split, and collection counts reproduce. The license is a
  declaration authenticated by the artifact, not a separate legal
  adjudication. Rendering consumes an owned copy, so later pathname replacement
  cannot change the admitted examples;
- a prediction plan first reruns the three-way Calibration,
  policy-validation, and acceptance-holdout overlap proof, then retains token
  ids from the Calibration split only. Completed assistant token `i` is bound
  to logits from the exact prefix ending at `i - 1`; generation prompts bind
  one next-token row. Global retained-rendered-byte, token, prefix, point, and
  generation-prompt bounds use checked arithmetic before any model work. The
  current production renderer still constructs and tokenizes one complete
  example before applying those aggregate bounds, so this is not yet a hard
  peak-render-memory claim;
- full-vocabulary logits are stored as finite F32 little-endian rows in a
  framed binary artifact. The same retained file is independently reread
  before atomic no-clobber publication; this slice defines no persisted replay
  authority. The verifier rejects gaps, overlaps, reordering,
  trailing bytes, mutation, truncation, vocabulary drift, and non-finite
  values, and recomputes each row hash, deterministic argmax/top-k (ties by
  ascending token id), and F64 log-sum-exp. Greedy evidence is fixed at exactly
  32 tokens and bound to the same generation prompts;
- the artifact returned by a caller-logit writer is named and typed as
  **structural only**. Self-hashed JSON, a well-framed target file, or the tiny
  F32 CPU oracle used by tests cannot mint exact-teacher execution authority,
  sensitivity, a policy, or allocator input;
- the synthetic zero-layer, all-zero Qwen CPU oracle proves full-logit framing
  and deterministic target collection only. It does not prove a loaded Qwen
  graph and is not the Qwen3.8-27B teacher: expanding the roughly
  27B BF16 source to whole-model F32 would consume about 110 GB before
  activations and output rows and is not a safe fallback on the 128 GiB target.

The structural target seam now admits a canonical row-at-a-time producer. A
separate preflight validates vocabulary closure, row and summary cardinality,
checked artifact bytes, and every retained plan token before a future family
runner allocates model weights or Metal buffers. The preflight creates no
output. Its consumed stream accepts only the next canonical prediction point
or greedy prompt, writes and hashes one full-vocabulary row at a time, and
requires exact plan closure before it can return the same structural artifact
type. The compatibility callback writer delegates to this stream, and the two
paths are byte-for-byte and receipt-identical in the focused gate.

The prediction plan also exposes each Calibration example once with its
contiguous scored points. That seam does not authorize a full-transcript
shortcut: an authoritative source teacher must prefill the first exact prefix
for an example, then advance the same fresh per-example cache with the
ground-truth suffix and emit a row at each scored point. Running the complete
transcript as one wider causal pass, or batching output-head rows, can select a
different Metal route and remains inadmissible until separately parity-proven
and bound. These additions are allocation and streaming prerequisites only;
they still establish no execution, completion, sensitivity, policy, or
allocator authority.

The next source-precision prerequisite retains the exact opened dense-Qwen
source config and safetensors shard inodes instead of reopening tensor paths.
It applies descriptor-relative no-follow opens, strict duplicate-key
safetensors header parsing, hard shard/header/tensor/source-byte ceilings, and
checked BF16/F16 geometry. Every retained tensor name, shape, dtype, byte
extent, raw SHA-256, and Variable/Fixed/Protected/Excluded disposition must
match the already verified D1 inventory and partition exactly. Vision tensors
are authenticated but must be Excluded; MTP tensors are authenticated but must
be Fixed or Protected; an untied `lm_head.weight` is required. Later Metal
upload may copy one tensor through a bounded `u16` view while hashing those
exact copied bytes, and the retained files can be rehashed after the pass.

This retained snapshot is still structural source authority only. It does not
authenticate tokenizer/template derivation, allocate a Metal buffer, apply a
Qwen semantic or layout transform, execute or complete a graph, produce
teacher logits, persist a cross-process receipt, compute sensitivity, or open
the allocator. Its file-identity and final-rehash checks detect persistent
mutation; the later loader remains responsible for hashing the exact bytes it
copies into immutable owned buffers. B1 does not consume or interpret DWQ or
learned affine state; packed/quantized source dtypes and F32 source fallback
are rejected. Exact family tensor-topology and semantic-transform admission
remain B2 work.

#### 2026-08-20 — B2a exact BF16 source-topology admission

The first B2 slice closes the dense-Qwen source topology before allocating a
Metal buffer. It consumes and owns the opaque B1 retained snapshot and returns
an opaque, process-local `VerifiedQwen35Bf16TopologyV1`. Conversion and B2a use
the same family-owned mapper-context constructor, so authenticated explicit
`layer_types`, wrapper namespace, linear-attention head geometry, and mapper
outcomes cannot drift between the GGUF and source-teacher paths.

B2a compares the snapshot and a config-derived expected inventory in both
directions. Every admitted text source has an exact name, outermost-first
shape, BF16 dtype, D1 disposition, production mapper result, and closed future
transform descriptor. Before constructing the projected layer schedule, a
checked `3 + full_layers*11 + linear_layers*14 + mtp_layers*15` source count
must equal the already bounded non-vision snapshot. The shared authenticated
config projection rejects malformed-present optional fields, zero or
nondivisible geometry, and more than 256 layers rather than allocating from an
untrusted declaration. The official pinned Qwen3.8 config produces 866 text
source records: 851 base autoregressive tensors and 15 fixed/protected MTP
tensors. Sixteen full-attention fused Q/gate parents fan out into two
role-distinct branches, giving 867 future base buffers: 514 BF16 projection or
table buffers and 353 future F32 control buffers. The exact transform profile
is 290 value-identity/widening outputs, 161 AddOne controls, 240 grouped-to-
tiled V-head reorders, 48 reorder-then-NegExp controls, 48 squeeze-plus-partial
V reorders, 48 per-row V reorders, and 32 interleaved Q/gate branches.

The transform vocabulary is family-owned rather than a caller-supplied bake
list. It binds the exact Q/gate head-interleaving geometry and every DeltaNet
slice, row, head, kernel, and squeeze parameter. These descriptors are future
obligations only: B2a does not read or transform tensor payloads and does not
claim that a BF16-to-F32 widening, AddOne, NegExp, layout reorder, or split has
executed. B1 may retain F16 structurally, but this BF16 teacher profile rejects
every non-vision F16 source before topology authority is created.

Vision remains authenticated and exactly Excluded by the dense-Qwen HF source
predicate and production mapper. Config-declared shared-embedding MTP remains
authenticated, fixed/protected, and explicitly non-executed by the base
autoregressive profile. B2a performs no Metal allocation, payload upload,
command encoding or completion, graph execution, teacher-logit production,
sensitivity measurement, candidate materialization, allocator admission, or
`--quant auto` activation. It introduces no DWQ, overlay, affine, calibration,
or training path. B2b must stream and hash the retained bytes into owned typed
buffers while discharging these transform obligations; the later family runner
must separately prove numerical execution and completion.

#### 2026-08-20 — B2b bounded host-populated Metal upload

B2b consumes the opaque B2a topology and returns an opaque, non-cloneable
`VerifiedQwen35Bf16MetalUploadV1`. Before the first Metal allocation it
reconciles every B2a source with the retained B1 record, revalidates all seven
closed transform geometries, computes every output dtype/shape/byte count with
checked arithmetic, and applies caller and hard bounds to output count, total
logical bytes, single-buffer bytes, host availability, and the device's
reported maximum and recommended working set. The capacity comparison accounts
for exact logical output payload plus one reusable 4 MiB source-read scratch;
caller reserves are the operator-selected allowance for unmeasured Rust/Metal
bookkeeping and allocation granularity.
It is not a measured peak-memory or full teacher-runtime-fit claim.

Each output uses a fresh exact-sized, CPU-writable `StorageModeShared`
`mlx-native` 0.10.16 allocation from the bound Metal device. The safe allocator
zero-initializes the allocation and stages residency-set membership when the
device supports it. B2b then positionally streams the already-retained source
inode, reproduces its source hash, and initializes the complete logical output.
BF16 identity, grouped-to-tiled/per-row reorders, and interleaved Q/gate fanout
preserve raw 16-bit words; future F32 controls widen BF16 exactly and reuse the
production AddOne and SLEEF-compatible reorder/NegExp semantics. The squeeze
obligation changes only the authenticated output shape while the required
partial V-head reorder writes the final layout. Final buffer dtype, shape,
underlying/logical byte length, zero offset, CPU writability, non-file-backed
storage, device registry id, distinct allocation identity, and SHA-256 are
checked before the buffer can enter the opaque catalog. MTP and vision records
remain explicit zero-output entries.

No partially filled buffer or retained snapshot escapes on allocation, read,
hash, transform, capacity, or final retained-file rehash failure. The canonical
process-local receipt binds B1/B2a parents, every source/use/disposition and
output transform, actual buffer-content hashes, device name/registry id,
residency mode, limits, and observed capacity. Volatile capacity observations
are excluded from the stable content-catalog hash and included in the upload
receipt hash. No `MlxBuffer` reference is exposed because it is cloneable and
CPU-writable; only the later family-owned teacher constructor may consume the
catalog.

B2b proves bounded host population and byte verification of owned shared Metal
storage only. It does not prove a GPU dispatch, command-buffer submission or
completion, numerical graph correctness, teacher logits, latency, energy,
measured peak memory, persisted replay, sensitivity, policy quality,
materialization, allocator admission, or `--quant auto`. The official config's
preflight oracle is 867 buffers: 514 BF16 buffers totaling 53,786,705,920 bytes
and 353 F32 buffers totaling 10,582,016 bytes, for 53,797,287,936 logical bytes;
the largest single buffer is 2,542,796,800 bytes. A full official-artifact load
time/RSS/Metal benchmark and the later execution-liveness reserve remain B3
gates. This slice introduces no DWQ, overlay, learned affine, or training path.

#### 2026-08-20 — B3a inert source-teacher graph preparation

B3a consumes either the B2a topology directly through the preferred combined
B2b+B3a transition or an already-created B2b upload through a narrower
promotion seam. The combined transition derives the authenticated dense-Qwen
config, checked future weight bytes, and a caller-bounded runtime envelope
before the first Metal weight allocation. An already-uploaded B2b catalog can
only be checked for the incremental runtime envelope because its weight
allocation has already happened. Both paths return an opaque, non-cloneable
`PreparedQwen35SourceTeacherV1` with no buffer, model, session, or forward
accessor.

Preparation destructively drains every B2b output exactly once into a closed
family-owned graph: BF16 token embedding and untied output head, F32 output
norm, exact LinearAttention or FullAttention plus dense-FFN slots for every
base layer, and the small authenticated F32 Delta controls needed by the later
runner. It rehashes each actual CPU-writable Metal buffer while checking its
node, source, transform, dtype, shape, byte length, zero offset, backing,
device, and B2b content hash. Missing, extra, duplicated, swapped, mutated, or
leftover nodes reject the whole consuming transition. No `Qwen35Model`, global
GPU cache, CPU-to-GPU constructor, Q4 repack, TQ path, or MTP execution is used.
The 15 config-declared MTP sources remain authenticated and non-executed; vision
remains authenticated and excluded.

The stable prepared-graph catalog hash binds the B1 snapshot, device-free B2a
topology, exact projected execution config (including floating-point bit
patterns), ordered role-to-node/content bindings, tensor counts and bytes, and
the fixed source-BF16/F32, no-Q4, no-DWQ, no-TQ, base-text scope. It excludes
volatile device and capacity observations. A separate preparation-receipt hash
binds that graph catalog to the B2b upload catalog and receipt, device identity,
and requested runtime envelope. This distinction permits identical source,
config, and graph content to have a stable identity across different devices
or capacity observations without losing the process-local preparation
evidence.

The v1 runtime envelope is deliberately narrow: at most 4,096 sequence tokens,
16,384 target rows, a 16 GiB target-artifact payload declaration, and 256 MiB
of Delta CPU control mirrors. It accounts for base full-attention cache
payload, base DeltaNet ping-pong state, the maximum input activation, one F32
vocabulary row, the control mirrors, and a caller-selected allowance for
unmeasured builder, arena, allocator, and residency costs. These are checked
logical lower bounds, not a reservation, measured peak, exact liveness proof,
or claim that the current 4,105-token validation workload is admitted by this
version. The target-artifact payload is not counted as resident memory, and
B3a does not prove how a later runner writes it.

B3a is an inert assembly authority. It proves no graph encoding, dispatch,
submission, completion, numerical result, streaming behavior, teacher target,
latency, energy, runtime fit, sensitivity, policy quality, Dynamic admission,
selector behavior, or `--quant auto`. B3b must first join and preflight the
opaque calibration prediction plan and target limits before the approximately
53.8 GB upload, then add a fixed source-specific base-text graph, fresh
base-only cache/state, exact-prefix teacher forcing, one-row output-head
materialization, and terminal Metal completion before any teacher authority is
minted. This slice introduces no DWQ, overlay, learned affine, or training path.

The first B3b prerequisite makes that pre-upload join structural and explicit.
Teacher-prediction plan schema v2 retains the exact `SourceIdentity` and
verified source-manifest hash already authenticated by the rendered
Calibration split. Rendering now recomputes the tensor-bundle identity from
that verified manifest before retaining either identity; both participate in
the plan's canonical hash and public validation. A consuming, opaque Qwen work
preflight compares those identities to B2a, validates vocabulary and target
framing, and computes checked counts for examples, forward calls, input tokens,
output-head evaluations, cache tokens, target bytes, and the fixed 32-token
greedy horizon. Those exact expectations remain in the consuming capability so
the later runner can compare observed work without reconstructing the plan.
V1 rejects a fresh full-attention prefix shorter than 16 tokens because the
current 256-wide Qwen path has no accepted bulk kernel for lengths 2 through
15. The work capability owns the exact topology and plan so neither can be
substituted after validation.

This prerequisite performs no target-file creation, Metal allocation, graph
preparation, encoding, completion, or publication. Its hash is work identity,
not teacher, numerical, performance, sensitivity, Dynamic, selector, or
`--quant auto` authority. The production B3b entrypoint must additionally
preflight the destination and runtime capacity, begin an unpublished target,
consume this exact work capability into B2b/B3a, execute with fresh base-only
state, terminally complete every required row, and publish only after the final
family receipt is built. No DWQ state or path is introduced.

The second B3b prerequisite makes target publication the final consuming
transition. Target begin refuses an existing destination before model-weight
or Metal allocation, canonicalizes that destination parent for stable reporting,
retains its exact directory descriptor as the publication location, and creates
its private temporary through that descriptor. Later changes to a lexical
symlink alias cannot redirect the retained destination. `finish_unpublished`
flushes, syncs, bounded-hashes, and
independently rereads that exact inode while leaving the destination absent. It
returns an opaque owner whose structural receipt can be joined to the future
family completion receipt. Under cooperative same-user directory writers,
dropping it best-effort removes its private name after an identity check.
`publish_noclobber` reverifies the inode and parent, syncs the retained file and
private directory entry, and then uses one descriptor-relative atomic
no-replace rename as its last fallible operation. That namespace transition
removes the private name and creates the final name for the same retained inode
without any rollback-by-path race. The returned open file, rather than a later
pathname lookup, remains the structural authority. Directory-entry durability
across a process or host crash is not claimed; a crash-visible target without
the future in-memory family receipt is uncommitted and cannot mint authority.
Pre-publication temporary cleanup is best-effort and assumes cooperative
same-user writers in the destination directory. The existing callback writer
uses the same two transitions and still returns structural-only authority. This
is process-local publication continuity, not a persisted sidecar/replay, graph,
completion, numerical, sensitivity, Dynamic, selector, or DWQ authority.

The third B3b prerequisite closes the base-text cache allocation boundary.
One checked, device-independent layout plan is now shared by B3a runtime
preflight and the actual cache constructor, so the two byte formulas cannot
drift. B3a derives that plan from its authenticated config; the cache
constructor does not clone or change the config it is given, including its
declared MTP fields, but allocates exactly one F32 base-text sequence: the
scheduled full-attention K/V slots plus the DeltaNet conv and recurrent
ping-pong state. It allocates no MTP slot, TQ buffers, speculative capture, or
auxiliary sequence. A fallible per-slot reset establishes zero Delta state and
canonical ping-pong parity before an opaque owner can be returned. Sealing then
checks the exact layer-to-slot schedule, every buffer's shape, dtype, logical
extent, backing, CPU writability, and Metal device,
plus zero cursors, absent MTP/TQ/capture state, checked payload totals, and
canonical layout/receipt hashes. The official dense 64-layer, 4,096-token
profile is pinned at 16 full-attention slots, 48 linear-attention slots,
536,870,912 full-attention bytes, 313,786,368 linear-state bytes, and
850,657,280 total bytes. Production exposes no cache or buffer from the opaque
prepared owner; a `cfg(test)`-only consuming seam supports the private parity
harness, while the production run-input owner remains inert. A later consuming
runner transition must join the cache to the opaque prepared teacher and
remains required. The cache object alone proves a fresh,
config-relative, process-local host-visible Metal layout only—not source
authority, residency, graph
dispatch, completion, logits, peak memory, performance, persisted replay,
sensitivity, Dynamic, selector, or DWQ authority.

The fourth B3b prerequisite makes those structural capabilities one ordered,
opaque set of future run inputs. Its sole production constructor consumes the
source-bound work proof, derives the exact cache-token and target-row bounds
from that proof, and creates an empty descriptor-retained target reservation
before invoking the B2b/B3a weight transition. The reservation writes only the
target magic under a private name and hash-binds the prediction-plan identity,
vocabulary, limits, row/trajectory counts, and exact final byte length. It
owns no logits and cannot finish or publish; the later runner must consume it
and rebind the unchanged opaque prediction plan to obtain the row stream. This
avoids cloning token material or constructing a self-referential plan/stream.
The reservation is revalidated after weight preparation, after cache
preparation, and again at plan rebind: the canonical parent and private inode,
exact magic bytes, receipt hash, and continued absence of the final name must
all reproduce. A same-length in-place mutation or destination created during
the expensive preparation window therefore rejects before execution.
After the weights are prepared, capacity is observed again with the exact
upload reserves retained by B3a, and the cache is allocated only from B3a's
private authenticated config and exact `MlxDevice`. Sealing checks the work,
topology, projected config, prepared graph, device, cache layout/bytes, target
reservation, and expected counters, then owns the plan, reservation, weights,
and cache inseparably. Failure at any point drops the private target and all
partial Metal ownership; the final destination remains absent. Stable catalog
identity excludes the pathname and volatile capacity/device receipts, while a
separate process receipt binds the B3a preparation, cache allocation, and
device. That process receipt also binds the fresh post-weight host/Metal
capacity observation, exact accounted runtime payload and unmeasured-runtime
allowance, retained upload reserves, checked host/Metal requirements, and
available Metal bytes; the stable catalog deliberately excludes those
volatile observations. This is a successful admission observation, not a
reservation or a runtime-liveness/peak proof. This run-input object remains
inert: it exposes no model, buffer,
cache, target writer, or forward method and proves no encoding, submission,
completion, logits, finished/published target, runtime liveness, peak memory,
performance, sensitivity, Dynamic, selector, autoquant, or DWQ authority.

The first execution-route prerequisite aligns DeltaNet source-BF16 decode
projections with the already-established full-attention and dense-FFN policy.
For BF16 weights and F32 activations, M=1 projections with an even output width
use mlx-native 0.10.16's paired-row `dense_gemv_bf16_f32`; M>1 and odd-width
projections remain on `dense_matmul_bf16_f32_tensor`. The odd-width fallback is
deliberate because the published GEMV shader forms two source-row pointers per
threadgroup before guarding the second write. Every official Qwen3.8 DeltaNet
projection width is even. Both allocating and caller-destination helpers use
one process-state-independent selector and reject non-F32 activation/output
buffers before invoking the native BF16 route. U8/GGML and legacy F32-cast
paths are unchanged. This establishes a safe route prerequisite only: without
a paired benchmark it makes no speed claim, and it proves no complete graph,
terminal completion, teacher target, sensitivity, Dynamic, selector,
autoquant, or DWQ authority.

The second execution-route prerequisite adds a `cfg(test)`-only,
non-authoritative source-BF16 parity harness. A distinct tagged thread-local
graph scope is cross-nonreentrant with the copied-GGML evidence scope and
rejects every Qwen
GGML/fused-quantized projection wrapper, and fixes the switches routed through
the shared Qwen execution-dispatch layer: no chunk scan, fused QKVG, fused
quantized gate/up, or dense-Q split profile; dense-Q arena reset, the
small-vector path, and fused full-attention stages A/B remain enabled. Other diagnostic
switches remain outside this prerequisite. The call substrate itself accepts
only the prepared base-text graph/cache, so MTP, TQ, and vision remain absent.
The scope deliberately does
not claim a complete native route: mlx-native 0.10.16 still resolves internal
BF16 tensor-MM choices outside this hf2q state.

The test-only call path consumes only the already-prepared BF16 projection and
F32-control buffers plus the prepared base-text cache. It widens only the
requested BF16 embedding rows into one checked F32 `[tokens, hidden]`
activation, never a whole embedding table. It runs the exact configured
Delta/full-attention and dense-FFN layer schedule, retaining Delta ping-pong and
full-attention KV state across calls. The output boundary slices only the last
hidden row, applies the authenticated F32 RMSNorm and untied BF16 output
weight, materializes one F32 vocabulary row, and terminally waits before host
readback. No `[sequence, vocabulary]` output is allocated.

The non-skipping Apple gate uses a finite authenticated two-layer fixture with
hidden size 256, intermediate size 512, vocabulary 32, one Delta layer, one
full-attention layer, and authenticated-but-nonexecuted shared MTP. On a fresh
named worker thread it proves finite, nonzero full-vocabulary logits, identical
top-1, and maximum absolute error at most `5e-3` against `forward_cpu` for both
a 16-token prefill and the cached one-token continuation at position 16. The
fixture also requires that independently removing the DeltaNet output, full
attention output, or dense-FFN outputs changes the CPU-oracle vocabulary row by
more than the accepted `5e-3` bound, so the parity gate cannot pass merely from
the residual stream. This is wiring and numerical-parity evidence for a
test-only call. No production forward or raw-cache transition is compiled by
this prerequisite. The path does
not traverse the verified prediction plan, own the one-shot worker/panic
lifecycle, reconcile complete work counters, finish or publish a target, or
mint a family teacher capability. Its caller-selected runtime allowance is not
a peak-memory or liveness proof. Official 27B execution, full one-shot
completion, route/performance evidence, target authority, sensitivity,
Dynamic/selector/autoquant admission, and DWQ remain out of scope.

The first authority-bearing B3b runner now closes that deliberately test-only
boundary with one consuming family transaction. Its only entrypoint accepts
the sealed run-input owner by value and moves the authenticated prediction
plan, retained source snapshot, prepared BF16/F32 graph, base-text cache, and
private target reservation onto a fresh named worker thread. The source graph
scope supplies a non-Clone, non-`Send`, lifetime-bound token. Both the raw-cache
transition and every source call require that token, so neither state can be
detached from the canonical tagged thread-local policy. The worker cannot be
resumed or used as a generic logit callback, and no model, cache, Metal buffer,
target stream, or logit row escapes it.

V1 makes each hf2q source-path command buffer a checked synchronous completion
point. This is intentionally slower than serving: mlx-native 0.10.16 does not
retain an error ledger for earlier asynchronous command buffers after their
handles are dropped, so a later empty wait alone cannot truthfully authorize
their status. Synchronous completion also keeps every local activation and
scratch owner alive through the corresponding Metal work even when native
retained references are disabled. The final output head still materializes
only one F32 vocabulary row. The worker retains the session outside its
`catch_unwind` boundary; a returned error or Rust panic poisons the session and
performs a same-queue terminal drain before cache, weights, thread-local
scratch, or the private target may be dropped. A failed drain cannot mint
completion. Process aborts and foreign Objective-C exceptions are not claimed
recoverable.

Execution follows the retained plan exactly. Every example receives a
fallible fresh base-cache reset. A completed transcript prefills its first
scored prefix once, advances one ground-truth token for every subsequent
prefix—including unscored gaps—and evaluates the output head only at retained
prediction points. A generation prompt evaluates and stores its one required
row, uses that row's canonical finite, lowest-token-ID-on-tie argmax as greedy
token zero, then performs exactly 31 one-token continuations for a 32-token
trajectory. Observed examples, resets, calls, input tokens, output-head
evaluations, terminal call completions, rows, trajectories, and cache
high-water must exactly equal the pre-upload work record. A second canonical
schedule hash binds every actual stable ID, position, input token, and
emit/advance decision; after target finalization it is independently
reconstructed from the retained plan and stored trajectory.

After all calls complete, the worker terminally drains again while the private
target stream remains live, verifies the exact counters, finishes the
structural target under its private name, verifies the exact schedule,
rehashes the same retained config/shard descriptors, drops the GPU graph/cache,
and constructs a non-deserializable family completion receipt.
That receipt joins the work, topology, plan, projected config, prepared graph,
cache, target reservation, source snapshot, graph policy, exact structural
receipt/artifact, device, and expected/observed work. It records source BF16
projections with F32 controls and base-text F32 cache, checked synchronous
source commits, and explicit `q4=false`, `ggml=false`, `dwq=false`, `tq=false`,
`mtp_executed=false`, and `vision_executed=false`. The worker returns only a
completed-but-unpublished private owner. Its caller joins the worker and then
performs the existing descriptor-relative no-replace rename as the final
fallible operation; only the infallible post-rename wrapper creates the opaque
completed authority. That authority retains the exact source snapshot, opaque
prediction plan, and published target inode.

The non-skipping Apple authority gate drives the finite authenticated H256
Delta-plus-full-attention fixture through the complete production entrypoint.
For the canonical two-example plan it checks exactly 34 calls, 64 processed
input tokens, 34 output-head evaluations, three full-vocabulary rows, cache
high-water 47, and one 32-token trajectory. All rows and the trajectory match
the BF16-derived CPU oracle with identical top-1 and row maximum absolute error
at most `5e-3`. A separate gapped transcript proves 35 forwards but only 34
head evaluations and three rows. Persistent retained-source mutation rejects,
an injected panic after a real completed call drains and publishes nothing,
and a destination created after worker join remains byte-exact when the final
no-replace publication rejects.

This is bounded, process-local source-teacher target authority, not official
Qwen3.8-27B acceptance. The receipt deliberately does not claim a complete
native kernel route, allocator or peak-memory proof, timing/performance,
sensitivity, Dynamic/selector/autoquant admission, cross-process replay, or
DWQ. The real 27B Apple run, matched external-reference parity, reproducible
resource measurement, and downstream one-option-at-a-time sensitivity remain
required before the wider D3/D4 policy stages can consume this lane.

Completing D3a for the official model now requires the pinned 27B Apple run and
matched source-reference validation through this family-owned path; an F32
GGUF/CPU run remains only a tiny-model oracle or an explicitly preflighted
small-model comparison. D3b then materializes exactly one D1 atomic option at
a time and computes full-distribution KL, top-1, and trajectory metrics. The transferable
Unsloth lessons are multi-domain native-template calibration, heterogeneous
precision, separate tuning/validation/holdout data, and full-distribution plus
trajectory gates—not an unpublished selector algorithm. No DWQ, learned
affine overlay, or training path is introduced.

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
nominal compression, or best theoretical bandwidth. The current completion
contract does not include DWQ; future learned-affine candidates, if separately
authorized, would compete under the same quality-before-speed gates rather than
becoming an assumed speed feature.

The design applies unchanged to vanilla and weight-modified checkpoints. An
intentional source behavior is preserved because the exact source is the
teacher and its required suites are gates, not because hf2q recognizes the
transformation's name.

Full production auto quantization is not claimed by Phase A. The remaining
work is measurable engineering—artifact ownership, kernel routing,
calibration, and real-model evidence—not an unresolved conceptual dependency.
