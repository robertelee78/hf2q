# What MLX affine could add to hf2q

Date: 2026-10-06. Research and planning only; no implementation, model
execution, or locally reproduced performance result.

Governing decision: [ADR-046](../adr/ADR-046-evidence-driven-apple-auto-quant.md).
The [existing support audit](mlx-affine-support-2026-09-23.md) remains the
implementation inventory. This supplement answers the narrower investment
question: **what could affine add when hf2q already runs packed GGUF weights
through its native Apple-Silicon backend?**

## Conclusion

Affine offers a simpler weight representation, directly represented per-group
quantization parameters, and access to the standard MLX affine artifact and
kernel contract. These can support faster unpacking and additional
quality/storage tradeoffs. They do not establish an inference speedup over
hf2q's existing GGUF kernels.

The closest same-runtime evidence found is particularly instructive: an
M5 Max implementation initially had a large affine advantage because its
K-quant path lacked the equivalent tensor-operation kernel. Adding that
kernel brought K-quants approximately level with affine at the tested shape.
Other synthetic work shows an advantage for simpler affine decoding, but
does not establish a full-model, quality-matched benefit.

**The evidence supports evaluating affine as a competing representation. It
does not support promising a percentage improvement, making affine the
default, or treating support itself as completion of the user's performance
objective.** Runtime-versus-runtime rankings are not evidence of affine's
incremental value to hf2q.

## The actual difference

MLX affine reconstructs each group as `w = scale * code + bias`, with packed
integer codes and separate scale/bias arrays. GGUF Q4_K also uses affine
reconstruction, but derives each sub-group's scale and offset from a
hierarchically packed super-block. Thus this is a comparison of particular
representations and kernels, not affine mathematics versus its absence.
[MLX quantization API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html),
[GGML block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h).

| Potential benefit | Why it could help | What must be established for hf2q |
|---|---|---|
| Less decoding work inside a matmul | Direct group parameters avoid extracting and reconstructing K-quant sub-scales/minima | Whether those instructions currently limit the actual target kernels; simpler arithmetic does not imply faster bandwidth-bound execution |
| A regular layout for specialized QMV/QMM | Contiguous packed codes and regular companion arrays give another tiling/vectorization target | A faster packed native implementation at the target shapes, including prompt processing and small physical batches |
| Additional quality/size choices | Bits, group sizes and independent floating-point group parameters give a different candidate space | Better useful quality per resident byte or less latency at the required quality; mixed precision is already possible with GGUF |
| Preserve compatible affine artifacts | Directly retain their codes and scale/bias values, including parameters produced by more advanced quantizers | Correct production conversion/import, metadata handling, complete graph execution and cache identity |

Compatibility has value, but is not sufficient justification for the user's
performance objective. Learned quantization is also separate from merely
supporting the encoding; DWQ remains outside this implementation program.

Affine is not required to use Metal, unified memory, fused dequantization,
small-batch weight reuse, or tensor operations. hf2q already has GGUF
implementations of these facilities. The research must not count them again
as benefits obtained simply by adding affine.

## Closest external evidence

### 1. Same-runtime affine/K-quant prefill: mlx-node on M5 Max

At revision `be0162f2dcd4c6639c7b921af059f83617144f6c`, mlx-node documents
a BF16 matrix test with `M=512, N=8192, K=8192`. Disabling the K-quant NAX
route made it take 3.40–3.49 times the affine time. With that route available,
the benchmark records K-quant/affine time ratios of 0.95–1.09 across three
sweeps. The control is affine 4-bit/group-32; the other arms are q4k/q5k/q6k.

The checked-in harness has synthetic weights, six warmups, nine rounds,
twelve evaluations per measurement, rotated ordering, two affine controls,
and trimmed means. It does not measure model quality or full-model serving;
the arms are not equal bit budgets. Its permissive 2.0 regression ceiling is
not a measured speedup or an hf2q acceptance target.

Inference: equivalent kernel coverage can remove a large apparent affine
advantage. This does not prove parity at decode or every shape.
[Pinned report](https://github.com/mlx-node/mlx-node/blob/be0162f2dcd4c6639c7b921af059f83617144f6c/docs/perf.md#k-quants-and-the-nax-tensor-op),
[benchmark source](https://github.com/mlx-node/mlx-node/blob/be0162f2dcd4c6639c7b921af059f83617144f6c/crates/mlx-core/tests/kquant_nax_bench.rs).

### 2. A simpler affine layout can win: M4 format-router experiment

The m4-prefill-engine project reports these single-projection timings for
its K=4096 synthetic case:

| Input rows | Q4_K time | Affine-style time | Calculated time ratio, Q4_K/affine |
|---|---:|---:|---:|
| 33 | 1.117 ms | 0.848 ms | 1.317 |
| 128 | 2.280 ms | 2.030 ms | 1.123 |
| 2048 | 24.622 ms | 21.744 ms | 1.132 |

Important limitation found by reading its source: its `MLX_4BIT` is a custom
interleaved group-32 block, 5.0 bits/weight, versus Q4_K's 4.5. It is not the
standard MLX three-array storage ABI. Weights are generated synthetically per
format. This is evidence for a simpler affine-style kernel representation,
not a quality-matched MLX-affine model win, a decode result, or an M5 forecast.
[Reported measurements](https://github.com/mohamedhossammohamed/m4-prefill-engine/blob/1138a30c271175a3e33e3e6178c9b7240e3c5e1d/README.md),
[format definitions](https://github.com/mohamedhossammohamed/m4-prefill-engine/blob/1138a30c271175a3e33e3e6178c9b7240e3c5e1d/quant_router.h),
[benchmark source](https://github.com/mohamedhossammohamed/m4-prefill-engine/blob/1138a30c271175a3e33e3e6178c9b7240e3c5e1d/bench_universal_router.mm).

### 3. Quality can reverse the size argument: mlx-kquant

The author's Qwen3.6-27B comparison reports:

| Recipe | Effective bits/weight | Mean KL divergence from BF16 |
|---|---:|---:|
| MLX affine 4-bit | 4.69 | 0.0577 |
| Q4_K_M with importance calibration | 4.88 | 0.0208 |
| MLX affine 5-bit | 5.68 | 0.0214 |

For these recipes, affine 5-bit approaches Q4_K_M's distribution fidelity
while using about 16.4% more bits/weight. That could outweigh a modest kernel
advantage in bandwidth-bound decode. This is an inference, not a measured
latency penalty. Recipes/calibration differ; this does not prove every affine
quantizer loses. KL divergence is not coding success, and its ratio is not a
multiple of behavioral accuracy. The evaluation used WikiText, not the
required multi-turn coding suite.
[Primary comparison and protocol](https://github.com/ml-explore/mlx/pull/3588#issue-contents).

### 4. Group choice has measured performance consequences within affine

MLX issue #3251 reports M2 Ultra/MLX 0.29.3 results comparing uniform
group-128 affine against mixed group-32/64/128 models of approximately the
same size. Generation was 7–14% lower for the mixed recipes; some prefill
times roughly doubled. This changes multiple per-layer choices, so it does
not isolate group size perfectly, and it is not current-M5 qualification.
It does show why a quality-improving group policy cannot be assigned the
performance of another affine recipe.
[Original measurements](https://github.com/ml-explore/mlx/issues/3251).

### 5. Affine kernel improvements are useful references, not format benefits

Upstream MLX PR #3764's refreshed table reports a 1.7x INT4 kernel speedup
on M5 Max at four input rows for a Gemma MLP shape. It changes weight reuse
inside the same encoding. Separately, Qwen3.8-27B experiments in issue #4265
report sequence-width-8 forward time dropping from 77.1 to 43.3 ms; their
custom matrix kernel is approximately tied at width four and slower at one.
These are useful implementation references. Neither compares affine with
hf2q's GGUF kernels, and speculative sequence width is not concurrent-client
count. Do not use these ratios as affine's expected incremental gain.
[Merged kernel change](https://github.com/ml-explore/mlx/pull/3764),
[corrected dependent-chain/model measurements](https://github.com/ml-explore/mlx/issues/4265#issuecomment-5301594380).

### Other projects screened out of the affine benefit estimate

BaseRT, SiliconBench, Rapid-MLX, LM Studio, vLLM Metal, vllm-mlx, Higgs and
AXEngine were examined for comparable measurements. Runtime comparisons,
scheduler/cache changes, speculative decoding and hardware differences
prevent attributing their headline gains to affine. Even same-GGUF-weight
MLX-versus-llama.cpp results do not measure the benefit of adding affine to
hf2q. These results therefore supply **no percentage target** here.

## What the storage arithmetic actually says

For b-bit weights with two 16-bit parameters per group of g weights,
the payload rate is `b + 32/g` bits/weight, before alignment, padding,
container metadata and protected tensors. This is arithmetic, not a benchmark.

| Encoding | Payload bits/weight | Consequence versus a Q4_K block |
|---|---:|---|
| Q4_K | 4.50 | Reference block rate; Q4_K_M is a mixed model recipe |
| Affine b4/g32 | 5.00 | 11.1% more payload bytes |
| Affine b4/g64 | 4.50 | No payload-byte advantage |
| Affine b4/g128 | 4.25 | 5.6% fewer payload bytes |

Under the deliberately idealized assumption that runtime is entirely weight
traffic with identical achieved bandwidth, 4.50/4.25 gives a 1.059x rate
ratio. That is **not a forecast**: quality, actual mixed tensor policies,
unpacking, activation traffic, attention/KV state and scheduling can dominate.
F32 metadata changes the formula to `b + 64/g`; silently widening metadata
can erase the intended storage benefit. Actual resident bytes must be measured.

## Implications for current hf2q

The local comparison is bound to hf2q main
`cca6425860c1fb770c6223c2bbdfa5d2d769dc70` and published `mlx-native 0.15.1`,
native commit `f92eb020c4d3f821700e648fbce15d6cf75c2cc6`.

- Native GGUF dispatch already contains Q4_K/Q5_K/Q6_K tensor-operation
  routes (`src/ops/quantized_matmul_ggml.rs`) and small-width weight-reuse
  kernels (`src/ops/mul_mv_ext.rs`). Source presence does not prove which
  route a particular future benchmark actually executes; record dispatch.
- [ADR-044](../adr/ADR-044-qwen38-native.md) already records qualified Qwen
  GGUF width-4–8 verification using `mul_mv_ext`. Comparing a future affine
  candidate only with an older ordinary-decode number would omit existing
  production optimizations. Freeze the currently supported matched modes.
- Current packed affine primitives and the separate `qmm_affine` path do
  not yet form the complete artifact/dtype/regime contract documented in
  ADR-046. Existing local affine fallbacks are not a fair proxy for what a
  completed implementation could achieve, nor proof it will beat GGUF.

No controlled, end-to-end, quality-matched affine-versus-current-hf2q result
was found or produced in this research. The external kernel comparisons are
useful evidence about mechanisms, with the limits above.

## Revised qualification plan for later execution

The user selected Qwen3.8 dense text, M5 Max with 128 GiB, and both interactive
agentic coding and four simultaneous coding sessions. The previously suggested
10% improvement and 5% regression numbers were unsupported proposals and
are withdrawn; no numerical product threshold has been approved.

Preserve the ADR's C.0 then C.1 ordering, making C.1 explicitly answer
**affine's incremental value** before broad integration:

1. Freeze current hf2q/native revisions, source weights, model policy,
   actual tensor sizes and shapes, activation/metadata dtypes, and resolved
   GGUF routes. Identify whether the four-client path forms physical matrix
   batches; client concurrency must not be substituted for matrix M.
2. After C.0 numerical fixtures, compare GGUF and packed affine in the same
   native harness at M=1, small widths including four, and representative
   prefill shapes. Include the currently optimized GGUF route. Keep
   activation dtype, accumulation, hardware state and timing boundaries
   consistent; expose casts, unpacked copies and fallbacks.
3. Separate a representation diagnostic from product quantization. A
   carefully constructed common-value fixture can isolate packing/kernel
   effects; independent real-source quantization requires source-teacher and
   behavioral quality checks. Preserve rounding semantics explicitly when
   deriving flat parameters from hierarchical scales; do not assume bitwise
   equivalence after storing their products in a narrower dtype.
4. Compare candidate frontiers at both actual bytes and required quality.
   An affine 4-bit artifact must not win by losing required model behavior;
   an affine 6-bit remedy must pay for all its extra bytes in the comparison.
5. Before acceptance runs, predeclare the quality and service-level gates
   using the existing workload contract and baseline variability. Publish
   every required regime, warmed repeated measurements and dispersion.
   Kernel speedup, full-model decode, cached semantic TTFT and completed
   four-session throughput remain distinct measurements.
6. Continue to C.2–C.6 only with an explicit rationale from C.1. A plausible
   kernel improvement warrants a complete graph trial, not a success claim.
   If equivalent GGUF kernels erase the gain, record that result and
   reconsider a performance-driven affine rollout. Compatibility work can
   still be valuable, but needs its own product justification.

No user-approved release scope is silently cancelled by that last condition.
It identifies the technical decision the future experiment must inform.
Task decomposition and release acceptance remain subject to completing the
readiness discussion; this research is not approval to execute feature work.

## Research validation and handoff

Primary project reports, exact source files and benchmark harnesses were read.
Numerical ratios and storage arithmetic were checked on the host. External
performance figures remain author-reported; no Metal benchmark or model was
executed for this document. No Rust, Metal, dependency or serving change is
part of this supplement.

The useful evidence is the same-runtime kernel comparison and its source,
the limited affine-style representation experiment, and the quality/storage
tradeoff. None supplies an evidence-backed universal percentage for hf2q.
