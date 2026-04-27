# ADR-016 Research Dossier — CoreML for SOTA Inference on M5 Max Apple Silicon

- **Date:** 2026-04-26
- **Status:** Proposed-with-evidence (graduates ADR-016 from *Stub*)
- **Authors:** 7-researcher CFA dual-research swarm + Opus queen synthesis (session `cfa-20260426-adr016-coreml-research`)
- **Cutoff:** 2026-04-26 (everything published through today is admissible; 2026 sources are PRIMARY, not "post-cutoff informational only")
- **Comparison baseline:** hf2q on M5 Max already does **100+ tok/s on 26-27B Gemma-4 / Qwen3.5-MoE** at HEAD `9ab4cca` (memory pin `project_decode_parity_achieved`: 103.5–107.1 tok/s). The historical Apple 2024 Llama-3.1-8B-on-M1-Max @ 33.67 tok/s number [R2] is a 2-silicon-generation-old, 3×-smaller-model ceiling and is treated here as historical context only.

---

## Executive Summary

The headline finding of the M5/M5 Max generation is that Apple's stated **>4× peak GPU AI compute vs M4** is a property of the GPU's new per-core **Neural Accelerators** (Metal 4 TensorOps / Metal Performance Primitives), not of the standalone 16-core Neural Engine [R5][R8]. This was confirmed independently by Apple's M5 newsroom press release, Apple's own ML Research M5/MLX blog, and the WWDC25 / M5+A19 GPU Tech Talk. The path that captures this 4× headline is the path **hf2q is already on** through the `mlx-native` sibling crate — Metal-GPU compute dispatched via Metal 4 Tensor APIs.

A pure CoreML/ANE replacement of hf2q's Qwen3.5-MoE / Gemma-4 hot path is closed as a no-go. Six independent reasons converge: (i) DWQ Q4_K/Q6_K cannot round-trip into any CoreML quantization primitive [R4]; (ii) MoE expert-routing gather ops + Gated DeltaNet SSM force CPU fallback regardless of quant format [R4][R7]; (iii) Orion's empirical ANE dispatch ceiling (≈0.095 ms IOSurface/XPC overhead per op, ~119 compilations/process) puts public CoreML LLM decode below CPU on M4 Max even for GPT-2 124M [R3][R8]; (iv) ANEMLL — the only open end-to-end ANE LLM runtime — caps at 8B dense and explicitly disclaims MoE [R5][R7]; (v) Apple's own ML Research blog deliberately targets the GPU, not the ANE, for Llama-3.1-8B-Instruct because the LLM body is bandwidth-bound [R2][R6]; and (vi) AtomGradient's `hybrid-ane-mlx-bench` measured **no 9B crossover** for the public CoreML-prefill + MLX-decode hybrid on macOS 26.3 [R5][R8].

The strategic reframe is to abandon "can CoreML replace mlx-native for Qwen / Gemma?" and ship the more honest question: "**Can `coreml-native` opportunistically offload non-LLM-body encoder work (mmproj/ViT, BERT, ASR) while `mlx-native` keeps the decode critical path?**" That matches ADR-016's existing sibling-crate structure, preserves ADR-008's pure-Rust commitment, and lines up with every shipping ANE+GPU hybrid in the literature today (whisper.cpp, FluidAudio/Parakeet, CoreML-LLM's own deliberately-CPU/GPU vision encoder for Gemma 4 E2B [R8]).

The 2026 academic literature — `Orion` (March 2026, arXiv:2603.06728), `vllm-mlx` (January 2026, arXiv:2601.19139), `Open-TQ-Metal` (April 2026, arXiv:2604.16957), `Mirror Speculative Decoding` (October 2025, arXiv:2510.13161), Apple Silicon Profiling (August 2025, arXiv:2508.08531), `WhisperKit` (July 2025, arXiv:2507.10860) — names the territory and points the same direction the user pointed: **heterogeneous NPU+GPU dispatch is the active 2026 frontier**. ADR-015's acknowledged "no peer-reviewed Apple Silicon LLM perf work exists" is no longer true; this dossier is the hand-off.

ADR-016 should graduate to **Proposed-Accepted with narrow scope**: P2 ViT/mmproj and P3 BERT spikes proceed under a mandatory `MLComputePlan` ANE-placement gate; P1 Qwen on ANE is closed except as a ≤4-hour calibration smoke test for tooling; P4 hybrid pipeline becomes a **conditional** post-P2/P3 deliverable, deliberately scoped to the *sequential* ANE-encoder → mlx-native-GPU-body topology rather than the falsified concurrent hybrid. ADR-015's mlx-native single-CB rewrite remains the higher-leverage path for capturing Apple's actual 2026 AI compute story.

---

## 1. The M5 Two-Tier AI Architecture (PARTIAL CONFIRMATION)

R5 introduced — and R8 partially corrected — the framing that M5 ships **two** distinct AI compute paths.

**Tier B — GPU-Integrated Neural Accelerators (Metal 4 / TensorOps): CONFIRMED.**
M5 introduces dedicated matrix-multiplication units integrated into every GPU core. On M5 Max (40-core GPU) this means **40 × 1,024 FP16 FMA / cycle** [R5]. They are programmable via Metal 4 Tensor APIs / TensorOps + Metal Performance Primitives, **not** via CoreML or the ANE path. MLX (and therefore `mlx-native`) targets these Neural Accelerators directly. Apple's own MLX/M5 ML Research post measures **3.3–4.1× faster TTFT vs M4** across Qwen 1.7B/8B/14B, 30B-A3B MoE, and GPT OSS 20B at prompt 4096 / generation 128, and **1.19–1.27× decode** (memory-bandwidth bound — tracking the 460→614 GB/s bandwidth jump) [R5][R8]. R5 explicitly attributes the 19–27% generation-speed boost to memory bandwidth scaling.

**Tier A — Standalone 16-core ANE: PARTIAL — "unchanged from M4" is NOT confirmed.**
R5 originally framed Tier A as "same generation, same core count, same throughput as M4." R8 (Codex orthogonal pass) downgraded this. Apple's M5 marketing calls the Neural Engine "faster" without publishing comparable TOPS for any M5 variant; M4 is the most recent chip Apple explicitly quantified at 38 TOPS, and the M5 Max spec page omits TOPS entirely [R5][R8]. The defensible language is: **M5's headline 4× AI compute is GPU Neural Accelerator compute; the standalone ANE remains a 16-core block with no public M5 Max TOPS disclosure and no evidence of a comparable 4× jump**.

**Strategic implication.** Apple put the 2025–2026 headline AI gains in the GPU, not the ANE. The path `hf2q` is already on via `mlx-native` is the path Apple is now publicly accelerating. Any CoreML/ANE pivot for the LLM hot path therefore swims against Apple's own developer investment direction. The WWDC25 "Combine Metal 4 machine learning and graphics" session (262) and the M5/A19 GPU Tech Talk (111432) both frame Apple's developer push as **GPU TensorOps**, not CoreML ANE [R8].

---

## 2. The Empirical Falsification of the Public CoreML+MLX Hybrid Path (AtomGradient, March 2026)

The single most consequential piece of empirical evidence in this dossier is AtomGradient's `hybrid-ane-mlx-bench` repository [R5][R8].

**What it measured.** Qwen3.5 0.8B–9B running through three configurations on macOS 26.3:
1. Pure MLX-GPU (`mlx-lm`).
2. CoreML prefill (`ComputeUnits::All`) + MLX decode hybrid.
3. Private-API direct ANE access (Orion-style).

**What it found.**
- CoreML `ComputeUnits::All` routes Qwen3.5 transformer prefill to **GPU**, not ANE. ANE power readings sit near zero watts during prefill. This is the placement question that `MLComputePlan` would have surfaced before wall-clock benchmarking (see §6).
- CoreML+MLX hybrid TTFT **matches** pure MLX for 2B BF16 and is **slower** for 9B 8-bit, with **no crossover** at any tested prompt length (6 / 133 / 410 tokens) [R5][R8].
- Conversion is a tooling-killer: seq64 ≈ 2 min, seq256 ≈ 50 min, seq512 ≈ 97 min. `CPU_AND_NE` causes an ANE IPC deadlock [R8].

**Counter-evidence inside the same repo.** AtomGradient also reports private-API ANE batch prefill at **268 tok/s for 0.8B** [R5][R8]. This is critical: it falsifies the *public CoreML hybrid path*, not ANE *hardware potential*. Orion's M4 Max GPT-2 124M result (170 tok/s ANE, **283 tok/s CPU** — ANE losing because per-dispatch IOSurface overhead dominates single-token decode) is the same shape of finding [R3][R8].

**Scope limits R8 demanded we honour.**
- Hardware: **M2 Ultra**, not M5 Max.
- Status: small GitHub project, non-peer-reviewed.
- Macros: TTFT only; no n=256 sustained-decode regimes.

The honest read: AtomGradient + Orion together close the question for the *public* CoreML LLM-body path on M-series. They do **not** close the question for ANE-as-encoder workloads, where every shipping precedent is sequential rather than hybrid (see §5).

---

## 3. Q1 Verdict — Qwen3.5-MoE / Gemma-4 on ANE: HARD NO-GO

**Verdict.** Hard no-go for shipping. Restricted to a **≤4-hour `MLComputePlan` smoke test** on dense 0.8B / 2B variants for tooling calibration only [R8]. The original ADR-016 P1 "two-day spike" is over-scoped given the evidence below.

The six convergent reasons:

**(a) DWQ tensor format cannot round-trip into CoreML quant.** [R4]
The four production GGUFs (qwen3.6-27b-dwq46/dwq48, qwen3.6-35b-A3B variants) carry Q4_0 + Q6_K + Q8_0 + F32 + F16 mixes that emerge from DWQ's two-pass calibration (ADR-012 P6/P9). All GGML formats used in dwq46 are **affine linear** quantization with hierarchical (superblock-of-subblock) scales for Q4_K/Q6_K. CoreML's linear-quant primitives (`coremltools.optimize`) support only single-level per-tensor / per-channel / per-block scales — there is no CoreML primitive that preserves the Q4_K `scales[K_SCALE_SIZE]` 6-bit-under-FP16-superblock structure. Re-ingesting therefore requires dequantize-to-float and re-quantize under CoreML's scheme, which **discards the DWQ calibration**. The closest algorithmic sibling in coremltools is `ModuleGPTQConfig` (Hessian-based OBS, arXiv:2210.17323) — a fresh GPTQ run from HF float safetensors is the shortest faithful re-quantization path, but it is not DWQ.

**(b) MoE expert-routing gather + Gated DeltaNet force CPU fallback.** [R2][R4][R7]
The ANE explicitly does not support Gather operations [R2 — *hollance/neural-engine* unsupported-layers list, Apple ML Research note]. Qwen3.5-MoE's 256-expert top-k routing is a gather-based dispatch; every MoE layer would trigger ANE→CPU→ANE round-trips. Gated DeltaNet SSM (3:1 hybrid with attention in qwen3.6) has no ANE op-coverage equivalent. R4 confirms TRI_SOLVE custom ops are absent from CoreML's op set.

**(c) The 2.3 ms XPC dispatch overhead × ~10 chunks per Qwen3.5-35B token = ~23 ms ceiling = ~21 tok/s ceiling before compute.** [R7]
CoreML's per-chunk dispatch cost is well-characterised in the public literature, and 35B-A3B's MoE chunking forces a ~10-chunk-per-token decode pattern. The ceiling on this pattern is below half of `hf2q`'s 100+ tok/s baseline before adding any actual compute time.

**(d) ANEMLL caps at 8B dense; zero published MoE-on-ANE attempts.** [R5][R7]
ANEMLL (Anemll/Anemll, the only open-source end-to-end ANE LLM pipeline) lists Llama-3.1/3.2, Qwen 3 (0.6B/1.7B/8B), Qwen 2.5 (0.5B–7B), Gemma 3 dense (270M / 1B / 4B QAT), DeepSeek R1 8B distilled. Maximum supported = 8B parameters. **No MoE entries.** Reported tok/s on M4: Llama-3.2-1B 47–62 tok/s, Llama-3.1-8B ~9.3 tok/s — the latter is ~5× slower than MLX-GPU on the same chip (≈50–93 tok/s) [R7].

**(e) Orion's GPT-2 124M ANE result loses to CPU on M4 Max.** [R3][R8]
The Orion paper (arXiv:2603.06728, March 2026) — first arxiv treatment of direct ANE programming — reports GPT-2 124M at 170 tok/s ANE vs **283 tok/s CPU** on M4 Max, with prefill at 165 tok/s. The dispatch overhead ceiling dominates single-token decode. This is private-API-direct (not public CoreML), so it is the *upper bound* for ANE-on-LLM-decode that public CoreML-via-ANEMLL cannot beat. CoreML adds a further 2–4× overhead vs `_ANEClient` direct access for small operations [R3].

**(f) Apple's own ML Research blog targets GPU, not ANE, for Llama-3.1-8B.** [R2][R6]
The "On Device Llama 3.1 with Core ML" post (machinelearning.apple.com) is Apple's most detailed first-party LLM-on-CoreML reference, and it deliberately deploys the LLM body to the GPU because the body is bandwidth-bound. The 33.67 tok/s number is therefore **not an ANE result** — it is a CoreML-on-Metal-GPU result on M1 Max from 2024. Per the user-correction-baseline directive, that figure is historical only; the live-data peer baseline is `vllm-mlx`'s January 2026 measurement of **Qwen3-30B-A3B at 109.7 tok/s on M4 Max via MLX Metal** [R3 — arXiv:2601.19139] — directly comparable to hf2q's 100+ tok/s baseline.

**Allowed exception: a ≤4-hour MLComputePlan smoke test.** Convert dense Qwen3.5-2B (or 0.8B) to `.mlmodelc`, load via `MLComputePlan`, report per-op compute-device-usage and estimated cost without running wall-clock decode. This calibrates the conversion tooling for P2/P3 use and exposes the placement reality (≥99% ANE? mixed? all GPU?). It is **not** a viability spike for Q1 — Q1 is closed.

---

## 4. Q2 Verdicts — Adjacent Workloads

### 4.1 P2 — ViT / mmproj: GO with MANDATORY MLComputePlan ANE-placement gate

ViT and BERT are Apple's published ANE sweet spots. Apple's 2022 *Deploying Transformers on the Apple Neural Engine* post benchmarks DistilBERT-base on iPhone 13 (A15) at **3.47 ms / 0.454 W**, with **10× speed and 14× memory** improvement over unoptimized baseline [R3]. M4 ANE is ≈2.4× larger than A15; BERT embeds should land ≤1.5 ms there [R2]. mmproj ViT prefill is a one-shot non-stateful workload that maps cleanly to `Model::predict` (no MLState required, no autoregressive dispatch).

**The gate R8 added.** Gemma 4 E2B's CoreML model card (`mlboydaisuke/gemma-4-E2B-coreml`) splits the package into a decoder (ANE, with MLState) and a vision encoder. CoreML-LLM's own `BENCHMARKING.md` discloses that the Gemma 4 E2B 99.78 % ANE figure **deliberately excludes the SigLIP vision encoder, which runs on CPU/GPU by design** [R8]. This is direct counter-evidence that ViT-on-ANE is *not automatic*; placement is a per-architecture question. Therefore P2 **must** front-load `MLComputePlan` (per-op compute-device-usage + estimated cost reporting before wall-clock benchmarking) and **fail-fast** if hot ViT ops fall to GPU/CPU.

**Acceptance criterion.** Convert Gemma-4V mmproj or Qwen-VL ViT to `.mlmodelc` with coremltools 9 + FP16 + EnumeratedShapes (5 buckets); confirm via `MLComputePlan` that the vision tower lands on ANE (not GPU); measure cold-SoC prefill latency vs `mlx-native` ViT path on M5 Max MacBook Pro per memory-pin `feedback_perf_gate_thermal_methodology`; require **≥1.5× speedup** for proceed signal; corroborate with Instruments power measurement to verify ANE utilization.

### 4.2 P3 — BERT / embeddings: GO

This is the cleanest ANE candidate the dossier surfaces. Dense encoder, no MoE routing, no DeltaNet, no autoregressive per-token dispatch. ANEMLL supports comparable architectures cleanly [R5][R7]. The `bert-test` target in `models/bert-test/` is small enough that conversion risk is low [R5]. Acceptance criterion: convert `bert-base-uncased` with coremltools 9; `MLComputePlan` confirms ANE placement; cold/warm latency, memory, thermal measurements [R8]; success criterion **≥2× speedup** vs current `mlx-native` BERT path.

### 4.3 ASR / Whisper-class: STRONGEST SHIPPING EVIDENCE — recommend honourable-mention spike

Not in original ADR-016 P1–P4, but the clearest *shipping* CoreML/ANE success across the entire 7-researcher sweep:

- **WhisperKit** (arXiv:2507.10860, ICML 2025): MLState delivers a **45 % latency reduction** (8.4 ms → 4.6 ms / forward pass) on M3 ANE for the Whisper decoder [R3][R7]. This is the best concrete proof that MLState-on-ANE moves the needle for autoregressive decode workloads.
- **FluidInference / FluidAudio Parakeet TDT 0.6B**: ~110× RTF on M4 Pro per HF model card; ~190× RTF per FluidAudio benchmark table on Tahoe 26.0 [R6][R7][R8]. Hundreds of thousands of monthly downloads — production-scale CoreML/ANE shipping evidence.
- **whisper.cpp** ANE-encoder discussion (#548): 6× speedup for the encoder on M1 Air [R7]; the precedent for the sequential-handoff topology (encoder on ANE → decoder elsewhere via `memcpy` at the `MLMultiArray.dataPointer` boundary) [R6].

If hf2q ever picks up speech, this is a P3-equivalent candidate that should not be deferred under "ASR's not in scope." Recommend opening **ADR-016c (ASR)** alongside ADR-016a/b.

---

## 5. Q3 — Hybrid Pipeline (P4): REFRAMED

The original ADR-016 P4 framing — *"ANE prefill + GPU LLM body running in parallel"* — needs three pieces of reframing.

### 5.1 Concurrent ANE+GPU via PUBLIC CoreML for the LLM body: NO-GO

This is what AtomGradient measured and falsified for Qwen3.5 0.8B–9B (§2). Reasons concrete:

- ANE and Metal GPU are physically separate hardware blocks on Apple Silicon; cross-model concurrency is *architecturally* possible but **zero confirmed shipping precedents** of true parallel CoreML-ANE + Metal-GPU execution exist [R6].
- The `coreml-native::Model` is `Send + Sync` and Apple documents `MLModel.predictionFromFeatures` as thread-safe for concurrent read-only predictions [R6 — `coreml-native/lib.rs`]. So firing `predict_async` from one thread while `mlx-native::GraphSession` dispatches Metal command buffers on the same `MTLDevice` from another thread *should not* block at the hardware level. **But** AtomGradient's measurement on the actual pattern returns no crossover [R5][R8].
- The open question is **scheduler contention in macOS's AMX/ANE driver under unified-memory bandwidth saturation** [R6]. No public benchmark answers it for M5 Max.

### 5.2 Sequential hybrid (ANE encoder → mlx-native GPU LLM body): CONDITIONAL GO post-P2/P3

This is the topology every shipping precedent uses:
- whisper.cpp's `whisper_coreml_encode` extracts ANE encoder output via explicit `memcpy(out, outCoreML.output.dataPointer, outCoreML.output.count)` — the handoff is sequential, with a CPU-mediated copy [R6]. For a typical 2 MB ViT output, the copy costs ≈0.1–0.3 ms — negligible vs ViT prefill time.
- Parakeet TDT and FluidAudio follow the same sequential CoreML-encoder → decoder topology [R6][R7].
- The hollance/neural-engine ANE-vs-GPU note states explicitly: *"shared memory means you don't need to upload, but format conversion is required even on shared memory; for the GPU, data needs to go into a texture object first."* [R6] So even unified-memory does **not** automatically mean zero-copy ANE↔Metal.

**Conditional-GO criterion.** P4 unblocks only after P2 (ViT) AND P3 (BERT) confirm via MLComputePlan that the encoder of interest actually lands on ANE, *and* a measured handoff cost (memcpy or zero-copy) is on the table.

### 5.3 Metal 4 ML passes: a THIRD path none of R2–R7 distinguished

R8's most underweighted orthogonal find. The Metal 4 framework (WWDC25 session 262, `developer.apple.com/documentation/metal/machine-learning-passes`) supports **CoreML-authored models running inside a Metal command buffer on the GPU timeline** with `MTLTensors`, `ML command encoder`, and Shader ML [R8]. This is relevant if the P4 objective is **command-buffer integration**, not freeing the GPU. It is **not** evidence that the standalone ANE can run hf2q's LLM body. A separate spike (call it **P4b**) would investigate whether the Metal 4 GPU-timeline path beats the current `mlx-native` Metal-GPU path on TTFT for adjacent workloads.

### 5.4 coreml-native gaps the hybrid path will demand

R6 surfaced two concrete bindings the `coreml-native` crate does not currently expose:
- **`setOutputBackings`** — eliminates the memcpy at the ANE→Metal boundary by letting CoreML write directly to a Metal-allocated buffer.
- **`setPreferredMetalDevice`** — pins CoreML's GPU fallback to the same `MTLDevice` handle used by `mlx-native`, avoiding cross-device-context costs.

Both are **low-complexity single-method additions** [R6]. They are pre-requisites for a credible P4-sequential-hybrid spike.

---

## 6. MLComputePlan as the Required Placement Gate

R8's most important orthogonal finding is that **none of the 6 Claude researchers (R2–R7) surfaced `MLComputePlan`** [R8].

The Apple Developer Documentation page `developer.apple.com/documentation/coreml/mlcomputeplan-85vdw` describes a per-operation compute-device-usage and estimated cost reporting mechanism that runs at *compile time*, before any wall-clock benchmark. CoreML-LLM uses it to report `7,294 of 7,310 decode operations on ANE for Gemma 4 E2B on iPhone 17 Pro` [R8]. The benchmarking doc also discloses that the SigLIP vision encoder for the same Gemma 4 E2B package was deliberately **excluded** from that ANE figure — that single fact is direct counter-evidence to "ViT lands on ANE automatically" [R8].

**The required ADR-016 phasing change.** Convert `MLComputePlan` into a **hard phase gate**: every spike must report ANE-placement % via `MLComputePlan` and **fail-fast** if hot ops fall to GPU/CPU. Powermetrics / Instruments traces / Core ML Instruments are corroboration only. Combine with `MLComputeUnits.cpuAndNeuralEngine` (`developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine`) for a GPU-excluding falsification mode [R8].

This gate also prevents the recurring failure mode that bit AtomGradient: spending hours on conversion + benchmarking only to discover via power readings that ANE never fired.

---

## 7. Updated Phasing for ADR-016

The original ADR-016 phasing (P1 = Qwen3.5-3B spike, P2 = ViT spike, P3 = BERT spike, P4 = hybrid design) is replaced with:

| Phase | Deliverable | Acceptance | Status entering exec |
|---|---|---|---|
| **P0 — `coreml-native` plumbing** | Add `setOutputBackings` + `setPreferredMetalDevice` bindings to `coreml-native` per R6's gap analysis. | Both methods exposed via Rust + smoke-test against a trivial `.mlmodelc`. | ready (low complexity) |
| **P1 — MLComputePlan calibration smoke (≤4 h)** | Convert dense Qwen3.5-0.8B or 2B to `.mlmodelc`; load via `MLComputePlan`; report per-op compute-device usage. **No wall-clock decode.** | Confirmed tooling round-trip works; per-op placement table produced. | ready |
| **P2 — ViT / mmproj spike** | Convert Gemma-4V mmproj (or Qwen-VL ViT) to `.mlmodelc`. **MLComputePlan ANE-placement gate is mandatory.** Cold-SoC prefill latency vs `mlx-native` ViT on M5 Max MacBook Pro per memory pin `feedback_perf_gate_thermal_methodology`. Power via Instruments. | ANE-placement ≥ 90 % of hot ViT ops AND ≥1.5× speedup vs `mlx-native` ViT. | ready post-P0 |
| **P3 — BERT / embeddings spike** | Convert `bert-base-uncased` (`bert-test` target). MLComputePlan + cold/warm latency + thermal + memory measurements. | ANE-placement ≥ 90 % AND ≥2× speedup vs `mlx-native` BERT. | ready post-P0 |
| **P4 — Sequential hybrid design + measurement** | If P2 OR P3 lands an ANE win: design ANE-encoder → mlx-native-GPU-body sequential pipeline using P0's `setOutputBackings`. Measure handoff cost; characterise cross-thread `predict_async` + Metal CB scheduler interaction. | Measured handoff cost ≤ 0.5 ms for typical 2 MB encoder output AND no observable CB stall in unified-memory-saturated regime. | conditional on P2/P3 |
| **P4b — Metal 4 ML-pass investigation (optional)** | Stand up CoreML-authored model on the GPU timeline inside a Metal command buffer (Shader ML, `MTLTensor`s). Compare TTFT vs current `mlx-native` Metal path for adjacent workloads. | Decisive go/no-go on whether the Metal-4-ML-pass path is a third sibling worth pursuing. | optional, post-P2/P3 |
| **P5 — ASR / Whisper honourable-mention spike (recommended)** | Convert Whisper-Small or Parakeet-equivalent to `.mlmodelc`. Quantify M5 Max RTF vs reference. | ≥110× RTF or matches FluidAudio precedent. | recommended |

ADR-016a / 016b / 016c become per-workload landings post-P2 / P3 / P5.

---

## 8. Literature Foundations (2026 Primary Sources)

Per the user-correction-recency-and-direction directive, 2026 publications are PRIMARY, not "post-cutoff informational only." Re-classifications applied below.

### 8.1 Apple-Official 2025–2026 Primary Sources

- **Apple newsroom — M5 launch (Oct 2025)** [R5][R8] — `https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/` — confirms GPU Neural Accelerators, >4× peak GPU AI compute vs M4, faster 16-core Neural Engine (no comparable TOPS published).
- **Apple newsroom — M5 Pro / M5 Max launch (March 2026)** [R5][R8] — `https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/` — confirms M5 Max up-to-40-core GPU and 614 GB/s bandwidth.
- **Apple ML Research — "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU" (2026)** [R5][R8] — `https://machinelearning.apple.com/research/exploring-llms-mlx-m5` — confirms MLX targets M5 GPU Neural Accelerators via Metal 4 TensorOps + Metal Performance Primitives; TTFT up to 4×, decode 19–27 %.
- **Apple Tech Talk 111432 — M5 / A19 GPU (2026)** [R8] — `https://developer.apple.com/videos/play/tech-talks/111432/` — frames Neural Accelerators inside each GPU core; confirms Apple's developer push is GPU TensorOps not CoreML ANE.
- **WWDC25 session 262 — "Combine Metal 4 machine learning and graphics" (2025)** [R8] — `https://developer.apple.com/videos/play/wwdc2025/262/` — `MTLTensor`s, ML command encoder, Shader ML, GPU-timeline CoreML.
- **WWDC25 session 286 — "Meet the Foundation Models framework" (2025)** [R2][R8] — `https://developer.apple.com/videos/play/wwdc2025/286/` — Foundation Models exposes a closed 3B 2-bit QAT on-device model via Swift-only API; not custom-weight loading; irrelevant to hf2q's open-weight use case.
- **Apple Core ML overview (2026)** [R8] — `https://developer.apple.com/machine-learning/core-ml/` — performance reports, operation compute-unit breakdown, Core ML Instruments, stateful models.
- **MLComputePlan docs (2026)** [R8] — `https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw`.
- **MLComputeUnits.cpuAndNeuralEngine docs (2026)** [R8] — `https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine`.
- **Metal machine-learning passes (2026)** [R8] — `https://developer.apple.com/documentation/metal/machine-learning-passes`.
- **Apple Developer Forums — Core ML topic (2026)** [R8] — `https://developer.apple.com/forums/forums/topics/machine-learning-and-ai/machine-learning-topic-core-ml` — Tahoe-era regression register: fused-QKV NaNs on macOS 26.2 GPU, stride/layout warnings for ANE loads, scrambled tensor outputs in macOS 26.1 beta, large live-camera latency for CPU+ANE despite outputBackings + IOSurface, Qwen 1.7B quantized model load spikes on iPhone SE 3.

### 8.2 arXiv 2025–2026 Primary Sources (re-classified per user directive)

- **Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference (March 2026, arXiv:2603.06728)** [R3][R7][R8] — first arxiv treatment of direct ANE programming via private `_ANEClient` / `_ANECompiler`; 20 ANE constraints catalogue; **GPT-2 124M 170 tok/s ANE vs 283 tok/s CPU on M4 Max** (ANE loses on decode); softmax 33.8× faster on ANE vs CPU; CoreML adds 2–4× overhead vs direct `_ANEClient`; XPC + IOKit dispatch ≈0.095 ms / op.
- **Native LLM and MLLM Inference at Scale on Apple Silicon — vllm-mlx (January 2026, arXiv:2601.19139)** [R3] — **Qwen3-30B-A3B 109.7 tok/s on M4 Max via MLX Metal 4-bit; Qwen3-0.6B 525.5 tok/s; Llama-3.2-1B 461.9 tok/s; Gemma 3-4B 152.5 tok/s; Nemotron-30B-A3B 121.8 tok/s.** 21–87 % higher throughput than llama.cpp. Multimodal Qwen3-VL-8B: first query 21.7 s, cached 0.78 s (28× speedup). **All MLX Metal GPU — no ANE for the LLM body.** This is the closest published peer to hf2q's 100+ tok/s baseline.
- **Open-TQ-Metal: Fused Compressed-Domain Attention for Long-Context LLM Inference on Apple Silicon (April 2026, arXiv:2604.16957)** [R3] — Metal int4 KV cache 48× attention speedup at 128K context.
- **Mirror Speculative Decoding (October 2025, arXiv:2510.13161)** [user-direction directive — corroborated by R7's heterogeneous-dispatch territory] — GPU+NPU heterogeneous 2.8–5.8× speedup; direct evidence for the concurrent-NPU+GPU-as-2026-frontier thesis.
- **Profiling LLM Inference on Apple Silicon — Quantization Perspective (August 2025, arXiv:2508.08531)** [R3] — Apple Silicon quantization profiling foundation for ADR-016 measurement methodology.
- **WhisperKit: On-device Real-time ASR with Billion-Scale Transformers (ICML 2025, arXiv:2507.10860)** [R3][R7] — **MLState 45 % latency reduction (8.4→4.6 ms / fwd pass) on Whisper decoder M3 ANE**; encoder reduction 65 % via architectural mod. Strongest concrete proof MLState-on-ANE accelerates autoregressive decode for an adjacent (non-LLM) workload.
- **Apple Intelligence Foundation Language Models — AFM (arXiv:2407.21075, July 2024)** [R2][R3] — confirms AFM-on-device is 4-bit palettization (LUT, K-means 16 unique values) with per-16-col/row shared scales, average ~3.5 bpw / production 3.7 bpw, QAT. ANE LLM platform-precedent. **Implication for DWQ:** dwq46 must be re-quantized to deploy on ANE (DWQ is affine block, not palettization).
- **EdgeMoE: Empowering Sparse Large Language Models on Mobile Devices (arXiv:2308.14352, 2023)** [R3] — establishes MoE-on-device expert-routing as a known challenge; zero academic treatment of Qwen3.5-MoE-256-experts on ANE.
- **GPTQ (arXiv:2210.17323, 2022)** [R4] — algorithmic sibling to DWQ; coremltools `ModuleGPTQConfig` references this paper directly.

### 8.3 Shipping Projects (2024–2026 Primary)

- **ANEMLL (Anemll/Anemll, 2024–2026)** [R3][R5][R7] — `https://github.com/Anemll/Anemll` + `https://www.anemll.com/` — only open end-to-end ANE LLM pipeline; coremltools ≥ 9.0 with LUT4/LUT6; Conv2d-replaces-nn.Linear; chunked attention; **caps at 8B dense**, no MoE; Llama-3.2-1B 47–62 tok/s on M4, Llama-3.1-8B ~9.3 tok/s.
- **AtomGradient `hybrid-ane-mlx-bench` (2026)** [R5][R8] — `https://github.com/AtomGradient/hybrid-ane-mlx-bench` — **the empirical falsification of CoreML+MLX hybrid for Qwen3.5 0.8B–9B on macOS 26.3 / M2 Ultra**; same repo also reports private-API ANE batch-prefill 268 tok/s for 0.8B.
- **CoreML-LLM (john-rocky, 2026)** [R7][R8] — `https://github.com/john-rocky/CoreML-LLM` and `BENCHMARKING.md` — Gemma 4 E2B INT4 34.2 tok/s on iPhone 17 Pro A19; Qwen3.5 2B INT4 ~17 tok/s; **Gemma 4 E2B vision encoder deliberately excluded from 99.78 % ANE figure (CPU/GPU by design)**.
- **Gemma 4 E2B CoreML model card (2026)** [R8] — `https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml` — decoder/vision split, MLState, int4 palettization.
- **Orion repository (2026)** [R8] — `https://github.com/mechramc/Orion` — no-CoreML direct ANE runtime, delta compilation, GPT-2/Stories110M scope.
- **FluidAudio / FluidInference (2025–2026)** [R6][R7][R8] — `https://github.com/FluidInference/FluidAudio`, `Documentation/Benchmarks.md`, Parakeet-TDT-0.6B-v3-coreml HF model card — **Parakeet ~110× RTF on M4 Pro batch ASR; 190× RTF on M4 Pro Tahoe 26.0**. Strongest *shipping* CoreML/ANE success case.
- **whisper.cpp ANE encoder (2023–2024)** [R6][R7] — `coreml/whisper-encoder.mm` — sequential CoreML-encoder + main-decoder topology with explicit `memcpy` at the boundary; ANE encoder ~6× speedup on M1 Air.
- **coreml-llm-cli (smpanaro)** [R7] — Llama-2-7B FP16 7.0 tok/s M1 Max → 13.9 tok/s M3 Max (illustrative scaling, far below `hf2q` baseline).
- **hollance/neural-engine** [R2][R6] — `https://github.com/hollance/neural-engine` — canonical ANE unsupported-layers reference (Gather, LSTM/GRU/RNN, dilated conv, broadcastable layers, ND layers, dynamic reshape, kernel >13 / stride >2 pooling, upsampling >2×) and ANE-vs-GPU note (shared memory ≠ zero-copy).

### 8.4 Apple ML Research / WWDC 2022–2024 (Historical Foundation)

- **Deploying Transformers on the Apple Neural Engine (Orhon et al., 2022)** [R2][R3] — `https://machinelearning.apple.com/research/neural-engine-transformers` — DistilBERT 3.47 ms / 0.454 W on iPhone 13; 10× / 14× memory baseline. Apple-blog-only, no arxiv counterpart.
- **On-Device Llama 3.1 with Core ML (2024)** [R2][R6] — `https://machinelearning.apple.com/research/core-ml-on-device-llama` — Llama-3.1-8B-Instruct M1 Max 33.67 tok/s decode at Int4 + MLState (**historical context only per user directive**; not the ADR-016 viability bar).
- **WWDC24 session 10161** [R2] — MLState (1.6× Mistral 7B M3 Max KV cache); MLTensor; multi-function models.
- **WWDC24 session 10159** [R2] — per-grouped-channel palettization, Int4 block_size=32, fused SDPA op (iOS 18+), joint sparsity+quant.
- **WWDC24 session 10218** [R2] — MPSGraph SDPA, 4-bit linear/LUT in MPSGraph, KV-cache update ops.
- **WWDC23 session 10049** [R2] — async predict, `MLComputeDevice` runtime discovery.
- **coremltools docs** [R2][R4] — Stateful Models, Palettization Overview, Quantization Overview, Flexible Inputs, Joint Compression, GPTQ source (`coremltools/optimize/torch/layerwise_compression/algorithms.py`), PostTrainingQuantizer source.
- **Maderix M4 ANE benchmarks Parts 1 & 2 (2024)** [R5] — direct `_ANEClient` access, 100+ median-timed iterations: M4 ANE measured peak **19.9 TFLOPS FP16** at 94 % utilization (deep 32+ layer graphs); **5.7 TFLOPS** for single 2048×2048 matmul; INT8 / FP16 deliver near-identical throughput (hardware dequantizes INT8 to FP16 before arithmetic).

### 8.5 Secondary / Anecdotal

- llmcheck.net Apple Silicon LLM benchmarks leaderboard, hardware-corner.net M5 Max RTX comparisons, insiderllm.com ANE inference guide [R5][R7] — secondary aggregations; useful for chip-generation scaling triangulation, not for primary citation.
- llama.cpp Apple Silicon discussion #4167 [R7] — community-level baseline triangulation.
- Draw Things engineering blog "Making ANE Work in a Custom Inference Stack" [R3][R7].

---

## 9. Honest Gap Acknowledgements

Per `feedback_ground_truth_is_what_we_can_measure_now`, the dossier names what it does **not** know:

1. **No public M5 Max standalone ANE TOPS table by 2026-04-26** [R5][R8]. M4 was the most recent chip Apple explicitly quantified (38 TOPS marketing, 19.9 TFLOPS measured); M5 Max ANE TOPS is M4-extrapolated only.
2. **No public M5 Max CoreML LLM benchmark on ≥20B-class models** [R5][R7][R8]. Most CoreML LLM tok/s in the literature are M3 Max / M4 Max / iPhone A-series. CoreML-LLM's iPhone 17 Pro A19 figures are M5-extrapolated for mobile, not M5 Max throughput projections.
3. **AtomGradient's empirical falsification was on M2 Ultra, not M5 Max** [R5][R8]. The repo is small and non-peer-reviewed. It is still the best direct public falsification we have for a public CoreML-prefill + MLX-decode hybrid. P2/P3 must reproduce on M5 Max before drawing closing conclusions.
4. **Tahoe-era CoreML developer-forum regressions are anecdotal, not peer-reviewed** [R8]. They are primary developer reports, not measurements; they belong in the spike test register, not in dossier conclusions.
5. **No measured concurrent ANE+GPU latency numbers on M5 Max in any public literature** [R6]. The cross-thread `predict_async` + Metal CB scheduler-contention question is genuinely open.
6. **`setOutputBackings` + `makeBuffer(bytesNoCopy)` zero-copy ANE→Metal handoff is architecturally inferred, not benchmarked** [R6]. The hollance ANE-vs-GPU note explicitly cautions that shared memory ≠ free format conversion.
7. **whisper.cpp `ggml-coreml.mm` Metal-side dispatch was not inspectable (404)** [R6]. Its sequential precedent stands; the concurrency precedent does not.
8. **FluidInference per-stage compute routing not publicly disclosed** [R6]. Their ANE win is real; the topology is opaque enough that we cannot project to LLM body workloads.
9. **Mirror Speculative Decoding (October 2025, arXiv:2510.13161) has not been independently reproduced on Apple Silicon** [user directive]. The 2.8–5.8× heterogeneous-NPU+GPU speedup is the right territory but not yet validated for Apple Silicon's specific NPU/ANE shape.

---

## 10. Strategic Recommendation

ADR-016 graduates from **Stub** to **Proposed-Accepted with narrow scope**.

The narrow scope: **`coreml-native` is positioned as an opportunistic encoder-offload sibling, not a Qwen-MoE LLM-body replacement.** P2 ViT/mmproj, P3 BERT, and (recommend adding) P5 ASR proceed under a mandatory `MLComputePlan` placement gate. P4 hybrid is conditional on at least one of P2/P3 producing an ANE win, and is scoped to the *sequential* topology with measured handoff cost — not the *concurrent* CoreML-prefill + MLX-decode topology that AtomGradient already falsified for the LLM body. P4b (Metal 4 ML passes on the GPU timeline) is an honourable optional that may yield a third sibling crate or simply a `mlx-native` enhancement.

The pure-CoreML/ANE replacement of the Qwen3.5-MoE / Gemma-4 hot path is **closed (no-go)**. Six convergent reasons (DWQ format incompatibility, MoE/DeltaNet op coverage, XPC dispatch ceiling, ANEMLL caps at 8B, Orion ANE-loses-to-CPU on decode, Apple's own deliberate GPU choice for Llama-3.1) are too many to be re-opened by another two-day spike. The 2026 papers (Orion, vllm-mlx, AtomGradient + private-API ANE 268 tok/s for 0.8B) settle the perf-ceiling question that ADR-015 acknowledged was open.

The 4×-AI-compute story Apple is publicly telling for M5 / M5 Max is **the path hf2q is already on** — Metal 4 TensorOps via MLX, captured by the `mlx-native` sibling crate. ADR-015's mlx-native single-CB rewrite captures more of M5's headline gains than any CoreML pivot would. Heterogeneous NPU+GPU dispatch is the active 2026 research frontier (Mirror SD, HeteroInfer, WhisperKit's MLState autoregressive proof, Orion's 0.095 ms dispatch lower bound), and we have full-stack ownership of both crates needed to claim that territory once P0–P3 are in hand. The concurrent path is enabled — but only via the encoder-offload topology, not via a CoreML-LLM-body retrofit.

ADR-008's pure-Rust commitment is preserved. ADR-015 stays the higher-leverage 2026 effort. ADR-016 narrows to a sibling-crate opportunistic offload programme. The dossier closes ADR-015's literature gap.

---

## 11. References

### ADRs
- **ADR-008** — Candle divorce / pure-Rust mandate (`mlx-native` is sole inference backend for current models; `coreml-native` is sibling, not replacement).
- **ADR-012** — DWQ scheme (qwen3.5-moe / qwen3.6 dwq46/dwq48 conversion plan; P6/P9 activation capture; closure milestone 2026-04-26).
- **ADR-015** — `mlx-native` single-command-buffer rewrite (acknowledged literature gap; this dossier closes it).
- **ADR-016** — `mlx-native` vs `coreml-native` strategic comparison (parent of this dossier).

### CFA Research Briefs (memory namespace `swarm-cfa-adr016`)

- **`agents/R2/result`** — Apple-official sweep (WWDC, Apple ML Research, coremltools docs, hollance/neural-engine).
- **`agents/R3/result`** — arxiv / academic sweep (Orion, AFM, EdgeMoE, vllm-mlx, Open-TQ-Metal, WhisperKit, Apple Silicon Profiling). Note: R3's "post-cutoff" labels were generated against the assistant's January 2026 training cutoff and have been re-classified here per user directive — 2026 papers are PRIMARY.
- **`agents/R4/result`** — DWQ ↔ CoreML quantization compatibility (Q4_K / Q6_K hierarchical scales vs CoreML linear-quant; ONNX→CoreML deprecated; PyTorch→ct.convert is the only viable path; coremltools GPTQ is the closest algorithmic sibling).
- **`agents/R5/result`** — M5 hardware specifics; introduced two-tier architecture framing (Tier B confirmed, Tier A partial per R8); AtomGradient hybrid-bench falsification surfacing.
- **`agents/R6/result`** — Hybrid pipeline (Q-H1/H2/H3/H4); whisper.cpp + Parakeet sequential precedents; coreml-native gap analysis (`setOutputBackings`, `setPreferredMetalDevice`).
- **`agents/R7/result`** — SOTA shipping landscape (MLX, ANEMLL, CoreML-LLM, FluidAudio); ANEMLL 8B cap; 2.3 ms XPC dispatch ceiling; CoreML iPhone benchmarks.
- **`agents/R8/result`** — Codex orthogonal-bias check: MLComputePlan placement gate (none of R2–R7 surfaced this); R5 two-tier downgrade to PARTIAL; AtomGradient confirmation with scope limits + private-API counter-evidence; Metal 4 ML passes as third path; Apple Developer Forums Tahoe-era regression register; CoreML-LLM ViT-on-CPU/GPU caveat.
- **`user-correction-baseline`** — comparison baseline = hf2q on M5 Max 100+ tok/s on 26-27B Gemma/Qwen MoE (NOT Apple's 2024 Llama-3.1-8B M1 Max @ 33.67 tok/s).
- **`user-correction-recency-and-direction`** — 2026 sources are PRIMARY (re-classify R3's "post-cutoff" labels against today's calendar); concurrent ANE+GPU is the strategic interest.

### Memory Pins (project / feedback)

- `project_decode_parity_achieved` — hf2q 103.5–107.1 tok/s decode at HEAD `9ab4cca` on M5 Max.
- `project_mlx_native_is_the_strategic_destination` — non-trivial future GPU compute work belongs in `mlx-native`.
- `project_mlx_native_crate` — separate pure-Rust crate at `/opt/mlx-native` for Metal GPU compute.
- `feedback_perf_gate_thermal_methodology` — cold-SoC bench discipline for thermal-honest measurement.
- `feedback_ground_truth_is_what_we_can_measure_now` — peer reference values must be re-measured on the day; live data supersedes historical figures.
- `feedback_hf2q_sovereignty` — Pure Rust, `mlx-native` is the only sibling dep; no Python, no runtime link to candle/llama.cpp.
- `feedback_no_shortcuts` / `feedback_correct_outcomes` — never fall back to lesser options or take shortcuts; fix the blocker.

### Verified URL list (from researcher briefs; deduplicated)

Apple newsroom + Apple ML Research:
- `https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/`
- `https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/`
- `https://www.apple.com/macbook-pro/specs/`
- `https://machinelearning.apple.com/research/exploring-llms-mlx-m5`
- `https://machinelearning.apple.com/research/neural-engine-transformers`
- `https://machinelearning.apple.com/research/core-ml-on-device-llama`
- `https://machinelearning.apple.com/research/introducing-apple-foundation-models`
- `https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025`
- `https://machinelearning.apple.com/research/openelm`
- `https://machinelearning.apple.com/research/apple-intelligence-foundation-language-models`

Apple developer docs / WWDC / Tech Talks:
- `https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw`
- `https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine`
- `https://developer.apple.com/documentation/metal/machine-learning-passes`
- `https://developer.apple.com/machine-learning/core-ml/`
- `https://developer.apple.com/forums/forums/topics/machine-learning-and-ai/machine-learning-topic-core-ml`
- `https://developer.apple.com/videos/play/wwdc2024/10161/`
- `https://developer.apple.com/videos/play/wwdc2024/10159/`
- `https://developer.apple.com/videos/play/wwdc2024/10218/`
- `https://developer.apple.com/videos/play/wwdc2023/10049/`
- `https://developer.apple.com/videos/play/wwdc2025/262/`
- `https://developer.apple.com/videos/play/wwdc2025/286/`
- `https://developer.apple.com/videos/play/wwdc2025/360/`
- `https://developer.apple.com/videos/play/wwdc2025/301/`
- `https://developer.apple.com/videos/play/tech-talks/111432/`

coremltools:
- `https://apple.github.io/coremltools/docs-guides/source/stateful-models.html`
- `https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html`
- `https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html`
- `https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html`
- `https://apple.github.io/coremltools/docs-guides/source/opt-joint-compression.html`
- `https://github.com/apple/coremltools/releases`
- `https://github.com/apple/coremltools/blob/main/coremltools/optimize/torch/layerwise_compression/algorithms.py`
- `https://github.com/apple/coremltools/blob/main/coremltools/optimize/torch/quantization/post_training_quantization.py`

arXiv 2025–2026:
- `https://arxiv.org/abs/2603.06728` (Orion, Mar 2026)
- `https://arxiv.org/pdf/2603.06728`
- `https://arxiv.org/abs/2601.19139` (vllm-mlx, Jan 2026)
- `https://arxiv.org/abs/2604.16957` (Open-TQ-Metal, Apr 2026)
- `https://arxiv.org/abs/2510.13161` (Mirror Speculative Decoding, Oct 2025)
- `https://arxiv.org/abs/2508.08531` (Apple Silicon Profiling, Aug 2025)
- `https://arxiv.org/abs/2507.10860` and `https://arxiv.org/html/2507.10860v1` (WhisperKit, ICML 2025)

arXiv historical foundation:
- `https://arxiv.org/abs/2407.21075` (AFM)
- `https://arxiv.org/abs/2308.14352` (EdgeMoE)
- `https://arxiv.org/abs/2306.00978` (AWQ)
- `https://arxiv.org/abs/2206.01861` (ZeroQuant)
- `https://arxiv.org/abs/2407.05858` (NPU on-device LLM)
- `https://arxiv.org/abs/2404.14619` (OpenELM)
- `https://arxiv.org/abs/2401.10774` (Medusa)
- `https://arxiv.org/abs/2402.02057` (Lookahead Decoding)
- `https://arxiv.org/abs/2210.17323` (GPTQ)
- `https://arxiv.org/abs/2301.00774` (SparseGPT)
- `https://arxiv.org/abs/2208.11580` (Optimal Brain Compression)

Shipping projects:
- `https://github.com/Anemll/Anemll`
- `https://www.anemll.com/`
- `https://github.com/AtomGradient/hybrid-ane-mlx-bench`
- `https://github.com/john-rocky/CoreML-LLM`
- `https://github.com/john-rocky/CoreML-LLM/blob/main/docs/BENCHMARKING.md`
- `https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml`
- `https://github.com/mechramc/Orion`
- `https://github.com/FluidInference/FluidAudio`
- `https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md`
- `https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml`
- `https://github.com/FluidInference/mobius/tree/main/models/stt/parakeet-tdt-v3-0.6b/coreml`
- `https://github.com/apple/ml-ane-transformers`
- `https://github.com/smpanaro/coreml-llm-cli`
- `https://github.com/ggml-org/whisper.cpp/discussions/548`
- `https://github.com/ggml-org/whisper.cpp/blob/master/coreml/whisper-encoder.mm`
- `https://github.com/ggml-org/llama.cpp/discussions/4167`
- `https://github.com/mlc-ai/mlc-llm/issues/2230`
- `https://github.com/ggerganov/ggml/blob/master/src/ggml-common.h`
- `https://raw.githubusercontent.com/hollance/neural-engine/master/docs/ane-vs-gpu.md`
- `https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md`

Engineering blogs / community:
- `https://engineering.drawthings.ai/p/making-apple-neural-engine-work-in`
- `https://maderix.substack.com/p/inside-the-m4-apple-neural-engine`
- `https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615`
- `https://insiderllm.com/guides/apple-neural-engine-llm-inference/`
- `https://llmcheck.net/benchmarks`
- `https://llmcheck.net/blog/apple-silicon-m5-max-local-ai-guide/`
- `https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/`
- `https://en.wikipedia.org/wiki/Apple_M5`
- `https://en.wikipedia.org/wiki/MacOS_Tahoe`
- `https://rockyshikoku.medium.com/running-gemma4-on-apple-neural-engine-79fa0cb39dd2`

---

*End of dossier. ADR-016 graduates to Proposed-Accepted (narrow scope) against this evidence base.*
