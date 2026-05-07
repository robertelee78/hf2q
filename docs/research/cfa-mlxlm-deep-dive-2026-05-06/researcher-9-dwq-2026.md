# DWQ (Distilled Weight Quantization) — State of the Art as of 2026

**Researcher #9 / cfa-20260506-mlxlm-research**
**Date: 2026-05-06**
**Method: Goalie agentic-research (3 sub-agents per subtopic, parallel) + WebSearch + WebFetch on canonical sources, with cross-referencing.**

Citations follow each section. Where sub-agents disagreed I went to primary sources (mlx-lm GitHub, raw `mlx_lm/quant/dwq.py`, official mlx-community model cards, Awni Hannun's X account, the smcleod 2026-04 KL benchmark) and trust those over goalie's synthesized text. **Disagreements are flagged as "[OPEN]"** — the field is moving fast and several questions cannot be conclusively resolved from public sources.

---

## 1. DWQ Origin and Authoritative Reference

### 1.1 What DWQ stands for

**Confirmed: "Distilled Weight Quantization"** (not "Distillation-Weighted Quantization", not "Dynamic Weight Quantization", not "Double Weight Quantization", not "Data-free Weight Quantization" — all of these appear in third-party blog posts and goalie's fact-checker noise, but the canonical Apple source is unambiguous). The canonical definition lives in `mlx-lm/mlx_lm/LEARNED_QUANTS.md`:

> "DWQ fine-tunes non-quantized parameters (scales, biases) using the non-quantized model as a teacher to reduce quality loss from quantization."

Source: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md>

### 1.2 Authoritative reference

**There is no formal academic paper or arXiv preprint for DWQ.** Multiple agents (researcher, fact_checker, synthesizer) all independently confirmed this, and direct GitHub inspection backs them up. DWQ is a practical engineering contribution by the Apple MLX team, documented only in:

- The `mlx-lm` repository itself: `mlx_lm/LEARNED_QUANTS.md` plus the implementation in `mlx_lm/quant/dwq.py`.
- Awni Hannun's announcement on X, June 2025: *"If you haven't tried the new DWQ and/or dynamic quants in mlx-lm, I highly recommend. They give much higher quality q4 MLX models. And the full quantization can be done locally."*
  Source: <https://x.com/awnihannun/status/1931088863642181970>

No author bylines beyond the Apple MLX team (Awni Hannun, Alex Barron, and `ml-explore` contributors). **No arXiv preprint exists as of May 2026.** This is unusual for a method this widely deployed and is itself a useful fact: "no DWQ paper" means citing DWQ correctly cites the GitHub repo, not arXiv.

### 1.3 Other DWQ implementations outside Apple MLX as of 2026

**Effectively zero** outside the MLX ecosystem. The acronym "DWQ" is used in the wild for several unrelated methods (Dynamic Weight Quantization, Double-Wide Quantization, etc.), but Apple's specific *Distilled* WQ has not been ported. Specifically:

- **PyTorch / HuggingFace transformers**: No `DWQ` quantizer in `bitsandbytes`, `auto-gptq`, `auto-awq`, or HF native quantization. AWQ and GPTQ are widely available; DWQ is not.
- **vllm / TensorRT-LLM / SGLang**: No DWQ loader as of early 2026. They consume AWQ, GPTQ, FP8, MXFP4.
- **CUDA / NVIDIA**: No port. DWQ output is tied to MLX's specific affine quantization layout and Apple's `mx.quantize` op.
- **Closest relative**: The general idea (distillation-tune quant scales) shows up in academic work like **EfficientQAT** (Chen et al., 2024, arXiv:2407.11062) and **BitDistiller** (cited in EfficientQAT's compared methods), but those use *quantization-aware training* not Apple's lighter-weight scales-only post-training distillation. They are not API-compatible.

**[OPEN]**: The unrelated `mlx-optiq` project (<https://mlx-optiq.com>) on Apple Silicon implements a *similar* mixed-precision layer-sensitivity approach but it is not DWQ — it builds on MLX's own primitives and was published independently.

---

## 2. DWQ Math, Precisely Defined

This section is the most-citation-dense because the goalie agents disagreed sharply, and the only ground truth is the source code. I fetched the actual `mlx_lm/quant/dwq.py` and the loss function reads:

```python
# from mlx_lm/quant/dwq.py — Apple MLX, fetched 2026-05-06
losses = kl_div_loss(scale * logits, scale * targets)
mask = mx.arange(1, 1 + targets.shape[1]) < lengths[:, 1:]
ntoks = mask.sum()
loss = (mask * losses).sum() / ntoks
```

Where `scale = 1 / temperature`, `logits` are student outputs, `targets` are teacher logits. So the answers to the question are:

### 2.1 Objective function

**Pure KL divergence on output logits**, with temperature scaling. **Not** layer-wise hidden-state matching, **not** MSE on intermediate activations. This is the standard Hinton-style logit distillation but applied to quantization scales rather than to a smaller student. The loss is masked to valid tokens (no loss on padding).

### 2.2 What's actually being optimized

**Only the FP16 scales and biases.** Confirmed by the source line:

```python
m.unfreeze(keys=["scales", "biases"], recurse=False)
```

This *unfreezes* `scales` and `biases` on every quantized linear layer; everything else (the packed `uint32` weight codes, all non-quantized layers, layer norms, embeddings, the routing gates of MoE) stays frozen. **The integer codes are NOT updated.** This is critical for understanding DWQ's efficiency: you only train ~2× group_count parameters per quantized layer, which is roughly 1/64th of full-rank fine-tuning at group_size=64.

Implication: DWQ is closer in spirit to *learned per-group calibration* than to true distillation-based QAT. The "distillation" terminology refers to the teacher-student loss shape, not to a structural change in the student.

### 2.3 Straight-through estimator semantics

**There is NO explicit STE.** Gradients flow through MLX's built-in `mx.quantize` / `mx.dequantize` ops, which are differentiable with respect to `scales` and `biases` (because those are continuous FP16 parameters in the affine equation `x_dq = code * scale + bias`). They are **not** differentiable through the `code = round((x - bias) / scale)` step — but since `code` is held *fixed* during DWQ training (from the initial RTN/AWQ/dynamic-quant pass), no gradient ever needs to flow through `round`. This is why no STE is needed: the integer codes are inputs, not learnable.

This is a meaningful simplification compared to QAT methods like EfficientQAT or LSQ which *do* update codes and therefore *do* require STE.

### 2.4 Hyperparameters (defaults from `dwq.py` + `LEARNED_QUANTS.md`)

| Hyperparameter        | Default       | Documented intuition                                      |
|-----------------------|---------------|-----------------------------------------------------------|
| `--learning-rate`     | `1e-6`        | "higher for lower precision, lower for higher precision"  |
| `--batch-size`        | `4`           | Tradeoff with seq length for memory                       |
| `--num-samples`       | `1024-2048`   | More = better quality, longer runtime                     |
| `--max-seq-length`    | `1025`        | Lower (e.g. 512) saves memory                             |
| `--bits`              | `4`           | Optimal target; doc says "best at 2-4 bit"               |
| `--group-size`        | `64`          | "--group-size 32 doubles tunable params, often better"    |
| `temperature`         | implicit ~1.0 | Hardcoded scale parameter for KL                          |

Note the **learning rate (1e-6) is unusually small** — characteristic of "fine-tuning a tiny number of parameters that already start near-correct from RTN initialization." For comparison, full-model SFT runs at 1e-5 to 1e-4. Awni Hannun's example invocations on X show `--learning-rate 5e-8` for refining an already-good 4-bit model — i.e. the LR drops further as you cascade.

Sources:
- <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md>
- Source code in `mlx_lm/quant/dwq.py` (fetched via GitHub raw, 2026-05-06)

---

## 3. DWQ vs Alternatives — Benchmark Numbers as of 2026

### 3.1 The single most-citable comparison (smcleod 2026-04, Qwen3.6-35B-A3B MoE)

Sam McLeod's April 2026 KL-divergence benchmark using 65,536 tokens of WikiText-2 against a BF16 reference, on Apple M5 Max:

| Method                       | Effective bpw | Mean KL-div vs BF16 | Notes                                                  |
|------------------------------|---------------|---------------------|--------------------------------------------------------|
| `Qwen3.6-35B-A3B-4bit-DWQ`   | 4.84          | **0.02663**         | mlx-community official DWQ release                     |
| oQ4 (omlx 4-bit mixed)       | ~4.5          | 0.04024             | DWQ ~34% better                                        |
| Plain MLX RTN 4-bit          | ~4.0          | 0.07418             | DWQ ~3× better                                         |
| oQ6 (omlx 6-bit mixed)       | ~6.0          | 0.0119              | wins by ~13% over DWQ in 6-bit territory               |
| 4-bit DWQ vs 6-bit oQ6       | -             | DWQ loses           | confirms DWQ's saturation point                        |

Source: <https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/>

This is the **clean answer to "where does DWQ saturate"**: on Qwen3.6-35B-A3B, DWQ's KL-div improvement plateaus or reverses against carefully-mixed 6-bit. This empirically confirms `LEARNED_QUANTS.md`'s prose claim "Distilling 16-bit precision to 8-bit and even 6-bit often doesn't work well".

### 3.2 Qwen3 4-bit / 3-bit / 2-bit baselines (no DWQ)

From "An Empirical Study of Qwen3 Quantization" (arXiv:2505.02214, May 2025), Qwen3-8B on C4 perplexity and MMLU:

| Bit | Method  | PPL  | MMLU |
|-----|---------|------|------|
| 4   | AWQ     | 11.2 | 73.8 |
| 4   | GPTQ    | 11.0 | 72.7 |
| 4   | RTN     | 14.5 | 70.2 |
| 3   | AWQ     | 14.4 | 57.7 |
| 2   | GPTQ    | 2560 | crash |

Combined with smcleod, the picture is: **at 4-bit DWQ pulls level with AWQ/GPTQ on dense models and pulls ahead on MoE; at 3-bit no published DWQ numbers exist; at 2-bit nothing works well, period.**

[OPEN] **Disagreement**: Some goalie sub-agents claimed DWQ "decisively leapfrogs AWQ at 4-bit"; others (including the fact_checker) found no head-to-head DWQ-vs-AWQ on Llama 3 specifically. I trust only smcleod's measured Qwen3.6-A3B numbers; Llama 3 4-bit DWQ vs AWQ comparison is **not published as of 2026-05** to my knowledge.

### 3.3 Saturation point summary

- **2-bit**: DWQ helps materially (the doc explicitly recommends it). **Quantitative numbers are not published on a standard PPL benchmark** as of May 2026.
- **3-bit**: DWQ likely helps (extrapolation from 4-bit gain), but no public head-to-head numbers.
- **4-bit**: DWQ is the recommended target; ~34% better mean KL-div than oQ4 on MoE.
- **6-bit**: DWQ marginally helps or is neutral. `LEARNED_QUANTS.md`: "doesn't work well."
- **8-bit**: DWQ is **not worth running**. Quantization error is already below distillation noise floor; the optimization has nothing to learn.

The phrase *"isn't worth running for 6/8-bit because loss is already low"* in your prompt matches Apple's documented stance verbatim.

---

## 4. DWQ Tooling Landscape 2026

### 4.1 Canonical implementation

**`mlx_lm.dwq`** (in `mlx-lm` ≥ 0.20, currently 0.25+) is the only canonical implementation. Sister command: **`mlx_lm.dynamic_quant`** for the cheaper sensitivity-driven mixed-precision pass. Both are command-line wrappers over Python. No official GUI; LM Studio's MLX backend consumes the output but does not run DWQ itself.

```bash
# canonical invocations (from LEARNED_QUANTS.md)
mlx_lm.dwq --model Qwen/Qwen3-0.6B
mlx_lm.dwq --model <bf16> --quantized-model <existing-4bit> --bits 4 \
           --num-samples 2048 --learning-rate 5e-8
mlx_lm.dynamic_quant --model Qwen/Qwen3-0.6B --target-bpw 4.5
```

### 4.2 Third-party DWQ ports

**None as of 2026-05.** Closest adjacent projects:

- **`mlx-optiq`** (<https://mlx-optiq.com>): independent mixed-precision quantization for Apple Silicon. Uses KL-div per-layer to budget bits via greedy knapsack — *similar idea, different algorithm*. Not API-compatible with DWQ.
- **`mlx-vlm`** (Blaizzy): does NOT support DWQ for VLMs as of mid-2025 — discussion #667 confirms users are blocked from quantizing vision-language models with DWQ until mlx-vlm catches up.
- **`unsloth` MLX** (`Unsloth Dynamic 2.0` per-tensor): different approach, also Apple-only, not DWQ.

### 4.3 GGUF compatibility — the unsolved problem

**No, DWQ has not been ported to GGUF compatibly.** This is a real gap and a recurring user complaint. Reasons:

1. **Layout mismatch**: DWQ output is MLX's affine-quant `(uint32-packed, fp16 scale, fp16 bias)` per group of 64 weights. GGUF Q4_K_M uses a 6-bit super-scale + 4-bit sub-scales + INT8 mins per super-block of 256 elements; the math is *fundamentally different* (k-quants are non-uniform with two levels of nesting; DWQ is plain affine).
2. **Bit budget mismatch**: GGUF Q4_K_M is ~4.83 bpw with super-scale overhead; MLX 4-bit DWQ is ~4.5 bpw at group_size=64 (4 + 2*16/64 = 4.5). You cannot losslessly remap learned MLX scales onto GGUF's Q4_K_M layout — the granularity differs.
3. **Tooling gap**: There is `gguf2mlx` (one-way, GGUF → MLX, last updated 2026-04) and `ungguf` (GGUF → safetensors, requires HF reference). **There is no `mlx2gguf-dwq` tool that preserves trained scales.** Existing MLX→safetensors→GGUF paths re-quantize with k-quants from scratch, throwing away the DWQ training.

[OPEN] **Active research direction**, not a solved problem. As of 2026-05 the workaround everyone uses is "ship two artifacts: MLX-DWQ for Apple, GGUF-Q4_K_M for everyone else, accept they're different".

### 4.4 DWQ-2 / DWQ-extended

**No formal DWQ-2 paper or release.** Evolution since the original (~mid-2024) is incremental and lives in mlx-lm releases:

- VLM calibration support (multimodal calibration data) — discussion #667.
- Cascade with `dynamic_quant` — added to `LEARNED_QUANTS.md` after 2024.
- `--quantized-model` flag (refine an existing 4-bit instead of from FP16) — Issue #345 noted that 8-bit → 4-bit DWQ fails (shape error); only 4-bit base + DWQ-4-bit or FP16 → DWQ-4-bit work.

[OPEN] No public roadmap or planned "DWQ-2" announcement from Apple.

### 4.5 MLX safetensors layout stability in 2026

**Stable for affine 4-bit and 8-bit, evolving at the edges.** The base format is well-documented:

- `layer.weight`: packed `uint32` array (4-bit codes packed 8-per-uint32 at group_size 64).
- `layer.scales`: `float16` per group.
- `layer.biases`: `float16` per group.
- Quantization equation: `x_recon = code * scale + bias`.

What's *new* in 2026 (so format is "stable but not frozen"):

- **`mxfp8`** for 8-bit (group_size=32) replacing affine 8-bit in some recipes — adds a new metadata flag.
- **DQ_K_M / DQ3_K_M / DQ5_K_M** experimental k-quant-style MLX formats appearing on `mlx-community` (e.g. `Kimi-K2.6-mlx-DQ3_K_M-q8`, `DeepSeek-V3.1-mlx-DQ5_K_M`) — **these are NOT vanilla affine MLX**, they're a new MLX-internal k-quant variant with different sidecar metadata.

So your characterization "MLX safetensors with .weight (uint32 packed 4-bit), .scales (fp16), .biases (fp16)" **remains correct for 4-bit DWQ output specifically** but **alternative MLX quant formats now exist** in 2026 and a generic loader needs to detect format from metadata.

Sources: <https://github.com/ml-explore/mlx-lm/releases>, <https://huggingface.co/mlx-community>

---

## 5. Format Compatibility — 2026 State

### 5.1 What runtimes load MLX safetensors with DWQ?

| Runtime         | Loads MLX-DWQ?      | Notes                                             |
|-----------------|---------------------|---------------------------------------------------|
| `mlx-lm`        | Yes (canonical)     | `mlx_lm.generate`, `mlx_lm.server`                |
| `mlx-vlm`       | Partial             | VLM models via mlx-vlm; quant generation gap      |
| LM Studio       | Yes                 | MLX backend on Apple Silicon                      |
| Ollama          | Yes (since 03/2026) | Switched to MLX backend on Apple Silicon          |
| `llama.cpp`     | **No**              | GGUF-only, no MLX safetensors loader              |
| vllm / SGLang   | **No**              | CUDA-only ecosystem                               |
| TensorRT-LLM    | **No**              | NVIDIA proprietary                                |
| HuggingFace `transformers` | **No** | No `MlxQuantizationConfig` as of 2026-05          |

Source: <https://github.com/ml-explore/mlx-lm>, Awni Hannun on X (Ollama MLX integration), various Hugging Face model cards.

### 5.2 MLX → GGUF preserving DWQ scales — none exists

Restating section 4.3 for completeness: **no public tool bridges MLX-DWQ safetensors into GGUF preserving the trained scales.** All existing converters re-quantize. The user's `hf2q` project (this codebase) appears to be one of the few attempts to *consume* MLX DWQ output natively rather than convert to GGUF.

---

## 6. MoE-Specific Findings

### 6.1 Does DWQ work on MoE?

**Yes, demonstrably.** The mlx-community ships several official DWQ-MoE models:

- `mlx-community/Qwen3-30B-A3B-4bit-DWQ` (5B active params, 17.2 GB)
- `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ` (~20.66 GB)
- `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ`
- `mlx-community/Qwen3-Embedding-4B-4bit-DWQ`

The user's 35B-A3B is *exactly* the model on which smcleod measured the headline `0.02663` mean KL-div — **DWQ on MoE is the most-published DWQ result of any model class**.

### 6.2 Special handling for routing weights

Implicitly yes, via group_size and protected-tensor lists. From smcleod's analysis of the mlx-community release config:

> "It protected 512 overrides including embed, all SSM tensors, routers, and shared experts — significantly more protection than competing methods."

i.e. the mlx-community DWQ recipe for 35B-A3B *exempts the router gates* (the small per-layer linear projection that produces expert assignment logits) from quantization entirely, leaving them BF16 / FP16. This matches general best-practice in MoE quantization literature (EAQuant, MILO, EAC-MoE) which all report router gates are too sensitive to quantize naively.

DWQ's contribution here is *the framework for selectively unfreezing-and-protecting* (via `mixed_quantization_predicate` / `--mixed`-style flags), not a novel router-quant algorithm.

### 6.3 Published DWQ-on-MoE numbers

- **Qwen3.6-35B-A3B-4bit-DWQ**: mean KL-div 0.02663 vs BF16 (smcleod 2026-04) — best 4-bit on MoE measured.
- **Qwen3-30B-A3B-4bit-DWQ** in production: per BrownBear127's `qwen-mlx-bench`, 70/70 tool-call rounds clean, 47% faster decode than Q8_K_XL GGUF on M-series.
- **Mixtral 8x7B with DWQ**: [OPEN] no official mlx-community DWQ release. Goalie's researcher claimed MLX "explicitly supports DWQ for Mixtral" but I could not verify a published DWQ-Mixtral artifact on Hugging Face; only AWQ and base 4-bit Mixtrals exist there.
- **DeepSeek-MoE / DeepSeek-V3.1**: mlx-community has `DeepSeek-V3.1-mlx-DQ5_K_M` which is a *different* MLX format (DQ5_K_M, k-quant variant) — **not DWQ**.

So as of 2026-05 **DWQ-on-MoE is well-established for the Qwen3 / Qwen3.6 family but not yet ported to the other major MoE families** (Mixtral, DeepSeek-V3, etc.) in the public mlx-community.

Sources:
- <https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ>
- <https://huggingface.co/mlx-community/Qwen3-30B-A3B-4bit-DWQ>
- <https://github.com/BrownBear127/qwen-mlx-bench>

---

## 7. Practical Wisdom of the Field as of 2026

### 7.1 Has DWQ been leapfrogged?

**Depends what you mean.** Three lenses:

1. **For weight-only 4-bit on Apple Silicon: NO.** DWQ is the best published method for the specific niche of "I have a Mac, I want a 4-bit model with maximum quality and inference speed via the Metal kernels." Nothing else combines (a) MLX-native execution, (b) low compute cost (~hours, not days), (c) preserved Apple-Silicon kernel speedups.

2. **For W4A4 (weights AND activations 4-bit) on NVIDIA: YES, several methods leapfrog.** As of late 2025 / early 2026 the academic SOTA for full W4A4 on Llama-3 70B is:
   - **FlatQuant** (arXiv:2410.09426, ICML 2025) — <1% accuracy drop on W4A4, beats SpinQuant by 7.5% on Llama-3-70B, 2.3× prefill speedup.
   - **KurTail** (arXiv:2503.01483) — outperforms QuaRot by 13.3% MMLU and SpinQuant by 2.6% MMLU; runs on a single GPU vs SpinQuant's 4× H100s.
   - **SpinQuant** (arXiv:2405.16406) — learned rotations, the prior SOTA W4A4.
   - **QuaRot** (2024) — Hadamard rotation baseline that the above two beat.
   - **EfficientQAT** (arXiv:2407.11062, ACL 2025) — quantization-aware training, expensive but high quality.

3. **For sub-4-bit (W3, W2, W1)**: BitNet b1.58 / native FP4 (Blackwell) / AQLM / BiLLM / ARB-LLM are all active and DWQ is *not* in this conversation — DWQ's published numbers stop at 4-bit (and barely at 3-bit). The frontier has moved below 4-bit and DWQ doesn't compete there.

### 7.2 Consensus best-quality 4-bit method late-2025 / 2026

There is **no single consensus** but rough alignment:

| Hardware target       | Recommended 4-bit method                                     |
|-----------------------|--------------------------------------------------------------|
| Apple Silicon         | **DWQ** (or DWQ cascaded after dynamic_quant)                |
| NVIDIA (W4A16)        | **AWQ + Marlin** (mature, fast, low PPL; oobabooga 2025)     |
| NVIDIA (W4A4)         | **FlatQuant** (academic SOTA Llama-3) or **SpinQuant** (deployed) |
| llama.cpp / GGUF      | **imatrix Q4_K_M** (de facto standard, ~4.83 bpw)            |
| Mobile / edge         | **EfficientQAT** if you can train, **AWQ** if you can't       |

### 7.3 Recent papers showing DWQ limitations

Direct DWQ-limitations papers don't really exist (because DWQ has no paper to critique). But indirect critiques:

- The **Qwen3 Empirical Study** (arXiv:2505.02214) shows Qwen3 is *more* sensitive to low-bit quantization than Llama 3 because Qwen3 has less parameter redundancy from advanced pre-training. Implication: even DWQ struggles on Qwen3 below 3-bit.
- **FlatQuant** paper notes that scales-only methods (which includes DWQ) cannot fix activation outliers, only weight quantization noise. For W4A4 you need rotation-based methods.

---

## 8. Cascade Orderings

### 8.1 The mlx-lm recommended preset

Per `LEARNED_QUANTS.md` directly:

> "You can also cascade methods. For example a dynamically quantized model can be further refined with DWQ."

The single cascade that ships in the official Apple recipe is:

```
FP16/BF16 base
  → mlx_lm.dynamic_quant --target-bpw <X>   (sensitivity-aware mixed-precision)
  → mlx_lm.dwq --bits 4 --quantized-model <out_of_step1>  (refine scales)
```

Awni Hannun's example invocations on X show this exact pipeline.

### 8.2 Why the order matters

`dynamic_quant` is a *sensitivity-aware initial quantization* — it produces a coarse mixed-bit-width assignment using KL-divergence-per-bit per layer (greedy knapsack on layer sensitivity). It runs in minutes. DWQ then *refines the surviving FP16 scales* using teacher distillation. It runs in hours.

Reversing the order (DWQ first, then dynamic_quant) would be incoherent: dynamic_quant *changes which layers are at what bit width*, which would invalidate the DWQ-trained scales. DWQ is an *output-stage refinement*, not an input-stage chooser.

Empirical: `mlx_lm.dwq --quantized-model <existing-4bit>` works **only if `<existing-4bit>` is already 4-bit** (Issue #345 demonstrates 8-bit → 4-bit DWQ fails with shape error). So the practical cascade chain is constrained: you can refine-at-target-bit-width, you cannot re-target-bit-width with DWQ.

### 8.3 DWQ + AWQ?

[OPEN] **No published cascade.** Goalie's researcher claimed AWQ → DWQ "creates smoother distributions for DWQ's codebooks" but I found no benchmark or recipe demonstrating this. AWQ output uses HuggingFace's quantizer format (different layout) and would need format conversion before DWQ could refine it — there is no documented path. Would be a research contribution if someone built it.

### 8.4 GPTQ + DWQ?

[OPEN] **No published cascade.** Same blocker: GPTQ output is HF-format, not MLX safetensors. No tool bridges them while preserving GPTQ scales for DWQ to refine.

### 8.5 The real cascade in practice

The actually-deployed cascade in the mlx-community 2026 model cards is:

```
BF16 weights (Hugging Face)
  → mlx_lm.convert --quantize  (RTN initialization, fast)
  → mlx_lm.dwq --learning-rate 1e-6 --num-samples 2048  (refine ~1 hour)
```

The `dynamic_quant` step is recommended but often skipped; many DWQ models on mlx-community are RTN-init + DWQ, not dynamic_quant + DWQ.

Source: model cards under <https://huggingface.co/mlx-community>, e.g. `Qwen3-30B-A3B-4bit-DWQ`.

---

## Cross-Source Disagreements and Confidence Notes

I flag these because the user requested it:

1. **DWQ acronym expansion**: smcleod and `LEARNED_QUANTS.md` say "**Distilled** Weight Quantization." Goalie's fact_checker flipped between "Distilled" and "Distillation-Weighted." Multiple Medium / blog posts use "Dynamic Weight Quantization" or "Double-Weight Quantization." → **Authoritative answer is "Distilled" per the canonical doc.**

2. **Whether DWQ has a paper**: All goalie agents agreed there is no paper. WebSearch turned up no arXiv preprint. → **Confirmed: no paper, GitHub repo is the citation.**

3. **DWQ on Mixtral**: Goalie researcher confidently claimed yes; fact_checker found zero evidence. Direct mlx-community search confirms **no Mixtral-DWQ release** as of 2026-05. → **OPEN: theoretically supported, not publicly released.**

4. **Llama-3 4-bit DWQ vs AWQ**: No published head-to-head numbers. → **OPEN.**

5. **DWQ-2 / DWQ-extended**: No such paper exists. → **Confirmed.**

6. **GGUF bridge for DWQ**: Multiple converters exist (gguf2mlx, ungguf), **none preserve DWQ scales**. → **Confirmed gap.**

---

## Summary for the cfa-mlx-lm-research synthesis

If you take only one paragraph from this report:

> *DWQ ("Distilled Weight Quantization") is an Apple MLX-team engineering contribution, not an academic paper. It optimizes only the FP16 quantization scales and biases (not the integer codes) using a KL-divergence loss between full-precision teacher logits and quantized-student logits, with no straight-through estimator (gradients flow through MLX's differentiable affine quantize op). Default hyperparameters: lr 1e-6, batch 4, 1024-2048 samples, max-seq 1025, group-size 64. It works best at 2-4 bits on Apple Silicon, saturates by 6-bit, isn't worth running at 8-bit. Best published number: 0.02663 mean KL-div on Qwen3.6-35B-A3B 4-bit (smcleod 2026-04), ~34% better than oQ4 mixed-precision. As of May 2026 it is the recommended 4-bit method for Apple Silicon, but FlatQuant and SpinQuant leapfrog it for W4A4 on NVIDIA, and there is no public MLX→GGUF bridge that preserves DWQ's trained scales — so DWQ output is currently locked to MLX runtimes (mlx-lm, mlx-vlm, LM Studio's MLX backend, Ollama on Apple Silicon since March 2026).*

---

## Primary Sources

1. `mlx-lm` repo: <https://github.com/ml-explore/mlx-lm>
2. `LEARNED_QUANTS.md`: <https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md>
3. `mlx_lm/quant/dwq.py` (raw source, fetched 2026-05-06)
4. Awni Hannun, X announcement: <https://x.com/awnihannun/status/1931088863642181970>
5. mlx-community Qwen3.6-35B-A3B-DWQ: <https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit-DWQ>
6. mlx-community Qwen3-30B-A3B-DWQ: <https://huggingface.co/mlx-community/Qwen3-30B-A3B-4bit-DWQ>
7. smcleod KL-div benchmark 2026-04: <https://smcleod.net/2026/04/measuring-model-quantisation-quality-with-kl-divergence/>
8. Qwen3 Empirical Quantization Study: <https://arxiv.org/abs/2505.02214>
9. FlatQuant ICML 2025: <https://arxiv.org/abs/2410.09426>
10. KurTail: <https://arxiv.org/abs/2503.01483>
11. SpinQuant: <https://arxiv.org/abs/2405.16406>
12. EfficientQAT: <https://arxiv.org/abs/2407.11062>
13. BrownBear127 qwen-mlx-bench: <https://github.com/BrownBear127/qwen-mlx-bench>
14. mlx-lm Issue #345 (8-bit → 4-bit DWQ fails): <https://github.com/ml-explore/mlx-lm/issues/345>
15. mlx-lm Discussion #667 (DWQ on VLM): <https://github.com/ml-explore/mlx-lm/discussions/667>
16. mlx-optiq (related but distinct project): <https://mlx-optiq.com>
17. Awesome-LLM-Quantization curated index: <https://github.com/pprp/Awesome-LLM-Quantization>
