# ADR-004: GGUF Compatibility with llama.cpp

**Status:** Implemented  
**Date:** 2026-04-09  
**Updated:** 2026-08-21 — restored the originally accepted automatic
multimodal-pair contract and made paired publication crash-recoverable with
runtime generation/digest enforcement.
**Decision Makers:** Robert, Claude

## Context

hf2q converts HuggingFace models to quantized GGUF format. The GGUF files must load and produce correct output in llama.cpp, ollama, and any tool using the ggml ecosystem. Prior to this work, hf2q's GGUF output was rejected by llama.cpp due to multiple format issues.

## Decisions

### 1. Q4_0/Q8_0 Block Repacking

**Problem:** hf2q's quantizer produces packed nibbles + separate scales. GGUF Q4_0 format requires interleaved 18-byte blocks (2-byte f16 scale + 16 packed nibbles per 32 elements).

**Decision:** Repack hf2q's internal format into proper ggml block format during GGUF writing. Streaming repack (one tensor at a time) to avoid OOM on 26B+ models.

**Alternatives considered:** Write raw data and let llama-quantize re-quantize. Rejected because we want pre-quantized GGUFs.

### 2. Dimension Ordering

**Problem:** GGUF stores dimensions in ggml order (innermost first), opposite of PyTorch/safetensors (outermost first).

**Decision:** Reverse shape dimensions with `.iter().rev()` in `write_tensor_info()`. Data bytes stay in row-major order (unchanged from safetensors).

### 3. Architecture-Aware Tensor Name Mapping

**Problem:** Same HF tensor name maps to different GGUF names depending on architecture (e.g., `post_attention_layernorm` → `ffn_norm` for LLaMA but `post_attention_norm` for Gemma4).

**Decision:** Per-arch layer_map tables selected by `model_type`. Shared base entries + architecture-specific overrides.

### 4. Text/Vision Split (mmproj)

**Problem:** llama.cpp expects multimodal models as two separate GGUF files.

**Decision:** Automatically produce both `model.gguf` (text) and
`model-mmproj.gguf` (vision) when `vision_config` and supported vision tensors
are present. One convert command, two files. `--text-only` is the explicit
opt-out; the existing `--mmproj` spelling remains projector-only repair mode.

The projector is staged first and its exact SHA-256 is written into the text
GGUF as `hf2q.mmproj_sha256`. Both GGUFs carry one UUID
`hf2q.pair_generation` and schema marker. Publication takes an exclusive
sibling lock, persists a same-directory recovery journal, moves every previous
member to a private same-filesystem backup, then promotes the projector,
receipts, and bound text GGUF in that order. The text rename is the sole commit
marker. Before it, an error rolls the complete old pair back; after it,
recovery cleans forward to the complete new pair. Regular files receive
`F_FULLFSYNC` on macOS; directories receive `fsync` after each rename, with
the destination directory synchronized before the source directory for every
cross-directory move.
Terminal cleanup removes and synchronizes the journal before deleting private
staging/backups, so interruption can leave only an inert private orphan—not a
recovery journal whose required state has already disappeared.

Serving holds the shared side of the same lock from before opening the text
GGUF through projector validation/load. On read-only mounts or cross-user
artifact stores where the private sibling lock is unavailable, readers lock
the current text GGUF inode instead; writers always take that inode
exclusively as well as the stable sibling lock before mutation. The staged new
text inode is exclusively locked before its commit-marker rename as well.
Generation-marked members must share the schema, UUID, and directory, and the
projector bytes must match the digest embedded in the text GGUF. The digest
check also applies to older local pairs that predate remote provenance
receipts; provenance classification cannot silently discard an embedded
projector binding. Both remote-source receipts bind the same resolved
revision, source bundle, and converter; projector receipts use `f16-mmproj`
regardless of the text quantization.

`vision_config` without matching source tensors, vision tensors without
`vision_config`, a missing Qwen processor config, an unsupported projector
family, or colliding output paths fails before either destination is written.
A multimodal dry run names both planned outputs but creates neither.

### 5. Tokenizer Embedding

**Problem:** llama.cpp requires tokenizer data (vocab, merges, special tokens) embedded in the GGUF.

**Decision:** Parse `tokenizer.json` and `tokenizer_config.json` from the model directory. Write all required keys including `tokenizer.ggml.model`, `tokens`, `merges`, `scores`, `token_type`, special token IDs, `pre`, and `chat_template`.

### 6. V=K Duplicate Tensors

**Problem:** Gemma4 full-attention layers tie V to K — no separate `v_proj.weight` in safetensors. llama.cpp expects `attn_v.weight` to exist.

**Decision:** Duplicate K tensor data as V for full-attention layers (5, 11, 17, 23, 29).

### 7. K-Quant Block Size Fallback

**Problem:** K-quant types (Q6_K, Q5_K, etc.) require `ne[0] % 256 == 0`. MoE expert tensors with `intermediate_size=704` don't satisfy this.

**Decision:** Match llama.cpp's `tensor_type_fallback()`: Q6_K→Q8_0, Q5_K→Q5_1, Q4_K→Q5_0, Q3_K/Q2_K→Q4_0. This is the SOTA behavior used by all major quantizers.

### 8. Layer-Streaming Forward Pass

**Problem:** Loading all 30 Gemma4 layers into GPU memory for activation-based DWQ calibration exceeds 128GB (48GB TensorMap + 48GB Candle tensors).

**Decision:** Load one transformer layer at a time from TensorMap. Peak memory: ~54GB instead of ~96GB.

### 9. 1D Tensors as F32

**Problem:** llama.cpp Metal kernels assert F32 for element-wise binary operations.

**Decision:** Convert all 1D preserved tensors (norms, scales, scalars) from F16 to F32 during GGUF writing.

### 10. Synthetic rope_freqs Tensor

**Problem:** Gemma4's partial RoPE on full-attention layers requires frequency scaling factors not present in safetensors.

**Decision:** Generate `rope_freqs.weight` as `[1.0]*n_rot + [1e30]*n_unrot` (matching llama.cpp's converter).

## Results

- GGUF loads in llama.cpp with correct, coherent text output
- 100+ tok/s decode on Apple M4 Metal
- 2.9x compression (48GB → 16.8GB)
- Mixed-precision: Q4_0 base + Q6_K/Q8_0 for sensitive layers

### Automatic-pair acceptance evidence (2026-08-21)

Commit `54fd9a089c2d9ebf2ec3ac20b8d24fdc1236c318` converted the exact
`jenerallee78/Qwen3.8-27B-Abliterated-SFT` revision
`08c2f075b43bc06456382db6b918a3dcabdcf4dd` with one ordinary `hf2q convert`
command and no projector flag. The command published:

- an 866-tensor, 16,810,714,944-byte Q4_K_M text GGUF with SHA-256
  `1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a`;
- a 496-tensor, 927,606,848-byte F16 projector with SHA-256
  `463b264713f8e081f0fae753c80d8089308e01b1e2ac0948dd9966d0711d8f1b`.

Both schema-v3 receipts bind the same 21-file source set, source-bundle
SHA-256 `8531e68e43a4a28ed6c0b9b41ac33dc9484e0fbb5eae96198a8c45ba9caf18d0`,
exact source revision, and converter commit. Runtime startup reopened the text
header, observed the projector digest in `hf2q.mmproj_sha256`, loaded the
projector as `qwen3vl_siglip` / `qwen3vl_merger`, and rejected no binding.

`scripts/test_qwen38_first_image_cache.sh` then passed on an Apple M5 Max. Its
cold text turn used 86,077 prompt tokens with zero cached tokens. The first
image follow-up reused 86,072 of 86,172 prompt tokens, ran the GPU ViT forward,
answered `The dominant color in this image is red.`, stopped after 73
completion tokens under a 64-token thinking budget, and left `/readyz` at HTTP
200. This is the exact real-model proof that automatic pair production and the
existing runtime binding/cache contract compose correctly.
