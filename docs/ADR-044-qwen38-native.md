# ADR-044: Qwen3.8 native conversion and inference

- Status: Accepted for implementation; real-model acceptance pending
- Date: 2026-08-16
- Owners: hf2q conversion, quantization, inference, and serving

## Context

`Qwen/Qwen3.8-27B` is an Apache-2.0 dense multimodal release. The exact
source revision selected for hf2q onboarding is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Its configuration declares
`Qwen3_5ForConditionalGeneration`, with the text decoder under
`model.language_model.*` and the vision tower under `model.visual.*`.

The text decoder is a 64-layer hybrid:

- hidden size 5,120 and dense FFN size 17,408;
- 48 linear-attention layers and 16 full-attention layers;
- full attention every fourth layer, with 24 query heads, four KV heads,
  and head dimension 256;
- linear attention with 16 key heads, 48 value heads, dimension 128, and
  convolution width four;
- one MTP layer with shared embeddings;
- vocabulary 248,320 and native context 262,144.

The official index contains 1,199 tensors: 851 main text tensors, 15 MTP
tensors, and 333 vision tensors. The source BF16 payload is 55,562,855,904
bytes across 18 shards.

hf2q already owns a native dense `qwen35` inference graph with the same
hybrid layer contract, but the active converter previously supported only
the MoE variants. The stale registry entry referenced a deleted converter,
so the official Qwen3.8 checkpoint correctly failed closed as unsupported.

## Decision

1. Add a distinct dense `ArchName::Qwen35`. Detection accepts only the
   explicit `qwen3_5` / `qwen3_5_text` model types and
   `Qwen3_5ForCausalLM` / `Qwen3_5ForConditionalGeneration` classes.
2. Implement dense conversion in `src/convert/arch/qwen35_dense.rs`. It
   shares the family’s linear-attention V-head transformation contract,
   maps dense FFNs and all 15 MTP tensors, and fails on unknown text tensors.
3. Emit `general.architecture=qwen35`, 65 total blocks, one NextN block, and
   `nextn.use_dedicated_embeddings=false`. The native loader subtracts the
   NextN count to recover the 64 verifier layers.
4. Text conversion drops only the known `model.visual.*` namespace. Vision
   support is a separate projector artifact and is not claimed until its
   converter, loader, image-token plumbing, and real-image parity gate pass.
5. hf2q remains the implementation owner. No external converter,
   quantizer, or inference runtime becomes a product dependency. An external
   implementation may be used only as a developer-side reference oracle.
6. The first real artifact is produced from the exact official source
   revision with hf2q itself. Pre-quantized downloads do not satisfy this ADR.

## Acceptance gates

### Model-free

- Official config detects as dense `qwen35` and no neighboring architecture.
- The 851 main text tensors plus 15 MTP tensors map to 866 unique GGUF names.
- The official 333 vision tensors are excluded only from the text artifact.
- Metadata parses back into `Qwen35Config` with the exact dimensions above.
- A synthetic sharded safetensors fixture completes convert, tokenizer
  emission, GGUF reopen, tensor-bake, and quantization checks.

### Real model on Apple Silicon

- Download and verify the exact 18-shard source revision.
- Produce the requested quantized GGUF with `hf2q convert`.
- Load through the Rust + `mlx-native` dense Qwen graph.
- Compare deterministic text and logits against the source-model reference;
  enforce the documented quality bound for the selected quantization.
- Prove unary and SSE OpenAI chat, native tool calls and tool-result
  continuation, thinking modes, cancellation, and non-empty semantic TTFT.
- Prove a normal follow-up reuses the stable prompt prefix and is byte-equal
  to a matched cold continuation.
- Record exact artifact hashes, prompt/settings, cached-token counts,
  prefill/decode rates, and hardware.

### Vision

Vision remains unavailable for Qwen3.8 until a separately reviewable gate
proves the projector bytes and at least one real image request. Text support
must never silently advertise vision support before that gate is green.

## Consequences

Qwen3.8 can reuse a mature native execution family without approximate
architecture routing, while conversion and evidence remain explicit. The
large source download is deferred until metadata and synthetic conversion
are green, reducing iteration cost and making any later failure attributable
to real weights or runtime behavior rather than basic dispatch.
