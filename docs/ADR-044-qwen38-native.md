# ADR-044: Qwen3.8 native conversion and inference

- Status: Accepted for native text conversion and serving; vision candidate
  is under exact-artifact acceptance
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
   uses a separate projector artifact produced by hf2q. The server may
   advertise it only after exact text/projector binding, production-graph
   warmup, image preprocessing, image-token plumbing, and cache isolation pass
   fail closed.
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

- Conversion requires and records the official processor configuration and
  emits an explicit all-false 27-layer DeepStack mask for the dense Qwen3.8
  projector.
- Startup and every image-bearing request validate the exact text/projector
  architecture and hidden-width contract before image I/O or GPU execution.
- The production graph must pass a real-projector warmup and reject non-finite
  or wrong-width output.
- Official bicubic preprocessing, the 200:1 aspect-ratio ceiling, the exact
  65,536 through 16,777,216 pixel range, multi-image order, and decoded-input
  bounds are part of the accepted wire contract.
- Projected embeddings use an immutable byte-budgeted cache with exact image
  identity, single-flight execution, and request cancellation before language
  admission. A changed image must not reuse image or language-model state.
- Unary and SSE image requests, an image-driven tool call, its tool-result
  continuation, exact prompt-prefix reuse, concurrent same-image requests, a
  disconnected client, and the official maximum image size must pass on the
  exact artifact.

## Consequences

Qwen3.8 reuses the native dense Qwen execution family without approximate
architecture routing, while conversion and evidence remain explicit.
Speculative MTP decode remains outside the accepted surface; the canonical
launcher selects ordinary autoregressive decode until its separate
transactional state and parity gates pass. Vision is a separately measured
candidate surface and does not inherit text-only performance authority.

## Acceptance evidence

The text acceptance gate ran on an Apple M5 Max on AC power. The source was
the exact revision named above; its configuration SHA-256 was
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` and
its tensor-index SHA-256 was
`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`.

Native `q4_k_m` conversion produced 866 text tensors in a
16,810,714,624-byte GGUF with SHA-256
`bddc9ada92212253cceb77781cc3267cb63da10f6e000c32e775abdee9cf69ea`.
An independently converted artifact had the same tensor count, dimensions,
types, and deterministic smoke output. Quantized tensor payloads were not
claimed byte-identical: 515 of 866 tensor payloads matched exactly and 351
differed. Both artifacts generated the same requested Rust function through
both runtimes used by the acceptance comparison.

The native quantized artifact was also compared with the 54,657,734,016-byte
BF16 source conversion in the independent runtime. Three deterministic prompts
covering Rust code, integer arithmetic, and a concurrency explanation produced
exactly identical complete text and the same first token ID in all three
cases. This is a focused functional/quantization discriminator, not a broad
language-quality benchmark.

On the native hf2q artifact, deterministic single-shot generation measured a
32-token prefill in 11.94 seconds and 32 decoded tokens in 2.37 seconds
(13.5 tokens/second). The independent runtime measured 27.8 tokens/second on
the same artifact, so decode performance remains a follow-up rather than an
inflated parity claim. Initial GPU materialization exceeded the generic
30-second request deadline; the accepted implementation gives only Qwen
startup warmup a finite 240-second supervisor deadline and keeps ordinary
request deadlines unchanged.

The exact native server artifact passed `/readyz`, unary and SSE text,
required-tool unary and SSE calls, schema-correct arguments, tool-result
continuation with 320 cached tokens, automatic thinking without private client
flags, cancellation with checkpoint recovery, and two simultaneous requests.
An ordinary three-message follow-up returned the remembered value with 27 of
54 prompt tokens reused. The first clean server load took 24.65 seconds and
its measured startup warmup took 33.43 seconds. The gate also found and fixed
an invalid cache invariant: verifier KV cursors must agree with one another,
but the optional MTP cursor is independent until speculative decoding runs.

These measurements establish functional native text support, not completion
of the vision surface, speculative MTP acceptance, or performance parity.

### Vision candidate evidence (2026-08-16)

The exact source revision above produced a 927,606,848-byte F16 projector with
SHA-256
`6fa039b75244c0a28a013da30b92b1d221c61029acc19f9efa882b75a495b0d0`.
The paired Q4_K_M text candidate was 16,810,714,752 bytes with SHA-256
`0fa8acc661d0edc60276c43705619fd848682dbf768ced9fe46cd8a572b8043d`.

On an Apple M5 Max, the production server correctly described an 8.65 MP
screenshot, a separately resized image, and a two-image request. A required
image-driven `record_observation` call emitted schema-valid arguments with the
grounded two-panel count; its tool-result continuation reused 1,514 of 1,550
prompt tokens and completed normally. The matching SSE request emitted 224
valid JSON chunks, one `tool_calls` finish, and one terminal `[DONE]`.

Two simultaneous cold requests for the same previously unseen image executed
one 838 ms vision forward; the follower reused the immutable embedding and
both returned the same grounded answer. A client disconnected after 203 ms;
thirty seconds later no vision result, language request, or cache state had
been published. The next identical request was therefore cold and performed
one 18.429 second vision forward.

The official 4,096 by 4,096 processor maximum completed without materializing
an unbounded attention matrix: 16,384 visual tokens and 83,886,080 projected
F32 values, with a 55.097 second vision forward and 93 second cold end-to-end
time. Its exact repeat hit the vision cache and reused 16,444 of 16,449 prompt
tokens, then returned the correct two-panel answer in 20 seconds. This closes
the former out-of-memory correctness boundary. It does not yet establish the
desired comparative latency, so vision performance remains an optimization
gate rather than an inflated release claim.
