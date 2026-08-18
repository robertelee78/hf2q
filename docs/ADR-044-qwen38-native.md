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

The first native hf2q performance spike expanded Q4_K feed-forward tensors to
dense storage. It measured a 32-token prefill in 11.94 seconds and 32 decoded
tokens in 2.37 seconds (13.5 tokens/second), versus 27.8 tokens/second in the
comparison runtime. That result was a useful falsifier, not the accepted
loader. The production loader now keeps Q4_K and Q6_K feed-forward tensors in
their native quantized representation and fails loudly instead of silently
expanding an unsupported quantization type. Initial GPU materialization fell
from 24.65 seconds to about 10.7 seconds.

On the exact 16,810,714,752-byte candidate above, an uninterrupted seven-run
temperature-zero API sequence generated 512 tokens per run. hf2q end-to-end
rates were 28.96, 30.10, 29.83, 29.49, 29.19, 28.74, and 27.99 tokens/second,
for a 29.19 tokens/second median. The matched single-process comparison on the
same artifact and user prompt produced a 24.87 tokens/second median across
seven runs. The sustained median advantage was 17.4 percent. Both paths
produced the same correct in-place Rust sort-and-deduplicate implementation
and the same required `calculate_sum` call with integer arguments 17 and 25;
the result continuation completed normally. This closes the original dense
fallback performance defect without weakening the quality gate.

A later matched long-context peer run used llama.cpp build 10451
(`10bf611e5`) with the same Q4_K_M artifact, one 131,072-token slot, Metal
flash attention, default F16 K/V, temperature zero, and thinking disabled. A
cold 105,029-token prefill took 493.839 seconds. Five exact-prefix 128-token
decode runs measured 15.734, 15.266, 15.934, 15.890, and 15.374 tok/s, a
15.734 tok/s median, with identical output. This is a production-default peer
comparison rather than cache-format parity because hf2q uses compressed TQ-HB
K/V. It replaces the former absence of any matched approximately 105K
llama.cpp evidence; it does not replace the exact hf2q legacy/Q2 release gate.

The exact native server artifact passed `/readyz`, unary and SSE text,
required-tool unary and SSE calls, schema-correct arguments, tool-result
continuation with 407 of 429 prompt tokens cached, automatic thinking without
private client flags, cancellation with checkpoint recovery, and two
simultaneous requests.
An ordinary three-message follow-up returned the remembered value with 27 of
54 prompt tokens reused. A separate coding follow-up reused the complete
96-token stable prefix and returned a valid function plus unit test. The gate
also found and fixed an invalid cache invariant: verifier KV cursors must agree
with one another, but the optional MTP cursor is independent until speculative
decoding runs.

These measurements establish functional native text support and sustained
single-request performance superiority for the measured artifact and workload.
They do not establish speculative MTP acceptance, cross-family vision
completion, or multi-slot aggregate superiority. In a four-request cold-prefix
run with 256 generated tokens per request, hf2q completed 1,024 tokens in
30.003 seconds (34.13 aggregate tokens/second). The matched four-slot
comparison completed in 19.103 seconds (53.60 aggregate tokens/second). Source
inspection explains the gap: `decode_batch_qwen35` currently loops through
four scalar forwards, while the faster runtime executes one width-four model
step. A native, state-isolated width-N Qwen body/head is therefore a blocking
performance follow-up; the single-request result must not be generalized to
concurrent serving.

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

### Multimodal continuation and answer-progress correction (2026-08-17)

A real 90,044-token OpenCode image turn exposed two independent serving
defects. The preceding 88,559-token text continuation had reused 84,522 tokens,
but adding the conversation's first image selected a cold slot and performed a
full prefill. Source tracing showed that the request-global vision fingerprint
added on 2026-08-16 excluded the text-only anchor before token-prefix affinity
could consider it. The same request then produced 1,889 reasoning tokens over
6 minutes 56 seconds and could legally continue to the 8,192-token completion
limit without emitting content or a tool call. The existing "semantic fragment
ready" log fired on raw decoded text before reasoning/tool routing and was
therefore not evidence of answer progress.

The accepted correction is:

1. Restore an idle text-only anchor into a first-image request only when its
   exact tokens precede the first soft span and the request proves ordinary
   text coordinates on all four mRoPE axes over the complete anchor. Reuse is
   limited to the snapshot. Active affinity, live retained state, and every
   image-bearing anchor remain exact-vision-fingerprint-only.
2. Expose the vLLM-compatible `thinking_token_budget` request field. The Qwen
   SlotAware decoder counts generated reasoning tokens in transport-neutral
   state; at the boundary it emits a tokenizer-derived transition and
   `</think>`, then continues decoding the answer. Natural `</think>` or
   `<tool_call>` ends budget enforcement. Required or named tool grammars and
   SerialFifo reject the option rather than accepting unsupported semantics.
3. The canonical Qwen launcher sets a 2,048-token default ceiling. Shorter
   completion windows adapt the ceiling to preserve roughly one quarter for
   the answer; `THINKING_TOKEN_BUDGET=0` disables it. Explicit per-request
   budgets are exact and fail when they leave no answer capacity.
4. Qwen `<tool_call>` implicitly closes a still-open reasoning channel, as in
   current Qwen-aware serving parsers, while preserving the native marker for
   structured tool routing. "First answer event" now means a successfully
   delivered non-empty content or structured tool-call delta after routing.
   Reasoning-only completion is logged explicitly and never reported as answer
   readiness.

The cache design comparator is vLLM's parent-hash block chain with typed extra
hashes: multimodal identity affects blocks containing the multimedia input,
not causally earlier text blocks. The longer-term Qwen cache layout should
encode an ordered text/media event chain, including model/template/position
namespace, media content hash, ordinal, token span, grid, and DeepStack shape.
That redesign is not required to repair the proven first-image suffix case and
must not weaken exact image identity.

Primary comparators:

- vLLM automatic prefix caching design:
  <https://docs.vllm.ai/en/latest/design/prefix_caching/>
- vLLM multimodal cache identity:
  <https://docs.vllm.ai/en/latest/features/multimodal_inputs/>
- vLLM thinking-budget control:
  <https://docs.vllm.ai/en/v0.20.1/features/reasoning_outputs/#thinking-budget-control>
- Qwen thinking-mode and budget guidance:
  <https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md>

Acceptance requires model-free rejection/acceptance matrices for mRoPE and
image identity, exact budget-boundary state tests, implicit tool-boundary and
post-router progress tests, then a real Qwen3.8 text-turn -> first-image-turn
SSE gate. The real gate must report nonzero cached prompt tokens on the image
turn, a delivered answer event before the completion limit, valid terminal
usage, coherent grounded output, and healthy `/readyz` after completion.

The exact-artifact gate passed on 2026-08-17 on an Apple M5 Max. The release
binary SHA-256 was
`24f47b0394b231ce58d00213e118d190f5c3efa1ad270846283626d61e5f4e90`;
the text and projector hashes remained the bound candidates above. A cold text
turn established an 86,072-token idle snapshot inside an 86,077-token prompt.
The first-image continuation then reported 86,072 cached of 86,172 prompt
tokens and prefilled only the 100-token suffix. Its real GPU vision forward
completed, the 64-token reasoning budget forced the tokenizer-derived close,
and the SSE stream delivered `The dominant color in this image is red.` with
82 completion tokens, `finish_reason=stop`, one terminal `[DONE]`, 9-second
wall time, and HTTP 200 readiness afterward. Server telemetry independently
recorded the 86,072-token prompt-boundary hit, the budget transition, and the
first delivered answer event.

The matched agentic gate also passed exact cached replay, automatic and
required tools, SSE reconstruction, tool-result continuation, and byte-exact
source arguments. The tool-result turn measured 14.303 seconds; the gate's
historical 10-second wall-clock bound missed by 732 ms on the first run, then
passed under an explicit 15-second bound without relaxing any semantic or
cache assertion. This residual latency is recorded rather than conflated with
the corrected first-image cache and reasoning failures.

### Long-context GQA-cooperative decode candidate (2026-08-18)

At a 104,966-token prefix, Qwen3.8's 24 query heads and four KV heads cause the
legacy TQ-HB kernel to request roughly 20.96 GB of KV traffic per generated
token: each six-head GQA group reloads and dequantizes its shared KV head per
query head. A bandwidth model built from the accepted 29.19 tok/s short-context
baseline predicts 12.99 tok/s; the observed agentic turn measured 13.4 tok/s.
The same model predicted 24.09 tok/s at 17,807 tokens versus 22.1 observed and
18.43 at 49,169 versus 18.0 observed. This identifies the long-context decode
loss as a kernel-layout limit, independent of the separate model/tool retry
loop.

Published `mlx-native 0.10.9` added the first bit-exact D=256
GQA-cooperative Q2 TQ-HB kernel. It shares one packed K/V load and
dequantization across two query heads without changing per-query
online-softmax state or the final reduction layout. Its first sealed hf2q
OFF/AUTO/AUTO/OFF run measured 17.1767 versus 19.5490 tok/s, a 13.8112% gain,
and correctly failed the fixed 15% release gate.

Split retuning did not close the gap: NSG4 remained faster than NSG2/NSG1,
and the best NWG point improved the candidate by only about 1.3%. The revised
kernel instead retains each lane's query slice in registers. It remains
bit-identical across TQ5/TQ6/TQ8 and reduces threadgroup memory from 11,264 to
10,240 bytes, crossing the 32 KiB three-workgroup occupancy boundary. Three
isolated 104,966-token processes measured 1.603x, 1.610x, and 1.615x; a
1,000-step run had a 0.999 first-versus-last median ratio. A local
path-patched hf2q spike then measured 16.5732 versus 20.6089 tok/s, a 24.3506%
gain, with identical semantic hashes. That spike proves the reformulated
hypothesis but is not release authority. The same register-resident
implementation is now published and checksum-pinned as `mlx-native 0.10.10`
without a Cargo patch; the packed hf2q short/long receipt remains the required
downstream authority. Q3 was not retained because its threadgroup-memory and
occupancy tradeoff did not justify a second production variant.

Upstream release workflow `32148168017` tested the exact source, packed
archive, and archive downloaded back from crates.io; it then verified the
GitHub release bytes. Tag `v0.10.10` resolves to
`c6c5092f6f5a0cc4f3c79e98c3caa63eef78d542`, and both public crate surfaces
have SHA-256 `b390c48281b0134b821d6da300b1e385580b9e6456f0536fd744e1bc711572cf`.

hf2q's candidate selector defaults to `auto`: it remains on the legacy kernel
below 8,192 KV tokens and requires the exact Qwen3.8 D=256/GQA/no-mask geometry
above that threshold. `HF2Q_QWEN_GQA_Q2=off` is the supported escape hatch;
`on` still cannot bypass hard geometry checks, and invalid values fail safe to
off. This default is accepted for merge only when the same packed hf2q binary
passes the shipping contract's OFF/AUTO/AUTO/OFF release receipt: identical
greedy output at both a sub-8,192 short prompt and near 105K, no more than 2%
short-context regression, at least 15% mean long-context decode gain, no arm
above 5% spread, lower independently measured long-request curl wall time,
one exactly-once SlotAware completion event per request derived from the
finalized result, and a continuous fair-or-better thermal envelope. The short
log snapshot must prove `auto` retained the scalar route before the long AUTO
request proves Q2 selection. Isolated mlx-native numbers are dependency
evidence, not hf2q release authority.

The next exact-output optimization sequence is a two-dimensional H2xP2
query-head/query-position verifier, then native Qwen3.8 MTP and a dynamic
suffix-automaton proposer behind measured acceptance/cost routing. Split-K
retuning and true packed TQ6/TQ5 storage follow only after the cooperative
kernel and verifier establish their new bandwidth/occupancy regime.
