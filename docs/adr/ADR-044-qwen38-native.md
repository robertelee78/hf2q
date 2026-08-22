# ADR-044: Qwen3.8 native conversion and inference

- Status: Accepted for native text conversion and serving; exact server
  speculation accepted for the measured one-slot workloads; vision candidate
  is under exact-artifact acceptance
- Date: 2026-08-16
- Updated: 2026-08-22 — the canonical server now owns exact fixed-K3 MTP and
  request-history speculation with per-proposer measured cost gates. GGUF
  inference preserves the artifact's declared weight encodings; the qualified
  width-four verifier substantially narrows the measured one-slot code gap.
  Q5_K_M artifacts with Q5_K token embeddings and Q6_K output heads now retain
  and execute those exact representations. The published dependency also makes
  equal-logit argmax choose the lowest vocabulary index deterministically.
  A one-slot worker now uses the measured 4,096-token prefill quantum while
  multi-slot workers retain the 2,048-token fairness ceiling. The remaining
  single-user gap and true physical multi-slot batching remain performance
  blockers. Exact Qwen3.8 gates are bound to the paired abliterated text
  artifact and reuse snapshot-bound model-verification receipts instead of
  repeatedly scanning an unchanged multi-gigabyte GGUF.
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
7. Conversion and inference have separate quantization authority. `hf2q
   convert` may quantize source safetensors into the requested GGUF. Inference
   must execute each GGUF weight in its declared stored representation; it
   must not dequantize and silently rebuild that weight as Q4_0, BF16, or any
   other codec. Explicit learned overlays own their format and fail closed
   where the native graph cannot execute it.

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
architecture routing, while conversion and evidence remain explicit. The
canonical launcher selects `HF2Q_QWEN_SPECULATION=auto`. **Updated
2026-08-21:** the qwen35 server engine now defaults to `auto` when
`HF2Q_QWEN_SPECULATION` is unset — the launcher-only default left every
bare `hf2q serve` on ordinary decode even though all admission paths fail
closed (unsupported semantics, prompt-cache hits, unavailable proposer, or
a negative per-generation cost decision stay on ordinary target decode, so
the worst case for default-on `auto` is ordinary decode plus telemetry).
Explicit `off` remains the operator escape. In the same change, loading a
Qwen3.8-identified model applies the launcher's qualified decode route
(`HF2Q_DECODE_MVN=0` + `HF2Q_DECODE_MV_EXT=1`) at engine load when those
variables are unset; the route stays Qwen3.8-scoped because `mul_mv_ext`
is not bit-exact and no other family carries the qualifying receipt.
Also since 2026-08-21, `hf2q setup` persists the qualified agentic serving
profile (repetition penalty 1.05, thinking budget 2048, tool-continuation
budget 512) into `config.toml` when the operator optimizes for long agent
and tool-use prompts, and `hf2q serve` applies those values only to
environment variables the operator has not exported. Vision is a
separately measured candidate surface and does not inherit text-only
performance authority.

## Acceptance evidence

### Native artifact execution invariant (2026-08-20)

The accepted Q4_K_M artifact has 866 tensors: 360 F32, 439 Q4_K, and 67
Q6_K. The production loader retains all 506 quantized tensors as native GGUF
blocks. That includes the Q4_K embedding, Q6_K output head, all 304 target
attention/DeltaNet projections, all 192 target FFN weights, and all eight
MTP-local quantized weights. The target and MTP share the same native
output-head buffer. Runtime activations, norms, and recurrent state remain in
their declared compute formats; no GGUF weight is dequantized and requantized
during model load or inference.

This corrects the former loader behavior, which expanded the embedding and
attention weights and rebuilt several target/MTP weights as Q4_0 or BF16.
That path both changed the converted model and retained roughly 39 GiB of
avoidable duplicate unified-memory storage. Synthetic fixtures retain their
explicit F32 paths. Native DWQ overlays fail closed rather than mutating the
artifact through the legacy F32-to-Q4_0 route.

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

A later matched long-context external-reference run used build 10451
(`10bf611e5`) with the same Q4_K_M artifact, one 131,072-token slot, Metal
flash attention, default F16 K/V, temperature zero, and thinking disabled. A
cold 105,029-token prefill took 493.839 seconds. Five exact-prefix 128-token
decode runs measured 15.734, 15.266, 15.934, 15.890, and 15.374 tok/s, a
15.734 tok/s median, with identical output. This is a production-default peer
comparison rather than cache-format parity because hf2q uses compressed TQ-HB
K/V. It replaces the former absence of any matched approximately 105K
external-reference evidence; it does not replace the exact hf2q legacy/Q2 release gate.

The exact native server artifact passed `/readyz`, unary and SSE text,
required-tool unary and SSE calls, schema-correct arguments, tool-result
continuation with 407 of 429 prompt tokens cached, automatic thinking without
private client flags, cancellation with checkpoint recovery, and two
simultaneous requests.
An ordinary three-message follow-up returned the remembered value with 27 of
54 prompt tokens reused. A separate coding follow-up reused the complete
96-token stable prefix and returned a valid function plus unit test. The gate
also found and fixed an invalid cache invariant: verifier KV cursors must agree
with one another. Ordinary prefixes may omit speculative metadata, but a
prefix that advertises reusable MTP state must have equal target and MTP
cursors at the exact published token count.

These earlier measurements establish functional native text support and
sustained single-request performance superiority for the measured artifact
and workload. They did not establish speculative MTP acceptance,
cross-family vision completion, or multi-slot aggregate superiority. In a
four-request cold-prefix
run with 256 generated tokens per request, hf2q completed 1,024 tokens in
30.003 seconds (34.13 aggregate tokens/second). The matched four-slot
comparison completed in 19.103 seconds (53.60 aggregate tokens/second). Source
inspection explains the gap: `decode_batch_qwen35` currently loops through
four scalar forwards, while the faster runtime executes one width-four model
step. A native, state-isolated width-N Qwen body/head is therefore a blocking
performance follow-up; the single-request result must not be generalized to
concurrent serving.

### Exact server speculation evidence (2026-08-20)

The launcher previously exported `HF2Q_SPEC_DECODE=0`, but the OpenAI worker
never read that CLI-only variable. The apparent forced-off policy was dead
configuration. The live control is now
`HF2Q_QWEN_SPECULATION=off|auto`; `serve_qwen38_opencode.sh` selects `auto`
and validates its launcher alias `QWEN38_SPECULATION` before model load.

The accepted Qwen3.8 transaction uses the target's post-output-RMSNorm hidden
rows and catches the MTP KV cursor up across the complete prompt. Each MTP
round drafts three tokens, discards draft-only MTP KV, verifies
`[seed,d0,d1,d2]` in one target forward, processes those target rows back
through MTP, and commits the accepted target/MTP/DeltaNet boundary. History
lookup instead uses a request-owned token-position index with exact 6-12-token
suffix comparison and at most three continuation tokens. Both proposers own
independent measured four-round cost controllers. A proposer is disabled only
after two consecutive unprofitable windows; one negative window is retained
as noise, and a profitable window clears that strike. This keeps the gate
cost-based without treating acceptance as a profitability proxy.

Real parity testing found that the multi-token DeltaNet capture path wrote
per-row convolution captures but did not write the next ping-pong convolution
state. Full acceptance therefore selected stale state and corrupted the next
block. The corrected path materializes the final captured convolution row
inside the same command buffer. A second correction computes multi-slot
capture offsets from physical capacity while slicing only the active depth;
grow-only K4 storage can no longer address another slot after a shorter
history block. The focused GPU gates compare the capture path's own final
convolution state byte-for-byte and exercise the physical-stride case.

A later correctness hardening enlarged cancellation recovery and the first
rebuilt receipt failed loudly: AUTO regressed code throughput 23.6% while
ordinary qL1 timing stayed flat and speculative qL4 rounds slowed about 50%.
Revert/reapply isolation showed LLVM had folded cold cancellation work into
the hot slot loop. Marking recovery `#[cold] #[inline(never)]` retained the
strict cursor/error handling and restored the prior speculative timing. The
final ABBA receipt below is from that rebuilt binary; the failed receipt was
not averaged into an accepted claim.

A final rustfmt-only wrap inside that cold recovery function changed the
release digest to
`7dba0d159cc9a2c9181e63ad00a02cb0dc257fa7a832f419a73243736a275287`.
Its exact ABBA rerun preserved all choices but failed the code floor at
-12.695%. That run also showed host-wide drift: ordinary internal decode fell
from roughly 29 tok/s in the first OFF arm to roughly 20 tok/s in the fourth,
so it does not by itself isolate code layout from system state. The artifact
was rejected. Restoring only the original source shape reproduced the
accepted release binary below byte-for-byte. Rustfmt remains informational
for this measured line until the hot-path layout is made insensitive to crate
span changes; the performance receipt, not cosmetic formatting, is the gate.

The reproducible one-slot receipt from
`scripts/qwen38_speculation_ab.sh` used release binary SHA-256
`e2b7a3ec831b8b85ddf52728cb7f46cbb6da8401ed06f355c996b029b4a6c190`
and the 16,810,714,752-byte Q4_K_M artifact SHA-256
`0fa8acc661d0edc60276c43705619fd848682dbf768ced9fe46cd8a572b8043d`.
Fresh one-slot servers ran in fixed OFF/AUTO/AUTO/OFF order. All 24
`choices[0]` values across three 128-token deterministic Rust prompts and
three repeat-heavy prompts matched byte-for-byte. The code-workload
six-sample median fell from 4.777795 to 3.886297 seconds, a 22.940%
throughput improvement. The repeat-workload median fell from 2.485324 to
1.729047 seconds, a 43.740% improvement. The two AUTO arms recorded 250
verified proposals and 662 accepted draft tokens. These are fixed-workload
one-process medians, not a universal speed claim.

The original matched external-reference run used reference commit
`521a64cd0197`, Metal
flash attention, one 262,144-token slot, fixed `--spec-draft-n-max 3`,
temperature zero, repetition penalty 1.05, and the same artifact/prompts.
Current HEAD has no adaptive-MTP controller. Its first code receipt was 47.02
tok/s median while the then-current hf2q receipt was 36.42 tok/s, a 22.5%
gap. That result is retained as the falsifying baseline, not current
performance authority.

Source profiling localized most of the gap to target width-four verification:
the exact-tree Q4_K/Q6_K mvN path cost roughly 98-147 ms per K3 target round.
The qualified Qwen3.8 launcher instead uses `mul_mv_ext` for K-quant widths
4-8 while leaving the process-wide default unchanged. The real Qwen3.8 gate proves the
Q4_K and Q6_K width-four routes were dispatched, all four target decisions
match four independent sequential forwards, all hybrid-cache cursors agree,
and eight subsequent sequential decisions remain exact. The verifier phase
then measured roughly 59-63 ms per round.

The final ABBA binary SHA-256
`c217e128e28a18d6dbe48ec88155a7bab8a0f633b7b691187f9f118fa2f24ce7`
preserved all 24 OFF/AUTO choices. Its six code samples had a 41.86 tok/s
internal median and improved wall time 51.66% over ordinary decode; its six
repeat samples had a 50.28 tok/s internal median and improved wall time
70.29%. The two AUTO arms recorded 314 verified proposer rounds and 804
accepted draft tokens, with zero cost-disabled generations. The adjacent
two-cycle external code receipt had a 45.49 tok/s median, leaving hf2q 7.98%
behind on this measured code workload. This supersedes both the original
22.5% gap and the intermediate near-tie receipt. It is not a universal speed
claim, and true physical multi-slot batching remains open.

A follow-up diagnostic tested whether the 1.05 target repetition penalty was
lowering MTP acceptance because the fused drafter uses raw argmax. With the
penalty removed, the three code cases accepted 244 of 300 drafted tokens
(81.33%), effectively unchanged from the final receipt's 252 of 309 (81.55%).
The single warm arm was faster, but that isolated run is not performance
authority. The acceptance hypothesis was rejected, so no penalty-scatter
kernel or proposal-policy change is included.

The final phase profile on the production route measured target verification
at roughly 58-61 ms of a 68-75 ms speculative round. Three chained draft
steps cost roughly 8-10 ms, native embedding 0.2-0.4 ms, MTP catch-up 0.4-0.7
ms, and partial-reject state recovery 5-7 ms only on rejected rounds. That
profiled code request reached 47.10 tok/s. Embedding and rollback are therefore
not credible explanations for the full adjacent-receipt gap; target
verification remains the dominant optimization surface, and the spread
between the 41.86 tok/s ABBA median and this warm run requires a better
interleaved variance-controlled receipt before another kernel change.

The agentic gate under AUTO returned a schema-valid `get_weather` tool call,
continued from its tool result with 298 cached of 363 prompt tokens, emitted a
valid 48-event SSE stream plus one `[DONE]`, recovered from client cancellation
with the verified checkpoint, and served a healthy following request. A
four-slot wave produced byte-identical choices for all four requests and no
runtime-unavailable fallback, but generated 450 tokens in 30.77 seconds
(14.62 aggregate tok/s). That is correctness evidence only. Qwen still
interleaves scalar slot forwards; true physical width-N batching remains the
blocking aggregate-throughput item.

### Native Q5_K_M execution and deterministic argmax (2026-08-22)

Issue 146 exposed a separate artifact-coverage failure: the server could parse
and report ready for a Q5_K_M file whose `token_embd.weight` was Q5_K, then
failed on the first forward because no admitted native gather existed. The
correction extends the one embedding-routing authority to Q5_K and Q6_K and
checks packed byte extent, storage dtype, executable capability, and resolved
kernel route before execution. Unsupported embedding encodings fail closed;
they are never expanded and rebuilt as another codec.

The accepted author-hosted artifact is repository
`jenerallee78/Qwen3.8-27B-Abliterated-SFT`, immutable revision
`0a72776892f98db49381fdf69f4b9982222ec9dc`, file
`gguf/qwen38-abliterated-sft-q5_k_m.gguf`, 19,535,701,568 bytes, SHA-256
`4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e`.
It contains 866 tensors and declares GGUF file type 17. Its token embedding and
MTP embedding-hidden projection are Q5_K; its output head is Q6_K.

The real Apple-Silicon storage/parity gate loads those bytes and asserts that
the legacy F32 embedding and output tables are empty, every target projection
remains in an artifact-declared K-quant representation, the MTP block borrows
the exact target output-head Metal allocation and byte offset, and a
four-position target verification followed by the shared-token handoff makes
the same greedy choices as serial one-position forwards. The gate passed from
hf2q commit `909dfd0b3dcce3635c54b2460771c91ee0f9ec2a` in 4.17 seconds.

That commit resolves published, checksum-pinned `mlx-native 0.11.2`. Its
protected release run `32567649315` verified the exact source, packed archive,
crates.io bytes, downloaded registry archive, and GitHub release bytes. Tag
`v0.11.2` and the release target resolve to
`54859646fd1aab1878f891c35060d0fb961bc1b2`; both public crate surfaces have
SHA-256
`22f4bd6661e77994c6f26a79fdd2c188f3d5252aa7e51616f5feb080b22da8e0`.
The release also corrects GPU argmax ties to choose the lowest vocabulary
index, matching the engine's CPU greedy rule and preventing exact speculative
verification from depending on the reduction tree.

The exact hf2q release binary built from
`909dfd0b3dcce3635c54b2460771c91ee0f9ec2a` has SHA-256
`ae0d4e566c6c8525ad87a1f9e9d0cbd53b9aba287fa48a1c7e497228c26f6afe`.
A prior matched Q5 run produced executable, byte-stable code and repeat
outputs but is rejected as performance evidence: hf2q's two ABBA arms drifted
by roughly 47-55% while macOS explicitly reported Low Power Mode. The current
gate therefore requires Automatic or High AC Energy Mode, three fixed
transcription warmups totaling at least 512 generated tokens before each arm,
within-engine semantic identity, no more than 5% aggregate and 10% per-case
spread, and non-overlapping observed speed bands. Current Q5 performance
authority remains pending that stable rerun; neither the rejected run nor this
correctness gate establishes parity or superiority.

The final binary above also passed the shared full-context coding contract with
a 20,584-character repository fixture and a real 230-byte Rust tool result.
It proved cold tool selection and arguments, exact cached replay, automatic
tool choice, SSE tool-call reconstruction, tool-result continuation, source
syntax preservation, and a cold-to-cached transition from zero to 7,039 reused
prompt tokens. The continuation added 127 tokens and completed in 1.525
seconds. The same gate with the 12,973-byte `Cargo.toml` tool result remained a
performance failure: 7,038 tokens were reused, but the 4,017-token uncached
suffix plus decode completed in 11.464 seconds against a 10-second limit. Its
semantics were correct; the latency failure remains open and is not waived by
the smaller passing fixture.

Merged commit `1aa7cdebcb2a` promotes a 4,096-token prefill quantum only when
the slot-aware worker owns one physical slot. Multi-slot workers and inline
surfaces retain the established 2,048-token ceiling, and every target
transaction remains hard-capped at 4,096 tokens. On the exact Q4_K_M artifact
above, the large continuation reused 7,020 of 10,887 prompt tokens and
processed the same 3,867-token suffix in both arms. Two current-base arms per
configuration measured 13.326 seconds at 2,048 versus 12.102 seconds at 4,096:
a 1.224-second, 9.2% median improvement. Cold and cached unary behavior,
automatic tool choice, SSE reconstruction, tool-result continuation, and
source syntax all passed in every arm. Lighter-load strict runs measured
9.766, 10.024, and 9.881 seconds at 4,096, so the absolute 10-second bound
remains host-sensitive and open; this evidence accepts the bounded incremental
speedup, not closure of the broader latency gate.

Post-merge validation on source
`cc0d99ae60a019c44133fe75a23f296a951e6afc` used release binary SHA-256
`6d7535c9a958d754e5c1518303ad359647537bd77682d7440051859ef3331234`
and the same Q4_K_M artifact. The opt-in real-artifact width-four gate executed
one test (not a filtered or skipped run) and passed four sequential-versus-
batched target decisions, native route assertions, hybrid-cache state, and
eight continuation decisions. A fresh OFF/AUTO/AUTO/OFF receipt preserved all
24 deterministic choices. Code wall-time median fell from 4.754997 to
3.092766 seconds, a 53.746% throughput improvement; repeat-heavy median fell
from 2.626848 to 1.444434 seconds, an 81.860% improvement. The AUTO arms
recorded 314 verified proposer rounds and 804 accepted draft tokens.

The full-context coding gate on that exact binary also passed required and
automatic tool calls, SSE reconstruction, tool-result continuation, source
syntax, and exact cached replay. The continuation reused 7,019 of 10,886
prompt tokens and processed a 3,867-token suffix in 9.954 seconds. This is an
exact-SHA pass of the 10-second gate, not evidence of comfortable latency
margin: an unrelated foreground application was consuming substantial CPU,
and the prior strict-run set still demonstrates host sensitivity. No thermal
or power warning was recorded. The paired speculation receipt remains valid
for relative OFF/AUTO behavior; the contended host state is not authority for
an absolute cross-stack throughput claim.

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
implementation was introduced in `mlx-native 0.10.10`; the release candidate
now resolves the additive, checksum-pinned `mlx-native 0.10.12` without a
Cargo patch. Version 0.10.12 bounds D512 prefill reads at a partial logical KV
tail and does not change the Qwen Q2 kernel. The packed hf2q short/long receipt
remains the required downstream authority. Q3 was not retained because its
threadgroup-memory and occupancy tradeoff did not justify a second production
variant.

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

The thermal producer compiles `scripts/macos_thermal_probe.swift` once before
the benchmark and reuses the resulting private executable for every sample.
The receipt binds the checked-in source digest, `/usr/bin/swiftc` digest, and
compiled binary digest; the independent verifier rehashes the checked-in
source. This identity-shape change advances the benchmark summary schema from
one to two; the enclosing release envelope remains schema one. This replaces
the prior per-sample `swift -e` launch after protected run `32336641261`
demonstrated an eight-second telemetry hole under load. It does not widen the
two-second sampling target, five-second maximum gap, Nominal-start requirement,
or Fair-only sustained allowance.

Test-ownership RCA (2026-08-18): commit `57d33b92` added three CPU-only GQA-Q2
policy tests directly to the Metal-owning `kv_cache.rs`. The cross-module GPU
discipline guard intentionally requires every test in such a file to acquire
the shared GPU lock first, so the full suite correctly reported 103 tests but
only 100 acquisitions. Adding locks to pure policy tests would have hidden the
ownership error and unnecessarily serialized them. The policy, parameter type,
and tests now live in CPU-only `gqa_q2_policy.rs`; `kv_cache` re-exports the
parameter type to preserve its caller contract. Hosted CI runs both the policy
tests and `iter230_a2_lock_discipline`, closing the coverage gap that let the
inconsistent module placement merge.

Native fixed-K3 MTP and request-history lookup now occupy the exact-output
server path. The history implementation is a request-owned token-position
index with exact 6-12-token slice comparison and up to three continuation
tokens; it is not a suffix automaton. The next performance sequence is to
close the matched single-user throughput gap, then replace scalar interleaving
with a true width-N Qwen body/head for multi-slot throughput. Adaptive depth
may follow only after K2/K3 transaction parity and measured cost show a win;
the current reference uses fixed depth three, and an unmerged adaptive-MTP
change is not treated as shipped source truth. Split-K retuning and true
packed TQ6/TQ5 storage remain downstream of the established bandwidth and
occupancy regime.

### Paired-artifact gate identity and verification lifecycle (2026-08-21)

The automatic multimodal conversion superseded the earlier text-only
Qwen3.8 gate artifact at the canonical model path. The accepted text GGUF is
now the 16,810,714,944-byte output bound to its automatically produced
projector, with SHA-256
`1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a`.
The paired 927,606,848-byte F16 projector has SHA-256
`463b264713f8e081f0fae753c80d8089308e01b1e2ac0948dd9966d0711d8f1b`;
these text-only performance gates do not pretend to exercise it.
The focused speculation harness had retained the removed vanilla filename,
while Cache lifecycle still accepted the preceding text-only abliterated
digest. Both were contract drift: a default benchmark could target no file,
and the release gate could not qualify the paired artifact now used by the
product.

The speculation and long-decode runners now use the shared one-time model
verification protocol. A standalone run hashes content once, records the
digest with the file's device, inode, size, modification time, and change
time, and reuses that receipt only while the complete snapshot is unchanged.
An enclosing release gate may pass its already-verified receipt. Every runner
checks the snapshot again after inference, so avoiding a second content scan
does not waive mutation detection. The speculation runner defaults only the
paired path and requires the caller's expected digest, preventing a baked-in
digest from silently drifting at the next artifact rotation. The model-free
contract pins the paired path and release-gate digest, requires receipt use in
both runners, and rejects any direct full-model `sha256_file` reread.
