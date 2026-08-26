# ADR-044: Qwen3.8 native conversion and inference

- Status: Accepted for native text conversion and serving; exact server
  speculation re-opened after a longitudinal verifier-coherence failure;
  vision candidate is under exact-artifact acceptance
- Date: 2026-08-16
- Updated: 2026-08-25 — the canonical server owns fixed-K3 MTP and
  request-history speculation with per-proposer measured cost gates. GGUF
  inference preserves the artifact's declared weight encodings. A 508-decision
  scalar-versus-width-four real-artifact gate found that the Qwen3.8-local
  `mul_mv_ext` default could change a target decision after repeated verifier
  rounds even though the earlier four-position gate passed. That non-exact
  model-local route is removed; speculation must re-pass longitudinal identity
  and performance on the shared native routing policy before re-acceptance.
  Q5_K_M artifacts with Q5_K token embeddings and Q6_K output heads now retain
  and execute those exact representations. The published dependency also makes
  equal-logit argmax choose the lowest vocabulary index deterministically.
  A one-slot worker now uses the measured 4,096-token prefill quantum while
  multi-slot workers retain the 2,048-token fairness ceiling. The qualified
  Q5_K_M one-slot workload now has a sealed, stable faster-than-baseline result.
  Universal Qwen3.8 acceptance now requires the same fail-closed correctness,
  physical-width, and matched-performance matrices over BF16, Q4_K_M, Q5_K_M,
  Q6_K, and Q8_0; individual cells do not authorize a universal claim. Exact
  Qwen3.8 gates are bound to their immutable abliterated text artifacts. The
  schema-v2 receipt is the single content-hash authority for both gates and
  server startup: it includes a shell-checkable portable snapshot plus the
  loader's full nanosecond identity. A valid v1 ledger is upgraded through one
  v2 hash before server use; it never authorizes runtime hash reuse. Exact
  cross-format swaps preload a bounded fail-closed directory of those same v2
  identities into the per-artifact policy registry so activation does not
  repeat an already-proven full-file scan. The exact five-format hardware
  swap matrix is sealed: five A→B→A cells plus a 17-generation BF16-hub
  chain passed with exact semantic replay and no stale cache or mapping state.
  Decode routing is immutable model-owned state; preflight, registries, cache
  identity, and dispatch use that same value, so A→B→A model swaps cannot leak
  policy through process environment. Model labels no longer change the shared
  coherent Qwen routing defaults. The protected cross-family release gate
  now also gives Qwen3.8 an authoritative agentic cache-lifecycle phase: the
  shared lifecycle fixture binds every successful unary and SSE response to
  the exact leased generation and artifact, and the terminal sealed manifest
  includes that receipt beside binary, dependency, and model identity. The
  five-format physical-width matrix is also sealed at widths 1, 2, 4, 8, and
  16 with exact scalar replay. A matched Q4_K_M ABBA run closed the former
  one-slot gap, but upstream advanced before the universal matched-physical
  matrix began; that result is retained as measured progress rather than
  current comparison authority. The matched gate now binds an externally
  frozen complete reference-runtime closure and records the clean exact gate
  harness separately from the immutable hf2q candidate source commit. A
  resumed packed-KV Stage-A/B fusion was also tested and rejected: it reduced
  an isolated verifier forward but did not improve the exact end-to-end server
  ABBA, so none of its runtime changes are part of this decision.
  A clean cross-family replay on the current main lineage then falsified the
  MTP MoE admission seal: the qualified Qwen3.6 MoE artifact stores
  `ffn_gate_inp_shexp.weight` as its native rank-one `[hidden]` vector, while
  the MTP topology checker alone had drifted to `[1, hidden]`. Target preflight
  and target loading already required the exact rank-one form. One shared
  descriptor validator now governs all three boundaries and a mutation test
  rejects rank-two squeeze semantics. The exact Qwen3.6 MoE cell and complete
  cross-family matrix must be rerun on the corrected commit before this replay
  can claim the prior seal.
  The subsequent five-codec replay also falsified BF16 admission before its
  first forward: the descriptor distinguished scalar BF16 for dispatch but
  still sent its byte extent through a block-quant-only helper. The exact
  native storage calculator now handles F32/F16/BF16 and block codecs from
  their declared GGML block geometry, and both ordinary and MTP embedding
  admission use it. Focused scalar extent mutations and the Metal-backed
  embedding admission matrix pass; BF16 and the complete five-codec artifact
  matrix must be rerun on the corrected commit.
  The proof authorities now reopen all five artifact snapshots immediately
  before final sealing, require the physical matrix's exact 64-token, 48-GiB,
  MVN-on/MV_EXT-off workload in every format, and seal the matched matrix's
  exact 48-GiB KV budget plus 100-ms launch-skew ceiling in both child and
  outer receipts. Ambient or cross-format weakening fails validation.
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

### 2026-08-23 representation-audit correction in validation

The universal matrix is a per-tensor contract, not a per-layer or per-artifact
shortcut. A source audit found remaining shortcuts that inferred gate and up
projection metadata from one another, rejected native Q4_0 embedding storage,
copied the rank-2 F32 convolution weight through host memory, and omitted native
Q2_K/Q3_K/Q5_1/IQ4 projection routes already available to the execution layer.
Those shortcuts cannot remain merely because the first measured Qwen3.8
artifact does not exercise every combination.

The correction under validation carries dense and expert gate/up/down codecs
and strides independently, admits only role/codec pairs with a native route,
keeps `ssm_conv1d.weight` in one mapped matrix view, and uses the exact Q4_0
embedding gather. The Q5_0 correction extends that same invariant to mapped
22-byte legacy blocks for embeddings, dense projections, and expert stacks;
Q4_0-only fused routes remain ineligible and fall through to ordinary native
Q5_0 execution without changing the weight. A shared activation row does not imply shared weight
representation, and a fused route is eligible only when the participating
stored codecs satisfy its exact contract. The dense deciding spike proved that
scalar/block cross-class siblings can execute through the same
representation-polymorphic per-projection dispatcher; a mixed
Q4_0/BF16/F16 SwiGLU canary now exercises that path after activating the BF16
route. A separate rank-3 execution canary uses Q4_0 gate, Q8_0 up, and Q5_1
down expert stacks with distinct strides; it matches an independent CPU oracle
at decode M=1 and matrix M=33 through both ordinary and prefill-arena paths.
Scalar/block mixtures inside rank-3 expert stacks remain a separate
admission question until the native expert-ID proof and hf2q wiring pass; they
may not be silently coerced.

This subsection records the reformulated contract. It does not activate a
universal claim. Acceptance still requires locked compile and regression
gates, tiny mixed-codec GPU proofs, the complete physical-width and multi-slot
matrix for every applicable supported family, A→B→A model swaps, exact text and
multimodal coherence, and matched performance receipts for BF16, Q4_K_M,
Q5_K_M, Q6_K, and Q8_0 artifacts.

### 2026-08-23 activation-lifetime correction under validation

A source audit separated server isolation from same-thread activation safety.
Production server generations own distinct worker threads, so their Qwen
thread-local state dies when the worker exits. The direct Qwen entry points,
tests, and any future reusable worker can replace a model on one thread,
however. On that path the model epoch and routing policy rejected the stale
`ForwardGpuCache`, but replacement uploaded the new weights before dropping
the old cache. The allocation-backed decode arena and legacy expert-ID scratch
also survived the epoch transition with the prior `MlxDevice` residency owner.
That is an allocation-lifetime and model-swap defect even when later dispatch
identity checks prevent stale arithmetic.

The correction candidate makes teardown precede the first allocation in the
replacement loader and fail closed: drop the stale weight/cache owner first;
reset and clear the decode arena; drop legacy and scalar expert scratch;
replace the residency-only weight pool; only then create B's model-load
device, map its weights, freeze its model-owned routing policy, and build its
forward cache. An activation failure uses the same teardown rather than
leaving a half-frozen registry or scratch owner alive. Legacy expert-ID scratch
is independently keyed by activation epoch and device-registry identity so
equal capacity is never reuse authority. Model-owned `MlxQWeight` dispatch
records remain safe: they retain pipeline and geometry metadata but no weight
buffer, and disappear with their own model. Process timing caches in
`mlx-native` remain pointer-free metadata. Installed native route plans are
epoch-, device-, shape-, and current-activation-authority bound, while the
model registry independently freezes the exact GGML routing policy.

The source candidate adds A→B→A scratch-allocation, same-address/new-epoch,
and changed-policy canaries. It is not runtime acceptance until locked tests
and the real model-swap residency gate run on the integrated dependency.

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
- Prove A→B→A releases A's weight cache and decode/expert scratch before B's
  first allocation, binds B's exact routing policy, and repeats the same
  release/rebind for A2 without process restart or retained high-water bytes.

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
Explicit `off` remains the operator escape. Loading a model resolves one
immutable routing policy from the shared native defaults plus explicit
operator overrides. Qwen3.8 no longer changes those defaults from its artifact
metadata. The former model-local values (`dense_decode_mvn=false`,
`dense_decode_mv_ext=true`) passed a four-position gate but failed the repeated
508-decision gate at completion token 206 because `mul_mv_ext` does not preserve
the scalar reduction tree. With the shared defaults restored (`mvN` enabled,
`mul_mv_ext` disabled), all 508 decisions were exact. The process environment
is never mutated, and the canonical launcher validates but does not synthesize
either routing variable.
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

The exact transaction includes semantic state as well as GPU cache bytes and
cursors. Forced-reasoning selection advances a private staged budget and
publishes that cursor only after the target decision succeeds, so a fallible
forward or constrained-sampling path cannot consume an uncommitted close
token. Coherent MTP warmup likewise publishes its newly derived carried hidden
row only after canonical semantic selection succeeds. Model-free regressions
prove that discarding the staged budget preserves the same retry decision and
pin the hidden-publication ordering; the real-model lifecycle gate remains the
authority for end-to-end behavior.

The bounded-prefill transaction also remains live through construction of the
initial decode semantic state. Target KV may already contain the final prompt
chunk when grammar-constrained sampling selects the first output token; an
error there now restores the chunk-entry target/MTP cursors and DeltaNet
ping-pong selection instead of relying on a later scheduler reset. The
`final_prefill_semantic_error_restores_the_chunk_entry_boundary` regression
mutates both slot cursors and every recurrent selector before injecting that
failure, while preserving the peer slot.

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
The rejected Qwen3.8-local `mul_mv_ext` policy reduced the verifier phase to
roughly 59-63 ms per round, but its different reduction tree is not a valid
exact-decoding optimization. The earlier gate proved only four decisions plus
eight continuation decisions; it did not cover enough repeated transactions
to expose the completion-token-206 fork. The replacement performance path is
a shared Q5_K multi-row kernel whose per-row accumulator and reduction order
are a literal scalar clone, admitted only after byte identity and longitudinal
model gates.

The earlier final ABBA binary SHA-256
`c217e128e28a18d6dbe48ec88155a7bab8a0f633b7b691187f9f118fa2f24ce7`
preserved all 24 OFF/AUTO choices. Its six code samples had a 41.86 tok/s
internal median and improved wall time 51.66% over ordinary decode; its six
repeat samples had a 50.28 tok/s internal median and improved wall time
70.29%. The two AUTO arms recorded 314 verified proposer rounds and 804
accepted draft tokens, with zero cost-disabled generations. The adjacent
two-cycle external code receipt had a 45.49 tok/s median, leaving hf2q 7.98%
behind on this measured code workload. This supersedes both the original
22.5% gap and the intermediate near-tie receipt. It is not a universal speed
claim, and at that point true physical multi-slot batching remained open.

The later exact-build Q4_K_M ABBA run used hf2q commit `9138cfaa` and binary
SHA-256
`3a3339a327224ea73e00351853a8237c053ad7302bc3320ad52d75f4caccfb76`
against then-current comparison commit `c060ca974c773c7c3d17fd1b66dc9d312bc292c0`.
The AC Automatic, nominal-thermal A-B-B-A receipt passed 12 complete-Rust
compile/evaluator receipts and 12 exact-transcription receipts. hf2q completed
the code groups in a 9.748788-second median versus 10.273607 seconds, a
1.053834x ratio, and repeat groups in 4.354343 seconds versus 5.049949 seconds,
a 1.159750x ratio. Median internal decode was 54.088493 versus 48.029688
tokens/second; median semantic streamed TTFT was 396.6325 versus 463.0040 ms.
The worst hf2q arm beat the best comparison arm in wall time and end-to-end
throughput for both groups, with stable non-overlapping bands. Summary
SHA-256 is
`3e28cea84b852622136a57f240617c6af098ffaa3032b6373e306d3db4b337aa`.
Upstream advanced to the pin below before the universal matched-physical
matrix began, so this receipt proves that the old 22.5% gap was closed for its
exact identities but does not authorize the current universal comparison.

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

### Verifier profile and rejected Stage-A/B fusion (2026-08-24)

The instrumented exact K3 boundary counted 163 verifier command buffers, 18
syncs, and 1,347 dispatches. Its mean phase split was 111.523 ms target verify,
18.787 ms three-step draft, 0.453 ms embedding, and 0.665 ms MTP catch-up. That
made a resumed packed-KV Stage-A/B fusion a legitimate deciding spike: it kept
projection, KV write, packed attention, and output projection inside one
encoder for each of Qwen3.8's 16 full-attention layers.

The isolated four-position target/cache-handoff proof was exact: OFF and ON
logit bits and kernel multisets matched, while command buffers fell from 163
to 115. Five independent-process pairs all favored the candidate; median
forward latency fell from 132.783 to 112.558 ms (15.232% less latency, or
17.969% more phase throughput). That result was necessary but not sufficient.

The full server ABBA then held the Q4_K_M artifact
`1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a`,
candidate binary
`5b313d54921f083a3a76335d29269f1bc3bb2f1f16a98f4c3def91f173b7512d`,
AUTO speculation, sampling, six 128-token cases, and one physical slot fixed.
Trial order was unfused/fused/fused/unfused. Every `choices[0]` value matched
exactly and both arms proposed and accepted the same token counts. After ABBA
aggregation, however, code median time changed from 2.962270 to 2.972309
seconds (-0.338% throughput) and repeat median from 1.387792 to 1.407205
seconds (-1.380%). The 5% floors failed. The runtime spike and its temporary
harness changes were removed; this fusion is **FALSIFIED for the measured
server workload**, not accepted on the strength of a microphase win.

The remaining verifier hypothesis is inside the unchanged target work itself:
quantized matrix-vector/matrix-matrix execution and wider exact verifier
batching may amortize material cost. Any such candidate must repeat the same
two-level proof: exact target/cache handoff first, then an interleaved complete
server receipt. Submission-count reduction alone is not authorization.

The earlier apparent short-prompt AUTO TTFT comparison was invalid because it
compared warmed OFF against first-use AUTO with different prompt/output shapes.
The matched first-transition observations were 528.750 ms OFF and 542.542 ms
AUTO, about 13.8 ms of AUTO-specific setup against roughly 3.706 seconds of
decode saved on that request. AUTO therefore remains enabled.

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

The accepted performance binary is built from
`8f999ca75c014c1298670f593d1a9dd606fe5e43` with exact embedded provenance and
SHA-256
`15e7959f6227833b889e0ad09d4be86794f06fb4c6b12992d0a3de37d2049f7c`.
The paired baseline is bound to commit
`9a286ac98d2cab74231bd3f1fc3f2b8bdf05422e` and executable SHA-256
`26fa7cb0f42c24468b95c7f6727e36ceb0eb4da3871e0ca0e3bcf7310705095b`.
The sealed `qwen38-q5-matched-8f999ca7-r3` receipt has summary SHA-256
`0ba9fd132bbbe7fffdc79e87d2d747e6b07e7e9e8388428e1babf5d6ba22528f`.

That AC Automatic, nominal-thermal A-B-B-A run passed 12 executable-code
quality receipts and 12 exact-transcription receipts. hf2q completed the code
groups in a 10.579800-second median versus 11.410625 seconds, a 1.078529x
speed ratio. It completed exact-repeat groups in 4.784610 seconds versus
5.485687 seconds, a 1.146528x ratio. Median internal decode was 48.642751
tokens/s versus 43.834526 tokens/s, and median semantic streamed TTFT was
429.711 ms versus 488.924 ms. hf2q recorded 324 proposer rounds and 918
accepted draft tokens; the baseline recorded 990 drafted and 936 accepted
tokens. The worst hf2q arm beat the best baseline arm in wall time and
end-to-end throughput for both groups. Maximum group wall/decode spread was
1.436742%, maximum per-case spread was 2.665646%, and the observed bands did
not overlap.

Two prior attempts remain explicitly rejected. A Low Power Mode run had large
hf2q arm drift, and the first Automatic run exceeded the 5% reference-repeat
group bound. A subsequent attempt changed from nominal to fair thermal pressure
during the second baseline arm and was terminated before measurements. That
failure proved a 30-second loaded-idle interval insufficient, so the accepted
gate requires 120 continuous nominal seconds before every arm, three fixed
transcription warmups totaling at least 512 generated tokens, within-engine
semantic identity, no more than 5% aggregate and 10% per-case spread, and
non-overlapping observed speed bands. The result establishes superiority for
this exact artifact and workload, not a universal Qwen3.8 speed claim.

### Cross-quant and physical-width acceptance contract (2026-08-22)

The accepted Q5_K_M receipt above is one required matrix cell, not an
exemption from artifact breadth or concurrency. Universal Qwen3.8 authority
requires all of the following exact author-hosted artifacts at immutable
revision `0a72776892f98db49381fdf69f4b9982222ec9dc`:

| Format | GGUF file | Bytes | SHA-256 | `general.file_type` |
|---|---|---:|---|---:|
| BF16 | `gguf/qwen38-abliterated-sft-bf16.gguf` | 54,657,734,208 | `f30d9a6ea40ca3c5265d0996a460ad1474173c40c8e7f04c0b03caf6084c2cee` | 32 |
| Q4_K_M | `gguf/qwen38-abliterated-sft-hf2q-q4_k_m.gguf` | 16,810,714,944 | `1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a` | 15 |
| Q5_K_M | `gguf/qwen38-abliterated-sft-q5_k_m.gguf` | 19,535,701,568 | `4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e` | 17 |
| Q6_K | `gguf/qwen38-abliterated-sft-q6_k.gguf` | 22,431,000,128 | `78f62a87ef851443d4e0c74c4e1eb1dfe73e3bf0ded3cf320ec80f763020ddb3` | 18 |
| Q8_0 | `gguf/qwen38-abliterated-sft-q8_0.gguf` | 29,047,084,608 | `53c076e5117be1391e76a9746998fbe2040e6b69a73aa47d1c1b0ca97a8a2c99` | 7 |

`scripts/qwen38_artifact_contract.sh` is the single machine-readable identity
authority. Every real-model runner validates format, byte length, digest, and
GGUF file type before model allocation. The four-position matrix additionally
proves native stored-width retention, shared target/MTP output-head allocation,
batched-versus-scalar decisions at four positions, and an eight-token handoff.
The physical matrix must pass exact concurrent-versus-scalar response parity
at widths 1, 2, 4, 8, and 16 for every artifact. The matched pinned-peer matrix
must pass the existing quality, transcript, stability, non-overlap, and
faster-than-peer criteria for every artifact. A matrix receipt is published
only after all cells pass; a missing or failed cell fails the matrix.

The physical authority's canonical workload is `max_tokens=64`,
`kv_cache_budget_bytes=51539607552`, `HF2Q_DECODE_MVN=1`, and
`HF2Q_DECODE_MV_EXT=0`. The matched physical authority uses that same KV
budget and a `0.100`-second maximum client launch skew. Both policies are
recorded in the outer receipt and every applicable child receipt; validators
require the exact values and reject cross-format drift. The four-position
runner reopens and compares every catalog artifact's hash-bound portable
snapshot after all five cells, so an early artifact cannot change while later
formats run and still contribute to a sealed matrix.

### Repeat work-unit correction (2026-08-24)

The exact-repeat arm measures a fixed expected response byte sequence, so its
cross-engine throughput denominator must describe that shared semantic work,
not either server's OpenAI usage accounting or SSE transport framing. Live
BF16 width-one closure evidence produced the same expected 349-byte response
from both engines, while hf2q reported `completion_tokens=66` and the pinned
reference reported `completion_tokens=67`. The reference also emitted 69 JSON
SSE events while hf2q emitted 66: role, content chunking, finish, and usage
frames are not decoder-token work. The extra reference usage token is
consistent with terminal-EOS accounting, but the contract does not infer or
subtract EOS because that policy can vary by model and engine.

The gate therefore records one `semantic_completion_tokens` value by invoking
the candidate hf2q binary's GGUF tokenizer on the immutable expected response
with special-token insertion disabled. Its receipt binds the expected-byte
SHA-256 and length, exact model SHA-256 plus snapshot, candidate binary SHA-256
and source commit, and token-ID-stream SHA-256. Repeat aggregate TPS uses that
canonical count; raw API `completion_tokens` remains recorded and must be
stable within each engine's two ABBA trials, but is never required to match
across engines. Every streamed and scalar response must still exactly match the
expected bytes, with one role, finish, and DONE event. Code-generation arms
remain evaluator-equivalent rather than cross-engine byte/token comparable, so
they retain raw API accounting as a diagnostic and do not claim normalized
semantic-token throughput.

The current pinned comparison source identity is
`a14dba686aaafba3a2d6b5eb8820b0df5c5d2d92`, read from
`data/llama_cpp_pin.txt` and verified against the upstream branch on
2026-08-24. No binary or performance receipt built from that pin has yet been
accepted. Regeneration at build 10610 left all 24 quantizer fixtures
byte-identical and bound the dynamically linked comparison runtime with a
complete non-system Mach-O closure manifest, because the launcher executable
alone remained byte-identical across distinct engine dylibs. The earlier
`3f545bec`/build-10587, `9a286`, and `c060ca`
identities remain historical evidence only and cannot satisfy the current
matched gate.

Before that pin advance, the provenance-bound hf2q candidate at commit
`9138cfaa` and binary SHA-256
`3a3339a327224ea73e00351853a8237c053ad7302bc3320ad52d75f4caccfb76`
sealed the full physical matrix. All 25 format-by-width cells observed target
body/head widths 1, 2, 4, 8, and 16 and matched exact per-lane scalar replay.
The five N=1/N=16 aggregate decode endpoints were respectively BF16
10.657887/28.033998, Q4_K_M 31.146512/48.123313, Q5_K_M
26.989447/38.001907, Q6_K 23.557019/38.083372, and Q8_0
19.127482/45.029667 tokens/second. The sealed matrix SHA-256 is
`d0f4d215a776a17a24cfc13df6c3ad09c3df9eed2ec835f589e8e1f2ecfc6800`;
the evidence-manifest SHA-256 is
`62c4527d3742b5c0ff1739a9c42840b3c1f26f30088ea3231cbd713c14515e5b`.
Because the pin is part of the exact hf2q source identity, the final
current-pin candidate must repeat this physical proof before the universal
matched matrix may consume it.

The first universal matched-physical launch then falsified a live-tip
requirement before allocating a model: the upstream branch advanced from
`a14dba68` to `71cc86fa` during the few minutes between pin refresh, exact
build, and gate start. Requiring a high-churn remote branch to remain unchanged
through a multi-hour 100-arm matrix makes completion depend on unrelated
upstream inactivity and cannot produce reproducible authority. The corrected
contract records the exact commit observed current at refresh time, commits
that pin, and freezes it for the entire run. Every arm revalidates the clean
exact source, launcher, complete non-system Mach-O runtime closure, model,
request manifest, and checked-in pin. A later upstream commit is a reason to
schedule the next comparison, not to invalidate measurements already in
progress against an immutable observed-current snapshot.

A source-only preflight audit then found that this frozen-snapshot contract
was still only internally consistent: each artifact child captured and sealed
its own reference runtime-closure digest, but the five-artifact parent did not
bind an expected digest or require all five children to share one. A sibling
dylib could therefore change between children and produce a coherently sealed
mixed-reference matrix. The corrected gate requires the operator-frozen
runtime-manifest SHA-256 before any evidence output, artifact hashing, or model
load; each child checks it at entry and exit, records both expected and
observed identity, and the parent rejects any cross-child closure or pin
cohort drift before publication. Because this gate-only repair advances the
harness commit without rebuilding the already qualified hf2q binary, binary
provenance is now supplied through a separate `HF2Q_SOURCE_DIR`: it must be a
clean exact worktree at the commit embedded in the binary. The harness has its
own clean exact recorded commit and launches hf2q through the candidate source
tree's launcher. This preserves exact binary/source identity rather than
relaxing it to ancestry or to a newer harness commit.

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
tokens; it is not a suffix automaton. True width-N Qwen body/head execution is
now proven by the exact 25-cell physical matrix rather than inferred from
request concurrency. The next acceptance sequence is the current-pin matched
physical matrix over all five formats and all five widths. Adaptive depth may
follow only after K2/K3 transaction parity and measured cost show a win; an
unmerged adaptive-MTP change is not treated as shipped source truth. Split-K
retuning and true packed TQ6/TQ5 storage remain downstream of the established
bandwidth and occupancy regime.

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
verification protocol. A sealed runtime recorder hashes content once and emits
the schema-v2 receipt consumed directly by the server; the shell checks its
portable snapshot while runtime checks the full nanosecond stamp. An enclosing
release gate reuses an unchanged v2 receipt without either a second recorder
call or `shasum` scan. A valid v1 ledger is retained only for shell
compatibility and is upgraded once before server use. Every runner checks the
snapshot again after inference, so avoiding a second content scan does not
waive mutation detection. The speculation runner defaults only the
paired path and requires the caller's expected digest, preventing a baked-in
digest from silently drifting at the next artifact rotation. The model-free
contract pins the paired path and release-gate digest, requires receipt use in
both runners, and rejects any direct full-model `sha256_file` reread.

The first exact large-artifact runtime proof used hf2q commit
`ab3e60803085b2966d287922b0f50dc67be143b9`, binary SHA-256
`3926e121bf54ca5ad083e408589bceaeb014ad47256026398bbbb6259d0910a5`,
and the 54,657,734,208-byte BF16 artifact with SHA-256
`f30d9a6ea40ca3c5265d0996a460ad1474173c40c8e7f04c0b03caf6084c2cee`.
Exactly one Rust recorder process performed the content scan and emitted the
schema-v2 receipt. Server startup with that receipt reported a 114.8 ms model
load, followed by its independent 4.14 s native-route calibration/warmup, and
published a healthy HTTP endpoint. The pre-fix fresh-server path spent roughly
90 seconds in `TextArtifactIdentity::inspect` hashing the same file before
mapping it. This proves the duplicate scan is removed for the exact artifact;
it does not waive the post-load stamp check or the final cross-format swap
matrix.

The same protected workflow previously verified the Qwen3.8 text artifact but
ran only its long-decode comparison. That was not agentic lifecycle authority:
Qwen3.6, Gemma, and DeepSeek ran the shared cancellation/rollback/isolation
fixture, while the Qwen3.8 family entry omitted it. The release gate now starts
`scripts/serve_qwen38_opencode.sh` in an explicit text-only four-slot phase,
runs `scripts/test_agentic_cache_lifecycle.sh`, and requires cold tool use,
retained-prefix continuation, active-SSE cancellation, queued exact-retry
reuse, and unrelated-slot isolation. Each of the four unary requests and the
cancelled SSE request must expose the same generation-bound execution receipt:
the qualified text-artifact digest, `qwen35` family, `qwen35` GGUF
architecture, pool key, and positive generation. The final manifest embeds
the validated lifecycle summary and its exact SHA-256 under Qwen3.8, then a
standalone verifier rechecks the summary and manifest sidecars, sealed binary
digest, packed-crate dependency receipt, and snapshot-bound model identity.
These checks authorize that concrete lifecycle route only; they add no decode
throughput or broad performance claim.
