# Gemma 4 Unsloth chat: resolution, MLX ABI, and native matrix RCA

Date: 2026-09-08. Source baseline: `bb20fab8ca8adae8a4948941f54e83170b4b803f`.
Governing contracts: ADR-051 (artifact resolution) and ADR-008 (native inference).

## Trigger and observed failure

```sh
hf2q chat unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M
```

The chat-owned server's retained log contained:

```text
WARN hf2q::serve: multimodal text model has no unambiguous matching hosted mmproj; serving text-only
Error: locally discovered GGUF changed before runtime activation
```

Artifact: `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`, 16,947,541,728 bytes,
repository revision `c099eb48e663fd284577b04978a94ffccb261841`, independently measured digest
`f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f`.
Dependency: published `mlx-native = 0.15.1`, locked without a local override.
The reported release binary's measured SHA-256 was
`706ff7a3705c3860668c9a06567e915f5112a4a50872c8a51dced1ed46013bab`;
its embedded source identity was not established from the version number alone.

The managed filename-digest directory contained a symlink to the Hub blob.
The legacy revision directory also contained an independent regular GGUF of
the same byte length. Following the symlink identified inode 157109179;
the legacy file had inode 157122095. No model files were removed or repaired.

## Causal chain

1. `managed_artifacts::resolution` selects and retains a compatible local
   model. Runtime architecture and tensor-layout admission have already passed.
2. A missing verified projector sends this candidate through
   `prepare_local_candidate_with_catalog_resolver` for exact-revision companion
   discovery. The extra cache-candidate progress line reports a candidate probe;
   it does not establish that the cache candidate won selection.
3. That function previously constructed the legacy managed destination even
   when `explicit_output` was `None`. `PreparedLocalArtifact` verified or
   materialized text there before returning. This could hash/copy a large model
   merely because automatic projector discovery ran.
4. It changed `candidate.path` to that destination. This happened even when no
   unambiguous projector existed and the function chose text-only fallback.
5. The caller retained the original descriptor because no explicit output had
   been requested. `Candidate::into_resolved` checked whether that original
   descriptor remained stable, but did not compare it with the returned path.
6. `cmd_serve` did compare the returned path with the descriptor before loading
   weights. It correctly rejected the different inode. The error described an
   apparent external file change, although hf2q itself had changed the path.
7. The child exited before its private READY frame. Chat reported the generic
   early-exit error and retained the private log, following its diagnostic
   privacy contract.

The projector warning is a supported text-only fallback, not an explanation
for terminating text chat. This failure precedes native model loading; it does
not demonstrate missing Gemma 4 inference support or corrupt weights.

## Introduction and missing coverage

The implicit relocation dates to `2c80994494d23eb83ab9be1636186e4bc69f06dc`.
The hosted-resolution change `8f217f72af10c4fa67b8539a6c3d4ed11d0ef485`
introduced filename-specific managed destinations, exposing the path discrepancy
for new cache-linked entries. Legacy-path entries can mask the defect when the
prepared destination is already the selected path.

The existing `markerless_local_multimodal_uses_candidate_revision_config_and_companion`
test supplied an explicit output directory equal to the input directory. It
proved exact-revision projector choice but not ordinary no-output activation.
Other descriptor tests covered external replacement, not a stable original
descriptor accidentally paired with a different output inode.

## Failures behind the initial startup error

Fixing resolution exposed two independent runtime defects on the same artifact.
These were reproduced sequentially; READY alone was not an acceptance result.

1. **Warmup argmax ABI.** Gemma allocated `argmax_params` as two F32 elements,
   then wrote a U32 vocabulary count through the CPU slice. mlx-native 0.11.2
   already documented U32 parameters but did not check their dtype. The published
   0.15.1 backend checks it and rejected the F32/F32-tagged parameter allocation:
   `argmax_f32 requires F32 input, U32 index, F32 value, and U32 params; got F32/U32/F32/F32`.
   Commit `8f217f72` upgraded the dependency from 0.11.2 to 0.15.1, exposing this
   existing hf2q ABI error. The fix allocates one actual U32 parameter.
2. **Batched head representation.** With correct argmax parameters the model
   loaded, warmed up, and announced READY. The first real chat request then
   failed with `lm_head_batched requires a Q6_K lm_head (production decode path)`.
   This artifact's tied embedding/output head is Q8_0. Header admission and the
   scalar head accepted it, but the production batched head required Q6_K.
   Changing the label or requantizing the model to Q6_K would hide this mismatch.

The startup wrapper also formatted an error with `{e}`, discarding the nested
warmup error chain. It now adds an anyhow context, retaining the underlying
backend diagnostic in the private server log.

## What the previous quantization implementation actually did

The source baseline does not support a blanket claim that Gemma quantization
was fake. Packed expert matrices already executed through GGML expert kernels.
However, it did not consistently execute every matrix in its stored format:

| Matrix role on baseline main | Actual behavior |
| --- | --- |
| Token embedding | Entire quantized table expanded into an F32 allocation for CPU gather |
| Attention and shared MLP | Original packed matrix plus default-on F16 weight shadows for prefill |
| Expert stacks | Original packed storage; native scalar-ID dispatch for F32/F16/BF16 |
| Output head | Native Q6_K when selected; native Q8 reuse added in `8f217f72`; other policy-selected heads could be expanded/requantized or cast |
| Batched output head | Q6_K-only despite broader scalar/head admission |
| Token selection | Host embedding-based reranking coupled to the old head policy |

The broader native-storage work on `feat/universal-native-q5-landing` at
`e7328e0b` was not an ancestor of main. Its presence in another worktree did not
mean those changes were running in the reported command. This fix reconciles
its Gemma representation work (`d364f63c`) and later exact-storage/Q8 gather
corrections with current main, retaining published mlx-native 0.15.1 and current
scalar-expert dispatch. It does not merge that branch's unrelated family or
global route-activation changes.

Header inspection of the exact UD-Q4_K_M artifact found 658 tensors:

| Stored type | Count | Payload bytes |
| --- | ---: | ---: |
| F32 | 392 | 46,056,568 |
| Q8_0 | 207 | 2,802,235,392 |
| Q5_1 | 29 | 5,519,179,776 |
| Q4_K | 30 | 8,564,244,480 |

There are no BF16 tensors in this particular artifact. The similarly named
UD-Q8_K_XL artifact discussed in earlier hosted-resolution work is different.
The Q8 embedding is `[262144,2816]` in runtime row-major order and occupies
784,334,848 stored bytes. Its former whole-table F32 copy occupied
2,952,790,016 bytes. The 205 quantized attention/shared-MLP matrices could add
3,290,890,240 bytes of F16 shadows. These are artifact-derived allocation sizes,
not measured RSS or a performance claim.

The 29 Q5_1 expert-down stacks have input width 704, legitimately divisible by
Q5_1's 32-element block width. Requiring 256-element K-quant alignment here would
incorrectly reject their native representation.

## Independent queue-liveness failure exposed by the full gate

The real-artifact `slot_aware_n4_per_slot_parity_vs_serial` gate stalled after
native loading. A process sample showed the worker blocked on request intake
while concurrent generation futures remained unresolved; it was not executing
Metal or waiting for shutdown. Short inline admission yields after one request
(`eb132d4df`). If that request completes at its seed or terminates before
installing an active slot, the scheduler can report Idle while more runnable
requests already reside in the worker's pending queue. The Idle branch then
waited for a new channel message instead of admitting buffered work.

The fix consults the existing pending-request readiness predicate before
parking, preserving free-slot and active-prefix constraints. The same
real-artifact reproducer then passed in 5.24 seconds. Its requests now have a
120-second timeout and the model is shut down before assertions, so a future
regression fails with cleanup instead of hanging indefinitely. The sample did
not record the particular seed token; the scheduling state and successful
before/after reproducer establish the missing pending-queue check.

## Coherent runtime correction

Gemma now maps the embedding, explicit/tied output head, dense projections,
router matrices, and expert stacks from their original GGUF payloads. A load-time
inventory rejects anonymous ordinary matrix storage or an F16 shadow. Native
gather touches only requested embedding rows; it produces F32 activations and
applies Gemma's input scaling. Scalar and batched heads consume the same native
matrix. Token selection uses those logits directly, without the transformed
head's host reranking machinery.

Prefill O-projection dispatch checks whether the stored codec supports native
head-major BF16 input. Otherwise it permutes the activation into the normal F32
layout and uses that same stored weight matrix. No weight is transformed to
satisfy an optimized kernel. Header admission shares the native gather, dense,
and expert-route contracts with loading. Existing tokenizer, template, feature,
and tensor-shape rejection safeguards remain active.

Elementwise norm/scaling state may still be converted to its graph's F32 state
representation. This is not a claim that all activation or scalar-state bytes
retain on-disk dtype. Explicitly requested DWQ overlays keep their separate
representation contract.

## Resolution correction and initial proof

Automatic serving now uses the existing in-place projector preparation helper.
Explicit-output publication continues through its existing verified path.
The resolver also rejects a returned path that does not match its retained
activation authority; the runtime checks remain enabled.

Two new regression tests failed on the unmodified baseline:

- No-output projector preparation returned a different managed path.
- A stable original descriptor incorrectly authorized a different file
  containing identical bytes.

The positive regression exercises six cases: regular and symlinked text files,
each with absent, uniquely available, and ambiguous companions. It checks the
returned path, projector result, retained identity, and descriptor-backed bytes.
The existing explicit-output, replacement, and symlink-retargeting tests remain
part of the managed-artifact suite.

Focused validation after the fix:

| Command / contract | Result |
| --- | --- |
| `cargo check --locked --all-targets --all-features` | Passed |
| `cargo build --release --locked` | Passed |
| `cargo test --locked --bin hf2q serve::managed_artifacts::` | 106 passed, zero ignored |
| `cargo test --locked --bin hf2q core::bounded_file::` | 4 passed, zero ignored |
| `cargo test --locked --bin hf2q chat::` | 55 passed, zero ignored |
| `cargo test --locked --lib --all-features` | 51 passed, zero ignored |
| `cargo test --locked --test convert_integration --all-features` | 16 passed; 1 existing ignored streaming-RSS fixture test |
| `scripts/test_frictionless_binary.sh` against the release binary | Passed |
| Unsafe activation matrix: clean / refused / explicitly acknowledged | All three passed |

The regression runs used a temporary `XDG_DATA_HOME` to isolate model bindings.
The local release binary SHA-256 after the fix was
`de8c1fe24b86fe68d39f71e2ff7df65b5047cd71e666f37ed6dafe6a78f6d2c0`.

The initial resolution-only commits passed GitHub CI at `cec9f25c` (run
34202930463). That result does not cover the subsequent native Gemma changes.
Full-model and native numerical evidence for the integration follows below.
GitHub checks are bound separately to PR #191's final head.

## Native integration validation (before final main reconciliation)

- 137 Gemma module regressions passed, zero ignored.
- Native-filter suite: 63 passed; one unrelated existing DeepSeek full-checkpoint
  test remained ignored. This includes nonzero Q8 embedding/head comparison to
  an independent F64 CPU calculation at widths 1/2/4/8/9/32, exact mapped-byte
  preservation, scalar-expert admission, and the U32 argmax ABI.
- The exact hosted-selector chat returned `GEMMA4_READY` twice, with 15 reused
  prompt tokens on the follow-up.
- Three fresh-server runs of `scripts/test_gemma4_agentic.sh` passed all tool,
  replay, automatic-tool, SSE, continuation and source-syntax predicates.
  Each used 7,105 prompt tokens and reused 7,098 on replay. Median cold/cached
  TTFT: 4,608.219 / 77.341 ms; completed cached SSE tool call: 352 ms; tool-result
  response: 5,236 ms. These are measured gate results, not a speedup claim.
- Ordinary load inventory: 296 mapped matrices, 16,928,913,408 stored matrix
  bytes, zero anonymous matrix bytes.
- Four distinct HTTP CSV transcription requests matched their serial outputs
  exactly under concurrent serving. The separate short-request internal parity
  gate caught the queue-liveness defect above.
- The first reference trial passed tool/continuation/SSE semantics but exhausted
  the transcription token budget in reasoning: peer default thinking differed
  from hf2q's rendered request. Matched-request trials therefore explicitly
  disable thinking on both sides; this failed trial is not a performance result.

Full-model host: Apple M5 Max, 128 GiB. Engines were loaded sequentially with
DeepSeek stopped under operator authorization. The available peer source is
`74a7c897f049c17e7080423aa2111776eff6ebbf`, clean, with executable and linked
library hashes recorded together. The repository pin remains unavailable as
already documented in the hosted-resolution RCA; no result here represents
validation of that unavailable pin.

## Final integrated evidence

Runtime source: `5787e2660ae745ee9a9c542754c797e1ccfb657f` (includes main's
`28394ee4` logprobs change). Clean locked release build passed. Tested executable
SHA-256: `b00d001def7445a2acd3fcc51b7f22a61e3d607d2828f0254056569fbe68bff9`.
Subsequent evidence/harness documentation does not change runtime source.

| Focused integrated check | Result |
| --- | --- |
| Gemma module regressions, release | 137 passed, zero ignored |
| Native route/storage regressions, release | 63 passed; one unrelated existing full-DeepSeek test ignored |
| EAGLE3 consumer regressions, release | 234 passed, zero ignored |
| Managed artifact regressions, release | 106 passed, zero ignored |
| Sampling/logprobs regressions, release | 22 passed, zero ignored |
| Unsloth four-slot vs serial real-model gate | Passed after the pending-queue fix |
| Unsloth fresh/reused 4,096/8,193-token boundary gate | Passed, 16.70 s |
| Existing locally produced Ara Q5_K_M agentic gate | Passed all predicates |

The existing artifact was `gemma4-ara-2pass-APEX-Q5_K_M.gguf`,
20,576,631,488 bytes, SHA-256
`82beae39cdee643824dde5bc3fb1a3d6e2e4f8701572930163b0d703298bcf82`.
It retained its native Q6_K tied embedding/head. Its matrix inventory contained
296 mapped views, 20,558,010,880 stored bytes, zero anonymous matrix bytes.
The tool fixture had 7,103 prompt tokens and reused 7,096; cached TTFT was
74.328 ms. This proves that native execution also works on the existing local
artifact, not only the new publisher's quantization mix.

### Matched reference requests

All three hf2q and all three peer trials passed tool selection/arguments,
tool-result continuation, complete valid SSE tool calls, and 24-row CSV
transcription. Both returned identical terminal sentinel and transcription
content. Empty tool-message content is `""` versus `null` between the APIs;
generated tool-call IDs are not compared. Reported completion-token counts
differ by one, so this is not a claim of identical token-accounting conventions
or bit-identical logits.

Frozen request SHA-256:
`2ceab515397730010fb83349f88cd10c7da56346fcdca0e20fddcbd965c4d714`.
Repository-context SHA-256:
`2c894c9ed9cf02d5454e9756e6836ffbeed4f256c9e35c544cc451636476b4ef`.
Manifest SHA-256:
`82aa8bee40e772edc78b97d26d0bb8e5868778743d409b95c8d52d93ea68780b`.
Common settings: same Unsloth artifact, same
request bytes, greedy temperature 0, explicit thinking disabled, repetition
penalty 1, one slot, 262,144 context capacity, 4,096 prompt transaction/batch
size. Native Gemma uses its production F16-K/TQ-HB-8 V cache; the peer uses
F16 K/V. The six fresh-process trials ran hf2q/peer/peer/hf2q/hf2q/peer with
normal engine warmup. This is a functional and latency comparison, not a
controlled optimization benchmark: cache representations differ and unrelated
host CPU work was present.

| Measured quantity (three-run median) | hf2q | Peer at 74a7c897 |
| --- | ---: | ---: |
| Cold tool prompt tokens / cached | 7,101 / 0 | 7,101 / 0 |
| Cold tool response | 5,571.6 ms | 3,280.9 ms |
| Cold tool prefill rate | 1,416.6 tok/s | 2,472.9 tok/s |
| Cached tool response | 517.5 ms | 406.5 ms |
| Cached tool reused tokens | 7,094 | 7,100 |
| Tool-result response | 5,733.0 ms | 2,454.2 ms |
| Tool-result prompt / reused tokens | 11,203 / 7,094 | 11,205 / 7,097 |
| SSE first semantic event after continuation | 313.2 ms | 3,128.5 ms |
| Complete streamed tool call after continuation | 511.2 ms | 3,445.8 ms |
| CSV transcription prompt / completion tokens | 270 / 239 | 270 / 240 |
| CSV decode rate | 89.50 tok/s | 97.37 tok/s |

Cold tool response samples (ms): hf2q `[4882.1, 5688.9, 5571.6]`, peer
`[3073.3, 3336.1, 3280.9]`. CSV decode samples (tok/s): hf2q
`[90.69, 89.50, 87.90]`, peer `[98.02, 95.87, 97.37]`.
The post-continuation SSE comparison includes each engine's different prefix
retention behavior; it is not a comparison of identical cache hits. The peer
was faster on cold prefill and sustained decode in these runs. No performance
win is claimed for removing Gemma's transformed matrix copies.

The reference build reports 10820 / `74a7c897f`, AppleClang 21 / Darwin arm64.
Its small executable depends on shared libraries; relevant SHA-256 identities:

| File | SHA-256 |
| --- | --- |
| `llama-server` | `1e4f1f5b71a67259598bff0c797b2366a67c2c07ac35b69f6e52fcfce9e6f5ed` |
| `libllama-server-impl.dylib` | `5abb989c51be0883ff793c78b8ec53e9ead9231495f4bed8157e828b7b221edb` |
| `libllama-common.dylib` | `274b63961253ef99bf891962ee61521aac894253753d9f4083b7e263cdf6ef58` |
| `libmtmd.dylib` | `b2c0726c85492e71f56d92d061de140b90dba7e9e14f292bc11966cfd99a006b` |
| `libllama.dylib` | `20b90ed8910e7c2fb1fb70530d43e28732f9702b7a7cb3c6db7fdfb5f66c3f59` |
| `libggml-metal.dylib` | `c0f8f702c033f88696ff77484f99b0b1bb5ccccb912ca588a9ce7316873779d1` |
| `libggml-cpu.dylib` | `57715ea6765d4fd1ef4167ed5ab24582903a6035561ccc720a95f526b842b35e` |
| `libggml-base.dylib` | `b9b42c13a4e4c0bda319620c907c7adf31b275d1d728ab904e52dbb2ab07c158` |

The full 26-file name-to-SHA manifest has digest
`a2478d0dc07a18595e0f6d9082db73e8147e3c028b807b73a188f5c155bddb08`
(SHA-256 of sorted compact JSON). This is explicitly an available-source
reference, not the unavailable repository pin or a release gate.

### Reproduction

Use an exclusive Apple Silicon window. The canonical launcher is
`scripts/serve_gemma4_opencode.sh`; set `HF2Q_BIN`, `MODEL`, `MMPROJ` to an
absent path for text-only tests, `PORT`, `MAX_SLOTS`, and `REP_PENALTY` explicitly.
Always stop a test server before starting another full-model engine.

```sh
# Against a running four-slot native server:
BASE_URL=http://127.0.0.1:8092 MODEL=Gemma-4-26B-A4B-It \
  scripts/test_gemma4_agentic.sh

# Export one request and make thinking explicit for both engines:
MODEL=Gemma-4-26B-A4B-It RUN_ID=gemma-native-matched-20260908 \
  HF2Q_AGENTIC_REQUEST_ONLY_OUTPUT="$EVIDENCE/base.json" \
  scripts/test_gemma4_agentic.sh
jq '. + {chat_template_kwargs:{enable_thinking:false},hf2q_enable_thinking:false}' \
  "$EVIDENCE/base.json" > "$EVIDENCE/request.json"

python3 scripts/test_gemma4_native_reference.py \
  --engine hf2q --base-url http://127.0.0.1:8092 \
  --request "$EVIDENCE/request.json" --tool-result "$PWD/Cargo.toml" \
  --output "$EVIDENCE/native-1"
# Add --concurrency against a four-slot server to compare four distinct
# concurrent outputs against their serial references.
```

For the reference, start the recorded peer build with the same artifact/alias,
`--parallel 1 --ctx-size 262144 --gpu-layers all --batch-size 4096
--ubatch-size 4096 --flash-attn on --cache-type-k f16 --cache-type-v f16
--jinja --temp 0 --repeat-penalty 1 --metrics`, then run the same harness with
`--engine peer` and its endpoint. Repeat each engine three times with a fresh
process and a new output directory. Keep launch logs, request/response files,
SSE bytes and binary/library hashes together.

The hardware tests use the same artifact via `HF2Q_BYTE_EQUIV_E2E_GGUF` and
`HF2Q_BYTE_EQUIV_E2E=1`:

```sh
cargo test --release --locked --bin hf2q \
  slot_aware_n4_per_slot_parity_vs_serial -- --nocapture --test-threads=1
cargo test --release --locked --bin hf2q \
  gemma_fresh_and_reused_4096_8193_bounded_outputs_match -- --nocapture --test-threads=1
```

Unfiltered `cargo test --locked`, full vision inference, speculative decoding
with a real drafter, and the unavailable pinned-peer artifact are not claimed
as validated. The focused EAGLE3 tests cover the updated target-embedding
consumer; the real-model acceptance above covers text chat and agentic serving.
