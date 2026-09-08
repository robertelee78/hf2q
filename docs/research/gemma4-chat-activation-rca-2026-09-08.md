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
Full-model and native numerical evidence for the final integration follows
below; acceptance remains pending until those gates complete.

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
