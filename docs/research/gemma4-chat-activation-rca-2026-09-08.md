# Gemma 4 chat startup: discovery-to-activation RCA

Date: 2026-09-08. Source baseline: `bb20fab8ca8adae8a4948941f54e83170b4b803f`.
Governing contract: ADR-051 (Accepted, amended 2026-09-08).

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

## Correction and proof

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

GitHub CI is pending for PR #191. Full-model validation requires an exclusive Apple
Silicon window; an existing DeepSeek server is currently preserved pending
operator permission to stop and restore it. This document does not yet claim
real-model startup or multi-turn acceptance.
