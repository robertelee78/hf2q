# ADR-035: GGUF codec ownership — move encode side into mlx-native

**Status**: Proposed (sequenced after ADR-033 §P1 byte-identity closure and ADR-034 ship)
**Date**: 2026-05-19
**Supersedes**: nothing
**Related**: ADR-008 (mlx-native as compute backend), ADR-020 (DWQ/mixed-precision — qdq_legacy load-bearing scaffolding), ADR-033 (unified quant/convert pipeline), ADR-034 (speculative decode end-to-end)

## Context

The GGUF format codec is currently split across two crates:

| Component                              | Current home                                                | Natural pair                           |
|----------------------------------------|-------------------------------------------------------------|----------------------------------------|
| GGUF **reader** (header + tensor parse) | `mlx_native::gguf::GgufFile`                                | writer                                 |
| GGUF **writer** (header + tensor emit)  | `hf2q::backends::gguf::writer::GgufWriter`                  | reader                                 |
| CPU **dequant** (Q4_0..Q6_K, IQ4_NL)    | `mlx_native::gguf::*` (in-module helpers)                   | CPU quantize                           |
| CPU **quantize** (Q2_K..Q8_0, IQ4_NL)   | `hf2q::quantize::ggml_quants::*`                            | CPU dequant                            |
| Metal dequant / quantized-matmul        | `mlx_native::ops::*` and `mlx_native::shaders::*`           | CPU encode/decode (same canonical ref) |
| GPU QDQ round-trip (Q4_0/Q8_0)          | `mlx_native::ops::qdq_legacy` (ADR-020 Track 1 scaffolding) | independent (load-bearing future use)  |
| imatrix corpus driver                   | `hf2q::quantize::imatrix::*` (convert-time orchestration)   | stays with convert                     |

mlx-native already owns half the GGUF codec (reader + CPU dequant + Metal dequant). hf2q owns the other half (writer + CPU quantize). The split is historical, not designed: hf2q was the convert tool that grew quantize code in-place; mlx-native is the runtime library that grew GGUF parsing because it consumes the format. Neither side intentionally claimed ownership of the codec as a whole.

This is a mess that should never have been allowed: format-level kernels (byte-level encode/decode of the GGUF representation) ended up in the application crate instead of the library crate that defines the format. The asymmetry creates real friction:

- Any future tool that wants to write GGUF (e.g. a standalone imatrix-only quantizer) must depend on `hf2q` — a 100-file convert pipeline — to get at the encode kernels.
- The canonical reference for each kernel (`/opt/llama.cpp/ggml/src/ggml-quants.c`) is one file containing matched `quantize_row_qX` and `dequantize_row_qX` pairs. Splitting them across crates breaks the natural co-location.
- Cross-crate refactors (e.g. adding a new ggml_type, MTP tensor remapping in ADR-034 P2) require coordinated PRs across both crates instead of one.

## Decision

After ADR-033 §P1 (hf2q convert producing byte-identical GGUFs to `convert_hf_to_gguf.py | llama-quantize`) closes and ADR-034 (MTP+DFlash) ships, move the GGUF encode side into mlx-native, giving mlx-native end-to-end ownership of the GGUF codec.

### Split principle (the rule that drives every placement question)

A component moves to mlx-native iff **both** are true:
1. **Kernel, not driver** — touches GGUF byte format or quant math directly, not convert-time orchestration or policy (what to quantize / which corpus / which arch).
2. **Reusable by third parties** — a hypothetical non-hf2q tool would plausibly want it.

A component stays in hf2q iff it is convert-time orchestration, source-format reading (safetensors → IR), arch-specific tensor mapping, or CLI/imatrix-driver glue.

### Concrete move list

1. **Move** `hf2q::backends::gguf::*` (entire directory: `writer.rs`, `types.rs`, `mod.rs`) → `mlx_native::gguf::writer::*` plus the shared types fold into `mlx_native::gguf`.
2. **Move** `hf2q::quantize::ggml_quants::*` (Q2_K..Q8_0, IQ4_NL, `common.rs`, `quantizer.rs`, `ggml_type.rs`, `llama_ftype.rs`, `tensor_ref.rs`, `vision.rs`, `apex/`, `standard_policy.rs`) → `mlx_native::quantize::ggml_quants::*`.
3. **Stays in hf2q**: `convert/` (orchestrator, source_reader, arch/), `quantize/imatrix/` (driver code that *uses* the kernels — including `imatrix/gguf_writer.rs` which is the imatrix-format writer, distinct from the GGUF tensor writer), CLI bindings.
4. **Stays in mlx-native, unmoved**: `mlx_native::ops::qdq_legacy` (GPU Metal-shader QDQ round-trip for ADR-020 Track 1 sensitivity computation — load-bearing future scaffolding). Internal kernel-kind rule: `mlx_native::ops::*` hosts GPU Metal dispatchers; `mlx_native::quantize::*` hosts CPU Rust kernels + the GGUF writer.
5. **Re-export surface**: `mlx_native::quantize::*` is the stable public path; external consumers (hf2q itself, third-party tools) import from there.
6. **Migrate tests in lockstep**: kernel-level fixture tests + byte-cmp tests follow the kernels into mlx-native; convert-orchestrator tests stay in hf2q.

### Module path

`mlx_native::quantize::ggml_quants::*` (flat top-level `quantize`, ggml_quants as sub-module) was chosen over `mlx_native::gguf::quantize::*` and bare `mlx_native::quantize::*` after surveying Rust crate conventions for matched-pair codecs:

- `image::codecs::png` co-locates encode + decode pairs per format.
- `flate2`, `zstd`, `gix-object` per-type all co-locate matched directions.
- `parquet::file::reader` / `writer` and `arrow-ipc` split *only* when reader and writer are independently-versioned subsystems with different state machines, which is not our case.

The `ggml_quants` sub-segment is preserved because ggml quants are tensor-format kernels reused by Metal matmul independent of GGUF I/O — coupling them to the file-format module (`mlx_native::gguf::quantize::*`) would create a misleading import path — and because it leaves room for future non-ggml schemes (AWQ, GPTQ, MX-formats) as siblings under `quantize/`.

### Structural notes for the move

- `mlx_native::gguf` is a single 1294-line `mod.rs` today. The move converts it to a directory: `mlx_native/src/gguf/{mod.rs, writer.rs, types.rs}`.
- Use `git mv` per-file; cross-crate `git log --follow` is preserved per-file. Reviewers should expect blame continuity on individual files, not on the directory as a whole.

## Why "after §P1 closes and ADR-034 ships"

Two reasons, in priority order:

1. **Don't mix code motion with semantic changes.** Doing the move concurrently with the FMA + F16-roundtrip correctness work would make review impossible and git bisect across the move boundary painful. Byte-identity must be locked down in the current structure first; then the move is a pure refactor verifiable by the same byte-cmp test from `/opt/hf2q/scripts/byte_cmp_gguf.py` and the kernel fixture tests.
2. **Don't churn imports while ADR-034 is mid-flight.** ADR-034 P2 adds Qwen-3.5/3.6 MTP tensor mapping that touches both crates; landing that against shifting module paths multiplies merge pain. ADR-034 lands against the current paths; ADR-035 then re-homes them.

## Consequences

**Positive**:
- One crate owns the full GGUF codec; canonical-reference fidelity is enforced in one place.
- New tools writing GGUF depend on a thin `mlx-native` library, not the full hf2q app.
- Reader/writer and quantize/dequant pairs co-locate, matching the canonical `ggml-quants.c` source layout.
- Future cross-crate work on the format (new K-variants, new ggml_types) is single-crate.

**Negative**:
- One-shot churn: ~10.6k LOC (639 writer + 5,966 ggml_quants kernels + 3,551 apex policy + 470 trait/common/types) moves crates. Imports change everywhere.
- mlx-native crate-size grows; build times rise marginally.
- Public API of mlx-native expands: `mlx_native::quantize::*` becomes a stability commitment.

**Mitigations**:
- The move is mechanical: paths change, code does not. Byte-cmp on hf2q convert output before and after the move proves zero semantic regression.
- Crate-size impact is one-time; the kernels were already going to be linked into the hf2q binary anyway, just transitively.

## Rollback policy

Forward-fix only. The move lands as one PR (or one PR per crate with the hf2q side blocked on the mlx-native release). Once landed, any surfaced bug gets fixed in place; no kill-switch, no feature flag, no shim layer (per `[[feedback-no-backwards-compat-2026-05-18]]`). This forces care up front — CI byte-cmp must be green before merge, not relied upon as a safety net after.

## Acceptance Criteria

- All `src/quantize/ggml_quants/` paths in hf2q resolve to `mlx_native::quantize::ggml_quants::*` imports (matching Decision §5 — single re-export path, no aliases).
- All `src/backends/gguf/` paths in hf2q resolve to `mlx_native::gguf::*` imports.
- `/opt/hf2q/scripts/byte_cmp_gguf.py` on a freshly-converted Q4_K_M model reports zero diffs against the same canonical reference used by ADR-033 §P1 — i.e. true byte-identity, not just byte-mix-equivalence.
- All K-quant fixture tests pass in their new home in mlx-native.
- hf2q's convert orchestrator and imatrix driver continue to work unchanged from the user's perspective.
- ADR-034 (MTP+DFlash) is shipped at the time of merge.

## Open questions

- Whether `mlx_native::quantize::imatrix::*` should eventually receive the imatrix-format loader/writer pair (currently in `hf2q::quantize::imatrix::{gguf_loader, gguf_writer}`) under the same split-principle reasoning. Deferred: imatrix's coupling to the convert-time driver makes this a follow-up question, not an ADR-035 question.
- Whether `mlx_native::ops::qdq_legacy` should also migrate into `mlx_native::quantize::*` for full co-location once ADR-020 Track 1 ships and the GPU/CPU kernel-kind boundary can be re-evaluated. Tracked under ADR-020 follow-up, not here.
