# ADR-035: GGUF codec ownership — move encode side into mlx-native

**Status**: Proposed (pending ADR-033 §P1 byte-equivalence closure)
**Date**: 2026-05-19
**Supersedes**: nothing
**Related**: ADR-008 (mlx-native as compute backend), ADR-033 (unified quant/convert pipeline)

## Context

The GGUF format codec is currently split across two crates:

| Component                              | Current home                                                | Natural pair                           |
|----------------------------------------|-------------------------------------------------------------|----------------------------------------|
| GGUF **reader** (header + tensor parse) | `mlx_native::gguf::GgufFile`                                | writer                                 |
| GGUF **writer** (header + tensor emit)  | `hf2q::backends::gguf::writer::GgufWriter`                  | reader                                 |
| CPU **dequant** (Q4_0..Q6_K, IQ4_NL)    | `mlx_native::gguf::*` and `mlx_native::ops::qdq_legacy`     | quantize                               |
| CPU **quantize** (Q2_K..Q8_0, IQ4_NL)   | `hf2q::quantize::ggml_quants::*`                            | dequant                                |
| Metal dequant / quantized-matmul        | `mlx_native::ops::*` and `mlx_native::shaders::*`           | CPU encode/decode (same canonical ref) |
| imatrix loader                          | `hf2q::quantize::imatrix::*` (uses `mlx_native::gguf`)      | quantize                               |

mlx-native already owns half the GGUF codec (reader + CPU dequant + Metal dequant). hf2q owns the other half (writer + CPU quantize). The split is historical: hf2q was the convert tool that grew quantize code in-place; mlx-native is the runtime library that grew GGUF parsing because it consumes the format. Neither side intentionally claimed ownership of the codec as a whole.

The asymmetry creates real friction:
- Any future tool that wants to write GGUF (e.g. a standalone imatrix-only quantizer, a checkpoint-shard merger, a vendored-quant exporter) must depend on `hf2q` — a 100-file convert pipeline — to get at the encode kernels.
- The canonical reference for each kernel (`/opt/llama.cpp/ggml/src/ggml-quants.c`) is one file containing matched `quantize_row_qX` and `dequantize_row_qX` pairs. Splitting them across crates breaks the natural co-location.
- Cross-crate refactors (e.g. adding a new ggml_type) require coordinated changes to both crates instead of one.

## Decision

After ADR-033 §P1 (hf2q convert producing byte-identical GGUFs to `convert_hf_to_gguf.py | llama-quantize`) closes, move the GGUF encode side into mlx-native, giving mlx-native end-to-end ownership of the GGUF codec:

1. Move `hf2q::backends::gguf::writer::GgufWriter` → `mlx_native::gguf::writer::GgufWriter`.
2. Move `hf2q::quantize::ggml_quants::*` (Q2_K..Q8_0, IQ4_NL, common.rs, quantizer trait, ggml_type, llama_ftype, vision, apex/, standard_policy.rs) → `mlx_native::quantize::*`.
3. Keep in hf2q: `convert/` (orchestrator, source_reader, arch/), `quantize/imatrix/` (driver code that *uses* the kernels), CLI bindings.
4. Re-export the moved types from `mlx_native::quantize` so external consumers (hf2q itself, third-party tools) have a single stable import surface.
5. Migrate tests in lockstep: kernel-level fixture tests + byte-cmp tests follow the kernels into mlx-native; convert-orchestrator tests stay in hf2q.

## Why "after §P1 closes"

Doing the move concurrently with the FMA + F16-roundtrip correctness work would mix code motion with semantic changes, making review impossible and making git bisect across the move boundary painful. Byte-equivalence must be locked down in the current structure first; then the move is a pure refactor that should be verifiable via the same byte-cmp test from `/opt/hf2q/scripts/byte_cmp_gguf.py` and the kernel fixture tests.

## Consequences

**Positive**:
- One crate owns the full GGUF codec; canonical-reference fidelity is enforced in one place.
- New tools writing GGUF depend on a thin `mlx-native` library, not the full hf2q app.
- Reader/writer and quantize/dequant pairs co-locate, matching the canonical `ggml-quants.c` source layout.
- Future cross-crate work on the format (e.g. Q8_K matmul, new K-variants) is single-crate.

**Negative**:
- One-shot churn: ~16k LOC moves crates. Imports change everywhere.
- mlx-native crate-size grows; build times rise marginally.
- Public API of mlx-native expands: `mlx_native::quantize::*` becomes a stability commitment.

**Mitigations**:
- The move is mechanical: paths change, code does not. Byte-cmp on hf2q convert output before and after the move proves zero semantic regression.
- Crate-size impact is one-time; the kernels were already going to be linked into the hf2q binary anyway, just transitively.

## Acceptance Criteria

- All `src/quantize/ggml_quants/` paths in hf2q resolve to `mlx_native::quantize::ggml_quants::*` imports.
- `/opt/hf2q/scripts/byte_cmp_gguf.py` on a freshly-converted Q4_K_M model reports zero diffs (same as the pre-move §P1-closed result).
- All K-quant fixture tests pass in their new home in mlx-native.
- hf2q's convert orchestrator and imatrix driver continue to work unchanged from the user's perspective.

## Open questions

- Naming under mlx-native: `mlx_native::quantize::*` vs `mlx_native::gguf::quantize::*`? The latter is more discoverable (lives next to the reader/writer) but verbose.
- Whether `qdq_legacy.rs` (existing Q4_0/Q8_0 oracle in mlx-native) should be retired in favor of the now-co-located production quantizers, or kept as a self-contained parity oracle.
- imatrix path lives in hf2q today; do we also move `quantize::imatrix::*` to mlx-native, or keep it in hf2q as a convert-time driver that calls into mlx-native's kernels?
