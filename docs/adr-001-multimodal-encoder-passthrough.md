# ADR-001: Multimodal Encoder Passthrough in MLX Backend

**Status:** Proposed
**Date:** 2026-04-07
**Authors:** Winston (Architect), Robert
**Deciders:** Robert

## Context

hf2q's MLX backend currently hardcodes two tensor name patterns as "vision passthrough" — weights matching `vision_tower` or `embed_vision` are preserved at original precision (bf16) instead of being quantized. All other 2D+ tensors go through affine quantization.

This works for the current Gemma4 26B-A4B model (vision + language only), but the Gemma4 family includes E2B and E4B variants that ship with a ~300M parameter **audio encoder** (`Gemma4AudioConfig`). Future models may add video encoders or other modality-specific components.

If a user converts an audio-capable model (e.g., `google/gemma-4-4b-it`) with hf2q today, the audio encoder weights would be silently quantized to 4-bit. Encoder weights are perceptual feature extractors — aggressive quantization degrades them disproportionately compared to language model weights. This would produce a model that loads without error but generates poor audio understanding, with no indication that quantization caused the problem.

### Models Affected

| Model | Modalities | Encoder Components |
|-------|-----------|-------------------|
| Gemma4 26B-A4B | Vision + Text | `vision_tower`, `embed_vision` |
| Gemma4 E4B | Vision + Audio + Text | `vision_tower`, `embed_vision`, `audio_tower`*, `embed_audio`* |
| Gemma4 E2B | Vision + Audio + Text | `vision_tower`, `embed_vision`, `audio_tower`*, `embed_audio`* |

*Exact tensor name prefixes TBD — need to verify against actual E2B/E4B weight maps once accessible (gated models).

## Decision

**Explicitly add audio encoder patterns alongside the existing vision passthrough.** Keep the same explicit-enumeration approach used for vision — add `audio_tower`, `embed_audio`, and `audio_encoder` as named patterns. No generic pattern matching; we add modalities as we verify them.

This is deliberately conservative. There are too many unknowns (exact tensor names in gated E2B/E4B checkpoints, potential audio encoder substructures) to justify a generic approach. We can generalize later once we have more data points.

### Audio Passthrough Patterns

Added to the existing vision check:
- `audio_tower` — the audio encoder tower (analogous to `vision_tower`)
- `embed_audio` — audio embedding projection (analogous to `embed_vision`)
- `audio_encoder` — alternate naming for the audio encoder component

These tensors are preserved at original precision (bf16), same as vision.

### Config Handling

- `audio_config` is already passed through via `raw_config` — no changes needed
- The quantization override block in the output `config.json` excludes audio encoder tensors (same as vision)

### Implementation

**File:** `src/backends/mlx.rs` — two locations changed:

1. Quantization decision (tensor write loop):
```rust
let is_vision = name.contains("vision_tower") || name.contains("embed_vision");
let is_audio = name.contains("audio_tower") || name.contains("embed_audio") || name.contains("audio_encoder");
let should_quantize = is_2d_plus && !is_norm_or_scalar && !is_vision && !is_audio;
```

2. Config override exclusion:
```rust
if tensor_name.contains("vision_tower") || tensor_name.contains("embed_vision")
    || tensor_name.contains("audio_tower") || tensor_name.contains("embed_audio")
    || tensor_name.contains("audio_encoder") {
    continue;
}
```

**No other files changed.** Config parsing and auto-quant memory estimation are left as follow-up work once we can verify against actual E2B/E4B checkpoints.

## Consequences

### Positive

- E2B/E4B audio models convert correctly without silent quality loss
- Explicit patterns are easy to audit — no hidden matching logic
- No user-facing flag or config change required — it just works
- Encoder precision is the safe default; users can opt into encoder quantization later if desired
- Zero risk to existing vision-only conversions (additive change only)

### Negative

- Cannot verify against actual E2B/E4B weight maps until model access is available (gated)
- Requires a code change per new modality (acceptable trade-off for safety)
- Memory estimation in auto-quant won't account for audio encoder size until that code is updated

### Risks

- **Unknown tensor naming:** The exact weight prefixes for audio in E2B/E4B need verification. The HuggingFace transformers code uses `audio_tower` in the config class, but the safetensors keys may differ. Mitigation: verify against actual checkpoint before merging.
- **MoE expert split for audio:** If audio encoders have any MoE-like structure, the expert splitting logic would need extension. Unlikely for a ~300M encoder, but worth checking.

## Alternatives Considered

### 1. Generic pattern-based detection
Match `*_tower`, `embed_*` (excluding `embed_tokens`), `*_encoder` (excluding `language_model.*`) to automatically handle any future modality.

**Deferred:** Too many unknowns. Risk of false positives on tensor names we haven't seen. Revisit when we have 3+ modalities to support.

### 2. Config-driven detection
Read `audio_config`, `vision_config`, etc. from the parsed config to determine which encoder components exist, then skip quantization for matching prefixes.

**Deferred:** More principled but requires config parsing changes. Good follow-up for cross-validation (warn if `audio_config` is non-null but zero audio tensors found in output).

### 3. User flag: `--preserve-encoders`
Let users explicitly specify which components to preserve.

**Deferred:** Good power-user escape hatch but shouldn't be required for correctness.

## Validation Plan

1. Download `google/gemma-4-4b-it` (requires HF access approval)
2. Inspect `model.safetensors.index.json` to confirm audio tensor name prefixes
3. Run `hf2q convert` and verify audio encoder weights appear unquantized in output
4. Compare output file sizes: audio encoder bytes should match original bf16 size
5. Test with `mlx-vlm` or equivalent to confirm audio inference works post-conversion
