//! Vision / audio tensor pattern gates.
//!
//! Per ADR-033 Decision §"Vision / audio tensor patterns" — these are the
//! **only** places modality-pattern membership is decided. The convert
//! dispatcher checks `is_vision_tensor_pattern(name) || is_audio_tensor_pattern(name)`
//! BEFORE calling `QuantPolicy::target_for`; tensors that match are
//! emitted F16 directly (modality-side), unmatched go through policy.
//!
//! `is_vision_tensor_pattern` is ported verbatim from
//! `src/quantize/layer_mix.rs:366` at HEAD. The audio sibling is new per
//! P-1 audit finding E (the inline filter at `src/backends/gguf.rs:322-333`
//! had an `audio_tower.` substring that the canonical layer_mix fn did
//! not — this module consolidates both into a single pair of fns).
//!
//! P6 deletes the layer_mix copy. The convert pipeline routes through
//! this module exclusively.
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: this is new code, not
//! a wrapper around the layer_mix copy. Once P6 deletes layer_mix the
//! two will be one fn, but at P2 they coexist as separate sources of
//! truth keyed to their respective callers.

/// Returns `true` iff the GGUF tensor name belongs to a vision tower /
/// projector / patch-embd path.
///
/// Substrings checked (in source order — `contains` returns on first
/// match):
/// 1. `model.visual.`     — Qwen3-VL `model.visual.*` (post-HF rename)
/// 2. `vision_tower.`     — Llama-3-Vision / Mllama style
/// 3. `vision_model.`     — CLIP / SigLIP style
/// 4. `vit.`              — generic ViT prefix (some custom converters)
/// 5. `visual.` (prefix)  — bare `visual.*` (Qwen3-VL pre-HF rename)
/// 6. `.visual.`          — namespaced `<arch>.visual.<rest>`
#[inline]
pub fn is_vision_tensor_pattern(tensor_name: &str) -> bool {
    tensor_name.contains("model.visual.")
        || tensor_name.contains("vision_tower.")
        || tensor_name.contains("vision_model.")
        || tensor_name.contains("vit.")
        || tensor_name.starts_with("visual.")
        || tensor_name.contains(".visual.")
        // Per codex 0d28ae3f review: the old inline filter at
        // `src/backends/gguf.rs:339` caught `embed_vision` with NO
        // trailing dot — e.g. Gemma's `model.embed_vision.weight`.
        // Adding here as the canonical source.
        || tensor_name.contains("embed_vision")
}

/// Returns `true` iff the GGUF tensor name belongs to an audio tower /
/// whisper-style encoder path.
///
/// Substrings checked (in source order):
/// 1. `audio_tower`   — HF multi-modal audio adapters (Gemma audio etc.).
///    NO trailing dot — matches the old inline filter at
///    `src/backends/gguf.rs:339` which used `contains("audio_tower")`
///    and caught e.g. `model.audio_tower_proj.weight`. Codex 0d28ae3f
///    review flagged the dot-bearing form as too strict.
/// 2. `audio_model.`  — mirror of `vision_model.` for audio-side towers.
///    Trailing dot kept; no prior canonical source — NEW per ADR.
/// 3. `whisper.`      — Whisper-derived encoders. Trailing dot kept to
///    avoid false-positives on model-name substrings like
///    `whispering`. NEW per ADR (no prior canonical source).
///
/// Per ADR-033 amendment E: NEW companion to `is_vision_tensor_pattern`.
#[inline]
pub fn is_audio_tensor_pattern(tensor_name: &str) -> bool {
    tensor_name.contains("audio_tower")
        || tensor_name.contains("audio_model.")
        || tensor_name.contains("whisper.")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- is_vision_tensor_pattern: 7 substrings × positive/negative ----

    #[test]
    fn vision_matches_model_visual_dot() {
        assert!(is_vision_tensor_pattern("model.visual.patch_embd.weight"));
        assert!(is_vision_tensor_pattern("model.visual.blocks.0.attn_q.weight"));
    }

    #[test]
    fn vision_matches_vision_tower_dot() {
        assert!(is_vision_tensor_pattern("vision_tower.vision_model.embeddings"));
        assert!(is_vision_tensor_pattern("language_model.vision_tower.patch_embd"));
    }

    #[test]
    fn vision_matches_vision_model_dot() {
        assert!(is_vision_tensor_pattern("vision_model.encoder.layers.0.attn.q"));
    }

    #[test]
    fn vision_matches_vit_dot() {
        assert!(is_vision_tensor_pattern("vit.patch_embd.weight"));
        assert!(is_vision_tensor_pattern("model.vit.blocks.0.norm1.weight"));
    }

    #[test]
    fn vision_matches_visual_prefix() {
        assert!(is_vision_tensor_pattern("visual.patch_embed.weight"));
        assert!(is_vision_tensor_pattern("visual.blocks.0.attn.q.weight"));
    }

    #[test]
    fn vision_matches_dot_visual_dot() {
        assert!(is_vision_tensor_pattern("qwen3vl.visual.norm.weight"));
        assert!(is_vision_tensor_pattern("clip.visual.embeddings"));
    }

    #[test]
    fn vision_does_not_match_text_paths() {
        assert!(!is_vision_tensor_pattern("blk.0.attn_q.weight"));
        assert!(!is_vision_tensor_pattern("token_embd.weight"));
        assert!(!is_vision_tensor_pattern("output_norm.weight"));
        // "visual" with no dot anywhere — must NOT match (no substring rule fires)
        assert!(!is_vision_tensor_pattern("visualization_weight"));
        // "vit" substring alone (no trailing dot) — substring rule needs `vit.`
        assert!(!is_vision_tensor_pattern("blk.0.vitamin.weight"));
    }

    /// Per codex 0d28ae3f review: the old inline filter at
    /// `src/backends/gguf.rs:339` caught `embed_vision` (no dot) for
    /// Gemma's `model.embed_vision.weight`. Lock the fix in.
    #[test]
    fn vision_matches_embed_vision_codex_0d28ae3f() {
        assert!(is_vision_tensor_pattern("model.embed_vision.weight"));
        assert!(is_vision_tensor_pattern("model.embed_vision.norm.weight"));
    }

    // ---- is_audio_tensor_pattern: 3 substrings × positive/negative ----

    #[test]
    fn audio_matches_audio_tower_dot() {
        assert!(is_audio_tensor_pattern("audio_tower.encoder.layers.0.attn"));
        assert!(is_audio_tensor_pattern("model.audio_tower.proj.weight"));
    }

    /// Per codex 0d28ae3f review: the old inline filter used
    /// `contains("audio_tower")` (no dot) and caught e.g.
    /// `model.audio_tower_proj.weight`. Lock the fix in.
    #[test]
    fn audio_matches_audio_tower_no_dot_codex_0d28ae3f() {
        assert!(is_audio_tensor_pattern("model.audio_tower_proj.weight"));
        assert!(is_audio_tensor_pattern("audio_tower_v2.encoder"));
    }

    #[test]
    fn audio_matches_audio_model_dot() {
        assert!(is_audio_tensor_pattern("audio_model.embeddings.weight"));
    }

    #[test]
    fn audio_matches_whisper_dot() {
        assert!(is_audio_tensor_pattern("whisper.encoder.layers.0.attn"));
        assert!(is_audio_tensor_pattern("model.audio_tower.whisper.proj"));
    }

    #[test]
    fn audio_does_not_match_text_paths() {
        assert!(!is_audio_tensor_pattern("blk.0.attn_q.weight"));
        assert!(!is_audio_tensor_pattern("token_embd.weight"));
        // bare "audio" without a trailing dot — substring rule needs `audio_tower.` etc.
        assert!(!is_audio_tensor_pattern("blk.0.audio_weight"));
        // "whispering" — must NOT match (no `whisper.`)
        assert!(!is_audio_tensor_pattern("blk.0.whispering.weight"));
    }

    // ---- combined dispatcher gate ----

    #[test]
    fn combined_gate_for_dispatcher() {
        // Mirrors the dispatcher call:
        // `is_vision_tensor_pattern(name) || is_audio_tensor_pattern(name)`
        let go_through_policy = |name: &str| {
            !(is_vision_tensor_pattern(name) || is_audio_tensor_pattern(name))
        };
        assert!(go_through_policy("blk.0.attn_q.weight"));
        assert!(!go_through_policy("vision_tower.encoder.q"));
        assert!(!go_through_policy("audio_tower.encoder.q"));
        assert!(!go_through_policy("whisper.encoder.q"));
        assert!(!go_through_policy("model.visual.patch_embd"));
    }
}
