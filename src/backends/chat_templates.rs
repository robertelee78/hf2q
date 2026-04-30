//! Convert-time chat-template auto-inject (ADR-012 follow-up, 2026-04-30).
//!
//! # Why this module exists
//!
//! `tokenizer.chat_template` is a GGUF metadata key consulted by
//! [`crate::serve::render_chat_template`] (priority 3, between `--chat-template`
//! CLI overrides and the hardcoded Gemma4 fallback). When the source HF model
//! ships no `chat_template` field — e.g. all 4 abliterated Qwen3.6
//! variants on disk as of 2026-04-30 — the convert path at
//! [`crate::backends::gguf::load_tokenizer_metadata`] used to silently skip
//! the emit, leaving the runtime to fall through to the Gemma4 hardcoded
//! string. Qwen3 was never trained on Gemma4 control tokens, so the
//! resulting chat session produced gibberish and contaminated ADR-013's
//! sourdough byte-parity gate.
//!
//! This module owns the *vendor-shipped* per-architecture default
//! templates. Each fixture under `chat_templates/*.jinja` is copied
//! verbatim from a known vendor release of a model in that architecture
//! family (see `[VENDOR PROVENANCE]` annotations on each constant). We
//! never invent a template; if no vendor reference exists for an arch,
//! we return `None` and the caller logs a structured WARN.
//!
//! # Priority chain (mirrored in `gguf.rs::load_tokenizer_metadata`)
//!
//! 1. `chat_template.jinja` file alongside the HF tokenizer
//! 2. `tokenizer_config.json[chat_template]`
//! 3. **arch-default from this module** (NEW)
//! 4. Graceful skip + WARN (operator-visible)
//!
//! [VENDOR PROVENANCE — qwen3-chatml.jinja]
//!   Source file: `/opt/hf2q/models/qwen3.6-27b-dwq46/tokenizer_config.json`
//!   Field: `chat_template`
//!   Length: exactly 7764 bytes (asserted in [`QWEN3_CHATML_LEN`] +
//!   in tests via [`vendor_chat_template_lengths_match_fixtures`]).
//!   Captured: 2026-04-30 by ADR-012 chat-template-auto-inject CFA session.

/// Vendor-shipped Qwen3 ChatML chat template, served verbatim from
/// `qwen3.6-27b-dwq46/tokenizer_config.json`.
///
/// Used by `qwen35` and `qwen35moe` when the source HF dir omits a
/// chat_template (e.g. all Qwen3.6 abliterated variants).
pub const QWEN3_CHATML: &str = include_str!("chat_templates/qwen3-chatml.jinja");

/// Compile-time-known length of [`QWEN3_CHATML`]. The fixture's length
/// must match the vendor's exactly — drift means someone trimmed or
/// re-encoded the template, and the byte-identical guarantee fails.
pub const QWEN3_CHATML_LEN: usize = 7764;

/// Look up the vendor-shipped chat template for `arch`.
///
/// Returns `None` for arches where we have no vendor-shipped reference
/// to embed. The convert path then logs a structured WARN and skips
/// emit (graceful degradation — the runtime falls through to the
/// hardcoded Gemma4 fallback as before, and the operator sees a
/// log line rather than silent gibberish at chat time).
///
/// # Arch coverage (2026-04-30)
///
/// | arch       | source                 | status     |
/// |------------|------------------------|------------|
/// | qwen35     | qwen3.6-27b-dwq46      | EMBEDDED   |
/// | qwen35moe  | qwen3.6-27b-dwq46      | EMBEDDED   |
/// | qwen2      | (no vendor ref yet)    | WARN-only  |
/// | qwen3      | (no vendor ref yet)    | WARN-only  |
/// | gemma3     | (no vendor ref yet)    | WARN-only  |
/// | gemma4     | (no vendor ref yet)    | WARN-only  |
/// | llama      | (no vendor ref yet)    | WARN-only  |
/// | mistral    | (no vendor ref yet)    | WARN-only  |
/// | phi        | (no vendor ref yet)    | WARN-only  |
///
/// New arch entries MUST be sourced from a published vendor model's
/// `tokenizer_config.json` and copied verbatim — never synthesized.
pub fn arch_default_chat_template(arch: &str) -> Option<&'static str> {
    match arch {
        "qwen35" | "qwen35moe" => Some(QWEN3_CHATML),
        // Other arches: research pending — see arch-coverage table above.
        // Until we capture a vendor reference, we return None and let
        // the caller WARN. No synthesized templates per
        // `feedback_prove_in_code.md` + `feedback_dont_guess.md`.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture's byte length must match the vendor's exactly. If
    /// this drifts, someone re-encoded line endings, trimmed
    /// whitespace, or otherwise altered the template — at which point
    /// we are no longer shipping the vendor's bytes verbatim and the
    /// "ChatML correctness" guarantee fails.
    #[test]
    fn vendor_chat_template_lengths_match_fixtures() {
        assert_eq!(
            QWEN3_CHATML.len(),
            QWEN3_CHATML_LEN,
            "Qwen3 ChatML fixture drifted from vendor: expected {} bytes, fixture has {}",
            QWEN3_CHATML_LEN,
            QWEN3_CHATML.len()
        );
    }

    #[test]
    fn arch_default_qwen35_resolves_to_qwen3_chatml() {
        assert_eq!(arch_default_chat_template("qwen35"), Some(QWEN3_CHATML));
        assert_eq!(arch_default_chat_template("qwen35moe"), Some(QWEN3_CHATML));
    }

    #[test]
    fn arch_default_unknown_arch_returns_none() {
        // Operator-visible WARN path — see `gguf.rs::load_tokenizer_metadata`.
        assert_eq!(arch_default_chat_template("unknown"), None);
        assert_eq!(arch_default_chat_template("qwen2"), None);
        assert_eq!(arch_default_chat_template("gemma4"), None);
        assert_eq!(arch_default_chat_template("llama"), None);
    }
}
