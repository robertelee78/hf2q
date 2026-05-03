//! Qwen3-VL text-LM forward path **scaffolding** (iter-228a) and the
//! sentinel surface that iter-228b replaces with the actual transformer
//! forward.
//!
//! # Why a scaffolding-only iter
//!
//! iter-228a's deliverable is the load surface (config parser + GGUF
//! weight loader + engine seam). The dense transformer forward path
//! (per-layer biased GQA + 3D-mRoPE + flash attention + SiLU FFN +
//! DeepStack residual + LM head) is non-trivial mlx-native integration
//! work that, if shipped on the same iter as the load surface, would
//! conflate "did the load work?" with "did the forward work?" debugging
//! signal. iter-215 → Wedge-3 took the same approach for Qwen3.5/3.6:
//! land the load + 501 sentinel, then wire forward in a follow-up.
//!
//! # Sentinel surface
//!
//! Until iter-228b lands, every chat / streaming / embed / soft-tokens
//! request that lands on a [`crate::inference::models::qwen3vl_text::Qwen3VlTextModel`]
//! returns a sentinel-tagged error via [`qwen3vl_text_forward_pending_err`].
//! The chat handler matches on [`QWEN3VL_TEXT_FORWARD_PENDING_SENTINEL`]
//! and maps the error to HTTP 501 with an operator-actionable message
//! pointing at this iter's docstring.
//!
//! This mirrors `engine_qwen35::QWEN35_NOT_IMPLEMENTED_SENTINEL` /
//! `qwen35_not_implemented_err` byte-for-byte.

use anyhow::Result;

/// Sentinel substring embedded in every iter-228a forward-path error.
///
/// The chat handler at [`crate::serve::api::handlers`] matches on this
/// substring to dispatch to a structured HTTP 501 response. Stable
/// across iters; the value is part of the operator-facing contract.
pub const QWEN3VL_TEXT_FORWARD_PENDING_SENTINEL: &str = "qwen3vl_text_forward_pending";

/// Operator-facing message body. Surfaced verbatim in the HTTP 501
/// response body (and in `tracing::warn` lines on the worker side).
pub const QWEN3VL_TEXT_FORWARD_PENDING_MESSAGE: &str =
    "Qwen3-VL text-LM dense forward path is iter-228b scope. iter-228a closed the load surface \
     (config + weights + engine seam) so the GGUF opens cleanly through `hf2q serve` without \
     the iter-227 actionable-error bail; the per-layer transformer forward (biased GQA + \
     per-head Q/K RMSNorm + 3D-mRoPE + GQA flash-attention + SiLU FFN + DeepStack residual \
     injection + tied LM head) is the next iteration. For text-only chat today, use a \
     Qwen3.5/3.6 GGUF (full chat path) or a Gemma 4 GGUF (full chat + image path).";

/// Construct the sentinel-tagged error returned by every dispatch arm
/// that lands on [`crate::inference::models::qwen3vl_text::Qwen3VlTextModel`]
/// in iter-228a. Mirrors the role of
/// [`crate::serve::api::engine::qwen35_not_implemented_err`] for the
/// Qwen3-VL text family.
pub fn qwen3vl_text_forward_pending_err<T>() -> Result<T> {
    Err(anyhow::anyhow!(
        "{}: {}",
        QWEN3VL_TEXT_FORWARD_PENDING_SENTINEL,
        QWEN3VL_TEXT_FORWARD_PENDING_MESSAGE
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinel_is_stable_across_iters() {
        // The sentinel value is part of the operator-facing contract.
        // The chat handler matches on this substring; changing it in a
        // refactor would silently break HTTP 501 dispatch.
        assert_eq!(
            QWEN3VL_TEXT_FORWARD_PENDING_SENTINEL,
            "qwen3vl_text_forward_pending"
        );
    }

    #[test]
    fn pending_err_carries_sentinel_substring() {
        let err: Result<()> = qwen3vl_text_forward_pending_err();
        let msg = format!("{:#}", err.unwrap_err());
        assert!(
            msg.contains(QWEN3VL_TEXT_FORWARD_PENDING_SENTINEL),
            "error message must carry the sentinel substring; got: {msg}"
        );
        assert!(
            msg.contains("iter-228b"),
            "error message must point at iter-228b; got: {msg}"
        );
    }

    #[test]
    fn pending_err_message_is_operator_actionable() {
        let err: Result<()> = qwen3vl_text_forward_pending_err();
        let msg = format!("{:#}", err.unwrap_err());
        // Operator-actionable: tells them what works today.
        assert!(
            msg.contains("Qwen3.5") || msg.contains("Gemma"),
            "error message must name a working alternative; got: {msg}"
        );
        // Operator-actionable: explains scope split.
        assert!(
            msg.contains("load surface"),
            "error message must explain what iter-228a DOES ship; got: {msg}"
        );
    }
}
