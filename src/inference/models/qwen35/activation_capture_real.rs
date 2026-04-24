//! Real (non-mock) `ActivationCapture` implementation for qwen35 / qwen35moe.
//!
//! ADR-012 P9 Decision 17. Replaces the `NoActivationCapture` guard in
//! `src/main.rs` once ADR-013 P12's weight loader + F16 forward land
//! bit-stable. Until P12 ships, this module exposes the `RealActivationCapture`
//! type with a `not_ready()` constructor that surfaces a structured
//! dependency error — NOT a silent stub. Callers choose between:
//!
//!   - `RealActivationCapture::new(...)` — attempts real forward (P12+).
//!   - `RealActivationCapture::not_ready()` — cites the ADR-013 P12
//!     blocker explicitly so the guard in `main.rs` can degrade
//!     deterministically and users see the actionable error.
//!
//! Once ADR-013 P12 is green (load + forward both bit-stable), the
//! `new(...)` body is filled in to drive the forward pass and capture
//! per-nn.Linear activations at hook points P12 exposes. The mantra
//! applies: "No stub, no fallback" — the `NotReady` variant is the
//! load-bearing error that prevents silent dissolve into weight-space.

use anyhow::Result;
use thiserror::Error;

use super::activation_capture::{ActivationCapture, LayerActivations};

/// Errors returned by `RealActivationCapture::run_calibration_prompt`.
///
/// `NotReady` is emitted pre-P12. When ADR-013 P12 ships, the
/// constructor either returns Ok(Self) and `run_calibration_prompt`
/// delivers real activations, OR it returns a precise `ForwardPass`
/// error naming the specific layer / tensor that failed — never
/// `NotReady`.
#[derive(Debug, Error)]
pub enum RealActivationCaptureError {
    /// ADR-013 P12 (weight loader + F16 forward) has not yet landed
    /// bit-stable on `main`. Returned by the pre-P12 shim constructor
    /// `not_ready()`. Callers must check this and decline to run the
    /// DWQ activation path — the correct action is to finish P12, not
    /// fall back to weight-space (see `feedback_never_ship_fallback_without_rootcause.md`).
    #[error(
        "ADR-013 P12 not ready: RealActivationCapture cannot drive the \
         qwen35 forward pass until the weight loader + F16 forward are \
         bit-stable. Finish ADR-013 P12 before invoking DWQ calibration \
         on qwen35 / qwen35moe. No fallback is provided by design."
    )]
    NotReady,

    /// Forward pass failed with a specific root cause. Reserved for
    /// post-P12 callers when the captured activations do not reach
    /// the expected shape; the `ForwardPass` variant must name the
    /// layer / tensor involved so the bisection point is obvious.
    #[error("qwen35 forward failed at layer {layer}: {reason}")]
    ForwardPass { layer: u32, reason: String },
}

/// Production ActivationCapture. See module doc for lifecycle.
#[derive(Debug)]
pub struct RealActivationCapture {
    /// Dependency status — flipped to `true` when ADR-013 P12 is
    /// wired; pre-P12 callers construct via `not_ready()` which sets
    /// this `false` and returns `NotReady` from every `run_*` call.
    _p12_ready: bool,
}

impl RealActivationCapture {
    /// Construct a NotReady shim — pre-ADR-013-P12 callers use this so
    /// the DWQ guard surfaces the dependency explicitly rather than
    /// silently falling back. This is the load-bearing piece that
    /// prevents fallback without root cause.
    pub fn not_ready() -> Self {
        Self { _p12_ready: false }
    }

    /// Construct for real. Post-P12 implementation is added in P9
    /// alongside the `main.rs:488-506` rewrite. Returns
    /// `RealActivationCaptureError::NotReady` pre-P12.
    ///
    /// NOTE: the `_model_gguf` and `_tokenizer` parameters pin the
    /// future signature so the `new()` call site in `main.rs` already
    /// passes what ADR-013 P12's `Qwen35Model::load_from_gguf` needs.
    pub fn new(
        _model_gguf: &std::path::Path,
        _tokenizer_json: &std::path::Path,
    ) -> std::result::Result<Self, RealActivationCaptureError> {
        Err(RealActivationCaptureError::NotReady)
    }
}

impl ActivationCapture for RealActivationCapture {
    fn run_calibration_prompt(
        &mut self,
        _tokens: &[u32],
    ) -> Result<LayerActivations> {
        if !self._p12_ready {
            return Err(anyhow::anyhow!(RealActivationCaptureError::NotReady));
        }
        // Post-P12 path (unreachable today; filled in when P12 ships).
        unreachable!("RealActivationCapture forward path lands with ADR-013 P12");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_ready_shim_returns_not_ready_error() {
        let mut cap = RealActivationCapture::not_ready();
        let err = cap.run_calibration_prompt(&[1, 2, 3]).unwrap_err();
        let s = format!("{}", err);
        assert!(
            s.contains("ADR-013 P12 not ready"),
            "error message must cite ADR-013 P12, got: {}",
            s
        );
        assert!(s.contains("No fallback is provided by design"),);
    }

    #[test]
    fn new_returns_not_ready_pre_p12() {
        let err = RealActivationCapture::new(
            std::path::Path::new("/tmp/nonexistent.gguf"),
            std::path::Path::new("/tmp/tokenizer.json"),
        )
        .unwrap_err();
        assert!(matches!(err, RealActivationCaptureError::NotReady));
    }

    #[test]
    fn error_display_names_the_blocking_adr() {
        let err = RealActivationCaptureError::NotReady;
        let s = format!("{}", err);
        assert!(s.contains("ADR-013 P12"));
        assert!(s.contains("no fallback") || s.contains("No fallback"));
    }

    #[test]
    fn forward_pass_error_carries_layer_and_reason() {
        let err = RealActivationCaptureError::ForwardPass {
            layer: 7,
            reason: "attn_qkv shape mismatch".into(),
        };
        let s = format!("{}", err);
        assert!(s.contains("layer 7"));
        assert!(s.contains("attn_qkv shape mismatch"));
    }
}
