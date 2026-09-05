//! GLP (GGUF Layer Projection) runtime steering — ADR-053.
//!
//! A GLP vector is a few hundred KB of per-layer directions applied to the
//! post-layer residual stream at inference time, without touching weights:
//!
//! - `project` mode: `h ← h − α(h·d̂)d̂` (delete the component along the direction)
//! - `add` mode:     `h ← h + α·v`     (llama.cpp control-vector convention)
//!
//! Container and conformance contract: msuiche/weightless `spec/GLP.md`.
//! Fail-closed everywhere the spec demands: unknown mode, unknown hook point,
//! `direction.0`, mismatched hook, or a `project` merge attempt are fatal.
//!
//! Scope of this module: loading + conformance + the arithmetic of applying
//! one vector to a layer-local activation tensor. Family wiring (where in
//! each forward graph the hook runs) lives in the family modules.

pub mod apply;
pub mod apply_gpu;
pub mod bind;
pub mod discovery;
pub mod reader;

pub use apply::{apply_layer_add, apply_layer_project};
pub use apply_gpu::{
    apply_layer_gpu, apply_layer_gpu_in_session, apply_layer_gpu_mhc, apply_layer_gpu_mhc_in_session,
};
pub use bind::BoundGlp;
pub use discovery::{ResolvedGlp, auto_discover_glp, validate_glp_provenance};
pub use reader::{GlpError, GlpMode, GlpVector};
