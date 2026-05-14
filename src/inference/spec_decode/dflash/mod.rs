//! DFlash block-diffusion speculative decode draft model (ADR-030).
//!
//! Rust port of `/opt/dflash/dflash/model_mlx.py` (582 LOC reference).
//! See `docs/ADR-030-dflash-block-diffusion-spec-decode.md` for the full
//! design.
//!
//! Phase 2 (in_progress iter-15): drafter Rust port — config, weights,
//! forward. No production wire-up yet.
//!
//! Lands behind `HF2Q_SPEC_DFLASH_PHASE=2` until Phase 3+ ships hidden-state
//! capture and the verify forward.

pub mod config;
pub mod forward;
pub mod hidden_capture;
pub mod kv_cache;
pub mod orchestrator;
pub mod tensors;
pub mod weights;
