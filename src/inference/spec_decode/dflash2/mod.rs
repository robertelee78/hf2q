//! Family-neutral DFlash2 speculative-decoding core (ADR-067).
//!
//! Family adapters bind target hidden capture, verifier state, rollback and
//! checkpoint identity. This module owns the shared checkpoint schema and draft
//! operations; it never selects an approximately compatible target family.

pub mod adapters;
pub mod config;
pub mod forward;
pub mod schema;
pub mod runtime;
pub mod selector;
pub mod tensors;
pub mod weights;
pub mod workspace;
