//! Bounded full-logit target artifact substrate for exact-teacher quality.
//!
//! This module deliberately stops at structural target storage. A caller may
//! supply logits to the structural writer, but that result cannot authorize
//! sensitivity, allocation, or `--quant auto`. Only a future family-owned,
//! completed source-precision runner may wrap this artifact in execution
//! authority.

mod target_artifact;
mod types;

pub(crate) use target_artifact::{
    preflight_structural_teacher_target, write_structural_teacher_target_artifact,
};
pub use types::*;

#[cfg(test)]
mod tests;
