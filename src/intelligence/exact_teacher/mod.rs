//! Bounded full-logit target artifact substrate for exact-teacher quality.
//!
//! This module deliberately stops at structural target storage. A caller may
//! supply logits to the structural writer, but that result cannot authorize
//! sensitivity, allocation, or `--quant auto`. Only the family-owned,
//! completed source-precision runner may wrap this artifact in its bounded
//! process-local execution authority.

mod target_artifact;
mod types;

#[cfg(test)]
pub(crate) use target_artifact::write_structural_teacher_target_artifact;
pub(crate) use target_artifact::{
    canonical_teacher_argmax, preflight_structural_teacher_target, StructuralTeacherTargetStream,
    UnpublishedStructuralTeacherTargetArtifact, UnpublishedStructuralTeacherTargetReservation,
};
pub use types::*;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod reservation_tests;
