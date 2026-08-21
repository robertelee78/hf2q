//! Bounded full-logit target artifact substrate for exact-teacher quality.
//!
//! This module deliberately stops at structural target storage. A caller may
//! supply logits to the structural writer, but that result cannot authorize
//! sensitivity, allocation, or `--quant auto`. Only the family-owned,
//! completed source-precision runner may wrap this artifact in its bounded
//! process-local execution authority.

mod reference;
mod target_artifact;
mod types;

pub(crate) use reference::{
    build_exact_teacher_reference_input, compare_exact_teacher_reference_targets,
    validate_canonical_exact_teacher_reference_comparison_receipt,
    validate_exact_teacher_reference_comparison_artifact,
    validate_exact_teacher_reference_input, ExactTeacherExternalReferenceEvidenceV1,
    ExactTeacherReferenceComparisonReceiptV1, ExactTeacherReferenceInputV1,
    ExactTeacherReferenceTrajectoryComparisonV1,
};

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
