//! Structured, source-bound calibration inputs for learned quantization.
//!
//! This module does not compute sensitivity. It freezes disjoint structured
//! inputs, renders them through the production chat path, tokenizes them, and
//! produces enough evidence for a later collector to fail closed.

mod corpus;
mod manifest;
mod partition;
mod prediction_plan;
mod render;
mod types;

#[cfg(test)]
pub(crate) use corpus::verify_calibration_corpus_artifact;
pub(crate) use corpus::verify_embedded_calibration_corpus_artifact;
pub use manifest::{build_structured_dataset_manifest, validate_structured_dataset_manifest};
pub use partition::verify_dataset_partition;
pub(crate) use prediction_plan::build_teacher_prediction_plan;
pub use prediction_plan::validate_teacher_prediction_plan;
#[cfg(test)]
pub(crate) use prediction_plan::{
    prediction_plan_for_test, prediction_plan_for_test_bound,
    prediction_plan_for_test_bound_with_first_prefix, prediction_plan_for_test_bound_with_gap,
};
pub(crate) use render::render_and_tokenize_verified_split;
pub use render::{render_and_tokenize_split, verify_rendered_dataset_from_source};
pub use types::*;

#[cfg(test)]
mod tests;
