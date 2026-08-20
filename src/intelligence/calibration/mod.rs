//! Structured, source-bound calibration inputs for learned quantization.
//!
//! This module does not compute sensitivity. It freezes disjoint structured
//! inputs, renders them through the production chat path, tokenizes them, and
//! produces enough evidence for a later collector to fail closed.

mod manifest;
mod partition;
mod render;
mod types;

pub use manifest::{build_structured_dataset_manifest, validate_structured_dataset_manifest};
pub use partition::verify_dataset_partition;
pub use render::{render_and_tokenize_split, verify_rendered_dataset_from_source};
pub use types::*;

#[cfg(test)]
mod tests;
