//! Source- and execution-bound Dynamic-style policy proposal substrate.
//!
//! This module enumerates a bounded exact Pareto frontier from comparable
//! per-unit sensitivity and Apple operation-cost evidence. It does not produce
//! calibration data, materialize artifacts, or make a candidate eligible. A
//! proposal must still pass full-model quality, behavior, serving, and matched
//! hardware gates in [`super::measured_auto_quant`].

mod solver;
mod types;

pub mod producer;

#[cfg(test)]
use solver::{allocate_dynamic_frontier, validate_policy_frontier};
pub use solver::{
    allocation_problem_sha256, canonical_frontier_bytes, canonical_policy_bytes,
    execution_manifest_catalog_sha256, final_executed_tensor_bundle_sha256,
    precision_policy_sha256, stored_payload_bytes, tensor_catalog_sha256, DynamicAllocationError,
};
pub use types::*;

#[cfg(test)]
mod tests;
