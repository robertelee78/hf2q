//! Source- and execution-bound Dynamic-style policy proposal substrate.
//!
//! This module enumerates a bounded exact Pareto frontier from comparable
//! per-unit sensitivity and Apple operation-cost evidence. It does not produce
//! calibration data, materialize artifacts, or make a candidate eligible. A
//! proposal must still pass full-model quality, behavior, serving, and matched
//! hardware gates in [`super::measured_auto_quant`].

mod solver;
mod types;

pub use solver::{
    allocate_dynamic_frontier, allocation_problem_sha256, canonical_frontier_bytes,
    canonical_policy_bytes, final_executed_tensor_bundle_sha256, precision_policy_sha256,
    tensor_catalog_sha256, validate_policy_frontier, DynamicAllocationError,
};
pub use types::*;

#[cfg(test)]
mod tests;
