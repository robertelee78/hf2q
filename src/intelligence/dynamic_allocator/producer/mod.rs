//! Source inventory, tensor partition, and structural activation-coverage
//! contracts for Dynamic-style proposal inputs.
//!
//! This module deliberately stops before sensitivity production. It proves
//! which verified source tensors are variable or protected and lets a later
//! collector fail closed against an explicitly supplied operation/tap topology.
//! D1 does not authenticate activation arrays; a family-owned collector and
//! materialization verifier are required before observations become measured
//! coverage evidence.

#[cfg(test)]
mod bindings;
mod coverage;
mod partition;
mod types;

#[cfg(test)]
pub use bindings::{
    validate_structural_dynamic_allocation_bindings, StructurallyBoundDynamicAllocationProblem,
};
pub use coverage::{
    build_coverage_contract, validate_coverage_contract, verify_collector_topology,
    verify_coverage_receipt,
};
pub use partition::{
    build_tensor_partition, derive_source_tensor_inventory, validate_source_tensor_inventory,
    validate_tensor_partition, verify_source_tensor_inventory_from_source,
};
pub use types::*;

#[cfg(test)]
mod tests;
