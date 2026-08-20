use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::intelligence::measured_auto_quant::SourceIdentity;

use super::super::{TensorMember, TensorOperation};

pub const DYNAMIC_PRODUCER_SCHEMA_VERSION: u32 = 1;

/// One tensor read directly from a verified safetensors source snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceTensorRecord {
    pub name: String,
    /// Hugging Face/safetensors source order, before any conversion mapping.
    pub source_shape: Vec<usize>,
    pub source_dtype: String,
    pub source_byte_len: u64,
    pub source_tensor_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceTensorInventoryManifest {
    pub schema_version: u32,
    pub source: SourceIdentity,
    /// Hash of the opaque, already-verified source-file manifest used to open
    /// this snapshot. The inventory cannot be constructed from caller tensor
    /// metadata alone.
    pub verified_source_manifest_sha256: String,
    pub tensors: Vec<SourceTensorRecord>,
    pub manifest_sha256: String,
}

/// Type-state proof returned only after reading the exact verified source.
#[derive(Debug, Clone)]
pub struct VerifiedSourceTensorInventory {
    pub(super) manifest: SourceTensorInventoryManifest,
}

impl VerifiedSourceTensorInventory {
    pub fn manifest(&self) -> &SourceTensorInventoryManifest {
        &self.manifest
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonVariableDisposition {
    Fixed,
    Protected,
    Excluded,
}

/// Source-only disposition. Stored/loaded/executed representation belongs to
/// the separate tensor-execution manifest and is intentionally absent here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonVariableTensor {
    pub source: SourceTensorRecord,
    pub disposition: NonVariableDisposition,
    pub reason: String,
}

/// Canonical atomic allocation grouping, including packed-expert topology.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VariableUnitDescriptor {
    pub unit_id: String,
    pub members: Vec<TensorMember>,
    pub expected_expert_ids: Vec<u32>,
    pub operations: Vec<TensorOperation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorPartitionManifest {
    pub schema_version: u32,
    pub source: SourceIdentity,
    pub source_inventory_manifest_sha256: String,
    pub source_tensor_count: usize,
    pub variable_units: Vec<VariableUnitDescriptor>,
    pub non_variable_tensors: Vec<NonVariableTensor>,
    pub tensor_catalog_sha256: String,
    pub manifest_sha256: String,
}

/// Canonical, allocation-unit-complete collector topology. D1 verifies its
/// structure; a later family-specific producer must be the authority that
/// supplies the operation ids and graph paths.
#[derive(Debug, Clone)]
pub struct VerifiedCollectorTopology {
    pub(super) units: Vec<UnitCoverageExpectation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitCoverageExpectation {
    pub unit_id: String,
    pub tensor_names: Vec<String>,
    pub expected_expert_ids: Vec<u32>,
    pub collector_operations: Vec<TensorOperation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageContract {
    pub schema_version: u32,
    pub tensor_partition_manifest_sha256: String,
    pub calibration_manifest_sha256: String,
    pub collector_revision: String,
    pub collector_execution_identity_sha256: String,
    pub minimum_activation_rows: u64,
    pub units: Vec<UnitCoverageExpectation>,
    pub contract_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectorTapObservation {
    pub operation_id: String,
    pub graph_path: String,
    pub tensor_names: Vec<String>,
    pub activation_rows: u64,
    pub expert_activation_rows: BTreeMap<u32, u64>,
    pub activation_materialization_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitCoverageObservation {
    pub unit_id: String,
    pub tensor_names: Vec<String>,
    pub collector_taps: Vec<CollectorTapObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Structurally validated observations supplied under a D1 coverage contract.
/// The hashes are provenance links, not proof of activation execution; a
/// family-owned collector must authenticate the referenced materializations.
pub struct CoverageReceipt {
    pub schema_version: u32,
    pub contract_sha256: String,
    pub observations: Vec<UnitCoverageObservation>,
    pub observed_unit_count: usize,
    pub observed_tensor_count: usize,
    pub receipt_sha256: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DynamicProducerError {
    #[error("invalid source inventory: {0}")]
    InvalidInventory(String),
    #[error("invalid tensor partition: {0}")]
    InvalidPartition(String),
    #[error("invalid activation coverage: {0}")]
    InvalidCoverage(String),
    #[error("invalid allocation evidence binding: {0}")]
    InvalidBinding(String),
    #[error("failed to serialize ordered producer evidence: {0}")]
    Serialization(String),
}
