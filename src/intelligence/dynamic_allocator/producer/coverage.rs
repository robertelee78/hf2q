use std::collections::BTreeSet;

use serde::Serialize;

use super::super::TensorAllocationUnit;
use super::partition::{
    descriptor_from_unit, producer_hash, valid_sha256, validate_tensor_partition,
};
use super::types::*;

#[derive(Serialize)]
struct CoverageContractHashView<'a> {
    schema_version: u32,
    tensor_partition_manifest_sha256: &'a str,
    calibration_manifest_sha256: &'a str,
    collector_revision: &'a str,
    collector_execution_identity_sha256: &'a str,
    minimum_activation_rows: u64,
    units: &'a [UnitCoverageExpectation],
}

fn contract_hash(contract: &CoverageContract) -> Result<String, DynamicProducerError> {
    producer_hash(&CoverageContractHashView {
        schema_version: contract.schema_version,
        tensor_partition_manifest_sha256: &contract.tensor_partition_manifest_sha256,
        calibration_manifest_sha256: &contract.calibration_manifest_sha256,
        collector_revision: &contract.collector_revision,
        collector_execution_identity_sha256: &contract.collector_execution_identity_sha256,
        minimum_activation_rows: contract.minimum_activation_rows,
        units: &contract.units,
    })
}

pub fn verify_collector_topology(
    units: &[TensorAllocationUnit],
) -> Result<VerifiedCollectorTopology, DynamicProducerError> {
    let mut global_operation_ids = BTreeSet::new();
    let mut descriptors: Vec<_> = units
        .iter()
        .map(descriptor_from_unit)
        .collect::<Result<_, _>>()?;
    descriptors.sort_by(|left, right| left.unit_id.cmp(&right.unit_id));
    let mut expectations = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        if descriptor
            .operations
            .iter()
            .any(|operation| !global_operation_ids.insert(operation.operation_id.clone()))
        {
            return Err(DynamicProducerError::InvalidCoverage(
                "logical operation ids must be globally unique".into(),
            ));
        }
        let tensor_names: Vec<_> = descriptor
            .members
            .iter()
            .map(|member| member.name.clone())
            .collect();
        expectations.push(UnitCoverageExpectation {
            unit_id: descriptor.unit_id,
            tensor_names,
            expected_expert_ids: descriptor.expected_expert_ids,
            collector_operations: descriptor.operations,
        });
    }
    expectations.sort_by(|left, right| left.unit_id.cmp(&right.unit_id));
    Ok(VerifiedCollectorTopology {
        units: expectations,
    })
}

pub fn build_coverage_contract(
    partition: &TensorPartitionManifest,
    inventory: &VerifiedSourceTensorInventory,
    calibration_manifest_sha256: String,
    collector_revision: String,
    collector_execution_identity_sha256: String,
    minimum_activation_rows: u64,
    units: &[TensorAllocationUnit],
    topology: &VerifiedCollectorTopology,
) -> Result<CoverageContract, DynamicProducerError> {
    validate_tensor_partition(partition, inventory, units)?;
    if minimum_activation_rows == 0
        || !valid_sha256(&calibration_manifest_sha256)
        || collector_revision.is_empty()
        || !valid_sha256(&collector_execution_identity_sha256)
    {
        return Err(DynamicProducerError::InvalidCoverage(
            "partition, calibration, collector identity, and positive row bound are required"
                .into(),
        ));
    }
    let mut expected_descriptors: Vec<_> = units
        .iter()
        .map(descriptor_from_unit)
        .collect::<Result<_, _>>()?;
    expected_descriptors.sort_by(|left, right| left.unit_id.cmp(&right.unit_id));
    if topology.units.len() != expected_descriptors.len()
        || topology
            .units
            .iter()
            .zip(expected_descriptors.iter())
            .any(|(actual, expected)| {
                actual.unit_id != expected.unit_id
                    || actual.tensor_names
                        != expected
                            .members
                            .iter()
                            .map(|member| member.name.clone())
                            .collect::<Vec<_>>()
                    || actual.expected_expert_ids != expected.expected_expert_ids
            })
    {
        return Err(DynamicProducerError::InvalidCoverage(
            "verified collector topology does not match allocation units".into(),
        ));
    }

    let mut contract = CoverageContract {
        schema_version: DYNAMIC_PRODUCER_SCHEMA_VERSION,
        tensor_partition_manifest_sha256: partition.manifest_sha256.clone(),
        calibration_manifest_sha256,
        collector_revision,
        collector_execution_identity_sha256,
        minimum_activation_rows,
        units: topology.units.clone(),
        contract_sha256: String::new(),
    };
    contract.contract_sha256 = contract_hash(&contract)?;
    Ok(contract)
}

#[allow(clippy::too_many_arguments)]
pub fn validate_coverage_contract(
    partition: &TensorPartitionManifest,
    inventory: &VerifiedSourceTensorInventory,
    units: &[TensorAllocationUnit],
    topology: &VerifiedCollectorTopology,
    contract: &CoverageContract,
) -> Result<(), DynamicProducerError> {
    let rebuilt = build_coverage_contract(
        partition,
        inventory,
        contract.calibration_manifest_sha256.clone(),
        contract.collector_revision.clone(),
        contract.collector_execution_identity_sha256.clone(),
        contract.minimum_activation_rows,
        units,
        topology,
    )?;
    if &rebuilt != contract {
        return Err(DynamicProducerError::InvalidCoverage(
            "coverage contract does not reproduce from the verified partition and topology".into(),
        ));
    }
    Ok(())
}

#[derive(Serialize)]
struct CoverageReceiptHashView<'a> {
    schema_version: u32,
    contract_sha256: &'a str,
    observations: &'a [UnitCoverageObservation],
    observed_unit_count: usize,
    observed_tensor_count: usize,
}

pub fn verify_coverage_receipt(
    contract: &CoverageContract,
    mut observations: Vec<UnitCoverageObservation>,
) -> Result<CoverageReceipt, DynamicProducerError> {
    if contract.schema_version != DYNAMIC_PRODUCER_SCHEMA_VERSION
        || contract.minimum_activation_rows == 0
        || contract.collector_revision.is_empty()
        || !valid_sha256(&contract.collector_execution_identity_sha256)
        || contract_hash(contract)? != contract.contract_sha256
    {
        return Err(DynamicProducerError::InvalidCoverage(
            "coverage contract identity is invalid".into(),
        ));
    }
    observations.sort_by(|left, right| left.unit_id.cmp(&right.unit_id));
    if observations.len() != contract.units.len() {
        return Err(DynamicProducerError::InvalidCoverage(
            "missing or unexpected allocation-unit observation".into(),
        ));
    }
    let mut observed_tensor_count = 0usize;
    for (expected, observed) in contract.units.iter().zip(&mut observations) {
        observed.tensor_names.sort();
        observed
            .collector_taps
            .sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
        for tap in &mut observed.collector_taps {
            tap.tensor_names.sort();
        }
        let expected_experts: BTreeSet<_> = expected.expected_expert_ids.iter().copied().collect();
        if observed.unit_id != expected.unit_id
            || observed.tensor_names != expected.tensor_names
            || observed.collector_taps.len() != expected.collector_operations.len()
            || observed
                .collector_taps
                .iter()
                .zip(expected.collector_operations.iter())
                .any(|(actual, required)| {
                    actual.operation_id != required.operation_id
                        || actual.graph_path != required.graph_path
                        || actual.tensor_names != required.tensor_names
                        || actual.activation_rows < contract.minimum_activation_rows
                        || actual
                            .expert_activation_rows
                            .keys()
                            .copied()
                            .collect::<BTreeSet<_>>()
                            != expected_experts
                        || actual
                            .expert_activation_rows
                            .values()
                            .any(|rows| *rows < contract.minimum_activation_rows)
                        || !valid_sha256(&actual.activation_materialization_sha256)
                })
        {
            return Err(DynamicProducerError::InvalidCoverage(format!(
                "coverage for unit {} is missing, duplicated, unexpected, or below its row floor",
                expected.unit_id
            )));
        }
        observed_tensor_count = observed_tensor_count
            .checked_add(observed.tensor_names.len())
            .ok_or_else(|| DynamicProducerError::InvalidCoverage("tensor count overflow".into()))?;
    }

    let observed_unit_count = observations.len();
    let mut receipt = CoverageReceipt {
        schema_version: DYNAMIC_PRODUCER_SCHEMA_VERSION,
        contract_sha256: contract.contract_sha256.clone(),
        observations,
        observed_unit_count,
        observed_tensor_count,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = producer_hash(&CoverageReceiptHashView {
        schema_version: receipt.schema_version,
        contract_sha256: &receipt.contract_sha256,
        observations: &receipt.observations,
        observed_unit_count: receipt.observed_unit_count,
        observed_tensor_count: receipt.observed_tensor_count,
    })?;
    Ok(receipt)
}
