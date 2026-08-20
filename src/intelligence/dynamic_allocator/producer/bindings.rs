use super::super::{
    allocate_dynamic_frontier, DynamicAllocationError, DynamicAllocationProblem, PolicyFrontier,
};
use super::{
    validate_coverage_contract, validate_tensor_partition, verify_coverage_receipt,
    CoverageContract, CoverageReceipt, DynamicProducerError, TensorPartitionManifest,
    VerifiedCollectorTopology, VerifiedSourceTensorInventory,
};
use crate::intelligence::calibration::{
    verify_dataset_partition, DatasetPartitionManifest, RenderedDataset,
};

fn validate_identity_links(
    problem: &DynamicAllocationProblem,
    dataset_partition: &DatasetPartitionManifest,
    tensor_partition: &TensorPartitionManifest,
    coverage_contract: &CoverageContract,
    coverage_receipt: &CoverageReceipt,
) -> Result<(), DynamicProducerError> {
    let variable_tensor_count = tensor_partition
        .variable_units
        .iter()
        .try_fold(0usize, |count, unit| count.checked_add(unit.members.len()))
        .ok_or_else(|| DynamicProducerError::InvalidBinding("tensor count overflow".into()))?;
    if problem.source != tensor_partition.source
        || problem.dataset_partition_manifest_sha256 != dataset_partition.manifest_sha256
        || problem.calibration_manifest_sha256 != dataset_partition.calibration_manifest_sha256
        || problem.tensor_partition_manifest_sha256 != tensor_partition.manifest_sha256
        || problem.tensor_catalog_sha256 != tensor_partition.tensor_catalog_sha256
        || problem.expected_tensor_count != variable_tensor_count
        || coverage_contract.tensor_partition_manifest_sha256 != tensor_partition.manifest_sha256
        || coverage_contract.calibration_manifest_sha256
            != dataset_partition.calibration_manifest_sha256
        || problem.sensitivity_model.coverage_contract_sha256 != coverage_contract.contract_sha256
        || problem.sensitivity_model.coverage_receipt_sha256 != coverage_receipt.receipt_sha256
        || coverage_receipt.contract_sha256 != coverage_contract.contract_sha256
    {
        return Err(DynamicProducerError::InvalidBinding(
            "allocation problem does not bind the exact datasets, tensor partition, catalog, or coverage evidence"
                .into(),
        ));
    }
    Ok(())
}

/// Type-state admission proof for the raw additive Pareto proposer. The raw
/// problem remains inspectable, but only this type is accepted by the public
/// producer entrypoint.
#[derive(Debug, Clone)]
pub struct VerifiedDynamicAllocationProblem {
    problem: DynamicAllocationProblem,
}

pub fn allocate_verified_dynamic_frontier(
    verified: &VerifiedDynamicAllocationProblem,
) -> Result<PolicyFrontier, DynamicAllocationError> {
    allocate_dynamic_frontier(&verified.problem)
}

/// Validate the complete model-free producer chain before admitting a problem
/// to the Pareto solver. SHA-shaped strings alone are never sufficient.
#[allow(clippy::too_many_arguments)]
pub fn validate_dynamic_allocation_bindings(
    problem: &DynamicAllocationProblem,
    dataset_partition: &DatasetPartitionManifest,
    calibration: &RenderedDataset,
    policy_validation: &RenderedDataset,
    acceptance_holdout: &RenderedDataset,
    tensor_partition: &TensorPartitionManifest,
    inventory: &VerifiedSourceTensorInventory,
    topology: &VerifiedCollectorTopology,
    coverage_contract: &CoverageContract,
    coverage_receipt: &CoverageReceipt,
) -> Result<VerifiedDynamicAllocationProblem, DynamicProducerError> {
    let rebuilt_dataset_partition =
        verify_dataset_partition(calibration, policy_validation, acceptance_holdout)
            .map_err(|error| DynamicProducerError::InvalidBinding(error.to_string()))?;
    if &rebuilt_dataset_partition != dataset_partition {
        return Err(DynamicProducerError::InvalidBinding(
            "dataset partition does not reproduce from its three rendered splits".into(),
        ));
    }
    if problem.source != calibration.manifest().source
        || calibration.manifest().verified_source_manifest_sha256
            != inventory.manifest().verified_source_manifest_sha256
    {
        return Err(DynamicProducerError::InvalidBinding(
            "dataset and tensor evidence do not come from the allocation source snapshot".into(),
        ));
    }
    validate_tensor_partition(tensor_partition, inventory, &problem.units)?;
    validate_coverage_contract(
        tensor_partition,
        inventory,
        &problem.units,
        topology,
        coverage_contract,
    )?;
    let rebuilt_coverage_receipt =
        verify_coverage_receipt(coverage_contract, coverage_receipt.observations.clone())?;
    if &rebuilt_coverage_receipt != coverage_receipt {
        return Err(DynamicProducerError::InvalidBinding(
            "coverage receipt does not reproduce from its contract and observations".into(),
        ));
    }
    validate_identity_links(
        problem,
        dataset_partition,
        tensor_partition,
        coverage_contract,
        coverage_receipt,
    )?;
    super::super::allocation_problem_sha256(problem)
        .map_err(|error| DynamicProducerError::InvalidBinding(error.to_string()))?;
    Ok(VerifiedDynamicAllocationProblem {
        problem: problem.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intelligence::calibration::{
        DatasetOverlapReceipt, OverlapPolicy, CALIBRATION_INPUT_SCHEMA_VERSION,
    };
    use crate::intelligence::dynamic_allocator::{
        SearchContract, SensitivityModelIdentity, DYNAMIC_ALLOCATION_SCHEMA_VERSION,
    };
    use crate::intelligence::measured_auto_quant::{ExecutionIdentity, SourceIdentity};

    fn digest(value: &str) -> String {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(value.as_bytes()))
    }

    fn source() -> SourceIdentity {
        SourceIdentity {
            model_id: "model".into(),
            revision: "revision".into(),
            config_sha256: digest("config"),
            tensor_bundle_sha256: digest("tensor"),
            tokenizer_bundle_sha256: digest("tokenizer"),
            chat_template_sha256: digest("template"),
        }
    }

    fn linked_objects() -> (
        DynamicAllocationProblem,
        DatasetPartitionManifest,
        TensorPartitionManifest,
        CoverageContract,
        CoverageReceipt,
    ) {
        let dataset = DatasetPartitionManifest {
            schema_version: CALIBRATION_INPUT_SCHEMA_VERSION,
            calibration_manifest_sha256: digest("calibration"),
            policy_validation_manifest_sha256: digest("validation"),
            acceptance_holdout_manifest_sha256: digest("holdout"),
            overlap_policy: OverlapPolicy::RejectSourceRecordRawRenderedOrTokenWindow,
            overlap_receipt: DatasetOverlapReceipt {
                source_record_overlap_count: 0,
                raw_overlap_count: 0,
                rendered_overlap_count: 0,
                token_window_overlap_count: 0,
                compared_example_count: 3,
                receipt_sha256: digest("overlap"),
            },
            manifest_sha256: digest("dataset-partition"),
        };
        let tensor = TensorPartitionManifest {
            schema_version: super::super::DYNAMIC_PRODUCER_SCHEMA_VERSION,
            source: source(),
            source_inventory_manifest_sha256: digest("inventory"),
            source_tensor_count: 0,
            variable_units: Vec::new(),
            non_variable_tensors: Vec::new(),
            tensor_catalog_sha256: digest("catalog"),
            manifest_sha256: digest("tensor-partition"),
        };
        let coverage = CoverageContract {
            schema_version: super::super::DYNAMIC_PRODUCER_SCHEMA_VERSION,
            tensor_partition_manifest_sha256: tensor.manifest_sha256.clone(),
            calibration_manifest_sha256: dataset.calibration_manifest_sha256.clone(),
            collector_revision: "collector-v1".into(),
            collector_execution_identity_sha256: digest("collector-execution"),
            minimum_activation_rows: 1,
            units: Vec::new(),
            contract_sha256: digest("coverage"),
        };
        let receipt = CoverageReceipt {
            schema_version: super::super::DYNAMIC_PRODUCER_SCHEMA_VERSION,
            contract_sha256: coverage.contract_sha256.clone(),
            observations: Vec::new(),
            observed_unit_count: 0,
            observed_tensor_count: 0,
            receipt_sha256: digest("receipt"),
        };
        let problem = DynamicAllocationProblem {
            schema_version: DYNAMIC_ALLOCATION_SCHEMA_VERSION,
            source: source(),
            execution: ExecutionIdentity {
                hf2q_revision: "revision".into(),
                mlx_native_version: "version".into(),
                hardware_id: "hardware".into(),
                os_build: "os".into(),
            },
            tensor_catalog_sha256: tensor.tensor_catalog_sha256.clone(),
            expected_tensor_count: 0,
            dataset_partition_manifest_sha256: dataset.manifest_sha256.clone(),
            tensor_partition_manifest_sha256: tensor.manifest_sha256.clone(),
            calibration_manifest_sha256: dataset.calibration_manifest_sha256.clone(),
            sensitivity_model: SensitivityModelIdentity {
                method: "method".into(),
                version: "v1".into(),
                fixed_point_scale: 1,
                component_weights_sha256: digest("components"),
                coverage_contract_sha256: coverage.contract_sha256.clone(),
                coverage_receipt_sha256: receipt.receipt_sha256.clone(),
            },
            capability_profile_sha256: digest("capability"),
            proposal_workload_profile_sha256: digest("workload"),
            required_regimes: Vec::new(),
            payload_budget_bytes: 1,
            minimum_expert_activation_rows: 1,
            search: SearchContract::ExactPareto { max_states: 1 },
            units: Vec::new(),
        };
        (problem, dataset, tensor, coverage, receipt)
    }

    #[test]
    fn one_field_manifest_substitution_fails_closed() {
        let (problem, dataset, tensor, coverage, receipt) = linked_objects();
        validate_identity_links(&problem, &dataset, &tensor, &coverage, &receipt).unwrap();

        let mut changed = problem.clone();
        changed.dataset_partition_manifest_sha256 = digest("other-dataset");
        assert!(validate_identity_links(&changed, &dataset, &tensor, &coverage, &receipt).is_err());

        let mut changed = problem.clone();
        changed.tensor_partition_manifest_sha256 = digest("other-tensors");
        assert!(validate_identity_links(&changed, &dataset, &tensor, &coverage, &receipt).is_err());

        let mut changed = coverage.clone();
        changed.calibration_manifest_sha256 = digest("other-calibration");
        assert!(validate_identity_links(&problem, &dataset, &tensor, &changed, &receipt).is_err());

        let mut changed = receipt;
        changed.contract_sha256 = digest("other-coverage");
        assert!(validate_identity_links(&problem, &dataset, &tensor, &coverage, &changed).is_err());
    }
}
