use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::super::measured_auto_quant::{ExecutionIdentity, InferenceRegime, SourceIdentity};
use crate::core::provenance::tensor_execution::{
    TensorExecutionManifest, TensorExecutionScope, TensorRuntimeBinding,
};

pub const DYNAMIC_ALLOCATION_SCHEMA_VERSION: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarDType {
    F16,
    Bf16,
    F32,
}

/// One tensor in the exact source catalog. Tied/fused tensors belong to the
/// same allocation unit, but every source tensor still appears exactly once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorMember {
    pub name: String,
    /// Exact Hugging Face/safetensors source order. Stored and executed
    /// layouts belong to `TensorExecutionPlan`, not the source catalog.
    pub shape: Vec<usize>,
    pub role: String,
    pub source_dtype: ScalarDType,
    pub source_tensor_sha256: String,
    pub layer_index: Option<u32>,
    pub expert_index: Option<u32>,
}

/// One source member's binding into a source-to-runtime physical DAG. Node
/// identities, payload bytes, transforms, and operations are derived from the
/// referenced manifest; callers cannot repeat or override them here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorExecutionPlan {
    pub source_tensor_name: String,
    pub execution_manifest_sha256: String,
    pub lineage_slice_sha256: String,
}

/// Exact shape/route Apple evidence for one operation and workload regime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegimeCost {
    pub regime: InferenceRegime,
    /// Exact physical binding ids included in this aggregate measurement.
    pub runtime_binding_ids: Vec<String>,
    /// Canonical hash of only those exact regime bindings, including their
    /// request/decision envelopes, input nodes, and invocation counts.
    pub runtime_binding_bundle_sha256: String,
    pub executable: bool,
    pub specialized_for_regime: bool,
    pub median_nanoseconds: u64,
    pub p95_nanoseconds: u64,
    pub warmup_runs: u32,
    pub measured_runs: u32,
    pub measurement_receipt_sha256: String,
}

/// Runtime evidence for one independently dispatched or fused operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationExecutionEvidence {
    pub operation_id: String,
    pub graph_path: String,
    pub source_tensor_names: Vec<String>,
    pub executed_tensor_node_ids: Vec<String>,
    /// Canonical bundle of regime-specific physical capability bindings.
    pub capability_binding_bundle_sha256: String,
    pub regime_costs: BTreeMap<InferenceRegime, RegimeCost>,
}

/// Stable logical graph operation shared by every precision option for one
/// allocation unit. Storage/runtime routes may differ; tensor coverage and
/// graph identity may not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorOperation {
    pub operation_id: String,
    pub graph_path: String,
    pub tensor_names: Vec<String>,
}

/// Shared definition and scale for additive proposal loss. Full-model quality
/// evidence remains authoritative because local tensor effects interact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensitivityModelIdentity {
    pub method: String,
    pub version: String,
    pub fixed_point_scale: u64,
    pub component_weights_sha256: String,
    pub coverage_contract_sha256: String,
    /// Structurally admitted D1 observation receipt. A later family-owned
    /// collector must authenticate the referenced materializations.
    pub coverage_receipt_sha256: String,
}

/// Auditable fixed-point local quality evidence for one unit option.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensitivityEvidence {
    pub calibration_manifest_sha256: String,
    pub sensitivity_receipt_sha256: String,
    /// Aggregate fixed-point proposal loss. Lower is better.
    pub loss_units: u64,
    pub imatrix_weighted_error_units: u64,
    /// Signed because a first-order teacher-KL proxy can be noisy.
    pub teacher_kl_alignment_units: i64,
    pub block_output_error_units: u64,
    pub uncertainty_units: u64,
    pub activation_rows: u64,
    /// Exact routed rows by expert id. Empty for a dense unit.
    pub expert_activation_rows: BTreeMap<u32, u64>,
    /// Canonical hash of every plan's final executed tensor hash and codec.
    pub final_executed_tensor_bundle_sha256: String,
}

/// One executable composite precision choice for an atomic tensor unit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorOption {
    pub option_id: String,
    pub execution_plans: Vec<TensorExecutionPlan>,
    pub operations: Vec<OperationExecutionEvidence>,
    pub payload_bytes: u64,
    pub storage_manifest_receipt_sha256: String,
    pub sensitivity: SensitivityEvidence,
    pub capability_profile_sha256: String,
}

/// Tensors that must be allocated atomically, plus their candidate options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorAllocationUnit {
    pub unit_id: String,
    pub members: Vec<TensorMember>,
    /// Logical routed experts represented by this unit, including packed
    /// rank-3 expert tensors which do not have one physical tensor per expert.
    pub expected_expert_ids: Vec<u32>,
    pub operations: Vec<TensorOperation>,
    pub options: Vec<TensorOption>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum SearchContract {
    /// Exact within the supplied local additive metrics. Exceeding the state
    /// bound is an error; the implementation never silently truncates.
    ExactPareto { max_states: usize },
}

/// Source-, corpus-, runtime-, and workload-bound input to the proposer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicAllocationProblem {
    pub schema_version: u32,
    pub source: SourceIdentity,
    pub execution: ExecutionIdentity,
    /// Exact graph/routing/capability/hardware context shared by every
    /// physical candidate manifest in this frontier.
    pub tensor_runtime: TensorRuntimeBinding,
    pub execution_scope: TensorExecutionScope,
    pub tensor_catalog_sha256: String,
    pub expected_tensor_count: usize,
    /// Three-way calibration, repair-validation, and untouched acceptance
    /// partition. Sensitivity production may consume only its calibration
    /// split; later candidate gates consume the other two phases.
    pub dataset_partition_manifest_sha256: String,
    /// Complete source inventory partition into variable allocation units and
    /// explicit fixed/protected/excluded tensors.
    pub tensor_partition_manifest_sha256: String,
    /// Canonical candidate physical execution manifests from which option
    /// payload, terminal tensor, and capability facts are derived. D2a
    /// validates their structure; D2b additionally authenticates the bytes.
    pub execution_manifest_catalog: Vec<TensorExecutionManifest>,
    pub execution_manifest_catalog_sha256: String,
    pub calibration_manifest_sha256: String,
    pub sensitivity_model: SensitivityModelIdentity,
    pub capability_profile_sha256: String,
    pub proposal_workload_profile_sha256: String,
    pub required_regimes: Vec<InferenceRegime>,
    /// Budget for variable candidate storage only. Fixed/protected/excluded
    /// base artifact bytes are bound and added by the D2b materializer.
    pub variable_payload_budget_bytes: u64,
    pub minimum_expert_activation_rows: u64,
    pub search: SearchContract,
    pub units: Vec<TensorAllocationUnit>,
}

/// Selected option for one allocation unit. The complete evidence is retained
/// so the policy hash binds route, transform, and sensitivity provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitDecision {
    pub unit_id: String,
    pub selected_option: TensorOption,
    pub regime_cost_nanoseconds: BTreeMap<InferenceRegime, u64>,
}

/// One nondominated proposal. It is not a production eligibility receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecisionPolicyManifest {
    pub schema_version: u32,
    pub allocation_problem_sha256: String,
    pub source: SourceIdentity,
    pub execution: ExecutionIdentity,
    pub tensor_runtime: TensorRuntimeBinding,
    pub execution_scope: TensorExecutionScope,
    pub tensor_catalog_sha256: String,
    pub dataset_partition_manifest_sha256: String,
    pub tensor_partition_manifest_sha256: String,
    pub execution_manifest_catalog_sha256: String,
    pub calibration_manifest_sha256: String,
    pub capability_profile_sha256: String,
    pub proposal_workload_profile_sha256: String,
    pub variable_payload_budget_bytes: u64,
    pub total_variable_payload_bytes: u64,
    pub total_loss_units: u64,
    pub total_regime_cost_nanoseconds: BTreeMap<InferenceRegime, u64>,
    pub decisions: Vec<UnitDecision>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchReceipt {
    pub algorithm: String,
    pub exhaustive_within_proxy_metrics: bool,
    pub state_limit: usize,
    pub states_generated: u64,
    pub states_pruned_dominated: u64,
    pub equivalent_states_collapsed: u64,
    pub peak_frontier_states: usize,
    pub frontier_size: usize,
}

/// Complete nondominated local proxy frontier for subsequent materialization,
/// full-model repair/evaluation, and exact Apple benchmark selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyFrontier {
    pub schema_version: u32,
    pub allocation_problem_sha256: String,
    pub policies: Vec<PrecisionPolicyManifest>,
    pub search_receipt: SearchReceipt,
}

impl PrecisionPolicyManifest {
    pub fn tensor_execution_plans(&self) -> BTreeMap<&str, &TensorExecutionPlan> {
        self.decisions
            .iter()
            .flat_map(|decision| decision.selected_option.execution_plans.iter())
            .map(|plan| (plan.source_tensor_name.as_str(), plan))
            .collect()
    }
}
