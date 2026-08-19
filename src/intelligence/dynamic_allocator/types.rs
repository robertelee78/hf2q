use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::super::measured_auto_quant::{ExecutionIdentity, InferenceRegime, SourceIdentity};

pub const DYNAMIC_ALLOCATION_SCHEMA_VERSION: u32 = 2;

/// GGUF encodings which can be offered only when the exact runtime capability
/// catalog says the required operation/shape routes are executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgufCodec {
    Q2K,
    Q3K,
    Q4_0,
    Q4K,
    Q5_0,
    Q5_1,
    Q5K,
    Q6K,
    Q8_0,
    Iq4Nl,
    Iq4Xs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarDType {
    F16,
    Bf16,
    F32,
}

/// Stored or executed representation of one tensor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TensorCodec {
    Gguf {
        codec: GgufCodec,
    },
    MlxAffine {
        bits: u8,
        group_size: u16,
        scale_dtype: ScalarDType,
        bias_dtype: ScalarDType,
        layout_abi: String,
    },
    Dense {
        dtype: ScalarDType,
    },
}

/// One tensor in the exact source catalog. Tied/fused tensors belong to the
/// same allocation unit, but every source tensor still appears exactly once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorMember {
    pub name: String,
    /// GGUF order: the innermost/input dimension is first.
    pub shape: Vec<usize>,
    pub role: String,
    pub source_dtype: ScalarDType,
    pub source_tensor_sha256: String,
    pub layer_index: Option<u32>,
    pub expert_index: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionTransformKind {
    Identity,
    LosslessRepack,
    LossyRequantize,
    DequantizeExpand,
}

/// One source-bound storage/load transformation. This prevents an artifact
/// stored as Q6 but silently executed as load-time Q4 from being costed as Q6.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionTransformStep {
    pub kind: ExecutionTransformKind,
    pub from: TensorCodec,
    pub to: TensorCodec,
    pub input_tensor_sha256: String,
    pub transform_receipt_sha256: String,
    pub output_tensor_sha256: String,
    /// Hash of canonical dequantized logical values before this step.
    pub input_logical_tensor_sha256: String,
    /// Equal to the input for identity, lossless repack, and dequant expansion.
    pub output_logical_tensor_sha256: String,
}

/// The complete path from bytes in the candidate artifact to bytes consumed
/// by the runtime operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorExecutionPlan {
    pub tensor_name: String,
    pub stored_codec: TensorCodec,
    pub stored_tensor_sha256: String,
    pub stored_payload_bytes: u64,
    pub executed_codec: TensorCodec,
    pub transformations: Vec<ExecutionTransformStep>,
    pub final_executed_tensor_sha256: String,
    /// Operation evidence may cover multiple tensors, for example fused
    /// gate+up. Cost is counted once per unique operation id.
    pub operation_id: String,
}

/// Exact shape/route Apple evidence for one operation and workload regime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegimeCost {
    pub regime: InferenceRegime,
    pub workload_shape_sha256: String,
    pub executable: bool,
    pub specialized_for_regime: bool,
    pub route: String,
    pub invocation_count: u64,
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
    pub tensor_names: Vec<String>,
    pub capability_decision_sha256: String,
    pub regime_costs: BTreeMap<InferenceRegime, RegimeCost>,
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
    pub shared_metadata_bytes: u64,
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
    pub tensor_catalog_sha256: String,
    pub expected_tensor_count: usize,
    pub calibration_manifest_sha256: String,
    pub sensitivity_model: SensitivityModelIdentity,
    pub capability_profile_sha256: String,
    pub proposal_workload_profile_sha256: String,
    pub required_regimes: Vec<InferenceRegime>,
    pub payload_budget_bytes: u64,
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
    pub tensor_catalog_sha256: String,
    pub calibration_manifest_sha256: String,
    pub capability_profile_sha256: String,
    pub proposal_workload_profile_sha256: String,
    pub payload_budget_bytes: u64,
    pub total_payload_bytes: u64,
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
            .map(|plan| (plan.tensor_name.as_str(), plan))
            .collect()
    }
}
