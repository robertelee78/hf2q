use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const RECEIPT_SCHEMA_VERSION: u32 = 2;

/// Exact source checkpoint identity. The weight hash is intentionally opaque:
/// vanilla, fine-tuned, merged, and weight-edited checkpoints use the same
/// contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceIdentity {
    pub model_id: String,
    pub revision: String,
    pub config_sha256: String,
    pub tensor_bundle_sha256: String,
    pub tokenizer_bundle_sha256: String,
    pub chat_template_sha256: String,
}

/// Exact executable environment for a benchmark receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionIdentity {
    pub hf2q_revision: String,
    pub mlx_native_version: String,
    pub hardware_id: String,
    pub os_build: String,
}

/// Offline algorithm used to produce a candidate. This is distinct from both
/// its stored encoding and the runtime kernel that executes that encoding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationAlgorithm {
    RoundToNearest,
    ImportanceMatrix,
    DynamicMixedPrecision,
    Awq,
    Gptq,
    Dwq,
}

/// On-disk weight representation consumed by the native runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WeightEncoding {
    Gguf {
        quant_type: String,
    },
    /// Default affine profile for the artifact. Tensor-specific bit-width and
    /// group-size overrides are authoritative in
    /// `precision_policy_manifest_sha256`; these fields are not a claim that
    /// every tensor uses one homogeneous profile.
    MlxAffine {
        bits: u8,
        group_size: u16,
    },
}

/// Reproducible conversion and execution recipe for one candidate artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateRecipe {
    pub candidate_id: String,
    /// Ordered offline transformation pipeline. Round-to-nearest is a
    /// standalone control; calibrated cascades can compose stages such as
    /// dynamic allocation, AWQ, and DWQ in the order they were executed.
    pub calibration_pipeline: Vec<CalibrationAlgorithm>,
    pub encoding: WeightEncoding,
    /// Hash of the complete per-tensor precision-policy manifest.
    pub precision_policy_manifest_sha256: String,
    /// Calibration corpus and stored teacher-target hashes, when applicable.
    pub calibration_corpus_sha256: Option<String>,
    /// Hash of the canonical template-rendered UTF-8 calibration stream.
    pub calibration_rendered_text_sha256: Option<String>,
    /// Hash of the canonical encoded token-id stream, including sequence
    /// boundaries and integer byte order defined by the calibration manifest.
    pub calibration_token_ids_sha256: Option<String>,
    /// Hash of the canonical calibration manifest that binds source, examples,
    /// rendering, token stream, collector, tensor/expert coverage, and imatrix.
    pub calibration_manifest_sha256: Option<String>,
    pub teacher_targets_sha256: Option<String>,
    /// Hash of the native capability decisions and tensor-to-kernel routes
    /// used by the measured server.
    pub kernel_profile_sha256: String,
    /// Hash of KV/cache, speculation, batching, and other server settings.
    pub server_config_sha256: String,
}

/// Exact pass count for a behavioral or structural suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseResult {
    pub passed: u32,
    pub total: u32,
}

impl CaseResult {
    pub(super) fn is_complete(self) -> bool {
        self.total > 0 && self.passed == self.total
    }
}

/// Quality evidence against the exact unquantized source checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityEvidence {
    /// Hash of the exact corpora, prompts, sampling, activation taps, metric
    /// settings, and named behavioral suites used to produce this evidence.
    pub quality_suite_sha256: String,
    pub source_integrity_passed: bool,
    pub teacher_kl: TeacherKlEvidence,
    pub top1_token_agreement: f64,
    pub activation_cosine_similarity: f64,
    pub perplexity_ratio: f64,
    /// Fixed-horizon free-running greedy comparison against the exact source.
    /// Both source and candidate generate exactly `tokens_per_prompt` tokens;
    /// stopping is disabled for this metric so its denominator is invariant.
    pub greedy_trajectory: GreedyTrajectoryEvidence,
    pub dataset_separation: DatasetSeparationEvidence,
    pub tool_call_cases: CaseResult,
    pub context_cases: CaseResult,
    pub cache_cases: CaseResult,
    pub multimodal_cases: Option<CaseResult>,
    /// Owner- or family-supplied behavior suites. Names and semantics are
    /// external to the selector; no model transformation is special-cased.
    pub behavioral_regressions: BTreeMap<String, CaseResult>,
}

/// Gate-0 proof that the exact emitted artifact is executable by the exact
/// runtime. Converter success alone is insufficient.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServabilityEvidence {
    pub conversion_completed: bool,
    pub artifact_catalog_passed: bool,
    pub runtime_load_cases: CaseResult,
    pub kernel_contract_cases: CaseResult,
}

/// Hard quality thresholds. They are eligibility gates, never ranking weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityContract {
    pub quality_suite_sha256: String,
    pub evaluation_manifest_sha256: String,
    pub deduplication_policy_sha256: String,
    pub max_teacher_kl_mean: f64,
    pub max_teacher_kl_p95: f64,
    pub max_teacher_kl_max: f64,
    pub min_teacher_kl_prompts: u32,
    pub min_teacher_kl_tokens: u64,
    pub min_top1_token_agreement: f64,
    pub min_activation_cosine_similarity: f64,
    pub max_perplexity_ratio: f64,
    pub min_greedy_trajectory_prompts: u32,
    pub min_greedy_trajectory_tokens_per_prompt: u32,
    pub min_greedy_trajectory_exact_match_rate: f64,
    pub min_greedy_trajectory_mean_common_prefix_ratio: f64,
    pub require_calibration_evaluation_disjoint: bool,
    pub require_tool_calls: bool,
    pub require_context: bool,
    pub require_cache: bool,
    pub require_multimodal: bool,
    pub required_behavioral_regressions: BTreeSet<String>,
}

/// Distribution-aware KL evidence against the exact source teacher.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherKlEvidence {
    pub mean: f64,
    pub p95: f64,
    pub max: f64,
    pub prompt_count: u32,
    pub token_count: u64,
    pub receipt_sha256: String,
}

/// Exact calibration/evaluation split and deduplication evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetSeparationEvidence {
    pub calibration_manifest_sha256: Option<String>,
    pub evaluation_manifest_sha256: String,
    pub deduplication_policy_sha256: String,
    pub overlap_count: u32,
    /// Hash of the canonical pairwise overlap report from which
    /// `overlap_count` is derived.
    pub receipt_sha256: String,
}

/// Integer evidence for a fixed-horizon greedy trajectory comparison.
///
/// Rates are derived by the selector instead of trusted from a producer:
/// `exact_match_prompts / prompt_count` and
/// `total_common_prefix_tokens / (prompt_count * tokens_per_prompt)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GreedyTrajectoryEvidence {
    pub prompt_count: u32,
    pub tokens_per_prompt: u32,
    pub exact_match_prompts: u32,
    pub total_common_prefix_tokens: u64,
    /// Hash of the canonical per-prompt source/candidate token trajectories.
    pub receipt_sha256: String,
}

/// A distinct execution shape. Quantization speed is evaluated separately for
/// prompt QMM, token QMV, width-N/concurrent decode, long context, and vision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceRegime {
    TextPrefill,
    TextDecodeM1,
    TextDecodeWidthN,
    LongContextDecode,
    MultimodalPrefill,
}

/// Measured median for one exact workload regime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceMeasurement {
    pub regime: InferenceRegime,
    pub workload_sha256: String,
    pub median_tokens_per_second: f64,
    pub median_semantic_ttft_ms: f64,
    pub peak_mlx_bytes: u64,
    pub warmup_runs: u32,
    pub measured_runs: u32,
    pub tokens_per_run: u32,
    /// Semantic/output checks performed on generations from this exact regime.
    pub output_quality_cases: CaseResult,
}

/// Minimum service level and evidence depth for a required regime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegimeRequirement {
    pub regime: InferenceRegime,
    /// Hash of prompts, templates, sampling, context, batch, and media inputs.
    pub workload_sha256: String,
    pub min_tokens_per_second: f64,
    /// Omit when the regime has no TTFT service-level objective.
    pub max_semantic_ttft_ms: Option<f64>,
    pub min_warmup_runs: u32,
    pub min_measured_runs: u32,
    pub min_tokens_per_run: u32,
}

/// Complete evidence receipt for a converted artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateReceipt {
    pub schema_version: u32,
    pub source: SourceIdentity,
    pub execution: ExecutionIdentity,
    pub recipe: CandidateRecipe,
    pub artifact_sha256: String,
    pub artifact_bytes: u64,
    pub servability: ServabilityEvidence,
    pub quality: QualityEvidence,
    pub performance: Vec<PerformanceMeasurement>,
}

/// User/workload-specific selection contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionContract {
    /// Hash of the canonical selection profile, thresholds, and constraints.
    pub selection_profile_sha256: String,
    pub source: SourceIdentity,
    pub execution: ExecutionIdentity,
    pub quality: QualityContract,
    pub required_regimes: Vec<RegimeRequirement>,
    pub primary_regime: InferenceRegime,
    pub max_peak_mlx_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateRejectionCode {
    ReceiptSchemaMismatch,
    SourceIdentityMismatch,
    ExecutionIdentityMismatch,
    ArtifactIdentityInvalid,
    WeightEncodingInvalid,
    RecipeIdentityInvalid,
    CalibrationCorpusIdentityInvalid,
    CalibrationRenderingIdentityInvalid,
    CalibrationManifestIdentityInvalid,
    TeacherTargetIdentityInvalid,
    ConversionGateFailed,
    ArtifactCatalogGateFailed,
    RuntimeLoadGateFailed,
    KernelContractGateFailed,
    QualitySuiteIdentityMismatch,
    CalibrationEvaluationIdentityMismatch,
    SourceIntegrityGateFailed,
    TeacherKlGateFailed,
    Top1AgreementGateFailed,
    ActivationCosineGateFailed,
    PerplexityGateFailed,
    GreedyTrajectoryGateFailed,
    CalibrationEvaluationLeakageGateFailed,
    RequiredCaseGateFailed,
    MultimodalGateFailed,
    BehavioralRegressionGateFailed,
    DuplicatePerformanceMeasurement,
    InvalidPerformanceMeasurement,
    MissingPerformanceRegime,
    ThroughputGateFailed,
    WorkloadIdentityMismatch,
    SemanticTtftGateFailed,
    MemoryGateFailed,
    InsufficientBenchmarkEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateRejectionReason {
    pub code: CandidateRejectionCode,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateRejection {
    pub candidate_id: String,
    pub reasons: Vec<CandidateRejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectionDecision {
    pub schema_version: u32,
    pub selection_profile_sha256: String,
    pub source: SourceIdentity,
    pub execution: ExecutionIdentity,
    pub selected_candidate_id: String,
    pub selected_artifact_sha256: String,
    pub eligible_candidate_ids: Vec<String>,
    pub rejected_candidates: Vec<CandidateRejection>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MeasuredAutoQuantError {
    #[error("selection contract is invalid: {0}")]
    InvalidContract(String),
    #[error("candidate ids must be unique: {0}")]
    DuplicateCandidateId(String),
    #[error("no candidate satisfies the exact evidence contract: {rejections:?}")]
    NoEligibleCandidate { rejections: Vec<CandidateRejection> },
}
