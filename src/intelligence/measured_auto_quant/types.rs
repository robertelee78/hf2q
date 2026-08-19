use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const RECEIPT_SCHEMA_VERSION: u32 = 1;

/// Exact source checkpoint identity. The weight hash is intentionally opaque:
/// vanilla, fine-tuned, merged, and weight-edited checkpoints use the same
/// contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    Gguf { quant_type: String },
    MlxAffine { bits: u8, group_size: u16 },
}

/// Reproducible conversion and execution recipe for one candidate artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateRecipe {
    pub candidate_id: String,
    pub algorithm: CalibrationAlgorithm,
    pub encoding: WeightEncoding,
    /// Hash of the complete per-tensor precision/calibration manifest.
    pub policy_sha256: String,
    /// Calibration corpus and stored teacher-target hashes, when applicable.
    pub calibration_corpus_sha256: Option<String>,
    pub teacher_targets_sha256: Option<String>,
    /// Hash of the native capability decisions and tensor-to-kernel routes
    /// used by the measured server.
    pub kernel_profile_sha256: String,
    /// Hash of KV/cache, speculation, batching, and other server settings.
    pub server_config_sha256: String,
}

/// Exact pass count for a behavioral or structural suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
pub struct QualityEvidence {
    /// Hash of the exact corpora, prompts, sampling, activation taps, metric
    /// settings, and named behavioral suites used to produce this evidence.
    pub quality_suite_sha256: String,
    pub source_integrity_passed: bool,
    pub teacher_kl_divergence: f64,
    pub top1_token_agreement: f64,
    pub activation_cosine_similarity: f64,
    pub perplexity_ratio: f64,
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
pub struct ServabilityEvidence {
    pub conversion_completed: bool,
    pub artifact_catalog_passed: bool,
    pub runtime_load_cases: CaseResult,
    pub kernel_contract_cases: CaseResult,
}

/// Hard quality thresholds. They are eligibility gates, never ranking weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityContract {
    pub quality_suite_sha256: String,
    pub max_teacher_kl_divergence: f64,
    pub min_top1_token_agreement: f64,
    pub min_activation_cosine_similarity: f64,
    pub max_perplexity_ratio: f64,
    pub require_tool_calls: bool,
    pub require_context: bool,
    pub require_cache: bool,
    pub require_multimodal: bool,
    pub required_behavioral_regressions: BTreeSet<String>,
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
    TeacherTargetIdentityInvalid,
    ConversionGateFailed,
    ArtifactCatalogGateFailed,
    RuntimeLoadGateFailed,
    KernelContractGateFailed,
    QualitySuiteIdentityMismatch,
    SourceIntegrityGateFailed,
    TeacherKlGateFailed,
    Top1AgreementGateFailed,
    ActivationCosineGateFailed,
    PerplexityGateFailed,
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
