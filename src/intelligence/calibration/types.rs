use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::intelligence::measured_auto_quant::SourceIdentity;
use crate::serve::api::schema::{ChatMessage, Tool};

pub const CALIBRATION_INPUT_SCHEMA_VERSION: u32 = 1;
pub const TEACHER_PREDICTION_PLAN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplit {
    Calibration,
    PolicyValidation,
    AcceptanceHoldout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderMode {
    GenerationPrompt,
    CompletedAssistantTranscript,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExampleProvenance {
    pub dataset_id: String,
    pub revision: String,
    pub record_id: String,
    pub license: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredExample {
    pub stable_id: String,
    pub provenance: ExampleProvenance,
    pub domains: Vec<String>,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<Tool>,
    pub render_mode: RenderMode,
    pub enable_thinking: bool,
    pub chat_template_kwargs: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredDatasetManifest {
    pub schema_version: u32,
    pub dataset_id: String,
    pub revision: String,
    pub license: String,
    pub split: DatasetSplit,
    pub seed: u64,
    /// Exact example order consumed by rendering and collection.
    pub example_order: Vec<String>,
    pub examples: Vec<StructuredExample>,
    pub source_record_sha256: BTreeMap<String, String>,
    pub raw_example_sha256: BTreeMap<String, String>,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationCorpusArtifactLimits {
    pub max_artifact_bytes: u64,
    pub max_examples: usize,
    pub max_messages: usize,
    pub max_tools: usize,
}

#[derive(Debug, Clone)]
pub struct VerifyCalibrationCorpusRequest {
    pub path: PathBuf,
    pub expected_sha256: String,
    pub expected_dataset_id: String,
    pub expected_revision: String,
    pub expected_declared_license: String,
    pub expected_split: DatasetSplit,
    pub limits: CalibrationCorpusArtifactLimits,
}

/// Owned, path-swap-resistant structured corpus authority. The artifact hash
/// authenticates the exact JSON bytes; the license is explicitly a declaration
/// bound into those bytes, not an independently adjudicated legal conclusion.
#[derive(Debug, Clone)]
pub(crate) struct VerifiedCalibrationCorpus {
    pub(super) artifact: crate::core::provenance::tensor_execution::ArtifactEvidence,
    pub(super) manifest: StructuredDatasetManifest,
}

impl VerifiedCalibrationCorpus {
    pub fn artifact(&self) -> &crate::core::provenance::tensor_execution::ArtifactEvidence {
        &self.artifact
    }

    pub fn manifest(&self) -> &StructuredDatasetManifest {
        &self.manifest
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderedExampleReceipt {
    pub stable_id: String,
    pub source_record_sha256: String,
    pub raw_example_sha256: String,
    pub rendered_utf8_sha256: String,
    pub token_ids_sha256: String,
    pub token_count: usize,
    pub scoring_ranges: Vec<TokenRange>,
    pub token_window_sha256: Vec<String>,
    pub add_generation_prompt: bool,
    pub requested_enable_thinking: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderedDatasetManifest {
    pub schema_version: u32,
    pub split: DatasetSplit,
    pub source: SourceIdentity,
    pub verified_source_manifest_sha256: String,
    pub structured_dataset_sha256: String,
    pub chat_template_source: crate::core::chat_template_resolver::ChatTemplateSource,
    pub chat_template_sha256: String,
    pub tokenizer_json_sha256: String,
    pub renderer_revision: String,
    pub max_tokens_per_example: usize,
    pub token_window_size: usize,
    pub examples: Vec<RenderedExampleReceipt>,
    pub rendered_text_stream_sha256: String,
    pub token_id_stream_sha256: String,
    pub manifest_sha256: String,
}

/// Opaque source-verified rendering. Its fields are visible only to the
/// calibration implementation and its adversarial tests; other subsystems can
/// obtain this type only by rendering from a verified source snapshot.
#[derive(Debug, Clone)]
pub struct RenderedDataset {
    pub(super) structured: StructuredDatasetManifest,
    pub(super) manifest: RenderedDatasetManifest,
    pub(super) rendered_utf8: BTreeMap<String, String>,
    pub(super) token_ids: BTreeMap<String, Vec<u32>>,
}

impl RenderedDataset {
    pub fn manifest(&self) -> &RenderedDatasetManifest {
        &self.manifest
    }
}

#[derive(Debug, Clone)]
pub struct RenderDatasetRequest {
    pub model_dir: PathBuf,
    pub arch: String,
    pub source: SourceIdentity,
    pub verified_source: crate::input::integrity::VerifiedSourceManifest,
    pub renderer_revision: String,
    pub max_tokens_per_example: usize,
    pub token_window_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverlapPolicy {
    RejectSourceRecordRawRenderedOrTokenWindow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetOverlapReceipt {
    pub source_record_overlap_count: usize,
    pub raw_overlap_count: usize,
    pub rendered_overlap_count: usize,
    pub token_window_overlap_count: usize,
    pub compared_example_count: usize,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetPartitionManifest {
    pub schema_version: u32,
    pub calibration_manifest_sha256: String,
    pub policy_validation_manifest_sha256: String,
    pub acceptance_holdout_manifest_sha256: String,
    pub overlap_policy: OverlapPolicy,
    pub overlap_receipt: DatasetOverlapReceipt,
    pub manifest_sha256: String,
}

/// Global bounds applied before exact-teacher target collection can begin.
/// These limits complement the renderer's per-example token bound and make
/// the total work and target artifact size preflightable with checked math.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherPredictionPlanLimits {
    pub max_examples: usize,
    pub max_total_tokens: usize,
    pub max_rendered_utf8_bytes: u64,
    pub max_prediction_points: usize,
    pub max_prefix_tokens: usize,
    pub max_generation_prompts: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum TeacherPredictionPointKind {
    /// Logits from `tokens[..target_token_index]` predict the exact token at
    /// `target_token_index` in a completed assistant transcript.
    TeacherForced {
        target_token_index: usize,
        target_token_id: u32,
    },
    /// Logits from the complete generation prompt predict the next token.
    GenerationNext,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherPredictionPointReceipt {
    pub point_ordinal: usize,
    pub stable_id: String,
    pub kind: TeacherPredictionPointKind,
    pub prefix_token_count: usize,
    pub prefix_token_ids_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherGreedyPromptReceipt {
    pub stable_id: String,
    pub prefix_token_count: usize,
    pub prefix_token_ids_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherPredictionExampleReceipt {
    pub stable_id: String,
    pub render_mode: RenderMode,
    pub token_count: usize,
    pub token_ids_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherPredictionPlanManifest {
    pub schema_version: u32,
    pub dataset_partition_manifest_sha256: String,
    pub calibration_corpus_artifact_sha256: String,
    pub calibration_manifest_sha256: String,
    pub rendered_token_stream_sha256: String,
    pub limits: TeacherPredictionPlanLimits,
    pub total_example_count: usize,
    pub total_token_count: usize,
    pub total_rendered_utf8_bytes: u64,
    pub examples: Vec<TeacherPredictionExampleReceipt>,
    pub prediction_points: Vec<TeacherPredictionPointReceipt>,
    pub greedy_prompts: Vec<TeacherGreedyPromptReceipt>,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone)]
pub(crate) struct VerifiedCalibrationPredictionPlan {
    pub(super) manifest: TeacherPredictionPlanManifest,
    pub(super) examples: Vec<TeacherPredictionExample>,
}

#[derive(Debug, Clone)]
pub(super) struct TeacherPredictionExample {
    pub token_ids: Vec<u32>,
    pub point_ordinals: Vec<usize>,
    pub greedy_prompt_ordinal: Option<usize>,
}

impl VerifiedCalibrationPredictionPlan {
    pub(crate) fn manifest(&self) -> &TeacherPredictionPlanManifest {
        &self.manifest
    }

    pub(crate) fn prediction_point_count(&self) -> usize {
        self.manifest.prediction_points.len()
    }

    pub(crate) fn visit_prediction_points<E>(
        &self,
        mut visit: impl FnMut(&TeacherPredictionPointReceipt, &[u32]) -> Result<(), E>,
    ) -> Result<(), E> {
        for example in &self.examples {
            for ordinal in &example.point_ordinals {
                let receipt = &self.manifest.prediction_points[*ordinal];
                visit(receipt, &example.token_ids[..receipt.prefix_token_count])?;
            }
        }
        Ok(())
    }

    pub(crate) fn visit_greedy_prompts<E>(
        &self,
        mut visit: impl FnMut(&TeacherGreedyPromptReceipt, &[u32]) -> Result<(), E>,
    ) -> Result<(), E> {
        for example in &self.examples {
            if let Some(ordinal) = example.greedy_prompt_ordinal {
                let receipt = &self.manifest.greedy_prompts[ordinal];
                visit(receipt, &example.token_ids)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum CalibrationInputError {
    #[error("invalid structured dataset: {0}")]
    InvalidDataset(String),
    #[error("chat-template resolution failed: {0}")]
    Template(#[from] crate::core::chat_template_resolver::ChatTemplateResolveError),
    #[error("no chat template is available for architecture {0}")]
    MissingTemplate(String),
    #[error("read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse {path}: {detail}")]
    Parse { path: PathBuf, detail: String },
    #[error("render example {stable_id}: {detail}")]
    Render { stable_id: String, detail: String },
    #[error("tokenize example {stable_id}: {detail}")]
    Tokenize { stable_id: String, detail: String },
    #[error("example {stable_id} has {tokens} tokens, above the bound {maximum}")]
    TokenLimit {
        stable_id: String,
        tokens: usize,
        maximum: usize,
    },
    #[error("example {0} contains media; dense text calibration is text-only")]
    MediaUnsupported(String),
    #[error("completed transcript example {0} must end with an assistant message")]
    MissingAssistantTarget(String),
    #[error("completed transcript prefix is not a token prefix for example {0}")]
    NonPrefixAssistantTarget(String),
    #[error("source chat-template hash does not match resolved bytes")]
    SourceTemplateMismatch,
    #[error("source tokenizer-bundle hash does not match resolved bytes")]
    SourceTokenizerBundleMismatch,
    #[error("example {stable_id} has {tokens} tokens, below overlap-window width {width}")]
    TokenWindowTooShort {
        stable_id: String,
        tokens: usize,
        width: usize,
    },
    #[error("dataset split mismatch or duplicate split")]
    SplitMismatch,
    #[error("calibration, policy-validation, and holdout inputs overlap: source_records={source_records}, raw={raw}, rendered={rendered}, token_windows={token_windows}")]
    DatasetOverlap {
        source_records: usize,
        raw: usize,
        rendered: usize,
        token_windows: usize,
    },
    #[error("ordered evidence serialization failed: {0}")]
    Serialization(String),
}
