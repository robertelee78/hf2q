use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::intelligence::measured_auto_quant::SourceIdentity;
use crate::serve::api::schema::{ChatMessage, Tool};

pub const CALIBRATION_INPUT_SCHEMA_VERSION: u32 = 1;

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
