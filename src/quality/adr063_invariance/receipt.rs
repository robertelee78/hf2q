//! Closed diagnostic identity/coverage schema. Declared bindings are not self-attestation.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Family {
    Qwen38,
    Deepseek4,
    Qwen36,
    Gemma4,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum Stage {
    Input,
    Norm,
    Qkv,
    Query,
    Key,
    Value,
    Rope,
    RopeQuery,
    RopeKey,
    QuantizedKv,
    Attention,
    Residual,
    FfnNorm,
    RouterSelectionBias,
    RouterLogits,
    RouterIds,
    RouterWeights,
    DenseOutput,
    MoeOutput,
    FfnWriter,
    LayerOutput,
    CacheState,
    CacheKey,
    CacheValue,
    CacheKeyPacked,
    CacheValuePacked,
    CacheKeyNorm,
    CacheValueNorm,
    CacheConv,
    CacheRecurrent,
    CacheWindow,
    CacheCompressed,
    CacheIndexer,
    CacheMainKv,
    CacheMainScore,
    CacheIndexerState,
    CacheIndexerScore,
    Logits,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Scalar {
    F32,
    F16,
    Bf16,
    I32,
    U32,
    U8,
}
impl Scalar {
    pub fn bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::U8 => 1,
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub family: Family,
    pub source_commit: String,
    pub binary_sha256: String,
    pub native_version: String,
    pub native_checksum: String,
    pub model_sha256: String,
    pub tokenizer_sha256: String,
    pub template_sha256: String,
    pub config_sha256: String,
    pub token_vectors_sha256: String,
}
fn hex(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
impl Identity {
    pub fn validate(&self) -> Result<(), String> {
        if !hex(&self.source_commit, 40)
            || ![
                &self.binary_sha256,
                &self.native_checksum,
                &self.model_sha256,
                &self.tokenizer_sha256,
                &self.template_sha256,
                &self.config_sha256,
                &self.token_vectors_sha256,
            ]
            .iter()
            .all(|s| hex(s, 64))
            || self.native_version != env!("HF2Q_MLX_NATIVE_VERSION")
        {
            return Err("invalid complete diagnostic identity".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Coordinate {
    pub logical_step: u64,
    pub layer: u32,
    pub stage: Stage,
    pub agent: String,
    pub logical_row_start: u64,
    pub segment: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureTarget {
    pub at: Coordinate,
    pub normalize_rows: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Page {
    pub offset: u64,
    pub bytes: usize,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub at: Coordinate,
    pub scalar: Scalar,
    pub shape: Vec<usize>,
    pub logical_bytes: u64,
    pub parent_identity: u64,
    pub parent_bytes: u64,
    pub parent_offset: u64,
    pub submission_id: Option<u64>,
    pub checked_drain: String,
    pub file: String,
    pub sha256: String,
    pub pages: Vec<Page>,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedTensor {
    pub at: Coordinate,
    pub scalar: Scalar,
    pub shape: Vec<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Coverage {
    pub expected: Vec<ExpectedTensor>,
    pub captured: Vec<Tensor>,
    pub errors: Vec<String>,
    pub missing_native_capabilities: Vec<String>,
    pub trajectory: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    pub schema: String,
    pub identity: Identity,
    pub comparison_key: super::acquisition::ComparisonKey,
    pub acquisition: Option<super::acquisition::Acquisition>,
    pub coverage: Coverage,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum NumericPolicy {
    Exact,
    Unspecified,
    SourceQualified {
        max_abs: f64,
        relative_rms: f64,
        source: String,
    },
}
impl NumericPolicy {
    pub fn validate(&self) -> Result<(), String> {
        if let Self::SourceQualified {
            max_abs,
            relative_rms,
            source,
        } = self
        {
            if !max_abs.is_finite()
                || *max_abs < 0.0
                || !relative_rms.is_finite()
                || *relative_rms < 0.0
                || source.trim().is_empty()
            {
                return Err("invalid numeric policy".into());
            }
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Disposition {
    ExactEqual,
    WithinSourceQualifiedPolicy,
    DivergedExactInvariant,
    DivergedSourceQualifiedPolicy,
    ExplicitInconclusiveNumericPolicy,
    LocalizationIncomplete,
    IdentityMismatch,
    InvalidEvidence,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Difference {
    pub at: Coordinate,
    pub logical_element: u64,
    pub logical_index: Vec<u64>,
    pub cache_token_row: Option<u64>,
    pub left_bits: String,
    pub right_bits: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Comparison {
    pub disposition: Disposition,
    pub first_difference: Option<Difference>,
    pub compared_bytes: u64,
    pub compared_elements: u64,
    pub max_abs: f64,
    pub rms: f64,
    pub relative_rms: Option<f64>,
    pub cosine: Option<f64>,
    pub nonfinite_elements: u64,
    pub reason: Option<String>,
    pub oracle_qualified: bool,
    pub runtime_acquisition_qualified: bool,
    pub captured_extent_qualified: bool,
}
