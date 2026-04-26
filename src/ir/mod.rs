//! Intermediate Representation — the central data contract for hf2q.
//!
//! Input produces `TensorMap` + `ModelMetadata`.
//! Quantize transforms `TensorMap` into `QuantizedModel`.
//! Backends consume `QuantizedModel`.
//!
//! All types are Send + Sync.
//!
//! ## Lazy IR (ADR-014 P0)
//!
//! The submodule [`lazy`] adds [`lazy::LazyTensor`] / [`lazy::LazyTensorMap`]:
//! a `FnOnce`-backed deferred materialization layer for the streaming
//! convert pipeline (ADR-014 Decisions 1+2). Eager [`TensorMap`] remains
//! the contract for P1-pre callers; ADR-014 P1 lifts the Phase 1.4–1.7
//! transforms to consume `LazyTensorMap`. During P0 the eager
//! `read_tensors` reader is implemented as
//! `read_tensors_lazy(...).materialize_all()` (Decision 2 bridge), so
//! the legacy callers continue to see byte-identical output while the
//! new lazy primitive becomes the source of truth.

pub mod lazy;

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from IR operations.
#[derive(Error, Debug)]
pub enum IrError {
    #[error("Unsupported dtype for conversion: {dtype}")]
    UnsupportedDtype { dtype: String },

    #[error("Tensor '{name}' has invalid shape: expected {expected} elements, got {actual}")]
    ShapeMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[allow(dead_code)]
    #[error("bf16 to f16 conversion failed for tensor '{name}': {reason}")]
    ConversionFailed { name: String, reason: String },
}

/// Data type of tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U8,
    U16,
    U32,
    Bool,
}

impl DType {
    /// Size in bytes of a single element.
    pub fn element_size(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 | DType::U16 => 2,
            DType::I64 => 8,
            DType::U8 | DType::Bool => 1,
        }
    }

    /// Parse a dtype string from safetensors metadata.
    pub fn from_safetensors_str(s: &str) -> Option<DType> {
        match s {
            "F32" => Some(DType::F32),
            "F16" => Some(DType::F16),
            "BF16" => Some(DType::BF16),
            "I32" => Some(DType::I32),
            "I64" => Some(DType::I64),
            "U8" => Some(DType::U8),
            "U16" => Some(DType::U16),
            "U32" => Some(DType::U32),
            "BOOL" => Some(DType::Bool),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "F32"),
            DType::F16 => write!(f, "F16"),
            DType::BF16 => write!(f, "BF16"),
            DType::I32 => write!(f, "I32"),
            DType::I64 => write!(f, "I64"),
            DType::U8 => write!(f, "U8"),
            DType::U16 => write!(f, "U16"),
            DType::U32 => write!(f, "U32"),
            DType::Bool => write!(f, "BOOL"),
        }
    }
}

/// A reference to a tensor that can provide lazy access to its data via mmap.
#[derive(Debug, Clone)]
pub struct TensorRef {
    /// Fully qualified tensor name (e.g., "model.language_model.layers.0.self_attn.q_proj.weight")
    pub name: String,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data type of the tensor
    pub dtype: DType,
    /// Raw data bytes (may be mmap'd)
    pub data: Vec<u8>,
}

// Safety: TensorRef contains only owned data (Vec<u8>) — inherently Send + Sync.
unsafe impl Send for TensorRef {}
unsafe impl Sync for TensorRef {}

impl TensorRef {
    /// Total number of elements in this tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes of this tensor's data.
    #[allow(dead_code)]
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.element_size()
    }

    /// Whether this tensor belongs to a vision encoder or multimodal projector.
    /// Vision tensors should be preserved at full precision (F16) regardless of
    /// whether they pass `is_weight()`, because quantizing vision components
    /// degrades image understanding quality significantly.
    pub fn is_vision_tensor(&self) -> bool {
        let n = &self.name;
        // Check raw HF names (before any prefix stripping)
        if n.contains("vision_tower") || n.contains("embed_vision") {
            return true;
        }
        // After language_model. prefix strip, these would start with vision_tower / embed_vision
        if let Some(rest) = n.strip_prefix("language_model.") {
            if rest.starts_with("vision_tower") || rest.starts_with("embed_vision") {
                return true;
            }
        }
        false
    }

    /// Whether this tensor is a weight tensor (as opposed to a norm, bias, scalar, etc.)
    /// Used to decide what to quantize vs preserve at full precision.
    pub fn is_weight(&self) -> bool {
        let n = &self.name;
        // Weight tensors are multi-dimensional projections
        // Non-weight: layernorm, rmsnorm, bias, scalar, router scale, embeddings
        if n.contains("layernorm") || n.contains("layer_norm") || n.contains("_norm.weight") {
            return false;
        }
        if n.contains("bias") {
            return false;
        }
        if n.contains("layer_scalar") || n.contains("router.scale") || n.contains("router.per_expert_scale") {
            return false;
        }
        if n.contains("embed_tokens") || n.contains("embedding_projection") {
            return false;
        }
        // ADR-012 P9b real-model finding (2026-04-25): tensors with a small
        // inner-dim (< 32 = Q4_0 block size) cannot be block-quantized at all
        // — Q4_0/Q5_0/Q8_0 require row_dim divisible by 32, K-quants by 256.
        // ssm_conv1d.weight (shape [channels, K=4]) and similar small-kernel
        // tensors must be preserved at F16/F32. Without this gate the DWQ
        // pipeline emits a Q4_0 ssm_conv1d which llama.cpp rejects with
        //   "tensor 'blk.0.ssm_conv1d.weight' of type 2 (q4_0) has 4 elements
        //    per row, not a multiple of block size (32)"
        if self.shape.len() >= 2 {
            let row_dim = *self.shape.last().unwrap();
            if row_dim < 32 {
                return false;
            }
        }
        // Multi-dimensional tensors with "weight" or "proj" in the name are quantizable
        if self.shape.len() >= 2 {
            return n.contains("weight") || n.contains("proj") || n.contains("experts.");
        }
        false
    }

    /// Convert bf16 data to f16 in-place, returning a new TensorRef.
    pub fn to_f16(&self) -> Result<TensorRef, IrError> {
        if self.dtype == DType::F16 {
            return Ok(self.clone());
        }
        if self.dtype != DType::BF16 {
            return Err(IrError::UnsupportedDtype {
                dtype: self.dtype.to_string(),
            });
        }

        let element_count = self.numel();
        let expected_bytes = element_count * 2;
        if self.data.len() != expected_bytes {
            return Err(IrError::ShapeMismatch {
                name: self.name.clone(),
                expected: expected_bytes,
                actual: self.data.len(),
            });
        }

        let mut f16_data = Vec::with_capacity(expected_bytes);

        // Convert bf16 -> f32 -> f16 using the half crate
        for chunk in self.data.chunks_exact(2) {
            let bf16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let bf16_val = half::bf16::from_bits(bf16_bits);
            let f32_val: f32 = bf16_val.to_f32();
            let f16_val = half::f16::from_f32(f32_val);
            f16_data.extend_from_slice(&f16_val.to_le_bytes());
        }

        Ok(TensorRef {
            name: self.name.clone(),
            shape: self.shape.clone(),
            dtype: DType::F16,
            data: f16_data,
        })
    }
}

/// A map of tensor names to their references. The central data structure for model weights.
#[derive(Debug)]
pub struct TensorMap {
    pub tensors: HashMap<String, TensorRef>,
}

impl TensorMap {
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
        }
    }

    /// Insert a tensor into the map.
    pub fn insert(&mut self, tensor: TensorRef) {
        self.tensors.insert(tensor.name.clone(), tensor);
    }

    /// Get a tensor by name.
    #[allow(dead_code)]
    pub fn get(&self, name: &str) -> Option<&TensorRef> {
        self.tensors.get(name)
    }

    /// Number of tensors in the map.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the tensor map is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Iterate over all tensors.
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorRef)> {
        self.tensors.iter()
    }

    /// Total size of all tensors in bytes.
    pub fn total_size_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.data.len()).sum()
    }

    /// Convert all bf16 tensors to f16.
    pub fn convert_bf16_to_f16(&mut self) -> Result<usize, IrError> {
        let bf16_names: Vec<String> = self
            .tensors
            .iter()
            .filter(|(_, t)| t.dtype == DType::BF16)
            .map(|(name, _)| name.clone())
            .collect();

        let count = bf16_names.len();
        for name in bf16_names {
            if let Some(tensor) = self.tensors.remove(&name) {
                let converted = tensor.to_f16()?;
                self.tensors.insert(name, converted);
            }
        }
        Ok(count)
    }
}

impl Default for TensorMap {
    fn default() -> Self {
        Self::new()
    }
}

/// RoPE (Rotary Position Embedding) parameters for hybrid architectures.
///
/// Qwen3.5-family models embed these as a nested `rope_parameters` object in config.json.
/// All fields are optional to preserve Chesterton's fence for existing architectures.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct RopeParameters {
    /// Whether interleaved MROPE is used (Qwen3.5 uses true).
    #[serde(default)]
    pub mrope_interleaved: bool,
    /// MROPE section sizes: [temporal, height, width] split of the head_dim/2 positions.
    /// For Qwen3.5-MoE apex: [11, 11, 10].
    #[serde(default)]
    pub mrope_section: Vec<u32>,
    /// Base frequency for RoPE. For Qwen3.5-MoE: 10_000_000.
    #[serde(default)]
    pub rope_theta: f64,
    /// RoPE variant string (e.g. "default", "linear", "dynamic").
    #[serde(default)]
    pub rope_type: String,
    /// Fraction of head_dim rotated. Qwen3.5 partial-rotary: 0.25.
    #[serde(default)]
    pub partial_rotary_factor: f32,
}

/// Metadata extracted from a HuggingFace model's config.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Architecture name (e.g., "Gemma4ForConditionalGeneration")
    pub architecture: String,
    /// Model type (e.g., "gemma4")
    pub model_type: String,
    /// Total parameter count
    pub param_count: u64,
    /// Hidden size of the model
    pub hidden_size: u64,
    /// Number of transformer layers
    pub num_layers: u32,
    /// Layer types (e.g., ["sliding_attention", "full_attention"]).
    /// For Qwen3.5-MoE: 40-element vec alternating linear_attention/full_attention.
    /// Populated by `resolved_layer_types()` logic in the parser.
    pub layer_types: Vec<String>,
    /// Number of attention heads
    pub num_attention_heads: u32,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: Option<u32>,
    /// Vocabulary size
    pub vocab_size: u64,
    /// Model dtype (as stated in config.json)
    pub dtype: String,
    /// Number of safetensors shards
    pub shard_count: u32,
    /// Number of experts (for MoE models)
    pub num_experts: Option<u32>,
    /// Top-k experts used per token (for MoE models)
    pub top_k_experts: Option<u32>,
    /// Intermediate (FFN) size
    pub intermediate_size: Option<u64>,
    /// All raw config values for passthrough to output
    pub raw_config: serde_json::Value,

    // --- ADR-012 Decision 2: Qwen3.5-family extended fields ---

    /// Explicit per-layer attention type enumeration (preferred over full_attention_interval).
    /// None when the config omits it (e.g. Gemma4 — Chesterton's fence: no behavior change).
    pub explicit_layer_types: Option<Vec<String>>,

    /// Computed layer-type interval: every N-th layer is full_attention, rest are linear_attention.
    /// Used as fallback when `explicit_layer_types` is absent.
    pub full_attention_interval: Option<u32>,

    /// Whether the attention output has a gating projection (Qwen3.5 DeltaNet).
    pub attn_output_gate: Option<bool>,

    /// Per-attention-head dimension. Explicitly parsed — NEVER derived from hidden_size/num_heads.
    /// Qwen3.5-MoE apex: 256. May differ from hidden_size/num_attention_heads.
    pub head_dim: Option<u32>,

    /// Fraction of head_dim that is rotated (top-level field, may duplicate rope_parameters).
    pub partial_rotary_factor: Option<f32>,

    /// Nested RoPE configuration object (Qwen3.5-family).
    pub rope_parameters: Option<RopeParameters>,

    // Linear-attention (Gated DeltaNet) kernel dimensions:

    /// Convolution kernel width for the linear-attention SSM state.
    pub linear_conv_kernel_dim: Option<u32>,
    /// Head dimension for linear-attention key projections.
    pub linear_key_head_dim: Option<u32>,
    /// Number of key heads in linear-attention layers.
    pub linear_num_key_heads: Option<u32>,
    /// Head dimension for linear-attention value projections.
    pub linear_value_head_dim: Option<u32>,
    /// Number of value heads in linear-attention layers.
    pub linear_num_value_heads: Option<u32>,

    /// dtype used for SSM state in Mamba/DeltaNet kernels.
    /// Validated as one of: "float32", "bfloat16", "float16".
    pub mamba_ssm_dtype: Option<String>,

    // MoE sizing (Qwen3.5-MoE):

    /// Size of each expert's FFN intermediate layer.
    pub moe_intermediate_size: Option<u32>,
    /// Intermediate size of the always-active shared expert (Qwen3.5-MoE).
    pub shared_expert_intermediate_size: Option<u32>,

    // Multi-Token Prediction (MTP) fields:

    /// Number of hidden layers in the MTP draft head.
    pub mtp_num_hidden_layers: Option<u32>,
    /// Whether the MTP head uses its own embedding table.
    pub mtp_use_dedicated_embeddings: Option<bool>,

    // Router fields:

    /// Whether to output router logits in the forward pass (training-time flag).
    pub output_router_logits: Option<bool>,
    /// Auxiliary load-balancing loss coefficient.
    pub router_aux_loss_coef: Option<f32>,
}

impl ModelMetadata {
    /// Unique layer types (deduplicated).
    pub fn unique_layer_types(&self) -> Vec<String> {
        let mut types: Vec<String> = self.layer_types.clone();
        types.sort();
        types.dedup();
        types
    }

    /// Whether this is a Mixture of Experts model.
    pub fn is_moe(&self) -> bool {
        self.num_experts.is_some() && self.num_experts.unwrap_or(0) > 1
    }

    /// Resolved layer type list for hybrid architectures (ADR-012 Decision 2).
    ///
    /// Preference order:
    /// 1. `explicit_layer_types` — when the config contains an explicit `layer_types` array.
    /// 2. Derive from `full_attention_interval` — every N-th layer (0-indexed) is
    ///    `"full_attention"`, all others are `"linear_attention"`.
    /// 3. Fall back to `layer_types` as populated at parse time (Gemma / llama / etc.).
    ///
    /// Callers in P2+ should use this rather than `layer_types` directly when they
    /// need to know whether a specific layer is linear or full attention.
    pub fn resolved_layer_types(&self) -> Vec<String> {
        // Prefer explicit enumeration
        if let Some(explicit) = &self.explicit_layer_types {
            return explicit.clone();
        }
        // Derive from interval
        if let Some(interval) = self.full_attention_interval {
            let n = self.num_layers as usize;
            if n > 0 && interval > 0 {
                return (0..n)
                    .map(|i| {
                        // interval-th layer (1-indexed): layers at positions interval-1, 2*interval-1, …
                        if (i + 1) % interval as usize == 0 {
                            "full_attention".to_string()
                        } else {
                            "linear_attention".to_string()
                        }
                    })
                    .collect();
            }
        }
        // Fall back to whatever the parser set in layer_types
        self.layer_types.clone()
    }
}

/// A quantized tensor — the result of quantization.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QuantizedTensor {
    /// Original tensor name
    pub name: String,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Original dtype before quantization
    pub original_dtype: DType,
    /// Quantized data bytes
    pub data: Vec<u8>,
    /// Quantization metadata
    pub quant_info: TensorQuantInfo,
}

/// Per-tensor quantization metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorQuantInfo {
    /// Quantization method applied (e.g., "q4", "f16", "passthrough")
    pub method: String,
    /// Bit width used
    pub bits: u8,
    /// Group size used
    pub group_size: usize,
    /// Whether this tensor was preserved at full precision
    pub preserved: bool,
    /// Scale factors (for dequantization)
    pub scales: Option<Vec<u8>>,
    /// Zero points (for asymmetric quantization) — stored as raw bytes
    pub biases: Option<Vec<u8>>,
    /// Optional exact GGML type name (e.g., "Q4_K_M", "Q6_K").
    /// When set, the GGUF backend uses this instead of the generic bits-based mapping.
    /// Used by Apex quantization to assign per-tensor optimal K-quant types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ggml_type: Option<String>,
}

/// A fully quantized model, ready for output backend consumption.
#[derive(Debug)]
pub struct QuantizedModel {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Quantized tensors
    pub tensors: HashMap<String, QuantizedTensor>,
    /// Global quantization config
    pub quant_method: String,
    /// Global group size
    pub group_size: usize,
    /// Global bit width
    pub bits: u8,
}

impl QuantizedModel {
    /// Total size of all quantized tensors in bytes.
    #[allow(dead_code)]
    pub fn total_size_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.data.len()).sum()
    }

    /// Number of tensors.
    #[allow(dead_code)]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

/// Manifest produced by an output backend after writing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputManifest {
    /// Output directory path
    pub output_dir: String,
    /// List of files written
    pub files: Vec<OutputFile>,
    /// Total output size in bytes
    pub total_size_bytes: u64,
    /// Number of output shards
    pub shard_count: usize,
}

/// A single file in the output manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputFile {
    pub filename: String,
    pub size_bytes: u64,
}

/// Format-specific warnings from backend validation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FormatWarning {
    pub message: String,
    pub severity: WarningSeverity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum WarningSeverity {
    Info,
    Warning,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_element_size() {
        assert_eq!(DType::F32.element_size(), 4);
        assert_eq!(DType::F16.element_size(), 2);
        assert_eq!(DType::BF16.element_size(), 2);
        assert_eq!(DType::U8.element_size(), 1);
    }

    #[test]
    fn test_dtype_from_safetensors_str() {
        assert_eq!(DType::from_safetensors_str("F32"), Some(DType::F32));
        assert_eq!(DType::from_safetensors_str("BF16"), Some(DType::BF16));
        assert_eq!(DType::from_safetensors_str("UNKNOWN"), None);
    }

    #[test]
    fn test_tensor_ref_numel() {
        let t = TensorRef {
            name: "test".to_string(),
            shape: vec![3, 4, 5],
            dtype: DType::F32,
            data: vec![0u8; 3 * 4 * 5 * 4],
        };
        assert_eq!(t.numel(), 60);
        assert_eq!(t.size_bytes(), 240);
    }

    #[test]
    fn test_tensor_ref_is_weight() {
        let weight = TensorRef {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            shape: vec![4096, 4096],
            dtype: DType::F16,
            data: vec![],
        };
        assert!(weight.is_weight());

        let norm = TensorRef {
            name: "model.layers.0.input_layernorm.weight".to_string(),
            shape: vec![4096],
            dtype: DType::F16,
            data: vec![],
        };
        assert!(!norm.is_weight());

        let bias = TensorRef {
            name: "model.layers.0.self_attn.o_proj.bias".to_string(),
            shape: vec![4096],
            dtype: DType::F16,
            data: vec![],
        };
        assert!(!bias.is_weight());
    }

    #[test]
    fn test_bf16_to_f16_conversion() {
        // Create a small bf16 tensor with known value: 1.0
        // bf16 1.0 = 0x3F80
        let bf16_one = half::bf16::from_f32(1.0);
        let bytes = bf16_one.to_le_bytes();

        let tensor = TensorRef {
            name: "test".to_string(),
            shape: vec![1],
            dtype: DType::BF16,
            data: bytes.to_vec(),
        };

        let converted = tensor.to_f16().unwrap();
        assert_eq!(converted.dtype, DType::F16);
        assert_eq!(converted.data.len(), 2);

        let f16_bits = u16::from_le_bytes([converted.data[0], converted.data[1]]);
        let f16_val = half::f16::from_bits(f16_bits);
        assert!((f16_val.to_f32() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_tensor_map_operations() {
        let mut map = TensorMap::new();
        assert!(map.is_empty());

        map.insert(TensorRef {
            name: "a".to_string(),
            shape: vec![2, 3],
            dtype: DType::F16,
            data: vec![0u8; 12],
        });

        assert_eq!(map.len(), 1);
        assert!(map.get("a").is_some());
        assert!(map.get("b").is_none());
    }

    #[test]
    fn test_is_vision_tensor() {
        let make = |name: &str| TensorRef {
            name: name.to_string(),
            shape: vec![4096, 4096],
            dtype: DType::F16,
            data: vec![],
        };

        // Vision tensors — should return true
        assert!(make("model.vision_tower.encoder.layers.0.self_attn.q_proj.weight").is_vision_tensor());
        assert!(make("model.vision_tower.patch_embedder.input_proj.weight").is_vision_tensor());
        assert!(make("model.embed_vision.embedding_projection.weight").is_vision_tensor());

        // Non-vision tensors — should return false
        assert!(!make("model.layers.0.self_attn.q_proj.weight").is_vision_tensor());
        assert!(!make("model.embed_tokens.weight").is_vision_tensor());
    }

    #[test]
    fn test_vision_weight_tensor_classification() {
        // A vision weight tensor should pass both is_weight() and is_vision_tensor()
        let vt = TensorRef {
            name: "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight".to_string(),
            shape: vec![4096, 4096],
            dtype: DType::F16,
            data: vec![],
        };
        assert!(vt.is_weight(), "vision weight should pass is_weight()");
        assert!(vt.is_vision_tensor(), "vision weight should pass is_vision_tensor()");
    }

    #[test]
    fn test_model_metadata_moe() {
        let meta = ModelMetadata {
            architecture: "Test".to_string(),
            model_type: "test".to_string(),
            param_count: 1000,
            hidden_size: 256,
            num_layers: 4,
            layer_types: vec!["attention".to_string()],
            num_attention_heads: 8,
            num_kv_heads: None,
            vocab_size: 32000,
            dtype: "bfloat16".to_string(),
            shard_count: 1,
            num_experts: Some(128),
            top_k_experts: Some(8),
            intermediate_size: Some(512),
            raw_config: serde_json::Value::Null,
            // ADR-012 P1 fields: None for non-qwen35 models (Chesterton's fence)
            explicit_layer_types: None,
            full_attention_interval: None,
            attn_output_gate: None,
            head_dim: None,
            partial_rotary_factor: None,
            rope_parameters: None,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_num_key_heads: None,
            linear_value_head_dim: None,
            linear_num_value_heads: None,
            mamba_ssm_dtype: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            mtp_use_dedicated_embeddings: None,
            output_router_logits: None,
            router_aux_loss_coef: None,
        };
        assert!(meta.is_moe());
    }
}
