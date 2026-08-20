use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const TENSOR_EXECUTION_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const TENSOR_LAYOUT_ROW_MAJOR_OUTERMOST_FIRST_V1: &str = "row-major-outermost-first-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorStateStage {
    Source,
    Converted,
    Stored,
    Loaded,
    Executed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorScalarDType {
    F16,
    Bf16,
    F32,
}

/// Stable workload class shared by lineage, capability, and allocator
/// evidence. Exact M/N/K shapes remain in the typed capability binding, so a
/// logical operation may have multiple bindings for one regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorExecutionRegime {
    TextPrefill,
    TextDecodeM1,
    TextDecodeWidthN,
    LongContextDecode,
    MultimodalPrefill,
}

/// Physical representation of one tensor state in the lineage DAG.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PhysicalTensorCodec {
    Dense {
        dtype: TensorScalarDType,
    },
    Ggml {
        wire_type_id: u32,
        type_name: String,
    },
    MlxAffine {
        bits: u8,
        group_size: u16,
        scale_dtype: TensorScalarDType,
        bias_dtype: TensorScalarDType,
        layout_abi: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactEvidence {
    pub artifact_id: String,
    pub role: String,
    pub byte_len: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRegion {
    pub artifact_id: String,
    pub byte_offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorStateNode {
    pub node_id: String,
    pub stage: TensorStateStage,
    pub semantic_name: String,
    /// Logical tensor order for this state, never an implicit backend order.
    /// Schema v1 accepts only `row-major-outermost-first-v1`; backends must
    /// project their native dimension order into this canonical order.
    pub shape: Vec<u64>,
    pub layout: String,
    pub codec: PhysicalTensorCodec,
    pub byte_len: u64,
    pub byte_sha256: String,
    /// SHA-256 over the explicitly framed canonical F32 logical values.
    pub logical_f32_sha256: String,
    pub artifact_region: Option<ArtifactRegion>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TensorTransformOperation {
    SourceDecode,
    ArchitectureBake {
        operation: String,
        parameters_sha256: String,
    },
    Concatenate {
        axis: u32,
    },
    F16Roundtrip,
    GgufQuantize {
        implementation_id: String,
        calibration_receipt_sha256: Option<String>,
    },
    /// Encode a canonical converted F32 tensor as an unquantized GGUF wire
    /// tensor. Quantized GGML block encodings must use `GgufQuantize`.
    GgufDenseStore {
        implementation_id: String,
    },
    GgufDequantize {
        implementation_id: String,
    },
    SplitInterleavedQGate {
        implementation_id: String,
        heads: u32,
        head_dim: u32,
        hidden_size: u32,
    },
    Transpose {
        permutation: Vec<u32>,
    },
    /// Remove one exact singleton dimension from a converted dense tensor.
    /// This represents Qwen DeltaNet conv1d `BakeOp::Squeeze` without a
    /// generic rank-changing escape hatch.
    Squeeze {
        axis: u32,
    },
    ZeroPad {
        axis: u32,
        before: u64,
        after: u64,
    },
    Qwen35LoadQ4Amax7V1,
    DirectBlockLoad,
    RuntimeBind,
}

/// One ordered physical transformation. Input/output order is semantic and is
/// represented by named ports, so top-level serialization order is immaterial.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformPort {
    pub role: String,
    pub node_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorTransformEdge {
    pub edge_id: String,
    pub inputs: Vec<TransformPort>,
    pub outputs: Vec<TransformPort>,
    pub operation: TensorTransformOperation,
    pub implementation_revision: String,
    pub receipt_sha256: String,
}

/// Canonical JSON with a separately checked digest. Schema v1 validates only
/// its bytes and digest; D2b must deserialize and recompute the exact
/// mlx-native request/decision types before this becomes runtime evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalJsonEvidence {
    pub canonical_json: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RuntimeCapabilityEvidence {
    Ggml {
        request: CanonicalJsonEvidence,
        decision: CanonicalJsonEvidence,
        requires_device_probe: bool,
        resolved_runtime_trace: Option<CanonicalJsonEvidence>,
    },
    NonGgml {
        implementation_id: String,
        contract_sha256: String,
        resolved_runtime_trace: Option<CanonicalJsonEvidence>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeOperationBinding {
    pub binding_id: String,
    pub operation_id: String,
    pub graph_path: String,
    pub entrypoint: String,
    pub workload_regime: TensorExecutionRegime,
    /// Number of identical invocations represented by this exact physical
    /// binding in the proposal workload. Regime measurements cover the
    /// aggregate of all bound invocations.
    pub invocation_count: u64,
    /// Exact source-tensor closure of the physical inputs.
    pub source_tensor_names: Vec<String>,
    pub inputs: Vec<TransformPort>,
    pub capability: RuntimeCapabilityEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceTensorDispositionKind {
    Variable,
    Fixed,
    Protected,
    Excluded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceTensorDisposition {
    pub source_node_id: String,
    pub disposition: SourceTensorDispositionKind,
    pub reason: String,
    pub terminal_node_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorRuntimeBinding {
    pub hf2q_revision: String,
    pub mlx_native_version: String,
    pub mlx_native_capability_schema_version: u32,
    pub routing_policy_sha256: String,
    pub graph_configuration_sha256: String,
    /// Exact capability/measurement profile used by allocator operation
    /// evidence. This is distinct from the broader host identity below.
    pub capability_profile_sha256: String,
    pub hardware_profile_sha256: String,
    /// D2/no-DWQ contract. Any overlay produces a different, unsupported
    /// execution graph and must fail admission.
    pub dwq_overlay_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorExecutionScope {
    pub model_family: String,
    pub profile: String,
    pub included_paths: Vec<String>,
    pub excluded_paths: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorExecutionManifest {
    pub schema_version: u32,
    pub source_manifest_sha256: String,
    pub source_tensor_inventory_sha256: String,
    pub tensor_partition_manifest_sha256: String,
    pub conversion_receipt_sha256: String,
    pub logical_hash_encoding: String,
    pub runtime: TensorRuntimeBinding,
    pub scope: TensorExecutionScope,
    pub artifacts: Vec<ArtifactEvidence>,
    pub nodes: Vec<TensorStateNode>,
    pub transforms: Vec<TensorTransformEdge>,
    pub operations: Vec<RuntimeOperationBinding>,
    pub dispositions: Vec<SourceTensorDisposition>,
    pub manifest_sha256: String,
}

/// Canonical verifier-derived projection for one source tensor. Callers must
/// never supply shadow node/byte/operation fields alongside this projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorLineageSlice {
    pub execution_manifest_sha256: String,
    pub source_tensor_name: String,
    pub nodes: Vec<TensorStateNode>,
    pub transforms: Vec<TensorTransformEdge>,
    pub operations: Vec<RuntimeOperationBinding>,
    pub slice_sha256: String,
}
