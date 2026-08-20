use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::types::*;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TensorExecutionManifestError {
    #[error("tensor execution manifest serialization failed: {0}")]
    Serialization(String),
    #[error("tensor execution manifest is invalid: {0}")]
    Invalid(String),
    #[error("tensor execution manifest digest mismatch")]
    DigestMismatch,
}

#[derive(Debug, Clone)]
/// Structurally validated evidence. D2b's source/GGUF/runtime producer is
/// responsible for independently rehashing the referenced physical bytes
/// before this can become production execution evidence.
pub struct ValidatedTensorExecutionManifest {
    manifest: TensorExecutionManifest,
}

impl ValidatedTensorExecutionManifest {
    pub fn manifest(&self) -> &TensorExecutionManifest {
        &self.manifest
    }

    pub fn into_manifest(self) -> TensorExecutionManifest {
        self.manifest
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn normalize_json(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut sorted = BTreeMap::new();
            for (key, value) in object {
                sorted.insert(key, normalize_json(value));
            }
            let mut normalized = serde_json::Map::new();
            for (key, value) in sorted {
                normalized.insert(key, value);
            }
            Value::Object(normalized)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(normalize_json).collect()),
        other => other,
    }
}

fn validate_json_evidence(evidence: &CanonicalJsonEvidence) -> Result<(), String> {
    if !is_sha256(&evidence.sha256) {
        return Err("canonical JSON evidence has an invalid digest".into());
    }
    let parsed: Value = serde_json::from_str(&evidence.canonical_json)
        .map_err(|error| format!("canonical JSON evidence is invalid JSON: {error}"))?;
    let canonical = serde_json::to_string(&normalize_json(parsed))
        .map_err(|error| format!("canonical JSON evidence cannot serialize: {error}"))?;
    if canonical != evidence.canonical_json
        || hex::encode(Sha256::digest(canonical.as_bytes())) != evidence.sha256
    {
        return Err("canonical JSON evidence bytes or digest do not match".into());
    }
    Ok(())
}

fn normalize_manifest(manifest: &TensorExecutionManifest) -> TensorExecutionManifest {
    let mut normalized = manifest.clone();
    normalized
        .artifacts
        .sort_by(|left, right| left.artifact_id.cmp(&right.artifact_id));
    normalized
        .nodes
        .sort_by(|left, right| left.node_id.cmp(&right.node_id));
    normalized
        .transforms
        .sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    normalized
        .operations
        .sort_by(|left, right| left.binding_id.cmp(&right.binding_id));
    normalized
        .dispositions
        .sort_by(|left, right| left.source_node_id.cmp(&right.source_node_id));
    normalized.scope.included_paths.sort();
    for disposition in &mut normalized.dispositions {
        disposition.terminal_node_ids.sort();
    }
    for edge in &mut normalized.transforms {
        edge.inputs
            .sort_by(|left, right| left.role.cmp(&right.role));
        edge.outputs
            .sort_by(|left, right| left.role.cmp(&right.role));
    }
    for operation in &mut normalized.operations {
        operation.source_tensor_names.sort();
        operation
            .inputs
            .sort_by(|left, right| left.role.cmp(&right.role));
    }
    normalized
}

pub fn canonicalized_tensor_execution_manifest(
    manifest: &TensorExecutionManifest,
) -> TensorExecutionManifest {
    normalize_manifest(manifest)
}

pub fn canonical_tensor_execution_manifest_bytes(
    manifest: &TensorExecutionManifest,
) -> Result<Vec<u8>, TensorExecutionManifestError> {
    let mut normalized = normalize_manifest(manifest);
    normalized.manifest_sha256.clear();
    serde_json::to_vec(&normalized)
        .map_err(|error| TensorExecutionManifestError::Serialization(error.to_string()))
}

pub fn tensor_execution_manifest_sha256(
    manifest: &TensorExecutionManifest,
) -> Result<String, TensorExecutionManifestError> {
    Ok(hex::encode(Sha256::digest(
        canonical_tensor_execution_manifest_bytes(manifest)?,
    )))
}

pub fn unique_stored_payload_bytes(
    validated: &ValidatedTensorExecutionManifest,
    node_ids: &[String],
) -> Result<u64, TensorExecutionManifestError> {
    let nodes: BTreeMap<_, _> = validated
        .manifest
        .nodes
        .iter()
        .map(|node| (node.node_id.as_str(), node))
        .collect();
    let mut unique = BTreeSet::new();
    let mut total = 0u64;
    for node_id in node_ids {
        if !unique.insert(node_id.as_str()) {
            continue;
        }
        let node = nodes.get(node_id.as_str()).ok_or_else(|| {
            TensorExecutionManifestError::Invalid(format!(
                "stored payload references unknown node `{node_id}`"
            ))
        })?;
        if node.stage != TensorStateStage::Stored {
            return Err(TensorExecutionManifestError::Invalid(format!(
                "payload node `{node_id}` is not stored"
            )));
        }
        total = total.checked_add(node.byte_len).ok_or_else(|| {
            TensorExecutionManifestError::Invalid("stored payload byte sum overflows u64".into())
        })?;
    }
    Ok(total)
}

fn normalize_lineage_slice(slice: &TensorLineageSlice) -> TensorLineageSlice {
    let mut normalized = slice.clone();
    normalized
        .nodes
        .sort_by(|left, right| left.node_id.cmp(&right.node_id));
    normalized
        .transforms
        .sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    normalized
        .operations
        .sort_by(|left, right| left.binding_id.cmp(&right.binding_id));
    for edge in &mut normalized.transforms {
        edge.inputs
            .sort_by(|left, right| left.role.cmp(&right.role));
        edge.outputs
            .sort_by(|left, right| left.role.cmp(&right.role));
    }
    for operation in &mut normalized.operations {
        operation.source_tensor_names.sort();
        operation
            .inputs
            .sort_by(|left, right| left.role.cmp(&right.role));
    }
    normalized
}

pub fn canonical_tensor_lineage_slice_bytes(
    slice: &TensorLineageSlice,
) -> Result<Vec<u8>, TensorExecutionManifestError> {
    let mut normalized = normalize_lineage_slice(slice);
    normalized.slice_sha256.clear();
    serde_json::to_vec(&normalized)
        .map_err(|error| TensorExecutionManifestError::Serialization(error.to_string()))
}

pub fn tensor_lineage_slice_sha256(
    slice: &TensorLineageSlice,
) -> Result<String, TensorExecutionManifestError> {
    Ok(hex::encode(Sha256::digest(
        canonical_tensor_lineage_slice_bytes(slice)?,
    )))
}

/// Hash the complete physical identity of one state node. Allocator evidence
/// must use this projection instead of repeating selected node fields.
pub fn tensor_state_node_sha256(
    node: &TensorStateNode,
) -> Result<String, TensorExecutionManifestError> {
    let bytes = serde_json::to_vec(node)
        .map_err(|error| TensorExecutionManifestError::Serialization(error.to_string()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn normalize_runtime_operation(operation: &RuntimeOperationBinding) -> RuntimeOperationBinding {
    let mut normalized = operation.clone();
    normalized.source_tensor_names.sort();
    normalized
        .inputs
        .sort_by(|left, right| left.role.cmp(&right.role));
    normalized
}

/// Hash every regime-specific physical binding for one stable logical
/// operation. This prevents an allocator option from substituting a cheaper
/// request, route, or executed tensor while retaining a real manifest hash.
pub fn runtime_capability_binding_bundle_sha256(
    validated: &ValidatedTensorExecutionManifest,
    operation_id: &str,
) -> Result<String, TensorExecutionManifestError> {
    let mut bindings: Vec<_> = validated
        .manifest()
        .operations
        .iter()
        .filter(|binding| binding.operation_id == operation_id)
        .map(normalize_runtime_operation)
        .collect();
    if bindings.is_empty() {
        return Err(TensorExecutionManifestError::Invalid(format!(
            "logical operation `{operation_id}` has no runtime bindings"
        )));
    }
    bindings.sort_by(|left, right| left.binding_id.cmp(&right.binding_id));
    let bytes = serde_json::to_vec(&bindings)
        .map_err(|error| TensorExecutionManifestError::Serialization(error.to_string()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

/// Hash the exact physical bindings for one logical operation and workload
/// regime. Regime cost evidence must reference this projection, never a
/// caller-authored route or shape label.
pub fn runtime_regime_binding_bundle_sha256(
    validated: &ValidatedTensorExecutionManifest,
    operation_id: &str,
    regime: TensorExecutionRegime,
) -> Result<String, TensorExecutionManifestError> {
    let mut bindings: Vec<_> = validated
        .manifest()
        .operations
        .iter()
        .filter(|binding| binding.operation_id == operation_id && binding.workload_regime == regime)
        .map(normalize_runtime_operation)
        .collect();
    if bindings.is_empty() {
        return Err(TensorExecutionManifestError::Invalid(format!(
            "logical operation `{operation_id}` has no {regime:?} runtime bindings"
        )));
    }
    bindings.sort_by(|left, right| left.binding_id.cmp(&right.binding_id));
    let bytes = serde_json::to_vec(&bindings)
        .map_err(|error| TensorExecutionManifestError::Serialization(error.to_string()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

pub fn tensor_lineage_slice(
    validated: &ValidatedTensorExecutionManifest,
    source_tensor_name: &str,
) -> Result<TensorLineageSlice, TensorExecutionManifestError> {
    let manifest = validated.manifest();
    let source = manifest
        .nodes
        .iter()
        .find(|node| {
            node.stage == TensorStateStage::Source && node.semantic_name == source_tensor_name
        })
        .ok_or_else(|| {
            TensorExecutionManifestError::Invalid(format!(
                "source tensor `{source_tensor_name}` is absent from execution manifest"
            ))
        })?;
    let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for edge in &manifest.transforms {
        for input in &edge.inputs {
            for output in &edge.outputs {
                adjacency
                    .entry(input.node_id.as_str())
                    .or_default()
                    .push(output.node_id.as_str());
            }
        }
    }
    let mut reachable = BTreeSet::new();
    let mut pending = vec![source.node_id.as_str()];
    while let Some(node_id) = pending.pop() {
        if reachable.insert(node_id) {
            if let Some(outputs) = adjacency.get(node_id) {
                pending.extend(outputs.iter().copied());
            }
        }
    }
    // Follow an edge when any input is reachable so either member of a fused
    // gate+up source pair discovers the shared output. Then close the set
    // backwards over every producer so the projection also contains all
    // co-input sources and producer edges. A physical node can therefore
    // never appear in a slice without the transformation that created it.
    let producer_by_output: BTreeMap<&str, &TensorTransformEdge> = manifest
        .transforms
        .iter()
        .flat_map(|edge| {
            edge.outputs
                .iter()
                .map(move |output| (output.node_id.as_str(), edge))
        })
        .collect();
    let mut closed = reachable.clone();
    let mut included_edges = BTreeSet::new();
    let mut pending: Vec<_> = reachable.iter().copied().collect();
    for operation in manifest.operations.iter().filter(|operation| {
        operation
            .source_tensor_names
            .iter()
            .any(|name| name == source_tensor_name)
    }) {
        for input in &operation.inputs {
            if closed.insert(input.node_id.as_str()) {
                pending.push(input.node_id.as_str());
            }
        }
    }
    while let Some(node_id) = pending.pop() {
        if let Some(edge) = producer_by_output.get(node_id) {
            included_edges.insert(edge.edge_id.as_str());
            for input in &edge.inputs {
                if closed.insert(input.node_id.as_str()) {
                    pending.push(input.node_id.as_str());
                }
            }
            for output in &edge.outputs {
                closed.insert(output.node_id.as_str());
            }
        }
    }

    let mut slice = TensorLineageSlice {
        execution_manifest_sha256: manifest.manifest_sha256.clone(),
        source_tensor_name: source_tensor_name.into(),
        nodes: manifest
            .nodes
            .iter()
            .filter(|node| closed.contains(node.node_id.as_str()))
            .cloned()
            .collect(),
        transforms: manifest
            .transforms
            .iter()
            .filter(|edge| included_edges.contains(edge.edge_id.as_str()))
            .cloned()
            .collect(),
        operations: manifest
            .operations
            .iter()
            .filter(|operation| {
                operation
                    .source_tensor_names
                    .iter()
                    .any(|name| name == source_tensor_name)
            })
            .cloned()
            .collect(),
        slice_sha256: String::new(),
    };
    slice.slice_sha256 = tensor_lineage_slice_sha256(&slice)?;
    Ok(normalize_lineage_slice(&slice))
}

fn validate_codec(codec: &PhysicalTensorCodec) -> bool {
    match codec {
        PhysicalTensorCodec::Dense { .. } => true,
        PhysicalTensorCodec::Ggml {
            wire_type_id,
            type_name,
        } => matches!(
            (*wire_type_id, type_name.as_str()),
            (0, "F32")
                | (1, "F16")
                | (2, "Q4_0")
                | (3, "Q4_1")
                | (6, "Q5_0")
                | (7, "Q5_1")
                | (8, "Q8_0")
                | (10, "Q2_K")
                | (11, "Q3_K")
                | (12, "Q4_K")
                | (13, "Q5_K")
                | (14, "Q6_K")
                | (20, "IQ4_NL")
                | (23, "IQ4_XS")
                | (30, "BF16")
        ),
        // Schema v1 models the current Qwen GGUF execution path. Affine
        // physical layout admission requires a separately proven MLX ABI.
        PhysicalTensorCodec::MlxAffine { .. } => false,
    }
}

fn expected_physical_bytes(node: &TensorStateNode) -> Option<u64> {
    let elements = node
        .shape
        .iter()
        .try_fold(1u64, |total, dim| total.checked_mul(*dim))?;
    let (block_values, block_bytes) = match &node.codec {
        PhysicalTensorCodec::Dense { dtype } => {
            return elements.checked_mul(match dtype {
                TensorScalarDType::F16 | TensorScalarDType::Bf16 => 2,
                TensorScalarDType::F32 => 4,
            });
        }
        PhysicalTensorCodec::Ggml {
            wire_type_id,
            type_name,
        } => match (*wire_type_id, type_name.as_str()) {
            (0, "F32") => (1, 4),
            (1, "F16") | (30, "BF16") => (1, 2),
            (2, "Q4_0") => (32, 18),
            (3, "Q4_1") => (32, 20),
            (6, "Q5_0") => (32, 22),
            (7, "Q5_1") => (32, 24),
            (8, "Q8_0") => (32, 34),
            (10, "Q2_K") => (256, 84),
            (11, "Q3_K") => (256, 110),
            (12, "Q4_K") => (256, 144),
            (13, "Q5_K") => (256, 176),
            (14, "Q6_K") => (256, 210),
            (20, "IQ4_NL") => (32, 18),
            (23, "IQ4_XS") => (256, 136),
            _ => return None,
        },
        PhysicalTensorCodec::MlxAffine { .. } => return None,
    };
    let row_width = *node.shape.last()?;
    if row_width % block_values != 0 {
        return None;
    }
    let outer_rows = elements / row_width;
    outer_rows
        .checked_mul(row_width / block_values)?
        .checked_mul(block_bytes)
}

pub fn verify_tensor_execution_manifest(
    manifest: &TensorExecutionManifest,
) -> Result<ValidatedTensorExecutionManifest, TensorExecutionManifestError> {
    let invalid = |message: String| TensorExecutionManifestError::Invalid(message);
    if manifest.schema_version != TENSOR_EXECUTION_MANIFEST_SCHEMA_VERSION
        || !is_sha256(&manifest.source_manifest_sha256)
        || !is_sha256(&manifest.source_tensor_inventory_sha256)
        || !is_sha256(&manifest.tensor_partition_manifest_sha256)
        || !is_sha256(&manifest.conversion_receipt_sha256)
        || manifest.logical_hash_encoding != "hf2q-framed-f32-le-v1"
        || !is_sha256(&manifest.runtime.routing_policy_sha256)
        || !is_sha256(&manifest.runtime.graph_configuration_sha256)
        || !is_sha256(&manifest.runtime.capability_profile_sha256)
        || !is_sha256(&manifest.runtime.hardware_profile_sha256)
        || manifest.runtime.dwq_overlay_sha256.is_some()
        || manifest.runtime.hf2q_revision.len() != 40
        || !manifest
            .runtime
            .hf2q_revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || manifest.runtime.mlx_native_version.is_empty()
        || manifest.runtime.mlx_native_capability_schema_version != 1
        || manifest.scope.model_family.is_empty()
        || manifest.scope.profile.is_empty()
        || manifest.scope.included_paths.is_empty()
    {
        return Err(invalid("manifest identity or scope is incomplete".into()));
    }
    let included_paths: BTreeSet<_> = manifest.scope.included_paths.iter().collect();
    if manifest.scope.included_paths.iter().any(String::is_empty)
        || included_paths.len() != manifest.scope.included_paths.len()
        || manifest.scope.excluded_paths.iter().any(|(path, reason)| {
            path.is_empty() || reason.is_empty() || manifest.scope.included_paths.contains(path)
        })
    {
        return Err(invalid("execution scope paths are invalid".into()));
    }

    let mut artifact_ids = BTreeSet::new();
    for artifact in &manifest.artifacts {
        if artifact.artifact_id.is_empty()
            || artifact.role.is_empty()
            || artifact.byte_len == 0
            || !is_sha256(&artifact.sha256)
            || !artifact_ids.insert(artifact.artifact_id.as_str())
        {
            return Err(invalid(
                "artifact evidence is incomplete or duplicated".into(),
            ));
        }
    }
    if artifact_ids.is_empty() {
        return Err(invalid("at least one artifact is required".into()));
    }

    let mut nodes = BTreeMap::new();
    let mut source_semantic_names = BTreeSet::new();
    for node in &manifest.nodes {
        if node.node_id.is_empty()
            || node.semantic_name.is_empty()
            || node.shape.is_empty()
            || node.shape.contains(&0)
            || node.layout != TENSOR_LAYOUT_ROW_MAJOR_OUTERMOST_FIRST_V1
            || !validate_codec(&node.codec)
            || expected_physical_bytes(node) != Some(node.byte_len)
            || node.byte_len == 0
            || !is_sha256(&node.byte_sha256)
            || !is_sha256(&node.logical_f32_sha256)
            || nodes.insert(node.node_id.as_str(), node).is_some()
        {
            return Err(invalid("tensor node is incomplete or duplicated".into()));
        }
        if node.stage == TensorStateStage::Source
            && !source_semantic_names.insert(node.semantic_name.as_str())
        {
            return Err(invalid("source semantic tensor name is duplicated".into()));
        }
        if let Some(region) = &node.artifact_region {
            let Some(artifact) = manifest
                .artifacts
                .iter()
                .find(|artifact| artifact.artifact_id == region.artifact_id)
            else {
                return Err(invalid(format!(
                    "node `{}` references an unknown artifact",
                    node.node_id
                )));
            };
            let Some(end) = region.byte_offset.checked_add(region.byte_len) else {
                return Err(invalid("artifact region arithmetic overflow".into()));
            };
            if region.byte_len != node.byte_len || end > artifact.byte_len {
                return Err(invalid(format!(
                    "node `{}` has an invalid artifact region",
                    node.node_id
                )));
            }
        }
        match node.stage {
            TensorStateStage::Source | TensorStateStage::Stored
                if node.artifact_region.is_none() =>
            {
                return Err(invalid(format!(
                    "physical node `{}` has no artifact region",
                    node.node_id
                )));
            }
            TensorStateStage::Converted | TensorStateStage::Loaded | TensorStateStage::Executed
                if node.artifact_region.is_some() =>
            {
                return Err(invalid(format!(
                    "ephemeral node `{}` has an artifact region",
                    node.node_id
                )));
            }
            _ => {}
        }
    }
    if nodes.is_empty() {
        return Err(invalid("at least one tensor node is required".into()));
    }

    let mut regions_by_artifact: BTreeMap<&str, Vec<(u64, u64, &str)>> = BTreeMap::new();
    for node in &manifest.nodes {
        if let Some(region) = &node.artifact_region {
            regions_by_artifact
                .entry(region.artifact_id.as_str())
                .or_default()
                .push((
                    region.byte_offset,
                    region.byte_offset + region.byte_len,
                    node.node_id.as_str(),
                ));
        }
    }
    for regions in regions_by_artifact.values_mut() {
        regions.sort_by_key(|region| region.0);
        for pair in regions.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(invalid(format!(
                    "artifact regions for `{}` and `{}` overlap",
                    pair[0].2, pair[1].2
                )));
            }
        }
    }

    let mut edge_ids = BTreeSet::new();
    let mut produced_by = BTreeMap::new();
    let mut edge_inputs_by_output: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for edge in &manifest.transforms {
        if edge.edge_id.is_empty()
            || edge.inputs.is_empty()
            || edge.outputs.is_empty()
            || edge.implementation_revision.len() != 40
            || !edge
                .implementation_revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || !is_sha256(&edge.receipt_sha256)
            || !edge_ids.insert(edge.edge_id.as_str())
        {
            return Err(invalid("transform edge is incomplete or duplicated".into()));
        }
        let input_ids: BTreeSet<_> = edge.inputs.iter().map(|port| &port.node_id).collect();
        let output_ids: BTreeSet<_> = edge.outputs.iter().map(|port| &port.node_id).collect();
        let input_roles: BTreeSet<_> = edge.inputs.iter().map(|port| &port.role).collect();
        let output_roles: BTreeSet<_> = edge.outputs.iter().map(|port| &port.role).collect();
        if edge
            .inputs
            .iter()
            .chain(&edge.outputs)
            .any(|port| port.role.is_empty())
            || input_ids.len() != edge.inputs.len()
            || output_ids.len() != edge.outputs.len()
            || input_roles.len() != edge.inputs.len()
            || output_roles.len() != edge.outputs.len()
            || !input_ids.is_disjoint(&output_ids)
        {
            return Err(invalid(format!("edge `{}` repeats a node", edge.edge_id)));
        }
        let input_nodes = edge
            .inputs
            .iter()
            .map(|port| nodes.get(port.node_id.as_str()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid(format!("edge `{}` has an unknown input", edge.edge_id)))?;
        let input_stages: BTreeSet<_> = input_nodes.iter().map(|node| node.stage).collect();
        let output_nodes = edge
            .outputs
            .iter()
            .map(|port| nodes.get(port.node_id.as_str()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid(format!("edge `{}` has an unknown output", edge.edge_id)))?;
        let output_stages: BTreeSet<_> = output_nodes.iter().map(|node| node.stage).collect();
        let legal_signature = match &edge.operation {
            TensorTransformOperation::SourceDecode => {
                input_stages == [TensorStateStage::Source].into()
                    && output_stages == [TensorStateStage::Converted].into()
            }
            TensorTransformOperation::ArchitectureBake { .. }
            | TensorTransformOperation::Concatenate { .. }
            | TensorTransformOperation::F16Roundtrip => {
                input_stages == [TensorStateStage::Converted].into()
                    && output_stages == [TensorStateStage::Converted].into()
            }
            TensorTransformOperation::GgufQuantize { .. } => {
                input_stages == [TensorStateStage::Converted].into()
                    && output_stages == [TensorStateStage::Stored].into()
            }
            TensorTransformOperation::GgufDenseStore { .. } => {
                input_stages == [TensorStateStage::Converted].into()
                    && output_stages == [TensorStateStage::Stored].into()
            }
            TensorTransformOperation::GgufDequantize { .. }
            | TensorTransformOperation::DirectBlockLoad => {
                input_stages == [TensorStateStage::Stored].into()
                    && output_stages == [TensorStateStage::Loaded].into()
            }
            TensorTransformOperation::SplitInterleavedQGate { .. } => {
                input_stages == [TensorStateStage::Loaded].into()
                    && output_stages == [TensorStateStage::Loaded].into()
            }
            TensorTransformOperation::Transpose { .. }
            | TensorTransformOperation::Squeeze { .. }
            | TensorTransformOperation::ZeroPad { .. } => {
                input_stages.len() == 1
                    && input_stages == output_stages
                    && input_stages.iter().all(|stage| {
                        matches!(
                            stage,
                            TensorStateStage::Converted | TensorStateStage::Loaded
                        )
                    })
            }
            TensorTransformOperation::Qwen35LoadQ4Amax7V1 => {
                input_stages == [TensorStateStage::Loaded].into()
                    && output_stages == [TensorStateStage::Executed].into()
            }
            TensorTransformOperation::RuntimeBind => {
                input_stages == [TensorStateStage::Loaded].into()
                    && output_stages == [TensorStateStage::Executed].into()
            }
        };
        if !legal_signature {
            return Err(invalid(format!(
                "edge `{}` has an illegal stage signature",
                edge.edge_id
            )));
        }
        let single_input_output = || {
            (input_nodes.len() == 1 && output_nodes.len() == 1)
                .then_some((input_nodes[0], output_nodes[0]))
        };
        let input_role_set: BTreeSet<_> =
            edge.inputs.iter().map(|port| port.role.as_str()).collect();
        let output_role_set: BTreeSet<_> =
            edge.outputs.iter().map(|port| port.role.as_str()).collect();
        let geometry_valid = match &edge.operation {
            TensorTransformOperation::SourceDecode => {
                single_input_output().is_some_and(|(input, output)| {
                    input_role_set == ["source"].into()
                        && output_role_set == ["converted"].into()
                        && input.shape == output.shape
                        && matches!(input.codec, PhysicalTensorCodec::Dense { .. })
                        && matches!(
                            output.codec,
                            PhysicalTensorCodec::Dense {
                                dtype: TensorScalarDType::F32
                            }
                        )
                        && input.logical_f32_sha256 == output.logical_f32_sha256
                })
            }
            TensorTransformOperation::ArchitectureBake { .. } => {
                single_input_output().is_some_and(|(input, output)| {
                    input.shape == output.shape
                        && matches!(
                            input.codec,
                            PhysicalTensorCodec::Dense {
                                dtype: TensorScalarDType::F32
                            }
                        )
                        && matches!(
                            output.codec,
                            PhysicalTensorCodec::Dense {
                                dtype: TensorScalarDType::F32
                            }
                        )
                })
            }
            TensorTransformOperation::Concatenate { axis } => {
                output_nodes.len() == 1
                    && usize::try_from(*axis).is_ok_and(|axis| {
                        let output = output_nodes[0];
                        axis < output.shape.len()
                            && input_nodes.iter().all(|input| {
                                input.shape.len() == output.shape.len()
                                    && matches!(
                                        input.codec,
                                        PhysicalTensorCodec::Dense {
                                            dtype: TensorScalarDType::F32
                                        }
                                    )
                                    && matches!(
                                        output.codec,
                                        PhysicalTensorCodec::Dense {
                                            dtype: TensorScalarDType::F32
                                        }
                                    )
                                    && input.shape.iter().zip(&output.shape).enumerate().all(
                                        |(index, (input_dim, output_dim))| {
                                            index == axis || input_dim == output_dim
                                        },
                                    )
                            })
                            && input_nodes
                                .iter()
                                .try_fold(0u64, |total, input| total.checked_add(input.shape[axis]))
                                == Some(output.shape[axis])
                    })
            }
            TensorTransformOperation::F16Roundtrip => {
                single_input_output().is_some_and(|(input, output)| {
                    input.shape == output.shape
                        && matches!(
                            input.codec,
                            PhysicalTensorCodec::Dense {
                                dtype: TensorScalarDType::F32
                            }
                        )
                        && matches!(
                            output.codec,
                            PhysicalTensorCodec::Dense {
                                dtype: TensorScalarDType::F32
                            }
                        )
                })
            }
            TensorTransformOperation::GgufQuantize { .. } => {
                single_input_output().is_some_and(|(input, output)| {
                    input_role_set == ["converted"].into()
                        && output_role_set == ["stored"].into()
                        && matches!(
                            input.codec,
                            PhysicalTensorCodec::Dense {
                                dtype: TensorScalarDType::F32
                            }
                        )
                        && matches!(
                            output.codec,
                            PhysicalTensorCodec::Ggml {
                                wire_type_id: 2 | 3 | 6 | 7 | 8 | 10 | 11 | 12 | 13 | 14 | 20 | 23,
                                ..
                            }
                        )
                        && input.shape == output.shape
                })
            }
            TensorTransformOperation::GgufDenseStore { .. } => {
                single_input_output().is_some_and(|(input, output)| {
                    input_role_set == ["converted"].into()
                        && output_role_set == ["stored"].into()
                        && matches!(
                            input.codec,
                            PhysicalTensorCodec::Dense {
                                dtype: TensorScalarDType::F32
                            }
                        )
                        && matches!(
                            output.codec,
                            PhysicalTensorCodec::Ggml {
                                wire_type_id: 0 | 1 | 30,
                                ..
                            }
                        )
                        && input.shape == output.shape
                })
            }
            TensorTransformOperation::GgufDequantize { .. } => {
                single_input_output().is_some_and(|(input, output)| {
                    input_role_set == ["stored"].into()
                        && output_role_set == ["loaded"].into()
                        && input.shape == output.shape
                        && matches!(input.codec, PhysicalTensorCodec::Ggml { .. })
                        && matches!(
                            output.codec,
                            PhysicalTensorCodec::Dense {
                                dtype: TensorScalarDType::F32
                            }
                        )
                        && input.logical_f32_sha256 == output.logical_f32_sha256
                })
            }
            TensorTransformOperation::DirectBlockLoad | TensorTransformOperation::RuntimeBind => {
                single_input_output().is_some_and(|(input, output)| {
                    let roles_match = match &edge.operation {
                        TensorTransformOperation::DirectBlockLoad => {
                            input_role_set == ["stored"].into()
                                && output_role_set == ["loaded"].into()
                        }
                        TensorTransformOperation::RuntimeBind => {
                            input_role_set == ["loaded"].into()
                                && output_role_set == ["executed"].into()
                        }
                        _ => unreachable!(),
                    };
                    roles_match
                        && input.shape == output.shape
                        && input.codec == output.codec
                        && input.byte_len == output.byte_len
                        && input.byte_sha256 == output.byte_sha256
                        && input.logical_f32_sha256 == output.logical_f32_sha256
                })
            }
            TensorTransformOperation::SplitInterleavedQGate {
                heads,
                head_dim,
                hidden_size,
                ..
            } => {
                let projection_rows = u64::from(*heads).checked_mul(u64::from(*head_dim));
                let fused_rows = projection_rows.and_then(|rows| rows.checked_mul(2));
                input_nodes.len() == 1
                    && output_nodes.len() == 2
                    && input_role_set == ["loaded"].into()
                    && output_role_set == ["gate", "q"].into()
                    && projection_rows.is_some_and(|rows| {
                        rows > 0
                            && fused_rows.is_some_and(|fused_rows| {
                                input_nodes[0].shape == [fused_rows, u64::from(*hidden_size)]
                            })
                            && output_nodes
                                .iter()
                                .all(|node| node.shape == [rows, u64::from(*hidden_size)])
                    })
                    && input_nodes.iter().chain(&output_nodes).all(|node| {
                        matches!(
                            node.codec,
                            PhysicalTensorCodec::Dense {
                                dtype: TensorScalarDType::F32
                            }
                        )
                    })
            }
            TensorTransformOperation::Transpose { permutation } => single_input_output()
                .is_some_and(|(input, output)| {
                    permutation.len() == input.shape.len()
                        && permutation.len() == output.shape.len()
                        && permutation
                            .iter()
                            .enumerate()
                            .all(|(output_axis, input_axis)| {
                                usize::try_from(*input_axis).is_ok_and(|input_axis| {
                                    input_axis < input.shape.len()
                                        && output.shape[output_axis] == input.shape[input_axis]
                                })
                            })
                        && input.codec == output.codec
                }),
            TensorTransformOperation::Squeeze { axis } => {
                single_input_output().is_some_and(|(input, output)| {
                    usize::try_from(*axis).is_ok_and(|axis| {
                        axis < input.shape.len()
                            && input.shape[axis] == 1
                            && input.shape.len() == output.shape.len() + 1
                            && input
                                .shape
                                .iter()
                                .enumerate()
                                .filter_map(|(index, dimension)| {
                                    (index != axis).then_some(*dimension)
                                })
                                .eq(output.shape.iter().copied())
                            && matches!(
                                input.codec,
                                PhysicalTensorCodec::Dense {
                                    dtype: TensorScalarDType::F32
                                }
                            )
                            && input.codec == output.codec
                            && input.logical_f32_sha256 == output.logical_f32_sha256
                    })
                })
            }
            TensorTransformOperation::ZeroPad {
                axis,
                before,
                after,
            } => single_input_output().is_some_and(|(input, output)| {
                usize::try_from(*axis).is_ok_and(|axis| {
                    axis < input.shape.len()
                        && input.shape.len() == output.shape.len()
                        && input.codec == output.codec
                        && input.shape.iter().zip(&output.shape).enumerate().all(
                            |(index, (input_dim, output_dim))| {
                                if index == axis {
                                    input_dim
                                        .checked_add(*before)
                                        .and_then(|value| value.checked_add(*after))
                                        == Some(*output_dim)
                                } else {
                                    input_dim == output_dim
                                }
                            },
                        )
                })
            }),
            TensorTransformOperation::Qwen35LoadQ4Amax7V1 => {
                input_nodes.len() == 1
                    && output_nodes.len() == 1
                    && input_role_set == ["loaded"].into()
                    && output_role_set == ["executed"].into()
                    && input_nodes.iter().all(|node| {
                        matches!(
                            node.codec,
                            PhysicalTensorCodec::Dense {
                                dtype: TensorScalarDType::F32
                            }
                        )
                    })
                    && output_nodes.iter().all(|output| {
                        matches!(
                            &output.codec,
                            PhysicalTensorCodec::Ggml {
                                wire_type_id: 2,
                                type_name
                            } if type_name == "Q4_0"
                        ) && input_nodes[0].shape == output.shape
                    })
            }
        };
        if !geometry_valid {
            return Err(invalid(format!(
                "edge `{}` has invalid codec or geometry semantics",
                edge.edge_id
            )));
        }
        for output_port in &edge.outputs {
            if produced_by
                .insert(output_port.node_id.as_str(), edge.edge_id.as_str())
                .is_some()
            {
                return Err(invalid(format!(
                    "edge `{}` has a multiply-produced output",
                    edge.edge_id
                )));
            }
            edge_inputs_by_output.insert(
                output_port.node_id.as_str(),
                edge.inputs
                    .iter()
                    .map(|port| port.node_id.as_str())
                    .collect(),
            );
        }
        match &edge.operation {
            TensorTransformOperation::ArchitectureBake {
                operation,
                parameters_sha256,
            } if operation.is_empty() || !is_sha256(parameters_sha256) => {
                return Err(invalid(format!(
                    "edge `{}` has invalid bake evidence",
                    edge.edge_id
                )));
            }
            TensorTransformOperation::GgufQuantize {
                implementation_id,
                calibration_receipt_sha256,
            } if implementation_id.is_empty()
                || calibration_receipt_sha256
                    .as_ref()
                    .is_some_and(|digest| !is_sha256(digest)) =>
            {
                return Err(invalid(format!(
                    "edge `{}` has invalid quantization evidence",
                    edge.edge_id
                )));
            }
            TensorTransformOperation::GgufDenseStore { implementation_id }
                if implementation_id.is_empty() =>
            {
                return Err(invalid(format!(
                    "edge `{}` has invalid dense-store evidence",
                    edge.edge_id
                )));
            }
            TensorTransformOperation::GgufDequantize { implementation_id }
            | TensorTransformOperation::SplitInterleavedQGate {
                implementation_id, ..
            } if implementation_id.is_empty() => {
                return Err(invalid(format!(
                    "edge `{}` has no implementation identity",
                    edge.edge_id
                )));
            }
            TensorTransformOperation::Transpose { permutation }
                if permutation.is_empty()
                    || permutation.iter().copied().collect::<BTreeSet<_>>().len()
                        != permutation.len() =>
            {
                return Err(invalid(format!(
                    "edge `{}` has an invalid permutation",
                    edge.edge_id
                )));
            }
            TensorTransformOperation::ZeroPad { before, after, .. }
                if *before == 0 && *after == 0 =>
            {
                return Err(invalid(format!("edge `{}` has an empty pad", edge.edge_id)));
            }
            _ => {}
        }
    }
    // The converter's quantized path rounds converted F32 through F16
    // immediately before quantization. Dense F32/F16/BF16 stores encode
    // directly from converted F32 and therefore must not invent this node.
    for edge in &manifest.transforms {
        if matches!(
            edge.operation,
            TensorTransformOperation::GgufQuantize { .. }
        ) {
            let input = edge.inputs.first().expect("store arity validated above");
            let Some(producer_id) = produced_by.get(input.node_id.as_str()) else {
                return Err(invalid(format!(
                    "edge `{}` does not consume a canonical F16 roundtrip",
                    edge.edge_id
                )));
            };
            let producer = manifest
                .transforms
                .iter()
                .find(|candidate| candidate.edge_id == **producer_id)
                .expect("producer id came from the manifest");
            if !matches!(producer.operation, TensorTransformOperation::F16Roundtrip) {
                return Err(invalid(format!(
                    "edge `{}` does not consume a canonical F16 roundtrip",
                    edge.edge_id
                )));
            }
        }
    }
    for node in manifest
        .nodes
        .iter()
        .filter(|node| node.stage != TensorStateStage::Source)
    {
        if !produced_by.contains_key(node.node_id.as_str()) {
            return Err(invalid(format!(
                "non-source node `{}` has no producing edge",
                node.node_id
            )));
        }
    }

    let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut indegree: BTreeMap<&str, usize> = manifest
        .nodes
        .iter()
        .map(|node| (node.node_id.as_str(), 0))
        .collect();
    for edge in &manifest.transforms {
        for input in &edge.inputs {
            for output in &edge.outputs {
                adjacency
                    .entry(input.node_id.as_str())
                    .or_default()
                    .push(output.node_id.as_str());
                *indegree
                    .get_mut(output.node_id.as_str())
                    .expect("edge output was validated above") += 1;
            }
        }
    }
    let mut ready: Vec<_> = indegree
        .iter()
        .filter_map(|(node_id, count)| (*count == 0).then_some(*node_id))
        .collect();
    let mut visited_count = 0usize;
    while let Some(node_id) = ready.pop() {
        visited_count += 1;
        for output in adjacency.get(node_id).into_iter().flatten() {
            let count = indegree
                .get_mut(output)
                .expect("edge output was validated above");
            *count -= 1;
            if *count == 0 {
                ready.push(output);
            }
        }
    }
    if visited_count != manifest.nodes.len() {
        return Err(invalid("tensor execution graph contains a cycle".into()));
    }

    let ancestor_stages = |node_id: &str| {
        let mut stages = BTreeSet::new();
        let mut pending = vec![node_id];
        let mut visited = BTreeSet::new();
        while let Some(current) = pending.pop() {
            if !visited.insert(current) {
                continue;
            }
            stages.insert(nodes[current].stage);
            if let Some(inputs) = edge_inputs_by_output.get(current) {
                pending.extend(inputs.iter().copied());
            }
        }
        stages
    };
    for node in &manifest.nodes {
        let stages = ancestor_stages(node.node_id.as_str());
        let valid = match node.stage {
            TensorStateStage::Source => true,
            TensorStateStage::Converted | TensorStateStage::Stored => {
                stages.contains(&TensorStateStage::Source)
            }
            TensorStateStage::Loaded => {
                stages.contains(&TensorStateStage::Source)
                    && stages.contains(&TensorStateStage::Stored)
            }
            TensorStateStage::Executed => {
                stages.contains(&TensorStateStage::Source)
                    && stages.contains(&TensorStateStage::Stored)
                    && stages.contains(&TensorStateStage::Loaded)
            }
        };
        if !valid {
            return Err(invalid(format!(
                "node `{}` does not have the required physical ancestry",
                node.node_id
            )));
        }
    }

    let mut binding_ids = BTreeSet::new();
    let mut consumed_executed_nodes = BTreeSet::new();
    for operation in &manifest.operations {
        if operation.binding_id.is_empty()
            || operation.operation_id.is_empty()
            || operation.graph_path.is_empty()
            || operation.entrypoint.is_empty()
            || operation.source_tensor_names.is_empty()
            || operation.inputs.is_empty()
            || !binding_ids.insert(operation.binding_id.as_str())
        {
            return Err(invalid(
                "runtime operation is incomplete or duplicated".into(),
            ));
        }
        let executed_ids: BTreeSet<_> = operation.inputs.iter().map(|port| &port.node_id).collect();
        let source_names: BTreeSet<&str> = operation
            .source_tensor_names
            .iter()
            .map(String::as_str)
            .collect();
        let input_roles: BTreeSet<_> = operation.inputs.iter().map(|port| &port.role).collect();
        if operation.inputs.iter().any(|port| port.role.is_empty())
            || operation.source_tensor_names.iter().any(String::is_empty)
            || source_names.len() != operation.source_tensor_names.len()
            || executed_ids.len() != operation.inputs.len()
            || input_roles.len() != operation.inputs.len()
            || operation.inputs.iter().any(|port| {
                nodes
                    .get(port.node_id.as_str())
                    .is_none_or(|node| node.stage != TensorStateStage::Executed)
            })
        {
            return Err(invalid(format!(
                "operation `{}` does not bind unique executed nodes",
                operation.operation_id
            )));
        }
        let mut source_closure = BTreeSet::new();
        let mut pending: Vec<_> = operation
            .inputs
            .iter()
            .map(|port| port.node_id.as_str())
            .collect();
        let mut visited = BTreeSet::new();
        while let Some(node_id) = pending.pop() {
            if !visited.insert(node_id) {
                continue;
            }
            let node = nodes[node_id];
            if node.stage == TensorStateStage::Source {
                source_closure.insert(node.semantic_name.as_str());
            } else if let Some(inputs) = edge_inputs_by_output.get(node_id) {
                pending.extend(inputs.iter().copied());
            }
        }
        if source_closure != source_names {
            return Err(invalid(format!(
                "operation `{}` has the wrong source-tensor closure",
                operation.operation_id
            )));
        }
        consumed_executed_nodes.extend(operation.inputs.iter().map(|port| port.node_id.as_str()));
        let ggml_input_count = operation
            .inputs
            .iter()
            .filter(|port| {
                matches!(
                    nodes[port.node_id.as_str()].codec,
                    PhysicalTensorCodec::Ggml { .. }
                )
            })
            .count();
        let has_ggml_input = ggml_input_count > 0;
        match &operation.capability {
            RuntimeCapabilityEvidence::Ggml {
                request,
                decision,
                requires_device_probe,
                resolved_runtime_trace,
            } => {
                if ggml_input_count != operation.inputs.len() {
                    return Err(invalid(format!(
                        "operation `{}` claims GGML capability without all-GGML inputs",
                        operation.operation_id
                    )));
                }
                validate_json_evidence(request).map_err(invalid)?;
                validate_json_evidence(decision).map_err(invalid)?;
                if *requires_device_probe && resolved_runtime_trace.is_none() {
                    return Err(invalid(format!(
                        "operation `{}` has no required device-probe evidence",
                        operation.operation_id
                    )));
                }
                if let Some(trace) = resolved_runtime_trace {
                    validate_json_evidence(trace).map_err(invalid)?;
                }
            }
            RuntimeCapabilityEvidence::NonGgml {
                implementation_id,
                contract_sha256,
                resolved_runtime_trace,
            } => {
                if has_ggml_input || implementation_id.is_empty() || !is_sha256(contract_sha256) {
                    return Err(invalid(format!(
                        "operation `{}` has invalid non-GGML evidence",
                        operation.operation_id
                    )));
                }
                if let Some(trace) = resolved_runtime_trace {
                    validate_json_evidence(trace).map_err(invalid)?;
                }
            }
        }
        if operation.invocation_count == 0 {
            return Err(invalid(format!(
                "operation `{}` has zero invocations",
                operation.operation_id
            )));
        }
    }
    let all_executed_nodes: BTreeSet<_> = manifest
        .nodes
        .iter()
        .filter(|node| node.stage == TensorStateStage::Executed)
        .map(|node| node.node_id.as_str())
        .collect();
    if all_executed_nodes != consumed_executed_nodes {
        return Err(invalid(
            "every executed tensor must be consumed by a runtime operation".into(),
        ));
    }

    let source_ids: BTreeSet<_> = manifest
        .nodes
        .iter()
        .filter(|node| node.stage == TensorStateStage::Source)
        .map(|node| node.node_id.as_str())
        .collect();
    let mut disposition_ids = BTreeSet::new();
    for disposition in &manifest.dispositions {
        let terminal_ids: BTreeSet<&str> = disposition
            .terminal_node_ids
            .iter()
            .map(String::as_str)
            .collect();
        if disposition.reason.is_empty()
            || !source_ids.contains(disposition.source_node_id.as_str())
            || !disposition_ids.insert(disposition.source_node_id.as_str())
            || terminal_ids.len() != disposition.terminal_node_ids.len()
            || disposition
                .terminal_node_ids
                .iter()
                .any(|node_id| !nodes.contains_key(node_id.as_str()))
            || (matches!(
                disposition.disposition,
                SourceTensorDispositionKind::Variable | SourceTensorDispositionKind::Fixed
            ) && disposition.terminal_node_ids.is_empty())
            || (matches!(
                disposition.disposition,
                SourceTensorDispositionKind::Excluded
            ) && !disposition.terminal_node_ids.is_empty())
        {
            return Err(invalid(
                "source disposition is incomplete or duplicated".into(),
            ));
        }
        if disposition
            .terminal_node_ids
            .iter()
            .any(|node_id| nodes[node_id.as_str()].stage != TensorStateStage::Executed)
        {
            return Err(invalid(
                "source dispositions may terminate only at executed tensors".into(),
            ));
        }
        let mut reachable_executed = BTreeSet::new();
        let mut pending = vec![disposition.source_node_id.as_str()];
        let mut reachable = BTreeSet::new();
        while let Some(node_id) = pending.pop() {
            if !reachable.insert(node_id) {
                continue;
            }
            if nodes[node_id].stage == TensorStateStage::Executed {
                reachable_executed.insert(node_id);
            }
            if let Some(outputs) = adjacency.get(node_id) {
                pending.extend(outputs.iter().copied());
            }
        }
        if terminal_ids != reachable_executed {
            return Err(invalid(format!(
                "source `{}` disposition does not cover its exact executed descendants",
                disposition.source_node_id
            )));
        }
        for terminal_id in &disposition.terminal_node_ids {
            let mut pending = vec![terminal_id.as_str()];
            let mut ancestors = BTreeSet::new();
            while let Some(node_id) = pending.pop() {
                if ancestors.insert(node_id) {
                    if let Some(inputs) = edge_inputs_by_output.get(node_id) {
                        pending.extend(inputs.iter().copied());
                    }
                }
            }
            if !ancestors.contains(disposition.source_node_id.as_str()) {
                return Err(invalid(format!(
                    "terminal `{terminal_id}` is not reachable from source `{}`",
                    disposition.source_node_id
                )));
            }
        }
    }
    if source_ids != disposition_ids {
        return Err(invalid(
            "every source tensor must have exactly one disposition".into(),
        ));
    }

    if !is_sha256(&manifest.manifest_sha256)
        || tensor_execution_manifest_sha256(manifest)? != manifest.manifest_sha256
    {
        return Err(TensorExecutionManifestError::DigestMismatch);
    }
    Ok(ValidatedTensorExecutionManifest {
        manifest: normalize_manifest(manifest),
    })
}
