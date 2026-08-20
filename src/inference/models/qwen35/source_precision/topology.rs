//! Exact dense-Qwen source-to-future-execution topology admission.
//!
//! This module validates names, shapes, dispositions, mapper transforms, and
//! the full-attention Q/gate fanout. It deliberately performs no payload
//! transform, Metal allocation, graph execution, or authority promotion.

use std::collections::{BTreeSet, HashSet};

use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::convert::arch::bake::BakeOp;
use crate::convert::arch::qwen35_dense::{
    context_from_config, is_qwen35_dense_vision_source_tensor, map_tensor_name,
};
use crate::convert::arch::qwen35moe_full::MappedTensor;

use super::super::source_config::qwen35_config_from_authenticated_source;
use super::super::{Qwen35LayerKind, Qwen35Variant};
use super::snapshot::VerifiedQwenSourceSnapshot;
use super::topology_expected::expected_sources;
use super::types::{SourcePrecisionDType, SourcePrecisionDisposition, SourcePrecisionTensorRecord};

const TOPOLOGY_SCHEMA_VERSION: u32 = 1;
const TOPOLOGY_PROFILE: &str = "dense_qwen35_source_bf16_topology_v1";

#[cfg(test)]
mod test_support;
#[cfg(test)]
pub(super) use test_support::expected_profile_for_config_for_test;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Qwen35FutureDType {
    Bf16,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Qwen35QGateBranch {
    Query,
    Gate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub(super) enum Qwen35SourceTransformV1 {
    Identity,
    AddOneF32,
    ReorderVHeads {
        num_key_heads: usize,
        num_values_per_key: usize,
        block_elements: usize,
        slice_start: Option<usize>,
        slice_end: Option<usize>,
    },
    ReorderVHeadsThenNegExpF32 {
        num_key_heads: usize,
        num_values_per_key: usize,
    },
    SqueezeAxis1ThenReorderVSlice {
        num_key_heads: usize,
        num_values_per_key: usize,
        value_head_dim: usize,
        kernel_width: usize,
        slice_start: usize,
        slice_end: usize,
    },
    ReorderVHeadsPerRow {
        row_count: usize,
        num_key_heads: usize,
        num_values_per_key: usize,
        value_head_dim: usize,
    },
    SplitInterleavedQGate {
        branch: Qwen35QGateBranch,
        num_query_heads: usize,
        head_dim: usize,
        hidden_size: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct Qwen35FutureTensorRecord {
    pub node_id: String,
    pub shape: Vec<usize>,
    pub dtype: Qwen35FutureDType,
    pub transform: Qwen35SourceTransformV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Qwen35SourceUse {
    FutureExecution,
    AuthenticatedNonExecutedMtp,
    ExcludedVision,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct Qwen35SourceTopologyRecord {
    pub(super) source_name: String,
    pub(super) source_shape: Vec<usize>,
    pub(super) source_byte_sha256: String,
    pub(super) disposition: SourcePrecisionDisposition,
    pub(super) source_use: Qwen35SourceUse,
    pub(super) outputs: Vec<Qwen35FutureTensorRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct Qwen35TopologyConfigV1 {
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: usize,
    pub(super) vocabulary_size: usize,
    pub(super) num_hidden_layers: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_key_value_heads: usize,
    pub(super) head_dim: usize,
    pub(super) linear_num_key_heads: usize,
    pub(super) linear_num_value_heads: usize,
    pub(super) linear_head_dim: usize,
    pub(super) linear_conv_kernel_dim: usize,
    pub(super) layer_types: Vec<&'static str>,
    pub(super) mtp_num_hidden_layers: usize,
    pub(super) multimodal_wrapping: bool,
}

#[derive(Serialize)]
struct TopologyHashView<'a> {
    schema_version: u32,
    profile: &'static str,
    source_snapshot_catalog_sha256: &'a str,
    verified_source_manifest_sha256: &'a str,
    source_inventory_manifest_sha256: &'a str,
    tensor_partition_manifest_sha256: &'a str,
    config: &'a Qwen35TopologyConfigV1,
    records: &'a [Qwen35SourceTopologyRecord],
}

/// Opaque structural proof that one retained dense-Qwen snapshot has exact
/// BF16 source topology for the future source-precision teacher.
pub(crate) struct VerifiedQwen35Bf16TopologyV1 {
    _snapshot: VerifiedQwenSourceSnapshot,
    records: Vec<Qwen35SourceTopologyRecord>,
    topology_sha256: String,
    future_bf16_tensors: usize,
    future_f32_tensors: usize,
}

impl VerifiedQwen35Bf16TopologyV1 {
    pub(crate) fn topology_sha256(&self) -> &str {
        &self.topology_sha256
    }

    pub(crate) fn source_count(&self) -> usize {
        self.records.len()
    }

    pub(crate) fn future_tensor_count(&self) -> usize {
        self.future_bf16_tensors + self.future_f32_tensors
    }

    pub(crate) fn future_bf16_tensor_count(&self) -> usize {
        self.future_bf16_tensors
    }

    pub(crate) fn future_f32_tensor_count(&self) -> usize {
        self.future_f32_tensors
    }

    pub(super) fn projected_config_for_teacher(&self) -> Result<super::super::Qwen35Config> {
        qwen35_config_from_authenticated_source(self._snapshot.config())
    }

    pub(super) fn planned_output_bytes(&self) -> Result<u64> {
        self.records
            .iter()
            .flat_map(|record| &record.outputs)
            .try_fold(0_u64, |total, output| {
                let elements = output.shape.iter().try_fold(1_u64, |product, dimension| {
                    product.checked_mul(u64::try_from(*dimension).ok()?)
                })?;
                let element_bytes = match output.dtype {
                    Qwen35FutureDType::Bf16 => 2,
                    Qwen35FutureDType::F32 => 4,
                };
                total.checked_add(elements.checked_mul(element_bytes)?)
            })
            .context("dense-Qwen planned output bytes overflow")
    }

    pub(super) fn into_upload_parts(
        self,
    ) -> (
        VerifiedQwenSourceSnapshot,
        Vec<Qwen35SourceTopologyRecord>,
        String,
        usize,
        usize,
    ) {
        (
            self._snapshot,
            self.records,
            self.topology_sha256,
            self.future_bf16_tensors,
            self.future_f32_tensors,
        )
    }

    #[cfg(test)]
    pub(super) fn records_for_test(&self) -> &[Qwen35SourceTopologyRecord] {
        &self.records
    }
}

#[derive(Debug)]
pub(super) struct ExpectedSource {
    pub(super) shape: Vec<usize>,
    pub(super) mapped_name: Option<String>,
    pub(super) mapped_bake: Option<BakeOp>,
    pub(super) outputs: Vec<Qwen35FutureTensorRecord>,
    pub(super) source_use: Qwen35SourceUse,
}

pub(crate) fn admit_qwen35_bf16_topology(
    snapshot: VerifiedQwenSourceSnapshot,
) -> Result<VerifiedQwen35Bf16TopologyV1> {
    let (records, future_bf16_tensors, future_f32_tensors, topology_sha256) =
        build_topology(&snapshot)?;
    Ok(VerifiedQwen35Bf16TopologyV1 {
        _snapshot: snapshot,
        records,
        topology_sha256,
        future_bf16_tensors,
        future_f32_tensors,
    })
}

fn build_topology(
    snapshot: &VerifiedQwenSourceSnapshot,
) -> Result<(Vec<Qwen35SourceTopologyRecord>, usize, usize, String)> {
    let mapper = context_from_config(snapshot.config())
        .context("authenticated dense-Qwen config cannot construct the production mapper")?;
    let text = snapshot
        .config()
        .get("text_config")
        .unwrap_or(snapshot.config());
    let mtp_num_hidden_layers = match text.get("mtp_num_hidden_layers") {
        Some(value) => usize::try_from(value.as_u64().context(
            "authenticated Qwen config mtp_num_hidden_layers must be an unsigned integer",
        )?)?,
        None => 0,
    };
    ensure!(
        mtp_num_hidden_layers <= 1,
        "source teacher v1 supports at most one nonexecuted MTP block"
    );
    let full_layers = mapper.num_hidden_layers / mapper.full_attention_interval;
    let linear_layers = mapper.num_hidden_layers - full_layers;
    let expected_text_sources = 3_usize
        .checked_add(
            full_layers
                .checked_mul(11)
                .context("Qwen topology source count overflow")?,
        )
        .and_then(|count| count.checked_add(linear_layers.checked_mul(14)?))
        .and_then(|count| count.checked_add(mtp_num_hidden_layers.checked_mul(15)?))
        .context("Qwen topology source count overflow")?;
    let actual_text_sources = snapshot
        .tensor_records()
        .iter()
        .filter(|tensor| !is_qwen35_dense_vision_source_tensor(&tensor.name))
        .count();
    ensure!(
        actual_text_sources == expected_text_sources,
        "authenticated dense-Qwen config requires {expected_text_sources} text sources, snapshot contains {actual_text_sources}"
    );

    let projected = qwen35_config_from_authenticated_source(snapshot.config())?;
    ensure!(
        projected.variant == Qwen35Variant::Dense
            && projected.moe.is_none()
            && projected.attn_output_gate,
        "source teacher requires dense Qwen with the fused attention output gate"
    );
    ensure!(
        projected.linear_key_head_dim == projected.linear_value_head_dim,
        "source teacher requires equal linear key/value head dimensions"
    );
    ensure!(
        projected.linear_num_key_heads > 0
            && projected.linear_num_value_heads > projected.linear_num_key_heads
            && projected.linear_num_value_heads % projected.linear_num_key_heads == 0,
        "source teacher v1 requires grouped linear value heads"
    );
    ensure!(
        projected.mtp_num_hidden_layers <= 1
            && (projected.mtp_num_hidden_layers == 0 || !projected.mtp_use_dedicated_embeddings),
        "source teacher v1 supports at most one shared-embedding nonexecuted MTP block"
    );
    ensure!(
        mapper.num_hidden_layers == usize::try_from(projected.num_hidden_layers)?
            && mapper.linear.linear_num_key_heads
                == usize::try_from(projected.linear_num_key_heads)?
            && mapper.linear.linear_num_value_heads
                == usize::try_from(projected.linear_num_value_heads)?
            && mapper.linear.linear_key_head_dim == usize::try_from(projected.linear_key_head_dim)?
            && mapper.linear.linear_value_head_dim
                == usize::try_from(projected.linear_value_head_dim)?,
        "production mapper context differs from authenticated execution config"
    );

    let config = topology_config(&projected, mapper.multimodal_wrapping)?;
    let expected = expected_sources(&config)?;
    let mut seen_expected = HashSet::new();
    let mut seen_nodes = BTreeSet::new();
    let mut topology_records = Vec::with_capacity(snapshot.tensor_records().len());
    let mut future_bf16_tensors = 0_usize;
    let mut future_f32_tensors = 0_usize;

    for source in snapshot.tensor_records() {
        if is_qwen35_dense_vision_source_tensor(&source.name) {
            ensure!(
                source.disposition == SourcePrecisionDisposition::Excluded
                    && matches!(
                        map_tensor_name(&source.name, &source.shape, &mapper),
                        Some(MappedTensor::Drop)
                    ),
                "vision source {} is not exactly excluded by the dense mapper",
                source.name
            );
            topology_records.push(source_record(
                source,
                Qwen35SourceUse::ExcludedVision,
                Vec::new(),
            ));
            continue;
        }
        let expected_source = expected
            .get(source.name.as_str())
            .with_context(|| format!("unexpected dense-Qwen text source {}", source.name))?;
        ensure!(
            seen_expected.insert(source.name.as_str()),
            "duplicate dense-Qwen source {}",
            source.name
        );
        ensure!(
            source.dtype == SourcePrecisionDType::Bf16,
            "dense-Qwen BF16 topology rejects non-BF16 text source {}",
            source.name
        );
        ensure!(
            source.shape == expected_source.shape,
            "dense-Qwen source {} has shape {:?}, expected {:?}",
            source.name,
            source.shape,
            expected_source.shape
        );
        verify_mapper(source, expected_source, &mapper)?;
        if expected_source.source_use == Qwen35SourceUse::AuthenticatedNonExecutedMtp {
            ensure!(
                matches!(
                    source.disposition,
                    SourcePrecisionDisposition::Fixed | SourcePrecisionDisposition::Protected
                ),
                "MTP source {} must remain fixed or protected",
                source.name
            );
        } else {
            ensure!(
                source.disposition != SourcePrecisionDisposition::Excluded,
                "executed text source {} cannot be excluded",
                source.name
            );
        }
        for output in &expected_source.outputs {
            ensure!(
                seen_nodes.insert(output.node_id.as_str()),
                "future tensor node {} is duplicated",
                output.node_id
            );
            match output.dtype {
                Qwen35FutureDType::Bf16 => future_bf16_tensors += 1,
                Qwen35FutureDType::F32 => future_f32_tensors += 1,
            }
        }
        topology_records.push(source_record(
            source,
            expected_source.source_use,
            expected_source.outputs.clone(),
        ));
    }
    ensure!(
        seen_expected.len() == expected.len(),
        "dense-Qwen source snapshot is missing {} required text tensors",
        expected.len().saturating_sub(seen_expected.len())
    );
    topology_records.sort_by(|left, right| left.source_name.cmp(&right.source_name));
    let topology_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(&TopologyHashView {
        schema_version: TOPOLOGY_SCHEMA_VERSION,
        profile: TOPOLOGY_PROFILE,
        source_snapshot_catalog_sha256: snapshot.catalog_sha256(),
        verified_source_manifest_sha256: snapshot.verified_source_manifest_sha256(),
        source_inventory_manifest_sha256: snapshot.source_inventory_manifest_sha256(),
        tensor_partition_manifest_sha256: snapshot.tensor_partition_manifest_sha256(),
        config: &config,
        records: &topology_records,
    })?));
    Ok((
        topology_records,
        future_bf16_tensors,
        future_f32_tensors,
        topology_sha256,
    ))
}

fn topology_config(
    config: &super::super::Qwen35Config,
    multimodal_wrapping: bool,
) -> Result<Qwen35TopologyConfigV1> {
    let intermediate_size = config
        .intermediate_size
        .context("dense Qwen config lacks intermediate size")?;
    Ok(Qwen35TopologyConfigV1 {
        hidden_size: usize::try_from(config.hidden_size)?,
        intermediate_size: usize::try_from(intermediate_size)?,
        vocabulary_size: usize::try_from(config.vocab_size)?,
        num_hidden_layers: usize::try_from(config.num_hidden_layers)?,
        num_attention_heads: usize::try_from(config.num_attention_heads)?,
        num_key_value_heads: usize::try_from(config.num_key_value_heads)?,
        head_dim: usize::try_from(config.head_dim)?,
        linear_num_key_heads: usize::try_from(config.linear_num_key_heads)?,
        linear_num_value_heads: usize::try_from(config.linear_num_value_heads)?,
        linear_head_dim: usize::try_from(config.linear_key_head_dim)?,
        linear_conv_kernel_dim: usize::try_from(config.linear_conv_kernel_dim)?,
        layer_types: config
            .layer_types
            .iter()
            .map(|kind| match kind {
                Qwen35LayerKind::LinearAttention => "linear_attention",
                Qwen35LayerKind::FullAttention => "full_attention",
            })
            .collect(),
        mtp_num_hidden_layers: usize::try_from(config.mtp_num_hidden_layers)?,
        multimodal_wrapping,
    })
}

fn source_record(
    source: &SourcePrecisionTensorRecord,
    source_use: Qwen35SourceUse,
    outputs: Vec<Qwen35FutureTensorRecord>,
) -> Qwen35SourceTopologyRecord {
    Qwen35SourceTopologyRecord {
        source_name: source.name.clone(),
        source_shape: source.shape.clone(),
        source_byte_sha256: source.byte_sha256.clone(),
        disposition: source.disposition,
        source_use,
        outputs,
    }
}

fn verify_mapper(
    source: &SourcePrecisionTensorRecord,
    expected: &ExpectedSource,
    mapper: &crate::convert::arch::qwen35_dense::Qwen35DenseCtx,
) -> Result<()> {
    let mapped = map_tensor_name(&source.name, &source.shape, mapper)
        .with_context(|| format!("production mapper rejected source {}", source.name))?;
    match (expected.mapped_bake.as_ref(), mapped) {
        (None, MappedTensor::Direct(name)) if Some(&name) == expected.mapped_name.as_ref() => {
            Ok(())
        }
        (Some(expected_bake), MappedTensor::DirectWithBake { gguf_name, bake })
            if Some(&gguf_name) == expected.mapped_name.as_ref() && &bake == expected_bake =>
        {
            Ok(())
        }
        (_, other) => bail!(
            "production mapper outcome for {} differs from the source-teacher topology: {other:?}",
            source.name
        ),
    }
}
