//! Executed cache/input observations derived from exact loaded evidence.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use crate::convert::tensor_lineage::ConversionSourceDisposition;
use crate::core::provenance::tensor_execution::logical_f32_sha256;

use super::loaded::{
    buffer_storage_bytes, f32_bytes_sha256, LoadedTensorCodec, VerifiedLoadedTensorCatalog,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum ExecutedTensorTransform {
    CpuF32Retained,
    GpuF32CopyRuntimeBind,
    DirectF32RuntimeBind,
    DirectGgmlRuntimeBind,
    DeltaConvTwoTransposesRuntimeBind {
        channels: u32,
        kernel_width: u32,
    },
    Qwen35LoadQ4Amax7V1,
    SplitInterleavedQGateThenQwen35LoadQ4Amax7V1 {
        branch_role: String,
        heads: u32,
        head_dim: u32,
        hidden_size: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct ExecutedTensorObservation {
    pub node_id: String,
    pub source_tensor_name: String,
    pub semantic_name: String,
    pub shape_outermost_first: Vec<u64>,
    pub transform: ExecutedTensorTransform,
    pub loaded_parent_bytes_sha256: String,
    pub loaded_parent_logical_f32_sha256: String,
    pub transformed_f32_bytes_sha256: Option<String>,
    pub transformed_logical_f32_sha256: Option<String>,
    pub executed_codec: LoadedTensorCodec,
    pub executed_byte_len: u64,
    pub executed_byte_sha256: String,
    pub executed_logical_f32_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct NonExecutedLoadedTensorObservation {
    pub tensor_name: String,
    pub source_hf_tensor_name: String,
    pub source_disposition: ConversionSourceDisposition,
    pub reason: String,
}

/// Exact cache/input observations after the copied Qwen model is primed.
/// This does not include dispatch traces or command-buffer completion.
pub(crate) struct VerifiedExecutedTensorCatalog {
    loaded_catalog_sha256: String,
    observations: Vec<ExecutedTensorObservation>,
    non_executed_loaded_tensors: Vec<NonExecutedLoadedTensorObservation>,
    catalog_sha256: String,
}

impl VerifiedExecutedTensorCatalog {
    pub(crate) fn observations(&self) -> &[ExecutedTensorObservation] {
        &self.observations
    }

    pub(crate) fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub(crate) fn loaded_catalog_sha256(&self) -> &str {
        &self.loaded_catalog_sha256
    }

    pub(crate) fn non_executed_loaded_tensors(&self) -> &[NonExecutedLoadedTensorObservation] {
        &self.non_executed_loaded_tensors
    }
}

pub(crate) struct ExecutedTensorCatalogBuilder<'a> {
    loaded: &'a VerifiedLoadedTensorCatalog,
    consumed_sources: BTreeSet<String>,
    observations: BTreeMap<String, ExecutedTensorObservation>,
}

impl<'a> ExecutedTensorCatalogBuilder<'a> {
    pub(crate) fn new(loaded: &'a VerifiedLoadedTensorCatalog) -> Self {
        Self {
            loaded,
            consumed_sources: BTreeSet::new(),
            observations: BTreeMap::new(),
        }
    }

    fn insert(&mut self, observation: ExecutedTensorObservation) -> Result<()> {
        if self
            .observations
            .insert(observation.node_id.clone(), observation)
            .is_some()
        {
            bail!("executed tensor node was observed more than once");
        }
        Ok(())
    }

    fn loaded_f32_identity(
        &mut self,
        source_tensor_name: &str,
        values: &[f32],
    ) -> Result<(String, String, Vec<u64>)> {
        let source = self.loaded.observation(source_tensor_name)?;
        if source.codec != LoadedTensorCodec::DenseF32 {
            bail!("{source_tensor_name} was not loaded as dense F32");
        }
        let raw = f32_bytes_sha256(values);
        let logical = logical_f32_sha256(&source.shape_outermost_first, values)?;
        if raw != source.byte_sha256 || logical != source.logical_f32_sha256 {
            bail!("runtime F32 source {source_tensor_name} differs from loaded evidence");
        }
        self.consumed_sources.insert(source_tensor_name.to_owned());
        Ok((raw, logical, source.shape_outermost_first.clone()))
    }

    pub(crate) fn add_cpu_f32(
        &mut self,
        source_tensor_name: &str,
        semantic_name: &str,
        values: &[f32],
    ) -> Result<()> {
        let (raw, logical, shape) = self.loaded_f32_identity(source_tensor_name, values)?;
        self.insert(ExecutedTensorObservation {
            node_id: format!("executed:{semantic_name}"),
            source_tensor_name: source_tensor_name.to_owned(),
            semantic_name: semantic_name.to_owned(),
            shape_outermost_first: shape,
            transform: ExecutedTensorTransform::CpuF32Retained,
            loaded_parent_bytes_sha256: raw.clone(),
            loaded_parent_logical_f32_sha256: logical.clone(),
            transformed_f32_bytes_sha256: Some(raw.clone()),
            transformed_logical_f32_sha256: Some(logical.clone()),
            executed_codec: LoadedTensorCodec::DenseF32,
            executed_byte_len: u64::try_from(values.len())?
                .checked_mul(4)
                .context("CPU F32 executed byte length overflow")?,
            executed_byte_sha256: raw,
            executed_logical_f32_sha256: logical,
        })
    }

    pub(crate) fn add_gpu_f32(
        &mut self,
        source_tensor_name: &str,
        semantic_name: &str,
        values: &[f32],
        executed: &mlx_native::MlxBuffer,
    ) -> Result<()> {
        let (raw, logical, shape) = self.loaded_f32_identity(source_tensor_name, values)?;
        let executed_values = executed
            .as_slice::<f32>()
            .map_err(|error| anyhow::anyhow!("read executed F32 {semantic_name}: {error}"))?;
        if executed_values != values || executed.data_byte_len() != values.len() * 4 {
            bail!("executed F32 {semantic_name} differs from loaded values");
        }
        self.insert(ExecutedTensorObservation {
            node_id: format!("executed:{semantic_name}"),
            source_tensor_name: source_tensor_name.to_owned(),
            semantic_name: semantic_name.to_owned(),
            shape_outermost_first: shape,
            transform: ExecutedTensorTransform::GpuF32CopyRuntimeBind,
            loaded_parent_bytes_sha256: raw.clone(),
            loaded_parent_logical_f32_sha256: logical.clone(),
            transformed_f32_bytes_sha256: Some(raw.clone()),
            transformed_logical_f32_sha256: Some(logical.clone()),
            executed_codec: LoadedTensorCodec::DenseF32,
            executed_byte_len: u64::try_from(executed.data_byte_len())?,
            executed_byte_sha256: raw,
            executed_logical_f32_sha256: logical,
        })
    }

    pub(crate) fn add_direct_f32(
        &mut self,
        source_tensor_name: &str,
        semantic_name: &str,
        executed: &mlx_native::MlxBuffer,
    ) -> Result<()> {
        if !executed.is_file_backed() {
            bail!("executed F32 {semantic_name} is not file backed");
        }
        let values = executed
            .as_slice::<f32>()
            .map_err(|error| anyhow::anyhow!("read direct F32 {semantic_name}: {error}"))?;
        let (raw, logical, shape) = self.loaded_f32_identity(source_tensor_name, values)?;
        if executed.data_byte_len() != values.len() * std::mem::size_of::<f32>() {
            bail!("direct F32 {semantic_name} has inconsistent byte extent");
        }
        self.insert(ExecutedTensorObservation {
            node_id: format!("executed:{semantic_name}"),
            source_tensor_name: source_tensor_name.to_owned(),
            semantic_name: semantic_name.to_owned(),
            shape_outermost_first: shape,
            transform: ExecutedTensorTransform::DirectF32RuntimeBind,
            loaded_parent_bytes_sha256: raw.clone(),
            loaded_parent_logical_f32_sha256: logical.clone(),
            transformed_f32_bytes_sha256: None,
            transformed_logical_f32_sha256: None,
            executed_codec: LoadedTensorCodec::DenseF32,
            executed_byte_len: u64::try_from(executed.data_byte_len())?,
            executed_byte_sha256: raw,
            executed_logical_f32_sha256: logical,
        })
    }

    pub(crate) fn add_direct_ggml(
        &mut self,
        source_tensor_name: &str,
        semantic_name: &str,
        executed: &mlx_native::MlxBuffer,
    ) -> Result<()> {
        let source = self.loaded.observation(source_tensor_name)?;
        let LoadedTensorCodec::Ggml { .. } = &source.codec else {
            bail!("{source_tensor_name} was not loaded in native GGUF storage");
        };
        let bytes = buffer_storage_bytes(executed)
            .map_err(|error| anyhow::anyhow!("read executed GGUF {semantic_name}: {error}"))?;
        let data_len = bytes.len();
        let byte_sha = hex::encode(Sha256::digest(bytes));
        if u64::try_from(data_len)? != source.byte_len || byte_sha != source.byte_sha256 {
            bail!("executed GGUF {semantic_name} differs from loaded storage");
        }
        self.consumed_sources.insert(source_tensor_name.to_owned());
        self.insert(ExecutedTensorObservation {
            node_id: format!("executed:{semantic_name}"),
            source_tensor_name: source_tensor_name.to_owned(),
            semantic_name: semantic_name.to_owned(),
            shape_outermost_first: source.shape_outermost_first.clone(),
            transform: ExecutedTensorTransform::DirectGgmlRuntimeBind,
            loaded_parent_bytes_sha256: source.byte_sha256.clone(),
            loaded_parent_logical_f32_sha256: source.logical_f32_sha256.clone(),
            transformed_f32_bytes_sha256: None,
            transformed_logical_f32_sha256: None,
            executed_codec: source.codec.clone(),
            executed_byte_len: source.byte_len,
            executed_byte_sha256: byte_sha,
            executed_logical_f32_sha256: source.logical_f32_sha256.clone(),
        })
    }

    pub(crate) fn add_delta_conv_roundtrip(
        &mut self,
        source_tensor_name: &str,
        semantic_name: &str,
        cpu_kernel_channels: &[f32],
        executed: &mlx_native::MlxBuffer,
        channels: u32,
        kernel_width: u32,
    ) -> Result<()> {
        let channels_usize = usize::try_from(channels)?;
        let kernel_usize = usize::try_from(kernel_width)?;
        let elements = channels_usize
            .checked_mul(kernel_usize)
            .context("Delta conv geometry overflow")?;
        if cpu_kernel_channels.len() != elements {
            bail!("Delta conv CPU transpose has the wrong element count");
        }
        let mut restored_channels_kernel = vec![0.0_f32; elements];
        for kernel in 0..kernel_usize {
            for channel in 0..channels_usize {
                restored_channels_kernel[channel * kernel_usize + kernel] =
                    cpu_kernel_channels[kernel * channels_usize + channel];
            }
        }
        let source = self.loaded.observation(source_tensor_name)?;
        let raw = f32_bytes_sha256(&restored_channels_kernel);
        let logical = logical_f32_sha256(&source.shape_outermost_first, &restored_channels_kernel)?;
        if source.codec != LoadedTensorCodec::DenseF32
            || raw != source.byte_sha256
            || logical != source.logical_f32_sha256
        {
            bail!("Delta conv transpose roundtrip differs from loaded evidence");
        }
        let executed_values = executed.as_slice::<f32>().map_err(|error| {
            anyhow::anyhow!("read executed Delta conv {semantic_name}: {error}")
        })?;
        if executed_values != restored_channels_kernel || executed.data_byte_len() != elements * 4 {
            bail!("executed Delta conv bytes differ after the second transpose");
        }
        self.consumed_sources.insert(source_tensor_name.to_owned());
        let transformed_raw = f32_bytes_sha256(cpu_kernel_channels);
        let transformed_logical = logical_f32_sha256(
            &[u64::from(kernel_width), u64::from(channels)],
            cpu_kernel_channels,
        )?;
        self.insert(ExecutedTensorObservation {
            node_id: format!("executed:{semantic_name}"),
            source_tensor_name: source_tensor_name.to_owned(),
            semantic_name: semantic_name.to_owned(),
            shape_outermost_first: source.shape_outermost_first.clone(),
            transform: ExecutedTensorTransform::DeltaConvTwoTransposesRuntimeBind {
                channels,
                kernel_width,
            },
            loaded_parent_bytes_sha256: raw.clone(),
            loaded_parent_logical_f32_sha256: logical.clone(),
            transformed_f32_bytes_sha256: Some(transformed_raw),
            transformed_logical_f32_sha256: Some(transformed_logical),
            executed_codec: LoadedTensorCodec::DenseF32,
            executed_byte_len: u64::try_from(elements)?
                .checked_mul(4)
                .context("executed Delta conv byte length overflow")?,
            executed_byte_sha256: raw,
            executed_logical_f32_sha256: logical,
        })
    }

    pub(crate) fn add_amax7_q4(
        &mut self,
        source_tensor_name: &str,
        semantic_name: &str,
        values: &[f32],
        executed: &mlx_native::MlxBuffer,
    ) -> Result<()> {
        let (raw, logical, shape) = self.loaded_f32_identity(source_tensor_name, values)?;
        let transformed_raw = f32_bytes_sha256(values);
        let transformed_logical = logical_f32_sha256(&shape, values)?;
        self.insert_amax7(
            source_tensor_name,
            semantic_name,
            values,
            executed,
            shape,
            raw,
            logical,
            transformed_raw,
            transformed_logical,
            ExecutedTensorTransform::Qwen35LoadQ4Amax7V1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_amax7(
        &mut self,
        source_tensor_name: &str,
        semantic_name: &str,
        values: &[f32],
        executed: &mlx_native::MlxBuffer,
        shape: Vec<u64>,
        loaded_parent_raw: String,
        loaded_parent_logical: String,
        transformed_raw: String,
        transformed_logical: String,
        transform: ExecutedTensorTransform,
    ) -> Result<()> {
        let expected_bytes = super::super::gpu_full_attn::encode_q4_0_blocks(values);
        let data_len = executed.data_byte_len();
        let allocated = executed
            .as_slice::<u8>()
            .map_err(|error| anyhow::anyhow!("read executed Q4_0 {semantic_name}: {error}"))?;
        let actual_bytes = allocated
            .get(..data_len)
            .context("executed Q4_0 byte extent exceeds allocation")?;
        if actual_bytes != expected_bytes {
            bail!("executed Q4_0 {semantic_name} differs from the production amax/7 encoder");
        }
        let decoded = decode_q4_0_amax7(actual_bytes)?;
        let executed_logical = logical_f32_sha256(&shape, &decoded)?;
        self.insert(ExecutedTensorObservation {
            node_id: format!("executed:{semantic_name}"),
            source_tensor_name: source_tensor_name.to_owned(),
            semantic_name: semantic_name.to_owned(),
            shape_outermost_first: shape,
            transform,
            loaded_parent_bytes_sha256: loaded_parent_raw,
            loaded_parent_logical_f32_sha256: loaded_parent_logical,
            transformed_f32_bytes_sha256: Some(transformed_raw),
            transformed_logical_f32_sha256: Some(transformed_logical),
            executed_codec: LoadedTensorCodec::Ggml {
                type_name: "q4_0".into(),
                wire_type_id: 2,
            },
            executed_byte_len: u64::try_from(data_len)?,
            executed_byte_sha256: hex::encode(Sha256::digest(actual_bytes)),
            executed_logical_f32_sha256: executed_logical,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_interleaved_q_gate_amax7(
        &mut self,
        source_tensor_name: &str,
        q_semantic_name: &str,
        gate_semantic_name: &str,
        q_values: &[f32],
        gate_values: &[f32],
        q_executed: &mlx_native::MlxBuffer,
        gate_executed: &mlx_native::MlxBuffer,
        heads: u32,
        head_dim: u32,
        hidden_size: u32,
    ) -> Result<()> {
        let rows = usize::try_from(heads)?
            .checked_mul(usize::try_from(head_dim)?)
            .context("Q/gate row count overflow")?;
        let hidden = usize::try_from(hidden_size)?;
        let branch_values = rows
            .checked_mul(hidden)
            .context("Q/gate branch element count overflow")?;
        if q_values.len() != branch_values || gate_values.len() != branch_values {
            bail!("Q/gate split branch geometry differs from the execution configuration");
        }
        let mut fused = vec![
            0.0_f32;
            branch_values
                .checked_mul(2)
                .context("fused Q size overflow")?
        ];
        let rows_per_head = usize::try_from(head_dim)?;
        for head in 0..usize::try_from(heads)? {
            let branch_start = head * rows_per_head * hidden;
            let fused_q_start = head * 2 * rows_per_head * hidden;
            let fused_gate_start = fused_q_start + rows_per_head * hidden;
            let branch_end = branch_start + rows_per_head * hidden;
            fused[fused_q_start..fused_q_start + rows_per_head * hidden]
                .copy_from_slice(&q_values[branch_start..branch_end]);
            fused[fused_gate_start..fused_gate_start + rows_per_head * hidden]
                .copy_from_slice(&gate_values[branch_start..branch_end]);
        }
        let source = self.loaded.observation(source_tensor_name)?;
        let fused_raw = f32_bytes_sha256(&fused);
        let fused_logical = logical_f32_sha256(&source.shape_outermost_first, &fused)?;
        if fused_raw != source.byte_sha256 || fused_logical != source.logical_f32_sha256 {
            bail!("Q/gate split does not reconstruct the exact loaded fused tensor");
        }
        self.consumed_sources.insert(source_tensor_name.to_owned());
        let branch_shape = vec![u64::try_from(rows)?, u64::from(hidden_size)];
        let q_raw = f32_bytes_sha256(q_values);
        let q_logical = logical_f32_sha256(&branch_shape, q_values)?;
        self.insert_amax7(
            source_tensor_name,
            q_semantic_name,
            q_values,
            q_executed,
            branch_shape.clone(),
            fused_raw.clone(),
            fused_logical.clone(),
            q_raw,
            q_logical,
            ExecutedTensorTransform::SplitInterleavedQGateThenQwen35LoadQ4Amax7V1 {
                branch_role: "q".into(),
                heads,
                head_dim,
                hidden_size,
            },
        )?;
        let gate_raw = f32_bytes_sha256(gate_values);
        let gate_logical = logical_f32_sha256(&branch_shape, gate_values)?;
        self.insert_amax7(
            source_tensor_name,
            gate_semantic_name,
            gate_values,
            gate_executed,
            branch_shape,
            fused_raw,
            fused_logical,
            gate_raw,
            gate_logical,
            ExecutedTensorTransform::SplitInterleavedQGateThenQwen35LoadQ4Amax7V1 {
                branch_role: "gate".into(),
                heads,
                head_dim,
                hidden_size,
            },
        )
    }

    pub(crate) fn finalize(self) -> Result<VerifiedExecutedTensorCatalog> {
        let mut expected_sources = BTreeSet::new();
        let mut non_executed_loaded_tensors = Vec::new();
        for observation in self.loaded.observations() {
            if observation.source_hf_tensor_name.starts_with("mtp.") {
                if !matches!(
                    observation.source_disposition,
                    ConversionSourceDisposition::Fixed | ConversionSourceDisposition::Protected
                ) {
                    bail!(
                        "MTP tensor {} must be fixed or protected in the dense text execution profile",
                        observation.source_hf_tensor_name
                    );
                }
                if self.consumed_sources.contains(&observation.tensor_name) {
                    bail!(
                        "MTP tensor {} was consumed by the main autoregressive execution graph",
                        observation.source_hf_tensor_name
                    );
                }
                non_executed_loaded_tensors.push(NonExecutedLoadedTensorObservation {
                    tensor_name: observation.tensor_name.clone(),
                    source_hf_tensor_name: observation.source_hf_tensor_name.clone(),
                    source_disposition: observation.source_disposition,
                    reason: "mtp_stored_fixed_or_protected_not_executed_by_base_text_profile_v1"
                        .into(),
                });
            } else {
                expected_sources.insert(observation.tensor_name.clone());
                let executed = self
                    .observations
                    .values()
                    .filter(|executed| executed.source_tensor_name == observation.tensor_name)
                    .collect::<Vec<_>>();
                if executed.iter().any(|executed| {
                    executed.loaded_parent_bytes_sha256 != observation.byte_sha256
                        || executed.loaded_parent_logical_f32_sha256
                            != observation.logical_f32_sha256
                }) {
                    bail!(
                        "executed lineage for {} does not bind the exact loaded parent",
                        observation.tensor_name
                    );
                }
                if observation.tensor_name.ends_with(".attn_q.weight") {
                    let native_fused = executed.len() == 1
                        && matches!(
                            executed[0].transform,
                            ExecutedTensorTransform::DirectGgmlRuntimeBind
                        );
                    let roles = executed
                        .iter()
                        .filter_map(|executed| {
                            match &executed.transform {
                            ExecutedTensorTransform::SplitInterleavedQGateThenQwen35LoadQ4Amax7V1 {
                                branch_role,
                                ..
                            } => Some(branch_role.as_str()),
                            _ => None,
                        }
                        })
                        .collect::<BTreeSet<_>>();
                    let split_reencoded =
                        executed.len() == 2 && roles == BTreeSet::from(["gate", "q"]);
                    if !native_fused && !split_reencoded {
                        bail!(
                            "fused Q source {} must remain one native projection or produce exactly q and gate executed branches",
                            observation.tensor_name
                        );
                    }
                } else if executed.len() != 1
                    || matches!(
                        executed[0].transform,
                        ExecutedTensorTransform::SplitInterleavedQGateThenQwen35LoadQ4Amax7V1 { .. }
                    )
                {
                    bail!(
                        "ordinary loaded source {} must produce exactly one executed tensor",
                        observation.tensor_name
                    );
                }
            }
        }
        if self.consumed_sources != expected_sources {
            bail!(
                "executed tensor coverage differs from loaded evidence: expected {:?}, observed {:?}",
                expected_sources,
                self.consumed_sources
            );
        }
        let observations = self.observations.into_values().collect::<Vec<_>>();
        let catalog_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(&(
            self.loaded.catalog_sha256(),
            &observations,
            &non_executed_loaded_tensors,
        ))?));
        Ok(VerifiedExecutedTensorCatalog {
            loaded_catalog_sha256: self.loaded.catalog_sha256().to_owned(),
            observations,
            non_executed_loaded_tensors,
            catalog_sha256,
        })
    }
}

fn decode_q4_0_amax7(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 18 != 0 {
        bail!("Q4_0 executed bytes are not an integral block stream");
    }
    let mut values = Vec::with_capacity(bytes.len() / 18 * 32);
    for block in bytes.chunks_exact(18) {
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let mut decoded = [0.0_f32; 32];
        for (index, packed) in block[2..].iter().copied().enumerate() {
            decoded[index] = f32::from((packed & 0x0f) as i8 - 8) * scale;
            decoded[index + 16] = f32::from((packed >> 4) as i8 - 8) * scale;
        }
        values.extend_from_slice(&decoded);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_mtp_is_retained_as_explicitly_non_executed() {
        let loaded = VerifiedLoadedTensorCatalog::for_test_mtp(ConversionSourceDisposition::Fixed);
        let catalog = ExecutedTensorCatalogBuilder::new(&loaded)
            .finalize()
            .unwrap();
        assert!(catalog.observations().is_empty());
        assert_eq!(catalog.non_executed_loaded_tensors().len(), 1);
        assert_eq!(
            catalog.non_executed_loaded_tensors()[0].reason,
            "mtp_stored_fixed_or_protected_not_executed_by_base_text_profile_v1"
        );
    }

    #[test]
    fn variable_mtp_is_rejected_by_the_base_text_profile() {
        let loaded =
            VerifiedLoadedTensorCatalog::for_test_mtp(ConversionSourceDisposition::Variable);
        let error = match ExecutedTensorCatalogBuilder::new(&loaded).finalize() {
            Ok(_) => panic!("variable MTP unexpectedly entered the base text profile"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("must be fixed or protected"));
    }

    #[test]
    fn amax7_decoder_uses_low_then_high_nibbles_with_exact_scale() {
        let mut block = vec![0_u8; 18];
        block[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        for (index, packed) in block[2..].iter_mut().enumerate() {
            let low = u8::try_from(index).unwrap();
            let high = 15 - low;
            *packed = low | (high << 4);
        }
        let decoded = decode_q4_0_amax7(&block).unwrap();
        for index in 0..16 {
            assert_eq!(decoded[index], (index as f32 - 8.0) * 0.5);
            assert_eq!(decoded[index + 16], (7.0 - index as f32) * 0.5);
        }
    }
}
