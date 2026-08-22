//! Loaded-buffer capture bound to the retained D2b artifact.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::convert::tensor_lineage::{
    ConversionSourceDisposition, VerifiedSourceToStoredConversion,
};
use crate::core::provenance::tensor_execution::logical_f32_sha256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum LoadedTensorCodec {
    DenseF32,
    Ggml {
        type_name: String,
        wire_type_id: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct LoadedTensorObservation {
    pub tensor_name: String,
    pub source_hf_tensor_name: String,
    pub source_disposition: ConversionSourceDisposition,
    pub materialization_count: u32,
    pub shape_outermost_first: Vec<u64>,
    pub codec: LoadedTensorCodec,
    pub byte_len: u64,
    pub byte_sha256: String,
    pub logical_f32_sha256: String,
}

/// Opaque catalog emitted only while the production loader consumes the
/// retained D2b artifact parser. It proves loaded bytes, not GPU execution.
pub(crate) struct VerifiedLoadedTensorCatalog {
    conversion_receipt_sha256: String,
    observations: Vec<LoadedTensorObservation>,
    catalog_sha256: String,
}

impl VerifiedLoadedTensorCatalog {
    pub(crate) fn conversion_receipt_sha256(&self) -> &str {
        &self.conversion_receipt_sha256
    }

    pub(crate) fn observations(&self) -> &[LoadedTensorObservation] {
        &self.observations
    }

    pub(crate) fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub(crate) fn observation(&self, tensor_name: &str) -> Result<&LoadedTensorObservation> {
        self.observations
            .binary_search_by(|observation| observation.tensor_name.as_str().cmp(tensor_name))
            .ok()
            .map(|index| &self.observations[index])
            .with_context(|| format!("loaded catalog does not contain {tensor_name}"))
    }

    #[cfg(test)]
    pub(crate) fn for_test_empty(conversion_receipt_sha256: String) -> Self {
        Self {
            catalog_sha256: hex::encode(Sha256::digest(
                serde_json::to_vec(&(
                    &conversion_receipt_sha256,
                    Vec::<LoadedTensorObservation>::new(),
                ))
                .unwrap(),
            )),
            conversion_receipt_sha256,
            observations: Vec::new(),
        }
    }
    #[cfg(test)]
    pub(crate) fn for_test_mtp(disposition: ConversionSourceDisposition) -> Self {
        Self {
            conversion_receipt_sha256: "a".repeat(64),
            observations: vec![LoadedTensorObservation {
                tensor_name: "mtp.0.proj.weight".into(),
                source_hf_tensor_name: "mtp.0.proj.weight".into(),
                source_disposition: disposition,
                materialization_count: 1,
                shape_outermost_first: vec![1, 32],
                codec: LoadedTensorCodec::DenseF32,
                byte_len: 128,
                byte_sha256: "b".repeat(64),
                logical_f32_sha256: "c".repeat(64),
            }],
            catalog_sha256: "d".repeat(64),
        }
    }
}

#[derive(Clone)]
struct ExpectedLoadedTensor {
    source_hf_tensor_name: String,
    source_disposition: ConversionSourceDisposition,
    shape_outermost_first: Vec<u64>,
    ggml_wire_type_id: u32,
    ggml_type_name: String,
    payload_bytes: u64,
    payload_sha256: String,
    stored_f32_bytes_sha256: String,
    stored_logical_f32_sha256: String,
}

struct LoadedObservationState {
    conversion_receipt_sha256: String,
    expected: BTreeMap<String, ExpectedLoadedTensor>,
    observed: BTreeMap<String, LoadedTensorObservation>,
}

thread_local! {
    static ACTIVE_LOAD_OBSERVATIONS: RefCell<Option<LoadedObservationState>> =
        const { RefCell::new(None) };
}

struct LoadedObservationGuard;

impl Drop for LoadedObservationGuard {
    fn drop(&mut self) {
        ACTIVE_LOAD_OBSERVATIONS.with(|slot| {
            slot.replace(None);
        });
    }
}

pub(crate) fn f32_bytes_sha256(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

/// Borrow the exact logical storage bytes without lying about the buffer's
/// element type. This is observation-only: it neither converts values nor
/// allocates a shadow representation.
pub(super) fn buffer_storage_bytes(buffer: &mlx_native::MlxBuffer) -> Result<&[u8]> {
    use mlx_native::DType;

    let bytes = match buffer.dtype() {
        DType::F32 => bytemuck::cast_slice(buffer.as_slice::<f32>()?),
        DType::F16 => bytemuck::cast_slice(buffer.as_slice::<half::f16>()?),
        DType::BF16 => bytemuck::cast_slice(buffer.as_slice::<half::bf16>()?),
        DType::U8 => buffer.as_slice::<u8>()?,
        DType::U16 => bytemuck::cast_slice(buffer.as_slice::<u16>()?),
        DType::U32 => bytemuck::cast_slice(buffer.as_slice::<u32>()?),
        DType::I32 => bytemuck::cast_slice(buffer.as_slice::<i32>()?),
    };
    if bytes.len() != buffer.data_byte_len() {
        bail!(
            "typed buffer byte view length {} != logical extent {}",
            bytes.len(),
            buffer.data_byte_len()
        );
    }
    Ok(bytes)
}

fn record_loaded_observation(
    state: &mut LoadedObservationState,
    tensor_name: &str,
    mut observation: LoadedTensorObservation,
) -> Result<()> {
    if let Some(existing) = state.observed.get_mut(tensor_name) {
        observation.materialization_count = existing.materialization_count;
        if *existing != observation {
            bail!("repeated loaded tensor {tensor_name} changed physical evidence");
        }
        existing.materialization_count = existing
            .materialization_count
            .checked_add(1)
            .context("loaded tensor materialization count overflow")?;
        return Ok(());
    }
    state.observed.insert(tensor_name.to_owned(), observation);
    Ok(())
}

pub(crate) fn capture_loaded_tensor_catalog<T>(
    conversion: &VerifiedSourceToStoredConversion,
    operation: impl FnOnce() -> Result<T>,
) -> Result<(T, VerifiedLoadedTensorCatalog)> {
    let mut expected = BTreeMap::new();
    for lineage in &conversion.receipt().tensor_lineages {
        if expected
            .insert(
                lineage.gguf_tensor_name.clone(),
                ExpectedLoadedTensor {
                    source_hf_tensor_name: lineage.hf_tensor_name.clone(),
                    source_disposition: lineage.disposition,
                    shape_outermost_first: lineage.stored.shape_outermost_first.clone(),
                    ggml_wire_type_id: lineage.stored.ggml_wire_type_id,
                    ggml_type_name: lineage.stored.ggml_type_name.clone(),
                    payload_bytes: lineage.stored.payload_bytes,
                    payload_sha256: lineage.stored.payload_sha256.clone(),
                    stored_f32_bytes_sha256: lineage.stored.stored_f32_bytes_sha256.clone(),
                    stored_logical_f32_sha256: lineage.stored.stored_logical_f32_sha256.clone(),
                },
            )
            .is_some()
        {
            bail!(
                "stored conversion receipt repeats GGUF tensor {}",
                lineage.gguf_tensor_name
            );
        }
    }
    ACTIVE_LOAD_OBSERVATIONS.with(|slot| -> Result<()> {
        if slot.borrow().is_some() {
            bail!("nested Qwen loaded-buffer observation is not admitted");
        }
        slot.replace(Some(LoadedObservationState {
            conversion_receipt_sha256: conversion.receipt().receipt_sha256.clone(),
            expected,
            observed: BTreeMap::new(),
        }));
        Ok(())
    })?;
    let guard = LoadedObservationGuard;
    let value = operation()?;
    let state = ACTIVE_LOAD_OBSERVATIONS.with(|slot| slot.replace(None));
    std::mem::forget(guard);
    let state = state.context("loaded-buffer observation state disappeared")?;
    let expected_names = state.expected.keys().collect::<Vec<_>>();
    let observed_names = state.observed.keys().collect::<Vec<_>>();
    if expected_names != observed_names {
        bail!(
            "loaded tensor coverage differs from the stored receipt: expected {:?}, observed {:?}",
            expected_names,
            observed_names
        );
    }
    let observations = state.observed.into_values().collect::<Vec<_>>();
    let catalog_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(&(
        &state.conversion_receipt_sha256,
        &observations,
    ))?));
    Ok((
        value,
        VerifiedLoadedTensorCatalog {
            conversion_receipt_sha256: state.conversion_receipt_sha256,
            observations,
            catalog_sha256,
        },
    ))
}

pub(crate) fn observe_loaded_f32(tensor_name: &str, values: &[f32]) -> Result<()> {
    ACTIVE_LOAD_OBSERVATIONS.with(|slot| {
        let mut state = slot.borrow_mut();
        let Some(state) = state.as_mut() else {
            return Ok(());
        };
        let expected = state
            .expected
            .get(tensor_name)
            .with_context(|| format!("loaded unexpected tensor {tensor_name}"))?;
        let logical_sha = logical_f32_sha256(&expected.shape_outermost_first, values)?;
        let byte_sha = f32_bytes_sha256(values);
        if logical_sha != expected.stored_logical_f32_sha256
            || byte_sha != expected.stored_f32_bytes_sha256
        {
            bail!("loaded F32 tensor {tensor_name} differs from reopened GGUF evidence");
        }
        let byte_len = u64::try_from(values.len())?
            .checked_mul(4)
            .context("loaded F32 byte length overflow")?;
        let observation = LoadedTensorObservation {
            tensor_name: tensor_name.to_owned(),
            source_hf_tensor_name: expected.source_hf_tensor_name.clone(),
            source_disposition: expected.source_disposition,
            materialization_count: 1,
            shape_outermost_first: expected.shape_outermost_first.clone(),
            codec: LoadedTensorCodec::DenseF32,
            byte_len,
            byte_sha256: byte_sha,
            logical_f32_sha256: logical_sha,
        };
        record_loaded_observation(state, tensor_name, observation)
    })
}

pub(crate) fn observe_loaded_ggml(tensor_name: &str, buffer: &mlx_native::MlxBuffer) -> Result<()> {
    ACTIVE_LOAD_OBSERVATIONS.with(|slot| {
        let mut state = slot.borrow_mut();
        let Some(state) = state.as_mut() else {
            return Ok(());
        };
        let expected = state
            .expected
            .get(tensor_name)
            .with_context(|| format!("loaded unexpected tensor {tensor_name}"))?;
        let bytes = buffer_storage_bytes(buffer)
            .map_err(|error| anyhow::anyhow!("read loaded GGUF tensor {tensor_name}: {error}"))?;
        let data_len = bytes.len();
        let byte_sha = hex::encode(Sha256::digest(bytes));
        if u64::try_from(data_len)? != expected.payload_bytes || byte_sha != expected.payload_sha256
        {
            bail!("loaded GGUF tensor {tensor_name} differs from reopened payload evidence");
        }
        let observation = LoadedTensorObservation {
            tensor_name: tensor_name.to_owned(),
            source_hf_tensor_name: expected.source_hf_tensor_name.clone(),
            source_disposition: expected.source_disposition,
            materialization_count: 1,
            shape_outermost_first: expected.shape_outermost_first.clone(),
            codec: LoadedTensorCodec::Ggml {
                type_name: expected.ggml_type_name.clone(),
                wire_type_id: expected.ggml_wire_type_id,
            },
            byte_len: expected.payload_bytes,
            byte_sha256: byte_sha,
            logical_f32_sha256: expected.stored_logical_f32_sha256.clone(),
        };
        record_loaded_observation(state, tensor_name, observation)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_bf16_storage_is_hashed_as_exact_bytes_without_an_untyped_view() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = mlx_native::MlxDevice::new().expect("Metal device");
        let bits = [0x3f80_u16, 0xbf00, 0x0001, 0x7f80];
        let mut buffer = device
            .alloc_buffer(bits.len() * 2, mlx_native::DType::BF16, vec![2, 2])
            .expect("BF16 buffer");
        buffer
            .as_mut_slice::<u16>()
            .expect("BF16 storage bits")
            .copy_from_slice(&bits);
        let expected = bits
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(buffer_storage_bytes(&buffer).unwrap(), expected);
    }

    #[test]
    fn repeated_shared_head_load_is_counted_only_when_bytes_are_identical() {
        let mut state = LoadedObservationState {
            conversion_receipt_sha256: "a".repeat(64),
            expected: BTreeMap::new(),
            observed: BTreeMap::new(),
        };
        let observation = LoadedTensorObservation {
            tensor_name: "output.weight".into(),
            source_hf_tensor_name: "lm_head.weight".into(),
            source_disposition: ConversionSourceDisposition::Fixed,
            materialization_count: 1,
            shape_outermost_first: vec![32, 32],
            codec: LoadedTensorCodec::DenseF32,
            byte_len: 4096,
            byte_sha256: "b".repeat(64),
            logical_f32_sha256: "c".repeat(64),
        };
        record_loaded_observation(&mut state, "output.weight", observation.clone()).unwrap();
        record_loaded_observation(&mut state, "output.weight", observation.clone()).unwrap();
        assert_eq!(state.observed["output.weight"].materialization_count, 2);
        let mut mutated = observation;
        mutated.byte_sha256 = "d".repeat(64);
        assert!(record_loaded_observation(&mut state, "output.weight", mutated).is_err());
    }
}
