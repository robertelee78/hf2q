use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};
use serde::Serialize;
use sha2::{Digest, Sha256};
use sysinfo::System;

use super::snapshot::VerifiedQwenSourceSnapshot;
use super::topology::{Qwen35FutureDType, Qwen35SourceTransformV1, VerifiedQwen35Bf16TopologyV1};
use super::types::SourcePrecisionDisposition;
use super::upload_plan::{
    preflight_upload, QwenSourceMetalCapacityV1, QwenSourceMetalUploadLimits,
    QwenSourceMetalUploadPreflightV1,
};
use super::upload_transform::{upload_source, UploadedTensorBuffer};

mod teacher_model;

pub(crate) use teacher_model::{
    prepare_qwen35_source_teacher, prepare_qwen35_source_teacher_run_inputs,
    prepare_uploaded_qwen35_source_teacher, PreparedQwen35SourceTeacherRunInputsV1,
    PreparedQwen35SourceTeacherV1, Qwen35SourceTeacherLimitsV1,
    Qwen35SourceTeacherPreparationPolicyV1,
};

const UPLOAD_SCHEMA_VERSION: u32 = 1;
const UPLOAD_PROFILE: &str = "dense_qwen35_source_bf16_host_verified_metal_upload_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct QwenUploadedOutputRecordV1 {
    node_id: String,
    shape: Vec<usize>,
    dtype: Qwen35FutureDType,
    transform: Qwen35SourceTransformV1,
    byte_len: u64,
    buffer_byte_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct QwenUploadedSourceRecordV1 {
    source_name: String,
    source_shape: Vec<usize>,
    source_byte_sha256: String,
    disposition: SourcePrecisionDisposition,
    source_use: super::topology::Qwen35SourceUse,
    outputs: Vec<QwenUploadedOutputRecordV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct QwenMetalDeviceIdentityV1 {
    name: String,
    registry_id: u64,
    residency_sets_enabled: bool,
}

#[derive(Serialize)]
struct UploadCatalogHashView<'a> {
    schema_version: u32,
    profile: &'static str,
    topology_sha256: &'a str,
    source_snapshot_catalog_sha256: &'a str,
    device: &'a QwenMetalDeviceIdentityV1,
    records: &'a [QwenUploadedSourceRecordV1],
}

#[derive(Serialize)]
struct UploadReceiptHashView<'a> {
    schema_version: u32,
    profile: &'static str,
    topology_sha256: &'a str,
    source_snapshot_catalog_sha256: &'a str,
    device: &'a QwenMetalDeviceIdentityV1,
    limits: QwenSourceMetalUploadLimits,
    capacity: QwenSourceMetalCapacityV1,
    preflight: QwenSourceMetalUploadPreflightV1,
    records: &'a [QwenUploadedSourceRecordV1],
    catalog_sha256: &'a str,
}

/// Process-local receipt for host-populated shared Metal storage. This proves
/// exact source-to-buffer bytes and allocation bounds only; it is not a GPU
/// dispatch, completion, numerical, teacher, sensitivity, or allocator proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct QwenSourceMetalUploadReceiptV1 {
    schema_version: u32,
    profile: &'static str,
    topology_sha256: String,
    source_snapshot_catalog_sha256: String,
    device: QwenMetalDeviceIdentityV1,
    limits: QwenSourceMetalUploadLimits,
    capacity: QwenSourceMetalCapacityV1,
    preflight: QwenSourceMetalUploadPreflightV1,
    records: Vec<QwenUploadedSourceRecordV1>,
    catalog_sha256: String,
    receipt_sha256: String,
}

/// Opaque ownership of the exact host-populated BF16/F32 Metal buffers.
/// `MlxBuffer` is cloneable and CPU-writable, so no buffer reference escapes
/// this type before the later family-owned teacher constructor consumes it.
pub(crate) struct VerifiedQwen35Bf16MetalUploadV1 {
    _snapshot: VerifiedQwenSourceSnapshot,
    _device: MlxDevice,
    _buffers: BTreeMap<String, MlxBuffer>,
    _buffer_addresses: BTreeMap<String, usize>,
    receipt: QwenSourceMetalUploadReceiptV1,
}

impl VerifiedQwen35Bf16MetalUploadV1 {
    pub(crate) fn catalog_sha256(&self) -> &str {
        &self.receipt.catalog_sha256
    }

    pub(crate) fn receipt_sha256(&self) -> &str {
        &self.receipt.receipt_sha256
    }

    pub(crate) fn output_tensor_count(&self) -> usize {
        self.receipt.preflight.output_tensor_count
    }

    pub(crate) fn total_output_bytes(&self) -> u64 {
        self.receipt.preflight.total_output_bytes
    }

    #[cfg(test)]
    pub(super) fn receipt_json_for_test(&self) -> serde_json::Value {
        serde_json::to_value(&self.receipt).expect("upload receipt must serialize")
    }

    #[cfg(test)]
    pub(super) fn validate_receipt_for_test(&self) -> Result<()> {
        let expected_catalog = hex::encode(Sha256::digest(serde_json::to_vec(
            &UploadCatalogHashView {
                schema_version: self.receipt.schema_version,
                profile: self.receipt.profile,
                topology_sha256: &self.receipt.topology_sha256,
                source_snapshot_catalog_sha256: &self.receipt.source_snapshot_catalog_sha256,
                device: &self.receipt.device,
                records: &self.receipt.records,
            },
        )?));
        ensure!(
            expected_catalog == self.receipt.catalog_sha256,
            "source Metal upload catalog hash does not reproduce"
        );
        let expected_receipt = hex::encode(Sha256::digest(serde_json::to_vec(
            &UploadReceiptHashView {
                schema_version: self.receipt.schema_version,
                profile: self.receipt.profile,
                topology_sha256: &self.receipt.topology_sha256,
                source_snapshot_catalog_sha256: &self.receipt.source_snapshot_catalog_sha256,
                device: &self.receipt.device,
                limits: self.receipt.limits,
                capacity: self.receipt.capacity,
                preflight: self.receipt.preflight,
                records: &self.receipt.records,
                catalog_sha256: &self.receipt.catalog_sha256,
            },
        )?));
        ensure!(
            expected_receipt == self.receipt.receipt_sha256,
            "source Metal upload receipt hash does not reproduce"
        );
        let mut output_count = 0_usize;
        for record in &self.receipt.records {
            for output in &record.outputs {
                let buffer = self
                    ._buffers
                    .get(&output.node_id)
                    .context("source Metal upload receipt output lacks its owned buffer")?;
                let bytes: &[u8] = match output.dtype {
                    Qwen35FutureDType::Bf16 => bytemuck::cast_slice(buffer.as_slice::<u16>()?),
                    Qwen35FutureDType::F32 => bytemuck::cast_slice(buffer.as_slice::<f32>()?),
                };
                ensure!(
                    u64::try_from(bytes.len())? == output.byte_len
                        && hex::encode(Sha256::digest(bytes)) == output.buffer_byte_sha256,
                    "source Metal upload output {} does not reproduce",
                    output.node_id
                );
                output_count += 1;
            }
        }
        ensure!(
            output_count == self.receipt.preflight.output_tensor_count
                && self._buffers.len() == output_count
                && self._buffer_addresses.len() == output_count,
            "source Metal upload output cardinality does not reproduce"
        );
        for (node_id, buffer) in &self._buffers {
            ensure!(
                self._buffer_addresses.get(node_id).copied()
                    == Some(buffer.contents_ptr() as usize),
                "source Metal upload output {node_id} changed allocation identity"
            );
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn buffer_bytes_for_test(&self, node_id: &str) -> Option<Vec<u8>> {
        let buffer = self._buffers.get(node_id)?;
        Some(match buffer.dtype() {
            DType::BF16 => bytemuck::cast_slice(buffer.as_slice::<u16>().ok()?).to_vec(),
            DType::F32 => bytemuck::cast_slice(buffer.as_slice::<f32>().ok()?).to_vec(),
            _ => return None,
        })
    }
}

pub(crate) fn upload_qwen35_bf16_topology_to_metal(
    topology: VerifiedQwen35Bf16TopologyV1,
    device: &MlxDevice,
    limits: QwenSourceMetalUploadLimits,
) -> Result<VerifiedQwen35Bf16MetalUploadV1> {
    let capacity = observe_capacity(device);
    upload_with_capacity(topology, device, limits, capacity, |bytes, dtype, shape| {
        device
            .alloc_buffer(bytes, dtype, shape)
            .map_err(anyhow::Error::from)
    })
}

fn observe_capacity(device: &MlxDevice) -> QwenSourceMetalCapacityV1 {
    let mut system = System::new();
    system.refresh_memory();
    let metal = device.metal_device();
    QwenSourceMetalCapacityV1 {
        host_available_bytes: system.available_memory(),
        metal_recommended_working_set_bytes: metal.recommended_max_working_set_size(),
        metal_current_allocated_bytes: metal.current_allocated_size() as u64,
        metal_max_buffer_bytes: metal.max_buffer_length() as u64,
    }
}

fn upload_with_capacity<A>(
    topology: VerifiedQwen35Bf16TopologyV1,
    device: &MlxDevice,
    limits: QwenSourceMetalUploadLimits,
    capacity: QwenSourceMetalCapacityV1,
    mut allocate: A,
) -> Result<VerifiedQwen35Bf16MetalUploadV1>
where
    A: FnMut(usize, DType, Vec<usize>) -> Result<MlxBuffer>,
{
    let (snapshot, topology_records, topology_sha256, expected_bf16, expected_f32) =
        topology.into_upload_parts();
    let preflight = preflight_upload(
        &snapshot,
        &topology_records,
        expected_bf16,
        expected_f32,
        limits,
        capacity,
    )?;
    let device_identity = QwenMetalDeviceIdentityV1 {
        name: device.name(),
        registry_id: device.registry_id(),
        residency_sets_enabled: device.residency_sets_enabled(),
    };

    let mut scratch = Vec::new();
    let mut buffers = BTreeMap::new();
    let mut buffer_addresses = BTreeMap::new();
    let mut allocation_addresses = std::collections::BTreeSet::new();
    let mut checked_allocate = |bytes, dtype, shape| {
        let buffer = allocate(bytes, dtype, shape)?;
        ensure!(
            allocation_addresses.insert(buffer.contents_ptr() as usize),
            "source Metal allocator returned an aliased allocation"
        );
        Ok(buffer)
    };
    let mut records = Vec::with_capacity(topology_records.len());
    for source in &topology_records {
        let uploaded = upload_source(
            &snapshot,
            source,
            device_identity.registry_id,
            &mut scratch,
            &mut checked_allocate,
        )
        .with_context(|| format!("failed to upload source {}", source.source_name))?;
        ensure!(
            uploaded.len() == source.outputs.len(),
            "uploaded source {} output count differs from B2a",
            source.source_name
        );
        let outputs = source
            .outputs
            .iter()
            .zip(uploaded)
            .map(|(planned, actual)| {
                ensure!(
                    planned.node_id == actual.node_id
                        && planned.shape == actual.shape
                        && planned.dtype == actual.dtype,
                    "uploaded output differs from its B2a descriptor"
                );
                let UploadedTensorBuffer {
                    node_id,
                    shape,
                    dtype,
                    byte_len,
                    buffer_byte_sha256,
                    buffer,
                } = actual;
                ensure!(
                    buffer_addresses
                        .insert(node_id.clone(), buffer.contents_ptr() as usize)
                        .is_none(),
                    "uploaded output node {node_id} has duplicated allocation identity"
                );
                ensure!(
                    buffers.insert(node_id.clone(), buffer).is_none(),
                    "uploaded output node {node_id} is duplicated"
                );
                Ok(QwenUploadedOutputRecordV1 {
                    node_id,
                    shape,
                    dtype,
                    transform: planned.transform.clone(),
                    byte_len,
                    buffer_byte_sha256,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        records.push(QwenUploadedSourceRecordV1 {
            source_name: source.source_name.clone(),
            source_shape: source.source_shape.clone(),
            source_byte_sha256: source.source_byte_sha256.clone(),
            disposition: source.disposition,
            source_use: source.source_use,
            outputs,
        });
    }
    ensure!(
        buffers.len() == preflight.output_tensor_count,
        "uploaded Metal buffer count differs from preflight"
    );
    snapshot.rehash_retained_files()?;

    let source_snapshot_catalog_sha256 = snapshot.catalog_sha256().to_owned();
    let catalog_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(
        &UploadCatalogHashView {
            schema_version: UPLOAD_SCHEMA_VERSION,
            profile: UPLOAD_PROFILE,
            topology_sha256: &topology_sha256,
            source_snapshot_catalog_sha256: &source_snapshot_catalog_sha256,
            device: &device_identity,
            records: &records,
        },
    )?));
    let receipt_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(
        &UploadReceiptHashView {
            schema_version: UPLOAD_SCHEMA_VERSION,
            profile: UPLOAD_PROFILE,
            topology_sha256: &topology_sha256,
            source_snapshot_catalog_sha256: &source_snapshot_catalog_sha256,
            device: &device_identity,
            limits,
            capacity,
            preflight,
            records: &records,
            catalog_sha256: &catalog_sha256,
        },
    )?));
    let receipt = QwenSourceMetalUploadReceiptV1 {
        schema_version: UPLOAD_SCHEMA_VERSION,
        profile: UPLOAD_PROFILE,
        topology_sha256,
        source_snapshot_catalog_sha256,
        device: device_identity,
        limits,
        capacity,
        preflight,
        records,
        catalog_sha256,
        receipt_sha256,
    };
    Ok(VerifiedQwen35Bf16MetalUploadV1 {
        _snapshot: snapshot,
        _device: device.clone(),
        _buffers: buffers,
        _buffer_addresses: buffer_addresses,
        receipt,
    })
}

#[cfg(test)]
pub(super) fn upload_with_capacity_for_test<A>(
    topology: VerifiedQwen35Bf16TopologyV1,
    device: &MlxDevice,
    limits: QwenSourceMetalUploadLimits,
    capacity: QwenSourceMetalCapacityV1,
    allocate: A,
) -> Result<VerifiedQwen35Bf16MetalUploadV1>
where
    A: FnMut(usize, DType, Vec<usize>) -> Result<MlxBuffer>,
{
    upload_with_capacity(topology, device, limits, capacity, allocate)
}
