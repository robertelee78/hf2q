use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::core::provenance::{compute_source_bundle_sha256, SourceShard};
use crate::input::integrity::VerifiedSourceManifest;
use crate::intelligence::measured_auto_quant::SourceIdentity;

use super::super::{tensor_catalog_sha256, ScalarDType, TensorAllocationUnit, TensorMember};
use super::types::*;

fn hash_serialized<T: Serialize>(value: &T) -> Result<String, DynamicProducerError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| DynamicProducerError::Serialization(error.to_string()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn source_valid(source: &SourceIdentity) -> bool {
    !source.model_id.is_empty()
        && !source.revision.is_empty()
        && is_sha256(&source.config_sha256)
        && is_sha256(&source.tensor_bundle_sha256)
        && is_sha256(&source.tokenizer_bundle_sha256)
        && is_sha256(&source.chat_template_sha256)
}

fn record_valid(record: &SourceTensorRecord) -> bool {
    !record.name.is_empty()
        && !record.source_shape.is_empty()
        && record.source_shape.iter().all(|dimension| *dimension > 0)
        && !record.source_dtype.is_empty()
        && record.source_byte_len > 0
        && is_sha256(&record.source_tensor_sha256)
}

fn scalar_dtype_name(dtype: &ScalarDType) -> &'static str {
    match dtype {
        ScalarDType::F16 => "F16",
        ScalarDType::Bf16 => "BF16",
        ScalarDType::F32 => "F32",
    }
}

fn member_matches_record(member: &TensorMember, record: &SourceTensorRecord) -> bool {
    member.name == record.name
        && member.shape == record.source_shape
        && scalar_dtype_name(&member.source_dtype) == record.source_dtype
        && member.source_tensor_sha256 == record.source_tensor_sha256
}

#[derive(Serialize)]
struct InventoryHashView<'a> {
    schema_version: u32,
    source: &'a SourceIdentity,
    verified_source_manifest_sha256: &'a str,
    tensors: &'a [SourceTensorRecord],
}

pub(super) fn inventory_hash(
    inventory: &SourceTensorInventoryManifest,
) -> Result<String, DynamicProducerError> {
    hash_serialized(&InventoryHashView {
        schema_version: inventory.schema_version,
        source: &inventory.source,
        verified_source_manifest_sha256: &inventory.verified_source_manifest_sha256,
        tensors: &inventory.tensors,
    })
}

fn verified_source_manifest_sha256(
    manifest: &VerifiedSourceManifest,
) -> Result<String, DynamicProducerError> {
    hash_serialized(manifest)
}

fn verified_source_bundle_sha256(manifest: &VerifiedSourceManifest) -> Option<String> {
    let shards: Vec<_> = manifest
        .records()
        .iter()
        .map(SourceShard::from_integrity)
        .collect();
    compute_source_bundle_sha256(&shards)
}

fn source_record_matches_bytes(
    record: &crate::core::integrity::ShardIntegrity,
    bytes: &[u8],
) -> bool {
    if u64::try_from(bytes.len()).ok() != Some(record.bytes) {
        return false;
    }
    if let Some(expected) = &record.sha256 {
        return hex::encode(Sha256::digest(bytes)).eq_ignore_ascii_case(expected);
    }
    use sha1::Sha1;
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", bytes.len()).as_bytes());
    hasher.update(bytes);
    hex::encode(hasher.finalize()).eq_ignore_ascii_case(record.hf_etag.trim().trim_matches('"'))
}

/// Read the exact tensor catalog and bytes from an already authenticated
/// source snapshot. This is intentionally the only production constructor for
/// [`VerifiedSourceTensorInventory`].
pub fn derive_source_tensor_inventory(
    model_dir: &Path,
    source: SourceIdentity,
    verified_source: &VerifiedSourceManifest,
) -> Result<VerifiedSourceTensorInventory, DynamicProducerError> {
    if verified_source.repo() != source.model_id || verified_source.revision() != source.revision {
        return Err(DynamicProducerError::InvalidInventory(
            "verified source manifest is bound to a different repository or revision".into(),
        ));
    }
    let reverified = crate::input::integrity::verify_conversion_manifest(
        &source.model_id,
        &source.revision,
        model_dir,
        verified_source.records().to_vec(),
    )
    .map_err(|error| DynamicProducerError::InvalidInventory(error.to_string()))?;
    let config_bytes = std::fs::read(model_dir.join("config.json"))
        .map_err(|error| DynamicProducerError::InvalidInventory(error.to_string()))?;
    let config_record = reverified
        .records()
        .iter()
        .find(|record| record.filename == "config.json")
        .ok_or_else(|| {
            DynamicProducerError::InvalidInventory(
                "verified source manifest is missing config.json".into(),
            )
        })?;
    if !source_record_matches_bytes(config_record, &config_bytes)
        || hex::encode(Sha256::digest(&config_bytes)) != source.config_sha256
    {
        return Err(DynamicProducerError::InvalidInventory(
            "source config bytes do not match config.json identity".into(),
        ));
    }
    if verified_source_bundle_sha256(&reverified).as_deref()
        != Some(source.tensor_bundle_sha256.as_str())
    {
        return Err(DynamicProducerError::InvalidInventory(
            "source tensor-bundle hash does not match the verified file manifest".into(),
        ));
    }
    let records: BTreeMap<_, _> = reverified
        .records()
        .iter()
        .map(|record| (record.filename.as_str(), record))
        .collect();
    let mut seen_names = BTreeSet::new();
    let mut tensors = Vec::new();
    for shard_name in reverified.required_weight_shards() {
        let record = records.get(shard_name.as_str()).ok_or_else(|| {
            DynamicProducerError::InvalidInventory(format!(
                "authenticated shard set is missing {shard_name}"
            ))
        })?;
        // Owned bytes are both authenticated and parsed. Peak evidence memory
        // is one source shard; no mutable file-backed mmap survives hashing.
        let bytes = std::fs::read(model_dir.join(shard_name))
            .map_err(|error| DynamicProducerError::InvalidInventory(error.to_string()))?;
        if !source_record_matches_bytes(record, &bytes) {
            return Err(DynamicProducerError::InvalidInventory(format!(
                "source shard {shard_name} changed after verification"
            )));
        }
        let parsed = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|error| DynamicProducerError::InvalidInventory(error.to_string()))?;
        let mut names = parsed.names();
        names.sort_unstable();
        for name in names {
            if !seen_names.insert(name.to_owned()) {
                return Err(DynamicProducerError::InvalidInventory(format!(
                    "tensor {name} is duplicated across source shards"
                )));
            }
            let tensor = parsed
                .tensor(name)
                .map_err(|error| DynamicProducerError::InvalidInventory(error.to_string()))?;
            let source_byte_len = u64::try_from(tensor.data().len()).map_err(|_| {
                DynamicProducerError::InvalidInventory(format!(
                    "tensor {name} byte length is not representable"
                ))
            })?;
            tensors.push(SourceTensorRecord {
                name: name.to_owned(),
                source_shape: tensor.shape().to_vec(),
                source_dtype: format!("{:?}", tensor.dtype()),
                source_byte_len,
                source_tensor_sha256: hex::encode(Sha256::digest(tensor.data())),
            });
        }
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let mut inventory = SourceTensorInventoryManifest {
        schema_version: DYNAMIC_PRODUCER_SCHEMA_VERSION,
        source,
        verified_source_manifest_sha256: verified_source_manifest_sha256(&reverified)?,
        tensors,
        manifest_sha256: String::new(),
    };
    inventory.manifest_sha256 = inventory_hash(&inventory)?;
    validate_source_tensor_inventory(&inventory)?;
    Ok(VerifiedSourceTensorInventory {
        manifest: inventory,
    })
}

/// Independently reread a claimed source inventory and compare every tensor
/// name, shape, dtype, byte count, and raw hash.
pub fn verify_source_tensor_inventory_from_source(
    claimed: &SourceTensorInventoryManifest,
    model_dir: &Path,
    source: SourceIdentity,
    verified_source: &VerifiedSourceManifest,
) -> Result<VerifiedSourceTensorInventory, DynamicProducerError> {
    validate_source_tensor_inventory(claimed)?;
    let rebuilt = derive_source_tensor_inventory(model_dir, source, verified_source)?;
    if rebuilt.manifest != *claimed {
        return Err(DynamicProducerError::InvalidInventory(
            "source inventory does not reproduce from verified safetensors".into(),
        ));
    }
    Ok(rebuilt)
}

pub fn validate_source_tensor_inventory(
    inventory: &SourceTensorInventoryManifest,
) -> Result<(), DynamicProducerError> {
    if inventory.schema_version != DYNAMIC_PRODUCER_SCHEMA_VERSION
        || !source_valid(&inventory.source)
        || !is_sha256(&inventory.verified_source_manifest_sha256)
        || inventory.tensors.is_empty()
        || inventory
            .tensors
            .windows(2)
            .any(|pair| pair[0].name >= pair[1].name)
        || inventory.tensors.iter().any(|record| !record_valid(record))
        || inventory_hash(inventory)? != inventory.manifest_sha256
    {
        return Err(DynamicProducerError::InvalidInventory(
            "identity, source-derived order, tensor metadata, or hash is invalid".into(),
        ));
    }
    Ok(())
}

pub(super) fn descriptor_from_unit(
    unit: &TensorAllocationUnit,
) -> Result<VariableUnitDescriptor, DynamicProducerError> {
    let mut members = unit.members.clone();
    members.sort_by(|left, right| left.name.cmp(&right.name));
    let mut expected_expert_ids = unit.expected_expert_ids.clone();
    expected_expert_ids.sort_unstable();
    let member_names: BTreeSet<_> = members.iter().map(|member| member.name.as_str()).collect();
    let mut operations = unit.operations.clone();
    let mut covered = BTreeSet::new();
    for operation in &mut operations {
        operation.tensor_names.sort();
        if operation.operation_id.is_empty()
            || operation.graph_path.is_empty()
            || operation.tensor_names.is_empty()
            || operation
                .tensor_names
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || operation
                .tensor_names
                .iter()
                .any(|name| !member_names.contains(name.as_str()))
        {
            return Err(DynamicProducerError::InvalidPartition(format!(
                "allocation unit {} has an invalid logical operation",
                unit.unit_id
            )));
        }
        covered.extend(operation.tensor_names.iter().cloned());
    }
    operations.sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
    if unit.unit_id.is_empty()
        || members.is_empty()
        || members.windows(2).any(|pair| pair[0].name >= pair[1].name)
        || operations.is_empty()
        || operations
            .windows(2)
            .any(|pair| pair[0].operation_id >= pair[1].operation_id)
        || covered.len() != member_names.len()
        || expected_expert_ids
            .windows(2)
            .any(|pair| pair[0] == pair[1])
    {
        return Err(DynamicProducerError::InvalidPartition(format!(
            "allocation unit {} is empty or contains duplicate members/experts",
            unit.unit_id
        )));
    }
    Ok(VariableUnitDescriptor {
        unit_id: unit.unit_id.clone(),
        members,
        expected_expert_ids,
        operations,
    })
}

#[derive(Serialize)]
struct PartitionHashView<'a> {
    schema_version: u32,
    source: &'a SourceIdentity,
    source_inventory_manifest_sha256: &'a str,
    source_tensor_count: usize,
    variable_units: &'a [VariableUnitDescriptor],
    non_variable_tensors: &'a [NonVariableTensor],
    tensor_catalog_sha256: &'a str,
}

fn partition_hash(partition: &TensorPartitionManifest) -> Result<String, DynamicProducerError> {
    hash_serialized(&PartitionHashView {
        schema_version: partition.schema_version,
        source: &partition.source,
        source_inventory_manifest_sha256: &partition.source_inventory_manifest_sha256,
        source_tensor_count: partition.source_tensor_count,
        variable_units: &partition.variable_units,
        non_variable_tensors: &partition.non_variable_tensors,
        tensor_catalog_sha256: &partition.tensor_catalog_sha256,
    })
}

pub fn build_tensor_partition(
    inventory: &VerifiedSourceTensorInventory,
    units: &[TensorAllocationUnit],
    mut non_variable_tensors: Vec<NonVariableTensor>,
) -> Result<TensorPartitionManifest, DynamicProducerError> {
    validate_source_tensor_inventory(&inventory.manifest)?;
    let mut variable_units: Vec<_> = units
        .iter()
        .map(descriptor_from_unit)
        .collect::<Result<_, _>>()?;
    variable_units.sort_by(|left, right| left.unit_id.cmp(&right.unit_id));
    non_variable_tensors.sort_by(|left, right| left.source.name.cmp(&right.source.name));
    let tensor_catalog_sha256 = tensor_catalog_sha256(units)
        .map_err(|error| DynamicProducerError::InvalidPartition(error.to_string()))?;
    let mut partition = TensorPartitionManifest {
        schema_version: DYNAMIC_PRODUCER_SCHEMA_VERSION,
        source: inventory.manifest.source.clone(),
        source_inventory_manifest_sha256: inventory.manifest.manifest_sha256.clone(),
        source_tensor_count: inventory.manifest.tensors.len(),
        variable_units,
        non_variable_tensors,
        tensor_catalog_sha256,
        manifest_sha256: String::new(),
    };
    partition.manifest_sha256 = partition_hash(&partition)?;
    validate_tensor_partition(&partition, inventory, units)?;
    Ok(partition)
}

pub fn validate_tensor_partition(
    partition: &TensorPartitionManifest,
    inventory: &VerifiedSourceTensorInventory,
    units: &[TensorAllocationUnit],
) -> Result<(), DynamicProducerError> {
    validate_source_tensor_inventory(&inventory.manifest)?;
    let source_records: BTreeMap<_, _> = inventory
        .manifest
        .tensors
        .iter()
        .map(|record| (record.name.as_str(), record))
        .collect();
    let mut covered = BTreeSet::new();
    let mut descriptors: Vec<_> = units
        .iter()
        .map(descriptor_from_unit)
        .collect::<Result<_, _>>()?;
    descriptors.sort_by(|left, right| left.unit_id.cmp(&right.unit_id));
    if descriptors
        .windows(2)
        .any(|pair| pair[0].unit_id >= pair[1].unit_id)
    {
        return Err(DynamicProducerError::InvalidPartition(
            "variable allocation-unit identities must be unique".into(),
        ));
    }
    for descriptor in &descriptors {
        for member in &descriptor.members {
            if !source_records
                .get(member.name.as_str())
                .is_some_and(|record| member_matches_record(member, record))
                || !covered.insert(member.name.as_str())
            {
                return Err(DynamicProducerError::InvalidPartition(format!(
                    "variable tensor {} is missing, drifted, or duplicated",
                    member.name
                )));
            }
        }
    }
    for tensor in &partition.non_variable_tensors {
        if tensor.reason.is_empty()
            || source_records.get(tensor.source.name.as_str()) != Some(&&tensor.source)
            || !covered.insert(tensor.source.name.as_str())
        {
            return Err(DynamicProducerError::InvalidPartition(format!(
                "non-variable tensor {} is invalid, drifted, or duplicated",
                tensor.source.name
            )));
        }
    }
    if partition.schema_version != DYNAMIC_PRODUCER_SCHEMA_VERSION
        || partition.source != inventory.manifest.source
        || partition.source_inventory_manifest_sha256 != inventory.manifest.manifest_sha256
        || partition.source_tensor_count != inventory.manifest.tensors.len()
        || partition.variable_units != descriptors
        || partition
            .non_variable_tensors
            .windows(2)
            .any(|pair| pair[0].source.name >= pair[1].source.name)
        || covered.len() != inventory.manifest.tensors.len()
        || !source_records.keys().all(|name| covered.contains(name))
        || partition.tensor_catalog_sha256
            != tensor_catalog_sha256(units)
                .map_err(|error| DynamicProducerError::InvalidPartition(error.to_string()))?
        || partition_hash(partition)? != partition.manifest_sha256
    {
        return Err(DynamicProducerError::InvalidPartition(
            "source coverage, atomic grouping, expert topology, catalog, or hash is invalid".into(),
        ));
    }
    Ok(())
}

pub(super) fn producer_hash<T: Serialize>(value: &T) -> Result<String, DynamicProducerError> {
    hash_serialized(value)
}

pub(super) fn valid_sha256(value: &str) -> bool {
    is_sha256(value)
}
