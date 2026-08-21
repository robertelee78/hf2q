use std::collections::{BTreeMap, BTreeSet};
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use safetensors::tensor::Dtype;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::core::provenance::{compute_source_bundle_sha256, SourceShard};
use crate::input::integrity::VerifiedSourceManifest;
use crate::intelligence::dynamic_allocator::producer::{
    validate_tensor_partition, TensorPartitionManifest, VerifiedSourceTensorInventory,
};
use crate::intelligence::dynamic_allocator::TensorAllocationUnit;
use crate::intelligence::measured_auto_quant::SourceIdentity;

use super::header::parse_unique_header;
use super::retained_io::{
    hash_region, hash_retained_file, open_and_read_config, open_retained_directory,
    open_retained_file, read_exact_at, visit_hashed_tensor_region, RetainedSourceFile,
};
use super::scope::{
    parse_unique_qwen_config, source_dispositions, validate_dense_qwen_source_config,
    validate_teacher_dispositions,
};
use super::types::*;

#[derive(Debug)]
struct RetainedSourceShard {
    source: RetainedSourceFile,
}

#[derive(Serialize)]
struct CatalogHashView<'a> {
    schema_version: u32,
    source: &'a SourceIdentity,
    verified_source_manifest_sha256: &'a str,
    source_inventory_manifest_sha256: &'a str,
    tensor_partition_manifest_sha256: &'a str,
    config_sha256: &'a str,
    tensors: &'a [SourcePrecisionTensorRecord],
}

/// Retained, exact-inode source tensor snapshot for the future dense-Qwen
/// source teacher. This type authenticates source bytes and D1 topology only;
/// it is not executable and cannot mint teacher or allocator authority.
#[derive(Debug)]
pub(crate) struct VerifiedQwenSourceSnapshot {
    source: SourceIdentity,
    verified_source_manifest_sha256: String,
    source_inventory_manifest_sha256: String,
    tensor_partition_manifest_sha256: String,
    config: serde_json::Value,
    config_file: RetainedSourceFile,
    shards: Vec<RetainedSourceShard>,
    tensors: Vec<SourcePrecisionTensorRecord>,
    tensor_indices: BTreeMap<String, usize>,
    catalog_sha256: String,
}

impl VerifiedQwenSourceSnapshot {
    pub(crate) fn source(&self) -> &SourceIdentity {
        &self.source
    }

    pub(crate) fn config(&self) -> &serde_json::Value {
        &self.config
    }

    pub(crate) fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub(super) fn tensor_record(&self, name: &str) -> Option<&SourcePrecisionTensorRecord> {
        self.tensor_indices
            .get(name)
            .and_then(|index| self.tensors.get(*index))
    }

    pub(super) fn tensor_records(&self) -> &[SourcePrecisionTensorRecord] {
        &self.tensors
    }

    pub(crate) fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub(crate) fn verified_source_manifest_sha256(&self) -> &str {
        &self.verified_source_manifest_sha256
    }

    pub(crate) fn source_inventory_manifest_sha256(&self) -> &str {
        &self.source_inventory_manifest_sha256
    }

    pub(crate) fn tensor_partition_manifest_sha256(&self) -> &str {
        &self.tensor_partition_manifest_sha256
    }

    #[cfg(test)]
    pub(super) fn tensor_names_for_test(&self) -> Vec<&str> {
        self.tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect()
    }

    /// Copy one retained BF16/F16 payload directly into its final u16 view.
    /// The source bytes are reread positionally and rehashed during the copy;
    /// no whole-tensor host allocation or pathname reopen occurs.
    #[cfg(test)]
    pub(crate) fn read_tensor_u16(&self, name: &str, output: &mut [u16]) -> Result<()> {
        let tensor = self
            .tensor_record(name)
            .ok_or_else(|| anyhow::anyhow!("source tensor {name} is absent"))?;
        let expected_elements = usize::try_from(tensor.byte_len / 2)
            .context("source tensor element count is not representable")?;
        ensure!(
            tensor.byte_len % 2 == 0 && output.len() == expected_elements,
            "source tensor {name} destination has the wrong u16 length"
        );
        let mut scratch = Vec::new();
        let mut output_offset = 0;
        self.visit_tensor_le_bytes(name, &mut scratch, |_, bytes| {
            for pair in bytes.chunks_exact(2) {
                output[output_offset] = u16::from_le_bytes([pair[0], pair[1]]);
                output_offset += 1;
            }
            Ok(())
        })?;
        ensure!(
            output_offset == output.len(),
            "source tensor {name} copied the wrong element count"
        );
        Ok(())
    }

    /// Visit one retained BF16/F16 payload as bounded little-endian byte
    /// chunks while reproducing its authenticated source hash. The caller
    /// owns and reuses `scratch`; this method never allocates a whole tensor
    /// and never reopens a source pathname.
    pub(super) fn visit_tensor_le_bytes<F>(
        &self,
        name: &str,
        scratch: &mut Vec<u8>,
        visit: F,
    ) -> Result<()>
    where
        F: FnMut(usize, &[u8]) -> Result<()>,
    {
        let tensor = self
            .tensor_record(name)
            .ok_or_else(|| anyhow::anyhow!("source tensor {name} is absent"))?;
        let shard = self
            .shards
            .iter()
            .find(|shard| shard.source.filename == tensor.shard_filename)
            .ok_or_else(|| anyhow::anyhow!("source tensor {name} lost its retained shard"))?;
        visit_hashed_tensor_region(
            &shard.source,
            tensor.payload_offset,
            tensor.byte_len,
            &tensor.byte_sha256,
            scratch,
            visit,
        )
    }

    /// Rehash every retained source file after a future upload pass. This is a
    /// persistent-mutation check, not a claim that writable source inodes are
    /// cryptographically immutable against transient write/restore races.
    pub(crate) fn rehash_retained_files(&self) -> Result<()> {
        ensure!(
            hash_retained_file(&self.config_file)? == self.config_file.sha256,
            "retained source config changed"
        );
        for shard in &self.shards {
            ensure!(
                hash_retained_file(&shard.source)? == shard.source.sha256,
                "retained source shard {} changed",
                shard.source.filename
            );
        }
        Ok(())
    }
}

pub(crate) fn open_verified_qwen_source_snapshot(
    model_dir: &Path,
    verified_source: &VerifiedSourceManifest,
    inventory: &VerifiedSourceTensorInventory,
    partition: &TensorPartitionManifest,
    units: &[TensorAllocationUnit],
    limits: QwenSourceSnapshotLimits,
) -> Result<VerifiedQwenSourceSnapshot> {
    limits.validate()?;
    validate_tensor_partition(partition, inventory, units)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let source = &inventory.manifest().source;
    ensure!(
        verified_source.repo() == source.model_id && verified_source.revision() == source.revision,
        "verified source repository/revision differs from D1"
    );
    let manifest_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(verified_source)?));
    ensure!(
        manifest_sha256 == inventory.manifest().verified_source_manifest_sha256,
        "verified source manifest hash differs from D1"
    );
    let bundle = compute_source_bundle_sha256(
        &verified_source
            .records()
            .iter()
            .map(SourceShard::from_integrity)
            .collect::<Vec<_>>(),
    );
    ensure!(
        bundle.as_deref() == Some(source.tensor_bundle_sha256.as_str()),
        "verified source tensor bundle differs from D1"
    );
    ensure!(
        !verified_source.required_weight_shards().is_empty()
            && verified_source.required_weight_shards().len() <= limits.max_shards,
        "source shard count exceeds the v1 bound"
    );

    let mut records = BTreeMap::new();
    for record in verified_source.records() {
        ensure!(
            records.insert(record.filename.as_str(), record).is_none(),
            "verified source manifest contains duplicate file {}",
            record.filename
        );
    }
    let root = open_retained_directory(model_dir)?;
    let config_record = records
        .get("config.json")
        .copied()
        .context("verified source manifest is missing config.json")?;
    ensure!(
        config_record.bytes <= limits.max_config_bytes,
        "source config exceeds the v1 byte bound"
    );
    let (config_file, config_bytes) = open_and_read_config(&root, config_record)?;
    ensure!(
        config_file.sha256 == source.config_sha256,
        "retained config bytes differ from D1 source identity"
    );
    let config = parse_unique_qwen_config(&config_bytes)?;
    validate_dense_qwen_source_config(&config)?;

    let inventory_records: BTreeMap<_, _> = inventory
        .manifest()
        .tensors
        .iter()
        .map(|record| (record.name.as_str(), record))
        .collect();
    let dispositions = source_dispositions(partition)?;
    ensure!(
        inventory_records.len() == dispositions.len()
            && inventory_records.keys().eq(dispositions.keys()),
        "D1 source records and dispositions do not have exact coverage"
    );
    validate_teacher_dispositions(&dispositions)?;

    let mut total_source_bytes = config_record.bytes;
    let mut total_header_bytes = 0_u64;
    let mut seen_names = BTreeSet::new();
    let mut shards = Vec::with_capacity(verified_source.required_weight_shards().len());
    let mut tensors = Vec::with_capacity(inventory_records.len());
    for shard_name in verified_source.required_weight_shards() {
        let record = records
            .get(shard_name.as_str())
            .copied()
            .with_context(|| format!("verified source is missing shard {shard_name}"))?;
        ensure!(
            record.is_lfs && record.sha256.is_some(),
            "source weight shard {shard_name} lacks a strong SHA-256 identity"
        );
        ensure!(
            record.bytes <= MAX_SOURCE_SHARD_BYTES,
            "source weight shard {shard_name} exceeds the file hard bound"
        );
        total_source_bytes = total_source_bytes
            .checked_add(record.bytes)
            .context("source snapshot byte count overflow")?;
        ensure!(
            total_source_bytes <= limits.max_total_source_bytes,
            "source snapshot exceeds its declared byte bound"
        );
        let file = open_retained_file(&root, shard_name)?;
        let metadata = file.metadata()?;
        ensure!(
            metadata.file_type().is_file() && metadata.len() == record.bytes,
            "source shard {shard_name} is not the expected regular file"
        );
        let identity = (metadata.dev(), metadata.ino());

        let mut length_bytes = [0_u8; 8];
        read_exact_at(&file, &mut length_bytes, 0)?;
        let header_bytes_u64 = u64::from_le_bytes(length_bytes);
        ensure!(
            header_bytes_u64 > 0 && header_bytes_u64 <= limits.max_header_bytes_per_shard,
            "source shard {shard_name} header exceeds its bound"
        );
        total_header_bytes = total_header_bytes
            .checked_add(header_bytes_u64)
            .context("source header byte count overflow")?;
        ensure!(
            total_header_bytes <= limits.max_total_header_bytes,
            "source headers exceed their aggregate bound"
        );
        let header_len = usize::try_from(header_bytes_u64)
            .context("source shard header length is not representable")?;
        let mut header_bytes = vec![0_u8; header_len];
        read_exact_at(&file, &mut header_bytes, 8)?;
        let parsed = parse_unique_header(&header_bytes)
            .with_context(|| format!("parse retained source shard {shard_name}"))?;
        let data_offset = 8_u64
            .checked_add(header_bytes_u64)
            .context("source shard data offset overflow")?;
        let data_len = u64::try_from(parsed.data_len())
            .context("source shard data length is not representable")?;
        ensure!(
            data_offset.checked_add(data_len) == Some(record.bytes),
            "source shard {shard_name} has trailing, missing, or overflowed bytes"
        );

        let mut shard_hasher = Sha256::new();
        shard_hasher.update(length_bytes);
        shard_hasher.update(&header_bytes);
        let mut hash_scratch = vec![0_u8; SOURCE_READ_CHUNK_BYTES];
        let mut expected_relative_offset = 0_usize;
        for name in parsed.offset_keys() {
            ensure!(
                !name.is_empty()
                    && name.len() <= MAX_SOURCE_TENSOR_NAME_BYTES
                    && name.is_ascii()
                    && !name.bytes().any(|byte| byte.is_ascii_control()),
                "source tensor name is outside the v1 grammar"
            );
            ensure!(
                seen_names.insert(name.clone()),
                "source tensor {name} is duplicated across shards"
            );
            ensure!(
                seen_names.len() <= limits.max_tensors,
                "source tensor count exceeds the v1 bound"
            );
            let info = parsed
                .info(&name)
                .with_context(|| format!("source tensor {name} lost header metadata"))?;
            ensure!(
                info.data_offsets.0 == expected_relative_offset
                    && info.data_offsets.1 >= info.data_offsets.0,
                "source tensor {name} payloads are not contiguous in header order"
            );
            expected_relative_offset = info.data_offsets.1;
            ensure!(
                !info.shape.is_empty()
                    && info.shape.len() <= MAX_SOURCE_TENSOR_RANK
                    && info.shape.iter().all(|dimension| *dimension > 0),
                "source tensor {name} has invalid rank or dimensions"
            );
            let dtype = match info.dtype {
                Dtype::BF16 => SourcePrecisionDType::Bf16,
                Dtype::F16 => SourcePrecisionDType::F16,
                other => {
                    bail!("source tensor {name} dtype {other:?} is outside BF16/F16 teacher scope")
                }
            };
            let byte_len = info
                .data_offsets
                .1
                .checked_sub(info.data_offsets.0)
                .and_then(|value| u64::try_from(value).ok())
                .context("source tensor byte length overflow")?;
            let numel = info
                .shape
                .iter()
                .try_fold(1_usize, |product, dimension| {
                    product.checked_mul(*dimension)
                })
                .context("source tensor element count overflow")?;
            ensure!(
                u64::try_from(numel)
                    .ok()
                    .and_then(|value| value.checked_mul(2))
                    == Some(byte_len),
                "source tensor {name} BF16/F16 geometry is inconsistent"
            );
            let inventory_record = inventory_records
                .get(name.as_str())
                .copied()
                .with_context(|| format!("source tensor {name} is absent from D1"))?;
            let dtype_name = match dtype {
                SourcePrecisionDType::Bf16 => "BF16",
                SourcePrecisionDType::F16 => "F16",
            };
            ensure!(
                inventory_record.source_shape == info.shape
                    && inventory_record.source_dtype == dtype_name
                    && inventory_record.source_byte_len == byte_len,
                "source tensor {name} geometry differs from D1"
            );
            let payload_offset = data_offset
                .checked_add(u64::try_from(info.data_offsets.0)?)
                .context("source tensor payload offset overflow")?;
            let tensor_sha256 = hash_region(
                &file,
                payload_offset,
                byte_len,
                &mut shard_hasher,
                &mut hash_scratch,
            )?;
            ensure!(
                tensor_sha256 == inventory_record.source_tensor_sha256,
                "source tensor {name} bytes differ from D1"
            );
            tensors.push(SourcePrecisionTensorRecord {
                name: name.clone(),
                shape: info.shape.clone(),
                dtype,
                byte_len,
                byte_sha256: tensor_sha256,
                shard_filename: shard_name.clone(),
                payload_offset,
                disposition: dispositions[&name],
            });
        }
        ensure!(
            expected_relative_offset == parsed.data_len(),
            "source shard {shard_name} payload coverage is incomplete"
        );
        let shard_sha256 = hex::encode(shard_hasher.finalize());
        ensure!(
            record
                .sha256
                .as_deref()
                .is_some_and(|expected| expected.eq_ignore_ascii_case(&shard_sha256)),
            "retained source shard {shard_name} differs from its manifest"
        );
        let after = file.metadata()?;
        ensure!(
            after.file_type().is_file()
                && after.len() == record.bytes
                && (after.dev(), after.ino()) == identity,
            "source shard {shard_name} changed identity during verification"
        );
        shards.push(RetainedSourceShard {
            source: RetainedSourceFile {
                filename: shard_name.clone(),
                file,
                byte_len: record.bytes,
                device: identity.0,
                inode: identity.1,
                sha256: shard_sha256,
            },
        });
    }
    ensure!(
        seen_names.len() == inventory_records.len()
            && inventory_records
                .keys()
                .all(|name| seen_names.contains(*name)),
        "retained source snapshot does not have exact D1 tensor coverage"
    );
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let tensor_indices = tensors
        .iter()
        .enumerate()
        .map(|(index, tensor)| (tensor.name.clone(), index))
        .collect();
    let catalog_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(&CatalogHashView {
        schema_version: SOURCE_SNAPSHOT_SCHEMA_VERSION,
        source,
        verified_source_manifest_sha256: &manifest_sha256,
        source_inventory_manifest_sha256: &inventory.manifest().manifest_sha256,
        tensor_partition_manifest_sha256: &partition.manifest_sha256,
        config_sha256: &config_file.sha256,
        tensors: &tensors,
    })?));
    Ok(VerifiedQwenSourceSnapshot {
        source: source.clone(),
        verified_source_manifest_sha256: manifest_sha256,
        source_inventory_manifest_sha256: inventory.manifest().manifest_sha256.clone(),
        tensor_partition_manifest_sha256: partition.manifest_sha256.clone(),
        config,
        config_file,
        shards,
        tensors,
        tensor_indices,
        catalog_sha256,
    })
}
