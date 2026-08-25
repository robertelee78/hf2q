//! Owned-byte source reader for conversion evidence.
//!
//! Ordinary conversion uses mmap for throughput. Provenance production uses
//! this narrower reader: it authenticates the exact source manifest, parses
//! owned shard bytes, then reopens and owns each tensor payload before hashing
//! and decoding it. No evidence statement depends on a mutable mmap view.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use safetensors::{tensor::Dtype, SafeTensors};
use sha2::{Digest, Sha256};

use crate::core::mlx_safetensors_loader::read_floats_to_f32;
use crate::input::integrity::{verify_conversion_manifest, VerifiedSourceManifest};
use crate::quantize::ggml_quants::SourceDtype;

use super::source_reader::{HfTensor, RawSourceTensorEvidence, SourceError, TensorMeta};

#[derive(Debug, Clone)]
struct EvidenceShard {
    path: PathBuf,
    artifact_id: String,
    header_byte_len: usize,
    byte_len: u64,
}

/// Exact dense-source snapshot used only by the D2b evidence conversion.
#[derive(Debug)]
pub(crate) struct VerifiedEvidenceSource {
    pub(crate) config: serde_json::Value,
    shards: Vec<EvidenceShard>,
    metas: Vec<TensorMeta>,
    by_name: HashMap<String, usize>,
    raw_tensor_sha256: HashMap<String, String>,
}

fn source_error(message: impl Into<String>) -> SourceError {
    SourceError::Safetensors(message.into())
}

fn owned_file_bytes(path: &Path) -> Result<Vec<u8>, SourceError> {
    std::fs::read(path).map_err(SourceError::Io)
}

impl VerifiedEvidenceSource {
    pub(crate) fn open(
        model_dir: &Path,
        verified_source: &VerifiedSourceManifest,
        expected_repo: &str,
        expected_revision: &str,
        expected_config_sha256: &str,
    ) -> Result<Self, SourceError> {
        if verified_source.repo() != expected_repo
            || verified_source.revision() != expected_revision
        {
            return Err(source_error(
                "verified source repository/revision differs from evidence request",
            ));
        }
        let reverified = verify_conversion_manifest(
            expected_repo,
            expected_revision,
            model_dir,
            verified_source.records().to_vec(),
        )
        .map_err(|error| source_error(error.to_string()))?;
        let config_bytes = owned_file_bytes(&model_dir.join("config.json"))?;
        if hex::encode(Sha256::digest(&config_bytes)) != expected_config_sha256 {
            return Err(source_error(
                "config.json does not match the exact source identity",
            ));
        }
        let config: serde_json::Value =
            serde_json::from_slice(&config_bytes).map_err(SourceError::ConfigParse)?;
        let records: BTreeMap<_, _> = reverified
            .records()
            .iter()
            .map(|record| (record.filename.as_str(), record))
            .collect();
        let mut shards = Vec::with_capacity(reverified.required_weight_shards().len());
        let mut metas = Vec::new();
        let mut by_name = HashMap::new();
        let mut raw_tensor_sha256 = HashMap::new();

        for shard_name in reverified.required_weight_shards() {
            let record = records.get(shard_name.as_str()).ok_or_else(|| {
                source_error(format!("verified source is missing shard {shard_name}"))
            })?;
            let path = model_dir.join(shard_name);
            let bytes = owned_file_bytes(&path)?;
            let byte_len = u64::try_from(bytes.len())
                .map_err(|_| source_error(format!("shard {shard_name} is too large")))?;
            let sha256 = hex::encode(Sha256::digest(&bytes));
            if byte_len != record.bytes || record.sha256.as_deref() != Some(sha256.as_str()) {
                return Err(source_error(format!(
                    "source shard {shard_name} changed after verification"
                )));
            }
            let (header_size, metadata) = SafeTensors::read_metadata(&bytes)
                .map_err(|error| source_error(format!("parse {shard_name}: {error}")))?;
            let header_byte_len = 8_usize
                .checked_add(header_size)
                .ok_or_else(|| source_error(format!("shard {shard_name} header overflow")))?;
            let shard_idx = shards.len();
            for name in metadata.offset_keys() {
                let info = metadata.info(&name).ok_or_else(|| {
                    source_error(format!("shard {shard_name} lost tensor {name}"))
                })?;
                let source_dtype = match info.dtype {
                    Dtype::F16 => SourceDtype::F16,
                    Dtype::BF16 => SourceDtype::BF16,
                    Dtype::F32 => SourceDtype::F32,
                    other => {
                        return Err(SourceError::UnsupportedSourceDtype {
                            tensor: name.to_string(),
                            dtype: format!("{other:?} is outside dense-Qwen source evidence scope"),
                        });
                    }
                };
                let index = metas.len();
                let raw_start = header_byte_len
                    .checked_add(info.data_offsets.0)
                    .ok_or_else(|| source_error(format!("tensor {name} offset overflow")))?;
                let raw_end = header_byte_len
                    .checked_add(info.data_offsets.1)
                    .ok_or_else(|| source_error(format!("tensor {name} end overflow")))?;
                let raw = bytes.get(raw_start..raw_end).ok_or_else(|| {
                    source_error(format!("tensor {name} payload is outside {shard_name}"))
                })?;
                metas.push(TensorMeta {
                    name: name.to_string(),
                    shape: info.shape.clone(),
                    source_dtype,
                    shard_idx,
                    data_off_start: info.data_offsets.0,
                    data_off_end: info.data_offsets.1,
                });
                if by_name.insert(name.to_string(), index).is_some() {
                    return Err(source_error(format!(
                        "tensor {name} is duplicated across source shards"
                    )));
                }
                raw_tensor_sha256.insert(name.to_string(), hex::encode(Sha256::digest(raw)));
            }
            shards.push(EvidenceShard {
                path,
                artifact_id: shard_name.clone(),
                header_byte_len,
                byte_len,
            });
            // The owned shard bytes are dropped here; peak source evidence
            // memory remains one shard during index construction.
        }
        Ok(Self {
            config,
            shards,
            metas,
            by_name,
            raw_tensor_sha256,
        })
    }

    pub(crate) fn tensor_metas(&self) -> impl Iterator<Item = &TensorMeta> {
        self.metas.iter()
    }

    pub(crate) fn materialize_tensor_with_evidence(
        &self,
        name: &str,
    ) -> Result<HfTensor, SourceError> {
        let index = self
            .by_name
            .get(name)
            .copied()
            .ok_or_else(|| source_error(format!("source tensor {name} is absent")))?;
        let meta = &self.metas[index];
        let shard = &self.shards[meta.shard_idx];
        let start = shard
            .header_byte_len
            .checked_add(meta.data_off_start)
            .ok_or_else(|| source_error(format!("source tensor {name} offset overflow")))?;
        let byte_len = meta
            .data_off_end
            .checked_sub(meta.data_off_start)
            .ok_or_else(|| source_error(format!("source tensor {name} range is invalid")))?;
        let start_u64 = u64::try_from(start)
            .map_err(|_| source_error(format!("source tensor {name} offset overflow")))?;
        let byte_len_u64 = u64::try_from(byte_len)
            .map_err(|_| source_error(format!("source tensor {name} length overflow")))?;
        let end = start_u64
            .checked_add(byte_len_u64)
            .ok_or_else(|| source_error(format!("source tensor {name} end overflow")))?;
        if end > shard.byte_len {
            return Err(source_error(format!(
                "source tensor {name} ends beyond shard {}",
                shard.artifact_id
            )));
        }
        let mut file = File::open(&shard.path).map_err(SourceError::Io)?;
        file.seek(SeekFrom::Start(start_u64))
            .map_err(SourceError::Io)?;
        let mut raw = vec![0_u8; byte_len];
        file.read_exact(&mut raw).map_err(SourceError::Io)?;
        let raw_sha256 = hex::encode(Sha256::digest(&raw));
        if self.raw_tensor_sha256.get(name) != Some(&raw_sha256) {
            return Err(source_error(format!(
                "source tensor {name} changed after evidence source verification"
            )));
        }
        let dtype = match meta.source_dtype {
            SourceDtype::F16 => Dtype::F16,
            SourceDtype::BF16 => Dtype::BF16,
            SourceDtype::F32 => Dtype::F32,
            _ => unreachable!("open admits only dense scalar source types"),
        };
        let data = read_floats_to_f32(&raw, dtype)
            .map_err(|error| source_error(format!("decode tensor {name}: {error:#}")))?;
        if data.len() != meta.numel() {
            return Err(source_error(format!(
                "source tensor {name} decoded {} values, expected {}",
                data.len(),
                meta.numel()
            )));
        }
        Ok(HfTensor {
            name: name.to_string(),
            shape: meta.shape.clone(),
            source_dtype: meta.source_dtype,
            data,
            raw_source: Some(RawSourceTensorEvidence {
                artifact_id: shard.artifact_id.clone(),
                absolute_byte_offset: start_u64,
                byte_len: byte_len_u64,
                sha256: raw_sha256,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use half::f16;
    use safetensors::tensor::TensorView;

    use crate::core::integrity::{compute_git_blob_sha1, ShardIntegrity};

    use super::*;

    fn fixture(dtype: Dtype, payload: Vec<u8>) -> (tempfile::TempDir, VerifiedSourceManifest) {
        let dir = tempfile::tempdir().unwrap();
        let config = br#"{"model_type":"qwen3_5"}"#;
        std::fs::write(dir.path().join("config.json"), config).unwrap();
        let view = TensorView::new(dtype, vec![2], &payload).unwrap();
        let shard =
            safetensors::tensor::serialize(vec![("model.norm.weight".to_owned(), &view)], None)
                .unwrap();
        std::fs::write(dir.path().join("model.safetensors"), &shard).unwrap();
        let config_etag = compute_git_blob_sha1(
            &dir.path().join("config.json"),
            u64::try_from(config.len()).unwrap(),
        )
        .unwrap();
        let shard_sha256 = hex::encode(Sha256::digest(&shard));
        let verified = VerifiedSourceManifest::for_test_bound(
            "org/qwen",
            "a".repeat(40),
            vec![
                ShardIntegrity {
                    filename: "config.json".into(),
                    bytes: u64::try_from(config.len()).unwrap(),
                    sha256: None,
                    hf_etag: config_etag,
                    is_lfs: false,
                },
                ShardIntegrity {
                    filename: "model.safetensors".into(),
                    bytes: u64::try_from(shard.len()).unwrap(),
                    sha256: Some(shard_sha256.clone()),
                    hf_etag: shard_sha256,
                    is_lfs: true,
                },
            ],
        );
        (dir, verified)
    }

    fn config_sha256(dir: &Path) -> String {
        hex::encode(Sha256::digest(
            std::fs::read(dir.join("config.json")).unwrap(),
        ))
    }

    #[test]
    fn exact_owned_tensor_bytes_drive_hash_and_decode() {
        let values = [f16::from_f32(1.5), f16::from_f32(-2.0)];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let expected_raw_sha256 = hex::encode(Sha256::digest(&payload));
        let (dir, manifest) = fixture(Dtype::F16, payload);
        let source = VerifiedEvidenceSource::open(
            dir.path(),
            &manifest,
            "org/qwen",
            &"a".repeat(40),
            &config_sha256(dir.path()),
        )
        .unwrap();
        let tensor = source
            .materialize_tensor_with_evidence("model.norm.weight")
            .unwrap();
        assert_eq!(tensor.data, vec![1.5, -2.0]);
        assert_eq!(
            tensor.raw_source.as_ref().unwrap().sha256,
            expected_raw_sha256
        );
        assert_eq!(source.tensor_metas().count(), 1);
    }

    #[test]
    fn tensor_mutation_after_open_fails_closed() {
        let payload: Vec<u8> = [f16::from_f32(1.0), f16::from_f32(2.0)]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let (dir, manifest) = fixture(Dtype::F16, payload);
        let source = VerifiedEvidenceSource::open(
            dir.path(),
            &manifest,
            "org/qwen",
            &"a".repeat(40),
            &config_sha256(dir.path()),
        )
        .unwrap();
        let meta = source.tensor_metas().next().unwrap();
        let shard = &source.shards[meta.shard_idx];
        let offset = shard.header_byte_len + meta.data_off_start;
        let mut bytes = std::fs::read(&shard.path).unwrap();
        bytes[offset] ^= 0x01;
        std::fs::write(&shard.path, bytes).unwrap();
        let error = source
            .materialize_tensor_with_evidence("model.norm.weight")
            .unwrap_err();
        assert!(error.to_string().contains("changed after evidence"));
    }

    #[test]
    fn unsupported_source_dtype_is_rejected_before_conversion() {
        let payload = vec![0_u8; 16];
        let (dir, manifest) = fixture(Dtype::I64, payload);
        let error = VerifiedEvidenceSource::open(
            dir.path(),
            &manifest,
            "org/qwen",
            &"a".repeat(40),
            &config_sha256(dir.path()),
        )
        .unwrap_err();
        assert!(error.to_string().contains("outside dense-Qwen"));
    }
}
