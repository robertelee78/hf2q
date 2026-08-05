//! Exact-input provenance and atomic success receipts for remote conversion.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::convert::orchestrator::TensorChunkStats;
use crate::core::provenance::source_shard::{compute_source_bundle_sha256, SourceShard};
use crate::core::sha256::compute_file_sha256;
use crate::input::integrity::VerifiedSourceManifest;

#[derive(Debug, thiserror::Error)]
pub enum ReceiptError {
    #[error("receipt I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("receipt JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("verified source manifest has no canonical LFS bundle SHA-256")]
    SourceBundleUnavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceFileReceipt {
    pub path: String,
    pub size: u64,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_lfs_sha256: Option<String>,
}

/// Remote source identity passed into conversion only after verification.
#[derive(Debug, Clone)]
pub struct RemoteConversionSource {
    pub(crate) repo: String,
    pub(crate) revision: String,
    pub(crate) source_sha256: String,
    pub(crate) files: Vec<SourceFileReceipt>,
}

impl RemoteConversionSource {
    pub fn from_verified(
        repo: String,
        revision: String,
        local_dir: &Path,
        verified: &VerifiedSourceManifest,
    ) -> Result<Self, ReceiptError> {
        let source_shards: Vec<_> = verified
            .records()
            .iter()
            .map(SourceShard::from_integrity)
            .collect();
        let source_sha256 = compute_source_bundle_sha256(&source_shards)
            .ok_or(ReceiptError::SourceBundleUnavailable)?;
        let mut files = Vec::with_capacity(verified.records().len());
        for record in verified.records() {
            let sha256 = match &record.sha256 {
                Some(verified_lfs_sha) => verified_lfs_sha.to_ascii_lowercase(),
                None => compute_file_sha256(&local_dir.join(&record.filename))?,
            };
            files.push(SourceFileReceipt {
                path: record.filename.clone(),
                size: record.bytes,
                sha256,
                hf_lfs_sha256: record.sha256.as_ref().map(|sha| sha.to_ascii_lowercase()),
            });
        }
        files.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(Self {
            repo,
            revision,
            source_sha256,
            files,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PeakChunkBoundReceipt {
    pub strategy: String,
    pub scope: String,
    pub max_chunk_elements: usize,
    pub max_input_f32_bytes: usize,
    pub max_f16_roundtrip_f32_bytes: usize,
    pub max_quantized_payload_bytes: usize,
    pub max_working_vec_bytes: usize,
}

impl Default for PeakChunkBoundReceipt {
    fn default() -> Self {
        Self {
            strategy: "row_aligned_tensor_chunks".into(),
            scope: "all_streamed_tensors".into(),
            max_chunk_elements: 0,
            max_input_f32_bytes: 0,
            max_f16_roundtrip_f32_bytes: 0,
            max_quantized_payload_bytes: 0,
            max_working_vec_bytes: 0,
        }
    }
}

impl PeakChunkBoundReceipt {
    pub fn observe(&mut self, stats: TensorChunkStats) {
        self.max_chunk_elements = self.max_chunk_elements.max(stats.max_chunk_elements);
        self.max_input_f32_bytes = self.max_input_f32_bytes.max(stats.max_input_f32_bytes);
        self.max_f16_roundtrip_f32_bytes = self
            .max_f16_roundtrip_f32_bytes
            .max(stats.max_f16_roundtrip_f32_bytes);
        self.max_quantized_payload_bytes = self
            .max_quantized_payload_bytes
            .max(stats.max_quantized_payload_bytes);
        self.max_working_vec_bytes = self.max_working_vec_bytes.max(stats.max_working_vec_bytes);
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversionReceipt {
    pub schema_version: u32,
    pub source: SourceReceipt,
    pub converter: ConverterReceipt,
    pub quant_selector: String,
    pub output: OutputReceipt,
    pub excluded_dspark: ExcludedDsparkReceipt,
    pub peak_chunk_bound: PeakChunkBoundReceipt,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceReceipt {
    pub repo: String,
    pub revision: String,
    pub bundle_sha256: String,
    pub files: Vec<SourceFileReceipt>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConverterReceipt {
    pub package: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutputReceipt {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExcludedDsparkReceipt {
    pub tensor_count: usize,
    pub status: String,
}

pub fn receipt_path(output: &Path) -> PathBuf {
    let mut name = output.as_os_str().to_os_string();
    name.push(".receipt.json");
    PathBuf::from(name)
}

pub fn clear_stale_receipt(output: &Path) -> Result<(), ReceiptError> {
    match fs::remove_file(receipt_path(output)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

pub fn write_success_receipt(
    output: &Path,
    remote: &RemoteConversionSource,
    quant_selector: &str,
    excluded_dspark_count: usize,
    peak_chunk_bound: PeakChunkBoundReceipt,
) -> Result<PathBuf, ReceiptError> {
    let output_meta = fs::metadata(output)?;
    let receipt = ConversionReceipt {
        schema_version: 1,
        source: SourceReceipt {
            repo: remote.repo.clone(),
            revision: remote.revision.clone(),
            bundle_sha256: remote.source_sha256.clone(),
            files: remote.files.clone(),
        },
        converter: ConverterReceipt {
            package: env!("CARGO_PKG_NAME").to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            git_commit: build_git_commit(),
        },
        quant_selector: quant_selector.to_string(),
        output: OutputReceipt {
            path: output.display().to_string(),
            size: output_meta.len(),
            sha256: compute_file_sha256(output)?,
        },
        excluded_dspark: ExcludedDsparkReceipt {
            tensor_count: excluded_dspark_count,
            status: if excluded_dspark_count == 0 {
                "none_detected".into()
            } else {
                "excluded_from_base_gguf".into()
            },
        },
        peak_chunk_bound,
    };

    let path = receipt_path(output);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut tmp, &receipt)?;
    tmp.write_all(b"\n")?;
    tmp.as_file().sync_all()?;
    tmp.persist(&path).map_err(|error| error.error)?;
    Ok(path)
}

fn build_git_commit() -> Option<String> {
    [
        option_env!("GIT_COMMIT_SHA"),
        option_env!("VERGEN_GIT_SHA"),
        option_env!("GITHUB_SHA"),
    ]
    .into_iter()
    .flatten()
    .find(|sha| sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit()))
    .map(|sha| sha.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn remote() -> RemoteConversionSource {
        RemoteConversionSource {
            repo: "org/model".into(),
            revision: "a".repeat(40),
            source_sha256: "b".repeat(64),
            files: vec![SourceFileReceipt {
                path: "model.safetensors".into(),
                size: 7,
                sha256: "c".repeat(64),
                hf_lfs_sha256: Some("c".repeat(64)),
            }],
        }
    }

    #[test]
    fn verified_manifest_builds_sorted_full_source_receipt() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();
        let weight_sha = compute_file_sha256(&dir.path().join("model.safetensors")).unwrap();
        let verified = crate::input::integrity::verify_conversion_manifest(
            "org/model",
            &"a".repeat(40),
            dir.path(),
            vec![
                crate::core::integrity::ShardIntegrity {
                    filename: "model.safetensors".into(),
                    bytes: 7,
                    sha256: Some(weight_sha.clone()),
                    hf_etag: weight_sha.clone(),
                    is_lfs: true,
                },
                crate::core::integrity::ShardIntegrity {
                    filename: "config.json".into(),
                    bytes: 2,
                    sha256: None,
                    hf_etag: "git-etag".into(),
                    is_lfs: false,
                },
            ],
        )
        .unwrap();
        let source = RemoteConversionSource::from_verified(
            "org/model".into(),
            "a".repeat(40),
            dir.path(),
            &verified,
        )
        .unwrap();
        assert_eq!(source.files[0].path, "config.json");
        assert_eq!(
            source.files[0].sha256,
            compute_file_sha256(&dir.path().join("config.json")).unwrap()
        );
        assert_eq!(source.files[1].hf_lfs_sha256, Some(weight_sha));
        assert_eq!(source.source_sha256.len(), 64);
    }

    #[test]
    fn receipt_path_appends_suffix() {
        assert_eq!(
            receipt_path(Path::new("model.gguf")),
            PathBuf::from("model.gguf.receipt.json")
        );
    }

    #[test]
    fn success_receipt_binds_output_and_replaces_stale_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("model.gguf");
        fs::write(&output, b"GGUFfixture").unwrap();
        fs::write(receipt_path(&output), b"stale").unwrap();
        let path = write_success_receipt(
            &output,
            &remote(),
            "q4_k_m",
            3,
            PeakChunkBoundReceipt {
                strategy: "row_aligned_tensor_chunks".into(),
                scope: "all_streamed_tensors".into(),
                max_chunk_elements: 8,
                max_input_f32_bytes: 32,
                max_f16_roundtrip_f32_bytes: 32,
                max_quantized_payload_bytes: 16,
                max_working_vec_bytes: 80,
            },
        )
        .unwrap();
        let parsed: ConversionReceipt = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(parsed.quant_selector, "q4_k_m");
        assert_eq!(parsed.output.size, 11);
        assert_eq!(parsed.output.sha256, compute_file_sha256(&output).unwrap());
        assert_eq!(parsed.excluded_dspark.tensor_count, 3);
        assert_eq!(parsed.source.revision, "a".repeat(40));
        assert_eq!(parsed.peak_chunk_bound.max_working_vec_bytes, 80);
    }

    #[test]
    fn clearing_missing_or_stale_receipt_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("model.gguf");
        clear_stale_receipt(&output).unwrap();
        fs::write(receipt_path(&output), b"stale").unwrap();
        clear_stale_receipt(&output).unwrap();
        assert!(!receipt_path(&output).exists());
    }
}
