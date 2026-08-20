use serde::Serialize;

pub(super) const SOURCE_SNAPSHOT_SCHEMA_VERSION: u32 = 1;
pub(super) const MAX_SOURCE_SHARDS: usize = 4_096;
pub(super) const MAX_SOURCE_TENSORS: usize = 262_144;
pub(super) const MAX_SOURCE_HEADER_BYTES_PER_SHARD: u64 = 64 * 1024 * 1024;
pub(super) const MAX_SOURCE_TOTAL_HEADER_BYTES: u64 = 256 * 1024 * 1024;
pub(super) const MAX_SOURCE_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
pub(super) const MAX_SOURCE_SHARD_BYTES: u64 = 64 * 1024 * 1024 * 1024;
pub(super) const MAX_SOURCE_TOTAL_BYTES: u64 = 256 * 1024 * 1024 * 1024;
pub(super) const MAX_SOURCE_TENSOR_NAME_BYTES: usize = 1_024;
pub(super) const MAX_SOURCE_TENSOR_RANK: usize = 8;
pub(super) const SOURCE_READ_CHUNK_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QwenSourceSnapshotLimits {
    pub max_shards: usize,
    pub max_tensors: usize,
    pub max_header_bytes_per_shard: u64,
    pub max_total_header_bytes: u64,
    pub max_config_bytes: u64,
    pub max_total_source_bytes: u64,
}

impl QwenSourceSnapshotLimits {
    pub(crate) fn validate(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.max_shards > 0
                && self.max_shards <= MAX_SOURCE_SHARDS
                && self.max_tensors > 0
                && self.max_tensors <= MAX_SOURCE_TENSORS
                && self.max_header_bytes_per_shard > 0
                && self.max_header_bytes_per_shard <= MAX_SOURCE_HEADER_BYTES_PER_SHARD
                && self.max_total_header_bytes > 0
                && self.max_total_header_bytes <= MAX_SOURCE_TOTAL_HEADER_BYTES
                && self.max_config_bytes > 0
                && self.max_config_bytes <= MAX_SOURCE_CONFIG_BYTES
                && self.max_total_source_bytes > 0
                && self.max_total_source_bytes <= MAX_SOURCE_TOTAL_BYTES,
            "source snapshot limits exceed the hard v1 envelope"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum SourcePrecisionDType {
    Bf16,
    F16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum SourcePrecisionDisposition {
    Variable,
    Fixed,
    Protected,
    Excluded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct SourcePrecisionTensorRecord {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: SourcePrecisionDType,
    pub byte_len: u64,
    pub byte_sha256: String,
    pub shard_filename: String,
    pub payload_offset: u64,
    pub disposition: SourcePrecisionDisposition,
}
