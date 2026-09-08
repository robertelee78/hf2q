//! Exact mapped GGUF storage and ownership checks.

use super::forward_mlx_shared::MlxQWeight;
use super::gpu::QuantWeightInfo;
use anyhow::Result;
use mlx_native::{DType, GgmlType, MlxBuffer};
use std::collections::BTreeSet;

impl MlxQWeight {
    /// Retain one exact, file-backed matrix view from a mapped GGUF.
    ///
    /// Family loaders remain responsible for capability admission before
    /// calling this constructor. This shared ownership primitive independently
    /// checks rank, dimensions, byte length, and file backing so every family
    /// can use the same unload/reload-safe representation.
    pub fn from_mapped_gguf_tensor(
        mapped: &mlx_native::gguf::GgufMappedTensorSet<'_>,
        info: &mlx_native::gguf::TensorInfo,
    ) -> Result<Self> {
        let [rows, cols] = info.shape.as_slice() else {
            anyhow::bail!(
                "native GGUF matrix '{}' must be rank 2, got {:?}",
                info.name,
                info.shape
            );
        };
        if *rows == 0 || *cols == 0 {
            anyhow::bail!("native GGUF matrix '{}' has a zero dimension", info.name);
        }
        Self::from_mapped_gguf_matrix_view(mapped, info, *rows, *cols)
    }

    /// Retain one exact file-backed GGUF tensor as a logical `[rows, cols]`
    /// matrix without changing its stored representation.
    pub(crate) fn from_mapped_gguf_matrix_view(
        mapped: &mlx_native::gguf::GgufMappedTensorSet<'_>,
        info: &mlx_native::gguf::TensorInfo,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            rows > 0 && cols > 0,
            "native GGUF matrix '{}' must have non-zero [rows, cols], got [{rows}, {cols}]",
            info.name
        );
        let source_elements = info.shape.iter().try_fold(1usize, |product, dimension| {
            product.checked_mul(*dimension).ok_or_else(|| {
                anyhow::anyhow!("native GGUF matrix '{}' shape product overflow", info.name)
            })
        })?;
        let matrix_elements = rows.checked_mul(cols).ok_or_else(|| {
            anyhow::anyhow!("native GGUF matrix '{}' [rows, cols] overflow", info.name)
        })?;
        anyhow::ensure!(
            source_elements == matrix_elements,
            "native GGUF matrix '{}' source shape {:?} has {source_elements} elements, not declared [{rows}, {cols}] ({matrix_elements})",
            info.name,
            info.shape
        );
        let expected_bytes = native_gguf_matrix_bytes(info.ggml_type, rows, cols)?;
        anyhow::ensure!(
            info.byte_len == expected_bytes,
            "native GGUF matrix '{}' metadata has {} bytes, expected exactly {expected_bytes} for {:?} [{rows}, {cols}]",
            info.name,
            info.byte_len,
            info.ggml_type
        );
        let buffer = map_native_gguf_tensor_view(mapped, info)?;
        anyhow::ensure!(
            buffer.dtype() == native_gguf_buffer_dtype(info.ggml_type)?,
            "native GGUF matrix '{}' did not retain its exact {:?} payload dtype",
            info.name,
            info.ggml_type
        );
        Ok(Self {
            buffer,
            info: QuantWeightInfo {
                ggml_dtype: info.ggml_type,
                rows,
                cols,
            },
            affine: None,
            f16_shadow: None,
            decode_record_q6k_m1: std::sync::OnceLock::new(),
        })
    }
}

/// Checked byte width of one row in an admitted native GGUF matrix.
pub(crate) fn native_gguf_matrix_row_bytes(ggml_dtype: GgmlType, cols: usize) -> Result<usize> {
    anyhow::ensure!(cols > 0, "native GGUF matrix columns must be non-zero");
    match ggml_dtype {
        GgmlType::F32 => cols
            .checked_mul(DType::F32.size_of())
            .ok_or_else(|| anyhow::anyhow!("F32 GGUF matrix row byte extent overflow")),
        GgmlType::F16 => cols
            .checked_mul(DType::F16.size_of())
            .ok_or_else(|| anyhow::anyhow!("F16 GGUF matrix row byte extent overflow")),
        GgmlType::BF16 => cols
            .checked_mul(DType::BF16.size_of())
            .ok_or_else(|| anyhow::anyhow!("BF16 GGUF matrix row byte extent overflow")),
        GgmlType::Q4_0
        | GgmlType::Q5_0
        | GgmlType::Q8_0
        | GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::Q5_1
        | GgmlType::IQ4_NL
        | GgmlType::IQ4_XS => {
            let cols = u32::try_from(cols)
                .map_err(|_| anyhow::anyhow!("native GGUF matrix columns exceed u32"))?;
            usize::try_from(mlx_native::ggml_packed_row_bytes(ggml_dtype, cols)?)
                .map_err(Into::into)
        }
        other => anyhow::bail!(
            "native GGUF matrix codec {other:?} is not admitted; no transform fallback is allowed"
        ),
    }
}

/// Checked exact payload bytes for an admitted native GGUF `[rows, cols]`
/// matrix.
pub(crate) fn native_gguf_matrix_bytes(
    ggml_dtype: GgmlType,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    anyhow::ensure!(rows > 0, "native GGUF matrix rows must be non-zero");
    native_gguf_matrix_row_bytes(ggml_dtype, cols)?
        .checked_mul(rows)
        .ok_or_else(|| anyhow::anyhow!("native GGUF matrix byte extent overflow"))
}

pub(crate) fn native_gguf_buffer_dtype(ggml_dtype: GgmlType) -> Result<DType> {
    match ggml_dtype {
        GgmlType::F32 => Ok(DType::F32),
        GgmlType::F16 => Ok(DType::F16),
        GgmlType::BF16 => Ok(DType::BF16),
        GgmlType::Q4_0
        | GgmlType::Q5_0
        | GgmlType::Q8_0
        | GgmlType::Q2_K
        | GgmlType::Q3_K
        | GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::Q5_1
        | GgmlType::IQ4_NL
        | GgmlType::IQ4_XS => Ok(DType::U8),
        other => anyhow::bail!(
            "native GGUF matrix codec {other:?} is not admitted; no transform fallback is allowed"
        ),
    }
}

/// Exact logical-byte accounting for model-owned matrix views. Views that
/// name the same Metal allocation range are counted once, so tied output
/// heads and MTP aliases cannot inflate residency reporting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeMatrixStorageSummary {
    pub unique_matrix_views: usize,
    pub file_backed_bytes: u64,
    pub anonymous_bytes: u64,
}

impl NativeMatrixStorageSummary {
    pub fn resident_bytes(self) -> u64 {
        self.file_backed_bytes + self.anonymous_bytes
    }
}

/// Summarize the physical backing of exact model matrix views. The identity
/// includes the underlying Metal resource and logical byte range: separate
/// tensors in one scoped GGUF mapping remain distinct, while ARC clones of a
/// tied tensor collapse to one entry.
pub fn summarize_native_matrix_storage<'a>(
    buffers: impl IntoIterator<Item = &'a MlxBuffer>,
) -> Result<NativeMatrixStorageSummary> {
    let mut seen = BTreeSet::new();
    let mut summary = NativeMatrixStorageSummary::default();
    for buffer in buffers {
        let identity = (
            buffer.contents_ptr() as usize,
            buffer.byte_offset(),
            buffer.data_byte_len(),
        );
        if !seen.insert(identity) {
            continue;
        }
        let bytes = u64::try_from(buffer.data_byte_len())?;
        summary.unique_matrix_views += 1;
        if buffer.is_file_backed() {
            summary.file_backed_bytes = summary
                .file_backed_bytes
                .checked_add(bytes)
                .ok_or_else(|| anyhow::anyhow!("file-backed matrix byte accounting overflow"))?;
        } else {
            summary.anonymous_bytes = summary
                .anonymous_bytes
                .checked_add(bytes)
                .ok_or_else(|| anyhow::anyhow!("anonymous matrix byte accounting overflow"))?;
        }
    }
    Ok(summary)
}

/// Retain one exact, file-backed tensor view from a mapped GGUF.
///
/// Rank-2 matrices normally use [`super::forward_mlx_shared::MlxQWeight::from_mapped_gguf_tensor`]. Expert
/// stacks use this raw form because their rank-3 dispatch metadata is carried
/// by the family graph.
pub fn map_native_gguf_tensor_view(
    mapped: &mlx_native::gguf::GgufMappedTensorSet<'_>,
    info: &mlx_native::gguf::TensorInfo,
) -> Result<MlxBuffer> {
    let buffer = mapped
        .load_tensor(&info.name)
        .map_err(|error| anyhow::anyhow!("map native GGUF tensor '{}': {error}", info.name))?;
    if !buffer.is_file_backed() || buffer.data_byte_len() != info.byte_len {
        anyhow::bail!(
            "native GGUF tensor '{}' did not retain its file-backed {}-byte payload",
            info.name,
            info.byte_len
        );
    }
    Ok(buffer)
}

#[cfg(test)]
#[path = "native_matrix_storage_tests.rs"]
mod tests;
