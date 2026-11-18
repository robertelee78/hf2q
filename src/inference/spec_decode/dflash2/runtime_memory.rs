//! Requested drafter payloads composed by allocation lifetime, not file size.
//!
//! This is not RSS, a physical-residency bound, or an admission verdict. Target
//! weights/head/KV/prefill scratch and caller-owned sampling inputs are excluded.

use super::{
    config::DFlash2Config,
    runtime::DFlash2Runtime,
    tensors::{DFlash2ModelTensors, DFlash2TensorsError, DFlash2WeightMemory, ProjectionFormat},
    workspace::{DFlash2Workspace, DFlash2WorkspaceError, DFlash2WorkspaceMemory},
};
use crate::inference::spec_decode::dflash::hidden_capture::DFlashCaptureSession;

#[derive(Debug, thiserror::Error)]
pub enum DFlash2RuntimeMemoryError {
    #[error("DFlash2 runtime memory: {0}")]
    Invalid(&'static str),
    #[error(transparent)]
    Weights(#[from] DFlash2TensorsError),
    #[error(transparent)]
    Workspace(#[from] DFlash2WorkspaceError),
}

type Result<T, E = DFlash2RuntimeMemoryError> = std::result::Result<T, E>;

pub(super) fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(DFlash2RuntimeMemoryError::Invalid("payload sum overflows"))
}

pub(super) fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(DFlash2RuntimeMemoryError::Invalid("payload product overflows"))
}

fn allocation(bytes: usize) -> Result<usize> {
    if bytes > isize::MAX as usize {
        return Err(DFlash2RuntimeMemoryError::Invalid("allocation exceeds isize"));
    }
    Ok(bytes)
}

/// Known runtime-owned Rust payloads. Preallocated Vec backing uses constructor
/// capacity requests; parsed config strings/Vecs use logical lengths only.
/// Parser spare capacity and allocator rounding are explicitly unaccounted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash2RuntimeMetadataMemory {
    pub runtime_inline_bytes: usize,
    pub config_heap_bytes: usize,
    pub lane_state_bytes: usize,
    pub lane_rewind_bytes: usize,
    pub verify_budget_label_bytes: usize,
    pub weight_layer_array_bytes: usize,
    pub workspace_ring_array_bytes: usize,
    pub retained_bytes: usize,
    pub prefetch_snapshot_bytes: usize,
    pub prefetch_host_lane_order_temporary_bytes: usize,
    /// Replacing one lane binding constructs its new rewind Vec before dropping
    /// the old binding. `align_lane_prefix` need not settle a prefetch first.
    pub lane_rebind_temporary_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Production Qwen DFlash2 capture sessions request `with_argmaxes = false`
/// and write directly into one shared layer-major slab (no download/clone Vec).
pub struct DFlash2CaptureIngestMemory {
    pub max_capture_rows: usize,
    pub max_ingest_rows: usize,
    pub padded_ingest_rows: usize,
    pub source_hidden_bytes: usize,
    pub source_layer_ids_bytes: usize,
    /// The shared source buffer owns a rank-three shape Vec.
    pub source_shape_bytes: usize,
    /// Absent in all three production DFlash2 capture constructors.
    pub source_argmax_bytes: usize,
    pub source_session_inline_bytes: usize,
    pub source_payload_bound_bytes: usize,
    /// Only the immutable zero-padding scalar is newly allocated. The bounded
    /// transpose tile is workspace.captured_target_hidden, charged as retained.
    pub padding_scalar_device_bytes: usize,
    pub position_device_bytes: usize,
    /// Each of the write-position and commit-count buffers has this payload.
    pub group_word_device_bytes: usize,
    pub ingest_device_bytes: usize,
    pub word_upload_host_temporary_peak_bytes: usize,
    /// Joint ingest holds up to two segment ranges and a two-range forward map.
    /// This is the maximum production range-Vec backing, not target output data.
    pub ingest_range_host_bound_bytes: usize,
    /// Overlapping zero scalar + GPU word buffers + current host word Vec.
    /// NOT `ingest_device_bytes + word_upload_host_temporary_peak_bytes`.
    pub ingest_temporary_peak_bytes: usize,
}

impl DFlash2CaptureIngestMemory {
    fn required(
        config: &DFlash2Config,
        lanes: usize,
        max_capture_rows: usize,
        max_ingest_rows: usize,
    ) -> Result<Self> {
        if max_ingest_rows > max_capture_rows
            || (max_capture_rows > 0 && max_ingest_rows == 0)
        {
            return Err(DFlash2RuntimeMemoryError::Invalid("invalid capture/ingest row bounds"));
        }
        let feature_width = mul(config.hidden_size, config.target_layer_ids.len())?;
        let pass_rows = mul(lanes, config.block_size)?;
        let passes = max_ingest_rows.div_ceil(pass_rows);
        let padded_ingest_rows = mul(passes, pass_rows)?;
        u32::try_from(padded_ingest_rows).map_err(|_| {
            DFlash2RuntimeMemoryError::Invalid("padded capture row count exceeds u32")
        })?;
        let source_hidden_bytes = allocation(mul(mul(max_capture_rows, feature_width)?, 4)?)?;
        let source_layer_ids_bytes = allocation(mul(
            config.target_layer_ids.len(), std::mem::size_of::<usize>(),
        )?)?;
        let source_shape_bytes = allocation(mul(3, std::mem::size_of::<usize>())?)?;
        let source_argmax_bytes = 0;
        let source_session_inline_bytes = std::mem::size_of::<DFlashCaptureSession>();
        let source_payload_bound_bytes = add(
            add(add(source_hidden_bytes, source_layer_ids_bytes)?, source_shape_bytes)?,
            source_session_inline_bytes,
        )?;
        let padding_scalar_device_bytes = if max_ingest_rows == 0 { 0 } else { 4 };
        let position_device_bytes = allocation(mul(padded_ingest_rows, 4)?)?;
        let group_word_device_bytes = allocation(mul(mul(passes, lanes)?, 4)?)?;
        let ingest_device_bytes = add(
            padding_scalar_device_bytes,
            add(position_device_bytes, mul(2, group_word_device_bytes)?)?,
        )?;
        let ingest_range_host_bound_bytes = if max_ingest_rows == 0 {
            0
        } else {
            mul(4, std::mem::size_of::<std::ops::Range<usize>>())?
        };
        // The zero scalar is allocated only after all word uploads complete.
        let ingest_temporary_peak_bytes = add(
            ingest_range_host_bound_bytes,
            ingest_word_upload_peak(position_device_bytes, group_word_device_bytes)?
                .max(ingest_device_bytes),
        )?;
        Ok(Self {
            max_capture_rows,
            max_ingest_rows,
            padded_ingest_rows,
            source_hidden_bytes,
            source_layer_ids_bytes,
            source_shape_bytes,
            source_argmax_bytes,
            source_session_inline_bytes,
            source_payload_bound_bytes,
            padding_scalar_device_bytes,
            position_device_bytes,
            group_word_device_bytes,
            ingest_device_bytes,
            ingest_range_host_bound_bytes,
            word_upload_host_temporary_peak_bytes: position_device_bytes.max(group_word_device_bytes),
            ingest_temporary_peak_bytes,
        })
    }
}

/// Owners for which this source-only payload API does not claim a byte bound.
/// Presence in this list is a limitation, never a zero-byte estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlash2UnaccountedMemoryOwner {
    /// Load-only safetensors deserialization, manifest names/shapes and maps.
    CheckpointParsingMetadata,
    /// Retained parsed config String/Vec spare capacity beyond logical lengths.
    ParsedConfigSpareCapacity,
    /// Process-lifetime cached canonical planning configs are outside a runtime owner.
    PlanningConfigCache,
    /// Nested native shape/view metadata; layer and ring Vec backing IS counted.
    BufferAndTensorMetadata,
    /// Command buffers/encoders, captured resource references and kernel caches.
    NativeCommandAndKernelRegistry,
    AllocatorMetalAndDriverOverhead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash2RuntimeMemory {
    pub weights: DFlash2WeightMemory,
    pub workspace: DFlash2WorkspaceMemory,
    pub metadata: DFlash2RuntimeMetadataMemory,
    pub capture_ingest: DFlash2CaptureIngestMemory,
    /// Includes the lazy detached position pair exactly once; no prepared head.
    pub retained_device_payload_bytes: usize,
    pub retained_payload_bytes: usize,
    /// Serial weight upload, before workspace or runtime lane/budget allocation.
    pub weight_upload_peak_bytes: usize,
    /// Known load payload only, excluding the source config file Vec and mmap.
    pub load_payload_peak_bytes: usize,
    pub capture_run_peak_bytes: usize,
    pub ingest_run_peak_bytes: usize,
    pub metadata_run_peak_bytes: usize,
    pub run_payload_peak_bytes: usize,
    pub overall_payload_peak_bytes: usize,
}

impl DFlash2RuntimeMemory {
    pub const UNACCOUNTED_OWNERS: &'static [DFlash2UnaccountedMemoryOwner] = &[
        DFlash2UnaccountedMemoryOwner::CheckpointParsingMetadata,
        DFlash2UnaccountedMemoryOwner::ParsedConfigSpareCapacity,
        DFlash2UnaccountedMemoryOwner::PlanningConfigCache,
        DFlash2UnaccountedMemoryOwner::BufferAndTensorMetadata,
        DFlash2UnaccountedMemoryOwner::NativeCommandAndKernelRegistry,
        DFlash2UnaccountedMemoryOwner::AllocatorMetalAndDriverOverhead,
    ];

    /// Explicit caller-proved source-session and per-invocation append bounds.
    /// Ingest ranges cannot repeat beyond `max_ingest_rows`. Neither bound is
    /// inferred from the ring window or KV limit; folded captures and their
    /// serial content/cue ingests can have different maxima.
    /// Covers the production Qwen path: one live CPU session without argmaxes.
    /// Additional caller-owned replay/clone/argmax sessions need their own bound.
    pub fn required(
        config: &DFlash2Config,
        lanes: usize,
        format: ProjectionFormat,
        staged: bool,
        has_mask: bool,
        max_capture_rows: usize,
        max_ingest_rows: usize,
    ) -> Result<Self> {
        if staged && format != ProjectionFormat::Af4G64 {
            return Err(DFlash2RuntimeMemoryError::Invalid("staged requires AF4G64 weights"));
        }
        let workspace = DFlash2Workspace::required_memory(config, lanes, staged)?;
        let weights = DFlash2ModelTensors::weight_memory(config, format, has_mask)?;
        let metadata = DFlash2Runtime::metadata_memory(config, lanes)?;
        let capture_ingest = DFlash2CaptureIngestMemory::required(
            config, lanes, max_capture_rows, max_ingest_rows,
        )?;
        let initial_device = add(weights.retained_bytes, workspace.allocated_bytes)?;
        let retained_device_payload_bytes = add(
            initial_device, workspace.detached_context_positions_bytes,
        )?;
        let retained_payload_bytes = add(retained_device_payload_bytes, metadata.retained_bytes)?;
        let upload_config = add(
            metadata.config_heap_bytes, std::mem::size_of::<DFlash2Config>(),
        )?;
        // The layer object Vec is allocated only after global uploads finish.
        // Never attach it to a global-phase high-water that preceded it.
        let weight_upload_peak_bytes = add(upload_config,
            weights.global_upload_peak_bytes.max(add(
                weights.layer_upload_peak_bytes, metadata.weight_layer_array_bytes,
            )?),
        )?;
        let load_payload_peak_bytes = weight_upload_peak_bytes.max(add(
            initial_device, metadata.retained_bytes,
        )?);
        let captured = add(retained_payload_bytes, capture_ingest.source_payload_bound_bytes)?;
        let capture_run_peak_bytes = add(captured, metadata.prefetch_snapshot_bytes)?;
        // ingest_target_capture_runs settles and drops the prefetch first.
        let ingest_run_peak_bytes = add(captured, capture_ingest.ingest_temporary_peak_bytes)?;
        // Production capture closures only forward, ingest and record tokens;
        // the session is dropped before later prefetch/lane-rebinding work.
        let metadata_run_peak_bytes = add(
            add(retained_payload_bytes, metadata.prefetch_snapshot_bytes)?,
            metadata.prefetch_host_lane_order_temporary_bytes.max(metadata.lane_rebind_temporary_bytes),
        )?;
        let run_payload_peak_bytes = capture_run_peak_bytes
            .max(ingest_run_peak_bytes).max(metadata_run_peak_bytes);
        let overall_payload_peak_bytes = load_payload_peak_bytes.max(run_payload_peak_bytes);
        Ok(Self {
            weights, workspace, metadata, capture_ingest,
            retained_device_payload_bytes, retained_payload_bytes,
            weight_upload_peak_bytes, load_payload_peak_bytes,
            capture_run_peak_bytes, ingest_run_peak_bytes, metadata_run_peak_bytes,
            run_payload_peak_bytes, overall_payload_peak_bytes,
        })
    }

    /// Attach VERIFIED file lengths to LOAD only. The read-only checkpoint mmap
    /// is pageable address space, not a second heap/GPU copy or pinned residency.
    /// `checkpoint_mmap_extent_bytes` is the descriptor's verified weights_bytes.
    pub fn with_load_file_extents(
        &self,
        config_file_bytes: usize,
        checkpoint_mmap_extent_bytes: usize,
    ) -> Result<DFlash2RuntimeLoadMemory> {
        if config_file_bytes == 0 || checkpoint_mmap_extent_bytes == 0 {
            return Err(DFlash2RuntimeMemoryError::Invalid("verified load files must be nonempty"));
        }
        let config_file_bytes = allocation(config_file_bytes)?;
        let checkpoint_mmap_extent_bytes = allocation(checkpoint_mmap_extent_bytes)?;
        let load_payload_peak_bytes = add(self.load_payload_peak_bytes, config_file_bytes)?;
        Ok(DFlash2RuntimeLoadMemory {
            config_file_bytes,
            checkpoint_mmap_extent_bytes,
            load_payload_peak_bytes,
            load_payload_plus_mapping_extent_bytes: add(
                load_payload_peak_bytes, checkpoint_mmap_extent_bytes,
            )?,
            overall_payload_peak_bytes: load_payload_peak_bytes.max(self.run_payload_peak_bytes),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash2RuntimeLoadMemory {
    pub config_file_bytes: usize,
    pub checkpoint_mmap_extent_bytes: usize,
    pub load_payload_peak_bytes: usize,
    /// Arithmetic payload + virtual mapping extent, NOT physical residency/RSS.
    pub load_payload_plus_mapping_extent_bytes: usize,
    pub overall_payload_peak_bytes: usize,
}

fn ingest_word_upload_peak(position: usize, group: usize) -> Result<usize> {
    // Each collect::<Vec<u32>>() dies after its words(...) statement. The
    // later write/count GPU buffers do not exist during position upload.
    Ok(mul(2, position)?.max(add(position, mul(3, group)?)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_upload_peak_respects_noncoexisting_buffers() {
        // Eight rows/group: position host+device wins before group buffers exist.
        assert_eq!(ingest_word_upload_peak(1_024, 128).unwrap(), 2_048);
        // Two rows/group: count host+device wins with the earlier GPU buffers.
        assert_eq!(ingest_word_upload_peak(256, 128).unwrap(), 640);
        assert_eq!(ingest_word_upload_peak(0, 0).unwrap(), 0);
    }

    #[test]
    fn folded_capture_and_content_ingest_have_distinct_bounds() {
        let config = super::super::schema::QWEN38_DFLASH2.allocation_config().unwrap();
        let memory = DFlash2RuntimeMemory::required(
            config, 4, ProjectionFormat::Af4G64, true, false, 2_080, 2_048,
        ).unwrap();
        let ingest = memory.capture_ingest;
        assert_eq!(ingest.padded_ingest_rows, 2_048);
        assert_eq!(ingest.source_hidden_bytes, 212_992_000);
        assert_eq!(ingest.source_layer_ids_bytes, 5 * std::mem::size_of::<usize>());
        assert_eq!(ingest.position_device_bytes, 8_192);
        assert_eq!(ingest.group_word_device_bytes, 1_024);
        assert_eq!(
            ingest.ingest_temporary_peak_bytes,
            16_384 + 4 * std::mem::size_of::<std::ops::Range<usize>>(),
        );
        assert_eq!(memory.metadata.prefetch_snapshot_bytes, 192);
        assert_eq!(memory.workspace.detached_context_positions_bytes, 144);
        assert_eq!(
            memory.ingest_run_peak_bytes,
            memory.retained_payload_bytes + ingest.source_payload_bound_bytes
                + ingest.ingest_temporary_peak_bytes,
        );
        assert_eq!(
            memory.capture_run_peak_bytes,
            memory.retained_payload_bytes + ingest.source_payload_bound_bytes + 192,
        );
    }

    #[test]
    fn capture_limits_reject_invalid_bounds_and_pad_by_the_actual_lane_tile() {
        let config = super::super::schema::QWEN38_DFLASH2.allocation_config().unwrap();
        let memory = DFlash2CaptureIngestMemory::required(config, 3, 2_080, 2_047).unwrap();
        assert_eq!(memory.padded_ingest_rows, 2_064);
        assert!(DFlash2CaptureIngestMemory::required(config, 4, 2_080, 2_081).is_err());
        assert!(DFlash2CaptureIngestMemory::required(config, 4, 2_080, 0).is_err());
        assert!(DFlash2CaptureIngestMemory::required(
            config, 4, usize::MAX, usize::MAX,
        ).is_err());
        assert!(DFlash2RuntimeMemory::required(
            config, 4, ProjectionFormat::Bf16, true, false, 2_080, 2_048,
        ).is_err());
    }

    #[test]
    fn payload_arithmetic_fails_closed() {
        assert!(add(usize::MAX, 1).is_err());
        assert!(mul(usize::MAX, 2).is_err());
        assert!(ingest_word_upload_peak(usize::MAX, 1).is_err());
        assert!(allocation(isize::MAX as usize + 1).is_err());
    }
}
