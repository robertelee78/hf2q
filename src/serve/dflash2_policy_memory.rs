//! Drafter payload/load extents composed with target scratch by whole lifetime.
//! Non-payload parser/native/allocator/driver owners remain outside this bound
//! and under the policy's unchanged process reserve and hardware qualification.

use anyhow::{bail, Context, Result};

use super::{switch, DFlash2CheckpointDescriptor, ResolvedRoutes, TargetFacts};
use crate::inference::models::qwen35::prefill_fold::MAX_CUE_ROWS;
use crate::inference::spec_decode::dflash2::{
    runtime_memory::DFlash2RuntimeMemory,
    tensors::ProjectionFormat,
    weights::required_tensor_count,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct DrafterMemoryBound {
    pub(super) load_bytes: u64,
    pub(super) run_bytes: u64,
}

impl DrafterMemoryBound {
    pub(super) fn required(
        descriptor: &DFlash2CheckpointDescriptor,
        facts: &TargetFacts,
        routes: &ResolvedRoutes,
    ) -> Result<Self> {
        let config = descriptor.allocation_config()?;
        let required_tensors = required_tensor_count(config)
            .context("DFlash2 manifest cardinality overflows")?;
        let has_mask = match descriptor.tensor_count.checked_sub(required_tensors) {
            Some(0) => false,
            Some(1) => true,
            _ => bail!("DFlash2 descriptor tensor count cannot bind allocation geometry"),
        };
        let lanes = facts.slots.unwrap_or(1).max(1) as usize;
        let content_rows = super::prefill_content_rows(facts)
            .context("DFlash2 capture capacity is unknown")?;
        let capture_rows = content_rows.checked_add(
            if routes.is_on(switch("HF2Q_QWEN_PREFILL_FOLD_CUE")) {
                MAX_CUE_ROWS
            } else {
                0
            },
        ).context("DFlash2 folded capture row bound overflows")?;
        let format = if routes.is_on(switch("HF2Q_DFLASH2_DRAFT_AF4_G64")) {
            ProjectionFormat::Af4G64
        } else {
            ProjectionFormat::Bf16
        };
        let payload = DFlash2RuntimeMemory::required(
            config, lanes, format,
            routes.is_on(switch("HF2Q_DFLASH2_DRAFT_STAGED")),
            has_mask, capture_rows, content_rows,
        )?;
        let load = payload.with_load_file_extents(
            descriptor.allocation_config_bytes()?.len(),
            usize::try_from(descriptor.weights_bytes)
                .context("DFlash2 checkpoint mapping extent exceeds usize")?,
        )?;
        Ok(Self {
            load_bytes: u64::try_from(load.load_payload_plus_mapping_extent_bytes)
                .context("DFlash2 load extent exceeds u64")?,
            run_bytes: u64::try_from(payload.run_payload_peak_bytes)
                .context("DFlash2 run payload exceeds u64")?,
        })
    }

    /// Increment beside the target's existing scratch charge. Load-only mmap
    /// and upload peaks never overlap the target's run-prefill scratch peak.
    pub(super) fn charge(self, scratch: u64) -> u64 {
        self.run_bytes.max(self.load_bytes.saturating_sub(scratch))
    }

    pub(super) fn total(self, planes: u64, scratch: u64) -> u64 {
        planes.saturating_add(scratch).saturating_add(self.charge(scratch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{memory_verdict, HostFacts, MemoryVerdict, MEMORY_RESERVE_BYTES};

    fn verdict(bound: DrafterMemoryBound, headroom: u64) -> MemoryVerdict {
        let host = HostFacts {
            physical_bytes: MEMORY_RESERVE_BYTES + headroom,
            available_bytes: MEMORY_RESERVE_BYTES + headroom,
            kv_budget_bytes: 0,
            resident_bytes: 0,
        };
        memory_verdict(host, 0, bound.total(10, 80), bound.total(0, 0)).0
    }

    #[test]
    fn load_mapping_does_not_overlap_prefill_but_still_prevents_underadmission() {
        let bound = DrafterMemoryBound { load_bytes: 160, run_bytes: 40 };
        // Load needs 160+10; run needs 40+80+10, not their sum.
        assert_eq!(verdict(bound, 170), MemoryVerdict::Full);
        assert_eq!(verdict(bound, 169), MemoryVerdict::DropPlanes);
        assert_eq!(verdict(bound, 159), MemoryVerdict::NoDrafter);
    }

    #[test]
    fn run_scratch_and_drafter_are_simultaneously_charged() {
        let bound = DrafterMemoryBound { load_bytes: 160, run_bytes: 300 };
        assert_eq!(verdict(bound, 390), MemoryVerdict::Full);
        assert_eq!(verdict(bound, 389), MemoryVerdict::DropPlanes);
        assert_eq!(verdict(bound, 299), MemoryVerdict::NoDrafter);
    }
}
