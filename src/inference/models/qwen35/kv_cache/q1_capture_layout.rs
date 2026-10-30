//! Test-only Q1 metadata readback of the SAME plan consumed by the allocator.
//! No Metal allocation and no independent cache arithmetic implementation.
use super::*;
pub(crate) struct RoleLayout {
    pub layer: u32,
    pub mtp: bool,
    pub slot: LayerSlot,
    pub roots: Vec<(DType, Vec<usize>)>,
}
pub(crate) fn layouts(cfg: &Qwen35Config, capacity: u32) -> Result<Vec<RoleLayout>> {
    let plan = HybridCacheAllocationPlan::new(cfg, capacity, 1, true, true)?;
    plan.slots
        .into_iter()
        .enumerate()
        .map(|(layer, slot)| {
            Ok(RoleLayout {
                layer: u32::try_from(layer)?,
                mtp: slot.mtp,
                slot: slot.slot,
                roots: slot
                    .roots
                    .into_iter()
                    .map(|root| (root.dtype, root.shape))
                    .collect(),
            })
        })
        .collect()
}
