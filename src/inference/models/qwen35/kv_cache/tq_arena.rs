use anyhow::{anyhow, ensure, Result};

/// Physical segment assigned to one logical sequence inside a shared TQ KV
/// arena. `base_tokens` is measured in per-head token rows; packed and norm
/// element offsets are derived from it with the model dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TqArenaSegment {
    pub(super) base_tokens: u32,
    pub(super) capacity_tokens: u32,
}

/// Family-local policy for a bounded, variable-capacity TQ KV arena.
///
/// Logical context remains a property of the owning `HybridKvCache`. This
/// layout describes only the rows physically allocated today. Segments are
/// contiguous and non-overlapping, so one Metal buffer per K/V component can
/// still serve a physically batched cohort with per-row base/capacity arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TqArenaLayout {
    segments: Vec<TqArenaSegment>,
    total_capacity_tokens: u32,
}

impl TqArenaLayout {
    pub(super) fn uniform(n_seqs: u32, capacity_tokens: u32) -> Result<Self> {
        ensure!(n_seqs > 0, "TQ arena requires at least one sequence");
        ensure!(
            capacity_tokens > 0,
            "TQ arena requires non-zero physical capacity"
        );
        Self::from_capacities(vec![capacity_tokens; n_seqs as usize])
    }

    pub(super) fn from_capacities(capacities: Vec<u32>) -> Result<Self> {
        ensure!(!capacities.is_empty(), "TQ arena has no sequences");
        let mut base_tokens = 0_u32;
        let mut segments = Vec::with_capacity(capacities.len());
        for (slot, capacity_tokens) in capacities.into_iter().enumerate() {
            ensure!(
                capacity_tokens > 0,
                "TQ arena slot {slot} has zero physical capacity"
            );
            segments.push(TqArenaSegment {
                base_tokens,
                capacity_tokens,
            });
            base_tokens = base_tokens.checked_add(capacity_tokens).ok_or_else(|| {
                anyhow!("TQ arena aggregate token capacity exceeds u32 addressability")
            })?;
        }
        Ok(Self {
            segments,
            total_capacity_tokens: base_tokens,
        })
    }

    pub(super) fn n_seqs(&self) -> usize {
        self.segments.len()
    }

    pub(super) fn total_capacity_tokens(&self) -> u32 {
        self.total_capacity_tokens
    }

    pub(super) fn segment(&self, slot: usize) -> Result<TqArenaSegment> {
        self.segments.get(slot).copied().ok_or_else(|| {
            anyhow!(
                "TQ arena slot {slot} out of range for {} sequences",
                self.segments.len()
            )
        })
    }

    pub(super) fn capacities(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        self.segments.iter().map(|segment| segment.capacity_tokens)
    }

    /// Return the exact compacted layout after one slot grows. Existing
    /// capacity is never reduced and the logical maximum is never exceeded.
    pub(super) fn grow_slot(
        &self,
        slot: usize,
        required_tokens: u32,
        logical_max_tokens: u32,
    ) -> Result<Self> {
        ensure!(
            required_tokens > 0,
            "TQ arena growth requires non-zero capacity"
        );
        ensure!(
            required_tokens <= logical_max_tokens,
            "TQ arena slot {slot} requires {required_tokens} tokens, exceeding logical context {logical_max_tokens}"
        );
        let current = self.segment(slot)?;
        if required_tokens <= current.capacity_tokens {
            return Ok(self.clone());
        }
        let mut capacities: Vec<u32> = self.capacities().collect();
        capacities[slot] = required_tokens;
        Self::from_capacities(capacities)
    }

    pub(super) fn packed_base_elements(
        &self,
        slot: usize,
        n_kv_heads: u32,
        head_dim: u32,
    ) -> Result<u64> {
        let segment = self.segment(slot)?;
        u64::from(segment.base_tokens)
            .checked_mul(u64::from(n_kv_heads))
            .and_then(|value| value.checked_mul(u64::from(head_dim)))
            .ok_or_else(|| anyhow!("TQ arena packed base offset overflow"))
    }

    pub(super) fn norms_base_elements(
        &self,
        slot: usize,
        n_kv_heads: u32,
        norms_per_pos: u32,
    ) -> Result<u64> {
        let segment = self.segment(slot)?;
        u64::from(segment.base_tokens)
            .checked_mul(u64::from(n_kv_heads))
            .and_then(|value| value.checked_mul(u64::from(norms_per_pos)))
            .ok_or_else(|| anyhow!("TQ arena norm base offset overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unequal_segments_are_compact_and_non_overlapping() {
        let layout = TqArenaLayout::from_capacities(vec![16, 64, 7]).unwrap();
        assert_eq!(layout.n_seqs(), 3);
        assert_eq!(
            layout.segment(0).unwrap(),
            TqArenaSegment {
                base_tokens: 0,
                capacity_tokens: 16
            }
        );
        assert_eq!(
            layout.segment(1).unwrap(),
            TqArenaSegment {
                base_tokens: 16,
                capacity_tokens: 64
            }
        );
        assert_eq!(
            layout.segment(2).unwrap(),
            TqArenaSegment {
                base_tokens: 80,
                capacity_tokens: 7
            }
        );
        assert_eq!(layout.total_capacity_tokens(), 87);
    }

    #[test]
    fn growth_rebases_deeper_slots_without_shrinking_them() {
        let before = TqArenaLayout::from_capacities(vec![16, 64, 7]).unwrap();
        let after = before.grow_slot(0, 48, 262_144).unwrap();
        assert_eq!(after.capacities().collect::<Vec<_>>(), vec![48, 64, 7]);
        assert_eq!(after.segment(1).unwrap().base_tokens, 48);
        assert_eq!(after.segment(2).unwrap().base_tokens, 112);
        assert_eq!(before.segment(1).unwrap().base_tokens, 16);
    }

    #[test]
    fn logical_context_is_independent_from_physical_capacity() {
        let layout = TqArenaLayout::uniform(16, 1).unwrap();
        assert_eq!(layout.total_capacity_tokens(), 16);
        let grown = layout.grow_slot(11, 262_144, 262_144).unwrap();
        assert_eq!(grown.segment(11).unwrap().capacity_tokens, 262_144);
        assert_eq!(grown.total_capacity_tokens(), 262_159);
        assert!(layout.grow_slot(11, 262_145, 262_144).is_err());
    }

    #[test]
    fn qwen38_n16_startup_does_not_create_four_gib_component_buffers() {
        let layout = TqArenaLayout::uniform(16, 1).unwrap();
        let packed_elements = u64::from(layout.total_capacity_tokens()) * 4 * 256;
        assert_eq!(packed_elements, 16_384);
        assert!(packed_elements < 4 * 1024 * 1024 * 1024_u64);
    }

    #[test]
    fn element_bases_track_component_geometry() {
        let layout = TqArenaLayout::from_capacities(vec![16, 64]).unwrap();
        assert_eq!(
            layout.packed_base_elements(1, 4, 256).unwrap(),
            16 * 4 * 256
        );
        assert_eq!(layout.norms_base_elements(1, 4, 1).unwrap(), 16 * 4);
    }
}
