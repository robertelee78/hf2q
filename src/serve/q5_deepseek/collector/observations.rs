//! Bounded current snapshots; no resource/lifetime/physical accounting reset.
use super::{with_current, Lifecycle};
use anyhow::{ensure, Context, Result};
pub(crate) fn snapshot_begin(gpu_registry_id: u64) -> Result<()> {
    with_current(|c| {
        ensure!(
            !c.poisoned && c.manifest.as_ref().is_none_or(|m| m.end_ns.is_some()),
            "Q5 snapshot overlap/health"
        );
        c.manifest_sequence = c
            .manifest_sequence
            .checked_add(1)
            .context("Q5 snapshot sequence overflow")?;
        c.roots.clear(); // CURRENT borrowed-role census, no resource lifetime change
        let lifecycle = c.lifecycle.map(|kind| match kind {
            Lifecycle::Provision => "provision",
            Lifecycle::Warmup => "warmup",
        });
        c.manifest = Some(super::super::manifest::CurrentManifest::new(
            c.manifest_sequence,
            gpu_registry_id,
            c.owner.as_deref().unwrap_or(&[]),
            lifecycle,
        )?);
        Ok(())
    })
}
pub(crate) fn cache_state(
    owner: &str,
    state: crate::inference::models::deepseek4::cache::Q5CacheState,
) -> Result<()> {
    with_current(|c| {
        ensure!(
            !owner.is_empty() && owner.len() <= 128,
            "Q5 cache owner metadata name"
        );
        let manifest = match c.mounted_pair.as_mut() {
            Some(pair) if pair.end_ns.is_none() => pair,
            _ => c
                .manifest
                .as_mut()
                .context("Q5 cache state outside actual snapshot")?,
        };
        ensure!(
            manifest.end_ns.is_none()
                && manifest.cache_states.len() < 15
                && !manifest.cache_states.contains_key(owner),
            "Q5 cache state duplicate/cap/closed"
        );
        manifest.cache_states.insert(owner.to_owned(), state);
        Ok(())
    })
}
pub(crate) fn snapshot_end() -> Result<()> {
    with_current(|c| {
        ensure!(!c.poisoned, "Q5 snapshot closure health");
        c.manifest
            .as_mut()
            .context("Q5 no actual snapshot")?
            .close()
    })
}

pub(crate) fn mounted_begin(gpu_registry_id: u64) -> Result<()> {
    with_current(|c| {
        ensure!(
            !c.poisoned
                && c.lifecycle.is_none()
                && c.owner.as_ref().is_some_and(|owners| owners.len() == 1)
                && c.mounted_pair.as_ref().is_none_or(|m| m.end_ns.is_some()),
            "Q5 swap outside exact single owning operation"
        );
        c.manifest_sequence = c
            .manifest_sequence
            .checked_add(1)
            .context("Q5 mounted sequence overflow")?;
        c.mounted = !c.mounted;
        let kind = if c.mounted {
            "actual_mounted_pair_partial"
        } else {
            "actual_unmounted_pair_partial"
        };
        c.mounted_pair = Some(crate::serve::q5_deepseek::manifest::CurrentManifest::new(
            c.manifest_sequence,
            gpu_registry_id,
            c.owner.as_deref().unwrap_or(&[]),
            Some(kind),
        )?);
        Ok(())
    })
}
pub(crate) fn mounted_end() -> Result<()> {
    with_current(|c| {
        c.mounted_pair
            .as_mut()
            .context("Q5 mounted snapshot absent")?
            .close()
    })
}
