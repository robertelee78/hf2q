//! SAME actual native handle; requested creation/live limits, not host admission.
use super::NativeDomain;
use anyhow::{ensure, Context, Result};
use mlx_native::{LiveRootLimits, LiveRootSnapshot, MlxDevice, NativeRootAuthority, RootCreationLimits, RootCreationSnapshot};

fn validate_limits(c: RootCreationLimits, l: LiveRootLimits) -> Result<()> {
    ensure!(c.max_attempted_requested_bytes > 0 && c.max_root_attempts > 0
        && c.max_raw_constructor_attempts >= c.max_root_attempts
        && c.max_registered_devices == 1 && c.max_rope_cache_entries > 0,
        "Q5 invalid selected native creation limits");
    ensure!(l.max_live_and_pending_requested_bytes > 0 && l.max_live_and_pending_roots > 0
        && l.max_live_and_pending_requested_bytes <= c.max_attempted_requested_bytes
        && l.max_live_and_pending_roots <= c.max_root_attempts,
        "Q5 invalid native live versus historical limits");
    Ok(())
}
pub(super) fn validate_snapshots(c: &RootCreationSnapshot, l: &LiveRootSnapshot, fresh: bool) -> Result<()> {
    ensure!(c.covered && !c.legacy_used && c.unknown.is_none() && l.covered && l.unknown.is_none(),
        "Q5 unknown/uncovered native domain");
    let creation = c.limits.context("Q5 missing native creation limits")?;
    let live = l.limits.context("Q5 missing native live limits")?;
    validate_limits(creation, live)?;
    let bytes = l.live_requested_bytes.checked_add(l.pending_requested_bytes).context("Q5 native live sum overflow")?;
    let roots = l.live_roots.checked_add(l.pending_roots).context("Q5 native root sum overflow")?;
    ensure!(c.denied == 0 && l.denied == 0 && c.inflight == 0 && l.pending_roots == 0
        && l.pending_requested_bytes == 0
        && c.attempted_requested_bytes <= creation.max_attempted_requested_bytes
        && c.root_attempts <= creation.max_root_attempts
        && c.raw_attempt_reservations <= creation.max_raw_constructor_attempts
        && c.registered_devices <= creation.max_registered_devices
        && bytes <= live.max_live_and_pending_requested_bytes
        && roots <= live.max_live_and_pending_roots
        && l.high_water_requested_bytes <= live.max_live_and_pending_requested_bytes,
        "Q5 native denied/inflight/over-limit checkpoint");
    if fresh {
        ensure!(c.attempted_requested_bytes == 0 && c.root_attempts == 0 && c.successful_roots == 0
            && c.raw_attempt_reservations == 0 && c.registered_devices == 0
            && c.cache_slot_history == 0 && c.published_cache_entries == 0
            && bytes == 0 && roots == 0 && l.high_water_requested_bytes == 0 && l.final_releases == 0,
            "Q5 authority was not bound before actual native/device history");
    }
    Ok(())
}
pub(in crate::serve::q5_deepseek::collector) fn install_reviewed(domain: &mut NativeDomain, c: RootCreationLimits, l: LiveRootLimits) -> Result<()> {
    ensure!(domain.handle.is_none(), "Q5 duplicate early native installation");
    validate_limits(c, l)?;
    // Native rejects any prior installation/legacy/import history. Never reset
    // creation history and never look up an ambient/global handle as fallback.
    let handle = NativeRootAuthority::install_early_with_live(c, l)?;
    domain.bind(handle)
}
pub(in crate::serve::q5_deepseek::collector) fn register_model_device(domain: &mut NativeDomain, device: &MlxDevice) -> Result<()> {
    ensure!(!domain.registered, "Q5 duplicate selected model device binding");
    let handle = domain.handle.context("Q5 selected model has no early native authority")?;
    validate_snapshots(&handle.snapshot()?, &handle.live_snapshot()?, true)?;
    handle.register_device(device)?;
    let c = handle.snapshot()?;
    validate_snapshots(&c, &handle.live_snapshot()?, false)?;
    ensure!(c.registered_devices == 1, "Q5 actual same model device not registered");
    domain.registered = true;
    domain.capture();
    Ok(())
}
pub(in crate::serve::q5_deepseek::collector) fn require_bound_health(domain: &mut NativeDomain) -> Result<()> {
    if let Some(handle) = domain.handle {
        ensure!(domain.registered, "Q5 actual model device not bound");
        let c = handle.snapshot()?;
        validate_snapshots(&c, &handle.live_snapshot()?, false)?;
        ensure!(c.registered_devices == 1, "Q5 selected device history changed");
        domain.capture();
    }
    Ok(())
}
#[cfg(test)]
#[path = "binding_tests.rs"]
mod tests;
