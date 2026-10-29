//! Actual public native requested-domain handle only; no global substitute.
use serde::Serialize;
#[derive(Debug, Serialize)]
#[serde(tag = "status")]
pub(super) enum Snapshot {
    Unavailable {
        reason: &'static str,
    },
    CapturedRequestedNotPhysical {
        scope: &'static str,
        attempted_bytes: usize,
        root_attempts: usize,
        successful_roots: usize,
        inflight: usize,
        raw_attempts: usize,
        creation_denied: usize,
        live_bytes: usize,
        pending_bytes: usize,
        live_roots: usize,
        pending_roots: usize,
        highwater_bytes: usize,
        live_denied: usize,
        final_releases: usize,
        creation_byte_limit: usize,
        creation_root_limit: usize,
        live_byte_limit: usize,
        live_root_limit: usize,
    },
}
pub(super) struct NativeDomain {
    handle: Option<mlx_native::RootAuthorityHandle>,
    registered: bool,
    pub(super) latest: Snapshot,
}
impl std::fmt::Debug for NativeDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.latest.fmt(f)
    }
}
impl Default for NativeDomain {
    fn default() -> Self {
        Self {
            handle: None,
            registered: false,
            latest: Snapshot::Unavailable {
                reason: "NoActualEarlyNativeHandleBound",
            },
        }
    }
}
impl NativeDomain {
    pub(super) fn capture(&mut self) {
        let result = || -> anyhow::Result<Snapshot> {
            let handle = self
                .handle
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no bound handle"))?;
            let creation = handle.snapshot()?;
            let live = handle.live_snapshot()?;
            binding::validate_snapshots(&creation, &live, false)?;
            let c = creation
                .limits
                .ok_or_else(|| anyhow::anyhow!("no creation ceiling"))?;
            let l = live
                .limits
                .ok_or_else(|| anyhow::anyhow!("no live ceiling"))?;
            Ok(Snapshot::CapturedRequestedNotPhysical {
                scope: creation.scope,
                attempted_bytes: creation.attempted_requested_bytes,
                root_attempts: creation.root_attempts,
                successful_roots: creation.successful_roots,
                inflight: creation.inflight,
                raw_attempts: creation.raw_attempt_reservations,
                creation_denied: creation.denied,
                live_bytes: live.live_requested_bytes,
                pending_bytes: live.pending_requested_bytes,
                live_roots: live.live_roots,
                pending_roots: live.pending_roots,
                highwater_bytes: live.high_water_requested_bytes,
                live_denied: live.denied,
                final_releases: live.final_releases,
                creation_byte_limit: c.max_attempted_requested_bytes,
                creation_root_limit: c.max_root_attempts,
                live_byte_limit: l.max_live_and_pending_requested_bytes,
                live_root_limit: l.max_live_and_pending_roots,
            })
        };
        self.latest = result().unwrap_or(Snapshot::Unavailable {
            reason: "MissingPoisonedUncoveredOrNoLiveAuthority",
        });
    }
    pub(super) fn bind(&mut self, handle: mlx_native::RootAuthorityHandle) -> anyhow::Result<()> {
        anyhow::ensure!(self.handle.is_none(), "duplicate native authority binding");
        binding::validate_snapshots(&handle.snapshot()?, &handle.live_snapshot()?, true)?;
        self.handle = Some(handle);
        self.capture();
        Ok(())
    }
}
#[cfg(test)]
pub(crate) fn bind_actual_for_tiny(handle: mlx_native::RootAuthorityHandle) -> anyhow::Result<()> {
    super::with_current(|c| c.native_domain.bind(handle))
}

#[path = "native_domain/binding.rs"]
mod binding;
pub(super) use binding::{install_reviewed, register_model_device, require_bound_health};
