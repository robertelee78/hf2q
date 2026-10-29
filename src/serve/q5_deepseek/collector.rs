//! Owning-thread callback sink. Metadata-only; never retains MlxBuffer/Metal.
//! No cross-thread/global fallback or physical-release inference.
use anyhow::{ensure, Context, Result};
use mlx_native::{MlxBuffer, RootOriginSnapshot};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(crate) enum Domain {
    Weight,
    Prefill,
    Decode,
    SubmissionInput,
    Persistent,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lifecycle {
    Provision,
    Warmup,
}
#[derive(Debug, serde::Serialize)]
struct Root {
    bytes: usize,
    domain: Option<Domain>,
    // FD dev/ino + registered file range + mapping file range, not addresses.
    file_span: Option<(u64, u64, u64, u64, u64, u64)>,
    // Actual OS page geometry observed at the NoCopy callback, not assumed.
    file_page_bytes: Option<u64>,
}
#[derive(Debug)]
struct Source {
    file: std::fs::File,
    expected: super::packet::FileIdentity,
    tensor_data_offset: u64,
}

#[derive(Debug)]
pub(super) struct Collector {
    roots: BTreeMap<usize, Root>,
    // Actual model-held weight parents survive observational checkpoint reset.
    weights: BTreeMap<usize, Root>,
    source: Option<Source>,
    native_domain: native_domain::NativeDomain,
    page_observations: BTreeMap<usize, super::page_observer::RootObservation>,
    events: u64,
    completed_results: u64,
    transient_observations: [(u64, u64); 4],
    pool_latest: [(usize, usize); 3],
    pool_observations: u64,
    pool_observed_highwater: [usize; 3],
    upper: usize,
    poisoned: bool,
    owner: Option<Vec<(u64, u32, u64)>>,
    completed_owners: [Option<(u64, u64)>; 4],
    lifecycle: Option<Lifecycle>,
    manifest_sequence: u64,
    manifest: Option<super::manifest::CurrentManifest>,
    mounted_pair: Option<super::manifest::CurrentManifest>,
    mounted: bool,
    attached: Option<std::thread::ThreadId>,
    next_chain: u64,
    commands: BTreeMap<(u64, usize), Option<bool>>,
}
impl Collector {
    fn buffer(&mut self, domain: Option<Domain>, label: &str, buffer: &MlxBuffer) -> Result<()> {
        ensure!(!self.poisoned, "Q5 collector poisoned");
        ensure!(
            !label.is_empty() && label.len() <= 256,
            "Q5 allocation role is unbounded"
        );
        self.events = self.events.checked_add(1).context("Q5 event overflow")?;
        // Cumulative event count is constant-space, not a per-case event cap.
        // Distinct roots are a bounded CHECKPOINT census, never creation history.
        let b = buffer.backing_snapshot()?;
        let RootOriginSnapshot::KnownNative {
            id,
            requested_bytes,
        } = b.origin
        else {
            anyhow::bail!("Q5 Unknown native allocation lineage");
        };
        ensure!(
            b.view_byte_offset
                .checked_add(u64::try_from(b.logical_bytes)?)
                .is_some_and(|end| end <= b.metal_root_bytes as u64),
            "Q5 view span invalid"
        );
        let transient_index = match domain {
            Some(Domain::Prefill) => Some(0),
            Some(Domain::Decode) => Some(1),
            Some(Domain::SubmissionInput) => Some(2),
            Some(Domain::Persistent) => Some(3),
            _ => None,
        };
        if let Some(index) = transient_index {
            let stats = &mut self.transient_observations[index];
            stats.0 = stats
                .0
                .checked_add(1)
                .context("Q5 allocation observation count overflow")?;
            stats.1 = stats
                .1
                .checked_add(u64::try_from(requested_bytes)?)
                .context("Q5 observation-byte sum overflow")?;
            // This is NOT unique/live/requested-creation byte accounting. Pools
            // can return aliases repeatedly. Exact private domain partition and
            // native live+pending residual coverage remain separate required cells.
            return Ok(());
        }
        if domain.is_none() {
            if let Some(weight) = self.weights.get(&id) {
                ensure!(
                    weight.bytes == requested_bytes,
                    "Q5 role/weight parent extent changed"
                );
                return Ok(()); // SAME actual native parent; not file-page dedup
            }
        }
        let roots = if domain == Some(Domain::Weight) {
            &mut self.weights
        } else {
            &mut self.roots
        };
        if let Some(root) = roots.get_mut(&id) {
            ensure!(
                root.bytes == requested_bytes,
                "Q5 native root changed requested extent"
            );
            if let Some(domain) = domain {
                ensure!(
                    root.domain.is_none_or(|prior| prior == domain),
                    "Q5 root crossed certified source-private allocation domains"
                );
                root.domain = Some(domain);
            }
        } else {
            ensure!(roots.len() < self.upper, "Q5 root capacity exceeded");
            roots.insert(
                id,
                Root {
                    bytes: requested_bytes,
                    domain,
                    file_span: None,
                    file_page_bytes: None,
                },
            );
        }
        Ok(())
    }
}

mod context;
mod coverage;
mod native_domain;
mod observations;
pub(crate) use context::{Handle, Scope};
pub(crate) use observations::{
    cache_state, mounted_begin, mounted_end, snapshot_begin, snapshot_end,
};
pub(super) type HeaderScope = Scope;
use context::with_current;

pub(crate) fn active() -> bool {
    context::active()
}

pub(crate) fn allocation(domain: Domain, label: &str, buffer: &MlxBuffer) -> Result<()> {
    // Sticky only: never abandon already committed work on an observation error.
    let _ = with_current(|c| c.buffer(Some(domain), label, buffer));
    Ok(())
}
pub(crate) fn constructed(label: &str, buffer: &MlxBuffer) {
    let _ = allocation(Domain::Persistent, label, buffer);
}
pub(crate) fn field(label: &str, buffer: Option<&MlxBuffer>) -> Result<()> {
    with_current(|c| {
        if let Some(buffer) = buffer {
            c.buffer(None, label, buffer)?;
        } else {
            ensure!(!c.poisoned, "Q5 absent field health");
            c.events = c
                .events
                .checked_add(1)
                .context("Q5 absent event overflow")?;
        }
        let manifest = match c.mounted_pair.as_mut() {
            Some(pair) if pair.end_ns.is_none() => Some(pair),
            _ => c.manifest.as_mut().filter(|m| m.end_ns.is_none()),
        };
        if let Some(manifest) = manifest {
            manifest.field(label, buffer)?;
        }
        Ok(())
    })
}

pub(crate) fn chain() -> u64 {
    let mut id = 0;
    let _ = with_current(|c| {
        id = c.next_chain;
        c.next_chain = id.checked_add(1).context("Q5 source chain ID overflow")?;
        Ok(())
    });
    id
}
pub(crate) fn command(chain: u64, index: usize, completed: Option<bool>) {
    // Record even after another metadata failure; all real own GPU checks must
    // still occur and actual SubmissionChain retains encoders through its drain.
    let _ = with_current(|c| {
        ensure!(chain != 0, "Q5 command has no source chain");
        c.events = c
            .events
            .checked_add(1)
            .context("Q5 command event overflow")?;
        match completed {
            None => {
                ensure!(
                    c.commands.len() < 256 && !c.commands.contains_key(&(chain, index)),
                    "Q5 pending command capacity/duplicate"
                );
                c.commands.insert((chain, index), None);
            }
            Some(ok) => {
                let result = c
                    .commands
                    .get_mut(&(chain, index))
                    .context("Q5 unsolicited checked Result")?;
                ensure!(result.is_none(), "Q5 duplicate checked Result");
                *result = Some(ok);
                if !ok {
                    c.poisoned = true;
                }
            }
        }
        Ok(())
    });
}
pub(crate) fn checked_chain(chain: u64, expected: usize) -> Result<()> {
    with_current(|c| {
        ensure!(
            !c.poisoned && chain != 0,
            "Q5 checked-chain health/identity"
        );
        let mut results = c.commands.range((chain, 0)..=(chain, usize::MAX));
        ensure!(
            expected <= 256
                && (0..expected).all(|index| {
                    results.next().is_some_and(|((_, actual), result)| {
                        *actual == index && *result == Some(true)
                    })
                })
                && results.next().is_none(),
            "Q5 own command Result missing/failed"
        );
        c.completed_results = c
            .completed_results
            .checked_add(expected as u64)
            .context("Q5 all-Results count overflow")?;
        c.commands.retain(|(id, _), _| *id != chain);
        // Metadata only. Actual completed encoders still belong to the chain.
        Ok(())
    })
}
pub(crate) fn checked_health() -> Result<()> {
    with_current(|c| {
        ensure!(
            !c.poisoned,
            "Q5 sticky observation failure after checked boundary"
        );
        native_domain::require_bound_health(&mut c.native_domain)?;
        Ok(())
    })
}
#[cfg(test)]
pub(crate) fn owner_begin(request: u64, slot_id: u32, generation: u64) -> Result<()> {
    owner_begin_many(&[(request, slot_id, generation)])
}
pub(crate) fn owner_begin_many(keys: &[(u64, u32, u64)]) -> Result<()> {
    with_current(|c| {
        ensure!(
            !c.poisoned
                && c.lifecycle.is_none()
                && c.owner.is_none()
                && c.commands.is_empty()
                && !keys.is_empty()
                && keys.len() <= 4,
            "Q5 overlapping/invalid owning transaction"
        );
        for (index, key) in keys.iter().enumerate() {
            ensure!(
                key.1 < 4
                    && !keys[..index]
                        .iter()
                        .any(|prior| prior.0 == key.0 || prior.1 == key.1),
                "Q5 duplicate/stale logical owner map"
            );
        }
        c.roots.clear(); // observational checkpoint only; no native/live refund
        c.mounted = false;
        c.owner = Some(keys.to_vec());
        Ok(())
    })
}
#[cfg(test)]
pub(crate) fn owner_finish(request: u64, slot_id: u32, generation: u64) -> Result<()> {
    owner_finish_many(&[(request, slot_id, generation)])
}
pub(crate) fn owner_finish_many(keys: &[(u64, u32, u64)]) -> Result<()> {
    with_current(|c| {
        ensure!(
            !c.poisoned && !c.mounted && c.commands.is_empty() && c.owner.as_deref() == Some(keys),
            "Q5 stale request/generation or callback health"
        );
        for &(request, slot, generation) in keys {
            c.completed_owners[slot as usize] = Some((request, generation));
        }
        c.owner = None;
        Ok(())
    })
}
pub(crate) fn retained_cleanup_request(slot: u32, generation: u64) -> Result<u64> {
    let mut request = None;
    with_current(|c| {
        ensure!(!c.poisoned && slot < 4, "Q5 cleanup witness health/slot");
        let prior =
            c.completed_owners[slot as usize].context("Q5 no actual completed owner witness")?;
        ensure!(prior.1 == generation, "Q5 stale terminal cleanup witness");
        request = Some(prior.0);
        Ok(())
    })?;
    request.context("Q5 terminal cleanup lacks actual attached collector")
}
pub(crate) fn invalidate(reason: &'static str) -> Result<()> {
    with_current(|_| anyhow::bail!("Q5 invalid observation/transaction: {reason}"))
}
pub(super) fn attach_source(
    file: &std::fs::File,
    expected: super::packet::FileIdentity,
    tensor_data_offset: u64,
) -> Result<()> {
    expected.matches(&file.metadata()?)?;
    with_current(|c| {
        ensure!(c.source.is_none(), "Q5 duplicate/changed SameFD source");
        c.source = Some(Source {
            file: file.try_clone()?,
            expected,
            tensor_data_offset,
        });
        Ok(())
    })
}
mod weight_observation;
pub(crate) use weight_observation::weight;

pub(crate) fn pools(snapshots: [mlx_native::PoolMemorySnapshot; 3]) -> Result<()> {
    with_current(|c| {
        ensure!(!c.poisoned, "Q5 pool callback health");
        c.pool_observations = c
            .pool_observations
            .checked_add(1)
            .context("Q5 pool observation counter overflow")?;
        for (index, snapshot) in snapshots.into_iter().enumerate() {
            ensure!(
                snapshot.retained.roots <= 16_384
                    && snapshot.free.bytes <= snapshot.retained.bytes
                    && snapshot.issued.bytes <= snapshot.retained.bytes
                    && snapshot.residency_hint.bytes <= snapshot.retained.bytes,
                "Q5 actual private-pool snapshot invalid"
            );
            c.pool_latest[index] = (snapshot.retained.roots, snapshot.retained.bytes);
            c.pool_observed_highwater[index] =
                c.pool_observed_highwater[index].max(snapshot.retained.bytes);
        }
        // Labels overlap; outside owners/physical/lifetime highwater unavailable.
        Ok(())
    })
}
#[cfg(test)]
pub(super) fn observed_root_totals() -> Result<(usize, usize)> {
    let mut out = None;
    with_current(|c| {
        ensure!(!c.poisoned, "Q5 test census health");
        let sum = c
            .roots
            .values()
            .try_fold(0usize, |sum, r| sum.checked_add(r.bytes))
            .context("Q5 test census overflow")?;
        out = Some((c.roots.len(), sum));
        Ok(())
    })?;
    out.context("Q5 test census missing scope")
}

pub(crate) fn lifecycle_begin(kind: Lifecycle) -> Result<()> {
    with_current(|c| {
        ensure!(
            !c.poisoned && c.owner.is_none() && c.lifecycle.is_none() && c.commands.is_empty(),
            "Q5 overlapping lifecycle/pending work"
        );
        c.roots.clear(); // observational only, never weights/live/physical refund
        c.lifecycle = Some(kind);
        Ok(())
    })
}
pub(crate) fn lifecycle_finish(kind: Lifecycle, healthy: bool) -> Result<()> {
    with_current(|c| {
        ensure!(
            healthy
                && !c.poisoned
                && c.owner.is_none()
                && c.lifecycle == Some(kind)
                && c.commands.is_empty(),
            "Q5 lifecycle failed/unjoined"
        );
        c.lifecycle = None;
        Ok(())
    })
}
pub(crate) fn require_lifecycle(kind: Lifecycle) -> Result<()> {
    with_current(|c| {
        ensure!(
            !c.poisoned && c.lifecycle == Some(kind),
            "Q5 lifecycle authority missing"
        );
        Ok(())
    })
}

pub(crate) fn source_unchanged() -> Result<()> {
    with_current(|c| {
        let source = c
            .source
            .as_ref()
            .context("Q5 model observation has no actual SameFD owner")?;
        source.expected.matches(&source.file.metadata()?)
    })
}

#[cfg(test)]
pub(crate) use native_domain::bind_actual_for_tiny;
#[cfg(test)]
pub(crate) fn capture_native_for_tiny() -> Result<()> {
    with_current(|c| {
        c.native_domain.capture();
        ensure!(
            matches!(
                c.native_domain.latest,
                native_domain::Snapshot::CapturedRequestedNotPhysical { .. }
            ),
            "actual tiny native requested authority unavailable"
        );
        Ok(())
    })
}

#[cfg(test)]
mod provider_control;

/// Selected-only real native installation; caller holds the private review token.
pub(super) fn install_reviewed_native(
    context: &Handle,
    budget: super::ReviewedSourceBudget,
) -> Result<()> {
    context.require_current_model(Some(context))?;
    let (creation, live) = budget.into_limits();
    with_current(|c| {
        ensure!(!c.poisoned, "Q5 poisoned early authority installation");
        native_domain::install_reviewed(&mut c.native_domain, creation, live)
    })
}
/// Actual selected model seam, before weights. No TLS means original path.
pub(crate) fn register_model_device(device: &mlx_native::MlxDevice) -> Result<()> {
    with_current(|c| {
        ensure!(!c.poisoned, "Q5 poisoned model device binding");
        native_domain::register_model_device(&mut c.native_domain, device)
    })
}
