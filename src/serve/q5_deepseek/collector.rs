//! Owning-thread callback sink. Metadata-only; never retains MlxBuffer/Metal.
//! No cross-thread/global fallback or physical-release inference.
use anyhow::{ensure, Context, Result};
use mlx_native::{MlxBuffer, RootOriginSnapshot};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Domain {
    Weight,
    Prefill,
    Decode,
    SubmissionInput,
    Persistent,
}
#[derive(Debug)]
struct Root {
    bytes: usize,
    domain: Option<Domain>,
    // FD dev/ino + registered file range + mapping file range, not addresses.
    file_span: Option<(u64, u64, u64, u64, u64, u64)>,
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
    events: u64,
    completed_results: u64,
    transient_observations: [(u64, u64); 4],
    pool_latest: [(usize, usize); 3],
    pool_observed_highwater: [usize; 3],
    upper: usize,
    poisoned: bool,
    owner: Option<Vec<(u64, u32, u64)>>,
    completed_owners: [Option<(u64, u64)>; 4],
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
                },
            );
        }
        Ok(())
    }
}

mod context;
pub(crate) use context::{Handle, Scope};
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
            c.buffer(None, label, buffer)
        } else {
            ensure!(!c.poisoned, "Q5 absent field health");
            c.events = c
                .events
                .checked_add(1)
                .context("Q5 absent event overflow")?;
            Ok(())
        }
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
            !c.poisoned && c.commands.is_empty() && c.owner.as_deref() == Some(keys),
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
pub(crate) fn weight(
    gguf: &mlx_native::gguf::GgufFile,
    name: &str,
    buffer: &MlxBuffer,
) -> Result<()> {
    with_current(|c| {
        let source = c
            .source
            .as_ref()
            .context("Q5 weight missing actual SameFD owner")?;
        source.expected.matches(&source.file.metadata()?)?;
        ensure!(
            source.tensor_data_offset == gguf.tensor_data_offset(),
            "Q5 loader tensor offset drift"
        );
        let info = gguf.tensor_info(name).context("Q5 actual tensor missing")?;
        let logical = source
            .tensor_data_offset
            .checked_add(info.offset)
            .context("Q5 file offset overflow")?;
        ensure!(
            logical
                .checked_add(info.byte_len as u64)
                .is_some_and(|end| end <= source.expected.bytes),
            "Q5 tensor escapes SameFD extent"
        );
        let b = buffer.backing_snapshot()?;
        let file_span = match b.provenance {
            mlx_native::BufferProvenance::NoCopy => {
                let n = b
                    .no_copy
                    .context("Q5 NoCopy missing actual mapping range")?;
                let within = n
                    .registered_base_address
                    .checked_sub(n.mapping_base_address)
                    .context("Q5 NoCopy range order")?;
                let registered = logical
                    .checked_sub(b.view_byte_offset)
                    .context("Q5 registered offset underflow")?;
                let mapping = registered
                    .checked_sub(within as u64)
                    .context("Q5 mapping offset underflow")?;
                ensure!(
                    b.logical_bytes == info.byte_len
                        && n.registered_bytes == b.metal_root_bytes
                        && mapping
                            .checked_add(n.mapping_bytes as u64)
                            .is_some_and(|end| end <= source.expected.bytes),
                    "Q5 actual mapped span mismatch"
                );
                Some((
                    source.expected.device,
                    source.expected.inode,
                    registered,
                    n.registered_bytes as u64,
                    mapping,
                    n.mapping_bytes as u64,
                ))
            }
            mlx_native::BufferProvenance::Owned => None,
            _ => anyhow::bail!("Q5 Unknown weight provenance"),
        };
        source.expected.matches(&source.file.metadata()?)?;
        c.buffer(Some(Domain::Weight), name, buffer)?;
        if let RootOriginSnapshot::KnownNative { id, .. } = b.origin {
            let root = c
                .weights
                .get_mut(&id)
                .context("Q5 actual weight root census missing")?;
            ensure!(
                root.file_span.is_none() || root.file_span == file_span,
                "Q5 one root changed actual SameFD span"
            );
            root.file_span = file_span;
        }
        Ok(())
    })
}
pub(crate) fn pools(snapshots: [mlx_native::PoolMemorySnapshot; 3]) -> Result<()> {
    with_current(|c| {
        ensure!(!c.poisoned, "Q5 pool callback health");
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
