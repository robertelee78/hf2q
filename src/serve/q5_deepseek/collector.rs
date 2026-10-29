//! Owning-thread callback sink. Metadata-only; never retains MlxBuffer/Metal.
//! No cross-thread/global fallback or physical-release inference.
use anyhow::{ensure, Context, Result};
use mlx_native::{MlxBuffer, RootOriginSnapshot};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
};

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
    source: Option<Source>,
    events: u64,
    completed_results: u64,
    transient_observations: [(u64, u64); 4],
    pool_latest: [(usize, usize); 3],
    pool_highwater: [usize; 3],
    upper: usize,
    poisoned: bool,
    owner: Option<(u64, u32, u64)>,
    next_chain: u64,
    commands: BTreeMap<(u64, usize), Option<bool>>,
}
thread_local! {
    static ACTIVE: RefCell<Option<Collector>> = const { RefCell::new(None) };
    static REENTRY_FAILED: Cell<bool> = const { Cell::new(false) };
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
        if let Some(root) = self.roots.get_mut(&id) {
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
            ensure!(self.roots.len() < self.upper, "Q5 root capacity exceeded");
            self.roots.insert(
                id,
                Root {
                    bytes: requested_bytes,
                    domain,
                },
            );
        }
        Ok(())
    }
}

/// Only the selected header entry can create this scope in production. It
/// cannot admit a model; no runtime permit exists in this checkpoint.
pub(super) struct HeaderScope;
impl HeaderScope {
    pub(super) fn enter() -> Result<Self> {
        ACTIVE.with(|slot| {
            let mut slot = slot.try_borrow_mut().context("Q5 collector reentry")?;
            ensure!(slot.is_none(), "Q5 overlapping diagnostic owner");
            REENTRY_FAILED.with(|flag| flag.set(false));
            *slot = Some(Collector {
                roots: BTreeMap::new(),
                source: None,
                events: 0,
                completed_results: 0,
                transient_observations: [(0, 0); 4],
                pool_latest: [(0, 0); 3],
                pool_highwater: [0; 3],
                upper: 16_384,
                poisoned: false,
                owner: None,
                next_chain: 1,
                commands: BTreeMap::new(),
            });
            Ok(Self)
        })
    }
}
impl Drop for HeaderScope {
    fn drop(&mut self) {
        ACTIVE.with(|slot| {
            slot.borrow_mut().take();
        });
    }
}

/// Default None performs no native query, allocation or metadata clone. The
/// callback runs only AFTER pool RefCell borrows are released by its caller.
pub(crate) fn allocation(domain: Domain, label: &str, buffer: &MlxBuffer) -> Result<()> {
    ACTIVE.with(|slot| {
        let Ok(mut slot) = slot.try_borrow_mut() else {
            REENTRY_FAILED.with(|flag| flag.set(true));
            return Ok(()); // existing checked boundary will fail after draining
        };
        let Some(c) = slot.as_mut() else {
            return Ok(());
        };
        if c.buffer(Some(domain), label, buffer).is_err() {
            c.poisoned = true;
        }
        // Never interrupt an in-progress pipeline and abandon already committed
        // commands merely because metadata failed. Existing checked boundaries
        // surface sticky health AFTER draining every own command Result.
        Ok(())
    })
}

/// A source-local identity, never a native command UUID or root-owner list.
pub(crate) fn chain() -> u64 {
    ACTIVE.with(|slot| {
        let Ok(mut slot) = slot.try_borrow_mut() else {
            REENTRY_FAILED.with(|flag| flag.set(true));
            return 0;
        };
        let Some(c) = slot.as_mut() else {
            return 0;
        };
        let id = c.next_chain;
        match id.checked_add(1) {
            Some(next) => c.next_chain = next,
            None => c.poisoned = true,
        }
        id
    })
}

/// Do not abandon committed work on metadata failure: poison, retain the real
/// encoder in SubmissionChain, and surface the error AFTER all existing checks.
pub(crate) fn command(chain: u64, index: usize, completed: Option<bool>) {
    ACTIVE.with(|slot| {
        let Ok(mut slot) = slot.try_borrow_mut() else {
            REENTRY_FAILED.with(|flag| flag.set(true));
            return;
        };
        let Some(c) = slot.as_mut() else {
            return;
        };
        if chain == 0 {
            c.poisoned = true;
            return;
        }
        let Some(next) = c.events.checked_add(1) else {
            c.poisoned = true;
            return;
        };
        c.events = next;
        match completed {
            None => {
                if c.commands.len() >= 256 || c.commands.insert((chain, index), None).is_some() {
                    c.poisoned = true;
                }
            }
            Some(ok) => match c.commands.get_mut(&(chain, index)) {
                Some(value) if value.is_none() => {
                    *value = Some(ok);
                    if !ok {
                        c.poisoned = true;
                    }
                }
                _ => c.poisoned = true,
            },
        }
        // Completed encoders are STILL retained by the chain. Keep their
        // records; no byte refund/physical retirement follows from completion.
    });
}

pub(crate) fn checked_chain(chain: u64, expected: usize) -> Result<()> {
    ACTIVE.with(|slot| {
        let mut slot = slot.try_borrow_mut().context("Q5 checked-chain reentry")?;
        let Some(c) = slot.as_mut() else {
            return Ok(());
        };
        ensure!(
            !c.poisoned && chain != 0,
            "Q5 submission observation poisoned"
        );
        let mut results = c.commands.range((chain, 0)..=(chain, usize::MAX));
        let complete = expected <= 256
            && (0..expected).all(|index| {
                results
                    .next()
                    .is_some_and(|((_, actual), result)| *actual == index && *result == Some(true))
            })
            && results.next().is_none()
            && !REENTRY_FAILED.with(Cell::get);
        if !complete {
            c.poisoned = true;
            anyhow::bail!("Q5 own command Result missing/failed");
        }
        let count = c.commands.range((chain, 0)..=(chain, usize::MAX)).count();
        c.completed_results = c
            .completed_results
            .checked_add(count as u64)
            .context("Q5 all-Results aggregate overflow")?;
        c.commands.retain(|(id, _), _| *id != chain);
        // This discards ONLY completed metadata records. It is NOT resource
        // retirement: encoders still belong to their actual SubmissionChain.
        Ok(())
    })
}

pub(crate) fn active() -> bool {
    ACTIVE.with(|slot| slot.borrow().is_some())
}

pub(crate) fn field(label: &str, buffer: Option<&MlxBuffer>) -> Result<()> {
    ACTIVE.with(|slot| {
        if let Some(buffer) = buffer {
            let mut slot = slot.try_borrow_mut().context("Q5 role field reentry")?;
            let Some(c) = slot.as_mut() else {
                return Ok(());
            };
            // A role observation is NOT an allocator-domain assertion. A live
            // logit can legitimately reference an allocation already observed.
            let result = c.buffer(None, label, buffer);
            if result.is_err() {
                c.poisoned = true;
            }
            return result;
        }
        let mut slot = slot.try_borrow_mut().context("Q5 absent field reentry")?;
        if let Some(c) = slot.as_mut() {
            ensure!(!c.poisoned, "Q5 absent field health");
            c.events = c
                .events
                .checked_add(1)
                .context("Q5 absent event overflow")?;
        }
        Ok(())
    })
}

pub(crate) fn owner_begin(request: u64, slot_id: u32, generation: u64) -> Result<()> {
    ACTIVE.with(|slot| {
        let mut slot = slot.try_borrow_mut().context("Q5 owner callback reentry")?;
        let Some(c) = slot.as_mut() else {
            return Ok(());
        };
        if c.poisoned || c.owner.is_some() || slot_id >= 4 {
            c.poisoned = true;
            anyhow::bail!("Q5 overlapping/invalid owning transaction");
        }
        ensure!(c.commands.is_empty(), "Q5 prior unjoined pending commands");
        // Reset only bounded observational census. All retained cache owners are
        // revisited below. No native ownership charge, live counter or physical
        // term is reset/refunded; native current-live coverage remains required.
        c.roots.clear();
        c.owner = Some((request, slot_id, generation)); // request zero is valid
        Ok(())
    })
}
pub(crate) fn owner_finish(request: u64, slot_id: u32, generation: u64) -> Result<()> {
    ACTIVE.with(|slot| {
        let mut slot = slot
            .try_borrow_mut()
            .context("Q5 owner completion reentry")?;
        let Some(c) = slot.as_mut() else {
            return Ok(());
        };
        if c.poisoned || c.owner != Some((request, slot_id, generation)) {
            c.poisoned = true;
            anyhow::bail!("Q5 stale request/generation or callback health failure");
        }
        c.owner = None;
        Ok(())
    })
}

pub(crate) fn checked_health() -> Result<()> {
    ACTIVE.with(|slot| {
        let slot = slot
            .try_borrow()
            .context("Q5 healthy checked boundary reentry")?;
        ensure!(
            !REENTRY_FAILED.with(Cell::get),
            "Q5 callback reentry failed"
        );
        if let Some(c) = slot.as_ref() {
            ensure!(
                !c.poisoned,
                "Q5 sticky metadata failure after checked boundary"
            );
        }
        Ok(())
    })
}

pub(super) fn attach_source(
    file: &std::fs::File,
    expected: super::packet::FileIdentity,
    tensor_data_offset: u64,
) -> Result<()> {
    expected.matches(&file.metadata()?)?;
    ACTIVE.with(|slot| {
        let mut slot = slot.try_borrow_mut().context("Q5 source reentry")?;
        let c = slot.as_mut().context("Q5 no owning source collector")?;
        ensure!(c.source.is_none(), "Q5 duplicate/changed SameFD source");
        c.source = Some(Source {
            file: file.try_clone()?,
            expected,
            tensor_data_offset,
        });
        Ok(())
    })
}

/// Actual residency-loop output, joined to the retained real loader FD, not a
/// second load/map. This CPU/map-only phase has no submitted forward commands;
/// a mismatch refuses before the next tensor constructor.
pub(crate) fn weight(
    gguf: &mlx_native::gguf::GgufFile,
    name: &str,
    buffer: &MlxBuffer,
) -> Result<()> {
    ACTIVE.with(|slot| {
        let mut slot = slot.try_borrow_mut().context("Q5 weight reentry")?;
        let Some(c) = slot.as_mut() else { return Ok(()); };
        let result = (|| -> Result<()> {
            let source = c.source.as_ref().context("Q5 weight missing actual SameFD owner")?;
            source.expected.matches(&source.file.metadata()?)?;
            ensure!(source.tensor_data_offset == gguf.tensor_data_offset(), "Q5 loader tensor offset drift");
            let info = gguf.tensor_info(name).context("Q5 actual tensor missing")?;
            let logical = source.tensor_data_offset.checked_add(info.offset).context("Q5 file offset overflow")?;
            ensure!(logical.checked_add(info.byte_len as u64).is_some_and(|end| end <= source.expected.bytes),
                "Q5 actual tensor escapes SameFD extent");
            let b = buffer.backing_snapshot()?;
            match b.provenance {
                mlx_native::BufferProvenance::NoCopy => {
                    let n = b.no_copy.context("Q5 NoCopy missing actual mapping range")?;
                    let within = n.registered_base_address.checked_sub(n.mapping_base_address).context("Q5 NoCopy range order")?;
                    let registered = logical.checked_sub(b.view_byte_offset).context("Q5 registered file offset underflow")?;
                    let mapping = registered.checked_sub(within as u64).context("Q5 mapping file offset underflow")?;
                    ensure!(b.logical_bytes == info.byte_len && n.registered_bytes == b.metal_root_bytes
                        && mapping.checked_add(n.mapping_bytes as u64).is_some_and(|end| end <= source.expected.bytes),
                        "Q5 SameFD mapped tensor span mismatch");
                }
                mlx_native::BufferProvenance::Owned => { /* F32 expansion or explicit owned load; separate parent. */ }
                _ => anyhow::bail!("Q5 Unknown weight provenance"),
            }
            source.expected.matches(&source.file.metadata()?)?;
            c.buffer(Some(Domain::Weight), name, buffer)
        })();
        if result.is_err() { c.poisoned = true; }
        // CPU/map-only residency load has no submitted forward commands.
        // Refuse before the next tensor constructor rather than finish all load.
        result
    })
}

#[cfg(test)]
pub(super) fn observed_root_totals() -> Result<(usize, usize)> {
    ACTIVE.with(|slot| {
        let slot = slot.try_borrow().context("Q5 test census borrow")?;
        let c = slot
            .as_ref()
            .context("Q5 test census missing actual scope")?;
        ensure!(!c.poisoned, "Q5 test census poisoned");
        let total = c
            .roots
            .values()
            .try_fold(0usize, |sum, root| sum.checked_add(root.bytes))
            .context("Q5 test census overflow")?;
        Ok((c.roots.len(), total))
    })
}

pub(crate) fn pools(snapshots: [mlx_native::PoolMemorySnapshot; 3]) -> Result<()> {
    ACTIVE.with(|slot| {
        let mut slot = slot.try_borrow_mut().context("Q5 pool callback reentry")?;
        let Some(c) = slot.as_mut() else {
            return Ok(());
        };
        ensure!(!c.poisoned, "Q5 pool callback poisoned");
        for (index, snapshot) in snapshots.into_iter().enumerate() {
            ensure!(
                snapshot.retained.roots <= 16_384
                    && snapshot.free.bytes <= snapshot.retained.bytes
                    && snapshot.issued.bytes <= snapshot.retained.bytes
                    && snapshot.residency_hint.bytes <= snapshot.retained.bytes,
                "Q5 actual private-pool snapshot invalid"
            );
            c.pool_latest[index] = (snapshot.retained.roots, snapshot.retained.bytes);
            c.pool_highwater[index] = c.pool_highwater[index].max(snapshot.retained.bytes);
        }
        // Three source-private pool values stay separate from cache-role roots,
        // native outside-pool residual and physical/highwater Unknowns. Do not
        // add overlapping free/issued/residency labels or infer a global union.
        Ok(())
    })
}

/// Non-throwing constructor tap for native-error-typed cache helpers. Sticky
/// failure is checked at the owning worker's existing terminal boundary.
pub(crate) fn constructed(label: &str, buffer: &MlxBuffer) {
    if allocation(Domain::Persistent, label, buffer).is_err() {
        REENTRY_FAILED.with(|flag| flag.set(true));
    }
}
