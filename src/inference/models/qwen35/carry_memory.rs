//! Requested backing capacity for target carry and KV-only MTP catch-up.
//! Not physical admission. The caller supplies the complete target-forward
//! bound, including its pool/TLS high-water; no FFN, vocabulary or SDPA runs here.
use std::cell::RefCell;
use anyhow::Result;
use mlx_native::{MlxBuffer, RootOriginSnapshot};

use super::Qwen35Config;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RootId {
    Native(usize),
    Metal(usize),
}

fn identity(buffer: &MlxBuffer) -> RootId {
    match buffer.root_origin_snapshot() {
        RootOriginSnapshot::KnownNative { id, .. } => RootId::Native(id),
        // The owning clone below prevents reuse of this live Metal identity.
        _ => RootId::Metal(
            (&**buffer.metal_buffer() as *const mlx_native::metal::BufferRef) as usize,
        ),
    }
}

/// DFlash2 precedence is a property of this transaction's lane match, not of
/// whether a drafter happens to be loaded on the model.
pub(crate) fn uses_mtp(dflash2_lane_matches: bool, mtp_allowed: bool) -> bool {
    !dflash2_lane_matches && mtp_allowed
}

#[derive(Debug)]
struct CarryMemoryError(String);
impl std::fmt::Display for CarryMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for CarryMemoryError {}

pub(crate) fn is_commitment_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| cause.is::<CarryMemoryError>())
}

fn require_fit(required: u64, limit: u64, phase: &str) -> Result<()> {
    if required > limit {
        return Err(CarryMemoryError(format!(
            "Qwen {phase} exceeds admitted carry commitment: {required} > {limit}",
        )).into());
    }
    Ok(())
}

/// One owning handle per distinct full backing, never per slot/row/view.
pub(crate) struct CarryOwners {
    roots: Vec<(RootId, MlxBuffer)>,
}

thread_local! {
    // Worker inventory plus one nested catch-up. Reuse metadata, not GPU roots.
    static SPARE_OWNERS: RefCell<[Vec<(RootId, MlxBuffer)>; 2]> =
        const { RefCell::new([Vec::new(), Vec::new()]) };
}

impl Default for CarryOwners {
    fn default() -> Self {
        let roots = SPARE_OWNERS.with(|cell| {
            let mut spare = cell.borrow_mut();
            let index = usize::from(spare[1].capacity() > spare[0].capacity());
            std::mem::take(&mut spare[index])
        });
        Self { roots }
    }
}

impl Drop for CarryOwners {
    fn drop(&mut self) {
        self.roots.clear();
        let _ = SPARE_OWNERS.try_with(|cell| {
            let mut spare = cell.borrow_mut();
            let index = usize::from(spare[1].capacity() < spare[0].capacity());
            if spare[index].capacity() < self.roots.capacity() {
                std::mem::swap(&mut spare[index], &mut self.roots);
            }
        });
    }
}

impl CarryOwners {
    pub(crate) fn include(&mut self, buffer: &MlxBuffer) {
        let id = identity(buffer);
        if !self.roots.iter().any(|(known, _)| *known == id) {
            self.roots.push((id, buffer.clone()));
        }
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.roots.iter().fold(0u64, |total, (_, root)| {
            total.saturating_add(root.byte_len() as u64)
        })
    }
}

fn bucket(bytes: u64) -> u64 {
    bytes.max(1).checked_next_power_of_two().unwrap_or(u64::MAX)
}

/// A complete asynchronous catch-up phase: all submitted commands keep their
/// operands until the checked drain. `new_pool_requested_bytes` assumes no
/// eligible reuse; prior retained pool capacity is a separate composition term.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CatchupPlan {
    pub(crate) parent_bytes: u64,
    /// Roots that cannot be step-pooled. Projection outputs are reserved in
    /// the bucket-rounded term below even when the direct route produces them.
    pub(crate) non_pool_bytes: u64,
    pub(crate) new_pool_requested_bytes: u64,
}

impl CatchupPlan {
    pub(crate) fn new(cfg: &Qwen35Config, rows: usize, dedicated: bool) -> Self {
        let rows = rows as u64;
        let hidden = rows.saturating_mul(cfg.hidden_size as u64).saturating_mul(4);
        let kv = rows.saturating_mul(cfg.num_key_value_heads as u64)
            .saturating_mul(cfg.head_dim as u64).saturating_mul(4);
        // Shared embedding (+ optional dedicated), shifted hidden, two norms,
        // 2H concat, projection, attention norm; K and V projections; positions;
        // three 8-byte RMS parameter roots plus a three-root 16-byte RoPE pack.
        // Native shared and optional dedicated gathers each retain a 4R token-ID
        // root through command drain; those roots overlap the projection chain.
        let chain = hidden.saturating_mul(8 + u64::from(dedicated))
            .saturating_add(kv.saturating_mul(2))
            .saturating_add(rows.saturating_mul(16))
            .saturating_add(rows.saturating_mul(4).saturating_mul(1 + u64::from(dedicated)))
            .saturating_add(24 + 48);
        let embedding = hidden.saturating_mul(2 + u64::from(dedicated))
            .saturating_add(rows.saturating_mul(4));
        Self {
            parent_bytes: hidden,
            non_pool_bytes: chain.max(embedding)
                .saturating_sub(hidden.saturating_add(kv.saturating_mul(2)))
                .saturating_sub(if rows == 1 { hidden } else { 0 }),
            // K per-head norm, its params, rotated K; projection output and
            // K/V output buckets. These replace their direct requested bytes,
            // not a second charge for the same projection roots.
            new_pool_requested_bytes: bucket(kv).saturating_mul(2).saturating_add(8)
                .saturating_add(bucket(hidden)).saturating_add(bucket(kv).saturating_mul(2))
                // The real shared gather pools its output only at one row.
                // Its bucket replaces the shared logical bytes removed above.
                .saturating_add(if rows == 1 { bucket(hidden) } else { 0 }),
        }
    }

    pub(crate) fn catchup_bytes(self) -> u64 {
        self.non_pool_bytes.saturating_add(self.new_pool_requested_bytes)
    }

    /// Complete target-forward versus catch-up lifetimes. `forward_without_parent`
    /// MUST exclude the durable normalized nextn parent. It is allocated inside
    /// the output head while forward owners/commands are still live, so it is
    /// charged on BOTH phases, not only on the catch-up arm.
    ///
    /// A caller whose forward bound already includes this parent must remove
    /// that one known parent charge before calling; do not feed an inclusive
    /// total here. Runtime pre-allocation callers use zero for this term.
    pub(crate) fn compose(
        self, forward_without_parent: u64, retained: u64, prior_pool_high_water: u64,
    ) -> u64 {
        retained.saturating_add(self.parent_bytes).saturating_add(
            forward_without_parent.max(
                self.catchup_bytes().saturating_add(prior_pool_high_water),
            ),
        )
    }
}

/// Active state and retained prefix can hold different full prefill parents.
/// Joint parents are bounded by the aggregate forward ceiling, NOT that ceiling
/// multiplied into each lane of one transaction. Spills/anchors have their own
/// byte quotas; their actual shared roots join the runtime inventory below.
/// Source cardinality: `Qwen35SlotWork` holds one active prefill/decode state
/// per slot; `Qwen35RetainedPrefix.spec` holds at most one prior boundary per
/// slot. Coherent/K3 entry clones alias the active root. Replacement's new root
/// is the separate prospective parent in `CatchupPlan`, not a third old cell.
/// N4 MTP uses the 2048 content ceiling: `cue_fold_rows` rejects MTP catch-up.
pub(crate) fn retained_bound(cfg: &Qwen35Config, rows: usize, slots: usize) -> u64 {
    (rows as u64).saturating_mul(cfg.hidden_size as u64).saturating_mul(4)
        .saturating_mul(slots as u64).saturating_mul(2)
}

/// RoPE retains one three-root 16-byte parameter pack per actual sequence length
/// (one fixed MTP K geometry). Every length 1..=rows can occur across requests.
pub(crate) fn rope_cache_bound(rows: usize) -> u64 {
    (rows as u64).saturating_mul(48)
}

struct WorkerOwners {
    owners: CarryOwners,
    retained_limit: u64,
    initial_retained: u64,
    phase_limit: u64,
}

thread_local! {
    static WORKER: RefCell<Option<WorkerOwners>> = const { RefCell::new(None) };
    static PHASE: RefCell<Option<CarryOwners>> = const { RefCell::new(None) };
    #[cfg(test)]
    static LAST_PHASE: RefCell<Option<(u64, u64, u64)>> = const { RefCell::new(None) };
    static CATCHUP_WINDOW: std::cell::Cell<Option<(u64, u64, u64)>> =
        const { std::cell::Cell::new(None) };
}

/// Installs the serving owner's census for one scheduler transaction. Clones
/// deliberately keep old roots charged until all commands of that transaction
/// have drained, even when a slot replaces its last-row view during settlement.
pub(crate) struct WorkerScope;
impl WorkerScope {
    pub(crate) fn open(owners: CarryOwners, retained_limit: u64, phase_limit: u64) -> Self {
        WORKER.with(|cell| {
            assert!(cell.borrow().is_none(), "nested carry worker scope");
            let initial_retained = owners.bytes();
            *cell.borrow_mut() = Some(WorkerOwners {
                owners, retained_limit, initial_retained, phase_limit,
            });
        });
        Self
    }
}
impl Drop for WorkerScope {
    fn drop(&mut self) {
        WORKER.with(|cell| { cell.borrow_mut().take(); });
    }
}

fn pool_retained_bytes() -> Result<u64> {
    super::decode_pool::worker_carry_pool_bytes()
        .map_err(|error| CarryMemoryError(format!("Qwen carry pool census unavailable: {error}")).into())
}

/// A prospective promise, made before target-parent/embedding allocation.
/// The load policy reserves the geometry; this checks the actual root owners
/// against that promise, including separately quota-bound spill/anchor owners.
pub(crate) fn commit_target(cfg: &Qwen35Config, rows: usize) -> Result<u64> {
    let plan = CatchupPlan::new(cfg, rows, cfg.mtp_use_dedicated_embeddings);
    if cfg.mtp_num_hidden_layers == 0 { return Ok(0); }
    let prior_pool = pool_retained_bytes()?;
    WORKER.with(|cell| {
        let worker = cell.borrow();
        let retained = worker.as_ref().map_or(0, |worker| worker.owners.bytes());
        if let Some(worker) = worker.as_ref() {
            // New transaction parents may have joined since the initial census;
            // the complete phase limit, not an old-owner-only cap, governs them.
            require_fit(worker.initial_retained, worker.retained_limit, "retained roots")?;
            let required = plan.compose(0, retained, prior_pool);
            // The prior pool is a separately committed owner in the complete
            // forward envelope. Add it to BOTH sides, not to a second ledger.
            require_fit(required, worker.phase_limit.saturating_add(prior_pool),
                "prospective target/catch-up")?;
        }
        let required = plan.compose(0, retained, prior_pool);
        tracing::trace!(
            carry_commitment_bytes = required, prior_pool_bytes = prior_pool,
            new_pool_requested_bytes = plan.new_pool_requested_bytes,
            "Qwen prospective carry/catch-up commitment before allocation"
        );
        Ok(required)
    })
}

/// The parent now exists. Join it with prior active/prefix/spill owners rather
/// than charging its row views again or prematurely releasing replaced roots.
pub(crate) fn commit_catchup(
    cfg: &Qwen35Config, rows: usize, target: &MlxBuffer, pending: Option<&MlxBuffer>,
) -> Result<()> {
    let plan = CatchupPlan::new(cfg, rows, cfg.mtp_use_dedicated_embeddings);
    WORKER.with(|cell| {
        if let Some(worker) = cell.borrow_mut().as_mut() {
            worker.owners.include(target);
            if let Some(pending) = pending { worker.owners.include(pending); }
            let required = worker.owners.bytes().saturating_add(plan.catchup_bytes());
            require_fit(worker.initial_retained, worker.retained_limit, "retained roots")?;
            require_fit(required, worker.phase_limit, "actual parent/catch-up")?;
        }
        Ok(())
    })
}

/// Local actual operands, including incoming old/new parent aliases. Kept until
/// the caller's final checked command drain, not merely until kernel submission.
pub(crate) struct CatchupScope {
    planned: u64,
    pool_before: u64,
    pool_growth_limit: u64,
    owned: bool,
}
impl CatchupScope {
    pub(crate) fn open(plan: CatchupPlan, roots: &[&MlxBuffer]) -> Result<Self> {
        if let Some((planned, pool_before, pool_growth_limit)) =
            CATCHUP_WINDOW.with(std::cell::Cell::get)
        {
            for root in roots { keep_operand(root); }
            return Ok(Self { planned, pool_before, pool_growth_limit, owned: false });
        }
        Self::open_owned(plan, roots, true)
    }

    fn open_owned(plan: CatchupPlan, roots: &[&MlxBuffer], shared_exists: bool) -> Result<Self> {
        let pool_before = pool_retained_bytes()?;
        let baseline = PHASE.with(|cell| {
            assert!(cell.borrow().is_none(), "nested MTP catch-up owner");
            let mut owners = CarryOwners::default();
            for root in roots { owners.include(root); }
            let baseline = owners.bytes();
            *cell.borrow_mut() = Some(owners);
            baseline
        });
        // A caller opening before the shared gather has not charged that
        // embedding root yet. Standalone callers supply an existing root.
        let planned = baseline.saturating_add(plan.catchup_bytes())
            .saturating_sub(if shared_exists { plan.parent_bytes } else { 0 });
        let pool_growth_limit = plan.new_pool_requested_bytes;
        CATCHUP_WINDOW.with(|cell| cell.set(Some((planned, pool_before, pool_growth_limit))));
        Ok(Self { planned, pool_before, pool_growth_limit, owned: true })
    }

    pub(crate) fn owns_phase(&self) -> bool { self.owned }

    pub(crate) fn finish(&self) -> Result<()> {
        let actual = PHASE.with(|cell| cell.borrow().as_ref().map_or(0, CarryOwners::bytes));
        require_fit(actual, self.planned, "actual catch-up roots")?;
        let pool_after = pool_retained_bytes()?;
        require_fit(
            pool_after.saturating_sub(self.pool_before), self.pool_growth_limit,
            "catch-up new pool roots",
        )?;
        // Pool snapshots are unions of free/issued/residency roots. Do not add
        // those labels or the tracked pooled operands to this high-water.
        tracing::trace!(
            carry_roots_bytes = actual,
            catchup_commitment_bytes = self.planned,
            pool_high_water_bytes = self.pool_before.max(pool_after),
            "Qwen MTP carry/catch-up checked owner boundary"
        );
        Ok(())
    }
}
impl Drop for CatchupScope {
    fn drop(&mut self) {
        if !self.owned { return; }
        CATCHUP_WINDOW.with(|cell| cell.set(None));
        PHASE.with(|cell| {
            let owners = cell.borrow_mut().take();
            #[cfg(test)]
            LAST_PHASE.with(|last| {
                *last.borrow_mut() = owners.as_ref().map(|owners| {
                    let token_ids = owners.roots.iter()
                        .filter(|(_, root)| root.dtype() == mlx_native::DType::U32)
                        .fold(0u64, |bytes, (_, root)| bytes.saturating_add(root.byte_len() as u64));
                    (owners.bytes(), self.planned, token_ids)
                });
            });
            let _ = (owners, self.planned);
        });
    }
}

/// The serving transaction starts BEFORE the shared embedding producer. It
/// owns both the census and the submitted-command list through the final drain.
/// process_target_batch joins this phase rather than replacing either scope.
pub(crate) struct CatchupTransaction {
    // Fields drop in declaration order: settle commands before releasing roots.
    _commands: super::encoder_stage::SubmittedCommands,
    carry: CatchupScope,
}

impl CatchupTransaction {
    pub(crate) fn open(
        cfg: &Qwen35Config, rows: usize, target: &MlxBuffer, pending: Option<&MlxBuffer>,
    ) -> Result<Self> {
        // Serving arrives after the target/verifier's checked drain. Its
        // normalized parents are direct roots, never issued pool scratch.
        // Reclaim old issued scratch BEFORE the shared gather, then enforce
        // the prior-pool bound even after an ordinary uncapped target route.
        if !super::decode_pool::default_pool() {
            return Err(CarryMemoryError("MTP catch-up requires the default bounded pool".into()).into());
        }
        let caps = super::prefill_prepared::prefill_pool_caps();
        super::decode_pool::cap_pool(Some(caps));
        let reset = super::decode_pool::checked_reset_decode_pool();
        // Keep caps through this phase and the next checked reset on every
        // exit path. Do not reset standalone process_target_batch inputs.
        super::decode_pool::end_caps_at_next_reset();
        reset?;
        require_fit(pool_retained_bytes()?, caps.retained as u64, "catch-up prior pool")?;
        commit_catchup(cfg, rows, target, pending)?;
        let plan = CatchupPlan::new(cfg, rows, cfg.mtp_use_dedicated_embeddings);
        let carry = CatchupScope::open_owned(
            plan, &[target, pending.unwrap_or(target)], false,
        )?;
        Ok(Self {
            _commands: super::encoder_stage::SubmittedCommands::open(true),
            carry,
        })
    }

    /// Keep normal submission asynchronous. A producer error must drain/check
    /// its already-submitted commands BEFORE the caller rolls back its slot.
    pub(crate) fn gather(
        &self, producer: impl FnOnce() -> Result<MlxBuffer>,
    ) -> Result<MlxBuffer> {
        match producer() {
            Ok(embedding) => {
                keep_operand(&embedding);
                Ok(embedding)
            }
            Err(error) => self.finish(Err(error)),
        }
    }

    /// Also used for an early host/supervision error after a successful gather.
    pub(crate) fn settle_error(&self, error: anyhow::Error) -> anyhow::Error {
        self.finish::<()>(Err(error)).expect_err("failed catch-up stays failed")
    }

    pub(crate) fn finish<T>(&self, result: Result<T>) -> Result<T> {
        let settled = super::encoder_stage::settle_submitted();
        match (result, settled) {
            (Err(host), Err(gpu)) => Err(gpu.context(format!(
                "catch-up had already failed on the host: {host:#}",
            ))),
            (_, Err(gpu)) => Err(gpu),
            (Err(host), Ok(())) => Err(host),
            (Ok(value), Ok(())) => {
                self.carry.finish()?;
                Ok(value)
            }
        }
    }
}

pub(crate) fn keep_operand(buffer: &MlxBuffer) {
    PHASE.with(|cell| {
        if let Some(owners) = cell.borrow_mut().as_mut() { owners.include(buffer); }
    });
}

#[cfg(test)]
pub(crate) fn last_phase() -> Option<(u64, u64)> {
    LAST_PHASE.with(|cell| (*cell.borrow()).map(|(actual, planned, _)| (actual, planned)))
}

#[cfg(test)]
pub(crate) fn last_phase_token_id_bytes() -> Option<u64> {
    LAST_PHASE.with(|cell| (*cell.borrow()).map(|(_, _, token_ids)| token_ids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::{DType, MlxDevice};

    #[test]
    fn carry_views_keep_full_parents_and_joined_aliases_until_last_owner() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Ok(device) = MlxDevice::new() else { return };
        let parent = device.alloc_buffer(4 * 8 * 32, DType::F32, vec![8, 32]).unwrap();
        let row = parent.slice_view(7 * 32 * 4, 32);
        let prefix = row.clone();
        let spill = parent.slice_view(0, 32);
        let bytes = parent.byte_len() as u64;
        drop(parent);
        let mut owners = CarryOwners::default();
        owners.include(&row);
        owners.include(&prefix);
        owners.include(&spill);
        assert_eq!(owners.bytes(), bytes, "three slots/owners share one padded parent");
        let replacement = device.alloc_buffer(4 * 3 * 32, DType::F32, vec![3, 32]).unwrap();
        owners.include(&replacement);
        assert_eq!(owners.bytes(), bytes + replacement.byte_len() as u64);
        drop(row);
        drop(prefix);
        drop(owners);
        let mut remaining = CarryOwners::default();
        remaining.include(&spill);
        remaining.include(&replacement);
        assert_eq!(remaining.bytes(), bytes + replacement.byte_len() as u64);
        drop(spill);
        drop(remaining);
        let mut final_owner = CarryOwners::default();
        final_owner.include(&replacement);
        assert_eq!(final_owner.bytes(), replacement.byte_len() as u64);
    }
}
