//! Multi-model in-memory pool for ADR-005 Phase 4 hot-swap (iter-206 W74).
//!
//! This module is the **pure data primitive** that Phase 4's hot-swap
//! orchestrator (iter-208) and AppState integration (iter-209) compose
//! against.  It contains *no* engine load, *no* GPU code, *no* `AppState`
//! wiring — those live downstream.  The pattern mirrors iter-201
//! `serve::quant_select`: a synthetic-fixture-tested data structure that
//! later iters glue into the live serve path.
//!
//! # What this module does
//!
//! - Tracks N currently-loaded model handles (handle metadata only).
//! - Enforces two simultaneous bounds:
//!   1. **Capacity:** at most `capacity_models` distinct repos resident.
//!   2. **Memory budget:** sum of `bytes_resident` across the pool must
//!      stay `<= memory_budget_bytes`.
//! - LRU eviction order: when an `insert` would breach either bound, the
//!   pool drops the least-recently-used handle(s) until both bounds hold.
//! - `touch(repo)` promotes an existing entry to MRU without changing
//!   resident bytes — used by handler entry points before they read
//!   from the engine.
//! - `from_hardware(&HardwareProfile, n)` adapter applies the spec'd
//!   80% of total unified memory ceiling (line 929 of ADR-005).
//!
//! # What this module does NOT do
//!
//! - Hold the actual `Engine` (or any `mlx_native` buffers) — `LoadedHandle`
//!   is a pure descriptor.  iter-208's `HotSwapManager` will own
//!   `Arc<Engine>` and use this pool for the eviction-policy decision
//!   only.
//! - Reach the disk or GGUF reader — entry construction is by-value;
//!   the orchestrator builds handles after the engine load succeeds.
//! - Touch `AppState`, `cmd_serve`, or any handler — wiring is iter-209.
//! - Mutate the on-disk cache manifest — `cache::touch(repo)` is the
//!   parallel on-disk LRU stamp; the in-memory pool's LRU is
//!   independent (different lifetime, different policy).
//!
//! # Eviction algorithm
//!
//! `insert(handle)` runs in two passes:
//!
//! 1. **Capacity pass:** if `len() == capacity_models` and `repo` is not
//!    already in the pool, evict the LRU entry once.
//! 2. **Budget pass:** while `total_resident_bytes + handle.bytes >
//!    memory_budget_bytes`, evict the LRU entry.  Stop when either
//!    the bound is satisfied or the pool is empty.  If the pool is
//!    empty and `handle.bytes > memory_budget_bytes`, the insert is
//!    refused with `PoolError::OversizedHandle`.
//!
//! Re-inserting an already-present `repo_id` is treated as an in-place
//! update: the existing handle's `bytes_resident` and `loaded_at`
//! refresh and the entry is promoted to MRU; no eviction runs unless
//! the new bytes push the total over budget (in which case OTHER
//! entries are evicted, not the re-inserted one — that would be a
//! self-eviction defect; the re-insert intent is "this model just
//! re-loaded with different bytes / weights").
//!
//! # Why a hand-rolled LRU and not a crate
//!
//! `Cargo.toml` is fenced this session (ADR-014 P7 actively bumping
//! deps), so adding `indexmap` or `lru` is not allowed.  The stdlib
//! `HashMap<String, LoadedHandle>` plus `Vec<String>` order list
//! delivers identical semantics for `N <= 8` (Phase 4 default N=3,
//! configurable) at O(N) per op — fine for a pool of ~3 entries
//! where every op is a request-rate event.
//!
//! # Tests
//!
//! Synthetic-fixture unit tests cover:
//!
//! - empty pool + capacity-1 overflow
//! - capacity-3 LRU eviction order
//! - memory-budget eviction without capacity overflow
//! - combined eviction (capacity + budget in the same `insert`)
//! - `touch` promotes MRU; `get` is read-only (does NOT touch)
//! - `remove` returns the dropped handle + updates `total_resident_bytes`
//! - idempotent insert (same `repo_id`) updates bytes + promotes
//! - zero-budget refuses any non-zero handle
//! - oversized single handle refuses (handle.bytes > budget)
//! - `from_hardware` produces the 80% ceiling
//! - `len`, `is_empty`, `iter` semantics
//!
//! # ADR-005 Phase 4 lineage
//!
//! - Lines 928–936: narrative spec.
//! - AC 5357 (line 5357): "Cached pool holds up to 3 loaded models with
//!   LRU eviction bounded by 80% of system unified memory (configurable)".
//!   This module is the foundation primitive that AC closes against.
//! - Plan: line 5354 (Phase 4 audit + iter-by-iter plan, iter 206 row).

use std::collections::HashMap;
use std::time::SystemTime;

use crate::intelligence::hardware::HardwareProfile;

/// Default pool capacity per ADR-005 Phase 4 narrative (line 929).
/// Configurable via [`LoadedPool::with_capacity_and_budget`].
pub const DEFAULT_POOL_CAPACITY: usize = 3;

/// Default memory budget fraction of total unified memory per
/// ADR-005 Phase 4 narrative (line 929: "memory ceiling of **80% of
/// system unified memory**").  Used by [`LoadedPool::from_hardware`].
pub const DEFAULT_MEMORY_BUDGET_FRACTION: f64 = 0.80;

/// Descriptor for one currently-loaded model handle.
///
/// Pure metadata — does NOT hold the actual `Engine` or any GPU
/// buffers.  The hot-swap orchestrator (iter-208) maintains a
/// parallel `HashMap<String, Arc<Engine>>` keyed by `repo_id` and
/// uses this pool's eviction decisions to drop entries from that map.
#[derive(Debug, Clone)]
pub struct LoadedHandle {
    /// HuggingFace repo id (`org/repo`), the canonical pool key.
    pub repo_id: String,
    /// Quantization the loaded weights were materialized at.
    /// Carried for observability + `/v1/models` extension fields.
    pub quant: String,
    /// Wall-clock when the handle was loaded.  Used for diagnostics
    /// (e.g. `Server-Timing: model_load=NNNms` headers); does NOT
    /// drive eviction — eviction reads the pool's internal MRU
    /// order list, which is updated on every `insert` and `touch`.
    pub loaded_at: SystemTime,
    /// On-GPU resident-bytes attributed to this handle.  Sum of all
    /// model weights + KV cache region + any pinned scratch.  The
    /// pool sums these to enforce `memory_budget_bytes`.
    pub bytes_resident: u64,
}

impl LoadedHandle {
    /// Construct a handle with `loaded_at = SystemTime::now()`.
    /// Tests use this; the orchestrator sets `loaded_at` from the
    /// engine-load completion timestamp.
    pub fn new(repo_id: impl Into<String>, quant: impl Into<String>, bytes_resident: u64) -> Self {
        Self {
            repo_id: repo_id.into(),
            quant: quant.into(),
            loaded_at: SystemTime::now(),
            bytes_resident,
        }
    }
}

/// Errors the pool can return from `insert`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// `handle.bytes_resident > memory_budget_bytes`: the pool can
    /// never accommodate this handle even when empty.  Caller must
    /// reject the load attempt with a clear operator-facing message.
    OversizedHandle {
        repo_id: String,
        handle_bytes: u64,
        budget_bytes: u64,
    },
    /// `capacity_models == 0`: the pool was constructed disabled.
    /// Caller is expected to refuse the insert with a config-error
    /// message; this exists so a misconfigured deployment fails
    /// loudly rather than silently dropping every load.
    ZeroCapacity,
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OversizedHandle {
                repo_id,
                handle_bytes,
                budget_bytes,
            } => write!(
                f,
                "model {repo_id} resident bytes {handle_bytes} exceed pool memory budget \
                 {budget_bytes}; cannot load (consider raising the budget via the \
                 Phase 4 hot-swap config knob, or pick a smaller quant)"
            ),
            Self::ZeroCapacity => write!(
                f,
                "multi-model pool is configured with capacity_models = 0; refusing \
                 every load.  Either raise capacity (default 3) or remove the pool \
                 entirely from the deployment config."
            ),
        }
    }
}

impl std::error::Error for PoolError {}

/// Bounded in-memory LRU pool of [`LoadedHandle`] descriptors.
///
/// See module docs for the eviction algorithm and what this type does
/// (and explicitly does NOT) do.
#[derive(Debug)]
pub struct LoadedPool {
    /// Maximum number of distinct repos resident.  See module docs.
    capacity_models: usize,
    /// Maximum sum of `bytes_resident` across the pool.  See module docs.
    memory_budget_bytes: u64,
    /// Repo → handle lookup.  O(1) reads.
    entries: HashMap<String, LoadedHandle>,
    /// LRU ordering.  Index 0 is the LRU; the last index is the MRU.
    /// `lru_order.len() == entries.len()` is an invariant.
    lru_order: Vec<String>,
    /// Cumulative `bytes_resident` across `entries`.  Maintained
    /// incrementally so capacity checks are O(1).
    total_resident_bytes: u64,
}

impl LoadedPool {
    /// Construct a pool with explicit capacity and memory budget.
    /// `capacity_models == 0` produces a pool that refuses every
    /// insert with [`PoolError::ZeroCapacity`] — the constructor
    /// allows it so misconfigured deployments fail at insert-time
    /// rather than at construct-time (where the error has less
    /// context).
    pub fn with_capacity_and_budget(capacity_models: usize, memory_budget_bytes: u64) -> Self {
        Self {
            capacity_models,
            memory_budget_bytes,
            entries: HashMap::with_capacity(capacity_models.max(1)),
            lru_order: Vec::with_capacity(capacity_models.max(1)),
            total_resident_bytes: 0,
        }
    }

    /// Construct a pool with [`DEFAULT_POOL_CAPACITY`] and the
    /// 80%-of-total-unified-memory budget per ADR-005 line 929.
    ///
    /// Reads `HardwareProfile::total_memory_bytes` (the pool reserves
    /// against the *physical* unified memory, not the
    /// available-now-bytes — an in-flight model that hasn't released
    /// memory yet should still be allowed to fit, the pool's own
    /// accounting determines that).
    pub fn from_hardware(hw: &HardwareProfile) -> Self {
        Self::from_hardware_with(hw, DEFAULT_POOL_CAPACITY, DEFAULT_MEMORY_BUDGET_FRACTION)
    }

    /// Same as [`Self::from_hardware`] with explicit capacity and
    /// fraction overrides.  `fraction` outside `(0.0, 1.0]` is clamped
    /// to that range — passing 1.5 yields full physical memory; passing
    /// 0.0 yields a zero-budget pool that refuses everything (matches
    /// `from_hardware_with(_, 0, _)` semantically).
    pub fn from_hardware_with(
        hw: &HardwareProfile,
        capacity_models: usize,
        fraction: f64,
    ) -> Self {
        let f = fraction.clamp(0.0, 1.0);
        // f64 → u64 with floor; never panics for finite f.
        let budget = ((hw.total_memory_bytes as f64) * f).floor() as u64;
        Self::with_capacity_and_budget(capacity_models, budget)
    }

    /// Capacity (number of distinct loaded models permitted).
    pub fn capacity_models(&self) -> usize {
        self.capacity_models
    }

    /// Memory budget in bytes.
    pub fn memory_budget_bytes(&self) -> u64 {
        self.memory_budget_bytes
    }

    /// Cumulative resident bytes across loaded handles.
    pub fn total_resident_bytes(&self) -> u64 {
        self.total_resident_bytes
    }

    /// Number of currently-resident handles.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if the pool has zero resident handles.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Read-only borrow of a handle by `repo_id`.  Does NOT touch the
    /// LRU order — call [`Self::touch`] explicitly if the caller is
    /// servicing a request and wants to promote.  Separating get
    /// from touch lets diagnostic / metrics endpoints peek without
    /// poisoning the eviction policy.
    pub fn get(&self, repo_id: &str) -> Option<&LoadedHandle> {
        self.entries.get(repo_id)
    }

    /// Promote `repo_id` to MRU.  Returns `true` if the entry existed
    /// and was promoted; `false` if it wasn't in the pool (no-op).
    pub fn touch(&mut self, repo_id: &str) -> bool {
        if !self.entries.contains_key(repo_id) {
            return false;
        }
        // Move the entry to the back of `lru_order` (MRU end).
        if let Some(pos) = self.lru_order.iter().position(|r| r == repo_id) {
            let key = self.lru_order.remove(pos);
            self.lru_order.push(key);
        }
        true
    }

    /// Iterate handles in LRU → MRU order (pool internals' canonical
    /// order).  Useful for `/v1/models` listing and pool diagnostics.
    pub fn iter(&self) -> impl Iterator<Item = &LoadedHandle> {
        self.lru_order
            .iter()
            .filter_map(move |k| self.entries.get(k))
    }

    /// Drop a handle by `repo_id`.  Returns the dropped handle if
    /// present, `None` otherwise.  Updates `total_resident_bytes`.
    /// Does NOT free GPU buffers — that's the orchestrator's
    /// responsibility (drop the parallel `Arc<Engine>`).
    pub fn remove(&mut self, repo_id: &str) -> Option<LoadedHandle> {
        let handle = self.entries.remove(repo_id)?;
        if let Some(pos) = self.lru_order.iter().position(|r| r == repo_id) {
            self.lru_order.remove(pos);
        }
        self.total_resident_bytes = self.total_resident_bytes.saturating_sub(handle.bytes_resident);
        Some(handle)
    }

    /// Insert a handle into the pool, evicting LRU entries as needed.
    ///
    /// Returns the list of evicted handles (LRU-first order — the
    /// first element is the oldest evictee).  An empty `Vec` means
    /// no eviction was needed.
    ///
    /// Re-inserting an existing `repo_id` updates the handle's
    /// `bytes_resident` + `loaded_at` and promotes the entry to MRU.
    /// In that case the `total_resident_bytes` delta may still drive
    /// budget eviction of *other* entries — but the re-inserted
    /// handle itself is never evicted from the same `insert` call.
    ///
    /// # Errors
    ///
    /// - [`PoolError::ZeroCapacity`] if `capacity_models == 0`.
    /// - [`PoolError::OversizedHandle`] if `handle.bytes_resident >
    ///   memory_budget_bytes` and the pool would have to be empty
    ///   to fit it (i.e. impossible to satisfy under any LRU
    ///   eviction sequence).
    pub fn insert(&mut self, handle: LoadedHandle) -> Result<Vec<LoadedHandle>, PoolError> {
        if self.capacity_models == 0 {
            return Err(PoolError::ZeroCapacity);
        }
        if handle.bytes_resident > self.memory_budget_bytes {
            return Err(PoolError::OversizedHandle {
                repo_id: handle.repo_id,
                handle_bytes: handle.bytes_resident,
                budget_bytes: self.memory_budget_bytes,
            });
        }

        let mut evicted: Vec<LoadedHandle> = Vec::new();

        // Re-insert path: update bytes + promote to MRU.  Eviction of
        // OTHER entries may still happen in the budget pass below.
        if let Some(existing) = self.entries.get_mut(&handle.repo_id) {
            // Drop the old contribution first to avoid an underflow in
            // `total_resident_bytes` if the new bytes are smaller.
            self.total_resident_bytes = self
                .total_resident_bytes
                .saturating_sub(existing.bytes_resident);
            existing.bytes_resident = handle.bytes_resident;
            existing.loaded_at = handle.loaded_at;
            existing.quant = handle.quant.clone();
            self.total_resident_bytes = self
                .total_resident_bytes
                .saturating_add(handle.bytes_resident);
            // Promote: move to MRU end in `lru_order`.
            if let Some(pos) = self.lru_order.iter().position(|r| r == &handle.repo_id) {
                let key = self.lru_order.remove(pos);
                self.lru_order.push(key);
            }
            // Budget pass: evict OTHERS until the bound holds.  The
            // re-inserted entry is now at the MRU end (last index)
            // so the LRU-first eviction loop will not touch it
            // unless it is the only entry (in which case
            // `total_resident_bytes <= memory_budget_bytes` already
            // by the OversizedHandle precheck).
            self.evict_until_within_budget(&mut evicted, &handle.repo_id);
            return Ok(evicted);
        }

        // Capacity pass: if at capacity, evict the LRU entry exactly
        // once.  (We only evict ONE here; the budget pass below may
        // evict more.)
        if self.entries.len() >= self.capacity_models {
            if let Some(victim_key) = self.lru_order.first().cloned() {
                if let Some(victim) = self.entries.remove(&victim_key) {
                    self.total_resident_bytes = self
                        .total_resident_bytes
                        .saturating_sub(victim.bytes_resident);
                    self.lru_order.remove(0);
                    evicted.push(victim);
                }
            }
        }

        // Budget pass (pre-insert).  Evict LRU until adding the new
        // handle would fit.  We hold `handle` outside the pool here
        // so the loop only touches existing entries.
        while !self.lru_order.is_empty()
            && self
                .total_resident_bytes
                .saturating_add(handle.bytes_resident)
                > self.memory_budget_bytes
        {
            let victim_key = self.lru_order.remove(0);
            if let Some(victim) = self.entries.remove(&victim_key) {
                self.total_resident_bytes = self
                    .total_resident_bytes
                    .saturating_sub(victim.bytes_resident);
                evicted.push(victim);
            }
        }

        // The OversizedHandle precheck guarantees the post-eviction
        // budget holds — `handle.bytes_resident <= memory_budget_bytes`
        // and we evicted until `total_resident_bytes + handle.bytes
        // <= memory_budget_bytes` OR the pool is empty.  In the
        // pool-empty case, `0 + handle.bytes <= memory_budget_bytes`
        // holds by the precheck.

        // Insert + push to MRU end.
        self.total_resident_bytes = self
            .total_resident_bytes
            .saturating_add(handle.bytes_resident);
        self.lru_order.push(handle.repo_id.clone());
        self.entries.insert(handle.repo_id.clone(), handle);

        Ok(evicted)
    }

    /// Helper used by the re-insert path: evict LRU entries until
    /// `total_resident_bytes <= memory_budget_bytes`, skipping the
    /// just-promoted entry (`spare_repo_id`).
    fn evict_until_within_budget(
        &mut self,
        evicted: &mut Vec<LoadedHandle>,
        spare_repo_id: &str,
    ) {
        while self.total_resident_bytes > self.memory_budget_bytes {
            // Find the LRU entry that is NOT the spare.
            let victim_idx = self
                .lru_order
                .iter()
                .position(|k| k != spare_repo_id);
            let Some(idx) = victim_idx else {
                // Only the spare remains; the precheck guaranteed
                // `spare.bytes <= budget`, so the loop terminates here.
                break;
            };
            let victim_key = self.lru_order.remove(idx);
            if let Some(victim) = self.entries.remove(&victim_key) {
                self.total_resident_bytes = self
                    .total_resident_bytes
                    .saturating_sub(victim.bytes_resident);
                evicted.push(victim);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests — synthetic fixtures, no engine load, no GPU.
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn h(repo: &str, bytes: u64) -> LoadedHandle {
        LoadedHandle::new(repo, "Q4_K_M", bytes)
    }

    // --- Construction --------------------------------------------------

    #[test]
    fn empty_pool_is_empty() {
        let p = LoadedPool::with_capacity_and_budget(3, 1_000);
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert_eq!(p.total_resident_bytes(), 0);
        assert_eq!(p.capacity_models(), 3);
        assert_eq!(p.memory_budget_bytes(), 1_000);
        assert!(p.iter().next().is_none());
    }

    /// Synthetic [`HardwareProfile`] for the from-hardware tests.
    /// `HardwareProfile` does NOT derive `Default` (the bandwidth
    /// field is computed from the chip model elsewhere), so tests
    /// construct one fixture-style here.
    fn synthetic_hw(total_memory_bytes: u64) -> HardwareProfile {
        HardwareProfile {
            chip_model: "Synthetic Test Chip".into(),
            total_memory_bytes,
            available_memory_bytes: total_memory_bytes,
            performance_cores: 8,
            efficiency_cores: 4,
            total_cores: 12,
            memory_bandwidth_gbs: 400.0,
        }
    }

    #[test]
    fn from_hardware_applies_eighty_percent_default() {
        let hw = synthetic_hw(128 * 1024 * 1024 * 1024); // 128 GiB
        let p = LoadedPool::from_hardware(&hw);
        assert_eq!(p.capacity_models(), DEFAULT_POOL_CAPACITY);
        // 0.80 × 128 GiB = 102.4 GiB exact (fp64 has the precision)
        let expected = ((128.0_f64 * 1024.0 * 1024.0 * 1024.0) * 0.80).floor() as u64;
        assert_eq!(p.memory_budget_bytes(), expected);
    }

    #[test]
    fn from_hardware_with_clamps_fraction() {
        let hw = synthetic_hw(1_000_000);
        // 1.5 → clamps to 1.0 → full physical
        let p_high = LoadedPool::from_hardware_with(&hw, 3, 1.5);
        assert_eq!(p_high.memory_budget_bytes(), 1_000_000);
        // -0.1 → clamps to 0.0 → zero budget
        let p_low = LoadedPool::from_hardware_with(&hw, 3, -0.1);
        assert_eq!(p_low.memory_budget_bytes(), 0);
    }

    // --- Insert: capacity-driven eviction ------------------------------

    #[test]
    fn capacity_one_evicts_on_second_insert() {
        let mut p = LoadedPool::with_capacity_and_budget(1, 1_000_000);
        let evicted_a = p.insert(h("a/1", 100)).unwrap();
        assert!(evicted_a.is_empty());
        let evicted_b = p.insert(h("b/2", 200)).unwrap();
        assert_eq!(evicted_b.len(), 1);
        assert_eq!(evicted_b[0].repo_id, "a/1");
        assert_eq!(p.len(), 1);
        assert!(p.get("a/1").is_none());
        assert!(p.get("b/2").is_some());
        assert_eq!(p.total_resident_bytes(), 200);
    }

    #[test]
    fn capacity_three_evicts_lru_first() {
        let mut p = LoadedPool::with_capacity_and_budget(3, 1_000_000_000);
        let _ = p.insert(h("a/1", 100)).unwrap(); // LRU
        let _ = p.insert(h("b/2", 200)).unwrap();
        let _ = p.insert(h("c/3", 300)).unwrap(); // MRU
        assert_eq!(p.len(), 3);
        // 4th insert evicts a/1 (LRU)
        let evicted = p.insert(h("d/4", 400)).unwrap();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].repo_id, "a/1");
        // pool order is now b, c, d (LRU → MRU)
        let order: Vec<&str> = p.iter().map(|h| h.repo_id.as_str()).collect();
        assert_eq!(order, vec!["b/2", "c/3", "d/4"]);
        assert_eq!(p.total_resident_bytes(), 200 + 300 + 400);
    }

    // --- Insert: budget-driven eviction --------------------------------

    #[test]
    fn budget_evicts_without_capacity_overflow() {
        // Capacity 5, budget 1_000.  Three handles totalling 900 fit;
        // 4th handle of 200 forces budget eviction even though capacity
        // is not breached.
        let mut p = LoadedPool::with_capacity_and_budget(5, 1_000);
        let _ = p.insert(h("a/1", 300)).unwrap(); // LRU, 300
        let _ = p.insert(h("b/2", 300)).unwrap(); // total 600
        let _ = p.insert(h("c/3", 300)).unwrap(); // total 900, MRU
        assert_eq!(p.total_resident_bytes(), 900);
        // 4th handle 200: 900+200=1100 > 1000 → evict LRU until fit.
        // Evict a/1 (300) → total 800, +200 = 1000 ≤ 1000. Stop.
        let evicted = p.insert(h("d/4", 200)).unwrap();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].repo_id, "a/1");
        assert_eq!(p.len(), 3);
        assert_eq!(p.total_resident_bytes(), 800);
    }

    #[test]
    fn budget_eviction_chains_until_fit() {
        // Three small handles + one big new handle that requires
        // evicting all three.
        let mut p = LoadedPool::with_capacity_and_budget(5, 1_000);
        let _ = p.insert(h("a/1", 300)).unwrap();
        let _ = p.insert(h("b/2", 300)).unwrap();
        let _ = p.insert(h("c/3", 300)).unwrap();
        let evicted = p.insert(h("big/1", 950)).unwrap();
        assert_eq!(evicted.len(), 3);
        assert_eq!(
            evicted.iter().map(|h| h.repo_id.as_str()).collect::<Vec<_>>(),
            vec!["a/1", "b/2", "c/3"]
        );
        assert_eq!(p.len(), 1);
        assert_eq!(p.total_resident_bytes(), 950);
    }

    #[test]
    fn capacity_and_budget_evict_in_one_insert() {
        // Capacity 2, budget 800.  Insert two handles at 400 each;
        // third at 500 trips capacity AND budget — both passes run
        // and the eviction list reflects both.
        let mut p = LoadedPool::with_capacity_and_budget(2, 800);
        let _ = p.insert(h("a/1", 400)).unwrap();
        let _ = p.insert(h("b/2", 400)).unwrap(); // total 800, capacity 2/2
        // Third: capacity pass evicts a/1 → total 400, len 1.  Budget
        // pass: 400+500=900 > 800 → evict b/2 → total 0.  +500 = 500.
        let evicted = p.insert(h("c/3", 500)).unwrap();
        assert_eq!(evicted.len(), 2);
        let names: Vec<&str> = evicted.iter().map(|h| h.repo_id.as_str()).collect();
        assert_eq!(names, vec!["a/1", "b/2"]);
        assert_eq!(p.len(), 1);
        assert_eq!(p.total_resident_bytes(), 500);
    }

    // --- touch / get separation ----------------------------------------

    #[test]
    fn touch_promotes_to_mru() {
        let mut p = LoadedPool::with_capacity_and_budget(3, 1_000_000);
        let _ = p.insert(h("a/1", 100)).unwrap(); // LRU
        let _ = p.insert(h("b/2", 200)).unwrap();
        let _ = p.insert(h("c/3", 300)).unwrap(); // MRU
        assert!(p.touch("a/1"));
        // Order should now be b, c, a (LRU → MRU).
        let order: Vec<&str> = p.iter().map(|h| h.repo_id.as_str()).collect();
        assert_eq!(order, vec!["b/2", "c/3", "a/1"]);
        // 4th insert evicts b/2 (the new LRU), not a/1.
        let evicted = p.insert(h("d/4", 400)).unwrap();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].repo_id, "b/2");
    }

    #[test]
    fn get_does_not_touch() {
        let mut p = LoadedPool::with_capacity_and_budget(3, 1_000_000);
        let _ = p.insert(h("a/1", 100)).unwrap(); // LRU
        let _ = p.insert(h("b/2", 200)).unwrap();
        let _ = p.insert(h("c/3", 300)).unwrap();
        let _ = p.get("a/1").unwrap();
        // a/1 must still be LRU — get is read-only.
        let evicted = p.insert(h("d/4", 400)).unwrap();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].repo_id, "a/1");
    }

    #[test]
    fn touch_unknown_is_noop() {
        let mut p = LoadedPool::with_capacity_and_budget(3, 1_000);
        assert!(!p.touch("nope/0"));
    }

    // --- remove --------------------------------------------------------

    #[test]
    fn remove_returns_handle_and_updates_bytes() {
        let mut p = LoadedPool::with_capacity_and_budget(3, 1_000);
        let _ = p.insert(h("a/1", 100)).unwrap();
        let _ = p.insert(h("b/2", 200)).unwrap();
        assert_eq!(p.total_resident_bytes(), 300);
        let removed = p.remove("a/1").unwrap();
        assert_eq!(removed.repo_id, "a/1");
        assert_eq!(removed.bytes_resident, 100);
        assert_eq!(p.total_resident_bytes(), 200);
        assert_eq!(p.len(), 1);
        // Removing again is a no-op.
        assert!(p.remove("a/1").is_none());
    }

    // --- Idempotent insert (re-load) -----------------------------------

    #[test]
    fn reinsert_same_repo_updates_bytes_and_promotes() {
        let mut p = LoadedPool::with_capacity_and_budget(3, 10_000);
        let _ = p.insert(h("a/1", 100)).unwrap(); // LRU
        let _ = p.insert(h("b/2", 200)).unwrap();
        let _ = p.insert(h("c/3", 300)).unwrap(); // MRU
        // Re-insert a/1 with new bytes 1500: it should NOT be evicted,
        // total goes 100→1500 = +1400, no other eviction (well under
        // budget).  a/1 promotes to MRU.
        let evicted = p.insert(h("a/1", 1500)).unwrap();
        assert!(evicted.is_empty(), "re-insert must not self-evict");
        let order: Vec<&str> = p.iter().map(|h| h.repo_id.as_str()).collect();
        assert_eq!(order, vec!["b/2", "c/3", "a/1"]);
        assert_eq!(p.total_resident_bytes(), 1500 + 200 + 300);
        assert_eq!(p.get("a/1").unwrap().bytes_resident, 1500);
    }

    #[test]
    fn reinsert_with_budget_overflow_evicts_others_not_self() {
        // Budget 1000.  a:300, b:300, c:300 = 900.  Re-insert b:800
        // would push total to 1400 → evict OTHERS (a is LRU because b
        // promoted; then c) but never b.
        let mut p = LoadedPool::with_capacity_and_budget(5, 1_000);
        let _ = p.insert(h("a/1", 300)).unwrap();
        let _ = p.insert(h("b/2", 300)).unwrap();
        let _ = p.insert(h("c/3", 300)).unwrap();
        // Re-insert b with bigger bytes.  After the bytes update the
        // total is 300+800+300 = 1400; we evict LRU (a) → 1100, still
        // > 1000, evict next non-spare LRU (c) → 800.  Stop.
        let evicted = p.insert(h("b/2", 800)).unwrap();
        let evicted_names: Vec<&str> = evicted.iter().map(|h| h.repo_id.as_str()).collect();
        assert_eq!(evicted_names, vec!["a/1", "c/3"]);
        assert!(p.get("b/2").is_some(), "self never evicted on re-insert");
        assert_eq!(p.len(), 1);
        assert_eq!(p.total_resident_bytes(), 800);
    }

    // --- Refusal cases -------------------------------------------------

    #[test]
    fn zero_capacity_refuses_every_insert() {
        let mut p = LoadedPool::with_capacity_and_budget(0, 1_000);
        let err = p.insert(h("a/1", 100)).unwrap_err();
        assert_eq!(err, PoolError::ZeroCapacity);
        let msg = format!("{err}");
        assert!(msg.contains("capacity_models = 0"), "msg = {msg}");
    }

    #[test]
    fn oversized_handle_refused() {
        let mut p = LoadedPool::with_capacity_and_budget(3, 1_000);
        let err = p.insert(h("big/1", 1_001)).unwrap_err();
        match err {
            PoolError::OversizedHandle {
                repo_id,
                handle_bytes,
                budget_bytes,
            } => {
                assert_eq!(repo_id, "big/1");
                assert_eq!(handle_bytes, 1_001);
                assert_eq!(budget_bytes, 1_000);
            }
            other => panic!("unexpected error: {other:?}"),
        }
        // Pool is unchanged.
        assert!(p.is_empty());
    }

    #[test]
    fn zero_budget_refuses_any_nonzero_handle() {
        let mut p = LoadedPool::with_capacity_and_budget(3, 0);
        let err = p.insert(h("a/1", 1)).unwrap_err();
        assert!(matches!(err, PoolError::OversizedHandle { .. }));
        // A zero-byte handle would technically fit a zero budget, but
        // a real handle is never zero bytes — the test below documents
        // that we do not gratuitously refuse it.
        let evicted = p.insert(h("zerobyte/0", 0)).unwrap();
        assert!(evicted.is_empty());
        assert_eq!(p.len(), 1);
    }

    // --- Iter order ----------------------------------------------------

    #[test]
    fn iter_yields_lru_to_mru() {
        let mut p = LoadedPool::with_capacity_and_budget(3, 10_000);
        let _ = p.insert(h("a/1", 100)).unwrap();
        let _ = p.insert(h("b/2", 200)).unwrap();
        let _ = p.insert(h("c/3", 300)).unwrap();
        let order: Vec<&str> = p.iter().map(|h| h.repo_id.as_str()).collect();
        assert_eq!(order, vec!["a/1", "b/2", "c/3"]);
    }

    // --- Display + error formatting ------------------------------------

    #[test]
    fn oversized_error_message_names_budget_and_repo() {
        let err = PoolError::OversizedHandle {
            repo_id: "huge/model".into(),
            handle_bytes: 9_000,
            budget_bytes: 1_000,
        };
        let msg = format!("{err}");
        assert!(msg.contains("huge/model"));
        assert!(msg.contains("9000"));
        assert!(msg.contains("1000"));
        assert!(msg.contains("Phase 4"));
    }
}
