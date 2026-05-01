//! ADR-017 §C.1 — `KvPersistRegistry`: per-`(repo, quant)` table of
//! [`crate::serve::kv_persist::EngineBindable`] hooks.
//!
//! ## Why this exists (Chesterton's fence)
//!
//! Phase A.3's [`crate::serve::kv_persist::BlockPrefixCacheSpiller`]
//! holds a parallel registration table keyed on `(repo, quant)` for
//! [`crate::serve::kv_persist::KvCacheSpill`] hooks. The spiller's
//! `post_admit` trigger fires from inside `HotSwapManager::load_or_get`
//! AFTER the engine load succeeds but BEFORE the manager publishes the
//! Arc into its engines map (see `multi_model.rs:687-728` for the
//! [`crate::serve::multi_model::KvSpiller`] trait surface). At that
//! trigger site the engine ref is held only locally — it is NOT
//! threaded through to the per-family hook (the hook gets `repo +
//! quant` only, see `spiller.rs:398-477`). So a per-family hook that
//! needs to mutate engine-side state (e.g. Gemma 4 dense K/V buffers
//! living inside `MlxModelWeights`) must have its engine reference
//! ALREADY WIRED at the moment `post_admit` fires.
//!
//! `KvPersistRegistry` is the only natural seam: it carries one entry
//! per `(repo, quant)` registered at startup, and Phase C.1's
//! [`crate::serve::kv_persist::loader_wrapper::LoaderWrapper`] calls
//! `bind_for(...)` from inside the load path BEFORE the spiller's
//! `post_admit` runs — closing the gap without modifying the
//! `KvSpiller` trait surface (which is stable per the iter-212 ship
//! contract).
//!
//! ## What this registry does NOT do
//!
//! - It does NOT own the per-family `KvCacheSpill` hook. That stays
//!   inside `BlockPrefixCacheSpiller::registrations`. The registry
//!   here holds only an [`crate::serve::kv_persist::EngineBindable`]
//!   view on the same hook (or a stateless stub when no engine
//!   binding is needed).
//! - It does NOT enforce that `bind_for` precedes `post_admit`. The
//!   ordering is the caller's responsibility (in production, the
//!   `LoaderWrapper` enforces it; tests assert the order via mocks).
//! - It does NOT panic on type mismatch in the underlying
//!   `Arc<dyn Any>` downcast. The hook's `bind_engine` impl is
//!   responsible for downcast handling and silent no-op on mismatch
//!   (the registry itself never inspects the Any payload).
//!
//! ## Concurrency
//!
//! `RwLock` on the inner `HashMap` so concurrent `bind_for` /
//! `unbind_for` calls don't serialize against `register`. Production
//! `cmd_serve` registers all hooks at startup (write-locked once),
//! then handlers and the loader-wrapper read-lock for every load
//! event. The two trigger paths (load and evict) never contend on
//! the same key in practice — the manager's outer `RwLock` already
//! serializes load/evict into a write-lock — but the inner RwLock
//! makes the registry's behavior independent of that outer
//! contract.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::serve::kv_persist::spiller::KvCacheSpill;
use crate::serve::kv_persist::EngineBindable;
use crate::serve::quant_select::QuantType;

/// Composite key for the per-family hook table. Mirrors the
/// `BlockPrefixCacheSpiller::FamilyKey` shape so the two registries
/// stay key-equivalent: an entry inserted into the spiller via
/// `register_family((repo, quant), hook)` should round-trip to the
/// same key shape `(repo, quant)` in this registry.
type FamilyKey = (String, &'static str);

/// Per-`(repo, quant)` factory for "lazy real-hook construction at
/// first load". Phase B-dense.2 adds this trait so a per-family hook
/// that needs the live engine to even be CONSTRUCTED (e.g.
/// [`crate::serve::kv_persist::families::gemma4_dense::Gemma4DenseSpill`]
/// — its shape config + dtype come from `MlxModelWeights`) can be
/// registered AS A FACTORY at startup, then materialized on first
/// load when the engine is finally available.
///
/// ## Why a factory not a direct hook
///
/// At `cmd_serve` startup the operator's GGUF has not yet been
/// loaded — `MlxModelWeights` does not exist. The C.1 stub
/// registration exists precisely so the spiller substrate has *some*
/// hook to dispatch into; B-dense.2 wires the real hook lazily on
/// the first load via this factory.
///
/// ## Substitution flow (atomic)
///
/// On first successful load for a `(repo, quant)` with a registered
/// factory, [`KvPersistRegistry::try_substitute_on_load`] invokes
/// `factory.try_construct(engine_dyn)` and atomically:
///
///   1. Updates the registry's `hooks` map (overwriting the prior
///      stub `EngineBindable`).
///   2. Returns the new `Arc<Mutex<dyn KvCacheSpill>>` so the caller
///      ([`crate::serve::kv_persist::loader_wrapper::LoaderWrapper`])
///      can also call `BlockPrefixCacheSpiller::register_family` —
///      keeping the two registration tables (registry + spiller) in
///      lock-step.
///
/// If `try_construct` returns `None` (engine type mismatch), no
/// substitution happens and the existing stub registration remains
/// the active hook for that `(repo, quant)`. The caller falls
/// through to the existing `bind_for` flow (which also no-ops on the
/// stub's mismatching downcast — see `StubGemma4Spill`'s
/// `EngineBindable` impl).
///
/// ## Send + Sync
///
/// Required because the registry's `factories` map lives behind an
/// `RwLock` accessed from concurrent load triggers (per the
/// `KvPersistRegistry` concurrency note in module docs).
pub trait FamilyHookFactory: Send + Sync {
    /// Try to construct a real per-family hook from the live engine.
    /// Returns `Some((kv_hook, bindable_hook))` on success — the
    /// caller substitutes BOTH simultaneously (spiller registration
    /// + registry registration). Returns `None` on engine type
    /// mismatch — the registry leaves the prior stub hook in place
    /// and the load proceeds with stub-Skipped semantics.
    ///
    /// ## Contract: no panic on type mismatch
    ///
    /// Implementations MUST use `Arc::downcast` (or equivalent
    /// non-panicking extraction) and return `None` on mismatch. A
    /// panic here would propagate up through `LoaderWrapper::load`
    /// and break the load path.
    ///
    /// ## Contract: no Arc retention
    ///
    /// Implementations MUST NOT keep a clone of `engine_dyn` itself.
    /// They may keep clones of the inner `Arc<...>` they downcast
    /// out of `engine_dyn` (e.g.
    /// `Arc<crate::serve::kv_persist::families::gemma4_dense::EngineHandle>`),
    /// but the type-erased outer Arc must drop at end of scope so
    /// the [`crate::serve::kv_persist::loader_wrapper::LoaderWrapper`]'s
    /// `Arc::try_unwrap` succeeds (mirrors the
    /// [`EngineBindable::bind_engine`] retention contract).
    fn try_construct(
        &self,
        engine_dyn: Arc<dyn Any + Send + Sync>,
    ) -> Option<(Arc<Mutex<dyn KvCacheSpill>>, Arc<dyn EngineBindable>)>;
}

/// Per-`(repo, quant)` registry of [`EngineBindable`] hooks. See
/// module docs for the design rationale.
///
/// `Default` via [`KvPersistRegistry::new`].
pub struct KvPersistRegistry {
    /// Inner map. `RwLock` for concurrent reads (per
    /// concurrency note in module docs).
    hooks: RwLock<HashMap<FamilyKey, Arc<dyn EngineBindable>>>,
    /// Per-`(repo, quant)` factories (Phase B-dense.2). Populated
    /// once at `cmd_serve` startup; `try_substitute_on_load`
    /// consumes a factory entry on first successful load to
    /// materialize the real hook. Factories are NOT removed after
    /// use — they remain for any subsequent re-load (HotSwap evict
    /// → readmit) so the materialization repeats with the fresh
    /// engine.
    factories: RwLock<HashMap<FamilyKey, Arc<dyn FamilyHookFactory>>>,
}

impl Default for KvPersistRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl KvPersistRegistry {
    /// Construct an empty registry. Production `cmd_serve` builds one
    /// per `HotSwapManager` instance; tests build per-test instances.
    pub fn new() -> Self {
        Self {
            hooks: RwLock::new(HashMap::new()),
            factories: RwLock::new(HashMap::new()),
        }
    }

    /// Register an [`EngineBindable`] hook for `(repo, quant)`.
    ///
    /// Re-registering the same key OVERWRITES the prior hook (matches
    /// `BlockPrefixCacheSpiller::register_family` semantics — the
    /// freshest registration wins). Idempotent enough that an
    /// operator running `cmd_serve` twice in a row doesn't accumulate
    /// stale hooks.
    pub fn register(&self, repo: String, quant: QuantType, hook: Arc<dyn EngineBindable>) {
        let mut g = self
            .hooks
            .write()
            .expect("KvPersistRegistry::hooks RwLock poisoned");
        g.insert((repo, quant.as_str()), hook);
    }

    /// Remove the registration for `(repo, quant)`. Returns `true` if
    /// a hook was removed, `false` if no hook was registered. Used by
    /// the manager when a model is permanently removed from the loaded
    /// pool (e.g. `cmd_cache clear --model`). Mirror of
    /// `BlockPrefixCacheSpiller::unregister_family`.
    pub fn unregister(&self, repo: &str, quant: QuantType) -> bool {
        let mut g = self
            .hooks
            .write()
            .expect("KvPersistRegistry::hooks RwLock poisoned");
        let key = (repo.to_string(), quant.as_str());
        g.remove(&key).is_some()
    }

    /// Number of currently-registered hooks. Used by tests + the
    /// `/v1/models` extension fields to surface "how many families
    /// have an engine-binding hook wired".
    pub fn registered_count(&self) -> usize {
        self.hooks
            .read()
            .expect("KvPersistRegistry::hooks RwLock poisoned")
            .len()
    }

    /// Bind a live engine reference to the hook registered for
    /// `(repo, quant)`. Calls
    /// [`EngineBindable::bind_engine`] on the hook with the
    /// type-erased `engine_dyn`. The hook is responsible for
    /// downcasting the `Arc<dyn Any>` to its expected concrete type
    /// and silently no-opping on type mismatch.
    ///
    /// No-op (returns without action) when no hook is registered for
    /// `(repo, quant)`. The C.1 contract is "best-effort": a
    /// missing registration is not an error — an operator may run
    /// `cmd_serve --kv-persist` against a family with no registered
    /// hook (e.g. early Phase B before `B-hybrid.1` lands), in which
    /// case the pool falls through to the spiller's own `Skipped`
    /// short-circuit at `post_admit` time.
    pub fn bind_for(
        &self,
        repo: &str,
        quant: QuantType,
        engine_dyn: Arc<dyn Any + Send + Sync>,
    ) {
        let g = self
            .hooks
            .read()
            .expect("KvPersistRegistry::hooks RwLock poisoned");
        let key = (repo.to_string(), quant.as_str());
        if let Some(hook) = g.get(&key) {
            hook.bind_engine(engine_dyn);
        }
    }

    /// Drop the live engine reference for `(repo, quant)`. Calls
    /// [`EngineBindable::unbind_engine`] on the registered hook.
    /// No-op when no hook is registered. Symmetric counterpart to
    /// [`Self::bind_for`].
    pub fn unbind_for(&self, repo: &str, quant: QuantType) {
        let g = self
            .hooks
            .read()
            .expect("KvPersistRegistry::hooks RwLock poisoned");
        let key = (repo.to_string(), quant.as_str());
        if let Some(hook) = g.get(&key) {
            hook.unbind_engine();
        }
    }

    /// Register a [`FamilyHookFactory`] for `(repo, quant)`. Phase
    /// B-dense.2's lazy real-hook construction seam.
    ///
    /// Re-registering the same key OVERWRITES the prior factory
    /// (matches `register` semantics — freshest wins). Idempotent.
    ///
    /// The factory remains registered after a substitution succeeds:
    /// a subsequent evict + readmit cycle re-materializes the hook
    /// against the freshly-loaded engine. This mirrors how
    /// `register_family` on the spiller is "one entry per
    /// `(repo, quant)` for the lifetime of the loaded model".
    pub fn register_factory(
        &self,
        repo: String,
        quant: QuantType,
        factory: Arc<dyn FamilyHookFactory>,
    ) {
        let mut g = self
            .factories
            .write()
            .expect("KvPersistRegistry::factories RwLock poisoned");
        g.insert((repo, quant.as_str()), factory);
    }

    /// Number of currently-registered factories. Symmetric to
    /// [`Self::registered_count`] for hooks. Used by tests to
    /// verify factory wiring without leaking the `Arc<dyn FamilyHookFactory>`.
    pub fn factory_count(&self) -> usize {
        self.factories
            .read()
            .expect("KvPersistRegistry::factories RwLock poisoned")
            .len()
    }

    /// Test / diagnostic accessor: returns `true` iff a factory is
    /// registered for `(repo, quant)`.
    pub fn contains_factory(&self, repo: &str, quant: QuantType) -> bool {
        let g = self
            .factories
            .read()
            .expect("KvPersistRegistry::factories RwLock poisoned");
        let key = (repo.to_string(), quant.as_str());
        g.contains_key(&key)
    }

    /// Phase B-dense.2 — try to substitute the registered stub hook
    /// for a real per-family hook by invoking the registered factory
    /// (if any) with the live `engine_dyn`. Returns the new
    /// `Arc<Mutex<dyn KvCacheSpill>>` so the caller (the
    /// `LoaderWrapper`) can also update the spiller's registration
    /// table — keeping registry + spiller in lock-step.
    ///
    /// Returns `None` when:
    ///   * no factory is registered for `(repo, quant)`, OR
    ///   * the factory's `try_construct` returned `None` (engine
    ///     type mismatch — e.g. a non-Gemma 4 family loaded against
    ///     a Gemma 4 factory).
    ///
    /// On `Some(_)`, the registry's `hooks` map is OVERWRITTEN
    /// atomically with the factory-produced `Arc<dyn EngineBindable>`.
    /// Subsequent `bind_for` calls hit the new hook directly.
    ///
    /// Lock discipline: write-locks `hooks` only when substitution
    /// succeeds; `factories` is read-only on this path so concurrent
    /// load triggers don't serialize against each other.
    pub fn try_substitute_on_load(
        &self,
        repo: &str,
        quant: QuantType,
        engine_dyn: Arc<dyn Any + Send + Sync>,
    ) -> Option<Arc<Mutex<dyn KvCacheSpill>>> {
        let key = (repo.to_string(), quant.as_str());
        // 1. Look up the factory under a read lock; clone the Arc
        //    out so we don't hold the lock across `try_construct`
        //    (which may run for non-trivial time as it inspects the
        //    type-erased engine).
        let factory = {
            let g = self
                .factories
                .read()
                .expect("KvPersistRegistry::factories RwLock poisoned");
            g.get(&key).cloned()
        };
        let factory = factory?;
        // 2. Invoke the factory. `None` = type mismatch ⇒ leave the
        //    existing stub registration in place.
        let (kv_hook, bindable_hook) = factory.try_construct(engine_dyn)?;
        // 3. Substitute the registry's hook entry atomically.
        {
            let mut g = self
                .hooks
                .write()
                .expect("KvPersistRegistry::hooks RwLock poisoned");
            g.insert(key, bindable_hook);
        }
        Some(kv_hook)
    }

    /// Test / diagnostic accessor: returns `true` iff a hook is
    /// registered for `(repo, quant)`. Avoids leaking the
    /// `Arc<dyn EngineBindable>` from the read path.
    pub fn contains(&self, repo: &str, quant: QuantType) -> bool {
        let g = self
            .hooks
            .read()
            .expect("KvPersistRegistry::hooks RwLock poisoned");
        let key = (repo.to_string(), quant.as_str());
        g.contains_key(&key)
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Mock [`EngineBindable`] that records every bind/unbind call
    /// for assertion in tests.
    struct MockBindable {
        binds: AtomicU32,
        unbinds: AtomicU32,
        /// Last engine_dyn type-id observed at bind. `0` means
        /// "never bound". We can't store the Arc itself because Any
        /// can't be cloned generically; the type-id is sufficient
        /// to assert downcast routing in tests.
        last_bound_type: std::sync::Mutex<Option<std::any::TypeId>>,
    }

    impl MockBindable {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                binds: AtomicU32::new(0),
                unbinds: AtomicU32::new(0),
                last_bound_type: std::sync::Mutex::new(None),
            })
        }

        fn bind_count(&self) -> u32 {
            self.binds.load(Ordering::SeqCst)
        }

        fn unbind_count(&self) -> u32 {
            self.unbinds.load(Ordering::SeqCst)
        }

        fn last_bound_type(&self) -> Option<std::any::TypeId> {
            *self.last_bound_type.lock().unwrap()
        }
    }

    impl EngineBindable for MockBindable {
        fn bind_engine(&self, engine_dyn: Arc<dyn Any + Send + Sync>) {
            self.binds.fetch_add(1, Ordering::SeqCst);
            // Record the concrete type-id for assertion. `Any::type_id`
            // on a `&dyn Any` returns the underlying type's TypeId.
            let id: std::any::TypeId = (*engine_dyn).type_id();
            *self.last_bound_type.lock().unwrap() = Some(id);
        }
        fn unbind_engine(&self) {
            self.unbinds.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Small concrete payload type used by the test's bind_for calls.
    #[derive(Debug)]
    struct DummyEnginePayload {
        _marker: u32,
    }

    // ===== Test 1 ==========================================================
    #[test]
    fn new_registry_has_zero_hooks() {
        let reg = KvPersistRegistry::new();
        assert_eq!(reg.registered_count(), 0);
        assert!(!reg.contains("acme/m1", QuantType::Q4_K_M));
    }

    // ===== Test 2 ==========================================================
    #[test]
    fn registry_register_then_bind_round_trip() {
        // R-C1-equivalent for the registry: registering a hook then
        // calling bind_for fires bind_engine on the hook with the
        // exact payload (verified via type-id).
        let reg = KvPersistRegistry::new();
        let mock = MockBindable::new();
        reg.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            mock.clone() as Arc<dyn EngineBindable>,
        );
        assert_eq!(reg.registered_count(), 1);
        assert!(reg.contains("acme/m1", QuantType::Q4_K_M));

        let payload = Arc::new(DummyEnginePayload { _marker: 0xABCD });
        let payload_dyn: Arc<dyn Any + Send + Sync> = payload.clone();
        reg.bind_for("acme/m1", QuantType::Q4_K_M, payload_dyn);

        assert_eq!(mock.bind_count(), 1, "bind_engine fired exactly once");
        assert_eq!(mock.unbind_count(), 0);
        assert_eq!(
            mock.last_bound_type(),
            Some(std::any::TypeId::of::<DummyEnginePayload>()),
            "type-id round-trip survives Arc<dyn Any> erasure"
        );
    }

    // ===== Test 3 ==========================================================
    #[test]
    fn registry_unbind_clears_engine_handle() {
        let reg = KvPersistRegistry::new();
        let mock = MockBindable::new();
        reg.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            mock.clone() as Arc<dyn EngineBindable>,
        );
        let payload = Arc::new(DummyEnginePayload { _marker: 1 });
        reg.bind_for(
            "acme/m1",
            QuantType::Q4_K_M,
            payload as Arc<dyn Any + Send + Sync>,
        );
        assert_eq!(mock.bind_count(), 1);
        reg.unbind_for("acme/m1", QuantType::Q4_K_M);
        assert_eq!(mock.unbind_count(), 1);
    }

    // ===== Test 4 ==========================================================
    #[test]
    fn registry_bind_for_unknown_repo_quant_is_noop() {
        // Spec test 3: bind_for on a key that wasn't registered must
        // not panic; the registry silently drops the bind call.
        let reg = KvPersistRegistry::new();
        let mock = MockBindable::new();
        reg.register(
            "acme/known".to_string(),
            QuantType::Q4_K_M,
            mock.clone() as Arc<dyn EngineBindable>,
        );

        let payload = Arc::new(DummyEnginePayload { _marker: 2 });
        // Wrong repo.
        reg.bind_for(
            "acme/unknown",
            QuantType::Q4_K_M,
            payload.clone() as Arc<dyn Any + Send + Sync>,
        );
        // Wrong quant.
        reg.bind_for(
            "acme/known",
            QuantType::Q8_0,
            payload as Arc<dyn Any + Send + Sync>,
        );

        // The mock for the registered key is untouched.
        assert_eq!(mock.bind_count(), 0, "no hook fired on unknown key");
        // unbind_for on unknown key is also no-op.
        reg.unbind_for("nobody/here", QuantType::Q4_K_M);
        assert_eq!(mock.unbind_count(), 0);
    }

    // ===== Test 5 ==========================================================
    #[test]
    fn registry_re_register_overwrites_prior_hook() {
        // Mirror of BlockPrefixCacheSpiller's re-register-overwrites
        // semantic. The COUNT stays at 1; the second hook fires on
        // bind_for, the first does not.
        let reg = KvPersistRegistry::new();
        let first = MockBindable::new();
        let second = MockBindable::new();

        reg.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            first.clone() as Arc<dyn EngineBindable>,
        );
        reg.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            second.clone() as Arc<dyn EngineBindable>,
        );
        assert_eq!(reg.registered_count(), 1, "no count growth on overwrite");

        let payload = Arc::new(DummyEnginePayload { _marker: 3 });
        reg.bind_for(
            "acme/m1",
            QuantType::Q4_K_M,
            payload as Arc<dyn Any + Send + Sync>,
        );
        assert_eq!(first.bind_count(), 0, "first registration was overwritten");
        assert_eq!(second.bind_count(), 1, "second registration is live");
    }

    // ===== Test 6 ==========================================================
    #[test]
    fn registry_unregister_idempotent() {
        // Symmetric to BlockPrefixCacheSpiller::unregister_family —
        // returns true on hit, false on miss. Idempotent across
        // repeated calls.
        let reg = KvPersistRegistry::new();
        let mock = MockBindable::new();
        reg.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            mock as Arc<dyn EngineBindable>,
        );
        assert!(reg.unregister("acme/m1", QuantType::Q4_K_M));
        assert_eq!(reg.registered_count(), 0);
        assert!(!reg.unregister("acme/m1", QuantType::Q4_K_M));
        assert!(!reg.unregister("nobody/here", QuantType::Q4_K_M));
    }

    // ===== Test 7 ==========================================================
    #[test]
    fn registry_distinct_quant_grows_table() {
        // Same repo, distinct quant → two entries (mirrors the
        // spiller's behavior).
        let reg = KvPersistRegistry::new();
        let m1 = MockBindable::new();
        let m2 = MockBindable::new();
        reg.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            m1.clone() as Arc<dyn EngineBindable>,
        );
        reg.register(
            "acme/m1".to_string(),
            QuantType::Q8_0,
            m2.clone() as Arc<dyn EngineBindable>,
        );
        assert_eq!(reg.registered_count(), 2);

        let payload = Arc::new(DummyEnginePayload { _marker: 4 });
        reg.bind_for(
            "acme/m1",
            QuantType::Q8_0,
            payload as Arc<dyn Any + Send + Sync>,
        );
        assert_eq!(m1.bind_count(), 0);
        assert_eq!(m2.bind_count(), 1);
    }

    // ===== Phase B-dense.2 — FamilyHookFactory tests ========================

    /// Mock [`KvCacheSpill`] that just remembers it was constructed.
    /// Used as the `kv_hook` half of the factory return tuple.
    struct MockKvSpillForFactory;

    impl crate::serve::kv_persist::spiller::KvCacheSpill for MockKvSpillForFactory {
        fn block_alignment(&self) -> u32 {
            crate::serve::kv_persist::format::BLOCK_TOKENS
        }
        fn snapshot_block(
            &self,
            _layer_rank: usize,
            _range: std::ops::Range<u32>,
        ) -> Option<Vec<u8>> {
            None
        }
        fn restore_block(
            &mut self,
            _layer_rank: usize,
            _range: std::ops::Range<u32>,
            _payload: &[u8],
        ) -> std::result::Result<(), crate::serve::multi_model::SpillErrorKind> {
            Ok(())
        }
    }

    /// Synthetic concrete engine type used by the factory tests. The
    /// factory expects to downcast to `ExpectedEngineHandle`; tests
    /// pass either this (matching) or some unrelated type (mismatch).
    #[derive(Debug)]
    struct ExpectedEngineHandle {
        marker: u32,
    }

    /// Mock factory — succeeds when the dyn Any downcasts to
    /// `Arc<ExpectedEngineHandle>`, fails otherwise.
    struct MockFactoryExpecting;

    impl FamilyHookFactory for MockFactoryExpecting {
        fn try_construct(
            &self,
            engine_dyn: Arc<dyn Any + Send + Sync>,
        ) -> Option<(Arc<Mutex<dyn KvCacheSpill>>, Arc<dyn EngineBindable>)> {
            // Use Arc::downcast — returns Result, never panics.
            match engine_dyn.downcast::<ExpectedEngineHandle>() {
                Ok(_handle) => {
                    let kv: Arc<Mutex<dyn KvCacheSpill>> =
                        Arc::new(Mutex::new(MockKvSpillForFactory));
                    let bindable: Arc<dyn EngineBindable> = MockBindable::new();
                    Some((kv, bindable))
                }
                Err(_) => None,
            }
        }
    }

    /// Factory that always returns None (simulates a stubbed-out
    /// non-matching family).
    struct MockFactoryAlwaysNone;

    impl FamilyHookFactory for MockFactoryAlwaysNone {
        fn try_construct(
            &self,
            _engine_dyn: Arc<dyn Any + Send + Sync>,
        ) -> Option<(Arc<Mutex<dyn KvCacheSpill>>, Arc<dyn EngineBindable>)> {
            None
        }
    }

    // ===== Test FH-1 =======================================================
    #[test]
    fn registry_register_factory_round_trip() {
        // B-dense.2 spec: register_factory + factory_count grow
        // alongside the existing hook registration.
        let reg = KvPersistRegistry::new();
        assert_eq!(reg.factory_count(), 0);
        assert!(!reg.contains_factory("acme/m1", QuantType::Q4_K_M));

        reg.register_factory(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            Arc::new(MockFactoryExpecting),
        );
        assert_eq!(reg.factory_count(), 1);
        assert!(reg.contains_factory("acme/m1", QuantType::Q4_K_M));
        // hooks count is independent of factory count.
        assert_eq!(reg.registered_count(), 0);
    }

    // ===== Test FH-2 =======================================================
    #[test]
    fn registry_try_substitute_on_load_succeeds_on_type_match() {
        // B-dense.2 Hypothesis 1: factory.try_construct yields
        // Some(...), registry's hooks map is OVERWRITTEN.
        let reg = KvPersistRegistry::new();
        // Pre-existing stub hook.
        let stub = MockBindable::new();
        reg.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            stub.clone() as Arc<dyn EngineBindable>,
        );
        assert_eq!(reg.registered_count(), 1);

        reg.register_factory(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            Arc::new(MockFactoryExpecting),
        );

        let payload = Arc::new(ExpectedEngineHandle { marker: 0xBEEF });
        let payload_dyn: Arc<dyn Any + Send + Sync> = payload;
        let result = reg.try_substitute_on_load("acme/m1", QuantType::Q4_K_M, payload_dyn);
        assert!(result.is_some(), "factory yields a kv_hook on type match");

        // Substitution: the registry's hook for (acme/m1, Q4_K_M) is
        // NOT the original stub anymore. We can't compare Arc
        // identity through Arc<dyn EngineBindable>, but we verify by
        // bind_for'ing a payload and observing the original stub's
        // bind_count remains 0.
        let next_payload = Arc::new(ExpectedEngineHandle { marker: 0x1234 });
        reg.bind_for(
            "acme/m1",
            QuantType::Q4_K_M,
            next_payload as Arc<dyn Any + Send + Sync>,
        );
        assert_eq!(
            stub.bind_count(),
            0,
            "after substitution, the original stub no longer fires"
        );
        // Counts are unchanged: registry replaces the value at the
        // existing key without growing the table.
        assert_eq!(reg.registered_count(), 1);
        assert_eq!(reg.factory_count(), 1);
    }

    // ===== Test FH-3 =======================================================
    #[test]
    fn registry_try_substitute_on_load_returns_none_on_type_mismatch() {
        // B-dense.2: factory returns None ⇒ substitution does NOT
        // happen ⇒ the existing stub remains the registered hook.
        let reg = KvPersistRegistry::new();
        let stub = MockBindable::new();
        reg.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            stub.clone() as Arc<dyn EngineBindable>,
        );
        // Factory expects ExpectedEngineHandle; we will give it a
        // `DummyEnginePayload` ⇒ Arc::downcast fails ⇒ None.
        reg.register_factory(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            Arc::new(MockFactoryExpecting),
        );

        let wrong_payload = Arc::new(DummyEnginePayload { _marker: 0xFEED });
        let result = reg.try_substitute_on_load(
            "acme/m1",
            QuantType::Q4_K_M,
            wrong_payload as Arc<dyn Any + Send + Sync>,
        );
        assert!(result.is_none(), "type mismatch ⇒ no substitution");

        // Stub still in place.
        let next_payload = Arc::new(DummyEnginePayload { _marker: 0xBABE });
        reg.bind_for(
            "acme/m1",
            QuantType::Q4_K_M,
            next_payload as Arc<dyn Any + Send + Sync>,
        );
        assert_eq!(
            stub.bind_count(),
            1,
            "stub remains registered after non-matching factory"
        );
    }

    // ===== Test FH-4 =======================================================
    #[test]
    fn registry_try_substitute_on_load_no_factory_is_noop() {
        // B-dense.2: no factory registered ⇒ substitution returns None
        // and the registry state is unchanged.
        let reg = KvPersistRegistry::new();
        let stub = MockBindable::new();
        reg.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            stub.clone() as Arc<dyn EngineBindable>,
        );

        let payload = Arc::new(ExpectedEngineHandle { marker: 0xC0DE });
        let result = reg.try_substitute_on_load(
            "acme/m1",
            QuantType::Q4_K_M,
            payload as Arc<dyn Any + Send + Sync>,
        );
        assert!(result.is_none(), "no factory ⇒ no substitution");
        assert_eq!(reg.factory_count(), 0);
        // Stub still bind-able.
        let payload2 = Arc::new(ExpectedEngineHandle { marker: 1 });
        reg.bind_for(
            "acme/m1",
            QuantType::Q4_K_M,
            payload2 as Arc<dyn Any + Send + Sync>,
        );
        assert_eq!(stub.bind_count(), 1);
    }

    // ===== Test FH-5 =======================================================
    #[test]
    fn registry_factory_always_none_does_not_substitute() {
        // Defensive: factory is registered, but its try_construct is
        // hard-coded to None (e.g. a stub family for which lazy
        // construction is intentionally not implemented). Substitute
        // returns None; the existing hook stays.
        let reg = KvPersistRegistry::new();
        let stub = MockBindable::new();
        reg.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            stub.clone() as Arc<dyn EngineBindable>,
        );
        reg.register_factory(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            Arc::new(MockFactoryAlwaysNone),
        );

        let payload = Arc::new(ExpectedEngineHandle { marker: 5 });
        let result = reg.try_substitute_on_load(
            "acme/m1",
            QuantType::Q4_K_M,
            payload as Arc<dyn Any + Send + Sync>,
        );
        assert!(result.is_none());
        // The stub is still wired.
        let payload2 = Arc::new(ExpectedEngineHandle { marker: 6 });
        reg.bind_for(
            "acme/m1",
            QuantType::Q4_K_M,
            payload2 as Arc<dyn Any + Send + Sync>,
        );
        assert_eq!(stub.bind_count(), 1);
    }

    // ===== Test FH-6 =======================================================
    #[test]
    fn registry_factory_re_register_overwrites_prior() {
        // Symmetric to register_re_register_overwrites_prior_hook but
        // for factories.
        let reg = KvPersistRegistry::new();
        reg.register_factory(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            Arc::new(MockFactoryAlwaysNone),
        );
        reg.register_factory(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            Arc::new(MockFactoryExpecting),
        );
        assert_eq!(reg.factory_count(), 1, "no count growth on overwrite");

        // Now substitute — proves the SECOND factory is live (the
        // first would have returned None unconditionally).
        let stub = MockBindable::new();
        reg.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            stub as Arc<dyn EngineBindable>,
        );
        let payload = Arc::new(ExpectedEngineHandle { marker: 7 });
        let result = reg.try_substitute_on_load(
            "acme/m1",
            QuantType::Q4_K_M,
            payload as Arc<dyn Any + Send + Sync>,
        );
        assert!(result.is_some(), "second-registered factory is live");
    }

    // ===== Test 8 ==========================================================
    #[test]
    fn registry_multi_threaded_bind_for_is_safe() {
        // Spawn N threads each calling bind_for on the same key;
        // assert no panic + total bind count equals N. Exercises
        // the RwLock read path concurrency.
        use std::thread;
        const N_THREADS: u32 = 8;
        const N_ITERS: u32 = 32;

        let reg = Arc::new(KvPersistRegistry::new());
        let mock = MockBindable::new();
        reg.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            mock.clone() as Arc<dyn EngineBindable>,
        );

        let handles: Vec<_> = (0..N_THREADS)
            .map(|_| {
                let reg = Arc::clone(&reg);
                thread::spawn(move || {
                    for i in 0..N_ITERS {
                        let payload = Arc::new(DummyEnginePayload { _marker: i });
                        reg.bind_for(
                            "acme/m1",
                            QuantType::Q4_K_M,
                            payload as Arc<dyn Any + Send + Sync>,
                        );
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("worker did not panic");
        }
        assert_eq!(mock.bind_count(), N_THREADS * N_ITERS);
    }
}
