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
use std::sync::{Arc, RwLock};

use crate::serve::kv_persist::EngineBindable;
use crate::serve::quant_select::QuantType;

/// Composite key for the per-family hook table. Mirrors the
/// `BlockPrefixCacheSpiller::FamilyKey` shape so the two registries
/// stay key-equivalent: an entry inserted into the spiller via
/// `register_family((repo, quant), hook)` should round-trip to the
/// same key shape `(repo, quant)` in this registry.
type FamilyKey = (String, &'static str);

/// Per-`(repo, quant)` registry of [`EngineBindable`] hooks. See
/// module docs for the design rationale.
///
/// `Default` via [`KvPersistRegistry::new`].
pub struct KvPersistRegistry {
    /// Inner map. `RwLock` for concurrent reads (per
    /// concurrency note in module docs).
    hooks: RwLock<HashMap<FamilyKey, Arc<dyn EngineBindable>>>,
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
