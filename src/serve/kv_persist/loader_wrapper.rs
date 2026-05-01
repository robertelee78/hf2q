//! ADR-017 §C.1 — `LoaderWrapper<E>`: a `ModelLoader<E>`-decorating
//! pass-through that drives [`crate::serve::kv_persist::KvPersistRegistry::bind_for`]
//! on every successful load and `unbind_for` at evict time.
//!
//! ## Why this exists (Chesterton's fence)
//!
//! Phase A.3's [`crate::serve::kv_persist::BlockPrefixCacheSpiller`]
//! has a per-family hook table keyed on `(repo, quant)`. Per-family
//! hooks that need engine-side state (e.g. Gemma 4 dense K/V buffers
//! living inside `MlxModelWeights`) need their engine handle wired
//! BEFORE the spiller's `post_admit` trigger fires (see
//! `spiller.rs:398-477`). The natural seam is the load path — the
//! moment a fresh `Engine` is constructed.
//!
//! `LoaderWrapper` decorates `Arc<dyn ModelLoader<E>>` to perform that
//! binding without modifying the `ModelLoader` trait surface (which
//! is stable per the iter-212 ship contract). On every successful
//! `load(path, config) -> E`, the wrapper:
//!
//! 1. Reads its `pending_bind` slot for `(repo, quant)`. The slot is
//!    populated by `cmd_serve` immediately before the `pool.load_or_get`
//!    call that drives this loader.
//! 2. Wraps the freshly-loaded `E` in `Arc<E>` and constructs an
//!    `Arc<dyn Any + Send + Sync>` view on the same allocation.
//! 3. Calls `registry.bind_for(repo, quant, engine_dyn)`. The hook's
//!    `EngineBindable::bind_engine` impl performs the downcast — for
//!    Gemma 4 dense it pulls out an `Arc<EngineHandle>`; for the C.1
//!    StubGemma4Spill it's a silent no-op.
//! 4. Reclaims the inner `E` via `Arc::try_unwrap` and returns it to
//!    the manager. This succeeds iff the bind hook didn't keep a clone
//!    of the `Arc`. The `EngineBindable` contract requires hooks to
//!    keep only what they downcast OUT of the Arc (e.g. the inner
//!    `Arc<EngineHandle>` reachable from the type-erased Any), not
//!    the Arc itself.
//!
//! ## Why a `pending_bind` slot instead of a thread-local
//!
//! The `ModelLoader::load(path, config)` signature carries no
//! `(repo, quant)` context — `HotSwapManager::load_or_get` knows
//! those values but the loader callsite inside `load_or_get` does
//! not pass them through (see `multi_model.rs::HotSwapManager`'s
//! load path; modifying that surface is out-of-scope for C.1).
//!
//! Two seams are available without touching `multi_model.rs`:
//!
//! 1. A thread-local set by `cmd_serve` before `load_or_get`. Fragile
//!    under tokio's work-stealing scheduler — the load can resume on
//!    a different thread than the setter.
//! 2. A `Mutex<Option<(String, QuantType)>>` "pending" slot on the
//!    wrapper itself, set by `cmd_serve` before the
//!    synchronous `load_or_get` call. The wrapper consumes the slot
//!    inside its `load(...)` impl. Production `cmd_serve` startup is
//!    single-threaded so there is no setter/load race. Tests assert
//!    the discipline directly (the bind only fires when a pending
//!    slot is set).
//!
//! Option 2 is the chosen design. It keeps the dependency graph
//! local (no env / TLS pollution) and exercises a synchronous
//! contract that's easy to test.
//!
//! ## What the wrapper does NOT do
//!
//! - Does NOT panic on `Arc::try_unwrap` failure. A failure here means
//!   the hook held a clone of the type-erased Arc (which violates the
//!   `EngineBindable` contract). The wrapper surfaces this as a
//!   `LoaderFailed`-shaped `anyhow::Error` so the manager treats the
//!   load as failed (same shape as a real load error). Production
//!   hooks NEVER hit this path; tests assert the shape of the error.
//! - Does NOT enforce that the registry has a hook registered for
//!   `(repo, quant)`. Missing-hook is a silent no-op at the registry
//!   layer (per `KvPersistRegistry::bind_for` semantics).
//! - Does NOT clear the pending slot on load failure — the inner
//!   loader's failure is the operator's signal to retry, and a stale
//!   pending slot is a no-op until the next successful load consumes
//!   it.

use std::any::Any;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::serve::kv_persist::registry::KvPersistRegistry;
use crate::serve::multi_model::{EngineConfig, ModelLoader};
use crate::serve::quant_select::QuantType;

/// Decorating `ModelLoader<E>` that drives the C.1 engine-binding
/// registry. See module docs for the full design rationale.
pub struct LoaderWrapper<E> {
    /// Underlying loader. Production wires `DefaultModelLoader`;
    /// tests substitute a mock.
    inner: Arc<dyn ModelLoader<E>>,
    /// C.1 registry — `bind_for` / `unbind_for` are routed through
    /// this `Arc`. Cheap-clones into the wrapper at construction.
    registry: Arc<KvPersistRegistry>,
    /// Operator-set `(repo, quant)` for the next pending load.
    /// Consumed (taken) inside `ModelLoader::load`. Always `None`
    /// outside of an in-flight `set_pending_bind` → `load_or_get`
    /// transition.
    pending_bind: Mutex<Option<(String, QuantType)>>,
    /// PhantomData to bind the `E` generic to the wrapper. The
    /// `fn(E)` form makes `LoaderWrapper<E>` invariant in `E` and
    /// avoids unnecessary auto-trait implications.
    _phantom: PhantomData<fn(E)>,
}

impl<E> LoaderWrapper<E>
where
    E: Send + Sync + 'static,
{
    /// Construct a wrapper around `inner` driving `registry`. The
    /// wrapper starts with no pending bind; `cmd_serve` arms the slot
    /// before each `pool.load_or_get` call.
    pub fn new(inner: Arc<dyn ModelLoader<E>>, registry: Arc<KvPersistRegistry>) -> Self {
        Self {
            inner,
            registry,
            pending_bind: Mutex::new(None),
            _phantom: PhantomData,
        }
    }

    /// Arm the pending-bind slot for the next `load(...)` call. Must
    /// be invoked synchronously immediately before the load_or_get
    /// driver call (see module docs for the synchronous-contract
    /// rationale).
    ///
    /// Calling twice without an intervening `load(...)` overwrites the
    /// prior pending — the second `(repo, quant)` wins.
    pub fn set_pending_bind(&self, repo: String, quant: QuantType) {
        let mut g = self
            .pending_bind
            .lock()
            .expect("LoaderWrapper::pending_bind Mutex poisoned");
        *g = Some((repo, quant));
    }

    /// Drop the pending-bind slot without firing a load. Used by
    /// `cmd_serve` if the load sequence is aborted before it can
    /// drive `load_or_get` (defensive — the production path always
    /// drives the load_or_get immediately so this should never fire).
    pub fn clear_pending_bind(&self) {
        let mut g = self
            .pending_bind
            .lock()
            .expect("LoaderWrapper::pending_bind Mutex poisoned");
        *g = None;
    }

    /// Test / diagnostic accessor: returns the current pending slot
    /// without consuming it.
    pub fn pending_bind(&self) -> Option<(String, QuantType)> {
        let g = self
            .pending_bind
            .lock()
            .expect("LoaderWrapper::pending_bind Mutex poisoned");
        g.clone()
    }

    /// Drive `unbind_for(repo, quant)` directly through the registry.
    /// Phase C.1's evict-path wire-up calls this from `cmd_serve`'s
    /// graceful-shutdown / explicit-evict paths so stale engine
    /// handles drop in lock-step with the manager's evict.
    ///
    /// Idempotent (matches `KvPersistRegistry::unbind_for`
    /// semantics — no-op when no hook is registered).
    pub fn drive_unbind(&self, repo: &str, quant: QuantType) {
        self.registry.unbind_for(repo, quant);
    }

    /// Test / diagnostic accessor: returns the registry Arc for
    /// downstream inspection.
    pub fn registry(&self) -> Arc<KvPersistRegistry> {
        Arc::clone(&self.registry)
    }
}

impl<E> ModelLoader<E> for LoaderWrapper<E>
where
    E: Send + Sync + 'static,
{
    /// Load via the inner loader, then drive the registry's
    /// `bind_for` for the pending `(repo, quant)` (if any). Returns
    /// the freshly-loaded `E` to the manager.
    ///
    /// Returns `Err(...)` iff the inner loader failed OR the bind
    /// hook held a clone of the type-erased Arc (contract violation;
    /// see module docs).
    fn load(&self, path: &Path, config: &EngineConfig) -> Result<E> {
        // 1. Drive the inner loader. Propagate errors verbatim — the
        //    wrapper does NOT bind on failure (per Hypothesis 2 in the
        //    spec; covered by the loader-wrapper-does-not-bind-on-load-failure
        //    test).
        let engine = self.inner.load(path, config)?;

        // 2. Consume the pending-bind slot. If empty, pass-through —
        //    the bind only fires when `cmd_serve` armed it.
        let pending = {
            let mut g = self
                .pending_bind
                .lock()
                .expect("LoaderWrapper::pending_bind Mutex poisoned");
            g.take()
        };
        let Some((repo, quant)) = pending else {
            return Ok(engine);
        };

        // 3. Build a type-erased Arc view on the freshly-loaded
        //    engine and drive the registry. The hook's
        //    `EngineBindable::bind_engine` is responsible for the
        //    downcast (and silent no-op on mismatch — e.g. when only
        //    a stub family is registered).
        let arc_engine: Arc<E> = Arc::new(engine);
        let dyn_view: Arc<dyn Any + Send + Sync> = Arc::clone(&arc_engine) as _;
        self.registry.bind_for(&repo, quant, dyn_view);

        // 4. Reclaim the inner E. `Arc::try_unwrap` succeeds iff the
        //    only outstanding strong reference is `arc_engine` itself.
        //    The contract on `EngineBindable::bind_engine` requires
        //    hooks to drop the type-erased Arc before returning (they
        //    keep at most a downcast-derived inner Arc — e.g.
        //    `Arc<EngineHandle>` rather than the original
        //    `Arc<dyn Any>`).
        Arc::try_unwrap(arc_engine).map_err(|_| {
            anyhow::anyhow!(
                "LoaderWrapper: registry hook for (repo={repo}, quant={}) \
                 retained a clone of the type-erased engine Arc; \
                 this violates the EngineBindable contract",
                quant.as_str()
            )
        })
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::kv_persist::EngineBindable;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Synthetic engine type used by the loader tests. Holds a marker
    /// the tests can read back to verify the engine round-trips intact
    /// through the wrapper.
    #[derive(Debug)]
    struct TestEngine {
        marker: u32,
    }

    /// Mock `ModelLoader<TestEngine>` that returns either a fresh
    /// `TestEngine` or an error, depending on its `fail_next` flag.
    /// Records every load call for assertion.
    struct MockLoader {
        load_count: AtomicU32,
        next_marker: AtomicU32,
        fail_next: AtomicU32, // 0 = succeed, 1 = fail
    }

    impl MockLoader {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                load_count: AtomicU32::new(0),
                next_marker: AtomicU32::new(0xC1A1),
                fail_next: AtomicU32::new(0),
            })
        }

        fn force_fail_next(&self) {
            self.fail_next.store(1, Ordering::SeqCst);
        }

        fn load_count(&self) -> u32 {
            self.load_count.load(Ordering::SeqCst)
        }
    }

    impl ModelLoader<TestEngine> for MockLoader {
        fn load(&self, _path: &Path, _config: &EngineConfig) -> Result<TestEngine> {
            self.load_count.fetch_add(1, Ordering::SeqCst);
            if self.fail_next.swap(0, Ordering::SeqCst) == 1 {
                anyhow::bail!("mock loader: synthetic load failure")
            }
            let m = self.next_marker.fetch_add(1, Ordering::SeqCst);
            Ok(TestEngine { marker: m })
        }
    }

    /// Mock `EngineBindable` that records every bind/unbind. Mirrors
    /// the registry tests' mock but lives here so the loader_wrapper
    /// tests are self-contained.
    struct LwMockBindable {
        binds: AtomicU32,
        unbinds: AtomicU32,
        /// If `true`, retain a clone of the engine_dyn Arc on bind.
        /// Used to simulate the contract-violation path.
        retain_arc: bool,
        retained: std::sync::Mutex<Option<Arc<dyn Any + Send + Sync>>>,
    }

    impl LwMockBindable {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                binds: AtomicU32::new(0),
                unbinds: AtomicU32::new(0),
                retain_arc: false,
                retained: std::sync::Mutex::new(None),
            })
        }

        fn new_retaining() -> Arc<Self> {
            Arc::new(Self {
                binds: AtomicU32::new(0),
                unbinds: AtomicU32::new(0),
                retain_arc: true,
                retained: std::sync::Mutex::new(None),
            })
        }

        fn bind_count(&self) -> u32 {
            self.binds.load(Ordering::SeqCst)
        }

        fn unbind_count(&self) -> u32 {
            self.unbinds.load(Ordering::SeqCst)
        }
    }

    impl EngineBindable for LwMockBindable {
        fn bind_engine(&self, engine_dyn: Arc<dyn Any + Send + Sync>) {
            self.binds.fetch_add(1, Ordering::SeqCst);
            if self.retain_arc {
                let mut g = self.retained.lock().unwrap();
                *g = Some(engine_dyn);
            }
            // else: drop the Arc (bind compliant — the wrapper's
            // try_unwrap will succeed).
        }
        fn unbind_engine(&self) {
            self.unbinds.fetch_add(1, Ordering::SeqCst);
            let mut g = self.retained.lock().unwrap();
            *g = None;
        }
    }

    fn dummy_path_and_config() -> (std::path::PathBuf, EngineConfig) {
        (
            std::path::PathBuf::from("/dev/null/synthetic.gguf"),
            EngineConfig {
                tokenizer_path: None,
                config_path: None,
                queue_capacity: 32,
                warmup_synchronously: false,
            },
        )
    }

    // ===== Test 1 ==========================================================
    #[test]
    fn loader_wrapper_passes_through_load_when_no_registry_match() {
        // Spec test 4: no pending bind ⇒ wrapper transparently
        // returns the inner loader's engine; no registry call fires.
        let inner = MockLoader::new();
        let registry = Arc::new(KvPersistRegistry::new());
        let mock_hook = LwMockBindable::new();
        registry.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            mock_hook.clone() as Arc<dyn EngineBindable>,
        );

        let wrapper: LoaderWrapper<TestEngine> =
            LoaderWrapper::new(inner.clone() as Arc<dyn ModelLoader<TestEngine>>, registry);
        let (path, cfg) = dummy_path_and_config();
        let engine = wrapper.load(&path, &cfg).expect("inner load OK");
        assert!(engine.marker >= 0xC1A1);
        assert_eq!(inner.load_count(), 1);
        // No pending bind ⇒ no bind fired.
        assert_eq!(mock_hook.bind_count(), 0);
    }

    // ===== Test 2 ==========================================================
    #[test]
    fn loader_wrapper_calls_bind_after_successful_load() {
        // Spec test 5: with a pending bind set + a registered hook,
        // the wrapper's load fires bind_engine on the hook and
        // returns the engine intact.
        let inner = MockLoader::new();
        let registry = Arc::new(KvPersistRegistry::new());
        let mock_hook = LwMockBindable::new();
        registry.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            mock_hook.clone() as Arc<dyn EngineBindable>,
        );

        let wrapper: LoaderWrapper<TestEngine> = LoaderWrapper::new(
            inner.clone() as Arc<dyn ModelLoader<TestEngine>>,
            Arc::clone(&registry),
        );
        wrapper.set_pending_bind("acme/m1".to_string(), QuantType::Q4_K_M);
        assert_eq!(
            wrapper.pending_bind(),
            Some(("acme/m1".to_string(), QuantType::Q4_K_M))
        );

        let (path, cfg) = dummy_path_and_config();
        let engine = wrapper.load(&path, &cfg).expect("inner load OK");
        assert!(engine.marker >= 0xC1A1);
        assert_eq!(inner.load_count(), 1);
        assert_eq!(mock_hook.bind_count(), 1, "bind_engine fired exactly once");
        // Pending slot consumed.
        assert_eq!(wrapper.pending_bind(), None);
    }

    // ===== Test 3 ==========================================================
    #[test]
    fn loader_wrapper_does_not_bind_on_load_failure() {
        // Spec test 6: when the inner loader returns Err, the
        // wrapper does NOT call bind_for (the engine doesn't exist).
        // The pending slot is left intact (operator may retry).
        let inner = MockLoader::new();
        inner.force_fail_next();
        let registry = Arc::new(KvPersistRegistry::new());
        let mock_hook = LwMockBindable::new();
        registry.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            mock_hook.clone() as Arc<dyn EngineBindable>,
        );
        let wrapper: LoaderWrapper<TestEngine> = LoaderWrapper::new(
            inner.clone() as Arc<dyn ModelLoader<TestEngine>>,
            Arc::clone(&registry),
        );
        wrapper.set_pending_bind("acme/m1".to_string(), QuantType::Q4_K_M);

        let (path, cfg) = dummy_path_and_config();
        let result = wrapper.load(&path, &cfg);
        assert!(result.is_err(), "inner failure propagates");
        assert_eq!(mock_hook.bind_count(), 0, "no bind on load failure");
        // Pending slot intact (operator may retry).
        assert_eq!(
            wrapper.pending_bind(),
            Some(("acme/m1".to_string(), QuantType::Q4_K_M))
        );
    }

    // ===== Test 4 ==========================================================
    #[test]
    fn loader_wrapper_drive_unbind_calls_registry_unbind() {
        // Spec test 7: explicit unbind through the wrapper drives
        // the registry's unbind_for, firing unbind_engine on the hook.
        let inner = MockLoader::new();
        let registry = Arc::new(KvPersistRegistry::new());
        let mock_hook = LwMockBindable::new();
        registry.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            mock_hook.clone() as Arc<dyn EngineBindable>,
        );
        let wrapper: LoaderWrapper<TestEngine> = LoaderWrapper::new(
            inner as Arc<dyn ModelLoader<TestEngine>>,
            Arc::clone(&registry),
        );

        wrapper.drive_unbind("acme/m1", QuantType::Q4_K_M);
        assert_eq!(mock_hook.unbind_count(), 1);
        // No-op on unknown key.
        wrapper.drive_unbind("unknown/repo", QuantType::Q4_K_M);
        assert_eq!(mock_hook.unbind_count(), 1);
    }

    // ===== Test 5 ==========================================================
    #[test]
    fn loader_wrapper_clear_pending_bind_drops_slot() {
        // Defensive path: clear_pending_bind drops the slot without
        // a load_or_get. Subsequent load is a pass-through.
        let inner = MockLoader::new();
        let registry = Arc::new(KvPersistRegistry::new());
        let mock_hook = LwMockBindable::new();
        registry.register(
            "acme/m1".to_string(),
            QuantType::Q4_K_M,
            mock_hook.clone() as Arc<dyn EngineBindable>,
        );
        let wrapper: LoaderWrapper<TestEngine> = LoaderWrapper::new(
            inner as Arc<dyn ModelLoader<TestEngine>>,
            Arc::clone(&registry),
        );

        wrapper.set_pending_bind("acme/m1".to_string(), QuantType::Q4_K_M);
        wrapper.clear_pending_bind();
        assert_eq!(wrapper.pending_bind(), None);

        let (path, cfg) = dummy_path_and_config();
        let _engine = wrapper.load(&path, &cfg).expect("load");
        assert_eq!(mock_hook.bind_count(), 0, "no bind after clear");
    }

    // ===== Test 6 ==========================================================
    #[test]
    fn loader_wrapper_contract_violation_surfaces_error() {
        // When a hook retains a clone of the type-erased engine Arc
        // (violates the EngineBindable contract), the wrapper's
        // `Arc::try_unwrap` fails → wrapper returns Err. The error
        // mentions the (repo, quant) for operator-actionable reporting.
        let inner = MockLoader::new();
        let registry = Arc::new(KvPersistRegistry::new());
        let bad_hook = LwMockBindable::new_retaining();
        registry.register(
            "acme/bad".to_string(),
            QuantType::Q4_K_M,
            bad_hook.clone() as Arc<dyn EngineBindable>,
        );
        let wrapper: LoaderWrapper<TestEngine> = LoaderWrapper::new(
            inner as Arc<dyn ModelLoader<TestEngine>>,
            Arc::clone(&registry),
        );
        wrapper.set_pending_bind("acme/bad".to_string(), QuantType::Q4_K_M);

        let (path, cfg) = dummy_path_and_config();
        let result = wrapper.load(&path, &cfg);
        assert!(result.is_err(), "contract violation surfaces as Err");
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("acme/bad"), "error mentions repo: {msg}");
        // Bind fired before the error.
        assert_eq!(bad_hook.bind_count(), 1);

        // Cleanup: unbind clears the retained Arc so the test's
        // ref-count cycle doesn't leak.
        wrapper.drive_unbind("acme/bad", QuantType::Q4_K_M);
    }

    // ===== Test 7 ==========================================================
    #[test]
    fn loader_wrapper_set_pending_bind_overwrites_prior() {
        // Calling set_pending_bind twice without an intervening load
        // overwrites — the second pair is what fires.
        let inner = MockLoader::new();
        let registry = Arc::new(KvPersistRegistry::new());
        let mock_a = LwMockBindable::new();
        let mock_b = LwMockBindable::new();
        registry.register(
            "acme/a".to_string(),
            QuantType::Q4_K_M,
            mock_a.clone() as Arc<dyn EngineBindable>,
        );
        registry.register(
            "acme/b".to_string(),
            QuantType::Q4_K_M,
            mock_b.clone() as Arc<dyn EngineBindable>,
        );
        let wrapper: LoaderWrapper<TestEngine> = LoaderWrapper::new(
            inner as Arc<dyn ModelLoader<TestEngine>>,
            Arc::clone(&registry),
        );

        wrapper.set_pending_bind("acme/a".to_string(), QuantType::Q4_K_M);
        wrapper.set_pending_bind("acme/b".to_string(), QuantType::Q4_K_M);

        let (path, cfg) = dummy_path_and_config();
        let _engine = wrapper.load(&path, &cfg).expect("load");
        assert_eq!(mock_a.bind_count(), 0);
        assert_eq!(mock_b.bind_count(), 1);
    }
}
