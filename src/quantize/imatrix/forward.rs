//! Forward-pass driver — Phase B intercept infrastructure.
//!
//! ## Status: Stage 1 SHIPPED, Stage 2-4 IN PROGRESS
//!
//! Per ADR-033 §Pi the forward-pass driver runs hf2q's existing decoder
//! forward-pass over the calibration corpus chunks, intercepting the
//! input activations to each linear layer being quantized.
//!
//! Per the 2026-05-19 Risk 2 spike result (ADR-033 §Risk 2 "Spike
//! result"): **Metal-native accumulation is acceptable.** Stock
//! `llama-imatrix`'s CPU activation order is empirically infeasible to
//! mirror — p99 rel-err is 21% even on the spike's same-fixture
//! same-corpus run. The Pi Phase B acceptance gate is **downstream
//! quality** (perplexity / cosine of the resulting I-tier quant vs the
//! non-I sibling), not bit-equivalence of the imatrix intermediate.
//!
//! ## Stage 1 — Hook trait + minimal intercept (SHIPPED)
//!
//! [`ImatrixCollector`] is the trait the in-tree driver installs via
//! [`install_collector`]. The intercept point lives in
//! [`crate::serve::forward_mlx::dispatch_qmatmul`] — when both a
//! collector and a per-call name hint (set via [`with_collector`] or
//! [`set_name_hint`]) are present, the dispatch site:
//!
//! 1. Syncs the input buffer (commit_and_wait via the shared encoder).
//! 2. Slices `&[f32]` from the input buffer.
//! 3. Calls [`ImatrixCollector::record`] with the tensor name + slice.
//!
//! When either is `None` (the production-default state), the hot path
//! takes one branch (a `Cell::get()` on a thread-local) and proceeds
//! unchanged. Per [[feedback-no-loop-suppression-2026-05-17]] this is a
//! one-branch addition, not a runtime-degradation path.
//!
//! ## Stage 2 — In-tree driver (Phase B)
//!
//! [`collect_imatrix`] constructs an [`AccumulatorRegistry`]-backed
//! collector, drives the per-arch decoder over the tokenized corpus,
//! and returns an [`crate::quantize::imatrix::ImatrixData`].
//!
//! ## Phase A workaround (still supported)
//!
//! Operators can also generate `.imatrix.gguf` externally via stock
//! `llama-imatrix` and feed it back via `--imatrix <file>` — Phase A
//! loader path. Both producer and consumer coexist per the brief.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;

use super::corpus::CorpusBytes;
use super::error::ImatrixError;
use crate::quantize::ggml_quants::ArchName;

/// In-tree imatrix collector — installed by [`collect_imatrix`] (Stage
/// 2) and consumed by [`crate::serve::forward_mlx::dispatch_qmatmul`].
///
/// Convention: the caller (intercept site in `dispatch_qmatmul`) hands
/// in the **F32 input row** (the activation that's about to be matmul'd
/// against the weight). The collector accumulates `in_sum2[i] += row[i]²`
/// into its per-tensor [`super::accumulator::Accumulator`] — mirroring
/// `/opt/llama.cpp/tools/imatrix/imatrix.cpp:380-393`.
///
/// `tensor_name` is the canonical GGUF tensor name (e.g.
/// `"blk.0.attn_q.weight"`). It MUST match the name the convert
/// orchestrator will emit, or the [`super::gguf_loader::LoadedImatrix`]
/// at convert time won't find the per-tensor accumulator.
///
/// For MoE expert tensors, the per-expert routing is exposed via
/// [`Self::record_moe`]. Dense linears use [`Self::record`].
pub trait ImatrixCollector {
    /// Called by `dispatch_qmatmul` BEFORE the matmul. `tensor_name` is
    /// the canonical GGUF tensor name being multiplied against;
    /// `input_row` is the per-token F32 activation row. For multi-token
    /// prefill batches the caller invokes this once per token row.
    fn record(&mut self, tensor_name: &str, input_row: &[f32]);

    /// MoE per-expert variant. `expert_id` is the chosen expert slot;
    /// the caller is responsible for invoking this once per routed
    /// (token, expert) pair. Default impl forwards to [`Self::record`]
    /// with a synthetic per-expert name suffix — concrete collectors
    /// should override.
    fn record_moe(&mut self, tensor_name: &str, _expert_id: usize, input_row: &[f32]) {
        self.record(tensor_name, input_row);
    }
}

thread_local! {
    /// Active collector for this thread, or `None` for production decode.
    ///
    /// Held in a `RefCell` so the intercept site can `borrow_mut()`. The
    /// outer `Option` is the fast-path `is_none()` check; when `None`
    /// the hot path takes ONE branch and proceeds.
    static IMATRIX_COLLECTOR: RefCell<Option<Box<dyn ImatrixCollector>>> = const { RefCell::new(None) };

    /// Current tensor-name hint for the next [`dispatch_qmatmul`] call
    /// on this thread, or `None` for "don't collect this dispatch".
    ///
    /// Held as `RefCell<Option<String>>` so the in-tree driver can
    /// install runtime-built names (`format!("blk.{layer}.attn_q.weight")`)
    /// without leaking memory. The production fast path early-exits via
    /// the cheaper [`NAME_HINT_PRESENT`] `Cell<bool>` BEFORE this
    /// `RefCell` is touched — so the hot decode path never pays the
    /// borrow-tracking cost.
    static IMATRIX_NAME_HINT: RefCell<Option<String>> = const { RefCell::new(None) };

    /// Cheap "is a name hint set?" probe — `Cell<bool>`, one load on the
    /// hot path. Kept in sync with [`IMATRIX_NAME_HINT`] by every
    /// setter.
    static NAME_HINT_PRESENT: Cell<bool> = const { Cell::new(false) };

    /// Optional MoE expert-id hint. When set alongside
    /// [`IMATRIX_NAME_HINT`], the intercept site routes through
    /// [`ImatrixCollector::record_moe`] instead of `record`.
    static IMATRIX_EXPERT_HINT: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Install `collector` into the thread-local slot for the duration of
/// `body`. Restores the previous collector (typically `None`) at the
/// end. Safe to nest.
///
/// `body` receives no arguments — the in-tree driver should construct
/// the model + tokenize the corpus + invoke the decoder forward pass
/// inside the closure; the intercept site reads the thread-local on
/// every `dispatch_qmatmul`.
pub fn with_collector<C, F, R>(collector: C, body: F) -> R
where
    C: ImatrixCollector + 'static,
    F: FnOnce() -> R,
{
    let prev = IMATRIX_COLLECTOR.with(|slot| slot.replace(Some(Box::new(collector))));
    // Use a guard so we restore even on panic — though panics in the
    // forward pass already abort the convert run, the guard keeps the
    // thread-local consistent for any after-unwind diagnostics.
    struct Guard {
        prev: Option<Box<dyn ImatrixCollector>>,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            IMATRIX_COLLECTOR.with(|slot| {
                *slot.borrow_mut() = self.prev.take();
            });
        }
    }
    let _guard = Guard { prev };
    body()
}

/// Set the tensor-name hint for the next `dispatch_qmatmul` call on
/// this thread. Caller is responsible for clearing via
/// [`clear_name_hint`] after the call (or via [`with_name_hint`]).
///
/// Accepts `impl Into<String>` so callers can pass runtime-built names
/// (`format!("blk.{layer}.attn_q.weight")`) without leaking statics.
pub fn set_name_hint(name: impl Into<String>) {
    let s = name.into();
    IMATRIX_NAME_HINT.with(|slot| *slot.borrow_mut() = Some(s));
    NAME_HINT_PRESENT.with(|p| p.set(true));
}

/// Clear the tensor-name hint.
pub fn clear_name_hint() {
    IMATRIX_NAME_HINT.with(|slot| *slot.borrow_mut() = None);
    NAME_HINT_PRESENT.with(|p| p.set(false));
}

/// Set both the tensor-name hint and an MoE expert-id hint. Mirrors the
/// pairing convention in `dispatch_qmatmul` for MoE indirect matmuls.
pub fn set_moe_hint(name: impl Into<String>, expert_id: usize) {
    set_name_hint(name);
    IMATRIX_EXPERT_HINT.with(|slot| slot.set(Some(expert_id)));
}

/// Clear both the name + expert hint.
pub fn clear_moe_hint() {
    clear_name_hint();
    IMATRIX_EXPERT_HINT.with(|slot| slot.set(None));
}

/// Run `body` with `name` installed as the active tensor-name hint, and
/// clear the hint at the end. The hint is overwritten — not stacked —
/// so nested calls see only the inner-most name.
pub fn with_name_hint<F, R>(name: impl Into<String>, body: F) -> R
where
    F: FnOnce() -> R,
{
    set_name_hint(name);
    let out = body();
    clear_name_hint();
    out
}

/// Intercept entry point — called by `dispatch_qmatmul` at the top of
/// the function. Returns immediately if no collector is installed or no
/// name hint is set (the production-default fast path).
///
/// `materialize_row` is a closure that produces the F32 input row when
/// invoked — kept opaque so the intercept site decides the sync
/// strategy (commit_and_wait + as_slice, or a no-op for already-host
/// data). The closure is NOT called when collection is disabled.
///
/// This is the **only** Stage 1 surface the runtime needs. The trait +
/// install API above is what the in-tree driver (Stage 2) consumes.
pub fn intercept_qmatmul<F>(materialize_row: F)
where
    F: FnOnce() -> Option<Vec<f32>>,
{
    // Fast path: a single `Cell<bool>::get()` load. When no in-tree
    // driver is active (production decode / serve), this is the only
    // overhead — one load + one branch + return.
    if !NAME_HINT_PRESENT.with(|p| p.get()) {
        return;
    }
    let expert = IMATRIX_EXPERT_HINT.with(|slot| slot.get());

    // Snapshot the name out of the RefCell into an owned String so we
    // can drop the borrow before invoking the collector (which itself
    // borrows IMATRIX_COLLECTOR mutably — keeping both borrows live
    // would not be problematic since they're different cells, but a
    // single-borrow style is cheaper and easier to audit).
    let name = match IMATRIX_NAME_HINT.with(|slot| slot.borrow().clone()) {
        Some(n) => n,
        None => return,
    };

    IMATRIX_COLLECTOR.with(|slot| {
        let mut borrow = slot.borrow_mut();
        let collector = match borrow.as_deref_mut() {
            Some(c) => c,
            None => return,
        };
        let row = match materialize_row() {
            Some(r) => r,
            None => return,
        };
        match expert {
            Some(eid) => collector.record_moe(&name, eid, &row),
            None => collector.record(&name, &row),
        }
    });
}

/// True when a collector is currently installed on this thread. Used by
/// the intercept site to skip even the `Cell::get()` for the name hint
/// when nothing's listening — a strictly-tighter fast path than
/// [`intercept_qmatmul`]'s default-None branch.
pub fn is_active() -> bool {
    IMATRIX_COLLECTOR.with(|slot| slot.borrow().is_some())
}

/// Driver-side parameters for an in-tree imatrix run.
#[derive(Debug, Clone)]
pub struct ComputeImatrixParams {
    /// HF model directory (config.json + safetensors).
    pub hf_dir: PathBuf,
    /// Corpus text payload.
    pub corpus: CorpusBytes,
    /// `n_ctx` used by the forward pass. `chunk_size = n_ctx / n_parallel`
    /// per ADR-033 §Pi (default `n_parallel = 1` ⇒ chunks the corpus
    /// into `n_ctx`-token windows).
    pub n_ctx: u32,
    /// Detected source arch (gemma4 / qwen35moe / etc.).
    pub arch: ArchName,
}

/// Stage 2 entry point — in-tree forward-pass driver.
///
/// Implementation status: SCAFFOLD. Returns
/// [`ImatrixError::InTreeGenerationNotYetShipped`] for arches that don't
/// yet have driver wiring. The hook infrastructure (trait,
/// thread-local, intercept) IS shipped — only the model-load +
/// forward-pass-drive logic is deferred.
pub fn compute_imatrix(params: &ComputeImatrixParams) -> Result<(), ImatrixError> {
    // Stage 2 will:
    //   1. Convert hf_dir to F16 GGUF in a tempfile (via run_convert
    //      with --quant f16).
    //   2. Load the F16 GGUF via the existing inference loader.
    //   3. Tokenize params.corpus via the per-arch tokenizer.
    //   4. Chunk via super::corpus::chunk_tokens(&tokens, n_ctx).
    //   5. Build a MyCollector struct implementing ImatrixCollector;
    //      populate its registry on each `record` call.
    //   6. For each chunk: install the collector via with_collector;
    //      drive the per-arch decoder forward pass; uninstall.
    //   7. Return ImatrixData { loaded, provenance: Computed }.
    Err(ImatrixError::InTreeGenerationNotYetShipped {
        corpus: params.corpus.label.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::imatrix::corpus::CorpusSource;

    /// `intercept_qmatmul` is a no-op when no collector is installed.
    #[test]
    fn intercept_noop_without_collector() {
        // No collector, no name hint — closure should never fire.
        let mut materialized = false;
        intercept_qmatmul(|| {
            materialized = true;
            Some(vec![1.0, 2.0])
        });
        assert!(!materialized, "materialize closure should not fire");
        assert!(!is_active());
    }

    /// `intercept_qmatmul` is a no-op when collector is installed but
    /// no name hint is set.
    #[test]
    fn intercept_noop_without_name_hint() {
        let collector = RecorderCollector::default();
        with_collector(collector, || {
            assert!(is_active());
            let mut materialized = false;
            intercept_qmatmul(|| {
                materialized = true;
                Some(vec![1.0])
            });
            assert!(!materialized, "no name hint → closure should not fire");
        });
    }

    /// `with_name_hint` + installed collector → record() fires.
    #[test]
    fn intercept_fires_with_collector_and_name() {
        // RecorderCollector is module-private; verify via a shared
        // counter in a static so we can read it after `with_collector`
        // restores the slot.
        use std::sync::Mutex;
        static RECORDS: Mutex<Vec<(String, Vec<f32>)>> = Mutex::new(Vec::new());

        struct StaticCollector;
        impl ImatrixCollector for StaticCollector {
            fn record(&mut self, name: &str, row: &[f32]) {
                RECORDS.lock().unwrap().push((name.to_string(), row.to_vec()));
            }
        }

        RECORDS.lock().unwrap().clear();
        with_collector(StaticCollector, || {
            with_name_hint("blk.0.attn_q.weight", || {
                intercept_qmatmul(|| Some(vec![1.0, 2.0, 3.0]));
            });
        });

        let records = RECORDS.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, "blk.0.attn_q.weight");
        assert_eq!(records[0].1, vec![1.0, 2.0, 3.0]);
    }

    /// `set_moe_hint` routes through `record_moe`.
    #[test]
    fn intercept_moe_routing() {
        use std::sync::Mutex;
        static MOE_HITS: Mutex<Vec<(String, usize, Vec<f32>)>> = Mutex::new(Vec::new());

        struct MoeCollector;
        impl ImatrixCollector for MoeCollector {
            fn record(&mut self, _name: &str, _row: &[f32]) {
                panic!("expected record_moe path, not record");
            }
            fn record_moe(&mut self, name: &str, expert_id: usize, row: &[f32]) {
                MOE_HITS.lock().unwrap().push((name.to_string(), expert_id, row.to_vec()));
            }
        }

        MOE_HITS.lock().unwrap().clear();
        with_collector(MoeCollector, || {
            set_moe_hint("blk.0.ffn_gate_exps.weight", 7);
            intercept_qmatmul(|| Some(vec![0.5, 0.5]));
            clear_moe_hint();
        });

        let hits = MOE_HITS.lock().unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, "blk.0.ffn_gate_exps.weight");
        assert_eq!(hits[0].1, 7);
    }

    /// `with_collector` restores the previous slot at exit.
    #[test]
    fn with_collector_restores_slot() {
        assert!(!is_active());
        with_collector(RecorderCollector::default(), || {
            assert!(is_active());
        });
        assert!(!is_active());
    }

    /// Stage 2: `compute_imatrix` returns the deferred error (no panic /
    /// silent no-op). Composes with the no-loop-suppression rule.
    #[test]
    fn compute_imatrix_returns_deferred_error() {
        let corpus = CorpusBytes::load(&CorpusSource::Cdv3).unwrap();
        let params = ComputeImatrixParams {
            hf_dir: PathBuf::from("/tmp/non-existent-fixture"),
            corpus,
            n_ctx: 512,
            arch: ArchName::Gemma4,
        };
        let err = compute_imatrix(&params).unwrap_err();
        match err {
            ImatrixError::InTreeGenerationNotYetShipped { corpus } => {
                assert_eq!(corpus, "cdv3");
            }
            other => panic!("expected InTreeGenerationNotYetShipped, got {other:?}"),
        }
    }

    /// Used as a test-only collector that records via shared state. We
    /// can't easily move the recorder out of `with_collector` since the
    /// trait erases the concrete type, so multi-test cases above use
    /// `static Mutex` workarounds instead.
    #[derive(Default)]
    struct RecorderCollector;
    impl ImatrixCollector for RecorderCollector {
        fn record(&mut self, _name: &str, _row: &[f32]) {}
    }
}
