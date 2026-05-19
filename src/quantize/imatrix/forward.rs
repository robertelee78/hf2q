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

use std::cell::RefCell;
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
/// For MoE FUSED tensors (`ffn_gate_up_exps.weight`,
/// `ffn_down_exps.weight`), the same input row is read by ALL routed
/// experts in one dispatch — canonical llama-imatrix accumulates
/// **per-input-row** sum-of-squares, which is invariant under which
/// expert reads it. So one [`Self::record`] call per fused-MoE
/// dispatch is correct.
pub trait ImatrixCollector {
    /// Called by `dispatch_qmatmul` BEFORE the matmul. `tensor_name`
    /// is the canonical GGUF tensor name being multiplied against;
    /// `input_row` is the per-token F32 activation row. For multi-token
    /// prefill batches the caller invokes this once per token row.
    fn record(&mut self, tensor_name: &str, input_row: &[f32]);
}

thread_local! {
    /// Active collector for this thread, or `None` for production decode.
    ///
    /// Held in a `RefCell` so the intercept site can `borrow_mut()`. The
    /// outer `Option` is the fast-path `is_none()` check; when `None`
    /// the hot path takes ONE branch and proceeds.
    static IMATRIX_COLLECTOR: RefCell<Option<Box<dyn ImatrixCollector>>> = const { RefCell::new(None) };
}

/// Cheap-to-construct hint that gets lazily formatted into a canonical
/// GGUF tensor name when (and only when) an [`ImatrixCollector`] is
/// active. The intercept site reads this enum inline — no thread-local
/// String, no `format!` allocation on the production fast path.
///
/// Stage 2 plumbing replaced Stage 1's thread-local `IMATRIX_NAME_HINT`
/// with this inline-hint API. Per [[feedback-no-backwards-compat-2026-05-18]]
/// the thread-local API has been deleted, not aliased.
#[derive(Debug, Clone, Copy)]
pub enum ImatrixHint<'a> {
    /// Skip — intercept is a no-op regardless of collector state. Use
    /// for matmuls whose inputs are post-RoPE / post-norm activations
    /// being read by SDPA (the `sdpa_out`-driven `o_proj` is named,
    /// but the *output* projection's "input" is the attention's output
    /// row — capturing it doesn't help an imatrix of the o_proj weight
    /// itself; the relevant capture is upstream).
    None,
    /// Global tensor: GGUF name is exactly `name` (no formatting).
    /// Example: `ImatrixHint::Global("token_embd.weight")`.
    Global(&'a str),
    /// Per-block tensor: GGUF name is `"blk.{layer}.{tag}.weight"`.
    /// `tag` is the canonical GGUF middle slot — e.g., `"attn_q"`,
    /// `"attn_k"`, `"attn_v"`, `"attn_output"`, `"ffn_gate"`,
    /// `"ffn_up"`, `"ffn_down"`, `"ffn_gate_inp"`,
    /// `"ffn_gate_up_exps"`, `"ffn_down_exps"`.
    Layered { tag: &'a str, layer: usize },
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

/// Intercept entry point — called by [`crate::serve::forward_mlx::dispatch_qmatmul`]
/// at the top of the function. Returns immediately if no collector is
/// installed (the production-default fast path).
///
/// `hint` carries the canonical GGUF tensor name (or `None` to skip
/// this dispatch). `materialize_row` is a closure that produces the
/// F32 input row when invoked — kept opaque so the intercept site
/// decides the sync strategy (`commit_and_wait + as_slice`, or a no-op
/// for already-host data). The closure is NOT called when collection
/// is disabled.
///
/// Fast path overhead: one `RefCell::borrow().is_none()` check (one
/// load + branch). Per [[feedback-no-loop-suppression-2026-05-17]] this
/// is a one-branch addition, not a runtime-degradation path.
pub fn intercept_qmatmul_with_hint<F>(hint: ImatrixHint<'_>, materialize_row: F)
where
    F: FnOnce() -> Option<Vec<f32>>,
{
    // Fast path: if no collector installed, return immediately.
    if !is_active() {
        return;
    }
    // No allocation until we know the collector wants this dispatch.
    let name = match hint {
        ImatrixHint::None => return,
        ImatrixHint::Global(s) => s.to_string(),
        ImatrixHint::Layered { tag, layer } => format!("blk.{layer}.{tag}.weight"),
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
        collector.record(&name, &row);
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

    /// `intercept_qmatmul_with_hint` is a no-op when no collector is installed.
    #[test]
    fn intercept_noop_without_collector() {
        let mut materialized = false;
        intercept_qmatmul_with_hint(
            ImatrixHint::Layered { tag: "attn_q", layer: 0 },
            || {
                materialized = true;
                Some(vec![1.0, 2.0])
            },
        );
        assert!(!materialized, "materialize closure should not fire");
        assert!(!is_active());
    }

    /// `intercept_qmatmul_with_hint(None, ...)` is a no-op even with collector
    /// installed (used for non-imatrix-tracked matmuls).
    #[test]
    fn intercept_noop_with_none_hint() {
        let collector = RecorderCollector::default();
        with_collector(collector, || {
            assert!(is_active());
            let mut materialized = false;
            intercept_qmatmul_with_hint(ImatrixHint::None, || {
                materialized = true;
                Some(vec![1.0])
            });
            assert!(!materialized, "None hint → closure should not fire");
        });
    }

    /// `Layered` hint + installed collector → record() fires with the
    /// formatted canonical GGUF name.
    #[test]
    fn intercept_fires_with_collector_and_layered_hint() {
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
            intercept_qmatmul_with_hint(
                ImatrixHint::Layered { tag: "attn_q", layer: 0 },
                || Some(vec![1.0, 2.0, 3.0]),
            );
        });

        let records = RECORDS.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, "blk.0.attn_q.weight");
        assert_eq!(records[0].1, vec![1.0, 2.0, 3.0]);
    }

    /// `Global` hint records under the verbatim name (no formatting).
    #[test]
    fn intercept_global_hint_records_verbatim() {
        use std::sync::Mutex;
        static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

        struct C;
        impl ImatrixCollector for C {
            fn record(&mut self, name: &str, _row: &[f32]) {
                RECORDS.lock().unwrap().push(name.to_string());
            }
        }

        RECORDS.lock().unwrap().clear();
        with_collector(C, || {
            intercept_qmatmul_with_hint(ImatrixHint::Global("token_embd.weight"), || {
                Some(vec![0.0; 4])
            });
        });
        let r = RECORDS.lock().unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], "token_embd.weight");
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
