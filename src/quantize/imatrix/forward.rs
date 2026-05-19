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
/// Two intercept paths feed this trait:
///
///   - [`intercept_qmatmul_with_hint`] — dense matmuls
///     (`dispatch_qmatmul`). Calls [`Self::record`] once per token
///     row.
///   - [`intercept_qmatmul_id_with_hint`] — MoE fused matmuls
///     (`quantized_matmul_id_ggml_pooled`,
///     [`mlx_native::GgmlQuantizedMatmulIdParams`]). Calls
///     [`Self::record_moe`] once per (token, routed-expert) pair —
///     up to `top_k` calls per token, with the SAME input row but a
///     different `expert_id` argument each time.
///
/// Both intercepts deliver exactly one token row per call. The MoE
/// per-expert split mirrors canonical `imatrix.cpp:310-330` for
/// `GGML_OP_MUL_MAT_ID`: each routed expert sees the activation in
/// its own per-expert `Accumulator` slot at
/// `values[expert_id * n_per_row + j]`.
///
/// Required methods: implementations MUST define both. Dense-only
/// test collectors typically impl `record_moe` as a panic / no-op;
/// MoE-aware production collectors dispatch via
/// [`super::accumulator::Accumulator::absorb_moe`].
pub trait ImatrixCollector {
    /// Called by `intercept_qmatmul_with_hint` BEFORE the matmul.
    /// `tensor_name` is the canonical GGUF tensor name being
    /// multiplied against; `input_row` is exactly ONE token's F32
    /// activation row, length `n_per_row`. The intercept site is
    /// responsible for slicing the m-row prefill buffer into
    /// per-token rows before invoking this — implementations can
    /// assume every call delivers exactly one row.
    fn record(&mut self, tensor_name: &str, input_row: &[f32]);

    /// Called by `intercept_qmatmul_id_with_hint` BEFORE the MoE
    /// fused matmul, ONCE per (token, routed-expert) pair.
    /// `tensor_name` is the canonical GGUF name (e.g.
    /// `"blk.5.ffn_gate_up_exps.weight"`); `expert_id` is the index
    /// of the routed expert in `0..n_experts`; `input_row` is the
    /// shared per-token F32 activation row (same value for every
    /// routed expert of that token).
    ///
    /// Mirrors `imatrix.cpp:310-330` for `GGML_OP_MUL_MAT_ID`. The
    /// collector typically stores per-expert sum-of-squares at
    /// `values[expert_id * n_per_row + j]` and bumps
    /// `counts[expert_id] += 1` per call.
    fn record_moe(&mut self, tensor_name: &str, expert_id: usize, input_row: &[f32]);
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
/// this dispatch). `m` is the number of token rows in the input buffer
/// (i.e. the M dimension of the matmul; decode m=1, prefill m=seq_len).
/// `n_per_row` is the per-row activation width (K dimension of the
/// matmul; equal to `weight.info.cols`). `materialize_buffer` is a
/// closure that produces the FULL F32 input as a single `Vec<f32>` of
/// length `m * n_per_row` when invoked — kept opaque so the intercept
/// site decides the sync strategy (`commit_and_wait + as_slice`, or a
/// no-op for already-host data). The closure is NOT called when
/// collection is disabled.
///
/// Per-row dispatch: the intercept slices the materialized buffer into
/// `m` contiguous chunks of `n_per_row` and calls
/// [`ImatrixCollector::record`] once per token row. This matches the
/// canonical llama-imatrix semantics at
/// `/opt/llama.cpp/tools/imatrix/imatrix.cpp:380-393` where the
/// per-row sum-of-squares accumulator advances `counts[mat_id] += 1`
/// per absorbed row — NOT once per dispatch.
///
/// If the materialized buffer length doesn't equal `m * n_per_row` the
/// intercept returns [`ImatrixError::ShapeMismatch`]. Per the codex
/// review 2026-05-19 + [[feedback-no-loop-suppression-2026-05-17]] this
/// is a typed error, not a silent skip — silently dropping activation
/// data would bias the imatrix output. The `dispatch_qmatmul` caller
/// propagates as an `anyhow::Error` and the forward pass aborts loudly
/// so the operator sees the wiring bug.
///
/// Fast path overhead: one `RefCell::borrow().is_none()` check (one
/// load + branch). Per [[feedback-no-loop-suppression-2026-05-17]] this
/// is a one-branch addition, not a runtime-degradation path.
pub fn intercept_qmatmul_with_hint<F>(
    hint: ImatrixHint<'_>,
    m: usize,
    n_per_row: usize,
    materialize_buffer: F,
) -> Result<(), ImatrixError>
where
    F: FnOnce() -> Option<Vec<f32>>,
{
    // Fast path: if no collector installed, return immediately.
    if !is_active() {
        return Ok(());
    }
    // No allocation until we know the collector wants this dispatch.
    let name = match hint {
        ImatrixHint::None => return Ok(()),
        ImatrixHint::Global(s) => s.to_string(),
        ImatrixHint::Layered { tag, layer } => format!("blk.{layer}.{tag}.weight"),
    };

    IMATRIX_COLLECTOR.with(|slot| -> Result<(), ImatrixError> {
        let mut borrow = slot.borrow_mut();
        let collector = match borrow.as_deref_mut() {
            Some(c) => c,
            None => return Ok(()),
        };
        let buf = match materialize_buffer() {
            Some(r) => r,
            None => return Ok(()),
        };
        let expected = m.saturating_mul(n_per_row);
        if buf.len() != expected {
            return Err(ImatrixError::ShapeMismatch {
                tensor: name,
                m,
                n_per_row,
                got: buf.len(),
                expected,
            });
        }
        if n_per_row == 0 || m == 0 {
            // Zero-row dispatch — nothing to absorb, not an error.
            return Ok(());
        }
        // imatrix.cpp:380-393 — accumulate per token row.
        for row in buf.chunks_exact(n_per_row) {
            collector.record(&name, row);
        }
        Ok(())
    })
}

/// True when a collector is currently installed on this thread. Used by
/// the intercept site to skip even the `Cell::get()` for the name hint
/// when nothing's listening — a strictly-tighter fast path than
/// [`intercept_qmatmul`]'s default-None branch.
pub fn is_active() -> bool {
    IMATRIX_COLLECTOR.with(|slot| slot.borrow().is_some())
}

/// Intercept entry point for MoE FUSED matmuls dispatched through
/// `mlx_native::quantized_matmul_id_ggml_pooled` (the
/// [`mlx_native::GgmlQuantizedMatmulIdParams`] path used by Qwen3.5/3.6
/// MoE and Gemma 4-A4B's MoE expert dispatches). The dense intercept
/// [`intercept_qmatmul_with_hint`] does NOT see these — they bypass
/// `dispatch_qmatmul` entirely.
///
/// Contract per `imatrix.cpp:310-330` (canonical llama-imatrix for
/// `GGML_OP_MUL_MAT_ID`):
///
/// ```text
/// for each token in 0..n_tokens:
///   row = input_buffer[token * n_per_row..(token+1) * n_per_row]
///   for j in 0..top_k:
///     expert_id = expert_ids[token * top_k + j]
///     e.values[expert_id * n_per_row + col] += row[col]² for col
///     e.counts[expert_id] += 1
/// ```
///
/// The intercept fires [`ImatrixCollector::record_moe`] exactly
/// `n_tokens * top_k` times: once per (token, routed-expert) pair,
/// with the SAME row but a different `expert_id` each iteration.
///
/// Materialization closures:
///   - `materialize_input` produces the F32 input buffer as a single
///     `Vec<f32>` of length `n_tokens * n_per_row`. The intercept
///     site does `commit_and_wait + as_slice::<f32>()` on the input
///     MlxBuffer (same as the dense path).
///   - `materialize_expert_ids` produces the routing buffer as a
///     `Vec<u32>` of length `n_tokens * top_k`. Read from
///     `moe_expert_ids`; that buffer is populated upstream by
///     `fused_moe_routing_f32`.
///
/// Either closure returning `None` causes the intercept to silently
/// skip this dispatch — the production materialization path can fail
/// if the GPU encoder is in a weird state, and one missing dispatch
/// is recoverable (the next chunk's data still feeds the imatrix).
/// Shape mismatches (closures return wrong-size buffers) ARE typed
/// errors per the no-loop-suppression rule.
///
/// Fast path: same single `is_active()` load + branch as the dense
/// intercept; closures are only invoked when a collector is installed
/// and the hint is non-`None`.
pub fn intercept_qmatmul_id_with_hint<FInput, FIds>(
    hint: ImatrixHint<'_>,
    n_tokens: usize,
    top_k: usize,
    n_per_row: usize,
    materialize_input: FInput,
    materialize_expert_ids: FIds,
) -> Result<(), ImatrixError>
where
    FInput: FnOnce() -> Option<Vec<f32>>,
    FIds: FnOnce() -> Option<Vec<u32>>,
{
    if !is_active() {
        return Ok(());
    }
    let name = match hint {
        ImatrixHint::None => return Ok(()),
        ImatrixHint::Global(s) => s.to_string(),
        ImatrixHint::Layered { tag, layer } => format!("blk.{layer}.{tag}.weight"),
    };

    IMATRIX_COLLECTOR.with(|slot| -> Result<(), ImatrixError> {
        let mut borrow = slot.borrow_mut();
        let collector = match borrow.as_deref_mut() {
            Some(c) => c,
            None => return Ok(()),
        };

        let input = match materialize_input() {
            Some(b) => b,
            None => return Ok(()),
        };
        let expert_ids = match materialize_expert_ids() {
            Some(b) => b,
            None => return Ok(()),
        };

        let expected_input = n_tokens.saturating_mul(n_per_row);
        if input.len() != expected_input {
            return Err(ImatrixError::ShapeMismatch {
                tensor: name,
                m: n_tokens,
                n_per_row,
                got: input.len(),
                expected: expected_input,
            });
        }
        let expected_ids = n_tokens.saturating_mul(top_k);
        if expert_ids.len() != expected_ids {
            return Err(ImatrixError::ShapeMismatch {
                tensor: format!("{name}::expert_ids"),
                m: n_tokens,
                n_per_row: top_k,
                got: expert_ids.len(),
                expected: expected_ids,
            });
        }

        if n_tokens == 0 || n_per_row == 0 || top_k == 0 {
            return Ok(());
        }

        // imatrix.cpp:310-330 — for each routed expert of each token,
        // accumulate the SAME shared row into the per-expert slot.
        for tok in 0..n_tokens {
            let row = &input[tok * n_per_row..(tok + 1) * n_per_row];
            for k_idx in 0..top_k {
                let expert_id = expert_ids[tok * top_k + k_idx] as usize;
                collector.record_moe(&name, expert_id, row);
            }
        }
        Ok(())
    })
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

/// Stage 3 — in-tree forward-pass driver for imatrix generation.
///
/// Pipeline (per ADR-033 §Pi):
///   1. Convert `hf_dir` to a temporary F16 GGUF via `run_convert`.
///   2. Load the F16 GGUF via `LoadedModel::load` (Gemma 4 only for
///      Stage 3.0; Qwen35Moe and others surface
///      [`ImatrixError::UnsupportedArchForDriver`]).
///   3. Tokenize `params.corpus` via the model's tokenizer.
///   4. Chunk tokens into `params.n_ctx`-sized windows via
///      [`super::corpus::chunk_tokens`]; partial trailing chunks
///      dropped (mirrors `imatrix.cpp:960`).
///   5. For each chunk: install a
///      [`AccumulatorRegistry`]-backed collector via
///      [`with_collector`]; call `forward_prefill(chunk, 1, &mut ctx)`;
///      collector is automatically dropped on scope exit.
///   6. Pack the accumulated registry into [`super::ImatrixData`]
///      with [`super::ImatrixProvenance::Computed`] provenance.
///
/// All failure modes surface as typed [`ImatrixError`] variants
/// (ConvertFailed / ModelLoadFailed / UnsupportedArchForDriver /
/// TokenizationFailed / ForwardPassFailed / CorpusTooShort) per the
/// no-loop-suppression rule.
///
/// Cost note: a single chunk of `forward_prefill` on a 26B-A4B Gemma
/// model takes several seconds; a full cdv3 corpus (~50k tokens at
/// n_ctx=512 ⇒ ~100 chunks) is operator-time, not CI-time. The
/// driver intentionally has no per-test fixture — operators invoke
/// it via `hf2q convert <hf-dir> --quant apex-i-balanced
/// --imatrix-corpus cdv3` once Stage 3c.2 wires the CLI surface.
pub fn compute_imatrix(
    params: &ComputeImatrixParams,
) -> Result<super::ImatrixData, ImatrixError> {
    use crate::quantize::ggml_quants::ArchName as Arch;
    use std::sync::{Arc, Mutex};

    // ---- 1. Validate input + create tempdir for F16 GGUF ---------------
    if !params.hf_dir.is_dir() {
        return Err(ImatrixError::ConvertFailed {
            detail: format!(
                "hf_dir `{}` does not exist or is not a directory",
                params.hf_dir.display()
            ),
        });
    }
    // Only Gemma 4 is wired for the driver in Stage 3.0. Other arches
    // surface a typed error pointing at the supported set.
    if !matches!(params.arch, Arch::Gemma4) {
        return Err(ImatrixError::UnsupportedArchForDriver {
            arch: params.arch.name().to_string(),
            supported: &["gemma4"],
        });
    }
    let tmp = tempfile::tempdir().map_err(ImatrixError::Io)?;
    let f16_path = tmp.path().join("model.f16.gguf");

    // ---- 2. Run convert to F16 GGUF -----------------------------------
    let convert_args = crate::convert::cli_driver::ConvertArgs {
        hf_dir: params.hf_dir.clone(),
        selector: crate::convert::quant_selector::QuantSelector::Standard(
            crate::quantize::ggml_quants::llama_ftype::LlamaFtype::MostlyF16,
        ),
        output: f16_path.clone(),
        imatrix: None,
        imatrix_corpus: None,
        imatrix_out: None,
    };
    crate::convert::cli_driver::run_convert(convert_args).map_err(|e| {
        ImatrixError::ConvertFailed {
            detail: format!("{e:?}"),
        }
    })?;

    // ---- 3. Load model via existing inference loader -------------------
    let load_opts = crate::serve::api::engine::LoadOptions {
        model_path: f16_path.clone(),
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
        kv_persist_dir: None,
    };
    let loaded = crate::serve::api::engine::LoadedModel::load(&load_opts).map_err(|e| {
        ImatrixError::ModelLoadFailed {
            detail: format!("{e:?}"),
        }
    })?;
    let mut gemma = match loaded {
        crate::serve::api::engine::LoadedModel::Gemma(g) => g,
        // Stage 3.0 only supports Gemma 4. Qwen35Moe / Qwen3VlText
        // would land here but we caught them at the arch-validate
        // check above; this arm is defensive against arch
        // double-detection drift.
        _ => {
            return Err(ImatrixError::UnsupportedArchForDriver {
                arch: format!("{:?}", params.arch),
                supported: &["gemma4"],
            })
        }
    };
    let n_experts = gemma.config.num_experts;

    // ---- 4. Tokenize corpus -------------------------------------------
    let encoding =
        gemma
            .tokenizer
            .encode(params.corpus.text.as_str(), false)
            .map_err(|e| ImatrixError::TokenizationFailed {
                detail: format!("{e:?}"),
            })?;
    let tokens: Vec<u32> = encoding.get_ids().to_vec();

    // ---- 5. Chunk -----------------------------------------------------
    let chunks = super::corpus::chunk_tokens(&tokens, params.n_ctx as usize);
    if chunks.is_empty() {
        return Err(ImatrixError::CorpusTooShort {
            corpus_label: params.corpus.label.clone(),
            token_count: tokens.len(),
            n_ctx: params.n_ctx,
        });
    }
    let chunk_count = chunks.len();

    // ---- 6. Build shared collector ------------------------------------
    let registry = Arc::new(Mutex::new(super::accumulator::AccumulatorRegistry::new()));

    /// MoE-aware collector backed by a shared `AccumulatorRegistry`.
    /// Registers each tensor lazily on first record; dense tensors
    /// get `n_mat=1`, MoE-routed tensors get `n_mat=n_experts`.
    /// Per [[feedback-no-loop-suppression-2026-05-17]]: shape /
    /// expert-id violations panic the forward pass loudly rather
    /// than silently corrupt the accumulator — collection is one-
    /// shot operator-time, recovery doesn't help.
    struct SharedCollector {
        registry: Arc<Mutex<super::accumulator::AccumulatorRegistry>>,
        n_experts: usize,
    }
    impl ImatrixCollector for SharedCollector {
        fn record(&mut self, name: &str, row: &[f32]) {
            let mut reg = self.registry.lock().expect("imatrix registry mutex poisoned");
            let acc = reg
                .register(name, row.len(), 1)
                .expect("imatrix dense register: shape mismatch (re-register with different n_per_row)");
            acc.absorb_dense(row)
                .expect("imatrix dense absorb: row length mismatch (intercept should have caught)");
        }
        fn record_moe(&mut self, name: &str, expert_id: usize, row: &[f32]) {
            let mut reg = self.registry.lock().expect("imatrix registry mutex poisoned");
            let acc = reg
                .register(name, row.len(), self.n_experts)
                .expect("imatrix moe register: shape mismatch (re-register with different shape)");
            acc.absorb_moe(expert_id, row)
                .expect("imatrix moe absorb: expert_id out of range or row mismatch");
        }
    }

    // ---- 7. Drive forward pass over each chunk ------------------------
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let collector = SharedCollector {
            registry: Arc::clone(&registry),
            n_experts,
        };
        // forward_prefill returns the argmax of the last-row logits.
        // We don't need it for imatrix; the activations were captured
        // via the installed collector during the prefill itself.
        let result: anyhow::Result<u32> = with_collector(collector, || {
            gemma
                .weights
                .forward_prefill(chunk, /* max_decode_tokens */ 1, &mut gemma.ctx)
        });
        result.map_err(|e| ImatrixError::ForwardPassFailed {
            chunk_index,
            chunk_count,
            detail: format!("{e:?}"),
        })?;
    }

    // ---- 8. Pack into ImatrixData -------------------------------------
    let registry = Arc::try_unwrap(registry)
        .map_err(|_| ImatrixError::ForwardPassFailed {
            chunk_index: 0,
            chunk_count: 0,
            detail:
                "internal: registry Arc had outstanding clones at pack time (collector leak)"
                    .to_string(),
        })?
        .into_inner()
        .expect("imatrix registry mutex poisoned");
    let loaded = super::gguf_loader::LoadedImatrix {
        source_path: format!("<computed:{}>", params.corpus.label),
        datasets: vec![params.corpus.label.clone()],
        chunk_count: chunk_count as u32,
        chunk_size: params.n_ctx,
        registry,
    };
    Ok(super::ImatrixData {
        loaded,
        provenance: super::ImatrixProvenance::Computed {
            corpus_label: params.corpus.label.clone(),
            n_ctx: params.n_ctx,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::imatrix::corpus::CorpusSource;

    /// `intercept_qmatmul_with_hint` is a no-op (Ok) when no collector
    /// is installed.
    #[test]
    fn intercept_noop_without_collector() {
        let mut materialized = false;
        let result = intercept_qmatmul_with_hint(
            ImatrixHint::Layered { tag: "attn_q", layer: 0 },
            /* m */ 1,
            /* n_per_row */ 2,
            || {
                materialized = true;
                Some(vec![1.0, 2.0])
            },
        );
        assert!(result.is_ok(), "no-collector path returns Ok");
        assert!(!materialized, "materialize closure should not fire");
        assert!(!is_active());
    }

    /// `intercept_qmatmul_with_hint(None, ...)` is a no-op (Ok) even
    /// with collector installed (used for non-imatrix-tracked matmuls).
    #[test]
    fn intercept_noop_with_none_hint() {
        let collector = RecorderCollector::default();
        with_collector(collector, || {
            assert!(is_active());
            let mut materialized = false;
            let result = intercept_qmatmul_with_hint(
                ImatrixHint::None,
                /* m */ 1,
                /* n_per_row */ 1,
                || {
                    materialized = true;
                    Some(vec![1.0])
                },
            );
            assert!(result.is_ok(), "None hint returns Ok");
            assert!(!materialized, "None hint → closure should not fire");
        });
    }

    /// `Layered` hint + installed collector + m=1 → record() fires
    /// exactly once with the formatted canonical GGUF name and the
    /// full single-row slice; returns Ok.
    #[test]
    fn intercept_fires_with_collector_and_layered_hint() {
        use std::sync::Mutex;
        static RECORDS: Mutex<Vec<(String, Vec<f32>)>> = Mutex::new(Vec::new());

        struct StaticCollector;
        impl ImatrixCollector for StaticCollector {
            fn record(&mut self, name: &str, row: &[f32]) {
                RECORDS.lock().unwrap().push((name.to_string(), row.to_vec()));
            }
            fn record_moe(&mut self, _name: &str, _expert_id: usize, _row: &[f32]) {
                unreachable!("dense-only test collector — record_moe not exercised");
            }
        }

        RECORDS.lock().unwrap().clear();
        with_collector(StaticCollector, || {
            let result = intercept_qmatmul_with_hint(
                ImatrixHint::Layered { tag: "attn_q", layer: 0 },
                /* m */ 1,
                /* n_per_row */ 3,
                || Some(vec![1.0, 2.0, 3.0]),
            );
            assert!(result.is_ok());
        });

        let records = RECORDS.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, "blk.0.attn_q.weight");
        assert_eq!(records[0].1, vec![1.0, 2.0, 3.0]);
    }

    /// Multi-token prefill (m > 1) → record() fires once per token row
    /// with the per-row slice of length n_per_row. Mirrors canonical
    /// llama-imatrix per-row accumulation (imatrix.cpp:380-393).
    #[test]
    fn intercept_chunks_multi_token_prefill_into_per_row_records() {
        use std::sync::Mutex;
        static RECORDS: Mutex<Vec<(String, Vec<f32>)>> = Mutex::new(Vec::new());

        struct StaticCollector;
        impl ImatrixCollector for StaticCollector {
            fn record(&mut self, name: &str, row: &[f32]) {
                RECORDS.lock().unwrap().push((name.to_string(), row.to_vec()));
            }
            fn record_moe(&mut self, _name: &str, _expert_id: usize, _row: &[f32]) {
                unreachable!("dense-only test collector — record_moe not exercised");
            }
        }

        RECORDS.lock().unwrap().clear();
        with_collector(StaticCollector, || {
            // m=3 tokens × n_per_row=2 = 6-wide buffer.
            let result = intercept_qmatmul_with_hint(
                ImatrixHint::Layered { tag: "ffn_gate", layer: 5 },
                /* m */ 3,
                /* n_per_row */ 2,
                || Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            );
            assert!(result.is_ok());
        });

        let records = RECORDS.lock().unwrap();
        assert_eq!(records.len(), 3, "one record per token row");
        assert!(records.iter().all(|r| r.0 == "blk.5.ffn_gate.weight"));
        assert_eq!(records[0].1, vec![1.0, 2.0]);
        assert_eq!(records[1].1, vec![3.0, 4.0]);
        assert_eq!(records[2].1, vec![5.0, 6.0]);
    }

    /// Buffer/shape mismatch returns a typed `ShapeMismatch` error
    /// (no records emitted). Per the codex review 2026-05-19 +
    /// [[feedback-no-loop-suppression-2026-05-17]] this is a typed
    /// error, not a silent skip — silently dropping activation data
    /// would bias the imatrix output.
    #[test]
    fn intercept_errors_typed_on_buffer_shape_mismatch() {
        use std::sync::Mutex;
        static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

        struct C;
        impl ImatrixCollector for C {
            fn record(&mut self, name: &str, _row: &[f32]) {
                RECORDS.lock().unwrap().push(name.to_string());
            }
            fn record_moe(&mut self, _name: &str, _expert_id: usize, _row: &[f32]) {
                unreachable!("dense-only test collector — record_moe not exercised");
            }
        }

        RECORDS.lock().unwrap().clear();
        with_collector(C, || {
            let result = intercept_qmatmul_with_hint(
                ImatrixHint::Layered { tag: "attn_q", layer: 0 },
                /* m */ 2,
                /* n_per_row */ 4, // expects 8 floats
                || Some(vec![1.0; 5]), // returns 5 — mismatch
            );
            match result {
                Err(ImatrixError::ShapeMismatch {
                    tensor,
                    m,
                    n_per_row,
                    got,
                    expected,
                }) => {
                    assert_eq!(tensor, "blk.0.attn_q.weight");
                    assert_eq!(m, 2);
                    assert_eq!(n_per_row, 4);
                    assert_eq!(got, 5);
                    assert_eq!(expected, 8);
                }
                other => panic!("expected ShapeMismatch, got {other:?}"),
            }
        });
        assert!(RECORDS.lock().unwrap().is_empty(), "no records on shape mismatch");
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
            fn record_moe(&mut self, _name: &str, _expert_id: usize, _row: &[f32]) {
                unreachable!("dense-only test collector — record_moe not exercised");
            }
        }

        RECORDS.lock().unwrap().clear();
        with_collector(C, || {
            let result = intercept_qmatmul_with_hint(
                ImatrixHint::Global("token_embd.weight"),
                /* m */ 1,
                /* n_per_row */ 4,
                || Some(vec![0.0; 4]),
            );
            assert!(result.is_ok());
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

    /// Stage 3 driver: `compute_imatrix` on a missing `hf_dir` should
    /// surface `ImatrixError::ConvertFailed` (it can't proceed past
    /// the hf_dir validation gate). Per
    /// [[feedback-no-loop-suppression-2026-05-17]] this is a typed
    /// error, not a silent no-op.
    #[test]
    fn compute_imatrix_errors_typed_on_missing_hf_dir() {
        let corpus = CorpusBytes::load(&CorpusSource::Cdv3).unwrap();
        let params = ComputeImatrixParams {
            hf_dir: PathBuf::from("/tmp/non-existent-fixture-imatrix-driver"),
            corpus,
            n_ctx: 512,
            arch: ArchName::Gemma4,
        };
        let err = compute_imatrix(&params).unwrap_err();
        match err {
            ImatrixError::ConvertFailed { detail } => {
                assert!(
                    detail.contains("does not exist") || detail.contains("not a directory"),
                    "detail should describe missing hf_dir, got: {detail}"
                );
            }
            other => panic!("expected ConvertFailed, got {other:?}"),
        }
    }

    /// Stage 3 driver: arches outside the Stage 3.0 supported set
    /// (Gemma 4 only) surface `UnsupportedArchForDriver` BEFORE
    /// touching the filesystem (cheap upfront validation per the
    /// "fail fast at boundaries" rule).
    #[test]
    fn compute_imatrix_errors_typed_on_unsupported_arch() {
        let corpus = CorpusBytes::load(&CorpusSource::Cdv3).unwrap();
        let params = ComputeImatrixParams {
            // Provide a real-ish path so the hf_dir-existence check
            // doesn't short-circuit first. /tmp is always a dir.
            hf_dir: PathBuf::from("/tmp"),
            corpus,
            n_ctx: 512,
            arch: ArchName::Qwen35Moe,
        };
        let err = compute_imatrix(&params).unwrap_err();
        match err {
            ImatrixError::UnsupportedArchForDriver { arch, supported } => {
                assert_eq!(arch, "qwen3moe");
                assert_eq!(supported, &["gemma4"]);
            }
            other => panic!("expected UnsupportedArchForDriver, got {other:?}"),
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
        fn record_moe(&mut self, _name: &str, _expert_id: usize, _row: &[f32]) {}
    }

    // ─────────────────────────────────────────────────────────────────
    // MoE intercept tests — `intercept_qmatmul_id_with_hint`.
    // Mirrors the canonical `imatrix.cpp:310-330` for
    // `GGML_OP_MUL_MAT_ID`: per (token, routed-expert) accumulation.
    // ─────────────────────────────────────────────────────────────────

    /// No collector → no-op (Ok), neither closure invoked.
    #[test]
    fn moe_intercept_noop_without_collector() {
        let mut input_materialized = false;
        let mut ids_materialized = false;
        let result = intercept_qmatmul_id_with_hint(
            ImatrixHint::Layered { tag: "ffn_gate_up_exps", layer: 0 },
            /* n_tokens */ 2,
            /* top_k */ 2,
            /* n_per_row */ 4,
            || {
                input_materialized = true;
                Some(vec![0.0; 8])
            },
            || {
                ids_materialized = true;
                Some(vec![0u32; 4])
            },
        );
        assert!(result.is_ok());
        assert!(!input_materialized, "input closure should not fire");
        assert!(!ids_materialized, "expert_ids closure should not fire");
    }

    /// `None` hint → no-op even with collector installed.
    #[test]
    fn moe_intercept_noop_with_none_hint() {
        with_collector(RecorderCollector::default(), || {
            let mut input_materialized = false;
            let mut ids_materialized = false;
            let result = intercept_qmatmul_id_with_hint(
                ImatrixHint::None,
                1,
                1,
                1,
                || {
                    input_materialized = true;
                    Some(vec![0.0])
                },
                || {
                    ids_materialized = true;
                    Some(vec![0u32])
                },
            );
            assert!(result.is_ok());
            assert!(!input_materialized);
            assert!(!ids_materialized);
        });
    }

    /// **MoE canonical accumulation invariant.**
    ///
    /// For n_tokens=2 × top_k=2 × n_per_row=3:
    ///   - input  = [t0_row.., t1_row..] = [1,2,3, 4,5,6]
    ///   - ids    = [t0_e0, t0_e1, t1_e0, t1_e1] = [7, 9, 9, 11]
    ///
    /// Expected `record_moe` calls (4 total, in
    /// `for tok { for k { ... } }` order):
    ///   (name="blk.5.ffn_gate_up_exps.weight", expert=7,  row=[1,2,3])
    ///   (name="blk.5.ffn_gate_up_exps.weight", expert=9,  row=[1,2,3])
    ///   (name="blk.5.ffn_gate_up_exps.weight", expert=9,  row=[4,5,6])
    ///   (name="blk.5.ffn_gate_up_exps.weight", expert=11, row=[4,5,6])
    ///
    /// This is the exact wiring `imatrix.cpp:310-330` produces.
    #[test]
    fn moe_intercept_fires_per_token_per_routed_expert() {
        use std::sync::Mutex;
        static RECORDS: Mutex<Vec<(String, usize, Vec<f32>)>> = Mutex::new(Vec::new());

        struct MoeCollector;
        impl ImatrixCollector for MoeCollector {
            fn record(&mut self, _name: &str, _row: &[f32]) {
                panic!("MoE intercept should only call record_moe, not record");
            }
            fn record_moe(&mut self, name: &str, expert_id: usize, row: &[f32]) {
                RECORDS
                    .lock()
                    .unwrap()
                    .push((name.to_string(), expert_id, row.to_vec()));
            }
        }

        RECORDS.lock().unwrap().clear();
        with_collector(MoeCollector, || {
            let result = intercept_qmatmul_id_with_hint(
                ImatrixHint::Layered { tag: "ffn_gate_up_exps", layer: 5 },
                /* n_tokens */ 2,
                /* top_k */ 2,
                /* n_per_row */ 3,
                || Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                || Some(vec![7u32, 9, 9, 11]),
            );
            assert!(result.is_ok());
        });

        let recs = RECORDS.lock().unwrap();
        assert_eq!(recs.len(), 4, "n_tokens * top_k = 2 * 2 = 4 calls");
        assert!(recs.iter().all(|r| r.0 == "blk.5.ffn_gate_up_exps.weight"));
        // Token 0 row=[1,2,3] routes to experts 7 and 9.
        assert_eq!(recs[0].1, 7);
        assert_eq!(recs[0].2, vec![1.0, 2.0, 3.0]);
        assert_eq!(recs[1].1, 9);
        assert_eq!(recs[1].2, vec![1.0, 2.0, 3.0]);
        // Token 1 row=[4,5,6] routes to experts 9 and 11.
        assert_eq!(recs[2].1, 9);
        assert_eq!(recs[2].2, vec![4.0, 5.0, 6.0]);
        assert_eq!(recs[3].1, 11);
        assert_eq!(recs[3].2, vec![4.0, 5.0, 6.0]);
    }

    /// Input buffer / shape mismatch → typed `ShapeMismatch` error
    /// (no records emitted). Same no-suppression contract as the
    /// dense intercept.
    #[test]
    fn moe_intercept_errors_typed_on_input_shape_mismatch() {
        use std::sync::Mutex;
        static CALLS: Mutex<u32> = Mutex::new(0);

        struct C;
        impl ImatrixCollector for C {
            fn record(&mut self, _name: &str, _row: &[f32]) {}
            fn record_moe(&mut self, _name: &str, _expert_id: usize, _row: &[f32]) {
                *CALLS.lock().unwrap() += 1;
            }
        }
        *CALLS.lock().unwrap() = 0;
        with_collector(C, || {
            let result = intercept_qmatmul_id_with_hint(
                ImatrixHint::Layered { tag: "ffn_gate_up_exps", layer: 0 },
                /* n_tokens */ 2,
                /* top_k */ 2,
                /* n_per_row */ 4, // expects 8 input floats
                || Some(vec![1.0; 5]), // returns 5 — mismatch
                || Some(vec![0u32; 4]),
            );
            match result {
                Err(ImatrixError::ShapeMismatch { tensor, expected, got, .. }) => {
                    assert_eq!(tensor, "blk.0.ffn_gate_up_exps.weight");
                    assert_eq!(expected, 8);
                    assert_eq!(got, 5);
                }
                other => panic!("expected ShapeMismatch, got {other:?}"),
            }
        });
        assert_eq!(*CALLS.lock().unwrap(), 0, "no records on shape mismatch");
    }

    /// expert_ids buffer length mismatch → typed `ShapeMismatch`
    /// (uses the `::expert_ids` suffix in `tensor` so the operator
    /// can distinguish input-buffer vs ids-buffer mismatch at a
    /// glance).
    #[test]
    fn moe_intercept_errors_typed_on_expert_ids_shape_mismatch() {
        with_collector(RecorderCollector::default(), || {
            let result = intercept_qmatmul_id_with_hint(
                ImatrixHint::Layered { tag: "ffn_down_exps", layer: 3 },
                /* n_tokens */ 2,
                /* top_k */ 4,
                /* n_per_row */ 2,
                || Some(vec![1.0; 4]), // input ok (2*2=4)
                || Some(vec![0u32; 7]), // expert_ids expects 2*4=8, got 7
            );
            match result {
                Err(ImatrixError::ShapeMismatch { tensor, expected, got, .. }) => {
                    assert_eq!(tensor, "blk.3.ffn_down_exps.weight::expert_ids");
                    assert_eq!(expected, 8);
                    assert_eq!(got, 7);
                }
                other => panic!("expected ShapeMismatch, got {other:?}"),
            }
        });
    }

    /// Zero-token / zero-row / zero-top_k → Ok with no calls.
    /// Defensive: a degenerate dispatch shouldn't crash but also
    /// shouldn't generate spurious accumulator entries.
    #[test]
    fn moe_intercept_zero_dims_no_calls() {
        use std::sync::Mutex;
        static CALLS: Mutex<u32> = Mutex::new(0);
        struct C;
        impl ImatrixCollector for C {
            fn record(&mut self, _name: &str, _row: &[f32]) {}
            fn record_moe(&mut self, _name: &str, _expert_id: usize, _row: &[f32]) {
                *CALLS.lock().unwrap() += 1;
            }
        }
        for (n_tokens, top_k, n_per_row) in
            [(0usize, 2usize, 4usize), (2, 0, 4), (2, 2, 0)]
        {
            *CALLS.lock().unwrap() = 0;
            with_collector(C, || {
                let result = intercept_qmatmul_id_with_hint(
                    ImatrixHint::Layered { tag: "ffn_gate_up_exps", layer: 0 },
                    n_tokens,
                    top_k,
                    n_per_row,
                    || Some(vec![0.0; n_tokens * n_per_row]),
                    || Some(vec![0u32; n_tokens * top_k]),
                );
                assert!(result.is_ok());
            });
            assert_eq!(*CALLS.lock().unwrap(), 0);
        }
    }
}
