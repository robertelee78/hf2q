//! Outer forward-pass dispatch for the Gemma 4 model (GPU path).
//!
//! Contains:
//! - `encode_parallel_layers_chunked` — ADR-031 parallel-encode worker.
//! - `forward_decode` — single-token autoregressive decode.
//! - `forward_decode_verify_serial` — serial multi-token verify.
//! - `rollback_kv` — KV-cache rollback.
//! - `forward_decode_kernel_profile` — per-kernel-type profiling.
//!
//! Moved from `src/serve/forward_mlx.rs` by ADR-038 Step 3.

use anyhow::{Context, Result};
use mlx_native::KernelRegistry;
use mlx_native::ops::dense_gemm::DenseGemmF16Params;
use mlx_native::ops::elementwise::CastDirection;
use std::time::Instant;

use crate::debug::{dumps, INVESTIGATION_ENV};
use crate::serve::config::LayerType;
use crate::serve::encoder_worker_singleton::submit_to_global_worker;
use crate::serve::gpu::GpuContext;
use crate::serve::layer_ctx::LayerCtx;
use mlx_native::ops::flash_attn_vec_tq::FlashAttnVecTqParams;
use crate::serve::forward_mlx_shared::{
    dispatch_qmatmul, dispatch_rms_norm_unit_perhead,
    RmsNormPerHeadArgs,
};
use crate::inference::models::gemma4::kv_cache::{
    HbKvBuffers, HybridKvBuffers, alloc_hybrid_kv_for_layer,
};
use super::profile::{merge_profiles, TokenProfile, KernelTypeProfile};
use super::model::MlxModelWeights;

impl MlxModelWeights {
    /// Encode two disjoint layer chunks in parallel using the global encoder
    /// worker (ADR-031 Phase B, Path D).
    ///
    /// The worker creates a fresh `GraphSession` from `exec`, encodes `range_a`
    /// layers on the encoder worker thread; the main thread concurrently encodes
    /// `range_b` layers into `session_b`.  GPU execution order is guaranteed by
    /// commit-order serialization (Pillar P3 / INV-7): the worker calls
    /// `session_a.commit()` BEFORE sending on `done_tx`, and main commits
    /// `session_b` only AFTER `done_rx.recv()` returns (main is blocked on recv
    /// during chunk-B encoding) — Metal queue receives CB-A before CB-B.
    ///
    /// Returns the worker's `KernelRegistry` in BOTH the Ok and Err arms so
    /// the caller can always restore it into `GpuContext::put_worker_registry`
    /// without leaking it on encode failure (R-B9 / FIX-3 / iter-2 final).
    ///
    /// Return shape: `(Option<KernelRegistry>, Result<()>)`.
    /// - `(Some(reg), Ok(()))` — both sides succeeded; reg is the worker's registry.
    /// - `(Some(reg), Err(e))` — at least one side errored AFTER the worker
    ///   thread accepted the closure; reg recovered through the mpsc payload.
    /// - `(None, Err(e))` — failure BEFORE the worker thread took ownership
    ///   (e.g. `submit_to_global_worker` lock-poisoned) OR the mpsc channel
    ///   dropped (worker panicked).  In the rare submit-failure case the
    ///   registry was already moved into the closure that never ran; the
    ///   registry is permanently lost for this `GpuContext` (process restart
    ///   recovers).  Phase C addresses this via Arc<Mutex<KernelRegistry>>.
    #[allow(clippy::too_many_arguments)]
    fn encode_parallel_layers_chunked<'sess>(
        &self,
        range_a: std::ops::Range<usize>,
        range_b: std::ops::Range<usize>,
        ctx: &LayerCtx<'_>,
        session_b: &mut mlx_native::graph::GraphSession<'sess>,
        exec: &'sess mlx_native::GraphExecutor,
        main_reg: &mut KernelRegistry,
        worker_reg: KernelRegistry,
        profile_main: &mut Option<TokenProfile>,
        per_layer_disp_log_main: &mut Vec<(usize, bool, u64)>,
        total_dispatches_main: &mut usize,
    ) -> (Option<KernelRegistry>, Result<()>) {
        // ADR-031 Phase C — diagnostic profiling.  Gated on
        // HF2Q_PARALLEL_PROFILE=1 env var (read once per token, branch
        // predictor handles cheaply when unset).  Prints per-phase timings
        // to stderr so the Phase C design choice (commit-overlap via
        // GraphSession::enqueue() vs CPU overhead reduction) can be made
        // based on measured data, not assumption.
        let profile_enabled = std::env::var("HF2Q_PARALLEL_PROFILE").as_deref() == Ok("1");
        let prof_t0 = if profile_enabled { Some(std::time::Instant::now()) } else { None };
        // Worker sends back (registry, accumulators) on both Ok and Err paths so
        // the caller can always restore the registry into GpuContext.  Err carries
        // the registry alongside the error string to avoid a registry-leak on
        // encode failure (R-B9 / FIX-3).
        type WorkerOk = (KernelRegistry, Vec<(usize, bool, u64)>, usize, Option<TokenProfile>);
        type WorkerErr = (KernelRegistry, String);
        type WorkerResult = Result<WorkerOk, WorkerErr>;
        let prof_t_pre_channel = prof_t0.map(|_| std::time::Instant::now());
        let (done_tx, done_rx) = std::sync::mpsc::channel::<WorkerResult>();
        let prof_channel_us = prof_t_pre_channel.map(|t| t.elapsed().as_micros());

        // Clone profile scratch for the worker.  If profiling is active,
        // profile_main already has per-layer Vecs pre-allocated to num_layers;
        // the worker writes only its range_a indices, main writes range_b.
        let profile_a: Option<TokenProfile> = profile_main.clone();
        let per_layer_disp_a: Vec<(usize, bool, u64)> = Vec::new();
        let total_dispatches_a: usize = 0;

        // SAFETY: We forge &'a-bound references into &'static so the 'static-
        // bounded closure required by submit_to_global_worker can capture them.
        // This is sound IFF done_rx.recv() below COMPLETES before this function
        // returns and before the enclosing forward_decode stack frame is unwound.
        // All forged-'static refs are ONLY dereferenced during the worker's
        // execution, which is bracketed by `submit_to_global_worker` (start) and
        // `done_rx.recv()` (end).  Encapsulating both sides in this single helper
        // makes the spawn-and-wait invariant atomic at the API level — a future
        // maintainer who removes done_rx.recv() will see the unsafe block in the
        // same file and the safety comment will flag the dependency immediately.
        //
        // This mirrors the crossbeam::thread::scope pattern but works against a
        // persistent global worker (per-token thread spawn was empirically
        // -43 tok/s; see forward_mlx.rs comment near line 5471-5474).
        //
        // No GraphSession value crosses the mpsc boundary (R-B1): session_a is
        // created inside the closure from exec_static.begin(), consumed by
        // commit() inside the closure, and never sent across mpsc.
        // SAFETY: We forge &'a-bound refs into &'static so the 'static-bounded
        // closure required by submit_to_global_worker can capture them.  This is
        // sound IFF done_rx.recv() below COMPLETES unconditionally before this
        // function returns — including on the Err path from chunk-B encoding.
        // See the mandatory-recv block below; its comment is load-bearing.
        let (self_static, ctx_static, exec_static) = unsafe {
            (
                std::mem::transmute::<&Self, &'static Self>(self),
                std::mem::transmute::<&crate::serve::layer_ctx::LayerCtx<'_>, &'static crate::serve::layer_ctx::LayerCtx<'static>>(ctx),
                std::mem::transmute::<&mlx_native::GraphExecutor, &'static mlx_native::GraphExecutor>(exec),
            )
        };

        // Submit chunk-A closure to the worker.  The closure moves: worker_reg,
        // profile_a, per_layer_disp_a, total_dispatches_a, done_tx, range_a.
        // The closure uses forged-'static: self_static, ctx_static, exec_static.
        // session_a is created INSIDE the closure from exec_static so its
        // GraphSession<'static> lifetime is consistent with exec_static's 'static.
        let prof_t_pre_submit = prof_t0.map(|_| std::time::Instant::now());
        if let Err(submit_err) = submit_to_global_worker(move || {
            // worker_reg kept in outer scope so it is always accessible for the
            // Err arm below — the inner encode loop borrows it mutably but does
            // not consume it, so it is still owned here on both Ok and Err paths.
            let mut worker_reg = worker_reg;
            let encode_result: Result<(Vec<(usize, bool, u64)>, usize, Option<TokenProfile>), String> = (|| {
                // ADR-031 Phase C profile: worker-side phase timings.
                let pw_begin = profile_enabled.then(std::time::Instant::now);
                let mut session_a = exec_static
                    .begin()
                    .map_err(|e| format!("worker begin session_a: {e}"))?;
                let pw_begin_us = pw_begin.map(|t| t.elapsed().as_micros());
                let pw_encode = profile_enabled.then(std::time::Instant::now);
                let mut profile_a = profile_a;
                let mut per_layer_disp_a = per_layer_disp_a;
                let mut total_dispatches_a = total_dispatches_a;
                for layer_idx in range_a {
                    self_static
                        .encode_one_layer(
                            layer_idx,
                            ctx_static,
                            &mut session_a,
                            exec_static,
                            &mut worker_reg,
                            &mut profile_a,
                            &mut per_layer_disp_a,
                            &mut total_dispatches_a,
                        )
                        .map_err(|e| format!("worker encode L{layer_idx}: {e}"))?;
                }
                let pw_encode_us = pw_encode.map(|t| t.elapsed().as_micros());
                let pw_commit = profile_enabled.then(std::time::Instant::now);
                // CRITICAL: commit chunk-A's CommandBuffer BEFORE signaling via
                // done_tx.send.  This ensures CB-A enters the Metal queue BEFORE
                // main thread commits CB-B (which happens after done_rx.recv()
                // returns).  Metal executes CBs in commit order → GPU runs
                // chunk_A then chunk_B → activations.hidden handoff is correct.
                // See INV-7 / R-B7.
                let _enc = session_a.commit();
                let pw_commit_us = pw_commit.map(|t| t.elapsed().as_micros());
                if profile_enabled {
                    eprintln!(
                        "[PARALLEL_PROFILE worker] begin={}µs encode_a={}µs commit_a={}µs",
                        pw_begin_us.unwrap_or(0),
                        pw_encode_us.unwrap_or(0),
                        pw_commit_us.unwrap_or(0),
                    );
                }
                Ok((per_layer_disp_a, total_dispatches_a, profile_a))
            })();
            // Registry always travels back — on both Ok and Err — so the caller
            // can restore it into GpuContext without leaking it (R-B9 / FIX-3).
            let result: WorkerResult = match encode_result {
                Ok((disp, dispatches, profile)) => Ok((worker_reg, disp, dispatches, profile)),
                Err(e) => Err((worker_reg, e)),
            };
            // Ignore send error: if done_rx was dropped, the recv below will
            // return Err and propagate the error up cleanly.
            let _ = done_tx.send(result);
        }) {
            // submit_to_global_worker failed BEFORE the closure ran.  Per the
            // signature contract, the registry is now lost (it was moved into
            // the closure that never executed).  Return (None, Err) so the
            // caller can propagate cleanly.  This is a rare path (only fires
            // if GLOBAL_ENCODER_WORKER mutex is poisoned — i.e. worker thread
            // already panicked).  Phase C: Arc<Mutex<KernelRegistry>> would
            // let the registry survive even submit failures.
            return (None, Err(anyhow::anyhow!("submit_to_global_worker failed: {submit_err}")));
        }

        let prof_submit_us = prof_t_pre_submit.map(|t| t.elapsed().as_micros());

        // Main thread encodes chunk-B concurrently with the worker's chunk-A.
        // Use a Result capture rather than `?` so we can unconditionally wait for
        // the worker before propagating any error (FIX-1 / INV-4 + INV-5).
        let prof_t_pre_encode_b = prof_t0.map(|_| std::time::Instant::now());
        let chunk_b_result: Result<()> = (|| {
            for layer_idx in range_b {
                self.encode_one_layer(
                    layer_idx,
                    ctx,
                    session_b,
                    exec,
                    main_reg,
                    profile_main,
                    per_layer_disp_log_main,
                    total_dispatches_main,
                )?;
            }
            Ok(())
        })();
        let prof_encode_b_us = prof_t_pre_encode_b.map(|t| t.elapsed().as_micros());

        // MANDATORY unconditional wait — load-bearing for the unsafe block's
        // soundness.  The forged-'static refs (self_static, ctx_static,
        // exec_static) are aliased on the worker thread until this recv()
        // completes.  If chunk_b_result is Err we must STILL wait; returning
        // early would let the original stack frame unwind while the worker is
        // still dereferencing those forged refs — UB.  Do NOT add any
        // early-return or `?` between the submit_to_global_worker call above
        // and this recv().
        let prof_t_pre_recv = prof_t0.map(|_| std::time::Instant::now());
        let worker_msg = match done_rx.recv() {
            Ok(msg) => msg,
            Err(_) => {
                // mpsc channel dropped — worker panicked.  Registry is gone
                // with the worker thread; nothing for us to return.
                return (None, Err(anyhow::anyhow!(
                    "parallel-encode worker channel closed unexpectedly — \
                     worker thread may have panicked"
                )));
            }
        };

        // Both sides have finished.  Extract registry + accumulators, then
        // propagate whichever error occurred.  R-B9 fix: in EVERY arm we
        // recover the worker's KernelRegistry (from the mpsc payload's Ok or
        // Err tuple) so the caller can restore it into GpuContext, even on
        // encode failure.  chunk-B errors take precedence over worker errors
        // since chunk-B is the originating frame's outcome.
        let (returned_worker_reg, mut per_layer_disp_a, dispatches_a, profile_a) =
            match (chunk_b_result, worker_msg) {
                (Ok(()), Ok(ok)) => ok,
                (Err(main_e), Ok((wreg, _disp, _dispatches, _profile))) => {
                    // chunk-B errored; worker succeeded.  Recover worker reg
                    // through the Ok tuple; discard worker's accumulator data
                    // (output is unreliable since main errored).
                    return (Some(wreg), Err(main_e));
                }
                (Ok(()), Err((wreg, worker_e))) => {
                    // Worker errored; chunk-B succeeded.  Recover worker reg
                    // through the Err tuple (FIX-3 already routed it there).
                    return (Some(wreg), Err(anyhow::anyhow!(
                        "parallel-encode worker error: {worker_e}"
                    )));
                }
                (Err(main_e), Err((wreg, worker_e))) => {
                    // Both errored.  Recover worker reg; chain errors.
                    return (Some(wreg), Err(anyhow::anyhow!(
                        "chunk-B encode error: {main_e}; worker also errored: {worker_e}"
                    )));
                }
            };

        let prof_recv_us = prof_t_pre_recv.map(|t| t.elapsed().as_micros());
        if profile_enabled {
            let total_us = prof_t0.map(|t| t.elapsed().as_micros()).unwrap_or(0);
            eprintln!(
                "[PARALLEL_PROFILE main]  channel={}µs submit={}µs encode_b={}µs recv_blocked={}µs total_helper={}µs",
                prof_channel_us.unwrap_or(0),
                prof_submit_us.unwrap_or(0),
                prof_encode_b_us.unwrap_or(0),
                prof_recv_us.unwrap_or(0),
                total_us,
            );
        }

        // Merge worker's accumulator back into main's:
        //  - per_layer_disp_log: order within the vec doesn't matter (post-loop
        //    dump iterates all entries); just append.
        per_layer_disp_log_main.append(&mut per_layer_disp_a);
        //  - total_dispatches: additive.
        *total_dispatches_main += dispatches_a;
        //  - profile: per-layer Vecs are indexed by layer_idx; range_a indices
        //    were written by worker, range_b by main — merge by summing each
        //    index (worker's range_b entries are all zero; main's range_a
        //    entries are all zero before this merge).
        merge_profiles(profile_main, profile_a);

        (Some(returned_worker_reg), Ok(()))
    }

    /// Returns: the next token ID (greedy decode)
    pub fn forward_decode(
        &mut self,
        input_token: u32,
        seq_pos: usize,
        gpu: &mut GpuContext,
        profile: &mut Option<TokenProfile>,
    ) -> Result<u32> {
        let token_start = Instant::now();
        let hs = self.hidden_size;
        let num_layers = self.layers.len();
        let vocab_size = self.vocab_size;

        // Pre-allocate profile vectors if profiling
        if let Some(ref mut p) = profile {
            p.layer_s1_us = vec![0.0; num_layers];
            p.layer_cpu1_us = vec![0.0; num_layers];
            p.layer_s2_us = vec![0.0; num_layers];
            p.layer_cpu2_us = vec![0.0; num_layers];
            p.layer_s3_us = vec![0.0; num_layers];
            p.layer_cpu3_us = vec![0.0; num_layers];
            p.layer_s4_us = vec![0.0; num_layers];
            p.layer_cpu4_us = vec![0.0; num_layers];
            p.s1_dispatches = vec![0; num_layers];
            p.s2_dispatches = vec![0; num_layers];
            p.s3_dispatches = vec![0; num_layers];
            p.s4_dispatches = vec![0; num_layers];
        }

        // ADR-009 Phase 3A: boundary dump at specific token position.
        // Temporary diagnostic — to be merged into parity capture workflow.
        let dump_pos: Option<usize> = INVESTIGATION_ENV.dump_boundary;
        // iter-23: HF2Q_DUMP_SDPA_MAX_POS=N — when set along with HF2Q_DUMP_ALL_CACHE=1,
        // dump sdpa_out for all layers at every decode STEP < N (decode-step index, not
        // absolute seq_pos, so it is prompt-length independent).
        // The counter increments on each forward_decode call regardless of layer.
        // This enables the Gate A cosine-sim harness without requiring N separate hf2q runs.
        //
        // W39 iter-112b: HF2Q_DUMP_SDPA_MAX_POS still uses a process-static
        // OnceLock (read once, never overridden — Gate H always wants the
        // same `tokens` window across both passes), but the decode-step
        // counter moved from a process-static `AtomicUsize` to per-instance
        // `self.decode_step_dump_counter` so `set_decode_regime` /
        // `set_replay_tokens` / `set_dump_overrides` can reset it between
        // the dense and TQ passes of a single Gate H run.  The
        // `dump_all_cache` read also consults the per-instance override
        // first (W39 iter-112b: `INVESTIGATION_ENV` LazyLock is frozen at
        // `main.rs::main` before parity_quality's runtime `set_var` lands).
        let dump_sdpa_max_pos: Option<usize> = {
            static MAX_POS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
            *MAX_POS.get_or_init(|| {
                std::env::var("HF2Q_DUMP_SDPA_MAX_POS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
            })
        };
        let decode_step_for_dump: usize = self.decode_step_dump_counter;
        self.decode_step_dump_counter = self.decode_step_dump_counter.saturating_add(1);
        let dump_all_cache_eff: bool = self
            .dump_all_cache_override
            .unwrap_or(INVESTIGATION_ENV.dump_all_cache);
        let dump_layers: bool = INVESTIGATION_ENV.dump_layers == Some(seq_pos)
            || dump_sdpa_max_pos.map_or(false, |max| {
                decode_step_for_dump < max && dump_all_cache_eff
            });

        // --- Pre-session CPU work ---
        // Write position buffer (same for all layers)
        {
            let pos_dst: &mut [u32] = self.activations.position.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("position write: {e}"))?;
            pos_dst[0] = seq_pos as u32;
        }

        // KV cache bookkeeping for all layers (CPU counters only, no GPU buffers)
        let mut kv_info: Vec<(bool, usize, usize, usize)> = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let is_sliding = self.kv_caches[layer_idx].is_sliding;
            let write_pos = self.kv_caches[layer_idx].write_pos;
            let capacity = self.kv_caches[layer_idx].capacity;
            self.kv_caches[layer_idx].write_pos += 1;
            self.kv_caches[layer_idx].seq_len = self.kv_caches[layer_idx].seq_len.saturating_add(1)
                .min(capacity);
            let seq_len = self.kv_caches[layer_idx].seq_len;
            kv_info.push((is_sliding, write_pos, capacity, seq_len));
        }

        // iter-21 Track B: lazy allocation of leg_hb_encoded on first decode.
        // forward_prefill may have already allocated this; if not, do it here.
        // We read from the INVESTIGATION_ENV LazyLock (parsed once at process start).
        {
            // ADR-007 post-close correction 2026-04-24: TQ-8-bit is the default when
            // env is unset. Explicit HF2Q_TQ_CODEBOOK_BITS=4 selects the legacy 4-bit
            // native flash_attn_vec_tq path (127-byte sourdough ceiling, not shippable
            // as default). Explicit =5/=6 select intermediate HB-SDPA. This MUST match
            // the primary gate at tq_codebook_bits below.
            // ADR-005 wave-1 T1.2: read from INVESTIGATION_ENV LazyLock (parsed once at
            // process start) instead of calling std::env::var per forward_decode call.
            let cb_bits: u32 = INVESTIGATION_ENV.tq_codebook_bits;
            // ADR-028 Phase 10c (iter-348): if hybrid_kv gate is on, route to
            // HybridKvBuffers (F16 K + TQ-HB V) instead of legacy HbKvBuffers
            // (TQ-HB K + TQ-HB V). Mutually exclusive at alloc time per the
            // Phase 10b struct comment; only one of `hybrid_kv` /
            // `leg_hb_encoded` is `Some(_)` for a given model instance.
            if cb_bits >= 5 && INVESTIGATION_ENV.hybrid_kv && self.hybrid_kv.is_none() {
                let (exec, _reg) = gpu.split();
                let dev = exec.device();
                let mut hybrid_vec: Vec<HybridKvBuffers> = Vec::with_capacity(num_layers);
                for layer_idx in 0..num_layers {
                    let nkv = self.layers[layer_idx].num_kv_heads;
                    let hd = self.layers[layer_idx].head_dim;
                    let is_ring = self.kv_caches[layer_idx].is_sliding;
                    let cap = self.kv_caches[layer_idx].capacity;
                    hybrid_vec.push(alloc_hybrid_kv_for_layer(dev, layer_idx, nkv, hd, cap, is_ring)?);
                }
                eprintln!("[ADR-028 Phase 10c] Allocated hybrid_kv ({} layers, F16 K + TQ-HB V {}-bit)",
                    num_layers, cb_bits);
                self.hybrid_kv = Some(hybrid_vec);
            } else if cb_bits >= 5 && !INVESTIGATION_ENV.hybrid_kv && self.leg_hb_encoded.is_none() {
                let (exec, _reg) = gpu.split();
                let dev = exec.device();
                // Use kv_caches[0] write_pos - 1 to infer linear capacity. In practice
                // we use the same capacity as kv_caches per layer (the KV cache was sized
                // for the full sequence at init time).
                let mut leg_hb_vec: Vec<HbKvBuffers> = Vec::with_capacity(num_layers);
                for layer_idx in 0..num_layers {
                    let nkv = self.layers[layer_idx].num_kv_heads;
                    let hd = self.layers[layer_idx].head_dim;
                    let is_ring = self.kv_caches[layer_idx].is_sliding;
                    let cap = self.kv_caches[layer_idx].capacity;
                    let norms_per_pos = (hd / 256).max(1);
                    let norms_n = nkv * cap * norms_per_pos;
                    // byte-packed: 1 byte per element
                    let k_packed = dev.alloc_buffer(nkv * cap * hd, mlx_native::DType::U8,
                        vec![nkv, cap, hd])
                        .map_err(|e| anyhow::anyhow!("leg_hb K packed L{layer_idx}: {e}"))?;
                    let k_norms = dev.alloc_buffer(norms_n * 4, mlx_native::DType::F32,
                        if norms_per_pos == 1 { vec![nkv, cap] } else { vec![nkv, cap, norms_per_pos] })
                        .map_err(|e| anyhow::anyhow!("leg_hb K norms L{layer_idx}: {e}"))?;
                    let v_packed = dev.alloc_buffer(nkv * cap * hd, mlx_native::DType::U8,
                        vec![nkv, cap, hd])
                        .map_err(|e| anyhow::anyhow!("leg_hb V packed L{layer_idx}: {e}"))?;
                    let v_norms = dev.alloc_buffer(norms_n * 4, mlx_native::DType::F32,
                        if norms_per_pos == 1 { vec![nkv, cap] } else { vec![nkv, cap, norms_per_pos] })
                        .map_err(|e| anyhow::anyhow!("leg_hb V norms L{layer_idx}: {e}"))?;
                    leg_hb_vec.push(HbKvBuffers {
                        k_packed, k_norms, v_packed, v_norms,
                        capacity: cap, is_sliding: is_ring, norms_per_pos,
                    });
                }
                eprintln!("[iter-21 Track B] Allocated leg_hb_encoded ({} layers, {}-bit)", num_layers, cb_bits);
                self.leg_hb_encoded = Some(leg_hb_vec);
            }
            // iter-222 (2026-05-01): the lazy-allocate of `leg_f_kvs` shadow
            // cache that lived here was deleted along with the iter-34
            // dense-on-shadow Leg F decode branch — `flash_attn_vec_tq_hb`
            // consumes `leg_hb_encoded` directly with no F32 round-trip.
        }

        // =====================================================================
        // SINGLE SESSION: Embedding + All 30 Layers + Head
        //
        // ONE begin() → all GPU dispatches → ONE finish().
        // Zero CPU readbacks.  All norms, adds, MoE routing, scalar multiplies,
        // softcap, and argmax run on GPU.
        // =====================================================================
        // iter-18 S2B: D=512 per-block scale factor for encoder+decoder ablation.
        // HF2Q_SCALE_FORMULA: bare (1.0), sqrt256 (16.0), sqrt512 (≈22.627).
        // Read once per decode call; passed to dispatch_hadamard_quantize_kv + SDPA params.
        let tq_scale_factor_d512: f32 = {
            static SCALE_FACTOR: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
            *SCALE_FACTOR.get_or_init(|| {
                match std::env::var("HF2Q_SCALE_FORMULA").as_deref() {
                    Ok("sqrt256") => {
                        eprintln!("[HF2Q_SCALE_FORMULA] D=512 scale_factor = sqrt(256) = 16.0");
                        16.0_f32
                    }
                    Ok("sqrt512") => {
                        let v = 512.0_f32.sqrt();
                        eprintln!("[HF2Q_SCALE_FORMULA] D=512 scale_factor = sqrt(512) = {v:.4}");
                        v
                    }
                    Ok("bare") | Err(_) => {
                        // Default: bare (iter-16 control state)
                        1.0_f32
                    }
                    Ok(other) => {
                        eprintln!("[HF2Q_SCALE_FORMULA] unknown value {other:?}; using bare (1.0)");
                        1.0_f32
                    }
                }
            })
        };

        // iter-222 (ADR-005 closure, 2026-05-01): the iter-34 `force_dense_sdpa_on_tq_kv`
        // gate that lived here was deleted — see file-level iter-222 closure
        // note above the (now-deleted) `dense_sdpa_on_tq_kv_enabled()` site for
        // rationale. TQ-regime SDPA now flows through the inline-fused
        // `flash_attn_vec_tq` (cb_bits=4) / `flash_attn_vec_tq_hb` (cb_bits>=5)
        // kernels unconditionally — peer-correct production paths that read
        // TQ-packed K/V directly with no F32 shadow-cache round-trip.

        // iter-21 Track B + 2026-04-24 post-close default correction.
        // HF2Q_TQ_CODEBOOK_BITS selects the KV codebook width.
        //   unset  (DEFAULT) = 8-bit native HB SDPA (2× memory savings vs F16, 0.017 PPL
        //                      absolute / 1.24% delta, cosine 0.9998 — meets TurboQuant
        //                      paper + KIVI + KVQuant + AmesianX + vLLM published gates)
        //   "4"              = legacy 4-bit native flash_attn_vec_tq (iter-16 control;
        //                      127-byte sourdough ceiling — not shippable as default)
        //   "5" | "6"        = intermediate higher-bit HB SDPA (Lloyd-Max native)
        //   "8"              = explicit 8-bit (same as unset)
        // MUST stay in lockstep with the `cb_bits` lazy-alloc gate above.
        let tq_codebook_bits: u32 = {
            static CODEBOOK_BITS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
            *CODEBOOK_BITS.get_or_init(|| {
                match std::env::var("HF2Q_TQ_CODEBOOK_BITS").as_deref() {
                    Ok("4") => {
                        eprintln!("[HF2Q_TQ_CODEBOOK_BITS] 4-bit legacy TQ (opt-in; 127-byte sourdough ceiling)");
                        0u32
                    }
                    Ok("5") => {
                        eprintln!("[HF2Q_TQ_CODEBOOK_BITS] 5-bit Lloyd-Max native HB SDPA");
                        5u32
                    }
                    Ok("6") => {
                        eprintln!("[HF2Q_TQ_CODEBOOK_BITS] 6-bit Lloyd-Max native HB SDPA");
                        6u32
                    }
                    Ok("8") | Err(_) => {
                        eprintln!("[HF2Q_TQ_CODEBOOK_BITS] 8-bit Lloyd-Max native HB SDPA (default)");
                        8u32
                    }
                    Ok(other) => {
                        eprintln!("[HF2Q_TQ_CODEBOOK_BITS] unknown value {:?}; defaulting to 8-bit", other);
                        8u32
                    }
                }
            })
        };
        // iter-24: native HB SDPA via `flash_attn_vec_tq_hb` for cb_bits >= 5
        // (default 8). Reads TQ-packed K/V directly from `leg_hb_encoded` —
        // no F32 shadow-cache round-trip.
        let use_native_hb_sdpa = tq_codebook_bits >= 5;

        let session_start = Instant::now();
        let mut total_dispatches = 0usize;

        // ADR-031 Phase B: take the worker registry from gpu BEFORE gpu.split()
        // borrows gpu.  `gpu.split()` mutably borrows all of gpu; we cannot call
        // gpu.take_worker_registry() inside the inner block.  The returned registry
        // travels through the inner block in a local Option and is restored into
        // gpu after the block exits.
        // Engage parallel-encode only when the env var is set AND seq_pos is
        // above the kv-depth threshold.  Below the threshold the worker overhead
        // (mpsc + second GraphSession + Metal CB contention) exceeds the benefit;
        // the serial path is used instead (non-regression at shallow KV depth).
        let parallel_encode = INVESTIGATION_ENV.parallel_encode_enabled()
            && seq_pos >= INVESTIGATION_ENV.parallel_encode_kv_threshold;
        let maybe_worker_reg: Option<KernelRegistry> = if parallel_encode {
            Some(gpu.take_worker_registry().ok_or_else(|| anyhow::anyhow!(
                "HF2Q_PARALLEL_ENCODE=1 is set but worker_registry was not pre-warmed. \
                 GpuContext::new() must be called AFTER HF2Q_PARALLEL_ENCODE=1 is set in \
                 the environment (it is read once at model load)."
            ))?)
        } else {
            None
        };
        let mut returned_worker_reg: Option<KernelRegistry> = None;

        // R-B9 / iter-2 final: wrap the gpu.split() inner block in an IIFE so
        // any `?`-propagated error inside it returns from the closure, NOT
        // from forward_decode.  This ensures the post-IIFE
        // put_worker_registry below runs UNCONDITIONALLY — restoring the
        // registry into GpuContext on every exit path (success, encode error,
        // or any internal `?` failure).  The final error propagation happens
        // AFTER the put so the registry handle always rejoins GpuContext.
        let parallel_block_result: Result<()> = (|| {
            let (exec, reg) = gpu.split();
            let dev = exec.device();
            let metal_dev = dev.metal_device();
            let use_graph_opt = INVESTIGATION_ENV.graph_opt;
            let mut s = if use_graph_opt {
                exec.begin_recorded().map_err(|e| anyhow::anyhow!("recorded session begin: {e}"))?
            } else {
                exec.begin().map_err(|e| anyhow::anyhow!("single session begin: {e}"))?
            };

            // --- 1. Embedding gather + scale (GPU) ---
            // Set pending buffer ranges for graph capture (Phase 4e.5): the
            // embedding dispatch reads embed_weight and writes hidden.
            if use_graph_opt {
                let read_ranges = vec![{
                    let s_ptr = self.embed_weight.contents_ptr() as usize;
                    (s_ptr, s_ptr + self.embed_weight.byte_len())
                }];
                let write_ranges = vec![{
                    let s_ptr = self.activations.hidden.contents_ptr() as usize;
                    (s_ptr, s_ptr + self.activations.hidden.byte_len())
                }];
                s.encoder_mut().set_pending_buffer_ranges(read_ranges, write_ranges);
            }
            mlx_native::ops::elementwise::embedding_gather_scale_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.embed_weight,
                &self.activations.hidden,
                input_token,
                hs,
                (hs as f32).sqrt(),
            ).map_err(|e| anyhow::anyhow!("embedding_gather_scale: {e}"))?;
            total_dispatches += 1;
            s.track_dispatch(&[&self.embed_weight], &[&self.activations.hidden]);

            // --- Dual command buffer: split encoding at a layer boundary ---
            // Default: split after layer 3 (~10% of dispatches committed early so
            // GPU starts while CPU encodes the remaining 90%). Measured +4.4 tok/s
            // (94.3→98.7) with zero correctness impact (sourdough gate PASS).
            //
            // Override: HF2Q_DUAL_BUFFER=N (split after layer N, 0=disabled).
            // ADR-028 iter-374: comma-separated list supported (e.g. "2,10,20").
            let dual_buffer_split: Option<usize> =
                INVESTIGATION_ENV.dual_buffer_split(num_layers);
            let dual_buffer_splits: Vec<usize> =
                INVESTIGATION_ENV.dual_buffer_splits(num_layers);
            let _ = dual_buffer_split;

            // --- 2. Transformer layers ---
            // Phase 3A: sub-layer detail dump (which specific layer to break down)
            let dump_detail_layer: Option<usize> = INVESTIGATION_ENV.dump_layer_detail;
            // iter-18 S2C: first-divergence dump for layer 0 (sliding, hd=256), decode positions 1..=10.
            // Gate: HF2Q_DUMP_SLIDING_LAYER_0=1 env var. Run name: HF2Q_DUMP_RUN_NAME (dense|tq).
            // ADR-005 wave-1 T1.2: read from INVESTIGATION_ENV LazyLock (parsed once at process start).
            let dump_sliding_l0: bool = INVESTIGATION_ENV.dump_sliding_layer_0;
            let dump_run_name: Option<&str> = INVESTIGATION_ENV.dump_run_name.as_deref();

            // ADR-028 iter-292: per-layer dispatch attribution
            // (HF2Q_PER_LAYER_DISP=1). Snapshot dispatch_count at layer start;
            // diff at layer end gives dispatches-per-layer-type.  Localizes
            // the iter-291 +72 mat-vec gap (sliding vs full-attn layers).
            let per_layer_disp_enabled = std::env::var("HF2Q_PER_LAYER_DISP").as_deref() == Ok("1");
            let mut per_layer_disp_log: Vec<(usize, bool, u64)> = Vec::new();
            let ctx = LayerCtx {
                seq_pos,
                hidden_size: hs,
                kv_info: &kv_info,
                dump_layers,
                dump_detail_layer,
                dump_sliding_l0,
                dump_run_name,
                dual_buffer_splits: &dual_buffer_splits,
                per_layer_disp_enabled,
                tq_scale_factor_d512,
                tq_codebook_bits,
                use_native_hb_sdpa,
                dump_all_cache_eff,
            };
            if let Some(worker_reg) = maybe_worker_reg {
                // ADR-031 Phase B: parallel-encode path.
                //
                // Serial pre-split: encode layers 0..PARALLEL_SPLIT_START.
                // This range includes the default dual_buffer_splits=[2] commit,
                // which fires inside encode_one_layer when (layer_idx+1)==2, i.e.,
                // after layer 1.  After the serial pre-split loop, `s` holds the
                // work for layers 2..PARALLEL_SPLIT_START (layers 0+1 were already
                // committed to buf0 inside the loop).
                const PARALLEL_SPLIT_START: usize = 4;
                for layer_idx in 0..PARALLEL_SPLIT_START.min(num_layers) {
                    self.encode_one_layer(
                        layer_idx, &ctx, &mut s, exec, reg,
                        profile, &mut per_layer_disp_log, &mut total_dispatches,
                    )?;
                }

                if PARALLEL_SPLIT_START < num_layers {
                    // Commit the post-serial-pre-split session (contains layers
                    // 2..PARALLEL_SPLIT_START work).  Its CB must be queued BEFORE
                    // the worker's chunk-A CB — commit-order serialization (INV-7).
                    let _enc = std::mem::replace(
                        &mut s,
                        exec.begin().map_err(|e| anyhow::anyhow!("parallel pre-commit begin: {e}"))?,
                    ).commit();

                    // Parallel chunk split: naive midpoint of PARALLEL_SPLIT_START..num_layers.
                    let mid = PARALLEL_SPLIT_START
                        + (num_layers - PARALLEL_SPLIT_START) / 2;
                    let range_a = PARALLEL_SPLIT_START..mid;
                    let range_b = mid..num_layers;

                    // Build a parallel LayerCtx with empty dual_buffer_splits so
                    // encode_one_layer's inline split-and-commit does NOT fire
                    // mid-chunk (INV-11 / R-B5).
                    let chunk_dual_buffer_splits: &[usize] = &[];
                    let parallel_ctx = LayerCtx {
                        dual_buffer_splits: chunk_dual_buffer_splits,
                        ..ctx
                    };

                    // encode_parallel_layers_chunked encodes range_b into s (session_b)
                    // on the main thread, while the worker encodes range_a into a
                    // fresh session_a it creates internally.
                    //
                    // R-B9 / iter-2 final: helper returns (Option<KernelRegistry>,
                    // Result<()>) so the registry travels back on BOTH success
                    // and encode-failure paths.  We store it into
                    // returned_worker_reg BEFORE propagating any error so the
                    // post-IIFE put_worker_registry always finds it.  Only the
                    // rare submit-failure / mpsc-drop paths return None (worker
                    // thread already destroyed the registry).
                    let (maybe_returned_reg, helper_outcome) = self.encode_parallel_layers_chunked(
                        range_a,
                        range_b,
                        &parallel_ctx,
                        &mut s,
                        exec,
                        reg,
                        worker_reg,
                        profile,
                        &mut per_layer_disp_log,
                        &mut total_dispatches,
                    );
                    if let Some(reg) = maybe_returned_reg {
                        returned_worker_reg = Some(reg);
                    }
                    helper_outcome?;
                } else {
                    // Edge case: num_layers <= PARALLEL_SPLIT_START (all layers done serially).
                    returned_worker_reg = Some(worker_reg);
                }
            } else {
                // Default serial path — zero behavior change vs HEAD e86831ab.
                for layer_idx in 0..num_layers {
                    self.encode_one_layer(
                        layer_idx, &ctx, &mut s, exec, reg,
                        profile, &mut per_layer_disp_log, &mut total_dispatches,
                    )?;
                }
            }

            // ADR-028 iter-292: dump per-layer dispatch counts post-loop.
            if per_layer_disp_enabled && !per_layer_disp_log.is_empty() {
                let n_sliding = per_layer_disp_log.iter().filter(|(_, s, _)| *s).count();
                let n_full = per_layer_disp_log.len() - n_sliding;
                let total_sliding: u64 = per_layer_disp_log.iter()
                    .filter(|(_, s, _)| *s).map(|(_, _, n)| *n).sum();
                let total_full: u64 = per_layer_disp_log.iter()
                    .filter(|(_, s, _)| !*s).map(|(_, _, n)| *n).sum();
                let avg_sliding = if n_sliding > 0 { total_sliding / n_sliding as u64 } else { 0 };
                let avg_full = if n_full > 0 { total_full / n_full as u64 } else { 0 };
                eprintln!("[PER_LAYER_DISP] sliding_layers={} (avg {} disp/layer, total {})",
                    n_sliding, avg_sliding, total_sliding);
                eprintln!("[PER_LAYER_DISP] full_layers={} (avg {} disp/layer, total {})",
                    n_full, avg_full, total_full);
                for (idx, sliding, count) in &per_layer_disp_log {
                    eprintln!("[PER_LAYER_DISP]   L{:02} {} {} disp",
                        idx, if *sliding { "SLID" } else { "FULL" }, count);
                }
            }

            // --- Body/Head timing split (HF2Q_SPLIT_TIMING=1) ---
            // Inserts a commit_and_wait between layers and head to measure each
            // GPU section separately. Adds ~50μs sync overhead — measurement only.
            let body_dispatches = total_dispatches;
            let split_timing = INVESTIGATION_ENV.split_timing;
            // ADR-028 iter-312 — group-stats dump for barrier audit.
            let group_stats_enabled = std::env::var("HF2Q_GROUP_STATS")
                .ok()
                .as_deref()
                .map_or(false, |v| v == "1");
            if split_timing {
                let body_barriers = s.barrier_count();
                if group_stats_enabled {
                    s.dump_group_stats();
                }
                let (enc_ns, gpu_ns) = s.finish_with_timing(session_start)
                    .map_err(|e| anyhow::anyhow!("body finish: {e}"))?;
                eprintln!("  [SPLIT] BODY: encode={:.2}ms gpu={:.2}ms dispatches={} barriers={}",
                    enc_ns as f64 / 1e6, gpu_ns as f64 / 1e6, body_dispatches, body_barriers);
                // Start a new session for the head
                s = exec.begin().map_err(|e| anyhow::anyhow!("head session: {e}"))?;
            } else if group_stats_enabled {
                s.dump_group_stats();
            }

            // --- 3. Final norm + lm_head + softcap + argmax (all GPU) ---

            // GPU final RMS norm: hidden → norm_out
            s.barrier_between(
                &[&self.activations.hidden, &self.final_norm],
                &[&self.activations.norm_out],
            );
            s.rms_norm(
                reg, metal_dev,
                &self.activations.hidden,
                &self.final_norm,
                &self.activations.norm_out,
                &self.activations.norm_params,
                1, hs as u32,
            ).map_err(|e| anyhow::anyhow!("final norm: {e}"))?;
            total_dispatches += 1;

            // --- ADR-009 Phase 3A: boundary dump at specific token position ---
            if dump_pos == Some(seq_pos) {
                // Finish session to read GPU buffers.
                s.finish().map_err(|e| anyhow::anyhow!("dump boundary finish: {e}"))?;
                // Pre-lm_head = final_norm applied to hidden.
                dumps::dump_f32(&self.activations.norm_out, hs,
                    "pre_lmhead", None, seq_pos)?;
                // Re-begin session for lm_head + argmax.
                s = exec.begin().map_err(|e| anyhow::anyhow!("dump boundary re-begin: {e}"))?;
            }

            // GPU lm_head: prefer Q6_K-native (HF2Q_LMHEAD_Q6K=1, ADR-028
            // iter-188), then Q8_0 (HF2Q_LMHEAD_Q8 auto for large vocab),
            // then F16 dense.
            if let Some(ref q6k) = self.lm_head_q6k {
                s.barrier_between(
                    &[&self.activations.norm_out, &q6k.buffer],
                    &[&self.activations.logits],
                );
                dispatch_qmatmul(
                    &mut s, reg, dev,
                    &self.activations.norm_out,
                    q6k,
                    &self.activations.logits,
                    1,
                    crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
                )?;
                total_dispatches += 1;
            } else if let Some(ref q8) = self.lm_head_q8 {
                s.barrier_between(
                    &[&self.activations.norm_out, &q8.buffer],
                    &[&self.activations.logits],
                );
                dispatch_qmatmul(
                    &mut s, reg, dev,
                    &self.activations.norm_out,
                    q8,
                    &self.activations.logits,
                    1,
                    crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
                )?;
                total_dispatches += 1;
            } else if let Some(ref lm_head_f16) = self.lm_head_f16 {
                // Mixed-precision mat-vec (F32 input × F16 weights → F32 output).
                // Single dispatch replaces the old 3-dispatch path (cast + gemm + cast).
                s.barrier_between(
                    &[&self.activations.norm_out, lm_head_f16],
                    &[&self.activations.logits],
                );
                let gemm_params = DenseGemmF16Params {
                    m: 1,
                    n: vocab_size as u32,
                    k: hs as u32,
                };
                mlx_native::ops::dense_gemm::dispatch_dense_matvec_f16w_f32io(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.norm_out,  // F32 input (no cast needed)
                    lm_head_f16,                 // F16 weights
                    &self.activations.logits,    // F32 output (no cast needed)
                    &gemm_params,
                ).map_err(|e| anyhow::anyhow!("lm_head mixed-precision: {e}"))?;
                total_dispatches += 1;
            } else {
                anyhow::bail!("Single-session forward requires GPU lm_head (F16 weight)");
            }

            // GPU softcap (if configured)
            if let Some(cap) = self.final_logit_softcapping {
                s.barrier_between(
                    &[&self.activations.logits],
                    &[&self.activations.logits],  // in-place
                );
                mlx_native::ops::softcap::dispatch_softcap(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.logits,
                    &self.activations.logits,
                    &self.activations.softcap_params,
                    cap,
                ).map_err(|e| anyhow::anyhow!("GPU softcap: {e}"))?;
                total_dispatches += 1;
            }

            // GPU argmax
            s.barrier_between(
                &[&self.activations.logits],
                &[&self.activations.argmax_index, &self.activations.argmax_value],
            );
            mlx_native::ops::argmax::dispatch_argmax_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.activations.logits,
                &self.activations.argmax_index,
                &self.activations.argmax_value,
                &self.activations.argmax_params,
                vocab_size as u32,
            ).map_err(|e| anyhow::anyhow!("GPU argmax: {e}"))?;
            total_dispatches += 1;


            // === ONE finish() for the entire forward pass ===
            let head_dispatches = total_dispatches - body_dispatches;
            let barrier_count = s.barrier_count();
            let is_recording = s.is_recording();
            if is_recording {
                let (enc_ns, gpu_ns, fusions, reordered, b0, b1) =
                    s.finish_optimized_with_timing(reg, metal_dev, session_start)
                        .map_err(|e| anyhow::anyhow!("optimized session finish: {e}"))?;
                if INVESTIGATION_ENV.mlx_timing {
                    eprintln!("  [TIMING] encode={:.2}ms gpu_wait={:.2}ms dispatches={} barriers={}",
                        enc_ns as f64 / 1e6, gpu_ns as f64 / 1e6, total_dispatches, barrier_count);
                    eprintln!("  [GRAPH_OPT] fusions={} reordered={} barriers={}+{}",
                        fusions, reordered, b0, b1);
                    // ADR-028 iter-400: per-barrier wall measurement.
                    let bns = mlx_native::barrier_total_ns();
                    if bns > 0 && barrier_count > 0 {
                        eprintln!("  [BARRIER_PROFILE] total_ns={} per_barrier_ns={}",
                            bns, bns / barrier_count as u64);
                    }
                }
            } else {
                let head_barriers = barrier_count; // snapshot before finish consumes s
                let (enc_ns, gpu_ns) = s.finish_with_timing(session_start)
                    .map_err(|e| anyhow::anyhow!("single session finish: {e}"))?;
                if INVESTIGATION_ENV.mlx_timing {
                    // ADR-028 iter-400: per-barrier wall measurement.
                    let bns = mlx_native::barrier_total_ns();
                    if bns > 0 && barrier_count > 0 {
                        eprintln!("  [BARRIER_PROFILE] total_ns={} per_barrier_ns={}",
                            bns, bns / barrier_count as u64);
                    }
                    if split_timing {
                        eprintln!("  [SPLIT] HEAD: encode={:.2}ms gpu={:.2}ms dispatches={} barriers={}",
                            enc_ns as f64 / 1e6, gpu_ns as f64 / 1e6, head_dispatches, head_barriers);
                    } else {
                        eprintln!("  [TIMING] encode={:.2}ms gpu_wait={:.2}ms dispatches={} barriers={}",
                            enc_ns as f64 / 1e6, gpu_ns as f64 / 1e6, total_dispatches, barrier_count);
                    }
                }
            }
            Ok(())
        })();
        // ADR-031 Phase B: restore worker_registry into gpu now that gpu.split()
        // borrow has been released (the inner block above has exited).
        // R-B9 / iter-2 final: this put runs UNCONDITIONALLY regardless of
        // whether the IIFE above succeeded or errored.  The registry handle
        // always rejoins GpuContext, eliminating the prior leak on encode-Err
        // paths.
        if let Some(wreg) = returned_worker_reg {
            gpu.put_worker_registry(wreg);
        }
        // Propagate any error from the IIFE AFTER the registry has been
        // safely restored.  See INV-4 / load-bearing recv comment in
        // encode_parallel_layers_chunked for the parallel symmetry: there
        // we wait unconditionally, here we restore unconditionally.
        parallel_block_result?;

        let session_us = session_start.elapsed().as_secs_f64() * 1e6;

        // --- ADR-009 Phase 3A: dump post-lm_head logits at boundary position ---
        if dump_pos == Some(seq_pos) {
            dumps::dump_f32(&self.activations.logits, vocab_size,
                "logits", None, seq_pos)?;
            // Also dump top-10 logits for quick inspection.
            let logits_data: &[f32] = self.activations.logits.as_slice()
                .map_err(|e| anyhow::anyhow!("dump top-10 read: {e}"))?;
            let mut indexed: Vec<(usize, f32)> = logits_data[..vocab_size]
                .iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("[DUMP] top-10 logits at pos {seq_pos}:");
            for (tok_id, logit) in indexed.iter().take(10) {
                eprintln!("  tok={tok_id:>6} logit={logit:.6}");
            }
        }

        // Read the Q8 GPU argmax result (8 bytes: 1 u32 index + 1 f32 value).
        let gpu_top1: u32 = {
            let idx: &[u32] = self.activations.argmax_index.as_slice()
                .map_err(|e| anyhow::anyhow!("argmax read: {e}"))?;
            idx[0]
        };

        // Q8 coarse → F32 exact rerank.
        //
        // Data shows Q8_0 lm_head adds ~2.5–5e-3 logit noise. Any token within
        // that envelope of top-1 has non-trivial chance of flipping. The pad
        // case is the most visible symptom; the mechanism is symmetric.
        //
        // Fix: keep Q8 for coarse scoring, but for the small set of tokens
        // plausibly eligible to win, recompute exact F32 logits from the F32
        // `embed_weight` (already resident) and take argmax on those.
        //
        // Candidate set (O(~100) tokens):
        //   - top-K Q8 tokens (K=64)
        //   - all tokens within delta=0.01 of Q8 top-1
        //   - special tokens: 0 <pad>, 1 <eos>, 2 <bos>, 105 <|turn>, 106 <turn|>
        //
        // Rerank is skipped when lm_head is already F16 (no coarse noise to
        // correct) or when HF2Q_LMHEAD_RERANK=0.
        // ADR-028 iter-188: Q6_K-direct lm_head also has quantization noise
        // and benefits from rerank against the F16 oracle (when in compare
        // mode).  Rerank fires for any quantized lm_head path.
        let rerank_active = (self.lm_head_q8.is_some() || self.lm_head_q6k.is_some())
            && !INVESTIGATION_ENV.lmhead_rerank_disabled;
        let token_id: u32 = if rerank_active {
            // CPU candidate selection via threshold scan over the full Q8
            // logits. GPU top-K was explored but a single-threadgroup
            // top-K on vocab=262144 serializes phase 2 onto one thread
            // and costs ~5 ms/token — worse than the ~40 μs CPU scan.
            //
            // Algorithm: read the Q8 top-1 value (from the GPU argmax
            // output), then collect all tokens with logit ≥ top1 - delta,
            // plus specials. Delta is chosen larger than the observed
            // Q8 noise envelope (~5e-3) so the true winner is always in
            // the set.
            let top1_q8_val: f32 = {
                let v: &[f32] = self.activations.argmax_value.as_slice()
                    .map_err(|e| anyhow::anyhow!("argmax_value read: {e}"))?;
                v[0]
            };
            // Headroom for Q8 noise. Empirical Q8 noise envelope is ~5e-3
            // per logit, so delta=0.5 is a comfortable ~100× margin. The
            // candidate set remains small (~10–100 tokens typically) because
            // real top-K distributions fall off quickly below the winner.
            let delta: f32 = 0.5;
            let threshold = top1_q8_val - delta;

            let logits: &[f32] = self.activations.logits.as_slice()
                .map_err(|e| anyhow::anyhow!("rerank logits read: {e}"))?;
            let hidden: &[f32] = self.activations.norm_out.as_slice()
                .map_err(|e| anyhow::anyhow!("rerank norm_out read: {e}"))?;
            let embed_f32: &[f32] = self.embed_weight.as_slice()
                .map_err(|e| anyhow::anyhow!("rerank embed read: {e}"))?;

            let mut candidates: Vec<u32> = Vec::with_capacity(64);
            for (i, &v) in logits[..vocab_size].iter().enumerate() {
                if v >= threshold {
                    candidates.push(i as u32);
                }
            }
            // Specials always included.
            for sp in [0u32, 1, 2, 105, 106] {
                if (sp as usize) < vocab_size {
                    candidates.push(sp);
                }
            }
            candidates.sort_unstable();
            candidates.dedup();

            // Exact F32 rerank via hidden · embed_row. Softcap is monotonic
            // so skipping it doesn't change argmax order. F64 accumulator
            // for precision; the set is tiny so cost is negligible.
            let mut best_tok: u32 = gpu_top1;
            let mut best_logit: f32 = f32::NEG_INFINITY;
            for &tok in &candidates {
                let row_off = (tok as usize) * hs;
                if row_off + hs > embed_f32.len() { continue; }
                let row = &embed_f32[row_off..row_off + hs];
                let mut acc: f64 = 0.0;
                for i in 0..hs {
                    acc += (hidden[i] as f64) * (row[i] as f64);
                }
                let l = acc as f32;
                if l > best_logit {
                    best_logit = l;
                    best_tok = tok;
                }
            }
            best_tok
        } else {
            gpu_top1
        };

        // Diagnostic: when <pad> (id 0) still wins AFTER rerank (or when
        // rerank is off and <pad> wins raw), dump top-10 Q8 logits so we
        // see whether pad is near-tie or a genuine model preference.
        if token_id == 0 {
            let logits: &[f32] = self.activations.logits.as_slice()
                .map_err(|e| anyhow::anyhow!("pad diag logits read: {e}"))?;
            let vocab = logits.len().min(vocab_size);
            let mut indexed: Vec<(usize, f32)> = logits[..vocab]
                .iter().enumerate().map(|(i, &v)| (i, v)).collect();
            // Sort descending by logit.  NaN logits would crash
            // `partial_cmp().unwrap()` — sort them as the smallest
            // possible value (they land at the end of the descending
            // sort so the top-10 print stays useful) and surface a
            // separate NaN-count line below so the operator can see
            // the model is producing garbage logits without losing
            // the diagnostic itself.  Surfaced 2026-04-25 by the
            // ADR-005 iter-103 vision smoke test (mmproj load was
            // making warmup logits NaN; the diagnostic block crashed
            // before printing anything).
            indexed.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let n_nan = indexed.iter().filter(|(_, v)| v.is_nan()).count();
            if n_nan > 0 {
                eprintln!(
                    "[PAD-DIAG] WARNING: {} of {} logits are NaN — model produced garbage; \
                     pad-win is a downstream symptom",
                    n_nan, vocab
                );
            }
            eprintln!("\n[PAD-DIAG] <pad> won at seq_pos={} (rerank={}). Top 10 Q8 logits:",
                seq_pos, if rerank_active { "on" } else { "off" });
            for (tok_id, logit) in indexed.iter().take(10) {
                eprintln!("  tok={tok_id:>6} logit={logit:>10.6}");
            }
            let pad_rank = indexed.iter().position(|&(i, _)| i == 0).unwrap_or(999);
            let pad_logit = logits[0];
            let rank1_logit = indexed[0].1;
            eprintln!("  <pad> rank={} logit={:.6}  vs top-1 logit={:.6}  gap={:.6e}",
                pad_rank, pad_logit, rank1_logit, (rank1_logit - pad_logit).abs());
        }

        if let Some(ref mut p) = profile {
            // Single session — report all time in S1, zero everything else
            let per_layer_us = session_us / num_layers as f64;
            for li in 0..num_layers {
                p.layer_s1_us[li] = per_layer_us;
                p.layer_cpu1_us[li] = 0.0;
                p.layer_s2_us[li] = 0.0;
                p.layer_cpu2_us[li] = 0.0;
                p.layer_s3_us[li] = 0.0;
                p.layer_cpu3_us[li] = 0.0;
                p.layer_s4_us[li] = 0.0;
                p.layer_cpu4_us[li] = 0.0;
                p.s2_dispatches[li] = 0;
                p.s3_dispatches[li] = 0;
                p.s4_dispatches[li] = 0;
            }
            p.head_session_us = 0.0; // included in single session
            p.head_cpu_us = 0.0;
            p.head_dispatches = total_dispatches;
            p.total_us = token_start.elapsed().as_secs_f64() * 1e6;
        }

        // -----------------------------------------------------------------
        // ADR-007 Gate H release-check plumbing (W12 iter-108a blocker #1).
        //
        // Three coupled hooks, gated by env vars cached at process start:
        //
        //   HF2Q_DECODE_INPUT_TOKENS  → replay fixed tokens (override pick).
        //                               The argmax + Q8 rerank above already
        //                               ran, so cosine/NLL captures see live
        //                               logits — only the *picked* token is
        //                               replaced.  Falls through to the
        //                               sampler's pick once the replay is
        //                               exhausted.
        //   HF2Q_EMIT_NLL             → emit `[HF2Q_NLL] step=N token=X
        //                               nll=Y` per token (matches the audit
        //                               binaries' `parse_nll_values` regex).
        //                               The NLL is computed on the FINAL
        //                               picked token (post-replay), so a
        //                               TQ-active run replaying dense
        //                               tokens reports each replayed
        //                               token's NLL under the TQ logits —
        //                               this is the ADR-007 Gate C PPL
        //                               input shape.
        //   HF2Q_DECODE_EMIT_TOKENS   → emit `[HF2Q_DECODE_EMIT] step=N
        //                               token=X` per token (matches
        //                               `parse_emitted_tokens`).
        //
        // All three were previously honored only by the audit-binary
        // wrappers (`src/bin/iter2{3,4,5}_audit.rs`) shelling out to a
        // separate hf2q subprocess; per-token NLL/replay never reached the
        // production decode path.  iter-108b will replace the audit
        // binaries with a release-check.sh-driven Gate 5; for that to
        // work, the production binary itself must honor the contract.
        //
        // iter-108a-fix (W15, 2026-04-25): the entire block below is gated
        // behind `self.gate_h_inactive` so that pre-iter-108a per-token
        // cost is restored when no Gate H hooks are armed (W14b regression
        // 95.0 → 100.6 tok/s baseline). The flag is computed once at
        // construction (`from_gguf_with_options`) and refreshed only by
        // `set_decode_regime`; reading it here is a single field load + a
        // single branch, which LLVM/the M-series CPU hoists out of any
        // surrounding loop and skips entirely on the false branch.
        // -----------------------------------------------------------------
        if !self.gate_h_inactive {
            let env = &*INVESTIGATION_ENV;
            let step = self.decode_step;
            // Replay first — substitute picked token before NLL/emit so both
            // downstream observers see the SAME token id (otherwise a replay
            // run would emit replay tokens but NLL the original argmax pick).
            // W21 iter-108b: per-instance `replay_tokens` takes precedence
            // over the frozen env-var vector so the in-process two-regime
            // Gate H harness can switch replay sources between passes.
            let final_token = if !self.replay_tokens.is_empty()
                && (step as usize) < self.replay_tokens.len()
            {
                self.replay_tokens[step as usize]
            } else if !env.decode_input_tokens.is_empty()
                && (step as usize) < env.decode_input_tokens.len()
            {
                env.decode_input_tokens[step as usize]
            } else {
                token_id
            };
            if env.emit_nll {
                // token_nll_from_logits asserts token_id < vocab_size; replay
                // tokens come from the user, so guard against an out-of-vocab
                // entry rather than panicking the whole decode loop.
                if (final_token as usize) < self.vocab_size {
                    match self.token_nll_from_logits(final_token) {
                        Ok(nll) => eprintln!(
                            "[HF2Q_NLL] step={step} token={final_token} nll={nll:.6}"
                        ),
                        Err(e) => eprintln!(
                            "[HF2Q_NLL] step={step} token={final_token} error={e}"
                        ),
                    }
                } else {
                    eprintln!(
                        "[HF2Q_NLL] step={step} token={final_token} \
                         error=token_id_out_of_vocab vocab_size={}",
                        self.vocab_size
                    );
                }
            }
            if env.decode_emit_tokens {
                eprintln!("[HF2Q_DECODE_EMIT] step={step} token={final_token}");
            }
            // Only mutate decode_step when Gate H is active — pre-iter-108a
            // the field did not exist, and writing to it every token even
            // when no observer reads it is a per-token RMW that defeats
            // the rest of the elision.
            self.decode_step = self.decode_step.saturating_add(1);
            return Ok(final_token);
        }

        Ok(token_id)
    }

    // forward_prefill() is defined in forward_prefill.rs (ADR-009 Track 1).

    /// ADR-028 iter-123 / ADR-029 Phase 2 Shape S — serial spec-decode verify.
    ///
    /// Forwards each `tokens[i]` through the model at position `seq_pos + i`,
    /// collecting the model's argmax at each step. Returns `Vec<u32>` of
    /// argmaxes (length == `tokens.len()`).
    ///
    /// **Shape S contract**: each token is a full `forward_decode` (own
    /// `GraphSession` begin/finish + `commit_and_wait`). This runs at
    /// `K × default-decode-latency` — NO speedup vs default decode.
    ///
    /// Use case: byte-identity correctness gate for the `accept_prefix`
    /// wiring + `rollback_kv` helper. Shape B (batched single-pass) lands
    /// later for the actual speed lift.
    ///
    /// At greedy temperature, `forward_decode_verify_serial(&[t0, t1, t2])`
    /// produces argmaxes byte-identical to calling `forward_decode(t0)`
    /// then `forward_decode(t1)` then `forward_decode(t2)` independently.
    pub fn forward_decode_verify_serial(
        &mut self,
        tokens: &[u32],
        seq_pos: usize,
        gpu: &mut GpuContext,
    ) -> Result<Vec<u32>> {
        let mut argmaxes = Vec::with_capacity(tokens.len());
        for (i, &tok) in tokens.iter().enumerate() {
            let mut prof: Option<TokenProfile> = None;
            let argmax = self.forward_decode(tok, seq_pos + i, gpu, &mut prof)?;
            argmaxes.push(argmax);
        }
        Ok(argmaxes)
    }

    /// ADR-028 iter-123 / ADR-029 Phase 2 — KV-cache rollback after partial accept.
    ///
    /// Rolls back the last `trim` writes across all layers. Sliding-window
    /// caches wrap (write_pos modulo capacity); full-attention caches go
    /// monotonic. The math is delegated to
    /// [`crate::inference::spec_decode::verifier::rollback_kv_state`] —
    /// see its tests for invariants.
    ///
    /// After this call, the next `forward_decode`/`forward_decode_verify_serial`
    /// invocation resumes at `current_seq_pos - trim`. The K_packed/V_packed
    /// data past the new `seq_len` is left as garbage; this is safe because
    /// kernels only read `< seq_len` and writes always go to current
    /// `write_pos`.
    pub fn rollback_kv(&mut self, trim: usize) {
        for cache in &mut self.kv_caches {
            let (wp, sl) = crate::inference::spec_decode::verifier::rollback_kv_state(
                cache.write_pos,
                cache.seq_len,
                cache.capacity,
                cache.is_sliding,
                trim,
            );
            cache.write_pos = wp;
            cache.seq_len = sl;
        }
    }

    /// Per-kernel-type profiling forward pass.
    ///
    /// Breaks the single session into one session PER KERNEL TYPE PER LAYER,
    /// using `finish_with_timing` to measure GPU wait time for each group.
    ///
    /// This is intentionally slow (many sessions = many sync points) but gives
    /// us per-kernel-type GPU timing to compare against candle Phase 0 data.
    ///
    /// Gated by `HF2Q_MLX_KERNEL_PROFILE=1`.
    pub fn forward_decode_kernel_profile(
        &mut self,
        input_token: u32,
        seq_pos: usize,
        gpu: &mut GpuContext,
    ) -> Result<(u32, KernelTypeProfile)> {
        let hs = self.hidden_size;
        let num_layers = self.layers.len();
        let vocab_size = self.vocab_size;

        let mut kp = KernelTypeProfile {
            qkv_matmuls_us: vec![0.0; num_layers],
            head_norms_rope_us: vec![0.0; num_layers],
            kv_cache_copy_us: vec![0.0; num_layers],
            sdpa_us: vec![0.0; num_layers],
            o_proj_us: vec![0.0; num_layers],
            mlp_matmuls_us: vec![0.0; num_layers],
            moe_us: vec![0.0; num_layers],
            norms_adds_us: vec![0.0; num_layers],
            ..Default::default()
        };

        // Write position buffer
        {
            let pos_dst: &mut [u32] = self.activations.position.as_mut_slice()
                .map_err(|e| anyhow::anyhow!("position write: {e}"))?;
            pos_dst[0] = seq_pos as u32;
        }

        // KV cache bookkeeping
        let mut kv_info: Vec<(bool, usize, usize, usize)> = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let is_sliding = self.kv_caches[layer_idx].is_sliding;
            let write_pos = self.kv_caches[layer_idx].write_pos;
            let capacity = self.kv_caches[layer_idx].capacity;
            self.kv_caches[layer_idx].write_pos += 1;
            self.kv_caches[layer_idx].seq_len = self.kv_caches[layer_idx].seq_len.saturating_add(1)
                .min(capacity);
            let seq_len = self.kv_caches[layer_idx].seq_len;
            kv_info.push((is_sliding, write_pos, capacity, seq_len));
        }

        // iter-18 S2B: scale factor for kernel profile path (same OnceLock as forward_decode).
        let tq_scale_factor_d512: f32 = {
            static SCALE_FACTOR_KP: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
            *SCALE_FACTOR_KP.get_or_init(|| match std::env::var("HF2Q_SCALE_FORMULA").as_deref() {
                Ok("sqrt256") => 16.0_f32,
                Ok("sqrt512") => 512.0_f32.sqrt(),
                _ => 1.0_f32,
            })
        };

        // --- Embedding (tiny, single session) ---
        {
            let (exec, reg) = gpu.split();
            let dev = exec.device();
            let metal_dev = dev.metal_device();
            let mut s = exec.begin().map_err(|e| anyhow::anyhow!("embed begin: {e}"))?;
            mlx_native::ops::elementwise::embedding_gather_scale_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.embed_weight, &self.activations.hidden,
                input_token, hs, (hs as f32).sqrt(),
            ).map_err(|e| anyhow::anyhow!("embedding: {e}"))?;
            s.finish().map_err(|e| anyhow::anyhow!("embed finish: {e}"))?;
            // Embedding time intentionally not reported: trivial cost relative
            // to the per-layer kernel sessions profiled below.
        }

        // --- Per-layer kernel-type sessions ---
        //
        // clippy::needless_range_loop stays off here: the body writes into 8
        // parallel `kp.*_us` profile vectors and indexes `kv_info`/`self.kv_caches`
        // by layer_idx. Zipping all of them into one iterator chain would be
        // much less readable than the index form. Migration note: the `layer`
        // binding covers the per-layer config (head_dim, num_kv_heads, layer_type)
        // and attn/moe/norms accesses.
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..num_layers {
            let layer = &self.layers[layer_idx];
            let hd = layer.head_dim;
            let nkv = layer.num_kv_heads;
            let nh = self.num_attention_heads;
            let is_sliding = layer.layer_type == LayerType::Sliding;
            let eps = self.rms_norm_eps;
            let (kv_is_sliding, kv_write_pos, kv_capacity, kv_seq_len) = kv_info[layer_idx];
            let v_is_k = layer.attn.v_proj.is_none();

            // ============================================================
            // GROUP 1: QKV matmuls (pre-attn norm + Q + K + V projections)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("qkv begin L{layer_idx}: {e}"))?;

                // pre-attn norm
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.hidden,
                    &self.layers[layer_idx].norms.input_layernorm,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("pre-attn norm L{layer_idx}: {e}"))?;

                // Q proj
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].attn.q_proj, &self.activations.attn_q, 1,
                    crate::quantize::imatrix::ImatrixHint::Layered { tag: "attn_q", layer: layer_idx })?;
                // K proj
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].attn.k_proj, &self.activations.attn_k, 1,
                    crate::quantize::imatrix::ImatrixHint::Layered { tag: "attn_k", layer: layer_idx })?;
                // V proj (if not k_eq_v)
                if !v_is_k {
                    dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                        self.layers[layer_idx].attn.v_proj.as_ref().unwrap(),
                        &self.activations.attn_v, 1,
                        crate::quantize::imatrix::ImatrixHint::Layered { tag: "attn_v", layer: layer_idx })?;
                }

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("qkv finish L{layer_idx}: {e}"))?;
                kp.qkv_matmuls_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 2: Head norms + RoPE (fused Q norm+RoPE, fused K norm+RoPE, V norm)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("norms begin L{layer_idx}: {e}"))?;

                let ff_gpu = if is_sliding { None } else { Some(&self.activations.rope_freq_factors_gpu) };
                let theta = if is_sliding { self.rope_theta_sliding } else { self.rope_theta_global };
                let half_rope = (hd / 2) as u32;

                // Fused Q norm+RoPE
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_q, &self.activations.attn_q_normed,
                    Some(&self.layers[layer_idx].attn.q_norm_weight),
                    &self.activations.position, ff_gpu,
                    nh as u32, hd as u32, half_rope, eps, theta,
                ).map_err(|e| anyhow::anyhow!("fused Q L{layer_idx}: {e}"))?;

                // Fused K norm+RoPE
                mlx_native::ops::fused_head_norm_rope::dispatch_fused_head_norm_rope_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_k, &self.activations.attn_k_normed,
                    Some(&self.layers[layer_idx].attn.k_norm_weight),
                    &self.activations.position, ff_gpu,
                    nkv as u32, hd as u32, half_rope, eps, theta,
                ).map_err(|e| anyhow::anyhow!("fused K L{layer_idx}: {e}"))?;

                // V norm
                let hd_norm_params = if is_sliding {
                    &self.activations.norm_params_sliding_hd
                } else {
                    &self.activations.norm_params_global_hd
                };
                if v_is_k {
                    dispatch_rms_norm_unit_perhead(
                        s.encoder_mut(), reg, metal_dev,
                        &RmsNormPerHeadArgs {
                            input: &self.activations.attn_k,
                            output: &self.activations.attn_v,
                            params_buf: hd_norm_params,
                            rows: nkv as u32,
                            dim: hd as u32,
                        },
                    )?;
                } else {
                    dispatch_rms_norm_unit_perhead(
                        s.encoder_mut(), reg, metal_dev,
                        &RmsNormPerHeadArgs {
                            input: &self.activations.attn_v,
                            output: &self.activations.moe_expert_out,
                            params_buf: hd_norm_params,
                            rows: nkv as u32,
                            dim: hd as u32,
                        },
                    )?;
                }

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("norms finish L{layer_idx}: {e}"))?;
                kp.head_norms_rope_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            let v_src = if v_is_k { &self.activations.attn_v } else { &self.activations.moe_expert_out };

            // ============================================================
            // GROUP 3: KV cache Hadamard-quantize (2 dispatches, ADR-007)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("kv begin L{layer_idx}: {e}"))?;

                let cache_pos_val = if kv_is_sliding {
                    (kv_write_pos % kv_capacity) as u32
                } else {
                    kv_write_pos as u32
                };
                mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_k_normed,
                    &self.kv_caches[layer_idx].k_packed,
                    &self.kv_caches[layer_idx].k_norms,
                    nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                    kv_is_sliding,
                    Some(tq_scale_factor_d512),
                    None,
                ).map_err(|e| anyhow::anyhow!("hadamard_quantize K L{layer_idx}: {e}"))?;
                mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv(
                    s.encoder_mut(), reg, metal_dev,
                    v_src,
                    &self.kv_caches[layer_idx].v_packed,
                    &self.kv_caches[layer_idx].v_norms,
                    nkv as u32, hd as u32, kv_capacity as u32, cache_pos_val,
                    kv_is_sliding,
                    Some(tq_scale_factor_d512),
                    None,
                ).map_err(|e| anyhow::anyhow!("hadamard_quantize V L{layer_idx}: {e}"))?;

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("kv finish L{layer_idx}: {e}"))?;
                kp.kv_cache_copy_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 4: SDPA TQ (FWHT fused — Q rotation + output inv-rotation in-kernel)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("sdpa begin L{layer_idx}: {e}"))?;

                {
                    // ADR-009 Track 2 + iter-25 Subtask B fix: ring_start must be the
                    // physical slot of the OLDEST entry (not newest).
                    // kv_write_pos is pre-increment (the slot just written this step).
                    // After wrap: oldest = (kv_write_pos + 1) % capacity.
                    // The kernel formula: logical_idx = (k_pos - ring_start + cap) % cap
                    // maps ring_start → logical 0 (oldest). Matches HB dispatch.
                    let ring_start = if kv_is_sliding && kv_seq_len >= kv_capacity {
                        ((kv_write_pos + 1) % kv_capacity) as u32
                    } else {
                        0
                    };
                    let p = FlashAttnVecTqParams {
                        num_heads: nh as u32,
                        num_kv_heads: nkv as u32,
                        head_dim: hd as u32,
                        kv_seq_len: kv_seq_len as u32,
                        kv_capacity: kv_capacity as u32,
                        scale: 1.0,
                        mask_type: if is_sliding { 2 } else { 1 },
                        sliding_window: if is_sliding { self.sliding_window as u32 } else { 0 },
                        softcap: 0.0,
                        ring_start,
                        scale_factor_d512: tq_scale_factor_d512,
                    };
                    mlx_native::ops::flash_attn_vec_tq::flash_attn_vec_tq(
                        s.encoder_mut(), reg, dev,
                        &self.activations.attn_q_normed,
                        &self.kv_caches[layer_idx].k_packed,
                        &self.kv_caches[layer_idx].k_norms,
                        &self.kv_caches[layer_idx].v_packed,
                        &self.kv_caches[layer_idx].v_norms,
                        &self.activations.sdpa_out,
                        &self.activations.sdpa_tmp,
                        &p,
                    ).map_err(|e| anyhow::anyhow!("flash_attn_vec_tq L{layer_idx}: {e}"))?;
                }

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("sdpa finish L{layer_idx}: {e}"))?;
                kp.sdpa_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 5: O-proj matmul (1 dispatch)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("oproj begin L{layer_idx}: {e}"))?;

                dispatch_qmatmul(&mut s, reg, dev, &self.activations.sdpa_out,
                    &self.layers[layer_idx].attn.o_proj, &self.activations.attn_out, 1,
                    crate::quantize::imatrix::ImatrixHint::Layered { tag: "attn_output", layer: layer_idx })?;

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("oproj finish L{layer_idx}: {e}"))?;
                kp.o_proj_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 6: MLP matmuls (post-attn norm+add, pre-FF norm, gate, up, gelu_mul, down)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("mlp begin L{layer_idx}: {e}"))?;

                // Fused post-attn norm+add (needed to produce residual for MLP)
                mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.hidden, &self.activations.attn_out,
                    &self.layers[layer_idx].norms.post_attention_layernorm,
                    &self.activations.residual,
                    hs as u32, 1, eps,
                ).map_err(|e| anyhow::anyhow!("post-attn norm+add L{layer_idx}: {e}"))?;

                // Pre-FF norm
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.residual,
                    &self.layers[layer_idx].norms.pre_feedforward_layernorm,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("pre-FF norm L{layer_idx}: {e}"))?;

                // gate
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].mlp.gate_proj, &self.activations.mlp_gate, 1,
                    crate::quantize::imatrix::ImatrixHint::Layered { tag: "ffn_gate", layer: layer_idx })?;
                // up
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].mlp.up_proj, &self.activations.mlp_up, 1,
                    crate::quantize::imatrix::ImatrixHint::Layered { tag: "ffn_up", layer: layer_idx })?;
                // fused gelu_mul
                {
                    use mlx_native::ops::encode_helpers::{encode_with_args, KernelArg};
                    let n_elements_bytes = (self.intermediate_size as u32).to_ne_bytes();
                    let pipeline = reg.get_pipeline("fused_gelu_mul", metal_dev)?;
                    encode_with_args(
                        s.encoder_mut(), pipeline,
                        &[
                            (0, KernelArg::Buffer(&self.activations.mlp_gate)),
                            (1, KernelArg::Buffer(&self.activations.mlp_up)),
                            (2, KernelArg::Buffer(&self.activations.mlp_fused)),
                            (3, KernelArg::Bytes(&n_elements_bytes)),
                        ],
                        mlx_native::MTLSize::new(self.intermediate_size as u64, 1, 1),
                        mlx_native::MTLSize::new(
                            std::cmp::min(256, self.intermediate_size as u64), 1, 1),
                    );
                }
                // down
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.mlp_fused,
                    &self.layers[layer_idx].mlp.down_proj, &self.activations.mlp_down, 1,
                    crate::quantize::imatrix::ImatrixHint::Layered { tag: "ffn_down", layer: layer_idx })?;

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("mlp finish L{layer_idx}: {e}"))?;
                kp.mlp_matmuls_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 7: MoE (routing + experts)
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("moe begin L{layer_idx}: {e}"))?;

                // Post-FF norm 1
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.mlp_down,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm_1,
                    &self.activations.attn_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("post-FF norm 1 L{layer_idx}: {e}"))?;

                // Pre-FF norm 2
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.residual,
                    &self.layers[layer_idx].norms.pre_feedforward_layernorm_2,
                    &self.activations.moe_norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("pre-FF norm 2 L{layer_idx}: {e}"))?;

                // Router norm
                s.rms_norm(
                    reg, metal_dev,
                    &self.activations.residual,
                    &self.layers[layer_idx].moe.router_combined_weight,
                    &self.activations.norm_out,
                    &self.activations.norm_params,
                    1, hs as u32,
                ).map_err(|e| anyhow::anyhow!("router norm L{layer_idx}: {e}"))?;

                // Router proj
                dispatch_qmatmul(&mut s, reg, dev, &self.activations.norm_out,
                    &self.layers[layer_idx].moe.router_proj,
                    &self.activations.moe_router_logits, 1,
                    crate::quantize::imatrix::ImatrixHint::Layered { tag: "ffn_gate_inp", layer: layer_idx })?;

                // Fused MoE routing
                let num_experts = self.num_experts;
                let top_k = self.layers[layer_idx].moe.top_k;
                mlx_native::ops::fused_norm_add::dispatch_fused_moe_routing_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_router_logits,
                    &self.activations.moe_expert_ids,
                    &self.activations.moe_routing_weights_gpu,
                    &self.layers[layer_idx].moe.per_expert_scale,
                    num_experts as u32, top_k as u32,
                ).map_err(|e| anyhow::anyhow!("fused MoE routing L{layer_idx}: {e}"))?;

                // MoE experts (fused _id path)
                let moe_int = self.layers[layer_idx].moe.moe_intermediate_size;
                let use_fused_id = self.layers[layer_idx].moe.stacked_gate_up.is_some()
                    && self.layers[layer_idx].moe.stacked_down.is_some();
                if !use_fused_id {
                    anyhow::bail!("Kernel profile requires fused _id path (stacked weights). Layer {layer_idx} missing.");
                }

                let ggml_type_gu = self.layers[layer_idx].moe.gate_up_ggml_dtype;
                let ggml_type_dn = self.layers[layer_idx].moe.down_ggml_dtype;

                // gate_up _id
                let gu_params = mlx_native::GgmlQuantizedMatmulIdParams {
                    n_tokens: 1,
                    top_k: top_k as u32,
                    n: (2 * moe_int) as u32,
                    k: hs as u32,
                    n_experts: num_experts as u32,
                    expert_stride: self.layers[layer_idx].moe.gate_up_expert_stride,
                    ggml_type: ggml_type_gu,
                };
                s.quantized_matmul_id_ggml(
                    reg, dev,
                    &self.activations.moe_norm_out,
                    self.layers[layer_idx].moe.stacked_gate_up.as_ref().unwrap(),
                    &self.activations.moe_expert_ids,
                    &self.activations.moe_gate_up_id_out,
                    &gu_params,
                ).map_err(|e| anyhow::anyhow!("gate_up _id L{layer_idx}: {e}"))?;

                // Batched SwiGLU
                mlx_native::ops::moe_dispatch::moe_swiglu_batch_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_gate_up_id_out,
                    &self.activations.moe_swiglu_id_out,
                    moe_int, top_k,
                ).map_err(|e| anyhow::anyhow!("swiglu batch L{layer_idx}: {e}"))?;

                // down _id
                let dn_params = mlx_native::GgmlQuantizedMatmulIdParams {
                    n_tokens: top_k as u32,
                    top_k: 1,
                    n: hs as u32,
                    k: moe_int as u32,
                    n_experts: num_experts as u32,
                    expert_stride: self.layers[layer_idx].moe.down_expert_stride,
                    ggml_type: ggml_type_dn,
                };
                s.quantized_matmul_id_ggml(
                    reg, dev,
                    &self.activations.moe_swiglu_id_out,
                    self.layers[layer_idx].moe.stacked_down.as_ref().unwrap(),
                    &self.activations.moe_expert_ids,
                    &self.activations.moe_down_id_out,
                    &dn_params,
                ).map_err(|e| anyhow::anyhow!("down _id L{layer_idx}: {e}"))?;

                // Weighted sum
                mlx_native::ops::moe_dispatch::moe_weighted_sum_encode(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.moe_down_id_out,
                    &self.activations.moe_routing_weights_gpu,
                    &self.activations.moe_accum,
                    hs, top_k,
                ).map_err(|e| anyhow::anyhow!("weighted_sum L{layer_idx}: {e}"))?;

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("moe finish L{layer_idx}: {e}"))?;
                kp.moe_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }

            // ============================================================
            // GROUP 8: Fused norms/adds/end-of-layer
            // ============================================================
            {
                let (exec, reg) = gpu.split();
                let dev = exec.device();
                let metal_dev = dev.metal_device();
                let t0 = Instant::now();
                let mut s = exec.begin().map_err(|e| anyhow::anyhow!("norms_end begin L{layer_idx}: {e}"))?;

                // Fused post-FF norm2 + combine MLP+MoE
                mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.attn_out, &self.activations.moe_accum,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm_2,
                    &self.activations.mlp_down,
                    hs as u32, 1, eps,
                ).map_err(|e| anyhow::anyhow!("fused post-FF norm2+combine L{layer_idx}: {e}"))?;

                // Fused end-of-layer: post-FF norm + residual add + scalar mul
                let scalar_is_vector = self.layers[layer_idx].layer_scalar.element_count() > 1;
                mlx_native::ops::fused_norm_add::dispatch_fused_norm_add_scalar_f32(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.residual, &self.activations.mlp_down,
                    &self.layers[layer_idx].norms.post_feedforward_layernorm,
                    &self.activations.hidden,
                    &self.layers[layer_idx].layer_scalar,
                    1, hs as u32, eps, scalar_is_vector,
                ).map_err(|e| anyhow::anyhow!("fused end-of-layer L{layer_idx}: {e}"))?;

                let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                    .map_err(|e| anyhow::anyhow!("norms_end finish L{layer_idx}: {e}"))?;
                kp.norms_adds_us[layer_idx] = gpu_ns as f64 / 1000.0;
            }
        }

        // --- Head: final norm + lm_head + softcap + argmax ---
        {
            let (exec, reg) = gpu.split();
            let dev = exec.device();
            let metal_dev = dev.metal_device();
            let t0 = Instant::now();
            let mut s = exec.begin().map_err(|e| anyhow::anyhow!("head begin: {e}"))?;

            // Final RMS norm
            s.rms_norm(
                reg, metal_dev,
                &self.activations.hidden, &self.final_norm,
                &self.activations.norm_out, &self.activations.norm_params,
                1, hs as u32,
            ).map_err(|e| anyhow::anyhow!("final norm: {e}"))?;

            // ADR-029 §Decision item 3: kernel-profile lm_head — mirror
            // production single-session path at ~4818: prefer Q6_K-native
            // (HF2Q_LMHEAD_Q6K=1, ADR-028 iter-188/345, default-on when
            // token_embd.weight is Q6_K on-disk), then Q8_0 (auto for
            // big-vocab models like gemma4 262144), then F16 dense.
            //
            // Pre-ADR-029 this path only checked Q8_0/F16 and hard-failed
            // on gemma4-APEX-Q5_K_M (Q6_K token_embd) — blocking all
            // per-kernel-type buckets needed for MoE-1/-2/-3 audits.
            if let Some(ref q6k) = self.lm_head_q6k {
                s.barrier_between(
                    &[&self.activations.norm_out, &q6k.buffer],
                    &[&self.activations.logits],
                );
                dispatch_qmatmul(
                    &mut s, reg, dev,
                    &self.activations.norm_out,
                    q6k,
                    &self.activations.logits,
                    1,
                    crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
                )?;
            } else if let Some(ref q8) = self.lm_head_q8 {
                s.barrier_between(
                    &[&self.activations.norm_out, &q8.buffer],
                    &[&self.activations.logits],
                );
                dispatch_qmatmul(
                    &mut s, reg, dev,
                    &self.activations.norm_out,
                    q8,
                    &self.activations.logits,
                    1,
                    crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
                )?;
            } else if let Some(ref lm_head_f16) = self.lm_head_f16 {
                mlx_native::ops::elementwise::cast(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.norm_out, &self.activations.hidden_f16,
                    hs, CastDirection::F32ToF16,
                ).map_err(|e| anyhow::anyhow!("cast F32->F16: {e}"))?;

                let gemm_params = DenseGemmF16Params {
                    m: 1, n: vocab_size as u32, k: hs as u32,
                };
                mlx_native::ops::dense_gemm::dispatch_dense_gemm_f16(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.hidden_f16, lm_head_f16,
                    &self.activations.logits_f16, &gemm_params,
                ).map_err(|e| anyhow::anyhow!("dense_gemm_f16: {e}"))?;

                mlx_native::ops::elementwise::cast(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.logits_f16, &self.activations.logits,
                    vocab_size, CastDirection::F16ToF32,
                ).map_err(|e| anyhow::anyhow!("cast F16->F32: {e}"))?;
            } else {
                anyhow::bail!(
                    "Kernel profile requires GPU lm_head (Q6_K, Q8_0, or F16 weight)"
                );
            }

            // Softcap (params pre-initialized at model load time)
            if let Some(cap) = self.final_logit_softcapping {
                mlx_native::ops::softcap::dispatch_softcap(
                    s.encoder_mut(), reg, metal_dev,
                    &self.activations.logits, &self.activations.logits,
                    &self.activations.softcap_params, cap,
                ).map_err(|e| anyhow::anyhow!("GPU softcap: {e}"))?;
            }

            // Argmax (params pre-initialized at model load time)
            mlx_native::ops::argmax::dispatch_argmax_f32(
                s.encoder_mut(), reg, metal_dev,
                &self.activations.logits, &self.activations.argmax_index,
                &self.activations.argmax_value, &self.activations.argmax_params,
                vocab_size as u32,
            ).map_err(|e| anyhow::anyhow!("GPU argmax: {e}"))?;

            let (_enc_ns, gpu_ns) = s.finish_with_timing(t0)
                .map_err(|e| anyhow::anyhow!("head finish: {e}"))?;
            kp.lm_head_us = gpu_ns as f64 / 1000.0;
        }

        // Read argmax result
        let token_id: u32 = {
            let idx: &[u32] = self.activations.argmax_index.as_slice()
                .map_err(|e| anyhow::anyhow!("argmax read: {e}"))?;
            idx[0]
        };

        Ok((token_id, kp))
    }

    /// ADR-038 G4-CFA-3: Gemma 4 tree-verify forward pass.
    ///
    /// Allocate per-layer F32 (K, V) caches shaped `[num_kv_heads, kv_capacity, head_dim]`.
    ///
    /// Per ADR-038 §3.4.6 risk 4: head_dim and num_kv_heads vary per layer
    /// (sliding: 16 KV heads × 256 head_dim; global: 2 KV heads × 512 head_dim).
    /// The returned Vec has exactly `self.layers.len()` entries.
    ///
    /// # Safety (ADR-031) + INV-ORCH-LIFETIME (ADR-038 G4-CFA-5c)
    ///
    /// The returned MlxBuffer values are independent device allocations with no
    /// parallel-encode entanglement. They must not outlive the MlxDevice they were
    /// allocated against. When threaded through `Gemma4Eagle3Orchestrator` (the
    /// production path), the orchestrator MUST be dropped before the
    /// `GpuContext` that exposed this device — see the INV-ORCH-LIFETIME doc
    /// block on `Gemma4Eagle3Orchestrator` (eagle3_orchestrator.rs).
    pub fn alloc_tree_verify_kv_caches(
        &self,
        device: &mlx_native::MlxDevice,
        kv_capacity: usize,
    ) -> Result<Vec<(mlx_native::MlxBuffer, mlx_native::MlxBuffer)>> {
        use anyhow::anyhow;
        use mlx_native::DType;
        let num_layers = self.layers.len();
        let mut caches = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let lw = &self.layers[layer_idx];
            let nkv = lw.num_kv_heads;
            let d = lw.head_dim;
            let kv_elems = nkv
                .checked_mul(kv_capacity)
                .and_then(|v| v.checked_mul(d))
                .ok_or_else(|| anyhow!(
                    "alloc_tree_verify_kv_caches: KV size overflow at layer {layer_idx}"
                ))?;
            let kv_bytes_layer = kv_elems * std::mem::size_of::<f32>();
            let k = device
                .alloc_buffer(kv_bytes_layer, DType::F32, vec![nkv, kv_capacity, d])
                .map_err(|e| anyhow!(
                    "alloc_tree_verify_kv_caches: alloc F32 K cache layer {layer_idx}: {e}"
                ))?;
            let v = device
                .alloc_buffer(kv_bytes_layer, DType::F32, vec![nkv, kv_capacity, d])
                .map_err(|e| anyhow!(
                    "alloc_tree_verify_kv_caches: alloc F32 V cache layer {layer_idx}: {e}"
                ))?;
            caches.push((k, v));
        }
        Ok(caches)
    }

    /// Runs `tree_seq_len` draft tokens through all model layers in parallel
    /// (single forward pass). Each layer uses the caller-owned **persistent F32
    /// KV cache** `kv_caches_f32`; new K/V entries are appended at positions
    /// `[prefix_len, prefix_len + tree_seq_len)`.
    ///
    /// `kv_caches_f32` must have exactly `self.layers.len()` entries allocated
    /// via [`Self::alloc_tree_verify_kv_caches`]. The cache is validated on
    /// entry (length + per-layer shape checks); pass the same `&mut` Vec across
    /// prefill and all subsequent `run_iteration` calls for persistent KV context.
    ///
    /// Hidden states at layers in `hidden_collector.target_layer_ids()` are
    /// captured into `hidden_collector` for the EAGLE-3 drafter input.
    ///
    /// Returns logits `[tree_seq_len, vocab_size]` as a flat F32 Vec (row-major).
    ///
    /// # Safety (ADR-031)
    ///
    /// This function takes `&self` (not `&mut self`). The `kv_caches_f32` Vec is
    /// a separate caller-owned object — it does NOT alias `self`. No
    /// parallel-encoded tensors share lifetimes with the cache buffers.
    ///
    /// # Acceptance criteria
    ///
    /// AC-G4-3.1 through AC-G4-3.6 per ADR-038 §4.4.
    /// AC-G4-5c.1 per ADR-038 §4 (new persistent-cache entry point).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tree_verify_gpu_with_cache(
        &self,
        tree_tokens: &[u32],
        tree_mask: &[f32],
        tree_positions: &[u32],
        prefix_len: usize,
        kv_capacity: usize,
        gpu: &mut GpuContext,
        kv_caches_f32: &mut Vec<(mlx_native::MlxBuffer, mlx_native::MlxBuffer)>,
        hidden_collector: &mut crate::inference::spec_decode::eagle3::multi_layer_hidden::Eagle3HiddenCollector,
    ) -> Result<Vec<f32>> {
        use anyhow::{anyhow, ensure};
        use mlx_native::DType;

        if tree_tokens.is_empty() {
            return Err(anyhow!("forward_tree_verify_gpu: tree_tokens must be non-empty"));
        }
        let tree_seq_len = tree_tokens.len();
        let mask_stride = prefix_len
            .checked_add(tree_seq_len)
            .ok_or_else(|| anyhow!("forward_tree_verify_gpu: prefix_len + tree_seq_len overflow"))?;
        ensure!(
            tree_mask.len() == tree_seq_len * mask_stride,
            "forward_tree_verify_gpu: tree_mask len {} != tree_seq_len({}) * mask_stride({})",
            tree_mask.len(),
            tree_seq_len,
            mask_stride
        );
        ensure!(
            tree_positions.len() == tree_seq_len,
            "forward_tree_verify_gpu: tree_positions len {} != tree_seq_len {}",
            tree_positions.len(),
            tree_seq_len
        );
        ensure!(
            hidden_collector.seq_len() == tree_seq_len,
            "forward_tree_verify_gpu: collector seq_len {} != tree_seq_len {}",
            hidden_collector.seq_len(),
            tree_seq_len
        );
        ensure!(
            hidden_collector.hidden_size() == self.hidden_size,
            "forward_tree_verify_gpu: collector hidden_size {} != model hidden_size {}",
            hidden_collector.hidden_size(),
            self.hidden_size
        );
        ensure!(
            prefix_len + tree_seq_len <= kv_capacity,
            "forward_tree_verify_gpu: prefix_len {} + tree_seq_len {} > kv_capacity {}",
            prefix_len,
            tree_seq_len,
            kv_capacity
        );

        // Validate caller-owned KV cache: length + per-layer shape.
        // head_dim and num_kv_heads are read from self.layers[i] at both the
        // alloc site (alloc_tree_verify_kv_caches) and here — same source of
        // truth ensures the shapes are guaranteed identical.
        ensure!(
            kv_caches_f32.len() == self.layers.len(),
            "forward_tree_verify_gpu: kv_caches_f32.len() {} != self.layers.len() {}",
            kv_caches_f32.len(),
            self.layers.len()
        );
        for (layer_idx, (k_buf, v_buf)) in kv_caches_f32.iter().enumerate() {
            let lw = &self.layers[layer_idx];
            let expected = vec![lw.num_kv_heads, kv_capacity, lw.head_dim];
            ensure!(
                k_buf.shape() == expected,
                "forward_tree_verify_gpu: K cache shape mismatch at layer {layer_idx}: \
                 got {:?} expected {:?}",
                k_buf.shape(), expected
            );
            ensure!(
                v_buf.shape() == expected,
                "forward_tree_verify_gpu: V cache shape mismatch at layer {layer_idx}: \
                 got {:?} expected {:?}",
                v_buf.shape(), expected
            );
        }

        hidden_collector.reset();

        let (exec, registry) = gpu.split();
        let device = exec.device();
        let metal_dev = device.metal_device();

        let h = self.hidden_size;
        let vocab_size = self.vocab_size;
        let eps = self.rms_norm_eps as f32;

        // --- Step 1: Embed tokens [tree_seq_len, hidden_size] F32 ---
        // Gemma 4 scales embeddings by sqrt(hidden_size).
        let hs_bytes = tree_seq_len * h * std::mem::size_of::<f32>();
        let mut hidden = {
            let scale = (h as f32).sqrt();
            let embed_f32: &[f32] = self.embed_weight
                .as_slice()
                .map_err(|e| anyhow!("forward_tree_verify_gpu: embed_weight slice: {e}"))?;
            let vocab_in_buf = embed_f32.len() / h;
            let mut cpu = vec![0.0f32; tree_seq_len * h];
            for (i, &tok) in tree_tokens.iter().enumerate() {
                ensure!(
                    (tok as usize) < vocab_in_buf,
                    "forward_tree_verify_gpu: token {} out of vocab {}",
                    tok, vocab_in_buf
                );
                let src = (tok as usize) * h;
                let dst = i * h;
                cpu[dst..dst + h].copy_from_slice(&embed_f32[src..src + h]);
            }
            for v in cpu.iter_mut() {
                *v *= scale;
            }
            let mut buf = device
                .alloc_buffer(hs_bytes, DType::F32, vec![tree_seq_len, h])
                .map_err(|e| anyhow!("forward_tree_verify_gpu: alloc embed: {e}"))?;
            buf.as_mut_slice::<f32>()
                .map_err(|e| anyhow!("forward_tree_verify_gpu: embed slice: {e}"))?
                .copy_from_slice(&cpu);
            buf
        };

        // --- Step 2: Upload tree_mask and tree_positions to GPU ---
        let mask_bytes = tree_seq_len * mask_stride * std::mem::size_of::<f32>();
        let mut tree_mask_buf = device
            .alloc_buffer(mask_bytes, DType::F32, vec![tree_seq_len, mask_stride])
            .map_err(|e| anyhow!("forward_tree_verify_gpu: alloc tree_mask: {e}"))?;
        tree_mask_buf
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("forward_tree_verify_gpu: tree_mask slice: {e}"))?
            .copy_from_slice(tree_mask);

        let pos_bytes = tree_seq_len * std::mem::size_of::<u32>();
        let mut tree_pos_buf = device
            .alloc_buffer(pos_bytes, DType::U32, vec![tree_seq_len])
            .map_err(|e| anyhow!("forward_tree_verify_gpu: alloc tree_pos: {e}"))?;
        tree_pos_buf
            .as_mut_slice::<u32>()
            .map_err(|e| anyhow!("forward_tree_verify_gpu: tree_pos slice: {e}"))?
            .copy_from_slice(tree_positions);

        // --- Step 3: Layer loop ---
        // Per ADR-038 §3.4.6 risk 3: build attn_shape_base INSIDE the loop
        // because head_dim and num_kv_heads vary per layer type.
        //
        // ADR-038 G4-CFA-5d diagnostic instrumentation (env-gated, zero cost
        // when disabled): set HF2Q_G4_TREE_VERIFY_NAN_DEBUG=1 to bisect which
        // layer first produces NaN/inf in the verifier forward pass. Dumps
        // per-layer first-5 hidden values + finite-ness count.
        let nan_debug = std::env::var("HF2Q_G4_TREE_VERIFY_NAN_DEBUG")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if nan_debug {
            let h_slice: &[f32] = hidden
                .as_slice()
                .map_err(|e| anyhow!("nan_debug: download embed: {e}"))?;
            let n_finite = h_slice.iter().filter(|x| x.is_finite()).count();
            let n_nan = h_slice.iter().filter(|x| x.is_nan()).count();
            let n_inf = h_slice.iter().filter(|x| x.is_infinite()).count();
            eprintln!(
                "[g4-nan-debug] post-embed: len={} finite={} nan={} inf={} first5={:?}",
                h_slice.len(), n_finite, n_nan, n_inf, &h_slice[..5.min(h_slice.len())]
            );
            // Also dump first prompt token's first 10 embedding values for
            // sanity check vs known-good model output. Real Gemma 4 31B
            // embeddings should be roughly N(0, ~0.05) scaled by sqrt(5376).
            let token_5_start = 5 * h.min(h_slice.len() / 6);  // 6th token (last for "The capital city of France is")
            let token_5_end = token_5_start + 10;
            if token_5_end <= h_slice.len() {
                eprintln!(
                    "[g4-nan-debug] embed[token5][0..10]={:?}",
                    &h_slice[token_5_start..token_5_end]
                );
            }
        }
        let num_layers = self.layers.len();
        for layer_idx in 0..num_layers {
            let lw = &self.layers[layer_idx];
            let nkv = lw.num_kv_heads;
            let d = lw.head_dim;
            let (rope_theta, freq_factors_present) = match lw.layer_type {
                LayerType::Sliding => (self.rope_theta_sliding, false),
                LayerType::Full => (self.rope_theta_global, true),
            };
            let freq_factors_buf: Option<&mlx_native::MlxBuffer> = if freq_factors_present {
                Some(&self.activations.rope_freq_factors_gpu)
            } else {
                None
            };
            let shape = super::gpu_full_attn::Gemma4TreeVerifyFullLayerShapeQ {
                attn: super::gpu_full_attn::Gemma4TreeVerifyLayerShape {
                    hidden_size: h as u32,
                    num_q_heads: self.num_attention_heads as u32,
                    num_kv_heads: nkv as u32,
                    head_dim: d as u32,
                    tree_seq_len: tree_seq_len as u32,
                    cache_prefix_len: prefix_len as u32,
                    kv_capacity: kv_capacity as u32,
                    mask_stride: mask_stride as u32,
                    rms_norm_eps: eps,
                    rope_theta: rope_theta as f32,
                    freq_factors_present,
                },
                intermediate_size: self.intermediate_size as u32,
            };
            let enc = device
                .command_encoder()
                .map_err(|e| anyhow!(
                    "forward_tree_verify_gpu: command_encoder layer {layer_idx}: {e}"
                ))?;
            let (ref mut k_cache, ref mut v_cache) = kv_caches_f32[layer_idx];
            hidden = super::gpu_full_attn::gemma4_tree_verify_full_layer_q(
                enc, device, registry,
                &hidden, &tree_mask_buf, &tree_pos_buf,
                k_cache, v_cache,
                lw, freq_factors_buf,
                shape,
            )
            .map_err(|e| anyhow!(
                "forward_tree_verify_gpu: layer {layer_idx}: {e}"
            ))?;

            // Capture hidden states for EAGLE-3 drafter input.
            if let Some(capture_idx) = hidden_collector.capture_index_for(layer_idx) {
                let slab: &[f32] = hidden
                    .as_slice()
                    .map_err(|e| anyhow!(
                        "forward_tree_verify_gpu: download hidden layer {layer_idx}: {e}"
                    ))?;
                hidden_collector
                    .write_layer_slab(capture_idx, slab)
                    .map_err(|e| anyhow!(
                        "forward_tree_verify_gpu: write capture layer {layer_idx}: {e}"
                    ))?;
            }

            if nan_debug {
                let h_slice: &[f32] = hidden
                    .as_slice()
                    .map_err(|e| anyhow!("nan_debug: download hidden layer {layer_idx}: {e}"))?;
                let n_nan = h_slice.iter().filter(|x| x.is_nan()).count();
                let n_inf = h_slice.iter().filter(|x| x.is_infinite()).count();
                let n_finite = h_slice.len() - n_nan - n_inf;
                // Only dump first time NaN/inf appears + every 5 layers + last layer
                let should_print = n_nan + n_inf > 0
                    || layer_idx % 5 == 0
                    || layer_idx == num_layers - 1;
                if should_print {
                    let max_abs = h_slice.iter()
                        .filter(|x| x.is_finite())
                        .fold(0.0f32, |acc, x| acc.max(x.abs()));
                    // ADR-038 CFA-5e: dump pos=0 AND pos=LAST so per-position
                    // bisection is possible (prior version only showed pos=0
                    // which is IDENTICAL across N for the same input token —
                    // useless for finding where pos≥1 diverges).
                    let pos0_first5: Vec<f32> = h_slice[..5.min(h_slice.len())].to_vec();
                    let last_pos_start = h_slice.len().saturating_sub(h);
                    let last_first5: Vec<f32> = h_slice[last_pos_start..last_pos_start + 5.min(h)].to_vec();
                    let last_max_abs = h_slice[last_pos_start..].iter()
                        .filter(|x| x.is_finite())
                        .fold(0.0f32, |acc, x| acc.max(x.abs()));
                    eprintln!(
                        "[g4-nan-debug] post-layer-{layer_idx} ({:?}): finite={} nan={} inf={} max_abs={:.3e} pos0_first5={:?} last_max_abs={:.3e} last_first5={:?}",
                        lw.layer_type, n_finite, n_nan, n_inf, max_abs,
                        pos0_first5, last_max_abs, last_first5,
                    );
                }
            }
        }

        ensure!(
            hidden_collector.is_complete(),
            "forward_tree_verify_gpu: hidden collector incomplete after {} layers",
            num_layers
        );

        // --- Step 5: Final norm + lm_head → logits [tree_seq_len, vocab_size] ---
        let rms_params_bytes = 2 * std::mem::size_of::<f32>();
        let mut rms_params_buf = device
            .alloc_buffer(rms_params_bytes, DType::F32, vec![2])
            .map_err(|e| anyhow!("forward_tree_verify_gpu: alloc rms_params: {e}"))?;
        {
            let s = rms_params_buf
                .as_mut_slice::<f32>()
                .map_err(|e| anyhow!("forward_tree_verify_gpu: rms_params slice: {e}"))?;
            s[0] = eps;
            s[1] = h as f32;
        }
        let normed_bytes = tree_seq_len * h * std::mem::size_of::<f32>();
        let normed = device
            .alloc_buffer(normed_bytes, DType::F32, vec![tree_seq_len, h])
            .map_err(|e| anyhow!("forward_tree_verify_gpu: alloc normed: {e}"))?;
        let logits_bytes = tree_seq_len * vocab_size * std::mem::size_of::<f32>();
        let logits = device
            .alloc_buffer(logits_bytes, DType::F32, vec![tree_seq_len, vocab_size])
            .map_err(|e| anyhow!("forward_tree_verify_gpu: alloc logits: {e}"))?;

        // Norm pass.
        let mut enc_head = device
            .command_encoder()
            .map_err(|e| anyhow!("forward_tree_verify_gpu: alloc enc_head: {e}"))?;
        mlx_native::ops::rms_norm::dispatch_rms_norm(
            &mut enc_head, registry, metal_dev,
            &hidden,
            &self.final_norm,
            &normed,
            &rms_params_buf,
            tree_seq_len as u32,
            h as u32,
        )
        .context("forward_tree_verify_gpu: final rms_norm")?;
        enc_head.memory_barrier();

        // lm_head projection: prefer quantized, fallback to F16 GEMM.
        if let Some(ref q6k) = self.lm_head_q6k {
            let mut s = exec
                .begin()
                .map_err(|e| anyhow!("forward_tree_verify_gpu: begin session q6k: {e}"))?;
            // Commit the norm encoder first.
            enc_head
                .commit_and_wait()
                .context("forward_tree_verify_gpu: enc_head commit (q6k path)")?;
            dispatch_qmatmul(
                &mut s, registry, device,
                &normed, q6k, &logits,
                tree_seq_len as u32,
                crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
            )
            .context("forward_tree_verify_gpu: lm_head q6k")?;
            s.finish()
                .map_err(|e| anyhow!("forward_tree_verify_gpu: finish q6k session: {e}"))?;
        } else if let Some(ref q8) = self.lm_head_q8 {
            enc_head
                .commit_and_wait()
                .context("forward_tree_verify_gpu: enc_head commit (q8 path)")?;
            let mut s = exec
                .begin()
                .map_err(|e| anyhow!("forward_tree_verify_gpu: begin session q8: {e}"))?;
            dispatch_qmatmul(
                &mut s, registry, device,
                &normed, q8, &logits,
                tree_seq_len as u32,
                crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
            )
            .context("forward_tree_verify_gpu: lm_head q8")?;
            s.finish()
                .map_err(|e| anyhow!("forward_tree_verify_gpu: finish q8 session: {e}"))?;
        } else if let Some(ref lm_head_f16) = self.lm_head_f16 {
            // Multi-row F16 lm_head: cast F32→F16, then GEMM, then cast F16→F32.
            let normed_f16_bytes = tree_seq_len * h * 2; // 2 bytes per f16
            let normed_f16 = device
                .alloc_buffer(normed_f16_bytes, DType::F16, vec![tree_seq_len, h])
                .map_err(|e| anyhow!("forward_tree_verify_gpu: alloc normed_f16: {e}"))?;
            let logits_f16_bytes = tree_seq_len * vocab_size * 2;
            let logits_f16 = device
                .alloc_buffer(logits_f16_bytes, DType::F16, vec![tree_seq_len, vocab_size])
                .map_err(|e| anyhow!("forward_tree_verify_gpu: alloc logits_f16: {e}"))?;

            mlx_native::ops::elementwise::cast(
                &mut enc_head, registry, metal_dev,
                &normed, &normed_f16,
                tree_seq_len * h, CastDirection::F32ToF16,
            )
            .map_err(|e| anyhow!("forward_tree_verify_gpu: cast F32→F16: {e}"))?;
            enc_head.memory_barrier();
            let gemm_params = DenseGemmF16Params {
                m: tree_seq_len as u32,
                n: vocab_size as u32,
                k: h as u32,
            };
            mlx_native::ops::dense_gemm::dispatch_dense_gemm_f16(
                &mut enc_head, registry, metal_dev,
                &normed_f16, lm_head_f16,
                &logits_f16, &gemm_params,
            )
            .map_err(|e| anyhow!("forward_tree_verify_gpu: dense_gemm_f16: {e}"))?;
            enc_head.memory_barrier();
            mlx_native::ops::elementwise::cast(
                &mut enc_head, registry, metal_dev,
                &logits_f16, &logits,
                tree_seq_len * vocab_size, CastDirection::F16ToF32,
            )
            .map_err(|e| anyhow!("forward_tree_verify_gpu: cast F16→F32: {e}"))?;
            enc_head
                .commit_and_wait()
                .context("forward_tree_verify_gpu: enc_head commit (f16 path)")?;
        } else {
            return Err(anyhow!(
                "forward_tree_verify_gpu: no lm_head weight available (need q6k, q8, or f16)"
            ));
        }

        // Optional final_logit_softcapping.
        if let Some(cap) = self.final_logit_softcapping {
            let n_elems = tree_seq_len * vocab_size;
            let softcap_bytes = 2 * std::mem::size_of::<f32>();
            let mut sc_params = device
                .alloc_buffer(softcap_bytes, DType::F32, vec![2])
                .map_err(|e| anyhow!("forward_tree_verify_gpu: alloc softcap_params: {e}"))?;
            {
                let p = sc_params
                    .as_mut_slice::<f32>()
                    .map_err(|e| anyhow!("forward_tree_verify_gpu: softcap_params slice: {e}"))?;
                p[0] = cap;
                p[1] = f32::from_bits(n_elems as u32);
            }
            let mut enc_sc = device
                .command_encoder()
                .map_err(|e| anyhow!("forward_tree_verify_gpu: alloc enc_sc: {e}"))?;
            mlx_native::ops::softcap::dispatch_softcap(
                &mut enc_sc, registry, metal_dev,
                &logits, &logits, &sc_params, cap,
            )
            .context("forward_tree_verify_gpu: softcap")?;
            enc_sc
                .commit_and_wait()
                .context("forward_tree_verify_gpu: softcap commit")?;
        }

        // Download logits to host.
        let logits_data = logits
            .as_slice::<f32>()
            .map_err(|e| anyhow!("forward_tree_verify_gpu: download logits: {e}"))?
            .to_vec();
        Ok(logits_data)
    }

    /// Back-compat delegating shim — same 8-parameter signature as pre-CFA-5c.
    ///
    /// Allocates a fresh per-layer F32 KV cache via
    /// [`Self::alloc_tree_verify_kv_caches`] and calls
    /// [`Self::forward_tree_verify_gpu_with_cache`]. All existing single-call
    /// tests (g4_cfa3_*) call this path and produce byte-identical results to
    /// the pre-CFA-5c behaviour because the fresh zero-init cache matches the
    /// old fresh-alloc-per-call behaviour.
    ///
    /// # Acceptance criteria (INV-1, AC-G4-5c.2)
    ///
    /// This shim exists solely for back-compat; new callers should allocate
    /// a cache via `alloc_tree_verify_kv_caches` and call
    /// `forward_tree_verify_gpu_with_cache` directly.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tree_verify_gpu(
        &self,
        tree_tokens: &[u32],
        tree_mask: &[f32],
        tree_positions: &[u32],
        prefix_len: usize,
        kv_capacity: usize,
        gpu: &mut GpuContext,
        hidden_collector: &mut crate::inference::spec_decode::eagle3::multi_layer_hidden::Eagle3HiddenCollector,
    ) -> Result<Vec<f32>> {
        let device = gpu.device().clone();
        let mut caches = self.alloc_tree_verify_kv_caches(&device, kv_capacity)?;
        self.forward_tree_verify_gpu_with_cache(
            tree_tokens,
            tree_mask,
            tree_positions,
            prefix_len,
            kv_capacity,
            gpu,
            &mut caches,
            hidden_collector,
        )
    }
}

// ── ADR-038 Step 4 G4-CFA-3 unit tests ──────────────────────────────────────
#[cfg(test)]
mod g4_cfa3_tests {
    use super::*;
    use mlx_native::{DType, MlxDevice};
    use crate::inference::spec_decode::eagle3::multi_layer_hidden::Eagle3HiddenCollector;
    use crate::inference::models::gemma4::model::{
        MlxModelWeights, MlxActivationBuffers, MlxDecoderLayerWeights,
        MlxAttentionWeights, MlxLayerNorms, MlxMlpWeights,
    };
    use crate::inference::models::gemma4::kv_cache::{MlxKvCache, DecodeRegime};
    use crate::serve::config::LayerType;
    use crate::serve::gpu::GpuContext;

    fn try_device() -> Option<MlxDevice> {
        MlxDevice::new().ok()
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn mk_rand_g4(seed: &mut u32, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| {
            *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((*seed as i32 as f32) / (i32::MAX as f32)) * scale
        }).collect()
    }

    fn alloc_f32_g4(data: &[f32], device: &MlxDevice) -> mlx_native::MlxBuffer {
        let n = data.len().max(1);
        let mut b = device.alloc_buffer(n * 4, DType::F32, vec![n]).expect("alloc_f32_g4");
        b.as_mut_slice::<f32>().expect("slice").copy_from_slice(&data[..n.min(data.len())]);
        b
    }

    fn alloc_placeholder_f32(device: &MlxDevice) -> mlx_native::MlxBuffer {
        device.alloc_buffer(4, DType::F32, vec![1]).expect("placeholder f32")
    }

    fn alloc_placeholder_u32(device: &MlxDevice) -> mlx_native::MlxBuffer {
        device.alloc_buffer(4, DType::U32, vec![1]).expect("placeholder u32")
    }

    fn mk_f32_qweight_g4(
        rows: usize, cols: usize, seed: &mut u32, scale: f32, device: &MlxDevice,
    ) -> crate::serve::forward_mlx_shared::MlxQWeight {
        use crate::serve::gpu::QuantWeightInfo;
        let data = mk_rand_g4(seed, rows * cols, scale);
        crate::serve::forward_mlx_shared::MlxQWeight {
            buffer: alloc_f32_g4(&data, device),
            info: QuantWeightInfo {
                ggml_dtype: mlx_native::GgmlType::F32,
                rows,
                cols,
            },
            affine: None,
            f16_shadow: None,
            decode_record_q6k_m1: std::sync::OnceLock::new(),
        }
    }

    fn mk_sliding_layer(
        hidden: usize, nq: usize, nkv: usize, head_dim: usize,
        intermediate: usize, seed: &mut u32, device: &MlxDevice,
    ) -> MlxDecoderLayerWeights {
        MlxDecoderLayerWeights {
            attn: MlxAttentionWeights {
                q_proj: mk_f32_qweight_g4(nq * head_dim, hidden, seed, 0.05, device),
                k_proj: mk_f32_qweight_g4(nkv * head_dim, hidden, seed, 0.05, device),
                v_proj: Some(mk_f32_qweight_g4(nkv * head_dim, hidden, seed, 0.05, device)),
                o_proj: mk_f32_qweight_g4(hidden, nq * head_dim, seed, 0.05, device),
                q_norm_weight: alloc_f32_g4(&vec![1.0f32; head_dim], device),
                k_norm_weight: alloc_f32_g4(&vec![1.0f32; head_dim], device),
            },
            mlp: MlxMlpWeights {
                gate_proj: mk_f32_qweight_g4(intermediate, hidden, seed, 0.05, device),
                up_proj: mk_f32_qweight_g4(intermediate, hidden, seed, 0.05, device),
                down_proj: mk_f32_qweight_g4(hidden, intermediate, seed, 0.05, device),
            },
            moe: crate::inference::models::gemma4::model::MlxMoeWeights::dense_placeholder(device)
                .expect("dense_placeholder"),
            norms: MlxLayerNorms {
                input_layernorm: alloc_f32_g4(&vec![1.0f32; hidden], device),
                post_attention_layernorm: alloc_f32_g4(&vec![1.0f32; hidden], device),
                pre_feedforward_layernorm: alloc_f32_g4(&vec![1.0f32; hidden], device),
                post_feedforward_layernorm: alloc_f32_g4(&vec![1.0f32; hidden], device),
                pre_feedforward_layernorm_2: alloc_f32_g4(&vec![1.0f32; hidden], device),
                post_feedforward_layernorm_1: alloc_f32_g4(&vec![1.0f32; hidden], device),
                post_feedforward_layernorm_2: alloc_f32_g4(&vec![1.0f32; hidden], device),
            },
            layer_scalar: {
                let mut b = device.alloc_buffer(4, DType::F32, vec![1]).expect("scalar");
                b.as_mut_slice::<f32>().expect("s")[0] = 1.0;
                b
            },
            head_dim,
            num_kv_heads: nkv,
            layer_type: LayerType::Sliding,
        }
    }

    fn mk_placeholder_kv_cache(device: &MlxDevice) -> MlxKvCache {
        let buf = || device.alloc_buffer(4, DType::F32, vec![1]).expect("kv_buf");
        MlxKvCache {
            k_packed: buf(), k_norms: buf(), v_packed: buf(), v_norms: buf(),
            capacity: 8, is_sliding: true, write_pos: 0, seq_len: 0,
        }
    }

    /// Build a minimal MlxModelWeights suitable for forward_tree_verify_gpu tests.
    ///
    /// Uses only Sliding layers (no global), so rope_freq_factors_gpu is never
    /// accessed and can stay as a 1-element placeholder.
    ///
    /// `n_layers`: number of sliding layers.
    /// `hidden`, `nq`, `nkv`, `head_dim`, `intermediate`, `vocab`: model dims.
    fn mk_tiny_g4_model(
        n_layers: usize,
        hidden: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
        intermediate: usize,
        vocab: usize,
        seed: &mut u32,
        device: &MlxDevice,
    ) -> MlxModelWeights {
        // embed_weight: [vocab, hidden] F32 with small random values.
        let embed_data = mk_rand_g4(seed, vocab * hidden, 0.01);
        let embed_weight = alloc_f32_g4(&embed_data, device);

        // final_norm: [hidden] F32 ones.
        let final_norm = alloc_f32_g4(&vec![1.0f32; hidden], device);

        // lm_head_f16: [vocab, hidden] F16 zero-initialized (test only).
        // Actual F32→F16 conversion requires a Metal kernel; zeros still
        // produce finite logits after the rms_norm.
        let lm_head_f16 = {
            let bytes = vocab * hidden * 2;
            device.alloc_buffer(bytes, DType::F16, vec![vocab, hidden]).expect("lm_head_f16")
        };

        // Build layer weights (all sliding).
        let layers: Vec<MlxDecoderLayerWeights> = (0..n_layers)
            .map(|_| mk_sliding_layer(hidden, nq, nkv, head_dim, intermediate, seed, device))
            .collect();

        // KV caches: one per layer (unused by forward_tree_verify_gpu).
        let kv_caches: Vec<MlxKvCache> = (0..n_layers)
            .map(|_| mk_placeholder_kv_cache(device))
            .collect();

        // MlxActivationBuffers: only rope_freq_factors_gpu is accessed by
        // forward_tree_verify_gpu (for Full/global layers, which we don't have).
        // All other fields are placeholder 1-element buffers.
        let activations = MlxActivationBuffers {
            hidden: alloc_placeholder_f32(device),
            attn_q: alloc_placeholder_f32(device),
            attn_k: alloc_placeholder_f32(device),
            attn_out: alloc_placeholder_f32(device),
            norm_out: alloc_placeholder_f32(device),
            residual: alloc_placeholder_f32(device),
            mlp_gate: alloc_placeholder_f32(device),
            mlp_up: alloc_placeholder_f32(device),
            mlp_fused: alloc_placeholder_f32(device),
            mlp_down: alloc_placeholder_f32(device),
            sdpa_out: alloc_placeholder_f32(device),
            sdpa_tmp: alloc_placeholder_f32(device),
            norm_params: alloc_placeholder_f32(device),
            position: alloc_placeholder_u32(device),
            softcap_params: alloc_placeholder_f32(device),
            argmax_index: alloc_placeholder_u32(device),
            argmax_value: alloc_placeholder_f32(device),
            argmax_params: alloc_placeholder_f32(device),
            logits: alloc_placeholder_f32(device),
            moe_router_logits: alloc_placeholder_f32(device),
            moe_expert_out: alloc_placeholder_f32(device),
            moe_accum: alloc_placeholder_f32(device),
            moe_norm_out: alloc_placeholder_f32(device),
            router_norm_out: alloc_placeholder_f32(device),
            moe_expert_ids: alloc_placeholder_u32(device),
            moe_gate_up_id_out: alloc_placeholder_f32(device),
            moe_down_id_out: alloc_placeholder_f32(device),
            moe_swiglu_id_out: alloc_placeholder_f32(device),
            hidden_f16: device.alloc_buffer(2, DType::F16, vec![1]).expect("hidden_f16"),
            logits_f16: device.alloc_buffer(2, DType::F16, vec![1]).expect("logits_f16"),
            norm_params_sliding_hd: alloc_placeholder_f32(device),
            norm_params_global_hd: alloc_placeholder_f32(device),
            // rope_freq_factors_gpu: only accessed for Full (global) layers.
            // Since all layers here are Sliding, 1-element placeholder is safe.
            rope_freq_factors_gpu: alloc_placeholder_f32(device),
            attn_v: alloc_placeholder_f32(device),
            attn_q_normed: alloc_placeholder_f32(device),
            attn_k_normed: alloc_placeholder_f32(device),
            moe_routing_weights_gpu: alloc_placeholder_f32(device),
        };

        MlxModelWeights {
            embed_weight,
            layers,
            final_norm,
            lm_head_f16: Some(lm_head_f16),
            lm_head_q8: None,
            lm_head_q6k: None,
            hidden_size: hidden,
            vocab_size: vocab,
            num_attention_heads: nq,
            rms_norm_eps: 1e-6,
            final_logit_softcapping: None,
            kv_caches,
            activations,
            sliding_window: 4096,
            rope_theta_sliding: 10000.0,
            rope_theta_global: 1_000_000.0,
            num_experts: 0,
            intermediate_size: intermediate,
            dense_kvs: None,
            dense_kvs_snapshot_for_lcp: None,
            dense_sdpa_tmp: None,
            leg_hb_encoded: None,
            hybrid_kv: None,
            decode_step: 0,
            decode_regime: DecodeRegime::Default,
            gate_h_inactive: true,
            replay_tokens: Vec::new(),
            dump_dir_override: None,
            dump_all_cache_override: None,
            decode_step_dump_counter: 0,
            dflash_capture: None,
            decode_record_rms_norm_f32_hs: std::sync::OnceLock::new(),
        }
    }

    /// Build a causal tree mask [tree_seq_len, mask_stride] where each query
    /// attends to all prior positions (prefix) and itself.
    fn causal_mask_g4(tree_seq_len: usize, prefix_len: usize) -> Vec<f32> {
        let mask_stride = prefix_len + tree_seq_len;
        const ATTEND: f32 = 0.0;
        const BLOCK: f32 = -65504.0;
        let mut m = vec![BLOCK; tree_seq_len * mask_stride];
        for i in 0..tree_seq_len {
            for j in 0..prefix_len + i + 1 {
                if j < mask_stride {
                    m[i * mask_stride + j] = ATTEND;
                }
            }
        }
        m
    }

    // ── AC-G4-3.1 — single-iter end-to-end smoke test ────────────────────────

    /// AC-G4-3.1 — forward_tree_verify_gpu end-to-end: 2 sliding layers,
    /// tree_seq=3, vocab=64, asserts output logits shape [3×64] + all-finite.
    #[test]
    fn g4_cfa3_single_iter_end_to_end_2026_05_23() {
        let device = match try_device() {
            Some(d) => d,
            None => { eprintln!("skip: no Metal device"); return; }
        };
        let mut gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(_) => { eprintln!("skip: no Metal device (gpu)"); return; }
        };

        // head_dim=256 required by Gemma4TreeVerifyLayerShape validator.
        let hidden = 256usize;
        let nq = 1usize;
        let nkv = 1usize;
        let head_dim = 256usize;
        let intermediate = 256usize;
        let vocab = 256usize;
        let n_layers = 2usize;
        let tree_seq = 3usize;
        let prefix = 4usize;
        let kv_cap = 16usize;

        let mut seed = 0xABCD_u32;
        let model = mk_tiny_g4_model(
            n_layers, hidden, nq, nkv, head_dim, intermediate, vocab,
            &mut seed, &device,
        );

        let tokens: Vec<u32> = (0..tree_seq as u32).collect();
        let mask = causal_mask_g4(tree_seq, prefix);
        let positions: Vec<u32> = (prefix as u32..prefix as u32 + tree_seq as u32).collect();

        // Capture from layer 0 only.
        let mut collector = Eagle3HiddenCollector::new(
            vec![0], tree_seq, hidden,
        ).expect("collector");

        let logits = model.forward_tree_verify_gpu(
            &tokens, &mask, &positions, prefix, kv_cap,
            &mut gpu, &mut collector,
        ).expect("forward_tree_verify_gpu");

        assert_eq!(logits.len(), tree_seq * vocab,
            "logits shape: expected {} got {}", tree_seq * vocab, logits.len());
        assert!(logits.iter().all(|v| v.is_finite()),
            "logits contain non-finite values");
        assert!(collector.is_complete(), "collector should be complete");
    }

    // ── AC-G4-3.2 — multi-iteration cache continuity ──────────────────────────

    /// AC-G4-3.2 — forward_tree_verify_gpu is called 3 times in sequence;
    /// each call allocates fresh per-layer KV caches (independent) and returns
    /// correct shape. Verifies that repeated calls don't crash or corrupt.
    #[test]
    fn g4_cfa3_multi_iter_cache_continuity_2026_05_23() {
        let device = match try_device() {
            Some(d) => d,
            None => { eprintln!("skip: no Metal device"); return; }
        };
        let mut gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(_) => { eprintln!("skip: no Metal device (gpu)"); return; }
        };

        // head_dim=256 required by Gemma4TreeVerifyLayerShape validator.
        let hidden = 256usize;
        let nq = 1usize;
        let nkv = 1usize;
        let head_dim = 256usize;
        let intermediate = 256usize;
        let vocab = 256usize;
        let n_layers = 2usize;
        let tree_seq = 2usize;
        let kv_cap = 32usize;

        let mut seed = 0x1234_u32;
        let model = mk_tiny_g4_model(
            n_layers, hidden, nq, nkv, head_dim, intermediate, vocab,
            &mut seed, &device,
        );

        let tokens: Vec<u32> = vec![1, 2];
        let mut prev_argmax = None::<u32>;

        for iter in 0..3usize {
            let prefix = iter * tree_seq;
            let mask = causal_mask_g4(tree_seq, prefix);
            let positions: Vec<u32> = (prefix as u32..prefix as u32 + tree_seq as u32).collect();

            let mut collector = Eagle3HiddenCollector::new(
                vec![0], tree_seq, hidden,
            ).expect("collector");

            let logits = model.forward_tree_verify_gpu(
                &tokens, &mask, &positions, prefix, kv_cap,
                &mut gpu, &mut collector,
            ).unwrap_or_else(|e| panic!("iter {iter} failed: {e}"));

            assert_eq!(logits.len(), tree_seq * vocab,
                "iter {iter}: logits len mismatch");
            assert!(logits.iter().all(|v| v.is_finite()),
                "iter {iter}: non-finite logits");

            // Argmax of first token's logits should be deterministic per model.
            let argmax = logits[..vocab]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            if let Some(prev) = prev_argmax {
                assert_eq!(argmax, prev,
                    "iter {iter}: argmax changed from {prev} to {argmax} (non-deterministic)");
            }
            prev_argmax = Some(argmax);
        }
    }

    // ── AC-G4-3.3 — per-layer dispatch branch: sliding vs global ─────────────

    /// AC-G4-3.3 — A model with 1 sliding + 1 global layer both dispatch
    /// without error, and the output is finite. Exercises both LayerType
    /// branches of the layer loop in forward_tree_verify_gpu.
    #[test]
    fn g4_cfa3_per_layer_dispatch_layer_type_branch_2026_05_23() {
        let device = match try_device() {
            Some(d) => d,
            None => { eprintln!("skip: no Metal device"); return; }
        };
        let mut gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(_) => { eprintln!("skip: no Metal device (gpu)"); return; }
        };

        // head_dim must be 256 (sliding) or 512 (global) per validator.
        // Use nq=1 for both layers so num_attention_heads is consistent.
        // hidden=512 so nq*d_sliding=1*256=256 and nq*d_global=1*512=512 both fit.
        let hidden = 512usize;
        let nq_sliding = 1usize;
        let nkv_sliding = 1usize;
        let d_sliding = 256usize;  // LayerType::Sliding
        let nq_global = 1usize;
        let nkv_global = 1usize;
        let d_global = 512usize;   // LayerType::Full (global)
        let intermediate = 256usize;
        let vocab = 256usize;
        let tree_seq = 2usize;
        let prefix = 2usize;
        let kv_cap = 16usize;
        let global_head_dim_half = d_global / 2; // rope_freq_factors_gpu size = 256

        let mut seed = 0xBEEF_u32;

        // Build sliding layer.
        let sliding_layer = mk_sliding_layer(
            hidden, nq_sliding, nkv_sliding, d_sliding, intermediate,
            &mut seed, &device,
        );

        // Build global layer (LayerType::Full) manually.
        let global_layer = {
            let mut lw = mk_sliding_layer(
                hidden, nq_global, nkv_global, d_global, intermediate,
                &mut seed, &device,
            );
            lw.layer_type = LayerType::Full;
            lw
        };

        let embed_data = mk_rand_g4(&mut seed, vocab * hidden, 0.01);
        let embed_weight = alloc_f32_g4(&embed_data, &device);
        let final_norm = alloc_f32_g4(&vec![1.0f32; hidden], &device);
        let lm_head_f16 = device.alloc_buffer(vocab * hidden * 2, DType::F16, vec![vocab, hidden])
            .expect("lm_head_f16");

        let kv_caches = vec![
            mk_placeholder_kv_cache(&device),
            mk_placeholder_kv_cache(&device),
        ];

        // rope_freq_factors_gpu must be sized [global_head_dim/2] F32.
        let freq_factors = alloc_f32_g4(
            &vec![1.0f32; global_head_dim_half],
            &device,
        );

        let activations = MlxActivationBuffers {
            hidden: alloc_placeholder_f32(&device),
            attn_q: alloc_placeholder_f32(&device),
            attn_k: alloc_placeholder_f32(&device),
            attn_out: alloc_placeholder_f32(&device),
            norm_out: alloc_placeholder_f32(&device),
            residual: alloc_placeholder_f32(&device),
            mlp_gate: alloc_placeholder_f32(&device),
            mlp_up: alloc_placeholder_f32(&device),
            mlp_fused: alloc_placeholder_f32(&device),
            mlp_down: alloc_placeholder_f32(&device),
            sdpa_out: alloc_placeholder_f32(&device),
            sdpa_tmp: alloc_placeholder_f32(&device),
            norm_params: alloc_placeholder_f32(&device),
            position: alloc_placeholder_u32(&device),
            softcap_params: alloc_placeholder_f32(&device),
            argmax_index: alloc_placeholder_u32(&device),
            argmax_value: alloc_placeholder_f32(&device),
            argmax_params: alloc_placeholder_f32(&device),
            logits: alloc_placeholder_f32(&device),
            moe_router_logits: alloc_placeholder_f32(&device),
            moe_expert_out: alloc_placeholder_f32(&device),
            moe_accum: alloc_placeholder_f32(&device),
            moe_norm_out: alloc_placeholder_f32(&device),
            router_norm_out: alloc_placeholder_f32(&device),
            moe_expert_ids: alloc_placeholder_u32(&device),
            moe_gate_up_id_out: alloc_placeholder_f32(&device),
            moe_down_id_out: alloc_placeholder_f32(&device),
            moe_swiglu_id_out: alloc_placeholder_f32(&device),
            hidden_f16: device.alloc_buffer(2, DType::F16, vec![1]).expect("hidden_f16"),
            logits_f16: device.alloc_buffer(2, DType::F16, vec![1]).expect("logits_f16"),
            norm_params_sliding_hd: alloc_placeholder_f32(&device),
            norm_params_global_hd: alloc_placeholder_f32(&device),
            rope_freq_factors_gpu: freq_factors,
            attn_v: alloc_placeholder_f32(&device),
            attn_q_normed: alloc_placeholder_f32(&device),
            attn_k_normed: alloc_placeholder_f32(&device),
            moe_routing_weights_gpu: alloc_placeholder_f32(&device),
        };

        let model = MlxModelWeights {
            embed_weight,
            layers: vec![sliding_layer, global_layer],
            final_norm,
            lm_head_f16: Some(lm_head_f16),
            lm_head_q8: None,
            lm_head_q6k: None,
            hidden_size: hidden,
            vocab_size: vocab,
            num_attention_heads: nq_sliding, // nq_sliding == nq_global == 1
            rms_norm_eps: 1e-6,
            final_logit_softcapping: None,
            kv_caches,
            activations,
            sliding_window: 4096,
            rope_theta_sliding: 10000.0,
            rope_theta_global: 1_000_000.0,
            num_experts: 0,
            intermediate_size: intermediate,
            dense_kvs: None,
            dense_kvs_snapshot_for_lcp: None,
            dense_sdpa_tmp: None,
            leg_hb_encoded: None,
            hybrid_kv: None,
            decode_step: 0,
            decode_regime: DecodeRegime::Default,
            gate_h_inactive: true,
            replay_tokens: Vec::new(),
            dump_dir_override: None,
            dump_all_cache_override: None,
            decode_step_dump_counter: 0,
            dflash_capture: None,
            decode_record_rms_norm_f32_hs: std::sync::OnceLock::new(),
        };

        let tokens: Vec<u32> = vec![0, 1];
        let mask = causal_mask_g4(tree_seq, prefix);
        let positions: Vec<u32> = vec![prefix as u32, prefix as u32 + 1];
        let mut collector = Eagle3HiddenCollector::new(vec![0, 1], tree_seq, hidden)
            .expect("collector");

        let logits = model.forward_tree_verify_gpu(
            &tokens, &mask, &positions, prefix, kv_cap,
            &mut gpu, &mut collector,
        ).expect("forward sliding+global layers");

        assert_eq!(logits.len(), tree_seq * vocab);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite with mixed layer types");
    }

    // ── AC-G4-3.4 — EAGLE-3 hidden capture correctness ───────────────────────

    /// AC-G4-3.4 — forward_tree_verify_gpu captures hidden states at the
    /// requested layer and writes non-zero values into the collector buffer.
    /// Tests the layer-capture branch (capture_index_for → write_layer_slab).
    #[test]
    fn g4_cfa3_eagle3_hidden_capture_correctness_2026_05_23() {
        let device = match try_device() {
            Some(d) => d,
            None => { eprintln!("skip: no Metal device"); return; }
        };
        let mut gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(_) => { eprintln!("skip: no Metal device (gpu)"); return; }
        };

        // head_dim=256 required by Gemma4TreeVerifyLayerShape validator.
        let hidden = 256usize;
        let nq = 1usize;
        let nkv = 1usize;
        let head_dim = 256usize;
        let intermediate = 256usize;
        let vocab = 256usize;
        let n_layers = 3usize;
        let tree_seq = 2usize;
        let prefix = 2usize;
        let kv_cap = 16usize;

        let mut seed = 0xCAFE_u32;
        let model = mk_tiny_g4_model(
            n_layers, hidden, nq, nkv, head_dim, intermediate, vocab,
            &mut seed, &device,
        );

        let tokens: Vec<u32> = vec![3, 7];
        let mask = causal_mask_g4(tree_seq, prefix);
        let positions: Vec<u32> = vec![prefix as u32, prefix as u32 + 1];

        // Capture at layer 0 and layer 2 (not layer 1).
        let mut collector = Eagle3HiddenCollector::new(
            vec![0, 2], tree_seq, hidden,
        ).expect("collector");

        model.forward_tree_verify_gpu(
            &tokens, &mask, &positions, prefix, kv_cap,
            &mut gpu, &mut collector,
        ).expect("forward with capture");

        assert!(collector.is_complete(), "collector must be complete after full forward");

        // The concatenated buffer [tree_seq, num_aux=2, hidden] must be non-zero.
        let buf = collector.concatenated_hidden().expect("concatenated_hidden");
        let total = tree_seq * 2 * hidden;
        assert_eq!(buf.len(), total, "buffer shape mismatch");
        assert!(buf.iter().any(|&v| v != 0.0),
            "capture buffer is all zeros — layer hidden states not written");
    }

    // ── AC-G4-3.5 — input validation rejects bad arguments ───────────────────

    /// AC-G4-3.5 — forward_tree_verify_gpu returns Err for: (a) empty tokens,
    /// (b) wrong mask length, (c) wrong positions length, (d) collector/model
    /// hidden_size mismatch, (e) prefix+tree > kv_capacity.
    /// Existing forward_decode path is unaffected (additive-only).
    #[test]
    fn g4_cfa3_regression_existing_forward_decode_passes_2026_05_23() {
        let device = match try_device() {
            Some(d) => d,
            None => { eprintln!("skip: no Metal device"); return; }
        };
        let mut gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(_) => { eprintln!("skip: no Metal device (gpu)"); return; }
        };

        // head_dim=256 required by Gemma4TreeVerifyLayerShape validator.
        let hidden = 256usize;
        let nq = 1usize;
        let nkv = 1usize;
        let head_dim = 256usize;
        let intermediate = 256usize;
        let vocab = 256usize;
        let n_layers = 2usize;
        let tree_seq = 3usize;
        let prefix = 4usize;
        let kv_cap = 16usize;

        let mut seed = 0xDEAD_u32;
        let model = mk_tiny_g4_model(
            n_layers, hidden, nq, nkv, head_dim, intermediate, vocab,
            &mut seed, &device,
        );

        let mask = causal_mask_g4(tree_seq, prefix);
        let positions: Vec<u32> = (prefix as u32..prefix as u32 + tree_seq as u32).collect();
        let mut collector = Eagle3HiddenCollector::new(vec![0], tree_seq, hidden)
            .expect("collector");

        // (a) Empty tokens.
        let err = model.forward_tree_verify_gpu(
            &[], &mask, &positions, prefix, kv_cap, &mut gpu, &mut collector,
        );
        assert!(err.is_err(), "empty tokens must fail");
        assert!(err.unwrap_err().to_string().contains("tree_tokens must be non-empty"),
            "expected 'tree_tokens must be non-empty' in error");

        // (b) Wrong mask length.
        let tokens: Vec<u32> = vec![0, 1, 2];
        let bad_mask = vec![0.0f32; 5]; // wrong size
        let err = model.forward_tree_verify_gpu(
            &tokens, &bad_mask, &positions, prefix, kv_cap, &mut gpu, &mut collector,
        );
        assert!(err.is_err(), "wrong mask length must fail");

        // (c) Wrong positions length.
        let bad_positions = vec![0u32]; // too short
        let err = model.forward_tree_verify_gpu(
            &tokens, &mask, &bad_positions, prefix, kv_cap, &mut gpu, &mut collector,
        );
        assert!(err.is_err(), "wrong positions length must fail");

        // (d) collector hidden_size mismatch.
        let mut wrong_collector = Eagle3HiddenCollector::new(vec![0], tree_seq, hidden + 8)
            .expect("wrong collector");
        let err = model.forward_tree_verify_gpu(
            &tokens, &mask, &positions, prefix, kv_cap, &mut gpu, &mut wrong_collector,
        );
        assert!(err.is_err(), "hidden_size mismatch must fail");

        // (e) prefix + tree_seq > kv_capacity.
        let tight_cap = prefix + tree_seq - 1; // too small
        let err = model.forward_tree_verify_gpu(
            &tokens, &mask, &positions, prefix, tight_cap, &mut gpu, &mut collector,
        );
        assert!(err.is_err(), "prefix+tree > kv_capacity must fail");
    }

    // ── AC-G4-5c.4 — multi-iter KV continuity (cached vs fresh) ─────────────

    /// AC-G4-5c.4 — `forward_tree_verify_gpu_with_cache`: 3 iterations with a
    /// shared `&mut kv_caches_f32` all produce finite logits, AND iter-1 hidden
    /// states BYTE-DIFFER from iter-1 run with a fresh cache (proves real reuse).
    ///
    /// We compare hidden states captured via Eagle3HiddenCollector at layer 0
    /// rather than logits, because the tiny model's lm_head is zero-initialized
    /// (unit-test fixture) and would always produce all-zero logits regardless of
    /// KV state. The hidden state at layer 0's output is directly determined by
    /// the attention over [0, prefix+tree_seq) — positions [0, iter*tree_seq)
    /// from prior iterations are zero in the fresh cache and real in the cached
    /// path — so the counterfactual byte-diff is load-bearing here.
    #[test]
    fn g4_cfa5c_multi_iter_kv_continuity_2026_05_23() {
        let device = match try_device() {
            Some(d) => d,
            None => { eprintln!("skip: no Metal device"); return; }
        };
        let mut gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(_) => { eprintln!("skip: no Metal device (gpu)"); return; }
        };

        let hidden = 256usize;
        let nq = 1usize;
        let nkv = 1usize;
        let head_dim = 256usize;
        let intermediate = 256usize;
        let vocab = 256usize;
        let n_layers = 2usize;
        let tree_seq = 2usize;
        let kv_cap = 32usize;

        let mut seed = 0x5C5C_u32;
        let model = mk_tiny_g4_model(
            n_layers, hidden, nq, nkv, head_dim, intermediate, vocab,
            &mut seed, &device,
        );

        // Allocate a single persistent cache shared across all iterations.
        let mut kv_caches = model.alloc_tree_verify_kv_caches(&device, kv_cap)
            .expect("alloc_tree_verify_kv_caches");

        let tokens: Vec<u32> = vec![1, 2];
        let mut iter1_cached_hidden: Option<Vec<f32>> = None;

        for iter in 0..3usize {
            let prefix = iter * tree_seq;
            let mask = causal_mask_g4(tree_seq, prefix);
            let positions: Vec<u32> = (prefix as u32..prefix as u32 + tree_seq as u32).collect();

            let mut collector = Eagle3HiddenCollector::new(
                vec![0], tree_seq, hidden,
            ).expect("collector");

            let logits = model.forward_tree_verify_gpu_with_cache(
                &tokens, &mask, &positions, prefix, kv_cap,
                &mut gpu, &mut kv_caches, &mut collector,
            ).unwrap_or_else(|e| panic!("cached iter {iter} failed: {e}"));

            assert_eq!(logits.len(), tree_seq * vocab, "iter {iter}: logits len");
            assert!(logits.iter().all(|v| v.is_finite()),
                "iter {iter}: cached logits contain non-finite values");

            if iter == 1 {
                let h = collector.concatenated_hidden().expect("cached iter-1 hidden");
                iter1_cached_hidden = Some(h.to_vec());
            }
        }

        // Counterfactual: run iter-1 again with a FRESH cache (back-compat shim).
        // The fresh shim allocates zero-init caches, so iter-0 K/V is absent.
        // The cached path had iter-0 K/V in cache → attention output over
        // positions [0, prefix) differs → layer-0 hidden MUST byte-differ.
        {
            let prefix = 1 * tree_seq; // iter=1
            let mask = causal_mask_g4(tree_seq, prefix);
            let positions: Vec<u32> = (prefix as u32..prefix as u32 + tree_seq as u32).collect();
            let mut collector_fresh = Eagle3HiddenCollector::new(
                vec![0], tree_seq, hidden,
            ).expect("collector fresh");
            model.forward_tree_verify_gpu(
                &tokens, &mask, &positions, prefix, kv_cap,
                &mut gpu, &mut collector_fresh,
            ).expect("fresh iter-1");

            let fresh_hidden = collector_fresh.concatenated_hidden().expect("fresh hidden");
            let cached = iter1_cached_hidden.expect("cached iter-1 hidden must be set");
            let byte_equal = cached.iter().zip(fresh_hidden.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(!byte_equal,
                "iter-1 layer-0 hidden states must BYTE-DIFFER between cached path \
                 (has iter-0 K/V at positions [0, prefix)) and fresh path \
                 (zero-init cache at those positions) — persistent KV cache is not \
                 being reused across iterations");
        }
    }

    // ── AC-G4-5c.2 — back-compat byte-identity ──────────────────────────────

    /// AC-G4-5c.2 — `forward_tree_verify_gpu` (old sig) and
    /// `forward_tree_verify_gpu_with_cache` (fresh cache via
    /// `alloc_tree_verify_kv_caches`) produce BIT-EQUAL logits on identical
    /// inputs at a single iteration. Locks in the shim's byte-identity guarantee.
    #[test]
    fn g4_cfa5c_old_signature_back_compat_2026_05_23() {
        let device = match try_device() {
            Some(d) => d,
            None => { eprintln!("skip: no Metal device"); return; }
        };
        let mut gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(_) => { eprintln!("skip: no Metal device (gpu)"); return; }
        };

        let hidden = 256usize;
        let nq = 1usize;
        let nkv = 1usize;
        let head_dim = 256usize;
        let intermediate = 256usize;
        let vocab = 256usize;
        let n_layers = 2usize;
        let tree_seq = 3usize;
        let prefix = 4usize;
        let kv_cap = 16usize;

        let mut seed = 0xBAC0_u32;
        let model = mk_tiny_g4_model(
            n_layers, hidden, nq, nkv, head_dim, intermediate, vocab,
            &mut seed, &device,
        );

        let tokens: Vec<u32> = vec![0, 1, 2];
        let mask = causal_mask_g4(tree_seq, prefix);
        let positions: Vec<u32> = (prefix as u32..prefix as u32 + tree_seq as u32).collect();

        // Old-signature call.
        let logits_old = {
            let mut collector = Eagle3HiddenCollector::new(
                vec![0], tree_seq, hidden,
            ).expect("collector old");
            model.forward_tree_verify_gpu(
                &tokens, &mask, &positions, prefix, kv_cap,
                &mut gpu, &mut collector,
            ).expect("forward old sig")
        };

        // New-entry-point call with a freshly-allocated cache.
        let logits_new = {
            let mut kv_caches = model.alloc_tree_verify_kv_caches(&device, kv_cap)
                .expect("alloc caches");
            let mut collector = Eagle3HiddenCollector::new(
                vec![0], tree_seq, hidden,
            ).expect("collector new");
            model.forward_tree_verify_gpu_with_cache(
                &tokens, &mask, &positions, prefix, kv_cap,
                &mut gpu, &mut kv_caches, &mut collector,
            ).expect("forward with cache")
        };

        assert_eq!(logits_old.len(), logits_new.len(), "logit vec length mismatch");
        let bit_equal = logits_old.iter().zip(logits_new.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(bit_equal,
            "old forward_tree_verify_gpu and forward_tree_verify_gpu_with_cache (fresh cache) \
             must be bit-equal on the same single-iter inputs");
    }
}
