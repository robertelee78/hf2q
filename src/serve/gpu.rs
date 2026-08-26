//! mlx-native integration layer for hf2q inference.
//!
//! Provides [`GpuContext`] — a thin wrapper that holds the [`GraphExecutor`]
//! and [`KernelRegistry`] for the mlx-native backend.
//!
//! # ADR-008: candle divorce
//!
//! All candle bridge functions have been removed.  Weights are loaded
//! directly from GGUF via `mlx_native::gguf::GgufFile` into `MlxBuffer`s.
//! The `QuantWeightInfo` struct now uses `mlx_native::GgmlType` directly.

use mlx_native::{
    DenseBf16CalibrationBatchReceipt, DenseMatmulIdCalibrationBatchReceipt, GraphExecutor,
    KernelRegistry, MlxDevice,
};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::inference::dense_bf16_activation::{
    activate_native_bf16_dense, DenseBf16Activation, NativeBf16Matrix,
};
use crate::inference::dense_expert_activation::{
    activate_native_scalar_experts, DenseExpertActivation, NativeScalarExpertMatrix,
};

/// GPU context for the mlx-native backend.
///
/// Owns the graph executor (which in turn owns the Metal device and command
/// queue) and the pre-warmed kernel registry.  Created once at model load;
/// lives for the duration of inference.
pub struct GpuContext {
    activation_epoch: u64,
    ggml_routing_policy: mlx_native::GgmlRoutingPolicy,
    dense_bf16_activated: bool,
    dense_bf16_receipt: Option<DenseBf16CalibrationBatchReceipt>,
    dense_expert_activated: bool,
    dense_expert_receipt: Option<DenseMatmulIdCalibrationBatchReceipt>,
    /// Batched dispatch executor — one `CommandEncoder` per forward pass.
    /// Also owns the `MlxDevice`.
    pub executor: GraphExecutor,
    /// Pre-compiled shader pipeline cache.
    pub registry: KernelRegistry,
    /// Secondary pre-warmed kernel registry for the parallel-encode worker
    /// thread (ADR-031 Phase B, Option A).  `Some` only when
    /// `HF2Q_PARALLEL_ENCODE=1` was set at process start; `None` otherwise,
    /// keeping the default path zero-cost.
    ///
    /// Life-cycle: `take_worker_registry` moves it out for one
    /// `forward_decode` call; `encode_parallel_layers_chunked` returns it
    /// via mpsc; `put_worker_registry` stores it back so the next token
    /// finds it here again.
    pub worker_registry: Option<KernelRegistry>,
    /// Epoch captured when the worker registry leaves this context. A return
    /// from another model lifetime is rejected and the foreign registry is
    /// dropped instead of becoming route authority for this model.
    worker_registry_checked_out_epoch: Option<u64>,
}

// SAFETY: The metal::DeviceRef is Send+Sync (MTLDevice is thread-safe).
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
    /// Initialize the mlx-native GPU context.
    ///
    /// Creates the Metal device, graph executor, and an empty kernel registry.
    /// Kernel pipelines are compiled lazily on first use (typically during the
    /// warmup forward passes).
    ///
    /// When `HF2Q_PARALLEL_ENCODE=1` is set at process start, also allocates
    /// and registers an identical secondary `KernelRegistry` for the
    /// parallel-encode worker thread.  One-time ~5 ms startup cost; paid only
    /// on opt-in.
    ///
    /// # Errors
    ///
    /// Returns an error if no Metal device is available.
    pub fn new() -> mlx_native::Result<Self> {
        static NEXT_ACTIVATION_EPOCH: AtomicU64 = AtomicU64::new(1);
        let activation_epoch = NEXT_ACTIVATION_EPOCH.fetch_add(1, Ordering::Relaxed);
        assert_ne!(activation_epoch, 0, "GPU activation epoch exhausted");
        let device = MlxDevice::new()?;
        let gpu_name = device.name();
        let executor = GraphExecutor::new(device);
        let mut registry = KernelRegistry::new();
        let ggml_routing_policy = mlx_native::ggml_routing_policy_from_environment();
        registry.freeze_ggml_routing_policy(ggml_routing_policy)?;
        tracing::info!(
            dense_decode_mvn = ggml_routing_policy.dense_decode_mvn,
            dense_decode_mv_ext = ggml_routing_policy.dense_decode_mv_ext,
            dense_q5k_canonical_q4x4 = ggml_routing_policy.dense_q5k_canonical_q4x4,
            "frozen dense GGML routing policy"
        );
        // Register all inference kernels.
        mlx_native::ops::hadamard_quantize_kv::register(&mut registry);
        mlx_native::ops::flash_attn_vec_tq::register(&mut registry);
        // F16 SDPA reduce kernels — reused by TQ SDPA with NWG>1.
        mlx_native::ops::flash_attn_vec::register(&mut registry);
        // Standalone FWHT for TQ SDPA pre/post rotation.
        let fwht_src = mlx_native::ops::fwht_standalone::FWHT_STANDALONE_SHADER_SOURCE;
        registry.register_source("fwht_standalone_f32_d256", fwht_src);
        registry.register_source("fwht_standalone_f32_d512", fwht_src);
        // ADR-011 Phase 2 Wave 4 (flash_attn_prefill wire-up):
        //   Flash-attention tiled prefill kernels replace sdpa/sdpa_sliding for
        //   batched prefill. Three registration calls cover (1) the D=256
        //   main kernel (bf16 Q/K/V/O, BQ=32, BK=16), (2) the D=512 NSG=8
        //   peer-derived main kernel (bf16, NQPSG=8, NCPSG=64), (3) the
        //   SWA / causal mask builder (Wave 2D, shape [qL, kL] broadcast
        //   across batch + heads), and (4) the tile-skip pre-pass classifier
        //   (Wave 2E, one byte per (qtile, ktile) from the mask). See
        //   docs/ADR-011-phase2-wave4-wire-up-verification.md.
        mlx_native::ops::flash_attn_prefill::register(&mut registry);
        mlx_native::ops::flash_attn_prefill_d512::register(&mut registry);
        mlx_native::ops::flash_attn_prefill_mask::register(&mut registry);
        mlx_native::ops::flash_attn_prefill_blk::register(&mut registry);
        mlx_native::ops::embedding_q2_k::register(&mut registry);
        mlx_native::ops::embedding_q8_0::register(&mut registry);
        mlx_native::ops::deepseek_hyper_connection::register(&mut registry);
        mlx_native::ops::deepseek_sparse_attention::register(&mut registry);
        mlx_native::ops::deepseek_sparse_prefill_mask::register(&mut registry);
        mlx_native::ops::deepseek_compressor::register(&mut registry);
        mlx_native::ops::deepseek_indexer::register(&mut registry);
        mlx_native::ops::deepseek_tail_rope::register(&mut registry);
        mlx_native::ops::deepseek_moe_routing::register(&mut registry);
        mlx_native::ops::deepseek_moe_activation::register(&mut registry);
        mlx_native::ops::repeat_tiled::register(&mut registry);
        // EAGLE3 speculative decode helpers are not part of
        // KernelRegistry::new(); register them once at model load so the
        // first draft step can compile and cache the pipelines.
        crate::inference::spec_decode::eagle3::forward::register_eagle3_forward_kernels(
            &mut registry,
        );

        // ADR-031 Phase B (Option A): allocate a second identical registry for
        // the parallel-encode worker thread, but ONLY when opt-in is set.
        // Using std::env::var directly here (not INVESTIGATION_ENV) because
        // LazyLock semantics allow either init order, and env::var is cheaper
        // and sufficient for this single binary decision at model load.
        let worker_registry = if std::env::var("HF2Q_PARALLEL_ENCODE").as_deref() == Ok("1") {
            let mut wreg = KernelRegistry::new();
            wreg.freeze_ggml_routing_policy(ggml_routing_policy)?;
            // Mirror the EXACT same registrations as the main registry above.
            // Chesterton's fence: if a new kernel family is added to the main
            // registry block, it MUST also be added here to keep the worker
            // registry warm for all decode-hot kernels.
            mlx_native::ops::hadamard_quantize_kv::register(&mut wreg);
            mlx_native::ops::flash_attn_vec_tq::register(&mut wreg);
            mlx_native::ops::flash_attn_vec::register(&mut wreg);
            wreg.register_source("fwht_standalone_f32_d256", fwht_src);
            wreg.register_source("fwht_standalone_f32_d512", fwht_src);
            mlx_native::ops::flash_attn_prefill::register(&mut wreg);
            mlx_native::ops::flash_attn_prefill_d512::register(&mut wreg);
            mlx_native::ops::flash_attn_prefill_mask::register(&mut wreg);
            mlx_native::ops::flash_attn_prefill_blk::register(&mut wreg);
            mlx_native::ops::embedding_q2_k::register(&mut wreg);
            mlx_native::ops::embedding_q8_0::register(&mut wreg);
            mlx_native::ops::deepseek_hyper_connection::register(&mut wreg);
            mlx_native::ops::deepseek_sparse_attention::register(&mut wreg);
            mlx_native::ops::deepseek_sparse_prefill_mask::register(&mut wreg);
            mlx_native::ops::deepseek_compressor::register(&mut wreg);
            mlx_native::ops::deepseek_indexer::register(&mut wreg);
            mlx_native::ops::deepseek_tail_rope::register(&mut wreg);
            mlx_native::ops::deepseek_moe_routing::register(&mut wreg);
            mlx_native::ops::deepseek_moe_activation::register(&mut wreg);
            mlx_native::ops::repeat_tiled::register(&mut wreg);
            crate::inference::spec_decode::eagle3::forward::register_eagle3_forward_kernels(
                &mut wreg,
            );
            tracing::info!(
                "mlx-native GpuContext: worker KernelRegistry pre-warmed (HF2Q_PARALLEL_ENCODE=1)"
            );
            Some(wreg)
        } else {
            None
        };

        tracing::info!("mlx-native GpuContext initialized on {}", gpu_name);
        Ok(Self {
            activation_epoch,
            ggml_routing_policy,
            dense_bf16_activated: false,
            dense_bf16_receipt: None,
            dense_expert_activated: false,
            dense_expert_receipt: None,
            executor,
            registry,
            worker_registry,
            worker_registry_checked_out_epoch: None,
        })
    }

    /// Borrow the underlying `MlxDevice`.
    #[inline]
    pub fn device(&self) -> &MlxDevice {
        self.executor.device()
    }

    /// Human-readable GPU name (e.g. "Apple M5 Max").
    pub fn gpu_name(&self) -> String {
        self.device().name()
    }

    pub(crate) fn activate_native_bf16_dense(
        &mut self,
        matrices: &[NativeBf16Matrix<'_>],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.worker_registry_checked_out_epoch.is_none(),
            "cannot activate native routes while the worker registry is checked out"
        );
        anyhow::ensure!(
            !self.dense_bf16_activated,
            "GPU native BF16 routes were already activated for epoch {}",
            self.activation_epoch
        );
        let activation = activate_native_bf16_dense(
            &mut self.registry,
            self.executor.device(),
            self.activation_epoch,
            matrices,
        )?;
        if let Some(DenseBf16Activation { plan, receipt }) = activation {
            if let Some(worker) = self.worker_registry.as_mut() {
                worker
                    .freeze_dense_bf16_plan(self.executor.device(), plan)
                    .map_err(|error| {
                        anyhow::anyhow!("freeze BF16 plan into worker registry: {error}")
                    })?;
            }
            self.dense_bf16_receipt = Some(receipt);
        }
        self.dense_bf16_activated = true;
        Ok(())
    }

    pub(crate) fn activate_native_scalar_experts(
        &mut self,
        matrices: &[NativeScalarExpertMatrix<'_>],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.worker_registry_checked_out_epoch.is_none(),
            "cannot activate native routes while the worker registry is checked out"
        );
        anyhow::ensure!(
            !self.dense_expert_activated,
            "GPU native scalar expert routes were already activated for epoch {}",
            self.activation_epoch
        );
        let activation = activate_native_scalar_experts(
            &mut self.registry,
            self.worker_registry.as_mut(),
            self.executor.device(),
            self.activation_epoch,
            matrices,
        )?;
        if let Some(DenseExpertActivation { plan: _, receipt }) = activation {
            self.dense_expert_receipt = Some(receipt);
        }
        self.dense_expert_activated = true;
        Ok(())
    }

    #[inline]
    pub(crate) fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn ggml_routing_policy(&self) -> &mlx_native::GgmlRoutingPolicy {
        &self.ggml_routing_policy
    }

    /// Split borrow: returns (&GraphExecutor, &mut KernelRegistry) to avoid
    /// conflicting borrows when methods need both the device (from executor)
    /// and mutable access to the registry.
    #[inline]
    pub fn split(&mut self) -> (&GraphExecutor, &mut KernelRegistry) {
        (&self.executor, &mut self.registry)
    }

    /// Move the worker registry out for use by `encode_parallel_layers_chunked`.
    ///
    /// Returns `None` if `HF2Q_PARALLEL_ENCODE=1` was not set at process start
    /// (i.e. the worker registry was never allocated) or if it has already been
    /// taken and not yet returned (panic-safe: caller gets `None` and can error
    /// cleanly via the `ok_or_else` pattern in B3).
    #[inline]
    pub fn take_worker_registry(&mut self) -> Option<KernelRegistry> {
        let registry = self.worker_registry.take()?;
        debug_assert!(self.worker_registry_checked_out_epoch.is_none());
        self.worker_registry_checked_out_epoch = Some(self.activation_epoch);
        Some(registry)
    }

    /// Return the worker registry after `encode_parallel_layers_chunked`
    /// completes.  Called unconditionally on every `PARALLEL=ON` forward_decode
    /// return path so the next token's parallel split finds the registry here.
    #[inline]
    pub fn put_worker_registry(&mut self, reg: KernelRegistry) -> anyhow::Result<()> {
        let checked_out_epoch = self
            .worker_registry_checked_out_epoch
            .take()
            .ok_or_else(|| anyhow::anyhow!("worker registry returned without an active lease"))?;
        anyhow::ensure!(
            checked_out_epoch == self.activation_epoch,
            "worker registry lease epoch {checked_out_epoch} != context epoch {}",
            self.activation_epoch
        );
        anyhow::ensure!(
            self.worker_registry.is_none(),
            "worker registry returned while another registry is already installed"
        );
        match (self.registry.dense_bf16_plan(), reg.dense_bf16_plan()) {
            (None, None) => {}
            (Some(primary), Some(worker)) => anyhow::ensure!(
                worker.activation_epoch() == self.activation_epoch
                    && worker.plan_id() == primary.plan_id()
                    && worker.decision_count() == primary.decision_count(),
                "worker BF16 plan does not match primary authority for context epoch {}",
                self.activation_epoch
            ),
            _ => anyhow::bail!("worker and primary BF16 plan presence differ"),
        }
        match (
            self.registry.dense_matmul_id_plan(),
            reg.dense_matmul_id_plan(),
        ) {
            (None, None) => {}
            (Some(primary), Some(worker)) => anyhow::ensure!(
                worker.activation_epoch() == self.activation_epoch
                    && worker.plan_id() == primary.plan_id()
                    && worker.activation_authority_digest()
                        == primary.activation_authority_digest()
                    && worker.decision_count() == primary.decision_count(),
                "worker scalar-expert plan does not match primary authority for context epoch {}",
                self.activation_epoch
            ),
            _ => anyhow::bail!("worker and primary scalar-expert plan presence differ"),
        }
        anyhow::ensure!(
            reg.ggml_routing_policy() == Some(&self.ggml_routing_policy),
            "worker registry routing policy does not match its model context"
        );
        anyhow::ensure!(
            reg.dense_q4_plan().is_none(),
            "decode worker registry must remain free of main-only Q4 route authority"
        );
        self.worker_registry = Some(reg);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Quantized weight metadata
// ---------------------------------------------------------------------------

/// Information about a quantized weight loaded from GGUF.
#[derive(Debug, Clone, Copy)]
pub struct QuantWeightInfo {
    /// GGML quantization type (Q4_0, Q6_K, Q8_0, etc.).
    pub ggml_dtype: mlx_native::GgmlType,
    /// Number of output rows (N dimension of the weight matrix).
    pub rows: usize,
    /// Number of input columns (K dimension of the weight matrix).
    pub cols: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_init() {
        let ctx = GpuContext::new().expect("GpuContext::new should succeed on Apple Silicon");
        assert!(!ctx.gpu_name().is_empty());
        assert!(
            ctx.worker_registry.is_none(),
            "worker_registry should be None when HF2Q_PARALLEL_ENCODE is unset"
        );
        println!("GpuContext GPU: {}", ctx.gpu_name());
    }

    #[test]
    fn test_worker_registry_round_trip() {
        let mut ctx = GpuContext::new().expect("GpuContext::new");
        let mut worker = KernelRegistry::new();
        worker
            .freeze_ggml_routing_policy(*ctx.ggml_routing_policy())
            .expect("bind worker policy");
        ctx.worker_registry = Some(worker);
        let reg = ctx
            .take_worker_registry()
            .expect("installed worker registry should lease");
        ctx.put_worker_registry(reg)
            .expect("same-epoch worker registry should return");
        assert!(ctx.worker_registry.is_some());
        let _reg = ctx.take_worker_registry();
        assert!(ctx.worker_registry.is_none());
    }

    #[test]
    fn worker_registry_return_without_lease_fails_closed() {
        let mut ctx = GpuContext::new().expect("GpuContext::new");
        let error = ctx
            .put_worker_registry(KernelRegistry::new())
            .expect_err("unleased registry must not become worker authority");
        assert!(error.to_string().contains("without an active lease"));
        assert!(ctx.worker_registry.is_none());
    }

    #[test]
    fn worker_registry_rejects_same_epoch_wrong_route_authority() {
        let mut ctx = GpuContext::new().expect("GpuContext::new");
        ctx.worker_registry = None;
        let target = ctx
            .device()
            .alloc_buffer(32 * 32 * 2, mlx_native::DType::BF16, vec![32, 32])
            .expect("target BF16 matrix");
        ctx.activate_native_bf16_dense(&[NativeBf16Matrix::unbatched_single_row(
            "primary", &target, 32, 32,
        )])
        .expect("activate primary plan");

        let foreign = ctx
            .device()
            .alloc_buffer(64 * 32 * 2, mlx_native::DType::BF16, vec![64, 32])
            .expect("foreign BF16 matrix");
        let mut wrong = KernelRegistry::new();
        wrong
            .freeze_ggml_routing_policy(*ctx.ggml_routing_policy())
            .expect("bind foreign registry policy");
        crate::inference::dense_bf16_activation::activate_native_bf16_dense(
            &mut wrong,
            ctx.device(),
            ctx.activation_epoch(),
            &[NativeBf16Matrix::unbatched_single_row(
                "wrong authority",
                &foreign,
                64,
                32,
            )],
        )
        .expect("activate wrong same-epoch plan");

        ctx.worker_registry = Some(wrong);
        let leased = ctx.take_worker_registry().expect("lease wrong registry");
        let error = ctx
            .put_worker_registry(leased)
            .expect_err("same epoch must not authorize a different plan");
        assert!(error
            .to_string()
            .contains("does not match primary authority"));
        assert!(ctx.worker_registry.is_none());
    }

    #[test]
    fn worker_registry_rejects_same_epoch_wrong_scalar_authority() {
        use mlx_native::{DenseMatmulIdInputLayout, DenseMatmulIdMultiplicity};

        let mut ctx = GpuContext::new().expect("GpuContext::new");
        ctx.worker_registry = None;
        let primary_weight = ctx
            .device()
            .alloc_buffer(4 * 32 * 32 * 2, mlx_native::DType::F16, vec![4, 32, 32])
            .expect("primary scalar experts");
        let primary = [NativeScalarExpertMatrix {
            label: "primary scalar authority",
            weight: &primary_weight,
            n: 32,
            k: 32,
            top_k: 2,
            n_experts: 4,
            expert_stride_bytes: 32 * 32 * 2,
            input_layout: DenseMatmulIdInputLayout::SharedPerToken,
            id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
            calibrated_m: vec![1],
        }];
        ctx.activate_native_scalar_experts(&primary)
            .expect("activate primary scalar plan");

        let foreign_weight = ctx
            .device()
            .alloc_buffer(4 * 64 * 32 * 2, mlx_native::DType::F16, vec![4, 64, 32])
            .expect("foreign scalar experts");
        let foreign = [NativeScalarExpertMatrix {
            label: "foreign scalar authority",
            weight: &foreign_weight,
            n: 64,
            k: 32,
            top_k: 2,
            n_experts: 4,
            expert_stride_bytes: 64 * 32 * 2,
            input_layout: DenseMatmulIdInputLayout::SharedPerToken,
            id_multiplicity: DenseMatmulIdMultiplicity::DistinctPerToken,
            calibrated_m: vec![1],
        }];
        let mut wrong = KernelRegistry::new();
        wrong
            .freeze_ggml_routing_policy(*ctx.ggml_routing_policy())
            .expect("bind foreign registry policy");
        crate::inference::dense_expert_activation::activate_native_scalar_experts(
            &mut wrong,
            None,
            ctx.device(),
            ctx.activation_epoch(),
            &foreign,
        )
        .expect("activate wrong same-epoch scalar plan");

        ctx.worker_registry = Some(wrong);
        let leased = ctx.take_worker_registry().expect("lease wrong registry");
        let error = ctx
            .put_worker_registry(leased)
            .expect_err("same epoch must not authorize a different scalar plan");
        assert!(error
            .to_string()
            .contains("scalar-expert plan does not match primary authority"));
        assert!(ctx.worker_registry.is_none());
    }

    #[test]
    fn native_bf16_activation_unions_exact_rows_and_freezes_once() {
        let mut ctx = GpuContext::new().expect("GpuContext::new");
        let mut target = ctx
            .device()
            .alloc_buffer(32 * 32 * 2, mlx_native::DType::BF16, vec![32, 32])
            .expect("target BF16 buffer");
        target
            .as_mut_slice::<u16>()
            .expect("target BF16 slice")
            .fill(0x3f80);
        let mut drafter = ctx
            .device()
            .alloc_buffer(64 * 32 * 2, mlx_native::DType::BF16, vec![64, 32])
            .expect("drafter BF16 buffer");
        drafter
            .as_mut_slice::<u16>()
            .expect("drafter BF16 slice")
            .fill(0x3f00);

        let matrices = [
            NativeBf16Matrix::unbatched_through("target", &target, 32, 32, 2).expect("target rows"),
            NativeBf16Matrix::unbatched_single_row("drafter", &drafter, 64, 32),
        ];
        ctx.activate_native_bf16_dense(&matrices)
            .expect("union activation");

        let receipt = ctx.dense_bf16_receipt.as_ref().expect("activation receipt");
        assert_eq!(receipt.activation_epoch, ctx.activation_epoch);
        assert_eq!(receipt.declared_shapes, 3);
        assert_eq!(receipt.decisions.len(), 3);
        assert_eq!(
            ctx.registry
                .dense_bf16_plan()
                .expect("frozen registry plan")
                .decision_count(),
            3
        );
        assert_eq!(
            receipt
                .decisions
                .iter()
                .map(|decision| (decision.shape.m, decision.shape.n, decision.shape.k))
                .collect::<std::collections::BTreeSet<_>>(),
            [(1, 32, 32), (1, 64, 32), (2, 32, 32)]
                .into_iter()
                .collect()
        );

        let error = ctx
            .activate_native_bf16_dense(&matrices)
            .expect_err("a live context must never widen or replace its frozen plan");
        assert!(
            error.to_string().contains("already activated"),
            "unexpected second-activation error: {error:#}"
        );
    }

    #[test]
    fn native_bf16_model_swap_a_b_a_reuses_only_process_timing_metadata() {
        fn activate_shape(n: usize) -> GpuContext {
            let mut ctx = GpuContext::new().expect("GpuContext::new");
            let mut weight = ctx
                .device()
                .alloc_buffer(n * 32 * 2, mlx_native::DType::BF16, vec![n, 32])
                .expect("BF16 swap fixture");
            weight
                .as_mut_slice::<u16>()
                .expect("BF16 swap slice")
                .fill(0x3f80);
            let matrix = [NativeBf16Matrix::unbatched_single_row(
                "swap fixture",
                &weight,
                u32::try_from(n).expect("fixture N"),
                32,
            )];
            ctx.activate_native_bf16_dense(&matrix)
                .expect("activate swap fixture");
            ctx
        }

        let a1 = activate_shape(32);
        let b = activate_shape(64);
        let a2 = activate_shape(32);
        let a1_receipt = a1.dense_bf16_receipt.as_ref().expect("A1 receipt");
        let b_receipt = b.dense_bf16_receipt.as_ref().expect("B receipt");
        let a2_receipt = a2.dense_bf16_receipt.as_ref().expect("A2 receipt");

        assert_ne!(a1.activation_epoch, b.activation_epoch);
        assert_ne!(b.activation_epoch, a2.activation_epoch);
        assert_ne!(a1_receipt.plan_id, b_receipt.plan_id);
        assert_ne!(b_receipt.plan_id, a2_receipt.plan_id);
        assert_ne!(a1_receipt.plan_id, a2_receipt.plan_id);
        assert_eq!(a2_receipt.declared_shapes, 1);
        assert_eq!(a2_receipt.process_cache_hits, 1);
        assert_eq!(a2_receipt.calibration_submissions, 0);
        assert!(a2_receipt.decisions[0].process_cache_hit);
    }
}
