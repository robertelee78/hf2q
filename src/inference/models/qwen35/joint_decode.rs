//! Layer-major joint decode over independent Qwen3.5/3.6/3.8 slots.
//!
//! One transaction advances `B` slots of the shared [`HybridKvCache`] by one
//! token each, every slot at its own position. Row-parallel work runs once
//! over the packed rows: embedding gather, RMS norms, the DeltaNet and
//! full-attention projections (per [`JointProductRoute`]), the fused
//! residual norm, the dense FFN (per [`JointDenseFfnRoute`]) and the output
//! head. Work that touches a slot's cache (DeltaNet conv + recurrence, KV
//! write + SDPA) runs per row against that slot's cursor and ping-pong
//! parity; see
//! [`super::super::joint_decode_core`]. MoE FFNs with native router and
//! shared-expert storage (`HF2Q_QWEN_MOE_BATCHED=1`) run one batched layer
//! over the packed rows ([`super::super::moe_batched`]), whose kernels are
//! row-exact, so serial decode runs the same route at one row. Otherwise
//! they run per row through the serial helper: the BF16 router and shared
//! expert are float projections whose packed route differs from `m = 1`.
//!
//! Admission ([`super::super::joint_decode_routes`]) runs before the first
//! write. Cursors and parities advance at encode time exactly as in the
//! serial graph, so a failed step leaves every row of the step needing the
//! caller's per-slot recovery; rows outside the step are never touched.
//!
//! This module is a child of `forward_gpu` so it shares the thread-local GPU
//! weight cache without widening that cache's visibility.

use anyhow::{anyhow, bail, ensure, Context, Result};
use mlx_native::ops::argmax::dispatch_argmax_f32;
use mlx_native::ops::copy::dispatch_copy_f32;
use mlx_native::ops::fused_norm_add::dispatch_fused_residual_norm_f32;
use mlx_native::ops::rms_norm;
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::{
    default_chain_n, dump_layer_n, embed_token_ids_gpu_into, embed_tokens_gpu_into,
    gpu_cache_matches_model, FfnWeightsGpu, ForwardGpuCache, LayerWeightsGpu, OutputHeadGpu,
    GPU_CACHE,
};
use crate::inference::models::qwen35::decode_pool;
use crate::inference::models::qwen35::dump_bisect;
use crate::inference::models::qwen35::ffn::MoeFfnShape;
use crate::inference::models::qwen35::gpu_ffn::{
    build_dense_ffn_layer_gpu_q_into_with_gate_up, build_moe_ffn_layer_gpu_q_into,
    DenseGateUpDispatch,
};
use crate::inference::models::qwen35::gpu_full_attn::apply_linear_projection_f32_into_with_ggml_type;
use crate::inference::models::qwen35::joint_decode_core::{
    encode_joint_delta_net_layer, encode_joint_full_attn_layer, row_view, DeltaNetReplay,
    DeltaNetRoutes, FullAttnRoutes, JointRows, JointWindow,
};
use crate::inference::models::qwen35::joint_decode_routes::{
    check_dense_ffn_route, check_mode_policy, dense_ffn_route, joint_decode_env_unavailable_reason,
    joint_mode_max_rows, joint_padded_width, projection_route, select_joint_route_mode,
    JointDenseFfnRoute, JointProductRoute, JointRouteMode, JointRoutingPolicy,
    DFLASH2_VERIFY_WIDTH, MAX_JOINT_DECODE_ROWS,
};
use crate::inference::models::qwen35::joint_projection::{
    JointProjectionKernel, JointProjectionScope,
};
use crate::inference::models::qwen35::joint_step_options::JointStepOptions;
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::model::{Qwen35FfnWeights, Qwen35Model};
use crate::inference::models::qwen35::moe_batched::{
    build_moe_ffn_layer_gpu_batched_into, DFlash2MoeRouteMask,
};
use crate::inference::models::qwen35::spec_decode::nth_hidden_row;
use crate::inference::models::qwen35::Qwen35Config;
use crate::inference::spec_decode::dflash2::adapters::DFlash2TargetCapture;
use crate::serve::multi_seq_kv::SlotId;

/// What the caller needs back for one joint row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen35JointOutput {
    /// GPU argmax over the row's logits (the serial greedy head's kernel).
    GreedyToken,
    /// Full logits for host sampling.
    Logits,
    /// Full logits plus the post-output-norm hidden row MTP consumes.
    LogitsAndHidden,
    /// GPU argmax plus the post-output-norm hidden row (a greedy MTP lane).
    GreedyTokenAndHidden,
}

impl Qwen35JointOutput {
    fn wants_argmax(self) -> bool {
        matches!(self, Self::GreedyToken | Self::GreedyTokenAndHidden)
    }

    fn wants_logits(self) -> bool {
        matches!(self, Self::Logits | Self::LogitsAndHidden)
    }

    fn wants_hidden(self) -> bool {
        matches!(self, Self::LogitsAndHidden | Self::GreedyTokenAndHidden)
    }
}

/// One slot's input to a joint step.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Qwen35JointRow {
    pub slot_id: SlotId,
    pub token: u32,
    /// Text-axis position; all four MRoPE axes use it, as in serial decode.
    pub position: i32,
    pub output: Qwen35JointOutput,
}

/// One slot's result of a joint step.
pub(crate) struct Qwen35JointRowOutput {
    pub greedy_token: Option<u32>,
    pub logits: Option<Vec<f32>>,
    /// `[1, hidden]` view of a durable buffer; outlives the decode pool.
    pub nextn_hidden: Option<MlxBuffer>,
}

enum LayerRoutes {
    Full(FullAttnRoutes),
    Linear(DeltaNetRoutes),
}

/// Admitted product routes of every layer and the output head.
pub(super) struct JointPlan {
    layers: Vec<LayerRoutes>,
    head: JointProductRoute,
}

/// The packed rows of one joint forward.
pub(super) struct JointPacking<'a> {
    /// Real rows in packed order.
    pub rows: &'a [Qwen35JointRow],
    /// Lane windows over `rows` (single rows for a decode step).
    pub windows: &'a [JointWindow],
    /// Packed width, at least `rows.len()`; padding rows copy row 0.
    pub width: usize,
}

fn plan_joint_routes(
    mode: JointRouteMode,
    cfg: &Qwen35Config,
    layer_weights: &[LayerWeightsGpu],
    output_head: &OutputHeadGpu,
) -> Result<JointPlan, String> {
    let mut layers = Vec::with_capacity(layer_weights.len());
    for (layer, weights) in layer_weights.iter().enumerate() {
        let (routes, ffn) = match weights {
            LayerWeightsGpu::FullAttn { attn, ffn } => {
                (LayerRoutes::Full(FullAttnRoutes::plan(layer, attn)?), ffn)
            }
            LayerWeightsGpu::LinearAttn { attn, ffn } => {
                (LayerRoutes::Linear(DeltaNetRoutes::plan(layer, attn)?), ffn)
            }
        };
        match ffn {
            FfnWeightsGpu::DenseQ(weights) => {
                dense_ffn_route(layer, weights.ggml_type_gate_up, weights.ggml_type_down)?
            }
            FfnWeightsGpu::MoeQ(_) if mode == JointRouteMode::Exact && cfg.moe.is_some() => {}
            FfnWeightsGpu::MoeQ(_) => {
                return Err(format!(
                    "layer {layer} MoE FFN is admitted only in exact mode with a MoE config"
                ))
            }
            FfnWeightsGpu::Dense(_) | FfnWeightsGpu::Moe(_) => {
                return Err(format!(
                    "layer {layer} uses expanded float FFN storage, which has no joint graph"
                ))
            }
        }
        layers.push(routes);
    }
    let head = projection_route(
        "output",
        output_head.lm_head.dtype(),
        output_head.lm_head_ggml_type,
    )?;
    Ok(JointPlan { layers, head })
}

/// Same chain grouping knobs the serial greedy graph honours.
fn joint_chain_n(cfg: &Qwen35Config, layer_weights: &[LayerWeightsGpu]) -> usize {
    if std::env::var_os("HF2Q_PARTIAL_CHAIN_LEGACY").is_some_and(|value| value == "1") {
        return 1;
    }
    std::env::var("HF2Q_PARTIAL_CHAIN_N")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or_else(|| default_chain_n(cfg, layer_weights))
}

/// Whether the joint layer chain closes (commits) after `layer_idx`.
fn closes_chain(layer_idx: usize, chain_n: usize, n_layers: usize) -> bool {
    (layer_idx + 1) % chain_n == 0 || layer_idx + 1 == n_layers
}

/// The observed native commands of one joint forward, in commit order: the
/// embedding gather (an observed command only on the native embedding route),
/// one command per closed layer chain, an open-chain flush (never, while every
/// chain closes at `n_layers`) and the output head.
fn joint_command_plan(n_layers: usize, chain_n: usize, native_embedding: bool) -> Vec<String> {
    let mut commands = Vec::new();
    if native_embedding {
        commands.push("embed".to_string());
    }
    let mut start = 0;
    for layer in 0..n_layers {
        if closes_chain(layer, chain_n, n_layers) {
            commands.push(format!("layers[{start},{})", layer + 1));
            start = layer + 1;
        }
    }
    if start < n_layers {
        commands.push(format!("layers.flush[{start},{n_layers})"));
    }
    commands.push("output_head".into());
    commands
}

fn host_buffer<T: bytemuck::Pod>(
    device: &MlxDevice,
    dtype: DType,
    values: &[T],
    label: &str,
) -> Result<MlxBuffer> {
    let mut buffer = device
        .alloc_buffer(std::mem::size_of_val(values), dtype, vec![values.len()])
        .map_err(|e| anyhow!("allocate Qwen joint {label}: {e}"))?;
    buffer
        .as_mut_slice::<T>()
        .map_err(|e| anyhow!("map Qwen joint {label}: {e}"))?
        .copy_from_slice(values);
    Ok(buffer)
}

impl Qwen35Model {
    /// Decode route of this model's MoE FFN layers, from the loaded weights:
    /// `batched_native` (every MoE layer has native router/shared storage,
    /// `HF2Q_QWEN_MOE_BATCHED=1`), `per_row_bf16`, `mixed`, or `none` for a
    /// dense artifact.
    pub(crate) fn moe_ffn_decode_route(&self) -> &'static str {
        let (mut moe, mut native) = (0_usize, 0_usize);
        for layer in &self.layers {
            match layer.ffn() {
                Qwen35FfnWeights::MoeQ(weights) => {
                    moe += 1;
                    native += usize::from(weights.native_dense.is_some());
                }
                Qwen35FfnWeights::Moe(_) => moe += 1,
                Qwen35FfnWeights::Dense(_) | Qwen35FfnWeights::DenseQ(_) => {}
            }
        }
        match (moe, native) {
            (0, _) => "none",
            (moe, native) if native == moe => "batched_native",
            (_, 0) => "per_row_bf16",
            _ => "mixed",
        }
    }

    /// Joint route mode for this artifact and its routing policy, or why decode must
    /// stay serial. Primes the GPU weight cache so storage types are known.
    pub(crate) fn joint_decode_mode(&self) -> Result<JointRouteMode, String> {
        if let Some(reason) = joint_decode_env_unavailable_reason() {
            return Err(reason);
        }
        if dump_bisect::is_enabled() || dump_layer_n().is_some() {
            return Err("layer-dump diagnostics require the serial graph".into());
        }
        if self.glp.is_some() {
            return Err("GLP steering requires the serial graph".into());
        }
        let policy = JointRoutingPolicy::from_ggml(&self.ggml_routing_policy);
        let mode = select_joint_route_mode(policy, self.cfg.moe.is_some())?;
        check_mode_policy(mode, policy)?;
        self.ensure_gpu_cache_primed()
            .map_err(|error| format!("GPU cache priming failed: {error:#}"))?;
        GPU_CACHE.with(|cell| -> Result<JointRouteMode, String> {
            let cache = cell.borrow();
            let cache = cache
                .as_ref()
                .ok_or_else(|| "GPU cache is not primed".to_string())?;
            if !gpu_cache_matches_model(cache, self) {
                return Err("GPU cache belongs to a different Qwen model".to_string());
            }
            plan_joint_routes(mode, &self.cfg, &cache.layer_weights, &cache.output_head)
                .map(|_| mode)
        })
    }

    /// [`Self::forward_gpu_decode_joint_with_options`] with the default
    /// step options (every switch off).
    #[cfg(test)]
    pub(crate) fn forward_gpu_decode_joint(
        &self,
        rows: &[Qwen35JointRow],
        kv_cache: &mut HybridKvCache,
        mode: JointRouteMode,
        dense_ffn: JointDenseFfnRoute,
    ) -> Result<Vec<Qwen35JointRowOutput>> {
        let options = JointStepOptions::default();
        self.forward_gpu_decode_joint_with_options(rows, kv_cache, mode, dense_ffn, options)
    }

    /// Advance `rows.len()` distinct slots by one token each in one
    /// layer-major transaction. Under [`JointRouteMode::Exact`] every row's
    /// logits, hidden, KV codes, recurrent state, cursor and parity are
    /// bitwise those of its serial one-token forward. `dense_ffn` must be
    /// admitted for `mode` ([`check_dense_ffn_route`]). `options` are
    /// resolved by the caller (the serving worker reads their switches once
    /// at start; the Q1 trace runs several in one process); the
    /// column-invariant MMA kernel is admitted only in the padded mode.
    /// Refusal (admission, shape, slot range) happens before any cache write.
    pub(crate) fn forward_gpu_decode_joint_with_options(
        &self,
        rows: &[Qwen35JointRow],
        kv_cache: &mut HybridKvCache,
        mode: JointRouteMode,
        dense_ffn: JointDenseFfnRoute,
        options: JointStepOptions,
    ) -> Result<Vec<Qwen35JointRowOutput>> {
        let max_rows = joint_mode_max_rows(mode, MAX_JOINT_DECODE_ROWS);
        ensure!(
            (1..=max_rows).contains(&rows.len()),
            "Qwen {} joint decode requires 1..={max_rows} rows, got {}",
            mode.label(),
            rows.len()
        );
        for (index, row) in rows.iter().enumerate() {
            ensure!(
                row.slot_id.0 < kv_cache.n_seqs,
                "Qwen joint slot {} outside n_seqs={}",
                row.slot_id.0,
                kv_cache.n_seqs
            );
            ensure!(
                rows[..index]
                    .iter()
                    .all(|earlier| earlier.slot_id != row.slot_id),
                "Qwen joint decode slot {} appears twice",
                row.slot_id.0
            );
            ensure!(
                row.position >= 0,
                "Qwen joint position must be non-negative"
            );
        }
        self.check_joint_step_admission(kv_cache, mode, dense_ffn, options)?;
        let slots = rows.iter().map(|row| row.slot_id).collect::<Vec<_>>();
        let windows = JointWindow::per_row(&slots);
        let packing = JointPacking {
            rows,
            windows: &windows,
            width: joint_padded_width(mode, rows.len()),
        };
        self.with_joint_forward(
            mode,
            |device, registry, layer_weights, output_head, plan| {
                self.encode_joint_decode(
                    device,
                    registry,
                    layer_weights,
                    output_head,
                    plan,
                    &packing,
                    kv_cache,
                    dense_ffn,
                    options,
                    None,
                    None,
                    None,
                )
                .map(|(outputs, _normed)| outputs)
            },
        )
    }

    /// Admission shared by every joint forward, before any cache write: no
    /// DeltaNet capture, no serial-only diagnostics or steering, the mode
    /// and dense FFN route admitted by the model's routing policy, and the
    /// MMA projections only in the padded mode.
    pub(super) fn check_joint_step_admission(
        &self,
        kv_cache: &HybridKvCache,
        mode: JointRouteMode,
        dense_ffn: JointDenseFfnRoute,
        options: JointStepOptions,
    ) -> Result<()> {
        ensure!(
            !kv_cache.la_capture_active(),
            "Qwen joint decode cannot run while a DeltaNet capture is active"
        );
        if let Some(reason) = joint_decode_env_unavailable_reason() {
            bail!("Qwen joint decode unavailable: {reason}");
        }
        ensure!(
            !dump_bisect::is_enabled() && dump_layer_n().is_none(),
            "Qwen joint decode unavailable: layer-dump diagnostics require the serial graph"
        );
        ensure!(
            self.glp.is_none(),
            "Qwen joint decode unavailable: GLP steering requires the serial graph"
        );
        check_mode_policy(
            mode,
            JointRoutingPolicy::from_ggml(&self.ggml_routing_policy),
        )
        .map_err(|reason| anyhow!("Qwen joint decode unavailable: {reason}"))?;
        check_dense_ffn_route(mode, dense_ffn)
            .map_err(|reason| anyhow!("Qwen joint decode unavailable: {reason}"))?;
        ensure!(
            options.projection == JointProjectionKernel::Canonical
                || mode == JointRouteMode::PaddedMvExt,
            "Qwen joint decode unavailable: the column-invariant MMA projections are \
             admitted only in the padded mode"
        );
        Ok(())
    }

    /// Run `encode` inside one joint forward: the primed GPU cache of this
    /// model, its admitted route plan and a freshly reset decode pool.
    pub(super) fn with_joint_forward<R>(
        &self,
        mode: JointRouteMode,
        encode: impl FnOnce(
            &MlxDevice,
            &mut KernelRegistry,
            &[LayerWeightsGpu],
            &OutputHeadGpu,
            &JointPlan,
        ) -> Result<R>,
    ) -> Result<R> {
        self.ensure_gpu_cache_primed()?;
        let pool_operation = decode_pool::profile::Operation::enter(self as *const _ as usize)?;
        let result = GPU_CACHE.with(|cell| -> Result<R> {
            let mut guard = cell.borrow_mut();
            let cache = guard
                .as_mut()
                .context("Qwen joint decode: GPU cache not primed")?;
            ensure!(
                gpu_cache_matches_model(cache, self),
                "Qwen joint decode: GPU cache belongs to a different Qwen model"
            );
            let ForwardGpuCache {
                device,
                registry,
                layer_weights,
                output_head,
                ..
            } = cache;
            let plan = plan_joint_routes(mode, &self.cfg, layer_weights, output_head)
                .map_err(|reason| anyhow!("Qwen joint decode unavailable: {reason}"))?;
            decode_pool::checked_reset_decode_pool()
                .map_err(|e| anyhow!("Qwen joint decode pool reset: {e}"))?;
            encode(device, registry, layer_weights, output_head, &plan)
        });
        pool_operation.complete()?;
        result
    }

    /// Encode and run one joint forward over `packing`. `replay` (MTP
    /// verify only) retains the multi-row windows' DeltaNet inputs. Returns
    /// the real rows' outputs and the packed `[width, hidden]`
    /// post-output-norm rows (a durable allocation).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_joint_decode(
        &self,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        layer_weights: &[LayerWeightsGpu],
        output_head: &OutputHeadGpu,
        plan: &JointPlan,
        packing: &JointPacking<'_>,
        kv_cache: &mut HybridKvCache,
        dense_ffn: JointDenseFfnRoute,
        options: JointStepOptions,
        gpu_token_ids: Option<&MlxBuffer>,
        mut replay: Option<&mut DeltaNetReplay>,
        capture: Option<DFlash2TargetCapture<'_>>,
    ) -> Result<(Vec<Qwen35JointRowOutput>, MlxBuffer)> {
        let arena_setup = crate::inference::stage_profile::CpuSpan::begin("per_row_arena_setup");
        let cfg = &self.cfg;
        let hidden_size = cfg.hidden_size;
        let h = hidden_size as usize;
        let eps = cfg.rms_norm_eps;
        let vocab = cfg.vocab_size as usize;
        let rows = packing.rows;
        let width = packing.width;
        ensure!(
            (1..=width).contains(&rows.len()) && width <= DFLASH2_VERIFY_WIDTH,
            "Qwen joint forward packs {} rows at width {width}",
            rows.len()
        );
        if let Some(capture) = capture {
            ensure!(
                capture.lane_indices.len() == packing.windows.len()
                    && capture
                        .lane_indices
                        .iter()
                        .enumerate()
                        .all(|(index, lane)| {
                            *lane < capture.lanes && !capture.lane_indices[..index].contains(lane)
                        })
                    && capture.rows_per_lane == 8
                    && !capture.layer_ids.is_empty()
                    && capture.feature_width == h * capture.layer_ids.len()
                    && capture.output.dtype() == DType::F32
                    && capture.output.element_count()
                        == capture.lanes * capture.rows_per_lane * capture.feature_width
                    && capture.layer_ids.windows(2).all(|pair| pair[0] < pair[1])
                    && capture
                        .layer_ids
                        .iter()
                        .all(|layer| *layer < layer_weights.len())
                    && packing
                        .windows
                        .iter()
                        .all(|window| window.rows <= capture.rows_per_lane),
                "invalid Qwen DFlash2 target capture geometry"
            );
        }
        let width_u32 = width as u32;
        let slots = rows.iter().map(|row| row.slot_id).collect::<Vec<_>>();
        let joint_rows = JointRows {
            slots: &slots,
            windows: packing.windows,
            width,
        };
        // Padding rows replay row 0's token and position; their outputs are
        // discarded and they never reach a cache.
        let packed = |index: usize| rows[if index < rows.len() { index } else { 0 }];
        let tokens = (0..width)
            .map(|index| packed(index).token)
            .collect::<Vec<_>>();
        let mut positions = vec![0_i32; 4 * width];
        for axis in 0..4 {
            for index in 0..width {
                positions[axis * width + index] = packed(index).position;
            }
        }
        let positions = host_buffer(device, DType::I32, positions.as_slice(), "positions")?;
        // The decode forward's actual input manifest (trace on only): one
        // row per packed slot in matrix order, padding rows owner-less. Its
        // drop at return closes the forward before the caller's per-lane
        // settle work. A verify window has no joint trace role yet.
        let _native_input = replay.is_none().then(|| {
            crate::inference::native_observation::joint_decode_input(
                rows.iter().map(|row| (row.slot_id.0, row.position)),
                width,
            )
        });
        let mut embedded = device
            .alloc_buffer(width * h * 4, DType::F32, vec![width, h])
            .map_err(|e| anyhow!("allocate Qwen joint embeddings: {e}"))?;
        if let Some(token_ids) = gpu_token_ids {
            ensure!(
                token_ids.dtype() == DType::U32 && token_ids.element_count() >= width,
                "Qwen GPU target token tile has {:?} with {} rows, expected at least {width} U32 rows",
                token_ids.dtype(),
                token_ids.element_count()
            );
            let narrowed;
            let token_ids = if token_ids.element_count() == width {
                token_ids
            } else {
                narrowed = token_ids.slice_view(0, width);
                &narrowed
            };
            embed_token_ids_gpu_into(
                token_ids,
                self.token_embd_native.as_ref(),
                hidden_size,
                device,
                registry,
                &mut embedded,
            )
            .context("Qwen joint GPU-token embedding gather")?;
        } else {
            embed_tokens_gpu_into(
                &tokens,
                &self.token_embd,
                self.token_embd_native.as_ref(),
                cfg.vocab_size,
                hidden_size,
                device,
                registry,
                &mut embedded,
                #[cfg(test)]
                None,
            )
            .context("Qwen joint embedding gather")?;
        }

        drop(arena_setup);
        let chain_n = joint_chain_n(cfg, layer_weights);
        let n_layers = layer_weights.len();
        ensure!(
            plan.layers.len() == n_layers,
            "Qwen joint plan covers {} of {n_layers} layers",
            plan.layers.len()
        );
        crate::inference::native_observation::joint_chain_plan(|| {
            let kinds: Vec<_> = layer_weights
                .iter()
                .map(|weights| match weights {
                    LayerWeightsGpu::FullAttn { .. } => "full",
                    LayerWeightsGpu::LinearAttn { .. } => "linear",
                })
                .collect();
            let native_embedding = self.token_embd_native.is_some();
            serde_json::json!({"model":"qwen35","n_layers":n_layers,"chain_n":chain_n,
                "native_embedding":native_embedding,"layer_kinds":kinds,
                "commands":joint_command_plan(n_layers, chain_n, native_embedding)})
        });
        let mut hidden = embedded;
        let mut retained = Vec::with_capacity(n_layers);
        let mut chain: Option<CommandEncoder> = None;
        // Layer chains only; the output head stays canonical and allocates
        // its outputs as before.
        let projection_scope = JointProjectionScope::enter(options.projection, width_u32);
        let pooled_scope = decode_pool::StepPooledOutputs::enter(options.pooled_outputs);
        for (layer_idx, (weights, routes)) in layer_weights.iter().zip(&plan.layers).enumerate() {
            if let Some(enc) = chain.as_mut() {
                // Cross-layer RAW: this layer reads the previous FFN output.
                enc.memory_barrier();
            } else {
                chain = Some(
                    device
                        .command_encoder()
                        .map(crate::inference::native_observation::encoder)
                        .with_context(|| format!("Qwen joint encoder layer {layer_idx}"))?,
                );
            }
            let enc = chain.as_mut().expect("joint layer encoder opened above");
            let (attention, post_norm, ffn) = match (weights, routes) {
                (LayerWeightsGpu::FullAttn { attn, ffn }, LayerRoutes::Full(routes)) => (
                    encode_joint_full_attn_layer(
                        enc,
                        device,
                        registry,
                        &hidden,
                        &positions,
                        attn,
                        routes,
                        kv_cache,
                        layer_idx,
                        &joint_rows,
                        cfg,
                    )
                    .with_context(|| format!("Qwen joint full attention layer {layer_idx}"))?,
                    &attn.post_attn_norm,
                    ffn,
                ),
                (LayerWeightsGpu::LinearAttn { attn, ffn }, LayerRoutes::Linear(routes)) => (
                    encode_joint_delta_net_layer(
                        enc,
                        device,
                        registry,
                        &hidden,
                        attn,
                        routes,
                        kv_cache,
                        layer_idx,
                        &joint_rows,
                        cfg,
                        replay.as_deref_mut(),
                    )
                    .with_context(|| format!("Qwen joint DeltaNet layer {layer_idx}"))?,
                    &attn.post_attn_norm,
                    ffn,
                ),
                _ => bail!("Qwen joint plan disagrees with layer {layer_idx} kind"),
            };
            enc.memory_barrier();
            let ffn_input =
                decode_pool::pooled_alloc_buffer(device, width * h * 4, DType::F32, vec![width, h])
                    .map_err(|e| anyhow!("allocate Qwen joint FFN input: {e}"))?;
            let ffn_residual =
                decode_pool::pooled_alloc_buffer(device, width * h * 4, DType::F32, vec![width, h])
                    .map_err(|e| anyhow!("allocate Qwen joint FFN residual: {e}"))?;
            dispatch_fused_residual_norm_f32(
                enc,
                registry,
                device.metal_device(),
                &hidden,
                &attention,
                post_norm,
                &ffn_input,
                Some(&ffn_residual),
                width_u32,
                hidden_size,
                eps,
            )
            .with_context(|| format!("Qwen joint fused residual norm layer {layer_idx}"))?;
            enc.memory_barrier();
            let output = match ffn {
                FfnWeightsGpu::DenseQ(weights) => build_dense_ffn_layer_gpu_q_into_with_gate_up(
                    enc,
                    device,
                    registry,
                    &ffn_input,
                    weights,
                    Some(&ffn_residual),
                    match dense_ffn {
                        JointDenseFfnRoute::SerialBuilder => DenseGateUpDispatch::PreferFused,
                        JointDenseFfnRoute::SharedWeights => DenseGateUpDispatch::Separate,
                    },
                )
                .with_context(|| format!("Qwen joint dense FFN layer {layer_idx}"))?,
                FfnWeightsGpu::MoeQ(weights) => {
                    let moe = cfg
                        .moe
                        .as_ref()
                        .with_context(|| format!("Qwen joint MoE config (layer {layer_idx})"))?;
                    let shape = MoeFfnShape {
                        hidden_size,
                        num_experts: moe.num_experts,
                        num_experts_per_tok: moe.num_experts_per_tok,
                        moe_intermediate_size: moe.moe_intermediate_size,
                        shared_intermediate_size: moe.shared_expert_intermediate_size,
                    };
                    if weights.dense_storage.is_native() {
                        let mut output = decode_pool::pooled_alloc_buffer(
                            device,
                            width * h * 4,
                            DType::F32,
                            vec![width, h],
                        )
                        .map_err(|e| anyhow!("allocate Qwen joint MoE output: {e}"))?;
                        let moe_rows = if capture.is_some() {
                            rows.len() as u32
                        } else {
                            width_u32
                        };
                        let route_mask = if gpu_token_ids.is_some() {
                            let capture = capture.context(
                                "Qwen GPU-target MoE verification requires DFlash2 capture",
                            )?;
                            let mut physical_lane = [0_u32; 4];
                            for (logical, &physical) in capture.lane_indices.iter().enumerate() {
                                physical_lane[logical] = u32::try_from(physical)
                                    .context("Qwen DFlash2 physical lane overflow")?;
                            }
                            Some(DFlash2MoeRouteMask {
                                verify_rows: capture.verify_rows,
                                params: mlx_native::ops::dflash2_attention_pack::DFlash2MoeRouteMaskParams {
                                    physical_lane,
                                    lanes: capture.lane_indices.len() as u32,
                                    rows_per_lane: capture.rows_per_lane as u32,
                                    top_k: shape.num_experts_per_tok,
                                    n_experts: shape.num_experts,
                                    width: moe_rows,
                                },
                            })
                        } else {
                            None
                        };
                        build_moe_ffn_layer_gpu_batched_into(
                            enc,
                            device,
                            registry,
                            &ffn_input,
                            weights,
                            shape,
                            Some(&ffn_residual),
                            &mut output,
                            moe_rows,
                            layer_idx,
                            route_mask,
                        )
                        .with_context(|| format!("Qwen joint batched MoE FFN layer {layer_idx}"))?;
                        output
                    } else {
                        let mut per_row = Vec::with_capacity(rows.len());
                        for row in 0..rows.len() {
                            if row > 0 {
                                enc.memory_barrier();
                            }
                            per_row.push(
                                build_moe_ffn_layer_gpu_q_into(
                                    enc,
                                    device,
                                    registry,
                                    &row_view(&ffn_input, row, vec![1, h])?,
                                    weights,
                                    shape,
                                    Some(&row_view(&ffn_residual, row, vec![1, h])?),
                                    layer_idx,
                                )
                                .with_context(|| {
                                    format!("Qwen joint MoE FFN layer {layer_idx} row {row}")
                                })?,
                            );
                        }
                        enc.memory_barrier();
                        let packed_output = device
                            .alloc_buffer(width * h * 4, DType::F32, vec![width, h])
                            .map_err(|e| anyhow!("allocate Qwen joint MoE output: {e}"))?;
                        for index in 0..width {
                            let source = &per_row[if index < rows.len() { index } else { 0 }];
                            dispatch_copy_f32(
                                enc,
                                registry,
                                device.metal_device(),
                                source,
                                &packed_output,
                                0,
                                index * h,
                                h,
                            )
                            .map_err(|e| anyhow!("gather Qwen joint MoE row {index}: {e}"))?;
                        }
                        packed_output
                    }
                }
                FfnWeightsGpu::Dense(_) | FfnWeightsGpu::Moe(_) => {
                    bail!("Qwen joint layer {layer_idx} has no admitted FFN arm")
                }
            };
            if let Some(capture) = capture {
                if let Some(capture_index) = capture
                    .layer_ids
                    .iter()
                    .position(|captured| *captured == layer_idx)
                {
                    enc.memory_barrier();
                    for (&lane, window) in capture.lane_indices.iter().zip(packing.windows) {
                        let source =
                            output.slice_view((window.start * h * 4) as u64, window.rows * h);
                        let destination = capture.output.slice_view(
                            (lane * capture.rows_per_lane * capture.feature_width * 4) as u64,
                            window.rows * capture.feature_width,
                        );
                        mlx_native::ops::feature_concat::dispatch_feature_concat_f32(
                            enc,
                            registry,
                            device.metal_device(),
                            &source,
                            &destination,
                            window.rows as u32,
                            hidden_size,
                            (capture_index * h) as u32,
                            capture.feature_width as u32,
                        )
                        .with_context(|| {
                            format!("capture Qwen DFlash2 layer {layer_idx} lane {lane}")
                        })?;
                    }
                }
            }
            crate::inference::stage_profile::boundary(enc, "adr065.ffn_moe")
                .with_context(|| format!("profile Qwen joint FFN layer {layer_idx}"))?;
            if closes_chain(layer_idx, chain_n, n_layers) {
                chain
                    .take()
                    .expect("joint layer encoder opened above")
                    .commit_labeled("qwen35.joint_decode.layers");
            }
            retained.push(std::mem::replace(&mut hidden, output));
        }
        if let Some(mut enc) = chain.take() {
            enc.commit_labeled("qwen35.joint_decode.layers.flush");
        }
        drop(pooled_scope);
        drop(projection_scope);

        // ---- Output head: final norm, lm_head, per-row argmax ----
        let mut enc = device
            .command_encoder()
            .map(crate::inference::native_observation::encoder)
            .context("Qwen joint output-head encoder")?;
        let normed = device
            .alloc_buffer(width * h * 4, DType::F32, vec![width, h])
            .map_err(|e| anyhow!("allocate Qwen joint output norm: {e}"))?;
        let norm_params = host_buffer(
            device,
            DType::F32,
            &[eps, hidden_size as f32],
            "norm params",
        )?;
        rms_norm::dispatch_rms_norm(
            &mut enc,
            registry,
            device.metal_device(),
            &hidden,
            &output_head.norm_w,
            &normed,
            &norm_params,
            width_u32,
            hidden_size,
        )
        .context("Qwen joint output norm")?;
        enc.memory_barrier();
        // Pooled: the head writes every row and every result leaves this
        // function as an owned copy before the next forward resets the pool,
        // so a step pays no fresh `[width, vocab]` allocation and CPU zeroing.
        let mut logits = decode_pool::pooled_alloc_buffer(
            device,
            width * vocab * 4,
            DType::F32,
            vec![width, vocab],
        )
        .map_err(|e| anyhow!("allocate Qwen joint logits: {e}"))?;
        match plan.head {
            JointProductRoute::Batched => apply_linear_projection_f32_into_with_ggml_type(
                &mut enc,
                registry,
                device,
                &normed,
                &output_head.lm_head,
                output_head.lm_head_ggml_type,
                &mut logits,
                width_u32,
                hidden_size,
                cfg.vocab_size,
            )
            .context("Qwen joint lm_head")?,
            JointProductRoute::PerRow => {
                for row in 0..rows.len() {
                    let mut destination = row_view(&logits, row, vec![1, vocab])?;
                    apply_linear_projection_f32_into_with_ggml_type(
                        &mut enc,
                        registry,
                        device,
                        &row_view(&normed, row, vec![1, h])?,
                        &output_head.lm_head,
                        output_head.lm_head_ggml_type,
                        &mut destination,
                        1,
                        hidden_size,
                        cfg.vocab_size,
                    )
                    .with_context(|| format!("Qwen joint lm_head row {row}"))?;
                }
            }
        }
        enc.memory_barrier();
        let argmax_params = host_buffer(device, DType::U32, &[cfg.vocab_size], "argmax params")?;
        let mut argmax = Vec::with_capacity(rows.len());
        for (row, input) in rows.iter().enumerate() {
            if !input.output.wants_argmax() {
                argmax.push(None);
                continue;
            }
            let index = decode_pool::pooled_alloc_buffer(device, 4, DType::U32, vec![1])
                .map_err(|e| anyhow!("allocate Qwen joint argmax index: {e}"))?;
            let value = decode_pool::pooled_alloc_buffer(device, 4, DType::F32, vec![1])
                .map_err(|e| anyhow!("allocate Qwen joint argmax value: {e}"))?;
            dispatch_argmax_f32(
                &mut enc,
                registry,
                device.metal_device(),
                &row_view(&logits, row, vec![1, vocab])?,
                &index,
                &value,
                &argmax_params,
                cfg.vocab_size,
            )
            .with_context(|| format!("Qwen joint argmax row {row}"))?;
            argmax.push(Some((index, value)));
        }
        let head_label = if crate::inference::stage_profile::enabled() {
            "adr065.head"
        } else {
            "qwen35.joint_decode.output_head"
        };
        enc.commit_and_wait_labeled(head_label)
            .map_err(|e| anyhow!("Qwen joint decode command buffer: {e}"))?;
        drop(retained);

        let readback = crate::inference::stage_profile::CpuSpan::begin("logit_readback");
        let all_logits = logits
            .as_slice::<f32>()
            .map_err(|e| anyhow!("read Qwen joint logits: {e}"))?;
        let outputs = rows
            .iter()
            .enumerate()
            .zip(argmax)
            .map(|((row, input), argmax)| -> Result<Qwen35JointRowOutput> {
                let greedy_token = argmax
                    .map(|(index, _value)| {
                        index
                            .as_slice::<u32>()
                            .map(|slice| slice[0])
                            .map_err(|e| anyhow!("read Qwen joint argmax row {row}: {e}"))
                    })
                    .transpose()?;
                let logits = input
                    .output
                    .wants_logits()
                    .then(|| all_logits[row * vocab..(row + 1) * vocab].to_vec());
                let nextn_hidden = input
                    .output
                    .wants_hidden()
                    .then(|| nth_hidden_row(&normed, hidden_size, row as u64))
                    .transpose()?;
                Ok(Qwen35JointRowOutput {
                    greedy_token,
                    logits,
                    nextn_hidden,
                })
            })
            .collect::<Result<Vec<_>>>();
        drop(readback);
        outputs.map(|outputs| (outputs, normed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::gpu_delta_net::apply_proj_with_ggml_type;
    use crate::inference::models::qwen35::gpu_ffn::{
        build_dense_ffn_layer_gpu_q_into, DenseFfnWeightsGpuQ,
    };
    use crate::inference::models::qwen35::gpu_full_attn::{
        apply_linear_projection_f32_pooled_with_ggml_type, upload_f32,
    };
    use crate::inference::models::qwen35::joint_decode_core::project_rows;
    use crate::inference::models::qwen35::joint_decode_test_support::{
        bits, random_f32, synthetic_blocks,
    };
    use mlx_native::ops::quantized_matmul_ggml::GgmlType;

    /// A registry frozen to the operator policy with `mul_mv_ext` off, so the
    /// exact-policy proofs below run whatever the process environment says.
    fn exact_policy_registry() -> KernelRegistry {
        let mut policy = mlx_native::ggml_routing_policy_from_environment();
        policy.dense_decode_mv_ext = false;
        let mut registry = KernelRegistry::new();
        registry
            .freeze_ggml_routing_policy(policy)
            .expect("freeze exact routing policy");
        registry
    }

    /// Admitted block storage: one packed dispatch at `m = 2..=4` must equal
    /// the serial `m = 1` dispatch of every row, bit for bit, on the default
    /// (exact) routing policy. A failing type must leave
    /// `joint_decode_routes::packed_block_type`.
    #[test]
    fn packed_block_projections_match_per_row_dispatch_bitwise() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(_) => return,
        };
        let mut registry = exact_policy_registry();
        let (k, n) = (512_u32, 256_u32);
        let mut seed = 0x51A7_0001_u32;
        for ggml_type in [
            GgmlType::Q4_0,
            GgmlType::Q8_0,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
        ] {
            let weight = synthetic_blocks(&device, ggml_type, n, k, &mut seed);
            for m in 2..=4_usize {
                let input = upload_f32(&random_f32(&mut seed, m * k as usize), &device)
                    .expect("upload input");
                let mut enc = device.command_encoder().expect("encoder");
                let run = |enc: &mut CommandEncoder,
                           registry: &mut KernelRegistry,
                           route: JointProductRoute| {
                    project_rows(
                        enc,
                        registry,
                        &device,
                        route,
                        &input,
                        m,
                        k as usize,
                        |enc, registry, input, seq| {
                            apply_proj_with_ggml_type(
                                enc, registry, &device, input, &weight, ggml_type, seq, k, n,
                            )
                            .map(|output| vec![output])
                        },
                    )
                    .expect("joint projection")
                    .remove(0)
                };
                let packed = run(&mut enc, &mut registry, JointProductRoute::Batched);
                let per_row = run(&mut enc, &mut registry, JointProductRoute::PerRow);
                enc.commit_and_wait_labeled("test.qwen.joint.projection")
                    .expect("complete joint projection");
                assert_eq!(
                    bits(&packed, m * n as usize),
                    bits(&per_row, m * n as usize),
                    "{ggml_type:?} packed m={m} differs from per-row m=1"
                );
            }
        }
    }

    /// The packed dense FFN (fused gate/up/SiLU, down, residual fold) must
    /// equal one serial `m = 1` FFN per row for every admitted type pair.
    #[test]
    fn packed_dense_ffn_matches_per_row_ffn_bitwise() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(_) => return,
        };
        let mut registry = exact_policy_registry();
        let (hidden, intermediate) = (256_u32, 512_u32);
        let h = hidden as usize;
        let mut seed = 0x51A7_0002_u32;
        for (gate_up, down) in [
            (GgmlType::Q4_K, GgmlType::Q6_K),
            (GgmlType::Q5_K, GgmlType::Q6_K),
            (GgmlType::Q8_0, GgmlType::Q8_0),
            (GgmlType::Q6_K, GgmlType::Q4_K),
        ] {
            let weights = DenseFfnWeightsGpuQ {
                gate_q: synthetic_blocks(&device, gate_up, intermediate, hidden, &mut seed),
                up_q: synthetic_blocks(&device, gate_up, intermediate, hidden, &mut seed),
                down_q: synthetic_blocks(&device, down, hidden, intermediate, &mut seed),
                ggml_type_gate_up: gate_up,
                ggml_type_down: down,
                intermediate_size: intermediate,
                hidden_size: hidden,
            };
            for m in 2..=4_usize {
                let x = upload_f32(&random_f32(&mut seed, m * h), &device)
                    .expect("upload x")
                    .with_shape(vec![m, h])
                    .expect("shape x");
                let residual = upload_f32(&random_f32(&mut seed, m * h), &device)
                    .expect("upload residual")
                    .with_shape(vec![m, h])
                    .expect("shape residual");
                let mut enc = device.command_encoder().expect("encoder");
                let packed = build_dense_ffn_layer_gpu_q_into(
                    &mut enc,
                    &device,
                    &mut registry,
                    &x,
                    &weights,
                    Some(&residual),
                )
                .expect("packed dense FFN");
                let mut per_row = Vec::with_capacity(m);
                for row in 0..m {
                    enc.memory_barrier();
                    per_row.push(
                        build_dense_ffn_layer_gpu_q_into(
                            &mut enc,
                            &device,
                            &mut registry,
                            &row_view(&x, row, vec![1, h]).expect("x row"),
                            &weights,
                            Some(&row_view(&residual, row, vec![1, h]).expect("residual row")),
                        )
                        .expect("per-row dense FFN"),
                    );
                }
                enc.commit_and_wait_labeled("test.qwen.joint.dense_ffn")
                    .expect("complete dense FFN");
                let packed = bits(&packed, m * h);
                for (row, output) in per_row.iter().enumerate() {
                    assert_eq!(
                        &packed[row * h..(row + 1) * h],
                        bits(output, h).as_slice(),
                        "{gate_up:?}/{down:?} packed m={m} row {row} differs from m=1"
                    );
                }
            }
        }
    }

    /// Packed width-4 products inside an MMA [`JointProjectionScope`]: each
    /// row is bitwise the MMA kernel's own result, equals the same row in a
    /// padded 1-real-row step (row copies in the other slots) and the
    /// width-1 dispatch, and repeats bitwise. Qwen3.8 K values included.
    #[test]
    fn mma_scope_rows_are_cohort_invariant_and_deterministic() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let width = 4_usize;
        let mut seed = 0x51A7_0003_u32;
        for (ggml_type, n, k) in [
            (GgmlType::Q4_K, 256_u32, 5120_u32),
            (GgmlType::Q6_K, 256, 5120),
            (GgmlType::Q4_K, 128, 17408),
            (GgmlType::Q6_K, 128, 17408),
            (GgmlType::Q4_K, 48, 6144),
        ] {
            let (nn, kk) = (n as usize, k as usize);
            let weight = synthetic_blocks(&device, ggml_type, n, k, &mut seed);
            let rows = random_f32(&mut seed, width * kk);
            let packed_input = upload_f32(&rows, &device).expect("upload rows");
            // Padded step for row r: row r first, copies of it after.
            let padded_inputs: Vec<MlxBuffer> = (0..width)
                .map(|r| {
                    let row = &rows[r * kk..(r + 1) * kk];
                    upload_f32(&row.repeat(width), &device).expect("upload padded")
                })
                .collect();
            let mut enc = device.command_encoder().expect("encoder");
            let project = |enc: &mut CommandEncoder,
                           registry: &mut KernelRegistry,
                           input: &MlxBuffer,
                           seq: u32| {
                apply_proj_with_ggml_type(
                    enc, registry, &device, input, &weight, ggml_type, seq, k, n,
                )
                .expect("scoped projection")
            };
            let (packed, repeat, padded, singles) = {
                let _scope =
                    JointProjectionScope::enter(JointProjectionKernel::ColumnInvariantMma, 4);
                let packed = project(&mut enc, &mut registry, &packed_input, 4);
                let repeat = project(&mut enc, &mut registry, &packed_input, 4);
                let padded: Vec<MlxBuffer> = padded_inputs
                    .iter()
                    .map(|input| project(&mut enc, &mut registry, input, 4))
                    .collect();
                let _single_scope =
                    JointProjectionScope::enter(JointProjectionKernel::ColumnInvariantMma, 1);
                let singles: Vec<MlxBuffer> = (0..width)
                    .map(|r| {
                        let input = row_view(&packed_input, r, vec![1, kk]).expect("row view");
                        project(&mut enc, &mut registry, &input, 1)
                    })
                    .collect();
                (packed, repeat, padded, singles)
            };
            let direct = device
                .alloc_buffer(width * nn * 4, DType::F32, vec![width, nn])
                .expect("direct output");
            mlx_native::quantized_matmul_ggml_column_invariant_mma(
                &mut enc,
                &mut registry,
                &device,
                &packed_input,
                &weight,
                &direct,
                &mlx_native::ops::quantized_matmul_ggml::GgmlQuantizedMatmulParams {
                    m: 4,
                    n,
                    k,
                    ggml_type,
                },
            )
            .expect("direct MMA");
            enc.commit_and_wait_labeled("test.qwen.joint.mma_scope")
                .expect("complete scoped projections");
            let label = format!("{ggml_type:?} {n}x{k}");
            let packed = bits(&packed, width * nn);
            assert_eq!(
                packed,
                bits(&direct, width * nn),
                "{label}: scope did not route to MMA"
            );
            assert_eq!(packed, bits(&repeat, width * nn), "{label}: repeat differs");
            for r in 0..width {
                let row = &packed[r * nn..(r + 1) * nn];
                assert_eq!(
                    row,
                    &bits(&padded[r], nn)[..],
                    "{label}: row {r} differs when padded"
                );
                assert_eq!(
                    row,
                    &bits(&singles[r], nn)[..],
                    "{label}: row {r} differs from m=1"
                );
            }
        }
    }

    /// The shared-weights dense FFN inside an MMA scope: every packed row is
    /// bitwise the 1-row FFN of that row (joint ≡ per-slot by construction).
    #[test]
    fn mma_scope_shared_dense_ffn_matches_per_row_bitwise() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let (hidden, intermediate) = (512_u32, 1024_u32);
        let h = hidden as usize;
        let m = 4_usize;
        let mut seed = 0x51A7_0004_u32;
        for down in [GgmlType::Q4_K, GgmlType::Q6_K] {
            let weights = DenseFfnWeightsGpuQ {
                gate_q: synthetic_blocks(&device, GgmlType::Q4_K, intermediate, hidden, &mut seed),
                up_q: synthetic_blocks(&device, GgmlType::Q4_K, intermediate, hidden, &mut seed),
                down_q: synthetic_blocks(&device, down, hidden, intermediate, &mut seed),
                ggml_type_gate_up: GgmlType::Q4_K,
                ggml_type_down: down,
                intermediate_size: intermediate,
                hidden_size: hidden,
            };
            let x = upload_f32(&random_f32(&mut seed, m * h), &device)
                .expect("upload x")
                .with_shape(vec![m, h])
                .expect("shape x");
            let residual = upload_f32(&random_f32(&mut seed, m * h), &device)
                .expect("upload residual")
                .with_shape(vec![m, h])
                .expect("shape residual");
            let mut enc = device.command_encoder().expect("encoder");
            let packed = {
                let _scope =
                    JointProjectionScope::enter(JointProjectionKernel::ColumnInvariantMma, 4);
                build_dense_ffn_layer_gpu_q_into_with_gate_up(
                    &mut enc,
                    &device,
                    &mut registry,
                    &x,
                    &weights,
                    Some(&residual),
                    DenseGateUpDispatch::Separate,
                )
                .expect("packed dense FFN")
            };
            let per_row: Vec<MlxBuffer> = {
                let _scope =
                    JointProjectionScope::enter(JointProjectionKernel::ColumnInvariantMma, 1);
                (0..m)
                    .map(|row| {
                        enc.memory_barrier();
                        build_dense_ffn_layer_gpu_q_into_with_gate_up(
                            &mut enc,
                            &device,
                            &mut registry,
                            &row_view(&x, row, vec![1, h]).expect("x row"),
                            &weights,
                            Some(&row_view(&residual, row, vec![1, h]).expect("residual row")),
                            DenseGateUpDispatch::Separate,
                        )
                        .expect("per-row dense FFN")
                    })
                    .collect()
            };
            enc.commit_and_wait_labeled("test.qwen.joint.mma_dense_ffn")
                .expect("complete dense FFN");
            let packed = bits(&packed, m * h);
            for (row, output) in per_row.iter().enumerate() {
                assert_eq!(
                    &packed[row * h..(row + 1) * h],
                    bits(output, h).as_slice(),
                    "Q4_K/{down:?} MMA packed row {row} differs from m=1"
                );
            }
        }
    }

    /// Pooled step outputs (Class A): the packed shared-weights dense FFN
    /// and an attention-style projection give the same bytes with and
    /// without [`decode_pool::StepPooledOutputs`].
    #[test]
    fn step_pooled_outputs_keep_packed_ffn_and_projection_bitwise() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let (hidden, intermediate) = (512_u32, 1024_u32);
        let h = hidden as usize;
        let m = 4_usize;
        let mut seed = 0x51A7_0005_u32;
        let weights = DenseFfnWeightsGpuQ {
            gate_q: synthetic_blocks(&device, GgmlType::Q4_K, intermediate, hidden, &mut seed),
            up_q: synthetic_blocks(&device, GgmlType::Q4_K, intermediate, hidden, &mut seed),
            down_q: synthetic_blocks(&device, GgmlType::Q6_K, hidden, intermediate, &mut seed),
            ggml_type_gate_up: GgmlType::Q4_K,
            ggml_type_down: GgmlType::Q6_K,
            intermediate_size: intermediate,
            hidden_size: hidden,
        };
        let projection = synthetic_blocks(&device, GgmlType::Q4_K, 256, hidden, &mut seed);
        let x = upload_f32(&random_f32(&mut seed, m * h), &device)
            .expect("upload x")
            .with_shape(vec![m, h])
            .expect("shape x");
        let residual = upload_f32(&random_f32(&mut seed, m * h), &device)
            .expect("upload residual")
            .with_shape(vec![m, h])
            .expect("shape residual");
        decode_pool::checked_reset_decode_pool().expect("reset pool");
        let mut enc = device.command_encoder().expect("encoder");
        let mut run = |enc: &mut CommandEncoder, pooled: bool| {
            let _scope = decode_pool::StepPooledOutputs::enter(pooled);
            let ffn = build_dense_ffn_layer_gpu_q_into_with_gate_up(
                enc,
                &device,
                &mut registry,
                &x,
                &weights,
                Some(&residual),
                DenseGateUpDispatch::Separate,
            )
            .expect("dense FFN");
            let proj = apply_linear_projection_f32_pooled_with_ggml_type(
                enc,
                &mut registry,
                &device,
                &x,
                &projection,
                GgmlType::Q4_K,
                m as u32,
                hidden,
                256,
            )
            .expect("projection");
            (ffn, proj)
        };
        let (fresh_ffn, fresh_proj) = run(&mut enc, false);
        let (pooled_ffn, pooled_proj) = run(&mut enc, true);
        enc.commit_and_wait_labeled("test.qwen.joint.pooled_outputs")
            .expect("complete pooled outputs");
        assert_eq!(bits(&fresh_ffn, m * h), bits(&pooled_ffn, m * h));
        assert_eq!(bits(&fresh_proj, m * 256), bits(&pooled_proj, m * 256));
        decode_pool::checked_reset_decode_pool().expect("reset pool");
    }

    #[test]
    fn joint_command_plan_follows_the_commit_points() {
        let plan = joint_command_plan(64, 4, true);
        assert_eq!(plan.len(), 18);
        assert_eq!(plan[0], "embed");
        assert_eq!(plan[1], "layers[0,4)");
        assert_eq!(plan[16], "layers[60,64)");
        assert_eq!(plan[17], "output_head");
        // A remainder group closes at the last layer; no open-chain flush.
        assert_eq!(
            joint_command_plan(10, 4, true),
            [
                "embed",
                "layers[0,4)",
                "layers[4,8)",
                "layers[8,10)",
                "output_head"
            ]
        );
        assert_eq!(joint_command_plan(3, 1, true).len(), 5);
        // Host embedding (no native table): no observed embed command.
        assert_eq!(
            joint_command_plan(8, 4, false),
            ["layers[0,4)", "layers[4,8)", "output_head"]
        );
        let closes = (0..10).filter(|&layer| closes_chain(layer, 4, 10)).count();
        assert_eq!(closes, 3);
    }
}
