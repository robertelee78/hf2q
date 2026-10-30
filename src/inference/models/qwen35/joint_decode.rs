//! Layer-major joint decode over independent Qwen3.5/3.6/3.8 slots.
//!
//! One transaction advances `B` slots of the shared [`HybridKvCache`] by one
//! token each, every slot at its own position. Row-parallel work runs once
//! over the packed rows: embedding gather, RMS norms, the DeltaNet and
//! full-attention projections (per [`JointProductRoute`]), the fused
//! residual norm, the dense FFN and the output head. Work that touches a
//! slot's cache (DeltaNet conv + recurrence, KV write + SDPA) runs per row
//! against that slot's cursor and ping-pong parity; see
//! [`super::super::joint_decode_core`]. MoE FFNs run per row through the
//! serial helper: the router and shared expert are float projections whose
//! packed route differs from `m = 1`, and routed experts are mostly distinct
//! per row, so packing them would trade exactness for little weight reuse.
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
    default_chain_n, dump_layer_n, embed_tokens_gpu_into, gpu_cache_matches_model, FfnWeightsGpu,
    ForwardGpuCache, LayerWeightsGpu, OutputHeadGpu, GPU_CACHE,
};
use crate::inference::models::qwen35::decode_pool;
use crate::inference::models::qwen35::dump_bisect;
use crate::inference::models::qwen35::ffn::MoeFfnShape;
use crate::inference::models::qwen35::gpu_ffn::{
    build_dense_ffn_layer_gpu_q_into, build_moe_ffn_layer_gpu_q_into,
};
use crate::inference::models::qwen35::gpu_full_attn::apply_linear_projection_f32_into_with_ggml_type;
use crate::inference::models::qwen35::joint_decode_core::{
    encode_joint_delta_net_layer, encode_joint_full_attn_layer, row_view, DeltaNetRoutes,
    FullAttnRoutes, JointRows,
};
use crate::inference::models::qwen35::joint_decode_routes::{
    check_mode_environment, dense_ffn_route, joint_decode_env_unavailable_reason,
    joint_padded_width, projection_route, select_joint_route_mode, JointProductRoute,
    JointRouteMode, JointRoutingEnvironment, MAX_JOINT_DECODE_ROWS,
};
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::inference::models::qwen35::spec_decode::nth_hidden_row;
use crate::inference::models::qwen35::Qwen35Config;
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

struct JointPlan {
    layers: Vec<LayerRoutes>,
    head: JointProductRoute,
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
    /// Joint route mode for this artifact and process, or why decode must
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
        let environment = JointRoutingEnvironment::from_process();
        let mode = select_joint_route_mode(environment, self.cfg.moe.is_some())?;
        check_mode_environment(mode, environment)?;
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

    /// Advance `rows.len()` distinct slots by one token each in one
    /// layer-major transaction. Under [`JointRouteMode::Exact`] every row's
    /// logits, hidden, KV codes, recurrent state, cursor and parity are
    /// bitwise those of its serial one-token forward. Refusal (admission,
    /// shape, slot range) happens before any cache write.
    pub(crate) fn forward_gpu_decode_joint(
        &self,
        rows: &[Qwen35JointRow],
        kv_cache: &mut HybridKvCache,
        mode: JointRouteMode,
    ) -> Result<Vec<Qwen35JointRowOutput>> {
        ensure!(
            (1..=MAX_JOINT_DECODE_ROWS).contains(&rows.len()),
            "Qwen joint decode requires 1..={MAX_JOINT_DECODE_ROWS} rows, got {}",
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
        check_mode_environment(mode, JointRoutingEnvironment::from_process())
            .map_err(|reason| anyhow!("Qwen joint decode unavailable: {reason}"))?;
        self.ensure_gpu_cache_primed()?;

        let pool_operation = decode_pool::profile::Operation::enter(self as *const _ as usize)?;
        let result = GPU_CACHE.with(|cell| -> Result<Vec<Qwen35JointRowOutput>> {
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
            self.encode_joint_decode(
                device,
                registry,
                layer_weights,
                output_head,
                &plan,
                rows,
                kv_cache,
                mode,
            )
        });
        pool_operation.complete()?;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_joint_decode(
        &self,
        device: &MlxDevice,
        registry: &mut KernelRegistry,
        layer_weights: &[LayerWeightsGpu],
        output_head: &OutputHeadGpu,
        plan: &JointPlan,
        rows: &[Qwen35JointRow],
        kv_cache: &mut HybridKvCache,
        mode: JointRouteMode,
    ) -> Result<Vec<Qwen35JointRowOutput>> {
        let cfg = &self.cfg;
        let hidden_size = cfg.hidden_size;
        let h = hidden_size as usize;
        let eps = cfg.rms_norm_eps;
        let vocab = cfg.vocab_size as usize;
        let width = joint_padded_width(mode, rows.len());
        let width_u32 = width as u32;
        let slots = rows.iter().map(|row| row.slot_id).collect::<Vec<_>>();
        let joint_rows = JointRows {
            slots: &slots,
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
        let mut embedded = device
            .alloc_buffer(width * h * 4, DType::F32, vec![width, h])
            .map_err(|e| anyhow!("allocate Qwen joint embeddings: {e}"))?;
        embed_tokens_gpu_into(
            &tokens,
            &self.token_embd,
            self.token_embd_native.as_ref(),
            cfg.vocab_size,
            hidden_size,
            device,
            registry,
            &mut embedded,
        )
        .context("Qwen joint embedding gather")?;

        let chain_n = joint_chain_n(cfg, layer_weights);
        let n_layers = layer_weights.len();
        ensure!(
            plan.layers.len() == n_layers,
            "Qwen joint plan covers {} of {n_layers} layers",
            plan.layers.len()
        );
        let mut hidden = embedded;
        let mut retained = Vec::with_capacity(n_layers);
        let mut chain: Option<CommandEncoder> = None;
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
                FfnWeightsGpu::DenseQ(weights) => build_dense_ffn_layer_gpu_q_into(
                    enc,
                    device,
                    registry,
                    &ffn_input,
                    weights,
                    Some(&ffn_residual),
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
                FfnWeightsGpu::Dense(_) | FfnWeightsGpu::Moe(_) => {
                    bail!("Qwen joint layer {layer_idx} has no admitted FFN arm")
                }
            };
            let last_in_group = (layer_idx + 1) % chain_n == 0 || layer_idx + 1 == n_layers;
            if last_in_group {
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
        let mut logits = device
            .alloc_buffer(width * vocab * 4, DType::F32, vec![width, vocab])
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
            if input.output != Qwen35JointOutput::GreedyToken {
                argmax.push(None);
                continue;
            }
            let index = device
                .alloc_buffer(4, DType::U32, vec![1])
                .map_err(|e| anyhow!("allocate Qwen joint argmax index: {e}"))?;
            let value = device
                .alloc_buffer(4, DType::F32, vec![1])
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
        enc.commit_and_wait_labeled("qwen35.joint_decode.output_head")
            .map_err(|e| anyhow!("Qwen joint decode command buffer: {e}"))?;
        drop(retained);

        let all_logits = logits
            .as_slice::<f32>()
            .map_err(|e| anyhow!("read Qwen joint logits: {e}"))?;
        rows.iter()
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
                let logits = (input.output != Qwen35JointOutput::GreedyToken)
                    .then(|| all_logits[row * vocab..(row + 1) * vocab].to_vec());
                let nextn_hidden = (input.output == Qwen35JointOutput::LogitsAndHidden)
                    .then(|| nth_hidden_row(&normed, hidden_size, row as u64))
                    .transpose()?;
                Ok(Qwen35JointRowOutput {
                    greedy_token,
                    logits,
                    nextn_hidden,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::gpu_delta_net::apply_proj_with_ggml_type;
    use crate::inference::models::qwen35::gpu_ffn::DenseFfnWeightsGpuQ;
    use crate::inference::models::qwen35::gpu_full_attn::{download_f32, upload_f32};
    use crate::inference::models::qwen35::joint_decode_core::project_rows;
    use mlx_native::ops::quantized_matmul_ggml::GgmlType;

    const F16_SCALE: [u8; 2] = 0x211F_u16.to_le_bytes(); // ≈ 0.01

    fn next(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        *seed
    }

    fn random_f32(seed: &mut u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| (next(seed) as i32 as f32 / i32::MAX as f32) * 0.5)
            .collect()
    }

    /// Valid GGML blocks with finite scales and random quants.
    fn synthetic_blocks(
        device: &MlxDevice,
        ggml_type: GgmlType,
        rows: u32,
        cols: u32,
        seed: &mut u32,
    ) -> MlxBuffer {
        let block_bytes = ggml_type.block_bytes() as usize;
        let blocks = (rows * cols / ggml_type.block_values()) as usize;
        let mut bytes = (0..blocks * block_bytes)
            .map(|_| (next(seed) >> 24) as u8)
            .collect::<Vec<_>>();
        let scale_offsets: &[usize] = match ggml_type {
            GgmlType::Q4_0 | GgmlType::Q8_0 => &[0],
            GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
            GgmlType::Q6_K => &[208],
            other => panic!("no synthetic layout for {other:?}"),
        };
        for block in bytes.chunks_exact_mut(block_bytes) {
            for &offset in scale_offsets {
                block[offset..offset + 2].copy_from_slice(&F16_SCALE);
            }
        }
        let mut buffer = device
            .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
            .expect("allocate synthetic GGML blocks");
        buffer
            .as_mut_slice::<u8>()
            .expect("map synthetic GGML blocks")
            .copy_from_slice(&bytes);
        buffer
    }

    fn bits(buffer: &MlxBuffer, count: usize) -> Vec<u32> {
        download_f32(buffer).expect("download")[..count]
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    /// Admitted block storage: one packed dispatch at `m = 2..=4` must equal
    /// the serial `m = 1` dispatch of every row, bit for bit, on the default
    /// (exact) routing policy. A failing type must leave
    /// `joint_decode_routes::packed_block_type`.
    #[test]
    fn packed_block_projections_match_per_row_dispatch_bitwise() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        if JointRoutingEnvironment::from_process().mv_ext {
            eprintln!("skipping: HF2Q_DECODE_MV_EXT=1 selects the non-exact policy");
            return;
        }
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
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
        if JointRoutingEnvironment::from_process().mv_ext {
            eprintln!("skipping: HF2Q_DECODE_MV_EXT=1 selects the non-exact policy");
            return;
        }
        let device = match MlxDevice::new() {
            Ok(device) => device,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
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
}
