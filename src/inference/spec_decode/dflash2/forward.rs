//! Allocation-free family-neutral DFlash2 draft forward and KV-ring commit.

use super::{
    config::DFlash2Config,
    staged::{dispatch_prepared_head, dispatch_ring_attention, staged_tile_takes},
    tensors::{DFlash2LayerTensors, DFlash2ModelTensors, DFlash2Projection},
    workspace::DFlash2Workspace,
};
use crate::serve::forward_mlx_shared::MlxQWeight;
use anyhow::{anyhow, Context, Result};
use mlx_native::ops::{
    dense_mm_bf16::{dense_matmul_bf16_f32_tensor, DenseMmBf16F32Params},
    dense_mm_f32_f32::{dense_matmul_f32_f32_tensor, DenseMmF32F32Params},
    dflash2_attention_pack::{
        dispatch_dflash2_attention_mask_bf16, dispatch_dflash2_commit_kv_bf16,
        dispatch_dflash2_pack_attention_bf16, DFlash2AttentionPackParams,
    },
    dflash2_dynamic_conv::{
        dispatch_dflash2_dynamic_conv_f32, DFlash2ConvolutionStage, DFlash2DynamicConvParams,
    },
    elementwise::{dispatch_cast_f32_to_bf16_with_encoder, scalar_mul_f32},
    flash_attn_prefill::{
        dispatch_flash_attn_prefill_bf16_d128, dispatch_flash_attn_prefill_bf16_d256,
        FlashAttnPrefillLayout, FlashAttnPrefillParams,
    },
    q_gate_deinterleave::{dispatch_q_gate_deinterleave_f32, QGateDeinterleaveParams},
    qkv_split::{dispatch_qkv_split_f32, QkvSplitParams},
    qmm_affine::{
        dispatch_qmm_affine_t_packed_simd4_b4, dispatch_qmm_affine_t_packed_simd4_b4_gs64_bf16,
    },
    quantized_matmul_ggml::{dispatch_mm_v2_f16, quantized_matmul_ggml},
    rms_norm::dispatch_rms_norm,
    rope::dispatch_rope_neox_f32,
    silu_mul::dispatch_silu_mul,
    softcap::dispatch_softcap,
    transpose::permute_021_bf16_to_f32,
};
use mlx_native::{CommandEncoder, GgmlType, KernelRegistry, MlxBuffer, MlxDevice};

/// Project F32 rows. On the staged route a projection the staged tile takes
/// reads their BF16 rounding.
fn project(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    workspace: &DFlash2Workspace,
    projection: &DFlash2Projection,
    input: &MlxBuffer,
    output: &MlxBuffer,
    rows: usize,
    input_width: usize,
    output_width: usize,
) -> Result<()> {
    if let (Some(staged), DFlash2Projection::Af4G64 { .. }) =
        (workspace.staged.as_ref(), projection)
    {
        if staged_tile_takes(output_width) {
            staged.round_input(encoder, registry, device, input, rows, input_width)?;
            return project_rounded(
                encoder,
                registry,
                device,
                workspace,
                projection,
                &staged.projection_input,
                output,
                rows,
                input_width,
                output_width,
            );
        }
    }
    match projection {
        DFlash2Projection::Dense { weight } => project_bf16(
            encoder,
            registry,
            device,
            weight,
            input,
            output,
            rows,
            input_width,
            output_width,
        ),
        DFlash2Projection::Af4G64 { weights, metadata } => {
            dispatch_qmm_affine_t_packed_simd4_b4_gs64_bf16(
                encoder,
                registry,
                device.metal_device(),
                input,
                &workspace.af4_table,
                &workspace.af4_sums,
                weights,
                metadata,
                output,
                u32::try_from(rows)?,
                u32::try_from(output_width)?,
                u32::try_from(input_width)?,
            )
            .map_err(anyhow::Error::from)
        }
    }
}

/// Project BF16 rows on the staged tile.
fn project_rounded(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    workspace: &DFlash2Workspace,
    projection: &DFlash2Projection,
    input: &MlxBuffer,
    output: &MlxBuffer,
    rows: usize,
    input_width: usize,
    output_width: usize,
) -> Result<()> {
    let (Some(staged), DFlash2Projection::Af4G64 { weights, metadata }) =
        (workspace.staged.as_ref(), projection)
    else {
        return Err(anyhow!(
            "DFlash2 BF16 projection rows require the staged AF4G64 route"
        ));
    };
    staged.project(
        encoder,
        registry,
        device,
        weights,
        metadata,
        input,
        output,
        rows,
        input_width,
        output_width,
    )
}

fn project_bf16(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxBuffer,
    input: &MlxBuffer,
    output: &MlxBuffer,
    rows: usize,
    input_width: usize,
    output_width: usize,
) -> Result<()> {
    dense_matmul_bf16_f32_tensor(
        encoder,
        registry,
        device,
        weight,
        input,
        output,
        &DenseMmBf16F32Params {
            m: u32::try_from(rows)?,
            n: u32::try_from(output_width)?,
            k: u32::try_from(input_width)?,
            src0_batch: 1,
            src1_batch: 1,
        },
    )
    .map_err(anyhow::Error::from)
}

fn norm(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &MlxBuffer,
    rows: usize,
    width: usize,
) -> Result<()> {
    dispatch_rms_norm(
        encoder,
        registry,
        device.metal_device(),
        input,
        weight,
        output,
        params,
        u32::try_from(rows)?,
        u32::try_from(width)?,
    )
    .map_err(anyhow::Error::from)
}

fn convolution_params(config: &DFlash2Config, lanes: usize) -> DFlash2DynamicConvParams {
    DFlash2DynamicConvParams {
        lanes: lanes as u32,
        rows_per_lane: config.block_size as u32,
        hidden: config.hidden_size as u32,
        groups: (config.hidden_size / config.conv_group_size) as u32,
        stage: 0,
    }
}

fn attention_params(config: &DFlash2Config, lanes: usize) -> DFlash2AttentionPackParams {
    DFlash2AttentionPackParams {
        lanes: lanes as u32,
        q_heads: config.num_attention_heads as u32,
        kv_heads: config.num_key_value_heads as u32,
        head_dim: config.head_dim as u32,
        rows: config.block_size as u32,
        window: config.sliding_window as u32,
        packed_len: (config.sliding_window - 1 + config.block_size) as u32,
        max_commit: config.block_size as u32,
    }
}

/// Populate the shared position and ring metadata immediately before encoding a
/// draft cycle. One workspace represents exactly one scheduler batch width.
pub fn prepare_decode_positions(
    workspace: &mut DFlash2Workspace,
    config: &DFlash2Config,
    logical_positions: &[u32],
) -> Result<()> {
    if logical_positions.len() != workspace.max_lanes {
        return Err(anyhow!(
            "DFlash2 logical position count {} != workspace lanes {}",
            logical_positions.len(),
            workspace.max_lanes
        ));
    }
    let lengths = workspace.cache_lengths.as_mut_slice::<u32>()?;
    let writes = workspace.write_positions.as_mut_slice::<u32>()?;
    let positions = workspace.rope_positions.as_mut_slice::<u32>()?;
    for (lane, logical) in logical_positions.iter().copied().enumerate() {
        lengths[lane] = logical;
        writes[lane] = logical % config.sliding_window as u32;
        for row in 0..config.block_size {
            positions[lane * config.block_size + row] = logical
                .checked_add(row as u32)
                .ok_or_else(|| anyhow!("DFlash2 position overflow"))?;
        }
    }
    Ok(())
}

/// Attention of the current rows over the layer's rings and themselves. The
/// staged route's ring attention leaves BF16 rows in its scratch `output`;
/// the packed attention leaves F32 rows in `workspace.attention`.
#[allow(clippy::too_many_arguments)]
fn dispatch_attention(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    config: &DFlash2Config,
    workspace: &DFlash2Workspace,
    layer_index: usize,
) -> Result<()> {
    let lanes = workspace.max_lanes;
    let block = config.block_size;
    let rows = lanes * block;
    let head_dim = config.head_dim;
    let q_heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;

    dispatch_rope_neox_f32(
        encoder,
        registry,
        device.metal_device(),
        &workspace.query_normalized,
        &workspace.query,
        &workspace.rope_params,
        &workspace.rope_positions,
        None,
        rows as u32,
        q_heads as u32,
        head_dim as u32,
        head_dim as u32,
    )?;
    dispatch_rope_neox_f32(
        encoder,
        registry,
        device.metal_device(),
        &workspace.key_normalized,
        &workspace.key,
        &workspace.rope_params,
        &workspace.rope_positions,
        None,
        rows as u32,
        kv_heads as u32,
        head_dim as u32,
        head_dim as u32,
    )?;
    encoder.memory_barrier();

    if let Some(scratch) = workspace
        .staged
        .as_ref()
        .and_then(|staged| staged.attention.as_ref())
    {
        dispatch_ring_attention(
            encoder,
            registry,
            device,
            config,
            workspace,
            scratch,
            layer_index,
        )?;
        encoder.memory_barrier();
        return Ok(());
    }

    let current_query = workspace
        .query
        .with_shape(vec![lanes, block, q_heads, head_dim])?;
    let current_key = workspace
        .key
        .with_shape(vec![lanes, block, kv_heads, head_dim])?;
    let current_value = workspace
        .value
        .with_shape(vec![lanes, block, kv_heads, head_dim])?;
    let pack = attention_params(config, lanes);
    dispatch_dflash2_pack_attention_bf16(
        encoder,
        registry,
        device,
        &workspace.ring_keys[layer_index],
        &workspace.ring_values[layer_index],
        &current_query,
        &current_key,
        &current_value,
        &workspace.cache_lengths,
        &workspace.write_positions,
        &workspace.packed_query,
        &workspace.packed_key,
        &workspace.packed_value,
        pack,
    )?;
    dispatch_dflash2_attention_mask_bf16(
        encoder,
        registry,
        device,
        &workspace.cache_lengths,
        &workspace.attention_mask,
        pack,
    )?;
    encoder.memory_barrier();

    let flash = FlashAttnPrefillParams {
        n_heads: q_heads as u32,
        n_kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        seq_len_q: block as u32,
        seq_len_k: pack.packed_len,
        batch: lanes as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
        do_causal: false,
    };
    match head_dim {
        128 => dispatch_flash_attn_prefill_bf16_d128(
            encoder,
            device,
            registry,
            &workspace.packed_query,
            &workspace.packed_key,
            &workspace.packed_value,
            Some(&workspace.attention_mask),
            &workspace.attention_bf16,
            &flash,
            FlashAttnPrefillLayout::HeadMajor,
        )?,
        256 => dispatch_flash_attn_prefill_bf16_d256(
            encoder,
            device,
            registry,
            &workspace.packed_query,
            &workspace.packed_key,
            &workspace.packed_value,
            Some(&workspace.attention_mask),
            &workspace.attention_bf16,
            &flash,
        )?,
        other => {
            return Err(anyhow!(
                "unsupported DFlash2 attention head dimension {other}"
            ));
        }
    }
    encoder.memory_barrier();

    let lane_elements = q_heads * block * head_dim;
    for lane in 0..lanes {
        let input = workspace
            .attention_bf16
            .slice_view((lane * lane_elements * 2) as u64, lane_elements);
        let output = workspace
            .attention
            .slice_view((lane * lane_elements * 4) as u64, lane_elements);
        permute_021_bf16_to_f32(
            encoder,
            registry,
            device.metal_device(),
            &input,
            &output,
            q_heads,
            block,
            head_dim,
        )?;
    }
    encoder.memory_barrier();
    Ok(())
}

pub(super) fn dispatch_layer(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    config: &DFlash2Config,
    tensors: &DFlash2LayerTensors,
    workspace: &DFlash2Workspace,
    layer_index: usize,
) -> Result<()> {
    let rows = workspace.max_lanes * config.block_size;
    let hidden = config.hidden_size;
    let q_width = config.num_attention_heads * config.head_dim;
    let kv_width = config.num_key_value_heads * config.head_dim;
    let groups = config.hidden_size / config.conv_group_size;
    let current = layer_index & 1;
    let next = current ^ 1;

    norm(
        encoder,
        registry,
        device,
        &workspace.hidden[current],
        &tensors.input_norm,
        &workspace.normalized,
        &workspace.norm_params,
        rows,
        hidden,
    )?;
    encoder.memory_barrier();
    project(
        encoder,
        registry,
        device,
        workspace,
        &tensors.attention_dynamic,
        &workspace.normalized,
        &workspace.dynamic,
        rows,
        hidden,
        4 * groups,
    )?;
    encoder.memory_barrier();
    dispatch_dflash2_dynamic_conv_f32(
        encoder,
        registry,
        device,
        &workspace.normalized,
        &workspace.dynamic,
        &tensors.attention_convolution,
        None,
        &workspace.convolved,
        convolution_params(config, workspace.max_lanes),
        DFlash2ConvolutionStage::Prepare,
    )?;
    encoder.memory_barrier();

    project(
        encoder,
        registry,
        device,
        workspace,
        &tensors.qkv_projection,
        &workspace.convolved,
        &workspace.qkv,
        rows,
        hidden,
        q_width + 2 * kv_width,
    )?;
    encoder.memory_barrier();
    dispatch_qkv_split_f32(
        encoder,
        registry,
        device.metal_device(),
        &workspace.qkv,
        &workspace.query,
        &workspace.key,
        &workspace.value,
        &QkvSplitParams {
            seq: u32::try_from(rows)?,
            q_sp: u32::try_from(q_width)?,
            k_sp: u32::try_from(kv_width)?,
            v_sp: u32::try_from(kv_width)?,
        },
    )?;
    encoder.memory_barrier();
    norm(
        encoder,
        registry,
        device,
        &workspace.query,
        &tensors.query_norm,
        &workspace.query_normalized,
        &workspace.head_norm_params,
        rows * config.num_attention_heads,
        config.head_dim,
    )?;
    encoder.memory_barrier();
    norm(
        encoder,
        registry,
        device,
        &workspace.key,
        &tensors.key_norm,
        &workspace.key_normalized,
        &workspace.head_norm_params,
        rows * config.num_key_value_heads,
        config.head_dim,
    )?;
    encoder.memory_barrier();
    dispatch_attention(encoder, registry, device, config, workspace, layer_index)?;
    match workspace
        .staged
        .as_ref()
        .and_then(|staged| staged.attention.as_ref())
    {
        Some(scratch) => project_rounded(
            encoder,
            registry,
            device,
            workspace,
            &tensors.output_projection,
            &scratch.output,
            &workspace.projected,
            rows,
            q_width,
            hidden,
        )?,
        None => project(
            encoder,
            registry,
            device,
            workspace,
            &tensors.output_projection,
            &workspace.attention,
            &workspace.projected,
            rows,
            q_width,
            hidden,
        )?,
    }
    encoder.memory_barrier();
    dispatch_dflash2_dynamic_conv_f32(
        encoder,
        registry,
        device,
        &workspace.projected,
        &workspace.dynamic,
        &tensors.attention_convolution,
        Some(&workspace.hidden[current]),
        &workspace.residual,
        convolution_params(config, workspace.max_lanes),
        DFlash2ConvolutionStage::Residual,
    )?;
    encoder.memory_barrier();

    norm(
        encoder,
        registry,
        device,
        &workspace.residual,
        &tensors.post_attention_norm,
        &workspace.normalized,
        &workspace.norm_params,
        rows,
        hidden,
    )?;
    encoder.memory_barrier();
    project(
        encoder,
        registry,
        device,
        workspace,
        &tensors.mlp_dynamic,
        &workspace.normalized,
        &workspace.dynamic,
        rows,
        hidden,
        4 * groups,
    )?;
    encoder.memory_barrier();
    dispatch_dflash2_dynamic_conv_f32(
        encoder,
        registry,
        device,
        &workspace.normalized,
        &workspace.dynamic,
        &tensors.mlp_convolution,
        None,
        &workspace.convolved,
        convolution_params(config, workspace.max_lanes),
        DFlash2ConvolutionStage::Prepare,
    )?;
    encoder.memory_barrier();
    dispatch_mlp(encoder, registry, device, config, tensors, workspace)?;
    encoder.memory_barrier();
    dispatch_dflash2_dynamic_conv_f32(
        encoder,
        registry,
        device,
        &workspace.projected,
        &workspace.dynamic,
        &tensors.mlp_convolution,
        Some(&workspace.residual),
        &workspace.hidden[next],
        convolution_params(config, workspace.max_lanes),
        DFlash2ConvolutionStage::Residual,
    )?;
    encoder.memory_barrier();
    Ok(())
}

/// `projected = down(SiLU(gate(convolved)) * up(convolved))`. The staged
/// route runs gate and up in one pass into BF16 rows the down projection
/// reads in place.
fn dispatch_mlp(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    config: &DFlash2Config,
    tensors: &DFlash2LayerTensors,
    workspace: &DFlash2Workspace,
) -> Result<()> {
    let rows = workspace.max_lanes * config.block_size;
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    if let (Some(staged), Some(halves)) =
        (workspace.staged.as_ref(), tensors.gate_up_halves.as_ref())
    {
        staged.round_input(
            encoder,
            registry,
            device,
            &workspace.convolved,
            rows,
            hidden,
        )?;
        staged.gate_up(
            encoder,
            registry,
            device,
            halves,
            rows,
            hidden,
            intermediate,
        )?;
        encoder.memory_barrier();
        return project_rounded(
            encoder,
            registry,
            device,
            workspace,
            &tensors.down_projection,
            &staged.gated,
            &workspace.projected,
            rows,
            intermediate,
            hidden,
        );
    }
    project(
        encoder,
        registry,
        device,
        workspace,
        &tensors.gate_up_projection,
        &workspace.convolved,
        &workspace.gate_up,
        rows,
        hidden,
        2 * intermediate,
    )?;
    encoder.memory_barrier();
    let fused_gate_up = workspace
        .gate_up
        .with_shape(vec![rows, 1, 2 * intermediate])?;
    let gate = workspace.gate.with_shape(vec![rows, 1, intermediate])?;
    let up = workspace.up.with_shape(vec![rows, 1, intermediate])?;
    dispatch_q_gate_deinterleave_f32(
        encoder,
        registry,
        device,
        &fused_gate_up,
        &gate,
        &up,
        QGateDeinterleaveParams {
            m: u32::try_from(rows)?,
            n_heads: 1,
            head_dim: u32::try_from(intermediate)?,
        },
    )?;
    encoder.memory_barrier();
    dispatch_silu_mul(
        encoder,
        registry,
        device.metal_device(),
        &workspace.gate,
        &workspace.up,
        &workspace.intermediate,
        &workspace.silu_params,
        (rows * intermediate) as u32,
    )?;
    encoder.memory_barrier();
    project(
        encoder,
        registry,
        device,
        workspace,
        &tensors.down_projection,
        &workspace.intermediate,
        &workspace.projected,
        rows,
        intermediate,
        hidden,
    )
}

/// Target-family vocabulary projection. Dense BF16 is useful for controlled
/// parity fixtures; production adapters pass the target's exact native GGUF
/// projection, including tied embeddings and affine overlays.
pub enum DFlash2VocabularyHead<'a> {
    DenseBf16(&'a MlxBuffer),
    Native(&'a MlxQWeight),
}

#[allow(clippy::too_many_arguments)]
fn dispatch_vocabulary_head(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    head: DFlash2VocabularyHead<'_>,
    input: &MlxBuffer,
    output: &MlxBuffer,
    head_meta: &mut MlxBuffer,
    rows: usize,
    hidden: usize,
    vocabulary: usize,
) -> Result<()> {
    let DFlash2VocabularyHead::Native(weight) = head else {
        let DFlash2VocabularyHead::DenseBf16(weight) = head else {
            unreachable!()
        };
        return project_bf16(
            encoder, registry, device, weight, input, output, rows, hidden, vocabulary,
        );
    };
    if weight.info.rows != vocabulary || weight.info.cols != hidden {
        return Err(anyhow!(
            "DFlash2 native vocabulary head is {}x{}, expected {}x{}",
            weight.info.rows,
            weight.info.cols,
            vocabulary,
            hidden
        ));
    }
    let m = u32::try_from(rows)?;
    let n = u32::try_from(vocabulary)?;
    let k = u32::try_from(hidden)?;
    if let Some(extra) = weight.affine.as_ref() {
        if extra.bits != 4 || extra.group_size != 32 {
            return Err(anyhow!(
                "DFlash2 affine vocabulary head requires bits=4, group_size=32"
            ));
        }
        head_meta
            .as_mut_slice::<u32>()?
            .copy_from_slice(&[m, n, k, extra.group_size]);
        return dispatch_qmm_affine_t_packed_simd4_b4(
            encoder,
            registry,
            device.metal_device(),
            input,
            &weight.buffer,
            &extra.scales,
            &extra.biases,
            output,
            head_meta,
            m,
            n,
            k,
            extra.group_size,
            extra.bits,
        )
        .map_err(anyhow::Error::from);
    }
    if rows > mlx_native::ops::quantized_matmul_ggml::MM_ROUTING_THRESHOLD as usize {
        if let Some(shadow) = weight.f16_shadow.as_ref() {
            return dispatch_mm_v2_f16(encoder, registry, device, shadow, input, output, m, n, k)
                .map_err(anyhow::Error::from);
        }
    }
    match weight.info.ggml_dtype {
        GgmlType::BF16 => dense_matmul_bf16_f32_tensor(
            encoder,
            registry,
            device,
            &weight.buffer,
            input,
            output,
            &DenseMmBf16F32Params {
                m,
                n,
                k,
                src0_batch: 1,
                src1_batch: 1,
            },
        )
        .map_err(anyhow::Error::from),
        GgmlType::F32 => dense_matmul_f32_f32_tensor(
            encoder,
            registry,
            device,
            &weight.buffer,
            input,
            output,
            &DenseMmF32F32Params {
                m,
                n,
                k,
                src0_batch: 1,
                src1_batch: 1,
            },
        )
        .map_err(anyhow::Error::from),
        GgmlType::F16 => dispatch_mm_v2_f16(
            encoder,
            registry,
            device,
            &weight.buffer,
            input,
            output,
            m,
            n,
            k,
        )
        .map_err(anyhow::Error::from),
        _ => quantized_matmul_ggml(
            encoder,
            registry,
            device,
            input,
            &weight.buffer,
            output,
            &weight.matmul_params(m)?,
        )
        .map_err(anyhow::Error::from),
    }
}

/// Encode the complete five/six-layer draft forward. The caller must populate
/// `workspace.hidden[0]` with anchor + mask embeddings and call
/// [`prepare_decode_positions`] first.
pub fn dispatch_dflash2_forward(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    config: &DFlash2Config,
    tensors: &DFlash2ModelTensors,
    vocabulary_head: DFlash2VocabularyHead<'_>,
    workspace: &mut DFlash2Workspace,
) -> Result<()> {
    if tensors.layers.len() != config.num_hidden_layers {
        return Err(anyhow!("DFlash2 layer tensor count mismatch"));
    }
    let rows = workspace.max_lanes * config.block_size;
    encoder.memory_barrier();
    for (index, layer) in tensors.layers.iter().enumerate() {
        dispatch_layer(encoder, registry, device, config, layer, workspace, index)
            .with_context(|| format!("DFlash2 layer {index}"))?;
    }

    norm(
        encoder,
        registry,
        device,
        &workspace.hidden[config.num_hidden_layers & 1],
        &tensors.final_norm,
        &workspace.final_hidden,
        &workspace.norm_params,
        rows,
        config.hidden_size,
    )?;
    encoder.memory_barrier();
    let prepared = match (&vocabulary_head, workspace.staged.as_mut()) {
        (DFlash2VocabularyHead::Native(weight), Some(staged))
            if weight.info.rows == config.vocab_size && weight.info.cols == config.hidden_size =>
        {
            dispatch_prepared_head(
                encoder,
                registry,
                device,
                staged,
                weight,
                &workspace.final_hidden,
                &workspace.logits,
                rows,
            )?
        }
        _ => false,
    };
    if !prepared {
        dispatch_vocabulary_head(
            encoder,
            registry,
            device,
            vocabulary_head,
            &workspace.final_hidden,
            &workspace.logits,
            &mut workspace.head_meta,
            rows,
            config.hidden_size,
            config.vocab_size,
        )?;
    }
    project(
        encoder,
        registry,
        device,
        workspace,
        &tensors.selector_projection,
        &workspace.final_hidden,
        &workspace.selector_hidden_f32,
        rows,
        config.hidden_size,
        config.selector_rank,
    )?;
    encoder.memory_barrier();
    if let Some(multiplier) = config.output_multiplier {
        scalar_mul_f32(
            encoder,
            registry,
            device.metal_device(),
            &workspace.logits,
            &workspace.logits,
            rows * config.vocab_size,
            multiplier,
        )?;
        encoder.memory_barrier();
    }
    if let Some(cap) = config.final_logit_softcapping {
        dispatch_softcap(
            encoder,
            registry,
            device.metal_device(),
            &workspace.logits,
            &workspace.logits,
            &workspace.softcap_params,
            cap,
        )?;
        encoder.memory_barrier();
    }
    dispatch_cast_f32_to_bf16_with_encoder(
        encoder,
        registry,
        device.metal_device(),
        &workspace.selector_hidden_f32,
        &workspace.selector_hidden_bf16,
        (rows * config.selector_rank) as u32,
    )?;
    encoder.memory_barrier();
    Ok(())
}

/// Project accepted target hidden states and append their per-layer K/V rows to
/// the persistent rings. `retained_counts` stays GPU-resident so rejected rows
/// never mutate draft state.
pub fn dispatch_dflash2_context_commit(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    config: &DFlash2Config,
    tensors: &DFlash2ModelTensors,
    retained_counts: &MlxBuffer,
    workspace: &DFlash2Workspace,
) -> Result<()> {
    let rows = workspace.max_lanes * config.block_size;
    project(
        encoder,
        registry,
        device,
        workspace,
        &tensors.context_projection,
        &workspace.captured_target_hidden,
        &workspace.hidden[0],
        rows,
        config.fc_input_dim(),
        config.hidden_size,
    )?;
    encoder.memory_barrier();
    norm(
        encoder,
        registry,
        device,
        &workspace.hidden[0],
        &tensors.hidden_norm,
        &workspace.normalized,
        &workspace.norm_params,
        rows,
        config.hidden_size,
    )?;
    encoder.memory_barrier();
    let pack = attention_params(config, workspace.max_lanes);
    let kv_width = config.num_key_value_heads * config.head_dim;
    // Every layer's key and value projection reads the same normalized rows;
    // the staged route rounds them once.
    let rounded = match workspace.staged.as_ref() {
        Some(staged) if staged_tile_takes(2 * kv_width) => {
            staged.round_input(
                encoder,
                registry,
                device,
                &workspace.normalized,
                rows,
                config.hidden_size,
            )?;
            Some(&staged.projection_input)
        }
        _ => None,
    };
    for (index, layer) in tensors.layers.iter().enumerate() {
        match rounded {
            Some(input) => project_rounded(
                encoder,
                registry,
                device,
                workspace,
                &layer.context_kv_projection,
                input,
                &workspace.qkv,
                rows,
                config.hidden_size,
                2 * kv_width,
            )?,
            None => project(
                encoder,
                registry,
                device,
                workspace,
                &layer.context_kv_projection,
                &workspace.normalized,
                &workspace.qkv,
                rows,
                config.hidden_size,
                2 * kv_width,
            )?,
        }
        encoder.memory_barrier();
        let fused_kv = workspace
            .qkv
            .slice_view(0, rows * 2 * kv_width)
            .with_shape(vec![rows, 1, 2 * kv_width])?;
        let key = workspace.key.with_shape(vec![rows, 1, kv_width])?;
        let value = workspace.value.with_shape(vec![rows, 1, kv_width])?;
        dispatch_q_gate_deinterleave_f32(
            encoder,
            registry,
            device,
            &fused_kv,
            &key,
            &value,
            QGateDeinterleaveParams {
                m: u32::try_from(rows)?,
                n_heads: 1,
                head_dim: u32::try_from(kv_width)?,
            },
        )?;
        encoder.memory_barrier();
        norm(
            encoder,
            registry,
            device,
            &workspace.key,
            &layer.key_norm,
            &workspace.key_normalized,
            &workspace.head_norm_params,
            rows * config.num_key_value_heads,
            config.head_dim,
        )?;
        encoder.memory_barrier();
        dispatch_rope_neox_f32(
            encoder,
            registry,
            device.metal_device(),
            &workspace.key_normalized,
            &workspace.key,
            &workspace.rope_params,
            &workspace.rope_positions,
            None,
            rows as u32,
            config.num_key_value_heads as u32,
            config.head_dim as u32,
            config.head_dim as u32,
        )?;
        encoder.memory_barrier();
        let commit_key = workspace.key.with_shape(vec![
            workspace.max_lanes,
            config.block_size,
            config.num_key_value_heads,
            config.head_dim,
        ])?;
        let commit_value = workspace.value.with_shape(vec![
            workspace.max_lanes,
            config.block_size,
            config.num_key_value_heads,
            config.head_dim,
        ])?;
        dispatch_dflash2_commit_kv_bf16(
            encoder,
            registry,
            device,
            &commit_key,
            &commit_value,
            retained_counts,
            &workspace.write_positions,
            &workspace.ring_keys[index],
            &workspace.ring_values[index],
            pack,
        )?;
        encoder.memory_barrier();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::dflash2::{
        config::DFlash2LayerType, selector::dispatch_dflash2_selection,
    };
    use mlx_native::{DType, MlxError};

    fn buffer(
        device: &MlxDevice,
        dtype: DType,
        shape: &[usize],
    ) -> std::result::Result<MlxBuffer, MlxError> {
        let elements = shape.iter().product::<usize>();
        device.alloc_buffer(elements * dtype.size_of(), dtype, shape.to_vec())
    }

    fn projection(device: &MlxDevice, shape: &[usize]) -> DFlash2Projection {
        let columns = *shape.last().unwrap();
        let rows = shape[..shape.len() - 1].iter().product::<usize>();
        let bytes = rows * columns * 2;
        DFlash2Projection::Dense {
            weight: device
                .alloc_buffer(bytes, DType::BF16, shape.to_vec())
                .unwrap(),
        }
    }

    fn norm_weight(device: &MlxDevice, width: usize) -> MlxBuffer {
        let mut output = buffer(device, DType::F32, &[width]).unwrap();
        output.as_mut_slice::<f32>().unwrap().fill(1.0);
        output
    }

    fn config() -> DFlash2Config {
        DFlash2Config {
            draft_model_type: "qwen3".into(),
            hidden_size: 128,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 128,
            vocab_size: 16,
            num_target_layers: 1,
            target_layer_ids: vec![0],
            layer_types: vec![DFlash2LayerType::SlidingAttention],
            max_position_embeddings: 128,
            sliding_window: 32,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            block_size: 8,
            conv_group_size: 16,
            conv_kernel_size: 2,
            selector_rank: 256,
            selector_top_k: 16,
            mask_token_id: 0,
            final_logit_softcapping: None,
            output_multiplier: None,
        }
    }

    fn tensors(device: &MlxDevice, cfg: &DFlash2Config) -> DFlash2ModelTensors {
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let groups = cfg.convolution_groups();
        let layer = DFlash2LayerTensors {
            attention_dynamic: projection(device, &[4 * groups, hidden]),
            attention_convolution: buffer(device, DType::BF16, &[2, 2, hidden]).unwrap(),
            input_norm: norm_weight(device, hidden),
            qkv_projection: projection(device, &[3 * hidden, hidden]),
            context_kv_projection: projection(device, &[2 * hidden, hidden]),
            output_projection: projection(device, &[hidden, hidden]),
            query_norm: norm_weight(device, cfg.head_dim),
            key_norm: norm_weight(device, cfg.head_dim),
            mlp_dynamic: projection(device, &[4 * groups, hidden]),
            post_attention_norm: norm_weight(device, hidden),
            gate_up_projection: projection(device, &[2 * intermediate, hidden]),
            gate_up_halves: None,
            down_projection: projection(device, &[hidden, intermediate]),
            mlp_convolution: buffer(device, DType::BF16, &[2, 2, hidden]).unwrap(),
        };
        DFlash2ModelTensors {
            context_projection: projection(device, &[hidden, cfg.fc_input_dim()]),
            hidden_norm: norm_weight(device, hidden),
            final_norm: norm_weight(device, hidden),
            selector_projection: projection(device, &[256, hidden]),
            predecessor_codebook: buffer(device, DType::BF16, &[cfg.vocab_size, 256]).unwrap(),
            successor_codebook: buffer(device, DType::BF16, &[cfg.vocab_size, 256]).unwrap(),
            mask_embedding: None,
            layers: vec![layer],
        }
    }

    #[test]
    fn synthetic_forward_selection_and_commit_run_on_metal_with_inactive_lanes() {
        let device = MlxDevice::new().unwrap();
        let mut registry = KernelRegistry::new();
        let cfg = config();
        let tensors = tensors(&device, &cfg);
        let head = buffer(&device, DType::BF16, &[cfg.vocab_size, cfg.hidden_size]).unwrap();
        let mut workspace = DFlash2Workspace::new(&device, &cfg, 4, false).unwrap();
        prepare_decode_positions(&mut workspace, &cfg, &[0; 4]).unwrap();
        workspace.uniforms.as_mut_slice::<f32>().unwrap().fill(0.5);
        let mut retained = buffer(&device, DType::U32, &[4]).unwrap();
        retained.as_mut_slice::<u32>().unwrap().fill(0);
        retained.as_mut_slice::<u32>().unwrap()[0] = 1;

        let mut encoder = device.command_encoder().unwrap();
        dispatch_dflash2_forward(
            &mut encoder,
            &mut registry,
            &device,
            &cfg,
            &tensors,
            DFlash2VocabularyHead::DenseBf16(&head),
            &mut workspace,
        )
        .unwrap();
        dispatch_dflash2_selection(
            &mut encoder,
            &mut registry,
            &device,
            &cfg,
            &tensors,
            &workspace,
            &[0; 4],
            &[1.0; 4],
            0,
            &[8, 1, 1, 1],
            &[0],
            None,
        )
        .unwrap();
        dispatch_dflash2_context_commit(
            &mut encoder,
            &mut registry,
            &device,
            &cfg,
            &tensors,
            &retained,
            &workspace,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        assert!(workspace
            .logits
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(
            workspace.proposed_tokens.as_slice::<u32>().unwrap(),
            &[0; 4 * 7]
        );
        assert_eq!(workspace.ring_keys[0].as_slice::<u16>().unwrap()[0], 0);
    }
}
