//! Shared one-token HC/QKV graph for compressed DeepSeek-V4 attention.

use anyhow::Result;
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_hyper_connection::{
    dispatch_hc_post, dispatch_hc_pre, dispatch_hc_split_sinkhorn,
};
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_bf16, DeepSeekTailRopeParams,
};
use mlx_native::ops::elementwise::dispatch_cast_bf16_to_f32_with_encoder;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::attention_weights::AttentionWeightsRef;
use super::forward_support::{alloc, grouped_output_a, raw_matmul, rms_params};
use super::Deepseek4Config;

pub(super) struct CompressedAttentionCoreArena {
    pub state_norm: MlxBuffer,
    pub mixes: MlxBuffer,
    pub pre: MlxBuffer,
    pub post: MlxBuffer,
    pub comb: MlxBuffer,
    pub attn_input: MlxBuffer,
    pub attn_norm: MlxBuffer,
    pub q_a: MlxBuffer,
    pub q_a_norm: MlxBuffer,
    pub q: MlxBuffer,
    pub q_norm: MlxBuffer,
    pub kv: MlxBuffer,
    pub kv_norm: MlxBuffer,
    pub attention_unrotated: MlxBuffer,
    pub attention_f32: MlxBuffer,
    pub output_a: MlxBuffer,
    pub output_b: MlxBuffer,
    pub output_state: MlxBuffer,
    pub hc_params: MlxBuffer,
    pub hidden_params: MlxBuffer,
    pub rank_params: MlxBuffer,
    pub head_params: MlxBuffer,
}

impl CompressedAttentionCoreArena {
    pub(super) fn new(device: &MlxDevice, cfg: &Deepseek4Config) -> Result<Self> {
        let hidden = cfg.hidden_size as usize;
        let hc = cfg.hyper_connection_count as usize;
        let hc_hidden = hc * hidden;
        let mix_width = (2 + hc) * hc;
        let heads = cfg.num_attention_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let q_rank = cfg.q_lora_rank as usize;
        let output_width = cfg.output_groups as usize * cfg.o_lora_rank as usize;
        Ok(Self {
            state_norm: alloc(device, DType::F32, vec![1, hc_hidden], "compressed HC norm")?,
            mixes: alloc(
                device,
                DType::F32,
                vec![1, mix_width],
                "compressed HC mixes",
            )?,
            pre: alloc(device, DType::F32, vec![1, hc], "compressed HC pre")?,
            post: alloc(device, DType::F32, vec![1, hc], "compressed HC post")?,
            comb: alloc(
                device,
                DType::F32,
                vec![1, hc, hc],
                "compressed HC combination",
            )?,
            attn_input: alloc(
                device,
                DType::F32,
                vec![1, hidden],
                "compressed attention input",
            )?,
            attn_norm: alloc(
                device,
                DType::F32,
                vec![1, hidden],
                "compressed attention norm",
            )?,
            q_a: alloc(device, DType::F32, vec![1, q_rank], "compressed query rank")?,
            q_a_norm: alloc(
                device,
                DType::F32,
                vec![1, q_rank],
                "compressed query rank norm",
            )?,
            q: alloc(
                device,
                DType::F32,
                vec![1, heads, head_dim],
                "compressed query heads",
            )?,
            q_norm: alloc(
                device,
                DType::F32,
                vec![1, 1, heads, head_dim],
                "compressed query norm",
            )?,
            kv: alloc(device, DType::F32, vec![1, head_dim], "compressed raw KV")?,
            kv_norm: alloc(
                device,
                DType::F32,
                vec![1, 1, 1, head_dim],
                "compressed raw KV norm",
            )?,
            attention_unrotated: alloc(
                device,
                DType::BF16,
                vec![1, 1, heads, head_dim],
                "compressed inverse attention",
            )?,
            attention_f32: alloc(
                device,
                DType::F32,
                vec![1, heads, head_dim],
                "compressed attention f32",
            )?,
            output_a: alloc(
                device,
                DType::F32,
                vec![1, output_width],
                "compressed output A",
            )?,
            output_b: alloc(device, DType::F32, vec![1, hidden], "compressed output B")?,
            output_state: alloc(
                device,
                DType::F32,
                vec![1, hc, hidden],
                "compressed HC output",
            )?,
            hc_params: rms_params(device, cfg.rms_norm_eps, hc_hidden, "compressed HC params")?,
            hidden_params: rms_params(
                device,
                cfg.rms_norm_eps,
                hidden,
                "compressed hidden params",
            )?,
            rank_params: rms_params(device, cfg.rms_norm_eps, q_rank, "compressed rank params")?,
            head_params: rms_params(device, cfg.rms_norm_eps, head_dim, "compressed head params")?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_compressed_attention_prelude(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    state: &MlxBuffer,
    weights: &AttentionWeightsRef<'_>,
    arena: &CompressedAttentionCoreArena,
) -> Result<()> {
    let hidden = cfg.hidden_size as usize;
    let hc = cfg.hyper_connection_count as usize;
    let hc_hidden = hc * hidden;
    let heads = cfg.num_attention_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_rank = cfg.q_lora_rank as usize;
    let state_flat = state.with_shape(vec![1, hc_hidden])?;
    session.barrier_between(&[&state_flat], &[&arena.state_norm]);
    session.rms_norm_no_scale_f32(
        registry,
        device.metal_device(),
        &state_flat,
        &arena.state_norm,
        &arena.hc_params,
        1,
        hc_hidden as u32,
    )?;
    raw_matmul(
        session,
        registry,
        device,
        &arena.state_norm,
        &weights.hc_fn,
        &arena.mixes,
        1,
        (2 + hc) * hc,
        hc_hidden,
        "compressed HC attention function",
    )?;
    session.barrier_between(
        &[&arena.mixes, weights.hc_scale, weights.hc_base],
        &[&arena.pre, &arena.post, &arena.comb],
    );
    dispatch_hc_split_sinkhorn(
        session.encoder_mut(),
        registry,
        device,
        &arena.mixes,
        weights.hc_scale,
        weights.hc_base,
        &arena.pre,
        &arena.post,
        &arena.comb,
        1,
    )?;
    session.barrier_between(&[state, &arena.pre], &[&arena.attn_input]);
    dispatch_hc_pre(
        session.encoder_mut(),
        registry,
        device,
        state,
        &arena.pre,
        &arena.attn_input,
        1,
        hidden as u32,
    )?;
    session.barrier_between(&[&arena.attn_input, weights.attn_norm], &[&arena.attn_norm]);
    session.rms_norm(
        registry,
        device.metal_device(),
        &arena.attn_input,
        weights.attn_norm,
        &arena.attn_norm,
        &arena.hidden_params,
        1,
        hidden as u32,
    )?;
    raw_matmul(
        session,
        registry,
        device,
        &arena.attn_norm,
        &weights.q_a,
        &arena.q_a,
        1,
        q_rank,
        hidden,
        "compressed query A",
    )?;
    raw_matmul(
        session,
        registry,
        device,
        &arena.attn_norm,
        &weights.kv,
        &arena.kv,
        1,
        head_dim,
        hidden,
        "compressed shared KV",
    )?;
    session.barrier_between(&[&arena.q_a, weights.q_a_norm], &[&arena.q_a_norm]);
    session.rms_norm(
        registry,
        device.metal_device(),
        &arena.q_a,
        weights.q_a_norm,
        &arena.q_a_norm,
        &arena.rank_params,
        1,
        q_rank as u32,
    )?;
    raw_matmul(
        session,
        registry,
        device,
        &arena.q_a_norm,
        &weights.q_b,
        &arena.q,
        1,
        heads * head_dim,
        q_rank,
        "compressed query B",
    )?;
    session.barrier_between(&[&arena.q], &[&arena.q_norm]);
    session.rms_norm_no_scale_f32(
        registry,
        device.metal_device(),
        &arena.q,
        &arena.q_norm,
        &arena.head_params,
        heads as u32,
        head_dim as u32,
    )?;
    session.barrier_between(&[&arena.kv, weights.kv_norm], &[&arena.kv_norm]);
    session.rms_norm(
        registry,
        device.metal_device(),
        &arena.kv,
        weights.kv_norm,
        &arena.kv_norm,
        &arena.head_params,
        1,
        head_dim as u32,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_compressed_attention_epilogue(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    state: &MlxBuffer,
    weights: &AttentionWeightsRef<'_>,
    arena: &CompressedAttentionCoreArena,
    attention: &MlxBuffer,
    positions: &MlxBuffer,
    frequencies: &MlxBuffer,
) -> Result<()> {
    let hidden = cfg.hidden_size as usize;
    let heads = cfg.num_attention_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let groups = cfg.output_groups as usize;
    let o_rank = cfg.o_lora_rank as usize;
    let inverse = DeepSeekTailRopeParams {
        batch: 1,
        seq_len: 1,
        heads: heads as u32,
        head_dim: head_dim as u32,
        rope_dim: cfg.rope_head_dim,
        inverse: 1,
    };
    session.barrier_between(
        &[attention, positions, frequencies],
        &[&arena.attention_unrotated],
    );
    dispatch_deepseek_tail_rope_bf16(
        session.encoder_mut(),
        registry,
        device,
        attention,
        positions,
        frequencies,
        &arena.attention_unrotated,
        &inverse,
    )?;
    session.barrier_between(&[&arena.attention_unrotated], &[&arena.attention_f32]);
    dispatch_cast_bf16_to_f32_with_encoder(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        &arena.attention_unrotated,
        &arena.attention_f32,
        (heads * head_dim) as u32,
    )?;
    grouped_output_a(
        session,
        registry,
        device,
        &arena.attention_f32,
        &weights.output_a,
        &arena.output_a,
        groups,
        o_rank,
        heads,
        head_dim,
    )?;
    raw_matmul(
        session,
        registry,
        device,
        &arena.output_a,
        &weights.output_b,
        &arena.output_b,
        1,
        hidden,
        groups * o_rank,
        "compressed output B",
    )?;
    session.barrier_between(
        &[&arena.output_b, state, &arena.post, &arena.comb],
        &[&arena.output_state],
    );
    dispatch_hc_post(
        session.encoder_mut(),
        registry,
        device,
        &arena.output_b,
        state,
        &arena.post,
        &arena.comb,
        &arena.output_state,
        1,
        hidden as u32,
    )?;
    Ok(())
}
