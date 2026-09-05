//! Exact one-token DeepSeek-V4 hash- and score-routed MoE layers.

use std::sync::Once;

use anyhow::{bail, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_hyper_connection::{
    dispatch_hc_post, dispatch_hc_pre, dispatch_hc_split_sinkhorn,
};
use mlx_native::ops::deepseek_moe_activation::{
    dispatch_deepseek_moe_swiglu, dispatch_deepseek_moe_weighted_reduce, DEEPSEEK_MOE_HIDDEN_DIM,
    DEEPSEEK_MOE_INTER_DIM,
};
use mlx_native::ops::deepseek_moe_routing::{
    dispatch_deepseek_moe_hash_route, dispatch_deepseek_moe_sanitize_indices,
    dispatch_deepseek_moe_score_route, DEEPSEEK_MOE_EXPERTS, DEEPSEEK_MOE_TOP_K,
};
use mlx_native::{DType, GraphExecutor, IdMmScratch, MlxBuffer, MM_ID_ROUTING_THRESHOLD};

use super::forward_support::{
    alloc, alloc_host_input, alloc_persistent, expert_matmul, expert_matmul_pair, raw_matmul,
    rms_params, ExpertMatmulRoute,
};
use super::submission::{finish_or_commit, SubmissionChain};
use super::Deepseek4Model;

static PAIRED_MOE_PREFILL_LOG: Once = Once::new();

fn split_profile_stage(session: &mut GraphSession<'_>, label: &str) -> Result<()> {
    if std::env::var("HF2Q_DEEPSEEK_ENCODER_STAGES").as_deref() == Ok("1") {
        session
            .encoder_mut()
            .profile_stage_boundary(label)
            .with_context(|| format!("profile DeepSeek-V4 {label}"))?;
    }
    Ok(())
}

fn use_paired_expert_prefill(
    rows: usize,
    route: ExpertMatmulRoute,
    has_scratch: bool,
    threshold_override_present: bool,
) -> bool {
    rows > MM_ID_ROUTING_THRESHOLD as usize
        && route == ExpertMatmulRoute::Auto
        && has_scratch
        && !threshold_override_present
}

impl Deepseek4Model {
    /// Run layer-0 FFN for one token with checkpoint hash routing.
    ///
    /// The projection, safe-ID bridge, six routed experts, shared expert,
    /// weighted reduction, and HC-post all share one Metal command buffer.
    pub fn forward_layer0_ffn_one(
        &mut self,
        state: &MlxBuffer,
        token_id: u32,
    ) -> Result<MlxBuffer> {
        self.forward_ffn_one(state, token_id, 0, None, None)
    }

    /// Run one verifier FFN layer for one token.
    ///
    /// Layers in the checkpoint hash prefix preserve the stored expert order;
    /// later layers select with the learned bias and weight with unbiased
    /// sqrt-softplus scores.
    pub(super) fn forward_ffn_one(
        &mut self,
        state: &MlxBuffer,
        token_id: u32,
        layer: usize,
        in_flight: Option<&mut SubmissionChain>,
        shared_session: Option<&mut GraphSession<'_>>,
    ) -> Result<MlxBuffer> {
        self.forward_ffn_rows(
            state,
            &[token_id],
            layer,
            in_flight,
            shared_session,
            None,
            None,
        )
    }

    /// Run one verifier FFN layer for a complete prompt in layer-major order.
    pub(super) fn forward_ffn_rows(
        &mut self,
        state: &MlxBuffer,
        token_ids: &[u32],
        layer: usize,
        mut in_flight: Option<&mut SubmissionChain>,
        shared_session: Option<&mut GraphSession<'_>>,
        reusable_output_state: Option<MlxBuffer>,
        mut id_mm_scratch: Option<&mut [IdMmScratch; 2]>,
    ) -> Result<MlxBuffer> {
        if layer >= self.cfg.num_hidden_layers as usize {
            bail!(
                "DeepSeek-V4 FFN layer index {layer} is outside 0..{}",
                self.cfg.num_hidden_layers
            );
        }
        let hidden = self.cfg.hidden_size as usize;
        let hc = self.cfg.hyper_connection_count as usize;
        let experts = self.cfg.num_experts as usize;
        let top_k = self.cfg.num_experts_per_tok as usize;
        let inter = self.cfg.expert_intermediate_size as usize;
        let rows = token_ids.len();
        if rows == 0 {
            bail!("DeepSeek-V4 FFN requires at least one token row");
        }
        let routed_rows = rows
            .checked_mul(top_k)
            .context("DeepSeek-V4 routed FFN row count overflow")?;
        if hidden != DEEPSEEK_MOE_HIDDEN_DIM
            || experts != DEEPSEEK_MOE_EXPERTS
            || top_k != DEEPSEEK_MOE_TOP_K
            || inter != DEEPSEEK_MOE_INTER_DIM
            || self.cfg.hash_layer_count == 0
        {
            bail!(
                "DeepSeek-V4 native MoE requires hidden/expert/top-k/inter/hash = 4096/256/6/2048/enabled"
            );
        }
        if state.dtype() != DType::F32 || state.shape() != [rows, hc, hidden] {
            bail!(
                "DeepSeek-V4 layer {layer} FFN state must be F32 [{rows}, {hc}, {hidden}], got {} {:?}",
                state.dtype(),
                state.shape()
            );
        }

        let hc_hidden = hc * hidden;
        let mix_width = (2 + hc) * hc;
        let prefix = format!("blk.{layer}");
        let hc_fn = self
            .weights
            .raw_matrix_ref(&format!("{prefix}.hc_ffn_fn.weight"))?;
        let hc_base = self
            .weights
            .f32_state(&format!("{prefix}.hc_ffn_base.weight"))?;
        let hc_scale = self
            .weights
            .f32_state(&format!("{prefix}.hc_ffn_scale.weight"))?;
        let ffn_norm_weight = self
            .weights
            .f32_state(&format!("{prefix}.ffn_norm.weight"))?;
        let gate_weight = self
            .weights
            .raw_matrix_ref(&format!("{prefix}.ffn_gate_inp.weight"))?;
        let lookup = (layer < self.cfg.hash_layer_count as usize)
            .then(|| {
                self.weights
                    .i32_lookup(&format!("{prefix}.ffn_gate_tid2eid.weight"))
            })
            .transpose()?;
        let selection_bias = (layer >= self.cfg.hash_layer_count as usize)
            .then(|| {
                self.weights
                    .f32_state(&format!("{prefix}.exp_probs_b.bias"))
            })
            .transpose()?;
        let gate_experts = self
            .weights
            .raw_matrix_ref(&format!("{prefix}.ffn_gate_exps.weight"))?;
        let up_experts = self
            .weights
            .raw_matrix_ref(&format!("{prefix}.ffn_up_exps.weight"))?;
        let down_experts = self
            .weights
            .raw_matrix_ref(&format!("{prefix}.ffn_down_exps.weight"))?;
        let gate_shared = self
            .weights
            .raw_matrix_ref(&format!("{prefix}.ffn_gate_shexp.weight"))?;
        let up_shared = self
            .weights
            .raw_matrix_ref(&format!("{prefix}.ffn_up_shexp.weight"))?;
        let down_shared = self
            .weights
            .raw_matrix_ref(&format!("{prefix}.ffn_down_shexp.weight"))?;

        let device = self.ctx.device().clone();
        let state_flat = state.with_shape(vec![rows, hc_hidden])?;
        let state_norm = alloc(&device, DType::F32, vec![rows, hc_hidden], "FFN HC norm")?;
        let mixes = alloc(&device, DType::F32, vec![rows, mix_width], "FFN HC mixes")?;
        let pre = alloc(&device, DType::F32, vec![rows, hc], "FFN HC pre")?;
        let post = alloc(&device, DType::F32, vec![rows, hc], "FFN HC post")?;
        let comb = alloc(
            &device,
            DType::F32,
            vec![rows, hc, hc],
            "FFN HC combination",
        )?;
        let ffn_input = alloc(&device, DType::F32, vec![rows, hidden], "FFN input")?;
        let ffn_norm = alloc(&device, DType::F32, vec![rows, hidden], "FFN norm")?;
        let logits = alloc(&device, DType::F32, vec![rows, experts], "MoE logits")?;
        let indices = alloc(&device, DType::I32, vec![rows, top_k], "MoE indices")?;
        let safe_indices = alloc(&device, DType::U32, vec![rows, top_k], "safe MoE indices")?;
        let mut invalid_status =
            alloc_persistent(&device, DType::U32, vec![1], "MoE invalid status")?;
        invalid_status.as_logical_mut_slice::<u32>()?[0] = 0;
        let route_weights = alloc(&device, DType::F32, vec![rows, top_k], "MoE weights")?;
        let routed_gate = alloc(&device, DType::F32, vec![routed_rows, inter], "routed gate")?;
        let routed_up = alloc(&device, DType::F32, vec![routed_rows, inter], "routed up")?;
        let routed_activated = alloc(
            &device,
            DType::F32,
            vec![routed_rows, inter],
            "routed activation",
        )?;
        let routed_down = alloc(
            &device,
            DType::F32,
            vec![rows, top_k, hidden],
            "routed output",
        )?;
        let shared_gate = alloc(&device, DType::F32, vec![rows, inter], "shared gate")?;
        let shared_up = alloc(&device, DType::F32, vec![rows, inter], "shared up")?;
        let shared_activated = alloc(&device, DType::F32, vec![rows, inter], "shared activation")?;
        let shared_down = alloc(&device, DType::F32, vec![rows, hidden], "shared output")?;
        let ffn_output = alloc(&device, DType::F32, vec![rows, hidden], "FFN output")?;
        let output_shape = vec![rows, hc, hidden];
        let output_state = if let Some(output_state) = reusable_output_state {
            if output_state.dtype() != DType::F32 || output_state.shape() != output_shape {
                bail!(
                    "DeepSeek-V4 reusable FFN output must be F32 {:?}, got {} {:?}",
                    output_shape,
                    output_state.dtype(),
                    output_state.shape()
                );
            }
            output_state
        } else {
            alloc_persistent(&device, DType::F32, output_shape, "FFN HC output")?
        };
        let hc_params = rms_params(&device, self.cfg.rms_norm_eps, hc_hidden, "FFN HC params")?;
        let hidden_params = rms_params(&device, self.cfg.rms_norm_eps, hidden, "FFN norm params")?;
        let mut routing_token_ids =
            alloc_host_input(&device, DType::I32, vec![rows], "MoE token IDs")?;
        for (destination, &token_id) in routing_token_ids
            .as_logical_mut_slice::<i32>()?
            .iter_mut()
            .zip(token_ids)
        {
            *destination = i32::try_from(token_id)
                .context("DeepSeek-V4 token ID exceeds signed routing range")?;
        }
        let rows_u32 = u32::try_from(rows).context("DeepSeek-V4 FFN rows exceed u32")?;
        let routed_projection = if lookup.is_some() && rows == 1 {
            ExpertMatmulRoute::ForceMv
        } else {
            ExpertMatmulRoute::Auto
        };

        let registry = &mut self.ctx.registry;
        let mut encode = |session: &mut GraphSession<'_>| -> Result<()> {
            session.barrier_between(&[&state_flat], &[&state_norm]);
            session.rms_norm_no_scale_f32(
                registry,
                device.metal_device(),
                &state_flat,
                &state_norm,
                &hc_params,
                rows_u32,
                hc_hidden as u32,
            )?;
            raw_matmul(
                session,
                registry,
                &device,
                &state_norm,
                &hc_fn,
                &mixes,
                rows,
                mix_width,
                hc_hidden,
                "HC FFN function",
            )?;
            session.barrier_between(&[&mixes, hc_scale, hc_base], &[&pre, &post, &comb]);
            dispatch_hc_split_sinkhorn(
                session.encoder_mut(),
                registry,
                &device,
                &mixes,
                hc_scale,
                hc_base,
                &pre,
                &post,
                &comb,
                rows_u32,
            )?;
            session.barrier_between(&[state, &pre], &[&ffn_input]);
            dispatch_hc_pre(
                session.encoder_mut(),
                registry,
                &device,
                state,
                &pre,
                &ffn_input,
                rows_u32,
                hidden as u32,
            )?;
            session.barrier_between(&[&ffn_input, ffn_norm_weight], &[&ffn_norm]);
            session.rms_norm(
                registry,
                device.metal_device(),
                &ffn_input,
                ffn_norm_weight,
                &ffn_norm,
                &hidden_params,
                rows_u32,
                hidden as u32,
            )?;
            split_profile_stage(session, "DeepSeek-V4 FFN hyper-connection prelude")?;
            raw_matmul(
                session,
                registry,
                &device,
                &ffn_norm,
                &gate_weight,
                &logits,
                rows,
                experts,
                hidden,
                "MoE gate",
            )?;
            if let Some(lookup) = lookup {
                session.barrier_between(
                    &[&logits, &routing_token_ids, lookup],
                    &[&indices, &route_weights],
                );
                dispatch_deepseek_moe_hash_route(
                    session.encoder_mut(),
                    registry,
                    &device,
                    &logits,
                    &routing_token_ids,
                    lookup,
                    &indices,
                    &route_weights,
                    rows,
                    self.cfg.vocab_size as usize,
                )?;
            } else if let Some(selection_bias) = selection_bias {
                session.barrier_between(&[&logits, selection_bias], &[&indices, &route_weights]);
                dispatch_deepseek_moe_score_route(
                    session.encoder_mut(),
                    registry,
                    &device,
                    &logits,
                    selection_bias,
                    &indices,
                    &route_weights,
                    rows,
                )?;
            } else {
                bail!("DeepSeek-V4 FFN layer {layer} has no routing source");
            }
            session.barrier_between(&[&indices], &[&safe_indices]);
            dispatch_deepseek_moe_sanitize_indices(
                session.encoder_mut(),
                registry,
                &device,
                &indices,
                &safe_indices,
                &invalid_status,
                rows,
            )?;
            split_profile_stage(session, "DeepSeek-V4 FFN router")?;
            if use_paired_expert_prefill(
                rows,
                routed_projection,
                id_mm_scratch.is_some(),
                std::env::var_os("HF2Q_MM_ID_ROUTING_THRESHOLD").is_some(),
            ) {
                let scratch = id_mm_scratch
                    .as_deref_mut()
                    .context("DeepSeek-V4 paired routed projections require scratch")?;
                PAIRED_MOE_PREFILL_LOG.call_once(|| {
                    tracing::info!(
                        rows,
                        layer,
                        "DeepSeek-V4 paired MoE gate/up schedule enabled for large prefill"
                    );
                });
                expert_matmul_pair(
                    session,
                    registry,
                    &device,
                    &ffn_norm,
                    &gate_experts,
                    &up_experts,
                    &safe_indices,
                    &routed_gate,
                    &routed_up,
                    rows,
                    top_k,
                    experts,
                    inter,
                    hidden,
                    &mut scratch[0],
                    "paired routed gate/up",
                )?;
            } else {
                expert_matmul(
                    session,
                    registry,
                    &device,
                    &ffn_norm,
                    &gate_experts,
                    &safe_indices,
                    &routed_gate,
                    rows,
                    top_k,
                    experts,
                    inter,
                    hidden,
                    routed_projection,
                    id_mm_scratch.as_deref_mut().map(|scratch| &mut scratch[0]),
                    "routed gate",
                )?;
                expert_matmul(
                    session,
                    registry,
                    &device,
                    &ffn_norm,
                    &up_experts,
                    &safe_indices,
                    &routed_up,
                    rows,
                    top_k,
                    experts,
                    inter,
                    hidden,
                    routed_projection,
                    id_mm_scratch.as_deref_mut().map(|scratch| &mut scratch[1]),
                    "routed up",
                )?;
            }
            raw_matmul(
                session,
                registry,
                &device,
                &ffn_norm,
                &gate_shared,
                &shared_gate,
                rows,
                inter,
                hidden,
                "shared gate",
            )?;
            raw_matmul(
                session,
                registry,
                &device,
                &ffn_norm,
                &up_shared,
                &shared_up,
                rows,
                inter,
                hidden,
                "shared up",
            )?;
            split_profile_stage(session, "DeepSeek-V4 FFN gate/up projections")?;
            session.barrier_between(
                &[&routed_gate, &routed_up, &shared_gate, &shared_up],
                &[&routed_activated, &shared_activated],
            );
            dispatch_deepseek_moe_swiglu(
                session.encoder_mut(),
                registry,
                &device,
                &routed_gate,
                &routed_up,
                None,
                &routed_activated,
                &invalid_status,
                routed_rows,
            )?;
            dispatch_deepseek_moe_swiglu(
                session.encoder_mut(),
                registry,
                &device,
                &shared_gate,
                &shared_up,
                None,
                &shared_activated,
                &invalid_status,
                rows,
            )?;
            split_profile_stage(session, "DeepSeek-V4 FFN activations")?;
            if rows > MM_ID_ROUTING_THRESHOLD as usize && id_mm_scratch.is_some() {
                expert_matmul(
                    session,
                    registry,
                    &device,
                    &routed_activated,
                    &down_experts,
                    &safe_indices,
                    &routed_down,
                    rows,
                    top_k,
                    experts,
                    hidden,
                    inter,
                    ExpertMatmulRoute::SlottedMm,
                    id_mm_scratch.as_deref_mut().map(|scratch| &mut scratch[0]),
                    "routed down",
                )?;
            } else {
                expert_matmul(
                    session,
                    registry,
                    &device,
                    &routed_activated,
                    &down_experts,
                    &safe_indices,
                    &routed_down,
                    routed_rows,
                    1,
                    experts,
                    hidden,
                    inter,
                    ExpertMatmulRoute::Auto,
                    None,
                    "routed down",
                )?;
            }
            raw_matmul(
                session,
                registry,
                &device,
                &shared_activated,
                &down_shared,
                &shared_down,
                rows,
                hidden,
                inter,
                "shared down",
            )?;
            split_profile_stage(session, "DeepSeek-V4 FFN down projections")?;
            session.barrier_between(
                &[&indices, &route_weights, &routed_down, &shared_down],
                &[&ffn_output],
            );
            dispatch_deepseek_moe_weighted_reduce(
                session.encoder_mut(),
                registry,
                &device,
                &indices,
                &route_weights,
                &routed_down,
                &shared_down,
                &ffn_output,
                &invalid_status,
                rows,
            )?;
            session.barrier_between(&[&ffn_output, state, &post, &comb], &[&output_state]);
            dispatch_hc_post(
                session.encoder_mut(),
                registry,
                &device,
                &ffn_output,
                state,
                &post,
                &comb,
                &output_state,
                rows_u32,
                hidden as u32,
            )?;
            Ok(())
        };
        let inspect_status = shared_session.is_none() && in_flight.is_none();
        if let Some(in_flight) = in_flight.as_deref_mut() {
            in_flight.retain_moe_status(layer, invalid_status.clone());
        }
        if let Some(session) = shared_session {
            encode(session)?;
        } else {
            let local_executor = GraphExecutor::new(device.clone());
            let mut session = local_executor
                .begin()
                .with_context(|| format!("begin DeepSeek-V4 layer {layer} FFN"))?;
            encode(&mut session)?;
            finish_or_commit(
                session,
                in_flight,
                format!("execute DeepSeek-V4 layer {layer} FFN"),
            )?;
        }
        if inspect_status && invalid_status.as_logical_slice::<u32>()?[0] != 0 {
            bail!("DeepSeek-V4 layer {layer} MoE kernels rejected invalid routing data");
        }
        Ok(output_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_expert_prefill_is_large_auto_scratch_only() {
        let large = MM_ID_ROUTING_THRESHOLD as usize + 1;
        assert!(use_paired_expert_prefill(
            large,
            ExpertMatmulRoute::Auto,
            true,
            false,
        ));
        assert!(!use_paired_expert_prefill(
            MM_ID_ROUTING_THRESHOLD as usize,
            ExpertMatmulRoute::Auto,
            true,
            false,
        ));
        assert!(!use_paired_expert_prefill(
            large,
            ExpertMatmulRoute::ForceMv,
            true,
            false,
        ));
        assert!(!use_paired_expert_prefill(
            large,
            ExpertMatmulRoute::SlottedMm,
            true,
            false,
        ));
        assert!(!use_paired_expert_prefill(
            large,
            ExpertMatmulRoute::Auto,
            false,
            false,
        ));
        assert!(!use_paired_expert_prefill(
            large,
            ExpertMatmulRoute::Auto,
            true,
            true,
        ));
    }
}
