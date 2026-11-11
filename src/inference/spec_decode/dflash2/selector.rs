//! Coherent DFlash2 proposal selection over the reusable decode workspace.

use super::{
    config::DFlash2Config, tensors::DFlash2ModelTensors, verify_budget::VerifyRowsModel,
    workspace::DFlash2Workspace,
};
use anyhow::{anyhow, Result};
use mlx_native::ops::dflash2_selector::{
    dispatch_dflash2_compose_fixed_verify_input, dispatch_dflash2_compose_verify_input,
    dispatch_dflash2_selector, dispatch_dflash2_verify_rows, DFlash2ComposeVerifyInputParams,
    DFlash2SelectorBuffers, DFlash2SelectorParams, DFlash2VerifyRowsParams, DFLASH2_MAX_LANES,
    DFLASH2_MAX_POSITIONS, DFLASH2_VERIFY_COSTS,
};
use mlx_native::{CommandEncoder, KernelRegistry, MlxDevice};

/// Encode top-16 candidate extraction, predecessor/successor edge scoring,
/// coherent-path selection, and either all-row or confidence-budgeted verify.
pub(crate) fn dispatch_dflash2_selection(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    config: &DFlash2Config,
    tensors: &DFlash2ModelTensors,
    workspace: &DFlash2Workspace,
    anchors: &[u32],
    temperatures: &[f32],
    sampling_mask: u32,
    maximum_verify_rows: &[u32],
    target_lane_order: &[u32],
    verify_model: Option<VerifyRowsModel>,
) -> Result<()> {
    let lanes = workspace.max_lanes;
    for (name, len) in [
        ("anchors", anchors.len()),
        ("temperatures", temperatures.len()),
        ("maximum verify rows", maximum_verify_rows.len()),
    ] {
        if len != lanes {
            return Err(anyhow!(
                "DFlash2 {name} count {len} != workspace lanes {lanes}"
            ));
        }
    }
    if target_lane_order.is_empty() || target_lane_order.len() > lanes {
        return Err(anyhow!(
            "DFlash2 target lane count {} must be in 1..={lanes}",
            target_lane_order.len()
        ));
    }
    let mut target_anchor = [0; DFLASH2_MAX_LANES];
    let mut target_physical_lane = [0; DFLASH2_MAX_LANES];
    let mut target_rows = [0; DFLASH2_MAX_LANES];
    for (logical, &physical_u32) in target_lane_order.iter().enumerate() {
        let physical = physical_u32 as usize;
        if physical >= lanes || target_lane_order[..logical].contains(&physical_u32) {
            return Err(anyhow!(
                "DFlash2 target logical lane {logical} names invalid physical lane {physical}"
            ));
        }
        target_anchor[logical] = anchors[physical];
        target_physical_lane[logical] = physical_u32;
        target_rows[logical] = maximum_verify_rows[physical];
    }
    let mut anchor = [0; DFLASH2_MAX_LANES];
    let mut temperature = [1.0; DFLASH2_MAX_LANES];
    let mut maximum = [0; DFLASH2_MAX_LANES];
    anchor[..lanes].copy_from_slice(anchors);
    temperature[..lanes].copy_from_slice(temperatures);
    maximum[..lanes].copy_from_slice(maximum_verify_rows);

    dispatch_dflash2_selector(
        encoder,
        registry,
        device,
        &DFlash2SelectorBuffers {
            logits: &workspace.logits,
            selector_hidden: &workspace.selector_hidden_bf16,
            predecessor_codebook: &tensors.predecessor_codebook,
            successor_codebook: &tensors.successor_codebook,
            uniforms: &workspace.uniforms,
            partial_ids: &workspace.partial_ids,
            partial_values: &workspace.partial_values,
            candidates: &workspace.candidates,
            unary: &workspace.unary,
            edges: &workspace.edges,
            tokens: &workspace.proposed_tokens,
            proposal_probabilities: &workspace.proposal_probabilities,
            confidence: &workspace.confidence,
        },
        DFlash2SelectorParams {
            anchor,
            temperature,
            lanes: lanes as u32,
            rows_per_lane: config.block_size as u32,
            vocabulary: config.vocab_size as u32,
            sampling_mask,
        },
    )?;
    encoder.memory_barrier();

    let mut cost = [1.0; DFLASH2_VERIFY_COSTS - 1];
    let mut cost_64 = 1.0;
    let mut intercept = [0.0; DFLASH2_MAX_POSITIONS];
    let mut slope = 0.0;
    let mut every_row = true;
    if let Some(model) = verify_model.as_ref() {
        slope = model.slope;
        intercept = model.intercept;
        cost.copy_from_slice(&model.costs[..DFLASH2_VERIFY_COSTS - 1]);
        cost_64 = model.costs[DFLASH2_VERIFY_COSTS - 1];
        every_row = model.every_row;
    }
    dispatch_dflash2_verify_rows(
        encoder,
        registry,
        device,
        &workspace.confidence,
        &workspace.verify_rows,
        DFlash2VerifyRowsParams {
            slope,
            intercept,
            cost,
            cost_64,
            physical_lane: target_physical_lane,
            maximum: target_rows,
            physical_lanes: lanes as u32,
            lanes: target_lane_order.len() as u32,
            rows_per_lane: config.block_size as u32,
            every_row: u32::from(every_row),
        },
    )?;
    encoder.memory_barrier();
    let mut compose = DFlash2ComposeVerifyInputParams {
        anchor: target_anchor,
        physical_lane: target_physical_lane,
        rows: target_rows,
        physical_lanes: lanes as u32,
        lanes: target_lane_order.len() as u32,
        rows_per_lane: config.block_size as u32,
        width: (workspace.max_lanes * workspace.rows_per_lane) as u32,
    };
    if verify_model.is_some() {
        // Qwen target verification is always the canonical width-32 route.
        // Fill its padding lanes even when the worker owns fewer physical
        // draft lanes.
        compose.width = workspace.target_tokens.element_count() as u32;
        dispatch_dflash2_compose_fixed_verify_input(
            encoder,
            registry,
            device,
            &workspace.proposed_tokens,
            &workspace.verify_rows,
            &workspace.target_tokens,
            compose,
        )?;
    } else {
        let target_tokens = workspace
            .target_tokens
            .slice_view(0, compose.width as usize);
        dispatch_dflash2_compose_verify_input(
            encoder,
            registry,
            device,
            &workspace.proposed_tokens,
            &target_tokens,
            compose,
        )?;
    }
    encoder.memory_barrier();
    Ok(())
}
