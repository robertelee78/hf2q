//! Coherent DFlash2 proposal selection over the reusable decode workspace.

use super::{config::DFlash2Config, tensors::DFlash2ModelTensors, workspace::DFlash2Workspace};
use anyhow::{anyhow, Result};
use mlx_native::ops::dflash2_selector::{
    dispatch_dflash2_selector, dispatch_dflash2_verify_rows, DFlash2SelectorBuffers,
    DFlash2SelectorParams, DFlash2VerifyRowsParams, DFLASH2_MAX_LANES, DFLASH2_MAX_POSITIONS,
    DFLASH2_VERIFY_COSTS,
};
use mlx_native::{CommandEncoder, KernelRegistry, MlxDevice};

/// Encode top-16 candidate extraction, predecessor/successor edge scoring,
/// coherent-path selection, and either all-row or confidence-budgeted verify.
pub fn dispatch_dflash2_selection(
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

    // The draft graph emits the correctness-first all-row plan. MoE serving
    // replaces it from the same confidence readback with its online host
    // budget before target verification; dense targets retain this plan.
    let cost = [1.0; DFLASH2_VERIFY_COSTS - 1];
    dispatch_dflash2_verify_rows(
        encoder,
        registry,
        device,
        &workspace.confidence,
        &workspace.verify_rows,
        DFlash2VerifyRowsParams {
            slope: 0.0,
            intercept: [0.0; DFLASH2_MAX_POSITIONS],
            cost,
            cost_64: 1.0,
            lanes: lanes as u32,
            rows_per_lane: config.block_size as u32,
            every_row: 1,
            maximum,
        },
    )?;
    encoder.memory_barrier();
    Ok(())
}
