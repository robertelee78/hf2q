//! Complete resource reads for selected DeepSeek RMS and indexer RoPE nodes.
//! Helpers annotate and immediately dispatch; numerical arguments stay unchanged.
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_f32_to_bf16, DeepSeekTailRopeParams,
};
use mlx_native::{KernelRegistry, MlxBuffer, MlxDevice};

#[allow(clippy::too_many_arguments)]
pub(super) fn rms_norm(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &mlx_native::metal::DeviceRef,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    output: &MlxBuffer,
    params: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> mlx_native::Result<()> {
    session.barrier_between(&[input, weight, params], &[output]);
    crate::inference::native_observation::witness_rms_norm(
        session,input,rows as usize,dim as usize,None,|session|
            session.rms_norm(registry,device,input,weight,output,params,rows,dim))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rms_norm_no_scale_f32(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &mlx_native::metal::DeviceRef,
    input: &MlxBuffer,
    output: &MlxBuffer,
    params: &MlxBuffer,
    rows: u32,
    dim: u32,
) -> mlx_native::Result<()> {
    session.barrier_between(&[input, params], &[output]);
    crate::inference::native_observation::witness_rms_norm(
        session,input,rows as usize,dim as usize,None,|session|
            session.rms_norm_no_scale_f32(registry,device,input,output,params,rows,dim))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn indexer_query_rope(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    positions: &MlxBuffer,
    frequencies: &MlxBuffer,
    output: &MlxBuffer,
    params: &DeepSeekTailRopeParams,
) -> mlx_native::Result<()> {
    session.barrier_between(&[input, positions, frequencies], &[output]);
    dispatch_deepseek_tail_rope_f32_to_bf16(
        session.encoder_mut(),
        registry,
        device,
        input,
        positions,
        frequencies,
        output,
        params,
    )
}

#[cfg(test)]
#[path = "prefill_annotations_tests.rs"]
mod tests;
