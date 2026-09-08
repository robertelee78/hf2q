//! Native projection of head-major attention activations.

use super::forward_mlx_shared::{dispatch_qmatmul, HeadMajorQmatmulRoute, MlxQWeight};
use crate::quantize::imatrix::ImatrixHint;
use anyhow::Result;
use mlx_native::{
    ggml_capability, GgmlCapabilityRequest, GgmlInvocation, GgmlRoutingPolicy, GgmlWorkloadClass,
    GraphSession, MlxBuffer, MlxDevice, GGML_CAPABILITY_SCHEMA_VERSION,
};

pub fn supports_native_perm021(weight: &MlxQWeight, m: u32, head_dim: u32) -> bool {
    if weight.affine.is_some() {
        return false;
    }
    let Ok(n) = u32::try_from(weight.info.rows) else {
        return false;
    };
    let Ok(k) = u32::try_from(weight.info.cols) else {
        return false;
    };
    ggml_capability(GgmlCapabilityRequest {
        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
        invocation: GgmlInvocation::DensePerm021Bf16 { m, n, k, head_dim },
        ggml_type: weight.info.ggml_dtype,
        workload: GgmlWorkloadClass::Prompt,
        routing: GgmlRoutingPolicy::default(),
    })
    .executable
}

/// Project a head-major BF16 activation through an artifact-native matrix.
///
/// Codecs with a declared direct kernel read `[heads, tokens, head_dim]`
/// without an intermediate. Every other admitted codec permutes only the
/// activation into the caller-provided F32 scratch and then uses the ordinary
/// native stored-weight route. This helper never transforms or shadows the
/// weight.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_qmatmul_head_major_bf16(
    session: &mut GraphSession<'_>,
    registry: &mut mlx_native::KernelRegistry,
    device: &MlxDevice,
    input_head_major: &MlxBuffer,
    seq_major_scratch: &MlxBuffer,
    weight: &MlxQWeight,
    output: &MlxBuffer,
    m: u32,
    n_heads: usize,
    head_dim: usize,
    imatrix_hint: ImatrixHint<'_>,
) -> Result<HeadMajorQmatmulRoute> {
    anyhow::ensure!(m > 0, "head-major projection token count must be positive");
    anyhow::ensure!(
        n_heads > 0 && head_dim > 0,
        "head-major projection dimensions must be positive"
    );
    let hidden = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow::anyhow!("head-major projection hidden width overflow"))?;
    anyhow::ensure!(
        hidden == weight.info.cols,
        "head-major projection width {hidden} does not match stored weight width {}",
        weight.info.cols
    );
    anyhow::ensure!(
        input_head_major.dtype() == mlx_native::DType::BF16,
        "head-major projection input must be BF16, got {:?}",
        input_head_major.dtype()
    );
    anyhow::ensure!(
        seq_major_scratch.dtype() == mlx_native::DType::F32,
        "head-major projection scratch must be F32, got {:?}",
        seq_major_scratch.dtype()
    );
    anyhow::ensure!(
        output.dtype() == mlx_native::DType::F32,
        "head-major projection output must be F32, got {:?}",
        output.dtype()
    );
    let head_dim_u32 = u32::try_from(head_dim)
        .map_err(|_| anyhow::anyhow!("head-major projection head_dim exceeds u32"))?;
    let n = u32::try_from(weight.info.rows)
        .map_err(|_| anyhow::anyhow!("head-major projection rows exceed u32"))?;
    let k = u32::try_from(weight.info.cols)
        .map_err(|_| anyhow::anyhow!("head-major projection cols exceed u32"))?;

    if supports_native_perm021(weight, m, head_dim_u32) {
        let params = mlx_native::GgmlQuantizedMatmulPerm021Params {
            m,
            n,
            k,
            head_dim: head_dim_u32,
            ggml_type: weight.info.ggml_dtype,
        };
        // `barrier_between` both resolves prior hazards and registers this
        // dispatch with the graph conflict tracker.
        session.barrier_between(&[input_head_major, &weight.buffer], &[output]);
        mlx_native::quantized_matmul_mm_tensor_perm021(
            session.encoder_mut(),
            registry,
            device,
            input_head_major,
            &weight.buffer,
            output,
            &params,
        )
        .map_err(|error| anyhow::anyhow!("native head-major projection failed: {error}"))?;
        return Ok(HeadMajorQmatmulRoute::DirectPerm021);
    }

    let m_usize = usize::try_from(m)
        .map_err(|_| anyhow::anyhow!("head-major projection token count exceeds usize"))?;
    session.barrier_between(&[input_head_major], &[seq_major_scratch]);
    mlx_native::ops::transpose::permute_021_bf16_to_f32(
        session.encoder_mut(),
        registry,
        device.metal_device(),
        input_head_major,
        seq_major_scratch,
        n_heads,
        m_usize,
        head_dim,
    )
    .map_err(|error| anyhow::anyhow!("head-major activation permute failed: {error}"))?;
    session.barrier_between(&[seq_major_scratch, &weight.buffer], &[output]);
    dispatch_qmatmul(
        session,
        registry,
        device,
        seq_major_scratch,
        weight,
        output,
        m,
        imatrix_hint,
    )?;
    Ok(HeadMajorQmatmulRoute::ActivationPermute)
}
