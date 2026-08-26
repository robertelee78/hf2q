use std::ops::Range;

use anyhow::{ensure, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::{DType, GgmlType, KernelRegistry, MlxBuffer, MlxDevice};

use super::rectangular_prefill::RectangularPrefillShape;

/// One contiguous element range inside a rectangular activation buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ElementSpan {
    pub(crate) offset: usize,
    pub(crate) elements: usize,
}

impl ElementSpan {
    fn range(self) -> Result<Range<usize>> {
        let end = self
            .offset
            .checked_add(self.elements)
            .context("rectangular FFN element-span overflow")?;
        Ok(self.offset..end)
    }
}

/// Exact lane-local buffer geometry for Gemma's rectangular FFN/MoE body.
///
/// The aggregate buffers remain row-major. Each span identifies one lane's
/// existing storage without copying or changing the artifact representation.
/// `scratch_reuse_barrier` is false only for lane zero: every later expert
/// dispatch reuses the same mutable mm-id routing scratch and therefore must
/// be preceded by an explicit Metal buffer barrier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RectangularFfnLanePlan {
    pub(crate) lane: usize,
    pub(crate) source_rows: u32,
    pub(crate) top_k: u32,
    pub(crate) input: ElementSpan,
    pub(crate) routing: ElementSpan,
    pub(crate) expert_gate_up: ElementSpan,
    pub(crate) scratch_reuse_barrier: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RectangularFfnPlan {
    pub(crate) shape: RectangularPrefillShape,
    pub(crate) lanes: Vec<RectangularFfnLanePlan>,
}

fn lane_span(shape: RectangularPrefillShape, lane: usize, columns: usize) -> Result<ElementSpan> {
    ensure!(
        lane < shape.lanes,
        "rectangular FFN lane {lane} is out of range"
    );
    ensure!(columns > 0, "rectangular FFN buffer has zero columns");
    let elements = shape
        .rows_per_lane
        .checked_mul(columns)
        .context("rectangular FFN lane extent overflow")?;
    let offset = lane
        .checked_mul(elements)
        .context("rectangular FFN lane offset overflow")?;
    Ok(ElementSpan { offset, elements })
}

/// Build the only body shape admitted by the current deciding spike:
/// two or four lanes, each retaining the canonical scalar prefill width 32.
pub(crate) fn plan_rectangular_ffn(
    shape: RectangularPrefillShape,
    hidden: usize,
    moe_intermediate: usize,
    top_k: usize,
) -> Result<RectangularFfnPlan> {
    ensure!(
        matches!(shape.lanes, 2 | 4),
        "rectangular Gemma FFN requires two or four lanes, got {}",
        shape.lanes
    );
    ensure!(
        shape.rows_per_lane == 32,
        "rectangular Gemma FFN requires scalar width 32, got {}",
        shape.rows_per_lane
    );
    ensure!(
        hidden > 0 && moe_intermediate > 0 && top_k > 0,
        "rectangular Gemma FFN dimensions must be nonzero"
    );

    let routed_gate_up_columns = top_k
        .checked_mul(2)
        .and_then(|value| value.checked_mul(moe_intermediate))
        .context("rectangular expert gate/up width overflow")?;
    let lanes = (0..shape.lanes)
        .map(|lane| {
            Ok(RectangularFfnLanePlan {
                lane,
                source_rows: u32::try_from(shape.rows_per_lane)
                    .context("rectangular FFN row count does not fit u32")?,
                top_k: u32::try_from(top_k).context("rectangular FFN top-k does not fit u32")?,
                input: lane_span(shape, lane, hidden)?,
                routing: lane_span(shape, lane, top_k)?,
                expert_gate_up: lane_span(shape, lane, routed_gate_up_columns)?,
                scratch_reuse_barrier: lane > 0,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RectangularFfnPlan { shape, lanes })
}

/// Produce a zero-copy typed view for a validated lane span.
pub(crate) fn checked_span_view(
    buffer: &MlxBuffer,
    span: ElementSpan,
    dtype: DType,
    operation: &str,
) -> Result<MlxBuffer> {
    ensure!(
        buffer.dtype() == dtype,
        "{operation}: expected {dtype:?}, got {:?}",
        buffer.dtype()
    );
    let range = span.range()?;
    ensure!(
        range.end <= buffer.element_count(),
        "{operation}: lane range {:?} exceeds {} elements",
        range,
        buffer.element_count()
    );
    let byte_offset = span
        .offset
        .checked_mul(dtype.size_of())
        .context("rectangular FFN lane byte offset overflow")?;
    let byte_end = range
        .end
        .checked_mul(dtype.size_of())
        .context("rectangular FFN lane byte extent overflow")?;
    ensure!(
        byte_end <= buffer.data_byte_len(),
        "{operation}: lane byte extent {byte_end} exceeds {} bytes",
        buffer.data_byte_len()
    );
    Ok(buffer.slice_view(
        u64::try_from(byte_offset).context("rectangular FFN byte offset does not fit u64")?,
        span.elements,
    ))
}

/// Dispatch the native/affine expert gate-up route once per physical lane.
/// Returns `false` only when the artifact uses a block codec that must take
/// the GGUF pooled route below.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_rectangular_native_gate_up(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    activation_epoch: u64,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    affine: Option<&crate::serve::forward_mlx_shared::MlxAffineMoeStack>,
    ggml_type: GgmlType,
    plan: &RectangularFfnPlan,
    output_width: u32,
    input_width: u32,
    n_experts: u32,
    expert_stride: u64,
) -> Result<bool> {
    let mut shared_route = None;
    for lane in &plan.lanes {
        if lane.scratch_reuse_barrier {
            session.encoder_mut().memory_barrier();
        }
        let lane_input = checked_span_view(
            input,
            lane.input,
            DType::F32,
            "rectangular expert gate/up input",
        )?;
        let lane_ids = checked_span_view(
            expert_ids,
            lane.routing,
            DType::U32,
            "rectangular expert gate/up IDs",
        )?;
        let lane_output = checked_span_view(
            output,
            lane.expert_gate_up,
            DType::F32,
            "rectangular expert gate/up output",
        )?;
        let label = format!("Gemma rectangular-prefill gate/up lane {}", lane.lane);
        let dispatched = super::expert_dispatch::dispatch_native_scalar_expert(
            session,
            registry,
            device,
            activation_epoch,
            &lane_input,
            weight,
            &lane_ids,
            &lane_output,
            affine,
            ggml_type,
            lane.source_rows,
            lane.top_k,
            output_width,
            input_width,
            n_experts,
            expert_stride,
            mlx_native::DenseMatmulIdInputLayout::SharedPerToken,
            super::expert_dispatch::DenseExpertScratchSlot::GateUp,
            &label,
        )?;
        if let Some(expected) = shared_route {
            ensure!(
                dispatched == expected,
                "rectangular expert gate/up selected inconsistent native routes across lanes"
            );
        } else {
            shared_route = Some(dispatched);
        }
    }
    Ok(shared_route.unwrap_or(false))
}

/// Dispatch the block-quantized expert gate-up route once per physical lane,
/// retaining the scalar M=32 mm-id plan and explicitly fencing shared routing
/// scratch before every reuse.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_rectangular_ggml_gate_up(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &MlxBuffer,
    expert_ids: &MlxBuffer,
    output: &MlxBuffer,
    scratch: &mut mlx_native::ops::quantized_matmul_id_ggml::IdMmScratch,
    plan: &RectangularFfnPlan,
    params: mlx_native::GgmlQuantizedMatmulIdParams,
) -> Result<()> {
    for lane in &plan.lanes {
        if lane.scratch_reuse_barrier {
            session.encoder_mut().memory_barrier();
        }
        let lane_input = checked_span_view(
            input,
            lane.input,
            DType::F32,
            "rectangular GGUF gate/up input",
        )?;
        let lane_ids = checked_span_view(
            expert_ids,
            lane.routing,
            DType::U32,
            "rectangular GGUF gate/up IDs",
        )?;
        let lane_output = checked_span_view(
            output,
            lane.expert_gate_up,
            DType::F32,
            "rectangular GGUF gate/up output",
        )?;
        session.barrier_between(&[&lane_input, &lane_ids, weight], &[&lane_output]);
        session
            .quantized_matmul_id_ggml_pooled(
                registry,
                device,
                &lane_input,
                weight,
                &lane_ids,
                &lane_output,
                scratch,
                &mlx_native::GgmlQuantizedMatmulIdParams {
                    n_tokens: lane.source_rows,
                    ..params
                },
            )
            .with_context(|| {
                format!(
                    "rectangular batched gate_up_id lane {} with m={}",
                    lane.lane, lane.source_rows
                )
            })?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "rectangular_ffn_tests.rs"]
mod tests;
