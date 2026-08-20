use anyhow::{Context, Result};

use super::{
    context_from_config, expected_sources, qwen35_config_from_authenticated_source,
    topology_config, Qwen35FutureDType, Qwen35SourceTransformV1,
};

pub(in crate::inference::models::qwen35::source_precision) fn expected_profile_for_config_for_test(
    source_config: &serde_json::Value,
) -> Result<(usize, usize, usize, u64, u64, u64, [usize; 7])> {
    let projected = qwen35_config_from_authenticated_source(source_config)?;
    let mapper = context_from_config(source_config).context("mapper context")?;
    let config = topology_config(&projected, mapper.multimodal_wrapping)?;
    let expected = expected_sources(&config)?;
    let mut bf16 = 0_usize;
    let mut f32 = 0_usize;
    let mut bf16_bytes = 0_u64;
    let mut f32_bytes = 0_u64;
    let mut max_bytes = 0_u64;
    let mut transforms = [0_usize; 7];
    for source in expected.values() {
        for output in &source.outputs {
            match output.dtype {
                Qwen35FutureDType::Bf16 => bf16 += 1,
                Qwen35FutureDType::F32 => f32 += 1,
            }
            let elements = output.shape.iter().try_fold(1_u64, |product, dimension| {
                product
                    .checked_mul(u64::try_from(*dimension)?)
                    .context("test profile element count overflow")
            })?;
            let bytes = elements
                .checked_mul(match output.dtype {
                    Qwen35FutureDType::Bf16 => 2,
                    Qwen35FutureDType::F32 => 4,
                })
                .context("test profile byte count overflow")?;
            match output.dtype {
                Qwen35FutureDType::Bf16 => bf16_bytes += bytes,
                Qwen35FutureDType::F32 => f32_bytes += bytes,
            }
            max_bytes = max_bytes.max(bytes);
            transforms[match output.transform {
                Qwen35SourceTransformV1::Identity => 0,
                Qwen35SourceTransformV1::AddOneF32 => 1,
                Qwen35SourceTransformV1::ReorderVHeads { .. } => 2,
                Qwen35SourceTransformV1::ReorderVHeadsThenNegExpF32 { .. } => 3,
                Qwen35SourceTransformV1::SqueezeAxis1ThenReorderVSlice { .. } => 4,
                Qwen35SourceTransformV1::ReorderVHeadsPerRow { .. } => 5,
                Qwen35SourceTransformV1::SplitInterleavedQGate { .. } => 6,
            }] += 1;
        }
    }
    Ok((
        expected.len(),
        bf16,
        f32,
        bf16_bytes,
        f32_bytes,
        max_bytes,
        transforms,
    ))
}
