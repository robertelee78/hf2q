//! Test-only preparation through the native checked consuming batch boundary.
use std::sync::Arc;
use mlx_native::ops::quantized_matmul_id_ggml::{
    prepare_quantized_matmul_id_ggml_grouped_m8_batch, GgmlQuantizedMatmulIdParams,
    GroupedM8Preparation,
};
use mlx_native::{KernelRegistry, MlxDevice};
use super::expert_storage::{CompletedMoeExperts, MoeExpertStorage, RawExpertRole};
use super::ffn::MoeFfnShape;
use super::gpu_ffn::MoeFfnWeightsGpuQ;

pub(super) fn complete_experts(
    mut weights: MoeFfnWeightsGpuQ,
    shape: MoeFfnShape,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
) -> MoeFfnWeightsGpuQ {
    let MoeExpertStorage::Raw(raw) = Arc::try_unwrap(weights.experts)
        .unwrap_or_else(|_| panic!("test preparation requires exclusive expert storage"))
    else { panic!("test preparation requires raw expert storage") };
    let source = |role| match role {
        RawExpertRole::Ggml(buffer) => buffer,
        RawExpertRole::Affine(_) => panic!("test preparation requires GGML expert storage"),
    };
    let gate = GgmlQuantizedMatmulIdParams {
        n_tokens: 1, top_k: 1, n: shape.moe_intermediate_size, k: shape.hidden_size,
        n_experts: shape.num_experts, expert_stride: weights.expert_gate_stride,
        ggml_type: weights.ggml_type_gate_up,
    };
    let requests = vec![
        GroupedM8Preparation { source: source(raw.gate), params: gate },
        GroupedM8Preparation {
            source: source(raw.up),
            params: GgmlQuantizedMatmulIdParams { expert_stride: weights.expert_up_stride, ..gate },
        },
        GroupedM8Preparation {
            source: source(raw.down),
            params: GgmlQuantizedMatmulIdParams {
                n: shape.hidden_size, k: shape.moe_intermediate_size,
                expert_stride: weights.expert_down_stride, ggml_type: weights.ggml_type_down,
                ..gate
            },
        },
    ];
    let mut completed = prepare_quantized_matmul_id_ggml_grouped_m8_batch(device, registry, requests)
        .expect("complete native expert preparation");
    assert_eq!(completed.len(), 3);
    let down = completed.pop().unwrap();
    let up = completed.pop().unwrap();
    let gate = completed.pop().unwrap();
    weights.experts = Arc::new(MoeExpertStorage::CompletedPrepared(CompletedMoeExperts { gate, up, down }));
    weights
}
