//! Valid fixed encoded test tensors, not model weights or a CPU attention oracle.
use super::*;
use crate::inference::models::deepseek4::{
    attention_weights::AttentionWeightsRef, residency::RawMatrixRef, Deepseek4Config,
};
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
pub(super) const GUARD: usize = 256;
pub(super) const HOSTILE: u32 = 0x7fc0cafe;
pub(super) fn cfg(ratio: u32) -> Deepseek4Config {
    Deepseek4Config {
        num_hidden_layers: 1,
        hidden_size: 4096,
        hidden_size_out: 16384,
        max_position_embeddings: 1048576,
        vocab_size: 129280,
        num_attention_heads: 64,
        num_key_value_heads: 1,
        head_dim: 512,
        rope_head_dim: 64,
        rope_theta: 10000.,
        rope_factor: 16.,
        original_context_length: 65536,
        yarn_beta_fast: 32.,
        yarn_beta_slow: 1.,
        q_lora_rank: 1024,
        o_lora_rank: 1024,
        output_groups: 8,
        sliding_window: 128,
        compress_ratios: vec![ratio],
        compress_rope_theta: 160000.,
        index_num_heads: 64,
        index_head_dim: 128,
        index_top_k: 512,
        rms_norm_eps: 1e-6,
        num_experts: 256,
        num_experts_per_tok: 6,
        num_shared_experts: 1,
        expert_intermediate_size: 2048,
        route_scale: 1.5,
        normalize_topk: true,
        swiglu_clamp_experts: vec![10.],
        swiglu_clamp_shared: vec![10.],
        hyper_connection_count: 4,
        hyper_connection_sinkhorn_iterations: 20,
        hyper_connection_epsilon: 1e-6,
        hash_layer_count: 1,
    }
}
pub(super) struct Weights {
    hc: MlxBuffer,
    base: MlxBuffer,
    scale: MlxBuffer,
    norm: MlxBuffer,
    qa: MlxBuffer,
    qan: MlxBuffer,
    qb: MlxBuffer,
    kv: MlxBuffer,
    kvn: MlxBuffer,
    sinks: MlxBuffer,
    oa: MlxBuffer,
    ob: MlxBuffer,
}
fn f32s(d: &MlxDevice, n: usize, value: f32) -> MlxBuffer {
    let mut b = alloc(d, DType::F32, vec![n], "workspace fixture F32").unwrap();
    b.as_logical_mut_slice::<f32>().unwrap().fill(value);
    b
}
fn q8(d: &MlxDevice, n: usize, k: usize, seed: usize) -> MlxBuffer {
    let bytes = n * k / 32 * 34;
    let mut b = alloc(d, DType::U8, vec![bytes], "valid fixed Q8 fixture").unwrap();
    let scale = half::f16::from_f32(1. / 1024.).to_bits().to_le_bytes();
    for (j, block) in b
        .as_logical_mut_slice::<u8>()
        .unwrap()
        .chunks_exact_mut(34)
        .enumerate()
    {
        block[..2].copy_from_slice(&scale);
        for (i, v) in block[2..].iter_mut().enumerate() {
            *v = (((i + j + seed) % 9) as i8 - 4) as u8;
        }
    }
    b
}
impl Weights {
    pub fn new(d: &MlxDevice) -> Self {
        Self {
            hc: q8(d, 24, 16384, 0),
            base: f32s(d, 24, 0.1),
            scale: f32s(d, 3, 0.01),
            norm: f32s(d, 4096, 1.),
            qa: q8(d, 1024, 4096, 1),
            qan: f32s(d, 1024, 1.),
            qb: q8(d, 32768, 1024, 2),
            kv: q8(d, 512, 4096, 3),
            kvn: f32s(d, 512, 1.),
            sinks: f32s(d, 64, 0.),
            oa: q8(d, 8192, 4096, 4),
            ob: q8(d, 4096, 8192, 5),
        }
    }
    pub fn refs(&self) -> AttentionWeightsRef<'_> {
        AttentionWeightsRef {
            hc_fn: RawMatrixRef {
                buffer: &self.hc,
                ggml_type: GgmlType::Q8_0,
                shape: &[24, 16384],
            },
            hc_base: &self.base,
            hc_scale: &self.scale,
            attn_norm: &self.norm,
            q_a: RawMatrixRef {
                buffer: &self.qa,
                ggml_type: GgmlType::Q8_0,
                shape: &[1024, 4096],
            },
            q_a_norm: &self.qan,
            q_b: RawMatrixRef {
                buffer: &self.qb,
                ggml_type: GgmlType::Q8_0,
                shape: &[32768, 1024],
            },
            kv: RawMatrixRef {
                buffer: &self.kv,
                ggml_type: GgmlType::Q8_0,
                shape: &[512, 4096],
            },
            kv_norm: &self.kvn,
            sinks: &self.sinks,
            output_a: RawMatrixRef {
                buffer: &self.oa,
                ggml_type: GgmlType::Q8_0,
                shape: &[8192, 4096],
            },
            output_b: RawMatrixRef {
                buffer: &self.ob,
                ggml_type: GgmlType::Q8_0,
                shape: &[4096, 8192],
            },
        }
    }
    pub fn bytes(&self) -> usize {
        [
            &self.hc,
            &self.base,
            &self.scale,
            &self.norm,
            &self.qa,
            &self.qan,
            &self.qb,
            &self.kv,
            &self.kvn,
            &self.sinks,
            &self.oa,
            &self.ob,
        ]
        .iter()
        .map(|b| b.byte_len())
        .sum()
    }
}
pub(super) fn guarded(d: &MlxDevice, n: usize, parents: &mut Vec<MlxBuffer>) -> MlxBuffer {
    let mut b = alloc(d, DType::F32, vec![n + 2 * GUARD], "workspace guards").unwrap();
    b.as_logical_mut_slice::<u32>().unwrap().fill(HOSTILE);
    let view = b.slice_view((GUARD * 4) as u64, n);
    parents.push(b);
    view
}
pub(super) fn workspace(
    d: &MlxDevice,
    p: Plan,
    shared: bool,
    parents: &mut Vec<MlxBuffer>,
) -> AttentionWorkspace {
    if shared {
        AttentionWorkspace::from_root(guarded(d, p.root_elements, parents), p).unwrap()
    } else {
        AttentionWorkspace {
            state_norm: guarded(d, p.norm_elements, parents)
                .with_shape(vec![p.rows, p.hc_hidden])
                .unwrap(),
            q: guarded(d, p.query_elements, parents)
                .with_shape(vec![p.rows, p.heads, p.dim])
                .unwrap(),
            attention_f32: guarded(d, p.query_elements, parents)
                .with_shape(vec![p.rows, p.heads, p.dim])
                .unwrap(),
        }
    }
}
pub(super) fn guards(parents: &[MlxBuffer]) {
    for b in parents {
        let v = b.as_logical_slice::<u32>().unwrap();
        assert!(
            v[..GUARD]
                .iter()
                .chain(&v[v.len() - GUARD..])
                .all(|&v| v == HOSTILE),
            "workspace guard overwritten"
        );
    }
}
