//! One consuming main-expert preparation transaction, before KV or GPU caches.
use std::sync::Arc;

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::{
    prepare_quantized_matmul_id_ggml_grouped_m8_batch, CompletedGroupedM8Experts,
    GgmlQuantizedMatmulIdParams, GgmlType, GroupedM8Preparation, KernelRegistry, MlxDevice,
};

use super::delta_net::DeltaNetLayerWeights;
use super::expert_storage::{
    CanonicalExpertStorageBytes, CompletedMoeExperts, MoeExpertStorage, RawExpertRole, RawMoeExperts,
};
use super::full_attn::FullAttnLayerWeights;
use super::gpu_delta_net::DeltaNetWeightsGpu;
use super::gpu_full_attn::FullAttnWeightsGpu;
use super::model::{Qwen35FfnWeights, Qwen35LayerWeights, Qwen35Model};
use super::weight_loader::{MoeFfnWeightsQ, MoeNativeDenseQ};

// This guard is outside the consuming frame. On failure/unwind, that frame's
// model/drafts drop first; only then does cleanup settle their actual queue.
struct PreparationQueue {
    device: MlxDevice,
    settled: bool,
}

impl PreparationQueue {
    fn settle(&mut self) -> Result<()> {
        self.settled = true;
        crate::inference::device_settle::settle_or_record(&self.device)
    }
}

impl Drop for PreparationQueue {
    fn drop(&mut self) {
        if !self.settled {
            let _ = self.settle();
        }
    }
}

fn eligible(weights: &MoeFfnWeightsQ) -> bool {
    matches!(weights.ggml_type_gate_up, GgmlType::Q4_K | GgmlType::Q6_K)
        && matches!(weights.ggml_type_down, GgmlType::Q4_K | GgmlType::Q6_K)
        && matches!(&*weights.experts, MoeExpertStorage::Raw(RawMoeExperts {
            gate: RawExpertRole::Ggml(_), up: RawExpertRole::Ggml(_), down: RawExpertRole::Ggml(_),
        }))
}

impl Qwen35Model {
    /// Finalize eligible main experts after overlays/policies and before any
    /// execution-cache, persistent-KV or drafter allocation. MTP is not visited.
    /// Retire overlay-replaced sources on their loading queue even when no
    /// expert triplet is eligible for preparation.
    pub(crate) fn finalize_canonical_experts(self) -> Result<Self> {
        let mut queue = self.weight_device.as_ref().map(|device| PreparationQueue {
            device: device.clone(),
            settled: false,
        });
        let model = self.prepare_canonical_experts(queue.as_mut())?;
        // Overlay replacements also stage source-residency removals when
        // no triplet is eligible. Publish neither path before checked retirement.
        if let Some(queue) = queue.as_mut() {
            queue.settle().context(
                "retire consumed or overlay-replaced weights from their source residency set",
            )?;
        }
        Ok(model)
    }

    fn prepare_canonical_experts(self, queue: Option<&mut PreparationQueue>) -> Result<Self> {
        if crate::quantize::imatrix::is_active() {
            return Ok(self);
        }
        let Some(moe) = self.cfg.moe.as_ref() else { return Ok(self) };
        if moe.num_experts == 0 || self.cfg.hidden_size == 0
            || self.cfg.hidden_size % 256 != 0 || moe.moe_intermediate_size == 0
            || moe.moe_intermediate_size % 256 != 0
        {
            return Ok(self);
        }
        let mut count = 0usize;
        for (index, layer) in self.layers.iter().enumerate() {
            if let Qwen35FfnWeights::MoeQ(weights) = layer.ffn() {
                if eligible(weights) {
                    ensure!(Arc::strong_count(&weights.experts) == 1,
                        "main expert layer {index} is already shared; finalize before cache construction");
                    count = count.checked_add(1).context("canonical layer count overflow")?;
                }
            }
        }
        if count == 0 { return Ok(self); }
        let queue = queue.context(
            "canonical main experts require their actual weight-loading device",
        )?;
        self.consume_canonical_experts(queue, count)
    }

    fn consume_canonical_experts(self, queue: &mut PreparationQueue, count: usize) -> Result<Self> {
        let moe = self.cfg.moe.as_ref().context("missing MoE configuration")?;
        let dimensions = (self.cfg.hidden_size, moe.moe_intermediate_size, moe.num_experts);
        let expected = count.checked_mul(3).context("canonical request count overflow")?;
        let mut requests = Vec::with_capacity(expected);
        let mut raw_bytes = 0u64;
        let drafts = self.layers.into_iter().map(|layer| {
            LayerDraft::take(layer, dimensions, &mut requests, &mut raw_bytes)
        }).collect::<Result<Vec<_>>>()?;
        ensure!(requests.len() == expected, "canonical request count changed during consumption");
        let mut registry = KernelRegistry::new();
        registry.freeze_ggml_routing_policy(self.ggml_routing_policy)?;
        let completed = prepare_quantized_matmul_id_ggml_grouped_m8_batch(
            &queue.device, &mut registry, requests,
        ).context("prepare the complete main-expert batch")?;
        ensure!(completed.len() == expected, "native preparation returned an incomplete batch");
        let prepared_bytes = completed.iter().try_fold(0u64, |sum, planes| {
            sum.checked_add(planes.storage_bytes() as u64)
                .context("canonical prepared byte count overflow")
        })?;
        let mut completed = completed.into_iter();
        let layers = drafts.into_iter().map(|draft| draft.finish(&mut completed))
            .collect::<Result<Vec<_>>>()?;
        ensure!(completed.next().is_none(), "native preparation returned unused expert planes");
        Ok(Self {
            layers,
            canonical_expert_storage_bytes: CanonicalExpertStorageBytes { raw_bytes, prepared_bytes },
            ..self
        })
    }
}

// A private transaction draft owns attention and non-routed weights without
// installing an empty/placeholder representation in any live model layer.
enum Attention {
    Full(FullAttnLayerWeights),
    Linear(DeltaNetLayerWeights),
    NativeFull(FullAttnWeightsGpu),
    NativeLinear(DeltaNetWeightsGpu),
}

struct LayerDraft {
    attention: Attention,
    ffn: FfnDraft,
}

enum FfnDraft {
    Ready(Qwen35FfnWeights),
    Preparing(MoeRemainder),
}

struct MoeRemainder {
    router: Vec<f32>,
    ggml_type_gate_up: GgmlType,
    ggml_type_down: GgmlType,
    shared_gate_logit: Vec<f32>,
    shared_gate: Vec<f32>,
    shared_up: Vec<f32>,
    shared_down: Vec<f32>,
    native_dense: Option<MoeNativeDenseQ>,
}

impl LayerDraft {
    fn take(layer: Qwen35LayerWeights, dimensions: (u32, u32, u32),
        requests: &mut Vec<GroupedM8Preparation>, raw_bytes: &mut u64) -> Result<Self> {
        let (attention, ffn) = match layer {
            Qwen35LayerWeights::FullAttn { attn, ffn } => (Attention::Full(attn), ffn),
            Qwen35LayerWeights::LinearAttn { attn, ffn } => (Attention::Linear(attn), ffn),
            Qwen35LayerWeights::NativeFullAttn { attn, ffn } => (Attention::NativeFull(attn), ffn),
            Qwen35LayerWeights::NativeLinearAttn { attn, ffn } => (Attention::NativeLinear(attn), ffn),
        };
        let ffn = match ffn {
            Qwen35FfnWeights::MoeQ(weights) if eligible(&weights) => {
                let MoeFfnWeightsQ { experts, router, ggml_type_gate_up, ggml_type_down,
                    shared_gate_logit, shared_gate, shared_up, shared_down, native_dense } = weights;
                let storage = Arc::try_unwrap(experts)
                    .map_err(|_| anyhow!("canonical expert ownership changed during consumption"))?;
                let MoeExpertStorage::Raw(RawMoeExperts { gate, up, down }) = storage else {
                    return Err(anyhow!("canonical expert storage changed during consumption"));
                };
                let (hidden, intermediate, experts) = dimensions;
                for (role, n, k, ggml_type) in [
                    (gate, intermediate, hidden, ggml_type_gate_up),
                    (up, intermediate, hidden, ggml_type_gate_up),
                    (down, hidden, intermediate, ggml_type_down),
                ] {
                    let RawExpertRole::Ggml(source) = role else {
                        return Err(anyhow!("canonical expert role changed during consumption"));
                    };
                    *raw_bytes = raw_bytes.checked_add(source.data_byte_len() as u64)
                        .context("canonical source byte count overflow")?;
                    let expert_stride = u64::from(n).checked_mul(u64::from(k / 256))
                        .and_then(|blocks| blocks.checked_mul(u64::from(ggml_type.block_bytes())))
                        .context("canonical expert stride overflow")?;
                    requests.push(GroupedM8Preparation { source, params: GgmlQuantizedMatmulIdParams {
                        n_tokens: 1, top_k: 1, n, k, n_experts: experts, expert_stride, ggml_type,
                    }});
                }
                FfnDraft::Preparing(MoeRemainder { router, ggml_type_gate_up, ggml_type_down,
                    shared_gate_logit, shared_gate, shared_up, shared_down, native_dense })
            }
            ready => FfnDraft::Ready(ready),
        };
        Ok(Self { attention, ffn })
    }

    fn finish(self, completed: &mut impl Iterator<Item = CompletedGroupedM8Experts>)
        -> Result<Qwen35LayerWeights> {
        let ffn = match self.ffn {
            FfnDraft::Ready(ffn) => ffn,
            FfnDraft::Preparing(MoeRemainder { router, ggml_type_gate_up, ggml_type_down,
                shared_gate_logit, shared_gate, shared_up, shared_down, native_dense }) => {
                let gate = completed.next().context("missing completed gate planes")?;
                let up = completed.next().context("missing completed up planes")?;
                let down = completed.next().context("missing completed down planes")?;
                Qwen35FfnWeights::MoeQ(MoeFfnWeightsQ {
                    router, experts: Arc::new(MoeExpertStorage::CompletedPrepared(
                        CompletedMoeExperts { gate, up, down })),
                    ggml_type_gate_up, ggml_type_down, shared_gate_logit,
                    shared_gate, shared_up, shared_down, native_dense,
                })
            }
        };
        Ok(match self.attention {
            Attention::Full(attn) => Qwen35LayerWeights::FullAttn { attn, ffn },
            Attention::Linear(attn) => Qwen35LayerWeights::LinearAttn { attn, ffn },
            Attention::NativeFull(attn) => Qwen35LayerWeights::NativeFullAttn { attn, ffn },
            Attention::NativeLinear(attn) => Qwen35LayerWeights::NativeLinearAttn { attn, ffn },
        })
    }
}
