//! Explicit target-family adapters for DFlash2 embedding and vocabulary heads.

use super::{
    config::{DFlash2Config, DFlash2TargetShape},
    forward::DFlash2VocabularyHead,
    tensors::DFlash2ModelTensors,
    workspace::DFlash2Workspace,
};
use crate::{
    inference::models::{
        gemma4::model::MlxModelWeights,
        qwen35::{model::Qwen35Model, DFlash2JointCapture},
    },
    serve::forward_mlx_shared::MlxQWeight,
};
use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::ops::dflash2_attention_pack::{
    dispatch_dflash2_mask_rows_f32, DFlash2MaskRowsParams,
};
use mlx_native::{CommandEncoder, GgmlType, KernelRegistry, MlxDevice};

pub trait DFlash2TargetFamily {
    fn family_name(&self) -> &'static str;
    fn target_shape(&self) -> DFlash2TargetShape<'_>;
    fn embedding(&self) -> Result<&MlxQWeight>;
    fn vocabulary_head(&self) -> Result<&MlxQWeight>;
    fn embedding_scale(&self) -> f32;
    /// Whether target verification should budget rows from selector confidence.
    ///
    /// Dense targets keep every row because reducing their packed width did
    /// not qualify. MoE targets benefit because fewer rows avoid expert work.
    fn adaptive_verify_budget(&self) -> bool {
        false
    }

    fn validate(&self, config: &DFlash2Config) -> Result<()> {
        config
            .validate_target(self.target_shape())
            .map_err(anyhow::Error::from)
    }
}

pub struct QwenDFlash2Adapter<'a> {
    model: &'a Qwen35Model,
    config: &'a DFlash2Config,
}

impl<'a> QwenDFlash2Adapter<'a> {
    pub fn new(model: &'a Qwen35Model, config: &'a DFlash2Config) -> Result<Self> {
        let adapter = Self { model, config };
        adapter.validate(config)?;
        adapter.embedding()?;
        adapter.vocabulary_head()?;
        Ok(adapter)
    }

    pub fn target_capture<'b>(
        &'b self,
        workspace: &'b DFlash2Workspace,
        lane_indices: &'b [usize],
    ) -> Result<DFlash2JointCapture<'b>> {
        ensure!(
            workspace.rows_per_lane == self.config.block_size
                && workspace.max_lanes <= 4
                && self.config.fc_input_dim()
                    == self.config.target_layer_ids.len() * self.model.cfg.hidden_size as usize,
            "Qwen DFlash2 target capture geometry does not match its workspace"
        );
        Ok(DFlash2JointCapture {
            layer_ids: &self.config.target_layer_ids,
            output: &workspace.captured_target_hidden,
            lane_indices,
            lanes: workspace.max_lanes,
            rows_per_lane: workspace.rows_per_lane,
            feature_width: self.config.fc_input_dim(),
        })
    }
}

impl DFlash2TargetFamily for QwenDFlash2Adapter<'_> {
    fn family_name(&self) -> &'static str {
        "qwen35"
    }

    fn target_shape(&self) -> DFlash2TargetShape<'_> {
        DFlash2TargetShape {
            architecture: self.family_name(),
            hidden_size: self.model.cfg.hidden_size as usize,
            vocab_size: self.model.cfg.vocab_size as usize,
            num_hidden_layers: self.model.cfg.num_hidden_layers as usize,
        }
    }

    fn embedding(&self) -> Result<&MlxQWeight> {
        self.model
            .token_embd_native
            .as_ref()
            .context("Qwen DFlash2 requires native token_embd.weight")
    }

    fn vocabulary_head(&self) -> Result<&MlxQWeight> {
        self.model
            .resolved_native_output_head()?
            .map(|(weight, _)| weight)
            .context("Qwen DFlash2 requires a native output head")
    }

    fn embedding_scale(&self) -> f32 {
        1.0
    }

    fn adaptive_verify_budget(&self) -> bool {
        self.model.cfg.moe.is_some()
    }
}

pub struct GemmaDFlash2Adapter<'a> {
    model: &'a MlxModelWeights,
}

impl<'a> GemmaDFlash2Adapter<'a> {
    pub fn new(model: &'a MlxModelWeights, config: &DFlash2Config) -> Result<Self> {
        let adapter = Self { model };
        adapter.validate(config)?;
        adapter.embedding()?;
        adapter.vocabulary_head()?;
        Ok(adapter)
    }
}

impl DFlash2TargetFamily for GemmaDFlash2Adapter<'_> {
    fn family_name(&self) -> &'static str {
        "gemma4"
    }

    fn target_shape(&self) -> DFlash2TargetShape<'_> {
        DFlash2TargetShape {
            architecture: self.family_name(),
            hidden_size: self.model.hidden_size,
            vocab_size: self.model.vocab_size,
            num_hidden_layers: self.model.layers.len(),
        }
    }

    fn embedding(&self) -> Result<&MlxQWeight> {
        Ok(&self.model.embed_weight)
    }

    fn vocabulary_head(&self) -> Result<&MlxQWeight> {
        Ok(self.model.resolved_lm_head())
    }

    fn embedding_scale(&self) -> f32 {
        (self.model.hidden_size as f32).sqrt()
    }
}

fn dispatch_embedding_rows(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    weight: &MlxQWeight,
    token_ids: &mlx_native::MlxBuffer,
    output: &mlx_native::MlxBuffer,
    rows: usize,
) -> Result<()> {
    ensure!(
        weight.affine.is_none(),
        "DFlash2 embedding cannot use affine metadata"
    );
    let vocab_size = weight.info.rows;
    let embed_dim = weight.info.cols;
    match weight.info.ggml_dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => mlx_native::embedding_gather_dense(
            encoder,
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::EmbeddingDenseParams {
                vocab_size,
                embed_dim,
                n_tokens: rows,
            },
        )?,
        GgmlType::Q4_0 => mlx_native::embedding_gather_q4_0(
            encoder,
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::EmbeddingQ4_0Params {
                vocab_size,
                embed_dim,
                n_tokens: rows,
            },
        )?,
        GgmlType::Q5_0 => mlx_native::embedding_gather_q5_0(
            encoder,
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::EmbeddingQ5_0Params {
                vocab_size,
                embed_dim,
                n_tokens: rows,
            },
        )?,
        GgmlType::Q2_K => mlx_native::embedding_gather_q2_k(
            encoder,
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::EmbeddingQ2KParams {
                vocab_size,
                embed_dim,
                n_tokens: rows,
            },
        )?,
        GgmlType::Q4_K => mlx_native::embedding_gather_q4_k(
            encoder,
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::EmbeddingQ4KParams {
                vocab_size,
                embed_dim,
                n_tokens: rows,
            },
        )?,
        GgmlType::Q5_K => mlx_native::embedding_gather_q5_k(
            encoder,
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::EmbeddingQ5KParams {
                vocab_size,
                embed_dim,
                n_tokens: rows,
            },
        )?,
        GgmlType::Q6_K => mlx_native::embedding_gather_q6_k(
            encoder,
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::EmbeddingQ6KParams {
                vocab_size,
                embed_dim,
                n_tokens: rows,
            },
        )?,
        GgmlType::Q8_0 => mlx_native::embedding_gather_q8_0(
            encoder,
            registry,
            device,
            &weight.buffer,
            token_ids,
            output,
            &mlx_native::EmbeddingQ8_0Params {
                vocab_size,
                embed_dim,
                n_tokens: rows,
            },
        )?,
        other => return Err(anyhow!("DFlash2 embedding does not support {other:?}")),
    }
    Ok(())
}

/// Fill the physical anchor+mask rows and encode the target family's native
/// embedding table directly into the reusable draft input buffer.
pub fn dispatch_dflash2_input<T: DFlash2TargetFamily>(
    tensors: &DFlash2ModelTensors,
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    config: &DFlash2Config,
    adapter: &T,
    anchors: &[u32],
    workspace: &mut DFlash2Workspace,
) -> Result<()> {
    if anchors.len() != workspace.max_lanes {
        return Err(anyhow!(
            "DFlash2 anchor count {} != workspace lanes {}",
            anchors.len(),
            workspace.max_lanes
        ));
    }
    adapter.validate(config)?;
    let tokens = workspace.row_tokens.as_mut_slice::<u32>()?;
    for (lane, anchor) in anchors.iter().copied().enumerate() {
        ensure!(
            (anchor as usize) < config.vocab_size,
            "DFlash2 anchor {anchor} exceeds vocabulary {}",
            config.vocab_size
        );
        let lane_rows = &mut tokens[lane * config.block_size..(lane + 1) * config.block_size];
        lane_rows.fill(config.mask_token_id);
        lane_rows[0] = anchor;
    }
    let embedding = adapter.embedding()?;
    ensure!(
        embedding.info.cols == config.hidden_size,
        "DFlash2 {} embedding width {} != draft hidden {}",
        adapter.family_name(),
        embedding.info.cols,
        config.hidden_size
    );
    let rows = workspace.max_lanes * config.block_size;
    dispatch_embedding_rows(
        encoder,
        registry,
        device,
        embedding,
        &workspace.row_tokens,
        &workspace.hidden[0],
        rows,
    )?;
    encoder.memory_barrier();
    let scale = adapter.embedding_scale();
    if scale != 1.0 {
        mlx_native::ops::elementwise::scalar_mul_f32(
            encoder,
            registry,
            device.metal_device(),
            &workspace.hidden[0],
            &workspace.hidden[0],
            rows * config.hidden_size,
            scale,
        )?;
        encoder.memory_barrier();
    }
    if let Some(mask_embedding) = tensors.mask_embedding.as_ref() {
        dispatch_dflash2_mask_rows_f32(
            encoder,
            registry,
            device,
            mask_embedding,
            &workspace.hidden[0],
            DFlash2MaskRowsParams {
                lanes: workspace.max_lanes as u32,
                rows: config.block_size as u32,
                hidden: config.hidden_size as u32,
            },
        )?;
        encoder.memory_barrier();
    }
    Ok(())
}

pub fn target_vocabulary_head<'a, T: DFlash2TargetFamily>(
    adapter: &'a T,
) -> Result<DFlash2VocabularyHead<'a>> {
    Ok(DFlash2VocabularyHead::Native(adapter.vocabulary_head()?))
}
