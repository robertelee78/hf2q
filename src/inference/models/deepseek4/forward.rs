//! First native DeepSeek-V4 forward boundary: token IDs to HC state.

use anyhow::{bail, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::ops::repeat_tiled::{dispatch_repeat_tiled_f32, RepeatTiledParams};
use mlx_native::{
    DType, EmbeddingQ2KParams, EmbeddingQ8_0Params, KernelRegistry, MlxBuffer, MlxDevice,
};

use super::Deepseek4Model;

/// Buffers that must stay alive until the active command buffer completes.
pub(super) struct EmbeddingArena {
    pub(super) ids: MlxBuffer,
    pub(super) gathered: MlxBuffer,
    pub(super) state: MlxBuffer,
}

impl Deepseek4Model {
    /// Gather quantized token rows and expand each activation into the four
    /// official Hyper-Connection streams in one Metal command buffer.
    pub fn embed_hyper_state(&mut self, token_ids: &[u32]) -> Result<MlxBuffer> {
        let arena = self.prepare_embedding_arena(token_ids)?;
        let embedding = self
            .weights
            .raw_matrix_ref("token_embd.weight")
            .context("DeepSeek-V4 token embedding residency")?;
        let vocab = self.cfg.vocab_size as usize;
        let hidden = self.cfg.hidden_size as usize;
        let hc = self.cfg.hyper_connection_count;
        let tokens = token_ids.len();
        let (executor, registry) = self.ctx.split();
        let mut session = executor
            .begin()
            .context("begin DeepSeek-V4 embedding graph")?;
        Self::encode_embedding_hyper_state(
            &mut session,
            registry,
            executor.device(),
            embedding.buffer,
            embedding.ggml_type,
            embedding.shape,
            &arena,
            tokens,
            vocab,
            hidden,
            hc,
        )?;
        session
            .finish()
            .context("execute DeepSeek-V4 embedding graph")?;
        Ok(arena.state)
    }

    pub(super) fn prepare_embedding_arena(&self, token_ids: &[u32]) -> Result<EmbeddingArena> {
        if token_ids.is_empty() {
            bail!("DeepSeek-V4 forward requires at least one token");
        }
        let hidden = self.cfg.hidden_size as usize;
        let tokens = token_ids.len();
        let device = self.ctx.device().clone();
        let mut ids = device
            .alloc_buffer(tokens * DType::U32.size_of(), DType::U32, vec![tokens])
            .context("allocate DeepSeek-V4 token IDs")?;
        ids.as_mut_slice::<u32>()?.copy_from_slice(token_ids);
        let gathered = device
            .alloc_buffer(
                tokens * hidden * DType::F32.size_of(),
                DType::F32,
                vec![tokens, 1, hidden],
            )
            .context("allocate DeepSeek-V4 gathered embeddings")?;
        let hc = self.cfg.hyper_connection_count as usize;
        let state = device
            .alloc_buffer(
                tokens * hc * hidden * DType::F32.size_of(),
                DType::F32,
                vec![tokens, hc, hidden],
            )
            .context("allocate DeepSeek-V4 Hyper-Connection state")?;
        Ok(EmbeddingArena {
            ids,
            gathered,
            state,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_embedding_hyper_state(
        session: &mut GraphSession<'_>,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        embedding: &MlxBuffer,
        embedding_type: GgmlType,
        embedding_shape: &[usize],
        arena: &EmbeddingArena,
        tokens: usize,
        vocab: usize,
        hidden: usize,
        hc: u32,
    ) -> Result<()> {
        if embedding_shape != [vocab, hidden] {
            bail!(
                "DeepSeek-V4 token embedding shape drift: got {embedding_shape:?}, expected [{vocab}, {hidden}]"
            );
        }
        session.barrier_between(&[embedding, &arena.ids], &[&arena.gathered]);
        match embedding_type {
            GgmlType::Q2_K => session
                .embedding_gather_q2_k(
                    registry,
                    device,
                    embedding,
                    &arena.ids,
                    &arena.gathered,
                    &EmbeddingQ2KParams {
                        vocab_size: vocab,
                        embed_dim: hidden,
                        n_tokens: tokens,
                    },
                )
                .context("encode DeepSeek-V4 Q2_K embedding gather")?,
            GgmlType::Q8_0 => session
                .embedding_gather_q8_0(
                    registry,
                    device,
                    embedding,
                    &arena.ids,
                    &arena.gathered,
                    &EmbeddingQ8_0Params {
                        vocab_size: vocab,
                        embed_dim: hidden,
                        n_tokens: tokens,
                    },
                )
                .context("encode DeepSeek-V4 Q8_0 embedding gather")?,
            other => bail!(
                "DeepSeek-V4 optimized runtime requires Q2_K or Q8_0 token embeddings, got {other:?}"
            ),
        }
        session.barrier_between(&[&arena.gathered], &[&arena.state]);
        dispatch_repeat_tiled_f32(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            &arena.gathered,
            &arena.state,
            &RepeatTiledParams {
                seq: u32::try_from(tokens).context("DeepSeek-V4 token count exceeds u32")?,
                hg: 1,
                h: hc,
                k: u32::try_from(hidden).context("DeepSeek-V4 hidden size exceeds u32")?,
            },
        )
        .context("encode DeepSeek-V4 Hyper-Connection expansion")?;
        Ok(())
    }
}
