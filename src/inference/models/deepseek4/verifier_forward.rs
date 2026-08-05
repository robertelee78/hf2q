//! Full one-token DeepSeek-V4 verifier execution and cache publication.

use anyhow::{Context, Result};
use mlx_native::MlxBuffer;

use super::cache::Deepseek4Cache;
use super::Deepseek4Model;

impl Deepseek4Model {
    /// Execute all verifier blocks for one token.
    ///
    /// Each block currently submits its own command buffer, so a failure after
    /// an earlier submission poisons the cache. The caller must reset and
    /// replay the request; retrying only the failed token is rejected.
    pub fn forward_verifier_one(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        let position = cache.position();
        let result = self.forward_verifier_one_uncommitted(token_id, cache);
        match result {
            Ok(state) => {
                if let Err(error) = cache.commit_step(position) {
                    cache.poison();
                    return Err(error).context("publish complete DeepSeek-V4 verifier token");
                }
                Ok(state)
            }
            Err(error) => {
                cache.poison();
                Err(error).context("DeepSeek-V4 verifier token partially executed; cache poisoned")
            }
        }
    }

    fn forward_verifier_one_uncommitted(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        let mut state = self
            .forward_uncompressed_attention_one(None, token_id, 0, cache, false)
            .context("execute DeepSeek-V4 layer-0 attention")?;
        state = self
            .forward_ffn_one(&state, token_id, 0)
            .context("execute DeepSeek-V4 layer-0 FFN")?;
        for layer in 1..self.cfg.num_hidden_layers as usize {
            state = if self.cfg.compress_ratios[layer] == 0 {
                self.forward_uncompressed_attention_one(Some(&state), token_id, layer, cache, false)
            } else {
                self.forward_compressed_attention_one(&state, layer, cache, false)
            }
            .with_context(|| format!("execute DeepSeek-V4 layer-{layer} attention"))?;
            state = self
                .forward_ffn_one(&state, token_id, layer)
                .with_context(|| format!("execute DeepSeek-V4 layer-{layer} FFN"))?;
        }
        Ok(state)
    }
}
