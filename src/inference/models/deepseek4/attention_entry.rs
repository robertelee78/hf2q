//! Public entry points for the exact DeepSeek-V4 attention graph.

use anyhow::Result;
use mlx_native::MlxBuffer;

use super::cache::Deepseek4Cache;
use super::Deepseek4Model;

impl Deepseek4Model {
    /// Embed one token and execute exact layer-0 attention in one command buffer.
    pub fn forward_layer0_attention_one(
        &mut self,
        token_id: u32,
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        self.forward_uncompressed_attention_one(None, token_id, 0, cache, true)
    }
}
