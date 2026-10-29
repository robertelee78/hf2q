//! Borrowed actual field traversal. No buffer clones or private-field exports.
use super::{Deepseek4Cache, Deepseek4CacheSnapshot};
use anyhow::{ensure, Result};
use mlx_native::MlxBuffer;

type Visitor<'a> = dyn FnMut(&str, Option<&MlxBuffer>) -> Result<()> + 'a;
impl Deepseek4Cache {
    pub(crate) fn q5_visit(&self, visit: &mut Visitor<'_>) -> Result<()> {
        ensure!(
            self.layers.len() == self.plan.layers.len(),
            "Q5 cache plan/layer drift"
        );
        for (index, layer) in self.layers.iter().enumerate() {
            for (role, buffer) in [
                ("attention_parent", Some(&layer.attention_kv)),
                ("window_view", Some(&layer.window_kv)),
                ("compressed_view", layer.compressed_kv.as_ref()),
                ("indexer_kv", layer.indexer_kv.as_ref()),
                ("main_kv_state", layer.main_kv_state.as_ref()),
                ("main_score_state", layer.main_score_state.as_ref()),
                ("indexer_kv_state", layer.indexer_kv_state.as_ref()),
                ("indexer_score_state", layer.indexer_score_state.as_ref()),
            ] {
                visit(&format!("layer/{index}/{role}"), buffer)?;
            }
        }
        Ok(())
    }
}
impl Deepseek4CacheSnapshot {
    pub(crate) fn q5_visit(&self, visit: &mut Visitor<'_>) -> Result<()> {
        ensure!(
            self.layers.len() == self.plan.layers.len(),
            "Q5 anchor plan/layer drift"
        );
        for (index, layer) in self.layers.iter().enumerate() {
            for (role, buffer) in [
                ("window_copy", Some(&layer.window_kv)),
                ("main_kv_state_copy", layer.main_kv_state.as_ref()),
                ("main_score_state_copy", layer.main_score_state.as_ref()),
                ("indexer_kv_state_copy", layer.indexer_kv_state.as_ref()),
                (
                    "indexer_score_state_copy",
                    layer.indexer_score_state.as_ref(),
                ),
            ] {
                visit(&format!("layer/{index}/{role}"), buffer)?;
            }
        }
        Ok(())
    }
}
