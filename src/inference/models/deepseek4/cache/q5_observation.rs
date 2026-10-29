//! Borrowed actual field traversal. No buffer clones or private-field exports.
use super::{CacheBufferPlan, Deepseek4Cache, Deepseek4CacheSnapshot};
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
            let plan = &self.plan.layers[index];
            for (role, expected, buffer) in [
                (
                    "attention_parent",
                    Some(&plan.attention_kv),
                    Some(&layer.attention_kv),
                ),
                ("window_view", Some(&plan.window_kv), Some(&layer.window_kv)),
                (
                    "compressed_view",
                    plan.compressed_kv.as_ref(),
                    layer.compressed_kv.as_ref(),
                ),
                (
                    "indexer_kv",
                    plan.indexer_kv.as_ref(),
                    layer.indexer_kv.as_ref(),
                ),
                (
                    "main_kv_state",
                    plan.main_kv_state.as_ref(),
                    layer.main_kv_state.as_ref(),
                ),
                (
                    "main_score_state",
                    plan.main_score_state.as_ref(),
                    layer.main_score_state.as_ref(),
                ),
                (
                    "indexer_kv_state",
                    plan.indexer_kv_state.as_ref(),
                    layer.indexer_kv_state.as_ref(),
                ),
                (
                    "indexer_score_state",
                    plan.indexer_score_state.as_ref(),
                    layer.indexer_score_state.as_ref(),
                ),
            ] {
                require_field(expected, buffer)?;
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
            let plan = &self.plan.layers[index];
            for (role, expected, buffer) in [
                ("window_copy", Some(&plan.window_kv), Some(&layer.window_kv)),
                (
                    "main_kv_state_copy",
                    plan.main_kv_state.as_ref(),
                    layer.main_kv_state.as_ref(),
                ),
                (
                    "main_score_state_copy",
                    plan.main_score_state.as_ref(),
                    layer.main_score_state.as_ref(),
                ),
                (
                    "indexer_kv_state_copy",
                    plan.indexer_kv_state.as_ref(),
                    layer.indexer_kv_state.as_ref(),
                ),
                (
                    "indexer_score_state_copy",
                    plan.indexer_score_state.as_ref(),
                    layer.indexer_score_state.as_ref(),
                ),
            ] {
                require_field(expected, buffer)?;
                visit(&format!("layer/{index}/{role}"), buffer)?;
            }
        }
        Ok(())
    }
}

fn require_field(expected: Option<&CacheBufferPlan>, actual: Option<&MlxBuffer>) -> Result<()> {
    match (expected, actual) {
        (None, None) => Ok(()),
        (Some(plan), Some(buffer)) => {
            ensure!(
                buffer.dtype() == plan.dtype
                    && buffer.shape() == plan.shape.as_slice()
                    && u64::try_from(buffer.data_byte_len())? == plan.bytes,
                "Q5 actual cache/anchor field differs from source plan"
            );
            Ok(())
        }
        _ => anyhow::bail!("Q5 cache/anchor Present/Absent differs from actual plan"),
    }
}
