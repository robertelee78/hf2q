//! Enforce the ordinary Gemma matrix representation at the load boundary.

use anyhow::{ensure, Result};
use mlx_native::MlxBuffer;

use super::model::MlxModelWeights;
use crate::serve::forward_mlx_shared::MlxQWeight;

fn retain<'a>(weight: &'a MlxQWeight, buffers: &mut Vec<&'a MlxBuffer>) -> Result<()> {
    ensure!(
        weight.affine.is_none(),
        "ordinary GGUF matrix has an affine replacement"
    );
    ensure!(
        weight.f16_shadow.is_none(),
        "ordinary Gemma matrix has an F16 shadow"
    );
    buffers.push(&weight.buffer);
    Ok(())
}

impl MlxModelWeights {
    /// Matrix bytes are mapped unchanged. Elementwise norm/scaling state,
    /// activations, KV state and explicitly requested DWQ overlays have their
    /// own contracts and are excluded from this ordinary-load inventory.
    pub(super) fn verify_native_matrix_storage(&self) -> Result<()> {
        let mut buffers = Vec::new();
        retain(&self.embed_weight, &mut buffers)?;
        retain(self.resolved_lm_head(), &mut buffers)?;
        for layer in &self.layers {
            for projection in [
                &layer.attn.q_proj,
                &layer.attn.k_proj,
                &layer.attn.o_proj,
                &layer.mlp.gate_proj,
                &layer.mlp.up_proj,
                &layer.mlp.down_proj,
            ] {
                retain(projection, &mut buffers)?;
            }
            if let Some(v) = &layer.attn.v_proj {
                retain(v, &mut buffers)?;
            }
            match (&layer.moe.stacked_gate_up, &layer.moe.stacked_down) {
                (Some(gate_up), Some(down)) => {
                    retain(&layer.moe.router_proj, &mut buffers)?;
                    buffers.extend([gate_up, down]);
                }
                (None, None) => {}
                _ => anyhow::bail!("Gemma layer has an incomplete expert matrix pair"),
            }
        }
        let storage =
            crate::serve::native_matrix_storage::summarize_native_matrix_storage(buffers)?;
        ensure!(
            storage.anonymous_bytes == 0,
            "Gemma load created {} anonymous matrix bytes",
            storage.anonymous_bytes
        );
        tracing::info!(
            matrices = storage.unique_matrix_views,
            stored_matrix_bytes = storage.resident_bytes(),
            anonymous_matrix_bytes = storage.anonymous_bytes,
            "Gemma native matrix storage verified"
        );
        Ok(())
    }
}
