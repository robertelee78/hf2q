//! ADR-065 E5 DeepSeek-V4 joint-step routed-expert diagnostic.
//!
//! `HF2Q_ADR065_EXPERT_UNION=1` blit-copies each joint FFN layer's sanitized
//! routed ids into one per-step buffer. The buffer is read only after the
//! transaction's existing pipeline drain, so the observer adds no host/GPU
//! synchronization. Each completed width-2..=4 joint step emits one JSON line.

use std::sync::LazyLock;

use anyhow::{ensure, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::forward_support::alloc;

pub(crate) const ENV: &str = "HF2Q_ADR065_EXPERT_UNION";
pub(crate) const SCHEMA: &str = "hf2q.adr065.deepseek-expert-union.v1";
const PREFIX: &str = "[ADR065_EXPERT_UNION_JSON] ";

static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var(ENV).as_deref() == Ok("1"));

pub(crate) fn enabled() -> bool {
    *ENABLED
}

pub(super) struct Capture {
    ids: MlxBuffer,
    layers: usize,
    rows: usize,
    top_k: usize,
    expert_bytes: Vec<u64>,
}

impl Capture {
    pub(super) fn for_joint_step(
        device: &MlxDevice,
        layers: usize,
        rows: usize,
        top_k: usize,
    ) -> Result<Option<Self>> {
        if !enabled() || !(2..=4).contains(&rows) {
            return Ok(None);
        }
        let count = layers
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(top_k))
            .context("E5 routed-id capture size overflow")?;
        Ok(Some(Self {
            ids: alloc(device, DType::U32, vec![count], "E5 routed expert ids")?,
            layers,
            rows,
            top_k,
            expert_bytes: vec![0; layers],
        }))
    }

    pub(super) fn encode_layer(
        &mut self,
        session: &mut GraphSession<'_>,
        layer: usize,
        safe_indices: &MlxBuffer,
        expert_bytes: u64,
    ) -> Result<()> {
        ensure!(layer < self.layers, "E5 layer {layer} exceeds capture");
        ensure!(
            safe_indices.dtype() == DType::U32 && safe_indices.shape() == [self.rows, self.top_k],
            "E5 layer {layer} routed ids must be U32 [{}, {}], got {} {:?}",
            self.rows,
            self.top_k,
            safe_indices.dtype(),
            safe_indices.shape()
        );
        self.expert_bytes[layer] = expert_bytes;
        let bytes = self
            .rows
            .checked_mul(self.top_k)
            .and_then(|value| value.checked_mul(4))
            .context("E5 layer copy size overflow")?;
        session
            .encoder_mut()
            .blit_copy_bytes(
                safe_indices,
                0,
                &self.ids,
                (layer * bytes) as u64,
                bytes as u64,
            )
            .with_context(|| format!("E5 routed-id copy for layer {layer}"))?;
        Ok(())
    }

    pub(super) fn emit_after_drain(self, hash_layers: usize) -> Result<()> {
        let ids = self
            .ids
            .as_logical_slice::<u32>()
            .context("read E5 routed-id capture after joint drain")?;
        let report = report(ids, self.rows, self.top_k, hash_layers, &self.expert_bytes)?;
        eprintln!("{PREFIX}{report}");
        Ok(())
    }
}

fn distinct_count(ids: &[u32]) -> usize {
    let mut unique = ids.to_vec();
    unique.sort_unstable();
    unique.dedup();
    unique.len()
}

fn report(
    ids: &[u32],
    rows: usize,
    top_k: usize,
    hash_layers: usize,
    expert_bytes: &[u64],
) -> Result<serde_json::Value> {
    let layer_stride = rows
        .checked_mul(top_k)
        .context("E5 layer stride overflow")?;
    ensure!(
        ids.len() == layer_stride * expert_bytes.len(),
        "E5 capture has {} ids, expected {}",
        ids.len(),
        layer_stride * expert_bytes.len()
    );
    let mut naive_total = 0u64;
    let mut unique_total = 0u64;
    let layers = ids
        .chunks_exact(layer_stride)
        .enumerate()
        .map(|(layer, layer_ids)| {
            let unique = distinct_count(layer_ids);
            let bytes = expert_bytes[layer];
            let naive = bytes
                .checked_mul(layer_stride as u64)
                .context("E5 naive routed-byte count overflow")?;
            let unique_bytes = bytes
                .checked_mul(unique as u64)
                .context("E5 unique routed-byte count overflow")?;
            naive_total = naive_total
                .checked_add(naive)
                .context("E5 total naive routed-byte count overflow")?;
            unique_total = unique_total
                .checked_add(unique_bytes)
                .context("E5 total unique routed-byte count overflow")?;
            Ok(serde_json::json!({
                "layer": layer,
                "routing": if layer < hash_layers { "hash" } else { "score" },
                "unique_experts": unique,
                "routed_expert_rows": layer_stride,
                "expert_bytes": bytes,
                "routed_bytes_naive": naive,
                "routed_bytes_unique": unique_bytes,
                "reduction_share": reduction_share(naive, unique_bytes),
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(serde_json::json!({
        "schema": SCHEMA,
        "width": rows,
        "top_k": top_k,
        "layers": layers,
        "routed_bytes_naive": naive_total,
        "routed_bytes_unique": unique_total,
        "reduction_share": reduction_share(naive_total, unique_total),
    }))
}

fn reduction_share(naive: u64, unique: u64) -> f64 {
    if naive == 0 {
        0.0
    } else {
        (naive - unique) as f64 / naive as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_separates_hash_and_score_layers_and_bytes() {
        let value = report(&[0, 1, 1, 2, 4, 4, 4, 5], 2, 2, 1, &[100, 200]).unwrap();
        assert_eq!(value["schema"], SCHEMA);
        assert_eq!(value["width"], 2);
        assert_eq!(value["layers"][0]["routing"], "hash");
        assert_eq!(value["layers"][0]["unique_experts"], 3);
        assert_eq!(value["layers"][0]["routed_bytes_naive"], 400);
        assert_eq!(value["layers"][0]["routed_bytes_unique"], 300);
        assert_eq!(value["layers"][1]["routing"], "score");
        assert_eq!(value["layers"][1]["unique_experts"], 2);
        assert_eq!(value["routed_bytes_naive"], 1200);
        assert_eq!(value["routed_bytes_unique"], 700);
    }

    #[test]
    fn report_rejects_incomplete_capture() {
        let error = report(&[1, 2, 3], 2, 2, 1, &[100]).unwrap_err();
        assert!(error.to_string().contains("expected 4"));
    }
}
