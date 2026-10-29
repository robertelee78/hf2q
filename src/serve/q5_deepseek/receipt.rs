//! Bounded output-only receipt. It cannot mint permission or claim physical fit.
use anyhow::{Context, Result};
use serde::Serialize;
use std::io::{self, Write};

struct BoundedBytes {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedBytes {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let size = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|size| *size <= self.limit)
            .ok_or_else(|| io::Error::other("Q5 receipt byte quota exceeded"))?;
        self.bytes
            .try_reserve(size - self.bytes.len())
            .map_err(|_| io::Error::other("Q5 receipt allocation failed"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(super) fn encode(value: &impl Serialize, limit: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(limit)
        .context("reserve finite Q5 receipt buffer")?;
    let mut out = BoundedBytes { bytes, limit };
    serde_json::to_writer(&mut out, value).context("serialize bounded Q5 receipt")?;
    Ok(out.bytes)
}

#[derive(Serialize)]
pub(super) struct PreloadReceipt<'a> {
    pub schema: &'static str,
    pub verdict: &'static str,
    pub model_loaded: bool,
    pub runtime_qualified: bool,
    pub physical_credit: bool,
    pub retained_coalesced_upper_bytes: u64,
    pub source_native_binary_refs: [&'a str; 3],
    pub reference_authority: &'static str,
    pub native_page_observer: &'static str,
    pub actual_header_plan: HeaderPlan,
    pub required_root_domains: Vec<(&'static str, &'static str)>,
}

/// Pure projection from the SAME parsed header; not current allocation or fit.
#[derive(Serialize)]
#[serde(tag = "status")]
pub(super) enum HeaderPlan {
    Unknown {
        reason: &'static str,
    },
    Parsed {
        layers: u32,
        compression_ratios: Vec<u32>,
        hash_layers: u32,
        context: usize,
        logical_slots: u32,
        one_cache_requested_bytes: u64,
    },
}
impl HeaderPlan {
    pub(super) fn actual(header: &mlx_native::gguf::GgufFile) -> Self {
        use crate::inference::models::deepseek4::{cache::Deepseek4CachePlan, Deepseek4Model};
        if !header
            .metadata_u32("deepseek4.block_count")
            .is_some_and(|n| (1..=64).contains(&n))
        {
            return Self::Unknown {
                reason: "MissingOrOutOfBoundActualLayerCount",
            };
        }
        let Ok(cfg) = Deepseek4Model::load_config_only(header) else {
            return Self::Unknown {
                reason: "ActualStrictConfigUnavailable",
            };
        };
        let Ok(plan) = Deepseek4CachePlan::for_context(&cfg, 262_144) else {
            return Self::Unknown {
                reason: "ActualCachePlanUnavailable",
            };
        };
        Self::Parsed {
            layers: cfg.num_hidden_layers,
            compression_ratios: cfg.compress_ratios,
            hash_layers: cfg.hash_layer_count,
            context: 262_144,
            logical_slots: 4,
            one_cache_requested_bytes: plan.resident_bytes,
        }
    }
}
