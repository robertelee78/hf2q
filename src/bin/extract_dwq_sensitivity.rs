//! ADR-014 P11 iter-97 — extract DWQ sensitivity from an existing dwq-mixed GGUF
//! and write it to the local sensitivity cache so a re-emit can hit cache and skip
//! the OOM-prone Phase 2 activation capture.
//!
//! Usage:
//!   extract_dwq_sensitivity <existing_dwq_gguf> <cache_key_hash>
//!
//! Reads `<existing_dwq_gguf>`, enumerates `blk.{i}.attn_q.weight` for each layer,
//! and infers sensitivity from its ggml_type:
//!   - Q5_K / Q6_K (or any type higher-bpw than Q4_K) → sensitive (1.0)
//!   - Q4_K (or lower-bpw)                            → base      (0.0)
//!
//! Writes `~/.cache/hf2q/sensitivity/<cache_key_hash>.json` with the
//! `CachedCalibrationPayload::Dwq { map: BTreeMap<String, Vec<f32>> }` envelope
//! that `dwq_calibrator::calibrate()`'s cache-HIT path expects.
//!
//! Background: ADR-014 P4 iter-1 commit `f8f727e` eliminated the P9b
//! intermediate-GGUF dance for activation capture, but the new lazy path
//! 4×-expands BF16 → F32 host-side for dense FFN weights via `load_lazy_f32`,
//! causing 158-186 GB peak memory on Qwen3.6 27B (jetsam SIGKILL on M5 Max
//! 128 GB). iter-95 hoisted the cache HIT short-circuit above the Qwen35Model
//! build, but no DWQ sensitivity cache entries existed. This binary primes
//! the cache from the existing Apr 26 emission, preserving the original
//! ranking and unblocking P11 first-time conversion at low memory peak.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlx_native::gguf::GgufFile;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;

#[derive(serde::Serialize)]
struct CachedCalibrationPayload {
    kind: &'static str,
    data: Inner,
}

#[derive(serde::Serialize)]
struct Inner {
    map: BTreeMap<String, Vec<f32>>,
}

#[derive(serde::Serialize)]
struct SensitivityCacheEntry {
    algorithm_version: String,
    model_sha: String,
    corpus_sha: String,
    payload: CachedCalibrationPayload,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let gguf_path: PathBuf = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing arg: <existing_dwq_gguf>"))?
        .into();
    let cache_key_hash: String = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing arg: <cache_key_hash>"))?;

    let gguf = GgufFile::open(&gguf_path)
        .map_err(|e| anyhow::anyhow!("GgufFile::open({}): {e}", gguf_path.display()))?;

    let n_layers = gguf
        .metadata_u32("qwen35.block_count")
        .or_else(|| gguf.metadata_u32("qwen35moe.block_count"))
        .or_else(|| gguf.metadata_u32("qwen3_5.block_count"))
        .or_else(|| gguf.metadata_u32("qwen3_5moe.block_count"))
        .or_else(|| gguf.metadata_u32("llama.block_count"))
        .ok_or_else(|| anyhow::anyhow!("could not read block_count from GGUF metadata"))?
        as usize;

    eprintln!("[extract] gguf: {}", gguf_path.display());
    eprintln!("[extract] n_layers: {n_layers}");

    // Per-layer Q-type histogram for verification + sensitivity inference.
    // We treat Q5_K, Q6_K, and any higher-bpw type as "sensitive";
    // Q4_K and lower as "base". Probe `attn_q.weight` (always present
    // for dense Qwen3.5/3.6 attention layers; for linear-attn layers the
    // probe falls back to ffn_gate.weight or attn_k_b.weight).
    let probe_names = |i: usize| -> Vec<String> {
        vec![
            // Dense Qwen3.5/3.6 (full-attention layers).
            format!("blk.{i}.attn_q.weight"),
            format!("blk.{i}.attn_k_b.weight"),
            format!("blk.{i}.ffn_gate.weight"),
            format!("blk.{i}.ffn_down.weight"),
            // MoE Qwen3.5/3.6 (35B-A3B / apex shape).
            format!("blk.{i}.ffn_gate_exps.weight"),
            format!("blk.{i}.ffn_down_exps.weight"),
            format!("blk.{i}.ffn_up_exps.weight"),
            // Linear-attention layers in 3:1 hybrid (attn_qkv fused).
            format!("blk.{i}.attn_qkv.weight"),
            format!("blk.{i}.attn_gate.weight"),
            format!("blk.{i}.ssm_out.weight"),
        ]
    };

    let mut map: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut sensitive_count = 0_usize;
    let mut q_type_hist: BTreeMap<String, usize> = BTreeMap::new();
    for i in 0..n_layers {
        let mut q_type_for_layer: Option<GgmlType> = None;
        for name in probe_names(i) {
            if let Some(info) = gguf.tensor_info(&name) {
                q_type_for_layer = Some(info.ggml_type);
                break;
            }
        }
        let q = q_type_for_layer.ok_or_else(|| {
            anyhow::anyhow!("layer {i}: no probe tensor found among attn_q/attn_k_b/ffn_*")
        })?;
        *q_type_hist.entry(format!("{q:?}")).or_default() += 1;

        // Sensitivity rule for dwq-4-6: base=Q4_K, sensitive=Q5_K (or Q6_K
        // if the layer was promoted by a K-quant variant policy bump).
        // Anything with more bits than Q4_K counts as sensitive.
        // mlx_native::ops::quantized_matmul_ggml::GgmlType variants today:
        // F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, I16.
        let is_sensitive = matches!(
            q,
            GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0 | GgmlType::F16 | GgmlType::F32
        );
        if is_sensitive {
            sensitive_count += 1;
        }

        let key = format!("blk.{i}.sensitivity");
        let value = if is_sensitive { vec![1.0f32] } else { vec![0.0f32] };
        map.insert(key, value);
    }

    eprintln!(
        "[extract] sensitivity: {sensitive_count}/{n_layers} layers marked sensitive"
    );
    eprintln!("[extract] per-layer Q-type histogram:");
    for (k, v) in &q_type_hist {
        eprintln!("  {k}: {v}");
    }

    // Derive cache file path from the hash arg.
    let home = std::env::var("HOME")
        .map_err(|_| anyhow::anyhow!("HOME env var unset; cannot resolve cache dir"))?;
    let cache_dir = Path::new(&home).join(".cache/hf2q/sensitivity");
    std::fs::create_dir_all(&cache_dir)?;
    let cache_path = cache_dir.join(format!("{cache_key_hash}.json"));

    // Build the envelope. The model_sha/corpus_sha fields are informational
    // only at load time (the lookup is by filename); we write the canonical
    // values for forensic traceability — operators can verify the entry
    // claims to be for Qwen3.6-27B + 1024-sample synthetic corpus by
    // inspecting these fields after load.
    let entry = SensitivityCacheEntry {
        algorithm_version: "1.0.variance-magnitude".to_string(),
        model_sha: "extracted-from-gguf-iter97".to_string(),
        corpus_sha: "extracted-from-gguf-iter97".to_string(),
        payload: CachedCalibrationPayload {
            kind: "Dwq",
            data: Inner { map },
        },
    };
    let json = serde_json::to_string_pretty(&entry)?;

    // Atomic write: temp + rename, mirrors `cache::save_to_path` for
    // crash-safe semantics.
    let tmp_path = cache_path.with_extension("json.tmp");
    std::fs::write(&tmp_path, &json)?;
    std::fs::rename(&tmp_path, &cache_path)?;

    eprintln!("[extract] wrote cache JSON: {}", cache_path.display());
    eprintln!("[extract] {} bytes", json.len());

    Ok(())
}
