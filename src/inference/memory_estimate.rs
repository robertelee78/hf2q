//! Memory estimation for inference model loading.
//!
//! Calculates the estimated memory footprint of a model before loading
//! and compares against available system unified memory. Fails fast
//! with a clear error message if the model won't fit.

use std::path::Path;

use thiserror::Error;
use tracing::{debug, info, warn};

use crate::inference::models::registry::ModelConfig;

/// Errors from memory estimation.
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error(
        "Insufficient memory to load model.\n\
         \n\
         Estimated requirement: {estimated_gb:.1} GB\n\
         Available memory:      {available_gb:.1} GB\n\
         \n\
         Try a smaller model or a more aggressively quantized variant."
    )]
    InsufficientMemory {
        estimated_gb: f64,
        available_gb: f64,
    },

    #[error("Failed to estimate memory: {reason}")]
    EstimationFailed { reason: String },
}

/// Memory estimation result.
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Estimated weight memory in bytes.
    pub weight_bytes: u64,
    /// Estimated KV cache memory in bytes (for a reasonable default context).
    pub kv_cache_bytes: u64,
    /// Total estimated memory in bytes.
    pub total_bytes: u64,
    /// Available system memory in bytes.
    pub available_bytes: u64,
}

impl MemoryEstimate {
    /// Check if the model fits in available memory (with a 90% threshold).
    pub fn fits(&self) -> bool {
        let threshold = (self.available_bytes as f64 * 0.90) as u64;
        self.total_bytes <= threshold
    }

    /// Percentage of available memory the model would use.
    pub fn usage_percent(&self) -> f64 {
        if self.available_bytes == 0 {
            return 100.0;
        }
        (self.total_bytes as f64 / self.available_bytes as f64) * 100.0
    }
}

/// Estimate the memory required to load a model for inference.
///
/// Reads the model config to determine:
/// - Weight memory: based on parameter count and quantization bits
/// - KV cache memory: based on layers, heads, head_dim, and a default context length
///
/// # Arguments
///
/// * `model_config` - Parsed model configuration
/// * `model_dir` - Path to model directory (for quantization_config.json)
/// * `max_seq_len` - Maximum sequence length to budget KV cache for
pub fn estimate_memory(
    model_config: &ModelConfig,
    model_dir: &Path,
    max_seq_len: u64,
) -> Result<MemoryEstimate, MemoryError> {
    // Determine quantization bit-width
    let quant_bits = read_quant_bits(model_dir);

    // Estimate weight memory
    let weight_bytes = estimate_weight_memory(model_config, quant_bits);

    // Estimate KV cache memory
    let kv_cache_bytes = estimate_kv_cache_memory(model_config, max_seq_len);

    // Get available system memory
    let available_bytes = get_available_memory();

    let total_bytes = weight_bytes + kv_cache_bytes;

    let estimate = MemoryEstimate {
        weight_bytes,
        kv_cache_bytes,
        total_bytes,
        available_bytes,
    };

    info!(
        weight_mb = weight_bytes / (1024 * 1024),
        kv_cache_mb = kv_cache_bytes / (1024 * 1024),
        total_mb = total_bytes / (1024 * 1024),
        available_mb = available_bytes / (1024 * 1024),
        usage_pct = format!("{:.1}%", estimate.usage_percent()),
        "Memory estimation complete"
    );

    Ok(estimate)
}

/// Perform the memory check and fail fast if insufficient.
pub fn check_memory(estimate: &MemoryEstimate) -> Result<(), MemoryError> {
    if !estimate.fits() {
        return Err(MemoryError::InsufficientMemory {
            estimated_gb: estimate.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            available_gb: estimate.available_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        });
    }
    Ok(())
}

/// Estimate weight memory based on config and quantization.
fn estimate_weight_memory(config: &ModelConfig, quant_bits: u8) -> u64 {
    let hidden = config.hidden_size();
    let layers = config.num_hidden_layers();
    let heads = config.num_attention_heads();
    let kv_heads = config.num_kv_heads();
    let intermediate = config.intermediate_size();
    let vocab = config.vocab_size();
    let num_experts = config.num_experts();
    let head_dim = config.head_dim();

    if hidden == 0 || layers == 0 {
        warn!("Config missing hidden_size or num_hidden_layers, using fallback estimation");
        return 0;
    }

    // Attention parameters per layer:
    //   q_proj: hidden * (heads * head_dim)
    //   k_proj: hidden * (kv_heads * head_dim)
    //   v_proj: hidden * (kv_heads * head_dim)
    //   o_proj: (heads * head_dim) * hidden
    let kv_h = if kv_heads > 0 { kv_heads } else { heads };
    let attn_params_per_layer = hidden * heads * head_dim  // q_proj
        + hidden * kv_h * head_dim                         // k_proj
        + hidden * kv_h * head_dim                         // v_proj
        + heads * head_dim * hidden;                       // o_proj

    // FFN parameters per layer (SwiGLU: gate + up + down)
    let ffn_params_per_layer = if intermediate > 0 {
        hidden * intermediate * 3 // gate, up, down
    } else {
        hidden * hidden * 4 * 3 / hidden // fallback
    };

    // MoE: multiply FFN by num_experts
    let ffn_total = if num_experts > 1 {
        ffn_params_per_layer * num_experts
    } else {
        ffn_params_per_layer
    };

    // Norms per layer (small, but count them)
    let norm_params_per_layer = hidden * 2; // input_norm + post_norm

    // Total per-layer params
    let params_per_layer = attn_params_per_layer + ffn_total + norm_params_per_layer;

    // Embedding + lm_head
    let embedding_params = vocab * hidden;
    let lm_head_params = vocab * hidden;

    // Global norms
    let global_norm_params = hidden * 2;

    let total_params = params_per_layer * layers + embedding_params + lm_head_params + global_norm_params;

    // Convert to bytes based on quantization
    let bytes_per_param = match quant_bits {
        2 => 0.25,   // 2 bits = 0.25 bytes
        3 => 0.375,  // 3 bits
        4 => 0.5,    // 4 bits
        6 => 0.75,   // 6 bits
        8 => 1.0,    // 8 bits
        16 => 2.0,   // f16
        _ => 0.5,    // default to 4-bit estimate
    };

    // Add ~10% overhead for scales, biases, and metadata
    let raw_bytes = (total_params as f64 * bytes_per_param) as u64;
    let overhead = raw_bytes / 10;

    debug!(
        total_params = total_params,
        quant_bits = quant_bits,
        raw_bytes = raw_bytes,
        "Weight memory estimation"
    );

    raw_bytes + overhead
}

/// Estimate KV cache memory for a given sequence length.
fn estimate_kv_cache_memory(config: &ModelConfig, max_seq_len: u64) -> u64 {
    let layers = config.num_hidden_layers();
    let kv_heads = config.num_kv_heads();
    let heads = config.num_attention_heads();
    let head_dim = config.head_dim();

    let kv_h = if kv_heads > 0 { kv_heads } else { heads };

    if layers == 0 || kv_h == 0 || head_dim == 0 {
        return 0;
    }

    // KV cache per layer: 2 (K+V) * kv_heads * head_dim * seq_len * 2 bytes (f16)
    let bytes_per_layer = 2 * kv_h * head_dim * max_seq_len * 2;
    let total = bytes_per_layer * layers;

    debug!(
        layers = layers,
        kv_heads = kv_h,
        head_dim = head_dim,
        max_seq_len = max_seq_len,
        total_mb = total / (1024 * 1024),
        "KV cache memory estimation"
    );

    total
}

/// Read the default quantization bit-width from quantization_config.json.
fn read_quant_bits(model_dir: &Path) -> u8 {
    let config_path = model_dir.join("quantization_config.json");
    if !config_path.exists() {
        // No quant config — assume f16 weights
        return 16;
    }

    match std::fs::read_to_string(&config_path) {
        Ok(content) => {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(&content);
            match parsed {
                Ok(v) => {
                    v.get("bits")
                        .and_then(|b| b.as_u64())
                        .map(|b| b as u8)
                        .unwrap_or(4)
                }
                Err(_) => 4,
            }
        }
        Err(_) => 4,
    }
}

/// Get available system unified memory in bytes.
///
/// On macOS, queries `sysctl hw.memsize` for total physical memory.
fn get_available_memory() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();

    // Total memory is a better metric on Apple Silicon since it's unified
    let total = sys.total_memory(); // in bytes

    if total == 0 {
        warn!("Could not determine system memory, defaulting to 8 GB");
        return 8 * 1024 * 1024 * 1024;
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config(json: &str) -> ModelConfig {
        let raw: serde_json::Value = serde_json::from_str(json).unwrap();
        ModelConfig {
            architecture: crate::inference::models::registry::ModelArchitecture::Gemma4,
            architecture_str: "Gemma4ForConditionalGeneration".to_string(),
            raw,
        }
    }

    #[test]
    fn test_estimate_weight_memory_basic() {
        let config = make_test_config(r#"{
            "text_config": {
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "vocab_size": 262144,
                "intermediate_size": 2112,
                "num_experts": 128,
                "head_dim": 256
            }
        }"#);

        let weight_bytes = estimate_weight_memory(&config, 4);
        // Should be a reasonable number for a ~26B MoE model at 4-bit
        assert!(weight_bytes > 0);
        let weight_gb = weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        // Rough sanity: 26B params at 4-bit ~= 13 GB, but MoE has many experts
        assert!(weight_gb > 1.0, "Expected > 1 GB, got {:.1} GB", weight_gb);
    }

    #[test]
    fn test_estimate_kv_cache_memory() {
        let config = make_test_config(r#"{
            "text_config": {
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 256
            }
        }"#);

        let kv_bytes = estimate_kv_cache_memory(&config, 4096);
        assert!(kv_bytes > 0);
        let kv_mb = kv_bytes as f64 / (1024.0 * 1024.0);
        // 30 layers * 2 * 8 * 256 * 4096 * 2 bytes = ~960 MB
        assert!(kv_mb > 100.0, "Expected > 100 MB KV cache, got {:.1} MB", kv_mb);
    }

    #[test]
    fn test_estimate_kv_cache_zero_config() {
        let config = make_test_config(r#"{}"#);
        let kv_bytes = estimate_kv_cache_memory(&config, 4096);
        assert_eq!(kv_bytes, 0);
    }

    #[test]
    fn test_memory_estimate_fits() {
        let estimate = MemoryEstimate {
            weight_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
            kv_cache_bytes: 1 * 1024 * 1024 * 1024, // 1 GB
            total_bytes: 11 * 1024 * 1024 * 1024,   // 11 GB
            available_bytes: 32 * 1024 * 1024 * 1024, // 32 GB
        };
        assert!(estimate.fits());
        assert!(estimate.usage_percent() < 50.0);
    }

    #[test]
    fn test_memory_estimate_does_not_fit() {
        let estimate = MemoryEstimate {
            weight_bytes: 28 * 1024 * 1024 * 1024,
            kv_cache_bytes: 2 * 1024 * 1024 * 1024,
            total_bytes: 30 * 1024 * 1024 * 1024,
            available_bytes: 32 * 1024 * 1024 * 1024,
        };
        assert!(!estimate.fits()); // 30/32 = 93.75% > 90%
    }

    #[test]
    fn test_check_memory_passes() {
        let estimate = MemoryEstimate {
            weight_bytes: 5_000_000_000,
            kv_cache_bytes: 500_000_000,
            total_bytes: 5_500_000_000,
            available_bytes: 32_000_000_000,
        };
        assert!(check_memory(&estimate).is_ok());
    }

    #[test]
    fn test_check_memory_fails() {
        let estimate = MemoryEstimate {
            weight_bytes: 30_000_000_000,
            kv_cache_bytes: 2_000_000_000,
            total_bytes: 32_000_000_000,
            available_bytes: 16_000_000_000,
        };
        let result = check_memory(&estimate);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Insufficient memory"));
    }

    #[test]
    fn test_read_quant_bits_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let bits = read_quant_bits(tmp.path());
        assert_eq!(bits, 16); // No config = f16
    }

    #[test]
    fn test_read_quant_bits_present() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("quantization_config.json"),
            r#"{"bits": 4, "group_size": 64}"#,
        )
        .unwrap();
        let bits = read_quant_bits(tmp.path());
        assert_eq!(bits, 4);
    }

    #[test]
    fn test_get_available_memory_nonzero() {
        let mem = get_available_memory();
        assert!(mem > 0, "Available memory should be > 0");
    }

    #[test]
    fn test_usage_percent() {
        let estimate = MemoryEstimate {
            weight_bytes: 0,
            kv_cache_bytes: 0,
            total_bytes: 8_000_000_000,
            available_bytes: 32_000_000_000,
        };
        let pct = estimate.usage_percent();
        assert!((pct - 25.0).abs() < 0.1);
    }
}
