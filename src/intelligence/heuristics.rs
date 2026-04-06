//! Rule-based heuristic quant selection for auto mode.
//!
//! When RuVector has no stored results for a hardware+model combination,
//! these heuristics provide a reasonable default based on memory fitting.
//!
//! Rules:
//! - Model fits at f16 with headroom -> f16
//! - Model fits at q8 with headroom -> mixed-4-6
//! - Model fits tight -> q4
//! - Model fits very tight -> q2

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info};

use super::fingerprint::ModelFingerprint;
use super::hardware::HardwareProfile;

/// Errors from heuristic resolution.
#[derive(Error, Debug)]
pub enum HeuristicsError {
    #[error("Heuristic resolution failed: {reason}")]
    ResolutionFailed { reason: String },
}

/// The result of heuristic quant selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeuristicResult {
    /// Recommended quantization method name
    pub quant_method: String,
    /// Recommended bit width
    pub bits: u8,
    /// Recommended group size
    pub group_size: usize,
    /// Confidence level of the recommendation (0.0 to 1.0)
    pub confidence: f64,
    /// Human-readable explanation of why this was chosen
    pub reasoning: String,
}

/// Memory headroom factor — we want at least this much free memory beyond
/// the model size for inference overhead, KV cache, etc.
const MEMORY_HEADROOM_FACTOR: f64 = 1.3;

/// Generous headroom — model comfortably fits with room to spare
const GENEROUS_HEADROOM_FACTOR: f64 = 1.8;

/// Default group size for quantized methods
const DEFAULT_GROUP_SIZE: usize = 64;

/// Select the best quantization method based on hardware and model fingerprint.
///
/// The heuristic works by estimating model size at various bit widths and
/// comparing against available memory. It prefers higher quality (more bits)
/// when memory allows.
pub fn select_quant(
    hardware: &HardwareProfile,
    fingerprint: &ModelFingerprint,
) -> Result<HeuristicResult, HeuristicsError> {
    let available_bytes = hardware.available_memory_bytes;
    let total_bytes = hardware.total_memory_bytes;

    // Use the higher of available vs 70% of total memory as our budget.
    // Available memory can be misleadingly low due to OS file caches that
    // are reclaimable. 70% of total is a conservative but realistic budget.
    let memory_budget = available_bytes.max((total_bytes as f64 * 0.7) as u64);

    let f16_size = fingerprint.estimated_f16_size_bytes();
    let q8_size = fingerprint.estimated_size_bytes(8);
    let q4_size = fingerprint.estimated_size_bytes(4);
    let q2_size = fingerprint.estimated_size_bytes(2);

    debug!(
        memory_budget_gb = memory_budget as f64 / 1e9,
        f16_size_gb = f16_size as f64 / 1e9,
        q8_size_gb = q8_size as f64 / 1e9,
        q4_size_gb = q4_size as f64 / 1e9,
        q2_size_gb = q2_size as f64 / 1e9,
        "Heuristic memory analysis"
    );

    // Rule 1: Model fits comfortably at f16 — no quantization needed
    if f16_size as f64 * GENEROUS_HEADROOM_FACTOR <= memory_budget as f64 {
        let confidence = 0.9;
        let result = HeuristicResult {
            quant_method: "f16".to_string(),
            bits: 16,
            group_size: 0,
            confidence,
            reasoning: format!(
                "Model ({:.1} GB at f16) fits comfortably in available memory ({:.1} GB) with generous headroom. \
                 f16 preserves full precision.",
                f16_size as f64 / 1e9,
                memory_budget as f64 / 1e9,
            ),
        };
        info!(
            method = "f16",
            confidence = confidence,
            "Heuristic: f16 — model fits with generous headroom"
        );
        return Ok(result);
    }

    // Rule 2: Model fits at f16 but without generous headroom — use q8 for safety
    if f16_size as f64 * MEMORY_HEADROOM_FACTOR <= memory_budget as f64 {
        let confidence = 0.75;
        let result = HeuristicResult {
            quant_method: "q8".to_string(),
            bits: 8,
            group_size: DEFAULT_GROUP_SIZE,
            confidence,
            reasoning: format!(
                "Model ({:.1} GB at f16) fits in memory ({:.1} GB) but without generous headroom. \
                 q8 reduces size by 2x with minimal quality loss.",
                f16_size as f64 / 1e9,
                memory_budget as f64 / 1e9,
            ),
        };
        info!(
            method = "q8",
            confidence = confidence,
            "Heuristic: q8 — model fits at f16 but tight"
        );
        return Ok(result);
    }

    // Rule 3: Model fits at q8 with headroom — use mixed-4-6 for good quality/size tradeoff
    if q8_size as f64 * MEMORY_HEADROOM_FACTOR <= memory_budget as f64 {
        let confidence = 0.7;
        let result = HeuristicResult {
            quant_method: "mixed-4-6".to_string(),
            bits: 4,
            group_size: DEFAULT_GROUP_SIZE,
            confidence,
            reasoning: format!(
                "Model ({:.1} GB at q8) fits with headroom in available memory ({:.1} GB). \
                 mixed-4-6 gives good quality with ~4x compression from f16.",
                q8_size as f64 / 1e9,
                memory_budget as f64 / 1e9,
            ),
        };
        info!(
            method = "mixed-4-6",
            confidence = confidence,
            "Heuristic: mixed-4-6 — q8 fits but want better compression"
        );
        return Ok(result);
    }

    // Rule 4: Model fits at q4 — standard quantization
    if q4_size as f64 * MEMORY_HEADROOM_FACTOR <= memory_budget as f64 {
        let confidence = 0.65;
        let result = HeuristicResult {
            quant_method: "q4".to_string(),
            bits: 4,
            group_size: DEFAULT_GROUP_SIZE,
            confidence,
            reasoning: format!(
                "Model ({:.1} GB at q4) fits in available memory ({:.1} GB). \
                 q4 provides ~4x compression from f16 with acceptable quality loss.",
                q4_size as f64 / 1e9,
                memory_budget as f64 / 1e9,
            ),
        };
        info!(
            method = "q4",
            confidence = confidence,
            "Heuristic: q4 — tight memory, standard quantization"
        );
        return Ok(result);
    }

    // Rule 5: Very tight — q2
    if q2_size as f64 * MEMORY_HEADROOM_FACTOR <= memory_budget as f64 {
        let confidence = 0.5;
        let result = HeuristicResult {
            quant_method: "q2".to_string(),
            bits: 2,
            group_size: DEFAULT_GROUP_SIZE,
            confidence,
            reasoning: format!(
                "Model requires aggressive quantization to fit in available memory ({:.1} GB). \
                 q2 provides ~8x compression from f16 but significant quality loss is expected.",
                memory_budget as f64 / 1e9,
            ),
        };
        info!(
            method = "q2",
            confidence = confidence,
            "Heuristic: q2 — very tight memory"
        );
        return Ok(result);
    }

    // Model doesn't fit even at q2
    Err(HeuristicsError::ResolutionFailed {
        reason: format!(
            "Model is too large for available memory even at q2 quantization. \
             Estimated q2 size: {:.1} GB, available memory budget: {:.1} GB. \
             Consider a smaller model or a machine with more memory.",
            q2_size as f64 / 1e9,
            memory_budget as f64 / 1e9,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hardware(total_gb: u64, available_gb: u64) -> HardwareProfile {
        HardwareProfile {
            chip_model: "Apple M5 Max".to_string(),
            total_memory_bytes: total_gb * 1024 * 1024 * 1024,
            available_memory_bytes: available_gb * 1024 * 1024 * 1024,
            performance_cores: 14,
            efficiency_cores: 4,
            total_cores: 18,
        }
    }

    fn make_fingerprint(param_billions: f64) -> ModelFingerprint {
        ModelFingerprint {
            architecture: "TestModel".to_string(),
            total_params: (param_billions * 1e9) as u64,
            layer_count: 32,
            expert_count: 0,
            attention_types: vec!["attention".to_string()],
            hidden_size: 4096,
            dtype: "bfloat16".to_string(),
            intermediate_size: Some(14336),
            num_attention_heads: 32,
            num_kv_heads: Some(8),
            vocab_size: 128256,
        }
    }

    #[test]
    fn test_small_model_on_large_machine_gets_f16() {
        // 3B model on 128GB machine — should easily fit at f16
        let hw = make_hardware(128, 100);
        let fp = make_fingerprint(3.0);

        let result = select_quant(&hw, &fp).unwrap();
        assert_eq!(result.quant_method, "f16");
        assert_eq!(result.bits, 16);
        assert!(result.confidence >= 0.8);
    }

    #[test]
    fn test_medium_model_on_large_machine_gets_f16() {
        // 8B model (~16GB at f16) on 128GB machine — fits with generous headroom
        let hw = make_hardware(128, 100);
        let fp = make_fingerprint(8.0);

        let result = select_quant(&hw, &fp).unwrap();
        assert_eq!(result.quant_method, "f16");
    }

    #[test]
    fn test_model_fits_tight_gets_q8() {
        // 27B model (~54GB at f16) on 128GB machine with 70GB available
        // f16 * 1.8 = 97.2 > 89.6 (budget) -> not generous
        // f16 * 1.3 = 70.2 < 89.6 -> fits tight -> q8
        let hw = make_hardware(128, 70);
        let fp = make_fingerprint(27.0);

        let result = select_quant(&hw, &fp).unwrap();
        assert_eq!(result.quant_method, "q8");
    }

    #[test]
    fn test_large_model_moderate_memory_gets_mixed() {
        // 27B model on 64GB machine
        // f16 = 54GB, f16 * 1.3 = 70.2 > 44.8 (budget=max(30,44.8)=44.8) -> no
        // q8 = 27GB, q8 * 1.3 = 35.1 < 44.8 -> fits -> mixed-4-6
        let hw = make_hardware(64, 30);
        let fp = make_fingerprint(27.0);

        let result = select_quant(&hw, &fp).unwrap();
        assert_eq!(result.quant_method, "mixed-4-6");
    }

    #[test]
    fn test_large_model_small_memory_gets_q4() {
        // 27B model on 36GB machine
        // budget = max(20, 25.2) = 25.2 GB
        // q8 = 27GB, q8 * 1.3 = 35.1 > 25.2 -> no
        // q4 = 13.5GB, q4 * 1.3 = 17.55 < 25.2 -> fits -> q4
        let hw = make_hardware(36, 20);
        let fp = make_fingerprint(27.0);

        let result = select_quant(&hw, &fp).unwrap();
        assert_eq!(result.quant_method, "q4");
    }

    #[test]
    fn test_huge_model_tiny_memory_gets_q2() {
        // 70B model on 36GB machine
        // budget = max(20, 25.2) = 25.2 GB
        // q4 = 35GB, q4 * 1.3 = 45.5 > 25.2 -> no
        // q2 = 17.5GB, q2 * 1.3 = 22.75 < 25.2 -> fits -> q2
        let hw = make_hardware(36, 20);
        let fp = make_fingerprint(70.0);

        let result = select_quant(&hw, &fp).unwrap();
        assert_eq!(result.quant_method, "q2");
    }

    #[test]
    fn test_model_too_large_errors() {
        // 405B model on 36GB machine — doesn't fit even at q2
        // q2 = ~101GB, q2 * 1.3 = 131 > 25.2 -> nope
        let hw = make_hardware(36, 20);
        let fp = make_fingerprint(405.0);

        let result = select_quant(&hw, &fp);
        assert!(result.is_err());
    }

    #[test]
    fn test_confidence_decreases_with_more_quantization() {
        let hw = make_hardware(128, 100);

        // Small model: f16 with high confidence
        let fp_small = make_fingerprint(3.0);
        let r_small = select_quant(&hw, &fp_small).unwrap();

        // Larger model that needs more quantization
        let fp_large = make_fingerprint(70.0);
        let r_large = select_quant(&hw, &fp_large).unwrap();

        assert!(r_small.confidence >= r_large.confidence);
    }

    #[test]
    fn test_reasoning_is_populated() {
        let hw = make_hardware(128, 100);
        let fp = make_fingerprint(8.0);

        let result = select_quant(&hw, &fp).unwrap();
        assert!(!result.reasoning.is_empty());
    }
}
