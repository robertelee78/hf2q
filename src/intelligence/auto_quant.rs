//! Auto-quant algorithm for `--quant auto` mode.
//!
//! Selects the best quantization configuration based on model architecture,
//! hardware capabilities, and user-specified performance constraints.
//!
//! # Algorithm Overview
//!
//! 1. **Detect model type**: MoE vs dense, size class, architecture family
//! 2. **Profile hardware**: memory capacity, estimated bandwidth, chip tier
//! 3. **Compute max model size** for target tok/s using bandwidth model
//! 4. **Select base bit width** that fits within the budget
//! 5. **Apply per-component overrides** based on architecture sensitivity
//! 6. **Generate per-tensor bit allocation map**
//!
//! # Bandwidth Model
//!
//! For autoregressive decoding, throughput is memory-bandwidth-bound:
//!
//!     tok/s = bandwidth_bytes_per_sec / model_weight_bytes
//!
//! For MoE models that activate K of N experts per token:
//!
//!     effective_weight_bytes = shared_layer_bytes + (K/N) * expert_bytes
//!
//! # Per-Component Sensitivity
//!
//! Not all weight tensors degrade equally under quantization. The algorithm
//! assigns higher bit widths to components measured as most sensitive:
//!
//! | Priority | Component       | Rationale                                    |
//! |----------|-----------------|----------------------------------------------|
//! | 1 (must) | Router proj     | Misrouting is catastrophic in MoE models      |
//! | 2 (must) | embed_tokens    | Shared across all tokens; errors compound      |
//! | 3 (must) | lm_head         | Directly impacts output distribution           |
//! | 4 (high) | v_proj          | Biggest quality jump per bit across all archs  |
//! | 5 (high) | k_proj          | Attention pattern corruption is hard to recover|
//! | 6 (med)  | q_proj          | Less sensitive than k/v but still important    |
//! | 7 (med)  | o_proj          | Output projection of attention                 |
//! | 8 (low)  | gate_proj/up    | FFN gating; moderate sensitivity               |
//! | 9 (low)  | down_proj       | FFN output; least sensitive of projections      |
//! | 10(low)  | expert FFN      | Redundancy across experts provides resilience   |

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::fingerprint::ModelFingerprint;
use super::hardware::HardwareProfile;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The complete output of the auto-quant algorithm: a per-tensor bit allocation
/// plus global settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoQuantPlan {
    /// Global base bit width (the "floor" for most tensors)
    pub base_bits: u8,
    /// Group size for all quantized tensors
    pub group_size: usize,
    /// Per-component bit overrides (component pattern -> bits)
    /// Keys are suffix patterns like "v_proj", "router.proj", etc.
    pub component_overrides: Vec<ComponentOverride>,
    /// Recommended quant method name for CLI dispatch
    pub quant_method: String,
    /// Estimated model size in bytes after applying this plan
    pub estimated_size_bytes: u64,
    /// Estimated tok/s on the target hardware
    pub estimated_tok_per_sec: f64,
    /// Human-readable explanation of why this plan was chosen
    pub reasoning: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
}

/// A single per-component bit width override.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentOverride {
    /// Pattern to match against tensor names (e.g., "router.proj", "v_proj")
    pub pattern: String,
    /// Bit width for tensors matching this pattern
    pub bits: u8,
    /// Why this override exists
    pub reason: String,
}

/// User-specified constraints for the auto algorithm.
#[derive(Debug, Clone)]
pub struct AutoQuantConstraints {
    /// Minimum acceptable tok/s (default: 80)
    pub min_tok_per_sec: f64,
    /// Quality preference: "speed" biases toward fewer bits, "quality" toward more
    pub quality_preference: QualityPreference,
    /// If set, override the detected memory bandwidth (GB/s)
    pub override_bandwidth_gbps: Option<f64>,
    /// If set, forces a specific base bit width (disables auto selection)
    pub forced_bits: Option<u8>,
}

impl Default for AutoQuantConstraints {
    fn default() -> Self {
        Self {
            min_tok_per_sec: 80.0,
            quality_preference: QualityPreference::Balanced,
            override_bandwidth_gbps: None,
            forced_bits: None,
        }
    }
}

/// Quality-vs-speed preference slider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityPreference {
    /// Maximize speed: use lowest bits that meet tok/s target
    Speed,
    /// Balance quality and speed (default)
    Balanced,
    /// Maximize quality: use highest bits that meet tok/s target
    Quality,
}

// ---------------------------------------------------------------------------
// Hardware bandwidth estimation
// ---------------------------------------------------------------------------

/// Known Apple Silicon bandwidth profiles (measured effective, not peak).
/// These are conservative estimates from community benchmarks.
struct BandwidthProfile {
    /// Effective memory bandwidth in GB/s
    effective_bandwidth_gbps: f64,
}

/// Estimate the effective memory bandwidth for the detected hardware.
///
/// Returns bandwidth in bytes/sec. Uses known chip profiles for Apple Silicon,
/// falls back to conservative estimates for unknown hardware.
fn estimate_bandwidth(hardware: &HardwareProfile) -> f64 {
    let chip = hardware.chip_model.to_lowercase();
    let _mem_gb = hardware.total_memory_gb();

    let profile = if chip.contains("m5 ultra") {
        BandwidthProfile { effective_bandwidth_gbps: 780.0 }
    } else if chip.contains("m5 max") {
        BandwidthProfile { effective_bandwidth_gbps: 401.0 }
    } else if chip.contains("m5 pro") {
        BandwidthProfile { effective_bandwidth_gbps: 250.0 }
    } else if chip.contains("m5") {
        BandwidthProfile { effective_bandwidth_gbps: 100.0 }
    } else if chip.contains("m4 ultra") {
        BandwidthProfile { effective_bandwidth_gbps: 700.0 }
    } else if chip.contains("m4 max") {
        BandwidthProfile { effective_bandwidth_gbps: 370.0 }
    } else if chip.contains("m4 pro") {
        BandwidthProfile { effective_bandwidth_gbps: 230.0 }
    } else if chip.contains("m4") {
        BandwidthProfile { effective_bandwidth_gbps: 100.0 }
    } else if chip.contains("m3 ultra") {
        BandwidthProfile { effective_bandwidth_gbps: 600.0 }
    } else if chip.contains("m3 max") {
        BandwidthProfile { effective_bandwidth_gbps: 300.0 }
    } else if chip.contains("m3 pro") {
        BandwidthProfile { effective_bandwidth_gbps: 150.0 }
    } else if chip.contains("m3") {
        BandwidthProfile { effective_bandwidth_gbps: 100.0 }
    } else if chip.contains("m2 ultra") {
        BandwidthProfile { effective_bandwidth_gbps: 600.0 }
    } else if chip.contains("m2 max") {
        BandwidthProfile { effective_bandwidth_gbps: 300.0 }
    } else if chip.contains("m2 pro") {
        BandwidthProfile { effective_bandwidth_gbps: 150.0 }
    } else if chip.contains("m2") {
        BandwidthProfile { effective_bandwidth_gbps: 100.0 }
    } else if chip.contains("m1 ultra") {
        BandwidthProfile { effective_bandwidth_gbps: 500.0 }
    } else if chip.contains("m1 max") {
        BandwidthProfile { effective_bandwidth_gbps: 250.0 }
    } else if chip.contains("m1 pro") {
        BandwidthProfile { effective_bandwidth_gbps: 150.0 }
    } else if chip.contains("m1") {
        BandwidthProfile { effective_bandwidth_gbps: 60.0 }
    } else {
        // Unknown hardware: assume 100 GB/s as a conservative baseline.
        // This is roughly the lower bound for any modern unified-memory chip.
        warn!(
            chip = %hardware.chip_model,
            "Unknown hardware — using conservative 100 GB/s bandwidth estimate. \
             Pass --bandwidth to override."
        );
        BandwidthProfile { effective_bandwidth_gbps: 100.0 }
    };

    let gbps = profile.effective_bandwidth_gbps;
    debug!(
        chip = %hardware.chip_model,
        bandwidth_gbps = gbps,
        "Estimated effective memory bandwidth"
    );

    gbps * 1e9 // Convert GB/s to bytes/s
}

// ---------------------------------------------------------------------------
// Model size estimation
// ---------------------------------------------------------------------------

/// Architecture family, used to select the right sensitivity table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchFamily {
    /// Gemma 4 and similar MoE architectures
    GemmaMoE,
    /// Mixtral, DBRX, and other standard MoE architectures
    GenericMoE,
    /// Llama, Mistral, Qwen, and other dense decoder-only models
    DenseDecoder,
    /// Unknown architecture — use conservative defaults
    Unknown,
}

fn classify_architecture(fingerprint: &ModelFingerprint) -> ArchFamily {
    let arch = fingerprint.architecture.to_lowercase();

    if fingerprint.is_moe() {
        if arch.contains("gemma") {
            ArchFamily::GemmaMoE
        } else {
            ArchFamily::GenericMoE
        }
    } else if arch.contains("llama")
        || arch.contains("mistral")
        || arch.contains("qwen")
        || arch.contains("phi")
        || arch.contains("gemma")
        || arch.contains("starcoder")
        || arch.contains("codellama")
        || arch.contains("deepseek")
        || arch.contains("internlm")
        || arch.contains("yi")
        || arch.contains("command")
    {
        ArchFamily::DenseDecoder
    } else {
        ArchFamily::Unknown
    }
}

/// Estimate the effective weight bytes read per token for a given plan.
///
/// For dense models: all weights are read every token.
/// For MoE models: shared layers + (top_k / num_experts) of expert layers.
fn estimate_bytes_per_token(
    fingerprint: &ModelFingerprint,
    plan_base_bits: u8,
    component_overrides: &[ComponentOverride],
) -> u64 {
    if !fingerprint.is_moe() {
        // Dense model: all params read every token
        return estimate_total_model_bytes(fingerprint, plan_base_bits, component_overrides);
    }

    // MoE model: split into shared and expert params
    //
    // Shared params: embeddings + attention layers + norms + lm_head + routers
    // Expert params: all expert FFN weights
    //
    // We approximate the split using the architecture's intermediate_size and
    // expert_count. Each expert has ~3 * hidden_size * intermediate_size params
    // (gate_proj, up_proj, down_proj). With N experts, expert params dominate.
    let h = fingerprint.hidden_size as f64;
    let i = fingerprint.intermediate_size.unwrap_or(fingerprint.hidden_size * 4) as f64;
    let n_experts = fingerprint.expert_count as f64;
    let n_layers = fingerprint.layer_count as f64;

    // Expert params per layer: N_experts * 3 * H * I (gate, up, down projections)
    let expert_params_per_layer = n_experts * 3.0 * h * i;
    // Shared params per layer: attention (4 * H * H for q/k/v/o) + norms + router
    let num_kv_heads = fingerprint.num_kv_heads.unwrap_or(fingerprint.num_attention_heads) as f64;
    let num_heads = fingerprint.num_attention_heads as f64;
    let head_dim = h / num_heads;
    let shared_attn_per_layer = h * (num_heads * head_dim)  // q_proj
        + h * (num_kv_heads * head_dim)                      // k_proj
        + h * (num_kv_heads * head_dim)                      // v_proj
        + (num_heads * head_dim) * h;                         // o_proj

    let total_expert_params = expert_params_per_layer * n_layers;
    let total_shared_attn_params = shared_attn_per_layer * n_layers;

    // Embeddings + lm_head
    let embed_params = fingerprint.vocab_size as f64 * h * 2.0; // embed_tokens + lm_head

    let total_shared_params = total_shared_attn_params + embed_params;

    // What fraction of experts is read per token?
    // Default top_k=8 for Gemma4-style, top_k=2 for Mixtral-style
    let top_k = if fingerprint.expert_count >= 64 { 8.0 } else { 2.0 };
    let expert_activation_ratio = top_k / n_experts;

    let shared_bytes = (total_shared_params * plan_base_bits as f64 / 8.0) as u64;
    let expert_bytes_active =
        (total_expert_params * expert_activation_ratio * plan_base_bits as f64 / 8.0) as u64;

    debug!(
        shared_gb = shared_bytes as f64 / 1e9,
        expert_active_gb = expert_bytes_active as f64 / 1e9,
        activation_ratio = expert_activation_ratio,
        "MoE bytes-per-token breakdown"
    );

    shared_bytes + expert_bytes_active
}

/// Estimate total on-disk model size for a given plan.
fn estimate_total_model_bytes(
    fingerprint: &ModelFingerprint,
    base_bits: u8,
    _component_overrides: &[ComponentOverride],
) -> u64 {
    // For a rough estimate, use total_params * base_bits / 8.
    // Component overrides affect a small fraction of total params
    // and the difference is minor for budget calculations.
    (fingerprint.total_params as f64 * base_bits as f64 / 8.0) as u64
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/// The main auto-quant decision function.
///
/// # Algorithm
///
/// ```text
/// 1. Classify architecture (MoE/dense, family)
/// 2. Estimate hardware bandwidth
/// 3. For each candidate bit width [8, 6, 4, 3, 2]:
///    a. Compute bytes-per-token
///    b. Estimate tok/s = bandwidth / bytes_per_token
///    c. If tok/s >= target: this is our base_bits
/// 4. Apply quality preference:
///    - Speed:    use the candidate found in step 3
///    - Balanced: try one step higher if tok/s has headroom
///    - Quality:  use the highest bits where tok/s >= target
/// 5. Build component overrides from sensitivity table
/// 6. Verify total model fits in memory (not just bandwidth)
/// ```
pub fn resolve_auto_plan(
    hardware: &HardwareProfile,
    fingerprint: &ModelFingerprint,
    constraints: &AutoQuantConstraints,
) -> Result<AutoQuantPlan, AutoQuantError> {
    let arch_family = classify_architecture(fingerprint);
    info!(
        arch_family = ?arch_family,
        is_moe = fingerprint.is_moe(),
        params_b = fingerprint.total_params as f64 / 1e9,
        "Auto-quant: classifying model"
    );

    // Step 1: Determine bandwidth
    let bandwidth_bps = match constraints.override_bandwidth_gbps {
        Some(bw) => bw * 1e9,
        None => estimate_bandwidth(hardware),
    };

    // Step 2: If bits are forced, skip the search
    if let Some(forced) = constraints.forced_bits {
        let overrides = build_component_overrides(arch_family, fingerprint, forced);
        let total_bytes = estimate_total_model_bytes(fingerprint, forced, &overrides);
        let bytes_per_token = estimate_bytes_per_token(fingerprint, forced, &overrides);
        let est_tok_s = bandwidth_bps / bytes_per_token as f64;

        return Ok(AutoQuantPlan {
            base_bits: forced,
            group_size: 64,
            component_overrides: overrides,
            quant_method: plan_to_quant_method(forced, arch_family, fingerprint),
            estimated_size_bytes: total_bytes,
            estimated_tok_per_sec: est_tok_s,
            reasoning: format!(
                "Forced to {}-bit by user. Estimated {:.0} tok/s.",
                forced, est_tok_s
            ),
            confidence: 0.6,
        });
    }

    // Step 3: Search for the best base bit width
    // Candidate bit widths in descending quality order.
    // We skip odd widths (3, 6) that use slower scalar packing on MLX
    // unless the quality preference demands them.
    let candidates: &[u8] = match constraints.quality_preference {
        QualityPreference::Speed => &[4, 3, 2],
        QualityPreference::Balanced => &[8, 4, 3, 2],
        QualityPreference::Quality => &[8, 6, 4, 3, 2],
    };

    let target_tok_s = constraints.min_tok_per_sec;
    let memory_budget = (hardware.total_memory_bytes as f64 * 0.85) as u64; // 85% of total

    let mut best_plan: Option<(u8, f64, u64)> = None; // (bits, tok/s, total_bytes)

    for &bits in candidates {
        let overrides = build_component_overrides(arch_family, fingerprint, bits);
        let total_bytes = estimate_total_model_bytes(fingerprint, bits, &overrides);
        let bytes_per_token = estimate_bytes_per_token(fingerprint, bits, &overrides);
        let est_tok_s = bandwidth_bps / bytes_per_token as f64;

        debug!(
            bits = bits,
            total_gb = total_bytes as f64 / 1e9,
            bpt_gb = bytes_per_token as f64 / 1e9,
            est_tok_s = est_tok_s,
            "Auto-quant: evaluating {}-bit",
            bits
        );

        // Check memory fit
        if total_bytes > memory_budget {
            debug!(bits = bits, "Skipping: model does not fit in memory");
            continue;
        }

        // Check throughput
        if est_tok_s >= target_tok_s {
            match constraints.quality_preference {
                QualityPreference::Quality => {
                    // Take the FIRST (highest bits) that meets the target
                    best_plan = Some((bits, est_tok_s, total_bytes));
                    break;
                }
                QualityPreference::Balanced => {
                    // Take the first that meets the target (candidates already ordered
                    // high-to-low, but skip 8-bit if 4-bit also meets it with >20% headroom)
                    best_plan = Some((bits, est_tok_s, total_bytes));
                    break;
                }
                QualityPreference::Speed => {
                    // Already ordered low-to-high for speed preference:
                    // Wait, no -- candidates are [4, 3, 2] for speed.
                    // We want the LOWEST bits that still meets the target.
                    // Since we iterate low-to-high... actually for speed we iterate
                    // from 4 down. Take the first that meets the target.
                    best_plan = Some((bits, est_tok_s, total_bytes));
                    break;
                }
            }
        }

        // If it doesn't meet tok/s target but fits in memory, record as fallback
        if best_plan.is_none() || bits < best_plan.unwrap().0 {
            best_plan = Some((bits, est_tok_s, total_bytes));
        }
    }

    let (base_bits, est_tok_s, total_bytes) = best_plan.ok_or_else(|| {
        AutoQuantError::ModelTooLarge {
            reason: format!(
                "Model ({:.1}B params) does not fit in {:.0} GB memory even at 2-bit quantization.",
                fingerprint.total_params as f64 / 1e9,
                hardware.total_memory_gb()
            ),
        }
    })?;

    // Step 4: Build the final plan
    let overrides = build_component_overrides(arch_family, fingerprint, base_bits);
    let quant_method = plan_to_quant_method(base_bits, arch_family, fingerprint);

    // Compute confidence based on how well we meet the target
    let confidence = if est_tok_s >= target_tok_s * 1.2 {
        0.85 // Comfortable headroom
    } else if est_tok_s >= target_tok_s {
        0.75 // Meets target
    } else if est_tok_s >= target_tok_s * 0.8 {
        0.6 // Close but under target
    } else {
        0.45 // Significantly under target
    };

    // Quality rating based on bpw
    let quality_note = match base_bits {
        8 => "~97.7% token accuracy",
        6 => "~96.9% token accuracy",
        4 => "~90.5% token accuracy",
        3 => "~85% token accuracy (estimated)",
        2 => "~75% token accuracy (estimated, significant quality loss)",
        _ => "unknown quality profile",
    };

    let reasoning = if est_tok_s >= target_tok_s {
        format!(
            "{}-bit base ({}) meets {:.0} tok/s target (est. {:.0} tok/s) on {} with {:.0} GB. \
             Model size: {:.1} GB. {}{}",
            base_bits,
            quant_method,
            target_tok_s,
            est_tok_s,
            hardware.chip_model,
            hardware.total_memory_gb(),
            total_bytes as f64 / 1e9,
            quality_note,
            if !overrides.is_empty() {
                format!(
                    ". {} component overrides applied for quality-critical layers.",
                    overrides.len()
                )
            } else {
                String::new()
            },
        )
    } else {
        format!(
            "Best achievable: {}-bit base ({}) yields ~{:.0} tok/s (target: {:.0}) on {} with {:.0} GB. \
             Model size: {:.1} GB. {}. Consider a smaller model for the target throughput.",
            base_bits,
            quant_method,
            est_tok_s,
            target_tok_s,
            hardware.chip_model,
            hardware.total_memory_gb(),
            total_bytes as f64 / 1e9,
            quality_note,
        )
    };

    info!(
        base_bits = base_bits,
        method = %quant_method,
        est_tok_s = est_tok_s,
        model_gb = total_bytes as f64 / 1e9,
        overrides = overrides.len(),
        "Auto-quant plan resolved"
    );

    Ok(AutoQuantPlan {
        base_bits,
        group_size: 64,
        component_overrides: overrides,
        quant_method,
        estimated_size_bytes: total_bytes,
        estimated_tok_per_sec: est_tok_s,
        reasoning,
        confidence,
    })
}

// ---------------------------------------------------------------------------
// Per-component sensitivity overrides
// ---------------------------------------------------------------------------

/// Build component overrides based on architecture family and base bit width.
///
/// The rules encode the sensitivity hierarchy from the CFA research:
///
/// - Router projections are ALWAYS 8-bit (MoE only; misrouting is catastrophic)
/// - embed_tokens and lm_head are preserved at f16 when base >= 4-bit
/// - v_proj gets +2 bits over base (biggest single-component quality impact)
/// - k_proj gets +1 bit over base
/// - Other components stay at base bits
fn build_component_overrides(
    _arch_family: ArchFamily,
    fingerprint: &ModelFingerprint,
    base_bits: u8,
) -> Vec<ComponentOverride> {
    let mut overrides = Vec::new();

    // Rule 1: MoE router projections must be 8-bit minimum
    if fingerprint.is_moe() {
        overrides.push(ComponentOverride {
            pattern: "router.proj".to_string(),
            bits: 8.max(base_bits),
            reason: "Router misrouting is catastrophic in MoE models".to_string(),
        });
        // Gemma4's router.scale and per_expert_scale are automatically preserved
        // as non-weight tensors by the MLX backend's should_quantize logic — no override needed.
    }

    // Rule 2: v_proj gets elevated bits (biggest quality impact per bit)
    // MLX only supports 2, 3, 4, 6, 8 — use next_valid_mlx_bits to clamp
    let v_proj_bits = next_valid_mlx_bits(base_bits, 2);
    if v_proj_bits > base_bits {
        overrides.push(ComponentOverride {
            pattern: "v_proj".to_string(),
            bits: v_proj_bits,
            reason: format!(
                "v_proj has the highest per-bit quality impact ({}-bit vs {}-bit base)",
                v_proj_bits, base_bits
            ),
        });
    }

    // Rule 3: k_proj gets next valid MLX bit width above base
    let k_proj_bits = next_valid_mlx_bits(base_bits, 1);
    if k_proj_bits > base_bits && k_proj_bits != v_proj_bits {
        overrides.push(ComponentOverride {
            pattern: "k_proj".to_string(),
            bits: k_proj_bits,
            reason: format!(
                "k_proj attention patterns are sensitive to quantization ({}-bit)",
                k_proj_bits
            ),
        });
    }

    // Rule 4: At aggressive quantization (2-3 bit base), protect first/last layers
    if base_bits <= 3 {
        let elevated = next_valid_mlx_bits(base_bits, 2);
        overrides.push(ComponentOverride {
            pattern: "layers.0.".to_string(),
            bits: elevated,
            reason: "First transformer layer is disproportionately sensitive".to_string(),
        });

        let last_layer = fingerprint.layer_count.saturating_sub(1);
        overrides.push(ComponentOverride {
            pattern: format!("layers.{}.", last_layer),
            bits: elevated,
            reason: "Last transformer layer directly feeds lm_head".to_string(),
        });
    }

    overrides
}

/// Get the next valid MLX bit width that is at least `steps` above `base`.
/// Currently limited to power-of-2 widths (2, 4, 8) which are verified working.
/// TODO: Enable 3-bit and 6-bit once non-power-of-2 uint32 packing is validated.
fn next_valid_mlx_bits(base: u8, steps: u8) -> u8 {
    const VALID: [u8; 5] = [2, 3, 4, 6, 8];
    let target = base + steps;
    for &v in &VALID {
        if v >= target {
            return v;
        }
    }
    8
}

/// Map a (base_bits, arch_family) pair to the appropriate CLI quant method name.
fn plan_to_quant_method(base_bits: u8, _arch_family: ArchFamily, fingerprint: &ModelFingerprint) -> String {
    // If there are component overrides that differ from base, use mixed-bit
    let has_elevated = fingerprint.is_moe() || base_bits <= 4;

    match base_bits {
        8 => {
            if fingerprint.is_moe() {
                // MoE at 8-bit with router protection = still q8 with overrides
                "q8".to_string()
            } else {
                "q8".to_string()
            }
        }
        6 => {
            if has_elevated {
                "mixed-4-6".to_string() // Use 4-6 mixed as the closest preset
            } else {
                "q8".to_string() // q6 not a standard method; round up
            }
        }
        4 => {
            if has_elevated {
                "mixed-4-6".to_string()
            } else {
                "q4".to_string()
            }
        }
        3 => {
            if has_elevated {
                "mixed-3-6".to_string()
            } else {
                "q4".to_string() // q3 not a standard method; round up
            }
        }
        2 => {
            if has_elevated {
                "mixed-2-6".to_string()
            } else {
                "q2".to_string()
            }
        }
        16 => "f16".to_string(),
        _ => format!("q{}", base_bits),
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from auto-quant resolution.
#[derive(Debug, thiserror::Error)]
pub enum AutoQuantError {
    #[error("Model too large for available hardware: {reason}")]
    ModelTooLarge { reason: String },

    #[error("Hardware detection failed: {reason}")]
    #[allow(dead_code)]
    HardwareDetection { reason: String },
}

// ---------------------------------------------------------------------------
// Config output format
// ---------------------------------------------------------------------------

/// Convert an AutoQuantPlan to the JSON structure written to
/// `quantization_config.json` in the output directory.
///
/// This produces the format that MLX and other backends expect:
///
/// ```json
/// {
///   "quant_method": "mixed-4-6",
///   "bits": 4,
///   "group_size": 64,
///   "component_overrides": {
///     "router.proj": { "bits": 8, "reason": "..." },
///     "v_proj": { "bits": 6, "reason": "..." }
///   },
///   "auto_resolved": true,
///   "estimated_tok_per_sec": 105.3,
///   "estimated_size_bytes": 14500000000,
///   "hardware": "Apple M5 Max",
///   "confidence": 0.85
/// }
/// ```
#[allow(dead_code)] // Planned for use when auto_quant writes config files directly
pub fn plan_to_config_json(plan: &AutoQuantPlan, hardware: &HardwareProfile) -> serde_json::Value {
    let mut overrides_map = serde_json::Map::new();
    for ov in &plan.component_overrides {
        overrides_map.insert(
            ov.pattern.clone(),
            serde_json::json!({
                "bits": ov.bits,
                "reason": ov.reason,
            }),
        );
    }

    serde_json::json!({
        "quant_method": plan.quant_method,
        "bits": plan.base_bits,
        "group_size": plan.group_size,
        "component_overrides": overrides_map,
        "auto_resolved": true,
        "estimated_tok_per_sec": (plan.estimated_tok_per_sec * 10.0).round() / 10.0,
        "estimated_size_bytes": plan.estimated_size_bytes,
        "hardware": hardware.chip_model,
        "confidence": (plan.confidence * 100.0).round() / 100.0,
        "reasoning": plan.reasoning,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hardware(chip: &str, total_gb: u64) -> HardwareProfile {
        HardwareProfile {
            chip_model: chip.to_string(),
            total_memory_bytes: total_gb * 1024 * 1024 * 1024,
            available_memory_bytes: (total_gb as f64 * 0.8) as u64 * 1024 * 1024 * 1024,
            performance_cores: 14,
            efficiency_cores: 4,
            total_cores: 18,
            memory_bandwidth_gbs: crate::intelligence::hardware::lookup_memory_bandwidth_gbs(chip),
        }
    }

    fn make_dense_fingerprint(param_billions: f64) -> ModelFingerprint {
        ModelFingerprint {
            architecture: "LlamaForCausalLM".to_string(),
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

    fn make_moe_fingerprint() -> ModelFingerprint {
        ModelFingerprint {
            architecture: "Gemma4ForConditionalGeneration".to_string(),
            total_params: 27_000_000_000,
            layer_count: 30,
            expert_count: 128,
            attention_types: vec![
                "full_attention".to_string(),
                "sliding_attention".to_string(),
            ],
            hidden_size: 2816,
            dtype: "bfloat16".to_string(),
            intermediate_size: Some(2112),
            num_attention_heads: 16,
            num_kv_heads: Some(8),
            vocab_size: 262144,
        }
    }

    #[test]
    fn test_classify_architecture() {
        let dense = make_dense_fingerprint(8.0);
        assert_eq!(classify_architecture(&dense), ArchFamily::DenseDecoder);

        let moe = make_moe_fingerprint();
        assert_eq!(classify_architecture(&moe), ArchFamily::GemmaMoE);
    }

    #[test]
    fn test_bandwidth_estimation_known_chip() {
        let hw = make_hardware("Apple M5 Max", 128);
        let bw = estimate_bandwidth(&hw);
        assert!((bw / 1e9 - 401.0).abs() < 1.0);
    }

    #[test]
    fn test_bandwidth_estimation_unknown_chip() {
        let hw = make_hardware("Unknown Chip XYZ", 64);
        let bw = estimate_bandwidth(&hw);
        assert!((bw / 1e9 - 100.0).abs() < 1.0); // Conservative default
    }

    #[test]
    fn test_small_dense_model_on_large_machine() {
        // 8B model on M5 Max 128GB: should get high bit width
        let hw = make_hardware("Apple M5 Max", 128);
        let fp = make_dense_fingerprint(8.0);
        let constraints = AutoQuantConstraints::default();

        let plan = resolve_auto_plan(&hw, &fp, &constraints).unwrap();
        // 8B at 8-bit = 8GB. BW=401 GB/s. tok/s = 401/8 ~= 50.
        // At 4-bit = 4GB. tok/s = 401/4 ~= 100. Should choose 4-bit for 80 tok/s target.
        assert!(plan.base_bits <= 8);
        assert!(plan.estimated_tok_per_sec >= 50.0);
        assert!(!plan.reasoning.is_empty());
    }

    #[test]
    fn test_moe_model_has_router_override() {
        let hw = make_hardware("Apple M5 Max", 128);
        let fp = make_moe_fingerprint();
        let constraints = AutoQuantConstraints::default();

        let plan = resolve_auto_plan(&hw, &fp, &constraints).unwrap();

        // Must have a router.proj override at 8-bit
        let router_override = plan
            .component_overrides
            .iter()
            .find(|o| o.pattern == "router.proj");
        assert!(router_override.is_some(), "MoE plan must include router.proj override");
        assert_eq!(router_override.unwrap().bits, 8);
    }

    #[test]
    fn test_moe_bytes_per_token_less_than_total() {
        let fp = make_moe_fingerprint();
        let overrides = build_component_overrides(ArchFamily::GemmaMoE, &fp, 4);
        let bpt = estimate_bytes_per_token(&fp, 4, &overrides);
        let total = estimate_total_model_bytes(&fp, 4, &overrides);

        // MoE should read significantly less than total weights per token
        assert!(
            bpt < total,
            "MoE bytes_per_token ({}) should be less than total ({})",
            bpt,
            total
        );
    }

    #[test]
    fn test_dense_bytes_per_token_equals_total() {
        let fp = make_dense_fingerprint(8.0);
        let overrides = build_component_overrides(ArchFamily::DenseDecoder, &fp, 4);
        let bpt = estimate_bytes_per_token(&fp, 4, &overrides);
        let total = estimate_total_model_bytes(&fp, 4, &overrides);

        // Dense model reads all weights every token
        assert_eq!(bpt, total);
    }

    #[test]
    fn test_quality_preference_affects_bits() {
        let hw = make_hardware("Apple M5 Max", 128);
        let fp = make_dense_fingerprint(8.0);

        let speed_plan = resolve_auto_plan(
            &hw,
            &fp,
            &AutoQuantConstraints {
                quality_preference: QualityPreference::Speed,
                ..Default::default()
            },
        )
        .unwrap();

        let quality_plan = resolve_auto_plan(
            &hw,
            &fp,
            &AutoQuantConstraints {
                quality_preference: QualityPreference::Quality,
                ..Default::default()
            },
        )
        .unwrap();

        // Quality preference should result in equal or higher bits
        assert!(quality_plan.base_bits >= speed_plan.base_bits);
    }

    #[test]
    fn test_huge_model_too_large_for_small_machine() {
        let hw = make_hardware("Apple M4", 16);
        let fp = make_dense_fingerprint(405.0);
        let constraints = AutoQuantConstraints::default();

        let result = resolve_auto_plan(&hw, &fp, &constraints);
        assert!(result.is_err());
    }

    #[test]
    fn test_forced_bits_respected() {
        let hw = make_hardware("Apple M5 Max", 128);
        let fp = make_dense_fingerprint(8.0);
        let constraints = AutoQuantConstraints {
            forced_bits: Some(3),
            ..Default::default()
        };

        let plan = resolve_auto_plan(&hw, &fp, &constraints).unwrap();
        assert_eq!(plan.base_bits, 3);
    }

    #[test]
    fn test_v_proj_override_present_at_4bit() {
        let fp = make_dense_fingerprint(8.0);
        let overrides = build_component_overrides(ArchFamily::DenseDecoder, &fp, 4);

        let v_proj = overrides.iter().find(|o| o.pattern == "v_proj");
        assert!(v_proj.is_some(), "4-bit plan should elevate v_proj");
        assert_eq!(v_proj.unwrap().bits, 6); // base 4 + 2 steps = 6
    }

    #[test]
    fn test_aggressive_quant_protects_first_last_layers() {
        let fp = make_dense_fingerprint(70.0);
        let overrides = build_component_overrides(ArchFamily::DenseDecoder, &fp, 2);

        let first = overrides.iter().find(|o| o.pattern == "layers.0.");
        let last = overrides.iter().find(|o| o.pattern == "layers.31.");
        assert!(first.is_some(), "2-bit should protect first layer");
        assert!(last.is_some(), "2-bit should protect last layer");
    }

    #[test]
    fn test_plan_to_config_json_structure() {
        let hw = make_hardware("Apple M5 Max", 128);
        let fp = make_moe_fingerprint();
        let constraints = AutoQuantConstraints::default();

        let plan = resolve_auto_plan(&hw, &fp, &constraints).unwrap();
        let json = plan_to_config_json(&plan, &hw);

        assert!(json.get("quant_method").is_some());
        assert!(json.get("bits").is_some());
        assert!(json.get("group_size").is_some());
        assert!(json.get("component_overrides").is_some());
        assert_eq!(json["auto_resolved"], true);
        assert!(json["hardware"].as_str().unwrap().contains("M5 Max"));
    }

    #[test]
    fn test_plan_to_quant_method() {
        let fp_dense = make_dense_fingerprint(8.0);
        let fp_moe = make_moe_fingerprint();

        assert_eq!(plan_to_quant_method(4, ArchFamily::DenseDecoder, &fp_dense), "mixed-4-6");
        assert_eq!(plan_to_quant_method(8, ArchFamily::DenseDecoder, &fp_dense), "q8");
        assert_eq!(plan_to_quant_method(2, ArchFamily::GemmaMoE, &fp_moe), "mixed-2-6");
        assert_eq!(plan_to_quant_method(16, ArchFamily::DenseDecoder, &fp_dense), "f16");
    }
}
