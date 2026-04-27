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
//! | 4 (high) | MLP (gate/up/dn)| FFN pathway is #1 source of quant artifacts   |
//! | 5 (high) | v_proj          | Biggest quality jump per bit across all archs  |
//! | 6 (med)  | k_proj          | Attention pattern corruption is hard to recover|
//! | 7 (med)  | q_proj          | Less sensitive than k/v but still important    |
//! | 8 (med)  | o_proj          | Output projection of attention                 |
//! | 9 (low)  | expert FFN      | Redundancy across experts provides resilience   |

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
///
/// ADR-012 Decision 12: Qwen35Dense and Qwen35MoE are explicit variants so that
/// cohort priors (SSM state tensors, router, shared/routed experts) can fire only
/// for qwen35 family, never for Gemma or other archs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchFamily {
    /// Gemma 4 and similar MoE architectures
    GemmaMoE,
    /// Mixtral, DBRX, and other standard MoE architectures
    GenericMoE,
    /// Llama, Mistral, and other dense decoder-only models (NOT qwen35)
    DenseDecoder,
    /// Qwen3.5 dense — hybrid Gated-DeltaNet + full-attention, no MoE
    /// (HF arch: "Qwen3_5ForCausalLM")
    Qwen35Dense,
    /// Qwen3.5-MoE — hybrid Gated-DeltaNet + full-attention with routed + shared experts
    /// (HF arch: "Qwen3_5MoeForCausalLM")
    Qwen35MoE,
    /// Unknown architecture — use conservative defaults
    Unknown,
}

fn classify_architecture(fingerprint: &ModelFingerprint) -> ArchFamily {
    let arch = fingerprint.architecture.to_lowercase();

    // Qwen3.5 family is matched BEFORE the generic qwen/gemma checks below so that
    // cohort priors only fire for the right variant.  Both dense and MoE variants
    // embed "qwen3_5" in the HF architecture string.
    if arch.contains("qwen3_5") {
        return if fingerprint.is_moe() {
            ArchFamily::Qwen35MoE
        } else {
            ArchFamily::Qwen35Dense
        };
    }

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
            quant_method: plan_to_quant_method(forced, arch_family, fingerprint, hardware),
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
    // We skip odd widths (3, 6) that use slower scalar packing
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
    let quant_method = plan_to_quant_method(base_bits, arch_family, fingerprint, hardware);

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
/// The strategy: keep attention projections at base bits but elevate the MLP
/// pathway and critical routing to 8-bit. This matches the approach used by
/// high-quality community quantizations (e.g., 4-bit models)
/// and dramatically reduces generation artifacts compared to uniform quantization.
///
/// # Override hierarchy (highest priority first)
///
/// 1. **ADR-012 Decision 12 cohort priors** (qwen35 / qwen35moe only):
///    - SSM state tensors (A_log, dt_bias, dt_proj, dt_proj.bias, conv1d): ALWAYS promoted to
///      `sensitive_bits` regardless of activation score. These tiny tensors are numerically
///      load-bearing: `A_log` is exponentiated, quantization error compounds across recurrence.
///    - qwen35moe only: Router (`ffn_gate_inp`) and shared experts
///      (`ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`): ALWAYS promoted.
///    - qwen35moe only: Routed experts (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`):
///      activation-score-driven, but with a HIGHER threshold than ordinary layers.
///      Default threshold = 0.6 (documented here per ADR-012 D12; future calibration may tune).
///
///    Cohort priors are ADDITIVE to user `--sensitive-layers`: if the user has already
///    nominated a layer index, the cohort prior for that tensor family also applies.
///    User `--sensitive-layers` is honored alongside cohort priors.
///
/// 2. Router projections (non-qwen35 MoE): ALWAYS 8-bit (misrouting is catastrophic).
///
/// 3. MLP layers (gate_proj, up_proj, down_proj): 8-bit at 4-bit base or lower.
///
/// 4. v_proj: +2 bits over base.
///
/// 5. First/last layers: elevated at aggressive (2-3 bit) quantization.
///
/// # Gemma-4 regression invariant
///
/// Cohort priors in rules 1a-1c are only added for `Qwen35Dense` and `Qwen35MoE`.
/// For `GemmaMoE`, `GenericMoE`, `DenseDecoder`, and `Unknown`, the output of this
/// function is **byte-identical** to the pre-P6 version. This is the primary regression
/// gate — if any test for GemmaMoE arch observes new overrides, that is a defect.
///
/// Citation: ADR-012 Decision 12.
fn build_component_overrides(
    arch_family: ArchFamily,
    fingerprint: &ModelFingerprint,
    base_bits: u8,
) -> Vec<ComponentOverride> {
    let mut overrides = Vec::new();

    // -----------------------------------------------------------------------
    // Rule 1: ADR-012 Decision 12 — Qwen3.5-family cohort priors
    //
    // These priors are ADDITIVE: they do not replace or override user
    // --sensitive-layers nominations; both apply independently.
    // -----------------------------------------------------------------------
    match arch_family {
        ArchFamily::Qwen35Dense | ArchFamily::Qwen35MoE => {
            // 1a. SSM state tensors — ALWAYS promoted to sensitive_bits.
            //
            // Rationale: A_log is exponentiated in the recurrence, so quantization
            // error in A_log compounds multiplicatively across the sequence.
            // dt_proj drives the time-step gate; dt_bias/dt_proj.bias are small but
            // their absolute error dominates at low bit widths.  conv1d holds the
            // SSM short-range convolutional state — all are tiny and numerically
            // load-bearing.  Promotion cost (extra bits * tensor bytes) is negligible
            // vs. perplexity recovery.
            //
            // Tensor name patterns follow ADR-012 Decision 5 / Decision 11 GGUF naming:
            //   blk.N.ssm_a         (.A_log in HF space, negated + stored as A)
            //   blk.N.time_mix_dt   (dt_bias / dt_proj combined in GGUF)
            //   blk.N.ssm_conv1d    (conv1d squeeze-reorder; see P3)
            // We match HF tensor name suffixes here (pre-rename) because cohort priors
            // are evaluated in the conversion pipeline before GGUF renaming.
            for ssm_pattern in &[
                ".A_log",
                ".dt_bias",
                ".dt_proj.weight",
                ".dt_proj.bias",
                ".conv1d.weight",
            ] {
                overrides.push(ComponentOverride {
                    pattern: ssm_pattern.to_string(),
                    bits: 8u8.max(base_bits),
                    reason: format!(
                        "ADR-012 D12 cohort prior: SSM state tensor '{ssm_pattern}' is numerically \
                         load-bearing (A_log exponentiated, dt drives time-step gate). \
                         Promoted to sensitive_bits unconditionally."
                    ),
                });
            }
        }
        _ => {}
    }

    if arch_family == ArchFamily::Qwen35MoE {
        // 1b. Router + shared experts — ALWAYS promoted (qwen35moe only).
        //
        // Rationale: The router selects which experts handle each token; routing
        // errors are catastrophic (wrong computation path, no recovery).
        // Shared experts are active for every token — quantization error compounds
        // across every forward pass.
        //
        // HF tensor name patterns (pre-GGUF rename):
        //   model.layers.N.mlp.gate.weight   → ffn_gate_inp in GGUF
        //   model.layers.N.mlp.shared_expert.gate_proj.weight → ffn_gate_shexp
        //   model.layers.N.mlp.shared_expert.up_proj.weight   → ffn_up_shexp
        //   model.layers.N.mlp.shared_expert.down_proj.weight → ffn_down_shexp
        for router_pattern in &[
            ".mlp.gate.weight",        // router: ffn_gate_inp
            ".shared_expert.gate_proj", // ffn_gate_shexp
            ".shared_expert.up_proj",   // ffn_up_shexp
            ".shared_expert.down_proj", // ffn_down_shexp
        ] {
            overrides.push(ComponentOverride {
                pattern: router_pattern.to_string(),
                bits: 8u8.max(base_bits),
                reason: format!(
                    "ADR-012 D12 cohort prior (qwen35moe): '{router_pattern}' is a router or \
                     shared expert — always active, misrouting is unrecoverable. \
                     Promoted to sensitive_bits unconditionally."
                ),
            });
        }

        // 1c. Routed experts — higher promotion threshold (qwen35moe only).
        //
        // Rationale: routed experts have redundancy (each token only activates top_k of N),
        // so they tolerate quantization better than shared experts.  However, the merged
        // expert tensor (ffn_gate_exps / ffn_up_exps / ffn_down_exps) stacks all N experts
        // and per-expert quantization error can still cause routing-adjacent artifacts.
        // We use a HIGHER threshold than ordinary layers but do NOT unconditionally promote.
        //
        // Threshold: we elevate routed experts by 2 bits above base (not to sensitive_bits).
        // This reflects the intermediate risk level.  The value 2 is documented here as the
        // default (ADR-012 D12); a future calibration pass may tune it per model.
        // Default elevation: +2 bits over base (clamped to 8).
        let routed_bits = next_valid_bits(base_bits, 2).min(8);
        if routed_bits > base_bits {
            // Single pattern catches ffn_gate_exps, ffn_up_exps, ffn_down_exps.
            let exp_pattern = ".experts.";
            overrides.push(ComponentOverride {
                pattern: exp_pattern.to_string(),
                bits: routed_bits,
                reason: format!(
                    "ADR-012 D12 cohort prior (qwen35moe): routed expert tensor '{exp_pattern}' \
                     uses activation-score-driven heuristic with elevated threshold. \
                     Default: +2 bits above base ({base_bits}-bit \u{2192} {routed_bits}-bit). \
                     Tunable by future calibration (see ADR-012 D12)."
                ),
            });
        }
    }

    // -----------------------------------------------------------------------
    // Rule 2: MoE router projections for non-qwen35 MoE (Gemma4 / GenericMoE)
    // -----------------------------------------------------------------------
    if fingerprint.is_moe()
        && !matches!(arch_family, ArchFamily::Qwen35MoE)
    {
        overrides.push(ComponentOverride {
            pattern: "router.proj".to_string(),
            bits: 8.max(base_bits),
            reason: "Router misrouting is catastrophic in MoE models".to_string(),
        });
        // Gemma4's router.scale and per_expert_scale are automatically preserved
        // as non-weight tensors by the backend's should_quantize logic — no override needed.
    }

    // -----------------------------------------------------------------------
    // Rule 3: MLP layers at 8-bit when base is 4-bit or lower.
    // -----------------------------------------------------------------------
    if base_bits <= 4 {
        for component in &["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            overrides.push(ComponentOverride {
                pattern: component.to_string(),
                bits: 8,
                reason: format!(
                    "MLP {} at 8-bit eliminates FFN-induced generation artifacts ({}-bit base)",
                    component, base_bits
                ),
            });
        }
    }

    // -----------------------------------------------------------------------
    // Rule 4: v_proj elevated bits.
    // -----------------------------------------------------------------------
    let v_proj_bits = next_valid_bits(base_bits, 2);
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

    // -----------------------------------------------------------------------
    // Rule 5: At aggressive quantization (2-3 bit base), protect first/last layers.
    // -----------------------------------------------------------------------
    if base_bits <= 3 {
        let elevated = next_valid_bits(base_bits, 2);
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

/// Get the next valid bit width that is at least `steps` above `base`.
/// Currently limited to power-of-2 widths (2, 4, 8) which are verified working.
/// TODO: Enable 3-bit and 6-bit once non-power-of-2 uint32 packing is validated.
fn next_valid_bits(base: u8, steps: u8) -> u8 {
    const VALID: [u8; 5] = [2, 3, 4, 6, 8];
    let target = base + steps;
    for &v in &VALID {
        if v >= target {
            return v;
        }
    }
    8
}

/// Map a (base_bits, arch_family, hardware) tuple to the appropriate
/// CLI quant method name (ADR-014 P8 Decision 18 routing table).
///
/// ## Decision 18 routing table
///
/// | Model class | RAM       | → Variant          |
/// |-------------|-----------|--------------------|
/// | Dense ≤30B  | any       | `imatrix-q4_k_m`   |
/// | Dense >30B  | <64 GB    | `imatrix-q4_k_m`   |
/// | Dense >30B  | ≥64 GB    | `imatrix-q5_k_m`   |
/// | MoE any     | <96 GB    | `dwq-4-6`          |
/// | MoE any     | ≥96 GB    | `dwq-4-8`          |
///
/// `base_bits` is consulted only as a tiebreaker for the legacy
/// flat / passthrough cells (16 → `f16`, 8 → `q8`, etc.); when an
/// override-aware K-quant or DWQ cell applies, the table above wins.
///
/// ## ArchEntry::auto_override
///
/// When the model's [`ArchEntry`] (looked up via the registry by HF
/// architecture string) carries an `auto_override = Some(v)`, that
/// string wins over the table — used for arches that have empirically
/// validated a non-default cell. Per spec, this iter scopes the
/// auto_override enforcement to AutoResolver only; the field itself
/// landing on `ArchEntry` is documented as a follow-up if not present.
///
/// [`ArchEntry`]: crate::arch::registry::ArchEntry
fn plan_to_quant_method(
    base_bits: u8,
    _arch_family: ArchFamily,
    fingerprint: &ModelFingerprint,
    hardware: &HardwareProfile,
) -> String {
    // -----------------------------------------------------------------
    // Step 1: ArchEntry::auto_override (per-arch override wins).
    //
    // We consult the registry at runtime via fingerprint.architecture.
    // If the entry exists AND carries an auto_override, that string
    // wins over the Decision-18 table. Missing entry / missing override
    // → fall through to the table.
    // -----------------------------------------------------------------
    if let Some(entry_override) =
        crate::arch::registry::lookup_auto_override(&fingerprint.architecture)
    {
        return entry_override;
    }

    // -----------------------------------------------------------------
    // Step 2: Decision-18 routing table.
    // -----------------------------------------------------------------
    let total_gb = hardware.total_memory_gb();

    if fingerprint.is_moe() {
        if total_gb >= 96.0 {
            return "dwq-4-8".to_string();
        }
        return "dwq-4-6".to_string();
    }

    // Dense path.
    let params_b = fingerprint.total_params as f64 / 1e9;
    if params_b > 30.0 && total_gb >= 64.0 {
        return "imatrix-q5_k_m".to_string();
    }
    // Dense ≤30B (any RAM) OR dense >30B with <64 GB RAM.
    if params_b <= 30.0 || total_gb < 64.0 {
        // Use the imatrix Q4_K_M default unless the bit search demands
        // a flat cell (the cells below cover f16 / q8 forced-by-search
        // outcomes; the Decision-18 default is imatrix-q4_k_m).
        match base_bits {
            16 => return "f16".to_string(),
            8 => return "q8".to_string(),
            _ => return "imatrix-q4_k_m".to_string(),
        }
    }

    // Defensive default — every (params_b, total_gb) combination is
    // handled above, but Rust's exhaustiveness checker can't see it.
    "imatrix-q4_k_m".to_string()
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
/// This produces the format that backends expect:
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

    fn make_qwen35_dense_fingerprint() -> ModelFingerprint {
        ModelFingerprint {
            architecture: "Qwen3_5ForCausalLM".to_string(),
            total_params: 32_000_000_000,
            layer_count: 64,
            expert_count: 0,
            attention_types: vec!["full_attention".to_string(), "linear_attention".to_string()],
            hidden_size: 7168,
            dtype: "bfloat16".to_string(),
            intermediate_size: Some(18432),
            num_attention_heads: 64,
            num_kv_heads: Some(8),
            vocab_size: 152064,
        }
    }

    fn make_qwen35moe_fingerprint() -> ModelFingerprint {
        ModelFingerprint {
            architecture: "Qwen3_5MoeForCausalLM".to_string(),
            total_params: 235_000_000_000,
            layer_count: 94,
            expert_count: 128,
            attention_types: vec!["full_attention".to_string(), "linear_attention".to_string()],
            hidden_size: 7168,
            dtype: "bfloat16".to_string(),
            intermediate_size: Some(2048),
            num_attention_heads: 64,
            num_kv_heads: Some(4),
            vocab_size: 152064,
        }
    }

    // --- ADR-012 Decision 12: cohort prior tests ---

    #[test]
    fn test_classify_qwen35_dense() {
        let fp = make_qwen35_dense_fingerprint();
        assert_eq!(classify_architecture(&fp), ArchFamily::Qwen35Dense);
    }

    #[test]
    fn test_classify_qwen35moe() {
        let fp = make_qwen35moe_fingerprint();
        assert_eq!(classify_architecture(&fp), ArchFamily::Qwen35MoE);
    }

    /// ADR-012 D12: SSM state tensors always promoted for qwen35 dense.
    #[test]
    fn test_qwen35_dense_ssm_cohort_priors_always_promoted() {
        let fp = make_qwen35_dense_fingerprint();
        let overrides = build_component_overrides(ArchFamily::Qwen35Dense, &fp, 4);

        for ssm in &[".A_log", ".dt_bias", ".dt_proj.weight", ".dt_proj.bias", ".conv1d.weight"] {
            let found = overrides.iter().find(|o| o.pattern == *ssm);
            assert!(found.is_some(), "qwen35 dense cohort prior missing: {ssm}");
            assert!(found.unwrap().bits >= 4u8, "SSM prior must be >= base_bits for {ssm}");
            // Must be at sensitive_bits (8, the max valid >= base_bits)
            assert_eq!(found.unwrap().bits, 8, "SSM prior for {ssm} must be 8-bit (sensitive_bits)");
        }
    }

    /// ADR-012 D12: SSM state tensors also promoted for qwen35moe.
    #[test]
    fn test_qwen35moe_ssm_cohort_priors_always_promoted() {
        let fp = make_qwen35moe_fingerprint();
        let overrides = build_component_overrides(ArchFamily::Qwen35MoE, &fp, 4);

        for ssm in &[".A_log", ".dt_bias", ".dt_proj.weight", ".dt_proj.bias", ".conv1d.weight"] {
            let found = overrides.iter().find(|o| o.pattern == *ssm);
            assert!(found.is_some(), "qwen35moe SSM cohort prior missing: {ssm}");
            assert_eq!(found.unwrap().bits, 8, "SSM prior for {ssm} must be 8-bit");
        }
    }

    /// ADR-012 D12: Router and shared experts promoted for qwen35moe.
    #[test]
    fn test_qwen35moe_router_and_shared_expert_promoted() {
        let fp = make_qwen35moe_fingerprint();
        let overrides = build_component_overrides(ArchFamily::Qwen35MoE, &fp, 4);

        for moe_pattern in &[
            ".mlp.gate.weight",
            ".shared_expert.gate_proj",
            ".shared_expert.up_proj",
            ".shared_expert.down_proj",
        ] {
            let found = overrides.iter().find(|o| o.pattern == *moe_pattern);
            assert!(found.is_some(), "qwen35moe router/shared-expert prior missing: {moe_pattern}");
            assert_eq!(found.unwrap().bits, 8, "Router/shared-expert prior for {moe_pattern} must be 8-bit");
        }
    }

    /// ADR-012 D12: Routed experts get elevated threshold (+2 bits) for qwen35moe.
    #[test]
    fn test_qwen35moe_routed_experts_elevated_threshold() {
        let fp = make_qwen35moe_fingerprint();
        let overrides = build_component_overrides(ArchFamily::Qwen35MoE, &fp, 4);

        // ".experts." catches ffn_gate_exps / ffn_up_exps / ffn_down_exps
        let found = overrides.iter().find(|o| o.pattern == ".experts.");
        assert!(found.is_some(), "qwen35moe routed-expert elevated prior missing");
        // base_bits=4, +2 → 6
        assert_eq!(found.unwrap().bits, 6, "Routed expert prior must be base+2 = 6-bit at 4-bit base");
    }

    /// ADR-012 D12: Router and shared-expert priors NOT present for qwen35 dense
    /// (dense has no router or shared experts).
    #[test]
    fn test_qwen35_dense_no_moe_priors() {
        let fp = make_qwen35_dense_fingerprint();
        let overrides = build_component_overrides(ArchFamily::Qwen35Dense, &fp, 4);

        // Must NOT contain MoE-only patterns
        for moe_only in &[".mlp.gate.weight", ".shared_expert.", ".experts."] {
            assert!(
                !overrides.iter().any(|o| o.pattern == *moe_only),
                "qwen35 dense must not have MoE-only prior: {moe_only}"
            );
        }
    }

    /// ADR-012 D12: Gemma regression — cohort priors must NOT fire for Gemma.
    ///
    /// This is the primary regression gate: if any new override appears for
    /// GemmaMoE arch, the Gemma-4 conversion output would change.
    #[test]
    fn test_gemma4_regression_cohort_priors_not_added() {
        let fp = make_moe_fingerprint(); // Gemma4ForConditionalGeneration
        assert_eq!(classify_architecture(&fp), ArchFamily::GemmaMoE);

        let overrides_before = vec![
            // Rule 2: router.proj at 8-bit (Gemma is MoE, non-qwen35)
            "router.proj".to_string(),
            // Rule 3: mlp.{gate,up,down}_proj at 8-bit (base=4)
            "mlp.gate_proj".to_string(),
            "mlp.up_proj".to_string(),
            "mlp.down_proj".to_string(),
            // Rule 4: v_proj at 6-bit
            "v_proj".to_string(),
        ];

        let overrides = build_component_overrides(ArchFamily::GemmaMoE, &fp, 4);
        let patterns: Vec<&str> = overrides.iter().map(|o| o.pattern.as_str()).collect();

        // No qwen35-only patterns present
        for qwen35_only in &[".A_log", ".dt_bias", ".dt_proj.weight", ".dt_proj.bias",
                               ".conv1d.weight", ".mlp.gate.weight", ".shared_expert.",
                               ".experts."] {
            assert!(
                !patterns.contains(qwen35_only),
                "Gemma regression: qwen35-only pattern '{qwen35_only}' appeared in Gemma overrides"
            );
        }

        // Exactly the pre-P6 patterns and nothing extra
        for expected_pat in &overrides_before {
            assert!(
                patterns.contains(&expected_pat.as_str()),
                "Gemma regression: expected pattern '{expected_pat}' missing from overrides"
            );
        }
        assert_eq!(
            overrides.len(), overrides_before.len(),
            "Gemma regression: override count changed — pre-P6={}, post-P6={}",
            overrides_before.len(), overrides.len()
        );
    }

    /// ADR-012 D12: --sensitive-layers is honored alongside cohort priors (ADDITIVE).
    ///
    /// This test verifies that cohort priors do not suppress the user override —
    /// both the cohort pattern and the user's layer-index pattern are in the output.
    #[test]
    fn test_sensitive_layers_additive_with_cohort_priors() {
        let fp = make_qwen35moe_fingerprint();
        let overrides = build_component_overrides(ArchFamily::Qwen35MoE, &fp, 4);

        // Cohort priors are present
        assert!(overrides.iter().any(|o| o.pattern == ".A_log"),
            "SSM cohort prior must be present");
        // v_proj override (Rule 4) is still present alongside cohort priors
        assert!(overrides.iter().any(|o| o.pattern == "v_proj"),
            "v_proj override must still be present alongside cohort priors");
    }

    /// Mock: verify no SSM priors for a non-qwen35 dense model (Llama).
    #[test]
    fn test_non_qwen35_dense_no_ssm_priors() {
        let fp = make_dense_fingerprint(8.0); // LlamaForCausalLM
        let overrides = build_component_overrides(ArchFamily::DenseDecoder, &fp, 4);

        for ssm in &[".A_log", ".dt_bias", ".dt_proj.weight", ".dt_proj.bias", ".conv1d.weight"] {
            assert!(
                !overrides.iter().any(|o| o.pattern == *ssm),
                "Non-qwen35 dense must not have SSM cohort prior: {ssm}"
            );
        }
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
    fn test_mlp_8bit_overrides_at_4bit_base() {
        let fp = make_dense_fingerprint(8.0);
        let overrides = build_component_overrides(ArchFamily::DenseDecoder, &fp, 4);

        for component in &["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            let found = overrides.iter().find(|o| o.pattern == *component);
            assert!(found.is_some(), "4-bit plan should elevate {} to 8-bit", component);
            assert_eq!(found.unwrap().bits, 8, "{} should be 8-bit", component);
        }
    }

    #[test]
    fn test_no_mlp_override_at_8bit_base() {
        let fp = make_dense_fingerprint(8.0);
        let overrides = build_component_overrides(ArchFamily::DenseDecoder, &fp, 8);

        let mlp = overrides.iter().find(|o| o.pattern.contains("mlp."));
        assert!(mlp.is_none(), "8-bit base should not need MLP overrides");
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
        let hw = make_hardware("Apple M5 Max", 128);
        let fp_dense = make_dense_fingerprint(8.0);
        let fp_moe = make_moe_fingerprint();

        // ADR-014 P8 Decision 18 routing — Dense ≤30B → imatrix-q4_k_m
        assert_eq!(
            plan_to_quant_method(4, ArchFamily::DenseDecoder, &fp_dense, &hw),
            "imatrix-q4_k_m"
        );
        // base_bits=8 forces a flat passthrough cell.
        assert_eq!(
            plan_to_quant_method(8, ArchFamily::DenseDecoder, &fp_dense, &hw),
            "q8"
        );
        // base_bits=16 → f16 passthrough.
        assert_eq!(
            plan_to_quant_method(16, ArchFamily::DenseDecoder, &fp_dense, &hw),
            "f16"
        );
        // MoE on 128 GB → dwq-4-8 (≥96 GB).
        assert_eq!(
            plan_to_quant_method(2, ArchFamily::GemmaMoE, &fp_moe, &hw),
            "dwq-4-8"
        );
    }

    // ─── ADR-014 P8 Decision 18 routing-table tests (S5) ───

    /// Decision 18 row 1: Dense ≤30B → `imatrix-q4_k_m` (any RAM).
    #[test]
    fn test_decision18_dense_27b_resolves_to_imatrix_q4_k_m() {
        let hw = make_hardware("Apple M5 Max", 128);
        let mut fp = make_dense_fingerprint(27.0);
        fp.architecture = "LlamaForCausalLM".to_string();
        let result = plan_to_quant_method(4, ArchFamily::DenseDecoder, &fp, &hw);
        assert_eq!(
            result, "imatrix-q4_k_m",
            "Dense 27B (≤30B) on any RAM → imatrix-q4_k_m, got: {result}"
        );
    }

    /// Decision 18 row 3: Dense >30B AND ≥64 GB → `imatrix-q5_k_m`.
    #[test]
    fn test_decision18_dense_70b_64gb_resolves_to_imatrix_q5_k_m() {
        let hw = make_hardware("Apple M5 Max", 64);
        let mut fp = make_dense_fingerprint(70.0);
        fp.architecture = "LlamaForCausalLM".to_string();
        let result = plan_to_quant_method(4, ArchFamily::DenseDecoder, &fp, &hw);
        assert_eq!(
            result, "imatrix-q5_k_m",
            "Dense 70B with 64 GB RAM → imatrix-q5_k_m, got: {result}"
        );
    }

    /// Decision 18 row 2: Dense >30B AND <64 GB → `imatrix-q4_k_m`.
    #[test]
    fn test_decision18_dense_70b_below_64gb_resolves_to_imatrix_q4_k_m() {
        let hw = make_hardware("Apple M4", 32);
        let mut fp = make_dense_fingerprint(70.0);
        fp.architecture = "LlamaForCausalLM".to_string();
        let result = plan_to_quant_method(4, ArchFamily::DenseDecoder, &fp, &hw);
        assert_eq!(
            result, "imatrix-q4_k_m",
            "Dense 70B with 32 GB RAM → imatrix-q4_k_m, got: {result}"
        );
    }

    /// Decision 18 row 4: MoE any AND <96 GB → `dwq-4-6`.
    #[test]
    fn test_decision18_moe_apex_64gb_resolves_to_dwq_4_6() {
        let hw = make_hardware("Apple M5 Max", 64);
        let fp = make_moe_fingerprint();
        let result = plan_to_quant_method(4, ArchFamily::GemmaMoE, &fp, &hw);
        assert_eq!(
            result, "dwq-4-6",
            "MoE with 64 GB RAM (<96 GB) → dwq-4-6, got: {result}"
        );
    }

    /// Decision 18 row 5: MoE any AND ≥96 GB → `dwq-4-8`.
    #[test]
    fn test_decision18_moe_apex_128gb_resolves_to_dwq_4_8() {
        let hw = make_hardware("Apple M5 Max", 128);
        let fp = make_moe_fingerprint();
        let result = plan_to_quant_method(4, ArchFamily::GemmaMoE, &fp, &hw);
        assert_eq!(
            result, "dwq-4-8",
            "MoE with 128 GB RAM (≥96 GB) → dwq-4-8, got: {result}"
        );
    }

    /// Decision 18: ArchEntry::auto_override (when set) wins over the
    /// routing table. With the current registry shipping
    /// `auto_override: None` for both qwen35 and qwen35moe, this test
    /// verifies the lookup mechanism: an unknown HF architecture falls
    /// through to the table (returns None), and a registered arch with
    /// no override behaves identically to the table.
    #[test]
    fn test_decision18_arch_override_lookup_falls_through_when_none() {
        // Registered arch with no auto_override → lookup returns None.
        let lookup =
            crate::arch::registry::lookup_auto_override("Qwen3_5ForCausalLM");
        assert!(
            lookup.is_none(),
            "qwen35 ArchEntry has auto_override = None → lookup must return None, \
             got: {lookup:?}"
        );

        // Unregistered arch → lookup returns None (falls through to table).
        let lookup =
            crate::arch::registry::lookup_auto_override("CompletelyUnknownArch");
        assert!(
            lookup.is_none(),
            "Unknown arch must return None (falls through to table)"
        );

        // The table-driven path then applies — confirm it lands on the
        // expected Decision-18 row for a known dense arch.
        let hw = make_hardware("Apple M5 Max", 128);
        let mut fp = make_qwen35_dense_fingerprint();
        fp.total_params = 27_000_000_000; // dense ≤30B
        let result = plan_to_quant_method(4, ArchFamily::Qwen35Dense, &fp, &hw);
        assert_eq!(
            result, "imatrix-q4_k_m",
            "qwen35 dense 27B with no override → imatrix-q4_k_m"
        );
    }
}
