//! Graft-to-model bind validation (ADR-059 gate 2), independent of
//! device allocation.
//!
//! Two halves, both fail-closed:
//!
//! 1. **Checkpoint identity** — the graft GGUF's `general.base_model.0.*`
//!    keys validated through the exact `CheckpointIdentity` trust boundary
//!    GLP uses (ADR-053): output-verified conversion receipts supply the
//!    served checkpoint's source identity for hf2q conversions, model-card
//!    ancestors never do, conflicts are fatal, and a declared revision the
//!    model cannot verify is rejected. No duplicated identity logic.
//! 2. **Structural bind** — the `full_attn_kv` site contract against the
//!    model GGUF's own config: every bank layer must be a full-attention
//!    layer of the served model, `n_kv_heads`/`head_dim` must match the
//!    model's GQA geometry, and the bank's RoPE identity (`rope_theta`,
//!    `rotary_dim`, `mrope_interleaved`, `position_base`) must match the
//!    model's — cache K is post-RoPE, so a RoPE mismatch means the bank's
//!    keys live in a different rotation space than the model's queries.
//!
//! Architectures that do not yet expose the `full_attn_kv` splice site are
//! refused here with a named error (the staged-site rollout, ADR-059) —
//! a graft never binds to a family whose cache medium it was not built
//! for.

use std::path::Path;

use anyhow::{bail, Context, Result};
use mlx_native::gguf::GgufFile;

use crate::inference::glp::{CheckpointIdentity, Compatibility};

use super::reader::{GraftBank, GraftHookPoint};

/// The `full_attn_kv` site's view of a model GGUF: which layers the site
/// covers and the geometry/RoPE identity a bank must match.
#[derive(Debug, Clone, PartialEq)]
pub struct GraftModelShape {
    pub arch: String,
    /// Graph indices of full-attention layers (0-based), ascending.
    pub full_attn_layers: Vec<u32>,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
    pub rotary_dim: u32,
    pub mrope_interleaved: bool,
}

impl GraftModelShape {
    /// Extract the site shape from a model GGUF. Supported families:
    /// `qwen35` / `qwen35moe` (the v1 splice family). Everything else is
    /// refused with a named error until its splice ships.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow::anyhow!("model GGUF missing 'general.architecture'"))?
            .to_owned();
        match arch.as_str() {
            "qwen35" | "qwen35moe" => {
                let required_u32 = |key: &str| -> Result<u32> {
                    gguf.metadata_u32(key).with_context(|| {
                        format!("graft bind: model GGUF missing required key '{key}'")
                    })
                };
                let block_count = required_u32(&format!("{arch}.block_count"))?;
                let interval = required_u32(&format!("{arch}.full_attention_interval"))?;
                // Same convention as qwen35::default_layer_types: layer i
                // is full attention iff (i + 1) % interval == 0.
                let full_attn_layers: Vec<u32> = if interval == 0 {
                    Vec::new()
                } else {
                    (0..block_count)
                        .filter(|i| (i + 1) % interval == 0)
                        .collect()
                };
                if full_attn_layers.is_empty() {
                    bail!(
                        "graft bind: {arch} model exposes no full-attention layers \
                         (full_attention_interval={interval}); the full_attn_kv site \
                         has no cache medium on this model"
                    );
                }
                Ok(Self {
                    n_kv_heads: required_u32(&format!("{arch}.attention.head_count_kv"))?,
                    head_dim: required_u32(&format!("{arch}.attention.key_length"))?,
                    rope_theta: gguf
                        .metadata_f32(&format!("{arch}.rope.freq_base"))
                        .with_context(|| {
                            format!("graft bind: model GGUF missing '{arch}.rope.freq_base'")
                        })?,
                    rotary_dim: required_u32(&format!("{arch}.rope.dimension_count"))?,
                    // Qwen3.5-family convention: IMROPE interleaved mode is
                    // always on (GGML_ROPE_TYPE_IMROPE; no metadata key).
                    mrope_interleaved: true,
                    arch,
                    full_attn_layers,
                })
            }
            other => bail!(
                "graft bind: architecture {other:?} does not expose the \
                 full_attn_kv graft site in this build (staged per ADR-059); \
                 refusing to bind"
            ),
        }
    }
}

/// Structural bind of a loaded bank against the site shape.
///
/// A zero-slot canary bank skips the tensor-geometry checks (it carries
/// none) but still requires a supported architecture — a canary on an
/// unsupported family proves nothing about that family's splice.
pub fn validate_graft_bank_for_model(bank: &GraftBank, model: &GraftModelShape) -> Result<()> {
    if bank.hook_point != GraftHookPoint::FullAttnKv {
        bail!(
            "graft bind: site {:?} is not implemented for any family in this build",
            bank.hook_point.as_str()
        );
    }
    if bank.position_base != 0 {
        bail!(
            "graft bind: graft.position_base {} is unsupported; the v1 \
             splice_prefix contract occupies positions 0..n_slots",
            bank.position_base
        );
    }
    if bank.n_slots == 0 {
        // Canary: no tensors to validate.
        return Ok(());
    }
    let offenders: Vec<String> = bank
        .layers
        .keys()
        .filter(|layer| !model.full_attn_layers.contains(layer))
        .map(|layer| layer.to_string())
        .collect();
    if !offenders.is_empty() {
        bail!(
            "graft bind: bank covers non-full-attention layer(s) [{}] on {} \
             (full-attention layers: [{}]); the full_attn_kv site only splices \
             full-attention-layer caches",
            offenders.join(", "),
            model.arch,
            model
                .full_attn_layers
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    // Complete site coverage: the bank must carry rows for EVERY
    // full-attention layer. Partial coverage would splice graft rows at
    // positions 0..N on some layers and nothing on others — per-layer
    // cursor divergence on a cache whose production invariant is
    // homogeneous cursors, plus per-layer position semantics that the
    // v1 splice_prefix contract does not define. Dose-by-layer-subset is
    // a spec-v2 question, not a silent option.
    let missing: Vec<String> = model
        .full_attn_layers
        .iter()
        .filter(|l| !bank.layers.contains_key(l))
        .map(|l| l.to_string())
        .collect();
    if !missing.is_empty() {
        bail!(
            "graft bind: bank does not cover full-attention layer(s) [{}] on {} \
             — the full_attn_kv site requires complete coverage of every \
             full-attention layer (partial-coverage banks are not a defined \
             v1 artifact)",
            missing.join(", "),
            model.arch
        );
    }
    if bank.n_kv_heads != model.n_kv_heads as usize {
        bail!(
            "graft bind: bank n_kv_heads {} does not match model {}",
            bank.n_kv_heads, model.n_kv_heads
        );
    }
    if bank.head_dim != model.head_dim as usize {
        bail!(
            "graft bind: bank head_dim {} does not match model {}",
            bank.head_dim, model.head_dim
        );
    }
    // RoPE identity: cache K is post-RoPE, so the bank's rotation space
    // must be the model's. GGUF f32 round-trips exactly, but producers
    // may reformat; compare with a tight relative tolerance.
    let theta_delta = (bank.rope_theta - model.rope_theta).abs();
    let theta_scale = bank.rope_theta.abs().max(model.rope_theta.abs()).max(f32::EPSILON);
    if !(theta_delta <= 1e-6 * theta_scale) {
        bail!(
            "graft bind: bank rope_theta {} does not match model {}",
            bank.rope_theta, model.rope_theta
        );
    }
    if bank.rotary_dim != model.rotary_dim {
        bail!(
            "graft bind: bank rotary_dim {} does not match model {}",
            bank.rotary_dim, model.rotary_dim
        );
    }
    if bank.mrope_interleaved != model.mrope_interleaved {
        bail!(
            "graft bind: bank mrope_interleaved {} does not match model {}",
            bank.mrope_interleaved, model.mrope_interleaved
        );
    }
    Ok(())
}

/// Full loader-boundary validation of a graft artifact against a served
/// model: loads and conformance-checks the bank, validates checkpoint
/// identity through the GLP trust boundary, and structurally binds it to
/// the model's site shape. Returns the loaded bank and the identity
/// compatibility level (callers warn below `Checkpoint`).
pub fn validate_graft_for_model(
    graft_path: &Path,
    model_path: &Path,
    model: &GgufFile,
) -> Result<(GraftBank, Compatibility)> {
    let bank = GraftBank::load(graft_path).context("load graft bank")?;
    let graft_gguf = GgufFile::open(graft_path).context("open graft GGUF metadata")?;
    let compatibility = CheckpointIdentity::from_gguf(&graft_gguf)?
        .validate_for(&CheckpointIdentity::for_model_path(model_path, model)?)?;
    let shape = GraftModelShape::from_gguf(model)?;
    validate_graft_bank_for_model(&bank, &shape)?;
    Ok((bank, compatibility))
}

#[cfg(test)]
#[path = "compatibility_tests.rs"]
mod tests;
