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
//! Architectures that do not yet expose a splice site are refused here
//! with a named error (the staged-site rollout, ADR-059) — a graft never
//! binds to a family whose cache medium it was not built for.
//!
//! 3. **The `compressed_kv` site bind (ADR-059, #308)** — DeepSeek-V4 only:
//!    the model's own per-layer `compress_ratios` schedule defines the
//!    covered layers (`compress_ratios[layer] != 0` — those layers own a
//!    compressed region), the bank's `graft.compress_ratios` array must be
//!    parallel and equal on every covered layer, `n_kv_heads` must be 1
//!    (shared-KV MQA), `head_dim` must match, the RoPE keys are validated
//!    against the model's COMPRESSOR rope identity for CHECKPOINT IDENTITY
//!    ONLY (compressed rows are merged, positionless state — no
//!    `position_base` semantics at this site), every value must be exactly
//!    bf16-representable (the splice encodes through the cache's BF16
//!    rows — exported bf16-rounded F32 values splice losslessly), and
//!    `n_slots` must fit the compressed region's capacity
//!    (`context_length / ratio` rows, binding at the max covered ratio)
//!    minus the real history the conversation needs (the per-request
//!    capacity accounting counts the graft rows).

use std::path::Path;

use anyhow::{bail, Context, Result};
use mlx_native::gguf::{GgufFile, MetadataValue};

use crate::inference::glp::{CheckpointIdentity, Compatibility};

use super::reader::{GraftBank, GraftHookPoint};

/// Round-to-nearest-even bf16 encoding of an f32, as bit pattern. This is
/// the exact conversion the DeepSeek-V4 compressed rows' BF16 storage
/// performs: a value that is already bf16-representable encodes to itself,
/// so the #309 producer's exported bf16-rounded F32 bank values splice
/// losslessly.
fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits + 0x7FFF + ((bits >> 16) & 1);
    (rounded >> 16) as u16
}

/// A bank value is splice-lossless iff it round-trips through the bf16
/// encode exactly (the compressed rows' storage dtype).
fn is_bf16_representable(value: f32) -> bool {
    if !value.is_finite() {
        return false;
    }
    let bits = f32_to_bf16_bits(value);
    f32::from_bits(u32::from(bits) << 16).to_bits() == value.to_bits()
}

/// The `compressed_kv` site's view of a DeepSeek-V4 model GGUF (ADR-059
/// #308): the covered layers, the per-layer compression schedule, and the
/// geometry/identity a fabricated-compressed-row bank must match.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressedKvShape {
    pub arch: String,
    /// The model's per-layer compression schedule, parallel to the bank's
    /// `graft.compress_ratios` array.
    pub compress_ratios: Vec<u32>,
    /// Graph indices of covered layers (`compress_ratios[layer] != 0` —
    /// exactly the layers that own a compressed region), ascending.
    pub covered_layers: Vec<u32>,
    /// DeepSeek-V4 MLA is shared-KV MQA: exactly one KV head.
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Checkpoint-identity RoPE (the compressor's rope space): the
    /// compress rope theta and the rope head dim. Identity only —
    /// compressed rows are merged, positionless state and no position
    /// semantics applies at this site.
    pub rope_theta: f32,
    pub rotary_dim: u32,
    /// The model's maximum context — the capacity bound source
    /// (`context_length / ratio` compressed rows per covered layer).
    pub context_length: u32,
}

impl CompressedKvShape {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow::anyhow!("model GGUF missing 'general.architecture'"))?
            .to_owned();
        if arch != "deepseek4" {
            bail!(
                "graft bind: architecture {arch:?} does not expose the \
                 compressed_kv graft site in this build (the site is \
                 DeepSeek-V4 only, ADR-059); refusing to bind"
            );
        }
        // Reuse the strict DeepSeek-V4 config parse — the validated
        // per-layer schedule (0/4/128, one per layer), the MLA head
        // geometry, and the compressor rope identity come from the same
        // single source of truth the loader uses.
        let cfg = crate::inference::models::deepseek4::Deepseek4Config::from_gguf(gguf)
            .context("graft bind: DeepSeek-V4 model config")?;
        let covered_layers: Vec<u32> = cfg
            .compress_ratios
            .iter()
            .enumerate()
            .filter(|(_, ratio)| **ratio != 0)
            .map(|(layer, _)| layer as u32)
            .collect();
        anyhow::ensure!(
            !covered_layers.is_empty(),
            "graft bind: deepseek4 model exposes no compressed layers \
             (compress_ratios are all zero); the compressed_kv site has no \
             cache medium on this model"
        );
        Ok(Self {
            arch,
            compress_ratios: cfg.compress_ratios.clone(),
            covered_layers,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rope_theta: cfg.compress_rope_theta,
            rotary_dim: cfg.rope_head_dim,
            context_length: cfg.max_position_embeddings,
        })
    }
}

/// Structural bind of a `compressed_kv` bank against the DeepSeek-V4 site
/// shape (ADR-059 #308 consumer items 1–2).
///
/// A zero-slot canary skips the tensor-geometry checks (it carries none)
/// but still validates the site schedule and the architecture — a canary
/// on the compressed site binds THIS checkpoint's schedule.
pub fn validate_compressed_kv_bank_for_model(
    bank: &GraftBank,
    model: &CompressedKvShape,
) -> Result<()> {
    // The schedule is parallel to the model's per-layer array and must be
    // equal on every layer (the gemma4 per-layer-array precedent): the
    // splice reserves the leading rows of every covered layer, so a
    // divergent schedule is a wrong-checkpoint bank.
    let bank_ratios = bank.compress_ratios.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "graft bind: graft.compress_ratios missing; the compressed_kv \
             site requires the per-layer compression schedule parallel to \
             the model's"
        )
    })?;
    if bank_ratios.len() != model.compress_ratios.len()
        || bank_ratios
            .iter()
            .zip(model.compress_ratios.iter())
            .any(|(b, m)| b != m)
    {
        bail!(
            "graft bind: graft.compress_ratios [{}] does not match the model's \
             per-layer schedule [{}] on {} — the compressed_kv splice reserves \
             the leading rows of every covered layer; a divergent schedule is \
             a wrong-checkpoint bank",
            bank_ratios
                .iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join(","),
            model
                .compress_ratios
                .iter()
                .map(|r| r.to_string())
                .collect::<Vec<_>>()
                .join(","),
            model.arch
        );
    }
    if bank.n_slots == 0 {
        // Canary: no tensors to validate; the site schedule above still
        // binds.
        return Ok(());
    }
    // A tensor at a ratio-0 layer is fatal: those layers own no compressed
    // region (sliding-attention layers keep their history only in the
    // 128-token circular window).
    let offenders: Vec<String> = bank
        .layers
        .keys()
        .filter(|layer| {
            model
                .compress_ratios
                .get(**layer as usize)
                .is_none_or(|ratio| *ratio == 0)
        })
        .map(|layer| layer.to_string())
        .collect();
    if !offenders.is_empty() {
        bail!(
            "graft bind: bank covers layer(s) [{}] with compress_ratios[layer]==0 \
             on {} — those layers own no compressed region (no long-range \
             history); the compressed_kv site splices only covered \
             (ratio != 0) layers",
            offenders.join(", "),
            model.arch
        );
    }
    // Complete site coverage (the full_attn_kv contract mirrors): the bank
    // must carry fabricated rows for EVERY covered layer.
    let missing: Vec<String> = model
        .covered_layers
        .iter()
        .filter(|l| !bank.layers.contains_key(l))
        .map(|l| l.to_string())
        .collect();
    if !missing.is_empty() {
        bail!(
            "graft bind: bank does not cover compressed layer(s) [{}] on {} \
             — the compressed_kv site requires complete coverage of every \
             covered (ratio != 0) layer (partial-coverage banks are not a \
             defined v1 artifact)",
            missing.join(", "),
            model.arch
        );
    }
    if bank.n_kv_heads != model.n_kv_heads as usize {
        bail!(
            "graft bind: bank n_kv_heads {} does not match model {}",
            bank.n_kv_heads,
            model.n_kv_heads
        );
    }
    if model.n_kv_heads != 1 {
        bail!(
            "graft bind: deepseek4 reports {} KV heads; the compressed_kv \
             site is shared-KV MQA (one K==V series) and binds only 1",
            model.n_kv_heads
        );
    }
    if bank.head_dim != model.head_dim as usize {
        bail!(
            "graft bind: bank head_dim {} does not match model {}",
            bank.head_dim,
            model.head_dim
        );
    }
    // RoPE identity — CHECKPOINT IDENTITY ONLY. Compressed rows are
    // merged, positionless state: the bank's compressor-rope identity
    // (theta = compress_rope_freq_base, rotary = the rope head dim) pins
    // the checkpoint it was derived on; no position semantics applies.
    let theta_delta = (bank.rope_theta - model.rope_theta).abs();
    let theta_scale = bank
        .rope_theta
        .abs()
        .max(model.rope_theta.abs())
        .max(f32::EPSILON);
    if !(theta_delta <= 1e-6 * theta_scale) {
        bail!(
            "graft bind: bank rope_theta {} does not match the model's \
             compress rope theta {} (checkpoint identity of the \
             compressed_kv site)",
            bank.rope_theta,
            model.rope_theta
        );
    }
    if bank.rotary_dim != model.rotary_dim {
        bail!(
            "graft bind: bank rotary_dim {} does not match the model's rope \
             head dim {} (checkpoint identity of the compressed_kv site)",
            bank.rotary_dim,
            model.rotary_dim
        );
    }
    // The static capacity bound: n_slots must fit the compressed region of
    // EVERY covered layer. The binding layer is the max covered ratio
    // (fewest rows). The real-history headroom beyond this is enforced by
    // the per-request capacity accounting, which counts the graft rows.
    let binding_capacity = model
        .compress_ratios
        .iter()
        .zip(model.covered_layers.iter())
        .filter(|(ratio, _)| **ratio != 0)
        .map(|(ratio, _)| model.context_length / *ratio)
        .min()
        .unwrap_or(0);
    if bank.n_slots as u64 > binding_capacity as u64 {
        bail!(
            "graft bind: graft n_slots {} exceeds the compressed-region \
             capacity {} rows (context {}/max covered ratio) — the graft \
             reserves leading rows the real history must not overwrite",
            bank.n_slots,
            binding_capacity,
            model.context_length
        );
    }
    // dtype: the splice encodes through the cache's BF16 rows (the same
    // storage every real prefill-compressed row uses), so every exported
    // F32 value must be exactly bf16-representable — the #309 to-gguf
    // contract rounds the bank through bf16 before export; anything else
    // would splice LOSSY and is refused here rather than silently encoded.
    for (layer, kv) in &bank.layers {
        for value in kv.k.iter().chain(kv.v.iter()) {
            if !is_bf16_representable(*value) {
                bail!(
                    "graft bind: layer {layer} bank value {value} is not \
                     bf16-representable — the compressed_kv splice encodes \
                     through the cache's BF16 rows; export bf16-rounded \
                     values (the #309 to-gguf contract) or the splice is lossy"
                );
            }
        }
    }
    Ok(())
}

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
            "gemma4" => {
                let required_u32 = |key: &str| -> Result<u32> {
                    gguf.metadata_u32(key).with_context(|| {
                        format!("graft bind: model GGUF missing required key '{key}'")
                    })
                };
                let block_count = required_u32("gemma4.block_count")?;
                // Layer types: `sliding_window_pattern` is a per-layer
                // bool array, True = sliding, False = full (the
                // serve/config.rs convention). Fallback when absent:
                // every 6th layer is full (the default pattern).
                let full_attn_layers: Vec<u32> = match gguf
                    .metadata("gemma4.attention.sliding_window_pattern")
                {
                    Some(MetadataValue::Array(arr)) => {
                        let pattern: Vec<bool> = arr
                            .iter()
                            .filter_map(|v| match v {
                                MetadataValue::Bool(b) => Some(*b),
                                _ => None,
                            })
                            .collect();
                        anyhow::ensure!(
                            pattern.len() == block_count as usize,
                            "graft bind: gemma4.attention.sliding_window_pattern length {} \
                             != block_count {}",
                            pattern.len(),
                            block_count
                        );
                        (0..block_count)
                            .filter(|i| !pattern[*i as usize])
                            .collect()
                    }
                    _ => (0..block_count).filter(|i| (i + 1) % 6 == 0).collect(),
                };
                anyhow::ensure!(
                    !full_attn_layers.is_empty(),
                    "graft bind: gemma4 model exposes no full-attention layers; the \
                     full_attn_kv site has no cache medium on this model"
                );
                // Per-layer KV head counts (i32 array, one per layer).
                // The site geometry is the FULL-attention layers' count —
                // same derivation as serve/config.rs (the first full
                // layer's value); sliding layers use a different head
                // count AND head dim and are not part of this site.
                let head_count_kv: Vec<u32> = match gguf
                    .metadata("gemma4.attention.head_count_kv")
                {
                    Some(MetadataValue::Array(arr)) => {
                        arr.iter().filter_map(|v| v.as_u32()).collect()
                    }
                    Some(other) => bail!(
                        "graft bind: gemma4.attention.head_count_kv has unexpected \
                         type {:?}",
                        std::mem::discriminant(other)
                    ),
                    None => bail!("graft bind: gemma4.attention.head_count_kv missing"),
                };
                anyhow::ensure!(
                    head_count_kv.len() == block_count as usize,
                    "graft bind: gemma4.attention.head_count_kv length {} != block_count {}",
                    head_count_kv.len(),
                    block_count
                );
                let first_full = full_attn_layers[0] as usize;
                let n_kv_heads = head_count_kv[first_full];
                anyhow::ensure!(
                    n_kv_heads > 0,
                    "graft bind: gemma4 full-attention layer {} reports 0 KV heads",
                    first_full
                );
                Ok(Self {
                    n_kv_heads,
                    // Global (full-attention) geometry — NOT the _swa
                    // variants, which describe sliding layers.
                    head_dim: required_u32("gemma4.attention.key_length")?,
                    rope_theta: gguf
                        .metadata_f32("gemma4.rope.freq_base")
                        .with_context(|| {
                            "graft bind: model GGUF missing 'gemma4.rope.freq_base'"
                        })?,
                    rotary_dim: required_u32("gemma4.rope.dimension_count")?,
                    // Gemma-4 uses standard RoPE; IMROPE interleaving is
                    // a Qwen3.5-family convention.
                    mrope_interleaved: false,
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
            "graft bind: site {:?} is not the full_attn_kv site; this bind \
             validates full_attn_kv banks on {} (the compressed_kv site \
             validates through CompressedKvShape)",
            bank.hook_point.as_str(),
            model.arch
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
    // Site dispatch (ADR-059 site matrix): the bank's hook point selects
    // the site bind; each site owns its model families and refuses every
    // other architecture by name — a graft never binds to a family whose
    // cache medium it was not built for.
    match bank.hook_point {
        GraftHookPoint::FullAttnKv => {
            let shape = GraftModelShape::from_gguf(model)?;
            validate_graft_bank_for_model(&bank, &shape)?;
        }
        GraftHookPoint::CompressedKv => {
            let shape = CompressedKvShape::from_gguf(model)?;
            validate_compressed_kv_bank_for_model(&bank, &shape)?;
        }
        staged => bail!(
            "graft bind: site {:?} is a staged site with no family in this \
             build (ADR-059 site matrix); refusing to bind",
            staged.as_str()
        ),
    }
    Ok((bank, compatibility))
}

#[cfg(test)]
#[path = "compatibility_tests.rs"]
mod tests;
