//! Top-level `Qwen35Model` struct — the load target for Qwen3.5 / Qwen3.5-MoE
//! GGUFs.
//!
//! Ties together every component that preceding phases delivered:
//!
//! - [`Qwen35Config`](super::Qwen35Config) from the GGUF metadata parser.
//! - Per-layer weights: `Qwen35LayerWeights` enum (FullAttn or LinearAttn).
//! - FFN variant: `Qwen35FfnWeights` enum (dense SwiGLU or MoE).
//! - Optional [`MtpWeights`](super::mtp::MtpWeights) for MTP speculative decoding.
//! - Global tensors: `token_embd`, `output_weight`, `output_norm`.

use anyhow::{anyhow, Context, Result};

use super::delta_net::DeltaNetLayerWeights;
use super::ffn::{DenseFfnWeights, MoeFfnWeights};
use super::full_attn::FullAttnLayerWeights;
use super::mtp::{load_mtp_weights_if_present, MtpWeights};
use super::weight_loader::{DenseFfnWeightsQ, MoeFfnWeightsQ};
use super::{weight_loader, Qwen35Config, Qwen35LayerKind, Qwen35Variant};

use mlx_native::gguf::GgufFile;
use mlx_native::MlxDevice;

// ================================================================
// Layer weight enums
// ================================================================

/// FFN weights for a single Qwen3.5 layer. Qwen3.5 dense uses SwiGLU;
/// Qwen3.5-MoE uses a 256-expert router with a sigmoid-gated shared
/// expert. Exactly one variant is populated per layer per model.
pub enum Qwen35FfnWeights {
    Dense(DenseFfnWeights),
    /// Quantized dense SwiGLU weights loaded directly from GGUF (production path).
    /// Gate/up/down tensors stay in GGML block format on the Metal device;
    /// no F32 expansion occurs during load. Used for 27B dense DWQ GGUFs.
    DenseQ(DenseFfnWeightsQ),
    /// F32-dequantized MoE weights (unit tests / synthetic models only).
    Moe(MoeFfnWeights),
    /// Quantized MoE weights loaded directly from GGUF (production path).
    /// Expert tensors stay in GGML block format on the Metal device;
    /// no F32 expansion occurs during load.
    MoeQ(MoeFfnWeightsQ),
}

impl Qwen35FfnWeights {
    pub fn variant(&self) -> &'static str {
        match self {
            Qwen35FfnWeights::Dense(_) => "dense",
            Qwen35FfnWeights::DenseQ(_) => "dense-q",
            Qwen35FfnWeights::Moe(_) => "moe",
            Qwen35FfnWeights::MoeQ(_) => "moe-q",
        }
    }
}

/// Per-layer weight bundle. The attention variant is determined by the
/// `Qwen35LayerKind` at the corresponding position in `cfg.layer_types`;
/// the FFN variant is determined by `cfg.variant` (Dense or Moe) and is
/// the same across all layers.
pub enum Qwen35LayerWeights {
    /// Gated full-attention layer (every 4th layer for Qwen3.5).
    FullAttn {
        attn: FullAttnLayerWeights,
        ffn: Qwen35FfnWeights,
    },
    /// Linear-attention (Gated DeltaNet) layer.
    LinearAttn {
        attn: DeltaNetLayerWeights,
        ffn: Qwen35FfnWeights,
    },
}

impl Qwen35LayerWeights {
    pub fn kind(&self) -> Qwen35LayerKind {
        match self {
            Qwen35LayerWeights::FullAttn { .. } => Qwen35LayerKind::FullAttention,
            Qwen35LayerWeights::LinearAttn { .. } => Qwen35LayerKind::LinearAttention,
        }
    }

    pub fn ffn(&self) -> &Qwen35FfnWeights {
        match self {
            Qwen35LayerWeights::FullAttn { ffn, .. } => ffn,
            Qwen35LayerWeights::LinearAttn { ffn, .. } => ffn,
        }
    }

    /// ADR-020 AC#5 Iter C2.4 — mutable accessor for the FFN slot, used
    /// by `Qwen35Model::apply_dwq_overlay` to populate the
    /// `MoeFfnWeightsQ::expert_*_affine` slots without re-allocating
    /// the layer.
    pub fn ffn_mut(&mut self) -> &mut Qwen35FfnWeights {
        match self {
            Qwen35LayerWeights::FullAttn { ffn, .. } => ffn,
            Qwen35LayerWeights::LinearAttn { ffn, .. } => ffn,
        }
    }
}

// ================================================================
// Top-level model
// ================================================================

/// Complete Qwen3.5 / Qwen3.5-MoE model: config + weights + MTP.
///
/// # Construction
///
/// Use [`Qwen35Model::load_from_gguf`] to populate from a GGUF file;
/// or [`Qwen35Model::empty_from_cfg`] for tests that construct zero-weight
/// models of a given shape.
pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    /// Per-layer weight bundles. `len() == cfg.num_hidden_layers`.
    pub layers: Vec<Qwen35LayerWeights>,
    /// Token-embedding table (vocab_size × hidden_size, row-major).
    pub token_embd: Vec<f32>,
    /// LM-head output projection (hidden_size × vocab_size, row-major per
    /// GGUF's `[out_dim, in_dim]` convention).
    pub output_weight: Vec<f32>,
    /// Final-layer RMSNorm weight, shape `[hidden_size]`.
    pub output_norm: Vec<f32>,
    /// Optional MTP draft block. Executed only by speculative decoding, never
    /// by the verifier's main layer loop.
    pub mtp: Option<MtpWeights>,
}

impl Qwen35Model {
    /// Parse only the [`Qwen35Config`] from a GGUF without allocating
    /// weights. Cheap — useful for introspection (e.g. `hf2q info`).
    pub fn load_config_only(gguf: &GgufFile) -> Result<Qwen35Config> {
        Qwen35Config::from_gguf(gguf).context("Qwen35Config::from_gguf")
    }

    /// Construct a zero-weighted model of the shape prescribed by `cfg`.
    ///
    /// Useful for test harnesses that want to populate weights by hand
    /// (e.g. fuzzing per-layer forward with synthetic data). The MTP
    /// field is `None` (no MTP in an empty model).
    pub fn empty_from_cfg(cfg: Qwen35Config) -> Self {
        let h = cfg.hidden_size as usize;
        let vocab = cfg.vocab_size as usize;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for kind in &cfg.layer_types {
            let ffn = empty_ffn_for(&cfg);
            let layer = match kind {
                Qwen35LayerKind::FullAttention => Qwen35LayerWeights::FullAttn {
                    attn: empty_full_attn_weights(&cfg),
                    ffn,
                },
                Qwen35LayerKind::LinearAttention => Qwen35LayerWeights::LinearAttn {
                    attn: empty_delta_net_weights(&cfg),
                    ffn,
                },
            };
            layers.push(layer);
        }

        Self {
            layers,
            token_embd: vec![0.0f32; vocab * h],
            output_weight: vec![0.0f32; h * vocab],
            output_norm: vec![1.0f32; h],
            mtp: None,
            cfg,
        }
    }

    /// Load a complete model from a GGUF file.
    ///
    /// Acquires a Metal GPU device for dequantization, loads the three
    /// global tensors (`token_embd`, `output`, `output_norm`), then
    /// iterates over all layers.
    ///
    /// For MoE models the expert weight tensors (`ffn_{gate,up,down}_exps`)
    /// are kept in their native GGML block quantization via
    /// [`weight_loader::load_moe_ffn_quantized`].  This prevents the ~128 GB
    /// Metal working-set OOM that an F32-expanded load would cause for the
    /// 35B-A3B apex model.
    ///
    /// For Dense models the behaviour is unchanged: weights are dequantized
    /// to f32 via [`weight_loader::load_layer`].
    ///
    /// `progress` drives the default-mode in-place `\r`-overwrite progress
    /// line on stderr (mirrors the Gemma path at [`crate::serve::forward_mlx::MlxModelWeights::load_from_gguf`]).
    /// It is a no-op when stderr isn't a TTY or verbosity > 0 (tracing
    /// debug events then cover per-layer detail). Pass a silent progress
    /// (`LoadProgress::new(false, 1, n_layers)`) from non-CLI call sites
    /// — `RealActivationCapture::new`, `ppl_driver`, the integration test
    /// at `model.rs:622` — to suppress output cleanly.
    pub fn load_from_gguf(
        gguf: &GgufFile,
        progress: &mut crate::serve::header::LoadProgress,
    ) -> Result<Self> {
        let mut cfg = Self::load_config_only(gguf)?;

        let device = MlxDevice::new()
            .map_err(|e| anyhow!("MlxDevice::new for weight loading: {e}"))?;

        let (mut token_embd, output_weight, output_norm) =
            weight_loader::load_global_tensors(gguf, &cfg, &device)
                .context("load_global_tensors")?;

        // ADR-012 P9b real-model finding (Qwen3.6-27B): the embedding table is
        // physically padded for alignment (e.g. 248320 rows) while the metadata
        // vocab_size reports the logical vocab (e.g. 248044). When they
        // disagree, take the tensor shape as authoritative for cfg.vocab_size
        // — the io_heads.rs assertion `token_embd.len() == vocab * hidden`
        // and the LM head matmul both require the row-count match. The logical
        // vocab is still recoverable from `tokenizer.ggml.tokens` metadata if
        // ever needed (e.g. for sampler masking of pad rows).
        let h = cfg.hidden_size as usize;
        if h > 0 {
            let physical_vocab = token_embd.len() / h;
            if (physical_vocab as u32) != cfg.vocab_size {
                tracing::info!(
                    metadata_vocab = cfg.vocab_size,
                    physical_vocab = physical_vocab,
                    "qwen35 vocab pad: metadata reports {} but token_embd has {} rows; using physical for cfg.vocab_size",
                    cfg.vocab_size,
                    physical_vocab,
                );
                cfg.vocab_size = physical_vocab as u32;
            }
        }

        // Special-token coverage fix (Qwen3.5 / Qwen3.6-27B dense):
        //
        // The GGUF embed table is truncated to the base tokenizer vocab (e.g.
        // 248044 entries derived from tokenizer.ggml.tokens), but the Qwen3.5
        // chat template inserts special tokens <|im_start|> (248045),
        // <|im_end|> (248046), … up to <|fim_suffix|> (248062).  These IDs
        // live in tokenizer.json's added_tokens section and are NOT reflected
        // in tokenizer.ggml.tokens or any GGUF metadata key.
        //
        // Fix: if cfg.vocab_size < QWEN35_FULL_VOCAB (248320 — the authoritative
        // vocab_size from the HF config, which covers all special tokens) and
        // the gap is small (< 2048 rows), extend token_embd in-place with zero
        // rows.  Zero-filled embeddings for structural special tokens
        // (<|im_start|> etc.) are functional: the model's attention pattern
        // treats these IDs as role delimiters regardless of embedding magnitude.
        // This avoids an OOB panic in embed_tokens when prompt tokens exceed
        // the GGUF's truncated vocab.
        //
        // output_weight (lm_head) is NOT extended: the model never generates
        // special token IDs through argmax over the trained 248044-wide logits.
        if h > 0 {
            const QWEN35_FULL_VOCAB: u32 = 248_320;
            let current_vocab = cfg.vocab_size;
            if current_vocab < QWEN35_FULL_VOCAB
                && (QWEN35_FULL_VOCAB - current_vocab) < 2048
            {
                let rows_to_add = (QWEN35_FULL_VOCAB - current_vocab) as usize;
                tracing::info!(
                    current_vocab,
                    extended_vocab = QWEN35_FULL_VOCAB,
                    rows_to_add,
                    "qwen35 special-token coverage: extending token_embd \
                     from {} to {} rows with zero embeddings",
                    current_vocab,
                    QWEN35_FULL_VOCAB,
                );
                token_embd.resize(QWEN35_FULL_VOCAB as usize * h, 0.0f32);
                // NOTE: cfg.vocab_size is NOT updated here.  cfg.vocab_size
                // reflects the LM-head output dimension (output_weight rows =
                // 248044).  The embed table now has more rows than cfg.vocab_size,
                // and embed_tokens_gpu uses token_embd.len()/h as its effective
                // vocab_size so it can look up any token in [0, 248320).
                // Keeping cfg.vocab_size at 248044 ensures the lm_head matmul
                // allocates the correct output buffer size.
            }
        }

        let mtp = load_mtp_weights_if_present(gguf, &cfg, &device)
            .context("load_mtp_weights_if_present")?;

        // ADR-012 item-2 architectural fix (2026-04-25): MoE experts MUST
        // be loaded as native ggml-quantized blocks (`MoeQ`). The previous
        // F16-detection / F32-expand fallback ("Moe" variant via
        // `weight_loader::load_moe_ffn`) was peer-misaligned — peers
        // (mlx-lm, llama.cpp, AutoAWQ) never F32-expand MoE experts at load
        // time. Apex 35B-A3B at F32 is ~128 GB which doesn't fit on a
        // 128 GB system. If a caller ever supplies F16/F32 experts
        // (e.g. legacy GGUFs), we fail loud at load time rather than
        // silently expanding.
        use mlx_native::ops::quantized_matmul_ggml::GgmlType;
        if cfg.variant == Qwen35Variant::Moe {
            if let Some(info) = gguf.tensor_info("blk.0.ffn_gate_exps.weight") {
                if matches!(info.ggml_type, GgmlType::F16 | GgmlType::F32) {
                    return Err(anyhow!(
                        "qwen35moe load: MoE expert tensor 'blk.0.ffn_gate_exps.weight' \
                         is dtype {:?}; native ggml-block quantization (Q4_0, Q5_K, Q6_K, Q8_0) \
                         is required. Re-emit the GGUF with quantized MoE experts — no \
                         F32-expansion fallback per ADR-012 item-2 (peer alignment).",
                        info.ggml_type
                    ));
                }
            }
        }

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for i in 0..cfg.num_hidden_layers {
            // Default-mode CLI progress line: `\r loading {i}/{n} layers`.
            // No-op for SERVE / tests / non-TTY callers (the `progress`
            // they pass is a `LoadProgress::new(false, 1, _)` silent
            // sentinel). The `i+1` form matches the Gemma path
            // (`forward_mlx::MlxModelWeights::load_from_gguf` which calls
            // `progress.on_layer(i + 1)` per layer).
            progress.on_layer(i as usize + 1);
            let layer = match cfg.variant {
                Qwen35Variant::Moe => {
                    let kind = cfg
                        .layer_types
                        .get(i as usize)
                        .copied()
                        .ok_or_else(|| anyhow!("layer_idx {i} out of range"))?;
                    // Production quantized experts (Q4_0/Q5_K/Q6_K/Q8_0) —
                    // keep native blocks on Metal; no F32 expansion.
                    let ffn_weights = {
                        let ffn = weight_loader::load_moe_ffn_quantized(gguf, i, &device)
                            .with_context(|| format!("load_moe_ffn_quantized layer {i}"))?;
                        Qwen35FfnWeights::MoeQ(ffn)
                    };
                    match kind {
                        Qwen35LayerKind::FullAttention => {
                            let attn = weight_loader::load_full_attn_layer(gguf, &cfg, i, &device)
                                .with_context(|| format!("load_full_attn layer {i}"))?;
                            Qwen35LayerWeights::FullAttn { attn, ffn: ffn_weights }
                        }
                        Qwen35LayerKind::LinearAttention => {
                            let attn = weight_loader::load_delta_net_layer(gguf, &cfg, i, &device)
                                .with_context(|| format!("load_delta_net layer {i}"))?;
                            Qwen35LayerWeights::LinearAttn { attn, ffn: ffn_weights }
                        }
                    }
                }
                Qwen35Variant::Dense => {
                    weight_loader::load_layer(gguf, &cfg, i, &device)
                        .with_context(|| format!("load_layer {i}"))?
                }
            };
            layers.push(layer);
        }

        // Clear the progress line so any subsequent stderr output
        // (e.g. tracing log lines, banner emission) starts on a clean
        // row. Mirrors the Gemma path's terminal-side hygiene.
        progress.finish();

        Ok(Self {
            cfg,
            layers,
            token_embd,
            output_weight,
            output_norm,
            mtp,
        })
    }

    /// Report the active FFN variant (Dense or Moe) determined by config.
    pub fn ffn_variant(&self) -> Qwen35Variant {
        self.cfg.variant
    }

    /// Per-layer metadata helper: number of linear-attention layers.
    pub fn num_linear_attn_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l, Qwen35LayerWeights::LinearAttn { .. }))
            .count()
    }

    /// Per-layer metadata helper: number of full-attention layers.
    pub fn num_full_attn_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l, Qwen35LayerWeights::FullAttn { .. }))
            .count()
    }

    /// Per-layer kind lookup.
    pub fn layer_kind(&self, idx: u32) -> Option<Qwen35LayerKind> {
        self.layers.get(idx as usize).map(|l| l.kind())
    }

    /// ADR-020 AC#5 Iter C2.4 — overlay a DWQ-trained mlx-affine
    /// safetensors file on top of an already-GGUF-loaded Qwen35 model.
    ///
    /// Walks the safetensors stems looking for SEPARATE per-expert
    /// gate/up/down stems (`blk.{i}.ffn_gate.{e}`, `ffn_up.{e}`,
    /// `ffn_down.{e}`).  Aggregates into stacked `MlxAffineMoeStack`
    /// per (layer, role) bucket and assigns to
    /// `MoeFfnWeightsQ.expert_{gate,up,down}_affine`.
    ///
    /// Note Qwen35 splits gate + up (no fused `ffn_gate_up.{e}`
    /// — that's Gemma 4's convention).  Dense Qwen35 layers are not
    /// covered (the production qwen35 path's dense layers go through
    /// DenseFfnWeightsQ, not MlxModelWeights).  The overlay's `format`
    /// metadata field is validated; `bits`/`group_size` come from
    /// metadata too (default 4 / 32 if absent).
    ///
    /// Returns the count of MoE buckets overridden.  Logs each
    /// unmatched stem at `tracing::warn!`.
    pub fn apply_dwq_overlay(
        &mut self,
        device: &mlx_native::MlxDevice,
        path: &std::path::Path,
    ) -> anyhow::Result<usize> {
        use crate::core::mlx_safetensors_loader::MlxAffineLinear;
        use crate::serve::forward_mlx::{
            parse_dwq_moe_expert_role, parse_dwq_overlay_metadata, MlxAffineMoeStack, MoeBaseRole,
        };
        use anyhow::Context;

        let bytes = std::fs::read(path)
            .with_context(|| format!("qwen35 apply_dwq_overlay: read {}", path.display()))?;
        let (_n, metadata_obj) = safetensors::SafeTensors::read_metadata(&bytes)
            .map_err(|e| anyhow::anyhow!("qwen35 apply_dwq_overlay: read_metadata: {e:?}"))?;
        let (bits, group_size) =
            parse_dwq_overlay_metadata(metadata_obj.metadata().as_ref())
                .with_context(|| format!("qwen35 apply_dwq_overlay: metadata of {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| anyhow::anyhow!("qwen35 apply_dwq_overlay: deserialize: {e:?}"))?;

        let mut moe_buckets: std::collections::HashMap<
            (usize, MoeBaseRole),
            Vec<(usize, MlxAffineLinear)>,
        > = std::collections::HashMap::new();
        let mut unknown_skipped: usize = 0;
        // ADR-020 AC#7 iter B1 — track whether lm_head ("output") was overlaid.
        // The lm_head weights live as `Vec<f32>` (NOT a per-layer MlxQWeight),
        // so the overlay path dequantizes the DWQ-trained codes back to f32
        // and overwrites `self.output_weight` in place.  The next forward
        // call re-quantizes these to Q4_0 via `upload_q4_0_from_f32` (in
        // `forward_gpu.rs::ensure_gpu_cache_primed`); the round-trip loses
        // the per-group DWQ bias term but preserves the trained scale +
        // codes — measurable AC#7 signal without requiring a new affine
        // matmul kernel path.
        let mut lm_head_overridden: bool = false;
        // ADR-020 AC#7 iter B2.A — count of dense attn projections overlaid
        // (Q/K/V/O across all FullAttn layers).  Same Vec<f32>+Q4_0 storage
        // path as lm_head; F2 round-trip drift was measured under 0.10 for
        // every attn Linear in the empirical 27B 20-step overlay.
        let mut overridden_dense_attn: usize = 0;
        // ADR-020 AC#7 iter B2.B — count of dense FFN projections overlaid
        // (Gate/Up/Down across all FullAttn layers with DenseQ variant).
        // Path: dequant DWQ → transpose to GGUF native → Q4_0 re-encode →
        // replace MlxBuffer in DenseFfnWeightsQ.  MoE FFN layers are NOT
        // counted here (they go through the existing MoE bucket path).
        let mut overridden_dense_ffn: usize = 0;

        for name in st.names() {
            let stem = match name.strip_suffix(".weight") {
                Some(s) => s,
                None => continue,
            };
            if st.tensor(&format!("{stem}.scales")).is_err()
                || st.tensor(&format!("{stem}.biases")).is_err()
            {
                continue;
            }
            // ── lm_head ("output.weight") special-case ─────────────────────
            // The Phase 3c trainer emits `output.weight` for the LM head
            // (per `dwq_loop.rs::3949`); it does NOT have the `blk.{i}.`
            // prefix, so dispatch it here before the per-layer parser.
            if stem == "output" {
                let linear = MlxAffineLinear::from_safetensors(&st, stem, bits, group_size)
                    .with_context(|| format!("qwen35 apply_dwq_overlay: parse {stem}"))?;
                let h = self.cfg.hidden_size as usize;
                let v = self.output_weight.len() / h.max(1);
                if linear.n != v || linear.k != h {
                    anyhow::bail!(
                        "qwen35 DWQ overlay: output shape ({}, {}) != model lm_head ({}, {})",
                        linear.n, linear.k, v, h,
                    );
                }
                // ADR-020 AC#7 foundation F2 — measure the Q4_0 round-trip
                // drift the iter-B1 path will incur when
                // `forward_gpu.rs::ensure_gpu_cache_primed::upload_q4_0_from_f32`
                // re-quantizes our dequantized lm_head Vec<f32>.  This
                // gives operators a deterministic readout of how much
                // DWQ signal is being LOST to the codec round-trip on
                // every overlay-load — independent of any training
                // run's actual KL improvement.
                match linear.q4_0_round_trip_drift() {
                    Ok(d) => {
                        eprintln!(
                            "[qwen35 DWQ overlay] lm_head Q4_0 round-trip drift: \
                             rms={rms:.4e} max={max:.4e} \
                             relative_rms={rrms:.4} bias_fraction={bf:.4} \
                             ({n}x{k}, gs={gs}, bits={bits}); \
                             read: relative_rms<0.1 ⇒ codec preserves signal, \
                             >0.5 ⇒ codec destroys it",
                            rms = d.rms_drift, max = d.max_abs_drift,
                            rrms = d.relative_rms, bf = d.bias_fraction,
                            n = d.n, k = d.k, gs = d.group_size, bits = d.bits,
                        );
                    }
                    Err(e) => {
                        // Never fatal — drift measurement is purely a
                        // diagnostic; alignment failure here would have
                        // already tripped the from_safetensors load
                        // earlier (Q4_0 alignment is a subset of DWQ's
                        // group_size constraint at gs=32).
                        tracing::warn!(error = %e,
                            "qwen35 DWQ overlay: Q4_0 round-trip drift measurement skipped");
                    }
                }
                self.output_weight = linear.dequantize_to_f32();
                lm_head_overridden = true;
                continue;
            }
            let after_blk = match stem.strip_prefix("blk.") {
                Some(s) => s,
                None => continue,
            };
            let dot = match after_blk.find('.') {
                Some(d) => d,
                None => continue,
            };
            let layer_idx: usize = match after_blk[..dot].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if layer_idx >= self.layers.len() {
                continue;
            }
            let role = &after_blk[(dot + 1)..];

            // ── ADR-020 AC#7 iter B2.A — dense attention Q/K/V/O ──────────────
            // The Phase 3c trainer trained these 4 Linears per FullAttn layer
            // (jointly with lm_head + dense FFN gate/up/down via cross-layer
            // KL gradients).  Without this branch they would hit
            // `unknown_skipped` and the model would receive a partial overlay
            // where the lm_head was tuned to match a fully-DWQ-trained dense
            // stack but the dense stack is still vanilla Q4_0 — the
            // partial-application mismatch documented in the F2 round-trip
            // measurement memory.
            //
            // Storage is `FullAttnLayerWeights.wq/wk/wv/wo: Vec<f32>`; the
            // GPU upload path goes through `upload_q4_0_from_f32` per
            // `gpu_full_attn.rs::FullAttnWeightsGpu::from_cpu` (lines
            // 334-340), the same Q4_0 codec the iter-B1 lm_head Vec<f32>
            // overwrite path uses.  F2 measurement (`hf2q dwq-overlay-drift`
            // on a real 27B 20-step overlay) confirmed all 4 attn roles
            // round-trip with `relative_rms < 0.10` (codec preserves >90%
            // of DWQ signal per Linear).
            //
            // DeltaNet (LinearAttention) layers are not trained by the
            // wrapper (skipped in `train_all_linears_full_model_dwq`'s
            // layer iter), so a DeltaNet stem would mean a malformed
            // overlay; we log + skip rather than panic.
            let attn_role_target: Option<AttnRole> = match role {
                "attn_q"      => Some(AttnRole::Q),
                "attn_k"      => Some(AttnRole::K),
                "attn_v"      => Some(AttnRole::V),
                "attn_output" => Some(AttnRole::Output),
                _ => None,
            };
            if let Some(role_kind) = attn_role_target {
                let linear = MlxAffineLinear::from_safetensors(&st, stem, bits, group_size)
                    .with_context(|| format!("qwen35 apply_dwq_overlay: parse {stem}"))?;
                let layer_kind_label = format!("{role_kind:?}");
                match overwrite_full_attn_f32_linear(
                    &mut self.layers[layer_idx],
                    role_kind,
                    &linear,
                    layer_idx,
                    stem,
                ) {
                    Ok(()) => {
                        // Diagnostic drift readout — never fatal (mirrors
                        // the lm_head path).
                        match linear.q4_0_round_trip_drift() {
                            Ok(d) => eprintln!(
                                "[qwen35 DWQ overlay] {role_label} \
                                 (layer {layer_idx}) Q4_0 round-trip drift: \
                                 rms={rms:.4e} max={max:.4e} \
                                 relative_rms={rrms:.4} bias_fraction={bf:.4}",
                                role_label = layer_kind_label,
                                rms = d.rms_drift, max = d.max_abs_drift,
                                rrms = d.relative_rms, bf = d.bias_fraction,
                            ),
                            Err(e) => tracing::warn!(error = %e,
                                "qwen35 DWQ overlay: {} layer {} drift skipped",
                                layer_kind_label, layer_idx),
                        }
                        overridden_dense_attn += 1;
                    }
                    Err(e) => {
                        tracing::warn!(error = %e,
                            "qwen35 DWQ overlay: {} layer {} skipped",
                            layer_kind_label, layer_idx);
                        unknown_skipped += 1;
                    }
                }
                continue;
            }

            // ── ADR-020 AC#7 iter B2.B — dense FFN gate/up/down ───────────────
            // The Phase 3c trainer trained these 3 Linears per FullAttn layer
            // (jointly with attn + lm_head).  Without this branch dense FFN
            // training would be silently dropped at serve, defeating the
            // cross-layer KL signal.  Routes only when ffn variant is DenseQ;
            // MoE per-expert stems (`ffn_gate.{e}` etc.) take the bucket
            // path below.
            //
            // Native storage is GGML Q4_0 blocks (MlxBuffer), not Vec<f32>,
            // so the helper does a full DWQ→native-shape→Q4_0-re-encode→
            // new MlxBuffer dance.  F2 round-trip measurement on the
            // empirical 27B 20-step overlay found relative_rms < 0.10 on
            // every dense FFN role (codec preserves >90% of DWQ signal).
            let dense_ffn_role: Option<DenseFfnRole> = match role {
                "ffn_gate" => Some(DenseFfnRole::Gate),
                "ffn_up"   => Some(DenseFfnRole::Up),
                "ffn_down" => Some(DenseFfnRole::Down),
                _ => None,
            };
            if let Some(ffn_role) = dense_ffn_role {
                let linear = MlxAffineLinear::from_safetensors(&st, stem, bits, group_size)
                    .with_context(|| format!("qwen35 apply_dwq_overlay: parse {stem}"))?;
                let role_label = format!("{ffn_role:?}");
                match overwrite_dense_ffn_q4_0_linear(
                    &mut self.layers[layer_idx], ffn_role, &linear, layer_idx, stem, device,
                ) {
                    Ok(()) => {
                        match linear.q4_0_round_trip_drift() {
                            Ok(d) => eprintln!(
                                "[qwen35 DWQ overlay] {role_label} \
                                 (layer {layer_idx}) Q4_0 round-trip drift: \
                                 rms={rms:.4e} max={max:.4e} \
                                 relative_rms={rrms:.4} bias_fraction={bf:.4}",
                                rms = d.rms_drift, max = d.max_abs_drift,
                                rrms = d.relative_rms, bf = d.bias_fraction,
                            ),
                            Err(e) => tracing::warn!(error = %e,
                                "qwen35 DWQ overlay: {} layer {} drift skipped",
                                role_label, layer_idx),
                        }
                        overridden_dense_ffn += 1;
                    }
                    Err(e) => {
                        // MoE FFN: the bucket path will pick it up under
                        // the per-expert stem pattern (ffn_gate.{e}); not a
                        // failure for that variant.  Errors from native-
                        // type / shape mismatch ARE surfaced.
                        tracing::warn!(error = %e,
                            "qwen35 DWQ overlay: {} layer {} skipped",
                            role_label, layer_idx);
                        unknown_skipped += 1;
                    }
                }
                continue;
            }

            // MoE per-expert stems (gate.{e}, up.{e}, down.{e}) — bucketed
            // and applied below.
            if let Some((base, expert_idx)) = parse_dwq_moe_expert_role(role) {
                let linear = MlxAffineLinear::from_safetensors(&st, stem, bits, group_size)
                    .with_context(|| format!("qwen35 apply_dwq_overlay: parse {stem}"))?;
                moe_buckets
                    .entry((layer_idx, base))
                    .or_default()
                    .push((expert_idx, linear));
            } else {
                unknown_skipped += 1;
            }
        }

        let mut moe_stacked: usize = 0;
        for ((layer_idx, base), mut linears) in moe_buckets.into_iter() {
            linears.sort_by_key(|(e, _)| *e);
            let n_experts = linears.len();
            for (i, (e, _)) in linears.iter().enumerate() {
                if *e != i {
                    anyhow::bail!(
                        "qwen35 DWQ overlay: bucket (layer={layer_idx}, base={:?}) non-contiguous expert idx (got {} at slot {})",
                        base, e, i,
                    );
                }
            }
            let n = linears[0].1.n;
            let k = linears[0].1.k;
            let bits_per = linears[0].1.bits;
            let gs_per = linears[0].1.group_size;
            for (e, l) in &linears[1..] {
                if l.n != n || l.k != k || l.bits != bits_per || l.group_size != gs_per {
                    anyhow::bail!(
                        "qwen35 DWQ overlay: bucket (layer={layer_idx}, base={:?}) expert {} shape mismatch",
                        base, e,
                    );
                }
            }
            if bits_per != 4 || gs_per != 32 {
                anyhow::bail!(
                    "qwen35 DWQ overlay: only bits=4 group_size=32 supported (got bits={}, gs={})",
                    bits_per, gs_per,
                );
            }
            let pack_factor = 32 / bits_per as usize;
            let k_packed = k / pack_factor;
            let groups_per_row = k / (gs_per as usize);

            // Pack + BF16 conversion + upload.
            let stack_words = n_experts * n * k_packed;
            let mut packed_stack: Vec<u32> = vec![0u32; stack_words];
            let mut scales_stack_bf16: Vec<u16> =
                vec![0u16; n_experts * n * groups_per_row];
            let mut biases_stack_bf16: Vec<u16> =
                vec![0u16; n_experts * n * groups_per_row];
            for (e, lin) in &linears {
                for row in 0..n {
                    for kp in 0..k_packed {
                        let mut word: u32 = 0;
                        for j in 0..pack_factor {
                            let code = lin.q_int[row * k + kp * pack_factor + j] as u32;
                            debug_assert!(code <= 0xF);
                            word |= (code & 0xF) << (j * 4);
                        }
                        packed_stack[((*e * n) + row) * k_packed + kp] = word;
                    }
                }
                let s_offset = e * n * groups_per_row;
                for (i, v) in lin.scales.iter().enumerate() {
                    scales_stack_bf16[s_offset + i] = half::bf16::from_f32(*v).to_bits();
                }
                for (i, v) in lin.biases.iter().enumerate() {
                    biases_stack_bf16[s_offset + i] = half::bf16::from_f32(*v).to_bits();
                }
            }

            let mut weight_buf = device
                .alloc_buffer(
                    stack_words * std::mem::size_of::<u32>(),
                    mlx_native::DType::U32,
                    vec![n_experts, n, k_packed],
                )
                .map_err(|e| anyhow::anyhow!("qwen35 MoE stack weight alloc: {e}"))?;
            weight_buf
                .as_mut_slice::<u32>()
                .map_err(|e| anyhow::anyhow!("qwen35 MoE stack weight slice: {e}"))?
                .copy_from_slice(&packed_stack);
            let mut scales_buf = device
                .alloc_buffer(
                    scales_stack_bf16.len() * std::mem::size_of::<u16>(),
                    mlx_native::DType::BF16,
                    vec![n_experts, n, groups_per_row],
                )
                .map_err(|e| anyhow::anyhow!("qwen35 MoE stack scales alloc: {e}"))?;
            scales_buf
                .as_mut_slice::<u16>()
                .map_err(|e| anyhow::anyhow!("qwen35 MoE stack scales slice: {e}"))?
                .copy_from_slice(&scales_stack_bf16);
            let mut biases_buf = device
                .alloc_buffer(
                    biases_stack_bf16.len() * std::mem::size_of::<u16>(),
                    mlx_native::DType::BF16,
                    vec![n_experts, n, groups_per_row],
                )
                .map_err(|e| anyhow::anyhow!("qwen35 MoE stack biases alloc: {e}"))?;
            biases_buf
                .as_mut_slice::<u16>()
                .map_err(|e| anyhow::anyhow!("qwen35 MoE stack biases slice: {e}"))?
                .copy_from_slice(&biases_stack_bf16);

            let stack = MlxAffineMoeStack {
                weight: weight_buf,
                scales: scales_buf,
                biases: biases_buf,
                n,
                k,
                bits: bits_per,
                group_size: gs_per as u32,
                num_experts: n_experts,
            };

            // Assign to the right MoE slot.
            let layer = &mut self.layers[layer_idx];
            if let Qwen35FfnWeights::MoeQ(moeq) = layer.ffn_mut() {
                match base {
                    MoeBaseRole::Gate => moeq.expert_gate_affine = Some(stack),
                    MoeBaseRole::Up => moeq.expert_up_affine = Some(stack),
                    MoeBaseRole::Down => moeq.expert_down_affine = Some(stack),
                    MoeBaseRole::GateUp => {
                        tracing::warn!(
                            layer_idx,
                            n_experts,
                            "qwen35 DWQ overlay: fused ffn_gate_up.{{e}} not supported (Gemma 4 convention); skipping bucket"
                        );
                        continue;
                    }
                }
                moe_stacked += 1;
                tracing::debug!(layer_idx, ?base, n_experts, n, k, "qwen35 DWQ overlay applied");
            } else {
                tracing::warn!(
                    layer_idx,
                    "qwen35 DWQ overlay: layer FFN is not MoeQ; skipping {:?} bucket",
                    base
                );
            }
        }

        tracing::info!(
            moe_stacked,
            overridden_dense_attn,
            overridden_dense_ffn,
            lm_head_overridden,
            unknown_skipped,
            bits,
            group_size,
            "qwen35 DWQ overlay applied: {moe_stacked} MoE expert stacks + \
             {overridden_dense_attn} dense attn Linears + \
             {overridden_dense_ffn} dense FFN Linears{}",
            if lm_head_overridden { " + lm_head" } else { "" }
        );
        Ok(moe_stacked
            + overridden_dense_attn
            + overridden_dense_ffn
            + if lm_head_overridden { 1 } else { 0 })
    }
}

/// ADR-020 AC#7 iter B2.A — internal role tag for the dense-attention
/// overlay path.  Decouples stem-string parsing (`"attn_q"` etc.) from
/// the Vec<f32> overwrite logic in [`overwrite_full_attn_f32_linear`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttnRole {
    Q,
    K,
    V,
    Output,
}

/// ADR-020 AC#7 iter B2.B — internal role tag for the dense-FFN overlay
/// path.  Decouples stem-string parsing (`"ffn_gate"` / `"ffn_up"` /
/// `"ffn_down"`) from the Q4_0-rebuild logic in
/// [`overwrite_dense_ffn_q4_0_linear`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DenseFfnRole {
    Gate,
    Up,
    Down,
}

/// ADR-020 AC#7 iter B2.B — pure CPU helper: dequantize a DWQ-trained
/// dense FFN Linear into the GGUF-native row-major F32 layout that
/// [`upload_q4_0_from_f32`] expects.
///
/// Shape contract (the trainer wrapper at
/// `dwq_loop.rs::3911-3925` / our iter-A transpose fix):
///   - `Gate`/`Up`: DWQ stores `[n=hidden, k=intermediate]`; GGUF native
///     is `[intermediate, hidden]` row-major.  Transpose direction:
///     `(hidden, intermediate)` → `(intermediate, hidden)`.
///   - `Down`: DWQ stores `[n=intermediate, k=hidden]`; GGUF native is
///     `[hidden, intermediate]` row-major.  Transpose direction:
///     `(intermediate, hidden)` → `(hidden, intermediate)`.
///
/// In every case the output's row length is `hidden` for gate/up and
/// `intermediate` for down — both multiples of `QK4_0=32` for the
/// Qwen3.6-27B-MTP fixture validated by F2 (`hidden=5120`,
/// `intermediate=17408`).  The downstream Q4_0 codec
/// (`quantize_row_q4_0_to_bytes`) row-quantizes one matrix row at a
/// time, so the transposed layout matches what the GGML Q4_0 matmul
/// kernel reads at serve.
fn dwq_to_native_q4_0_f32(
    linear: &crate::core::mlx_safetensors_loader::MlxAffineLinear,
    role: DenseFfnRole,
    intermediate: usize,
    hidden: usize,
) -> anyhow::Result<Vec<f32>> {
    // Validate DWQ shape against role-expected (n, k).  These must hold
    // because the trainer wrapper packs every dense FFN Linear with
    // these exact dimensions; a mismatch here means the overlay was
    // produced for a different model.
    let (expect_n, expect_k) = match role {
        DenseFfnRole::Gate | DenseFfnRole::Up => (hidden, intermediate),
        DenseFfnRole::Down => (intermediate, hidden),
    };
    if linear.n != expect_n || linear.k != expect_k {
        anyhow::bail!(
            "dwq_to_native_q4_0_f32: {role:?} shape ({}, {}) != expected ({}, {})",
            linear.n, linear.k, expect_n, expect_k,
        );
    }

    let dwq_flat = linear.dequantize_to_f32();   // [n, k] row-major
    debug_assert_eq!(dwq_flat.len(), expect_n * expect_k);

    // Transpose into GGUF native layout (rows = output dim).
    // src/dst sizes are role-specific; src_cols is what we index into
    // dwq_flat with (it equals dst's row dim, by construction of transpose).
    let (out_rows, out_cols, src_cols) = match role {
        DenseFfnRole::Gate | DenseFfnRole::Up => {
            // src: [hidden rows, intermediate cols] → dst: [intermediate rows, hidden cols]
            (intermediate, hidden, intermediate)
        }
        DenseFfnRole::Down => {
            // src: [intermediate rows, hidden cols] → dst: [hidden rows, intermediate cols]
            (hidden, intermediate, hidden)
        }
    };
    let mut out = vec![0f32; out_rows * out_cols];
    for r in 0..out_rows {
        for c in 0..out_cols {
            // dst[r, c] = src[c, r]  (transpose)
            out[r * out_cols + c] = dwq_flat[c * src_cols + r];
        }
    }
    Ok(out)
}

/// ADR-020 AC#7 iter B2.B — overwrite a single dense FFN projection
/// (Gate, Up, or Down) on a [`Qwen35LayerWeights::FullAttn`] layer
/// whose FFN variant is [`Qwen35FfnWeights::DenseQ`] (the production
/// path for non-MoE Qwen3.5/3.6 models).
///
/// Unlike the iter-B2.A attn path (which overwrites a `Vec<f32>` and
/// lets the next forward's `upload_q4_0_from_f32` handle the GPU
/// upload), DenseFfnWeightsQ stores raw GGML blocks directly as
/// `MlxBuffer`.  This helper:
///   1. Dequantizes the DWQ-trained values via `dwq_to_native_q4_0_f32`
///      (transpose-included)
///   2. Re-quantizes to Q4_0 bytes via `upload_q4_0_from_f32`
///   3. Replaces the target buffer slot (gate_q / up_q / down_q)
///
/// Errors:
///   - Layer is `LinearAttention`: DeltaNet doesn't have dense FFN
///     overrideable via this stem; trainer never emits FFN stems for
///     LinearAttn layers anyway.
///   - FFN variant is not DenseQ (Dense/Moe/MoeQ): MoeQ goes through
///     the existing MoE bucket path; Dense/Moe wouldn't appear in a
///     production GGUF-loaded model.
///   - Native ggml type is not Q4_0: the trainer was validated against
///     Qwen3.6-27B-MTP (all-Q4_0 dense FFN, confirmed by gguf-dump).
///     Q8_0 / Q6_K paths are deferred until operator validates a
///     fixture that uses them.
fn overwrite_dense_ffn_q4_0_linear(
    layer: &mut Qwen35LayerWeights,
    role: DenseFfnRole,
    linear: &crate::core::mlx_safetensors_loader::MlxAffineLinear,
    layer_idx: usize,
    stem: &str,
    device: &mlx_native::MlxDevice,
) -> anyhow::Result<()> {
    use mlx_native::ops::quantized_matmul_ggml::GgmlType;

    let ffn = match layer {
        Qwen35LayerWeights::FullAttn { ffn, .. } => ffn,
        Qwen35LayerWeights::LinearAttn { .. } => {
            anyhow::bail!(
                "qwen35 DWQ overlay: layer {layer_idx} is LinearAttention; \
                 FFN-role stem '{stem}' not applicable"
            );
        }
    };
    let dq = match ffn {
        Qwen35FfnWeights::DenseQ(d) => d,
        other => {
            anyhow::bail!(
                "qwen35 DWQ overlay: layer {layer_idx} ffn variant is {} \
                 (expected DenseQ); MoeQ goes through the MoE bucket path",
                other.variant()
            );
        }
    };
    // Native ggml type gate: trainer validated against Q4_0 only.
    let native_t = match role {
        DenseFfnRole::Gate | DenseFfnRole::Up => dq.ggml_type_gate_up,
        DenseFfnRole::Down => dq.ggml_type_down,
    };
    if native_t != GgmlType::Q4_0 {
        anyhow::bail!(
            "qwen35 DWQ overlay: layer {layer_idx} {role:?} native ggml type \
             is {native_t:?}; only Q4_0 is supported by iter-B2.B (Q8_0 / \
             Q6_K paths deferred pending operator-validated fixture)"
        );
    }

    let intermediate = dq.intermediate_size as usize;
    let hidden = dq.hidden_size as usize;
    let native_f32 = dwq_to_native_q4_0_f32(linear, role, intermediate, hidden)?;
    let new_buf =
        crate::inference::models::qwen35::gpu_full_attn::upload_q4_0_from_f32(
            &native_f32, device,
        )
        .with_context(|| format!("qwen35 DWQ overlay: re-Q4_0 upload {stem}"))?;
    match role {
        DenseFfnRole::Gate => dq.gate_q = new_buf,
        DenseFfnRole::Up   => dq.up_q   = new_buf,
        DenseFfnRole::Down => dq.down_q = new_buf,
    }
    Ok(())
}

/// ADR-020 AC#7 iter B2.A — overwrite a single dense attention F32
/// projection (Q, K, V, or Output) on a [`Qwen35LayerWeights::FullAttn`]
/// layer with DWQ-trained values.
///
/// Mirrors the iter-B1 lm_head pattern: dequantize the affine-DWQ codes
/// to f32 and overwrite the existing `Vec<f32>` slot.  The next forward's
/// [`gpu_full_attn::FullAttnWeightsGpu::from_cpu`] call re-quantizes via
/// `upload_q4_0_from_f32` — F2 measurement on the empirical 27B 20-step
/// overlay confirms `relative_rms < 0.10` for every attn role (codec
/// preserves >90% of DWQ signal).
///
/// Errors when:
///   - The layer at `layer_idx` is `LinearAttention` (DeltaNet); the
///     wrapper does not train DeltaNet layers, so a DeltaNet stem in
///     the overlay is malformed and not silently applied.
///   - The DWQ tensor's (n, k) does not match the live weight shape;
///     this would indicate the overlay was produced from a different
///     model architecture.
fn overwrite_full_attn_f32_linear(
    layer: &mut Qwen35LayerWeights,
    role: AttnRole,
    linear: &crate::core::mlx_safetensors_loader::MlxAffineLinear,
    layer_idx: usize,
    stem: &str,
) -> anyhow::Result<()> {
    let attn = match layer {
        Qwen35LayerWeights::FullAttn { attn, .. } => attn,
        Qwen35LayerWeights::LinearAttn { .. } => {
            anyhow::bail!(
                "qwen35 DWQ overlay: layer {layer_idx} is LinearAttention \
                 (DeltaNet); attn-role stem '{stem}' is not applicable \
                 (the wrapper does not train DeltaNet layers)"
            );
        }
    };
    // Target Vec<f32> slot for this role (mut borrow needed; pick after
    // the LinearAttn rejection).
    let (slot, expected_label): (&mut Vec<f32>, &'static str) = match role {
        AttnRole::Q      => (&mut attn.wq, "wq"),
        AttnRole::K      => (&mut attn.wk, "wk"),
        AttnRole::V      => (&mut attn.wv, "wv"),
        AttnRole::Output => (&mut attn.wo, "wo"),
    };
    if linear.n * linear.k != slot.len() {
        anyhow::bail!(
            "qwen35 DWQ overlay: {expected_label} layer {layer_idx} shape \
             mismatch — overlay [{} x {}] = {} elements vs live slot {} elements",
            linear.n, linear.k, linear.n * linear.k, slot.len(),
        );
    }
    *slot = linear.dequantize_to_f32();
    Ok(())
}

// ================================================================
// Empty-weight constructors (for tests)
// ================================================================

fn empty_full_attn_weights(cfg: &Qwen35Config) -> FullAttnLayerWeights {
    let h = cfg.hidden_size as usize;
    let nh = cfg.num_attention_heads as usize;
    let nkv = cfg.num_key_value_heads as usize;
    let d = cfg.head_dim as usize;
    let q_total = nh * d;
    let kv_total = nkv * d;
    FullAttnLayerWeights {
        attn_norm: vec![1.0f32; h],
        post_attn_norm: vec![1.0f32; h],
        wq: vec![0.0f32; q_total * h],
        wk: vec![0.0f32; kv_total * h],
        wv: vec![0.0f32; kv_total * h],
        w_gate: vec![0.0f32; q_total * h],
        attn_q_norm: vec![1.0f32; d],
        attn_k_norm: vec![1.0f32; d],
        wo: vec![0.0f32; h * q_total],
    }
}

fn empty_delta_net_weights(cfg: &Qwen35Config) -> DeltaNetLayerWeights {
    let h = cfg.hidden_size as usize;
    let nk = cfg.linear_num_key_heads as usize;
    let nv = cfg.linear_num_value_heads as usize;
    let dk = cfg.linear_key_head_dim as usize;
    let dv = cfg.linear_value_head_dim as usize;
    let k_width = cfg.linear_conv_kernel_dim as usize;
    let qkv_channels = 2 * nk * dk + nv * dv;
    let z_channels = nv * dv;
    DeltaNetLayerWeights {
        attn_norm: vec![1.0f32; h],
        post_attn_norm: vec![1.0f32; h],
        attn_qkv: vec![0.0f32; qkv_channels * h],
        attn_gate: vec![0.0f32; z_channels * h],
        ssm_conv1d: vec![0.0f32; k_width * qkv_channels],
        ssm_alpha: vec![0.0f32; nv * h],
        ssm_dt_bias: vec![0.0f32; nv],
        ssm_beta: vec![0.0f32; nv * h],
        ssm_a: vec![0.0f32; nv],
        // ssm_norm shape is [D_v], broadcast across n_v_heads per token.
        ssm_norm: vec![1.0f32; dv],
        ssm_out: vec![0.0f32; h * z_channels],
    }
}

fn empty_ffn_for(cfg: &Qwen35Config) -> Qwen35FfnWeights {
    match cfg.variant {
        Qwen35Variant::Dense => {
            let h = cfg.hidden_size as usize;
            let m = cfg
                .intermediate_size
                .expect("dense variant requires intermediate_size") as usize;
            Qwen35FfnWeights::Dense(DenseFfnWeights {
                gate: vec![0.0f32; m * h],
                up: vec![0.0f32; m * h],
                down: vec![0.0f32; h * m],
            })
        }
        Qwen35Variant::Moe => {
            let moe_cfg = cfg
                .moe
                .as_ref()
                .expect("moe variant requires moe config");
            let h = cfg.hidden_size as usize;
            let ne = moe_cfg.num_experts as usize;
            let m_moe = moe_cfg.moe_intermediate_size as usize;
            let m_sh = moe_cfg.shared_expert_intermediate_size as usize;
            Qwen35FfnWeights::Moe(MoeFfnWeights {
                router: vec![0.0f32; ne * h],
                expert_gate: vec![0.0f32; ne * m_moe * h],
                expert_up: vec![0.0f32; ne * m_moe * h],
                expert_down: vec![0.0f32; ne * h * m_moe],
                shared_gate_logit: vec![0.0f32; h],
                shared_gate: vec![0.0f32; m_sh * h],
                shared_up: vec![0.0f32; m_sh * h],
                shared_down: vec![0.0f32; h * m_sh],
            })
        }
    }
}


// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::mlx_safetensors_loader::MlxAffineLinear;
    use crate::inference::models::qwen35::{
        default_layer_types, Qwen35LayerKind, Qwen35MoeConfig, Qwen35Variant,
    };

    fn moe_cfg_40() -> Qwen35Config {
        Qwen35Config {
            variant: Qwen35Variant::Moe,
            hidden_size: 64, // small for fast tests
            num_hidden_layers: 40,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            linear_num_key_heads: 4,
            linear_num_value_heads: 8,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            layer_types: default_layer_types(40, 4),
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            rotary_dim: 4,
            mrope_section: [1, 1, 0, 0],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 1024,
            vocab_size: 256,
            attn_output_gate: true,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: true,
            intermediate_size: None,
            moe: Some(Qwen35MoeConfig {
                moe_intermediate_size: 16,
                num_experts: 4,
                num_experts_per_tok: 2,
                shared_expert_intermediate_size: 16,
            }),
        }
    }

    fn dense_cfg_12() -> Qwen35Config {
        let mut cfg = moe_cfg_40();
        cfg.variant = Qwen35Variant::Dense;
        cfg.num_hidden_layers = 12;
        cfg.layer_types = default_layer_types(12, 4);
        cfg.intermediate_size = Some(32);
        cfg.moe = None;
        cfg
    }

    #[test]
    fn empty_moe_40layer_has_correct_slot_counts() {
        let cfg = moe_cfg_40();
        let m = Qwen35Model::empty_from_cfg(cfg.clone());
        assert_eq!(m.layers.len(), 40);
        assert_eq!(m.num_full_attn_layers(), 10);
        assert_eq!(m.num_linear_attn_layers(), 30);
        assert!(m.mtp.is_none());
        assert_eq!(m.token_embd.len(), 256 * 64);
        assert_eq!(m.output_weight.len(), 64 * 256);
        assert_eq!(m.output_norm.len(), 64);
    }

    #[test]
    fn empty_dense_12layer_uses_swiglu_ffn() {
        let cfg = dense_cfg_12();
        let m = Qwen35Model::empty_from_cfg(cfg);
        for l in &m.layers {
            assert_eq!(l.ffn().variant(), "dense");
        }
    }

    #[test]
    fn empty_moe_12layer_uses_moe_ffn() {
        let cfg = moe_cfg_40();
        let m = Qwen35Model::empty_from_cfg(cfg);
        for l in &m.layers {
            assert_eq!(l.ffn().variant(), "moe");
        }
    }

    #[test]
    fn layer_kind_matches_config() {
        let cfg = moe_cfg_40();
        let m = Qwen35Model::empty_from_cfg(cfg.clone());
        for i in 0..40 {
            assert_eq!(
                m.layer_kind(i).unwrap(),
                cfg.layer_types[i as usize]
            );
        }
    }

    #[test]
    fn layer_kind_out_of_bounds_is_none() {
        let cfg = moe_cfg_40();
        let m = Qwen35Model::empty_from_cfg(cfg);
        assert_eq!(m.layer_kind(40), None);
        assert_eq!(m.layer_kind(9999), None);
    }

    #[test]
    fn full_attn_layer_has_q_and_kv_weights() {
        let cfg = moe_cfg_40();
        let m = Qwen35Model::empty_from_cfg(cfg.clone());
        // Layer 3 is full-attention (interval=4).
        let l3 = &m.layers[3];
        match l3 {
            Qwen35LayerWeights::FullAttn { attn, .. } => {
                let h = cfg.hidden_size as usize;
                let nh = cfg.num_attention_heads as usize;
                let d = cfg.head_dim as usize;
                assert_eq!(attn.wq.len(), nh * d * h);
                assert_eq!(attn.attn_q_norm.len(), d);
            }
            _ => panic!("expected layer 3 to be FullAttn"),
        }
    }

    #[test]
    fn linear_attn_layer_has_ssm_weights() {
        let cfg = moe_cfg_40();
        let m = Qwen35Model::empty_from_cfg(cfg.clone());
        // Layer 0 is linear-attention.
        let l0 = &m.layers[0];
        match l0 {
            Qwen35LayerWeights::LinearAttn { attn, .. } => {
                let nv = cfg.linear_num_value_heads as usize;
                assert_eq!(attn.ssm_a.len(), nv);
                assert_eq!(attn.ssm_dt_bias.len(), nv);
                let qkv_channels =
                    (2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
                        + cfg.linear_num_value_heads * cfg.linear_value_head_dim)
                        as usize;
                let k_width = cfg.linear_conv_kernel_dim as usize;
                assert_eq!(attn.ssm_conv1d.len(), k_width * qkv_channels);
            }
            _ => panic!("expected layer 0 to be LinearAttn"),
        }
    }

    #[test]
    fn ffn_variant_reported_via_config() {
        let cfg_moe = moe_cfg_40();
        let m_moe = Qwen35Model::empty_from_cfg(cfg_moe);
        assert_eq!(m_moe.ffn_variant(), Qwen35Variant::Moe);
        let cfg_dense = dense_cfg_12();
        let m_dense = Qwen35Model::empty_from_cfg(cfg_dense);
        assert_eq!(m_dense.ffn_variant(), Qwen35Variant::Dense);
    }

    // ── ADR-020 AC#7 iter B2.A — overwrite_full_attn_f32_linear tests ──
    // Pure CPU; no GPU.  Validates the dense-attention overlay overwrite
    // helper that drives `apply_dwq_overlay`'s 4 new role handlers
    // (attn_q / attn_k / attn_v / attn_output).

    /// Build a synthetic MlxAffineLinear of `[n, k]` shape with codes
    /// chosen so dequantize_to_f32 gives a deterministic vector.
    fn synth_attn_linear(n: usize, k: usize) -> MlxAffineLinear {
        let group_size = 32usize;
        let groups_per_row = k / group_size;
        let q_int: Vec<u8> = (0..(n * k)).map(|i| (i % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.01 + (i as f32) * 1e-4)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.05 + (i as f32) * 1e-4)
            .collect();
        MlxAffineLinear { n, k, group_size, bits: 4, q_int, scales, biases }
    }

    /// Build a 1-FullAttn-layer Qwen35Config (for cheap unit tests of
    /// the per-layer overwrite helper).  Sized to satisfy Q4_0
    /// alignment (k=32 multiple) and DWQ alignment (k=group_size
    /// multiple) without paying the cost of a 27B-shape fixture.
    fn one_full_attn_cfg() -> Qwen35Config {
        let mut cfg = dense_cfg_12();
        // Ensure layer 0 is FullAttention; default_layer_types puts the
        // FullAttn at indices 3, 7, 11, ...  override layer_types[0] for
        // tests so `layers[0]` is the easy-to-target FullAttn variant.
        cfg.layer_types[0] = Qwen35LayerKind::FullAttention;
        cfg
    }

    #[test]
    fn overwrite_full_attn_q_replaces_wq_vec() {
        let cfg = one_full_attn_cfg();
        let mut model = Qwen35Model::empty_from_cfg(cfg);
        let n_q  = model.cfg.num_attention_heads as usize * model.cfg.head_dim as usize;
        let h    = model.cfg.hidden_size as usize;
        let lin  = synth_attn_linear(n_q, h);
        let expected = lin.dequantize_to_f32();

        // Sanity: empty_from_cfg gives wq filled with 0s; ensure overlay changes it.
        match &model.layers[0] {
            Qwen35LayerWeights::FullAttn { attn, .. } => {
                assert!(attn.wq.iter().all(|&v| v == 0.0), "empty wq must start zeroed");
            }
            _ => panic!("layer 0 must be FullAttn after one_full_attn_cfg override"),
        }

        overwrite_full_attn_f32_linear(
            &mut model.layers[0], AttnRole::Q, &lin, 0, "blk.0.attn_q",
        ).expect("overwrite must succeed on shape match");

        match &model.layers[0] {
            Qwen35LayerWeights::FullAttn { attn, .. } => {
                assert_eq!(attn.wq.len(), expected.len());
                assert_eq!(attn.wq, expected,
                    "wq must equal dequantize_to_f32 output bit-identically");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn overwrite_full_attn_role_to_slot_mapping_is_correct() {
        // Each AttnRole must hit its own slot; no cross-pollination.
        let cfg = one_full_attn_cfg();
        let n_q  = cfg.num_attention_heads as usize * cfg.head_dim as usize;
        let n_kv = cfg.num_key_value_heads as usize * cfg.head_dim as usize;
        let h    = cfg.hidden_size as usize;

        let cases: &[(AttnRole, usize, usize)] = &[
            (AttnRole::Q,      n_q,  h   ),
            (AttnRole::K,      n_kv, h   ),
            (AttnRole::V,      n_kv, h   ),
            (AttnRole::Output, h,    n_q ),
        ];
        for (role, n, k) in cases.iter().copied() {
            let mut model = Qwen35Model::empty_from_cfg(cfg.clone());
            let lin = synth_attn_linear(n, k);
            let expected = lin.dequantize_to_f32();
            overwrite_full_attn_f32_linear(
                &mut model.layers[0], role, &lin, 0, "test",
            ).expect("must succeed");
            match &model.layers[0] {
                Qwen35LayerWeights::FullAttn { attn, .. } => {
                    let actual = match role {
                        AttnRole::Q      => &attn.wq,
                        AttnRole::K      => &attn.wk,
                        AttnRole::V      => &attn.wv,
                        AttnRole::Output => &attn.wo,
                    };
                    assert_eq!(actual, &expected,
                        "role {role:?} must hit its own slot bit-identically");
                    // Other roles' slots remain zeroed (no cross-pollination).
                    let untouched: &[&Vec<f32>] = &[&attn.wq, &attn.wk, &attn.wv, &attn.wo];
                    for (idx, slot) in untouched.iter().enumerate() {
                        let role_idx = match role {
                            AttnRole::Q => 0, AttnRole::K => 1,
                            AttnRole::V => 2, AttnRole::Output => 3,
                        };
                        if idx == role_idx { continue; }
                        assert!(slot.iter().all(|&v| v == 0.0),
                            "non-target slot {idx} for role {role:?} was modified");
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn overwrite_full_attn_rejects_linear_attention_layer() {
        let cfg = dense_cfg_12();
        // Find a LinearAttention layer (default arrangement has them
        // at every non-(every-Nth) position).
        let mut model = Qwen35Model::empty_from_cfg(cfg);
        let linear_idx = model.layers.iter()
            .position(|l| matches!(l, Qwen35LayerWeights::LinearAttn { .. }))
            .expect("dense_cfg_12 must contain at least one LinearAttn layer");
        let n_q = model.cfg.num_attention_heads as usize * model.cfg.head_dim as usize;
        let h   = model.cfg.hidden_size as usize;
        let lin = synth_attn_linear(n_q, h);
        let err = overwrite_full_attn_f32_linear(
            &mut model.layers[linear_idx], AttnRole::Q, &lin, linear_idx, "blk.X.attn_q",
        ).expect_err("overlay on LinearAttn layer must error");
        let msg = format!("{err}");
        assert!(msg.contains("LinearAttention"),
            "error must name the layer kind; got: {msg}");
    }

    #[test]
    fn overwrite_full_attn_rejects_shape_mismatch() {
        let cfg = one_full_attn_cfg();
        let mut model = Qwen35Model::empty_from_cfg(cfg);
        let h = model.cfg.hidden_size as usize;
        // wrong N: pass an arbitrary too-small (n, k) pair.
        let bad = synth_attn_linear(32, h);
        let err = overwrite_full_attn_f32_linear(
            &mut model.layers[0], AttnRole::Q, &bad, 0, "blk.0.attn_q",
        ).expect_err("shape mismatch must error");
        let msg = format!("{err}");
        assert!(msg.contains("shape mismatch"),
            "error must name shape mismatch; got: {msg}");
    }

    // ── ADR-020 AC#7 iter B2.B — dwq_to_native_q4_0_f32 transpose tests ──
    // Pure CPU; no GPU.  Validates the shape contract + transpose math
    // for the dense-FFN overlay path (gate/up: hidden×inter → inter×hidden;
    // down: inter×hidden → hidden×inter).

    /// Build a tiny synthetic MlxAffineLinear with codes such that
    /// dequantize_to_f32 returns a deterministic, easy-to-check vector.
    fn synth_ffn_linear(n: usize, k: usize) -> MlxAffineLinear {
        // scales=1, biases=0, codes = (i*k + j) — encodes the flat index
        // directly so transpose math is verifiable element-by-element.
        // Code values must fit in 4 bits ([0,15]) so we mod 16.
        let group_size = 32usize;
        let groups_per_row = k / group_size;
        let q_int: Vec<u8> = (0..n)
            .flat_map(|i| (0..k).map(move |j| ((i * k + j) % 16) as u8))
            .collect();
        let scales = vec![1.0f32; n * groups_per_row];
        let biases = vec![0.0f32; n * groups_per_row];
        MlxAffineLinear { n, k, group_size, bits: 4, q_int, scales, biases }
    }

    #[test]
    fn dwq_to_native_q4_0_f32_gate_transposes_hidden_to_intermediate_rows() {
        // Gate: DWQ shape (n=hidden=64, k=inter=128) → native (inter, hidden)
        // = (128 rows, 64 cols).
        let hidden = 64usize;
        let inter = 128usize;
        let lin = synth_ffn_linear(hidden, inter);
        // Sanity: dwq_flat[i*inter + j] = ((i*inter + j) % 16) as f32.
        let dwq_flat = lin.dequantize_to_f32();
        assert_eq!(dwq_flat.len(), hidden * inter);
        // Native should be transposed: native[r=inter_row, c=hidden_col]
        // = dwq_flat[c=hidden_row, r=inter_col]. Re-index check:
        //   native[r * hidden + c] = dwq_flat[c * inter + r]
        let native = dwq_to_native_q4_0_f32(
            &lin, DenseFfnRole::Gate, inter, hidden,
        ).expect("must succeed on shape match");
        assert_eq!(native.len(), inter * hidden);
        for r in 0..inter {
            for c in 0..hidden {
                let want = ((c * inter + r) % 16) as f32;
                let got = native[r * hidden + c];
                assert_eq!(got, want,
                    "gate transpose [r={r}, c={c}]: got {got} want {want}");
            }
        }
    }

    #[test]
    fn dwq_to_native_q4_0_f32_down_transposes_intermediate_to_hidden_rows() {
        // Down: DWQ shape (n=inter=128, k=hidden=64) → native (hidden, inter)
        // = (64 rows, 128 cols).
        let hidden = 64usize;
        let inter = 128usize;
        let lin = synth_ffn_linear(inter, hidden);
        let native = dwq_to_native_q4_0_f32(
            &lin, DenseFfnRole::Down, inter, hidden,
        ).expect("must succeed on shape match");
        assert_eq!(native.len(), hidden * inter);
        // native[r=hidden_row, c=inter_col] = dwq_flat[c=inter_row, r=hidden_col]
        for r in 0..hidden {
            for c in 0..inter {
                let want = ((c * hidden + r) % 16) as f32;
                let got = native[r * inter + c];
                assert_eq!(got, want,
                    "down transpose [r={r}, c={c}]: got {got} want {want}");
            }
        }
    }

    #[test]
    fn dwq_to_native_q4_0_f32_role_validates_shape() {
        // Gate expects (n=hidden, k=inter); supplying (n=inter, k=hidden)
        // must error rather than silently transposing wrong data.
        let hidden = 64usize;
        let inter = 128usize;
        let bad = synth_ffn_linear(inter, hidden);  // wrong order for Gate
        let err = dwq_to_native_q4_0_f32(
            &bad, DenseFfnRole::Gate, inter, hidden,
        ).expect_err("wrong-shape MlxAffineLinear must error for Gate role");
        let msg = format!("{err}");
        assert!(msg.contains("shape"),
            "error message must mention shape; got: {msg}");
    }

    #[test]
    fn dwq_to_native_q4_0_f32_round_trip_dimensions() {
        // Sanity: total element count is preserved across the transpose
        // and the output shape's row length is what Q4_0 will see.
        let hidden = 64usize;
        let inter = 128usize;
        let cases: &[(DenseFfnRole, usize, usize, usize)] = &[
            (DenseFfnRole::Gate, hidden, inter, hidden), // row_len = hidden
            (DenseFfnRole::Up,   hidden, inter, hidden),
            (DenseFfnRole::Down, inter,  hidden, inter), // row_len = inter
        ];
        for (role, dwq_n, dwq_k, expected_row_len) in cases.iter().copied() {
            let lin = synth_ffn_linear(dwq_n, dwq_k);
            let native = dwq_to_native_q4_0_f32(
                &lin, role, inter, hidden,
            ).expect("must succeed");
            assert_eq!(native.len(), hidden * inter,
                "{role:?}: total element count must be hidden*inter");
            // Row length determines Q4_0 quantize_row block alignment;
            // both 64 and 128 are multiples of QK4_0=32.
            assert_eq!(expected_row_len % 32, 0,
                "{role:?}: row_len {expected_row_len} must align to QK4_0=32");
        }
    }

    // ── ADR-020 AC#7 — apply_dwq_overlay end-to-end dispatch tests ──
    // Drive the FULL `Qwen35Model::apply_dwq_overlay` path (stem matcher
    // → role helper → counter increment → return-value computation)
    // against a synthetic safetensors file written to a tempfile.
    // Catches dispatch-loop integration bugs the per-helper unit tests
    // can't see (e.g. wrong stem regex, mis-routed role enum, off-by-one
    // in the return count, lm_head/attn ordering interaction).
    //
    // Uses a real MlxDevice for buffer alloc inside apply_dwq_overlay's
    // re-Q4_0 path; runtime-skips if Metal isn't available.

    fn write_synthetic_overlay(
        path: &std::path::Path,
        triplets: &[(String, MlxAffineLinear)],
        bits: u32,
        group_size: usize,
    ) {
        use crate::core::mlx_safetensors_loader::MlxAffineLinearBytes;
        use safetensors::tensor::Dtype;
        // Owned per-Linear bytes must outlive the borrowed views.
        // F32 scales/biases for byte-identical round-trip in the test;
        // production overlays save as BF16 (writer_round_trips_with_reader_bf16_scales
        // covers that path).  F32 keeps the assertion deterministic at
        // dequantize_to_f32 bit-identity.
        let owned: Vec<(String, MlxAffineLinearBytes)> = triplets.iter()
            .map(|(stem, lin)| (stem.clone(),
                lin.to_safetensors_bytes(Dtype::F32).expect("to_safetensors_bytes")))
            .collect();
        let mut entries: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
        for (stem, bytes) in &owned {
            let (w, s, b) = bytes.to_safetensors_views().expect("views");
            entries.push((format!("{stem}.weight"), w));
            entries.push((format!("{stem}.scales"), s));
            entries.push((format!("{stem}.biases"), b));
        }
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("format".to_string(), "mlx-affine-dwq-v1".to_string());
        metadata.insert("bits".to_string(), bits.to_string());
        metadata.insert("group_size".to_string(), group_size.to_string());
        let serialized = safetensors::tensor::serialize(entries, Some(metadata))
            .expect("safetensors serialize");
        std::fs::write(path, &serialized).expect("write tempfile");
    }

    #[test]
    fn apply_dwq_overlay_e2e_lm_head_plus_attn_q() {
        // Build a tiny dense Qwen35Model with layer 0 as FullAttention.
        let mut cfg = dense_cfg_12();
        cfg.layer_types[0] = Qwen35LayerKind::FullAttention;
        let n_q = cfg.num_attention_heads as usize * cfg.head_dim as usize;
        let h   = cfg.hidden_size as usize;
        let v   = cfg.vocab_size as usize;
        let mut model = Qwen35Model::empty_from_cfg(cfg);

        // Synthetic: lm_head (output) and one attn_q on layer 0.
        let lin_output = synth_attn_linear(v, h);
        let lin_attn_q = synth_attn_linear(n_q, h);
        let triplets = vec![
            ("output".to_string(), lin_output.clone()),
            ("blk.0.attn_q".to_string(), lin_attn_q.clone()),
        ];

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("overlay.safetensors");
        write_synthetic_overlay(&path, &triplets, 4, 32);

        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[apply_dwq_overlay_e2e] SKIP: no Metal device: {e}");
                return;
            }
        };

        let n_overrides = model.apply_dwq_overlay(&device, &path)
            .expect("apply_dwq_overlay must succeed on a clean overlay");
        assert_eq!(n_overrides, 2,
            "expected 2 overrides (1 lm_head + 1 attn_q); got {n_overrides}");

        // lm_head: Vec<f32> matches dequantized DWQ codes bit-identically.
        assert_eq!(model.output_weight, lin_output.dequantize_to_f32(),
            "lm_head Vec<f32> must equal MlxAffineLinear::dequantize_to_f32 output");

        // attn_q: layer 0's wq matches dequantized DWQ codes.
        match &model.layers[0] {
            Qwen35LayerWeights::FullAttn { attn, .. } => {
                assert_eq!(attn.wq, lin_attn_q.dequantize_to_f32(),
                    "layer-0 wq must equal MlxAffineLinear::dequantize_to_f32 output");
            }
            _ => panic!("layer 0 must be FullAttn after override"),
        }
    }

    #[test]
    fn apply_dwq_overlay_e2e_dense_ffn_stem_skipped_on_cpu_dense_variant() {
        // The B2.B handler errors when ffn variant is `Dense` (CPU
        // test-only; production uses `DenseQ`).  This test confirms the
        // dispatch loop catches that error path: ffn_gate stem on a
        // Dense (not DenseQ) layer increments unknown_skipped and the
        // overall return count is 0.
        let mut cfg = dense_cfg_12();
        cfg.layer_types[0] = Qwen35LayerKind::FullAttention;
        let h     = cfg.hidden_size as usize;
        let inter = cfg.intermediate_size.expect("dense_cfg_12 sets intermediate_size") as usize;
        let mut model = Qwen35Model::empty_from_cfg(cfg);

        // Synthetic ffn_gate sized correctly but routed at a Dense (not
        // DenseQ) FFN; helper must error → handler logs warn + skips.
        let lin_gate = synth_attn_linear(h, inter);
        let triplets = vec![
            ("blk.0.ffn_gate".to_string(), lin_gate),
        ];

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("overlay.safetensors");
        write_synthetic_overlay(&path, &triplets, 4, 32);

        let device = match mlx_native::MlxDevice::new() {
            Ok(d) => d,
            Err(e) => { eprintln!("SKIP: {e}"); return; }
        };

        let n_overrides = model.apply_dwq_overlay(&device, &path)
            .expect("apply_dwq_overlay must succeed even when ffn variant gates");
        assert_eq!(n_overrides, 0,
            "expected 0 overrides — Dense (not DenseQ) variant must be skipped; got {n_overrides}");
    }

    /// Integration smoke: load_from_gguf on the real apex returns a fully-
    /// shaped model with 40 layers (10 full-attn + 30 linear-attn), no MTP.
    /// Runtime-skips when artefact absent (existing path-exists check).
    #[test]
    fn load_from_real_apex_has_correct_shape() {
        let path = std::path::PathBuf::from(
            "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/\
             APEX-Q5_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: apex GGUF not at expected path");
            return;
        }
        if let Err(e) = MlxDevice::new() {
            eprintln!("skipping: no Metal device: {e}");
            return;
        }
        let gguf = match GgufFile::open(&path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        // Silent progress: heavyweight test runs without TTY hooks.
        let mut progress = crate::serve::header::LoadProgress::new(false, 1, 0);
        let m = Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load");
        assert_eq!(m.cfg.variant, Qwen35Variant::Moe);
        assert_eq!(m.layers.len(), 40);
        assert_eq!(m.num_full_attn_layers(), 10);
        assert_eq!(m.num_linear_attn_layers(), 30);
        assert!(m.mtp.is_none(), "apex MTP stripped per 2026-04-23 dump");
    }
}
