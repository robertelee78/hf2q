//! Qwen3-VL text-LM weight loader.
//!
//! Reads all 310 tensors emitted by hf2q's Wedge-4f converter for
//! `Qwen/Qwen3-VL-2B-Instruct` (and the 4B variant when present).
//!
//! # Shape conventions
//!
//! GGUF stores dims innermost-first; mlx-native's loader reverses them
//! at parse time (`mlx_native::gguf::mod.rs:858-861`) so the Rust-side
//! `MlxBuffer::shape()` returns dims in **`[outer, inner]` order**
//! (matching the candle / hf2q `[rows, cols]` convention used elsewhere
//! in the loader code). Concretely, gguf-dump shows
//! `2048, 151936` on disk for `token_embd.weight`; that becomes
//! `[151936, 2048]` after the reverse, and `validate_shape` below
//! expects the post-reverse shape:
//!
//! ```text
//!   token_embd.weight                F16     [vocab_size, hidden_size]
//!   blk.{N}.attn_norm.weight         F32     [hidden_size]
//!   blk.{N}.attn_q.weight            Q4_0    [hidden_size, hidden_size]
//!   blk.{N}.attn_k.weight            Q4_0    [n_kv_heads*head_dim, hidden_size]
//!   blk.{N}.attn_v.weight            Q4_0    [n_kv_heads*head_dim, hidden_size]
//!   blk.{N}.attn_q_norm.weight       F32     [head_dim]            -- per-head Q RMSNorm
//!   blk.{N}.attn_k_norm.weight       F32     [head_dim]            -- per-head K RMSNorm
//!   blk.{N}.attn_output.weight       Q4_0    [hidden_size, hidden_size]
//!   blk.{N}.ffn_norm.weight          F32     [hidden_size]
//!   blk.{N}.ffn_gate.weight          Q4_0    [intermediate_size, hidden_size]
//!   blk.{N}.ffn_up.weight            Q4_0    [intermediate_size, hidden_size]
//!   blk.{N}.ffn_down.weight          Q4_0    [hidden_size, intermediate_size]
//!   output_norm.weight               F32     [hidden_size]
//!   [output.weight                   ?       [vocab_size, hidden_size]]  -- only when untied
//! ```
//!
//! # What this module does NOT do
//!
//! - **No forward pass** — see iter-228b for the dense transformer
//!   implementation. This module only loads bytes into [`MlxBuffer`]s
//!   the forward path will consume.
//! - **No KV-cache allocation** — the cache layout depends on prefill
//!   chunking + GQA semantics that are forward-path concerns. iter-228b
//!   wires the cache.
//! - **No bias support** — the canonical peer source `qwen3vl.cpp:65`
//!   references attention biases (`wo_b`), but the Qwen3-VL-2B q4_0
//!   GGUF emitted by hf2q's converter does NOT include bias tensors
//!   (gguf-dump confirms 11 tensors per block, none named `*.bias`).
//!   When/if a future Qwen3-VL converter emits biases, this loader will
//!   need an optional second pass; the docstring on
//!   [`Qwen3VlTextLayerWeights`] tracks the field shape that would land
//!   it.

use anyhow::{anyhow, Context, Result};
use mlx_native::gguf::GgufFile;
use mlx_native::{MlxBuffer, MlxDevice};

use crate::serve::header::LoadProgress;

use super::Qwen3VlTextConfig;

/// All weights for one transformer layer.
///
/// All tensors are owned (not borrowed) so [`Qwen3VlTextWeights`] can
/// move into a `LoadedModel` and live for the lifetime of the engine
/// worker thread without any lifetime/borrow gymnastics.
///
/// Quantized tensors (`attn_q`, `attn_k`, `attn_v`, `attn_output`,
/// `ffn_gate`, `ffn_up`, `ffn_down`) are loaded as raw GGML Q4_0 blocks
/// via [`GgufFile::load_tensor`] (DType::U8 on Metal). The forward
/// path consumes them via mlx-native's `quantized_matmul_ggml` /
/// `qmatmul` / `qmatmul_t` kernels which dequantize on-device per
/// matmul tile.
///
/// Norm tensors (`attn_norm`, `attn_q_norm`, `attn_k_norm`, `ffn_norm`)
/// are loaded as F32 via [`GgufFile::load_tensor_f32`] — the
/// rms-norm kernel reads them at f32 precision.
///
/// All shapes below are **post-reverse `[outer, inner]`** order — see
/// the module-level docstring for the GGUF-disk vs in-memory shape
/// convention.
///
/// **Bias fields**: not present in iter-228a's loader. If a future
/// converter emits attention biases, add `attn_q_bias: Option<MlxBuffer>`
/// (etc.) here and a second loader pass under
/// [`load_layer`].
pub struct Qwen3VlTextLayerWeights {
    // -------------------- Attention --------------------
    /// Input layer-norm (RMSNorm) weight, shape `[hidden_size]`, F32.
    pub attn_norm: MlxBuffer,
    /// Q projection weight, shape `[hidden_size, hidden_size]`, Q4_0.
    pub attn_q: MlxBuffer,
    /// K projection weight, shape `[n_kv_heads * head_dim, hidden_size]`, Q4_0.
    pub attn_k: MlxBuffer,
    /// V projection weight, shape `[n_kv_heads * head_dim, hidden_size]`, Q4_0.
    pub attn_v: MlxBuffer,
    /// Per-head Q RMSNorm weight, shape `[head_dim]`, F32.
    pub attn_q_norm: MlxBuffer,
    /// Per-head K RMSNorm weight, shape `[head_dim]`, F32.
    pub attn_k_norm: MlxBuffer,
    /// Output projection weight, shape `[hidden_size, hidden_size]`, Q4_0.
    pub attn_output: MlxBuffer,

    // -------------------- FFN --------------------
    /// Post-attention RMSNorm weight, shape `[hidden_size]`, F32.
    pub ffn_norm: MlxBuffer,
    /// FFN gate projection (SiLU branch), shape `[intermediate_size, hidden_size]`, Q4_0.
    pub ffn_gate: MlxBuffer,
    /// FFN up projection (linear branch), shape `[intermediate_size, hidden_size]`, Q4_0.
    pub ffn_up: MlxBuffer,
    /// FFN down projection, shape `[hidden_size, intermediate_size]`, Q4_0.
    pub ffn_down: MlxBuffer,
}

/// All weights for the Qwen3-VL text LM.
///
/// Owns every byte the forward path will consume. When
/// [`Self::tied_word_embeddings`] is `true`, [`Self::output`] is `None`
/// and the LM head re-uses [`Self::token_embd`] (the F16 embedding
/// table). When `false`, [`Self::output`] holds the dedicated `output.weight`
/// tensor.
pub struct Qwen3VlTextWeights {
    /// Token embedding table, shape `[hidden_size, vocab_size]`, F16.
    pub token_embd: MlxBuffer,
    /// Per-layer weights, indexed by layer id.
    pub layers: Vec<Qwen3VlTextLayerWeights>,
    /// Final pre-LM-head RMSNorm, shape `[hidden_size]`, F32.
    pub output_norm: MlxBuffer,
    /// Dedicated LM-head projection, shape `[hidden_size, vocab_size]`.
    /// `None` iff [`Qwen3VlTextConfig::tied_word_embeddings`] is `true`
    /// (the LM head re-uses [`Self::token_embd`]).
    pub output: Option<MlxBuffer>,
    /// Mirror of [`Qwen3VlTextConfig::tied_word_embeddings`] — kept in
    /// the weights struct so the forward path can dispatch on it
    /// without needing the config back.
    pub tied_word_embeddings: bool,
    /// Mirror of [`Qwen3VlTextConfig::hidden_size`] — convenience.
    pub hidden_size: usize,
    /// Mirror of [`Qwen3VlTextConfig::vocab_size`] — convenience.
    pub vocab_size: usize,
    /// Mirror of [`Qwen3VlTextConfig::num_hidden_layers`] — convenience.
    pub num_hidden_layers: usize,
}

impl Qwen3VlTextWeights {
    /// Load all weights for a Qwen3-VL text LM from an open GGUF.
    ///
    /// Validates every tensor against [`Qwen3VlTextConfig`]: shapes
    /// must match exactly (no silent reshape). Any missing tensor or
    /// wrong shape produces a clear error message naming the tensor.
    ///
    /// `progress` drives the per-layer `\r loading i/n layers` banner.
    /// Pass a no-op [`LoadProgress`] if banner output is undesired
    /// (silent under non-TTY stderr or `-v` verbosity).
    pub fn load_from_gguf(
        gguf: &GgufFile,
        cfg: &Qwen3VlTextConfig,
        device: &MlxDevice,
        progress: &mut LoadProgress,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size as usize;
        let n_kv_heads = cfg.num_key_value_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let kv_dim = n_kv_heads * head_dim;
        let intermediate = cfg.intermediate_size as usize;
        let vocab = cfg.vocab_size as usize;

        // -------------------- Global tensors --------------------

        let token_embd = gguf
            .load_tensor("token_embd.weight", device)
            .map_err(|e| anyhow!("token_embd.weight load: {e}"))?;
        validate_shape(
            "token_embd.weight",
            &token_embd,
            &[vocab, hidden],
        )?;

        let output_norm = gguf
            .load_tensor_f32("output_norm.weight", device)
            .map_err(|e| anyhow!("output_norm.weight load: {e}"))?;
        validate_shape("output_norm.weight", &output_norm, &[hidden])?;

        let output = if cfg.tied_word_embeddings {
            None
        } else {
            let buf = gguf
                .load_tensor("output.weight", device)
                .map_err(|e| anyhow!("output.weight load: {e}"))?;
            validate_shape("output.weight", &buf, &[vocab, hidden])?;
            Some(buf)
        };

        // -------------------- Per-layer tensors --------------------

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for il in 0..cfg.num_hidden_layers as usize {
            let layer = load_layer(
                gguf,
                device,
                il,
                hidden,
                kv_dim,
                head_dim,
                intermediate,
            )
            .with_context(|| format!("Qwen3-VL text LM: layer {il}"))?;
            layers.push(layer);
            progress.on_layer(il + 1);
        }
        progress.finish();

        Ok(Self {
            token_embd,
            layers,
            output_norm,
            output,
            tied_word_embeddings: cfg.tied_word_embeddings,
            hidden_size: hidden,
            vocab_size: vocab,
            num_hidden_layers: cfg.num_hidden_layers as usize,
        })
    }
}

/// Validate that `buf`'s shape matches `expected` exactly. Names the
/// tensor in the error message so a mismatch points operators directly
/// at the offending tensor.
fn validate_shape(name: &str, buf: &MlxBuffer, expected: &[usize]) -> Result<()> {
    let actual = buf.shape();
    if actual != expected {
        return Err(anyhow!(
            "{name}: shape mismatch — got {actual:?}, expected {expected:?}"
        ));
    }
    Ok(())
}

/// iter-228a Phase-2c (Codex review of 3811f5d, finding #2 medium):
/// validate that the GGUF on-disk tensor type matches what the
/// downstream forward path expects. Without this, a same-shape tensor
/// of unexpected type (e.g. F16 in place of Q4_0) would load past the
/// shape check and produce garbage when iter-228b's forward dispatches
/// the tensor through the wrong qmatmul kernel.
///
/// Accepted ggml_types per tensor class on the canonical Qwen3-VL-2B
/// q4_0 GGUF (verified: 196 Q4_0, 113 F32, 1 F16, 0 attn biases):
///   - Quantized projections (attn_q/k/v/output, ffn_gate/up/down):
///     Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0 — any quantized type the
///     mlx-native qmatmul kernel registry accepts.
///   - Norms (attn_norm, ffn_norm, attn_q_norm, attn_k_norm,
///     output_norm): F32 (already enforced by load_tensor_f32 — kept
///     as a fact-of-life pin for future reorgs).
///   - Embedding (token_embd, output): F16 OR Q4_0 OR another quantized
///     type — operator may convert via different quants. Anything
///     gguf::load_tensor accepts is fine.
fn validate_quantized_proj_type(
    gguf: &mlx_native::gguf::GgufFile,
    name: &str,
) -> Result<()> {
    use mlx_native::ops::quantized_matmul_ggml::GgmlType;
    let info = gguf.tensor_info(name).ok_or_else(|| {
        anyhow!("{name}: tensor info missing (loader can't validate ggml_type)")
    })?;
    match info.ggml_type {
        GgmlType::Q4_0 | GgmlType::Q4_K
        | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0 => Ok(()),
        other => Err(anyhow!(
            "{name}: ggml_type {other:?} is not a quantized projection \
             type (expected Q4_0/Q4_K/Q5_K/Q6_K/Q8_0). Loading would \
             route through the wrong qmatmul kernel and produce garbage."
        )),
    }
}

fn load_layer(
    gguf: &GgufFile,
    device: &MlxDevice,
    il: usize,
    hidden: usize,
    kv_dim: usize,
    head_dim: usize,
    intermediate: usize,
) -> Result<Qwen3VlTextLayerWeights> {
    let attn_norm = gguf
        .load_tensor_f32(&format!("blk.{il}.attn_norm.weight"), device)
        .map_err(|e| anyhow!("blk.{il}.attn_norm.weight: {e}"))?;
    validate_shape("attn_norm", &attn_norm, &[hidden])?;

    let attn_q_name = format!("blk.{il}.attn_q.weight");
    validate_quantized_proj_type(gguf, &attn_q_name)?;
    let attn_q = gguf
        .load_tensor(&attn_q_name, device)
        .map_err(|e| anyhow!("{attn_q_name}: {e}"))?;
    validate_shape("attn_q", &attn_q, &[hidden, hidden])?;

    let attn_k_name = format!("blk.{il}.attn_k.weight");
    validate_quantized_proj_type(gguf, &attn_k_name)?;
    let attn_k = gguf
        .load_tensor(&attn_k_name, device)
        .map_err(|e| anyhow!("{attn_k_name}: {e}"))?;
    validate_shape("attn_k", &attn_k, &[kv_dim, hidden])?;

    let attn_v_name = format!("blk.{il}.attn_v.weight");
    validate_quantized_proj_type(gguf, &attn_v_name)?;
    let attn_v = gguf
        .load_tensor(&attn_v_name, device)
        .map_err(|e| anyhow!("{attn_v_name}: {e}"))?;
    validate_shape("attn_v", &attn_v, &[kv_dim, hidden])?;

    let attn_q_norm = gguf
        .load_tensor_f32(&format!("blk.{il}.attn_q_norm.weight"), device)
        .map_err(|e| anyhow!("blk.{il}.attn_q_norm.weight: {e}"))?;
    validate_shape("attn_q_norm", &attn_q_norm, &[head_dim])?;

    let attn_k_norm = gguf
        .load_tensor_f32(&format!("blk.{il}.attn_k_norm.weight"), device)
        .map_err(|e| anyhow!("blk.{il}.attn_k_norm.weight: {e}"))?;
    validate_shape("attn_k_norm", &attn_k_norm, &[head_dim])?;

    let attn_output_name = format!("blk.{il}.attn_output.weight");
    validate_quantized_proj_type(gguf, &attn_output_name)?;
    let attn_output = gguf
        .load_tensor(&attn_output_name, device)
        .map_err(|e| anyhow!("{attn_output_name}: {e}"))?;
    validate_shape("attn_output", &attn_output, &[hidden, hidden])?;

    let ffn_norm = gguf
        .load_tensor_f32(&format!("blk.{il}.ffn_norm.weight"), device)
        .map_err(|e| anyhow!("blk.{il}.ffn_norm.weight: {e}"))?;
    validate_shape("ffn_norm", &ffn_norm, &[hidden])?;

    let ffn_gate_name = format!("blk.{il}.ffn_gate.weight");
    validate_quantized_proj_type(gguf, &ffn_gate_name)?;
    let ffn_gate = gguf
        .load_tensor(&ffn_gate_name, device)
        .map_err(|e| anyhow!("{ffn_gate_name}: {e}"))?;
    validate_shape("ffn_gate", &ffn_gate, &[intermediate, hidden])?;

    let ffn_up_name = format!("blk.{il}.ffn_up.weight");
    validate_quantized_proj_type(gguf, &ffn_up_name)?;
    let ffn_up = gguf
        .load_tensor(&ffn_up_name, device)
        .map_err(|e| anyhow!("{ffn_up_name}: {e}"))?;
    validate_shape("ffn_up", &ffn_up, &[intermediate, hidden])?;

    let ffn_down_name = format!("blk.{il}.ffn_down.weight");
    validate_quantized_proj_type(gguf, &ffn_down_name)?;
    let ffn_down = gguf
        .load_tensor(&ffn_down_name, device)
        .map_err(|e| anyhow!("{ffn_down_name}: {e}"))?;
    validate_shape("ffn_down", &ffn_down, &[hidden, intermediate])?;

    Ok(Qwen3VlTextLayerWeights {
        attn_norm,
        attn_q,
        attn_k,
        attn_v,
        attn_q_norm,
        attn_k_norm,
        attn_output,
        ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
    })
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`Qwen3VlTextWeights`] use the operator-gated real
    //! GGUF (synthetic GGUFs cannot test the load path because that
    //! path actually reads tensor bytes — the synthetic helper in
    //! `mod.rs::test_fixtures` only fabricates header bytes). The
    //! integration test at `tests/qwen3vl_text_lm_forward.rs` runs the
    //! loader against the real 1.32 GB Qwen3-VL-2B GGUF when
    //! `HF2Q_QWEN3VL_LM_LOAD=1`.

    use super::*;

    #[test]
    fn load_real_qwen3vl_2b_gguf_when_operator_gated() {
        if std::env::var("HF2Q_QWEN3VL_LM_LOAD").ok().as_deref() != Some("1") {
            eprintln!("skip: HF2Q_QWEN3VL_LM_LOAD!=1");
            return;
        }
        let p = std::path::PathBuf::from(
            "/opt/hf2q/.cfa-archive/wedge4f-out/qwen3-vl-2b-q4_0.gguf",
        );
        if !p.exists() {
            eprintln!("skip: real GGUF fixture not present at {}", p.display());
            return;
        }
        let gguf = GgufFile::open(&p).expect("open real Qwen3-VL-2B GGUF");
        let cfg = Qwen3VlTextConfig::from_gguf(&gguf).expect("parse config");
        let device = MlxDevice::new().expect("Metal device init");
        let mut progress = LoadProgress::new(false, 0, cfg.num_hidden_layers as usize);
        let weights =
            Qwen3VlTextWeights::load_from_gguf(&gguf, &cfg, &device, &mut progress)
                .expect("load weights from real Qwen3-VL-2B GGUF");

        assert_eq!(weights.num_hidden_layers, 28);
        assert_eq!(weights.hidden_size, 2048);
        assert_eq!(weights.vocab_size, 151936);
        assert!(
            weights.tied_word_embeddings,
            "Qwen3-VL-2B has tied word embeddings"
        );
        assert!(
            weights.output.is_none(),
            "tied → no dedicated output buffer"
        );
        assert_eq!(weights.layers.len(), 28);
        // Pin one layer's tensor shapes to the post-reverse [outer, inner]
        // canonical values (gguf-dump shows on-disk innermost-first;
        // mlx-native reverses on read).
        let l0 = &weights.layers[0];
        assert_eq!(l0.attn_norm.shape(), &[2048]);
        assert_eq!(l0.attn_q.shape(), &[2048, 2048]);
        // disk: 2048, 1024 → in-memory [1024, 2048] = [n_kv_heads*head_dim, hidden]
        assert_eq!(l0.attn_k.shape(), &[1024, 2048]);
        assert_eq!(l0.attn_v.shape(), &[1024, 2048]);
        assert_eq!(l0.attn_q_norm.shape(), &[128]);
        assert_eq!(l0.attn_k_norm.shape(), &[128]);
        assert_eq!(l0.attn_output.shape(), &[2048, 2048]);
        assert_eq!(l0.ffn_norm.shape(), &[2048]);
        // disk: 2048, 6144 → in-memory [6144, 2048] = [intermediate, hidden]
        assert_eq!(l0.ffn_gate.shape(), &[6144, 2048]);
        assert_eq!(l0.ffn_up.shape(), &[6144, 2048]);
        // disk: 6144, 2048 → in-memory [2048, 6144] = [hidden, intermediate]
        assert_eq!(l0.ffn_down.shape(), &[2048, 6144]);
    }
}
