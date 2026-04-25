//! BERT GGUF weight loader (ADR-005 Phase 2b, Task #13).
//!
//! Reads every tensor from a parsed `GgufFile` onto the Metal device as
//! F32 buffers, dequantizing Q-type tensors on the CPU first via
//! `mlx_native::gguf::GgufFile::load_tensor_f32`. Mirrors the Phase 2c
//! `LoadedMmprojWeights` pattern: arch-agnostic walk of
//! `gguf.tensor_names()`, per-tensor `MlxBuffer`s keyed by GGUF name,
//! shortcut accessors for the stem + per-block tensors that the encoder
//! forward pass needs.
//!
//! # Sequencing
//!
//! 1. `BertConfig::from_gguf` — parse architecture metadata (hidden
//!    size, layer count, pooling, etc.). Cheap.
//! 2. `validate_tensor_set` (this module) — confirm every required
//!    tensor name is present BEFORE the expensive load. Saves operators
//!    from waiting through a multi-GB load only to fail on a missing
//!    `blk.5.attn_q.weight`.
//! 3. `LoadedBertWeights::load` — pull every tensor onto the device as
//!    F32. Total cost dominated by file read + dequant; ~50–500 MB
//!    depending on the model.
//! 4. (future iter) `bert_forward.rs` — encoder forward pass + pooling.
//!
//! # Day-one supported models (per ADR-005 Phase 2b)
//!
//! - `nomic-embed-text-v1.5` (137M params, hidden=768, layers=12)
//! - `mxbai-embed-large-v1` (335M params, hidden=1024, layers=24)
//! - `bge-small-en-v1.5` (33M params, hidden=384, layers=12)
//!
//! All three share the llama.cpp `bert.*` GGUF metadata convention and
//! the per-layer tensor names below. Variants (e.g. nomic's RoPE-based
//! version of BERT, or mxbai's v2 with longer context) are validated as
//! day-one models surface; the loader is structured so adding a new BERT
//! variant is a tensor-set diff, not a rewrite.
//!
//! # Optional tensors
//!
//! Some BERTs lack `token_types.weight` (single-segment models) or
//! `attn_*.bias` (bias-free variants). The validator below treats those
//! as **optional** — `validate_tensor_set` does not flag them missing —
//! and the accessor returns `None` (not `Err`) so the forward pass can
//! branch on presence. The required minimum is the LayerNorm + linear
//! weight set that every BERT variant ships.

#![allow(dead_code)] // forward pass + handler wiring lands in iter 56+

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use mlx_native::gguf::GgufFile;
use mlx_native::{MlxBuffer, MlxDevice};

use super::config::{
    bert_layer_tensor, BertConfig, TENSOR_EMBED_NORM_BIAS, TENSOR_EMBED_NORM_WEIGHT,
    TENSOR_POS_EMBD, TENSOR_TOKEN_EMBD, TENSOR_TOKEN_TYPES,
};

// ---------------------------------------------------------------------------
// Per-layer tensor suffixes
// ---------------------------------------------------------------------------

/// Required per-layer suffixes (every BERT variant ships these). The
/// validator + loader use this list to confirm the GGUF is complete
/// before dispatching the forward pass.
///
/// Listed in approximate forward-pass order:
///   QKV linear weights → output projection → post-attn LN →
///   FFN up/down → post-FFN LN.
pub const BERT_BLOCK_REQUIRED_SUFFIXES: &[&str] = &[
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "attn_output_norm.weight",
    "attn_output_norm.bias",
    "ffn_up.weight",
    "ffn_down.weight",
    "layer_output_norm.weight",
    "layer_output_norm.bias",
];

/// Optional per-layer suffixes — present in *most* BERT variants but
/// some bias-free / fused variants drop them. Loader treats absence as
/// "no bias" (forward pass branches on presence).
pub const BERT_BLOCK_OPTIONAL_SUFFIXES: &[&str] = &[
    "attn_q.bias",
    "attn_k.bias",
    "attn_v.bias",
    "attn_output.bias",
    "ffn_up.bias",
    "ffn_down.bias",
];

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

/// Confirm every required tensor exists in the GGUF before the
/// expensive load. Returns the missing-name list on failure (sorted for
/// stable error output).
///
/// `cfg.num_hidden_layers` drives the per-block expansion. Stem tensors
/// (`token_embd.weight`, `position_embd.weight`, `token_embd_norm.*`)
/// are required unconditionally; `token_types.weight` is treated as
/// optional because some encoders lack a segment table.
pub fn validate_tensor_set(gguf: &GgufFile, cfg: &BertConfig) -> Result<()> {
    let names: std::collections::HashSet<&str> = gguf.tensor_names().into_iter().collect();
    let mut missing: Vec<String> = Vec::new();

    // Required stem tensors.
    for n in &[
        TENSOR_TOKEN_EMBD,
        TENSOR_POS_EMBD,
        TENSOR_EMBED_NORM_WEIGHT,
        TENSOR_EMBED_NORM_BIAS,
    ] {
        if !names.contains(*n) {
            missing.push((*n).to_string());
        }
    }

    // Per-layer required suffixes.
    for layer_idx in 0..cfg.num_hidden_layers {
        for suffix in BERT_BLOCK_REQUIRED_SUFFIXES {
            let key = bert_layer_tensor(layer_idx, suffix);
            if !names.contains(key.as_str()) {
                missing.push(key);
            }
        }
    }

    if !missing.is_empty() {
        missing.sort();
        return Err(anyhow!(
            "BERT GGUF missing {} tensor(s): {}",
            missing.len(),
            missing.join(", ")
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// LoadedBertWeights
// ---------------------------------------------------------------------------

/// Collection of BERT tensors loaded onto a Metal device as F32.
///
/// Cheap to move; cloning requires the caller to pay the GPU-alloc cost
/// again (not implemented here — wrap in `Arc` at the call site if
/// cheap-cloning is needed). Field access goes through the named
/// shortcut methods; `get(name)` is the escape hatch for tensors not
/// covered by a shortcut.
pub struct LoadedBertWeights {
    /// Keyed by the tensor's GGUF name. Values are F32 `MlxBuffer`s
    /// with shape preserved from the source GGUF.
    tensors: HashMap<String, MlxBuffer>,
    /// Device handle kept alive for the lifetime of the buffers. Held
    /// for RAII even though public accessors go through `tensors`.
    _device: MlxDevice,
}

impl std::fmt::Debug for LoadedBertWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedBertWeights")
            .field("tensor_count", &self.tensors.len())
            .finish()
    }
}

impl LoadedBertWeights {
    /// Load every tensor from the GGUF onto the supplied device as F32.
    /// Arch-agnostic — walks `gguf.tensor_names()` rather than driving
    /// off a hardcoded list, so optional tensors and any extra
    /// debug-only tensors a producer might emit are loaded transparently.
    /// Caller should run `validate_tensor_set` first to fail fast on a
    /// missing required tensor.
    pub fn load(gguf: &GgufFile, _cfg: &BertConfig, device: MlxDevice) -> Result<Self> {
        let names = gguf.tensor_names();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in &names {
            let buf = gguf
                .load_tensor_f32(*name, &device)
                .map_err(|e| anyhow!("BERT load_tensor_f32('{}'): {e}", name))?;
            tensors.insert((*name).to_string(), buf);
        }
        Ok(Self {
            tensors,
            _device: device,
        })
    }

    /// Open + load convenience: opens the file at `path`, creates a
    /// default `MlxDevice`, validates, and loads. Used by the server
    /// startup path when the operator passes `--embedding-model X.gguf`.
    pub fn load_from_path(path: &Path, cfg: &BertConfig) -> Result<Self> {
        let gguf = GgufFile::open(path)
            .map_err(|e| anyhow!("open BERT GGUF {}: {e}", path.display()))?;
        validate_tensor_set(&gguf, cfg)?;
        let device = MlxDevice::new().map_err(|e| anyhow!("create MlxDevice for BERT load: {e}"))?;
        Self::load(&gguf, cfg, device)
    }

    /// Empty placeholder — used by tests that need a `LoadedBertWeights`
    /// shape but do not drive a forward pass. Every shortcut accessor
    /// returns `Err` on this instance; `get()` returns `None`.
    pub fn empty(device: MlxDevice) -> Self {
        Self {
            tensors: HashMap::new(),
            _device: device,
        }
    }

    /// Build a `LoadedBertWeights` from a name→buffer map. Test-only
    /// escape hatch for the full-forward parity test (iter 61) — the
    /// production code path is `load`/`load_from_path`. The function
    /// is `pub(crate)` so it can be invoked from sibling test modules
    /// (e.g. `bert_gpu`'s `apply_bert_full_forward_gpu` test) without
    /// constructing a synthetic GGUF on disk.
    #[cfg(test)]
    pub(crate) fn from_tensors_for_test(
        tensors: HashMap<String, MlxBuffer>,
        device: MlxDevice,
    ) -> Self {
        Self {
            tensors,
            _device: device,
        }
    }

    /// Total tensor count.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Empty when no tensors loaded (only possible from `Self::empty`).
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Look up a tensor by exact GGUF name. `None` when absent.
    pub fn get(&self, name: &str) -> Option<&MlxBuffer> {
        self.tensors.get(name)
    }

    // -----------------------------------------------------------------------
    // Stem shortcuts. Errors carry the missing-tensor name in the message
    // so a forward-pass failure is debuggable from the log alone.
    // -----------------------------------------------------------------------

    pub fn token_embd_weight(&self) -> Result<&MlxBuffer> {
        self.tensors
            .get(TENSOR_TOKEN_EMBD)
            .ok_or_else(|| anyhow!("BERT missing '{}'", TENSOR_TOKEN_EMBD))
    }

    pub fn position_embd_weight(&self) -> Result<&MlxBuffer> {
        self.tensors
            .get(TENSOR_POS_EMBD)
            .ok_or_else(|| anyhow!("BERT missing '{}'", TENSOR_POS_EMBD))
    }

    /// Optional — returns `None` when the model has no segment table.
    pub fn token_types_weight(&self) -> Option<&MlxBuffer> {
        self.tensors.get(TENSOR_TOKEN_TYPES)
    }

    pub fn embed_norm_weight(&self) -> Result<&MlxBuffer> {
        self.tensors
            .get(TENSOR_EMBED_NORM_WEIGHT)
            .ok_or_else(|| anyhow!("BERT missing '{}'", TENSOR_EMBED_NORM_WEIGHT))
    }

    pub fn embed_norm_bias(&self) -> Result<&MlxBuffer> {
        self.tensors
            .get(TENSOR_EMBED_NORM_BIAS)
            .ok_or_else(|| anyhow!("BERT missing '{}'", TENSOR_EMBED_NORM_BIAS))
    }

    // -----------------------------------------------------------------------
    // Per-block accessors. The forward pass calls these in a layer loop.
    // -----------------------------------------------------------------------

    /// Required per-block tensor (errors if missing).
    pub fn block_required(&self, layer_idx: usize, suffix: &str) -> Result<&MlxBuffer> {
        let key = bert_layer_tensor(layer_idx, suffix);
        self.tensors
            .get(&key)
            .ok_or_else(|| anyhow!("BERT missing '{}'", key))
    }

    /// Optional per-block tensor (returns `None` when absent — e.g. a
    /// bias-free model, or `attn_q.bias` on a fused-attention variant).
    pub fn block_optional(&self, layer_idx: usize, suffix: &str) -> Option<&MlxBuffer> {
        let key = bert_layer_tensor(layer_idx, suffix);
        self.tensors.get(&key)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::config::PoolingType;
    use super::*;

    /// Build a synthetic config that drives `validate_tensor_set` and
    /// the per-block accessor loop. `num_hidden_layers=2` keeps the
    /// expected-tensor list short enough to enumerate inline.
    fn synthetic_cfg(layers: usize) -> BertConfig {
        BertConfig {
            hidden_size: 384,
            num_attention_heads: 12,
            num_hidden_layers: layers,
            intermediate_size: 1536,
            max_position_embeddings: 512,
            vocab_size: 30522,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
            pooling_type: PoolingType::Mean,
            causal_attention: false,
        }
    }

    /// Required tensor names for the synthetic 2-layer config — the
    /// minimum set `validate_tensor_set` must accept without error.
    fn synthetic_required_names(cfg: &BertConfig) -> Vec<String> {
        let mut out = vec![
            TENSOR_TOKEN_EMBD.to_string(),
            TENSOR_POS_EMBD.to_string(),
            TENSOR_EMBED_NORM_WEIGHT.to_string(),
            TENSOR_EMBED_NORM_BIAS.to_string(),
        ];
        for layer_idx in 0..cfg.num_hidden_layers {
            for suffix in BERT_BLOCK_REQUIRED_SUFFIXES {
                out.push(bert_layer_tensor(layer_idx, suffix));
            }
        }
        out
    }

    #[test]
    fn block_required_suffixes_cover_every_forward_pass_op() {
        // Spot-check the suffix list matches what a transformer block
        // forward pass actually needs. Changes here must update the
        // forward pass + this test in lockstep.
        for s in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_output_norm.weight",
            "attn_output_norm.bias",
            "ffn_up.weight",
            "ffn_down.weight",
            "layer_output_norm.weight",
            "layer_output_norm.bias",
        ] {
            assert!(
                BERT_BLOCK_REQUIRED_SUFFIXES.contains(&s),
                "missing required suffix '{}'",
                s
            );
        }
    }

    #[test]
    fn block_optional_suffixes_are_biases_only() {
        // Optional list must be biases — variants drop biases, never
        // weights. If a future BERT variant turns out to drop a weight,
        // the model is genuinely incompatible and should error in the
        // validator, not silently load.
        for s in BERT_BLOCK_OPTIONAL_SUFFIXES {
            assert!(s.ends_with(".bias"), "optional must be .bias: '{}'", s);
        }
    }

    #[test]
    fn synthetic_required_names_count_matches_config() {
        let cfg = synthetic_cfg(2);
        let names = synthetic_required_names(&cfg);
        // 4 stem + 10 per-block × 2 blocks = 24 required.
        assert_eq!(names.len(), 4 + BERT_BLOCK_REQUIRED_SUFFIXES.len() * 2);
        // Spot-check expansion.
        assert!(names.contains(&"blk.0.attn_q.weight".to_string()));
        assert!(names.contains(&"blk.1.layer_output_norm.bias".to_string()));
    }

    #[test]
    fn empty_loaded_weights_returns_errs_from_shortcuts() {
        let device = MlxDevice::new().expect("create device");
        let w = LoadedBertWeights::empty(device);
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
        assert!(w.token_embd_weight().is_err());
        assert!(w.position_embd_weight().is_err());
        assert!(w.embed_norm_weight().is_err());
        assert!(w.embed_norm_bias().is_err());
        assert!(w.token_types_weight().is_none());
        assert!(w.block_required(0, "attn_q.weight").is_err());
        assert!(w.block_optional(0, "attn_q.bias").is_none());
        assert!(w.get("anything").is_none());
    }

    /// Real BERT GGUFs aren't on disk (Phase 2b downloads them in iter
    /// 57+), but the vocab GGUFs at /opt/llama.cpp/models/ exercise the
    /// architecture-validation branch. Vocab GGUFs deliberately lack
    /// the weight tensors, so `validate_tensor_set` must report a
    /// specific missing-list rather than panic.
    #[test]
    fn validate_tensor_set_on_vocab_only_gguf_reports_missing_tensors() {
        let path = Path::new("/opt/llama.cpp/models/ggml-vocab-bert-bge.gguf");
        if !path.exists() {
            eprintln!(
                "skipping: vocab GGUF fixture not found at {}",
                path.display()
            );
            return;
        }
        let gguf = GgufFile::open(path).expect("open vocab gguf");
        // The vocab-only GGUF doesn't have a parseable BertConfig (no
        // `bert.embedding_length` etc. — it's tokenizer metadata only).
        // Drive the validator with a synthetic cfg so the test stays
        // independent of GGUF contents.
        let cfg = synthetic_cfg(2);
        let err = validate_tensor_set(&gguf, &cfg).expect_err("vocab-only must miss tensors");
        let msg = format!("{}", err);
        assert!(msg.contains("missing"), "error names missing: {msg}");
        // Naming a specific stem tensor proves the message is useful
        // for debugging.
        assert!(
            msg.contains(TENSOR_TOKEN_EMBD)
                || msg.contains("blk.0")
                || msg.contains("attn_q.weight"),
            "error should name a specific missing tensor: {msg}"
        );
    }
}
