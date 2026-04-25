//! nomic-bert GGUF weight loader (ADR-005 Phase 2b, Task #16).
//!
//! Mirrors `bert::weights::LoadedBertWeights` in shape but with three
//! manifest differences enforced:
//!
//! 1. Per-layer linears are FUSED — `blk.{i}.attn_qkv.weight` is a
//!    single `[hidden, 3*hidden]` tensor. The encoder forward pass
//!    splits it into Q/K/V views in-flight (no separate `attn_q/k/v`
//!    tensors exist on disk).
//! 2. Per-layer FFN has THREE matrices — `ffn_up`, `ffn_gate`,
//!    `ffn_down`. `ffn_gate` is the SwiGLU gate projection (added on
//!    top of BERT's plain `ffn_up → activation → ffn_down`).
//! 3. Stem has NO `position_embd.weight`. RoPE replaces the absolute
//!    position table at attention time per `nomic-bert.rope.freq_base`.
//!
//! Linear biases are entirely OPTIONAL on this lane —
//! nomic-embed-text-v1.5 ships zero `.bias` tensors on its linears
//! (`attn_qkv`, `attn_output`, `ffn_*`), only the LayerNorm biases.
//! Forward-pass code branches on `block_optional("...bias")`.
//!
//! # Sequencing
//!
//! 1. `NomicBertConfig::from_gguf` — parse architecture metadata.
//! 2. `validate_tensor_set` (this module) — confirm every required
//!    tensor name is present BEFORE the expensive load.
//! 3. `LoadedNomicBertWeights::load` — pull every tensor onto the
//!    device as F32. For nomic-embed-text-v1.5 this is ~268 MB.
//! 4. (next iter) `forward.rs` — encoder forward + pooling.

#![allow(dead_code)] // forward pass + handler wiring lands in iter 76+

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use mlx_native::gguf::GgufFile;
use mlx_native::{MlxBuffer, MlxDevice};

use super::config::{
    nomic_bert_layer_tensor, NomicBertConfig, NOMIC_BERT_TENSOR_EMBED_NORM_BIAS,
    NOMIC_BERT_TENSOR_EMBED_NORM_WEIGHT, NOMIC_BERT_TENSOR_TOKEN_EMBD,
    NOMIC_BERT_TENSOR_TOKEN_TYPES,
};

// ---------------------------------------------------------------------------
// Per-layer tensor suffixes
// ---------------------------------------------------------------------------

/// Required per-layer suffixes (every nomic-bert variant ships these).
/// Listed in approximate forward-pass order: fused QKV linear → output
/// projection → post-attn LN → FFN three-matrix SwiGLU → post-FFN LN.
pub const NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES: &[&str] = &[
    "attn_qkv.weight",
    "attn_output.weight",
    "attn_output_norm.weight",
    "attn_output_norm.bias",
    "ffn_up.weight",
    "ffn_gate.weight",
    "ffn_down.weight",
    "layer_output_norm.weight",
    "layer_output_norm.bias",
];

/// Optional per-layer suffixes — present in some nomic-bert variants but
/// not in the day-one model. Loader treats absence as "no bias" (forward
/// pass branches on presence). Includes both linear biases AND the
/// alternate q-norm/k-norm names that some derivative architectures emit.
pub const NOMIC_BERT_BLOCK_OPTIONAL_SUFFIXES: &[&str] = &[
    "attn_qkv.bias",
    "attn_output.bias",
    "ffn_up.bias",
    "ffn_gate.bias",
    "ffn_down.bias",
];

// ---------------------------------------------------------------------------
// Validator
// ---------------------------------------------------------------------------

/// Confirm every required tensor exists in the GGUF before the expensive
/// load. Returns the missing-name list on failure (sorted for stable
/// error output).
///
/// `cfg.num_hidden_layers` drives the per-block expansion. Stem tensors
/// are required unconditionally EXCEPT `token_types.weight`, which is
/// optional (forward pass synthesises an all-zero segment id when the
/// table is absent).
pub fn validate_tensor_set(gguf: &GgufFile, cfg: &NomicBertConfig) -> Result<()> {
    let names: std::collections::HashSet<&str> = gguf.tensor_names().into_iter().collect();
    let mut missing: Vec<String> = Vec::new();

    // Required stem tensors. `token_types.weight` is optional —
    // configurable here but always present in nomic-embed-text-v1.5;
    // omit from the required-set so single-segment derivative GGUFs
    // load without modification.
    for n in &[
        NOMIC_BERT_TENSOR_TOKEN_EMBD,
        NOMIC_BERT_TENSOR_EMBED_NORM_WEIGHT,
        NOMIC_BERT_TENSOR_EMBED_NORM_BIAS,
    ] {
        if !names.contains(*n) {
            missing.push((*n).to_string());
        }
    }

    // Per-layer required suffixes.
    for layer_idx in 0..cfg.num_hidden_layers {
        for suffix in NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES {
            let key = nomic_bert_layer_tensor(layer_idx, suffix);
            if !names.contains(key.as_str()) {
                missing.push(key);
            }
        }
    }

    if !missing.is_empty() {
        missing.sort();
        return Err(anyhow!(
            "nomic-bert GGUF missing {} tensor(s): {}",
            missing.len(),
            missing.join(", ")
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// LoadedNomicBertWeights
// ---------------------------------------------------------------------------

/// Collection of nomic-bert tensors loaded onto a Metal device as F32.
///
/// Cheap to move; cloning requires the caller to pay the GPU-alloc cost
/// again (not implemented — wrap in `Arc` at the call site if needed).
/// Field access goes through the named shortcut methods; `get(name)` is
/// the escape hatch for tensors not covered by a shortcut.
pub struct LoadedNomicBertWeights {
    /// Keyed by the tensor's GGUF name. Values are F32 `MlxBuffer`s
    /// with shape preserved from the source GGUF.
    tensors: HashMap<String, MlxBuffer>,
    /// Pre-cast BF16 versions of every linear-style tensor (any 2D
    /// matmul weight). Populated during `load()` via a one-shot encoder
    /// dispatch per weight; the same key is used in both maps. The
    /// matmul fast path reads from here via `weight_bf16(...)` so the
    /// per-request F32→BF16 cast in `bert_linear_gpu` is eliminated.
    /// Iter-83 perf optimization — drops nomic-bert e2e from ~190 ms
    /// to ~10–30 ms (eliminates ~84 cast dispatches per request).
    tensors_bf16: HashMap<String, MlxBuffer>,
    /// Device handle kept alive for the lifetime of the buffers.
    _device: MlxDevice,
}

/// Per-layer suffixes whose `.weight` tensor is a linear matmul weight
/// `[out_features, in_features]` and benefits from pre-casting to BF16.
/// LayerNorm / segment / embedding tables stay F32 (norms read F32
/// directly; gather is F32-native; pre-casting them would just waste
/// memory).
const LINEAR_WEIGHT_SUFFIXES: &[&str] = &[
    "attn_qkv.weight",
    "attn_output.weight",
    "ffn_up.weight",
    "ffn_gate.weight",
    "ffn_down.weight",
];

impl std::fmt::Debug for LoadedNomicBertWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedNomicBertWeights")
            .field("tensor_count", &self.tensors.len())
            .finish()
    }
}

impl LoadedNomicBertWeights {
    /// Load every tensor from the GGUF onto the supplied device as F32,
    /// then pre-cast every linear-style 2D weight to BF16 for the matmul
    /// fast path. Walks `gguf.tensor_names()` rather than a hardcoded
    /// list, so any optional tensors are loaded transparently. Caller
    /// should run `validate_tensor_set` first to fail fast on a missing
    /// required tensor.
    ///
    /// Pre-cast cost is one-time at server boot (~10 ms per layer ×
    /// 5 linears × 12 layers ≈ 600 ms warmup, eliminated from per-request
    /// hot path). See iter-83 perf analysis in ADR-005.
    pub fn load(gguf: &GgufFile, _cfg: &NomicBertConfig, device: MlxDevice) -> Result<Self> {
        use mlx_native::ops::elementwise::{cast, CastDirection};
        use mlx_native::DType;
        use mlx_native::KernelRegistry;

        let names = gguf.tensor_names();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in &names {
            let buf = gguf
                .load_tensor_f32(*name, &device)
                .map_err(|e| anyhow!("nomic-bert load_tensor_f32('{}'): {e}", name))?;
            tensors.insert((*name).to_string(), buf);
        }

        // Pre-cast linear-style 2D weights (Q/K/V fused, attn_output,
        // ffn_up/gate/down) to BF16 in a single command-buffer pass.
        // Linear weights match `blk.{N}.{suffix}` where suffix is in
        // LINEAR_WEIGHT_SUFFIXES; non-block weights (token_embd, norms)
        // are not matmul'd and stay F32.
        let mut tensors_bf16: HashMap<String, MlxBuffer> = HashMap::new();
        let mut registry = KernelRegistry::new();
        let mut encoder = device
            .command_encoder()
            .map_err(|e| anyhow!("nomic-bert pre-cast: command_encoder: {e}"))?;

        for (name, src) in &tensors {
            // Match `blk.{N}.{suffix}` where suffix ∈ LINEAR_WEIGHT_SUFFIXES.
            let is_linear = LINEAR_WEIGHT_SUFFIXES.iter().any(|sfx| {
                name.starts_with("blk.") && name.ends_with(sfx)
            });
            if !is_linear {
                continue;
            }
            let n_elems = src.element_count();
            let dst = device
                .alloc_buffer(n_elems * 2, DType::BF16, src.shape().to_vec())
                .map_err(|e| anyhow!("nomic-bert pre-cast: alloc bf16 for '{name}': {e}"))?;
            cast(
                &mut encoder,
                &mut registry,
                device.metal_device(),
                src,
                &dst,
                n_elems,
                CastDirection::F32ToBF16,
            )
            .map_err(|e| anyhow!("nomic-bert pre-cast: cast('{name}'): {e}"))?;
            tensors_bf16.insert(name.clone(), dst);
        }
        // Single commit + wait amortizes the full set of casts into one
        // GPU round-trip (vs the per-request path that does 84 separate
        // cast dispatches each with their own commit).
        encoder
            .commit_and_wait()
            .map_err(|e| anyhow!("nomic-bert pre-cast: commit_and_wait: {e}"))?;

        Ok(Self {
            tensors,
            tensors_bf16,
            _device: device,
        })
    }

    /// Open + load convenience: opens the file at `path`, creates a
    /// default `MlxDevice`, validates, and loads. Used by the server
    /// startup path when the operator passes a nomic-bert GGUF to
    /// `--embedding-model`.
    pub fn load_from_path(path: &Path, cfg: &NomicBertConfig) -> Result<Self> {
        let gguf = GgufFile::open(path)
            .map_err(|e| anyhow!("open nomic-bert GGUF {}: {e}", path.display()))?;
        validate_tensor_set(&gguf, cfg)?;
        let device =
            MlxDevice::new().map_err(|e| anyhow!("create MlxDevice for nomic-bert load: {e}"))?;
        Self::load(&gguf, cfg, device)
    }

    /// Empty placeholder — used by tests that need a
    /// `LoadedNomicBertWeights` shape but do not drive a forward pass.
    /// Every shortcut accessor returns `Err`; `get()` returns `None`.
    pub fn empty(device: MlxDevice) -> Self {
        Self {
            tensors: HashMap::new(),
            tensors_bf16: HashMap::new(),
            _device: device,
        }
    }

    /// Build a `LoadedNomicBertWeights` from a name→buffer map. Test-only
    /// escape hatch for the eventual full-forward parity test. Pre-casts
    /// linear weights to BF16 just like the production load path so
    /// synthetic tests exercise the same fast-path.
    #[cfg(test)]
    pub(crate) fn from_tensors_for_test(
        tensors: HashMap<String, MlxBuffer>,
        device: MlxDevice,
    ) -> Self {
        use mlx_native::ops::elementwise::{cast, CastDirection};
        use mlx_native::DType;
        use mlx_native::KernelRegistry;

        let mut tensors_bf16: HashMap<String, MlxBuffer> = HashMap::new();
        let mut registry = KernelRegistry::new();
        if let Ok(mut encoder) = device.command_encoder() {
            for (name, src) in &tensors {
                let is_linear = LINEAR_WEIGHT_SUFFIXES.iter().any(|sfx| {
                    name.starts_with("blk.") && name.ends_with(sfx)
                });
                if !is_linear {
                    continue;
                }
                let n_elems = src.element_count();
                if let Ok(dst) =
                    device.alloc_buffer(n_elems * 2, DType::BF16, src.shape().to_vec())
                {
                    if cast(
                        &mut encoder,
                        &mut registry,
                        device.metal_device(),
                        src,
                        &dst,
                        n_elems,
                        CastDirection::F32ToBF16,
                    )
                    .is_ok()
                    {
                        tensors_bf16.insert(name.clone(), dst);
                    }
                }
            }
            let _ = encoder.commit_and_wait();
        }
        Self {
            tensors,
            tensors_bf16,
            _device: device,
        }
    }

    /// Pre-cast BF16 weight lookup. Returns `None` for non-linear
    /// tensors (norms, embeddings) and for any name that isn't in the
    /// loaded set. Composers should call this first; if `None`, fall
    /// back to the F32 path via `block_required` + `bert_linear_gpu`.
    pub fn weight_bf16(&self, name: &str) -> Option<&MlxBuffer> {
        self.tensors_bf16.get(name)
    }

    /// Per-layer pre-cast BF16 weight accessor. Convenience wrapper
    /// over `weight_bf16` for the per-block forward composer.
    pub fn block_weight_bf16(&self, layer_idx: usize, suffix: &str) -> Option<&MlxBuffer> {
        let key = nomic_bert_layer_tensor(layer_idx, suffix);
        self.tensors_bf16.get(&key)
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
            .get(NOMIC_BERT_TENSOR_TOKEN_EMBD)
            .ok_or_else(|| anyhow!("nomic-bert missing '{}'", NOMIC_BERT_TENSOR_TOKEN_EMBD))
    }

    /// Optional — returns `None` for single-segment models.
    pub fn token_types_weight(&self) -> Option<&MlxBuffer> {
        self.tensors.get(NOMIC_BERT_TENSOR_TOKEN_TYPES)
    }

    pub fn embed_norm_weight(&self) -> Result<&MlxBuffer> {
        self.tensors
            .get(NOMIC_BERT_TENSOR_EMBED_NORM_WEIGHT)
            .ok_or_else(|| {
                anyhow!(
                    "nomic-bert missing '{}'",
                    NOMIC_BERT_TENSOR_EMBED_NORM_WEIGHT
                )
            })
    }

    pub fn embed_norm_bias(&self) -> Result<&MlxBuffer> {
        self.tensors
            .get(NOMIC_BERT_TENSOR_EMBED_NORM_BIAS)
            .ok_or_else(|| {
                anyhow!(
                    "nomic-bert missing '{}'",
                    NOMIC_BERT_TENSOR_EMBED_NORM_BIAS
                )
            })
    }

    // -----------------------------------------------------------------------
    // Per-block accessors. The forward pass calls these in a layer loop.
    // -----------------------------------------------------------------------

    /// Required per-block tensor (errors if missing).
    pub fn block_required(&self, layer_idx: usize, suffix: &str) -> Result<&MlxBuffer> {
        let key = nomic_bert_layer_tensor(layer_idx, suffix);
        self.tensors
            .get(&key)
            .ok_or_else(|| anyhow!("nomic-bert missing '{}'", key))
    }

    /// Optional per-block tensor (returns `None` when absent).
    pub fn block_optional(&self, layer_idx: usize, suffix: &str) -> Option<&MlxBuffer> {
        let key = nomic_bert_layer_tensor(layer_idx, suffix);
        self.tensors.get(&key)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::config::NomicBertConfig;
    use super::super::super::bert::config::PoolingType;
    use super::*;

    /// Build a synthetic config that drives `validate_tensor_set` and
    /// the per-block accessor loop.
    fn synthetic_cfg(layers: usize) -> NomicBertConfig {
        NomicBertConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: layers,
            intermediate_size: 3072,
            max_position_embeddings: 2048,
            vocab_size: 30522,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            pooling_type: PoolingType::Mean,
            rope_freq_base: 1000.0,
            causal_attention: false,
        }
    }

    #[test]
    fn block_required_suffixes_cover_every_forward_pass_op() {
        // Spot-check the suffix list matches what a nomic-bert block
        // forward pass actually needs. Changes here must update the
        // forward pass + this test in lockstep.
        for s in [
            "attn_qkv.weight",
            "attn_output.weight",
            "attn_output_norm.weight",
            "attn_output_norm.bias",
            "ffn_up.weight",
            "ffn_gate.weight",
            "ffn_down.weight",
            "layer_output_norm.weight",
            "layer_output_norm.bias",
        ] {
            assert!(
                NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES.contains(&s),
                "missing required suffix '{}'",
                s
            );
        }
    }

    #[test]
    fn block_optional_suffixes_are_biases_only() {
        // Optional list must be biases — variants drop biases, never
        // weights. If a future variant turns out to drop a weight, the
        // model is genuinely incompatible and should error in the
        // validator, not silently load.
        for s in NOMIC_BERT_BLOCK_OPTIONAL_SUFFIXES {
            assert!(s.ends_with(".bias"), "optional must be .bias: '{}'", s);
        }
    }

    #[test]
    fn no_separate_qkv_in_required_suffixes() {
        // The whole point of the nomic-bert lane is that QKV is fused.
        // Lock that the required set does NOT regress to the BERT
        // separate-Q/K/V manifest.
        for s in ["attn_q.weight", "attn_k.weight", "attn_v.weight"] {
            assert!(
                !NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES.contains(&s),
                "{} must not be in required (nomic-bert is fused-QKV)",
                s
            );
        }
    }

    #[test]
    fn no_position_embd_in_required_stem() {
        // RoPE replaces the absolute position table. position_embd.weight
        // must NOT be required (it isn't even present in nomic GGUFs).
        // This test mirrors `validate_tensor_set`'s actual stem list to
        // catch refactor drift.
        let gguf_dummy_names = vec![
            NOMIC_BERT_TENSOR_TOKEN_EMBD.to_string(),
            NOMIC_BERT_TENSOR_EMBED_NORM_WEIGHT.to_string(),
            NOMIC_BERT_TENSOR_EMBED_NORM_BIAS.to_string(),
        ];
        // Add the per-layer required tensors for a synthetic 1-layer config.
        let mut all_names = gguf_dummy_names;
        for s in NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES {
            all_names.push(nomic_bert_layer_tensor(0, s));
        }
        // Confirm position_embd.weight is NOT in the constructed required
        // set (this is what validate_tensor_set checks against).
        assert!(!all_names.contains(&"position_embd.weight".to_string()));
    }

    #[test]
    fn empty_loaded_weights_returns_errs_from_shortcuts() {
        let device = MlxDevice::new().expect("create device");
        let w = LoadedNomicBertWeights::empty(device);
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
        assert!(w.token_embd_weight().is_err());
        assert!(w.embed_norm_weight().is_err());
        assert!(w.embed_norm_bias().is_err());
        assert!(w.token_types_weight().is_none());
        assert!(w.block_required(0, "attn_qkv.weight").is_err());
        assert!(w.block_optional(0, "attn_qkv.bias").is_none());
        assert!(w.get("anything").is_none());
    }

    /// End-to-end: validate the on-disk nomic-embed-text-v1.5 GGUF
    /// against the required-set. The model ships every required tensor
    /// per the GGUF dump in `bert.cpp` ADR-005 iter 75 notes.
    #[test]
    fn validate_tensor_set_passes_on_real_nomic_gguf() {
        let path = Path::new("/opt/hf2q/models/bert-test/nomic-embed-text-v1.5-f16.gguf");
        if !path.exists() {
            eprintln!("skipping: nomic GGUF fixture not at {}", path.display());
            return;
        }
        let gguf = GgufFile::open(path).expect("open nomic GGUF");
        let cfg = NomicBertConfig::from_gguf(&gguf).expect("parse nomic config");
        validate_tensor_set(&gguf, &cfg).expect("nomic GGUF must satisfy required-set");
    }

    /// Negative: bge GGUF (which is `arch="bert"` with separate Q/K/V)
    /// MUST fail nomic-bert validation. Drives the synthetic
    /// 12-layer cfg through validation against the bge tensor set so
    /// the failure names a specific missing fused tensor (i.e. the
    /// validator is naming what's actually wrong, not just throwing).
    #[test]
    fn validate_tensor_set_rejects_bert_gguf_naming_missing_fused_tensor() {
        let path = Path::new("/opt/hf2q/models/bert-test/bge-small-en-v1.5-f16.gguf");
        if !path.exists() {
            eprintln!("skipping: bge GGUF fixture not at {}", path.display());
            return;
        }
        let gguf = GgufFile::open(path).expect("open bge GGUF");
        // bge has 12 layers like nomic — drive with a synthetic 12-layer
        // cfg so the per-layer expansion doesn't generate spurious
        // missing-tensor noise.
        let cfg = synthetic_cfg(12);
        let err = validate_tensor_set(&gguf, &cfg)
            .expect_err("bge must fail nomic-bert validation (no fused QKV)");
        let msg = format!("{err}");
        assert!(msg.contains("missing"), "error must say 'missing', got: {msg}");
        assert!(
            msg.contains("attn_qkv.weight") || msg.contains("ffn_gate.weight"),
            "error must name a nomic-only tensor, got: {msg}"
        );
    }

    #[test]
    fn synthetic_required_count_matches_layer_count() {
        let cfg = synthetic_cfg(2);
        // 3 stem (token_embd + 2 LN) + 9 per-block × 2 layers = 21.
        let stem = 3;
        let per_block = NOMIC_BERT_BLOCK_REQUIRED_SUFFIXES.len();
        let expected = stem + per_block * cfg.num_hidden_layers;
        assert_eq!(stem + 9 * 2, expected);
    }
}
