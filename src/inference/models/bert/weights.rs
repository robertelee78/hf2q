//! BERT GGUF weight loader (ADR-005 Phase 2b, Task #13).
//!
//! Matrix tensors retain their exact GGUF bytes in file-backed Metal views.
//! Only declared vector state (normalization parameters and biases) expands
//! to F32.  Shape and kernel capability checks complete before mapping.
//!
//! # Sequencing
//!
//! 1. `BertConfig::from_gguf` — parse architecture metadata (hidden
//!    size, layer count, pooling, etc.). Cheap.
//! 2. `validate_tensor_set` (this module) — confirm every required
//!    tensor name is present BEFORE the expensive load. Saves operators
//!    from waiting through a multi-GB load only to fail on a missing
//!    `blk.5.attn_q.weight`.
//! 3. `LoadedBertWeights::load` — map every matrix in its native GGUF
//!    representation and expand only declared vector state to F32.
//! 4. `bert_gpu.rs` — native encoder forward pass + pooling.
//!
//! # Day-one supported models (per ADR-005 Phase 2b)
//!
//! - `nomic-embed-text-v1.5` (137M params, hidden=768, layers=12)
//! - `mxbai-embed-large-v1` (335M params, hidden=1024, layers=24)
//! - `bge-small-en-v1.5` (33M params, hidden=384, layers=12)
//!
//! bge and mxbai share the `bert.*` GGUF metadata convention and the
//! per-layer tensor names below. Nomic's RoPE/fused-QKV graph has its own
//! explicit loader under `nomic_bert`; it is never approximated through this
//! plain-BERT tensor catalog.
//!
//! # Optional tensors
//!
//! Some BERTs lack `token_types.weight` (single-segment models) or
//! `attn_*.bias` (bias-free variants). The validator below treats those
//! as **optional** — `validate_tensor_set` does not flag them missing —
//! and the accessor returns `None` (not `Err`) so the forward pass can
//! branch on presence. The required minimum is the LayerNorm + linear
//! weight set that every BERT variant ships.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use mlx_native::gguf::GgufFile;
use mlx_native::metal::foreign_types::ForeignType;
use mlx_native::{MlxBuffer, MlxDevice};

use crate::serve::forward_mlx_shared::MlxQWeight;

use super::native_storage::{
    mapped_qweight, preflight_embedding, preflight_linear, preflight_state, MatrixPlan,
    NativeStorageStats, StatePlan,
};

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

/// BERT matrices retained as native mapped GGUF views plus explicit F32
/// vector state.
///
/// Wrap one loaded generation in `Arc` for request sharing; reloading creates
/// a fresh mapping and explicit state allocation. Field access goes through
/// named shortcuts, with `get(name)` retained for diagnostics/tests.
pub struct LoadedBertWeights {
    matrices: HashMap<String, MlxQWeight>,
    states: HashMap<String, MlxBuffer>,
    stats: NativeStorageStats,
    /// Device handle kept alive for the lifetime of the buffers. Held
    /// for RAII even though public accessors go through `tensors`.
    _device: MlxDevice,
}

impl std::fmt::Debug for LoadedBertWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedBertWeights")
            .field("matrix_count", &self.matrices.len())
            .field("state_count", &self.states.len())
            .field("storage", &self.stats)
            .finish()
    }
}

impl LoadedBertWeights {
    pub fn load(gguf: &GgufFile, cfg: &BertConfig, device: MlxDevice) -> Result<Self> {
        validate_tensor_set(gguf, cfg)?;
        let (matrix_plans, state_plans) = preflight_plans(gguf, cfg)?;

        // This is the first payload operation. Every required name, shape,
        // storage type, and execution regime has already passed above.
        let mapped = gguf
            .map_tensor_data(&device)
            .map_err(|e| anyhow!("map BERT GGUF tensor payload: {e}"))?;
        let mut matrices = HashMap::with_capacity(matrix_plans.len());
        let mut mapped_resources = std::collections::HashSet::new();
        let mut file_backed_bytes = 0u64;
        for plan in &matrix_plans {
            let info = gguf
                .tensor_info(&plan.name)
                .ok_or_else(|| anyhow!("BERT native tensor '{}' disappeared", plan.name))?;
            let weight = mapped_qweight(plan, &mapped, info)
                .map_err(|e| anyhow!("map BERT native tensor '{}': {e}", plan.name))?;
            mapped_resources.insert(weight.buffer.metal_buffer().as_ptr() as usize);
            file_backed_bytes = file_backed_bytes
                .checked_add(plan.byte_len as u64)
                .ok_or_else(|| anyhow!("BERT file-backed byte accounting overflow"))?;
            matrices.insert(plan.name.clone(), weight);
        }

        let mut states = HashMap::with_capacity(state_plans.len());
        let mut anonymous_state_bytes = 0u64;
        for plan in &state_plans {
            let buffer = gguf
                .load_tensor_f32(&plan.name, &device)
                .map_err(|e| anyhow!("load BERT F32 state '{}': {e}", plan.name))?;
            anonymous_state_bytes = anonymous_state_bytes
                .checked_add((plan.elements as u64) * 4)
                .ok_or_else(|| anyhow!("BERT state byte accounting overflow"))?;
            states.insert(plan.name.clone(), buffer);
        }
        let resident_bytes = file_backed_bytes
            .checked_add(anonymous_state_bytes)
            .ok_or_else(|| anyhow!("BERT resident byte accounting overflow"))?;
        Ok(Self {
            matrices,
            states,
            stats: NativeStorageStats {
                resident_bytes,
                file_backed_bytes,
                anonymous_state_bytes,
                mapped_segment_count: mapped_resources.len(),
            },
            _device: device,
        })
    }

    /// Open + load convenience: opens the file at `path`, creates a
    /// default `MlxDevice`, validates, and loads. Used by the server
    /// startup path when the operator passes `--embedding-model X.gguf`.
    pub fn load_from_path(path: &Path, cfg: &BertConfig) -> Result<Self> {
        let gguf =
            GgufFile::open(path).map_err(|e| anyhow!("open BERT GGUF {}: {e}", path.display()))?;
        validate_tensor_set(&gguf, cfg)?;
        let device =
            MlxDevice::new().map_err(|e| anyhow!("create MlxDevice for BERT load: {e}"))?;
        Self::load(&gguf, cfg, device)
    }

    /// Empty placeholder — used by tests that need a `LoadedBertWeights`
    /// shape but do not drive a forward pass. Every shortcut accessor
    /// returns `Err` on this instance; `get()` returns `None`.
    pub fn empty(device: MlxDevice) -> Self {
        Self {
            matrices: HashMap::new(),
            states: HashMap::new(),
            stats: NativeStorageStats::default(),
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
        let mut matrices = HashMap::new();
        let mut states = HashMap::new();
        for (name, buffer) in tensors {
            if buffer.shape().len() == 2 {
                let rows = buffer.shape()[0];
                let cols = buffer.shape()[1];
                matrices.insert(
                    name,
                    super::native_storage::f32_qweight_for_test(buffer, rows, cols),
                );
            } else {
                states.insert(name, buffer);
            }
        }
        Self {
            matrices,
            states,
            stats: NativeStorageStats::default(),
            _device: device,
        }
    }

    /// Total tensor count.
    pub fn len(&self) -> usize {
        self.matrices.len() + self.states.len()
    }

    /// Empty when no tensors loaded (only possible from `Self::empty`).
    pub fn is_empty(&self) -> bool {
        self.matrices.is_empty() && self.states.is_empty()
    }

    /// Look up a tensor by exact GGUF name. `None` when absent.
    pub fn get(&self, name: &str) -> Option<&MlxBuffer> {
        self.states
            .get(name)
            .or_else(|| self.matrices.get(name).map(|weight| &weight.buffer))
    }

    pub fn storage_stats(&self) -> NativeStorageStats {
        self.stats
    }

    // -----------------------------------------------------------------------
    // Stem shortcuts. Errors carry the missing-tensor name in the message
    // so a forward-pass failure is debuggable from the log alone.
    // -----------------------------------------------------------------------

    pub fn token_embd_weight(&self) -> Result<&MlxQWeight> {
        self.matrices
            .get(TENSOR_TOKEN_EMBD)
            .ok_or_else(|| anyhow!("BERT missing '{}'", TENSOR_TOKEN_EMBD))
    }

    pub fn position_embd_weight(&self) -> Result<&MlxQWeight> {
        self.matrices
            .get(TENSOR_POS_EMBD)
            .ok_or_else(|| anyhow!("BERT missing '{}'", TENSOR_POS_EMBD))
    }

    /// Optional — returns `None` when the model has no segment table.
    pub fn token_types_weight(&self) -> Option<&MlxQWeight> {
        self.matrices.get(TENSOR_TOKEN_TYPES)
    }

    pub fn embed_norm_weight(&self) -> Result<&MlxBuffer> {
        self.states
            .get(TENSOR_EMBED_NORM_WEIGHT)
            .ok_or_else(|| anyhow!("BERT missing '{}'", TENSOR_EMBED_NORM_WEIGHT))
    }

    pub fn embed_norm_bias(&self) -> Result<&MlxBuffer> {
        self.states
            .get(TENSOR_EMBED_NORM_BIAS)
            .ok_or_else(|| anyhow!("BERT missing '{}'", TENSOR_EMBED_NORM_BIAS))
    }

    // -----------------------------------------------------------------------
    // Per-block accessors. The forward pass calls these in a layer loop.
    // -----------------------------------------------------------------------

    /// Required per-block tensor (errors if missing).
    pub fn block_required(&self, layer_idx: usize, suffix: &str) -> Result<&MlxBuffer> {
        let key = bert_layer_tensor(layer_idx, suffix);
        self.states
            .get(&key)
            .ok_or_else(|| anyhow!("BERT missing '{}'", key))
    }

    /// Optional per-block tensor (returns `None` when absent — e.g. a
    /// bias-free model, or `attn_q.bias` on a fused-attention variant).
    pub fn block_optional(&self, layer_idx: usize, suffix: &str) -> Option<&MlxBuffer> {
        let key = bert_layer_tensor(layer_idx, suffix);
        self.states.get(&key)
    }

    pub fn block_matrix(&self, layer_idx: usize, suffix: &str) -> Result<&MlxQWeight> {
        let key = bert_layer_tensor(layer_idx, suffix);
        self.matrices
            .get(&key)
            .ok_or_else(|| anyhow!("BERT missing native matrix '{}'", key))
    }
}

fn preflight_plans(gguf: &GgufFile, cfg: &BertConfig) -> Result<(Vec<MatrixPlan>, Vec<StatePlan>)> {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let mut matrices = vec![
        preflight_embedding(
            gguf,
            TENSOR_TOKEN_EMBD,
            cfg.vocab_size,
            h,
            cfg.max_position_embeddings,
        )?,
        preflight_embedding(
            gguf,
            TENSOR_POS_EMBD,
            cfg.max_position_embeddings,
            h,
            cfg.max_position_embeddings,
        )?,
    ];
    if gguf.tensor_info(TENSOR_TOKEN_TYPES).is_some() {
        matrices.push(preflight_embedding(
            gguf,
            TENSOR_TOKEN_TYPES,
            cfg.type_vocab_size,
            h,
            cfg.max_position_embeddings,
        )?);
    }
    let mut states = vec![
        preflight_state(gguf, TENSOR_EMBED_NORM_WEIGHT, h)?,
        preflight_state(gguf, TENSOR_EMBED_NORM_BIAS, h)?,
    ];
    for layer in 0..cfg.num_hidden_layers {
        for suffix in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
        ] {
            matrices.push(preflight_linear(
                gguf,
                bert_layer_tensor(layer, suffix),
                h,
                h,
                cfg.max_position_embeddings,
            )?);
        }
        matrices.push(preflight_linear(
            gguf,
            bert_layer_tensor(layer, "ffn_up.weight"),
            i,
            h,
            cfg.max_position_embeddings,
        )?);
        matrices.push(preflight_linear(
            gguf,
            bert_layer_tensor(layer, "ffn_down.weight"),
            h,
            i,
            cfg.max_position_embeddings,
        )?);
        for suffix in [
            "attn_output_norm.weight",
            "attn_output_norm.bias",
            "layer_output_norm.weight",
            "layer_output_norm.bias",
        ] {
            states.push(preflight_state(gguf, bert_layer_tensor(layer, suffix), h)?);
        }
        for (suffix, elements) in [
            ("attn_q.bias", h),
            ("attn_k.bias", h),
            ("attn_v.bias", h),
            ("attn_output.bias", h),
            ("ffn_up.bias", i),
            ("ffn_down.bias", h),
        ] {
            let name = bert_layer_tensor(layer, suffix);
            if gguf.tensor_info(&name).is_some() {
                states.push(preflight_state(gguf, name, elements)?);
            }
        }
    }
    Ok((matrices, states))
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
        let _gpu = crate::inference::hf2q_gpu_test_lock();
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
        let _gpu = crate::inference::hf2q_gpu_test_lock();
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
        let _gpu = crate::inference::hf2q_gpu_test_lock();
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
        let _gpu = crate::inference::hf2q_gpu_test_lock();
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
    /// 57+), but the peer's vocab GGUF fixtures exercise the
    /// architecture-validation branch. Vocab GGUFs deliberately lack
    /// the weight tensors, so `validate_tensor_set` must report a
    /// specific missing-list rather than panic.
    #[test]
    fn validate_tensor_set_on_vocab_only_gguf_reports_missing_tensors() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
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
