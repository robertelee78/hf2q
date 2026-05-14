//! DFlash drafter weight loader (ADR-030 §3.3).
//!
//! Mirrors the safetensors-loading half of `/opt/dflash/dflash/model_mlx.py`
//! lines 207-240 (`load_draft`). The Python path uses `mx.load(path)` which
//! returns a `dict[str, mx.array]`; we instead build a strict manifest from
//! [`super::config::DFlashConfig`] and validate every tensor's name +
//! dtype + shape at load time. Missing/wrong-shape/wrong-dtype tensors
//! produce [`WeightsError`] — no fallback, no stub (per mantra).
//!
//! ## Tensor inventory (58 tensors for the gemma-4-26B-A4B-it drafter)
//!
//! Verified iter-16 via `mx.load` on the cached safetensors file:
//!
//! | Tensor                                          | Shape                  |
//! |-------------------------------------------------|------------------------|
//! | `fc.weight`                                     | `[H, T*H]`             |
//! | `hidden_norm.weight`                            | `[H]`                  |
//! | `norm.weight`                                   | `[H]`                  |
//! | `layers.{i}.input_layernorm.weight`             | `[H]`                  |
//! | `layers.{i}.post_attention_layernorm.weight`    | `[H]`                  |
//! | `layers.{i}.self_attn.q_proj.weight`            | `[Qh*Dh, H]`           |
//! | `layers.{i}.self_attn.k_proj.weight`            | `[Kh*Dh, H]`           |
//! | `layers.{i}.self_attn.v_proj.weight`            | `[Kh*Dh, H]`           |
//! | `layers.{i}.self_attn.o_proj.weight`            | `[H, Qh*Dh]`           |
//! | `layers.{i}.self_attn.q_norm.weight`            | `[Dh]`                 |
//! | `layers.{i}.self_attn.k_norm.weight`            | `[Dh]`                 |
//! | `layers.{i}.mlp.gate_proj.weight`               | `[I, H]`               |
//! | `layers.{i}.mlp.up_proj.weight`                 | `[I, H]`               |
//! | `layers.{i}.mlp.down_proj.weight`               | `[H, I]`               |
//!
//! Where `H=hidden_size`, `T=num_target_layers_used`, `Qh=num_q_heads`,
//! `Kh=num_kv_heads`, `Dh=head_dim`, `I=intermediate_size`.
//!
//! All BF16. No `embed_tokens` / `lm_head` — those are shared with target
//! via `DFlashDraftModel.bind()` at model construction (model_mlx.py:153-168).

use super::config::DFlashConfig;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    #[error("dflash weights IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("dflash weights safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("dflash weights: missing tensor `{0}`")]
    Missing(String),
    #[error("dflash weights: tensor `{name}` has dtype {actual:?}, expected {expected:?}")]
    Dtype {
        name: String,
        actual: Dtype,
        expected: Dtype,
    },
    #[error("dflash weights: tensor `{name}` has shape {actual:?}, expected {expected:?}")]
    Shape {
        name: String,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("dflash weights: unexpected extra tensor `{0}` not in DFlash manifest")]
    Extra(String),
}

/// Expected dtype for every drafter tensor.
///
/// All drafter weights are BF16 per the published `z-lab/gemma-4-26B-A4B-it-DFlash`
/// model card and verified via `mx.load` on the cached safetensors.
pub const DRAFTER_WEIGHT_DTYPE: Dtype = Dtype::BF16;

/// A single expected tensor in the drafter manifest.
#[derive(Debug, Clone)]
pub struct ExpectedTensor {
    pub name: String,
    pub shape: Vec<usize>,
}

/// Build the full expected-tensor manifest from a validated config.
///
/// The manifest must match the safetensors file exactly — no missing,
/// no extra tensors. Order is stable so tests can rely on indexing.
pub fn expected_manifest(cfg: &DFlashConfig) -> Vec<ExpectedTensor> {
    let h = cfg.hidden_size;
    let fc_in = cfg.fc_input_dim();
    let qh_dh = cfg.num_attention_heads * cfg.head_dim;
    let kh_dh = cfg.num_key_value_heads * cfg.head_dim;
    let dh = cfg.head_dim;
    let inter = cfg.intermediate_size;

    let mut m = Vec::with_capacity(3 + cfg.num_hidden_layers * 11);

    // Globals (ordered to match common safetensors traversal).
    m.push(ExpectedTensor { name: "fc.weight".into(), shape: vec![h, fc_in] });
    m.push(ExpectedTensor { name: "hidden_norm.weight".into(), shape: vec![h] });

    // Per-layer (5 × 11 = 55 tensors for the 5-layer gemma-4 drafter).
    for i in 0..cfg.num_hidden_layers {
        let p = format!("layers.{i}");
        m.push(ExpectedTensor { name: format!("{p}.input_layernorm.weight"), shape: vec![h] });
        m.push(ExpectedTensor { name: format!("{p}.mlp.down_proj.weight"), shape: vec![h, inter] });
        m.push(ExpectedTensor { name: format!("{p}.mlp.gate_proj.weight"), shape: vec![inter, h] });
        m.push(ExpectedTensor { name: format!("{p}.mlp.up_proj.weight"), shape: vec![inter, h] });
        m.push(ExpectedTensor { name: format!("{p}.post_attention_layernorm.weight"), shape: vec![h] });
        m.push(ExpectedTensor { name: format!("{p}.self_attn.k_norm.weight"), shape: vec![dh] });
        m.push(ExpectedTensor { name: format!("{p}.self_attn.k_proj.weight"), shape: vec![kh_dh, h] });
        m.push(ExpectedTensor { name: format!("{p}.self_attn.o_proj.weight"), shape: vec![h, qh_dh] });
        m.push(ExpectedTensor { name: format!("{p}.self_attn.q_norm.weight"), shape: vec![dh] });
        m.push(ExpectedTensor { name: format!("{p}.self_attn.q_proj.weight"), shape: vec![qh_dh, h] });
        m.push(ExpectedTensor { name: format!("{p}.self_attn.v_proj.weight"), shape: vec![kh_dh, h] });
    }

    m.push(ExpectedTensor { name: "norm.weight".into(), shape: vec![h] });

    m
}

/// Memmapped safetensors file + parsed metadata.
///
/// Owns the file mapping for the lifetime of the loaded tensors. The
/// `tensors` view borrows from `bytes`. Hold this struct alive for the
/// duration of any in-flight inference call.
pub struct DFlashWeightsFile {
    _mmap: memmap2::Mmap,
    bytes: &'static [u8],
}

impl DFlashWeightsFile {
    /// Memmap a safetensors file. The mapping is read-only and lives
    /// as long as `self`.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, WeightsError> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        // SAFETY: we keep `_mmap` alive on this struct, so the slice
        // is valid for the lifetime of `self`. Callers borrow tensors
        // through `view()` which constrains the lifetime correctly.
        let bytes: &'static [u8] =
            unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };
        Ok(Self { _mmap: mmap, bytes })
    }

    /// Borrow the file bytes (must remain alive as long as the returned slice).
    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }
}

/// View into a loaded drafter safetensors file, validated against config.
///
/// `tensors[i]` corresponds to `manifest[i]` — same order, 1:1.
pub struct DFlashWeights<'data> {
    pub manifest: Vec<ExpectedTensor>,
    pub tensors: Vec<TensorView<'data>>,
}

impl<'data> DFlashWeights<'data> {
    /// Parse and validate the safetensors file bytes against the
    /// expected-tensor manifest derived from `cfg`. Strict: every
    /// expected tensor MUST be present, in BF16, with the expected
    /// shape; no extra tensors allowed (mantra: no shortcuts).
    pub fn load<'cfg>(
        bytes: &'data [u8],
        cfg: &'cfg DFlashConfig,
    ) -> Result<Self, WeightsError> {
        let st = SafeTensors::deserialize(bytes)?;
        let manifest = expected_manifest(cfg);

        // Build a name set for the "no extras" check.
        let expected_names: std::collections::HashSet<&str> =
            manifest.iter().map(|t| t.name.as_str()).collect();
        for name in st.names() {
            // `st.names()` returns Vec<&String>; dereference to &str for HashSet lookup.
            let name_str: &str = name;
            if !expected_names.contains(name_str) {
                return Err(WeightsError::Extra(name.to_string()));
            }
        }

        // Validate + collect tensors in manifest order.
        let mut tensors = Vec::with_capacity(manifest.len());
        for exp in &manifest {
            let view = st.tensor(&exp.name).map_err(|e| match e {
                safetensors::SafeTensorError::TensorNotFound(_) => {
                    WeightsError::Missing(exp.name.clone())
                }
                other => WeightsError::Safetensors(other),
            })?;
            if view.dtype() != DRAFTER_WEIGHT_DTYPE {
                return Err(WeightsError::Dtype {
                    name: exp.name.clone(),
                    actual: view.dtype(),
                    expected: DRAFTER_WEIGHT_DTYPE,
                });
            }
            let actual: Vec<usize> = view.shape().to_vec();
            if actual != exp.shape {
                return Err(WeightsError::Shape {
                    name: exp.name.clone(),
                    actual,
                    expected: exp.shape.clone(),
                });
            }
            tensors.push(view);
        }

        Ok(DFlashWeights { manifest, tensors })
    }

    /// Look up a tensor by name (linear scan — manifest is ~58 entries).
    pub fn tensor(&self, name: &str) -> Option<&TensorView<'data>> {
        self.manifest
            .iter()
            .position(|t| t.name == name)
            .map(|i| &self.tensors[i])
    }

    /// Number of bytes occupied by all tensor data (excludes header).
    /// At BF16: total params × 2 bytes. Useful for memory accounting.
    pub fn total_data_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.data().len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::dflash::config::DFlashConfig;

    fn gemma4_26b_a4b_dflash_config() -> DFlashConfig {
        DFlashConfig::from_json_str(super::super::config::tests::GEMMA4_26B_A4B_DFLASH_CONFIG)
            .expect("test fixture must parse")
    }

    #[test]
    fn manifest_has_expected_tensor_count() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let m = expected_manifest(&cfg);
        // 2 leading globals (fc, hidden_norm) + 5 × 11 per-layer + 1 trailing (norm) = 58
        assert_eq!(m.len(), 58);
    }

    #[test]
    fn manifest_fc_shape_is_h_times_fc_in() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let m = expected_manifest(&cfg);
        let fc = m.iter().find(|t| t.name == "fc.weight").expect("fc.weight in manifest");
        assert_eq!(fc.shape, vec![2816, 6 * 2816]);
    }

    #[test]
    fn manifest_layer_qkv_shapes_match_qwen3_style() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let m = expected_manifest(&cfg);
        let q = m.iter().find(|t| t.name == "layers.0.self_attn.q_proj.weight").unwrap();
        let k = m.iter().find(|t| t.name == "layers.0.self_attn.k_proj.weight").unwrap();
        let v = m.iter().find(|t| t.name == "layers.0.self_attn.v_proj.weight").unwrap();
        let o = m.iter().find(|t| t.name == "layers.0.self_attn.o_proj.weight").unwrap();
        // Drafter: 32 q heads × 128 dim = 4096; 8 kv heads × 128 dim = 1024
        assert_eq!(q.shape, vec![4096, 2816]);
        assert_eq!(k.shape, vec![1024, 2816]);
        assert_eq!(v.shape, vec![1024, 2816]);
        assert_eq!(o.shape, vec![2816, 4096]);
    }

    #[test]
    fn manifest_norm_shapes() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let m = expected_manifest(&cfg);
        // q_norm/k_norm are per-head_dim (128)
        let qn = m.iter().find(|t| t.name == "layers.0.self_attn.q_norm.weight").unwrap();
        assert_eq!(qn.shape, vec![128]);
        // input_layernorm / post_attention_layernorm are per-hidden_size (2816)
        let il = m.iter().find(|t| t.name == "layers.0.input_layernorm.weight").unwrap();
        assert_eq!(il.shape, vec![2816]);
    }

    #[test]
    fn manifest_no_embed_tokens_no_lm_head() {
        // Per model_mlx.py:153-168 the drafter shares embed_tokens + lm_head
        // with the target via `bind()`. They MUST NOT appear in our manifest.
        let cfg = gemma4_26b_a4b_dflash_config();
        let m = expected_manifest(&cfg);
        assert!(m.iter().all(|t| t.name != "embed_tokens.weight"));
        assert!(m.iter().all(|t| t.name != "lm_head.weight"));
    }

    /// Integration test against the actual cached safetensors file.
    /// Skipped by default; run with `cargo test --bin hf2q --
    /// dflash::weights::tests::loads_real_drafter_file --ignored --nocapture`.
    #[test]
    #[ignore = "requires ~/.cache/huggingface drafter download"]
    fn loads_real_drafter_file() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let home = std::env::var("HOME").expect("HOME set");
        let path = format!("{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors");
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let w = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        assert_eq!(w.manifest.len(), 58);
        assert_eq!(w.tensors.len(), 58);
        // Sanity: total bytes ≈ 425M params × 2 bytes ≈ 850MB.
        let bytes = w.total_data_bytes();
        assert!((780_000_000..=900_000_000).contains(&bytes),
            "expected ~820MB data, got {bytes}");
    }
}
