//! ADR-037 Phase E3b — EAGLE-3 drafter weights schema + safetensors loader.
//!
//! Strict manifest-based loader: every tensor's name + dtype + shape is
//! validated against [`Eagle3DrafterConfig`] at load time. No fallback,
//! no stub. Mirrors the pattern shipped at
//! `src/inference/spec_decode/dflash/weights.rs` (DFlash drafter).
//!
//! ## Tensor inventory
//!
//! The manifest is config-driven — these gates control conditional
//! tensors:
//!
//! | Config flag                | Tensors gated                          |
//! |----------------------------|----------------------------------------|
//! | always present             | `embed_tokens.weight`, `fc.weight`,    |
//! |                            | `norm.weight`, `layers.0.input_lay…`,  |
//! |                            | `layers.0.hidden_norm.weight`,         |
//! |                            | `layers.0.post_attention_lay…`, all    |
//! |                            | self_attn projections, mlp projections |
//! | `norm_before_fc = true`    | `input_norm.weight` `[fc_input_size]` |
//! | `fc_norm = true`           | `fc_norm.{i}.weight` × `num_aux`,      |
//! |                            | shape `[target_hidden_size]`           |
//! | `use_qk_norm = true`       | `layers.0.self_attn.{q,k}_norm.weight` |
//! |                            | shape `[head_dim]`                     |
//! | `attention_bias = true`    | `{q,k,v,o}_proj.bias`                  |
//! | `tie_lm_head = false`      | `lm_head.weight`                       |
//! | `include_draft_id_mapping  | `draft_id_to_target_id` I64            |
//! |  = true`                   |                                        |
//!
//! ## First-layer Q/K/V input width contract
//!
//! Per vLLM `llama_eagle3.py:53`, the EAGLE-3 first decoder layer has
//! `qkv_input_size = 2 * hidden_size` (concat of `input_layernorm(embeds)`
//! + `hidden_norm(hidden_states)`). The `q_proj` / `k_proj` / `v_proj`
//! tensors therefore have **second-dim = 2 * hidden_size** — NOT
//! `hidden_size` like in a normal transformer layer. Tests verify this
//! invariant (`adr_037_e3b_layer0_qkv_input_width_is_2x_hidden`).
//!
//! ## Why we expect SEPARATE q/k/v/gate/up tensors (not stacked)
//!
//! vLLM stacks QKV + gate/up IN MEMORY for efficiency (peer line 254-260),
//! but the SAFETENSORS files published from EAGLE-3 training carry
//! separate tensors. Our loader expects the published format.

use super::config::Eagle3DrafterConfig;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum Eagle3WeightsError {
    #[error("eagle3 weights IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("eagle3 weights safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("eagle3 weights: missing tensor `{0}`")]
    Missing(String),
    #[error(
        "eagle3 weights: tensor `{name}` has dtype {actual:?}, expected {expected:?}"
    )]
    Dtype {
        name: String,
        actual: Dtype,
        expected: Dtype,
    },
    #[error(
        "eagle3 weights: tensor `{name}` has shape {actual:?}, expected {expected:?}"
    )]
    Shape {
        name: String,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("eagle3 weights: unexpected extra tensor `{0}` not in manifest")]
    Extra(String),
    #[error("eagle3 weights: invalid config: {0}")]
    Config(String),
}

/// Default dtype for floating tensors (BF16 per EAGLE-3 paper +
/// vLLM default training).
pub const EAGLE3_FLOAT_DTYPE: Dtype = Dtype::BF16;

/// Dtype for the integer `draft_id_to_target_id` mapping (I64 per
/// vLLM line 333 `torch.zeros(... dtype=torch.long)`).
pub const EAGLE3_DRAFT_ID_DTYPE: Dtype = Dtype::I64;

/// A single expected tensor in the manifest.
#[derive(Debug, Clone)]
pub struct ExpectedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl ExpectedTensor {
    fn float(name: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype: EAGLE3_FLOAT_DTYPE,
        }
    }
    fn int_i64(name: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype: EAGLE3_DRAFT_ID_DTYPE,
        }
    }
}

/// Build the full expected-tensor manifest from a validated config.
///
/// The manifest must match the safetensors file exactly — no missing,
/// no extra tensors (strict mode per mantra "no fallback").
///
/// Order is stable so tests + downstream consumers can rely on
/// `manifest[i]` indexing. The ordering follows the natural
/// "globals → layer-internals (alphabetical) → trailing-globals"
/// pattern from DFlash for consistency.
pub fn expected_manifest(cfg: &Eagle3DrafterConfig) -> Vec<ExpectedTensor> {
    let h = cfg.hidden_size;
    let fc_in = cfg.fc_input_size();
    let qkv_in = cfg.qkv_input_width();
    let qh_dh = cfg.q_proj_out();
    let kh_dh = cfg.kv_proj_out();
    let dh = cfg.head_dim;
    let inter = cfg.intermediate_size;
    let th = cfg.target_hidden_size;
    let num_aux = cfg.num_aux_hidden_states;

    // Capacity: ~3 leading globals + (optional input_norm) +
    // (optional fc_norm × num_aux) + ~10 layer tensors + (optional
    // 2 qk_norms) + (optional 4 biases) + 2 trailing globals +
    // (optional lm_head) + (optional draft_id_to_target_id).
    let est = 3 + 1 + num_aux + 10 + 2 + 4 + 2 + 1 + 1;
    let mut m = Vec::with_capacity(est);

    // === Leading globals ===
    // embed_tokens is OPTIONAL — vLLM peer (llama_eagle3.py:449-450)
    // treats missing EAGLE embed weights as valid and shares the
    // target's embedding table. Codex /cfa E3 Major (2026-05-22).
    if cfg.has_own_embed_tokens {
        m.push(ExpectedTensor::float(
            "embed_tokens.weight",
            vec![cfg.vocab_size, h],
        ));
    }
    m.push(ExpectedTensor::float("fc.weight", vec![h, fc_in]));

    // Optional: input_norm (single RMSNorm over fc_input_size).
    if cfg.norm_before_fc {
        m.push(ExpectedTensor::float("input_norm.weight", vec![fc_in]));
    }

    // Optional: per-aux RMSNorms each [target_hidden_size]. Indexed
    // by capture position, NOT target layer ID.
    if cfg.fc_norm {
        for i in 0..num_aux {
            m.push(ExpectedTensor::float(
                format!("fc_norm.{i}.weight"),
                vec![th],
            ));
        }
    }

    // === The single layer (layers.0) ===
    // Layer norms (all hidden_size unless specified).
    m.push(ExpectedTensor::float(
        "layers.0.input_layernorm.weight",
        vec![h],
    ));
    m.push(ExpectedTensor::float(
        "layers.0.hidden_norm.weight",
        vec![h],
    ));
    m.push(ExpectedTensor::float(
        "layers.0.post_attention_layernorm.weight",
        vec![h],
    ));

    // Self-attention projections. NOTE: q/k/v input width is
    // 2*hidden_size for the EAGLE-3 first layer (concat input).
    m.push(ExpectedTensor::float(
        "layers.0.self_attn.q_proj.weight",
        vec![qh_dh, qkv_in],
    ));
    m.push(ExpectedTensor::float(
        "layers.0.self_attn.k_proj.weight",
        vec![kh_dh, qkv_in],
    ));
    m.push(ExpectedTensor::float(
        "layers.0.self_attn.v_proj.weight",
        vec![kh_dh, qkv_in],
    ));
    m.push(ExpectedTensor::float(
        "layers.0.self_attn.o_proj.weight",
        vec![h, qh_dh],
    ));

    // Optional Q/K head-norm (Qwen-style). Per-head_dim.
    if cfg.use_qk_norm {
        m.push(ExpectedTensor::float(
            "layers.0.self_attn.q_norm.weight",
            vec![dh],
        ));
        m.push(ExpectedTensor::float(
            "layers.0.self_attn.k_norm.weight",
            vec![dh],
        ));
    }

    // Optional attention biases.
    if cfg.attention_bias {
        m.push(ExpectedTensor::float(
            "layers.0.self_attn.q_proj.bias",
            vec![qh_dh],
        ));
        m.push(ExpectedTensor::float(
            "layers.0.self_attn.k_proj.bias",
            vec![kh_dh],
        ));
        m.push(ExpectedTensor::float(
            "layers.0.self_attn.v_proj.bias",
            vec![kh_dh],
        ));
        m.push(ExpectedTensor::float(
            "layers.0.self_attn.o_proj.bias",
            vec![h],
        ));
    }

    // MLP — SwiGLU (gate / up / down).
    m.push(ExpectedTensor::float(
        "layers.0.mlp.gate_proj.weight",
        vec![inter, h],
    ));
    m.push(ExpectedTensor::float(
        "layers.0.mlp.up_proj.weight",
        vec![inter, h],
    ));
    m.push(ExpectedTensor::float(
        "layers.0.mlp.down_proj.weight",
        vec![h, inter],
    ));

    // === Trailing globals ===
    m.push(ExpectedTensor::float("norm.weight", vec![h]));

    if !cfg.tie_lm_head {
        m.push(ExpectedTensor::float(
            "lm_head.weight",
            vec![cfg.draft_vocab_size, h],
        ));
    }

    if cfg.include_draft_id_mapping {
        m.push(ExpectedTensor::int_i64(
            "draft_id_to_target_id",
            vec![cfg.draft_vocab_size],
        ));
    }

    m
}

/// Memmapped safetensors file. Mirrors DFlash's `DFlashWeightsFile`.
pub struct Eagle3WeightsFile {
    _mmap: memmap2::Mmap,
    bytes: &'static [u8],
}

impl Eagle3WeightsFile {
    /// Memmap a safetensors file. The mapping is read-only and lives
    /// as long as `self`.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Eagle3WeightsError> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        // SAFETY: we keep `_mmap` alive on this struct, so the slice
        // is valid for the lifetime of `self`. Borrowers via
        // `Eagle3Weights::load` get a constrained lifetime tied to
        // the &self borrow.
        let bytes: &'static [u8] =
            unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };
        Ok(Self { _mmap: mmap, bytes })
    }

    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }
}

/// View into a loaded EAGLE-3 safetensors file, validated against config.
///
/// `tensors[i]` corresponds to `manifest[i]` — same order, 1:1.
#[derive(Debug)]
pub struct Eagle3Weights<'data> {
    pub manifest: Vec<ExpectedTensor>,
    pub tensors: Vec<TensorView<'data>>,
}

impl<'data> Eagle3Weights<'data> {
    /// Parse + validate the safetensors file bytes against the
    /// expected-tensor manifest derived from `cfg`. Strict: every
    /// expected tensor MUST be present with the expected dtype + shape;
    /// no extra tensors allowed.
    ///
    /// Codex /cfa E3 Major (2026-05-22): vLLM EAGLE-3 checkpoints
    /// sometimes use `d2t` as the safetensors key for the draft→target
    /// vocab mapping and `t2d` for the inverse. Per vLLM
    /// `llama_eagle3.py:415-419`, `d2t` is canonically renamed to
    /// `draft_id_to_target_id` and `t2d` is skipped. We apply the
    /// same normalization here so vLLM-format checkpoints load
    /// without manual remapping.
    pub fn load(
        bytes: &'data [u8],
        cfg: &Eagle3DrafterConfig,
    ) -> Result<Self, Eagle3WeightsError> {
        // Codex /cfa E3 Major (2026-05-22): defensive cfg.validate()
        // at the loader entry. Without this, an invalid config silently
        // builds a wrong manifest and surfaces as confusing
        // Missing/Shape errors at load time.
        cfg.validate()
            .map_err(|e| Eagle3WeightsError::Config(e.to_string()))?;

        let st = SafeTensors::deserialize(bytes)?;
        let manifest = expected_manifest(cfg);

        // Apply vLLM EAGLE-3 name normalization: d2t → draft_id_to_target_id;
        // t2d skipped. verifier_lm_head.weight / verifier_norm.weight are in
        // `_keys_to_ignore_on_save` per Speculators eagle3/core.py — skip them
        // silently (ADR-038 §3.4.3, AC-G4-4.3).
        let resolve_name = |incoming: &str| -> Option<String> {
            if incoming == "t2d"
                || incoming == "verifier_lm_head.weight"
                || incoming == "verifier_norm.weight"
            {
                None // skipped — not an error
            } else if incoming == "d2t" {
                Some("draft_id_to_target_id".to_string())
            } else {
                Some(incoming.to_string())
            }
        };

        // Build name set for the "no extras" check.
        let expected_names: std::collections::HashSet<&str> =
            manifest.iter().map(|t| t.name.as_str()).collect();
        for name in st.names() {
            let name_str: &str = name;
            match resolve_name(name_str) {
                None => continue, // skipped tensor — not an error
                Some(canonical) => {
                    if !expected_names.contains(canonical.as_str()) {
                        return Err(Eagle3WeightsError::Extra(name.to_string()));
                    }
                }
            }
        }

        // For lookup, build a map from canonical → raw safetensors name.
        let mut canonical_to_raw: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for raw_name in st.names() {
            let raw: &str = raw_name;
            if let Some(canonical) = resolve_name(raw) {
                canonical_to_raw.insert(canonical, raw.to_string());
            }
        }

        let mut tensors = Vec::with_capacity(manifest.len());
        for exp in &manifest {
            let raw_name = canonical_to_raw
                .get(exp.name.as_str())
                .ok_or_else(|| Eagle3WeightsError::Missing(exp.name.clone()))?;
            let view = st.tensor(raw_name).map_err(|e| match e {
                safetensors::SafeTensorError::TensorNotFound(_) => {
                    Eagle3WeightsError::Missing(exp.name.clone())
                }
                other => Eagle3WeightsError::Safetensors(other),
            })?;
            if view.dtype() != exp.dtype {
                return Err(Eagle3WeightsError::Dtype {
                    name: exp.name.clone(),
                    actual: view.dtype(),
                    expected: exp.dtype,
                });
            }
            let actual: Vec<usize> = view.shape().to_vec();
            if actual != exp.shape {
                return Err(Eagle3WeightsError::Shape {
                    name: exp.name.clone(),
                    actual,
                    expected: exp.shape.clone(),
                });
            }
            tensors.push(view);
        }

        Ok(Self { manifest, tensors })
    }

    /// Look up a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorView<'data>> {
        self.manifest
            .iter()
            .position(|t| t.name == name)
            .map(|i| &self.tensors[i])
    }

    /// Total bytes occupied by all tensor data (excludes header).
    pub fn total_data_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.data().len()).sum()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::eagle3::config::tests::qwen35_default;

    // -------------------------------------------------------------
    // Manifest structure tests (no actual safetensors load required)
    // -------------------------------------------------------------

    #[test]
    fn adr_037_e3b_default_qwen35_manifest_count_2026_05_22() {
        // Default config: 3 aux, fc_norm=true, use_qk_norm=true,
        // norm_before_fc=false, attention_bias=false, tie_lm_head=false,
        // include_draft_id_mapping=true.
        //
        // Expected:
        //  - embed_tokens.weight
        //  - fc.weight
        //  - fc_norm.0/1/2.weight (3)
        //  - layers.0.input_layernorm
        //  - layers.0.hidden_norm
        //  - layers.0.post_attention_layernorm
        //  - layers.0.self_attn.q/k/v/o_proj.weight (4)
        //  - layers.0.self_attn.q_norm + k_norm (2)
        //  - layers.0.mlp.gate/up/down_proj.weight (3)
        //  - norm.weight
        //  - lm_head.weight
        //  - draft_id_to_target_id
        // Total: 2 + 3 + 3 + 4 + 2 + 3 + 1 + 1 + 1 = 20
        let m = expected_manifest(&qwen35_default());
        assert_eq!(m.len(), 20, "got: {:?}", m.iter().map(|t| &t.name).collect::<Vec<_>>());
    }

    #[test]
    fn adr_037_e3b_layer0_qkv_input_width_is_2x_hidden_2026_05_22() {
        // Critical invariant per vLLM line 53: EAGLE-3 first-layer
        // qkv_input_size = 2 * hidden_size. Validates the
        // concat-input-width handling.
        let cfg = qwen35_default();
        let m = expected_manifest(&cfg);

        let q = m
            .iter()
            .find(|t| t.name == "layers.0.self_attn.q_proj.weight")
            .expect("q_proj in manifest");
        assert_eq!(q.shape, vec![cfg.q_proj_out(), cfg.qkv_input_width()]);
        assert_eq!(q.shape[1], 2 * cfg.hidden_size);

        let k = m
            .iter()
            .find(|t| t.name == "layers.0.self_attn.k_proj.weight")
            .unwrap();
        assert_eq!(k.shape, vec![cfg.kv_proj_out(), cfg.qkv_input_width()]);
        let v = m
            .iter()
            .find(|t| t.name == "layers.0.self_attn.v_proj.weight")
            .unwrap();
        assert_eq!(v.shape, vec![cfg.kv_proj_out(), cfg.qkv_input_width()]);

        // o_proj is unaffected — outputs hidden_size from
        // q_proj_out width (standard).
        let o = m
            .iter()
            .find(|t| t.name == "layers.0.self_attn.o_proj.weight")
            .unwrap();
        assert_eq!(o.shape, vec![cfg.hidden_size, cfg.q_proj_out()]);
    }

    #[test]
    fn adr_037_e3b_fc_weight_shape_matches_fc_input_size_2026_05_22() {
        let cfg = qwen35_default();
        let m = expected_manifest(&cfg);
        let fc = m.iter().find(|t| t.name == "fc.weight").unwrap();
        assert_eq!(fc.shape, vec![cfg.hidden_size, cfg.fc_input_size()]);
    }

    #[test]
    fn adr_037_e3b_norm_before_fc_gates_input_norm_2026_05_22() {
        // norm_before_fc=false → no input_norm
        let mut cfg = qwen35_default();
        cfg.norm_before_fc = false;
        let m = expected_manifest(&cfg);
        assert!(
            m.iter().all(|t| t.name != "input_norm.weight"),
            "input_norm should be absent when norm_before_fc=false"
        );

        // norm_before_fc=true → input_norm present, shape [fc_input_size]
        cfg.norm_before_fc = true;
        let m = expected_manifest(&cfg);
        let n = m
            .iter()
            .find(|t| t.name == "input_norm.weight")
            .expect("input_norm present when norm_before_fc=true");
        assert_eq!(n.shape, vec![cfg.fc_input_size()]);
    }

    #[test]
    fn adr_037_e3b_fc_norm_gates_per_aux_norms_2026_05_22() {
        let mut cfg = qwen35_default();
        cfg.fc_norm = false;
        let m = expected_manifest(&cfg);
        assert!(
            m.iter().all(|t| !t.name.starts_with("fc_norm.")),
            "fc_norm.* should be absent when fc_norm=false"
        );

        cfg.fc_norm = true;
        let m = expected_manifest(&cfg);
        // Exactly num_aux fc_norm tensors, indexed 0..num_aux, each
        // [target_hidden_size].
        for i in 0..cfg.num_aux_hidden_states {
            let n = m
                .iter()
                .find(|t| t.name == format!("fc_norm.{i}.weight"))
                .unwrap_or_else(|| panic!("fc_norm.{i}.weight missing"));
            assert_eq!(n.shape, vec![cfg.target_hidden_size]);
        }
        // No extras: fc_norm.{num_aux}.weight should NOT be present.
        let extra = format!("fc_norm.{}.weight", cfg.num_aux_hidden_states);
        assert!(
            m.iter().all(|t| t.name != extra),
            "fc_norm.{} should NOT be present",
            cfg.num_aux_hidden_states
        );
    }

    #[test]
    fn adr_037_e3b_use_qk_norm_gates_qk_norm_tensors_2026_05_22() {
        let mut cfg = qwen35_default();
        cfg.use_qk_norm = false;
        let m = expected_manifest(&cfg);
        assert!(
            m.iter().all(|t| !t.name.contains("q_norm") && !t.name.contains("k_norm")),
            "qk_norm tensors absent when use_qk_norm=false"
        );

        cfg.use_qk_norm = true;
        let m = expected_manifest(&cfg);
        let qn = m
            .iter()
            .find(|t| t.name == "layers.0.self_attn.q_norm.weight")
            .unwrap();
        assert_eq!(qn.shape, vec![cfg.head_dim]);
        let kn = m
            .iter()
            .find(|t| t.name == "layers.0.self_attn.k_norm.weight")
            .unwrap();
        assert_eq!(kn.shape, vec![cfg.head_dim]);
    }

    #[test]
    fn adr_037_e3b_attention_bias_gates_4_biases_2026_05_22() {
        let mut cfg = qwen35_default();
        cfg.attention_bias = false;
        let m = expected_manifest(&cfg);
        assert!(
            m.iter().all(|t| !t.name.ends_with(".bias")),
            "no biases when attention_bias=false"
        );

        cfg.attention_bias = true;
        let m = expected_manifest(&cfg);
        let q_bias = m
            .iter()
            .find(|t| t.name == "layers.0.self_attn.q_proj.bias")
            .unwrap();
        assert_eq!(q_bias.shape, vec![cfg.q_proj_out()]);
        let k_bias = m
            .iter()
            .find(|t| t.name == "layers.0.self_attn.k_proj.bias")
            .unwrap();
        assert_eq!(k_bias.shape, vec![cfg.kv_proj_out()]);
        let v_bias = m
            .iter()
            .find(|t| t.name == "layers.0.self_attn.v_proj.bias")
            .unwrap();
        assert_eq!(v_bias.shape, vec![cfg.kv_proj_out()]);
        let o_bias = m
            .iter()
            .find(|t| t.name == "layers.0.self_attn.o_proj.bias")
            .unwrap();
        assert_eq!(o_bias.shape, vec![cfg.hidden_size]);
    }

    #[test]
    fn adr_037_e3b_tie_lm_head_gates_lm_head_tensor_2026_05_22() {
        let mut cfg = qwen35_default();
        cfg.tie_lm_head = true;
        let m = expected_manifest(&cfg);
        assert!(
            m.iter().all(|t| t.name != "lm_head.weight"),
            "lm_head.weight should be absent when tied"
        );

        cfg.tie_lm_head = false;
        let m = expected_manifest(&cfg);
        let lh = m.iter().find(|t| t.name == "lm_head.weight").unwrap();
        assert_eq!(lh.shape, vec![cfg.draft_vocab_size, cfg.hidden_size]);
    }

    #[test]
    fn adr_037_e3b_draft_id_mapping_is_i64_2026_05_22() {
        let cfg = qwen35_default();
        let m = expected_manifest(&cfg);
        let map = m
            .iter()
            .find(|t| t.name == "draft_id_to_target_id")
            .unwrap();
        assert_eq!(map.shape, vec![cfg.draft_vocab_size]);
        assert_eq!(map.dtype, EAGLE3_DRAFT_ID_DTYPE);
        assert_eq!(map.dtype, Dtype::I64);
    }

    #[test]
    fn adr_037_e3b_smaller_draft_vocab_shrinks_lm_head_and_mapping_2026_05_22() {
        // "Fast vocab projection" optimization: draft_vocab_size smaller
        // than target vocab_size shrinks both lm_head + the integer
        // mapping. This is the variant the EAGLE-3 paper recommends
        // for sequences where most tokens fall in a hot subset.
        let mut cfg = qwen35_default();
        cfg.draft_vocab_size = 32000; // ≪ 152064
        let m = expected_manifest(&cfg);

        let lh = m.iter().find(|t| t.name == "lm_head.weight").unwrap();
        assert_eq!(lh.shape, vec![32000, cfg.hidden_size]);

        let map = m
            .iter()
            .find(|t| t.name == "draft_id_to_target_id")
            .unwrap();
        assert_eq!(map.shape, vec![32000]);

        // embed_tokens stays at full vocab (drafter still consumes
        // full-vocab input tokens).
        let emb = m.iter().find(|t| t.name == "embed_tokens.weight").unwrap();
        assert_eq!(emb.shape, vec![152064, cfg.hidden_size]);
    }

    #[test]
    fn adr_037_e3b_all_gates_off_minimum_manifest_2026_05_22() {
        // Slimmest possible config: no input_norm, no fc_norm,
        // no qk_norm, no attention_bias, tied lm_head, no draft_id
        // mapping. Validates that every gate works independently.
        let mut cfg = qwen35_default();
        cfg.norm_before_fc = false;
        cfg.fc_norm = false;
        cfg.use_qk_norm = false;
        cfg.attention_bias = false;
        cfg.tie_lm_head = true;
        cfg.include_draft_id_mapping = false;
        let m = expected_manifest(&cfg);
        // Just: embed_tokens, fc, 3 layer norms (input_layernorm,
        // hidden_norm, post_attention_layernorm), 4 projections
        // (qkvo), 3 mlp, norm.weight = 2 + 3 + 4 + 3 + 1 = 13.
        assert_eq!(m.len(), 13);

        // Verify NO conditional tensors snuck in.
        for t in &m {
            assert!(!t.name.contains("input_norm"));
            assert!(!t.name.starts_with("fc_norm."));
            assert!(!t.name.contains("q_norm") && !t.name.contains("k_norm"));
            assert!(!t.name.ends_with(".bias"));
            assert!(t.name != "lm_head.weight");
            assert!(t.name != "draft_id_to_target_id");
        }
    }

    #[test]
    fn adr_037_e3b_all_gates_on_maximum_manifest_2026_05_22() {
        // Every gate enabled.
        let mut cfg = qwen35_default();
        cfg.norm_before_fc = true;
        cfg.fc_norm = true;
        cfg.use_qk_norm = true;
        cfg.attention_bias = true;
        cfg.tie_lm_head = false;
        cfg.include_draft_id_mapping = true;
        let m = expected_manifest(&cfg);
        // Base 13 + 1 (input_norm) + 3 (fc_norm × num_aux=3) + 2 (qk_norm)
        // + 4 (biases) + 1 (lm_head) + 1 (draft_id_to_target_id) = 25.
        assert_eq!(m.len(), 25);
    }

    #[test]
    fn adr_037_e3b_manifest_names_are_unique_2026_05_22() {
        // Catch typos / accidental duplication in expected_manifest.
        let cfg = qwen35_default();
        let m = expected_manifest(&cfg);
        let names: std::collections::HashSet<&str> =
            m.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate name in manifest");
    }

    #[test]
    fn adr_037_e3b_float_dtype_is_bf16_2026_05_22() {
        let cfg = qwen35_default();
        let m = expected_manifest(&cfg);
        for t in &m {
            if t.name != "draft_id_to_target_id" {
                assert_eq!(
                    t.dtype,
                    EAGLE3_FLOAT_DTYPE,
                    "tensor {} should be BF16",
                    t.name
                );
                assert_eq!(t.dtype, Dtype::BF16);
            }
        }
    }

    // -------------------------------------------------------------
    // Safetensors load tests using synthetic in-memory blobs.
    // -------------------------------------------------------------
    //
    // We don't have a real EAGLE-3 checkpoint for Qwen 3.6 27B yet (it
    // comes from E2 training). To validate the loader logic now, we
    // build a synthetic safetensors blob matching the expected
    // manifest. This exercises the strict-validation paths without
    // committing test fixture binaries to the repo.

    use std::collections::BTreeMap;

    /// Build a synthetic safetensors blob containing tensors at the
    /// shapes the manifest expects. Float tensors get zeros (BF16);
    /// integer mapping gets zeros (I64).
    fn build_synthetic_safetensors(
        manifest: &[ExpectedTensor],
    ) -> Vec<u8> {
        let mut storage: Vec<Vec<u8>> = Vec::with_capacity(manifest.len());
        let mut tensors: BTreeMap<String, TensorView> = BTreeMap::new();
        // Need to keep storage alive for view lifetime.
        for exp in manifest {
            let elem_bytes = match exp.dtype {
                Dtype::BF16 => 2,
                Dtype::I64 => 8,
                _ => panic!("unexpected dtype in test"),
            };
            let nelem: usize = exp.shape.iter().product();
            storage.push(vec![0u8; nelem * elem_bytes]);
        }
        // Second pass to build views (now that storage is stable).
        for (i, exp) in manifest.iter().enumerate() {
            let view = TensorView::new(
                exp.dtype,
                exp.shape.clone(),
                storage[i].as_slice(),
            )
            .expect("synthetic tensor view");
            tensors.insert(exp.name.clone(), view);
        }
        safetensors::serialize(&tensors, None::<std::collections::HashMap<String, String>>)
            .expect("serialize synthetic")
    }

    #[test]
    fn adr_037_e3b_load_synthetic_safetensors_succeeds_2026_05_22() {
        let cfg = qwen35_default();
        let manifest = expected_manifest(&cfg);
        let blob = build_synthetic_safetensors(&manifest);
        let w = Eagle3Weights::load(&blob, &cfg)
            .expect("synthetic blob matches manifest exactly");
        assert_eq!(w.tensors.len(), manifest.len());
        // Lookup by name returns same tensor.
        let fc = w.tensor("fc.weight").expect("fc.weight reachable");
        assert_eq!(fc.shape(), &[cfg.hidden_size, cfg.fc_input_size()]);
    }

    #[test]
    fn adr_037_e3b_load_rejects_missing_tensor_2026_05_22() {
        let cfg = qwen35_default();
        let mut manifest = expected_manifest(&cfg);
        manifest.retain(|t| t.name != "fc.weight"); // drop one
        let blob = build_synthetic_safetensors(&manifest);
        let err = Eagle3Weights::load(&blob, &cfg).unwrap_err();
        assert!(
            matches!(err, Eagle3WeightsError::Missing(ref n) if n == "fc.weight"),
            "got: {err:?}"
        );
    }

    #[test]
    fn adr_037_e3b_load_rejects_extra_tensor_2026_05_22() {
        let cfg = qwen35_default();
        let mut manifest = expected_manifest(&cfg);
        manifest.push(ExpectedTensor::float("extra_tensor.weight", vec![16]));
        let blob = build_synthetic_safetensors(&manifest);
        let err = Eagle3Weights::load(&blob, &cfg).unwrap_err();
        assert!(
            matches!(err, Eagle3WeightsError::Extra(ref n) if n == "extra_tensor.weight"),
            "got: {err:?}"
        );
    }

    #[test]
    fn adr_037_e3b_load_rejects_wrong_shape_2026_05_22() {
        let cfg = qwen35_default();
        // Build correct manifest but corrupt one tensor's shape.
        let mut manifest = expected_manifest(&cfg);
        // Find fc.weight and corrupt its shape.
        let fc_idx = manifest
            .iter()
            .position(|t| t.name == "fc.weight")
            .unwrap();
        manifest[fc_idx].shape = vec![cfg.hidden_size, cfg.fc_input_size() + 1];
        let blob = build_synthetic_safetensors(&manifest);
        // Load with ORIGINAL config — expects fc_input_size().
        let err = Eagle3Weights::load(&blob, &cfg).unwrap_err();
        assert!(
            matches!(err, Eagle3WeightsError::Shape { ref name, .. } if name == "fc.weight"),
            "got: {err:?}"
        );
    }

    // -------------------------------------------------------------
    // Codex /cfa E3 gate (2026-05-22) — negative-path validation tests.
    // Each proves a specific codex finding fix actually fires.
    // -------------------------------------------------------------

    #[test]
    fn adr_037_e3_gate_has_own_embed_tokens_gates_embed_tensor_2026_05_22() {
        // Codex Major 2: vLLM peer treats missing EAGLE embed weights
        // as valid (drafter shares target's embeddings).
        let mut cfg = qwen35_default();
        cfg.has_own_embed_tokens = false;
        let m = expected_manifest(&cfg);
        assert!(
            m.iter().all(|t| t.name != "embed_tokens.weight"),
            "embed_tokens.weight should be absent when has_own_embed_tokens=false"
        );
        // Loading a synthetic blob without embed_tokens still works.
        let blob = build_synthetic_safetensors(&m);
        Eagle3Weights::load(&blob, &cfg).expect("loader accepts shared-embed manifest");
    }

    #[test]
    fn adr_037_e3_gate_d2t_canonicalized_to_draft_id_to_target_id_2026_05_22() {
        // Codex Major 1: vLLM-format checkpoints emit `d2t` as the
        // safetensors key; loader must canonicalize to
        // `draft_id_to_target_id`. We build a synthetic blob using
        // the raw `d2t` name and the canonical-named manifest's
        // other tensors, then verify load succeeds.
        let cfg = qwen35_default();
        let canonical_manifest = expected_manifest(&cfg);
        // Rebuild manifest with `d2t` instead of `draft_id_to_target_id`
        // to drive the synthetic-blob builder.
        let mut raw_manifest = canonical_manifest.clone();
        for t in raw_manifest.iter_mut() {
            if t.name == "draft_id_to_target_id" {
                t.name = "d2t".to_string();
            }
        }
        let blob = build_synthetic_safetensors(&raw_manifest);
        // Loader sees `d2t` in the file, canonicalizes to
        // `draft_id_to_target_id`, then matches the canonical manifest.
        let w = Eagle3Weights::load(&blob, &cfg).expect("d2t normalization should work");
        // Lookup uses canonical name — works because manifest stored
        // canonical names, even though the file used `d2t`.
        let map = w.tensor("draft_id_to_target_id").expect("found by canonical");
        assert_eq!(map.shape(), &[cfg.draft_vocab_size]);
    }

    #[test]
    fn adr_037_e3_gate_t2d_skipped_at_load_2026_05_22() {
        // Codex Major 1 part 2: vLLM peer skips `t2d` (inverse mapping)
        // when loading. Adding a `t2d` tensor to the synthetic blob
        // should NOT trigger Extra error.
        let cfg = qwen35_default();
        let mut manifest = expected_manifest(&cfg);
        manifest.push(ExpectedTensor::int_i64(
            "t2d",
            vec![cfg.vocab_size],
        ));
        let blob = build_synthetic_safetensors(&manifest);
        // Original cfg expected manifest — load with cfg, NOT
        // augmented manifest, so loader does its own resolution.
        Eagle3Weights::load(&blob, &cfg).expect("t2d should be silently skipped");
    }

    #[test]
    fn adr_037_e3_gate_load_validates_config_2026_05_22() {
        // Codex Major 3: load() must call cfg.validate() at entry.
        // Construct an invalid config and a (vacuously) tiny blob —
        // load should fail with Config(), not panic on later math.
        //
        // ADR-038 G4-CFA-5 (2026-05-23): the `num_q_heads * head_dim ==
        // hidden_size` invariant was relaxed (Llama-style drafters
        // legitimately violate it). Trigger a still-valid invariant:
        // GQA divisibility (`num_q_heads % num_kv_heads == 0`).
        let mut cfg = qwen35_default();
        cfg.num_kv_heads = 7; // 40 % 7 != 0 → GQA invariant violated
        // Build SOME synthetic blob (empty manifest is fine since
        // validate fires first).
        let blob = build_synthetic_safetensors(&[]);
        let err = Eagle3Weights::load(&blob, &cfg).unwrap_err();
        assert!(
            matches!(err, Eagle3WeightsError::Config(_)),
            "expected Config error, got: {err:?}"
        );
    }

    #[test]
    fn adr_037_e3b_load_rejects_wrong_dtype_2026_05_22() {
        let cfg = qwen35_default();
        let mut manifest = expected_manifest(&cfg);
        // Find norm.weight (float) and corrupt its dtype to I64.
        let nidx = manifest.iter().position(|t| t.name == "norm.weight").unwrap();
        manifest[nidx].dtype = Dtype::I64;
        let blob = build_synthetic_safetensors(&manifest);
        // Original cfg expects BF16; synthetic blob has I64.
        let err = Eagle3Weights::load(&blob, &cfg).unwrap_err();
        assert!(
            matches!(err, Eagle3WeightsError::Dtype { ref name, .. } if name == "norm.weight"),
            "got: {err:?}"
        );
    }

    /// AC-G4-4.3 — verifier_lm_head.weight and verifier_norm.weight are silently
    /// skipped (in `_keys_to_ignore_on_save` per Speculators eagle3/core.py).
    /// Adding both to a synthetic blob must NOT trigger Extra errors.
    #[test]
    fn g4_cfa4_verifier_tensors_silently_skipped_2026_05_23() {
        let cfg = qwen35_default();
        let mut manifest = expected_manifest(&cfg);
        // Inject both verifier tensors into the blob — they should be silently
        // dropped by resolve_name returning None, not cause Eagle3WeightsError::Extra.
        manifest.push(ExpectedTensor {
            name: "verifier_lm_head.weight".to_string(),
            shape: vec![cfg.vocab_size, cfg.hidden_size],
            dtype: Dtype::BF16,
        });
        manifest.push(ExpectedTensor {
            name: "verifier_norm.weight".to_string(),
            shape: vec![cfg.hidden_size],
            dtype: Dtype::BF16,
        });
        let blob = build_synthetic_safetensors(&manifest);
        // Load with the canonical config (which does NOT include verifier tensors).
        // Must succeed — verifier tensors skipped, not flagged as extra.
        Eagle3Weights::load(&blob, &cfg)
            .expect("verifier_lm_head.weight + verifier_norm.weight must be silently skipped");
    }
}
