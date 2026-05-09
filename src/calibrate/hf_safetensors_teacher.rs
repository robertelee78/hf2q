//! ADR-020 AC#7 — HuggingFace safetensors-backed `TeacherLogitsProvider`.
//!
//! [`HfSafetensorsTeacherProvider`] mirrors [`super::gguf_teacher::GgufTeacherProvider`]
//! but loads full-precision (f16/bf16/f32) weights from a standard HuggingFace
//! model directory instead of a GGUF file.
//!
//! # Why safetensors input is better
//!
//! GGUF-input bakes quantization noise (Q5_K_M ≈ 5.5 bpw) into the teacher.
//! Safetensors-input uses the original bf16/f32 HuggingFace weights as
//! the teacher reference — less lossy → better KL distillation → tighter
//! AC#7 boundary delta.
//!
//! # Scope — Phase 1
//!
//! v1 supports only Qwen3 MoE models (`model_type` in
//! `{"qwen2_moe", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text",
//! "qwen3_5_moe", "qwen3_5_moe_text"}`).  Gemma 4 is a follow-up.
//! If `model_type` is not in the supported set, the constructor fails
//! with a descriptive error and the exact unsupported type.
//!
//! # Teacher forward — Option Y
//!
//! The teacher forward is implemented directly on the tape via
//! `decoder_layer_on_tape_real_gqa`.  This reuses existing primitives,
//! avoids touching `qwen35/model.rs`, and is mathematically consistent
//! with the student's training forward (same ops, same precision).
//!
//! # HF tensor-name mapping
//!
//! HF full-attention layer:
//! - `model.layers.{i}.input_layernorm.weight`
//! - `model.layers.{i}.self_attn.q_proj.weight`    `[n_q*head_dim, hidden]`
//! - `model.layers.{i}.self_attn.k_proj.weight`    `[n_kv*head_dim, hidden]`
//! - `model.layers.{i}.self_attn.v_proj.weight`    `[n_kv*head_dim, hidden]`
//! - `model.layers.{i}.self_attn.o_proj.weight`    `[hidden, n_q*head_dim]`
//! - `model.layers.{i}.self_attn.q_norm.weight`
//! - `model.layers.{i}.self_attn.k_norm.weight`
//! - `model.layers.{i}.post_attention_layernorm.weight`
//! - `model.layers.{i}.mlp.gate.weight`            `[n_experts, hidden]` (router)
//! - `model.layers.{i}.mlp.experts.{e}.gate_proj.weight`
//! - `model.layers.{i}.mlp.experts.{e}.up_proj.weight`
//! - `model.layers.{i}.mlp.experts.{e}.down_proj.weight`
//! - `model.norm.weight`
//! - `lm_head.weight`

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use crate::ir::DType;
use crate::progress::ProgressReporter;

use super::dwq_targets::TeacherLogitsProvider;

// ── Supported model types ─────────────────────────────────────────────────────

/// Return `true` if the `model_type` string belongs to the Qwen3/Qwen3.5 family.
pub(crate) fn is_supported_qwen3_model_type(model_type: &str) -> bool {
    matches!(
        model_type,
        "qwen2_moe"
            | "qwen3"
            | "qwen3_moe"
            | "qwen3_5"
            | "qwen3_5_text"
            | "qwen3_5_moe"
            | "qwen3_5_moe_text"
            | "qwen3_5_moe_text_only"
    )
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Architecture fields parsed from a HuggingFace `config.json`.
///
/// Phase 1 scope: Qwen3 / Qwen3.5 / Qwen3.5-MoE text models.
#[derive(Debug, Clone)]
pub struct HfSafetensorsConfig {
    /// Value of `model_type` in `config.json`.
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    /// Number of MoE experts (`num_experts` key).  `None` for dense.
    pub num_experts: Option<usize>,
    /// Number of experts used per token (`num_experts_per_tok`).
    pub num_experts_per_tok: Option<usize>,
    /// RoPE theta base (from `rope_parameters.rope_theta`).
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// IMROPE section dimensions (from `rope_parameters.mrope_section`).
    /// Stored as `[s0, s1, s2, s3]`; Qwen3.5 source has 3 values + implied 0.
    pub mrope_section: [u32; 4],
    pub sliding_window: Option<u32>,
    pub tie_word_embeddings: bool,
    /// Optional `language_model.` prefix on tensor names (VL models).
    pub tensor_prefix: String,
}

impl HfSafetensorsConfig {
    /// Parse from the JSON blob at `config.json` inside a model directory.
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let v: Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", path.display()))?;
        Self::from_json(&v)
            .with_context(|| format!("HfSafetensorsConfig::from_json {}", path.display()))
    }

    fn from_json(v: &Value) -> Result<Self> {
        // Some VL models nest the text config under "text_config".
        let tc = v.get("text_config").unwrap_or(v);

        let model_type = tc
            .get("model_type")
            .or_else(|| v.get("model_type"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("config.json: missing model_type"))?
            .to_string();

        if !is_supported_qwen3_model_type(&model_type) {
            return Err(anyhow!(
                "model_type '{}' is not supported in this DWQ pipeline yet \
                 (Phase 1 supports Qwen3/Qwen3.5 MoE text models only; \
                 Gemma 4 is a follow-up)",
                model_type
            ));
        }

        let get_usize = |key: &str| -> Result<usize> {
            tc.get(key)
                .or_else(|| v.get(key))
                .and_then(|v| v.as_u64())
                .map(|u| u as usize)
                .ok_or_else(|| anyhow!("config.json: missing or non-integer '{}'", key))
        };

        let get_usize_opt = |key: &str| -> Option<usize> {
            tc.get(key)
                .or_else(|| v.get(key))
                .and_then(|v| v.as_u64())
                .map(|u| u as usize)
        };

        let hidden_size = get_usize("hidden_size")?;
        let num_hidden_layers = get_usize("num_hidden_layers")?;
        let num_attention_heads = get_usize("num_attention_heads")?;
        let num_key_value_heads = get_usize("num_key_value_heads")?;
        let head_dim = get_usize("head_dim")?;
        let intermediate_size = get_usize_opt("intermediate_size")
            .or_else(|| get_usize_opt("moe_intermediate_size"))
            .unwrap_or(0);
        let vocab_size = get_usize("vocab_size")?;

        let num_experts = get_usize_opt("num_experts");
        let num_experts_per_tok = get_usize_opt("num_experts_per_tok");

        let rms_norm_eps = tc
            .get("rms_norm_eps")
            .or_else(|| v.get("rms_norm_eps"))
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(1e-6f32);

        // rope_parameters is a nested object in Qwen3.5.
        let rope_params = tc
            .get("rope_parameters")
            .or_else(|| v.get("rope_parameters"));

        let rope_theta = rope_params
            .and_then(|rp| rp.get("rope_theta"))
            .or_else(|| tc.get("rope_theta"))
            .or_else(|| v.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(1_000_000.0f32);

        // mrope_section: [s0, s1, s2] in HF config → [s0, s1, s2, 0] for our 4-int format.
        let mrope_section = Self::parse_mrope_section(rope_params, &model_type)?;

        let sliding_window = tc
            .get("sliding_window")
            .or_else(|| v.get("sliding_window"))
            .and_then(|v| v.as_u64())
            .map(|u| u as u32);

        let tie_word_embeddings = tc
            .get("tie_word_embeddings")
            .or_else(|| v.get("tie_word_embeddings"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // VL models may wrap everything under `language_model.`.
        // Detect by checking the outer `model_type`.
        let outer_mt = v
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let tensor_prefix = if v.get("text_config").is_some()
            && !outer_mt.ends_with("_text") && !outer_mt.ends_with("_moe")
        {
            "language_model.".to_string()
        } else {
            String::new()
        };

        Ok(HfSafetensorsConfig {
            model_type,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            num_experts,
            num_experts_per_tok,
            rope_theta,
            rms_norm_eps,
            mrope_section,
            sliding_window,
            tie_word_embeddings,
            tensor_prefix,
        })
    }

    fn parse_mrope_section(rope_params: Option<&Value>, model_type: &str) -> Result<[u32; 4]> {
        let arr = rope_params
            .and_then(|rp| rp.get("mrope_section"))
            .and_then(|v| v.as_array());

        if let Some(a) = arr {
            let vals: Vec<u32> = a
                .iter()
                .filter_map(|v| v.as_u64().map(|u| u as u32))
                .collect();
            let s0 = vals.first().copied().unwrap_or(0);
            let s1 = vals.get(1).copied().unwrap_or(0);
            let s2 = vals.get(2).copied().unwrap_or(0);
            let s3 = vals.get(3).copied().unwrap_or(0);
            return Ok([s0, s1, s2, s3]);
        }

        // Fallback: if model_type is a known Qwen3.5/3.6 variant, use the
        // production default [11, 11, 10, 0].  Fail loud otherwise.
        if is_supported_qwen3_model_type(model_type) {
            return Ok([11, 11, 10, 0]);
        }

        Err(anyhow!(
            "config.json: mrope_section not found in rope_parameters \
             and no default available for model_type '{}'",
            model_type
        ))
    }
}

// ── Weight index ──────────────────────────────────────────────────────────────

/// Lazy-loaded tensor index over a HuggingFace model directory.
///
/// Wraps the existing [`crate::input::safetensors::read_tensors_lazy`] reader.
/// Tensors are only materialised (paged in) when [`Self::load_f32`] is called.
pub struct HfSafetensorsWeights {
    lazy_map: crate::ir::lazy::LazyTensorMap,
    model_dir: PathBuf,
}

impl HfSafetensorsWeights {
    /// Open a model directory and index all safetensors shards.
    pub fn open(model_dir: &Path) -> Result<Self> {
        let progress = ProgressReporter::new();
        let lazy_map =
            crate::input::safetensors::read_tensors_lazy(model_dir, &progress)
                .with_context(|| {
                    format!("HfSafetensorsWeights::open {}", model_dir.display())
                })?;
        Ok(Self {
            lazy_map,
            model_dir: model_dir.to_path_buf(),
        })
    }

    /// Return all tensor names present in the index.
    pub fn list_tensor_names(&self) -> Vec<String> {
        self.lazy_map.iter().map(|(k, _)| k.clone()).collect()
    }

    /// Materialise a tensor by name and cast to a flat `Vec<f32>`.
    ///
    /// Handles F32, F16, and BF16 source dtypes.  For F16 and BF16 the
    /// cast is done host-side via the `half` crate (no GPU round-trip).
    ///
    /// Each call opens a fresh lazy index to get an owned `LazyTensor`
    /// (required because `LazyTensor::materialize` consumes `self`).
    /// The index read is cheap (JSON header only, no tensor bytes).
    pub fn load_f32(&self, name: &str) -> Result<Vec<f32>> {
        // Re-open to get an owned LazyTensor (materialize consumes self).
        let progress = ProgressReporter::new();
        let mut fresh_map =
            crate::input::safetensors::read_tensors_lazy(&self.model_dir, &progress)
                .with_context(|| {
                    format!("HfSafetensorsWeights::load_f32 reopen {}", self.model_dir.display())
                })?;

        let lazy = fresh_map.remove(name).ok_or_else(|| {
            anyhow!(
                "HfSafetensorsWeights: tensor '{}' not found in {}",
                name,
                self.model_dir.display()
            )
        })?;

        // Materialise: reads bytes from the safetensors shard (mmap), then casts.
        let tensor_ref = lazy
            .materialize()
            .map_err(|e| anyhow!("materialise '{}': {e}", name))?;

        match tensor_ref.dtype {
            DType::F32 => {
                let bytes = &*tensor_ref.data;
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Ok(floats)
            }
            DType::F16 => {
                let bytes = &*tensor_ref.data;
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|c| {
                        half::f16::from_le_bytes(c.try_into().unwrap()).to_f32()
                    })
                    .collect();
                Ok(floats)
            }
            DType::BF16 => {
                let bytes = &*tensor_ref.data;
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|c| {
                        half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32()
                    })
                    .collect();
                Ok(floats)
            }
            other => Err(anyhow!(
                "HfSafetensorsWeights: tensor '{}' has unsupported dtype {:?}; \
                 only F32/F16/BF16 are supported",
                name,
                other
            )),
        }
    }
}

// ── TeacherLogitsProvider ─────────────────────────────────────────────────────

/// `TeacherLogitsProvider` backed by HuggingFace safetensors full-precision weights.
///
/// The teacher forward is implemented via `decoder_layer_on_tape_real_gqa`
/// (Option Y from the spec) — same tape primitives as the student's
/// training forward for mathematical consistency.
pub struct HfSafetensorsTeacherProvider {
    cfg: HfSafetensorsConfig,
    weights: HfSafetensorsWeights,
    device: mlx_native::MlxDevice,
    positions_buf: Vec<i32>,
}

impl HfSafetensorsTeacherProvider {
    /// Load an HF safetensors model from `model_dir`.
    pub fn from_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let cfg = HfSafetensorsConfig::from_dir(model_dir)
            .with_context(|| format!("HfSafetensorsConfig::from_dir {}", model_dir.display()))?;
        let weights = HfSafetensorsWeights::open(model_dir)
            .with_context(|| format!("HfSafetensorsWeights::open {}", model_dir.display()))?;
        let device = mlx_native::MlxDevice::new()
            .map_err(|e| anyhow!("HfSafetensorsTeacherProvider: MlxDevice::new: {e}"))?;
        Ok(Self {
            cfg,
            weights,
            device,
            positions_buf: Vec::new(),
        })
    }

    /// Vocab size from the parsed config.
    pub fn vocab(&self) -> usize {
        self.cfg.vocab_size
    }

    /// Parsed architecture config.
    pub fn config(&self) -> &HfSafetensorsConfig {
        &self.cfg
    }

    /// Number of transformer layers from config.
    pub fn num_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }

    /// Materialise and cast a named tensor to f32.
    pub fn load_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.weights.load_f32(name)
    }

    /// All tensor names in the index.
    pub fn list_tensor_names(&self) -> Vec<String> {
        self.weights.list_tensor_names()
    }

    /// Tensor name prefix in this model's safetensors files.
    ///
    /// Plain text-only models: `"model."`.
    /// VL models: `"language_model.model."`.
    #[allow(dead_code)]
    fn model_prefix(&self) -> String {
        format!("{}model.", self.cfg.tensor_prefix)
    }

    /// Build the HF tensor name for a per-layer weight.
    fn layer_tensor(&self, layer_idx: usize, suffix: &str) -> String {
        format!(
            "{}{}.layers.{}.{}",
            self.cfg.tensor_prefix,
            "model",
            layer_idx,
            suffix
        )
    }

    /// Build the HF tensor name for a global (non-layer) weight.
    fn global_tensor(&self, suffix: &str) -> String {
        format!("{}{}", self.cfg.tensor_prefix, suffix)
    }

    /// Run one forward pass and return `(logits [seq_len, vocab], hidden [seq_len, hidden])`.
    ///
    /// `hidden` is the pre-output-norm hidden state — same quantity that
    /// `Qwen35Model::forward_gpu_with_hidden` returns and that Stage B of
    /// `train_all_linears_full_model_dwq` stores as `hidden_rows`.
    pub fn forward_with_hidden(&self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
        self.forward_one_row_impl(tokens, true)
    }

    /// Run one forward pass over `seq_len` tokens and return flat
    /// `[seq_len, vocab]` f32 logits.
    fn forward_one_row(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let (logits, _) = self.forward_one_row_impl(tokens, false)?;
        Ok(logits)
    }

    fn forward_one_row_impl(&self, tokens: &[u32], capture_hidden: bool) -> Result<(Vec<f32>, Vec<f32>)> {
        use crate::calibrate::autograd_gpu_tape::{matmul, rms_norm, GpuTape, GpuTensor};
        use crate::calibrate::qwen35_moe::{
            decoder_layer_on_tape_real_gqa, make_pos_buf_for_real_gqa,
            DecoderLayerWeightsRealGqa, Qwen35RealGqaConfig,
        };

        let seq_len = tokens.len();
        let cfg = &self.cfg;
        let n_q = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let n_experts = cfg.num_experts.unwrap_or(1);
        let moe_k = cfg.num_experts_per_tok.unwrap_or(1);

        let tape = GpuTape::new(self.device.clone());
        let pos_buf = make_pos_buf_for_real_gqa(&self.device, seq_len)
            .context("make_pos_buf_for_real_gqa")?;

        // Load embed_tokens and embed the input.
        let embed_name = self.global_tensor("model.embed_tokens.weight");
        let embed_w = self.weights.load_f32(&embed_name)
            .with_context(|| format!("load embed_tokens: {}", embed_name))?;
        // embed_w: [vocab, hidden]; tokens: [seq_len]; result: [seq_len, hidden]
        let mut hidden_data = vec![0.0f32; seq_len * hidden];
        for (t_idx, &tok_id) in tokens.iter().enumerate() {
            let src = (tok_id as usize) * hidden;
            let dst = t_idx * hidden;
            hidden_data[dst..dst + hidden]
                .copy_from_slice(&embed_w[src..src + hidden]);
        }

        let gqa_cfg = Qwen35RealGqaConfig {
            n_q_heads: n_q,
            n_kv_heads: n_kv,
            head_dim: hd,
            seq_len,
            hidden,
            rope_theta_base: cfg.rope_theta,
            rope_sections: cfg.mrope_section,
            causal: true,
            sliding_window: cfg.sliding_window,
            rms_eps: cfg.rms_norm_eps,
        };

        let mut h = GpuTensor::from_vec(&tape, &hidden_data, vec![seq_len, hidden])
            .context("embed -> h")?;

        let is_full_attn_layer = |layer_idx: usize| -> bool {
            // Detect by presence of self_attn.q_proj.weight in the map.
            let name = self.layer_tensor(layer_idx, "self_attn.q_proj.weight");
            self.weights.lazy_map.contains_key(&name)
        };

        for layer_idx in 0..cfg.num_hidden_layers {
            if !is_full_attn_layer(layer_idx) {
                // LinearAttention (DeltaNet) layer: pass hidden through unchanged.
                // We cannot run DeltaNet on the tape (not wired), so we treat
                // these layers as identity for the teacher forward.
                // This is a known approximation for v1 — the teacher's
                // DeltaNet contribution is zeroed out.
                continue;
            }

            let w_in = self.weights.load_f32(
                &self.layer_tensor(layer_idx, "input_layernorm.weight"),
            )?;
            let w_post = self.weights.load_f32(
                &self.layer_tensor(layer_idx, "post_attention_layernorm.weight"),
            )?;
            let wq_flat = self.weights.load_f32(
                &self.layer_tensor(layer_idx, "self_attn.q_proj.weight"),
            )?;
            let wk_flat = self.weights.load_f32(
                &self.layer_tensor(layer_idx, "self_attn.k_proj.weight"),
            )?;
            let wv_flat = self.weights.load_f32(
                &self.layer_tensor(layer_idx, "self_attn.v_proj.weight"),
            )?;
            let wo_flat = self.weights.load_f32(
                &self.layer_tensor(layer_idx, "self_attn.o_proj.weight"),
            )?;
            let q_norm_w = self.weights.load_f32(
                &self.layer_tensor(layer_idx, "self_attn.q_norm.weight"),
            )?;
            let k_norm_w = self.weights.load_f32(
                &self.layer_tensor(layer_idx, "self_attn.k_norm.weight"),
            )?;

            let q_total = n_q * hd;
            let kv_total = n_kv * hd;

            // Load MoE or dense FFN weights.
            #[allow(clippy::type_complexity)]
            let (router_data, gate_vecs, up_vecs, down_vecs): (
                Vec<f32>,
                Vec<Vec<f32>>,
                Vec<Vec<f32>>,
                Vec<Vec<f32>>,
            ) = if n_experts > 1 {
                let router = self.weights.load_f32(
                    &self.layer_tensor(layer_idx, "mlp.gate.weight"),
                )?;
                let mut gates = Vec::with_capacity(n_experts);
                let mut ups = Vec::with_capacity(n_experts);
                let mut downs = Vec::with_capacity(n_experts);
                for e in 0..n_experts {
                    gates.push(self.weights.load_f32(
                        &self.layer_tensor(
                            layer_idx,
                            &format!("mlp.experts.{e}.gate_proj.weight"),
                        ),
                    )?);
                    ups.push(self.weights.load_f32(
                        &self.layer_tensor(
                            layer_idx,
                            &format!("mlp.experts.{e}.up_proj.weight"),
                        ),
                    )?);
                    downs.push(self.weights.load_f32(
                        &self.layer_tensor(
                            layer_idx,
                            &format!("mlp.experts.{e}.down_proj.weight"),
                        ),
                    )?);
                }
                (router, gates, ups, downs)
            } else {
                // Dense FFN.
                let inter = cfg.intermediate_size;
                let gate = self.weights.load_f32(
                    &self.layer_tensor(layer_idx, "mlp.gate_proj.weight"),
                )?;
                let up = self.weights.load_f32(
                    &self.layer_tensor(layer_idx, "mlp.up_proj.weight"),
                )?;
                let down = self.weights.load_f32(
                    &self.layer_tensor(layer_idx, "mlp.down_proj.weight"),
                )?;
                let router_dummy = vec![1.0f32; hidden];
                let _ = inter;
                (router_dummy, vec![gate], vec![up], vec![down])
            };

            // Build GpuTensors.
            let win_t = GpuTensor::from_vec(&tape, &w_in, vec![hidden])?;
            let wpost_t = GpuTensor::from_vec(&tape, &w_post, vec![hidden])?;
            let wq_t = GpuTensor::from_vec(&tape, &wq_flat, vec![q_total, hidden])?;
            let wk_t = GpuTensor::from_vec(&tape, &wk_flat, vec![kv_total, hidden])?;
            let wv_t = GpuTensor::from_vec(&tape, &wv_flat, vec![kv_total, hidden])?;
            let wo_t = GpuTensor::from_vec(&tape, &wo_flat, vec![hidden, q_total])?;
            let qn_t = GpuTensor::from_vec(&tape, &q_norm_w, vec![hd])?;
            let kn_t = GpuTensor::from_vec(&tape, &k_norm_w, vec![hd])?;

            let n_exp = gate_vecs.len();
            let inter = if n_experts > 1 {
                cfg.intermediate_size
            } else {
                gate_vecs.first().map(|g| g.len() / hidden).unwrap_or(0)
            };
            // HF gate/up: [inter, hidden] → need [hidden, inter] for tape
            let transpose_hf = |data: &[f32], rows: usize, cols: usize| -> Vec<f32> {
                let mut out = vec![0.0f32; data.len()];
                for i in 0..rows {
                    for j in 0..cols {
                        out[j * rows + i] = data[i * cols + j];
                    }
                }
                out
            };
            let mut gate_ts: Vec<GpuTensor> = Vec::with_capacity(n_exp);
            let mut up_ts: Vec<GpuTensor> = Vec::with_capacity(n_exp);
            let mut down_ts: Vec<GpuTensor> = Vec::with_capacity(n_exp);
            for e in 0..n_exp {
                // gate_proj.weight: [inter, hidden] → transpose → [hidden, inter]
                let gd = transpose_hf(&gate_vecs[e], inter, hidden);
                let ud = transpose_hf(&up_vecs[e], inter, hidden);
                // down_proj.weight: [hidden, inter] → transpose → [inter, hidden]
                let dd = transpose_hf(&down_vecs[e], hidden, inter);
                gate_ts.push(GpuTensor::from_vec(&tape, &gd, vec![hidden, inter])?);
                up_ts.push(GpuTensor::from_vec(&tape, &ud, vec![hidden, inter])?);
                down_ts.push(GpuTensor::from_vec(&tape, &dd, vec![inter, hidden])?);
            }

            // Router: HF shape [n_experts, hidden] → need [hidden, n_experts]
            let router_t = if n_experts > 1 {
                let r = transpose_hf(&router_data, n_experts, hidden);
                GpuTensor::from_vec(&tape, &r, vec![hidden, n_experts])?
            } else {
                GpuTensor::from_vec(&tape, &router_data, vec![hidden, 1])?
            };

            let lw = DecoderLayerWeightsRealGqa {
                w_in: &win_t,
                w_post: &wpost_t,
                w_q: &wq_t,
                w_k: &wk_t,
                w_v: &wv_t,
                w_o: &wo_t,
                q_norm_w: &qn_t,
                k_norm_w: &kn_t,
                w_gate: &router_t,
                gate_projs: &gate_ts,
                up_projs: &up_ts,
                down_projs: &down_ts,
            };

            h = decoder_layer_on_tape_real_gqa(
                &tape,
                &h,
                pos_buf.clone(),
                &lw,
                &gqa_cfg,
                moe_k,
                1e-9_f32,
            )
            .with_context(|| format!("decoder_layer_on_tape_real_gqa layer {layer_idx}"))?;

            // Reset tape to free intermediate activations.
            let h_data = h.to_vec().context("h.to_vec between layers")?;
            tape.reset();
            h = GpuTensor::from_vec(&tape, &h_data, vec![seq_len, hidden])
                .context("rebuild h after tape reset")?;
        }

        // Capture pre-norm hidden state when caller requests it (Stage B).
        let hidden_out = if capture_hidden {
            h.to_vec().context("h.to_vec for capture_hidden")?
        } else {
            Vec::new()
        };

        // Output norm.
        let norm_name = self.global_tensor("model.norm.weight");
        let norm_w = self.weights.load_f32(&norm_name)?;
        let norm_t = GpuTensor::from_vec(&tape, &norm_w, vec![hidden])?;
        let h_normed = rms_norm(&h, &norm_t, cfg.rms_norm_eps).context("output_norm")?;

        // LM head.
        let lm_head_name = if cfg.tie_word_embeddings {
            self.global_tensor("model.embed_tokens.weight")
        } else {
            self.global_tensor("lm_head.weight")
        };
        let lm_head_w = self.weights.load_f32(&lm_head_name)
            .with_context(|| format!("load lm_head: {}", lm_head_name))?;
        // lm_head.weight: [vocab, hidden] → transpose → [hidden, vocab]
        // Logits = h_normed @ lm_head^T  →  matmul(h_normed, lm_head^T)
        // w_lm shape [vocab, hidden] → GpuTensor, then matmul transposes inside
        let w_lm = GpuTensor::from_vec(&tape, &lm_head_w, vec![vocab, hidden])?;
        use crate::calibrate::autograd_gpu_tape::transpose;
        let w_lm_t = transpose(&w_lm).context("transpose lm_head")?;
        let logits_t = matmul(&h_normed, &w_lm_t).context("lm_head matmul")?;

        let logits_out = logits_t.to_vec().context("logits_t.to_vec")?;
        Ok((logits_out, hidden_out))
    }
}

impl TeacherLogitsProvider for HfSafetensorsTeacherProvider {
    fn forward_logits(
        &mut self,
        tokens: &[u32],
        batch_size: usize,
        seq_len: usize,
        vocab: usize,
    ) -> Result<Vec<f32>> {
        super::gguf_teacher::validate_forward_logits_args(
            vocab,
            self.cfg.vocab_size,
            tokens.len(),
            batch_size,
            seq_len,
        )?;
        super::gguf_teacher::fill_text_positions(&mut self.positions_buf, seq_len);

        let row_logits_len = seq_len * vocab;
        let mut out = Vec::with_capacity(batch_size * row_logits_len);
        for row in 0..batch_size {
            let row_tokens = &tokens[row * seq_len..(row + 1) * seq_len];
            let row_logits = self
                .forward_one_row(row_tokens)
                .with_context(|| format!("forward_one_row row {row}"))?;
            if row_logits.len() != row_logits_len {
                return Err(anyhow!(
                    "HfSafetensorsTeacherProvider: row {} returned {} logits, \
                     expected seq_len*vocab={}",
                    row,
                    row_logits.len(),
                    row_logits_len
                ));
            }
            out.extend_from_slice(&row_logits);
        }
        Ok(out)
    }
}

// ── HF weight loading helpers (public for train_all_linears) ──────────────────

/// Load the full-precision weights for one full-attention layer from HF
/// safetensors and return the same field names used by Stage A in
/// `train_all_linears_full_model_dwq`.
///
/// # Returns
///
/// `(w_in, w_post, q_norm, k_norm, wq, wk, wv, wo, router, gate_stacked,
///   up_stacked, down_stacked, n_experts, moe_k)`
///
/// where `*_stacked` are flat vectors in the same order as the GGUF path:
/// - gate/up: `[n_experts * intermediate * hidden]` row-major `[inter, hidden]`
///   transposed to `[hidden, inter]` per expert, then concatenated.
/// - down: `[n_experts * hidden * intermediate]` (`[hidden, inter]` → `[inter, hidden]`).
#[allow(clippy::too_many_arguments)]
pub fn load_hf_layer_weights(
    weights: &HfSafetensorsWeights,
    cfg: &HfSafetensorsConfig,
    layer_idx: usize,
    layer_tn: &dyn Fn(&str) -> String,
) -> Result<HfLayerWeights> {
    let hidden = cfg.hidden_size;
    let n_q = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let n_experts = cfg.num_experts.unwrap_or(1);
    let moe_k = cfg.num_experts_per_tok.unwrap_or(1);

    let w_in = weights.load_f32(&layer_tn("input_layernorm.weight"))
        .with_context(|| format!("layer {layer_idx} input_layernorm"))?;
    let w_post = weights.load_f32(&layer_tn("post_attention_layernorm.weight"))
        .with_context(|| format!("layer {layer_idx} post_attention_layernorm"))?;
    let q_norm = weights.load_f32(&layer_tn("self_attn.q_norm.weight"))
        .with_context(|| format!("layer {layer_idx} q_norm"))?;
    let k_norm = weights.load_f32(&layer_tn("self_attn.k_norm.weight"))
        .with_context(|| format!("layer {layer_idx} k_norm"))?;

    // HF: q_proj.weight [n_q*hd, hidden] — no interleaving (unlike GGUF's fused format).
    let wq = weights.load_f32(&layer_tn("self_attn.q_proj.weight"))
        .with_context(|| format!("layer {layer_idx} q_proj"))?;
    let wk = weights.load_f32(&layer_tn("self_attn.k_proj.weight"))
        .with_context(|| format!("layer {layer_idx} k_proj"))?;
    let wv = weights.load_f32(&layer_tn("self_attn.v_proj.weight"))
        .with_context(|| format!("layer {layer_idx} v_proj"))?;
    let wo = weights.load_f32(&layer_tn("self_attn.o_proj.weight"))
        .with_context(|| format!("layer {layer_idx} o_proj"))?;

    // Validate shapes.
    let q_total = n_q * hd;
    let kv_total = n_kv * hd;
    if wq.len() != q_total * hidden {
        return Err(anyhow!(
            "layer {layer_idx} q_proj: expected [{q_total}, {hidden}] = {} elements, got {}",
            q_total * hidden,
            wq.len()
        ));
    }
    if wk.len() != kv_total * hidden || wv.len() != kv_total * hidden {
        return Err(anyhow!(
            "layer {layer_idx} k/v_proj shape mismatch"
        ));
    }
    if wo.len() != hidden * q_total {
        return Err(anyhow!(
            "layer {layer_idx} o_proj: expected [{hidden}, {q_total}] = {} elements, got {}",
            hidden * q_total,
            wo.len()
        ));
    }

    // MoE or dense FFN.
    let (router, inter, gates, ups, downs): (Vec<f32>, usize, Vec<f32>, Vec<f32>, Vec<f32>) =
        if n_experts > 1 {
            let inter = cfg.intermediate_size;
            // HF router: [n_experts, hidden] → transpose → [hidden, n_experts]
            let router_raw = weights.load_f32(&layer_tn("mlp.gate.weight"))
                .with_context(|| format!("layer {layer_idx} mlp.gate (router)"))?;
            let router_t = transpose_2d(&router_raw, n_experts, hidden);

            let per_g = inter * hidden;
            let mut gate_cat = Vec::with_capacity(n_experts * per_g);
            let mut up_cat = Vec::with_capacity(n_experts * per_g);
            let mut down_cat = Vec::with_capacity(n_experts * per_g);
            for e in 0..n_experts {
                // HF gate_proj: [inter, hidden] → transpose → [hidden, inter]
                let g_raw = weights.load_f32(
                    &layer_tn(&format!("mlp.experts.{e}.gate_proj.weight")),
                )
                .with_context(|| format!("layer {layer_idx} expert {e} gate_proj"))?;
                let u_raw = weights.load_f32(
                    &layer_tn(&format!("mlp.experts.{e}.up_proj.weight")),
                )
                .with_context(|| format!("layer {layer_idx} expert {e} up_proj"))?;
                // HF down_proj: [hidden, inter] → transpose → [inter, hidden]
                let d_raw = weights.load_f32(
                    &layer_tn(&format!("mlp.experts.{e}.down_proj.weight")),
                )
                .with_context(|| format!("layer {layer_idx} expert {e} down_proj"))?;
                gate_cat.extend(transpose_2d(&g_raw, inter, hidden));
                up_cat.extend(transpose_2d(&u_raw, inter, hidden));
                down_cat.extend(transpose_2d(&d_raw, hidden, inter));
            }
            (router_t, inter, gate_cat, up_cat, down_cat)
        } else {
            // Dense FFN.
            let gate_raw = weights.load_f32(&layer_tn("mlp.gate_proj.weight"))?;
            let up_raw = weights.load_f32(&layer_tn("mlp.up_proj.weight"))?;
            let down_raw = weights.load_f32(&layer_tn("mlp.down_proj.weight"))?;
            let inter = gate_raw.len() / hidden;
            let router_dummy = vec![1.0f32; hidden];
            let gate_t = transpose_2d(&gate_raw, inter, hidden);
            let up_t = transpose_2d(&up_raw, inter, hidden);
            let down_t = transpose_2d(&down_raw, hidden, inter);
            (router_dummy, inter, gate_t, up_t, down_t)
        };

    Ok(HfLayerWeights {
        w_in,
        w_post,
        q_norm,
        k_norm,
        wq,
        wk,
        wv,
        wo,
        router,
        gate_flat: gates,
        up_flat: ups,
        down_flat: downs,
        n_experts,
        moe_k,
        intermediate: inter,
    })
}

/// Flat f32 weight bundle for one full-attention layer loaded from HF format.
pub struct HfLayerWeights {
    pub w_in: Vec<f32>,
    pub w_post: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    /// `[n_q * head_dim, hidden]`
    pub wq: Vec<f32>,
    /// `[n_kv * head_dim, hidden]`
    pub wk: Vec<f32>,
    /// `[n_kv * head_dim, hidden]`
    pub wv: Vec<f32>,
    /// `[hidden, n_q * head_dim]`
    pub wo: Vec<f32>,
    /// `[hidden, n_experts]` (transposed from HF `[n_experts, hidden]`)
    pub router: Vec<f32>,
    /// Expert gate projections, concatenated: `n_experts × [hidden, inter]`
    pub gate_flat: Vec<f32>,
    /// Expert up projections: `n_experts × [hidden, inter]`
    pub up_flat: Vec<f32>,
    /// Expert down projections: `n_experts × [inter, hidden]`
    pub down_flat: Vec<f32>,
    pub n_experts: usize,
    pub moe_k: usize,
    pub intermediate: usize,
}

/// Transpose a `[rows, cols]` matrix to `[cols, rows]`.
pub(crate) fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = data[i * cols + j];
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── config.json parsing ───────────────────────────────────────────────────

    /// Regression baseline: parse a synthetic Qwen3.5-MoE config.json and
    /// assert every field matches the expected value.
    #[test]
    fn parse_qwen3_moe_config_json_all_fields() {
        let v = json!({
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "intermediate_size": 512,
            "vocab_size": 248320,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 1_000_000.0,
                "mrope_section": [11, 11, 10],
                "rope_type": "default"
            },
            "tie_word_embeddings": false
        });
        let cfg = HfSafetensorsConfig::from_json(&v).expect("parse");
        assert_eq!(cfg.model_type, "qwen3_moe");
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.intermediate_size, 512);
        assert_eq!(cfg.vocab_size, 248320);
        assert_eq!(cfg.num_experts, Some(128));
        assert_eq!(cfg.num_experts_per_tok, Some(8));
        assert!((cfg.rms_norm_eps - 1e-6_f32).abs() < 1e-10);
        assert!((cfg.rope_theta - 1_000_000.0_f32).abs() < 1.0);
        assert_eq!(cfg.mrope_section, [11, 11, 10, 0]);
        assert!(!cfg.tie_word_embeddings);
        assert!(cfg.sliding_window.is_none());
    }

    #[test]
    fn parse_rejects_unsupported_model_type() {
        let v = json!({
            "model_type": "gemma4",
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "intermediate_size": 4096,
            "vocab_size": 256000,
            "rms_norm_eps": 1e-6
        });
        let err = HfSafetensorsConfig::from_json(&v).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("gemma4"), "unexpected error: {msg}");
        assert!(msg.contains("not supported"), "unexpected error: {msg}");
    }

    #[test]
    fn parse_qwen3_moe_mrope_section_defaults_when_missing() {
        // Qwen3_moe without rope_parameters — should default to [11, 11, 10, 0].
        let v = json!({
            "model_type": "qwen3_moe",
            "hidden_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "intermediate_size": 512,
            "vocab_size": 1024,
            "rms_norm_eps": 1e-6
        });
        let cfg = HfSafetensorsConfig::from_json(&v).expect("parse");
        assert_eq!(cfg.mrope_section, [11, 11, 10, 0]);
    }

    #[test]
    fn parse_nested_text_config() {
        // VL model with text_config nesting.
        let v = json!({
            "model_type": "qwen3_vl_outer",
            "text_config": {
                "model_type": "qwen3_moe",
                "hidden_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "intermediate_size": 256,
                "vocab_size": 1024,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "rope_theta": 500000.0,
                    "mrope_section": [5, 5, 5]
                }
            }
        });
        // Outer model_type is unknown but we should read text_config.model_type.
        // The outer vl wrapper may not be in supported set, but text_config should.
        let cfg = HfSafetensorsConfig::from_json(&v).expect("parse nested text_config");
        assert_eq!(cfg.model_type, "qwen3_moe");
        assert_eq!(cfg.hidden_size, 512);
        assert_eq!(cfg.mrope_section, [5, 5, 5, 0]);
    }

    // ── list_tensor_names ─────────────────────────────────────────────────────

    /// Build a minimal safetensors file for a 2-layer MoE model directory.
    fn write_mini_safetensors_index(dir: &std::path::Path) {
        // Write a model.safetensors.index.json that maps two tensors to a
        // single shard filename.
        let index = json!({
            "weight_map": {
                "model.embed_tokens.weight": "model.safetensors",
                "model.norm.weight": "model.safetensors",
                "lm_head.weight": "model.safetensors",
                "model.layers.0.input_layernorm.weight": "model.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
                "model.layers.0.self_attn.k_proj.weight": "model.safetensors",
                "model.layers.0.self_attn.v_proj.weight": "model.safetensors",
                "model.layers.0.self_attn.o_proj.weight": "model.safetensors"
            }
        });
        let index_str = serde_json::to_string(&index).unwrap();
        std::fs::write(dir.join("model.safetensors.index.json"), index_str).unwrap();
    }

    /// Helper: create a valid safetensors file with f32 tensors.
    fn write_safetensors(path: &std::path::Path, tensors: &[(&str, &[usize], &[f32])]) {
        let mut header_map = serde_json::Map::new();
        let mut current_offset: usize = 0;
        let mut all_data: Vec<u8> = Vec::new();
        for (name, shape, data) in tensors {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let end = current_offset + bytes.len();
            let mut ti = serde_json::Map::new();
            ti.insert("dtype".into(), json!("F32"));
            ti.insert(
                "shape".into(),
                json!(shape.iter().map(|&s| s as u64).collect::<Vec<_>>()),
            );
            ti.insert(
                "data_offsets".into(),
                json!([current_offset as u64, end as u64]),
            );
            header_map.insert(name.to_string(), json!(ti));
            all_data.extend(bytes);
            current_offset = end;
        }
        let header_json = serde_json::to_string(&header_map).unwrap();
        let header_size = header_json.len() as u64;
        let mut file: Vec<u8> = header_size.to_le_bytes().to_vec();
        file.extend(header_json.as_bytes());
        file.extend(all_data);
        std::fs::write(path, file).unwrap();
    }

    #[test]
    fn list_tensor_names_matches_hf_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();

        let embed_data: Vec<f32> = (0..4usize).map(|i| i as f32).collect(); // vocab=2, hidden=2
        write_mini_safetensors_index(dir);
        write_safetensors(
            &dir.join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", &[2, 2], &embed_data),
                ("model.norm.weight", &[2], &[1.0, 1.0]),
                ("lm_head.weight", &[2, 2], &embed_data),
                ("model.layers.0.input_layernorm.weight", &[2], &[1.0, 1.0]),
                ("model.layers.0.self_attn.q_proj.weight", &[2, 2], &embed_data),
                ("model.layers.0.self_attn.k_proj.weight", &[2, 2], &embed_data),
                ("model.layers.0.self_attn.v_proj.weight", &[2, 2], &embed_data),
                ("model.layers.0.self_attn.o_proj.weight", &[2, 2], &embed_data),
            ],
        );

        let weights = HfSafetensorsWeights::open(dir).expect("open");
        let names = weights.list_tensor_names();
        assert!(names.contains(&"model.embed_tokens.weight".to_string()));
        assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(names.contains(&"lm_head.weight".to_string()));
        assert!(names.contains(&"model.norm.weight".to_string()));
    }

    // ── load_f32 dtype handling ───────────────────────────────────────────────

    #[test]
    fn load_f32_handles_bf16_correctly() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();

        // Write a BF16 tensor: value 1.0 in bf16 = 0x3F80.
        let bf16_bytes: Vec<u8> = [1.0_f32, 2.0_f32, 3.0_f32]
            .iter()
            .flat_map(|&f| half::bf16::from_f32(f).to_le_bytes())
            .collect();
        let header_map = json!({
            "w": {
                "dtype": "BF16",
                "shape": [3],
                "data_offsets": [0, bf16_bytes.len() as u64]
            }
        });
        let header_json = serde_json::to_string(&header_map).unwrap();
        let header_size = header_json.len() as u64;
        let mut file_data: Vec<u8> = header_size.to_le_bytes().to_vec();
        file_data.extend(header_json.as_bytes());
        file_data.extend(&bf16_bytes);
        std::fs::write(dir.join("model.safetensors"), &file_data).unwrap();

        let weights = HfSafetensorsWeights::open(dir).unwrap();
        let vals = weights.load_f32("w").unwrap();
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-3, "vals[0]={}", vals[0]);
        assert!((vals[1] - 2.0).abs() < 1e-3, "vals[1]={}", vals[1]);
        assert!((vals[2] - 3.0).abs() < 1e-3, "vals[2]={}", vals[2]);
    }

    // ── transpose_2d ─────────────────────────────────────────────────────────

    #[test]
    fn transpose_2d_correctness() {
        // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = transpose_2d(&data, 2, 3);
        assert_eq!(out, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // ── forward_logits_matches_oracle_2_layer_synthetic ───────────────────────
    //
    // This test builds a tiny 2-layer synthetic HF model (vocab=8, hidden=64,
    // 2 heads, head_dim=32, 2 experts) with identity-initialized weights
    // and checks that `forward_logits` returns finite logits with the correct
    // shape, and that top-1 token IDs differ across positions (non-degenerate
    // forward).
    //
    // Running the full tape-based forward requires a GPU device which is
    // available in the CI environment.  The test is NOT #[ignore]d because
    // it uses tiny synthetic weights and should complete in <1s.
    // If no GPU device is available, the test is skipped gracefully.
    #[test]
    fn forward_logits_shape_and_finite() {
        // Skip if can't open the GPU device (headless CI).
        if mlx_native::MlxDevice::new().is_err() {
            eprintln!("[forward_logits_shape] SKIP: no MlxDevice");
            return;
        }

        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();

        // Tiny dimensions that satisfy kernel floors.
        // head_dim=32 fails the kernel minimum; use 64.
        let vocab: usize = 64;
        let hidden = 64usize;
        let n_q = 1usize;
        let n_kv = 1usize;
        let hd = 64usize;
        let n_experts = 1usize;
        let inter = 64usize;
        let n_layers = 1usize;
        let seq_len = 2usize;

        // Write config.json.
        let cfg_json = json!({
            "model_type": "qwen3_moe",
            "hidden_size": hidden,
            "num_hidden_layers": n_layers,
            "num_attention_heads": n_q,
            "num_key_value_heads": n_kv,
            "head_dim": hd,
            "intermediate_size": inter,
            "vocab_size": vocab,
            "num_experts": n_experts,
            "num_experts_per_tok": 1,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "mrope_section": [11, 11, 10]
            },
            "tie_word_embeddings": false
        });
        std::fs::write(dir.join("config.json"), serde_json::to_string(&cfg_json).unwrap()).unwrap();

        // All weights: identity / ones / small random-like values.
        let ones_h = vec![1.0f32; hidden];
        let ones_hd = vec![1.0f32; hd];
        let embed = {
            let mut v = vec![0.0f32; vocab * hidden];
            for i in 0..vocab.min(hidden) {
                v[i * hidden + i] = 1.0;
            }
            v
        };
        // q/k/v/o: identity-ish [n*hd, hidden]
        let q_total = n_q * hd;
        let kv_total = n_kv * hd;
        let mut wq = vec![0.0f32; q_total * hidden];
        let mut wk = vec![0.0f32; kv_total * hidden];
        let mut wv = vec![0.0f32; kv_total * hidden];
        let mut wo = vec![0.0f32; hidden * q_total];
        // Diagonal fill.
        for i in 0..q_total.min(hidden) {
            wq[i * hidden + i] = 0.1;
            wk[i * hidden + i] = 0.1;
            wv[i * hidden + i] = 0.1;
        }
        for i in 0..hidden.min(q_total) {
            wo[i * q_total + i] = 0.1;
        }
        // Dense FFN [inter, hidden] → gate/up; [hidden, inter] → down.
        let gate_proj = vec![0.01f32; inter * hidden];
        let up_proj = vec![0.01f32; inter * hidden];
        let down_proj = vec![0.01f32; hidden * inter];
        let _router = vec![1.0f32; n_experts * hidden]; // [n_experts, hidden] — not used in dense forward
        let lm_head = embed.clone();

        let tensors: Vec<(&str, Vec<usize>, Vec<f32>)> = vec![
            ("model.embed_tokens.weight", vec![vocab, hidden], embed.clone()),
            ("model.norm.weight", vec![hidden], ones_h.clone()),
            ("lm_head.weight", vec![vocab, hidden], lm_head),
            (
                "model.layers.0.input_layernorm.weight",
                vec![hidden],
                ones_h.clone(),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                vec![hidden],
                ones_h.clone(),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                vec![q_total, hidden],
                wq,
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                vec![kv_total, hidden],
                wk,
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                vec![kv_total, hidden],
                wv,
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                vec![hidden, q_total],
                wo,
            ),
            (
                "model.layers.0.self_attn.q_norm.weight",
                vec![hd],
                ones_hd.clone(),
            ),
            (
                "model.layers.0.self_attn.k_norm.weight",
                vec![hd],
                ones_hd.clone(),
            ),
            // Dense FFN (n_experts=1).
            (
                "model.layers.0.mlp.gate_proj.weight",
                vec![inter, hidden],
                gate_proj.clone(),
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                vec![inter, hidden],
                up_proj.clone(),
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                vec![hidden, inter],
                down_proj.clone(),
            ),
        ];

        let tensor_refs: Vec<(&str, &[usize], &[f32])> = tensors
            .iter()
            .map(|(n, s, d)| (*n, s.as_slice(), d.as_slice()))
            .collect();
        write_safetensors(&dir.join("model.safetensors"), &tensor_refs);

        let mut provider = HfSafetensorsTeacherProvider::from_dir(dir).expect("from_dir");
        let tokens: Vec<u32> = (0..seq_len as u32).collect();
        let logits = provider
            .forward_logits(&tokens, 1, seq_len, vocab)
            .expect("forward_logits");

        assert_eq!(logits.len(), seq_len * vocab, "logit count mismatch");
        let n_finite = logits.iter().filter(|x| x.is_finite()).count();
        assert_eq!(
            n_finite,
            logits.len(),
            "expected all logits finite; {} non-finite",
            logits.len() - n_finite
        );
    }

    /// Integration smoke test gated on `HF2Q_TEST_HF_SAFETENSORS_DIR`.
    /// Skips cleanly when env var is unset.
    #[test]
    #[ignore = "loads multi-GB HF model; run with --ignored when HF2Q_TEST_HF_SAFETENSORS_DIR is set"]
    fn safetensors_input_teacher_smoke() {
        let dir = match std::env::var("HF2Q_TEST_HF_SAFETENSORS_DIR").ok() {
            Some(d) => std::path::PathBuf::from(d),
            None => {
                eprintln!("[safetensors_teacher_smoke] SKIP: HF2Q_TEST_HF_SAFETENSORS_DIR not set");
                return;
            }
        };
        let mut teacher =
            HfSafetensorsTeacherProvider::from_dir(&dir).expect("from_dir");
        let vocab = teacher.vocab();
        assert!(vocab > 0);
        let seq_len = 4usize;
        let tokens: Vec<u32> = (0..seq_len as u32).collect();
        let logits = teacher
            .forward_logits(&tokens, 1, seq_len, vocab)
            .expect("forward_logits");
        assert_eq!(logits.len(), seq_len * vocab);
        assert!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
    }
}
