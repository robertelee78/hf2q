//! ADR-037 Phase E4b.1 — GPU-resident EAGLE-3 drafter weights.
//!
//! Mirrors `DFlashModelTensors` pattern at
//! `src/inference/spec_decode/dflash/tensors.rs` but adapted for the
//! EAGLE-3 1-layer drafter schema (single decoder layer; concat-input
//! qkv with `qkv_input_width = 2 * hidden_size`; optional gates per
//! [`Eagle3DrafterConfig`]).
//!
//! ## Dtype policy
//!
//! Same as DFlash (ADR-030 iter-106):
//! - **BF16** on-disk → BF16 on-GPU for **projection** weights
//!   (q/k/v/o_proj, mlp gate/up/down, fc, embed_tokens, lm_head, biases).
//! - **BF16** on-disk → **F32** on-GPU for **RMSNorm** weights
//!   (input_layernorm, hidden_norm, post_attention_layernorm, q_norm,
//!   k_norm, input_norm, fc_norm.{i}, final norm.weight). mlx-native's
//!   `rms_norm_f32` kernel requires F32 weight; passing BF16 over the
//!   F32 contract reads adjacent BF16 elements as misinterpreted
//!   F32 bits (the bug ADR-030 iter-106 fixed).
//! - **I64** for `draft_id_to_target_id` (vocab remap). Stored on
//!   CPU as `Vec<i64>` since mlx-native's DType has no I64 variant
//!   and this mapping is consumed post-logits-download anyway.

use super::config::Eagle3DrafterConfig;
use super::weights::{Eagle3Weights, Eagle3WeightsError};
use mlx_native::{DType, MlxBuffer, MlxDevice, MlxError};
use safetensors::tensor::TensorView;

#[derive(Debug, thiserror::Error)]
pub enum Eagle3TensorsError {
    #[error("eagle3 tensors mlx: {0}")]
    Mlx(#[from] MlxError),
    #[error("eagle3 tensors weights: {0}")]
    Weights(#[from] Eagle3WeightsError),
    #[error("eagle3 tensors: missing manifest entry `{0}`")]
    MissingEntry(String),
}

/// GPU-resident EAGLE-3 drafter weights.
///
/// Fields gated by [`Eagle3DrafterConfig`] are `Option<MlxBuffer>`;
/// always-present fields are `MlxBuffer`. The struct shape matches
/// the manifest schema in `weights.rs::expected_manifest`.
pub struct Eagle3DrafterTensors {
    // === Globals ===
    /// `[vocab_size, hidden_size]` BF16. None when drafter shares
    /// target's embedding table (`has_own_embed_tokens = false`).
    pub embed_tokens: Option<MlxBuffer>,
    /// `[hidden_size, fc_input_size]` BF16. Projects concat
    /// multi-aux hidden into drafter hidden space.
    pub fc: MlxBuffer,
    /// `[fc_input_size]` F32 RMSNorm — applied BEFORE `fc` when
    /// `norm_before_fc = true`. None otherwise.
    pub input_norm: Option<MlxBuffer>,
    /// `num_aux_hidden_states` × `[target_hidden_size]` F32 RMSNorms,
    /// applied per-aux chunk before concat+fc when `fc_norm = true`.
    /// Empty Vec when `fc_norm = false`.
    pub fc_norm: Vec<MlxBuffer>,
    /// `[hidden_size]` F32 — final RMSNorm before lm_head.
    pub norm: MlxBuffer,
    /// `[draft_vocab_size, hidden_size]` BF16. None when tied with
    /// embed_tokens (`tie_lm_head = true`).
    pub lm_head: Option<MlxBuffer>,
    /// `[draft_vocab_size]` I64 — draft-vocab → target-vocab mapping.
    /// **Stored on CPU (Vec<i64>)** because mlx-native's DType enum
    /// has no I64 variant; this mapping is consumed on the CPU side
    /// after logits download (per vLLM `llama_eagle3.py:375-385`
    /// `targets = base + draft_id_to_target_id; logits_new[:, targets]
    /// = logits`), so a CPU buffer is functionally correct.
    /// None when `include_draft_id_mapping = false`.
    pub draft_id_to_target_id: Option<Vec<i64>>,

    // === Layer 0 (the only decoder layer) ===
    /// `[hidden_size]` F32 — RMSNorm of embeds branch.
    pub input_layernorm: MlxBuffer,
    /// `[hidden_size]` F32 — RMSNorm of hidden_states branch
    /// (post-fc). Per vLLM `llama_eagle3.py:69`.
    pub hidden_norm: MlxBuffer,
    /// `[hidden_size]` F32 — RMSNorm between attn residual + MLP.
    pub post_attention_layernorm: MlxBuffer,
    /// `[num_q_heads * head_dim, 2*hidden_size]` BF16. NOTE: input
    /// width is `2*hidden_size` (not `hidden_size`) because EAGLE-3
    /// layer-0 receives the CONCAT of normed-embeds + normed-hidden.
    pub q_proj: MlxBuffer,
    /// `[num_kv_heads * head_dim, 2*hidden_size]` BF16. Same
    /// 2x-input-width invariant as q_proj.
    pub k_proj: MlxBuffer,
    /// `[num_kv_heads * head_dim, 2*hidden_size]` BF16.
    pub v_proj: MlxBuffer,
    /// `[hidden_size, num_q_heads * head_dim]` BF16. Standard
    /// hidden-out-from-q-heads shape.
    pub o_proj: MlxBuffer,
    /// `[head_dim]` F32. None when `use_qk_norm = false`.
    pub q_norm: Option<MlxBuffer>,
    /// `[head_dim]` F32. None when `use_qk_norm = false`.
    pub k_norm: Option<MlxBuffer>,
    /// `[num_q_heads * head_dim]` F32 (cast from BF16 at upload for
    /// add_bias_row_2d_f32 compatibility). Present when
    /// `attention_bias = true`.
    pub q_bias: Option<MlxBuffer>,
    /// `[num_kv_heads * head_dim]` F32.
    pub k_bias: Option<MlxBuffer>,
    /// `[num_kv_heads * head_dim]` F32.
    pub v_bias: Option<MlxBuffer>,
    /// `[hidden_size]` F32.
    pub o_bias: Option<MlxBuffer>,
    /// `[intermediate_size, hidden_size]` BF16 SwiGLU gate proj.
    pub mlp_gate: MlxBuffer,
    /// `[intermediate_size, hidden_size]` BF16 SwiGLU up proj.
    pub mlp_up: MlxBuffer,
    /// `[hidden_size, intermediate_size]` BF16 SwiGLU down proj.
    pub mlp_down: MlxBuffer,
}

/// Decode little-endian BF16 bytes into F32 values.
///
/// Re-implementation of `dflash::tensors::decode_bf16_bytes_to_f32`
/// (kept module-local so the two spec-decode flavors stay
/// independently shippable). BF16 = top 16 bits of an F32;
/// reconstruction is `f32_bits = (bf16_bits as u32) << 16`.
fn decode_bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, Eagle3TensorsError> {
    let n_elem = bytes.len() / 2;
    if bytes.len() != n_elem * 2 {
        return Err(Eagle3TensorsError::Mlx(MlxError::InvalidArgument(format!(
            "decode_bf16_bytes_to_f32: data len {} not BF16-aligned (odd byte count)",
            bytes.len()
        ))));
    }
    let mut out = Vec::with_capacity(n_elem);
    for i in 0..n_elem {
        let lo = bytes[i * 2] as u32;
        let hi = bytes[i * 2 + 1] as u32;
        let bf16_bits = lo | (hi << 8);
        out.push(f32::from_bits(bf16_bits << 16));
    }
    Ok(out)
}

fn upload_bf16(
    device: &MlxDevice,
    view: &TensorView<'_>,
) -> Result<MlxBuffer, Eagle3TensorsError> {
    let shape: Vec<usize> = view.shape().to_vec();
    let byte_len = view.data().len();
    let mut buf = device.alloc_buffer(byte_len, DType::BF16, shape)?;
    let dst: &mut [u8] = buf.as_mut_slice::<u8>().map_err(|e| {
        Eagle3TensorsError::Mlx(MlxError::InvalidArgument(format!("buffer slice: {e}")))
    })?;
    debug_assert_eq!(dst.len(), byte_len);
    dst.copy_from_slice(view.data());
    Ok(buf)
}

fn upload_bf16_as_f32(
    device: &MlxDevice,
    view: &TensorView<'_>,
) -> Result<MlxBuffer, Eagle3TensorsError> {
    let f32_values = decode_bf16_bytes_to_f32(view.data())?;
    let n_elem = f32_values.len();
    let shape: Vec<usize> = view.shape().to_vec();
    let mut buf = device.alloc_buffer(n_elem * 4, DType::F32, shape)?;
    let dst: &mut [f32] = buf.as_mut_slice::<f32>().map_err(|e| {
        Eagle3TensorsError::Mlx(MlxError::InvalidArgument(format!("f32 slice: {e}")))
    })?;
    debug_assert_eq!(dst.len(), n_elem);
    dst.copy_from_slice(&f32_values);
    Ok(buf)
}

fn decode_i64_le(view: &TensorView<'_>) -> Result<Vec<i64>, Eagle3TensorsError> {
    let bytes = view.data();
    if bytes.len() % 8 != 0 {
        return Err(Eagle3TensorsError::Mlx(MlxError::InvalidArgument(format!(
            "decode_i64_le: data len {} not I64-aligned (not multiple of 8)",
            bytes.len()
        ))));
    }
    let n = bytes.len() / 8;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
        out.push(i64::from_le_bytes(arr));
    }
    Ok(out)
}

fn fetch<'a, 'b>(
    weights: &'a Eagle3Weights<'b>,
    name: &str,
) -> Result<&'a TensorView<'b>, Eagle3TensorsError> {
    weights
        .tensor(name)
        .ok_or_else(|| Eagle3TensorsError::MissingEntry(name.to_string()))
}

impl Eagle3DrafterTensors {
    /// Upload validated EAGLE-3 weights to GPU.
    ///
    /// Pre-conditions: `weights` was loaded against the SAME `cfg`
    /// (the manifest schemas must agree). The loader at
    /// `weights.rs::Eagle3Weights::load` already validates this
    /// against the safetensors file at load time.
    pub fn upload(
        device: &MlxDevice,
        cfg: &Eagle3DrafterConfig,
        weights: &Eagle3Weights<'_>,
    ) -> Result<Self, Eagle3TensorsError> {
        // --- Globals ---
        let embed_tokens = if cfg.has_own_embed_tokens {
            Some(upload_bf16(device, fetch(weights, "embed_tokens.weight")?)?)
        } else {
            None
        };
        let fc = upload_bf16(device, fetch(weights, "fc.weight")?)?;
        let input_norm = if cfg.norm_before_fc {
            Some(upload_bf16_as_f32(device, fetch(weights, "input_norm.weight")?)?)
        } else {
            None
        };
        let fc_norm = if cfg.fc_norm {
            let mut v = Vec::with_capacity(cfg.num_aux_hidden_states);
            for i in 0..cfg.num_aux_hidden_states {
                v.push(upload_bf16_as_f32(
                    device,
                    fetch(weights, &format!("fc_norm.{i}.weight"))?,
                )?);
            }
            v
        } else {
            Vec::new()
        };
        let norm = upload_bf16_as_f32(device, fetch(weights, "norm.weight")?)?;
        let lm_head = if cfg.tie_lm_head {
            None
        } else {
            Some(upload_bf16(device, fetch(weights, "lm_head.weight")?)?)
        };
        let draft_id_to_target_id = if cfg.include_draft_id_mapping {
            Some(decode_i64_le(fetch(weights, "draft_id_to_target_id")?)?)
        } else {
            None
        };

        // --- Layer 0 ---
        let input_layernorm = upload_bf16_as_f32(
            device,
            fetch(weights, "layers.0.input_layernorm.weight")?,
        )?;
        let hidden_norm =
            upload_bf16_as_f32(device, fetch(weights, "layers.0.hidden_norm.weight")?)?;
        let post_attention_layernorm = upload_bf16_as_f32(
            device,
            fetch(weights, "layers.0.post_attention_layernorm.weight")?,
        )?;
        let q_proj =
            upload_bf16(device, fetch(weights, "layers.0.self_attn.q_proj.weight")?)?;
        let k_proj =
            upload_bf16(device, fetch(weights, "layers.0.self_attn.k_proj.weight")?)?;
        let v_proj =
            upload_bf16(device, fetch(weights, "layers.0.self_attn.v_proj.weight")?)?;
        let o_proj =
            upload_bf16(device, fetch(weights, "layers.0.self_attn.o_proj.weight")?)?;
        let (q_norm, k_norm) = if cfg.use_qk_norm {
            (
                Some(upload_bf16_as_f32(
                    device,
                    fetch(weights, "layers.0.self_attn.q_norm.weight")?,
                )?),
                Some(upload_bf16_as_f32(
                    device,
                    fetch(weights, "layers.0.self_attn.k_norm.weight")?,
                )?),
            )
        } else {
            (None, None)
        };
        // Q/K/V/O biases cast BF16 → F32 at upload time so the
        // `add_bias_row_2d_f32` kernel can consume them directly.
        // Same rationale as RMSNorm weights (ADR-030 iter-106): the
        // bias-add kernel declares F32 bias and would mis-stride a
        // BF16 buffer.
        let (q_bias, k_bias, v_bias, o_bias) = if cfg.attention_bias {
            (
                Some(upload_bf16_as_f32(
                    device,
                    fetch(weights, "layers.0.self_attn.q_proj.bias")?,
                )?),
                Some(upload_bf16_as_f32(
                    device,
                    fetch(weights, "layers.0.self_attn.k_proj.bias")?,
                )?),
                Some(upload_bf16_as_f32(
                    device,
                    fetch(weights, "layers.0.self_attn.v_proj.bias")?,
                )?),
                Some(upload_bf16_as_f32(
                    device,
                    fetch(weights, "layers.0.self_attn.o_proj.bias")?,
                )?),
            )
        } else {
            (None, None, None, None)
        };
        let mlp_gate =
            upload_bf16(device, fetch(weights, "layers.0.mlp.gate_proj.weight")?)?;
        let mlp_up =
            upload_bf16(device, fetch(weights, "layers.0.mlp.up_proj.weight")?)?;
        let mlp_down =
            upload_bf16(device, fetch(weights, "layers.0.mlp.down_proj.weight")?)?;

        let tensors = Self {
            embed_tokens,
            fc,
            input_norm,
            fc_norm,
            norm,
            lm_head,
            draft_id_to_target_id,
            input_layernorm,
            hidden_norm,
            post_attention_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            q_bias,
            k_bias,
            v_bias,
            o_bias,
            mlp_gate,
            mlp_up,
            mlp_down,
        };

        // Codex /cfa E4b.1 Minor (2026-05-22): post-upload dtype
        // tripwire. Mirrors DFlash's invariant guards. Defense against
        // ADR-030 iter-106-style RMSNorm-over-BF16 silent corruption:
        // if a future refactor mistakenly routes an RMSNorm weight
        // through `upload_bf16` instead of `upload_bf16_as_f32`, this
        // catches it BEFORE the rms_norm_f32 kernel reads adjacent
        // BF16 elements as misinterpreted F32 bits.
        debug_assert_eq!(tensors.fc.dtype(), DType::BF16);
        debug_assert_eq!(tensors.norm.dtype(), DType::F32);
        debug_assert_eq!(tensors.input_layernorm.dtype(), DType::F32);
        debug_assert_eq!(tensors.hidden_norm.dtype(), DType::F32);
        debug_assert_eq!(tensors.post_attention_layernorm.dtype(), DType::F32);
        debug_assert_eq!(tensors.q_proj.dtype(), DType::BF16);
        debug_assert_eq!(tensors.k_proj.dtype(), DType::BF16);
        debug_assert_eq!(tensors.v_proj.dtype(), DType::BF16);
        debug_assert_eq!(tensors.o_proj.dtype(), DType::BF16);
        debug_assert_eq!(tensors.mlp_gate.dtype(), DType::BF16);
        debug_assert_eq!(tensors.mlp_up.dtype(), DType::BF16);
        debug_assert_eq!(tensors.mlp_down.dtype(), DType::BF16);
        if let Some(b) = &tensors.embed_tokens {
            debug_assert_eq!(b.dtype(), DType::BF16);
        }
        if let Some(b) = &tensors.input_norm {
            debug_assert_eq!(b.dtype(), DType::F32);
        }
        for b in &tensors.fc_norm {
            debug_assert_eq!(b.dtype(), DType::F32);
        }
        if let Some(b) = &tensors.lm_head {
            debug_assert_eq!(b.dtype(), DType::BF16);
        }
        if let Some(b) = &tensors.q_norm {
            debug_assert_eq!(b.dtype(), DType::F32);
        }
        if let Some(b) = &tensors.k_norm {
            debug_assert_eq!(b.dtype(), DType::F32);
        }
        // Biases now cast to F32 at upload (matches add_bias_row_2d_f32
        // kernel input expectation).
        for opt in [&tensors.q_bias, &tensors.k_bias, &tensors.v_bias, &tensors.o_bias] {
            if let Some(b) = opt {
                debug_assert_eq!(b.dtype(), DType::F32);
            }
        }

        Ok(tensors)
    }

    /// Total GPU-resident bytes across all uploaded Metal buffers.
    ///
    /// Codex /cfa E4b.1 Minor (2026-05-22): excludes
    /// `draft_id_to_target_id` — that mapping lives on CPU as
    /// `Vec<i64>`, NOT in a Metal buffer. Use `cpu_resident_bytes()`
    /// for that and `total_resident_bytes()` for the sum.
    pub fn gpu_resident_bytes(&self) -> usize {
        let mut total = 0;
        total += self.fc.byte_len();
        total += self.norm.byte_len();
        total += self.input_layernorm.byte_len();
        total += self.hidden_norm.byte_len();
        total += self.post_attention_layernorm.byte_len();
        total += self.q_proj.byte_len();
        total += self.k_proj.byte_len();
        total += self.v_proj.byte_len();
        total += self.o_proj.byte_len();
        total += self.mlp_gate.byte_len();
        total += self.mlp_up.byte_len();
        total += self.mlp_down.byte_len();
        if let Some(b) = &self.embed_tokens {
            total += b.byte_len();
        }
        if let Some(b) = &self.input_norm {
            total += b.byte_len();
        }
        for b in &self.fc_norm {
            total += b.byte_len();
        }
        if let Some(b) = &self.lm_head {
            total += b.byte_len();
        }
        // draft_id_to_target_id is CPU-resident; not counted here
        // (codex /cfa E4b.1 Minor fix 2026-05-22).
        if let Some(b) = &self.q_norm {
            total += b.byte_len();
        }
        if let Some(b) = &self.k_norm {
            total += b.byte_len();
        }
        if let Some(b) = &self.q_bias {
            total += b.byte_len();
        }
        if let Some(b) = &self.k_bias {
            total += b.byte_len();
        }
        if let Some(b) = &self.v_bias {
            total += b.byte_len();
        }
        if let Some(b) = &self.o_bias {
            total += b.byte_len();
        }
        total
    }

    /// Bytes resident on CPU (currently only the `draft_id_to_target_id`
    /// vocab-remap vector when present).
    pub fn cpu_resident_bytes(&self) -> usize {
        self.draft_id_to_target_id
            .as_ref()
            .map_or(0, |v| v.len() * std::mem::size_of::<i64>())
    }

    /// Total resident bytes = GPU + CPU.
    pub fn total_resident_bytes(&self) -> usize {
        self.gpu_resident_bytes() + self.cpu_resident_bytes()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::eagle3::weights::{
        expected_manifest, Eagle3Weights, ExpectedTensor,
    };
    use safetensors::tensor::Dtype as SafeDtype;
    use std::collections::BTreeMap;

    // ----------------------------------------------------------------
    // Bit-decode tests (pure CPU — no Metal device needed)
    // ----------------------------------------------------------------

    fn bf16_bytes_from_f32(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for v in values {
            let bf16_bits = (v.to_bits() >> 16) as u16;
            out.push((bf16_bits & 0xff) as u8);
            out.push(((bf16_bits >> 8) & 0xff) as u8);
        }
        out
    }

    #[test]
    fn adr_037_e4b_decode_bf16_round_trips_canonical_2026_05_22() {
        // Values with all-zero low 16 bits survive BF16 truncation.
        let canonical = [0.0f32, 1.0, -1.0, 2.0, 0.5, -0.5];
        let bytes = bf16_bytes_from_f32(&canonical);
        let decoded = decode_bf16_bytes_to_f32(&bytes).expect("decode ok");
        for (i, (got, want)) in decoded.iter().zip(canonical.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "canonical[{i}] = {want} round-trip got {got}"
            );
        }
    }

    #[test]
    fn adr_037_e4b_decode_bf16_rejects_odd_length_2026_05_22() {
        let err = decode_bf16_bytes_to_f32(&[0x80, 0x3f, 0x00]).unwrap_err();
        assert!(format!("{err}").contains("not BF16-aligned"), "got: {err}");
    }

    // ----------------------------------------------------------------
    // GPU upload tests using synthetic safetensors blobs
    // ----------------------------------------------------------------

    /// Build a synthetic safetensors blob matching the expected
    /// manifest. Float tensors get zero bytes; I64 tensors get zero
    /// bytes (which is a valid I64 0). Owns its storage so the
    /// returned bytes are usable by `Eagle3Weights::load`.
    fn build_synthetic_safetensors(manifest: &[ExpectedTensor]) -> Vec<u8> {
        let mut storage: Vec<Vec<u8>> = Vec::with_capacity(manifest.len());
        let mut tensors: BTreeMap<String, TensorView> = BTreeMap::new();
        for exp in manifest {
            let elem_bytes = match exp.dtype {
                SafeDtype::BF16 => 2,
                SafeDtype::I64 => 8,
                _ => panic!("unexpected dtype in test"),
            };
            let nelem: usize = exp.shape.iter().product();
            storage.push(vec![0u8; nelem * elem_bytes]);
        }
        for (i, exp) in manifest.iter().enumerate() {
            let view =
                TensorView::new(exp.dtype, exp.shape.clone(), storage[i].as_slice())
                    .expect("synthetic tensor view");
            tensors.insert(exp.name.clone(), view);
        }
        safetensors::serialize(
            &tensors,
            None::<std::collections::HashMap<String, String>>,
        )
        .expect("serialize synthetic")
    }

    /// Smaller config than qwen35_default so synthetic tests run fast.
    fn tiny_cfg() -> Eagle3DrafterConfig {
        Eagle3DrafterConfig {
            hidden_size: 256,
            intermediate_size: 512,
            head_dim: 32,
            num_q_heads: 8,
            num_kv_heads: 4,
            vocab_size: 1000,
            draft_vocab_size: 1000,
            target_hidden_size: 256,
            num_aux_hidden_states: 3,
            rms_norm_eps: 1e-6,
            norm_before_fc: false,
            fc_norm: true,
            use_qk_norm: true,
            attention_bias: false,
            tie_lm_head: false,
            include_draft_id_mapping: true,
            has_own_embed_tokens: true,
            rope_theta: 1_000_000.0,
            rope_dim: 32,
        }
    }

    #[test]
    fn adr_037_e4b1_upload_default_qwen35_config_succeeds_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return, // skip when no Metal available
        };
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_synthetic_safetensors(&manifest);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors = Eagle3DrafterTensors::upload(&device, &cfg, &weights)
            .expect("upload to GPU");
        // Always-present tensors: fc, norm, layer norms (3), projections (4),
        // mlp (3) = 12.
        assert_eq!(tensors.fc.dtype(), DType::BF16);
        assert_eq!(tensors.norm.dtype(), DType::F32, "norm cast to F32 on upload");
        assert_eq!(tensors.input_layernorm.dtype(), DType::F32);
        assert_eq!(tensors.hidden_norm.dtype(), DType::F32);
        assert_eq!(tensors.post_attention_layernorm.dtype(), DType::F32);
        assert_eq!(tensors.q_proj.dtype(), DType::BF16);
        // Conditional tensors per tiny_cfg() gates:
        assert!(
            tensors.embed_tokens.is_some(),
            "has_own_embed_tokens = true"
        );
        assert!(tensors.input_norm.is_none(), "norm_before_fc = false");
        assert_eq!(
            tensors.fc_norm.len(),
            3,
            "fc_norm = true, num_aux = 3"
        );
        assert!(tensors.q_norm.is_some(), "use_qk_norm = true");
        assert!(tensors.k_norm.is_some(), "use_qk_norm = true");
        assert!(tensors.q_bias.is_none(), "attention_bias = false");
        assert!(tensors.lm_head.is_some(), "tie_lm_head = false");
        assert!(
            tensors.draft_id_to_target_id.is_some(),
            "include_draft_id_mapping = true"
        );
    }

    #[test]
    fn adr_037_e4b1_upload_all_gates_off_minimum_tensors_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut cfg = tiny_cfg();
        cfg.norm_before_fc = false;
        cfg.fc_norm = false;
        cfg.use_qk_norm = false;
        cfg.attention_bias = false;
        cfg.tie_lm_head = true;
        cfg.include_draft_id_mapping = false;
        cfg.has_own_embed_tokens = false;
        let manifest = expected_manifest(&cfg);
        let blob = build_synthetic_safetensors(&manifest);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors = Eagle3DrafterTensors::upload(&device, &cfg, &weights)
            .expect("minimum-config upload");
        // All Option<MlxBuffer> fields should be None.
        assert!(tensors.embed_tokens.is_none());
        assert!(tensors.input_norm.is_none());
        assert!(tensors.fc_norm.is_empty());
        assert!(tensors.lm_head.is_none());
        assert!(tensors.draft_id_to_target_id.is_none());
        assert!(tensors.q_norm.is_none() && tensors.k_norm.is_none());
        assert!(tensors.q_bias.is_none());
    }

    #[test]
    fn adr_037_e4b1_upload_all_gates_on_maximum_tensors_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut cfg = tiny_cfg();
        cfg.norm_before_fc = true;
        cfg.fc_norm = true;
        cfg.use_qk_norm = true;
        cfg.attention_bias = true;
        cfg.tie_lm_head = false;
        cfg.include_draft_id_mapping = true;
        cfg.has_own_embed_tokens = true;
        let manifest = expected_manifest(&cfg);
        let blob = build_synthetic_safetensors(&manifest);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let tensors = Eagle3DrafterTensors::upload(&device, &cfg, &weights)
            .expect("maximum-config upload");
        assert!(tensors.embed_tokens.is_some());
        assert!(tensors.input_norm.is_some());
        assert_eq!(tensors.fc_norm.len(), cfg.num_aux_hidden_states);
        assert!(tensors.lm_head.is_some());
        assert!(tensors.draft_id_to_target_id.is_some());
        assert!(tensors.q_norm.is_some() && tensors.k_norm.is_some());
        assert!(tensors.q_bias.is_some());
        assert!(tensors.k_bias.is_some());
        assert!(tensors.v_bias.is_some());
        assert!(tensors.o_bias.is_some());
    }

    #[test]
    fn adr_037_e4b1_gpu_resident_bytes_includes_f32_cast_expansion_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let cfg = tiny_cfg();
        let manifest = expected_manifest(&cfg);
        let blob = build_synthetic_safetensors(&manifest);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("weights load");
        let safetensors_data_bytes: usize =
            weights.tensors.iter().map(|t| t.data().len()).sum();
        let tensors = Eagle3DrafterTensors::upload(&device, &cfg, &weights)
            .expect("upload to GPU");
        // Test uses TOTAL bytes (gpu + cpu) since safetensors_data_bytes
        // counts the I64 mapping which is CPU-resident (codex /cfa
        // E4b.1 Minor 2026-05-22).
        let total_bytes = tensors.total_resident_bytes();
        // F32-cast RMSNorm tensors double in size (BF16 2bpe → F32 4bpe).
        // tiny_cfg has: input_layernorm + hidden_norm + post_attention_layernorm
        // + q_norm + k_norm + 3 fc_norm + norm = 8 cast norms.
        // Each is hidden_size or head_dim or target_hidden_size elements.
        let cast_elems = cfg.hidden_size                       // input_layernorm
            + cfg.hidden_size                                  // hidden_norm
            + cfg.hidden_size                                  // post_attention_layernorm
            + cfg.head_dim                                     // q_norm
            + cfg.head_dim                                     // k_norm
            + cfg.num_aux_hidden_states * cfg.target_hidden_size // fc_norm × num_aux
            + cfg.hidden_size; // final norm
        let cast_expansion_bytes = cast_elems * 2; // BF16(2) → F32(4) adds 2 bytes per elem
        assert_eq!(
            total_bytes,
            safetensors_data_bytes + cast_expansion_bytes,
            "total bytes (gpu+cpu) = safetensors data ({safetensors_data_bytes}) + F32 cast expansion ({cast_expansion_bytes})"
        );
        // Also verify gpu/cpu split is consistent.
        assert_eq!(
            tensors.gpu_resident_bytes() + tensors.cpu_resident_bytes(),
            total_bytes
        );
        // CPU bytes = draft_vocab_size * 8 bytes (i64 mapping).
        assert_eq!(
            tensors.cpu_resident_bytes(),
            cfg.draft_vocab_size * 8,
            "CPU bytes should be draft_vocab_size * 8 (i64 mapping)"
        );
    }
}
