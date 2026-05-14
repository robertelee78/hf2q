//! DFlash drafter GPU weight residence (ADR-030 §3.3).
//!
//! Uploads validated BF16 safetensors data ([`super::weights::DFlashWeights`])
//! to mlx-native [`MlxBuffer`] resident on the Metal device. After this
//! step the drafter forward pass operates purely on GPU memory.
//!
//! Per [`super::config::DFlashConfig`] / `model_mlx.py:DFlashAttention.__init__`,
//! every weight is stored row-major BF16 in safetensors. Upload is a single
//! `copy_from_slice` per tensor (no transpose, no dequantization). Shapes
//! are re-validated post-upload so a corrupted MlxBuffer state fails fast.
//!
//! ## What this module does NOT do
//!
//! - **Embed_tokens / lm_head**: drafter shares these with the target
//!   via `bind()` per `model_mlx.py:153-168`. They live on the target
//!   model's [`MlxBuffer`]s and are referenced (not copied) at forward
//!   time.
//! - **Forward math**: the actual SDPA / RMSNorm / RoPE / MLP dispatch
//!   lives in `forward.rs` (Phase 2 next piece).
//! - **KV cache allocation**: lives in `kv_cache.rs` (Phase 2 next-next
//!   piece) — caches are sized at first-forward-time, not at weight-load
//!   time.

use super::config::DFlashConfig;
use super::weights::{DFlashWeights, WeightsError};
use mlx_native::{DType, MlxBuffer, MlxDevice, MlxError};
use safetensors::tensor::TensorView;

#[derive(Debug, thiserror::Error)]
pub enum TensorsError {
    #[error("dflash tensors mlx: {0}")]
    Mlx(#[from] MlxError),
    #[error("dflash tensors weights: {0}")]
    Weights(#[from] WeightsError),
    #[error("dflash tensors: missing manifest entry `{0}`")]
    MissingEntry(String),
}

/// Per-layer GPU-resident weights for one DFlash decoder layer.
///
/// Mirrors the Python `DFlashDecoderLayer` constituents
/// (`model_mlx.py:119-129`): one [`DFlashAttention`] + one `nn.MLP`
/// (qwen3-style SwiGLU) + two RMSNorms.
///
/// ## DType policy
///
/// Most weights stay BF16 (the safetensors-on-disk dtype) for memory
/// efficiency — they are consumed by `dispatch_dense_mm_bf16`-class
/// matmul kernels that operate on BF16 weights + F32 activations.
///
/// **All RMSNorm weights are stored as F32** (ADR-030 iter-106): `q_norm`,
/// `k_norm`, `input_layernorm`, `post_attention_layernorm`, model-level
/// `hidden_norm`, and final `norm`. mlx-native's `dispatch_rms_norm`
/// kernel selects pipeline by INPUT dtype only; for F32 input it
/// dispatches `rms_norm_f32` which declares `device const float*
/// weight [[buffer(1)]]` (`shaders/rms_norm.metal:20`).  Passing a
/// BF16-backed weight buffer to that kernel causes the kernel to
/// read F32 words over the BF16 byte layout — values become bit-
/// misinterpreted combinations of adjacent BF16 elements AND reads
/// past the buffer end (BF16 buffer is dim*2 bytes; F32 kernel reads
/// dim*4 bytes).  Matching qwen35's convention (`forward_gpu.rs:332`
/// uploaded via `upload_f32_weight`).  Memory cost is small: all
/// drafter norms total 5 × (2 head + 2 layer) + 2 model = 14 RMSNorm
/// vectors, each at most `hidden_size = 2816` floats → ~158 KB.
pub struct DFlashLayerTensors {
    /// `[hidden_size]` **F32** — applied before attention.
    /// Cast from BF16 at upload time (ADR-030 iter-106).
    pub input_layernorm: MlxBuffer,
    /// `[hidden_size]` **F32** — applied between attention residual and MLP.
    /// Cast from BF16 at upload time (ADR-030 iter-106).
    pub post_attention_layernorm: MlxBuffer,
    /// `[num_q_heads * head_dim, hidden_size]` BF16 — Q projection.
    pub q_proj: MlxBuffer,
    /// `[num_kv_heads * head_dim, hidden_size]` BF16 — K projection.
    pub k_proj: MlxBuffer,
    /// `[num_kv_heads * head_dim, hidden_size]` BF16 — V projection.
    pub v_proj: MlxBuffer,
    /// `[hidden_size, num_q_heads * head_dim]` BF16 — O projection.
    pub o_proj: MlxBuffer,
    /// `[head_dim]` **F32** — per-head RMSNorm on Q (qwen3-style).
    /// Cast from BF16 at upload time for mlx-native rms_norm-f32 path.
    pub q_norm: MlxBuffer,
    /// `[head_dim]` **F32** — per-head RMSNorm on K (qwen3-style).
    /// Cast from BF16 at upload time.
    pub k_norm: MlxBuffer,
    /// `[intermediate_size, hidden_size]` BF16 — SwiGLU gate projection.
    pub mlp_gate: MlxBuffer,
    /// `[intermediate_size, hidden_size]` BF16 — SwiGLU up projection.
    pub mlp_up: MlxBuffer,
    /// `[hidden_size, intermediate_size]` BF16 — SwiGLU down projection.
    pub mlp_down: MlxBuffer,
}

/// Full drafter model GPU weights: globals + per-layer tensors.
pub struct DFlashModelTensors {
    /// `[hidden_size, num_target_layers_used * hidden_size]` — projects
    /// concatenated target hidden states into draft hidden space.
    pub fc: MlxBuffer,
    /// `[hidden_size]` **F32** — RMSNorm applied to `fc` output (h_ctx).
    /// Cast from BF16 at upload time (ADR-030 iter-106).
    pub hidden_norm: MlxBuffer,
    /// `[hidden_size]` **F32** — final RMSNorm before lm_head.
    /// Cast from BF16 at upload time (ADR-030 iter-106).
    pub final_norm: MlxBuffer,
    /// One per draft layer (5 for gemma-4-26B-A4B-it drafter).
    pub layers: Vec<DFlashLayerTensors>,
}

/// Upload a single BF16 [`TensorView`] to a fresh GPU [`MlxBuffer`].
fn upload_bf16(device: &MlxDevice, view: &TensorView<'_>) -> Result<MlxBuffer, TensorsError> {
    let shape: Vec<usize> = view.shape().to_vec();
    let byte_len = view.data().len();
    let mut buf = device.alloc_buffer(byte_len, DType::BF16, shape)?;
    // SAFETY: alloc_buffer returns a fresh zeroed buffer; we have exclusive
    // access (no dispatch references it yet). `as_mut_slice::<u8>()` over
    // BF16 storage is well-defined (any byte pattern is a valid u8).
    let dst: &mut [u8] = buf
        .as_mut_slice::<u8>()
        .map_err(|e| TensorsError::Mlx(MlxError::InvalidArgument(format!("buffer slice: {e}"))))?;
    debug_assert_eq!(dst.len(), byte_len);
    dst.copy_from_slice(view.data());
    Ok(buf)
}

/// Decode little-endian BF16 bytes into F32 values via bit-shift
/// reconstruction.
///
/// BF16 = top 16 bits of an F32 (matching exponent range, truncated
/// mantissa).  Reconstruction is `f32_bits = (bf16_bits as u32) << 16`
/// (the dropped low 16 mantissa bits become zero).
///
/// Factored out of [`upload_bf16_as_f32`] so the bit math is unit-
/// testable without a Metal device (ADR-030 iter-108).  Catches future
/// regressions in the iter-106 cast path.
///
/// Returns an error if `bytes.len()` is odd (not BF16-aligned).
pub(super) fn decode_bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, TensorsError> {
    let n_elem = bytes.len() / 2;
    if bytes.len() != n_elem * 2 {
        return Err(TensorsError::Mlx(MlxError::InvalidArgument(format!(
            "decode_bf16_bytes_to_f32: data len {} not even (not BF16-aligned)",
            bytes.len()
        ))));
    }
    let mut out = Vec::with_capacity(n_elem);
    for i in 0..n_elem {
        let lo = bytes[i * 2] as u32;
        let hi = bytes[i * 2 + 1] as u32;
        let bf16_bits = lo | (hi << 8);
        let f32_bits = bf16_bits << 16;
        out.push(f32::from_bits(f32_bits));
    }
    Ok(out)
}

/// Convert a BF16 [`TensorView`] to F32 and upload to a fresh GPU
/// [`MlxBuffer`]. Used for the RMSNorm weights (q_norm, k_norm,
/// input_layernorm, post_attention_layernorm, hidden_norm, final_norm
/// — see ADR-030 iter-106) where mlx-native's `rms_norm_f32` kernel
/// declares `device const float*` for the weight buffer and requires
/// the weight dtype to match the F32 input.
///
/// Bit decoding factored into [`decode_bf16_bytes_to_f32`] for
/// device-free unit testability (ADR-030 iter-108).
fn upload_bf16_as_f32(device: &MlxDevice, view: &TensorView<'_>) -> Result<MlxBuffer, TensorsError> {
    let f32_values = decode_bf16_bytes_to_f32(view.data())?;
    let n_elem = f32_values.len();
    let shape: Vec<usize> = view.shape().to_vec();
    let mut buf = device.alloc_buffer(n_elem * 4, DType::F32, shape)?;
    let dst: &mut [f32] = buf
        .as_mut_slice::<f32>()
        .map_err(|e| TensorsError::Mlx(MlxError::InvalidArgument(format!("f32 slice: {e}"))))?;
    debug_assert_eq!(dst.len(), n_elem);
    dst.copy_from_slice(&f32_values);
    Ok(buf)
}

impl DFlashModelTensors {
    /// Upload all 58 drafter tensors to the GPU. The order of upload
    /// matches `DFlashWeights::manifest` (stable). Total upload cost:
    /// 58 × `copy_from_slice` over BF16 bytes (~820 MB for gemma-4-26B-A4B-it).
    ///
    /// After this returns, the drafter is ready for forward — embed_tokens
    /// and lm_head are NOT loaded (shared from target at forward time).
    pub fn upload(
        device: &MlxDevice,
        cfg: &DFlashConfig,
        weights: &DFlashWeights<'_>,
    ) -> Result<Self, TensorsError> {
        // Globals
        let fc = upload_bf16(device, fetch(weights, "fc.weight")?)?;
        // ADR-030 iter-106 — hidden_norm + final_norm uploaded as F32.
        // dispatch_rms_norm selects pipeline by INPUT dtype (F32 here);
        // rms_norm_f32 declares weight as `device const float*` so a BF16
        // buffer would be bit-misinterpreted + out-of-bounds-read.
        let hidden_norm = upload_bf16_as_f32(device, fetch(weights, "hidden_norm.weight")?)?;
        let final_norm = upload_bf16_as_f32(device, fetch(weights, "norm.weight")?)?;

        // Per-layer
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("layers.{i}");
            let lw = DFlashLayerTensors {
                // ADR-030 iter-106 — see DFlashLayerTensors docstring.
                // F32 cast at upload mirrors qwen35 forward_gpu.rs:332.
                input_layernorm: upload_bf16_as_f32(
                    device,
                    fetch(weights, &format!("{p}.input_layernorm.weight"))?,
                )?,
                post_attention_layernorm: upload_bf16_as_f32(
                    device,
                    fetch(weights, &format!("{p}.post_attention_layernorm.weight"))?,
                )?,
                q_proj: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.self_attn.q_proj.weight"))?,
                )?,
                k_proj: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.self_attn.k_proj.weight"))?,
                )?,
                v_proj: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.self_attn.v_proj.weight"))?,
                )?,
                o_proj: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.self_attn.o_proj.weight"))?,
                )?,
                q_norm: upload_bf16_as_f32(
                    device,
                    fetch(weights, &format!("{p}.self_attn.q_norm.weight"))?,
                )?,
                k_norm: upload_bf16_as_f32(
                    device,
                    fetch(weights, &format!("{p}.self_attn.k_norm.weight"))?,
                )?,
                mlp_gate: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.mlp.gate_proj.weight"))?,
                )?,
                mlp_up: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.mlp.up_proj.weight"))?,
                )?,
                mlp_down: upload_bf16(
                    device,
                    fetch(weights, &format!("{p}.mlp.down_proj.weight"))?,
                )?,
            };
            layers.push(lw);
        }

        // ADR-030 iter-107 — defensive dtype-invariant guard.  The
        // iter-106 bug (rms_norm_f32 reading F32 over BF16-allocated
        // weight buffer → bit-misinterpretation + OOB reads) silently
        // produced corrupted hidden states.  These asserts catch future
        // regressions at upload time instead of letting them surface
        // as opaque drafter-clustering symptoms downstream.
        debug_assert_eq!(fc.dtype(), DType::BF16, "fc weight must stay BF16 for dense_matmul");
        debug_assert_eq!(hidden_norm.dtype(), DType::F32, "hidden_norm weight must be F32 for rms_norm_f32 kernel");
        debug_assert_eq!(final_norm.dtype(), DType::F32, "final_norm weight must be F32 for rms_norm_f32 kernel");
        for (idx, l) in layers.iter().enumerate() {
            debug_assert_eq!(l.input_layernorm.dtype(), DType::F32,
                "layer {idx}: input_layernorm weight must be F32 for rms_norm_f32 kernel");
            debug_assert_eq!(l.post_attention_layernorm.dtype(), DType::F32,
                "layer {idx}: post_attention_layernorm weight must be F32 for rms_norm_f32 kernel");
            debug_assert_eq!(l.q_norm.dtype(), DType::F32,
                "layer {idx}: q_norm weight must be F32 for rms_norm_f32 head_norm kernel");
            debug_assert_eq!(l.k_norm.dtype(), DType::F32,
                "layer {idx}: k_norm weight must be F32 for rms_norm_f32 head_norm kernel");
            debug_assert_eq!(l.q_proj.dtype(), DType::BF16, "layer {idx}: q_proj must stay BF16 for dense_matmul");
            debug_assert_eq!(l.k_proj.dtype(), DType::BF16, "layer {idx}: k_proj must stay BF16 for dense_matmul");
            debug_assert_eq!(l.v_proj.dtype(), DType::BF16, "layer {idx}: v_proj must stay BF16 for dense_matmul");
            debug_assert_eq!(l.o_proj.dtype(), DType::BF16, "layer {idx}: o_proj must stay BF16 for dense_matmul");
            debug_assert_eq!(l.mlp_gate.dtype(), DType::BF16, "layer {idx}: mlp_gate must stay BF16 for dense_matmul");
            debug_assert_eq!(l.mlp_up.dtype(), DType::BF16, "layer {idx}: mlp_up must stay BF16 for dense_matmul");
            debug_assert_eq!(l.mlp_down.dtype(), DType::BF16, "layer {idx}: mlp_down must stay BF16 for dense_matmul");
        }

        Ok(DFlashModelTensors {
            fc,
            hidden_norm,
            final_norm,
            layers,
        })
    }

    /// Total bytes resident on GPU after upload. Used for memory
    /// accounting / regression tests.
    pub fn gpu_resident_bytes(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(|l| {
                l.input_layernorm.byte_len()
                    + l.post_attention_layernorm.byte_len()
                    + l.q_proj.byte_len()
                    + l.k_proj.byte_len()
                    + l.v_proj.byte_len()
                    + l.o_proj.byte_len()
                    + l.q_norm.byte_len()
                    + l.k_norm.byte_len()
                    + l.mlp_gate.byte_len()
                    + l.mlp_up.byte_len()
                    + l.mlp_down.byte_len()
            })
            .sum();
        self.fc.byte_len() + self.hidden_norm.byte_len() + self.final_norm.byte_len() + layer_bytes
    }
}

fn fetch<'a, 'b>(
    weights: &'a DFlashWeights<'b>,
    name: &str,
) -> Result<&'a TensorView<'b>, TensorsError> {
    weights
        .tensor(name)
        .ok_or_else(|| TensorsError::MissingEntry(name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::dflash::config::DFlashConfig;
    use crate::inference::spec_decode::dflash::weights::{DFlashWeights, DFlashWeightsFile};

    fn gemma4_26b_a4b_dflash_config() -> DFlashConfig {
        DFlashConfig::from_json_str(
            super::super::config::tests::GEMMA4_26B_A4B_DFLASH_CONFIG,
        )
        .expect("test fixture must parse")
    }

    /// Build a BF16 byte buffer (little-endian) from a F32 value by
    /// truncating the F32 to its top 16 bits.  Round-trip helper for
    /// the tests below.
    fn bf16_bytes_from_f32(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for v in values {
            let bits = v.to_bits();
            // BF16 = top 16 bits.  Round-to-nearest-even would add half
            // a low-bit of the BF16 mantissa range, but for canonical
            // test values (1.0, -1.0, 0.0, finite powers of two) the
            // truncated form IS the round-to-nearest representative
            // because the low 16 bits are already zero.
            let bf16_bits = (bits >> 16) as u16;
            out.push((bf16_bits & 0xff) as u8);
            out.push(((bf16_bits >> 8) & 0xff) as u8);
        }
        out
    }

    /// ADR-030 iter-108 — locks down the bit-decode path that the iter-106
    /// fix relies on.  Constructs synthetic BF16 byte buffers with known
    /// canonical values and asserts the F32 round-trip via
    /// `decode_bf16_bytes_to_f32` produces the originating F32 values
    /// bit-exactly.
    #[test]
    fn decode_bf16_bytes_round_trips_canonical_values() {
        // Values whose F32 form has all-zero low 16 bits: these survive
        // BF16 truncation losslessly.
        let canonical = [0.0f32, 1.0, -1.0, 2.0, 0.5, -0.5, 256.0, -256.0];
        let bytes = bf16_bytes_from_f32(&canonical);
        let decoded = decode_bf16_bytes_to_f32(&bytes).expect("decode");
        assert_eq!(decoded.len(), canonical.len());
        for (i, (got, want)) in decoded.iter().zip(canonical.iter()).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(),
                "canonical[{i}] = {want} round-trip got {got}");
        }
    }

    /// Verify the OOB / corruption signature that iter-106 fixed.
    /// Reading F32 over BF16 bytes (the pre-fix bug) would read 8 BF16
    /// elements as 4 F32 elements with mangled bits.  This test
    /// constructs a small BF16 buffer + reads the FIRST 4 F32 words from
    /// it; the result should NOT equal the BF16 values themselves.
    #[test]
    fn bf16_over_f32_misinterpret_signature() {
        let canonical = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let bf16_bytes = bf16_bytes_from_f32(&canonical);
        assert_eq!(bf16_bytes.len(), canonical.len() * 2);
        // BF16 1.0 bytes are [0x80, 0x3f] little-endian.
        assert_eq!(&bf16_bytes[0..2], &[0x80, 0x3f]);

        // The PRE-FIX bug: rms_norm_f32 kernel reads F32 over this BF16
        // buffer.  Simulate by reinterpreting 4 bytes at a time as F32.
        let mut misread = Vec::with_capacity(canonical.len() / 2);
        for chunk in bf16_bytes.chunks_exact(4) {
            let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            misread.push(f32::from_bits(bits));
        }
        // Two BF16 1.0 bytes [0x80 0x3f 0x80 0x3f] reinterpret as
        // F32 bits 0x3f803f80 — a value close to 1.0 but NOT equal
        // (mantissa contains the second BF16's bits).
        assert_eq!(misread.len(), 4);
        let bug_value = f32::from_bits(0x3f803f80);
        assert_eq!(misread[0].to_bits(), bug_value.to_bits(),
            "pre-iter-106 bug: reads BF16 1.0 + 1.0 as F32 ≈ {} (NOT 1.0)",
            bug_value);
        assert_ne!(misread[0], 1.0,
            "this is the silent-corruption signature iter-106 fixed");
    }

    #[test]
    fn decode_bf16_bytes_rejects_odd_length() {
        let bytes = vec![0x80, 0x3f, 0x00]; // 3 bytes — odd
        let err = decode_bf16_bytes_to_f32(&bytes).unwrap_err();
        assert!(format!("{err}").contains("not even"));
    }

    /// Integration test: upload all 58 drafter tensors to GPU and
    /// verify the resident byte total matches the safetensors data
    /// total plus the F32 expansion for the cast RMSNorm tensors
    /// (ADR-030 iter-106).
    ///
    /// Requires both a Metal device AND the cached drafter download.
    #[test]
    #[ignore = "requires Metal device + drafter HF cache"]
    fn uploads_real_drafter_to_gpu() {
        let cfg = gemma4_26b_a4b_dflash_config();
        let device = MlxDevice::new().expect("Metal device available on M5 Max");
        let home = std::env::var("HOME").expect("HOME set");
        let path = format!(
            "{home}/.cache/huggingface/hub/models--z-lab--gemma-4-26B-A4B-it-DFlash/snapshots/77d4202772dfe50b2396ec7bac9cfffc7b9e7057/model.safetensors"
        );
        let file = DFlashWeightsFile::open(&path).expect("file open");
        let weights = DFlashWeights::load(file.bytes(), &cfg).expect("validated load");
        let safetensors_data_bytes = weights.total_data_bytes();

        let tensors = DFlashModelTensors::upload(&device, &cfg, &weights).expect("GPU upload");
        let gpu_bytes = tensors.gpu_resident_bytes();

        // ADR-030 iter-106: RMSNorm weights are BF16 on disk but uploaded
        // as F32 (2× expansion).  Per-layer F32-cast norms: q_norm +
        // k_norm (head_dim each) + input_layernorm + post_attention_layernorm
        // (hidden_size each).  Model-level: hidden_norm + final_norm
        // (hidden_size each).  Each cast adds `n_elem * 2` bytes vs
        // straight BF16 copy.
        let per_layer_cast_elems = 2 * cfg.head_dim + 2 * cfg.hidden_size;
        let model_cast_elems = 2 * cfg.hidden_size;
        let cast_expansion_bytes =
            (cfg.num_hidden_layers * per_layer_cast_elems + model_cast_elems) * 2;
        assert_eq!(
            gpu_bytes,
            safetensors_data_bytes + cast_expansion_bytes,
            "GPU resident bytes must equal safetensors data + F32 cast expansion \
             (cfg.num_hidden_layers={}, head_dim={}, hidden_size={}, expansion={})",
            cfg.num_hidden_layers, cfg.head_dim, cfg.hidden_size, cast_expansion_bytes,
        );
        assert_eq!(tensors.layers.len(), cfg.num_hidden_layers);
    }
}
