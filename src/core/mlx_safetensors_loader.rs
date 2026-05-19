//! ADR-020 iter-16 — hf2q reader for mlx-lm DWQ-quantized model files.
//!
//! Direct port of mlx-lm's affine-quantized save format
//! (`mlx-lm/mlx_lm/utils.py:save_model` + `mlx/ops.cpp:4762-4798`
//! `affine_quantize`):
//!
//! For each `nn.QuantizedLinear(in=K, out=N, bits=4, group_size=GS,
//! mode="affine")`, the safetensors file contains three tensors per
//! quantized Linear:
//!
//! | tensor          | dtype                        | shape       |
//! |-----------------|------------------------------|-------------|
//! | `<path>.weight` | `U32`                        | `[N, K/8]`  |
//! | `<path>.scales` | original weight dtype (BF16/F16/F32) | `[N, K/GS]` |
//! | `<path>.biases` | same as scales               | `[N, K/GS]` |
//!
//! The `K/8` packing factor at bits=4: each u32 holds 8 nibbles,
//! element `i` (0..7 within one u32) at bits `[i*bits, (i+1)*bits)`.
//! The lowest-index element occupies the LOW bits of the uint32
//! (`mlx/ops.cpp:4762-4772` left_shift loop).  This matches the GPU
//! dequant at `mlx/backend/metal/kernels/quantized.h:521-526` (`b &
//! 0x0f` = even byte index, `(b & 0xf0) >> 4` = odd byte index).
//!
//! `config.json` contains `quantization.{bits, group_size, mode}` and
//! a duplicate `quantization_config.{bits, group_size, mode}` per
//! `mlx-lm/mlx_lm/utils.py:813-846`.  Per-layer overrides may be
//! present as `quantization.{path}: bool | dict`
//! (`mlx-lm/mlx_lm/utils.py:351-352`).
//!
//! For integration with iter-13b / iter-15 / iter-15b, this loader
//! UNPACKS the u32-packed weight into one-byte-per-code uint8 layout
//! (the iter-13b layout) and CASTS scales/biases to f32 (the kernel
//! input convention).  Bit-packed read kernels (matching mlx's GPU
//! dequant directly without a host-side unpack pass) are deferred to
//! iter-16b — once iter-15 / iter-15b are validated against the
//! unpacked format.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use safetensors::{tensor::Dtype, SafeTensors};
use serde::{Deserialize, Serialize};

/// Top-level mlx-lm quantization config (read from `config.json`).
///
/// Captures both the global default and any per-layer overrides.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MlxQuantConfig {
    /// Default bits-per-weight (`config.quantization.bits`).
    pub bits: u32,
    /// Default group size (`config.quantization.group_size`).
    pub group_size: u32,
    /// Quantization mode — must equal `"affine"` for this loader.
    pub mode: String,
    /// Per-path overrides.  Value is either `bool` (to enable/disable
    /// quantization for that path against the global default) or a
    /// dict of param overrides — represented here as a flexible JSON
    /// blob so future modes can ride along.
    pub per_path: BTreeMap<String, serde_json::Value>,
}

impl MlxQuantConfig {
    /// Parse `config.json` from a model directory and extract the
    /// quantization metadata.  Returns `Ok(None)` if the model has no
    /// quantization key (i.e. unquantized FP model).
    pub fn from_config_json(model_dir: &Path) -> Result<Option<Self>> {
        let path = model_dir.join("config.json");
        if !path.exists() {
            return Err(anyhow!(
                "MlxQuantConfig: config.json not found at {}",
                path.display()
            ));
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse json {}", path.display()))?;

        // Prefer "quantization", fall back to "quantization_config"
        // (both are written identically per mlx_lm/utils.py:914-915,
        // but legacy models may have only one).
        let q = v.get("quantization").or_else(|| v.get("quantization_config"));
        let Some(q) = q else { return Ok(None); };
        let q_obj = q
            .as_object()
            .ok_or_else(|| anyhow!("'quantization' is not a JSON object: {q}"))?;

        let bits = q_obj
            .get("bits")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing/invalid 'bits' in quantization config"))?
            as u32;
        let group_size = q_obj
            .get("group_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("missing/invalid 'group_size' in quantization config"))?
            as u32;
        let mode = q_obj
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("affine")
            .to_string();

        // Per-path overrides: any key in q_obj that is NOT one of the
        // global keys is a path override.
        const GLOBAL_KEYS: &[&str] = &["bits", "group_size", "mode"];
        let mut per_path = BTreeMap::new();
        for (k, v) in q_obj {
            if GLOBAL_KEYS.contains(&k.as_str()) {
                continue;
            }
            per_path.insert(k.clone(), v.clone());
        }

        Ok(Some(Self {
            bits,
            group_size,
            mode,
            per_path,
        }))
    }
}

/// One affine-quantized Linear loaded into hf2q's runtime layout.
///
/// `q_int` is UNPACKED (one byte per code, value in `[0, 2^bits)`)
/// to match iter-13b / iter-15 / iter-15b.  `scales` and `biases`
/// are cast to f32 regardless of source dtype.  `n` is the output
/// dim; `k` is the input dim; `group_size` is the per-group axis
/// length along K.
#[derive(Debug, Clone)]
pub struct MlxAffineLinear {
    pub n: usize,
    pub k: usize,
    pub group_size: usize,
    pub bits: u32,
    pub q_int: Vec<u8>,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
}

impl MlxAffineLinear {
    /// Load one affine-quantized Linear from a safetensors view.
    /// Caller passes the stem `path` (e.g. `"model.layers.0.self_attn.q_proj"`)
    /// and the loader pulls `<path>.weight`, `<path>.scales`,
    /// `<path>.biases` from the same `SafeTensors` instance.
    pub fn from_safetensors(
        st: &SafeTensors,
        path: &str,
        bits: u32,
        group_size: usize,
    ) -> Result<Self> {
        if !(bits == 4 || bits == 8) {
            return Err(anyhow!(
                "MlxAffineLinear: only bits=4 and bits=8 supported in iter-16; got {bits}"
            ));
        }
        let weight_key = format!("{path}.weight");
        let scales_key = format!("{path}.scales");
        let biases_key = format!("{path}.biases");

        let w_t = st
            .tensor(&weight_key)
            .map_err(|e| anyhow!("missing {weight_key}: {e:?}"))?;
        let s_t = st
            .tensor(&scales_key)
            .map_err(|e| anyhow!("missing {scales_key}: {e:?}"))?;
        let b_t = st
            .tensor(&biases_key)
            .map_err(|e| anyhow!("missing {biases_key}: {e:?}"))?;

        if w_t.dtype() != Dtype::U32 {
            return Err(anyhow!(
                "MlxAffineLinear: {weight_key} dtype {:?} != U32 (mlx affine packed weight is uint32)",
                w_t.dtype()
            ));
        }
        let w_shape = w_t.shape();
        if w_shape.len() != 2 {
            return Err(anyhow!(
                "MlxAffineLinear: {weight_key} shape {:?} not 2-D",
                w_shape
            ));
        }
        let n = w_shape[0];
        let pack_factor = 32 / bits as usize; // 8 for bits=4, 4 for bits=8
        let k = w_shape[1] * pack_factor;
        let groups_per_row = k / group_size;

        // Validate scales/biases shapes.
        for (label, t) in [("scales", &s_t), ("biases", &b_t)] {
            let sh = t.shape();
            if sh.len() != 2 || sh[0] != n || sh[1] != groups_per_row {
                return Err(anyhow!(
                    "MlxAffineLinear: {label} shape {:?} != [{}, {}] (n, k/group_size)",
                    sh,
                    n,
                    groups_per_row
                ));
            }
        }

        let q_int = unpack_u32_packed(w_t.data(), n * w_shape[1], bits)?;
        debug_assert_eq!(q_int.len(), n * k);
        let scales = read_floats_to_f32(s_t.data(), s_t.dtype())?;
        let biases = read_floats_to_f32(b_t.data(), b_t.dtype())?;
        if scales.len() != n * groups_per_row {
            return Err(anyhow!(
                "MlxAffineLinear: scales element count {} != {}",
                scales.len(),
                n * groups_per_row
            ));
        }
        if biases.len() != n * groups_per_row {
            return Err(anyhow!(
                "MlxAffineLinear: biases element count {} != {}",
                biases.len(),
                n * groups_per_row
            ));
        }

        Ok(Self {
            n,
            k,
            group_size,
            bits,
            q_int,
            scales,
            biases,
        })
    }

    /// Dequantize the affine-DWQ Linear back to row-major flat F32:
    ///
    ///   `w[i * k + j] = scales[i, j / group_size] * q_int[i, j] as f32
    ///                 + biases[i, j / group_size]`
    ///
    /// Returns a `Vec<f32>` of length `n * k`.  Pure CPU computation;
    /// no GPU.  Used by the AC#7 round-trip drift benchmark
    /// (`q4_0_round_trip_drift`) and by `Qwen35Model::apply_dwq_overlay`'s
    /// lm_head Vec<f32> overwrite path.
    pub fn dequantize_to_f32(&self) -> Vec<f32> {
        let groups_per_row = self.k / self.group_size;
        let mut w = vec![0f32; self.n * self.k];
        for i in 0..self.n {
            for j in 0..self.k {
                let g = j / self.group_size;
                let scale = self.scales[i * groups_per_row + g];
                let bias  = self.biases[i * groups_per_row + g];
                w[i * self.k + j] = scale * (self.q_int[i * self.k + j] as f32) + bias;
            }
        }
        w
    }

    /// ADR-020 AC#7 foundation F2 — measure drift introduced by routing
    /// this DWQ-trained Linear's dequantized values through hf2q's
    /// production Q4_0 codec (`crate::quantize::q_legacy::quantize_row_q4_0_to_bytes`
    /// + `dequantize_row_q4_0_bytes`).
    ///
    /// This is the lossiness incurred by `Qwen35Model::apply_dwq_overlay`'s
    /// iter-B1 lm_head path: the trained `(s, b, q_int)` are dequantized
    /// to a row of f32, which is later re-quantized to Q4_0 by
    /// `forward_gpu.rs::ensure_gpu_cache_primed::upload_q4_0_from_f32`.
    /// Q4_0 has no per-group bias term, so any non-zero `biases[g]`
    /// in the DWQ output is necessarily LOST in the round-trip — this
    /// helper quantifies how much drift that introduces in absolute
    /// element-space, per row.
    ///
    /// See `RoundTripDrift` for the metrics returned.
    ///
    /// Errors if `k` is not a multiple of the Q4_0 block size (32) —
    /// the same alignment requirement that `quantize_row_q4_0` enforces.
    pub fn q4_0_round_trip_drift(&self) -> Result<RoundTripDrift> {
        use crate::quantize::ggml_quants::q4_0::{quantize as q4_0_quantize, BLOCK_BYTES as Q4_0_BLOCK_BYTES, QK4_0};
        use half::f16;

        let qk = QK4_0; // 32 — production Q4_0 codec constant.
        if self.k % qk != 0 {
            return Err(anyhow!(
                "q4_0_round_trip_drift: k={} not a multiple of Q4_0 block size {}",
                self.k, qk
            ));
        }

        let w_dwq = self.dequantize_to_f32();
        let mut max_abs_drift = 0f32;
        let mut sum_abs_drift = 0.0f64;
        let mut sum_sq_drift  = 0.0f64;
        let mut sum_sq_signal = 0.0f64;
        let total_elems = self.n * self.k;

        // Per-row Q4_0 round-trip — production codec is row-aligned to
        // QK4_0.  For each row of length k we quantize → bytes → dequant
        // and accumulate drift. The dequantize is a trivial unpack of the
        // 18-byte block (2-byte f16 scale + 16-byte nibble payload), so
        // we inline it here rather than re-introducing a legacy helper.
        let mut w_rt_row = vec![0f32; self.k];
        for i in 0..self.n {
            let row = &w_dwq[i * self.k..(i + 1) * self.k];
            let q4_0_bytes = q4_0_quantize(row, self.k, None);
            // Dequantize each block: bytes = [d_lo, d_hi, qs_0..qs_15].
            // For j ∈ [0, qk/2): y[j] = ((qs[j] & 0xF) - 8) × d
            //                   y[j + qk/2] = ((qs[j] >> 4) - 8) × d
            let nb = self.k / qk;
            for b in 0..nb {
                let blk = &q4_0_bytes[b * Q4_0_BLOCK_BYTES..(b + 1) * Q4_0_BLOCK_BYTES];
                let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                let qs = &blk[2..2 + qk / 2];
                for j in 0..(qk / 2) {
                    let x0 = ((qs[j] & 0x0F) as i32 - 8) as f32;
                    let x1 = ((qs[j] >> 4) as i32 - 8) as f32;
                    w_rt_row[b * qk + j] = x0 * d;
                    w_rt_row[b * qk + j + qk / 2] = x1 * d;
                }
            }
            for j in 0..self.k {
                let signal = row[j];
                let drift = (signal - w_rt_row[j]).abs();
                if drift > max_abs_drift { max_abs_drift = drift; }
                sum_abs_drift += drift as f64;
                sum_sq_drift  += (drift as f64) * (drift as f64);
                sum_sq_signal += (signal as f64) * (signal as f64);
            }
        }

        let mean_abs_drift = (sum_abs_drift / total_elems as f64) as f32;
        let rms_drift      = ((sum_sq_drift / total_elems as f64).sqrt()) as f32;
        let rms_signal     = ((sum_sq_signal / total_elems as f64).sqrt()) as f32;
        let relative_rms   = if rms_signal.abs() > f32::EPSILON {
            rms_drift / rms_signal
        } else {
            f32::NAN
        };

        // Bias-explained fraction: how much of the DWQ output's energy
        // comes specifically from the per-group bias term.  If biases
        // are all zero this is 0.0 and Q4_0 round-trip is information-
        // preserving on the (s, code) part.  If biases dominate, the
        // round-trip is going to discard most of that energy.
        let groups_per_row = self.k / self.group_size;
        let mut sum_sq_bias_signal = 0.0f64;
        for i in 0..self.n {
            for g in 0..groups_per_row {
                let b = self.biases[i * groups_per_row + g] as f64;
                sum_sq_bias_signal += b * b * (self.group_size as f64);
            }
        }
        let rms_bias = ((sum_sq_bias_signal / total_elems as f64).sqrt()) as f32;
        let bias_fraction = if rms_signal.abs() > f32::EPSILON {
            rms_bias / rms_signal
        } else {
            f32::NAN
        };

        Ok(RoundTripDrift {
            n: self.n,
            k: self.k,
            group_size: self.group_size,
            bits: self.bits,
            mean_abs_drift,
            max_abs_drift,
            rms_drift,
            rms_signal,
            relative_rms,
            rms_bias,
            bias_fraction,
        })
    }
}

/// ADR-020 AC#7 foundation F2 — drift metrics from
/// [`MlxAffineLinear::q4_0_round_trip_drift`].
///
/// All fields are in element space (f32 absolute drift between the
/// DWQ-dequantized values and the Q4_0-roundtripped values).
///
/// `relative_rms = rms_drift / rms_signal` is the dimensionless
/// quantity to read first:
///   - `< 0.1` ⇒ Q4_0 round-trip is preserving most of the DWQ signal
///   - `~ 0.5` ⇒ half the signal is being destroyed by the round-trip
///   - `~ 1.0+` ⇒ Q4_0 is essentially randomizing the values
///
/// `bias_fraction = rms_bias / rms_signal` quantifies how much of the
/// DWQ output's energy is in the per-group bias term that Q4_0 cannot
/// represent — high `bias_fraction` paired with high `relative_rms`
/// indicates the lm_head Vec<f32> overwrite path is fundamentally
/// inadequate and the iter-B2 affine kernel route is required.
#[derive(Debug, Clone, PartialEq)]
pub struct RoundTripDrift {
    pub n: usize,
    pub k: usize,
    pub group_size: usize,
    pub bits: u32,
    pub mean_abs_drift: f32,
    pub max_abs_drift: f32,
    pub rms_drift: f32,
    pub rms_signal: f32,
    pub relative_rms: f32,
    pub rms_bias: f32,
    pub bias_fraction: f32,
}

/// Unpack a u32-packed buffer of `[..., n_u32]` codes into one byte per
/// code (size `n_u32 * pack_factor` bytes).  Element at flat index
/// `i_u32 * pack_factor + j` (j in 0..pack_factor) reads the j-th
/// `bits`-wide slot of `u32_buf[i_u32]`, with `j=0` at the LOW bits
/// (matches mlx's `left_shift(i*bits)` packing at `mlx/ops.cpp:4762-4772`).
pub fn unpack_u32_packed(bytes: &[u8], n_u32: usize, bits: u32) -> Result<Vec<u8>> {
    if bytes.len() != n_u32 * 4 {
        return Err(anyhow!(
            "unpack_u32_packed: bytes.len()={} != n_u32 * 4 = {}",
            bytes.len(),
            n_u32 * 4
        ));
    }
    if !(bits == 4 || bits == 8) {
        return Err(anyhow!(
            "unpack_u32_packed: bits must be 4 or 8 in iter-16; got {bits}"
        ));
    }
    let pack_factor = 32 / bits as usize;
    let mask: u32 = (1u32 << bits) - 1;
    let mut out = Vec::with_capacity(n_u32 * pack_factor);
    for chunk in bytes.chunks_exact(4) {
        // Safetensors raw bytes are little-endian u32.
        let word = u32::from_le_bytes(chunk.try_into().unwrap());
        for j in 0..pack_factor {
            let v = (word >> (j as u32 * bits)) & mask;
            out.push(v as u8);
        }
    }
    Ok(out)
}

/// Read raw safetensors bytes of `Dtype::{F32, F16, BF16}` into a
/// `Vec<f32>`.  All inputs are little-endian per the safetensors
/// spec.
pub fn read_floats_to_f32(bytes: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(anyhow!(
                    "read_floats_to_f32: F32 byte len {} not a multiple of 4",
                    bytes.len()
                ));
            }
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for c in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
            Ok(out)
        }
        Dtype::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(anyhow!(
                    "read_floats_to_f32: F16 byte len {} not a multiple of 2",
                    bytes.len()
                ));
            }
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for c in bytes.chunks_exact(2) {
                let h = half::f16::from_le_bytes(c.try_into().unwrap());
                out.push(h.to_f32());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            if bytes.len() % 2 != 0 {
                return Err(anyhow!(
                    "read_floats_to_f32: BF16 byte len {} not a multiple of 2",
                    bytes.len()
                ));
            }
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for c in bytes.chunks_exact(2) {
                let bf = half::bf16::from_le_bytes(c.try_into().unwrap());
                out.push(bf.to_f32());
            }
            Ok(out)
        }
        other => Err(anyhow!(
            "read_floats_to_f32: dtype {other:?} not supported (expected F32/F16/BF16)"
        )),
    }
}

/// Pack a flat `[bits]`-wide code stream into mlx's u32-packed
/// little-endian byte layout — inverse of [`unpack_u32_packed`].
///
/// `codes.len() % pack_factor` must be 0, where `pack_factor = 32 /
/// bits` (8 for bits=4, 4 for bits=8).  Element index `j` within a
/// pack group occupies bits `[j*bits, (j+1)*bits)` of the resulting
/// u32, LOW bits first — matches mlx's `left_shift(i*bits)` packing
/// at `mlx/ops.cpp:4762-4772`.  Bytes are emitted little-endian
/// per safetensors raw-tensor convention.
///
/// Returns the byte buffer that the caller can wrap in a
/// `safetensors::tensor::TensorView` of dtype `Dtype::U32`.
pub fn pack_u32_codes(codes: &[u8], bits: u32) -> Result<Vec<u8>> {
    if !(bits == 4 || bits == 8) {
        return Err(anyhow!(
            "pack_u32_codes: bits must be 4 or 8 in iter-16/16b; got {bits}"
        ));
    }
    let pack_factor = (32 / bits) as usize;
    if codes.len() % pack_factor != 0 {
        return Err(anyhow!(
            "pack_u32_codes: codes.len()={} not divisible by pack_factor={}",
            codes.len(),
            pack_factor
        ));
    }
    let mask: u32 = (1u32 << bits) - 1;
    let mut out = Vec::with_capacity(codes.len() / pack_factor * 4);
    for chunk in codes.chunks_exact(pack_factor) {
        let mut word: u32 = 0;
        for (j, &c) in chunk.iter().enumerate() {
            if (c as u32) > mask {
                return Err(anyhow!(
                    "pack_u32_codes: code {} at chunk position {j} exceeds {bits}-bit mask {mask}",
                    c
                ));
            }
            word |= ((c as u32) & mask) << (j as u32 * bits);
        }
        out.extend_from_slice(&word.to_le_bytes());
    }
    Ok(out)
}

/// Cast a `Vec<f32>` to safetensors-ready raw bytes in the requested
/// dtype.  `dtype` must be one of `F32`, `F16`, `BF16` — the same
/// types accepted by [`read_floats_to_f32`].  Bytes are little-endian.
pub fn write_floats_from_f32(values: &[f32], dtype: Dtype) -> Result<Vec<u8>> {
    match dtype {
        Dtype::F32 => Ok(values.iter().flat_map(|f| f.to_le_bytes()).collect()),
        Dtype::F16 => Ok(values
            .iter()
            .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
            .collect()),
        Dtype::BF16 => Ok(values
            .iter()
            .flat_map(|f| half::bf16::from_f32(*f).to_le_bytes())
            .collect()),
        other => Err(anyhow!(
            "write_floats_from_f32: dtype {other:?} not supported (expected F32/F16/BF16)"
        )),
    }
}

/// Owned safetensors-ready byte buffers + shape/dtype metadata for
/// one mlx affine-quantized Linear.  Caller wraps these into
/// `TensorView`s via [`MlxAffineLinear::to_safetensors_views`] and
/// passes them to `safetensors::serialize`.  The owned buffers must
/// outlive the views.
pub struct MlxAffineLinearBytes {
    pub weight: Vec<u8>,
    pub scales: Vec<u8>,
    pub biases: Vec<u8>,
    pub n: usize,
    pub k_packed: usize, // K * bits / 32  — last dim of weight
    pub n_groups: usize, // K / group_size — last dim of scales/biases
    pub float_dtype: Dtype,
}

impl MlxAffineLinear {
    /// Serialize this Linear's `(weight, scales, biases)` triplet
    /// into safetensors-ready owned byte buffers + shape metadata.
    /// `float_dtype` controls the on-disk dtype of `scales` and
    /// `biases` — mlx-lm typically saves at `BF16`
    /// (`mlx-lm/quant/dwq.py:78` `dtype: mx.Dtype = mx.bfloat16`).
    ///
    /// Caller wraps the resulting `MlxAffineLinearBytes` via
    /// [`MlxAffineLinear::to_safetensors_views`] (or directly
    /// constructs `TensorView`s) and concatenates with views from
    /// other layers before calling `safetensors::serialize`.
    pub fn to_safetensors_bytes(&self, float_dtype: Dtype) -> Result<MlxAffineLinearBytes> {
        // Validate q_int range against the bit-width.
        let max_code = (1u32 << self.bits) as usize - 1;
        for (i, &c) in self.q_int.iter().enumerate() {
            if c as usize > max_code {
                return Err(anyhow!(
                    "to_safetensors_bytes: q_int[{i}]={c} exceeds {}-bit max {max_code}",
                    self.bits
                ));
            }
        }
        let pack_factor = (32 / self.bits) as usize;
        if self.k % pack_factor != 0 {
            return Err(anyhow!(
                "to_safetensors_bytes: K ({}) must be divisible by pack_factor {pack_factor}",
                self.k
            ));
        }
        let weight = pack_u32_codes(&self.q_int, self.bits)?;
        let n_groups = self.k / self.group_size;
        let scales = write_floats_from_f32(&self.scales, float_dtype)?;
        let biases = write_floats_from_f32(&self.biases, float_dtype)?;
        Ok(MlxAffineLinearBytes {
            weight,
            scales,
            biases,
            n: self.n,
            k_packed: self.k / pack_factor,
            n_groups,
            float_dtype,
        })
    }
}

impl MlxAffineLinearBytes {
    /// Wrap the owned byte buffers into `TensorView`s ready for
    /// `safetensors::serialize`.  The returned views borrow from
    /// `self`; `self` must live until `serialize` returns.
    ///
    /// Returns `(weight_view, scales_view, biases_view)` ready to be
    /// keyed by `<path>.weight`, `<path>.scales`, `<path>.biases` per
    /// mlx-lm's flat-parameter naming convention.
    pub fn to_safetensors_views(
        &self,
    ) -> Result<(
        safetensors::tensor::TensorView<'_>,
        safetensors::tensor::TensorView<'_>,
        safetensors::tensor::TensorView<'_>,
    )> {
        use safetensors::tensor::TensorView;
        let w = TensorView::new(Dtype::U32, vec![self.n, self.k_packed], &self.weight)
            .map_err(|e| anyhow!("weight TensorView: {e:?}"))?;
        let s = TensorView::new(
            self.float_dtype,
            vec![self.n, self.n_groups],
            &self.scales,
        )
        .map_err(|e| anyhow!("scales TensorView: {e:?}"))?;
        let b = TensorView::new(
            self.float_dtype,
            vec![self.n, self.n_groups],
            &self.biases,
        )
        .map_err(|e| anyhow!("biases TensorView: {e:?}"))?;
        Ok((w, s, b))
    }
}

/// Discover the model.safetensors shard files in a directory.
/// Returns the index map (tensor_name → shard_filename) if a sharded
/// `model.safetensors.index.json` is present, or a single `[(*, "model.safetensors")]`
/// equivalent if the model is a single-file save.  Mirrors mlx-lm's
/// `save_model` output convention (utils.py:728-771).
pub fn discover_shards(
    model_dir: &Path,
) -> Result<BTreeMap<String, PathBuf>> {
    let single = model_dir.join("model.safetensors");
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let raw = std::fs::read_to_string(&index_path)
            .with_context(|| format!("read {}", index_path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse {}", index_path.display()))?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| anyhow!("index has no 'weight_map' object"))?;
        let mut out = BTreeMap::new();
        for (k, shard) in map {
            let shard_name = shard
                .as_str()
                .ok_or_else(|| anyhow!("weight_map[{k}] not a string"))?;
            out.insert(k.clone(), model_dir.join(shard_name));
        }
        Ok(out)
    } else if single.exists() {
        // Single-file: caller will deserialize once + iterate; we
        // signal "single file" with an empty map sentinel via an
        // explicit marker below.  Since we don't have the tensor
        // names without parsing the file, just return an empty map
        // and let the caller use the single-file fast path.
        let mut out = BTreeMap::new();
        out.insert("__single__".to_string(), single);
        Ok(out)
    } else {
        Err(anyhow!(
            "discover_shards: neither model.safetensors.index.json nor model.safetensors found in {}",
            model_dir.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    /// Pack a flat Vec<u8> of `bits`-wide codes into the u32-packed
    /// layout that mlx's `affine_quantize` writes.  Inverse of
    /// `unpack_u32_packed`.  Test-internal alias for the public
    /// `pack_u32_codes` function (kept for legacy test references).
    fn pack_u32(codes: &[u8], bits: u32) -> Vec<u8> {
        super::pack_u32_codes(codes, bits).expect("pack_u32_codes")
    }

    #[test]
    fn unpack_u32_packed_round_trip_4bit() {
        // Synthetic codes 0..16 cycling, 24 elements (3 u32 groups
        // of 8 nibbles each).
        let codes: Vec<u8> = (0..24u8).map(|i| i % 16).collect();
        let packed = pack_u32(&codes, 4);
        assert_eq!(packed.len(), 12); // 3 u32 = 12 bytes
        let unpacked = unpack_u32_packed(&packed, 3, 4).unwrap();
        assert_eq!(unpacked, codes);
    }

    #[test]
    fn unpack_u32_packed_round_trip_8bit() {
        // 8-bit: pack_factor = 4, 12 codes → 3 u32.
        let codes: Vec<u8> = (0..12u8).collect();
        let packed = pack_u32(&codes, 8);
        let unpacked = unpack_u32_packed(&packed, 3, 8).unwrap();
        assert_eq!(unpacked, codes);
    }

    #[test]
    fn unpack_u32_low_bits_are_lowest_index() {
        // mlx's pack: element 0 in low 4 bits, element 1 next 4 bits, ...
        // Pack codes [0xA, 0x3, 0x7, 0x1, 0x5, 0xE, 0x2, 0x9] →
        // u32 = 0x9 << 28 | 0x2 << 24 | 0xE << 20 | 0x5 << 16 | 0x1 << 12
        //     | 0x7 <<  8 | 0x3 <<  4 | 0xA <<  0
        let codes: Vec<u8> = vec![0xA, 0x3, 0x7, 0x1, 0x5, 0xE, 0x2, 0x9];
        let packed = pack_u32(&codes, 4);
        // Manual: byte 0 low=0xA high=0x3 → 0x3A; byte 1 low=0x7 high=0x1
        // → 0x17; byte 2 low=0x5 high=0xE → 0xE5; byte 3 low=0x2
        // high=0x9 → 0x92.  u32 = 0x92E5_173A; LE bytes = [0x3A, 0x17,
        // 0xE5, 0x92].
        assert_eq!(packed.len(), 4);
        assert_eq!(packed, vec![0x3Au8, 0x17, 0xE5, 0x92]);
        let unpacked = unpack_u32_packed(&packed, 1, 4).unwrap();
        assert_eq!(unpacked, codes);
    }

    #[test]
    fn read_floats_to_f32_handles_all_three_dtypes() {
        // F32 round-trip
        let f32_in: Vec<f32> = vec![1.5, -0.25, 3.14];
        let f32_bytes: Vec<u8> = f32_in
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        assert_eq!(
            read_floats_to_f32(&f32_bytes, Dtype::F32).unwrap(),
            f32_in
        );

        // F16 round-trip (within f16 precision)
        let f16_vals: Vec<f32> = vec![1.0, -0.5, 2.5];
        let f16_bytes: Vec<u8> = f16_vals
            .iter()
            .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
            .collect();
        let got = read_floats_to_f32(&f16_bytes, Dtype::F16).unwrap();
        for (a, b) in got.iter().zip(f16_vals.iter()) {
            assert!((a - b).abs() < 1e-3);
        }

        // BF16 round-trip
        let bf16_vals: Vec<f32> = vec![0.125, -16.0, 100.0];
        let bf16_bytes: Vec<u8> = bf16_vals
            .iter()
            .flat_map(|f| half::bf16::from_f32(*f).to_le_bytes())
            .collect();
        let got = read_floats_to_f32(&bf16_bytes, Dtype::BF16).unwrap();
        for (a, b) in got.iter().zip(bf16_vals.iter()) {
            assert!((a - b).abs() < 0.5 * b.abs().max(1.0));
        }
    }

    #[test]
    fn mlx_affine_linear_round_trip_synthetic() {
        // Build a synthetic affine-quantized Linear matching mlx's
        // exact on-disk layout: weight U32-packed, scales+biases F32.
        let n = 2usize;
        let k = 16usize; // multiple of group_size + multiple of 8
        let group_size = 8usize;
        let bits = 4u32;
        let groups_per_row = k / group_size;

        let q_int_unpacked: Vec<u8> = (0..(n * k)).map(|i| (i % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.05 + i as f32 * 0.01)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.1 + i as f32 * 0.02)
            .collect();

        // Pack weight per mlx's u32 convention, byte order LE.
        let w_packed = pack_u32(&q_int_unpacked, bits);
        let w_n_u32_per_row = k / 8; // bits=4 → 8 elem/u32
        assert_eq!(w_packed.len(), n * w_n_u32_per_row * 4);

        // F32 byte buffers.
        let scales_bytes: Vec<u8> = scales
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let biases_bytes: Vec<u8> = biases
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // Wrap in safetensors views + serialize.
        let w_view = TensorView::new(
            Dtype::U32,
            vec![n, w_n_u32_per_row],
            &w_packed,
        )
        .unwrap();
        let s_view = TensorView::new(
            Dtype::F32,
            vec![n, groups_per_row],
            &scales_bytes,
        )
        .unwrap();
        let b_view = TensorView::new(
            Dtype::F32,
            vec![n, groups_per_row],
            &biases_bytes,
        )
        .unwrap();
        let bytes = safetensors::tensor::serialize(
            [
                ("model.layers.0.q_proj.weight".to_string(), &w_view),
                ("model.layers.0.q_proj.scales".to_string(), &s_view),
                ("model.layers.0.q_proj.biases".to_string(), &b_view),
            ],
            None,
        )
        .unwrap();

        // Deserialize + read.
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let lin = MlxAffineLinear::from_safetensors(
            &st,
            "model.layers.0.q_proj",
            bits,
            group_size,
        )
        .unwrap();

        assert_eq!(lin.n, n);
        assert_eq!(lin.k, k);
        assert_eq!(lin.group_size, group_size);
        assert_eq!(lin.bits, bits);
        assert_eq!(lin.q_int, q_int_unpacked);
        assert_eq!(lin.scales, scales);
        assert_eq!(lin.biases, biases);
    }

    #[test]
    fn mlx_affine_linear_handles_bf16_scales() {
        // Round-trip with BF16 scales/biases (matches mlx-lm's default
        // `dtype=mx.bfloat16` save).
        let n = 1usize;
        let k = 16usize;
        let group_size = 8usize;
        let bits = 4u32;
        let groups_per_row = k / group_size;

        let q_int_unpacked: Vec<u8> = (0..k).map(|i| (i % 16) as u8).collect();
        let scales_f32: Vec<f32> = vec![0.0625, 0.125]; // representable in bf16
        let biases_f32: Vec<f32> = vec![-0.5, 0.25];

        let w_packed = pack_u32(&q_int_unpacked, bits);
        let scales_bf16_bytes: Vec<u8> = scales_f32
            .iter()
            .flat_map(|f| half::bf16::from_f32(*f).to_le_bytes())
            .collect();
        let biases_bf16_bytes: Vec<u8> = biases_f32
            .iter()
            .flat_map(|f| half::bf16::from_f32(*f).to_le_bytes())
            .collect();

        let w_view = TensorView::new(
            Dtype::U32,
            vec![n, k / 8],
            &w_packed,
        )
        .unwrap();
        let s_view = TensorView::new(
            Dtype::BF16,
            vec![n, groups_per_row],
            &scales_bf16_bytes,
        )
        .unwrap();
        let b_view = TensorView::new(
            Dtype::BF16,
            vec![n, groups_per_row],
            &biases_bf16_bytes,
        )
        .unwrap();

        let bytes = safetensors::tensor::serialize(
            [
                ("layer.weight".to_string(), &w_view),
                ("layer.scales".to_string(), &s_view),
                ("layer.biases".to_string(), &b_view),
            ],
            None,
        )
        .unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let lin = MlxAffineLinear::from_safetensors(&st, "layer", bits, group_size)
            .unwrap();
        assert_eq!(lin.q_int, q_int_unpacked);
        // BF16 has ~7-bit mantissa; round-trip is exact for these
        // chosen powers of 2 and small integers.
        assert_eq!(lin.scales, scales_f32);
        assert_eq!(lin.biases, biases_f32);
    }

    #[test]
    fn mlx_affine_linear_rejects_dtype_mismatch() {
        // Pass an F32 weight where U32 is expected — must reject.
        let bytes_f32 = vec![0u8; 8]; // 2 floats = 8 bytes
        let s_bytes = vec![0u8; 8]; // 2 f32
        let b_bytes = vec![0u8; 8];
        let w_view = TensorView::new(Dtype::F32, vec![1, 2], &bytes_f32).unwrap();
        let s_view = TensorView::new(Dtype::F32, vec![1, 2], &s_bytes).unwrap();
        let bb_view = TensorView::new(Dtype::F32, vec![1, 2], &b_bytes).unwrap();
        let bytes = safetensors::tensor::serialize(
            [
                ("l.weight".to_string(), &w_view),
                ("l.scales".to_string(), &s_view),
                ("l.biases".to_string(), &bb_view),
            ],
            None,
        )
        .unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let res = MlxAffineLinear::from_safetensors(&st, "l", 4, 8);
        assert!(res.is_err());
        let msg = format!("{:?}", res.err().unwrap());
        assert!(msg.contains("U32"), "unexpected error: {msg}");
    }

    #[test]
    fn mlx_quant_config_parses_global_keys() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({
            "vocab_size": 32000,
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine"
            },
            "quantization_config": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine"
            }
        });
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::to_string_pretty(&cfg).unwrap(),
        )
        .unwrap();
        let mq = MlxQuantConfig::from_config_json(tmp.path()).unwrap().unwrap();
        assert_eq!(mq.bits, 4);
        assert_eq!(mq.group_size, 64);
        assert_eq!(mq.mode, "affine");
        assert!(mq.per_path.is_empty());
    }

    #[test]
    fn mlx_quant_config_parses_per_path_overrides() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
                "model.layers.0.lm_head": false,
                "model.layers.0.attention.q_proj": {"bits": 8, "group_size": 64}
            }
        });
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::to_string_pretty(&cfg).unwrap(),
        )
        .unwrap();
        let mq = MlxQuantConfig::from_config_json(tmp.path()).unwrap().unwrap();
        assert_eq!(mq.bits, 4);
        assert_eq!(mq.per_path.len(), 2);
        // bool override: lm_head is excluded from quantization.
        assert_eq!(
            mq.per_path["model.layers.0.lm_head"],
            serde_json::Value::Bool(false)
        );
        // dict override: attention.q_proj uses bits=8.
        let q_obj = &mq.per_path["model.layers.0.attention.q_proj"];
        assert_eq!(q_obj["bits"], 8);
    }

    #[test]
    fn mlx_quant_config_returns_none_for_unquantized_model() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({
            "vocab_size": 32000,
            "hidden_size": 4096
        });
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::to_string_pretty(&cfg).unwrap(),
        )
        .unwrap();
        let mq = MlxQuantConfig::from_config_json(tmp.path()).unwrap();
        assert!(mq.is_none());
    }

    #[test]
    fn discover_shards_handles_index_json() {
        let tmp = tempfile::tempdir().unwrap();
        let idx = serde_json::json!({
            "metadata": {"total_size": 12345},
            "weight_map": {
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors"
            }
        });
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&idx).unwrap(),
        )
        .unwrap();
        let map = discover_shards(tmp.path()).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(
            map["model.layers.0.weight"].file_name().unwrap(),
            "model-00001-of-00002.safetensors"
        );
    }

    #[test]
    fn discover_shards_handles_single_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("model.safetensors"), [0u8; 8]).unwrap();
        let map = discover_shards(tmp.path()).unwrap();
        assert_eq!(map.len(), 1);
        assert!(map.contains_key("__single__"));
    }

    /// iter-16b — writer round-trips with reader for the round-trip
    /// fixture: serialize an MlxAffineLinear via `to_safetensors_bytes`
    /// + `to_safetensors_views`, concatenate via
    /// `safetensors::serialize`, deserialize via the iter-16 reader,
    /// and assert byte-identical recovery of (q_int, scales, biases).
    #[test]
    fn writer_round_trips_with_reader_f32_scales() {
        let n = 4usize;
        let k = 32usize;
        let group_size = 8usize;
        let bits = 4u32;
        let groups_per_row = k / group_size;

        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 11 + 3) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.05 + i as f32 * 0.011)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.13 + i as f32 * 0.029)
            .collect();
        let lin = MlxAffineLinear {
            n,
            k,
            group_size,
            bits,
            q_int: q_int.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
        };

        let bytes_owned = lin.to_safetensors_bytes(Dtype::F32).unwrap();
        let (w_view, s_view, b_view) = bytes_owned.to_safetensors_views().unwrap();
        let bytes = safetensors::tensor::serialize(
            [
                ("layer.weight".to_string(), &w_view),
                ("layer.scales".to_string(), &s_view),
                ("layer.biases".to_string(), &b_view),
            ],
            None,
        )
        .unwrap();

        let st = SafeTensors::deserialize(&bytes).unwrap();
        let lin2 =
            MlxAffineLinear::from_safetensors(&st, "layer", bits, group_size).unwrap();
        assert_eq!(lin2.n, n);
        assert_eq!(lin2.k, k);
        assert_eq!(lin2.group_size, group_size);
        assert_eq!(lin2.bits, bits);
        assert_eq!(lin2.q_int, q_int);
        assert_eq!(lin2.scales, scales);
        assert_eq!(lin2.biases, biases);
    }

    /// iter-16b — round-trip with BF16 scales (matches mlx-lm's
    /// default save dtype).  Scales+biases must be representable in
    /// bf16 (small powers of 2 + small integers) for byte-identical
    /// recovery; otherwise compare within bf16 precision.
    #[test]
    fn writer_round_trips_with_reader_bf16_scales() {
        let n = 2usize;
        let k = 16usize;
        let group_size = 8usize;
        let bits = 4u32;

        let q_int: Vec<u8> = (0..(n * k)).map(|i| (i % 16) as u8).collect();
        // BF16-representable: powers of 2 and small integers.
        let scales: Vec<f32> = vec![0.0625, 0.125, 0.25, 0.5];
        let biases: Vec<f32> = vec![-1.0, 0.5, 2.0, -0.25];
        let lin = MlxAffineLinear {
            n,
            k,
            group_size,
            bits,
            q_int: q_int.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
        };

        let bytes_owned = lin.to_safetensors_bytes(Dtype::BF16).unwrap();
        let (w_view, s_view, b_view) = bytes_owned.to_safetensors_views().unwrap();
        let bytes = safetensors::tensor::serialize(
            [
                ("l.weight".to_string(), &w_view),
                ("l.scales".to_string(), &s_view),
                ("l.biases".to_string(), &b_view),
            ],
            None,
        )
        .unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let lin2 =
            MlxAffineLinear::from_safetensors(&st, "l", bits, group_size).unwrap();
        assert_eq!(lin2.q_int, q_int);
        // BF16-exact for bf16-representable values.
        assert_eq!(lin2.scales, scales);
        assert_eq!(lin2.biases, biases);
    }

    /// iter-16b — pack convention parity: writer must produce the
    /// exact byte layout that mlx's left_shift(i*bits) packing
    /// produces.  Same hand-computed fixture as
    /// `unpack_u32_low_bits_are_lowest_index`.
    #[test]
    fn writer_pack_convention_matches_canonical_fixture() {
        let codes: Vec<u8> = vec![0xA, 0x3, 0x7, 0x1, 0x5, 0xE, 0x2, 0x9];
        let packed = pack_u32_codes(&codes, 4).unwrap();
        // Same canonical bytes as iter-16's reader test:
        //   byte 0 = 0xA | (0x3 << 4) = 0x3A
        //   byte 1 = 0x7 | (0x1 << 4) = 0x17
        //   byte 2 = 0x5 | (0xE << 4) = 0xE5
        //   byte 3 = 0x2 | (0x9 << 4) = 0x92
        assert_eq!(packed, vec![0x3Au8, 0x17, 0xE5, 0x92]);
    }

    /// iter-16b — writer rejects out-of-range codes for the requested
    /// bit-width (catches bugs where an 8-bit code accidentally lands
    /// in a 4-bit Linear).
    #[test]
    fn writer_rejects_out_of_range_codes() {
        let codes = vec![0u8, 1, 2, 3, 4, 5, 6, 16]; // 16 > 4-bit max
        let res = pack_u32_codes(&codes, 4);
        assert!(res.is_err());
        let msg = format!("{:?}", res.err().unwrap());
        assert!(msg.contains("exceeds"), "unexpected error: {msg}");
    }

    /// iter-16b — multi-Linear save+load: verify the writer composes
    /// correctly when multiple Linears are batched into one
    /// safetensors file.  Mirrors mlx-lm's `save_model` flow where
    /// many `<path>.{weight, scales, biases}` triplets share one
    /// safetensors file.
    #[test]
    fn writer_multi_linear_save_load() {
        let group_size = 8usize;
        let bits = 4u32;
        let make_lin = |n: usize, k: usize, seed: u32| -> MlxAffineLinear {
            let groups_per_row = k / group_size;
            let q_int: Vec<u8> = (0..(n * k))
                .map(|i| ((i as u32 * 7 + seed) % 16) as u8)
                .collect();
            let scales: Vec<f32> = (0..(n * groups_per_row))
                .map(|i| 0.04 + (i as f32 + seed as f32) * 0.001)
                .collect();
            let biases: Vec<f32> = (0..(n * groups_per_row))
                .map(|i| -0.05 + (i as f32 + seed as f32) * 0.002)
                .collect();
            MlxAffineLinear {
                n,
                k,
                group_size,
                bits,
                q_int,
                scales,
                biases,
            }
        };
        let lin_q = make_lin(8, 16, 1);
        let lin_k = make_lin(8, 16, 2);
        let lin_v = make_lin(8, 16, 3);

        // Owned buffers must outlive the views (borrow checker
        // enforces this, hence the explicit binding).
        let bytes_q = lin_q.to_safetensors_bytes(Dtype::BF16).unwrap();
        let bytes_k = lin_k.to_safetensors_bytes(Dtype::BF16).unwrap();
        let bytes_v = lin_v.to_safetensors_bytes(Dtype::BF16).unwrap();
        let (qw, qs, qb) = bytes_q.to_safetensors_views().unwrap();
        let (kw, ks, kb) = bytes_k.to_safetensors_views().unwrap();
        let (vw, vs, vb) = bytes_v.to_safetensors_views().unwrap();
        let serialized = safetensors::tensor::serialize(
            [
                ("q_proj.weight".to_string(), &qw),
                ("q_proj.scales".to_string(), &qs),
                ("q_proj.biases".to_string(), &qb),
                ("k_proj.weight".to_string(), &kw),
                ("k_proj.scales".to_string(), &ks),
                ("k_proj.biases".to_string(), &kb),
                ("v_proj.weight".to_string(), &vw),
                ("v_proj.scales".to_string(), &vs),
                ("v_proj.biases".to_string(), &vb),
            ],
            None,
        )
        .unwrap();
        let st = SafeTensors::deserialize(&serialized).unwrap();

        for (name, expected) in &[("q_proj", &lin_q), ("k_proj", &lin_k), ("v_proj", &lin_v)] {
            let got = MlxAffineLinear::from_safetensors(&st, name, bits, group_size).unwrap();
            // q_int is integer-exact regardless of float dtype.
            assert_eq!(got.q_int, expected.q_int);
            // scales/biases: BF16 has ~7-bit mantissa, so non-power-of-2
            // values round-trip with ~0.4% precision loss.  Compare
            // within bf16 precision, not byte-identical.
            for (i, (a, b)) in
                got.scales.iter().zip(expected.scales.iter()).enumerate()
            {
                let tol = 0.01 * b.abs().max(1e-3);
                assert!(
                    (a - b).abs() < tol,
                    "{name}.scales[{i}]: got {a} expected {b} tol {tol}"
                );
            }
            for (i, (a, b)) in
                got.biases.iter().zip(expected.biases.iter()).enumerate()
            {
                let tol = 0.01 * b.abs().max(1e-3);
                assert!(
                    (a - b).abs() < tol,
                    "{name}.biases[{i}]: got {a} expected {b} tol {tol}"
                );
            }
        }
    }

    /// End-to-end iter-15 → iter-16 integration: load an mlx-format
    /// quantized Linear via the loader, run the iter-15 fused
    /// matmul kernel against it, verify the result matches a
    /// hand-computed dequant + matmul oracle.  This proves the
    /// loader's UNPACKED output format is byte-compatible with
    /// iter-15's qmm_affine_t kernel.
    #[test]
    fn loader_output_is_qmm_affine_compatible() {
        use mlx_native::ops::qmm_affine::dispatch_qmm_affine_t_f32;
        use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

        let n = 8usize;
        let k = 32usize;
        let group_size = 8usize;
        let bits = 4u32;
        let groups_per_row = k / group_size;

        // Build a known affine-quantized Linear, save as mlx
        // safetensors, load via iter-16, then run iter-15's kernel
        // and verify against host oracle.
        let q_int: Vec<u8> = (0..(n * k)).map(|i| ((i * 13 + 1) % 16) as u8).collect();
        let scales: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| 0.04 + i as f32 * 0.005)
            .collect();
        let biases: Vec<f32> = (0..(n * groups_per_row))
            .map(|i| -0.05 + i as f32 * 0.011)
            .collect();

        let w_packed = pack_u32(&q_int, bits);
        let scales_bytes: Vec<u8> =
            scales.iter().flat_map(|f| f.to_le_bytes()).collect();
        let biases_bytes: Vec<u8> =
            biases.iter().flat_map(|f| f.to_le_bytes()).collect();

        let w_view = TensorView::new(Dtype::U32, vec![n, k / 8], &w_packed).unwrap();
        let s_view =
            TensorView::new(Dtype::F32, vec![n, groups_per_row], &scales_bytes).unwrap();
        let b_view =
            TensorView::new(Dtype::F32, vec![n, groups_per_row], &biases_bytes).unwrap();
        let bytes = safetensors::tensor::serialize(
            [
                ("l.weight".to_string(), &w_view),
                ("l.scales".to_string(), &s_view),
                ("l.biases".to_string(), &b_view),
            ],
            None,
        )
        .unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let lin =
            MlxAffineLinear::from_safetensors(&st, "l", bits, group_size).unwrap();

        // Build activations + run iter-15 fused kernel.
        let m = 4usize;
        let x: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.013).sin() * 0.4).collect();
        let device = MlxDevice::new().expect("device");
        let mut registry = KernelRegistry::new();

        fn alloc_f32(device: &MlxDevice, n: usize, shape: Vec<usize>) -> MlxBuffer {
            device.alloc_buffer(n * 4, DType::F32, shape).expect("alloc f32")
        }
        fn alloc_u8(device: &MlxDevice, n: usize, shape: Vec<usize>) -> MlxBuffer {
            device.alloc_buffer(n, DType::U8, shape).expect("alloc u8")
        }
        let mut x_buf = alloc_f32(&device, m * k, vec![m, k]);
        x_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&x);
        let mut q_buf = alloc_u8(&device, n * k, vec![n, k]);
        q_buf.as_mut_slice::<u8>().unwrap().copy_from_slice(&lin.q_int);
        let mut s_buf = alloc_f32(&device, lin.scales.len(), vec![n, groups_per_row]);
        s_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&lin.scales);
        let mut b_buf = alloc_f32(&device, lin.biases.len(), vec![n, groups_per_row]);
        b_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&lin.biases);
        let y_buf = alloc_f32(&device, m * n, vec![m, n]);
        let mut meta = device.alloc_buffer(16, DType::U32, vec![4]).unwrap();
        meta.as_mut_slice::<u32>().unwrap().copy_from_slice(&[
            m as u32,
            n as u32,
            k as u32,
            group_size as u32,
        ]);

        let mut encoder = device.command_encoder().unwrap();
        dispatch_qmm_affine_t_f32(
            &mut encoder,
            &mut registry,
            device.metal_device(),
            &x_buf,
            &q_buf,
            &s_buf,
            &b_buf,
            &y_buf,
            &meta,
            m as u32,
            n as u32,
            k as u32,
            group_size as u32,
        )
        .unwrap();
        encoder.commit_and_wait().unwrap();

        let gpu = y_buf.as_slice::<f32>().unwrap();
        // Host oracle: y = x @ dequant(q_int, scales, biases)^T.
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for g in 0..groups_per_row {
                    let s = lin.scales[col * groups_per_row + g] as f64;
                    let b = lin.biases[col * groups_per_row + g] as f64;
                    for i in 0..group_size {
                        let kk = g * group_size + i;
                        let q = lin.q_int[col * k + kk] as f64;
                        acc += (x[r * k + kk] as f64) * (q * s + b);
                    }
                }
                let expected = acc as f32;
                let got = gpu[r * n + col];
                assert!(
                    (got - expected).abs() < 1e-3 * expected.abs().max(1.0),
                    "y[{r},{col}]: got {got} expected {expected}"
                );
            }
        }
    }

    // ── ADR-020 AC#7 foundation F2 — Q4_0 round-trip drift tests ────
    // Pure CPU; no GPU.  Validates the metric used to decide whether
    // the iter-B1 lm_head Vec<f32> overwrite path can carry DWQ
    // improvement to inference, or whether the iter-B2 affine kernel
    // route is required.

    /// Helper: build a synthetic Linear with explicit (s, b) per group.
    fn make_synthetic_linear(
        n: usize, k: usize, group_size: usize, bits: u32,
        scales_per_group: f32, biases_per_group: f32,
        codes: impl Fn(usize, usize) -> u8,
    ) -> MlxAffineLinear {
        let groups_per_row = k / group_size;
        let mut q_int = vec![0u8; n * k];
        for i in 0..n {
            for j in 0..k {
                q_int[i * k + j] = codes(i, j);
            }
        }
        let scales = vec![scales_per_group; n * groups_per_row];
        let biases = vec![biases_per_group; n * groups_per_row];
        MlxAffineLinear { n, k, group_size, bits, q_int, scales, biases }
    }

    #[test]
    fn round_trip_drift_zero_signal_is_zero_drift() {
        // All codes = 0, scales = 0 ⇒ dequant = 0 ⇒ Q4_0 round-trip = 0
        // ⇒ drift exactly 0 and rms_signal == 0 ⇒ relative_rms = NaN
        // (degenerate signal, sentinel).
        let lin = make_synthetic_linear(2, 32, 32, 4, 0.0, 0.0, |_, _| 0);
        let d = lin.q4_0_round_trip_drift().expect("drift");
        assert_eq!(d.mean_abs_drift, 0.0);
        assert_eq!(d.max_abs_drift, 0.0);
        assert_eq!(d.rms_drift, 0.0);
        assert_eq!(d.rms_signal, 0.0);
        assert!(d.relative_rms.is_nan(),
                "rms_signal == 0 ⇒ relative_rms = NaN; got {}", d.relative_rms);
    }

    #[test]
    fn round_trip_drift_q4_0_already_idempotent() {
        // Construct DWQ values that ARE already on the Q4_0 grid.
        //
        // Q4_0 (per `q_legacy::quantize_row_q4_0`) picks per-block
        // `d = max / -8.0` where `max` is the value with the largest
        // |v| (signed value retained); then encodes
        // `xi = clamp((v/d + 8.5) as i32, 0, 15)` and decodes
        // `y = (xi - 8) * d`.
        //
        // Fixture: scale=1.0, bias=-8.0, codes (j % 16) ∈ [0, 15]
        //   ⇒ DWQ-dequant values v = code - 8 ∈ {-8, -7, ..., 7}
        //   ⇒ amax=8, max=-8, d = -8/-8 = 1.0, id = 1.0
        //   ⇒ for v=-8: xi=clamp((-8 + 8.5) as i32, 0, 15) = 0,
        //     y = (0 - 8) * 1.0 = -8 ✓ drift=0
        //   ⇒ for v=7:  xi=clamp((7 + 8.5) as i32, 0, 15) = 15,
        //     y = (15 - 8) * 1.0 = 7  ✓ drift=0
        //   ⇒ all 16 grid points round-trip exactly.
        //
        // This proves the round-trip codec is an identity on Q4_0-grid
        // inputs (the codec's natural fixed point), and falsifies any
        // implementation bug that would introduce drift on this input.
        let lin = make_synthetic_linear(
            1, 32, 32, 4,
            1.0, -8.0,
            |_, j| (j % 16) as u8,  // codes 0..15, repeated twice in 32-block
        );
        let d = lin.q4_0_round_trip_drift().expect("drift");
        assert!(
            d.max_abs_drift < 1e-5,
            "max_abs_drift {} should be ~0 for already-Q4_0-grid values",
            d.max_abs_drift,
        );
        assert!(
            d.rms_drift < 1e-5,
            "rms_drift {} should be ~0", d.rms_drift,
        );
        assert!(
            d.relative_rms < 1e-3,
            "relative_rms {} should be ~0", d.relative_rms,
        );
    }

    #[test]
    fn round_trip_drift_increases_with_bias_magnitude() {
        // The bias term has no Q4_0 representation.  As bias grows
        // relative to scale*codes, more of the signal lives in bias
        // and gets DESTROYED by the round-trip — observable as
        // increasing `bias_fraction` AND increasing `relative_rms`.
        // Fixture: bias-dominated input where Q4_0 cannot represent
        // the bias offset because scale would have to be huge AND the
        // codes are clustered.

        // Case A: small bias relative to scale (low bias_fraction).
        let lin_low = make_synthetic_linear(
            2, 64, 32, 4,
            0.1, 0.01,                // scale=0.1, bias=0.01
            |_, j| (j % 16) as u8,    // codes 0..15
        );
        // Case B: large bias dominates (high bias_fraction).
        let lin_high = make_synthetic_linear(
            2, 64, 32, 4,
            0.01, 1.0,                // scale=0.01, bias=1.0
            |_, j| (j % 16) as u8,    // same codes
        );
        let d_low  = lin_low.q4_0_round_trip_drift().expect("low");
        let d_high = lin_high.q4_0_round_trip_drift().expect("high");

        // Sanity on bias_fraction direction.
        assert!(
            d_high.bias_fraction > d_low.bias_fraction,
            "high bias_fraction {} should exceed low {}",
            d_high.bias_fraction, d_low.bias_fraction,
        );
        // The high-bias case lives almost entirely in the bias term:
        // expect bias_fraction near 1.0 (bias dominates signal energy).
        assert!(
            d_high.bias_fraction > 0.9,
            "bias-dominated case should have bias_fraction > 0.9; got {}",
            d_high.bias_fraction,
        );
    }

    #[test]
    fn round_trip_drift_rejects_misaligned_k() {
        // Q4_0 production codec requires k % 32 == 0.
        let lin = make_synthetic_linear(1, 16, 16, 4, 1.0, 0.0, |_, _| 0);
        let err = lin.q4_0_round_trip_drift().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("not a multiple of Q4_0 block size"),
            "expected alignment error, got: {msg}"
        );
    }

    #[test]
    fn round_trip_drift_per_element_metrics_are_consistent() {
        // mean_abs_drift <= max_abs_drift, rms_drift <= max_abs_drift,
        // and relative_rms = rms_drift / rms_signal (within fp).
        let lin = make_synthetic_linear(
            4, 64, 32, 4,
            0.1, 0.05,
            |i, j| ((i + j) % 16) as u8,
        );
        let d = lin.q4_0_round_trip_drift().expect("drift");
        assert!(d.mean_abs_drift <= d.max_abs_drift,
                "mean {} > max {}", d.mean_abs_drift, d.max_abs_drift);
        assert!(d.rms_drift <= d.max_abs_drift,
                "rms {} > max {}", d.rms_drift, d.max_abs_drift);
        // Reconstruct relative_rms and check it agrees.
        if d.rms_signal > 0.0 {
            let expected = d.rms_drift / d.rms_signal;
            assert!(
                (d.relative_rms - expected).abs() < 1e-5 * expected.abs().max(1.0),
                "relative_rms {} != rms_drift {} / rms_signal {} = {}",
                d.relative_rms, d.rms_drift, d.rms_signal, expected,
            );
        }
    }
}
