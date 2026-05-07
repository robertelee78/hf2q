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
    /// `unpack_u32_packed`.
    fn pack_u32(codes: &[u8], bits: u32) -> Vec<u8> {
        assert!(bits == 4 || bits == 8);
        let pack_factor = (32 / bits) as usize;
        assert!(codes.len() % pack_factor == 0);
        let mask: u32 = (1u32 << bits) - 1;
        let mut out = Vec::with_capacity(codes.len() / pack_factor * 4);
        for chunk in codes.chunks_exact(pack_factor) {
            let mut word: u32 = 0;
            for (j, &c) in chunk.iter().enumerate() {
                word |= ((c as u32) & mask) << (j as u32 * bits);
            }
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
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
}
