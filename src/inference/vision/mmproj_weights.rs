//! mmproj GGUF weight loader (ADR-005 Phase 2c, Task #15 iter 31).
//!
//! Reads every required tensor from a parsed `GgufFile` onto the Metal
//! device as F32 buffers, dequantizing Q-type tensors on the CPU first
//! via `mlx_native::gguf::GgufFile::load_tensor_f32`. The produced
//! `LoadedMmprojWeights` holds one `MlxBuffer` per tensor keyed by its
//! GGUF name (e.g. `"v.patch_embd.weight"`).
//!
//! # Sequencing vs iter 30's validator
//!
//! Iter 30 (`validate_tensor_set`) proves the tensors EXIST at startup
//! before the expensive load runs. Iter 31 (this module) actually
//! reads them onto the GPU. Caller should invoke the validator first
//! and bail early on missing tensors — that keeps the operator's
//! error message specific (missing list) rather than a generic
//! "tensor not found" from mid-load.
//!
//! # GPU cost
//!
//! Gemma 4 vision tower ≈ 400 MB of F32 after dequant (221 tensors).
//! The load sequentially dispatches a small allocation per tensor;
//! total time on M5 Max ≈ 150-300ms for a cold-page-cache load of the
//! Gemma 4 mmproj. Deliberately NOT parallelized: mlx-native's
//! `load_tensor_f32` serializes through the GGUF `BufReader` mutex,
//! and the cost is already dominated by the page-cache fill rather
//! than CPU dequant.
//!
//! # Not in this iter
//!
//! - Handler wiring. The loader is usable in isolation; the
//!   `process_multimodal_content` short-circuit at 501 is unchanged.
//!   iter 32+ threads the loaded weights through `patch_embed_forward`
//!   etc. as the ViT forward pass ports block-by-block.
//! - Lazy tensor loading. Every required tensor is loaded eagerly at
//!   `load()` time. A future iter can add a per-layer lazy mode if a
//!   memory-constrained deployment needs it.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use mlx_native::{MlxBuffer, MlxDevice};
use mlx_native::gguf::GgufFile;

use super::mmproj::{vit_layer_tensor, MmprojConfig};

/// Collection of mmproj tensors loaded onto a Metal device as F32.
///
/// Cheap to move; cloning requires the caller to pay the GPU-alloc
/// cost again (not implemented here — if a use case needs cheap
/// cloning, wrap in `Arc` at the call site).
pub struct LoadedMmprojWeights {
    /// Keyed by the tensor's GGUF name. Values are F32 `MlxBuffer`s
    /// with shape preserved from the source GGUF.
    tensors: HashMap<String, MlxBuffer>,
    /// Device handle kept alive for the lifetime of the buffers.
    /// Held for RAII even though public accessors go through `tensors`.
    _device: MlxDevice,
}

impl std::fmt::Debug for LoadedMmprojWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedMmprojWeights")
            .field("tensor_count", &self.tensors.len())
            .finish()
    }
}

impl LoadedMmprojWeights {
    /// Load every tensor from the GGUF file onto the supplied device
    /// as F32. Arch-agnostic — walks `gguf.tensor_names()` and doesn't
    /// assume a particular naming convention, so it transparently
    /// handles both Gemma 4's SigLIP-style tower AND classic CLIP
    /// producers. Callers should run `validate_tensor_set` + detect
    /// `ArchProfile` first to know what the forward-pass dispatch
    /// branch needs.
    ///
    /// `_cfg` is accepted but currently unused — kept in the signature
    /// because a future lazy/tiered loader will partition tensor loads
    /// by cfg (e.g., load only stem + first few blocks on cold start,
    /// lazy-load remaining blocks on first request).
    pub fn load(
        gguf: &GgufFile,
        _cfg: &MmprojConfig,
        device: MlxDevice,
    ) -> Result<Self> {
        let names = gguf.tensor_names();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in &names {
            let buf = gguf
                .load_tensor_f32(*name, &device)
                .map_err(|e| anyhow!("mmproj load_tensor_f32('{}'): {e}", name))?;
            tensors.insert((*name).to_string(), buf);
        }
        Ok(Self {
            tensors,
            _device: device,
        })
    }

    /// Load from a GGUF file path. Opens the file, creates a default
    /// MlxDevice, and loads every tensor. Convenience wrapper for the
    /// common startup path.
    pub fn load_from_path(path: &Path, cfg: &MmprojConfig) -> Result<Self> {
        let gguf = GgufFile::open(path)
            .map_err(|e| anyhow!("open mmproj GGUF {}: {e}", path.display()))?;
        let device = MlxDevice::new()
            .map_err(|e| anyhow!("create MlxDevice for mmproj load: {e}"))?;
        Self::load(&gguf, cfg, device)
    }

    /// Look up a tensor by its GGUF name. `None` when absent (optional
    /// tensors like biases — callers gate the forward-pass branch on
    /// `Some`).
    pub fn get(&self, name: &str) -> Option<&MlxBuffer> {
        self.tensors.get(name)
    }

    /// Build an empty `LoadedMmprojWeights` with no tensors. Useful for
    /// tests that need an `AppState.mmproj` shape but don't need to
    /// drive a forward pass. The shortcut accessors all return `Err`
    /// (as the real accessors would on a broken-producer file).
    pub fn empty(device: MlxDevice) -> Self {
        Self {
            tensors: HashMap::new(),
            _device: device,
        }
    }

    /// Test-only: build a `LoadedMmprojWeights` from a pre-populated
    /// tensor map. Used by parity tests that synthesize block weights
    /// in-process rather than load a real GGUF (which would require a
    /// fixture file on disk and the full 400 MB dequant cost).
    #[cfg(test)]
    pub fn from_tensors_for_test(
        tensors: HashMap<String, MlxBuffer>,
        device: MlxDevice,
    ) -> Self {
        Self {
            tensors,
            _device: device,
        }
    }

    /// Number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Empty when no tensors were loaded (only possible from an empty
    /// `expected_tensor_names`; normal load paths return ≥ 5 tensors).
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    // -----------------------------------------------------------------------
    // Stem shortcuts. Each returns the buffer when present OR errors with
    // a specific name — matches what a forward-pass call-site needs.
    // -----------------------------------------------------------------------

    pub fn patch_embd_weight(&self) -> Result<&MlxBuffer> {
        self.tensors
            .get(super::mmproj::TENSOR_PATCH_EMBD)
            .ok_or_else(|| anyhow!("mmproj missing '{}'", super::mmproj::TENSOR_PATCH_EMBD))
    }

    pub fn position_embd_weight(&self) -> Result<&MlxBuffer> {
        self.tensors
            .get(super::mmproj::TENSOR_POS_EMBD)
            .ok_or_else(|| anyhow!("mmproj missing '{}'", super::mmproj::TENSOR_POS_EMBD))
    }

    /// Read the gemma4v dual position-embed table as a typed
    /// `[2, pos_size, hidden]` 3-D view.
    ///
    /// Returns `(buf, pos_size, hidden)` where `buf` is the same backing
    /// `MlxBuffer` returned by `position_embd_weight()` (no copy). The
    /// gemma4v vision tower stores this as `model.embed_vision.
    /// position_embedding_table`, mapped to `v.position_embd.weight` by
    /// `src/backends/gguf.rs:1782-1786`. The first dim is fixed at 2
    /// (X-axis table at `[0, ..]`, Y-axis table at `[1, ..]`).
    ///
    /// # Errors
    ///
    /// - tensor missing
    /// - shape isn't 3-D, or first dim isn't 2
    /// - product of dims doesn't match the buffer's element count
    ///   (catches a stale GGUF write or a producer mismatch)
    ///
    /// # Why a sibling accessor instead of changing
    /// `position_embd_weight()`
    ///
    /// SigLIP-49's vision tower stores a 2-D
    /// `[num_patches (+1 cls), hidden]` table; gemma4v's is 3-D. The
    /// untyped accessor returns the raw buffer for both — callers that
    /// need typed shape information branch on `ArchProfile`. This
    /// addition lands the gemma4v branch without churning the SigLIP
    /// path.
    pub fn position_embd_table_3d(&self) -> Result<(&MlxBuffer, u32, u32)> {
        let buf = self.position_embd_weight()?;
        let shape = buf.shape();
        if shape.len() != 3 {
            return Err(anyhow!(
                "v.position_embd.weight: expected 3-D [2, pos_size, hidden], got shape {:?}",
                shape
            ));
        }
        if shape[0] != 2 {
            return Err(anyhow!(
                "v.position_embd.weight: expected first dim 2 (gemma4v dual table), got {}",
                shape[0]
            ));
        }
        let pos_size = shape[1] as u32;
        let hidden = shape[2] as u32;
        if pos_size == 0 || hidden == 0 {
            return Err(anyhow!(
                "v.position_embd.weight: pos_size ({pos_size}) and hidden ({hidden}) must be > 0"
            ));
        }
        // Buffer-element-count cross-check: 2 * pos_size * hidden f32s.
        let expected_bytes =
            2usize * (pos_size as usize) * (hidden as usize) * std::mem::size_of::<f32>();
        if buf.byte_len() < expected_bytes {
            return Err(anyhow!(
                "v.position_embd.weight: byte_len {} < expected {} (2 * {} * {} * 4)",
                buf.byte_len(),
                expected_bytes,
                pos_size,
                hidden
            ));
        }
        Ok((buf, pos_size, hidden))
    }

    pub fn post_ln_weight(&self) -> Result<&MlxBuffer> {
        self.tensors
            .get(super::mmproj::TENSOR_POST_LN_WEIGHT)
            .ok_or_else(|| anyhow!("mmproj missing '{}'", super::mmproj::TENSOR_POST_LN_WEIGHT))
    }

    /// Per-block tensor accessor.
    ///
    /// `suffix` is the block-relative name ("attn_q.weight",
    /// "ffn_down.weight", etc. — see `BLOCK_REQUIRED_SUFFIXES`).
    pub fn block_tensor(&self, layer_idx: usize, suffix: &str) -> Result<&MlxBuffer> {
        let key = vit_layer_tensor(layer_idx, suffix);
        self.tensors
            .get(&key)
            .ok_or_else(|| anyhow!("mmproj missing '{}'", key))
    }

    /// MLP projector accessors.
    pub fn mm_0_weight(&self) -> Result<&MlxBuffer> {
        self.tensors
            .get(super::mmproj::TENSOR_MM_0_WEIGHT)
            .ok_or_else(|| anyhow!("mmproj missing '{}'", super::mmproj::TENSOR_MM_0_WEIGHT))
    }

    pub fn mm_2_weight(&self) -> Result<&MlxBuffer> {
        self.tensors
            .get(super::mmproj::TENSOR_MM_2_WEIGHT)
            .ok_or_else(|| anyhow!("mmproj missing '{}'", super::mmproj::TENSOR_MM_2_WEIGHT))
    }

    // -----------------------------------------------------------------------
    // Gemma4ClippableLinear scalar bounds for `mm.0.weight`.
    //
    // Per `/opt/llama.cpp/tools/mtmd/clip.cpp:1935-1959`, gemma4v emits
    // four optional scalar f32 tensors as siblings of `mm.0.weight`:
    //   - `mm.0.input_min`, `mm.0.input_max` (clamps applied BEFORE matmul)
    //   - `mm.0.output_min`, `mm.0.output_max` (clamps applied AFTER matmul)
    //
    // Each is a 1-element f32 tensor (the converter `unsqueeze(0)`s the
    // 0-D scalar so GGUF round-trips it as a 1-D `[1]` tensor; see
    // `/opt/llama.cpp/convert_hf_to_gguf.py:7851-7853`).
    //
    // Returns `Some(value)` when the tensor is present and decodes to
    // exactly one f32, else `None`. Callers compose the four into a
    // `Gemma4ClippableLinearBounds` via `mm_0_bounds()`.
    // -----------------------------------------------------------------------

    fn read_scalar_f32(&self, name: &str) -> Option<f32> {
        let buf = self.tensors.get(name)?;
        let slice = buf.as_slice::<f32>().ok()?;
        // Defensive: clamp scalars are 1-element. If we ever load a
        // mis-shaped sibling (e.g. converter wrote a vector), reject
        // cleanly rather than silently picking element 0.
        if slice.len() != 1 {
            return None;
        }
        Some(slice[0])
    }

    /// Read the `mm.0.input_min` scalar bound (clamp BEFORE matmul).
    /// `None` when absent OR mis-shaped — caller treats absence as
    /// `f32::NEG_INFINITY` (no-op) per llama.cpp's default.
    pub fn mm_0_input_min(&self) -> Option<f32> {
        self.read_scalar_f32("mm.0.input_min")
    }
    /// Read the `mm.0.input_max` scalar bound. See `mm_0_input_min`.
    pub fn mm_0_input_max(&self) -> Option<f32> {
        self.read_scalar_f32("mm.0.input_max")
    }
    /// Read the `mm.0.output_min` scalar bound (clamp AFTER matmul).
    pub fn mm_0_output_min(&self) -> Option<f32> {
        self.read_scalar_f32("mm.0.output_min")
    }
    /// Read the `mm.0.output_max` scalar bound. See `mm_0_output_min`.
    pub fn mm_0_output_max(&self) -> Option<f32> {
        self.read_scalar_f32("mm.0.output_max")
    }

    /// Compose the four clamp scalars into a single
    /// `Gemma4ClippableLinearBounds`. All-`None` result means the
    /// projector is byte-equivalent to a plain Linear (no clamps).
    pub fn mm_0_bounds(&self) -> super::vit::Gemma4ClippableLinearBounds {
        super::vit::Gemma4ClippableLinearBounds {
            input_min: self.mm_0_input_min(),
            input_max: self.mm_0_input_max(),
            output_min: self.mm_0_output_min(),
            output_max: self.mm_0_output_max(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::mmproj::ProjectorType;

    /// Gemma 4 26B mmproj — present on this dev machine. Tests gate on
    /// existence so CI without the fixture skips them cleanly.
    const GEMMA4_MMPROJ_PATH: &str =
        "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq-mmproj.gguf";

    #[test]
    fn load_gemma4_mmproj_populates_arch_tensors() {
        // Real Gemma 4 mmproj (SigLIP variant): 356 tensors total —
        //   5 non-block (patch_embd, pos_embd, std_bias, std_scale, mm.0.weight)
        //   13/block × 27 blocks = 351
        //   No v.post_ln.weight, no mm.2.weight.
        // See /opt/hf2q/docs/ADR-005 iter 31 for the real tensor manifest.
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found at {}", GEMMA4_MMPROJ_PATH);
            return;
        }
        let gguf = GgufFile::open(path).expect("open gemma4 mmproj");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("parse mmproj config");
        // Sanity: Gemma 4's 27-layer SigLIP at 224×224 with 16×16 patches.
        assert_eq!(cfg.num_hidden_layers, 27);
        assert_eq!(cfg.image_size, 224);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.hidden_size, 1152);
        assert_eq!(cfg.projector, ProjectorType::Mlp);

        let device = MlxDevice::new().expect("create device");
        let weights = LoadedMmprojWeights::load(&gguf, &cfg, device).expect("load weights");
        // Gemma 4 mmproj has 356 tensors total.
        assert_eq!(weights.len(), 356);
        // Arch-agnostic shortcuts present.
        weights.patch_embd_weight().expect("patch_embd_weight");
        weights.position_embd_weight().expect("position_embd_weight");
        weights.mm_0_weight().expect("mm_0_weight");
        // post_ln + mm.2 do NOT exist in Gemma 4 mmproj.
        assert!(weights.post_ln_weight().is_err());
        assert!(weights.mm_2_weight().is_err());
        // Every layer's arch-agnostic QKV+output suffixes present.
        for layer_idx in 0..27 {
            for suffix in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
            ] {
                weights
                    .block_tensor(layer_idx, suffix)
                    .unwrap_or_else(|_| panic!("layer {} {}", layer_idx, suffix));
            }
        }
    }

    #[test]
    fn load_gemma4_mmproj_patch_embd_has_expected_shape_and_values() {
        // `v.patch_embd.weight` in Gemma 4 is a 2D tensor [hidden,
        // 3*patch*patch] = [1152, 768] = 884,736 f32 elements. This is
        // the flattened Conv2d kernel ready for a matmul against the
        // flattened patch pixels.
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found at {}", GEMMA4_MMPROJ_PATH);
            return;
        }
        let gguf = GgufFile::open(path).expect("open gemma4 mmproj");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("parse mmproj config");
        let device = MlxDevice::new().expect("create device");
        let weights = LoadedMmprojWeights::load(&gguf, &cfg, device).expect("load weights");
        let patch = weights.patch_embd_weight().expect("patch_embd");
        let expected_elems = (cfg.hidden_size as usize)
            * 3
            * (cfg.patch_size as usize)
            * (cfg.patch_size as usize);
        let data: &[f32] = patch.as_slice().expect("as_slice f32");
        assert_eq!(data.len(), expected_elems, "patch_embd element count");
        // Sanity: loaded f32 dequant output is non-trivial.
        let sum_abs: f32 = data.iter().take(1024).map(|v| v.abs()).sum();
        assert!(
            sum_abs > 0.0,
            "first 1024 elements all zero — probable load bug"
        );
    }

    #[test]
    fn load_from_path_wraps_gguf_open_and_device_create() {
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!("skipping: mmproj fixture not found at {}", GEMMA4_MMPROJ_PATH);
            return;
        }
        let gguf = GgufFile::open(path).expect("open for cfg");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let weights = LoadedMmprojWeights::load_from_path(path, &cfg).expect("load_from_path");
        assert_eq!(weights.len(), 356);
    }

    #[test]
    fn accessors_return_err_with_specific_name_when_missing() {
        // Synthetic LoadedMmprojWeights with an empty tensor map — every
        // accessor should return Err naming the missing tensor.
        let weights = LoadedMmprojWeights {
            tensors: HashMap::new(),
            _device: MlxDevice::new().expect("device"),
        };
        let err = weights.patch_embd_weight().unwrap_err();
        assert!(format!("{err}").contains("v.patch_embd.weight"));

        let err = weights.position_embd_weight().unwrap_err();
        assert!(format!("{err}").contains("v.position_embd.weight"));

        let err = weights.block_tensor(5, "attn_q.weight").unwrap_err();
        assert!(format!("{err}").contains("v.blk.5.attn_q.weight"));

        let err = weights.mm_0_weight().unwrap_err();
        assert!(format!("{err}").contains("mm.0.weight"));
    }

    #[test]
    fn empty_weights_report_len_zero_and_is_empty_true() {
        let weights = LoadedMmprojWeights {
            tensors: HashMap::new(),
            _device: MlxDevice::new().expect("device"),
        };
        assert_eq!(weights.len(), 0);
        assert!(weights.is_empty());
    }

    #[test]
    fn get_returns_none_for_absent_tensor() {
        let weights = LoadedMmprojWeights {
            tensors: HashMap::new(),
            _device: MlxDevice::new().expect("device"),
        };
        assert!(weights.get("v.patch_embd.weight").is_none());
    }

    #[test]
    fn position_embd_table_3d_rejects_non_3d_shape() {
        // Synthesize a LoadedMmprojWeights with a 2-D position-embd
        // (the SigLIP shape). The 3-D accessor must reject it cleanly.
        let device = MlxDevice::new().expect("device");
        let buf = device
            .alloc_buffer(64 * 4, mlx_native::DType::F32, vec![8, 8])
            .expect("alloc");
        let mut tensors = HashMap::new();
        tensors.insert(super::super::mmproj::TENSOR_POS_EMBD.to_string(), buf);
        let weights = LoadedMmprojWeights {
            tensors,
            _device: device,
        };
        let err = weights.position_embd_table_3d().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("expected 3-D"),
            "wrong error msg: {msg}"
        );
    }

    #[test]
    fn position_embd_table_3d_rejects_first_dim_not_two() {
        let device = MlxDevice::new().expect("device");
        let buf = device
            .alloc_buffer(96 * 4, mlx_native::DType::F32, vec![3, 4, 8])
            .expect("alloc");
        let mut tensors = HashMap::new();
        tensors.insert(super::super::mmproj::TENSOR_POS_EMBD.to_string(), buf);
        let weights = LoadedMmprojWeights {
            tensors,
            _device: device,
        };
        let err = weights.position_embd_table_3d().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("first dim 2"),
            "wrong error msg: {msg}"
        );
    }

    #[test]
    fn position_embd_table_3d_returns_dims_for_valid_shape() {
        let device = MlxDevice::new().expect("device");
        let pos_size = 27usize;
        let hidden = 1152usize;
        let buf = device
            .alloc_buffer(2 * pos_size * hidden * 4, mlx_native::DType::F32, vec![2, pos_size, hidden])
            .expect("alloc");
        let mut tensors = HashMap::new();
        tensors.insert(super::super::mmproj::TENSOR_POS_EMBD.to_string(), buf);
        let weights = LoadedMmprojWeights {
            tensors,
            _device: device,
        };
        let (buf, ps, h) = weights.position_embd_table_3d().expect("3d ok");
        assert_eq!(ps, pos_size as u32);
        assert_eq!(h, hidden as u32);
        assert_eq!(buf.shape(), &[2, pos_size, hidden]);
    }

    #[test]
    fn position_embd_table_3d_propagates_missing_tensor_error() {
        let device = MlxDevice::new().expect("device");
        let weights = LoadedMmprojWeights::empty(device);
        let err = weights.position_embd_table_3d().unwrap_err();
        assert!(format!("{err}").contains("v.position_embd.weight"));
    }

    #[test]
    fn empty_constructor_produces_zero_tensor_weights() {
        // `empty(device)` is the pub constructor for test/scaffolding
        // call sites that need a LoadedMmprojWeights shape without a
        // real 400MB load. Should len == 0, is_empty == true, and
        // every shortcut accessor should return Err.
        let device = MlxDevice::new().expect("device");
        let weights = LoadedMmprojWeights::empty(device);
        assert_eq!(weights.len(), 0);
        assert!(weights.is_empty());
        assert!(weights.patch_embd_weight().is_err());
        assert!(weights.position_embd_weight().is_err());
        assert!(weights.post_ln_weight().is_err());
        assert!(weights.mm_0_weight().is_err());
        assert!(weights.mm_2_weight().is_err());
        assert!(weights.block_tensor(0, "attn_q.weight").is_err());
    }

    /// iter-115: mm.0 Gemma4ClippableLinear scalar bounds accessors.
    /// Build a synthetic `LoadedMmprojWeights` carrying only the four
    /// 1-element clamp-scalar tensors and assert each accessor returns
    /// the expected scalar.
    #[test]
    fn mm_0_clamp_scalar_accessors_round_trip_single_element_tensors() {
        use mlx_native::DType;
        let device = MlxDevice::new().expect("device");
        let put = |tensors: &mut HashMap<String, MlxBuffer>,
                   dev: &MlxDevice,
                   name: &str,
                   value: f32| {
            // 1-element f32 tensor with shape [1] — matches what
            // convert_hf_to_gguf.py emits for the unsqueeze(0)'d scalar.
            let buf = dev.alloc_buffer(4, DType::F32, vec![1]).expect("alloc scalar");
            let s: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut f32, 1)
            };
            s[0] = value;
            tensors.insert(name.to_string(), buf);
        };
        let mut tensors: HashMap<String, MlxBuffer> = HashMap::new();
        put(&mut tensors, &device, "mm.0.input_min", -2.5);
        put(&mut tensors, &device, "mm.0.input_max", 2.5);
        put(&mut tensors, &device, "mm.0.output_min", -10.0);
        put(&mut tensors, &device, "mm.0.output_max", 10.0);
        let weights = LoadedMmprojWeights::from_tensors_for_test(tensors, device);

        assert_eq!(weights.mm_0_input_min(), Some(-2.5));
        assert_eq!(weights.mm_0_input_max(), Some(2.5));
        assert_eq!(weights.mm_0_output_min(), Some(-10.0));
        assert_eq!(weights.mm_0_output_max(), Some(10.0));

        let bounds = weights.mm_0_bounds();
        assert!(bounds.any());
        assert_eq!(bounds.input_min, Some(-2.5));
        assert_eq!(bounds.output_max, Some(10.0));
    }

    /// Absence of any clamp-scalar tensor → all accessors return None,
    /// `mm_0_bounds().any()` is false (the projector degrades to a
    /// plain Linear, byte-equivalent to the no-clamp path).
    #[test]
    fn mm_0_clamp_scalar_accessors_return_none_when_absent() {
        let device = MlxDevice::new().expect("device");
        let weights = LoadedMmprojWeights::empty(device);
        assert_eq!(weights.mm_0_input_min(), None);
        assert_eq!(weights.mm_0_input_max(), None);
        assert_eq!(weights.mm_0_output_min(), None);
        assert_eq!(weights.mm_0_output_max(), None);
        let bounds = weights.mm_0_bounds();
        assert!(!bounds.any());
    }
}
