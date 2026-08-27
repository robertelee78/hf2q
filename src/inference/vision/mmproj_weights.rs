//! mmproj GGUF weight loader (ADR-005 Phase 2c, Task #15 iter 31).
//!
//! Maps every required tensor from a parsed `GgufFile` into Metal without
//! changing its stored representation. Matrix weights have role-typed
//! [`MlxQWeight`] ownership; F32 runtime state has a separate role. The
//! `MlxBuffer` map is a zero-copy source/state index; production matrix
//! execution resolves only through the typed matrix map.
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
//! Physical storage accounting is allocation-aware: all tensor views into one
//! mapped Metal segment count that segment exactly once, and fused-QKV aliases
//! never add storage bytes.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{anyhow, Result};
use mlx_native::gguf::{GgufFile, TensorInfo};
use mlx_native::{DType, GgmlType, MlxBuffer, MlxDevice};

use super::mmproj::{vit_layer_tensor, MmprojConfig};
use crate::serve::forward_mlx_shared::{native_gguf_matrix_bytes, MlxQWeight};

/// Runtime role of one mmproj GGUF tensor.
///
/// Matrices retain their source codec and are eligible for native matrix
/// dispatch. The admitted stored formats are F32, F16, BF16, Q4_0, Q5_0,
/// Q8_0, Q4_K, Q5_K, and Q6_K. F32 state is intentionally not admitted
/// through matrix kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmprojTensorRole {
    Matrix,
    F32State,
}

/// Allocation-aware storage receipt for one loaded mmproj generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MmprojStorageAccounting {
    pub source_tensor_count: usize,
    pub source_matrix_count: usize,
    pub source_f32_state_count: usize,
    pub alias_view_count: usize,
    pub unique_backing_allocations: usize,
    pub mapped_backing_bytes: u64,
    pub anonymous_backing_bytes: u64,
    pub source_logical_bytes: u64,
    pub matrix_logical_bytes: u64,
    pub f32_state_logical_bytes: u64,
}

impl MmprojStorageAccounting {
    pub fn total_backing_bytes(self) -> Result<u64> {
        self.mapped_backing_bytes
            .checked_add(self.anonymous_backing_bytes)
            .ok_or_else(|| anyhow!("mmproj backing-byte accounting overflow"))
    }
}

fn classify_mmproj_tensor_role(name: &str) -> Result<MmprojTensorRole> {
    let is_scalar_bound = ["input_min", "input_max", "output_min", "output_max"]
        .iter()
        .any(|suffix| name.ends_with(suffix));
    let is_f32_state = crate::models::vit::convert::vit_emission_is_f32(name)
        || matches!(name, "v.std_bias" | "v.std_scale")
        || is_scalar_bound;
    if is_f32_state {
        return Ok(MmprojTensorRole::F32State);
    }
    if name.ends_with(".weight") || name.ends_with(".weight.1") {
        return Ok(MmprojTensorRole::Matrix);
    }
    Err(anyhow!(
        "mmproj tensor '{name}' has no admitted Matrix or F32State role"
    ))
}

fn block_alias_suffix(suffix: &str) -> Option<&'static str> {
    match suffix {
        "attn_output.weight" => Some("attn_out.weight"),
        "attn_output.bias" => Some("attn_out.bias"),
        "attn_out.weight" => Some("attn_output.weight"),
        "attn_out.bias" => Some("attn_output.bias"),
        "post_ffw_norm.weight" => Some("ffn_post_norm.weight"),
        "post_ffw_norm.bias" => Some("ffn_post_norm.bias"),
        "ffn_post_norm.weight" => Some("post_ffw_norm.weight"),
        "ffn_post_norm.bias" => Some("post_ffw_norm.bias"),
        _ => None,
    }
}

fn checked_shape_elements(info: &TensorInfo) -> Result<usize> {
    anyhow::ensure!(
        !info.shape.is_empty(),
        "mmproj tensor '{}' must not be scalar-rank metadata",
        info.name
    );
    info.shape.iter().try_fold(1usize, |product, dimension| {
        anyhow::ensure!(
            *dimension > 0,
            "mmproj tensor '{}' has a zero dimension in {:?}",
            info.name,
            info.shape
        );
        product
            .checked_mul(*dimension)
            .ok_or_else(|| anyhow!("mmproj tensor '{}' shape product overflow", info.name))
    })
}

/// Return the exact logical matrix view for one admitted matrix tensor.
/// Rank > 2 flattening is limited to the two contiguous patch-convolution
/// weights whose contract is `[N, C, ...] -> [N, K]`.
fn matrix_shape(info: &TensorInfo) -> Result<(usize, usize)> {
    anyhow::ensure!(
        info.shape.len() >= 2,
        "mmproj matrix '{}' must have rank >= 2, got {:?}",
        info.name,
        info.shape
    );
    if info.shape.len() == 2 {
        return Ok((info.shape[0], info.shape[1]));
    }
    anyhow::ensure!(
        matches!(
            info.name.as_str(),
            "v.patch_embd.weight" | "v.patch_embd.weight.1"
        ),
        "mmproj matrix '{}' may not flatten rank-{} shape {:?}; only patch weights have a rank>2 matrix-view contract",
        info.name,
        info.shape.len(),
        info.shape
    );
    let rows = info.shape[0];
    let cols = info.shape[1..]
        .iter()
        .try_fold(1usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| anyhow!("mmproj patch matrix '{}' flattened K overflow", info.name))
        })?;
    Ok((rows, cols))
}

fn validate_mmproj_tensor_role_codec(info: &TensorInfo, role: MmprojTensorRole) -> Result<()> {
    let elements = checked_shape_elements(info)?;
    match role {
        MmprojTensorRole::Matrix => {
            let (rows, cols) = matrix_shape(info)?;
            anyhow::ensure!(
                rows.checked_mul(cols) == Some(elements),
                "mmproj matrix '{}' flattened shape does not preserve element count",
                info.name
            );
            let expected = native_gguf_matrix_bytes(info.ggml_type, rows, cols)?;
            anyhow::ensure!(
                info.byte_len == expected,
                "mmproj matrix '{}' {:?} metadata has {} bytes, expected exactly {expected} for [{rows}, {cols}]",
                info.name,
                info.ggml_type,
                info.byte_len
            );
        }
        MmprojTensorRole::F32State => {
            anyhow::ensure!(
                info.ggml_type == GgmlType::F32,
                "mmproj state '{}' must be stored as F32, got {:?}; no dequantization or widening fallback is allowed",
                info.name,
                info.ggml_type
            );
            let expected = elements
                .checked_mul(DType::F32.size_of())
                .ok_or_else(|| anyhow!("mmproj F32 state '{}' byte extent overflow", info.name))?;
            anyhow::ensure!(
                info.byte_len == expected,
                "mmproj F32 state '{}' metadata has {} bytes, expected exactly {expected}",
                info.name,
                info.byte_len
            );
        }
    }
    Ok(())
}

fn validate_source_tensor_roles(
    gguf: &GgufFile,
    names: &[&str],
) -> Result<HashMap<String, MmprojTensorRole>> {
    let mut roles = HashMap::with_capacity(names.len());
    for &name in names {
        let info = gguf
            .tensor_info(name)
            .ok_or_else(|| anyhow!("mmproj tensor metadata disappeared for '{name}'"))?;
        let role = classify_mmproj_tensor_role(name)?;
        validate_mmproj_tensor_role_codec(info, role)?;
        roles.insert(name.to_string(), role);
    }
    Ok(roles)
}

fn checked_add_bytes(total: &mut u64, bytes: usize, label: &str) -> Result<()> {
    let bytes = u64::try_from(bytes).map_err(|_| anyhow!("{label} exceeds u64"))?;
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| anyhow!("{label} accounting overflow"))?;
    Ok(())
}

fn summarize_source_storage(
    tensors: &HashMap<String, MlxBuffer>,
    roles: &HashMap<String, MmprojTensorRole>,
) -> Result<MmprojStorageAccounting> {
    anyhow::ensure!(
        tensors.len() == roles.len(),
        "mmproj storage accounting requires one role per source tensor"
    );
    let mut storage = MmprojStorageAccounting {
        source_tensor_count: tensors.len(),
        ..MmprojStorageAccounting::default()
    };
    let mut allocations = HashSet::new();
    for (name, buffer) in tensors {
        let role = roles
            .get(name)
            .ok_or_else(|| anyhow!("mmproj storage accounting missing role for '{name}'"))?;
        checked_add_bytes(
            &mut storage.source_logical_bytes,
            buffer.data_byte_len(),
            "mmproj source logical bytes",
        )?;
        match role {
            MmprojTensorRole::Matrix => {
                storage.source_matrix_count += 1;
                checked_add_bytes(
                    &mut storage.matrix_logical_bytes,
                    buffer.data_byte_len(),
                    "mmproj matrix logical bytes",
                )?;
            }
            MmprojTensorRole::F32State => {
                storage.source_f32_state_count += 1;
                checked_add_bytes(
                    &mut storage.f32_state_logical_bytes,
                    buffer.data_byte_len(),
                    "mmproj F32-state logical bytes",
                )?;
            }
        }
        let allocation = (buffer.contents_ptr() as usize, buffer.byte_len());
        if allocations.insert(allocation) {
            storage.unique_backing_allocations += 1;
            if buffer.is_file_backed() {
                checked_add_bytes(
                    &mut storage.mapped_backing_bytes,
                    buffer.byte_len(),
                    "mmproj mapped backing bytes",
                )?;
            } else {
                checked_add_bytes(
                    &mut storage.anonymous_backing_bytes,
                    buffer.byte_len(),
                    "mmproj anonymous backing bytes",
                )?;
            }
        }
    }
    Ok(storage)
}

/// Collection of role-typed mmproj tensors mapped onto a Metal device.
///
/// Cheap to move; cloning requires the caller to pay the GPU-alloc
/// cost again (not implemented here — if a use case needs cheap
/// cloning, wrap in `Arc` at the call site).
pub struct LoadedMmprojWeights {
    /// Zero-copy source/state index. Production matrix execution does not
    /// consume untyped entries from this map.
    tensors: HashMap<String, MlxBuffer>,
    /// Exact native stored-format matrices. Fused-QKV split aliases are
    /// represented here as independent logical `MlxQWeight`s sharing the
    /// source allocation.
    matrix_weights: HashMap<String, MlxQWeight>,
    /// Names admitted as F32 runtime state. Their buffers live in `tensors`.
    f32_state_names: HashSet<String>,
    storage: MmprojStorageAccounting,
    /// Exact page-rounded segment sizes of the file-backed Metal mappings.
    /// Alias views do not appear here because they retain existing segments.
    mapped_segment_physical_bytes: Vec<u64>,
    /// Exact physical bytes owned by this generation's primary mappings (or
    /// by synthetic anonymous buffers in test-only constructors).
    owned_bytes: u64,
    /// Device handle kept alive for the lifetime of the buffers.
    /// Held for RAII even though public accessors go through `tensors`.
    _device: MlxDevice,
}

impl std::fmt::Debug for LoadedMmprojWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedMmprojWeights")
            .field("tensor_count", &self.tensors.len())
            .field("matrix_count", &self.matrix_weights.len())
            .field("f32_state_count", &self.f32_state_names.len())
            .field("storage", &self.storage)
            .field(
                "mapped_segment_physical_bytes",
                &self.mapped_segment_physical_bytes,
            )
            .field("owned_bytes", &self.owned_bytes)
            .finish()
    }
}

impl LoadedMmprojWeights {
    /// Validate the complete stored-format role/codec contract without
    /// creating mmap or Metal resources.
    pub fn validate_native_storage(gguf: &GgufFile) -> Result<()> {
        let names = gguf.tensor_names();
        validate_source_tensor_roles(gguf, &names).map(|_| ())
    }

    /// Map every source tensor once, validate its semantic role and codec,
    /// then build zero-copy role-typed views. Unknown names, quantized state,
    /// and unsupported matrix codecs fail before any runtime publication.
    pub fn load(gguf: &GgufFile, cfg: &MmprojConfig, device: MlxDevice) -> Result<Self> {
        let names = gguf.tensor_names();
        // Preflight every semantic role, codec, shape, and byte extent before
        // creating mmap or Metal resources. A later mapping error therefore
        // cannot leave a partially admitted projector contract.
        let roles = validate_source_tensor_roles(gguf, &names)?;
        let mapped = gguf
            .map_tensor_data(&device)
            .map_err(|error| anyhow!("map mmproj GGUF tensor data: {error}"))?;
        let mapped_segment_physical_bytes = mapped
            .storage_plan()
            .segment_physical_byte_lens()
            .iter()
            .map(|&bytes| u64::try_from(bytes).map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        let owned_bytes = u64::try_from(mapped.physical_byte_len())?;
        let mut tensors = HashMap::with_capacity(names.len());
        let mut matrix_weights = HashMap::with_capacity(names.len());
        let mut f32_state_names = HashSet::with_capacity(names.len());
        for name in &names {
            let info = gguf
                .tensor_info(name)
                .ok_or_else(|| anyhow!("mmproj tensor metadata disappeared for '{name}'"))?;
            let role = *roles
                .get(*name)
                .ok_or_else(|| anyhow!("mmproj role preflight disappeared for '{name}'"))?;
            match role {
                MmprojTensorRole::Matrix => {
                    let (rows, cols) = matrix_shape(info)?;
                    let weight =
                        MlxQWeight::from_mapped_gguf_matrix_view(&mapped, info, rows, cols)?;
                    tensors.insert((*name).to_string(), weight.buffer.clone());
                    matrix_weights.insert((*name).to_string(), weight);
                }
                MmprojTensorRole::F32State => {
                    let buffer = mapped
                        .load_tensor(name)
                        .map_err(|error| anyhow!("map mmproj F32 state '{name}': {error}"))?;
                    anyhow::ensure!(
                        buffer.is_file_backed()
                            && buffer.dtype() == DType::F32
                            && buffer.data_byte_len() == info.byte_len,
                        "mmproj F32 state '{name}' did not retain its exact file-backed payload"
                    );
                    tensors.insert((*name).to_string(), buffer);
                    f32_state_names.insert((*name).to_string());
                }
            }
        }

        let mut storage = summarize_source_storage(&tensors, &roles)?;
        anyhow::ensure!(
            storage.anonymous_backing_bytes == 0
                && storage.mapped_backing_bytes == owned_bytes
                && storage.unique_backing_allocations == mapped_segment_physical_bytes.len(),
            "mmproj realized mapping receipt disagrees with mapped tensor storage plan: storage={storage:?}, segments={mapped_segment_physical_bytes:?}"
        );
        let source_tensor_count = tensors.len();
        Self::install_native_fused_attn_qkv_views(
            &mut tensors,
            &mut matrix_weights,
            &mut f32_state_names,
            cfg,
        )?;
        storage.alias_view_count = tensors
            .len()
            .checked_sub(source_tensor_count)
            .ok_or_else(|| anyhow!("mmproj alias-view accounting underflow"))?;

        Ok(Self {
            tensors,
            matrix_weights,
            f32_state_names,
            storage,
            mapped_segment_physical_bytes,
            owned_bytes,
            _device: device,
        })
    }

    /// Install codec-aware split matrix/state aliases for fused QKV tensors.
    /// Every alias shares the source allocation; no payload is transformed.
    fn install_native_fused_attn_qkv_views(
        tensors: &mut HashMap<String, MlxBuffer>,
        matrix_weights: &mut HashMap<String, MlxQWeight>,
        f32_state_names: &mut HashSet<String>,
        cfg: &MmprojConfig,
    ) -> Result<()> {
        let hidden = cfg.hidden_size as usize;
        if hidden == 0 {
            return Ok(());
        }
        for layer_idx in 0..cfg.num_hidden_layers as usize {
            let fused_w = vit_layer_tensor(layer_idx, "attn_qkv.weight");
            let fused_b = vit_layer_tensor(layer_idx, "attn_qkv.bias");
            let split_q_w = vit_layer_tensor(layer_idx, "attn_q.weight");
            let split_k_w = vit_layer_tensor(layer_idx, "attn_k.weight");
            let split_v_w = vit_layer_tensor(layer_idx, "attn_v.weight");
            let split_q_b = vit_layer_tensor(layer_idx, "attn_q.bias");
            let split_k_b = vit_layer_tensor(layer_idx, "attn_k.bias");
            let split_v_b = vit_layer_tensor(layer_idx, "attn_v.bias");

            let has_fused_w = matrix_weights.contains_key(&fused_w);
            let has_split_w = matrix_weights.contains_key(&split_q_w)
                || matrix_weights.contains_key(&split_k_w)
                || matrix_weights.contains_key(&split_v_w);
            let has_fused_b = tensors.contains_key(&fused_b);
            let has_split_b = tensors.contains_key(&split_q_b)
                || tensors.contains_key(&split_k_b)
                || tensors.contains_key(&split_v_b);

            if (has_fused_w && has_split_w) || (has_fused_b && has_split_b) {
                return Err(anyhow!(
                    "mmproj loader: block {layer_idx} has BOTH fused and split \
                     attn_q/k/v tensors (fused weight '{}') — refusing to mix conventions \
                     (validator should have caught this at startup; bypassing \
                     validator is unsupported)",
                    fused_w
                ));
            }
            anyhow::ensure!(
                has_fused_w || !has_fused_b,
                "mmproj loader: block {layer_idx} has fused bias '{fused_b}' without fused weight '{fused_w}'"
            );
            if !has_fused_w {
                continue;
            }

            let (q_weight, k_weight, v_weight) = {
                let fused = matrix_weights
                    .get(&fused_w)
                    .expect("has_fused_w=true contract violated by matrix_weights.get");
                let expected_rows = hidden
                    .checked_mul(3)
                    .ok_or_else(|| anyhow!("fused QKV row count overflow"))?;
                anyhow::ensure!(
                    fused.info.rows == expected_rows && fused.info.cols == hidden,
                    "mmproj loader: fused '{fused_w}' is {:?} [{}, {}], expected [3*{hidden}, {hidden}]",
                    fused.info.ggml_dtype,
                    fused.info.rows,
                    fused.info.cols
                );
                (
                    fused.exact_row_range(0, hidden)?,
                    fused.exact_row_range(hidden, hidden)?,
                    fused.exact_row_range(
                        hidden
                            .checked_mul(2)
                            .ok_or_else(|| anyhow!("fused QKV V-row offset overflow"))?,
                        hidden,
                    )?,
                )
            };
            tensors.insert(split_q_w.clone(), q_weight.buffer.clone());
            tensors.insert(split_k_w.clone(), k_weight.buffer.clone());
            tensors.insert(split_v_w.clone(), v_weight.buffer.clone());
            matrix_weights.insert(split_q_w, q_weight);
            matrix_weights.insert(split_k_w, k_weight);
            matrix_weights.insert(split_v_w, v_weight);

            if has_fused_b {
                anyhow::ensure!(
                    f32_state_names.contains(&fused_b),
                    "mmproj loader: fused QKV bias '{fused_b}' is not admitted F32 state"
                );
                let fused_bias_buf = tensors
                    .get(&fused_b)
                    .expect("tensors.contains_key(&fused_b) contract violated by tensors.get");
                let bias_chunk_bytes = hidden
                    .checked_mul(DType::F32.size_of())
                    .ok_or_else(|| anyhow!("fused QKV bias chunk byte extent overflow"))?;
                let bias_expected = bias_chunk_bytes
                    .checked_mul(3)
                    .ok_or_else(|| anyhow!("fused QKV bias byte extent overflow"))?;
                anyhow::ensure!(
                    fused_bias_buf.dtype() == DType::F32
                        && fused_bias_buf.shape() == [3 * hidden]
                        && fused_bias_buf.data_byte_len() == bias_expected,
                    "mmproj loader: fused '{fused_b}' must be exact F32 [3*{hidden}] state (shape={:?}, logical_bytes={})",
                    fused_bias_buf.shape(),
                    fused_bias_buf.data_byte_len()
                );
                let k_offset = u64::try_from(bias_chunk_bytes)
                    .map_err(|_| anyhow!("fused QKV bias K offset exceeds u64"))?;
                let v_offset_bytes = bias_chunk_bytes
                    .checked_mul(2)
                    .ok_or_else(|| anyhow!("fused QKV bias V offset overflow"))?;
                let v_offset = u64::try_from(v_offset_bytes)
                    .map_err(|_| anyhow!("fused QKV bias V offset exceeds u64"))?;
                let q_bias = fused_bias_buf.slice_view(0, hidden);
                let k_bias = fused_bias_buf.slice_view(k_offset, hidden);
                let v_bias = fused_bias_buf.slice_view(v_offset, hidden);
                tensors.insert(split_q_b.clone(), q_bias);
                tensors.insert(split_k_b.clone(), k_bias);
                tensors.insert(split_v_b.clone(), v_bias);
                f32_state_names.insert(split_q_b);
                f32_state_names.insert(split_k_b);
                f32_state_names.insert(split_v_b);
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn install_fused_attn_qkv_slice_views(
        tensors: &mut HashMap<String, MlxBuffer>,
        cfg: &MmprojConfig,
    ) -> Result<()> {
        let mut matrix_weights = HashMap::new();
        let mut f32_state_names = HashSet::new();
        for (name, buffer) in tensors.iter() {
            if name.ends_with(".bias") {
                f32_state_names.insert(name.clone());
            } else if name.ends_with(".weight") {
                let [rows, cols] = buffer.shape() else {
                    return Err(anyhow!("test matrix '{name}' must be rank 2"));
                };
                matrix_weights.insert(
                    name.clone(),
                    MlxQWeight::from_test_buffer(buffer.clone(), GgmlType::F32, *rows, *cols),
                );
            }
        }
        Self::install_native_fused_attn_qkv_views(
            tensors,
            &mut matrix_weights,
            &mut f32_state_names,
            cfg,
        )
    }

    /// Load from a GGUF file path. Opens the file, creates a default
    /// MlxDevice, and loads every tensor. Convenience wrapper for the
    /// common startup path.
    pub fn load_from_path(path: &Path, cfg: &MmprojConfig) -> Result<Self> {
        let gguf = GgufFile::open(path)
            .map_err(|e| anyhow!("open mmproj GGUF {}: {e}", path.display()))?;
        let device =
            MlxDevice::new().map_err(|e| anyhow!("create MlxDevice for mmproj load: {e}"))?;
        Self::load(&gguf, cfg, device)
    }

    /// Look up a tensor by its GGUF name. `None` when absent (optional
    /// tensors like biases — callers gate the forward-pass branch on
    /// `Some`).
    pub fn get(&self, name: &str) -> Option<&MlxBuffer> {
        self.tensors.get(name)
    }

    /// Resolve an admitted native matrix by semantic role.
    pub fn matrix(&self, name: &str) -> Result<&MlxQWeight> {
        self.matrix_weights
            .get(name)
            .ok_or_else(|| anyhow!("mmproj matrix '{name}' is absent or has a non-matrix role"))
    }

    /// Resolve admitted F32 runtime state by semantic role.
    pub fn f32_state(&self, name: &str) -> Result<&MlxBuffer> {
        anyhow::ensure!(
            self.f32_state_names.contains(name),
            "mmproj F32 state '{name}' is absent or has a non-state role"
        );
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("mmproj F32 state index is inconsistent for '{name}'"))
    }

    pub fn storage_accounting(&self) -> MmprojStorageAccounting {
        self.storage
    }

    pub fn owned_bytes(&self) -> u64 {
        self.owned_bytes
    }

    pub fn mapped_segment_physical_bytes(&self) -> &[u64] {
        &self.mapped_segment_physical_bytes
    }

    /// CPU-oracle utility for F32/F16 buffers.
    ///
    /// Callers that need an `&[f32]` view (CPU patch reference or test
    /// parity) must explicitly convert. For an F32 buffer this helper
    /// allocates and copies, for an F16 buffer it widens via `half::f16`.
    /// It fails closed for BF16 and packed matrices. Production request
    /// forward code does not call it and uses native stored-format dispatch.
    ///
    /// Cost: O(N) heap allocation plus any F16 widening.
    ///
    /// # Errors
    ///
    /// Returns `Err` when (a) the tensor's `as_slice::<…>` fails for
    /// the underlying buffer's dtype, or (b) the dtype is anything
    /// other than F32 or F16.
    pub fn tensor_as_f32_owned(&self, buf: &MlxBuffer) -> Result<Vec<f32>> {
        use mlx_native::DType;
        match buf.dtype() {
            DType::F32 => buf
                .as_slice::<f32>()
                .map(|s| s.to_vec())
                .map_err(|e| anyhow!("tensor_as_f32_owned (F32): {e}")),
            DType::F16 => {
                let raw = buf
                    .as_slice::<u16>()
                    .map_err(|e| anyhow!("tensor_as_f32_owned (F16 u16 view): {e}"))?;
                Ok(raw
                    .iter()
                    .map(|&u| half::f16::from_bits(u).to_f32())
                    .collect())
            }
            other => Err(anyhow!(
                "tensor_as_f32_owned: unsupported dtype {other:?} (only F32/F16)"
            )),
        }
    }

    /// Build an empty `LoadedMmprojWeights` with no tensors. Useful for
    /// tests that need an `AppState.mmproj` shape but don't need to
    /// drive a forward pass. The shortcut accessors all return `Err`
    /// (as the real accessors would on a broken-producer file).
    pub fn empty(device: MlxDevice) -> Self {
        Self {
            tensors: HashMap::new(),
            matrix_weights: HashMap::new(),
            f32_state_names: HashSet::new(),
            storage: MmprojStorageAccounting::default(),
            mapped_segment_physical_bytes: Vec::new(),
            owned_bytes: 0,
            _device: device,
        }
    }

    /// Test-only: build a `LoadedMmprojWeights` from a pre-populated
    /// tensor map. Used by parity tests that synthesize block weights
    /// in-process rather than load a real GGUF (which would require a
    /// fixture file and its mapped Metal storage).
    #[cfg(test)]
    pub fn from_tensors_for_test(tensors: HashMap<String, MlxBuffer>, device: MlxDevice) -> Self {
        let mut roles = HashMap::with_capacity(tensors.len());
        let mut matrix_weights = HashMap::with_capacity(tensors.len());
        let mut f32_state_names = HashSet::with_capacity(tensors.len());
        for (name, buffer) in &tensors {
            if classify_mmproj_tensor_role(name).ok() == Some(MmprojTensorRole::Matrix) {
                let shape = buffer.shape();
                let (rows, cols) = if shape.len() == 2 {
                    (shape[0], shape[1])
                } else if matches!(
                    name.as_str(),
                    "v.patch_embd.weight" | "v.patch_embd.weight.1"
                ) && shape.len() > 2
                {
                    (shape[0], shape[1..].iter().copied().product())
                } else {
                    panic!("synthetic mmproj matrix '{name}' has invalid shape {shape:?}");
                };
                let codec = match buffer.dtype() {
                    DType::F32 => GgmlType::F32,
                    DType::F16 => GgmlType::F16,
                    DType::BF16 => GgmlType::BF16,
                    other => panic!(
                        "synthetic mmproj matrix '{name}' has ambiguous dtype {other:?}; construct its MlxQWeight explicitly"
                    ),
                };
                roles.insert(name.clone(), MmprojTensorRole::Matrix);
                matrix_weights.insert(
                    name.clone(),
                    MlxQWeight::from_test_buffer(buffer.clone(), codec, rows, cols),
                );
            } else {
                roles.insert(name.clone(), MmprojTensorRole::F32State);
                f32_state_names.insert(name.clone());
            }
        }
        let storage = summarize_source_storage(&tensors, &roles)
            .expect("synthetic mmproj storage accounting must not overflow");
        let owned_bytes = storage
            .total_backing_bytes()
            .expect("synthetic mmproj backing bytes must not overflow");
        Self {
            tensors,
            matrix_weights,
            f32_state_names,
            storage,
            mapped_segment_physical_bytes: Vec::new(),
            owned_bytes,
            _device: device,
        }
    }

    /// Test-only realized-storage receipt without allocating a production
    /// matrix. This lets serving tests mutate admission totals/topology while
    /// keeping production construction restricted to real mapped resources.
    #[cfg(test)]
    pub(crate) fn empty_with_mapped_storage_for_test(
        device: MlxDevice,
        mapped_segment_physical_bytes: Vec<u64>,
    ) -> Self {
        let owned_bytes = mapped_segment_physical_bytes
            .iter()
            .try_fold(0u64, |total, &bytes| total.checked_add(bytes))
            .expect("synthetic mapped storage bytes must not overflow");
        Self {
            tensors: HashMap::new(),
            matrix_weights: HashMap::new(),
            f32_state_names: HashSet::new(),
            storage: MmprojStorageAccounting {
                unique_backing_allocations: mapped_segment_physical_bytes.len(),
                mapped_backing_bytes: owned_bytes,
                ..MmprojStorageAccounting::default()
            },
            mapped_segment_physical_bytes,
            owned_bytes,
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

    pub fn patch_embd_matrix(&self) -> Result<&MlxQWeight> {
        self.matrix(super::mmproj::TENSOR_PATCH_EMBD)
    }

    pub fn patch_embd_matrix_1(&self) -> Result<&MlxQWeight> {
        self.matrix("v.patch_embd.weight.1")
    }

    pub fn position_embd_weight(&self) -> Result<&MlxBuffer> {
        self.f32_state(super::mmproj::TENSOR_POS_EMBD)
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
        if buf.data_byte_len() != expected_bytes {
            return Err(anyhow!(
                "v.position_embd.weight: logical byte extent {} != expected {} (2 * {} * {} * 4)",
                buf.data_byte_len(),
                expected_bytes,
                pos_size,
                hidden
            ));
        }
        Ok((buf, pos_size, hidden))
    }

    pub fn post_ln_weight(&self) -> Result<&MlxBuffer> {
        self.f32_state(super::mmproj::TENSOR_POST_LN_WEIGHT)
    }

    /// Per-block tensor accessor.
    ///
    /// `suffix` is the block-relative name ("attn_q.weight",
    /// "ffn_down.weight", etc. — see `BLOCK_REQUIRED_SUFFIXES`).
    ///
    /// W41 iter-116i: vision-namespace tensor names migrated to
    /// the peer's short-form convention in W34 iter-116e (writer
    /// side) but the runtime forward path still uses the
    /// pre-migration suffixes. `block_tensor` accepts both: if the
    /// caller asks for a legacy name we fall back to the canonical
    /// short form. The mapping is bidirectional so a producer
    /// emitting either convention loads cleanly.
    ///
    /// Mappings (legacy ↔ canonical short form, per the peer's mmproj
    /// naming):
    ///   - `attn_output.{w,b}`  ↔ `attn_out.{w,b}`
    ///   - `post_ffw_norm.{w,b}` ↔ `ffn_post_norm.{w,b}`
    pub fn block_tensor(&self, layer_idx: usize, suffix: &str) -> Result<&MlxBuffer> {
        let key = vit_layer_tensor(layer_idx, suffix);
        if let Some(b) = self.tensors.get(&key) {
            return Ok(b);
        }
        // Try the legacy/canonical alias.
        let alias_suffix = block_alias_suffix(suffix);
        if let Some(alt) = alias_suffix {
            let alt_key = vit_layer_tensor(layer_idx, alt);
            if let Some(b) = self.tensors.get(&alt_key) {
                return Ok(b);
            }
        }
        Err(anyhow!("mmproj missing '{}'", key))
    }

    pub fn block_matrix(&self, layer_idx: usize, suffix: &str) -> Result<&MlxQWeight> {
        let key = vit_layer_tensor(layer_idx, suffix);
        if let Some(weight) = self.matrix_weights.get(&key) {
            return Ok(weight);
        }
        if let Some(alias) = block_alias_suffix(suffix) {
            let alias_key = vit_layer_tensor(layer_idx, alias);
            if let Some(weight) = self.matrix_weights.get(&alias_key) {
                return Ok(weight);
            }
        }
        Err(anyhow!(
            "mmproj matrix '{}' is absent or has a non-matrix role",
            key
        ))
    }

    pub fn block_f32_state(&self, layer_idx: usize, suffix: &str) -> Result<&MlxBuffer> {
        let key = vit_layer_tensor(layer_idx, suffix);
        if let Ok(state) = self.f32_state(&key) {
            return Ok(state);
        }
        if let Some(alias) = block_alias_suffix(suffix) {
            let alias_key = vit_layer_tensor(layer_idx, alias);
            if let Ok(state) = self.f32_state(&alias_key) {
                return Ok(state);
            }
        }
        Err(anyhow!(
            "mmproj F32 state '{}' is absent or has a non-state role",
            key
        ))
    }

    /// Projector head weight tensor.
    ///
    /// W41 iter-116i: looks up the CLIP-classic name `mm.0.weight` first,
    /// then falls back to gemma4v's `mm.input_projection.weight`.
    /// Both name back the same logical tensor — the writer chose the
    /// projector-specific base per the peer's convention (which
    /// hard-requires `mm.input_projection` for the gemma4v projector
    /// type), and the runtime forward path uses whichever is present.
    ///
    /// The accessor name is preserved (`mm_0_weight`) for source-compat
    /// across `vit_gpu.rs` callers; the fallback is invisible to them.
    pub fn mm_0_weight(&self) -> Result<&MlxBuffer> {
        if let Some(b) = self.tensors.get(super::mmproj::TENSOR_MM_0_WEIGHT) {
            return Ok(b);
        }
        if let Some(b) = self
            .tensors
            .get(super::mmproj::TENSOR_MM_INPUT_PROJECTION_WEIGHT)
        {
            return Ok(b);
        }
        Err(anyhow!(
            "mmproj missing '{}' (and gemma4v fallback '{}')",
            super::mmproj::TENSOR_MM_0_WEIGHT,
            super::mmproj::TENSOR_MM_INPUT_PROJECTION_WEIGHT,
        ))
    }

    pub fn mm_0_matrix(&self) -> Result<&MlxQWeight> {
        if let Some(weight) = self.matrix_weights.get(super::mmproj::TENSOR_MM_0_WEIGHT) {
            return Ok(weight);
        }
        if let Some(weight) = self
            .matrix_weights
            .get(super::mmproj::TENSOR_MM_INPUT_PROJECTION_WEIGHT)
        {
            return Ok(weight);
        }
        Err(anyhow!(
            "mmproj matrix missing '{}' (and gemma4v fallback '{}')",
            super::mmproj::TENSOR_MM_0_WEIGHT,
            super::mmproj::TENSOR_MM_INPUT_PROJECTION_WEIGHT,
        ))
    }

    pub fn mm_2_weight(&self) -> Result<&MlxBuffer> {
        self.tensors
            .get(super::mmproj::TENSOR_MM_2_WEIGHT)
            .ok_or_else(|| anyhow!("mmproj missing '{}'", super::mmproj::TENSOR_MM_2_WEIGHT))
    }

    pub fn mm_2_matrix(&self) -> Result<&MlxQWeight> {
        self.matrix(super::mmproj::TENSOR_MM_2_WEIGHT)
    }

    // -----------------------------------------------------------------------
    // Gemma4ClippableLinear scalar bounds for `mm.0.weight`.
    //
    // Gemma4v mmproj files carry four optional scalar f32 tensors as
    // siblings of `mm.0.weight`:
    //   - `mm.0.input_min`, `mm.0.input_max` (clamps applied BEFORE matmul)
    //   - `mm.0.output_min`, `mm.0.output_max` (clamps applied AFTER matmul)
    //
    // Each is a 1-element f32 tensor (the converter `unsqueeze(0)`s the
    // 0-D scalar so GGUF round-trips it as a 1-D `[1]` tensor).
    //
    // Returns `Some(value)` when the tensor is present and decodes to
    // exactly one f32, else `None`. Callers compose the four into a
    // `Gemma4ClippableLinearBounds` via `mm_0_bounds()`.
    // -----------------------------------------------------------------------

    fn read_scalar_f32(&self, name: &str) -> Option<f32> {
        let buf = self.f32_state(name).ok()?;
        let slice = buf.as_slice::<f32>().ok()?;
        // Defensive: clamp scalars are 1-element. If we ever load a
        // mis-shaped sibling (e.g. converter wrote a vector), reject
        // cleanly rather than silently picking element 0.
        if slice.len() != 1 {
            return None;
        }
        Some(slice[0])
    }

    /// Read a clamp-scalar bound under either the CLIP-classic
    /// `mm.0.<suffix>` or the gemma4v `mm.input_projection.<suffix>`
    /// base name. W41 iter-116i: gemma4v's projector head is named
    /// `mm.input_projection`,
    /// so the optional clamp scalars share that base. Returns the
    /// first match in (`mm.0`, `mm.input_projection`) order.
    fn read_projector_scalar(&self, suffix: &str) -> Option<f32> {
        let mm0 = format!("mm.0.{suffix}");
        if let Some(v) = self.read_scalar_f32(&mm0) {
            return Some(v);
        }
        let mm_inp = format!("mm.input_projection.{suffix}");
        self.read_scalar_f32(&mm_inp)
    }

    /// Read the `mm.0.input_min` (or gemma4v's `mm.input_projection.input_min`)
    /// scalar bound (clamp BEFORE matmul). `None` when absent OR
    /// mis-shaped — caller treats absence as `f32::NEG_INFINITY` (no-op)
    /// per the peer's default.
    pub fn mm_0_input_min(&self) -> Option<f32> {
        self.read_projector_scalar("input_min")
    }
    /// See `mm_0_input_min`.
    pub fn mm_0_input_max(&self) -> Option<f32> {
        self.read_projector_scalar("input_max")
    }
    /// Read the output_min scalar bound (clamp AFTER matmul).
    pub fn mm_0_output_min(&self) -> Option<f32> {
        self.read_projector_scalar("output_min")
    }
    /// See `mm_0_output_min`.
    pub fn mm_0_output_max(&self) -> Option<f32> {
        self.read_projector_scalar("output_max")
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
    use super::super::mmproj::ProjectorType;
    use super::*;

    /// Gemma 4 26B mmproj — present on this dev machine. Tests gate on
    /// existence so CI without the fixture skips them cleanly.
    const GEMMA4_MMPROJ_PATH: &str =
        "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq-mmproj.gguf";

    fn info(name: &str, shape: Vec<usize>, ggml_type: GgmlType, byte_len: usize) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            shape,
            ggml_type,
            offset: 0,
            byte_len,
        }
    }

    #[test]
    fn role_classifier_separates_native_matrices_from_f32_state() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        for name in [
            "v.patch_embd.bias",
            "v.position_embd.weight",
            "v.blk.0.ln1.weight",
            "v.blk.0.attn_norm.weight",
            "v.blk.0.attn_q_norm.weight",
            "v.blk.0.attn_k_norm.weight",
            "v.blk.0.attn_post_norm.weight",
            "v.deepstack.7.norm.weight",
            "v.std_bias",
            "v.std_scale",
            "mm.input_norm.weight",
            "mm.input_projection.input_min",
        ] {
            assert_eq!(
                classify_mmproj_tensor_role(name).expect("known F32 state"),
                MmprojTensorRole::F32State,
                "{name}"
            );
        }
        for name in [
            "v.patch_embd.weight",
            "v.patch_embd.weight.1",
            "v.blk.0.attn_qkv.weight",
            "v.blk.0.ffn_up.weight",
            "v.deepstack.7.fc1.weight",
            "mm.0.weight",
        ] {
            assert_eq!(
                classify_mmproj_tensor_role(name).expect("known matrix"),
                MmprojTensorRole::Matrix,
                "{name}"
            );
        }
        assert!(classify_mmproj_tensor_role("v.future.untyped_payload").is_err());
    }

    #[test]
    fn matrix_codec_admission_is_exact_and_fail_closed() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        for (codec, cols) in [
            (GgmlType::F32, 7usize),
            (GgmlType::F16, 7),
            (GgmlType::BF16, 7),
            (GgmlType::Q4_0, 32),
            (GgmlType::Q5_0, 32),
            (GgmlType::Q8_0, 32),
            (GgmlType::Q4_K, 256),
            (GgmlType::Q5_K, 256),
            (GgmlType::Q6_K, 256),
        ] {
            let bytes = native_gguf_matrix_bytes(codec, 3, cols).expect("admitted codec bytes");
            validate_mmproj_tensor_role_codec(
                &info("v.blk.0.ffn_up.weight", vec![3, cols], codec, bytes),
                MmprojTensorRole::Matrix,
            )
            .unwrap_or_else(|error| panic!("{codec:?} must be admitted: {error}"));
        }

        let unsupported = info("v.blk.0.ffn_up.weight", vec![3, 256], GgmlType::I16, 1);
        let error = validate_mmproj_tensor_role_codec(&unsupported, MmprojTensorRole::Matrix)
            .expect_err("unsupported matrix codec must fail closed");
        assert!(format!("{error}").contains("not admitted"));

        let quantized_state = info("v.blk.0.ln1.weight", vec![256], GgmlType::Q4_0, 144);
        let error = validate_mmproj_tensor_role_codec(&quantized_state, MmprojTensorRole::F32State)
            .expect_err("quantized runtime state must fail closed");
        assert!(format!("{error}").contains("must be stored as F32"));

        let exact = native_gguf_matrix_bytes(GgmlType::Q5_K, 3, 256).unwrap();
        let oversized = info(
            "v.blk.0.ffn_up.weight",
            vec![3, 256],
            GgmlType::Q5_K,
            exact + 1,
        );
        assert!(validate_mmproj_tensor_role_codec(&oversized, MmprojTensorRole::Matrix).is_err());
    }

    #[test]
    fn rank_greater_than_two_flattens_only_patch_matrices() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        for (codec, tail) in [
            (GgmlType::F32, vec![2usize, 2, 2]),
            (GgmlType::F16, vec![2, 2, 2]),
            (GgmlType::BF16, vec![2, 2, 2]),
            (GgmlType::Q4_0, vec![2, 2, 8]),
            (GgmlType::Q5_0, vec![2, 2, 8]),
            (GgmlType::Q8_0, vec![2, 2, 8]),
            (GgmlType::Q4_K, vec![2, 8, 16]),
            (GgmlType::Q5_K, vec![2, 8, 16]),
            (GgmlType::Q6_K, vec![2, 8, 16]),
        ] {
            let rows = 3usize;
            let cols = tail.iter().product::<usize>();
            let mut shape = vec![rows];
            shape.extend(tail);
            let bytes = native_gguf_matrix_bytes(codec, rows, cols).unwrap();
            let patch = info("v.patch_embd.weight", shape, codec, bytes);
            assert_eq!(matrix_shape(&patch).unwrap(), (rows, cols), "{codec:?}");
            validate_mmproj_tensor_role_codec(&patch, MmprojTensorRole::Matrix)
                .unwrap_or_else(|error| panic!("{codec:?} patch view: {error}"));
        }

        let projection = info(
            "v.blk.0.attn_q.weight",
            vec![2, 32, 32],
            GgmlType::F16,
            2 * 32 * 32 * 2,
        );
        let error = matrix_shape(&projection).expect_err("non-patch rank-3 must fail");
        assert!(format!("{error}").contains("only patch weights"));
    }

    #[test]
    fn production_loader_has_no_transform_fallback_canary() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let source = include_str!("mmproj_weights.rs");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("production source prefix");
        assert!(production.contains(".map_tensor_data("));
        assert!(production.contains("from_mapped_gguf_matrix_view"));
        assert!(!production.contains(".load_tensor_f32("));
        assert!(!production.contains("dispatch_dequant"));
    }

    #[test]
    fn production_vision_source_wiring_smoke_has_native_role_typed_callsites() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        fn between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
            source
                .split_once(start)
                .unwrap_or_else(|| panic!("missing source start marker: {start}"))
                .1
                .split_once(end)
                .unwrap_or_else(|| panic!("missing source end marker: {end}"))
                .0
        }

        let vit = include_str!("vit_gpu.rs");
        let linear = between(
            vit,
            "pub(crate) fn vit_linear_gpu",
            "fn vit_patch_embed_native_gpu",
        );
        for token in ["GgmlType::F32", "GgmlType::F16", "GgmlType::BF16"] {
            assert!(linear.contains(token), "missing scalar route {token}");
        }
        for token in [
            "GgmlType::Q4_0",
            "GgmlType::Q5_0",
            "GgmlType::Q8_0",
            "GgmlType::Q4_K",
            "GgmlType::Q5_K",
            "GgmlType::Q6_K",
        ] {
            assert!(linear.contains(token), "missing quantized route {token}");
        }
        assert!(linear.contains("quantized_matmul_ggml"));
        assert!(!linear.contains("F32ToBF16"));
        assert!(!linear.contains("dispatch_dequant"));

        let classic = between(
            vit,
            "pub fn apply_vit_full_forward_gpu",
            "pub fn warmup_vit_gpu",
        );
        assert!(classic.contains("patch_embd_matrix"));
        assert!(classic.contains("mm_0_matrix"));
        assert!(classic.contains("f32_state(\"v.std_bias\")"));
        assert!(!classic.contains("tensor_as_f32_owned"));
        assert!(!classic.contains("patch_embed_cpu"));

        let classic_block = between(
            vit,
            "pub fn apply_vit_block_forward_gpu",
            "pub fn apply_vit_blocks_loop_gpu",
        );
        assert!(classic_block.contains(".block_matrix("));
        assert!(classic_block.contains(".block_f32_state("));
        assert!(!classic_block.contains(".block_tensor("));

        let gemma_block = between(
            vit,
            "pub fn gemma4v_block_forward_gpu",
            "pub(crate) fn gemma4v_clippable_linear_gpu",
        );
        assert!(gemma_block.contains(".block_matrix("));
        assert!(gemma_block.contains(".block_f32_state("));
        assert!(!gemma_block.contains(".block_tensor("));

        let gemma_forward = between(
            vit,
            "pub fn gemma4v_apply_full_forward_gpu",
            "#[cfg(test)]\nmod tests",
        );
        assert!(gemma_forward.contains(".patch_embd_matrix("));
        assert!(gemma_forward.contains(".mm_0_matrix("));
        assert!(!gemma_forward.contains(".patch_embd_weight("));
        assert!(!gemma_forward.contains(".mm_0_weight("));
        assert!(!gemma_forward.contains("tensor_as_f32_owned"));

        let qwen = include_str!("vit_gpu_qwen.rs");
        let stage_a = between(
            qwen,
            "fn qwen_vision_stage_a_dispatch",
            "pub(crate) fn qwen_vision_resize_position_embeddings_bilinear",
        );
        assert!(stage_a.contains("weight_0: &MlxQWeight"));
        assert!(stage_a.contains("weight_1: &MlxQWeight"));
        assert!(stage_a.contains("vit_linear_gpu"));
        assert!(!stage_a.contains("weight_0: &[f32]"));
        assert!(!stage_a.contains("alloc weight_0"));
        assert!(!stage_a.contains("tensor_as_f32_owned"));

        let qwen_block = between(
            qwen,
            "fn apply_qwen_vision_block_forward_gpu",
            "fn apply_qwen_vision_deepstack_head_gpu",
        );
        assert!(qwen_block.contains(".block_matrix("));
        assert!(qwen_block.contains(".block_f32_state("));
        assert!(!qwen_block.contains(".block_tensor("));

        let qwen_forward = between(
            qwen,
            "pub fn compute_vision_embeddings_gpu_qwen",
            "#[cfg(test)]\nmod tests",
        );
        assert!(qwen_forward.contains("patch_embd_matrix"));
        assert!(qwen_forward.contains("patch_embd_matrix_1"));
        assert!(!qwen_forward.contains("tensor_as_f32_owned"));
        assert!(!qwen_forward.contains("patch_embd_f32"));
    }

    #[test]
    fn backing_storage_accounting_deduplicates_aliases() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let base = device
            .alloc_buffer(64, DType::F32, vec![16])
            .expect("base allocation");
        let allocation_bytes = base.byte_len();
        let mut tensors = HashMap::new();
        tensors.insert("v.blk.0.attn_q.weight".to_string(), base.clone());
        tensors.insert("v.blk.0.attn_k.weight".to_string(), base.slice_view(16, 4));
        tensors.insert("v.blk.0.attn_v.weight".to_string(), base.slice_view(32, 4));
        let roles = tensors
            .keys()
            .map(|name| (name.clone(), MmprojTensorRole::Matrix))
            .collect();
        let storage = summarize_source_storage(&tensors, &roles).expect("storage receipt");
        assert_eq!(storage.unique_backing_allocations, 1);
        assert_eq!(storage.anonymous_backing_bytes, allocation_bytes as u64);
        assert_eq!(storage.mapped_backing_bytes, 0);
        assert_eq!(storage.source_logical_bytes, (64 + 16 + 16) as u64);
    }

    #[test]
    fn fused_qkv_native_views_are_codec_aware_and_zero_copy() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        for (codec, hidden, dtype) in [
            (GgmlType::F32, 8usize, DType::F32),
            (GgmlType::F16, 8, DType::F16),
            (GgmlType::BF16, 8, DType::BF16),
            (GgmlType::Q4_0, 32, DType::U8),
            (GgmlType::Q5_0, 32, DType::U8),
            (GgmlType::Q8_0, 32, DType::U8),
            (GgmlType::Q4_K, 256, DType::U8),
            (GgmlType::Q5_K, 256, DType::U8),
            (GgmlType::Q6_K, 256, DType::U8),
        ] {
            let rows = hidden * 3;
            let bytes = native_gguf_matrix_bytes(codec, rows, hidden).unwrap();
            let elements = bytes / dtype.size_of();
            let buffer = device
                .alloc_buffer(bytes, dtype, vec![elements])
                .unwrap_or_else(|error| panic!("allocate {codec:?}: {error}"));
            let base_ptr = buffer.contents_ptr();
            let mut tensors =
                HashMap::from([("v.blk.0.attn_qkv.weight".to_string(), buffer.clone())]);
            let mut matrices = HashMap::from([(
                "v.blk.0.attn_qkv.weight".to_string(),
                MlxQWeight::from_test_buffer(buffer, codec, rows, hidden),
            )]);
            let mut state = HashSet::new();
            let cfg = synth_qwen_vision_loader_cfg(hidden as u32, 1);
            LoadedMmprojWeights::install_native_fused_attn_qkv_views(
                &mut tensors,
                &mut matrices,
                &mut state,
                &cfg,
            )
            .unwrap_or_else(|error| panic!("slice {codec:?}: {error}"));
            let row_bytes = native_gguf_matrix_bytes(codec, 1, hidden).unwrap();
            for (suffix, row_start) in [
                ("attn_q.weight", 0usize),
                ("attn_k.weight", hidden),
                ("attn_v.weight", hidden * 2),
            ] {
                let alias = matrices
                    .get(&format!("v.blk.0.{suffix}"))
                    .unwrap_or_else(|| panic!("missing {codec:?} {suffix}"));
                assert_eq!(alias.info.ggml_dtype, codec);
                assert_eq!(alias.info.rows, hidden);
                assert_eq!(alias.info.cols, hidden);
                assert_eq!(alias.buffer.contents_ptr(), base_ptr);
                assert_eq!(alias.buffer.byte_offset(), (row_start * row_bytes) as u64);
                assert_eq!(
                    alias.buffer.data_byte_len(),
                    native_gguf_matrix_bytes(codec, hidden, hidden).unwrap()
                );
            }
        }
    }

    #[test]
    fn load_gemma4_mmproj_populates_arch_tensors() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Real Gemma 4 mmproj (SigLIP variant): 356 tensors total —
        //   5 non-block (patch_embd, pos_embd, std_bias, std_scale, mm.0.weight)
        //   13/block × 27 blocks = 351
        //   No v.post_ln.weight, no mm.2.weight.
        // See /opt/hf2q/docs/ADR-005 iter 31 for the real tensor manifest.
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!(
                "skipping: mmproj fixture not found at {}",
                GEMMA4_MMPROJ_PATH
            );
            return;
        }
        let gguf = GgufFile::open(path).expect("open gemma4 mmproj");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("parse mmproj config");
        // Sanity: Gemma 4's 27-layer SigLIP at 224×224 with 16×16 patches.
        assert_eq!(cfg.num_hidden_layers, 27);
        assert_eq!(cfg.image_size, 224);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.hidden_size, 1152);
        // W41 iter-116i: hf2q-emitted gemma4 mmproj writes
        // `clip.projector_type = "gemma4v"` (matches the peer's
        // projector-type literal).
        // Pre-iter-116i the loader parsed this to `Other("gemma4v")`
        // and `is_supported()` returned false, blocking serve startup.
        assert_eq!(cfg.projector, ProjectorType::Gemma4v);

        let device = MlxDevice::new().expect("create device");
        let weights = LoadedMmprojWeights::load(&gguf, &cfg, device).expect("load weights");
        // Gemma 4 mmproj has 356 tensors total.
        assert_eq!(weights.len(), 356);
        // Arch-agnostic shortcuts present.
        weights.patch_embd_weight().expect("patch_embd_weight");
        weights
            .position_embd_weight()
            .expect("position_embd_weight");
        weights.mm_0_weight().expect("mm_0_weight");
        weights
            .matrix(super::super::mmproj::TENSOR_PATCH_EMBD)
            .expect("role-typed patch matrix");
        weights
            .f32_state(super::super::mmproj::TENSOR_POS_EMBD)
            .expect("role-typed position state");
        assert!(weights
            .matrix(super::super::mmproj::TENSOR_POS_EMBD)
            .is_err());
        let storage = weights.storage_accounting();
        assert_eq!(
            storage.source_matrix_count + storage.source_f32_state_count,
            storage.source_tensor_count
        );
        assert!(weights.owned_bytes() > 0);
        // post_ln + mm.2 do NOT exist in Gemma 4 mmproj.
        assert!(weights.post_ln_weight().is_err());
        assert!(weights.mm_2_weight().is_err());
        // Every layer's arch-agnostic QKV+output suffixes present.
        // W41/W42 iter-116i: vision-namespace short-form `attn_out`
        // (`v.blk.{N}.attn_out.*` per the peer's naming); W34 iter-116e
        // fixed the writer to emit this short form and `validate_tensor_set`
        // requires the same. The pre-iter-116e long-form
        // `attn_output.weight` is no longer present.
        for layer_idx in 0..27 {
            for suffix in [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_out.weight",
            ] {
                weights
                    .block_tensor(layer_idx, suffix)
                    .unwrap_or_else(|_| panic!("layer {} {}", layer_idx, suffix));
            }
        }
    }

    #[test]
    #[ignore = "requires HF2Q_QWEN_VISION_MMPROJ and Apple Metal"]
    fn real_mapping_plan_equals_realized_storage_and_fused_aliases_add_no_bytes() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let path = std::env::var("HF2Q_QWEN_VISION_MMPROJ")
            .expect("set HF2Q_QWEN_VISION_MMPROJ to a fused-QKV Qwen projector");
        let gguf = GgufFile::open(Path::new(&path)).expect("open Qwen projector");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("parse projector config");
        let device = MlxDevice::new().expect("Metal device");
        let plan = gguf
            .mapped_tensor_storage_plan(device.metal_device())
            .expect("plan mapped storage");
        let expected_segments = plan
            .segment_physical_byte_lens()
            .iter()
            .map(|&bytes| u64::try_from(bytes).expect("segment bytes fit u64"))
            .collect::<Vec<_>>();
        let expected_total = u64::try_from(plan.physical_byte_len()).expect("total fits u64");

        let weights = LoadedMmprojWeights::load(&gguf, &cfg, device).expect("map projector");
        assert_eq!(weights.owned_bytes(), expected_total);
        assert_eq!(
            weights.mapped_segment_physical_bytes(),
            expected_segments.as_slice()
        );
        assert!(
            weights.storage_accounting().alias_view_count >= 3,
            "fused QKV fixture must install at least Q/K/V aliases"
        );

        let fused = weights
            .matrix("v.blk.0.attn_qkv.weight")
            .expect("fused QKV matrix");
        for suffix in ["attn_q.weight", "attn_k.weight", "attn_v.weight"] {
            let alias = weights
                .matrix(&format!("v.blk.0.{suffix}"))
                .unwrap_or_else(|error| panic!("missing fused alias {suffix}: {error}"));
            assert_eq!(alias.buffer.contents_ptr(), fused.buffer.contents_ptr());
        }
        assert_eq!(weights.owned_bytes(), expected_total);
    }

    #[test]
    fn load_gemma4_mmproj_patch_embd_has_expected_shape_and_values() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // `v.patch_embd.weight` in Gemma 4 is a 2D tensor [hidden,
        // 3*patch*patch] = [1152, 768] = 884,736 elements. The GGUF
        // stores it as F16 (per W58 iter-127 audit + gguf_dump.py
        // verification). The loader retains the exact F16 file bytes so
        // Phase 2 can dispatch the matching native matrix kernel.
        //
        // This test asserts shape (element count) AND non-trivial
        // content (non-zero patch weights) without depending on the
        // storage dtype.
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!(
                "skipping: mmproj fixture not found at {}",
                GEMMA4_MMPROJ_PATH
            );
            return;
        }
        let gguf = GgufFile::open(path).expect("open gemma4 mmproj");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("parse mmproj config");
        let device = MlxDevice::new().expect("create device");
        let weights = LoadedMmprojWeights::load(&gguf, &cfg, device).expect("load weights");
        let patch = weights.patch_embd_weight().expect("patch_embd");
        let expected_elems =
            (cfg.hidden_size as usize) * 3 * (cfg.patch_size as usize) * (cfg.patch_size as usize);
        // Element-count check is dtype-agnostic via element_count(); it
        // matches expected_elems regardless of F16/F32 storage.
        assert_eq!(
            patch.element_count(),
            expected_elems,
            "patch_embd element count"
        );
        // Non-zero sanity: read the underlying bytes (works for both
        // F16 and F32). For F16 storage, every f16 has ≥1 nonzero bit
        // when its value is nonzero; for F32 the same holds. A patch
        // weight tensor with the first 1024 elements all zero would be
        // a load bug — assert that fewer than 95% of the first 2048
        // bytes (= 1024 f16 OR 512 f32) are zero.
        let raw: &[u8] = patch.as_slice().expect("as_slice raw bytes");
        let scan = raw.len().min(2048);
        let nonzero = raw[..scan].iter().filter(|&&b| b != 0).count();
        assert!(
            nonzero > scan * 5 / 100,
            "patch_embd loads to mostly-zero bytes (probable load bug): \
             {nonzero}/{scan} nonzero in first {scan} bytes"
        );
        // Dtype-specific spot-check: the gemma4v patch_embd is F16 in
        // storage; if the iter-128 path is wired correctly the buffer
        // dtype reflects that. (For SigLIP / classic CLIP producers
        // the tensor may be F32 instead — both paths are valid here;
        // we just print so the test record shows which one was loaded.)
        eprintln!(
            "patch_embd dtype: {:?}, element_count: {}",
            patch.dtype(),
            patch.element_count()
        );
    }

    #[test]
    fn load_from_path_wraps_gguf_open_and_device_create() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let path = Path::new(GEMMA4_MMPROJ_PATH);
        if !path.exists() {
            eprintln!(
                "skipping: mmproj fixture not found at {}",
                GEMMA4_MMPROJ_PATH
            );
            return;
        }
        let gguf = GgufFile::open(path).expect("open for cfg");
        let cfg = MmprojConfig::from_gguf(&gguf).expect("cfg");
        let weights = LoadedMmprojWeights::load_from_path(path, &cfg).expect("load_from_path");
        assert_eq!(weights.len(), 356);
    }

    #[test]
    fn accessors_return_err_with_specific_name_when_missing() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Synthetic LoadedMmprojWeights with an empty tensor map — every
        // accessor should return Err naming the missing tensor.
        let weights = LoadedMmprojWeights::empty(MlxDevice::new().expect("device"));
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
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let weights = LoadedMmprojWeights::empty(MlxDevice::new().expect("device"));
        assert_eq!(weights.len(), 0);
        assert!(weights.is_empty());
    }

    #[test]
    fn get_returns_none_for_absent_tensor() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let weights = LoadedMmprojWeights::empty(MlxDevice::new().expect("device"));
        assert!(weights.get("v.patch_embd.weight").is_none());
    }

    #[test]
    fn position_embd_table_3d_rejects_non_3d_shape() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Synthesize a LoadedMmprojWeights with a 2-D position-embd
        // (the SigLIP shape). The 3-D accessor must reject it cleanly.
        let device = MlxDevice::new().expect("device");
        let buf = device
            .alloc_buffer(64 * 4, mlx_native::DType::F32, vec![8, 8])
            .expect("alloc");
        let mut tensors = HashMap::new();
        tensors.insert(super::super::mmproj::TENSOR_POS_EMBD.to_string(), buf);
        let weights = LoadedMmprojWeights::from_tensors_for_test(tensors, device);
        let err = weights.position_embd_table_3d().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("expected 3-D"), "wrong error msg: {msg}");
    }

    #[test]
    fn position_embd_table_3d_rejects_first_dim_not_two() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let buf = device
            .alloc_buffer(96 * 4, mlx_native::DType::F32, vec![3, 4, 8])
            .expect("alloc");
        let mut tensors = HashMap::new();
        tensors.insert(super::super::mmproj::TENSOR_POS_EMBD.to_string(), buf);
        let weights = LoadedMmprojWeights::from_tensors_for_test(tensors, device);
        let err = weights.position_embd_table_3d().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("first dim 2"), "wrong error msg: {msg}");
    }

    #[test]
    fn position_embd_table_3d_returns_dims_for_valid_shape() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let pos_size = 27usize;
        let hidden = 1152usize;
        let buf = device
            .alloc_buffer(
                2 * pos_size * hidden * 4,
                mlx_native::DType::F32,
                vec![2, pos_size, hidden],
            )
            .expect("alloc");
        let mut tensors = HashMap::new();
        tensors.insert(super::super::mmproj::TENSOR_POS_EMBD.to_string(), buf);
        let weights = LoadedMmprojWeights::from_tensors_for_test(tensors, device);
        let (buf, ps, h) = weights.position_embd_table_3d().expect("3d ok");
        assert_eq!(ps, pos_size as u32);
        assert_eq!(h, hidden as u32);
        assert_eq!(buf.shape(), &[2, pos_size, hidden]);
    }

    #[test]
    fn position_embd_table_3d_propagates_missing_tensor_error() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let weights = LoadedMmprojWeights::empty(device);
        let err = weights.position_embd_table_3d().unwrap_err();
        assert!(format!("{err}").contains("v.position_embd.weight"));
    }

    #[test]
    fn empty_constructor_produces_zero_tensor_weights() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
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
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use mlx_native::DType;
        let device = MlxDevice::new().expect("device");
        let put =
            |tensors: &mut HashMap<String, MlxBuffer>, dev: &MlxDevice, name: &str, value: f32| {
                // 1-element f32 tensor with shape [1] — matches what
                // convert_hf_to_gguf.py emits for the unsqueeze(0)'d scalar.
                let buf = dev
                    .alloc_buffer(4, DType::F32, vec![1])
                    .expect("alloc scalar");
                let s: &mut [f32] =
                    unsafe { std::slice::from_raw_parts_mut(buf.contents_ptr() as *mut f32, 1) };
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
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let weights = LoadedMmprojWeights::empty(device);
        assert_eq!(weights.mm_0_input_min(), None);
        assert_eq!(weights.mm_0_input_max(), None);
        assert_eq!(weights.mm_0_output_min(), None);
        assert_eq!(weights.mm_0_output_max(), None);
        let bounds = weights.mm_0_bounds();
        assert!(!bounds.any());
    }

    // -----------------------------------------------------------------------
    // Wedge-4c.5: install_fused_attn_qkv_slice_views.
    // -----------------------------------------------------------------------

    /// Build a small synthetic Qwen3-VL-style MmprojConfig that
    /// `install_fused_attn_qkv_slice_views` consumes for hidden_size +
    /// num_hidden_layers. Other fields don't affect the slice-view path.
    fn synth_qwen_vision_loader_cfg(hidden: u32, num_layers: u32) -> MmprojConfig {
        MmprojConfig {
            image_size: 32,
            patch_size: 16,
            num_patches_side: 2,
            hidden_size: hidden,
            intermediate_size: hidden * 4,
            num_attention_heads: 4,
            num_hidden_layers: num_layers,
            layer_norm_eps: 1e-6,
            projector: super::super::mmproj::ProjectorType::QwenVisionMerger,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            image_min_pixels: None,
            image_max_pixels: None,
            spatial_merge_size: Some(2),
            projection_dim: Some(hidden),
            deepstack_indexes: Some(vec![]),
        }
    }

    /// Allocate an MlxBuffer of `n_elements` F32 values, populated by
    /// `f(i) -> f32` for i in 0..n_elements.
    fn alloc_f32_with<F: FnMut(usize) -> f32>(
        device: &MlxDevice,
        n_elements: usize,
        shape: Vec<usize>,
        mut f: F,
    ) -> MlxBuffer {
        use mlx_native::DType;
        let mut buf = device
            .alloc_buffer(n_elements * 4, DType::F32, shape)
            .expect("alloc f32 buffer");
        {
            let dst: &mut [f32] = buf.as_mut_slice::<f32>().expect("as_mut_slice f32");
            for (i, slot) in dst.iter_mut().enumerate().take(n_elements) {
                *slot = f(i);
            }
        }
        buf
    }

    #[test]
    fn install_fused_attn_qkv_splits_weight_into_three_slice_views() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Single block, hidden=4 → fused weight is 3*4*4 = 48 floats.
        // Q chunk = floats [0..16] @ byte_offset 0,
        // K = [16..32] @ byte_offset 64 (16 floats × 4 bytes),
        // V = [32..48] @ byte_offset 128 (32 floats × 4 bytes).
        //
        // We verify the slice views by inspecting (byte_offset, shape,
        // dtype, element_count). Direct CPU readback via `as_slice`
        // does NOT honor `byte_offset` — `MlxBuffer::contents_ptr()`
        // returns the start of the whole storage — so we read the
        // backing storage manually using `contents_ptr` + the recorded
        // `byte_offset` to verify the kernel-dispatch contract.
        // This pattern matches what the encoder does at
        // /opt/mlx-native/src/encoder.rs:218-220
        // (`set_buffer(index, metal_buffer(), buf.byte_offset())`).
        let device = MlxDevice::new().expect("device");
        let cfg = synth_qwen_vision_loader_cfg(4, 1);
        let mut tensors: HashMap<String, MlxBuffer> = HashMap::new();
        let fused_buf = alloc_f32_with(&device, 48, vec![12, 4], |i| i as f32);
        tensors.insert("v.blk.0.attn_qkv.weight".to_string(), fused_buf);
        LoadedMmprojWeights::install_fused_attn_qkv_slice_views(&mut tensors, &cfg)
            .expect("install on fused-only tensor map must succeed");
        let q = tensors.get("v.blk.0.attn_q.weight").expect("Q view");
        let k = tensors.get("v.blk.0.attn_k.weight").expect("K view");
        let v = tensors.get("v.blk.0.attn_v.weight").expect("V view");
        // byte_offset matches the per-chunk offset.
        assert_eq!(q.byte_offset(), 0);
        assert_eq!(k.byte_offset(), 64); // 16 floats × 4 bytes.
        assert_eq!(v.byte_offset(), 128); // 32 floats × 4 bytes.
                                          // shape was flattened to a 1-D view of n_elements = hidden*hidden.
        assert_eq!(q.element_count(), 16);
        assert_eq!(k.element_count(), 16);
        assert_eq!(v.element_count(), 16);
        // Verify the underlying bytes are the right region by reading
        // contents_ptr + byte_offset directly. This is what the encoder
        // does on dispatch.
        let read_slice = |buf: &MlxBuffer, n: usize| -> Vec<f32> {
            let ptr = buf.contents_ptr() as *const u8;
            let off = buf.byte_offset() as usize;
            // SAFETY: synthetic test buffer alloc'd above; we hold a
            // shared ref via `tensors.get` and no GPU work is in flight.
            unsafe { std::slice::from_raw_parts((ptr.add(off)) as *const f32, n).to_vec() }
        };
        let q_s = read_slice(q, 16);
        let k_s = read_slice(k, 16);
        let v_s = read_slice(v, 16);
        for i in 0..16 {
            assert_eq!(q_s[i], i as f32, "Q[{i}] must equal fused[{i}]");
            assert_eq!(
                k_s[i],
                (16 + i) as f32,
                "K[{i}] must equal fused[{}]",
                16 + i
            );
            assert_eq!(
                v_s[i],
                (32 + i) as f32,
                "V[{i}] must equal fused[{}]",
                32 + i
            );
        }
    }

    #[test]
    fn install_fused_attn_qkv_handles_optional_bias() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // hidden=8, single block. Fused bias is 3*8 = 24 f32 values.
        // Q bias = [0..8] @ off 0; K bias = [8..16] @ off 32;
        // V bias = [16..24] @ off 64.
        let device = MlxDevice::new().expect("device");
        let cfg = synth_qwen_vision_loader_cfg(8, 1);
        let mut tensors: HashMap<String, MlxBuffer> = HashMap::new();
        let weight_buf = alloc_f32_with(&device, 3 * 8 * 8, vec![24, 8], |i| i as f32);
        let bias_buf = alloc_f32_with(&device, 24, vec![24], |i| -(i as f32));
        tensors.insert("v.blk.0.attn_qkv.weight".to_string(), weight_buf);
        tensors.insert("v.blk.0.attn_qkv.bias".to_string(), bias_buf);
        LoadedMmprojWeights::install_fused_attn_qkv_slice_views(&mut tensors, &cfg)
            .expect("install with optional bias must succeed");
        let q_b = tensors.get("v.blk.0.attn_q.bias").expect("Q bias view");
        let k_b = tensors.get("v.blk.0.attn_k.bias").expect("K bias view");
        let v_b = tensors.get("v.blk.0.attn_v.bias").expect("V bias view");
        assert_eq!(q_b.byte_offset(), 0);
        assert_eq!(k_b.byte_offset(), 32); // 8 f32 = 32 bytes.
        assert_eq!(v_b.byte_offset(), 64);
        assert_eq!(q_b.element_count(), 8);
        assert_eq!(k_b.element_count(), 8);
        assert_eq!(v_b.element_count(), 8);
        // Read the underlying region via contents_ptr + byte_offset.
        let read_slice = |buf: &MlxBuffer, n: usize| -> Vec<f32> {
            let ptr = buf.contents_ptr() as *const u8;
            let off = buf.byte_offset() as usize;
            unsafe { std::slice::from_raw_parts(ptr.add(off) as *const f32, n).to_vec() }
        };
        for (label, view, n_off) in [("Q", q_b, 0usize), ("K", k_b, 8), ("V", v_b, 16)] {
            let s = read_slice(view, 8);
            for i in 0..8 {
                assert_eq!(s[i], -((n_off + i) as f32), "{label} bias [{i}] mismatch");
            }
        }
    }

    #[test]
    fn install_fused_attn_qkv_split_only_is_a_noop() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Pre-existing split tensors must pass through untouched —
        // the helper is a no-op on split-only inputs.
        let device = MlxDevice::new().expect("device");
        let cfg = synth_qwen_vision_loader_cfg(4, 1);
        let mut tensors: HashMap<String, MlxBuffer> = HashMap::new();
        for suffix in ["attn_q.weight", "attn_k.weight", "attn_v.weight"] {
            let key = format!("v.blk.0.{suffix}");
            let buf = alloc_f32_with(&device, 16, vec![4, 4], |i| i as f32);
            tensors.insert(key, buf);
        }
        let n_before = tensors.len();
        LoadedMmprojWeights::install_fused_attn_qkv_slice_views(&mut tensors, &cfg)
            .expect("split-only must be a no-op");
        assert_eq!(
            tensors.len(),
            n_before,
            "split-only tensor map must be unchanged"
        );
    }

    #[test]
    fn install_fused_attn_qkv_rejects_mixed_block() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // A block with BOTH fused and split is a producer bug — the
        // validator catches it normally; this test guards the
        // defense-in-depth check inside the loader.
        let device = MlxDevice::new().expect("device");
        let cfg = synth_qwen_vision_loader_cfg(4, 1);
        let mut tensors: HashMap<String, MlxBuffer> = HashMap::new();
        tensors.insert(
            "v.blk.0.attn_qkv.weight".to_string(),
            alloc_f32_with(&device, 48, vec![12, 4], |i| i as f32),
        );
        // Add a stray split tensor as well — this should trigger the
        // mixed-state error path.
        tensors.insert(
            "v.blk.0.attn_q.weight".to_string(),
            alloc_f32_with(&device, 16, vec![4, 4], |_| 0.0),
        );
        let err = LoadedMmprojWeights::install_fused_attn_qkv_slice_views(&mut tensors, &cfg)
            .expect_err("mixed fused+split per block must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("BOTH fused") && msg.contains("attn_qkv"),
            "loader error must call out the mixed-state case; got: {msg}"
        );
    }

    #[test]
    fn install_fused_attn_qkv_rejects_mixed_or_orphan_bias_state() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new().expect("device");
        let cfg = synth_qwen_vision_loader_cfg(4, 1);

        let mut mixed = HashMap::new();
        mixed.insert(
            "v.blk.0.attn_qkv.weight".to_string(),
            alloc_f32_with(&device, 48, vec![12, 4], |_| 0.0),
        );
        mixed.insert(
            "v.blk.0.attn_qkv.bias".to_string(),
            alloc_f32_with(&device, 12, vec![12], |_| 0.0),
        );
        mixed.insert(
            "v.blk.0.attn_q.bias".to_string(),
            alloc_f32_with(&device, 4, vec![4], |_| 0.0),
        );
        let error = LoadedMmprojWeights::install_fused_attn_qkv_slice_views(&mut mixed, &cfg)
            .expect_err("mixed fused/split bias must fail closed");
        assert!(format!("{error}").contains("BOTH fused and split"));

        let mut orphan = HashMap::from([(
            "v.blk.0.attn_qkv.bias".to_string(),
            alloc_f32_with(&device, 12, vec![12], |_| 0.0),
        )]);
        let error = LoadedMmprojWeights::install_fused_attn_qkv_slice_views(&mut orphan, &cfg)
            .expect_err("fused bias without fused weight must fail closed");
        assert!(format!("{error}").contains("without fused weight"));
    }

    #[test]
    fn install_fused_attn_qkv_rejects_undersized_fused_weight() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Fused weight that's smaller than 3*hidden*hidden floats is a
        // converter bug — slicing would silently return wrong data.
        // Reject loud.
        let device = MlxDevice::new().expect("device");
        let cfg = synth_qwen_vision_loader_cfg(4, 1);
        let mut tensors: HashMap<String, MlxBuffer> = HashMap::new();
        // Allocate only 2*4*4 = 32 floats instead of 48.
        let undersized = alloc_f32_with(&device, 32, vec![8, 4], |i| i as f32);
        tensors.insert("v.blk.0.attn_qkv.weight".to_string(), undersized);
        let err = LoadedMmprojWeights::install_fused_attn_qkv_slice_views(&mut tensors, &cfg)
            .expect_err("undersized fused weight must fail loud");
        let msg = format!("{err}");
        assert!(
            msg.contains("fused") && msg.contains("expected"),
            "loader error must name the fused tensor + expected shape; got: {msg}"
        );
    }

    #[test]
    fn install_fused_attn_qkv_multi_block_batches_correctly() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        // Multi-block: every block independently slices its fused
        // tensor. Block N's Q view points at block N's storage at
        // byte_offset 0 — distinct backing storage from block M (M≠N).
        let device = MlxDevice::new().expect("device");
        let cfg = synth_qwen_vision_loader_cfg(4, 3);
        let mut tensors: HashMap<String, MlxBuffer> = HashMap::new();
        for layer_idx in 0..3 {
            let fused = alloc_f32_with(&device, 48, vec![12, 4], |i| (layer_idx * 100 + i) as f32);
            tensors.insert(format!("v.blk.{layer_idx}.attn_qkv.weight"), fused);
        }
        LoadedMmprojWeights::install_fused_attn_qkv_slice_views(&mut tensors, &cfg)
            .expect("multi-block install must succeed");
        let read_slice = |buf: &MlxBuffer, n: usize| -> Vec<f32> {
            let ptr = buf.contents_ptr() as *const u8;
            let off = buf.byte_offset() as usize;
            unsafe { std::slice::from_raw_parts(ptr.add(off) as *const f32, n).to_vec() }
        };
        for layer_idx in 0..3 {
            let q = tensors
                .get(&format!("v.blk.{layer_idx}.attn_q.weight"))
                .unwrap_or_else(|| panic!("Q for block {layer_idx}"));
            assert_eq!(q.byte_offset(), 0, "Q view always at fused-tensor start");
            let q_s = read_slice(q, 16);
            for i in 0..16 {
                assert_eq!(
                    q_s[i],
                    (layer_idx * 100 + i) as f32,
                    "block {layer_idx} Q[{i}] mismatch"
                );
            }
        }
    }
}
