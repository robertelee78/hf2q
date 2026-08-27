//! Role-aware DeepSeek-V4 weight residency.
//!
//! Matmul weights stay in their GGUF representation. Only tensors consumed
//! by elementwise kernels are expanded to F32; hash routing remains I32.

use std::collections::HashMap;

use anyhow::Context;
use mlx_native::gguf::GgufFile;
use mlx_native::ops::quantized_matmul_ggml::{GgmlType, MM_ROUTING_THRESHOLD};
use mlx_native::{
    ggml_capability, ggml_matrix_bytes, DType, DenseMatmulIdInputLayout, DenseMatmulIdMultiplicity,
    DenseMatmulIdParams, DenseMatmulIdRoute, GgmlCapabilityRequest, GgmlExpertInputLayout,
    GgmlExpertShape, GgmlInvocation, GgmlRoutingPolicy, GgmlWorkloadClass, MlxBuffer, MlxDevice,
    MlxError, GGML_CAPABILITY_SCHEMA_VERSION, MM_ID_ROUTING_THRESHOLD,
};
use thiserror::Error;

use crate::inference::dense_bf16_activation::NativeBf16Matrix;
use crate::inference::dense_expert_activation::NativeScalarExpertMatrix;

use super::weights::{validate_tensor_catalog, TensorRole, WeightCatalogError};
use super::Deepseek4Config;

#[derive(Debug, Error)]
pub enum WeightResidencyError {
    #[error(transparent)]
    Catalog(#[from] WeightCatalogError),
    #[error("tensor '{name}' with role {role:?} cannot use GGML storage {actual:?}")]
    StorageType {
        name: String,
        role: TensorRole,
        actual: GgmlType,
    },
    #[error(
        "tensor '{name}' with role {role:?} and GGML storage {actual:?} has no executable native route: {diagnostic}"
    )]
    ExecutionCapability {
        name: String,
        role: TensorRole,
        actual: GgmlType,
        diagnostic: String,
    },
    #[error("failed to make DeepSeek-V4 tensor '{name}' resident: {source}")]
    Load {
        name: String,
        #[source]
        source: MlxError,
    },
    #[error("resident tensor '{name}' has dtype {actual}, expected {expected}")]
    LoadedDtype {
        name: String,
        expected: DType,
        actual: DType,
    },
    #[error("resident tensor '{name}' uses {actual} bytes, expected {expected}")]
    LoadedBytes {
        name: String,
        expected: u64,
        actual: u64,
    },
    #[error("resident-byte accounting overflowed while adding tensor '{name}'")]
    ByteOverflow { name: String },
    #[error("tensor '{name}' disappeared after exact catalog validation")]
    MissingAfterCatalog { name: String },
    #[error(
        "hash-routing tensor '{name}' row {row} slot {slot} selects expert {expert}, outside 0..{expert_count}"
    )]
    HashExpertOutOfRange {
        name: String,
        row: usize,
        slot: usize,
        expert: i32,
        expert_count: u32,
    },
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum WeightLookupError {
    #[error("DeepSeek-V4 tensor '{name}' is not resident")]
    Missing { name: String },
    #[error("DeepSeek-V4 tensor '{name}' has role {actual:?}, requested {expected:?}")]
    RoleMismatch {
        name: String,
        expected: TensorRole,
        actual: TensorRole,
    },
}

struct ResidentTensor {
    role: TensorRole,
    ggml_type: GgmlType,
    shape: Vec<usize>,
    buffer: MlxBuffer,
}

pub struct RawMatrixRef<'a> {
    pub buffer: &'a MlxBuffer,
    pub ggml_type: GgmlType,
    pub shape: &'a [usize],
}

pub struct Deepseek4Weights {
    tensors: HashMap<String, ResidentTensor>,
    hash_route_ids_distinct: Vec<bool>,
    resident_bytes: u64,
    file_backed_bytes: u64,
    anonymous_bytes: u64,
    mapped_segment_count: usize,
    _device: MlxDevice,
}

impl Deepseek4Weights {
    fn preflight_specs(
        gguf: &GgufFile,
        cfg: &Deepseek4Config,
    ) -> Result<Vec<super::weights::TensorSpec>, WeightResidencyError> {
        let specs = validate_tensor_catalog(gguf, cfg)?;
        for spec in &specs {
            let info = gguf.tensor_info(&spec.name).ok_or_else(|| {
                WeightResidencyError::MissingAfterCatalog {
                    name: spec.name.clone(),
                }
            })?;
            validate_storage_type(&spec.name, spec.role, info.ggml_type, &info.shape, cfg)?;
        }
        Ok(specs)
    }

    /// Validate the exact catalog, shapes, storage roles, and executable
    /// native routes without mapping tensor payloads or allocating Metal
    /// buffers.
    pub fn preflight_gguf(
        gguf: &GgufFile,
        cfg: &Deepseek4Config,
    ) -> Result<(), WeightResidencyError> {
        Self::preflight_specs(gguf, cfg).map(drop)
    }

    /// Validate the complete verifier catalog, then load each tensor according
    /// to its runtime role. No allocation occurs before catalog and storage
    /// types have both passed validation.
    pub fn load_from_gguf(
        gguf: &GgufFile,
        cfg: &Deepseek4Config,
        device: MlxDevice,
    ) -> Result<Self, WeightResidencyError> {
        let specs = Self::preflight_specs(gguf, cfg)?;

        let mut tensors = HashMap::with_capacity(specs.len());
        let mut resident_bytes = 0_u64;
        let mut file_backed_bytes = 0_u64;
        let mut anonymous_bytes = 0_u64;
        let map_raw_weights = std::env::var("HF2Q_DEEPSEEK_MMAP_WEIGHTS").as_deref() != Ok("0");
        let mapped_tensor_data = map_raw_weights
            .then(|| {
                gguf.map_tensor_data(&device)
                    .map_err(|source| WeightResidencyError::Load {
                        name: "GGUF tensor data".to_string(),
                        source,
                    })
            })
            .transpose()?;
        let mapped_segment_count = mapped_tensor_data
            .as_ref()
            .map_or(0, |mapping| mapping.segment_count());
        for spec in specs {
            let info = gguf.tensor_info(&spec.name).ok_or_else(|| {
                WeightResidencyError::MissingAfterCatalog {
                    name: spec.name.clone(),
                }
            })?;
            let buffer =
                match spec.role {
                    TensorRole::EmbeddingMatrix
                    | TensorRole::DenseMatrix
                    | TensorRole::GroupedMatrix
                    | TensorRole::ExpertStack
                    | TensorRole::IntegerLookupI32
                        if map_raw_weights =>
                    {
                        mapped_tensor_data
                            .as_ref()
                            .expect("mapped tensor data exists when mmap weights are enabled")
                            .load_tensor(&spec.name)
                            .map_err(|source| WeightResidencyError::Load {
                                name: spec.name.clone(),
                                source,
                            })?
                    }
                    TensorRole::EmbeddingMatrix
                    | TensorRole::DenseMatrix
                    | TensorRole::GroupedMatrix
                    | TensorRole::ExpertStack
                    | TensorRole::IntegerLookupI32 => gguf
                        .load_tensor(&spec.name, &device)
                        .map_err(|source| WeightResidencyError::Load {
                            name: spec.name.clone(),
                            source,
                        })?,
                    TensorRole::ElementwiseF32
                        if info.ggml_type == GgmlType::F32 && map_raw_weights =>
                    {
                        mapped_tensor_data
                            .as_ref()
                            .expect("mapped tensor data exists when mmap weights are enabled")
                            .load_tensor(&spec.name)
                            .map_err(|source| WeightResidencyError::Load {
                                name: spec.name.clone(),
                                source,
                            })?
                    }
                    TensorRole::ElementwiseF32 if info.ggml_type == GgmlType::F32 => gguf
                        .load_tensor(&spec.name, &device)
                        .map_err(|source| WeightResidencyError::Load {
                            name: spec.name.clone(),
                            source,
                        })?,
                    TensorRole::ElementwiseF32 => gguf
                        .load_tensor_f32(&spec.name, &device)
                        .map_err(|source| WeightResidencyError::Load {
                            name: spec.name.clone(),
                            source,
                        })?,
                };
            validate_loaded_buffer(
                &spec.name,
                spec.role,
                info.ggml_type,
                info.byte_len,
                &info.shape,
                &buffer,
            )?;
            let bytes = u64::try_from(buffer.data_byte_len()).map_err(|_| {
                WeightResidencyError::ByteOverflow {
                    name: spec.name.clone(),
                }
            })?;
            resident_bytes = resident_bytes.checked_add(bytes).ok_or_else(|| {
                WeightResidencyError::ByteOverflow {
                    name: spec.name.clone(),
                }
            })?;
            if buffer.is_file_backed() {
                file_backed_bytes = file_backed_bytes.checked_add(bytes).ok_or_else(|| {
                    WeightResidencyError::ByteOverflow {
                        name: spec.name.clone(),
                    }
                })?;
            } else {
                anonymous_bytes = anonymous_bytes.checked_add(bytes).ok_or_else(|| {
                    WeightResidencyError::ByteOverflow {
                        name: spec.name.clone(),
                    }
                })?;
            }
            tensors.insert(
                spec.name,
                ResidentTensor {
                    role: spec.role,
                    ggml_type: info.ggml_type,
                    shape: info.shape.clone(),
                    buffer,
                },
            );
        }
        let hash_route_ids_distinct = validate_hash_route_tables(&tensors, cfg)?;
        Ok(Self {
            tensors,
            hash_route_ids_distinct,
            resident_bytes,
            file_backed_bytes,
            anonymous_bytes,
            mapped_segment_count,
            _device: device,
        })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Sum of logical Metal tensor bytes, including F32 expansion only for
    /// elementwise state and excluding page-alignment padding.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Immutable raw weight bytes served directly from the GGUF mapping.
    pub fn file_backed_bytes(&self) -> u64 {
        self.file_backed_bytes
    }

    /// Weight bytes copied or expanded into anonymous Metal allocations.
    pub fn anonymous_bytes(&self) -> u64 {
        self.anonymous_bytes
    }

    /// Number of shared Metal resources spanning the file-backed tensors.
    pub fn mapped_segment_count(&self) -> usize {
        self.mapped_segment_count
    }

    pub fn tensor(
        &self,
        name: &str,
        expected: TensorRole,
    ) -> Result<&MlxBuffer, WeightLookupError> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| WeightLookupError::Missing {
                name: name.to_string(),
            })?;
        if entry.role != expected {
            return Err(WeightLookupError::RoleMismatch {
                name: name.to_string(),
                expected,
                actual: entry.role,
            });
        }
        Ok(&entry.buffer)
    }

    pub fn raw_matrix(&self, name: &str) -> Result<&MlxBuffer, WeightLookupError> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| WeightLookupError::Missing {
                name: name.to_string(),
            })?;
        if !entry.role.is_native_matrix() {
            return Err(WeightLookupError::RoleMismatch {
                name: name.to_string(),
                expected: TensorRole::DenseMatrix,
                actual: entry.role,
            });
        }
        Ok(&entry.buffer)
    }

    pub fn raw_matrix_ref(&self, name: &str) -> Result<RawMatrixRef<'_>, WeightLookupError> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| WeightLookupError::Missing {
                name: name.to_string(),
            })?;
        if !entry.role.is_native_matrix() {
            return Err(WeightLookupError::RoleMismatch {
                name: name.to_string(),
                expected: TensorRole::DenseMatrix,
                actual: entry.role,
            });
        }
        Ok(RawMatrixRef {
            buffer: &entry.buffer,
            ggml_type: entry.ggml_type,
            shape: &entry.shape,
        })
    }

    pub fn f32_state(&self, name: &str) -> Result<&MlxBuffer, WeightLookupError> {
        self.tensor(name, TensorRole::ElementwiseF32)
    }

    pub fn i32_lookup(&self, name: &str) -> Result<&MlxBuffer, WeightLookupError> {
        self.tensor(name, TensorRole::IntegerLookupI32)
    }

    /// Whether one token's checkpoint-selected experts are unique in this
    /// artifact. Score-routed layers are unique by construction. Hash-routed
    /// layers are classified from their loaded I32 `tid2eid` payload so the
    /// fast grouped route is never enabled from metadata or architecture name
    /// alone.
    pub(crate) fn expert_ids_are_distinct_per_token(&self, layer: usize) -> bool {
        self.hash_route_ids_distinct
            .get(layer)
            .copied()
            .unwrap_or(true)
    }

    pub(crate) fn native_bf16_dense_matrices(
        &self,
        output_groups: usize,
    ) -> anyhow::Result<Vec<NativeBf16Matrix<'_>>> {
        let mut matrices = Vec::new();
        for (name, entry) in &self.tensors {
            if !matches!(
                entry.role,
                TensorRole::DenseMatrix | TensorRole::GroupedMatrix
            ) {
                continue;
            }
            anyhow::ensure!(
                (entry.buffer.dtype() == DType::BF16) == (entry.ggml_type == GgmlType::BF16),
                "DeepSeek-V4 {name} storage {} disagrees with declared type {:?}",
                entry.buffer.dtype(),
                entry.ggml_type
            );
            if entry.ggml_type != GgmlType::BF16 {
                continue;
            }
            let [rows, columns] = entry.shape.as_slice() else {
                anyhow::bail!(
                    "DeepSeek-V4 native BF16 dense matrix {name} must be rank two, got {:?}",
                    entry.shape
                );
            };
            if entry.role == TensorRole::DenseMatrix {
                matrices.push(NativeBf16Matrix::unbatched(
                    name,
                    &entry.buffer,
                    u32::try_from(*rows)?,
                    u32::try_from(*columns)?,
                ));
                continue;
            }
            anyhow::ensure!(
                output_groups > 0 && rows % output_groups == 0,
                "DeepSeek-V4 grouped BF16 matrix {name} rows {rows} are not divisible by {output_groups} output groups"
            );
            let rank = rows / output_groups;
            matrices.push(NativeBf16Matrix {
                label: name,
                weight: &entry.buffer,
                n: u32::try_from(rank)?,
                k: u32::try_from(*columns)?,
                src0_batch: u32::try_from(output_groups)?,
                src1_batch: u32::try_from(output_groups)?,
                reachable_row_mask: u16::MAX,
            });
        }
        Ok(matrices)
    }

    pub(crate) fn native_scalar_expert_matrices(
        &self,
        cfg: &Deepseek4Config,
    ) -> anyhow::Result<Vec<NativeScalarExpertMatrix<'_>>> {
        use mlx_native::{DenseMatmulIdInputLayout, DenseMatmulIdMultiplicity};

        let source_widths = super::native_expert_activation_widths();
        let mut matrices = Vec::new();
        for (name, entry) in &self.tensors {
            if entry.role != TensorRole::ExpertStack
                || !matches!(
                    entry.ggml_type,
                    GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
                )
            {
                continue;
            }
            let [experts, n, k] = entry.shape.as_slice() else {
                anyhow::bail!(
                    "DeepSeek-V4 scalar expert matrix {name} must be rank three, got {:?}",
                    entry.shape
                );
            };
            anyhow::ensure!(
                *experts == cfg.num_experts as usize,
                "DeepSeek-V4 scalar expert count drift for {name}: {experts} != {}",
                cfg.num_experts
            );
            let expected_dtype = match entry.ggml_type {
                GgmlType::F32 => DType::F32,
                GgmlType::F16 => DType::F16,
                GgmlType::BF16 => DType::BF16,
                _ => unreachable!(),
            };
            anyhow::ensure!(
                entry.buffer.dtype() == expected_dtype,
                "DeepSeek-V4 {name} declared {:?} but maps as {}",
                entry.ggml_type,
                entry.buffer.dtype()
            );
            let expert_stride_bytes = u64::try_from(*n)?
                .checked_mul(u64::try_from(*k)?)
                .and_then(|elements| elements.checked_mul(expected_dtype.size_of() as u64))
                .context("DeepSeek-V4 scalar expert stride overflow")?;
            let layer = name
                .strip_prefix("blk.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|layer| layer.parse::<u32>().ok())
                .with_context(|| format!("parse DeepSeek-V4 expert layer from {name}"))?;
            let id_multiplicity = if self.expert_ids_are_distinct_per_token(layer as usize) {
                DenseMatmulIdMultiplicity::DistinctPerToken
            } else {
                DenseMatmulIdMultiplicity::MayRepeat
            };
            let is_down = name.contains("ffn_down_exps.weight");
            let base = NativeScalarExpertMatrix {
                label: name,
                weight: &entry.buffer,
                n: u32::try_from(*n)?,
                k: u32::try_from(*k)?,
                top_k: cfg.num_experts_per_tok,
                n_experts: cfg.num_experts,
                expert_stride_bytes,
                input_layout: if is_down {
                    DenseMatmulIdInputLayout::Slotted
                } else {
                    DenseMatmulIdInputLayout::SharedPerToken
                },
                id_multiplicity,
                calibrated_m: source_widths.clone(),
            };
            matrices.push(base);
        }
        Ok(matrices)
    }
}

pub(super) fn classify_hash_route_table(
    name: &str,
    values: &[i32],
    top_k: usize,
    expert_count: u32,
) -> Result<bool, WeightResidencyError> {
    let mut all_distinct = true;
    for (row, ids) in values.chunks_exact(top_k).enumerate() {
        for (slot, &expert) in ids.iter().enumerate() {
            if expert < 0 || expert >= expert_count as i32 {
                return Err(WeightResidencyError::HashExpertOutOfRange {
                    name: name.to_string(),
                    row,
                    slot,
                    expert,
                    expert_count,
                });
            }
            if ids[..slot].contains(&expert) {
                all_distinct = false;
            }
        }
    }
    Ok(all_distinct)
}

fn validate_hash_route_tables(
    tensors: &HashMap<String, ResidentTensor>,
    cfg: &Deepseek4Config,
) -> Result<Vec<bool>, WeightResidencyError> {
    let top_k = cfg.num_experts_per_tok as usize;
    let mut distinct = Vec::with_capacity(cfg.hash_layer_count as usize);
    for layer in 0..cfg.hash_layer_count as usize {
        let name = format!("blk.{layer}.ffn_gate_tid2eid.weight");
        let entry = tensors
            .get(&name)
            .ok_or_else(|| WeightResidencyError::MissingAfterCatalog { name: name.clone() })?;
        let values = entry.buffer.as_logical_slice::<i32>().map_err(|source| {
            WeightResidencyError::Load {
                name: name.clone(),
                source,
            }
        })?;
        distinct.push(classify_hash_route_table(
            &name,
            values,
            top_k,
            cfg.num_experts,
        )?);
    }
    Ok(distinct)
}

fn validate_storage_type(
    name: &str,
    role: TensorRole,
    actual: GgmlType,
    shape: &[usize],
    cfg: &Deepseek4Config,
) -> Result<(), WeightResidencyError> {
    let valid = match role {
        TensorRole::EmbeddingMatrix
        | TensorRole::DenseMatrix
        | TensorRole::GroupedMatrix
        | TensorRole::ExpertStack => !matches!(actual, GgmlType::I16 | GgmlType::I32),
        TensorRole::ElementwiseF32 => !matches!(actual, GgmlType::I16 | GgmlType::I32),
        TensorRole::IntegerLookupI32 => actual == GgmlType::I32,
    };
    if !valid {
        return Err(WeightResidencyError::StorageType {
            name: name.to_string(),
            role,
            actual,
        });
    }
    if !role.is_native_matrix() {
        return Ok(());
    }

    let dim = |index: usize, label: &str| -> Result<u32, WeightResidencyError> {
        shape
            .get(index)
            .copied()
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| WeightResidencyError::ExecutionCapability {
                name: name.to_string(),
                role,
                actual,
                diagnostic: format!("invalid or missing {label} dimension in {shape:?}"),
            })
    };
    let mut requests = Vec::new();
    match role {
        TensorRole::EmbeddingMatrix => {
            requests.push(GgmlCapabilityRequest {
                schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
                invocation: GgmlInvocation::EmbeddingGather {
                    n_tokens: 1,
                    vocab_size: dim(0, "vocabulary")?,
                    embed_dim: dim(1, "embedding")?,
                },
                ggml_type: actual,
                workload: GgmlWorkloadClass::Embedding,
                routing: GgmlRoutingPolicy::default(),
            });
        }
        TensorRole::DenseMatrix | TensorRole::GroupedMatrix => {
            let (n, k) = if role == TensorRole::DenseMatrix {
                (dim(0, "output")?, dim(1, "input")?)
            } else {
                let groups = cfg.output_groups;
                let rows = dim(0, "grouped output")?;
                if groups == 0 || rows % groups != 0 {
                    return Err(WeightResidencyError::ExecutionCapability {
                        name: name.to_string(),
                        role,
                        actual,
                        diagnostic: format!(
                            "grouped rows {rows} are not divisible by output_groups {groups}"
                        ),
                    });
                }
                (rows / groups, dim(1, "group input")?)
            };
            for (m, workload) in [
                (1, GgmlWorkloadClass::DecodeSingle),
                (2, GgmlWorkloadClass::ContinuousWidth),
                (33, GgmlWorkloadClass::Prompt),
            ] {
                requests.push(GgmlCapabilityRequest {
                    schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
                    invocation: GgmlInvocation::DenseAuto { m, n, k },
                    ggml_type: actual,
                    workload,
                    routing: GgmlRoutingPolicy::default(),
                });
            }
        }
        TensorRole::ExpertStack => {
            let layer = name
                .strip_prefix("blk.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|layer| layer.parse::<u32>().ok())
                .unwrap_or(cfg.num_hidden_layers);
            let hash_layer = layer < cfg.hash_layer_count;
            // Score routing selects unique indices by construction. Hash
            // routing may use the same fast route only after the loaded
            // `tid2eid` payload is classified by `validate_hash_route_tables`.
            // Capability preflight proves both routes so a repeat-bearing
            // custom artifact remains executable through the exact fallback.
            let ids_are_distinct_per_token = true;
            let n_experts = dim(0, "expert count")?;
            let n = dim(1, "expert output")?;
            let k = dim(2, "expert input")?;
            let expert_stride_bytes = ggml_matrix_bytes(actual, n, k).map_err(|error| {
                WeightResidencyError::ExecutionCapability {
                    name: name.to_string(),
                    role,
                    actual,
                    diagnostic: error.to_string(),
                }
            })?;
            let down_projection = name.contains("ffn_down_exps");
            if matches!(actual, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
                let dtype = match actual {
                    GgmlType::F32 => DType::F32,
                    GgmlType::F16 => DType::F16,
                    GgmlType::BF16 => DType::BF16,
                    _ => unreachable!(),
                };
                for n_tokens in super::native_expert_activation_widths() {
                    let multiplicities = if hash_layer {
                        [
                            Some(DenseMatmulIdMultiplicity::DistinctPerToken),
                            Some(DenseMatmulIdMultiplicity::MayRepeat),
                        ]
                    } else {
                        [Some(DenseMatmulIdMultiplicity::DistinctPerToken), None]
                    };
                    for id_multiplicity in multiplicities.into_iter().flatten() {
                        let params = DenseMatmulIdParams {
                            m: n_tokens,
                            n,
                            k,
                            top_k: cfg.num_experts_per_tok,
                            n_experts,
                            expert_stride_bytes,
                            input_layout: if down_projection {
                                DenseMatmulIdInputLayout::Slotted
                            } else {
                                DenseMatmulIdInputLayout::SharedPerToken
                            },
                            id_multiplicity,
                            route: DenseMatmulIdRoute::Direct,
                        };
                        mlx_native::dense_matmul_id_capability(dtype, &params).map_err(
                            |error| WeightResidencyError::ExecutionCapability {
                                name: name.to_string(),
                                role,
                                actual,
                                diagnostic: format!(
                                    "native scalar expert source M={n_tokens} multiplicity={id_multiplicity:?}: {error}"
                                ),
                            },
                        )?;
                    }
                }
                return Ok(());
            }
            for n_tokens in [1, 8, 9, 33] {
                let large_slotted = down_projection
                    && matches!(cfg.num_experts_per_tok, 1 | 6 | 8)
                    && n_tokens > MM_ID_ROUTING_THRESHOLD;
                let (shape_tokens, shape_top_k, input_layout) = if down_projection && !large_slotted
                {
                    (
                        n_tokens
                            .checked_mul(cfg.num_experts_per_tok)
                            .ok_or_else(|| WeightResidencyError::ExecutionCapability {
                                name: name.to_string(),
                                role,
                                actual,
                                diagnostic: "flattened expert-row count overflow".to_string(),
                            })?,
                        1,
                        GgmlExpertInputLayout::SharedPerToken,
                    )
                } else {
                    (
                        n_tokens,
                        cfg.num_experts_per_tok,
                        if large_slotted {
                            GgmlExpertInputLayout::Slotted
                        } else {
                            GgmlExpertInputLayout::SharedPerToken
                        },
                    )
                };
                let shape = GgmlExpertShape {
                    n_tokens: shape_tokens,
                    n,
                    k,
                    top_k: shape_top_k,
                    n_experts,
                    expert_stride_bytes,
                    // A flattened down-projection exposes one selected expert
                    // ID per physical row. Distinctness is therefore
                    // vacuously true for the capability request even when the
                    // original hash-router top-k may repeat IDs.
                    ids_are_distinct_per_token: shape_top_k == 1 || ids_are_distinct_per_token,
                    ids_within_expert_range: true,
                };
                let paired_gate_up = !down_projection
                    && matches!(cfg.num_experts_per_tok, 1 | 6 | 8)
                    && n_tokens > MM_ID_ROUTING_THRESHOLD;
                let workload = match shape_tokens {
                    1 => GgmlWorkloadClass::DecodeSingle,
                    2..=MM_ROUTING_THRESHOLD => GgmlWorkloadClass::ContinuousWidth,
                    _ => GgmlWorkloadClass::Prompt,
                };
                requests.push(GgmlCapabilityRequest {
                    schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
                    invocation: if paired_gate_up {
                        GgmlInvocation::ExpertPooledPair { shape }
                    } else {
                        GgmlInvocation::ExpertPooled {
                            shape,
                            input_layout,
                        }
                    },
                    ggml_type: actual,
                    workload,
                    routing: GgmlRoutingPolicy::default(),
                });
                if hash_layer && !down_projection {
                    requests.push(GgmlCapabilityRequest {
                        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
                        invocation: GgmlInvocation::ExpertForceMv {
                            shape: GgmlExpertShape {
                                n_tokens,
                                n,
                                k,
                                top_k: cfg.num_experts_per_tok,
                                n_experts,
                                expert_stride_bytes,
                                ids_are_distinct_per_token: false,
                                ids_within_expert_range: true,
                            },
                        },
                        ggml_type: actual,
                        workload,
                        routing: GgmlRoutingPolicy::default(),
                    });
                } else if hash_layer && down_projection && large_slotted {
                    requests.push(GgmlCapabilityRequest {
                        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
                        invocation: GgmlInvocation::ExpertPooled {
                            shape: GgmlExpertShape {
                                n_tokens: n_tokens
                                    .checked_mul(cfg.num_experts_per_tok)
                                    .ok_or_else(|| WeightResidencyError::ExecutionCapability {
                                        name: name.to_string(),
                                        role,
                                        actual,
                                        diagnostic: "flattened fallback expert-row count overflow"
                                            .to_string(),
                                    })?,
                                n,
                                k,
                                top_k: 1,
                                n_experts,
                                expert_stride_bytes,
                                ids_are_distinct_per_token: true,
                                ids_within_expert_range: true,
                            },
                            input_layout: GgmlExpertInputLayout::SharedPerToken,
                        },
                        ggml_type: actual,
                        workload,
                        routing: GgmlRoutingPolicy::default(),
                    });
                }
            }
        }
        TensorRole::ElementwiseF32 | TensorRole::IntegerLookupI32 => unreachable!(),
    }
    for request in requests {
        let capability = ggml_capability(request);
        if !capability.executable {
            return Err(WeightResidencyError::ExecutionCapability {
                name: name.to_string(),
                role,
                actual,
                diagnostic: format!("{}; request={request:?}", capability.diagnostic),
            });
        }
    }
    Ok(())
}

fn validate_loaded_buffer(
    name: &str,
    role: TensorRole,
    storage: GgmlType,
    disk_bytes: usize,
    shape: &[usize],
    buffer: &MlxBuffer,
) -> Result<(), WeightResidencyError> {
    let expected_dtype = match role {
        TensorRole::ElementwiseF32 => Some(DType::F32),
        TensorRole::IntegerLookupI32 => Some(DType::I32),
        role if role.is_native_matrix() => None,
        _ => unreachable!("every DeepSeek tensor role is classified above"),
    };
    if let Some(expected) = expected_dtype {
        if buffer.dtype() != expected {
            return Err(WeightResidencyError::LoadedDtype {
                name: name.to_string(),
                expected,
                actual: buffer.dtype(),
            });
        }
    } else if !matches!(
        buffer.dtype(),
        DType::F32 | DType::F16 | DType::BF16 | DType::U8
    ) {
        return Err(WeightResidencyError::LoadedDtype {
            name: name.to_string(),
            expected: DType::U8,
            actual: buffer.dtype(),
        });
    }

    let expected_bytes = if role == TensorRole::ElementwiseF32 && storage != GgmlType::F32 {
        shape
            .iter()
            .try_fold(4usize, |bytes, dim| bytes.checked_mul(*dim))
            .ok_or_else(|| WeightResidencyError::ByteOverflow {
                name: name.to_string(),
            })?
    } else {
        disk_bytes
    };
    let expected =
        u64::try_from(expected_bytes).map_err(|_| WeightResidencyError::ByteOverflow {
            name: name.to_string(),
        })?;
    let actual =
        u64::try_from(buffer.data_byte_len()).map_err(|_| WeightResidencyError::ByteOverflow {
            name: name.to_string(),
        })?;
    if actual != expected {
        return Err(WeightResidencyError::LoadedBytes {
            name: name.to_string(),
            expected,
            actual,
        });
    }
    Ok(())
}
