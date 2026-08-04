//! Role-aware DeepSeek-V4 weight residency.
//!
//! Matmul weights stay in their GGUF representation. Only tensors consumed
//! by elementwise kernels are expanded to F32; hash routing remains I32.

use std::collections::HashMap;

use mlx_native::gguf::GgufFile;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, MlxBuffer, MlxDevice, MlxError};
use thiserror::Error;

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
    buffer: MlxBuffer,
}

pub struct Deepseek4Weights {
    tensors: HashMap<String, ResidentTensor>,
    resident_bytes: u64,
    _device: MlxDevice,
}

impl Deepseek4Weights {
    /// Validate the complete verifier catalog, then load each tensor according
    /// to its runtime role. No allocation occurs before catalog and storage
    /// types have both passed validation.
    pub fn load_from_gguf(
        gguf: &GgufFile,
        cfg: &Deepseek4Config,
        device: MlxDevice,
    ) -> Result<Self, WeightResidencyError> {
        let specs = validate_tensor_catalog(gguf, cfg)?;
        for spec in &specs {
            let info = gguf.tensor_info(&spec.name).ok_or_else(|| {
                WeightResidencyError::MissingAfterCatalog {
                    name: spec.name.clone(),
                }
            })?;
            validate_storage_type(&spec.name, spec.role, info.ggml_type)?;
        }

        let mut tensors = HashMap::with_capacity(specs.len());
        let mut resident_bytes = 0_u64;
        for spec in specs {
            let info = gguf.tensor_info(&spec.name).ok_or_else(|| {
                WeightResidencyError::MissingAfterCatalog {
                    name: spec.name.clone(),
                }
            })?;
            let buffer = match spec.role {
                TensorRole::RawMatrix | TensorRole::IntegerLookupI32 => gguf
                    .load_tensor(&spec.name, &device)
                    .map_err(|source| WeightResidencyError::Load {
                        name: spec.name.clone(),
                        source,
                    })?,
                TensorRole::ElementwiseF32 => {
                    gguf.load_tensor_f32(&spec.name, &device)
                        .map_err(|source| WeightResidencyError::Load {
                            name: spec.name.clone(),
                            source,
                        })?
                }
            };
            validate_loaded_buffer(&spec.name, spec.role, info.byte_len, &buffer)?;
            let bytes = u64::try_from(buffer.byte_len()).map_err(|_| {
                WeightResidencyError::ByteOverflow {
                    name: spec.name.clone(),
                }
            })?;
            resident_bytes = resident_bytes.checked_add(bytes).ok_or_else(|| {
                WeightResidencyError::ByteOverflow {
                    name: spec.name.clone(),
                }
            })?;
            tensors.insert(
                spec.name,
                ResidentTensor {
                    role: spec.role,
                    buffer,
                },
            );
        }
        Ok(Self {
            tensors,
            resident_bytes,
            _device: device,
        })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Sum of actual Metal buffer lengths, including F32 expansion only for
    /// elementwise state.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
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
        self.tensor(name, TensorRole::RawMatrix)
    }

    pub fn f32_state(&self, name: &str) -> Result<&MlxBuffer, WeightLookupError> {
        self.tensor(name, TensorRole::ElementwiseF32)
    }

    pub fn i32_lookup(&self, name: &str) -> Result<&MlxBuffer, WeightLookupError> {
        self.tensor(name, TensorRole::IntegerLookupI32)
    }
}

fn validate_storage_type(
    name: &str,
    role: TensorRole,
    actual: GgmlType,
) -> Result<(), WeightResidencyError> {
    let valid = match role {
        TensorRole::RawMatrix => !matches!(actual, GgmlType::I16 | GgmlType::I32),
        TensorRole::ElementwiseF32 => actual != GgmlType::I32,
        TensorRole::IntegerLookupI32 => actual == GgmlType::I32,
    };
    if valid {
        Ok(())
    } else {
        Err(WeightResidencyError::StorageType {
            name: name.to_string(),
            role,
            actual,
        })
    }
}

fn validate_loaded_buffer(
    name: &str,
    role: TensorRole,
    disk_bytes: usize,
    buffer: &MlxBuffer,
) -> Result<(), WeightResidencyError> {
    let expected_dtype = match role {
        TensorRole::ElementwiseF32 => Some(DType::F32),
        TensorRole::IntegerLookupI32 => Some(DType::I32),
        TensorRole::RawMatrix => None,
    };
    if let Some(expected) = expected_dtype {
        if buffer.dtype() != expected {
            return Err(WeightResidencyError::LoadedDtype {
                name: name.to_string(),
                expected,
                actual: buffer.dtype(),
            });
        }
    } else if !matches!(buffer.dtype(), DType::F32 | DType::F16 | DType::U8) {
        return Err(WeightResidencyError::LoadedDtype {
            name: name.to_string(),
            expected: DType::U8,
            actual: buffer.dtype(),
        });
    }

    let expected = match role {
        TensorRole::ElementwiseF32 => buffer
            .element_count()
            .checked_mul(DType::F32.size_of())
            .and_then(|bytes| u64::try_from(bytes).ok()),
        _ => u64::try_from(disk_bytes).ok(),
    }
    .ok_or_else(|| WeightResidencyError::ByteOverflow {
        name: name.to_string(),
    })?;
    let actual =
        u64::try_from(buffer.byte_len()).map_err(|_| WeightResidencyError::ByteOverflow {
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
