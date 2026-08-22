use std::collections::BTreeMap;

use anyhow::{ensure, Result};
use mlx_native::GgmlType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TensorRole {
    F32State,
    Embedding,
    DenseProjection,
    FfnGateUp,
    FfnDown,
}

impl TensorRole {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::F32State => "f32_state",
            Self::Embedding => "embedding",
            Self::DenseProjection => "dense_projection",
            Self::FfnGateUp => "ffn_gate_up",
            Self::FfnDown => "ffn_down",
        }
    }
}

/// Internal type vocabulary used by the model-free artifact matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TensorStorage {
    Parsed(GgmlType),
}

impl TensorStorage {
    fn label(self) -> String {
        match self {
            Self::Parsed(kind) => format!("{kind:?}"),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Qwen35GgufPreflightReceipt {
    pub(super) required_tensor_count: usize,
    pub(super) role_counts: BTreeMap<&'static str, usize>,
    pub(super) storage_counts: BTreeMap<String, usize>,
}

impl Qwen35GgufPreflightReceipt {
    pub(super) fn record(&mut self, role: TensorRole, storage: TensorStorage) {
        self.required_tensor_count += 1;
        *self.role_counts.entry(role.label()).or_default() += 1;
        *self.storage_counts.entry(storage.label()).or_default() += 1;
    }
}

pub(super) fn admit_storage_for_role(
    name: &str,
    role: TensorRole,
    storage: TensorStorage,
) -> Result<()> {
    let TensorStorage::Parsed(kind) = storage;

    let admitted = match role {
        TensorRole::F32State => kind == GgmlType::F32,
        TensorRole::Embedding => matches!(
            kind,
            GgmlType::F32
                | GgmlType::F16
                | GgmlType::BF16
                | GgmlType::Q2_K
                | GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::Q8_0
        ),
        TensorRole::DenseProjection => matches!(
            kind,
            GgmlType::F32
                | GgmlType::F16
                | GgmlType::BF16
                | GgmlType::Q4_0
                | GgmlType::Q8_0
                | GgmlType::Q2_K
                | GgmlType::Q3_K
                | GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::Q5_1
                | GgmlType::IQ4_NL
                | GgmlType::IQ4_XS
        ),
        TensorRole::FfnGateUp | TensorRole::FfnDown => matches!(
            kind,
            GgmlType::F16
                | GgmlType::BF16
                | GgmlType::F32
                | GgmlType::Q4_0
                | GgmlType::Q8_0
                | GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
        ),
    };
    ensure!(
        admitted,
        "Qwen GGUF preflight rejects tensor '{name}' ({}) at storage {kind:?}; no substitution or runtime re-encoding is permitted",
        role.label()
    );
    Ok(())
}

pub(super) fn admit_mtp_tensor_presence(
    dedicated: bool,
    embed_present: bool,
    head_present: bool,
) -> Result<()> {
    if dedicated {
        ensure!(
            embed_present && head_present,
            "Qwen GGUF preflight: dedicated MTP requires both dedicated embedding and head tensors"
        );
    } else {
        ensure!(
            !embed_present && !head_present,
            "Qwen GGUF preflight: shared MTP forbids dedicated embedding or head tensors"
        );
    }
    Ok(())
}
