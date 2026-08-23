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
    ExpertStack,
}

impl TensorRole {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::F32State => "f32_state",
            Self::Embedding => "embedding",
            Self::DenseProjection => "dense_projection",
            Self::FfnGateUp => "ffn_gate_up",
            Self::FfnDown => "ffn_down",
            Self::ExpertStack => "expert_stack",
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
pub(in crate::inference::models::qwen35) struct Qwen35GgufPreflightReceipt {
    pub(super) required_tensor_count: usize,
    pub(super) role_counts: BTreeMap<&'static str, usize>,
    pub(super) storage_counts: BTreeMap<String, usize>,
    /// Unique rank-2/rank-3 matrices expected in the loaded model. Tied and
    /// shared-MTP aliases are not separate tensors and therefore count once.
    pub(in crate::inference::models::qwen35) matrix_tensor_count: usize,
    pub(in crate::inference::models::qwen35) matrix_bytes: u64,
}

impl Qwen35GgufPreflightReceipt {
    pub(super) fn record(&mut self, role: TensorRole, storage: TensorStorage) {
        self.required_tensor_count += 1;
        *self.role_counts.entry(role.label()).or_default() += 1;
        *self.storage_counts.entry(storage.label()).or_default() += 1;
    }

    pub(super) fn record_tensor(
        &mut self,
        role: TensorRole,
        storage: TensorStorage,
        byte_len: usize,
    ) -> Result<()> {
        self.record(role, storage);
        if role != TensorRole::F32State {
            self.matrix_tensor_count += 1;
            self.matrix_bytes = self
                .matrix_bytes
                .checked_add(u64::try_from(byte_len)?)
                .ok_or_else(|| anyhow::anyhow!("Qwen matrix byte receipt overflow"))?;
        }
        Ok(())
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
        TensorRole::ExpertStack => matches!(
            kind,
            GgmlType::Q4_0
                | GgmlType::Q5_1
                | GgmlType::Q8_0
                | GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::IQ4_NL
                | GgmlType::IQ4_XS
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
