//! Routed expert ownership. Storage never selects a different arithmetic route.
use crate::serve::forward_mlx_shared::MlxAffineMoeStack;
use mlx_native::{CompletedGroupedM8Experts, GroupedM8ExpertPlanes, MlxBuffer};

/// A legacy role owns exactly the representation its dispatcher consumes.
pub enum RawExpertRole {
    Ggml(MlxBuffer),
    Affine(MlxAffineMoeStack),
}

pub struct RawMoeExperts {
    pub gate: RawExpertRole,
    pub up: RawExpertRole,
    pub down: RawExpertRole,
}

pub struct CompletedMoeExperts {
    pub gate: CompletedGroupedM8Experts,
    pub up: CompletedGroupedM8Experts,
    pub down: CompletedGroupedM8Experts,
}

/// Converted main expert allocations, published only after source retirement.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CanonicalExpertStorageBytes {
    pub raw_bytes: u64,
    pub prepared_bytes: u64,
}

/// Shared by the loaded model and execution caches. A completed triplet cannot
/// retain an unused GGML source or a partially converted raw role.
pub enum MoeExpertStorage {
    Raw(RawMoeExperts),
    CompletedPrepared(CompletedMoeExperts),
}

#[derive(Clone, Copy)]
pub enum ExpertRole {
    Gate,
    Up,
    Down,
}

/// Borrowed dispatch input; creating it neither clones nor reconstructs weights.
pub enum ExpertRoleRef<'a> {
    Ggml(&'a MlxBuffer),
    Affine(&'a MlxAffineMoeStack),
    Prepared(&'a GroupedM8ExpertPlanes),
}

impl RawExpertRole {
    fn borrow(&self) -> ExpertRoleRef<'_> {
        match self {
            Self::Ggml(buffer) => ExpertRoleRef::Ggml(buffer),
            Self::Affine(stack) => ExpertRoleRef::Affine(stack),
        }
    }
}

impl MoeExpertStorage {
    pub fn from_ggml(gate: MlxBuffer, up: MlxBuffer, down: MlxBuffer) -> Self {
        Self::Raw(RawMoeExperts {
            gate: RawExpertRole::Ggml(gate),
            up: RawExpertRole::Ggml(up),
            down: RawExpertRole::Ggml(down),
        })
    }

    pub fn role(&self, role: ExpertRole) -> ExpertRoleRef<'_> {
        match self {
            Self::Raw(raw) => match role {
                ExpertRole::Gate => raw.gate.borrow(),
                ExpertRole::Up => raw.up.borrow(),
                ExpertRole::Down => raw.down.borrow(),
            },
            Self::CompletedPrepared(planes) => ExpertRoleRef::Prepared(match role {
                ExpertRole::Gate => &planes.gate,
                ExpertRole::Up => &planes.up,
                ExpertRole::Down => &planes.down,
            }),
        }
    }

    pub fn completed(&self) -> Option<&CompletedMoeExperts> {
        match self {
            Self::CompletedPrepared(planes) => Some(planes),
            Self::Raw(_) => None,
        }
    }

    pub fn has_affine(&self) -> bool {
        match self {
            Self::Raw(raw) => [&raw.gate, &raw.up, &raw.down]
                .into_iter()
                .any(|role| matches!(role, RawExpertRole::Affine(_))),
            Self::CompletedPrepared(_) => false,
        }
    }

    pub fn raw_mut(&mut self) -> Option<&mut RawMoeExperts> {
        match self {
            Self::Raw(raw) => Some(raw),
            Self::CompletedPrepared(_) => None,
        }
    }
}
