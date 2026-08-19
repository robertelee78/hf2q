use super::super::InstallStateError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::distribution) enum IdentityBarrier {
    IntentNameSync,
    IntentContentSync,
    IntentFullSync,
    PrecommitReopen,
    IdentityRename,
    FinalReopen,
    FinalFullSync,
    UpdateSync,
    RootSync,
    LockFullSync,
}

impl IdentityBarrier {
    #[cfg(test)]
    pub(super) const ALL: [Self; 10] = [
        Self::IntentNameSync,
        Self::IntentContentSync,
        Self::IntentFullSync,
        Self::PrecommitReopen,
        Self::IdentityRename,
        Self::FinalReopen,
        Self::FinalFullSync,
        Self::UpdateSync,
        Self::RootSync,
        Self::LockFullSync,
    ];

    #[cfg(test)]
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::IntentNameSync => "intent-name-sync",
            Self::IntentContentSync => "intent-content-sync",
            Self::IntentFullSync => "intent-full-sync",
            Self::PrecommitReopen => "precommit-reopen",
            Self::IdentityRename => "identity-rename",
            Self::FinalReopen => "final-reopen",
            Self::FinalFullSync => "final-full-sync",
            Self::UpdateSync => "update-sync",
            Self::RootSync => "root-sync",
            Self::LockFullSync => "lock-full-sync",
        }
    }

    #[cfg(test)]
    pub(super) fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|barrier| barrier.name() == value)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(in crate::distribution) struct IdentityFaultPlan {
    selected: Option<IdentityBarrier>,
}

impl IdentityFaultPlan {
    #[cfg(test)]
    pub(in crate::distribution) fn once(barrier: IdentityBarrier) -> Self {
        Self {
            selected: Some(barrier),
        }
    }
}

pub(super) fn trip(
    faults: IdentityFaultPlan,
    barrier: IdentityBarrier,
) -> Result<(), InstallStateError> {
    if faults.selected != Some(barrier) {
        return Ok(());
    }
    #[cfg(test)]
    if std::env::var_os("HF2Q_IDENTITY_ABORT_ON_FAULT").is_some() {
        std::process::abort();
    }
    Err(InstallStateError::std_io(
        "injected installation-identity durability barrier",
        std::io::Error::other("test-only installation-identity barrier failure"),
    ))
}
