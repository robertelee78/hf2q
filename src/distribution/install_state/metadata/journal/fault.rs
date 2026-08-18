use super::super::MetadataJournalError;

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::distribution::install_state::metadata) enum Barrier {
    PendingDirectory,
    GenerationFiles,
    GenerationPublish,
    GenerationsSync,
    SelectorFile,
    MetadataPrecommitSync,
    SelectorCommit,
    SelectorFullSync,
    GenerationPostcommitSync,
    MetadataPostcommitSync,
    UpdatePostcommitSync,
    RootPostcommitSync,
    PredecessorPruneRename,
    PredecessorPruneEntryRemoval(usize),
    PredecessorPruneRemoval,
    PredecessorPruneFullSync,
}

#[cfg(test)]
impl Barrier {
    pub(in crate::distribution::install_state::metadata) fn name(self) -> String {
        match self {
            Self::PendingDirectory => "pending-directory".to_owned(),
            Self::GenerationFiles => "generation-files".to_owned(),
            Self::GenerationPublish => "generation-publish".to_owned(),
            Self::GenerationsSync => "generations-sync".to_owned(),
            Self::SelectorFile => "selector-file".to_owned(),
            Self::MetadataPrecommitSync => "metadata-precommit-sync".to_owned(),
            Self::SelectorCommit => "selector-commit".to_owned(),
            Self::SelectorFullSync => "selector-full-sync".to_owned(),
            Self::GenerationPostcommitSync => "generation-postcommit-sync".to_owned(),
            Self::MetadataPostcommitSync => "metadata-postcommit-sync".to_owned(),
            Self::UpdatePostcommitSync => "update-postcommit-sync".to_owned(),
            Self::RootPostcommitSync => "root-postcommit-sync".to_owned(),
            Self::PredecessorPruneRename => "predecessor-prune-rename".to_owned(),
            Self::PredecessorPruneEntryRemoval(step) => {
                format!("predecessor-prune-entry-{step}")
            }
            Self::PredecessorPruneRemoval => "predecessor-prune-removal".to_owned(),
            Self::PredecessorPruneFullSync => "predecessor-prune-full-sync".to_owned(),
        }
    }

    pub(in crate::distribution::install_state::metadata) fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "pending-directory" => Self::PendingDirectory,
            "generation-files" => Self::GenerationFiles,
            "generation-publish" => Self::GenerationPublish,
            "generations-sync" => Self::GenerationsSync,
            "selector-file" => Self::SelectorFile,
            "metadata-precommit-sync" => Self::MetadataPrecommitSync,
            "selector-commit" => Self::SelectorCommit,
            "selector-full-sync" => Self::SelectorFullSync,
            "generation-postcommit-sync" => Self::GenerationPostcommitSync,
            "metadata-postcommit-sync" => Self::MetadataPostcommitSync,
            "update-postcommit-sync" => Self::UpdatePostcommitSync,
            "root-postcommit-sync" => Self::RootPostcommitSync,
            "predecessor-prune-rename" => Self::PredecessorPruneRename,
            "predecessor-prune-removal" => Self::PredecessorPruneRemoval,
            "predecessor-prune-full-sync" => Self::PredecessorPruneFullSync,
            _ => {
                let step = value
                    .strip_prefix("predecessor-prune-entry-")?
                    .parse()
                    .ok()?;
                Self::PredecessorPruneEntryRemoval(step)
            }
        })
    }
}
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
pub(in crate::distribution::install_state::metadata) struct FaultPlan {
    pub(in crate::distribution::install_state::metadata) barrier: Option<Barrier>,
}

#[cfg(not(test))]
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct FaultPlan {
    _private: (),
}

impl FaultPlan {
    #[cfg(test)]
    fn trip(self, barrier: Barrier) -> Result<(), MetadataJournalError> {
        if self.barrier == Some(barrier) {
            if std::env::var_os("HF2Q_METADATA_ABORT_ON_FAULT").is_some() {
                std::process::abort();
            }
            return Err(MetadataJournalError::Invalid(
                "injected metadata journal failure",
            ));
        }
        Ok(())
    }

    #[cfg(not(test))]
    fn trip(self, _barrier: ()) -> Result<(), MetadataJournalError> {
        Ok(())
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) enum TestBarrier {
    PendingDirectory,
    GenerationFiles,
    GenerationPublish,
    GenerationsSync,
    SelectorFile,
    MetadataPrecommitSync,
    SelectorCommit,
    SelectorFullSync,
    GenerationPostcommitSync,
    MetadataPostcommitSync,
    UpdatePostcommitSync,
    RootPostcommitSync,
    PredecessorPruneRename,
    PredecessorPruneEntryRemoval(usize),
    PredecessorPruneRemoval,
    PredecessorPruneFullSync,
}

#[cfg(not(test))]
#[derive(Clone, Copy)]
pub(super) enum TestBarrier {
    PendingDirectory,
    GenerationFiles,
    GenerationPublish,
    GenerationsSync,
    SelectorFile,
    MetadataPrecommitSync,
    SelectorCommit,
    SelectorFullSync,
    GenerationPostcommitSync,
    MetadataPostcommitSync,
    UpdatePostcommitSync,
    RootPostcommitSync,
    PredecessorPruneRename,
    PredecessorPruneEntryRemoval(usize),
    PredecessorPruneRemoval,
    PredecessorPruneFullSync,
}

pub(super) fn trip(faults: FaultPlan, barrier: TestBarrier) -> Result<(), MetadataJournalError> {
    #[cfg(test)]
    {
        let selected = match barrier {
            TestBarrier::PendingDirectory => Barrier::PendingDirectory,
            TestBarrier::GenerationFiles => Barrier::GenerationFiles,
            TestBarrier::GenerationPublish => Barrier::GenerationPublish,
            TestBarrier::GenerationsSync => Barrier::GenerationsSync,
            TestBarrier::SelectorFile => Barrier::SelectorFile,
            TestBarrier::MetadataPrecommitSync => Barrier::MetadataPrecommitSync,
            TestBarrier::SelectorCommit => Barrier::SelectorCommit,
            TestBarrier::SelectorFullSync => Barrier::SelectorFullSync,
            TestBarrier::GenerationPostcommitSync => Barrier::GenerationPostcommitSync,
            TestBarrier::MetadataPostcommitSync => Barrier::MetadataPostcommitSync,
            TestBarrier::UpdatePostcommitSync => Barrier::UpdatePostcommitSync,
            TestBarrier::RootPostcommitSync => Barrier::RootPostcommitSync,
            TestBarrier::PredecessorPruneRename => Barrier::PredecessorPruneRename,
            TestBarrier::PredecessorPruneEntryRemoval(step) => {
                Barrier::PredecessorPruneEntryRemoval(step)
            }
            TestBarrier::PredecessorPruneRemoval => Barrier::PredecessorPruneRemoval,
            TestBarrier::PredecessorPruneFullSync => Barrier::PredecessorPruneFullSync,
        };
        return faults.trip(selected);
    }
    #[cfg(not(test))]
    {
        let _ = (faults, barrier);
        Ok(())
    }
}

pub(super) fn trip_prune_entry(
    faults: FaultPlan,
    removal_step: &mut usize,
) -> Result<(), MetadataJournalError> {
    *removal_step = removal_step
        .checked_add(1)
        .ok_or(MetadataJournalError::Invalid(
            "metadata prune step overflowed",
        ))?;
    trip(
        faults,
        TestBarrier::PredecessorPruneEntryRemoval(*removal_step),
    )
}
