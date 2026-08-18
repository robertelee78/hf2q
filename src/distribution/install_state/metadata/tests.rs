use super::schema::{
    MetadataGenerationReceiptV2, MetadataSelectorV2, MAX_GENERATION_RECEIPT_BYTES, MAX_ROOT_CHAIN,
    MAX_SELECTOR_BYTES,
};
use super::{
    ExactMetadataRole, MetadataCommitOutcome, MetadataJournalError, MetadataStateAuthorization,
    VerifiedMetadataCandidate,
};
use crate::distribution::install_state::locked::LockedInstallation;
use crate::distribution::install_state::metadata::journal::{
    cleanup_selected_with_hook_for_test, commit_candidate_for_test,
    commit_candidate_with_hook_for_test, commit_candidate_with_precommit_hook_for_test,
    discard_unselected_for_test, discard_unselected_with_hook_for_test,
    hold_metadata_lock_for_test, read_selected, Barrier, FaultPlan,
};
use crate::distribution::install_state::ExplicitRootAuthorization;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn role(name: &str, version: u64, marker: &str) -> ExactMetadataRole {
    ExactMetadataRole::for_test(
        name,
        version,
        format!("{{\"signed\":\"{marker}\"}}").into_bytes(),
    )
}

fn candidate(
    started: &str,
    completed: &str,
    root_version: u64,
    timestamp_version: u64,
) -> VerifiedMetadataCandidate {
    let mut root_chain = Vec::new();
    for version in 2..=root_version {
        root_chain.push(role(
            &format!("{version}.root.json"),
            version,
            &format!("root-{version}"),
        ));
    }
    VerifiedMetadataCandidate::for_test(
        "7c907c7a-3125-4a40-a8b3-1c125080e46a".to_owned(),
        "/Users/example/.hf2q".to_owned(),
        started.parse().expect("valid start"),
        completed.parse().expect("valid completion"),
        role("1.root.json", 1, "root-1"),
        root_chain,
        role(
            &format!("{root_version}.root.json"),
            root_version,
            &format!("root-{root_version}"),
        ),
        role("timestamp.json", timestamp_version, "timestamp"),
        role(
            &format!("{timestamp_version}.snapshot.json"),
            timestamp_version,
            "snapshot",
        ),
        role(
            &format!("{timestamp_version}.targets.json"),
            timestamp_version,
            "targets",
        ),
    )
}

fn candidate_at(
    root: &std::path::Path,
    started: &str,
    completed: &str,
    root_version: u64,
    timestamp_version: u64,
) -> VerifiedMetadataCandidate {
    let mut candidate = candidate(started, completed, root_version, timestamp_version);
    candidate.set_state_root_for_test(root.to_str().expect("UTF-8 root"));
    candidate
}

fn authorization(root: &std::path::Path) -> MetadataStateAuthorization {
    MetadataStateAuthorization::for_test(
        ExplicitRootAuthorization::new(root).expect("explicit root authorization"),
        "7c907c7a-3125-4a40-a8b3-1c125080e46a",
    )
}

fn test_root(parent: &tempfile::TempDir) -> std::path::PathBuf {
    // macOS exposes /var as a symlink; descriptor-relative production
    // traversal intentionally rejects it.
    parent
        .path()
        .canonicalize()
        .expect("canonical temp path")
        .join("state")
}

fn initial_transaction_barriers() -> Vec<Barrier> {
    vec![
        Barrier::PendingDirectory,
        Barrier::GenerationFiles,
        Barrier::GenerationPublish,
        Barrier::GenerationsSync,
        Barrier::SelectorFile,
        Barrier::MetadataPrecommitSync,
        Barrier::SelectorCommit,
        Barrier::SelectorFullSync,
        Barrier::GenerationPostcommitSync,
        Barrier::MetadataPostcommitSync,
        Barrier::UpdatePostcommitSync,
        Barrier::RootPostcommitSync,
    ]
}

fn successor_transaction_barriers() -> Vec<Barrier> {
    let mut barriers = initial_transaction_barriers();
    barriers.push(Barrier::PredecessorPruneRename);
    barriers.extend(prune_entry_barriers(1));
    barriers.push(Barrier::PredecessorPruneRemoval);
    barriers.push(Barrier::PredecessorPruneFullSync);
    barriers
}

fn prune_entry_barriers(root_history_entries: usize) -> impl Iterator<Item = Barrier> {
    // One barrier follows each root-history unlink, the root-chain rmdir,
    // five top-level role unlinks, and the receipt unlink.
    (1..=root_history_entries + 7).map(Barrier::PredecessorPruneEntryRemoval)
}

fn successor_discard_barriers(root_history_entries: usize) -> Vec<Barrier> {
    let mut barriers = vec![
        Barrier::SuccessorDiscardSelector,
        Barrier::SuccessorDiscardRename,
    ];
    // Reverse removal visits six fixed files, each root-history file, and the
    // root-chain directory before removing the pending generation itself.
    barriers.extend((1..=root_history_entries + 7).map(Barrier::SuccessorDiscardEntryRemoval));
    barriers.push(Barrier::SuccessorDiscardDirectory);
    barriers.push(Barrier::SuccessorDiscardGenerationsSync);
    barriers.push(Barrier::SuccessorDiscardMetadataSync);
    barriers.push(Barrier::SuccessorDiscardUpdateSync);
    barriers.push(Barrier::SuccessorDiscardRootSync);
    barriers.push(Barrier::SuccessorDiscardFullSync);
    barriers
}

mod journal_cases;
mod process_cases;
mod recovery_cases;
mod schema_cases;
