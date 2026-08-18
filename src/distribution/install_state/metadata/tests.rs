use super::schema::{
    MetadataGenerationReceiptV1, MetadataSelectorV1, MAX_GENERATION_RECEIPT_BYTES, MAX_ROOT_CHAIN,
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
    hold_metadata_lock_for_test, read_selected, Barrier, FaultPlan,
};
use crate::distribution::install_state::ExplicitRootAuthorization;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn role(name: &str, version: u64, marker: &str) -> ExactMetadataRole {
    ExactMetadataRole {
        request_name: name.to_owned(),
        version,
        bytes: format!("{{\"signed\":\"{marker}\"}}").into_bytes().into(),
    }
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
    VerifiedMetadataCandidate {
        installation_id: "7c907c7a-3125-4a40-a8b3-1c125080e46a".to_owned(),
        state_root: "/Users/example/.hf2q".to_owned(),
        repository_id: "hf2q".to_owned(),
        channel: "stable".to_owned(),
        verification_started_at: started.parse().expect("valid start"),
        verification_completed_at: completed.parse().expect("valid completion"),
        anchor_root: role("1.root.json", 1, "root-1"),
        trusted_root: role(
            &format!("{root_version}.root.json"),
            root_version,
            &format!("root-{root_version}"),
        ),
        root_chain,
        timestamp: role("timestamp.json", timestamp_version, "timestamp"),
        snapshot: role(
            &format!("{timestamp_version}.snapshot.json"),
            timestamp_version,
            "snapshot",
        ),
        targets: role(
            &format!("{timestamp_version}.targets.json"),
            timestamp_version,
            "targets",
        ),
    }
}

fn candidate_at(
    root: &std::path::Path,
    started: &str,
    completed: &str,
    root_version: u64,
    timestamp_version: u64,
) -> VerifiedMetadataCandidate {
    let mut candidate = candidate(started, completed, root_version, timestamp_version);
    candidate.state_root = root.to_str().expect("UTF-8 root").to_owned();
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

mod journal_cases;
mod process_cases;
mod schema_cases;
