use std::collections::BTreeSet;

use super::super::schema::{
    MetadataGenerationReceiptV2, MetadataSelectorV2, MAX_GENERATION_RECEIPT_BYTES,
    MAX_SELECTOR_BYTES,
};
use super::super::{
    MetadataJournalError, MetadataStateAuthorization, StoredMetadataGeneration,
    VerifiedMetadataCandidate,
};
use super::cleanup::{verify_cleanup_residue, verify_pending_shape};
use super::{HistoryMode, CURRENT, GENERATIONS, METADATA};
use crate::distribution::install_state::file;
use crate::distribution::install_state::unix::{self, Directory};
use crate::distribution::install_state::InstallStateError;

pub(in crate::distribution) fn read_selected(
    authorization: &MetadataStateAuthorization,
) -> Result<Option<StoredMetadataGeneration>, MetadataJournalError> {
    let root = match unix::open_existing_root(&authorization.root.path) {
        Ok(root) => root,
        Err(InstallStateError::Missing(_)) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let update = match unix::entry_identity(&root, "update")? {
        None => return Ok(None),
        Some(_) => unix::open_directory_at(&root, "update", Some(0o700), true)?,
    };
    let metadata = match unix::entry_identity(&update, METADATA)? {
        None => return Ok(None),
        Some(_) => unix::open_directory_at(&update, METADATA, Some(0o700), true)?,
    };
    let generations = unix::open_directory_at(&metadata, GENERATIONS, Some(0o700), true)?;
    let stored = read_selected_with_mode(&metadata, &generations, HistoryMode::Authority)?;
    bind_selection(&stored, authorization)?;

    // Reopen and re-read every named boundary before returning bytes. A
    // detached descriptor must never silently become selected authority.
    let fresh_root = unix::open_existing_root(&authorization.root.path)?;
    require_same_directory(&fresh_root, &root)?;
    let fresh_update = unix::open_directory_at(&fresh_root, "update", Some(0o700), true)?;
    require_same_directory(&fresh_update, &update)?;
    let fresh_metadata = unix::open_directory_at(&fresh_update, METADATA, Some(0o700), true)?;
    require_same_directory(&fresh_metadata, &metadata)?;
    let fresh_generations =
        unix::open_directory_at(&fresh_metadata, GENERATIONS, Some(0o700), true)?;
    require_same_directory(&fresh_generations, &generations)?;
    let repeated =
        read_selected_with_mode(&fresh_metadata, &fresh_generations, HistoryMode::Authority)?;
    bind_selection(&repeated, authorization)?;
    match (&stored, &repeated) {
        (None, None) => Ok(None),
        (Some(first), Some(second))
            if first.sequence == second.sequence
                && first.generation_receipt == second.generation_receipt =>
        {
            Ok(stored)
        }
        _ => Err(MetadataJournalError::Invalid(
            "selected metadata changed while it was being read",
        )),
    }
}

pub(super) fn write_generation(
    directory: &Directory,
    candidate: &VerifiedMetadataCandidate,
    receipt: &[u8],
) -> Result<(), MetadataJournalError> {
    let root_chain = unix::ensure_private_directory(directory, "root-chain")?;
    for root in candidate.root_chain() {
        let name = format!("{:020}.root.json", root.version());
        let file = file::write_or_resume_private_file(&root_chain, &name, root.bytes())?;
        unix::full_sync_file(&file)?;
    }
    unix::sync_directory(&root_chain)?;
    for (name, bytes) in [
        ("anchor-root.json", candidate.anchor_root().bytes()),
        ("trusted-root.json", candidate.trusted_root().bytes()),
        ("timestamp.json", candidate.timestamp().bytes()),
        ("snapshot.json", candidate.snapshot().bytes()),
        ("targets.json", candidate.targets().bytes()),
        ("generation.json", receipt),
    ] {
        let file = file::write_or_resume_private_file(directory, name, bytes)?;
        unix::full_sync_file(&file)?;
    }
    Ok(())
}

pub(super) fn verify_generation(
    directory: &Directory,
    receipt: &MetadataGenerationReceiptV2,
    candidate: Option<&VerifiedMetadataCandidate>,
) -> Result<StoredMetadataGeneration, MetadataJournalError> {
    let expected = BTreeSet::from([
        "anchor-root.json".to_owned(),
        "generation.json".to_owned(),
        "root-chain".to_owned(),
        "snapshot.json".to_owned(),
        "targets.json".to_owned(),
        "timestamp.json".to_owned(),
        "trusted-root.json".to_owned(),
    ]);
    if unix::list_names(directory)? != expected {
        return Err(MetadataJournalError::Invalid(
            "metadata generation inventory is not exact",
        ));
    }
    let (_, receipt_bytes, _) = file::read_regular_file(
        directory,
        "generation.json",
        0o600,
        MAX_GENERATION_RECEIPT_BYTES,
    )?;
    if MetadataGenerationReceiptV2::parse(&receipt_bytes)? != *receipt {
        return Err(MetadataJournalError::Invalid(
            "stored metadata receipt changed after validation",
        ));
    }
    if candidate.is_some_and(|candidate| !receipt.matches_candidate(candidate)) {
        return Err(MetadataJournalError::Invalid(
            "metadata receipt does not bind the verified candidate",
        ));
    }
    let root_chain_dir = unix::open_directory_at(directory, "root-chain", Some(0o700), true)?;
    let expected_roots: BTreeSet<_> = receipt.expected_root_names().into_iter().collect();
    if unix::list_names(&root_chain_dir)? != expected_roots {
        return Err(MetadataJournalError::Invalid(
            "stored root history inventory is not exact",
        ));
    }
    let mut root_chain = Vec::with_capacity(expected_roots.len());
    for name in expected_roots {
        let (_, bytes, _) =
            file::read_regular_file(&root_chain_dir, &name, 0o600, receipt.root_limit(&name)?)?;
        receipt.validate_root_bytes(&name, &bytes)?;
        root_chain.push(bytes.into_boxed_slice());
    }
    let anchor_root = read_role(directory, receipt, "anchor-root.json")?;
    let trusted_root = read_role(directory, receipt, "trusted-root.json")?;
    let timestamp = read_role(directory, receipt, "timestamp.json")?;
    let snapshot = read_role(directory, receipt, "snapshot.json")?;
    let targets = read_role(directory, receipt, "targets.json")?;
    Ok(StoredMetadataGeneration {
        sequence: receipt.sequence(),
        generation_receipt: receipt_bytes.into_boxed_slice(),
        anchor_root,
        root_chain,
        trusted_root,
        timestamp,
        snapshot,
        targets,
    })
}

pub(super) fn read_role(
    directory: &Directory,
    receipt: &MetadataGenerationReceiptV2,
    name: &str,
) -> Result<Box<[u8]>, MetadataJournalError> {
    let (_, bytes, _) = file::read_regular_file(directory, name, 0o600, receipt.role_limit(name)?)?;
    receipt.validate_role_bytes(name, &bytes)?;
    Ok(bytes.into_boxed_slice())
}

pub(super) fn read_selector_with_mode(
    metadata: &Directory,
    generations: &Directory,
    mode: HistoryMode,
) -> Result<Option<MetadataSelectorV2>, MetadataJournalError> {
    let selector = unix::entry_identity(metadata, CURRENT)?
        .map(|_| {
            let (_, bytes, _) =
                file::read_regular_file(metadata, CURRENT, 0o600, MAX_SELECTOR_BYTES)?;
            MetadataSelectorV2::parse(&bytes)
        })
        .transpose()?;
    verify_history(metadata, generations, selector.as_ref(), mode)?;
    Ok(selector)
}

pub(super) fn read_selected_with_mode(
    metadata: &Directory,
    generations: &Directory,
    mode: HistoryMode,
) -> Result<Option<StoredMetadataGeneration>, MetadataJournalError> {
    let selector = read_selector_with_mode(metadata, generations, mode)?;
    selector
        .as_ref()
        .map(|selector| {
            let generation = unix::open_directory_at(
                generations,
                &format!("{:020}", selector.sequence()),
                Some(0o700),
                true,
            )?;
            let receipt = read_receipt(generations, selector)?;
            verify_generation(&generation, &receipt, None)
        })
        .transpose()
}

fn verify_history(
    metadata: &Directory,
    generations: &Directory,
    selector: Option<&MetadataSelectorV2>,
    mode: HistoryMode,
) -> Result<(), MetadataJournalError> {
    let next = selector.map_or(Ok(1), |selector| {
        selector
            .sequence()
            .checked_add(1)
            .ok_or(MetadataJournalError::Invalid(
                "metadata generation sequence overflowed",
            ))
    })?;
    let pending_generation = format!(".pending-{next:020}");
    let next_generation = format!("{next:020}");
    let pending_selector = format!(".current-{next:020}.json");
    let metadata_names = unix::list_names(metadata)?;
    let mut expected_metadata = BTreeSet::from([GENERATIONS.to_owned()]);
    if selector.is_some() {
        expected_metadata.insert(CURRENT.to_owned());
    }
    let has_pending_selector = metadata_names.contains(&pending_selector);
    if has_pending_selector {
        expected_metadata.insert(pending_selector.clone());
    }
    if metadata_names != expected_metadata {
        return Err(MetadataJournalError::Invalid(
            "metadata journal inventory is not exact",
        ));
    }

    let generation_names = unix::list_names(generations)?;
    let has_pending_generation = generation_names.contains(&pending_generation);
    let has_next_generation = generation_names.contains(&next_generation);
    if has_pending_generation && has_next_generation {
        return Err(MetadataJournalError::Invalid(
            "pending and published successor generations coexist",
        ));
    }
    if has_pending_selector && !has_next_generation {
        return Err(MetadataJournalError::Invalid(
            "pending selector lacks a published successor",
        ));
    }
    if mode == HistoryMode::Authority && (has_pending_generation || has_next_generation) {
        return Err(MetadataJournalError::Invalid(
            "metadata successor transaction requires lock-held recovery",
        ));
    }

    let mut expected_generations = BTreeSet::new();
    if let Some(selector) = selector {
        expected_generations.insert(format!("{:020}", selector.sequence()));
        if selector.sequence() > 1 {
            let predecessor = selector.sequence() - 1;
            let normal = format!("{predecessor:020}");
            let prune = format!(".prune-{predecessor:020}");
            if generation_names.contains(&normal) {
                expected_generations.insert(normal);
            }
            if generation_names.contains(&prune) {
                expected_generations.insert(prune);
            }
        }
    }
    if has_pending_generation {
        expected_generations.insert(pending_generation.clone());
    }
    if has_next_generation {
        expected_generations.insert(next_generation.clone());
    }
    if generation_names != expected_generations {
        return Err(MetadataJournalError::Invalid(
            "metadata generation inventory is not bounded and exact",
        ));
    }
    if has_pending_generation {
        verify_pending_shape(&unix::open_directory_at(
            generations,
            &pending_generation,
            Some(0o700),
            true,
        )?)?;
    }
    if let Some(selector) = selector {
        let receipt = read_receipt(generations, selector)?;
        let selected = unix::open_directory_at(
            generations,
            &format!("{:020}", selector.sequence()),
            Some(0o700),
            true,
        )?;
        verify_generation(&selected, &receipt, None)?;
        verify_cleanup_residue(generations, selector, &receipt)?;
    }
    if has_next_generation {
        let successor = unix::open_directory_at(generations, &next_generation, Some(0o700), true)?;
        let successor_receipt = read_receipt_from_directory(&successor)?;
        if successor_receipt.sequence() != next {
            return Err(MetadataJournalError::Invalid(
                "published successor sequence is inconsistent",
            ));
        }
        verify_generation(&successor, &successor_receipt, None)?;
        if let Some(selected) = selector {
            let prior = read_receipt(generations, selected)?;
            successor_receipt.validate_successor(&prior, selected.generation_sha256())?;
        }
        if has_pending_selector {
            let (_, bytes, _) =
                file::read_regular_file(metadata, &pending_selector, 0o600, MAX_SELECTOR_BYTES)?;
            let staged = MetadataSelectorV2::parse(&bytes)?;
            if staged.sequence() != next
                || staged.generation_sha256() != successor_receipt.digest()?
            {
                return Err(MetadataJournalError::Invalid(
                    "pending metadata selector does not bind its successor",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn read_receipt(
    generations: &Directory,
    selector: &MetadataSelectorV2,
) -> Result<MetadataGenerationReceiptV2, MetadataJournalError> {
    let generation = unix::open_directory_at(
        generations,
        &format!("{:020}", selector.sequence()),
        Some(0o700),
        true,
    )?;
    let receipt = read_receipt_from_directory(&generation)?;
    if receipt.sequence() != selector.sequence()
        || receipt.digest()? != selector.generation_sha256()
    {
        return Err(MetadataJournalError::Invalid(
            "metadata selector does not bind its generation receipt",
        ));
    }
    Ok(receipt)
}

pub(super) fn read_receipt_from_directory(
    directory: &Directory,
) -> Result<MetadataGenerationReceiptV2, MetadataJournalError> {
    let (_, bytes, _) = file::read_regular_file(
        directory,
        "generation.json",
        0o600,
        MAX_GENERATION_RECEIPT_BYTES,
    )?;
    MetadataGenerationReceiptV2::parse(&bytes)
}

pub(super) fn generation_matches_candidate(
    generations: &Directory,
    sequence: u64,
    candidate: &VerifiedMetadataCandidate,
) -> Result<bool, MetadataJournalError> {
    let name = format!("{sequence:020}");
    let generation = unix::open_directory_at(generations, &name, Some(0o700), true)?;
    let receipt = read_receipt_from_directory(&generation)?;
    if !receipt.matches_candidate(candidate) {
        return Ok(false);
    }
    verify_generation(&generation, &receipt, Some(candidate)).map(|_| true)
}

pub(super) fn require_same_directory(
    actual: &Directory,
    expected: &Directory,
) -> Result<(), MetadataJournalError> {
    if !actual.same_object(expected) {
        return Err(MetadataJournalError::Invalid(
            "metadata journal namespace changed after verification",
        ));
    }
    Ok(())
}

fn bind_selection(
    selected: &Option<StoredMetadataGeneration>,
    authorization: &MetadataStateAuthorization,
) -> Result<(), MetadataJournalError> {
    if let Some(selected) = selected {
        MetadataGenerationReceiptV2::parse(&selected.generation_receipt)?.validate_state_identity(
            &authorization.installation_id,
            authorization.root.canonical.as_str(),
        )?;
    }
    Ok(())
}
