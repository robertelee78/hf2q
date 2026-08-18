use std::collections::BTreeSet;

use super::super::schema::{
    MetadataGenerationReceiptV1, MetadataSelectorV1, MAX_GENERATION_RECEIPT_BYTES,
};
use super::super::MetadataJournalError;
use super::fault::{trip_prune_entry, FaultPlan};
use super::validation::{read_receipt_from_directory, read_role, verify_generation};
use crate::distribution::install_state::file;
use crate::distribution::install_state::unix::{self, Directory};

pub(super) fn verify_cleanup_residue(
    generations: &Directory,
    selector: &MetadataSelectorV1,
    selected_receipt: &MetadataGenerationReceiptV1,
) -> Result<(), MetadataJournalError> {
    if selector.sequence() == 1 {
        return Ok(());
    }
    let predecessor = selector.sequence() - 1;
    let normal = format!("{predecessor:020}");
    let prune = format!(".prune-{predecessor:020}");
    let normal_exists = unix::entry_identity(generations, &normal)?.is_some();
    let prune_exists = unix::entry_identity(generations, &prune)?.is_some();
    if normal_exists && prune_exists {
        return Err(MetadataJournalError::Invalid(
            "metadata predecessor and prune residue coexist",
        ));
    }
    let Some(name) = normal_exists
        .then_some(normal)
        .or_else(|| prune_exists.then_some(prune))
    else {
        return Ok(());
    };
    let directory = unix::open_directory_at(generations, &name, Some(0o700), true)?;
    let expected_digest =
        selected_receipt
            .predecessor_digest()
            .ok_or(MetadataJournalError::Invalid(
                "selected generation lacks predecessor binding",
            ))?;
    let receipt = if name.starts_with(".prune-") {
        verify_prune_prefix(&directory, expected_digest)?
    } else {
        let receipt = read_receipt_from_directory(&directory)?;
        if receipt.digest()? != expected_digest {
            return Err(MetadataJournalError::Invalid(
                "metadata cleanup residue is not the selected predecessor",
            ));
        }
        verify_generation(&directory, &receipt, None)?;
        Some(receipt)
    };
    if receipt
        .as_ref()
        .is_some_and(|receipt| receipt.sequence() != predecessor)
    {
        return Err(MetadataJournalError::Invalid(
            "metadata cleanup residue is not the selected predecessor",
        ));
    }
    Ok(())
}

pub(super) fn verify_pending_shape(directory: &Directory) -> Result<(), MetadataJournalError> {
    let names = unix::list_names(directory)?;
    let allowed = BTreeSet::from([
        "anchor-root.json".to_owned(),
        "generation.json".to_owned(),
        "root-chain".to_owned(),
        "snapshot.json".to_owned(),
        "targets.json".to_owned(),
        "timestamp.json".to_owned(),
        "trusted-root.json".to_owned(),
    ]);
    if !names.is_subset(&allowed) {
        return Err(MetadataJournalError::Invalid(
            "pending metadata generation contains unexpected state",
        ));
    }
    if names.contains("root-chain") {
        let chain = unix::open_directory_at(directory, "root-chain", Some(0o700), true)?;
        if unix::list_names(&chain)?.iter().any(|name| {
            name.len() != 30
                || !name.ends_with(".root.json")
                || !name[..20].bytes().all(|byte| byte.is_ascii_digit())
        }) {
            return Err(MetadataJournalError::Invalid(
                "pending metadata root history is invalid",
            ));
        }
    }
    Ok(())
}

pub(super) fn remove_generation(
    generations: &Directory,
    name: &str,
    directory: &Directory,
    receipt: Option<&MetadataGenerationReceiptV1>,
    expected_digest: &str,
    faults: FaultPlan,
) -> Result<(), MetadataJournalError> {
    let Some(receipt) = receipt else {
        unix::remove_empty_directory(generations, name, directory)?;
        return Ok(());
    };
    verify_prune_prefix(directory, expected_digest)?;
    let mut removal_step = 0_usize;
    if unix::entry_identity(directory, "root-chain")?.is_some() {
        let root_chain = unix::open_directory_at(directory, "root-chain", Some(0o700), true)?;
        for root_name in receipt.expected_root_names() {
            if unix::entry_identity(&root_chain, &root_name)?.is_none() {
                continue;
            }
            let (file_handle, _, identity) = file::read_regular_file(
                &root_chain,
                &root_name,
                0o600,
                receipt.root_limit(&root_name)?,
            )?;
            let _ = file_handle;
            unix::remove_named_regular_file(&root_chain, &root_name, identity)?;
            unix::sync_directory(&root_chain)?;
            trip_prune_entry(faults, &mut removal_step)?;
        }
        unix::remove_empty_directory(directory, "root-chain", &root_chain)?;
        unix::sync_directory(directory)?;
        trip_prune_entry(faults, &mut removal_step)?;
    }
    for file_name in [
        "anchor-root.json",
        "trusted-root.json",
        "timestamp.json",
        "snapshot.json",
        "targets.json",
    ] {
        if unix::entry_identity(directory, file_name)?.is_none() {
            continue;
        }
        let maximum = receipt.role_limit(file_name)?;
        let (_, _, identity) = file::read_regular_file(directory, file_name, 0o600, maximum)?;
        unix::remove_named_regular_file(directory, file_name, identity)?;
        unix::sync_directory(directory)?;
        trip_prune_entry(faults, &mut removal_step)?;
    }
    if unix::entry_identity(directory, "generation.json")?.is_some() {
        let (_, _, identity) = file::read_regular_file(
            directory,
            "generation.json",
            0o600,
            MAX_GENERATION_RECEIPT_BYTES,
        )?;
        unix::remove_named_regular_file(directory, "generation.json", identity)?;
        unix::sync_directory(directory)?;
        trip_prune_entry(faults, &mut removal_step)?;
    }
    unix::remove_empty_directory(generations, name, directory)?;
    unix::sync_directory(generations)?;
    Ok(())
}

/// Validate the exact suffix left by the deterministic prune order.
///
/// `generation.json` is removed last, so every non-empty partial directory
/// stays bound to the selected generation's predecessor digest. The only
/// receipt-less shape is an empty directory awaiting its final rmdir.
pub(super) fn verify_prune_prefix(
    directory: &Directory,
    expected_digest: &str,
) -> Result<Option<MetadataGenerationReceiptV1>, MetadataJournalError> {
    let names = unix::list_names(directory)?;
    if names.is_empty() {
        return Ok(None);
    }
    if !names.contains("generation.json") {
        return Err(MetadataJournalError::Invalid(
            "partial metadata prune lost its binding receipt",
        ));
    }
    let receipt = read_receipt_from_directory(directory)?;
    if receipt.digest()? != expected_digest {
        return Err(MetadataJournalError::Invalid(
            "partial metadata prune receipt digest is invalid",
        ));
    }

    let removal_order = [
        "root-chain",
        "anchor-root.json",
        "trusted-root.json",
        "timestamp.json",
        "snapshot.json",
        "targets.json",
        "generation.json",
    ];
    let first_remaining = removal_order
        .iter()
        .position(|name| names.contains(*name))
        .ok_or(MetadataJournalError::Invalid(
            "partial metadata prune inventory is invalid",
        ))?;
    let expected_names: BTreeSet<_> = removal_order[first_remaining..]
        .iter()
        .map(|name| (*name).to_owned())
        .collect();
    if names != expected_names {
        return Err(MetadataJournalError::Invalid(
            "partial metadata prune is not an exact removal prefix",
        ));
    }

    if names.contains("root-chain") {
        let chain = unix::open_directory_at(directory, "root-chain", Some(0o700), true)?;
        let expected = receipt.expected_root_names();
        let actual = unix::list_names(&chain)?;
        let first_remaining = expected
            .iter()
            .position(|name| actual.contains(name))
            .unwrap_or(expected.len());
        let expected_remaining: BTreeSet<_> = expected[first_remaining..].iter().cloned().collect();
        if actual != expected_remaining {
            return Err(MetadataJournalError::Invalid(
                "partial root-history prune is not an exact removal prefix",
            ));
        }
        for name in actual {
            let (_, bytes, _) =
                file::read_regular_file(&chain, &name, 0o600, receipt.root_limit(&name)?)?;
            receipt.validate_root_bytes(&name, &bytes)?;
        }
    }
    for name in [
        "anchor-root.json",
        "trusted-root.json",
        "timestamp.json",
        "snapshot.json",
        "targets.json",
    ] {
        if names.contains(name) {
            let _ = read_role(directory, &receipt, name)?;
        }
    }
    Ok(Some(receipt))
}
