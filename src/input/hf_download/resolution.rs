use std::collections::BTreeSet;
use std::fmt;

use super::{validate_repo_inventory, DownloadError};
use crate::input::hf_reference::{HfModelReference, ResolvedHfModelReference};

/// One bounded repository inventory resolved by the pinned in-process Hub
/// client before any model payload is transferred.
///
/// This value is structural resolution evidence only. It grants no download,
/// conversion, filesystem, or serving authority and is deliberately
/// non-Clone. Its inventory remains private so unrelated repository entries
/// cannot become caller-selected inputs.
pub struct ResolvedModelRepository {
    reference: ResolvedHfModelReference,
    inventory: BTreeSet<String>,
}

impl fmt::Debug for ResolvedModelRepository {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResolvedModelRepository")
            .field("repository_id", &self.reference.repo_id())
            .field("revision", &self.reference.revision())
            .field("inventory_len", &self.inventory.len())
            .finish()
    }
}

impl ResolvedModelRepository {
    pub fn reference(&self) -> &ResolvedHfModelReference {
        &self.reference
    }

    pub fn inventory_len(&self) -> usize {
        self.inventory.len()
    }

    #[cfg(test)]
    pub(in crate::input) fn contains(&self, filename: &str) -> bool {
        self.inventory.contains(filename)
    }

    fn new(reference: ResolvedHfModelReference, inventory: BTreeSet<String>) -> Self {
        Self {
            reference,
            inventory,
        }
    }

    pub(super) fn into_download_parts(self) -> (ResolvedHfModelReference, BTreeSet<String>) {
        (self.reference, self.inventory)
    }
}

pub(super) fn resolve_repository_info(
    reference: HfModelReference,
    requested_revision: &str,
    returned_sha: &str,
    filenames: impl IntoIterator<Item = impl AsRef<str>>,
) -> Result<ResolvedModelRepository, DownloadError> {
    let requested_exact = requested_revision.len() == 40
        && requested_revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit());
    if requested_exact && !requested_revision.eq_ignore_ascii_case(returned_sha) {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "repository lookup returned commit `{}` instead of explicitly requested `{requested_revision}`",
                returned_sha
            ),
        });
    }
    let resolved = reference.resolve(returned_sha)?;
    let filenames = filenames
        .into_iter()
        .map(|filename| filename.as_ref().to_owned())
        .collect::<Vec<_>>();
    let inventory = validate_repo_inventory(filenames.iter().map(String::as_str))?;
    Ok(ResolvedModelRepository::new(resolved, inventory))
}

#[cfg(test)]
pub(in crate::input) fn resolve_repository_info_for_test(
    reference: HfModelReference,
    requested_revision: &str,
    returned_sha: &str,
    filenames: &[&str],
) -> Result<ResolvedModelRepository, DownloadError> {
    resolve_repository_info(
        reference,
        requested_revision,
        returned_sha,
        filenames.iter(),
    )
}
