use serde::{Deserialize, Serialize};

use super::{
    descriptor, validate_descriptor, validate_descriptor_bytes, MetadataGenerationReceiptV2,
    MetadataRoleDescriptorV2, RoleKind,
};
use crate::distribution::install_state::metadata::{
    ExactMetadataRole, MetadataJournalError, VerifiedMetadataCandidate,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MetadataFloorResetV2 {
    from_trusted_root: MetadataRoleDescriptorV2,
    to_trusted_root: MetadataRoleDescriptorV2,
}

pub(super) fn candidate(candidate: &VerifiedMetadataCandidate) -> Option<MetadataFloorResetV2> {
    candidate
        .timestamp_snapshot_floor_reset_from_root()
        .map(|from_trusted_root| MetadataFloorResetV2 {
            from_trusted_root: descriptor(from_trusted_root),
            to_trusted_root: descriptor(candidate.trusted_root()),
        })
}

pub(super) fn validate_successor(
    receipt: &MetadataGenerationReceiptV2,
    prior: &MetadataGenerationReceiptV2,
) -> Result<(), MetadataJournalError> {
    match &receipt.timestamp_snapshot_floor_reset {
        Some(reset)
            if reset.from_trusted_root == prior.trusted_root
                && reset.to_trusted_root == receipt.trusted_root
                && receipt.root_chain.len() > prior.root_chain.len() =>
        {
            Ok(())
        }
        Some(_) => Err(MetadataJournalError::Invalid(
            "online-role floor reset is not bound to the new root transition",
        )),
        None => Ok(()),
    }
}

pub(super) fn validate_receipt(
    receipt: &MetadataGenerationReceiptV2,
) -> Result<(), MetadataJournalError> {
    let Some(reset) = &receipt.timestamp_snapshot_floor_reset else {
        return Ok(());
    };
    validate_descriptor(&reset.from_trusted_root, RoleKind::Root)?;
    validate_descriptor(&reset.to_trusted_root, RoleKind::Root)?;
    if receipt.sequence == 1
        || reset.to_trusted_root != receipt.trusted_root
        || reset.from_trusted_root.version >= reset.to_trusted_root.version
        || (reset.from_trusted_root != receipt.anchor_root
            && !receipt
                .root_chain
                .iter()
                .any(|root| root == &reset.from_trusted_root))
    {
        return Err(MetadataJournalError::Invalid(
            "online-role floor reset descriptor is invalid",
        ));
    }
    Ok(())
}

impl MetadataGenerationReceiptV2 {
    pub(in crate::distribution) fn timestamp_snapshot_floor_reset_from_root_version(
        &self,
    ) -> Option<u64> {
        self.timestamp_snapshot_floor_reset
            .as_ref()
            .map(|reset| reset.from_trusted_root.version)
    }

    pub(in crate::distribution) fn validate_timestamp_snapshot_floor_reset(
        &self,
        anchor_root: &[u8],
        root_chain: &[ExactMetadataRole],
        trusted_root: &[u8],
        binding_change_observed: bool,
    ) -> Result<(), MetadataJournalError> {
        let Some(reset) = &self.timestamp_snapshot_floor_reset else {
            return Ok(());
        };
        let from_bytes = if reset.from_trusted_root == self.anchor_root {
            anchor_root
        } else {
            root_chain
                .iter()
                .find(|root| root.version() == reset.from_trusted_root.version)
                .map(ExactMetadataRole::bytes)
                .ok_or(MetadataJournalError::Invalid(
                    "online-role floor reset root is absent",
                ))?
        };
        validate_descriptor_bytes(&reset.from_trusted_root, from_bytes)?;
        validate_descriptor_bytes(&reset.to_trusted_root, trusted_root)?;
        if !binding_change_observed {
            return Err(MetadataJournalError::Invalid(
                "online-role floor reset lacks an authenticated key rotation",
            ));
        }
        Ok(())
    }
}
