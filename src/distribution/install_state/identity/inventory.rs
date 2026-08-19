use std::collections::BTreeSet;

use super::{IDENTITY_FILE, IDENTITY_INTENT_PREFIX, IDENTITY_INTENT_SUFFIX};
use crate::distribution::install_state::file;
use crate::distribution::install_state::unix::{self, Directory};
use crate::distribution::install_state::{
    ExplicitRootAuthorization, InstallStateError, PENDING_ACTIVATION, PENDING_CURRENT,
};
use crate::distribution::schema::{
    InstallationId, InstallationIdentityV1, MAX_INSTALLATION_IDENTITY_BYTES,
};

pub(super) const MAX_UPDATE_INVENTORY: usize = 8;

const BASE_UPDATE_NAMES: [&str; 3] = [".noreplace-source", ".noreplace-target", "install.lock"];
const POST_IDENTITY_UPDATE_NAMES: [&str; 4] = ["metadata", "downloads", "extractions", "prepared"];
const PRE_IDENTITY_ROOT_NAMES: [&str; 6] = [
    "versions",
    "activations",
    "current",
    PENDING_ACTIVATION,
    PENDING_CURRENT,
    "uninstall",
];

pub(super) struct IdentityInventory {
    pub(super) final_present: bool,
    pub(super) intent_id: Option<InstallationId>,
    names: BTreeSet<String>,
}

pub(super) fn classify_identity_inventory(
    _root: &Directory,
    update: &Directory,
) -> Result<IdentityInventory, InstallStateError> {
    let names = unix::list_names_bounded(update, MAX_UPDATE_INVENTORY)?;
    let final_present = names.contains(IDENTITY_FILE);
    let mut intent_id = None;
    for name in &names {
        if name == IDENTITY_FILE {
            continue;
        }
        if name.starts_with(".installation-identity") {
            let parsed = parse_intent_name(name)?;
            if intent_id.replace(parsed).is_some() {
                return Err(InstallStateError::InvalidLayout(
                    "multiple installation-identity intents coexist",
                ));
            }
        }
    }
    if final_present && intent_id.is_some() {
        return Err(InstallStateError::InvalidLayout(
            "final installation identity and intent coexist",
        ));
    }
    if (final_present || intent_id.is_some())
        && BASE_UPDATE_NAMES.iter().any(|name| !names.contains(*name))
    {
        return Err(InstallStateError::InvalidLayout(
            "installation-identity state is missing a bootstrap entry",
        ));
    }
    let mut allowed: BTreeSet<String> = BASE_UPDATE_NAMES
        .iter()
        .map(|name| (*name).to_owned())
        .collect();
    if final_present {
        allowed.insert(IDENTITY_FILE.to_owned());
        allowed.extend(
            POST_IDENTITY_UPDATE_NAMES
                .iter()
                .map(|name| (*name).to_owned()),
        );
    }
    if let Some(id) = &intent_id {
        allowed.insert(intent_name(id));
    }
    if !names.is_subset(&allowed) {
        return Err(InstallStateError::InvalidLayout(
            "update directory contains an unexpected installation-owned entry",
        ));
    }
    Ok(IdentityInventory {
        final_present,
        intent_id,
        names,
    })
}

pub(super) fn require_pre_identity_state(
    root: &Directory,
    update: &Directory,
    inventory: &IdentityInventory,
) -> Result<(), InstallStateError> {
    require_no_root_dependent_state(root)?;
    if inventory
        .names
        .iter()
        .any(|name| POST_IDENTITY_UPDATE_NAMES.contains(&name.as_str()))
    {
        return Err(InstallStateError::InvalidLayout(
            "installation-owned update state exists without an identity",
        ));
    }
    validate_intent_attributes(update, inventory.intent_id.as_ref())
}

pub(super) fn require_no_root_dependent_state(root: &Directory) -> Result<(), InstallStateError> {
    for name in PRE_IDENTITY_ROOT_NAMES {
        if unix::entry_identity(root, name)?.is_some() {
            return Err(InstallStateError::InvalidLayout(
                "installation-owned state exists without an identity",
            ));
        }
    }
    Ok(())
}

fn validate_intent_attributes(
    update: &Directory,
    intent_id: Option<&InstallationId>,
) -> Result<(), InstallStateError> {
    let Some(id) = intent_id else {
        return Ok(());
    };
    let _ = file::read_regular_file(
        update,
        &intent_name(id),
        0o600,
        MAX_INSTALLATION_IDENTITY_BYTES,
    )?;
    Ok(())
}

pub(super) fn validate_intent_prefix(
    authorization: &ExplicitRootAuthorization,
    update: &Directory,
    intent_id: Option<&InstallationId>,
) -> Result<(), InstallStateError> {
    let Some(id) = intent_id else {
        return Ok(());
    };
    let record = InstallationIdentityV1::new(id.clone(), authorization.canonical.clone());
    let expected = record.to_deterministic_json()?;
    let (_, prefix, _) = file::read_regular_file(
        update,
        &intent_name(id),
        0o600,
        MAX_INSTALLATION_IDENTITY_BYTES,
    )?;
    if !expected.starts_with(&prefix) {
        return Err(InstallStateError::InvalidLayout(
            "installation-identity intent conflicts with expected bytes",
        ));
    }
    Ok(())
}

fn parse_intent_name(name: &str) -> Result<InstallationId, InstallStateError> {
    let value = name
        .strip_prefix(IDENTITY_INTENT_PREFIX)
        .and_then(|value| value.strip_suffix(IDENTITY_INTENT_SUFFIX))
        .ok_or(InstallStateError::InvalidLayout(
            "installation-identity intent name is malformed",
        ))?;
    InstallationId::parse(value.to_owned()).map_err(InstallStateError::from)
}

pub(super) fn intent_name(installation_id: &InstallationId) -> String {
    format!(
        "{IDENTITY_INTENT_PREFIX}{}{IDENTITY_INTENT_SUFFIX}",
        installation_id.as_str()
    )
}
