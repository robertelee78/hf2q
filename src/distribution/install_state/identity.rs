use std::fs::File;

use super::file;
use super::locked::{LiveLockedNamespace, LockedInstallation};
use super::unix::{self, Directory, EntryIdentity};
use super::{ExplicitRootAuthorization, InstallStateError};
use crate::distribution::schema::{
    AbsoluteInstallPath, InstallationId, InstallationIdentityV1, MAX_INSTALLATION_IDENTITY_BYTES,
};

mod fault;
mod inventory;
use fault::trip;
pub(in crate::distribution) use fault::{IdentityBarrier, IdentityFaultPlan};
#[cfg(test)]
use inventory::MAX_UPDATE_INVENTORY;
use inventory::{
    classify_identity_inventory, intent_name, require_no_root_dependent_state,
    require_pre_identity_state, validate_intent_prefix,
};

const IDENTITY_FILE: &str = "installation-identity.json";
const LOCK_FILE: &str = "install.lock";
const IDENTITY_INTENT_PREFIX: &str = ".installation-identity-v1-";
const IDENTITY_INTENT_SUFFIX: &str = ".partial";

/// Immutable descriptor/inode-bound proof of one root's durable identity.
///
/// This capability is intentionally neither cloneable nor serializable. A
/// copied UUID or parsed JSON value cannot substitute for it.
pub(in crate::distribution) struct DurableInstallationIdentity {
    authorization: ExplicitRootAuthorization,
    root: Directory,
    update: Directory,
    lock_file: File,
    lock_identity: EntryIdentity,
    file: File,
    file_identity: EntryIdentity,
    exact_bytes: Box<[u8]>,
    record: InstallationIdentityV1,
}

/// The shared installation lock plus a live binding to the immutable identity.
pub(in crate::distribution) struct LockedInstallationIdentity {
    locked: LockedInstallation,
    file: File,
    file_identity: EntryIdentity,
    exact_bytes: Box<[u8]>,
    record: InstallationIdentityV1,
}

pub(super) struct LiveLockedInstallationIdentity {
    pub(super) root: Directory,
    pub(super) update: Directory,
    pub(super) identity_file: File,
}

pub(super) struct LiveInstallationIdentity {
    pub(super) root: Directory,
    pub(super) update: Directory,
    pub(super) _identity_file: File,
}

pub(in crate::distribution) enum InstallationIdentityBootstrap {
    Created(DurableInstallationIdentity),
    AlreadyCreated(DurableInstallationIdentity),
}

impl std::fmt::Debug for DurableInstallationIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DurableInstallationIdentity")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for LockedInstallationIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LockedInstallationIdentity")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for InstallationIdentityBootstrap {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Created(_) => formatter.write_str("Created(DurableInstallationIdentity { .. })"),
            Self::AlreadyCreated(_) => {
                formatter.write_str("AlreadyCreated(DurableInstallationIdentity { .. })")
            }
        }
    }
}

impl InstallationIdentityBootstrap {
    pub(in crate::distribution) fn into_identity(self) -> DurableInstallationIdentity {
        match self {
            Self::Created(identity) | Self::AlreadyCreated(identity) => identity,
        }
    }
}

pub(in crate::distribution) fn bootstrap_installation_identity(
    authorization: ExplicitRootAuthorization,
) -> Result<InstallationIdentityBootstrap, InstallStateError> {
    bootstrap_with_id(authorization, None, IdentityFaultPlan::default())
}

#[cfg(test)]
pub(in crate::distribution) fn bootstrap_installation_identity_for_test(
    authorization: ExplicitRootAuthorization,
    installation_id: &str,
    faults: IdentityFaultPlan,
) -> Result<InstallationIdentityBootstrap, InstallStateError> {
    let installation_id = InstallationId::parse(installation_id.to_owned())?;
    bootstrap_with_id(authorization, Some(installation_id), faults)
}

fn bootstrap_with_id(
    authorization: ExplicitRootAuthorization,
    requested_id: Option<InstallationId>,
    faults: IdentityFaultPlan,
) -> Result<InstallationIdentityBootstrap, InstallStateError> {
    preflight_bootstrap_without_mutation(&authorization)?;
    let locked = LockedInstallation::acquire(&authorization.path)?;
    let inventory = classify_identity_inventory(locked.root(), locked.update())?;
    if inventory.final_present {
        let identity = repair_committed_identity(&locked, authorization, faults)?;
        if requested_id
            .as_ref()
            .is_some_and(|expected| expected != identity.installation_id())
        {
            return Err(InstallStateError::InvalidLayout(
                "installation identity differs from the requested test identity",
            ));
        }
        return Ok(InstallationIdentityBootstrap::AlreadyCreated(identity));
    }

    require_pre_identity_state(locked.root(), locked.update(), &inventory)?;
    let installation_id = match (inventory.intent_id, requested_id) {
        (Some(intent), Some(requested)) if intent != requested => {
            return Err(InstallStateError::InvalidLayout(
                "durable installation-identity intent differs from the requested identity",
            ))
        }
        (Some(intent), _) => intent,
        (None, Some(requested)) => requested,
        (None, None) => InstallationId::parse(uuid::Uuid::new_v4().hyphenated().to_string())?,
    };
    let intent_name = intent_name(&installation_id);
    let record =
        InstallationIdentityV1::new(installation_id.clone(), authorization.canonical.clone());
    let expected = record.to_deterministic_json()?;
    let intent_file = file::write_or_resume_private_file_with_create_hook(
        locked.update(),
        &intent_name,
        &expected,
        || trip(faults, IdentityBarrier::IntentNameSync),
    )?;
    trip(faults, IdentityBarrier::IntentContentSync)?;
    unix::full_sync_file(&intent_file)?;
    trip(faults, IdentityBarrier::IntentFullSync)?;

    let live = locked.reopen()?;
    require_precommit_intent(&live, &intent_name, &intent_file, &expected)?;
    trip(faults, IdentityBarrier::PrecommitReopen)?;
    unix::rename_noreplace(&live.update, &intent_name, &live.update, IDENTITY_FILE)?;
    let result = finish_committed_identity(
        &locked,
        authorization,
        &installation_id,
        Some(unix::regular_file_identity(
            &intent_file,
            live.update.device(),
        )?),
        faults,
    );
    match result {
        Ok(identity) => Ok(InstallationIdentityBootstrap::Created(identity)),
        Err(error) => Err(error.after_identity_commit(&installation_id)),
    }
}

fn preflight_bootstrap_without_mutation(
    authorization: &ExplicitRootAuthorization,
) -> Result<(), InstallStateError> {
    let root = match unix::open_existing_root(&authorization.path) {
        Ok(root) => root,
        Err(InstallStateError::Missing(_)) => return Ok(()),
        Err(error) => return Err(error),
    };
    if unix::entry_identity(&root, "update")?.is_none() {
        return require_no_root_dependent_state(&root);
    }
    let update = unix::open_directory_at(&root, "update", Some(0o700), true)?;
    let inventory = classify_identity_inventory(&root, &update)?;
    if inventory.final_present {
        Ok(())
    } else {
        require_pre_identity_state(&root, &update, &inventory)?;
        validate_intent_prefix(authorization, &update, inventory.intent_id.as_ref())
    }
}

fn repair_committed_identity(
    locked: &LockedInstallation,
    authorization: ExplicitRootAuthorization,
    faults: IdentityFaultPlan,
) -> Result<DurableInstallationIdentity, InstallStateError> {
    let live = locked.reopen()?;
    let (_, bytes, _) = file::read_regular_file(
        &live.update,
        IDENTITY_FILE,
        0o600,
        MAX_INSTALLATION_IDENTITY_BYTES,
    )?;
    let record = InstallationIdentityV1::parse_and_validate(&bytes)?;
    require_record_binding(&record, &authorization, None)?;
    let installation_id = record.installation_id().clone();
    finish_committed_identity(locked, authorization, &installation_id, None, faults)
        .map_err(|error| error.after_identity_commit(&installation_id))
}

fn finish_committed_identity(
    locked: &LockedInstallation,
    authorization: ExplicitRootAuthorization,
    expected_id: &InstallationId,
    expected_inode: Option<EntryIdentity>,
    faults: IdentityFaultPlan,
) -> Result<DurableInstallationIdentity, InstallStateError> {
    trip(faults, IdentityBarrier::IdentityRename)?;
    let live = locked.reopen()?;
    let inventory = classify_identity_inventory(&live.root, &live.update)?;
    if !inventory.final_present || inventory.intent_id.is_some() {
        return Err(InstallStateError::InvalidLayout(
            "committed installation identity has transaction residue",
        ));
    }
    let (file, bytes, file_identity) = file::read_regular_file(
        &live.update,
        IDENTITY_FILE,
        0o600,
        MAX_INSTALLATION_IDENTITY_BYTES,
    )?;
    let record = InstallationIdentityV1::parse_and_validate(&bytes)?;
    require_record_binding(&record, &authorization, Some(expected_id))?;
    if expected_inode.is_some_and(|identity| identity != file_identity) {
        return Err(InstallStateError::InvalidLayout(
            "published installation identity is not the prepared inode",
        ));
    }
    trip(faults, IdentityBarrier::FinalReopen)?;
    unix::full_sync_file(&file)?;
    trip(faults, IdentityBarrier::FinalFullSync)?;
    unix::sync_directory(&live.update)?;
    trip(faults, IdentityBarrier::UpdateSync)?;
    unix::sync_directory(&live.root)?;
    trip(faults, IdentityBarrier::RootSync)?;
    locked.full_sync_endpoint()?;
    trip(faults, IdentityBarrier::LockFullSync)?;
    open_bound_under_lock(locked, authorization, Some(expected_id))
}

pub(in crate::distribution) fn open_existing_installation_identity(
    authorization: ExplicitRootAuthorization,
) -> Result<Option<DurableInstallationIdentity>, InstallStateError> {
    open_existing_installation_identity_with_hook(authorization, || {})
}

fn open_existing_installation_identity_with_hook(
    authorization: ExplicitRootAuthorization,
    hook: impl FnOnce(),
) -> Result<Option<DurableInstallationIdentity>, InstallStateError> {
    let root = match unix::open_existing_root(&authorization.path) {
        Ok(root) => root,
        Err(InstallStateError::Missing(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    let update_identity = unix::entry_identity(&root, "update")?;
    let Some(_) = update_identity else {
        require_no_root_dependent_state(&root)?;
        return Ok(None);
    };
    let update = unix::open_directory_at(&root, "update", Some(0o700), true)?;
    let inventory = classify_identity_inventory(&root, &update)?;
    if !inventory.final_present {
        require_pre_identity_state(&root, &update, &inventory)?;
        validate_intent_prefix(&authorization, &update, inventory.intent_id.as_ref())?;
        return Ok(None);
    }
    let first = open_bound(authorization, root, update, None)?;
    hook();
    let fresh_root = unix::open_existing_root(&first.authorization.path)?;
    if !fresh_root.same_object(&first.root) {
        return Err(InstallStateError::InvalidLayout(
            "named state root changed while reading installation identity",
        ));
    }
    let fresh_update = unix::open_directory_at(&fresh_root, "update", Some(0o700), true)?;
    if !fresh_update.same_object(&first.update) {
        return Err(InstallStateError::InvalidLayout(
            "named update directory changed while reading installation identity",
        ));
    }
    let second = open_bound(
        ExplicitRootAuthorization::new(&first.authorization.path)?,
        fresh_root,
        fresh_update,
        Some(first.installation_id()),
    )?;
    if second.file_identity != first.file_identity
        || second.exact_bytes != first.exact_bytes
        || second.lock_identity != first.lock_identity
    {
        return Err(InstallStateError::InvalidLayout(
            "installation identity changed between read snapshots",
        ));
    }
    Ok(Some(second))
}

impl DurableInstallationIdentity {
    pub(in crate::distribution) fn installation_id(&self) -> &InstallationId {
        self.record.installation_id()
    }

    pub(in crate::distribution) fn state_root(&self) -> &AbsoluteInstallPath {
        self.record.state_root()
    }

    pub(super) fn authorization(&self) -> &ExplicitRootAuthorization {
        &self.authorization
    }

    pub(super) fn reopen(&self) -> Result<LiveInstallationIdentity, InstallStateError> {
        let root = unix::open_existing_root(&self.authorization.path)?;
        if !root.same_object(&self.root) {
            return Err(InstallStateError::InvalidLayout(
                "named state root changed after installation identity authorization",
            ));
        }
        let update = unix::open_directory_at(&root, "update", Some(0o700), true)?;
        if !update.same_object(&self.update) {
            return Err(InstallStateError::InvalidLayout(
                "named update directory changed after installation identity authorization",
            ));
        }
        unix::verify_named_identity(&update, LOCK_FILE, self.lock_identity)?;
        if unix::regular_file_identity(&self.lock_file, update.device())? != self.lock_identity {
            return Err(InstallStateError::InvalidLayout(
                "retained installation lock changed after authorization",
            ));
        }
        let (identity_file, bytes, file_identity) = file::read_regular_file(
            &update,
            IDENTITY_FILE,
            0o600,
            MAX_INSTALLATION_IDENTITY_BYTES,
        )?;
        if file_identity != self.file_identity || bytes.as_slice() != self.exact_bytes.as_ref() {
            return Err(InstallStateError::InvalidLayout(
                "installation identity changed after authorization",
            ));
        }
        if unix::regular_file_identity(&self.file, update.device())? != self.file_identity {
            return Err(InstallStateError::InvalidLayout(
                "retained installation identity changed after authorization",
            ));
        }
        Ok(LiveInstallationIdentity {
            root,
            update,
            _identity_file: identity_file,
        })
    }

    pub(super) fn lock(&self) -> Result<LockedInstallationIdentity, InstallStateError> {
        let locked = LockedInstallation::acquire(&self.authorization.path)?;
        let live = locked.reopen()?;
        if !live.root.same_object(&self.root) || !live.update.same_object(&self.update) {
            return Err(InstallStateError::InvalidLayout(
                "installation namespace changed before lock acquisition",
            ));
        }
        if locked.lock_identity() != self.lock_identity
            || unix::regular_file_identity(&self.lock_file, live.update.device())?
                != self.lock_identity
        {
            return Err(InstallStateError::InvalidLayout(
                "installation lock changed before lock acquisition",
            ));
        }
        let (file, bytes, file_identity) = file::read_regular_file(
            &live.update,
            IDENTITY_FILE,
            0o600,
            MAX_INSTALLATION_IDENTITY_BYTES,
        )?;
        if file_identity != self.file_identity || bytes.as_slice() != self.exact_bytes.as_ref() {
            return Err(InstallStateError::InvalidLayout(
                "installation identity changed before lock acquisition",
            ));
        }
        let retained = unix::regular_file_identity(&self.file, live.update.device())?;
        if retained != self.file_identity {
            return Err(InstallStateError::InvalidLayout(
                "retained installation identity changed before lock acquisition",
            ));
        }
        unix::full_sync_file(&file)?;
        unix::sync_directory(&live.update)?;
        unix::sync_directory(&live.root)?;
        locked.full_sync_endpoint()?;
        let live = locked.reopen()?;
        let (file, bytes, file_identity) = file::read_regular_file(
            &live.update,
            IDENTITY_FILE,
            0o600,
            MAX_INSTALLATION_IDENTITY_BYTES,
        )?;
        if file_identity != self.file_identity || bytes.as_slice() != self.exact_bytes.as_ref() {
            return Err(InstallStateError::InvalidLayout(
                "installation identity changed while repairing durability",
            ));
        }
        let record = InstallationIdentityV1::parse_and_validate(&bytes)?;
        Ok(LockedInstallationIdentity {
            locked,
            file,
            file_identity,
            exact_bytes: bytes.into_boxed_slice(),
            record,
        })
    }
}

impl LockedInstallationIdentity {
    pub(super) fn root(&self) -> &Directory {
        self.locked.root()
    }

    pub(super) fn update(&self) -> &Directory {
        self.locked.update()
    }

    pub(super) fn installation_id(&self) -> &InstallationId {
        self.record.installation_id()
    }

    pub(super) fn state_root(&self) -> &AbsoluteInstallPath {
        self.record.state_root()
    }

    pub(super) fn reopen(&self) -> Result<LiveLockedInstallationIdentity, InstallStateError> {
        let live = self.locked.reopen()?;
        let (identity_file, bytes, file_identity) = file::read_regular_file(
            &live.update,
            IDENTITY_FILE,
            0o600,
            MAX_INSTALLATION_IDENTITY_BYTES,
        )?;
        if file_identity != self.file_identity || bytes.as_slice() != self.exact_bytes.as_ref() {
            return Err(InstallStateError::InvalidLayout(
                "installation identity changed while its lock was held",
            ));
        }
        if unix::regular_file_identity(&self.file, live.update.device())? != self.file_identity {
            return Err(InstallStateError::InvalidLayout(
                "retained installation identity changed while its lock was held",
            ));
        }
        Ok(LiveLockedInstallationIdentity {
            root: live.root,
            update: live.update,
            identity_file,
        })
    }

    pub(super) fn full_sync_endpoint(&self) -> Result<(), InstallStateError> {
        let live = self.reopen()?;
        unix::full_sync_file(&live.identity_file)?;
        unix::sync_directory(&live.update)?;
        unix::sync_directory(&live.root)?;
        self.locked.full_sync_endpoint()
    }
}

fn open_bound_under_lock(
    locked: &LockedInstallation,
    authorization: ExplicitRootAuthorization,
    expected_id: Option<&InstallationId>,
) -> Result<DurableInstallationIdentity, InstallStateError> {
    let live = locked.reopen()?;
    open_bound(authorization, live.root, live.update, expected_id)
}

fn open_bound(
    authorization: ExplicitRootAuthorization,
    root: Directory,
    update: Directory,
    expected_id: Option<&InstallationId>,
) -> Result<DurableInstallationIdentity, InstallStateError> {
    let inventory = classify_identity_inventory(&root, &update)?;
    if !inventory.final_present || inventory.intent_id.is_some() {
        return Err(InstallStateError::InvalidLayout(
            "durable installation identity inventory is not exact",
        ));
    }
    let (file, bytes, file_identity) = file::read_regular_file(
        &update,
        IDENTITY_FILE,
        0o600,
        MAX_INSTALLATION_IDENTITY_BYTES,
    )?;
    let record = InstallationIdentityV1::parse_and_validate(&bytes)?;
    require_record_binding(&record, &authorization, expected_id)?;
    let (lock_file, lock_identity) = unix::open_private_regular_file(&update, LOCK_FILE)?;
    Ok(DurableInstallationIdentity {
        authorization,
        root,
        update,
        lock_file,
        lock_identity,
        file,
        file_identity,
        exact_bytes: bytes.into_boxed_slice(),
        record,
    })
}

fn require_precommit_intent(
    live: &LiveLockedNamespace,
    expected_intent_name: &str,
    intent_file: &File,
    expected: &[u8],
) -> Result<(), InstallStateError> {
    let inventory = classify_identity_inventory(&live.root, &live.update)?;
    require_pre_identity_state(&live.root, &live.update, &inventory)?;
    if inventory.intent_id.as_ref().map(intent_name).as_deref() != Some(expected_intent_name) {
        return Err(InstallStateError::InvalidLayout(
            "installation-identity intent changed before publication",
        ));
    }
    if unix::entry_identity(&live.update, IDENTITY_FILE)?.is_some() {
        return Err(InstallStateError::InvalidLayout(
            "installation identity appeared before publication",
        ));
    }
    let (named, bytes, identity) = file::read_regular_file(
        &live.update,
        expected_intent_name,
        0o600,
        MAX_INSTALLATION_IDENTITY_BYTES,
    )?;
    if bytes != expected
        || unix::regular_file_identity(intent_file, live.update.device())? != identity
        || unix::regular_file_identity(&named, live.update.device())? != identity
    {
        return Err(InstallStateError::InvalidLayout(
            "installation-identity intent changed before publication",
        ));
    }
    Ok(())
}

fn require_record_binding(
    record: &InstallationIdentityV1,
    authorization: &ExplicitRootAuthorization,
    expected_id: Option<&InstallationId>,
) -> Result<(), InstallStateError> {
    if record.state_root() != &authorization.canonical
        || expected_id.is_some_and(|expected| expected != record.installation_id())
    {
        return Err(InstallStateError::InvalidLayout(
            "installation identity is bound to a different root or UUID",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "identity_tests.rs"]
mod tests;
