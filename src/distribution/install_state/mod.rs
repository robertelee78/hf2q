//! Descriptor-relative standalone installation state.
//!
//! This bounded context intentionally implements only the first activation of
//! an already authenticated, fully prepared standalone release. It does not
//! establish update authority, executable ownership, or deletion authority.

mod file;
mod host;
mod locked;
pub(in crate::distribution) mod metadata;
mod unix;
mod verify;

#[cfg(test)]
mod test_fixture;
#[cfg(test)]
mod tests;

use std::fs::File;
use std::path::{Path, PathBuf};

use schema::{
    AbsoluteInstallPath, InstallReceiptError, InstallReceiptV1, OwnerFamily, ReleaseManifestError,
    TransitionKind, UpdateRoute,
};

use self::locked::LockedInstallation;
use self::unix::Directory;
use self::verify::VerifiedPreparedVersion;
use super::schema;

const FIRST_SEQUENCE: u64 = 1;
const FIRST_GENERATION: &str = "00000000000000000001";
const PENDING_ACTIVATION: &str = ".pending-00000000000000000001";
const PENDING_CURRENT: &str = ".current-00000000000000000001";
const CURRENT_TARGET: &str = "activations/00000000000000000001";

#[cfg(test)]
std::thread_local! {
    static FAIL_AFTER_CURRENT_COMMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FAIL_RECOVERY_BARRIER: std::cell::Cell<Option<RecoveryBarrier>> = const { std::cell::Cell::new(None) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecoveryBarrier {
    ActivationDirectory,
    ActivationsParent,
    RootDirectory,
    ReceiptFullSync,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum InstallStateError {
    #[error("another hf2q installation transition is already running")]
    Busy,
    #[error("invalid explicit installation root: {0}")]
    InvalidRoot(&'static str),
    #[error("invalid standalone installation layout: {0}")]
    InvalidLayout(&'static str),
    #[error("required standalone installation {0} is missing")]
    Missing(&'static str),
    #[error("{operation} failed: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Receipt(#[from] InstallReceiptError),
    #[error(transparent)]
    Manifest(#[from] ReleaseManifestError),
    #[error(
        "activation sequence {sequence} was committed, but its final durability is unknown: {source}"
    )]
    CommittedDurabilityUnknown {
        sequence: u64,
        #[source]
        source: Box<InstallStateError>,
    },
}

impl InstallStateError {
    fn io(operation: &'static str, source: rustix::io::Errno) -> Self {
        Self::Io {
            operation,
            source: std::io::Error::from_raw_os_error(source.raw_os_error()),
        }
    }

    fn std_io(operation: &'static str, source: std::io::Error) -> Self {
        Self::Io { operation, source }
    }

    fn after_commit(self) -> Self {
        Self::CommittedDurabilityUnknown {
            sequence: FIRST_SEQUENCE,
            source: Box::new(self),
        }
    }
}

/// Explicit user authorization for one exact standalone installation root.
#[derive(Debug)]
pub(crate) struct ExplicitRootAuthorization {
    path: PathBuf,
    canonical: AbsoluteInstallPath,
}

impl ExplicitRootAuthorization {
    pub(crate) fn new(path: &Path) -> Result<Self, InstallStateError> {
        let text = path
            .to_str()
            .ok_or(InstallStateError::InvalidRoot("root path must be UTF-8"))?;
        let canonical = AbsoluteInstallPath::parse("installation_root", text.to_owned())?;
        Ok(Self {
            path: path.to_owned(),
            canonical,
        })
    }
}

/// Exact receipt bytes whose release targets were authenticated upstream.
///
/// There is deliberately no production constructor yet. The future signed
/// update adapter may construct this capability only after authenticating the
/// exact manifest, archive, and receipt inputs and durably publishing the
/// prepared version under the authorized root. Parsing JSON is insufficient;
/// activation preparation independently re-verifies and re-syncs every byte.
#[derive(Debug)]
pub(crate) struct AuthenticatedPreparedVersion {
    receipt_bytes: Vec<u8>,
}

#[cfg(test)]
impl AuthenticatedPreparedVersion {
    fn for_test_only(receipt_bytes: Vec<u8>) -> Self {
        Self { receipt_bytes }
    }
}

#[derive(Debug)]
pub(crate) enum FirstActivationPreparation {
    Ready(PreparedFirstActivation),
    AlreadyCommitted { sequence: u64 },
}

/// A lock-held, descriptor-backed capability to commit only sequence one.
///
/// The capability is intentionally neither `Clone` nor serializable and does
/// not grant update, overwrite, entry-point, pruning, or deletion authority.
#[derive(Debug)]
pub(crate) struct PreparedFirstActivation {
    locked: LockedInstallation,
    versions: Directory,
    activations: Directory,
    receipt: InstallReceiptV1,
    receipt_bytes: Vec<u8>,
    version: VerifiedPreparedVersion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FirstActivationOutcome {
    pub(crate) sequence: u64,
}

pub(crate) fn prepare_first_activation(
    authorization: ExplicitRootAuthorization,
    authenticated: AuthenticatedPreparedVersion,
) -> Result<FirstActivationPreparation, InstallStateError> {
    let receipt =
        verify::validate_first_receipt(&authenticated.receipt_bytes, &authorization.canonical)?;
    let locked = LockedInstallation::acquire(&authorization.path)?;
    let versions = unix::open_directory_at(locked.root(), "versions", Some(0o700), true)?;
    let activations = unix::ensure_private_directory(locked.root(), "activations")?;

    if unix::entry_identity(locked.root(), "current")?.is_some() {
        let repair = || -> Result<(), InstallStateError> {
            if unix::entry_identity(locked.root(), PENDING_CURRENT)?.is_some() {
                return Err(InstallStateError::InvalidLayout(
                    "committed activation coexists with a pending current entry",
                ));
            }
            let version = verify::verify_prepared_version(&versions, &receipt)?;
            if unix::read_symlink(locked.root(), "current")? != CURRENT_TARGET {
                return Err(InstallStateError::InvalidLayout(
                    "current does not select the canonical first activation",
                ));
            }
            require_exact_activation_parent(&activations)?;
            let activation = verify::verify_committed_first_activation(
                locked.root(),
                &activations,
                &receipt,
                &authenticated.receipt_bytes,
                &version,
            )?;
            repeat_postcommit_barriers(&activation, &activations, locked.root())?;
            Ok(())
        };
        repair().map_err(InstallStateError::after_commit)?;
        return Ok(FirstActivationPreparation::AlreadyCommitted {
            sequence: FIRST_SEQUENCE,
        });
    }

    let version = verify::verify_prepared_version(&versions, &receipt)?;

    Ok(FirstActivationPreparation::Ready(PreparedFirstActivation {
        locked,
        versions,
        activations,
        receipt,
        receipt_bytes: authenticated.receipt_bytes,
        version,
    }))
}

impl PreparedFirstActivation {
    pub(crate) fn commit(self) -> Result<FirstActivationOutcome, InstallStateError> {
        let (activations, version) = self.reopen_verified_namespace()?;
        if unix::entry_identity(self.locked.root(), "current")?.is_some() {
            return Err(InstallStateError::InvalidLayout(
                "current appeared after first-activation preparation",
            ));
        }

        self.publish_or_adopt_activation(&activations, &version)?;
        self.stage_current_link()?;

        // Re-open every root-relative directory and re-verify the complete
        // activation immediately before the sole commit point. This prevents
        // a stale descriptor from authorizing a different named namespace.
        let (live_activations, live_version) = self.reopen_verified_namespace()?;
        require_exact_activation_parent(&live_activations)?;
        verify::verify_activation(
            &live_activations,
            FIRST_GENERATION,
            &self.receipt,
            &self.receipt_bytes,
            &live_version,
        )?;
        if unix::entry_identity(self.locked.root(), "current")?.is_some()
            || unix::read_symlink(self.locked.root(), PENDING_CURRENT)? != CURRENT_TARGET
        {
            return Err(InstallStateError::InvalidLayout(
                "current namespace changed before activation commit",
            ));
        }
        // This no-replace rename is the sole activation commit point.
        unix::rename_noreplace(
            self.locked.root(),
            PENDING_CURRENT,
            self.locked.root(),
            "current",
        )?;

        #[cfg(test)]
        if FAIL_AFTER_CURRENT_COMMIT.replace(false) {
            return Err(InstallStateError::std_io(
                "injected post-commit durability step",
                std::io::Error::other("test-only post-commit failure"),
            )
            .after_commit());
        }

        let finalize = || -> Result<(), InstallStateError> {
            let (fresh_activations, fresh_version) = self.reopen_verified_namespace()?;
            require_exact_activation_parent(&fresh_activations)?;
            let activation = verify::verify_committed_first_activation(
                self.locked.root(),
                &fresh_activations,
                &self.receipt,
                &self.receipt_bytes,
                &fresh_version,
            )?;
            repeat_postcommit_barriers(&activation, &fresh_activations, self.locked.root())?;
            Ok(())
        };
        finalize().map_err(InstallStateError::after_commit)?;
        Ok(FirstActivationOutcome {
            sequence: FIRST_SEQUENCE,
        })
    }

    fn publish_or_adopt_activation(
        &self,
        activations: &Directory,
        version: &VerifiedPreparedVersion,
    ) -> Result<File, InstallStateError> {
        let names = unix::list_names(activations)?;
        let allowed = std::collections::BTreeSet::from([
            FIRST_GENERATION.to_owned(),
            PENDING_ACTIVATION.to_owned(),
        ]);
        if !names.is_subset(&allowed) {
            return Err(InstallStateError::InvalidLayout(
                "first activation parent contains an unexpected generation",
            ));
        }
        let final_exists = names.contains(FIRST_GENERATION);
        let pending_exists = names.contains(PENDING_ACTIVATION);
        if final_exists && pending_exists {
            return Err(InstallStateError::InvalidLayout(
                "published and pending first activations coexist",
            ));
        }
        if final_exists {
            let activation = verify::verify_activation(
                activations,
                FIRST_GENERATION,
                &self.receipt,
                &self.receipt_bytes,
                version,
            )?;
            // A prior process may have stopped between publishing the
            // activation and syncing its parent. Re-establish both durability
            // barriers before allowing `current` to select it.
            unix::sync_directory(activations)?;
            unix::sync_directory(&activation.directory)?;
            unix::full_sync_file(&activation.receipt_file)?;
            return Ok(activation.receipt_file);
        }

        let pending = unix::ensure_private_directory(activations, PENDING_ACTIVATION)?;
        verify::resume_activation_prefix(&pending, &self.receipt, &self.receipt_bytes, version)?;
        let receipt_file = verify::verify_activation_directory(
            &pending,
            &self.receipt,
            &self.receipt_bytes,
            version,
        )?;
        unix::sync_directory(&pending)?;
        // Preflight the platform's strongest durability primitive before the
        // `current` commit point.
        unix::full_sync_file(&receipt_file)?;
        unix::rename_noreplace(
            activations,
            PENDING_ACTIVATION,
            activations,
            FIRST_GENERATION,
        )?;
        unix::sync_directory(activations)?;
        Ok(receipt_file)
    }

    fn reopen_verified_namespace(
        &self,
    ) -> Result<(Directory, VerifiedPreparedVersion), InstallStateError> {
        let live = self.locked.reopen()?;

        let versions = unix::open_directory_at(&live.root, "versions", Some(0o700), true)?;
        if !versions.same_object(&self.versions) {
            return Err(InstallStateError::InvalidLayout(
                "named versions directory changed after preparation",
            ));
        }
        let version = verify::verify_prepared_version(&versions, &self.receipt)?;
        if !version.directory.same_object(&self.version.directory) {
            return Err(InstallStateError::InvalidLayout(
                "named prepared version changed after preparation",
            ));
        }

        let activations = unix::open_directory_at(&live.root, "activations", Some(0o700), true)?;
        if !activations.same_object(&self.activations) {
            return Err(InstallStateError::InvalidLayout(
                "named activations directory changed after preparation",
            ));
        }
        Ok((activations, version))
    }

    fn stage_current_link(&self) -> Result<(), InstallStateError> {
        match unix::entry_identity(self.locked.root(), PENDING_CURRENT)? {
            None => {
                unix::create_symlink(self.locked.root(), PENDING_CURRENT, CURRENT_TARGET)?;
                unix::sync_directory(self.locked.root())?;
            }
            Some(_)
                if unix::read_symlink(self.locked.root(), PENDING_CURRENT)? == CURRENT_TARGET => {}
            Some(_) => {
                return Err(InstallStateError::InvalidLayout(
                    "pending current link has conflicting contents",
                ))
            }
        }
        Ok(())
    }
}

fn validate_receipt_shape(receipt: &InstallReceiptV1) -> Result<(), InstallStateError> {
    let transition =
        receipt
            .last_successful_transition()
            .ok_or(InstallStateError::InvalidLayout(
                "first activation lacks its transition",
            ))?;
    if receipt.owner_family() != OwnerFamily::Standalone
        || receipt.update_route() != Some(UpdateRoute::Standalone)
        || !receipt.retained().is_empty()
        || transition.sequence() != FIRST_SEQUENCE
        || transition.transition_type() != TransitionKind::Install
    {
        return Err(InstallStateError::InvalidLayout(
            "receipt is not a standalone sequence-one activation",
        ));
    }
    Ok(())
}

fn require_exact_activation_parent(activations: &Directory) -> Result<(), InstallStateError> {
    if unix::list_names(activations)?
        != std::collections::BTreeSet::from([FIRST_GENERATION.to_owned()])
    {
        return Err(InstallStateError::InvalidLayout(
            "first activation parent inventory is not exact",
        ));
    }
    Ok(())
}

fn repeat_postcommit_barriers(
    activation: &verify::VerifiedActivation,
    activations: &Directory,
    root: &Directory,
) -> Result<(), InstallStateError> {
    maybe_fail_recovery_barrier(RecoveryBarrier::ActivationDirectory)?;
    unix::sync_directory(&activation.directory)?;
    maybe_fail_recovery_barrier(RecoveryBarrier::ActivationsParent)?;
    unix::sync_directory(activations)?;
    maybe_fail_recovery_barrier(RecoveryBarrier::RootDirectory)?;
    unix::sync_directory(root)?;
    maybe_fail_recovery_barrier(RecoveryBarrier::ReceiptFullSync)?;
    unix::full_sync_file(&activation.receipt_file)
}

#[cfg(test)]
fn maybe_fail_recovery_barrier(barrier: RecoveryBarrier) -> Result<(), InstallStateError> {
    let should_fail = FAIL_RECOVERY_BARRIER.with(|selected| {
        if selected.get() == Some(barrier) {
            selected.set(None);
            true
        } else {
            false
        }
    });
    if should_fail {
        return Err(InstallStateError::std_io(
            "injected post-commit recovery barrier",
            std::io::Error::other("test-only recovery barrier failure"),
        ));
    }
    Ok(())
}

#[cfg(not(test))]
fn maybe_fail_recovery_barrier(_barrier: RecoveryBarrier) -> Result<(), InstallStateError> {
    Ok(())
}
