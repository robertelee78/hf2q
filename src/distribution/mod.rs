//! Secure distribution, installation, and update primitives.
//!
//! ADR-045 deliberately starts with data-only, hostile-input-safe schemas.
//! Parsing a document in this module proves structural validity only; it does
//! not authenticate bytes, establish package ownership, or authorize a
//! filesystem mutation.

// Install/update mutation authority remains unreachable from production
// dispatch until the real trust root and public authority exist. Setup reaches
// only the read-only state-root/identity verifier below.
#[allow(dead_code)]
pub(crate) mod install_state;
#[allow(dead_code)]
mod prepared_release;
pub mod schema;
#[allow(dead_code)]
mod update_auth;
#[allow(dead_code)]
mod update_transport;

/// Descriptor-bound read-only consistency proof for a setup-selected state root.
///
/// This never bootstraps installation identity. A config-only root is valid,
/// but existing installation-owned state must satisfy the same descriptor-
/// bound identity rules as install and update transitions.
pub(crate) struct SetupStateRootBinding {
    path: std::path::PathBuf,
    identity: Option<install_state::DurableInstallationIdentity>,
}

impl SetupStateRootBinding {
    pub(crate) fn revalidate(&self) -> Result<(), install_state::InstallStateError> {
        match &self.identity {
            Some(identity) => identity.revalidate(),
            None => {
                let authorization = install_state::ExplicitRootAuthorization::new(&self.path)?;
                if install_state::open_existing_installation_identity(authorization)?.is_some() {
                    return Err(install_state::InstallStateError::InvalidLayout(
                        "installation identity appeared during setup",
                    ));
                }
                Ok(())
            }
        }
    }
}

pub(crate) fn verify_setup_state_root(
    path: &std::path::Path,
) -> Result<SetupStateRootBinding, install_state::InstallStateError> {
    let authorization = install_state::ExplicitRootAuthorization::new(path)?;
    let identity = install_state::open_existing_installation_identity(authorization)?;
    Ok(SetupStateRootBinding {
        path: path.to_owned(),
        identity,
    })
}

#[cfg(test)]
pub(crate) fn bootstrap_setup_test_identity(
    path: &std::path::Path,
) -> Result<(), install_state::InstallStateError> {
    let authorization = install_state::ExplicitRootAuthorization::new(path)?;
    let _identity = install_state::bootstrap_installation_identity_for_test(
        authorization,
        "11111111-2222-4333-8444-555555555555",
        install_state::IdentityFaultPlan::default(),
    )?;
    Ok(())
}
