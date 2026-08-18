use std::collections::BTreeSet;

use serde::Deserialize;

use super::*;

const MAX_RETAINED_RELEASES: usize = 2;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInstallReceiptV1 {
    kind: String,
    schema_version: u32,
    package: String,
    state_layout_schema: u32,
    installation_layout_schema: Option<u32>,
    installation_id: String,
    state_root: String,
    installation_root: String,
    owner_family: String,
    update_route: Option<String>,
    active: RawInstalledReleaseV1,
    retained: Vec<RawInstalledReleaseV1>,
    last_successful_transition: Option<RawSuccessfulTransitionV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInstalledReleaseV1 {
    version: String,
    target: String,
    bundle: Option<RawRecordedBundleIdentityV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRecordedBundleIdentityV1 {
    release_manifest_sha256: String,
    archive_sha256: String,
    installed_version_marker_sha256: Option<String>,
    installation_sequence: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSuccessfulTransitionV1 {
    sequence: u64,
    #[serde(rename = "type")]
    transition_type: String,
    from: Option<RawTransitionEndpointV1>,
    to: RawTransitionEndpointV1,
    authority: RawRecordedTransitionEvidenceV1,
    completed_at_unix_seconds: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTransitionEndpointV1 {
    owner_family: String,
    release: RawInstalledReleaseV1,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum RawRecordedTransitionEvidenceV1 {
    #[serde(rename = "verified-update-metadata")]
    VerifiedUpdateMetadata {
        root_version: u64,
        timestamp_version: u64,
        snapshot_version: u64,
        targets_version: u64,
    },
    #[serde(rename = "package-manager")]
    PackageManager { route: String },
    #[serde(rename = "retained-release")]
    RetainedRelease { release_manifest_sha256: String },
}

pub(super) fn parse_install_receipt(bytes: &[u8]) -> Result<InstallReceiptV1, InstallReceiptError> {
    if bytes.len() > MAX_INSTALL_RECEIPT_BYTES {
        return Err(InstallReceiptError::InputTooLarge {
            document: "install receipt",
            limit: MAX_INSTALL_RECEIPT_BYTES,
            actual: bytes.len(),
        });
    }
    let raw: RawInstallReceiptV1 = serde_json::from_slice(bytes)
        .map_err(|error| sanitize_json_error("install receipt", error))?;
    validate_install_receipt(raw)
}

fn validate_install_receipt(
    raw: RawInstallReceiptV1,
) -> Result<InstallReceiptV1, InstallReceiptError> {
    validate_envelope(
        &raw.kind,
        raw.schema_version,
        &raw.package,
        INSTALL_RECEIPT_KIND,
        INSTALL_RECEIPT_SCHEMA_VERSION,
        "install receipt",
    )?;
    if raw.state_layout_schema != STATE_LAYOUT_SCHEMA_V1 {
        return Err(InstallReceiptError::invalid(
            "state_layout_schema",
            "must equal the supported v1 state layout schema",
        ));
    }
    if raw.retained.len() > MAX_RETAINED_RELEASES {
        return Err(InstallReceiptError::TooManyRetained {
            limit: MAX_RETAINED_RELEASES,
            actual: raw.retained.len(),
        });
    }

    let installation_id = InstallationId::parse(raw.installation_id)?;
    let state_root = AbsoluteInstallPath::parse("state_root", raw.state_root)?;
    let installation_root = AbsoluteInstallPath::parse("installation_root", raw.installation_root)?;
    if is_at_or_below_versions(&state_root, &installation_root) {
        return Err(InstallReceiptError::invalid(
            "state_root",
            "cannot be the installation versions directory or one of its descendants",
        ));
    }

    let owner_family = parse_owner("owner_family", &raw.owner_family)?;
    let update_route = raw
        .update_route
        .as_deref()
        .map(|route| parse_route("update_route", route))
        .transpose()?;
    validate_owner_layout_and_route(
        owner_family,
        update_route,
        raw.installation_layout_schema,
        &state_root,
        &installation_root,
    )?;

    let active = parse_release("active", raw.active)?;
    let retained = raw
        .retained
        .into_iter()
        .map(|release| parse_release("retained[]", release))
        .collect::<Result<Vec<_>, _>>()?;
    validate_release_set(owner_family, &active, &retained)?;

    let last_successful_transition = raw
        .last_successful_transition
        .map(parse_transition)
        .transpose()?;
    validate_transition(
        owner_family,
        update_route,
        &active,
        &retained,
        last_successful_transition.as_ref(),
    )?;

    Ok(InstallReceiptV1 {
        kind: raw.kind,
        schema_version: raw.schema_version,
        package: raw.package,
        state_layout_schema: raw.state_layout_schema,
        installation_layout_schema: raw.installation_layout_schema,
        installation_id,
        state_root,
        installation_root,
        owner_family,
        update_route,
        active,
        retained,
        last_successful_transition,
    })
}

fn validate_owner_layout_and_route(
    owner: OwnerFamily,
    route: Option<UpdateRoute>,
    layout: Option<u32>,
    state_root: &AbsoluteInstallPath,
    installation_root: &AbsoluteInstallPath,
) -> Result<(), InstallReceiptError> {
    let route_matches = matches!(
        (owner, route),
        (OwnerFamily::Standalone, Some(UpdateRoute::Standalone))
            | (OwnerFamily::Homebrew, Some(UpdateRoute::Brew))
            | (
                OwnerFamily::CargoRegistry,
                Some(UpdateRoute::CargoInstall | UpdateRoute::CargoBinstall)
            )
            | (OwnerFamily::UnknownManual, None)
    );
    if !route_matches {
        return Err(InstallReceiptError::OwnerRouteMismatch);
    }

    match owner {
        OwnerFamily::Standalone => {
            if layout != Some(INSTALLATION_LAYOUT_SCHEMA_V1) {
                return Err(InstallReceiptError::invalid(
                    "installation_layout_schema",
                    "standalone ownership requires the supported v1 layout schema",
                ));
            }
            if state_root != installation_root {
                return Err(InstallReceiptError::invalid(
                    "state_root",
                    "standalone state and installation roots must be identical",
                ));
            }
        }
        _ if layout.is_some() => {
            return Err(InstallReceiptError::invalid(
                "installation_layout_schema",
                "manager/manual ownership cannot claim an hf2q-owned installation layout",
            ));
        }
        _ => {}
    }
    Ok(())
}

fn validate_release_set(
    owner: OwnerFamily,
    active: &InstalledReleaseV1,
    retained: &[InstalledReleaseV1],
) -> Result<(), InstallReceiptError> {
    match owner {
        OwnerFamily::Standalone => {
            if !release_has_standalone_identity(active)
                || retained
                    .iter()
                    .any(|release| !release_has_standalone_identity(release))
            {
                return Err(InstallReceiptError::invalid(
                    "active.bundle",
                    "standalone active and retained releases require bundle and installed-marker identity",
                ));
            }
        }
        _ => {
            if !retained.is_empty() {
                return Err(InstallReceiptError::invalid(
                    "retained",
                    "manager/manual ownership cannot claim standalone retained releases",
                ));
            }
            if active
                .bundle
                .as_ref()
                .is_some_and(RecordedBundleIdentityV1::is_standalone_installation)
            {
                return Err(InstallReceiptError::invalid(
                    "active.bundle",
                    "manager/manual ownership cannot claim a standalone installed-version marker or sequence",
                ));
            }
        }
    }

    let mut versions = BTreeSet::new();
    let mut installation_sequences = BTreeSet::new();
    for release in std::iter::once(active).chain(retained) {
        if release.target != active.target {
            return Err(InstallReceiptError::invalid(
                "retained[].target",
                "all retained releases must match the active target",
            ));
        }
        if !versions.insert(release.version.as_str().to_owned()) {
            return Err(InstallReceiptError::DuplicateVersion(
                release.version.as_str().to_owned(),
            ));
        }
        if let Some(bundle) = &release.bundle {
            if bundle
                .installation_sequence
                .is_some_and(|sequence| !installation_sequences.insert(sequence))
            {
                return Err(InstallReceiptError::invalid(
                    "bundle.installation_sequence",
                    "installation sequences must be unique across active and retained releases",
                ));
            }
        }
    }
    Ok(())
}

fn parse_release(
    field: &'static str,
    raw: RawInstalledReleaseV1,
) -> Result<InstalledReleaseV1, InstallReceiptError> {
    let bundle = raw
        .bundle
        .map(|bundle| {
            if bundle.installation_sequence == Some(0) {
                return Err(InstallReceiptError::invalid(
                    "bundle.installation_sequence",
                    "must be nonzero",
                ));
            }
            if bundle.installed_version_marker_sha256.is_some()
                != bundle.installation_sequence.is_some()
            {
                return Err(InstallReceiptError::invalid(
                    "bundle",
                    "installed marker digest and installation sequence must either both be present or both be absent",
                ));
            }
            Ok(RecordedBundleIdentityV1 {
                release_manifest_sha256: Sha256Digest::parse(
                    "bundle.release_manifest_sha256",
                    bundle.release_manifest_sha256,
                )?,
                archive_sha256: Sha256Digest::parse(
                    "bundle.archive_sha256",
                    bundle.archive_sha256,
                )?,
                installed_version_marker_sha256: bundle
                    .installed_version_marker_sha256
                    .map(|digest| {
                        Sha256Digest::parse("bundle.installed_version_marker_sha256", digest)
                    })
                    .transpose()?,
                installation_sequence: bundle.installation_sequence,
            })
        })
        .transpose()?;
    Ok(InstalledReleaseV1 {
        version: ReleaseVersion::parse_stable(field, raw.version)?,
        target: TargetTriple::parse(field, raw.target)?,
        bundle,
    })
}

fn release_has_standalone_identity(release: &InstalledReleaseV1) -> bool {
    release
        .bundle
        .as_ref()
        .is_some_and(RecordedBundleIdentityV1::is_standalone_installation)
}

fn parse_transition(
    raw: RawSuccessfulTransitionV1,
) -> Result<SuccessfulTransitionV1, InstallReceiptError> {
    if raw.sequence == 0 || raw.completed_at_unix_seconds == 0 {
        return Err(InstallReceiptError::invalid(
            "last_successful_transition",
            "sequence and completion time must be nonzero",
        ));
    }
    Ok(SuccessfulTransitionV1 {
        sequence: raw.sequence,
        transition_type: parse_transition_kind(&raw.transition_type)?,
        from: raw.from.map(parse_endpoint).transpose()?,
        to: parse_endpoint(raw.to)?,
        recorded_evidence: parse_recorded_evidence(raw.authority)?,
        completed_at_unix_seconds: raw.completed_at_unix_seconds,
    })
}

fn parse_endpoint(
    raw: RawTransitionEndpointV1,
) -> Result<TransitionEndpointV1, InstallReceiptError> {
    Ok(TransitionEndpointV1 {
        owner_family: parse_owner("last_successful_transition.owner_family", &raw.owner_family)?,
        release: parse_release("last_successful_transition.release", raw.release)?,
    })
}

fn parse_recorded_evidence(
    raw: RawRecordedTransitionEvidenceV1,
) -> Result<RecordedTransitionEvidenceV1, InstallReceiptError> {
    match raw {
        RawRecordedTransitionEvidenceV1::VerifiedUpdateMetadata {
            root_version,
            timestamp_version,
            snapshot_version,
            targets_version,
        } => {
            if [
                root_version,
                timestamp_version,
                snapshot_version,
                targets_version,
            ]
            .contains(&0)
            {
                return Err(InstallReceiptError::invalid(
                    "last_successful_transition.authority",
                    "verified metadata role versions must be nonzero",
                ));
            }
            Ok(RecordedTransitionEvidenceV1::UpdateMetadataVersions {
                root_version,
                timestamp_version,
                snapshot_version,
                targets_version,
            })
        }
        RawRecordedTransitionEvidenceV1::PackageManager { route } => {
            Ok(RecordedTransitionEvidenceV1::PackageManagerRoute {
                route: parse_route("last_successful_transition.authority.route", &route)?,
            })
        }
        RawRecordedTransitionEvidenceV1::RetainedRelease {
            release_manifest_sha256,
        } => Ok(RecordedTransitionEvidenceV1::RetainedReleaseManifest {
            release_manifest_sha256: Sha256Digest::parse(
                "last_successful_transition.authority.release_manifest_sha256",
                release_manifest_sha256,
            )?,
        }),
    }
}

fn validate_transition(
    owner: OwnerFamily,
    route: Option<UpdateRoute>,
    active: &InstalledReleaseV1,
    retained: &[InstalledReleaseV1],
    transition: Option<&SuccessfulTransitionV1>,
) -> Result<(), InstallReceiptError> {
    if owner == OwnerFamily::UnknownManual {
        return if transition.is_none() {
            Ok(())
        } else {
            Err(InstallReceiptError::TransitionMismatch(
                "unknown/manual ownership cannot claim a successful transition",
            ))
        };
    }
    if owner == OwnerFamily::Standalone && transition.is_none() {
        return Err(InstallReceiptError::TransitionMismatch(
            "standalone ownership requires a last successful transition",
        ));
    }
    let Some(transition) = transition else {
        return Ok(());
    };
    if transition.to.owner_family != owner || transition.to.release != *active {
        return Err(InstallReceiptError::TransitionMismatch(
            "transition destination must equal the current owner and active release",
        ));
    }
    if owner == OwnerFamily::Standalone {
        let active_installation = active
            .bundle
            .as_ref()
            .and_then(RecordedBundleIdentityV1::installation_sequence)
            .expect("standalone active bundle validated");
        let newest_retained_installation = retained
            .iter()
            .filter_map(|release| release.bundle.as_ref())
            .filter_map(RecordedBundleIdentityV1::installation_sequence)
            .max()
            .unwrap_or(0);
        if transition.transition_type == TransitionKind::Rollback {
            require(
                transition.sequence > active_installation.max(newest_retained_installation),
                "rollback sequence must follow every installed release",
            )?;
        } else {
            require(
                transition.sequence == active_installation,
                "transition sequence must equal the newly installed release sequence",
            )?;
            require(
                active_installation > newest_retained_installation,
                "newly installed release sequence must follow every retained release",
            )?;
        }
    }

    match transition.transition_type {
        TransitionKind::Install => {
            require(transition.from.is_none(), "install origin must be absent")?;
            require(
                owner == OwnerFamily::Standalone,
                "install must end standalone",
            )?;
            require(
                retained.is_empty(),
                "initial install cannot retain releases",
            )?;
            require_recorded_update_metadata(&transition.recorded_evidence)?;
        }
        TransitionKind::Update => {
            let from = transition
                .from
                .as_ref()
                .ok_or(InstallReceiptError::TransitionMismatch(
                    "update origin must be present",
                ))?;
            require(from.owner_family == owner, "update owner cannot change")?;
            require(
                from.release.version != active.version,
                "update must change the release version",
            )?;
            match owner {
                OwnerFamily::Standalone => {
                    require(
                        release_has_standalone_identity(&from.release),
                        "standalone update origin needs installed-marker identity",
                    )?;
                    require(
                        retained.first() == Some(&from.release),
                        "standalone update must retain the prior active release first",
                    )?;
                    require_recorded_update_metadata(&transition.recorded_evidence)?;
                }
                OwnerFamily::Homebrew | OwnerFamily::CargoRegistry => {
                    require(
                        !release_has_standalone_identity(&from.release),
                        "manager update origin cannot claim installed-marker identity",
                    )?;
                    require_recorded_manager_route(&transition.recorded_evidence, route)?;
                }
                OwnerFamily::UnknownManual => unreachable!(),
            }
        }
        TransitionKind::Rollback => {
            require(
                owner == OwnerFamily::Standalone,
                "rollback must end standalone",
            )?;
            let from = transition
                .from
                .as_ref()
                .ok_or(InstallReceiptError::TransitionMismatch(
                    "rollback origin must be present",
                ))?;
            require(
                from.owner_family == OwnerFamily::Standalone,
                "rollback origin must be standalone",
            )?;
            require(
                retained.first() == Some(&from.release),
                "rollback must retain the prior active release first",
            )?;
            let active_manifest = active
                .bundle
                .as_ref()
                .expect("standalone bundle validated")
                .release_manifest_sha256();
            match &transition.recorded_evidence {
                RecordedTransitionEvidenceV1::RetainedReleaseManifest {
                    release_manifest_sha256,
                } if release_manifest_sha256 == active_manifest => {}
                _ => {
                    return Err(InstallReceiptError::TransitionMismatch(
                        "rollback authority must bind the newly active retained manifest",
                    ));
                }
            }
        }
        TransitionKind::ConfirmedMigration => {
            require(
                owner == OwnerFamily::Standalone,
                "migration must end standalone",
            )?;
            let from = transition
                .from
                .as_ref()
                .ok_or(InstallReceiptError::TransitionMismatch(
                    "migration origin must be present",
                ))?;
            require(
                from.owner_family == OwnerFamily::UnknownManual,
                "migration must originate from unknown/manual ownership",
            )?;
            require(
                !release_has_standalone_identity(&from.release),
                "migration origin cannot claim installed-marker identity",
            )?;
            require(
                retained.is_empty(),
                "confirmed migration cannot invent retained releases",
            )?;
            require_recorded_update_metadata(&transition.recorded_evidence)?;
        }
    }
    Ok(())
}

fn require_recorded_update_metadata(
    evidence: &RecordedTransitionEvidenceV1,
) -> Result<(), InstallReceiptError> {
    require(
        matches!(
            evidence,
            RecordedTransitionEvidenceV1::UpdateMetadataVersions { .. }
        ),
        "standalone transition must record update metadata versions",
    )
}

fn require_recorded_manager_route(
    evidence: &RecordedTransitionEvidenceV1,
    expected: Option<UpdateRoute>,
) -> Result<(), InstallReceiptError> {
    require(
        matches!(
            evidence,
            RecordedTransitionEvidenceV1::PackageManagerRoute { route } if Some(*route) == expected
        ),
        "recorded manager route must match the selected route",
    )
}

fn require(condition: bool, reason: &'static str) -> Result<(), InstallReceiptError> {
    if condition {
        Ok(())
    } else {
        Err(InstallReceiptError::TransitionMismatch(reason))
    }
}

fn parse_owner(field: &'static str, value: &str) -> Result<OwnerFamily, InstallReceiptError> {
    match value {
        "standalone" => Ok(OwnerFamily::Standalone),
        "homebrew" => Ok(OwnerFamily::Homebrew),
        "cargo-registry" => Ok(OwnerFamily::CargoRegistry),
        "unknown/manual" => Ok(OwnerFamily::UnknownManual),
        _ => Err(InstallReceiptError::invalid(
            field,
            "unsupported owner family",
        )),
    }
}

fn parse_route(field: &'static str, value: &str) -> Result<UpdateRoute, InstallReceiptError> {
    match value {
        "standalone" => Ok(UpdateRoute::Standalone),
        "brew" => Ok(UpdateRoute::Brew),
        "cargo-install" => Ok(UpdateRoute::CargoInstall),
        "cargo-binstall" => Ok(UpdateRoute::CargoBinstall),
        _ => Err(InstallReceiptError::invalid(
            field,
            "unsupported update route",
        )),
    }
}

fn parse_transition_kind(value: &str) -> Result<TransitionKind, InstallReceiptError> {
    match value {
        "install" => Ok(TransitionKind::Install),
        "update" => Ok(TransitionKind::Update),
        "rollback" => Ok(TransitionKind::Rollback),
        "confirmed-migration" => Ok(TransitionKind::ConfirmedMigration),
        _ => Err(InstallReceiptError::invalid(
            "last_successful_transition.type",
            "unsupported transition type",
        )),
    }
}

fn is_at_or_below_versions(
    state_root: &AbsoluteInstallPath,
    installation_root: &AbsoluteInstallPath,
) -> bool {
    let versions = format!("{}/versions", installation_root.as_str());
    state_root.as_str() == versions
        || state_root
            .as_str()
            .strip_prefix(&versions)
            .is_some_and(|suffix| suffix.starts_with('/'))
}
