use super::*;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
static ABORT_AFTER_NORMALIZATION_BARRIER: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static FAIL_AFTER_NORMALIZATION_BARRIER: AtomicUsize = AtomicUsize::new(0);

#[cfg(target_os = "macos")]
pub(in crate::distribution::install_state) fn with_extracted_executable<R, E>(
    locked: &LockedInstallationIdentity,
    retained: &ExtractedReleaseTree,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    operation: impl FnOnce(&std::path::Path, &std::fs::File, ExecutableReleaseBinding) -> Result<R, E>,
) -> Result<R, E>
where
    E: From<ExtractionError>,
{
    with_retained_executable(
        locked,
        &retained._extractions,
        &retained._stage,
        &retained.stage_name,
        exact_manifest,
        manifest,
        false,
        operation,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::distribution::install_state) fn with_normalized_executable<R, E>(
    locked: &LockedInstallationIdentity,
    retained: &NormalizedExtractedReleaseTree,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    operation: impl FnOnce(&std::path::Path, &std::fs::File, ExecutableReleaseBinding) -> Result<R, E>,
) -> Result<R, E>
where
    E: From<ExtractionError>,
{
    with_retained_executable(
        locked,
        &retained._extractions,
        &retained._stage,
        &retained.stage_name,
        exact_manifest,
        manifest,
        true,
        operation,
    )
}

pub(in crate::distribution::install_state) fn normalize_release_tree(
    locked: &LockedInstallationIdentity,
    retained: ExtractedReleaseTree,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
) -> Result<NormalizedExtractedReleaseTree, ExtractionError> {
    if manifest
        .to_deterministic_json()
        .map_err(|_| ExtractionError::Integrity)?
        != exact_manifest
    {
        return Err(ExtractionError::Integrity);
    }
    let files = expected_files(exact_manifest, manifest);
    let directories = manifest.derived_directories().to_vec();
    let expected = expected_tree(&directories, &files)?;

    let live = locked.reopen()?;
    let extractions = unix::open_directory_at(&live.update, EXTRACTIONS, Some(0o700), true)?;
    if !extractions.same_object(&retained._extractions) {
        return Err(ExtractionError::Integrity);
    }
    let stage = unix::open_directory_at(&extractions, &retained.stage_name, Some(0o700), true)?;
    if !stage.same_object(&retained._stage) {
        return Err(ExtractionError::Integrity);
    }
    let states = scan_tree(&stage, &expected, &files, &directories)?;
    validate_prefix(&states)?;
    if states.iter().any(|state| !state.is_complete()) {
        return Err(ExtractionError::Integrity);
    }

    normalize_files(&stage, &files, &states)?;
    normalize_directories(&stage, &directories)?;
    unix::sync_directory(&stage)?;
    normalization_barrier()?;
    unix::sync_directory(&extractions)?;
    normalization_barrier()?;
    unix::sync_directory(&live.update)?;
    normalization_barrier()?;
    unix::sync_directory(&live.root)?;
    normalization_barrier()?;

    let rebound = locked.reopen()?;
    let rebound_extractions =
        unix::open_directory_at(&rebound.update, EXTRACTIONS, Some(0o700), true)?;
    if !rebound_extractions.same_object(&extractions) {
        return Err(ExtractionError::Integrity);
    }
    let rebound_stage = unix::open_directory_at(
        &rebound_extractions,
        &retained.stage_name,
        Some(0o700),
        true,
    )?;
    if !rebound_stage.same_object(&stage) {
        return Err(ExtractionError::Integrity);
    }
    let rebound_states = scan_tree(&rebound_stage, &expected, &files, &directories)?;
    if rebound_states
        .iter()
        .any(|state| *state != FileState::CompleteFinal)
        || directory_normalization_order(&directories)
            .into_iter()
            .any(|path| {
                !matches!(
                    existing_directory_mode(&rebound_stage, path),
                    Ok(Some(0o755))
                )
            })
    {
        return Err(ExtractionError::Integrity);
    }
    normalization_barrier()?;
    locked.full_sync_endpoint()?;
    normalization_barrier()?;
    Ok(NormalizedExtractedReleaseTree {
        _extractions: rebound_extractions,
        _stage: rebound_stage,
        stage_name: retained.stage_name,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::distribution::install_state) fn normalize_developer_id_verified_release(
    locked: &LockedInstallationIdentity,
    developer_id: crate::distribution::prepared_release::DeveloperIdVerification,
    retained: ExtractedReleaseTree,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
) -> Result<NormalizedExtractedReleaseTree, ExtractionError> {
    let current = executable_binding(
        locked,
        &retained._extractions,
        &retained._stage,
        &retained.stage_name,
        exact_manifest,
        manifest,
        false,
    )?;
    if !developer_id.matches(current) {
        return Err(ExtractionError::Integrity);
    }
    normalize_release_tree(locked, retained, exact_manifest, manifest)
}

pub(in crate::distribution::install_state) fn verify_normalized_release_tree(
    locked: &LockedInstallationIdentity,
    retained: &NormalizedExtractedReleaseTree,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
) -> Result<(), ExtractionError> {
    if manifest
        .to_deterministic_json()
        .map_err(|_| ExtractionError::Integrity)?
        != exact_manifest
    {
        return Err(ExtractionError::Integrity);
    }
    let files = expected_files(exact_manifest, manifest);
    let directories = manifest.derived_directories().to_vec();
    let expected = expected_tree(&directories, &files)?;
    let live = locked.reopen()?;
    let extractions = unix::open_directory_at(&live.update, EXTRACTIONS, Some(0o700), true)?;
    if !extractions.same_object(&retained._extractions) {
        return Err(ExtractionError::Integrity);
    }
    let stage = unix::open_directory_at(&extractions, &retained.stage_name, Some(0o700), true)?;
    if !stage.same_object(&retained._stage) {
        return Err(ExtractionError::Integrity);
    }
    let states = scan_tree(&stage, &expected, &files, &directories)?;
    if states
        .iter()
        .any(|state| *state != FileState::CompleteFinal)
        || directory_normalization_order(&directories)
            .into_iter()
            .any(|path| !matches!(existing_directory_mode(&stage, path), Ok(Some(0o755))))
    {
        return Err(ExtractionError::Integrity);
    }
    Ok(())
}

fn normalize_files(
    stage: &Directory,
    files: &[ExpectedFile],
    states: &[FileState],
) -> Result<(), ExtractionError> {
    for (expected, state) in files.iter().zip(states) {
        let (parent, name) = open_existing_parent(stage, &expected.path)?;
        let current_mode = match state {
            FileState::CompletePrivate => 0o600,
            FileState::CompleteFinal => expected.final_mode,
            FileState::Absent | FileState::Partial(_) => return Err(ExtractionError::Integrity),
        };
        let (file, identity) = unix::open_regular_file_with_mode(&parent, name, current_mode)?;
        verify_file(&file, identity, expected)?;
        unix::verify_named_identity(&parent, name, identity)?;
        let normalized = if current_mode == expected.final_mode {
            identity
        } else {
            unix::set_regular_file_mode(&file, parent.device(), current_mode, expected.final_mode)?
        };
        normalization_barrier()?;
        unix::verify_named_identity(&parent, name, normalized)?;
        unix::full_sync_file(&file)?;
        normalization_barrier()?;
        verify_file(&file, normalized, expected)?;
        unix::verify_named_identity(&parent, name, normalized)?;
    }
    Ok(())
}

fn normalize_directories(stage: &Directory, directories: &[String]) -> Result<(), ExtractionError> {
    for path in directory_normalization_order(directories) {
        let directory = open_existing_directory(stage, path)?;
        let normalized = match directory.mode() {
            0o700 => unix::set_directory_mode(directory, 0o700, 0o755)?,
            0o755 => directory,
            _ => return Err(ExtractionError::Integrity),
        };
        normalization_barrier()?;
        unix::sync_directory(&normalized)?;
        normalization_barrier()?;
        let reopened = open_existing_directory(stage, path)?;
        if reopened.mode() != 0o755 || !reopened.same_object(&normalized) {
            return Err(ExtractionError::Integrity);
        }
    }
    Ok(())
}

#[cfg(not(test))]
fn normalization_barrier() -> Result<(), ExtractionError> {
    Ok(())
}

#[cfg(test)]
fn normalization_barrier() -> Result<(), ExtractionError> {
    let remaining = ABORT_AFTER_NORMALIZATION_BARRIER.load(Ordering::SeqCst);
    if remaining != 0 && ABORT_AFTER_NORMALIZATION_BARRIER.fetch_sub(1, Ordering::SeqCst) == 1 {
        std::process::abort();
    }
    let remaining = FAIL_AFTER_NORMALIZATION_BARRIER.load(Ordering::SeqCst);
    if remaining != 0 && FAIL_AFTER_NORMALIZATION_BARRIER.fetch_sub(1, Ordering::SeqCst) == 1 {
        return Err(ExtractionError::Integrity);
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn abort_after_normalization_barrier(barrier: usize) {
    assert!(barrier != 0, "normalization barrier is one-based");
    ABORT_AFTER_NORMALIZATION_BARRIER.store(barrier, Ordering::SeqCst);
}

#[cfg(test)]
pub(super) fn fail_after_normalization_barrier(barrier: usize) {
    assert!(barrier != 0, "normalization barrier is one-based");
    FAIL_AFTER_NORMALIZATION_BARRIER.store(barrier, Ordering::SeqCst);
}

#[cfg(test)]
pub(super) fn normalization_barrier_count(manifest: &ReleaseManifestV1) -> usize {
    // Every expected file and derived directory has a mode-transition and a
    // full-sync barrier. The four namespace syncs, live-tree rebind, and lock
    // endpoint full-sync are the six transaction-wide barriers.
    2 * (manifest.files().len() + 1 + manifest.derived_directories().len()) + 6
}

#[cfg(target_os = "macos")]
fn with_retained_executable<R, E>(
    locked: &LockedInstallationIdentity,
    retained_extractions: &Directory,
    retained_stage: &Directory,
    stage_name: &str,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    require_final_mode: bool,
    operation: impl FnOnce(&std::path::Path, &std::fs::File, ExecutableReleaseBinding) -> Result<R, E>,
) -> Result<R, E>
where
    E: From<ExtractionError>,
{
    if manifest
        .to_deterministic_json()
        .map_err(|_| ExtractionError::Integrity)
        .map_err(E::from)?
        != exact_manifest
    {
        return Err(E::from(ExtractionError::Integrity));
    }
    let files = expected_files(exact_manifest, manifest);
    let binary = files
        .iter()
        .find(|file| file.path == "bin/hf2q")
        .ok_or(ExtractionError::Integrity)
        .map_err(E::from)?;
    let (file, identity, path) = open_live_executable(
        locked,
        retained_extractions,
        retained_stage,
        stage_name,
        binary,
        require_final_mode,
    )
    .map_err(E::from)?;
    let binding = binding_from_live(
        retained_extractions,
        retained_stage,
        identity,
        path.clone(),
        exact_manifest,
    );
    let result = operation(&path, &file, binding)?;
    let (reopened, reopened_identity, reopened_path) = open_live_executable(
        locked,
        retained_extractions,
        retained_stage,
        stage_name,
        binary,
        require_final_mode,
    )
    .map_err(E::from)?;
    if identity != reopened_identity || path != reopened_path {
        return Err(E::from(ExtractionError::Integrity));
    }
    verify_file(&reopened, reopened_identity, binary).map_err(E::from)?;
    Ok(result)
}

#[cfg(target_os = "macos")]
fn executable_binding(
    locked: &LockedInstallationIdentity,
    retained_extractions: &Directory,
    retained_stage: &Directory,
    stage_name: &str,
    exact_manifest: &[u8],
    manifest: &ReleaseManifestV1,
    require_final_mode: bool,
) -> Result<ExecutableReleaseBinding, ExtractionError> {
    if manifest
        .to_deterministic_json()
        .map_err(|_| ExtractionError::Integrity)?
        != exact_manifest
    {
        return Err(ExtractionError::Integrity);
    }
    let files = expected_files(exact_manifest, manifest);
    let binary = files
        .iter()
        .find(|file| file.path == "bin/hf2q")
        .ok_or(ExtractionError::Integrity)?;
    let (_file, identity, path) = open_live_executable(
        locked,
        retained_extractions,
        retained_stage,
        stage_name,
        binary,
        require_final_mode,
    )?;
    Ok(binding_from_live(
        retained_extractions,
        retained_stage,
        identity,
        path,
        exact_manifest,
    ))
}

#[cfg(target_os = "macos")]
fn binding_from_live(
    extractions: &Directory,
    stage: &Directory,
    executable: EntryIdentity,
    executable_path: std::path::PathBuf,
    exact_manifest: &[u8],
) -> ExecutableReleaseBinding {
    ExecutableReleaseBinding {
        extractions_device: extractions.device(),
        extractions_inode: extractions.inode(),
        stage_device: stage.device(),
        stage_inode: stage.inode(),
        executable,
        executable_path,
        manifest_sha256: Sha256::digest(exact_manifest).into(),
    }
}

#[cfg(target_os = "macos")]
fn open_live_executable(
    locked: &LockedInstallationIdentity,
    retained_extractions: &Directory,
    retained_stage: &Directory,
    stage_name: &str,
    binary: &ExpectedFile,
    require_final_mode: bool,
) -> Result<(std::fs::File, EntryIdentity, std::path::PathBuf), ExtractionError> {
    let live = locked.reopen()?;
    let extractions = unix::open_directory_at(&live.update, EXTRACTIONS, Some(0o700), true)?;
    if !extractions.same_object(retained_extractions) {
        return Err(ExtractionError::Integrity);
    }
    let stage = unix::open_directory_at(&extractions, stage_name, Some(0o700), true)?;
    if !stage.same_object(retained_stage) {
        return Err(ExtractionError::Integrity);
    }
    let (parent, name) = open_existing_parent(&stage, &binary.path)?;
    let named = unix::entry_identity(&parent, name)?.ok_or(ExtractionError::Integrity)?;
    let mode = if named.mode == binary.final_mode {
        binary.final_mode
    } else if !require_final_mode && named.mode == 0o600 {
        0o600
    } else {
        return Err(ExtractionError::Integrity);
    };
    let (file, identity) = unix::open_regular_file_with_mode(&parent, name, mode)?;
    verify_file(&file, identity, binary)?;
    unix::verify_named_identity(&parent, name, identity)?;
    let path = unix::file_descriptor_path(&file)?;
    let expected_path = unix::directory_descriptor_path(&stage)?.join(&binary.path);
    if path != expected_path {
        return Err(ExtractionError::Integrity);
    }
    Ok((file, identity, path))
}
