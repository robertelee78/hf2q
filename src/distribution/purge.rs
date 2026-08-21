//! Explicit, bounded operator-data purge used only by `hf2q uninstall`.

use std::fs::{self, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};

use crate::serve::cache::{
    CacheManifest, ModelCache, MANIFEST_SCHEMA_MIN_SUPPORTED, MANIFEST_SCHEMA_VERSION,
};

const MAX_CACHE_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub(crate) enum PurgeError {
    #[error("invalid hf2q cache purge target: {0}")]
    Invalid(String),
    #[error("hf2q cache purge failed: {0}")]
    Cache(#[from] anyhow::Error),
    #[error("cache purge filesystem operation `{operation}` failed: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
}

impl PurgeError {
    fn io(operation: &'static str, source: std::io::Error) -> Self {
        Self::Io { operation, source }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CachePurgePlan {
    pub(crate) root: PathBuf,
    pub(crate) models: PathBuf,
    pub(crate) manifest: PathBuf,
    identity: Option<(u64, u64)>,
    present: bool,
}

impl CachePurgePlan {
    pub(crate) fn contains_data(&self) -> bool {
        self.present
    }
}

pub(crate) fn prepare_cache_purge() -> Result<CachePurgePlan, PurgeError> {
    let root = crate::serve::cache::default_root().map_err(PurgeError::Cache)?;
    prepare_cache_purge_at(&root)
}

pub(crate) fn execute_cache_purge(plan: &CachePurgePlan) -> Result<u64, PurgeError> {
    let current = prepare_cache_purge_at(&plan.root)?;
    if current != *plan {
        return Err(PurgeError::Invalid(format!(
            "{} changed after the purge preview",
            plan.root.display()
        )));
    }
    if !plan.present {
        return Ok(0);
    }
    let mut cache = ModelCache::open_at(&plan.root).map_err(PurgeError::Cache)?;
    cache.purge().map_err(PurgeError::Cache)
}

pub(super) fn prepare_cache_purge_at(root: &Path) -> Result<CachePurgePlan, PurgeError> {
    validate_absolute_non_root(root)?;
    let models = root.join("models");
    let manifest = root.join("manifest.json");
    let root_metadata = match fs::symlink_metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CachePurgePlan {
                root: root.to_owned(),
                models,
                manifest,
                identity: None,
                present: false,
            })
        }
        Err(error) => return Err(PurgeError::io("inspect cache root", error)),
    };
    let canonical =
        fs::canonicalize(root).map_err(|error| PurgeError::io("canonicalize cache root", error))?;
    if canonical != root
        || !root_metadata.is_dir()
        || root_metadata.uid() != rustix::process::geteuid().as_raw()
        || root_metadata.mode() & 0o022 != 0
    {
        return Err(PurgeError::Invalid(format!(
            "{} must be a canonical current-user-owned directory that is not group/world-writable",
            root.display()
        )));
    }
    validate_models_directory(&models, &root_metadata)?;
    let identity = Some((root_metadata.dev(), root_metadata.ino()));
    let manifest_metadata = match fs::symlink_metadata(&manifest) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if directory_has_entries(&models)? {
                return Err(PurgeError::Invalid(format!(
                    "{} contains model data but has no hf2q cache manifest",
                    root.display()
                )));
            }
            return Ok(CachePurgePlan {
                root: root.to_owned(),
                models,
                manifest,
                identity,
                present: false,
            });
        }
        Err(error) => return Err(PurgeError::io("inspect cache manifest", error)),
    };
    let bytes = read_manifest(&manifest, &root_metadata, &manifest_metadata)?;
    let parsed: CacheManifest = serde_json::from_slice(&bytes)
        .map_err(|_| PurgeError::Invalid("cache manifest is not valid schema JSON".to_owned()))?;
    if parsed.schema_version < MANIFEST_SCHEMA_MIN_SUPPORTED
        || parsed.schema_version > MANIFEST_SCHEMA_VERSION
    {
        return Err(PurgeError::Invalid(format!(
            "cache manifest schema {} is unsupported",
            parsed.schema_version
        )));
    }
    Ok(CachePurgePlan {
        root: root.to_owned(),
        models,
        manifest,
        identity,
        present: true,
    })
}

fn validate_absolute_non_root(root: &Path) -> Result<(), PurgeError> {
    if !root.is_absolute() || root == Path::new("/") || root.as_os_str().len() > 1024 {
        return Err(PurgeError::Invalid(
            "cache root must be an absolute non-root path of at most 1024 bytes".to_owned(),
        ));
    }
    if root.components().any(|component| {
        matches!(
            component,
            Component::CurDir | Component::ParentDir | Component::Prefix(_)
        )
    }) {
        return Err(PurgeError::Invalid(
            "cache root may not contain relative components".to_owned(),
        ));
    }
    Ok(())
}

fn validate_models_directory(
    models: &Path,
    root_metadata: &fs::Metadata,
) -> Result<(), PurgeError> {
    match fs::symlink_metadata(models) {
        Ok(metadata)
            if metadata.is_dir()
                && metadata.uid() == rustix::process::geteuid().as_raw()
                && metadata.dev() == root_metadata.dev()
                && metadata.mode() & 0o022 == 0 =>
        {
            Ok(())
        }
        Ok(_) => Err(PurgeError::Invalid(format!(
            "{} is not an owned non-writable cache directory",
            models.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(PurgeError::io("inspect cache models directory", error)),
    }
}

fn directory_has_entries(path: &Path) -> Result<bool, PurgeError> {
    match fs::read_dir(path) {
        Ok(mut entries) => Ok(entries
            .next()
            .transpose()
            .map_err(|error| PurgeError::io("inspect cache models contents", error))?
            .is_some()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(PurgeError::io("open cache models directory", error)),
    }
}

fn read_manifest(
    path: &Path,
    root_metadata: &fs::Metadata,
    named: &fs::Metadata,
) -> Result<Vec<u8>, PurgeError> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| PurgeError::io("open cache manifest", error))?;
    let opened = file
        .metadata()
        .map_err(|error| PurgeError::io("inspect cache manifest", error))?;
    if !opened.is_file()
        || opened.uid() != rustix::process::geteuid().as_raw()
        || opened.mode() & 0o022 != 0
        || opened.nlink() != 1
        || opened.dev() != root_metadata.dev()
        || opened.len() == 0
        || opened.len() > MAX_CACHE_MANIFEST_BYTES
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
    {
        return Err(PurgeError::Invalid(format!(
            "{} is not a bounded current-user-controlled cache manifest",
            path.display()
        )));
    }
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    file.by_ref()
        .take(MAX_CACHE_MANIFEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| PurgeError::io("read cache manifest", error))?;
    let after = file
        .metadata()
        .map_err(|error| PurgeError::io("reinspect cache manifest", error))?;
    if opened.dev() != after.dev()
        || opened.ino() != after.ino()
        || opened.len() != after.len()
        || bytes.len() as u64 > MAX_CACHE_MANIFEST_BYTES
    {
        return Err(PurgeError::Invalid(
            "cache manifest changed while it was read".to_owned(),
        ));
    }
    Ok(bytes)
}
