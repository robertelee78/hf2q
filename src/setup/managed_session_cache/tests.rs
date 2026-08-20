use std::fs;
use std::io::Cursor;
use std::os::unix::fs::{symlink, MetadataExt, PermissionsExt};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use super::super::host::{HostObservation, HostProbe};
use super::super::runtime_policy::{
    authorize_session_cache_policy, SessionCachePolicyAuthorization,
};
use super::super::schema::{ConfiguredShell, HardwareProfileV1};
use super::super::{execute, SetupError};
use super::catalog::{sha256_hex, CatalogEntryV1, CatalogV1};
use super::transaction::{
    commit_with_test_hook, reset_test_object_write_failure_len, set_test_volume_space,
    take_test_object_write_failure_len, with_test_io_fault, with_test_volume_space, TestIoFault,
};
use super::{
    encode_object, ManagedCacheBarrier, ManagedCheckpointKey, ManagedCommitOutcome,
    ManagedSessionCache, ManagedSessionCacheError, CATALOG_PARTIAL, OBJECT_PARTIAL,
};
use crate::cli::{SessionCacheChoice, SetupArgs};

const GIB: u64 = 1024 * 1024 * 1024;

pub(super) fn abort_at_managed_cache_barrier(barrier: ManagedCacheBarrier) {
    if std::env::var("HF2Q_MANAGED_CACHE_ABORT_AT").as_deref() == Ok(barrier.as_str()) {
        unsafe { libc::raise(libc::SIGABRT) };
        std::process::abort();
    }
}

struct FakeProbe;

impl HostProbe for FakeProbe {
    fn observe(&self, _state_root: &Path) -> Result<HostObservation, SetupError> {
        Ok(HostObservation {
            hardware: HardwareProfileV1 {
                target: "aarch64-apple-darwin".to_owned(),
                chip_model: "Apple M5 Max".to_owned(),
                unified_memory_bytes: 128 * GIB,
                metal_device_name: "Apple M5 Max".to_owned(),
                metal_recommended_working_set_bytes: 96 * GIB,
            },
            macos_version: "15.6.1".to_owned(),
            configured_shell: ConfiguredShell::Zsh,
            performance_level0_name: "Super".to_owned(),
            performance_level0_cores: 4,
            performance_level1_name: "Performance".to_owned(),
            performance_level1_cores: 12,
            open_file_soft_limit: 10240,
            volume_total_bytes: 500 * GIB,
            volume_available_bytes: 200 * GIB,
        })
    }
}

fn root(temp: &TempDir, name: &str) -> PathBuf {
    temp.path().canonicalize().unwrap().join(name)
}

fn configure(root: &Path, limit: &str) {
    let args = SetupArgs {
        session_cache: Some(SessionCacheChoice::On),
        session_cache_limit: Some(limit.to_owned()),
        state_root: Some(root.to_owned()),
    };
    execute(
        args,
        &FakeProbe,
        false,
        &mut Cursor::new(Vec::<u8>::new()),
        &mut Vec::new(),
    )
    .unwrap();
}

fn open_store(root: &Path) -> Result<ManagedSessionCache, ManagedSessionCacheError> {
    let authorization = authorize_session_cache_policy(root).unwrap();
    let SessionCachePolicyAuthorization::Enabled(authorization) = authorization else {
        panic!("positive setup policy must mint enabled authority");
    };
    authorization.into_managed_store()
}

fn with_room<T>(action: impl FnOnce() -> T) -> T {
    with_test_volume_space(500 * GIB, 200 * GIB, 4096, action)
}

fn snapshot_tree(root: &Path) -> Vec<(PathBuf, u64, u64, u32)> {
    fn visit(base: &Path, current: &Path, entries: &mut Vec<(PathBuf, u64, u64, u32)>) {
        let Ok(directory) = fs::read_dir(current) else {
            return;
        };
        let mut children: Vec<_> = directory.map(|entry| entry.unwrap().path()).collect();
        children.sort();
        for child in children {
            let metadata = fs::symlink_metadata(&child).unwrap();
            entries.push((
                child.strip_prefix(base).unwrap().to_owned(),
                metadata.ino(),
                metadata.len(),
                metadata.mode(),
            ));
            if metadata.is_dir() {
                visit(base, &child, entries);
            }
        }
    }
    let mut entries = Vec::new();
    visit(root, root, &mut entries);
    entries
}

include!("tests/basic.rs");
include!("tests/hostile.rs");
include!("tests/storage.rs");
include!("tests/crash.rs");
