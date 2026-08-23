use std::fs;
use std::io::{self, BufRead, Cursor, Read};
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};

use clap::Parser;
use tempfile::TempDir;

use super::fs as setup_fs;
use super::host::{
    nearest_existing_directory, require_supported_macos, HostObservation, HostProbe,
};
use super::host::{validate_performance_levels, HardwareProfile, PerformanceLevel};
use super::schema::{ConfiguredScheduler, ConfiguredShell, OperatorConfigV2};
use super::{execute, SetupError};
use crate::cli::{Cli, Command, SchedulerArg, SetupArgs};

const GIB: u64 = 1024 * 1024 * 1024;

pub(super) fn abort_at_setup_barrier(barrier: setup_fs::SetupBarrier) {
    if std::env::var("HF2Q_SETUP_ABORT_AT").as_deref() == Ok(barrier.as_str()) {
        unsafe { libc::raise(libc::SIGABRT) };
        std::process::abort();
    }
}

struct FakeProbe {
    total: u64,
    available: u64,
}

impl HostProbe for FakeProbe {
    fn observe(&self, _state_root: &Path) -> Result<HostObservation, SetupError> {
        Ok(HostObservation {
            hardware: fixture_hardware(),
            macos_version: "15.6.1".to_owned(),
            configured_shell: ConfiguredShell::Zsh,
            performance_levels: vec![
                PerformanceLevel {
                    name: "Super".to_owned(),
                    logical_cores: 4,
                },
                PerformanceLevel {
                    name: "Performance".to_owned(),
                    logical_cores: 12,
                },
            ],
            logical_cores: 16,
            open_file_soft_limit: 10240,
            volume_total_bytes: self.total,
            volume_available_bytes: self.available,
        })
    }
}

fn fixture_hardware() -> HardwareProfile {
    HardwareProfile {
        target: "aarch64-apple-darwin".to_owned(),
        chip_model: "Apple M5 Max".to_owned(),
        unified_memory_bytes: 128 * GIB,
        metal_device_name: "Apple M5 Max".to_owned(),
        metal_recommended_working_set_bytes: 96 * GIB,
    }
}

#[test]
fn named_performance_levels_accept_one_or_more_levels_and_reject_incoherent_facts() {
    let one = [PerformanceLevel {
        name: "Performance".to_owned(),
        logical_cores: 4,
    }];
    validate_performance_levels(&one, 4).unwrap();

    let two = [
        PerformanceLevel {
            name: "Super".to_owned(),
            logical_cores: 6,
        },
        PerformanceLevel {
            name: "Performance".to_owned(),
            logical_cores: 12,
        },
    ];
    validate_performance_levels(&two, 18).unwrap();
    assert!(validate_performance_levels(&two, 17).is_err());
    assert!(validate_performance_levels(&[], 0).is_err());

    let duplicate = [
        PerformanceLevel {
            name: "Performance".to_owned(),
            logical_cores: 2,
        },
        PerformanceLevel {
            name: "Performance".to_owned(),
            logical_cores: 2,
        },
    ];
    assert!(validate_performance_levels(&duplicate, 4).is_err());
}

fn fixture_config(limit_bytes: u64) -> (OperatorConfigV2, Vec<u8>) {
    let mut config = OperatorConfigV2::guide_defaults().unwrap();
    if limit_bytes != 0 {
        config.convert.quant = "q5_k_m".to_owned();
        config.serve.port = 9090;
    }
    let bytes = config.to_canonical_bytes().unwrap();
    (config, bytes)
}

struct TestInvocation {
    root: PathBuf,
    args: SetupArgs,
}

fn args(root: &Path) -> TestInvocation {
    TestInvocation {
        root: root.to_owned(),
        args: SetupArgs {
            default_quant: None,
            serve_host: None,
            serve_port: None,
            serve_scheduler: None,
            serve_max_slots: None,
            serve_kv_persist_budget: None,
            accept_defaults: true,
        },
    }
}

fn prompt_args(root: &Path) -> TestInvocation {
    let mut invocation = args(root);
    invocation.args.accept_defaults = false;
    invocation
}

fn test_root(temp: &TempDir, name: &str) -> PathBuf {
    temp.path().canonicalize().unwrap().join(name)
}

#[test]
fn config_purge_preview_is_non_mutating_and_execution_removes_only_owned_names() {
    let temp = TempDir::new().unwrap();
    let missing = test_root(&temp, "missing-state");
    let plan = super::prepare_config_purge(Some(&missing)).unwrap();
    assert_eq!(plan.root, missing);
    assert!(!plan.root.exists(), "preview must not create a state root");
    assert!(super::execute_config_purge(&plan).unwrap().is_empty());
    assert!(!plan.root.exists(), "missing purge remains a no-op");

    let root = test_root(&temp, "state");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    for (name, bytes) in [
        ("config.toml", b"config".as_slice()),
        (".config.toml.partial", b"partial".as_slice()),
        (".config.toml.lock", b"".as_slice()),
        ("operator-note", b"preserve".as_slice()),
    ] {
        fs::write(root.join(name), bytes).unwrap();
        fs::set_permissions(root.join(name), fs::Permissions::from_mode(0o600)).unwrap();
    }
    let before_note = fs::read(root.join("operator-note")).unwrap();
    let plan = super::prepare_config_purge(Some(&root)).unwrap();
    assert!(root.join("config.toml").exists());
    let removed = super::execute_config_purge(&plan).unwrap();
    assert_eq!(removed.len(), 3);
    for path in plan.paths {
        assert!(!path.exists(), "purged {}", path.display());
    }
    assert_eq!(fs::read(root.join("operator-note")).unwrap(), before_note);
    assert!(
        root.exists(),
        "purge preserves the selected root and siblings"
    );
}

#[test]
fn config_purge_rejects_hostile_leaf_and_busy_setup_without_removing_config() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "hostile-state");
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    let outside = test_root(&temp, "outside");
    fs::write(&outside, b"outside").unwrap();
    std::os::unix::fs::symlink(&outside, root.join("config.toml")).unwrap();
    assert!(super::prepare_config_purge(Some(&root)).is_err());
    assert_eq!(fs::read(&outside).unwrap(), b"outside");

    fs::remove_file(root.join("config.toml")).unwrap();
    fs::write(root.join("config.toml"), b"config").unwrap();
    fs::set_permissions(root.join("config.toml"), fs::Permissions::from_mode(0o600)).unwrap();
    let held = setup_fs::hold_setup_lock(&root).unwrap();
    let plan = super::prepare_config_purge(Some(&root)).unwrap();
    assert!(matches!(
        super::execute_config_purge(&plan),
        Err(SetupError::Busy)
    ));
    assert_eq!(fs::read(root.join("config.toml")).unwrap(), b"config");
    drop(held);
}

fn execute_with_probe(
    invocation: TestInvocation,
    interactive: bool,
    input: &str,
    total: u64,
    available: u64,
) -> Result<String, SetupError> {
    let mut input = Cursor::new(input.as_bytes());
    let mut output = Vec::new();
    execute(
        invocation.args,
        Some(&invocation.root),
        &FakeProbe { total, available },
        interactive,
        &mut input,
        &mut output,
    )?;
    Ok(String::from_utf8(output).expect("setup output is UTF-8"))
}

fn execute_fake(
    invocation: TestInvocation,
    interactive: bool,
    input: &str,
) -> Result<String, SetupError> {
    execute_with_probe(invocation, interactive, input, 500 * GIB, 200 * GIB)
}

#[test]
fn cli_parses_the_closed_noninteractive_surface() {
    let raw = [
        "hf2q",
        "--state-root",
        "/tmp/hf2q-state",
        "setup",
        "--default-quant",
        "q4_k_m",
        "--serve-host",
        "127.0.0.1",
        "--serve-port",
        "8081",
        "--serve-scheduler",
        "inflight-batched",
        "--serve-max-slots",
        "1",
        "--serve-kv-persist-budget",
        "32GiB",
    ];
    let cli = Cli::try_parse_from(raw).unwrap();
    let Command::Setup(args) = cli.command else {
        panic!("setup command was not selected");
    };
    assert_eq!(cli.state_root.unwrap(), Path::new("/tmp/hf2q-state"));
    assert_eq!(args.default_quant.as_deref(), Some("q4_k_m"));
    assert_eq!(args.serve_host.as_deref(), Some("127.0.0.1"));
    assert_eq!(args.serve_port, Some(8081));
    assert_eq!(args.serve_scheduler, Some(SchedulerArg::InflightBatched));
    assert_eq!(args.serve_max_slots, Some(1));
    assert_eq!(args.serve_kv_persist_budget.as_deref(), Some("32GiB"));

    let cli = Cli::try_parse_from([
        "hf2q",
        "setup",
        "--accept-defaults",
        "--state-root",
        "/tmp/hf2q-state-after",
    ])
    .unwrap();
    assert_eq!(
        cli.state_root.as_deref(),
        Some(Path::new("/tmp/hf2q-state-after"))
    );
}

#[test]
fn host_version_and_selected_storage_ancestor_rules_are_closed() {
    for valid in ["14.0", "15.6.1", "99.255.255"] {
        require_supported_macos(valid).unwrap();
    }
    for invalid in ["13.9", "14", "14.0.0.1", "014.0", "14.00", "14.a", ""] {
        assert!(
            require_supported_macos(invalid).is_err(),
            "accepted {invalid:?}"
        );
    }
    let temp = TempDir::new().unwrap();
    let parent = temp.path().canonicalize().unwrap();
    assert_eq!(
        nearest_existing_directory(&parent.join("one/two/state")).unwrap(),
        parent
    );
    let file = parent.join("not-a-directory");
    fs::write(&file, b"evidence").unwrap();
    assert!(nearest_existing_directory(&file).is_err());
}

#[test]
fn fresh_and_repeated_noninteractive_setup_are_idempotent() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "state");
    let output = execute_fake(args(&root), false, "").unwrap();
    assert!(output.contains("Configured"));
    assert!(output.contains("No model was downloaded"));
    assert!(output.contains("q4_k_m"));
    assert!(output.contains("127.0.0.1:8081"));
    assert!(output.contains("inflight_batched (max slots 1)"));
    let before = fs::read(root.join("config.toml")).unwrap();
    assert_eq!(before, include_bytes!("testdata/config_v2.toml"));
    let metadata = fs::metadata(root.join("config.toml")).unwrap();
    assert_eq!(metadata.mode() & 0o777, 0o600);
    assert_eq!(metadata.nlink(), 1);
    assert_eq!(fs::metadata(&root).unwrap().mode() & 0o777, 0o700);
    assert!(!root.join("cache").exists());
    assert!(!root.join(".config.toml.partial").exists());
    assert!(!root.join("update").exists());

    let output = execute_fake(args(&root), false, "").unwrap();
    assert!(output.contains("Verified"));
    assert_eq!(fs::read(root.join("config.toml")).unwrap(), before);
    let after = fs::metadata(root.join("config.toml")).unwrap();
    assert_eq!(after.ino(), metadata.ino());
    assert_eq!(after.mtime(), metadata.mtime());
    assert_eq!(after.mtime_nsec(), metadata.mtime_nsec());
}

#[test]
fn interactive_setup_uses_current_values_and_records_explicit_operator_choices() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "state");
    execute_fake(args(&root), false, "").unwrap();
    let output = execute_fake(
        prompt_args(&root),
        true,
        " q5_k_m \n n \n y \n 9090 \n 24GiB \n",
    )
    .unwrap();
    assert!(output.contains("[Y/n]"));
    let config = OperatorConfigV2::parse(&fs::read(root.join("config.toml")).unwrap()).unwrap();
    assert_eq!(config.convert.quant, "q5_k_m");
    assert_eq!(config.serve.scheduler, ConfiguredScheduler::FifoSerial);
    assert_eq!(config.serve.max_slots, 1);
    assert_eq!(config.serve.host, "0.0.0.0");
    assert_eq!(config.serve.port, 9090);
    assert_eq!(config.serve.kv_persist_budget.as_deref(), Some("24GiB"));
}

#[test]
fn cancelled_or_interrupted_interactive_setup_creates_nothing() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "never-created");
    let output = execute_fake(prompt_args(&root), true, "").unwrap();
    assert!(output.contains("Setup cancelled"));
    assert!(!root.exists());

    struct Interrupted;
    impl Read for Interrupted {
        fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::Interrupted))
        }
    }
    impl BufRead for Interrupted {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            Err(io::Error::from(io::ErrorKind::Interrupted))
        }
        fn consume(&mut self, _amount: usize) {}
        fn read_line(&mut self, _buffer: &mut String) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::Interrupted))
        }
    }
    let mut input = Interrupted;
    let mut output = Vec::new();
    let invocation = prompt_args(&root);
    execute(
        invocation.args,
        Some(&invocation.root),
        &FakeProbe {
            total: 500 * GIB,
            available: 200 * GIB,
        },
        true,
        &mut input,
        &mut output,
    )
    .unwrap();
    assert!(!root.exists());
}

#[test]
fn hostile_prompt_input_is_bounded_utf8_and_creates_nothing() {
    for mut input in [vec![b'x'; 257], vec![0xff, b'\n']] {
        let temp = TempDir::new().unwrap();
        let root = test_root(&temp, "state");
        let mut output = Vec::new();
        let invocation = prompt_args(&root);
        let error = execute(
            invocation.args,
            Some(&invocation.root),
            &FakeProbe {
                total: 500 * GIB,
                available: 200 * GIB,
            },
            true,
            &mut Cursor::new(&mut input),
            &mut output,
        )
        .unwrap_err();
        assert!(matches!(error, SetupError::Input(_)));
        assert!(!root.exists());
    }
}

#[test]
fn noninteractive_setup_requires_complete_choices_or_accept_defaults() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "state");
    assert!(execute_fake(prompt_args(&root), false, "").is_err());
    assert!(!root.exists());

    let mut explicit = prompt_args(&root);
    explicit.args.default_quant = Some("q5_k_m".to_owned());
    explicit.args.serve_host = Some("127.0.0.1".to_owned());
    explicit.args.serve_port = Some(9090);
    explicit.args.serve_scheduler = Some(SchedulerArg::InflightBatched);
    explicit.args.serve_max_slots = Some(2);
    explicit.args.serve_kv_persist_budget = Some("16GiB".to_owned());
    execute_fake(explicit, false, "").unwrap();
    let parsed = OperatorConfigV2::parse(&fs::read(root.join("config.toml")).unwrap()).unwrap();
    assert_eq!(parsed.convert.quant, "q5_k_m");
    assert_eq!(parsed.serve.port, 9090);
    assert_eq!(parsed.serve.max_slots, 2);
    assert_eq!(parsed.serve.kv_persist_budget.as_deref(), Some("16GiB"));

    let before = fs::read(root.join("config.toml")).unwrap();
    execute_fake(args(&root), false, "").unwrap();
    assert_eq!(fs::read(root.join("config.toml")).unwrap(), before);
}

#[test]
fn malformed_or_future_config_is_preserved_without_transaction_residue() {
    for body in [
        b"not = [valid".as_slice(),
        b"kind = \"hf2q.config\"\nschema_version = 2\n",
        b"kind = \"hf2q.config\"\nschema_version = 1\npackage = \"hf2q\"\n\n[session_cache]\nlimit_bytes = 0\n",
        b"kind = \"hf2q.config\"\nschema_version = 3\npackage = \"hf2q\"\n",
    ] {
        let temp = TempDir::new().unwrap();
        let root = test_root(&temp, "state");
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(root.join("config.toml"), body).unwrap();
        fs::set_permissions(root.join("config.toml"), fs::Permissions::from_mode(0o600)).unwrap();
        let before = fs::read(root.join("config.toml")).unwrap();
        let metadata = fs::metadata(root.join("config.toml")).unwrap();
        assert!(execute_fake(args(&root), false, "").is_err());
        assert_eq!(fs::read(root.join("config.toml")).unwrap(), before);
        assert_eq!(
            fs::metadata(root.join("config.toml")).unwrap().ino(),
            metadata.ino()
        );
        assert!(!root.join(".config.toml.lock").exists());
        assert!(!root.join(".config.toml.partial").exists());
        assert!(!root.join("cache").exists());
    }
}

#[test]
fn private_partial_resumes_and_conflicting_private_partial_is_reconstructed() {
    for (name, partial) in [
        (
            "prefix",
            include_bytes!("testdata/config_v2.toml")[..37].to_vec(),
        ),
        ("conflict", b"unrelated private residue".to_vec()),
    ] {
        let temp = TempDir::new().unwrap();
        let root = test_root(&temp, name);
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(root.join(".config.toml.partial"), partial).unwrap();
        fs::set_permissions(
            root.join(".config.toml.partial"),
            fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        execute_fake(args(&root), false, "").unwrap();
        assert_eq!(
            fs::read(root.join("config.toml")).unwrap(),
            include_bytes!("testdata/config_v2.toml")
        );
        assert!(!root.join(".config.toml.partial").exists());
    }
}

#[test]
fn lock_contention_is_busy_and_mints_no_config() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "state");
    let lock = setup_fs::hold_setup_lock(&root).unwrap();
    let error = execute_fake(args(&root), false, "").unwrap_err();
    assert!(matches!(error, SetupError::Busy));
    assert!(!root.join("config.toml").exists());
    drop(lock);
    execute_fake(args(&root), false, "").unwrap();
}

#[test]
fn exact_read_rejects_leaf_replacement_after_reading_the_open_inode() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "state");
    execute_fake(args(&root), false, "").unwrap();
    let bytes = fs::read(root.join("config.toml")).unwrap();
    let error = setup_fs::read_existing_config_with_test_hook(&root, || {
        fs::rename(root.join("config.toml"), root.join("retained-evidence")).unwrap();
        fs::write(root.join("config.toml"), &bytes).unwrap();
        fs::set_permissions(root.join("config.toml"), fs::Permissions::from_mode(0o600)).unwrap();
    })
    .unwrap_err();
    assert!(matches!(error, SetupError::Filesystem(_)));
    assert!(root.join("retained-evidence").exists());
}

#[test]
fn state_root_replacement_before_commit_cannot_redirect_config_publication() {
    for replacement_barrier in [
        setup_fs::SetupBarrier::LockAcquired,
        setup_fs::SetupBarrier::BeforeRename,
    ] {
        let temp = TempDir::new().unwrap();
        let root = test_root(&temp, "state");
        let detached = test_root(&temp, "detached");
        let (config, bytes) = fixture_config(0);
        let mut replaced = false;
        let error = setup_fs::persist_with_test_hook(&root, &config, &bytes, |barrier| {
            if barrier == replacement_barrier && !replaced {
                fs::rename(&root, &detached).unwrap();
                fs::create_dir(&root).unwrap();
                fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
                replaced = true;
            }
            Ok(())
        })
        .unwrap_err();
        assert!(matches!(error, SetupError::Filesystem(_)));
        assert!(!root.join("config.toml").exists());
        assert!(detached.join(".config.toml.lock").exists());
        assert!(!detached.join("config.toml").exists());
    }
}

#[test]
fn partial_leaf_replacement_before_rename_is_rejected_without_publication() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "state");
    let retained = test_root(&temp, "retained-partial");
    let (config, bytes) = fixture_config(0);
    let error = setup_fs::persist_with_test_hook(&root, &config, &bytes, |barrier| {
        if barrier == setup_fs::SetupBarrier::BeforeRename {
            fs::rename(root.join(".config.toml.partial"), &retained).unwrap();
            fs::write(root.join(".config.toml.partial"), &bytes).unwrap();
            fs::set_permissions(
                root.join(".config.toml.partial"),
                fs::Permissions::from_mode(0o600),
            )
            .unwrap();
        }
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, SetupError::Filesystem(_)));
    assert!(!root.join("config.toml").exists());
    assert!(retained.exists());
}

#[test]
fn config_lock_and_partial_leaf_replacements_never_return_success() {
    use setup_fs::SetupBarrier::*;

    for barrier in [PartialSynced, BeforeRename] {
        let temp = TempDir::new().unwrap();
        let root = test_root(&temp, barrier.as_str());
        execute_fake(args(&root), false, "").unwrap();
        let old = fs::read(root.join("config.toml")).unwrap();
        let retained = root.join("retained-config");
        let (config, bytes) = fixture_config(32 * GIB);
        let error = setup_fs::persist_with_test_hook(&root, &config, &bytes, |seen| {
            if seen == barrier {
                fs::rename(root.join("config.toml"), &retained).unwrap();
                fs::write(root.join("config.toml"), &old).unwrap();
                fs::set_permissions(root.join("config.toml"), fs::Permissions::from_mode(0o600))
                    .unwrap();
            }
            Ok(())
        })
        .unwrap_err();
        assert!(matches!(error, SetupError::Filesystem(_)));
        assert_eq!(fs::read(root.join("config.toml")).unwrap(), old);
        assert!(retained.exists());
    }

    for replacement_barrier in [LockAcquired, PartialPrefixVerified] {
        let temp = TempDir::new().unwrap();
        let root = test_root(&temp, "lock");
        let retained = test_root(&temp, "retained-lock");
        let (config, bytes) = fixture_config(0);
        let error = setup_fs::persist_with_test_hook(&root, &config, &bytes, |barrier| {
            if barrier == replacement_barrier {
                fs::rename(root.join(".config.toml.lock"), &retained).unwrap();
                fs::write(root.join(".config.toml.lock"), b"").unwrap();
                fs::set_permissions(
                    root.join(".config.toml.lock"),
                    fs::Permissions::from_mode(0o600),
                )
                .unwrap();
            }
            Ok(())
        })
        .unwrap_err();
        assert!(matches!(error, SetupError::Filesystem(_)));
        assert!(retained.exists());
        assert!(!root.join("config.toml").exists());
    }

    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "idempotent-config");
    execute_fake(args(&root), false, "").unwrap();
    let (config, bytes) = fixture_config(0);
    let retained = test_root(&temp, "idempotent-retained");
    let error = setup_fs::persist_with_test_hook(&root, &config, &bytes, |barrier| {
        if barrier == EndpointFullSynced {
            fs::rename(root.join("config.toml"), &retained).unwrap();
            fs::write(root.join("config.toml"), &bytes).unwrap();
            fs::set_permissions(root.join("config.toml"), fs::Permissions::from_mode(0o600))
                .unwrap();
        }
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, SetupError::Filesystem(_)));
    assert!(retained.exists());
}

#[test]
fn stale_prompt_snapshot_and_partial_prefix_replacement_fail_closed() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "stale-prompt");
    execute_fake(args(&root), false, "").unwrap();
    let observed = setup_fs::observe_existing_config(&root).unwrap();
    let (changed, changed_bytes) = fixture_config(32 * GIB);
    fs::write(root.join("config.toml"), &changed_bytes).unwrap();
    fs::set_permissions(root.join("config.toml"), fs::Permissions::from_mode(0o600)).unwrap();
    let (requested, requested_bytes) = fixture_config(0);
    assert!(setup_fs::persist_observed_with_test_hook(
        &root,
        &requested,
        &requested_bytes,
        &observed,
        |_| Ok(())
    )
    .is_err());
    assert_eq!(fs::read(root.join("config.toml")).unwrap(), changed_bytes);
    assert_eq!(changed.convert.quant, "q5_k_m");

    let root = test_root(&temp, "replaced-after-prompt");
    execute_fake(args(&root), false, "").unwrap();
    let observed = setup_fs::observe_existing_config(&root).unwrap();
    let retained_root = test_root(&temp, "retained-prompt-root");
    fs::rename(&root, &retained_root).unwrap();
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    let (requested, requested_bytes) = fixture_config(0);
    assert!(setup_fs::persist_observed_with_test_hook(
        &root,
        &requested,
        &requested_bytes,
        &observed,
        |_| Ok(())
    )
    .is_err());
    assert!(!root.join("config.toml").exists());
    assert!(retained_root.join("config.toml").exists());

    let root = test_root(&temp, "disappeared-after-prompt");
    execute_fake(args(&root), false, "").unwrap();
    let observed = setup_fs::observe_existing_config(&root).unwrap();
    let retained_root = test_root(&temp, "retained-disappeared-root");
    fs::rename(&root, &retained_root).unwrap();
    assert!(setup_fs::persist_observed_with_test_hook(
        &root,
        &requested,
        &requested_bytes,
        &observed,
        |_| Ok(())
    )
    .is_err());
    assert!(!root.exists());
    assert!(retained_root.join("config.toml").exists());

    let root = test_root(&temp, "appeared-after-prompt");
    let observed = setup_fs::observe_existing_config(&root).unwrap();
    fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    assert!(setup_fs::persist_observed_with_test_hook(
        &root,
        &requested,
        &requested_bytes,
        &observed,
        |_| Ok(())
    )
    .is_err());
    assert!(!root.join("config.toml").exists());

    let root = test_root(&temp, "partial-prefix");
    let retained = test_root(&temp, "retained-prefix");
    let (config, bytes) = fixture_config(0);
    let error = setup_fs::persist_with_test_hook(&root, &config, &bytes, |barrier| {
        if barrier == setup_fs::SetupBarrier::PartialPrefixVerified {
            let measured = fs::read(root.join(".config.toml.partial")).unwrap();
            fs::rename(root.join(".config.toml.partial"), &retained).unwrap();
            fs::write(root.join(".config.toml.partial"), measured).unwrap();
            fs::set_permissions(
                root.join(".config.toml.partial"),
                fs::Permissions::from_mode(0o600),
            )
            .unwrap();
        }
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, SetupError::Filesystem(_)));
    assert!(retained.exists());
    assert!(!root.join("config.toml").exists());
}

#[test]
fn precommit_faults_are_retryable_and_postcommit_faults_are_typed_unknown() {
    use setup_fs::SetupBarrier::*;

    for barrier in [
        RootOpened,
        LockAcquired,
        PartialPrefixVerified,
        PartialSynced,
        BeforeRename,
        ConfigRenamed,
        RootSynced,
        ConfigFullSynced,
        EndpointFullSynced,
    ] {
        let temp = TempDir::new().unwrap();
        let root = test_root(&temp, barrier.as_str());
        let (config, bytes) = fixture_config(0);
        let error = setup_fs::persist_with_test_hook(&root, &config, &bytes, |seen| {
            if seen == barrier {
                Err(SetupError::Filesystem(format!("injected at {barrier:?}")))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        if matches!(
            barrier,
            ConfigRenamed | RootSynced | ConfigFullSynced | EndpointFullSynced
        ) {
            assert!(matches!(error, SetupError::DurabilityUnknown(_)));
        } else {
            assert!(matches!(error, SetupError::Filesystem(_)));
        }
        execute_fake(args(&root), false, "").unwrap();
        assert_eq!(fs::read(root.join("config.toml")).unwrap(), bytes);
        assert!(!root.join(".config.toml.partial").exists());
    }
}

#[test]
fn sigabrt_at_every_publication_barrier_recovers_in_a_fresh_process() {
    const CHILD: &str = "HF2Q_SETUP_CRASH_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let root = std::env::var_os("HF2Q_SETUP_CRASH_ROOT").unwrap();
        execute_fake(args(Path::new(&root)), false, "").unwrap();
        return;
    }

    use setup_fs::SetupBarrier::*;
    for barrier in [
        RootOpened,
        LockAcquired,
        PartialPrefixVerified,
        PartialSynced,
        BeforeRename,
        ConfigRenamed,
        RootSynced,
        ConfigFullSynced,
        EndpointFullSynced,
    ] {
        let temp = TempDir::new().unwrap();
        let root = test_root(&temp, barrier.as_str());
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("setup::tests::sigabrt_at_every_publication_barrier_recovers_in_a_fresh_process")
            .arg("--exact")
            .env(CHILD, "1")
            .env("HF2Q_SETUP_CRASH_ROOT", &root)
            .env("HF2Q_SETUP_ABORT_AT", barrier.as_str())
            .status()
            .unwrap();
        assert_eq!(status.signal(), Some(libc::SIGABRT), "barrier {barrier:?}");
        execute_fake(args(&root), false, "").unwrap();
        assert_eq!(
            OperatorConfigV2::parse(&fs::read(root.join("config.toml")).unwrap()).unwrap(),
            OperatorConfigV2::guide_defaults().unwrap()
        );
        assert!(!root.join(".config.toml.partial").exists());
        for name in ["update", "versions", "activations", "current"] {
            assert!(!root.join(name).exists());
        }
    }
}

#[test]
fn hostile_root_config_lock_and_partial_nodes_fail_closed() {
    let temp = TempDir::new().unwrap();
    let parent = temp.path().canonicalize().unwrap();

    let wrong_mode = parent.join("wrong-mode");
    fs::create_dir(&wrong_mode).unwrap();
    fs::set_permissions(&wrong_mode, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(execute_fake(args(&wrong_mode), false, "").is_err());

    let special_mode = parent.join("special-mode");
    fs::create_dir(&special_mode).unwrap();
    fs::set_permissions(&special_mode, fs::Permissions::from_mode(0o1700)).unwrap();
    assert!(execute_fake(args(&special_mode), false, "").is_err());

    let real = parent.join("real");
    fs::create_dir(&real).unwrap();
    fs::set_permissions(&real, fs::Permissions::from_mode(0o700)).unwrap();
    let symlink_root = parent.join("symlink-root");
    std::os::unix::fs::symlink(&real, &symlink_root).unwrap();
    assert!(execute_fake(args(&symlink_root), false, "").is_err());

    for name in ["config.toml", ".config.toml.lock", ".config.toml.partial"] {
        let root = parent.join(name.replace('.', "-"));
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
        let target = parent.join(format!("target-{}", name.replace('.', "-")));
        fs::write(&target, b"evidence").unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o600)).unwrap();
        std::os::unix::fs::symlink(&target, root.join(name)).unwrap();
        assert!(execute_fake(args(&root), false, "").is_err());
        assert_eq!(fs::read(&target).unwrap(), b"evidence");
    }

    let hardlink_root = parent.join("hardlink");
    fs::create_dir(&hardlink_root).unwrap();
    fs::set_permissions(&hardlink_root, fs::Permissions::from_mode(0o700)).unwrap();
    let (_, bytes) = fixture_config(0);
    fs::write(hardlink_root.join("config.toml"), bytes).unwrap();
    fs::set_permissions(
        hardlink_root.join("config.toml"),
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    fs::hard_link(
        hardlink_root.join("config.toml"),
        parent.join("hardlink-evidence"),
    )
    .unwrap();
    assert!(execute_fake(args(&hardlink_root), false, "").is_err());
    assert_eq!(
        fs::metadata(parent.join("hardlink-evidence"))
            .unwrap()
            .nlink(),
        2
    );

    let special_file_root = parent.join("special-file");
    fs::create_dir(&special_file_root).unwrap();
    fs::set_permissions(&special_file_root, fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(
        special_file_root.join("config.toml"),
        include_bytes!("testdata/config_v2.toml"),
    )
    .unwrap();
    fs::set_permissions(
        special_file_root.join("config.toml"),
        fs::Permissions::from_mode(0o4600),
    )
    .unwrap();
    assert!(execute_fake(args(&special_file_root), false, "").is_err());

    let nonempty_lock_root = parent.join("nonempty-lock");
    fs::create_dir(&nonempty_lock_root).unwrap();
    fs::set_permissions(&nonempty_lock_root, fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(nonempty_lock_root.join(".config.toml.lock"), b"not empty").unwrap();
    fs::set_permissions(
        nonempty_lock_root.join(".config.toml.lock"),
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    assert!(execute_fake(args(&nonempty_lock_root), false, "").is_err());
}

#[test]
fn root_authority_rejects_relative_root_missing_ancestor_and_fifo_config() {
    let relative = Path::new("relative-state");
    assert!(execute_fake(args(relative), false, "").is_err());

    let temp = TempDir::new().unwrap();
    let parent = temp.path().canonicalize().unwrap();
    let missing_ancestor = parent.join("missing/state");
    assert!(execute_fake(args(&missing_ancestor), false, "").is_err());
    assert!(!parent.join("missing").exists());

    for name in ["config.toml", ".config.toml.lock", ".config.toml.partial"] {
        let root = parent.join(format!("fifo-{}", name.replace('.', "-")));
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
        let path = std::ffi::CString::new(root.join(name).as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        assert!(execute_fake(args(&root), false, "").is_err());
        assert!(fs::symlink_metadata(root.join(name))
            .unwrap()
            .file_type()
            .is_fifo());
    }
}

#[test]
fn restrictive_umask_cannot_change_owned_file_or_directory_modes() {
    const CHILD: &str = "HF2Q_SETUP_UMASK_CHILD";
    if std::env::var_os(CHILD).is_some() {
        unsafe { libc::umask(0o777) };
        let root = std::env::var_os("HF2Q_SETUP_UMASK_ROOT").unwrap();
        execute_fake(args(Path::new(&root)), false, "").unwrap();
        return;
    }
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "state");
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("setup::tests::restrictive_umask_cannot_change_owned_file_or_directory_modes")
        .arg("--exact")
        .env(CHILD, "1")
        .env("HF2Q_SETUP_UMASK_ROOT", &root)
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(fs::metadata(&root).unwrap().mode() & 0o777, 0o700);
    for name in ["config.toml", ".config.toml.lock"] {
        assert_eq!(fs::metadata(root.join(name)).unwrap().mode() & 0o777, 0o600);
    }
}
