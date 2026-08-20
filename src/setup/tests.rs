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
use super::policy::{parse_byte_size, recommended_limit};
use super::runtime_policy::{authorize_session_cache_policy, SessionCachePolicyAuthorization};
use super::schema::{
    ConfigV1, ConfiguredShell, HardwareProfileV1, SessionCachePolicyV1, MAX_CONFIG_BYTES,
};
use super::{execute, SetupError};
use crate::cli::{Cli, Command, SessionCacheChoice, SetupArgs};

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
            performance_level0_name: "Super".to_owned(),
            performance_level0_cores: 4,
            performance_level1_name: "Performance".to_owned(),
            performance_level1_cores: 12,
            open_file_soft_limit: 10240,
            volume_total_bytes: self.total,
            volume_available_bytes: self.available,
        })
    }
}

fn fixture_hardware() -> HardwareProfileV1 {
    HardwareProfileV1 {
        target: "aarch64-apple-darwin".to_owned(),
        chip_model: "Apple M5 Max".to_owned(),
        unified_memory_bytes: 128 * GIB,
        metal_device_name: "Apple M5 Max".to_owned(),
        metal_recommended_working_set_bytes: 96 * GIB,
    }
}

fn fixture_config(limit_bytes: u64) -> (ConfigV1, Vec<u8>) {
    let config = ConfigV1::new(fixture_hardware(), SessionCachePolicyV1 { limit_bytes }).unwrap();
    let bytes = config.to_canonical_bytes().unwrap();
    (config, bytes)
}

fn args(root: &Path, cache: Option<SessionCacheChoice>, limit: Option<&str>) -> SetupArgs {
    SetupArgs {
        session_cache: cache,
        session_cache_limit: limit.map(str::to_owned),
        state_root: Some(root.to_owned()),
    }
}

fn test_root(temp: &TempDir, name: &str) -> PathBuf {
    temp.path().canonicalize().unwrap().join(name)
}

fn execute_with_probe(
    args: SetupArgs,
    interactive: bool,
    input: &str,
    total: u64,
    available: u64,
) -> Result<String, SetupError> {
    let mut input = Cursor::new(input.as_bytes());
    let mut output = Vec::new();
    execute(
        args,
        &FakeProbe { total, available },
        interactive,
        &mut input,
        &mut output,
    )?;
    Ok(String::from_utf8(output).expect("setup output is UTF-8"))
}

fn execute_fake(args: SetupArgs, interactive: bool, input: &str) -> Result<String, SetupError> {
    execute_with_probe(args, interactive, input, 500 * GIB, 200 * GIB)
}

#[test]
fn cli_parses_the_closed_noninteractive_surface() {
    let raw = [
        "hf2q",
        "setup",
        "--state-root",
        "/tmp/hf2q-state",
        "--session-cache",
        "on",
        "--session-cache-limit",
        "32GiB",
    ];
    let cli = Cli::try_parse_from(raw).unwrap();
    assert!(crate::invocation_mentions_setup(
        &raw.into_iter().map(Into::into).collect::<Vec<_>>()
    ));
    let Command::Setup(args) = cli.command else {
        panic!("setup command was not selected");
    };
    assert_eq!(args.state_root.unwrap(), Path::new("/tmp/hf2q-state"));
    assert_eq!(args.session_cache, Some(SessionCacheChoice::On));
    assert_eq!(args.session_cache_limit.as_deref(), Some("32GiB"));
    assert!(crate::invocation_mentions_setup(&[
        "hf2q".into(),
        "setup".into(),
        "--help".into(),
    ]));
    assert!(crate::invocation_mentions_setup(&[
        "hf2q".into(),
        "setup".into(),
        "--not-a-real-flag".into(),
    ]));
    assert!(crate::invocation_mentions_setup(&[
        "hf2q".into(),
        "--log-format".into(),
        "json".into(),
        "-vv".into(),
        "setup".into(),
    ]));
    assert!(!crate::invocation_mentions_setup(&[
        "hf2q".into(),
        "info".into(),
        "setup".into(),
    ]));
    assert!(!crate::invocation_mentions_setup(&[
        "hf2q".into(),
        "--log-level".into(),
        "setup".into(),
    ]));
}

#[test]
fn recommendation_uses_every_disk_bound_with_checked_flooring() {
    assert_eq!(recommended_limit(500 * GIB, 200 * GIB).unwrap(), 50 * GIB);
    assert_eq!(
        recommended_limit(2_000 * GIB, 1_000 * GIB).unwrap(),
        100 * GIB
    );
    assert_eq!(recommended_limit(500 * GIB, 40 * GIB).unwrap(), 0);
    assert_eq!(recommended_limit(100 * GIB, 50 * GIB).unwrap(), 10 * GIB);
    assert!(recommended_limit(10, 11).is_err());
}

#[test]
fn byte_size_parser_is_closed_and_overflow_safe() {
    assert_eq!(parse_byte_size("1KiB").unwrap(), 1024);
    assert_eq!(parse_byte_size("32GiB").unwrap(), 32 * GIB);
    assert_eq!(parse_byte_size("7B").unwrap(), 7);
    assert_eq!(parse_byte_size("0").unwrap(), 0);
    for invalid in [
        "",
        "01GiB",
        "1GB",
        "1 GiB",
        "-1",
        "+1",
        "1.5GiB",
        "1gib",
        "18446744073709551615TiB",
    ] {
        assert!(parse_byte_size(invalid).is_err(), "accepted {invalid:?}");
    }
}

#[test]
fn config_v1_golden_is_exact_and_hostile_input_is_rejected() {
    let config = ConfigV1::new(
        fixture_hardware(),
        SessionCachePolicyV1 {
            limit_bytes: 32 * GIB,
        },
    )
    .unwrap();
    let bytes = config.to_canonical_bytes().unwrap();
    assert_eq!(bytes, include_bytes!("testdata/config_v1.toml"));
    assert_eq!(ConfigV1::parse(&bytes).unwrap(), config);

    let text = String::from_utf8(bytes).unwrap();
    for hostile in [
        text.replace("schema_version = 1", "schema_version = 2"),
        text.replace("package = \"hf2q\"", "package = \"other\""),
        text.replace("limit_bytes = 34359738368", "limit_bytes = -1"),
        text.replace(
            "target = \"aarch64-apple-darwin\"",
            "target = \"x86_64-apple-darwin\"",
        ),
        text.replace(
            "chip_model = \"Apple M5 Max\"",
            "chip_model = \"bad\\nmodel\"",
        ),
        format!("{text}unknown = true\n"),
        text.replacen(
            "schema_version = 1",
            "schema_version = 1\nschema_version = 1",
            1,
        ),
    ] {
        assert!(ConfigV1::parse(hostile.as_bytes()).is_err());
    }
    let padding_len = MAX_CONFIG_BYTES - text.len() - 2;
    let mut exact_cap = text.clone();
    exact_cap.push('#');
    exact_cap.push_str(&"x".repeat(padding_len));
    exact_cap.push('\n');
    assert_eq!(exact_cap.len(), MAX_CONFIG_BYTES);
    assert_eq!(ConfigV1::parse(exact_cap.as_bytes()).unwrap(), config);
    assert!(ConfigV1::parse(&vec![b'a'; MAX_CONFIG_BYTES + 1]).is_err());
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
    let output = execute_fake(
        args(&root, Some(SessionCacheChoice::On), Some("32GiB")),
        false,
        "",
    )
    .unwrap();
    assert!(output.contains("Configured"));
    assert!(output.contains("No model was downloaded"));
    let before = fs::read(root.join("config.toml")).unwrap();
    let metadata = fs::metadata(root.join("config.toml")).unwrap();
    assert_eq!(metadata.mode() & 0o777, 0o600);
    assert_eq!(metadata.nlink(), 1);
    assert_eq!(fs::metadata(&root).unwrap().mode() & 0o777, 0o700);
    assert_eq!(
        fs::metadata(root.join("cache/sessions")).unwrap().mode() & 0o777,
        0o700
    );
    assert!(!root.join(".config.toml.partial").exists());
    assert!(!root.join("update").exists());

    let output = execute_fake(
        args(&root, Some(SessionCacheChoice::On), Some("32GiB")),
        false,
        "",
    )
    .unwrap();
    assert!(output.contains("Verified"));
    assert_eq!(fs::read(root.join("config.toml")).unwrap(), before);
    let after = fs::metadata(root.join("config.toml")).unwrap();
    assert_eq!(after.ino(), metadata.ino());
    assert_eq!(after.mtime(), metadata.mtime());
    assert_eq!(after.mtime_nsec(), metadata.mtime_nsec());
}

#[test]
fn interactive_defaults_follow_current_policy_and_canonicalize_valid_toml() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "state");
    execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
    let output = execute_fake(args(&root, None, None), true, "\n").unwrap();
    assert!(output.contains("[y/N]"));
    let config = ConfigV1::parse(&fs::read(root.join("config.toml")).unwrap()).unwrap();
    assert_eq!(config.session_cache.limit_bytes, 0);

    execute_fake(
        args(&root, Some(SessionCacheChoice::On), Some("32GiB")),
        false,
        "",
    )
    .unwrap();
    let mut noncanonical = b"# managed by hf2q\n".to_vec();
    noncanonical.extend(fs::read(root.join("config.toml")).unwrap());
    fs::write(root.join("config.toml"), noncanonical).unwrap();
    fs::set_permissions(root.join("config.toml"), fs::Permissions::from_mode(0o600)).unwrap();
    let output = execute_fake(args(&root, None, None), true, "\n\n").unwrap();
    assert!(output.contains("[Y/n]"));
    assert!(output.contains("Session cache limit [32 GiB]"));
    assert_eq!(
        fs::read(root.join("config.toml")).unwrap(),
        include_bytes!("testdata/config_v1.toml")
    );
}

#[test]
fn fresh_interactive_yes_uses_recommendation_and_no_records_disabled() {
    let temp = TempDir::new().unwrap();
    let enabled = test_root(&temp, "enabled");
    execute_fake(args(&enabled, None, None), true, "\n\n").unwrap();
    let config = ConfigV1::parse(&fs::read(enabled.join("config.toml")).unwrap()).unwrap();
    assert_eq!(config.session_cache.limit_bytes, 50 * GIB);

    let disabled = test_root(&temp, "disabled");
    execute_fake(args(&disabled, None, None), true, "n\n").unwrap();
    let config = ConfigV1::parse(&fs::read(disabled.join("config.toml")).unwrap()).unwrap();
    assert_eq!(config.session_cache.limit_bytes, 0);

    let explicit = test_root(&temp, "explicit");
    execute_fake(args(&explicit, None, None), true, "y\n32GiB\n").unwrap();
    let config = ConfigV1::parse(&fs::read(explicit.join("config.toml")).unwrap()).unwrap();
    assert_eq!(config.session_cache.limit_bytes, 32 * GIB);

    let whitespace = test_root(&temp, "whitespace");
    assert!(execute_fake(args(&whitespace, None, None), true, "y\n 32GiB\n").is_err());
    assert!(!whitespace.exists());
}

#[test]
fn cancelled_or_interrupted_interactive_setup_creates_nothing() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "never-created");
    let output = execute_fake(args(&root, None, None), true, "").unwrap();
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
    execute(
        args(&root, None, None),
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
    for mut input in [
        {
            let mut bytes = b"y\n".to_vec();
            bytes.extend(vec![b'x'; 257]);
            bytes
        },
        vec![b'y', b'\n', 0xff, b'\n'],
    ] {
        let temp = TempDir::new().unwrap();
        let root = test_root(&temp, "state");
        let mut output = Vec::new();
        let error = execute(
            args(&root, None, None),
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
fn noninteractive_policy_never_guesses_and_zero_is_disabled_only() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "state");
    assert!(execute_fake(args(&root, None, None), false, "").is_err());
    assert!(execute_fake(args(&root, Some(SessionCacheChoice::On), None), false, "").is_err());
    assert!(execute_fake(
        args(&root, Some(SessionCacheChoice::On), Some("0")),
        false,
        ""
    )
    .is_err());
    assert!(execute_fake(
        args(&root, Some(SessionCacheChoice::Off), Some("1GiB")),
        false,
        ""
    )
    .is_err());

    execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
    let parsed = ConfigV1::parse(&fs::read(root.join("config.toml")).unwrap()).unwrap();
    assert_eq!(parsed.session_cache.limit_bytes, 0);
}

#[test]
fn overrides_warn_and_zero_safe_band_rejects_enabling_without_mutation() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "state");
    let output = execute_fake(
        args(&root, Some(SessionCacheChoice::On), Some("60GiB")),
        false,
        "",
    )
    .unwrap();
    assert!(output.contains("Warning"));
    let before = fs::read(root.join("config.toml")).unwrap();
    assert!(execute_with_probe(
        args(&root, Some(SessionCacheChoice::On), Some("1GiB")),
        false,
        "",
        500 * GIB,
        40 * GIB,
    )
    .is_err());
    assert_eq!(fs::read(root.join("config.toml")).unwrap(), before);
    execute_with_probe(
        args(&root, Some(SessionCacheChoice::Off), None),
        false,
        "",
        500 * GIB,
        40 * GIB,
    )
    .unwrap();
}

#[test]
fn malformed_or_future_config_is_preserved_without_transaction_residue() {
    for body in [
        b"not = [valid".as_slice(),
        b"kind = \"hf2q.config\"\nschema_version = 2\n",
    ] {
        let temp = TempDir::new().unwrap();
        let root = test_root(&temp, "state");
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(root.join("config.toml"), body).unwrap();
        fs::set_permissions(root.join("config.toml"), fs::Permissions::from_mode(0o600)).unwrap();
        let before = fs::read(root.join("config.toml")).unwrap();
        let metadata = fs::metadata(root.join("config.toml")).unwrap();
        assert!(execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").is_err());
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
fn config_only_root_remains_preidentity_and_bad_update_state_is_rejected() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "config-only");
    execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
    crate::distribution::verify_setup_state_root(&root).unwrap();
    assert!(!root.join("update").exists());

    let hostile = test_root(&temp, "hostile");
    fs::create_dir(&hostile).unwrap();
    fs::set_permissions(&hostile, fs::Permissions::from_mode(0o700)).unwrap();
    fs::create_dir(hostile.join("update")).unwrap();
    fs::set_permissions(hostile.join("update"), fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(hostile.join("update/unexpected"), b"evidence").unwrap();
    assert!(execute_fake(
        args(&hostile, Some(SessionCacheChoice::Off), None),
        false,
        ""
    )
    .is_err());
    assert!(!hostile.join("config.toml").exists());
}

#[test]
fn matching_existing_installation_identity_is_read_only_and_accepted() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "identified");
    crate::distribution::bootstrap_setup_test_identity(&root).unwrap();
    let identity_path = root.join("update/installation-identity.json");
    let before = fs::read(&identity_path).unwrap();
    let metadata = fs::metadata(&identity_path).unwrap();
    execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
    assert_eq!(fs::read(&identity_path).unwrap(), before);
    let after = fs::metadata(&identity_path).unwrap();
    assert_eq!(after.ino(), metadata.ino());
    assert_eq!(after.mtime(), metadata.mtime());
}

#[test]
fn installation_identity_appearance_or_replacement_during_setup_is_rejected() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "identified");
    let alternate = test_root(&temp, "alternate");
    let retained = test_root(&temp, "retained-update");
    crate::distribution::bootstrap_setup_test_identity(&root).unwrap();
    crate::distribution::bootstrap_setup_test_identity(&alternate).unwrap();
    let (config, bytes) = fixture_config(0);
    let error = setup_fs::persist_with_test_hook(&root, &config, &bytes, |barrier| {
        if barrier == setup_fs::SetupBarrier::PartialSynced {
            fs::rename(root.join("update"), &retained).unwrap();
            fs::rename(alternate.join("update"), root.join("update")).unwrap();
        }
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, SetupError::Filesystem(_)));
    assert!(retained.join("installation-identity.json").exists());
    assert!(!root.join("config.toml").exists());

    let root = test_root(&temp, "identity-appears");
    let error = setup_fs::persist_with_test_hook(&root, &config, &bytes, |barrier| {
        if barrier == setup_fs::SetupBarrier::PartialSynced {
            crate::distribution::bootstrap_setup_test_identity(&root).unwrap();
        }
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, SetupError::Filesystem(_)));
    assert!(root.join("update/installation-identity.json").exists());
    assert!(!root.join("config.toml").exists());

    let root = test_root(&temp, "identified-idempotent");
    let alternate = test_root(&temp, "alternate-idempotent");
    let retained = test_root(&temp, "retained-idempotent-update");
    crate::distribution::bootstrap_setup_test_identity(&root).unwrap();
    crate::distribution::bootstrap_setup_test_identity(&alternate).unwrap();
    execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
    let (config, bytes) = fixture_config(0);
    let error = setup_fs::persist_with_test_hook(&root, &config, &bytes, |barrier| {
        if barrier == setup_fs::SetupBarrier::SessionDirectoriesSynced {
            fs::rename(root.join("update"), &retained).unwrap();
            fs::rename(alternate.join("update"), root.join("update")).unwrap();
        }
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, SetupError::Filesystem(_)));
    assert!(retained.join("installation-identity.json").exists());
}

#[test]
fn private_partial_resumes_and_conflicting_private_partial_is_reconstructed() {
    for (name, partial) in [
        (
            "prefix",
            include_bytes!("testdata/config_v1.toml")[..37].to_vec(),
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
        execute_fake(
            args(&root, Some(SessionCacheChoice::On), Some("32GiB")),
            false,
            "",
        )
        .unwrap();
        assert_eq!(
            fs::read(root.join("config.toml")).unwrap(),
            include_bytes!("testdata/config_v1.toml")
        );
        assert!(!root.join(".config.toml.partial").exists());
    }
}

#[test]
fn lock_contention_is_busy_and_mints_no_config() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "state");
    let lock = setup_fs::hold_setup_lock(&root).unwrap();
    let error =
        execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap_err();
    assert!(matches!(error, SetupError::Busy));
    assert!(!root.join("config.toml").exists());
    drop(lock);
    execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
}

#[test]
fn exact_read_rejects_leaf_replacement_after_reading_the_open_inode() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "state");
    execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
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
fn config_lock_partial_and_session_leaf_replacements_never_return_success() {
    use setup_fs::SetupBarrier::*;

    for barrier in [PartialSynced, BeforeRename] {
        let temp = TempDir::new().unwrap();
        let root = test_root(&temp, barrier.as_str());
        execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
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
    let root = test_root(&temp, "session-directory");
    let detached = test_root(&temp, "detached-cache");
    let (config, bytes) = fixture_config(0);
    let error = setup_fs::persist_with_test_hook(&root, &config, &bytes, |barrier| {
        if barrier == SessionDirectoriesSynced {
            fs::rename(root.join("cache"), &detached).unwrap();
            fs::create_dir(root.join("cache")).unwrap();
            fs::set_permissions(root.join("cache"), fs::Permissions::from_mode(0o700)).unwrap();
            fs::create_dir(root.join("cache/sessions")).unwrap();
            fs::set_permissions(
                root.join("cache/sessions"),
                fs::Permissions::from_mode(0o700),
            )
            .unwrap();
        }
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(error, SetupError::DurabilityUnknown(_)));
    assert!(detached.join("sessions").is_dir());

    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "idempotent-config");
    execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
    let (config, bytes) = fixture_config(0);
    let retained = test_root(&temp, "idempotent-retained");
    let error = setup_fs::persist_with_test_hook(&root, &config, &bytes, |barrier| {
        if barrier == SessionDirectoriesSynced {
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
    execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
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
    assert_eq!(changed.session_cache.limit_bytes, 32 * GIB);

    let root = test_root(&temp, "replaced-after-prompt");
    execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
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
    execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
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
        SessionDirectoriesSynced,
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
            ConfigRenamed
                | RootSynced
                | ConfigFullSynced
                | SessionDirectoriesSynced
                | EndpointFullSynced
        ) {
            assert!(matches!(error, SetupError::DurabilityUnknown(_)));
        } else {
            assert!(matches!(error, SetupError::Filesystem(_)));
        }
        execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
        assert_eq!(fs::read(root.join("config.toml")).unwrap(), bytes);
        assert!(!root.join(".config.toml.partial").exists());
    }
}

#[test]
fn sigabrt_at_every_publication_barrier_recovers_in_a_fresh_process() {
    const CHILD: &str = "HF2Q_SETUP_CRASH_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let root = std::env::var_os("HF2Q_SETUP_CRASH_ROOT").unwrap();
        execute_fake(
            args(Path::new(&root), Some(SessionCacheChoice::Off), None),
            false,
            "",
        )
        .unwrap();
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
        SessionDirectoriesSynced,
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
        execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
        assert_eq!(
            ConfigV1::parse(&fs::read(root.join("config.toml")).unwrap())
                .unwrap()
                .session_cache
                .limit_bytes,
            0
        );
        assert!(!root.join(".config.toml.partial").exists());
        for name in ["update", "versions", "activations", "current"] {
            assert!(!root.join(name).exists());
        }
    }
}

#[test]
fn hostile_root_config_lock_partial_and_cache_nodes_fail_closed() {
    let temp = TempDir::new().unwrap();
    let parent = temp.path().canonicalize().unwrap();

    let wrong_mode = parent.join("wrong-mode");
    fs::create_dir(&wrong_mode).unwrap();
    fs::set_permissions(&wrong_mode, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(execute_fake(
        args(&wrong_mode, Some(SessionCacheChoice::Off), None),
        false,
        ""
    )
    .is_err());

    let special_mode = parent.join("special-mode");
    fs::create_dir(&special_mode).unwrap();
    fs::set_permissions(&special_mode, fs::Permissions::from_mode(0o1700)).unwrap();
    assert!(execute_fake(
        args(&special_mode, Some(SessionCacheChoice::Off), None),
        false,
        ""
    )
    .is_err());

    let real = parent.join("real");
    fs::create_dir(&real).unwrap();
    fs::set_permissions(&real, fs::Permissions::from_mode(0o700)).unwrap();
    let symlink_root = parent.join("symlink-root");
    std::os::unix::fs::symlink(&real, &symlink_root).unwrap();
    assert!(execute_fake(
        args(&symlink_root, Some(SessionCacheChoice::Off), None),
        false,
        ""
    )
    .is_err());

    for name in ["config.toml", ".config.toml.lock", ".config.toml.partial"] {
        let root = parent.join(name.replace('.', "-"));
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
        let target = parent.join(format!("target-{}", name.replace('.', "-")));
        fs::write(&target, b"evidence").unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o600)).unwrap();
        std::os::unix::fs::symlink(&target, root.join(name)).unwrap();
        assert!(execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").is_err());
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
    assert!(execute_fake(
        args(&hardlink_root, Some(SessionCacheChoice::Off), None),
        false,
        ""
    )
    .is_err());
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
        include_bytes!("testdata/config_v1.toml"),
    )
    .unwrap();
    fs::set_permissions(
        special_file_root.join("config.toml"),
        fs::Permissions::from_mode(0o4600),
    )
    .unwrap();
    assert!(execute_fake(
        args(&special_file_root, Some(SessionCacheChoice::Off), None),
        false,
        ""
    )
    .is_err());

    let nonempty_lock_root = parent.join("nonempty-lock");
    fs::create_dir(&nonempty_lock_root).unwrap();
    fs::set_permissions(&nonempty_lock_root, fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(nonempty_lock_root.join(".config.toml.lock"), b"not empty").unwrap();
    fs::set_permissions(
        nonempty_lock_root.join(".config.toml.lock"),
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    assert!(execute_fake(
        args(&nonempty_lock_root, Some(SessionCacheChoice::Off), None),
        false,
        ""
    )
    .is_err());

    let cache_root = parent.join("cache-symlink");
    execute_fake(
        args(&cache_root, Some(SessionCacheChoice::Off), None),
        false,
        "",
    )
    .unwrap();
    fs::remove_dir(cache_root.join("cache/sessions")).unwrap();
    fs::remove_dir(cache_root.join("cache")).unwrap();
    std::os::unix::fs::symlink(&real, cache_root.join("cache")).unwrap();
    assert!(execute_fake(
        args(&cache_root, Some(SessionCacheChoice::Off), None),
        false,
        ""
    )
    .is_err());
}

#[test]
fn root_authority_rejects_relative_root_missing_ancestor_and_fifo_config() {
    let relative = Path::new("relative-state");
    assert!(execute_fake(
        args(relative, Some(SessionCacheChoice::Off), None),
        false,
        ""
    )
    .is_err());

    let temp = TempDir::new().unwrap();
    let parent = temp.path().canonicalize().unwrap();
    let missing_ancestor = parent.join("missing/state");
    assert!(execute_fake(
        args(&missing_ancestor, Some(SessionCacheChoice::Off), None),
        false,
        ""
    )
    .is_err());
    assert!(!parent.join("missing").exists());

    for name in ["config.toml", ".config.toml.lock", ".config.toml.partial"] {
        let root = parent.join(format!("fifo-{}", name.replace('.', "-")));
        fs::create_dir(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
        let path = std::ffi::CString::new(root.join(name).as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        assert!(execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").is_err());
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
        execute_fake(
            args(Path::new(&root), Some(SessionCacheChoice::Off), None),
            false,
            "",
        )
        .unwrap();
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

#[test]
fn runtime_session_policy_is_closed_zero_safe_and_read_only() {
    let temp = TempDir::new().unwrap();

    let absent = test_root(&temp, "absent-runtime-policy");
    assert!(matches!(
        authorize_session_cache_policy(&absent).unwrap(),
        SessionCachePolicyAuthorization::Absent
    ));
    assert!(!absent.exists());

    let disabled_root = test_root(&temp, "disabled-runtime-policy");
    execute_fake(
        args(&disabled_root, Some(SessionCacheChoice::Off), None),
        false,
        "",
    )
    .unwrap();
    let disabled = authorize_session_cache_policy(&disabled_root).unwrap();
    let SessionCachePolicyAuthorization::Disabled(disabled) = disabled else {
        panic!("zero must authorize no persistor");
    };
    disabled.revalidate().unwrap();

    let enabled_root = test_root(&temp, "enabled-runtime-policy");
    execute_fake(
        args(&enabled_root, Some(SessionCacheChoice::On), Some("32GiB")),
        false,
        "",
    )
    .unwrap();
    let enabled = authorize_session_cache_policy(&enabled_root).unwrap();
    let SessionCachePolicyAuthorization::Enabled(enabled) = enabled else {
        panic!("positive config must mint an enabled authorization");
    };
    assert_eq!(enabled.limit_bytes().get(), 32 * GIB);
    enabled.revalidate().unwrap();
    let debug = format!("{enabled:?}");
    assert!(!debug.contains(enabled_root.to_str().unwrap()));
    assert!(!debug.contains("config.toml"));
    assert!(!debug.contains(&(32 * GIB).to_string()));
}

#[test]
fn runtime_session_policy_fails_closed_on_hostile_or_changed_state() {
    let temp = TempDir::new().unwrap();
    let malformed = test_root(&temp, "malformed-runtime-policy");
    fs::create_dir(&malformed).unwrap();
    fs::set_permissions(&malformed, fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(malformed.join("config.toml"), b"not = [toml\n").unwrap();
    fs::set_permissions(
        malformed.join("config.toml"),
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    assert!(authorize_session_cache_policy(&malformed).is_err());
    assert!(!malformed.join("cache").exists());

    let root = test_root(&temp, "changed-runtime-policy");
    execute_fake(
        args(&root, Some(SessionCacheChoice::On), Some("1GiB")),
        false,
        "",
    )
    .unwrap();
    let authorization = authorize_session_cache_policy(&root).unwrap();
    let SessionCachePolicyAuthorization::Enabled(authorization) = authorization else {
        panic!("positive config must mint an enabled authorization");
    };
    let replacement = root.join("replacement.toml");
    fs::copy(root.join("config.toml"), &replacement).unwrap();
    fs::set_permissions(&replacement, fs::Permissions::from_mode(0o600)).unwrap();
    fs::rename(&replacement, root.join("config.toml")).unwrap();
    assert!(authorization.revalidate().is_err());

    let root = test_root(&temp, "changed-runtime-session-directory");
    execute_fake(
        args(&root, Some(SessionCacheChoice::On), Some("1GiB")),
        false,
        "",
    )
    .unwrap();
    let authorization = authorize_session_cache_policy(&root).unwrap();
    let SessionCachePolicyAuthorization::Enabled(authorization) = authorization else {
        panic!("positive config must mint an enabled authorization");
    };
    fs::rename(root.join("cache/sessions"), root.join("retained-sessions")).unwrap();
    fs::create_dir(root.join("cache/sessions")).unwrap();
    fs::set_permissions(
        root.join("cache/sessions"),
        fs::Permissions::from_mode(0o700),
    )
    .unwrap();
    assert!(authorization.revalidate().is_err());

    let root = test_root(&temp, "changed-runtime-installation-identity");
    crate::distribution::bootstrap_setup_test_identity(&root).unwrap();
    execute_fake(args(&root, Some(SessionCacheChoice::Off), None), false, "").unwrap();
    let authorization = authorize_session_cache_policy(&root).unwrap();
    let SessionCachePolicyAuthorization::Disabled(authorization) = authorization else {
        panic!("zero must mint only a disabled authorization");
    };
    let identity = root.join("update/installation-identity.json");
    let replacement = root.join("update/replacement-identity.json");
    fs::copy(&identity, &replacement).unwrap();
    fs::set_permissions(&replacement, fs::Permissions::from_mode(0o600)).unwrap();
    fs::rename(&replacement, &identity).unwrap();
    assert!(authorization.revalidate().is_err());
}

#[test]
fn runtime_session_policy_retains_no_writable_regular_file_descriptors() {
    let temp = TempDir::new().unwrap();
    let root = test_root(&temp, "read-only-runtime-policy");
    crate::distribution::bootstrap_setup_test_identity(&root).unwrap();
    execute_fake(
        args(&root, Some(SessionCacheChoice::On), Some("1GiB")),
        false,
        "",
    )
    .unwrap();
    let authorization = authorize_session_cache_policy(&root).unwrap();
    let SessionCachePolicyAuthorization::Enabled(authorization) = authorization else {
        panic!("positive config must mint an enabled authorization");
    };
    assert!(authorization
        .retained_regular_files_are_read_only_for_test()
        .unwrap());
}
