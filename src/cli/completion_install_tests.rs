use super::*;
#[test]
fn public_filter_removes_every_internal_command_and_argument_record() {
    for hidden in [
        "__standalone-install",
        "__fetch-hub-gguf",
        "__catalog-hub-gguf",
        "__verify-local-gguf",
        "source-teacher",
        "source-teacher-reference",
        "source-teacher-acceptance-verify",
        "--chat-parent-lifeline-fd",
        "--chat-startup-progress-fd",
    ] {
        assert!(!public_record(hidden), "{hidden} leaked");
        assert!(
            !public_record(&format!("{hidden}\tdescription")),
            "{hidden} leaked"
        );
        assert!(
            !public_record(&format!("{hidden}:description")),
            "{hidden} leaked"
        );
    }
    for public in ["convert", "serve", "--model", "Q4_K_M"] {
        assert!(public_record(public), "{public} was filtered");
    }
}

#[test]
fn every_registration_is_owned_bound_and_invokes_dynamic_protocol() {
    for shell in [&BASH, &ZSH, &FISH] {
        let bytes = render_registration(shell).expect(shell.name);
        let text = String::from_utf8(bytes.clone()).unwrap();
        assert!(is_hf2q_managed(&bytes, shell.marker_line));
        assert_eq!(text.matches(BINDING_PREFIX).count(), 1);
        assert!(text.contains("HF2Q_COMPLETE"));
        assert!(text.contains("hf2q"));
        assert!(!text.contains(BINDING_PLACEHOLDER));
    }
}

#[test]
fn zsh_keeps_compdef_first_and_first_tab_dispatch() {
    let text = String::from_utf8(render_registration(&ZSH).unwrap()).unwrap();
    assert!(text.starts_with("#compdef hf2q\n"), "{text}");
    assert_eq!(text.lines().nth(1), Some(MARKER));
    assert!(text.contains("_clap_dynamic_completer_hf2q"));
    assert!(text.contains("funcstack[(Ie)_hf2q]"));
}

#[test]
fn bash_adapter_restores_noglob_and_rechecks_dead_binary() {
    let text = String::from_utf8(render_registration(&BASH).unwrap()).unwrap();
    assert!(text.contains("set -f"));
    assert!(text.contains("set +f"));
    assert!(text.contains("type -P hf2q"));
    assert!(text.contains("[[ ! -f $_hf2q_completer || ! -x $_hf2q_completer ]]"));
}

#[test]
fn fish_adapter_is_autoload_shaped_and_rechecks_dead_binary() {
    let text = String::from_utf8(render_registration(&FISH).unwrap()).unwrap();
    assert!(text.contains("function __hf2q_dynamic_completer"));
    assert!(text.contains("command -v hf2q"));
    assert!(text.contains("--arguments \"(__hf2q_dynamic_completer)\""));
}

#[test]
fn absent_atomic_commit_never_clobbers_a_racing_target() {
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("hf2q");
    let result = atomic_replace_with_hook(
        root.path(),
        &target,
        b"managed",
        0o600,
        &ExpectedTarget::Absent,
        "race-test",
        || fs::write(&target, b"foreign").unwrap(),
    );
    assert!(result.is_err());
    assert_eq!(fs::read(target).unwrap(), b"foreign");
}

#[test]
fn changed_atomic_commit_never_replaces_a_racing_symlink() {
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("hf2q");
    fs::write(&target, b"old").unwrap();
    let expected = capture_regular_target(&target).unwrap();
    let referent = root.path().join("operator");
    fs::write(&referent, b"operator").unwrap();
    let result = atomic_replace_with_hook(
        root.path(),
        &target,
        b"managed",
        0o600,
        &expected,
        "race-test",
        || {
            fs::remove_file(&target).unwrap();
            std::os::unix::fs::symlink(&referent, &target).unwrap();
        },
    );
    assert!(result.is_err());
    assert!(fs::symlink_metadata(&target)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(fs::read(referent).unwrap(), b"operator");
}

#[test]
fn reconcile_is_idempotent_and_adopts_foreign_bytes_losslessly() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("bash");
    let first = try_reconcile_in(&BASH, &dir).unwrap();
    assert!(matches!(first, Outcome::Wrote(_)));
    let target = dir.join(BASH.file);
    let first_bytes = fs::read(&target).unwrap();
    let second = try_reconcile_in(&BASH, &dir).unwrap();
    assert!(matches!(second, Outcome::UpToDate(_)));
    assert_eq!(fs::read(&target).unwrap(), first_bytes);

    fs::write(&target, b"operator completion\n").unwrap();
    fs::set_permissions(&target, fs::Permissions::from_mode(0o640)).unwrap();
    let adopted = try_reconcile_in(&BASH, &dir).unwrap();
    let Outcome::Adopted {
        backup: Some(backup),
        ..
    } = adopted
    else {
        panic!("expected lossless adoption");
    };
    assert_eq!(fs::read(backup).unwrap(), b"operator completion\n");
    assert!(is_hf2q_managed(&fs::read(target).unwrap(), 0));
}

/// An isolated environment for destination resolution: lookups never consult
/// (or mutate) the process environment.
fn isolated_env(vars: &[(&str, &Path)]) -> impl Fn(&str) -> Option<OsString> {
    let vars = vars
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.as_os_str().to_owned()))
        .collect::<Vec<_>>();
    move |key| {
        vars.iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.clone())
    }
}

#[test]
fn explicit_destinations_work_without_automatic_policy() {
    let root = tempfile::tempdir().unwrap();
    let bash = root.path().join("bash");
    let zsh = root.path().join("zsh");
    let fish = root.path().join("fish");
    let env = isolated_env(&[
        ("BASH_COMPLETION_USER_DIR", &bash),
        (ZSH_DIR_VAR, &zsh),
        (FISH_DIR_VAR, &fish),
    ]);
    assert_eq!(
        bash_target_dirs_with_automatic_user(false, &env),
        vec![bash.join("completions")]
    );
    assert_eq!(
        zsh_target_dirs_with_automatic_locations(false, &env),
        vec![zsh]
    );
    assert_eq!(
        fish_target_dirs_with_automatic_user(false, &env),
        vec![fish]
    );
}

#[test]
fn zsh_automatic_destination_is_only_the_per_user_dir() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("home");
    let brew = root.path().join("brew");
    fs::create_dir_all(brew.join("share/zsh/site-functions")).unwrap();
    // Even with a Homebrew prefix exported, only the per-user dir is used.
    let env = isolated_env(&[("HOME", &home), ("HOMEBREW_PREFIX", &brew)]);
    assert_eq!(
        zsh_target_dirs_with_automatic_locations(true, &env),
        vec![home.join(".local/share/zsh/site-functions")]
    );
    let env = isolated_env(&[]);
    assert!(zsh_target_dirs_with_automatic_locations(true, &env).is_empty());
}

/// A fresh owned install resolved against a temporary HOME writes bash, zsh,
/// and Fish registrations only under that HOME, registers only those, and
/// uninstall removes exactly those plus the receipt.
#[test]
fn fresh_install_and_uninstall_stay_inside_home() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("home");
    fs::create_dir_all(&home).unwrap();
    let env = isolated_env(&[("HOME", &home)]);

    let mut registrations = Vec::new();
    for shell in [&BASH, &ZSH, &FISH] {
        let dirs = completions_dirs(shell, true, &env);
        assert_eq!(dirs.len(), 1, "{}: {dirs:?}", shell.name);
        for dir in dirs {
            match try_reconcile_in(shell, &dir).unwrap() {
                Outcome::Wrote(path) => registrations.push(path),
                other => panic!("{}: unexpected {other:?}", shell.name),
            }
        }
    }
    assert_eq!(
        registrations,
        vec![
            home.join(".local/share/bash-completion/completions/hf2q"),
            home.join(".local/share/zsh/site-functions/_hf2q"),
            home.join(".config/fish/completions/hf2q.fish"),
        ]
    );

    let receipt_path = home.join(".local/state/hf2q/completion-ownership-v1.json");
    super::super::completion_receipt::record_in(&receipt_path, Some(&home), &registrations, &[])
        .unwrap();

    // Every file the install produced lives under HOME.
    fn files_under(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if fs::symlink_metadata(&path).unwrap().is_dir() {
                files_under(&path, out);
            } else {
                out.push(path);
            }
        }
    }
    let mut written = Vec::new();
    files_under(root.path(), &mut written);
    written.sort();
    let mut expected = registrations.clone();
    expected.push(receipt_path.clone());
    expected.sort();
    assert_eq!(written, expected);

    let mut owned =
        super::super::completion_receipt::owned_paths_in(&receipt_path, Some(&home)).unwrap();
    owned.sort();
    assert_eq!(owned, expected);

    let cleanup = super::super::completion_receipt::cleanup_owned_in(&receipt_path, Some(&home));
    assert!(cleanup.preserved.is_empty(), "{:?}", cleanup.preserved);
    assert_eq!(cleanup.removed, expected);
    let mut remaining = Vec::new();
    files_under(root.path(), &mut remaining);
    assert!(remaining.is_empty(), "{remaining:?}");
}

#[test]
fn lifecycle_cleanup_is_detected_after_global_arguments_without_matching_nearby_commands() {
    let args = |values: &[&str]| values.iter().map(OsString::from).collect::<Vec<_>>();

    assert!(lifecycle_cleanup_requested(&args(&[
        "hf2q",
        "--state-root",
        "/tmp/state",
        "uninstall",
        "--yes",
    ])));
    assert!(lifecycle_cleanup_requested(&args(&[
        "hf2q",
        "update",
        "--rollback",
    ])));
    assert!(!lifecycle_cleanup_requested(&args(&["hf2q", "update"])));
    assert!(!lifecycle_cleanup_requested(&args(&[
        "hf2q",
        "convert",
        "uninstall",
    ])));
    assert!(!lifecycle_cleanup_requested(&args(&["hf2q", "--help",])));
}
