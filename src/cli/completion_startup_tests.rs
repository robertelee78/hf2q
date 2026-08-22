use super::*;
use std::os::unix::fs::PermissionsExt as _;

const TEST_BINDING: &str = "# hf2q-completion-binding sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn effective_zdotdir_probe_accepts_only_framed_absolute_output() {
    let root = tempfile::tempdir().unwrap();
    let shell = root.path().join("zsh");
    let expected = root.path().join("effective");
    let mut stdout = ZDOTDIR_PROBE_MARKER.to_vec();
    stdout.extend_from_slice(expected.to_str().unwrap().as_bytes());
    stdout.push(0);
    let result = probe_effective_zdotdir_with_runner(&shell, |program, args, timeout| {
        assert_eq!(program, Path::new("/usr/bin/env"));
        assert_eq!(timeout, 2);
        assert_eq!(args[0], "HF2Q_COMPLETION_ZDOTDIR_PROBE=1");
        assert_eq!(args[1], "HF2Q_NO_COMPLETION_INSTALL=1");
        Some(std::process::Output {
            status: std::os::unix::process::ExitStatusExt::from_raw(0),
            stdout,
            stderr: Vec::new(),
        })
    })
    .unwrap();
    assert_eq!(result, expected);
}

#[test]
fn append_refresh_and_malformed_preservation_are_exact() {
    let desired = bash_block(Path::new("/tmp/hf2q completion/hf2q"), TEST_BINDING).unwrap();
    let original = b"# operator line\nexport KEEP=yes\n";
    let appended = reconcile_block(original, &desired).unwrap().unwrap();
    assert!(appended.starts_with(original));
    assert!(appended.ends_with(&desired));

    let refreshed = bash_block(Path::new("/new/hf2q"), TEST_BINDING).unwrap();
    let replaced = reconcile_block(&appended, &refreshed).unwrap().unwrap();
    assert!(replaced.starts_with(original));
    assert!(replaced.ends_with(&refreshed));
    assert!(!String::from_utf8_lossy(&replaced).contains("/tmp/hf2q completion"));

    let malformed = b"keep\n# >>> hf2q managed completion >>>\npartial\n";
    assert!(reconcile_block(malformed, &desired).unwrap().is_none());
    let duplicate = [desired.as_slice(), desired.as_slice()].concat();
    assert!(reconcile_block(&duplicate, &desired).unwrap().is_none());
}

#[test]
fn reconcile_preserves_operator_content_mode_and_symlink() {
    let root = tempfile::tempdir().unwrap();
    let referent = root.path().join("dotfiles/zshrc");
    fs::create_dir_all(referent.parent().unwrap()).unwrap();
    fs::write(&referent, b"# operator content\n").unwrap();
    fs::set_permissions(&referent, fs::Permissions::from_mode(0o640)).unwrap();
    let logical = root.path().join(".zshrc");
    std::os::unix::fs::symlink(&referent, &logical).unwrap();
    let block = zsh_block(Path::new("/tmp/functions dir"), TEST_BINDING).unwrap();

    assert!(matches!(
        reconcile_file(&logical, &block),
        Outcome::Wrote(_)
    ));
    assert!(fs::symlink_metadata(&logical)
        .unwrap()
        .file_type()
        .is_symlink());
    let content = fs::read(&referent).unwrap();
    assert!(content.starts_with(b"# operator content\n"));
    assert!(content.ends_with(&block));
    assert_eq!(
        fs::metadata(&referent).unwrap().permissions().mode() & 0o7777,
        0o640
    );
    assert!(matches!(
        reconcile_file(&logical, &block),
        Outcome::UpToDate(_)
    ));
}

#[test]
fn exact_cleanup_removes_only_the_managed_block() {
    let root = tempfile::tempdir().unwrap();
    let startup = root.path().join(".bashrc");
    fs::write(&startup, b"before\n").unwrap();
    let block = bash_block(Path::new("/tmp/hf2q"), TEST_BINDING).unwrap();
    assert!(matches!(
        reconcile_file(&startup, &block),
        Outcome::Wrote(_)
    ));
    let digest = managed_block_digest(&startup).unwrap().unwrap();
    assert!(matches!(
        remove_managed_block(&startup, &digest).unwrap(),
        StartupCleanup::Removed
    ));
    assert_eq!(fs::read(&startup).unwrap(), b"before\n\n");
}

#[test]
fn cleanup_preserves_a_modified_managed_block() {
    let root = tempfile::tempdir().unwrap();
    let startup = root.path().join(".zshrc");
    let block = zsh_block(Path::new("/tmp/hf2q"), TEST_BINDING).unwrap();
    fs::write(&startup, &block).unwrap();
    let digest = managed_block_digest(&startup).unwrap().unwrap();
    let changed = String::from_utf8(block)
        .unwrap()
        .replace("return 0", "return 7");
    fs::write(&startup, changed.as_bytes()).unwrap();
    let result = remove_managed_block(&startup, &digest).unwrap();
    assert!(matches!(result, StartupCleanup::Preserved(_)));
    assert_eq!(fs::read(&startup).unwrap(), changed.as_bytes());
}

#[test]
fn generated_blocks_parse_in_real_shells() {
    let root = tempfile::tempdir().unwrap();
    let bash = root.path().join("bashrc");
    let zsh = root.path().join("zshrc");
    fs::write(
        &bash,
        bash_block(Path::new("/tmp/hf2q"), TEST_BINDING).unwrap(),
    )
    .unwrap();
    fs::write(
        &zsh,
        zsh_block(Path::new("/tmp/functions"), TEST_BINDING).unwrap(),
    )
    .unwrap();

    for shell in ["/bin/bash", "/bin/sh"] {
        if Path::new(shell).is_file() {
            let output = std::process::Command::new(shell)
                .arg("-n")
                .arg(&bash)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{shell}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    if Path::new("/bin/zsh").is_file() {
        let output = std::process::Command::new("/bin/zsh")
            .arg("-n")
            .arg(&zsh)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
