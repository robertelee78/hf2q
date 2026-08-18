use std::fs;

use super::*;

#[test]
fn generated_registrations_keep_shell_headers_and_cli_grammar() {
    for shell in [&BASH, &ZSH, &FISH] {
        let rendered = render_registration(shell).expect("render registration");
        assert!(is_managed(&rendered, shell.marker_line));
        let rendered = String::from_utf8(rendered).expect("UTF-8 registration");
        assert!(
            rendered.contains("convert"),
            "{} grammar missing",
            shell.name
        );
        assert!(rendered.contains("serve"), "{} grammar missing", shell.name);
    }

    let zsh = String::from_utf8(render_registration(&ZSH).unwrap()).unwrap();
    assert!(zsh.starts_with("#compdef hf2q\n"));
    assert_eq!(zsh.lines().nth(1), Some(MARKER));
}

#[test]
fn registration_refreshes_only_owned_regular_files() {
    let root = tempfile::tempdir().expect("temporary completion root");
    let target = root.path().join(BASH.file);

    fs::write(&target, b"foreign completion\n").expect("write foreign occupant");
    assert!(reconcile_registration(&BASH, root.path())
        .expect("preserve foreign")
        .is_none());
    assert_eq!(fs::read(&target).unwrap(), b"foreign completion\n");

    fs::write(&target, format!("{MARKER}\nstale\n")).expect("write stale managed file");
    let installed = reconcile_registration(&BASH, root.path())
        .expect("refresh managed")
        .expect("exact registration installed");
    assert_eq!(installed, target);
    assert_eq!(
        fs::read(&target).unwrap(),
        render_registration(&BASH).unwrap()
    );
}

#[test]
fn startup_block_is_bounded_idempotent_and_preserves_malformed_layouts() {
    let block = startup_block(OsStr::new("zsh"), Path::new("/tmp/hf2q completions/_hf2q"))
        .expect("zsh block");
    let block_text = String::from_utf8(block.clone()).expect("UTF-8 startup block");
    assert!(block_text.contains("! -L \"$completion_file\""));
    assert!(block_text.contains(MARKER_PREFIX));
    assert!(block_text.contains("source \"$completion_file\" || true"));
    let first = reconcile_block(b"before\n", &block).expect("append block");
    assert!(first.starts_with(b"before\n"));
    assert_eq!(exact_line_ranges(&first, BEGIN).len(), 1);
    assert_eq!(exact_line_ranges(&first, END).len(), 1);
    assert_eq!(reconcile_block(&first, &block).unwrap(), first);

    let replacement =
        startup_block(OsStr::new("zsh"), Path::new("/new/_hf2q")).expect("replacement block");
    let replaced = reconcile_block(&first, &replacement).expect("replace block");
    let replaced = String::from_utf8(replaced).unwrap();
    assert!(replaced.starts_with("before\n"));
    assert!(replaced.contains("'/new/_hf2q'"));
    assert!(!replaced.contains("/tmp/hf2q completions"));

    assert!(reconcile_block(b"before\n# >>> hf2q managed completion >>>\n", &block).is_none());
}

#[test]
fn startup_reconciliation_preserves_a_symlink_and_surrounding_bytes() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("temporary startup root");
        let target = root.path().join("real-zshrc");
        let logical = root.path().join(".zshrc");
        fs::write(&target, b"operator-before\n").unwrap();
        symlink(&target, &logical).unwrap();
        let block = startup_block(OsStr::new("zsh"), Path::new("/tmp/_hf2q")).unwrap();

        reconcile_startup_file(&logical, &block).expect("reconcile symlink referent");
        assert!(fs::symlink_metadata(&logical)
            .unwrap()
            .file_type()
            .is_symlink());
        let bytes = fs::read(&target).unwrap();
        assert!(bytes.starts_with(b"operator-before\n"));
        assert_eq!(exact_line_ranges(&bytes, BEGIN).len(), 1);
        assert_eq!(exact_line_ranges(&bytes, END).len(), 1);
    }
}
