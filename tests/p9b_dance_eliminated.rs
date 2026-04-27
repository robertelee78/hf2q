use std::path::Path;

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("failed to read {path}: {err}");
    })
}

fn walk_rs_files(root: &Path, files: &mut Vec<std::path::PathBuf>) {
    let entries = std::fs::read_dir(root).unwrap_or_else(|err| {
        panic!("failed to read dir {}: {err}", root.display());
    });
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files(&path, files);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            files.push(path);
        }
    }
}

#[test]
fn p9b_intermediate_quantizer_file_is_removed() {
    assert!(
        !Path::new("src/quantize/intermediate_moe_q8.rs").exists(),
        "the P9b temporary quantizer module must stay deleted"
    );
}

#[test]
fn qwen35_dwq_capture_does_not_create_temp_artifact() {
    let main_rs = read("src/main.rs");
    let old_temp_gguf = concat!("intermediate", "-f16.gguf");
    assert!(main_rs.contains("requires_activation_capture"));
    assert!(
        !main_rs.contains(old_temp_gguf),
        "DWQ activation capture must not create the old temporary GGUF"
    );
    assert!(
        !main_rs.contains("tempfile::tempdir()"),
        "DWQ activation capture must not allocate a tempdir in cmd_convert"
    );
}

#[test]
fn removed_symbols_do_not_reappear_in_src_or_tests() {
    let removed_emit = concat!("emit_gguf_from_tensor", "_map");
    let removed_quantizer = concat!("Intermediate", "MoeQ8Quantizer");
    let mut files = Vec::new();
    walk_rs_files(Path::new("src"), &mut files);
    walk_rs_files(Path::new("tests"), &mut files);

    let this_file = Path::new("tests/p9b_dance_eliminated.rs");
    for path in files {
        if path == this_file {
            continue;
        }
        let text = read(path.to_str().expect("utf8 path"));
        assert!(
            !text.contains(removed_emit),
            "{} still references removed GGUF emit helper",
            path.display()
        );
        assert!(
            !text.contains(removed_quantizer),
            "{} still references removed temporary quantizer",
            path.display()
        );
    }
}

#[test]
#[ignore = "P11 hardware RSS gate: run with the apex MoE fixture on target hardware"]
fn test_apex_moe_capture_peak_rss() {
    let main_rs = read("src/main.rs");
    let old_temp_gguf = concat!("intermediate", "-f16.gguf");
    assert!(main_rs.contains("with_activation_capture_lazy"));
    assert!(!main_rs.contains(old_temp_gguf));
}
