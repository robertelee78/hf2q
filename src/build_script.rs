//! Build-time converter provenance without invoking `git` or another helper.
//!
//! Registry packages carry Cargo's `.cargo_vcs_info.json`; source builds can
//! provide one of the explicit CI/release SHA variables. Dirty or unversioned
//! source trees deliberately receive no inferred commit and remote conversion
//! fails closed instead of claiming a provenance identity it cannot prove.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine as _;

const TERMINAL_RASTER_WIDTH: u32 = 224;
const ANSI_RASTER_WIDTH: u32 = 20;
const SITE_CANVAS_RGBA: (u8, u8, u8, u8) = (0x0b, 0x0c, 0x0e, 0xff);

fn exact_git_sha(value: &str) -> Option<String> {
    let value = value.trim();
    (value.len() == 40 && value.chars().all(|character| character.is_ascii_hexdigit()))
        .then(|| value.to_ascii_lowercase())
}

fn packaged_vcs_info_path(manifest_dir: &Path) -> Option<PathBuf> {
    let path = manifest_dir.join(".cargo_vcs_info.json");
    path.is_file().then_some(path)
}

fn packaged_vcs_sha(path: &Path) -> Option<String> {
    let contents = fs::read_to_string(path).ok()?;
    let after_key = contents.split_once("\"sha1\"")?.1;
    let after_colon = after_key.split_once(':')?.1;
    let quoted = after_colon.split_once('"')?.1;
    let value = quoted.split_once('"')?.0;
    exact_git_sha(value)
}

fn render_svg(
    svg: &[u8],
    width: u32,
    even_height: bool,
) -> Result<resvg::tiny_skia::Pixmap, String> {
    let options = resvg::usvg::Options::default();
    let tree = resvg::usvg::Tree::from_data(svg, &options)
        .map_err(|error| format!("parse assets/head.svg: {error}"))?;
    let source_size = tree.size();
    let scale = width as f32 / source_size.width();
    let scaled_height = source_size.height() * scale;
    let mut height = if even_height {
        scaled_height.round() as u32
    } else {
        scaled_height.ceil() as u32
    };
    if even_height && height % 2 != 0 {
        height += 1;
    }
    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height)
        .ok_or_else(|| format!("allocate {width}x{height} head.svg raster"))?;
    // The authoritative SVG is transparent white/black artwork. hf2q.us
    // presents it on this fixed canvas; baking that canvas keeps the exact
    // mark visible on both light and dark terminal themes without recoloring.
    pixmap.fill(resvg::tiny_skia::Color::from_rgba8(
        SITE_CANVAS_RGBA.0,
        SITE_CANVAS_RGBA.1,
        SITE_CANVAS_RGBA.2,
        SITE_CANVAS_RGBA.3,
    ));
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(scale, scale),
        &mut pixmap.as_mut(),
    );
    Ok(pixmap)
}

fn compile_terminal_assets(manifest_dir: &Path, out_dir: &Path) -> Result<(), String> {
    let source = manifest_dir.join("assets/head.svg.base64");
    println!("cargo:rerun-if-changed={}", source.display());
    let encoded = fs::read_to_string(&source)
        .map_err(|error| format!("read {}: {error}", source.display()))?;
    let svg = base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .map_err(|error| format!("decode exact head.svg asset: {error}"))?;
    fs::write(out_dir.join("hf2q-head.svg"), &svg)
        .map_err(|error| format!("write compiled exact head SVG: {error}"))?;

    let native = render_svg(&svg, TERMINAL_RASTER_WIDTH, false)?;
    let png = native
        .encode_png()
        .map_err(|error| format!("encode terminal head PNG: {error}"))?;
    fs::write(out_dir.join("hf2q-head.png"), png)
        .map_err(|error| format!("write compiled terminal head PNG: {error}"))?;

    // tiny-skia's byte representation is premultiplied RGBA. It is retained
    // verbatim and unpremultiplied only for the handful of visible ANSI cells
    // at runtime, avoiding PNG decode/resampling on every process start.
    let ansi = render_svg(&svg, ANSI_RASTER_WIDTH, true)?;
    fs::write(out_dir.join("hf2q-head-ansi.rgba"), ansi.data())
        .map_err(|error| format!("write compiled ANSI head raster: {error}"))?;
    Ok(())
}

fn main() {
    for name in ["GIT_COMMIT_SHA", "VERGEN_GIT_SHA", "GITHUB_SHA"] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    let explicit = ["GIT_COMMIT_SHA", "VERGEN_GIT_SHA", "GITHUB_SHA"]
        .into_iter()
        .filter_map(|name| env::var(name).ok())
        .find_map(|value| exact_git_sha(&value));
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR").map(std::path::PathBuf::from);
    let out_dir = env::var_os("OUT_DIR").map(std::path::PathBuf::from);
    compile_terminal_assets(
        manifest_dir
            .as_deref()
            .expect("Cargo must provide CARGO_MANIFEST_DIR"),
        out_dir.as_deref().expect("Cargo must provide OUT_DIR"),
    )
    .expect("compile exact hf2q terminal logo assets");
    let packaged_vcs_info = manifest_dir.as_deref().and_then(packaged_vcs_info_path);

    // Cargo treats a missing `rerun-if-changed` input as perpetually stale.
    // `.cargo_vcs_info.json` exists in a packaged crate but not in a normal Git
    // checkout, so only register the file dependency when Cargo supplied it.
    if let Some(path) = packaged_vcs_info.as_deref() {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let commit = explicit.or_else(|| packaged_vcs_info.as_deref().and_then(packaged_vcs_sha));

    if let Some(commit) = commit {
        println!("cargo:rustc-env=HF2Q_BUILD_GIT_SHA={commit}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_sha_validation_is_fail_closed() {
        assert_eq!(exact_git_sha(&"A".repeat(40)), Some("a".repeat(40)));
        assert_eq!(exact_git_sha("abc"), None);
        assert_eq!(exact_git_sha(&"g".repeat(40)), None);
    }

    #[test]
    fn packaged_vcs_info_supplies_registry_commit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".cargo_vcs_info.json");
        fs::write(
            &path,
            format!("{{\"git\":{{\"sha1\":\"{}\"}}}}", "D".repeat(40)),
        )
        .unwrap();
        assert_eq!(packaged_vcs_info_path(dir.path()), Some(path.clone()));
        assert_eq!(packaged_vcs_sha(&path), Some("d".repeat(40)));
    }

    #[test]
    fn missing_packaged_vcs_info_is_not_a_cargo_file_dependency() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(packaged_vcs_info_path(dir.path()), None);
    }
}
