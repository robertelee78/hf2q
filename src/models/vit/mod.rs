//! Pure-Rust mmproj (vision-tower) emitter — ADR-012 Decision 18 / P10.
//!
//! Converts HF `model.vision_tower.*` + `model.multi_modal_projector.*`
//! safetensors tensors to a standalone `mmproj-<slug>-F16.gguf` file
//! carrying the ViT weights and the cross-modal projector. Sovereignty:
//! read `/opt/llama.cpp/tools/mtmd/clip-model.h` and `clip.cpp` as
//! spec sources; produce all code + tests natively.
//!
//! # Layer decomposition (P10's four-layer defense)
//!
//! - **Layer A (structural)**: synthetic tiny-ViT convert → read back via
//!   hf2q's own GGUF reader; assert every expected tensor name + metadata
//!   key is present with correct shape / dtype. Tests in
//!   `tests/convert_vision_tower_integration.rs`.
//!
//! - **Layer B (ADR-005 round-trip)**: synthetic + real-model load via
//!   `src/inference/vision/mmproj.rs`. `tests/convert_vision_tower_adr005_roundtrip.rs`.
//!
//! - **Layer C (spec-driven layout)**: hand-authored bytes on the four
//!   highest-risk mappings (fc1↔fc2, linear_1↔linear_2, patch-embd
//!   transpose, pos-embd dtype). Unit tests in `src/models/vit/convert.rs`.
//!
//! Layer D (external oracle) was removed per the 2026-04-24 sovereignty
//! audit — using an external mmproj output to prove our correctness is
//! exactly the pattern `feedback_hf2q_sovereignty.md` rejects.
//!
//! # Silent-skip semantics
//!
//! `convert_vision_tower` returns `Ok(None)` when the HF config has no
//! `vision_config` field. Gemma4 (no vision_config) and Qwen3.6-35B-A3B
//! MoE (vision_config dropped by the publisher) both silently skip;
//! only Qwen3.6-27B dense emits.

pub mod config;
pub mod convert;
pub mod gguf_emit;

use std::path::{Path, PathBuf};

pub use config::{VisionConfig, VisionConfigError};

/// Top-level errors the convert-vision-tower pipeline may surface.
///
/// Distinct from `VisionConfigError` so callers can choose between
/// "config was bad" (don't emit, surface a user-facing error) and
/// "emission pipeline bug" (panic-worthy for tests).
#[derive(Debug, thiserror::Error)]
pub enum VitConvertError {
    #[error("vision config parse error: {0}")]
    Config(#[from] VisionConfigError),

    #[error("safetensors read error: {0}")]
    Safetensors(String),

    #[error("GGUF emit error: {0}")]
    GgufEmit(String),

    #[error("tensor {name}: expected shape {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convert the vision tower of `hf_repo_dir` into an mmproj GGUF under
/// `output_dir`. Returns:
///
///   - `Ok(Some(path))` on successful emission.
///   - `Ok(None)` when the HF config has no `vision_config` — silent skip
///     path per Decision 18 §3. Gemma4 and Qwen3.6-35B-A3B MoE both hit
///     this branch (no regression on shipped dense-only arches).
///   - `Err(...)` on any conversion-side failure.
///
/// The output filename is `mmproj-<slug>-F16.gguf` where `<slug>` is
/// derived from the HF repo's `config.json::_name_or_path` or falls
/// back to the last segment of `hf_repo_dir`.
pub fn convert_vision_tower(
    hf_repo_dir: &Path,
    output_dir: &Path,
) -> Result<Option<PathBuf>, VitConvertError> {
    let config_path = hf_repo_dir.join("config.json");
    if !config_path.exists() {
        return Err(VitConvertError::Config(VisionConfigError::NoConfigJson));
    }

    // Silent-skip check: parse config.json as JSON and look for
    // vision_config key. No vision_config → return Ok(None).
    let raw = std::fs::read_to_string(&config_path)
        .map_err(|e| VitConvertError::Config(VisionConfigError::Io(e.to_string())))?;
    let root: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        VitConvertError::Config(VisionConfigError::BadJson(e.to_string()))
    })?;
    if root.get("vision_config").is_none() {
        return Ok(None);
    }

    // Load + validate the vision_config block.
    let vision_config = VisionConfig::from_hf_config(&root)?;

    // Compute output slug and path.
    let slug = compute_slug(&root, hf_repo_dir);
    let output = output_dir.join(format!("mmproj-{}-F16.gguf", slug));
    std::fs::create_dir_all(output_dir)
        .map_err(|e| VitConvertError::GgufEmit(format!("mkdir output_dir: {}", e)))?;

    // Build tensor map from safetensors + emit GGUF.
    let tensors = convert::load_vision_tensors(hf_repo_dir, &vision_config)?;
    gguf_emit::write_mmproj_gguf(&output, &vision_config, &tensors)?;

    Ok(Some(output))
}

/// Derive a filesystem-safe slug for the mmproj filename.
/// Priority: config.json::_name_or_path → last segment of hf_repo_dir.
pub fn compute_slug(config_root: &serde_json::Value, hf_repo_dir: &Path) -> String {
    if let Some(name) = config_root
        .get("_name_or_path")
        .and_then(|v| v.as_str())
    {
        return sanitize_slug(name);
    }
    hf_repo_dir
        .file_name()
        .and_then(|s| s.to_str())
        .map(sanitize_slug)
        .unwrap_or_else(|| "model".to_string())
}

fn sanitize_slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn no_vision_config_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("no-vision");
        fs::create_dir_all(&input).unwrap();
        fs::write(
            input.join("config.json"),
            r#"{"architectures":["Qwen3_5MoeForCausalLM"],"hidden_size":64}"#,
        )
        .unwrap();
        let out_dir = tmp.path().join("out");

        let result = convert_vision_tower(&input, &out_dir).expect("no error");
        assert!(result.is_none(), "no vision_config → Ok(None)");
    }

    #[test]
    fn missing_config_json_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("no-config");
        fs::create_dir_all(&input).unwrap();
        let out_dir = tmp.path().join("out");

        let err = convert_vision_tower(&input, &out_dir).unwrap_err();
        assert!(matches!(
            err,
            VitConvertError::Config(VisionConfigError::NoConfigJson)
        ));
    }

    #[test]
    fn malformed_json_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("bad-json");
        fs::create_dir_all(&input).unwrap();
        fs::write(input.join("config.json"), "not json").unwrap();
        let out_dir = tmp.path().join("out");

        let err = convert_vision_tower(&input, &out_dir).unwrap_err();
        assert!(matches!(
            err,
            VitConvertError::Config(VisionConfigError::BadJson(_))
        ));
    }

    #[test]
    fn compute_slug_from_name_or_path() {
        let root = serde_json::json!({"_name_or_path": "Qwen/Qwen3.6-27B"});
        let slug = compute_slug(&root, Path::new("/tmp/ignored"));
        assert_eq!(slug, "qwen-qwen3.6-27b");
    }

    #[test]
    fn compute_slug_from_directory_when_no_name() {
        let root = serde_json::json!({});
        let slug = compute_slug(&root, Path::new("/tmp/qwen3.6-27B-apex"));
        assert_eq!(slug, "qwen3.6-27b-apex");
    }

    #[test]
    fn sanitize_slug_strips_bad_chars() {
        assert_eq!(sanitize_slug("Foo/Bar_Baz.V2"), "foo-bar-baz.v2");
        assert_eq!(sanitize_slug("---leading"), "leading");
    }

    #[test]
    fn gemma4_config_returns_none_silent_regression_gate() {
        // Gemma4's config.json has no vision_config. This is the
        // regression gate from Decision 18 §4 — --emit-vision-tower
        // against Gemma4 must NOT emit a file and must NOT error.
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("gemma4-fixture");
        fs::create_dir_all(&input).unwrap();
        fs::write(
            input.join("config.json"),
            r#"{
                "architectures": ["Gemma4ForCausalLM"],
                "hidden_size": 2048,
                "num_hidden_layers": 26
            }"#,
        )
        .unwrap();
        let out_dir = tmp.path().join("out");

        let result = convert_vision_tower(&input, &out_dir).expect("gemma4 must not error");
        assert!(
            result.is_none(),
            "gemma4 has no vision_config — must silently skip"
        );
        // Out-dir is not even created (no reason to mkdir).
        assert!(!out_dir.exists(), "no output dir created on silent-skip");
    }

    #[test]
    fn qwen35moe_without_vision_config_silently_skips() {
        // The Robert-named 35B-A3B MoE target's config.json dropped
        // vision_config — --emit-vision-tower must silent-skip, not
        // emit a mmproj, not error.
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("qwen35moe-no-vc");
        fs::create_dir_all(&input).unwrap();
        fs::write(
            input.join("config.json"),
            r#"{
                "architectures": ["Qwen3_5MoeForCausalLM"],
                "hidden_size": 2048,
                "num_hidden_layers": 40,
                "num_experts": 256
            }"#,
        )
        .unwrap();
        let out_dir = tmp.path().join("out");

        let result = convert_vision_tower(&input, &out_dir).unwrap();
        assert!(result.is_none(), "MoE without vision_config must silent-skip");
    }
}
