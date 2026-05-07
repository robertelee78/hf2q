//! mlx-lm dynamic_quant subprocess wrapper — Path B GO/NO-GO gate (ADR-020 iter-8).
//!
//! mlx-native has no autograd surface (verified iter-8 audit: zero
//! `value_and_grad` / `VJP` / `JVP` / `tape` / `gradient` symbols across
//! `/opt/mlx-native/src/`).  Building reverse-mode autograd in Rust is
//! months of work — out of scope for an iter.  Per the same gate-first
//! pattern as Track 2 Path B (ADR §8.3), iter 8 wraps mlx-lm's CLI and
//! captures its sensitivity JSON output, so iter 9 can measure
//! perplexity gap on real models BEFORE committing to native port.
//!
//! ## Subprocess invocation
//!
//! `python -m mlx_lm.quant.dynamic_quant --model <input> --mlx-path <out>
//!  --target-bpw <f64> --low-bits <u8> --high-bits <u8>
//!  [--low-group-size <usize>] [--high-group-size <usize>]
//!  [--accumulation-dtype {float32,bfloat16}] [--grad-checkpoint]`
//!
//! mlx-lm writes:
//! - `<out>/` — MLX safetensors directory (the quantized model)
//! - `<model_basename>_sensitivities.json` in CWD — `[[path, score], …]`
//!
//! This wrapper:
//! 1. Builds the argv from a typed config.
//! 2. Runs the subprocess with a `--target-dir`-style WORKING directory
//!    so the auto-named JSON lands somewhere predictable.
//! 3. Parses the resulting JSON into a `BTreeMap<String, f64>`.
//! 4. Returns paths + the parsed sensitivity map.
//!
//! ## What this is NOT
//!
//! - NOT a permanent runtime dependency. Iter 9 measures the value of
//!   the algorithm; iter 13+ ports natively if the gap justifies the
//!   autograd investment.
//! - NOT used by the default `hf2q convert` flow. Wired through a
//!   gated CLI subcommand or a test harness only.
//! - NOT format-converting MLX safetensors → GGUF. The algorithm
//!   evaluation runs the MLX output via `mlx_lm.generate` for PPL;
//!   GGUF emit is a separate concern (per ADR §5 the formats are
//!   incompatible).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use thiserror::Error;

/// Configuration for invoking `mlx_lm.quant.dynamic_quant`.  Field
/// defaults match mlx-lm CLI defaults at
/// `/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py:151-185`.
#[derive(Debug, Clone)]
pub struct MlxLmDynamicQuantConfig {
    /// Input HF model path or repo id.  Maps to `--model`.
    pub model: String,
    /// Output directory for MLX safetensors.  Maps to `--mlx-path`.
    pub mlx_path: PathBuf,
    /// Target average bits-per-weight.  Maps to `--target-bpw`.  Default 5.0.
    pub target_bpw: f64,
    /// Bits for non-sensitive layers.  Maps to `--low-bits`.  Default 4.
    pub low_bits: u8,
    /// Group size for low-bit tensors.  Maps to `--low-group-size`.  Default 64.
    pub low_group_size: u32,
    /// Bits for sensitive layers.  Maps to `--high-bits`.  Default 5.
    pub high_bits: u8,
    /// Group size for high-bit tensors.  Maps to `--high-group-size`.  Default 64.
    pub high_group_size: u32,
    /// `--accumulation-dtype`.  `bfloat16` halves grad-accum memory
    /// (per researcher-2 finding).  Default `float32` to match mlx-lm.
    pub accumulation_dtype: AccumulationDtype,
    /// `--grad-checkpoint`.  Default false.
    pub grad_checkpoint: bool,
    /// `--seed`.  Default 123.
    pub seed: u64,
    /// Pre-computed sensitivities path (`--sensitivities`).  When set,
    /// mlx-lm skips its forward+backward pass and reuses the JSON.
    pub precomputed_sensitivities: Option<PathBuf>,
    /// Working directory the subprocess runs in.  The
    /// `<model>_sensitivities.json` output lands here.  Required —
    /// without a known cwd, the auto-named JSON is unrecoverable.
    pub working_dir: PathBuf,
    /// `python` executable path.  Defaults to `python3`.
    pub python_executable: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulationDtype {
    Float32,
    Bfloat16,
}

impl AccumulationDtype {
    pub const fn as_arg(&self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Bfloat16 => "bfloat16",
        }
    }
}

impl Default for MlxLmDynamicQuantConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            mlx_path: PathBuf::from("mlx_model"),
            target_bpw: 5.0,
            low_bits: 4,
            low_group_size: 64,
            high_bits: 5,
            high_group_size: 64,
            accumulation_dtype: AccumulationDtype::Float32,
            grad_checkpoint: false,
            seed: 123,
            precomputed_sensitivities: None,
            working_dir: PathBuf::from("."),
            python_executable: "python3".to_string(),
        }
    }
}

impl MlxLmDynamicQuantConfig {
    /// Build the argv for `python -m mlx_lm.quant.dynamic_quant ...`.
    ///
    /// Returns the full argv with `python_executable` as argv[0].
    pub fn argv(&self) -> Vec<String> {
        let mut args: Vec<String> = vec![
            self.python_executable.clone(),
            "-m".to_string(),
            "mlx_lm.quant.dynamic_quant".to_string(),
            "--model".to_string(),
            self.model.clone(),
            "--mlx-path".to_string(),
            self.mlx_path.to_string_lossy().into_owned(),
            "--target-bpw".to_string(),
            format!("{}", self.target_bpw),
            "--low-bits".to_string(),
            self.low_bits.to_string(),
            "--low-group-size".to_string(),
            self.low_group_size.to_string(),
            "--high-bits".to_string(),
            self.high_bits.to_string(),
            "--high-group-size".to_string(),
            self.high_group_size.to_string(),
            "--accumulation-dtype".to_string(),
            self.accumulation_dtype.as_arg().to_string(),
            "--seed".to_string(),
            self.seed.to_string(),
        ];
        if self.grad_checkpoint {
            args.push("--grad-checkpoint".to_string());
        }
        if let Some(p) = &self.precomputed_sensitivities {
            args.push("--sensitivities".to_string());
            args.push(p.to_string_lossy().into_owned());
        }
        args
    }
}

/// Result of running `mlx_lm.quant.dynamic_quant` to completion.
#[derive(Debug, Clone)]
pub struct MlxLmDynamicQuantResult {
    /// Path to the auto-named `<model_basename>_sensitivities.json`
    /// that mlx-lm wrote.  See
    /// `/opt/mlx-lm/mlx_lm/quant/dynamic_quant.py:204-206`.
    pub sensitivities_json: PathBuf,
    /// Parsed `path → score` map (mlx-lm's JSON is `[[path, score], …]`).
    pub sensitivities: BTreeMap<String, f64>,
    /// Output MLX safetensors directory (the `--mlx-path`).
    pub mlx_safetensors_dir: PathBuf,
    /// Full stdout captured from the subprocess.
    pub stdout: String,
    /// Full stderr captured.
    pub stderr: String,
}

/// Errors from the subprocess wrapper.
#[derive(Error, Debug)]
pub enum MlxLmDynamicQuantError {
    #[error("mlx-lm subprocess: failed to spawn `{argv0}`: {source}")]
    Spawn {
        argv0: String,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "mlx-lm subprocess: exited with status {status}; stderr tail:\n{stderr_tail}"
    )]
    NonZeroExit {
        status: i32,
        stderr_tail: String,
    },

    #[error("mlx-lm subprocess: working_dir {path} does not exist or is not a directory")]
    WorkingDirInvalid { path: String },

    #[error("mlx-lm subprocess: model field is empty (--model is required)")]
    EmptyModel,

    #[error("mlx-lm subprocess: cannot locate sensitivities JSON in {path} (looked for *_sensitivities.json)")]
    SensitivitiesJsonMissing { path: String },

    #[error("mlx-lm sensitivities JSON: I/O error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("mlx-lm sensitivities JSON: parse error at {path}: {source}")]
    JsonParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("mlx-lm sensitivities JSON: malformed entry at index {index}: expected [str, float], got {got}")]
    MalformedEntry { index: usize, got: String },
}

/// Parse mlx-lm's `*_sensitivities.json` format: `[[path, score], …]`
/// — list of two-element arrays, where the first element is a string
/// and the second is a numeric value.  Stripped `.weight` suffix per
/// `dynamic_quant.py:104`.
///
/// Returns a deterministic `BTreeMap` (alphabetical by path).
pub fn parse_mlx_lm_sensitivities_json(
    path: impl AsRef<Path>,
) -> Result<BTreeMap<String, f64>, MlxLmDynamicQuantError> {
    let path_ref = path.as_ref();
    let bytes = std::fs::read(path_ref).map_err(|e| MlxLmDynamicQuantError::Io {
        path: path_ref.display().to_string(),
        source: e,
    })?;

    // mlx-lm writes a list-of-pairs.  Use serde_json::Value to validate
    // the structure carefully — wrong-shape entries produce a
    // MalformedEntry error with the offending entry's debug form, not
    // a cryptic serde error.
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| MlxLmDynamicQuantError::JsonParse {
            path: path_ref.display().to_string(),
            source: e,
        })?;
    let arr = value
        .as_array()
        .ok_or_else(|| MlxLmDynamicQuantError::MalformedEntry {
            index: 0,
            got: format!("top-level was {:?}, expected array", short_type(&value)),
        })?;

    let mut out = BTreeMap::new();
    for (i, entry) in arr.iter().enumerate() {
        let pair = entry
            .as_array()
            .ok_or_else(|| MlxLmDynamicQuantError::MalformedEntry {
                index: i,
                got: format!("{:?}, expected [str, float]", short_type(entry)),
            })?;
        if pair.len() != 2 {
            return Err(MlxLmDynamicQuantError::MalformedEntry {
                index: i,
                got: format!("array of len {} (expected 2)", pair.len()),
            });
        }
        let key = pair[0]
            .as_str()
            .ok_or_else(|| MlxLmDynamicQuantError::MalformedEntry {
                index: i,
                got: format!("first element is {:?}, expected string", short_type(&pair[0])),
            })?
            .to_string();
        let score = pair[1]
            .as_f64()
            .ok_or_else(|| MlxLmDynamicQuantError::MalformedEntry {
                index: i,
                got: format!("second element is {:?}, expected number", short_type(&pair[1])),
            })?;
        out.insert(key, score);
    }
    Ok(out)
}

fn short_type(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Find the auto-named `<model_basename>_sensitivities.json` mlx-lm
/// writes in `cwd`.  Looks for any `*_sensitivities.json` file matching
/// (mlx-lm uses `args.model.replace("/", "_") + "_sensitivities.json"`
/// per `dynamic_quant.py:204`).
pub fn locate_sensitivities_json(
    working_dir: impl AsRef<Path>,
) -> Result<PathBuf, MlxLmDynamicQuantError> {
    let dir = working_dir.as_ref();
    let entries = std::fs::read_dir(dir).map_err(|e| MlxLmDynamicQuantError::Io {
        path: dir.display().to_string(),
        source: e,
    })?;
    for entry in entries.flatten() {
        let p = entry.path();
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if name.ends_with("_sensitivities.json") {
                return Ok(p);
            }
        }
    }
    Err(MlxLmDynamicQuantError::SensitivitiesJsonMissing {
        path: dir.display().to_string(),
    })
}

/// Run `mlx_lm.quant.dynamic_quant` to completion + parse the
/// resulting sensitivities JSON.
///
/// Caller must have a working Python environment with `mlx-lm`
/// installed (verified iter-8: `mlx_lm 0.31.2` with Metal).
///
/// **Memory profile** of the subprocess: per researcher-2 §6, mlx-lm's
/// estimate_sensitivities phase needs `model + q_model + grad_accum +
/// 2× activations` resident.  For a 35B BF16 model with default
/// `--accumulation-dtype float32` that's ≥ 280 GB — won't fit on
/// 128 GB.  Use `Bfloat16` accumulation + `grad_checkpoint=true` for
/// large models; small (Qwen3-0.6B) fits easily.
pub fn run_mlx_lm_dynamic_quant(
    config: &MlxLmDynamicQuantConfig,
) -> Result<MlxLmDynamicQuantResult, MlxLmDynamicQuantError> {
    if config.model.is_empty() {
        return Err(MlxLmDynamicQuantError::EmptyModel);
    }
    if !config.working_dir.is_dir() {
        return Err(MlxLmDynamicQuantError::WorkingDirInvalid {
            path: config.working_dir.display().to_string(),
        });
    }

    let argv = config.argv();
    let (argv0, args) = argv.split_first().expect("argv is non-empty");

    let output = Command::new(argv0)
        .args(args)
        .current_dir(&config.working_dir)
        .output()
        .map_err(|e| MlxLmDynamicQuantError::Spawn {
            argv0: argv0.clone(),
            source: e,
        })?;

    if !output.status.success() {
        let stderr_tail = String::from_utf8_lossy(&output.stderr).into_owned();
        // Cap to last 4 KiB to keep error readable in logs.
        let tail_start = stderr_tail.len().saturating_sub(4096);
        return Err(MlxLmDynamicQuantError::NonZeroExit {
            status: output.status.code().unwrap_or(-1),
            stderr_tail: stderr_tail[tail_start..].to_string(),
        });
    }

    let json_path = locate_sensitivities_json(&config.working_dir)?;
    let sensitivities = parse_mlx_lm_sensitivities_json(&json_path)?;

    Ok(MlxLmDynamicQuantResult {
        sensitivities_json: json_path,
        sensitivities,
        mlx_safetensors_dir: config.mlx_path.clone(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_json(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix("_sensitivities.json")
            .tempfile()
            .expect("tempfile");
        f.write_all(content.as_bytes()).expect("write");
        f.flush().expect("flush");
        f
    }

    #[test]
    fn parse_valid_two_entry_json() {
        // Mirror exact mlx-lm output shape (dynamic_quant.py:205-206).
        let f = write_json(r#"[["model.layers.0.self_attn.q_proj", 0.0123], ["model.layers.0.self_attn.k_proj", -0.0007]]"#);
        let m = parse_mlx_lm_sensitivities_json(f.path()).expect("parse OK");
        assert_eq!(m.len(), 2);
        assert!((m["model.layers.0.self_attn.q_proj"] - 0.0123).abs() < 1e-12);
        assert!((m["model.layers.0.self_attn.k_proj"] - (-0.0007)).abs() < 1e-12);
    }

    #[test]
    fn parse_empty_array_returns_empty_map() {
        let f = write_json("[]");
        let m = parse_mlx_lm_sensitivities_json(f.path()).expect("parse OK");
        assert!(m.is_empty(), "empty JSON array → empty map");
    }

    #[test]
    fn parse_signed_scores_preserved() {
        // mlx-lm sensitivities are SIGNED (researcher-2 §2).  Negative
        // scores mean low-bits actually helps loss — must be preserved.
        let f = write_json(r#"[["a", -1.5], ["b", 2.5]]"#);
        let m = parse_mlx_lm_sensitivities_json(f.path()).expect("parse OK");
        assert_eq!(m["a"], -1.5);
        assert_eq!(m["b"], 2.5);
    }

    #[test]
    fn parse_integer_score_promoted_to_float() {
        // mlx-lm uses `s.item()` which is fp; if a JSON arrives with
        // an integer (e.g. 0), parser must accept it.
        let f = write_json(r#"[["a", 0], ["b", 1]]"#);
        let m = parse_mlx_lm_sensitivities_json(f.path()).expect("parse OK");
        assert_eq!(m["a"], 0.0);
        assert_eq!(m["b"], 1.0);
    }

    #[test]
    fn parse_top_level_object_errors() {
        let f = write_json(r#"{"a": 1.0}"#);
        let e = parse_mlx_lm_sensitivities_json(f.path()).expect_err("must error");
        assert!(matches!(e, MlxLmDynamicQuantError::MalformedEntry { .. }));
    }

    #[test]
    fn parse_wrong_pair_length_errors() {
        let f = write_json(r#"[["a", 1.0, "extra"]]"#);
        let e = parse_mlx_lm_sensitivities_json(f.path()).expect_err("must error");
        match e {
            MlxLmDynamicQuantError::MalformedEntry { index, got } => {
                assert_eq!(index, 0);
                assert!(got.contains("len 3"), "got: {got}");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn parse_non_string_key_errors() {
        let f = write_json(r#"[[42, 1.0]]"#);
        let e = parse_mlx_lm_sensitivities_json(f.path()).expect_err("must error");
        assert!(matches!(e, MlxLmDynamicQuantError::MalformedEntry { .. }));
    }

    #[test]
    fn parse_non_numeric_score_errors() {
        let f = write_json(r#"[["a", "not-a-number"]]"#);
        let e = parse_mlx_lm_sensitivities_json(f.path()).expect_err("must error");
        assert!(matches!(e, MlxLmDynamicQuantError::MalformedEntry { .. }));
    }

    #[test]
    fn parse_invalid_json_propagates_serde_error() {
        let f = write_json("not even close to JSON");
        let e = parse_mlx_lm_sensitivities_json(f.path()).expect_err("must error");
        assert!(matches!(e, MlxLmDynamicQuantError::JsonParse { .. }));
    }

    #[test]
    fn parse_missing_file_io_error() {
        let e = parse_mlx_lm_sensitivities_json("/tmp/__definitely_does_not_exist_53194.json")
            .expect_err("must error");
        assert!(matches!(e, MlxLmDynamicQuantError::Io { .. }));
    }

    #[test]
    fn argv_default_config_well_formed() {
        let cfg = MlxLmDynamicQuantConfig {
            model: "mlx-community/Qwen3-0.6B-base".to_string(),
            ..Default::default()
        };
        let argv = cfg.argv();
        assert_eq!(argv[0], "python3");
        assert_eq!(argv[1], "-m");
        assert_eq!(argv[2], "mlx_lm.quant.dynamic_quant");
        // Required flags MUST appear.
        for required in [
            "--model",
            "--mlx-path",
            "--target-bpw",
            "--low-bits",
            "--high-bits",
            "--low-group-size",
            "--high-group-size",
            "--accumulation-dtype",
            "--seed",
        ] {
            assert!(
                argv.iter().any(|a| a == required),
                "missing required flag: {required}; argv={argv:?}"
            );
        }
        // grad_checkpoint=false → flag must NOT appear.
        assert!(
            !argv.iter().any(|a| a == "--grad-checkpoint"),
            "grad_checkpoint=false should omit flag"
        );
        // No precomputed_sensitivities → flag must NOT appear.
        assert!(
            !argv.iter().any(|a| a == "--sensitivities"),
            "no precomputed_sensitivities should omit flag"
        );
    }

    #[test]
    fn argv_grad_checkpoint_flag_emitted() {
        let cfg = MlxLmDynamicQuantConfig {
            model: "x".to_string(),
            grad_checkpoint: true,
            ..Default::default()
        };
        let argv = cfg.argv();
        assert!(argv.iter().any(|a| a == "--grad-checkpoint"));
    }

    #[test]
    fn argv_precomputed_sensitivities_emitted() {
        let cfg = MlxLmDynamicQuantConfig {
            model: "x".to_string(),
            precomputed_sensitivities: Some(PathBuf::from("/tmp/sens.json")),
            ..Default::default()
        };
        let argv = cfg.argv();
        let pos = argv.iter().position(|a| a == "--sensitivities").expect("flag");
        assert_eq!(argv[pos + 1], "/tmp/sens.json");
    }

    #[test]
    fn argv_bfloat16_accumulation_flag_value() {
        let cfg = MlxLmDynamicQuantConfig {
            model: "x".to_string(),
            accumulation_dtype: AccumulationDtype::Bfloat16,
            ..Default::default()
        };
        let argv = cfg.argv();
        let pos = argv
            .iter()
            .position(|a| a == "--accumulation-dtype")
            .expect("flag");
        assert_eq!(argv[pos + 1], "bfloat16");
    }

    #[test]
    fn run_empty_model_errors() {
        let cfg = MlxLmDynamicQuantConfig {
            model: String::new(),
            working_dir: std::env::temp_dir(),
            ..Default::default()
        };
        let r = run_mlx_lm_dynamic_quant(&cfg);
        assert!(matches!(r, Err(MlxLmDynamicQuantError::EmptyModel)));
    }

    #[test]
    fn run_invalid_working_dir_errors() {
        let cfg = MlxLmDynamicQuantConfig {
            model: "x".to_string(),
            working_dir: PathBuf::from("/tmp/__definitely_not_a_dir_84572"),
            ..Default::default()
        };
        let r = run_mlx_lm_dynamic_quant(&cfg);
        assert!(matches!(r, Err(MlxLmDynamicQuantError::WorkingDirInvalid { .. })));
    }

    #[test]
    fn locate_sensitivities_json_finds_auto_named_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Drop a file matching mlx-lm's pattern.
        let path = dir.path().join("mlx-community_Qwen3-0.6B-base_sensitivities.json");
        std::fs::write(&path, "[]").expect("write");
        let found = locate_sensitivities_json(dir.path()).expect("locate OK");
        assert_eq!(found, path);
    }

    #[test]
    fn locate_sensitivities_json_missing_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Empty dir.
        let r = locate_sensitivities_json(dir.path());
        assert!(matches!(r, Err(MlxLmDynamicQuantError::SensitivitiesJsonMissing { .. })));
    }

    /// Real subprocess test — gated `#[ignore]` because it requires:
    /// (a) Python + mlx-lm installed (verified iter-8: mlx_lm 0.31.2),
    /// (b) Network or local cache of `Qwen/Qwen3-0.6B-base`,
    /// (c) ~30 seconds runtime.
    /// Run manually with `cargo test mlx_lm_dynamic_quant_subprocess_e2e_qwen3_0_6b -- --ignored`.
    #[test]
    #[ignore]
    fn mlx_lm_dynamic_quant_subprocess_e2e_qwen3_0_6b() {
        let workdir = tempfile::tempdir().expect("tempdir");
        let cfg = MlxLmDynamicQuantConfig {
            model: "Qwen/Qwen3-0.6B-base".to_string(),
            mlx_path: workdir.path().join("out"),
            target_bpw: 5.0,
            low_bits: 4,
            high_bits: 5,
            working_dir: workdir.path().to_path_buf(),
            accumulation_dtype: AccumulationDtype::Bfloat16,
            grad_checkpoint: true,
            ..Default::default()
        };
        let r = run_mlx_lm_dynamic_quant(&cfg).expect("subprocess succeeds");
        assert!(!r.sensitivities.is_empty(), "got sensitivities");
        // 0.6B Qwen has 28 transformer blocks × ~6 quantizable Linears
        // each + embeddings ≈ ~170 entries, but exact count depends on
        // mlx-lm's `to_quantized` filter.  Sanity-check magnitude only.
        assert!(
            r.sensitivities.len() >= 50,
            "sensitivities count smells right; got {}",
            r.sensitivities.len()
        );
    }
}
