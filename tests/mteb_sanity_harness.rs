//! ADR-005 Phase 2b — MTEB 5-task sanity harness.
//!
//! This integration test drives the AC at
//! `/opt/hf2q/docs/ADR-005-inference-server.md:3979`:
//!
//!   > MTEB 5-task sanity suite recovers published scores within ±1 pt
//!   > per supported model.
//!
//! It is **gated** behind `HF2Q_MTEB_E2E=1` because each leg of the
//! comparison loads a BERT-family embedding GGUF and shells out to a
//! Python `mteb` runner that downloads the task datasets on first use.
//! Default `cargo test` runs no model — only the harness compile path
//! and the cheap fixture / parser unit tests are exercised.
//!
//! Per the project's standing OOM-prevention directive (see memory:
//! "Check RAM before inference") the matrix loads **one model at a
//! time**: spawn server with `--embedding-model <gguf>`, run the 5-task
//! subset, SIGTERM, then load the next model. The 3 day-one BERT-family
//! models from ADR-005 Phase 2b are:
//!
//!   * `nomic-ai/nomic-embed-text-v1.5`
//!   * `mixedbread-ai/mxbai-embed-large-v1`
//!   * `BAAI/bge-small-en-v1.5`
//!
//! Closure criterion (Worker Z design, 2026-05-01): for every
//! (model × task) cell, `|measured - expected| <= 1.0`. 15/15 PASS =
//! AC line 3979 closes. The published scores baseline is checked in at
//! `tests/fixtures/mteb/expected_scores.json`.
//!
//! ENV knobs (only consulted when `HF2Q_MTEB_E2E=1`):
//!
//! | Env var | Default | Notes |
//! |---|---|---|
//! | `HF2Q_MTEB_E2E` | (unset) | gate; "1" enables the live matrix |
//! | `HF2Q_MTEB_GGUF_NOMIC` | `/opt/hf2q/models/bert-test/nomic-embed-text-v1.5-f16.gguf` | nomic-embed-text-v1.5 |
//! | `HF2Q_MTEB_GGUF_MXBAI` | `/opt/hf2q/models/bert-test/mxbai-embed-large-v1-f16.gguf` | mxbai-embed-large-v1 |
//! | `HF2Q_MTEB_GGUF_BGE` | `/opt/hf2q/models/bert-test/bge-small-en-v1.5-f16.gguf` | bge-small-en-v1.5 |
//! | `HF2Q_MTEB_PYTHON` | `python3` | python binary for the runner |
//! | `HF2Q_MTEB_PORT_BASE` | `8765` | hf2q server port (each spawn reuses it after the prev SIGTERM drains) |
//! | `HF2Q_MTEB_RESULTS_FOLDER` | `/tmp/mteb-output` | mteb's output_folder; flat scores JSONs land here |
//! | `HF2Q_MTEB_FLOOR` | `1.0` | per-cell drift floor in points; default matches AC line 3979 |
//!
//! Mirrors the env-gated subprocess pattern of
//! `tests/vision_e2e_vs_mlx_vlm.rs` (iter-101..104 vision E2E harness).
//!
//! See `/tmp/cfa-adr005-audit/mteb-harness-design.md` (Worker Z,
//! 2026-05-01) for the full design rationale + task-selection
//! justification.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

// -----------------------------------------------------------------------------
// Constants — task list, model list, expected_scores.json path
// -----------------------------------------------------------------------------

/// 5 MTEB tasks, one per major category (see Worker Z design §2):
/// STS, Classification, Retrieval, Clustering, Reranking. Each task is
/// the smallest representative dataset in its category that has a
/// published-leaderboard row for all 3 day-one models.
///
/// Order is the canonical wire order — every cell in the report is
/// keyed on (`model_idx`, `task_idx`) which indexes into these arrays.
const MTEB_TASKS: &[&str] = &[
    "BIOSSES",                       // STS — 100 sentence pairs
    "Banking77Classification",       // Classification — 13,083 / 77 intents
    "NFCorpus",                      // Retrieval — 3,237 q × 3,633 docs (BEIR)
    "TwentyNewsgroupsClustering",    // Clustering — 20,000 docs / 20 clusters
    "SciDocsRR",                     // Reranking — 19,394 q-doc pairs
];

/// 3 day-one BERT-family embedding models supported by the server
/// (`--embedding-model`).
///
/// `model_id` is the HuggingFace canonical id (used as the JSON key in
/// `expected_scores.json` and as the `--model-id` arg passed to
/// `runner.py`).
///
/// `gguf_env` is the env-var name an operator sets to point at the
/// GGUF on disk; `gguf_default` is the fallback used by the
/// in-tree bench harness (`scripts/bench_embedding.sh`).
struct DayOneModel {
    model_id: &'static str,
    gguf_env: &'static str,
    gguf_default: &'static str,
}

const DAY_ONE_MODELS: &[DayOneModel] = &[
    DayOneModel {
        model_id: "nomic-ai/nomic-embed-text-v1.5",
        gguf_env: "HF2Q_MTEB_GGUF_NOMIC",
        gguf_default: "/opt/hf2q/models/bert-test/nomic-embed-text-v1.5-f16.gguf",
    },
    DayOneModel {
        model_id: "mixedbread-ai/mxbai-embed-large-v1",
        gguf_env: "HF2Q_MTEB_GGUF_MXBAI",
        gguf_default: "/opt/hf2q/models/bert-test/mxbai-embed-large-v1-f16.gguf",
    },
    DayOneModel {
        model_id: "BAAI/bge-small-en-v1.5",
        gguf_env: "HF2Q_MTEB_GGUF_BGE",
        gguf_default: "/opt/hf2q/models/bert-test/bge-small-en-v1.5-f16.gguf",
    },
];

/// Path to the checked-in `expected_scores.json` baseline. The file is
/// the source of truth for the ±1pt floor; drift = re-pin via PR (see
/// the file's `_re_pin_policy` field).
fn expected_scores_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("mteb")
        .join("expected_scores.json")
}

/// Path to the Python runner that adapts mteb.Encoder to hf2q's
/// `/v1/embeddings` HTTP API. Invoked via subprocess from the gated
/// matrix driver below.
fn runner_py_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("mteb")
        .join("runner.py")
}

// -----------------------------------------------------------------------------
// Cheap unit tests (always run, even without the env-gate)
// -----------------------------------------------------------------------------

#[test]
fn mteb_tasks_are_exactly_five_and_unique() {
    assert_eq!(
        MTEB_TASKS.len(),
        5,
        "AC line 3979 mandates a 5-task suite — adjusting this count requires an ADR amendment"
    );
    use std::collections::HashSet;
    let mut seen: HashSet<&str> = HashSet::new();
    for (i, t) in MTEB_TASKS.iter().enumerate() {
        assert!(seen.insert(t), "MTEB_TASKS[{i}] = {t} duplicated");
        assert!(!t.is_empty(), "MTEB_TASKS[{i}] is empty");
    }
}

#[test]
fn day_one_models_are_exactly_three_and_unique_ids() {
    assert_eq!(DAY_ONE_MODELS.len(), 3, "Phase 2b ships 3 day-one models");
    use std::collections::HashSet;
    let mut ids: HashSet<&str> = HashSet::new();
    let mut envs: HashSet<&str> = HashSet::new();
    for (i, m) in DAY_ONE_MODELS.iter().enumerate() {
        assert!(ids.insert(m.model_id), "DAY_ONE_MODELS[{i}] model_id dup");
        assert!(envs.insert(m.gguf_env), "DAY_ONE_MODELS[{i}] gguf_env dup");
        assert!(
            m.gguf_env.starts_with("HF2Q_MTEB_GGUF_"),
            "gguf_env must follow HF2Q_MTEB_GGUF_<NAME> convention; got {}",
            m.gguf_env
        );
    }
}

#[test]
fn expected_scores_json_parses_and_covers_all_15_cells_or_uses_nan_placeholder() {
    let path = expected_scores_path();
    assert!(
        path.exists(),
        "expected_scores.json missing at {}",
        path.display()
    );
    let text = std::fs::read_to_string(&path).expect("read expected_scores.json");
    let v: serde_json::Value =
        serde_json::from_str(&text).expect("expected_scores.json is valid JSON");
    let obj = v.as_object().expect("top level must be a JSON object");
    // Metadata fields starting with `_` are ignored by the matrix
    // driver. They MUST be present so the file documents itself.
    assert!(obj.contains_key("_source"), "_source metadata missing");
    assert!(
        obj.contains_key("_re_pin_policy"),
        "_re_pin_policy metadata missing"
    );
    // Every day-one model needs an entry; every entry needs all 5
    // tasks. Scores are either a finite number or `null` (TODO
    // placeholder — verified-but-unfilled). Anything else is a hard
    // schema fail and forces a re-pin PR.
    for m in DAY_ONE_MODELS {
        let entry = obj
            .get(m.model_id)
            .unwrap_or_else(|| panic!("expected_scores.json missing entry for {}", m.model_id));
        let entry_obj = entry
            .as_object()
            .unwrap_or_else(|| panic!("entry for {} is not a JSON object", m.model_id));
        for task in MTEB_TASKS {
            let cell = entry_obj.get(*task).unwrap_or_else(|| {
                panic!(
                    "expected_scores.json: missing {} cell for {}",
                    task, m.model_id
                )
            });
            // Either a finite number or null (placeholder). NaN-as-null
            // is the design report's recommendation when a leaderboard
            // value couldn't be verified at harness-land time; the
            // matrix driver treats null as "skip with WARN".
            if cell.is_null() {
                continue;
            }
            let n = cell.as_f64().unwrap_or_else(|| {
                panic!(
                    "expected_scores.json: cell ({}, {}) is neither number nor null: {cell}",
                    m.model_id, task
                )
            });
            assert!(
                n.is_finite() && (0.0..=100.0).contains(&n),
                "expected_scores.json: cell ({}, {}) = {n} is not a sane MTEB score (0..=100)",
                m.model_id,
                task
            );
        }
    }
}

#[test]
fn runner_py_exists_and_imports_required_modules() {
    let path = runner_py_path();
    assert!(path.exists(), "runner.py missing at {}", path.display());
    let text = std::fs::read_to_string(&path).expect("read runner.py");
    // Spot-check that the runner imports the canonical mteb + numpy +
    // requests modules (any drift will surface here before a live run
    // wastes operator time on a slow venv install).
    for needle in &["import mteb", "import numpy", "import requests"] {
        assert!(
            text.contains(needle),
            "runner.py is missing `{needle}`"
        );
    }
    // The Encoder adapter must POST to /v1/embeddings.
    assert!(
        text.contains("/v1/embeddings"),
        "runner.py does not target /v1/embeddings"
    );
}

#[test]
fn parse_runner_output_round_trips_well_formed_json() {
    let raw = r#"{"model": "BAAI/bge-small-en-v1.5", "tasks": {"BIOSSES": 87.4, "NFCorpus": 31.1}}"#;
    let parsed = parse_runner_output(raw).expect("well-formed runner JSON");
    assert_eq!(parsed.model, "BAAI/bge-small-en-v1.5");
    assert!((parsed.tasks["BIOSSES"] - 87.4).abs() < 1e-6);
    assert!((parsed.tasks["NFCorpus"] - 31.1).abs() < 1e-6);
}

#[test]
fn parse_runner_output_rejects_invalid_json() {
    assert!(parse_runner_output("not json").is_err());
    assert!(parse_runner_output("{}").is_err()); // missing "model"
}

#[test]
fn delta_floor_passes_within_one_point_and_fails_beyond() {
    assert!(within_floor(87.4, 87.4, 1.0));
    assert!(within_floor(87.4, 86.5, 1.0));
    assert!(within_floor(87.4, 88.4, 1.0));
    // Inclusive at exactly 1.0
    assert!(within_floor(87.4, 88.4 + 1e-12, 1.0));
    // Beyond floor
    assert!(!within_floor(87.4, 89.0, 1.0));
    assert!(!within_floor(87.4, 85.0, 1.0));
}

// -----------------------------------------------------------------------------
// Gated end-to-end matrix (only when HF2Q_MTEB_E2E=1)
// -----------------------------------------------------------------------------

#[test]
fn mteb_sanity_matrix_5_tasks_3_models() {
    if std::env::var("HF2Q_MTEB_E2E").as_deref() != Ok("1") {
        eprintln!(
            "mteb_sanity_matrix_5_tasks_3_models: SKIPPED \
             (HF2Q_MTEB_E2E != \"1\"; live matrix needs 3 BERT GGUFs + a python venv with mteb)"
        );
        return;
    }

    // Pre-flight existence-check on every GGUF before spawning anything
    // (per pattern_env_override_close_gates: existence-check first; no
    // sentinel fallback). Lets a missing model surface as a single
    // clear error instead of a confusing /readyz timeout three minutes
    // in.
    let mut gguf_paths: Vec<PathBuf> = Vec::with_capacity(DAY_ONE_MODELS.len());
    for m in DAY_ONE_MODELS {
        let p =
            std::env::var(m.gguf_env).unwrap_or_else(|_| m.gguf_default.to_string());
        let path = PathBuf::from(&p);
        assert!(
            path.exists(),
            "{} = {} does not exist (set the env var or place the GGUF at the default path)",
            m.gguf_env,
            p
        );
        gguf_paths.push(path);
    }
    let runner = runner_py_path();
    assert!(runner.exists(), "runner.py missing at {}", runner.display());
    let python = std::env::var("HF2Q_MTEB_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let port_base: u16 = std::env::var("HF2Q_MTEB_PORT_BASE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8765);
    let results_folder = std::env::var("HF2Q_MTEB_RESULTS_FOLDER")
        .unwrap_or_else(|_| "/tmp/mteb-output".to_string());
    let floor: f64 = std::env::var("HF2Q_MTEB_FLOOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    // Read expected scores once up front.
    let expected = load_expected_scores(&expected_scores_path()).expect("load expected scores");

    // Drive the matrix: load model i, run all 5 tasks for that model,
    // SIGTERM the server, then move on to model i+1. Captures every
    // cell into `report.cells`.
    let mut report = MatrixReport {
        cells: Vec::with_capacity(DAY_ONE_MODELS.len() * MTEB_TASKS.len()),
        floor,
        expected_scores_path: expected_scores_path().to_string_lossy().into_owned(),
        runner_py_path: runner.to_string_lossy().into_owned(),
        results_folder: results_folder.clone(),
        port_base,
    };

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let report_path = PathBuf::from(format!("/tmp/mteb-results-{ts}.json"));

    for (mi, m) in DAY_ONE_MODELS.iter().enumerate() {
        let gguf_path = &gguf_paths[mi];
        eprintln!(
            "mteb harness: model {}/{}: spawning hf2q with --embedding-model {}",
            mi + 1,
            DAY_ONE_MODELS.len(),
            gguf_path.display()
        );
        let cells = match run_one_model_arm(RunArm {
            model_id: m.model_id,
            gguf_path,
            python: &python,
            runner_py: &runner,
            port: port_base,
            results_folder: &results_folder,
            tasks: MTEB_TASKS,
        }) {
            Ok(per_model) => per_model,
            Err(e) => {
                // A spawn / readyz / runner failure for one model is
                // surfaced loud-and-clear in the report so the operator
                // can triage; the other models still run for context.
                eprintln!(
                    "mteb harness: model {} ARM FAILED: {e}",
                    m.model_id
                );
                MTEB_TASKS
                    .iter()
                    .map(|t| MeasuredCell {
                        model_id: m.model_id.to_string(),
                        task: t.to_string(),
                        measured: f64::NAN,
                        error: Some(format!("{e:#}")),
                    })
                    .collect()
            }
        };

        for c in cells {
            let exp = expected
                .get(m.model_id)
                .and_then(|t| t.get(&c.task))
                .copied()
                .flatten();
            let delta = match (exp, c.measured.is_finite()) {
                (Some(e), true) => Some(c.measured - e),
                _ => None,
            };
            let pass = match (exp, delta) {
                (Some(_), Some(d)) => d.abs() <= floor,
                (None, _) => false, // null placeholder = WARN not PASS
                (Some(_), None) => false,
            };
            report.cells.push(ReportCell {
                model_id: c.model_id,
                task: c.task,
                measured: c.measured,
                expected: exp,
                delta,
                pass,
                error: c.error,
            });
        }
    }

    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    if let Err(e) = std::fs::write(&report_path, &json) {
        eprintln!(
            "mteb harness: WARN failed to write report at {}: {e}",
            report_path.display()
        );
    } else {
        eprintln!("mteb harness: wrote report at {}", report_path.display());
    }

    // Closure criterion: 15 cells, all PASS. Print a per-cell line for
    // operator triage on failure.
    let mut total = 0usize;
    let mut passed = 0usize;
    let mut warned = 0usize;
    for c in &report.cells {
        total += 1;
        let status = if c.pass {
            passed += 1;
            "PASS"
        } else if c.expected.is_none() {
            warned += 1;
            "WARN(no-baseline)"
        } else {
            "FAIL"
        };
        eprintln!(
            "  [{status}] {} / {} measured={:.4} expected={:?} delta={:?} err={:?}",
            c.model_id, c.task, c.measured, c.expected, c.delta, c.error
        );
    }
    eprintln!(
        "mteb harness: {passed}/{total} PASS, {warned} WARN(no-baseline), report at {}",
        report_path.display()
    );
    let expected_total = DAY_ONE_MODELS.len() * MTEB_TASKS.len();
    assert_eq!(
        total, expected_total,
        "report has {total} cells, expected {expected_total}"
    );
    assert_eq!(
        passed + warned,
        total,
        "AC line 3979 closure: every cell must be PASS or WARN(no-baseline); \
         passed={passed} warned={warned} total={total}; see report at {}",
        report_path.display()
    );
    // Hard close gate: every cell with a baseline must PASS. WARN
    // (null placeholder) does not block the harness — it surfaces in
    // stderr as a re-pin TODO.
    assert!(
        warned == 0 || std::env::var("HF2Q_MTEB_ALLOW_WARN").as_deref() == Ok("1"),
        "{warned} cells are WARN(no-baseline) — re-pin tests/fixtures/mteb/expected_scores.json \
         with verified leaderboard values, or set HF2Q_MTEB_ALLOW_WARN=1 to accept the gap."
    );
}

// -----------------------------------------------------------------------------
// Internal: matrix driver primitives + report types
// -----------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct MatrixReport {
    cells: Vec<ReportCell>,
    floor: f64,
    expected_scores_path: String,
    runner_py_path: String,
    results_folder: String,
    port_base: u16,
}

#[derive(serde::Serialize)]
struct ReportCell {
    model_id: String,
    task: String,
    measured: f64,
    expected: Option<f64>,
    delta: Option<f64>,
    pass: bool,
    error: Option<String>,
}

struct MeasuredCell {
    model_id: String,
    task: String,
    measured: f64,
    error: Option<String>,
}

struct RunArm<'a> {
    model_id: &'a str,
    gguf_path: &'a Path,
    python: &'a str,
    runner_py: &'a Path,
    port: u16,
    results_folder: &'a str,
    tasks: &'a [&'a str],
}

/// Spawn `hf2q serve --embedding-model <gguf>`, poll `/readyz` until
/// 200 or 600s elapsed, run the Python runner once for the full task
/// set, parse its JSON output into per-task measured scores, and tear
/// down the server.
///
/// Failure mode: any step that doesn't land successfully is wrapped in
/// an `anyhow::Error` (via `Result<_, String>` so the test crate
/// doesn't need anyhow). The caller treats a per-model failure as a
/// row of NaN cells with `error = Some(...)` and continues with the
/// next model — the report is most useful when partial.
fn run_one_model_arm(arm: RunArm) -> Result<Vec<MeasuredCell>, String> {
    // 1. Resolve the hf2q binary (cargo sets CARGO_BIN_EXE_<name> for
    //    integration tests; fall back to target/release/hf2q for
    //    out-of-cargo invocations).
    let hf2q_bin = std::env::var("CARGO_BIN_EXE_hf2q").unwrap_or_else(|_| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/release/hf2q")
            .to_string_lossy()
            .into_owned()
    });

    let port_str = arm.port.to_string();
    let gguf_str = arm
        .gguf_path
        .to_str()
        .ok_or_else(|| "gguf path is not valid UTF-8".to_string())?;

    let mut child = std::process::Command::new(&hf2q_bin)
        .arg("serve")
        .args(["--embedding-model", gguf_str])
        .args(["--port", &port_str])
        .args(["--host", "127.0.0.1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn hf2q serve: {e}"))?;

    // 2. Poll /readyz. 600s ceiling matches the vision_e2e harness —
    //    BERT-family GGUFs load fast (<10s typically), but a cold mmap
    //    on a slow disk can still spike past the 30s bench script
    //    default.
    let base = format!("http://127.0.0.1:{}", arm.port);
    let ready_url = format!("{base}/readyz");
    let started = Instant::now();
    let mut ready = false;
    while started.elapsed() < Duration::from_secs(600) {
        std::thread::sleep(Duration::from_millis(500));
        let status = std::process::Command::new("curl")
            .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", &ready_url])
            .output();
        if let Ok(out) = status {
            if std::str::from_utf8(&out.stdout).unwrap_or("") == "200" {
                ready = true;
                break;
            }
        }
        // If the child has died early, stop polling and surface the
        // exit status instead of waiting for a 10-minute timeout.
        if let Ok(Some(status)) = child.try_wait() {
            let _ = child.kill();
            return Err(format!(
                "hf2q server exited early before /readyz flipped (exit status: {status})"
            ));
        }
    }
    if !ready {
        let _ = child.kill();
        let _ = child.wait();
        return Err("hf2q /readyz did not return 200 within 600s".to_string());
    }

    // 3. Run the Python runner once for all 5 tasks. The runner emits
    //    a JSON file at --out which the Rust side reads back.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let out_path = std::env::temp_dir()
        .join(format!("mteb-runner-{}-{ts}.json", sanitize_for_filename(arm.model_id)));
    let tasks_csv = arm.tasks.join(",");
    let runner_status = std::process::Command::new(arm.python)
        .arg(arm.runner_py)
        .args(["--model-id", arm.model_id])
        .args(["--server", &base])
        .args(["--tasks", &tasks_csv])
        .args(["--out", out_path.to_str().unwrap_or("/tmp/mteb-runner-fallback.json")])
        .args(["--results-folder", arm.results_folder])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|c| c.wait_with_output());

    // 4. Tear down the server BEFORE parsing — we hold no lock on the
    //    runner JSON and want the GPU freed for the next arm even if
    //    parsing fails.
    let _ = child.kill();
    let drain = child.wait_with_output();
    let server_stderr_tail = match drain {
        Ok(out) => tail_bytes(&out.stderr, 8 * 1024),
        Err(e) => format!("(stderr drain failed: {e})"),
    };

    let runner_out = runner_status.map_err(|e| {
        format!(
            "spawn runner.py: {e}; hf2q stderr tail: {server_stderr_tail}"
        )
    })?;
    if !runner_out.status.success() {
        let stdout = String::from_utf8_lossy(&runner_out.stdout);
        let stderr = String::from_utf8_lossy(&runner_out.stderr);
        return Err(format!(
            "runner.py exited {:?}\n--- runner stdout ---\n{stdout}\n--- runner stderr ---\n{stderr}\n--- hf2q stderr tail ---\n{server_stderr_tail}",
            runner_out.status.code()
        ));
    }

    let body = std::fs::read_to_string(&out_path).map_err(|e| {
        format!(
            "read runner.py output {}: {e}",
            out_path.display()
        )
    })?;
    let parsed = parse_runner_output(&body)
        .map_err(|e| format!("parse runner.py output: {e}; raw body: {body}"))?;

    // 5. Re-shape into per-task MeasuredCell rows. Tasks the runner
    //    didn't return surface as NaN cells with an explicit error so
    //    the report flags the gap.
    let mut cells: Vec<MeasuredCell> = Vec::with_capacity(arm.tasks.len());
    for t in arm.tasks {
        let score = parsed.tasks.get(*t).copied();
        match score {
            Some(s) if s.is_finite() => cells.push(MeasuredCell {
                model_id: arm.model_id.to_string(),
                task: t.to_string(),
                measured: s,
                error: None,
            }),
            _ => cells.push(MeasuredCell {
                model_id: arm.model_id.to_string(),
                task: t.to_string(),
                measured: f64::NAN,
                error: Some(format!(
                    "runner.py did not return a finite score for task {t}"
                )),
            }),
        }
    }
    Ok(cells)
}

// -----------------------------------------------------------------------------
// JSON parsing + helpers
// -----------------------------------------------------------------------------

/// Minimal struct mirroring the runner.py output:
///
/// ```json
/// {"model": "<id>", "tasks": {"BIOSSES": 87.4, ...}}
/// ```
#[derive(Debug)]
struct RunnerOutput {
    model: String,
    tasks: BTreeMap<String, f64>,
}

fn parse_runner_output(raw: &str) -> Result<RunnerOutput, String> {
    let v: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| format!("not JSON: {e}"))?;
    let obj = v
        .as_object()
        .ok_or_else(|| "top level is not an object".to_string())?;
    let model = obj
        .get("model")
        .and_then(|m| m.as_str())
        .ok_or_else(|| "missing or non-string 'model' field".to_string())?
        .to_string();
    let tasks_v = obj
        .get("tasks")
        .ok_or_else(|| "missing 'tasks' field".to_string())?;
    let tasks_obj = tasks_v
        .as_object()
        .ok_or_else(|| "'tasks' is not a JSON object".to_string())?;
    let mut tasks: BTreeMap<String, f64> = BTreeMap::new();
    for (k, val) in tasks_obj {
        if let Some(n) = val.as_f64() {
            tasks.insert(k.clone(), n);
        }
    }
    Ok(RunnerOutput { model, tasks })
}

/// Load the checked-in expected-scores JSON. Returns a nested map
/// keyed on (`model_id`, `task`) with `Option<f64>` values; `None`
/// means the cell is a `null` placeholder (TODO re-pin) and is
/// surfaced as WARN not FAIL by the matrix driver.
fn load_expected_scores(
    path: &Path,
) -> Result<BTreeMap<String, BTreeMap<String, Option<f64>>>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("parse {}: {e}", path.display()))?;
    let obj = v
        .as_object()
        .ok_or_else(|| "expected_scores top level is not a JSON object".to_string())?;
    let mut out: BTreeMap<String, BTreeMap<String, Option<f64>>> = BTreeMap::new();
    for (model_id, entry) in obj {
        if model_id.starts_with('_') {
            continue; // metadata
        }
        let entry_obj = match entry.as_object() {
            Some(o) => o,
            None => continue,
        };
        let mut per_task: BTreeMap<String, Option<f64>> = BTreeMap::new();
        for (task, val) in entry_obj {
            if val.is_null() {
                per_task.insert(task.clone(), None);
            } else if let Some(n) = val.as_f64() {
                if n.is_finite() {
                    per_task.insert(task.clone(), Some(n));
                } else {
                    per_task.insert(task.clone(), None);
                }
            }
        }
        out.insert(model_id.clone(), per_task);
    }
    Ok(out)
}

/// `|measured - expected| <= floor` (inclusive). The +1e-9 fudge is
/// belt-and-braces against rounding artifacts in the published scores;
/// the per-cell floor is 1.0 by AC, so an FP rounding tie at exactly
/// 1.0 should land on the PASS side.
fn within_floor(expected: f64, measured: f64, floor: f64) -> bool {
    (measured - expected).abs() <= floor + 1e-9
}

/// Sanitize a model id (which contains `/`) into something safe to
/// use as a filename component. Replaces every non-alphanumeric (and
/// non-`-_.`) char with `_`.
fn sanitize_for_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Last `n_bytes` of `buf` as a UTF-8 lossy string. Mirrors the
/// vision_e2e harness helper of the same name.
fn tail_bytes(buf: &[u8], n_bytes: usize) -> String {
    let start = buf.len().saturating_sub(n_bytes);
    String::from_utf8_lossy(&buf[start..]).into_owned()
}

// -----------------------------------------------------------------------------
// Helper unit tests (always run)
// -----------------------------------------------------------------------------

#[test]
fn sanitize_for_filename_replaces_slashes() {
    assert_eq!(
        sanitize_for_filename("nomic-ai/nomic-embed-text-v1.5"),
        "nomic-ai_nomic-embed-text-v1.5"
    );
    assert_eq!(
        sanitize_for_filename("BAAI/bge-small-en-v1.5"),
        "BAAI_bge-small-en-v1.5"
    );
}

#[test]
fn sanitize_for_filename_preserves_safe_chars() {
    let s = "abcDEF-_.0123";
    assert_eq!(sanitize_for_filename(s), s);
}

#[test]
fn tail_bytes_returns_last_n_bytes() {
    let buf = b"abcdefghij";
    assert_eq!(tail_bytes(buf, 3), "hij");
}

#[test]
fn tail_bytes_returns_full_when_n_exceeds_len() {
    assert_eq!(tail_bytes(b"hi", 100), "hi");
}

#[test]
fn load_expected_scores_handles_metadata_keys_and_null_placeholders() {
    let dir = std::env::temp_dir();
    let p = dir.join(format!(
        "expected_scores_test_{}.json",
        std::process::id()
    ));
    let body = r#"{
      "_source": "MTEB leaderboard snapshot 2026-05-01",
      "_re_pin_policy": "Drift = re-pin via PR.",
      "BAAI/bge-small-en-v1.5": {
        "BIOSSES": 87.4,
        "Banking77Classification": 84.4,
        "NFCorpus": 31.0,
        "TwentyNewsgroupsClustering": 47.6,
        "SciDocsRR": 81.0
      },
      "nomic-ai/nomic-embed-text-v1.5": {
        "BIOSSES": null,
        "Banking77Classification": null,
        "NFCorpus": null,
        "TwentyNewsgroupsClustering": null,
        "SciDocsRR": null
      }
    }"#;
    std::fs::write(&p, body).expect("write tmp expected_scores");
    let m = load_expected_scores(&p).expect("parse");
    assert!(!m.contains_key("_source"));
    assert!(!m.contains_key("_re_pin_policy"));
    let bge = m.get("BAAI/bge-small-en-v1.5").expect("bge entry");
    assert_eq!(bge["BIOSSES"], Some(87.4));
    let nomic = m.get("nomic-ai/nomic-embed-text-v1.5").expect("nomic entry");
    assert_eq!(nomic["BIOSSES"], None);
    let _ = std::fs::remove_file(&p);
}
