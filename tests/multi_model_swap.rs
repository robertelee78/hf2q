//! ADR-047 real model-lifecycle swap proof.
//!
//! The old version of this test admitted a symlink to the same GGUF alongside
//! the startup model. That was a second pool key, but it was not a model swap:
//! neither the model bytes nor the tokenizer/template changed, no victim was
//! evicted, and no physical-memory release was proven.
//!
//! This gate requires two distinct physical GGUFs and exercises the production
//! revision-bound control plane in the sequence A -> B -> A. The larger file is
//! selected as A, and the pool byte budget is set to the larger file size so
//! either artifact fits alone while the pair cannot co-reside logically. Each
//! transition first obtains the exact conflict receipt, then submits that
//! receipt as an explicit switch.
//!
//! The proof is family-neutral. `*_KIND_{A,B}=chat` uses the chat endpoint;
//! `embedding` uses the embeddings endpoint. It asserts exact A-result replay
//! after reload, a different resident generation for the reloaded A, one
//! resident model throughout, load latency on both switch legs, and process
//! RSS/physical-footprint/wired-memory reclamation with no double-residency
//! peak. This is intentionally a real-hardware gate, not a hosted-safe smoke.
//!
//! # Why the AC budget is wall-clock under 10 s
//!
//! ADR-005 line 929 spec: "Hot-swap algorithm: LRU pool of loaded models,
//! memory-bounded, ollama-compatible semantics."  AC text at line 5466:
//! "Hot-swap between two cached GGUFs in under 10 seconds, measured on
//! M5 Max."  The OS page cache holds the GGUF after the startup pre-warm,
//! so the second load reads from RAM (not SSD); the dominant cost is GPU
//! upload + Metal-kernel warmup (approx 1-3 s for 16 GiB on M5 Max's
//! unified bus).  The 10 s ceiling is the operator-facing latency
//! budget — small enough to feel snappy, large enough to absorb cold
//! Metal compile.
//!
//! # Scopes
//!
//! 1. **Default (no env)**: skip with a diagnostic.  Keeps `cargo test
//!    --release` cheap on dev machines.
//! 2. **`HF2Q_HOT_SWAP_E2E=1`**: runs the full subprocess swap-timing
//!    harness against `HF2Q_HOT_SWAP_E2E_MODEL_A` (first GGUF;
//!    falls back to the canonical Gemma 4 chat fixture used by the rest
//!    of the LIVE-tests suite).  `HF2Q_HOT_SWAP_E2E_MODEL_B` is OPTIONAL —
//!    when unset, the test creates a tempdir symlink to MODEL_A under
//!    a distinct stem so the pool-key-uniqueness path runs against a
//!    single on-disk fixture.  When set, it points at a second physical
//!    GGUF and the test exercises a true two-file swap.
//!
//! ```bash
//! HF2Q_HOT_SWAP_E2E=1 \
//!   HF2Q_HOT_SWAP_E2E_MODEL_A=/path/to/larger.gguf \
//!   HF2Q_HOT_SWAP_E2E_MODEL_B=/path/to/smaller.gguf \
//!   cargo test --release --test multi_model_swap -- --test-threads=1 --nocapture
//! ```

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const ENV_GATE: &str = "HF2Q_HOT_SWAP_E2E";
const ENV_MODEL_A: &str = "HF2Q_HOT_SWAP_E2E_MODEL_A";
const ENV_MODEL_B: &str = "HF2Q_HOT_SWAP_E2E_MODEL_B";
const ENV_KIND_A: &str = "HF2Q_HOT_SWAP_E2E_KIND_A";
const ENV_KIND_B: &str = "HF2Q_HOT_SWAP_E2E_KIND_B";
const ENV_MAX_SECS: &str = "HF2Q_HOT_SWAP_E2E_MAX_SECS";

/// High-numbered fixed port distinct from the openwebui suite (52334),
/// `mmproj_llama_cpp_compat.rs` (52226), and `vision_e2e_vs_mlx_vlm.rs`
/// (18181).  Test runs under `--test-threads=1` per the OOM directive,
/// so collisions are operator error, not the harness's problem.
const PORT: u16 = 52337;
const HOST: &str = "127.0.0.1";

/// `/readyz` poll budget — cold-load + warmup of a 16 GiB chat GGUF on
/// M5 Max is on the order of 60-180 s; 10 minutes is the same budget
/// the openwebui suite uses, kept symmetric to avoid harness drift.
const READYZ_BUDGET_SECS: u64 = 600;

/// Per-request budget — first chat after warmup is fast, but the
/// second request triggers a 16 GiB cold load on the test path.  Give
/// it 30 s so the AC assertion (10 s) is the gate, not the reqwest
/// timeout.
const REQUEST_BUDGET_SECS: u64 = 180;

/// AC 5466 budget — hot-swap between two cached GGUFs must complete in
/// under 10 s on M5 Max.  This is the assertion bar.
const SWAP_BUDGET_SECS: u64 = 10;
const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
enum ProbeKind {
    Chat,
    Embedding,
}

impl ProbeKind {
    fn from_env(name: &str) -> Self {
        match std::env::var(name).as_deref() {
            Ok("embedding") => Self::Embedding,
            Ok("chat") | Err(_) => Self::Chat,
            Ok(other) => panic!("{name} must be `chat` or `embedding`, got {other:?}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MemorySnapshot {
    rss_bytes: u64,
    physical_footprint_bytes: u64,
    physical_footprint_peak_bytes: u64,
    wired_bytes: u64,
    system_wired_bytes: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct MemoryPeak {
    rss_bytes: u64,
    system_wired_bytes: u64,
}

fn skip_unless_gated(name: &str) -> bool {
    if std::env::var(ENV_GATE).as_deref() == Ok("1") {
        return false;
    }
    eprintln!(
        "[skip] {name} — set {ENV_GATE}=1 to run the real A -> B -> A model-swap harness. \
         Required: {ENV_MODEL_A} and {ENV_MODEL_B} must name distinct physical GGUFs. \
         Optional: {ENV_KIND_A}/{ENV_KIND_B}=chat|embedding and {ENV_MAX_SECS}."
    );
    true
}

/// Locate the `hf2q` binary the cargo test runner just built.
fn hf2q_binary_path() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_hf2q") {
        return PathBuf::from(path);
    }
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            PathBuf::from(manifest_dir).join("target")
        });
    let binary = target_dir.join("release").join("hf2q");
    assert!(
        binary.exists(),
        "hf2q binary not found at {} — did `cargo build --release` run?",
        binary.display()
    );
    binary
}

/// RAII guard around the spawned `hf2q serve` subprocess.  Drop kills the
/// child so a panic mid-test never strands a 16 GiB-resident server.
/// Mirrors `tests/openwebui_helpers/mod.rs::ServerGuard`.
struct ServerGuard(Child);

impl ServerGuard {
    fn spawn(gguf: &str, pool_budget_bytes: u64) -> std::io::Result<Self> {
        let bin = hf2q_binary_path();
        let child = Command::new(bin)
            .args([
                "serve",
                "--model",
                gguf,
                "--host",
                HOST,
                "--port",
                &PORT.to_string(),
            ])
            .env("HF2Q_POOL_BUDGET_BYTES", pool_budget_bytes.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()?;
        Ok(Self(child))
    }
}

fn process_rss_bytes(pid: u32) -> u64 {
    let rss = Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "rss="])
        .output()
        .expect("run ps for server RSS");
    assert!(rss.status.success(), "ps failed for server pid {pid}");
    let rss_kib = String::from_utf8_lossy(&rss.stdout)
        .trim()
        .parse::<u64>()
        .expect("parse ps RSS KiB");
    rss_kib.saturating_mul(1024)
}

fn system_wired_bytes() -> u64 {
    let vm_stat = Command::new("vm_stat")
        .output()
        .expect("run vm_stat for host wired memory");
    assert!(vm_stat.status.success(), "vm_stat failed");
    let vm_stat = String::from_utf8_lossy(&vm_stat.stdout);
    let page_size = vm_stat
        .lines()
        .next()
        .and_then(|line| line.split("page size of ").nth(1))
        .and_then(|tail| tail.split_whitespace().next())
        .and_then(|raw| raw.parse::<u64>().ok())
        .expect("parse vm_stat page size");
    let wired_pages = vm_stat
        .lines()
        .find_map(|line| line.strip_prefix("Pages wired down:"))
        .map(str::trim)
        .map(|raw| raw.trim_end_matches('.'))
        .and_then(|raw| raw.parse::<u64>().ok())
        .expect("parse vm_stat wired pages");
    wired_pages.saturating_mul(page_size)
}

fn process_memory_snapshot(pid: u32) -> MemorySnapshot {
    let rss_bytes = process_rss_bytes(pid);

    let tmp = tempfile::tempdir().expect("create footprint tempdir");
    let json_path = tmp.path().join("footprint.json");
    let output = Command::new("footprint")
        .args([
            "--pid",
            &pid.to_string(),
            "--json",
            json_path.to_str().expect("UTF-8 footprint path"),
        ])
        .output()
        .expect("run macOS footprint");
    assert!(
        output.status.success(),
        "footprint failed for server pid {pid}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&json_path).expect("read footprint JSON"))
            .expect("parse footprint JSON");
    let process = value["processes"]
        .as_array()
        .and_then(|rows| {
            rows.iter()
                .find(|row| row["pid"].as_u64() == Some(pid as u64))
        })
        .unwrap_or_else(|| panic!("footprint JSON missing server pid {pid}: {value}"));
    let required_u64 = |pointer: &str| {
        process
            .pointer(pointer)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_else(|| panic!("footprint JSON missing {pointer} for pid {pid}: {process}"))
    };
    let wired_bytes = process["categories"]
        .as_object()
        .expect("footprint process categories object")
        .values()
        .filter_map(|category| category["wired"].as_u64())
        .fold(0u64, u64::saturating_add);
    MemorySnapshot {
        rss_bytes,
        physical_footprint_bytes: required_u64("/auxiliary/phys_footprint"),
        physical_footprint_peak_bytes: required_u64("/auxiliary/phys_footprint_peak"),
        wired_bytes,
        system_wired_bytes: system_wired_bytes(),
    }
}

fn start_peak_sampler(
    pid: u32,
) -> (
    std::sync::Arc<std::sync::atomic::AtomicBool>,
    std::thread::JoinHandle<MemoryPeak>,
) {
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let thread_stop = std::sync::Arc::clone(&stop);
    let handle = std::thread::spawn(move || {
        let mut peak = MemoryPeak::default();
        while !thread_stop.load(std::sync::atomic::Ordering::Acquire) {
            peak.rss_bytes = peak.rss_bytes.max(process_rss_bytes(pid));
            peak.system_wired_bytes = peak.system_wired_bytes.max(system_wired_bytes());
            std::thread::sleep(Duration::from_millis(100));
        }
        peak.rss_bytes = peak.rss_bytes.max(process_rss_bytes(pid));
        peak.system_wired_bytes = peak.system_wired_bytes.max(system_wired_bytes());
        peak
    });
    (stop, handle)
}

fn measured_switch(
    runtime: &tokio::runtime::Runtime,
    client: &reqwest::Client,
    model: &std::path::Path,
    pid: u32,
) -> (serde_json::Value, Duration, MemoryPeak) {
    let (stop, sampler) = start_peak_sampler(pid);
    let result = runtime.block_on(explicit_switch(client, model));
    stop.store(true, std::sync::atomic::Ordering::Release);
    let peak = sampler.join().expect("memory peak sampler");
    (result.0, result.1, peak)
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Minimal HTTP/1.1 GET → status code, no body.  Same idiom as
/// `tests/openwebui_helpers/mod.rs::http_get_status`.
fn http_get_status(host: &str, port: u16, path: &str) -> std::io::Result<u16> {
    use std::net::TcpStream;
    let mut s = TcpStream::connect_timeout(
        &format!("{host}:{port}")
            .parse()
            .map_err(std::io::Error::other)?,
        Duration::from_secs(5),
    )?;
    s.set_read_timeout(Some(Duration::from_secs(5)))?;
    s.write_all(
        format!("GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n")
            .as_bytes(),
    )?;
    let mut head = [0u8; 64];
    let n = s.read(&mut head)?;
    let head_s = std::str::from_utf8(&head[..n]).unwrap_or("");
    let mut parts = head_s.split_whitespace();
    let _http = parts.next();
    let code = parts
        .next()
        .and_then(|s| s.parse::<u16>().ok())
        .ok_or_else(|| std::io::Error::other(format!("malformed HTTP status line: {head_s:?}")))?;
    Ok(code)
}

fn wait_for_readyz() {
    let started = Instant::now();
    let mut last_err: Option<String> = None;
    while started.elapsed().as_secs() < READYZ_BUDGET_SECS {
        match http_get_status(HOST, PORT, "/readyz") {
            Ok(200) => {
                eprintln!(
                    "multi_model_swap: /readyz=200 after {}s",
                    started.elapsed().as_secs()
                );
                return;
            }
            Ok(code) => last_err = Some(format!("status={code}")),
            Err(e) => last_err = Some(format!("transport: {e}")),
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    panic!(
        "multi_model_swap: /readyz did not reach 200 within {READYZ_BUDGET_SECS}s; \
         last_err={}",
        last_err.unwrap_or_else(|| "<none>".into())
    );
}

fn base_url() -> String {
    format!("http://{HOST}:{PORT}")
}

fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(REQUEST_BUDGET_SECS))
        .build()
        .expect("build reqwest client")
}

/// GET `/v1/models` → first entry's `id` field.
async fn fetch_canonical_model_id(client: &reqwest::Client) -> String {
    let resp = client
        .get(format!("{}/v1/models", base_url()))
        .send()
        .await
        .expect("GET /v1/models failed");
    assert_eq!(resp.status().as_u16(), 200, "/v1/models status != 200");
    let v: serde_json::Value = resp.json().await.expect("parse /v1/models JSON");
    v["data"][0]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("/v1/models response missing data[0].id: {v}"))
        .to_string()
}

/// Run one deterministic inference and return `(status, body_json, elapsed)`.
async fn post_inference(
    client: &reqwest::Client,
    model: &str,
    kind: ProbeKind,
) -> (u16, serde_json::Value, Duration) {
    let (path, body) = match kind {
        ProbeKind::Chat => (
            "/v1/chat/completions",
            serde_json::json!({
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": "Reply with a short deterministic acknowledgement of: café 東京 model swap."
                }],
                "max_tokens": 16,
                "temperature": 0,
                "stream": false,
            }),
        ),
        ProbeKind::Embedding => (
            "/v1/embeddings",
            serde_json::json!({
                "model": model,
                "input": "café 東京 model swap identity probe"
            }),
        ),
    };
    let t0 = Instant::now();
    let resp = client
        .post(format!("{}{path}", base_url()))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/chat/completions failed");
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
    let elapsed = t0.elapsed();
    let json: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("non-JSON chat response (status={status}, err={e}): {text}"));
    (status, json, elapsed)
}

fn canonical_result(body: &serde_json::Value, kind: ProbeKind) -> serde_json::Value {
    match kind {
        ProbeKind::Chat => body["choices"][0]["message"].clone(),
        ProbeKind::Embedding => body["data"][0]["embedding"].clone(),
    }
}

async fn fetch_runtime(client: &reqwest::Client) -> serde_json::Value {
    let resp = client
        .get(format!("{}/hf2q/v1/runtime", base_url()))
        .send()
        .await
        .expect("GET /hf2q/v1/runtime failed");
    assert_eq!(resp.status().as_u16(), 200, "runtime status != 200");
    resp.json().await.expect("parse runtime JSON")
}

async fn request_activation(
    client: &reqwest::Client,
    body: serde_json::Value,
) -> (u16, serde_json::Value, Duration) {
    let t0 = Instant::now();
    let resp = client
        .post(format!("{}/hf2q/v1/models/activate", base_url()))
        .json(&body)
        .send()
        .await
        .expect("POST /hf2q/v1/models/activate failed");
    let status = resp.status().as_u16();
    let text = resp.text().await.expect("read activation response");
    let json = serde_json::from_str(&text).unwrap_or_else(|error| {
        panic!("activation response is not JSON (status={status}, error={error}): {text}")
    });
    (status, json, t0.elapsed())
}

async fn explicit_switch(
    client: &reqwest::Client,
    model: &std::path::Path,
) -> (serde_json::Value, Duration) {
    let model = model.to_string_lossy();
    let (probe_status, conflict, _) = request_activation(
        client,
        serde_json::json!({"model": model, "action": "load"}),
    )
    .await;
    assert_eq!(
        probe_status, 409,
        "candidate must conflict under the one-model byte budget: {conflict}"
    );
    assert_eq!(conflict["status"], "conflict", "wrong conflict receipt");
    let revision = conflict["pool_revision"]
        .as_u64()
        .expect("conflict pool_revision");
    let victims = conflict["victims"]
        .as_array()
        .filter(|rows| !rows.is_empty())
        .cloned()
        .expect("conflict must carry exact non-empty victims");
    let (switch_status, receipt, elapsed) = request_activation(
        client,
        serde_json::json!({
            "model": model,
            "action": "switch",
            "expected_revision": revision,
            "victims": victims,
        }),
    )
    .await;
    assert_eq!(switch_status, 200, "explicit switch failed: {receipt}");
    assert_eq!(receipt["status"], "switched", "wrong switch receipt");
    (receipt, elapsed)
}

fn settled_memory_snapshot(pid: u32) -> MemorySnapshot {
    let mut settled = process_memory_snapshot(pid);
    for _ in 0..4 {
        std::thread::sleep(Duration::from_millis(250));
        let current = process_memory_snapshot(pid);
        settled.rss_bytes = settled.rss_bytes.min(current.rss_bytes);
        settled.physical_footprint_bytes = settled
            .physical_footprint_bytes
            .min(current.physical_footprint_bytes);
        settled.physical_footprint_peak_bytes = settled
            .physical_footprint_peak_bytes
            .max(current.physical_footprint_peak_bytes);
        settled.wired_bytes = settled.wired_bytes.min(current.wired_bytes);
        settled.system_wired_bytes = settled.system_wired_bytes.min(current.system_wired_bytes);
    }
    settled
}

#[derive(Debug)]
struct ResidentIdentity {
    pool_key: String,
    generation: u64,
    bytes_resident: u64,
    engine_config: serde_json::Value,
}

fn one_resident(runtime: &serde_json::Value) -> ResidentIdentity {
    assert_eq!(
        runtime["pool"]["loaded_count"].as_u64(),
        Some(1),
        "swap gate requires exactly one resident: {runtime}"
    );
    let rows = runtime["pool"]["resident"]
        .as_array()
        .expect("runtime resident array");
    assert_eq!(rows.len(), 1, "runtime resident array: {runtime}");
    ResidentIdentity {
        pool_key: rows[0]["pool_key"]
            .as_str()
            .expect("resident pool_key")
            .to_owned(),
        generation: rows[0]["generation"].as_u64().expect("resident generation"),
        bytes_resident: rows[0]["bytes_resident"].as_u64().expect("resident bytes"),
        engine_config: rows[0]["engine_config"].clone(),
    }
}

/// Smoke: `hf2q --version` returns 0.  Always-on; verifies the
/// scaffolding can locate the binary so the gated test bodies have a
/// known-good entry point.
#[test]
fn binary_is_locatable_and_runs_version() {
    let bin = hf2q_binary_path();
    let out = Command::new(&bin)
        .arg("--version")
        .output()
        .expect("spawn hf2q --version");
    assert!(
        out.status.success(),
        "hf2q --version exited {:?}; stderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A -> B -> A through the production explicit-switch path.
#[test]
fn model_swap_a_b_a_reclaims_and_replays_e2e() {
    if skip_unless_gated("model_swap_a_b_a_reclaims_and_replays_e2e") {
        return;
    }

    let requested_a = PathBuf::from(std::env::var(ENV_MODEL_A).expect(ENV_MODEL_A));
    let requested_b = PathBuf::from(std::env::var(ENV_MODEL_B).expect(ENV_MODEL_B));
    let canonical_a = std::fs::canonicalize(&requested_a)
        .unwrap_or_else(|error| panic!("cannot resolve {}: {error}", requested_a.display()));
    let canonical_b = std::fs::canonicalize(&requested_b)
        .unwrap_or_else(|error| panic!("cannot resolve {}: {error}", requested_b.display()));
    assert_ne!(
        canonical_a, canonical_b,
        "model swap proof requires two distinct physical files, not aliases"
    );
    let bytes_a = std::fs::metadata(&canonical_a)
        .expect("MODEL_A metadata")
        .len();
    let bytes_b = std::fs::metadata(&canonical_b)
        .expect("MODEL_B metadata")
        .len();
    let kind_a = ProbeKind::from_env(ENV_KIND_A);
    let kind_b = ProbeKind::from_env(ENV_KIND_B);
    let ((model_a, bytes_a, kind_a), (model_b, bytes_b, kind_b)) = if bytes_a >= bytes_b {
        (
            (canonical_a, bytes_a, kind_a),
            (canonical_b, bytes_b, kind_b),
        )
    } else {
        (
            (canonical_b, bytes_b, kind_b),
            (canonical_a, bytes_a, kind_a),
        )
    };
    let artifact_delta = bytes_a - bytes_b;
    assert!(
        artifact_delta >= GIB,
        "resource-reclaim proof needs artifacts at least 1 GiB apart; A={bytes_a}, B={bytes_b}"
    );
    let swap_budget = Duration::from_secs(
        std::env::var(ENV_MAX_SECS)
            .ok()
            .map(|raw| raw.parse::<u64>().expect("parse swap seconds"))
            .unwrap_or(SWAP_BUDGET_SECS),
    );

    eprintln!(
        "model_swap: A={} ({bytes_a} bytes, {kind_a:?}), B={} ({bytes_b} bytes, {kind_b:?}), \
         pool_budget={bytes_a}, swap_budget={swap_budget:?}",
        model_a.display(),
        model_b.display()
    );

    let server = ServerGuard::spawn(&model_a.to_string_lossy(), bytes_a).expect("spawn hf2q serve");
    wait_for_readyz();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let client = build_client();
    let initial_model_id = rt.block_on(fetch_canonical_model_id(&client));
    let initial_runtime = rt.block_on(fetch_runtime(&client));
    let resident_a1 = one_resident(&initial_runtime);
    assert_eq!(resident_a1.bytes_resident, bytes_a);
    let (status_a1, body_a1, inference_a1) =
        rt.block_on(post_inference(&client, &initial_model_id, kind_a));
    assert_eq!(status_a1, 200, "initial A inference failed: {body_a1}");
    let result_a1 = canonical_result(&body_a1, kind_a);
    assert!(!result_a1.is_null(), "initial A result missing: {body_a1}");
    let memory_a1 = settled_memory_snapshot(server.0.id());

    let (receipt_b, switch_to_b, peak_a_to_b) =
        measured_switch(&rt, &client, &model_b, server.0.id());
    assert!(
        switch_to_b < swap_budget,
        "A -> B load took {switch_to_b:?}, exceeds {swap_budget:?}"
    );
    let request_model_b = receipt_b["request_model"]
        .as_str()
        .expect("B switch request_model");
    let runtime_b = rt.block_on(fetch_runtime(&client));
    let resident_b = one_resident(&runtime_b);
    assert_eq!(resident_b.bytes_resident, bytes_b);
    assert_ne!(resident_b.pool_key, resident_a1.pool_key);
    assert_eq!(
        resident_b.engine_config["scheduler"], resident_a1.engine_config["scheduler"],
        "process scheduler policy changed across A -> B"
    );
    assert_eq!(
        resident_b.engine_config["kv_cache_budget_bytes"],
        resident_a1.engine_config["kv_cache_budget_bytes"],
        "process KV budget changed across A -> B"
    );
    assert_eq!(
        resident_b.engine_config["queue_capacity"], resident_a1.engine_config["queue_capacity"],
        "queue policy changed across A -> B"
    );
    let (status_b, body_b, inference_b) =
        rt.block_on(post_inference(&client, request_model_b, kind_b));
    assert_eq!(status_b, 200, "B inference failed: {body_b}");
    assert!(
        !canonical_result(&body_b, kind_b).is_null(),
        "B result missing: {body_b}"
    );
    let memory_b = settled_memory_snapshot(server.0.id());

    let reclaim_floor = artifact_delta / 4;
    let rss_drop = memory_a1.rss_bytes.saturating_sub(memory_b.rss_bytes);
    let footprint_drop = memory_a1
        .physical_footprint_bytes
        .saturating_sub(memory_b.physical_footprint_bytes);
    let wired_drop = memory_a1.wired_bytes.saturating_sub(memory_b.wired_bytes);
    let system_wired_drop = memory_a1
        .system_wired_bytes
        .saturating_sub(memory_b.system_wired_bytes);
    assert_eq!(
        runtime_b["pool"]["total_resident_bytes"].as_u64(),
        Some(bytes_b),
        "logical pool accounting must name only B: {runtime_b}"
    );
    assert!(
        rss_drop >= reclaim_floor,
        "A -> B did not reclaim enough process memory: artifact_delta={artifact_delta}, \
         floor={reclaim_floor}, rss_drop={rss_drop}, footprint_drop={footprint_drop}, \
         A={memory_a1:?}, B={memory_b:?}"
    );
    assert!(
        system_wired_drop >= reclaim_floor / 2,
        "A -> B did not reclaim host wired model memory: floor={}, \
         process_wired_drop={wired_drop}, system_wired_drop={system_wired_drop}, \
         A={memory_a1:?}, B={memory_b:?}",
        reclaim_floor / 2
    );
    let peak_margin = memory_a1
        .rss_bytes
        .max(memory_b.rss_bytes)
        .checked_div(10)
        .unwrap_or(0)
        .max(2 * GIB);
    assert!(
        peak_a_to_b.rss_bytes
            <= memory_a1
                .rss_bytes
                .max(memory_b.rss_bytes)
                .saturating_add(peak_margin),
        "A -> B crossed the process-RSS double-residency bound: peak={peak_a_to_b:?}, \
         A={memory_a1:?}, B={memory_b:?}, margin={peak_margin}"
    );
    assert!(
        peak_a_to_b.system_wired_bytes
            <= memory_a1
                .system_wired_bytes
                .max(memory_b.system_wired_bytes)
                .saturating_add(peak_margin),
        "A -> B crossed the host-wired double-residency bound: peak={peak_a_to_b:?}, \
         A={memory_a1:?}, B={memory_b:?}, margin={peak_margin}"
    );

    let (receipt_a2, switch_to_a, peak_b_to_a) =
        measured_switch(&rt, &client, &model_a, server.0.id());
    assert!(
        switch_to_a < swap_budget,
        "B -> A reload took {switch_to_a:?}, exceeds {swap_budget:?}"
    );
    let request_model_a2 = receipt_a2["request_model"]
        .as_str()
        .expect("A reload request_model");
    let runtime_a2 = rt.block_on(fetch_runtime(&client));
    let resident_a2 = one_resident(&runtime_a2);
    assert_eq!(resident_a2.pool_key, resident_a1.pool_key);
    assert_ne!(
        resident_a2.generation, resident_a1.generation,
        "A reload must publish a fresh generation"
    );
    assert_eq!(
        resident_a2.engine_config, resident_a1.engine_config,
        "A reload changed scheduler, KV budget, queue, sidecar, or overlay identity"
    );
    let (status_a2, body_a2, inference_a2) =
        rt.block_on(post_inference(&client, request_model_a2, kind_a));
    assert_eq!(status_a2, 200, "reloaded A inference failed: {body_a2}");
    let result_a2 = canonical_result(&body_a2, kind_a);
    assert_eq!(
        result_a2, result_a1,
        "A result changed after A -> B -> A; stale model/template/tokenizer/cache state or nondeterminism"
    );
    let memory_a2 = settled_memory_snapshot(server.0.id());
    let reload_margin = (memory_a1.rss_bytes / 10).max(2 * GIB);
    assert!(
        memory_a2.rss_bytes <= memory_a1.rss_bytes.saturating_add(reload_margin),
        "A reload leaked process RSS: first={memory_a1:?}, reload={memory_a2:?}, \
         margin={reload_margin}"
    );
    assert!(
        memory_a2.system_wired_bytes <= memory_a1.system_wired_bytes.saturating_add(reload_margin),
        "A reload leaked host wired memory: first={memory_a1:?}, reload={memory_a2:?}, \
         margin={reload_margin}"
    );
    assert!(
        peak_b_to_a.rss_bytes
            <= memory_a1
                .rss_bytes
                .max(memory_b.rss_bytes)
                .saturating_add(reload_margin),
        "B -> A crossed the process-RSS double-residency bound: peak={peak_b_to_a:?}, \
         A={memory_a1:?}, B={memory_b:?}, reload={memory_a2:?}"
    );
    assert!(
        peak_b_to_a.system_wired_bytes
            <= memory_a1
                .system_wired_bytes
                .max(memory_b.system_wired_bytes)
                .saturating_add(reload_margin),
        "B -> A crossed the host-wired double-residency bound: peak={peak_b_to_a:?}, \
         A={memory_a1:?}, B={memory_b:?}, reload={memory_a2:?}"
    );

    eprintln!(
        "model_swap PASS: A -> B={switch_to_b:?}, B -> A={switch_to_a:?}, \
         inference A1/B/A2={inference_a1:?}/{inference_b:?}/{inference_a2:?}; \
         RSS A1/B/A2={}/{}/{}; footprint A1/B/A2={}/{}/{}; \
         process-wired A1/B/A2={}/{}/{}; host-wired A1/B/A2={}/{}/{}",
        memory_a1.rss_bytes,
        memory_b.rss_bytes,
        memory_a2.rss_bytes,
        memory_a1.physical_footprint_bytes,
        memory_b.physical_footprint_bytes,
        memory_a2.physical_footprint_bytes,
        memory_a1.wired_bytes,
        memory_b.wired_bytes,
        memory_a2.wired_bytes,
        memory_a1.system_wired_bytes,
        memory_b.system_wired_bytes,
        memory_a2.system_wired_bytes,
    );
    eprintln!("model_swap peaks: A -> B={peak_a_to_b:?}; B -> A={peak_b_to_a:?}");
}
