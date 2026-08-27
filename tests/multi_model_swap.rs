//! ADR-047 real model-lifecycle swap proof.
//!
//! The old version of this test admitted a symlink to the same GGUF alongside
//! the startup model. That was a second pool key, but it was not a model swap:
//! neither the model bytes nor the tokenizer/template changed, no victim was
//! evicted, and no physical-memory release was proven.
//!
//! This gate requires two distinct physical GGUFs and exercises the production
//! revision-bound control plane in the sequence A -> B -> A. The caller's A/B
//! identity is preserved, and the pool byte budget is set to the larger file
//! size so either artifact fits alone while the pair cannot co-reside
//! logically. Each transition first obtains the exact conflict receipt, then
//! submits that receipt as an explicit switch.
//!
//! The proof covers pool-resident generative engines. Dedicated BERT/Nomic
//! embedding models have a separate process-global lifecycle and are not
//! counted by this gate. It asserts exact A-result replay after reload, a
//! different resident generation for the reloaded A, one
//! resident model throughout, load latency on both switch legs, exact live-file
//! ownership/reclaim for file-backed artifacts, and bounded process
//! RSS/physical-footprint/wired memory with no double-residency peak. This is
//! intentionally a real-hardware gate, not a hosted-safe smoke.
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
//!    harness against required `HF2Q_HOT_SWAP_E2E_MODEL_A` and
//!    `HF2Q_HOT_SWAP_E2E_MODEL_B` paths. Their SHA-256, raw GGUF
//!    architecture, and architecture-family variables are also required.
//!    The paths must resolve to distinct physical GGUFs; aliases and symlinks
//!    to A are rejected because they do not exercise model replacement.
//!
//! ```bash
//! HF2Q_HOT_SWAP_E2E=1 \
//!   HF2Q_HOT_SWAP_E2E_MODEL_A=/path/to/larger.gguf \
//!   HF2Q_HOT_SWAP_E2E_MODEL_B=/path/to/smaller.gguf \
//!   HF2Q_HOT_SWAP_E2E_MODEL_A_SHA256=<lowercase-sha256> \
//!   HF2Q_HOT_SWAP_E2E_MODEL_B_SHA256=<lowercase-sha256> \
//!   HF2Q_HOT_SWAP_E2E_MODEL_A_ARCHITECTURE=gemma4 \
//!   HF2Q_HOT_SWAP_E2E_MODEL_B_ARCHITECTURE=qwen35 \
//!   HF2Q_HOT_SWAP_E2E_MODEL_A_ARCH_FAMILY=gemma4 \
//!   HF2Q_HOT_SWAP_E2E_MODEL_B_ARCH_FAMILY=qwen35 \
//!   cargo test --release --test multi_model_swap -- --test-threads=1 --nocapture
//! ```

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

const ENV_GATE: &str = "HF2Q_HOT_SWAP_E2E";
const ENV_MODEL_A: &str = "HF2Q_HOT_SWAP_E2E_MODEL_A";
const ENV_MODEL_B: &str = "HF2Q_HOT_SWAP_E2E_MODEL_B";
const ENV_MODEL_A_SHA256: &str = "HF2Q_HOT_SWAP_E2E_MODEL_A_SHA256";
const ENV_MODEL_B_SHA256: &str = "HF2Q_HOT_SWAP_E2E_MODEL_B_SHA256";
const ENV_MODEL_A_ARCHITECTURE: &str = "HF2Q_HOT_SWAP_E2E_MODEL_A_ARCHITECTURE";
const ENV_MODEL_B_ARCHITECTURE: &str = "HF2Q_HOT_SWAP_E2E_MODEL_B_ARCHITECTURE";
const ENV_MODEL_A_ARCH_FAMILY: &str = "HF2Q_HOT_SWAP_E2E_MODEL_A_ARCH_FAMILY";
const ENV_MODEL_B_ARCH_FAMILY: &str = "HF2Q_HOT_SWAP_E2E_MODEL_B_ARCH_FAMILY";
const ENV_MAX_SECS: &str = "HF2Q_HOT_SWAP_E2E_MAX_SECS";
const ENV_EXACT_RECEIPT: &str = "HF2Q_HOT_SWAP_EXACT_RECEIPT";
const ENV_EXACT_PAIR_ID: &str = "HF2Q_HOT_SWAP_EXACT_PAIR_ID";
const ENV_EXACT_FORMAT_A: &str = "HF2Q_HOT_SWAP_EXACT_FORMAT_A";
const ENV_EXACT_FORMAT_B: &str = "HF2Q_HOT_SWAP_EXACT_FORMAT_B";
const ENV_EXACT_FILE_A: &str = "HF2Q_HOT_SWAP_EXACT_FILE_A";
const ENV_EXACT_FILE_B: &str = "HF2Q_HOT_SWAP_EXACT_FILE_B";
const ENV_EXACT_FILE_TYPE_A: &str = "HF2Q_HOT_SWAP_EXACT_FILE_TYPE_A";
const ENV_EXACT_FILE_TYPE_B: &str = "HF2Q_HOT_SWAP_EXACT_FILE_TYPE_B";
const ENV_EXACT_SOURCE_COMMIT: &str = "HF2Q_HOT_SWAP_EXACT_SOURCE_COMMIT";
const ENV_EXACT_BINARY_SHA256: &str = "HF2Q_HOT_SWAP_EXACT_BINARY_SHA256";
const ENV_EXACT_BINARY_GIT_COMMIT: &str = "HF2Q_HOT_SWAP_EXACT_BINARY_GIT_COMMIT";
const ENV_EXACT_MLX_VERSION: &str = "HF2Q_HOT_SWAP_EXACT_MLX_VERSION";
const ENV_EXACT_MLX_SOURCE: &str = "HF2Q_HOT_SWAP_EXACT_MLX_SOURCE";
const ENV_EXACT_MLX_CHECKSUM: &str = "HF2Q_HOT_SWAP_EXACT_MLX_CHECKSUM";
const ENV_EXECUTABLE: &str = "HF2Q_HOT_SWAP_EXECUTABLE";
const ENV_EXACT_CHAIN_RECEIPT: &str = "HF2Q_HOT_SWAP_EXACT_CHAIN_RECEIPT";
const ENV_EXACT_CHAIN_SPEC: &str = "HF2Q_HOT_SWAP_EXACT_CHAIN_SPEC";
const ENV_GENERATIVE_CHAIN_RECEIPT: &str = "HF2Q_GENERATIVE_SWAP_CHAIN_RECEIPT";
const ENV_GENERATIVE_CHAIN_SPEC: &str = "HF2Q_GENERATIVE_SWAP_CHAIN_SPEC";
const SWAP_SENTINEL: &str = "HF2Q_SWAP_OK";

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
const GENERATIVE_SWAP_BUDGET_SECS: u64 = 60;
const GIB: u64 = 1024 * 1024 * 1024;

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
         Their *_SHA256, *_ARCHITECTURE, and *_ARCH_FAMILY identities are also required. \
         Prefer scripts/run_generative_swap_matrix.sh. Optional: {ENV_MAX_SECS}."
    );
    true
}

/// Locate the `hf2q` binary the cargo test runner just built.
fn hf2q_binary_path() -> PathBuf {
    if let Some(path) = std::env::var_os(ENV_EXECUTABLE) {
        let path = PathBuf::from(path);
        assert!(
            path.is_absolute() && path.is_file(),
            "{ENV_EXECUTABLE} must name an absolute regular executable"
        );
        return path;
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExecutionReceipt {
    pool_key: String,
    generation: u64,
    text_artifact_sha256: String,
    arch_family: String,
    architecture: String,
}

fn required_receipt_header(resp: &reqwest::Response, name: &str) -> String {
    resp.headers()
        .get(name)
        .unwrap_or_else(|| panic!("completion is missing required {name} execution receipt"))
        .to_str()
        .unwrap_or_else(|error| panic!("completion {name} is not ASCII: {error}"))
        .to_owned()
}

fn execution_receipt(resp: &reqwest::Response) -> ExecutionReceipt {
    use base64::Engine as _;

    let encoded_pool = required_receipt_header(resp, "x-hf2q-execution-pool-key-b64");
    let pool_key = String::from_utf8(
        base64::engine::general_purpose::STANDARD
            .decode(encoded_pool)
            .expect("decode execution pool key"),
    )
    .expect("execution pool key UTF-8");
    ExecutionReceipt {
        pool_key,
        generation: required_receipt_header(resp, "x-hf2q-execution-generation")
            .parse()
            .expect("parse execution generation"),
        text_artifact_sha256: required_receipt_header(resp, "x-hf2q-execution-artifact-sha256"),
        arch_family: required_receipt_header(resp, "x-hf2q-execution-arch-family"),
        architecture: required_receipt_header(resp, "x-hf2q-execution-architecture"),
    }
}

/// Run one deterministic inference and return the execution-bound receipt in
/// addition to the OpenAI body. HTTP 200 without this receipt is not swap
/// proof: a stale route can return a valid completion from the wrong engine.
async fn post_inference(
    client: &reqwest::Client,
    model: &str,
) -> (u16, serde_json::Value, ExecutionReceipt, Duration) {
    post_inference_with_canary_and_limit(client, model, SWAP_SENTINEL, 16).await
}

async fn post_inference_with_canary(
    client: &reqwest::Client,
    model: &str,
    canary: &str,
) -> (u16, serde_json::Value, ExecutionReceipt, Duration) {
    post_inference_with_canary_and_limit(client, model, canary, 24).await
}

async fn post_inference_with_canary_and_limit(
    client: &reqwest::Client,
    model: &str,
    canary: &str,
    max_tokens: u64,
) -> (u16, serde_json::Value, ExecutionReceipt, Duration) {
    let path = "/v1/chat/completions";
    let body = serde_json::json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": format!("Reply with exactly {canary} and nothing else.")
        }],
        "max_tokens": max_tokens,
        "temperature": 0,
        "hf2q_enable_thinking": false,
        "stream": false,
    });
    let t0 = Instant::now();
    let resp = client
        .post(format!("{}{path}", base_url()))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/chat/completions failed");
    let status = resp.status().as_u16();
    let receipt = execution_receipt(&resp);
    let text = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
    let elapsed = t0.elapsed();
    let json: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("non-JSON chat response (status={status}, err={e}): {text}"));
    (status, json, receipt, elapsed)
}

fn required_expected_identity(prefix: &str) -> String {
    std::env::var(prefix).unwrap_or_else(|_| panic!("{prefix} is required for exact swap proof"))
}

fn assert_execution_receipt(
    receipt: &ExecutionReceipt,
    resident: &ResidentIdentity,
    expected_sha256: &str,
    expected_arch_family: &str,
    expected_architecture: &str,
    phase: &str,
) {
    assert_eq!(receipt.pool_key, resident.pool_key, "{phase}: pool key");
    assert_eq!(
        receipt.generation, resident.generation,
        "{phase}: generation"
    );
    assert_eq!(
        receipt.text_artifact_sha256, expected_sha256,
        "{phase}: artifact SHA-256"
    );
    assert_eq!(
        receipt.arch_family, expected_arch_family,
        "{phase}: architecture family"
    );
    assert_eq!(
        receipt.architecture, expected_architecture,
        "{phase}: GGUF architecture"
    );
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn sha256_file(path: &std::path::Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn sha256_json(value: &serde_json::Value) -> String {
    sha256_bytes(&serde_json::to_vec(value).expect("serialize swap receipt value"))
}

#[derive(Debug, Clone)]
struct ValidatedSwapResult {
    message: serde_json::Value,
    role: String,
    content: String,
    finish_reason: String,
    completion_tokens: u64,
    cached_tokens: u64,
}

fn validated_swap_result(body: &serde_json::Value, phase: &str) -> ValidatedSwapResult {
    validated_swap_result_for_canary(body, phase, SWAP_SENTINEL)
}

fn validated_swap_result_for_canary(
    body: &serde_json::Value,
    phase: &str,
    canary: &str,
) -> ValidatedSwapResult {
    let choice = &body["choices"][0];
    let message = choice["message"].clone();
    let role = message["role"]
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| panic!("{phase}: response missing assistant role: {body}"));
    let content = message["content"]
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| panic!("{phase}: response missing text content: {body}"));
    let finish_reason = choice["finish_reason"]
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| panic!("{phase}: response missing finish reason: {body}"));
    let completion_tokens = body["usage"]["completion_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("{phase}: response missing completion-token count: {body}"));
    let cached_tokens = body["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("{phase}: response missing cached-token count: {body}"));
    assert_eq!(role, "assistant", "{phase}: wrong response role");
    assert_eq!(content, canary, "{phase}: incoherent swap sentinel");
    assert_eq!(
        finish_reason, "stop",
        "{phase}: response did not finish cleanly"
    );
    assert!(
        completion_tokens > 0,
        "{phase}: no semantic token was generated"
    );
    assert_eq!(
        cached_tokens, 0,
        "{phase}: a fresh resident generation reused stale KV state"
    );
    ValidatedSwapResult {
        message,
        role,
        content,
        finish_reason,
        completion_tokens,
        cached_tokens,
    }
}

fn exact_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required for exact swap receipt"))
}

fn memory_receipt(snapshot: MemorySnapshot) -> serde_json::Value {
    serde_json::json!({
        "rss_bytes": snapshot.rss_bytes,
        "physical_footprint_bytes": snapshot.physical_footprint_bytes,
        "physical_footprint_peak_bytes": snapshot.physical_footprint_peak_bytes,
        "wired_bytes": snapshot.wired_bytes,
        "host_wired_bytes": snapshot.system_wired_bytes,
    })
}

fn phase_receipt(
    phase: &str,
    format: &str,
    resident: &ResidentIdentity,
    execution: &ExecutionReceipt,
    result: &ValidatedSwapResult,
) -> serde_json::Value {
    let pool_key_sha256 = sha256_bytes(resident.pool_key.as_bytes());
    assert_eq!(
        pool_key_sha256,
        sha256_bytes(execution.pool_key.as_bytes()),
        "{phase}: path-free pool-key digest"
    );
    serde_json::json!({
        "phase": phase,
        "format": format,
        "resident": {
            "pool_key_sha256": pool_key_sha256,
            "generation": resident.generation,
            "bytes": resident.bytes_resident,
            "engine_config_sha256": sha256_json(&resident.engine_config),
        },
        "execution": {
            "pool_key_sha256": sha256_bytes(execution.pool_key.as_bytes()),
            "generation": execution.generation,
            "artifact_sha256": execution.text_artifact_sha256,
            "arch_family": execution.arch_family,
            "architecture": execution.architecture,
        },
        "result_sha256": sha256_json(&result.message),
        "semantic": {
            "role": result.role.as_str(),
            "content": result.content.as_str(),
            "finish_reason": result.finish_reason.as_str(),
            "completion_tokens": result.completion_tokens,
            "cached_tokens": result.cached_tokens,
        },
    })
}

fn write_exact_receipt(path: &std::path::Path, receipt: &serde_json::Value) {
    use std::fs::OpenOptions;

    assert!(
        path.is_absolute(),
        "exact swap receipt path must be absolute"
    );
    assert!(
        !path.exists() && std::fs::symlink_metadata(path).is_err(),
        "exact swap receipt destination already exists: {}",
        path.display()
    );
    let parent = path.parent().expect("exact swap receipt parent");
    assert!(parent.is_dir(), "exact swap receipt parent is missing");
    let tmp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(std::ffi::OsStr::to_str)
            .expect("UTF-8 receipt name"),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .expect("create exact swap receipt temp file");
    serde_json::to_writer_pretty(&mut file, receipt).expect("write exact swap receipt");
    file.write_all(b"\n").expect("terminate exact swap receipt");
    file.sync_all().expect("sync exact swap receipt");
    drop(file);
    std::fs::rename(&tmp, path).expect("publish exact swap receipt");
}

fn semantic_ttft(body: &serde_json::Value) -> Duration {
    let milliseconds = body["x_hf2q_timing"]["time_to_first_token_ms"]
        .as_f64()
        .unwrap_or_else(|| panic!("completion is missing semantic TTFT: {body}"));
    assert!(
        milliseconds.is_finite() && milliseconds > 0.0,
        "completion semantic TTFT must be finite and positive: {milliseconds}"
    );
    Duration::from_secs_f64(milliseconds / 1000.0)
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
struct ArtifactMappingEvidence {
    path: PathBuf,
    inode: u64,
    lsof_live: bool,
    vmmap_live: bool,
}

impl ArtifactMappingEvidence {
    fn is_live(&self) -> bool {
        self.lsof_live || self.vmmap_live
    }
}

fn vmmap_contains_exact_artifact(vmmap_stdout: &str, artifact: &std::path::Path) -> bool {
    let artifact = artifact.to_string_lossy();
    vmmap_stdout.lines().any(|line| {
        let line = line.trim_end();
        let Some(prefix) = line.strip_suffix(artifact.as_ref()) else {
            return false;
        };
        prefix.is_empty()
            || prefix
                .as_bytes()
                .last()
                .is_some_and(|byte| byte.is_ascii_whitespace())
    })
}

#[test]
fn vmmap_artifact_match_rejects_a_basename_embedded_in_another_artifact() {
    let qwen = std::path::Path::new(
        "/opt/hf2q/models/generative-swap-text-only-v1/qwen-moe/APEX-Q5_K_M.gguf",
    );
    let qwen_line = concat!(
        "mapped file            104000000-108000000 [ 64.0M] r--/r-- SM=COW  ",
        "/opt/hf2q/models/generative-swap-text-only-v1/qwen-moe/APEX-Q5_K_M.gguf"
    );
    let gemma_line = concat!(
        "mapped file            104000000-108000000 [ 64.0M] r--/r-- SM=COW  ",
        "/opt/hf2q/models/generative-swap-text-only-v1/gemma/",
        "gemma4-ara-2pass-APEX-Q5_K_M.gguf"
    );

    assert!(vmmap_contains_exact_artifact(qwen_line, qwen));
    assert!(!vmmap_contains_exact_artifact(gemma_line, qwen));
}

/// Query both the process file table and virtual-memory map for one physical
/// artifact. A mapped GGUF may legitimately close its original file descriptor
/// while retaining file-backed pages, so absence is proven only when both
/// views agree. `lsof` selects by the exact canonical file identity (including
/// inode); `vmmap -wide` is the path-visible fallback for descriptor-free maps.
fn artifact_mapping_evidence(pid: u32, artifact: &std::path::Path) -> ArtifactMappingEvidence {
    use std::os::unix::fs::MetadataExt;

    let path = std::fs::canonicalize(artifact)
        .unwrap_or_else(|error| panic!("cannot canonicalize {}: {error}", artifact.display()));
    let inode = std::fs::metadata(&path)
        .unwrap_or_else(|error| panic!("cannot stat {}: {error}", path.display()))
        .ino();
    let lsof = Command::new("lsof")
        .args(["-nP", "-a", "-p", &pid.to_string(), "--"])
        .arg(&path)
        .output()
        .expect("run lsof for artifact mapping");
    let lsof_live = lsof.status.success();
    assert!(
        lsof_live || lsof.status.code() == Some(1),
        "lsof failed for pid {pid}, artifact {}: status={:?}, stderr={}",
        path.display(),
        lsof.status,
        String::from_utf8_lossy(&lsof.stderr)
    );

    let vmmap = Command::new("vmmap")
        .args(["-wide", &pid.to_string()])
        .output()
        .expect("run vmmap for artifact mapping");
    assert!(
        vmmap.status.success(),
        "vmmap failed for pid {pid}, artifact {}: {}",
        path.display(),
        String::from_utf8_lossy(&vmmap.stderr)
    );
    let vmmap_stdout = String::from_utf8_lossy(&vmmap.stdout);
    // `vmmap -wide` prints the complete mapped-file path. Match that path as
    // a whole trailing field: basename substring matching can otherwise call
    // APEX-Q5_K_M.gguf live while the mapped file is the distinct
    // gemma4-ara-2pass-APEX-Q5_K_M.gguf artifact.
    let vmmap_live = vmmap_contains_exact_artifact(&vmmap_stdout, &path);

    ArtifactMappingEvidence {
        path,
        inode,
        lsof_live,
        vmmap_live,
    }
}

fn assert_artifact_mapping_state(
    pid: u32,
    present: &std::path::Path,
    absent: &std::path::Path,
    phase: &str,
    require_present_mapping: bool,
) -> (ArtifactMappingEvidence, ArtifactMappingEvidence) {
    let present = artifact_mapping_evidence(pid, present);
    let absent = artifact_mapping_evidence(pid, absent);
    assert_ne!(
        present.path, absent.path,
        "{phase}: artifacts resolved to the same canonical path"
    );
    assert_ne!(
        present.inode, absent.inode,
        "{phase}: artifacts are aliases of the same physical file"
    );
    if require_present_mapping {
        assert!(
            present.is_live(),
            "{phase}: file-backed resident artifact has no live open/mapped-file ownership: {present:?}"
        );
    }
    assert!(
        !absent.is_live(),
        "{phase}: evicted artifact still has live open/mapped-file ownership: {absent:?}"
    );
    (present, absent)
}

#[derive(Debug, Clone)]
struct ResidentIdentity {
    pool_key: String,
    quant: String,
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
        quant: rows[0]["quant"]
            .as_str()
            .expect("resident quant")
            .to_owned(),
        generation: rows[0]["generation"].as_u64().expect("resident generation"),
        bytes_resident: rows[0]["bytes_resident"].as_u64().expect("resident bytes"),
        engine_config: rows[0]["engine_config"].clone(),
    }
}

#[derive(Debug, Clone)]
struct ExactChainArtifact {
    format: String,
    path: PathBuf,
    file: String,
    bytes: u64,
    sha256: String,
    gguf_file_type: u64,
}

fn exact_chain_artifacts() -> Vec<ExactChainArtifact> {
    let spec = exact_env(ENV_EXACT_CHAIN_SPEC);
    let value: serde_json::Value =
        serde_json::from_str(&spec).expect("parse exact five-format chain specification");
    let rows = value["artifacts"]
        .as_array()
        .expect("exact chain specification artifacts");
    assert_eq!(
        rows.len(),
        5,
        "exact chain specification must name five artifacts"
    );
    rows.iter()
        .map(|row| ExactChainArtifact {
            format: row["format"]
                .as_str()
                .expect("chain artifact format")
                .to_owned(),
            path: PathBuf::from(row["path"].as_str().expect("chain artifact path")),
            file: row["file"]
                .as_str()
                .expect("chain artifact file")
                .to_owned(),
            bytes: row["bytes"].as_u64().expect("chain artifact bytes"),
            sha256: row["sha256"]
                .as_str()
                .expect("chain artifact SHA-256")
                .to_owned(),
            gguf_file_type: row["gguf_file_type"]
                .as_u64()
                .expect("chain artifact GGUF file type"),
        })
        .collect()
}

fn exact_chain_storage_state(
    pid: u32,
    artifacts: &[ExactChainArtifact],
    current: usize,
    phase: &str,
) -> &'static str {
    let mut current_live = false;
    for (index, artifact) in artifacts.iter().enumerate() {
        let evidence = artifact_mapping_evidence(pid, &artifact.path);
        if index == current {
            current_live = evidence.is_live();
        } else {
            assert!(
                !evidence.is_live(),
                "{phase}: evicted {} artifact still has live ownership: {evidence:?}",
                artifact.format
            );
        }
    }
    if current_live {
        "file_backed"
    } else {
        "anonymous_accounted"
    }
}

#[derive(Debug)]
struct ExactChainObservation {
    resident: ResidentIdentity,
    execution: ExecutionReceipt,
    result: ValidatedSwapResult,
    memory: MemorySnapshot,
    storage: &'static str,
}

fn observe_exact_chain_phase(
    runtime: &tokio::runtime::Runtime,
    client: &reqwest::Client,
    request_model: &str,
    artifact: &ExactChainArtifact,
    artifact_index: usize,
    artifacts: &[ExactChainArtifact],
    phase: &str,
    pid: u32,
) -> ExactChainObservation {
    let state = runtime.block_on(fetch_runtime(client));
    let resident = one_resident(&state);
    assert_eq!(
        resident.bytes_resident, artifact.bytes,
        "{phase}: resident bytes"
    );
    assert_eq!(
        state["pool"]["total_resident_bytes"].as_u64(),
        Some(artifact.bytes),
        "{phase}: logical pool accounting"
    );
    let (status, body, execution, _) = runtime.block_on(post_inference(client, request_model));
    assert_eq!(status, 200, "{phase}: exact chain inference failed: {body}");
    assert_execution_receipt(
        &execution,
        &resident,
        &artifact.sha256,
        "qwen35",
        "qwen35",
        phase,
    );
    let result = validated_swap_result(&body, phase);
    let _semantic_ttft = semantic_ttft(&body);
    let memory = settled_memory_snapshot(pid);
    let storage = exact_chain_storage_state(pid, artifacts, artifact_index, phase);
    ExactChainObservation {
        resident,
        execution,
        result,
        memory,
        storage,
    }
}

fn memory_margin(bytes: u64) -> u64 {
    (bytes / 10).max(2 * GIB)
}

fn single_resident_host_wired_bound(
    previous: MemorySnapshot,
    current: MemorySnapshot,
    destination_artifact_bytes: u64,
    margin: u64,
) -> u64 {
    let destination_bound = previous
        .system_wired_bytes
        .min(current.system_wired_bytes)
        .saturating_add(destination_artifact_bytes)
        .saturating_add(margin);
    let previous_bound = previous.system_wired_bytes.saturating_add(margin);
    let current_bound = current.system_wired_bytes.saturating_add(margin);
    destination_bound.max(previous_bound).max(current_bound)
}

fn replay_host_wired_bound(
    phase_host_wired_bytes: impl IntoIterator<Item = u64>,
    replay_artifact_bytes: u64,
    margin: u64,
) -> u64 {
    phase_host_wired_bytes
        .into_iter()
        .min()
        .expect("replay bound requires at least one phase")
        .saturating_add(replay_artifact_bytes)
        .saturating_add(margin)
}

#[test]
fn host_wired_bound_allows_one_destination_artifact_but_not_two() {
    let snapshot = |system_wired_bytes| MemorySnapshot {
        rss_bytes: 4 * GIB,
        physical_footprint_bytes: 4 * GIB,
        physical_footprint_peak_bytes: 4 * GIB,
        wired_bytes: 0,
        system_wired_bytes,
    };
    let previous = snapshot(8 * GIB);
    let current = snapshot(9 * GIB);
    let destination = 54 * GIB;
    let previous_artifact = 29 * GIB;
    let margin = 2 * GIB;
    let bound = single_resident_host_wired_bound(previous, current, destination, margin);

    assert_eq!(bound, 64 * GIB);
    assert!(8 * GIB + destination <= bound);
    assert!(8 * GIB + previous_artifact + destination > bound);
    assert_eq!(
        replay_host_wired_bound([9 * GIB, 8 * GIB], destination, margin),
        bound
    );

    let source_dominant =
        single_resident_host_wired_bound(snapshot(108 * GIB), snapshot(8 * GIB), 25 * GIB, margin);
    assert_eq!(source_dominant, 110 * GIB);
    assert!(108 * GIB <= source_dominant);
    assert!(8 * GIB + 107 * GIB + 25 * GIB > source_dominant);
}

fn assert_chain_transition_memory(
    previous: MemorySnapshot,
    current: MemorySnapshot,
    peak: MemoryPeak,
    destination_artifact_bytes: u64,
    phase: &str,
) -> (u64, u64) {
    let margin = memory_margin(previous.rss_bytes.max(current.rss_bytes));
    let rss_bound = previous
        .rss_bytes
        .max(current.rss_bytes)
        .saturating_add(margin);
    let host_wired_bound =
        single_resident_host_wired_bound(previous, current, destination_artifact_bytes, margin);
    assert!(
        peak.rss_bytes <= rss_bound,
        "{phase}: process RSS crossed the no-double-residency bound: \
         peak={} bound={} before={} after={} destination={} margin={}",
        peak.rss_bytes,
        rss_bound,
        previous.rss_bytes,
        current.rss_bytes,
        destination_artifact_bytes,
        margin,
    );
    assert!(
        peak.system_wired_bytes <= host_wired_bound,
        "{phase}: host wired memory crossed the no-double-residency bound: \
         peak={} bound={} before={} after={} destination={} margin={}",
        peak.system_wired_bytes,
        host_wired_bound,
        previous.system_wired_bytes,
        current.system_wired_bytes,
        destination_artifact_bytes,
        margin,
    );
    (rss_bound, host_wired_bound)
}

fn chain_memory_receipt(snapshot: MemorySnapshot) -> serde_json::Value {
    memory_receipt(snapshot)
}

fn executed_build_info(binary: &std::path::Path) -> serde_json::Value {
    let output = Command::new(binary)
        .arg("__build-info")
        .output()
        .expect("read executed hf2q build provenance");
    assert!(
        output.status.success(),
        "executed hf2q does not expose build provenance: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse executed hf2q build provenance");
    assert_eq!(value["schema"], "hf2q.build-info.v1");
    value
}

fn exact_chain_phase_receipt(
    index: usize,
    phase: &str,
    process_pid: u32,
    artifact: &ExactChainArtifact,
    observation: &ExactChainObservation,
) -> serde_json::Value {
    let mut receipt = phase_receipt(
        phase,
        &artifact.format,
        &observation.resident,
        &observation.execution,
        &observation.result,
    );
    receipt["index"] = serde_json::json!(index);
    receipt["process_pid"] = serde_json::json!(process_pid);
    receipt["artifact"] = serde_json::json!({
        "format": artifact.format.as_str(),
        "file": artifact.file.as_str(),
        "bytes": artifact.bytes,
        "sha256": artifact.sha256.as_str(),
        "gguf_file_type": artifact.gguf_file_type,
    });
    receipt["memory"] = chain_memory_receipt(observation.memory);
    receipt["storage"] = serde_json::json!(observation.storage);
    receipt
}

#[derive(Debug, Clone)]
struct GenerativeChainArtifact {
    id: String,
    architecture: String,
    arch_family: String,
    path: PathBuf,
    file: String,
    bytes: u64,
    sha256: String,
    canary: String,
}

fn generative_chain_spec() -> (Vec<GenerativeChainArtifact>, Vec<String>, u64) {
    let spec = exact_env(ENV_GENERATIVE_CHAIN_SPEC);
    let value: serde_json::Value =
        serde_json::from_str(&spec).expect("parse generative chain specification");
    let rows = value["artifacts"]
        .as_array()
        .expect("generative chain specification artifacts");
    assert_eq!(rows.len(), 4, "generative chain requires four artifacts");
    let artifacts = rows
        .iter()
        .map(|row| GenerativeChainArtifact {
            id: row["id"]
                .as_str()
                .expect("generative artifact id")
                .to_owned(),
            architecture: row["architecture"]
                .as_str()
                .expect("generative artifact architecture")
                .to_owned(),
            arch_family: row["arch_family"]
                .as_str()
                .expect("generative artifact family")
                .to_owned(),
            path: PathBuf::from(row["path"].as_str().expect("generative artifact path")),
            file: row["file"]
                .as_str()
                .expect("generative artifact file")
                .to_owned(),
            bytes: row["bytes"].as_u64().expect("generative artifact bytes"),
            sha256: row["sha256"]
                .as_str()
                .expect("generative artifact SHA-256")
                .to_owned(),
            canary: row["canary"]
                .as_str()
                .expect("generative artifact canary")
                .to_owned(),
        })
        .collect::<Vec<_>>();
    let sequence = value["sequence"]
        .as_array()
        .expect("generative chain sequence")
        .iter()
        .map(|id| id.as_str().expect("generative sequence id").to_owned())
        .collect::<Vec<_>>();
    let load_budget_seconds = value["load_budget_seconds"]
        .as_u64()
        .expect("generative chain load budget");
    (artifacts, sequence, load_budget_seconds)
}

fn generative_chain_storage_state(
    pid: u32,
    artifacts: &[GenerativeChainArtifact],
    current: usize,
    phase: &str,
) -> &'static str {
    let mut current_live = false;
    for (index, artifact) in artifacts.iter().enumerate() {
        let evidence = artifact_mapping_evidence(pid, &artifact.path);
        if index == current {
            current_live = evidence.is_live();
        } else {
            assert!(
                !evidence.is_live(),
                "{phase}: evicted {} artifact still has live ownership: {evidence:?}",
                artifact.id
            );
        }
    }
    if current_live {
        "file_backed"
    } else {
        "anonymous_accounted"
    }
}

#[derive(Debug)]
struct GenerativeChainObservation {
    resident: ResidentIdentity,
    execution: ExecutionReceipt,
    result: ValidatedSwapResult,
    process_policy: serde_json::Value,
    process_policy_sha256: String,
    q5_canonical_route_dispatches: u64,
    memory: MemorySnapshot,
    storage: &'static str,
}

fn runtime_process_policy(runtime: &serde_json::Value, phase: &str) -> (serde_json::Value, String) {
    let policy = runtime["process_policy"].clone();
    assert_eq!(
        policy["schema_version"].as_u64(),
        Some(1),
        "{phase}: typed process policy schema"
    );
    let sha256 = runtime["process_policy_sha256"]
        .as_str()
        .unwrap_or_else(|| panic!("{phase}: process policy SHA-256"))
        .to_owned();
    assert_eq!(
        sha256_json(&policy),
        sha256,
        "{phase}: process policy hash must bind the emitted typed policy"
    );
    (policy, sha256)
}

fn runtime_q5_canonical_dispatches(runtime: &serde_json::Value, phase: &str) -> u64 {
    assert_eq!(
        runtime["routing_observation"]["q5_canonical_route"].as_str(),
        Some("dense_q5k_canonical_q4x4"),
        "{phase}: canonical Q5 route identity"
    );
    runtime["routing_observation"]["q5_canonical_route_dispatches"]
        .as_u64()
        .unwrap_or_else(|| panic!("{phase}: canonical Q5 dispatch counter"))
}

fn observe_generative_chain_phase(
    runtime: &tokio::runtime::Runtime,
    client: &reqwest::Client,
    request_model: &str,
    artifact: &GenerativeChainArtifact,
    artifact_index: usize,
    artifacts: &[GenerativeChainArtifact],
    phase: &str,
    pid: u32,
) -> GenerativeChainObservation {
    let state_before = runtime.block_on(fetch_runtime(client));
    let resident = one_resident(&state_before);
    let (process_policy_before, process_policy_sha256_before) =
        runtime_process_policy(&state_before, phase);
    let q5_dispatches_before = runtime_q5_canonical_dispatches(&state_before, phase);
    assert_eq!(
        resident.bytes_resident, artifact.bytes,
        "{phase}: text-only generative gate admitted sidecar/projector bytes"
    );
    assert_eq!(
        state_before["pool"]["total_resident_bytes"].as_u64(),
        Some(artifact.bytes),
        "{phase}: logical pool accounting"
    );
    let (status, body, execution, _) = runtime.block_on(post_inference_with_canary(
        client,
        request_model,
        &artifact.canary,
    ));
    assert_eq!(
        status, 200,
        "{phase}: generative chain inference failed: {body}"
    );
    assert_execution_receipt(
        &execution,
        &resident,
        &artifact.sha256,
        &artifact.arch_family,
        &artifact.architecture,
        phase,
    );
    let result = validated_swap_result_for_canary(&body, phase, &artifact.canary);
    let _semantic_ttft = semantic_ttft(&body);
    let state_after = runtime.block_on(fetch_runtime(client));
    let resident_after = one_resident(&state_after);
    assert_eq!(
        resident_after.pool_key, resident.pool_key,
        "{phase}: resident changed during inference"
    );
    assert_eq!(
        resident_after.generation, resident.generation,
        "{phase}: generation changed during inference"
    );
    let (process_policy, process_policy_sha256) = runtime_process_policy(&state_after, phase);
    assert_eq!(
        process_policy, process_policy_before,
        "{phase}: process policy changed during inference"
    );
    assert_eq!(
        process_policy_sha256, process_policy_sha256_before,
        "{phase}: process policy hash changed during inference"
    );
    let q5_dispatches_after = runtime_q5_canonical_dispatches(&state_after, phase);
    assert!(
        q5_dispatches_after >= q5_dispatches_before,
        "{phase}: canonical Q5 dispatch counter regressed"
    );
    let q5_canonical_route_dispatches = q5_dispatches_after - q5_dispatches_before;
    let memory = settled_memory_snapshot(pid);
    let storage = generative_chain_storage_state(pid, artifacts, artifact_index, phase);
    GenerativeChainObservation {
        resident,
        execution,
        result,
        process_policy,
        process_policy_sha256,
        q5_canonical_route_dispatches,
        memory,
        storage,
    }
}

fn generative_chain_phase_receipt(
    index: usize,
    phase: &str,
    process_pid: u32,
    artifact: &GenerativeChainArtifact,
    observation: &GenerativeChainObservation,
) -> serde_json::Value {
    let mut receipt = phase_receipt(
        phase,
        &artifact.id,
        &observation.resident,
        &observation.execution,
        &observation.result,
    );
    receipt["index"] = serde_json::json!(index);
    receipt["process_pid"] = serde_json::json!(process_pid);
    receipt["artifact"] = serde_json::json!({
        "id": artifact.id.as_str(),
        "architecture": artifact.architecture.as_str(),
        "arch_family": artifact.arch_family.as_str(),
        "file": artifact.file.as_str(),
        "bytes": artifact.bytes,
        "sha256": artifact.sha256.as_str(),
        "canary": artifact.canary.as_str(),
    });
    receipt["process_policy"] = observation.process_policy.clone();
    receipt["process_policy_sha256"] =
        serde_json::json!(observation.process_policy_sha256.as_str());
    // The qualified Qwen MoE artifact carries dense/shared Q5_K tensors.
    // The staged Gemma Q5_K_M-labelled artifact does not; do not fabricate a
    // route claim from its container-level quant identity.
    let q5_route_required =
        artifact.arch_family == "qwen35" && observation.resident.quant == "Q5_K_M";
    let q5_policy_enabled = observation.process_policy["ggml_routing"]["dense_q5k_canonical_q4x4"]
        .as_bool()
        .expect("typed Q5 routing policy");
    if q5_route_required {
        assert!(
            q5_policy_enabled,
            "{phase}: Q5_K_M artifact requires canonical Q5 policy"
        );
        assert!(
            observation.q5_canonical_route_dispatches > 0,
            "{phase}: Q5_K_M inference encoded no canonical Q5 dispatch"
        );
    }
    receipt["q5_route"] = if q5_route_required {
        serde_json::json!({
            "policy_enabled": q5_policy_enabled,
            "route": "dense_q5k_canonical_q4x4",
            "route_observed": true,
        })
    } else {
        serde_json::json!({
            "policy_enabled": q5_policy_enabled,
            "route": "N/A",
            "route_observed": false,
        })
    };
    receipt["memory"] = memory_receipt(observation.memory);
    receipt["storage"] = serde_json::json!(observation.storage);
    receipt
}

#[test]
fn execution_receipt_join_rejects_stale_generation_artifact_and_family() {
    let resident = ResidentIdentity {
        pool_key: "model/a@Q4_K_M".to_owned(),
        quant: "Q4_K_M".to_owned(),
        generation: 7,
        bytes_resident: 1,
        engine_config: serde_json::json!({}),
    };
    let receipt = ExecutionReceipt {
        pool_key: resident.pool_key.clone(),
        generation: resident.generation,
        text_artifact_sha256: "a".repeat(64),
        arch_family: "qwen35".to_owned(),
        architecture: "qwen35".to_owned(),
    };
    assert_execution_receipt(
        &receipt,
        &resident,
        &"a".repeat(64),
        "qwen35",
        "qwen35",
        "fixture",
    );

    let mut mutations = Vec::new();
    let mut stale_pool = receipt.clone();
    stale_pool.pool_key = "model/b@Q4_K_M".to_owned();
    mutations.push(stale_pool);
    let mut stale_generation = receipt.clone();
    stale_generation.generation += 1;
    mutations.push(stale_generation);
    let mut wrong_artifact = receipt.clone();
    wrong_artifact.text_artifact_sha256 = "b".repeat(64);
    mutations.push(wrong_artifact);
    let mut wrong_family = receipt.clone();
    wrong_family.arch_family = "gemma4".to_owned();
    mutations.push(wrong_family);
    let mut wrong_architecture = receipt.clone();
    wrong_architecture.architecture = "qwen35moe".to_owned();
    mutations.push(wrong_architecture);

    for mutation in mutations {
        assert!(
            std::panic::catch_unwind(|| {
                assert_execution_receipt(
                    &mutation,
                    &resident,
                    &"a".repeat(64),
                    "qwen35",
                    "qwen35",
                    "mutated fixture",
                )
            })
            .is_err(),
            "mutated execution receipt passed: {mutation:?}"
        );
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
    let expected_a_sha256 = required_expected_identity(ENV_MODEL_A_SHA256);
    let expected_b_sha256 = required_expected_identity(ENV_MODEL_B_SHA256);
    let expected_a_architecture = required_expected_identity(ENV_MODEL_A_ARCHITECTURE);
    let expected_b_architecture = required_expected_identity(ENV_MODEL_B_ARCHITECTURE);
    let expected_a_arch_family = required_expected_identity(ENV_MODEL_A_ARCH_FAMILY);
    let expected_b_arch_family = required_expected_identity(ENV_MODEL_B_ARCH_FAMILY);
    for digest in [&expected_a_sha256, &expected_b_sha256] {
        assert!(
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "expected artifact identity must be lowercase SHA-256: {digest}"
        );
    }
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
    let model_a = canonical_a;
    let model_b = canonical_b;
    let pool_budget = bytes_a.max(bytes_b);
    let swap_budget = Duration::from_secs(
        std::env::var(ENV_MAX_SECS)
            .ok()
            .map(|raw| raw.parse::<u64>().expect("parse swap seconds"))
            .unwrap_or(SWAP_BUDGET_SECS),
    );

    eprintln!(
        "model_swap: A={} ({bytes_a} bytes), B={} ({bytes_b} bytes), \
         pool_budget={pool_budget}, swap_budget={swap_budget:?}",
        model_a.display(),
        model_b.display()
    );

    let server =
        ServerGuard::spawn(&model_a.to_string_lossy(), pool_budget).expect("spawn hf2q serve");
    wait_for_readyz();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let client = build_client();
    let initial_model_id = rt.block_on(fetch_canonical_model_id(&client));
    let initial_runtime = rt.block_on(fetch_runtime(&client));
    let resident_a1 = one_resident(&initial_runtime);
    assert_eq!(resident_a1.bytes_resident, bytes_a);
    let (status_a1, body_a1, execution_a1, inference_a1) =
        rt.block_on(post_inference(&client, &initial_model_id));
    assert_eq!(status_a1, 200, "initial A inference failed: {body_a1}");
    assert_execution_receipt(
        &execution_a1,
        &resident_a1,
        &expected_a_sha256,
        &expected_a_arch_family,
        &expected_a_architecture,
        "A1",
    );
    let result_a1 = validated_swap_result(&body_a1, "A1");
    let ttft_a1 = semantic_ttft(&body_a1);
    let memory_a1 = settled_memory_snapshot(server.0.id());
    let mappings_a1 = assert_artifact_mapping_state(server.0.id(), &model_a, &model_b, "A1", true);

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
    assert_ne!(
        resident_b.generation, resident_a1.generation,
        "B switch must publish a fresh resident generation"
    );
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
    assert_eq!(
        resident_b.engine_config["warmup_synchronously"],
        resident_a1.engine_config["warmup_synchronously"],
        "warmup policy changed across A -> B"
    );
    assert_eq!(
        resident_b.engine_config["kv_metrics_sink"], resident_a1.engine_config["kv_metrics_sink"],
        "KV metrics wiring changed across A -> B"
    );
    let (status_b, body_b, execution_b, inference_b) =
        rt.block_on(post_inference(&client, request_model_b));
    assert_eq!(status_b, 200, "B inference failed: {body_b}");
    assert_execution_receipt(
        &execution_b,
        &resident_b,
        &expected_b_sha256,
        &expected_b_arch_family,
        &expected_b_architecture,
        "B",
    );
    let result_b = validated_swap_result(&body_b, "B");
    let ttft_b = semantic_ttft(&body_b);
    let memory_b = settled_memory_snapshot(server.0.id());
    // Detect the resident representation from OS ownership evidence. A future
    // native-mapped B automatically enters the file-backed branch and must be
    // absent again after B -> A. A copied B has no file map; in that case its
    // fresh generation, successful model-specific inference, and exact pool
    // byte accounting are the explicit anonymous-storage proof.
    let mappings_b = assert_artifact_mapping_state(server.0.id(), &model_b, &model_a, "B", false);

    assert_eq!(
        runtime_b["pool"]["total_resident_bytes"].as_u64(),
        Some(bytes_b),
        "logical pool accounting must name only B: {runtime_b}"
    );
    let b_storage_contract = if mappings_b.0.is_live() {
        "file_backed"
    } else {
        assert!(
            memory_b.rss_bytes > 0 && memory_b.physical_footprint_bytes > 0,
            "anonymous B storage has no physical-memory accounting: {memory_b:?}"
        );
        "anonymous_accounted"
    };
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
            <= single_resident_host_wired_bound(memory_a1, memory_b, bytes_b, peak_margin),
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
    let (status_a2, body_a2, execution_a2, inference_a2) =
        rt.block_on(post_inference(&client, request_model_a2));
    assert_eq!(status_a2, 200, "reloaded A inference failed: {body_a2}");
    assert_execution_receipt(
        &execution_a2,
        &resident_a2,
        &expected_a_sha256,
        &expected_a_arch_family,
        &expected_a_architecture,
        "A2",
    );
    let result_a2 = validated_swap_result(&body_a2, "A2");
    let ttft_a2 = semantic_ttft(&body_a2);
    assert_eq!(
        result_a2.message, result_a1.message,
        "A result changed after A -> B -> A; stale model/template/tokenizer/cache state or nondeterminism"
    );
    let memory_a2 = settled_memory_snapshot(server.0.id());
    let mappings_a2 = assert_artifact_mapping_state(server.0.id(), &model_a, &model_b, "A2", true);
    let reload_margin = (memory_a1.rss_bytes / 10).max(2 * GIB);
    assert!(
        memory_a2.rss_bytes <= memory_a1.rss_bytes.saturating_add(reload_margin),
        "A reload leaked process RSS: first={memory_a1:?}, reload={memory_a2:?}, \
         margin={reload_margin}"
    );
    let footprint_reload_margin = (memory_a1.physical_footprint_bytes / 10).max(2 * GIB);
    assert!(
        memory_a2.physical_footprint_bytes
            <= memory_a1
                .physical_footprint_bytes
                .saturating_add(footprint_reload_margin),
        "A reload leaked physical footprint: first={memory_a1:?}, reload={memory_a2:?}, \
         margin={footprint_reload_margin}"
    );
    let wired_reload_margin = (memory_a1.wired_bytes / 10).max(2 * GIB);
    assert!(
        memory_a2.wired_bytes <= memory_a1.wired_bytes.saturating_add(wired_reload_margin),
        "A reload leaked process wired memory: first={memory_a1:?}, reload={memory_a2:?}, \
         margin={wired_reload_margin}"
    );
    let host_wired_replay_bound = replay_host_wired_bound(
        [
            memory_a1.system_wired_bytes,
            memory_b.system_wired_bytes,
            memory_a2.system_wired_bytes,
        ],
        bytes_a,
        reload_margin,
    );
    assert!(
        memory_a2.system_wired_bytes <= host_wired_replay_bound,
        "A reload leaked host wired memory: first={memory_a1:?}, reload={memory_a2:?}, \
         bound={host_wired_replay_bound}"
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
            <= single_resident_host_wired_bound(memory_b, memory_a2, bytes_a, reload_margin),
        "B -> A crossed the host-wired double-residency bound: peak={peak_b_to_a:?}, \
         A={memory_a1:?}, B={memory_b:?}, reload={memory_a2:?}"
    );

    if let Ok(receipt_path) = std::env::var(ENV_EXACT_RECEIPT) {
        let pair_id = exact_env(ENV_EXACT_PAIR_ID);
        let format_a = exact_env(ENV_EXACT_FORMAT_A);
        let format_b = exact_env(ENV_EXACT_FORMAT_B);
        let file_a = exact_env(ENV_EXACT_FILE_A);
        let file_b = exact_env(ENV_EXACT_FILE_B);
        let file_type_a = exact_env(ENV_EXACT_FILE_TYPE_A)
            .parse::<u64>()
            .expect("parse exact A GGUF file type");
        let file_type_b = exact_env(ENV_EXACT_FILE_TYPE_B)
            .parse::<u64>()
            .expect("parse exact B GGUF file type");
        let source_commit = exact_env(ENV_EXACT_SOURCE_COMMIT);
        let binary_sha256 = exact_env(ENV_EXACT_BINARY_SHA256);
        let binary_git_commit = exact_env(ENV_EXACT_BINARY_GIT_COMMIT);
        let mlx_version = exact_env(ENV_EXACT_MLX_VERSION);
        let mlx_source = exact_env(ENV_EXACT_MLX_SOURCE);
        let mlx_checksum = exact_env(ENV_EXACT_MLX_CHECKSUM);
        assert!(
            source_commit.len() == 40
                && source_commit
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "exact source commit must be lowercase SHA-1"
        );
        for (name, digest) in [
            ("binary", binary_sha256.as_str()),
            ("mlx-native", mlx_checksum.as_str()),
        ] {
            assert!(
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
                "exact {name} identity must be lowercase SHA-256"
            );
        }
        let executed_binary = hf2q_binary_path();
        assert_eq!(
            sha256_file(&executed_binary).expect("hash executed hf2q binary"),
            binary_sha256,
            "exact receipt binary differs from the executable used by the swap gate"
        );
        let build_info_output = Command::new(&executed_binary)
            .arg("__build-info")
            .output()
            .expect("read executed hf2q build provenance");
        assert!(
            build_info_output.status.success(),
            "executed hf2q does not expose build provenance: {}",
            String::from_utf8_lossy(&build_info_output.stderr)
        );
        let build_info: serde_json::Value = serde_json::from_slice(&build_info_output.stdout)
            .expect("parse executed hf2q build provenance");
        assert_eq!(build_info["schema"], "hf2q.build-info.v1");
        assert_eq!(
            build_info["git_commit"].as_str(),
            Some(binary_git_commit.as_str())
        );
        assert_eq!(binary_git_commit, source_commit);
        assert_eq!(
            swap_budget,
            Duration::from_secs(SWAP_BUDGET_SECS),
            "exact swap proof must use the fixed ten-second load budget"
        );
        let receipt = serde_json::json!({
            "schema": 1,
            "verdict": "pass",
            "pair": {"id": pair_id, "a": format_a, "b": format_b},
            "binding": {
                "source_commit": source_commit,
                "binary_sha256": binary_sha256,
                "binary_git_commit": binary_git_commit,
                "dependency": {
                    "name": "mlx-native",
                    "version": mlx_version,
                    "source": mlx_source,
                    "checksum": mlx_checksum,
                },
            },
            "artifacts": {
                "a": {
                    "format": format_a,
                    "file": file_a,
                    "bytes": bytes_a,
                    "sha256": expected_a_sha256,
                    "gguf_file_type": file_type_a,
                },
                "b": {
                    "format": format_b,
                    "file": file_b,
                    "bytes": bytes_b,
                    "sha256": expected_b_sha256,
                    "gguf_file_type": file_type_b,
                },
            },
            "pool_budget_bytes": pool_budget,
            "load_budget_seconds": SWAP_BUDGET_SECS,
            "proof": {
                "one_resident_every_phase": true,
                "fresh_generation_per_activation": true,
                "execution_receipts_joined": true,
                "bounded_residency": true,
                "no_double_residency": true,
                "evicted_artifact_absent": true,
                "exact_a_replay": true,
            },
            "phases": [
                phase_receipt("A1", &format_a, &resident_a1, &execution_a1, &result_a1),
                phase_receipt("B", &format_b, &resident_b, &execution_b, &result_b),
                phase_receipt("A2", &format_a, &resident_a2, &execution_a2, &result_a2),
            ],
            "transitions": {
                "a_to_b": {
                    "load_seconds": switch_to_b.as_secs_f64(),
                    "peak_rss_bytes": peak_a_to_b.rss_bytes,
                    "peak_host_wired_bytes": peak_a_to_b.system_wired_bytes,
                    "rss_bound_bytes": memory_a1.rss_bytes.max(memory_b.rss_bytes)
                        .saturating_add(peak_margin),
                    "host_wired_bound_bytes": single_resident_host_wired_bound(
                        memory_a1, memory_b, bytes_b, peak_margin),
                },
                "b_to_a": {
                    "load_seconds": switch_to_a.as_secs_f64(),
                    "peak_rss_bytes": peak_b_to_a.rss_bytes,
                    "peak_host_wired_bytes": peak_b_to_a.system_wired_bytes,
                    "rss_bound_bytes": memory_a1.rss_bytes.max(memory_b.rss_bytes)
                        .saturating_add(reload_margin),
                    "host_wired_bound_bytes": single_resident_host_wired_bound(
                        memory_b, memory_a2, bytes_a, reload_margin),
                },
            },
            "memory": {
                "a1": memory_receipt(memory_a1),
                "b": memory_receipt(memory_b),
                "a2": memory_receipt(memory_a2),
            },
            "replay_bounds": {
                "rss_bytes": memory_a1.rss_bytes.saturating_add(reload_margin),
                "physical_footprint_bytes": memory_a1.physical_footprint_bytes
                    .saturating_add(footprint_reload_margin),
                "wired_bytes": memory_a1.wired_bytes.saturating_add(wired_reload_margin),
                "host_wired_bytes": host_wired_replay_bound,
            },
            "storage": {
                "a1_file_backed": mappings_a1.0.is_live(),
                "b": b_storage_contract,
                "a2_file_backed": mappings_a2.0.is_live(),
            },
        });
        write_exact_receipt(std::path::Path::new(&receipt_path), &receipt);
    }

    eprintln!(
        "model_swap PASS: A -> B={switch_to_b:?}, B -> A={switch_to_a:?}, \
         inference A1/B/A2={inference_a1:?}/{inference_b:?}/{inference_a2:?}; \
         semantic-TTFT A1/B/A2={ttft_a1:?}/{ttft_b:?}/{ttft_a2:?}; \
         switch-to-first-semantic B/A2={:?}/{:?}; \
         RSS A1/B/A2={}/{}/{}; footprint A1/B/A2={}/{}/{}; \
         process-wired A1/B/A2={}/{}/{}; host-wired A1/B/A2={}/{}/{}",
        switch_to_b + ttft_b,
        switch_to_a + ttft_a2,
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
    eprintln!(
        "model_swap mappings: A1={mappings_a1:?}; B_contract={b_storage_contract} \
         B={mappings_b:?}; A2={mappings_a2:?}"
    );
}

/// Two complete five-format cycles in one long-lived server process. Pairwise
/// A -> B -> A cells cannot prove cumulative reclamation because restarting the
/// process erases sub-threshold leaks and stale generation state.
#[test]
fn qwen38_exact_five_format_two_cycle_e2e() {
    let Ok(receipt_path) = std::env::var(ENV_EXACT_CHAIN_RECEIPT) else {
        eprintln!(
            "[skip] qwen38_exact_five_format_two_cycle_e2e — \
             {ENV_EXACT_CHAIN_RECEIPT} is not set"
        );
        return;
    };
    if skip_unless_gated("qwen38_exact_five_format_two_cycle_e2e") {
        return;
    }

    let expected_formats = ["BF16", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"];
    let mut artifacts = exact_chain_artifacts();
    assert_eq!(
        artifacts
            .iter()
            .map(|artifact| artifact.format.as_str())
            .collect::<Vec<_>>(),
        expected_formats
    );
    let mut canonical_paths = std::collections::HashSet::new();
    for artifact in &mut artifacts {
        artifact.path = std::fs::canonicalize(&artifact.path).unwrap_or_else(|error| {
            panic!("cannot resolve exact {} artifact: {error}", artifact.format)
        });
        assert!(
            canonical_paths.insert(artifact.path.clone()),
            "exact chain artifacts must be distinct physical paths"
        );
        assert_eq!(
            std::fs::metadata(&artifact.path)
                .expect("exact chain artifact metadata")
                .len(),
            artifact.bytes,
            "{} exact chain byte identity",
            artifact.format
        );
    }

    let source_commit = exact_env(ENV_EXACT_SOURCE_COMMIT);
    let binary_sha256 = exact_env(ENV_EXACT_BINARY_SHA256);
    let binary_git_commit = exact_env(ENV_EXACT_BINARY_GIT_COMMIT);
    let mlx_version = exact_env(ENV_EXACT_MLX_VERSION);
    let mlx_source = exact_env(ENV_EXACT_MLX_SOURCE);
    let mlx_checksum = exact_env(ENV_EXACT_MLX_CHECKSUM);
    assert_eq!(binary_git_commit, source_commit);
    let executed_binary = hf2q_binary_path();
    assert_eq!(
        sha256_file(&executed_binary).expect("hash exact-chain hf2q binary"),
        binary_sha256
    );
    assert_eq!(
        executed_build_info(&executed_binary)["git_commit"].as_str(),
        Some(source_commit.as_str())
    );

    // BF16 is the eviction hub. A max-artifact pool permits any smaller
    // adjacent pair to co-reside, so a direct Q4 -> Q5 -> Q6 -> Q8 chain
    // cannot prove replacement in one process. The pairwise cells retain
    // those direct edges; this cumulative chain forces every transition.
    let sequence = [0usize, 1, 0, 2, 0, 3, 0, 4, 0, 1, 0, 2, 0, 3, 0, 4, 0];
    let pool_budget = artifacts
        .iter()
        .map(|artifact| artifact.bytes)
        .max()
        .expect("non-empty exact chain");
    let server = ServerGuard::spawn(&artifacts[0].path.to_string_lossy(), pool_budget)
        .expect("spawn exact-chain hf2q serve");
    wait_for_readyz();
    let runtime = tokio::runtime::Runtime::new().expect("exact-chain tokio runtime");
    let client = build_client();
    let mut request_model = runtime.block_on(fetch_canonical_model_id(&client));
    let pid = server.0.id();

    let mut phases = Vec::with_capacity(sequence.len());
    let mut transitions = Vec::with_capacity(sequence.len() - 1);
    let mut generation_ids = std::collections::HashSet::new();
    let mut format_identities = std::collections::BTreeMap::new();
    let mut unique_pool_keys = std::collections::HashSet::new();

    let first = observe_exact_chain_phase(
        &runtime,
        &client,
        &request_model,
        &artifacts[0],
        0,
        &artifacts,
        "P00-BF16",
        pid,
    );
    assert!(generation_ids.insert(first.resident.generation));
    unique_pool_keys.insert(first.resident.pool_key.clone());
    format_identities.insert(
        artifacts[0].format.clone(),
        (
            sha256_bytes(first.resident.pool_key.as_bytes()),
            sha256_json(&first.resident.engine_config),
        ),
    );
    let baseline_memory = first.memory;
    let baseline_result = first.result.message.clone();
    phases.push(exact_chain_phase_receipt(
        0,
        "P00-BF16",
        pid,
        &artifacts[0],
        &first,
    ));
    let mut previous_memory = first.memory;

    for (step, &artifact_index) in sequence.iter().enumerate().skip(1) {
        let artifact = &artifacts[artifact_index];
        let phase = format!("P{step:02}-{}", artifact.format);
        let previous_format = artifacts[sequence[step - 1]].format.as_str();
        let (switch_receipt, load_time, peak) =
            measured_switch(&runtime, &client, &artifact.path, pid);
        assert!(
            load_time < Duration::from_secs(SWAP_BUDGET_SECS),
            "{phase}: load took {load_time:?}, exceeds fixed ten-second budget"
        );
        request_model = switch_receipt["request_model"]
            .as_str()
            .expect("exact chain switch request_model")
            .to_owned();
        let observation = observe_exact_chain_phase(
            &runtime,
            &client,
            &request_model,
            artifact,
            artifact_index,
            &artifacts,
            &phase,
            pid,
        );
        assert!(
            generation_ids.insert(observation.resident.generation),
            "{phase}: resident generation was reused"
        );
        let pool_key_sha = sha256_bytes(observation.resident.pool_key.as_bytes());
        let engine_config_sha = sha256_json(&observation.resident.engine_config);
        if let Some((expected_pool, expected_config)) = format_identities.get(&artifact.format) {
            assert_eq!(&pool_key_sha, expected_pool, "{phase}: pool identity drift");
            assert_eq!(
                &engine_config_sha, expected_config,
                "{phase}: engine configuration drift"
            );
        } else {
            assert!(
                unique_pool_keys.insert(observation.resident.pool_key.clone()),
                "{phase}: distinct formats aliased one pool identity"
            );
            format_identities.insert(
                artifact.format.clone(),
                (pool_key_sha.clone(), engine_config_sha.clone()),
            );
        }
        let (rss_bound, host_wired_bound) = assert_chain_transition_memory(
            previous_memory,
            observation.memory,
            peak,
            artifact.bytes,
            &phase,
        );
        transitions.push(serde_json::json!({
            "index": step - 1,
            "from": previous_format,
            "to": artifact.format.as_str(),
            "load_seconds": load_time.as_secs_f64(),
            "peak_rss_bytes": peak.rss_bytes,
            "peak_host_wired_bytes": peak.system_wired_bytes,
            "rss_bound_bytes": rss_bound,
            "host_wired_bound_bytes": host_wired_bound,
        }));
        if artifact_index == 0 {
            assert_eq!(
                observation.result.message, baseline_result,
                "{phase}: BF16 replay diverged"
            );
        }
        previous_memory = observation.memory;
        phases.push(exact_chain_phase_receipt(
            step,
            &phase,
            pid,
            artifact,
            &observation,
        ));
    }

    assert_eq!(generation_ids.len(), sequence.len());
    assert_eq!(unique_pool_keys.len(), artifacts.len());
    let rss_replay_bound = baseline_memory
        .rss_bytes
        .saturating_add(memory_margin(baseline_memory.rss_bytes));
    let footprint_replay_bound = baseline_memory
        .physical_footprint_bytes
        .saturating_add(memory_margin(baseline_memory.physical_footprint_bytes));
    let wired_replay_bound = baseline_memory
        .wired_bytes
        .saturating_add(memory_margin(baseline_memory.wired_bytes));
    let host_wired_replay_bound = replay_host_wired_bound(
        phases
            .iter()
            .map(|phase| phase["memory"]["host_wired_bytes"].as_u64().unwrap()),
        artifacts[0].bytes,
        memory_margin(baseline_memory.rss_bytes),
    );
    for index in [8usize, 16] {
        let memory = &phases[index]["memory"];
        assert!(memory["rss_bytes"].as_u64().unwrap() <= rss_replay_bound);
        assert!(memory["physical_footprint_bytes"].as_u64().unwrap() <= footprint_replay_bound);
        assert!(memory["wired_bytes"].as_u64().unwrap() <= wired_replay_bound);
        assert!(
            memory["host_wired_bytes"].as_u64().unwrap() <= host_wired_replay_bound,
            "BF16 replay host-wired memory exceeded one-artifact bound: phase={index}, \
             observed={}, bound={host_wired_replay_bound}",
            memory["host_wired_bytes"].as_u64().unwrap()
        );
    }

    let artifact_receipts = artifacts
        .iter()
        .map(|artifact| {
            serde_json::json!({
                "format": artifact.format.as_str(),
                "file": artifact.file.as_str(),
                "bytes": artifact.bytes,
                "sha256": artifact.sha256.as_str(),
                "gguf_file_type": artifact.gguf_file_type,
            })
        })
        .collect::<Vec<_>>();
    let sequence_formats = sequence
        .iter()
        .map(|&index| artifacts[index].format.as_str())
        .collect::<Vec<_>>();
    let receipt = serde_json::json!({
        "schema": 1,
        "verdict": "pass",
        "gate": "qwen38-exact-five-format-two-cycle",
        "binding": {
            "source_commit": source_commit,
            "binary_sha256": binary_sha256,
            "binary_git_commit": binary_git_commit,
            "dependency": {
                "name": "mlx-native",
                "version": mlx_version,
                "source": mlx_source,
                "checksum": mlx_checksum,
            },
        },
        "artifacts": artifact_receipts,
        "pool_budget_bytes": pool_budget,
        "load_budget_seconds": SWAP_BUDGET_SECS,
        "process": {"pid": pid},
        "sequence": sequence_formats,
        "proof": {
            "one_long_lived_process": true,
            "two_complete_cycles": true,
            "fresh_generation_every_activation": true,
            "cold_generation_cache": true,
            "execution_receipts_joined": true,
            "bounded_every_transition": true,
            "evicted_artifacts_absent": true,
            "exact_bf16_replay": true,
        },
        "phases": phases,
        "transitions": transitions,
        "cycle_replay_phase_indexes": [8, 16],
        "replay_bounds": {
            "rss_bytes": rss_replay_bound,
            "physical_footprint_bytes": footprint_replay_bound,
            "wired_bytes": wired_replay_bound,
            "host_wired_bytes": host_wired_replay_bound,
        },
    });
    write_exact_receipt(std::path::Path::new(&receipt_path), &receipt);
}

/// Two complete architecture-family cycles in one long-lived server process.
/// This is the universal generative lifecycle authority: fresh-server pair
/// rows cannot expose stale family state or cumulative residency leaks.
#[test]
fn generative_cross_family_two_cycle_e2e() {
    use std::os::unix::fs::MetadataExt as _;

    let Ok(receipt_path) = std::env::var(ENV_GENERATIVE_CHAIN_RECEIPT) else {
        eprintln!(
            "[skip] generative_cross_family_two_cycle_e2e — \
             {ENV_GENERATIVE_CHAIN_RECEIPT} is not set"
        );
        return;
    };
    if skip_unless_gated("generative_cross_family_two_cycle_e2e") {
        return;
    }

    let (mut artifacts, sequence_ids, load_budget_seconds) = generative_chain_spec();
    let expected_ids = ["qwen-dense", "qwen-moe", "gemma", "deepseek"];
    let expected_architectures = ["qwen35", "qwen35moe", "gemma4", "deepseek4"];
    let expected_families = ["qwen35", "qwen35", "gemma4", "deepseek4"];
    assert_eq!(
        artifacts
            .iter()
            .map(|artifact| artifact.id.as_str())
            .collect::<Vec<_>>(),
        expected_ids
    );
    assert_eq!(
        artifacts
            .iter()
            .map(|artifact| artifact.architecture.as_str())
            .collect::<Vec<_>>(),
        expected_architectures
    );
    assert_eq!(
        artifacts
            .iter()
            .map(|artifact| artifact.arch_family.as_str())
            .collect::<Vec<_>>(),
        expected_families
    );
    assert_eq!(load_budget_seconds, GENERATIVE_SWAP_BUDGET_SECS);
    assert_eq!(
        sequence_ids,
        [
            "qwen-dense",
            "deepseek",
            "qwen-moe",
            "deepseek",
            "gemma",
            "deepseek",
            "qwen-dense",
            "deepseek",
            "qwen-moe",
            "deepseek",
            "gemma",
            "deepseek",
            "qwen-dense",
        ]
    );

    let mut physical_files = std::collections::HashSet::new();
    let mut canaries = std::collections::HashSet::new();
    for artifact in &mut artifacts {
        artifact.path = std::fs::canonicalize(&artifact.path)
            .unwrap_or_else(|error| panic!("cannot resolve {} artifact: {error}", artifact.id));
        let metadata = std::fs::metadata(&artifact.path)
            .unwrap_or_else(|error| panic!("cannot stat {}: {error}", artifact.id));
        assert!(
            physical_files.insert((metadata.dev(), metadata.ino())),
            "generative chain artifacts must be distinct physical files"
        );
        assert_eq!(
            metadata.len(),
            artifact.bytes,
            "{} byte identity",
            artifact.id
        );
        assert_eq!(
            artifact.path.file_name().and_then(std::ffi::OsStr::to_str),
            Some(artifact.file.as_str()),
            "{} basename identity",
            artifact.id
        );
        assert!(
            canaries.insert(artifact.canary.clone()),
            "family semantic canaries must be unique"
        );
    }

    let source_commit = exact_env(ENV_EXACT_SOURCE_COMMIT);
    let binary_sha256 = exact_env(ENV_EXACT_BINARY_SHA256);
    let binary_git_commit = exact_env(ENV_EXACT_BINARY_GIT_COMMIT);
    let mlx_version = exact_env(ENV_EXACT_MLX_VERSION);
    let mlx_source = exact_env(ENV_EXACT_MLX_SOURCE);
    let mlx_checksum = exact_env(ENV_EXACT_MLX_CHECKSUM);
    assert_eq!(binary_git_commit, source_commit);
    let executed_binary = hf2q_binary_path();
    assert_eq!(
        sha256_file(&executed_binary).expect("hash cross-family hf2q binary"),
        binary_sha256
    );
    assert_eq!(
        executed_build_info(&executed_binary)["git_commit"].as_str(),
        Some(source_commit.as_str())
    );

    let sequence = sequence_ids
        .iter()
        .map(|id| {
            artifacts
                .iter()
                .position(|artifact| &artifact.id == id)
                .unwrap_or_else(|| panic!("sequence names unknown artifact: {id}"))
        })
        .collect::<Vec<_>>();
    let pool_budget = artifacts
        .iter()
        .map(|artifact| artifact.bytes)
        .max()
        .expect("non-empty generative chain");
    let server = ServerGuard::spawn(&artifacts[0].path.to_string_lossy(), pool_budget)
        .expect("spawn cross-family hf2q serve");
    wait_for_readyz();
    let runtime = tokio::runtime::Runtime::new().expect("cross-family tokio runtime");
    let client = build_client();
    let mut request_model = runtime.block_on(fetch_canonical_model_id(&client));
    let pid = server.0.id();

    let mut phases = Vec::with_capacity(sequence.len());
    let mut transitions = Vec::with_capacity(sequence.len() - 1);
    let mut generation_ids = std::collections::HashSet::new();
    let mut artifact_identities = std::collections::BTreeMap::new();
    let mut replay_results = std::collections::BTreeMap::new();
    let mut unique_pool_keys = std::collections::HashSet::new();

    let first = observe_generative_chain_phase(
        &runtime,
        &client,
        &request_model,
        &artifacts[0],
        0,
        &artifacts,
        "P00-qwen-dense",
        pid,
    );
    assert!(generation_ids.insert(first.resident.generation));
    unique_pool_keys.insert(first.resident.pool_key.clone());
    artifact_identities.insert(
        artifacts[0].id.clone(),
        (
            sha256_bytes(first.resident.pool_key.as_bytes()),
            sha256_json(&first.resident.engine_config),
        ),
    );
    replay_results.insert(artifacts[0].id.clone(), first.result.message.clone());
    let process_policy = first.process_policy.clone();
    let process_policy_sha256 = first.process_policy_sha256.clone();
    let baseline_memory = first.memory;
    phases.push(generative_chain_phase_receipt(
        0,
        "P00-qwen-dense",
        pid,
        &artifacts[0],
        &first,
    ));
    let mut previous_memory = first.memory;

    for (step, &artifact_index) in sequence.iter().enumerate().skip(1) {
        let artifact = &artifacts[artifact_index];
        let phase = format!("P{step:02}-{}", artifact.id);
        let previous_id = artifacts[sequence[step - 1]].id.as_str();
        let (switch_receipt, load_time, peak) =
            measured_switch(&runtime, &client, &artifact.path, pid);
        assert!(
            load_time < Duration::from_secs(GENERATIVE_SWAP_BUDGET_SECS),
            "{phase}: load took {load_time:?}, exceeds fixed cross-family budget"
        );
        request_model = switch_receipt["request_model"]
            .as_str()
            .expect("cross-family switch request_model")
            .to_owned();
        let observation = observe_generative_chain_phase(
            &runtime,
            &client,
            &request_model,
            artifact,
            artifact_index,
            &artifacts,
            &phase,
            pid,
        );
        assert!(
            generation_ids.insert(observation.resident.generation),
            "{phase}: resident generation was reused"
        );
        let pool_key_sha = sha256_bytes(observation.resident.pool_key.as_bytes());
        let engine_config_sha = sha256_json(&observation.resident.engine_config);
        assert_eq!(
            observation.process_policy, process_policy,
            "{phase}: process serving policy changed across family replacement"
        );
        assert_eq!(
            observation.process_policy_sha256, process_policy_sha256,
            "{phase}: process serving policy hash changed across family replacement"
        );
        if let Some((expected_pool, expected_config)) = artifact_identities.get(&artifact.id) {
            assert_eq!(&pool_key_sha, expected_pool, "{phase}: pool identity drift");
            assert_eq!(
                &engine_config_sha, expected_config,
                "{phase}: engine configuration drift"
            );
            assert_eq!(
                replay_results.get(&artifact.id),
                Some(&observation.result.message),
                "{phase}: exact family replay diverged"
            );
        } else {
            assert!(
                unique_pool_keys.insert(observation.resident.pool_key.clone()),
                "{phase}: distinct families aliased one pool identity"
            );
            artifact_identities.insert(
                artifact.id.clone(),
                (pool_key_sha.clone(), engine_config_sha.clone()),
            );
            replay_results.insert(artifact.id.clone(), observation.result.message.clone());
        }
        let (rss_bound, host_wired_bound) = assert_chain_transition_memory(
            previous_memory,
            observation.memory,
            peak,
            artifact.bytes,
            &phase,
        );
        transitions.push(serde_json::json!({
            "index": step - 1,
            "from": previous_id,
            "to": artifact.id.as_str(),
            "load_seconds": load_time.as_secs_f64(),
            "peak_rss_bytes": peak.rss_bytes,
            "peak_host_wired_bytes": peak.system_wired_bytes,
            "rss_bound_bytes": rss_bound,
            "host_wired_bound_bytes": host_wired_bound,
        }));
        previous_memory = observation.memory;
        phases.push(generative_chain_phase_receipt(
            step,
            &phase,
            pid,
            artifact,
            &observation,
        ));
    }

    assert_eq!(generation_ids.len(), sequence.len());
    assert_eq!(unique_pool_keys.len(), artifacts.len());
    assert_eq!(artifact_identities.len(), artifacts.len());
    assert_eq!(replay_results.len(), artifacts.len());
    let rss_replay_bound = baseline_memory
        .rss_bytes
        .saturating_add(memory_margin(baseline_memory.rss_bytes));
    let footprint_replay_bound = baseline_memory
        .physical_footprint_bytes
        .saturating_add(memory_margin(baseline_memory.physical_footprint_bytes));
    let wired_replay_bound = baseline_memory
        .wired_bytes
        .saturating_add(memory_margin(baseline_memory.wired_bytes));
    let host_wired_replay_bound = replay_host_wired_bound(
        phases
            .iter()
            .map(|phase| phase["memory"]["host_wired_bytes"].as_u64().unwrap()),
        artifacts[sequence[0]].bytes,
        memory_margin(baseline_memory.rss_bytes),
    );
    for index in [6usize, 12] {
        let memory = &phases[index]["memory"];
        assert!(memory["rss_bytes"].as_u64().unwrap() <= rss_replay_bound);
        assert!(memory["physical_footprint_bytes"].as_u64().unwrap() <= footprint_replay_bound);
        assert!(memory["wired_bytes"].as_u64().unwrap() <= wired_replay_bound);
        assert!(
            memory["host_wired_bytes"].as_u64().unwrap() <= host_wired_replay_bound,
            "family replay host-wired memory exceeded one-artifact bound: phase={index}, \
             observed={}, bound={host_wired_replay_bound}",
            memory["host_wired_bytes"].as_u64().unwrap()
        );
    }

    let artifact_receipts = artifacts
        .iter()
        .map(|artifact| {
            serde_json::json!({
                "id": artifact.id.as_str(),
                "architecture": artifact.architecture.as_str(),
                "arch_family": artifact.arch_family.as_str(),
                "file": artifact.file.as_str(),
                "bytes": artifact.bytes,
                "sha256": artifact.sha256.as_str(),
                "canary": artifact.canary.as_str(),
            })
        })
        .collect::<Vec<_>>();
    let receipt = serde_json::json!({
        "schema": 1,
        "verdict": "pass",
        "gate": "generative-cross-family-two-cycle",
        "binding": {
            "source_commit": source_commit,
            "binary_sha256": binary_sha256,
            "binary_git_commit": binary_git_commit,
            "dependency": {
                "name": "mlx-native",
                "version": mlx_version,
                "source": mlx_source,
                "checksum": mlx_checksum,
            },
        },
        "artifacts": artifact_receipts,
        "pool_budget_bytes": pool_budget,
        "load_budget_seconds": GENERATIVE_SWAP_BUDGET_SECS,
        "process": {"pid": pid},
        "sequence": sequence_ids,
        "proof": {
            "one_long_lived_process": true,
            "two_complete_cycles": true,
            "every_required_family": true,
            "unique_semantic_canary_per_family": true,
            "fresh_generation_every_activation": true,
            "cold_generation_cache": true,
            "execution_receipts_joined": true,
            "process_policy_preserved": true,
            "q5_policy_preserved_and_observed": true,
            "bounded_every_transition": true,
            "evicted_artifacts_absent": true,
            "exact_family_replay": true,
        },
        "phases": phases,
        "transitions": transitions,
        "cycle_replay_phase_indexes": [6, 12],
        "replay_bounds": {
            "rss_bytes": rss_replay_bound,
            "physical_footprint_bytes": footprint_replay_bound,
            "wired_bytes": wired_replay_bound,
            "host_wired_bytes": host_wired_replay_bound,
        },
    });
    write_exact_receipt(std::path::Path::new(&receipt_path), &receipt);
}
