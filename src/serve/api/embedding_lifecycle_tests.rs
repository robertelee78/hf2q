//! Dedicated encoder-model server lifecycle gates.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

use super::router::build_router;
use super::state::{AppState, ServerConfig};

const LONG_EMBED_PROMPT: &str = "A local inference server must preserve exact artifact identity, tokenizer configuration, native matrix storage, output coherence, and memory ownership while switching repeatedly between encoder families under realistic multi-token embedding requests. The returned vector should remain stable after unloading and reloading the original artifact, while another architecture uses its own vocabulary, dimensions, pooling behavior, kernel registry, and mapped weight generation without leaking state across requests.";
// The first quiet-host real run measured fresh-A deltas of 1.49 MiB settled
// RSS and zero wired-byte delta versus initial A. A 16 MiB bound leaves
// allocator headroom without being wide enough to hide either complete
// 25.4 MiB A or 83.1 MiB B artifact. Peak RSS uses the 1 ms in-process Mach
// sampler below; the former 20 ms `ps` subprocess sampler was falsified after
// it missed initial-A's short allocation peak.
const EMBEDDING_REPLAY_MEMORY_TOLERANCE_BYTES: u64 = 16 * 1024 * 1024;

fn state_default() -> AppState {
    AppState::new(ServerConfig::default())
}

async fn body_string(response: axum::response::Response) -> String {
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    String::from_utf8_lossy(&bytes).into_owned()
}

async fn post_embedding(app: &axum::Router, model: &str) -> (Vec<f64>, Duration) {
    post_embedding_text(app, model, "hello").await
}

async fn post_embedding_text(app: &axum::Router, model: &str, input: &str) -> (Vec<f64>, Duration) {
    let started = Instant::now();
    let body = serde_json::json!({"model": model, "input": input}).to_string();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .expect("embedding response");
    assert_eq!(response.status(), StatusCode::OK);
    let value: serde_json::Value =
        serde_json::from_str(&body_string(response).await).expect("embedding JSON");
    assert_eq!(
        value["model"], model,
        "embedding response must be bound to the requested active generation"
    );
    assert!(
        value["usage"]["prompt_tokens"]
            .as_u64()
            .is_some_and(|tokens| tokens > 0),
        "embedding response must report nonzero prompt-token usage"
    );
    let vector = value["data"][0]["embedding"]
        .as_array()
        .expect("embedding vector")
        .iter()
        .map(|value| value.as_f64().expect("finite embedding component"))
        .collect();
    (vector, started.elapsed())
}

async fn assert_embedding_model_unavailable(app: &axum::Router, model: &str, phase: &str) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({"model": model, "input": "hello"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("unavailable embedding response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{phase}");
    let body = body_string(response).await;
    assert!(body.contains("model_not_loaded"), "{phase}: {body}");
}

async fn activate_embedding_candidate(
    app: &axum::Router,
    model: &str,
    candidate_id: Option<&str>,
    action: &str,
    expected_generation: Option<u64>,
) -> (StatusCode, serde_json::Value, Duration) {
    let started = Instant::now();
    let mut body = serde_json::json!({
        "model": model,
        "kind": "embedding",
        "action": action,
    });
    if let Some(candidate_id) = candidate_id {
        body["candidate_id"] = candidate_id.into();
    }
    if let Some(generation) = expected_generation {
        body["expected_revision"] = generation.into();
    }
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/hf2q/v1/models/activate")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("embedding activation response");
    let elapsed = started.elapsed();
    let status = response.status();
    let value =
        serde_json::from_str(&body_string(response).await).expect("embedding activation JSON");
    (status, value, elapsed)
}

async fn embedding_runtime(app: &axum::Router) -> serde_json::Value {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/hf2q/v1/runtime")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("embedding runtime response");
    assert_eq!(response.status(), StatusCode::OK);
    serde_json::from_str(&body_string(response).await).expect("embedding runtime JSON")
}

fn register_real_embedding_candidate(
    state: &AppState,
    path: &Path,
    repository: &str,
    revision: &str,
    expected_sha256: &str,
) -> String {
    use crate::serve::api::local_artifacts::{
        LocalArtifactCatalog, LocalArtifactProvenance, LocalGgufArtifact,
    };

    let canonical = path.canonicalize().expect("canonical embedding artifact");
    let sha256 = crate::core::sha256::compute_file_sha256(&canonical)
        .expect("hash exact embedding artifact");
    assert_eq!(sha256, expected_sha256, "embedding artifact SHA drift");
    let header = mlx_native::gguf::GgufFile::open(&canonical).expect("embedding GGUF header");
    let file_type = header
        .metadata_u32("general.file_type")
        .expect("embedding general.file_type");
    let quant_hint = crate::quantize::ggml_quants::GgufFtype::try_from(file_type)
        .expect("known embedding file type")
        .name()
        .to_ascii_uppercase();
    assert!(matches!(
        header.metadata_string("general.architecture"),
        Some("bert" | "nomic-bert")
    ));
    let root = canonical
        .parent()
        .expect("embedding artifact parent")
        .to_path_buf();
    let filename = canonical
        .file_name()
        .expect("embedding filename")
        .to_string_lossy()
        .into_owned();
    let bytes = std::fs::metadata(&canonical)
        .expect("embedding metadata")
        .len();
    let view = state
        .artifact_catalog
        .register_local(LocalArtifactCatalog {
            artifacts: vec![LocalGgufArtifact {
                repository: repository.into(),
                revision: revision.into(),
                filename,
                root,
                path: canonical,
                bytes,
                sha256,
                quant_hint,
                file_type,
                quant: crate::serve::quant_select::QuantType::from_gguf_file_type(file_type),
                role: "embedding_model".into(),
                selectable: true,
                unavailable_reason: None,
                provenance: LocalArtifactProvenance::ConversionReceipt,
            }],
            warnings: Vec::new(),
        })
        .expect("register exact embedding candidate");
    view.candidates[0]
        .candidate_id
        .clone()
        .expect("embedding candidate id")
}

fn write_header_valid_but_unloadable_embedding(path: &Path) {
    let file_type_key = b"general.file_type";
    let arch_key = b"general.architecture";
    let arch_value = b"bert";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    bytes.extend_from_slice(&(file_type_key.len() as u64).to_le_bytes());
    bytes.extend_from_slice(file_type_key);
    bytes.extend_from_slice(&4_u32.to_le_bytes()); // GGUF_TYPE_UINT32
    bytes.extend_from_slice(&2_u32.to_le_bytes()); // Q4_0 file type
    bytes.extend_from_slice(&(arch_key.len() as u64).to_le_bytes());
    bytes.extend_from_slice(arch_key);
    bytes.extend_from_slice(&8_u32.to_le_bytes()); // GGUF_TYPE_STRING
    bytes.extend_from_slice(&(arch_value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(arch_value);
    bytes.resize(256, 0);
    std::fs::write(path, bytes).expect("write invalid embedding GGUF");
}

#[derive(Debug, Clone, Copy)]
struct EmbeddingProcessMemory {
    rss_bytes: u64,
    wired_bytes: u64,
}

#[derive(Debug)]
struct EmbeddingArtifactMapping {
    path: PathBuf,
    inode: u64,
    lsof_live: bool,
    vmmap_live: bool,
}

impl EmbeddingArtifactMapping {
    fn is_live(&self) -> bool {
        self.lsof_live || self.vmmap_live
    }
}

/// A mapped GGUF may close its descriptor while its file-backed pages remain.
/// Require both the exact-file `lsof` view and the virtual-memory view to agree
/// before declaring an evicted embedding artifact absent.
fn embedding_artifact_mapping(path: &Path) -> EmbeddingArtifactMapping {
    use std::os::unix::fs::MetadataExt;

    let path = path
        .canonicalize()
        .expect("canonical embedding mapping path");
    let inode = std::fs::metadata(&path)
        .expect("embedding mapping metadata")
        .ino();
    let pid = std::process::id().to_string();
    let lsof = std::process::Command::new("lsof")
        .args(["-nP", "-a", "-p", &pid, "--"])
        .arg(&path)
        .output()
        .expect("run lsof for embedding mapping");
    let lsof_live = lsof.status.success();
    assert!(
        lsof_live || lsof.status.code() == Some(1),
        "lsof failed for embedding artifact {}: status={:?}, stderr={}",
        path.display(),
        lsof.status,
        String::from_utf8_lossy(&lsof.stderr)
    );
    let vmmap = std::process::Command::new("vmmap")
        .args(["-wide", &pid])
        .output()
        .expect("run vmmap for embedding mapping");
    assert!(
        vmmap.status.success(),
        "vmmap failed for embedding artifact {}: {}",
        path.display(),
        String::from_utf8_lossy(&vmmap.stderr)
    );
    let vmmap_stdout = String::from_utf8_lossy(&vmmap.stdout);
    let path_text = path.to_string_lossy();
    let vmmap_live = vmmap_stdout.contains(path_text.as_ref());
    EmbeddingArtifactMapping {
        path,
        inode,
        lsof_live,
        vmmap_live,
    }
}

fn assert_embedding_mapping_state(
    present: &Path,
    absent: &Path,
    phase: &str,
) -> (EmbeddingArtifactMapping, EmbeddingArtifactMapping) {
    let present = embedding_artifact_mapping(present);
    let absent = embedding_artifact_mapping(absent);
    assert_ne!(present.path, absent.path, "{phase}: artifact path alias");
    assert_ne!(present.inode, absent.inode, "{phase}: artifact inode alias");
    assert!(
        present.is_live(),
        "{phase}: current embedding artifact has no mapped ownership: {present:?}"
    );
    assert!(
        !absent.is_live(),
        "{phase}: evicted embedding artifact remains mapped: {absent:?}"
    );
    (present, absent)
}

fn current_process_rss_bytes() -> u64 {
    let mut info = std::mem::MaybeUninit::<libc::mach_task_basic_info>::zeroed();
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    #[allow(deprecated)]
    let task = unsafe { libc::mach_task_self() };
    let status = unsafe {
        libc::task_info(
            task,
            libc::MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast::<libc::integer_t>(),
            &mut count,
        )
    };
    assert_eq!(status, libc::KERN_SUCCESS, "read test-process Mach RSS");
    assert_eq!(
        count,
        libc::MACH_TASK_BASIC_INFO_COUNT,
        "Mach RSS info width"
    );
    unsafe { info.assume_init().resident_size }
}

fn current_process_memory() -> EmbeddingProcessMemory {
    let pid = std::process::id();
    let directory = tempfile::tempdir().expect("footprint directory");
    let json = directory.path().join("footprint.json");
    let output = std::process::Command::new("footprint")
        .args([
            "--pid",
            &pid.to_string(),
            "--json",
            json.to_str().expect("footprint path"),
        ])
        .output()
        .expect("run footprint");
    assert!(
        output.status.success(),
        "footprint failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(json).expect("footprint JSON"))
            .expect("parse footprint JSON");
    let process = value["processes"]
        .as_array()
        .and_then(|rows| {
            rows.iter()
                .find(|row| row["pid"].as_u64() == Some(pid as u64))
        })
        .expect("footprint process row");
    let wired_bytes = process["categories"]
        .as_object()
        .expect("footprint categories")
        .values()
        .filter_map(|category| category["wired"].as_u64())
        .fold(0u64, u64::saturating_add);
    EmbeddingProcessMemory {
        rss_bytes: current_process_rss_bytes(),
        wired_bytes,
    }
}

fn start_embedding_rss_peak_sampler() -> (
    Arc<std::sync::atomic::AtomicBool>,
    std::thread::JoinHandle<u64>,
) {
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(0);
    let handle = std::thread::spawn(move || {
        let mut peak = current_process_rss_bytes();
        started_tx
            .send(())
            .expect("publish first embedding RSS sample");
        while !thread_stop.load(std::sync::atomic::Ordering::Acquire) {
            peak = peak.max(current_process_rss_bytes());
            std::thread::sleep(Duration::from_millis(1));
        }
        peak.max(current_process_rss_bytes())
    });
    started_rx
        .recv()
        .expect("embedding RSS sampler must take a baseline sample");
    (stop, handle)
}

#[test]
fn embedding_mach_rss_peak_sampler_cannot_return_a_vacuous_zero() {
    let (stop, handle) = start_embedding_rss_peak_sampler();
    stop.store(true, std::sync::atomic::Ordering::Release);
    assert!(
        handle.join().expect("embedding RSS peak sampler") > 1024 * 1024,
        "the in-process Mach RSS observer must return a plausible live-process sample"
    );
}

async fn steady_embedding_median(app: &axum::Router, model: &str) -> Duration {
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        samples.push(post_embedding(app, model).await.1);
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

#[tokio::test(flavor = "current_thread")]
async fn dedicated_embedding_server_a_b_a_replays_and_reclaims() {
    use crate::inference::models::bert::native_storage::test_support::{
        bert_tensors, write_fixture,
    };
    use crate::serve::api::state::load_synthetic_native_embedding_model;

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let directory = tempfile::tempdir().expect("fixture directory");
    let path_a = directory.path().join("a.gguf");
    let path_b = directory.path().join("b.gguf");
    write_fixture(&path_a, &bert_tensors(6, 0xA9, 2), true);
    write_fixture(&path_b, &bert_tensors(6, 0x67, 2), true);

    let state = state_default().with_embedding_model(
        load_synthetic_native_embedding_model(&path_a, "a", 4).expect("load initial A"),
    );
    let app = build_router(state.clone());
    let first_a_lease = state.acquire_embedding_model().expect("A generation");
    assert_eq!(first_a_lease.model.encode("hello", false), vec![4]);
    let first_a_registry = Arc::downgrade(&first_a_lease.model.registry);
    let weak_a = Arc::downgrade(&first_a_lease.model);
    drop(first_a_lease);
    let (first_a, first_a_latency) = post_embedding(&app, "a").await;

    let switch_b_started = Instant::now();
    let receipt_b = state
        .try_swap_embedding_model(|| {
            assert!(
                weak_a.upgrade().is_none(),
                "A must be reclaimed before B loads"
            );
            load_synthetic_native_embedding_model(&path_b, "b", 5)
        })
        .expect("switch A to B");
    let (b, b_first_latency) = post_embedding(&app, "b").await;
    let switch_to_first_b = switch_b_started.elapsed();
    assert_ne!(
        first_a, b,
        "distinct native artifacts must not share output"
    );
    assert_eq!(receipt_b.reclaimed_bytes, receipt_b.resident_bytes);
    assert!(receipt_b.resident_bytes > 0);
    let b_lease = state.acquire_embedding_model().expect("B generation");
    assert_eq!(b_lease.model.encode("hello", false), vec![5]);
    assert!(
        first_a_registry.upgrade().is_none(),
        "A registry must not survive the B publication"
    );
    let weak_b = Arc::downgrade(&b_lease.model);
    drop(b_lease);

    let switch_a_started = Instant::now();
    let receipt_a = state
        .try_swap_embedding_model(|| {
            assert!(
                weak_b.upgrade().is_none(),
                "B must be reclaimed before A reloads"
            );
            load_synthetic_native_embedding_model(&path_a, "a", 4)
        })
        .expect("switch B to fresh A");
    let (second_a, a_first_latency) = post_embedding(&app, "a").await;
    let switch_to_first_a = switch_a_started.elapsed();
    assert_eq!(first_a, second_a, "fresh A reload must replay exactly");
    assert_eq!(receipt_a.generation, 3);
    assert_eq!(receipt_a.reclaimed_bytes, receipt_a.resident_bytes);
    let second_a_lease = state
        .acquire_embedding_model()
        .expect("reloaded A generation");
    assert_eq!(second_a_lease.model.model_id, "a");
    assert_eq!(second_a_lease.model.encode("hello", false), vec![4]);
    assert!(first_a_registry.upgrade().is_none());
    assert_eq!(
        state
            .embedding_slot
            .read()
            .expect("embedding slot")
            .logical_resident_bytes(),
        receipt_a.resident_bytes,
        "slot accounting must contain exactly the current generation"
    );

    eprintln!(
        "dedicated embedding lifecycle: initial_first={}us switch_B_load={}us B_first={}us switch_to_first_B={}us switch_A_load={}us A_first={}us switch_to_first_A={}us",
        first_a_latency.as_micros(),
        receipt_b.load_elapsed.as_micros(),
        b_first_latency.as_micros(),
        switch_to_first_b.as_micros(),
        receipt_a.load_elapsed.as_micros(),
        a_first_latency.as_micros(),
        switch_to_first_a.as_micros(),
    );
    assert!(
        switch_to_first_b < Duration::from_secs(5) && switch_to_first_a < Duration::from_secs(5),
        "synthetic switch-to-first embedding must stay below five seconds"
    );
}

/// Real source-derived lifecycle gate for the dedicated embedding slot. The
/// inputs must be hf2q-converted GGUFs, not downloaded quantized artifacts.
#[tokio::test(flavor = "current_thread")]
async fn dedicated_embedding_real_bert_nomic_bert_replays_and_reclaims() {
    if std::env::var("HF2Q_EMBED_SWAP_E2E").as_deref() != Ok("1") {
        eprintln!("skipping: set HF2Q_EMBED_SWAP_E2E=1 for the real embedding swap gate");
        return;
    }
    let path_a = PathBuf::from(
        std::env::var_os("HF2Q_EMBED_SWAP_E2E_MODEL_A")
            .expect("HF2Q_EMBED_SWAP_E2E_MODEL_A must name an hf2q-converted BERT GGUF"),
    );
    let path_b = PathBuf::from(
        std::env::var_os("HF2Q_EMBED_SWAP_E2E_MODEL_B")
            .expect("HF2Q_EMBED_SWAP_E2E_MODEL_B must name an hf2q-converted NomicBert GGUF"),
    );
    let helper = PathBuf::from(
        std::env::var_os("HF2Q_TEST_CONTROL_HELPER_BIN")
            .expect("HF2Q_TEST_CONTROL_HELPER_BIN must name the exact release hf2q binary"),
    );
    assert!(
        helper.is_file(),
        "control helper must be a built hf2q binary"
    );
    let _gpu = crate::inference::hf2q_gpu_test_lock();

    let state = state_default();
    let candidate_a = register_real_embedding_candidate(
        &state,
        &path_a,
        "BAAI/bge-small-en-v1.5",
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "1e55ff235dc9e7ea1d0fb1f5e588b3c774b316ba272365d668403b9e457549d6",
    );
    let candidate_b = register_real_embedding_candidate(
        &state,
        &path_b,
        "nomic-ai/nomic-embed-text-v1.5",
        "e9b6763023c676ca8431644204f50c2b100d9aab",
        "99d5c1378a62669cd0b199ae0506b91f81600d438f0bcb0cab37c4e733078e6a",
    );
    let app = build_router(state.clone());

    let baseline_memory = current_process_memory();
    let (stop_a, peak_a_thread) = start_embedding_rss_peak_sampler();
    let initial_started = Instant::now();
    let (initial_status, initial_receipt, initial_activation_latency) =
        activate_embedding_candidate(
            &app,
            "BAAI/bge-small-en-v1.5",
            Some(&candidate_a),
            "load",
            None,
        )
        .await;
    assert_eq!(initial_status, StatusCode::OK, "{initial_receipt}");
    assert_eq!(initial_receipt["kind"], "embedding");
    assert_eq!(initial_receipt["status"], "loaded");
    assert_eq!(
        initial_receipt["candidate"]["repo"],
        "BAAI/bge-small-en-v1.5"
    );
    assert!(initial_receipt["candidate"]["exact_selection"]
        .as_str()
        .is_some_and(
            |selection| selection.contains("5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
                && selection.contains(&candidate_a)
        ));
    assert!(!initial_receipt
        .to_string()
        .contains(path_a.to_string_lossy().as_ref()));
    let model_a_id = initial_receipt["embedding"]["model_id"]
        .as_str()
        .expect("initial A model id")
        .to_owned();
    assert_eq!(initial_receipt["request_model"], model_a_id);

    let first_a_lease = state.acquire_embedding_model().expect("A generation");
    let initial_load = first_a_lease.model.load_timing;
    assert_eq!(
        initial_receipt["timing_us"]["load_ready"].as_u64(),
        Some(initial_load.weight_load_elapsed.as_micros() as u64)
    );
    assert_eq!(
        initial_receipt["timing_us"]["post_warm"].as_u64(),
        Some(initial_load.total_elapsed.as_micros() as u64)
    );
    let first_a_generation = first_a_lease.generation;
    let first_a_registry = Arc::downgrade(&first_a_lease.model.registry);
    let first_a_tokenizer = Arc::downgrade(&first_a_lease.model.tokenizer);
    let first_a_vocab = Arc::downgrade(&first_a_lease.model.vocab);
    let first_a_tokens = first_a_lease.model.encode("hello", true);
    let first_a_arch = first_a_lease
        .model
        .arch
        .as_ref()
        .expect("production A arch");
    assert_eq!(first_a_arch.arch_name(), "bert");
    assert_eq!(first_a_arch.hidden_size(), 384);
    let a_resident_bytes = first_a_lease.model.resident_bytes();
    let weak_a = Arc::downgrade(&first_a_lease.model);
    drop(first_a_lease);

    let (first_a, first_a_latency) = post_embedding(&app, &model_a_id).await;
    let (first_a_long, _) = post_embedding_text(&app, &model_a_id, LONG_EMBED_PROMPT).await;
    let initial_to_first = initial_started.elapsed();
    stop_a.store(true, std::sync::atomic::Ordering::Release);
    let peak_a_rss = peak_a_thread.join().expect("initial A RSS peak sampler");
    let a_steady = steady_embedding_median(&app, &model_a_id).await;
    let memory_a = current_process_memory();
    let mappings_a = assert_embedding_mapping_state(&path_a, &path_b, "A1");
    assert_unit_embedding("initial BERT A", &first_a, 384);
    assert_eq!(first_a_generation, 1);
    let (probe_a_status, probe_a, _) = activate_embedding_candidate(
        &app,
        "BAAI/bge-small-en-v1.5",
        Some(&candidate_a),
        "probe",
        None,
    )
    .await;
    assert_eq!(probe_a_status, StatusCode::OK);
    assert_eq!(probe_a["status"], "resident");
    assert_eq!(probe_a["embedding"]["generation"], first_a_generation);

    let (stop_b, peak_b_thread) = start_embedding_rss_peak_sampler();
    let switch_b_started = Instant::now();
    let (conflict_b_status, conflict_b, conflict_b_latency) = activate_embedding_candidate(
        &app,
        "nomic-ai/nomic-embed-text-v1.5",
        Some(&candidate_b),
        "load",
        None,
    )
    .await;
    assert_eq!(conflict_b_status, StatusCode::CONFLICT);
    assert_eq!(conflict_b["code"], "embedding_model_resident");
    assert_eq!(conflict_b["embedding"]["generation"], first_a_generation);
    assert_eq!(conflict_b["requires_explicit_switch"], true);
    assert!(conflict_b["candidate"]["exact_selection"]
        .as_str()
        .is_some_and(
            |selection| selection.contains("e9b6763023c676ca8431644204f50c2b100d9aab")
                && selection.contains(&candidate_b)
        ));
    let (switch_b_status, receipt_b, switch_b_latency) = activate_embedding_candidate(
        &app,
        "nomic-ai/nomic-embed-text-v1.5",
        Some(&candidate_b),
        "switch",
        Some(first_a_generation),
    )
    .await;
    assert_eq!(switch_b_status, StatusCode::OK, "{receipt_b}");
    assert_eq!(receipt_b["kind"], "embedding");
    assert_eq!(receipt_b["status"], "switched");
    assert!(
        weak_a.upgrade().is_none(),
        "BERT A must be reclaimed by the public switch before NomicBert B is published"
    );
    assert!(first_a_registry.upgrade().is_none());
    assert!(first_a_tokenizer.upgrade().is_none());
    assert!(first_a_vocab.upgrade().is_none());
    let b_lease = state.acquire_embedding_model().expect("B generation");
    let b_model_id = b_lease.model.model_id.clone();
    assert_eq!(receipt_b["request_model"], b_model_id);
    let b_load = b_lease.model.load_timing;
    assert_eq!(
        receipt_b["timing_us"]["load_ready"].as_u64(),
        Some(b_load.weight_load_elapsed.as_micros() as u64)
    );
    assert_eq!(
        receipt_b["timing_us"]["post_warm"].as_u64(),
        Some(b_load.total_elapsed.as_micros() as u64)
    );
    let b_tokens = b_lease.model.encode("hello", true);
    assert_eq!(first_a_tokens.first(), Some(&101));
    assert_eq!(first_a_tokens.last(), Some(&102));
    assert_eq!(b_tokens.first(), Some(&101));
    assert_eq!(b_tokens.last(), Some(&102));
    let b_arch = b_lease.model.arch.as_ref().expect("production B arch");
    assert_eq!(b_arch.arch_name(), "nomic-bert");
    assert_eq!(b_arch.hidden_size(), 768);
    let b_generation = receipt_b["embedding"]["generation"]
        .as_u64()
        .expect("B generation receipt");
    let b_reclaimed_bytes = receipt_b["reclaimed_bytes"]
        .as_u64()
        .expect("B reclaimed bytes");
    let b_resident_bytes = receipt_b["resident_bytes"]
        .as_u64()
        .expect("B resident bytes");
    assert_eq!(b_generation, first_a_generation + 1);
    assert_eq!(b_reclaimed_bytes, a_resident_bytes);
    assert_eq!(b_resident_bytes, b_lease.model.resident_bytes());
    let runtime_b = embedding_runtime(&app).await;
    assert_eq!(runtime_b["embedding"]["generation"], b_generation);
    assert_eq!(runtime_b["embedding"]["arch"], "nomic-bert");
    assert_eq!(runtime_b["embedding"]["resident_bytes"], b_resident_bytes);
    assert!(runtime_b["embedding"].get("gguf_path").is_none());
    let weak_b = Arc::downgrade(&b_lease.model);
    let weak_b_registry = Arc::downgrade(&b_lease.model.registry);
    let weak_b_tokenizer = Arc::downgrade(&b_lease.model.tokenizer);
    let weak_b_vocab = Arc::downgrade(&b_lease.model.vocab);
    drop(b_lease);
    let (b, b_first_latency) = post_embedding(&app, &b_model_id).await;
    let (b_long, _) = post_embedding_text(&app, &b_model_id, LONG_EMBED_PROMPT).await;
    let switch_to_first_b = switch_b_started.elapsed();
    stop_b.store(true, std::sync::atomic::Ordering::Release);
    let peak_b_rss = peak_b_thread.join().expect("B RSS peak sampler");
    let b_steady = steady_embedding_median(&app, &b_model_id).await;
    let memory_b = current_process_memory();
    let mappings_b = assert_embedding_mapping_state(&path_b, &path_a, "B");
    assert_unit_embedding("NomicBert B", &b, 768);
    assert_ne!(
        first_a, b,
        "different families must not reuse an output buffer"
    );
    let (probe_old_a_status, probe_old_a, _) = activate_embedding_candidate(
        &app,
        "BAAI/bge-small-en-v1.5",
        Some(&candidate_a),
        "probe",
        None,
    )
    .await;
    assert_eq!(probe_old_a_status, StatusCode::NOT_FOUND);
    assert_eq!(probe_old_a["status"], "not_resident");
    assert_eq!(probe_old_a["embedding"]["generation"], b_generation);
    let (probe_b_status, probe_b, _) = activate_embedding_candidate(
        &app,
        "nomic-ai/nomic-embed-text-v1.5",
        Some(&candidate_b),
        "probe",
        None,
    )
    .await;
    assert_eq!(probe_b_status, StatusCode::OK);
    assert_eq!(probe_b["status"], "resident");
    assert_embedding_model_unavailable(&app, &model_a_id, "B must reject evicted A identity").await;

    let (stop_a2, peak_a2_thread) = start_embedding_rss_peak_sampler();
    let switch_a2_started = Instant::now();
    let (conflict_a2_status, conflict_a2, conflict_a2_latency) = activate_embedding_candidate(
        &app,
        "BAAI/bge-small-en-v1.5",
        Some(&candidate_a),
        "load",
        None,
    )
    .await;
    assert_eq!(conflict_a2_status, StatusCode::CONFLICT);
    assert_eq!(conflict_a2["code"], "embedding_model_resident");
    assert_eq!(conflict_a2["embedding"]["generation"], b_generation);
    let (switch_a2_status, receipt_a2, switch_a2_latency) = activate_embedding_candidate(
        &app,
        "BAAI/bge-small-en-v1.5",
        Some(&candidate_a),
        "switch",
        Some(b_generation),
    )
    .await;
    assert_eq!(switch_a2_status, StatusCode::OK, "{receipt_a2}");
    assert!(
        weak_b.upgrade().is_none(),
        "NomicBert B must be reclaimed by the public switch before fresh BERT A is published"
    );
    assert!(weak_b_registry.upgrade().is_none());
    assert!(weak_b_tokenizer.upgrade().is_none());
    assert!(weak_b_vocab.upgrade().is_none());
    let a2_lease = state.acquire_embedding_model().expect("fresh A generation");
    let a2_model_id = a2_lease.model.model_id.clone();
    assert_eq!(receipt_a2["request_model"], a2_model_id);
    let a2_load = a2_lease.model.load_timing;
    assert_eq!(
        receipt_a2["timing_us"]["load_ready"].as_u64(),
        Some(a2_load.weight_load_elapsed.as_micros() as u64)
    );
    assert_eq!(
        receipt_a2["timing_us"]["post_warm"].as_u64(),
        Some(a2_load.total_elapsed.as_micros() as u64)
    );
    assert_eq!(a2_lease.model.arch.as_ref().unwrap().arch_name(), "bert");
    assert_eq!(a2_lease.model.arch.as_ref().unwrap().hidden_size(), 384);
    assert_eq!(a2_lease.model.encode("hello", true), first_a_tokens);
    let a2_generation = receipt_a2["embedding"]["generation"]
        .as_u64()
        .expect("A2 generation receipt");
    let a2_reclaimed_bytes = receipt_a2["reclaimed_bytes"]
        .as_u64()
        .expect("A2 reclaimed bytes");
    let a2_resident_bytes = receipt_a2["resident_bytes"]
        .as_u64()
        .expect("A2 resident bytes");
    assert_eq!(a2_generation, b_generation + 1);
    assert_eq!(a2_reclaimed_bytes, b_resident_bytes);
    assert_eq!(a2_resident_bytes, a_resident_bytes);
    let runtime_a2 = embedding_runtime(&app).await;
    assert_eq!(runtime_a2["embedding"]["generation"], a2_generation);
    assert_eq!(runtime_a2["embedding"]["arch"], "bert");
    let weak_a2 = Arc::downgrade(&a2_lease.model);
    drop(a2_lease);
    let (second_a, a2_first_latency) = post_embedding(&app, &a2_model_id).await;
    let (second_a_long, _) = post_embedding_text(&app, &a2_model_id, LONG_EMBED_PROMPT).await;
    let switch_to_first_a2 = switch_a2_started.elapsed();
    stop_a2.store(true, std::sync::atomic::Ordering::Release);
    let peak_a2_rss = peak_a2_thread.join().expect("A2 RSS peak sampler");
    let a2_steady = steady_embedding_median(&app, &a2_model_id).await;
    let memory_a2 = current_process_memory();
    let mappings_a2 = assert_embedding_mapping_state(&path_a, &path_b, "A2");
    assert!(
        memory_a2.rss_bytes
            <= memory_a
                .rss_bytes
                .saturating_add(EMBEDDING_REPLAY_MEMORY_TOLERANCE_BYTES),
        "fresh A settled RSS must return within the measured replay bound: A={} A2={} tolerance={}",
        memory_a.rss_bytes,
        memory_a2.rss_bytes,
        EMBEDDING_REPLAY_MEMORY_TOLERANCE_BYTES
    );
    assert!(
        memory_a2.wired_bytes
            <= memory_a
                .wired_bytes
                .saturating_add(EMBEDDING_REPLAY_MEMORY_TOLERANCE_BYTES),
        "fresh A wired bytes must return within the measured replay bound: A={} A2={} tolerance={}",
        memory_a.wired_bytes,
        memory_a2.wired_bytes,
        EMBEDDING_REPLAY_MEMORY_TOLERANCE_BYTES
    );
    assert!(
        peak_a2_rss
            <= peak_a_rss.saturating_add(EMBEDDING_REPLAY_MEMORY_TOLERANCE_BYTES),
        "fresh A activation peak RSS must return within the measured replay bound: A={} A2={} tolerance={}",
        peak_a_rss,
        peak_a2_rss,
        EMBEDDING_REPLAY_MEMORY_TOLERANCE_BYTES
    );
    assert_eq!(first_a, second_a, "fresh BERT A must replay exactly");
    assert_eq!(
        first_a_long, second_a_long,
        "fresh BERT A must replay the longer prompt exactly"
    );
    let (probe_old_b_status, probe_old_b, _) = activate_embedding_candidate(
        &app,
        "nomic-ai/nomic-embed-text-v1.5",
        Some(&candidate_b),
        "probe",
        None,
    )
    .await;
    assert_eq!(probe_old_b_status, StatusCode::NOT_FOUND);
    assert_eq!(probe_old_b["status"], "not_resident");
    assert_eq!(probe_old_b["embedding"]["generation"], a2_generation);
    assert_embedding_model_unavailable(&app, &b_model_id, "A2 must reject evicted B identity")
        .await;
    let reference_a = reference_embedding(&path_a, "cls", "hello");
    let reference_a_long = reference_embedding(&path_a, "cls", LONG_EMBED_PROMPT);
    let reference_b = reference_embedding(&path_b, "mean", "hello");
    let reference_b_long = reference_embedding(&path_b, "mean", LONG_EMBED_PROMPT);
    let cosine_a = embedding_cosine(&first_a, &reference_a);
    let cosine_a_long = embedding_cosine(&first_a_long, &reference_a_long);
    let cosine_b = embedding_cosine(&b, &reference_b);
    let cosine_b_long = embedding_cosine(&b_long, &reference_b_long);
    assert!(
        cosine_a >= 0.999,
        "BERT reference cosine {cosine_a:.6} below 0.999"
    );
    assert!(
        cosine_a_long >= 0.999,
        "BERT long-prompt reference cosine {cosine_a_long:.6} below 0.999"
    );
    assert!(
        cosine_b >= 0.999,
        "NomicBert reference cosine {cosine_b:.6} below 0.999"
    );
    assert!(
        cosine_b_long >= 0.999,
        "NomicBert long-prompt reference cosine {cosine_b_long:.6} below 0.999"
    );
    assert_eq!(
        state
            .embedding_slot
            .read()
            .expect("embedding slot")
            .logical_resident_bytes(),
        a2_resident_bytes,
        "accounting must contain exactly one current embedding generation"
    );

    // A failed public replacement must not resurrect A2 or fall through to a
    // generative embedding path. Recovery is another explicit public load of
    // the exact A artifact and publishes a fresh generation.
    let invalid_dir = tempfile::tempdir().expect("invalid embedding fixture directory");
    let invalid_path = invalid_dir.path().join("header-valid-unloadable-q4_0.gguf");
    write_header_valid_but_unloadable_embedding(&invalid_path);
    let invalid_sha = crate::core::sha256::compute_file_sha256(&invalid_path)
        .expect("hash invalid embedding fixture");
    let invalid_candidate = register_real_embedding_candidate(
        &state,
        &invalid_path,
        "hf2q/test-invalid-embedding",
        "fixture-v1",
        &invalid_sha,
    );
    let (failed_status, failed_receipt, failed_activation_latency) = activate_embedding_candidate(
        &app,
        "hf2q/test-invalid-embedding",
        Some(&invalid_candidate),
        "switch",
        Some(a2_generation),
    )
    .await;
    assert_eq!(failed_status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(failed_receipt["code"], "embedding_activation_failed");
    assert_eq!(failed_receipt["embedding"]["generation"], a2_generation);
    assert_eq!(failed_receipt["embedding"]["resident_bytes"], 0);
    assert_eq!(failed_receipt["embedding"]["last_load_failed"], true);
    assert!(!failed_receipt
        .to_string()
        .contains(invalid_path.to_string_lossy().as_ref()));
    assert!(!failed_receipt.to_string().contains("vocab"));
    let failed_runtime = embedding_runtime(&app).await;
    assert_eq!(failed_runtime["embedding"]["generation"], a2_generation);
    assert_eq!(failed_runtime["embedding"]["resident_bytes"], 0);
    assert_eq!(failed_runtime["embedding"]["last_load_failed"], true);
    assert!(
        weak_a2.upgrade().is_none(),
        "failed replacement must reclaim A2 before attempting the invalid load"
    );

    let unavailable = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({"model":a2_model_id,"input":"hello"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("unavailable embedding response");
    assert_eq!(unavailable.status(), StatusCode::BAD_REQUEST);

    let (recovery_status, recovery_receipt, recovery_activation_latency) =
        activate_embedding_candidate(
            &app,
            "BAAI/bge-small-en-v1.5",
            Some(&candidate_a),
            "load",
            None,
        )
        .await;
    assert_eq!(recovery_status, StatusCode::OK, "{recovery_receipt}");
    assert_eq!(recovery_receipt["status"], "loaded");
    assert_eq!(
        recovery_receipt["embedding"]["generation"],
        a2_generation + 1
    );
    let recovered_runtime = embedding_runtime(&app).await;
    assert_eq!(
        recovered_runtime["embedding"]["generation"],
        a2_generation + 1
    );
    assert_eq!(recovered_runtime["embedding"]["last_load_failed"], false);
    assert_eq!(
        recovered_runtime["embedding"]["resident_bytes"],
        a_resident_bytes
    );
    let recovery_model_id = recovery_receipt["embedding"]["model_id"]
        .as_str()
        .expect("recovery model id");
    let (recovered_a, recovery_first_latency) = post_embedding(&app, recovery_model_id).await;
    assert_eq!(
        first_a, recovered_a,
        "failed-switch recovery must replay A exactly"
    );

    eprintln!(
        "real embedding public A→B→A: cosine_A_short={:.6} cosine_A_long={:.6} cosine_B_short={:.6} cosine_B_long={:.6} baseline_rss={} baseline_wired={} A_activation={}us A_unload={}us A_load={}us A_weight_load={}us A_warm={}us A_first={}us A_steady={}us initial_to_first={}us A_rss={} A_wired={} A_peak_rss={} A_map_lsof={} A_map_vmmap={} B_conflict={}us B_activation={}us B_unload={}us B_load={}us B_weight_load={}us B_warm={}us B_first={}us B_steady={}us switch_to_first_B={}us B_rss={} B_wired={} B_peak_rss={} B_map_lsof={} B_map_vmmap={} A2_conflict={}us A2_activation={}us A2_unload={}us A2_load={}us A2_weight_load={}us A2_warm={}us A2_first={}us A2_steady={}us switch_to_first_A2={}us A2_rss={} A2_wired={} A2_peak_rss={} A2_map_lsof={} A2_map_vmmap={} failed_activation={}us recovery_activation={}us recovery_first={}us A_bytes={} B_bytes={}",
        cosine_a,
        cosine_a_long,
        cosine_b,
        cosine_b_long,
        baseline_memory.rss_bytes,
        baseline_memory.wired_bytes,
        initial_activation_latency.as_micros(),
        initial_receipt["timing_us"]["unload"].as_u64().unwrap(),
        initial_load.total_elapsed.as_micros(),
        initial_load.weight_load_elapsed.as_micros(),
        initial_load.registry_warm_elapsed.as_micros(),
        first_a_latency.as_micros(),
        a_steady.as_micros(),
        initial_to_first.as_micros(),
        memory_a.rss_bytes,
        memory_a.wired_bytes,
        peak_a_rss,
        mappings_a.0.lsof_live,
        mappings_a.0.vmmap_live,
        conflict_b_latency.as_micros(),
        switch_b_latency.as_micros(),
        receipt_b["timing_us"]["unload"].as_u64().unwrap(),
        receipt_b["timing_us"]["load"].as_u64().unwrap(),
        b_load.weight_load_elapsed.as_micros(),
        b_load.registry_warm_elapsed.as_micros(),
        b_first_latency.as_micros(),
        b_steady.as_micros(),
        switch_to_first_b.as_micros(),
        memory_b.rss_bytes,
        memory_b.wired_bytes,
        peak_b_rss,
        mappings_b.0.lsof_live,
        mappings_b.0.vmmap_live,
        conflict_a2_latency.as_micros(),
        switch_a2_latency.as_micros(),
        receipt_a2["timing_us"]["unload"].as_u64().unwrap(),
        receipt_a2["timing_us"]["load"].as_u64().unwrap(),
        a2_load.weight_load_elapsed.as_micros(),
        a2_load.registry_warm_elapsed.as_micros(),
        a2_first_latency.as_micros(),
        a2_steady.as_micros(),
        switch_to_first_a2.as_micros(),
        memory_a2.rss_bytes,
        memory_a2.wired_bytes,
        peak_a2_rss,
        mappings_a2.0.lsof_live,
        mappings_a2.0.vmmap_live,
        failed_activation_latency.as_micros(),
        recovery_activation_latency.as_micros(),
        recovery_first_latency.as_micros(),
        a2_resident_bytes,
        b_resident_bytes,
    );
}

fn reference_embedding(path: &Path, pooling: &str, prompt: &str) -> Vec<f64> {
    let binary = std::env::var_os("HF2Q_EMBED_SWAP_E2E_REFERENCE_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/opt/llama.cpp/build/bin/llama-embedding".into());
    let output = std::process::Command::new(&binary)
        .args(["-m"])
        .arg(path)
        .args([
            "-p",
            prompt,
            "--pooling",
            pooling,
            "--embd-output-format",
            "raw",
            "--no-perf",
            // The embedding binary writes its machine-readable payload through
            // output-level `LOG(...)`, which is stdout. `--log-disable`
            // suppresses that payload as well as diagnostics and makes the
            // quality gate vacuous. Info/warning diagnostics use stderr, so
            // leaving logging enabled preserves a numeric-only stdout stream.
            "--offline",
        ])
        .output()
        .unwrap_or_else(|error| panic!("run reference embedding binary {binary:?}: {error}"));
    assert!(
        output.status.success(),
        "reference embedding failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("reference output UTF-8")
        .split_whitespace()
        .map(|value| value.parse::<f64>().expect("reference embedding number"))
        .collect()
}

fn embedding_cosine(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len(), "embedding width mismatch");
    let dot = left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
    let left_norm = left.iter().map(|value| value * value).sum::<f64>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    dot / (left_norm * right_norm)
}

fn assert_unit_embedding(label: &str, values: &[f64], expected_len: usize) {
    assert_eq!(values.len(), expected_len, "{label} output width");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{label} finite"
    );
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    assert!(
        (norm - 1.0).abs() <= 1.0e-5,
        "{label} must be unit normalized, norm={norm}"
    );
}
