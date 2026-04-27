//! ADR-005 Phase 2c iter-211 (W79) — Open WebUI vision multi-turn E2E
//! (Scenario 4 of the openwebui_* suite). Closes Phase 2c residual ACs:
//!
//!   * AC 3103: "Open WebUI with image uploads: full multi-turn vision
//!     chat works end-to-end."
//!   * AC 3106: "OpenAI-format `image_url` content parts (base64 data
//!     URIs) parse and route to ViT correctly."
//!
//! AC 3104 (vision accuracy gate) is BLOCKED on iter-119 HF-auth and
//! is NOT in scope here.
//!
//! # What this test exercises
//!
//! Open WebUI sends image uploads as the OpenAI `image_url` content-part
//! shape:
//!
//! ```json
//! {"role": "user", "content": [
//!   {"type": "text",      "text":  "Describe what you see."},
//!   {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
//! ]}
//! ```
//!
//! Testing the protocol path AT the OpenAI shape closes both ACs in one
//! test — Open WebUI is OpenAI-compatible, so Open-WebUI-against-hf2q
//! interop reduces to OpenAI-against-hf2q at the multimodal frontier.
//!
//! ## Scenarios
//!
//!   1. `scenario4_image_url_multi_turn` (LIVE) — turn 1 with image
//!      content parts → turn 2 text-only follow-up → T=0 determinism
//!      check on turn 1 → record/replay fixture (~50 KB cap).
//!   2. `image_url_jpeg_data_uri_supported` (LIVE) — same shape with a
//!      JPEG payload; confirms `parse_image_url` + `preprocess_gemma4v`
//!      handle the second mandated mime type symmetrically.
//!   3. `image_url_https_url_fetched` (LIVE, additionally gated on
//!      `HF2Q_NETWORK_TESTS=1`) — confirms HTTPS URLs are now fetched
//!      successfully via `fetch_https_image` (ADR-005 Phase 2c T1.4);
//!      previously asserted 400; now asserts 200 + non-empty content.
//!   4. `image_url_webp_data_uri_rejected` (LIVE) — confirms WEBP
//!      data URIs return a clear 400 (parse_image_url rejects mime
//!      types outside png/jpeg per `mod.rs:104-109`); same scope-doc
//!      role as scenario 3.
//!
//! All scenarios share a single `hf2q serve --mmproj` instance per
//! `--test-threads=1`. Spawn cost (~16 GB chat GGUF + 1.19 GB mmproj
//! cold-mmap warmup) dominates everything else.
//!
//! # Env gates
//!
//!   * `HF2Q_OPENWEBUI_E2E=1`              — required to run any LIVE
//!     test; absent → skip-with-note.
//!   * `HF2Q_OPENWEBUI_E2E_GGUF=<path>`    — chat GGUF override.
//!   * `HF2Q_OPENWEBUI_E2E_MMPROJ=<path>`  — mmproj GGUF override
//!     (default = `helpers::DEFAULT_MMPROJ_GGUF`).
//!   * `HF2Q_OPENWEBUI_E2E_RECORD=1`       — record turn-1 SSE chunks
//!     to `tests/fixtures/openwebui_multiturn/scenario4_vision_chunks.txt`.
//!     Subsequent runs without RECORD replay-and-assert.
//!   * `HF2Q_NETWORK_TESTS=1`              — additionally required for
//!     scenario 3 (`image_url_https_url_fetched`); enables real HTTPS
//!     egress to httpbin.org. Absent → test skips with a note.
//!
//! # Run
//!
//! ```ignore
//! HF2Q_OPENWEBUI_E2E=1 \
//! cargo test --release --test openwebui_vision -- --test-threads=1 --nocapture
//! ```

use std::path::PathBuf;

#[path = "openwebui_helpers/mod.rs"]
mod helpers;

use helpers::{
    assert_streaming_invariants, assistant_msg, base_url, fetch_canonical_model_id,
    nonstreaming_chat_status_and_body, replay_fixture_assert, streaming_chat_with_max_tokens,
    user_msg, user_msg_with_image_url, wait_for_readyz, write_fixture, ServerGuard,
    DEFAULT_CHAT_GGUF, DEFAULT_MMPROJ_GGUF, ENV_GATE, ENV_GGUF, ENV_MMPROJ, ENV_NETWORK,
    ENV_RECORD, HOST, PORT,
};

// ---------------------------------------------------------------------------
// Constants — fixture paths + image fixture loader
// ---------------------------------------------------------------------------

/// Recorded turn-1 SSE-chunks fixture for the vision scenario.
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/openwebui_multiturn/scenario4_vision_chunks.txt")
}

/// The four-dots-in-corners fixture used since iter-121 + the W56-W62
/// vision-encoder peer-overlap iters. Path is stable across the
/// vision-test family.
fn four_dots_png_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/vision/four_dots_in_corners_128x128.png")
}

/// Read PNG bytes → base64 → `data:image/png;base64,<...>` data URI.
fn read_as_data_uri(path: &PathBuf, mime: &str) -> String {
    use base64::Engine;
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("failed to read fixture {path:?}: {e}"));
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    format!("data:{mime};base64,{b64}")
}

/// Generate a tiny in-memory JPEG (8x8 black square) and return the
/// `data:image/jpeg;base64,...` data URI. JPEG is the second of the two
/// mime types `parse_image_url` accepts (`mod.rs:104-109`); a synthesized
/// fixture sidesteps adding another binary blob to the tree just to prove
/// JPEG decode works.
fn synthesize_jpeg_data_uri() -> String {
    use base64::Engine;
    use image::{ImageBuffer, Rgb};
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_pixel(8, 8, Rgb([0u8, 0, 0]));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg)
        .expect("encode synthetic 8x8 black JPEG");
    let bytes = buf.into_inner();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    format!("data:image/jpeg;base64,{b64}")
}

/// Generate a tiny WEBP placeholder data URI. We don't need the bytes
/// to be a real WEBP — `parse_image_url` rejects WEBP at the mime-type
/// guard before it ever decodes the payload (`src/inference/vision/mod.rs:104-109`),
/// and the `image` crate is opted in to `png + jpeg` features only
/// (`Cargo.toml:80`). The 400-response shape is the contract this
/// scenario verifies; the payload validity is irrelevant.
fn synthesize_webp_data_uri() -> String {
    use base64::Engine;
    let payload = b"placeholder-webp-bytes-that-never-decode";
    let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
    format!("data:image/webp;base64,{b64}")
}

// ---------------------------------------------------------------------------
// Common skip / spawn boilerplate
// ---------------------------------------------------------------------------

/// Returns `Some((gguf, mmproj))` when the gate is set + both fixtures
/// exist, `None` otherwise (test should print a skip note + return Ok).
fn resolve_fixtures_or_skip(test_name: &str) -> Option<(String, String)> {
    if std::env::var(ENV_GATE).as_deref() != Ok("1") {
        eprintln!(
            "{test_name}: {ENV_GATE} != \"1\" — skipping. Set {ENV_GATE}=1 to run \
             (loads ~17 GB of chat + mmproj GGUFs; first cold-mmap warmup is multi-minute)."
        );
        return None;
    }
    let gguf = std::env::var(ENV_GGUF).unwrap_or_else(|_| DEFAULT_CHAT_GGUF.into());
    let mmproj = std::env::var(ENV_MMPROJ).unwrap_or_else(|_| DEFAULT_MMPROJ_GGUF.into());
    if !PathBuf::from(&gguf).exists() {
        panic!(
            "{test_name}: chat GGUF not found at {gguf:?} — set {ENV_GGUF} or \
             place a fixture at the default path"
        );
    }
    if !PathBuf::from(&mmproj).exists() {
        panic!(
            "{test_name}: mmproj GGUF not found at {mmproj:?} — set {ENV_MMPROJ} or \
             run the iter-116b emit step"
        );
    }
    Some((gguf, mmproj))
}

// ---------------------------------------------------------------------------
// Scenario 4 — vision multi-turn (AC 3103 + 3106 primary closure)
// ---------------------------------------------------------------------------

/// Words that have appeared in mlx-lm and llama.cpp peer-overlap runs
/// against the four-dots fixture across the W56-W62 vision iters. The
/// model's response is "image-aware" if it contains at least one of
/// these. This is the same image-aware bar Phase 2c iter-132 used for
/// peer-precision-parity closure.
const PEER_OVERLAP_WORDS: &[&str] = &[
    "square", "squares", "frame", "corner", "corners", "four", "dots", "dot",
    "black", "white", "background", "pattern", "image", "shape", "shapes",
];

fn assert_image_aware(label: &str, text: &str) {
    let lower = text.to_lowercase();
    let hit = PEER_OVERLAP_WORDS
        .iter()
        .find(|w| lower.contains(*w))
        .copied();
    assert!(
        hit.is_some(),
        "{label}: response did not contain any peer-overlap word \
         (expected at least one of {PEER_OVERLAP_WORDS:?}); response={text:?}"
    );
    eprintln!(
        "{label}: image-aware OK — matched peer-overlap word {:?} in response",
        hit.unwrap_or("?")
    );
}

#[test]
fn scenario4_image_url_multi_turn() {
    let Some((gguf, mmproj)) = resolve_fixtures_or_skip("scenario4_image_url_multi_turn") else {
        return;
    };

    eprintln!(
        "openwebui_vision: spawning hf2q serve at {HOST}:{PORT} with model={gguf} \
         + mmproj={mmproj} (base_url={})",
        base_url()
    );
    let _server = ServerGuard::spawn_with_mmproj(&gguf, &mmproj)
        .expect("spawn hf2q serve --mmproj");
    wait_for_readyz();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    let model_id = rt.block_on(fetch_canonical_model_id());
    eprintln!("openwebui_vision: canonical model_id={model_id}");

    // Build the OpenAI multimodal turn-1 message with the four-dots
    // fixture as a base64 PNG data URI. The encoder produced this fixture
    // image-aware in iter-132 peer-precision-parity work; if the soft-token
    // path regressed between then and iter-211, this test fails loud.
    let four_dots_uri = read_as_data_uri(&four_dots_png_path(), "image/png");
    eprintln!(
        "openwebui_vision: four-dots data-URI prefix={:?}, len={}",
        &four_dots_uri[..four_dots_uri.find(',').unwrap_or(64).min(64)],
        four_dots_uri.len()
    );
    let user1_text = "Describe what you see.";
    let messages_t1 = vec![user_msg_with_image_url(user1_text, &four_dots_uri)];

    // -----------------------------------------------------------------
    // Turn 1 — image content parts → image-aware response
    // -----------------------------------------------------------------
    //
    // max_tokens=32: the chat-template emits a few formatting tokens
    // before the model says anything substantive; 16 (the default for
    // the iter-A `streaming_chat`) caps too tight for a peer-overlap
    // word to land. 32 is the same cap used by the operator-level
    // smoke `tests/vision_e2e_vs_mlx_vlm.rs` from iter-132.
    let (turn1_chunks, turn1_text) =
        rt.block_on(streaming_chat_with_max_tokens(&model_id, &messages_t1, 32));
    assert_streaming_invariants("turn1_vision", &turn1_chunks);
    assert!(
        !turn1_text.trim().is_empty(),
        "turn1 accumulated content was empty; chunks={:?}",
        turn1_chunks
    );
    eprintln!("openwebui_vision: turn1_text={turn1_text:?}");
    assert_image_aware("turn1_vision", &turn1_text);

    // -----------------------------------------------------------------
    // Turn 2 — text-only follow-up referencing image
    // -----------------------------------------------------------------
    //
    // The follow-up keeps the original image in the conversation history
    // so the model has the visual context to answer. We do NOT assert a
    // specific number; we assert (a) protocol invariants hold and (b) the
    // accumulated content is non-empty. Asking the model to "count"
    // exercises whether the multi-turn KV/embedding plumbing carries
    // image-derived state into the second turn.
    let messages_t2 = vec![
        user_msg_with_image_url(user1_text, &four_dots_uri),
        assistant_msg(&turn1_text),
        user_msg("How many objects did you count?"),
    ];
    let (turn2_chunks, turn2_text) =
        rt.block_on(streaming_chat_with_max_tokens(&model_id, &messages_t2, 32));
    assert_streaming_invariants("turn2_vision_followup", &turn2_chunks);
    assert!(
        !turn2_text.trim().is_empty(),
        "turn2 accumulated content was empty; chunks={:?}",
        turn2_chunks
    );
    eprintln!("openwebui_vision: turn2_text={turn2_text:?}");

    // -----------------------------------------------------------------
    // Determinism — re-run turn 1 at temperature=0
    // -----------------------------------------------------------------
    //
    // T=0 must produce byte-identical accumulated content even with the
    // image preprocessing + ViT forward + soft-token expansion in the
    // path. Same rationale as iter A's text-only determinism check
    // (`openwebui_multiturn.rs:194-209`); broader path here.
    let (rerun_chunks, rerun_text) =
        rt.block_on(streaming_chat_with_max_tokens(&model_id, &messages_t1, 32));
    assert_streaming_invariants("turn1_vision_rerun", &rerun_chunks);
    assert_eq!(
        turn1_text, rerun_text,
        "turn1 vision determinism violation at temperature=0:\n  first:  {turn1_text:?}\n  \
         rerun:  {rerun_text:?}"
    );
    eprintln!(
        "openwebui_vision: determinism PASS — turn1 byte-identical at temperature=0"
    );

    // -----------------------------------------------------------------
    // Fixture record / replay
    // -----------------------------------------------------------------
    let record = std::env::var(ENV_RECORD).as_deref() == Ok("1");
    let fp = fixture_path();
    if record {
        write_fixture(&fp, &turn1_chunks);
        eprintln!("openwebui_vision: recorded fixture at {fp:?}");
    } else if fp.exists() {
        replay_fixture_assert(&fp, &turn1_chunks);
        eprintln!("openwebui_vision: replayed fixture at {fp:?} — content-shape match");
    } else {
        eprintln!(
            "openwebui_vision: no fixture at {fp:?} — set {ENV_RECORD}=1 once \
             to record a baseline; subsequent runs replay-and-assert"
        );
    }
}

// ---------------------------------------------------------------------------
// Scenario 4b — JPEG data URI supported (companion to AC 3106)
// ---------------------------------------------------------------------------

#[test]
fn image_url_jpeg_data_uri_supported() {
    let Some((gguf, mmproj)) = resolve_fixtures_or_skip("image_url_jpeg_data_uri_supported") else {
        return;
    };

    let _server = ServerGuard::spawn_with_mmproj(&gguf, &mmproj)
        .expect("spawn hf2q serve --mmproj");
    wait_for_readyz();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    let model_id = rt.block_on(fetch_canonical_model_id());

    // Tiny synthetic 8×8 black JPEG. The image-awareness assertion is
    // weaker than scenario 4 (no peer-overlap-word check) because an
    // 8×8 black square has nothing semantic to describe; what we DO
    // assert is that the JPEG decode + preprocess + ViT forward + LM
    // forward path returns a non-empty assistant response over SSE
    // without erroring at the mime-type or decode boundary.
    let jpeg_uri = synthesize_jpeg_data_uri();
    eprintln!(
        "openwebui_vision: jpeg-data-uri len={} prefix={:?}",
        jpeg_uri.len(),
        &jpeg_uri[..jpeg_uri.find(',').unwrap_or(80).min(80)]
    );
    let messages = vec![user_msg_with_image_url(
        "Describe what you see in one sentence.",
        &jpeg_uri,
    )];
    let (chunks, text) =
        rt.block_on(streaming_chat_with_max_tokens(&model_id, &messages, 16));
    assert_streaming_invariants("jpeg_data_uri", &chunks);
    assert!(
        !text.trim().is_empty(),
        "jpeg companion: accumulated content empty; chunks={:?}",
        chunks
    );
    eprintln!("openwebui_vision: jpeg companion text={text:?}");
}

// ---------------------------------------------------------------------------
// Scenario 4c — WEBP data URI rejected (AC-3106 scope doc)
// ---------------------------------------------------------------------------

#[test]
fn image_url_webp_data_uri_rejected() {
    let Some((gguf, mmproj)) = resolve_fixtures_or_skip("image_url_webp_data_uri_rejected") else {
        return;
    };

    let _server = ServerGuard::spawn_with_mmproj(&gguf, &mmproj)
        .expect("spawn hf2q serve --mmproj");
    wait_for_readyz();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    let model_id = rt.block_on(fetch_canonical_model_id());
    let webp_uri = synthesize_webp_data_uri();

    // The handler maps `parse_image_url` mime-type rejection to a 400
    // invalid_request with `param = "messages[0].content[1]"` (handlers.rs:1707-1717).
    // We assert (a) status==400, (b) error.type == invalid_request_error,
    // (c) the message names the unsupported mime so an operator sees
    // *which* part of *which* message failed.
    let body = serde_json::json!({
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe."},
                    {"type": "image_url", "image_url": {"url": webp_uri}},
                ]
            }
        ],
        "stream": false,
        "max_tokens": 16,
        "temperature": 0,
    });
    let (status, json) = rt.block_on(nonstreaming_chat_status_and_body(body));
    assert_eq!(status, 400, "WEBP data URI: expected 400, got {status}; body={json}");
    let err_type = json["error"]["type"].as_str().unwrap_or("");
    assert_eq!(
        err_type, "invalid_request_error",
        "WEBP data URI: expected error.type=invalid_request_error; body={json}"
    );
    let err_msg = json["error"]["message"].as_str().unwrap_or("");
    assert!(
        err_msg.contains("image/webp") || err_msg.contains("not supported")
            || err_msg.contains("parse failed"),
        "WEBP data URI: error.message lacks identification of the unsupported \
         mime; body={json}"
    );
    eprintln!(
        "openwebui_vision: WEBP rejected (AC-3106 scope doc) — status=400 \
         error.message={err_msg:?}"
    );
}

// ---------------------------------------------------------------------------
// Scenario 4d — HTTPS URL fetched successfully (ADR-005 Phase 2c T1.4)
// ---------------------------------------------------------------------------
//
// Previously this scenario asserted a 400 because `load_image_bytes`
// unconditionally rejected HTTP(S) URLs. Since T1.4 (ADR-005 Phase 2c
// wave-1 Worker C) added `fetch_https_image` with a 10 s timeout and
// 20 MB cap, the path now fetches the remote image and preprocesses it.
//
// Fixture: `https://httpbin.org/image/png` returns a small (few-KB) PNG
// square — stable, no auth required, deterministic image format. The
// assertion does NOT check image-awareness (an httpbin test card is not
// the four-dots fixture the peer-overlap words were tuned on); it asserts
// only that the request succeeds (HTTP 200) and the response carries a
// non-empty assistant content, proving end-to-end HTTPS-fetch routing works.
//
// If httpbin.org is unreachable the fetch path returns 400 with a network
// error message. The test is gated on HF2Q_NETWORK_TESTS=1 to prevent
// unintended egress in offline CI.

#[test]
fn image_url_https_url_fetched() {
    if std::env::var(ENV_NETWORK).as_deref() != Ok("1") {
        eprintln!(
            "image_url_https_url_fetched: {ENV_NETWORK} != \"1\" — \
             skipping. Set {ENV_NETWORK}=1 alongside {ENV_GATE}=1 to run \
             (requires outbound HTTPS to httpbin.org)."
        );
        return;
    }
    let Some((gguf, mmproj)) = resolve_fixtures_or_skip("image_url_https_url_fetched")
    else {
        return;
    };

    let _server = ServerGuard::spawn_with_mmproj(&gguf, &mmproj)
        .expect("spawn hf2q serve --mmproj");
    wait_for_readyz();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    let model_id = rt.block_on(fetch_canonical_model_id());

    // httpbin.org/image/png returns a small PNG — stable public fixture.
    // The 10 s timeout in fetch_https_image covers typical latency; if the
    // host is unreachable this assertion fires with a clear 400 body.
    let https_url = "https://httpbin.org/image/png";
    let body = serde_json::json!({
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what you see."},
                    {"type": "image_url", "image_url": {"url": https_url}},
                ]
            }
        ],
        "stream": false,
        "max_tokens": 16,
        "temperature": 0,
    });
    let (status, json) = rt.block_on(nonstreaming_chat_status_and_body(body));
    assert_eq!(
        status, 200,
        "HTTPS URL fetch: expected 200, got {status}; body={json}"
    );
    // The response must carry a non-empty assistant content field.
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        !content.trim().is_empty(),
        "HTTPS URL fetch: expected non-empty assistant content; body={json}"
    );
    eprintln!(
        "openwebui_vision: HTTPS URL fetched OK (ADR-005 T1.4) — \
         status=200 content={content:?}"
    );
}
