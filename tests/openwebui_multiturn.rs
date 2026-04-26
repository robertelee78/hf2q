//! ADR-005 Phase 2a iter-133 (Iter A) — Open WebUI multi-turn streaming chat
//! E2E test (Scenario 1 of three).
//!
//! # What this test exercises
//!
//! Open WebUI is OpenAI-compatible. Testing OpenAI streaming chat-completions
//! semantics against `hf2q serve` IS testing Open WebUI compatibility — we do
//! NOT need to spin up Open WebUI itself; we test that hf2q's HTTP server
//! speaks the protocol Open WebUI expects.
//!
//! Scenario 1 covers the **text-stream multi-turn** portion of Phase 2a AC
//! line 2509 ("Open WebUI on separate host: multi-turn chat works (streaming,
//! tool use, reasoning-panel display). Image input required at 2c, not 2a."):
//!
//!   1. Spawn `hf2q serve` as a subprocess, point it at a real chat GGUF.
//!   2. Wait for `/readyz` 200 (engine warmup completes synchronously per
//!      iter-103 — see `cmd_serve` in `src/serve/mod.rs`).
//!   3. Resolve the canonical model id from `/v1/models` (per W42 iter-116i:
//!      the loaded model id is `general.name` from GGUF metadata, not the
//!      file stem).
//!   4. **Turn 1** — POST `/v1/chat/completions` with `stream: true`, parse
//!      SSE chunks, assert protocol invariants (role chunk first, content
//!      delta chunks, final chunk with `finish_reason`, `data: [DONE]`
//!      terminator).
//!   5. **Turn 2** — POST again with the prior turn's `assistant` reply
//!      appended to the message array; assert same SSE invariants.
//!   6. **Turn 3** — third user turn referencing earlier context; assert
//!      same SSE invariants.
//!   7. **Determinism** — re-run turn 1 at `temperature=0`, assert
//!      byte-identical accumulated content.
//!   8. **Non-streaming companion** — turn 1 with `stream: false`, assert
//!      response shape `{"id": ..., "object": "chat.completion", ...}`.
//!
//! Tool use (Scenario 2) and reasoning-panel display (Scenario 3) land in
//! iters B + C respectively; AC 2509 stays `[ ]` until all three scenarios
//! are green.
//!
//! # Env gates
//!
//! Default-off — matches the iter-101 `HF2Q_VISION_E2E` env-gate pattern.
//! Loading a 16GB chat GGUF + waiting for warmup is multi-minute on M5 Max;
//! every-`cargo-test` runs would be hostile.
//!
//!   * `HF2Q_OPENWEBUI_E2E=1`              — required to run the test at
//!     all; absent ⇒ test prints a skip note and returns Ok.
//!   * `HF2Q_OPENWEBUI_E2E_GGUF=<path>`    — chat GGUF (text-only is fine).
//!     Default falls back to the canonical gemma-4 chat GGUF used by
//!     `tests/mmproj_llama_cpp_compat.rs` and `tests/vision_e2e_vs_mlx_vlm.rs`.
//!   * `HF2Q_OPENWEBUI_E2E_RECORD=1`       — when set ALONGSIDE the gate,
//!     the test writes turn 1's SSE chunks to
//!     `tests/fixtures/openwebui_multiturn/turn1_chunks.txt` (one chunk
//!     per line, with `id` + `created` normalized). Subsequent runs without
//!     `_RECORD=1` replay the fixture and assert byte-identical chunk
//!     content modulo per-request fields.
//!
//! # Run
//!
//! ```ignore
//! HF2Q_OPENWEBUI_E2E=1 cargo test --release --test openwebui_multiturn \
//!     -- --test-threads=1 --nocapture
//! ```
//!
//! `--test-threads=1` per the OOM-prevention directive — only one
//! model-loading inference at a time.
//!
//! # Why subprocess spawning, not in-process axum
//!
//! Same rationale as `tests/vision_e2e_vs_mlx_vlm.rs` and
//! `tests/mmproj_llama_cpp_compat.rs`: subprocess testing exercises the
//! full server stack including the iter-103 synchronous warmup ordering,
//! the tokio runtime composition in `cmd_serve`, and the Engine FIFO
//! worker thread. An in-process axum oneshot would skip warmup and the
//! worker thread entirely, missing the integration surface that Open WebUI
//! actually consumes.

use std::path::PathBuf;

#[path = "openwebui_helpers/mod.rs"]
mod helpers;

use helpers::{
    assert_nonstreaming_invariants, assert_streaming_invariants, assistant_msg, base_url,
    fetch_canonical_model_id, nonstreaming_chat, replay_fixture_assert, streaming_chat,
    user_msg, wait_for_readyz, write_fixture, ServerGuard, DEFAULT_CHAT_GGUF, ENV_GATE,
    ENV_GGUF, ENV_RECORD, HOST, PORT,
};

/// Fixture path for the turn-1 SSE chunk recording. Truncated at
/// `helpers::FIXTURE_MAX_BYTES` (~50KB) per the iter-133 directive — turn 1
/// with `max_tokens=16` fits easily.
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/openwebui_multiturn/turn1_chunks.txt")
}

#[test]
fn openwebui_multiturn_streaming_chat_scenario_1() {
    if std::env::var(ENV_GATE).as_deref() != Ok("1") {
        eprintln!(
            "{ENV_GATE} != \"1\" — skipping. Set {ENV_GATE}=1 to run \
             (loads a 16GB chat GGUF; first cold-mmap warmup is multi-minute)."
        );
        return;
    }

    let gguf = std::env::var(ENV_GGUF).unwrap_or_else(|_| DEFAULT_CHAT_GGUF.into());
    if !PathBuf::from(&gguf).exists() {
        panic!(
            "chat GGUF not found at {gguf:?} — set {ENV_GGUF} to a valid \
             text-capable chat GGUF, or place a fixture at the default path"
        );
    }

    eprintln!(
        "openwebui_multiturn: spawning hf2q serve at {HOST}:{PORT} with model={gguf} \
         (base_url={})",
        base_url()
    );
    let _server = ServerGuard::spawn(&gguf).expect("spawn hf2q serve");
    wait_for_readyz();

    // Build the runtime that drives every async request. Single-threaded —
    // every assertion is synchronous from the operator's perspective. We
    // deliberately do NOT use `#[tokio::test]` (would force a shared
    // runtime config); the current shape matches `tests/serve_ux.rs`.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    let model_id = rt.block_on(fetch_canonical_model_id());
    eprintln!("openwebui_multiturn: canonical model_id={model_id}");

    let user1 = "Say hello in one word.";
    let user2 = "Now say goodbye in one word.";
    let user3 = "What did I ask first?";

    // -----------------------------------------------------------------
    // Turn 1: streaming
    // -----------------------------------------------------------------
    let messages_t1 = vec![user_msg(user1)];
    let (turn1_chunks, turn1_text) = rt.block_on(streaming_chat(&model_id, &messages_t1));
    assert_streaming_invariants("turn1", &turn1_chunks);
    assert!(
        !turn1_text.trim().is_empty(),
        "turn1 accumulated content was empty; chunks={:?}",
        turn1_chunks
    );
    eprintln!("openwebui_multiturn: turn1_text={turn1_text:?}");

    // -----------------------------------------------------------------
    // Turn 2: streaming, multi-turn (turn 1 reply baked into history)
    // -----------------------------------------------------------------
    let messages_t2 = vec![
        user_msg(user1),
        assistant_msg(&turn1_text),
        user_msg(user2),
    ];
    let (turn2_chunks, turn2_text) = rt.block_on(streaming_chat(&model_id, &messages_t2));
    assert_streaming_invariants("turn2", &turn2_chunks);
    assert!(
        !turn2_text.trim().is_empty(),
        "turn2 accumulated content was empty; chunks={:?}",
        turn2_chunks
    );
    eprintln!("openwebui_multiturn: turn2_text={turn2_text:?}");

    // -----------------------------------------------------------------
    // Turn 3: streaming, three-turn (asks the model to recall earlier context)
    // -----------------------------------------------------------------
    //
    // We don't gate on substring matches against the original prompts —
    // the chat-model's quality is not the bar (protocol compliance is).
    // What we DO gate is that the response is non-empty: a multi-turn
    // history that produced an empty stream would mean the SSE pipeline
    // collapsed.
    let messages_t3 = vec![
        user_msg(user1),
        assistant_msg(&turn1_text),
        user_msg(user2),
        assistant_msg(&turn2_text),
        user_msg(user3),
    ];
    let (turn3_chunks, turn3_text) = rt.block_on(streaming_chat(&model_id, &messages_t3));
    assert_streaming_invariants("turn3", &turn3_chunks);
    assert!(
        !turn3_text.trim().is_empty(),
        "turn3 accumulated content was empty; chunks={:?}",
        turn3_chunks
    );
    eprintln!("openwebui_multiturn: turn3_text={turn3_text:?}");

    // -----------------------------------------------------------------
    // Determinism check — re-run turn 1 at temperature=0
    // -----------------------------------------------------------------
    //
    // OpenAI compatibility's "multi-turn chat works" claim implicitly
    // requires reproducibility under `temperature=0` for the AC's value
    // proposition (the operator can ask the same thing twice and not get
    // gibberish). The accumulated content must match byte-for-byte; we
    // do NOT assert byte-identity on the chunk *boundaries* because TCP
    // packet boundaries shift the SSE framing and that's not a protocol
    // violation. We assert byte-identity on the *content text*.
    let (rerun_chunks, rerun_text) = rt.block_on(streaming_chat(&model_id, &messages_t1));
    assert_streaming_invariants("turn1_rerun", &rerun_chunks);
    assert_eq!(
        turn1_text, rerun_text,
        "turn1 determinism violation at temperature=0:\n  first:  {turn1_text:?}\n  rerun:  {rerun_text:?}"
    );
    eprintln!("openwebui_multiturn: determinism PASS — turn1 byte-identical at temperature=0");

    // -----------------------------------------------------------------
    // Non-streaming companion — turn 1 with stream:false
    // -----------------------------------------------------------------
    //
    // The same handler (`chat_completions`) forks on `req.stream` at
    // `src/serve/api/handlers.rs:80`. Exercising the non-streaming branch
    // with the same prompt both confirms the wire shape OpenAI clients
    // see when streaming is off, and confirms that determinism (above)
    // is in the engine, not in the streaming chunker.
    let nonstream_resp = rt.block_on(nonstreaming_chat(&model_id, &messages_t1));
    assert_nonstreaming_invariants(&nonstream_resp);
    eprintln!(
        "openwebui_multiturn: non-streaming companion PASS — id={}, finish_reason={}",
        nonstream_resp["id"].as_str().unwrap_or("<missing>"),
        nonstream_resp["choices"][0]["finish_reason"].as_str().unwrap_or("<missing>")
    );

    // -----------------------------------------------------------------
    // Fixture record / replay
    // -----------------------------------------------------------------
    //
    // The recorded fixture is a regression check beyond "the test passed":
    // it pins the SSE wire shape (chunk count, frame-boundary semantics,
    // delta sequence, terminator) so a future refactor that *passes the
    // assertions* but emits a different stream-shape regresses loudly.
    let record = std::env::var(ENV_RECORD).as_deref() == Ok("1");
    let fp = fixture_path();
    if record {
        write_fixture(&fp, &turn1_chunks);
        eprintln!("openwebui_multiturn: recorded fixture at {fp:?}");
    } else if fp.exists() {
        replay_fixture_assert(&fp, &turn1_chunks);
        eprintln!("openwebui_multiturn: replayed fixture at {fp:?} — content-shape match");
    } else {
        eprintln!(
            "openwebui_multiturn: no fixture at {fp:?} — set {ENV_RECORD}=1 once \
             to record a baseline; subsequent runs replay-and-assert"
        );
    }
}
