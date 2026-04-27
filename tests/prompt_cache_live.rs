//! ADR-005 Phase 2a T1.10 — PromptCache live-validation harness.
//!
//! # What this test exercises
//!
//! The PromptCache (`engine.rs:385-504`) is a full-equality + full-response-
//! replay cache owned by the worker thread inside `LoadedModel`.  It was
//! introduced in iter-96 (commit a9eeb54) under a "live-validation-gated"
//! label in the wave-1 hive-mind snapshot.
//!
//! Wave-2 T1.10 audit (2026-04-26) confirmed the cache is already un-gated:
//! `lookup` fires unconditionally at `engine.rs:1138` and `store` fires at
//! `:1428` (non-streaming) and `:2129` (streaming).  No `if false`, no
//! `const ENABLED: bool = false`, no `HF2Q_PROMPT_CACHE_DISABLED` env var
//! was ever present.  The task reduces to: write the proof (this test) that
//! was always missing.
//!
//! # Cache contract being validated
//!
//! 1. **Miss → `cached_tokens = 0`** in `usage.prompt_tokens_details`.
//! 2. **Hit → `cached_tokens = prompt_len`**, content byte-identical to miss.
//! 3. **Hit is faster** — `x_hf2q_timing.decode_time_secs = 0.0` on hit
//!    (full decode skipped); > 0 on miss.
//! 4. **Invalidation** — a different prompt gets a fresh miss, not a stale
//!    hit; the response is free to differ.
//!
//! Only non-streaming requests are tested here.  The streaming path stores
//! to the cache on completion but does NOT consult it on input (per the
//! comment at `engine.rs:2108-2113`): "Streaming currently does NOT consult
//! the cache on input — would require fake-emitting Delta events from cached
//! text — iter-97 follow-up."  That asymmetry is by design; this test
//! exercises the path that does consult the cache.
//!
//! # Env gate
//!
//! Default-off per the iter-101 env-gate pattern.  Loading a 16GB chat
//! GGUF + warmup is multi-minute on M5 Max; always-on would be hostile.
//!
//!   * `HF2Q_PROMPT_CACHE_LIVE_TEST=1`   — required to run; absent ⇒ skip.
//!   * `HF2Q_PROMPT_CACHE_LIVE_GGUF=<path>` — GGUF to serve.  Falls back
//!     to the canonical gemma-4-26B chat GGUF used by other E2E tests.
//!
//! # Run
//!
//! ```ignore
//! HF2Q_PROMPT_CACHE_LIVE_TEST=1 cargo test --release --test prompt_cache_live \
//!     -- --test-threads=1 --nocapture
//! ```
//!
//! `--test-threads=1` per the OOM-prevention directive — only one model-
//! loading inference process at a time.
//!
//! # Validation corpus (wave-2 T1.10, 2026-04-26)
//!
//! The three-prompt corpus used to validate the un-gated cache:
//!
//!   Prompt A: "Say hello in one word."
//!   Prompt B: "Say goodbye in one word."
//!
//! Both are short, deterministic (temperature=0), and produce stable greedy
//! outputs on Gemma 4 26B.  They are strictly different so a false-positive
//! hit across A→B would surface immediately.

#![allow(dead_code)]

use std::path::PathBuf;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ENV_GATE: &str = "HF2Q_PROMPT_CACHE_LIVE_TEST";
const ENV_GGUF: &str = "HF2Q_PROMPT_CACHE_LIVE_GGUF";
const DEFAULT_CHAT_GGUF: &str = concat!(
    "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/",
    "gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf"
);
const HOST: &str = "127.0.0.1";
/// Port distinct from mmproj_llama_cpp_compat (52226), vision_e2e_vs_mlx_vlm
/// (18181), and openwebui_multiturn (52334).
const PORT: u16 = 52335;
const READYZ_BUDGET_SECS: u64 = 600;
const REQUEST_TIMEOUT_SECS: u64 = 180;
const MAX_TOKENS: u64 = 16;

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

struct ServerGuard(std::process::Child);

impl ServerGuard {
    fn spawn(gguf: &str) -> std::io::Result<Self> {
        use std::process::{Command, Stdio};
        let bin = std::env::var("CARGO_BIN_EXE_hf2q").unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target/release/hf2q")
                .to_string_lossy()
                .into_owned()
        });
        let child = Command::new(&bin)
            .args([
                "serve",
                "--model",
                gguf,
                "--host",
                HOST,
                "--port",
                &PORT.to_string(),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        Ok(Self(child))
    }
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Poll `/readyz` until 200 or timeout.
fn wait_for_readyz() {
    use std::io::{Read, Write};
    use std::net::TcpStream;

    let started = Instant::now();
    let mut last_err: Option<String> = None;
    while started.elapsed().as_secs() < READYZ_BUDGET_SECS {
        let result = (|| -> std::io::Result<u16> {
            let mut s = TcpStream::connect_timeout(
                &format!("{HOST}:{PORT}")
                    .parse()
                    .map_err(std::io::Error::other)?,
                Duration::from_secs(5),
            )?;
            s.set_read_timeout(Some(Duration::from_secs(5)))?;
            s.write_all(
                format!(
                    "GET /readyz HTTP/1.1\r\nHost: {HOST}:{PORT}\r\nConnection: close\r\n\r\n"
                )
                .as_bytes(),
            )?;
            let mut head = [0u8; 64];
            let n = s.read(&mut head)?;
            let head_s = std::str::from_utf8(&head[..n]).unwrap_or("");
            let code = head_s
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<u16>().ok())
                .ok_or_else(|| {
                    std::io::Error::other(format!("malformed HTTP status line: {head_s:?}"))
                })?;
            Ok(code)
        })();
        match result {
            Ok(200) => {
                eprintln!(
                    "prompt_cache_live: /readyz=200 after {}s",
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
        "prompt_cache_live: /readyz did not reach 200 within {READYZ_BUDGET_SECS}s; \
         last_err={}",
        last_err.unwrap_or_else(|| "<none>".into())
    );
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// GET `/v1/models` → first chat-eligible model id (context_length != null).
async fn fetch_model_id(client: &reqwest::Client) -> String {
    let url = format!("http://{HOST}:{PORT}/v1/models");
    let resp = client
        .get(&url)
        .send()
        .await
        .expect("GET /v1/models failed");
    assert_eq!(
        resp.status().as_u16(),
        200,
        "/v1/models status != 200: {:?}",
        resp.status()
    );
    let v: serde_json::Value = resp.json().await.expect("parse /v1/models JSON");
    let data = v
        .get("data")
        .and_then(|d| d.as_array())
        .unwrap_or_else(|| panic!("/v1/models missing data array: {v}"));
    let id = data
        .iter()
        .find(|m| {
            m.get("context_length")
                .map(|cl| !cl.is_null())
                .unwrap_or(false)
        })
        .and_then(|m| m.get("id"))
        .and_then(|s| s.as_str())
        .unwrap_or_else(|| {
            v["data"][0]["id"]
                .as_str()
                .expect("no model id in /v1/models response")
        });
    id.to_string()
}

/// POST a non-streaming chat-completions request with a single user turn at
/// temperature=0 (greedy — cache-eligible).  Returns the parsed JSON.
async fn chat_once(
    client: &reqwest::Client,
    model_id: &str,
    user_text: &str,
) -> serde_json::Value {
    let body = serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": user_text}],
        "stream": false,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    });
    let url = format!("http://{HOST}:{PORT}/v1/chat/completions");
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .expect("POST /v1/chat/completions failed");
    let status = resp.status().as_u16();
    let text = resp
        .text()
        .await
        .unwrap_or_else(|_| "<unreadable>".into());
    assert_eq!(
        status,
        200,
        "/v1/chat/completions status != 200: body={text}"
    );
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("response is not JSON ({e}): {text}"))
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

fn content_text(resp: &serde_json::Value) -> &str {
    resp["choices"][0]["message"]["content"]
        .as_str()
        .expect("missing choices[0].message.content")
}

fn cached_tokens(resp: &serde_json::Value) -> u64 {
    resp["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0)
}

fn prompt_tokens(resp: &serde_json::Value) -> u64 {
    resp["usage"]["prompt_tokens"]
        .as_u64()
        .expect("missing usage.prompt_tokens")
}

fn decode_time_secs(resp: &serde_json::Value) -> f64 {
    resp["x_hf2q_timing"]["decode_time_secs"]
        .as_f64()
        .unwrap_or(-1.0)
}

fn prefill_time_secs(resp: &serde_json::Value) -> f64 {
    resp["x_hf2q_timing"]["prefill_time_secs"]
        .as_f64()
        .unwrap_or(-1.0)
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

/// Live validation for the engine's PromptCache (wave-2 T1.10).
///
/// Sends three non-streaming greedy requests to a running hf2q server and
/// asserts the cache hit/miss contract described in the module doc.
#[test]
fn prompt_cache_hit_miss_and_invalidation() {
    if std::env::var(ENV_GATE).as_deref() != Ok("1") {
        eprintln!(
            "{ENV_GATE} != \"1\" — skipping. Set {ENV_GATE}=1 to run \
             (loads a chat GGUF; cold warmup is multi-minute)."
        );
        return;
    }

    let gguf = std::env::var(ENV_GGUF).unwrap_or_else(|_| DEFAULT_CHAT_GGUF.into());
    if !PathBuf::from(&gguf).exists() {
        panic!(
            "chat GGUF not found at {gguf:?} — set {ENV_GGUF}=<path> or place a \
             GGUF at the default path"
        );
    }

    eprintln!(
        "prompt_cache_live: spawning hf2q serve at {HOST}:{PORT} model={gguf}"
    );
    let _server = ServerGuard::spawn(&gguf).expect("spawn hf2q serve");
    wait_for_readyz();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    rt.block_on(async {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .build()
            .expect("build reqwest client");

        let model_id = fetch_model_id(&client).await;
        eprintln!("prompt_cache_live: model_id={model_id}");

        // ── Prompt A, first send: cache MISS ─────────────────────────────
        //
        // The worker thread has no prior cached prompt, so this request
        // runs the full prefill+decode pipeline.  Expected:
        //   cached_tokens == 0
        //   decode_time_secs > 0
        let prompt_a = "Say hello in one word.";
        let miss = chat_once(&client, &model_id, prompt_a).await;
        let miss_content = content_text(&miss).to_string();
        let miss_cached = cached_tokens(&miss);
        let miss_decode_secs = decode_time_secs(&miss);
        let prompt_len = prompt_tokens(&miss);

        eprintln!(
            "prompt_cache_live: prompt_A MISS — \
             content={miss_content:?} cached_tokens={miss_cached} \
             decode_secs={miss_decode_secs:.4} prompt_tokens={prompt_len}"
        );
        assert_eq!(
            miss_cached,
            0,
            "Expected cache MISS (cached_tokens=0) on first request; got {miss_cached}. \
             The PromptCache struct initialises with an empty token slice so the first \
             request can never hit."
        );
        assert!(
            !miss_content.trim().is_empty(),
            "Cache MISS produced empty content — model may have failed to decode."
        );
        assert!(
            miss_decode_secs > 0.0,
            "Cache MISS had decode_time_secs={miss_decode_secs:.6} which is not > 0; \
             the decode pipeline should have run."
        );

        // ── Prompt A, second send: cache HIT ─────────────────────────────
        //
        // Identical prompt + greedy decode → the worker thread's
        // PromptCache.lookup() returns the previously cached
        // GenerationResult without running the decoder.  Expected:
        //   cached_tokens == prompt_len (full prompt count, per iter-96 spec)
        //   decode_time_secs == 0.0     (full decode skipped)
        //   content == miss_content     (byte-identical replay)
        let hit = chat_once(&client, &model_id, prompt_a).await;
        let hit_content = content_text(&hit).to_string();
        let hit_cached = cached_tokens(&hit);
        let hit_decode_secs = decode_time_secs(&hit);
        let hit_prefill_secs = prefill_time_secs(&hit);

        eprintln!(
            "prompt_cache_live: prompt_A HIT — \
             content={hit_content:?} cached_tokens={hit_cached} \
             decode_secs={hit_decode_secs:.4} prefill_secs={hit_prefill_secs:.4}"
        );

        assert_eq!(
            hit_content,
            miss_content,
            "Cache HIT content differs from cache MISS:\n  miss: {miss_content:?}\n   hit: {hit_content:?}\n\
             Full-response-replay must produce byte-identical output."
        );
        assert_eq!(
            hit_cached,
            prompt_len,
            "Cache HIT: expected cached_tokens={prompt_len} (full prompt length), \
             got {hit_cached}. The PromptCache stores the prompt token slice and \
             sets cached_tokens=prompt_len on replay (engine.rs:474)."
        );
        assert_eq!(
            hit_decode_secs,
            0.0,
            "Cache HIT: expected decode_time_secs=0.0 (decode skipped), \
             got {hit_decode_secs:.6}. The cache fast-path returns \
             decode_duration=Duration::ZERO (engine.rs:472)."
        );
        assert_eq!(
            hit_prefill_secs,
            0.0,
            "Cache HIT: expected prefill_time_secs=0.0 (prefill skipped), \
             got {hit_prefill_secs:.6}. The cache fast-path returns \
             prefill_duration=Duration::ZERO (engine.rs:471)."
        );
        eprintln!(
            "prompt_cache_live: HIT latency is effectively 0 decode; MISS \
             decode_secs={miss_decode_secs:.4} — cache is working."
        );

        // ── Prompt B: cache MISS (invalidation check) ────────────────────
        //
        // A different prompt must not return the cached prompt-A response.
        // The cache holds only the single most-recent result; prompt B has
        // a different token sequence so lookup() returns None (O(N) equality
        // compare fails at the first differing token).  Expected:
        //   cached_tokens == 0
        //   content may differ from prompt A (no correctness assertion — only
        //   the miss-shape is required)
        let prompt_b = "Say goodbye in one word.";
        let b_miss = chat_once(&client, &model_id, prompt_b).await;
        let b_content = content_text(&b_miss).to_string();
        let b_cached = cached_tokens(&b_miss);
        let b_decode_secs = decode_time_secs(&b_miss);

        eprintln!(
            "prompt_cache_live: prompt_B MISS — \
             content={b_content:?} cached_tokens={b_cached} \
             decode_secs={b_decode_secs:.4}"
        );
        assert_eq!(
            b_cached,
            0,
            "Expected cache MISS (cached_tokens=0) for a NEW prompt; got {b_cached}. \
             A cache hit here would mean prompt B is matching prompt A's cached \
             token sequence — a correctness bug in PromptCache.lookup()."
        );
        assert!(
            b_decode_secs > 0.0,
            "Prompt B cache MISS had decode_time_secs={b_decode_secs:.6}; \
             expected > 0 (fresh decode)."
        );
        // The response CONTENT for prompt B is intentionally left unchecked:
        // model quality is not the bar — only the cache-miss shape matters.

        eprintln!(
            "prompt_cache_live: ALL ASSERTIONS PASSED\n\
             \n\
             Summary (wave-2 T1.10 live validation, 2026-04-26):\n\
             \n\
             model:             {model_id}\n\
             corpus:            [\"Say hello in one word.\", \"Say goodbye in one word.\"]\n\
             prompt_A_MISS:     cached_tokens=0  decode_secs={miss_decode_secs:.4}\n\
             prompt_A_HIT:      cached_tokens={hit_cached}  decode_secs={hit_decode_secs:.4}  (content byte-identical)\n\
             prompt_B_MISS:     cached_tokens=0  decode_secs={b_decode_secs:.4}\n\
             \n\
             The PromptCache is live and correct. No un-gate required — the cache\n\
             was never gated. This test is the proof that was missing."
        );
    });
}
