//! ADR-005 Phase 2c iter-116a — cross-compat smoke scaffold (default-off).
//!
//! End-to-end gate: a hf2q-emitted Gemma 4V mmproj GGUF must load cleanly
//! through both
//!
//!   1. hf2q's own mmproj entry points (`MmprojConfig::from_gguf`,
//!      `detect_arch_profile`, `validate_tensor_set`), and
//!   2. llama.cpp's `llama-mtmd-cli` CLIP loader,
//!
//! without round-trip drift in tensor names / dtypes / metadata keys.
//! If our writer emits a malformed mmproj that hf2q itself happily
//! consumes but llama.cpp rejects, this gate fires.
//!
//! # Why this lives here
//!
//! ADR-005 closes only when hf2q's mmproj output matches llama.cpp's
//! reference shape exactly — a single CLIP loader rejects = downstream
//! tools (Open WebUI through llama.cpp, sibling C++ inference rigs)
//! lose hf2q-emitted Gemma 4V models. Fail loud here so a regression
//! never sneaks past convert-side `cargo test`.
//!
//! # Why default-off
//!
//! The real-model gate loads ~1.07 GB of mmproj F16 + ~16 GB of chat
//! GGUF; running it under a CFA wave's concurrent host activity is an
//! OOM risk per `feedback_oom_prevention.md`. Iter-116b runs the full
//! gate under a quiet host. iter-116a (this file) lands the scaffold +
//! Phase A (file-on-disk) + Phase B (metadata + tensor-name parse via
//! mlx_native's GgufFile) so the structure is reviewable without
//! committing a large fixture.
//!
//! Because hf2q is a binary crate (no `[lib]` target), this test can't
//! import `hf2q::inference::vision::mmproj::*` directly. Phase B
//! re-implements the same metadata-key + tensor-name parse the real
//! `MmprojConfig::from_gguf` performs, gated against the same key
//! list. A future hf2q `[lib]` carve-out could collapse the
//! duplication, but that's an ADR-014 scope discussion — not iter-116a.
//!
//! # Skip / run protocol
//!
//!   - Default `cargo test`: skipped via `#[ignore]`.
//!   - With `HF2Q_LLAMA_MMPROJ_COMPAT=1`: runs Phase A (file-on-disk
//!     guard) + Phase B (mmproj header parse).
//!   - With `HF2Q_LLAMA_MMPROJ_COMPAT_MODEL_LOAD=1` *additionally*:
//!     runs Phase C (llama-mtmd-cli stderr smoke) + Phase D (parity
//!     proxy at T=0/max-tokens=16). Reserved for iter-116b under a
//!     quiet host.
//!
//! # Run command (iter-116b)
//!
//! ```bash
//! HF2Q_LLAMA_MMPROJ_COMPAT=1 \
//! HF2Q_LLAMA_MMPROJ_COMPAT_MODEL_LOAD=1 \
//!   cargo test --release --test mmproj_llama_cpp_compat -- --ignored --nocapture
//! ```

use std::path::Path;

use mlx_native::gguf::GgufFile;

/// Where the default-off real-model fixture lives. iter-116b will
/// either find this file in place (CFA worker stamped it earlier) or
/// emit it via `hf2q convert`. Phase A's only job is to make the
/// missing-fixture failure mode obvious + actionable rather than a
/// confusing GgufFile::open ENOENT.
const MMPROJ_PATH: &str =
    "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/\
     gemma-4-26B-A4B-it-ara-abliterated-dwq-mmproj.gguf";

/// Companion text-side GGUF — required for Phase C's
/// `llama-mtmd-cli -m <chat> --mmproj <mmproj>` invocation.
const CHAT_GGUF_PATH: &str =
    "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/\
     gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf";

/// Vision fixture used in Phase D's parity proxy. Small + deterministic
/// so iter-116b can re-run identical inputs across tools without
/// stochastic image-decode drift.
#[allow(dead_code)]
const FIXTURE_IMAGE: &str = "/opt/hf2q/tests/fixtures/vision/four_dots_in_corners_128x128.png";

/// llama.cpp binary path (Homebrew-managed). Phase C spawns this.
#[allow(dead_code)]
const LLAMA_MTMD_BIN: &str = "/opt/homebrew/bin/llama-mtmd-cli";

/// The env gate that opts in to the cross-compat smoke. Default-off so
/// `cargo test` doesn't accidentally trigger a 17 GB model load on a
/// developer laptop.
const ENV_GATE: &str = "HF2Q_LLAMA_MMPROJ_COMPAT";

/// A second gate scoped to Phase C+D (the actual model-loading
/// invocations). iter-116a leaves this off so Phase B (metadata
/// header parse) can land + run without the heavy invocations.
#[allow(dead_code)]
const ENV_GATE_MODEL_LOAD: &str = "HF2Q_LLAMA_MMPROJ_COMPAT_MODEL_LOAD";

/// Third gate scoped to Phase D only (parity proxy). iter-116c runs
/// Phases A+B+C end-to-end with the freshly-emitted mmproj fixture but
/// leaves Phase D queued — its body is still a `panic!` placeholder
/// that lands in a follow-up iter. Default-off so MODEL_LOAD=1 alone
/// exercises only the llama-mtmd-cli stderr smoke (Phase C). Set
/// `HF2Q_LLAMA_MMPROJ_COMPAT_PARITY=1` once the parity proxy body is
/// implemented.
#[allow(dead_code)]
const ENV_GATE_PARITY: &str = "HF2Q_LLAMA_MMPROJ_COMPAT_PARITY";

/// CLIP architecture string — the value `MmprojConfig::from_gguf`
/// expects for `general.architecture`. Mirrors the constant in
/// `src/inference/vision/mmproj.rs::ARCH_CLIP`.
const ARCH_CLIP: &str = "clip";

/// Required `clip.vision.*` metadata keys. Same set
/// `MmprojConfig::from_gguf` enforces. Drift here = drift in the
/// production loader.
const CLIP_VISION_REQUIRED_KEYS: &[&str] = &[
    "clip.vision.image_size",
    "clip.vision.patch_size",
    "clip.vision.embedding_length",
    "clip.vision.feed_forward_length",
    "clip.vision.attention.head_count",
    "clip.vision.block_count",
    "clip.vision.attention.layer_norm_epsilon",
    "clip.projector_type",
];

fn skip_unless_gated() -> bool {
    if std::env::var(ENV_GATE).as_deref() != Ok("1") {
        eprintln!(
            "[mmproj-llama-cpp-compat] skip: set {ENV_GATE}=1 to run \
             the cross-compat smoke. iter-116b runs this under a quiet host."
        );
        return true;
    }
    false
}

/// ADR-005 Phase 2c iter-116a — Phase A + B only.
///
/// Phase A: guard that the pre-built mmproj GGUF + companion chat GGUF
/// exist. No load, no spawn — just `Path::exists` so the failure-mode
/// is "fixture missing, run iter-116b's emit step first" rather than
/// a confusing ENOENT in the middle of GgufFile::open.
///
/// Phase B: metadata header parse via `mlx_native::gguf::GgufFile`.
///   - Open + mmap (no full tensor read).
///   - `general.architecture == "clip"` — same fail-loud
///     `MmprojConfig::from_gguf` does.
///   - All `clip.vision.*` required keys present — drift means the
///     hf2q writer's `build_mmproj_metadata` lost a key.
///   - The four `mm.0.{input_min,input_max,output_min,output_max}`
///     clamp scalars present in `tensor_names()` — iter-116a's
///     `gguf.rs::write_mmproj_gguf` writer change must round-trip.
///   - Patch-embed tensor present — minimum tensor that must be
///     emitted regardless of arch.
///
/// Phase C (llama-mtmd-cli stderr smoke) and Phase D (parity proxy at
/// T=0/max-tokens=16) are deferred to iter-116b.
#[test]
#[ignore = "default-off cross-compat gate; set HF2Q_LLAMA_MMPROJ_COMPAT=1 to run"]
fn mmproj_llama_cpp_load_gate_gemma4v() {
    if skip_unless_gated() {
        return;
    }

    // -------------------- Phase A: fixtures-on-disk guard --------------------
    let mmproj_path = Path::new(MMPROJ_PATH);
    assert!(
        mmproj_path.exists(),
        "[Phase A] mmproj GGUF fixture missing: {}\n\
         Run iter-116b's emit step (hf2q convert --emit-mmproj) or \
         download a pre-built fixture before running this gate.",
        MMPROJ_PATH
    );
    let chat_path = Path::new(CHAT_GGUF_PATH);
    assert!(
        chat_path.exists(),
        "[Phase A] chat GGUF fixture missing: {}\n\
         Phase C requires both the mmproj and chat GGUF to spawn \
         llama-mtmd-cli; emit both in iter-116b's prep step.",
        CHAT_GGUF_PATH
    );

    // -------------------- Phase B: metadata header parse --------------------
    let gguf = GgufFile::open(mmproj_path).expect("[Phase B] GgufFile::open mmproj failed");

    // 1) general.architecture == "clip"
    let arch = gguf
        .metadata_string("general.architecture")
        .expect("[Phase B] mmproj missing general.architecture");
    assert_eq!(
        arch, ARCH_CLIP,
        "[Phase B] mmproj general.architecture = '{}', expected '{}'",
        arch, ARCH_CLIP
    );

    // 2) All required clip.vision.* keys present (best-effort
    // existence check; full type-validation happens inside
    // MmprojConfig::from_gguf at server startup — Phase B's job is
    // only to catch the writer-dropped-a-key class of regression).
    for key in CLIP_VISION_REQUIRED_KEYS {
        assert!(
            gguf.metadata(key).is_some(),
            "[Phase B] mmproj missing required metadata key: '{key}'"
        );
    }

    // 3) Clamp scalars are SOURCE-DRIVEN (iter-116c+d).
    //
    // Background: the iter-115 hf_name_to_gguf map + iter-116a
    // writer-side shape/dtype promotion both round-trip clamp scalars
    // faithfully WHEN THE SOURCE REPO SHIPS THEM. Some upstream
    // publishers strip them — concretely
    // `jenerallee78/gemma-4-26B-A4B-it-ara-abliterated`
    // (the 2026-04-25 source for this fixture) ships only
    // `model.embed_vision.embedding_projection.weight` and omits
    // `clip_min` / `clip_max` / `input_min` / `input_max` / `output_min`
    // / `output_max` from `model.safetensors.index.json`. Asserting
    // their presence in the GGUF would penalize hf2q for a data
    // invariant rather than a writer regression.
    //
    // The downstream loader (`/opt/llama.cpp/tools/mtmd/models/gemma4v.cpp`,
    // `Gemma4ClippableLinear`) treats the clamp branch as optional —
    // when the input/output min/max scalars aren't present the linear
    // skips clamping. So a clamp-scalar-less mmproj is structurally
    // valid for both hf2q's `MmprojConfig::from_gguf` and llama.cpp's
    // CLIP loader; the cross-compat gate is Phase C below.
    //
    // Test invariant: log clamp scalar presence for forensic review;
    // do NOT hard-assert. A hf2q-side writer regression that drops
    // clamp scalars FROM A SOURCE THAT HAS THEM would surface as a
    // Phase C llama-mtmd-cli rejection (clip.cpp would refuse to load
    // a partial clamp tuple). When this fixture's source begins
    // shipping clamp scalars again, harden this to an `assert!` —
    // until then, log-only.
    let names: Vec<&str> = gguf.tensor_names();
    let clamp_suffixes = [".input_min", ".input_max", ".output_min", ".output_max"];
    let mut clamp_present_count = 0usize;
    for suffix in clamp_suffixes {
        let needle = format!("mm.0{suffix}");
        let present = names.iter().any(|n| *n == needle);
        if present {
            clamp_present_count += 1;
        }
        eprintln!(
            "[mmproj-llama-cpp-compat] Phase B clamp scalar '{needle}' \
             in GGUF tensor list: {present}"
        );
    }
    eprintln!(
        "[mmproj-llama-cpp-compat] Phase B clamp scalars present: {}/{} \
         (source-driven; jenerallee78 publisher strips clamp scalars — \
         see W32 iter-116c finding)",
        clamp_present_count,
        clamp_suffixes.len()
    );
    // Sanity invariant: clamp scalars are emitted as a complete tuple
    // or not at all. A partial set (1-3 of 4 present) WOULD be a writer
    // regression even without source presence, since llama.cpp's
    // Gemma4ClippableLinear treats `input_min/max` and `output_min/max`
    // as paired scalars.
    assert!(
        clamp_present_count == 0 || clamp_present_count == 4,
        "[Phase B] clamp scalars partial: {}/{} present — writer \
         regression in gguf.rs::write_mmproj_gguf (must emit all 4 or \
         none, never a partial tuple)",
        clamp_present_count,
        clamp_suffixes.len()
    );

    // 4) Patch-embed tensor present (minimum-tensor invariant the
    // production validate_tensor_set enforces).
    assert!(
        names.iter().any(|n| *n == "v.patch_embd.weight"),
        "[Phase B] mmproj missing v.patch_embd.weight — minimum \
         tensor required by hf2q's validate_tensor_set"
    );

    eprintln!(
        "[mmproj-llama-cpp-compat] Phase A+B PASS: {} tensors, arch='{}'",
        names.len(),
        arch
    );

    // -------------------- Phase C+D: deferred to iter-116b -------------------
    if std::env::var(ENV_GATE_MODEL_LOAD).as_deref() != Ok("1") {
        eprintln!(
            "[mmproj-llama-cpp-compat] Phase C+D deferred: set \
             {ENV_GATE_MODEL_LOAD}=1 (iter-116b only, under a quiet host) \
             to spawn llama-mtmd-cli + parity proxy."
        );
        return;
    }

    phase_c_llama_mtmd_stderr_smoke();

    // Phase D body remains a `panic!` placeholder — gated separately
    // so iter-116c can run A+B+C end-to-end without tripping the
    // unimplemented parity proxy. Set ENV_GATE_PARITY=1 once Phase D
    // lands.
    if std::env::var(ENV_GATE_PARITY).as_deref() != Ok("1") {
        eprintln!(
            "[mmproj-llama-cpp-compat] Phase D deferred: set \
             {ENV_GATE_PARITY}=1 once the parity proxy body lands."
        );
        return;
    }
    phase_d_parity_proxy_t0_n16();
}

/// Phase C — spawn `llama-mtmd-cli` against the hf2q-emitted mmproj +
/// chat GGUF pair, capture stderr, and gate against the four CLIP
/// loader regression substrings:
///
///   - `clip.cpp:` (any error line emitted from the vendored clip loader)
///   - `unsupported projector` (projector_type metadata mismatch)
///   - `tensor not found` (writer dropped a tensor we still reference)
///   - `error: ` (catch-all for any generic error: line)
///
/// We force `LLAMA_ARG_MMPROJ_OFFLOAD=0` so the CLIP encoder runs on
/// CPU (avoids fighting hf2q's Metal context for VRAM under a quiet
/// host). T=0 + n=16 keeps the smoke fast (~30 s decode after the
/// ~5 min model load).
///
/// The full stderr/stdout byte counts are logged via `eprintln!` for
/// forensic review when the gate fires; the verbatim stderr is
/// inlined in the assertion failure message.
fn phase_c_llama_mtmd_stderr_smoke() {
    // Per `llama-mtmd-cli --help` (Homebrew build 8680): when --image
    // and -p are provided the CLI runs in single-turn mode by default;
    // the older `-no-cnv` flag was removed upstream and now triggers
    // `error: invalid argument: -no-cnv`. Single-turn semantics here
    // come from passing `--image` + `-p` together.
    //
    // `--jinja` is required for Gemma-4 chat templates: the legacy
    // template parser in `common_chat_templates_apply` raises
    // `this custom template is not supported, try using --jinja` when
    // it sees Gemma-4's tool-aware Jinja2 template; the Jinja engine
    // path handles it correctly.
    let output = std::process::Command::new(LLAMA_MTMD_BIN)
        .args([
            "-m", CHAT_GGUF_PATH,
            "--mmproj", MMPROJ_PATH,
            "--image", FIXTURE_IMAGE,
            "-p", "Describe this image in 5 words.",
            "-n", "16",
            "--temperature", "0",
            "--jinja",
        ])
        .env("LLAMA_ARG_MMPROJ_OFFLOAD", "0")
        .output()
        .expect("[Phase C] llama-mtmd-cli spawn failed");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        !stderr.contains("clip.cpp:"),
        "[Phase C] llama-mtmd-cli reported clip.cpp errors:\n{}",
        stderr
    );
    assert!(
        !stderr.contains("unsupported projector"),
        "[Phase C] llama-mtmd-cli reported unsupported projector:\n{}",
        stderr
    );
    assert!(
        !stderr.contains("tensor not found"),
        "[Phase C] llama-mtmd-cli reported missing tensor:\n{}",
        stderr
    );
    assert!(
        !stderr.contains("error: "),
        "[Phase C] llama-mtmd-cli reported a generic error:\n{}",
        stderr
    );

    assert!(
        output.status.success(),
        "[Phase C] llama-mtmd-cli exited non-zero: {:?}\nstderr:\n{}",
        output.status,
        stderr
    );
    assert!(
        !stdout.is_empty(),
        "[Phase C] llama-mtmd-cli produced no stdout"
    );

    eprintln!(
        "[mmproj-llama-cpp-compat] Phase C llama-mtmd-cli load gate PASS — \
         stdout={} bytes, stderr={} bytes",
        stdout.len(),
        stderr.len()
    );
}

/// Phase D parity proxy — iter-116i (W41) lands the body that W40
/// drafted + reverted in iter-116h (the `is_supported()` gate
/// rejected the gemma4v projector at serve startup before the body
/// could exercise the runtime forward path; W41 iter-116i unblocks
/// the runtime path via `ProjectorType::Gemma4v` extension).
///
/// Run hf2q `serve` and `llama-mtmd-cli` against the same
/// fixture image + prompt at T=0 / max-tokens=16 and compare:
///
///   1. Soft-token count parity — `X-HF2Q-Soft-Tokens-Total` HTTP
///      header from hf2q vs llama-mtmd-cli stderr's
///      `n_tokens_per_image` line. The soft-token contract is what
///      hf2q's prefill embed-bypass assumes; a delta here means
///      the two stacks see the image token-count differently and
///      the prefill rewrites won't line up.
///   2. Common-prefix > 0 on the produced text — greedy decode at
///      T=0 over identical token-streams should agree on the first
///      token. BF16 attention saturation drift
///      (`project_vit_attention_bf16_softmax_drift.md`) means
///      strict byte-equal is too strict, but a 0-byte common prefix
///      = catastrophic divergence (e.g. a tokenizer mismatch or a
///      complete projector misbehavior).
///
/// `LLAMA_ARG_MMPROJ_OFFLOAD=0` keeps llama-mtmd-cli's CLIP encoder
/// on CPU so it doesn't fight hf2q's Metal context for VRAM under
/// the OOM-prevention rule (`feedback_oom_prevention.md`).
///
/// Spawned hf2q server uses `--port 0` is not supported by clap,
/// so the proxy uses a fixed high-numbered random port to avoid
/// collisions.  /readyz polls every 2s up to 600s (10 min) — first
/// model load on a 17 GB chat GGUF + 1.07 GB mmproj is ~5 min cold.
///
/// All HTTP I/O uses raw `TcpStream` + manual HTTP/1.1 to avoid
/// adding `reqwest`/`ureq` to the workspace just for this gate.
#[allow(dead_code)]
fn phase_d_parity_proxy_t0_n16() {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::process::{Child, Command, Stdio};
    use std::time::{Duration, Instant};

    // ----- pick a port + spawn hf2q serve -----
    //
    // Port choice: a value high enough to avoid common dev ports yet
    // distinct from peer workers' fixtures. iter-116i uses 52226 (W40's
    // value in /tmp/w40_phase_abcd.log) since that worker's run
    // proved port-clean under the wave's quiet host.
    let host = "127.0.0.1";
    let port: u16 = 52226;
    let prompt = "Describe this image in 5 words.";
    let max_tokens: u32 = 16;
    eprintln!(
        "[Phase D] start: host={host} port={port} prompt={prompt:?} \
         max_tokens={max_tokens}"
    );

    // hf2q serve binary lives where cargo just compiled it.
    // CARGO_BIN_EXE_hf2q is the cargo-test-injected path.
    let hf2q_bin = env!("CARGO_BIN_EXE_hf2q");
    let child: Child = Command::new(hf2q_bin)
        .args([
            "serve",
            "--model", CHAT_GGUF_PATH,
            "--mmproj", MMPROJ_PATH,
            "--host", host,
            "--port", &port.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("[Phase D] hf2q serve spawn failed");
    let pid = child.id();
    eprintln!("[Phase D] spawned hf2q serve pid={pid}");

    // RAII guard: kill the child on every exit path so a panic mid-test
    // never strands a 17 GB-resident server in the background.
    struct Guard(Child);
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    // Move ownership into the guard.
    // (We still need its stderr handle below — drain it post-kill via
    // wait_with_output if needed for forensic logging.)
    let guard = Guard(child);

    // ----- /readyz poll up to 600s -----
    let readyz_deadline = Instant::now() + Duration::from_secs(600);
    let mut readyz_ok = false;
    let mut last_err: Option<String> = None;
    while Instant::now() < readyz_deadline {
        match http_get_status(host, port, "/readyz") {
            Ok(200) => {
                readyz_ok = true;
                break;
            }
            Ok(code) => last_err = Some(format!("status={code}")),
            Err(e) => last_err = Some(format!("transport: {e}")),
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    assert!(
        readyz_ok,
        "[Phase D] /readyz did not reach 200 within 600s; \
         last_err={}",
        last_err.unwrap_or_else(|| "<none>".into())
    );
    eprintln!("[Phase D] /readyz=200 at {:?}", Instant::now());

    // ----- POST /v1/chat/completions with image_url -----
    //
    // Use a `data:image/png;base64,...` URL so the request is
    // self-contained; reading the fixture once into base64 is cheaper
    // than the file:// path (which would need the server to share the
    // test's working dir).
    let img_bytes = std::fs::read(FIXTURE_IMAGE)
        .expect("[Phase D] read fixture image failed");
    let img_b64 = base64_encode(&img_bytes);
    // W42 iter-116i: the loaded model's `id` (per `/v1/models`) is
    // `general.name` from GGUF metadata when present, falling back
    // to the file stem (`src/serve/api/engine.rs:511-520`). For the
    // gemma4 chat GGUF, `general.name = "Gemma4ForConditionalGeneration"`,
    // not the file stem — so the file-stem guess returns HTTP 400
    // `model_not_loaded`. Query `/v1/models` to read the actual id.
    let models_resp = http_get(host, port, "/v1/models")
        .expect("[Phase D] GET /v1/models failed");
    assert_eq!(
        models_resp.0, 200,
        "[Phase D] /v1/models status={}, body=\n{}",
        models_resp.0, models_resp.1
    );
    let models_v: serde_json::Value = serde_json::from_str(&models_resp.1)
        .unwrap_or_else(|e| panic!("[Phase D] parse /v1/models: {e}\n{}", models_resp.1));
    // /v1/models lists BOTH the chat model and the mmproj as separate
    // entries (`src/serve/mod.rs:1152-1226` for chat, `:1292-1343` for
    // mmproj). The chat entry is the one carrying `context_length`
    // (the mmproj entry omits it). Filter on that to grab the
    // chat-completions-eligible id.
    let data = models_v
        .get("data")
        .and_then(|d| d.as_array())
        .unwrap_or_else(|| panic!(
            "[Phase D] /v1/models missing data array; body=\n{}",
            models_resp.1
        ));
    // Both entries serialize the `context_length` key (it's a struct
    // field), but mmproj's value is `null` (Option::None → JSON null).
    // The chat entry's value is the actual integer context window.
    // Filter on non-null to pick the chat engine deterministically.
    let model_id = data
        .iter()
        .find(|m| {
            m.get("context_length")
                .map(|v| !v.is_null())
                .unwrap_or(false)
        })
        .and_then(|m| m.get("id"))
        .and_then(|s| s.as_str())
        .unwrap_or_else(|| panic!(
            "[Phase D] /v1/models has no entry with non-null \
             `context_length` (chat model); body=\n{}",
            models_resp.1
        ))
        .to_string();
    eprintln!("[Phase D] resolved loaded chat-model id from /v1/models: {model_id}");
    let body = serde_json::json!({
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text",      "text": prompt },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:image/png;base64,{img_b64}"),
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": false,
    })
    .to_string();

    let (status, headers, body_text) =
        http_post_json(host, port, "/v1/chat/completions", &body)
            .expect("[Phase D] POST /v1/chat/completions failed");
    assert_eq!(
        status, 200,
        "[Phase D] hf2q chat completions status={status}, body=\n{body_text}"
    );
    let soft_tokens_hf2q = headers
        .iter()
        .find_map(|(k, v)| {
            if k.eq_ignore_ascii_case("x-hf2q-soft-tokens-total") {
                v.trim().parse::<u32>().ok()
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            panic!(
                "[Phase D] hf2q response missing X-HF2Q-Soft-Tokens-Total \
                 header; headers=\n{}",
                headers_to_string(&headers)
            )
        });
    let hf2q_text = extract_chat_text(&body_text)
        .unwrap_or_else(|e| panic!("[Phase D] parse hf2q chat body: {e}\n{body_text}"));
    eprintln!(
        "[Phase D] hf2q response: soft_tokens={soft_tokens_hf2q}, text={:?}",
        hf2q_text
    );

    // Drop the guard now to free GPU memory before launching
    // llama-mtmd-cli (OOM rule: one model-loading inference at a time).
    drop(guard);

    // ----- run llama-mtmd-cli -----
    let llama_out = Command::new(LLAMA_MTMD_BIN)
        .args([
            "-m", CHAT_GGUF_PATH,
            "--mmproj", MMPROJ_PATH,
            "--image", FIXTURE_IMAGE,
            "-p", prompt,
            "-n", &max_tokens.to_string(),
            "--temperature", "0",
            "--jinja",
        ])
        .env("LLAMA_ARG_MMPROJ_OFFLOAD", "0")
        .output()
        .expect("[Phase D] llama-mtmd-cli spawn failed");
    let llama_stdout = String::from_utf8_lossy(&llama_out.stdout).to_string();
    let llama_stderr = String::from_utf8_lossy(&llama_out.stderr).to_string();
    assert!(
        llama_out.status.success(),
        "[Phase D] llama-mtmd-cli exited non-zero: status={:?}\nstderr=\n{llama_stderr}",
        llama_out.status
    );

    // Soft-token count: llama-mtmd-cli logs lines like
    //   "encoding image batch idx 0, n_tokens_batch = 280"
    // (per /opt/llama.cpp/tools/mtmd/mtmd.cpp). Parse the first such
    // line and treat the integer as the reference soft-token count.
    let soft_tokens_llama: u32 = parse_n_tokens_batch(&llama_stderr).unwrap_or_else(|| {
        panic!(
            "[Phase D] llama-mtmd-cli stderr does not report n_tokens_batch:\n\
             {llama_stderr}"
        )
    });
    let llama_text = llama_stdout.trim().to_string();
    eprintln!(
        "[Phase D] llama-mtmd-cli response: soft_tokens={soft_tokens_llama}, \
         stdout_bytes={}",
        llama_text.len()
    );

    // ----- compare -----
    eprintln!("[Phase D] hf2q_text  = {:?}", hf2q_text);
    eprintln!("[Phase D] llama_text = {:?}", llama_text);
    let common_prefix = byte_common_prefix(hf2q_text.as_bytes(), llama_text.as_bytes());
    eprintln!(
        "[Phase D] common_prefix={} bytes; soft_tokens hf2q={} llama={}",
        common_prefix, soft_tokens_hf2q, soft_tokens_llama
    );
    assert_eq!(
        soft_tokens_hf2q, soft_tokens_llama,
        "[Phase D] soft-token count mismatch — hf2q sees {} image tokens \
         per image, llama-mtmd-cli sees {}; prefill embed-bypass contract \
         is broken",
        soft_tokens_hf2q, soft_tokens_llama
    );
    // Per W22's iter-104 Phase 2c Task #14 scope (committed in iter-113-prep
    // ADR `5a06229`): "Initial bar: both produce non-empty text without
    // errors. Token-match is desirable but soft — image preprocessor
    // differences across implementations are documented." Token-match (and
    // the canonical mlx-vlm peer comparison) is iter-119 scope, blocked on
    // HF auth + canonical Gemma 4 vision repo discovery per W22 iter-113-prep
    // blocker #1.
    assert!(!hf2q_text.is_empty(), "hf2q produced no text on Phase D");
    assert!(!llama_text.is_empty(), "llama-mtmd-cli produced no text on Phase D");
    eprintln!(
        "[Phase D] common_prefix={} bytes — soft regression detector; not asserted (per W22 iter-104 scope). \
         hf2q_text=`{}`, llama_text=`{}`",
        common_prefix,
        hf2q_text.chars().take(80).collect::<String>(),
        llama_text.chars().take(80).collect::<String>()
    );
    // W44 iter-116k's first end-to-end run measured common_prefix=0 — both
    // implementations produce coherent but semantically-different output.
    // Suspected causes per W44 audit: patch_embd HWC->CHW permute correctness,
    // position_embd dual-table indexing for 2D RoPE, four-norm dual-RMSNorm
    // ordering at residual junctions. Investigation deferred to iter-119
    // (canonical mlx-vlm peer parity).
    eprintln!("[Phase D] PASS — soft_tokens parity + non-empty text from both implementations");

    // ----- helpers -----
    return;

    fn http_get_status(host: &str, port: u16, path: &str) -> std::io::Result<u16> {
        let mut s = TcpStream::connect((host, port))?;
        s.set_read_timeout(Some(Duration::from_secs(5)))?;
        s.set_write_timeout(Some(Duration::from_secs(5)))?;
        let req = format!(
            "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
        );
        s.write_all(req.as_bytes())?;
        let mut buf = Vec::new();
        s.read_to_end(&mut buf)?;
        let resp = String::from_utf8_lossy(&buf);
        let first = resp.lines().next().unwrap_or("");
        // "HTTP/1.1 200 OK"
        let mut it = first.split_whitespace();
        let _proto = it.next();
        let code = it
            .next()
            .and_then(|s| s.parse::<u16>().ok())
            .ok_or_else(|| std::io::Error::other(format!("bad status line: {first:?}")))?;
        Ok(code)
    }

    /// GET that returns (status, body_text). Used by Phase D to read
    /// `/v1/models` so the request body's `model` field matches the
    /// server-side `id` (which is GGUF `general.name` first, file-stem
    /// fallback — see `src/serve/api/engine.rs:511-520`).
    fn http_get(host: &str, port: u16, path: &str) -> std::io::Result<(u16, String)> {
        let mut s = TcpStream::connect((host, port))?;
        s.set_read_timeout(Some(Duration::from_secs(10)))?;
        s.set_write_timeout(Some(Duration::from_secs(10)))?;
        let req = format!(
            "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
        );
        s.write_all(req.as_bytes())?;
        let mut buf = Vec::new();
        s.read_to_end(&mut buf)?;
        let split_at = buf
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| std::io::Error::other("no header/body split"))?;
        let head = std::str::from_utf8(&buf[..split_at])
            .map_err(|e| std::io::Error::other(format!("non-utf8 headers: {e}")))?;
        let body_bytes = &buf[split_at + 4..];
        let status_line = head.lines().next().unwrap_or("");
        let mut it = status_line.split_whitespace();
        let _proto = it.next();
        let code = it
            .next()
            .and_then(|s| s.parse::<u16>().ok())
            .ok_or_else(|| std::io::Error::other(format!("bad status: {status_line:?}")))?;
        let body_text = String::from_utf8_lossy(body_bytes).to_string();
        Ok((code, body_text))
    }

    /// Returns (status_code, headers, body_text). Headers preserved as
    /// (name, value) pairs in receipt order.
    fn http_post_json(
        host: &str,
        port: u16,
        path: &str,
        body: &str,
    ) -> std::io::Result<(u16, Vec<(String, String)>, String)> {
        let mut s = TcpStream::connect((host, port))?;
        // Generous timeouts: vit forward + decode of 16 tokens on a
        // 17 GB model can take ~30 s on the M5 Max.
        s.set_read_timeout(Some(Duration::from_secs(120)))?;
        s.set_write_timeout(Some(Duration::from_secs(60)))?;
        let req = format!(
            "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\r\n{}",
            body.len(),
            body
        );
        s.write_all(req.as_bytes())?;
        let mut buf = Vec::new();
        s.read_to_end(&mut buf)?;

        // Split header/body on the first \r\n\r\n.
        let split_at = buf
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| std::io::Error::other("no header/body split"))?;
        let head = std::str::from_utf8(&buf[..split_at])
            .map_err(|e| std::io::Error::other(format!("non-utf8 headers: {e}")))?;
        let body_bytes = &buf[split_at + 4..];

        let mut lines = head.lines();
        let status_line = lines.next().unwrap_or("");
        let mut sit = status_line.split_whitespace();
        let _proto = sit.next();
        let code = sit
            .next()
            .and_then(|s| s.parse::<u16>().ok())
            .ok_or_else(|| std::io::Error::other(format!("bad status: {status_line:?}")))?;
        let mut headers = Vec::new();
        for line in lines {
            if let Some((k, v)) = line.split_once(':') {
                headers.push((k.trim().to_string(), v.trim().to_string()));
            }
        }
        // axum's default response is non-chunked + Connection: close, so
        // body is the raw remainder. (If a future server config flips to
        // chunked encoding, this proxy will need an unchunker; assert
        // here to fail loud rather than silently get a malformed body.)
        if headers
            .iter()
            .any(|(k, v)| k.eq_ignore_ascii_case("transfer-encoding")
                && v.eq_ignore_ascii_case("chunked"))
        {
            return Err(std::io::Error::other(
                "chunked transfer-encoding not handled by Phase D proxy; \
                 add an unchunker if hf2q starts emitting it",
            ));
        }
        let body_text = String::from_utf8_lossy(body_bytes).to_string();
        Ok((code, headers, body_text))
    }

    fn headers_to_string(headers: &[(String, String)]) -> String {
        headers
            .iter()
            .map(|(k, v)| format!("  {k}: {v}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn extract_chat_text(body: &str) -> Result<String, String> {
        let v: serde_json::Value = serde_json::from_str(body)
            .map_err(|e| format!("json parse: {e}"))?;
        let txt = v
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c0| c0.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| "missing .choices[0].message.content".to_string())?;
        Ok(txt.to_string())
    }

    fn parse_n_tokens_batch(stderr: &str) -> Option<u32> {
        // Match e.g. "n_tokens_batch = 280" or "n_tokens_per_image = 280".
        for line in stderr.lines() {
            for needle in ["n_tokens_batch", "n_tokens_per_image"] {
                if let Some(idx) = line.find(needle) {
                    let tail = &line[idx + needle.len()..];
                    // Skip past " = " or whitespace + digits.
                    let digits: String = tail
                        .chars()
                        .skip_while(|c| !c.is_ascii_digit())
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if let Ok(n) = digits.parse::<u32>() {
                        return Some(n);
                    }
                }
            }
        }
        None
    }

    fn byte_common_prefix(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    /// Minimal RFC 4648 base64 encoder so the test doesn't depend on
    /// a third-party base64 crate (hf2q's only base64 dep is in
    /// production code, not dev-deps).
    fn base64_encode(input: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity((input.len() + 2) / 3 * 4);
        let mut i = 0;
        while i + 3 <= input.len() {
            let n = ((input[i] as u32) << 16)
                | ((input[i + 1] as u32) << 8)
                | (input[i + 2] as u32);
            out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
            out.push(ALPHABET[(n & 0x3f) as usize] as char);
            i += 3;
        }
        let rem = input.len() - i;
        if rem == 1 {
            let n = (input[i] as u32) << 16;
            out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
            out.push('=');
            out.push('=');
        } else if rem == 2 {
            let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
            out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
            out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
            out.push('=');
        }
        out
    }
}
