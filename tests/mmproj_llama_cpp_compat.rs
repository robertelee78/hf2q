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

    // 3) Clamp scalars round-tripped (iter-115 hf_name_to_gguf map +
    // iter-116a writer-side shape/dtype promotion). Their absence
    // means the writer dropped them somewhere in the convert pipeline.
    let names: Vec<&str> = gguf.tensor_names();
    for suffix in [".input_min", ".input_max", ".output_min", ".output_max"] {
        let needle = format!("mm.0{suffix}");
        assert!(
            names.iter().any(|n| *n == needle),
            "[Phase B] mmproj missing clamp scalar '{needle}' — hf2q \
             writer regression in gguf.rs::write_mmproj_gguf"
        );
    }

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
    let output = std::process::Command::new(LLAMA_MTMD_BIN)
        .args([
            "-m", CHAT_GGUF_PATH,
            "--mmproj", MMPROJ_PATH,
            "--image", FIXTURE_IMAGE,
            "-p", "Describe this image in 5 words.",
            "-n", "16",
            "--temperature", "0",
            "-no-cnv",
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

/// Phase D placeholder — iter-116b lands the parity proxy: same
/// fixture image + prompt at T=0/max-tokens=16 through both hf2q's
/// `/v1/chat/completions` and `llama-mtmd-cli`, then compares output
/// token sequences. Soft equality (greedy decode shouldn't drift but
/// the BF16 attention saturation in hf2q's vit_gpu can perturb
/// logits — see `project_vit_attention_bf16_softmax_drift.md`); a
/// strict byte-equal proxy is too strict.
///
/// Iter-116b's proxy: assert N_image_tokens matches between the two
/// (hf2q's `X-HF2Q-Soft-Tokens-Total` header vs llama-mtmd-cli's
/// stderr image-token report). Output text need not match — the
/// soft-token contract is what we're gating on.
#[allow(dead_code)]
fn phase_d_parity_proxy_t0_n16() {
    panic!(
        "[Phase D] parity proxy not implemented in iter-116a. \
         Lands in iter-116b."
    );
}
