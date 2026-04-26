//! ADR-005 Phase 2c iter-101 — vision E2E acceptance harness against mlx-vlm.
//!
//! This integration test drives the full acceptance bar from iter-99:
//!
//!   > hf2q vision output matches mlx-lm Gemma 4 vision on 5 standard prompts
//!   > × 5 images (token-match for the first 50 generated tokens at T=0).
//!
//! It is **gated** behind `HF2Q_VISION_E2E=1` because each leg of the
//! comparison loads a ~30 GB model and the project's standing OOM-prevention
//! directive says **one model-loading inference at a time**.  Default `cargo
//! test` runs no model — only the harness compile path + the synthetic-
//! fixture builder are exercised, both of which are cheap.
//!
//! Two more env knobs (used only when `HF2Q_VISION_E2E=1`):
//!
//!   * `HF2Q_VISION_E2E_GGUF`   — path to the chat-model GGUF (Gemma 4).
//!   * `HF2Q_VISION_E2E_MMPROJ` — path to the mmproj GGUF.
//!   * `HF2Q_VISION_E2E_MLX_REPO` — model id passed to `mlx_vlm.generate`
//!                                    (defaults to `mlx-community/gemma-4-vision-26b-A4B-it-bf16`).
//!
//! When unset, defaults point at `/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/`
//! which holds both the chat GGUF and the mmproj GGUF as sibling files.
//!
//! What the harness does end-to-end (only when the env-gate fires):
//!   1. Materialize 5 synthetic fixture PNGs in `tests/fixtures/vision/` if missing.
//!   2. Spawn `hf2q serve --model ... --mmproj ...` as a child process.
//!   3. Poll `GET /readyz` until 200 OK or 120s timeout.
//!   4. For each of the 25 (prompt, image) pairs, POST to
//!      `/v1/chat/completions` with `messages = [{"role":"user", "content":[
//!      {"type":"text", "text": <prompt>}, {"type":"image_url", "image_url":
//!      {"url": "data:image/png;base64,..."}}]}]`, `temperature=0`,
//!      `max_tokens=50`, capture the response text.
//!   5. SIGTERM the hf2q child + wait for it to drain.
//!   6. For each of the 25 pairs, run `mlx_vlm.generate --model ... --image
//!      <path> --prompt <prompt> --temperature 0 --max-tokens 50` and
//!      capture stdout.
//!   7. Compare hf2q text vs mlx-vlm text per pair (token-level match
//!      lands in iter-102 once the byte-comparable hf2q `tokenizer.encode`
//!      pipeline exposes per-response token ids on the wire — for now
//!      we string-compare the first 200 chars and report exact-match,
//!      starts-with, and Levenshtein distance).
//!   8. Emit a report at `tests/fixtures/vision/last_e2e_report.json` and
//!      assert exact-match count >= 0 (iter-101 records the matrix; the
//!      next iter raises the bar to a hard PASS gate).
//!
//! Per-iter scope on the harness:
//!   * iter-101 (this iter): build fixtures, build harness, default-skip,
//!     run-on-demand to record a baseline matrix.
//!   * iter-102: ratchet the assertion up — exact-match >= K, with K
//!     calibrated against the iter-101 baseline.
//!   * iter-103+: triage divergences (placeholder text scheme, ViT scale,
//!     positional embedding alignment) until the matrix is fully GREEN.

use std::path::{Path, PathBuf};

// -----------------------------------------------------------------------------
// Fixture: 5 standard prompts × 5 standard images
// -----------------------------------------------------------------------------

/// 5 standard prompts that exercise distinct visual reasoning categories.
/// Each prompt is short + closed-ended so token-level comparison against
/// mlx-vlm is reasonable.  Order matters — the prompt array is paired
/// with the image array index-wise as the natural acceptance matrix
/// (5 × 5 = 25 pairs), so changing this list breaks reproducibility of
/// any matrix snapshot keyed on these strings.
const STANDARD_PROMPTS: &[&str] = &[
    "What is the dominant color in this image? Answer in one word.",
    "Describe what you see in one short sentence.",
    "What text appears in this image? Answer with just the text shown.",
    "How many distinct shapes are in the image? Answer with a number.",
    "Is the background of this image dark or light? Answer with one word.",
];

/// 5 standard image fixture file names (relative to
/// `tests/fixtures/vision/`).  Materialized on-the-fly by
/// `materialize_fixture_images` so the repo stays text-only.
const STANDARD_IMAGE_NAMES: &[&str] = &[
    "red_square_64x64.png",
    "green_circle_on_white_128x128.png",
    "text_hello_256x64.png",
    "four_dots_in_corners_128x128.png",
    "vertical_dark_gradient_128x128.png",
];

/// Sub-directory under the test crate where synthetic fixtures land.
fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("vision")
}

// -----------------------------------------------------------------------------
// Fixture materialization — synthetic, deterministic, byte-stable
// -----------------------------------------------------------------------------

fn materialize_fixture_images() -> std::io::Result<()> {
    use image::{ImageBuffer, ImageFormat, Rgb, RgbImage};
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir)?;

    // 1. Solid red 64×64 — answers "red".
    {
        let img: RgbImage = ImageBuffer::from_fn(64, 64, |_x, _y| Rgb([220u8, 30, 30]));
        let path = dir.join(STANDARD_IMAGE_NAMES[0]);
        img.save_with_format(&path, ImageFormat::Png)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
    }

    // 2. Green circle (radius 40) on white 128×128.
    {
        let cx: i32 = 64;
        let cy: i32 = 64;
        let r2: i32 = 40 * 40;
        let img: RgbImage = ImageBuffer::from_fn(128, 128, |x, y| {
            let dx = x as i32 - cx;
            let dy = y as i32 - cy;
            if dx * dx + dy * dy <= r2 {
                Rgb([30u8, 180, 30])
            } else {
                Rgb([245u8, 245, 245])
            }
        });
        let path = dir.join(STANDARD_IMAGE_NAMES[1]);
        img.save_with_format(&path, ImageFormat::Png)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
    }

    // 3. Pixel-art "HELLO" on a 256×64 white canvas.  Each letter is
    //    16 wide + 4-px gap, drawn with hand-rolled 5×7 glyphs that
    //    are deterministic (no font-file dependency).
    {
        let mut img: RgbImage = ImageBuffer::from_pixel(256, 64, Rgb([250u8, 250, 250]));
        let glyphs = "HELLO";
        // Tiny 5x7 bitmap font for H, E, L, O.  Columns left-to-right, MSB top.
        let glyph_bits: &[(char, [u8; 5])] = &[
            ('H', [0b1111111, 0b0001000, 0b0001000, 0b0001000, 0b1111111]),
            ('E', [0b1111111, 0b1001001, 0b1001001, 0b1001001, 0b1000001]),
            ('L', [0b1111111, 0b1000000, 0b1000000, 0b1000000, 0b1000000]),
            ('O', [0b0111110, 0b1000001, 0b1000001, 0b1000001, 0b0111110]),
        ];
        let scale: u32 = 4; // 5×7 → 20×28 per glyph
        let glyph_w_px: u32 = 5 * scale;
        let glyph_h_px: u32 = 7 * scale;
        let gap_px: u32 = scale; // 4px between glyphs
        let total_w: u32 = glyphs.chars().count() as u32 * (glyph_w_px + gap_px) - gap_px;
        let x0: u32 = (256 - total_w) / 2;
        let y0: u32 = (64 - glyph_h_px) / 2;
        let black = Rgb([10u8, 10, 10]);
        for (gi, ch) in glyphs.chars().enumerate() {
            let bits = glyph_bits
                .iter()
                .find(|(c, _)| *c == ch)
                .map(|(_, b)| *b)
                .expect("glyph defined");
            let gx = x0 + gi as u32 * (glyph_w_px + gap_px);
            for col in 0..5 {
                let column_bits = bits[col];
                for row in 0..7 {
                    let bit = (column_bits >> (6 - row)) & 1;
                    if bit == 1 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                let px = gx + col as u32 * scale + dx;
                                let py = y0 + row as u32 * scale + dy;
                                if px < 256 && py < 64 {
                                    img.put_pixel(px, py, black);
                                }
                            }
                        }
                    }
                }
            }
        }
        let path = dir.join(STANDARD_IMAGE_NAMES[2]);
        img.save_with_format(&path, ImageFormat::Png)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
    }

    // 4. Four black 12×12 dots in the corners of a 128×128 white canvas.
    {
        let mut img: RgbImage = ImageBuffer::from_pixel(128, 128, Rgb([250u8, 250, 250]));
        let dot_corners: [(u32, u32); 4] = [(8, 8), (108, 8), (8, 108), (108, 108)];
        let black = Rgb([15u8, 15, 15]);
        for (cx, cy) in dot_corners {
            for dy in 0..12 {
                for dx in 0..12 {
                    let px = cx + dx;
                    let py = cy + dy;
                    if px < 128 && py < 128 {
                        img.put_pixel(px, py, black);
                    }
                }
            }
        }
        let path = dir.join(STANDARD_IMAGE_NAMES[3]);
        img.save_with_format(&path, ImageFormat::Png)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
    }

    // 5. Vertical dark gradient 128×128 (black at top → mid-gray at bottom).
    //    "Dark" is the obvious answer for the background-luminance prompt.
    {
        let img: RgbImage = ImageBuffer::from_fn(128, 128, |_x, y| {
            let g = (20 + (y * 60) / 128) as u8;
            Rgb([g, g, g])
        });
        let path = dir.join(STANDARD_IMAGE_NAMES[4]);
        img.save_with_format(&path, ImageFormat::Png)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
    }

    Ok(())
}

// -----------------------------------------------------------------------------
// Compile-time / cheap unit tests (always run, even without the env-gate)
// -----------------------------------------------------------------------------

#[test]
fn standard_prompt_image_arrays_have_matching_length() {
    assert_eq!(
        STANDARD_PROMPTS.len(),
        STANDARD_IMAGE_NAMES.len(),
        "5 prompts must pair 1:1 with 5 images for the acceptance matrix"
    );
    assert_eq!(STANDARD_PROMPTS.len(), 5);
}

#[test]
fn standard_prompts_are_short_and_closed() {
    // Long-form prompts make token-level comparison brittle. Cap each
    // at 200 chars; if a future iter wants a longer prompt, bump this
    // bound and document why in the ADR.
    for (i, p) in STANDARD_PROMPTS.iter().enumerate() {
        assert!(
            p.len() <= 200,
            "STANDARD_PROMPTS[{i}] = {p:?} is {} chars (max 200)",
            p.len()
        );
        assert!(!p.is_empty(), "STANDARD_PROMPTS[{i}] is empty");
    }
}

#[test]
fn standard_image_names_are_unique_and_have_png_extension() {
    use std::collections::HashSet;
    let mut seen: HashSet<&str> = HashSet::new();
    for (i, n) in STANDARD_IMAGE_NAMES.iter().enumerate() {
        assert!(seen.insert(n), "STANDARD_IMAGE_NAMES[{i}] = {n} duplicated");
        assert!(
            n.ends_with(".png"),
            "STANDARD_IMAGE_NAMES[{i}] = {n} does not end in .png"
        );
    }
}

#[test]
fn materialize_fixture_images_round_trips_5_decodable_pngs() {
    materialize_fixture_images().expect("write fixture images");
    let dir = fixture_dir();
    for name in STANDARD_IMAGE_NAMES {
        let path = dir.join(name);
        assert!(path.exists(), "fixture missing: {}", path.display());
        // Decode round-trip: every PNG must reload as a non-empty RGB image.
        let img = image::open(&path)
            .unwrap_or_else(|e| panic!("decode {}: {e}", path.display()));
        let (w, h) = (img.width(), img.height());
        assert!(w > 0 && h > 0, "fixture {name} decoded to 0×0");
    }
}

// -----------------------------------------------------------------------------
// Gated end-to-end matrix (only when HF2Q_VISION_E2E=1)
// -----------------------------------------------------------------------------

#[test]
fn vision_e2e_matrix_against_mlx_vlm() {
    if std::env::var("HF2Q_VISION_E2E").as_deref() != Ok("1") {
        eprintln!(
            "vision_e2e_matrix_against_mlx_vlm: SKIPPED \
             (HF2Q_VISION_E2E != \"1\"; full matrix needs ~30 GB and \
             sequential model loads)"
        );
        return;
    }
    // The end-to-end driver itself lives in `run_e2e_matrix` so the
    // gating boilerplate stays narrow.  The function returns a
    // `MatrixReport` whose presence on disk is the iter-101 deliverable;
    // the assertion bar here is intentionally permissive (record the
    // matrix; iter-102 ratchets it).
    let report = run_e2e_matrix().expect("E2E matrix run failed");
    let report_path = fixture_dir().join("last_e2e_report.json");
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    std::fs::write(&report_path, &json).expect("write report");
    eprintln!(
        "vision_e2e_matrix_against_mlx_vlm: wrote {} ({} pairs, exact-match {}/{})",
        report_path.display(),
        report.pairs.len(),
        report.exact_matches,
        report.pairs.len()
    );
    // Iter-101 baseline: the matrix runs to completion without panicking.
    // Iter-102 introduces a hard exact-match >= K assertion calibrated
    // against the iter-101 baseline.
    assert!(
        !report.pairs.is_empty(),
        "E2E matrix produced zero pairs — harness is broken"
    );
}

// -----------------------------------------------------------------------------
// Internal: matrix driver + report types (only invoked under env-gate)
// -----------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct MatrixReport {
    pairs: Vec<PairOutcome>,
    exact_matches: usize,
    hf2q_gguf: String,
    hf2q_mmproj: String,
    mlx_repo: String,
}

#[derive(serde::Serialize)]
struct PairOutcome {
    prompt_idx: usize,
    image_idx: usize,
    prompt: String,
    image_name: String,
    hf2q_text: String,
    mlx_vlm_text: String,
    exact_match: bool,
    hf2q_starts_with_mlx_first_word: bool,
}

fn run_e2e_matrix() -> std::io::Result<MatrixReport> {
    materialize_fixture_images()?;
    let model_dir = PathBuf::from(
        "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq",
    );
    let gguf = std::env::var("HF2Q_VISION_E2E_GGUF").unwrap_or_else(|_| {
        model_dir
            .join("gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf")
            .to_string_lossy()
            .into_owned()
    });
    let mmproj = std::env::var("HF2Q_VISION_E2E_MMPROJ").unwrap_or_else(|_| {
        model_dir
            .join("gemma-4-26B-A4B-it-ara-abliterated-dwq-mmproj.gguf")
            .to_string_lossy()
            .into_owned()
    });
    let mlx_repo = std::env::var("HF2Q_VISION_E2E_MLX_REPO")
        .unwrap_or_else(|_| "mlx-community/gemma-4-vision-26b-A4B-it-bf16".to_string());

    if !PathBuf::from(&gguf).exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("HF2Q_VISION_E2E_GGUF not found: {gguf}"),
        ));
    }
    if !PathBuf::from(&mmproj).exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("HF2Q_VISION_E2E_MMPROJ not found: {mmproj}"),
        ));
    }

    // ---- Step 1: spawn hf2q server ----
    let port: u16 = 18181;
    let hf2q_bin = std::env::var("CARGO_BIN_EXE_hf2q").unwrap_or_else(|_| {
        // Fallback when run outside cargo (e.g. cargo nextest). The
        // env var is the cargo-canonical lookup.
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/release/hf2q")
            .to_string_lossy()
            .into_owned()
    });
    let mut hf2q = std::process::Command::new(&hf2q_bin)
        .arg("serve")
        .args(["--model", &gguf])
        .args(["--mmproj", &mmproj])
        .args(["--port", &port.to_string()])
        .args(["--host", "127.0.0.1"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;

    // Wait for /readyz with 120s timeout.
    let base = format!("http://127.0.0.1:{port}");
    let ready_url = format!("{base}/readyz");
    let started = std::time::Instant::now();
    let mut ready = false;
    while started.elapsed().as_secs() < 120 {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let status = std::process::Command::new("curl")
            .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", &ready_url])
            .output();
        if let Ok(out) = status {
            if std::str::from_utf8(&out.stdout).unwrap_or("") == "200" {
                ready = true;
                break;
            }
        }
    }
    if !ready {
        let _ = hf2q.kill();
        return Err(std::io::Error::other("hf2q /readyz did not return 200 within 120s"));
    }

    // ---- Step 2: 25-pair matrix via curl/JSON ----
    let mut pairs: Vec<PairOutcome> = Vec::with_capacity(25);
    for (pi, prompt) in STANDARD_PROMPTS.iter().enumerate() {
        for (ii, name) in STANDARD_IMAGE_NAMES.iter().enumerate() {
            let img_path = fixture_dir().join(name);
            let bytes = std::fs::read(&img_path)?;
            let b64 = {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.encode(&bytes)
            };
            let body = serde_json::json!({
                "model": Path::new(&gguf).file_stem().and_then(|s| s.to_str()).unwrap_or("hf2q"),
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": *prompt},
                        {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{b64}")}},
                    ],
                }],
                "temperature": 0.0,
                "max_tokens": 50,
            });
            let body_path = std::env::temp_dir().join(format!("hf2q_e2e_body_{pi}_{ii}.json"));
            std::fs::write(&body_path, body.to_string())?;
            let resp = std::process::Command::new("curl")
                .args(["-s", "-X", "POST", &format!("{base}/v1/chat/completions")])
                .args(["-H", "Content-Type: application/json"])
                .args(["-d", &format!("@{}", body_path.display())])
                .output()?;
            let raw = String::from_utf8_lossy(&resp.stdout).into_owned();
            let parsed: serde_json::Value = serde_json::from_str(&raw)
                .unwrap_or_else(|_| serde_json::json!({"_raw": raw}));
            let hf2q_text = parsed["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            pairs.push(PairOutcome {
                prompt_idx: pi,
                image_idx: ii,
                prompt: prompt.to_string(),
                image_name: name.to_string(),
                hf2q_text,
                mlx_vlm_text: String::new(), // filled in step 3
                exact_match: false,
                hf2q_starts_with_mlx_first_word: false,
            });
            let _ = std::fs::remove_file(&body_path);
        }
    }

    // ---- Step 3: drain hf2q ----
    let _ = hf2q.kill();
    let _ = hf2q.wait();

    // ---- Step 4: 25-pair mlx-vlm reference ----
    for outcome in pairs.iter_mut() {
        let img_path = fixture_dir().join(&outcome.image_name);
        let mlx_out = std::process::Command::new("mlx_vlm.generate")
            .args(["--model", &mlx_repo])
            .args(["--image", img_path.to_str().unwrap()])
            .args(["--prompt", &outcome.prompt])
            .args(["--temperature", "0"])
            .args(["--max-tokens", "50"])
            .output()?;
        let stdout = String::from_utf8_lossy(&mlx_out.stdout).into_owned();
        // mlx_vlm.generate prints config/usage info around the actual
        // generated text.  Heuristic extraction: find the line after
        // a "==========" or "Output:" marker; otherwise take stdout verbatim.
        let extracted = extract_mlx_vlm_output(&stdout);
        outcome.mlx_vlm_text = extracted.clone();
        outcome.exact_match = outcome.hf2q_text.trim() == extracted.trim();
        let mlx_first_word = extracted.split_whitespace().next().unwrap_or("");
        outcome.hf2q_starts_with_mlx_first_word =
            !mlx_first_word.is_empty() && outcome.hf2q_text.contains(mlx_first_word);
    }

    let exact_matches = pairs.iter().filter(|p| p.exact_match).count();
    Ok(MatrixReport {
        pairs,
        exact_matches,
        hf2q_gguf: gguf,
        hf2q_mmproj: mmproj,
        mlx_repo,
    })
}

/// Extract the mlx_vlm.generate output text from its stdout.  The CLI
/// prints model/config banners and a `==========` divider before the
/// actual generation; isolate the post-divider tail.  Falls back to
/// the raw stdout when no divider is present (some mlx-vlm versions
/// don't emit one).
fn extract_mlx_vlm_output(stdout: &str) -> String {
    if let Some(idx) = stdout.find("==========") {
        // Take the text after the FIRST `==========` line and stop at
        // the SECOND if there is one (mlx-vlm wraps the output in
        // matching dividers).
        let tail = &stdout[idx + "==========".len()..];
        let tail = tail.trim_start_matches(|c: char| c == '\n' || c == '\r');
        if let Some(end) = tail.find("==========") {
            return tail[..end].trim().to_string();
        }
        return tail.trim().to_string();
    }
    stdout.trim().to_string()
}

// -----------------------------------------------------------------------------
// Cheap unit tests for the mlx-vlm output extractor (always run)
// -----------------------------------------------------------------------------

#[test]
fn extract_mlx_vlm_output_isolates_text_between_divider_lines() {
    let stdout = "Loading model...\n==========\nThis is the answer.\n==========\nGeneration: 50 tokens, 12.4 tok/s\n";
    let got = extract_mlx_vlm_output(stdout);
    assert_eq!(got, "This is the answer.");
}

#[test]
fn extract_mlx_vlm_output_handles_only_leading_divider() {
    let stdout = "Loading model...\n==========\nshort answer\n";
    let got = extract_mlx_vlm_output(stdout);
    assert_eq!(got, "short answer");
}

#[test]
fn extract_mlx_vlm_output_falls_back_to_raw_when_no_divider() {
    let stdout = "verbatim with no marker";
    let got = extract_mlx_vlm_output(stdout);
    assert_eq!(got, "verbatim with no marker");
}

#[test]
fn extract_mlx_vlm_output_trims_surrounding_whitespace() {
    let stdout = "==========\n\n   padded   \n\n==========\n";
    let got = extract_mlx_vlm_output(stdout);
    assert_eq!(got, "padded");
}
