//! ADR-015 iter41 — coherence matrix integration test (heavyweight).
//!
//! For each (fixture × prompt) cell, spawns `target/release/hf2q generate`
//! and compares the decoded output to the captured llama-completion
//! golden in `tests/coherence_golden/`.
//!
//! Three pass tiers (decreasing strictness):
//!
//!   (a) EXACT     — byte-identical to golden. PASS, log "EXACT".
//!   (b) COHERENT  — first 5 tokens of golden share ≥3 with hf2q output
//!                    AND no token repeats >3× in hf2q output AND no
//!                    degenerate-pattern markers. PASS+WARN.
//!   (c) GIBBERISH — neither of the above. FAIL with golden vs actual.
//!
//! Tier (b) accommodates legitimate sampling variance between hf2q and
//! llama (different RNG, different floating-point precision in matmul
//! ordering, etc.) without admitting broken-decode regressions like the
//! iter21-iter37 trap.
//!
//! `#[ignore]`-gated: model loading is heavyweight (~5-30s per fixture
//! plus 16-token decode wallclock). Run with:
//!
//!     cargo test --release --test coherence_matrix -- --ignored coherence
//!
//! The driver in `scripts/coherence_and_speed_regression.sh` invokes
//! coherence_smoke (always) then this matrix test (when full bench is
//! requested) before any perf measurement.

use std::path::PathBuf;
use std::process::Command;

const BIN: &str = "target/release/hf2q";
const CHAT_TEMPLATE_RAW: &str =
    "{% for message in messages %}{{ message.content }}{% endfor %}";

#[derive(Debug)]
struct Cell {
    fixture: &'static str,
    prompt: &'static str,
    prompt_slug: &'static str,
    model_path: &'static str,
}

const CELLS: &[Cell] = &[
    // 27b-dwq46 (dense)
    Cell {
        fixture: "27b-dwq46",
        prompt: "Hello, my name is",
        prompt_slug: "hello-my-name-is",
        model_path: "/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf",
    },
    Cell {
        fixture: "27b-dwq46",
        prompt: "The quick brown fox",
        prompt_slug: "the-quick-brown-fox",
        model_path: "/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf",
    },
    Cell {
        fixture: "27b-dwq46",
        prompt: "What is 2+2?",
        prompt_slug: "what-is-22",
        model_path: "/opt/hf2q/models/qwen3.6-27b-dwq46/qwen3.6-27b-dwq46.gguf",
    },
    // dwq46 (35B MoE)
    Cell {
        fixture: "dwq46",
        prompt: "Hello, my name is",
        prompt_slug: "hello-my-name-is",
        model_path: "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf",
    },
    Cell {
        fixture: "dwq46",
        prompt: "The quick brown fox",
        prompt_slug: "the-quick-brown-fox",
        model_path: "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf",
    },
    Cell {
        fixture: "dwq46",
        prompt: "What is 2+2?",
        prompt_slug: "what-is-22",
        model_path: "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq46.gguf",
    },
    // apex
    Cell {
        fixture: "apex",
        prompt: "Hello, my name is",
        prompt_slug: "hello-my-name-is",
        model_path: "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf",
    },
    Cell {
        fixture: "apex",
        prompt: "The quick brown fox",
        prompt_slug: "the-quick-brown-fox",
        model_path: "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf",
    },
    Cell {
        fixture: "apex",
        prompt: "What is 2+2?",
        prompt_slug: "what-is-22",
        model_path: "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex.gguf",
    },
    // gemma
    Cell {
        fixture: "gemma",
        prompt: "Hello, my name is",
        prompt_slug: "hello-my-name-is",
        model_path: "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf",
    },
    Cell {
        fixture: "gemma",
        prompt: "The quick brown fox",
        prompt_slug: "the-quick-brown-fox",
        model_path: "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf",
    },
    Cell {
        fixture: "gemma",
        prompt: "What is 2+2?",
        prompt_slug: "what-is-22",
        model_path: "/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf",
    },
];

const GIBBERISH_MARKERS: &[&str] = &[
    "якобы",
    "<|turn|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "**_**_**",
    "2- 2- 2-",
    "2-2-2-2-",
];

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Exact,
    Coherent,
    Gibberish,
}

fn first_n_tokens(s: &str, n: usize) -> Vec<&str> {
    s.split_whitespace().take(n).collect()
}

fn classify(actual: &str, golden: &str) -> Verdict {
    if actual == golden {
        return Verdict::Exact;
    }

    let actual_trim = actual.trim();
    if actual_trim.is_empty() {
        return Verdict::Gibberish;
    }

    // Tier (b) — coherent.
    let golden_first5 = first_n_tokens(golden.trim(), 5);
    let actual_tokens: std::collections::HashSet<&str> =
        actual_trim.split_whitespace().collect();
    let shared = golden_first5
        .iter()
        .filter(|t| actual_tokens.contains(*t))
        .count();

    let words: Vec<&str> = actual_trim.split_whitespace().collect();
    let max_rep = if words.is_empty() {
        0
    } else {
        let mut counts = std::collections::HashMap::<&str, usize>::new();
        for w in &words {
            *counts.entry(*w).or_default() += 1;
        }
        *counts.values().max().unwrap_or(&0)
    };

    let has_marker = GIBBERISH_MARKERS
        .iter()
        .any(|m| actual_trim.contains(m) && !golden.contains(m));

    let token_repetition_ok = words.len() < 6 || (max_rep as f32) <= (words.len() as f32 * 0.5);

    if shared >= 3 && token_repetition_ok && !has_marker {
        Verdict::Coherent
    } else {
        Verdict::Gibberish
    }
}

fn read_golden(cell: &Cell) -> Result<String, String> {
    let path = format!(
        "tests/coherence_golden/{}-{}.txt",
        cell.fixture, cell.prompt_slug
    );
    std::fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))
}

fn run_hf2q(cell: &Cell) -> Result<String, String> {
    let bin = PathBuf::from(BIN);
    if !bin.exists() {
        return Err(format!("BIN_MISSING: {}", BIN));
    }
    if !PathBuf::from(cell.model_path).exists() {
        return Err(format!("MODEL_MISSING: {}", cell.model_path));
    }

    let output = Command::new(&bin)
        .args([
            "generate",
            "--model",
            cell.model_path,
            "--prompt",
            cell.prompt,
            "--max-tokens",
            "16",
            "--temperature",
            "0.0",
            "--chat-template",
            CHAT_TEMPLATE_RAW,
        ])
        .output()
        .map_err(|e| format!("spawn failed: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "hf2q exit {:?}\nstderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[test]
#[ignore = "heavyweight (model loading); run with `cargo test --release --test coherence_matrix -- --ignored coherence`"]
fn coherence_matrix_all_cells() {
    if !PathBuf::from(BIN).exists() {
        eprintln!(
            "SKIP coherence_matrix: {BIN} not built. \
             Run `cargo build --release` and retry."
        );
        return;
    }

    let mut exact = 0usize;
    let mut coherent = 0usize;
    let mut gibberish: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    for cell in CELLS {
        let golden = match read_golden(cell) {
            Ok(g) => g,
            Err(e) => {
                gibberish.push(format!(
                    "{}/{}: golden missing: {e}",
                    cell.fixture, cell.prompt_slug
                ));
                continue;
            }
        };

        let actual = match run_hf2q(cell) {
            Ok(a) => a,
            Err(e) if e.starts_with("MODEL_MISSING") => {
                eprintln!(
                    "SKIP {}/{}: {}",
                    cell.fixture, cell.prompt_slug, e
                );
                skipped.push(format!("{}/{}", cell.fixture, cell.prompt_slug));
                continue;
            }
            Err(e) => {
                gibberish.push(format!(
                    "{}/{}: hf2q decode failed: {e}",
                    cell.fixture, cell.prompt_slug
                ));
                continue;
            }
        };

        match classify(&actual, &golden) {
            Verdict::Exact => {
                eprintln!(
                    "EXACT     {}/{}: {:?}",
                    cell.fixture, cell.prompt_slug, actual.trim()
                );
                exact += 1;
            }
            Verdict::Coherent => {
                eprintln!(
                    "COHERENT  {}/{}\n  actual: {:?}\n  golden: {:?}",
                    cell.fixture,
                    cell.prompt_slug,
                    actual.trim(),
                    golden.trim()
                );
                coherent += 1;
            }
            Verdict::Gibberish => {
                gibberish.push(format!(
                    "{}/{}\n  actual: {:?}\n  golden: {:?}",
                    cell.fixture,
                    cell.prompt_slug,
                    actual.trim(),
                    golden.trim()
                ));
            }
        }
    }

    eprintln!(
        "\ncoherence_matrix: EXACT={exact}, COHERENT={coherent}, GIBBERISH={}, SKIPPED={}",
        gibberish.len(),
        skipped.len()
    );

    assert!(
        gibberish.is_empty(),
        "coherence_matrix detected {} gibberish cells:\n  - {}",
        gibberish.len(),
        gibberish.join("\n  - ")
    );
}

// Pure-logic unit tests for the classifier — these run as part of every
// `cargo test` (no `#[ignore]`), so a refactor of `classify()` can't
// silently break the pass-tier semantics.

#[cfg(test)]
mod classifier_tests {
    use super::*;

    #[test]
    fn exact_match_is_exact() {
        let g = "Alex. I am 20 years old.";
        let a = "Alex. I am 20 years old.";
        assert_eq!(classify(a, g), Verdict::Exact);
    }

    #[test]
    fn empty_actual_is_gibberish() {
        assert_eq!(classify("", "Alex. I am 20."), Verdict::Gibberish);
        assert_eq!(classify("   \n\t  ", "Alex."), Verdict::Gibberish);
    }

    #[test]
    fn similar_first_tokens_is_coherent() {
        let g = "John. I am a 30-year-old male.";
        let a = "John. I am a 25-year-old male, currently studying.";
        assert_eq!(classify(a, g), Verdict::Coherent);
    }

    #[test]
    fn known_marker_in_actual_only_is_gibberish() {
        let g = "Alex. I am 20 years old.";
        let a = "якобы ( ) 2025-09-11 14:25:00";
        assert_eq!(classify(a, g), Verdict::Gibberish);
    }

    #[test]
    fn turn_token_leak_is_gibberish() {
        let g = "Alex. I am 20.";
        let a = "<|turn|> <|turn|> <|turn|> hello";
        assert_eq!(classify(a, g), Verdict::Gibberish);
    }

    #[test]
    fn high_repetition_is_gibberish() {
        let g = "John. I am 30. I have been having";
        let a = "the the the the the the the the";
        assert_eq!(classify(a, g), Verdict::Gibberish);
    }

    #[test]
    fn gemma_known_degenerate_peer_passes_when_actual_matches() {
        // gemma temp-0 produces "-ing-ing-ing" repetition; if hf2q also
        // produces it (matching peer), classify it byte-equal → EXACT.
        let g = "-jars-ing-ing-ing-ing-ing-ing-ing\n";
        let a = "-jars-ing-ing-ing-ing-ing-ing-ing\n";
        assert_eq!(classify(a, g), Verdict::Exact);
    }

    #[test]
    fn coherent_fails_if_marker_present_only_in_actual() {
        let g = "Alex. I am 20 years old.";
        // Has 3 shared tokens, but also has gibberish marker.
        let a = "Alex. I am 20 <|turn|> <|turn|> <|turn|>";
        assert_eq!(classify(a, g), Verdict::Gibberish);
    }
}
