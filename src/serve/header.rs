//! Prefill-stat printer + load-progress reporter for `cmd_generate`.
//!
//! Both surfaces frame the generation stream on stdout (product output,
//! not logs). At TTY, rendered dim so the generation stream itself reads
//! at full visual weight. Stripped of ANSI on non-TTY so piped/redirected
//! stdout captures plain text.
//!
//! Pre-ADR-018 this module also owned the 2-line `print_header_top` /
//! `HeaderInfoTop` surface for backend + model summary lines. ADR-018 C3
//! superseded that with the unified 13-line `crate::serve::load_info::print_banner`,
//! and ADR-018 C5 retired the legacy surface entirely (no production
//! callers remained after C3; the legacy-2-line-header feature flag was
//! also dropped). The prefill-stat surface (`print_header_prefill`) and
//! the load-progress reporter (`LoadProgress`) live on; both are called
//! exclusively from CLI generate paths and are NOT load facts in the
//! `LoadInfo` sense (they are wall-clock prefill stats / streaming
//! progress UI, respectively).
//!
//! See `docs/generate-ux-cleanup.md` §3 for the full spec.

use std::io::{self, Write};

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// Strip vendor prefix from GPU chip labels for compact header display.
/// Current strippable prefixes: "Apple ". Falls back to the full string
/// if no known prefix is present.
pub fn short_chip_label(name: &str) -> String {
    name.strip_prefix("Apple ").unwrap_or(name).to_string()
}

/// Printable info for the prefill-stats line (shown after prefill
/// completes, before the first generation token renders).
pub struct HeaderInfoPrefill {
    /// Number of prompt tokens fed into prefill.
    pub prefill_n: usize,
    /// Prefill wall-clock in milliseconds.
    pub prefill_ms: f64,
    /// Prefill throughput in tokens per second.
    pub prefill_tok_s: f64,
}

/// Write the prefill-stats line and the blank line that frames the
/// generation stream. Called once prefill finishes, just before the first
/// decode token prints.
pub fn print_header_prefill<W: Write>(
    w: &mut W,
    info: &HeaderInfoPrefill,
    tty: bool,
) -> io::Result<()> {
    let (d, r) = if tty { (DIM, RESET) } else { ("", "") };
    writeln!(
        w,
        "{d}prefill: {} tok in {:.0}ms ({:.0} tok/s){r}",
        info.prefill_n, info.prefill_ms, info.prefill_tok_s,
    )?;
    writeln!(w)?;
    w.flush()
}

/// In-place weight-load progress reporter using `\r` overwrite on stderr.
///
/// Active only when (a) stderr is a TTY and (b) verbosity == 0. At higher
/// verbosity levels the tracing debug events emitted by the loader already
/// provide per-layer detail, and mixing a `\r` progress line with them
/// produces garbled output. On non-TTY stderr the reporter is silent —
/// redirected/piped stderr gets only the tracing-level events.
pub struct LoadProgress {
    enabled: bool,
    n_layers: usize,
    last_width: usize,
}

impl LoadProgress {
    /// `stderr_is_tty` + `verbosity` determine whether output is emitted.
    pub fn new(stderr_is_tty: bool, verbosity: u8, n_layers: usize) -> Self {
        Self {
            enabled: stderr_is_tty && verbosity == 0,
            n_layers,
            last_width: 0,
        }
    }

    /// Render `\r loading {i}/{n} layers`. No-op if not enabled.
    pub fn on_layer(&mut self, i: usize) {
        if !self.enabled {
            return;
        }
        let line = format!("loading {}/{} layers", i, self.n_layers);
        self.last_width = line.len().max(self.last_width);
        eprint!("\r{line}");
        let _ = io::stderr().flush();
    }

    /// Clear the progress line. Call once before any non-progress output
    /// follows on stderr (or before the header prints on stdout).
    pub fn finish(&mut self) {
        if !self.enabled {
            return;
        }
        eprint!("\r{}\r", " ".repeat(self.last_width));
        let _ = io::stderr().flush();
        self.enabled = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_chip_label_strips_apple_prefix() {
        assert_eq!(short_chip_label("Apple M5 Max"), "M5 Max");
        assert_eq!(short_chip_label("Apple M3 Pro"), "M3 Pro");
    }

    #[test]
    fn short_chip_label_passes_through_unknown_prefix() {
        assert_eq!(short_chip_label("AMD Radeon Pro"), "AMD Radeon Pro");
        assert_eq!(short_chip_label(""), "");
    }

    /// ADR-018 C5: replacement for the legacy 2-line `header_top_*`
    /// tests retired alongside `print_header_top` / `HeaderInfoTop`.
    ///
    /// Asserts the new banner shape against a fixture LoadInfo. The
    /// banner is line-shape-stable across cold/warm loads and across
    /// both arch families (gemma4 / qwen35); the fixture below mirrors
    /// the design doc §6.1 Gemma sample (CLI, M5 Max, Q4_K_M) so the
    /// test pins both the per-line composition AND the ordering. A
    /// further set of golden tests at `crate::serve::load_info::tests`
    /// (`print_banner_golden_qwen35moe`, `print_banner_handles_absent_optional_fields`)
    /// pins the same shape against the Qwen35 / vision-absent variants;
    /// this test ensures the header.rs sibling-module path keeps the
    /// dependency on the unified surface live.
    #[test]
    fn print_banner_default_arm_smoke() {
        use crate::serve::load_info::{
            print_banner, ArchFamily, ChatTemplateSource, LoadInfo, MoeShape, TokenizerSource,
        };
        use crate::core::provenance::Provenance;
        use std::path::PathBuf;
        use std::time::Duration;

        let info = LoadInfo {
            model_id: "gemma-4-27b-it-Q4_K_M".to_string(),
            arch_str: "gemma4".to_string(),
            arch_family: ArchFamily::Gemma4,
            model_path: PathBuf::from("/cache/gemma-4-27b-it-Q4_K_M.gguf"),
            on_disk_bytes: (16.91_f64 * 1024.0 * 1024.0 * 1024.0).round() as u64,
            backend_chip: "Apple M5 Max".to_string(),
            backend: "mlx-native",
            n_layers: 62,
            hidden_size: 5376,
            vocab_size: 262_144,
            n_attention_heads: 32,
            n_key_value_heads: 16,
            head_dim: 128,
            sliding_window: Some(4096),
            full_attention_interval: None,
            max_context_length: Some(131_072),
            moe: Some(MoeShape {
                n_experts: 0,
                n_experts_per_tok: 0,
            }),
            quant_label: Some("Q4_K".to_string()),
            quant_bpw: Some(4.83),
            tokenizer_source: TokenizerSource::HfTokenizerJson {
                path: PathBuf::from("/cache/tokenizer.json"),
            },
            eos_token_ids: vec![1, 106],
            bos_token_id: Some(2),
            chat_template_source: ChatTemplateSource::GgufEmbedded,
            provenance: Provenance::External,
            vision_projector: None,
            load_wall_clock: Duration::from_secs_f64(2.41),
            resident_weight_bytes: Some((16.42_f64 * 1024.0 * 1024.0 * 1024.0).round() as u64),
            kv_cache_budget_bytes: None,
            kv_spill_active: false,
            tq_kv_active: false,
        };
        let mut buf = Vec::new();
        print_banner(&info, &mut buf, false).expect("print banner");
        let s = String::from_utf8(buf).expect("utf8");
        // ADR-027 Phase B iter-17: banner gained `tq_kv = ...` line, so
        // the count is now 14 (was 13 per design-doc §5.1).
        assert_eq!(
            s.lines().count(),
            14,
            "expected 14-line banner (post-iter-17 tq_kv addition), got\n{s}"
        );
        // Per-line shape: every line begins with `hf2q load: `.
        for line in s.lines() {
            assert!(
                line.starts_with("hf2q load: "),
                "line shape diverged: {line:?}"
            );
        }
        // Spot-check the architecture / quant-label / chip strings the
        // unified surface is responsible for forwarding from `LoadInfo`.
        assert!(s.contains("backend = mlx-native (M5 Max)"));
        assert!(s.contains("arch = gemma4, family = gemma4"));
        assert!(s.contains("quant = Q4_K dominant"));
        assert!(s.contains("ready in 2.41 s"));
    }

    #[test]
    fn header_prefill_adds_blank_line() {
        let info = HeaderInfoPrefill {
            prefill_n: 15,
            prefill_ms: 260.0,
            prefill_tok_s: 57.6,
        };
        let mut buf = Vec::new();
        print_header_prefill(&mut buf, &info, false).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s, "prefill: 15 tok in 260ms (58 tok/s)\n\n");
    }

    #[test]
    fn load_progress_disabled_when_not_tty() {
        let mut p = LoadProgress::new(false, 0, 30);
        p.on_layer(1);
        p.finish();
    }

    #[test]
    fn load_progress_disabled_when_verbose() {
        let mut p = LoadProgress::new(true, 1, 30);
        p.on_layer(1);
        p.finish();
    }
}
