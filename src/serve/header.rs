//! Default-mode header printer and load-progress reporter for `cmd_generate`.
//!
//! The three header lines frame the generation stream. They render on stdout
//! (not stderr) because they are product output, not logs. At TTY, rendered
//! dim so the generation stream itself reads at full visual weight. Stripped
//! of ANSI on non-TTY so piped/redirected stdout captures plain text.
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

/// Printable info for header lines 1 and 2 (shown after weights load,
/// before prefill begins).
pub struct HeaderInfoTop {
    /// Short-form chip label, e.g. "M5 Max". See [`short_chip_label`].
    pub chip: String,
    /// Backend identifier. `"mlx-native"` is the sole option today (ADR-008).
    pub backend: &'static str,
    /// Human-readable model name (GGUF `general.name` or file-stem fallback).
    pub model: String,
    /// Wall-clock seconds from GPU-init start through weights-resident.
    pub load_s: f64,
    /// Number of transformer layers actually resident in memory.
    pub n_layers: usize,
    /// GGUF file size in GB (authoritative; matches `ls -la` on the model).
    pub total_gb: f64,
}

/// Printable info for header line 3 (shown after prefill completes, before
/// the first generation token renders).
pub struct HeaderInfoPrefill {
    /// Number of prompt tokens fed into prefill.
    pub prefill_n: usize,
    /// Prefill wall-clock in milliseconds.
    pub prefill_ms: f64,
    /// Prefill throughput in tokens per second.
    pub prefill_tok_s: f64,
}

/// Write header lines 1 and 2 (backend + model summary).
/// `tty` controls ANSI dimming. Renders to `w` (typically stdout).
pub fn print_header_top<W: Write>(
    w: &mut W,
    info: &HeaderInfoTop,
    tty: bool,
) -> io::Result<()> {
    let (d, r) = if tty { (DIM, RESET) } else { ("", "") };
    writeln!(w, "{d}hf2q · {} · {}{r}", info.chip, info.backend)?;
    writeln!(
        w,
        "{d}{} · loaded in {:.1}s · {} layers · {:.1} GB{r}",
        info.model, info.load_s, info.n_layers, info.total_gb,
    )?;
    w.flush()
}

/// Write header line 3 (prefill stats) and the blank line that frames the
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

    #[test]
    fn header_top_no_tty_has_no_ansi() {
        let info = HeaderInfoTop {
            chip: "M5 Max".to_string(),
            backend: "mlx-native",
            model: "gemma-4-26B".to_string(),
            load_s: 2.4,
            n_layers: 30,
            total_gb: 16.9,
        };
        let mut buf = Vec::new();
        print_header_top(&mut buf, &info, false).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(!s.contains("\x1b["), "no ANSI when not tty");
        assert_eq!(
            s,
            "hf2q · M5 Max · mlx-native\n\
             gemma-4-26B · loaded in 2.4s · 30 layers · 16.9 GB\n"
        );
    }

    #[test]
    fn header_top_tty_has_dim() {
        let info = HeaderInfoTop {
            chip: "M5 Max".to_string(),
            backend: "mlx-native",
            model: "m".to_string(),
            load_s: 1.0,
            n_layers: 1,
            total_gb: 1.0,
        };
        let mut buf = Vec::new();
        print_header_top(&mut buf, &info, true).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("\x1b[2m"));
        assert!(s.contains("\x1b[0m"));
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
