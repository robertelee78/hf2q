//! Scrollback-preserving model-preparation UI for owned diagnostic chat.
//!
//! Interactive chat stdout or direct-serve stderr gets one live indicatif row.
//! Pipes and test captures keep stable line-oriented text with no ANSI. This
//! module never enters raw mode or an alternate screen.

use std::io::Write;
use std::time::Duration;

use anyhow::Result;
use console::Style;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use crate::serve::startup_progress::{
    format_duration, human_bytes, render_verified_ready, terminal_safe_text as clean, StartupEvent,
};

const COPPER_256: u8 = 166;

pub(crate) fn interactive_startup_enabled(output_is_tty: bool) -> bool {
    interactive_startup_enabled_for(
        output_is_tty,
        std::env::var_os("CI").is_some(),
        std::env::var("TERM").ok().as_deref(),
    )
}

fn interactive_startup_enabled_for(output_is_tty: bool, ci: bool, term: Option<&str>) -> bool {
    output_is_tty && !ci && term != Some("dumb")
}

#[derive(Clone, Copy)]
pub(crate) enum StartupOutput {
    Stdout,
    Stderr,
}

pub(crate) struct StartupUi<'a, W: Write> {
    output: &'a mut W,
    bar: Option<ProgressBar>,
    use_color: bool,
    current_phase: String,
    started: std::time::Instant,
}

impl<'a, W: Write> StartupUi<'a, W> {
    pub(crate) fn new(output: &'a mut W, interactive: bool, target: StartupOutput) -> Self {
        let use_color =
            interactive && std::env::var_os("NO_COLOR").is_none_or(|value| value.is_empty());
        let bar = interactive.then(|| {
            let draw_target = match target {
                StartupOutput::Stdout => ProgressDrawTarget::stdout_with_hz(12),
                StartupOutput::Stderr => ProgressDrawTarget::stderr_with_hz(12),
            };
            let bar = ProgressBar::with_draw_target(None, draw_target);
            bar.set_style(spinner_style(use_color));
            bar.enable_steady_tick(Duration::from_millis(90));
            bar
        });
        Self {
            output,
            bar,
            use_color,
            current_phase: "waiting for the owned server's first preparation event".into(),
            started: std::time::Instant::now(),
        }
    }

    pub(crate) fn announce_owned(&mut self, targeted: bool) -> Result<()> {
        if let Some(bar) = self.bar.as_ref() {
            bar.set_message(if targeted {
                "Inspecting local model stores"
            } else {
                "Starting an owned local hf2q server"
            });
        } else if targeted {
            writeln!(
                self.output,
                "starting an owned hf2q server for the requested model"
            )?;
            writeln!(
                self.output,
                "waiting for the owned server to inspect local model stores"
            )?;
            self.output.flush()?;
        } else {
            writeln!(self.output, "no local hf2q server found; starting one")?;
            self.output.flush()?;
        }
        Ok(())
    }

    pub(crate) fn announce_direct(&mut self, target: Option<&str>) -> Result<()> {
        let message = target.map_or_else(
            || "Starting the local hf2q server".to_owned(),
            |target| format!("Preparing {}", clean(target)),
        );
        self.current_phase = message.clone();
        if let Some(bar) = self.bar.as_ref() {
            bar.set_message(message);
        } else {
            writeln!(self.output, "{message}")?;
            self.output.flush()?;
        }
        Ok(())
    }

    pub(crate) fn event(&mut self, event: StartupEvent) -> Result<()> {
        self.current_phase = event.heartbeat_label();
        let Some(bar) = self.bar.as_ref() else {
            writeln!(self.output, "{}", event.render())?;
            self.output.flush()?;
            return Ok(());
        };

        match event {
            StartupEvent::LocalSearch {
                repository,
                requested_quant,
            } => {
                bar.set_style(spinner_style(self.use_color));
                bar.set_message(format!(
                    "Inspecting local stores for {}{}",
                    clean(&repository),
                    requested_quant
                        .map(|quant| format!(":{}", clean(&quant)))
                        .unwrap_or_default()
                ));
            }
            StartupEvent::LocalCandidate {
                quant,
                origin,
                bytes,
                filename,
            } => {
                history(
                    bar,
                    format!(
                        "found local {} `{}` • {} • {origin}",
                        clean(&quant),
                        clean(&filename),
                        human_bytes(bytes),
                        origin = origin.label()
                    ),
                    HistoryKind::Found,
                    self.use_color,
                );
                bar.set_style(spinner_style(self.use_color));
                bar.set_message("Inspecting bounded GGUF metadata and tensor directory");
            }
            StartupEvent::VerifyStart {
                artifact,
                bytes,
                filename,
            } => {
                bar.set_length(bytes);
                bar.set_position(0);
                bar.set_style(bytes_style(self.use_color));
                bar.set_message(format!(
                    "Verifying local {} `{}`",
                    clean(&artifact),
                    clean(&filename)
                ));
            }
            StartupEvent::VerifyProgress {
                artifact,
                completed_bytes,
                total_bytes,
                ..
            } => {
                bar.set_length(total_bytes);
                bar.set_position(completed_bytes);
                bar.set_style(bytes_style(self.use_color));
                bar.set_message(format!("Verifying local {}", clean(&artifact)));
            }
            StartupEvent::LocalReady {
                quant,
                origin,
                filename,
            } => {
                history(
                    bar,
                    format!(
                        "compatible local {} `{}` • {} • no model download or full-file hash needed",
                        clean(&quant),
                        clean(&filename),
                        origin.label()
                    ),
                    HistoryKind::Success,
                    self.use_color,
                );
                reset_spinner(bar, self.use_color, "Preparing the compatible local model");
            }
            StartupEvent::ModelPrepared {
                quant,
                origin,
                filename,
            } => {
                history(
                    bar,
                    format!(
                        "prepared {} `{}` • {} • managed model store",
                        clean(&quant),
                        clean(&filename),
                        origin.label()
                    ),
                    HistoryKind::Success,
                    self.use_color,
                );
                reset_spinner(bar, self.use_color, "Loading the prepared model");
            }
            StartupEvent::HubMetadata { repository } => {
                reset_spinner(
                    bar,
                    self.use_color,
                    format!(
                        "Querying exact-revision metadata for {} • no payload yet",
                        clean(&repository)
                    ),
                );
            }
            StartupEvent::HostedDownload { filename, bytes } => {
                history(
                    bar,
                    format!(
                        "selected hosted `{}` • {} • destination: managed model store",
                        clean(&filename),
                        human_bytes(bytes)
                    ),
                    HistoryKind::Download,
                    self.use_color,
                );
                reset_spinner(
                    bar,
                    self.use_color,
                    format!(
                        "Downloading `{}` • {}",
                        clean(&filename),
                        human_bytes(bytes)
                    ),
                );
            }
            StartupEvent::ProjectorPrepare { filename, bytes } => {
                history(
                    bar,
                    format!(
                        "multimodal projector `{}` • {} • exact repository revision",
                        clean(&filename),
                        human_bytes(bytes)
                    ),
                    HistoryKind::Download,
                    self.use_color,
                );
                reset_spinner(
                    bar,
                    self.use_color,
                    format!("Locating or downloading projector `{}`", clean(&filename)),
                );
            }
            StartupEvent::NativeConversion { repository, quant } => {
                history(
                    bar,
                    format!(
                        "no compatible hosted GGUF • native {} conversion selected",
                        clean(&quant)
                    ),
                    HistoryKind::Convert,
                    self.use_color,
                );
                reset_spinner(
                    bar,
                    self.use_color,
                    format!(
                        "Checking/downloading missing source weights for {}, then converting to {}",
                        clean(&repository),
                        clean(&quant)
                    ),
                );
            }
            StartupEvent::TextLoad {
                quant,
                bytes,
                filename,
            } => reset_spinner(
                bar,
                self.use_color,
                format!(
                    "Loading {} `{}` into Metal and warming kernels • {}",
                    clean(&quant),
                    clean(&filename),
                    human_bytes(bytes)
                ),
            ),
            StartupEvent::TextReady { elapsed_ms } => {
                history(
                    bar,
                    format!(
                        "text model loaded and warmed in {}",
                        format_duration(Duration::from_millis(elapsed_ms))
                    ),
                    HistoryKind::Success,
                    self.use_color,
                );
                reset_spinner(bar, self.use_color, "Finishing server startup");
            }
            StartupEvent::ProjectorLoad { bytes, filename } => reset_spinner(
                bar,
                self.use_color,
                format!(
                    "Loading multimodal projector `{}` • {}",
                    clean(&filename),
                    human_bytes(bytes)
                ),
            ),
            StartupEvent::ProjectorReady { elapsed_ms } => {
                history(
                    bar,
                    format!(
                        "multimodal projector loaded and warmed in {}",
                        format_duration(Duration::from_millis(elapsed_ms))
                    ),
                    HistoryKind::Success,
                    self.use_color,
                );
                reset_spinner(bar, self.use_color, "Finishing server startup");
            }
            StartupEvent::TextOnlyFallback { reason } => {
                history(
                    bar,
                    format!("serving text-only • {}", reason.label()),
                    HistoryKind::Warning,
                    self.use_color,
                );
                reset_spinner(bar, self.use_color, "Finishing text-only server startup");
            }
        }
        Ok(())
    }

    pub(crate) fn heartbeat(&mut self, elapsed: Duration) -> Result<()> {
        if let Some(bar) = self.bar.as_ref() {
            bar.set_message(format!(
                "Still {} • {} elapsed",
                self.current_phase,
                format_duration(elapsed)
            ));
        } else {
            writeln!(
                self.output,
                "still {} ({}s elapsed)",
                self.current_phase,
                elapsed.as_secs()
            )?;
            self.output.flush()?;
        }
        Ok(())
    }

    pub(crate) fn ready(&mut self, address: &str) -> Result<()> {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
            writeln!(
                self.output,
                "{}  {}  {}",
                styled("✓ ready", BrandStyle::Success, self.use_color),
                clean(address),
                styled(
                    &format!("• startup {}", format_duration(self.started.elapsed())),
                    BrandStyle::Dim,
                    self.use_color,
                )
            )?;
        } else {
            writeln!(self.output, "{}", render_verified_ready(address))?;
        }
        self.output.flush()?;
        Ok(())
    }

    /// Finish direct-serve preparation without claiming HTTP health.
    ///
    /// Owned chat calls `ready` only after its parent has performed the HTTP
    /// health check. Direct serve has merely retained and bound the listener
    /// at this point; `axum::serve` has not begun polling yet.
    pub(crate) fn listener_bound(&mut self, address: &str, model_prepared: bool) -> Result<()> {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
            let detail = if model_prepared {
                format!(
                    "• model prepared in {} • starting HTTP service",
                    format_duration(self.started.elapsed())
                )
            } else {
                "• starting HTTP service • no model preloaded".to_owned()
            };
            writeln!(
                self.output,
                "{}  {}  {}",
                styled("✓ listener bound", BrandStyle::Success, self.use_color),
                clean(address),
                styled(&detail, BrandStyle::Dim, self.use_color)
            )?;
        } else if model_prepared {
            writeln!(
                self.output,
                "listen: model prepared; listener bound at {}; starting HTTP service",
                clean(address)
            )?;
        } else {
            writeln!(
                self.output,
                "listen: listener bound at {}; starting HTTP service; no model preloaded",
                clean(address)
            )?;
        }
        self.output.flush()?;
        Ok(())
    }
}

impl<W: Write> Drop for StartupUi<'_, W> {
    fn drop(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
    }
}

fn reset_spinner(bar: &ProgressBar, use_color: bool, message: impl Into<String>) {
    bar.reset();
    bar.set_style(spinner_style(use_color));
    bar.set_message(message.into());
}

fn spinner_style(use_color: bool) -> ProgressStyle {
    ProgressStyle::with_template(if use_color {
        "  {spinner:.yellow} {wide_msg:.white}  {elapsed_precise:.dim}"
    } else {
        "  {spinner} {wide_msg}  {elapsed_precise}"
    })
    .expect("static startup spinner template")
    .tick_strings(&["▹▹▹▹▹", "▸▹▹▹▹", "▹▸▹▹▹", "▹▹▸▹▹", "▹▹▹▸▹", "▹▹▹▹▸"])
}

fn bytes_style(use_color: bool) -> ProgressStyle {
    ProgressStyle::with_template(if use_color {
        "  {spinner:.yellow} {msg:.42} {wide_bar:.blue/dim} {bytes:>9}/{total_bytes:<9} {bytes_per_sec:>10} ETA {eta}"
    } else {
        "  {spinner} {msg:.42} {wide_bar} {bytes:>9}/{total_bytes:<9} {bytes_per_sec:>10} ETA {eta}"
    })
    .expect("static startup byte template")
    .progress_chars("━╸─")
    .tick_strings(&["▹▹▹▹▹", "▸▹▹▹▹", "▹▸▹▹▹", "▹▹▸▹▹", "▹▹▹▸▹", "▹▹▹▹▸"])
}

#[derive(Clone, Copy)]
enum HistoryKind {
    Found,
    Success,
    Download,
    Convert,
    Warning,
}

fn history(bar: &ProgressBar, message: String, kind: HistoryKind, use_color: bool) {
    let (symbol, style) = match kind {
        HistoryKind::Found => ("●", BrandStyle::Cobalt),
        HistoryKind::Success => ("✓", BrandStyle::Success),
        HistoryKind::Download => ("↓", BrandStyle::Cobalt),
        HistoryKind::Convert => ("◆", BrandStyle::Copper),
        HistoryKind::Warning => ("!", BrandStyle::Warning),
    };
    bar.println(format!("  {}  {message}", styled(symbol, style, use_color)));
}

#[derive(Clone, Copy)]
enum BrandStyle {
    Copper,
    Cobalt,
    Success,
    Warning,
    Dim,
}

fn styled(value: &str, style: BrandStyle, enabled: bool) -> String {
    if !enabled {
        return value.to_owned();
    }
    let style = match style {
        BrandStyle::Copper => Style::new().color256(COPPER_256),
        BrandStyle::Cobalt => Style::new().blue(),
        BrandStyle::Success => Style::new().green(),
        BrandStyle::Warning => Style::new().yellow(),
        BrandStyle::Dim => Style::new().dim(),
    };
    style.apply_to(value).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_startup_remains_stable_line_oriented_text() {
        let mut output = Vec::new();
        {
            let mut ui = StartupUi::new(&mut output, false, StartupOutput::Stdout);
            ui.announce_owned(true).unwrap();
            ui.event(StartupEvent::LocalReady {
                quant: "Q6_K".into(),
                origin: crate::serve::startup_progress::StartupOrigin::ManualDownload,
                filename: "model.gguf".into(),
            })
            .unwrap();
            ui.ready("http://127.0.0.1:9123").unwrap();
        }
        let rendered = String::from_utf8(output).unwrap();
        assert!(!rendered.contains('\u{1b}'));
        assert!(rendered.contains("no model download or full-file hash needed"));
        assert!(rendered.contains("ready: verified hf2q endpoint"));
    }

    #[test]
    fn direct_listener_state_never_claims_http_verification() {
        let mut output = Vec::new();
        {
            let mut ui = StartupUi::new(&mut output, false, StartupOutput::Stderr);
            ui.announce_direct(Some("owner/model:Q4_K_M")).unwrap();
            ui.listener_bound("http://127.0.0.1:9123", true).unwrap();
        }
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("listener bound"), "{rendered}");
        assert!(rendered.contains("starting HTTP service"), "{rendered}");
        assert!(!rendered.contains("verified"), "{rendered}");
        assert!(!rendered.contains("\u{1b}[?1049"), "{rendered}");
    }

    #[test]
    fn model_less_listener_never_claims_model_preparation() {
        let mut output = Vec::new();
        {
            let mut ui = StartupUi::new(&mut output, false, StartupOutput::Stderr);
            ui.announce_direct(None).unwrap();
            ui.listener_bound("http://127.0.0.1:9123", false).unwrap();
        }
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("no model preloaded"), "{rendered}");
        assert!(!rendered.contains("model prepared"), "{rendered}");
    }

    #[test]
    fn interactive_event_text_neutralizes_controls_and_bidi() {
        let hostile = "owner/model\u{1b}[31m\u{202e}";
        assert_eq!(clean(hostile), "owner/model�[31m�");
        let event = StartupEvent::LocalSearch {
            repository: hostile.into(),
            requested_quant: Some("Q4_K_M\n\u{2066}".into()),
        };
        let StartupEvent::LocalSearch {
            repository,
            requested_quant,
        } = event
        else {
            unreachable!()
        };
        let rendered = format!(
            "Inspecting local stores for {}:{}",
            clean(&repository),
            clean(requested_quant.as_deref().unwrap())
        );
        assert!(!rendered.contains('\u{1b}'));
        assert!(!rendered.contains('\n'));
        assert!(!rendered.contains('\u{202e}'));
        assert!(!rendered.contains('\u{2066}'));
    }

    #[test]
    fn automatic_ui_rejects_ci_and_dumb_term() {
        assert!(!interactive_startup_enabled_for(
            true,
            true,
            Some("xterm-256color")
        ));
        assert!(!interactive_startup_enabled_for(true, false, Some("dumb")));
        assert!(interactive_startup_enabled_for(
            true,
            false,
            Some("xterm-256color")
        ));
        assert!(!interactive_startup_enabled_for(
            false,
            false,
            Some("xterm-256color")
        ));
    }
}
