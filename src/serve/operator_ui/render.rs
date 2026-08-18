use std::io::{self, Write};
use std::time::Duration;

use super::{DashboardState, Phase, RequestView};

pub(super) fn draw(state: &DashboardState) {
    let width = usize::from(console::Term::stderr().size().1).clamp(72, 160);
    let inner = width.saturating_sub(4);
    let mut frame = String::with_capacity(width.saturating_mul(20));
    frame.push_str("\x1b[H\x1b[2J\x1b[1;36m");
    frame.push_str(&format!(
        "hf2q live  {}\x1b[0m\n",
        truncate(&state.model, inner - 12)
    ));
    if let Some(reason) = state.unhealthy.as_deref() {
        frame.push_str(&format!(
            "\x1b[31m● unhealthy — restart required\x1b[0m  {}  ·  up {}\n",
            state.endpoint,
            format_duration(state.started.elapsed())
        ));
        frame.push_str(&format!("  {}\n", truncate(reason, inner)));
    } else {
        frame.push_str(&format!(
            "\x1b[32m● ready\x1b[0m  {}  ·  {} slots  ·  up {}\n",
            state.endpoint,
            state.max_slots,
            format_duration(state.started.elapsed())
        ));
    }
    frame.push_str(&format!("{}\n", "─".repeat(width.min(160))));
    if state.requests.is_empty() {
        frame.push_str("  idle — waiting for OpenAI-compatible requests\n");
    } else {
        for request in state.requests.values() {
            frame.push_str(&render_request(request, inner));
            frame.push('\n');
        }
    }
    frame.push_str(&format!("{}\n", "─".repeat(width.min(160))));
    frame.push_str("recent events\n");
    if state.logs.is_empty() {
        frame.push_str("  no warnings or lifecycle events\n");
    } else {
        for line in &state.logs {
            frame.push_str("  ");
            frame.push_str(&truncate(line, inner));
            frame.push('\n');
        }
    }
    frame.push_str(&format!(
        "\x1b[2m{} · Ctrl-C shuts down cleanly\x1b[0m",
        state.family
    ));
    let mut stderr = io::stderr().lock();
    let _ = stderr.write_all(frame.as_bytes());
    let _ = stderr.flush();
}

pub(super) fn render_request(request: &RequestView, width: usize) -> String {
    let label = request
        .slot
        .map_or_else(|| format!("r{}", request.key.id), |slot| format!("s{slot}"));
    let phase = match request.phase {
        Phase::Queued => "queued ",
        Phase::Prefill => "prefill",
        Phase::Decode => "decode ",
        Phase::Complete => "done   ",
        Phase::Cancelled => "cancel ",
        Phase::Failed => "failed ",
    };
    let detail = match request.phase {
        Phase::Prefill => {
            let percent = if request.work_tokens == 0 {
                100.0
            } else {
                100.0 * request.completed_tokens as f64 / request.work_tokens as f64
            };
            let eta = if request.rate > 0.0 {
                Duration::from_secs_f64(
                    request.work_tokens.saturating_sub(request.completed_tokens) as f64
                        / request.rate,
                )
            } else {
                Duration::ZERO
            };
            format!(
                "{} {:5.1}%  {:>7}/{:<7} new  ·  cache {:>7}  ·  {:>6.1} tok/s  ·  ETA {}",
                progress_bar(percent, 18),
                percent,
                request.completed_tokens,
                request.work_tokens,
                request.cached_tokens,
                request.rate,
                if request.rate > 0.0 {
                    format_duration(eta)
                } else {
                    "—".into()
                }
            )
        }
        Phase::Decode => {
            let thinking = match (request.thinking_tokens, request.thinking_budget) {
                (Some(tokens), Some(budget)) if request.thinking_forced_closed => {
                    format!(
                        "  ·  think capped {tokens}/{budget}  ·  output {}",
                        if request.answer_event_delivered {
                            "started"
                        } else {
                            "pending"
                        }
                    )
                }
                (Some(tokens), Some(budget)) => format!("  ·  think {tokens}/{budget}"),
                _ => String::new(),
            };
            format!(
                "completion {:>5}/{:<5}{thinking}  ·  prompt {} cache {}  ·  {:>5.1} tok/s  ·  elapsed {}",
                request.generated_tokens,
                request.max_tokens,
                request.prompt_tokens,
                request.cached_tokens,
                request.rate,
                format_duration(request.started.elapsed())
            )
        }
        Phase::Queued => format!(
            "prompt {} tokens  ·  completion budget {}",
            request.prompt_tokens, request.max_tokens
        ),
        _ => format!(
            "prompt {}  ·  cache {}  ·  generated {}  ·  {}",
            request.prompt_tokens,
            request.cached_tokens,
            request.generated_tokens,
            format_duration(request.started.elapsed())
        ),
    };
    truncate(
        &format!("{label:<4} {phase}  {detail}  [{}]", request.mode),
        width,
    )
}

pub(super) fn progress_bar(percent: f64, width: usize) -> String {
    let filled = ((percent.clamp(0.0, 100.0) / 100.0) * width as f64).round() as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

pub(super) fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds >= 3_600 {
        format!("{}h{:02}m", seconds / 3_600, (seconds / 60) % 60)
    } else if seconds >= 60 {
        format!("{}m{:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{}s", seconds)
    }
}

fn truncate(value: &str, width: usize) -> String {
    let mut chars = value.chars();
    let mut out: String = chars.by_ref().take(width).collect();
    if chars.next().is_some() && width > 1 {
        out.pop();
        out.push('…');
    }
    out
}

pub(super) fn enter_terminal() {
    let mut stderr = io::stderr().lock();
    let _ = stderr.write_all(b"\x1b[?1049h\x1b[?25l\x1b[H\x1b[2J");
    let _ = stderr.flush();
}

pub(super) fn restore_terminal() {
    let mut stderr = io::stderr().lock();
    let _ = stderr.write_all(b"\x1b[?25h\x1b[?1049l");
    let _ = stderr.flush();
}
