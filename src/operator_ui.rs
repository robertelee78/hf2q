//! Global scrollback-safe operator presentation.
//!
//! The rabbit is never character art. The exact hf2q.us SVG is compiled into
//! deterministic raster assets, then emitted through a native terminal image
//! protocol when the terminal is positively identified. Unsupported native
//! image protocols get an exact-source ANSI raster. stdout remains a command-
//! data boundary; this module writes stderr only.

use std::io::{self, IsTerminal, Write};

use base64::Engine as _;
use console::Style;

use crate::cli::{Cli, Command, LogFormat, TerminalGraphicsArg};

#[cfg(test)]
const HEAD_SVG_FILE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/hf2q-head.svg"));
const HEAD_PNG_FILE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/hf2q-head.png"));
const ANSI_HEAD_RGBA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/hf2q-head-ansi.rgba"));
const LOGO_COLUMNS: u16 = 14;
const LOGO_ROWS: u16 = 10;
const ANSI_LOGO_COLUMNS: u16 = 20;
const ANSI_LOGO_PIXEL_ROWS: u16 = 28;
const LOGO_MIN_COLUMNS: u16 = ANSI_LOGO_COLUMNS + 4;
const KITTY_CHUNK_BYTES: usize = 4096;
const COPPER_256: u8 = 166;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphicsProtocol {
    Kitty,
    Iterm2,
    Ansi,
}

#[derive(Debug)]
struct TerminalFacts {
    stdout_tty: bool,
    stderr_tty: bool,
    ci: bool,
    no_color: bool,
    columns: u16,
    term: Option<String>,
    term_program: Option<String>,
    lc_terminal: Option<String>,
    kitty_window_id: bool,
    ghostty: bool,
    cmux: bool,
    tmux: bool,
    screen: bool,
}

impl TerminalFacts {
    fn current() -> Self {
        let term = nonempty_env("TERM");
        Self {
            stdout_tty: io::stdout().is_terminal(),
            stderr_tty: io::stderr().is_terminal(),
            ci: std::env::var_os("CI").is_some(),
            no_color: std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty()),
            columns: console::Term::stderr().size().1,
            term: term.clone(),
            term_program: nonempty_env("TERM_PROGRAM"),
            lc_terminal: nonempty_env("LC_TERMINAL"),
            kitty_window_id: std::env::var_os("KITTY_WINDOW_ID")
                .is_some_and(|value| !value.is_empty()),
            ghostty: std::env::var_os("GHOSTTY_RESOURCES_DIR")
                .is_some_and(|value| !value.is_empty()),
            cmux: std::env::var_os("CMUX_SURFACE_ID").is_some_and(|value| !value.is_empty()),
            tmux: std::env::var_os("TMUX").is_some_and(|value| !value.is_empty()),
            screen: term
                .as_deref()
                .is_some_and(|value| value.starts_with("screen")),
        }
    }

    #[cfg(test)]
    fn interactive(term_program: Option<&str>) -> Self {
        Self {
            stdout_tty: true,
            stderr_tty: true,
            ci: false,
            no_color: true,
            columns: 100,
            term: Some("xterm-256color".into()),
            term_program: term_program.map(str::to_owned),
            lc_terminal: None,
            kitty_window_id: false,
            ghostty: false,
            cmux: false,
            tmux: false,
            screen: false,
        }
    }
}

pub(crate) fn print_global_banner(cli: &Cli) {
    let facts = TerminalFacts::current();
    if !banner_enabled(cli, &facts) {
        return;
    }

    let mut stderr = io::stderr().lock();
    // Branding must never prevent the requested command from running. Exact
    // rasterization/protocol failure degrades only to the wordmark.
    let _ = render_banner(&mut stderr, cli.terminal_graphics, &facts);
}

/// Brand the interactive bare-command overview before Clap exits.
///
/// There is no parsed [`Cli`] on this path because its required subcommand is
/// deliberately absent. The bare invocation has no operator overrides, so it
/// uses the documented defaults while retaining every TTY/CI suppression
/// boundary used by accepted commands.
pub(crate) fn print_bare_invocation_banner() {
    let facts = TerminalFacts::current();
    if !interactive_banner_enabled(LogFormat::Text, TerminalGraphicsArg::Auto, &facts) {
        return;
    }

    let mut stderr = io::stderr().lock();
    let _ = render_banner(&mut stderr, TerminalGraphicsArg::Auto, &facts);
}

fn render_banner(
    output: &mut impl Write,
    requested: TerminalGraphicsArg,
    facts: &TerminalFacts,
) -> io::Result<()> {
    if facts.columns >= LOGO_MIN_COLUMNS {
        if let Some(protocol) = select_protocol(requested, facts) {
            let emitted = match protocol {
                GraphicsProtocol::Kitty => {
                    emit_kitty(output, HEAD_PNG_FILE, LOGO_COLUMNS, LOGO_ROWS, facts.tmux)
                }
                GraphicsProtocol::Iterm2 => {
                    emit_iterm2(output, HEAD_PNG_FILE, LOGO_COLUMNS, LOGO_ROWS, facts.tmux)
                }
                GraphicsProtocol::Ansi => {
                    debug_assert!(facts.columns >= ANSI_LOGO_COLUMNS + 4);
                    emit_ansi_raster(output)
                }
            };
            if emitted.is_err() {
                // A partially written terminal sequence cannot be repaired,
                // but the normal wordmark below still identifies the tool.
                let _ = output.flush();
            }
        }
    }

    let use_color = !facts.no_color;
    let name = styled("hf2q", Style::new().color256(COPPER_256).bold(), use_color);
    let version = styled(
        &format!("v{}", env!("CARGO_PKG_VERSION")),
        Style::new().dim(),
        use_color,
    );
    let tagline = styled(
        "Hugging Face → native GGUF → Apple Silicon",
        Style::new().dim(),
        use_color,
    );
    writeln!(output, "  {name}  {version}")?;
    writeln!(output, "  {tagline}")?;
    writeln!(output)?;
    output.flush()
}

fn banner_enabled(cli: &Cli, facts: &TerminalFacts) -> bool {
    interactive_banner_enabled(cli.log_format, cli.terminal_graphics, facts)
        && !matches!(
            cli.command,
            Command::StandaloneInstall(_)
                | Command::FetchHubGguf(_)
                | Command::CatalogHubGguf(_)
                | Command::VerifyLocalGguf(_)
                | Command::SourceTeacher(_)
                | Command::SourceTeacherReference(_)
                | Command::SourceTeacherAcceptanceVerify(_)
                | Command::Completions(_)
        )
        && !matches!(&cli.command, Command::Serve(args) if args.quiet)
}

fn interactive_banner_enabled(
    log_format: LogFormat,
    terminal_graphics: TerminalGraphicsArg,
    facts: &TerminalFacts,
) -> bool {
    facts.stdout_tty
        && facts.stderr_tty
        && !facts.ci
        && facts.term.as_deref() != Some("dumb")
        && matches!(log_format, LogFormat::Text)
        && terminal_graphics != TerminalGraphicsArg::Off
}

fn select_protocol(
    requested: TerminalGraphicsArg,
    facts: &TerminalFacts,
) -> Option<GraphicsProtocol> {
    match requested {
        TerminalGraphicsArg::Off => None,
        TerminalGraphicsArg::Kitty => Some(GraphicsProtocol::Kitty),
        TerminalGraphicsArg::Iterm2 => Some(GraphicsProtocol::Iterm2),
        TerminalGraphicsArg::Ansi => Some(GraphicsProtocol::Ansi),
        TerminalGraphicsArg::Auto => detect_protocol(facts),
    }
}

fn detect_protocol(facts: &TerminalFacts) -> Option<GraphicsProtocol> {
    // Environment hints can describe an outer terminal while running under a
    // multiplexer whose passthrough policy is unknown. Auto therefore refuses
    // graphics there; an explicit selector is the operator's opt-in.
    if facts.tmux || facts.screen {
        return Some(GraphicsProtocol::Ansi);
    }

    let program = facts.term_program.as_deref().unwrap_or_default();
    let lc_terminal = facts.lc_terminal.as_deref().unwrap_or_default();
    let term = facts.term.as_deref().unwrap_or_default();
    if facts.cmux
        || facts.kitty_window_id
        || facts.ghostty
        || contains_folded(program, "kitty")
        || contains_folded(program, "ghostty")
        || contains_folded(term, "kitty")
    {
        return Some(GraphicsProtocol::Kitty);
    }
    if [program, lc_terminal].iter().any(|value| {
        ["iterm", "wezterm", "warpterminal", "rio", "mintty"]
            .iter()
            .any(|needle| contains_folded(value, needle))
    }) {
        return Some(GraphicsProtocol::Iterm2);
    }
    Some(GraphicsProtocol::Ansi)
}

#[cfg(test)]
fn exact_head_svg() -> &'static [u8] {
    HEAD_SVG_FILE
}

fn emit_kitty(
    output: &mut impl Write,
    png: &[u8],
    columns: u16,
    rows: u16,
    tmux: bool,
) -> io::Result<()> {
    let encoded = base64::engine::general_purpose::STANDARD.encode(png);
    let chunks: Vec<&[u8]> = encoded.as_bytes().chunks(KITTY_CHUNK_BYTES).collect();
    for (index, chunk) in chunks.iter().enumerate() {
        let more = usize::from(index + 1 != chunks.len());
        let mut sequence = Vec::with_capacity(chunk.len() + 96);
        if index == 0 {
            write!(
                sequence,
                "\x1b_Ga=T,f=100,c={columns},r={rows},C=1,q=2,m={more};"
            )?;
        } else {
            write!(sequence, "\x1b_Gm={more};")?;
        }
        sequence.extend_from_slice(chunk);
        sequence.extend_from_slice(b"\x1b\\");
        emit_terminal_sequence(output, &sequence, tmux)?;
    }
    for _ in 0..rows {
        output.write_all(b"\r\n")?;
    }
    Ok(())
}

fn emit_iterm2(
    output: &mut impl Write,
    png: &[u8],
    columns: u16,
    rows: u16,
    tmux: bool,
) -> io::Result<()> {
    let encoded = base64::engine::general_purpose::STANDARD.encode(png);
    let mut sequence = Vec::with_capacity(encoded.len() + 160);
    write!(
        sequence,
        "\x1b]1337;File=inline=1;preserveAspectRatio=1;size={};width={columns};height={rows}:{}\x07",
        png.len(),
        encoded
    )?;
    emit_terminal_sequence(output, &sequence, tmux)?;
    output.write_all(b"\r\n")
}

fn emit_ansi_raster(output: &mut impl Write) -> io::Result<()> {
    debug_assert_eq!(
        ANSI_HEAD_RGBA.len(),
        usize::from(ANSI_LOGO_COLUMNS) * usize::from(ANSI_LOGO_PIXEL_ROWS) * 4
    );
    for y in (0..ANSI_LOGO_PIXEL_ROWS).step_by(2) {
        output.write_all(b"  ")?;
        for x in 0..ANSI_LOGO_COLUMNS {
            let top = ansi_pixel(x, y);
            let bottom = ansi_pixel(x, y + 1);
            write_half_block(output, top, bottom)?;
        }
        output.write_all(b"\x1b[0m\r\n")?;
    }
    Ok(())
}

fn ansi_pixel(x: u16, y: u16) -> [u8; 4] {
    let offset = (usize::from(y) * usize::from(ANSI_LOGO_COLUMNS) + usize::from(x)) * 4;
    let premultiplied: [u8; 4] = ANSI_HEAD_RGBA[offset..offset + 4]
        .try_into()
        .expect("compiled ANSI pixel is RGBA");
    let alpha = premultiplied[3];
    if alpha == 0 {
        return [0, 0, 0, 0];
    }
    let unpremultiply = |channel: u8| {
        ((u32::from(channel) * 255 + u32::from(alpha) / 2) / u32::from(alpha)).min(255) as u8
    };
    [
        unpremultiply(premultiplied[0]),
        unpremultiply(premultiplied[1]),
        unpremultiply(premultiplied[2]),
        alpha,
    ]
}

fn write_half_block(output: &mut impl Write, top: [u8; 4], bottom: [u8; 4]) -> io::Result<()> {
    const VISIBLE_ALPHA: u8 = 32;
    match (top[3] >= VISIBLE_ALPHA, bottom[3] >= VISIBLE_ALPHA) {
        (false, false) => output.write_all(b"\x1b[0m "),
        (true, false) => write!(
            output,
            "\x1b[49m\x1b[38;2;{};{};{}m▀",
            top[0], top[1], top[2]
        ),
        (false, true) => write!(
            output,
            "\x1b[49m\x1b[38;2;{};{};{}m▄",
            bottom[0], bottom[1], bottom[2]
        ),
        (true, true) => write!(
            output,
            "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀",
            top[0], top[1], top[2], bottom[0], bottom[1], bottom[2]
        ),
    }
}

fn emit_terminal_sequence(output: &mut impl Write, sequence: &[u8], tmux: bool) -> io::Result<()> {
    if !tmux {
        return output.write_all(sequence);
    }
    output.write_all(b"\x1bPtmux;")?;
    for byte in sequence {
        if *byte == 0x1b {
            output.write_all(b"\x1b")?;
        }
        output.write_all(std::slice::from_ref(byte))?;
    }
    output.write_all(b"\x1b\\")
}

fn styled(value: &str, style: Style, enabled: bool) -> String {
    if enabled {
        style.apply_to(value).to_string()
    } else {
        value.to_owned()
    }
}

fn nonempty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn contains_folded(value: &str, needle: &str) -> bool {
    value.to_ascii_lowercase().contains(needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use sha2::{Digest, Sha256};

    #[test]
    fn embedded_svg_is_the_exact_hf2q_us_head_asset() {
        assert_eq!(exact_head_svg().len(), 1_387);
        assert_eq!(
            hex::encode(Sha256::digest(exact_head_svg())),
            "645f8a42049a9a1fd7074a98568c35ec0da947d2e2e997151a1d88c8ce9f2c4c"
        );
    }

    #[test]
    fn exact_svg_compiles_to_a_nonempty_native_size_png() {
        assert_eq!(
            hex::encode(Sha256::digest(HEAD_PNG_FILE)),
            "fe8cc15cc2693c38ab8510724566a22455b2d33bc7332229deedb88bc5e28aad"
        );
        assert_eq!(
            hex::encode(Sha256::digest(ANSI_HEAD_RGBA)),
            "b639f60b2ea65c985e312fc3d8a62519427e8ef4f88107ef0aec26e419c57538"
        );
        assert!(HEAD_PNG_FILE.starts_with(b"\x89PNG\r\n\x1a\n"));
        let image = image::load_from_memory(HEAD_PNG_FILE)
            .expect("decode compiled PNG")
            .to_rgba8();
        assert_eq!((image.width(), image.height()), (224, 315));
        assert_eq!(ANSI_HEAD_RGBA.len(), 20 * 28 * 4);
        // The authoritative left ear begins at SVG coordinate (0, 0), so
        // both native and ANSI rasters must preserve its antialiased edge.
        assert_eq!(image.get_pixel(0, 0).0, [0xa3, 0xa3, 0xa4, 0xff]);
        assert_eq!(&ANSI_HEAD_RGBA[..4], &[0xa3, 0xa3, 0xa4, 0xff]);
        assert!(image
            .pixels()
            .any(|pixel| pixel.0 == [0xff, 0xff, 0xff, 0xff]));
    }

    #[test]
    fn auto_selects_only_supported_non_multiplexed_protocols() {
        assert_eq!(
            detect_protocol(&TerminalFacts::interactive(Some("ghostty"))),
            Some(GraphicsProtocol::Kitty)
        );
        assert_eq!(
            detect_protocol(&TerminalFacts::interactive(Some("WezTerm"))),
            Some(GraphicsProtocol::Iterm2)
        );
        assert_eq!(
            detect_protocol(&TerminalFacts::interactive(Some("Apple_Terminal"))),
            Some(GraphicsProtocol::Ansi)
        );
        let mut tmux = TerminalFacts::interactive(Some("WezTerm"));
        tmux.tmux = true;
        assert_eq!(detect_protocol(&tmux), Some(GraphicsProtocol::Ansi));

        let mut cmux = TerminalFacts::interactive(Some("Apple_Terminal"));
        cmux.cmux = true;
        assert_eq!(detect_protocol(&cmux), Some(GraphicsProtocol::Kitty));
        let mut kitty = TerminalFacts::interactive(Some("Apple_Terminal"));
        kitty.kitty_window_id = true;
        assert_eq!(detect_protocol(&kitty), Some(GraphicsProtocol::Kitty));
        let mut ghostty = TerminalFacts::interactive(Some("Apple_Terminal"));
        ghostty.ghostty = true;
        assert_eq!(detect_protocol(&ghostty), Some(GraphicsProtocol::Kitty));
        assert_eq!(
            detect_protocol(&TerminalFacts::interactive(Some("Alacritty"))),
            Some(GraphicsProtocol::Ansi)
        );
    }

    #[test]
    fn kitty_and_iterm_frames_carry_exact_raster_without_alt_screen() {
        let mut kitty = Vec::new();
        emit_kitty(&mut kitty, HEAD_PNG_FILE, LOGO_COLUMNS, LOGO_ROWS, false).unwrap();
        assert!(kitty.starts_with(b"\x1b_Ga=T,f=100,c=14,r=10,C=1,q=2,m=1;"));
        assert!(kitty.ends_with(b"\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n\r\n"));
        assert!(!kitty.windows(8).any(|part| part == b"\x1b[?1049h"));

        let mut iterm = Vec::new();
        emit_iterm2(&mut iterm, HEAD_PNG_FILE, LOGO_COLUMNS, LOGO_ROWS, false).unwrap();
        assert!(iterm.starts_with(b"\x1b]1337;File=inline=1;preserveAspectRatio=1;"));
        assert!(iterm.ends_with(b"\x07\r\n"));
        assert!(!iterm.windows(8).any(|part| part == b"\x1b[?1049h"));
    }

    #[test]
    fn apple_terminal_gets_source_derived_ansi_raster_not_fake_rabbit_art() {
        let cli = Cli::parse_from(["hf2q", "doctor"]);
        let facts = TerminalFacts::interactive(Some("Apple_Terminal"));
        let mut output = Vec::new();
        render_banner(&mut output, cli.terminal_graphics, &facts).unwrap();
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("hf2q"));
        assert!(rendered.contains("Hugging Face → native GGUF"));
        assert!(!rendered.contains("╲"));
        assert!(!rendered.contains("◢"));
        assert!(rendered.contains("\x1b[38;2;"));
        assert!(rendered.contains('▀') || rendered.contains('▄'));
        assert!(!rendered.contains("\x1b[?1049h"));
    }

    #[test]
    fn ansi_backend_is_bounded_and_uses_only_pixel_blocks() {
        let mut output = Vec::new();
        emit_ansi_raster(&mut output).unwrap();
        let rendered = String::from_utf8(output).unwrap();
        assert_eq!(rendered.matches("\r\n").count(), 14);
        assert!(rendered.contains('▀') || rendered.contains('▄'));
        assert!(!rendered.contains("╲"));
        assert!(!rendered.contains("◢"));
        assert!(!rendered.contains("\x1b[?1049h"));
    }

    #[test]
    fn structured_quiet_protocol_and_non_tty_paths_suppress_banner() {
        let mut facts = TerminalFacts::interactive(Some("WezTerm"));
        let doctor = Cli::parse_from(["hf2q", "doctor"]);
        assert!(banner_enabled(&doctor, &facts));

        facts.stderr_tty = false;
        assert!(!banner_enabled(&doctor, &facts));
        facts.stderr_tty = true;

        facts.stdout_tty = false;
        assert!(!banner_enabled(&doctor, &facts));
        facts.stdout_tty = true;

        let json = Cli::parse_from(["hf2q", "--log-format", "json", "doctor"]);
        assert!(!banner_enabled(&json, &facts));
        let quiet = Cli::parse_from(["hf2q", "serve", "--quiet"]);
        assert!(!banner_enabled(&quiet, &facts));
        let completions = Cli::parse_from(["hf2q", "completions", "--shell", "bash"]);
        assert!(!banner_enabled(&completions, &facts));
        let off = Cli::parse_from(["hf2q", "--terminal-graphics", "off", "doctor"]);
        assert!(!banner_enabled(&off, &facts));
        let hidden = Cli::parse_from([
            "hf2q",
            "source-teacher",
            "--model-dir",
            "/tmp/model",
            "--output",
            "/tmp/target",
            "--evaluation-split",
            "calibration",
        ]);
        assert!(!banner_enabled(&hidden, &facts));
    }
}
