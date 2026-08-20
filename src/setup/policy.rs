use std::io::{BufRead, Write};

use crate::cli::{SessionCacheChoice, SetupArgs};

use super::schema::SessionCachePolicyV1;
use super::SetupError;

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
const GIB: u64 = 1024 * MIB;
const TIB: u64 = 1024 * GIB;
const MAX_SESSION_CACHE_BYTES: u64 = 100 * GIB;
const RESERVED_BYTES: u64 = 20 * GIB;
const MAX_PROMPT_BYTES: usize = 256;

pub(super) enum PolicyResolution {
    Cancelled,
    Selected(SessionCachePolicyV1),
}

pub(super) fn recommended_limit(total: u64, available: u64) -> Result<u64, SetupError> {
    if total == 0 || available > total {
        return Err(SetupError::Host(
            "filesystem reported invalid capacity facts".to_owned(),
        ));
    }
    let volume_tenth = total / 10;
    let free_quarter = available / 4;
    let fifteen_percent = (total / 100)
        .checked_mul(15)
        .and_then(|whole| whole.checked_add((total % 100) * 15 / 100))
        .ok_or_else(|| SetupError::Host("filesystem capacity arithmetic overflowed".to_owned()))?;
    let reserve = RESERVED_BYTES.max(fifteen_percent);
    Ok(MAX_SESSION_CACHE_BYTES
        .min(volume_tenth)
        .min(free_quarter)
        .min(available.saturating_sub(reserve)))
}

pub(super) fn resolve_policy<R: BufRead, W: Write>(
    args: &SetupArgs,
    current: Option<&SessionCachePolicyV1>,
    interactive: bool,
    recommendation: u64,
    input: &mut R,
    output: &mut W,
) -> Result<PolicyResolution, SetupError> {
    match (args.session_cache, args.session_cache_limit.as_deref()) {
        (Some(SessionCacheChoice::Off), None) => Ok(PolicyResolution::Selected(disabled())),
        (Some(SessionCacheChoice::Off), Some(_)) => Err(SetupError::Input(
            "--session-cache-limit is invalid when --session-cache off".to_owned(),
        )),
        (Some(SessionCacheChoice::On), Some(value)) => Ok(PolicyResolution::Selected(enabled(
            value,
            recommendation,
            output,
        )?)),
        (Some(SessionCacheChoice::On), None) if interactive => {
            prompt_limit(current, recommendation, input, output)
        }
        (Some(SessionCacheChoice::On), None) => Err(SetupError::Input(
            "non-interactive setup requires --session-cache on --session-cache-limit SIZE"
                .to_owned(),
        )),
        (None, Some(_)) => Err(SetupError::Input(
            "--session-cache-limit requires --session-cache on".to_owned(),
        )),
        (None, None) if interactive => {
            let current_enabled = current.map_or(true, |policy| policy.limit_bytes > 0);
            write!(
                output,
                "Keep inactive sessions on disk for fast resume? {} ",
                if current_enabled { "[Y/n]" } else { "[y/N]" }
            )?;
            output.flush()?;
            let Some(answer) = read_prompt_line(input)? else {
                return Ok(PolicyResolution::Cancelled);
            };
            match answer.trim().to_ascii_lowercase().as_str() {
                "" if !current_enabled => Ok(PolicyResolution::Selected(disabled())),
                "" | "y" | "yes" => prompt_limit(current, recommendation, input, output),
                "n" | "no" => Ok(PolicyResolution::Selected(disabled())),
                _ => Err(SetupError::Input(
                    "expected y, yes, n, no, or Enter".to_owned(),
                )),
            }
        }
        (None, None) => Err(SetupError::Input(
            "non-interactive setup requires --session-cache off or --session-cache on --session-cache-limit SIZE"
                .to_owned(),
        )),
    }
}

fn prompt_limit<R: BufRead, W: Write>(
    current: Option<&SessionCachePolicyV1>,
    recommendation: u64,
    input: &mut R,
    output: &mut W,
) -> Result<PolicyResolution, SetupError> {
    if recommendation == 0 {
        return Err(SetupError::Input(
            "session persistence cannot be enabled: no positive safe disk band remains".to_owned(),
        ));
    }
    let default = current
        .map(|policy| policy.limit_bytes)
        .filter(|limit| *limit > 0)
        .unwrap_or(recommendation);
    write!(output, "Session cache limit [{}]: ", format_bytes(default))?;
    output.flush()?;
    let Some(answer) = read_prompt_line(input)? else {
        return Ok(PolicyResolution::Cancelled);
    };
    let limit = if answer.is_empty() {
        default
    } else {
        parse_byte_size(&answer)?
    };
    validate_enabled_limit(limit, recommendation, output)?;
    Ok(PolicyResolution::Selected(SessionCachePolicyV1 {
        limit_bytes: limit,
    }))
}

fn enabled<W: Write>(
    value: &str,
    recommendation: u64,
    output: &mut W,
) -> Result<SessionCachePolicyV1, SetupError> {
    let limit = parse_byte_size(value)?;
    validate_enabled_limit(limit, recommendation, output)?;
    Ok(SessionCachePolicyV1 { limit_bytes: limit })
}

fn validate_enabled_limit<W: Write>(
    limit: u64,
    recommendation: u64,
    output: &mut W,
) -> Result<(), SetupError> {
    if limit == 0 {
        return Err(SetupError::Input(
            "--session-cache on requires a positive limit; zero means disabled".to_owned(),
        ));
    }
    if recommendation == 0 {
        return Err(SetupError::Input(
            "session persistence cannot be enabled: no positive safe disk band remains".to_owned(),
        ));
    }
    if limit > i64::MAX as u64 {
        return Err(SetupError::Input(
            "session cache limit exceeds TOML's signed integer range".to_owned(),
        ));
    }
    if limit > recommendation {
        writeln!(
            output,
            "Warning: requested {} exceeds the disk-aware recommendation {}.",
            format_bytes(limit),
            format_bytes(recommendation)
        )?;
    }
    Ok(())
}

fn disabled() -> SessionCachePolicyV1 {
    SessionCachePolicyV1 { limit_bytes: 0 }
}

pub(super) fn parse_byte_size(value: &str) -> Result<u64, SetupError> {
    if value.is_empty() || value.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err(SetupError::Input(
            "SIZE must be an integer with optional B, KiB, MiB, GiB, or TiB suffix".to_owned(),
        ));
    }
    let (digits, multiplier) = [
        ("KiB", KIB),
        ("MiB", MIB),
        ("GiB", GIB),
        ("TiB", TIB),
        ("B", 1),
    ]
    .into_iter()
    .find_map(|(suffix, multiplier)| {
        value
            .strip_suffix(suffix)
            .map(|digits| (digits, multiplier))
    })
    .unwrap_or((value, 1));
    if digits.is_empty()
        || (digits.len() > 1 && digits.starts_with('0'))
        || !digits.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(SetupError::Input(
            "SIZE must use canonical unsigned decimal digits".to_owned(),
        ));
    }
    digits
        .parse::<u64>()
        .ok()
        .and_then(|number| number.checked_mul(multiplier))
        .ok_or_else(|| SetupError::Input("SIZE exceeds u64".to_owned()))
}

pub(super) fn format_bytes(bytes: u64) -> String {
    for (suffix, unit) in [("TiB", TIB), ("GiB", GIB), ("MiB", MIB), ("KiB", KIB)] {
        if bytes >= unit && bytes % unit == 0 {
            return format!("{} {suffix}", bytes / unit);
        }
    }
    format!("{bytes} bytes")
}

fn read_prompt_line<R: BufRead>(input: &mut R) -> Result<Option<String>, SetupError> {
    let mut bytes = Vec::new();
    loop {
        let available = match input.fill_buf() {
            Ok(available) if available.is_empty() => {
                if bytes.is_empty() {
                    return Ok(None);
                }
                break;
            }
            Ok(available) => available,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let consumed = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        if bytes.len().saturating_add(consumed) > MAX_PROMPT_BYTES {
            return Err(SetupError::Input(format!(
                "setup prompt input exceeds {MAX_PROMPT_BYTES} bytes"
            )));
        }
        let finished = available[..consumed].ends_with(b"\n");
        bytes.extend_from_slice(&available[..consumed]);
        input.consume(consumed);
        if finished {
            break;
        }
    }
    if bytes.ends_with(b"\n") {
        bytes.pop();
        if bytes.ends_with(b"\r") {
            bytes.pop();
        }
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_| SetupError::Input("setup prompt input must be valid UTF-8".to_owned()))
}
