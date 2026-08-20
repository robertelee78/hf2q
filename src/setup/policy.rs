use std::io::{BufRead, Write};

use crate::cli::{SchedulerArg, SetupArgs};

use super::schema::{ConfiguredScheduler, OperatorConfigV2};
use super::SetupError;

const MAX_PROMPT_BYTES: usize = 256;

pub(super) enum PreferenceResolution {
    Cancelled,
    Selected(OperatorConfigV2),
}

pub(super) fn resolve_preferences<R: BufRead, W: Write>(
    args: &SetupArgs,
    current: Option<&OperatorConfigV2>,
    interactive: bool,
    input: &mut R,
    output: &mut W,
) -> Result<PreferenceResolution, SetupError> {
    if !interactive
        && current.is_none()
        && !args.accept_defaults
        && !fresh_noninteractive_is_complete(args)
    {
        return Err(SetupError::Input(
            "fresh non-interactive setup requires --accept-defaults or explicit values for --default-quant, --serve-host, --serve-port, --serve-scheduler, and (for inflight) --serve-max-slots"
                .to_owned(),
        ));
    }

    let mut selected = current
        .cloned()
        .unwrap_or(OperatorConfigV2::guide_defaults()?);
    apply_explicit(args, &mut selected);

    if !interactive || args.accept_defaults {
        return Ok(PreferenceResolution::Selected(OperatorConfigV2::new(
            selected.convert,
            selected.serve,
        )?));
    }

    if args.default_quant.is_none() {
        write!(
            output,
            "Default conversion quantization [{}]: ",
            selected.convert.quant
        )?;
        output.flush()?;
        let Some(answer) = read_prompt_line(input)? else {
            return Ok(PreferenceResolution::Cancelled);
        };
        if !answer.is_empty() {
            selected.convert.quant = answer;
        }
    }

    if args.serve_scheduler.is_none() {
        write!(
            output,
            "Optimize serving for long agent and tool-use prompts? {} ",
            if selected.serve.scheduler == ConfiguredScheduler::InflightBatched {
                "[Y/n]"
            } else {
                "[y/N]"
            }
        )?;
        output.flush()?;
        let Some(answer) = read_prompt_line(input)? else {
            return Ok(PreferenceResolution::Cancelled);
        };
        selected.serve.scheduler = match yes_no(
            &answer,
            selected.serve.scheduler == ConfiguredScheduler::InflightBatched,
        )? {
            true => ConfiguredScheduler::InflightBatched,
            false => ConfiguredScheduler::FifoSerial,
        };
        if selected.serve.scheduler == ConfiguredScheduler::FifoSerial {
            selected.serve.max_slots = 1;
        }
    }

    if selected.serve.scheduler == ConfiguredScheduler::InflightBatched
        && args.serve_max_slots.is_none()
    {
        write!(
            output,
            "Maximum simultaneous active requests [{}]: ",
            selected.serve.max_slots
        )?;
        output.flush()?;
        let Some(answer) = read_prompt_line(input)? else {
            return Ok(PreferenceResolution::Cancelled);
        };
        if !answer.is_empty() {
            selected.serve.max_slots = parse_positive_u32("active request count", &answer)?;
        }
    }

    if args.serve_host.is_none() {
        write!(
            output,
            "Allow other devices on the LAN to connect? {} ",
            if selected.serve.host == "0.0.0.0" {
                "[Y/n]"
            } else {
                "[y/N]"
            }
        )?;
        output.flush()?;
        let Some(answer) = read_prompt_line(input)? else {
            return Ok(PreferenceResolution::Cancelled);
        };
        selected.serve.host = if yes_no(&answer, selected.serve.host == "0.0.0.0")? {
            "0.0.0.0"
        } else {
            "127.0.0.1"
        }
        .to_owned();
    }

    if args.serve_port.is_none() {
        write!(output, "Default API port [{}]: ", selected.serve.port)?;
        output.flush()?;
        let Some(answer) = read_prompt_line(input)? else {
            return Ok(PreferenceResolution::Cancelled);
        };
        if !answer.is_empty() {
            selected.serve.port = parse_port(&answer)?;
        }
    }

    Ok(PreferenceResolution::Selected(OperatorConfigV2::new(
        selected.convert,
        selected.serve,
    )?))
}

fn fresh_noninteractive_is_complete(args: &SetupArgs) -> bool {
    args.default_quant.is_some()
        && args.serve_host.is_some()
        && args.serve_port.is_some()
        && args.serve_scheduler.is_some()
        && (args.serve_scheduler == Some(SchedulerArg::FifoSerial)
            || args.serve_max_slots.is_some())
}

fn apply_explicit(args: &SetupArgs, selected: &mut OperatorConfigV2) {
    if let Some(quant) = &args.default_quant {
        selected.convert.quant.clone_from(quant);
    }
    if let Some(host) = &args.serve_host {
        selected.serve.host.clone_from(host);
    }
    if let Some(port) = args.serve_port {
        selected.serve.port = port;
    }
    if let Some(scheduler) = args.serve_scheduler {
        selected.serve.scheduler = match scheduler {
            SchedulerArg::FifoSerial => ConfiguredScheduler::FifoSerial,
            SchedulerArg::InflightBatched => ConfiguredScheduler::InflightBatched,
        };
        if scheduler == SchedulerArg::FifoSerial && args.serve_max_slots.is_none() {
            selected.serve.max_slots = 1;
        }
    }
    if let Some(max_slots) = args.serve_max_slots {
        selected.serve.max_slots = max_slots;
    }
}

fn yes_no(answer: &str, default: bool) -> Result<bool, SetupError> {
    match answer.to_ascii_lowercase().as_str() {
        "" => Ok(default),
        "y" | "yes" => Ok(true),
        "n" | "no" => Ok(false),
        _ => Err(SetupError::Input(
            "expected y, yes, n, no, or Enter".to_owned(),
        )),
    }
}

fn parse_positive_u32(field: &str, value: &str) -> Result<u32, SetupError> {
    value
        .parse::<u32>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| SetupError::Input(format!("{field} must be a positive integer")))
}

fn parse_port(value: &str) -> Result<u16, SetupError> {
    value
        .parse::<u16>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| SetupError::Input("API port must be in 1..=65535".to_owned()))
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
        .map(|answer| Some(answer.trim().to_owned()))
        .map_err(|_| SetupError::Input("setup prompt input must be valid UTF-8".to_owned()))
}
