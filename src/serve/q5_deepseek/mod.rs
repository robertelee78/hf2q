//! Connected ADR-063 selected DeepSeek diagnostic entry, default off.
//! This first slice deliberately cannot mint a model-load permit.
mod packet;
#[cfg(test)]
mod tests;

use anyhow::{ensure, Context, Result};
use mlx_native::gguf::GgufFile;
use std::{fs::File, io::Read, path::Path};
use packet::{Packet, Phase};

const SELECTOR: &str = "HF2Q_DEEPSEEK_OWNER_PACKET";
const MAX_PACKET: u64 = 65_536;

/// Called before serve initializes phase observers, devices or warmups. A
/// direct family loader repeats this check, so bypassing cmd_serve is unsafe
/// neither accidentally nor through a malformed selector.
pub(crate) fn validate_activation() -> Result<()> {
    activation(std::env::var_os(SELECTOR), std::env::var_os("HF2Q_UNSAFE_EXPERIMENTS"))
}

/// Run after local MODEL resolution but before listeners/workers/devices. The
/// closed selected diagnostic does not auto-download or fall through to another
/// family when its selected path is missing or mismatched.
pub(crate) fn preflight_model(model: Option<&Path>) -> Result<()> {
    if std::env::var_os(SELECTOR).is_some() {
        let model = model.context("Q5 packet requires an explicit local MODEL")?;
        return load(model, || anyhow::bail!("Q5 entry unexpectedly lost selector"));
    }
    Ok(())
}

fn activation(path: Option<std::ffi::OsString>, ack: Option<std::ffi::OsString>) -> Result<()> {
    if let Some(path) = path {
        ensure!(!path.is_empty(), "Q5 owner packet selector must not be empty");
        ensure!(ack.as_deref() == Some(std::ffi::OsStr::new("1")),
            "Q5 owner packet requires HF2Q_UNSAFE_EXPERIMENTS=1");
    }
    Ok(())
}

/// The diagnostic branch opens the actual selected file ONCE and constructs
/// its header from a duplicate of THAT FD. The existing real-load callback
/// cannot execute without admission. No standalone map/load is performed.
/// None invokes the original closure, preserving normal constructor behavior.
pub(crate) fn load<T>(model: &Path, normal: impl FnOnce() -> Result<T>) -> Result<T> {
    validate_activation()?;
    match std::env::var_os(SELECTOR) {
        None => normal(),
        Some(path) => {
            let mut bytes = Vec::new();
            File::open(path)?.take(MAX_PACKET + 1).read_to_end(&mut bytes)?;
            ensure!(bytes.len() as u64 <= MAX_PACKET, "Q5 packet exceeds 64KiB");
            let packet: Packet = serde_json::from_slice(&bytes).context("closed Q5 packet")?;
            enter(&packet, model, normal)
        }
    }
}

fn enter<T>(packet: &Packet, path: &Path, _real_load: impl FnOnce() -> Result<T>) -> Result<T> {
    packet.validate()?;
    let file = File::open(path).context("open Q5 actual readonly GGUF FD")?;
    let before = file.metadata()?;
    ensure!(before.is_file(), "Q5 model FD is not a regular file");
    packet.file.matches(&before)?;
    let gguf = GgufFile::from_file(file.try_clone()?).context("Q5 same-FD header")?;
    ensure!(gguf.metadata_string("general.architecture") == Some("deepseek4"),
        "Q5 selected DeepSeek architecture mismatch");
    packet.file.matches(&file.metadata()?)?;
    reject_preload(packet)
}

fn reject_preload<T>(packet: &Packet) -> Result<T> {
    // Fixed retained coalesced envelope. No lower caller-provided estimate,
    // Available/Cached estimate or desired weight-window policy can replace it.
    let requested = 120_783_134_084_u64;
    let positive_operating = 14_092_861_440_u64;
    let upper = requested.checked_add(positive_operating)
        .and_then(|v| v.checked_add(packet.collector_upper_bytes))
        .context("Q5 preload bound overflow")?;
    let missing = packet.requirements(Phase::Preload)?;
    anyhow::bail!("Q5 RejectedPreload: retained_coalesced_upper_bytes={upper}; required={missing:?};         native_page_observer=UnavailableUntilPublishedPin; external_identity_and_operating_review=Unverified;         private_permit=NotMinted; model_loaded=false; physical_credit=false")
}
