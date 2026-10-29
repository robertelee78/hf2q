//! Connected ADR-063 selected DeepSeek diagnostic entry, default off.
//! This first slice deliberately cannot mint a model-load permit.
pub(crate) mod collector;
pub(crate) mod manifest;
mod packet;
mod page_observer;
mod receipt;
mod requirements;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tiny_control;

use anyhow::{ensure, Context, Result};
use mlx_native::gguf::GgufFile;
use packet::{Packet, Phase};
use std::{fs::File, io::Read, path::Path};

/// Private real-load carrier. No serde/external constructor or permit boolean.
/// Present only after a future independently reviewed admission implementation.
pub(crate) struct PreparedLoad {
    header: GgufFile,
    context: collector::Handle,
}

// No constructor/serde exists. Only a future independent budget validator in
// this private module may mint one; packet hashes/booleans cannot construct it.
mod reviewed_budget {
    pub(super) struct ReviewedSourceBudget {
        creation: mlx_native::RootCreationLimits,
        live: mlx_native::LiveRootLimits,
    }
    impl ReviewedSourceBudget {
        pub(super) fn into_limits(self) -> (mlx_native::RootCreationLimits, mlx_native::LiveRootLimits) {
            (self.creation, self.live)
        }
    }
}
use reviewed_budget::ReviewedSourceBudget;
impl PreparedLoad {
    fn after_review(header: GgufFile, context: collector::Handle, budget: ReviewedSourceBudget) -> Result<Self> {
        collector::install_reviewed_native(&context, budget)?;
        Ok(Self { header, context })
    }
    pub(crate) fn into_parts(self) -> (GgufFile, collector::Handle) {
        (self.header, self.context)
    }
}

const MAX_PACKET: u64 = 65_536;

/// Called before serve initializes phase observers, devices or warmups. A
/// direct family loader repeats this check, so bypassing cmd_serve is unsafe
/// neither accidentally nor through a malformed selector.
pub(crate) fn validate_activation() -> Result<()> {
    selected().map(|_| ())
}
fn selected() -> Result<Option<&'static Path>> {
    crate::debug::investigation_env::deepseek_owner_packet::selected().map_err(anyhow::Error::msg)
}

/// Run after local MODEL resolution but before listeners/workers/devices. The
/// closed selected diagnostic does not auto-download or fall through to another
/// family when its selected path is missing or mismatched.
pub(crate) fn preflight_model(model: Option<&Path>) -> Result<()> {
    if selected()?.is_some() {
        let model = model.context("Q5 packet requires an explicit local MODEL")?;
        let path = selected()?.context("Q5 preflight selector changed")?;
        let packet = read_packet(path)?;
        let (gguf, _scope) = observe_header(&packet, model)?;
        // Dry preflight never installs native authority or executes a real-load
        // callback, even when a later source-budget validator is implemented.
        return reject_preload(&packet, receipt::HeaderPlan::actual(&gguf));
    }
    Ok(())
}

#[cfg(test)]
fn activation(path: Option<std::ffi::OsString>, ack: Option<std::ffi::OsString>) -> Result<()> {
    crate::debug::investigation_env::deepseek_owner_packet::Mode::from_os(
        path.as_deref(),
        ack.as_deref() == Some(std::ffi::OsStr::new("1")),
    )
    .validate()
    .map(|_| ())
    .map_err(anyhow::Error::msg)
}

fn diagnostic_open(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    ensure!(
        file.metadata()?.is_file(),
        "Q5 diagnostic input FD must be regular"
    );
    Ok(file)
}

/// The diagnostic branch opens the actual selected file ONCE and constructs
/// its header from a duplicate of THAT FD. The existing real-load callback
/// cannot execute without admission. No standalone map/load is performed.
/// None invokes the original closure, preserving normal constructor behavior.
pub(crate) fn load<T>(
    model: &Path,
    normal: impl FnOnce(Option<PreparedLoad>) -> Result<T>,
) -> Result<T> {
    load_selected(selected()?, model, normal)
}

fn load_selected<T>(
    selected: Option<&Path>,
    model: &Path,
    normal: impl FnOnce(Option<PreparedLoad>) -> Result<T>,
) -> Result<T> {
    match selected {
        None => normal(None),
        Some(path) => {
            let packet = read_packet(path)?;
            enter(&packet, model, normal)
        }
    }
}

fn read_packet(path: &Path) -> Result<Packet> {
    let mut bytes = Vec::new();
    diagnostic_open(path)?.take(MAX_PACKET + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= MAX_PACKET, "Q5 packet exceeds 64KiB");
    serde_json::from_slice(&bytes).context("closed Q5 packet")
}

fn enter<T>(
    packet: &Packet,
    path: &Path,
    real_load: impl FnOnce(Option<PreparedLoad>) -> Result<T>,
) -> Result<T> {
    let (gguf, scope) = observe_header(packet, path)?;
    let observed_header_metadata = scope.handle().summary()?;
    let compiled_source = option_env!("HF2Q_BUILD_GIT_SHA").unwrap_or("Unknown");
    let compiled_native_pin = env!("HF2Q_MLX_NATIVE_VERSION");
    let pid = std::process::id();
    let budget: ReviewedSourceBudget = reject_preload(packet, receipt::HeaderPlan::actual(&gguf)).with_context(|| format!("Q5 same-FD header observation (events,checked_results,checkpoint_roots,retained_fd)={observed_header_metadata:?}; compiled_metadata_source={compiled_source}; compiled_native_pin={compiled_native_pin}; pid={pid}; declared_hashes_unverified; not runtime coverage"))?;
    // Currently unreachable: reject_preload cannot construct ReviewedSourceBudget.
    // If a real validator replaces it, this installs once BEFORE target Device.
    let prepared = PreparedLoad::after_review(gguf, scope.handle(), budget)?;
    real_load(Some(prepared))
}

fn observe_header(packet: &Packet, path: &Path) -> Result<(GgufFile, collector::HeaderScope)> {
    packet.validate()?;
    let _collector = collector::HeaderScope::enter()?;
    let file = diagnostic_open(path).context("open Q5 actual readonly regular GGUF FD")?;
    let before = file.metadata()?;
    ensure!(before.is_file(), "Q5 model FD is not a regular file");
    packet.file.matches(&before)?;
    let gguf = GgufFile::from_file(file.try_clone()?).context("Q5 same-FD header")?;
    ensure!(
        gguf.metadata_string("general.architecture") == Some("deepseek4"),
        "Q5 selected DeepSeek architecture mismatch"
    );
    packet.file.matches(&file.metadata()?)?;
    collector::attach_source(&file, packet.file, gguf.tensor_data_offset())?;
    Ok((gguf, _collector))
}

fn reject_preload<T>(packet: &Packet, actual_header_plan: receipt::HeaderPlan) -> Result<T> {
    // Fixed retained coalesced envelope. No lower caller-provided estimate,
    // Available/Cached estimate or desired weight-window policy can replace it.
    let requested = 120_783_134_084_u64;
    let positive_operating = 14_092_861_440_u64;
    let upper = requested
        .checked_add(positive_operating)
        .and_then(|v| v.checked_add(packet.collector_upper_bytes))
        .context("Q5 preload bound overflow")?;
    let missing = packet.requirements(Phase::Preload)?;
    let root_domains = requirements::describe(Phase::Preload);
    use page_observer::PublishedPageObserver;
    let unavailable = page_observer::Unavailable
        .require_runtime_support()
        .err()
        .context("Q5 unavailable observer unexpectedly claimed support")?;
    let record = receipt::PreloadReceipt {
        schema: "hf2q.q5.deepseek.preload-receipt.v1",
        verdict: "RejectedPreload",
        model_loaded: false,
        runtime_qualified: false,
        physical_credit: false,
        retained_coalesced_upper_bytes: upper,
        source_native_binary_refs: [
            &packet.hf_source_sha256,
            &packet.native_package_sha256,
            &packet.binary_sha256,
        ],
        reference_authority: "CallerDeclared_Unverified_NoPermit",
        native_page_observer: "Public14Available_NoWholeDomainPermit",
        actual_header_plan,
        required_root_domains: root_domains,
    };
    let json =
        String::from_utf8(receipt::encode(&record, 65_536)?).context("Q5 receipt encoding")?;
    anyhow::bail!(
        "Q5 RejectedPreload: {json}; required={missing:?}; {unavailable}; private_permit=NotMinted"
    )
}

#[cfg(test)]
pub(crate) fn actual_header_plan_control(header: &GgufFile) -> Result<(u32, u64)> {
    match receipt::HeaderPlan::actual(header) {
        receipt::HeaderPlan::Parsed {
            layers,
            one_cache_requested_bytes,
            ..
        } => Ok((layers, one_cache_requested_bytes)),
        receipt::HeaderPlan::Unknown { reason } => anyhow::bail!(reason),
    }
}
