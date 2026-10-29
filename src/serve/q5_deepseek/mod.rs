//! Connected ADR-063 selected DeepSeek diagnostic entry, default off.
//! Legacy packets refuse; a private reviewed quantitative gate may authorize
//! one capped operating attempt, never physical or Q5 qualification.
pub(crate) mod collector;
pub(crate) mod manifest;
mod packet;
mod page_observer;
mod receipt;
mod requirements;
mod startup;
pub(crate) mod selected_profile;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tiny_control;

use anyhow::{ensure, Context, Result};
use mlx_native::gguf::GgufFile;
use packet::{Packet, Phase};
use std::{fs::File, io::Read, path::Path};

/// Private real-load carrier. No serde/external constructor or permit boolean.
/// Present only after the private independently reviewed operating validator.
pub(crate) struct PreparedLoad {
    header: GgufFile,
    context: collector::Handle,
}

mod reviewed_budget;
pub(crate) mod operating_case;
mod operating_os;
use reviewed_budget::ReviewedSourceBudget;
impl PreparedLoad {
    fn after_review(
        header: GgufFile,
        context: collector::Handle,
        budget: ReviewedSourceBudget,
    ) -> Result<Self> {
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
        let (gguf, _scope, startup) = observe_header(&packet, model)?;
        // Dry preflight never installs native authority or executes a real-load
        // callback, even when a later source-budget validator is implemented.
        if packet.operating_case.is_some() {
            reviewed_budget::dry_validate(&packet,&gguf,&startup)?;
            return Ok(()); // No token, native installation or model callback.
        }
        return reject_preload_with_startup(&packet, receipt::HeaderPlan::actual(&gguf), startup);
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
    opts: &crate::serve::api::engine::LoadOptions,
    effective_context: Option<u32>,
    normal: impl FnOnce(Option<PreparedLoad>) -> Result<T>,
) -> Result<T> {
    match selected()? {
        None => normal(None), // Do not inspect options/env or change normal load.
        Some(path) => {
            startup::check_origin(opts.tokenizer_path.is_some())?;
            let profile=selected_profile::SelectedLoadProfile::from_actual(opts,effective_context)?;
            let packet=read_packet(path)?;
            enter_with_profile(&packet,&opts.model_path,Some(profile),normal)
        }
    }
}

#[cfg(test)]
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
    diagnostic_open(path)?
        .take(MAX_PACKET + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= MAX_PACKET, "Q5 packet exceeds 64KiB");
    serde_json::from_slice(&bytes).context("closed Q5 packet")
}

#[cfg(test)]
fn enter<T>(
    packet: &Packet,
    path: &Path,
    real_load: impl FnOnce(Option<PreparedLoad>) -> Result<T>,
) -> Result<T> {
    enter_with_profile(packet,path,None,real_load)
}

fn enter_with_profile<T>(
    packet: &Packet,
    path: &Path,
    profile: Option<selected_profile::SelectedLoadProfile>,
    real_load: impl FnOnce(Option<PreparedLoad>) -> Result<T>,
) -> Result<T> {
    let (gguf, scope, startup) = observe_header(packet, path)?;
    if let Some(profile)=profile { collector::bind_selected_load_profile(profile)?; }
    let observed_header_metadata = scope.handle().summary()?;
    let compiled_source = option_env!("HF2Q_BUILD_GIT_SHA").unwrap_or("Unknown");
    let compiled_native_pin = env!("HF2Q_MLX_NATIVE_VERSION");
    let pid = std::process::id();
    let budget: ReviewedSourceBudget = if packet.operating_case.is_some() {
        ensure!(profile.is_some(), "Q5 operating case requires actual typed load profile");
        reviewed_budget::validate(packet,&gguf,&startup)?
    } else {
        reject_preload_with_startup(packet, receipt::HeaderPlan::actual(&gguf), startup).with_context(|| format!("Q5 same-FD header observation (events,checked_results,checkpoint_roots,retained_fd)={observed_header_metadata:?}; compiled_metadata_source={compiled_source}; compiled_native_pin={compiled_native_pin}; pid={pid}; declared_hashes_unverified; not runtime coverage"))?
    };
    // The selected operating validator authorizes one capped attempt, not success.
    // Install exactly once BEFORE target Device; failure never retries/replays.
    let prepared = PreparedLoad::after_review(gguf, scope.handle(), budget)?;
    real_load(Some(prepared))
}

fn observe_header(
    packet: &Packet,
    path: &Path,
) -> Result<(GgufFile, collector::HeaderScope, startup::Facts)> {
    packet.validate()?;
    let _collector = collector::HeaderScope::enter()?;
    let file = diagnostic_open(path).context("open Q5 actual readonly regular GGUF FD")?;
    let before = file.metadata()?;
    ensure!(before.is_file(), "Q5 model FD is not a regular file");
    packet.file.matches(&before)?;
    let gguf = GgufFile::from_file_with_limits(file.try_clone()?, startup::limits())
        .context("Q5 bounded same-FD header")?;
    ensure!(
        gguf.metadata_string("general.architecture") == Some("deepseek4"),
        "Q5 selected DeepSeek architecture mismatch"
    );
    packet.file.matches(&file.metadata()?)?;
    collector::attach_source(&file, packet.file, gguf.tensor_data_offset())?;
    let startup = startup::inspect(&gguf)?;
    Ok((gguf, _collector, startup))
}

#[cfg(test)]
fn reject_preload<T>(packet: &Packet, actual_header_plan: receipt::HeaderPlan) -> Result<T> {
    reject_preload_with_startup(
        packet,
        actual_header_plan,
        startup::Facts::Unknown {
            reason: "HostControlWithoutStartupHeader",
        },
    )
}

fn reject_preload_with_startup<T>(
    packet: &Packet,
    actual_header_plan: receipt::HeaderPlan,
    startup: startup::Facts,
) -> Result<T> {
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
        startup,
        actual_load_profile: collector::selected_load_profile()?,
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

/// Selected surrounding metadata reads use the SAME bounded parser. Normal
/// loading retains its original parser and behavior; no device/mapping occurs.
pub(crate) fn metadata_header(path:&Path)->Result<GgufFile>{
    if selected()?.is_none(){return Ok(GgufFile::open(path)?);}
    let result=(||{let packet=read_packet(selected()?.context("Q5 selected metadata source lost")?)?;
        let file=diagnostic_open(path)?;packet.file.matches(&file.metadata()?)?;
        let g=GgufFile::from_file_with_limits(file.try_clone()?,startup::limits())?;
        packet.file.matches(&file.metadata()?)?;Ok(g)})();
    if result.is_err(){operating_case::record_failure();}result
}

/// Dry preflight eligibility is never a direct native model-load authority.
pub(crate) fn require_prepared_model_load()->Result<()>{
    if selected()?.is_some(){ensure!(operating_case::is_active(),"Q5 direct model load requires the private operating carrier");collector::require_prepared_before_device()?;}
    Ok(())
}


/// Must run before MODEL parsing/resolution or any persistence/projector work.
pub(crate) fn preflight_serve_args(args: &crate::cli::ServeArgs) -> Result<()> {
    check_serve_args(selected()?.is_some(), args)
}
fn check_serve_args(active: bool, args: &crate::cli::ServeArgs) -> Result<()> {
    if !active {
        return Ok(());
    }
    ensure!(
        args.tokenizer.is_none()
            && args.config.is_none()
            && args.dwq_overlay.is_none()
            && args.glp.is_none()
            && args.glp_alpha.is_none()
            && args.kv_graft.is_none()
            && args.embedding_model.is_none()
            && args.mmproj.is_none()
            && args.planning.kv_persist_path.is_none()
            && args.planning.kv_persist_budget.is_none()
            && args.output.is_none(),
        "Q5 selected startup preparation option unsupported"
    );
    if let Some(target) = args.target.as_deref() {
        ensure!(
            matches!(
                crate::model_spec::parse_model_spec(target)?,
                crate::model_spec::ModelSpec::Path(_)
            ),
            "Q5 selected startup requires local MODEL, not inventory/repository resolution"
        );
    } else {
        let path = args
            .model
            .as_deref()
            .context("Q5 selected startup requires explicit local MODEL")?;
        ensure!(
            !matches!(
                crate::model_spec::parse_model_spec(&path.to_string_lossy())?,
                crate::model_spec::ModelSpec::Repository(_) | crate::model_spec::ModelSpec::List
            ),
            "Q5 selected startup forbids repository/inventory resolution"
        );
    }
    Ok(())
}
pub(crate) fn preflight_engine_config(
    config: &crate::serve::multi_model::EngineConfig,
) -> Result<()> {
    if selected()?.is_none() {
        return Ok(());
    }
    ensure!(
        config.tokenizer_path.is_none()
            && config.config_path.is_none()
            && config.dwq_overlay_path.is_none()
            && config.glp_reference.is_none()
            && config.glp_alpha.is_none()
            && config.kv_graft_path.is_none()
            && config.kv_persist_dir.is_none()
            && config.kv_persist_budget_bytes == 0,
        "Q5 selected effective engine preparation options unsupported"
    );
    Ok(())
}
pub(crate) fn allows_automatic_projector() -> Result<bool> {
    Ok(selected()?.is_none())
}

#[cfg(test)]
mod early_serve_tests {
    use clap::Parser;
    fn args() -> crate::cli::ServeArgs {
        let cli = crate::cli::Cli::try_parse_from([
            "hf2q",
            "serve",
            "--model",
            "/definitely-missing-q5/model.gguf",
        ])
        .unwrap();
        match cli.command {
            crate::cli::Command::Serve(args) => args,
            _ => panic!("serve"),
        }
    }
    #[test]
    fn unsupported_preparation_refuses_before_missing_model_or_side_effects() {
        let mut a = args();
        // Local missing path is NOT opened by this pure early boundary.
        super::check_serve_args(true, &a).unwrap();
        for i in 0..9 {
            match i {
                0 => a.tokenizer = Some("/missing/tokenizer".into()),
                1 => a.config = Some("/missing/config".into()),
                2 => a.dwq_overlay = Some("/missing/dwq".into()),
                3 => a.glp = Some("/missing/glp".into()),
                4 => a.kv_graft = Some("/missing/graft".into()),
                5 => a.embedding_model = Some("/missing/embed".into()),
                6 => a.mmproj = Some("/missing/projector".into()),
                7 => a.planning.kv_persist_path = Some("/missing/persist".into()),
                _ => a.target = Some("other/repository".into()),
            }
            let side_effect = std::cell::Cell::new(false);
            let result = super::check_serve_args(true, &a).map(|_| side_effect.set(true));
            assert!(result.is_err());
            assert!(!side_effect.get());
            super::check_serve_args(false, &a).unwrap();
            a = args();
        }
    }
}
