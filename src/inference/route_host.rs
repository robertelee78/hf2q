//! Which fast routes this host's GPU can run.
//!
//! Most fast-route kernels are portable Metal and run on every Mac hf2q
//! supports, or sit behind an mlx-native probe that takes a portable kernel
//! by itself (the dense, quantized and expert matmuls, the tiled verify
//! attention). A few kernel families exist only on the Metal tensor API and
//! have no portable twin: the prepared GGUF matmul, the staged draft tiles,
//! the prepared expert planes and the MoE live tiles. The tensor API is
//! absent in two cases: the runtime compiler has no `<metal_tensor>` header
//! (macOS before 26), or the GPU is outside the Metal 4 family (virtual
//! machines; pipeline creation fails there). On such a host a switch that
//! selects one of those families cannot run.
//!
//! [`SWITCHES`] names, for every fast-route switch of the Qwen DFlash2 route
//! profile, how its reader decides "on" and the tensor-only families its
//! route needs. The answers come from mlx-native's device-scoped queries
//! (`tensor_kernel_family_available`), asked once per process by [`probe`].
//! Two consumers:
//!
//! - a model load calls [`require_explicit`]: a tensor-only switch the
//!   operator set that this host cannot run fails the load with the switch
//!   and the reason. A load that selects no such switch never fails here,
//!   whatever the probe says;
//! - the admission points of the prepared GGUF matmul ask [`runs`], so a
//!   prepared-route switch that is on for any other reason (a route profile)
//!   leaves its route off on such a host and the stock route serves. The
//!   staged draft and the prepared MoE expert routes have no such point:
//!   selected on such a host they fail the load with mlx-native's typed
//!   error, and whoever selects them asks [`HostRoutes::switch`] first.
//!
//! A probe that fails for any reason other than a missing tensor API is an
//! error in mlx-native: a shader or toolchain regression on a Metal 4 device
//! must not look like an old Mac. Here it fails the load of a model that
//! selects a tensor-only switch, is logged for one that selects none, and
//! is returned to a caller of [`probe`] to decide.

use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use mlx_native::{tensor_kernel_family_available, KernelRegistry, MlxDevice, TensorKernelFamily};

use TensorKernelFamily::{
    Dflash2Af4g64Staged, Dflash2RingAttention, GroupedM8Experts, MoeExpertTiles, RepackedGgufMm,
};

/// The families [`probe`] asks: those a switch of [`SWITCHES`] or another
/// admission point of this crate ([`runs`]) depends on.
const FAMILIES: [TensorKernelFamily; 6] = [
    RepackedGgufMm,
    Dflash2Af4g64Staged,
    Dflash2RingAttention,
    GroupedM8Experts,
    MoeExpertTiles,
    TensorKernelFamily::MmPerm021,
];

/// How a switch's reader decides that the switch is on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Grammar {
    /// Exactly `1`; any other value leaves the route off (or is refused by
    /// the reader).
    One,
    /// `1`, `on` or `true`, trimmed and in any case.
    OnOffTrue,
}

impl Grammar {
    /// Whether `value` turns a switch of this grammar on.
    pub(crate) fn is_on(self, value: &str) -> bool {
        match self {
            Self::One => value == "1",
            Self::OnOffTrue => matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "on" | "true"
            ),
        }
    }
}

/// One fast-route switch.
pub(crate) struct SwitchRow {
    pub(crate) name: &'static str,
    /// The grammar of the switch's own reader.
    pub(crate) grammar: Grammar,
    /// Tensor-only kernel families the route needs. Empty for a route of
    /// portable kernels and for one that falls back by itself where a
    /// kernel is unavailable.
    pub(crate) families: &'static [TensorKernelFamily],
}

const fn row(
    name: &'static str,
    grammar: Grammar,
    families: &'static [TensorKernelFamily],
) -> SwitchRow {
    SwitchRow {
        name,
        grammar,
        families,
    }
}

use Grammar::{OnOffTrue, One};

/// Every fast-route switch of the Qwen DFlash2 route profile.
#[rustfmt::skip]
pub(crate) const SWITCHES: &[SwitchRow] = &[
    // The AF4G64 register tile: portable simdgroup kernels.
    row("HF2Q_DFLASH2_DRAFT_AF4_G64",     One,       &[]),
    // Staged projections, ring attention, and the prepared head tile.
    row("HF2Q_DFLASH2_DRAFT_STAGED",      One,       &[Dflash2Af4g64Staged, Dflash2RingAttention, RepackedGgufMm]),
    // Token gather and selection kernels: portable.
    row("HF2Q_DFLASH2_GPU_TARGET_TOKENS", OnOffTrue, &[]),
    row("HF2Q_REPACKED_GGUF_M32",         One,       &[RepackedGgufMm]),
    row("HF2Q_REPACKED_GGUF_M32_PROJ",    One,       &[RepackedGgufMm]),
    row("HF2Q_REPACKED_GGUF_M32_HEAD",    One,       &[RepackedGgufMm]),
    // Modifiers of the prepared routes: they select nothing else.
    row("HF2Q_REPACKED_GGUF_ANY_WIDTH",   One,       &[RepackedGgufMm]),
    row("HF2Q_REPACKED_GGUF_SHARED_HEAD", One,       &[RepackedGgufMm]),
    // The tile probes its pipelines per layer and keeps the per-row chain.
    row("HF2Q_QWEN_VERIFY_ATTN_TILE",     One,       &[]),
    row("HF2Q_QWEN_PREFILL_PREPARED",     OnOffTrue, &[RepackedGgufMm]),
    // Folds only inside a routed prepared prefill; two forwards otherwise.
    row("HF2Q_QWEN_PREFILL_FOLD_CUE",     OnOffTrue, &[]),
    // A portable fused core; its wave follows the prepared routes.
    row("HF2Q_QWEN_VERIFY_MIXER_FUSED",   One,       &[]),
    // Host-side storage and scheduling.
    row("HF2Q_QWEN_VERIFY_REPLAY_REUSE",  One,       &[]),
    row("HF2Q_DFLASH2_CONTEXT_FIRST",     One,       &[]),
    // Row-exact matvec, top-k and reduce kernels: portable.
    row("HF2Q_QWEN_MOE_BATCHED",          OnOffTrue, &[]),
    row("HF2Q_QWEN_MOE_STAGED_M8",        OnOffTrue, &[GroupedM8Experts]),
    row("HF2Q_QWEN_MOE_LIVE_TILES",       OnOffTrue, &[GroupedM8Experts, MoeExpertTiles]),
];

/// Whether this host can run a switch's route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HostRoute {
    /// Every kernel of the route runs here, or the route falls back kernel
    /// by kernel.
    Runs,
    /// The route needs this tensor-only kernel family, which the host
    /// cannot run.
    Unavailable(TensorKernelFamily),
}

impl HostRoute {
    /// Why the route cannot run, for an operator; `None` when it runs.
    pub(crate) fn reason(self) -> Option<String> {
        match self {
            Self::Runs => None,
            Self::Unavailable(family) => Some(format!(
                "{} cannot run on this host: {}",
                family.name(),
                mlx_native::tensor_kernel_family_unavailable_reason()
            )),
        }
    }
}

/// The answers of one Metal device.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HostRoutes {
    device: String,
    registry_id: u64,
    /// The families of [`FAMILIES`] this device cannot run.
    unavailable: Vec<TensorKernelFamily>,
}

impl HostRoutes {
    /// Ask mlx-native about every family on `device`. `registry` caches the
    /// answers and keeps the probe pipelines it built.
    pub(crate) fn ask(device: &MlxDevice, registry: &mut KernelRegistry) -> Result<Self> {
        let mut unavailable = Vec::new();
        for family in FAMILIES {
            let available = tensor_kernel_family_available(registry, device, family)
                .with_context(|| format!("probe {} on {}", family.name(), device.name()))?;
            if !available {
                unavailable.push(family);
            }
        }
        Ok(Self {
            device: device.name(),
            registry_id: device.registry_id(),
            unavailable,
        })
    }

    /// Whether the device runs `family`.
    pub(crate) fn runs(&self, family: TensorKernelFamily) -> bool {
        !self.unavailable.contains(&family)
    }

    /// The answer for the switch `name`; `None` if [`SWITCHES`] has no such
    /// switch.
    pub(crate) fn switch(&self, name: &str) -> Option<HostRoute> {
        SWITCHES
            .iter()
            .find(|row| row.name == name)
            .map(|row| self.route(row))
    }

    fn route(&self, row: &SwitchRow) -> HostRoute {
        row.families
            .iter()
            .find(|family| !self.runs(**family))
            .map_or(HostRoute::Runs, |family| HostRoute::Unavailable(*family))
    }

    /// Fail if a switch that `env` turns on, by its reader's own grammar,
    /// cannot run on this device.
    pub(crate) fn require_explicit(&self, env: &dyn Fn(&str) -> Option<String>) -> Result<()> {
        for row in selected_tensor_only(env) {
            if let route @ HostRoute::Unavailable(_) = self.route(row) {
                bail!("{}", self.explicit_refusal(row.name, route));
            }
        }
        Ok(())
    }

    /// What an operator reads when the switch `name`, which they set,
    /// resolved to `route` on this device.
    pub(crate) fn explicit_refusal(&self, name: &str, route: HostRoute) -> String {
        format!(
            "{name} is on, but {} ({}). Unset {name}: the stock route serves this host.",
            route
                .reason()
                .unwrap_or_else(|| "its route runs on this host".to_owned()),
            self.device
        )
    }

    /// One line for the load log.
    pub(crate) fn summary(&self) -> String {
        if self.unavailable.is_empty() {
            format!("{}: every fast-route kernel family runs", self.device)
        } else {
            let off = SWITCHES
                .iter()
                .filter(|row| self.route(row) != HostRoute::Runs)
                .map(|row| row.name)
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{}: {}; these switches cannot run: {off}",
                self.device,
                mlx_native::tensor_kernel_family_unavailable_reason()
            )
        }
    }
}

/// The tensor-only switches `env` turns on, each by its reader's grammar.
fn selected_tensor_only(
    env: &dyn Fn(&str) -> Option<String>,
) -> impl Iterator<Item = &'static SwitchRow> + '_ {
    SWITCHES.iter().filter(move |row| {
        !row.families.is_empty() && env(row.name).is_some_and(|value| row.grammar.is_on(&value))
    })
}

/// The load-time rule over one probe result: a tensor-only switch `env`
/// turns on must be runnable, and a load that selects none never fails
/// here, also when the probe itself failed.
fn require_selected(
    env: &dyn Fn(&str) -> Option<String>,
    probe: impl FnOnce() -> Result<HostRoutes>,
) -> Result<()> {
    match probe() {
        Ok(host) => host.require_explicit(env),
        Err(error) => {
            let selected = selected_tensor_only(env)
                .map(|row| row.name)
                .collect::<Vec<_>>();
            if selected.is_empty() {
                // No optional route is asked for: the stock route does not
                // depend on this probe.
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    tracing::warn!(
                        error = format!("{error:#}"),
                        "fast-route host capability probe failed; no tensor-only switch is \
                         selected, the stock route is unaffected"
                    );
                });
                Ok(())
            } else {
                Err(error.context(format!(
                    "{} selected, and the kernel families it needs could not be probed",
                    selected.join(", ")
                )))
            }
        }
    }
}

static HOST: OnceLock<HostRoutes> = OnceLock::new();

#[cfg(test)]
thread_local! {
    static TEST_HOST: std::cell::RefCell<Option<HostRoutes>> =
        const { std::cell::RefCell::new(None) };
}

/// The answers a test's thread runs under instead of the process's, until
/// dropped. Joint steps, loads and admissions run on the calling thread.
#[cfg(test)]
pub(crate) struct HostForTest {
    previous: Option<HostRoutes>,
}

#[cfg(test)]
impl HostForTest {
    pub(crate) fn set(host: HostRoutes) -> Self {
        Self {
            previous: TEST_HOST.with(|cell| cell.replace(Some(host))),
        }
    }

    /// A host on which every family of [`FAMILIES`] runs, whatever this
    /// process's device answers: for a test of an admission rule itself.
    pub(crate) fn with_tensor_api() -> Self {
        Self::set(HostRoutes {
            device: "host with the Metal tensor API (test)".to_owned(),
            registry_id: 0,
            unavailable: Vec::new(),
        })
    }

    /// A host without the Metal tensor API: no family of [`FAMILIES`] runs.
    /// Registries this thread builds for a model also compile as that host's
    /// runtime compiler does ([`simulate_in`]).
    pub(crate) fn without_tensor_api() -> Self {
        Self::set(HostRoutes {
            device: "host without the Metal tensor API (test)".to_owned(),
            registry_id: 0,
            unavailable: FAMILIES.to_vec(),
        })
    }
}

#[cfg(test)]
impl Drop for HostForTest {
    fn drop(&mut self) {
        TEST_HOST.with(|cell| *cell.borrow_mut() = self.previous.take());
    }
}

/// Under [`HostForTest::without_tensor_api`], make `registry` build
/// pipelines as a runtime compiler without `<metal_tensor>` does, so the
/// model's own kernels take their portable routes on any test host.
#[cfg(test)]
pub(crate) fn simulate_in(registry: &mut KernelRegistry) -> Result<()> {
    let simulated = TEST_HOST.with(|cell| {
        cell.borrow()
            .as_ref()
            .is_some_and(|host| host.unavailable.len() == FAMILIES.len())
    });
    if simulated {
        registry
            .simulate_missing_tensor_header()
            .context("simulate a runtime compiler without <metal_tensor>")?;
    }
    Ok(())
}

/// This process's answers for `device`, asked on the first call (one probe
/// pipeline per family, in a registry of their own) and kept for the
/// process. An error is not kept.
pub(crate) fn probe(device: &MlxDevice) -> Result<HostRoutes> {
    #[cfg(test)]
    if let Some(host) = TEST_HOST.with(|cell| cell.borrow().clone()) {
        return Ok(host);
    }
    if let Some(host) = HOST.get() {
        if host.registry_id != device.registry_id() {
            // Not the device the process asked first: its answers are its own.
            return HostRoutes::ask(device, &mut KernelRegistry::new());
        }
        return Ok(host.clone());
    }
    let host = HostRoutes::ask(device, &mut KernelRegistry::new())?;
    tracing::info!(host = %host.summary(), "fast-route host capability");
    Ok(HOST.get_or_init(|| host).clone())
}

/// [`probe`], then fail if a tensor-only switch the process environment
/// turns on cannot run on `device`. Called where a model's weights are
/// uploaded, before the upload is paid for. A load that selects no
/// tensor-only switch does not fail here: a probe error is then logged once
/// and the stock route loads.
pub(crate) fn require_explicit(device: &MlxDevice) -> Result<()> {
    require_selected(&|name| std::env::var(name).ok(), || probe(device))
}

/// Whether the tensor-only family `family` may be selected: `false` only
/// when the process has asked its device ([`probe`]) and the device cannot
/// run the family. The admission points of the tensor-only routes ask this
/// on every decision; it reads one initialised cell.
pub(crate) fn runs(family: TensorKernelFamily) -> bool {
    #[cfg(test)]
    if let Some(runs) = TEST_HOST.with(|cell| cell.borrow().as_ref().map(|host| host.runs(family)))
    {
        return runs;
    }
    HOST.get().is_none_or(|host| host.runs(family))
}

#[cfg(test)]
mod tests;
