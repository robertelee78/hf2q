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
//! profile, the tensor-only families it needs. The answers come from
//! mlx-native's device-scoped queries (`tensor_kernel_family_available`),
//! asked once per process by [`probe`]. Two consumers:
//!
//! - a model load calls [`require_explicit`]: a switch the operator set that
//!   this host cannot run fails the load with the switch and the reason;
//! - the admission points of the tensor-only routes ask [`runs`], so a
//!   switch that is on for any other reason (a route profile) leaves its
//!   route off on such a host and the stock route serves.
//!
//! A probe that fails for any reason other than a missing tensor API is an
//! error, as it is in mlx-native: a shader or toolchain regression on a
//! Metal 4 device must not look like an old Mac.

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

/// Every fast-route switch and the tensor-only kernel families its route
/// needs. An empty list is a route of portable kernels, or one that falls
/// back by itself where a kernel is unavailable.
#[rustfmt::skip]
pub(crate) const SWITCHES: &[(&str, &[TensorKernelFamily])] = &[
    // The AF4G64 register tile: portable simdgroup kernels.
    ("HF2Q_DFLASH2_DRAFT_AF4_G64",      &[]),
    // Staged projections, ring attention, and the prepared head tile.
    ("HF2Q_DFLASH2_DRAFT_STAGED",       &[Dflash2Af4g64Staged, Dflash2RingAttention, RepackedGgufMm]),
    // Token gather and selection kernels: portable.
    ("HF2Q_DFLASH2_GPU_TARGET_TOKENS",  &[]),
    ("HF2Q_REPACKED_GGUF_M32",          &[RepackedGgufMm]),
    ("HF2Q_REPACKED_GGUF_M32_PROJ",     &[RepackedGgufMm]),
    ("HF2Q_REPACKED_GGUF_M32_HEAD",     &[RepackedGgufMm]),
    // Modifiers of the prepared routes: they select nothing else.
    ("HF2Q_REPACKED_GGUF_ANY_WIDTH",    &[RepackedGgufMm]),
    ("HF2Q_REPACKED_GGUF_SHARED_HEAD",  &[RepackedGgufMm]),
    // The tile probes its pipelines per layer and keeps the per-row chain.
    ("HF2Q_QWEN_VERIFY_ATTN_TILE",      &[]),
    ("HF2Q_QWEN_PREFILL_PREPARED",      &[RepackedGgufMm]),
    // Folds only inside a routed prepared prefill; two forwards otherwise.
    ("HF2Q_QWEN_PREFILL_FOLD_CUE",      &[]),
    // A portable fused core; its wave follows the prepared routes.
    ("HF2Q_QWEN_VERIFY_MIXER_FUSED",    &[]),
    // Host-side storage and scheduling.
    ("HF2Q_QWEN_VERIFY_REPLAY_REUSE",   &[]),
    ("HF2Q_DFLASH2_CONTEXT_FIRST",      &[]),
    // Row-exact matvec, top-k and reduce kernels: portable.
    ("HF2Q_QWEN_MOE_BATCHED",           &[]),
    ("HF2Q_QWEN_MOE_STAGED_M8",         &[GroupedM8Experts]),
    ("HF2Q_QWEN_MOE_LIVE_TILES",        &[GroupedM8Experts, MoeExpertTiles]),
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
                mlx_native::TENSOR_API_UNAVAILABLE
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
        let (_, families) = SWITCHES.iter().find(|(switch, _)| *switch == name)?;
        Some(
            families
                .iter()
                .find(|family| !self.runs(**family))
                .map_or(HostRoute::Runs, |family| HostRoute::Unavailable(*family)),
        )
    }

    /// Fail if a switch that `env` turns on cannot run on this device.
    pub(crate) fn require_explicit(&self, env: &dyn Fn(&str) -> Option<String>) -> Result<()> {
        for (name, _) in SWITCHES {
            let on = env(name).is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "on" | "true"
                )
            });
            if let (true, Some(route @ HostRoute::Unavailable(_))) = (on, self.switch(name)) {
                bail!("{}", self.explicit_refusal(name, route));
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
                .filter(|(name, _)| self.switch(name) != Some(HostRoute::Runs))
                .map(|(name, _)| *name)
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{}: {}; these switches cannot run and stay off: {off}",
                self.device,
                mlx_native::TENSOR_API_UNAVAILABLE
            )
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

/// [`probe`], then fail if a switch the process environment turns on cannot
/// run on `device`. Called where a model's weights are uploaded, before the
/// upload is paid for.
pub(crate) fn require_explicit(device: &MlxDevice) -> Result<()> {
    probe(device)?.require_explicit(&|name| std::env::var(name).ok())
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
