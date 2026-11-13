//! Prompt-prefill route through the prepared Q4_K / Q6_K row-tile MM
//! (`HF2Q_QWEN_PREFILL_PREPARED=1`).
//!
//! The stock GGML matmul runs a chunk of more than eight rows on 128-row
//! tiles and pays a whole tile for every started one, so a 31-row chunk costs
//! what 128 rows cost and a 157-row chunk what 256 cost. The prepared kernel
//! works in 32-row tiles, the last one partial, and its output row depends
//! only on that input row and the weight: not on the chunk size and not on
//! the row's position in the chunk. It is the kernel the width-32 joint
//! target step already runs (`HF2Q_REPACKED_GGUF_M32`,
//! `HF2Q_REPACKED_GGUF_M32_PROJ`), and the prefill route shares that step's
//! load-once planes, so a prefilled row and a decoded row see one kernel.
//!
//! Against the stock route the activation is rounded to BF16 and the
//! accumulation order differs (Q1 Class B). The route applies inside
//! [`PrefillScope`] only: the bounded prompt prefill opens it around its
//! target forward; MTP catch-up, verify windows and decode run outside it
//! and keep their kernels.

use std::cell::Cell;
use std::sync::LazyLock;

use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, MlxBuffer};

use super::joint_step_options::{parse_switch, switch_from_process};

/// Switch, default off: `1`/`on`/`true` routes prompt-prefill chunks through
/// the prepared row-tile MM.
pub(crate) const QWEN_PREFILL_PREPARED_ENV: &str = "HF2Q_QWEN_PREFILL_PREPARED";

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    switch_from_process(QWEN_PREFILL_PREPARED_ENV).unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = false,
            "invalid Qwen prepared-prefill switch"
        );
        false
    })
});

/// Whether the operator switched the prepared prefill route on.
pub(crate) fn enabled() -> bool {
    *ENABLED
}

/// Layers between host waits of a prompt-prefill chunk of more than 32 rows,
/// `1` (default) to `8`.
///
/// The layer loop of a large chunk waits for the GPU after every layer, which
/// bounds how long a Metal fault goes unobserved; the GPU then idles while
/// the host encodes the next layer. With a value above 1 a layer's command
/// buffer is committed without a wait and the host waits once per that many
/// layers, so the GPU runs a layer while the host encodes the next one. A
/// layer whose output the host copies (a DFlash2 target layer, a calibration
/// capture) still waits, and so does the last layer. The dispatches, their
/// order and their operands do not change, so neither does any bit.
pub(crate) const QWEN_PREFILL_WAIT_LAYERS_ENV: &str = "HF2Q_QWEN_PREFILL_WAIT_LAYERS";

/// Longest stretch of layers a large prefill chunk may run without a wait:
/// the cadence chunks of at most 32 rows already have.
pub(crate) const MAX_WAIT_LAYERS: usize = 8;

/// [`QWEN_PREFILL_WAIT_LAYERS_ENV`] as an operator wrote it: unset or blank
/// is 1, anything outside `1..=MAX_WAIT_LAYERS` is an error.
pub(crate) fn parse_wait_layers(raw: Option<&str>) -> Result<usize, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(1),
        Some(value) => value
            .parse::<usize>()
            .ok()
            .filter(|layers| (1..=MAX_WAIT_LAYERS).contains(layers))
            .ok_or_else(|| {
                format!(
                    "{QWEN_PREFILL_WAIT_LAYERS_ENV} must be 1..={MAX_WAIT_LAYERS}, got {value:?}"
                )
            }),
    }
}

static WAIT_LAYERS: LazyLock<usize> = LazyLock::new(|| {
    parse_wait_layers(std::env::var(QWEN_PREFILL_WAIT_LAYERS_ENV).ok().as_deref()).unwrap_or_else(
        |error| {
            tracing::warn!(error, fallback = 1, "invalid Qwen prefill wait-layers value");
            1
        },
    )
});

thread_local! {
    static SCOPE: Cell<bool> = const { Cell::new(false) };
    static WAIT: Cell<usize> = const { Cell::new(1) };
    /// The wait cadence [`PrefillScope::forced`] opens its scopes with.
    #[cfg(test)]
    static FORCED_WAIT: Cell<usize> = const { Cell::new(1) };
}

/// While alive, the scopes [`PrefillScope::forced`] opens on this thread
/// wait every `wait_layers` layers, so a gate written against the forced
/// route can be repeated at another cadence.
#[cfg(test)]
pub(crate) struct ForcedWaitLayers {
    previous: usize,
}

#[cfg(test)]
impl ForcedWaitLayers {
    pub(crate) fn set(wait_layers: usize) -> Self {
        Self {
            previous: FORCED_WAIT.with(|cell| cell.replace(wait_layers)),
        }
    }
}

#[cfg(test)]
impl Drop for ForcedWaitLayers {
    fn drop(&mut self) {
        FORCED_WAIT.with(|cell| cell.set(self.previous));
    }
}

/// Marks this thread's forward as a prompt-prefill chunk until dropped.
pub(crate) struct PrefillScope {
    previous: bool,
    previous_wait: usize,
}

impl PrefillScope {
    /// Open the scope; the route is taken only when the switch is on.
    pub(crate) fn enter() -> Self {
        #[cfg(test)]
        if let Some(switches) = SwitchesForTest::current() {
            return Self::with(switches.prepared, switches.wait_layers);
        }
        Self::with(*ENABLED, *WAIT_LAYERS)
    }

    /// Open the scope with the route forced on or off, whatever the switch.
    #[cfg(test)]
    pub(crate) fn forced(route: bool) -> Self {
        Self::with(route, FORCED_WAIT.with(Cell::get))
    }

    fn with(route: bool, wait_layers: usize) -> Self {
        Self {
            previous: SCOPE.with(|cell| cell.replace(route)),
            previous_wait: WAIT.with(|cell| cell.replace(wait_layers)),
        }
    }
}

impl Drop for PrefillScope {
    fn drop(&mut self) {
        SCOPE.with(|cell| cell.set(self.previous));
        WAIT.with(|cell| cell.set(self.previous_wait));
    }
}

/// Pooled scratch a forward that skips host waits
/// ([`QWEN_PREFILL_WAIT_LAYERS_ENV`] above 1) may have out when a layer
/// ends without a wait. The worker's scratch pool takes its buffers back
/// only at a host wait, so layers that do not wait accumulate theirs; a
/// layer that ends with more than this out waits. What is out between two
/// waits is therefore at most this plus one layer's scratch, and a forward
/// that waits at every layer has one layer's scratch out as well.
pub(crate) const WAIT_LAYERS_WINDOW_SCRATCH_BYTES: usize = 1 << 30;

/// Pooled scratch a worker keeps between host waits once a joint forward or
/// a forward that skips host waits has run: above it the pool's free
/// buffers are released at the next reset
/// ([`super::decode_pool::release_free_above`]).
pub(crate) const PREFILL_RETAINED_SCRATCH_BYTES: usize = 2 << 30;

#[cfg(test)]
thread_local! {
    /// Test override of the two caps: (window, retained).
    static SCRATCH_CAPS_FOR_TEST: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

/// [`WAIT_LAYERS_WINDOW_SCRATCH_BYTES`] (a test may lower it).
pub(crate) fn window_scratch_cap() -> usize {
    #[cfg(test)]
    if let Some((window, _)) = SCRATCH_CAPS_FOR_TEST.with(Cell::get) {
        return window;
    }
    WAIT_LAYERS_WINDOW_SCRATCH_BYTES
}

/// [`PREFILL_RETAINED_SCRATCH_BYTES`] (a test may lower it).
pub(crate) fn retained_scratch_cap() -> usize {
    #[cfg(test)]
    if let Some((_, retained)) = SCRATCH_CAPS_FOR_TEST.with(Cell::get) {
        return retained;
    }
    PREFILL_RETAINED_SCRATCH_BYTES
}

/// Until dropped, this thread's forwards run under these caps instead of
/// the two constants.
#[cfg(test)]
pub(crate) struct ScratchCapsForTest {
    previous: Option<(usize, usize)>,
}

#[cfg(test)]
impl ScratchCapsForTest {
    pub(crate) fn set(window: usize, retained: usize) -> Self {
        Self {
            previous: SCRATCH_CAPS_FOR_TEST.with(|cell| cell.replace(Some((window, retained)))),
        }
    }
}

#[cfg(test)]
impl Drop for ScratchCapsForTest {
    fn drop(&mut self) {
        SCRATCH_CAPS_FOR_TEST.with(|cell| cell.set(self.previous));
    }
}

/// Upper bound of the pooled scratch `HF2Q_QWEN_PREFILL_JOINT` keeps between
/// steps: enforced, so it holds for every model geometry, slot count and
/// sequence of forwards.
pub(crate) const fn joint_scratch_bound_bytes() -> u64 {
    PREFILL_RETAINED_SCRATCH_BYTES as u64
}

/// Upper bound of the pooled scratch a worker holds because of
/// `HF2Q_QWEN_PREFILL_WAIT_LAYERS=wait_layers`, beyond what the same
/// forwards hold when they wait at every layer: what the pool keeps between
/// waits plus what a window may add before a layer waits. Enforced, so it
/// holds for every model geometry, slot count and sequence of forwards; 0
/// at the default cadence.
pub(crate) const fn wait_layers_scratch_bound_bytes(wait_layers: usize) -> u64 {
    if wait_layers > 1 {
        (PREFILL_RETAINED_SCRATCH_BYTES + WAIT_LAYERS_WINDOW_SCRATCH_BYTES) as u64
    } else {
        0
    }
}

/// Layers between host waits for a chunk of more than 32 rows on this
/// thread: the operator's value inside a prompt-prefill scope, 1 outside.
pub(crate) fn wait_layers() -> usize {
    WAIT.with(Cell::get)
}

thread_local! {
    static SINGLE_COMMAND_LAYERS: Cell<bool> = const { Cell::new(false) };
}

/// While alive, the layer builders of this thread's forward keep every
/// stage of a layer in the encoder session's open command buffer instead of
/// committing each stage on its own. The layer loop sets it for a forward
/// that submits one command buffer per layer
/// ([`QWEN_PREFILL_WAIT_LAYERS_ENV`] above 1).
pub(crate) struct SingleCommandLayers {
    previous: bool,
}

impl SingleCommandLayers {
    pub(crate) fn set(on: bool) -> Self {
        Self {
            previous: SINGLE_COMMAND_LAYERS.with(|cell| cell.replace(on)),
        }
    }
}

impl Drop for SingleCommandLayers {
    fn drop(&mut self) {
        SINGLE_COMMAND_LAYERS.with(|cell| cell.set(self.previous));
    }
}

/// Whether this thread's forward keeps a layer in one command buffer.
pub(crate) fn single_command_layers() -> bool {
    SINGLE_COMMAND_LAYERS.with(Cell::get)
}

/// The prompt-prefill switches of a SlotAware Qwen worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrefillSwitches {
    pub(crate) prepared: bool,
    pub(crate) fold_cue: bool,
    pub(crate) joint: bool,
    /// Layers between host waits of a chunk of more than 32 rows.
    pub(crate) wait_layers: usize,
    /// Whether a decoding worker keeps the prefill-only weights fresh.
    pub(crate) keep_fresh: bool,
}

#[cfg(test)]
thread_local! {
    static TEST_SWITCHES: Cell<Option<PrefillSwitches>> = const { Cell::new(None) };
}

/// The switches a test's thread runs under instead of the process's, until
/// dropped: what [`PrefillScope::enter`] and the cue fold read in a
/// scheduler-level test.
#[cfg(test)]
pub(crate) struct SwitchesForTest {
    previous: Option<PrefillSwitches>,
}

#[cfg(test)]
impl SwitchesForTest {
    pub(crate) fn set(switches: PrefillSwitches) -> Self {
        Self {
            previous: TEST_SWITCHES.with(|cell| cell.replace(Some(switches))),
        }
    }

    pub(crate) fn current() -> Option<PrefillSwitches> {
        TEST_SWITCHES.with(Cell::get)
    }
}

#[cfg(test)]
impl Drop for SwitchesForTest {
    fn drop(&mut self) {
        TEST_SWITCHES.with(|cell| cell.set(self.previous));
    }
}

/// The switches as an operator wrote them, read strictly: an invalid value
/// is an error, not a fallback, and so are the two combinations that cannot
/// keep their promise. With the chunk-scan DeltaNet investigation route on,
/// alpha and beta would stay on the stock kernel beside prepared qkv, gate
/// and output, and a row's bits would depend on whether its chunk is a
/// multiple of 64 rows. The cue fold and the joint prefill are exact only on
/// the prepared kernels.
pub(crate) fn resolve_switches(
    prepared: Option<&str>,
    fold_cue: Option<&str>,
    joint: Option<&str>,
    wait_layers: Option<&str>,
    keep_fresh: Option<&str>,
    chunk_scan_prefill: bool,
) -> Result<PrefillSwitches, String> {
    use super::prefill_fold::QWEN_PREFILL_FOLD_CUE_ENV;
    use super::prefill_joint::QWEN_PREFILL_JOINT_ENV;
    let switches = PrefillSwitches {
        prepared: parse_switch(QWEN_PREFILL_PREPARED_ENV, prepared)?,
        fold_cue: parse_switch(QWEN_PREFILL_FOLD_CUE_ENV, fold_cue)?,
        joint: parse_switch(QWEN_PREFILL_JOINT_ENV, joint)?,
        wait_layers: parse_wait_layers(wait_layers)?,
        keep_fresh: parse_switch(
            super::prefill_keep_fresh::QWEN_PREFILL_KEEP_FRESH_ENV,
            keep_fresh,
        )?,
    };
    if switches.prepared && chunk_scan_prefill {
        return Err(format!(
            "{QWEN_PREFILL_PREPARED_ENV} cannot be combined with the chunk-scan prefill \
             investigation route: its alpha/beta projections keep the stock kernel"
        ));
    }
    if switches.fold_cue && !switches.prepared {
        return Err(format!(
            "{QWEN_PREFILL_FOLD_CUE_ENV} requires {QWEN_PREFILL_PREPARED_ENV}"
        ));
    }
    if switches.joint && !switches.prepared {
        return Err(format!(
            "{QWEN_PREFILL_JOINT_ENV} requires {QWEN_PREFILL_PREPARED_ENV}"
        ));
    }
    Ok(switches)
}

/// [`resolve_switches`] with the refusal an operator reads: the spawn path
/// prints an error's outer message only, so the variable and the reason are
/// in it.
pub(crate) fn switches_or_refusal(
    prepared: Option<&str>,
    fold_cue: Option<&str>,
    joint: Option<&str>,
    wait_layers: Option<&str>,
    keep_fresh: Option<&str>,
    chunk_scan_prefill: bool,
) -> anyhow::Result<PrefillSwitches> {
    resolve_switches(
        prepared,
        fold_cue,
        joint,
        wait_layers,
        keep_fresh,
        chunk_scan_prefill,
    )
    .map_err(|cause| anyhow::anyhow!("Qwen prompt prefill switches: {cause}"))
}

/// The switches of this process, read strictly. Called before a Qwen model
/// is loaded, so a mistyped value is refused before the load is paid for.
pub(crate) fn switches_from_process() -> anyhow::Result<PrefillSwitches> {
    let read = |name: &str| std::env::var(name).ok();
    switches_or_refusal(
        read(QWEN_PREFILL_PREPARED_ENV).as_deref(),
        read(super::prefill_fold::QWEN_PREFILL_FOLD_CUE_ENV).as_deref(),
        read(super::prefill_joint::QWEN_PREFILL_JOINT_ENV).as_deref(),
        read(QWEN_PREFILL_WAIT_LAYERS_ENV).as_deref(),
        read(super::prefill_keep_fresh::QWEN_PREFILL_KEEP_FRESH_ENV).as_deref(),
        crate::debug::INVESTIGATION_ENV.chunk_scan_prefill,
    )
}

/// Validate the switches when a SlotAware worker provisions its slots, so a
/// mistyped value stops the server instead of warning at the first prefill,
/// and log what the worker will run.
pub(crate) fn validate_at_worker_start() -> anyhow::Result<PrefillSwitches> {
    let switches = switches_from_process()?;
    tracing::info!(
        target: "hf2q::serve::api::engine_qwen35::progress",
        prepared = switches.prepared,
        fold_cue = switches.fold_cue,
        joint = switches.joint,
        wait_layers = switches.wait_layers,
        keep_fresh = switches.keep_fresh,
        // Enforced caps on the pooled scratch the two switches add: what
        // memory admission charges them.
        joint_scratch_bound_bytes = if switches.joint {
            joint_scratch_bound_bytes()
        } else {
            0
        },
        wait_layers_scratch_bound_bytes = wait_layers_scratch_bound_bytes(switches.wait_layers),
        "Qwen prompt prefill configuration"
    );
    Ok(switches)
}

/// Whether the prepared kernel serves this weight shape.
fn shape_admitted(ggml_type: GgmlType, in_features: u32, out_features: u32) -> bool {
    matches!(ggml_type, GgmlType::Q4_K | GgmlType::Q6_K)
        && out_features % 64 == 0
        && in_features % 256 == 0
}

/// Whether this product of a multi-row prefill chunk takes the prepared
/// route: the scope is open with the route on, the weight is a raw GGML block
/// matrix, and the prepared kernel serves its shape.
pub(crate) fn admits(
    rows: u32,
    weight: &MlxBuffer,
    ggml_type: GgmlType,
    in_features: u32,
    out_features: u32,
) -> bool {
    rows > 1
        && SCOPE.with(Cell::get)
        && weight.dtype() == DType::U8
        && shape_admitted(ggml_type, in_features, out_features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::joint_step_options::parse_switch;
    use mlx_native::MlxDevice;

    #[test]
    fn switch_is_off_unless_set_on() {
        let parse = |raw| parse_switch(QWEN_PREFILL_PREPARED_ENV, raw);
        assert_eq!(parse(None), Ok(false));
        assert_eq!(parse(Some(" ")), Ok(false));
        assert_eq!(parse(Some("1")), Ok(true));
        assert_eq!(parse(Some("off")), Ok(false));
        assert!(parse(Some("rows")).is_err());
    }

    #[test]
    fn worker_start_rejects_invalid_values_and_combinations_that_break_a_promise() {
        let resolve =
            |prepared, fold, chunk_scan| resolve_switches(prepared, fold, None, None, None, chunk_scan);
        assert_eq!(
            resolve(None, None, false),
            Ok(PrefillSwitches {
                prepared: false,
                fold_cue: false,
                joint: false,
                wait_layers: 1,
                keep_fresh: false
            })
        );
        assert_eq!(
            resolve(Some("1"), Some("on"), false),
            Ok(PrefillSwitches {
                prepared: true,
                fold_cue: true,
                joint: false,
                wait_layers: 1,
                keep_fresh: false
            })
        );
        assert_eq!(
            resolve_switches(Some("1"), None, Some("1"), Some("4"), Some("1"), false),
            Ok(PrefillSwitches {
                prepared: true,
                fold_cue: false,
                joint: true,
                wait_layers: 4,
                keep_fresh: true
            })
        );
        // The keep-alive reads weights; it does not depend on a route.
        assert_eq!(
            resolve_switches(None, None, None, None, Some("on"), false).map(|s| s.keep_fresh),
            Ok(true)
        );
        assert!(
            resolve_switches(Some("1"), None, None, None, Some("250"), false).is_err(),
            "invalid keep-fresh value"
        );
        // The wait cadence is a property of the layer loop, not of a route.
        assert_eq!(
            resolve_switches(None, None, None, Some("8"), None, false).map(|s| s.wait_layers),
            Ok(8)
        );
        for invalid in ["0", "9", "on", "-1"] {
            assert!(
                resolve_switches(Some("1"), None, None, Some(invalid), None, false).is_err(),
                "wait layers {invalid}"
            );
        }
        assert!(
            resolve_switches(None, None, Some("1"), None, None, false).is_err(),
            "joint without prepared"
        );
        assert!(
            resolve_switches(Some("1"), None, Some("lanes"), None, None, false).is_err(),
            "invalid joint value"
        );
        // The stock route may run beside the chunk-scan investigation route.
        assert!(resolve(None, None, true).is_ok());
        for (prepared, fold, chunk_scan, what, names) in [
            (
                Some("yes"),
                None,
                false,
                "invalid prepared value",
                QWEN_PREFILL_PREPARED_ENV,
            ),
            (
                Some("1"),
                Some("cue"),
                false,
                "invalid fold value",
                super::super::prefill_fold::QWEN_PREFILL_FOLD_CUE_ENV,
            ),
            (
                Some("1"),
                None,
                true,
                "prepared beside chunk scan",
                QWEN_PREFILL_PREPARED_ENV,
            ),
            (
                None,
                Some("1"),
                false,
                "fold without prepared",
                super::super::prefill_fold::QWEN_PREFILL_FOLD_CUE_ENV,
            ),
        ] {
            assert!(resolve(prepared, fold, chunk_scan).is_err(), "{what}");
            // What the spawn path prints is the outer message alone.
            let printed = switches_or_refusal(prepared, fold, None, None, None, chunk_scan)
                .expect_err(what)
                .to_string();
            assert!(
                printed.starts_with("Qwen prompt prefill switches: ") && printed.contains(names),
                "{what}: the refusal reads {printed:?}"
            );
        }
    }

    #[test]
    fn a_test_override_replaces_the_process_switches_on_its_thread_only() {
        assert_eq!(SwitchesForTest::current(), None);
        let on = PrefillSwitches {
            prepared: true,
            fold_cue: true,
            joint: false,
            wait_layers: 1,
            keep_fresh: false,
        };
        {
            let _switches = SwitchesForTest::set(on);
            assert_eq!(SwitchesForTest::current(), Some(on));
            assert!(super::super::prefill_fold::enabled());
            let _scope = PrefillScope::enter();
            assert!(SCOPE.with(Cell::get));
            std::thread::spawn(|| assert_eq!(SwitchesForTest::current(), None))
                .join()
                .expect("other thread");
        }
        assert_eq!(SwitchesForTest::current(), None);
    }

    #[test]
    fn wait_layers_default_to_one_and_apply_only_inside_a_scope() {
        assert_eq!(parse_wait_layers(None), Ok(1));
        assert_eq!(parse_wait_layers(Some(" ")), Ok(1));
        assert_eq!(parse_wait_layers(Some("4")), Ok(4));
        assert_eq!(parse_wait_layers(Some("8")), Ok(8));
        assert_eq!(wait_layers(), 1, "no scope");
        {
            let _scope = PrefillScope::forced(true);
            assert_eq!(wait_layers(), 1, "a forced scope waits every layer");
            {
                let _cadence = ForcedWaitLayers::set(4);
                assert_eq!(wait_layers(), 1, "an open scope keeps its cadence");
                let _inner = PrefillScope::forced(true);
                assert_eq!(wait_layers(), 4, "an inner scope overrides");
            }
            assert_eq!(wait_layers(), 1, "restored on drop");
        }
        assert_eq!(wait_layers(), 1, "closed");
        {
            let _cadence = ForcedWaitLayers::set(8);
            assert_eq!(wait_layers(), 1, "the cadence applies inside a scope only");
        }
    }

    #[test]
    fn route_is_admitted_only_inside_a_routed_scope_for_prepared_shapes() {
        let device = MlxDevice::new().expect("device");
        let blocks = device
            .alloc_buffer(64, DType::U8, vec![64])
            .expect("block bytes");
        let floats = device
            .alloc_buffer(64, DType::F32, vec![16])
            .expect("float weight");
        let q4k = |rows| admits(rows, &blocks, GgmlType::Q4_K, 5120, 17408);
        assert!(!q4k(157), "no scope");
        {
            let _off = PrefillScope::forced(false);
            assert!(!q4k(157), "scope with the route off");
        }
        {
            let _on = PrefillScope::forced(true);
            assert!(q4k(157));
            assert!(q4k(2), "any multi-row chunk");
            assert!(!q4k(1), "a one-row forward is the decode graph");
            assert!(admits(31, &blocks, GgmlType::Q6_K, 17408, 5120));
            assert!(
                !admits(31, &floats, GgmlType::Q4_K, 5120, 17408),
                "not GGML blocks"
            );
            assert!(
                !admits(31, &blocks, GgmlType::Q8_0, 5120, 17408),
                "not a K-quant"
            );
            assert!(
                !admits(31, &blocks, GgmlType::Q4_K, 5120, 48),
                "N % 64 != 0"
            );
            assert!(
                !admits(31, &blocks, GgmlType::Q4_K, 5120 + 32, 6144),
                "K % 256 != 0"
            );
            {
                let _inner = PrefillScope::forced(false);
                assert!(!q4k(157), "an inner scope overrides");
            }
            assert!(q4k(157), "restored on drop");
        }
        assert!(!q4k(157), "closed");
    }
}
