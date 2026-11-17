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

use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, MlxBuffer};

use super::joint_step_options::parse_switch;
use crate::inference::spec_decode::dflash2::route_profile;

/// Switch: `1`/`on`/`true` routes prompt-prefill chunks through the prepared
/// row-tile MM. On in the route profile of a qualified DFlash2 pair, off for
/// every other artifact; an explicit value wins either way.
pub(crate) const QWEN_PREFILL_PREPARED_ENV: &str = "HF2Q_QWEN_PREFILL_PREPARED";

/// The switch as resolved for the model this thread serves. An invalid value
/// warns once and keeps the route off.
pub(crate) fn enabled() -> bool {
    const SWITCH: route_profile::Switch = route_profile::switch(QWEN_PREFILL_PREPARED_ENV);
    route_profile::with_value(SWITCH, |raw| parse_switch(QWEN_PREFILL_PREPARED_ENV, raw))
        .unwrap_or_else(|error| {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    error,
                    fallback = false,
                    "invalid Qwen prepared-prefill switch"
                );
            });
            false
        })
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

fn resolved_wait_layers() -> usize {
    const SWITCH: route_profile::Switch = route_profile::switch(QWEN_PREFILL_WAIT_LAYERS_ENV);
    route_profile::with_value(SWITCH, parse_wait_layers).unwrap_or_else(|error| {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(error, fallback = 1, "invalid Qwen prefill wait-layers value");
        });
        1
    })
}

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
        Self::with(enabled(), resolved_wait_layers())
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
/// layer that ends with more than this out waits. This is a threshold, not
/// a cap: the layer that crosses it has already allocated, so between two
/// host waits up to this plus that one layer's scratch is out.
pub(crate) const WAIT_LAYERS_WINDOW_SCRATCH_BYTES: usize = 1 << 27;

/// Free bytes retained after a checked reset. At up to 2080 rows Qwen3.8's
/// prepared dense-FFN BF16 hidden activation requests 2080 * 17408 * 2 =
/// 72,417,280 bytes, requiring a 128 MiB bucket. FA Q/gate and BF16 staging
/// now borrow their forward arenas instead of warming pooled roots. Keep one
/// large prepared bucket/window warm; reset releases free buffers above it,
/// not merely their accounting charge.
pub(crate) const PREFILL_RETAINED_SCRATCH_BYTES: usize = 1 << 27;

/// What the pool may hold at any time while a joint forward or a forward
/// that skips host waits runs, and until the reset after it. Checked before
/// every pooled allocation ([`super::decode_pool::cap_pool`]): a request
/// that could take the pool over it is served by a plain buffer that is
/// released at the next host wait. A hard cap, so it holds for every model
/// geometry, slot count, shared-prefix layout and sequence of forwards.
pub(crate) const PREFILL_POOL_CAP_BYTES: usize =
    PREFILL_RETAINED_SCRATCH_BYTES + WAIT_LAYERS_WINDOW_SCRATCH_BYTES;

#[cfg(test)]
thread_local! {
    /// Test override: (window threshold, pool caps).
    static SCRATCH_CAPS_FOR_TEST: Cell<Option<(usize, super::decode_pool::PoolCaps)>> =
        const { Cell::new(None) };
}

/// [`WAIT_LAYERS_WINDOW_SCRATCH_BYTES`] (a test may lower it).
pub(crate) fn window_scratch_threshold() -> usize {
    #[cfg(test)]
    if let Some((window, _)) = SCRATCH_CAPS_FOR_TEST.with(Cell::get) {
        return window;
    }
    WAIT_LAYERS_WINDOW_SCRATCH_BYTES
}

/// The pool caps of a joint forward and of a forward that skips host waits
/// (a test may lower them).
pub(crate) fn prefill_pool_caps() -> super::decode_pool::PoolCaps {
    #[cfg(test)]
    if let Some((_, caps)) = SCRATCH_CAPS_FOR_TEST.with(Cell::get) {
        return caps;
    }
    super::decode_pool::PoolCaps {
        pool: PREFILL_POOL_CAP_BYTES,
        retained: PREFILL_RETAINED_SCRATCH_BYTES,
    }
}

/// Until dropped, this thread's forwards run under this window threshold
/// and these pool caps instead of the constants.
#[cfg(test)]
pub(crate) struct ScratchCapsForTest {
    previous: Option<(usize, super::decode_pool::PoolCaps)>,
}

#[cfg(test)]
impl ScratchCapsForTest {
    pub(crate) fn set(window: usize, retained: usize, pool: usize) -> Self {
        let caps = super::decode_pool::PoolCaps { pool, retained };
        Self {
            previous: SCRATCH_CAPS_FOR_TEST.with(|cell| cell.replace(Some((window, caps)))),
        }
    }
}

#[cfg(test)]
impl Drop for ScratchCapsForTest {
    fn drop(&mut self) {
        SCRATCH_CAPS_FOR_TEST.with(|cell| cell.set(self.previous));
    }
}

/// Maximum sum of buckets for at most two nonempty segments with combined
/// rows <= `rows`, for one allocation of `bytes_per_row` per segment.
///
/// Let T = rows*C and B = next_pow2(T). If neither payload exceeds B/2,
/// their two buckets sum to <= B. Otherwise exactly one does, rounding to
/// B. Its smallest possible row count is floor((B/2)/C)+1; assigning every
/// remaining row to the other segment maximizes that segment's bucket.
/// If no row remains, only the one-segment bucket B is possible. Monotone
/// buckets also cover lanes with fewer than `rows` rows. Checked arithmetic
/// rejects zero/overflow instead of inventing a smaller allocation bound.
fn two_segment_bucket_bytes(rows: u128, bytes_per_row: u128) -> Option<u128> {
    if rows == 0 || bytes_per_row == 0 {
        return None;
    }
    let bucket = rows.checked_mul(bytes_per_row)?.checked_next_power_of_two()?;
    let first = (bucket / 2 / bytes_per_row).checked_add(1)?;
    if first < rows {
        let tail = (rows - first).checked_mul(bytes_per_row)?.checked_next_power_of_two()?;
        bucket.checked_add(tail)
    } else {
        Some(bucket)
    }
}

/// Upper bound of the scratch one layer of a prompt-prefill forward on the
/// prepared route requests from the worker's scratch pool (bucket bytes,
/// whether the pool or a plain buffer serves them), for a forward of
/// `forward_rows` rows over `lanes` slots.
///
/// Derived from the requests the prepared route makes (`pooled_alloc_buffer`
/// call sites; each request occupies its next-power-of-two bucket):
/// - A DeltaNet layer runs its projections once over all rows `R`: the BF16
///   copy of the normed input (`2Rh`), the fused alpha/beta logits
///   (`4R·ab`, `ab` = 2·heads rounded up to 64), their two halves
///   (`4R·heads` each), the BF16 copy of the gated values (`2R·z`,
///   `z` = value heads × value head dim) and the output projection (`4Rh`).
/// - A full-attention layer's multi-row segments reuse the combined Q helper
///   root for fused Q/gate and sequence staging for gated BF16 inputs. The
///   normalized BF16 input also borrows staging when h <= q. The output
///   projection (`4rh`) and an unshared attention result (`4rq`) remain pooled.
///   A one-row segment bypasses the arenas, so charge its input/Q-gate/gated
///   copies separately. Up to two <=32-row segments/lane add direct-TQ tile
///   scratch (`4·32·32·heads·(head dim + 2)`) and two row-id lists.
///   `PlanLane::segments` splits only at the optional content/cue boundary:
///   combined rows, including a private shared-prefix copy, are <= R, since
///   `plan` validates R = shared_rows + sum(each lane's own rows). Independent
///   remaining pooled roots use `two_segment_bucket_bytes`; the segmented
///   forward also owns its joined output (`4Rh`).
/// - The dense FFN requests the BF16 copies of its input (`2Rh`) and of its
///   hidden activations (`2Rm`); the MoE FFN the shared expert's output
///   (`4rh`) per call: one call over the long segments' rows and one per
///   short segment.
/// - Outside the layers, in the same window: the token embedding when it is
///   not gathered natively (`4Rh`) and, per slot, two output-head norm rows.
///
/// It holds for the prepared route (`HF2Q_QWEN_PREFILL_PREPARED`, every
/// projection of the layer on the prepared kernel, per-forward arenas),
/// which is the route joint forwards and the served profiles run; the
/// legacy routes request other scratch and are not covered. The forward
/// checks every layer against it and reports a layer that exceeds it
/// (`decode_pool::note_layer_scratch`).
/// `shared_attention_result` is a proven route capability: joint D256
/// attention, TQ-only cache, and the enabled fast resume workspace. Its
/// large fresh/resumed results alias one forward root. The <=32-row direct
/// route's plain outputs are charged across their actual held window by
/// `prefill_memory::bound`, not disguised as pooled requests. Unknown and
/// legacy routes must pass false. Short MoE calls include staged gathers and
/// both live-tile implementations, not just the shared expert's output.
pub(crate) fn prefill_layer_scratch_bound_bytes(
    cfg: &super::Qwen35Config,
    forward_rows: usize,
    lanes: usize,
    shared_attention_result: bool,
) -> u64 {
    if forward_rows == 0 || lanes == 0
        || forward_rows > u32::MAX as usize || lanes > u32::MAX as usize
    {
        return u64::MAX;
    }
    let bucket = |bytes: u128| bytes.next_power_of_two();
    let rows = forward_rows as u128;
    let lanes = lanes as u128;
    let h = u128::from(cfg.hidden_size);
    let heads = u128::from(cfg.num_attention_heads);
    let head_dim = u128::from(cfg.head_dim);
    let q = heads * head_dim;
    let value_heads = u128::from(cfg.linear_num_value_heads);
    let z = value_heads * u128::from(cfg.linear_value_head_dim);
    if h == 0 || q == 0 || z == 0 || cfg.intermediate_size == Some(0)
        || q > u32::MAX as u128 || z > u32::MAX as u128
    {
        return u64::MAX;
    }
    let ab = (2 * value_heads).div_ceil(64) * 64;
    let short_segments = 2 * lanes;
    // A joint TQ-fast forward routes every >32-row result to its shared
    // root. Direct <=32-row results are plain, not pooled, and their held
    // lifetime is charged by prefill_memory::bound instead.
    let Some(mut per_lane) = two_segment_bucket_bytes(rows, 4 * h) else {
        return u64::MAX;
    };
    if h > q {
        let Some(input) = two_segment_bucket_bytes(rows, 2 * h) else {
            return u64::MAX;
        };
        per_lane += input;
    }
    // seq=1 bypasses the FA arenas; it can occur at either segment boundary.
    let single_row_copies = bucket(8 * q) + bucket(2 * q)
        + if h <= q { bucket(2 * h) } else { 0 };
    if !shared_attention_result {
        let Some(result) = two_segment_bucket_bytes(rows, 4 * q) else {
            return u64::MAX;
        };
        per_lane += result;
    }
    let attention = lanes * per_lane
        + short_segments * single_row_copies
        + short_segments * (bucket(4 * 32 * 32 * heads * (head_dim + 2)) + 2 * bucket(4 * 32))
        + bucket(4 * rows * h);
    let delta_net = bucket(2 * rows * h)
        + bucket(4 * rows * ab)
        + 2 * bucket(4 * rows * value_heads)
        + bucket(2 * rows * z)
        + bucket(4 * rows * h);
    let ffn = match cfg.intermediate_size {
        Some(m) => bucket(2 * rows * h) + bucket(2 * rows * u128::from(m)),
        None => 2 * 4 * h * rows
            + short_segments * u128::from(super::prefill_memory::short_moe_scratch_bytes(cfg)),
    };
    let outside = bucket(4 * rows * h) + 2 * lanes * (bucket(4 * h) + bucket(8));
    u64::try_from(attention.max(delta_net) + ffn + outside).unwrap_or(u64::MAX)
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
    // A forward skips host waits only on the prepared route: its scratch
    // bound is derived for that route alone.
    if switches.wait_layers > 1 && !switches.prepared {
        return Err(format!(
            "{QWEN_PREFILL_WAIT_LAYERS_ENV} above 1 requires {QWEN_PREFILL_PREPARED_ENV}"
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

/// Whether this thread's forward runs in a prefill scope with the prepared
/// route on.
pub(crate) fn route_in_scope() -> bool {
    SCOPE.with(Cell::get)
}

/// The switches as resolved for the model this thread loads or serves, read
/// strictly. Called before a Qwen model is loaded, so a mistyped value is
/// refused before the load is paid for.
pub(crate) fn switches_from_process() -> anyhow::Result<PrefillSwitches> {
    let routes = route_profile::current();
    let read = |name| routes.value(route_profile::switch(name));
    switches_or_refusal(
        read(QWEN_PREFILL_PREPARED_ENV),
        read(super::prefill_fold::QWEN_PREFILL_FOLD_CUE_ENV),
        read(super::prefill_joint::QWEN_PREFILL_JOINT_ENV),
        read(QWEN_PREFILL_WAIT_LAYERS_ENV),
        read(super::prefill_keep_fresh::QWEN_PREFILL_KEEP_FRESH_ENV),
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
/// matrix, the prepared kernel serves its shape, and this host's GPU runs
/// the prepared kernels ([`super::gpu_ffn::prepared_mm_runs`]). The host is
/// asked here, at every decision, not when the scope opens: a scope opened
/// before the host was probed must not admit a product the host cannot run.
/// The cue fold asks this for every shared product, so it keeps its two
/// forwards on such a host.
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
        && super::gpu_ffn::prepared_mm_runs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::MlxDevice;

    #[test]
    fn layer_bucket_bound_rejects_zero_and_overflowing_native_geometry() {
        let source: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/qwen38/config.json"
        )).unwrap();
        let cfg = super::super::source_config::qwen35_config_from_authenticated_source(&source).unwrap();
        for (rows, lanes) in [(0, 1), (1, 0), (u32::MAX as usize + 1, 1), (1, u32::MAX as usize + 1)] {
            assert_eq!(prefill_layer_scratch_bound_bytes(&cfg, rows, lanes, true), u64::MAX);
        }
        for field in 0..6 {
            let mut invalid = cfg.clone();
            match field {
                0 => invalid.hidden_size = 0,
                1 => invalid.num_attention_heads = 0,
                2 => invalid.head_dim = 0,
                3 => invalid.linear_num_value_heads = 0,
                4 => invalid.intermediate_size = Some(0),
                _ => invalid.num_attention_heads = u32::MAX,
            }
            assert_eq!(prefill_layer_scratch_bound_bytes(&invalid, 2080, 4, true), u64::MAX);
        }
    }

    #[test]
    fn segment_bucket_bound_is_exact_for_every_split_and_rejects_invalid_sizes() {
        for rows in 1..=130u128 {
            for width in 1..=131u128 {
                let whole = (rows * width).next_power_of_two();
                let actual = (1..rows).map(|first| {
                    (first * width).next_power_of_two()
                        + ((rows - first) * width).next_power_of_two()
                }).fold(whole, u128::max);
                assert_eq!(two_segment_bucket_bytes(rows, width), Some(actual));
            }
        }
        // Include the served capacities, with non-power-of-two row widths
        // (Qwen3.8 h/q) as well as Qwen3.6's power-of-two widths.
        for rows in [2048u128, 2080] {
            for width in [4096, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152] {
                let actual = (1..rows).map(|first| {
                    (first * width).next_power_of_two()
                        + ((rows - first) * width).next_power_of_two()
                }).max().unwrap();
                assert_eq!(two_segment_bucket_bytes(rows, width), Some(actual));
            }
        }
        for (rows, width) in [(0, 4), (4, 0), (u128::MAX, 2), (u128::MAX, 1)] {
            assert_eq!(two_segment_bucket_bytes(rows, width), None);
        }
    }

    #[test]
    fn segment_bucket_bound_covers_real_unequal_tail_and_capacity_allocations() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let Ok(device) = MlxDevice::new() else { return };
        // The five pooled requests per attention segment: BF16 input,
        // fused F32 Q/gate, BF16 gated values, F32 projection and (without
        // shared output) F32 attention result. Exercise the native allocator,
        // not a second copy of its bucket formula.
        for (h, q) in [(2048usize, 16 * 256), (5120, 24 * 256)] {
            for width in [2 * h, 8 * q, 2 * q, 4 * h, 4 * q] {
                for rows in [38usize, 72, 2048, 2080] {
                    let bucket = (rows * width).next_power_of_two();
                    let first = bucket / 2 / width + 1;
                    for split in [1, rows - 1, rows - 5, rows - 32, first.min(rows - 1)] {
                        let mut pool = mlx_native::MlxBufferPool::new();
                        let roots: Vec<_> = [split, rows - split].into_iter().map(|count| {
                            pool.alloc(&device, count * width, DType::U8, vec![count * width])
                                .expect("real segment allocation")
                        }).collect();
                        let actual: usize = roots.iter().map(MlxBuffer::byte_len).sum();
                        let bound = two_segment_bucket_bytes(rows as u128, width as u128).unwrap();
                        assert!(actual as u128 <= bound, "{rows} rows split at {split}, width {width}");
                        assert_eq!(pool.memory_snapshot().unwrap().retained.bytes, actual);
                        if split == first && first < rows {
                            assert_eq!(actual as u128, bound, "extremal partition attains the bound");
                        }
                        drop(roots);
                        pool.reset();
                        assert_eq!(pool.free_bytes(), actual, "both segment roots remain owned");
                        pool.clear();
                        assert_eq!(pool.memory_snapshot().unwrap().retained.bytes, 0);
                    }
                }
            }
        }
    }

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
        // A forward skips host waits only on the prepared route.
        assert_eq!(
            resolve_switches(Some("1"), None, None, Some("8"), None, false).map(|s| s.wait_layers),
            Ok(8)
        );
        assert_eq!(
            resolve_switches(None, None, None, Some("1"), None, false).map(|s| s.wait_layers),
            Ok(1)
        );
        assert!(
            resolve_switches(None, None, None, Some("8"), None, false).is_err(),
            "wait layers without prepared"
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
        use crate::inference::route_host::HostForTest;
        // The rule itself, on a host that runs the prepared kernels.
        let _host = HostForTest::with_tensor_api();
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
            // The host is asked at every decision: a scope that was opened
            // with the route on admits nothing once the host is known not
            // to run the prepared kernels.
            {
                let _incapable = HostForTest::without_tensor_api();
                assert!(!q4k(157), "a host without the prepared kernels");
                let _reopened = PrefillScope::forced(true);
                assert!(!q4k(157), "whatever scope is opened there");
            }
            assert!(q4k(157), "the host is restored on drop");
        }
        assert!(!q4k(157), "closed");
    }
}
