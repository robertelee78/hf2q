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
pub(crate) const WAIT_LAYERS_WINDOW_SCRATCH_BYTES: usize = 1 << 30;

/// Free bytes the pool keeps at a reset once a joint forward or a forward
/// that skips host waits has run: above it the reset releases the free
/// buffers.
pub(crate) const PREFILL_RETAINED_SCRATCH_BYTES: usize = 2 << 30;

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

/// Upper bound of the scratch one layer of a prompt-prefill forward on the
/// prepared route requests from the worker's scratch pool (bucket bytes,
/// whether the pool or a plain buffer serves them), for a forward of
/// `forward_rows` rows over `lanes` slots.
///
/// Derived from the requests the prepared route makes (`pooled_alloc_buffer`
/// call sites; a buffer of `n` bytes occupies the power of two it rounds up
/// to, at most `2n`, which is exact for the single-call terms and the
/// factor on the per-segment sums):
/// - A DeltaNet layer runs its projections once over all rows `R`: the BF16
///   copy of the normed input (`2Rh`), the fused alpha/beta logits
///   (`4R·ab`, `ab` = 2·heads rounded up to 64), their two halves
///   (`4R·heads` each), the BF16 copy of the gated values (`2R·z`,
///   `z` = value heads × value head dim) and the output projection (`4Rh`).
/// - A full-attention layer is built per slot and segment; a segment of `r`
///   rows requests the BF16 input copy (`2rh`), the fused query/gate
///   projection (`4r·2q`, `q` = heads × head dim), the BF16 copy of the gated
///   attention output (`2rq`), the output projection (`4rh`) and the
///   attention output (`4rq`); a segment of at most 32 rows on the direct TQ
///   route adds its tile scratch (`4·32·32·heads·(head dim + 2)` at 32 rows)
///   and two row-id lists. A lane's segments cover its own rows and its
///   private copy of a shared prefix, at most `R` rows per lane, and at most
///   two of them are short (a short content chunk and the cue). A forward
///   with segments also requests the joined attention output (`4Rh`).
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
pub(crate) fn prefill_layer_scratch_bound_bytes(
    cfg: &super::Qwen35Config,
    forward_rows: usize,
    lanes: usize,
) -> u64 {
    let bucket = |bytes: u64| bytes.max(1).next_power_of_two();
    let rows = forward_rows as u64;
    let lanes = lanes.max(1) as u64;
    let lane_rows = rows * lanes;
    let h = u64::from(cfg.hidden_size);
    let heads = u64::from(cfg.num_attention_heads);
    let head_dim = u64::from(cfg.head_dim);
    let q = heads * head_dim;
    let value_heads = u64::from(cfg.linear_num_value_heads);
    let z = value_heads * u64::from(cfg.linear_value_head_dim);
    let ab = u64::from(super::prepared_projection::alpha_beta_rows(cfg.linear_num_value_heads));
    let short_segments = 2 * lanes;
    let attention = 2 * lane_rows * (2 * h + 8 * q + 2 * q + 4 * h + 4 * q)
        + short_segments * (bucket(4 * 32 * 32 * heads * (head_dim + 2)) + 2 * bucket(4 * 32))
        + bucket(4 * rows * h);
    let delta_net = bucket(2 * rows * h)
        + bucket(4 * rows * ab)
        + 2 * bucket(4 * rows * value_heads)
        + bucket(2 * rows * z)
        + bucket(4 * rows * h);
    let ffn = match cfg.intermediate_size {
        Some(m) => bucket(2 * rows * h) + bucket(2 * rows * u64::from(m)),
        None => 2 * 4 * h * (rows + 32 * short_segments),
    };
    let outside = bucket(4 * rows * h) + 2 * lanes * (bucket(4 * h) + bucket(8));
    attention.max(delta_net) + ffn + outside
}

/// Forward-wide device allocations beside the pool, including the arenas
/// also used by a solo prefill. They remain live through the terminal wait;
/// being common to the default route does not remove their absolute charge.
fn common_arena_bound_bytes(cfg: &super::Qwen35Config, forward_rows: usize) -> u64 {
    let rows = forward_rows as u64;
    let h = u64::from(cfg.hidden_size);
    // Positions, layer-boundary pair, and the output ring of each FFN kind.
    let mut bytes = 16 * rows + 8 * rows * h;
    if let Some(m) = cfg.intermediate_size {
        // Three hidden activations, down projection, parameters, output ring.
        bytes += 12 * rows * u64::from(m) + 12 * rows * h + 4;
    }
    if cfg.moe.is_some() {
        bytes += 8 * rows * h;
    }
    if cfg.layer_types.contains(&super::Qwen35LayerKind::LinearAttention) {
        let keys = u64::from(cfg.linear_num_key_heads) * u64::from(cfg.linear_key_head_dim);
        let values = u64::from(cfg.linear_num_value_heads);
        let dk = u64::from(cfg.linear_key_head_dim);
        let dv = u64::from(cfg.linear_value_head_dim);
        // All 23 DnPrefillArena roots, including projection and split outputs.
        bytes += rows * (36 * keys + 24 * values * dv + 4 * h + 16 * values) + 92;
        // All seven ChunkAllocsArena roots (allocated even off the chunk path).
        bytes += rows * values * (12 * dk + 4 * dv + 4);
        // The constructor accepts only multiples of 64. Its largest possible
        // allocation below the transaction ceiling bounds every shorter chunk.
        let chunk_rows = (forward_rows / 64 * 64) as u32;
        if chunk_rows > 0 {
            bytes += mlx_native::ops::chunk_gated_delta_rule::ChunkInternalArenaPlan::for_shape(
                1,
                chunk_rows,
                cfg.linear_num_value_heads,
                cfg.linear_num_value_heads,
                cfg.linear_key_head_dim,
                cfg.linear_value_head_dim,
                64,
            )
            .map_or(0, |plan| plan.requested_bytes());
        }
    }
    // Page rounding for at most 64 common roots, separate from segment roots.
    bytes + 64 * 16 * 1024
}

/// Absolute bound of device buffers beside the pool for a joint forward
/// (also bounds a solo forward with one slot).
///
/// Per-segment attention/projection/position/gather arenas cover at most
/// `forward_rows` rows per slot, including private shared-prefix copies.
/// Resumed TQ K/V backing is shared across segments in queue order (12 bytes
/// per KV element). The fast-off bridge can overlap its pending old K/V
/// removals (12 bytes) with the next F32 pair allocated before submission
/// (8 bytes): charge 20 bytes per KV element at full cache capacity. Results
/// remain per segment; transient BF16 Q/output buffers are charged too.
/// Common arenas and plain attention results held across deferred waits are
/// included, not subtracted as default-route scratch. Stable checkpoints
/// remain under the anchor store's quota with and without the switch.
pub(crate) fn joint_arena_bound_bytes(
    cfg: &super::Qwen35Config,
    forward_rows: usize,
    slots: usize,
    kv_rows: usize,
) -> u64 {
    let rows = forward_rows as u64;
    let slots = slots.max(1) as u64;
    let lane_rows = rows * slots;
    let h = u64::from(cfg.hidden_size);
    let q = u64::from(cfg.num_attention_heads) * u64::from(cfg.head_dim);
    let kv = u64::from(cfg.num_key_value_heads) * u64::from(cfg.head_dim);
    let attention = lane_rows * (4 * (h + 5 * q + 4 * kv) + 6 * q + 8 * kv + 16);
    let resume = 20 * kv * kv_rows as u64 + 4 * q * lane_rows + 6 * q * rows;
    let gather = 4 * h * lane_rows;
    let moe = cfg.moe.as_ref().map_or(0, |moe| {
        let k = u64::from(moe.num_experts_per_tok);
        let per_row = 8 * k
            + 12 * k * u64::from(moe.moe_intermediate_size)
            + 4 * k * h
            + 12 * u64::from(moe.shared_expert_intermediate_size)
            + 4 * u64::from(moe.num_experts)
            + 4;
        (rows + 2 * slots * 32) * per_row
    });
    let channels = 2 * u64::from(cfg.linear_num_key_heads) * u64::from(cfg.linear_key_head_dim)
        + u64::from(cfg.linear_num_value_heads) * u64::from(cfg.linear_value_head_dim);
    let conv_state = 4 * channels * u64::from(cfg.linear_conv_kernel_dim.saturating_sub(1));
    let recurrent_state = 4
        * u64::from(cfg.linear_key_head_dim)
        * u64::from(cfg.linear_value_head_dim)
        * u64::from(cfg.linear_num_value_heads);
    let delta_net = (1 + slots) * conv_state + recurrent_state;
    // Large forwards wait within MAX_WAIT_LAYERS; a short (<=32-row)
    // forward can defer through every layer under the terminal-K override.
    // Count even plain results not served by the capped scratch pool.
    let held_rows = (MAX_WAIT_LAYERS as u64 * rows)
        .max(32 * u64::from(cfg.num_hidden_layers));
    let deferred_outputs = 4 * q * slots * held_rows;
    // Last-row/boundary logits held on the host, plus the current device
    // head result and normed row (heads run one row at a time).
    let head = (2 * slots + 1) * 4 * u64::from(cfg.vocab_size) + 4 * h;
    // Page rounding and parameter blocks of at most 64 segment roots per slot.
    let slack = slots * 64 * 16 * 1024;
    attention + resume + gather + moe + delta_net + deferred_outputs + head + slack
        + common_arena_bound_bytes(cfg, forward_rows)
}

/// Upper bound of what a worker holds for `HF2Q_QWEN_PREFILL_JOINT`
/// forwards of at most `forward_rows` rows over `slots` slots on a cache of
/// `kv_rows` rows per slot: what the pool keeps at a reset, its enforced
/// window threshold, one layer's requests, and all forward-owned buffers.
/// The threshold also covers short joint forwards that defer host waits.
/// Absolute: nothing of the default route is subtracted.
pub(crate) fn joint_scratch_bound_bytes(
    cfg: &super::Qwen35Config,
    forward_rows: usize,
    slots: usize,
    kv_rows: usize,
) -> u64 {
    (PREFILL_RETAINED_SCRATCH_BYTES + WAIT_LAYERS_WINDOW_SCRATCH_BYTES) as u64
        + prefill_layer_scratch_bound_bytes(cfg, forward_rows, slots)
        + joint_arena_bound_bytes(cfg, forward_rows, slots, kv_rows)
}

/// Upper bound of the scratch a worker holds for forwards of at most
/// `forward_rows` rows over `lanes` slots under
/// `HF2Q_QWEN_PREFILL_WAIT_LAYERS=wait_layers`: what the pool keeps at a
/// reset, what a window may have out when its last layer starts
/// ([`WAIT_LAYERS_WINDOW_SCRATCH_BYTES`], enforced) and that layer's
/// requests (derived); 0 at the default cadence. These are the pool bytes
/// only: [`prefill_switch_scratch_bound_bytes`] adds every forward-owned
/// arena, including those a waiting forward also allocates. With both
/// switches the caps are one pool's, not one per switch.
pub(crate) fn wait_layers_scratch_bound_bytes(
    cfg: &super::Qwen35Config,
    forward_rows: usize,
    lanes: usize,
    wait_layers: usize,
) -> u64 {
    if wait_layers > 1 {
        (PREFILL_RETAINED_SCRATCH_BYTES + WAIT_LAYERS_WINDOW_SCRATCH_BYTES) as u64
            + prefill_layer_scratch_bound_bytes(cfg, forward_rows, lanes)
    } else {
        0
    }
}

/// What memory admission charges a worker for `HF2Q_QWEN_PREFILL_JOINT`
/// (`joint`) and `HF2Q_QWEN_PREFILL_WAIT_LAYERS` (`wait_layers`) together:
/// the two switches cap one pool, so their figures do not add.
/// `forward_rows`: the worker's per-transaction row ceiling
/// (`qwen35_slot_prefill_chunk_tokens(slots)` in the engine); `kv_rows`:
/// the rows a slot's cache can hold (the resolved context length).
/// - neither: 0;
/// - joint alone: [`joint_scratch_bound_bytes`];
/// - wait layers alone: the one-lane pool bound plus its forward arenas;
/// - both: the joint pool bound plus its forward arenas.
pub(crate) fn prefill_switch_scratch_bound_bytes(
    cfg: &super::Qwen35Config,
    forward_rows: usize,
    slots: usize,
    kv_rows: usize,
    joint: bool,
    wait_layers: usize,
) -> u64 {
    match (joint, wait_layers > 1) {
        (false, false) => 0,
        (true, false) => joint_scratch_bound_bytes(cfg, forward_rows, slots, kv_rows),
        (false, true) => {
            wait_layers_scratch_bound_bytes(cfg, forward_rows, 1, wait_layers)
                + joint_arena_bound_bytes(cfg, forward_rows, 1, kv_rows)
        }
        (true, true) => {
            wait_layers_scratch_bound_bytes(cfg, forward_rows, slots, wait_layers)
                + joint_arena_bound_bytes(cfg, forward_rows, slots, kv_rows)
        }
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
