//! Generation-cue fold for the bounded prompt prefill
//! (`HF2Q_QWEN_PREFILL_FOLD_CUE=1`).
//!
//! A served prompt ends in a short generation cue. The rows before it are the
//! stable prompt prefix, and the prompt cache keeps a checkpoint exactly
//! there: the slot's DeltaNet state, its cursors and the logits of the last
//! prefix row. The bounded prefill therefore stops at that boundary, takes
//! the checkpoint, and runs the cue as a second forward of a few rows, which
//! streams every layer's weights a second time.
//!
//! The fold runs the last content chunk and the cue as one forward and takes
//! the checkpoint inside it. Every operation whose output row depends on that
//! input row alone (norms, the prepared row-tile projections, the dense FFN,
//! the elementwise stages) runs once over all rows. The operations that mix
//! rows run as the two forwards would have run them:
//!
//! - DeltaNet convolution and recurrence are dispatched twice, first over the
//!   prefix rows from the slot state into the checkpoint image, then over the
//!   cue rows from the checkpoint image into the slot state.
//! - A full-attention layer and a MoE FFN layer are built twice, once per
//!   segment, with the row count, cursor and scratch each forward would have
//!   had.
//! - The output head projects the last prefix row and the last row.
//!
//! Each segment therefore runs the kernels, parameters and inputs of its own
//! forward, and the slot state, the checkpoint, the captured target layers and
//! both logit rows are bit-identical to the two-forward path
//! (`prefill_fold_tests`). That holds only while every shared product is row
//! exact, so the fold is taken only inside a routed
//! [`super::prefill_prepared::PrefillScope`] and only for models whose shared
//! products all take the prepared kernel ([`Qwen35Model::prefill_fold_exact`]).
//!
//! The forward side is [`super::prefill_joint`]: a folded chunk is a joint
//! prefill of one lane with a boundary. This module keeps the switch and the
//! shapes the fold serves.
//!
//! [`Qwen35Model::prefill_fold_exact`]: super::model::Qwen35Model

use super::joint_step_options::parse_switch;
use crate::inference::spec_decode::dflash2::route_profile;

/// Switch, default off: `1`/`on`/`true` folds the generation cue into the
/// last content chunk of a bounded prompt prefill.
pub(crate) const QWEN_PREFILL_FOLD_CUE_ENV: &str = "HF2Q_QWEN_PREFILL_FOLD_CUE";

/// Longest cue the fold accepts. The served cues are four to eight tokens;
/// the bound keeps the second segment on the short-chunk kernels the tests
/// cover.
pub(crate) const MAX_CUE_ROWS: usize = 32;

/// The switch as resolved for the model this thread serves. An invalid value
/// warns once and keeps the fold off.
fn resolved() -> bool {
    const SWITCH: route_profile::Switch = route_profile::switch(QWEN_PREFILL_FOLD_CUE_ENV);
    route_profile::with_value(SWITCH, |raw| parse_switch(QWEN_PREFILL_FOLD_CUE_ENV, raw))
        .unwrap_or_else(|error| {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(error, fallback = false, "invalid Qwen cue-fold switch");
            });
            false
        })
}

/// Whether the operator switched the fold on.
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(switches) = super::prefill_prepared::SwitchesForTest::current() {
        return switches.fold_cue;
    }
    resolved()
}

/// Forward routes that have no folded form: with any of them a prompt
/// prefill keeps its two forwards. GLP steering writes the residual stream
/// between layers, the dump-bisect instrument downloads per-layer tensors,
/// and the dense-Q split profile commits each FFN stage on its own; none of
/// the three is split at the prompt boundary. And a forward that does not
/// recycle its pooled scratch at a layer's host wait may not pool a folded
/// or joint forward's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UnfoldedRoutes {
    pub(crate) glp_bound: bool,
    pub(crate) dump_bisect: bool,
    pub(crate) dense_q_split_profile: bool,
    /// `HF2Q_MM_ID_ROUTING_THRESHOLD` is set (any value, as for the shared
    /// expert schedule in `gpu_ffn`): mlx-native may change the MoE FFN's
    /// kernel family at another row count than the 32 rows a folded or
    /// joint forward groups its MoE segments by. Concerns MoE models only
    /// and is therefore not part of [`Self::any`].
    pub(crate) moe_route_override: bool,
    /// `HF2Q_DENSE_Q_ARENA_RESET=0`: a forward does not take its pooled
    /// scratch back at a layer's host wait. A folded or joint forward pools
    /// per-segment scratch and runs under pool caps that rely on that
    /// reset, so neither is taken.
    pub(crate) scratch_reset_off: bool,
}

/// Whether the process overrides the row count at which the MoE FFN changes
/// kernel family ([`UnfoldedRoutes::moe_route_override`]).
pub(crate) fn moe_route_override() -> bool {
    static OVERRIDE: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("HF2Q_MM_ID_ROUTING_THRESHOLD").is_some());
    *OVERRIDE
}

impl UnfoldedRoutes {
    pub(crate) fn any(self) -> bool {
        self.glp_bound
            || self.dump_bisect
            || self.dense_q_split_profile
            || self.scratch_reset_off
    }
}

/// Whether a chunk of `prefix_rows` content rows followed by `cue_rows` cue
/// rows has the shape the fold serves: both segments are multi-row forwards
/// (a one-row forward is the decode graph) and the cue is short.
pub(crate) fn rows_admitted(prefix_rows: usize, cue_rows: usize) -> bool {
    prefix_rows >= 2 && (2..=MAX_CUE_ROWS).contains(&cue_rows)
}

/// Whether folding these rows can pay. A content chunk of at most 32 rows
/// runs on the short-chunk wait cadence and one row tile; a fold that
/// carries it past 32 rows waits after every layer and starts a second tile
/// (measured about 10 ms slower per turn on Qwen3.8), so such a chunk keeps
/// its two forwards.
pub(crate) fn rows_worth_folding(prefix_rows: usize, cue_rows: usize) -> bool {
    const SHORT_CHUNK_ROWS: usize = 32;
    prefix_rows > SHORT_CHUNK_ROWS || prefix_rows + cue_rows <= SHORT_CHUNK_ROWS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::joint_step_options::parse_switch;

    #[test]
    fn switch_is_off_unless_set_on() {
        let parse = |raw| parse_switch(QWEN_PREFILL_FOLD_CUE_ENV, raw);
        assert_eq!(parse(None), Ok(false));
        assert_eq!(parse(Some(" ")), Ok(false));
        assert_eq!(parse(Some("on")), Ok(true));
        assert_eq!(parse(Some("0")), Ok(false));
        assert!(parse(Some("cue")).is_err());
    }

    #[test]
    fn a_short_chunk_is_not_folded_past_the_short_chunk_cadence() {
        assert!(rows_worth_folding(720, 5));
        assert!(
            rows_worth_folding(157, 5),
            "a new tile, but one forward less"
        );
        assert!(rows_worth_folding(33, 5));
        assert!(rows_worth_folding(27, 5), "32 rows: still a short chunk");
        assert!(!rows_worth_folding(28, 5));
        assert!(!rows_worth_folding(31, 5));
        assert!(!rows_worth_folding(32, 2));
    }

    #[test]
    fn only_two_multi_row_segments_with_a_short_cue_fold() {
        assert!(rows_admitted(720, 5));
        assert!(rows_admitted(2, 2));
        assert!(rows_admitted(31, MAX_CUE_ROWS));
        assert!(!rows_admitted(1, 5), "a one-row prefix is the decode graph");
        assert!(!rows_admitted(720, 1), "a one-row cue is the decode graph");
        assert!(!rows_admitted(720, 0));
        assert!(!rows_admitted(720, MAX_CUE_ROWS + 1));
    }
}
