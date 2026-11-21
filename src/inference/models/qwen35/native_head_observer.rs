//! Test-only copies of the actual projected output-head inputs, read only after
//! the head's existing checked completion. No norm or projection is replayed.
use std::cell::RefCell;

use anyhow::{ensure, Context, Result};
use mlx_native::MlxBuffer;

use super::super::prefill_joint::PlanLane;
use crate::serve::api::engine_qwen35::anchor_exactness_tests::trace;

thread_local! {
    static HEAD: RefCell<Option<Option<Vec<f32>>>> = const { RefCell::new(None) };
}

/// Bounds the diagnostic copy to this forward, including its error exits.
pub(super) struct Scope;

impl Scope {
    pub(super) fn begin() -> Result<Option<Self>> {
        if !trace::enabled() {
            return Ok(None);
        }
        HEAD.with(|cell| {
            let mut head = cell.borrow_mut();
            ensure!(head.is_none(), "nested native output-head observation");
            *head = Some(None);
            Ok(Some(Self))
        })
    }

    pub(super) fn take(&self) -> Result<Vec<f32>> {
        HEAD.with(|cell| {
            cell.borrow_mut().as_mut().and_then(Option::take)
                .context("actual output-head normalized rows were not observed")
        })
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        HEAD.with(|cell| { cell.borrow_mut().take(); });
    }
}

/// `normed` is exactly the row extent sent to the vocabulary projection.
pub(super) fn completed(normed: &MlxBuffer) -> Result<()> {
    HEAD.with(|cell| {
        if let Some(head) = cell.borrow_mut().as_mut() {
            ensure!(head.is_none(), "native output-head rows were not consumed");
            *head = Some(normed.as_logical_slice::<f32>()?.to_vec());
        }
        Ok(())
    })
}

/// Reconstruct the real lane segment, including its shared-prefix inputs,
/// while keeping just its actually projected final logit/normalized-hidden row.
#[allow(clippy::too_many_arguments)]
pub(super) fn lane(
    lane: &PlanLane,
    first: u32,
    rows: u32,
    tokens: &[u32],
    positions: &[i32],
    route: &str,
    logits: &[f32],
    vocab: usize,
    normed: &[f32],
    hidden_width: usize,
) -> Result<()> {
    let runs = lane.runs(first, rows);
    let start = runs.first().context("empty native prefill lane segment")?.first as usize;
    let mut input = Vec::with_capacity(rows as usize);
    for run in runs {
        let first = run.first as usize;
        input.extend_from_slice(&tokens[first..first + run.rows as usize]);
    }
    ensure!(logits.len() == vocab && normed.len() == hidden_width,
        "native prefill lane must export one matching projected row");
    trace::forward(lane.slot, usize::try_from(positions[start])?, &input, route,
        logits, vocab, Some(normed), hidden_width)
}
