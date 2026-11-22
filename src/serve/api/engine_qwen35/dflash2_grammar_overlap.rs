//! Read-only proposal preparation while the submitted target command executes.

use super::{
    PreparedDFlash2Grammar, Qwen35DFlash2Window, Qwen35DecodeState, Qwen35LoadedModel,
    DFLASH2_PROPOSALS_PER_LANE,
};
use crate::inference::spec_decode::dflash2::runtime::DFlash2Runtime;
use anyhow::{Context, Result};

pub(super) type PreparedBySlot = [Result<Option<PreparedDFlash2Grammar>>; 4];

const SLOT_MARKS: [(&str, &str); 4] = [
    ("grammar.slot0.simulate_begin", "grammar.slot0.simulate_end"),
    ("grammar.slot1.simulate_begin", "grammar.slot1.simulate_end"),
    ("grammar.slot2.simulate_begin", "grammar.slot2.simulate_end"),
    ("grammar.slot3.simulate_begin", "grammar.slot3.simulate_end"),
];

pub(super) fn needed(
    states: &[&mut Qwen35DecodeState],
    verifying: &[usize],
    tokens: &[Option<Vec<u32>>],
) -> bool {
    verifying.iter().any(|&index| {
        states[index].grammar_runtime.is_some()
            && tokens[index].as_ref().is_some_and(|tokens| tokens.len() > 1)
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
    qwen: &Qwen35LoadedModel,
    runtime: &DFlash2Runtime,
    states: &[&mut Qwen35DecodeState],
    verifying: &[usize],
    windows: &[Option<Qwen35DFlash2Window>],
    tokens: &[Option<Vec<u32>>],
    gpu_target_tokens: bool,
    prepared: &mut PreparedBySlot,
    collect_work: bool,
) -> Result<()> {
    // These accessors check the actual retained draft producer, not merely
    // same-queue order or a promise that the later target will complete.
    let gpu_outputs = if gpu_target_tokens {
        Some((runtime.proposals()?, runtime.verify_rows()?))
    } else {
        None
    };
    mlx_native::command_timeline::mark("grammar.simulate_begin");
    for &index in verifying {
        let state = &states[index];
        if state.grammar_runtime.is_none() {
            continue;
        }
        let slot = state.slot_id.0 as usize;
        let output = prepared
            .get_mut(slot)
            .context("DFlash2 grammar slot outside the four-lane profile")?;
        if collect_work {
            mlx_native::command_timeline::mark(SLOT_MARKS[slot].0);
        }
        // A malformed lane is rejected only after the target's checked wait;
        // other lanes still receive their own independent preparation result.
        *output = (|| {
            let window = windows[index].as_ref().context("DFlash2 grammar window")?;
            let drafts = if let Some((proposals, verify_rows)) = gpu_outputs {
                let count = *verify_rows.get(slot).context("DFlash2 grammar verify-row slot")?
                    as usize;
                anyhow::ensure!(
                    (1..=window.maximum_rows).contains(&count),
                    "DFlash2 grammar selected {count} rows outside 1..={}",
                    window.maximum_rows
                );
                let start = slot * DFLASH2_PROPOSALS_PER_LANE;
                proposals
                    .get(start..start + count - 1)
                    .context("DFlash2 grammar proposal extent")?
            } else {
                tokens[index]
                    .as_deref()
                    .context("DFlash2 grammar host tokens")?
                    .get(1..)
                    .context("DFlash2 grammar anchor")?
            };
            if drafts.is_empty() && !collect_work {
                return Ok(None);
            }
            state.prepare_dflash2_grammar(qwen, drafts, collect_work)
        })();
        if collect_work {
            mlx_native::command_timeline::mark(SLOT_MARKS[slot].1);
        }
    }
    mlx_native::command_timeline::mark("grammar.simulate_end");
    Ok(())
}
