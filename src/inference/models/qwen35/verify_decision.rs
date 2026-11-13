//! GPU-side candidates of a DFlash2 verify row's greedy decision
//! (`HF2Q_DFLASH2_GPU_DECISIONS`).
//!
//! A verify step whose lanes sample on the host (a repetition penalty, a
//! thinking budget, a lazy tool grammar) copies 32 rows of 248,320 logits
//! out of the head's buffer and scans each row once for the penalty and
//! once for the argmax, all while the GPU idles. For a request at
//! temperature zero whose grammar is absent or still waiting for its
//! trigger, the sampler's result for a row is
//! `sample_greedy(apply_repetition_penalty(logits, history))`, and the
//! penalty changes at most [`DECISION_SLOTS`] logits. So the head's command
//! also
//!
//! 1. gathers, per row, the logits of the row's history tokens (the lane's
//!    repetition window after the window's earlier drafts were accepted),
//! 2. masks those tokens in the logits with the lowest finite float (one
//!    writer per logits cell: a token that repeats in a row's history is
//!    written by the thread of its first slot only), and
//! 3. runs the stock argmax over the masked row,
//!
//! and the host finishes the decision from at most 65 candidates with the
//! sampler's own functions ([`JointRowDecision::greedy_token`]). The GPU
//! performs no arithmetic on a logit: it copies and compares. The result is
//! the host sampler's token whenever the candidates decide; a row they
//! cannot decide (a non-finite logit, a history that is not the one the
//! kernel assumed, an active grammar, logprobs) takes the host sampler over
//! the row's own logits ([`JointRowDecision::logits`]), which are the head's
//! bits again once the gathered values are written back.
//!
//! The kernels live in `shaders/verify_decision.metal` and are registered
//! with the model's registry on first use.

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::metal::MTLSize;
use mlx_native::ops::argmax::dispatch_argmax_f32;
use mlx_native::ops::encode_helpers::{as_bytes, encode_with_args, KernelArg};
use mlx_native::{CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::decode_pool::pooled_alloc_buffer;
use super::joint_decode_core::JointWindow;
use crate::serve::sampler_pure::{apply_repetition_penalty, sample_greedy};

/// History tokens a row's decision can carry: the sampler's repetition
/// window must fit.
pub(crate) const DECISION_SLOTS: usize = 64;

const SHADER: &str = include_str!("shaders/verify_decision.metal");
const GATHER: &str = "qwen_verify_decision_gather";
const MASK: &str = "qwen_verify_decision_mask";
const UNUSED: u32 = u32::MAX;
/// What the mask kernel leaves at a history token: `-FLT_MAX`.
const MASKED: f32 = f32::MIN;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    rows: u32,
    vocab: u32,
    slots: u32,
    row_token_count: u32,
}

/// Buffers of one step's decision dispatches.
pub(super) struct EncodedDecisions {
    history_tokens: MlxBuffer,
    history_logits: MlxBuffer,
    outside_tokens: MlxBuffer,
    outside_logits: MlxBuffer,
    /// Per real row: whether it carries a decision.
    decided: Vec<bool>,
}

/// Encode the decision candidates of every row of the windows that have a
/// base history (`histories[i]` belongs to `windows[i]`). `logits` is the
/// head's `[width, vocab]` F32 output, already ordered before this by a
/// barrier; `row_tokens` holds the input token of every packed row. The
/// history tokens of the decided rows are left masked in `logits`.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    logits: &MlxBuffer,
    windows: &[JointWindow],
    histories: &[Option<&[u32]>],
    row_tokens: &MlxBuffer,
    rows: usize,
    vocab: usize,
) -> Result<EncodedDecisions> {
    ensure!(
        histories.len() == windows.len(),
        "Qwen verify decisions: {} histories for {} windows",
        histories.len(),
        windows.len()
    );
    ensure!(
        logits.dtype() == DType::F32
            && logits.element_count() >= rows * vocab
            && row_tokens.dtype() == DType::U32
            && u32::try_from(rows * vocab).is_ok(),
        "Qwen verify decisions: logits {:?} x {} or row tokens {:?} do not fit {rows} rows of {vocab}",
        logits.dtype(),
        logits.element_count(),
        row_tokens.dtype()
    );
    let mut table = vec![UNUSED; 4 * rows];
    let mut base = Vec::new();
    let mut decided = vec![false; rows];
    for (window, history) in windows.iter().zip(histories) {
        let Some(history) = history else {
            continue;
        };
        ensure!(
            history.len() <= DECISION_SLOTS && window.start + window.rows <= rows,
            "Qwen verify decisions: a history of {} tokens or a window past row {rows}",
            history.len()
        );
        // Row r reads the inputs of rows 1..=r of its window.
        ensure!(
            window.start + window.rows <= row_tokens.element_count(),
            "Qwen verify decisions: the row tokens end inside a window"
        );
        let offset = base.len() as u32;
        base.extend_from_slice(history);
        for r in 0..window.rows {
            let row = window.start + r;
            table[4 * row..4 * row + 4].copy_from_slice(&[
                window.start as u32,
                r as u32,
                offset,
                history.len() as u32,
            ]);
            decided[row] = true;
        }
    }
    // Metal does not bind an empty buffer.
    base.push(UNUSED);
    let words = |values: &[u32], what: &str| -> Result<MlxBuffer> {
        let mut buffer = pooled_alloc_buffer(
            device,
            std::mem::size_of_val(values),
            DType::U32,
            vec![values.len()],
        )
        .map_err(|e| anyhow!("allocate Qwen verify decision {what}: {e}"))?;
        buffer
            .as_mut_slice::<u32>()
            .map_err(|e| anyhow!("{e}"))?
            .copy_from_slice(values);
        Ok(buffer)
    };
    let table = words(&table, "row table")?;
    let base = words(&base, "base histories")?;
    let slots = rows * DECISION_SLOTS;
    let scratch = |dtype: DType, elements: usize, what: &str| {
        pooled_alloc_buffer(device, elements * 4, dtype, vec![elements])
            .map_err(|e| anyhow!("allocate Qwen verify decision {what}: {e}"))
    };
    let history_tokens = scratch(DType::U32, slots, "history tokens")?;
    let history_logits = scratch(DType::F32, slots, "history logits")?;
    let outside_tokens = scratch(DType::U32, rows, "argmax tokens")?;
    let outside_logits = scratch(DType::F32, rows, "argmax logits")?;
    let params = Params {
        rows: rows as u32,
        vocab: vocab as u32,
        slots: DECISION_SLOTS as u32,
        row_token_count: u32::try_from(row_tokens.element_count())
            .context("Qwen verify decisions: row token count")?,
    };
    let grid = MTLSize::new(DECISION_SLOTS as u64, rows as u64, 1);
    let group = MTLSize::new(DECISION_SLOTS as u64, 1, 1);
    registry.register_source(GATHER, SHADER);
    registry.register_source(MASK, SHADER);
    let gather = registry
        .get_pipeline(GATHER, device.metal_device())
        .context("Qwen verify decision gather pipeline")?;
    ensure!(
        gather.max_total_threads_per_threadgroup() >= DECISION_SLOTS as u64,
        "Qwen verify decision kernels need {DECISION_SLOTS} threads per threadgroup"
    );
    encode_with_args(
        enc,
        gather,
        &[
            (0, KernelArg::Buffer(logits)),
            (1, KernelArg::Buffer(&table)),
            (2, KernelArg::Buffer(&base)),
            (3, KernelArg::Buffer(row_tokens)),
            (4, KernelArg::Buffer(&history_tokens)),
            (5, KernelArg::Buffer(&history_logits)),
            (6, KernelArg::Bytes(as_bytes(&params))),
        ],
        grid,
        group,
    );
    enc.memory_barrier();
    let mask = registry
        .get_pipeline(MASK, device.metal_device())
        .context("Qwen verify decision mask pipeline")?;
    ensure!(
        mask.max_total_threads_per_threadgroup() >= DECISION_SLOTS as u64,
        "Qwen verify decision kernels need {DECISION_SLOTS} threads per threadgroup"
    );
    encode_with_args(
        enc,
        mask,
        &[
            (0, KernelArg::Buffer(logits)),
            (1, KernelArg::Buffer(&history_tokens)),
            (2, KernelArg::Bytes(as_bytes(&params))),
        ],
        grid,
        group,
    );
    enc.memory_barrier();
    let argmax_params = words(&[vocab as u32], "argmax params")?;
    for row in (0..rows).filter(|&row| decided[row]) {
        let view = logits
            .slice_view((row * vocab * 4) as u64, vocab)
            .with_shape(vec![1, vocab])
            .map_err(|e| anyhow!("shape Qwen verify decision row {row}: {e}"))?;
        dispatch_argmax_f32(
            enc,
            registry,
            device.metal_device(),
            &view,
            &outside_tokens.slice_view((row * 4) as u64, 1),
            &outside_logits.slice_view((row * 4) as u64, 1),
            &argmax_params,
            vocab as u32,
        )
        .with_context(|| format!("Qwen verify decision argmax row {row}"))?;
    }
    Ok(EncodedDecisions {
        history_tokens,
        history_logits,
        outside_tokens,
        outside_logits,
        decided,
    })
}

impl EncodedDecisions {
    /// Row `row`'s candidates, once the command completed; `None` for a row
    /// without a decision. `logits_row` is the row's `[1, vocab]` view of
    /// the step's logits.
    pub(super) fn read(
        &self,
        row: usize,
        logits_row: MlxBuffer,
    ) -> Result<Option<JointRowDecision>> {
        if !self.decided.get(row).copied().unwrap_or(false) {
            return Ok(None);
        }
        let slots = row * DECISION_SLOTS..(row + 1) * DECISION_SLOTS;
        let tokens = &self
            .history_tokens
            .as_slice::<u32>()
            .map_err(|e| anyhow!("read Qwen verify decision history tokens: {e}"))?[slots.clone()];
        let logits = &self
            .history_logits
            .as_slice::<f32>()
            .map_err(|e| anyhow!("read Qwen verify decision history logits: {e}"))?[slots];
        let mut decision = JointRowDecision {
            outside_token: self
                .outside_tokens
                .as_slice::<u32>()
                .map_err(|e| anyhow!("read Qwen verify decision argmax tokens: {e}"))?[row],
            outside_logit: self
                .outside_logits
                .as_slice::<f32>()
                .map_err(|e| anyhow!("read Qwen verify decision argmax logits: {e}"))?[row],
            history_tokens: [UNUSED; DECISION_SLOTS],
            history_logits: [0.0; DECISION_SLOTS],
            history_len: tokens.iter().take_while(|&&token| token != UNUSED).count(),
            row: logits_row,
        };
        decision.history_tokens.copy_from_slice(tokens);
        decision.history_logits.copy_from_slice(logits);
        Ok(Some(decision))
    }
}

/// The candidates of one verify row's greedy decision.
pub(crate) struct JointRowDecision {
    /// Argmax over the row with its history tokens masked: the largest
    /// logit outside the history, at its lowest token id. `u32::MAX` with a
    /// NaN logit when the masked row holds a non-finite value.
    outside_token: u32,
    outside_logit: f32,
    /// The row's history as the kernel assembled it, oldest first, and the
    /// head's logit of each token: the first `history_len` slots.
    history_tokens: [u32; DECISION_SLOTS],
    history_logits: [f32; DECISION_SLOTS],
    history_len: usize,
    /// The row's `[1, vocab]` logits, masked at the history tokens. A view
    /// of a pooled buffer: readable until the next joint forward.
    row: MlxBuffer,
}

impl JointRowDecision {
    /// The token `sampler_pure::sample_token` returns at temperature zero
    /// for this row's logits with `penalty` over `history` (the sampler's
    /// `previous_tokens`), or `None` when the candidates cannot decide it:
    /// `history` is not the list the kernel assembled, a candidate is not
    /// finite, or a history token is outside the vocabulary. The caller
    /// then samples from [`Self::logits`].
    ///
    /// Why it is the sampler's token: the penalty changes only the history
    /// tokens' logits, so over the other tokens the sampler's maximum and
    /// its lowest index are the masked argmax; the sampler's result is the
    /// lowest token id among the largest of that value and the penalised
    /// history logits. The penalty and the selection below are the
    /// sampler's functions applied to those candidates in token-id order,
    /// held in fixed arrays (this function itself allocates nothing).
    pub(crate) fn greedy_token(&self, history: &[u32], penalty: f64) -> Option<u32> {
        /// Candidates: every history slot and the token outside the history.
        const CANDIDATES: usize = DECISION_SLOTS + 1;
        /// Positions of the candidates, the "token ids" the penalty is
        /// applied to in the compact row.
        const POSITIONS: [u32; CANDIDATES] = {
            let mut positions = [0; CANDIDATES];
            let mut index = 0;
            while index < CANDIDATES {
                positions[index] = index as u32;
                index += 1;
            }
            positions
        };
        let vocab = self.row.element_count();
        let (history_tokens, history_logits) = self.history();
        if history_tokens != history
            || self.outside_token as usize >= vocab
            || !self.outside_logit.is_finite()
            || self.outside_logit <= MASKED
        {
            return None;
        }
        // Unique history tokens with their logits, in token-id order (an
        // insertion sort over at most 64 entries, on the stack).
        let mut tokens = [0_u32; CANDIDATES];
        let mut logits = [0.0_f32; CANDIDATES];
        let mut count = 0;
        for (&token, &logit) in history_tokens.iter().zip(history_logits) {
            if token as usize >= vocab || !logit.is_finite() || token == self.outside_token {
                return None;
            }
            let at = tokens[..count].partition_point(|&earlier| earlier < token);
            if at < count && tokens[at] == token {
                continue;
            }
            tokens.copy_within(at..count, at + 1);
            logits.copy_within(at..count, at + 1);
            tokens[at] = token;
            logits[at] = logit;
            count += 1;
        }
        if penalty != 1.0 && !history.is_empty() {
            apply_repetition_penalty(&mut logits[..count], &POSITIONS[..count], penalty);
        }
        let at = tokens[..count].partition_point(|&token| token < self.outside_token);
        tokens.copy_within(at..count, at + 1);
        logits.copy_within(at..count, at + 1);
        tokens[at] = self.outside_token;
        logits[at] = self.outside_logit;
        count += 1;
        if logits[..count].iter().any(|logit| logit.is_nan()) {
            return None;
        }
        Some(tokens[sample_greedy(&logits[..count]) as usize])
    }

    /// The row's history as the kernel assembled it, oldest first, and the
    /// head's logit of each of its tokens.
    pub(crate) fn history(&self) -> (&[u32], &[f32]) {
        (
            &self.history_tokens[..self.history_len],
            &self.history_logits[..self.history_len],
        )
    }

    /// The row's logits as the head wrote them: the buffer's row with the
    /// gathered values back at the history tokens.
    pub(crate) fn logits(&self) -> Result<Vec<f32>> {
        let mut logits = self
            .row
            .as_slice::<f32>()
            .map_err(|e| anyhow!("read Qwen verify decision row: {e}"))?
            .to_vec();
        self.restore(&mut logits);
        Ok(logits)
    }

    /// Undo the mask in a copy of the row.
    pub(super) fn restore(&self, logits: &mut [f32]) {
        let (tokens, raw) = self.history();
        for (&token, &logit) in tokens.iter().zip(raw) {
            if let Some(slot) = logits.get_mut(token as usize) {
                *slot = logit;
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        outside: (u32, f32),
        history_tokens: &[u32],
        history_logits: &[f32],
        row: MlxBuffer,
    ) -> Self {
        let mut decision = Self {
            outside_token: outside.0,
            outside_logit: outside.1,
            history_tokens: [UNUSED; DECISION_SLOTS],
            history_logits: [0.0; DECISION_SLOTS],
            history_len: history_tokens.len(),
            row,
        };
        decision.history_tokens[..history_tokens.len()].copy_from_slice(history_tokens);
        decision.history_logits[..history_logits.len()].copy_from_slice(history_logits);
        decision
    }
}

/// The candidates of every row of a `[width, vocab]` logits tile, through
/// the kernels: `windows` are `(first packed row, rows)`, `histories` their
/// base histories, `row_tokens` the packed rows' input tokens. For tests of
/// the callers' use of the candidates.
#[cfg(test)]
pub(crate) fn candidates_for_test(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    vocab: usize,
    windows: &[(usize, usize)],
    histories: &[Option<&[u32]>],
    row_tokens: &[u32],
    logits: &[f32],
) -> Vec<Option<JointRowDecision>> {
    super::decode_pool::checked_reset_decode_pool().unwrap();
    let windows = windows
        .iter()
        .enumerate()
        .map(|(slot, &(start, rows))| JointWindow {
            slot: crate::serve::multi_seq_kv::SlotId(slot as u32),
            start,
            rows,
        })
        .collect::<Vec<_>>();
    let rows = windows
        .iter()
        .map(|window| window.start + window.rows)
        .max()
        .unwrap();
    let width = logits.len() / vocab;
    let mut tile = device
        .alloc_buffer(logits.len() * 4, DType::F32, vec![width, vocab])
        .unwrap();
    tile.as_mut_slice::<f32>().unwrap().copy_from_slice(logits);
    let mut tokens = device
        .alloc_buffer(row_tokens.len() * 4, DType::U32, vec![row_tokens.len()])
        .unwrap();
    tokens
        .as_mut_slice::<u32>()
        .unwrap()
        .copy_from_slice(row_tokens);
    let mut enc = device.command_encoder().unwrap();
    let encoded = encode(
        &mut enc, registry, device, &tile, &windows, histories, &tokens, rows, vocab,
    )
    .unwrap();
    enc.commit_and_wait().unwrap();
    (0..rows)
        .map(|row| {
            let view = tile
                .slice_view((row * vocab * 4) as u64, vocab)
                .with_shape(vec![1, vocab])
                .unwrap();
            encoded.read(row, view).unwrap()
        })
        .collect()
}

#[cfg(test)]
#[path = "verify_decision_tests.rs"]
mod tests;
