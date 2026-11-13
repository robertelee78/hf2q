//! Isolated spike: bias proposals using the target's already-known sampler history.
//! Target verification is unchanged; unknown within-block history is not guessed.

use anyhow::{ensure, Context, Result};
use mlx_native::{
    metal::MTLSize,
    ops::encode_helpers::{as_bytes, encode_with_args, KernelArg},
    CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice,
};

pub const HISTORY_CAPACITY: usize = 64;
const MAX_LANES: usize = mlx_native::ops::dflash2_selector::DFLASH2_MAX_LANES;
const KERNEL: &str = "hf2q_dflash2_proposal_repetition";

/// Bitwise inputs also form part of a prefetched draft's identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct DraftRepetitionPrior {
    active: u32,
    penalties: [u32; MAX_LANES],
    counts: [u32; MAX_LANES],
    history: [[u32; HISTORY_CAPACITY]; MAX_LANES],
}

impl Default for DraftRepetitionPrior {
    fn default() -> Self {
        Self {
            active: 0,
            penalties: [1.0_f32.to_bits(); MAX_LANES],
            counts: [0; MAX_LANES],
            history: [[0; HISTORY_CAPACITY]; MAX_LANES],
        }
    }
}

impl DraftRepetitionPrior {
    pub fn set_lane(
        &mut self,
        lane: usize,
        penalty: f32,
        history: impl DoubleEndedIterator<Item = u32>,
    ) -> Result<()> {
        ensure!(lane < MAX_LANES, "draft repetition lane out of range");
        ensure!(
            penalty.is_finite() && penalty > 0.0,
            "invalid draft repetition penalty"
        );
        self.active &= !(1 << lane);
        self.penalties[lane] = penalty.to_bits();
        self.counts[lane] = 0;
        self.history[lane].fill(0);
        if penalty == 1.0 {
            return Ok(());
        }
        for token in history.rev().take(HISTORY_CAPACITY) {
            let count = self.counts[lane] as usize;
            self.history[lane][count] = token;
            self.counts[lane] += 1;
        }
        self.history[lane][..self.counts[lane] as usize].reverse();
        if self.counts[lane] != 0 {
            self.active |= 1 << lane;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Params {
    lanes: u32,
    rows_per_lane: u32,
    vocabulary: u32,
}

pub(super) fn encode(
    encoder: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    logits: &MlxBuffer,
    lanes: usize,
    rows_per_lane: usize,
    vocabulary: usize,
    prior: &DraftRepetitionPrior,
) -> Result<()> {
    ensure!(
        (1..=MAX_LANES).contains(&lanes),
        "invalid draft repetition lanes"
    );
    ensure!(
        (2..=16).contains(&rows_per_lane),
        "invalid draft repetition rows"
    );
    ensure!(
        prior.active >> lanes == 0,
        "draft repetition addresses an absent lane"
    );
    ensure!(
        logits.dtype() == DType::F32
            && logits.element_count() == lanes * rows_per_lane * vocabulary,
        "draft repetition logits geometry mismatch"
    );
    ensure!(
        u32::try_from(logits.element_count()).is_ok(),
        "draft repetition index overflow"
    );
    if prior.active == 0 {
        return Ok(());
    }
    for lane in 0..lanes {
        ensure!(
            prior.history[lane][..prior.counts[lane] as usize]
                .iter()
                .all(|&token| (token as usize) < vocabulary),
            "draft repetition history token outside vocabulary"
        );
    }
    let params = Params {
        lanes: lanes as u32,
        rows_per_lane: rows_per_lane as u32,
        vocabulary: vocabulary as u32,
    };
    registry.register_source(KERNEL, include_str!("shaders/proposal_repetition.metal"));
    let pipeline = registry
        .get_pipeline(KERNEL, device.metal_device())
        .context("build draft repetition prior")?;
    encode_with_args(
        encoder,
        pipeline,
        &[
            (0, KernelArg::Buffer(logits)),
            (1, KernelArg::Bytes(as_bytes(prior))),
            (2, KernelArg::Bytes(as_bytes(&params))),
        ],
        MTLSize::new(
            HISTORY_CAPACITY as u64,
            (rows_per_lane - 1) as u64,
            lanes as u64,
        ),
        MTLSize::new(HISTORY_CAPACITY as u64, 1, 1),
    );
    encoder.memory_barrier();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposal_repetition_changes_each_history_token_once_without_cross_lane_or_anchor_writes(
    ) -> Result<()> {
        let _lock = crate::inference::hf2q_gpu_test_lock();
        let device = MlxDevice::new()?;
        let mut registry = KernelRegistry::new();
        let lanes = 4;
        let rows = 8;
        let vocabulary = 97;
        let original = (0..lanes * rows * vocabulary)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.25)
            .collect::<Vec<_>>();
        let mut logits = device.alloc_buffer(
            original.len() * 4,
            DType::F32,
            vec![lanes, rows, vocabulary],
        )?;
        let mut long_history = (0..70_u32).collect::<Vec<_>>();
        long_history.extend([70, 70, 96]);
        let histories: [&[u32]; 4] = [&long_history, &[0, 45, 45], &[], &[1, 96, 1]];
        let penalties = [1.05_f32, 0.8, 2.0, 1.0];
        let mut prior = DraftRepetitionPrior::default();
        let mut expected = original.clone();
        for lane in 0..lanes {
            prior.set_lane(lane, penalties[lane], histories[lane].iter().copied())?;
            let history =
                &histories[lane][histories[lane].len().saturating_sub(HISTORY_CAPACITY)..];
            for row in 1..rows {
                for (index, &token) in history.iter().enumerate() {
                    if history[..index].contains(&token) {
                        continue;
                    }
                    let value = &mut expected[(lane * rows + row) * vocabulary + token as usize];
                    let real = f64::from(*value);
                    let penalty = f64::from(penalties[lane]);
                    *value = if real > 0.0 {
                        real / penalty
                    } else {
                        real * penalty
                    } as f32;
                }
            }
        }
        let mut first_bits = None;
        for _ in 0..2 {
            logits.as_mut_slice::<f32>()?.copy_from_slice(&original);
            let mut encoder = device.command_encoder()?;
            encode(
                &mut encoder,
                &mut registry,
                &device,
                &logits,
                lanes,
                rows,
                vocabulary,
                &prior,
            )?;
            encoder.commit_and_wait()?;
            let actual = logits.as_slice::<f32>()?;
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 2.0e-6,
                    "logit {index}: {actual} != {expected}"
                );
            }
            if let Some(bits) = &first_bits {
                assert_eq!(logits.as_slice::<u32>()?, bits);
            } else {
                first_bits = Some(logits.as_slice::<u32>()?.to_vec());
            }
        }
        Ok(())
    }
}
