//! Session ledger half of DeepSeek-V4 joint decode: select the slot caches,
//! run one supervised joint transaction, then publish each slot's token and
//! next-token logits exactly as the serial `commit_generated_token` does.

use anyhow::{Context, Result};
use mlx_native::MlxBuffer;

use crate::inference::models::deepseek4::cache::Deepseek4Cache;

use super::super::engine_supervisor::EngineSupervisor;
use super::{Deepseek4LoadedModel, Deepseek4Session, GPU_TRANSACTION_TIMEOUT};

impl Deepseek4Session {
    /// Whether this slot can join a joint decode step: its cache cursor
    /// matches its token ledger, it holds next-token logits, and its cache can
    /// plan one more step (not poisoned, within capacity and graft bounds).
    pub(in crate::serve::api) fn joint_decode_ready(&self) -> bool {
        self.cache.position() == self.committed_tokens.len()
            && self.live_logits.is_some()
            && self.cache.plan_next_step().is_ok()
    }
}

impl Deepseek4LoadedModel {
    /// Advance several slots by one sampled token each in one joint
    /// transaction. `lanes` pairs strictly increasing slot indices with each
    /// slot's token; the result holds each lane's `[1, vocab]` logits in the
    /// same order. Failure leaves every lane's ledger unchanged; a submitted
    /// failure poisons every lane's cache.
    pub(in crate::serve::api) fn commit_generated_tokens_joint(
        &mut self,
        lanes: &[(usize, u32)],
        sessions: &mut [Deepseek4Session],
        supervisor: &EngineSupervisor,
    ) -> Result<Vec<MlxBuffer>> {
        anyhow::ensure!(
            lanes.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "DeepSeek-V4 joint decode slot indices must be unique and sorted"
        );
        let mut wanted = lanes.iter().copied().peekable();
        let mut selected = Vec::with_capacity(lanes.len());
        let mut token_ids = Vec::with_capacity(lanes.len());
        for (index, session) in sessions.iter_mut().enumerate() {
            let Some(&(slot, token)) = wanted.peek() else {
                break;
            };
            if slot != index {
                continue;
            }
            anyhow::ensure!(
                session.cache.position() == session.committed_tokens.len(),
                "DeepSeek-V4 joint decode slot {index} cache position {} disagrees with token ledger {}",
                session.cache.position(),
                session.committed_tokens.len()
            );
            session
                .committed_tokens
                .try_reserve(1)
                .context("reserve DeepSeek-V4 joint decode token ledger")?;
            selected.push(session);
            token_ids.push(token);
            wanted.next();
        }
        anyhow::ensure!(
            wanted.next().is_none() && selected.len() == lanes.len(),
            "DeepSeek-V4 joint decode slot index is out of range"
        );

        let lease = supervisor.arm("deepseek4_joint_decode", GPU_TRANSACTION_TIMEOUT)?;
        let state = {
            let mut caches = selected
                .iter_mut()
                .map(|session| &mut session.cache)
                .collect::<Vec<&mut Deepseek4Cache>>();
            self.model
                .forward_verifier_decode_joint_with_commit_gate(&token_ids, &mut caches, || {
                    lease.finish()
                })
                .context("execute DeepSeek-V4 joint decode verifier")?
        };
        for (session, &token) in selected.iter_mut().zip(&token_ids) {
            session.committed_tokens.push(token);
        }
        let logits = supervisor.run(
            "deepseek4_joint_decode_head",
            GPU_TRANSACTION_TIMEOUT,
            || {
                self.model
                    .forward_logits(&state)
                    .context("execute DeepSeek-V4 joint decode output head")
            },
        )?;
        let vocab = self.model.cfg.vocab_size as usize;
        anyhow::ensure!(
            logits.dtype() == mlx_native::DType::F32 && logits.shape() == [lanes.len(), vocab],
            "DeepSeek-V4 joint decode logits must be F32 [{}, {vocab}], got {} {:?}",
            lanes.len(),
            logits.dtype(),
            logits.shape()
        );
        let row_bytes = vocab
            .checked_mul(mlx_native::DType::F32.size_of())
            .context("DeepSeek-V4 joint decode logits row overflow")?;
        let rows = (0..lanes.len())
            .map(|lane| {
                logits
                    .slice_view((lane * row_bytes) as u64, vocab)
                    .with_shape(vec![1, vocab])
                    .context("view DeepSeek-V4 joint decode logits row")
            })
            .collect::<Result<Vec<_>>>()?;
        for (session, row) in selected.iter_mut().zip(&rows) {
            session.live_logits = Some(row.clone());
        }
        Ok(rows)
    }
}
