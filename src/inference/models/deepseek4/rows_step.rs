//! Exact multi-row prompt steps for DeepSeek-V4 token-wise prompt work.
//!
//! The recovery tail (the last `RECOVERY_TAIL_TOKENS` prompt tokens after the
//! rollback anchor) and cached suffixes shorter than the matrix floor run as
//! one-token verifier steps. A rows step advances one cache by up to
//! [`MAX_JOINT_DECODE_ROWS`] consecutive tokens in one layer-major
//! transaction built from the joint-decode graph: every product stays on its
//! one-token route (independent-row F16 and Q8_0 products, `mv_id` experts)
//! and the cache-dependent cores run per row, in row order, on the one cache
//! ([`RowCaches::Sequential`]). Layer `L` of token `p + 1` reads only layer-`L`
//! cache entries of tokens `<= p + 1`, so layer-major order gives every row
//! the inputs of the token-major one-token steps: each row's state and every
//! cache byte equal `rows` successive `forward_verifier_one` calls.
//!
//! `HF2Q_DEEPSEEK_TAIL_ROWS_STEP` (read once at model load, default off)
//! enables it in serving.

use anyhow::{ensure, Context, Result};
use mlx_native::MlxBuffer;

use super::cache::Deepseek4Cache;
use super::forward_support::{begin_decode_pool_token, end_decode_pool_token};
use super::joint_decode::MAX_JOINT_DECODE_ROWS;
use super::joint_decode_core::RowCaches;
use super::Deepseek4Model;

/// Serving switch for exact rows steps over token-wise prompt work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TailRowsStep {
    /// One `forward_verifier_one` transaction per token-wise prompt token.
    #[default]
    Off,
    /// Token-wise prompt tokens run as rows steps of at most
    /// [`MAX_JOINT_DECODE_ROWS`] tokens when the joint-decode routes are
    /// available for this artifact and process.
    On,
}

impl TailRowsStep {
    /// Process switch read by [`Self::from_env`].
    pub const ENV: &'static str = "HF2Q_DEEPSEEK_TAIL_ROWS_STEP";

    /// `HF2Q_DEEPSEEK_TAIL_ROWS_STEP`: `1` selects [`Self::On`], `0` or unset
    /// [`Self::Off`]; anything else is an error.
    pub fn from_env() -> Result<Self> {
        Self::resolve(std::env::var_os(Self::ENV).as_deref())
    }

    /// Resolution rule of [`Self::from_env`].
    pub fn resolve(value: Option<&std::ffi::OsStr>) -> Result<Self> {
        match value.map(|value| value.to_str()) {
            None | Some(Some("0")) => Ok(Self::Off),
            Some(Some("1")) => Ok(Self::On),
            Some(_) => anyhow::bail!("{} must be 0 or 1 (got {value:?})", Self::ENV),
        }
    }
}

/// Tokens per transaction for token-wise prompt work: `1` (one-token steps)
/// unless rows steps are enabled and available.
pub(crate) fn token_wise_step_rows(switch: TailRowsStep, unavailable: Option<&str>) -> usize {
    match (switch, unavailable) {
        (TailRowsStep::On, None) => MAX_JOINT_DECODE_ROWS,
        _ => 1,
    }
}

/// Publish a drained rows step: prevalidate the span, then the supervisor
/// gate, then advance the cursor infallibly. Any rejection poisons the cache.
fn publish_rows_step_after_gate(
    cache: &mut Deepseek4Cache,
    start: usize,
    rows: usize,
    commit_gate: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let ticket = match cache.validate_prefill_commit(start, rows) {
        Ok(ticket) => ticket,
        Err(error) => {
            cache.poison();
            return Err(error).context("prevalidate DeepSeek-V4 rows step publication");
        }
    };
    if let Err(error) = commit_gate() {
        cache.poison();
        return Err(error).context("DeepSeek-V4 rows step rejected before cache publication");
    }
    cache.publish_prefill_end(ticket);
    Ok(())
}

impl Deepseek4Model {
    /// Tokens per transaction for this model's token-wise prompt work.
    pub(crate) fn token_wise_step_rows(&self) -> usize {
        token_wise_step_rows(
            self.tail_rows_step,
            self.joint_decode_unavailable_reason().as_deref(),
        )
    }

    /// Advance `cache` by `token_ids.len()` (2..=[`MAX_JOINT_DECODE_ROWS`])
    /// consecutive tokens and return their `[rows, hc, hidden]` final states
    /// in token order (see the module doc for the exactness argument). The
    /// cache publishes every row at once after the drain and `commit_gate`;
    /// a submitted failure poisons it.
    pub(crate) fn forward_verifier_rows_step_with_commit_gate(
        &mut self,
        token_ids: &[u32],
        cache: &mut Deepseek4Cache,
        commit_gate: impl FnOnce() -> Result<()>,
    ) -> Result<MlxBuffer> {
        let rows = token_ids.len();
        ensure!(
            (2..=MAX_JOINT_DECODE_ROWS).contains(&rows),
            "DeepSeek-V4 rows step requires 2..={MAX_JOINT_DECODE_ROWS} tokens, got {rows}"
        );
        if let Some(reason) = self.joint_decode_unavailable_reason() {
            anyhow::bail!("DeepSeek-V4 rows step unavailable: {reason}");
        }
        let start = cache.position();
        let steps = (start..start + rows)
            .map(|position| {
                cache
                    .plan_step_at(position)
                    .context("plan DeepSeek-V4 rows step cache transaction")
            })
            .collect::<Result<Vec<_>>>()?;
        let (result, submitted_any) = {
            let views = vec![&*cache; rows];
            begin_decode_pool_token();
            let outcome = self.forward_verifier_rows_uncommitted(
                token_ids,
                &views,
                &steps,
                RowCaches::Sequential,
            );
            end_decode_pool_token();
            outcome
        };
        match result {
            Ok(state) => {
                publish_rows_step_after_gate(cache, start, rows, commit_gate)?;
                Ok(state)
            }
            Err(error) => {
                if submitted_any {
                    cache.poison();
                }
                Err(error).context("DeepSeek-V4 rows step transaction failed")
            }
        }
    }

    #[cfg(test)]
    pub(super) fn forward_verifier_rows_step(
        &mut self,
        token_ids: &[u32],
        cache: &mut Deepseek4Cache,
    ) -> Result<MlxBuffer> {
        self.forward_verifier_rows_step_with_commit_gate(token_ids, cache, || Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::*;

    #[test]
    fn switch_parses_zero_one_and_unset_and_rejects_anything_else() {
        assert_eq!(TailRowsStep::resolve(None).unwrap(), TailRowsStep::Off);
        assert_eq!(
            TailRowsStep::resolve(Some(OsStr::new("0"))).unwrap(),
            TailRowsStep::Off
        );
        assert_eq!(
            TailRowsStep::resolve(Some(OsStr::new("1"))).unwrap(),
            TailRowsStep::On
        );
        for bad in ["", "2", "on", "true", " 1"] {
            assert!(TailRowsStep::resolve(Some(OsStr::new(bad))).is_err(), "{bad:?}");
        }
        assert_eq!(TailRowsStep::default(), TailRowsStep::Off);
    }

    #[test]
    fn token_wise_rows_need_the_switch_and_available_joint_routes() {
        assert_eq!(token_wise_step_rows(TailRowsStep::Off, None), 1);
        assert_eq!(token_wise_step_rows(TailRowsStep::Off, Some("x")), 1);
        assert_eq!(token_wise_step_rows(TailRowsStep::On, Some("x")), 1);
        assert_eq!(
            token_wise_step_rows(TailRowsStep::On, None),
            MAX_JOINT_DECODE_ROWS
        );
        assert_eq!(MAX_JOINT_DECODE_ROWS, 8, "the recovery tail is one rows step");
    }
}
