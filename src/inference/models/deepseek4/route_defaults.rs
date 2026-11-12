//! DeepSeek-V4 default routes (v0.1.30).
//!
//! The three routes of the DeepSeek-V4 release gate (ADR-063 Q4-DS4-RACE,
//! campaign `q4-deepseek4-ds4file-release-gate-v4`) are the defaults:
//! row-exact few-row products, rows steps over token-wise prompt work and
//! batched joint attention cores. Each is bitwise equal to the route it
//! replaces, so the defaults change dispatch counts and never an output
//! bit. Each switch keeps `0` as its kill switch and `1` as the explicit
//! selection; any other value fails the model load, as before.
//!
//! None of the three can refuse an artifact. A product, layer or step the
//! route does not admit keeps its previous route:
//! - row-exact products: `forward_support::row_exact_storage` (storage type
//!   and the stock Q8_0 route of the model's routing policy);
//! - rows steps: `rows_step::token_wise_step_rows` (one-token steps while
//!   joint decode is unavailable for this artifact or process);
//! - batched cores: `joint_decode_rows::batched_cores_admit` (per layer and
//!   step).

use std::ffi::OsStr;

use anyhow::Result;

use crate::inference::models::route_default::Resolved;

use super::joint_decode_rows::AttnCores;
use super::residency::MatvecKernels;
use super::rows_step::{token_wise_step_rows, TailRowsStep};

/// `0` selects `off`, `1` or unset `on`; anything else is an error.
pub(super) fn resolve_zero_one<T>(
    env: &str,
    value: Option<&OsStr>,
    off: T,
    on: T,
) -> Result<Resolved<T>> {
    match value.map(|value| value.to_str()) {
        None => Ok(Resolved::default_value(on)),
        Some(Some("0")) => Ok(Resolved::environment(off)),
        Some(Some("1")) => Ok(Resolved::environment(on)),
        Some(_) => anyhow::bail!("{env} must be 0 or 1 (got {value:?})"),
    }
}

/// The three resolved routes of one model load.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DefaultRoutes {
    pub(super) matvec: Resolved<MatvecKernels>,
    pub(super) tail_rows_step: Resolved<TailRowsStep>,
    pub(super) attn_cores: Resolved<AttnCores>,
}

impl DefaultRoutes {
    /// Reads the three switches; an invalid value is an error.
    pub(super) fn from_env() -> Result<Self> {
        Ok(Self {
            matvec: MatvecKernels::from_env()?,
            tail_rows_step: TailRowsStep::from_env()?,
            attn_cores: AttnCores::from_env()?,
        })
    }

    /// One audit line per model load: each resolved route, whether it is the
    /// default or an operator setting, and what the artifact and process
    /// admit. `joint_unavailable` is the model's
    /// `joint_decode_unavailable_reason`: while it is set, token-wise prompt
    /// work runs one-token steps and no joint step runs, whatever the
    /// switches say.
    pub(super) fn log(&self, joint_unavailable: Option<&str>) {
        tracing::info!(
            row_exact_matvec = ?self.matvec.value,
            row_exact_matvec_source = self.matvec.source.as_str(),
            tail_rows_step = ?self.tail_rows_step.value,
            tail_rows_step_source = self.tail_rows_step.source.as_str(),
            batched_attn_cores = ?self.attn_cores.value,
            batched_attn_cores_source = self.attn_cores.source.as_str(),
            token_wise_step_rows =
                token_wise_step_rows(self.tail_rows_step.value, joint_unavailable),
            joint_routes = joint_unavailable.unwrap_or("available"),
            "DeepSeek-V4 default routes resolved"
        );
        if let Some(reason) = joint_unavailable {
            if self.tail_rows_step.value == TailRowsStep::On {
                tracing::warn!(
                    reason,
                    "DeepSeek-V4 rows steps are unavailable; token-wise prompt work runs \
                     one-token steps"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::route_default::RouteSource;

    fn os(value: &str) -> Option<&OsStr> {
        Some(OsStr::new(value))
    }

    /// Unset: the release-gate configuration, each reported as the default.
    #[test]
    fn unset_switches_select_the_release_gate_configuration() {
        let matvec = MatvecKernels::resolve(None).unwrap();
        let tail = TailRowsStep::resolve(None).unwrap();
        let cores = AttnCores::resolve(None).unwrap();
        assert_eq!(matvec, Resolved::default_value(MatvecKernels::RowExact));
        assert_eq!(tail, Resolved::default_value(TailRowsStep::On));
        assert_eq!(cores, Resolved::default_value(AttnCores::Batched));
    }

    /// `0` on each switch restores the route set every release through
    /// v0.1.29 served (the types' `Default`, which fixtures build with).
    #[test]
    fn kill_switches_restore_the_previous_route_set() {
        let matvec = MatvecKernels::resolve(os("0")).unwrap();
        let tail = TailRowsStep::resolve(os("0")).unwrap();
        let cores = AttnCores::resolve(os("0")).unwrap();
        assert_eq!(matvec, Resolved::environment(MatvecKernels::Stock));
        assert_eq!(tail, Resolved::environment(TailRowsStep::Off));
        assert_eq!(cores, Resolved::environment(AttnCores::PerRow));
        assert_eq!(matvec.value, MatvecKernels::default());
        assert_eq!(tail.value, TailRowsStep::default());
        assert_eq!(cores.value, AttnCores::default());
        assert_eq!(token_wise_step_rows(tail.value, None), 1);
    }

    /// `1` selects the same routes as unset and is reported as a setting.
    #[test]
    fn explicit_one_equals_the_default_routes() {
        let matvec = MatvecKernels::resolve(os("1")).unwrap();
        let tail = TailRowsStep::resolve(os("1")).unwrap();
        let cores = AttnCores::resolve(os("1")).unwrap();
        assert_eq!(matvec.value, MatvecKernels::resolve(None).unwrap().value);
        assert_eq!(tail.value, TailRowsStep::resolve(None).unwrap().value);
        assert_eq!(cores.value, AttnCores::resolve(None).unwrap().value);
        for source in [matvec.source, tail.source, cores.source] {
            assert_eq!(source, RouteSource::Environment);
        }
    }

    /// Any other value fails the load on every switch (fail closed).
    #[test]
    fn invalid_values_fail_closed() {
        for bad in ["", "2", "on", "true", " 1", "off"] {
            assert!(MatvecKernels::resolve(os(bad)).is_err(), "{bad:?}");
            assert!(TailRowsStep::resolve(os(bad)).is_err(), "{bad:?}");
            assert!(AttnCores::resolve(os(bad)).is_err(), "{bad:?}");
        }
    }

    /// The default never refuses an artifact whose joint routes are
    /// unproven: rows steps fall back to one-token steps.
    #[test]
    fn default_rows_step_falls_back_when_joint_routes_are_unavailable() {
        let tail = TailRowsStep::resolve(None).unwrap().value;
        assert_eq!(token_wise_step_rows(tail, Some("unproven Q4_K route")), 1);
        assert_eq!(
            token_wise_step_rows(tail, None),
            super::super::joint_decode::MAX_JOINT_DECODE_ROWS
        );
    }
}
