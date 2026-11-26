//! Q4-GEMMA-DECODE F2: exact grouped Q4_K gate/up plus GELU.
//!
//! Resolve once per model load. The route remains opt-in until its Class-A,
//! in-situ GPU and same-binary serving gates pass; `0` is the kill switch.
//! Other families, serial decode and unqualified widths/storage stay unchanged.

use std::ffi::OsStr;

use anyhow::{bail, ensure, Result};
use mlx_native::{GgmlQuantizedMatmulIdParams, GgmlRoutingPolicy, GgmlType};

use crate::inference::models::route_default::Resolved;

pub(crate) const ENV: &str = "HF2Q_GEMMA_DECODE_KERNELS";

fn parse(raw: Option<&OsStr>) -> Result<Resolved<bool>> {
    match raw {
        None => Ok(Resolved::default_value(false)),
        Some(value) if value == "0" => Ok(Resolved::environment(false)),
        Some(value) if value == "1" => Ok(Resolved::environment(true)),
        Some(value) => bail!("{ENV}={value:?} must be `0` or `1`"),
    }
}

fn require_reference_policy(policy: &GgmlRoutingPolicy) -> Result<()> {
    ensure!(policy.dense_q6k_mv_nr2, "{ENV}=1 requires dense_q6k_mv_nr2=true");
    ensure!(policy.dense_q8_0_mv_nr2, "{ENV}=1 requires dense_q8_0_mv_nr2=true");
    ensure!(!policy.expert_q8_0_mv_nr2, "{ENV}=1 requires expert_q8_0_mv_nr2=false");
    ensure!(!policy.dense_decode_mv_ext, "{ENV}=1 requires dense_decode_mv_ext=false");
    Ok(())
}

/// Validate before mapping weights, against this model's frozen native policy.
pub(crate) fn resolve(policy: &GgmlRoutingPolicy) -> Result<Resolved<bool>> {
    let raw = std::env::var_os(ENV);
    let resolved = parse(raw.as_deref())?;
    if resolved.value {
        require_reference_policy(policy)?;
        for name in ["HF2Q_HYBRID_NWG", "HF2Q_TQ_NSG"] {
            ensure!(std::env::var_os(name).is_none(), "{ENV}=1 requires {name} to be unset");
        }
        ensure!(
            std::env::var_os("HF2Q_BATCHED_FLASH").as_deref() != Some(OsStr::new("0")),
            "{ENV}=1 requires HF2Q_BATCHED_FLASH not to be `0`"
        );
    }
    Ok(resolved)
}

/// Only the story's qualified decode domain; native validation owns buffers,
/// strides and IDs. In particular, n5/n8 and non-Q4_K experts keep stock MV.
pub(crate) fn admits(params: &GgmlQuantizedMatmulIdParams) -> bool {
    (1..=4).contains(&params.n_tokens)
        && (1..=8).contains(&params.top_k)
        && params.ggml_type == GgmlType::Q4_K
        && params.n > 0
        && params.n % 2 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unqualified_decode_widths_and_storage_do_not_take_fused_route() {
        let mut params = GgmlQuantizedMatmulIdParams {
            n_tokens: 4,
            top_k: 8,
            n: 4608,
            k: 2816,
            n_experts: 128,
            expert_stride: 7_299_072,
            ggml_type: GgmlType::Q4_K,
        };
        assert!(admits(&params));
        for width in [0, 5, 8, 16] {
            params.n_tokens = width;
            assert!(!admits(&params));
        }
        params.n_tokens = 1;
        assert!(admits(&params));
        for dtype in [GgmlType::Q5_0, GgmlType::Q8_0, GgmlType::F32] {
            params.ggml_type = dtype;
            assert!(!admits(&params));
        }
        params.ggml_type = GgmlType::Q4_K;
        params.top_k = 9;
        assert!(!admits(&params));
        params.top_k = 8;
        params.n = 4607;
        assert!(!admits(&params));
    }

    #[test]
    fn changed_reference_policy_is_refused() {
        let reference = GgmlRoutingPolicy {
            dense_q6k_mv_nr2: true,
            dense_q8_0_mv_nr2: true,
            expert_q8_0_mv_nr2: false,
            dense_decode_mv_ext: false,
            ..GgmlRoutingPolicy::default()
        };
        for policy in [
            GgmlRoutingPolicy { dense_q6k_mv_nr2: false, ..reference },
            GgmlRoutingPolicy { dense_q8_0_mv_nr2: false, ..reference },
            GgmlRoutingPolicy { expert_q8_0_mv_nr2: true, ..reference },
            GgmlRoutingPolicy { dense_decode_mv_ext: true, ..reference },
        ] {
            assert!(require_reference_policy(&policy).is_err());
        }
    }

    #[test]
    fn malformed_operator_switch_is_refused() {
        for raw in ["", "2", "true", "off", " 1"] {
            assert!(parse(Some(OsStr::new(raw))).is_err());
        }
        assert!(!parse(Some(OsStr::new("0"))).unwrap().value);
        assert!(parse(Some(OsStr::new("1"))).unwrap().value);
    }
}
