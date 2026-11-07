//! Per-model GGML routing policy for Qwen text models (#729).
//!
//! ADR-044 qualifies the K-quant width-four decode route
//! (`dense_decode_mvn = false`, `dense_decode_mv_ext = true`) for Qwen3.8
//! only. That default is a property of the loaded model, so it is resolved
//! once at load into a [`GgmlRoutingPolicy`] value stored on the model and
//! frozen into every `KernelRegistry` the model's forwards use. The process
//! environment is never mutated: another model loaded in the same process,
//! before or after, keeps the environment default, and no request can
//! observe a policy that depends on model-load order.
//!
//! Explicit operator values of `HF2Q_DECODE_MVN` / `HF2Q_DECODE_MV_EXT`
//! always win and, like every routing variable, apply to every model in the
//! process.

use std::path::Path;

use mlx_native::GgmlRoutingPolicy;

/// Operator override for the exact multi-column Q4_K/Q6_K matvec route.
pub(crate) const DECODE_MVN_ENV: &str = "HF2Q_DECODE_MVN";
/// Operator override for the weight-amortized `mul_mv_ext` route.
pub(crate) const DECODE_MV_EXT_ENV: &str = "HF2Q_DECODE_MV_EXT";

/// Whether `model_path` identifies a Qwen3.8 artifact. The match mirrors the
/// serve registry's deliberately fuzzy substring identification.
pub(crate) fn is_qwen38_model_path(model_path: &Path) -> bool {
    let id = model_path.to_string_lossy().to_ascii_lowercase();
    id.contains("qwen38") || id.contains("qwen3.8")
}

/// Resolve the routing policy for the Qwen model at `model_path` from its
/// identity and the explicit operator environment. Never writes the
/// environment.
pub fn resolve_qwen_ggml_routing_policy(model_path: &Path) -> GgmlRoutingPolicy {
    resolve_with(
        model_path,
        mlx_native::ggml_routing_policy_from_environment(),
        |key| std::env::var_os(key).is_some(),
    )
}

fn resolve_with(
    model_path: &Path,
    environment: GgmlRoutingPolicy,
    operator_set: impl Fn(&str) -> bool,
) -> GgmlRoutingPolicy {
    let mut policy = environment;
    if is_qwen38_model_path(model_path) {
        if !operator_set(DECODE_MVN_ENV) {
            policy.dense_decode_mvn = false;
        }
        if !operator_set(DECODE_MV_EXT_ENV) {
            policy.dense_decode_mv_ext = true;
        }
    }
    policy
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN38: &str = "/models/Qwen3.8-27B-Q4_K_M.gguf";
    const QWEN38_COMPACT: &str = "/models/qwen38-27b/model.gguf";
    const QWEN36: &str = "/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    const GEMMA: &str = "/models/gemma-4-26b-it.gguf";

    fn env_default() -> GgmlRoutingPolicy {
        GgmlRoutingPolicy {
            dense_decode_mvn: true,
            dense_decode_mv_ext: false,
            ..GgmlRoutingPolicy::default()
        }
    }

    fn unset(_: &str) -> bool {
        false
    }

    #[test]
    fn qwen38_identity_match_is_case_insensitive() {
        for path in [
            "/opt/hf2q/models/qwen38-stock/Qwen3.8-27B-vanilla.gguf",
            "/models/QWEN3.8-27B.gguf",
            "/models/QwEn38/model.gguf",
        ] {
            assert!(is_qwen38_model_path(Path::new(path)), "{path}");
        }
        for path in [QWEN36, GEMMA, "/models/qwen3-8b.gguf"] {
            assert!(!is_qwen38_model_path(Path::new(path)), "{path}");
        }
    }

    #[test]
    fn qwen38_identity_selects_width_four_route_without_operator_env() {
        for path in [QWEN38, QWEN38_COMPACT] {
            let policy = resolve_with(Path::new(path), env_default(), unset);
            assert!(!policy.dense_decode_mvn, "{path}");
            assert!(policy.dense_decode_mv_ext, "{path}");
            assert_eq!(
                GgmlRoutingPolicy {
                    dense_decode_mvn: true,
                    dense_decode_mv_ext: false,
                    ..policy
                },
                env_default(),
                "only the two qualified fields change"
            );
        }
    }

    #[test]
    fn other_models_keep_environment_policy_in_either_load_order() {
        let environment = env_default();
        let resolve = |path: &str| resolve_with(Path::new(path), environment, unset);
        for order in [[QWEN38, QWEN36, GEMMA], [GEMMA, QWEN36, QWEN38]] {
            for path in order {
                let policy = resolve(path);
                if is_qwen38_model_path(Path::new(path)) {
                    assert!(!policy.dense_decode_mvn && policy.dense_decode_mv_ext);
                } else {
                    assert_eq!(policy, environment, "{path}");
                }
            }
        }
    }

    #[test]
    fn explicit_operator_env_wins_per_variable() {
        let operator = GgmlRoutingPolicy {
            dense_decode_mvn: true,
            dense_decode_mv_ext: false,
            ..env_default()
        };
        let both = resolve_with(Path::new(QWEN38), operator, |_| true);
        assert_eq!(both, operator);
        let mvn_only = resolve_with(Path::new(QWEN38), operator, |key| key == DECODE_MVN_ENV);
        assert!(mvn_only.dense_decode_mvn);
        assert!(mvn_only.dense_decode_mv_ext);
        let mv_ext_only =
            resolve_with(Path::new(QWEN38), operator, |key| key == DECODE_MV_EXT_ENV);
        assert!(!mv_ext_only.dense_decode_mvn);
        assert!(!mv_ext_only.dense_decode_mv_ext);
    }

    /// The process resolver never writes the environment, so a Qwen3.8 load
    /// cannot change the policy any later (or earlier) model resolves.
    #[test]
    fn process_resolver_leaves_environment_untouched_in_either_order() {
        if std::env::var_os(DECODE_MVN_ENV).is_some()
            || std::env::var_os(DECODE_MV_EXT_ENV).is_some()
        {
            eprintln!("skipping: operator routing env is set");
            return;
        }
        let environment = mlx_native::ggml_routing_policy_from_environment();
        for order in [[QWEN38, GEMMA, QWEN36], [GEMMA, QWEN36, QWEN38]] {
            for path in order {
                let policy = resolve_qwen_ggml_routing_policy(Path::new(path));
                if is_qwen38_model_path(Path::new(path)) {
                    assert!(!policy.dense_decode_mvn, "{path}");
                    assert!(policy.dense_decode_mv_ext, "{path}");
                } else {
                    assert_eq!(policy, environment, "{path}");
                }
                assert!(std::env::var_os(DECODE_MVN_ENV).is_none());
                assert!(std::env::var_os(DECODE_MV_EXT_ENV).is_none());
            }
        }
        assert_eq!(mlx_native::ggml_routing_policy_from_environment(), environment);
    }
}
