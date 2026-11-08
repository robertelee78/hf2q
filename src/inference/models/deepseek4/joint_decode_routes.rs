//! Fail-closed admission of joint decode by matrix storage type.
//!
//! Joint decode is admissible only when every product it issues with
//! `m = B` rows is proven bitwise equal to the one-row product the serial
//! graph issues. Proven on the official artifact (bitwise hardware test) and
//! the Metal row tests: Q8_0, Q2_K and Q3_K at `m <= 8` on the default matvec
//! route; dense float weights only where `row_matmul` splits them per row
//! (including a float `output_hc_fn`, which `forward_logits` splits); IQ2_XXS
//! routed experts on the forced `mv_id` route (mlx-native
//! `tests/iq2_xxs_deepseek.rs::iq2_xxs_mv_id_row_bitwise`, m = 1..8, and the
//! ds4-file S3 joint == serial hardware check, ADR-063 Q4-DS4-RACE L7);
//! grouped output-A only for Q8_0. Any other storage (for example a float
//! output head, or K-quant `mN` routes) keeps the serial path.

use mlx_native::ops::quantized_matmul_ggml::GgmlType;

use super::residency::Deepseek4Weights;
use super::Deepseek4Config;

/// How joint decode issues one batched product.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum JointProduct {
    /// `row_matmul(IndependentRows)`: float storage is split per row.
    RowMatmul,
    /// `forward_logits` at `m = B` through `raw_matmul` (never split):
    /// `output.weight`, and `output_hc_fn` in block storage.
    OutputHead,
    /// `grouped_output_a_batched` at `rows = B`.
    GroupedOutputA,
    /// Routed experts through the forced per-token `mv_id` kernel.
    RoutedExperts,
}

fn proven_block_type(ggml_type: GgmlType) -> bool {
    matches!(ggml_type, GgmlType::Q8_0 | GgmlType::Q2_K | GgmlType::Q3_K)
}

fn dense_float_type(ggml_type: GgmlType) -> bool {
    matches!(ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

pub(super) const OUTPUT_HC_FN: &str = "output_hc_fn.weight";

/// How `forward_logits` issues `output_hc_fn`: float storage is split per
/// row through `row_matmul(IndependentRows)` (L6), block storage keeps one
/// `m = B` `raw_matmul`.
pub(super) fn output_hc_fn_product(ggml_type: GgmlType) -> JointProduct {
    if dense_float_type(ggml_type) {
        JointProduct::RowMatmul
    } else {
        JointProduct::OutputHead
    }
}

pub(super) fn joint_route_is_exact(product: JointProduct, ggml_type: GgmlType) -> bool {
    match product {
        JointProduct::RowMatmul => proven_block_type(ggml_type) || dense_float_type(ggml_type),
        JointProduct::OutputHead => proven_block_type(ggml_type),
        JointProduct::RoutedExperts => {
            proven_block_type(ggml_type) || ggml_type == GgmlType::IQ2_XXS
        }
        JointProduct::GroupedOutputA => ggml_type == GgmlType::Q8_0,
    }
}

/// First product whose storage is outside the proven set, as a log reason.
pub(super) fn first_unproven_joint_route<'a>(
    products: impl IntoIterator<Item = (JointProduct, &'a str, GgmlType)>,
) -> Option<String> {
    products
        .into_iter()
        .find(|&(product, _, ggml_type)| !joint_route_is_exact(product, ggml_type))
        .map(|(product, name, ggml_type)| {
            format!("{name} is {ggml_type:?}, which has no proven-exact joint {product:?} route")
        })
}

/// Every matrix joint decode multiplies with `m = B`. Compressor and indexer
/// projections are exact for any storage without being listed: the per-row
/// cores run them at `m = 1`, and the batched cores
/// (`joint_decode_rows::projection_takes_single_dispatch`) use one `m = B`
/// dispatch only on routes proven here or row-exact, else `m = 1` per row.
fn joint_product_names(cfg: &Deepseek4Config) -> Vec<(JointProduct, String)> {
    let mut names = Vec::new();
    for layer in 0..cfg.num_hidden_layers as usize {
        let prefix = format!("blk.{layer}");
        for suffix in [
            "hc_attn_fn",
            "attn_q_a",
            "attn_q_b",
            "attn_kv",
            "attn_output_b",
            "hc_ffn_fn",
            "ffn_gate_inp",
            "ffn_gate_shexp",
            "ffn_up_shexp",
            "ffn_down_shexp",
        ] {
            names.push((JointProduct::RowMatmul, format!("{prefix}.{suffix}.weight")));
        }
        names.push((
            JointProduct::GroupedOutputA,
            format!("{prefix}.attn_output_a.weight"),
        ));
        for suffix in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            names.push((
                JointProduct::RoutedExperts,
                format!("{prefix}.{suffix}.weight"),
            ));
        }
    }
    names.push((JointProduct::OutputHead, OUTPUT_HC_FN.to_string()));
    names.push((JointProduct::OutputHead, "output.weight".to_string()));
    names
}

/// `None` when every batched product of this artifact is proven exact.
pub(super) fn unproven_joint_route(
    cfg: &Deepseek4Config,
    weights: &Deepseek4Weights,
) -> Option<String> {
    let mut products = Vec::new();
    for (product, name) in joint_product_names(cfg) {
        match weights.raw_matrix_ref(&name) {
            Ok(matrix) => {
                let product = if name == OUTPUT_HC_FN {
                    output_hc_fn_product(matrix.ggml_type)
                } else {
                    product
                };
                products.push((product, name, matrix.ggml_type))
            }
            Err(error) => return Some(format!("joint decode route inventory: {error}")),
        }
    }
    first_unproven_joint_route(
        products
            .iter()
            .map(|(product, name, ggml_type)| (*product, name.as_str(), *ggml_type)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn official_products(head: GgmlType) -> Vec<(JointProduct, &'static str, GgmlType)> {
        vec![
            (
                JointProduct::RowMatmul,
                "blk.0.attn_q_b.weight",
                GgmlType::Q8_0,
            ),
            (
                JointProduct::RowMatmul,
                "blk.0.ffn_gate_inp.weight",
                GgmlType::F32,
            ),
            (
                JointProduct::RowMatmul,
                "blk.0.ffn_gate_shexp.weight",
                GgmlType::Q2_K,
            ),
            (
                JointProduct::RowMatmul,
                "blk.0.ffn_down_shexp.weight",
                GgmlType::Q3_K,
            ),
            (
                JointProduct::GroupedOutputA,
                "blk.0.attn_output_a.weight",
                GgmlType::Q8_0,
            ),
            (
                JointProduct::RoutedExperts,
                "blk.0.ffn_up_exps.weight",
                GgmlType::Q2_K,
            ),
            (
                JointProduct::RoutedExperts,
                "blk.0.ffn_down_exps.weight",
                GgmlType::Q3_K,
            ),
            (
                JointProduct::OutputHead,
                "output_hc_fn.weight",
                GgmlType::Q8_0,
            ),
            (JointProduct::OutputHead, "output.weight", head),
        ]
    }

    #[test]
    fn official_artifact_types_admit_joint_decode() {
        assert_eq!(
            first_unproven_joint_route(official_products(GgmlType::Q8_0)),
            None
        );
    }

    #[test]
    fn float_output_head_falls_back_to_serial() {
        for head in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
            let reason = first_unproven_joint_route(official_products(head))
                .expect("float head must keep serial decode");
            assert!(reason.contains("output.weight"), "{reason}");
        }
    }

    #[test]
    fn unproven_block_types_fall_back_to_serial() {
        for (product, ggml_type) in [
            (JointProduct::RowMatmul, GgmlType::Q4_K),
            (JointProduct::RowMatmul, GgmlType::Q6_K),
            (JointProduct::RowMatmul, GgmlType::Q4_0),
            (JointProduct::GroupedOutputA, GgmlType::Q2_K),
            (JointProduct::GroupedOutputA, GgmlType::F32),
            (JointProduct::RoutedExperts, GgmlType::Q4_K),
            (JointProduct::OutputHead, GgmlType::Q6_K),
        ] {
            assert!(
                !joint_route_is_exact(product, ggml_type),
                "{product:?} {ggml_type:?}"
            );
            let mut products = official_products(GgmlType::Q8_0);
            products.push((product, "blk.1.unproven.weight", ggml_type));
            assert!(first_unproven_joint_route(products).is_some());
        }
    }

    #[test]
    fn row_split_float_weights_are_exact() {
        for ggml_type in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
            assert!(joint_route_is_exact(JointProduct::RowMatmul, ggml_type));
        }
    }

    /// ds4's 0731 file storage types per class (design §2.3), with
    /// `output_hc_fn` classified the way `forward_logits` issues it.
    fn ds4_file_products() -> Vec<(JointProduct, &'static str, GgmlType)> {
        vec![
            (
                JointProduct::RowMatmul,
                "blk.0.hc_attn_fn.weight",
                GgmlType::F16,
            ),
            (
                JointProduct::RowMatmul,
                "blk.0.attn_q_a.weight",
                GgmlType::Q8_0,
            ),
            (
                JointProduct::RowMatmul,
                "blk.0.attn_q_b.weight",
                GgmlType::Q8_0,
            ),
            (
                JointProduct::RowMatmul,
                "blk.0.attn_kv.weight",
                GgmlType::Q8_0,
            ),
            (
                JointProduct::RowMatmul,
                "blk.0.attn_output_b.weight",
                GgmlType::Q8_0,
            ),
            (
                JointProduct::RowMatmul,
                "blk.0.hc_ffn_fn.weight",
                GgmlType::F16,
            ),
            (
                JointProduct::RowMatmul,
                "blk.0.ffn_gate_inp.weight",
                GgmlType::F16,
            ),
            (
                JointProduct::RowMatmul,
                "blk.0.ffn_gate_shexp.weight",
                GgmlType::Q8_0,
            ),
            (
                JointProduct::RowMatmul,
                "blk.0.ffn_up_shexp.weight",
                GgmlType::Q8_0,
            ),
            (
                JointProduct::RowMatmul,
                "blk.0.ffn_down_shexp.weight",
                GgmlType::Q8_0,
            ),
            (
                JointProduct::GroupedOutputA,
                "blk.0.attn_output_a.weight",
                GgmlType::Q8_0,
            ),
            (
                JointProduct::RoutedExperts,
                "blk.0.ffn_gate_exps.weight",
                GgmlType::IQ2_XXS,
            ),
            (
                JointProduct::RoutedExperts,
                "blk.0.ffn_up_exps.weight",
                GgmlType::IQ2_XXS,
            ),
            (
                JointProduct::RoutedExperts,
                "blk.0.ffn_down_exps.weight",
                GgmlType::Q2_K,
            ),
            (
                output_hc_fn_product(GgmlType::F16),
                OUTPUT_HC_FN,
                GgmlType::F16,
            ),
            (JointProduct::OutputHead, "output.weight", GgmlType::Q8_0),
        ]
    }

    #[test]
    fn ds4_file_types_admit_joint_decode() {
        assert_eq!(first_unproven_joint_route(ds4_file_products()), None);
    }

    #[test]
    fn float_output_hc_fn_is_row_split_and_block_storage_stays_output_head() {
        for ggml_type in [GgmlType::F32, GgmlType::F16, GgmlType::BF16] {
            assert_eq!(output_hc_fn_product(ggml_type), JointProduct::RowMatmul);
        }
        // hf2q's artifact (Q8_0) keeps its existing classification.
        for ggml_type in [
            GgmlType::Q8_0,
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
        ] {
            assert_eq!(output_hc_fn_product(ggml_type), JointProduct::OutputHead);
        }
        // Without the L6 row split the F16 hc head would fall back to serial.
        let reason =
            first_unproven_joint_route([(JointProduct::OutputHead, OUTPUT_HC_FN, GgmlType::F16)])
                .expect("unsplit float output_hc_fn is not proven");
        assert!(reason.contains(OUTPUT_HC_FN), "{reason}");
    }

    #[test]
    fn iq2_xxs_is_proven_only_for_routed_experts() {
        assert!(joint_route_is_exact(
            JointProduct::RoutedExperts,
            GgmlType::IQ2_XXS
        ));
        for product in [
            JointProduct::RowMatmul,
            JointProduct::OutputHead,
            JointProduct::GroupedOutputA,
        ] {
            assert!(
                !joint_route_is_exact(product, GgmlType::IQ2_XXS),
                "{product:?}"
            );
        }
        for unproven in [
            GgmlType::IQ4_XS,
            GgmlType::IQ4_NL,
            GgmlType::Q4_K,
            GgmlType::Q6_K,
        ] {
            assert!(
                !joint_route_is_exact(JointProduct::RoutedExperts, unproven),
                "{unproven:?}"
            );
        }
    }
}
