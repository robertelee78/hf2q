//! Fail-closed admission of joint decode by matrix storage type.
//!
//! Joint decode is admissible only when every product it issues with
//! `m = B` rows is proven bitwise equal to the one-row product the serial
//! graph issues. Proven on the official artifact (bitwise hardware test) and
//! the Metal row tests: Q8_0, Q2_K and Q3_K at `m <= 8` on the default matvec
//! route; dense float weights only where `row_matmul` splits them per row;
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
    /// `forward_logits` at `m = B` through `raw_matmul` (never split).
    OutputHead,
    /// `grouped_output_a_batched` at `rows = B`.
    GroupedOutputA,
    /// Routed experts through the forced per-token `mv_id` kernel.
    RoutedExperts,
}

fn proven_block_type(ggml_type: GgmlType) -> bool {
    matches!(ggml_type, GgmlType::Q8_0 | GgmlType::Q2_K | GgmlType::Q3_K)
}

pub(super) fn joint_route_is_exact(product: JointProduct, ggml_type: GgmlType) -> bool {
    match product {
        JointProduct::RowMatmul => {
            proven_block_type(ggml_type)
                || matches!(ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
        }
        JointProduct::OutputHead | JointProduct::RoutedExperts => proven_block_type(ggml_type),
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
/// projections run per row (`m = 1`) and are exact for any storage.
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
    for name in ["output_hc_fn.weight", "output.weight"] {
        names.push((JointProduct::OutputHead, name.to_string()));
    }
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
            Ok(matrix) => products.push((product, name, matrix.ggml_type)),
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
}
