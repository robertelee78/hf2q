//! Exact dense-Qwen source names, shapes, and future tensor descriptors.

use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};

use crate::convert::arch::bake::BakeOp;

use super::topology::{
    ExpectedSource, Qwen35FutureDType, Qwen35FutureTensorRecord, Qwen35QGateBranch,
    Qwen35SourceTransformV1, Qwen35SourceUse, Qwen35TopologyConfigV1,
};
use super::topology_expected_mtp::add_mtp_sources;

pub(super) fn expected_sources(
    config: &Qwen35TopologyConfigV1,
) -> Result<BTreeMap<String, ExpectedSource>> {
    let mut expected = BTreeMap::new();
    let root = if config.multimodal_wrapping {
        "model.language_model"
    } else {
        "model"
    };
    let h = config.hidden_size;
    let i = config.intermediate_size;

    add_identity(
        &mut expected,
        format!("{root}.embed_tokens.weight"),
        "token_embd.weight".into(),
        vec![config.vocabulary_size, h],
        Qwen35FutureDType::Bf16,
    )?;
    add_baked(
        &mut expected,
        format!("{root}.norm.weight"),
        "output_norm.weight".into(),
        vec![h],
        vec![h],
        BakeOp::AddOne,
        Qwen35FutureDType::F32,
        Qwen35SourceTransformV1::AddOneF32,
    )?;
    add_identity(
        &mut expected,
        "lm_head.weight".into(),
        "output.weight".into(),
        vec![config.vocabulary_size, h],
        Qwen35FutureDType::Bf16,
    )?;

    ensure!(
        config.layer_types.len() == config.num_hidden_layers,
        "dense-Qwen layer schedule length differs from num_hidden_layers"
    );
    for (layer, kind) in config.layer_types.iter().enumerate() {
        let source_prefix = format!("{root}.layers.{layer}");
        add_norms_and_ffn(&mut expected, &source_prefix, layer, h, i)?;
        match *kind {
            "full_attention" => add_full_attention(&mut expected, &source_prefix, layer, config)?,
            "linear_attention" => {
                add_linear_attention(&mut expected, &source_prefix, layer, config)?
            }
            other => anyhow::bail!("unsupported dense-Qwen layer kind {other}"),
        }
    }
    if config.mtp_num_hidden_layers == 1 {
        add_mtp_sources(&mut expected, config)?;
    }
    Ok(expected)
}

fn add_norms_and_ffn(
    expected: &mut BTreeMap<String, ExpectedSource>,
    source_prefix: &str,
    layer: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<()> {
    let mapped = |suffix: &str| format!("blk.{layer}.{suffix}");
    for (source_suffix, mapped_suffix) in [
        ("input_layernorm.weight", "attn_norm.weight"),
        (
            "post_attention_layernorm.weight",
            "post_attention_norm.weight",
        ),
    ] {
        add_baked(
            expected,
            format!("{source_prefix}.{source_suffix}"),
            mapped(mapped_suffix),
            vec![hidden],
            vec![hidden],
            BakeOp::AddOne,
            Qwen35FutureDType::F32,
            Qwen35SourceTransformV1::AddOneF32,
        )?;
    }
    for (source_suffix, mapped_suffix, shape) in [
        (
            "mlp.gate_proj.weight",
            "ffn_gate.weight",
            vec![intermediate, hidden],
        ),
        (
            "mlp.up_proj.weight",
            "ffn_up.weight",
            vec![intermediate, hidden],
        ),
        (
            "mlp.down_proj.weight",
            "ffn_down.weight",
            vec![hidden, intermediate],
        ),
    ] {
        add_identity(
            expected,
            format!("{source_prefix}.{source_suffix}"),
            mapped(mapped_suffix),
            shape,
            Qwen35FutureDType::Bf16,
        )?;
    }
    Ok(())
}

fn add_full_attention(
    expected: &mut BTreeMap<String, ExpectedSource>,
    source_prefix: &str,
    layer: usize,
    config: &Qwen35TopologyConfigV1,
) -> Result<()> {
    let h = config.hidden_size;
    let d = config.head_dim;
    let query_rows = checked_mul(config.num_attention_heads, d)?;
    let key_value_rows = checked_mul(config.num_key_value_heads, d)?;
    let fused_query_gate_rows = checked_mul(2, query_rows)?;
    let mapped_query = format!("blk.{layer}.attn_q.weight");
    let query_source = format!("{source_prefix}.self_attn.q_proj.weight");
    insert_expected(
        expected,
        query_source,
        ExpectedSource {
            shape: vec![fused_query_gate_rows, h],
            mapped_name: Some(mapped_query),
            mapped_bake: None,
            outputs: [
                (Qwen35QGateBranch::Query, "q"),
                (Qwen35QGateBranch::Gate, "gate"),
            ]
            .into_iter()
            .map(|(branch, suffix)| Qwen35FutureTensorRecord {
                node_id: format!("blk.{layer}.attn_q.{suffix}"),
                shape: vec![query_rows, h],
                dtype: Qwen35FutureDType::Bf16,
                transform: Qwen35SourceTransformV1::SplitInterleavedQGate {
                    branch,
                    num_query_heads: config.num_attention_heads,
                    head_dim: d,
                    hidden_size: h,
                },
            })
            .collect(),
            source_use: Qwen35SourceUse::FutureExecution,
        },
    )?;
    for (source_suffix, mapped_suffix, shape) in [
        (
            "self_attn.k_proj.weight",
            "attn_k.weight",
            vec![key_value_rows, h],
        ),
        (
            "self_attn.v_proj.weight",
            "attn_v.weight",
            vec![key_value_rows, h],
        ),
        (
            "self_attn.o_proj.weight",
            "attn_output.weight",
            vec![h, query_rows],
        ),
    ] {
        add_identity(
            expected,
            format!("{source_prefix}.{source_suffix}"),
            format!("blk.{layer}.{mapped_suffix}"),
            shape,
            Qwen35FutureDType::Bf16,
        )?;
    }
    for (source_suffix, mapped_suffix) in [
        ("self_attn.q_norm.weight", "attn_q_norm.weight"),
        ("self_attn.k_norm.weight", "attn_k_norm.weight"),
    ] {
        add_baked(
            expected,
            format!("{source_prefix}.{source_suffix}"),
            format!("blk.{layer}.{mapped_suffix}"),
            vec![d],
            vec![d],
            BakeOp::AddOne,
            Qwen35FutureDType::F32,
            Qwen35SourceTransformV1::AddOneF32,
        )?;
    }
    Ok(())
}

fn add_linear_attention(
    expected: &mut BTreeMap<String, ExpectedSource>,
    source_prefix: &str,
    layer: usize,
    config: &Qwen35TopologyConfigV1,
) -> Result<()> {
    let h = config.hidden_size;
    let nk = config.linear_num_key_heads;
    let nv = config.linear_num_value_heads;
    let d = config.linear_head_dim;
    let kernel = config.linear_conv_kernel_dim;
    let nv_per_k = nv / nk;
    let qk_rows = checked_mul(checked_mul(2, nk)?, d)?;
    let value_rows = checked_mul(nv, d)?;
    let qkv_rows = checked_add(qk_rows, value_rows)?;
    let mapped = |suffix: &str| format!("blk.{layer}.{suffix}");

    add_identity(
        expected,
        format!("{source_prefix}.linear_attn.norm.weight"),
        mapped("ssm_norm.weight"),
        vec![d],
        Qwen35FutureDType::F32,
    )?;
    add_baked(
        expected,
        format!("{source_prefix}.linear_attn.dt_bias"),
        mapped("ssm_dt.bias"),
        vec![nv],
        vec![nv],
        BakeOp::ReorderVHeads {
            num_k_heads: nk,
            num_v_per_k: nv_per_k,
            head_dim: 1,
            slice: None,
        },
        Qwen35FutureDType::F32,
        Qwen35SourceTransformV1::ReorderVHeads {
            num_key_heads: nk,
            num_values_per_key: nv_per_k,
            block_elements: 1,
            slice_start: None,
            slice_end: None,
        },
    )?;
    for (source_suffix, mapped_suffix) in [
        ("in_proj_a.weight", "ssm_alpha.weight"),
        ("in_proj_b.weight", "ssm_beta.weight"),
    ] {
        add_baked(
            expected,
            format!("{source_prefix}.linear_attn.{source_suffix}"),
            mapped(mapped_suffix),
            vec![nv, h],
            vec![nv, h],
            BakeOp::ReorderVHeads {
                num_k_heads: nk,
                num_v_per_k: nv_per_k,
                head_dim: h,
                slice: None,
            },
            Qwen35FutureDType::Bf16,
            Qwen35SourceTransformV1::ReorderVHeads {
                num_key_heads: nk,
                num_values_per_key: nv_per_k,
                block_elements: h,
                slice_start: None,
                slice_end: None,
            },
        )?;
    }
    let qkv_slice_start = checked_mul(qk_rows, h)?;
    let qkv_slice_end = checked_add(qkv_slice_start, checked_mul(value_rows, h)?)?;
    add_baked(
        expected,
        format!("{source_prefix}.linear_attn.in_proj_qkv.weight"),
        mapped("attn_qkv.weight"),
        vec![qkv_rows, h],
        vec![qkv_rows, h],
        BakeOp::ReorderVHeads {
            num_k_heads: nk,
            num_v_per_k: nv_per_k,
            head_dim: checked_mul(d, h)?,
            slice: Some(qkv_slice_start..qkv_slice_end),
        },
        Qwen35FutureDType::Bf16,
        Qwen35SourceTransformV1::ReorderVHeads {
            num_key_heads: nk,
            num_values_per_key: nv_per_k,
            block_elements: checked_mul(d, h)?,
            slice_start: Some(qkv_slice_start),
            slice_end: Some(qkv_slice_end),
        },
    )?;
    add_baked(
        expected,
        format!("{source_prefix}.linear_attn.in_proj_z.weight"),
        mapped("attn_gate.weight"),
        vec![value_rows, h],
        vec![value_rows, h],
        BakeOp::ReorderVHeads {
            num_k_heads: nk,
            num_v_per_k: nv_per_k,
            head_dim: checked_mul(d, h)?,
            slice: None,
        },
        Qwen35FutureDType::Bf16,
        Qwen35SourceTransformV1::ReorderVHeads {
            num_key_heads: nk,
            num_values_per_key: nv_per_k,
            block_elements: checked_mul(d, h)?,
            slice_start: None,
            slice_end: None,
        },
    )?;
    add_baked(
        expected,
        format!("{source_prefix}.linear_attn.A_log"),
        mapped("ssm_a"),
        vec![nv],
        vec![nv],
        BakeOp::Sequence(vec![
            BakeOp::ReorderVHeads {
                num_k_heads: nk,
                num_v_per_k: nv_per_k,
                head_dim: 1,
                slice: None,
            },
            BakeOp::NegExp,
        ]),
        Qwen35FutureDType::F32,
        Qwen35SourceTransformV1::ReorderVHeadsThenNegExpF32 {
            num_key_heads: nk,
            num_values_per_key: nv_per_k,
        },
    )?;
    let conv_slice_start = checked_mul(qk_rows, kernel)?;
    let conv_slice_end = checked_add(conv_slice_start, checked_mul(value_rows, kernel)?)?;
    add_baked(
        expected,
        format!("{source_prefix}.linear_attn.conv1d.weight"),
        mapped("ssm_conv1d.weight"),
        vec![qkv_rows, 1, kernel],
        vec![qkv_rows, kernel],
        BakeOp::Sequence(vec![
            BakeOp::Squeeze,
            BakeOp::ReorderVHeads {
                num_k_heads: nk,
                num_v_per_k: nv_per_k,
                head_dim: checked_mul(d, kernel)?,
                slice: Some(conv_slice_start..conv_slice_end),
            },
        ]),
        Qwen35FutureDType::F32,
        Qwen35SourceTransformV1::SqueezeAxis1ThenReorderVSlice {
            num_key_heads: nk,
            num_values_per_key: nv_per_k,
            value_head_dim: d,
            kernel_width: kernel,
            slice_start: conv_slice_start,
            slice_end: conv_slice_end,
        },
    )?;
    add_baked(
        expected,
        format!("{source_prefix}.linear_attn.out_proj.weight"),
        mapped("ssm_out.weight"),
        vec![h, value_rows],
        vec![h, value_rows],
        BakeOp::ReorderVHeadsPerRow {
            row_count: h,
            num_k_heads: nk,
            num_v_per_k: nv_per_k,
            head_dim_in_row: d,
        },
        Qwen35FutureDType::Bf16,
        Qwen35SourceTransformV1::ReorderVHeadsPerRow {
            row_count: h,
            num_key_heads: nk,
            num_values_per_key: nv_per_k,
            value_head_dim: d,
        },
    )?;
    Ok(())
}

fn add_identity(
    expected: &mut BTreeMap<String, ExpectedSource>,
    source_name: String,
    mapped_name: String,
    shape: Vec<usize>,
    dtype: Qwen35FutureDType,
) -> Result<()> {
    let output = Qwen35FutureTensorRecord {
        node_id: mapped_name.clone(),
        shape: shape.clone(),
        dtype,
        transform: Qwen35SourceTransformV1::Identity,
    };
    insert_expected(
        expected,
        source_name,
        ExpectedSource {
            shape,
            mapped_name: Some(mapped_name),
            mapped_bake: None,
            outputs: vec![output],
            source_use: Qwen35SourceUse::FutureExecution,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn add_baked(
    expected: &mut BTreeMap<String, ExpectedSource>,
    source_name: String,
    mapped_name: String,
    source_shape: Vec<usize>,
    output_shape: Vec<usize>,
    bake: BakeOp,
    dtype: Qwen35FutureDType,
    transform: Qwen35SourceTransformV1,
) -> Result<()> {
    let output = Qwen35FutureTensorRecord {
        node_id: mapped_name.clone(),
        shape: output_shape,
        dtype,
        transform,
    };
    insert_expected(
        expected,
        source_name,
        ExpectedSource {
            shape: source_shape,
            mapped_name: Some(mapped_name),
            mapped_bake: Some(bake),
            outputs: vec![output],
            source_use: Qwen35SourceUse::FutureExecution,
        },
    )
}

pub(super) fn insert_expected(
    expected: &mut BTreeMap<String, ExpectedSource>,
    source_name: String,
    source: ExpectedSource,
) -> Result<()> {
    ensure!(
        expected.insert(source_name.clone(), source).is_none(),
        "duplicate expected dense-Qwen source {source_name}"
    );
    Ok(())
}

pub(super) fn checked_mul(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .context("Qwen topology size overflow")
}

fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .context("Qwen topology size overflow")
}
