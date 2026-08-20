//! Exact authenticated-but-nonexecuted MTP topology for the base teacher.

use std::collections::BTreeMap;

use anyhow::Result;

use crate::convert::arch::bake::BakeOp;

use super::topology::{ExpectedSource, Qwen35SourceUse, Qwen35TopologyConfigV1};
use super::topology_expected::{checked_mul, insert_expected};

pub(super) fn add_mtp_sources(
    expected: &mut BTreeMap<String, ExpectedSource>,
    config: &Qwen35TopologyConfigV1,
) -> Result<()> {
    let h = config.hidden_size;
    let i = config.intermediate_size;
    let d = config.head_dim;
    let query_rows = checked_mul(config.num_attention_heads, d)?;
    let key_value_rows = checked_mul(config.num_key_value_heads, d)?;
    let block = config.num_hidden_layers;
    let mapped = |suffix: &str| format!("blk.{block}.{suffix}");
    for (name, shape, mapped_name, bake) in [
        (
            "mtp.fc.weight",
            vec![h, checked_mul(2, h)?],
            mapped("nextn.eh_proj.weight"),
            None,
        ),
        (
            "mtp.layers.0.input_layernorm.weight",
            vec![h],
            mapped("attn_norm.weight"),
            Some(BakeOp::AddOne),
        ),
        (
            "mtp.layers.0.post_attention_layernorm.weight",
            vec![h],
            mapped("post_attention_norm.weight"),
            Some(BakeOp::AddOne),
        ),
        (
            "mtp.layers.0.mlp.gate_proj.weight",
            vec![i, h],
            mapped("ffn_gate.weight"),
            None,
        ),
        (
            "mtp.layers.0.mlp.up_proj.weight",
            vec![i, h],
            mapped("ffn_up.weight"),
            None,
        ),
        (
            "mtp.layers.0.mlp.down_proj.weight",
            vec![h, i],
            mapped("ffn_down.weight"),
            None,
        ),
        (
            "mtp.layers.0.self_attn.q_proj.weight",
            vec![checked_mul(2, query_rows)?, h],
            mapped("attn_q.weight"),
            None,
        ),
        (
            "mtp.layers.0.self_attn.k_proj.weight",
            vec![key_value_rows, h],
            mapped("attn_k.weight"),
            None,
        ),
        (
            "mtp.layers.0.self_attn.v_proj.weight",
            vec![key_value_rows, h],
            mapped("attn_v.weight"),
            None,
        ),
        (
            "mtp.layers.0.self_attn.o_proj.weight",
            vec![h, query_rows],
            mapped("attn_output.weight"),
            None,
        ),
        (
            "mtp.layers.0.self_attn.q_norm.weight",
            vec![d],
            mapped("attn_q_norm.weight"),
            Some(BakeOp::AddOne),
        ),
        (
            "mtp.layers.0.self_attn.k_norm.weight",
            vec![d],
            mapped("attn_k_norm.weight"),
            Some(BakeOp::AddOne),
        ),
        (
            "mtp.norm.weight",
            vec![h],
            mapped("nextn.shared_head_norm.weight"),
            Some(BakeOp::AddOne),
        ),
        (
            "mtp.pre_fc_norm_embedding.weight",
            vec![h],
            mapped("nextn.enorm.weight"),
            Some(BakeOp::AddOne),
        ),
        (
            "mtp.pre_fc_norm_hidden.weight",
            vec![h],
            mapped("nextn.hnorm.weight"),
            Some(BakeOp::AddOne),
        ),
    ] {
        insert_expected(
            expected,
            name.into(),
            ExpectedSource {
                shape,
                mapped_name: Some(mapped_name),
                mapped_bake: bake,
                outputs: Vec::new(),
                source_use: Qwen35SourceUse::AuthenticatedNonExecutedMtp,
            },
        )?;
    }
    Ok(())
}
