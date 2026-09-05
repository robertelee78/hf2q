use std::fs::File;

use mlx_native::gguf::GgufFile;

use crate::backends::gguf::types::MetaValue;
use crate::backends::gguf::writer::GgufWriter;
use crate::quantize::ggml_quants::GgmlType;

use super::admission::validate;

const HIDDEN: u64 = 32;
const VOCAB: u64 = 4;

#[derive(Clone)]
struct Tensor {
    name: String,
    shape: Vec<u64>,
    ty: GgmlType,
}

#[derive(Clone, Copy, Default)]
struct Fixture {
    experts: bool,
    omit_q: bool,
    wrong_q_shape: bool,
    unsupported_q_storage: bool,
    untied_output: bool,
    unsupported_graph_feature: bool,
    invalid_expert_geometry: bool,
}

fn tensor(name: impl Into<String>, shape: &[u64], ty: GgmlType) -> Tensor {
    Tensor {
        name: name.into(),
        shape: shape.to_vec(),
        ty,
    }
}

fn metadata(fixture: Fixture) -> Vec<(&'static str, MetaValue)> {
    let mut values = vec![
        ("general.architecture", MetaValue::String("gemma4".into())),
        ("gemma4.block_count", MetaValue::U32(1)),
        ("gemma4.embedding_length", MetaValue::U32(HIDDEN as u32)),
        ("gemma4.feed_forward_length", MetaValue::U32(HIDDEN as u32)),
        ("gemma4.attention.head_count", MetaValue::U32(1)),
        (
            "gemma4.attention.head_count_kv",
            MetaValue::ArrayU32(vec![1]),
        ),
        ("gemma4.attention.key_length", MetaValue::U32(HIDDEN as u32)),
        (
            "gemma4.attention.key_length_swa",
            MetaValue::U32(HIDDEN as u32),
        ),
        (
            "gemma4.attention.layer_norm_rms_epsilon",
            MetaValue::F32(1e-6),
        ),
        ("gemma4.rope.freq_base", MetaValue::F32(1_000_000.0)),
        ("gemma4.rope.freq_base_swa", MetaValue::F32(10_000.0)),
        ("gemma4.attention.sliding_window", MetaValue::U32(32)),
        ("gemma4.context_length", MetaValue::U32(64)),
        (
            "gemma4.attention.sliding_window_pattern",
            MetaValue::ArrayBool(vec![true]),
        ),
        ("tokenizer.ggml.model", MetaValue::String("gemma4".into())),
        ("tokenizer.chat_template", MetaValue::String(include_str!(
            "../../../serve/api/test_fixtures/gemma4-apex-embedded-chat-template.jinja"
        ).into())),
        (
            "tokenizer.ggml.tokens",
            MetaValue::ArrayString(vec![
                "<unk>".into(),
                "<bos>".into(),
                "<eos>".into(),
                "x".into(),
            ]),
        ),
        ("tokenizer.ggml.merges", MetaValue::ArrayString(Vec::new())),
        ("tokenizer.ggml.unknown_token_id", MetaValue::U32(0)),
        ("tokenizer.ggml.bos_token_id", MetaValue::U32(1)),
        ("tokenizer.ggml.eos_token_id", MetaValue::U32(2)),
    ];
    if fixture.experts || fixture.invalid_expert_geometry {
        values.extend([
            ("gemma4.expert_count", MetaValue::U32(2)),
            (
                "gemma4.expert_used_count",
                MetaValue::U32(if fixture.invalid_expert_geometry {
                    3
                } else {
                    1
                }),
            ),
            (
                "gemma4.expert_feed_forward_length",
                MetaValue::U32(HIDDEN as u32),
            ),
        ]);
    }
    if fixture.unsupported_graph_feature {
        values.push(("gemma4.attention.shared_kv_layers", MetaValue::U32(1)));
    }
    values
}

fn tensors(fixture: Fixture) -> Vec<Tensor> {
    let mut tensors = vec![
        tensor("token_embd.weight", &[VOCAB, HIDDEN], GgmlType::Q4_0),
        tensor("output_norm.weight", &[HIDDEN], GgmlType::F32),
    ];
    if !fixture.omit_q {
        tensors.push(tensor(
            "blk.0.attn_q.weight",
            &[if fixture.wrong_q_shape { 31 } else { HIDDEN }, HIDDEN],
            if fixture.unsupported_q_storage {
                GgmlType::I32
            } else {
                GgmlType::Q4_0
            },
        ));
    }
    for name in [
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ] {
        tensors.push(tensor(
            format!("blk.0.{name}.weight"),
            &[HIDDEN, HIDDEN],
            GgmlType::Q4_0,
        ));
    }
    for name in [
        "attn_q_norm",
        "attn_k_norm",
        "attn_norm",
        "post_attention_norm",
        "ffn_norm",
        "post_ffw_norm",
    ] {
        tensors.push(tensor(
            format!("blk.0.{name}.weight"),
            &[HIDDEN],
            GgmlType::F32,
        ));
    }
    tensors.push(tensor(
        "blk.0.layer_output_scale.weight",
        &[1],
        GgmlType::F32,
    ));
    if fixture.experts {
        tensors.extend([
            tensor(
                "blk.0.ffn_gate_up_exps.weight",
                &[2, 64, HIDDEN],
                GgmlType::BF16,
            ),
            tensor(
                "blk.0.ffn_down_exps.weight",
                &[2, HIDDEN, HIDDEN],
                GgmlType::BF16,
            ),
            tensor("blk.0.ffn_gate_inp.weight", &[2, HIDDEN], GgmlType::F32),
            tensor("blk.0.ffn_gate_inp.scale", &[HIDDEN], GgmlType::F32),
            tensor("blk.0.ffn_down_exps.scale", &[2], GgmlType::F32),
        ]);
    }
    if fixture.untied_output {
        tensors.push(tensor("output.weight", &[VOCAB, HIDDEN], GgmlType::Q4_0));
    }
    tensors
}

fn open_fixture(fixture: Fixture) -> (tempfile::TempDir, GgufFile) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.gguf");
    let metadata = metadata(fixture);
    let tensors = tensors(fixture);
    let mut writer = GgufWriter::new(File::create(&path).unwrap());
    writer
        .write_header(tensors.len() as u64, metadata.len() as u64)
        .unwrap();
    for (key, value) in &metadata {
        writer.write_metadata_kv(key, value).unwrap();
    }
    for entry in &tensors {
        let gguf_dims = entry.shape.iter().rev().copied().collect::<Vec<_>>();
        writer
            .reserve_tensor_info(&entry.name, &gguf_dims, entry.ty)
            .unwrap();
    }
    writer.pad_to_alignment().unwrap();
    for (index, entry) in tensors.iter().enumerate() {
        let cols = entry.shape.last().copied().unwrap_or(0) as usize;
        let rows = entry.shape[..entry.shape.len().saturating_sub(1)]
            .iter()
            .product::<u64>() as usize;
        writer
            .stream_tensor_payload(index, &vec![0; rows * entry.ty.row_size(cols)])
            .unwrap();
    }
    writer.finalize().unwrap();
    drop(writer);
    let gguf = GgufFile::open(&path).unwrap();
    (directory, gguf)
}

fn rejection(fixture: Fixture) -> String {
    let (_directory, gguf) = open_fixture(fixture);
    validate(&gguf).unwrap_err().to_string()
}

#[test]
fn admits_tiny_dense_runtime_contract() {
    let (_directory, gguf) = open_fixture(Fixture::default());
    validate(&gguf).unwrap();
}

#[test]
fn admits_mixed_bf16_expert_runtime_contract() {
    let (_directory, gguf) = open_fixture(Fixture {
        experts: true,
        ..Fixture::default()
    });
    validate(&gguf).unwrap();
}

#[test]
fn rejects_missing_required_tensor() {
    assert!(rejection(Fixture {
        omit_q: true,
        ..Fixture::default()
    })
    .contains("missing required tensor"));
}

#[test]
fn rejects_wrong_runtime_shape() {
    assert!(rejection(Fixture {
        wrong_q_shape: true,
        ..Fixture::default()
    })
    .contains("shape"));
}

#[test]
fn rejects_storage_without_a_runtime_route() {
    let error = rejection(Fixture {
        unsupported_q_storage: true,
        ..Fixture::default()
    });
    assert!(
        error.contains("attn_q.weight") && error.contains("I32"),
        "{error}"
    );
}

#[test]
fn rejects_distinct_output_head() {
    assert!(rejection(Fixture {
        untied_output: true,
        ..Fixture::default()
    })
    .contains("distinct output.weight"));
}

#[test]
fn rejects_metadata_that_requires_an_unimplemented_graph() {
    assert!(rejection(Fixture {
        unsupported_graph_feature: true,
        ..Fixture::default()
    })
    .contains("shared_kv_layers"));
}

#[test]
fn rejects_invalid_expert_geometry_before_tensor_dispatch() {
    assert!(rejection(Fixture {
        invalid_expert_geometry: true,
        ..Fixture::default()
    })
    .contains("invalid expert geometry"));
}
