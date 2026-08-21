use half::bf16;
use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::*;
use crate::inference::models::qwen35::delta_net::DeltaNetLayerWeights;
use crate::inference::models::qwen35::ffn::DenseFfnWeights;
use crate::inference::models::qwen35::full_attn::FullAttnLayerWeights;
use crate::inference::models::qwen35::gpu_full_attn::FullAttnQGateWeightsGpu;
use crate::inference::models::qwen35::kv_cache::prepare_qwen35_base_text_cache;
use crate::inference::models::qwen35::model::{Qwen35FfnWeights, Qwen35LayerWeights, Qwen35Model};
use crate::inference::models::qwen35::source_precision::topology::admit_qwen35_bf16_topology;
use crate::inference::models::qwen35::source_precision::topology_tests::{
    finite_bf16_fixture, open, TensorSpec,
};
use crate::inference::models::qwen35::source_precision::upload::teacher_model::{
    prepare_qwen35_source_teacher, Qwen35SourceTeacherLimitsV1,
};
use crate::inference::models::qwen35::source_precision::upload_plan::QwenSourceMetalUploadLimits;

fn spec(name: impl Into<String>, shape: &[usize]) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        shape: shape.to_vec(),
    }
}

pub(in crate::inference::models::qwen35::source_precision::upload::teacher_model) fn h256_fixture(
) -> crate::inference::models::qwen35::source_precision::topology_tests::TopologyFixture {
    finite_bf16_fixture(|config, specs| {
        *config = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 256,
                "intermediate_size": 512,
                "vocab_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "max_position_embeddings": 128,
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "attn_output_gate": true,
                "rms_norm_eps": 0.000001
            }
        });
        specs.clear();
        specs.extend([
            spec("model.language_model.embed_tokens.weight", &[32, 256]),
            spec("model.language_model.norm.weight", &[256]),
            spec("lm_head.weight", &[32, 256]),
        ]);
        for layer in 0..2 {
            let prefix = format!("model.language_model.layers.{layer}");
            specs.extend([
                spec(format!("{prefix}.input_layernorm.weight"), &[256]),
                spec(format!("{prefix}.post_attention_layernorm.weight"), &[256]),
                spec(format!("{prefix}.mlp.gate_proj.weight"), &[512, 256]),
                spec(format!("{prefix}.mlp.up_proj.weight"), &[512, 256]),
                spec(format!("{prefix}.mlp.down_proj.weight"), &[256, 512]),
            ]);
            if layer == 0 {
                specs.extend([
                    spec(format!("{prefix}.linear_attn.A_log"), &[2]),
                    spec(format!("{prefix}.linear_attn.conv1d.weight"), &[512, 1, 4]),
                    spec(format!("{prefix}.linear_attn.dt_bias"), &[2]),
                    spec(format!("{prefix}.linear_attn.in_proj_a.weight"), &[2, 256]),
                    spec(format!("{prefix}.linear_attn.in_proj_b.weight"), &[2, 256]),
                    spec(
                        format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                        &[512, 256],
                    ),
                    spec(
                        format!("{prefix}.linear_attn.in_proj_z.weight"),
                        &[256, 256],
                    ),
                    spec(format!("{prefix}.linear_attn.norm.weight"), &[128]),
                    spec(format!("{prefix}.linear_attn.out_proj.weight"), &[256, 256]),
                ]);
            } else {
                specs.extend([
                    spec(format!("{prefix}.self_attn.q_proj.weight"), &[512, 256]),
                    spec(format!("{prefix}.self_attn.k_proj.weight"), &[256, 256]),
                    spec(format!("{prefix}.self_attn.v_proj.weight"), &[256, 256]),
                    spec(format!("{prefix}.self_attn.o_proj.weight"), &[256, 256]),
                    spec(format!("{prefix}.self_attn.q_norm.weight"), &[256]),
                    spec(format!("{prefix}.self_attn.k_norm.weight"), &[256]),
                ]);
            }
        }
        specs.extend([
            spec("mtp.fc.weight", &[256, 512]),
            spec("mtp.layers.0.input_layernorm.weight", &[256]),
            spec("mtp.layers.0.post_attention_layernorm.weight", &[256]),
            spec("mtp.layers.0.mlp.gate_proj.weight", &[512, 256]),
            spec("mtp.layers.0.mlp.up_proj.weight", &[512, 256]),
            spec("mtp.layers.0.mlp.down_proj.weight", &[256, 512]),
            spec("mtp.layers.0.self_attn.q_proj.weight", &[512, 256]),
            spec("mtp.layers.0.self_attn.k_proj.weight", &[256, 256]),
            spec("mtp.layers.0.self_attn.v_proj.weight", &[256, 256]),
            spec("mtp.layers.0.self_attn.o_proj.weight", &[256, 256]),
            spec("mtp.layers.0.self_attn.q_norm.weight", &[256]),
            spec("mtp.layers.0.self_attn.k_norm.weight", &[256]),
            spec("mtp.norm.weight", &[256]),
            spec("mtp.pre_fc_norm_embedding.weight", &[256]),
            spec("mtp.pre_fc_norm_hidden.weight", &[256]),
            spec("model.visual.patch.weight", &[2, 2]),
        ]);
    })
}

fn bf16_values(buffer: &MlxBuffer) -> Vec<f32> {
    assert_eq!(buffer.dtype(), DType::BF16);
    buffer
        .as_slice::<u16>()
        .unwrap()
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect()
}

fn f32_values(buffer: &MlxBuffer) -> Vec<f32> {
    assert_eq!(buffer.dtype(), DType::F32);
    buffer.as_slice::<f32>().unwrap().to_vec()
}

pub(in crate::inference::models::qwen35::source_precision::upload::teacher_model) fn cpu_model(
    teacher: &PreparedQwen35SourceTeacherV1,
) -> Qwen35Model {
    let mut model = Qwen35Model::empty_from_cfg(teacher.config.clone());
    model.token_embd = bf16_values(&teacher.embedding);
    model.output_norm = f32_values(&teacher.output_norm);
    model.output_weight = bf16_values(&teacher.output);
    model.layers = teacher
        .layers
        .iter()
        .map(|layer| {
            let ffn = Qwen35FfnWeights::Dense(DenseFfnWeights {
                gate: bf16_values(&layer.ffn.gate),
                up: bf16_values(&layer.ffn.up),
                down: bf16_values(&layer.ffn.down),
            });
            match &layer.attention {
                PreparedQwen35SourceAttentionV1::Full(weights) => {
                    let FullAttnQGateWeightsGpu::Split { wq, w_gate, .. } = &weights.q_gate else {
                        panic!("source teacher must retain split Q/gate weights");
                    };
                    Qwen35LayerWeights::FullAttn {
                        attn: FullAttnLayerWeights {
                            attn_norm: f32_values(&weights.attn_norm),
                            post_attn_norm: f32_values(&weights.post_attn_norm),
                            wq: bf16_values(wq),
                            wk: bf16_values(&weights.wk),
                            wv: bf16_values(&weights.wv),
                            w_gate: bf16_values(w_gate),
                            attn_q_norm: f32_values(&weights.attn_q_norm),
                            attn_k_norm: f32_values(&weights.attn_k_norm),
                            wo: bf16_values(&weights.wo),
                        },
                        ffn,
                    }
                }
                PreparedQwen35SourceAttentionV1::Linear(weights) => {
                    let channels = weights.ssm_conv1d.shape()[0];
                    let width = weights.ssm_conv1d.shape()[1];
                    let gpu_conv = f32_values(&weights.ssm_conv1d);
                    let mut cpu_conv = vec![0.0_f32; gpu_conv.len()];
                    for channel in 0..channels {
                        for offset in 0..width {
                            cpu_conv[offset * channels + channel] =
                                gpu_conv[channel * width + offset];
                        }
                    }
                    Qwen35LayerWeights::LinearAttn {
                        attn: DeltaNetLayerWeights {
                            attn_norm: f32_values(&weights.attn_norm),
                            post_attn_norm: f32_values(&weights.post_attn_norm),
                            attn_qkv: bf16_values(&weights.attn_qkv),
                            attn_gate: bf16_values(&weights.attn_gate),
                            ssm_conv1d: cpu_conv,
                            ssm_alpha: bf16_values(&weights.ssm_alpha),
                            ssm_dt_bias: f32_values(&weights.ssm_dt_bias),
                            ssm_beta: bf16_values(&weights.ssm_beta),
                            ssm_a: f32_values(&weights.ssm_a),
                            ssm_norm: f32_values(&weights.ssm_norm),
                            ssm_out: bf16_values(&weights.ssm_out),
                        },
                        ffn,
                    }
                }
            }
        })
        .collect();
    model
}

pub(in crate::inference::models::qwen35::source_precision::upload::teacher_model) fn last_cpu_logits(
    model: &Qwen35Model,
    tokens: &[u32],
) -> Vec<f32> {
    let positions: Vec<[i32; 4]> = (0..tokens.len())
        .map(|position| [position as i32; 4])
        .collect();
    let logits = model.forward_cpu(tokens, &positions).unwrap();
    logits[logits.len() - model.cfg.vocab_size as usize..].to_vec()
}

fn assert_logits(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|value| value.is_finite()));
    assert!(actual.iter().any(|value| *value != 0.0));
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let actual_top = actual
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    let expected_top = expected
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert_eq!(actual_top, expected_top, "{label} top-1");
    assert!(max_abs <= 5.0e-3, "{label} max_abs={max_abs}");
}

fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

fn zero_delta_output(model: &mut Qwen35Model) {
    match &mut model.layers[0] {
        Qwen35LayerWeights::LinearAttn { attn, .. } => attn.ssm_out.fill(0.0),
        _ => panic!("fixture layer 0 is not DeltaNet"),
    }
}

fn zero_full_attention_output(model: &mut Qwen35Model) {
    match &mut model.layers[1] {
        Qwen35LayerWeights::FullAttn { attn, .. } => attn.wo.fill(0.0),
        _ => panic!("fixture layer 1 is not full attention"),
    }
}

fn zero_dense_ffn_outputs(model: &mut Qwen35Model) {
    for layer in &mut model.layers {
        match layer.ffn_mut() {
            Qwen35FfnWeights::Dense(ffn) => ffn.down.fill(0.0),
            _ => panic!("fixture FFN is not dense"),
        }
    }
}

#[test]
fn source_teacher_private_runner_matches_cpu_for_prefill_and_cached_decode() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let caller_thread = std::thread::current().id();
    let fixture = h256_fixture();
    let topology = admit_qwen35_bf16_topology(open(&fixture).unwrap()).unwrap();
    let device = MlxDevice::new().unwrap();
    let teacher = prepare_qwen35_source_teacher(
        topology,
        &device,
        QwenSourceMetalUploadLimits::default(),
        Qwen35SourceTeacherLimitsV1 {
            max_sequence_tokens: 17,
            max_target_rows: 1,
            max_cpu_control_mirror_bytes: 1024 * 1024,
            unmeasured_runtime_reserve_bytes: 512 * 1024 * 1024,
        },
    )
    .unwrap();
    let prepared_cache = prepare_qwen35_base_text_cache(&teacher.config, &device, 17).unwrap();
    let model = cpu_model(&teacher);
    let prefix: Vec<u32> = (0..16).map(|value| value % 32).collect();
    let next = 19_u32;
    let mut full = prefix.clone();
    full.push(next);
    let mut no_delta = cpu_model(&teacher);
    zero_delta_output(&mut no_delta);
    let mut no_full = cpu_model(&teacher);
    zero_full_attention_output(&mut no_full);
    let mut no_ffn = cpu_model(&teacher);
    zero_dense_ffn_outputs(&mut no_ffn);
    for (phase, tokens) in [
        ("prefill", prefix.as_slice()),
        ("cached decode continuation", full.as_slice()),
    ] {
        let baseline = last_cpu_logits(&model, tokens);
        for (label, perturbed_model) in [
            ("DeltaNet", &no_delta),
            ("full attention", &no_full),
            ("dense FFN", &no_ffn),
        ] {
            let perturbed = last_cpu_logits(perturbed_model, tokens);
            let difference = max_abs_difference(&baseline, &perturbed);
            assert!(
                difference > 5.0e-3,
                "fixture does not discriminate a missing {label} contribution during {phase}: {difference}"
            );
        }
    }

    std::thread::Builder::new()
        .name("hf2q-qwen35-source-teacher-test".into())
        .spawn(move || {
            assert_ne!(std::thread::current().id(), caller_thread);
            assert_eq!(
                std::thread::current().name(),
                Some("hf2q-qwen35-source-teacher-test")
            );
            crate::inference::models::qwen35::execution_dispatch::with_source_teacher_graph_scope(
                |scope| {
                    let mut session = SourceTeacherSessionV1::new(scope, teacher, prepared_cache)?;
                    let fresh_cursor = session.cache.cache.full_attn[0].current_len.clone();
                    let fresh_parity = session.cache.cache.linear_attn[0].pp_flipped.clone();
                    assert!(session.run_call(&prefix[..15], true).is_err());
                    assert_eq!(session.cache.cache.full_attn[0].current_len, fresh_cursor);
                    assert_eq!(session.cache.cache.linear_attn[0].pp_flipped, fresh_parity);
                    assert!(
                        !session.poisoned,
                        "preflight rejection poisoned the session"
                    );
                    let prefill = session.run_call(&prefix, true)?.unwrap();
                    assert_eq!(prefill.graph_policy_sha256.len(), 64);
                    assert_logits(
                        &prefill.logits,
                        &last_cpu_logits(&model, &prefix),
                        "prefill",
                    );

                    let decode = session.run_call(&[next], true)?.unwrap();
                    assert_eq!(
                        prefill.graph_policy_sha256, decode.graph_policy_sha256,
                        "prefill/decode did not share the canonical source graph policy"
                    );
                    assert_logits(&decode.logits, &last_cpu_logits(&model, &full), "decode");
                    assert_eq!(
                        session.cache.cache.full_attn[0].current_len.as_slice(),
                        &[17]
                    );
                    assert!(!session.cache.cache.linear_attn[0].pp_flipped[0]);

                    session.cache.cache.full_attn[0].current_len[0] = 16;
                    let cursor_before = session.cache.cache.full_attn[0].current_len.clone();
                    let parity_before = session.cache.cache.linear_attn[0].pp_flipped.clone();
                    assert!(session.run_call(&[7], true).is_err());
                    assert_eq!(session.cache.cache.full_attn[0].current_len, cursor_before);
                    assert_eq!(session.cache.cache.linear_attn[0].pp_flipped, parity_before);
                    assert!(
                        !session.poisoned,
                        "preflight rejection poisoned the session"
                    );
                    Ok(())
                },
            )
            .unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}
