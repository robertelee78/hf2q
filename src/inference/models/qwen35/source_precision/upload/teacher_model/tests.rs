use std::collections::BTreeSet;

use anyhow::Result;
use mlx_native::MlxDevice;
use safetensors::Dtype as SafeDtype;

use super::*;
use crate::inference::models::qwen35::gpu_full_attn::FullAttnQGateWeightsGpu;
use crate::inference::models::qwen35::source_precision::topology::admit_qwen35_bf16_topology;
use crate::inference::models::qwen35::source_precision::topology_tests::{fixture, open};
use crate::inference::models::qwen35::source_precision::upload::upload_with_capacity_for_test;

fn upload_limits() -> QwenSourceMetalUploadLimits {
    QwenSourceMetalUploadLimits {
        max_output_tensors: 4_096,
        max_total_output_bytes: 128 * 1024 * 1024 * 1024,
        max_single_buffer_bytes: 8 * 1024 * 1024 * 1024,
        host_reserve_bytes: 0,
        metal_reserve_bytes: 0,
    }
}

fn teacher_limits() -> Qwen35SourceTeacherLimitsV1 {
    Qwen35SourceTeacherLimitsV1 {
        max_sequence_tokens: 32,
        max_target_rows: 16,
        max_cpu_control_mirror_bytes: 1024 * 1024,
        unmeasured_runtime_reserve_bytes: 0,
    }
}

fn capacity(extra: u64) -> QwenSourceMetalCapacityV1 {
    QwenSourceMetalCapacityV1 {
        host_available_bytes: 256 * 1024 * 1024 * 1024 + extra,
        metal_recommended_working_set_bytes: 256 * 1024 * 1024 * 1024 + extra,
        metal_current_allocated_bytes: 0,
        metal_max_buffer_bytes: 16 * 1024 * 1024 * 1024,
    }
}

#[test]
fn combined_capacity_preflight_reports_exact_boundary_without_allocating() -> Result<()> {
    let runtime = Qwen35SourceTeacherRuntimeEnvelopeV1 {
        max_sequence_tokens: 32,
        max_target_rows: 16,
        base_full_attention_cache_bytes: 40,
        base_linear_attention_state_bytes: 50,
        max_input_activation_bytes: 30,
        one_logit_row_bytes: 20,
        target_payload_upper_bound_bytes: 320,
        cpu_control_mirror_bytes: 60,
        accounted_runtime_payload_bytes: 200,
        unmeasured_runtime_reserve_bytes: 300,
    };
    let limits = QwenSourceMetalUploadLimits {
        host_reserve_bytes: 11,
        metal_reserve_bytes: 13,
        ..upload_limits()
    };
    let planned_weight_bytes = 100;
    let accounted = planned_weight_bytes + 4 * 1024 * 1024 + 200 + 300;
    let exact = QwenSourceMetalCapacityV1 {
        host_available_bytes: accounted + 11,
        metal_recommended_working_set_bytes: accounted + 13 + 17,
        metal_current_allocated_bytes: 17,
        metal_max_buffer_bytes: 16 * 1024 * 1024 * 1024,
    };
    let accepted = combined_capacity_preflight(planned_weight_bytes, &runtime, limits, exact)?;
    assert!(accepted.eligible);
    assert_eq!(accepted.host_required_bytes, accounted + 11);
    assert_eq!(accepted.metal_required_bytes, accounted + 13);
    assert_eq!(accepted.metal_available_bytes, accounted + 13);

    let host_short = combined_capacity_preflight(
        planned_weight_bytes,
        &runtime,
        limits,
        QwenSourceMetalCapacityV1 {
            host_available_bytes: exact.host_available_bytes - 1,
            ..exact
        },
    )?;
    assert!(!host_short.eligible);
    let metal_short = combined_capacity_preflight(
        planned_weight_bytes,
        &runtime,
        limits,
        QwenSourceMetalCapacityV1 {
            metal_recommended_working_set_bytes: exact.metal_recommended_working_set_bytes - 1,
            ..exact
        },
    )?;
    assert!(!metal_short.eligible);
    assert!(validate_combined_capacity(planned_weight_bytes, &runtime, limits, exact).is_ok());
    assert!(validate_combined_capacity(
        planned_weight_bytes,
        &runtime,
        limits,
        QwenSourceMetalCapacityV1 {
            host_available_bytes: exact.host_available_bytes - 1,
            ..exact
        }
    )
    .is_err());
    Ok(())
}

fn upload(
    device: &MlxDevice,
    observed: QwenSourceMetalCapacityV1,
) -> Result<(
    VerifiedQwen35Bf16MetalUploadV1,
    Qwen35Config,
    Qwen35SourceTeacherRuntimeEnvelopeV1,
)> {
    let invalid_fixture = fixture(SafeDtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&invalid_fixture)?)?;
    let config = topology.projected_config_for_teacher()?;
    let runtime = runtime_envelope(&config, teacher_limits())?;
    let uploaded = upload_with_capacity_for_test(
        topology,
        device,
        upload_limits(),
        observed,
        |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
    )?;
    Ok((uploaded, config, runtime))
}

fn prepared_buffer_addresses(prepared: &PreparedQwen35SourceTeacherV1) -> BTreeSet<usize> {
    let mut addresses = BTreeSet::from([
        prepared.embedding.contents_ptr() as usize,
        prepared.output_norm.contents_ptr() as usize,
        prepared.output.contents_ptr() as usize,
    ]);
    for layer in &prepared.layers {
        addresses.extend([
            layer.ffn.gate.contents_ptr() as usize,
            layer.ffn.up.contents_ptr() as usize,
            layer.ffn.down.contents_ptr() as usize,
        ]);
        match &layer.attention {
            PreparedQwen35SourceAttentionV1::Full(weights) => {
                let FullAttnQGateWeightsGpu::Split { wq, w_gate, .. } = &weights.q_gate else {
                    panic!("source teacher must retain split Q/gate weights");
                };
                addresses.extend([
                    weights.attn_norm.contents_ptr() as usize,
                    weights.post_attn_norm.contents_ptr() as usize,
                    wq.contents_ptr() as usize,
                    weights.wk.contents_ptr() as usize,
                    weights.wv.contents_ptr() as usize,
                    w_gate.contents_ptr() as usize,
                    weights.attn_q_norm.contents_ptr() as usize,
                    weights.attn_k_norm.contents_ptr() as usize,
                    weights.wo.contents_ptr() as usize,
                ])
            }
            PreparedQwen35SourceAttentionV1::Linear(weights) => addresses.extend([
                weights.attn_norm.contents_ptr() as usize,
                weights.post_attn_norm.contents_ptr() as usize,
                weights.attn_qkv.contents_ptr() as usize,
                weights.attn_gate.contents_ptr() as usize,
                weights.ssm_conv1d.contents_ptr() as usize,
                weights.ssm_alpha.contents_ptr() as usize,
                weights.ssm_dt_bias.contents_ptr() as usize,
                weights.ssm_beta.contents_ptr() as usize,
                weights.ssm_a.contents_ptr() as usize,
                weights.ssm_norm.contents_ptr() as usize,
                weights.ssm_out.contents_ptr() as usize,
            ]),
        }
    }
    addresses
}

#[test]
fn prepared_teacher_drains_exact_delta_full_graph_without_new_metal_weights() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };
    // `upload` drops the fixture directory before returning. Promotion must
    // rely only on the retained snapshot/config authority.
    let (uploaded, _config, _runtime) = upload(&device, capacity(0))?;
    let before: BTreeSet<_> = uploaded
        ._buffers
        .values()
        .map(|buffer| buffer.contents_ptr() as usize)
        .collect();
    let prepared =
        prepare_uploaded_with_capacity_for_test(uploaded, teacher_limits(), capacity(0))?;
    prepared.validate_for_test()?;
    assert_eq!(prepared.layer_count(), 2);
    assert_eq!(prepared.accounted_runtime_payload_bytes(), 1_208);
    assert_eq!(prepared_buffer_addresses(&prepared), before);

    let receipt = prepared.receipt_json_for_test();
    assert_eq!(receipt["bf16_tensor_count"], 18);
    assert_eq!(receipt["f32_tensor_count"], 11);
    assert_eq!(receipt["weight_slots"].as_array().unwrap().len(), 29);
    assert_eq!(receipt["authenticated_nonexecuted_mtp_sources"], 15);
    assert_eq!(receipt["excluded_vision_sources"], 1);
    assert_eq!(receipt["weight_precision"], "source_bf16_controls_f32");
    assert_eq!(receipt["q4_repack"], false);
    assert_eq!(receipt["dwq"], false);
    assert_eq!(receipt["tq"], false);
    assert_eq!(receipt["mtp_executed"], false);
    assert_eq!(receipt["graph_executed"], false);
    assert_eq!(receipt["runtime_liveness_proven"], false);
    Ok(())
}

#[test]
fn preferred_combined_transition_prepares_the_exact_graph() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };
    let source_fixture = fixture(SafeDtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&source_fixture)?)?;
    let prepared = prepare_with_capacity_for_test(
        topology,
        &device,
        upload_limits(),
        teacher_limits(),
        capacity(0),
        |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
    )?;
    prepared.validate_for_test()?;
    assert_eq!(prepared.layer_count(), 2);
    Ok(())
}

#[test]
fn prepared_graph_catalog_excludes_volatile_capacity_and_device_but_receipt_binds_them(
) -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };
    let (first_upload, first_config, first_runtime) = upload(&device, capacity(0))?;
    let first = assemble::assemble(first_upload, first_config, first_runtime)?;
    let (second_upload, second_config, second_runtime) = upload(&device, capacity(1))?;
    let second = assemble::assemble(second_upload, second_config, second_runtime)?;
    first.validate_for_test()?;
    second.validate_for_test()?;
    assert_eq!(first.graph_catalog_sha256(), second.graph_catalog_sha256());
    assert_ne!(
        first.preparation_receipt_sha256(),
        second.preparation_receipt_sha256()
    );

    let reproduced_graph = hex::encode(Sha256::digest(serde_json::to_vec(
        &PreparedGraphHashView {
            schema_version: first.receipt.schema_version,
            profile: first.receipt.profile,
            topology_sha256: &first.receipt.topology_sha256,
            source_snapshot_catalog_sha256: &first.receipt.source_snapshot_catalog_sha256,
            projected_execution_config_sha256: &first.receipt.projected_execution_config_sha256,
            weight_slots: &first.receipt.weight_slots,
            bf16_tensor_count: first.receipt.bf16_tensor_count,
            f32_tensor_count: first.receipt.f32_tensor_count,
            bf16_bytes: first.receipt.bf16_bytes,
            f32_bytes: first.receipt.f32_bytes,
            authenticated_nonexecuted_mtp_sources: first
                .receipt
                .authenticated_nonexecuted_mtp_sources,
            excluded_vision_sources: first.receipt.excluded_vision_sources,
            weight_precision: first.receipt.weight_precision,
            q4_repack: first.receipt.q4_repack,
            dwq: first.receipt.dwq,
            tq: first.receipt.tq,
            mtp_executed: first.receipt.mtp_executed,
            graph_executed: first.receipt.graph_executed,
        },
    )?));
    assert_eq!(reproduced_graph, first.graph_catalog_sha256());
    let synthetic_upload_catalog = "f".repeat(64);
    let synthetic_device_name = format!("{}-synthetic", first.receipt.device_name);
    let synthetic_receipt = hex::encode(Sha256::digest(serde_json::to_vec(
        &PreparedReceiptHashView {
            schema_version: first.receipt.schema_version,
            profile: first.receipt.profile,
            graph_catalog_sha256: first.graph_catalog_sha256(),
            upload_catalog_sha256: &synthetic_upload_catalog,
            upload_receipt_sha256: &first.receipt.upload_receipt_sha256,
            device_name: &synthetic_device_name,
            device_registry_id: first.receipt.device_registry_id.wrapping_add(1),
            runtime: &first.receipt.runtime,
            runtime_liveness_proven: first.receipt.runtime_liveness_proven,
        },
    )?));
    assert_ne!(synthetic_receipt, first.preparation_receipt_sha256());
    Ok(())
}

#[test]
fn prepared_teacher_rejects_missing_swapped_and_mutated_owned_buffers() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };

    let (missing, config, runtime) = upload(&device, capacity(0))?;
    let error = assemble::assemble_with_removed_buffer_for_test(
        missing,
        config,
        runtime,
        "blk.1.attn_q.gate",
    )
    .err()
    .expect("missing gate buffer must reject");
    assert!(format!("{error:#}").contains("output count differs from B2b"));

    let (mut swapped, config, runtime) = upload(&device, capacity(0))?;
    let left = swapped._buffers.remove("blk.0.attn_norm.weight").unwrap();
    let right = swapped
        ._buffers
        .remove("blk.0.post_attention_norm.weight")
        .unwrap();
    swapped
        ._buffers
        .insert("blk.0.attn_norm.weight".into(), right);
    swapped
        ._buffers
        .insert("blk.0.post_attention_norm.weight".into(), left);
    let error = assemble::assemble(swapped, config, runtime)
        .err()
        .expect("same-shape norm swap must reject");
    assert!(format!("{error:#}").contains("changed B2b allocation identity"));

    let (mut mutated, _config, _runtime) = upload(&device, capacity(0))?;
    mutated
        ._buffers
        .get_mut("blk.1.attn_q.q")
        .unwrap()
        .as_mut_slice::<u16>()?[0] ^= 1;
    let error = prepare_uploaded_with_capacity_for_test(mutated, teacher_limits(), capacity(0))
        .err()
        .expect("late buffer mutation must reject");
    assert!(format!("{error:#}").contains("differs from B2b"));
    Ok(())
}

#[test]
fn production_promotion_uses_live_capacity_and_retained_config_authority() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };
    // `upload` drops the fixture directory before returning. Even when this
    // host is intentionally contended, production must use the retained config
    // and either prepare or reject only at the live-capacity boundary.
    let (uploaded, _config, _runtime) = upload(&device, capacity(0))?;
    match prepare_uploaded_qwen35_source_teacher(uploaded, teacher_limits()) {
        Ok(prepared) => prepared.validate_for_test()?,
        Err(error) => ensure!(
            format!("{error:#}")
                .contains("incremental runtime requirement exceeds observed capacity"),
            "production promotion failed outside its live-capacity gate: {error:#}"
        ),
    }
    Ok(())
}

#[test]
fn combined_preflight_rejects_limits_and_capacity_before_weight_allocation() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };
    let limit_fixture = fixture(SafeDtype::BF16, |_, _| {});
    let mut allocation_calls = 0_usize;
    let topology = admit_qwen35_bf16_topology(open(&limit_fixture)?)?;
    let mut invalid_limits = teacher_limits();
    invalid_limits.max_sequence_tokens = 4_097;
    assert!(prepare_with_capacity_for_test(
        topology,
        &device,
        upload_limits(),
        invalid_limits,
        capacity(0),
        |bytes, dtype, shape| {
            allocation_calls += 1;
            Ok(device.alloc_buffer(bytes, dtype, shape)?)
        },
    )
    .is_err());
    assert_eq!(allocation_calls, 0);

    let capacity_fixture = fixture(SafeDtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&capacity_fixture)?)?;
    let mut insufficient = capacity(0);
    insufficient.host_available_bytes = 1;
    assert!(prepare_with_capacity_for_test(
        topology,
        &device,
        upload_limits(),
        teacher_limits(),
        insufficient,
        |bytes, dtype, shape| {
            allocation_calls += 1;
            Ok(device.alloc_buffer(bytes, dtype, shape)?)
        },
    )
    .is_err());
    assert_eq!(allocation_calls, 0);

    let promotion_fixture = fixture(SafeDtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&promotion_fixture)?)?;
    let mut reserved_upload_limits = upload_limits();
    reserved_upload_limits.host_reserve_bytes = 1;
    let uploaded = upload_with_capacity_for_test(
        topology,
        &device,
        reserved_upload_limits,
        capacity(0),
        |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
    )?;
    let config =
        crate::inference::models::qwen35::source_config::qwen35_config_from_authenticated_source(
            uploaded._snapshot.config(),
        )?;
    let runtime = runtime_envelope(&config, teacher_limits())?;
    let mut promotion_capacity = capacity(0);
    promotion_capacity.host_available_bytes = runtime.accounted_runtime_payload_bytes;
    assert!(prepare_uploaded_with_capacity_for_test(
        uploaded,
        teacher_limits(),
        promotion_capacity,
    )
    .is_err());

    let reserve_fixture = fixture(SafeDtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&reserve_fixture)?)?;
    let planned_weight_bytes = topology.planned_output_bytes()?;
    let runtime = runtime_envelope(&topology.projected_config_for_teacher()?, teacher_limits())?;
    let mut reserve_capacity = capacity(0);
    reserve_capacity.host_available_bytes =
        planned_weight_bytes + 4 * 1024 * 1024 + runtime.accounted_runtime_payload_bytes;
    let mut reserved_upload_limits = upload_limits();
    reserved_upload_limits.host_reserve_bytes = 1;
    assert!(prepare_with_capacity_for_test(
        topology,
        &device,
        reserved_upload_limits,
        teacher_limits(),
        reserve_capacity,
        |bytes, dtype, shape| {
            allocation_calls += 1;
            Ok(device.alloc_buffer(bytes, dtype, shape)?)
        },
    )
    .is_err());
    assert_eq!(allocation_calls, 0);
    Ok(())
}

#[test]
fn prepared_source_contains_no_quantized_or_global_cache_construction() {
    let source = concat!(
        include_str!("../teacher_model.rs"),
        include_str!("../teacher_model/assemble.rs"),
        include_str!("../teacher_model/layers.rs")
    );
    for forbidden in [
        "FullAttnWeightsGpu::from_cpu",
        "DeltaNetWeightsGpu::from_cpu",
        "DenseFfnWeightsGpuQ",
        "Qwen35Model",
        "ensure_gpu_cache_primed",
        "apply_dwq_overlay",
    ] {
        assert!(
            !source.contains(forbidden),
            "forbidden B3a seam: {forbidden}"
        );
    }
}
