use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};

use crate::inference::models::qwen35::gpu_ffn::DenseFfnWeightsGpu;
use crate::inference::models::qwen35::{Qwen35Config, Qwen35LayerKind};

use super::super::{QwenSourceMetalUploadReceiptV1, VerifiedQwen35Bf16MetalUploadV1};
use super::layers::{full_attention, linear_attention, slot_totals, take_slot, OutputEntry};
use super::{
    PreparedGraphHashView, PreparedQwen35SourceAttentionV1, PreparedQwen35SourceLayerV1,
    PreparedQwen35SourceTeacherReceiptV1, PreparedQwen35SourceTeacherV1, PreparedReceiptHashView,
    Qwen35SourceTeacherRuntimeEnvelopeV1, PREPARED_PROFILE, PREPARED_SCHEMA_VERSION,
};
use crate::inference::models::qwen35::source_precision::topology::{
    Qwen35FutureDType, Qwen35SourceUse,
};
use crate::inference::models::qwen35::source_precision::types::SourcePrecisionDisposition;

pub(super) fn assemble(
    upload: VerifiedQwen35Bf16MetalUploadV1,
    config: Qwen35Config,
    runtime: Qwen35SourceTeacherRuntimeEnvelopeV1,
) -> Result<PreparedQwen35SourceTeacherV1> {
    let VerifiedQwen35Bf16MetalUploadV1 {
        _snapshot: snapshot,
        _device: device,
        _buffers: mut buffers,
        _buffer_addresses: buffer_addresses,
        receipt,
    } = upload;
    let QwenSourceMetalUploadReceiptV1 {
        topology_sha256,
        source_snapshot_catalog_sha256,
        device: device_identity,
        records,
        catalog_sha256: upload_catalog_sha256,
        receipt_sha256: upload_receipt_sha256,
        preflight,
        ..
    } = receipt;
    ensure!(
        source_snapshot_catalog_sha256 == snapshot.catalog_sha256(),
        "prepared source teacher snapshot differs from B2b"
    );
    ensure!(
        device_identity.registry_id == device.registry_id()
            && device_identity.name == device.name(),
        "prepared source teacher device differs from B2b"
    );

    let expected_mtp = usize::try_from(config.mtp_num_hidden_layers)?
        .checked_mul(15)
        .context("prepared source teacher MTP count overflow")?;
    let mut mtp_sources = 0_usize;
    let mut vision_sources = 0_usize;
    let mut outputs = BTreeMap::new();
    for source in records {
        match source.source_use {
            Qwen35SourceUse::FutureExecution => ensure!(
                !source.outputs.is_empty()
                    && source.disposition != SourcePrecisionDisposition::Excluded,
                "future source {} has no executable output or is excluded",
                source.source_name
            ),
            Qwen35SourceUse::AuthenticatedNonExecutedMtp => {
                mtp_sources += 1;
                ensure!(
                    source.outputs.is_empty()
                        && matches!(
                            source.disposition,
                            SourcePrecisionDisposition::Fixed
                                | SourcePrecisionDisposition::Protected
                        ),
                    "MTP source {} entered the prepared base graph",
                    source.source_name
                );
            }
            Qwen35SourceUse::ExcludedVision => {
                vision_sources += 1;
                ensure!(
                    source.outputs.is_empty()
                        && source.disposition == SourcePrecisionDisposition::Excluded,
                    "vision source {} entered the prepared text graph",
                    source.source_name
                );
            }
        }
        for output in source.outputs {
            let node_id = output.node_id.clone();
            ensure!(
                outputs
                    .insert(
                        node_id.clone(),
                        OutputEntry {
                            source_name: source.source_name.clone(),
                            output,
                        },
                    )
                    .is_none(),
                "prepared source teacher output node {node_id} is duplicated"
            );
        }
    }
    ensure!(
        mtp_sources == expected_mtp,
        "prepared source teacher MTP source count differs from config"
    );
    ensure!(
        outputs.len() == preflight.output_tensor_count
            && buffers.len() == outputs.len()
            && buffer_addresses.len() == outputs.len(),
        "prepared source teacher output count differs from B2b"
    );
    for (node_id, buffer) in &buffers {
        ensure!(
            buffer_addresses.get(node_id).copied() == Some(buffer.contents_ptr() as usize),
            "prepared source teacher buffer {node_id} changed B2b allocation identity"
        );
    }

    let hidden = usize::try_from(config.hidden_size)?;
    let vocabulary = usize::try_from(config.vocab_size)?;
    let intermediate = usize::try_from(
        config
            .intermediate_size
            .context("prepared dense teacher lacks intermediate size")?,
    )?;
    let mut slots = Vec::with_capacity(outputs.len());
    let embedding = take_slot(
        &mut buffers,
        &mut outputs,
        &mut slots,
        "global.embedding",
        "token_embd.weight",
        vec![vocabulary, hidden],
        Qwen35FutureDType::Bf16,
        device.registry_id(),
    )?;
    let output_norm = take_slot(
        &mut buffers,
        &mut outputs,
        &mut slots,
        "global.output_norm",
        "output_norm.weight",
        vec![hidden],
        Qwen35FutureDType::F32,
        device.registry_id(),
    )?;
    let output = take_slot(
        &mut buffers,
        &mut outputs,
        &mut slots,
        "global.output",
        "output.weight",
        vec![vocabulary, hidden],
        Qwen35FutureDType::Bf16,
        device.registry_id(),
    )?;

    let mut layers = Vec::with_capacity(config.layer_types.len());
    for (layer_index, kind) in config.layer_types.iter().enumerate() {
        let prefix = format!("blk.{layer_index}");
        let attn_norm = take_slot(
            &mut buffers,
            &mut outputs,
            &mut slots,
            &format!("layer.{layer_index}.attn_norm"),
            &format!("{prefix}.attn_norm.weight"),
            vec![hidden],
            Qwen35FutureDType::F32,
            device.registry_id(),
        )?;
        let post_attn_norm = take_slot(
            &mut buffers,
            &mut outputs,
            &mut slots,
            &format!("layer.{layer_index}.post_attn_norm"),
            &format!("{prefix}.post_attention_norm.weight"),
            vec![hidden],
            Qwen35FutureDType::F32,
            device.registry_id(),
        )?;
        let ffn = DenseFfnWeightsGpu {
            gate: take_slot(
                &mut buffers,
                &mut outputs,
                &mut slots,
                &format!("layer.{layer_index}.ffn_gate"),
                &format!("{prefix}.ffn_gate.weight"),
                vec![intermediate, hidden],
                Qwen35FutureDType::Bf16,
                device.registry_id(),
            )?,
            up: take_slot(
                &mut buffers,
                &mut outputs,
                &mut slots,
                &format!("layer.{layer_index}.ffn_up"),
                &format!("{prefix}.ffn_up.weight"),
                vec![intermediate, hidden],
                Qwen35FutureDType::Bf16,
                device.registry_id(),
            )?,
            down: take_slot(
                &mut buffers,
                &mut outputs,
                &mut slots,
                &format!("layer.{layer_index}.ffn_down"),
                &format!("{prefix}.ffn_down.weight"),
                vec![hidden, intermediate],
                Qwen35FutureDType::Bf16,
                device.registry_id(),
            )?,
        };
        let attention = match kind {
            Qwen35LayerKind::FullAttention => {
                PreparedQwen35SourceAttentionV1::Full(full_attention(
                    &mut buffers,
                    &mut outputs,
                    &mut slots,
                    layer_index,
                    &config,
                    attn_norm,
                    post_attn_norm,
                    device.registry_id(),
                )?)
            }
            Qwen35LayerKind::LinearAttention => {
                PreparedQwen35SourceAttentionV1::Linear(linear_attention(
                    &mut buffers,
                    &mut outputs,
                    &mut slots,
                    layer_index,
                    &config,
                    attn_norm,
                    post_attn_norm,
                    device.registry_id(),
                )?)
            }
        };
        layers.push(PreparedQwen35SourceLayerV1 { attention, ffn });
    }
    ensure!(
        outputs.is_empty() && buffers.is_empty(),
        "prepared source teacher left {} output descriptors and {} buffers unconsumed",
        outputs.len(),
        buffers.len()
    );

    let (bf16_tensor_count, f32_tensor_count, bf16_bytes, f32_bytes) = slot_totals(&slots)?;
    ensure!(
        bf16_tensor_count + f32_tensor_count == preflight.output_tensor_count
            && bf16_bytes
                .checked_add(f32_bytes)
                .context("prepared source teacher bytes overflow")?
                == preflight.total_output_bytes,
        "prepared source teacher totals differ from B2b"
    );
    let projected_execution_config_sha256 = super::config_hash(&config)?;
    let graph_catalog_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(
        &PreparedGraphHashView {
            schema_version: PREPARED_SCHEMA_VERSION,
            profile: PREPARED_PROFILE,
            topology_sha256: &topology_sha256,
            source_snapshot_catalog_sha256: &source_snapshot_catalog_sha256,
            projected_execution_config_sha256: &projected_execution_config_sha256,
            weight_slots: &slots,
            bf16_tensor_count,
            f32_tensor_count,
            bf16_bytes,
            f32_bytes,
            authenticated_nonexecuted_mtp_sources: mtp_sources,
            excluded_vision_sources: vision_sources,
            weight_precision: "source_bf16_controls_f32",
            q4_repack: false,
            dwq: false,
            tq: false,
            mtp_executed: false,
            graph_executed: false,
        },
    )?));
    let preparation_receipt_sha256 = hex::encode(Sha256::digest(serde_json::to_vec(
        &PreparedReceiptHashView {
            schema_version: PREPARED_SCHEMA_VERSION,
            profile: PREPARED_PROFILE,
            graph_catalog_sha256: &graph_catalog_sha256,
            upload_catalog_sha256: &upload_catalog_sha256,
            upload_receipt_sha256: &upload_receipt_sha256,
            device_name: &device_identity.name,
            device_registry_id: device_identity.registry_id,
            runtime: &runtime,
            runtime_liveness_proven: false,
        },
    )?));
    let receipt = PreparedQwen35SourceTeacherReceiptV1 {
        schema_version: PREPARED_SCHEMA_VERSION,
        profile: PREPARED_PROFILE,
        topology_sha256,
        upload_catalog_sha256,
        upload_receipt_sha256,
        source_snapshot_catalog_sha256,
        device_name: device_identity.name,
        device_registry_id: device_identity.registry_id,
        projected_execution_config_sha256,
        runtime,
        weight_slots: slots,
        bf16_tensor_count,
        f32_tensor_count,
        bf16_bytes,
        f32_bytes,
        authenticated_nonexecuted_mtp_sources: mtp_sources,
        excluded_vision_sources: vision_sources,
        weight_precision: "source_bf16_controls_f32",
        q4_repack: false,
        dwq: false,
        tq: false,
        mtp_executed: false,
        graph_executed: false,
        runtime_liveness_proven: false,
        graph_catalog_sha256,
        preparation_receipt_sha256,
    };
    Ok(PreparedQwen35SourceTeacherV1 {
        snapshot,
        device,
        config,
        embedding,
        output_norm,
        output,
        layers,
        receipt,
    })
}

#[cfg(test)]
pub(super) fn assemble_with_removed_buffer_for_test(
    mut upload: VerifiedQwen35Bf16MetalUploadV1,
    config: Qwen35Config,
    runtime: Qwen35SourceTeacherRuntimeEnvelopeV1,
    node_id: &str,
) -> Result<PreparedQwen35SourceTeacherV1> {
    upload._buffers.remove(node_id);
    assemble(upload, config, runtime)
}
