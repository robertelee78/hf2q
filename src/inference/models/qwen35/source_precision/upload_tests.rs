use std::collections::BTreeSet;

use anyhow::{bail, Result};
use half::bf16;
use mlx_native::{DType, MlxBuffer, MlxDevice};
use safetensors::Dtype as SafeDtype;
use sha2::{Digest, Sha256};

use super::topology::{
    admit_qwen35_bf16_topology, Qwen35FutureDType, Qwen35QGateBranch, Qwen35SourceTopologyRecord,
    Qwen35SourceTransformV1,
};
use super::topology_tests::{fixture, open};
use super::upload::upload_with_capacity_for_test;
use super::upload_plan::{
    preflight_upload, QwenSourceMetalCapacityV1, QwenSourceMetalUploadLimits,
};
use crate::convert::arch::bake::{apply_bake_op, BakeOp};

fn test_limits() -> QwenSourceMetalUploadLimits {
    QwenSourceMetalUploadLimits {
        max_output_tensors: 4_096,
        max_total_output_bytes: 128 * 1024 * 1024 * 1024,
        max_single_buffer_bytes: 8 * 1024 * 1024 * 1024,
        host_reserve_bytes: 0,
        metal_reserve_bytes: 0,
    }
}

fn ample_capacity() -> QwenSourceMetalCapacityV1 {
    QwenSourceMetalCapacityV1 {
        host_available_bytes: 256 * 1024 * 1024 * 1024,
        metal_recommended_working_set_bytes: 256 * 1024 * 1024 * 1024,
        metal_current_allocated_bytes: 0,
        metal_max_buffer_bytes: 16 * 1024 * 1024 * 1024,
    }
}

fn bake_for_transform(transform: &Qwen35SourceTransformV1) -> Option<BakeOp> {
    match transform {
        Qwen35SourceTransformV1::Identity => None,
        Qwen35SourceTransformV1::AddOneF32 => Some(BakeOp::AddOne),
        Qwen35SourceTransformV1::ReorderVHeads {
            num_key_heads,
            num_values_per_key,
            block_elements,
            slice_start,
            slice_end,
        } => Some(BakeOp::ReorderVHeads {
            num_k_heads: *num_key_heads,
            num_v_per_k: *num_values_per_key,
            head_dim: *block_elements,
            slice: match (*slice_start, *slice_end) {
                (Some(start), Some(end)) => Some(start..end),
                (None, None) => None,
                _ => panic!("preflight accepted a partial reorder slice"),
            },
        }),
        Qwen35SourceTransformV1::ReorderVHeadsThenNegExpF32 {
            num_key_heads,
            num_values_per_key,
        } => Some(BakeOp::Sequence(vec![
            BakeOp::ReorderVHeads {
                num_k_heads: *num_key_heads,
                num_v_per_k: *num_values_per_key,
                head_dim: 1,
                slice: None,
            },
            BakeOp::NegExp,
        ])),
        Qwen35SourceTransformV1::SqueezeAxis1ThenReorderVSlice {
            num_key_heads,
            num_values_per_key,
            value_head_dim,
            kernel_width,
            slice_start,
            slice_end,
        } => Some(BakeOp::Sequence(vec![
            BakeOp::Squeeze,
            BakeOp::ReorderVHeads {
                num_k_heads: *num_key_heads,
                num_v_per_k: *num_values_per_key,
                head_dim: value_head_dim * kernel_width,
                slice: Some(*slice_start..*slice_end),
            },
        ])),
        Qwen35SourceTransformV1::ReorderVHeadsPerRow {
            row_count,
            num_key_heads,
            num_values_per_key,
            value_head_dim,
        } => Some(BakeOp::ReorderVHeadsPerRow {
            row_count: *row_count,
            num_k_heads: *num_key_heads,
            num_v_per_k: *num_values_per_key,
            head_dim_in_row: *value_head_dim,
        }),
        Qwen35SourceTransformV1::SplitInterleavedQGate { .. } => None,
    }
}

fn split_reference(
    source: &[u16],
    branch: Qwen35QGateBranch,
    num_query_heads: usize,
    head_dim: usize,
    hidden_size: usize,
) -> Vec<u16> {
    let mut output = Vec::with_capacity(source.len() / 2);
    for head in 0..num_query_heads {
        let branch_offset = match branch {
            Qwen35QGateBranch::Query => 0,
            Qwen35QGateBranch::Gate => head_dim,
        };
        let row_start = (head * 2 * head_dim + branch_offset) * hidden_size;
        output.extend_from_slice(&source[row_start..row_start + head_dim * hidden_size]);
    }
    output
}

fn expected_bytes(
    source_words: &[u16],
    transform: &Qwen35SourceTransformV1,
    dtype: Qwen35FutureDType,
) -> Vec<u8> {
    if let Qwen35SourceTransformV1::SplitInterleavedQGate {
        branch,
        num_query_heads,
        head_dim,
        hidden_size,
    } = transform
    {
        return split_reference(
            source_words,
            *branch,
            *num_query_heads,
            *head_dim,
            *hidden_size,
        )
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    }
    let source_f32: Vec<f32> = source_words
        .iter()
        .map(|word| bf16::from_bits(*word).to_f32())
        .collect();
    match dtype {
        Qwen35FutureDType::Bf16 => {
            let positions: Vec<f32> = (0..source_words.len()).map(|index| index as f32).collect();
            let positions = match bake_for_transform(transform) {
                Some(op) => apply_bake_op(positions, &op).unwrap(),
                None => positions,
            };
            positions
                .into_iter()
                .flat_map(|position| source_words[position as usize].to_le_bytes())
                .collect()
        }
        Qwen35FutureDType::F32 => match bake_for_transform(transform) {
            Some(op) => apply_bake_op(source_f32, &op).unwrap(),
            None => source_f32,
        }
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect(),
    }
}

fn assert_outputs_match_reference(
    reference: &super::snapshot::VerifiedQwenSourceSnapshot,
    records: &[Qwen35SourceTopologyRecord],
    upload: &super::upload::VerifiedQwen35Bf16MetalUploadV1,
) {
    let mut transform_kinds = BTreeSet::new();
    for source in records {
        let retained = reference.tensor_record(&source.source_name).unwrap();
        let mut words = vec![0_u16; usize::try_from(retained.byte_len / 2).unwrap()];
        reference
            .read_tensor_u16(&source.source_name, &mut words)
            .unwrap();
        for output in &source.outputs {
            transform_kinds.insert(match output.transform {
                Qwen35SourceTransformV1::Identity => 0,
                Qwen35SourceTransformV1::AddOneF32 => 1,
                Qwen35SourceTransformV1::ReorderVHeads { .. } => 2,
                Qwen35SourceTransformV1::ReorderVHeadsThenNegExpF32 { .. } => 3,
                Qwen35SourceTransformV1::SqueezeAxis1ThenReorderVSlice { .. } => 4,
                Qwen35SourceTransformV1::ReorderVHeadsPerRow { .. } => 5,
                Qwen35SourceTransformV1::SplitInterleavedQGate { .. } => 6,
            });
            let expected = expected_bytes(&words, &output.transform, output.dtype);
            let actual = upload.buffer_bytes_for_test(&output.node_id).unwrap();
            assert_eq!(
                actual.len(),
                expected.len(),
                "output {} length differs",
                output.node_id
            );
            assert_eq!(
                Sha256::digest(&actual),
                Sha256::digest(&expected),
                "output {} content digest differs",
                output.node_id
            );
        }
    }
    assert_eq!(transform_kinds.len(), 7);
}

#[test]
fn preflight_rejects_capacity_source_and_q_gate_drift_before_allocation() {
    let fixture = fixture(SafeDtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&fixture).unwrap()).unwrap();
    let (snapshot, mut records, _, bf16, f32) = topology.into_upload_parts();

    let mut low_capacity = ample_capacity();
    low_capacity.host_available_bytes = 1;
    assert!(preflight_upload(&snapshot, &records, bf16, f32, test_limits(), low_capacity).is_err());

    let original_hash = records[0].source_byte_sha256.clone();
    records[0].source_byte_sha256 = "0".repeat(64);
    assert!(preflight_upload(
        &snapshot,
        &records,
        bf16,
        f32,
        test_limits(),
        ample_capacity()
    )
    .is_err());
    records[0].source_byte_sha256 = original_hash;

    let q = records
        .iter_mut()
        .find(|record| record.outputs.len() == 2)
        .unwrap();
    let Qwen35SourceTransformV1::SplitInterleavedQGate { head_dim, .. } =
        &mut q.outputs[1].transform
    else {
        panic!("gate output is not a split")
    };
    *head_dim += 1;
    let error = preflight_upload(
        &snapshot,
        &records,
        bf16,
        f32,
        test_limits(),
        ample_capacity(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("roles or parameters differ"));
}

#[test]
fn metal_upload_matches_all_transform_oracles_and_chunk_boundary() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };
    let fixture = fixture(SafeDtype::BF16, |config, specs| {
        const LARGE_VOCAB: usize = 524_289;
        config["text_config"]["vocab_size"] = serde_json::json!(LARGE_VOCAB);
        for tensor in specs.iter_mut().filter(|tensor| {
            tensor.name == "model.language_model.embed_tokens.weight"
                || tensor.name == "lm_head.weight"
        }) {
            tensor.shape = vec![LARGE_VOCAB, 4];
        }
    });
    let reference = open(&fixture)?;
    let topology = admit_qwen35_bf16_topology(open(&fixture)?)?;
    let records = topology.records_for_test().to_vec();
    let shard_path = fixture.temp.path().join("model.safetensors");
    let replacement_bytes = vec![0_u8; std::fs::metadata(&shard_path)?.len() as usize];
    std::fs::rename(
        &shard_path,
        fixture.temp.path().join("retained-original.safetensors"),
    )?;
    std::fs::write(&shard_path, replacement_bytes)?;
    // This test proves the exact Metal allocation and transform path. Keep
    // volatile process-wide capacity observations out of the oracle: the
    // broad suite runs many Metal tests concurrently and can legitimately
    // exhaust or temporarily hide the working-set observation.
    let upload = upload_with_capacity_for_test(
        topology,
        &device,
        test_limits(),
        ample_capacity(),
        |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
    )?;
    assert_eq!(upload.output_tensor_count(), 29);
    assert_eq!(
        upload.receipt_json_for_test()["device"]["registry_id"],
        device.registry_id()
    );
    assert_outputs_match_reference(&reference, &records, &upload);
    upload.validate_receipt_for_test()?;
    Ok(())
}

#[test]
fn injected_allocation_failure_alias_and_source_mutation_mint_no_upload() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };

    let capacity_fixture = fixture(SafeDtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&capacity_fixture)?)?;
    let mut allocation_calls = 0_usize;
    let mut low_capacity = ample_capacity();
    low_capacity.host_available_bytes = 1;
    assert!(upload_with_capacity_for_test(
        topology,
        &device,
        test_limits(),
        low_capacity,
        |bytes, dtype, shape| {
            allocation_calls += 1;
            Ok(device.alloc_buffer(bytes, dtype, shape)?)
        },
    )
    .is_err());
    assert_eq!(allocation_calls, 0);

    let allocation_fixture = fixture(SafeDtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&allocation_fixture)?)?;
    let mut calls = 0_usize;
    let error = upload_with_capacity_for_test(
        topology,
        &device,
        test_limits(),
        ample_capacity(),
        |bytes, dtype, shape| {
            calls += 1;
            if calls == 3 {
                bail!("injected allocation failure");
            }
            Ok(device.alloc_buffer(bytes, dtype, shape)?)
        },
    )
    .err()
    .expect("partial allocation must fail");
    assert!(format!("{error:#}").contains("injected allocation failure"));

    let alias_fixture = fixture(SafeDtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&alias_fixture)?)?;
    let mut shared_f32: Option<MlxBuffer> = None;
    let error = upload_with_capacity_for_test(
        topology,
        &device,
        test_limits(),
        ample_capacity(),
        |bytes, dtype, shape| {
            if bytes == 16 && dtype == DType::F32 && shape == [4] {
                if let Some(buffer) = &shared_f32 {
                    return Ok(buffer.clone());
                }
                let buffer = device.alloc_buffer(bytes, dtype, shape)?;
                shared_f32 = Some(buffer.clone());
                return Ok(buffer);
            }
            Ok(device.alloc_buffer(bytes, dtype, shape)?)
        },
    )
    .err()
    .expect("aliased allocation must fail before mutation");
    assert!(format!("{error:#}").contains("aliased allocation"));

    let mutation_fixture = fixture(SafeDtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&mutation_fixture)?)?;
    let shard_path = mutation_fixture.temp.path().join("model.safetensors");
    let mut bytes = std::fs::read(&shard_path)?;
    let last = bytes.last_mut().unwrap();
    *last ^= 0x5a;
    std::fs::write(&shard_path, bytes)?;
    assert!(upload_with_capacity_for_test(
        topology,
        &device,
        test_limits(),
        ample_capacity(),
        |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
    )
    .is_err());
    Ok(())
}

#[test]
fn fixed_capacity_receipt_is_canonical_and_records_nonexecuted_sources() -> Result<()> {
    let Some(device) = MlxDevice::new().ok() else {
        return Ok(());
    };
    let fixture = fixture(SafeDtype::BF16, |_, _| {});
    let first = upload_with_capacity_for_test(
        admit_qwen35_bf16_topology(open(&fixture)?)?,
        &device,
        test_limits(),
        ample_capacity(),
        |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
    )?;
    let second = upload_with_capacity_for_test(
        admit_qwen35_bf16_topology(open(&fixture)?)?,
        &device,
        test_limits(),
        ample_capacity(),
        |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
    )?;
    first.validate_receipt_for_test()?;
    second.validate_receipt_for_test()?;
    assert_eq!(first.catalog_sha256(), second.catalog_sha256());
    assert_eq!(first.receipt_sha256(), second.receipt_sha256());
    assert_eq!(first.total_output_bytes(), second.total_output_bytes());

    let receipt = first.receipt_json_for_test();
    let records = receipt["records"].as_array().unwrap();
    let mtp: Vec<_> = records
        .iter()
        .filter(|record| record["source_use"] == "authenticated_non_executed_mtp")
        .collect();
    let vision: Vec<_> = records
        .iter()
        .filter(|record| record["source_use"] == "excluded_vision")
        .collect();
    assert_eq!(mtp.len(), 15);
    assert_eq!(vision.len(), 1);
    assert!(mtp
        .iter()
        .all(|record| record["outputs"].as_array().unwrap().is_empty()));
    assert!(vision[0]["outputs"].as_array().unwrap().is_empty());
    assert_eq!(vision[0]["disposition"], "excluded");
    Ok(())
}
