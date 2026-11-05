//! Nonzero numerical proof for the publisher's Q8 embedding/head contract.

use super::batched_body::dispatch_dense_rowident;
use super::native_matrix::{encode_embedding, preflight_io};
use crate::backends::gguf::writer::GgufWriter;
use crate::quantize::ggml_quants::GgmlType as StoredType;
use crate::quantize::imatrix::ImatrixHint;
use crate::serve::forward_mlx_shared::MlxQWeight;
use mlx_native::{DType, GraphExecutor, KernelRegistry, MlxDevice};

#[test]
fn native_q8_embedding_and_batched_head_match_stored_values() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal required for native Gemma IO proof");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("native-q8-io.gguf");
    let (vocab, hidden) = (32_usize, 256_usize);
    // Construct Q8_0 blocks directly: a half-precision scale followed by 32
    // signed codes. The CPU oracle below uses these declared values, without
    // calling either hf2q's quantizer or the backend's dequantizer.
    let mut bytes = Vec::new();
    let mut values = Vec::new();
    for row in 0..vocab {
        for block in 0..hidden / 32 {
            let scale = (block + 1) as f32 / 16.0;
            bytes.extend(half::f16::from_f32(scale).to_bits().to_le_bytes());
            for col in 0..32 {
                let code = ((row * 7 + block * 3 + col) % 31) as i8 - 15;
                bytes.push(code as u8);
                values.push(scale * f32::from(code));
            }
        }
    }
    {
        let mut writer = GgufWriter::new(std::fs::File::create(&path).unwrap());
        writer.write_header(1, 0).unwrap();
        let tensor = writer
            .reserve_tensor_info(
                "token_embd.weight",
                &[hidden as u64, vocab as u64],
                StoredType::Q8_0,
            )
            .unwrap();
        writer.pad_to_alignment().unwrap();
        writer.stream_tensor_payload(tensor, &bytes).unwrap();
        writer.finalize().unwrap();
    }
    let gguf = mlx_native::gguf::GgufFile::open(&path).unwrap();
    let plan = preflight_io(&gguf, vocab, hidden).unwrap();
    assert!(plan.output.is_none());
    let mapped = gguf.map_tensor_data(&device).unwrap();
    let weight = MlxQWeight::from_mapped_gguf_tensor(
        &mapped,
        gguf.tensor_info(&plan.embedding.name).unwrap(),
    )
    .unwrap();
    let executor = GraphExecutor::new(device.clone());
    let mut registry = KernelRegistry::new();
    for m in [1_usize, 2, 4, 8, 9, 32] {
        let ids: Vec<u32> = (0..m).map(|i| ((i * 5 + 3) % vocab) as u32).collect();
        let mut token_ids = device.alloc_buffer(m * 4, DType::U32, vec![m]).unwrap();
        token_ids
            .as_mut_slice::<u32>()
            .unwrap()
            .copy_from_slice(&ids);
        let gathered = device
            .alloc_buffer(m * hidden * 4, DType::F32, vec![m, hidden])
            .unwrap();
        let logits = device
            .alloc_buffer(m * vocab * 4, DType::F32, vec![m, vocab])
            .unwrap();
        let mut session = executor.begin().unwrap();
        encode_embedding(
            &mut session,
            &mut registry,
            &device,
            &weight,
            &token_ids,
            &gathered,
            m,
        )
        .unwrap();
        session.barrier_between(&[&gathered, &weight.buffer], &[&logits]);
        dispatch_dense_rowident(
            &mut session,
            &mut registry,
            &device,
            &gathered,
            &weight,
            &logits,
            m,
            hidden,
            vocab,
            ImatrixHint::None,
        )
        .unwrap();
        session.finish().unwrap();
        let actual_gather = gathered.as_slice::<f32>().unwrap();
        let actual_logits = logits.as_slice::<f32>().unwrap();
        for (row, id) in ids.iter().enumerate() {
            let reference_row = &values[*id as usize * hidden..(*id as usize + 1) * hidden];
            for k in 0..hidden {
                assert_eq!(actual_gather[row * hidden + k], reference_row[k] * 16.0);
            }
            for out in 0..vocab {
                let expected: f64 = (0..hidden)
                    .map(|k| {
                        f64::from(reference_row[k] * 16.0) * f64::from(values[out * hidden + k])
                    })
                    .sum();
                let actual = f64::from(actual_logits[row * vocab + out]);
                assert!(
                    (actual - expected).abs() <= 1e-4,
                    "m={m} row={row} output={out}: {actual} != {expected}"
                );
            }
        }
        assert!(weight.buffer.is_file_backed());
        assert!(weight.f16_shadow.is_none());
        assert_eq!(weight.buffer.as_slice::<u8>().unwrap(), bytes);
    }
}
