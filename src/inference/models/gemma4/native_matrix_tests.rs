use super::*;
use mlx_native::MlxDevice;

fn stored_bytes(dtype: GgmlType, rows: usize, cols: usize) -> usize {
    match dtype {
        GgmlType::F32 => rows * cols * std::mem::size_of::<f32>(),
        GgmlType::F16 | GgmlType::BF16 => rows * cols * std::mem::size_of::<u16>(),
        _ => mlx_native::ggml_matrix_bytes(dtype, rows as u32, cols as u32).unwrap() as usize,
    }
}

fn info(
    ggml_type: GgmlType,
    rows: usize,
    cols: usize,
    byte_len: usize,
) -> mlx_native::gguf::TensorInfo {
    mlx_native::gguf::TensorInfo {
        name: "token_embd.weight".into(),
        shape: vec![rows, cols],
        ggml_type,
        offset: 0,
        byte_len,
    }
}

#[test]
fn admitted_embedding_types_are_exact_native_intersection() {
    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q4_0,
        GgmlType::Q5_0,
        GgmlType::Q2_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q8_0,
    ] {
        let bytes = stored_bytes(dtype, 4, 256);
        let admitted = admit_embedding("token_embd.weight", &info(dtype, 4, 256, bytes), 4, 256)
            .unwrap_or_else(|error| panic!("{dtype:?} should be native: {error}"));
        assert_eq!(admitted.ggml_type, dtype);
        assert_eq!(admitted.byte_len, bytes);
    }
}

#[test]
fn native_matrix_embedding_dispatches_every_admitted_codec() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Some(device) = MlxDevice::new().ok() else {
        eprintln!("[skip] Metal device unavailable");
        return;
    };
    let rows = 4usize;
    let cols = 256usize;
    let n_tokens = 2usize;
    let mut ids = device
        .alloc_buffer(
            n_tokens * std::mem::size_of::<u32>(),
            mlx_native::DType::U32,
            vec![n_tokens],
        )
        .unwrap();
    ids.as_mut_slice::<u32>().unwrap().copy_from_slice(&[0, 3]);

    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q4_0,
        GgmlType::Q5_0,
        GgmlType::Q2_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q8_0,
    ] {
        let byte_len = stored_bytes(dtype, rows, cols);
        let storage_dtype = match dtype {
            GgmlType::F32 => mlx_native::DType::F32,
            GgmlType::F16 => mlx_native::DType::F16,
            GgmlType::BF16 => mlx_native::DType::BF16,
            _ => mlx_native::DType::U8,
        };
        let storage_shape = if storage_dtype == mlx_native::DType::U8 {
            vec![byte_len]
        } else {
            vec![rows, cols]
        };
        let mut stored = device
            .alloc_buffer(byte_len, storage_dtype, storage_shape)
            .unwrap();
        match storage_dtype {
            mlx_native::DType::F32 => stored.as_mut_slice::<f32>().unwrap().fill(0.0),
            mlx_native::DType::F16 | mlx_native::DType::BF16 => {
                stored.as_mut_slice::<u16>().unwrap().fill(0)
            }
            mlx_native::DType::U8 => stored.as_mut_slice::<u8>().unwrap().fill(0),
            other => panic!("unexpected embedding storage dtype {other:?}"),
        }
        let weight = MlxQWeight {
            buffer: stored,
            info: crate::serve::gpu::QuantWeightInfo {
                ggml_dtype: dtype,
                rows,
                cols,
            },
            affine: None,
            f16_shadow: None,
            decode_record_q6k_m1: std::sync::OnceLock::new(),
        };
        let output = device
            .alloc_buffer(
                n_tokens * cols * std::mem::size_of::<f32>(),
                mlx_native::DType::F32,
                vec![n_tokens, cols],
            )
            .unwrap();
        let mut registry = mlx_native::KernelRegistry::new();
        let executor = mlx_native::GraphExecutor::new(device.clone());
        let mut session = executor.begin().unwrap();
        encode_embedding_rows(
            &mut session,
            &mut registry,
            &device,
            &weight,
            &ids,
            &output,
            n_tokens,
        )
        .unwrap_or_else(|error| panic!("{dtype:?} embedding dispatch failed: {error}"));
        session.finish().unwrap();
        assert!(
            output
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0),
            "{dtype:?} zero embedding did not produce numeric zeros"
        );
    }
}

#[test]
fn unsupported_and_malformed_embeddings_fail_before_loading() {
    let unsupported = info(GgmlType::Q3_K, 4, 256, 440);
    assert!(admit_embedding("token_embd.weight", &unsupported, 4, 256)
        .unwrap_err()
        .to_string()
        .contains("rejects stored Q3_K"));

    let mut malformed = info(GgmlType::Q6_K, 4, 256, 1);
    assert!(admit_embedding("token_embd.weight", &malformed, 4, 256)
        .unwrap_err()
        .to_string()
        .contains("payload is 1 bytes"));
    malformed.shape = vec![4, 2, 128];
    assert!(admit_embedding("token_embd.weight", &malformed, 4, 256)
        .unwrap_err()
        .to_string()
        .contains("must be rank 2"));

    let wrong_projection_shape = info(GgmlType::Q6_K, 8, 256, stored_bytes(GgmlType::Q6_K, 8, 256));
    assert!(
        admit_projection("blk.0.attn_q.weight", &wrong_projection_shape, 4, 256)
            .unwrap_err()
            .to_string()
            .contains("does not match expected [4, 256]")
    );
}

#[test]
fn projections_admit_every_native_dense_codec_without_transforming_it() {
    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q4_0,
        GgmlType::Q5_0,
        GgmlType::Q8_0,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q5_1,
        GgmlType::IQ4_NL,
        GgmlType::IQ4_XS,
    ] {
        let bytes = stored_bytes(dtype, 4, 256);
        let admitted = admit_projection("output.weight", &info(dtype, 4, 256, bytes), 4, 256)
            .unwrap_or_else(|error| panic!("{dtype:?} should be native: {error}"));
        assert_eq!(admitted.ggml_type, dtype);
        assert_eq!(admitted.byte_len, bytes);
    }
}

#[test]
fn native_matrix_projection_dispatches_every_admitted_codec_and_width() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Some(device) = MlxDevice::new().ok() else {
        eprintln!("[skip] Metal device unavailable");
        return;
    };
    let rows = 4usize;
    let cols = 256usize;
    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q4_0,
        GgmlType::Q5_0,
        GgmlType::Q8_0,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q5_1,
        GgmlType::IQ4_NL,
        GgmlType::IQ4_XS,
    ] {
        let byte_len = stored_bytes(dtype, rows, cols);
        let storage_dtype = match dtype {
            GgmlType::F32 => mlx_native::DType::F32,
            GgmlType::F16 => mlx_native::DType::F16,
            GgmlType::BF16 => mlx_native::DType::BF16,
            _ => mlx_native::DType::U8,
        };
        let storage_shape = if storage_dtype == mlx_native::DType::U8 {
            vec![byte_len]
        } else {
            vec![rows, cols]
        };
        let mut stored = device
            .alloc_buffer(byte_len, storage_dtype, storage_shape)
            .unwrap();
        match storage_dtype {
            mlx_native::DType::F32 => stored.as_mut_slice::<f32>().unwrap().fill(0.0),
            mlx_native::DType::F16 | mlx_native::DType::BF16 => {
                stored.as_mut_slice::<u16>().unwrap().fill(0)
            }
            mlx_native::DType::U8 => stored.as_mut_slice::<u8>().unwrap().fill(0),
            other => panic!("unexpected projection storage dtype {other:?}"),
        }
        let weight = MlxQWeight {
            buffer: stored,
            info: crate::serve::gpu::QuantWeightInfo {
                ggml_dtype: dtype,
                rows,
                cols,
            },
            affine: None,
            f16_shadow: None,
            decode_record_q6k_m1: std::sync::OnceLock::new(),
        };
        assert_eq!(
            supports_native_perm021(&weight, 9, 256),
            matches!(
                dtype,
                GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 | GgmlType::Q6_K
            ),
            "{dtype:?} perm021 admission drifted from the runtime fallback contract"
        );
        for m in [1u32, 2, 9] {
            let mut input = device
                .alloc_buffer(
                    m as usize * cols * std::mem::size_of::<f32>(),
                    mlx_native::DType::F32,
                    vec![m as usize, cols],
                )
                .unwrap();
            input.as_mut_slice::<f32>().unwrap().fill(0.0);
            let output = device
                .alloc_buffer(
                    m as usize * rows * std::mem::size_of::<f32>(),
                    mlx_native::DType::F32,
                    vec![m as usize, rows],
                )
                .unwrap();
            let mut registry = mlx_native::KernelRegistry::new();
            let executor = mlx_native::GraphExecutor::new(device.clone());
            let mut session = executor.begin().unwrap();
            crate::serve::forward_mlx_shared::dispatch_qmatmul(
                &mut session,
                &mut registry,
                &device,
                &input,
                &weight,
                &output,
                m,
                crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
            )
            .unwrap_or_else(|error| {
                panic!("{dtype:?} projection dispatch failed at m={m}: {error}")
            });
            session.finish().unwrap();
            assert!(
                output
                    .as_slice::<f32>()
                    .unwrap()
                    .iter()
                    .all(|value| *value == 0.0),
                "{dtype:?} zero projection at m={m} did not produce numeric zeros"
            );
        }

        // Exercise the exact production O-projection helper, not just its
        // capability predicate: direct codecs consume head-major BF16;
        // every other admitted codec takes the activation-only fallback.
        let m = 9u32;
        let mut head_major = device
            .alloc_buffer(
                m as usize * cols * std::mem::size_of::<u16>(),
                mlx_native::DType::BF16,
                vec![1, m as usize, cols],
            )
            .unwrap();
        head_major.as_mut_slice::<u16>().unwrap().fill(0);
        let scratch = device
            .alloc_buffer(
                m as usize * cols * std::mem::size_of::<f32>(),
                mlx_native::DType::F32,
                vec![m as usize, cols],
            )
            .unwrap();
        let output = device
            .alloc_buffer(
                m as usize * rows * std::mem::size_of::<f32>(),
                mlx_native::DType::F32,
                vec![m as usize, rows],
            )
            .unwrap();
        let mut registry = mlx_native::KernelRegistry::new();
        let executor = mlx_native::GraphExecutor::new(device.clone());
        let mut session = executor.begin().unwrap();
        let route = crate::serve::forward_mlx_shared::dispatch_qmatmul_head_major_bf16(
            &mut session,
            &mut registry,
            &device,
            &head_major,
            &scratch,
            &weight,
            &output,
            m,
            1,
            cols,
            crate::quantize::imatrix::ImatrixHint::Global("output.weight"),
        )
        .unwrap_or_else(|error| panic!("{dtype:?} production O-projection route failed: {error}"));
        session.finish().unwrap();
        let expected_route = if matches!(
            dtype,
            GgmlType::Q4_0 | GgmlType::Q5_0 | GgmlType::Q8_0 | GgmlType::Q6_K
        ) {
            crate::serve::forward_mlx_shared::HeadMajorQmatmulRoute::DirectPerm021
        } else {
            crate::serve::forward_mlx_shared::HeadMajorQmatmulRoute::ActivationPermute
        };
        assert_eq!(route, expected_route, "{dtype:?} O-projection route drift");
        assert!(
            output
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0),
            "{dtype:?} production O-projection did not produce numeric zeros"
        );
    }
}

#[test]
fn tied_resolution_reuses_the_embedding_object() {
    let embedding = String::from("artifact-native embedding");
    let explicit = String::from("artifact-native output");
    assert!(std::ptr::eq(
        resolve_tied_or_explicit(&embedding, None),
        &embedding
    ));
    assert!(std::ptr::eq(
        resolve_tied_or_explicit(&embedding, Some(&explicit)),
        &explicit
    ));
}

#[test]
fn expert_stacks_admit_native_scalar_and_block_codecs() {
    let experts = 8;
    let rows = 512;
    let cols = 256;
    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q4_0,
        GgmlType::Q5_0,
        GgmlType::Q8_0,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q5_1,
        GgmlType::IQ4_NL,
        GgmlType::IQ4_XS,
    ] {
        let mut tensor = info(dtype, rows, cols, experts * stored_bytes(dtype, rows, cols));
        tensor.name = "blk.0.ffn_gate_up_exps.weight".into();
        tensor.shape = vec![experts, rows, cols];
        for role in [ExpertMatrixRole::GateUp, ExpertMatrixRole::Down] {
            admit_expert_stack(&tensor.name, &tensor, experts, rows, cols, 8, role).unwrap_or_else(
                |error| panic!("{dtype:?} {role:?} expert stack should be native: {error}"),
            );
        }
    }
}

#[test]
fn mapped_weight_owns_file_pages_across_an_unlinked_fixture() {
    use crate::backends::gguf::writer::GgufWriter;
    use crate::quantize::ggml_quants::GgmlType as WriterGgmlType;

    let Some(device) = MlxDevice::new().ok() else {
        eprintln!("[skip] Metal device unavailable");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("native-matrix.gguf");
    let rows = 4usize;
    let cols = 256usize;
    let payload: Vec<f32> = (0..rows * cols).map(|value| value as f32).collect();
    let payload_bytes: &[u8] = bytemuck::cast_slice(&payload);
    {
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = GgufWriter::new(file);
        writer.write_header(1, 0).unwrap();
        let tensor = writer
            .reserve_tensor_info(
                "token_embd.weight",
                &[cols as u64, rows as u64],
                WriterGgmlType::F32,
            )
            .unwrap();
        writer.pad_to_alignment().unwrap();
        writer.stream_tensor_payload(tensor, payload_bytes).unwrap();
        writer.finalize().unwrap();
    }

    let gguf = mlx_native::gguf::GgufFile::open(&path).unwrap();
    let spec = admit_embedding(
        "token_embd.weight",
        gguf.tensor_info("token_embd.weight").unwrap(),
        rows,
        cols,
    )
    .unwrap();
    let mapped = gguf.map_tensor_data(&device).unwrap();
    let weight =
        MlxQWeight::from_mapped_gguf_tensor(&mapped, gguf.tensor_info(&spec.name).unwrap())
            .unwrap();
    assert!(weight.buffer.is_file_backed());
    assert_eq!(weight.buffer.data_byte_len(), payload_bytes.len());

    drop(mapped);
    drop(gguf);
    std::fs::remove_file(&path).unwrap();
    let retained = weight.buffer.as_slice::<f32>().unwrap();
    assert_eq!(retained[0], 0.0);
    assert_eq!(retained[513], 513.0);
    assert_eq!(retained[rows * cols - 1], (rows * cols - 1) as f32);
}
