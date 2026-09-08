use super::*;
use crate::backends::gguf::writer::GgufWriter;
use crate::quantize::ggml_quants::GgmlType as WriterGgmlType;
use crate::serve::forward_mlx_shared::MlxQWeight;
use mlx_native::MlxDevice;

#[test]
fn production_mapped_constructor_accepts_every_native_matrix_codec() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Some(device) = MlxDevice::new().ok() else {
        eprintln!("[skip] Metal device unavailable");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let rows = 4usize;
    let cols = 256usize;
    let codecs = [
        (GgmlType::F32, WriterGgmlType::F32, DType::F32),
        (GgmlType::F16, WriterGgmlType::F16, DType::F16),
        (GgmlType::BF16, WriterGgmlType::BF16, DType::BF16),
        (GgmlType::Q2_K, WriterGgmlType::Q2_K, DType::U8),
        (GgmlType::Q3_K, WriterGgmlType::Q3_K, DType::U8),
        (GgmlType::Q4_0, WriterGgmlType::Q4_0, DType::U8),
        (GgmlType::Q5_0, WriterGgmlType::Q5_0, DType::U8),
        (GgmlType::Q5_1, WriterGgmlType::Q5_1, DType::U8),
        (GgmlType::Q8_0, WriterGgmlType::Q8_0, DType::U8),
        (GgmlType::Q4_K, WriterGgmlType::Q4_K, DType::U8),
        (GgmlType::Q5_K, WriterGgmlType::Q5_K, DType::U8),
        (GgmlType::Q6_K, WriterGgmlType::Q6_K, DType::U8),
        (GgmlType::IQ4_NL, WriterGgmlType::IQ4_NL, DType::U8),
        (GgmlType::IQ4_XS, WriterGgmlType::IQ4_XS, DType::U8),
    ];

    for (index, (runtime_codec, writer_codec, storage_dtype)) in codecs.into_iter().enumerate() {
        let path = tmp.path().join(format!("codec-{index}.gguf"));
        let byte_len = native_gguf_matrix_bytes(runtime_codec, rows, cols).unwrap();
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut writer = GgufWriter::new(file);
            writer.write_header(1, 0).unwrap();
            let tensor = writer
                .reserve_tensor_info("output.weight", &[cols as u64, rows as u64], writer_codec)
                .unwrap();
            writer.pad_to_alignment().unwrap();
            writer
                .stream_tensor_payload(tensor, &vec![0u8; byte_len])
                .unwrap();
            writer.finalize().unwrap();
        }

        let gguf = mlx_native::gguf::GgufFile::open(&path).unwrap();
        let info = gguf.tensor_info("output.weight").unwrap();
        let mapped = gguf.map_tensor_data(&device).unwrap();
        let weight = MlxQWeight::from_mapped_gguf_tensor(&mapped, info)
            .unwrap_or_else(|error| panic!("{runtime_codec:?} mapped load failed: {error}"));
        assert_eq!(weight.info.ggml_dtype, runtime_codec);
        assert_eq!((weight.info.rows, weight.info.cols), (rows, cols));
        assert_eq!(weight.buffer.dtype(), storage_dtype);
        assert_eq!(weight.buffer.data_byte_len(), byte_len);
        assert!(weight.buffer.is_file_backed());
    }
}

#[test]
fn production_mapped_constructor_rejects_malformed_shape_and_extent() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Some(device) = MlxDevice::new().ok() else {
        eprintln!("[skip] Metal device unavailable");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("malformed-q5k.gguf");
    let rows = 4usize;
    let cols = 256usize;
    let byte_len = native_gguf_matrix_bytes(GgmlType::Q5_K, rows, cols).unwrap();
    {
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = GgufWriter::new(file);
        writer.write_header(1, 0).unwrap();
        let tensor = writer
            .reserve_tensor_info(
                "output.weight",
                &[cols as u64, rows as u64],
                WriterGgmlType::Q5_K,
            )
            .unwrap();
        writer.pad_to_alignment().unwrap();
        writer
            .stream_tensor_payload(tensor, &vec![0u8; byte_len])
            .unwrap();
        writer.finalize().unwrap();
    }

    let gguf = mlx_native::gguf::GgufFile::open(&path).unwrap();
    let mapped = gguf.map_tensor_data(&device).unwrap();
    let info = gguf.tensor_info("output.weight").unwrap();

    let mut rank_three = info.clone();
    rank_three.shape = vec![rows, 1, cols];
    assert!(MlxQWeight::from_mapped_gguf_tensor(&mapped, &rank_three)
        .err()
        .expect("rank-three descriptor must fail")
        .to_string()
        .contains("must be rank 2"));

    assert!(
        MlxQWeight::from_mapped_gguf_matrix_view(&mapped, info, rows + 1, cols)
            .err()
            .expect("mismatched declared geometry must fail")
            .to_string()
            .contains("not declared")
    );

    let mut truncated = info.clone();
    truncated.byte_len -= 1;
    assert!(MlxQWeight::from_mapped_gguf_tensor(&mapped, &truncated)
        .err()
        .expect("truncated descriptor must fail")
        .to_string()
        .contains("metadata has"));

    let mut mismatched_mapping = info.clone();
    mismatched_mapping.shape = vec![1, cols];
    mismatched_mapping.byte_len = native_gguf_matrix_bytes(GgmlType::Q5_K, 1, cols).unwrap();
    assert!(
        MlxQWeight::from_mapped_gguf_tensor(&mapped, &mismatched_mapping)
            .err()
            .expect("mapped extent mismatch must fail")
            .to_string()
            .contains("did not retain its file-backed")
    );
}

#[test]
fn native_matrix_mapped_lifecycle_is_path_independent_across_a_b_a() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Some(device) = MlxDevice::new().ok() else {
        eprintln!("[skip] Metal device unavailable");
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let rows = 4usize;
    let cols = 256usize;

    let load_once = |name: &str, base: f32| {
        let path = tmp.path().join(name);
        let payload: Vec<f32> = (0..rows * cols).map(|value| base + value as f32).collect();
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
            writer
                .stream_tensor_payload(tensor, bytemuck::cast_slice(&payload))
                .unwrap();
            writer.finalize().unwrap();
        }

        let gguf = mlx_native::gguf::GgufFile::open(&path).unwrap();
        let info = gguf.tensor_info("token_embd.weight").unwrap();
        let mapped = gguf.map_tensor_data(&device).unwrap();
        let weight = MlxQWeight::from_mapped_gguf_tensor(&mapped, info).unwrap();
        drop(mapped);
        drop(gguf);
        std::fs::remove_file(&path).unwrap();

        let tied_head = weight.clone();
        let storage = summarize_native_matrix_storage([&weight.buffer, &tied_head.buffer]).unwrap();
        assert_eq!(storage.unique_matrix_views, 1);
        assert_eq!(storage.file_backed_bytes, (rows * cols * 4) as u64);
        assert_eq!(storage.anonymous_bytes, 0);

        let retained = weight.buffer.as_slice::<f32>().unwrap();
        assert_eq!(retained[0], base);
        assert_eq!(retained[rows * cols - 1], base + (rows * cols - 1) as f32);
        drop(weight);
    };

    // Only one mapping is resident at a time. Reopening A at the same path
    // after B must derive state solely from the new A artifact bytes.
    load_once("model-a.gguf", 1000.0);
    load_once("model-b.gguf", 2000.0);
    load_once("model-a.gguf", 1000.0);
}
