use super::*;
use crate::backends::gguf::writer::GgufWriter;
use crate::quantize::ggml_quants::GgmlType;
use std::fs;

#[test]
fn native_q8_head_requires_selected_route_and_q8_source() {
    assert!(native_q8_head_selected(
        true,
        Some(mlx_native::GgmlType::Q8_0)
    ));
    assert!(!native_q8_head_selected(
        false,
        Some(mlx_native::GgmlType::Q8_0)
    ));
    assert!(!native_q8_head_selected(
        true,
        Some(mlx_native::GgmlType::Q6_K)
    ));
    assert!(!native_q8_head_selected(true, None));
}

#[test]
fn native_q8_head_loader_preserves_original_blocks() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = mlx_native::MlxDevice::new() else {
        eprintln!("skipping native Q8 head loader test: no MlxDevice");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-q8-head.gguf");
    let original: Vec<u8> = (0..68).map(|index| (index * 37 + 11) as u8).collect();
    let mut writer = GgufWriter::new(fs::File::create(&path).unwrap());
    writer.write_header(1, 0).unwrap();
    writer
        .reserve_tensor_info("token_embd.weight", &[32, 2], GgmlType::Q8_0)
        .unwrap();
    writer.pad_to_alignment().unwrap();
    writer.stream_tensor_payload(0, &original).unwrap();
    writer.finalize().unwrap();

    let gguf = mlx_native::gguf::GgufFile::open(&path).unwrap();
    let loaded = load_gguf_qweight(&gguf, "token_embd.weight", &device).unwrap();
    assert_eq!(loaded.info.ggml_dtype, mlx_native::GgmlType::Q8_0);
    assert_eq!((loaded.info.rows, loaded.info.cols), (2, 32));
    assert_eq!(loaded.buffer.dtype(), mlx_native::DType::U8);
    assert_eq!(loaded.buffer.as_slice::<u8>().unwrap(), original);
    assert!(loaded.f16_shadow.is_none());
}

#[test]
fn native_bf16_gguf_loader_preserves_original_storage() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = mlx_native::MlxDevice::new() else {
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("scalar.gguf");
    let values = [1.125_f32, -2.5, 0.03125, 7.0, -0.75, 16.0];
    let original = values.map(|v| half::bf16::from_f32(v).to_bits());
    let bytes = original
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let mut writer = GgufWriter::new(fs::File::create(&path).unwrap());
    writer.write_header(1, 0).unwrap();
    writer
        .reserve_tensor_info("projection.weight", &[3, 2], GgmlType::BF16)
        .unwrap();
    writer.pad_to_alignment().unwrap();
    writer.stream_tensor_payload(0, &bytes).unwrap();
    writer.finalize().unwrap();
    let gguf = mlx_native::gguf::GgufFile::open(&path).unwrap();
    let loaded = load_gguf_qweight(&gguf, "projection.weight", &device).unwrap();
    assert_eq!(loaded.info.ggml_dtype, mlx_native::GgmlType::BF16);
    assert_eq!(loaded.buffer.dtype(), mlx_native::DType::BF16);
    assert_eq!(loaded.buffer.as_slice::<u16>().unwrap(), original);
    assert!(loaded.f16_shadow.is_none());
}
