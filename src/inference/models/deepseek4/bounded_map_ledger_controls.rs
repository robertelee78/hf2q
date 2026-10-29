use super::*;
use crate::backends::gguf::writer::GgufWriter;
use crate::quantize::ggml_quants::GgmlType;
fn fixture(payload: bool) -> tempfile::NamedTempFile {
    let f = tempfile::NamedTempFile::new().unwrap();
    let kv = super::super::config::tests::official_metadata();
    let mut writer = GgufWriter::new(f.reopen().unwrap());
    writer.write_header(1, kv.len() as u64).unwrap();
    for (key, value) in kv {
        writer.write_metadata_kv(&key, &value).unwrap();
    }
    let index = writer
        .reserve_tensor_info("unused", &[1], GgmlType::F32)
        .unwrap();
    writer.pad_to_alignment().unwrap();
    writer
        .stream_tensor_payload(index, &1f32.to_le_bytes())
        .unwrap();
    writer.finalize().unwrap();
    if !payload {
        let g = GgufFile::from_file_with_limits(f.reopen().unwrap(), limits()).unwrap();
        f.as_file().set_len(g.tensor_data_offset()).unwrap();
    }
    f
}
fn geometry() -> Geometry {
    Geometry {
        page_bytes: 16384,
        max_buffer_bytes: 86586540032,
    }
}
#[test]
fn actual_bounded_entry_reports_unused_required_and_native_exact_plan_without_device() {
    let f = fixture(true);
    let out = inspect(open(f.path()).unwrap(), geometry()).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["catalog_exact"], false);
    assert_eq!(v["all_count"], 1);
    assert_eq!(v["unused_extras"][0], "unused");
    assert!(v["missing_required"].as_array().unwrap().len() > 10);
    assert_eq!(v["all_logical_file_bytes"], 4);
    assert_eq!(v["unused_logical_file_bytes"], 4);
    assert_eq!(v["required_present_count"], 0);
    assert_eq!(v["anonymous_f32_requested_bytes"], 0);
    assert_eq!(v["all_coalesced_root_requested_bytes"], 16384);
    assert_eq!(v["registered_file_page_union_bytes"], 16384);
    assert_eq!(v["physical_credit_bytes"], 0);
    assert_eq!(v["device_created"], false);
    assert_eq!(
        v["geometry_authority"],
        "UnverifiedGeometry_CallerAssumed_NotDeviceCapability"
    );
    assert!(v["header_read_bytes"].as_u64().unwrap() <= 64 * MIB);
}
#[test]
fn actual_entry_refuses_missing_payload_extent_without_reading_it() {
    let f = fixture(false);
    assert!(inspect(open(f.path()).unwrap(), geometry()).is_err());
}
#[test]
fn actual_geometry_and_output_caps_refuse_bad_inputs() {
    for g in [
        Geometry {
            page_bytes: 0,
            ..geometry()
        },
        Geometry {
            page_bytes: 3000,
            ..geometry()
        },
        Geometry {
            max_buffer_bytes: 1,
            ..geometry()
        },
        Geometry {
            max_buffer_bytes: usize::MAX,
            ..geometry()
        },
    ] {
        assert!(g.validate().is_err());
    }
    assert!(encode(&"x".repeat(OUTPUT_BYTES)).is_err());
    assert!(cpu_reservation(MAX_TENSORS + 1).is_err());
    assert!(cpu_reservation(MAX_TENSORS).unwrap() <= 16 * MIB);
    let mut sum = u64::MAX;
    assert!(add(&mut sum, 1).is_err());
}
#[test]
fn actual_regular_fd_boundary_refuses_symlink_and_fifo_before_read() {
    use std::{ffi::CString, os::unix::fs::symlink};
    let d = tempfile::tempdir().unwrap();
    let file = d.path().join("file");
    std::fs::write(&file, b"x").unwrap();
    let alias = d.path().join("alias");
    symlink(&file, &alias).unwrap();
    assert!(open(&alias).is_err());
    let fifo = d.path().join("fifo");
    let name = CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: valid nul-terminated owned path, creates only an isolated control FIFO.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(open(&fifo).is_err());
}
#[test]
fn actual_production_storage_predicate_remains_role_specific() {
    use mlx_native::ops::quantized_matmul_ggml::GgmlType as Native;
    let spec = super::super::weights::TensorSpec {
        name: "lookup".into(),
        shape: vec![1],
        role: TensorRole::IntegerLookupI32,
    };
    assert!(super::super::residency::metadata_storage_valid(
        &spec,
        Native::I32
    ));
    assert!(!super::super::residency::metadata_storage_valid(
        &spec,
        Native::F32
    ));
}

#[test]
fn actual_complete_tiny_catalog_reports_extra_f32_roots_separately_from_mapped_all() {
    let mut cfg = super::super::residency_tests::tiny_config();
    cfg.max_position_embeddings = 262_144;
    let (dir, header) = super::super::residency_tests::open_fixture(&cfg, false, false);
    let expected = super::super::weights::required_tensor_specs(&cfg);
    let anon = expected
        .iter()
        .filter(|s| s.role == TensorRole::ElementwiseF32)
        .map(|s| s.shape.iter().product::<usize>() * 4)
        .sum::<usize>();
    drop(header);
    let out = inspect(open(&dir.path().join("weights.gguf")).unwrap(), geometry()).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["catalog_exact"], true);
    assert_eq!(v["required_count"].as_u64(), Some(expected.len() as u64));
    assert_eq!(v["required_present_count"], v["required_count"]);
    assert_eq!(v["missing_required"], serde_json::json!([]));
    assert_eq!(v["unused_extras"], serde_json::json!([]));
    assert_eq!(
        v["anonymous_f32_requested_bytes"].as_u64(),
        Some(anon as u64)
    );
    assert!(anon > 0);
    assert!(v["all_coalesced_root_requested_bytes"].as_u64().unwrap() > 0);
}
