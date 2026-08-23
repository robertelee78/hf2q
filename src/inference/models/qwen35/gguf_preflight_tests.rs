use mlx_native::GgmlType;

use super::*;
use crate::backends::gguf::writer::GgufWriter;
use crate::quantize::ggml_quants::GgmlType as WriterGgmlType;

fn row_projection_fixture(dimensions: &[u64]) -> tempfile::NamedTempFile {
    let file = tempfile::NamedTempFile::new().expect("row projection fixture");
    let sink = std::fs::File::create(file.path()).expect("create row projection fixture");
    let mut writer = GgufWriter::new(sink);
    writer.write_header(1, 0).expect("header");
    let tensor = writer
        .reserve_tensor_info(
            "blk.0.ffn_gate_inp_shexp.weight",
            dimensions,
            WriterGgmlType::F32,
        )
        .expect("tensor info");
    writer.pad_to_alignment().expect("alignment");
    let elements = dimensions.iter().product::<u64>() as usize;
    writer
        .stream_tensor_payload(tensor, &vec![0; elements * 4])
        .expect("tensor payload");
    writer.finalize().expect("finalize");
    file
}

#[derive(Clone, Copy)]
struct ProfileEntry {
    count: usize,
    role: TensorRole,
    storage: TensorStorage,
}

fn validate_model_free_profile(entries: &[ProfileEntry]) -> Result<Qwen35GgufPreflightReceipt> {
    let mut receipt = Qwen35GgufPreflightReceipt::default();
    for entry in entries {
        for index in 0..entry.count {
            let name = format!("profile.{}.{}", entry.role.label(), index);
            admit_storage_for_role(&name, entry.role, entry.storage)?;
            receipt.record(entry.role, entry.storage);
        }
    }
    Ok(receipt)
}

fn mixed_k_profile(primary: GgmlType) -> Vec<ProfileEntry> {
    vec![
        ProfileEntry {
            count: 360,
            role: TensorRole::F32State,
            storage: TensorStorage::Parsed(GgmlType::F32),
        },
        ProfileEntry {
            count: 1,
            role: TensorRole::Embedding,
            storage: TensorStorage::Parsed(primary),
        },
        ProfileEntry {
            count: 1,
            role: TensorRole::DenseProjection,
            storage: TensorStorage::Parsed(GgmlType::Q6_K),
        },
        ProfileEntry {
            count: 276,
            role: TensorRole::DenseProjection,
            storage: TensorStorage::Parsed(primary),
        },
        ProfileEntry {
            count: 33,
            role: TensorRole::DenseProjection,
            storage: TensorStorage::Parsed(GgmlType::Q6_K),
        },
        ProfileEntry {
            count: 130,
            role: TensorRole::FfnGateUp,
            storage: TensorStorage::Parsed(primary),
        },
        ProfileEntry {
            count: 32,
            role: TensorRole::FfnDown,
            storage: TensorStorage::Parsed(primary),
        },
        ProfileEntry {
            count: 33,
            role: TensorRole::FfnDown,
            storage: TensorStorage::Parsed(GgmlType::Q6_K),
        },
    ]
}

fn uniform_profile(storage: TensorStorage) -> Vec<ProfileEntry> {
    vec![
        ProfileEntry {
            count: 360,
            role: TensorRole::F32State,
            storage: TensorStorage::Parsed(GgmlType::F32),
        },
        ProfileEntry {
            count: 1,
            role: TensorRole::Embedding,
            storage,
        },
        ProfileEntry {
            count: 310,
            role: TensorRole::DenseProjection,
            storage,
        },
        ProfileEntry {
            count: 130,
            role: TensorRole::FfnGateUp,
            storage,
        },
        ProfileEntry {
            count: 65,
            role: TensorRole::FfnDown,
            storage,
        },
    ]
}

#[test]
fn pinned_qwen38_q4_k_m_and_q5_k_m_profiles_are_admitted_without_substitution() {
    for primary in [GgmlType::Q4_K, GgmlType::Q5_K] {
        let receipt = validate_model_free_profile(&mixed_k_profile(primary))
            .unwrap_or_else(|error| panic!("{primary:?} profile rejected: {error:#}"));
        assert_eq!(receipt.required_tensor_count, 866);
        assert_eq!(receipt.storage_counts.get("F32"), Some(&360));
        assert_eq!(
            receipt.storage_counts.get(&format!("{primary:?}")),
            Some(&439)
        );
        assert_eq!(receipt.storage_counts.get("Q6_K"), Some(&67));
    }
}

#[test]
fn pinned_qwen38_q6_k_and_q8_0_profiles_are_admitted_without_substitution() {
    for kind in [GgmlType::Q6_K, GgmlType::Q8_0] {
        let receipt = validate_model_free_profile(&uniform_profile(TensorStorage::Parsed(kind)))
            .unwrap_or_else(|error| panic!("{kind:?} profile rejected: {error:#}"));
        assert_eq!(receipt.required_tensor_count, 866);
        assert_eq!(receipt.storage_counts.get("F32"), Some(&360));
        assert_eq!(receipt.storage_counts.get(&format!("{kind:?}")), Some(&506));
    }
}

#[test]
fn pinned_qwen38_dense_kernel_regimes_are_available_without_a_device() {
    for kind in [
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::Q8_0,
    ] {
        let info = TensorInfo {
            name: format!("profile.{kind:?}.weight"),
            shape: vec![256, 256],
            ggml_type: kind,
            offset: 0,
            byte_len: checked_tensor_bytes_for_profile(kind, 256, 256),
        };
        ensure_dense_capability(&info.name, &info)
            .unwrap_or_else(|error| panic!("{kind:?} dense regimes rejected: {error:#}"));
    }
}

#[test]
fn production_matrix_preflight_retains_non_power_and_prompt_boundaries() {
    let widths = REQUIRED_MATRIX_WIDTHS.map(|(width, _)| width);
    assert_eq!(widths, [1, 2, 3, 4, 8, 9, 16, 17]);
    assert_eq!(
        REQUIRED_MATRIX_WIDTHS[2].1,
        GgmlWorkloadClass::ContinuousWidth
    );
    assert_eq!(REQUIRED_MATRIX_WIDTHS[5].1, GgmlWorkloadClass::Prompt);
    assert_eq!(REQUIRED_MATRIX_WIDTHS[7].1, GgmlWorkloadClass::Prompt);
}

#[test]
fn shared_expert_gate_preflight_accepts_only_exact_rank_one_storage() {
    let cols = 256usize;
    let exact_file = row_projection_fixture(&[cols as u64]);
    let exact = GgufFile::open(exact_file.path()).expect("open exact row vector");
    let mut receipt = Qwen35GgufPreflightReceipt::default();
    require_row_projection(
        &exact,
        "blk.0.ffn_gate_inp_shexp.weight",
        cols,
        &mut receipt,
    )
    .expect("exact rank-one shared gate must be admitted");
    assert_eq!(receipt.matrix_tensor_count, 1);
    assert_eq!(receipt.matrix_bytes, (cols * 4) as u64);

    let squeezed_file = row_projection_fixture(&[cols as u64, 1]);
    let squeezed = GgufFile::open(squeezed_file.path()).expect("open rank-two row");
    let error = require_row_projection(
        &squeezed,
        "blk.0.ffn_gate_inp_shexp.weight",
        cols,
        &mut Qwen35GgufPreflightReceipt::default(),
    )
    .expect_err("rank-two storage must not acquire implicit squeeze semantics");
    assert!(error.to_string().contains("shape [1, 256]"));
}

/// Header-only gate for an exact pinned artifact. It parses the tensor
/// directory and runs the allocation-free preflight; it never creates a
/// Metal device or reads model tensor payloads.
#[test]
#[ignore = "requires HF2Q_TEST_QWEN38_GGUF"]
fn pinned_qwen38_real_header_passes_before_model_or_device_allocation() {
    let path = std::env::var_os("HF2Q_TEST_QWEN38_GGUF")
        .map(std::path::PathBuf::from)
        .expect("set HF2Q_TEST_QWEN38_GGUF to a pinned Qwen3.8 GGUF");
    let gguf = GgufFile::open(&path).expect("parse pinned Qwen3.8 GGUF header");
    assert_eq!(gguf.tensor_count(), 866, "unexpected artifact tensor count");
    assert!(
        matches!(
            gguf.metadata_u32("general.file_type"),
            Some(7 | 15 | 17 | 18 | 32)
        ),
        "header gate requires a pinned BF16, Q8_0, Q4_K_M, Q5_K_M, or Q6_K artifact"
    );
    let cfg = Qwen35Config::from_gguf(&gguf).expect("parse Qwen3.8 config metadata");
    preflight_dense_qwen35_gguf(&gguf, &cfg)
        .expect("pinned Qwen3.8 header must pass native execution preflight");
}

fn checked_tensor_bytes_for_profile(kind: GgmlType, rows: usize, cols: usize) -> usize {
    rows * (cols / kind.block_values() as usize) * kind.block_bytes() as usize
}

#[test]
fn pinned_qwen38_bf16_profile_is_admitted_without_substitution() {
    let receipt =
        validate_model_free_profile(&uniform_profile(TensorStorage::Parsed(GgmlType::BF16)))
            .expect("BF16 must be native across every Qwen tensor role");
    assert_eq!(receipt.required_tensor_count, 866);
    assert_eq!(receipt.storage_counts.get("F32"), Some(&360));
    assert_eq!(receipt.storage_counts.get("BF16"), Some(&506));
}

#[test]
fn matrix_rejects_known_non_native_role_combinations() {
    for (role, kind) in [
        (TensorRole::F32State, GgmlType::Q8_0),
        (TensorRole::Embedding, GgmlType::Q4_0),
        (TensorRole::DenseProjection, GgmlType::I16),
        (TensorRole::FfnGateUp, GgmlType::Q2_K),
    ] {
        let error = admit_storage_for_role("canary.weight", role, TensorStorage::Parsed(kind))
            .expect_err("unsupported route must fail closed");
        let message = format!("{error:#}");
        assert!(message.contains("canary.weight"));
        assert!(message.contains(role.label()));
    }
}

#[test]
fn shared_mtp_preflight_rejects_a_dedicated_head_without_creating_a_device() {
    let error = admit_mtp_tensor_presence(false, false, true)
        .expect_err("shared MTP must reject a dedicated head-only artifact");
    assert!(error.to_string().contains("shared MTP"));
    admit_mtp_tensor_presence(false, false, false).expect("valid shared MTP tensor presence");
    admit_mtp_tensor_presence(true, true, true).expect("valid dedicated MTP tensor presence");
}
