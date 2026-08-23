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
    let scheduler = include_str!("../../../serve/api/engine.rs");
    assert!(scheduler.contains("const QWEN35_SLOT_PREFILL_CHUNK_TOKENS: u32 = 2_048;"));
    assert!(scheduler.contains("const QWEN35_SINGLE_SLOT_PREFILL_CHUNK_CEILING: u32 = 4_096;"));
    assert_eq!(REQUIRED_EXPERT_SCHEDULER_WIDTHS, [2_048, 4_096]);
}

#[test]
fn qwen_expert_down_preflight_matches_flattened_runtime_rows() {
    const EXPERTS: u32 = 256;
    const TOP_K: u32 = 8;
    const HIDDEN: u32 = 2_048;
    const EXPERT_WIDTH: u32 = 512;
    const Q6_K_BLOCK_VALUES: u64 = 256;
    const Q6_K_BLOCK_BYTES: u64 = 210;
    let expert_stride =
        u64::from(HIDDEN) * u64::from(EXPERT_WIDTH) / Q6_K_BLOCK_VALUES * Q6_K_BLOCK_BYTES;
    let routing = GgmlRoutingPolicy::default();

    let decode = expert_capability_request(
        GgmlType::Q6_K,
        HIDDEN,
        EXPERT_WIDTH,
        TOP_K,
        EXPERTS,
        expert_stride,
        1,
        ExpertExecution::FlattenedRoutedRows,
        routing,
    )
    .expect("decode down request");
    let GgmlInvocation::ExpertPooled {
        shape,
        input_layout,
    } = decode.invocation
    else {
        panic!("down request must use the pooled expert entry point")
    };
    assert_eq!(shape.n_tokens, TOP_K);
    assert_eq!(shape.top_k, 1);
    assert_eq!(input_layout, GgmlExpertInputLayout::SharedPerToken);
    assert_eq!(decode.workload, GgmlWorkloadClass::ContinuousWidth);
    let decode_capability = mlx_native::ggml_capability(decode);
    assert!(
        decode_capability.executable,
        "flattened decode rows must retain the production matvec route: {}",
        decode_capability.diagnostic
    );
    assert!(matches!(
        decode_capability.route,
        Some(mlx_native::GgmlKernelRoute::ExpertMv | mlx_native::GgmlKernelRoute::ExpertMvNr2)
    ));

    let last_mv = expert_capability_request(
        GgmlType::Q6_K,
        HIDDEN,
        EXPERT_WIDTH,
        TOP_K,
        EXPERTS,
        expert_stride,
        4,
        ExpertExecution::FlattenedRoutedRows,
        routing,
    )
    .expect("last matvec down request");
    let GgmlInvocation::ExpertPooled { shape, .. } = last_mv.invocation else {
        panic!("down request must use the pooled expert entry point")
    };
    assert_eq!(shape.n_tokens, routing.expert_mm_threshold);
    assert_eq!(shape.top_k, 1);
    let last_mv_capability = mlx_native::ggml_capability(last_mv);
    assert!(matches!(
        last_mv_capability.route,
        Some(mlx_native::GgmlKernelRoute::ExpertMv | mlx_native::GgmlKernelRoute::ExpertMvNr2)
    ));

    let first_mm = expert_capability_request(
        GgmlType::Q6_K,
        HIDDEN,
        EXPERT_WIDTH,
        TOP_K,
        EXPERTS,
        expert_stride,
        5,
        ExpertExecution::FlattenedRoutedRows,
        routing,
    )
    .expect("first mm_id down request");
    let GgmlInvocation::ExpertPooled { shape, .. } = first_mm.invocation else {
        panic!("down request must use the pooled expert entry point")
    };
    assert_eq!(shape.n_tokens, 40);
    assert_eq!(shape.top_k, 1);
    assert_eq!(first_mm.workload, GgmlWorkloadClass::Prompt);
    let prompt_capability = mlx_native::ggml_capability(first_mm);
    assert!(
        prompt_capability.executable,
        "flattened prompt rows must use the supported mm_id route: {}",
        prompt_capability.diagnostic
    );
    assert!(matches!(
        prompt_capability.route,
        Some(
            mlx_native::GgmlKernelRoute::ExpertPooledMmSimdgroup
                | mlx_native::GgmlKernelRoute::ExpertPooledMmDeviceSelected
        )
    ));

    let info = TensorInfo {
        name: "blk.0.ffn_down_exps.weight".into(),
        shape: vec![EXPERTS as usize, HIDDEN as usize, EXPERT_WIDTH as usize],
        ggml_type: GgmlType::Q6_K,
        offset: 0,
        byte_len: usize::try_from(expert_stride * u64::from(EXPERTS)).unwrap(),
    };
    ensure_expert_capability(
        &info.name,
        &info,
        HIDDEN as usize,
        EXPERT_WIDTH as usize,
        TOP_K,
        EXPERTS,
        ExpertExecution::FlattenedRoutedRows,
    )
    .expect("the complete production down-width matrix must be admitted");
}

#[test]
fn qwen_expert_gate_up_proves_q5_k_exact_mv_to_mm_boundary() {
    let routing = GgmlRoutingPolicy::default();
    assert_eq!(routing.expert_mm_threshold, 32);
    let expert_stride = 720_896;
    for (source_tokens, expected_mm) in [(32, false), (33, true)] {
        let request = expert_capability_request(
            GgmlType::Q5_K,
            512,
            2_048,
            8,
            256,
            expert_stride,
            source_tokens,
            ExpertExecution::SharedPerSourceToken,
            routing,
        )
        .expect("Q5_K gate/up request");
        let capability = mlx_native::ggml_capability(request);
        assert!(capability.executable, "{}", capability.diagnostic);
        let mm_route = matches!(
            capability.route,
            Some(
                mlx_native::GgmlKernelRoute::ExpertPooledMmSimdgroup
                    | mlx_native::GgmlKernelRoute::ExpertPooledMmDeviceSelected
            )
        );
        assert_eq!(
            mm_route, expected_mm,
            "source M={source_tokens} must respect strict > threshold routing"
        );
    }
}

#[test]
fn qwen_expert_widths_follow_active_policy_and_scheduler_maxima() {
    let default = GgmlRoutingPolicy::default();
    let gate_up = required_expert_source_widths(ExpertExecution::SharedPerSourceToken, 8, default)
        .expect("default gate/up widths");
    for width in [31, 32, 33, 2_048, 4_096] {
        assert!(gate_up.contains(&width), "missing gate/up M={width}");
    }
    let down = required_expert_source_widths(ExpertExecution::FlattenedRoutedRows, 8, default)
        .expect("default down widths");
    for width in [4, 5, 2_048, 4_096] {
        assert!(down.contains(&width), "missing down source M={width}");
    }

    let overridden = GgmlRoutingPolicy {
        expert_mm_threshold: 77,
        ..default
    };
    let gate_up =
        required_expert_source_widths(ExpertExecution::SharedPerSourceToken, 8, overridden)
            .expect("overridden gate/up widths");
    for width in [76, 77, 78] {
        assert!(gate_up.contains(&width), "missing overridden M={width}");
    }
    assert!(!gate_up.contains(&33));
    let down = required_expert_source_widths(ExpertExecution::FlattenedRoutedRows, 8, overridden)
        .expect("overridden down widths");
    assert!(down.contains(&9));
    assert!(down.contains(&10));

    let force_mv = GgmlRoutingPolicy {
        expert_mm_threshold: u32::MAX,
        ..default
    };
    for execution in [
        ExpertExecution::SharedPerSourceToken,
        ExpertExecution::FlattenedRoutedRows,
    ] {
        let widths = required_expert_source_widths(execution, 8, force_mv)
            .expect("force-MV widths must remain representable");
        assert_eq!(widths.last(), Some(&4_096));
        assert!(widths.iter().all(|width| *width <= 4_096));
        for source_tokens in widths {
            let request = expert_capability_request(
                GgmlType::Q5_K,
                512,
                2_048,
                8,
                256,
                720_896,
                source_tokens,
                execution,
                force_mv,
            )
            .expect("force-MV request geometry");
            let capability = mlx_native::ggml_capability(request);
            assert!(
                capability.executable,
                "force-MV source M={source_tokens} rejected: {}",
                capability.diagnostic
            );
            assert!(matches!(
                capability.route,
                Some(
                    mlx_native::GgmlKernelRoute::ExpertMv
                        | mlx_native::GgmlKernelRoute::ExpertMvNr2
                )
            ));
        }
    }
}

#[test]
fn qwen_expert_down_preflight_keeps_slotted_and_overflow_canaries_closed() {
    let mut slotted = expert_capability_request(
        GgmlType::Q6_K,
        2_048,
        512,
        8,
        256,
        860_160,
        1,
        ExpertExecution::FlattenedRoutedRows,
        GgmlRoutingPolicy::default(),
    )
    .expect("decode down request");
    let GgmlInvocation::ExpertPooled { input_layout, .. } = &mut slotted.invocation else {
        panic!("down request must use the pooled expert entry point")
    };
    *input_layout = GgmlExpertInputLayout::Slotted;
    let rejected = mlx_native::ggml_capability(slotted);
    assert!(!rejected.executable);
    assert!(
        rejected
            .diagnostic
            .contains("paired/slotted pooled expert entry point requires the mm_id route"),
        "unexpected slotted rejection: {}",
        rejected.diagnostic
    );

    let error = expert_capability_request(
        GgmlType::Q6_K,
        2_048,
        512,
        2,
        256,
        860_160,
        u32::MAX,
        ExpertExecution::FlattenedRoutedRows,
        GgmlRoutingPolicy::default(),
    )
    .expect_err("flattened routed-row overflow must fail before capability evaluation");
    assert!(error.to_string().contains("row count overflows u32"));
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
