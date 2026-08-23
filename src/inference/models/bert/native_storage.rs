//! Shared native-GGUF storage contract for encoder-only embedding models.
//!
//! Matrix-shaped tensors remain in their exact stored representation.  Only
//! explicitly declared vector state (normalization parameters and biases) is
//! expanded to F32.  Capability and shape validation are pure header work and
//! therefore complete before the GGUF payload is mapped or a model buffer is
//! allocated.

use anyhow::{anyhow, ensure, Result};
use mlx_native::gguf::{GgufFile, TensorInfo};
#[cfg(test)]
use mlx_native::MlxBuffer;
use mlx_native::{
    ggml_capability, GgmlCapabilityRequest, GgmlInvocation, GgmlRoutingPolicy, GgmlType,
    GgmlWorkloadClass, GGML_CAPABILITY_SCHEMA_VERSION,
};

use crate::serve::forward_mlx_shared::MlxQWeight;
#[cfg(test)]
use crate::serve::gpu::QuantWeightInfo;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeStorageStats {
    /// Logical tensor bytes owned by this model instance.
    pub resident_bytes: u64,
    /// Exact GGUF payload bytes retained through read-only file mappings.
    pub file_backed_bytes: u64,
    /// Expanded F32 vector-state bytes allocated anonymously.
    pub anonymous_state_bytes: u64,
    /// Distinct mapped Metal resources referenced by this model instance.
    pub mapped_segment_count: usize,
}

#[derive(Debug, Clone)]
pub struct MatrixPlan {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub ggml_type: GgmlType,
    pub byte_len: usize,
}

#[derive(Debug, Clone)]
pub struct StatePlan {
    pub name: String,
    pub elements: usize,
}

fn exact_shape(info: &TensorInfo, name: &str, expected: &[usize]) -> Result<()> {
    ensure!(
        info.shape == expected,
        "embedding model tensor '{name}' shape {:?} != expected {expected:?}",
        info.shape
    );
    Ok(())
}

fn checked_u32(value: usize, label: &str, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| anyhow!("embedding model tensor '{name}' {label} exceeds u32"))
}

fn require_capability(
    name: &str,
    ggml_type: GgmlType,
    invocation: GgmlInvocation,
    workload: GgmlWorkloadClass,
) -> Result<()> {
    let capability = ggml_capability(GgmlCapabilityRequest {
        schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
        invocation,
        ggml_type,
        workload,
        routing: GgmlRoutingPolicy::default(),
    });
    ensure!(
        capability.executable,
        "embedding model tensor '{name}' stored as {ggml_type:?} has no native {workload:?} route: {}",
        capability.diagnostic
    );
    Ok(())
}

/// Validate one embedding table from GGUF metadata only.
pub fn preflight_embedding(
    gguf: &GgufFile,
    name: impl Into<String>,
    rows: usize,
    cols: usize,
    max_service_tokens: usize,
) -> Result<MatrixPlan> {
    let name = name.into();
    let info = gguf
        .tensor_info(&name)
        .ok_or_else(|| anyhow!("embedding model missing '{name}'"))?;
    exact_shape(info, &name, &[rows, cols])?;
    let rows_u32 = checked_u32(rows, "row count", &name)?;
    let cols_u32 = checked_u32(cols, "column count", &name)?;
    let max_tokens = checked_u32(max_service_tokens, "maximum service token count", &name)?;
    ensure!(
        max_tokens > 0,
        "embedding model tensor '{name}' maximum service token count must be positive"
    );
    ensure!(
        native_embedding_codec_supported(info.ggml_type),
        "embedding model tensor '{name}' stored as {:?} has no direct native gather",
        info.ggml_type
    );
    // mlx-native 0.12.1's embedding capability route is codec/shape based and
    // token-count invariant once n_tokens is nonzero; the regression below
    // pins that contract across every dense-matmul routing boundary. Check
    // both service-envelope ends here for dimension conversion and overflow.
    for n_tokens in [1, max_tokens] {
        require_capability(
            &name,
            info.ggml_type,
            GgmlInvocation::EmbeddingGather {
                n_tokens,
                vocab_size: rows_u32,
                embed_dim: cols_u32,
            },
            GgmlWorkloadClass::Embedding,
        )?;
    }
    Ok(MatrixPlan {
        name,
        rows,
        cols,
        ggml_type: info.ggml_type,
        byte_len: info.byte_len,
    })
}

pub fn native_embedding_codec_supported(ggml_type: GgmlType) -> bool {
    matches!(
        ggml_type,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q2_K
            | GgmlType::Q4_0
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
    )
}

/// File-level catalog filter corresponding to the tensor codecs above.
///
/// This intentionally accepts hf2q's GGUF file-type enum rather than casting
/// it to mlx-native's similarly numbered execution enum.  A selectable file
/// still goes through tensor-by-tensor `preflight_*` validation before any
/// payload mapping occurs.
pub const fn native_embedding_file_type_supported(
    file_type: crate::quantize::ggml_quants::GgufFtype,
) -> bool {
    use crate::quantize::ggml_quants::GgufFtype;
    matches!(
        file_type,
        GgufFtype::AllF32
            | GgufFtype::MostlyF16
            | GgufFtype::BF16
            | GgufFtype::MostlyQ2_K
            | GgufFtype::MostlyQ2_K_S
            | GgufFtype::MostlyQ4_0
            | GgufFtype::MostlyQ4_K_S
            | GgufFtype::MostlyQ4_K_M
            | GgufFtype::MostlyQ5_K_S
            | GgufFtype::MostlyQ5_K_M
            | GgufFtype::MostlyQ6_K
            | GgufFtype::MostlyQ8_0
    )
}

/// Validate one linear matrix from GGUF metadata only.
///
/// Encoder service must accept short requests, continuous batching, and
/// prompt-width execution.  Checking all three shapes prevents a type that
/// happens to have an M=1 kernel from entering a model that cannot serve its
/// normal width-N forward pass.
pub fn preflight_linear(
    gguf: &GgufFile,
    name: impl Into<String>,
    rows: usize,
    cols: usize,
    max_service_rows: usize,
) -> Result<MatrixPlan> {
    let name = name.into();
    let info = gguf
        .tensor_info(&name)
        .ok_or_else(|| anyhow!("embedding model missing '{name}'"))?;
    exact_shape(info, &name, &[rows, cols])?;
    let n = checked_u32(rows, "row count", &name)?;
    let k = checked_u32(cols, "column count", &name)?;
    for (m, workload) in linear_service_regimes(max_service_rows, &name)? {
        require_capability(
            &name,
            info.ggml_type,
            GgmlInvocation::DenseAuto { m, n, k },
            workload,
        )?;
    }
    Ok(MatrixPlan {
        name,
        rows,
        cols,
        ggml_type: info.ggml_type,
        byte_len: info.byte_len,
    })
}

/// Distinct row-count regimes selected by `quantized_matmul_ggml` for an
/// encoder request. Every continuous-width value is represented because
/// codecs may specialize individual M values. Above the routing threshold the
/// planner has one prompt regime, so its first row count and the model's
/// maximum admitted row count pin both the boundary and the full service
/// envelope without pretending one representative prompt is sufficient.
fn linear_service_regimes(
    max_service_rows: usize,
    name: &str,
) -> Result<Vec<(u32, GgmlWorkloadClass)>> {
    let max_rows = checked_u32(max_service_rows, "maximum service row count", name)?;
    ensure!(
        max_rows > 0,
        "embedding model tensor '{name}' maximum service row count must be positive"
    );
    let threshold = mlx_native::ops::quantized_matmul_ggml::MM_ROUTING_THRESHOLD;
    let mut regimes = Vec::with_capacity(threshold as usize + 2);
    regimes.push((1, GgmlWorkloadClass::DecodeSingle));
    for m in 2..=max_rows.min(threshold) {
        regimes.push((m, GgmlWorkloadClass::ContinuousWidth));
    }
    if max_rows > threshold {
        regimes.push((threshold + 1, GgmlWorkloadClass::Prompt));
        if max_rows > threshold + 1 {
            regimes.push((max_rows, GgmlWorkloadClass::Prompt));
        }
    }
    Ok(regimes)
}

/// Validate one small elementwise tensor from GGUF metadata only.
pub fn preflight_state(
    gguf: &GgufFile,
    name: impl Into<String>,
    elements: usize,
) -> Result<StatePlan> {
    let name = name.into();
    let info = gguf
        .tensor_info(&name)
        .ok_or_else(|| anyhow!("embedding model missing '{name}'"))?;
    exact_shape(info, &name, &[elements])?;
    ensure!(
        matches!(
            info.ggml_type,
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
        ),
        "embedding model state tensor '{name}' stored as {:?} must be F32, F16, or BF16",
        info.ggml_type
    );
    Ok(StatePlan { name, elements })
}

pub fn mapped_qweight(
    plan: &MatrixPlan,
    mapped: &mlx_native::gguf::GgufMappedTensorSet<'_>,
    info: &TensorInfo,
) -> Result<MlxQWeight> {
    ensure!(
        info.name == plan.name
            && info.shape.as_slice() == [plan.rows, plan.cols]
            && info.ggml_type == plan.ggml_type
            && info.byte_len == plan.byte_len,
        "native matrix '{}' preflight identity changed before mapping",
        plan.name,
    );
    MlxQWeight::from_mapped_gguf_tensor(mapped, info)
}

#[cfg(test)]
pub fn f32_qweight_for_test(buffer: MlxBuffer, rows: usize, cols: usize) -> MlxQWeight {
    MlxQWeight {
        buffer,
        info: QuantWeightInfo {
            ggml_dtype: GgmlType::F32,
            rows,
            cols,
        },
        affine: None,
        decode_record_q6k_m1: std::sync::OnceLock::new(),
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    use crate::inference::models::bert::config::{BertConfig, PoolingType};

    pub(crate) struct TestTensor {
        pub(crate) name: String,
        pub(crate) outer_shape: Vec<usize>,
        pub(crate) ggml_type: u32,
        pub(crate) bytes: Vec<u8>,
    }

    pub(crate) fn q4_rows(rows: usize, cols: usize, code: u8) -> Vec<u8> {
        assert_eq!(cols % 32, 0);
        let mut bytes = Vec::with_capacity(rows * cols / 32 * 18);
        for _ in 0..rows * cols / 32 {
            bytes.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
            bytes.extend(std::iter::repeat(code).take(16));
        }
        bytes
    }

    pub(crate) fn f32_state(elements: usize, value: f32) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(elements * 4);
        for _ in 0..elements {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    pub(crate) fn write_fixture(path: &Path, tensors: &[TestTensor], include_payload: bool) {
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend_from_slice(&3u32.to_le_bytes());
        header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());
        let mut offsets = Vec::with_capacity(tensors.len());
        let mut cursor = 0usize;
        for tensor in tensors {
            cursor = (cursor + 31) / 32 * 32;
            offsets.push(cursor);
            cursor += tensor.bytes.len();
            header.extend_from_slice(&(tensor.name.len() as u64).to_le_bytes());
            header.extend_from_slice(tensor.name.as_bytes());
            header.extend_from_slice(&(tensor.outer_shape.len() as u32).to_le_bytes());
            for dim in tensor.outer_shape.iter().rev() {
                header.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
            header.extend_from_slice(&tensor.ggml_type.to_le_bytes());
            header.extend_from_slice(&(offsets.last().copied().unwrap() as u64).to_le_bytes());
        }
        while header.len() % 32 != 0 {
            header.push(0);
        }
        if include_payload {
            let mut payload = vec![0u8; cursor];
            for (tensor, offset) in tensors.iter().zip(offsets) {
                payload[offset..offset + tensor.bytes.len()].copy_from_slice(&tensor.bytes);
            }
            header.extend_from_slice(&payload);
        }
        let mut file = File::create(path).expect("create native embedding fixture");
        file.write_all(&header)
            .expect("write native embedding fixture");
        file.flush().expect("flush native embedding fixture");
    }

    pub(crate) fn bert_cfg(vocab_size: usize) -> BertConfig {
        BertConfig {
            hidden_size: 32,
            num_attention_heads: 1,
            num_hidden_layers: 0,
            intermediate_size: 64,
            max_position_embeddings: 32,
            vocab_size,
            type_vocab_size: 2,
            layer_norm_eps: 1e-5,
            hidden_act: "gelu".into(),
            pooling_type: PoolingType::Mean,
            causal_attention: false,
        }
    }

    pub(crate) fn bert_tensors(
        vocab_size: usize,
        code: u8,
        embedding_type: u32,
    ) -> Vec<TestTensor> {
        let embedding_bytes = if embedding_type == 2 {
            q4_rows(vocab_size, 32, code)
        } else {
            vec![0u8; vocab_size * 24]
        };
        vec![
            TestTensor {
                name: "token_embd.weight".into(),
                outer_shape: vec![vocab_size, 32],
                ggml_type: embedding_type,
                bytes: embedding_bytes,
            },
            TestTensor {
                name: "position_embd.weight".into(),
                outer_shape: vec![32, 32],
                ggml_type: 2,
                bytes: q4_rows(32, 32, 0x88),
            },
            TestTensor {
                name: "token_embd_norm.weight".into(),
                outer_shape: vec![32],
                ggml_type: 0,
                bytes: f32_state(32, 1.0),
            },
            TestTensor {
                name: "token_embd_norm.bias".into(),
                outer_shape: vec![32],
                ggml_type: 0,
                bytes: f32_state(32, 0.0),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::inference::models::bert::config::PoolingType;
    use crate::inference::models::bert::native_gpu::{
        bert_embed_gather_native_gpu, register_native_embedding_shaders,
    };
    use crate::inference::models::bert::weights::LoadedBertWeights;
    use crate::inference::models::nomic_bert::{LoadedNomicBertWeights, NomicBertConfig};
    use mlx_native::{DType, KernelRegistry, MlxDevice};

    fn gather_first(weights: &LoadedBertWeights, expected: f32) {
        let device = MlxDevice::new().expect("device");
        let mut ids = device.alloc_buffer(4, DType::U32, vec![1]).expect("ids");
        ids.as_mut_slice::<u32>().expect("ids slice")[0] = 0;
        let mut registry = KernelRegistry::new();
        register_native_embedding_shaders(&mut registry);
        let mut encoder = device.command_encoder().expect("encoder");
        let output = bert_embed_gather_native_gpu(
            &mut encoder,
            &mut registry,
            &device,
            weights.token_embd_weight().expect("token table"),
            &ids,
            4,
            32,
            1,
        )
        .expect("native gather");
        encoder.commit_and_wait().expect("complete native gather");
        assert!(output
            .as_slice::<f32>()
            .expect("output")
            .iter()
            .all(|value| value.to_bits() == expected.to_bits()));
    }

    #[test]
    fn bert_q4_storage_is_native_and_a_b_a_reload_isolated() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let directory = tempfile::tempdir().expect("tempdir");
        let a = directory.path().join("a.gguf");
        let b = directory.path().join("b.gguf");
        write_fixture(&a, &bert_tensors(4, 0x99, 2), true);
        write_fixture(&b, &bert_tensors(4, 0x77, 2), true);
        let cfg = bert_cfg(4);

        let a_first = LoadedBertWeights::load_from_path(&a, &cfg).expect("load A");
        assert_eq!(
            a_first.token_embd_weight().unwrap().info.ggml_dtype,
            GgmlType::Q4_0
        );
        assert_eq!(
            a_first.token_embd_weight().unwrap().buffer.dtype(),
            DType::U8
        );
        assert!(a_first.token_embd_weight().unwrap().buffer.is_file_backed());
        assert!(!a_first.embed_norm_weight().unwrap().is_file_backed());
        let a_stats = a_first.storage_stats();
        assert!(a_stats.file_backed_bytes > a_stats.anonymous_state_bytes);
        assert!(a_stats.mapped_segment_count > 0);
        gather_first(&a_first, 1.0);
        drop(a_first);

        let b_loaded = LoadedBertWeights::load_from_path(&b, &cfg).expect("load B");
        assert_eq!(b_loaded.storage_stats(), a_stats);
        gather_first(&b_loaded, -1.0);
        drop(b_loaded);

        let a_again = LoadedBertWeights::load_from_path(&a, &cfg).expect("reload A");
        assert_eq!(
            a_again.storage_stats(),
            a_stats,
            "accounting must not accumulate across swaps"
        );
        gather_first(&a_again, 1.0);
    }

    #[test]
    fn unsupported_embedding_fails_before_truncated_payload_mapping() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("unsupported-truncated.gguf");
        write_fixture(&path, &bert_tensors(4, 0x88, 7), false);
        let gguf = GgufFile::open(&path).expect("header-only GGUF opens");
        let error = LoadedBertWeights::load(&gguf, &bert_cfg(4), MlxDevice::new().expect("device"))
            .expect_err("Q5_1 embedding has no native gather");
        let message = error.to_string();
        assert!(
            message.contains("Q5_1") && message.contains("native"),
            "{message}"
        );
        assert!(
            !message.contains("map") && !message.contains("payload"),
            "{message}"
        );
    }

    #[test]
    fn q5_k_linear_preflight_covers_decode_width_and_prompt_regimes() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("q5-k-linear.gguf");
        write_fixture(
            &path,
            &[TestTensor {
                name: "blk.0.ffn_up.weight".into(),
                outer_shape: vec![256, 256],
                ggml_type: 13,
                bytes: vec![0u8; 256 * 176],
            }],
            true,
        );
        let gguf = GgufFile::open(&path).expect("open Q5_K fixture");
        let plan = preflight_linear(&gguf, "blk.0.ffn_up.weight", 256, 256, 512)
            .expect("Q5_K must execute across embedding service widths");
        assert_eq!(plan.ggml_type, GgmlType::Q5_K);
        assert_eq!(plan.byte_len, 256 * 176);
    }

    #[test]
    fn linear_preflight_enumerates_every_route_boundary_and_model_maximum() {
        let threshold = mlx_native::ops::quantized_matmul_ggml::MM_ROUTING_THRESHOLD;
        assert_eq!(
            threshold, 8,
            "update the encoder preflight if routing changes"
        );
        assert_eq!(
            linear_service_regimes(512, "test.weight").expect("service regimes"),
            vec![
                (1, GgmlWorkloadClass::DecodeSingle),
                (2, GgmlWorkloadClass::ContinuousWidth),
                (3, GgmlWorkloadClass::ContinuousWidth),
                (4, GgmlWorkloadClass::ContinuousWidth),
                (5, GgmlWorkloadClass::ContinuousWidth),
                (6, GgmlWorkloadClass::ContinuousWidth),
                (7, GgmlWorkloadClass::ContinuousWidth),
                (8, GgmlWorkloadClass::ContinuousWidth),
                (9, GgmlWorkloadClass::Prompt),
                (512, GgmlWorkloadClass::Prompt),
            ]
        );
        assert_eq!(
            linear_service_regimes(8, "short.weight").expect("short service regimes"),
            vec![
                (1, GgmlWorkloadClass::DecodeSingle),
                (2, GgmlWorkloadClass::ContinuousWidth),
                (3, GgmlWorkloadClass::ContinuousWidth),
                (4, GgmlWorkloadClass::ContinuousWidth),
                (5, GgmlWorkloadClass::ContinuousWidth),
                (6, GgmlWorkloadClass::ContinuousWidth),
                (7, GgmlWorkloadClass::ContinuousWidth),
                (8, GgmlWorkloadClass::ContinuousWidth),
            ]
        );
    }

    #[test]
    fn native_embedding_codec_matrix_includes_q5_k_and_q6_k() {
        for ggml_type in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q2_K,
            GgmlType::Q4_0,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
        ] {
            assert!(
                native_embedding_codec_supported(ggml_type),
                "{ggml_type:?} must use a direct native gather"
            );
        }
        for ggml_type in [
            GgmlType::Q3_K,
            GgmlType::Q5_1,
            GgmlType::IQ4_NL,
            GgmlType::IQ4_XS,
            GgmlType::I16,
            GgmlType::I32,
        ] {
            assert!(
                !native_embedding_codec_supported(ggml_type),
                "{ggml_type:?} must fail before allocation"
            );
        }

        use crate::quantize::ggml_quants::GgufFtype;
        for file_type in [
            GgufFtype::AllF32,
            GgufFtype::MostlyF16,
            GgufFtype::BF16,
            GgufFtype::MostlyQ2_K,
            GgufFtype::MostlyQ2_K_S,
            GgufFtype::MostlyQ4_0,
            GgufFtype::MostlyQ4_K_S,
            GgufFtype::MostlyQ4_K_M,
            GgufFtype::MostlyQ5_K_S,
            GgufFtype::MostlyQ5_K_M,
            GgufFtype::MostlyQ6_K,
            GgufFtype::MostlyQ8_0,
        ] {
            assert!(
                native_embedding_file_type_supported(file_type),
                "{file_type:?} must remain selectable for native embedding execution"
            );
        }
        for file_type in [
            GgufFtype::MostlyQ3_K_M,
            GgufFtype::MostlyQ4_1,
            GgufFtype::MostlyQ5_1,
            GgufFtype::MostlyIQ4_NL,
        ] {
            assert!(
                !native_embedding_file_type_supported(file_type),
                "{file_type:?} must remain non-selectable without a native gather"
            );
        }
    }

    #[test]
    fn embedding_capability_route_is_token_count_invariant() {
        for ggml_type in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q2_K,
            GgmlType::Q4_0,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
        ] {
            let capability = |n_tokens| {
                ggml_capability(GgmlCapabilityRequest {
                    schema_version: GGML_CAPABILITY_SCHEMA_VERSION,
                    invocation: GgmlInvocation::EmbeddingGather {
                        n_tokens,
                        vocab_size: 32,
                        embed_dim: 256,
                    },
                    ggml_type,
                    workload: GgmlWorkloadClass::Embedding,
                    routing: GgmlRoutingPolicy::default(),
                })
            };
            let expected = capability(1);
            assert!(expected.executable, "{ggml_type:?} n_tokens=1");
            for n_tokens in [2, 3, 4, 5, 6, 7, 8, 9, 512] {
                let observed = capability(n_tokens);
                assert!(observed.executable, "{ggml_type:?} n_tokens={n_tokens}");
                assert_eq!(
                    observed.route, expected.route,
                    "{ggml_type:?} embedding routing changed at n_tokens={n_tokens}; preflight must enumerate the new regimes"
                );
            }
        }
    }

    #[test]
    fn nomic_q4_loader_retains_native_table_without_shadows() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("nomic.gguf");
        let tensors = vec![
            TestTensor {
                name: "token_embd.weight".into(),
                outer_shape: vec![4, 64],
                ggml_type: 2,
                bytes: q4_rows(4, 64, 0x99),
            },
            TestTensor {
                name: "token_embd_norm.weight".into(),
                outer_shape: vec![64],
                ggml_type: 0,
                bytes: f32_state(64, 1.0),
            },
            TestTensor {
                name: "token_embd_norm.bias".into(),
                outer_shape: vec![64],
                ggml_type: 0,
                bytes: f32_state(64, 0.0),
            },
        ];
        write_fixture(&path, &tensors, true);
        let cfg = NomicBertConfig {
            hidden_size: 64,
            num_attention_heads: 1,
            num_hidden_layers: 0,
            intermediate_size: 128,
            max_position_embeddings: 32,
            vocab_size: 4,
            type_vocab_size: 2,
            layer_norm_eps: 1e-5,
            pooling_type: PoolingType::Mean,
            rope_freq_base: 1000.0,
            causal_attention: false,
        };
        let gguf = GgufFile::open(&path).expect("open nomic fixture");
        let weights = LoadedNomicBertWeights::load(&gguf, &cfg, MlxDevice::new().expect("device"))
            .expect("load nomic native storage");
        let table = weights.token_embd_weight().expect("table");
        assert_eq!(table.info.ggml_dtype, GgmlType::Q4_0);
        assert!(table.buffer.is_file_backed());
        assert!(table.affine.is_none());
        assert_eq!(weights.len(), 3);
    }

    #[test]
    fn nomic_unsupported_embedding_fails_before_truncated_payload_mapping() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("nomic-unsupported-truncated.gguf");
        write_fixture(
            &path,
            &[
                TestTensor {
                    name: "token_embd.weight".into(),
                    outer_shape: vec![4, 64],
                    ggml_type: 7,
                    bytes: vec![0u8; 4 * 48],
                },
                TestTensor {
                    name: "token_embd_norm.weight".into(),
                    outer_shape: vec![64],
                    ggml_type: 0,
                    bytes: f32_state(64, 1.0),
                },
                TestTensor {
                    name: "token_embd_norm.bias".into(),
                    outer_shape: vec![64],
                    ggml_type: 0,
                    bytes: f32_state(64, 0.0),
                },
            ],
            false,
        );
        let cfg = NomicBertConfig {
            hidden_size: 64,
            num_attention_heads: 1,
            num_hidden_layers: 0,
            intermediate_size: 128,
            max_position_embeddings: 32,
            vocab_size: 4,
            type_vocab_size: 2,
            layer_norm_eps: 1e-5,
            pooling_type: PoolingType::Mean,
            rope_freq_base: 1000.0,
            causal_attention: false,
        };
        let gguf = GgufFile::open(&path).expect("header-only Nomic GGUF opens");
        let error = LoadedNomicBertWeights::load(&gguf, &cfg, MlxDevice::new().expect("device"))
            .expect_err("Q5_1 Nomic embedding has no native gather");
        let message = error.to_string();
        assert!(
            message.contains("Q5_1") && message.contains("native"),
            "{message}"
        );
        assert!(
            !message.contains("map") && !message.contains("payload"),
            "{message}"
        );
    }
}
