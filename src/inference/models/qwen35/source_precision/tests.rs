use std::collections::BTreeMap;
use std::ffi::CString;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::ffi::OsStrExt;

use safetensors::tensor::TensorView;
use safetensors::Dtype;
use sha1::Sha1;
use sha2::{Digest, Sha256};

use super::*;
use crate::core::integrity::ShardIntegrity;
use crate::core::provenance::{compute_source_bundle_sha256, SourceShard};
use crate::input::integrity::verify_conversion_manifest;
use crate::intelligence::dynamic_allocator::producer::{
    build_tensor_partition, derive_source_tensor_inventory, NonVariableDisposition,
    NonVariableTensor, TensorPartitionManifest, VerifiedSourceTensorInventory,
};
use crate::intelligence::dynamic_allocator::{
    ScalarDType, TensorAllocationUnit, TensorMember, TensorOperation,
};
use crate::intelligence::measured_auto_quant::SourceIdentity;

const MODEL_ID: &str = "Qwen/Qwen3.8-test";
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

struct SnapshotFixture {
    temp: tempfile::TempDir,
    verified: crate::input::integrity::VerifiedSourceManifest,
    inventory: VerifiedSourceTensorInventory,
    partition: TensorPartitionManifest,
    units: Vec<TensorAllocationUnit>,
    lm_head_words: Vec<u16>,
}

fn raw_u16(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn git_record(filename: &str, bytes: &[u8]) -> ShardIntegrity {
    let mut git = Sha1::new();
    git.update(format!("blob {}\0", bytes.len()).as_bytes());
    git.update(bytes);
    ShardIntegrity {
        filename: filename.into(),
        bytes: bytes.len() as u64,
        sha256: None,
        hf_etag: hex::encode(git.finalize()),
        is_lfs: false,
    }
}

fn lfs_record(filename: &str, bytes: &[u8]) -> ShardIntegrity {
    let sha256 = hex::encode(Sha256::digest(bytes));
    let uppercase = sha256.to_ascii_uppercase();
    ShardIntegrity {
        filename: filename.into(),
        bytes: bytes.len() as u64,
        sha256: Some(uppercase.clone()),
        hf_etag: uppercase,
        is_lfs: true,
    }
}

fn member(
    inventory: &VerifiedSourceTensorInventory,
    name: &str,
    source_dtype: ScalarDType,
) -> TensorMember {
    let source = inventory
        .manifest()
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap();
    TensorMember {
        name: name.into(),
        shape: source.source_shape.clone(),
        role: "lm_head".into(),
        source_dtype,
        source_tensor_sha256: source.source_tensor_sha256.clone(),
        layer_index: None,
        expert_index: None,
    }
}

fn non_variable(
    inventory: &VerifiedSourceTensorInventory,
    name: &str,
    disposition: NonVariableDisposition,
) -> NonVariableTensor {
    NonVariableTensor {
        source: inventory
            .manifest()
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap()
            .clone(),
        disposition,
        reason: "source-teacher-v1 disposition".into(),
    }
}

fn fixture(dtype: Dtype) -> SnapshotFixture {
    let temp = tempfile::tempdir().unwrap();
    let config = serde_json::to_vec(&serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": { "model_type": "qwen3_5_text" }
    }))
    .unwrap();
    std::fs::write(temp.path().join("config.json"), &config).unwrap();

    let word_width = dtype.bitsize() / 8;
    let make_payload = |start: u16| -> Vec<u8> {
        if word_width == 2 {
            raw_u16(&[start, start + 1, start + 2, start + 3])
        } else {
            (0..(4 * word_width))
                .map(|offset| (usize::from(start) + offset) as u8)
                .collect()
        }
    };
    let embed = make_payload(1);
    let lm_head = make_payload(11);
    let visual = make_payload(21);
    let mtp = make_payload(31);
    let embed_view = TensorView::new(dtype, vec![2, 2], &embed).unwrap();
    let lm_head_view = TensorView::new(dtype, vec![2, 2], &lm_head).unwrap();
    let visual_view = TensorView::new(dtype, vec![2, 2], &visual).unwrap();
    let mtp_view = TensorView::new(dtype, vec![2, 2], &mtp).unwrap();
    let shard = safetensors::tensor::serialize(
        vec![
            (
                "model.language_model.embed_tokens.weight".to_owned(),
                &embed_view,
            ),
            ("lm_head.weight".to_owned(), &lm_head_view),
            ("model.visual.patch.weight".to_owned(), &visual_view),
            ("mtp.norm.weight".to_owned(), &mtp_view),
        ],
        None,
    )
    .unwrap();
    std::fs::write(temp.path().join("model.safetensors"), &shard).unwrap();

    let config_record = git_record("config.json", &config);
    let shard_record = lfs_record("model.safetensors", &shard);
    let verified = verify_conversion_manifest(
        MODEL_ID,
        REVISION,
        temp.path(),
        vec![config_record, shard_record.clone()],
    )
    .unwrap();
    let source = SourceIdentity {
        model_id: MODEL_ID.into(),
        revision: REVISION.into(),
        config_sha256: hex::encode(Sha256::digest(&config)),
        tensor_bundle_sha256: compute_source_bundle_sha256(&[SourceShard::from_integrity(
            &shard_record,
        )])
        .unwrap(),
        tokenizer_bundle_sha256: "1".repeat(64),
        chat_template_sha256: "2".repeat(64),
    };
    let inventory = derive_source_tensor_inventory(temp.path(), source, &verified).unwrap();
    let source_dtype = match dtype {
        Dtype::BF16 => ScalarDType::Bf16,
        Dtype::F16 => ScalarDType::F16,
        Dtype::F32 => ScalarDType::F32,
        _ => panic!("fixture dtype is outside the test surface"),
    };
    let units = vec![TensorAllocationUnit {
        unit_id: "lm-head".into(),
        members: vec![member(&inventory, "lm_head.weight", source_dtype)],
        expected_expert_ids: Vec::new(),
        operations: vec![TensorOperation {
            operation_id: "lm-head".into(),
            graph_path: "qwen35.output_head".into(),
            tensor_names: vec!["lm_head.weight".into()],
        }],
        options: Vec::new(),
    }];
    let partition = build_tensor_partition(
        &inventory,
        &units,
        vec![
            non_variable(
                &inventory,
                "model.language_model.embed_tokens.weight",
                NonVariableDisposition::Protected,
            ),
            non_variable(
                &inventory,
                "model.visual.patch.weight",
                NonVariableDisposition::Excluded,
            ),
            non_variable(
                &inventory,
                "mtp.norm.weight",
                NonVariableDisposition::Protected,
            ),
        ],
    )
    .unwrap();
    SnapshotFixture {
        temp,
        verified,
        inventory,
        partition,
        units,
        lm_head_words: if word_width == 2 {
            vec![11, 12, 13, 14]
        } else {
            Vec::new()
        },
    }
}

fn limits(fixture: &SnapshotFixture) -> QwenSourceSnapshotLimits {
    QwenSourceSnapshotLimits {
        max_shards: 1,
        max_tensors: 4,
        max_header_bytes_per_shard: 16 * 1024,
        max_total_header_bytes: 16 * 1024,
        max_config_bytes: 16 * 1024,
        max_total_source_bytes: fixture
            .verified
            .records()
            .iter()
            .map(|record| record.bytes)
            .sum(),
    }
}

fn open(fixture: &SnapshotFixture) -> anyhow::Result<VerifiedQwenSourceSnapshot> {
    open_verified_qwen_source_snapshot(
        fixture.temp.path(),
        &fixture.verified,
        &fixture.inventory,
        &fixture.partition,
        &fixture.units,
        limits(fixture),
    )
}

#[test]
fn retained_snapshot_reads_exact_u16_and_rejects_persistent_mutation() {
    let fixture = fixture(Dtype::BF16);
    let snapshot = open(&fixture).unwrap();
    assert_eq!(snapshot.source().model_id, MODEL_ID);
    assert_eq!(
        snapshot.config()["text_config"]["model_type"],
        "qwen3_5_text"
    );
    assert_eq!(snapshot.tensor_count(), 4);
    assert_eq!(snapshot.catalog_sha256().len(), 64);
    assert_eq!(
        snapshot.verified_source_manifest_sha256(),
        fixture.inventory.manifest().verified_source_manifest_sha256
    );
    assert_eq!(
        snapshot.source_inventory_manifest_sha256(),
        fixture.inventory.manifest().manifest_sha256
    );
    assert_eq!(
        snapshot.tensor_partition_manifest_sha256(),
        fixture.partition.manifest_sha256
    );
    let mut words = [0_u16; 4];
    snapshot
        .read_tensor_u16("lm_head.weight", &mut words)
        .unwrap();
    assert_eq!(words, fixture.lm_head_words.as_slice());
    snapshot.rehash_retained_files().unwrap();

    let old = fixture.temp.path().join("retained-model.safetensors");
    let current = fixture.temp.path().join("model.safetensors");
    std::fs::rename(&current, &old).unwrap();
    std::fs::write(&current, b"untrusted replacement").unwrap();
    words.fill(0);
    snapshot
        .read_tensor_u16("lm_head.weight", &mut words)
        .unwrap();
    assert_eq!(words, fixture.lm_head_words.as_slice());

    let mut file = OpenOptions::new().write(true).open(&old).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    file.write_all(&vec![0_u8; file.metadata().unwrap().len() as usize])
        .unwrap();
    file.sync_all().unwrap();
    assert!(snapshot
        .read_tensor_u16("lm_head.weight", &mut words)
        .is_err());
    assert!(snapshot.rehash_retained_files().is_err());
}

#[test]
fn retained_snapshot_accepts_f16_without_expanding_it() {
    let fixture = fixture(Dtype::F16);
    let snapshot = open(&fixture).unwrap();
    let mut words = [0_u16; 4];
    snapshot
        .read_tensor_u16("lm_head.weight", &mut words)
        .unwrap();
    assert_eq!(words, fixture.lm_head_words.as_slice());
}

#[test]
fn snapshot_rejects_f32_and_disposition_drift() {
    let f32_fixture = fixture(Dtype::F32);
    assert!(open(&f32_fixture)
        .unwrap_err()
        .to_string()
        .contains("outside BF16/F16 teacher scope"));

    let mut disposition_fixture = fixture(Dtype::BF16);
    let non_variable = |name: &str, disposition| NonVariableTensor {
        source: disposition_fixture
            .inventory
            .manifest()
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap()
            .clone(),
        disposition,
        reason: "disposition test".into(),
    };
    disposition_fixture.partition = build_tensor_partition(
        &disposition_fixture.inventory,
        &disposition_fixture.units,
        vec![
            non_variable(
                "model.language_model.embed_tokens.weight",
                NonVariableDisposition::Protected,
            ),
            non_variable(
                "model.visual.patch.weight",
                NonVariableDisposition::Excluded,
            ),
            non_variable("mtp.norm.weight", NonVariableDisposition::Excluded),
        ],
    )
    .unwrap();
    assert!(open(&disposition_fixture)
        .unwrap_err()
        .to_string()
        .contains("text source tensor mtp.norm.weight cannot be excluded"));
}

#[test]
fn scope_uses_family_source_namespace_and_exact_wrapper_config() {
    let dispositions = BTreeMap::from([
        (
            "lm_head.weight".into(),
            super::types::SourcePrecisionDisposition::Variable,
        ),
        (
            "model.visual.patch.weight".into(),
            super::types::SourcePrecisionDisposition::Excluded,
        ),
    ]);
    super::scope::validate_teacher_dispositions(&dispositions).unwrap();
    let invalid_alias = BTreeMap::from([
        (
            "lm_head.weight".into(),
            super::types::SourcePrecisionDisposition::Variable,
        ),
        (
            "visual.patch.weight".into(),
            super::types::SourcePrecisionDisposition::Excluded,
        ),
    ]);
    assert!(super::scope::validate_teacher_dispositions(&invalid_alias).is_err());
    let invalid_mtp = BTreeMap::from([
        (
            "lm_head.weight".into(),
            super::types::SourcePrecisionDisposition::Variable,
        ),
        (
            "mtp.norm.weight".into(),
            super::types::SourcePrecisionDisposition::Variable,
        ),
    ]);
    assert!(super::scope::validate_teacher_dispositions(&invalid_mtp)
        .unwrap_err()
        .to_string()
        .contains("must be fixed or protected"));

    let mut invalid = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "unrelated_wrapper",
        "text_config": { "model_type": "qwen3_5_text" }
    });
    assert!(super::scope::validate_dense_qwen_source_config(&invalid).is_err());
    invalid["model_type"] = serde_json::json!("qwen3_5");
    super::scope::validate_dense_qwen_source_config(&invalid).unwrap();
    let causal = serde_json::json!({
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text"
    });
    super::scope::validate_dense_qwen_source_config(&causal).unwrap();
    let mut invalid_causal = causal;
    invalid_causal["text_config"] = serde_json::json!({"model_type": "qwen3_5_text"});
    assert!(super::scope::validate_dense_qwen_source_config(&invalid_causal).is_err());

    let duplicate_nested = br#"{
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {"model_type": "qwen3_5_text", "model_type": "other"}
    }"#;
    assert!(super::scope::parse_unique_qwen_config(duplicate_nested)
        .unwrap_err()
        .to_string()
        .contains("duplicate config key model_type"));
}

#[test]
fn retained_open_rejects_fifo_without_blocking() {
    let fixture = fixture(Dtype::BF16);
    let config_path = fixture.temp.path().join("config.json");
    std::fs::remove_file(&config_path).unwrap();
    let path = CString::new(config_path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    assert!(open(&fixture)
        .unwrap_err()
        .to_string()
        .contains("source config is not the expected regular file"));
}

#[test]
fn snapshot_limits_are_hard_bounded() {
    let fixture = fixture(Dtype::BF16);
    let mut unbounded = limits(&fixture);
    unbounded.max_total_source_bytes = u64::MAX;
    assert!(open_verified_qwen_source_snapshot(
        fixture.temp.path(),
        &fixture.verified,
        &fixture.inventory,
        &fixture.partition,
        &fixture.units,
        unbounded,
    )
    .is_err());

    let mut tiny = limits(&fixture);
    tiny.max_header_bytes_per_shard = 1;
    assert!(open_verified_qwen_source_snapshot(
        fixture.temp.path(),
        &fixture.verified,
        &fixture.inventory,
        &fixture.partition,
        &fixture.units,
        tiny,
    )
    .is_err());
}

#[test]
fn snapshot_catalog_is_name_sorted() {
    let fixture = fixture(Dtype::BF16);
    let snapshot = open(&fixture).unwrap();
    let names = snapshot.tensor_names_for_test();
    let expected = BTreeMap::from([
        ("lm_head.weight", ()),
        ("model.language_model.embed_tokens.weight", ()),
        ("model.visual.patch.weight", ()),
        ("mtp.norm.weight", ()),
    ]);
    assert_eq!(names, expected.keys().copied().collect::<Vec<_>>());
}
