//! Synthetic CPU/filesystem fixtures for the actual raw_audit entrypoint.
//! No model/device or real acquisition claim: parent/drain/identity fields below
//! are explicitly synthetic. Each test owns one 16,220,288-byte temp fixture.
use super::*;

const RAW_BYTES: u64 = 16_220_288;
// libtest may run tests concurrently; hold this lease through tempdir cleanup
// so the entire module owns at most one full raw payload fixture at a time.
static FIXTURE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
struct Fixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
    trace: Trace,
    frozen: Frozen,
    _serial: std::sync::MutexGuard<'static, ()>,
}

fn pages(bytes: &[u8]) -> Vec<Page> {
    bytes.chunks(PAGE_BYTES).enumerate().map(|(i, chunk)| Page {
        offset: (i * PAGE_BYTES) as u64,
        bytes: chunk.len(),
        sha256: hash(chunk),
    }).collect()
}

impl Fixture {
    fn new() -> Self {
        let serial = FIXTURE_SERIAL.lock().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        let cfg = qwen_cache::selected_header_fixture(true);
        let frozen = Frozen {
            workload: Workload {
                family: Family::Qwen38,
                vocab_size: cfg.vocab_size,
                lanes: (0..4).map(|lane| {
                    let mut ids = vec![7; 144];
                    ids[0] = lane as u32;
                    ids[130] = 31 + lane as u32;
                    ids[131] = 41 + lane as u32;
                    ids
                }).collect(),
            },
            final_carry: std::array::from_fn(|lane| Carry {
                token: 55 + lane as u32, terminal: false,
            }),
            forced_bad_ids: [0; 4],
        };
        let identity = Identity {
            family: Family::Qwen38,
            source_commit: "1".repeat(40), binary_sha256: "2".repeat(64),
            native_version: "0.15.7".into(), native_checksum: "3".repeat(64),
            model_sha256: "4".repeat(64), tokenizer_sha256: "5".repeat(64),
            template_sha256: "6".repeat(64), config_sha256: "7".repeat(64),
            token_vectors_sha256: "8".repeat(64),
        };
        let expected = qwen_speculation::raw(&cfg);
        assert_eq!(expected.len(), 20);
        let mut captured = Vec::new();
        for (index, item) in expected.iter().enumerate() {
            let lane = item.at.agent.strip_prefix("agent-").unwrap().parse::<usize>().unwrap();
            // At most one full raw tensor host Vec at a time; no duplicated
            // sixteen-MiB payload owner or native allocation is constructed.
            let data = match item.scalar {
                Scalar::F32 => {
                    let elements = item.shape.iter().product::<usize>();
                    let mut data = Vec::with_capacity(elements * 4);
                    for i in 0..elements {
                        let value = ((i % 257) as f32 - 128.0) / 512.0 + index as f32 / 64.0;
                        data.extend_from_slice(&value.to_le_bytes());
                    }
                    data
                }
                Scalar::U32 => {
                    let ids = match item.at.stage {
                        Stage::SpecOriginalDraftIds => vec![101 + lane as u32, 111 + lane as u32, 121 + lane as u32],
                        Stage::SpecForcedDraftIds => vec![31 + lane as u32, 0, 121 + lane as u32],
                        Stage::SpecCanonicalIds => vec![31 + lane as u32, 41 + lane as u32],
                        _ => panic!("unexpected source-authoritative ID stage"),
                    };
                    ids.into_iter().flat_map(u32::to_le_bytes).collect()
                }
                _ => panic!("unexpected source-authoritative scalar"),
            };
            assert!(data.len() <= 4 * 1024 * 1024);
            let file = format!("tensor-{index}.bin");
            OpenOptions::new().write(true).create_new(true).mode(0o600)
                .open(root.join(&file)).unwrap().write_all(&data).unwrap();
            captured.push(Tensor {
                at: item.at.clone(), scalar: item.scalar, shape: item.shape.clone(),
                logical_bytes: data.len() as u64, parent_identity: index as u64 + 1,
                parent_bytes: data.len() as u64 + 32, parent_offset: 32,
                submission_id: None,
                checked_drain: "SYNTHETIC unit fixture; not a GPU completion attestation".into(),
                file, sha256: hash(&data), pages: pages(&data),
            });
        }
        assert_eq!(captured.iter().map(|t| t.logical_bytes).sum::<u64>(), RAW_BYTES);
        let acquisition = acquisition::Acquisition {
            packet_sha256: "9".repeat(64), nonce: "a".repeat(64), output_root: root.clone(),
            actual_binary_sha256: identity.binary_sha256.clone(),
            composition_sha256: "b".repeat(64), independent_review_sha256: "c".repeat(64),
            opened_model: acquisition::FileIdentity {
                device: 1, inode: 2, bytes: 3, mtime: 4, mtime_ns: 5, ctime: 6, ctime_ns: 7,
            },
            observed_model_opens: 2, required_model_opens: 2,
            device_registry_id: 1, queue_identity: 2, model_instance: 3,
            provenance: "SYNTHETIC unit fixture; not a reviewed runtime acquisition".into(),
        };
        let trace = Trace {
            schema: "hf2q.adr063.invariance.v2".into(),
            comparison_key: (&identity).into(), identity, acquisition: Some(acquisition),
            coverage: Coverage {
                expected, captured, errors: vec![], missing_native_capabilities: vec![],
                trajectory: "QwenT5:ForcedProposal:raw-verifier".into(),
            },
        };
        let fixture = Self { _temporary: temporary, root, trace, frozen, _serial: serial };
        // Every mutation begins only AFTER a genuine complete positive call
        // through BOTH production helper entrypoints on that same fresh fixture.
        fixture.positive();
        fixture
    }

    fn positive(&self) {
        manifest(&self.trace, &self.trace.identity, &self.trace.coverage.expected,
            "QwenT5:ForcedProposal:raw-verifier", &self.root).unwrap();
        let report = raw_audit(&self.root, &self.trace, &self.frozen).unwrap();
        assert_eq!(report["records"], 20);
        assert_eq!(report["payload_bytes"], RAW_BYTES);
        assert_eq!(report["all_f32_finite"], true);
        assert_eq!(report["rejected_tail_compared_to_baseline"], false);
    }

    fn index(&self, stage: Stage) -> usize {
        self.trace.coverage.captured.iter().position(|t|
            t.at.agent == "agent-0" && t.at.stage == stage).unwrap()
    }

    fn rewrite(&mut self, index: usize, edit: impl FnOnce(&mut [u8]), rehash_pages: bool) {
        let tensor = &mut self.trace.coverage.captured[index];
        let path = self.root.join(&tensor.file);
        let mut data = std::fs::read(&path).unwrap();
        edit(&mut data);
        std::fs::write(&path, &data).unwrap();
        tensor.sha256 = hash(&data);
        if rehash_pages { tensor.pages = pages(&data); }
    }

    fn reject(&self, reason: &str) {
        let error = raw_audit(&self.root, &self.trace, &self.frozen).unwrap_err();
        assert!(format!("{error:#}").contains(reason), "wrong rejection: {error:#}");
    }
}

#[test]
fn raw_audit_complete_synthetic_fixture_passes_without_runtime_claim() {
    let fixture = Fixture::new();
    assert_eq!(fixture.trace.coverage.captured.len(), 20);
}

#[test]
fn raw_audit_payload_change_without_full_hash_rebind_is_refused() {
    let fixture = Fixture::new();
    let tensor = &fixture.trace.coverage.captured[fixture.index(Stage::SpecVerifierLogits)];
    let path = fixture.root.join(&tensor.file);
    let mut data = std::fs::read(&path).unwrap();
    data[0] ^= 1;
    std::fs::write(path, data).unwrap();
    fixture.reject("bound file hash");
}

#[test]
fn raw_audit_page_hash_failure_after_correct_full_rehash_is_refused() {
    let mut fixture = Fixture::new();
    let index = fixture.index(Stage::SpecVerifierLogits);
    fixture.rewrite(index, |data| data[0] ^= 1, false);
    fixture.reject("raw contiguous full page hash");
}

#[test]
fn raw_audit_page_offset_gap_is_refused_with_valid_payload_hashes() {
    let mut fixture = Fixture::new();
    let index = fixture.index(Stage::SpecVerifierLogits);
    fixture.trace.coverage.captured[index].pages[1].offset += 1;
    fixture.reject("raw contiguous full page hash");
}

#[test]
fn raw_audit_nan_logits_after_all_hashes_rebound_reaches_finite_guard() {
    let mut fixture = Fixture::new();
    let index = fixture.index(Stage::SpecVerifierLogits);
    fixture.rewrite(index, |data| data[..4].copy_from_slice(&f32::NAN.to_le_bytes()), true);
    fixture.reject("nonfinite raw verifier value");
}

#[test]
fn raw_audit_infinite_hidden_after_all_hashes_rebound_reaches_finite_guard() {
    let mut fixture = Fixture::new();
    let index = fixture.index(Stage::SpecVerifierHidden);
    fixture.rewrite(index, |data| data[..4].copy_from_slice(&f32::INFINITY.to_le_bytes()), true);
    fixture.reject("nonfinite raw verifier value");
}

#[test]
fn raw_audit_duplicate_coordinate_with_twenty_records_is_refused() {
    let mut fixture = Fixture::new();
    fixture.trace.coverage.captured[1] = fixture.trace.coverage.captured[0].clone();
    assert_eq!(fixture.trace.coverage.captured.len(), 20);
    fixture.reject("raw duplicate/shape drift");
}

#[test]
fn raw_audit_missing_record_fails_manifest_and_actual_twenty_record_guard() {
    let mut fixture = Fixture::new();
    fixture.trace.coverage.captured.pop();
    let error = manifest(&fixture.trace, &fixture.trace.identity, &fixture.trace.coverage.expected,
        "QwenT5:ForcedProposal:raw-verifier", &fixture.root).unwrap_err();
    assert!(format!("{error:#}").contains("complete selected-family manifest"));
    fixture.reject("all20 raw tensors");
}

#[test]
fn raw_audit_forced_first_id_after_rehash_reaches_identity_guard() {
    let mut fixture = Fixture::new();
    let index = fixture.index(Stage::SpecForcedDraftIds);
    fixture.rewrite(index, |data| data[..4].copy_from_slice(&32u32.to_le_bytes()), true);
    fixture.reject("raw proposal/canonical identity");
}

#[test]
fn raw_audit_forced_third_id_must_preserve_original_tail_after_rehash() {
    let mut fixture = Fixture::new();
    let index = fixture.index(Stage::SpecForcedDraftIds);
    fixture.rewrite(index, |data| data[8..12].copy_from_slice(&122u32.to_le_bytes()), true);
    fixture.reject("raw proposal/canonical identity");
}

#[test]
fn raw_audit_canonical_id_after_rehash_reaches_frozen_lineage_guard() {
    let mut fixture = Fixture::new();
    let index = fixture.index(Stage::SpecCanonicalIds);
    fixture.rewrite(index, |data| data[..4].copy_from_slice(&32u32.to_le_bytes()), true);
    fixture.reject("raw proposal/canonical identity");
}

#[test]
fn raw_audit_original_out_of_vocabulary_id_after_rehash_is_refused() {
    let mut fixture = Fixture::new();
    let index = fixture.index(Stage::SpecOriginalDraftIds);
    let invalid = fixture.frozen.workload.vocab_size;
    fixture.rewrite(index, |data| data[..4].copy_from_slice(&invalid.to_le_bytes()), true);
    fixture.reject("raw proposal/canonical identity");
}

#[test]
fn raw_audit_bad_proposal_cannot_equal_expected_rejected_id() {
    let mut fixture = Fixture::new();
    let index = fixture.index(Stage::SpecForcedDraftIds);
    let bad = fixture.frozen.workload.lanes[0][131];
    fixture.frozen.forced_bad_ids[0] = bad;
    fixture.rewrite(index, |data| data[4..8].copy_from_slice(&bad.to_le_bytes()), true);
    fixture.reject("raw proposal/canonical identity");
}
