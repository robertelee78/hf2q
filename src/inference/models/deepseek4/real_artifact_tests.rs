use std::path::PathBuf;

use mlx_native::gguf::GgufFile;

use super::model::Deepseek4Model;
use super::weights::required_tensor_specs;

pub(super) fn official_artifact() -> (PathBuf, GgufFile) {
    let path = PathBuf::from(
        std::env::var("HF2Q_DEEPSEEK4_GGUF")
            .expect("set HF2Q_DEEPSEEK4_GGUF to the locally converted official GGUF"),
    );
    let gguf = GgufFile::open(&path)
        .unwrap_or_else(|error| panic!("open official artifact {}: {error}", path.display()));
    (path, gguf)
}

#[test]
#[ignore = "requires the locally converted 89.65 GiB official checkpoint"]
fn official_artifact_metadata_and_catalog_are_exact() {
    let (path, gguf) = official_artifact();
    let cfg = Deepseek4Model::load_config_only(&gguf).expect("strict official metadata");
    let specs = required_tensor_specs(&cfg);
    assert_eq!(cfg.num_hidden_layers, 43);
    assert_eq!(cfg.vocab_size, 129_280);
    assert_eq!(cfg.compress_ratios.len(), 43);
    assert_eq!(specs.len(), gguf.tensor_names().len());
    eprintln!("validated {} tensors in {}", specs.len(), path.display());
}

#[test]
#[ignore = "loads the locally converted 89.65 GiB official checkpoint onto Metal"]
fn official_artifact_executes_native_verifier_and_logits() {
    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    let mut cache = model.allocate_cache(128).expect("allocate 128-token cache");
    let state = model
        .forward_verifier_one(0, &mut cache)
        .expect("execute all native verifier layers and publish one token");
    let values = state.as_slice::<f32>().expect("read final HC state");
    assert_eq!(state.shape(), &[1, 4, 4096]);
    assert_eq!(cache.position(), 1);
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| *value != 0.0));
    let logits = model
        .forward_logits(&state)
        .expect("collapse final HC state and execute vocabulary projection");
    let logit_values = logits.as_slice::<f32>().expect("read vocabulary logits");
    assert_eq!(logits.shape(), &[1, 129_280]);
    assert!(logit_values.iter().all(|value| value.is_finite()));
    assert!(logit_values.iter().any(|value| *value != 0.0));
    let greedy = model
        .greedy_token(&logits)
        .expect("select greedy token on Metal");
    assert!(greedy < 129_280);
    eprintln!(
        "executed all verifier layers from {} with {} resident weight bytes; greedy token {}",
        path.display(),
        model.weights.resident_bytes(),
        greedy
    );
}

#[test]
#[ignore = "loads the locally converted 89.65 GiB official checkpoint onto Metal"]
fn invalid_moe_receipt_poison_prevents_publish_and_fresh_transaction_recovers() {
    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    let mut rejected = model.allocate_cache(128).expect("allocate rejected cache");
    model.force_invalid_moe_status_once_for_test();
    let error = model
        .forward_verifier_one(0, &mut rejected)
        .err()
        .expect("injected invalid MoE receipt must reject the verifier transaction");
    assert!(error.to_string().contains("invalid MoE state"));
    assert_eq!(rejected.position(), 0, "rejected token must not publish");
    assert!(
        rejected.is_poisoned(),
        "rejected transaction must poison cache"
    );

    let mut fresh = model.allocate_cache(128).expect("allocate fresh cache");
    model
        .seed_invalid_moe_status_for_test()
        .expect("seed prior sticky status");
    model
        .forward_verifier_one(0, &mut fresh)
        .expect("fresh transaction must clear prior sticky status");
    assert_eq!(fresh.position(), 1);
    assert!(!fresh.is_poisoned());

    let mut rejected_prefill = model
        .allocate_cache(128)
        .expect("allocate rejected prefill cache");
    model.force_invalid_moe_status_once_for_test();
    let error = model
        .forward_verifier_prefill(&[0, 1], &mut rejected_prefill)
        .err()
        .expect("injected invalid MoE receipt must reject single prefill");
    assert!(error.to_string().contains("invalid MoE state"));
    assert_eq!(rejected_prefill.position(), 0);
    assert!(rejected_prefill.is_poisoned());

    let mut first = model
        .allocate_cache(128)
        .expect("allocate first cohort cache");
    let mut second = model
        .allocate_cache(128)
        .expect("allocate second cohort cache");
    let mut cohort = [&mut first, &mut second];
    let first_tokens = [0_u32];
    let second_tokens = [1_u32];
    let token_batches = [first_tokens.as_slice(), second_tokens.as_slice()];
    model.force_invalid_moe_status_once_for_test();
    let error = model
        .forward_verifier_prefill_cohort(&token_batches, &mut cohort)
        .err()
        .expect("injected invalid MoE receipt must reject the cohort transaction");
    assert!(error.to_string().contains("invalid MoE state"));
    for cache in cohort {
        assert_eq!(cache.position(), 0, "rejected cohort must not publish");
        assert!(
            cache.is_poisoned(),
            "rejected cohort must poison every lane"
        );
    }

    let mut lane0 = model.allocate_cache(128).expect("allocate B4 lane 0");
    let mut lane1 = model.allocate_cache(128).expect("allocate B4 lane 1");
    let mut lane2 = model.allocate_cache(128).expect("allocate B4 lane 2");
    let mut lane3 = model.allocate_cache(128).expect("allocate B4 lane 3");
    let mut lanes = [&mut lane0, &mut lane1, &mut lane2, &mut lane3];
    model.force_invalid_moe_status_once_for_test();
    let error = model
        .forward_verifier_decode_cohort([0, 1, 2, 3], &mut lanes)
        .err()
        .expect("injected invalid MoE receipt must reject B4 decode");
    assert!(error.to_string().contains("invalid MoE state"));
    for cache in lanes {
        assert_eq!(cache.position(), 0, "rejected B4 lane must not publish");
        assert!(cache.is_poisoned(), "rejected B4 must poison every lane");
    }
}
