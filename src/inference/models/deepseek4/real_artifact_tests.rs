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
