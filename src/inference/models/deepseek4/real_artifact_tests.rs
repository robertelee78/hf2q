use std::path::PathBuf;

use mlx_native::gguf::GgufFile;

use super::model::Deepseek4Model;
use super::weights::required_tensor_specs;

fn official_artifact() -> (PathBuf, GgufFile) {
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
fn official_artifact_executes_native_uncompressed_prefix() {
    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    let mut cache = model.allocate_cache(128).expect("allocate 128-token cache");
    let state = model
        .forward_uncompressed_attention_one(None, 0, 0, &mut cache, false)
        .expect("execute native layer-0 attention without early cache publication");
    let state = model
        .forward_layer0_ffn_one(&state, 0)
        .expect("execute native layer-0 hash-routed FFN");
    assert_eq!(cache.position(), 0);
    let state = model
        .forward_uncompressed_attention_one(Some(&state), 0, 1, &mut cache, false)
        .expect("execute native layer-1 attention from preceding HC state");
    let state = model
        .forward_ffn_one(&state, 0, 1)
        .expect("execute native layer-1 hash-routed FFN");
    cache.commit_step(0).expect("publish complete prefix state");
    let values = state.as_slice::<f32>().expect("read final HC state");
    assert_eq!(state.shape(), &[1, 4, 4096]);
    assert_eq!(cache.position(), 1);
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().any(|value| *value != 0.0));
    eprintln!(
        "executed uncompressed layers 0-1 from {} with {} resident weight bytes",
        path.display(),
        model.weights.resident_bytes()
    );
}
