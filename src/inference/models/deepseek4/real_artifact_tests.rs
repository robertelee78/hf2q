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
#[ignore = "requires the locally converted GGUF + a probe GLP on disk"]
fn same_boot_glp_steering_gate() {
    // ADR-053 gate: within one process (no restart noise), prove (a) the
    // unsteered forward logits are reproducible against themselves, and
    // (b) the GLP hook shifts logits above the within-process noise floor.
    // Cross-restart comparison is documented kernel noise for this stack
    // (MARLIN atomic-add MoE path); this test never leaves the process.
    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    let glp_path = std::env::var("HF2Q_DEEPSEEK4_GLP")
        .expect("set HF2Q_DEEPSEEK4_GLP to a probe GLP GGUF");

    // Baseline: unsteered, twice — must be identical within the process.
    let baseline = |model: &mut Deepseek4Model| -> Vec<f32> {
        model.glp = None;
        let mut cache = model.allocate_cache(128).expect("cache");
        let state = model
            .forward_verifier_one(0, &mut cache)
            .expect("verifier one");
        let logits = model.forward_logits(&state).expect("logits");
        logits.as_slice::<f32>().expect("slice").to_vec()
    };
    let base1 = baseline(&mut model);
    let base2 = baseline(&mut model);
    let within_noise: f32 = base1
        .iter()
        .zip(&base2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        within_noise < 1e-4,
        "within-process baseline must be reproducible, got within_noise {within_noise}"
    );

    // Steered: bind the probe vector, rerun the same single-token forward.
    let device = model.ctx.device().clone();
    let vector = crate::inference::glp::GlpVector::load(std::path::Path::new(&glp_path))
        .expect("load probe GLP");
    let bound = crate::inference::glp::BoundGlp::bind(vector, Some(1.0), &device)
        .expect("bind probe GLP");
    let steered_values = {
        model.glp = Some(bound);
        let mut cache = model.allocate_cache(128).expect("cache");
        let state = model
            .forward_verifier_one(0, &mut cache)
            .expect("verifier one steered");
        let logits = model.forward_logits(&state).expect("logits steered");
        logits.as_slice::<f32>().expect("slice").to_vec()
    };
    let shift: f32 = base1
        .iter()
        .zip(&steered_values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    eprintln!("GLP same-boot gate: within_noise={within_noise:.6} steered shift={shift:.6}");
    assert!(
        shift > 1e-3,
        "GLP hook at a real dose must shift logits > 1e-3, got {shift}"
    );
}
