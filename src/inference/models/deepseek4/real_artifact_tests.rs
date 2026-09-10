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
    let bound = crate::inference::glp::BoundGlp::bind(
        vector,
        Some(1.0),
        &device,
        crate::inference::glp::GlpHookPoint::FfnOutPreResidual,
        model.cfg.num_hidden_layers,
        model.cfg.hidden_size,
    )
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

#[test]
#[ignore = "requires the locally converted GGUF on disk"]
fn glp_layer_mapping_differential_probe() {
    // ADR-053 gate 2 / spec GLP.md: "direction.N applies at layer N. No
    // offset." A one-layer shift degrades silently — adjacent layers'
    // refusal directions have cosine 0.555–0.979, so a shifted stack still
    // ablates and still passes a smoke test; only this differential probe
    // finds it. Method: bind a synthetic vector holding ONE direction at
    // layer K, then capture the post-layer state at K-1 and at K, steered
    // and unsteered. The capture at K-1 (completed before the steered
    // layer runs) must be identical; the capture at K must differ. An
    // off-by-one apply site fails exactly one of the two assertions.
    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    let num_layers = model.cfg.num_hidden_layers as usize;
    let hidden = model.cfg.hidden_size as usize;
    let k: usize = 29.min(num_layers - 2);
    assert!(k >= 1, "probe layer must be expressible (>= 1)");

    // One non-trivial direction at exactly layer K (0-based graph layer,
    // exported/loaded verbatim per spec).
    let direction: Vec<f32> = (0..hidden).map(|i| 0.001 * (i % 7 + 1) as f32).collect();
    let vector = crate::inference::glp::GlpVector {
        mode: crate::inference::glp::GlpMode::Project,
        hook_point: crate::inference::glp::GlpHookPoint::FfnOutPreResidual,
        derived_at: None,
        alpha_default: 1.0,
        rank: 1,
        layers: std::collections::BTreeMap::from([(k as u32, direction)]),
        width: hidden,
        content_sha256: None,
        method: None,
        base_model_name: None,
    };
    let device = model.ctx.device().clone();
    let bound = crate::inference::glp::BoundGlp::bind(
        vector,
        Some(1.0),
        &device,
        crate::inference::glp::GlpHookPoint::FfnOutPreResidual,
        num_layers as u32,
        hidden as u32,
    )
    .expect("bind synthetic single-layer vector");

    // Capture helper: fresh cache per call (capture contract), fixed probe
    // tokens, post-layer state at the requested layer.
    let capture = |model: &mut Deepseek4Model, layer: usize| -> Vec<f32> {
        let mut cache = model.allocate_cache(128).expect("cache");
        model
            .forward_verifier_prefill_capture_layer(&[1, 2, 3], &mut cache, layer)
            .unwrap_or_else(|error| panic!("capture at layer {layer}: {error:#}"))
    };

    let below = capture(&mut model, k - 1);
    let at = capture(&mut model, k);

    model.glp = Some(bound);
    let below_steered = capture(&mut model, k - 1);
    let at_steered = capture(&mut model, k);
    model.glp = None;

    let below_delta: f32 = below
        .iter()
        .zip(&below_steered)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let at_delta: f32 = at
        .iter()
        .zip(&at_steered)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);

    assert_eq!(
        below_delta, 0.0,
        "direction.{k} must NOT touch the post-layer-{} state (applied one layer early?)",
        k - 1
    );
    assert!(
        at_delta > 1e-4,
        "direction.{k} must change the post-layer-{k} state (applied one layer late, or dead hook?)"
    );
    eprintln!(
        "differential probe: direction.{k} — below-layer delta {below_delta:.8}, at-layer delta {at_delta:.6}"
    );
}
