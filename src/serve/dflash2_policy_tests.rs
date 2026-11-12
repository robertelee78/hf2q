//! Model-free tests of the DFlash2 default policy: every policy state against
//! every kind of target, with the drafter store and the host stubbed.

use std::cell::Cell;
use std::collections::HashMap;

use super::*;
use crate::inference::spec_decode::dflash2::route_profile::{switch, SwitchSource, SWITCHES};
use crate::inference::spec_decode::dflash2::schema::{
    QWEN36_DFLASH2, QWEN36_TARGET_GEOMETRY, QWEN38_DFLASH2, QWEN38_TARGET_GEOMETRY,
};

const GIB: u64 = 1 << 30;

fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let map: HashMap<String, String> = pairs
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect();
    move |name| map.get(name).cloned()
}

fn tensor(name: &str, ggml_type: GgmlType, shape: &[usize]) -> TensorFact {
    TensorFact {
        name: name.to_owned(),
        ggml_type,
        shape: shape.to_vec(),
    }
}

fn hf2q(source: &str) -> Provenance {
    Provenance::Hf2q {
        producer_version: "hf2q test".into(),
        source_sha256: source.into(),
        mmproj_sha256: None,
    }
}

/// The tensor directory of the Qwen3.8 gate artifact as far as the policy
/// reads it: real shapes, one storage type per weight kind.
fn qwen38() -> TargetFacts {
    let geometry = QWEN38_TARGET_GEOMETRY.clone();
    let mut tensors = vec![
        tensor("output.weight", GgmlType::Q6_K, &[248_320, 5_120]),
        tensor("token_embd.weight", GgmlType::Q4_K, &[248_320, 5_120]),
    ];
    for block in 0..geometry.layers {
        let name = |leaf: &str| format!("blk.{block}.{leaf}.weight");
        tensors.extend([
            tensor(&name("ffn_gate"), GgmlType::Q4_K, &[17_408, 5_120]),
            tensor(&name("ffn_up"), GgmlType::Q4_K, &[17_408, 5_120]),
            tensor(&name("ffn_down"), GgmlType::Q6_K, &[5_120, 17_408]),
            tensor(&name("attn_qkv"), GgmlType::Q6_K, &[10_240, 5_120]),
            tensor(&name("attn_gate"), GgmlType::Q4_K, &[6_144, 5_120]),
            tensor(&name("ssm_out"), GgmlType::Q4_K, &[5_120, 6_144]),
            tensor(&name("attn_norm"), GgmlType::F32, &[5_120]),
        ]);
    }
    // The MTP block's projection is not a route plane.
    tensors.push(tensor(
        "blk.64.nextn.eh_proj.weight",
        GgmlType::Q4_K,
        &[5_120, 10_240],
    ));
    TargetFacts {
        architecture: "qwen35".into(),
        provenance: hf2q(QWEN38_DFLASH2.target_source_sha256.expect("identity")),
        slots: Some(4),
        model_file_bytes: 16_810_714_976,
        tensors,
        geometry: Some(geometry),
        dwq_overlay: false,
        chunk_scan_prefill: false,
    }
}

/// The Qwen3.6 gate artifact, likewise.
fn qwen36() -> TargetFacts {
    let geometry = QWEN36_TARGET_GEOMETRY.clone();
    let mut tensors = vec![
        tensor("output.weight", GgmlType::Q6_K, &[248_320, 2_048]),
        tensor("token_embd.weight", GgmlType::Q4_K, &[248_320, 2_048]),
    ];
    for block in 0..geometry.layers {
        let name = |leaf: &str| format!("blk.{block}.{leaf}.weight");
        tensors.extend([
            tensor(&name("ffn_gate_inp"), GgmlType::F32, &[256, 2_048]),
            tensor(&name("ffn_gate_inp_shexp"), GgmlType::F32, &[2_048]),
            tensor(&name("ffn_gate_shexp"), GgmlType::Q4_K, &[512, 2_048]),
            tensor(&name("ffn_up_shexp"), GgmlType::Q4_K, &[512, 2_048]),
            tensor(&name("ffn_down_shexp"), GgmlType::Q6_K, &[2_048, 512]),
            tensor(&name("ffn_gate_exps"), GgmlType::Q4_K, &[256, 512, 2_048]),
            tensor(&name("ffn_up_exps"), GgmlType::Q4_K, &[256, 512, 2_048]),
            tensor(&name("ffn_down_exps"), GgmlType::Q6_K, &[256, 2_048, 512]),
            tensor(&name("attn_qkv"), GgmlType::Q4_K, &[8_192, 2_048]),
        ]);
    }
    TargetFacts {
        architecture: "qwen35moe".into(),
        provenance: hf2q(QWEN36_DFLASH2.target_source_sha256.expect("identity")),
        slots: Some(4),
        model_file_bytes: 21 * GIB,
        tensors,
        geometry: Some(geometry),
        dwq_overlay: false,
        chunk_scan_prefill: false,
    }
}

/// `facts` with every tensor whose name contains one of `leaves` stored as
/// `ggml_type`.
fn restored(mut facts: TargetFacts, leaves: &[&str], ggml_type: GgmlType) -> TargetFacts {
    for tensor in &mut facts.tensors {
        if leaves.iter().any(|leaf| tensor.name.contains(leaf)) {
            tensor.ggml_type = ggml_type;
        }
    }
    facts
}

const ROOMY: HostFacts = HostFacts {
    physical_bytes: 128 * GIB,
    kv_budget_bytes: 48 * GIB,
};

/// A drafter store with a fixed cache state, download outcome and binding
/// outcome that counts its downloads.
struct Store {
    cached: bool,
    download: std::result::Result<(), &'static str>,
    bind: std::result::Result<(), &'static str>,
    downloads: Cell<u32>,
}

impl Store {
    fn cached() -> Self {
        Self {
            cached: true,
            download: Err("the cache already holds the drafter"),
            bind: Ok(()),
            downloads: Cell::new(0),
        }
    }

    fn empty(download: std::result::Result<(), &'static str>) -> Self {
        Self {
            cached: false,
            ..Self::cached()
        }
        .with_download(download)
    }

    fn with_download(mut self, download: std::result::Result<(), &'static str>) -> Self {
        self.download = download;
        self
    }

    fn files(descriptor: &DFlash2CheckpointDescriptor) -> (PathBuf, PathBuf) {
        let root = PathBuf::from("/hub").join(descriptor.revision);
        (root.join("config.json"), root.join("model.safetensors"))
    }
}

impl DrafterStore for Store {
    fn cached(&self, descriptor: &DFlash2CheckpointDescriptor) -> Option<(PathBuf, PathBuf)> {
        self.cached.then(|| Self::files(descriptor))
    }

    fn download(
        &self,
        descriptor: &DFlash2CheckpointDescriptor,
    ) -> std::result::Result<(PathBuf, PathBuf), String> {
        self.downloads.set(self.downloads.get() + 1);
        self.download
            .map(|()| Self::files(descriptor))
            .map_err(str::to_owned)
    }

    fn bind(
        &self,
        descriptor: &DFlash2CheckpointDescriptor,
        _config_path: &Path,
        target: DFlash2TargetShape<'_>,
    ) -> std::result::Result<(), String> {
        // The policy hands the adapter's check the target's own shape.
        assert_eq!(target.architecture, descriptor.target_architecture);
        assert_eq!(target.hidden_size, descriptor.target_hidden_size);
        assert_eq!(target.vocab_size, descriptor.target_vocab_size);
        assert_eq!(target.num_hidden_layers, descriptor.target_layers);
        self.bind.map_err(str::to_owned)
    }
}

/// A store the policy must not consult.
struct Untouched;

impl DrafterStore for Untouched {
    fn cached(&self, _: &DFlash2CheckpointDescriptor) -> Option<(PathBuf, PathBuf)> {
        panic!("the policy looked for a drafter")
    }

    fn download(
        &self,
        _: &DFlash2CheckpointDescriptor,
    ) -> std::result::Result<(PathBuf, PathBuf), String> {
        panic!("the policy downloaded a drafter")
    }

    fn bind(
        &self,
        _: &DFlash2CheckpointDescriptor,
        _: &Path,
        _: DFlash2TargetShape<'_>,
    ) -> std::result::Result<(), String> {
        panic!("the policy bound a drafter")
    }
}

fn decided(
    facts: &TargetFacts,
    pairs: &[(&str, &str)],
    store: &dyn DrafterStore,
) -> Result<Selection> {
    decide(facts, ROOMY, &env(pairs), store)
}

/// The route set main resolves for `pairs` with no policy: the environment
/// alone.
fn main_routes(pairs: &[(&str, &str)]) -> String {
    ResolvedRoutes::resolve(None, &|_| None, &env(pairs)).audit()
}

/// Asserts `selection` is the prior route: no drafter and every table switch
/// exactly as the environment alone resolves it.
fn assert_prior_route(selection: &Selection, pairs: &[(&str, &str)]) {
    assert!(!selection.selected(), "{}", selection.reason);
    assert!(selection.plan.is_none());
    assert_eq!(selection.routes.profile(), None);
    assert_eq!(selection.routes.audit(), main_routes(pairs));
    assert!(!selection.falls_back_on_load_error());
}

fn withheld(selection: &Selection, name: &str) -> bool {
    selection.routes.source(switch(name)) == SwitchSource::Withheld
        && selection.routes.value(switch(name)).is_none()
}

#[test]
fn policy_value_is_three_states_and_rejects_anything_else() {
    assert_eq!(parse_policy(None), Ok(PolicyRequest::Auto));
    assert_eq!(parse_policy(Some("")), Ok(PolicyRequest::Auto));
    assert_eq!(parse_policy(Some("  ")), Ok(PolicyRequest::Auto));
    for off in ["0", "off", "false", "OFF"] {
        assert_eq!(parse_policy(Some(off)), Ok(PolicyRequest::Off), "{off}");
    }
    for on in ["1", "on", "true", " True "] {
        assert_eq!(parse_policy(Some(on)), Ok(PolicyRequest::Required), "{on}");
    }
    assert!(parse_policy(Some("auto")).is_err());
    assert!(parse_policy(Some("2")).is_err());
    let error = decided(&qwen38(), &[(DFLASH2_ENV, "maybe")], &Untouched).unwrap_err();
    assert!(
        error.to_string().contains("HF2Q_DFLASH2 must be"),
        "{error}"
    );
}

#[test]
fn auto_selects_each_pinned_pair_with_its_whole_profile() {
    for (facts, descriptor) in [(qwen38(), &QWEN38_DFLASH2), (qwen36(), &QWEN36_DFLASH2)] {
        let store = Store::cached();
        let selection = decided(&facts, &[], &store).expect("auto");
        assert_eq!(selection.request, PolicyRequest::Auto);
        let plan = selection.plan.as_ref().expect("selected");
        assert_eq!(plan.descriptor.id, descriptor.id);
        assert_eq!(plan.class, QualifiedClass::PinnedPair);
        assert_eq!(plan.source, DrafterSource::Cache);
        assert_eq!(
            (plan.config_path.clone(), plan.weights_path.clone()),
            Store::files(descriptor)
        );
        assert_eq!(store.downloads.get(), 0, "a cache hit never downloads");
        assert_eq!(selection.reason, "pinned pair");
        assert_eq!(
            Some(selection.target_source.as_str()),
            descriptor.target_source_sha256
        );
        assert_eq!(selection.routes.profile(), RouteProfile::of(descriptor));
        assert!(!selection.routes.any_withheld());
        assert_eq!(selection.profile_label(), "full");
        assert!(selection.falls_back_on_load_error());
        assert!(selection.warnings.is_empty());
        assert_eq!(
            selection.routes.audit(),
            ResolvedRoutes::resolve(RouteProfile::of(descriptor), &|_| None, &env(&[])).audit()
        );
        assert_eq!(selection.descriptor.map(|d| d.id), Some(descriptor.id));
    }
}

#[test]
fn required_selects_a_pinned_pair_and_never_retries_on_the_prior_route() {
    let selection = decided(&qwen38(), &[(DFLASH2_ENV, "1")], &Store::cached()).expect("required");
    assert_eq!(selection.request, PolicyRequest::Required);
    assert!(selection.selected());
    assert_eq!(selection.routes.profile(), Some(RouteProfile::Qwen38));
    assert!(!selection.falls_back_on_load_error());
}

#[test]
fn the_kill_switch_resolves_exactly_the_route_set_of_main() {
    // Each environment an operator could already have set on main.
    let environments: [&[(&str, &str)]; 3] = [
        &[],
        &[("HF2Q_QWEN_PREFILL_PREPARED", "1")],
        &[
            ("HF2Q_QWEN_MOE_BATCHED", "1"),
            ("HF2Q_REPACKED_GGUF_M32", "1"),
            ("HF2Q_DFLASH2_GPU_TARGET_TOKENS", "0"),
        ],
    ];
    let fine_tune = TargetFacts {
        provenance: hf2q(&"f".repeat(64)),
        ..qwen38()
    };
    for facts in [qwen38(), qwen36(), fine_tune] {
        for pairs in environments {
            for off in ["0", "off", "false"] {
                let mut with_switch = pairs.to_vec();
                with_switch.push((DFLASH2_ENV, off));
                // The kill switch looks for no drafter and reads no memory.
                let selection = decided(&facts, &with_switch, &Untouched).expect("off");
                assert_eq!(selection.request, PolicyRequest::Off);
                assert_prior_route(&selection, pairs);
                assert!(selection.memory.is_none());
            }
        }
    }
    // With nothing set, nothing in the table is on.
    let selection = decided(&qwen36(), &[(DFLASH2_ENV, "0")], &Untouched).expect("off");
    for row in SWITCHES {
        assert_eq!(
            selection.routes.value(switch(row.name)),
            None,
            "{}",
            row.name
        );
    }
}

#[test]
fn a_target_with_the_pinned_architecture_and_shapes_qualifies_without_the_source_identity() {
    let source = "f".repeat(64);
    let fine_tune = TargetFacts {
        provenance: hf2q(&source),
        ..qwen38()
    };
    let external = TargetFacts {
        provenance: Provenance::External,
        ..qwen36()
    };
    let cases = [
        (&fine_tune, &QWEN38_DFLASH2, source.as_str()),
        (&external, &QWEN36_DFLASH2, "external"),
    ];
    for (facts, descriptor, target_source) in cases {
        // Auto and required accept the class alike.
        for pairs in [&[][..], &[(DFLASH2_ENV, "1")][..]] {
            let selection = decided(facts, pairs, &Store::cached()).expect("qualifies");
            let plan = selection.plan.as_ref().expect("selected");
            assert_eq!(plan.descriptor.id, descriptor.id);
            assert_eq!(plan.class, QualifiedClass::SameShapes);
            assert_eq!(
                selection.reason,
                format!("same architecture and shapes as {}", descriptor.id)
            );
            assert_eq!(selection.target_source, target_source);
            assert_eq!(selection.routes.profile(), RouteProfile::of(descriptor));
            assert!(!selection.routes.any_withheld());
        }
    }
}

#[test]
fn a_target_that_differs_in_one_shape_keeps_the_route_set_of_main() {
    let reshaped = |change: &dyn Fn(&mut QwenTargetGeometry), base: TargetFacts| {
        let mut facts = TargetFacts {
            provenance: hf2q(&"f".repeat(64)),
            ..base
        };
        change(facts.geometry.as_mut().expect("geometry"));
        facts
    };
    let wide = TargetFacts {
        slots: Some(5),
        ..qwen38()
    };
    let headerless = TargetFacts {
        provenance: Provenance::External,
        geometry: None,
        ..qwen38()
    };
    // The Qwen3.6 source in a dense container is not the Qwen3.6 pair.
    let crossed = TargetFacts {
        architecture: "qwen35".into(),
        ..qwen36()
    };
    let cases = [
        (
            reshaped(&|g| g.hidden_size = 4_096, qwen38()),
            "hidden_size 4096 != pinned 5120",
        ),
        (reshaped(&|g| g.layers = 48, qwen38()), "layers 48 != pinned 64"),
        (
            reshaped(&|g| g.attention_kv_heads = 8, qwen38()),
            "attention_kv_heads 8 != pinned 4",
        ),
        (
            reshaped(&|g| g.tokenizer_sha256 = "0".repeat(64).into(), qwen38()),
            "tokenizer_sha256",
        ),
        (
            reshaped(&|g| g.head_shape = [248_320, 4_096], qwen38()),
            "head_shape",
        ),
        (
            reshaped(&|g| g.moe_top_k = 4, qwen36()),
            "moe_top_k 4 != pinned 8",
        ),
        (
            reshaped(&|g| g.moe_experts = 128, qwen36()),
            "moe_experts 128 != pinned 256",
        ),
        (wide, "1..=4 physical slots, got 5"),
        (headerless, "does not describe a complete Qwen target"),
        (crossed, "architecture"),
    ];
    let pairs: &[(&str, &str)] = &[("HF2Q_QWEN_PREFILL_PREPARED", "1")];
    for (facts, reason) in &cases {
        for environment in [&[][..], pairs] {
            let selection = decided(facts, environment, &Untouched).expect("auto never fails");
            assert_prior_route(&selection, environment);
            assert!(selection.reason.contains(reason), "{}", selection.reason);
            assert!(selection.warnings.is_empty());
        }
        let error = decided(facts, &[(DFLASH2_ENV, "1")], &Untouched).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("HF2Q_DFLASH2=1 cannot be satisfied"),
            "{message}"
        );
        assert!(message.contains(reason), "{message}");
    }
}

#[test]
fn configurations_dflash2_never_served_stay_unserved_without_an_error() {
    let serial = TargetFacts {
        slots: None,
        ..qwen38()
    };
    let other_family = TargetFacts {
        architecture: "gemma4".into(),
        tensors: Vec::new(),
        geometry: None,
        ..qwen38()
    };
    for (facts, reason) in [
        (&serial, "not SlotAware"),
        (&other_family, "no qualified DFlash2 pair"),
    ] {
        let auto = decided(facts, &[], &Untouched).expect("auto");
        assert_prior_route(&auto, &[]);
        assert!(auto.reason.contains(reason), "{}", auto.reason);
        assert!(auto.warnings.is_empty());
        // `HF2Q_DFLASH2=1` was ignored here before the policy existed.
        let required = decided(facts, &[(DFLASH2_ENV, "1")], &Untouched).expect("ignored");
        assert_prior_route(&required, &[]);
        assert_eq!(required.warnings.len(), 1);
        assert!(required.warnings[0].contains("has no effect"));
    }
}

#[test]
fn a_missing_drafter_is_downloaded_once_before_the_model_is_loaded() {
    let store = Store::empty(Ok(()));
    let selection = decided(&qwen38(), &[], &store).expect("auto");
    assert_eq!(store.downloads.get(), 1);
    assert_eq!(
        selection.plan.as_ref().map(|plan| plan.source),
        Some(DrafterSource::Downloaded)
    );
    assert_eq!(selection.routes.profile(), Some(RouteProfile::Qwen38));
}

#[test]
fn a_failed_download_falls_back_under_auto_and_fails_when_required() {
    let store = Store::empty(Err("HF_HUB_OFFLINE is set"));
    let selection = decided(&qwen36(), &[], &store).expect("auto never fails");
    assert_eq!(store.downloads.get(), 1);
    assert_prior_route(&selection, &[]);
    assert_eq!(selection.descriptor.map(|d| d.id), Some(QWEN36_DFLASH2.id));
    assert!(selection.reason.contains("drafter unavailable"));
    let [warning] = selection.warnings.as_slice() else {
        panic!("one WARN line, got {:?}", selection.warnings);
    };
    for part in [
        QWEN36_DFLASH2.repository,
        QWEN36_DFLASH2.revision,
        "HF_HUB_OFFLINE is set",
        "serving the prior route",
    ] {
        assert!(warning.contains(part), "{warning}");
    }

    let error = decided(&qwen36(), &[(DFLASH2_ENV, "1")], &store).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("HF2Q_DFLASH2=1 cannot be satisfied"),
        "{message}"
    );
    assert!(message.contains(QWEN36_DFLASH2.repository), "{message}");
    assert!(message.contains("HF_HUB_OFFLINE is set"), "{message}");
}

#[test]
fn a_checkpoint_the_adapter_check_does_not_bind_is_not_used() {
    let store = Store {
        bind: Err("hidden_size 5120 != draft 4096"),
        ..Store::cached()
    };
    let auto = decided(&qwen38(), &[], &store).expect("auto never fails");
    assert_prior_route(&auto, &[]);
    assert!(
        auto.reason.contains("does not bind to this target"),
        "{}",
        auto.reason
    );
    let error = decided(&qwen38(), &[(DFLASH2_ENV, "1")], &store).unwrap_err();
    assert!(
        error.to_string().contains("hidden_size 5120 != draft 4096"),
        "{error}"
    );
}

#[test]
fn explicit_drafter_paths_win_and_are_used_only_together() {
    let both = [
        (DFLASH2_CONFIG_ENV, "/drafter/config.json"),
        (DFLASH2_WEIGHTS_ENV, "/drafter/model.safetensors"),
    ];
    for policy in [None, Some("1")] {
        let mut pairs = both.to_vec();
        pairs.extend(policy.map(|value| (DFLASH2_ENV, value)));
        // An explicit checkpoint that is not on disk is provisioning's to
        // report: the policy neither looks in the cache nor binds it.
        let selection = decided(&qwen38(), &pairs, &Untouched).expect("explicit");
        let plan = selection.plan.expect("selected");
        assert_eq!(plan.source, DrafterSource::Explicit);
        assert_eq!(plan.config_path, PathBuf::from("/drafter/config.json"));
        assert_eq!(
            plan.weights_path,
            PathBuf::from("/drafter/model.safetensors")
        );
    }
    let one = [(DFLASH2_CONFIG_ENV, "/drafter/config.json")];
    let selection = decided(&qwen38(), &one, &Store::cached()).expect("auto");
    assert_eq!(
        selection.plan.as_ref().map(|plan| plan.source),
        Some(DrafterSource::Cache)
    );
    assert_eq!(selection.warnings.len(), 1);
    assert!(selection.warnings[0].contains("used only together"));
    let error = decided(&qwen38(), &[one[0], (DFLASH2_ENV, "1")], &Store::cached()).unwrap_err();
    assert!(error.to_string().contains("used only together"), "{error}");
}

#[test]
fn the_profile_is_applied_switch_by_switch_to_storage_it_does_not_admit() {
    // The Qwen3.6 launcher default: Q5_K experts, shared expert and
    // projections, a Q8_0 head, no hf2q provenance.
    let apex = restored(
        restored(
            TargetFacts {
                provenance: Provenance::External,
                ..qwen36()
            },
            &["_exps", "_shexp", "attn_qkv"],
            GgmlType::Q5_K,
        ),
        &["output.weight"],
        GgmlType::Q8_0,
    );
    for pairs in [&[][..], &[(DFLASH2_ENV, "1")][..]] {
        let selection = decided(&apex, pairs, &Store::cached()).expect("nothing fails startup");
        assert!(selection.selected(), "{}", selection.reason);
        assert_eq!(selection.profile_label(), "partial");
        for name in [
            "HF2Q_QWEN_MOE_BATCHED",
            "HF2Q_QWEN_MOE_STAGED_M8",
            "HF2Q_QWEN_MOE_LIVE_TILES",
            "HF2Q_REPACKED_GGUF_M32_PROJ",
            "HF2Q_REPACKED_GGUF_M32_HEAD",
            "HF2Q_QWEN_PREFILL_PREPARED",
            "HF2Q_QWEN_PREFILL_FOLD_CUE",
        ] {
            assert!(withheld(&selection, name), "{name}");
        }
        // The drafter serves with every route that does admit the artifact.
        for name in [
            "HF2Q_DFLASH2_DRAFT_AF4_G64",
            "HF2Q_DFLASH2_DRAFT_STAGED",
            "HF2Q_DFLASH2_GPU_TARGET_TOKENS",
            "HF2Q_DFLASH2_CONTEXT_FIRST",
            "HF2Q_REPACKED_GGUF_M32",
            "HF2Q_REPACKED_GGUF_ANY_WIDTH",
            "HF2Q_REPACKED_GGUF_SHARED_HEAD",
            "HF2Q_QWEN_VERIFY_ATTN_TILE",
            "HF2Q_QWEN_VERIFY_MIXER_FUSED",
            "HF2Q_QWEN_VERIFY_REPLAY_REUSE",
        ] {
            assert!(selection.routes.is_one(switch(name)), "{name}");
        }
        let audit = selection.routes.withheld_audit();
        for part in [
            "HF2Q_QWEN_MOE_BATCHED: blk.0.ffn_gate_shexp.weight is Q5_K, not Q4_K or Q6_K",
            "HF2Q_QWEN_MOE_STAGED_M8: HF2Q_QWEN_MOE_BATCHED is off",
            "HF2Q_REPACKED_GGUF_M32_PROJ: the prepared kernels take Q4_K and Q6_K; these \
             weights are Q5_K",
            "HF2Q_REPACKED_GGUF_M32_HEAD: the prepared kernels take Q4_K and Q6_K; these \
             weights are Q8_0",
            "HF2Q_QWEN_PREFILL_PREPARED: the prepared kernels take Q4_K and Q6_K; these \
             weights are Q5_K",
            "HF2Q_QWEN_PREFILL_FOLD_CUE: HF2Q_QWEN_PREFILL_PREPARED is off",
        ] {
            assert!(audit.contains(part), "{part}\n{audit}");
        }
        assert_eq!(selection.memory.map(|memory| memory.plane_bytes), Some(0));
    }

    // A weight kind with some admitted storage keeps its row: the kernels
    // admit weight by weight.
    let mixed = restored(qwen38(), &["blk.0.ffn_gate", "blk.0.ffn_up"], GgmlType::Q5_K);
    let selection = decided(&mixed, &[], &Store::cached()).expect("auto");
    assert!(!selection.routes.any_withheld());

    // Conditions that fail a load when the switch is set by hand.
    let overlay = TargetFacts {
        dwq_overlay: true,
        ..qwen36()
    };
    let selection = decided(&overlay, &[], &Store::cached()).expect("auto");
    assert!(selection.selected());
    assert!(withheld(&selection, "HF2Q_QWEN_MOE_BATCHED"));
    assert!(selection.routes.withheld_audit().contains("DWQ overlay"));
    assert!(selection.routes.is_one(switch("HF2Q_REPACKED_GGUF_M32")));

    let f16_router = restored(qwen36(), &["blk.7.ffn_gate_inp."], GgmlType::F16);
    let selection = decided(&f16_router, &[], &Store::cached()).expect("auto");
    assert!(selection
        .routes
        .withheld_audit()
        .contains("blk.7.ffn_gate_inp.weight is F16, not F32"));

    let q8_experts = restored(qwen36(), &["blk.3.ffn_up_exps"], GgmlType::Q8_0);
    let selection = decided(&q8_experts, &[], &Store::cached()).expect("auto");
    assert!(selection.routes.is_one(switch("HF2Q_QWEN_MOE_BATCHED")));
    assert!(withheld(&selection, "HF2Q_QWEN_MOE_STAGED_M8"));
    assert!(withheld(&selection, "HF2Q_QWEN_MOE_LIVE_TILES"));
    assert!(selection
        .routes
        .withheld_audit()
        .contains("blk.3.ffn_up_exps.weight is Q8_0"));

    let chunk_scan = TargetFacts {
        chunk_scan_prefill: true,
        ..qwen38()
    };
    let selection = decided(&chunk_scan, &[], &Store::cached()).expect("auto");
    assert!(withheld(&selection, "HF2Q_QWEN_PREFILL_PREPARED"));
    assert!(withheld(&selection, "HF2Q_QWEN_PREFILL_FOLD_CUE"));

    // A switch set by hand is the operator's: it is never withheld, and its
    // reader refuses storage it does not admit as it always has.
    let forced = decided(
        &apex,
        &[("HF2Q_QWEN_MOE_BATCHED", "1")],
        &Store::cached(),
    )
    .expect("auto");
    assert_eq!(
        forced.routes.source(switch("HF2Q_QWEN_MOE_BATCHED")),
        SwitchSource::Env
    );
}

#[test]
fn an_explicit_switch_overrides_the_selected_profile_in_both_directions() {
    let pairs = [
        ("HF2Q_QWEN_PREFILL_PREPARED", "0"),
        ("HF2Q_QWEN_VERIFY_ATTN_TILE", "0"),
        ("HF2Q_DFLASH2_GPU_TARGET_TOKENS", "0"),
        ("HF2Q_QWEN_MOE_BATCHED", "1"),
    ];
    let selection = decided(&qwen38(), &pairs, &Store::cached()).expect("auto");
    assert!(selection.selected());
    let routes = &selection.routes;
    for (name, value) in pairs {
        assert_eq!(routes.value(switch(name)), Some(value), "{name}");
        assert_eq!(routes.source(switch(name)), SwitchSource::Env, "{name}");
    }
    let m32 = switch("HF2Q_REPACKED_GGUF_M32");
    assert!(routes.is_one(m32));
    assert_eq!(routes.source(m32), SwitchSource::Profile);
    // The cue fold cannot run without the prepared prefill the operator
    // switched off; it is left off instead of refusing to start.
    assert!(withheld(&selection, "HF2Q_QWEN_PREFILL_FOLD_CUE"));
    let audit = routes.audit();
    assert!(
        audit.contains("HF2Q_QWEN_PREFILL_PREPARED=0(env)"),
        "{audit}"
    );
    assert!(audit.contains("HF2Q_QWEN_MOE_BATCHED=1(env)"), "{audit}");
}

#[test]
fn prepared_plane_bytes_follow_the_native_layout() {
    // Q4_K: 16 bytes of codes per 32 weights and 16 of scales per 256, the
    // size of the GGUF blocks themselves (144 bytes per 256 weights).
    assert_eq!(prepared_plane_bytes(GgmlType::Q4_K, 256), 144);
    assert_eq!(
        prepared_plane_bytes(GgmlType::Q4_K, 17_408 * 5_120),
        50_135_040
    );
    // Q6_K: 24 and 20.
    assert_eq!(prepared_plane_bytes(GgmlType::Q6_K, 256), 212);
    assert_eq!(
        prepared_plane_bytes(GgmlType::Q6_K, 17_408 * 5_120),
        73_809_920
    );
    assert_eq!(
        prepared_plane_bytes(GgmlType::Q6_K, 248_320 * 5_120),
        1_052_876_800
    );
    for other in [GgmlType::F32, GgmlType::F16, GgmlType::Q8_0, GgmlType::Q5_K] {
        assert_eq!(prepared_plane_bytes(other, 1 << 20), 0);
    }

    let dense = qwen38();
    let per_block_ffn = 2 * 50_135_040 + 73_809_920;
    assert_eq!(
        plane_bytes(&dense.tensors, &[PlaneSet::DenseFfn]),
        64 * per_block_ffn
    );
    let per_block_projections = prepared_plane_bytes(GgmlType::Q6_K, 10_240 * 5_120)
        + 2 * prepared_plane_bytes(GgmlType::Q4_K, 6_144 * 5_120);
    assert_eq!(
        plane_bytes(&dense.tensors, &[PlaneSet::Projections]),
        64 * per_block_projections
    );
    assert_eq!(
        plane_bytes(&dense.tensors, &[PlaneSet::Head]),
        1_052_876_800
    );
    // The embedding, the norms and the MTP projection carry no planes.
    assert_eq!(
        plane_bytes(
            &dense.tensors,
            &[PlaneSet::DenseFfn, PlaneSet::Projections, PlaneSet::Head]
        ),
        64 * per_block_ffn + 64 * per_block_projections + 1_052_876_800
    );
    assert_eq!(plane_bytes(&dense.tensors, &[PlaneSet::MoeExperts]), 0);
    assert_eq!(plane_bytes(&dense.tensors, &[]), 0);

    // One copy of each routed expert stack: 40 blocks of two Q4_K and one
    // Q6_K stack.
    let moe = qwen36();
    assert_eq!(
        plane_bytes(&moe.tensors, &[PlaneSet::MoeExperts]),
        40 * (2 * 150_994_944 + 222_298_112)
    );
    assert_eq!(
        plane_bytes(&moe.tensors, &[PlaneSet::MoeShared]),
        40 * (2 * prepared_plane_bytes(GgmlType::Q4_K, 512 * 2_048)
            + prepared_plane_bytes(GgmlType::Q6_K, 2_048 * 512))
    );
}

#[test]
fn memory_rule_is_headroom_against_drafter_and_planes() {
    let host = |physical_gib: u64, kv_gib: u64| HostFacts {
        physical_bytes: physical_gib * GIB,
        kv_budget_bytes: kv_gib * GIB,
    };
    // headroom = 64 - 16 - 0 - 8 = 40 GiB.
    let (verdict, headroom) = memory_verdict(host(64, 0), 16 * GIB, 4 * GIB, 16 * GIB);
    assert_eq!(headroom, 40 * GIB);
    assert_eq!(verdict, MemoryVerdict::Full);
    // Exactly at the bound is admitted; one byte over is not.
    assert_eq!(
        memory_verdict(host(64, 0), 16 * GIB, 4 * GIB, 36 * GIB).0,
        MemoryVerdict::Full
    );
    assert_eq!(
        memory_verdict(host(64, 0), 16 * GIB, 4 * GIB, 36 * GIB + 1).0,
        MemoryVerdict::DropPlanes
    );
    // The KV budget comes out of the headroom: 64 - 16 - 32 - 8 = 8 GiB.
    let (verdict, headroom) = memory_verdict(host(64, 32), 16 * GIB, 4 * GIB, 16 * GIB);
    assert_eq!(headroom, 8 * GIB);
    assert_eq!(verdict, MemoryVerdict::DropPlanes);
    assert_eq!(
        memory_verdict(host(64, 32), 16 * GIB, 8 * GIB, 0).0,
        MemoryVerdict::Full
    );
    assert_eq!(
        memory_verdict(host(64, 32), 16 * GIB, 8 * GIB + 1, 0).0,
        MemoryVerdict::NoDrafter
    );
    // A machine smaller than the model has no headroom, not a wrapped one.
    let (verdict, headroom) = memory_verdict(host(16, 48), 21 * GIB, GIB, 0);
    assert_eq!(headroom, 0);
    assert_eq!(verdict, MemoryVerdict::NoDrafter);
}

#[test]
fn insufficient_memory_drops_the_planes_then_the_drafter() {
    let facts = qwen38();
    let drafter = QWEN38_DFLASH2.weights_bytes;
    let planes = plane_bytes(
        &facts.tensors,
        &[PlaneSet::DenseFfn, PlaneSet::Projections, PlaneSet::Head],
    );
    let host_with_headroom = |headroom: u64| HostFacts {
        physical_bytes: facts.model_file_bytes + MEMORY_RESERVE_BYTES + 4 * GIB + headroom,
        kv_budget_bytes: 4 * GIB,
    };
    let decide_on = |headroom: u64, pairs: &[(&str, &str)]| {
        decide(
            &facts,
            host_with_headroom(headroom),
            &env(pairs),
            &Store::cached(),
        )
    };

    let full = decide_on(drafter + planes, &[]).expect("fits");
    assert!(full.selected() && !full.routes.any_withheld());
    assert_eq!(
        full.memory,
        Some(MemoryEstimate {
            headroom_bytes: drafter + planes,
            drafter_bytes: drafter,
            plane_bytes: planes,
        })
    );

    // One byte short: DFlash2 stays, every plane-carrying profile row goes,
    // and so does a row that cannot run without one of them.
    let dropped = decide_on(drafter + planes - 1, &[]).expect("drafter fits");
    assert!(dropped.selected());
    assert_eq!(dropped.profile_label(), "partial");
    assert!(
        dropped.reason.contains("dropped from the profile"),
        "{}",
        dropped.reason
    );
    assert!(dropped.reason.contains("headroom"), "{}", dropped.reason);
    for row in SWITCHES {
        let index = switch(row.name);
        let depends_on_planes = row.requires.is_some_and(|required| {
            !SWITCHES[SWITCHES
                .iter()
                .position(|other| other.name == required)
                .expect("row")]
            .planes
            .is_empty()
        });
        if row.planes.is_empty() && !depends_on_planes {
            assert_eq!(
                dropped.routes.value(index),
                full.routes.value(index),
                "{}",
                row.name
            );
        } else {
            assert_eq!(dropped.routes.value(index), None, "{}", row.name);
        }
    }
    assert!(dropped
        .routes
        .withheld_audit()
        .contains("HF2Q_REPACKED_GGUF_M32: its prepared planes do not fit in memory"));
    assert_eq!(dropped.memory.map(|memory| memory.plane_bytes), Some(0));

    // An explicit plane switch is the operator's and stays.
    let kept = decide_on(drafter, &[("HF2Q_REPACKED_GGUF_M32", "1")]).expect("drafter fits");
    assert!(kept.routes.is_one(switch("HF2Q_REPACKED_GGUF_M32")));
    assert_eq!(
        kept.memory.map(|memory| memory.plane_bytes),
        Some(plane_bytes(&facts.tensors, &[PlaneSet::DenseFfn]))
    );

    // The drafter alone does not fit: auto keeps the prior route.
    let none = decide_on(drafter - 1, &[]).expect("auto never fails");
    assert_prior_route(&none, &[]);
    assert!(
        none.reason.contains("does not fit in memory"),
        "{}",
        none.reason
    );
    // Required keeps the drafter, without the planes, and says so.
    let required = decide_on(drafter - 1, &[(DFLASH2_ENV, "1")]).expect("required");
    assert!(required.selected());
    assert!(required.routes.any_withheld());
    assert!(required.warnings.iter().any(|w| w.contains("does not fit")));
}

#[test]
fn a_failed_load_is_retried_on_the_route_of_main() {
    let selection = decided(&qwen36(), &[], &Store::cached()).expect("auto");
    assert!(selection.falls_back_on_load_error());
    let prior = selection.without_dflash2("the DFlash2 route failed to load: test");
    assert!(!prior.selected());
    assert_eq!(prior.routes.profile(), None);
    assert_eq!(prior.descriptor.map(|d| d.id), Some(QWEN36_DFLASH2.id));
    assert_eq!(prior.target_source, selection.target_source);
    assert!(!prior.falls_back_on_load_error(), "one retry only");
}

/// Real-artifact gate, header only (no weights are read, no GPU):
/// `HF2Q_QWEN38_GGUF=… HF2Q_QWEN36_GGUF=… <test binary> pinned_geometry_is_the_gate_artifacts --ignored`.
/// `HF2Q_DFLASH2_SAME_SHAPE_GGUFS` may name further GGUFs, separated by `:`,
/// that must qualify for a descriptor by shape.
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF and HF2Q_QWEN36_GGUF"]
fn pinned_geometry_is_the_gate_artifacts() {
    let open = |path: &std::ffi::OsStr| GgufFile::open(Path::new(path)).expect("open GGUF");
    for (variable, pinned) in [
        ("HF2Q_QWEN38_GGUF", &QWEN38_TARGET_GEOMETRY),
        ("HF2Q_QWEN36_GGUF", &QWEN36_TARGET_GEOMETRY),
    ] {
        let path = std::env::var_os(variable).expect(variable);
        let geometry = read_geometry(&open(&path)).expect("geometry");
        assert_eq!(geometry.mismatch(pinned), None, "{variable}: {geometry:?}");
        assert_eq!(&geometry, pinned);
    }
    for path in std::env::var("HF2Q_DFLASH2_SAME_SHAPE_GGUFS")
        .unwrap_or_default()
        .split(':')
        .filter(|path| !path.is_empty())
    {
        let gguf = open(path.as_ref());
        let facts = TargetFacts::read(
            &gguf,
            Path::new(path),
            EngineMode::SlotAware { max_slots: 4 },
            false,
        );
        let selection = decided(&facts, &[], &Store::cached()).expect("auto");
        eprintln!(
            "[SAME_SHAPE] {path} selected={} reason={} target_source={} withheld={}",
            selection.selected(),
            selection.reason,
            selection.target_source,
            selection.routes.withheld_audit()
        );
        assert!(selection.selected(), "{path}: {}", selection.reason);
    }
}
