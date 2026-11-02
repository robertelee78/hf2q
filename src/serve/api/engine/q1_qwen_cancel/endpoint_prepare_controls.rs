//! Hosted controls: the emitted identity digest and cache manifests are exactly
//! what `qwen_actual` recomputes from the header config; no model/device.
use super::*;
use crate::adr063_invariance::qwen_cache::selected_header_fixture;

fn work(family: Family, vocab_size: u32) -> Workload {
    Workload {
        family,
        vocab_size,
        lanes: (0..4u32)
            .map(|lane| (0..144).map(|row| 1000 + 144 * lane + row).collect())
            .collect(),
    }
}

#[test]
fn q1_qwen_endpoint_prepare_matches_actual_recipe_and_cache() {
    for (q38, family) in [(true, Family::Qwen38), (false, Family::Qwen36)] {
        let cfg = selected_header_fixture(q38);
        let work = work(family, cfg.vocab_size);
        let plan = plan(&cfg, &work, 262144, 2 * 1024 * 1024 * 1024, [0, 0]).unwrap();
        // The literal recipe `qwen_actual` checked before this cutover.
        let literal = acquisition::ConfigRecipe {
            family,
            capacity: 262144,
            slots: 1,
            output_allowance: 0,
            cache_mode: "QwenTQ8",
            route: "endpoint",
            actual_model_config_sha256: hash(format!("{:?}", &cfg).as_bytes()),
        };
        assert_eq!(plan.config_sha256, acquisition::config_digest(&literal).unwrap());
        assert_eq!(plan.config_debug_sha256, literal.actual_model_config_sha256);
        assert_eq!(plan.recipe, serde_json::to_value(&literal).unwrap());
        assert_eq!(plan.token_vectors_sha256, acquisition::tokens_digest(&work).unwrap());
        assert_eq!(plan.packets.len(), TRAJECTORIES.len());
        for (trajectory, cache, planned) in &plan.packets {
            let actual =
                qwen_checkpoint::expected_cache(&cfg, family, *trajectory, 262144, [0, 0]).unwrap();
            assert_eq!(cache, &actual);
            assert_eq!(
                *planned,
                qwen_checkpoint::planned_payload(&actual, cfg.vocab_size as usize, 2 * 1024 * 1024 * 1024)
                    .unwrap()
            );
        }
    }
}

#[test]
fn q1_qwen_endpoint_prepare_negative_controls() {
    let cfg = selected_header_fixture(false);
    let base = plan(&cfg, &work(Family::Qwen36, cfg.vocab_size), 262144, 2 << 30, [0, 0]).unwrap();
    // A context-overridden header (the continuation tool's recipe input) must
    // not reproduce the endpoint identity digest.
    let mut overridden = cfg.clone();
    overridden.max_position_embeddings = overridden.max_position_embeddings.wrapping_add(1);
    let other = plan(&overridden, &work(Family::Qwen36, cfg.vocab_size), 262144, 2 << 30, [0, 0]).unwrap();
    assert_ne!(other.config_sha256, base.config_sha256);
    // Wrong family/variant, vocabulary, capacity, Q36 MTP cursors, quota.
    assert!(plan(&cfg, &work(Family::Qwen38, cfg.vocab_size), 262144, 2 << 30, [0, 0]).is_err());
    assert!(plan(&cfg, &work(Family::Qwen36, cfg.vocab_size - 1), 262144, 2 << 30, [0, 0]).is_err());
    assert!(plan(&cfg, &work(Family::Qwen36, cfg.vocab_size), 143, 2 << 30, [0, 0]).is_err());
    assert!(plan(&cfg, &work(Family::Qwen36, cfg.vocab_size), 262144, 2 << 30, [64, 64]).is_err());
    assert!(plan(&cfg, &work(Family::Qwen36, cfg.vocab_size), 262144, 1, [0, 0]).is_err());
}

#[test]
fn q1_qwen36_suffix144_workload_is_last_window_and_refuses_shared_or_short() {
    let prefix: Vec<u32> = (0..444).collect();
    let rounds: Vec<Vec<u32>> = (0..4u32)
        .map(|round| {
            let mut tokens = prefix.clone();
            tokens.extend((0..144 + round).map(|i| 5000 + 1000 * round + i));
            tokens
        })
        .collect();
    let work = suffix144_workload(&rounds, 248320).unwrap();
    assert_eq!(work.family, Family::Qwen36);
    for (lane, tokens) in work.lanes.iter().zip(&rounds) {
        assert_eq!(lane.as_slice(), &tokens[tokens.len() - 144..]);
    }
    // Whole-prompt prefixes are identical across rounds: refused as lanes.
    let prefixes: Vec<Vec<u32>> = rounds.iter().map(|t| t[..144].to_vec()).collect();
    assert!(suffix144_workload(&prefixes, 248320).is_err());
    let mut short = rounds.clone();
    short[2].truncate(143);
    assert!(suffix144_workload(&short, 248320).is_err());
    assert!(suffix144_workload(&rounds[..3], 248320).is_err());
    assert!(suffix144_workload(&rounds, 5000).is_err());
}

fn localizer(routes: serde_json::Value) -> LocalizerConfig {
    serde_json::from_value(serde_json::json!({
        "agent": 1,
        "layers": [3, 39],
        "operations": ["embed", "attn_out", "ffn_out", "layer_out"],
        "normalize_rows": true,
        "routes": routes,
    }))
    .unwrap()
}

fn t0_t3_routes() -> serde_json::Value {
    serde_json::json!([
        {"rerun_id": "loc-r128", "prefill_chunks": [128]},
        {"rerun_id": "loc-r64", "prefill_chunks": [64, 64]},
    ])
}

#[test]
fn q1_qwen_localizer_recipe_pins_todays_stage_localizer_literal() {
    for (q38, family) in [(true, Family::Qwen38), (false, Family::Qwen36)] {
        let cfg = selected_header_fixture(q38);
        let recipe = qwen_localizer::localizer_recipe(family, 262144, &cfg);
        // The literal both localizer config checks used before this cutover.
        assert_eq!(
            serde_json::to_value(&recipe).unwrap(),
            serde_json::json!({"family": family, "capacity": 262144, "slots": 1,
                "output_allowance": 0, "cache_mode": "QwenTQ8", "route": "stage-localizer",
                "actual_model_config_sha256": hash(format!("{:?}", &cfg).as_bytes())})
        );
        let endpoint = qwen_checkpoint::endpoint_recipe(family, 262144, &cfg);
        assert_ne!(
            acquisition::config_digest(&recipe).unwrap(),
            acquisition::config_digest(&endpoint).unwrap()
        );
        let work = work(family, cfg.vocab_size);
        let plan = localizer_plan(&cfg, &work, 262144, &localizer(t0_t3_routes())).unwrap();
        assert_eq!(plan.config_sha256, acquisition::config_digest(&recipe).unwrap());
        assert_eq!(plan.recipe, serde_json::to_value(&recipe).unwrap());
    }
}

#[test]
fn q1_qwen_localizer_routes_freeze_identical_coordinates_within_cap() {
    for (q38, family) in [(true, Family::Qwen38), (false, Family::Qwen36)] {
        let cfg = selected_header_fixture(q38);
        let hidden = cfg.hidden_size as usize;
        let plan =
            localizer_plan(&cfg, &work(family, cfg.vocab_size), 262144, &localizer(t0_t3_routes()))
                .unwrap();
        assert_eq!(plan.routes.len(), 2);
        let (_, _, (t128, e128, b128)) = &plan.routes[0];
        let (_, chunks, (t64, e64, b64)) = &plan.routes[1];
        assert_eq!(chunks, &vec![64, 64]);
        assert_eq!(e128, e64);
        assert_eq!(b128, b64);
        // embed + 2 layers x 3 operations, 128 F32 rows each.
        assert_eq!(e128.len(), 7 * 128);
        assert_eq!(*b128, (7 * 128 * hidden * 4) as u64);
        assert!(*b128 <= qwen_localizer::HOST_PAYLOAD_CAP);
        let coordinates: std::collections::BTreeSet<_> =
            e128.iter().map(|e| e.at.clone()).collect();
        assert_eq!(coordinates.len(), e128.len());
        assert_eq!(e128[0].at.layer, u32::MAX);
        assert_eq!(e128[0].at.stage, crate::adr063_invariance::receipt::Stage::Input);
        assert_eq!(e128[0].shape, vec![1, hidden]);
        assert_eq!(e128[128].at.layer, 3);
        assert!(e128.iter().all(|e| e.at.agent == "agent-1" && e.at.logical_step == 0));
        assert_eq!(t128.len(), 7);
        assert_eq!(t64.len(), 14);
        for target in t128.iter().chain(t64.iter()) {
            assert!(target.source_step < 2 && target.target.normalize_rows);
            assert_eq!(target.target.at.logical_row_start, 64 * target.source_step);
            assert_eq!(target.target.at.logical_step, 0);
            assert_eq!(
                target.target.at.layer,
                target.source_layer.map_or(u32::MAX, |l| l as u32)
            );
        }
        assert!(t128.iter().all(|t| t.source_step == 0));
    }
}

#[test]
fn q1_qwen_localizer_plan_refuses_unbound_scope() {
    let cfg = selected_header_fixture(false);
    let work = work(Family::Qwen36, cfg.vocab_size);
    let refused = |value: serde_json::Value| {
        let l: LocalizerConfig = serde_json::from_value(value).unwrap();
        localizer_plan(&cfg, &work, 262144, &l).is_err()
    };
    let base = serde_json::json!({"agent": 0, "layers": [3], "operations": ["layer_out"],
        "normalize_rows": true, "routes": [{"rerun_id": "loc-r128", "prefill_chunks": [128]}]});
    assert!(!refused(base.clone()));
    let mut v = base.clone();
    v["routes"] = serde_json::json!([{"rerun_id": "a", "prefill_chunks": [128]},
        {"rerun_id": "b", "prefill_chunks": [64]}]);
    assert!(refused(v), "64- and 128-token routes cannot share expected coordinates");
    let mut v = base.clone();
    v["normalize_rows"] = serde_json::json!(false);
    v["routes"] = t0_t3_routes();
    assert!(refused(v), "unnormalized per-chunk captures differ across routes");
    let mut v = base.clone();
    v["routes"] = serde_json::json!([{"rerun_id": "a", "prefill_chunks": [128]},
        {"rerun_id": "a", "prefill_chunks": [64, 64]}]);
    assert!(refused(v));
    let mut v = base.clone();
    v["routes"][0]["rerun_id"] = serde_json::json!("../x");
    assert!(refused(v));
    let mut v = base.clone();
    v["routes"][0]["prefill_chunks"] = serde_json::json!([64, 63]);
    assert!(refused(v));
    let mut v = base.clone();
    v["agent"] = serde_json::json!(4);
    assert!(refused(v));
    for (layers, operations) in [
        (serde_json::json!([40]), serde_json::json!(["layer_out"])),
        (serde_json::json!([3, 3]), serde_json::json!(["layer_out"])),
        (serde_json::json!([]), serde_json::json!(["layer_out"])),
        (serde_json::json!([3]), serde_json::json!(["embed"])),
        (serde_json::json!([3]), serde_json::json!(["fa_q_rope"])),
        (serde_json::json!([3]), serde_json::json!(["layer_out", "layer_out"])),
        (serde_json::json!([3]), serde_json::json!([])),
    ] {
        let mut v = base.clone();
        v["layers"] = layers;
        v["operations"] = operations;
        assert!(refused(v));
    }
    let mut embed_only = base.clone();
    embed_only["layers"] = serde_json::json!([]);
    embed_only["operations"] = serde_json::json!(["embed"]);
    assert!(!refused(embed_only));
    let l: LocalizerConfig = serde_json::from_value(base.clone()).unwrap();
    assert!(localizer_plan(&cfg, &work, 143, &l).is_err());
    let mut unknown = base;
    unknown["tokens_len"] = serde_json::json!(128);
    assert!(serde_json::from_value::<LocalizerConfig>(unknown).is_err());
}
