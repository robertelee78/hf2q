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
