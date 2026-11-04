//! Four-lane Gemma stage localizer on the ACTUAL Q1 endpoint routes.
//!
//! The packed route (`T2ActualCohort`: one 512-row prefill whose rows sit at
//! `lane*128`, then four-row batched decode) and its All128 reference
//! (`T0ColdSerial`: four serial 128-row prefills, single-row decode) both run
//! through `gemma_actual::run` with the SAME endpoint cache/logit capture and
//! routing witness as the b3 endpoint stores, plus a nested stage store at
//! `<output>/stages`. Both routes freeze this one route-agnostic manifest, so
//! the two stage stores compare coordinate for coordinate, and each endpoint
//! store is the observer-effect control against the matching b3 store.
//! Packed versus All128 is a Class B route pair: a localized divergence
//! implies no fix.
use super::{
    acquisition::ConfigRecipe,
    family::OWNED_AGENT,
    gemma_localizer::{prefill_row, PREFILL_STAGES},
    protocol::Trajectory,
    receipt::*,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub(super) const SCHEMA: &str = "hf2q.adr063.q1.cohort-stage-localizer.v1";
pub(super) const ROUTE: &str = "gemma-four-lane-stage-localizer-native-routing1511-v1";
pub(super) const MARKER: &str = "Q1_GEMMA_COHORT_STAGE_LOCALIZER_COMPLETE";
pub(super) const STAGE_PAYLOAD_CAP: u64 = 1024 * 1024 * 1024;
const PROMPT: usize = 128;
const TOKENS: usize = 144;
/// Roles the single-row decode hooks (`gpu_full_attn.rs`) and the batched
/// decode hooks (`batched_body.rs`) both emit, in source order.
pub(super) const DECODE_STAGES: [Stage; 12] = [
    Stage::Norm,
    Stage::Query,
    Stage::Key,
    Stage::RopeQuery,
    Stage::RopeKey,
    Stage::Value,
    Stage::Attention,
    Stage::FfnNorm,
    Stage::RouterLogits,
    Stage::RouterIds,
    Stage::RouterWeights,
    Stage::LayerOutput,
];
pub(super) type Targets = BTreeMap<(u64, u32, Stage), CaptureTarget>;
/// The packet's `stage_localizer` block. Its frozen record list is the
/// packet's `expected_stores["stages"]` or the bound `stage_manifest` file's
/// (plan-only form); either must equal [`manifest`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Plan {
    pub(super) rerun_id: String,
    pub(super) layers: Vec<u32>,
    pub(super) prefill_stages: Vec<Stage>,
    pub(super) decode_stages: Vec<Stage>,
    pub(super) host_payload_limit: u64,
}
/// Store names of a cohort localizer packet's `expected_stores`: the endpoint
/// store at the output root and the nested stage store.
pub(super) const ENDPOINT_STORE: &str = "endpoint";
pub(super) const STAGE_STORE: &str = "stages";
/// Bound manifest file of the plan-only packet form (the supervisor's
/// manifest-form extent): exactly these keys, stores `endpoint` and `stages`.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ManifestFile {
    pub(super) lanes: Vec<usize>,
    pub(super) host_payload_limit: u64,
    pub(super) expected_stores: BTreeMap<String, Vec<ExpectedTensor>>,
}
pub(super) const MANIFEST_FILE_BYTES: u64 = 64 * 1024 * 1024;
/// Witness-OFF All128 build-drift route label: its own composition key.
pub(super) const DRIFT_ROUTE: &str = "gemma-all128-stage-drift-witness-off-v1";
/// Localizer mode. `Cohort`: the witness-ON All128/packed pair. `DriftAll128`:
/// witness-OFF All128 prefill-only build-drift localizer; routing comes from
/// the production dispatch. This module names no witness item; the Cohort
/// witness checks live in `gemma_actual::routing` (the adapter the `540a4708`
/// backport replaces), so drift's code is witness-free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum Mode {
    #[default]
    Cohort,
    DriftAll128,
}
impl Mode {
    pub(super) fn route_label(self) -> &'static str {
        match self {
            Mode::Cohort => ROUTE,
            Mode::DriftAll128 => DRIFT_ROUTE,
        }
    }
    /// Every mode: the identity native must be the build's own `Cargo.toml`
    /// pin (`build`). Cohort additionally requires the published witness
    /// native (`gemma_actual::routing::require_published`); DriftAll128 does not.
    pub(super) fn admit_native(self, native: &str, build: &str) -> Result<(), String> {
        if native != build {
            return Err(format!("{self:?} localizer identity native {native} must be the build's own Cargo.toml pin {build}"));
        }
        Ok(())
    }
}
/// The ordinary endpoint recipe under the mode's own route label: both cohort
/// routes share the witness-ON one, so their stage stores share one comparison
/// key; the drift route is witness-OFF and keyed apart.
pub(super) fn recipe(cfg: &crate::serve::config::Gemma4Config, mode: Mode) -> ConfigRecipe<'static> {
    let mut r = super::gemma_actual::selected_recipe(cfg, mode == Mode::Cohort);
    r.route = mode.route_label();
    r
}
/// Which executed stage route a localizer trajectory is.
pub(super) fn route(t: Trajectory) -> Result<&'static str, String> {
    match t {
        Trajectory::T0ColdSerial => Ok("SerialAll128PrefillSingleRowDecode"),
        Trajectory::T2ActualCohort => Ok("PackedPrefill512OwnersBatchedDecodeRows"),
        _ => Err("cohort stage localizer runs only All128 (T0ColdSerial) or packed (T2ActualCohort)".into()),
    }
}
pub(super) fn ordered(all: &[Stage], wanted: &[Stage]) -> Result<Vec<Stage>, String> {
    let ordered: Vec<Stage> = all.iter().copied().filter(|s| wanted.contains(s)).collect();
    if ordered.len() != wanted.len() || ordered != wanted {
        return Err("localizer stages must be distinct roles in source order".into());
    }
    Ok(ordered)
}
pub(super) fn row(cfg: &crate::serve::config::Gemma4Config, layer: usize, stage: Stage) -> Result<(Scalar, Vec<usize>), String> {
    match stage {
        Stage::Norm => Ok((Scalar::F32, vec![cfg.hidden_size])),
        _ => prefill_row(cfg, layer, stage).ok_or_else(|| "unsupported Gemma stage role".into()),
    }
}
/// `(targets, expected, planned bytes)`. Every target is an owned-row
/// template; the source names each row's `(lane, position)`. `expected` is in
/// per-lane execution order: prefill layer -> stage -> lane -> row 0..128,
/// then decode position 128..144 -> layer -> stage -> lane.
pub(super) fn manifest(
    cfg: &crate::serve::config::Gemma4Config,
    layers: &[u32],
    prefill_stages: &[Stage],
    decode_stages: &[Stage],
) -> Result<(Targets, Vec<ExpectedTensor>, u64), String> {
    if layers.is_empty()
        || layers.windows(2).any(|w| w[0] >= w[1])
        || layers.iter().any(|&l| l as usize >= cfg.num_hidden_layers)
    {
        return Err("localizer layers must be ascending distinct actual Gemma layers".into());
    }
    let prefill = ordered(&PREFILL_STAGES, prefill_stages)?;
    let decode = ordered(&DECODE_STAGES, decode_stages)?;
    if prefill.is_empty() && decode.is_empty() {
        return Err("localizer needs at least one prefill or decode stage".into());
    }
    let at = |layer: u32, stage: Stage, agent: String, row: usize| Coordinate {
        logical_step: 0,
        layer,
        stage,
        agent,
        logical_row_start: row as u64,
        segment: 0,
    };
    let mut targets = Targets::new();
    let mut expected = Vec::new();
    let mut push = |layer: u32, stage: Stage, lane: usize, position: usize| -> Result<(), String> {
        let (scalar, axes) = row(cfg, layer as usize, stage)?;
        expected.push(ExpectedTensor {
            at: at(layer, stage, format!("agent-{lane}"), position),
            scalar,
            shape: std::iter::once(1).chain(axes).collect(),
        });
        Ok(())
    };
    for &layer in layers {
        for &stage in &prefill {
            targets.insert((0, layer, stage), CaptureTarget { at: at(layer, stage, OWNED_AGENT.into(), 0), normalize_rows: true });
            for lane in 0..4 {
                for position in 0..PROMPT {
                    push(layer, stage, lane, position)?;
                }
            }
        }
    }
    for position in PROMPT..TOKENS {
        for &layer in layers {
            for &stage in &decode {
                targets.insert(
                    (position as u64, layer, stage),
                    CaptureTarget { at: at(layer, stage, OWNED_AGENT.into(), position), normalize_rows: true },
                );
                for lane in 0..4 {
                    push(layer, stage, lane, position)?;
                }
            }
        }
    }
    if targets.len() > 4096 || expected.len() > 65536 {
        return Err("localizer target/record quota exceeded".into());
    }
    let bytes = expected
        .iter()
        .try_fold(0u64, |sum, e| {
            e.shape
                .iter()
                .try_fold(e.scalar.bytes() as u64, |n, &d| n.checked_mul(d as u64))
                .and_then(|n| sum.checked_add(n))
        })
        .ok_or("localizer payload overflow")?;
    if bytes > STAGE_PAYLOAD_CAP {
        return Err("localizer payload exceeds the 1 GiB stage cap".into());
    }
    Ok((targets, expected, bytes))
}
/// A packet plan is admitted only when it is exactly the source manifest.
pub(super) fn admit(
    cfg: &crate::serve::config::Gemma4Config,
    plan: &Plan,
    frozen: &[ExpectedTensor],
) -> Result<(Targets, u64), String> {
    if plan.rerun_id.is_empty()
        || plan.rerun_id.len() > 64
        || !plan.rerun_id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("file-safe localizer rerun id required".into());
    }
    let (targets, expected, bytes) = manifest(cfg, &plan.layers, &plan.prefill_stages, &plan.decode_stages)?;
    if expected != frozen {
        return Err("packet stage manifest differs from the source manifest".into());
    }
    if bytes > plan.host_payload_limit || plan.host_payload_limit > STAGE_PAYLOAD_CAP {
        return Err("stage payload exceeds the packet bound or the 1 GiB cap".into());
    }
    Ok((targets, bytes))
}

#[cfg(test)]
mod controls {
    use super::*;
    use crate::adr063_invariance::gemma_actual::controls::cfg;
    fn plan(layers: &[u32]) -> (Plan, Vec<ExpectedTensor>) {
        let cfg = cfg();
        let (_, expected, bytes) = manifest(&cfg, layers, &PREFILL_STAGES, &DECODE_STAGES).unwrap();
        (Plan {
            rerun_id: "cohort-packed".into(),
            layers: layers.to_vec(),
            prefill_stages: PREFILL_STAGES.to_vec(),
            decode_stages: DECODE_STAGES.to_vec(),
            host_payload_limit: bytes,
        }, expected)
    }
    #[test]
    fn q1_gemma_cohort_manifest_covers_four_lanes_in_execution_order() {
        let cfg = cfg();
        let (targets, expected, bytes) = manifest(&cfg, &[0, 1], &PREFILL_STAGES, &DECODE_STAGES).unwrap();
        // 2 layers x (14 prefill roles x 4 lanes x 128 rows + 16 decode steps x 12 roles x 4 lanes).
        assert_eq!(expected.len(), 2 * (14 * 4 * 128 + 16 * 12 * 4));
        assert_eq!(targets.len(), 2 * (14 + 16 * 12));
        assert!(targets.values().all(|t| t.at.agent == OWNED_AGENT && t.normalize_rows));
        assert_eq!(targets[&(0, 1, Stage::Key)].at.logical_row_start, 0);
        assert_eq!(targets[&(130, 0, Stage::Norm)].at.logical_row_start, 130);
        // Layer 0 is sliding (8 KV heads x 256), every record one row.
        assert_eq!(expected[0].at, Coordinate { logical_step: 0, layer: 0, stage: Stage::Query,
            agent: "agent-0".into(), logical_row_start: 0, segment: 0 });
        assert_eq!(expected[0].shape, vec![1, 16, 256]);
        assert_eq!(expected[128].at.agent, "agent-1");
        assert!(expected.iter().all(|e| e.shape[0] == 1));
        let unique: std::collections::BTreeSet<_> = expected.iter().map(|e| e.at.clone()).collect();
        assert_eq!(unique.len(), expected.len());
        for lane in 0..4 {
            let agent = format!("agent-{lane}");
            let rows: std::collections::BTreeSet<_> = expected.iter().filter(|e| e.at.agent == agent)
                .map(|e| e.at.logical_row_start).collect();
            assert_eq!(rows, (0..144).collect());
        }
        // Per lane, prefill precedes decode and decode is position-major.
        let lane0: Vec<_> = expected.iter().filter(|e| e.at.agent == "agent-0").collect();
        let first_decode = lane0.iter().position(|e| e.at.logical_row_start >= 128).unwrap();
        assert!(lane0[first_decode..].windows(2).all(|w| w[0].at.logical_row_start <= w[1].at.logical_row_start));
        assert!(lane0[..first_decode].iter().all(|e| e.at.logical_row_start < 128));
        // Two sliding layers, 4 lanes: prefill 512 rows x (3x[16,256] + 3x[8,256]
        // + 5x[2816] F32 + 128 logits + 2x top-8), decode 64 rows x (Norm + Q,K,
        // RopeQ,RopeK,V,Attn + FfnNorm + router + LayerOutput).
        let prefill_row = 3 * 16 * 256 * 4 + 3 * 8 * 256 * 4 + 5 * 2816 * 4 + 128 * 4 + 8 * 4 + 8 * 4;
        let decode_row = 3 * 2816 * 4 + 3 * 16 * 256 * 4 + 3 * 8 * 256 * 4 + 128 * 4 + 8 * 4 + 8 * 4;
        assert_eq!(bytes, 2 * (512 * prefill_row + 64 * decode_row) as u64);
        assert_eq!(bytes, 147_595_264);
        assert!(bytes <= STAGE_PAYLOAD_CAP);
        // Global layer 5: 2 KV heads x 512.
        let (_, global, _) = manifest(&cfg, &[5], &[Stage::Key], &[]).unwrap();
        assert_eq!(global.len(), 512);
        assert_eq!(global[0].shape, vec![1, 2, 512]);
    }
    #[test]
    fn q1_gemma_cohort_manifest_refuses_unbound_scope() {
        let cfg = cfg();
        assert!(manifest(&cfg, &[], &PREFILL_STAGES, &DECODE_STAGES).is_err());
        assert!(manifest(&cfg, &[1, 0], &PREFILL_STAGES, &DECODE_STAGES).is_err());
        assert!(manifest(&cfg, &[1, 1], &PREFILL_STAGES, &DECODE_STAGES).is_err());
        assert!(manifest(&cfg, &[30], &PREFILL_STAGES, &DECODE_STAGES).is_err());
        assert!(manifest(&cfg, &[0], &[], &[]).is_err());
        // Norm is decode-only (pf_norm_out is overwritten before the prefill terminal).
        assert!(manifest(&cfg, &[0], &[Stage::Norm], &[]).is_err());
        assert!(manifest(&cfg, &[0], &[], &[Stage::Residual]).is_err());
        assert!(manifest(&cfg, &[0], &[Stage::Key, Stage::Query], &[]).is_err());
        assert!(manifest(&cfg, &[0], &[Stage::Key, Stage::Key], &[]).is_err());
        // Eight layers of every role exceed the record quota.
        assert!(manifest(&cfg, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], &PREFILL_STAGES, &DECODE_STAGES).is_err());
    }
    #[test]
    fn q1_gemma_cohort_plan_admits_only_the_source_manifest() {
        let cfg = cfg();
        let (good, frozen) = plan(&[0, 1]);
        let (targets, bytes) = admit(&cfg, &good, &frozen).unwrap();
        assert_eq!(bytes, good.host_payload_limit);
        assert_eq!(targets.len(), 2 * (14 + 16 * 12));
        let mut swapped = frozen.clone();
        swapped.swap(0, 1);
        assert!(admit(&cfg, &good, &swapped).is_err());
        let mut relabelled = frozen.clone();
        relabelled[3].at.agent = OWNED_AGENT.into();
        assert!(admit(&cfg, &good, &relabelled).is_err());
        let mut bad = good.clone();
        bad.host_payload_limit -= 1;
        assert!(admit(&cfg, &bad, &frozen).is_err());
        let mut bad = good.clone();
        bad.host_payload_limit = STAGE_PAYLOAD_CAP + 1;
        assert!(admit(&cfg, &bad, &frozen).is_err());
        let mut bad = good.clone();
        bad.rerun_id = "../x".into();
        assert!(admit(&cfg, &bad, &frozen).is_err());
        let mut bad = good;
        bad.layers = vec![0];
        assert!(admit(&cfg, &bad, &frozen).is_err());
    }
    #[test]
    fn q1_gemma_cohort_routes_share_one_recipe_distinct_from_endpoints() {
        let cfg = cfg();
        assert_eq!(route(Trajectory::T0ColdSerial).unwrap(), "SerialAll128PrefillSingleRowDecode");
        assert_eq!(route(Trajectory::T2ActualCohort).unwrap(), "PackedPrefill512OwnersBatchedDecodeRows");
        for t in [Trajectory::T1Resume64, Trajectory::T3Chunk64, Trajectory::T6CancelRestore] {
            assert!(route(t).is_err());
        }
        let digest = |r: &ConfigRecipe<'_>| super::super::acquisition::config_digest(r).unwrap();
        let endpoint = super::super::gemma_actual::selected_recipe(&cfg, true);
        let localizer = recipe(&cfg, Mode::Cohort);
        assert_eq!((localizer.route, localizer.capacity, localizer.slots), (ROUTE, 262144, 4));
        assert_eq!(localizer.cache_mode, endpoint.cache_mode);
        assert_ne!(digest(&localizer), digest(&endpoint));
        assert_ne!(digest(&localizer), digest(&super::super::gemma_localizer::localizer_recipe(&cfg, 262144)));
    }
    #[test]
    fn q1_gemma_drift_all128_route_is_keyed_apart_and_admits_only_the_build_pin() {
        let cfg = cfg();
        let digest = |r: &ConfigRecipe<'_>| super::super::acquisition::config_digest(r).unwrap();
        assert_eq!(Mode::default(), Mode::Cohort);
        assert_ne!(DRIFT_ROUTE, ROUTE);
        assert_eq!((Mode::Cohort.route_label(), Mode::DriftAll128.route_label()), (ROUTE, DRIFT_ROUTE));
        let drift = recipe(&cfg, Mode::DriftAll128);
        assert_eq!(drift.route, "gemma-all128-stage-drift-witness-off-v1");
        assert_eq!(drift.cache_mode, recipe(&cfg, Mode::Cohort).cache_mode);
        assert_ne!(digest(&drift), digest(&recipe(&cfg, Mode::Cohort)));
        assert_ne!(digest(&drift), digest(&super::super::gemma_actual::selected_recipe(&cfg, false)));
        assert_eq!(serde_json::to_value(Mode::DriftAll128).unwrap(), serde_json::json!("DriftAll128"));
        // Drift accepts the build's own pin (live or a 0.15.9 diagnostic build), never another.
        let build = env!("HF2Q_MLX_NATIVE_VERSION");
        assert!(Mode::DriftAll128.admit_native(build, build).is_ok());
        assert!(Mode::DriftAll128.admit_native("0.15.9", "0.15.9").is_ok());
        assert!(Mode::DriftAll128.admit_native("0.15.9", "0.15.16").is_err());
        // Cohort also needs the build pin; its published-witness native check
        // is the witness-ON adapter's, which drift never calls.
        assert!(Mode::Cohort.admit_native(build, build).is_ok());
        assert!(Mode::Cohort.admit_native("0.15.9", "0.15.16").is_err());
        use super::super::gemma_actual::routing::require_published;
        assert!(require_published(super::super::gemma_routing_witness::WITNESS_NATIVE_VERSION).is_ok());
        assert!(require_published("0.15.9").is_err());
    }
    #[test]
    fn q1_gemma_owned_rows_refuse_fixed_lane_targets_and_relabelling() {
        use super::super::{family, store::Store};
        let root = std::env::temp_dir().join(format!("hf2q-q1-owned-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let root = std::fs::canonicalize(std::env::temp_dir()).unwrap().join(root.file_name().unwrap());
        let identity = Identity { family: Family::Gemma4, source_commit: "0".repeat(40), binary_sha256: "1".repeat(64),
            native_version: env!("HF2Q_MLX_NATIVE_VERSION").into(), native_checksum: "2".repeat(64),
            model_sha256: "3".repeat(64), tokenizer_sha256: "4".repeat(64), template_sha256: "5".repeat(64),
            config_sha256: "6".repeat(64), token_vectors_sha256: "7".repeat(64) };
        let at = |agent: &str, row: u64| Coordinate { logical_step: 0, layer: 0, stage: Stage::RouterIds,
            agent: agent.into(), logical_row_start: row, segment: 0 };
        let expected = (0..4).map(|lane| ExpectedTensor { at: at(&format!("agent-{lane}"), 0), scalar: Scalar::U32, shape: vec![1, 2] })
            .collect::<Vec<_>>();
        let store = Store::new(&root, identity, "HOST_ONLY_CONTROL".into(), expected, 1024, 4).unwrap();
        let mut targets = Targets::new();
        targets.insert((0, 0, Stage::RouterIds), CaptureTarget { at: at(OWNED_AGENT, 0), normalize_rows: true });
        let _scope = family::install(Family::Gemma4, store, targets).unwrap();
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device = mlx_native::MlxDevice::new().unwrap();
        let mut buf = device.alloc_buffer(32, mlx_native::DType::U32, vec![4, 2]).unwrap();
        buf.as_mut_slice::<u32>().unwrap().copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);
        // An owned template cannot capture without owners, nor with a wrong-length owner map.
        assert!(family::drained(Family::Gemma4, 0, 0, Stage::RouterIds, &buf, "HOST").is_err());
        assert!(family::drained_owned_rows(Family::Gemma4, 0, 0, Stage::RouterIds, &buf, &[(0, 0)], "HOST").is_err());
        assert!(family::drained_owned_rows(Family::Gemma4, 0, 0, Stage::RouterIds, &buf, &[(0, 0), (1, 0), (2, 0), (4, 0)], "HOST").is_err());
        family::drained_owned_rows(Family::Gemma4, 0, 0, Stage::RouterIds, &buf, &[(0, 0), (1, 0), (2, 0), (3, 0)], "HOST").unwrap();
        let trace = family::finish().unwrap();
        // Every row landed at its own lane; the earlier refusals are recorded.
        assert_eq!(trace.coverage.captured.len(), 4);
        for (lane, t) in trace.coverage.captured.iter().enumerate() {
            assert_eq!(t.at.agent, format!("agent-{lane}"));
            assert_eq!(t.parent_offset, (lane * 8) as u64);
        }
        assert_eq!(trace.coverage.errors.len(), 3);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
