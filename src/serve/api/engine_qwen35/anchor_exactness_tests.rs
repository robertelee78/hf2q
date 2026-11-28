//! ADR-063 C2 real-artifact qualification. These tests are deliberately ignored:
//! selecting one requires an independently reviewed acquisition packet and Metal.
//! A missing artifact/device/route is an error, never a successful skip.
//! Each invocation loads one artifact. The root driver owns host admission.
use std::{fs, io::Read, path::{Path, PathBuf}};
use std::os::unix::fs::PermissionsExt;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use super::{anchor_store::*, Qwen35DecodeState, Qwen35LoadedModel,
    Qwen35PrefillAdvance, Qwen35PrefillState, Qwen35SpecPrefixBoundary};
use crate::adr063_invariance::{acquisition, protocol::Workload, receipt::*, store::{hash, Store}};
use crate::inference::models::qwen35::{kv_cache::HybridKvCache, spec_decode::positions_for_range};
use crate::serve::{api::{engine::{LoadOptions, LoadedModel, SamplingParams},
    engine_supervisor::EngineSupervisor, qwen35_speculation::{QwenSpeculationController, QwenSpeculationPolicy}},
    multi_seq_kv::SlotId};

#[path = "anchor_exactness/trace.rs"]
pub(crate) mod trace;
#[path = "anchor_exactness/lifecycle.rs"]
pub(crate) mod lifecycle;
#[path = "anchor_exactness/divergent.rs"]
mod divergent;
#[path = "anchor_exactness/trace_pair.rs"]
mod trace_pair;
#[path = "anchor_exactness/state_pair.rs"]
mod state_pair;
#[path = "anchor_exactness/graft.rs"]
mod graft;
#[path = "anchor_exactness/post_commit.rs"]
mod post_commit;
#[path = "anchor_exactness/extent_export.rs"]
mod extent_export;
#[path = "anchor_exactness/transaction_trace.rs"]
mod transaction_trace;
#[path = "anchor_exactness/obligations.rs"]
mod obligations;
#[path = "anchor_exactness/state_layout.rs"]
mod state_layout;
#[path = "anchor_exactness/forward_plan.rs"]
mod forward_plan;
#[path = "anchor_exactness/matrix_plan.rs"]
mod matrix_plan;
#[path = "anchor_exactness/metadata.rs"]
mod metadata;
#[path = "anchor_exactness/registration.rs"]
pub(crate) mod registration;
#[path = "anchor_exactness/checkpoint_witness.rs"]
mod checkpoint_witness;
#[path = "anchor_exactness/registered_plan.rs"]
mod registered_plan;
#[path = "anchor_exactness/case_evidence.rs"]
mod case_evidence;
#[path = "anchor_exactness/live_ram.rs"]
mod live_ram;
#[path = "anchor_exactness/request_kind.rs"]
mod request_kind;
#[path = "anchor_exactness/matrix_catalog_export.rs"]
mod matrix_catalog_export;
#[path = "anchor_exactness/native_inputs.rs"]
pub(crate) mod native_inputs;
#[path = "anchor_exactness/frozen_params.rs"]
pub(crate) mod frozen_params;
#[path = "anchor_exactness/frozen_script.rs"]
pub(crate) mod frozen_script;
#[path = "anchor_exactness/fullw_http.rs"]
pub(crate) mod fullw_http;
#[path = "anchor_exactness/cancellation.rs"]
mod cancellation;
#[path = "anchor_exactness/original_witness.rs"]
pub(crate) mod original_witness;
#[path = "anchor_exactness/fresh_reference.rs"]
pub(crate) mod fresh_reference;
#[path = "anchor_exactness/fresh_http.rs"]
pub(crate) mod fresh_http;
#[path = "anchor_exactness/fresh_prefix.rs"]
mod fresh_prefix;
#[path = "anchor_exactness/fresh_admission.rs"]
mod fresh_admission;
#[path = "anchor_exactness/config_recipe_export.rs"]
mod config_recipe_export;
#[path = "anchor_exactness/localization_extent.rs"]
mod localization_extent;

const Q38: &str = "cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7";
const Q36: &str = "815a22cab21649948352c0eea6e9be0d9d15606e2024d86e7bdb71fb9df08220";
const SLOT: SlotId = SlotId(0);
const SLOTS: u32 = 4;
const CHUNK: usize = 16;
const DECODE_LIMIT: usize = 64;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
enum Route { Off, MtpAuto, CurrentDefault }
impl Route {
    fn label(self) -> &'static str { match self {
        Self::Off => "C2-SlotAware4-OFF", Self::MtpAuto => "C2-SlotAware4-AUTO-K3",
        Self::CurrentDefault => "C2-SlotAware4-current-default",
    }}
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    execution_approved: bool,
    acquisition: acquisition::Binding,
    model: PathBuf,
    output: PathBuf,
    identity: Identity,
    workload: Workload,
    route: Route,
    graft: Option<PathBuf>,
    changed_graft: Option<PathBuf>,
    capacity: u32,
    registration: Option<acquisition::FileBinding>,
    host_payload_limit: u64,
}
struct Fixture {
    loaded: Box<LoadedModel>,
    packet: Packet,
    acquisition: acquisition::Scope,
    supervisor: EngineSupervisor,
    restore_route: crate::inference::models::qwen35::kv_cache::LaRestoreRoute,
    _routes: crate::inference::spec_decode::dflash2::route_profile::Bound,
    joint: Option<crate::serve::api::engine::qwen35_joint_decode::Qwen35JointDecodeConfig>,
    matrix: Option<std::rc::Rc<std::cell::RefCell<matrix_plan::Budget>>>,
    _metadata_ram: Option<live_ram::Reservation>,
    _ram_scope: Option<live_ram::Bound>,
}
impl Fixture {
    fn load(family: Family, mode: Option<matrix_plan::Mode>) -> Result<Self> {
        let path = PathBuf::from(std::env::var_os("HF2Q_C2_EXACTNESS_PACKET")
            .context("HF2Q_C2_EXACTNESS_PACKET must name the root-reviewed packet")?);
        let mut raw = Vec::new();
        fs::File::open(path)?.take(4 * 1024 * 1024 + 1).read_to_end(&mut raw)?;
        ensure!(raw.len() <= 4 * 1024 * 1024, "C2 packet exceeds4MiB");
        let packet: Packet = serde_json::from_slice(&raw)?;
        ensure!(packet.execution_approved, "root execution approval missing");
        let matrix = match (mode, packet.registration.as_ref()) {
            (Some(mode), Some(binding)) => {
                if mode == matrix_plan::Mode::Lifecycle && packet.graft.is_some() {
                    ensure!(packet.changed_graft.is_some(), "grafted matrix requires its real changed-identity bank");
                }
                Some(std::rc::Rc::new(std::cell::RefCell::new(matrix_plan::Budget::new(
                    matrix_plan::Manifest::load(binding, &packet.identity, mode, packet.graft.is_some())?))))
            }
            (None, None) => None,
            _ => anyhow::bail!("selected matrix endpoint and pre-load registration manifest differ"),
        };
        packet.workload.validate().map_err(anyhow::Error::msg)?;
        ensure!(packet.identity.family == family && packet.workload.family == family,
            "selected test and packet family differ");
        let digest = match family { Family::Qwen38 => Q38, Family::Qwen36 => Q36,
            _ => anyhow::bail!("C2 only admits the two original Qwen artifacts") };
        ensure!(packet.identity.model_sha256 == digest, "not the original vanilla C2 artifact");
        ensure!(packet.model.is_absolute() && packet.model.is_file(), "required artifact unavailable");
        ensure!(packet.capacity >= 512 && packet.capacity <= 262144, "explicit bounded capacity required");
        ensure!(packet.host_payload_limit > 0 && packet.host_payload_limit <= 16 * 1024 * 1024 * 1024,
            "finite diagnostic payload grant required");
        output_preflight(&packet.output)?;
        let acquisition = acquisition::Scope::begin(&packet.acquisition, &packet.identity,
            &packet.workload, &packet.model, &packet.output, &raw)?;
        use crate::serve::dflash2_policy;
        let requested = std::env::var("HF2Q_DFLASH2").ok();
        let policy = dflash2_policy::parse_policy(requested.as_deref()).map_err(anyhow::Error::msg)?;
        if packet.route != Route::CurrentDefault {
            ensure!(policy == dflash2_policy::PolicyRequest::Off,
                "OFF/original-K3 qualification requires explicit HF2Q_DFLASH2=0; speculation policy alone does not disable it");
        } else {
            ensure!(policy != dflash2_policy::PolicyRequest::Off, "current-default packet disabled DFlash2");
            for name in ["HF2Q_DFLASH2_CONFIG", "HF2Q_DFLASH2_WEIGHTS"] {
                let path = PathBuf::from(std::env::var_os(name)
                    .with_context(|| format!("selected default diagnostic requires existing {name}; no implicit download"))?);
                ensure!(path.is_absolute() && path.is_file(), "required default artifact unavailable: {name}");
            }
        }
        let config = crate::serve::multi_model::EngineConfig {
            engine_mode: crate::serve::api::engine::EngineMode::SlotAware { max_slots: SLOTS },
            requested_context: crate::serve::operator_settings::requested_context(Some(packet.capacity), None)
                .map_err(anyhow::Error::msg)?,
            kv_graft_path: packet.graft.clone(),
            ..Default::default()
        };
        let selection = dflash2_policy::resolve_for_load(&packet.model, &config, 0)?;
        if packet.route == Route::CurrentDefault {
            ensure!(selection.plan.is_some(), "selected default route unavailable: {}", selection.reason);
        } else { ensure!(selection.plan.is_none(), "OFF/K3 packet selected a drafter"); }
        let routes = crate::inference::spec_decode::dflash2::route_profile::bind(selection.routes.clone());
        let options = load_options(&packet.model, packet.graft.clone());
        let mut loaded = Box::new(LoadedModel::load(&options)?);
        let LoadedModel::Qwen35(qwen) = loaded.as_mut() else { anyhow::bail!("not a Qwen model") };
        ensure!(qwen.model.mtp.is_some(), "original artifact must carry native MTP weights");
        ensure!(qwen.vocab_size == packet.workload.vocab_size as usize, "workload vocabulary mismatch");
        ensure!(qwen.tq_kv_active, "C2 selected hardware matrix requires production TQ cache");
        qwen.dflash2_plan = selection.plan.clone();
        qwen.provision_dflash2(SLOTS as usize)?;
        let restore_value = std::env::var("HF2Q_QWEN_LA_GPU_RESTORE").ok();
        let restore_route = crate::inference::models::qwen35::kv_cache::LaRestoreRoute::resolve(
            restore_value.as_deref(), qwen.model.cfg.variant)?;
        match packet.route {
            Route::Off => {
                ensure!(qwen.dflash2.is_none(),
                    "OFF packet loaded DFlash2; explicitly configure a genuine OFF arm, do not relabel it");
                qwen.speculation = QwenSpeculationController::new(QwenSpeculationPolicy::Off);
            }
            Route::MtpAuto => {
                ensure!(qwen.dflash2.is_none(),
                    "original K3 arm is distinct from DFlash2 default; configure it explicitly");
                qwen.speculation = QwenSpeculationController::new(QwenSpeculationPolicy::Auto);
            }
            Route::CurrentDefault => {
                ensure!(qwen.dflash2.is_some(), "current-default arm requires its actual draft runtime/artifact");
            }
        }
        let joint = crate::serve::api::engine::qwen35_joint_decode::qwen35_joint_decode_config(&qwen.model, SLOTS as usize);
        qwen.speculation.resolve_for_worker(SLOTS as usize, qwen.model.cfg.moe.is_some(), joint.is_some());
        let joint = joint.map(|config| crate::serve::api::engine::qwen35_joint_decode::Qwen35JointDecodeConfig {
            joint_mtp: qwen.speculation.select_joint_mtp(config.joint_mtp), ..config
        });
        qwen.model.ensure_gpu_cache_primed()?;
        let config_sha = hash(&serde_json::to_vec(&qwen.model.cfg)?);
        let recipe_route = format!("{};anchor_restore={}", packet.route.label(), restore_route.label());
        acquisition.check_config(&packet.identity, &acquisition::ConfigRecipe {
            family, capacity: packet.capacity as u64, slots: SLOTS,
            output_allowance: DECODE_LIMIT as u64, cache_mode: "TQ",
            route: &recipe_route, actual_model_config_sha256: config_sha,
        })?;
        fs::create_dir(&packet.output)?;
        fs::set_permissions(&packet.output, fs::Permissions::from_mode(0o700))?;
        fs::write(packet.output.join("resolved-route.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"hf2q.c2.actual-route.v1","declared":packet.route,
            "production_policy_reason":selection.reason,"production_default_selected":selection.plan.is_some(),
            "anchor_restore":restore_route.label(),"source":"production resolve_for_load + provision_dflash2",
            "diagnostic_not_timing_proof":true
        }))?)?;
        let mut fixture = Self { loaded, packet, acquisition, supervisor: EngineSupervisor::new(),
            restore_route, joint, _routes: routes, matrix, _metadata_ram: None, _ram_scope: None };
        if let Some(budget) = &fixture.matrix {
            fixture._metadata_ram = Some(live_ram::preflight(&fixture)?);
            fixture._ram_scope = Some(live_ram::bind(budget)?);
        }
        Ok(fixture)
    }
    fn qwen(&mut self) -> &mut Qwen35LoadedModel {
        let LoadedModel::Qwen35(qwen) = self.loaded.as_mut() else { unreachable!() }; qwen
    }
    fn cache(&self) -> Result<HybridKvCache> {
        let LoadedModel::Qwen35(qwen) = self.loaded.as_ref() else { unreachable!() };
        qwen.model.with_gpu_cache_mut(|device, _| {
            let mut cache = HybridKvCache::new_with_options(
                &qwen.model.cfg, device, self.packet.capacity, SLOTS, qwen.tq_kv_active)?;
            cache.set_la_restore_route(self.restore_route);
            Ok(cache)
        })
    }
    fn observed(&self, root: &Path) -> Result<acquisition::Acquisition> {
        let LoadedModel::Qwen35(qwen) = self.loaded.as_ref() else { unreachable!() };
        qwen.model.with_gpu_cache_mut(|device, _| self.acquisition.observed(
            device, &qwen.model as *const _ as usize as u64, 1))?.for_store_root(root)
    }
    fn matrix_budget(&self) -> Result<std::rc::Rc<std::cell::RefCell<matrix_plan::Budget>>> {
        self.matrix.as_ref().cloned().context("required C2 matrix registration is absent")
    }
    fn selected(&self, name: &str) -> Result<bool> {
        Ok(self.matrix_budget()?.borrow().manifest.selected(name))
    }
    fn completed_case(&self, name: &str) -> Result<()> {
        self.matrix_budget()?.borrow_mut().completed_case(name)
    }
    fn finish(&self) -> Result<()> {
        self.acquisition.check_unchanged()?;
        if let Some(matrix) = &self.matrix {
            let mut budget = matrix.borrow_mut();
            budget.finish()?;
            let full = matrix_plan::catalog(budget.manifest.mode, self.packet.graft.is_some());
            let result = serde_json::to_vec_pretty(&serde_json::json!({
                "schema":"hf2q.c2.matrix-shard.v1","identity":self.packet.identity,
                "acquisition":self.observed(&self.packet.output)?,"mode":budget.manifest.mode,
                "full_case_catalog":full,"catalog_sha256":budget.manifest.catalog_sha256,
                "selected_complete_cases":budget.manifest.cases,
                "all_selected_cases_completed":true,
                "whole_catalog_in_this_scope":budget.manifest.cases.len()==full.len(),
                "route":self.packet.route,"grants":budget.manifest.grants,"spent_before_index":budget.spent,
                "artifacts":budget.artifacts(),
                "independent_full_matrix_qualified":false,"timing_quality_release_qualified":false,
            }))?;
            budget.reserve_metadata(u64::try_from(result.len())?)?;
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new().create_new(true).write(true).mode(0o600)
                .open(self.packet.output.join("matrix-shard.json"))?.write_all(&result)?;
        }
        Ok(())
    }
}
fn load_options(model: &Path, graft: Option<PathBuf>) -> LoadOptions {
    LoadOptions { model_path: model.into(), tokenizer_path: None, config_path: None,
        dwq_overlay_path: None, kv_persist_dir: None, kv_persist_budget_bytes: 0,
        glp_path: None, glp_alpha: None, kv_graft_path: graft,
        lcp_resume_budget: crate::serve::kv_persist::lcp_registry::LcpResumeBudget::DEFAULT }
}
fn output_preflight(path: &Path) -> Result<()> {
    ensure!(path.is_absolute(), "absolute output path required");
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        _ => anyhow::bail!("output already exists or is not checkable"),
    }
    let parent = path.parent().context("output parent")?;
    ensure!(parent.canonicalize()? == parent && !parent.ancestors().any(|p| p.join(".git").exists()),
        "output must have a canonical parent outside Git");
    Ok(())
}

#[test]
#[ignore = "root-owned original Qwen3.8 artifact, reviewed C2 packet and exclusive Metal lease"]
fn qwen38_c2_anchor_real_lifecycle_matrix() { lifecycle::run(Family::Qwen38).unwrap(); }
#[test]
#[ignore = "root-owned original Qwen3.6 artifact, reviewed C2 packet and exclusive Metal lease"]
fn qwen36_c2_anchor_real_lifecycle_matrix() { lifecycle::run(Family::Qwen36).unwrap(); }
#[test]
#[ignore = "root-owned original Qwen3.8 artifact, reviewed C2 packet and exclusive Metal lease"]
fn qwen38_c2_anchor_o1_high_cursor_and_negative_control() { lifecycle::o1(Family::Qwen38).unwrap(); }
#[test]
#[ignore = "root-owned original Qwen3.6 artifact, reviewed C2 packet and exclusive Metal lease"]
fn qwen36_c2_anchor_o1_high_cursor_and_negative_control() { lifecycle::o1(Family::Qwen36).unwrap(); }
#[test]
#[ignore = "root-owned original Qwen3.8 artifact, reviewed frozen C2 input/extent packet and Metal"]
fn qwen38_c2_divergent_all_rows() { divergent::run(Family::Qwen38).unwrap(); }
#[test]
#[ignore = "root-owned original Qwen3.6 artifact, reviewed frozen C2 input/extent packet and Metal"]
fn qwen36_c2_divergent_all_rows() { divergent::run(Family::Qwen36).unwrap(); }

#[test]
#[ignore = "root-owned Qwen3.8 artifact, reviewed C2 packet and actual GPU anchor restore route"]
fn qwen38_c2_anchor_post_commit_failure() { post_commit::run(Family::Qwen38).unwrap(); }
#[test]
#[ignore = "root-owned Qwen3.6 artifact, reviewed C2 packet and actual GPU anchor restore route"]
fn qwen36_c2_anchor_post_commit_failure() { post_commit::run(Family::Qwen36).unwrap(); }

#[test]
#[ignore = "root-owned real production HTTP server; full original four-lane corpus and full262144 slots"]
fn qwen38_c2_full_original_http_capture_replay() { fullw_http::run(Family::Qwen38).unwrap(); }
#[test]
#[ignore = "root-owned real production HTTP server; full original four-lane corpus and full262144 slots"]
fn qwen36_c2_full_original_http_capture_replay() { fullw_http::run(Family::Qwen36).unwrap(); }

#[test]
#[ignore = "full original frozen-prefix All diagnostic; real artifact/full262144×4 cache"]
fn qwen38_c2_full_original_prefix_all() { divergent::prefix(Family::Qwen38).unwrap(); }
#[test]
#[ignore = "full original frozen-prefix All diagnostic; real artifact/full262144×4 cache"]
fn qwen36_c2_full_original_prefix_all() { divergent::prefix(Family::Qwen36).unwrap(); }

#[test]
#[ignore = "root-owned fresh fixed-input numerical acquisition/replay; full262144x4 real HTTP"]
fn qwen38_c2_fresh_fixed_input_http() { fresh_http::run(Family::Qwen38).unwrap(); }
#[test]
#[ignore = "root-owned fresh fixed-input numerical acquisition/replay; full262144x4 real HTTP"]
fn qwen36_c2_fresh_fixed_input_http() { fresh_http::run(Family::Qwen36).unwrap(); }
#[test]
#[ignore = "offline independent fresh fixed-input reference registration; no model/device"]
fn qwen_c2_register_fresh_fixed_input_reference() { fresh_reference::register().unwrap(); }

#[test]
#[ignore = "offline independent registered fresh header/tape extents; no model/device/inference"]
fn qwen_c2_fresh_fixed_input_extent_export() { fresh_prefix::extents().unwrap(); }
#[test]
#[ignore = "independently admitted complete registered fresh prefix-All; not serving/default proof"]
fn qwen38_c2_fresh_fixed_input_prefix_all() { fresh_prefix::prefix(Family::Qwen38).unwrap(); }
#[test]
#[ignore = "independently admitted complete registered fresh prefix-All; not serving/default proof"]
fn qwen36_c2_fresh_fixed_input_prefix_all() { fresh_prefix::prefix(Family::Qwen36).unwrap(); }
