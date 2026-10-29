//! Same-candidate AUTO semantic-tail gate, not scheduler/performance authority.
//! Two cold preparations on ONE loaded model/cache avoid invented state clones
//! and compact-snapshot restoration of unequal per-owner cursors.
use super::*;
use crate::inference::models::qwen35::{forward_gpu::shared_dense as shared, kv_cache::FullAttnKvSlot};
use crate::serve::scheduler::SlotHandle;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::{self, OpenOptions}, io::{Read, Write}, os::unix::fs::{OpenOptionsExt, PermissionsExt}, path::{Path, PathBuf}};

const PAYLOAD_LIMIT: u64 = 4 * 1024 * 1024 * 1024;
const META_LIMIT: usize = 16 * 1024 * 1024;
const WORKLOAD_SHA: &str = "0d657bc85dcb2c8ca4a7e1c7deb07b80ec8cda40f6e0559b66471d4b1a6d03a6";
const ORDER: [usize; 4] = [2, 0, 3, 1];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    schema: String, width: usize, model: PathBuf, workload: PathBuf, workload_sha256: String,
    request: PathBuf, request_sha256: String, output: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Workload { family: String, lanes: Vec<Vec<u32>>, vocab_size: usize }
#[derive(Serialize, PartialEq, Eq, Clone)]
struct Record { shared_arm: bool, key: String, bytes: usize, sha256: String, shape: Vec<usize>, file: String, source_view_shape: Vec<usize>, source_parent_bytes: usize, encoding: String }
struct Artifacts {
    output: PathBuf, bytes: u64, raw_payload:bool, records: Vec<Record>,
    reference: BTreeMap<String, Record>, mismatches: Vec<String>, shared: bool,
    boundaries: BTreeMap<String, Value>, incomparable: Vec<String>, selected: Vec<usize>, audit_skip: Option<usize>, audit_reference: &'static str, audit_checks: Vec<Value>, injection: Option<Value>,
}
fn require_retained_workload(w: &Workload) -> Result<()> {
    anyhow::ensure!(w.family == "Qwen38" && w.vocab_size == 248320 && w.lanes.len() == 4
        && w.lanes.iter().all(|l| l.len() == 144 && l.iter().all(|t| *t < 248320)), "retained four-lane workload");
    Ok(())
}
fn sha(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn read_bound(path: &Path, expected: &str, cap: u64) -> Result<Vec<u8>> {
    anyhow::ensure!(path.is_absolute() && fs::metadata(path)?.len() <= cap, "bounded absolute input");
    let mut bytes = Vec::new();
    fs::File::open(path)?.take(cap + 1).read_to_end(&mut bytes)?;
    anyhow::ensure!(bytes.len() as u64 <= cap, "input grew beyond cap");
    anyhow::ensure!(sha(&bytes) == expected, "input digest mismatch"); Ok(bytes)
}
fn active_spans(dims: &[usize], bytes: usize, owner: usize, live: usize) -> Result<Vec<std::ops::Range<usize>>> {
    anyhow::ensure!(dims.len() == 4 && dims[0] == 4 && dims.iter().all(|n| *n > 0) && owner < 4 && live > 0 && live <= dims[2], "active sequence shape/cursor");
    let rows = dims[0].checked_mul(dims[1]).and_then(|v| v.checked_mul(dims[2])).context("row product overflow")?;
    anyhow::ensure!(bytes > 0 && bytes % rows == 0, "sequence byte layout");
    let width = bytes / rows;
    (0..dims[1]).map(|head| {
        let offset = ((owner*dims[1]+head)*dims[2]).checked_mul(width).context("offset overflow")?;
        let end = live.checked_mul(width).and_then(|v| offset.checked_add(v)).context("extent overflow")?;
        anyhow::ensure!(end <= bytes, "source parent extent"); Ok(offset..end)
    }).collect()
}
fn output_preflight(path: &Path) -> Result<()> {
    anyhow::ensure!(path.is_absolute() && !path.exists(), "new absolute output required");
    let parent = path.parent().context("output parent")?;
    anyhow::ensure!(parent.canonicalize()? == parent, "canonical output parent required");
    anyhow::ensure!(!parent.ancestors().any(|p| p.join(".git").exists()), "output outside Git"); Ok(())
}
impl Artifacts {
    fn save(&mut self, key: String, shape: Vec<usize>, chunks: Vec<&[u8]>, float: bool) -> Result<()> {
        if let Some(skip) = self.audit_skip {
            if key.rsplit_once("/owner").and_then(|(_,i)|i.parse::<usize>().ok()) == Some(skip) {return Ok(());}
            let (_,suffix)=key.split_once('/').context("audit coordinate")?;
            let reference=self.records.iter().rev().find(|r|r.shared_arm==self.shared && r.key==format!("{}/{suffix}",self.audit_reference)).context("actual final shared reference")?;
            let mut hash=Sha256::new();let mut bytes=0usize;
            for chunk in chunks {hash.update(chunk);bytes+=chunk.len();}
            let digest=format!("{:x}",hash.finalize());
            anyhow::ensure!((reference.bytes,&reference.shape,&reference.sha256)==(bytes,&shape,&digest),"owner untouched by final diagnostic injection changed: {key}");
            self.audit_checks.push(json!({"key":key,"shared_arm":self.shared,"bytes_checked":bytes,"sha256":digest,"retained_reference_file":reference.file,"exact":true}));
            return Ok(());
        }
        let bytes: usize = chunks.iter().map(|x| x.len()).sum();
        self.bytes = self.bytes.checked_add(bytes as u64).context("artifact byte overflow")?;
        anyhow::ensure!(self.bytes <= if self.raw_payload{PAYLOAD_LIMIT}else{256*1024*1024*1024}, "bounded raw-or-digest scan cap");
        let file = format!("{:05}.bin", self.records.len());
        let mut out = if self.raw_payload{Some(OpenOptions::new().write(true).create_new(true).mode(0o600).open(self.output.join(&file))?)}else{None};
        let mut hash = Sha256::new();
        for chunk in chunks {
            if float {
                anyhow::ensure!(chunk.len() % 4 == 0 && chunk.chunks_exact(4).all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()), "nonfinite F32 {key}");
            }
            hash.update(chunk); if let Some(out)=out.as_mut(){out.write_all(chunk)?;}
        }
        let record = Record { shared_arm:self.shared, encoding:if float { "F32LE" } else if key.contains(".packed/") { "PackedFloatingCacheU8" } else { "Utf8SemanticJson" }.to_owned(), key: key.clone(), source_view_shape:shape.clone(), source_parent_bytes:bytes, shape, bytes, sha256: format!("{:x}", hash.finalize()), file:if self.raw_payload{file}else{String::new()} };
        if (key.starts_with("tail/") || key.starts_with("next/")) && key.rsplit_once("/owner").and_then(|(_,i)| i.parse::<usize>().ok()).is_some_and(|i| !self.selected.contains(&i)) {
            let (_, suffix) = key.split_once('/').context("phase coordinate")?;
            let before = self.reference.get(&format!("pre/{suffix}")).context("untouched pre coordinate")?;
            anyhow::ensure!((before.bytes, &before.shape, &before.sha256) == (record.bytes, &record.shape, &record.sha256), "untouched owner changed at {key}");
        }
        if self.shared && !self.incomparable.iter().any(|phase| key.starts_with(&format!("{phase}/"))) {
            let old = self.reference.get(&key).context("missing scalar coordinate")?;
            if (old.bytes, &old.shape, &old.sha256) != (record.bytes, &record.shape, &record.sha256) { self.mismatches.push(key); }
        } else if !self.shared { anyhow::ensure!(self.reference.insert(key, record.clone()).is_none(), "duplicate scalar coordinate"); }
        anyhow::ensure!(self.raw_payload||self.records.len()<10000,"bounded digest record count");
        self.records.push(record); Ok(())
    }
    fn buffer(&mut self, key: String, b: &MlxBuffer) -> Result<()> {
        self.save(key, vec![b.element_count()], vec![b.as_slice::<u8>()?], b.dtype() == mlx_native::DType::F32)?;
        if self.audit_skip.is_none() {
            let record = self.records.last_mut().context("buffer record")?;
            record.source_view_shape = b.shape().to_vec();
            record.source_parent_bytes = b.byte_len();
        }
        Ok(())
    }
    fn sequence(&mut self, phase: &str, role: &str, fa: &FullAttnKvSlot) -> Result<()> {
        anyhow::ensure!(fa.k.is_none() && fa.v.is_none() && fa.current_len.len() == 4, "selected TQ-only layout");
        let tq = fa.tq.as_ref().context("actual TQ payload missing")?;
        for (name, b) in [("key.packed", &tq.k_packed), ("key.norm", &tq.k_norms), ("value.packed", &tq.v_packed), ("value.norm", &tq.v_norms)] {
            let dims = b.shape();
            anyhow::ensure!(dims.len() == 4 && dims[0] == 4 && dims.iter().all(|n| *n > 0), "actual sequence dimensions");
            let raw = b.as_slice::<u8>()?;
            for owner in 0..4 {
                let live = fa.current_len[owner] as usize;
                let spans = active_spans(dims, raw.len(), owner, live)?;
                let parts = spans.iter().map(|r| &raw[r.clone()]).collect();
                self.save(format!("{phase}/{role}/{name}/owner{owner}"), vec![dims[1], live, dims[3]], parts, b.dtype() == mlx_native::DType::F32)?;
                if self.audit_skip.is_none() {
                    let record = self.records.last_mut().context("sequence record")?;
                    record.source_view_shape = dims.to_vec(); record.source_parent_bytes = b.byte_len();
                }
            }
        } Ok(())
    }
    fn boundary(&mut self, phase: &str, states: &[Qwen35DecodeState], kv: &HybridKvCache) -> Result<()> {
        let mut rows = Vec::new();
        for s in states { rows.push(json!({"slot":s.slot_id.0,"target":kv.sequence_len_for_slot(s.slot_id)?,
            "mtp":mtp_cursor_for_slot(kv,s.slot_id)?,"generated":s.generated_tokens,"pending":s.pending_speculation_output})); }
        let value = Value::Array(rows);
        if self.shared {
            let old = self.boundaries.get(phase).context("missing actual scalar boundary")?;
            if old != &value { self.incomparable.push(phase.to_owned()); }
            self.boundaries.insert(format!("shared/{phase}"),value);
        } else { self.boundaries.insert(phase.to_owned(),value); }
        Ok(())
    }
    fn cache(&mut self, phase: &str, kv: &HybridKvCache) -> Result<()> {
        for (i, fa) in kv.full_attn.iter().enumerate() { self.sequence(phase, &format!("fa{i}"), fa)?; }
        self.sequence(phase, "mtp", kv.mtp_slot.as_ref().context("actual active MTP slot missing")?)?;
        for (i, la) in kv.linear_attn.iter().enumerate() {
            for owner in 0..4 {
                let slot = SlotId(owner as u32);
                for (role, b) in [("conv", la.conv_bufs_for_slot(slot).0), ("recurrent", la.recurrent_bufs_for_slot(slot).0)] {
                    let raw = b.as_slice::<u8>()?;
                    anyhow::ensure!(raw.len() % 4 == 0, "current per-owner GDN extent");
                    let n = raw.len() / 4;
                    self.save(format!("{phase}/la{i}/{role}/owner{owner}"), vec![n / 4], vec![&raw[owner*n..(owner+1)*n]], true)?;
                    if self.audit_skip.is_none() {
                        let record = self.records.last_mut().context("current GDN record")?;
                        record.source_view_shape = b.shape().to_vec(); record.source_parent_bytes = b.byte_len();
                    }
                }
            }
        } Ok(())
    }
    fn states(&mut self, phase: &str, states: &[Qwen35DecodeState], kv: &HybridKvCache) -> Result<()> {
        for (i, s) in states.iter().enumerate() {
            let carry = &s.mtp.as_ref().context("actual carry absent")?.verifier_hidden;
            anyhow::ensure!(carry.element_count() == 5120 && carry.dtype() == mlx_native::DType::F32, "actual durable single-row MTP carry");
            self.buffer(format!("{phase}/carry/owner{i}"), carry)?;
            let value = json!({"owner":i,"target":kv.sequence_len_for_slot(s.slot_id)?,"mtp":mtp_cursor_for_slot(kv,s.slot_id)?,
                "step":s.step,"next_token":s.next_token,"generated":s.generated_tokens,"sampling_history":s.sampling_history,
                "decoded_text":s.decoded_text,"finish_reason":s.finish_reason,"pending":s.pending_speculation_output,
                "terminal_after_pending":s.terminal_after_pending,"history_present":s.history_lookup.is_some(),
                "grammar_present":s.grammar_runtime.is_some(),"grammar_runtime":format!("{:?}",s.grammar_runtime),
                "thinking_budget":format!("{:?}",s.thinking_budget),"action":format!("{:?}",s.next_decode_action())});
            let bytes = serde_json::to_vec(&value)?;
            self.save(format!("{phase}/semantic/owner{i}"), vec![bytes.len()], vec![&bytes], false)?;
        } Ok(())
    }
    fn flush(&self) -> Result<()> {
        let value = json!({"schema":"hf2q.s3.coherent-tail-artifacts.v1","payload_bytes":self.bytes,"records":self.records,
            "mismatches":self.mismatches,"actual_boundaries":self.boundaries,"cpu_only_refusal_and_post_injection_checks":self.audit_checks,
            "injection_after_actual_healthy_checked_copy":shared::fixture_final_injection_observed(),"final_diagnostic_injection":self.injection,"real_command_buffer_error_recovery_qualified":false,"incomparable_phases":self.incomparable,
            "auto_boundary_policy":"NoForcedBranchOrTokenAlignment; differing actual next boundary is Inconclusive","numeric_policy":"ExactInvariantGateNoTolerance", "production_cost_policy_qualified":false,"cost_api":"actual PreparedOrdinaryTick::measured_target and finish_prepared_ordinary; excludes artifact writes","request_usage":"Production tool-grammar compiler; fixed retained tokenvectors and explicit diagnostic sampling; not canonical chat rendering/thinking-policy qualification",
            "selected_diagnostic_owners":self.selected,"configured_cache_slots":4,"untouched_owner_pre_post_exact_required":true,
            "scheduler_owner_authority":false,"performance_qualified":false,"whole_body_b4":false});
        let bytes = serde_json::to_vec_pretty(&value)?;
        anyhow::ensure!(bytes.len() <= META_LIMIT, "16MiB metadata cap");
        OpenOptions::new().write(true).create_new(true).mode(0o600).open(self.output.join("manifest.json"))?.write_all(&bytes)?; Ok(())
    }
}
fn require_coherent_seed(finished: bool, mtp: bool, history: bool, greedy: bool, leaf: ordinary_action::OrdinaryLeaf, target: u32, mtp_cursor: u32, expected: u32) -> Result<()> {
    anyhow::ensure!(!finished && mtp && history && !greedy && leaf == ordinary_action::OrdinaryLeaf::CoherentMtpFullLogits
        && target == expected && mtp_cursor == expected, "real AUTO cold preparation did not expose coherent ordinary leaf/cursors");
    Ok(())
}
fn prepare(q: &mut Qwen35LoadedModel, kv: &mut HybridKvCache, lanes: &[Vec<u32>], params: &SamplingParams,
           reg: &ModelRegistration, supervisor: &EngineSupervisor) -> Result<Vec<Qwen35DecodeState>> {
    let mut result = Vec::new();
    for owner in 0..4 {
        let mut pending = Qwen35PrefillState::begin(lanes[owner][..128+owner].to_vec(), params.clone(), Some(reg), kv,
            SlotId(owner as u32), 0, None, None, None, q.hidden_size, None)?;
        let state = loop {
            match shared::checked_fixture(|| pending.advance(q, Some(reg), kv, 2048, supervisor))? {
                Qwen35PrefillAdvance::Pending { state, .. } => pending = state,
                Qwen35PrefillAdvance::Ready { state, .. } => break state,
            }
        };
        require_coherent_seed(state.finished_at_seed(), state.mtp.is_some(), state.history_lookup.is_some(), state.is_greedy,
            state.next_decode_action().ordinary, kv.sequence_len_for_slot(SlotId(owner as u32))?,
            mtp_cursor_for_slot(kv,SlotId(owner as u32))?, (128+owner) as u32)?;
        result.push(state);
    } Ok(result)
}

#[test]
#[ignore = "one selected Q38 model; reviewed runtime/resource packet required; no timing claim"]
fn qwen38_real_auto_shared_target_preserves_mtp_tail_and_next_decision() -> Result<()> {
    let mut raw = Vec::new();
    fs::File::open(std::env::var_os("HF2Q_S3_COHERENT_PARITY_PACKET").context("reviewed packet required")?)?
        .take(1024*1024+1).read_to_end(&mut raw)?;
    anyhow::ensure!(raw.len() <= 1024*1024, "packet cap");
    let p: Packet = serde_json::from_slice(&raw)?;
    anyhow::ensure!(p.schema == "hf2q.s3.coherent-tail-packet.v1" && (2..=4).contains(&p.width), "closed packet schema");
    output_preflight(&p.output)?;
    anyhow::ensure!(p.model.is_absolute() && p.model.is_file() && p.workload_sha256 == WORKLOAD_SHA, "selected retained artifact/workload required; root acquisition proof remains separate");
    let workload: Workload = serde_json::from_slice(&read_bound(&p.workload, &p.workload_sha256, 1024*1024)?)?;
    require_retained_workload(&workload)?;
    let request_raw = read_bound(&p.request, &p.request_sha256, 1024*1024)?;
    let request_value: Value = serde_json::from_slice(&request_raw)?;
    anyhow::ensure!(request_value["temperature"].as_f64() == Some(0.0) && request_value["repetition_penalty"] == json!(1.05)
        && request_value["max_completion_tokens"] == json!(8192)
        && ["seed","stop","logit_bias","response_format"].iter().all(|k| request_value.get(*k).is_none_or(Value::is_null))
        && ["top_k","min_p","frequency_penalty","presence_penalty","top_logprobs"].iter().all(|k| request_value.get(*k).is_none_or(|v| v.as_f64() == Some(0.0)))
        && request_value.get("top_p").is_none_or(|v| v.as_f64() == Some(1.0))
        && request_value.get("parallel_tool_calls").is_none_or(|v| v == &json!(false))
        && request_value.get("logprobs").is_none_or(|v| v == &json!(false)) && request_value["tools"].as_array().is_some_and(|x| !x.is_empty()), "actual tool request settings");
    let request: crate::serve::api::schema::ChatCompletionRequest = serde_json::from_slice(&request_raw)?;
    let reg = super::super::registry::find_for("Qwen3.8-27B").context("actual selected registration")?;
    let grammar = super::super::handlers::coherent_parity_tool_grammar(&request, &reg).map_err(anyhow::Error::msg)?.context("actual auto tool grammar required")?;
    let bits = crate::debug::INVESTIGATION_ENV.tq_codebook_bits;
    anyhow::ensure!(!matches!(bits, 5 | 6), "selected effective TQ8 required");
    let profile = crate::inference::models::qwen35::decode_pool::profile::selected();
    anyhow::ensure!(profile, "same selected ON24 profile required; no implicit stock substitution");
    crate::inference::models::qwen35::decode_pool::profile::prepare(true, Some(4), true)?;
    let opts = LoadOptions { model_path:p.model,tokenizer_path:None,config_path:None,dwq_overlay_path:None,kv_persist_dir:None,
        kv_persist_budget_bytes:0,glp_path:None,glp_alpha:None,kv_graft_path:None };
    let mut q = Qwen35LoadedModel::load_with_context(&opts, Some(262144))?;
    anyhow::ensure!(q.speculation.policy() == super::super::qwen35_speculation::QwenSpeculationPolicy::Auto
        && q.model.mtp.is_some() && q.tq_kv_active && q.vocab_size == 248320, "actual AUTO/MTP/TQ selected model");
    q.provision_multi_seq_kv_for_slot_aware(4)?;
    crate::inference::models::qwen35::decode_pool::profile::register_worker(&q.model as *const _ as usize)?;
    let mut kv = q.persistent_kv_cache.take().context("real four-slot cache")?;
    let table = Arc::new((0..q.vocab_size as u32).map(|id| q.tokenizer.decode(&[id], false).unwrap_or_default().into_bytes()).collect());
    let params = SamplingParams { temperature:0.0,repetition_penalty:1.05,max_tokens:8192,grammar:Some(grammar),token_bytes:Some(table),
        grammar_kind:super::super::engine::GrammarKind::ToolCallBodyAuto,tool_call_policy:ToolCallPolicy::AutoLazyGrammar,
        parallel_tool_calls:request.parallel_tool_calls.unwrap_or(false),..SamplingParams::default() };
    fs::create_dir(&p.output)?; fs::set_permissions(&p.output, fs::Permissions::from_mode(0o700))?;
    let mut artifacts = Artifacts {output:p.output,bytes:0,raw_payload:true,records:vec![],reference:BTreeMap::new(),mismatches:vec![],shared:false,boundaries:BTreeMap::new(),incomparable:vec![],selected:vec![],audit_skip:None,audit_reference:"next",audit_checks:vec![],injection:None};
    let selected = ORDER[..p.width].to_vec();
    artifacts.selected = selected.clone();
    let supervisor = EngineSupervisor::new();
    let result = (|| -> Result<()> {
        for arm in 0..2 {
            artifacts.shared = arm == 1;
            let mut states = prepare(&mut q, &mut kv, &workload.lanes, &params, &reg, &supervisor)?;
            artifacts.boundary("pre", &states, &kv)?; artifacts.cache("pre", &kv)?; artifacts.states("pre", &states, &kv)?;
            anyhow::ensure!(artifacts.mismatches.is_empty() && artifacts.incomparable.is_empty(), "cold prestate/carry mismatch BEFORE shared target");
            // Fixed diagnostic descriptors, NOT actual reply-cancellation or
            // scheduler-generation authority. Both refuse before graph.begin.
            let probe_rows = selected.iter().map(|&i| {
                let (token,position,slot)=states[i].prepare_coherent_ordinary_target(&kv)?.input();
                Ok(shared::Row{request:i as u64,handle:SlotHandle{slot_id:slot,generation:1},position:position as u32,token})
            }).collect::<Result<Vec<_>>>()?;
            for kind in ["cancelled_descriptor","stale_descriptor"] {
                let mut owners=probe_rows.iter().map(|r|shared::Owner{request:r.request,handle:r.handle,position:r.position,cancelled:false,speculative:false,vision:false}).collect::<Vec<_>>();
                if kind=="cancelled_descriptor" {owners[0].cancelled=true;} else {owners[0].handle.generation+=1;}
                let mut invocation=shared::Invocation::new();
                anyhow::ensure!(shared::run(&q.model,&mut kv,&mut invocation,&probe_rows,&owners).is_err()
                    && !invocation.is_failed_stop() && !invocation.completed(),"descriptor refusal must precede mutation");
                artifacts.audit_skip=Some(usize::MAX); artifacts.audit_reference="pre";
                artifacts.cache(kind,&kv)?; artifacts.states(kind,&states,&kv)?;
                artifacts.audit_skip=None;
            }
            let mut prepared = (0..4).map(|_| None).collect::<Vec<_>>();
            for &i in &selected {
                prepared[i] = Some(match states[i].prepare_shared_ordinary_tick(&kv)? {
                    ordinary_prepared::Preparation::Ordinary(p) if p.can_share() => p,
                    _ => anyhow::bail!("actual caller preparation did not select coherent ordinary action"),
                });
            }
            let targets = if arm == 0 {
                let mut targets = Vec::new();
                for &i in &selected {
                    let started = Instant::now();
                    let t = shared::checked_fixture(|| states[i].produce_prepared_scalar(&mut q,&mut kv,&supervisor,prepared[i].as_ref().unwrap()))?;
                    states[i].validate_prepared_ordinary(&q,&kv,prepared[i].as_ref().unwrap(),&t)?;
                    // Same actual measured service API, width one, excluding
                    // other-owner wait and artifact writes from controller cost.
                    prepared[i].as_mut().unwrap().measured_target(started.elapsed(),1)?;
                    targets.push((i,t));
                }
                targets
            } else {
                let rows = selected.iter().map(|&i| {let(token,position,slot)=prepared[i].as_ref().unwrap().input();shared::Row{request:i as u64,
                    handle:SlotHandle{slot_id:slot,generation:1},position:position as u32,token}}).collect::<Vec<_>>();
                let owners = rows.iter().map(|r| shared::Owner{request:r.request,handle:r.handle,position:r.position,cancelled:false,speculative:false,vision:false}).collect::<Vec<_>>();
                let mut invocation = shared::Invocation::new();
                let started = Instant::now();
                let out = shared::run(&q.model,&mut kv,&mut invocation,&rows,&owners)?;
                anyhow::ensure!(invocation.completed() && !out.whole_body_b4, "actual partial completion");
                let targets = selected.iter().copied().zip(out.logits.into_iter().zip(out.nextn_hidden)).map(|(i,(logits,h))|
                    (i,Qwen35DecodeState::supplied_ordinary_result(prepared[i].as_ref().unwrap(),logits,h))).collect::<Vec<_>>();
                for (i,t) in &targets { states[*i].validate_prepared_ordinary(&q,&kv,prepared[*i].as_ref().unwrap(),t)?; }
                let raw = started.elapsed();
                for &i in &selected { prepared[i].as_mut().unwrap().measured_target(raw,selected.len())?; }
                targets
            };
            // The production typed all-member validation still precedes ALL
            // completions. Artifact I/O is outside measured target/tail cost.
            for (i,t) in &targets { states[*i].validate_prepared_ordinary(&q,&kv,prepared[*i].as_ref().unwrap(),t)?; }
            for (i,t) in &targets {
                let ordinary_prepared::PreparedTargetResult::Coherent(t) = t else {anyhow::bail!("actual coherent result required")};
                let (logits, hidden) = t.fixture_parts();
                anyhow::ensure!(logits.len() == 248320, "full actual target vocabulary");
                let raw = logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                artifacts.save(format!("target/logits/owner{i}"), vec![logits.len()], vec![&raw], true)?;
                artifacts.buffer(format!("target/post_rms/owner{i}"), hidden)?;
            }
            for (i,t) in targets { let p=prepared[i].take().context("single-use preparation")?;
                shared::checked_fixture(|| states[i].finish_prepared_ordinary(&mut q,&mut kv,&supervisor,p,t))?; }
            artifacts.boundary("tail", &states, &kv)?; artifacts.cache("tail", &kv)?; artifacts.states("tail", &states, &kv)?;
            // Actual AUTO decides history/MTP/action. Never overwrite costs or
            // proposals; a differing committed boundary remains an explicit
            // retained mismatch, not force-aligned tokens or a tolerance pass.
            for &i in &selected { shared::checked_fixture(|| states[i].decode_tick(&mut q,&mut kv,&supervisor))?; }
            artifacts.boundary("next", &states, &kv)?; artifacts.cache("next", &kv)?; artifacts.states("next", &states, &kv)?;
            if arm==1 {
                anyhow::ensure!(artifacts.mismatches.is_empty() && artifacts.incomparable.is_empty(),
                    "all healthy/subsequent comparisons must pass BEFORE final diagnostic injection");
                let rows=selected.iter().map(|&i| {
                    // The final injected transaction is diagnostic, not an
                    // owning AUTO action. Use actual cursor and known verified
                    // token, without mutating/pretending to drain its queue.
                    let token=states[i].pending_speculation_output.back().or_else(||states[i].generated_tokens.last()).copied().context("verified diagnostic token")?;
                    Ok(shared::Row{request:i as u64,handle:SlotHandle{slot_id:SlotId(i as u32),generation:1},
                        position:kv.sequence_len_for_slot(SlotId(i as u32))?,token})
                }).collect::<Result<Vec<_>>>()?;
                let owners=rows.iter().map(|r|shared::Owner{request:r.request,handle:r.handle,position:r.position,cancelled:false,speculative:false,vision:false}).collect::<Vec<_>>();
                shared::fixture_arm_final_injection()?;
                let mut invocation=shared::Invocation::new();
                let error=match shared::run(&q.model,&mut kv,&mut invocation,&rows,&owners) {
                    Err(e)=>e,Ok(_)=>anyhow::bail!("final diagnostic injection did not fire"),
                };
                anyhow::ensure!(error.to_string().contains("shared target failed") && shared::fixture_final_injection_observed()
                    && invocation.is_failed_stop() && crate::inference::models::qwen35::decode_pool::profile::unhealthy(),
                    "expected sticky injected failure/quarantine, not real CB-error recovery");
                // Same failed Invocation rejects BEFORE any new operational
                // work. Existing checked scope also refuses before its closure.
                anyhow::ensure!(shared::run(&q.model,&mut kv,&mut invocation,&rows,&owners).is_err(),"failed invocation revived");
                let entered=std::cell::Cell::new(false);
                anyhow::ensure!(shared::checked_fixture(||{entered.set(true);Ok(())}).is_err() && !entered.get(),"sticky checked scope revived");
                artifacts.injection=Some(json!({"kind":"InjectedDiagnosticErrNotCommandBufferError",
                    "error_chain":format!("{error:#}"),"checked_copy_label":"s3.singleton_attention_row_copy",
                    "layer":0,"physical_row":0,"mutated_diagnostic_owner":selected[0],
                    "invocation_failed_stop":invocation.is_failed_stop(),"profile_quarantined":true,
                    "same_invocation_replay_rejected":true,"checked_scope_reentry_rejected":true,
                    "no_subsequent_gpu_work":true,"scheduler_fanout_authority":false}));
                // All GPU work has stopped. CPU-only active reads verify every
                // OTHER owner against its retained final healthy shared image.
                artifacts.audit_skip=Some(selected[0]); artifacts.audit_reference="next";
                artifacts.cache("after_final_injection",&kv)?;
                artifacts.states("after_final_injection",&states,&kv)?;
                artifacts.audit_skip=None;
                // No reset, replay, subsequent output or GPU submission after
                // this point. Quarantined roots remain process-owned.
            }
            drop(states);
            if arm==0 {for owner in 0..4 {kv.reset_for_slot(SlotId(owner))?;}}
        }
        anyhow::ensure!(artifacts.mismatches.is_empty() && artifacts.incomparable.is_empty(), "coherent tail differs or actual next AUTO boundary is Inconclusive; inspect retained artifacts"); Ok(())
    })();
    artifacts.flush()?;
    result?;
    println!("S3_COHERENT_TAIL_COMPLETE configured_owners=4 selected_width={} cold_preparations=2 full_active_cache=true actual_mtp=true next_auto_tick=true exact=true final_diagnostic_injection=true real_cb_error_recovery=false scheduler_authority=false performance_qualified=false",selected.len());
    Ok(())
}

#[test]
fn active_sequence_span_preserves_each_owner_head_and_cursor() {
    assert_eq!(active_spans(&[4,2,8,3],192,2,5).unwrap(), vec![96..111,120..135]);
    assert!(active_spans(&[4,2,8,3],192,2,9).is_err());
    assert!(active_spans(&[4,2,8,3],192,4,5).is_err());
    assert!(active_spans(&[4,2,8,3],191,2,5).is_err());
    assert!(active_spans(&[4,usize::MAX,8,3],192,2,5).is_err());
}

#[test]
fn real_cold_leaf_refuses_absent_carry_and_lagging_mtp_before_target() {
    use ordinary_action::OrdinaryLeaf::*;
    require_coherent_seed(false,true,true,false,CoherentMtpFullLogits,129,129,129).unwrap();
    assert!(require_coherent_seed(false,false,true,false,CoherentMtpFullLogits,129,129,129).is_err());
    assert!(require_coherent_seed(false,true,true,false,CoherentMtpFullLogits,129,128,129).is_err());
    assert!(require_coherent_seed(false,true,true,false,HistoryDecisionRequired,129,129,129).is_err());
    assert!(require_coherent_seed(true,true,true,false,CoherentMtpFullLogits,129,129,129).is_err());
}

#[test]
fn actual_artifact_entry_detects_packed_byte_change_and_untouched_mutation() {
    let tmp = tempfile::tempdir().unwrap();
    let mut a = Artifacts {output:tmp.path().to_owned(),bytes:0,raw_payload:true,records:vec![],reference:BTreeMap::new(),
        mismatches:vec![],shared:false,boundaries:BTreeMap::new(),incomparable:vec![],selected:vec![2,0],audit_skip:None,audit_reference:"next",audit_checks:vec![],injection:None};
    a.save("pre/fa0/key.packed/owner1".into(),vec![2],vec![&[1,2]],false).unwrap();
    a.shared = true;
    a.save("pre/fa0/key.packed/owner1".into(),vec![2],vec![&[1,3]],false).unwrap();
    assert_eq!(a.mismatches,vec!["pre/fa0/key.packed/owner1"]);
    assert_eq!(a.records.last().unwrap().encoding,"PackedFloatingCacheU8");
    assert!(a.save("tail/fa0/key.packed/owner1".into(),vec![2],vec![&[1,3]],false).is_err());
    a.bytes = PAYLOAD_LIMIT;
    assert!(a.save("too-large".into(),vec![1],vec![&[0]],false).is_err());
}

#[test]
fn final_cpu_only_audit_uses_retained_next_reference_without_new_payload() {
    let tmp=tempfile::tempdir().unwrap();
    let mut a=Artifacts{output:tmp.path().to_owned(),bytes:0,raw_payload:true,records:vec![],reference:BTreeMap::new(),
        mismatches:vec![],shared:false,boundaries:BTreeMap::new(),incomparable:vec![],selected:vec![2,0],
        audit_skip:None,audit_reference:"next",audit_checks:vec![],injection:None};
    // Exercise the actual pre -> next lifecycle for an untouched owner;
    // writing next without pre must not be treated as an accepted reference.
    a.save("pre/fa0/key.packed/owner1".into(),vec![2],vec![&[1,2]],false).unwrap();
    a.save("next/fa0/key.packed/owner1".into(),vec![2],vec![&[1,2]],false).unwrap();
    a.shared=true; a.audit_skip=Some(2);
    assert!(a.save("after_final_injection/fa0/key.packed/owner1".into(),vec![2],vec![&[1,2]],false).is_err(),
        "scalar next reference cannot substitute for missing shared next");
    assert!(a.audit_checks.is_empty());
    a.audit_skip=None;
    a.save("pre/fa0/key.packed/owner1".into(),vec![2],vec![&[1,2]],false).unwrap();
    a.save("next/fa0/key.packed/owner1".into(),vec![2],vec![&[1,2]],false).unwrap();
    let actual_shared_reference=a.records.last().unwrap().file.clone();
    let bytes=a.bytes;let records=a.records.len();
    a.audit_skip=Some(2);
    a.save("after_final_injection/fa0/key.packed/owner1".into(),vec![2],vec![&[1,2]],false).unwrap();
    assert_eq!((a.bytes,a.records.len()),(bytes,records));
    assert_eq!(a.audit_checks.len(),1);
    assert_eq!(a.audit_checks[0]["retained_reference_file"],json!(actual_shared_reference));
    assert!(a.save("after_final_injection/fa0/key.packed/owner1".into(),vec![2],vec![&[1,3]],false).is_err());
    // The only excluded owner is the actual first row already mutated before
    // the final injected Err. No retained reference or uninitialized read.
    a.save("after_final_injection/fa0/key.packed/owner2".into(),vec![2],vec![&[9,9]],false).unwrap();
    assert_eq!(a.audit_checks.len(),1);
}

#[test]
fn actual_retained_workload_label_is_case_exact() {
    let mut w=Workload{family:"Qwen38".into(),lanes:(0..4).map(|i|vec![i;144]).collect(),vocab_size:248320};
    require_retained_workload(&w).unwrap();
    w.family="qwen38".into();
    assert!(require_retained_workload(&w).is_err());
}

#[path="coherent_shared_parity_tests/target_timing.rs"]
mod target_timing;

#[path="coherent_shared_parity_tests/grouped_target_timing.rs"]
mod grouped_target_timing;
