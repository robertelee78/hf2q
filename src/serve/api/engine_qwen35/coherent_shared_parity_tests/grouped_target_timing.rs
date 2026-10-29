//! Producer-only teacher forcing; NOT complete AUTO/tail/performance acceptance.
use super::*;
use crate::inference::models::qwen35::{model::Qwen35Model,kv_cache::target_timing_requested_cache_bytes};
use shared::target_timing_hooks;
use std::time::Instant;
use std::os::unix::fs::MetadataExt;
const SELECTED:[usize;2]=[3,0];
const GIB:u64=1024*1024*1024;
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct Bound{path:PathBuf,sha256:String}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct Input{config:Bound,report:Bound}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct Admission{
 available_working_bytes:u64,process_limit_bytes:u64,runtime_floor_bytes:u64,
}
#[derive(Deserialize)]#[serde(deny_unknown_fields)]struct TimingPacket{
 schema:String,timing_profile:String,model:PathBuf,model_original:PathBuf,model_identity:Bound,model_bytes:u64,inputs:Vec<Input>,output:PathBuf,admission:Admission,
}
fn bound(b:&Bound,cap:u64)->Result<Vec<u8>>{read_bound(&b.path,&b.sha256,cap)}
fn schedule()->Vec<(bool,usize)>{
 let mut s=vec![(false,0),(false,1),(false,0),(false,1)];
 for _ in 0..5{for arm in [0,1,1,0]{s.push((true,arm));}}s
}
#[derive(Serialize)]struct Budget{cache_each:u64,caches:u64,weight_mapping:u64,mapping_padding:u64,
 pool:u64,host:u64,metadata_io:u64,stable_outputs:u64,driver_allowance:u64,free_margin:u64,named_upper:u64}
fn budget(cache:u64,weight:u64,tensors:usize,page:u64)->Result<Budget>{
 anyhow::ensure!(cache>0&&weight==16810714976&&tensors>0&&tensors<=4096&&page.is_power_of_two()&&page<=65536,"selected finite named resources");
 let caches=cache; // ONE retained-capacity cache, reused by full cold admission.
 let mapping_padding=u64::try_from(tensors)?.checked_mul(page).and_then(|x|x.checked_mul(2)).context("mapped-root padding overflow")?;
 let mut b=Budget{cache_each:cache,caches,weight_mapping:weight,mapping_padding,pool:24*GIB,host:GIB,
 metadata_io:64*1024*1024,stable_outputs:16*1024*1024,driver_allowance:2*GIB,free_margin:4*GIB,named_upper:0};
 b.named_upper=[b.caches,b.weight_mapping,b.mapping_padding,b.pool,b.host,b.metadata_io,b.stable_outputs,b.driver_allowance,b.free_margin]
   .into_iter().try_fold(0u64,|a,v|a.checked_add(v).context("named upper overflow"))?;Ok(b)
}
fn admit(b:&Budget,a:&Admission)->Result<()>{
 anyhow::ensure!(a.process_limit_bytes==96*GIB&&a.runtime_floor_bytes==4*GIB,"unchanged operating limits");
 anyhow::ensure!(b.named_upper<=a.available_working_bytes&&b.named_upper<=a.process_limit_bytes,"single cache named upper NOT admitted; no model construction or reduced-context fallback");Ok(())
}
fn suffix(v:&[u32],prefix:usize)->Result<Vec<u32>>{
 let x=v.get(prefix..prefix.checked_add(12).context("suffix index")?).context("canonical nextturn suffix extent")?;
 anyhow::ensure!(x.iter().all(|&id|id<248320),"valid frozen suffix");Ok(x.to_vec())
}
fn cold_inputs(prefixes:&[Vec<u32>],tails:&[Vec<u32>],step:usize)->Result<Vec<Vec<u32>>>{
 anyhow::ensure!(prefixes.len()==4&&tails.len()==4&&step<12,"fixed full-cold extent");
 let mut out=prefixes.to_vec();for &owner in &SELECTED{out[owner].extend_from_slice(tails[owner].get(..step).context("missing declared suffix")?);}Ok(out)
}
fn frozen_vectors(p:&TimingPacket)->Result<(Vec<Vec<u32>>,Vec<Vec<u32>>,Value)>{
 anyhow::ensure!(p.inputs.len()==4,"all4 canonical owners required");
 let(mut prefixes,mut tails,mut refs)=(Vec::new(),Vec::new(),Vec::new());
 for(owner,input)in p.inputs.iter().enumerate(){
  let raw=bound(&input.config,1024*1024)?;let report_raw=bound(&input.report,4*1024*1024)?;
  let report:Value=serde_json::from_slice(&report_raw)?;
  anyhow::ensure!(report["config_sha256"]==sha(&raw)&&report["full_prefix_reuse"]=="PassedExactSevenWireReplay"&&report["server_slots"]==4,"actual canonical exact prefix source");
  let vectors=super::super::super::handlers::target_timing_exact_vectors(&raw);
  anyhow::ensure!(vectors.len()==7,"actual seven-turn oracle");
  for turn in 0..7{
   let o=&report["observations"][turn];
   anyhow::ensure!(o["turn"]==turn&&o["prompt_tokens"]==vectors[turn].len()&&o["prompt_token_ids_sha256"]==sha(&serde_json::to_vec(&vectors[turn])?),"actual canonical vector hash/count join");
  }
  let prefix=vectors[5].clone();anyhow::ensure!((1900..=2100).contains(&prefix.len()),"fixed canonical turn5 extent");
  let tail=suffix(&vectors[6],prefix.len())?;
  refs.push(json!({"owner":owner,"turn":5,"prompt_tokens":prefix.len(),"prompt_sha256":sha(&serde_json::to_vec(&prefix)?),"suffix_rule":"turn6 vector[prompt_turn5_length..+12]; diagnostic NOT sampled output","suffix":tail,"suffix_sha256":sha(&serde_json::to_vec(&tail)?),"config_sha256":input.config.sha256,"report_sha256":input.report.sha256}));
  prefixes.push(prefix);tails.push(tail);
 }
 Ok((prefixes,tails,Value::Array(refs)))
}
fn digest_artifacts(output:&Path)->Artifacts{Artifacts{output:output.to_owned(),bytes:0,raw_payload:false,
 records:vec![],reference:BTreeMap::new(),mismatches:vec![],shared:false,boundaries:BTreeMap::new(),incomparable:vec![],selected:SELECTED.to_vec(),audit_skip:None,audit_reference:"next",audit_checks:vec![],injection:None}}
fn prepare_cold(q:&mut Qwen35LoadedModel,kv:&mut HybridKvCache,prefixes:&[Vec<u32>],params:&SamplingParams,reg:&ModelRegistration,supervisor:&EngineSupervisor)->Result<Vec<Qwen35DecodeState>>{
 let mut states=Vec::new();
 for owner in 0..4{
  let mut pending=Qwen35PrefillState::begin(prefixes[owner].clone(),params.clone(),Some(reg),kv,SlotId(owner as u32),0,None,None,None,q.hidden_size,None)?;
  let state=loop{match shared::checked_fixture(||pending.advance(q,Some(reg),kv,2048,supervisor))?{
   Qwen35PrefillAdvance::Pending{state,..}=>pending=state,Qwen35PrefillAdvance::Ready{state,..}=>break state}};
  require_coherent_seed(state.finished_at_seed(),state.mtp.is_some(),state.history_lookup.is_some(),state.is_greedy,
   state.next_decode_action().ordinary,kv.sequence_len_for_slot(SlotId(owner as u32))?,mtp_cursor_for_slot(kv,SlotId(owner as u32))?,prefixes[owner].len()as u32)?;
  states.push(state);
 }Ok(states)
}
struct Target{logits:Vec<Vec<f32>>,hidden:Vec<MlxBuffer>}
struct CpuTarget{logits:Vec<Vec<f32>>,hidden:Vec<Vec<f32>>}
impl Target{fn download(self)->Result<CpuTarget>{let hidden=self.hidden.iter().map(|b|b.as_slice::<f32>().map(|s|s.to_vec())).collect::<std::result::Result<Vec<_>,_>>()?;Ok(CpuTarget{logits:self.logits,hidden})}}
fn produce(q:&Qwen35LoadedModel,kv:&mut HybridKvCache,prefixes:&[Vec<u32>],tails:&[Vec<u32>],step:usize,shared_arm:bool)->Result<Target>{
 let rows=SELECTED.iter().map(|&owner|shared::Row{request:owner as u64,handle:SlotHandle{slot_id:SlotId(owner as u32),generation:1},position:(prefixes[owner].len()+step)as u32,token:tails[owner][step]}).collect::<Vec<_>>();
 for r in &rows{kv.validate_sequence_len_for_slot(r.handle.slot_id,r.position as usize)?;}
 if shared_arm{
  let owners=rows.iter().map(|r|shared::Owner{request:r.request,handle:r.handle,position:r.position,cancelled:false,speculative:false,vision:false}).collect::<Vec<_>>();
  let mut invocation=shared::Invocation::new();let out=shared::run(&q.model,kv,&mut invocation,&rows,&owners)?;
  anyhow::ensure!(invocation.completed()&&!out.whole_body_b4,"healthy partial graph");
  Ok(Target{logits:out.logits,hidden:out.nextn_hidden})
 }else{
  anyhow::ensure!(!shared::checked_active(),"NORMAL scalar must not acquire checked_fixture fences");
  let(mut logits,mut hidden)=(Vec::new(),Vec::new());
  for r in rows{let(v,h)=q.model.forward_gpu_with_nextn_hidden(&[r.token],&[r.position as i32;4],kv,r.handle.slot_id)?;logits.push(v);hidden.push(h);}
  Ok(Target{logits,hidden})
 }
}
fn outputs_equal(a:&CpuTarget,b:&CpuTarget)->Result<Value>{
 anyhow::ensure!(a.logits.len()==2&&b.logits.len()==2&&a.hidden.len()==2&&b.hidden.len()==2,"complete selected outputs");
 let mut values=Vec::new();for row in 0..2{
  let(x,y)=(&a.logits[row],&b.logits[row]);anyhow::ensure!(x.len()==248320&&y.len()==248320&&x.iter().chain(y).all(|v|v.is_finite()),"full finite logits");
  anyhow::ensure!(x.iter().zip(y).all(|(x,y)|x.to_bits()==y.to_bits()),"full target logits differ");
  let(x,y)=(&a.hidden[row],&b.hidden[row]);
  anyhow::ensure!(x.len()==5120&&y.len()==5120&&x.iter().chain(y).all(|v|v.is_finite())&&x.iter().zip(y).all(|(x,y)|x.to_bits()==y.to_bits()),"actual postRMS nextn differs");
  values.push(json!({"owner":SELECTED[row],"logit_values":248320,"hidden_values":5120,"exact":true}));
 }Ok(Value::Array(values))
}
// Root-address multiset, not active ping-pong field names. No ownership clone.
// reset_for_slot changes cursors/zeroes these existing roots in place; ping-pong
// may swap their names. Capture allocations are forbidden in this producer gate.
fn cache_roots(kv:&HybridKvCache)->Result<Vec<(usize,usize)>>{
 let mut roots=Vec::new();
 let mut add=|b:&MlxBuffer|{roots.push((b.contents_ptr()as usize,b.byte_len()));};
 for fa in kv.full_attn.iter().chain(kv.mtp_slot.iter()){
  anyhow::ensure!(fa.k.is_none()&&fa.v.is_none(),"selected TQ roots");let tq=fa.tq.as_ref().context("TQ roots absent")?;
  for b in [&tq.k_packed,&tq.k_norms,&tq.v_packed,&tq.v_norms]{add(b);}
 }
 for la in &kv.linear_attn{
  anyhow::ensure!(la.capture_states.is_none()&&la.conv_capture_states.is_none(),"no speculative capture allocation in producer-only gate");
  for b in [&la.conv_state,&la.conv_state_scratch,&la.recurrent,&la.recurrent_scratch]{add(b);}
 }
 roots.sort_unstable();anyhow::ensure!(roots.len()==260&&roots.windows(2).all(|w|w[0].0!=w[1].0),"17TQx4 +48GDNx4 unique roots");Ok(roots)
}
fn output_hashes(t:&CpuTarget)->Value{
 let hash=|v:&Vec<f32>|{let mut h=Sha256::new();for x in v{h.update(x.to_bits().to_le_bytes());}format!("{:x}",h.finalize())};
 json!({"logits":t.logits.iter().map(hash).collect::<Vec<_>>(),"nextn":t.hidden.iter().map(hash).collect::<Vec<_>>()})
}
fn write_meta(path:&Path,value:&Value)->Result<()>{let bytes=serde_json::to_vec_pretty(value)?;anyhow::ensure!(bytes.len()<=META_LIMIT,"bounded metadata");let prior=fs::read_dir(path.parent().context("metadata parent")?)?.try_fold(0u64,|a,e|{let n=e?.metadata()?.len();a.checked_add(n).context("metadata history overflow")})?;anyhow::ensure!(prior.checked_add(bytes.len()as u64).is_some_and(|n|n<=64*1024*1024),"64MiB total owned metadata/I/O cap");OpenOptions::new().create_new(true).write(true).mode(0o600).open(path)?.write_all(&bytes)?;Ok(())}
fn grouped_order()->Vec<(&'static str,usize,Option<u32>)>{
 let mut out=Vec::new();for layer in 0..64{
  out.push(("ffn-anchor-checked",layer,None));for owner in [3,0]{
   if layer%4!=3{out.push(("retained-delta-checked",layer,Some(owner)));}
   out.push(("retained-row-copy-checked",layer,Some(owner)));
  }}out
}
fn validate_grouped_order(events:&[target_timing_hooks::GroupedEvent])->Result<()>{
 let expected=grouped_order();anyhow::ensure!(events.len()==expected.len(),"complete grouped event set");
 for (ordinal,(e,(stage,layer,owner))) in events.iter().zip(expected).enumerate(){
  anyhow::ensure!(e.event_ordinal==ordinal&&e.stage==stage&&e.layer==Some(layer)&&e.owner==owner,
   "actual FFN-first/submission-order grouped events");
 }Ok(())
}
#[test]
#[ignore="root-owned RELEASE target-only one-cache FULL-cold-preparation fixed ABBA; resource gate BEFORE model"]
fn qwen38_normal_unchecked_vs_layer_group_target_timing()->Result<()>{
 anyhow::ensure!(!cfg!(debug_assertions),"release required");
 let path=std::env::var_os("HF2Q_S3_TARGET_TIMING_PACKET").context("reviewed timing packet required")?;
 let mut raw=Vec::new();fs::File::open(path)?.take(1024*1024+1).read_to_end(&mut raw)?;anyhow::ensure!(raw.len()<=1024*1024,"packet cap");let p:TimingPacket=serde_json::from_slice(&raw)?;
 anyhow::ensure!(p.schema=="hf2q.s3.layer-group-target-timing.v2"&&p.timing_profile==target_timing_hooks::GROUPED_PROFILE&&p.model.is_absolute()&&fs::metadata(&p.model)?.len()==p.model_bytes,"closed model packet");
 let identity:Value=serde_json::from_slice(&bound(&p.model_identity,1024*1024)?)?;
 anyhow::ensure!(identity["passed"]==true&&identity["sha256"]=="cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7"&&identity["bytes_read"]==p.model_bytes&&identity["before"]==identity["after"],"retained exact model payload identity");
 let stat=fs::metadata(&p.model)?;let original=fs::metadata(&p.model_original)?;
 anyhow::ensure!((stat.dev(),stat.ino(),stat.len(),stat.mtime(),stat.mtime_nsec())==(original.dev(),original.ino(),original.len(),original.mtime(),original.mtime_nsec()),"actual same-inode model alias");
 let stat_value=json!({"dev":stat.dev(),"ino":stat.ino(),"size":stat.len(),"mtime_ns":stat.mtime()as i128*1000000000+stat.mtime_nsec()as i128});
 anyhow::ensure!(identity["before"]==stat_value,"model stat differs from retained payload identity");
 output_preflight(&p.output)?;fs::create_dir(&p.output)?;fs::set_permissions(&p.output,fs::Permissions::from_mode(0o700))?;
 let(prefixes,tails,inputs)=frozen_vectors(&p)?;write_meta(&p.output.join("frozen-inputs.json"),&inputs)?;
 let header=mlx_native::gguf::GgufFile::open(&p.model)?;let cfg=Qwen35Model::load_config_only(&header)?;
 anyhow::ensure!(matches!(cfg.variant,crate::inference::models::qwen35::Qwen35Variant::Dense)&&cfg.hidden_size==5120&&cfg.num_hidden_layers==64&&cfg.max_position_embeddings==262144&&cfg.vocab_size==248320&&cfg.intermediate_size==Some(17408)&&cfg.mtp_num_hidden_layers==1&&cfg.layer_types.iter().filter(|k|matches!(k,crate::inference::models::qwen35::Qwen35LayerKind::LinearAttention)).count()==48,"selected actual Q38 shape");
 let bits=match crate::debug::INVESTIGATION_ENV.tq_codebook_bits{b@(5|6|8)=>b,_=>8};
 anyhow::ensure!(bits==8,"actual Qwen dispatch authority must select TQ8");
 anyhow::ensure!(!mlx_native::command_buffer_uses_unretained_references(),"retained graph mode authority before model");
 let cache=target_timing_requested_cache_bytes(&cfg)?;
 anyhow::ensure!(cache==38332792832,"actual production planner must match selected full-capacity primary+MTP root upper");
 extern "C"{fn getpagesize()->i32;}
 let page=u64::try_from(unsafe{getpagesize()})?;let resources=budget(cache,p.model_bytes,header.tensor_count(),page)?;drop(header);
 let resource_result=admit(&resources,&p.admission);
 write_meta(&p.output.join("resource-preflight.json"),&json!({"budget":resources,"admitted":resource_result.is_ok(),"physical_admission":false,"unwritten_capacity_credit":0,"full_context":262144,"configured_slots":4,"includes_MTP_cache_even_though_tail_not_run":true}))?;resource_result?;
 // ALL constructors below this line are gated by pure named upper/fresh root policy.
 anyhow::ensure!(crate::inference::models::qwen35::decode_pool::profile::selected(),"actual ON24 required");
 crate::inference::models::qwen35::decode_pool::profile::prepare(true,Some(4),true)?;
 let opts=LoadOptions{model_path:p.model.clone(),tokenizer_path:None,config_path:None,dwq_overlay_path:None,kv_persist_dir:None,kv_persist_budget_bytes:0,glp_path:None,glp_alpha:None,kv_graft_path:None};
 let mut q=Qwen35LoadedModel::load_with_context(&opts,Some(262144))?;
 anyhow::ensure!(format!("{:?}",q.model.cfg)==format!("{:?}",cfg)&&q.model.mtp.is_some()&&q.model.token_embd_native.is_some()&&q.tq_kv_active&&q.speculation.policy()==super::super::super::qwen35_speculation::QwenSpeculationPolicy::Auto,"same actual AUTO model config");
 anyhow::ensure!(q.persistent_kv_cache.is_none(),"no displaced prior cache");
 q.provision_multi_seq_kv_for_slot_aware(4)?;let mut cache=q.persistent_kv_cache.take().context("ONE actual cache")?;
 let root_capacity=cache_roots(&cache)?;
 crate::inference::models::qwen35::decode_pool::profile::register_worker(&q.model as *const _ as usize)?;
 let registration=super::super::super::registry::find_for("Qwen3.8-27B").context("selected registry")?;
 // Same actual cold-preparation grammar path as existing coherent fixture.
 let config:Value=serde_json::from_slice(&bound(&p.inputs[0].config,1024*1024)?)?;
 let request_bound=&config["turns"][5]["request"];let request:crate::serve::api::schema::ChatCompletionRequest=serde_json::from_slice(&read_bound(Path::new(request_bound["path"].as_str().context("request path")?),request_bound["sha256"].as_str().context("request hash")?,1024*1024)?)?;
 let grammar=super::super::super::handlers::coherent_parity_tool_grammar(&request,&registration).map_err(anyhow::Error::msg)?.context("actual tool grammar")?;
 let token_bytes:Arc<Vec<Vec<u8>>>=Arc::new((0..q.vocab_size as u32).map(|id|q.tokenizer.decode(&[id],false).unwrap_or_default().into_bytes()).collect());
 anyhow::ensure!(token_bytes.iter().map(Vec::len).sum::<usize>()<512*1024*1024,"host token table bound");
 let params=SamplingParams{temperature:0.,repetition_penalty:1.05,max_tokens:8192,grammar:Some(grammar),token_bytes:Some(token_bytes),grammar_kind:super::super::super::engine::GrammarKind::ToolCallBodyAuto,tool_call_policy:ToolCallPolicy::AutoLazyGrammar,..SamplingParams::default()};
 let supervisor=EngineSupervisor::new();
 // No cache clones, snapshots or replacement allocators: each arm/sample begins
 // with the same real begin(cached_tokens=0)->reset_for_slot->advance path.
 let mut pending:Vec<Vec<Option<(CpuTarget,Vec<Record>)>>>=(0..2).map(|_|(0..12).map(|_|None).collect()).collect();
 let mut steps=[0usize;2];let mut samples=Vec::new();let mut pairs=Vec::new();let mut scanned=0u64;
 for(index,(timed,arm))in schedule().into_iter().enumerate(){
  let step=steps[arm];let cold_prefixes=cold_inputs(&prefixes,&tails,step)?;
  anyhow::ensure!(cache_roots(&cache)?==root_capacity,"capacity changed BEFORE full cold preparation");
  let states=prepare_cold(&mut q,&mut cache,&cold_prefixes,&params,&registration,&supervisor)?;
  anyhow::ensure!(cache_roots(&cache)?==root_capacity,"cold reset/advance replaced primary/MTP/GDN capacity");
  let mut digest=digest_artifacts(&p.output);
  digest.cache("pre",&cache)?;
  for owner in 0..4{digest.buffer(format!("pre/carry/owner{owner}"),&states[owner].mtp.as_ref().context("cold carry")?.verifier_hidden)?;}
  let scope=if arm==1{Some(target_timing_hooks::GroupedScope::enter()?)}else{None};
  let clock=Instant::now();let result=produce(&q,&mut cache,&prefixes,&tails,step,arm==1);let elapsed=u64::try_from(clock.elapsed().as_nanos())?;
  let events=match scope{Some(s)=>s.take()?,None=>Vec::new()};
  samples.push(json!({"schema":"hf2q.s3.layer-group-target-sample.v2","timing_profile":target_timing_hooks::GROUPED_PROFILE,"chosen_event_order":"ffn_anchor_then_retained_submission_order","row_copy_submission_cost":"inside_whole_target_only_not_wait_only_event","index":index,"timed":timed,"arm":arm,"step":step,"elapsed_ns":elapsed,"producer_ok":result.is_ok(),"chosen_shared_cb_events":events,"normal_dropped_cb_gpu_times":"Unknown","unhooked_shared_stages":"Unknown:LayerEncoder/full-attention/embedding/head","sum_is_gpu_union":false}));
  write_meta(&p.output.join(format!("sample-{index:02}.json")),samples.last().unwrap())?;
  let result=match result{Ok(x)=>x,Err(error)=>{write_meta(&p.output.join("failure.json"),&json!({"index":index,"arm":arm,"step":step,"error":format!("{error:#}"),"replay":false,"qualified":false}))?;return Err(error);}};
  if arm==1{validate_grouped_order(&events)?;anyhow::ensure!(events.len()==288&&events.iter().filter(|e|e.stage=="retained-row-copy-checked").count()==128&&events.iter().filter(|e|e.stage=="ffn-anchor-checked").count()==64&&events.iter().filter(|e|e.stage=="retained-delta-checked").count()==96,"complete chosen-stage census, not whole CB coverage");}else{anyhow::ensure!(events.is_empty(),"no normal scalar profiling/fence substitution");}
  // Existing producer return has downloaded logits after healthy completion.
  // Read durable nextn outputs/cache NOW, before dropping state/full cold reset.
  let cpu=result.download()?;
  outputs_equal(&cpu,&cpu)?; // all output finite/shape checks even before paired arm.
  digest.cache("tail",&cache)?;
  for owner in 0..4{
   digest.buffer(format!("tail/carry/owner{owner}"),&states[owner].mtp.as_ref().context("cold carry retained")?.verifier_hidden)?;
   let expected=cold_prefixes[owner].len()+usize::from(SELECTED.contains(&owner));
   cache.validate_sequence_len_for_slot(SlotId(owner as u32),expected)?;
   anyhow::ensure!(mtp_cursor_for_slot(&cache,SlotId(owner as u32))?==cold_prefixes[owner].len()as u32,"producer never advances MTP");
  }
  for record in &digest.records{if record.key.starts_with("tail/")&&(record.key.contains("/mtp/")||record.key.contains("/carry/")){
   let key=record.key.replacen("tail/","pre/",1);let pre=digest.reference.get(&key).context("untouched MTP/carry reference")?;
   anyhow::ensure!((record.bytes,&record.shape,&record.sha256)==(pre.bytes,&pre.shape,&pre.sha256),"producer-only MTP/carry mutated");
  }}
  anyhow::ensure!(cache_roots(&cache)?==root_capacity,"producer replaced retained capacity");
  scanned=scanned.checked_add(digest.bytes).context("scan total overflow")?;
  write_meta(&p.output.join(format!("state-{index:02}.json")),&json!({"index":index,"arm":arm,"step":step,"records":digest.records,"scanned_bytes":digest.bytes,"raw_tensor_bytes":0,"capacity_unchanged":true,"cold_prompt_lengths":cold_prefixes.iter().map(Vec::len).collect::<Vec<_>>(),"output_hashes":output_hashes(&cpu)}))?;
  drop(states); // request-local owners released only AFTER all exact hashes.
  pending[arm][step]=Some((cpu,digest.records));steps[arm]+=1;
  if pending[0][step].is_some()&&pending[1][step].is_some(){
   let(a,ar)=pending[0][step].take().unwrap();let(b,br)=pending[1][step].take().unwrap();
   let exact=outputs_equal(&a,&b)?;
   anyhow::ensure!(ar==br,"paired FULL active pre/post/cache/carry manifests differ");
   pairs.push(json!({"step":step,"outputs":exact,"full_active_pre_and_post_records_exact":true,"records":ar.len()}));
  }
 }
 anyhow::ensure!(steps==[12,12]&&pairs.len()==12,"fixed complete paired schedule");
 let final_stat=fs::metadata(&p.model)?;anyhow::ensure!((final_stat.dev(),final_stat.ino(),final_stat.len(),final_stat.mtime(),final_stat.mtime_nsec())==(stat.dev(),stat.ino(),stat.len(),stat.mtime(),stat.mtime_nsec()),"model changed during fixture");
 let mut medians=Vec::new();for arm in 0..2{let mut ns=samples.iter().filter(|s|s["timed"]==true&&s["arm"]==arm).map(|s|s["elapsed_ns"].as_u64().context("sample ns")).collect::<Result<Vec<_>>>()?;ns.sort_unstable();anyhow::ensure!(ns.len()==10,"fixed median population");medians.push((ns[4]as f64+ns[5]as f64)/2.);}
 let report=json!({"schema":"hf2q.s3.layer-group-target-timing-result.v2","timing_profile":target_timing_hooks::GROUPED_PROFILE,"chosen_event_order":"ffn_anchor_then_retained_submission_order","row_copy_submission_cost":"inside_whole_target_only_not_wait_only_event","target_wall_median_ns_by_arm":medians,"median_population":"all10fixedtimed_samples;normal0/shared1;nofiltering","fixed_samples_per_arm":10,"warm_steps_per_arm":2,"samples":samples,"pairs":pairs,"digest_scanned_bytes":scanned,"raw_tensor_bytes_written":0,"selected":SELECTED,"configured_slots":4,"cold_preparations":24,"actual_cache_instances":1,"capacity_roots":root_capacity,"tail_executed":false,"MTP_remains_at_each_cold_prefix":true,"cost_controller_updated":false,"sampler_updated":false,"sum_CB_durations_is_not_union":true,"physical_admission":false,"performance_qualified":false,"full_auto_qualified":false});
 write_meta(&p.output.join("result.json"),&report)?;
 println!("S3_LAYER_GROUP_TARGET_TIMING_COMPLETE timed_per_arm=10 warm_per_arm=2 exact_pairs=12 caches=1 full_cold_preparations=24 raw_tensor_bytes=0 tail=false performance_qualified=false");Ok(())
}

#[test]fn fixed_schedule_and_resource_refusal(){let s=schedule();assert_eq!(s.len(),24);assert_eq!(s.iter().filter(|(t,a)|*t&&*a==0).count(),10);assert_eq!(s.iter().filter(|(t,a)|*t&&*a==1).count(),10);let b=budget(1,16810714976,1,16384).unwrap();assert!(admit(&b,&Admission{available_working_bytes:128*GIB,process_limit_bytes:96*GIB,runtime_floor_bytes:4*GIB}).is_ok());assert!(admit(&b,&Admission{available_working_bytes:0,process_limit_bytes:96*GIB,runtime_floor_bytes:4*GIB}).is_err());assert!(budget(u64::MAX,16810714976,1,16384).is_err());}
#[test]fn suffix_boundary_does_not_pad_or_reselect(){assert_eq!(suffix(&(0..30).collect::<Vec<_>>(),5).unwrap(),(5..17).collect::<Vec<_>>());assert!(suffix(&[1;10],0).is_err());assert!(suffix(&[248320;12],0).is_err());}

#[test]fn cold_inputs_rebuild_selected_owners_without_partial_restore(){let p=vec![vec![1],vec![2],vec![3],vec![4]];let t=vec![vec![5;12];4];let x=cold_inputs(&p,&t,3).unwrap();assert_eq!(x,vec![vec![1,5,5,5],vec![2],vec![3],vec![4,5,5,5]]);assert_eq!(cold_inputs(&p,&t,0).unwrap(),p);assert!(cold_inputs(&p,&t,12).is_err());assert!(cold_inputs(&p,&[vec![];0],0).is_err());}

#[test]fn actual_grouped_event_order_is_ffn_then_submission_order(){let x=grouped_order();assert_eq!(x.len(),288);assert_eq!(&x[..5],&[("ffn-anchor-checked",0,None),("retained-delta-checked",0,Some(3)),("retained-row-copy-checked",0,Some(3)),("retained-delta-checked",0,Some(0)),("retained-row-copy-checked",0,Some(0))]);assert_eq!(&x[15..18],&[("ffn-anchor-checked",3,None),("retained-row-copy-checked",3,Some(3)),("retained-row-copy-checked",3,Some(0))]);}
