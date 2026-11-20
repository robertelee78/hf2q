//! Real production router/handler/engine capture and canonical replay.
//! This is an ignored diagnostic, never a unary/SSE quality or timing waiver.
use super::*;
use std::{sync::{Mutex,LazyLock,Arc,atomic::{AtomicBool,Ordering}},time::{Duration,Instant}};
use crate::serve::api::schema::{ChatCompletionRequest,ChatMessage,Tool};
use super::{frozen_script::{Corpus,Script,Validated,Prepared,Terminal,OriginalCorpus,OriginalWitness},frozen_params::{TokenTable,FrozenParams}};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HttpPacket {
    execution_approved:bool, family:Family, identity:Identity, acquisition:acquisition::Binding,
    model:PathBuf, output:PathBuf, port:u16, server_argv:Vec<String>, corpus:acquisition::FileBinding,
    frozen_script:Option<acquisition::FileBinding>, selected_cell:Option<[usize;2]>,
    arm:String, keep:usize, route:Route, expected:Option<acquisition::FileBinding>,
    payload_limit:u64, ready_timeout_seconds:u64,
}
struct Row { logits:Vec<f32>, hidden:Vec<f32>, route:String }
struct Request {
    lane:usize,turn:usize,sha:String,messages:Vec<ChatMessage>,model:String,tools:Vec<Tool>,
    original:Arc<OriginalWitness>,stream:bool,
    prepared:Option<Prepared>,canonical:Option<Terminal>,terminal:Option<Terminal>,
    slot:Option<u32>,origin:usize,cached:usize,pending:std::collections::BTreeMap<usize,Row>,
    decisions:Vec<u32>,raw_decisions:Vec<u32>,gpu_argmax:Vec<(usize,u32)>,gpu_selector:bool,
    rows:usize,routes:Vec<String>,
}
struct Run {
    packet:HttpPacket, table:Option<TokenTable>,requests:Vec<Request>,
    architecture:String,worker:Option<acquisition::c2::C2WorkerBinding>,
    writer:Option<Store>,error:Option<String>,
}
static RUN:LazyLock<Mutex<Option<Run>>>=LazyLock::new(||Mutex::new(None));
fn run_lock()->std::sync::MutexGuard<'static,Option<Run>> { RUN.lock().unwrap_or_else(|e|e.into_inner()) }
pub(crate) fn historical_enabled()->bool {run_lock().is_some()}
pub(crate) fn enabled()->bool {historical_enabled() || super::fresh_http::enabled()}
fn bad(run:&mut Run,error:anyhow::Error) {if run.error.is_none(){run.error=Some(format!("{error:#}"));}}
fn recipe(packet:&HttpPacket,model:&Qwen35LoadedModel)->Result<String> {
    ensure!(model.tq_kv_active,"original full W requires actual TQ substrate");
    // K is the controlled store lever, not a target numerical-route change.
    // The exact packet and observed environment still bind/validate its value.
    Ok(format!("C2-full-original-HTTP-{:?};quota={}",packet.route,
        std::env::var("HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX")?))
}
pub(crate) fn model_loaded(model:&Qwen35LoadedModel,architecture:&str)->Result<()> {
    if super::fresh_http::enabled() {return super::fresh_http::model_loaded(model,architecture);}
    let mut global=run_lock(); let Some(run)=global.as_mut() else{return Ok(())};
    ensure!(run.worker.is_none(),"selected full-W process loaded multiple Qwen models");
    ensure!(model.model.kv_graft.is_none(),"original full-W canonical authority is vanilla; graft qualification is the separate complete declared-axis matrix");
    let route=recipe(&run.packet,model)?;
    let config=acquisition::ConfigRecipe {family:run.packet.family,capacity:262144,slots:4,
        output_allowance:8192,cache_mode:"TQ",route:&route,
        actual_model_config_sha256:hash(&serde_json::to_vec(&model.model.cfg)?)};
    run.worker=Some(acquisition::Scope::c2_loaded_binding(&run.packet.identity,&config,
        model.c2_model_witness.clone())?);
    run.architecture=architecture.to_owned(); Ok(())
}
pub(crate) fn prepared_http(headers:&axum::http::HeaderMap,req:&ChatCompletionRequest,prompt:&[u32],params:&mut SamplingParams)->Result<()> {
    if super::fresh_http::enabled() {return super::fresh_http::prepared(headers,req,prompt,params);}
    let mut global=run_lock(); let Some(run)=global.as_mut() else{return Ok(())};
    ensure!(run.error.is_none(),"previous selected acquisition error");
    let mut candidates=run.requests.iter().enumerate()
        .filter(|(_,r)|r.prepared.is_none()&&r.model==req.model&&r.messages==req.messages)
        .map(|(index,_)|index);
    let index=candidates.next().context("AcquisitionBlocked: actual prepared request is not enrolled in the exact HTTP wave")?;
    ensure!(candidates.next().is_none(),"AcquisitionBlocked: actual prepared request has ambiguous HTTP-wave binding");
    ensure!(prompt==run.requests[index].original.prompt_tokens,
        "AcquisitionBlocked: actual prepared prompt changed original native token witness");
    if let Some(values)=&params.token_bytes {
        if run.table.is_none() {run.table=Some(TokenTable::actual(values.clone())?);}
    }
    let frozen=FrozenParams::actual(params,run.table.as_ref(),&run.requests[index].tools)?;
    let prepared=Prepared {request_sha256:run.requests[index].sha.clone(),prompt:prompt.to_vec(),
        params:frozen,registration_architecture:run.architecture.clone(),cached_tokens:None,transactions:Vec::new()};
    ensure!(prepared.params.max_tokens==8192 && prompt.len()+8192<=262144,
        "actual request allowance/context differs from original W");
    params.c2_ticket=Some(format!("historical-{}-{}",run.requests[index].lane,run.requests[index].turn));
    run.requests[index].prepared=Some(prepared); Ok(())
}
pub(crate) fn admitted(model:&Qwen35LoadedModel,cache:&HybridKvCache,slot:SlotId,
    prompt:&[u32],cached:usize,params:&SamplingParams)->Result<()> {
    if super::fresh_http::enabled() {return super::fresh_http::admitted(model,cache,slot,prompt,cached,params);}
    let mut global=run_lock();let Some(run)=global.as_mut() else{return Ok(())};
    ensure!(cache.max_seq_len==262144&&cache.n_seqs==4,"full W must provision actual262144×4, not matrix cache");
    if run.packet.route==Route::CurrentDefault {
        ensure!(model.dflash2.is_some(),"selected default runtime unavailable; cannot count ordinary fallback as default acquisition");
    }
    let mut candidates=run.requests.iter().enumerate()
        .filter(|(_,r)|r.slot.is_none()&&params.c2_ticket.as_deref()==Some(format!("historical-{}-{}",r.lane,r.turn).as_str())
            &&r.prepared.as_ref().is_some_and(|p|p.prompt==prompt))
        .map(|(index,_)|index);
    let index=candidates.next().context("AcquisitionBlocked: native admission has no actual prepared raw-request binding")?;
    ensure!(candidates.next().is_none(),"AcquisitionBlocked: native admission has ambiguous raw-request binding");
    let request=&run.requests[index];
    ensure!(request.original.decision_tokens.iter().all(|&token|(token as usize)<model.vocab_size),
        "AcquisitionBlocked: original native token witness exceeds actual model vocabulary");
    let captured=FrozenParams::actual(params,run.table.as_ref(),&request.tools)?;
    ensure!(serde_json::to_vec(&captured)?==serde_json::to_vec(&request.prepared.as_ref().context("prepared missing")?.params)?,
        "sampling/grammar semantics changed between actual preparation and admission");
    let observed=model.model.with_gpu_cache_mut(|device,_|run.worker.as_ref()
        .context("actual loader witness missing")?.observed(&model.c2_model_witness,device,
            &model.model as *const _ as usize as u64))?;
    if run.writer.is_none() && run.packet.selected_cell==Some([request.lane,request.turn]) {
        let binding=run.packet.expected.as_ref().context("independent expected extents missing")?;
        let supplied:Vec<ExpectedTensor>=serde_json::from_slice(&acquisition::checked_file(binding,64*1024*1024)?)?;
        let canonical=request.canonical.as_ref().context("canonical decisions missing")?;
        let expected=super::divergent::decision_expected(request.lane,request.turn,prompt.len(),canonical.decisions.len(),model.vocab_size,model.hidden_size);
        super::divergent::require_complete(supplied,&expected)?;
        let root=run.packet.output.join("decisions");
        let mut writer=Store::new(&root,run.packet.identity.clone(),
            "C2-ClassB-real-HTTP-actual-resolved-target-teacher-forced-selector-diagnostic".into(),
            expected.clone(),run.packet.payload_limit,expected.len()).map_err(anyhow::Error::msg)?;
        writer.bind_acquisition(observed.for_store_root(&root)?).map_err(anyhow::Error::msg)?;run.writer=Some(writer);
    }
    let request=&mut run.requests[index];request.slot=Some(slot.0);request.cached=cached;
    request.prepared.as_mut().context("prepared missing")?.cached_tokens=Some(cached);
    request.origin=prompt.len()-1+model.model.kv_graft.as_ref().map_or(0,|g|g.bank.n_slots as usize);Ok(())
}
pub(crate) fn forward(slot:SlotId,start:usize,input:&[u32],route:&str,logits:&[f32],vocab:usize,
    hidden:Option<&[f32]>,hidden_width:usize)->Result<()> {
    if super::fresh_http::enabled() {return super::fresh_http::forward(slot,start,input,route,logits,vocab,hidden,hidden_width);}
    let mut global=run_lock();let Some(run)=global.as_mut() else{return Ok(())};
    let Some(request)=run.requests.iter_mut().find(|r|r.slot==Some(slot.0)&&r.terminal.is_none()) else{return Ok(())};
    request.prepared.as_mut().context("prepared missing")?.transactions.push(super::frozen_script::ForwardBoundary {
        start,input:input.to_vec(),logit_rows:if vocab==0{0}else{logits.len()/vocab},
        hidden_rows:if hidden_width==0{0}else{hidden.map_or(0,|h|h.len()/hidden_width)},route:route.into(),
    });
    if run.packet.selected_cell!=Some([request.lane,request.turn]) {return Ok(())}
    let canonical=request.canonical.as_ref().context("decision trace requires canonical reference")?;
    ensure!(vocab>0&&logits.len()%vocab==0,"native logits are not full vocabulary");
    let rows=logits.len()/vocab;ensure!(rows<=input.len(),"native row/input extent mismatch");
    let hidden=hidden.context("actual normalized target hidden missing")?;
    ensure!(hidden_width>0&&hidden.len()==rows*hidden_width,"actual hidden/logits extents differ");
    for row in 0..rows {
        let position=start+input.len()-rows+row;
        let Some(index)=position.checked_sub(request.origin) else{continue};
        if index>=canonical.decisions.len(){continue}
        let prompt=&request.prepared.as_ref().context("prepared missing")?.prompt;
        let graft=request.origin+1-prompt.len();
        let consumed=input.len()-rows+row;
        let canonical_prefix=input[..=consumed].iter().enumerate().all(|(offset,&token)| {
            let Some(logical)=(start+offset).checked_sub(graft) else{return false};
            let expected=if logical<prompt.len(){prompt.get(logical)}
                else{canonical.decisions.get(logical-prompt.len())};
            expected.is_some_and(|&expected|expected==token)
        });
        if !canonical_prefix {continue}
        request.pending.insert(index,Row {logits:logits[row*vocab..(row+1)*vocab].to_vec(),
            hidden:hidden[row*hidden_width..(row+1)*hidden_width].to_vec(),route:route.into()});
    }
    Ok(())
}
pub(crate) fn decision(slot:SlotId,index:usize,actual:u32)->Result<u32> {
    if super::fresh_http::enabled() {return super::fresh_http::decision(slot,index,actual);}
    let mut global=run_lock();let Some(run)=global.as_mut() else{return Ok(actual)};
    let Some(which)=run.requests.iter().position(|r|r.slot==Some(slot.0)&&r.terminal.is_none()) else{return Ok(actual)};
    let request=&mut run.requests[which];
    ensure!(index==request.decisions.len(),"native canonical decision index gap/duplicate");
    let selected=request.canonical.as_ref().map(|c|c.decisions.get(index).copied()
        .context("AcquisitionBlocked: actual continuation exceeds complete frozen extent")).transpose()?.unwrap_or(actual);
    request.raw_decisions.push(actual);request.decisions.push(selected);
    if run.packet.selected_cell==Some([request.lane,request.turn]) {
        let row=request.pending.remove(&index).context("AcquisitionBlocked: actual canonical decision has no retained real target row")?;
        let writer=run.writer.as_mut().context("actual acquisition writer missing")?;
        let prepared=request.prepared.as_ref().context("prepared missing")?;
        let at=|stage|trace::coordinate(&format!("agent-{}",request.lane),request.turn,prepared.prompt.len()-1+index,stage);
        let vocabulary=row.logits.len();let hidden=row.hidden.len();
        trace::capture_f32(writer,at(Stage::Logits),&row.logits,vec![1,vocabulary])?;
        trace::capture_f32(writer,at(Stage::SpecVerifierHidden),&row.hidden,vec![1,hidden])?;
        writer.capture(at(Stage::SpecCanonicalIds),Scalar::U32,vec![1],&selected as *const u32 as usize as u64,
            4,0,None,"actual semantic commit after checked target drain; frozen diagnostic selector",&selected.to_le_bytes())
            .map_err(anyhow::Error::msg)?;
        request.rows+=1;request.routes.push(row.route);
    }
    Ok(selected)
}
pub(crate) fn gpu_token(slot:SlotId,position:usize)->Result<Option<u32>> {
    if super::fresh_http::enabled() {return super::fresh_http::gpu_token(slot,position);}
    let global=run_lock();let Some(run)=global.as_ref() else{return Ok(None)};
    let Some(request)=run.requests.iter().find(|r|r.slot==Some(slot.0)&&r.terminal.is_none()) else{return Ok(None)};
    let Some(canonical)=&request.canonical else{return Ok(None)};
    Ok(position.checked_sub(request.origin).and_then(|i|canonical.decisions.get(i).copied()))
}
pub(crate) fn gpu_selected(slot:SlotId) {
    if super::fresh_http::enabled() {super::fresh_http::gpu_selected(slot);return;}
    if let Some(run)=run_lock().as_mut(){if let Some(r)=run.requests.iter_mut().find(|r|r.slot==Some(slot.0)&&r.terminal.is_none()){r.gpu_selector=true;}}
}
pub(crate) fn argmax(slot:SlotId,position:usize,token:u32) {
    if super::fresh_http::enabled(){super::fresh_http::argmax(slot,position,token);return;}
    if let Some(run)=run_lock().as_mut(){if let Some(r)=run.requests.iter_mut().find(|r|r.slot==Some(slot.0)&&r.terminal.is_none()){r.gpu_argmax.push((position,token));}}
}
pub(crate) fn finished(state:&Qwen35DecodeState) {
    if super::fresh_http::enabled() {super::fresh_http::finished(state);return;}
    let mut global=run_lock();let Some(run)=global.as_mut() else{return};
    let result=(||->Result<()> {
        let request=run.requests.iter_mut().find(|r|state.params.c2_ticket.as_deref()==Some(format!("historical-{}-{}",r.lane,r.turn).as_str())
            && r.slot==Some(state.slot_id.0)).context("terminal request ticket/generation not enrolled")?;
        let terminal=Terminal {request_sha256:request.sha.clone(),slot:state.slot_id.0,
            decisions:request.decisions.clone(),generated:state.generated_tokens.clone(),finish_reason:state.finish_reason.to_owned(),
            native_step:state.step};
        if let Some(expected)=&request.terminal {ensure!(expected==&terminal,"actual terminal changed after completion");return Ok(())}
        request.original.require_native(request.prepared.as_ref().context("actual preparation missing")?,&terminal)?;
        if let Some(canonical)=&request.canonical {
            ensure!(terminal.decisions==canonical.decisions&&terminal.generated==canonical.generated
                &&terminal.finish_reason==canonical.finish_reason,"AcquisitionBlocked: canonical continuation/terminal not acquired");
        }
        request.terminal=Some(terminal);Ok(())
    })();if let Err(error)=result{bad(run,error)}
}
fn binding(path:&Path)->Result<serde_json::Value> {let bytes=fs::read(path)?;Ok(serde_json::json!({"path":path,"sha256":hash(&bytes)}))}
fn save(path:&Path,value:&impl Serialize)->Result<serde_json::Value> {fs::write(path,serde_json::to_vec(value)?)?;binding(path)}
fn enrolled(lane:usize,turn:usize,raw:&[u8],original:Arc<OriginalWitness>,canonical:Option<Terminal>)->Result<Request> {
    let req:ChatCompletionRequest=serde_json::from_slice(raw)?;
    let wire:serde_json::Value=serde_json::from_slice(raw)?;
    ensure!(original.request_sha256==hash(raw),"AcquisitionBlocked: original witness raw request binding changed");
    if let Some(expected)=&canonical {ensure!(expected.request_sha256==hash(raw),"canonical replay changed exact raw request bytes");}
    Ok(Request {lane,turn,sha:hash(raw),messages:req.messages,model:req.model,tools:req.tools.unwrap_or_default(),
        original,stream:wire["stream"]==true,
        prepared:None,canonical,terminal:None,slot:None,origin:0,cached:0,pending:Default::default(),
        decisions:Vec::new(),raw_decisions:Vec::new(),gpu_argmax:Vec::new(),gpu_selector:false,rows:0,routes:Vec::new()})
}
fn driver(originals:OriginalCorpus,validated:Option<Validated>,stop:Arc<AtomicBool>)->Result<()> {
    let raws=&originals.requests;
    let (port,timeout,output,selected,arm)={let lock=run_lock();let run=lock.as_ref().context("HTTP run missing")?;
        (run.packet.port,run.packet.ready_timeout_seconds,run.packet.output.clone(),run.packet.selected_cell,run.packet.arm.clone())};
    let client=reqwest::blocking::Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none()).timeout(Duration::from_secs(1800)).build()?;
    let base=format!("http://127.0.0.1:{port}");let start=Instant::now();
    loop {
        ensure!(!stop.load(Ordering::Acquire),"real server exited before readiness");
        if let Ok(response)=client.get(format!("{base}/readyz")).send(){if response.status().is_success(){let value:serde_json::Value=response.json()?;if value["ready"]==true{break}}}
        ensure!(start.elapsed()<Duration::from_secs(timeout),"actual selected server readiness deadline");std::thread::sleep(Duration::from_millis(100));
    }
    let target=selected.map(|c|c[1]);let cold=arm=="cold";
    let last=target.unwrap_or_else(||raws.iter().map(Vec::len).max().unwrap_or(0)-1);
    let mut reference_lanes:Vec<Vec<serde_json::Value>>=(0..4).map(|_|Vec::new()).collect();
    for turn in 0..=last {
        if cold&&Some(turn)!=target {continue}
        let mut wave=Vec::new();
        {let mut lock=run_lock();let run=lock.as_mut().context("HTTP run missing")?;
            ensure!(run.requests.iter().all(|r|r.terminal.is_some()),"previous real HTTP wave did not finish");run.requests.clear();
            for lane in 0..4 {if let Some(raw)=raws[lane].get(turn){
                let canonical=validated.as_ref().map(|v|v.turns[lane][turn].terminal.clone());
                run.requests.push(enrolled(lane,turn,raw,originals.witnesses[lane][turn].clone(),canonical)?);wave.push((lane,raw.clone()));
            }}
        }
        let barrier=Arc::new(std::sync::Barrier::new(wave.len()));let mut jobs=Vec::new();
        for (lane,raw) in wave {let client=client.clone();let url=format!("{base}/v1/chat/completions");let barrier=barrier.clone();
            jobs.push((lane,std::thread::spawn(move||->Result<Vec<u8>>{barrier.wait();let response=client.post(url).header("content-type","application/json").body(raw).send()?;
                ensure!(response.status().is_success(),"actual HTTP response rejected: {}",response.status());
                let mut bytes=Vec::new();response.take(64*1024*1024+1).read_to_end(&mut bytes)?;ensure!(bytes.len()<=64*1024*1024,"real response exceeded bound");Ok(bytes)})));
        }
        for (lane,job) in jobs {
            let response=job.join().map_err(|_|anyhow::anyhow!("actual HTTP client panicked"))??;
            let mut lock=run_lock();let run=lock.as_mut().context("HTTP run missing")?;
            ensure!(run.error.is_none(),"selected acquisition failed: {:?}",run.error);
            let request=run.requests.iter().find(|r|r.lane==lane).context("actual lane missing")?;
            let prepared=request.prepared.as_ref().context("actual preparation observer missing")?;
            let terminal=request.terminal.as_ref().context("actual terminal observer missing")?;
            request.original.require_response(&response,request.stream,prepared,terminal)?;
            if let Some(v)=&validated {let expected=&v.turns[lane][turn].prepared;
                ensure!(prepared.prompt==expected.prompt&&serde_json::to_vec(&prepared.params)?==serde_json::to_vec(&expected.params)?
                    &&prepared.registration_architecture==expected.registration_architecture,"actual preparation differs from independent canonical reference");}
            let dir=output.join(format!("agent-{lane}-turn-{turn}"));fs::create_dir(&dir)?;fs::set_permissions(&dir,fs::Permissions::from_mode(0o700))?;
            let p=save(&dir.join("prepared.json"),prepared)?;let t=save(&dir.join("terminal.json"),terminal)?;
            fs::write(dir.join("response.bin"),&response)?;let response_binding=binding(&dir.join("response.bin"))?;
            reference_lanes[lane].push(serde_json::json!({"turn":turn,"prepared":p,"terminal":t,"response":response_binding}));
            let selector_sha256=request.canonical.as_ref().map(|canonical|
                serde_json::to_vec(&(request.origin,&canonical.decisions)).map(|bytes|hash(&bytes))).transpose()?;
            save(&dir.join("route-observation.json"),&serde_json::json!({"cached_tokens":request.cached,"slot":request.slot,
                "raw_decisions":request.raw_decisions,"actual_gpu_argmax":request.gpu_argmax,"gpu_frozen_selector_used":request.gpu_selector,
                "selector_sha256":selector_sha256,"selector_route":"TeacherForcedGpuAcceptanceDiagnosticWhenGpuTailActive",
                "actual_target_routes":request.routes,"observed_decision_rows":request.rows,"not_default_quality_or_timing_proof":true}))?;
            if selected==Some([lane,turn]) {
                if arm=="cold" {ensure!(request.cached==0,"cold arm reused history");}
                if arm=="resumed"&&run.packet.keep==4&&matches!(turn,3|5) {
                    ensure!(request.cached>0,"AcquisitionBlocked: original divergent K4 cell did not actually reuse an older boundary");
                }
            }
        }
    }
    let mut lock=run_lock();let run=lock.as_mut().context("HTTP run missing")?;
    if let Some(writer)=run.writer.take(){super::divergent::complete(writer)?;}else{ensure!(selected.is_none(),"selected all-row acquisition never opened");}
    if validated.is_none() {
        let corpus:serde_json::Value=serde_json::from_slice(&acquisition::checked_file(&run.packet.corpus,4*1024*1024)?)?;
        let table=run.table.as_ref().map(|table|save(&output.join("token-table.json"),table.values.as_ref())).transpose()?;
        let case:serde_json::Value=serde_json::from_slice(&acquisition::checked_file(&serde_json::from_value::<Corpus>(corpus.clone())?.case,4*1024*1024)?)?;
        save(&output.join("frozen-script.json"),&serde_json::json!({"schema":"hf2q.c2.full-original-frozen-script.v1","corpus":corpus,
            "reference_identity":run.packet.identity,"reference_route":"actual-production-HTTP-OFF-canonical","token_table":table,
            "lanes":reference_lanes.into_iter().enumerate().map(|(lane,turns)|serde_json::json!({"lane":lane,"turns":turns})).collect::<Vec<_>>(),
            "prefill_chunk":crate::serve::api::engine::qwen35_slot_prefill_chunk_tokens(4),
            "anchor_bytes_per_slot_max":case["workload"]["anchor_bytes_per_slot_max"]}))?;
        Script::validate_bytes(&fs::read(output.join("frozen-script.json"))?,run.packet.family)?;
    }
    save(&output.join("acquisition-result.json"),&serde_json::json!({"schema":"hf2q.c2.full-original-http-acquisition.v1",
        "complete_original_corpus_input_authority":true,"complete_original_corpus_acquisition":selected.is_none(),
        "executed_through_turn":last,"selected_cell":selected,"arm":arm,"declared_route":run.packet.route,
        "anchors_per_slot":run.packet.keep,
        "token_count_convention":{"generated":"verbatim native generated_tokens; never trimmed/padded",
            "decisions":"all actual semantic selections including any observed terminal EOS",
            "native_step":"verbatim decoder counter; not substituted for HTTP usage",
            "expected_extents":"original bound HTTP usage and complete independent canonical forward/token tapes"},
        "numerical_acceptance":"UNPROVED: consumer must compare independently complete traces with q1_decision_policy.envelope D<=0.25nat",
        "teacher_forced_diagnostic":validated.is_some(),"not_whole_C2_pass":true,
        "original_same_native_parent_refactor_gate":"51a3923fafbe46e4a18c22c5fdc24ce20a46593c vs62a5326422e7ebb8e5621c978cfd6d38b23facd8; both published mlx-native0.18.0; separate historical proof"}))?;
    drop(lock);client.post(format!("{base}/shutdown")).send()?.error_for_status()?;Ok(())
}
pub(super) fn run(family:Family)->Result<()> {
    use clap::Parser;
    let path=PathBuf::from(std::env::var_os("HF2Q_C2_FULLW_PACKET").context("full original-W HTTP packet missing")?);
    let raw=fs::read(path)?;ensure!(raw.len()<=4*1024*1024,"full-W control packet exceeds4MiB");
    let packet:HttpPacket=serde_json::from_slice(&raw)?;
    ensure!(packet.execution_approved&&packet.family==family&&packet.identity.family==family,"root full-W grant/family missing");
    ensure!(packet.port>=1024&&packet.ready_timeout_seconds>0&&packet.ready_timeout_seconds<=3600,"bounded owned loopback endpoint required");
    let reservation=std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST,packet.port)).context("selected port unavailable before model loading")?;
    ensure!(matches!(packet.keep,1|4)&&packet.model.is_absolute()&&packet.model.is_file(),"original artifact/depth missing");
    ensure!(packet.identity.model_sha256==if family==Family::Qwen38{Q38}else{Q36},"wrong original artifact");
    let corpus_raw=acquisition::checked_file(&packet.corpus,4*1024*1024)?;
    let corpus:Corpus=serde_json::from_slice(&corpus_raw)?;
    let cli=crate::cli::Cli::try_parse_from(&packet.server_argv)?;
    let crate::cli::Command::Serve(args)=cli.command else{anyhow::bail!("actual production serve CLI required")};
    ensure!(args.model.as_ref()==Some(&packet.model)&&args.host.as_deref()==Some("127.0.0.1")&&args.port==Some(packet.port)
        &&args.planning.ctx==Some(262144)&&args.planning.max_slots==Some(4)
        &&args.planning.scheduler==Some(crate::cli::SchedulerArg::InflightBatched),"actual original262144/four-slot production server required");
    let case:serde_json::Value=serde_json::from_slice(&acquisition::checked_file(&corpus.case,4*1024*1024)?)?;
    let kv=case["serve"]["env"]["KV_CACHE_BUDGET_BYTES"].as_str().context("source-bound aggregate KV budget missing")?;
    ensure!(args.planning.kv_cache_budget.as_deref()==Some(kv),"actual KV budget differs from original case bytes");
    ensure!(std::env::var("HF2Q_QWEN_ANCHOR_DEPTH")?==packet.keep.to_string()
        &&std::env::var("HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX")?==case["workload"]["anchor_bytes_per_slot_max"].to_string(),"actual source-bound anchor/capture budget missing");
    ensure!(packet.payload_limit>0&&packet.payload_limit<=8*1024*1024*1024*1024,"finite independently admitted diagnostic disk limit required");
    let (scope,validated,originals)=if let Some(binding)=&packet.frozen_script {
        ensure!(packet.selected_cell.is_some()&&matches!(packet.arm.as_str(),"cold"|"resumed"),"bounded full-W replay cell/arm required");
        let script=acquisition::checked_file(binding,32*1024*1024)?;
        let (scope,mut validated)=acquisition::Scope::begin_c2(&packet.acquisition,&packet.identity,&packet.model,&packet.output,&raw,&script)?;
        let script_value:serde_json::Value=serde_json::from_slice(&script)?;
        ensure!(script_value["corpus"]==serde_json::from_slice::<serde_json::Value>(&corpus_raw)?,
            "AcquisitionBlocked: control corpus differs from independently frozen original corpus binding");
        let [lane,turn]=packet.selected_cell.context("cell missing")?;
        ensure!(lane<4&&turn<validated.turns[lane].len(),"selected full-W cell outside independent complete corpus");
        let originals=std::mem::take(&mut validated.originals);
        (scope,Some(validated),originals)
    }else{
        ensure!(packet.route==Route::Off&&packet.selected_cell.is_none()&&packet.arm=="canonical","canonical capture is real unforced OFF, not default evidence");
        let (scope,originals)=acquisition::Scope::begin_c2_capture(&packet.acquisition,&packet.identity,&packet.model,&packet.output,&raw,&corpus_raw)?;
        (scope,None,originals)
    };
    match packet.route {
        Route::Off=>ensure!(std::env::var("HF2Q_DFLASH2")?=="0"&&std::env::var("HF2Q_QWEN_SPECULATION")?=="off","OFF must really disable both speculation engines"),
        Route::MtpAuto=>ensure!(std::env::var("HF2Q_DFLASH2")?=="0"&&std::env::var("HF2Q_QWEN_SPECULATION")?=="auto","actual K3 route prerequisite missing"),
        Route::CurrentDefault=>{
            ensure!(std::env::var("HF2Q_DFLASH2").unwrap_or_default()!="0","current default was disabled");
            for name in ["HF2Q_DFLASH2_CONFIG","HF2Q_DFLASH2_WEIGHTS"] {
                let path=PathBuf::from(std::env::var_os(name).with_context(||format!("required published default artifact {name} missing"))?);
                ensure!(path.is_absolute()&&path.is_file(),"required default artifact unavailable: {name}; no implicit download/fallback");
            }
        },
    }
    output_preflight(&packet.output)?;fs::create_dir(&packet.output)?;fs::set_permissions(&packet.output,fs::Permissions::from_mode(0o700))?;
    let _gpu=crate::inference::hf2q_gpu_test_lock();
    scope.c2_hold()?;
    {let mut global=run_lock();ensure!(global.is_none(),"another full-W scope is active");
        *global=Some(Run {packet,table:None,requests:Vec::new(),architecture:String::new(),worker:None,writer:None,error:None});}
    let stop=Arc::new(AtomicBool::new(false));let signal=stop.clone();drop(reservation);
    let worker=std::thread::spawn(move||{let result=driver(originals,validated,signal);if result.is_err(){unsafe{libc::raise(libc::SIGTERM);}}result});
    let served=crate::serve::cmd_serve(args,cli.log_format,None,None);stop.store(true,Ordering::Release);
    let captured=worker.join().map_err(|_|anyhow::anyhow!("real HTTP acquisition driver panicked"))?;
    let release=acquisition::Scope::c2_release();run_lock().take();served?;captured?;release
}

pub(crate) fn route(slot:SlotId,route:&'static str) {
    if super::fresh_http::enabled(){super::fresh_http::route(slot,route);return;}
    if let Some(run)=run_lock().as_mut() {
        if let Some(request)=run.requests.iter_mut().find(|r|r.slot==Some(slot.0)&&r.terminal.is_none()) {
            request.routes.push(route.to_owned());
        }
    }
}
