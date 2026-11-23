//! Real cmd_serve fixed-input capture, followed ONLY by independently registered
//! frozen-selector replays. No canonical client/tool implementation lives here.
use super::*;
use super::{fresh_reference as reference, frozen_script::{Prepared,Terminal,ForwardBoundary}, frozen_params::{FrozenParams,TokenTable}};
use crate::serve::api::schema::{ChatCompletionRequest,Tool};
use std::{collections::BTreeMap,sync::{Arc,Mutex,LazyLock,atomic::{AtomicBool,Ordering}},time::{Duration,Instant},io::Write};
use serde_json::Value;
#[path = "c23_localization.rs"]
mod localization;
pub(crate) use localization::{anchor_created, anchor_staged, anchor_restore_ticket, localization_cohort};
pub(in crate::serve::api) use localization::anchor_restored;
const HEADER:&str="x-hf2q-c2-ticket";
#[derive(Clone,Copy,Deserialize,PartialEq,Eq)]
#[serde(rename_all="snake_case")]
enum Mode {Capture,Replay}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    schema:String,execution_approved:bool,mode:Mode,owner_decision:acquisition::FileBinding,
    family:Family,identity:Identity,acquisition:acquisition::Binding,model:PathBuf,output:PathBuf,
    port:u16,server_argv:Vec<String>,inputs:acquisition::FileBinding,
    registration:Option<acquisition::FileBinding>,selected_cell:Option<[usize;2]>,
    arm:String,keep:usize,route:Route,expected:Option<acquisition::FileBinding>,
    payload_limit:u64,ready_timeout_seconds:u64,
    localization:Option<acquisition::FileBinding>,
}
struct Row {logits:Vec<f32>,hidden:Vec<f32>,route:String}
struct Request {
    lane:usize,turn:usize,ticket:String,raw:axum::body::Bytes,tools:Vec<Tool>,stream:bool,dir:PathBuf,
    ingress:bool,prepared:Option<Prepared>,terminal:Option<Terminal>,stop:Option<reference::StopSelection>,
    slot:Option<u32>,generation:u64,acquisition:Option<acquisition::Acquisition>,
    decisions:Vec<u32>,raw_decisions:Vec<u32>,canonical:Option<Terminal>,origin:usize,
    pending:BTreeMap<usize,Row>,routes:Vec<String>,gpu_selector:bool,
    gpu_argmax:Vec<(usize,u32)>,
    wire:Option<(u16,Value,String,usize)>,
    published:Option<usize>,
}
#[derive(Default)]
struct Owners {active:BTreeMap<u32,(String,u64)>,next:u64}
impl Owners {
    fn admit(&mut self,slot:u32,ticket:&str)->Result<u64> {
        ensure!(!self.active.contains_key(&slot),"admission overwrote an active native generation");
        self.next=self.next.checked_add(1).context("native generation overflow")?;
        self.active.insert(slot,(ticket.into(),self.next));Ok(self.next)
    }
    fn finish(&mut self,slot:u32,ticket:&str,generation:u64)->Result<()> {
        ensure!(self.active.get(&slot).is_some_and(|(owner,current)|owner==ticket&&*current==generation),
            "stale/cross-request terminal cannot release a reused slot");
        self.active.remove(&slot);Ok(())
    }
}
struct Run {
    packet:Packet,requests:Vec<Request>,owners:Owners,
    prefix:String,architecture:String,table:Option<TokenTable>,worker:Option<acquisition::c2::C2WorkerBinding>,
    writer:Option<Store>,error:Option<String>,
    localization:Option<localization::Session>,
}
static RUN:LazyLock<Mutex<Option<Run>>>=LazyLock::new(||Mutex::new(None));
fn lock()->std::sync::MutexGuard<'static,Option<Run>> {RUN.lock().unwrap_or_else(|e|e.into_inner())}
pub(crate) fn enabled()->bool {lock().is_some()}
fn request<'a>(run:&'a mut Run,ticket:&str)->Result<&'a mut Request> {
    run.requests.iter_mut().find(|r|r.ticket==ticket).context("C2 unknown/stale request ticket")
}
fn owner(run:&Run,slot:SlotId)->Result<usize> {
    let (ticket,generation)=run.owners.active.get(&slot.0).context("C2 native callback has no active request owner")?;
    run.requests.iter().position(|r| &r.ticket==ticket && r.generation==*generation && r.slot==Some(slot.0)
        && r.terminal.is_none()).context("C2 stale/cross-request native slot generation")
}
fn fail(ticket:&str,error:anyhow::Error) {
    if let Some(run)=lock().as_mut() {if run.error.is_none(){run.error=Some(format!("{ticket}: {error:#}"));}}
}
static DISK_LIMIT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
static DISK_USED:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
struct DiskBudget;
impl DiskBudget {
    fn begin(limit:u64)->Result<Self> {
        DISK_LIMIT.compare_exchange(0,limit,Ordering::AcqRel,Ordering::Acquire).map_err(|_|anyhow::anyhow!("another diagnostic disk grant active"))?;
        DISK_USED.store(0,Ordering::Release);Ok(Self)
    }
}
impl Drop for DiskBudget {fn drop(&mut self){DISK_LIMIT.store(0,Ordering::Release);}}
fn charge(bytes:usize)->Result<()> {
    let limit=DISK_LIMIT.load(Ordering::Acquire);
    ensure!(limit>0,"diagnostic disk grant absent");
    let mut used=DISK_USED.load(Ordering::Acquire);
    loop {
        let total=used.checked_add(bytes as u64).filter(|total|*total<=limit)
            .context("diagnostic disk grant exhausted; capture is incomplete")?;
        match DISK_USED.compare_exchange_weak(used,total,Ordering::AcqRel,Ordering::Acquire) {
            Ok(_)=>return Ok(()),
            Err(current)=>used=current,
        }
    }
}
fn binding(path:&Path)->Result<acquisition::FileBinding> {
    Ok(acquisition::FileBinding{path:path.to_path_buf(),sha256:hash(&fs::read(path)?)})
}
fn save(path:&Path,value:&impl Serialize)->Result<acquisition::FileBinding> {save_bytes(path,&serde_json::to_vec(value)?)}
fn save_bytes(path:&Path,raw:&[u8])->Result<acquisition::FileBinding> {
    charge(raw.len())?;
    let mut file=fs::OpenOptions::new().write(true).create_new(true).open(path)?;file.write_all(raw)?;
    Ok(acquisition::FileBinding{path:path.to_path_buf(),sha256:hash(raw)})
}
fn header(headers:&axum::http::HeaderMap)->Result<String> {
    let mut values=headers.get_all(HEADER).iter();let value=values.next().context("C2 request ticket header missing")?.to_str()?;
    ensure!(values.next().is_none() && !value.is_empty() && value.len()<=192 && value.bytes().all(|c|c.is_ascii_alphanumeric()||b"-_".contains(&c)),"invalid/duplicate C2 request ticket");
    Ok(value.into())
}
struct WireGuard {ticket:String,complete:bool}
impl Drop for WireGuard {fn drop(&mut self){if !self.complete{fail(&self.ticket,anyhow::anyhow!("raw HTTP body ended without complete capture"));}}}
/// Test-only body tee. Both endpoints retain the actual entity bytes; transfer
/// framing is not represented as model tokens or as reconstructed SSE text.
pub(crate) async fn middleware(req:axum::extract::Request,next:axum::middleware::Next)->axum::response::Response {
    use axum::response::IntoResponse;
    use futures::StreamExt;
    use sha2::{Digest,Sha256};
    if !enabled() || req.uri().path()!="/v1/chat/completions" {return next.run(req).await;}
    let result=(||->Result<String>{header(req.headers())})();
    let ticket=match result {Ok(ticket)=>ticket,Err(error)=>{fail("ingress",error);return axum::http::StatusCode::BAD_REQUEST.into_response();}};
    let (parts,body)=req.into_parts();
    let raw=match axum::body::to_bytes(body,256*1024).await {Ok(raw)=>raw,Err(error)=>{fail(&ticket,error.into());return axum::http::StatusCode::BAD_REQUEST.into_response();}};
    let checked=(||->Result<PathBuf>{let mut global=lock();let run=global.as_mut().context("capture disappeared")?;
        let r=request(run,&ticket)?;ensure!(!r.ingress&&r.raw==raw.as_ref(),"duplicate ticket or changed exact HTTP request body");
        r.ingress=true;save_bytes(&r.dir.join("server-request.bin"),&raw)?;
        save(&r.dir.join("request-headers.json"),&serde_json::json!({"ticket":ticket,"headers":parts.headers.iter().map(|(k,v)|(k.as_str(),v.as_bytes())).collect::<Vec<_>>()}))?;
        Ok(r.dir.clone())})();
    let dir=match checked {Ok(dir)=>dir,Err(error)=>{fail(&ticket,error);return axum::http::StatusCode::BAD_REQUEST.into_response();}};
    let response=next.run(axum::extract::Request::from_parts(parts,axum::body::Body::from(raw))).await;
    let (parts,body)=response.into_parts();let status=parts.status.as_u16();
    let headers=serde_json::json!({"status":status,"headers":parts.headers.iter().map(|(k,v)|(k.as_str(),v.as_bytes())).collect::<Vec<_>>()});
    let body:std::pin::Pin<Box<dyn futures::Stream<Item=std::result::Result<axum::body::Bytes,std::io::Error>>+Send>>=Box::pin(async_stream::try_stream! {
        let mut guard=WireGuard{ticket:ticket.clone(),complete:false};
        let mut file=fs::OpenOptions::new().write(true).create_new(true).open(dir.join("server-response.bin"))?;
        let mut stream=body.into_data_stream();let mut digest=Sha256::new();let mut count=0usize;
        while let Some(chunk)=stream.next().await {
            let chunk=chunk.map_err(std::io::Error::other)?;
            count=count.checked_add(chunk.len()).ok_or_else(||std::io::Error::other("HTTP byte count overflow"))?;
            if count>64*1024*1024 {Err(std::io::Error::other("raw response exceeds64MiB"))?;}
            charge(chunk.len()).map_err(|error|std::io::Error::other(format!("{error:#}")))?;
            file.write_all(&chunk)?;digest.update(&chunk);yield chunk;
        }
        let result=(||->Result<()> {let mut global=lock();let run=global.as_mut().context("capture disappeared")?;
            let r=request(run,&ticket)?;ensure!(r.wire.is_none(),"duplicate raw HTTP terminal");
            r.wire=Some((status,headers,hex::encode(digest.finalize()),count));Ok(())})();
        result.map_err(|error|std::io::Error::other(format!("{error:#}")))?;guard.complete=true;
    });
    axum::response::Response::from_parts(parts,axum::body::Body::from_stream(body))
}
pub(crate) fn prepared(headers:&axum::http::HeaderMap,req:&ChatCompletionRequest,prompt:&[u32],params:&mut SamplingParams)->Result<()> {
    let ticket=header(headers)?;let mut global=lock();let run=global.as_mut().context("fresh capture not active")?;
    ensure!(run.error.is_none(),"previous fresh capture error: {:?}",run.error);
    let index=run.requests.iter().position(|r|r.ticket==ticket).context("prepared unknown request ticket")?;
    ensure!(run.requests[index].ingress&&run.requests[index].prepared.is_none(),"preparation without unique raw ingress");
    let actual:ChatCompletionRequest=serde_json::from_slice(&run.requests[index].raw)?;
    ensure!(actual.model==req.model && actual.messages==req.messages,"handler changed fixed-input request identity");
    if let Some(table)=&params.token_bytes {if run.table.is_none(){run.table=Some(TokenTable::actual(table.clone())?);}}
    let frozen=FrozenParams::actual(params,run.table.as_ref(),&run.requests[index].tools)?;
    ensure!(!prompt.is_empty()&&prompt.len()+8192<=262144&&frozen.max_tokens==8192,"fixed-input context/output budget changed");
    run.requests[index].prepared=Some(Prepared{request_sha256:hash(&run.requests[index].raw),prompt:prompt.to_vec(),params:frozen,
        registration_architecture:run.architecture.clone(),cached_tokens:None,transactions:Vec::new()});
    params.c2_ticket=Some(ticket);Ok(())
}

pub(super) fn config_recipe<'a>(
    family: Family, route: &'a str, actual_model_config_sha256: String,
) -> acquisition::ConfigRecipe<'a> {
    acquisition::ConfigRecipe { family, capacity: 262144, slots: 4,
        output_allowance: 8192, cache_mode: "TQ", route, actual_model_config_sha256 }
}

pub(crate) fn model_loaded(model:&Qwen35LoadedModel,architecture:&str)->Result<()> {
    let mut global=lock();let run=global.as_mut().context("fresh capture absent")?;
    ensure!(run.worker.is_none()&&model.model.kv_graft.is_none()&&model.tq_kv_active,"single vanilla TQ model required");
    let route=format!("C2-fresh-fixed-HTTP-{:?};quota={}",run.packet.route,std::env::var("HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX")?);
    run.worker=Some(acquisition::Scope::c2_loaded_binding(&run.packet.identity,
        &config_recipe(run.packet.family,&route,hash(&serde_json::to_vec(&model.model.cfg)?)),
        model.c2_model_witness.clone())?);
    run.architecture=architecture.into();Ok(())
}
pub(crate) fn admitted(model:&Qwen35LoadedModel,cache:&HybridKvCache,slot:SlotId,prompt:&[u32],cached:usize,params:&SamplingParams)->Result<()> {
    let mut global=lock();let run=global.as_mut().context("fresh capture absent")?;
    if run.requests.is_empty()&&params.c2_ticket.is_none(){return Ok(())}
    ensure!(cache.max_seq_len==262144&&cache.n_seqs==4&&!run.owners.active.contains_key(&slot.0),"actual four-slot capacity/slot generation differs");
    match run.packet.route {
        Route::Off=>ensure!(model.dflash2.is_none()&&model.speculation.policy()==QwenSpeculationPolicy::Off,"actual OFF route differs"),
        Route::MtpAuto=>ensure!(model.dflash2.is_none()&&model.speculation.policy()==QwenSpeculationPolicy::Auto,"actual MTP route differs"),
        Route::CurrentDefault=>ensure!(model.dflash2.is_some(),"default route fell back"),
    }
    let ticket=params.c2_ticket.as_deref().context("native admission lost HTTP request ticket")?;
    let index=run.requests.iter().position(|r|r.ticket==ticket).context("native admission ticket not enrolled")?;
    let r=&run.requests[index];let p=r.prepared.as_ref().context("native admission before actual preparation")?;
    ensure!(r.slot.is_none()&&p.prompt==prompt&&serde_json::to_vec(&p.params)?==serde_json::to_vec(&FrozenParams::actual(params,run.table.as_ref(),&r.tools)?)?,"native admission changed request/settings");
    let observed=model.model.with_gpu_cache_mut(|device,_|run.worker.as_ref().context("loader witness absent")?
        .observed(&model.c2_model_witness,device,&model.model as *const _ as usize as u64))?;
    if run.packet.selected_cell==Some([r.lane,r.turn]) {
        let canonical=r.canonical.as_ref().context("numerical replay missing independent selector")?;
        let supplied:Vec<ExpectedTensor>=serde_json::from_slice(&acquisition::checked_file(run.packet.expected.as_ref().context("independent extent missing")?,64*1024*1024)?)?;
        let expected=super::divergent::decision_expected(r.lane,r.turn,prompt.len(),canonical.decisions.len(),model.vocab_size,model.hidden_size);
        super::divergent::require_complete(supplied,&expected)?;
        ensure!(run.writer.is_none(),"duplicate selected numerical store");let root=run.packet.output.join("decisions");
        let mut writer=Store::new(&root,run.packet.identity.clone(),"C2-fresh-fixed-input-teacher-forced-numerical-replay".into(),expected.clone(),run.packet.payload_limit,expected.len()).map_err(anyhow::Error::msg)?;
        writer.bind_acquisition(observed.clone().for_store_root(&root)?).map_err(anyhow::Error::msg)?;run.writer=Some(writer);
    }
    let generation=run.owners.admit(slot.0,ticket)?;
    let r=&mut run.requests[index];r.generation=generation;r.slot=Some(slot.0);r.origin=prompt.len()-1;r.acquisition=Some(observed);
    r.prepared.as_mut().unwrap().cached_tokens=Some(cached);
    localization::admitted(run,model,cache,slot,cached)?;
    Ok(())
}
pub(crate) fn require_owner(slot:SlotId,ticket:Option<&str>)->Result<()> {
    let global=lock();let run=global.as_ref().context("fresh capture absent")?;
    if run.requests.is_empty()&&ticket.is_none(){return Ok(())}
    let index=owner(run,slot)?;
    ensure!(Some(run.requests[index].ticket.as_str())==ticket,"stale/cross-request native decision owner");Ok(())
}
pub(crate) fn forward(slot:SlotId,start:usize,input:&[u32],route:&str,logits:&[f32],vocab:usize,hidden:Option<&[f32]>,width:usize)->Result<()> {
    let mut global=lock();let run=global.as_mut().context("fresh capture absent")?;
    if run.requests.is_empty() {return Ok(())} // Startup warmup is outside the enrolled HTTP workload.
    let index=owner(run,slot)?;
    ensure!(vocab>0&&logits.len()%vocab==0,"partial vocabulary observation");let rows=logits.len()/vocab;
    let r=&mut run.requests[index];r.routes.push(route.into());r.prepared.as_mut().context("forward before prepare")?.transactions.push(ForwardBoundary{
        start,input:input.to_vec(),logit_rows:rows,hidden_rows:if width==0{0}else{hidden.map_or(0,|h|h.len()/width)},route:route.into()});
    if run.packet.selected_cell!=Some([r.lane,r.turn]){return Ok(())}
    let canonical=r.canonical.as_ref().context("numerical selector absent")?;let h=hidden.context("actual normalized hidden absent")?;
    ensure!(width>0&&h.len()==rows*width&&rows<=input.len(),"target row/hidden extent differs");
    for row in 0..rows {let position=start+input.len()-rows+row;let Some(i)=position.checked_sub(r.origin) else{continue};
        if i>=canonical.decisions.len(){continue}let p=&r.prepared.as_ref().unwrap().prompt;
        if !input.iter().take(input.len()-rows+row+1).enumerate().all(|(offset,token)| {let at=start+offset;
            p.get(at).or_else(||at.checked_sub(p.len()).and_then(|j|canonical.decisions.get(j)))==Some(token)}){continue}
        r.pending.insert(i,Row{logits:logits[row*vocab..(row+1)*vocab].to_vec(),hidden:h[row*width..(row+1)*width].to_vec(),route:route.into()});
    }Ok(())
}
pub(crate) fn decision(slot:SlotId,index:usize,actual:u32)->Result<u32> {
    let mut global=lock();let run=global.as_mut().context("fresh capture absent")?;
    if run.requests.is_empty(){return Ok(actual)}
    let at=owner(run,slot)?;let r=&mut run.requests[at];
    ensure!(index==r.decisions.len(),"native decision index duplicate/gap");
    let selected=match &r.canonical {Some(c)=>*c.decisions.get(index).context("replay exceeded registered terminal extent")?,None=>actual};
    r.raw_decisions.push(actual);r.decisions.push(selected);
    if run.packet.selected_cell==Some([r.lane,r.turn]) {let row=r.pending.remove(&index).context("canonical row lacks complete consumed-prefix target observation")?;
        charge(row.logits.len().checked_add(row.hidden.len()).and_then(|n|n.checked_mul(4)).and_then(|n|n.checked_add(4)).context("numerical payload byte overflow")?)?;
        let writer=run.writer.as_mut().context("numerical writer absent")?;let at=|stage|trace::coordinate(&format!("agent-{}",r.lane),r.turn,r.origin+index,stage);
        trace::capture_f32(writer,at(Stage::Logits),&row.logits,vec![1,row.logits.len()])?;
        trace::capture_f32(writer,at(Stage::SpecVerifierHidden),&row.hidden,vec![1,row.hidden.len()])?;
        writer.capture(at(Stage::SpecCanonicalIds),Scalar::U32,vec![1],&selected as *const u32 as usize as u64,4,0,None,
            "registered fresh fixed-input frozen numerical selector",&selected.to_le_bytes()).map_err(anyhow::Error::msg)?;
        r.routes.push(row.route);
    }Ok(selected)
}
pub(crate) fn gpu_token(slot:SlotId,position:usize)->Result<Option<u32>> {
    let global=lock();let run=global.as_ref().context("fresh capture absent")?;
    if run.requests.is_empty(){return Ok(None)}
    let r=&run.requests[owner(run,slot)?];
    Ok(r.canonical.as_ref().and_then(|c|position.checked_sub(r.origin).and_then(|i|c.decisions.get(i).copied())))
}
pub(crate) fn gpu_selected(slot:SlotId) {let result=(||->Result<()>{let mut global=lock();let run=global.as_mut().context("fresh capture absent")?;
    let at=owner(run,slot)?;ensure!(run.packet.mode==Mode::Replay,"unforced acquisition attempted GPU selector");run.requests[at].gpu_selector=true;Ok(())})();if let Err(e)=result{fail("gpu-selector",e)}}
pub(crate) fn stopping(slot:SlotId,ticket:Option<&str>,token:u32,cause:&str)->Result<()> {
    require_owner(slot,ticket)?;let mut global=lock();let run=global.as_mut().unwrap();
    if run.requests.is_empty()&&ticket.is_none(){return Ok(())}
    let at=owner(run,slot)?;
    ensure!(run.requests[at].stop.is_none(),"duplicate native stopping selection");run.requests[at].stop=Some(reference::StopSelection{token,cause:cause.into()});Ok(())
}
pub(crate) fn published(state:&Qwen35DecodeState,count:usize)->Result<()> {
    let mut global=lock();let run=global.as_mut().context("fresh capture absent")?;
    if run.requests.is_empty()&&state.params.c2_ticket.is_none(){return Ok(())}
    let r=request(run,state.params.c2_ticket.as_deref().context("native publication lost request ticket")?)?;
    ensure!(r.terminal.as_ref().is_some_and(|t|t.slot==state.slot_id.0&&t.generated.len()==count)
        &&r.published.is_none(),"native completion publication lacks its unique terminal/owner");
    r.published=Some(count);Ok(())
}
pub(crate) fn finished(state:&Qwen35DecodeState) {
    let result=(||->Result<()> {let mut global=lock();let run=global.as_mut().context("fresh capture absent")?;
        if run.requests.is_empty()&&state.params.c2_ticket.is_none(){return Ok(())}
        let ticket=state.params.c2_ticket.as_deref().context("native terminal lost request ticket")?;
        let r=request(run,ticket)?;
        let terminal=Terminal{request_sha256:hash(&r.raw),slot:state.slot_id.0,decisions:r.decisions.clone(),generated:state.generated_tokens.clone(),finish_reason:state.finish_reason.into(),native_step:state.step};
        if let Some(old)=&r.terminal {ensure!(old==&terminal,"duplicate native terminal changed");return Ok(())}
        let index=owner(run,state.slot_id)?;ensure!(run.requests[index].ticket==ticket,"native terminal belongs to another slot generation");
        let r=&mut run.requests[index];if let Some(c)=&r.canonical {ensure!(terminal.decisions==c.decisions&&terminal.generated==c.generated&&terminal.finish_reason==c.finish_reason,"frozen numerical replay changed terminal tape");}
        let generation=r.generation;r.terminal=Some(terminal);
        localization::finished(run,state.slot_id)?;
        run.owners.finish(state.slot_id.0,ticket,generation)?;Ok(())})();if let Err(e)=result{fail("native-terminal",e)}
}
pub(crate) fn argmax(slot:SlotId,position:usize,token:u32) {
    let result=(||->Result<()> {let mut global=lock();let run=global.as_mut().context("fresh capture absent")?;
        if run.requests.is_empty(){return Ok(())}let at=owner(run,slot)?;
        run.requests[at].gpu_argmax.push((position,token));Ok(())})();if let Err(e)=result{fail("native-argmax",e)}
}
pub(crate) fn route(slot:SlotId,route:&str) {
    let result=(||->Result<()> {let mut global=lock();let run=global.as_mut().context("fresh capture absent")?;
        if run.requests.is_empty(){return Ok(())}let at=owner(run,slot)?;
        run.requests[at].routes.push(route.into());Ok(())})();if let Err(e)=result{fail("native-route",e)}
}
fn driver(requests:Vec<Vec<Vec<u8>>>,reference:Option<reference::Validated>,stop:Arc<AtomicBool>)->Result<()> {
    let requests=requests.into_iter().map(|lane|lane.into_iter().map(axum::body::Bytes::from).collect::<Vec<_>>()).collect::<Vec<_>>();
    let (port,timeout,output,selected,cold,prefix)={let global=lock();let run=global.as_ref().unwrap();(run.packet.port,run.packet.ready_timeout_seconds,run.packet.output.clone(),run.packet.selected_cell,run.packet.arm=="cold",run.prefix.clone())};
    let client=reqwest::blocking::Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none()).timeout(Duration::from_secs(1800)).build()?;
    let base=format!("http://127.0.0.1:{port}");let start=Instant::now();
    loop {ensure!(!stop.load(Ordering::Acquire),"server exited before readiness");if let Ok(r)=client.get(format!("{base}/readyz")).send(){if r.status().is_success()&&r.json::<Value>()?["ready"]==true{break}}
        ensure!(start.elapsed()<Duration::from_secs(timeout),"real server readiness deadline");std::thread::sleep(Duration::from_millis(100));}
    let health:Value=client.get(format!("{base}/health")).send()?.error_for_status()?.json()?;
    let runtime:Value=client.get(format!("{base}/hf2q/v1/runtime")).send()?.error_for_status()?.json()?;
    let initial:Value=serde_json::from_slice(&requests[0][0])?;
    ensure!(health["context_length"]==262144&&health["model"]==initial["model"]&&runtime["serve"]["max_slots"]==4
        &&runtime["measurement_snapshot"]["active_controls"]["vision_projector"]==false,
        "actual canonical full-context/text-only readiness differs");
    save(&output.join("readiness.json"),&serde_json::json!({"health":health,"runtime":runtime}))?;
    let last=selected.map_or_else(||requests.iter().map(Vec::len).max().unwrap()-1,|cell|cell[1]);let mut lanes:Vec<Vec<acquisition::FileBinding>>=vec![Vec::new();4];let mut nonce=None;
    for turn in 0..=last {if cold&&turn!=last {continue}
        {let mut global=lock();let run=global.as_mut().unwrap();ensure!(run.owners.active.is_empty()&&run.requests.iter().all(|r|r.terminal.is_some()&&r.wire.is_some()),"previous wave incomplete");run.requests.clear();
            for lane in 0..4 {let Some(raw)=requests[lane].get(turn).cloned() else{continue};let req:ChatCompletionRequest=serde_json::from_slice(&raw)?;let dir=output.join(format!("agent-{lane}-turn-{turn}"));fs::create_dir(&dir)?;
                save_bytes(&dir.join("request.json"),&raw)?;
                run.requests.push(Request{lane,turn,ticket:format!("{prefix}-{lane}-{turn}"),raw,tools:req.tools.unwrap_or_default(),stream:req.stream.unwrap_or(false),dir,
                    ingress:false,prepared:None,terminal:None,stop:None,slot:None,generation:0,acquisition:None,decisions:Vec::new(),raw_decisions:Vec::new(),
                    canonical:reference.as_ref().map(|v|v.observations[lane][turn].terminal.clone()),origin:0,pending:BTreeMap::new(),routes:Vec::new(),gpu_selector:false,gpu_argmax:Vec::new(),wire:None,published:None});
            }}
        let wave=requests.iter().enumerate().filter_map(|(lane,turns)|turns.get(turn).map(|raw|(lane,raw.clone()))).collect::<Vec<_>>();
        let barrier=Arc::new(std::sync::Barrier::new(wave.len()));let mut jobs=Vec::new();
        for (lane,raw) in wave {let client=client.clone();let url=format!("{base}/v1/chat/completions");let ticket=format!("{prefix}-{lane}-{turn}");let barrier=barrier.clone();
            jobs.push((lane,std::thread::spawn(move||->Result<Vec<u8>>{barrier.wait();let response=client.post(url).header("content-type","application/json").header(HEADER,ticket).body(raw).send()?;
                ensure!(response.status().is_success(),"fresh fixed-input HTTP request rejected: {}",response.status());let mut bytes=Vec::new();response.take(64*1024*1024+1).read_to_end(&mut bytes)?;ensure!(bytes.len()<=64*1024*1024,"raw response exceeded bound");Ok(bytes)})));}
        for (lane,job) in jobs {let response=job.join().map_err(|_|anyhow::anyhow!("fixed-input client panicked"))??;
            let mut global=lock();let run=global.as_mut().unwrap();ensure!(run.error.is_none(),"fresh acquisition failed: {:?}",run.error);
            let r=run.requests.iter().find(|r|r.lane==lane).context("joined HTTP lane not enrolled")?;let (status,headers,sha,bytes)=r.wire.as_ref().context("server HTTP body incomplete")?;
            ensure!(*status==200&&sha==&hash(&response)&&*bytes==response.len(),"raw HTTP endpoint bodies differ");
            let response_binding=save_bytes(&r.dir.join("response.bin"),&response)?;let headers_binding=save(&r.dir.join("response-headers.json"),headers)?;
            let wire=super::original_witness::wire_terminal(&response,r.stream)?;
            let observation=reference::Observation{ticket:r.ticket.clone(),generation:r.generation,prepared:r.prepared.clone().context("actual prepare missing")?,terminal:r.terminal.clone().context("native terminal missing")?,
                stop_selection:r.stop.clone(),request:acquisition::FileBinding{path:r.dir.join("request.json"),sha256:hash(&r.raw)},response:response_binding,response_headers:headers_binding,
                server_request:binding(&r.dir.join("server-request.bin"))?,request_headers:binding(&r.dir.join("request-headers.json"))?,
                server_response:binding(&r.dir.join("server-response.bin"))?,
                server_response_sha256:sha.clone(),server_response_bytes:*bytes,http_finish_reason:wire.finish_reason,completion_counter:"generated_tokens".into(),published_completion_tokens:r.published.context("actual native completion publication missing")?,
                acquisition:r.acquisition.clone().context("actual opened-model provenance missing")?,actual_routes:r.routes.clone(),raw_decisions:r.raw_decisions.clone(),gpu_frozen_selector_used:r.gpu_selector,actual_gpu_argmax:r.gpu_argmax.clone()};
            drop(response);
            observation.validate(&r.raw,r.stream,&run.packet.identity,run.packet.mode==Mode::Capture)?;
            if let Some(v)=&reference {let p=&v.observations[lane][turn].prepared;
                ensure!(observation.prepared.prompt==p.prompt&&serde_json::to_vec(&observation.prepared.params)?==serde_json::to_vec(&p.params)?&&observation.prepared.registration_architecture==p.registration_architecture,"registered preparation/settings changed");}
            if selected==Some([lane,turn]) {
                if run.packet.route==Route::CurrentDefault {
                    ensure!(observation.actual_routes.iter().any(|route|route=="DFlash2-actual-runner-completed"),
                        "selected default replay did not exercise the actual DFlash2 runner; ordinary fallback is not default evidence");
                }
                let cached=observation.prepared.cached_tokens.unwrap();
                if cold {ensure!(cached==0,"cold replay reused history");}
                if !cold&&run.packet.keep==4&&matches!(turn,3|5){ensure!(cached>0,"resumed divergent replay did not restore older boundary");}
            }
            if let Some(old)=&nonce {ensure!(old==&observation.acquisition.nonce,"mixed acquisition nonce");}else{nonce=Some(observation.acquisition.nonce.clone());}
            lanes[lane].push(save(&r.dir.join("observation.json"),&observation)?);
        }
    }
    {let mut global=lock();let run=global.as_mut().unwrap();if let Some(writer)=run.writer.take(){super::divergent::complete(writer)?;}else{ensure!(selected.is_none(),"selected replay store absent");}
        localization::complete(run)?;
        let table=run.table.as_ref().map(|t|save(&output.join("token-table.json"),t.values.as_ref())).transpose()?;
        if run.packet.mode==Mode::Capture {let captured=reference::Capture{schema:reference::CAPTURE_SCHEMA.into(),inputs:run.packet.inputs.clone(),owner_decision:run.packet.owner_decision.clone(),identity:run.packet.identity.clone(),
            packet:acquisition::FileBinding{path:output.join("packet.json"),sha256:hash(&fs::read(output.join("packet.json"))?)},token_table:table,
            lanes:lanes.into_iter().enumerate().map(|(lane,observations)|reference::Lane{lane,observations}).collect(),reference_route:"fresh-fixed-input-HTTP-OFF-unforced-K4".into(),
            prefill_chunk:crate::serve::api::engine::qwen35_slot_prefill_chunk_tokens(4) as usize,acquisition_nonce:nonce.context("no acquisition nonce")?,complete:true,unforced:true,historical_native_identity_recovered:false,whole_c2_pass:false};
            ensure!(captured.lanes.iter().enumerate().all(|(lane,observed)|observed.observations.len()==requests[lane].len()),
                "capture omitted independently declared per-lane turns");
            save(&output.join("capture.json"),&captured)?;
        } else {save(&output.join("replay.json"),&serde_json::json!({"schema":"hf2q.c2.fresh-fixed-input-replay.v1","registration":run.packet.registration,
            "selected_cell":selected,"arm":run.packet.arm,"route":run.packet.route,"lanes":lanes,"acquisition_nonce":nonce,"teacher_forced_diagnostic":true,"whole_c2_pass":false}))?;}}
    client.post(format!("{base}/shutdown")).send()?.error_for_status()?;Ok(())
}
pub(super) fn run(family:Family)->Result<()> {
    use clap::Parser;
    let path=PathBuf::from(std::env::var_os("HF2Q_C2_FRESH_HTTP_PACKET").context("fresh HTTP packet missing")?);
    let raw=fs::read(path)?;ensure!(raw.len()<=4*1024*1024,"fresh packet exceeds4MiB");let packet:Packet=serde_json::from_slice(&raw)?;
    ensure!(packet.schema=="hf2q.c2.fresh-fixed-input-http-packet.v1"&&packet.execution_approved&&packet.family==family&&packet.identity.family==family,"fresh packet approval/schema/family");
    reference::validate_current_identity(&packet.identity)?;
    let _:Value=serde_json::from_slice(&acquisition::checked_file(&packet.owner_decision,4*1024*1024)?)?;
    ensure!(packet.identity.model_sha256==if family==Family::Qwen38{Q38}else{Q36},"wrong declared artifact");
    let input_raw=acquisition::checked_file(&packet.inputs,4*1024*1024)?;let inputs:reference::Inputs=serde_json::from_slice(&input_raw)?;let requests=inputs.validate(family)?;
    let localization=packet.localization.as_ref().map(|binding|localization::Session::prepare(binding,&packet)).transpose()?;
    let (scope,reference)=match packet.mode {
        Mode::Capture=>{ensure!(packet.registration.is_none()&&packet.selected_cell.is_none()&&packet.expected.is_none()&&packet.arm=="canonical"&&packet.route==Route::Off&&packet.keep==4,"capture must be complete unforced OFF/K4 with no selector");
            (acquisition::Scope::begin_c2_fresh_capture(&packet.acquisition,&packet.identity,&packet.model,&packet.output,&raw,&input_raw)?,None)},
        Mode::Replay=>{let binding=packet.registration.as_ref().context("independent fresh registration missing")?;
            ensure!(packet.selected_cell.is_some()&&matches!(packet.arm.as_str(),"cold"|"resumed"),"replay cell/arm missing");
            let (scope,reference)=acquisition::Scope::begin_c2_registered_replay(&packet.acquisition,&packet.identity,&packet.model,&packet.output,&raw,binding)?;
            ensure!(reference.capture.inputs.sha256==packet.inputs.sha256,"replay substituted fixed inputs");
            let [lane,turn]=packet.selected_cell.unwrap();ensure!(lane<4&&turn<reference.requests[lane].len(),"replay outside complete per-lane reference");(scope,Some(reference))},
    };
    ensure!(packet.port>=1024&&packet.ready_timeout_seconds>0&&packet.ready_timeout_seconds<=3600&&matches!(packet.keep,1|4)
        &&packet.payload_limit>0&&packet.payload_limit<=8*1024*1024*1024*1024,"bounded port/depth/time/disk grant required");
    let reservation=std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST,packet.port))?;
    let cli=crate::cli::Cli::try_parse_from(&packet.server_argv)?;let crate::cli::Command::Serve(args)=cli.command else{anyhow::bail!("actual production Serve CLI required")};
    ensure!(args.model.as_ref()==Some(&packet.model)&&args.host.as_deref()==Some("127.0.0.1")&&args.port==Some(packet.port)&&args.planning.ctx==Some(262144)
        &&args.planning.max_slots==Some(4)&&args.planning.scheduler==Some(crate::cli::SchedulerArg::InflightBatched),"actual full262144x4 production server required");
    ensure!(args.behavior.default_repetition_penalty==Some(1.05)
        &&args.behavior.default_thinking_token_budget==Some(2048)
        &&args.behavior.default_tool_thinking_token_budget==Some(512),
        "actual canonical launcher sampling/thinking settings differ");
    let case:Value=serde_json::from_slice(&acquisition::checked_file(&inputs.case,4*1024*1024)?)?;
    let quota=case["workload"]["anchor_bytes_per_slot_max"].as_u64().context("anchor quota missing")?;
    ensure!(quota%5==0&&std::env::var("HF2Q_QWEN_ANCHOR_DEPTH")?==packet.keep.to_string()
        &&std::env::var("HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX")?==(quota/5*(packet.keep as u64+1)).to_string()
        &&args.planning.kv_cache_budget.as_deref()==case["serve"]["env"]["KV_CACHE_BUDGET_BYTES"].as_str(),"source-bound KV/anchor allocation differs");
    match packet.route {Route::Off=>ensure!(std::env::var("HF2Q_DFLASH2")?=="0"&&std::env::var("HF2Q_QWEN_SPECULATION")?=="off","OFF route changed"),
        Route::MtpAuto=>ensure!(std::env::var("HF2Q_DFLASH2")?=="0"&&std::env::var("HF2Q_QWEN_SPECULATION")?=="auto","MTP route changed"),
        Route::CurrentDefault=>{
            ensure!(std::env::var("HF2Q_DFLASH2").unwrap_or_default()!="0","default disabled");
            for name in ["HF2Q_DFLASH2_CONFIG","HF2Q_DFLASH2_WEIGHTS"] {
                let path=PathBuf::from(std::env::var_os(name).context("default artifact missing")?);
                ensure!(path.is_absolute()&&path.is_file(),"published default artifact missing");
            }
        }
    }
    let _disk=DiskBudget::begin(packet.payload_limit)?;
    output_preflight(&packet.output)?;fs::create_dir(&packet.output)?;fs::set_permissions(&packet.output,fs::Permissions::from_mode(0o700))?;save_bytes(&packet.output.join("packet.json"),&raw)?;
    let mut random=[0u8;32];fs::File::open("/dev/urandom")?.read_exact(&mut random)?;
    let _gpu=crate::inference::hf2q_gpu_test_lock();scope.c2_hold()?;
    {let mut global=lock();ensure!(global.is_none()&&!super::fullw_http::historical_enabled(),"another HTTP acquisition active");*global=Some(Run{packet,requests:Vec::new(),owners:Owners::default(),
        prefix:hex::encode(random),architecture:String::new(),table:None,worker:None,writer:None,error:None,localization});}
    let stop=Arc::new(AtomicBool::new(false));let signal=stop.clone();drop(reservation);
    let worker=std::thread::spawn(move||{let result=driver(requests,reference,signal);if result.is_err(){unsafe{libc::raise(libc::SIGTERM);}}result});
    let served=crate::serve::cmd_serve(args,cli.log_format,None,None);stop.store(true,Ordering::Release);
    let captured=worker.join().map_err(|_|anyhow::anyhow!("fresh HTTP driver panicked")).and_then(|result|result);
    let release=acquisition::Scope::c2_release();
    let mut retained=lock().take();
    let error=retained.as_ref().and_then(|run|run.error.clone());
    let failed=served.is_err()||captured.is_err()||release.is_err()||error.is_some();
    let diagnostic_retention=if failed {
        let reason=format!("serve={:?}; capture={:?}; release={:?}; observation={error:?}",
            served.as_ref().err(),captured.as_ref().err(),release.as_ref().err());
        retained.as_mut().map(|run|localization::failed(run,&reason)).transpose()
    }else{Ok(None)};
    served?;captured?;ensure!(error.is_none(),"late native/HTTP acquisition error: {error:?}");
    diagnostic_retention?;release
}

#[test]
fn fresh_owner_rejects_late_completion_after_slot_reuse() {
    let mut owners=Owners::default();
    let first=owners.admit(2,"request-a").unwrap();
    assert!(owners.admit(2,"request-b").is_err());
    owners.finish(2,"request-a",first).unwrap();
    let second=owners.admit(2,"request-b").unwrap();
    assert_ne!(first,second);
    assert!(owners.finish(2,"request-a",first).is_err());
    assert_eq!(owners.active.get(&2),Some(&(String::from("request-b"),second)));
    assert!(owners.finish(2,"request-b",first).is_err());
    owners.finish(2,"request-b",second).unwrap();
    assert!(owners.active.is_empty());
}
