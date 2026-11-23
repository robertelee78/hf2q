//! Separately authorized diagnostics, never an original-route qualification.
//! Snapshot blits can split compute encoders. Interpretation requires exact
//! all-60 endpoint equality against the saved original same-K acquisition.
use super::*;
use crate::inference::models::qwen35::{c23_layer_capture as layers, kv_cache::{FullAttnKvSlot, HybridKvSlotAnchor}};
use mlx_native::MlxBuffer;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    schema:String, execution_approved:bool, operator_authorization:acquisition::FileBinding,
    selected_cell:[usize;2], creation_cell:[usize;2], first_row:usize, end_row_exclusive:usize,
    layer_expected:acquisition::FileBinding, anchor_expected:acquisition::FileBinding, payload_limit:u64,
}
pub(super) struct Session {
    packet:Packet, expected:Vec<ExpectedTensor>, layer_expected:Vec<ExpectedTensor>, store:Option<Store>,
    creation:Option<Creation>, restored:Option<Value>, installed:bool, layer_result:Option<Value>,
    lifecycle:Vec<Value>, extent_validated:bool, staged:Option<Value>, restore_ticket:Option<Value>,
}
struct Creation { prefix:Vec<u32>, cursors:Vec<u32>, image:Vec<u32>, kv:Vec<Vec<u8>>, owner:Value, identity:(usize,u64,u32) }
fn planned_bytes(expected:&[ExpectedTensor])->Result<usize> {
    expected.iter().try_fold(0usize,|n,e|e.shape.iter().try_fold(e.scalar.bytes(),|n,d|n.checked_mul(*d))
        .and_then(|bytes|n.checked_add(bytes)).context("localization extent overflow"))
}
impl Session {
    pub(super) fn prepare(binding:&acquisition::FileBinding, outer:&super::Packet)->Result<Self> {
        let packet:Packet=serde_json::from_slice(&acquisition::checked_file(binding,1024*1024)?)?;
        ensure!(packet.schema=="hf2q.c23.localization.v1"&&packet.execution_approved,"localization approval/schema missing");
        ensure!(outer.mode==Mode::Replay&&outer.family==Family::Qwen38&&outer.route==Route::Off
            &&outer.arm=="resumed"&&outer.selected_cell==Some([2,3])&&matches!(outer.keep,1|4),"localization requires separate C23 OFF replay");
        ensure!(packet.selected_cell==[2,3]&&packet.creation_cell==[2,1]&&packet.first_row==914&&packet.end_row_exclusive==1923,
            "localization logical extent changed");
        ensure!(packet.payload_limit>0&&packet.payload_limit<=outer.payload_limit,"localization grant must fit unchanged fresh diagnostic disk grant");
        ensure!(packet.operator_authorization.sha256!=outer.owner_decision.sha256,"old fresh authority does not authorize localization");
        let auth:Value=serde_json::from_slice(&acquisition::checked_file(&packet.operator_authorization,1024*1024)?)?;
        ensure!(auth["schema"]=="hf2q.private.c23-separate-localization-authorization.v1"
            &&auth["answer"]["localization_authority"]=="authorize_separate_max4"
            &&auth["maximum_additional_model_loads"]==4&&auth["retries"]==0&&auth["warmups"]==0
            &&auth["original_C23_model_budget"]=="PERMANENTLY_CLOSED4/4"
            &&auth["original_failure_preserved"]==true&&auth["requires_own_calibrations"]==true
            &&auth["requires_route_preservation_proof_before_execution"]==true
            &&auth["all_safety_guards_and_release_gates_unchanged"]==true&&auth["frozen_D_limit"]==0.25
            &&auth["full_frozen_trajectory_and_all_60_decision_rows_required"]==true
            &&auth["budget_reset_authorized"]==false&&auth["relabel_original_failure_authorized"]==false
            &&auth["release_qualified"]==false,"separate four-load authorization missing");
        for name in ["HF2Q_DUMP_LAYER","HF2Q_DUMP_BISECT"] {ensure!(std::env::var_os(name).is_none(),"localization cannot use route-changing dump controls");}
        let expected:Vec<ExpectedTensor>=serde_json::from_slice(&acquisition::checked_file(&packet.anchor_expected,4*1024*1024)?)?;
        let layer_expected:Vec<ExpectedTensor>=serde_json::from_slice(&acquisition::checked_file(&packet.layer_expected,4*1024*1024)?)?;
        ensure!(!expected.is_empty()&&expected.len()<=2048&&!layer_expected.is_empty()&&layer_expected.len()<=2048
            &&planned_bytes(&expected)?.checked_add(planned_bytes(&layer_expected)?).is_some_and(|n|n<=packet.payload_limit as usize),
            "combined anchor/layer extent outside grant");
        Ok(Self{packet,expected,layer_expected,store:None,creation:None,restored:None,installed:false,layer_result:None,lifecycle:Vec::new(),extent_validated:false,staged:None,restore_ticket:None})
    }
    fn event(&mut self,value:Value)->Result<()> {ensure!(self.lifecycle.len()<256,"localization lifecycle bound exceeded");self.lifecycle.push(value);Ok(())}
}
fn owner_value(r:&Request)->Value {serde_json::json!({"ticket":r.ticket,"slot":r.slot,"generation":r.generation,"lane":r.lane,"turn":r.turn})}
fn at(boundary:u64,segment:u32)->Coordinate {super::super::localization_extent::anchor_coordinate(boundary,segment)}
fn spec(boundary:u64,segment:u32,scalar:Scalar,len:usize)->ExpectedTensor {ExpectedTensor{at:at(boundary,segment),scalar,shape:vec![len]}}
fn full_buffers(layer:&FullAttnKvSlot)->Vec<&MlxBuffer> {
    let mut buffers=Vec::new();if let Some(tq)=&layer.tq {buffers.extend([&tq.k_packed,&tq.k_norms,&tq.v_packed,&tq.v_norms]);}
    buffers.extend([layer.k.as_ref(),layer.v.as_ref()].into_iter().flatten());buffers
}
fn region_lengths(cache:&HybridKvCache)->Result<Vec<usize>> {
    let mut lengths=Vec::new();
    for (layer_index,layer) in cache.full_attn.iter().chain(cache.mtp_slot.iter()).enumerate() {for buffer in full_buffers(layer) {
        let valid=if layer_index==cache.full_attn.len(){0}else{914};
        let s=buffer.shape();ensure!(s.len()==4&&s[0]==cache.n_seqs as usize&&s[2]>=914,"anchor KV geometry");
        lengths.push(s[1].checked_mul(valid).and_then(|n|n.checked_mul(s[3])).and_then(|n|n.checked_mul(buffer.dtype().size_of())).context("anchor KV extent overflow")?);
    }}Ok(lengths)
}
fn image_words(cache:&HybridKvCache)->Result<usize> {
    cache.linear_attn.iter().try_fold(0usize,|total,l| {
        let s=l.conv_state.shape();ensure!(s.len()==3&&s[2]==cache.n_seqs as usize,"anchor convolution geometry");
        let conv=s[0].checked_mul(s[1]).context("anchor convolution overflow")?;
        ensure!(l.recurrent.element_count()%cache.n_seqs as usize==0,"anchor recurrent slot geometry");
        total.checked_add(l.recurrent.element_count()/cache.n_seqs as usize).and_then(|n|n.checked_add(conv)).context("anchor image overflow")
    })
}
fn runtime_geometry(cache:&HybridKvCache,keep:usize)->Result<Vec<ExpectedTensor>> {
    let cursors=cache.full_attn.len()+usize::from(cache.mtp_slot.is_some());let kv=region_lengths(cache)?;
    let mut out=vec![spec(0,0,Scalar::U32,cursors),spec(0,u32::MAX-1,Scalar::U32,image_words(cache)?),spec(0,u32::MAX,Scalar::U32,914)];
    out.extend(kv.iter().enumerate().filter(|(_,n)|**n>0).map(|(i,n)|spec(0,i as u32+2,Scalar::U8,*n)));
    if keep==4 {
        out.push(spec(1,0,Scalar::U32,cursors));out.push(spec(1,1,Scalar::U8,cache.linear_attn.len()));
        out.extend(kv.iter().enumerate().filter(|(_,n)|**n>0).map(|(i,n)|spec(1,i as u32+2,Scalar::U8,*n)));
        let mut segment=kv.len() as u32+2;
        for l in &cache.linear_attn {let s=l.conv_state.shape();
            out.push(spec(1,segment,Scalar::U8,s[0]*s[1]*4));segment+=1;
            out.push(spec(1,segment,Scalar::U8,l.recurrent.element_count()/cache.n_seqs as usize*4));segment+=1;
        }
        out.push(spec(1,u32::MAX,Scalar::U32,914));
    }Ok(out)
}
fn capture(store:&mut Store,boundary:u64,segment:u32,scalar:Scalar,bytes:&[u8])->Result<()> {
    store.capture(at(boundary,segment),scalar,vec![bytes.len()/scalar.bytes()],bytes.as_ptr() as usize as u64,bytes.len() as u64,0,None,
        "actual completed anchor snapshot/restore; existing checked native retirement; no added wait",bytes).map_err(anyhow::Error::msg)
}
fn words(values:&[u32])->Vec<u8> {values.iter().flat_map(|v|v.to_le_bytes()).collect()}
fn kv_read(cache:&HybridKvCache,slot:SlotId)->Result<Vec<Vec<u8>>> {
    cache.full_attn.iter().enumerate().map(|(_,l)|(l,914)).chain(cache.mtp_slot.iter().map(|l|(l,0)))
        .flat_map(|(l,valid)|full_buffers(l).into_iter().map(move|b|(b,valid))).map(|(b,valid)|trace::rows(b,slot,valid)).collect()
}
/// Called after the checked forward returns its real folded checkpoint. The
/// live cache may be later than914: only append-only full-KV prefix rows are
/// read there. Linear bytes come from the actual retained image, not live state.
pub(crate) fn anchor_created(cache:&HybridKvCache,slot:SlotId,anchor:&HybridKvSlotAnchor,prefix:&[u32]) {
    let result=(||->Result<()> {let mut global=lock();let Some(run)=global.as_mut() else{return Ok(())};
        if run.localization.is_none()||prefix.len()!=914{return Ok(())}
        let index=owner(run,slot)?;let r=&run.requests[index];
        let session=run.localization.as_ref().unwrap();if [r.lane,r.turn]!=session.packet.creation_cell{return Ok(())}
        ensure!(session.creation.is_none(),"duplicate creation914 witness");
        ensure!(session.extent_validated,"header-only independent extents not validated");
        let declared=session.expected.clone();
        super::super::divergent::require_complete(runtime_geometry(cache,run.packet.keep)?,&declared)?;
        let root=run.packet.output.join("localization-anchor");
        // Charge the entire independently declared raw extent before CPU mapping.
        charge(planned_bytes(&declared)?.checked_add(crate::adr063_invariance::store::METADATA_BYTES as usize).context("anchor disk grant overflow")?)?;
        let mut store=Store::new(&root,run.packet.identity.clone(),"C23-separate-folded-anchor-creation-and-actual-restore".into(),declared.clone(),session.packet.payload_limit,declared.len()).map_err(anyhow::Error::msg)?;
        store.bind_acquisition(r.acquisition.as_ref().context("creation lacks actual owning acquisition")?.clone().for_store_root(&root)?).map_err(anyhow::Error::msg)?;
        let (len,generation,graft,cursors,mtp,image)=anchor.test_words();
        ensure!(len==914&&graft==0&&cursors.iter().all(|n|*n==914)&&mtp.is_none_or(|n|n==0)&&image.len()==image_words(cache)?,"actual folded anchor extent differs");
        let cursors:Vec<_>=cursors.into_iter().chain(mtp).collect();let kv=kv_read(cache,slot)?;
        let who=owner_value(r);let session=run.localization.as_mut().unwrap();
        session.store=Some(store);
        let store=session.store.as_mut().unwrap();
        capture(store,0,0,Scalar::U32,&words(&cursors))?;
        for (i,bytes) in kv.iter().enumerate(){if !bytes.is_empty(){capture(store,0,i as u32+2,Scalar::U8,bytes)?;}}
        capture(store,0,u32::MAX-1,Scalar::U32,&words(&image))?;
        capture(store,0,u32::MAX,Scalar::U32,&words(prefix))?;
        session.creation=Some(Creation{prefix:prefix.to_vec(),cursors,image,kv,owner:who.clone(),identity:(len,generation,graft)});
        let layout=anchor_layout(cache)?;
        save(&run.packet.output.join("localization-anchor-layout.json"),&layout)?;
        session.event(serde_json::json!({"event":"folded-anchor-created","owner":who,"anchor_state_generation":generation,"graft_len":graft,"creation_live_parity":null,"boundary":"actual retained folded image; append-only fullKV prefix after checked forward"}))?;Ok(())})();
    if let Err(error)=result{fail("localization-anchor-creation",error)}
}
pub(crate) fn anchor_staged(slot:usize,origin:u64,epoch:u64,id:u64,request_id:u64,prefix:usize) {
    let result=(||->Result<()> {let mut global=lock();let Some(run)=global.as_mut() else{return Ok(())};
        if run.localization.is_none()||prefix!=914{return Ok(())}
        let slot=SlotId(u32::try_from(slot)?);let r=&run.requests[owner(run,slot)?];let who=owner_value(r);
        let session=run.localization.as_mut().unwrap();if [r.lane,r.turn]!=session.packet.creation_cell{return Ok(())}
        ensure!(session.staged.is_none()&&session.creation.as_ref().is_some_and(|c|c.owner==who),"staged914 missing matching actual creation owner");
        let event=serde_json::json!({"event":"anchor-staged","owner":who,"store_origin":origin,"store_epoch":epoch,"anchor_id":id,"anchor_request":request_id});
        session.staged=Some(event.clone());session.event(event)?;Ok(())})();
    if let Err(error)=result{fail("localization-anchor-stage",error)}
}
pub(crate) fn anchor_restore_ticket(slot:SlotId,origin:u64,epoch:u64,id:u64,operation:u64) {
    let result=(||->Result<()> {let mut global=lock();let Some(run)=global.as_mut() else{return Ok(())};let Some(session)=run.localization.as_mut() else{return Ok(())};
        if !run.requests.iter().any(|r|[r.lane,r.turn]==session.packet.selected_cell){return Ok(())}
        let Some(staged)=&session.staged else{return Ok(())};if staged["store_origin"]!=origin{return Ok(())}
        ensure!(session.restore_ticket.is_none()&&staged["anchor_id"]==id,"selected anchor ticket lineage changed");
        let event=serde_json::json!({"event":"anchor-restore-ticket-completed","slot":slot.0,"store_origin":origin,"ticket_epoch":epoch,"anchor_id":id,"operation":operation});
        session.restore_ticket=Some(event.clone());session.event(event)?;Ok(())})();
    if let Err(error)=result{fail("localization-anchor-restore-ticket",error)}
}
/// Restoration happens before ordinary admission binds the incoming generation.
/// Retain its actual authenticated ticket now; admitted() completes the join.
pub(in crate::serve::api) fn anchor_restored(cache:&HybridKvCache,slot:SlotId,anchor:&Qwen35Anchor,params:&SamplingParams) {
    let result=(||->Result<()> {let mut global=lock();let Some(run)=global.as_mut() else{return Ok(())};if run.localization.is_none(){return Ok(())}
        let r=request(run,params.c2_ticket.as_deref().context("restore lost authenticated ticket")?)?;
        if [r.lane,r.turn]!=[2,3]{return Ok(())}let ticket=r.ticket.clone();
        let session=run.localization.as_mut().unwrap();ensure!(session.restored.is_none(),"duplicate selected restoration");
        let creation=session.creation.as_ref().context("restore914 has no actual creation witness")?;
        let staged=session.staged.as_ref().context("restoration missing actual staged lineage")?;
        let transaction=session.restore_ticket.as_ref().context("restoration missing successful checked ticket")?;
        ensure!(staged["anchor_id"]==anchor.id&&staged["anchor_request"]==anchor.request
            &&transaction["slot"]==slot.0&&transaction["anchor_id"]==anchor.id,"restoration ticket/staging ownership mismatch");
        let (length,generation,graft,saved_cursors,saved_mtp,image)=anchor.kv.test_words();
        ensure!(anchor.prompt_tokens==creation.prefix&&length==914,"restoration changed anchor prefix");
        ensure!((length,generation,graft)==creation.identity&&image==creation.image
            &&saved_cursors.iter().copied().chain(saved_mtp).eq(creation.cursors.iter().copied()),
            "restoration changed physical anchor lineage/image");
        drop(image);drop(saved_cursors);
        let state=trace::State::read(cache,slot)?;let cursors:Vec<_>=state.target.iter().copied().chain(state.mtp).collect();
        let store=session.store.as_mut().context("anchor store missing")?;
        capture(store,1,0,Scalar::U32,&words(&cursors))?;capture(store,1,1,Scalar::U8,&state.parity.iter().map(|p|u8::from(*p)).collect::<Vec<_>>())?;
        for (i,bytes) in state.regions.iter().enumerate(){if !bytes.is_empty(){capture(store,1,i as u32+2,Scalar::U8,bytes)?;}}
        capture(store,1,u32::MAX,Scalar::U32,&words(&anchor.prompt_tokens))?;
        let kv_count=creation.kv.len();ensure!(state.regions.len()==kv_count+cache.linear_attn.len()*2,"restored semantic region count");
        let mut offset=0usize;let mut linear_equal=true;
        for (i,l) in cache.linear_attn.iter().enumerate(){let recurrent=l.recurrent.element_count()/cache.n_seqs as usize;let s=l.conv_state.shape();let count=s[0]*s[1];
            let recurrent_bytes=words(&creation.image[offset..offset+recurrent]);offset+=recurrent;
            let conv=conv_to_live(&creation.image[offset..offset+count],s[0],s[1])?;offset+=count;
            linear_equal&=state.regions[kv_count+i*2]==words(&conv)&&state.regions[kv_count+i*2+1]==recurrent_bytes;
        }
        let kv_equal=state.regions[..kv_count]==creation.kv;let cursors_equal=cursors==creation.cursors;
        // A mismatch is evidence, not an instrumentation failure or a pass.
        session.restored=Some(serde_json::json!({"ticket":ticket,"slot":slot.0,"pending_admission_join":true,"anchor_id":anchor.id,"anchor_request":anchor.request,
            "anchor_state_generation":generation,"graft_len":graft,"raw_kv_equal":kv_equal,"semantic_linear_equal":linear_equal,"cursors_equal":cursors_equal,
            "actual_restored_parity":state.parity,"creation_live_parity":null,"complete_parity_pass":false,
            "conv_mapping":"la_restore.rs: image[k*channels+channel] -> live[channel*(K-1)+k]; recurrent copied unchanged"}));Ok(())})();
    if let Err(error)=result{fail("localization-anchor-restore",error)}
}
fn conv_to_live(image:&[u32],channels:usize,k:usize)->Result<Vec<u32>> {
    ensure!(channels.checked_mul(k)==Some(image.len()),"convolution image extent mismatch");let mut out=vec![0;image.len()];
    for channel in 0..channels {for row in 0..k {out[channel*k+row]=image[row*channels+channel];}}Ok(out)
}
pub(super) fn admitted(run:&mut Run,model:&Qwen35LoadedModel,cache:&HybridKvCache,slot:SlotId,cached:usize)->Result<()> {
    if run.localization.is_none(){return Ok(())}let index=owner(run,slot)?;let r=&run.requests[index];let who=owner_value(r);
    let session=run.localization.as_mut().unwrap();session.event(serde_json::json!({"event":"admitted","owner":who,"cached_tokens":cached}))?;
    if !session.extent_validated {
        let independently_planned=super::super::localization_extent::anchor_expected(&model.model.cfg,run.packet.keep)?;
        super::super::divergent::require_complete(session.expected.clone(),&independently_planned)?;
        super::super::divergent::require_complete(session.layer_expected.clone(),
            &layers::expected(&model.model.cfg,2,3,914,1923)?)?;
        session.extent_validated=true;
    }
    if [r.lane,r.turn]!=session.packet.selected_cell{return Ok(())}
    ensure!(r.prepared.as_ref().context("selected prepare missing")?.prompt.len()==1864&&r.canonical.as_ref().context("selector missing")?.decisions.len()==60,"C23 canonical extent changed");
    if run.packet.keep==4 {ensure!(cached==914,"K4 did not restore actual914");let restored=session.restored.as_mut().context("actual914 restore observation missing")?;
        join_restoration(restored,&r.ticket,slot.0)?;restored["owner"]=who;restored["pending_admission_join"]=Value::Bool(false);
    }else{ensure!(cached==0&&session.restored.is_none(),"K1 fabricated/unexpected restoration");}
    ensure!(!session.installed,"duplicate layer installation");let expected=layers::expected(&model.model.cfg,r.lane,r.turn,session.packet.first_row,session.packet.end_row_exclusive)?;
    super::super::divergent::require_complete(session.layer_expected.clone(),&expected)?;
    charge(planned_bytes(&expected)?.checked_add(crate::adr063_invariance::store::METADATA_BYTES as usize).context("layer disk grant overflow")?)?;
    let root=run.packet.output.join("localization-layers");let mut store=Store::new(&root,run.packet.identity.clone(),"C23-separate-instrumented-layer-localization-not-original-route".into(),expected.clone(),session.packet.payload_limit,expected.len()).map_err(anyhow::Error::msg)?;
    store.bind_acquisition(r.acquisition.as_ref().context("layer actual acquisition missing")?.clone().for_store_root(&root)?).map_err(anyhow::Error::msg)?;
    layers::install(layers::Owner{slot,generation:r.generation,lane:r.lane,turn:r.turn,ticket:r.ticket.clone()},store,session.packet.first_row,session.packet.end_row_exclusive)?;
    session.installed=true;let _=cache;Ok(())
}
pub(crate) fn localization_cohort(rows:&[(SlotId,usize)])->Result<Value> {
    let global=lock();let Some(run)=global.as_ref() else{return Ok(Value::Null)};if run.localization.is_none(){return Ok(Value::Null)}
    ensure!(rows.len()<=65536,"physical cohort metadata bound");let mut out=Vec::with_capacity(rows.len());
    for (physical,(slot,position)) in rows.iter().enumerate(){let r=&run.requests[owner(run,*slot)?];out.push(serde_json::json!({"physical_row":physical,"logical_position":position,"owner":owner_value(r)}));}
    Ok(Value::Array(out))
}
pub(super) fn finished(run:&mut Run,slot:SlotId)->Result<()> {
    let Some(session)=run.localization.as_mut() else{return Ok(())};let (ticket,generation)=run.owners.active.get(&slot.0).context("terminal missing active owner")?;
    let r=run.requests.iter().find(|r|&r.ticket==ticket&&r.generation==*generation).context("terminal lineage missing")?;
    session.event(serde_json::json!({"event":"finished-after-checked-output","owner":owner_value(r)}))?;
    if [r.lane,r.turn]==session.packet.selected_cell {
        ensure!(session.installed&&session.layer_result.is_none(),"layer terminal without unique installation");
        if let Some(error)=&run.error {layers::abort(error)?;anyhow::bail!("localization failed before terminal: {error}")}
        session.layer_result=Some(layers::finish()?);
    }Ok(())
}
pub(super) fn complete(run:&mut Run)->Result<()> {
    let Some(session)=run.localization.as_mut() else{return Ok(())};ensure!(run.error.is_none(),"localization previous observation failure");
    let creation=session.creation.as_ref().context("creation914 missing")?;let layer=session.layer_result.as_ref().context("checked layer finalization missing")?;
    ensure!(session.staged.is_some(),"creation914 was not actually staged");
    ensure!(run.packet.keep==1||session.restored.as_ref().is_some_and(|v|v["pending_admission_join"]==false),"restoration/admission join missing");
    let trace=session.store.take().context("anchor store missing")?.finish().map_err(anyhow::Error::msg)?;
    ensure!(trace.coverage.errors.is_empty(),"incomplete anchor raw extent: {:?}",trace.coverage.errors);
    save(&run.packet.output.join("localization.json"),&serde_json::json!({"schema":"hf2q.c23.localization-result.v1","localization_packet":run.packet.localization,
        "operator_authorization":session.packet.operator_authorization,"creation_owner":creation.owner,"staged_anchor":session.staged,"restore_ticket":session.restore_ticket,"restoration":session.restored,
        "restoration_status":if run.packet.keep==1{"not-performed-K1-cached0"}else{"actual914-observed"},"lifecycle":session.lifecycle,"layers":layer,
        "diagnostic_only":true,"original_route_success":false,"whole_c2_pass":false,"creation_live_parity":null,
        "interpretation_requirement":"exact all60 saved original sameK endpoint equality before original-route localization","compute_encoder_splitting_possible":true}))?;Ok(())
}

fn join_restoration(restored:&Value,ticket:&str,slot:u32)->Result<()> {
    ensure!(restored["ticket"]==ticket&&restored["slot"]==slot&&restored["pending_admission_join"]==true,
        "restore/admission owner mismatch or duplicate join");Ok(())
}
fn anchor_layout(cache:&HybridKvCache)->Result<Value> {
    let mut kv=Vec::new();let mut segment=2usize;
    for (layer_index,layer) in cache.full_attn.iter().chain(cache.mtp_slot.iter()).enumerate() {
        let names=if layer.tq.is_some(){vec!["tq-k-packed","tq-k-norms","tq-v-packed","tq-v-norms"]}else{Vec::new()};
        let mut names=names;if layer.k.is_some(){names.push("f32-k");}if layer.v.is_some(){names.push("f32-v");}
        for (buffer,name) in full_buffers(layer).into_iter().zip(names) {kv.push(serde_json::json!({"layer_index":layer_index,"segment":segment,"semantic":name,
            "source_shape":buffer.shape(),"valid_rows":if layer_index==cache.full_attn.len(){0}else{914},"source_slot_axis":0,"source_row_axis":2,"packed_order":"head,valid-row,last-axis","dtype":format!("{:?}",buffer.dtype())}));segment+=1;}
    }
    let mut linear=Vec::new();let mut offset=0usize;
    for (layer_index,l) in cache.linear_attn.iter().enumerate(){let recurrent=l.recurrent.element_count()/cache.n_seqs as usize;let s=l.conv_state.shape();
        linear.push(serde_json::json!({"linear_layer":layer_index,"image_recurrent_word_offset":offset,"recurrent_words":recurrent,
            "image_conv_word_offset":offset+recurrent,"conv_words":s[0]*s[1],"channels":s[0],"k_minus_one":s[1],
            "restored_conv_segment":segment,"restored_recurrent_segment":segment+1,
            "image_conv_order":"k,channel","live_conv_order":"channel,k","conv_descriptor_slot_axis":2,"recurrent_descriptor_slot_axis":3,
            "actual_linear_slot_addressing":"contiguous slot_index*(element_count/nseq), not descriptor-axis striding"}));
        offset+=recurrent+s[0]*s[1];segment+=2;
    }
    Ok(serde_json::json!({"schema":"hf2q.c23.anchor-layout.v1","full_kv":kv,"linear_image":linear,
        "image_segment":u32::MAX-1,"prefix_segment":u32::MAX,"creation_live_parity":null,"source":"kv_cache/la_restore.rs anchor_layout and apply_anchor_restore"}))
}
pub(super) fn failed(run:&mut Run,reason:&str)->Result<()> {
    let Some(session)=run.localization.as_mut() else{return Ok(())};
    if let Some(mut store)=session.store.take(){store.fail(reason);store.finish().map_err(anyhow::Error::msg)?;}
    save(&run.packet.output.join("localization-failure.json"),&serde_json::json!({"schema":"hf2q.c23.localization-failure.v1",
        "reason":reason,"lifecycle":session.lifecycle,"restoration":session.restored,"layer_result":session.layer_result,
        "complete":false,"original_route_success":false,"whole_c2_pass":false}))?;Ok(())
}

#[test]
fn c23_localization_join_rejects_reused_slot_ticket() {
    let restored=serde_json::json!({"ticket":"old-ticket","slot":2,"pending_admission_join":true});
    assert!(join_restoration(&restored,"old-ticket",2).is_ok());
    assert!(join_restoration(&restored,"new-ticket",2).is_err());
    assert!(join_restoration(&restored,"old-ticket",1).is_err());
    let mut joined=restored;joined["pending_admission_join"]=Value::Bool(false);
    assert!(join_restoration(&joined,"old-ticket",2).is_err());
}

#[test]
fn c23_localization_conv_mapping_keeps_every_raw_word() {
    assert_eq!(conv_to_live(&[1,2,3,4,5,6],3,2).unwrap(),vec![1,4,2,5,3,6]);
    assert!(conv_to_live(&[1,2,3],2,2).is_err());
}
#[test]
fn c23_localization_extent_fails_on_overflow() {
    let e=spec(0,0,Scalar::U32,usize::MAX);assert!(planned_bytes(&[e]).is_err());
    assert_ne!(at(0,2),at(1,2));assert_ne!(at(0,u32::MAX-1),at(0,u32::MAX));
}
