//! Metadata-only preparation from retained ACTUAL Rust-tokenized Gemma inputs.
//! No model, context, tensor loader, device, native command or inference.
use super::{acquisition::{self,FileBinding,FileIdentity},protocol::Workload,receipt::{Family,Stage},store::hash};
use anyhow::{ensure,Context,Result};
use serde::Deserialize;
use serde_json::Value;
use std::{fs::{File,OpenOptions},io::{Read,Write},os::unix::fs::{OpenOptionsExt,PermissionsExt},path::{Path,PathBuf}};
const MODEL_SHA:&str="bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {path:PathBuf,retained_full_sha256:String,file_identity:FileIdentity}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default)] routing_recipes:bool,
    /// Optional stage-localizer packets; absent leaves every output unchanged.
    #[serde(default)] localizer:Option<Localizer>,
    /// Optional four-lane cohort stage-localizer packets (All128 + packed T2).
    #[serde(default)] cohort_localizer:Option<CohortLocalizer>,
    model:Model,full_content_receipt:FileBinding,tokenizer:FileBinding,template:FileBinding,
    producer_metadata:FileBinding,producer_build_receipt:FileBinding,
    producer_export_receipt:FileBinding,producer_config:FileBinding,
    producer_vectors:[FileBinding;4],expected_exporter_source_commit:String,
    expected_exporter_binary_sha256:String,output:PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalizerRoute {rerun_id:String,prefill_chunks:Vec<usize>}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Localizer {agent:usize,layers:Vec<u32>,stages:Vec<Stage>,normalize_rows:bool,routes:Vec<LocalizerRoute>}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CohortLocalizer {layers:Vec<u32>,prefill_stages:Vec<Stage>,decode_stages:Vec<Stage>,
    /// `Cohort` (All128 + packed, witness-ON) or `DriftAll128` (All128 only,
    /// witness-OFF, prefill-only); absent = `Cohort`.
    #[serde(default)] mode:super::gemma_cohort_localizer::Mode,
    /// Where each packet's store extent lives; absent = `Inline`.
    #[serde(default)] packet_form:PacketForm,
    /// Endpoint store the All128 route's endpoint store must be `ExactEqual`
    /// to (e.g. `gemma-t0-b5g`); required for `Cohort`, null for `DriftAll128`.
    observer_effect_reference:Option<String>,
    /// Endpoint store the packed route's endpoint store must be `ExactEqual`
    /// to; required for `Cohort`, null for `DriftAll128`.
    packed_observer_effect_reference:Option<String>}
/// `Inline`: the packet's `expected_stores` (#652's form, ≤16 MiB read).
/// `PlanOnly`: the packet binds a manifest file (`stage_manifest`) holding the
/// supervisor manifest-form extent, so the packet stays ≤1 MiB.
#[derive(Clone,Copy,Debug,Default,Deserialize,serde::Serialize,PartialEq,Eq)]
enum PacketForm {#[default] Inline,PlanOnly}
fn exact_hex(s:&str,n:usize)->bool {s.len()==n && s.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b))}
fn output_preflight(path:&Path)->Result<()> {
    ensure!(path.is_absolute() && !path.ancestors().any(|p|p.join(".git").exists()),"private output outsideGit");
    ensure!(path.parent().context("output parent")?.canonicalize()?==path.parent().unwrap(),"canonical outputparent");
    match std::fs::symlink_metadata(path) {Err(e) if e.kind()==std::io::ErrorKind::NotFound=>Ok(()),
        _=>anyhow::bail!("new output only")}
}
fn require_header(architecture:Option<&str>,embedded:Option<&str>,expected:&str)->Result<()> {
    ensure!(architecture==Some("gemma4") && embedded==Some(expected),
        "explicit Gemma architecture and exact embedded template required");Ok(())
}
fn derive(vectors:&[Vec<u32>;4],vocab:u32)->Result<Workload> {
    ensure!(vocab>0 && vectors.iter().all(|v|v.len()==253952 && v.iter().all(|&id|id<vocab)),
        "actual complete selected producer vectors/vocabulary required");
    let work=Workload {family:Family::Gemma4,vocab_size:vocab,
        lanes:vectors.iter().map(|v|v[..144].to_vec()).collect()};
    work.validate().map_err(anyhow::Error::msg)?;Ok(work)
}
fn le_hash(ids:&[u32])->String {use sha2::{Digest,Sha256};let mut h=Sha256::new();
    for id in ids {h.update(id.to_le_bytes());}hex::encode(h.finalize())}
fn planned(cfg:&crate::serve::config::Gemma4Config)->Result<Value> {
    let logit_bytes=4u64.checked_mul(144).and_then(|n|n.checked_mul(cfg.vocab_size as u64))
        .and_then(|n|n.checked_mul(4)).context("Gemma logits overflow")?;
    let expected=super::gemma_cache::expected(cfg,&[0,1,2,3],&[(64,false),(64,true),(128,false),(144,false)])
        .map_err(anyhow::Error::msg)?;
    let cache_bytes=expected.iter().try_fold(0u64,|sum,e|e.shape.iter()
        .try_fold(e.scalar.bytes() as u64,|n,&d|n.checked_mul(d as u64)).and_then(|n|sum.checked_add(n)))
        .context("Gemma cache capture overflow")?;
    let total=logit_bytes.checked_add(cache_bytes).context("Gemma total capture overflow")?;
    ensure!(total<=2*1024*1024*1024,"actual endpoint payload exceeds existing2GiB ceiling");
    let anchor= (0..cfg.num_hidden_layers).filter(|&l|!cfg.is_full_attention(l)).try_fold(0u64,|sum,l| {
        let hd=cfg.head_dim_for_layer(l) as u64;let heads=cfg.num_kv_heads_for_layer(l) as u64;
        heads.checked_mul(cfg.sliding_window as u64).and_then(|n|n.checked_mul(3*hd+4*(hd/256)))
            .and_then(|n|sum.checked_add(n))
    }).context("Gemma anchor overflow")?;
    Ok(serde_json::json!({"max_endpoint_payload":total,"full144_logits":logit_bytes,
        "active_cache_with_exact64_pair":cache_bytes,"one_live_serial_anchor_payload":anchor,
        "metadata_store_ceiling":134217728,"physical_admission":false,
        "operating_budget_disposition":"root must charge owned I/O/copies/model/fullC4scaffolds and positive margins separately"}))
}
fn routing_recipe_export(cfg:&crate::serve::config::Gemma4Config)->Result<Vec<Value>> {
    let mut rows=Vec::new();
    for routing in [false,true] {
        for (adapter,recipe) in [
            ("ordinary",super::gemma_actual::selected_recipe(cfg,routing)),
            ("T5",super::gemma_ngram::configuration_recipe(cfg,routing)),
        ] {
            let digest=acquisition::config_digest(&recipe)?;
            rows.push(serde_json::json!({"adapter":adapter,"routing_witness":routing,
                "config_sha256":digest,"recipe":recipe}));
        }
    }
    Ok(rows)
}
/// One manifest per route from the localizer's own builder. Every route must
/// freeze the identical `expected` list, or `compare` cannot localize them.
fn localizer_routes<'a>(cfg:&crate::serve::config::Gemma4Config,l:&'a Localizer)
    ->Result<Vec<(&'a LocalizerRoute,super::gemma_localizer::Manifest)>> {
    ensure!(!l.routes.is_empty() && l.routes.len()<=8,"one to eight localizer routes");
    let mut out:Vec<(&LocalizerRoute,super::gemma_localizer::Manifest)>=Vec::new();
    for r in &l.routes {
        ensure!(!r.rerun_id.is_empty() && r.rerun_id.len()<=64
            &&r.rerun_id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'||b==b'_')
            &&out.iter().all(|(o,_)|o.rerun_id!=r.rerun_id),"distinct file-safe localizer rerun ids");
        let m=super::gemma_localizer::localizer_manifest(cfg,l.agent,&r.prefill_chunks,&l.layers,&l.stages,l.normalize_rows)
            .map_err(anyhow::Error::msg)?;
        if let Some((_,first))=out.first() {ensure!(first.1==m.1,"localizer routes must freeze identical expected coordinates");}
        out.push((r,m));
    }
    Ok(out)
}
struct LocalizerExport {files:Vec<(String,Vec<u8>)>,routes:usize,agent:usize,records:usize}
/// `localizer-prepared.json` plus one DISABLED packet per route. The supervisor
/// fills `acquisition`, `output` and `identity.native_checksum` and flips
/// `execution_approved`; the identity is the key of a SEPARATE localizer
/// composition (its `config_sha256` is the stage-localizer recipe digest).
#[allow(clippy::too_many_arguments)]
fn localizer_export(c:&Config,l:&Localizer,cfg:&crate::serve::config::Gemma4Config,work:&Workload,source:&str,
    binary:&str,raw:&[u8],model_file:&FileIdentity,header_data_offset:impl serde::Serialize)->Result<LocalizerExport> {
    let routes=localizer_routes(cfg,l)?;
    let recipe=super::gemma_localizer::localizer_recipe(cfg,262144);
    let config_sha256=acquisition::config_digest(&recipe)?;
    let token_vectors_sha256=acquisition::tokens_digest(work)?;
    let identity=serde_json::json!({"family":Family::Gemma4,"source_commit":source,"binary_sha256":binary,
        "native_version":env!("HF2Q_MLX_NATIVE_VERSION"),"native_checksum":null,"model_sha256":MODEL_SHA,
        "tokenizer_sha256":c.tokenizer.sha256,"template_sha256":c.template.sha256,
        "config_sha256":config_sha256,"token_vectors_sha256":token_vectors_sha256});
    let mut files=Vec::new();let mut planned=serde_json::Map::new();
    for (route,(targets,expected,bytes)) in &routes {
        planned.insert(route.rerun_id.clone(),serde_json::json!(bytes));
        let packet=serde_json::json!({"execution_approved":false,"acquisition":null,"identity":identity,
            "model":c.model.path,"output":null,"tokens":work.lanes[l.agent],"workload":work,"agent":l.agent,
            "configured_context":262144,"configured_output_allowance":8192,"expected":expected,"targets":targets,
            "host_payload_limit":bytes,"rerun_id":route.rerun_id,"prefill_chunks":route.prefill_chunks});
        let encoded=serde_json::to_vec_pretty(&packet)?;
        ensure!(encoded.len()+65536<=8*1024*1024,"packet must fit the localizer 8 MiB read");
        files.push((format!("localizer-packet-{}.json",route.rerun_id),encoded));
    }
    let records=routes[0].1.1.len();
    let value=serde_json::json!({"schema":"hf2q.q1.gemma-localizer-preparation.v1","exporter_source_commit":source,
        "exporter_binary_sha256":binary,"native_version":env!("HF2Q_MLX_NATIVE_VERSION"),"config_input_sha256":hash(raw),
        "family":Family::Gemma4,"model_path":c.model.path,"model_full_sha256_retained":MODEL_SHA,
        "model_file_identity":model_file,"header_data_offset":header_data_offset,
        "actual_gemma_config_debug":format!("{cfg:?}"),"actual_gemma_config_debug_sha256":recipe.actual_model_config_sha256,
        "config_sha256":config_sha256,"recipe":recipe,"identity":identity,
        "comparison_key_scope":"SEPARATE stage-localizer composition/acquisition review/bindings; not the endpoint composition",
        "tokenizer_file_sha256":c.tokenizer.sha256,"template_file_sha256":c.template.sha256,
        "token_vectors_sha256":token_vectors_sha256,"agent":l.agent,"layers":l.layers,"stages":l.stages,
        "normalize_rows":l.normalize_rows,"configured_context":262144,"configured_output_allowance":8192,
        "records_per_route":records,"planned_payload":planned,
        "host_payload_cap":super::gemma_localizer::HOST_PAYLOAD_CAP,
        "packets":files.iter().map(|(name,bytes)|serde_json::json!({"file":name,"sha256":hash(bytes)})).collect::<Vec<_>>(),
        "packed_t2_route":"NotThisRoute: cohort_localizer block (gemma_cohort_localizer)",
        "model_loaded":false,"device_created":false,"inference_performed":false,"acquisition_attested":false,
        "execution_approved":false,"composition_reviewed":false,"physical_admission":false,"whole_q1_qualified":false});
    let encoded=serde_json::to_vec_pretty(&value)?;ensure!(encoded.len()<=2*1024*1024,"bounded localizer metadata");
    files.insert(0,("localizer-prepared.json".into(),encoded));
    Ok(LocalizerExport {files,routes:routes.len(),agent:l.agent,records})
}
/// `cohort-localizer-prepared.json` plus DISABLED multistore packets. `Cohort`:
/// the All128 reference (`T0ColdSerial`) and the packed route
/// (`T2ActualCohort`), witness-ON; `DriftAll128`: the All128 route only,
/// witness-OFF and prefill-only. Every packet freezes the same endpoint extent
/// as the b3 T0/T2 stores and one route-agnostic stage manifest, inline or (in
/// the `PlanOnly` form) in a bound manifest file; their identity is a SEPARATE
/// composition key per mode (`gemma_cohort_localizer::recipe`).
fn cohort_localizer_export(c:&Config,l:&CohortLocalizer,cfg:&crate::serve::config::Gemma4Config,work:&Workload,source:&str,
    binary:&str,raw:&[u8])->Result<Vec<(String,Vec<u8>)>> {
    use super::{gemma_actual::routing,gemma_cohort_localizer as cohort};
    let drift=l.mode==cohort::Mode::DriftAll128;
    // Identity native is this build's own pin in both modes; only Cohort (via
    // the witness-ON adapter) also requires the published witness native.
    if drift {
        ensure!(l.observer_effect_reference.is_none() && l.packed_observer_effect_reference.is_none(),
            "DriftAll128 has no observer-effect reference stores");
        ensure!(l.decode_stages.is_empty(),"DriftAll128 plan is prefill-only (decode_stages: [])");
        ensure!(l.packet_form==PacketForm::PlanOnly,"DriftAll128 packets use the plan-only form (bound stage manifest)");
    } else {
        routing::require_published(env!("HF2Q_MLX_NATIVE_VERSION"))?;
        ensure!([&l.observer_effect_reference,&l.packed_observer_effect_reference].iter()
            .all(|r|r.as_deref().is_some_and(super::gemma_actual::file_safe)),"explicit file-safe observer-effect reference stores");
    }
    let (_,stages,stage_bytes)=cohort::manifest(cfg,&l.layers,&l.prefill_stages,&l.decode_stages).map_err(anyhow::Error::msg)?;
    let recipe=cohort::recipe(cfg,l.mode);let config_sha256=acquisition::config_digest(&recipe)?;
    let token_vectors_sha256=acquisition::tokens_digest(work)?;
    let identity=serde_json::json!({"family":Family::Gemma4,"source_commit":source,"binary_sha256":binary,
        "native_version":env!("HF2Q_MLX_NATIVE_VERSION"),"native_checksum":null,"model_sha256":MODEL_SHA,
        "tokenizer_sha256":c.tokenizer.sha256,"template_sha256":c.template.sha256,
        "config_sha256":config_sha256,"token_vectors_sha256":token_vectors_sha256});
    // DriftAll128 is witness-OFF: null budget, no charge.
    let (budget,charge)=if drift {(Value::Null,0)} else {(routing::budget_json(),routing::CHARGE)};
    let all128=("all128",super::protocol::Trajectory::T0ColdSerial,"adr063_invariance::gemma_actual::q1_gemma_actual_all128_stage_localizer",&l.observer_effect_reference);
    let plan=if drift {vec![all128]} else {vec![all128,
        ("packed",super::protocol::Trajectory::T2ActualCohort,"adr063_invariance::gemma_actual::q1_gemma_actual_cohort_stage_localizer",&l.packed_observer_effect_reference)]};
    let mut files=Vec::new();let mut routes=Vec::new();
    for (name,trajectory,filter,reference) in plan {
        let endpoint=super::gemma_actual::endpoint_expected(cfg,trajectory)?;
        let endpoint_bytes=endpoint.iter().try_fold(0u64,|sum,e|e.shape.iter()
            .try_fold(e.scalar.bytes() as u64,|n,&d|n.checked_mul(d as u64)).and_then(|n|sum.checked_add(n)))
            .context("endpoint payload overflow")?;
        let host=endpoint_bytes.checked_add(charge).and_then(|n|n.checked_add(stage_bytes)).context("host payload overflow")?;
        ensure!(host<=2*1024*1024*1024,"endpoint+routing+stage payload exceeds the 2 GiB endpoint ceiling");
        let rerun_id=if drift {format!("drift-{name}")} else {format!("cohort-{name}")};
        let mut packet=serde_json::json!({"execution_approved":false,"acquisition":null,"identity":identity,
            "workload":work,"trajectory":trajectory,"model":c.model.path,"output":null,
            "configured_context":262144,"configured_output_allowance":8192,"host_payload_limit":host,
            "routing_witness":budget,"stage_localizer":{"rerun_id":rerun_id,"layers":l.layers,
                "prefill_stages":l.prefill_stages,"decode_stages":l.decode_stages,"host_payload_limit":stage_bytes},
            "lanes":[0,1,2,3],"localizer_mode":l.mode,"observer_effect_reference":reference});
        let stores=serde_json::json!({cohort::ENDPOINT_STORE:endpoint,cohort::STAGE_STORE:stages});
        let manifest=match l.packet_form {
            PacketForm::Inline=>{packet["expected_stores"]=stores;None}
            PacketForm::PlanOnly=>{
                let bytes=serde_json::to_vec_pretty(&serde_json::json!({"lanes":[0,1,2,3],"host_payload_limit":host,"expected_stores":stores}))?;
                ensure!(bytes.len() as u64<=cohort::MANIFEST_FILE_BYTES,"stage manifest file exceeds 64 MiB");
                let file=format!("cohort-localizer-manifest-{name}.json");
                packet["stage_manifest"]=serde_json::json!({"path":c.output.join(&file),"sha256":hash(&bytes)});
                Some((file,bytes))
            }
        };
        let encoded=serde_json::to_vec_pretty(&packet)?;
        // The supervisor rewrites the packet indent-2 (the same layout) and fills
        // its bindings; keep room below the localizer read (16 MiB inline, 1 MiB plan-only).
        ensure!(match l.packet_form {PacketForm::Inline=>encoded.len()+1024*1024<=16*1024*1024,
            PacketForm::PlanOnly=>encoded.len()+65536<=1024*1024},"cohort packet must fit the localizer read after the supervisor fill");
        let file=format!("cohort-localizer-packet-{name}.json");
        routes.push(serde_json::json!({"rerun_id":rerun_id,"trajectory":trajectory,"filter":filter,
            "stage_route":cohort::route(trajectory).map_err(anyhow::Error::msg)?,"packet":{"file":file,"sha256":hash(&encoded)},
            "manifest":manifest.as_ref().map(|(f,b)|serde_json::json!({"file":f,"sha256":hash(b)})),
            "endpoint_records":endpoint.len(),"endpoint_planned_payload":endpoint_bytes,"routing_witness_charge":charge,
            "stage_planned_payload":stage_bytes,"host_payload_limit":host,"observer_effect_reference":reference}));
        files.push((file,encoded));
        files.extend(manifest);
    }
    let value=serde_json::json!({"schema":"hf2q.q1.gemma-cohort-localizer-preparation.v1","exporter_source_commit":source,
        "exporter_binary_sha256":binary,"native_version":env!("HF2Q_MLX_NATIVE_VERSION"),"config_input_sha256":hash(raw),
        "family":Family::Gemma4,"model_path":c.model.path,"model_full_sha256_retained":MODEL_SHA,
        "actual_gemma_config_debug_sha256":recipe.actual_model_config_sha256,"config_sha256":config_sha256,"recipe":recipe,
        "identity":identity,"localizer_mode":l.mode,"route":l.mode.route_label(),"packet_form":l.packet_form,
        "comparison_key_scope":if drift {"SEPARATE DriftAll128 composition/acquisition review/bindings; witness-OFF, no routing_witness binding"}
            else {"SEPARATE cohort-localizer composition/acquisition review/bindings; both routes share it"},
        "layers":l.layers,"prefill_stages":l.prefill_stages,"decode_stages":l.decode_stages,"lanes":[0,1,2,3],
        "stage_records":stages.len(),"stage_planned_payload":stage_bytes,"routes":routes,
        "supervisor_case":{"kind":"multistore","stores":{cohort::ENDPOINT_STORE:".",cohort::STAGE_STORE:cohort::STAGE_STORE},
            "checks":["gemma_loc4"],"packet_env":"HF2Q_Q1_GEMMA_ENDPOINT_PACKET","completion_marker":cohort::MARKER,"marker_stream":"stdout",
            "manifest":if l.packet_form==PacketForm::PlanOnly {"each route's manifest file (bound as the case `manifest`)"} else {"none: extent in the packet"}},
        "workload_scope":"the prep's derived first-144 vectors; a runtime dataset swap must replace workload and identity.token_vectors_sha256 only (store manifests are token-independent)",
        "route_class":if drift {"build-drift localization: witness-OFF production routing; no fix implied"} else {"ClassB packed-versus-All128; no fix implied"},
        "model_loaded":false,"device_created":false,"inference_performed":false,"acquisition_attested":false,
        "execution_approved":false,"composition_reviewed":false,"physical_admission":false,"whole_q1_qualified":false});
    let encoded=serde_json::to_vec_pretty(&value)?;ensure!(encoded.len()<=2*1024*1024,"bounded cohort localizer metadata");
    files.insert(0,("cohort-localizer-prepared.json".into(),encoded));
    Ok(files)
}
#[test]
#[ignore="root-supervised header/retainedJSON only; no model/device"]
fn q1_gemma_endpoint_metadata_prepare(){run().unwrap();}
fn run()->Result<()> {
    let path=std::env::var_os("HF2Q_Q1_GEMMA_PREP_CONFIG").context("explicit Gemma prep config")?;
    let mut raw=Vec::new();File::open(path)?.take(1024*1024+1).read_to_end(&mut raw)?;
    ensure!(raw.len()<=1024*1024,"bounded metadata config");let c:Config=serde_json::from_slice(&raw)?;
    output_preflight(&c.output)?;
    if c.routing_recipes {super::gemma_actual::routing::require_published(env!("HF2Q_MLX_NATIVE_VERSION"))
        .context("routing recipe metadata requires the exact published routing-witness native build")?;}
    ensure!(exact_hex(&c.expected_exporter_source_commit,40)&&exact_hex(&c.expected_exporter_binary_sha256,64)
        &&c.model.retained_full_sha256==MODEL_SHA,"exact selected source/binary/model bindings");
    let source=option_env!("HF2Q_BUILD_GIT_SHA").context("compiled source stamp")?;
    ensure!(source==c.expected_exporter_source_commit,"actual exporter source mismatch");
    let binary=acquisition::executable_digest()?;ensure!(binary==c.expected_exporter_binary_sha256,"actual executable mismatch");
    let bindings=[(&c.full_content_receipt,16*1024*1024),(&c.tokenizer,64*1024*1024),
        (&c.template,1024*1024),(&c.producer_metadata,2*1024*1024),(&c.producer_build_receipt,2*1024*1024),
        (&c.producer_export_receipt,2*1024*1024),(&c.producer_config,1024*1024)];
    let bound=bindings.iter().map(|(b,n)|acquisition::checked_file(b,*n)).collect::<Result<Vec<_>>>()?;
    acquisition::full_content_receipt_has_model(&bound[0],MODEL_SHA)?;
    std::str::from_utf8(&bound[1])?;let template=std::str::from_utf8(&bound[2])?;
    let metadata:Value=serde_json::from_slice(&bound[3])?;let build:Value=serde_json::from_slice(&bound[4])?;
    let export:Value=serde_json::from_slice(&bound[5])?;let producer_config:Value=serde_json::from_slice(&bound[6])?;
    ensure!(metadata["schema"]=="hf2q.gemma-largest-prompt-metadata.v1" && metadata["family"]=="gemma4"
        &&metadata["prompt_tokens"]==253952 &&metadata["context"]==262144
        &&metadata["model_inference_performed"]==false &&metadata["config_sha256"]==c.producer_config.sha256
        &&metadata["tokenizer_raw_sha256"]==c.tokenizer.sha256
        &&metadata["template_raw_sha256"]==c.template.sha256,"retained actual Gemma metadata bindings");
    ensure!(build["source_clean"]==true &&build["model_loaded"]==false
        &&build["source_commit"]==build["explicit_git_commit_sha"] &&build["source_commit"]==metadata["compiled_build_git_sha"]
        &&export["helper_source"]==build["source_commit"] &&export["actual_metadata_export_passed"]==true
        &&export["model_loaded"]==false &&export["metadata_sha256"]==c.producer_metadata.sha256,
        "retained metadata build/export source join; not new execution approval");
    ensure!(producer_config["family"]=="gemma4" &&producer_config["tokenizer"]["sha256"]==c.tokenizer.sha256
        &&producer_config["template"]["sha256"]==c.template.sha256,"producer tokenizer/template config join");
    let mut vectors:[Vec<u32>;4]=std::array::from_fn(|_|Vec::new());let mut vector_raw=Vec::new();
    let requests=metadata["requests"].as_array().context("producer four requests")?;ensure!(requests.len()==4,"four producer vectors");
    for lane in 0..4 {
        let b=&c.producer_vectors[lane];let name=format!("agent-{lane}-token-ids.json");
        ensure!(b.path.parent()==c.producer_metadata.path.parent() &&b.path.file_name().and_then(|s|s.to_str())==Some(name.as_str()),"actual ordered producer filenames");
        let bytes=acquisition::checked_file(b,4*1024*1024)?;vectors[lane]=serde_json::from_slice(&bytes)?;
        ensure!(requests[lane]["agent"]==lane &&requests[lane]["token_file_sha256"]==b.sha256
            &&requests[lane]["token_ids_sha256_le_u32"]==le_hash(&vectors[lane])
            &&export["artifacts"][name.as_str()]==b.sha256,"actual raw/LE producer vector identity");
        vector_raw.push(bytes);
    }
    ensure!(c.model.path.is_absolute(),"absolute model path");let fd=File::open(&c.model.path)?;
    let before=FileIdentity::observed(&fd)?;ensure!(before==c.model.file_identity,"actual model descriptor/stat mismatch");
    let gguf=mlx_native::gguf::GgufFile::from_file(fd.try_clone()?)?;
    ensure!(gguf.tensor_data_offset()<=32*1024*1024,"bounded header only");
    require_header(gguf.metadata_string("general.architecture"),gguf.metadata_string("tokenizer.chat_template"),template)?;
    let mut cfg=crate::serve::config::Gemma4Config::from_gguf(&gguf)?;cfg.max_position_embeddings=262144;
    let work=derive(&vectors,u32::try_from(cfg.vocab_size)?)?;
    let recipe=super::gemma_actual::recipe(&cfg);let plan=planned(&cfg)?;
    let workload_bytes=serde_json::to_vec_pretty(&work)?;
    let value=serde_json::json!({"schema":"hf2q.q1.gemma-endpoint-preparation.v1","exporter_source_commit":source,
        "exporter_binary_sha256":binary,"native_version":env!("HF2Q_MLX_NATIVE_VERSION"),
        "family":Family::Gemma4,"configured_context":262144,"output_allowance":8192,"logical_lanes":4,"tokens_per_lane":144,
        "config_input_sha256":hash(&raw),"model_path":c.model.path,"workload_file_sha256":hash(&workload_bytes),
        "full_content_receipt_path":c.full_content_receipt.path,"full_content_receipt_sha256":c.full_content_receipt.sha256,
        "model_full_sha256_retained":MODEL_SHA,"model_file_identity":before,
        "model_payload_rehashed":false,"header_data_offset":gguf.tensor_data_offset(),
        "tokenizer_file_sha256":c.tokenizer.sha256,"template_file_sha256":c.template.sha256,
        "producer_metadata_sha256":c.producer_metadata.sha256,"producer_vector_sha256":c.producer_vectors.iter().map(|b|&b.sha256).collect::<Vec<_>>(),
        "token_vectors_sha256":acquisition::tokens_digest(&work)?,"actual_gemma_config_debug":format!("{cfg:?}"),
        "actual_gemma_config_debug_sha256":recipe.actual_model_config_sha256,"config_sha256":acquisition::config_digest(&recipe)?,
        "recipe":recipe,"payload_plan":plan,"derivation":"first144 retained actual Rust input IDs:128prompt+16forced; not generated canonical continuation",
        "native_checksum_verification":"independent published build composition required","model_loaded":false,
        "device_created":false,"inference_performed":false,"tokenizer_semantic_attestation":false,
        "execution_approved":false,"composition_reviewed":false,"acquisition_attested":false,"whole_q1_qualified":false});
    let encoded=serde_json::to_vec_pretty(&value)?;ensure!(encoded.len()<=2*1024*1024,"metadata output bound");
    // Additive file only; omitted/false flag preserves both old output bytes.
    let routing_encoded=if c.routing_recipes {
        let recipes=routing_recipe_export(&cfg)?;
        let v=serde_json::json!({"schema":"hf2q.q1.gemma-routing-recipes-metadata.v1",
            "exporter_source_commit":source,"exporter_binary_sha256":binary,
            "native_version":env!("HF2Q_MLX_NATIVE_VERSION"),"config_input_sha256":hash(&raw),
            "model_full_sha256_retained":MODEL_SHA,"model_file_identity":before,
            "header_data_offset":gguf.tensor_data_offset(),"actual_gemma_config_debug":format!("{cfg:?}"),
            "actual_gemma_config_debug_sha256":recipe.actual_model_config_sha256,
            "tokenizer_file_sha256":c.tokenizer.sha256,"template_file_sha256":c.template.sha256,
            "recipes":recipes,"runtime_dataset_binding":"SEPARATE independently reviewed842 greedy144 dataset; no dataset identity attested by recipe export",
            "model_loaded":false,"device_created":false,"inference_performed":false,
            "acquisition_attested":false,"execution_approved":false,"physical_admission":false,"whole_q1_qualified":false});
        let encoded=serde_json::to_vec_pretty(&v)?;ensure!(encoded.len()<=128*1024,"bounded recipe metadata");Some(encoded)
    }else{None};
    let localizer=match &c.localizer {
        Some(l)=>Some(localizer_export(&c,l,&cfg,&work,source,&binary,&raw,&before,gguf.tensor_data_offset())?),
        None=>None,
    };
    let cohort=match &c.cohort_localizer {
        Some(l)=>Some(cohort_localizer_export(&c,l,&cfg,&work,source,&binary,&raw)?),
        None=>None,
    };
    for ((b,n),old) in bindings.iter().zip(&bound) {ensure!(acquisition::checked_file(b,*n)?==*old,"bound input changed");}
    for (b,old) in c.producer_vectors.iter().zip(&vector_raw) {ensure!(acquisition::checked_file(b,4*1024*1024)?==*old,"producer vector changed");}
    ensure!(FileIdentity::observed(&fd)?==before,"model descriptor changed during metadata read");
    output_preflight(&c.output)?;std::fs::create_dir(&c.output)?;std::fs::set_permissions(&c.output,std::fs::Permissions::from_mode(0o700))?;
    for (name,bytes) in [("prepared.json",encoded),("workload.json",workload_bytes)] {
        OpenOptions::new().create_new(true).write(true).mode(0o600).open(c.output.join(name))?.write_all(&bytes)?;
    }
    if let Some(bytes)=routing_encoded {
        OpenOptions::new().create_new(true).write(true).mode(0o600).open(c.output.join("routing-recipes.json"))?.write_all(&bytes)?;
        println!("Q1_GEMMA_ROUTING_RECIPES_METADATA_PREPARED recipes=4 dataset_attested=false model=false device=false execution=false");
    }
    if let Some(export)=localizer {
        for (name,bytes) in &export.files {
            OpenOptions::new().create_new(true).write(true).mode(0o600).open(c.output.join(name))?.write_all(bytes)?;
        }
        println!("Q1_GEMMA_LOCALIZER_METADATA_PREPARED routes={} agent={} records_per_route={} device=false execution_approved=false",
            export.routes,export.agent,export.records);
    }
    if let Some(files)=cohort {
        for (name,bytes) in &files {
            OpenOptions::new().create_new(true).write(true).mode(0o600).open(c.output.join(name))?.write_all(bytes)?;
        }
        let routes=files.iter().filter(|(name,_)|name.starts_with("cohort-localizer-packet-")).count();
        println!("Q1_GEMMA_COHORT_LOCALIZER_METADATA_PREPARED routes={routes} lanes=4 device=false execution_approved=false");
    }
    println!("Q1_GEMMA_ENDPOINT_METADATA_PREPARED lanes=4 tokens=144 device=false execution=false");Ok(())
}
#[cfg(test)]
#[path="gemma_prepare_controls.rs"]
mod controls;
