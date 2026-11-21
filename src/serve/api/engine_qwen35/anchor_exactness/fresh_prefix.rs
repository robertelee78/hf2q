//! Independent registered-reference extents and separately labelled prefix-All.
//! Neither operation can register an acquisition or replace real HTTP row proof.
use super::*;
use super::{fresh_reference,divergent};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtentPacket {
    schema:String,registration:acquisition::FileBinding,model:PathBuf,
    file_identity:acquisition::FileIdentity,output:PathBuf,chunk_rows:usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefixPacket {
    schema:String,execution_approved:bool,registration:acquisition::FileBinding,
    acquisition:acquisition::Binding,identity:Identity,model:PathBuf,
    file_identity:acquisition::FileIdentity,output:PathBuf,selected_cell:[usize;2],
    chunk_rows:usize,payload_limit:u64,expected:acquisition::FileBinding,
}
fn packet(name:&str)->Result<Vec<u8>> {
    let path=PathBuf::from(std::env::var_os(name).with_context(||format!("{name} missing"))?);
    let mut bytes=Vec::new();fs::File::open(path)?.take(4*1024*1024+1).read_to_end(&mut bytes)?;
    ensure!(bytes.len()<=4*1024*1024,"fresh prefix/extent packet exceeds4MiB");Ok(bytes)
}
pub(super) fn extents()->Result<()> {
    let packet:ExtentPacket=serde_json::from_slice(&packet("HF2Q_C2_FRESH_EXTENT_PACKET")?)?;
    ensure!(packet.schema=="hf2q.c2.fresh-fixed-input-extent-packet.v1"&&packet.chunk_rows>0&&packet.chunk_rows<=64,
        "fresh independent extent schema/chunk differs");
    let reference=fresh_reference::load_registration(&packet.registration)?;
    let (file,cfg)=divergent::header(&packet.model,&packet.file_identity)?;
    output_preflight(&packet.output)?;fs::create_dir(&packet.output)?;
    fs::set_permissions(&packet.output,fs::Permissions::from_mode(0o700))?;
    let mut bindings=std::collections::BTreeMap::new();
    for (lane,turns) in reference.observations.iter().enumerate() {for (turn,observed) in turns.iter().enumerate() {
        ensure!(observed.acquisition.opened_model==packet.file_identity,"extent header is not registered opened model");
        ensure!(observed.prepared.prompt.iter().chain(observed.terminal.decisions.iter()).all(|&token|token<cfg.vocab_size),
            "registered token exceeds actual header vocabulary");
        for (kind,items) in [
            ("decisions",divergent::decision_expected(lane,turn,observed.prepared.prompt.len(),observed.terminal.decisions.len(),cfg.vocab_size as usize,cfg.hidden_size as usize)),
            ("prefix",divergent::prefix_expected(lane,turn,observed.prepared.prompt.len(),cfg.vocab_size as usize,packet.chunk_rows)),
        ] {
            let name=format!("agent-{lane}-turn-{turn}-{kind}");let path=packet.output.join(format!("{name}.expected.json"));
            let bytes=serde_json::to_vec(&items)?;fs::write(&path,&bytes)?;
            bindings.insert(name,serde_json::json!({"path":path,"sha256":hash(&bytes)}));
        }
    }}
    ensure!(acquisition::FileIdentity::observed(&file)?==packet.file_identity,"extent header changed");
    fs::write(packet.output.join("expected-bindings.json"),serde_json::to_vec_pretty(&bindings)?)?;
    fs::write(packet.output.join("extent-authority.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "schema":"hf2q.c2.fresh-fixed-input-extents.v1","registration":packet.registration,"chunk_rows":packet.chunk_rows,
        "model_loaded":false,"metal_allocated":false,"inference_executed":false,"whole_c2_pass":false,
        "source":"independently registered complete per-lane native tape plus actual bound model header; never a compared store"}))?)?;
    Ok(())
}
pub(super) fn prefix(family:Family)->Result<()> {
    let raw=packet("HF2Q_C2_FRESH_PREFIX_PACKET")?;let packet:PrefixPacket=serde_json::from_slice(&raw)?;
    ensure!(packet.schema=="hf2q.c2.fresh-fixed-input-prefix-packet.v1"&&packet.execution_approved&&packet.identity.family==family,
        "fresh prefix schema/approval/family differs");
    ensure!(packet.chunk_rows>0&&packet.chunk_rows<=64&&packet.payload_limit>0&&packet.payload_limit<=8*1024*1024*1024*1024,
        "bounded independently admitted full-prefix diagnostic required");
    ensure!(std::env::var("HF2Q_DFLASH2")?=="0"&&std::env::var("HF2Q_QWEN_SPECULATION")?=="off",
        "prefix-All is a separate OFF diagnostic, not actual default/MTP serving");
    output_preflight(&packet.output)?;
    let (scope,reference)=acquisition::Scope::begin_c2_registered_replay(&packet.acquisition,&packet.identity,
        &packet.model,&packet.output,&raw,&packet.registration)?;
    let [lane,turn]=packet.selected_cell;
    ensure!(lane<4&&turn<reference.observations[lane].len(),"prefix cell outside registered per-lane extent");
    let observed=&reference.observations[lane][turn];let prompt=&observed.prepared.prompt;
    ensure!(observed.acquisition.opened_model==packet.file_identity,"prefix header is not registered opened model");
    let (file,cfg)=divergent::header(&packet.model,&packet.file_identity)?;
    let expected=divergent::prefix_expected(lane,turn,prompt.len(),cfg.vocab_size as usize,packet.chunk_rows);
    divergent::require_complete(serde_json::from_slice(&acquisition::checked_file(&packet.expected,64*1024*1024)?)?,&expected)?;
    let bytes=(prompt.len() as u64).checked_mul(cfg.vocab_size as u64).and_then(|n|n.checked_mul(4)).context("prefix extent overflow")?;
    ensure!(packet.payload_limit>=bytes,"complete prefix output space not independently admitted");
    let _gpu=crate::inference::hf2q_gpu_test_lock();
    let model=Qwen35LoadedModel::load_with_context(&load_options(&packet.model,None),Some(262144))?;
    ensure!(model.tq_kv_active&&model.vocab_size==cfg.vocab_size as usize,"actual model/header/TQ differs");
    let route="C2-fresh-fixed-independent-prefix-All-not-serving-prefill";
    scope.check_config(&packet.identity,&acquisition::ConfigRecipe{family,capacity:262144,slots:4,output_allowance:8192,
        cache_mode:"TQ",route,actual_model_config_sha256:hash(&serde_json::to_vec(&model.model.cfg)?)})?;
    let mut cache=model.model.with_gpu_cache_mut(|device,_|HybridKvCache::new_with_options(&model.model.cfg,device,262144,4,model.tq_kv_active))?;
    let case:serde_json::Value=serde_json::from_slice(&acquisition::checked_file(&reference.inputs.case,4*1024*1024)?)?;
    let budget=case["serve"]["env"]["KV_CACHE_BUDGET_BYTES"].as_str().context("declared KV budget missing")?.parse::<usize>()?;
    ensure!(cache.max_seq_len==262144&&cache.n_seqs==4&&cache.total_bytes()<=budget,"actual full-prefix cache allocation differs");
    fs::create_dir(&packet.output)?;fs::set_permissions(&packet.output,fs::Permissions::from_mode(0o700))?;
    let root=packet.output.join("prefix");
    let mut store=Store::new(&root,packet.identity.clone(),"C2-ClassB-fresh-registered-independent-prefix-All-not-default-serving".into(),
        expected.clone(),packet.payload_limit,expected.len()).map_err(anyhow::Error::msg)?;
    let actual=model.model.with_gpu_cache_mut(|device,_|scope.observed(device,&model.model as *const _ as usize as u64,1))?;
    store.bind_acquisition(actual.for_store_root(&root)?).map_err(anyhow::Error::msg)?;
    divergent::capture_prefix(&model,&mut cache,prompt,lane,turn,packet.chunk_rows,&mut store)?;
    divergent::complete(store)?;
    ensure!(acquisition::FileIdentity::observed(&file)?==packet.file_identity,"prefix header changed");
    fs::write(packet.output.join("extent-authority.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "schema":"hf2q.c2.fresh-fixed-input-prefix-acquisition.v1","registration":packet.registration,
        "selected_cell":packet.selected_cell,"expected_rows":prompt.len(),"independent_prefix_All_not_serving_prefill":true,
        "numerical_acceptance":"UNPROVED D<=0.25nat","whole_c2_pass":false}))?)?;
    scope.check_unchanged()
}
