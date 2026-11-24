//! Full original W: all four lanes, all original7-or9 requests, true turns3/5.
//! Prefix-All is a separately labelled diagnostic, never substituted for actual
//! HTTP default/K3 target rows. Acceptance is D<=0.25nat, not ADR067D2.
use super::*;
use super::trace::coordinate;
pub(super) fn run(family:Family)->Result<()> {super::fullw_http::run(family)}
pub(super) fn decision_expected(lane:usize,turn:usize,prompt:usize,count:usize,
    vocab:usize,hidden:usize)->Vec<ExpectedTensor> {
    let mut expected=Vec::with_capacity(count*3);
    for index in 0..count {
        for (stage,scalar,shape) in [
            (Stage::Logits,Scalar::F32,vec![1,vocab]),
            (Stage::SpecVerifierHidden,Scalar::F32,vec![1,hidden]),
            (Stage::SpecCanonicalIds,Scalar::U32,vec![1]),
        ] {
            expected.push(ExpectedTensor {at:coordinate(&format!("agent-{lane}"),turn,
                prompt-1+index,stage),scalar,shape});
        }
    }
    expected
}
pub(super) fn prefix_expected(lane:usize,turn:usize,prompt:usize,vocab:usize,chunk:usize)->Vec<ExpectedTensor> {
    (0..prompt).step_by(chunk).map(|begin|ExpectedTensor {
        at:coordinate(&format!("agent-{lane}"),turn,begin,Stage::Logits),scalar:Scalar::F32,
        shape:vec![(prompt-begin).min(chunk),vocab],
    }).collect()
}
pub(super) fn require_complete(mut supplied:Vec<ExpectedTensor>,expected:&[ExpectedTensor])->Result<()> {
    let mut independent=expected.to_vec();
    supplied.sort_by(|a,b|a.at.cmp(&b.at));independent.sort_by(|a,b|a.at.cmp(&b.at));
    ensure!(supplied==independent,"Incomplete: registered extents differ from independently complete original W");
    Ok(())
}
pub(super) fn complete(store:Store)->Result<()> {
    let trace=store.finish().map_err(anyhow::Error::msg)?;
    ensure!(trace.coverage.errors.is_empty()&&trace.coverage.expected.len()==trace.coverage.captured.len(),
        "Incomplete: actual all-row acquisition omitted independent original-W extent");
    Ok(())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PrefixPacket {
    execution_approved:bool,acquisition:acquisition::Binding,identity:Identity,
    model:PathBuf,file_identity:acquisition::FileIdentity,frozen_script:acquisition::FileBinding,
    output:PathBuf,selected_cell:[usize;2],chunk_rows:usize,payload_limit:u64,
    expected:acquisition::FileBinding,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtentPacket {
    model:PathBuf,file_identity:acquisition::FileIdentity,frozen_script:acquisition::FileBinding,
    family:Family,output:PathBuf,chunk_rows:usize,
}
pub(super) fn header(model:&Path,expected:&acquisition::FileIdentity)
    ->Result<(fs::File,crate::inference::models::qwen35::Qwen35Config)> {
    let file=fs::File::open(model)?;
    ensure!(&acquisition::FileIdentity::observed(&file)?==expected,"independent header identity differs");
    let gguf=mlx_native::gguf::GgufFile::from_file(file.try_clone()?)?;
    ensure!(gguf.tensor_data_offset()<=32*1024*1024,"bounded actual GGUF header required");
    Ok((file,crate::inference::models::qwen35::Qwen35Config::from_gguf(&gguf)?))
}
pub(super) fn prefix(family:Family)->Result<()> {
    let raw=fs::read(PathBuf::from(std::env::var_os("HF2Q_C2_FULLW_PREFIX_PACKET")
        .context("full original prefix packet missing")?))?;
    ensure!(raw.len()<=4*1024*1024,"prefix control packet exceeds4MiB");
    let packet:PrefixPacket=serde_json::from_slice(&raw)?;
    ensure!(packet.execution_approved&&packet.identity.family==family,"root full-prefix grant missing");
    ensure!(packet.chunk_rows>0&&packet.chunk_rows<=64&&packet.payload_limit>0
        &&packet.payload_limit<=8*1024*1024*1024*1024,"bounded admitted prefix diagnostic required");
    ensure!(packet.identity.model_sha256==if family==Family::Qwen38{Q38}else{Q36},"not original artifact");
    ensure!(std::env::var("HF2Q_DFLASH2")?=="0"&&std::env::var("HF2Q_QWEN_SPECULATION")?=="off",
        "independent prefix-All must be explicitly separate from actual default/K3 route");
    output_preflight(&packet.output)?;
    let script=acquisition::checked_file(&packet.frozen_script,32*1024*1024)?;
    let (scope,validated)=acquisition::Scope::begin_c2(&packet.acquisition,&packet.identity,
        &packet.model,&packet.output,&raw,&script)?;
    let [lane,turn]=packet.selected_cell;
    ensure!(lane<4&&turn<validated.turns[lane].len(),"cell outside complete original W");
    let prompt=&validated.turns[lane][turn].prepared.prompt;
    let (file,cfg)=header(&packet.model,&packet.file_identity)?;
    let expected=prefix_expected(lane,turn,prompt.len(),cfg.vocab_size as usize,packet.chunk_rows);
    let supplied=serde_json::from_slice(&acquisition::checked_file(&packet.expected,64*1024*1024)?)?;
    require_complete(supplied,&expected)?;
    let bytes=(prompt.len() as u64).checked_mul(cfg.vocab_size as u64).and_then(|n|n.checked_mul(4))
        .context("full original prefix extent overflow")?;
    ensure!(packet.payload_limit>=bytes,"selected full prefix lacks independently admitted output space");
    let _gpu=crate::inference::hf2q_gpu_test_lock();
    let model=Qwen35LoadedModel::load_with_context(&load_options(&packet.model,None),Some(262144))?;
    ensure!(model.tq_kv_active&&model.vocab_size==cfg.vocab_size as usize,"actual model/header/substrate differs");
    let route="C2-full-original-independent-prefix-All-not-serving-prefill";
    scope.check_config(&packet.identity,&acquisition::ConfigRecipe {family,capacity:262144,slots:4,
        output_allowance:8192,cache_mode:"TQ",route,
        actual_model_config_sha256:hash(&serde_json::to_vec(&model.model.cfg)?)})?;
    let mut cache=model.model.with_gpu_cache_mut(|device,_|HybridKvCache::new_with_options(
        &model.model.cfg,device,262144,4,model.tq_kv_active))?;
    let case:serde_json::Value=serde_json::from_slice(&acquisition::checked_file(&validated.script.corpus.case,4*1024*1024)?)?;
    let budget=case["serve"]["env"]["KV_CACHE_BUDGET_BYTES"].as_str()
        .context("source-bound original KV budget missing")?.parse::<usize>()?;
    ensure!(cache.max_seq_len==262144&&cache.n_seqs==4&&cache.total_bytes()<=budget,
        "actual full-prefix cache differs from original262144/four-slot/source-bound budget");
    fs::create_dir(&packet.output)?;fs::set_permissions(&packet.output,fs::Permissions::from_mode(0o700))?;
    let root=packet.output.join("prefix");
    let mut store=Store::new(&root,packet.identity.clone(),
        "C2-ClassB-independent-complete-original-frozen-prefix-All-diagnostic-not-default-route".into(),
        expected.clone(),packet.payload_limit,expected.len()).map_err(anyhow::Error::msg)?;
    let observed=model.model.with_gpu_cache_mut(|device,_|scope.observed(device,
        &model.model as *const _ as usize as u64,1))?;
    store.bind_acquisition(observed.for_store_root(&root)?).map_err(anyhow::Error::msg)?;
    capture_prefix(&model,&mut cache,prompt,lane,turn,packet.chunk_rows,&mut store)?;
    complete(store)?;
    ensure!(acquisition::FileIdentity::observed(&file)?==packet.file_identity,"source header changed");
    fs::write(packet.output.join("extent-authority.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "schema":"hf2q.c2.full-original-prefix-acquisition.v1","selected_cell":packet.selected_cell,
        "expected_rows":prompt.len(),"source":"complete independently captured original frozen-script prompt vectors",
        "required_cells":validated.turns.iter().enumerate().flat_map(|(lane,turns)|turns.iter().enumerate()
            .map(move|(turn,t)|serde_json::json!({"lane":lane,"turn":turn,"prefix_rows":t.prepared.prompt.len(),"decision_rows":t.terminal.decisions.len()}))).collect::<Vec<_>>(),
        "independent_prefix_All_not_serving_prefill":true,"numerical_acceptance":"UNPROVED D<=0.25nat",
        "not_whole_C2_pass":true
    }))?)?;
    scope.check_unchanged()
}
pub(super) fn capture_prefix(model:&Qwen35LoadedModel,cache:&mut HybridKvCache,prompt:&[u32],
    lane:usize,turn:usize,chunk:usize,store:&mut Store)->Result<()> {
    for begin in (0..prompt.len()).step_by(chunk) {
        let end=(begin+chunk).min(prompt.len());
        let positions=positions_for_range(i32::try_from(begin)?,end-begin);
        let logits=model.model.forward_gpu(&prompt[begin..end],&positions,cache,SlotId(lane as u32))?;
        ensure!(logits.len()==(end-begin)*model.vocab_size
            &&cache.sequence_len_for_slot(SlotId(lane as u32))? as usize==end,
            "actual prefix-All cursor/complete vocabulary extent mismatch");
        trace::capture_f32(store,coordinate(&format!("agent-{lane}"),turn,begin,Stage::Logits),
            &logits,vec![end-begin,model.vocab_size])?;
    }
    Ok(())
}
#[test]
#[ignore="independent original-W header/script extents only; no model/device/inference"]
fn qwen_c2_full_original_extent_export() {
    let run=||->Result<()> {
        let raw=fs::read(PathBuf::from(std::env::var_os("HF2Q_C2_FULLW_EXTENT_PACKET")
            .context("full-W independent extent packet missing")?))?;
        ensure!(raw.len()<=4*1024*1024,"extent control packet exceeds4MiB");
        let packet:ExtentPacket=serde_json::from_slice(&raw)?;
        ensure!(packet.chunk_rows>0&&packet.chunk_rows<=64,"bounded full-prefix tile required");
        let script=acquisition::checked_file(&packet.frozen_script,32*1024*1024)?;
        let validated=super::frozen_script::Script::validate_bytes(&script,packet.family)?;
        let (file,cfg)=header(&packet.model,&packet.file_identity)?;
        output_preflight(&packet.output)?;fs::create_dir(&packet.output)?;
        fs::set_permissions(&packet.output,fs::Permissions::from_mode(0o700))?;
        let mut bindings=std::collections::BTreeMap::new();
        for (lane,turns) in validated.turns.iter().enumerate() {for (turn,t) in turns.iter().enumerate() {
            for (kind,items) in [
                ("decisions",decision_expected(lane,turn,t.prepared.prompt.len(),t.terminal.decisions.len(),cfg.vocab_size as usize,cfg.hidden_size as usize)),
                ("prefix",prefix_expected(lane,turn,t.prepared.prompt.len(),cfg.vocab_size as usize,packet.chunk_rows)),
            ] {
                let name=format!("agent-{lane}-turn-{turn}-{kind}");let path=packet.output.join(format!("{name}.expected.json"));
                let bytes=serde_json::to_vec(&items)?;fs::write(&path,&bytes)?;
                bindings.insert(name,serde_json::json!({"path":path,"sha256":hash(&bytes)}));
            }
        }}
        fs::write(packet.output.join("expected-bindings.json"),serde_json::to_vec_pretty(&bindings)?)?;
        ensure!(acquisition::FileIdentity::observed(&file)?==packet.file_identity,"header changed");
        Ok(())
    };run().unwrap();
}
#[test]
fn c2_independent_full_extent_rejects_equal_truncation() {
    let independent=decision_expected(3,5,12000,8192,248320,5120);
    let mut both_truncated=independent.clone();both_truncated.truncate(48);
    assert!(require_complete(both_truncated,&independent).is_err());
    assert!(require_complete(independent.clone(),&independent).is_ok());
    let prefix=prefix_expected(3,5,12001,248320,16);
    assert_eq!(prefix.iter().map(|e|e.shape[0]).sum::<usize>(),12001);
}
