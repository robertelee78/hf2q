//! Finite first-fork diagnostic. Test-only; no serving/math/ordering changes.
use super::*;
use crate::adr063_invariance::qwen_continuation::payload;

pub(super) struct Captures {
    pub(super) post_draft: Option<Store>,
    pub(super) pre_catchup: Store,
    pub(super) post_catchup: Store,
    post_draft_seen: bool,
    pre_rows: Vec<usize>,
    post_rows: Vec<usize>,
}
pub(super) fn committed(cfg: &Qwen35Config) -> Result<Vec<ExpectedTensor>> {
    Ok(plan::committed(cfg,262144)?.into_iter().filter(|e| e.at.agent == "agent-0" &&
        if e.at.stage == Stage::Logits {e.at.logical_row_start < 131}
        else {e.at.logical_step < 144}).collect())
}
pub(super) fn raw(cfg: &Qwen35Config) -> Vec<ExpectedTensor> {
    plan::raw(cfg).into_iter().filter(|e|e.at.agent == "agent-0").map(|mut e| {
        if matches!(e.at.stage,Stage::SpecVerifierLogits|Stage::SpecVerifierHidden) {e.shape[0]=2;}
        e
    }).collect()
}
pub(super) fn output_plan(cfg: &Qwen35Config) -> Vec<ExpectedTensor> {
    let mut result=Vec::new();
    for position in 129..131 {
        for (stage,width) in [(Stage::SpecVerifierLogits,cfg.vocab_size),
            (Stage::SpecVerifierHidden,cfg.hidden_size)] {
            result.push(ExpectedTensor {at:plan::at(0,position,cfg.num_hidden_layers,stage),
                scalar:Scalar::F32,shape:vec![1,width as usize]});
        }
    }
    result
}
pub(super) fn draft_plan(cfg: &Qwen35Config) -> Result<Vec<ExpectedTensor>> {
    qwen_cache::manifest(cfg,262144,"agent-0",129,129)
}
pub(super) fn payload_plan(cfg: &Qwen35Config, forced: bool) -> Result<u64> {
    let base=payload(&committed(cfg)?)?.checked_add(payload(&output_plan(cfg))?.checked_mul(2)
        .context("output payload overflow")?).context("diagnostic payload overflow")?;
    if forced { base.checked_add(payload(&raw(cfg))?).context("raw payload overflow")?
        .checked_add(payload(&draft_plan(cfg)?)?).context("forced payload overflow") }
    else {Ok(base)}
}
pub(super) fn make(
    cfg: &Qwen35Config, forced: bool,
    make: &impl Fn(&str,Vec<ExpectedTensor>)->Result<Store>,
) -> Result<Captures> {
    let pair=payload_plan(cfg,false)?.checked_add(payload_plan(cfg,true)?)
        .context("pair overflow")?;
    anyhow::ensure!(pair <= 1024*1024*1024,"first-fork tensor pair exceeds1GiB; no truncation");
    Ok(Captures {post_draft:if forced {Some(make("post-draft-discard129",draft_plan(cfg)?)?)}else{None},
        pre_catchup:make("target-before-mtp",output_plan(cfg))?,
        post_catchup:make("target-after-mtp",output_plan(cfg))?,
        post_draft_seen:false,pre_rows:vec![],post_rows:vec![]})
}
fn outputs(store: &mut Store, cfg: &Qwen35Config, position: usize,
    logits: &[f32], hidden: &[f32], rows: usize) -> Result<()> {
    anyhow::ensure!(position >=129 && position+rows <=131 &&
        logits.len() >= rows*cfg.vocab_size as usize && hidden.len() >= rows*cfg.hidden_size as usize,
        "complete first-two output rows");
    for row in 0..rows {
        for (stage,width,values) in [(Stage::SpecVerifierLogits,cfg.vocab_size as usize,logits),
            (Stage::SpecVerifierHidden,cfg.hidden_size as usize,hidden)] {
            let begin=row*width;let end=begin+width;
            store.capture(plan::at(0,position+row,cfg.num_hidden_layers,stage),Scalar::F32,
                vec![1,width],values.as_ptr() as usize as u64,std::mem::size_of_val(values) as u64,
                (begin*4) as u64,None,"actual target checked return; first-fork copy before/after MTP",
                bytemuck::cast_slice(&values[begin..end])).map_err(anyhow::Error::msg)?;
        }
    }
    Ok(())
}
pub(crate) fn after_draft_discard(position: i32, kv: &HybridKvCache) -> Result<()> {
    update(|a| {
        let Some(c)=a.first_fork.as_mut() else {return Ok(())};
        anyhow::ensure!(a.mode==Mode::ForcedProposal && a.lane==0 && position==129 &&
            !c.post_draft_seen,"one actual post-draft checkpoint only");
        anyhow::ensure!(kv.mtp_slot.as_ref().is_some_and(|m|m.current_len[0]==129),
            "real speculative draft cache discarded to129");
        qwen_cache::after_checked_terminal(c.post_draft.as_mut().context("forced checkpoint absent")?,
            kv,a.cfg.num_hidden_layers,"agent-0",129)?;
        c.post_draft_seen=true;Ok(())
    })
}
pub(crate) fn verifier_pre_catchup(position: i32, values: &MlxBuffer, hidden: &MlxBuffer) -> Result<()> {
    update(|a| {
        let Some(c)=a.first_fork.as_mut() else {return Ok(())};
        anyhow::ensure!(a.mode==Mode::ForcedProposal && a.lane==0 && position==129 &&
            c.post_draft_seen && c.pre_rows.is_empty() && values.dtype()==DType::F32 &&
            hidden.dtype()==DType::F32 && values.element_count()==4*a.cfg.vocab_size as usize &&
            hidden.element_count()==4*a.cfg.hidden_size as usize,"actual checked R4 target before MTP");
        outputs(&mut c.pre_catchup,&a.cfg,129,values.as_logical_slice::<f32>()?,
            hidden.as_logical_slice::<f32>()?,2)?;
        c.pre_rows.extend([129,130]);Ok(())
    })
}
pub(crate) fn ordinary_pre_catchup(position: i32, logits: &[f32], hidden: &MlxBuffer) -> Result<()> {
    ordinary(position,logits,hidden,true)
}
pub(crate) fn ordinary_post_catchup(position: i32, logits: &[f32], hidden: &MlxBuffer) -> Result<()> {
    ordinary(position,logits,hidden,false)
}
fn ordinary(position: i32, logits: &[f32], hidden: &MlxBuffer, before: bool) -> Result<()> {
    update(|a| {
        let Some(c)=a.first_fork.as_mut() else {return Ok(())};
        if position==128 {return Ok(())}
        anyhow::ensure!(a.mode==Mode::OrdinaryBaseline && a.lane==0 && (129..131).contains(&position)
            && hidden.dtype()==DType::F32 && hidden.element_count()==a.cfg.hidden_size as usize &&
            logits.len()==a.cfg.vocab_size as usize,"actual scalar129/130 output boundary");
        let position=position as usize;
        let (store,seen)=if before {(&mut c.pre_catchup,&mut c.pre_rows)}
            else {(&mut c.post_catchup,&mut c.post_rows)};
        anyhow::ensure!(seen.len()==position-129,"scalar capture order");
        outputs(store,&a.cfg,position,logits,hidden.as_logical_slice::<f32>()?,1)?;
        seen.push(position);Ok(())
    })
}
pub(super) fn post_catchup(a: &mut Active, position: i32, values: &MlxBuffer,
    hidden: &MlxBuffer) -> Result<()> {
    let Some(c)=a.first_fork.as_mut() else {return Ok(())};
    anyhow::ensure!(a.lane==0 && position==129 && c.pre_rows==[129,130] &&
        c.post_rows.is_empty(),"one checked verifier catch-up copy");
    outputs(&mut c.post_catchup,&a.cfg,129,values.as_logical_slice::<f32>()?,
        hidden.as_logical_slice::<f32>()?,2)?;
    c.post_rows.extend([129,130]);Ok(())
}
fn lane_scope(mode: Mode, lane: usize, counts: [usize;4], ordinary: &[Vec<usize>;4],
    pre: &[usize], post: &[usize], pending: bool, draft: bool, transitions: usize) -> bool {
    if lane!=0 || counts!=[4,0,0,0] || ordinary[1..].iter().any(|v|!v.is_empty()) ||
        pending || pre!=[129,130] || post!=[129,130] {return false}
    match mode {
        Mode::OrdinaryBaseline => ordinary[0]==[128,129,130] && transitions==0 && !draft,
        Mode::ForcedProposal => ordinary[0]==[128] && transitions==1 && draft,
        Mode::Preparation => false,
    }
}
pub(super) fn validate(a: &Active) -> Result<()> {
    let c=a.first_fork.as_ref().context("missing explicit first-fork capture")?;
    anyhow::ensure!(lane_scope(a.mode,a.lane,std::array::from_fn(|i|a.selected[i].len()),
        &a.ordinary_rows,&c.pre_rows,&c.post_rows,a.pending_logits.is_some(),
        c.post_draft_seen,a.transitions.len()),"finite actual lane0/target/MTP route incomplete");
    Ok(())
}
pub(super) fn finish(mut c: Captures, failure: Option<&str>) -> Result<serde_json::Value> {
    let mut stores=vec![("target-before-mtp",c.pre_catchup),("target-after-mtp",c.post_catchup)];
    if let Some(s)=c.post_draft.take() {stores.push(("post-draft-discard129",s));}
    let mut result=serde_json::Map::new();
    for (name,mut store) in stores {
        if let Some(error)=failure {store.fail(error.into());}
        let trace=store.finish().map_err(anyhow::Error::msg)?;
        result.insert(name.into(),serde_json::json!({"records":trace.coverage.captured.len(),
            "errors":trace.coverage.errors,"target_and_mtp_separate_by_layer":true}));
    }
    Ok(serde_json::Value::Object(result))
}
#[test]
#[ignore="explicit frozen agent0 first-fork packet only; actual model; root supervised"]
fn q1_qwen_t5_first_fork_actual() {super::runner::run_first_fork().unwrap();}
#[test]
fn first_fork_scope_and_disk_bound_do_not_relabel_four_lanes_or_sixteen_rows() {
    let cfg=qwen_cache::selected_header_fixture(true);
    let c=committed(&cfg).unwrap();
    assert!(c.iter().all(|e|e.at.agent=="agent-0"));
    assert_eq!(c.iter().filter(|e|e.at.stage==Stage::Logits).count(),3);
    assert!(c.iter().all(|e| if e.at.stage==Stage::Logits {(128..131).contains(&e.at.logical_row_start)}
        else {[129,131].contains(&e.at.logical_step)}));
    assert_eq!(output_plan(&cfg).len(),4);
    assert_eq!(raw(&cfg).len(),5);
    assert!(raw(&cfg).iter().filter(|e|matches!(e.at.stage,Stage::SpecVerifierLogits|Stage::SpecVerifierHidden))
        .all(|e|e.shape[0]==2));
    assert_eq!(payload_plan(&cfg,false).unwrap(),330_014_848);
    assert_eq!(payload_plan(&cfg,true).unwrap(),493_497_024);
    assert_eq!(payload_plan(&cfg,false).unwrap()+payload_plan(&cfg,true).unwrap(),823_511_872);
    assert!(payload_plan(&cfg,false).unwrap()+payload_plan(&cfg,true).unwrap()<1024*1024*1024);
}

#[test]
fn first_fork_actual_axis_refuses_stale_missing_or_other_lane_credit() {
    let ordinary=[vec![128,129,130],vec![],vec![],vec![]];
    let forced=[vec![128],vec![],vec![],vec![]];
    let check=|mode,lane,counts,rows:&[Vec<usize>;4],pre:&[usize],post:&[usize],pending,draft,rounds|
        lane_scope(mode,lane,counts,rows,pre,post,pending,draft,rounds);
    assert!(check(Mode::OrdinaryBaseline,0,[4,0,0,0],&ordinary,&[129,130],&[129,130],false,false,0));
    assert!(check(Mode::ForcedProposal,0,[4,0,0,0],&forced,&[129,130],&[129,130],false,true,1));
    assert!(!check(Mode::ForcedProposal,1,[4,0,0,0],&forced,&[129,130],&[129,130],false,true,1));
    assert!(!check(Mode::ForcedProposal,0,[4,1,0,0],&forced,&[129,130],&[129,130],false,true,1));
    assert!(!check(Mode::ForcedProposal,0,[4,0,0,0],&forced,&[129],&[129,130],false,true,1));
    assert!(!check(Mode::ForcedProposal,0,[4,0,0,0],&forced,&[129,130],&[129,130],true,true,1));
    assert!(!check(Mode::ForcedProposal,0,[4,0,0,0],&forced,&[129,130],&[129,130],false,false,1));
    assert!(!check(Mode::ForcedProposal,0,[4,0,0,0],&forced,&[129,130],&[129,130],false,true,0));
    assert!(!check(Mode::Preparation,0,[4,0,0,0],&ordinary,&[129,130],&[129,130],false,false,0));
}
