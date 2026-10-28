//! Q1 test adapter over actual production packed Gemma primitives and the SAME
//! checked cursor publication helper. No scheduler membership is inferred.
use super::*;
fn require_slots(slots:&[SlotId])->Result<()> {
    anyhow::ensure!(slots.len()==4 && slots.iter().map(|slot|slot.0)
        .collect::<std::collections::BTreeSet<_>>()==[0,1,2,3].into_iter().collect(),
        "four distinct actual Gemma slots required");Ok(())
}
pub(crate) fn prefill(model:&mut GemmaLoadedModel,seqs:&[(Vec<u32>,SlotId)]) -> Result<()> {
    anyhow::ensure!(seqs.len()==4 && seqs.iter().all(|(t,_)|t.len()==128),"frozen four128 Gemma inputs");
    require_slots(&seqs.iter().map(|(_,slot)|*slot).collect::<Vec<_>>())?;
    let mut guard=Gemma4KvGuard::take(model)?;
    let commits=seqs.iter().map(|(_,slot)|(*slot,0,128)).collect::<Vec<_>>();
    for &(slot,before,_) in &commits {validate_gemma4_slot_cursor(&guard,slot,before)?;}
    let lm=&mut *guard.model;
    let out=lm.weights.forward_prefill_batched_multi_seq(seqs,
        guard.hybrid.as_ref().context("Gemma actual Hybrid cohort scaffold")?,8192,&mut lm.ctx)?;
    anyhow::ensure!(out.first_tokens.len()==4 && out.logits.len()==4,"actual packed head output cardinality");
    commit_gemma4_slot_cursors(&mut guard,&commits)?;
    clear_gemma4_self_mounts(guard.model);
    Ok(())
}
pub(crate) fn decode(model:&mut GemmaLoadedModel,tokens:&[u32],slots:&[SlotId],position:usize)
    ->Result<Vec<f32>> {
    anyhow::ensure!(std::env::var("HF2Q_BATCHED_BODY").as_deref()!=Ok("0"),
        "Gemma T2 requires actual selected packed body; no override of opt-out");
    anyhow::ensure!(tokens.len()==4 && slots.len()==4 && (128..144).contains(&position),
        "Gemma actual four-row decode geometry");
    require_slots(slots)?;
    let mut guard=Gemma4KvGuard::take(model)?;
    for &slot in slots {validate_gemma4_slot_cursor(&guard,slot,position)?;}
    clear_gemma4_self_mounts(guard.model);
    let mut fused=None;
    let lm=&mut *guard.model;
    let mut regime=crate::inference::models::gemma4::batched_body::BatchedKvRegime::Hybrid(
        guard.hybrid.as_mut().context("Gemma actual Hybrid decode scaffold")?);
    let hidden=lm.weights.forward_decode_body_batched(tokens,slots,&[position;4],&mut regime,&mut fused,&mut lm.ctx)?;
    let head=if let Some(head)=fused {head}else{lm.weights.lm_head_batched(&hidden,4,&mut lm.ctx)?};
    anyhow::ensure!(head.logits.len()==4*lm.weights.vocab_size,"Gemma actual four-row logits extent");
    let commits=slots.iter().map(|&slot|(slot,position+1)).collect::<Vec<_>>();
    commit_gemma4_decode_cursors(&mut guard,&commits)?;
    Ok(head.logits)
}

#[test]
fn q1_gemma_actual_cohort_cursor_adapter_refuses_duplicate_or_missing_slots() {
    assert!(require_slots(&[SlotId(0),SlotId(1),SlotId(2),SlotId(3)]).is_ok());
    assert!(require_slots(&[SlotId(3),SlotId(2),SlotId(1),SlotId(0)]).is_ok());
    assert!(require_slots(&[SlotId(0),SlotId(1),SlotId(2),SlotId(2)]).is_err());
    assert!(require_slots(&[SlotId(0),SlotId(1),SlotId(2)]).is_err());
}
