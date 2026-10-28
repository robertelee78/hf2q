use super::*;
/// Caller invokes ONLY after its existing successful checked production return.
/// No native status inference, additional commit/wait or CPU score recomputation.
pub fn end_checked() -> Result<()> {
    checked("existing successful production return; no extra wait")
}
pub(super) fn primitive_end_checked() -> Result<()> {
    checked("primitive-only single checked drain; no model or production return")
}
fn checked(boundary_label: &str) -> Result<()> {
    ACTIVE.with(|s|{let mut s=s.borrow_mut();let Some(a)=s.as_mut()else{return Ok(())};
    let Some(c)=a.call.as_ref()else{ensure!(!a.pending,"pending routing without invocation");return Ok(())};
    ensure!(c.seen.len()==c.rows.len()*LAYERS,"missing required producer layer/row");
    a.pending=false;
    while a.read<a.records.len(){let rec=a.records[a.read].clone();let (scores,ids,status,selected,weights)=views(&a.bank,a.read,1)?;
        let witness=SelectionWitness{scores:&scores,ids:&ids,status:&status};
        let boundary=read_after_checked_drain(witness,1,E as u32,K as u32);
        let native_valid=boundary.as_ref().is_ok_and(|b|matches!(b.as_slice(),[SelectionBoundary::Scored{..}]));
        let selected=selected.as_logical_slice::<u32>()?;let weights=weights.as_logical_slice::<f32>()?;
        let values=scores.as_logical_slice::<f32>()?;let ids=ids.as_logical_slice::<u32>()?;
        let selection_valid=selected_valid(selected,weights,ids,values);
        event(a,json!({"kind":"routing_row","at":rec,"scores_f32_bits":values.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
            "boundary_ids":ids,"status":status.as_logical_slice::<u32>()?,"selected_ids":selected,"selected_weights_f32_bits":weights.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
            "native_boundary_valid":native_valid,"selected_copy_valid":selection_valid,
            "native_decode_error":boundary.as_ref().err().map(ToString::to_string),
            "checked_boundary":boundary_label,"native_boundary":boundary.as_ref().ok().map(|v|v.iter().map(project_boundary).collect::<Vec<_>>()),"qualified":false}))?;
        // Retain raw invalid status/bit patterns BEFORE refusing the case.
        ensure!(native_valid&&selection_valid,"invalid/N/A native boundary or actual selected copy");
        a.read+=1;
    }
    a.call=None;Ok(())
})
}
pub fn finish() -> Result<()> {
    ACTIVE.with(|s|{let mut s=s.borrow_mut();let Some(a)=s.as_mut()else{return Ok(())};
    ensure!(!a.pending&&a.call.is_none()&&a.read==a.records.len(),"unresolved routing invocation");
    for lane in 0..4 {for layer in 0..LAYERS {match a.mode {
        Mode::Ordinary|Mode::Cancelled=>for pos in 0..144 {ensure!(a.seen.contains(&(Namespace::Canonical,lane,pos,layer)),"canonical routing coverage incomplete");},
        Mode::Transaction=>{for (ns,end) in [(Namespace::T5Initial,128),(Namespace::T5Verify,132)] {for pos in 0..end {ensure!(a.seen.contains(&(ns,lane,pos,layer)),"T5 initial/verify routing missing");}}
            let tail=(129..144).filter(|&p|a.seen.contains(&(Namespace::T5Continuation,lane,p,layer))).collect::<Vec<_>>();
            ensure!(!tail.is_empty()&&(129..=132).contains(&tail[0])&&tail==((tail[0])..144).collect::<Vec<_>>(),"T5 continuation routing missing");}
    }if a.mode==Mode::Cancelled {for pos in 64..128 {ensure!(a.seen.contains(&(Namespace::UnpublishedCancelled,lane,pos,layer)),"cancelled-tail namespace missing");}}}}
    let count=a.records.len();let metadata=a.metadata;
    event(a,json!({"kind":"end","records":count,"payload_used":count*84,"metadata_bytes":metadata,
        "all_scored_rows_captured":true,"full_Q1_qualified":false,"native_model_parity_qualified":false}))?;s.take();Ok(())
})
}

pub(super) fn selected_valid(
    selected: &[u32],
    weights: &[f32],
    ids: &[u32],
    scores: &[f32],
) -> bool {
    selected.len() == K
        && weights.len() == K
        && ids.len() == 2
        && scores.len() == 2
        && selected.iter().all(|&id| id < E as u32)
        && selected.iter().collect::<BTreeSet<_>>().len() == K
        && weights.iter().all(|x| x.is_finite())
        && scores.iter().all(|x| x.is_finite())
        && ids[0] == selected[K - 1]
        && !selected.contains(&ids[1])
        && ids[1] < E as u32
        && scores[0] >= scores[1]
}

// Native diagnostic types intentionally need no serde dependency. Retain raw
// score bits, including invalid/nonfinite patterns, without JSON float coercion.
pub(super) fn project_boundary(value:&SelectionBoundary)->serde_json::Value {
    match value {
        SelectionBoundary::Scored{kth_id,kth_score,next_id,next_score}=>json!({"kind":"Scored","kth_id":kth_id,"kth_score_f32_bits":kth_score.to_bits(),"next_id":next_id,"next_score_f32_bits":next_score.to_bits()}),
        SelectionBoundary::AllExpertsSelected{kth_id,kth_score}=>json!({"kind":"AllExpertsSelected","kth_id":kth_id,"kth_score_f32_bits":kth_score.to_bits()}),
        SelectionBoundary::Invalid=>json!({"kind":"Invalid"}),
    }
}
