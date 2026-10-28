//! Actual input manifests and narrowly scoped arithmetic witnesses.
//! An input declaration or an auxiliary encode is never forward progress.
use super::*;
use std::sync::Arc;

struct Descriptor {
    id: u64,
    source: &'static str,
    rows: Vec<crate::serve::phase_trace::PhysicalRow>,
    total: u64,
    role: &'static str,
    witness_claimed: AtomicBool,
}
thread_local! {
    static INPUT: RefCell<Option<Arc<Descriptor>>> = const { RefCell::new(None) };
}
pub(crate) struct Input {
    previous: Option<Arc<Descriptor>>,
    active: bool,
    _thread_bound: PhantomData<Rc<()>>,
}
fn source_layout(source: &str, spans: &[InputSpan]) -> bool {
    if spans.is_empty() || spans.len() > 256 {
        return false;
    }
    match source {
        "qwen35.greedy.actual_tokens"
        | "gemma4.scalar_forward"
        | "deepseek4.scalar_verifier"
        | "deepseek4.sorted_cache_lane" => spans.len() == 1 && spans[0].count == 1,
        "qwen35.forward.actual_tokens"
        | "gemma4.prefill.actual_single_input"
        | "deepseek4.prefill.actual_cache_span"
        | "deepseek4.prefill.actual_cooperative_lane" => spans.len() == 1,
        "gemma4.decode.actual_token_slot_position_arrays"
        | "gemma4.prefill.actual_concat_descriptor"
        | "deepseek4.prefill.actual_cooperative_pack" => {
            (source != "deepseek4.prefill.actual_cooperative_pack"
                || (2..=4).contains(&spans.len()))
                && spans.iter().enumerate().all(|(index, span)| {
                    spans[..index].iter().all(|old| old.slot != span.slot)
                        && (source != "gemma4.decode.actual_token_slot_position_arrays"
                            || span.count == 1)
                })
        }
        _ => false,
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        if self.active {
            INPUT.with(|value| *value.borrow_mut() = self.previous.take());
        }
    }
}

/// Call only where the actual ordered token tensor is constructed. Spans are
/// packed in this order; a scheduler's desired width is not an input manifest.
pub(crate) fn input(source: &'static str, spans: &[InputSpan]) -> Input {
    let mut descriptor = None;
    let active = ARMED.load(Ordering::Acquire);
    if active {
        let rows = if source_layout(source, spans) {
            SELECTED.with(|selected| {
                OWNERS.with(|owners| resolve_selected(spans, &selected.borrow(), &owners.borrow()))
            })
        } else {
            None
        };
        if let (Some(mut rows), Some(id)) = (rows, next_id()) {
            let role = ROLE.with(Cell::get);
            if role == "target_verify" {
                for row in &mut rows {
                    row.phase = "verify";
                }
            }
            let total = rows
                .iter()
                .try_fold(0u64, |sum, row| sum.checked_add(u64::from(row.token_count)));
            if let Some(total) = total.filter(|_| EPOCH.with(Cell::get) != 0) {
                let mut offset = 0u64;
                let packed: Vec<_> = rows.iter().map(|row| {
                    let begin = offset; offset += u64::from(row.token_count);
                    serde_json::json!({"matrix_row_start":begin,"matrix_row_end":offset,"logical":row})
                }).collect();
                crate::serve::phase_trace::publish(
                    "forward_input",
                    serde_json::json!({
                    "forward_id":id,"epoch":EPOCH.with(Cell::get),"source":source,
                    "role":role,"matrix_rows":total,"packed_rows":packed,
                    "operator_site":if source == "deepseek4.prefill.actual_cooperative_pack" {
                        "ffn_hc_input" } else { "pre_attention_input" },
                    "execution_proven":false}),
                );
                descriptor = Some(Arc::new(Descriptor {
                    id,
                    source,
                    rows,
                    total,
                    role,
                    witness_claimed: AtomicBool::new(false),
                }));
            }
        }
        if descriptor.is_none() {
            crate::serve::phase_trace::invalidate(
                "actual forward input lacks current owner/epoch binding",
            );
        }
    }
    let previous = if active {
        INPUT.with(|value| value.replace(descriptor))
    } else {
        None
    };
    Input {
        previous,
        active,
        _thread_bound: PhantomData,
    }
}

pub(crate) fn single_input(source: &'static str, start: usize, count: usize) -> Input {
    if !ARMED.load(Ordering::Acquire) {
        return Input {
            previous: None,
            active: false,
            _thread_bound: PhantomData,
        };
    }
    let slot = SELECTED.with(|value| {
        let owners = value.borrow();
        (owners.len() == 1).then(|| owners[0].slot_id)
    });
    match (slot, u64::try_from(start), u32::try_from(count)) {
        (Some(slot), Ok(start), Ok(count)) => input(source, &[InputSpan { slot, start, count }]),
        _ => input(source, &[]),
    }
}

fn footprint(
    total: u64,
    rows: usize,
    cols: usize,
    elements: usize,
    bytes: usize,
    offset: u64,
    parent_bytes: usize,
    element_bytes: usize,
) -> Option<usize> {
    if rows == 0 || cols == 0 || u64::try_from(rows).ok()? != total {
        return None;
    }
    let expected = rows.checked_mul(cols)?;
    if elements != expected {
        return None;
    }
    let read_bytes = expected.checked_mul(element_bytes)?;
    if read_bytes > bytes
        || offset.checked_add(u64::try_from(read_bytes).ok()?)?
            > u64::try_from(parent_bytes).ok()?
    {
        return None;
    }
    Some(read_bytes)
}

/// The caller wraps ONLY the existing RMS-norm encode (one native arithmetic
/// dispatch), not its surrounding body, QKV wrapper or per-slot kernels.
/// A malformed observer descriptor invalidates metadata, never the real result.
fn prepare(
    matrix: &mlx_native::MlxBuffer,
    rows: usize,
    cols: usize,
    layer: Option<usize>,
) -> Option<(Operation, Option<ObservationScope>)> {
    if !ARMED.load(Ordering::Acquire) {
        return None;
    }
    let descriptor = INPUT.with(|value| value.borrow().clone());
    let Some(descriptor) = descriptor else {
        crate::serve::phase_trace::publish(
            "forward_witness_unmapped",
            serde_json::json!({
            "operator":"target_input_rms_norm","reason":"NoActualInputManifest",
            "progress_eligible":false}),
        );
        return None;
    };
    if descriptor.witness_claimed.swap(true, Ordering::AcqRel) {
        return None;
    }
    let bytes = footprint(
        descriptor.total,
        rows,
        cols,
        matrix.element_count(),
        matrix.data_byte_len(),
        matrix.byte_offset(),
        matrix.byte_len(),
        matrix.dtype().size_of(),
    );
    let Some(bytes) = bytes else {
        crate::serve::phase_trace::invalidate(
            "actual RMS matrix shape/view disagrees with forward input",
        );
        return None;
    };
    let spans: Vec<_> = descriptor
        .rows
        .iter()
        .map(|row| InputSpan {
            slot: row.slot_id,
            start: row.token_position,
            count: row.token_count,
        })
        .collect();
    let scope_guard = operation("target.input_rms_norm", &spans, descriptor.role);
    let scope = CURRENT.with(|value| *value.borrow());
    crate::serve::phase_trace::publish(
        "forward_witness",
        serde_json::json!({
        "forward_id":descriptor.id,"scope":scope,"source":descriptor.source,
        "operator":"target_input_rms_norm","layer":layer,"role":descriptor.role,
        "operator_site":if descriptor.source == "deepseek4.prefill.actual_cooperative_pack" {
            "ffn_hc_input" } else { "pre_attention_input" },
        "rows":rows,"columns":cols,"dtype":format!("{:?}",matrix.dtype()),
        "input_parent_identity":matrix.contents_ptr() as usize,
        "input_byte_offset":matrix.byte_offset(),"input_read_bytes":bytes,
        "input_data_bytes":matrix.data_byte_len(),"input_parent_bytes":matrix.byte_len(),
        "dispatches_expected":1,"execution_proven":false,"full_body_batching_proven":false}),
    );
    Some((scope_guard, scope))
}

pub(crate) fn witness_rms_norm<'a, R>(
    session: &mut mlx_native::graph::GraphSession<'a>,
    matrix: &mlx_native::MlxBuffer,
    rows: usize,
    cols: usize,
    layer: Option<usize>,
    encode: impl FnOnce(&mut mlx_native::graph::GraphSession<'a>) -> R,
) -> R {
    let Some((scope_guard, scope)) = prepare(matrix, rows, cols, layer) else {
        return encode(session);
    };
    session.encoder_mut().set_observation_scope(scope);
    // Restore metadata even if the existing encode panics; inference panic and
    // Result semantics are unchanged. No raw borrowed-encoder pointer survives.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encode(session)));
    let epoch = EPOCH.with(Cell::get);
    session
        .encoder_mut()
        .set_observation_scope(auxiliary(epoch));
    drop(scope_guard);
    match result {
        Ok(value) => value,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

pub(crate) fn witness_command_rms_norm<R>(
    encoder: &mut mlx_native::CommandEncoder,
    matrix: &mlx_native::MlxBuffer,
    rows: usize,
    cols: usize,
    encode: impl FnOnce(&mut mlx_native::CommandEncoder) -> R,
) -> R {
    let Some((scope_guard, scope)) = prepare(matrix, rows, cols, None) else {
        return encode(encoder);
    };
    encoder.set_observation_scope(scope);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encode(encoder)));
    encoder.set_observation_scope(auxiliary(EPOCH.with(Cell::get)));
    drop(scope_guard);
    match result {
        Ok(value) => value,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_witness_footprint_requires_all_rows_and_checked_view_extent() {
        assert_eq!(footprint(4, 4, 8, 32, 128, 64, 192, 4), Some(128));
        assert_eq!(footprint(4, 1, 8, 8, 32, 0, 32, 4), None);
        assert_eq!(footprint(4, 4, 8, 32, 127, 64, 192, 4), None);
        assert_eq!(footprint(4, 4, 8, 32, 128, 65, 192, 4), None);
        assert_eq!(footprint(4, 4, 8, 31, 128, 64, 192, 4), None);
        assert_eq!(footprint(4, 4, usize::MAX, 0, 0, 0, 0, 4), None);
        assert_eq!(footprint(4, 4, 8, 32, 128, u64::MAX, 192, 4), None);
    }
    #[test]
    fn actual_source_layout_does_not_relabel_scalar_or_temporal_as_independent_quad() {
        let scalar = [InputSpan {
            slot: 0,
            start: 3,
            count: 1,
        }];
        assert!(source_layout("qwen35.greedy.actual_tokens", &scalar));
        let temporal = [InputSpan {
            slot: 0,
            start: 3,
            count: 4,
        }];
        assert!(!source_layout("qwen35.greedy.actual_tokens", &temporal));
        assert!(source_layout("qwen35.forward.actual_tokens", &temporal));
        assert!(!source_layout(
            "gemma4.decode.actual_token_slot_position_arrays",
            &temporal
        ));
        let independent = std::array::from_fn::<_, 4, _>(|slot| InputSpan {
            slot: slot as u32,
            start: 3,
            count: 1,
        });
        assert!(source_layout(
            "gemma4.decode.actual_token_slot_position_arrays",
            &independent
        ));
        assert!(!source_layout("qwen35.forward.actual_tokens", &independent));
        assert!(!source_layout(
            "gemma4.decode.actual_token_slot_position_arrays",
            &[scalar[0], scalar[0]]
        ));
    }
}
