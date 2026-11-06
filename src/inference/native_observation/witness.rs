//! Actual input manifests and narrowly scoped arithmetic witnesses.
//! An input declaration or an auxiliary encode is never forward progress.
use super::*;
use std::sync::Arc;

/// Qwen joint ordinary decode: one actual one-token span per packed matrix
/// row. The only source whose physical matrix may carry owner-less padding.
pub(super) const JOINT_SOURCE: &str = "qwen35.joint_decode.actual_slot_rows";

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
/// Closes a joint forward's own encoding: forward id and invocation epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EndMarker {
    forward_id: u64,
    epoch: u64,
}
pub(crate) struct Input {
    previous: Option<Arc<Descriptor>>,
    active: bool,
    end: Option<EndMarker>,
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
        | "deepseek4.prefill.actual_cooperative_pack"
        | JOINT_SOURCE => {
            (source != "deepseek4.prefill.actual_cooperative_pack"
                || (2..=4).contains(&spans.len()))
                && (source != JOINT_SOURCE || spans.len() <= MAX_JOINT_DECODE_ROWS)
                && spans.iter().enumerate().all(|(index, span)| {
                    spans[..index].iter().all(|old| old.slot != span.slot)
                        && (!matches!(
                            source,
                            "gemma4.decode.actual_token_slot_position_arrays" | JOINT_SOURCE
                        ) || span.count == 1)
                })
        }
        _ => false,
    }
}
impl Input {
    /// The end-marker decision: only an active guard that published a joint
    /// input closes it, exactly once (drop, early return or unwinding alike).
    fn take_end(&mut self) -> Option<EndMarker> {
        if self.active {
            self.end.take()
        } else {
            None
        }
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        if let Some(marker) = self.take_end() {
            publish_end(marker);
        }
        if self.active {
            INPUT.with(|value| *value.borrow_mut() = self.previous.take());
        }
    }
}

/// `scope_watermark` is the highest id handed out so far: every scope created
/// inside this forward lies in `(forward_id, scope_watermark]`.
fn publish_end(marker: EndMarker) {
    note(Hook::Marker);
    let watermark = NEXT.load(Ordering::Acquire).saturating_sub(1);
    match crate::serve::phase_trace::monotonic_ns() {
        Ok(end) => crate::serve::phase_trace::publish(
            "forward_input_end",
            serde_json::json!({
            "forward_id":marker.forward_id,"epoch":marker.epoch,
            "scope_watermark":watermark,"end_monotonic_ns":end}),
        ),
        Err(_) => crate::serve::phase_trace::invalidate("forward input end clock unavailable"),
    }
}

/// Hook decisions, counted per thread in tests only (publish is a no-op there).
#[derive(Clone, Copy)]
enum Hook {
    Input,
    Witness,
    Unmapped,
    Marker,
}
#[cfg(test)]
thread_local! {
    static HOOKS: Cell<[u32; 4]> = const { Cell::new([0; 4]) };
}
#[cfg(test)]
fn note(hook: Hook) {
    HOOKS.with(|value| {
        let mut counts = value.get();
        counts[hook as usize] += 1;
        value.set(counts);
    });
}
#[cfg(not(test))]
#[inline(always)]
fn note(_: Hook) {}
/// This thread's `[inputs, witnesses, unmapped, end markers]` since the last call.
#[cfg(test)]
pub(crate) fn take_hook_counts() -> [u32; 4] {
    HOOKS.with(|value| value.replace([0; 4]))
}

/// Call only where the actual ordered token tensor is constructed. Spans are
/// packed in this order; a scheduler's desired width is not an input manifest.
pub(crate) fn input(source: &'static str, spans: &[InputSpan]) -> Input {
    input_rows(source, spans, 0)
}

/// Logical rows plus the physical matrix row count. Padding rows are
/// owner-less physical rows and exist only for the joint source; that source
/// is ordinary target decode only (a joint verify is refused, not relabelled).
fn manifest(
    source: &str,
    spans: &[InputSpan],
    padding_rows: u32,
    role: &'static str,
    selected: &[crate::serve::phase_trace::SlotState],
    owners: &[crate::serve::phase_trace::SlotState],
) -> Option<(Vec<crate::serve::phase_trace::PhysicalRow>, u64)> {
    let joint = source == JOINT_SOURCE;
    if (padding_rows != 0 && !joint) || (joint && role != "target") || !source_layout(source, spans)
    {
        return None;
    }
    let mut rows = resolve_selected(spans, selected, owners)?;
    if role == "target_verify" {
        for row in &mut rows {
            row.phase = "verify";
        }
    }
    if joint && rows.iter().any(|row| row.phase != "ordinary") {
        return None;
    }
    let total = rows.iter().try_fold(u64::from(padding_rows), |sum, row| {
        sum.checked_add(u64::from(row.token_count))
    })?;
    Some((rows, total))
}

pub(super) fn input_rows(source: &'static str, spans: &[InputSpan], padding_rows: u32) -> Input {
    let mut descriptor = None;
    let mut end = None;
    let active = ARMED.load(Ordering::Acquire);
    if active {
        let role = ROLE.with(Cell::get);
        let rows = SELECTED.with(|selected| {
            OWNERS.with(|owners| {
                manifest(
                    source,
                    spans,
                    padding_rows,
                    role,
                    &selected.borrow(),
                    &owners.borrow(),
                )
            })
        });
        if let (Some((rows, total)), Some(id)) = (rows, next_id()) {
            let epoch = EPOCH.with(Cell::get);
            if epoch != 0 {
                let mut offset = 0u64;
                let packed: Vec<_> = rows.iter().map(|row| {
                    let begin = offset; offset += u64::from(row.token_count);
                    serde_json::json!({"matrix_row_start":begin,"matrix_row_end":offset,"logical":row})
                }).collect();
                let mut payload = serde_json::json!({
                "forward_id":id,"epoch":epoch,"source":source,
                "role":role,"matrix_rows":total,"packed_rows":packed,
                "operator_site":if source == "deepseek4.prefill.actual_cooperative_pack" {
                    "ffn_hc_input" } else { "pre_attention_input" },
                "execution_proven":false});
                if source == JOINT_SOURCE {
                    payload["padding_rows"] = serde_json::json!(padding_rows);
                    end = Some(EndMarker {
                        forward_id: id,
                        epoch,
                    });
                }
                crate::serve::phase_trace::publish("forward_input", payload);
                note(Hook::Input);
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
        end,
        _thread_bound: PhantomData,
    }
}

pub(crate) fn single_input(source: &'static str, start: usize, count: usize) -> Input {
    if !ARMED.load(Ordering::Acquire) {
        return Input {
            previous: None,
            active: false,
            end: None,
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

/// What one witnessed RMS call does with the current input descriptor.
enum Claim {
    /// No manifest is in place: publish `forward_witness_unmapped`.
    Unmapped,
    /// This forward's one witness was already claimed: publish nothing.
    Silent,
    Witness(Arc<Descriptor>),
}
fn claim(descriptor: Option<Arc<Descriptor>>) -> Claim {
    match descriptor {
        None => Claim::Unmapped,
        Some(descriptor) if descriptor.witness_claimed.swap(true, Ordering::AcqRel) => {
            Claim::Silent
        }
        Some(descriptor) => Claim::Witness(descriptor),
    }
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
    let descriptor = match claim(INPUT.with(|value| value.borrow().clone())) {
        Claim::Unmapped => {
            note(Hook::Unmapped);
            crate::serve::phase_trace::publish(
                "forward_witness_unmapped",
                serde_json::json!({
                "operator":"target_input_rms_norm","reason":"NoActualInputManifest",
                "progress_eligible":false}),
            );
            return None;
        }
        Claim::Silent => return None,
        Claim::Witness(descriptor) => descriptor,
    };
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
    note(Hook::Witness);
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
#[path = "witness_tests.rs"]
mod tests;
