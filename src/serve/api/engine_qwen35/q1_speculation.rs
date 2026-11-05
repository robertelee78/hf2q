//! Q1 test-only real MTP transaction hooks. No synthetic accept/reject decision.
use super::*;
use crate::adr063_invariance::{
    capture, protocol::Workload, qwen_cache, qwen_speculation as plan, receipt::*, store::Store,
};
use crate::inference::models::qwen35::Qwen35Config;
use mlx_native::DType;
use std::cell::RefCell;
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub(super) enum Mode {
    Preparation,
    OrdinaryBaseline,
    ForcedProposal,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Carry {
    pub token: u32,
    pub terminal: bool,
}
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Frozen {
    pub workload: Workload,
    pub final_carry: [Carry; 4],
    pub forced_bad_ids: [u32; 4],
}
#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct Decision {
    pub position: usize,
    pub row: usize,
    pub token: u32,
    pub terminal: bool,
}
#[derive(serde::Serialize)]
pub(super) struct Transition {
    pub lane: usize,
    pub entry: usize,
    pub matched: usize,
    pub rejected: usize,
    pub valid_inputs: usize,
    pub carry_row: usize,
    pub committed: usize,
    pub actual_output: Vec<u32>,
}
pub(super) struct Active {
    id: u64,
    pub mode: Mode,
    pub cfg: Qwen35Config,
    pub capacity: u32,
    pub prefixes: Vec<Vec<u32>>,
    pub frozen: Option<Frozen>,
    pub eos: Vec<u32>,
    pub lane: usize,
    pub committed: Option<Store>,
    pub raw: Option<Store>,
    pub selected: [Vec<Carry>; 4],
    pub transitions: Vec<Transition>,
    pub ordinary_rows: [Vec<usize>; 4],
    pending_logits: Option<Vec<f32>>,
    first_fork: Option<first_fork::Captures>,
    canonical: Vec<Decision>,
}
impl Active {
    pub fn new(
        mode: Mode,
        cfg: Qwen35Config,
        prefixes: Vec<Vec<u32>>,
        frozen: Option<Frozen>,
        eos: Vec<u32>,
        committed: Option<Store>,
        raw: Option<Store>,
    ) -> Self {
        Self {
            id: 0,
            mode,
            cfg,
            capacity: 262144,
            prefixes,
            frozen,
            eos,
            lane: 0,
            committed,
            raw,
            selected: std::array::from_fn(|_| Vec::new()),
            transitions: Vec::new(),
            ordinary_rows: std::array::from_fn(|_| Vec::new()),
            pending_logits: None,
            first_fork: None,
            canonical: Vec::new(),
        }
    }
}
thread_local! {static ACTIVE:RefCell<Option<Active>>=const{RefCell::new(None)};}
static NEXT_SCOPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
pub(super) struct Scope {
    id: u64,
    _same_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|s| {
            let mut s = s.borrow_mut();
            if s.as_ref().is_some_and(|a| a.id == self.id) {
                s.take();
            }
        });
    }
}
pub(super) fn install(mut active: Active) -> Result<Scope> {
    ACTIVE.with(|s| {
        anyhow::ensure!(s.borrow().is_none(), "nested T5 scope");
        // Same loop as the deprecated `fetch_update`; `try_update` is newer than the MSRV.
        use std::sync::atomic::Ordering::Relaxed;
        let mut current = NEXT_SCOPE.load(Relaxed);
        let id = loop {
            let next = current
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("T5 scope ID overflow"))?;
            match NEXT_SCOPE.compare_exchange_weak(current, next, Relaxed, Relaxed) {
                Ok(previous) => break previous,
                Err(actual) => current = actual,
            }
        };
        active.id = id;
        *s.borrow_mut() = Some(active);
        Ok(Scope {
            id,
            _same_thread: std::marker::PhantomData,
        })
    })
}
pub(super) fn finish() -> Result<Active> {
    ACTIVE.with(|s| s.borrow_mut().take().context("T5 scope absent"))
}
fn update(f: impl FnOnce(&mut Active) -> Result<()>) -> Result<()> {
    ACTIVE.with(|s| {
        let mut s = s.borrow_mut();
        if let Some(active) = s.as_mut() {
            f(active)
        } else {
            Ok(())
        }
    })
}
pub(super) fn enter_lane(lane: usize) -> Result<()> {
    update(|a| {
        anyhow::ensure!(lane < 4 && a.pending_logits.is_none(), "bad/stale T5 lane");
        a.lane = lane;
        a.canonical.clear();
        Ok(())
    })
}
fn selected(a: &mut Active, position: usize, row: usize, token: u32, terminal: bool) -> Result<()> {
    let index = a.selected[a.lane].len();
    anyhow::ensure!(
        index <= 16 && token < a.cfg.vocab_size,
        "canonical decision extent"
    );
    anyhow::ensure!(
        !terminal || index == 16,
        "preparation/trajectory stopped before requiredg15"
    );
    if let Some(f) = &a.frozen {
        let expected = if index < 16 {
            Carry {
                token: f.workload.lanes[a.lane][128 + index],
                terminal: false,
            }
        } else {
            f.final_carry[a.lane]
        };
        anyhow::ensure!(
            Carry { token, terminal } == expected,
            "actual canonical lineage differs from frozen ordinary preparation"
        );
    }
    a.selected[a.lane].push(Carry { token, terminal });
    let _ = (position, row);
    Ok(())
}
pub(super) fn seed(token: u32, terminal: bool) -> Result<()> {
    update(|a| {
        anyhow::ensure!(a.selected[a.lane].is_empty(), "duplicate seed");
        selected(a, 127, 0, token, terminal)
    })
}
fn logits(
    store: &mut Store,
    cfg: &Qwen35Config,
    lane: usize,
    position: usize,
    all: &[f32],
    row: usize,
) -> Result<()> {
    let v = cfg.vocab_size as usize;
    let begin = row.checked_mul(v).context("logit offset")?;
    let slice = all
        .get(begin..begin + v)
        .context("complete target row absent")?;
    store
        .capture(
            plan::at(lane, position, cfg.num_hidden_layers, Stage::Logits),
            Scalar::F32,
            vec![1, v],
            all.as_ptr() as usize as u64,
            std::mem::size_of_val(all) as u64,
            (begin * 4) as u64,
            None,
            "actual raw target before canonical CPU semantic mutation; checked target+MTP drain",
            bytemuck::cast_slice(slice),
        )
        .map_err(anyhow::Error::msg)
}
fn ids(
    store: &mut Store,
    cfg: &Qwen35Config,
    lane: usize,
    stage: Stage,
    values: &[u32],
) -> Result<()> {
    let bytes = bytemuck::cast_slice(values);
    store
        .capture(
            plan::at(lane, 129, cfg.num_hidden_layers, stage),
            Scalar::U32,
            vec![values.len()],
            values.as_ptr() as usize as u64,
            bytes.len() as u64,
            0,
            None,
            "actual native proposal/canonical decision IDs, not predicted event counts",
            bytes,
        )
        .map_err(anyhow::Error::msg)
}
fn cache(a: &mut Active, cache: &HybridKvCache, position: usize) -> Result<()> {
    if plan::CHECKPOINTS.contains(&position) {
        anyhow::ensure!(
            cache
                .mtp_slot
                .as_ref()
                .is_some_and(|m| m.current_len[0] as usize == position),
            "active coherent MTP prefix required"
        );
        if let Some(store) = a.committed.as_mut() {
            qwen_cache::after_checked_terminal(
                store,
                cache,
                a.cfg.num_hidden_layers,
                &format!("agent-{}", a.lane),
                position,
            )?;
        }
    }
    Ok(())
}
pub(super) fn ordinary_buffers(position: i32, kv: &HybridKvCache, values: &[f32]) -> Result<()> {
    update(|a| {
        let position = usize::try_from(position)?;
        anyhow::ensure!(
            (128..144).contains(&position) && values.len() == a.cfg.vocab_size as usize,
            "ordinary raw-logit extent"
        );
        if let Some(store) = a.committed.as_mut() {
            logits(store, &a.cfg, a.lane, position, values, 0)?;
        }
        a.ordinary_rows[a.lane].push(position);
        cache(a, kv, position + 1)
    })
}
pub(super) fn ordinary_decision(position: i32, token: u32, terminal: bool) -> Result<()> {
    update(|a| selected(a, usize::try_from(position)?, 0, token, terminal))
}
pub(super) fn after_drafts(
    position: i32,
    target: u32,
    mtp: u32,
    original: Vec<u32>,
) -> Result<Vec<u32>> {
    ACTIVE.with(|s| {
        let mut s = s.borrow_mut();
        let Some(a) = s.as_mut() else {
            return Ok(original);
        };
        anyhow::ensure!(
            a.mode == Mode::ForcedProposal
                && position == 129
                && target == 129
                && mtp == 129
                && original.len() == 3
                && original.iter().all(|&id| id < a.cfg.vocab_size),
            "explicit actual K3 entry required"
        );
        anyhow::ensure!(
            a.selected[a.lane].len() == 2 && a.transitions.len() == a.lane,
            "exactly one K3 round per lane"
        );
        let f = a.frozen.as_ref().context("frozen fixture absent")?;
        let good = f.workload.lanes[a.lane][130];
        let bad = f.forced_bad_ids[a.lane];
        anyhow::ensure!(
            bad == plan::bad_proposal(f.workload.lanes[a.lane][131], a.cfg.vocab_size, &a.eos)?,
            "proposal wasn't frozen deterministic valid reject"
        );
        let forced = vec![good, bad, original[2]];
        let raw = a.raw.as_mut().context("raw store absent")?;
        ids(raw, &a.cfg, a.lane, Stage::SpecOriginalDraftIds, &original)?;
        ids(raw, &a.cfg, a.lane, Stage::SpecForcedDraftIds, &forced)?;
        Ok(forced)
    })
}
pub(super) fn verified_buffers(
    position: i32,
    values: &MlxBuffer,
    hidden: &MlxBuffer,
) -> Result<()> {
    update(|a| {
        anyhow::ensure!(
            a.mode == Mode::ForcedProposal && position == 129 && a.pending_logits.is_none(),
            "unexpected verifier capture"
        );
        anyhow::ensure!(
            values.dtype() == DType::F32
                && values.element_count() == 4 * a.cfg.vocab_size as usize
                && hidden.dtype() == DType::F32
                && hidden.element_count() == 4 * a.cfg.hidden_size as usize,
            "raw verifier extent/scalar"
        );
        first_fork::post_catchup(a, position, values, hidden)?;
        let captured_rows = if a.first_fork.is_some() { 2 } else { 4 };
        let raw = a.raw.as_mut().context("raw store")?;
        for (stage, buffer, shape) in [
            (
                Stage::SpecVerifierLogits,
                values,
                vec![captured_rows, a.cfg.vocab_size as usize],
            ),
            (
                Stage::SpecVerifierHidden,
                hidden,
                vec![captured_rows, a.cfg.hidden_size as usize],
            ),
        ] {
            let elements = shape.iter().product();
            let view = if a.first_fork.is_some() {
                buffer.slice_view(0, elements).with_shape(shape.clone())?
            } else { buffer.with_shape(shape.clone())? };
            capture::after_checked_drain(
                raw,
                plan::at(a.lane, 129, a.cfg.num_hidden_layers, stage),
                &view,
                &shape,
                None,
                "real four-row verifier +MTP catchup checked; before canonical mutation",
            )
            .map_err(anyhow::Error::msg)?;
        }
        a.pending_logits = Some(values.as_logical_slice::<f32>()?.to_vec());
        Ok(())
    })
}
pub(super) fn canonical_decision(
    position: i32,
    row: usize,
    token: u32,
    terminal: bool,
) -> Result<()> {
    update(|a| {
        anyhow::ensure!(
            a.mode == Mode::ForcedProposal
                && position == 129
                && row == a.canonical.len()
                && row < 4,
            "actual canonical walk order"
        );
        a.canonical.push(Decision {
            position: position as usize,
            row,
            token,
            terminal,
        });
        Ok(())
    })
}
pub(super) fn state_committed(
    position: i32,
    kv: &HybridKvCache,
    p: &Qwen35VerifiedBlockPlan,
) -> Result<()> {
    update(|a| {
        let cursor = kv.sequence_len_for_slot(SlotId(0))? as usize;
        a.transitions.push(Transition {
            lane: a.lane,
            entry: usize::try_from(position)?,
            matched: p.matched_drafts,
            rejected: p.rejected_drafts,
            valid_inputs: p.valid_input_tokens,
            carry_row: p.carry_hidden_row,
            committed: cursor,
            actual_output: p.output.iter().copied().collect(),
        });
        anyhow::ensure!(
            accepted_rejected_transition(a.mode, position, cursor, p, a.canonical.len()),
            "RequiredAcceptRejectNotObserved: actual plan differs, no retry/relabel"
        );
        let canonical = a.canonical.iter().map(|d| d.token).collect::<Vec<_>>();
        ids(
            a.raw.as_mut().context("raw store")?,
            &a.cfg,
            a.lane,
            Stage::SpecCanonicalIds,
            &canonical,
        )?;
        let values = a
            .pending_logits
            .take()
            .context("raw presemantic verifier copy missing")?;
        for row in 0..2 {
            logits(
                a.committed.as_mut().context("committed store")?,
                &a.cfg,
                a.lane,
                129 + row,
                &values,
                row,
            )?;
        }
        cache(a, kv, 131)
    })
}
pub(super) fn observe_k3_return(token: u32) -> Result<()> {
    update(|a| {
        anyhow::ensure!(a.selected[a.lane].len() == 2, "actual K3 return order");
        selected(a, 129, 0, token, false)
    })
}
pub(super) fn observe_pending(token: u32) -> Result<()> {
    update(|a| {
        let index = a.selected[a.lane].len();
        anyhow::ensure!(index == 3, "corrected pending output order");
        selected(a, 130, 1, token, false)
    })
}
#[path = "q1_speculation/controls.rs"]
mod controls;
#[path = "q1_speculation/runner.rs"]
mod runner;

#[path = "q1_speculation/trajectory.rs"]
mod trajectory;

fn accepted_rejected_transition(
    mode: Mode,
    position: i32,
    cursor: usize,
    p: &Qwen35VerifiedBlockPlan,
    canonical: usize,
) -> bool {
    mode == Mode::ForcedProposal
        && position == 129
        && p.matched_drafts == 1
        && p.rejected_drafts == 1
        && p.valid_input_tokens == 2
        && p.carry_hidden_row == 1
        && !p.terminal_after_pending
        && cursor == 131
        && canonical == 2
}

#[path = "q1_speculation/first_fork.rs"]
pub(super) mod first_fork;
#[path = "q1_speculation/extent_export.rs"]
mod extent_export;
