//! Stage every fallible resource before transferring the original bootstrap.
use super::{Deepseek4LoadedModel, Deepseek4Session, INITIAL_CACHE_LENGTH};
use crate::inference::models::deepseek4::{
    cache::{Deepseek4Cache, Deepseek4CachePlan},
    Deepseek4Config,
};
use anyhow::{ensure, Context, Result};

struct BootstrapState {
    capacity: usize,
    position: usize,
    poisoned: bool,
    grafted: bool,
    published: bool,
    empty: [bool; 8],
}
impl BootstrapState {
    fn reusable(&self, initial: usize) -> bool {
        self.capacity == initial
            && self.position == 0
            && !self.poisoned
            && !self.grafted
            && !self.published
            && self.empty.iter().all(|&empty| empty)
    }
}
struct Layout {
    slots: usize,
    reuse: bool,
    initial: Deepseek4CachePlan,
    small: Option<Deepseek4CachePlan>,
}
impl Layout {
    fn new(
        cfg: &Deepseek4Config,
        context: usize,
        slots: u32,
        state: &BootstrapState,
    ) -> Result<Self> {
        ensure!(slots > 0, "DeepSeek-V4 max_slots must be nonzero");
        let initial_capacity = context.min(INITIAL_CACHE_LENGTH);
        let initial = Deepseek4CachePlan::for_context(cfg, initial_capacity)?;
        let reuse = state.reusable(initial_capacity);
        if crate::serve::q5_deepseek::operating_case::is_active(){
            ensure!(slots==4 && reuse,"Q5 reviewed four-cache+C128 branch requires pristine bootstrap; no five-cache fallback");
        }
        let small = reuse
            .then(|| Deepseek4CachePlan::for_context(cfg, 128))
            .transpose()?;
        Ok(Self {
            slots: slots as usize,
            reuse,
            initial,
            small,
        })
    }
    fn allocation(&self, index: usize) -> (&Deepseek4CachePlan, bool) {
        if index == 0 {
            if let Some(small) = &self.small {
                return (small, true);
            }
        }
        (&self.initial, false)
    }
}

/// The production reserve/allocate/validate sequence, also used by fault controls.
fn stage<T>(
    layout: &Layout,
    reserve: impl FnOnce(&mut Vec<T>, usize) -> Result<()>,
    mut allocate: impl FnMut(usize, &Deepseek4CachePlan, bool) -> Result<T>,
    mut validate: impl FnMut(usize, &T, &Deepseek4CachePlan) -> Result<()>,
) -> Result<Vec<T>> {
    let mut sessions = Vec::new();
    reserve(&mut sessions, layout.slots)?;
    ensure!(
        sessions.is_empty() && sessions.capacity() >= layout.slots,
        "invalid staged slot reservation"
    );
    for index in 0..layout.slots {
        let (plan, eager) = layout.allocation(index);
        sessions.push(allocate(index, plan, eager)?);
    }
    for (index, session) in sessions.iter().enumerate() {
        validate(index, session, layout.allocation(index).0)?;
    }
    Ok(sessions)
}

/// Called only with a fully prepared nonempty vector; this commit cannot fail.
fn transfer<C, S>(bootstrap: &mut C, sessions: &mut [S], cache: fn(&mut S) -> &mut C) {
    std::mem::swap(bootstrap, cache(&mut sessions[0]));
}

fn validate_cache(cache: &Deepseek4Cache, plan: &Deepseek4CachePlan) -> Result<()> {
    ensure!(
        cache.capacity() == plan.context_length
            && cache.resident_bytes() == plan.resident_bytes
            && cache.layers().len() == plan.layers.len()
            && cache.position() == 0
            && !cache.is_poisoned()
            && cache.graft_slots() == 0,
        "staged DeepSeek-V4 cache differs from its plan"
    );
    for (actual, expected) in cache.layers().iter().zip(&plan.layers) {
        for (buffer, descriptor) in [
            (Some(&actual.attention_kv), Some(&expected.attention_kv)),
            (Some(&actual.window_kv), Some(&expected.window_kv)),
            (
                actual.compressed_kv.as_ref(),
                expected.compressed_kv.as_ref(),
            ),
            (actual.indexer_kv.as_ref(), expected.indexer_kv.as_ref()),
            (
                actual.main_kv_state.as_ref(),
                expected.main_kv_state.as_ref(),
            ),
            (
                actual.main_score_state.as_ref(),
                expected.main_score_state.as_ref(),
            ),
            (
                actual.indexer_kv_state.as_ref(),
                expected.indexer_kv_state.as_ref(),
            ),
            (
                actual.indexer_score_state.as_ref(),
                expected.indexer_score_state.as_ref(),
            ),
        ] {
            match (buffer, descriptor) {
                (None, None) => (),
                (Some(b), Some(d))
                    if b.dtype() == d.dtype
                        && b.shape() == d.shape.as_slice()
                        && b.data_byte_len() as u64 == d.bytes =>
                {
                    ()
                }
                _ => anyhow::bail!("staged DeepSeek-V4 buffer differs from its descriptor"),
            }
        }
    }
    Ok(())
}

pub(super) fn provision(loaded: &mut Deepseek4LoadedModel, max_slots: u32) -> Result<()> {
    // Keep the existing zero-width error before planning or allocating anything.
    ensure!(max_slots > 0, "DeepSeek-V4 max_slots must be nonzero");
    let state = BootstrapState {
        capacity: loaded.cache.capacity(),
        position: loaded.cache.position(),
        poisoned: loaded.cache.is_poisoned(),
        grafted: loaded.model.kv_graft.is_some() || loaded.cache.graft_slots() != 0,
        published: loaded.slot_sessions.is_some(),
        empty: [
            loaded.committed_tokens.is_empty(),
            loaded.live_logits.is_none(),
            loaded.turn_anchor.is_none(),
            loaded.turn_anchor_tokens.is_empty(),
            loaded.pending_turn_anchor.is_none(),
            loaded.pending_turn_anchor_tokens.is_empty(),
            !loaded.keep_turn_anchor_on_success,
            !loaded.request_anchor_transaction_active,
        ],
    };
    let layout = Layout::new(&loaded.model.cfg, loaded.context_limit(), max_slots, &state)?;
    if layout.reuse {
        validate_cache(&loaded.cache, &layout.initial)?;
    }
    let mut sessions = stage(
        &layout,
        |sessions, count| {
            sessions
                .try_reserve_exact(count)
                .context("reserve DeepSeek-V4 slot metadata")
        },
        |index, plan, eager| {
            if eager {
                let cache = loaded
                    .model
                    .allocate_cache(plan.context_length)
                    .context("allocate small DeepSeek-V4 bootstrap swap cache")?;
                Ok(Deepseek4Session::from_empty_cache(loaded, cache))
            } else {
                Deepseek4Session::new(loaded)
                    .with_context(|| format!("reserve DeepSeek-V4 logical slot {index}"))
            }
        },
        |_, session, plan| {
            // Grafted/nonpristine/repeated paths retain ordinary construction.
            if layout.reuse {
                validate_cache(&session.cache, plan)?;
                ensure!(
                    session.committed_tokens.is_empty()
                        && session.live_logits.is_none()
                        && session.turn_anchor.is_none()
                        && session.turn_anchor_tokens.is_empty()
                        && session.pending_turn_anchor.is_none()
                        && session.pending_turn_anchor_tokens.is_empty()
                        && !session.keep_turn_anchor_on_success
                        && !session.request_anchor_transaction_active,
                    "staged DeepSeek-V4 session is not pristine"
                );
            }
            Ok(())
        },
    )?;
    tracing::info!(
        max_slots,
        logical_context_per_slot = loaded.context_limit(),
        bootstrap_reused = layout.reuse,
        "DeepSeek-V4 full-context agent slots reserved"
    );
    // No fallible work after moving the original cache. Both host bundles are empty.
    if layout.reuse {
        transfer(&mut loaded.cache, &mut sessions, |session| {
            &mut session.cache
        });
    }
    loaded.slot_sessions = Some(sessions);
    Ok(())
}

#[cfg(test)]
mod tests;
