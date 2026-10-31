//! Gemma-only full-input checkpoint (ADR-063). A native Gemma tool-result
//! input renders identically with and without the generation cue, so the
//! completed input P is the next turn's exact prefix. The slot captures its
//! Hybrid sliding rings at P (no extra forward/sample) and the next turn
//! restores them and prefills only the appended suffix. Never relax the
//! strict cue boundary for any other family or request shape.
use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};

use crate::debug::investigation_env::gemma_full_input as activation;
use crate::inference::models::gemma4::kv_cache::{anchor_budget, MultiSeqHybridKvBuffers};

#[derive(Clone, Debug)]
pub(crate) struct Checkpoint {
    len: usize,
    tokens_sha256: [u8; 32],
}
fn digest(tokens: &[u32]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    hash.finalize().into()
}
/// `enabled` comes from [`enabled_for_engine`]; the tag binds the exact
/// nonempty text/tool-result input whose cue-less render equals the prompt.
pub(crate) fn request_tag(
    enabled: bool,
    last_is_tool: bool,
    prompt: &[u32],
    stable: &[u32],
) -> Option<Checkpoint> {
    (enabled && last_is_tool && !prompt.is_empty() && prompt == stable).then(|| Checkpoint {
        len: prompt.len(),
        tokens_sha256: digest(prompt),
    })
}
impl Checkpoint {
    pub(crate) fn validate(&self, prompt: &[u32]) -> Result<()> {
        anyhow::ensure!(
            self.len == prompt.len() && self.tokens_sha256 == digest(prompt),
            "Gemma full-input checkpoint token identity changed"
        );
        Ok(())
    }
}

/// Static substrate: Gemma, SlotAware 1–4, Hybrid KV, TQ8 codebook.
fn substrate_supported(gemma: bool, slots: Option<u32>, hybrid: bool, bits: u32) -> bool {
    gemma && matches!(slots, Some(1..=4)) && hybrid && bits == 8
}
fn env_substrate_supported(gemma: bool, slots: Option<u32>) -> bool {
    let env = &crate::debug::INVESTIGATION_ENV;
    substrate_supported(gemma, slots, env.hybrid_kv, env.tq_codebook_bits)
}
fn slot_count(mode: super::EngineMode) -> Option<u32> {
    match mode {
        super::EngineMode::SlotAware { max_slots } => Some(max_slots),
        _ => None,
    }
}

/// Spawn-time refusal of invalid or unsatisfiable explicit intent, before
/// any provisioning. Default mode never refuses.
pub(super) fn validate_spawn(gemma: bool, mode: super::EngineMode) -> Result<(), String> {
    let intent = activation::mode();
    intent.validate()?;
    intent.decide(env_substrate_supported(gemma, slot_count(mode)))?;
    Ok(())
}

/// HTTP side: tag requests only for an engine whose worker can activate.
/// The worker additionally requires the provisioned geometry to fit the
/// four-slot anchor quota; if it does not, [`effective`] ignores the tag.
pub(crate) fn enabled_for_engine(gemma: bool, mode: super::EngineMode) -> bool {
    activation::mode()
        .decide(env_substrate_supported(gemma, slot_count(mode)))
        .unwrap_or(false)
}

/// Provisioned decision shared by the spawn arm (refusal) and the worker
/// thread (activation). `max_prompt_tokens` bounds the token ledger. The kill
/// switch is decided before any planning, and in default mode a plan error
/// means "unsupported", so only an explicit `=1` can refuse here.
pub(super) fn decide_for_scaffold(
    slots: u32,
    scaffold: Option<&[MultiSeqHybridKvBuffers]>,
    max_prompt_tokens: usize,
) -> Result<(bool, Option<anchor_budget::QuotaPlan>)> {
    let intent = activation::mode();
    if intent == activation::Mode::Off {
        return Ok((false, None));
    }
    let plan = match scaffold
        .map(|scaffold| anchor_budget::quota_plan(scaffold, max_prompt_tokens))
        .transpose()
    {
        Ok(plan) => plan,
        Err(error) if intent == activation::Mode::Default => {
            tracing::warn!(error = %error, "Gemma full-input checkpoint quota plan failed; staying off");
            None
        }
        Err(error) => return Err(error),
    };
    let supported =
        env_substrate_supported(true, Some(slots)) && plan.is_some_and(|plan| plan.fits());
    let active = intent.decide(supported).map_err(|e| anyhow!(e))?;
    Ok((active, plan))
}

/// Install the worker decision on the calling (worker) thread.
pub(super) fn activate_worker(
    slots: u32,
    scaffold: Option<&[MultiSeqHybridKvBuffers]>,
    max_prompt_tokens: usize,
) -> Result<bool> {
    let (active, plan) = decide_for_scaffold(slots, scaffold, max_prompt_tokens)?;
    activation::set_worker_active(active);
    tracing::info!(
        active,
        mode = ?activation::mode(),
        slots,
        per_slot_requested_bytes = plan.map(|plan| plan.per_slot_bytes),
        worst_case_requested_bytes = plan.and_then(|plan| plan.total_bytes),
        quota_bytes = anchor_budget::LIMIT,
        "Gemma full-input checkpoint worker decision"
    );
    Ok(active)
}

/// Worker side: a tag only takes effect on an activated worker thread.
pub(super) fn effective(params: &super::SamplingParams) -> Option<&Checkpoint> {
    params
        .gemma_full_input_checkpoint
        .as_ref()
        .filter(|_| activation::worker_active())
}

pub(super) fn is_quota_refusal(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<anchor_budget::ReservationError>()
        .is_some()
}

/// The Hybrid anchor capture roles that reserve quota on an activated worker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AnchorRole {
    FullInput,
    ExactImage,
    StableBoundary,
    StableBatchBoundary,
    RequestStartRollback,
}

/// Every Hybrid anchor capture on an activated worker reserves quota first.
/// A refusal never fails a request: the capture is skipped (`Ok(None)`), the
/// slot keeps whatever anchor it already had, and the request is served as
/// with the checkpoint off (a later cancellation may then reset cold).
/// Identity, cursor, substrate and allocation failures are returned as-is.
pub(super) fn capture_or_skip<T>(
    slot: crate::serve::multi_seq_kv::SlotId,
    role: AnchorRole,
    capture: Result<T>,
) -> Result<Option<T>> {
    match capture {
        Ok(anchor) => Ok(Some(anchor)),
        Err(error) if is_quota_refusal(&error) => {
            tracing::warn!(
                slot = slot.0,
                ?role,
                error = %error,
                "Gemma anchor capture skipped: CPU anchor quota refused; previous anchor kept"
            );
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// Test witnesses for which resume path a request took (process-wide; the
/// real-artifact tests run one at a time).
#[cfg(test)]
pub(super) static ADMISSION_ANCHOR_RESTORES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(super) static CANCELLATION_ANCHOR_RESTORES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
thread_local! {
    static CAPTURE_COUNTS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}
#[cfg(test)]
pub(super) fn capture_counts() -> (usize, usize) {
    CAPTURE_COUNTS.with(|counts| counts.get())
}

/// Production capture seam: finished input cursor, never a zero-length forward.
pub(super) fn capture(
    guard: &super::Gemma4KvGuard<'_>,
    slot: crate::serve::multi_seq_kv::SlotId,
    prompt: &[u32],
    tag: &Checkpoint,
) -> Result<crate::inference::models::gemma4::kv_cache::GemmaHybridSlotAnchor> {
    #[cfg(test)]
    CAPTURE_COUNTS.with(|counts| {
        let (attempts, successes) = counts.get();
        counts.set((attempts + 1, successes));
    });
    tag.validate(prompt)?;
    super::validate_gemma4_slot_cursor(guard, slot, prompt.len())?;
    let hybrid = guard
        .hybrid
        .as_ref()
        .ok_or_else(|| anyhow!("Gemma full input requires actual Hybrid KV"))?;
    let anchor = crate::inference::models::gemma4::kv_cache::snapshot_gemma_hybrid_slot_anchor(
        hybrid,
        slot,
        guard.model.weights.kv_graft_len() + prompt.len(),
    )?;
    tracing::info!(slot = slot.0, prompt_tokens = prompt.len(),
        input_tokens_sha256 = %hex::encode(tag.tokens_sha256), cpu_payload_bytes = anchor.total_bytes(),
        "Gemma full-input checkpoint captured; publication pending");
    #[cfg(test)]
    CAPTURE_COUNTS.with(|counts| {
        let (attempts, successes) = counts.get();
        counts.set((attempts, successes + 1));
    });
    Ok(anchor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_full_input_identity_requires_exact_family_tool_and_full_render() {
        let p = [1, 2, 3];
        let tag = request_tag(true, true, &p, &p).unwrap();
        assert!(tag.validate(&p).is_ok());
        assert!(tag.validate(&[1, 9, 3]).is_err());
        assert!(tag.validate(&[1, 2]).is_err());
        assert!(request_tag(false, true, &p, &p).is_none());
        assert!(request_tag(true, false, &p, &p).is_none());
        assert!(request_tag(true, true, &p, &[1, 2]).is_none());
        assert!(request_tag(true, true, &[], &[]).is_none());
    }

    #[test]
    fn gemma_full_input_substrate_is_gemma_hybrid_tq8_slot_aware_one_to_four() {
        for n in 1..=4 {
            assert!(substrate_supported(true, Some(n), true, 8));
        }
        for (gemma, slots, hybrid, bits) in [
            (false, Some(4), true, 8),
            (true, None, true, 8),
            (true, Some(0), true, 8),
            (true, Some(5), true, 8),
            (true, Some(4), false, 8),
            (true, Some(4), true, 6),
        ] {
            assert!(!substrate_supported(gemma, slots, hybrid, bits));
        }
        assert_eq!(slot_count(super::super::EngineMode::SerialFifo), None);
        assert_eq!(
            slot_count(super::super::EngineMode::SlotAware { max_slots: 4 }),
            Some(4)
        );
    }

    #[test]
    fn gemma_full_input_tag_is_inert_on_an_inactive_worker_thread() {
        let p = [7, 8, 9];
        let params = super::super::SamplingParams {
            gemma_full_input_checkpoint: request_tag(true, true, &p, &p),
            ..super::super::SamplingParams::default()
        };
        activation::set_worker_active(false);
        assert!(effective(&params).is_none(), "inactive worker ignores tags");
        activation::set_worker_active(true);
        assert!(effective(&params).is_some());
        activation::set_worker_active(false);
    }

    #[test]
    fn gemma_anchor_capture_quota_refusal_skips_every_role_and_keeps_other_errors() {
        use crate::serve::multi_seq_kv::SlotId;
        for role in [
            AnchorRole::FullInput,
            AnchorRole::ExactImage,
            AnchorRole::StableBoundary,
            AnchorRole::StableBatchBoundary,
            AnchorRole::RequestStartRollback,
        ] {
            assert_eq!(capture_or_skip(SlotId(1), role, Ok(7)).unwrap(), Some(7));
            let refused: Result<u8> = Err(anchor_budget::refusal_for_test());
            assert_eq!(capture_or_skip(SlotId(1), role, refused).unwrap(), None);
            let wrapped: Result<u8> =
                Err(anchor_budget::refusal_for_test().context("capture Gemma anchor"));
            assert_eq!(capture_or_skip(SlotId(1), role, wrapped).unwrap(), None);
            let identity: Result<u8> = Err(anyhow!(
                "Gemma full-input checkpoint token identity changed"
            ));
            assert!(
                capture_or_skip(SlotId(1), role, identity).is_err(),
                "{role:?}"
            );
        }
    }
}

#[cfg(test)]
mod routing_controls {
    use super::super::{gemma4_cross_batch_candidate, Request, SamplingParams};
    use super::*;
    #[test]
    fn gemma_full_input_tag_leaves_cold_batch_only_on_an_active_worker() {
        let p = [1, 2, 3];
        let (reply, _) = tokio::sync::oneshot::channel();
        let cold = Request::Generate {
            prompt_tokens: p.to_vec(),
            params: SamplingParams::default(),
            reply,
        };
        assert!(gemma4_cross_batch_candidate(&cold).is_some());
        let (reply, _) = tokio::sync::oneshot::channel();
        let full = Request::Generate {
            prompt_tokens: p.to_vec(),
            params: SamplingParams {
                gemma_full_input_checkpoint: request_tag(true, true, &p, &p),
                ..SamplingParams::default()
            },
            reply,
        };
        activation::set_worker_active(true);
        assert!(gemma4_cross_batch_candidate(&full).is_none());
        // An inactive worker ignores the tag: tool turns keep cold batching,
        // exactly as with HF2Q_GEMMA_FULL_INPUT_CHECKPOINT=0.
        activation::set_worker_active(false);
        assert!(gemma4_cross_batch_candidate(&full).is_some());
    }
}
