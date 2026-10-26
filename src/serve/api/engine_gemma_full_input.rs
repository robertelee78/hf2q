//! Experimental Gemma-only input identity. Never relax the strict cue boundary.
use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};
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
pub(crate) fn request_tag(
    gemma: bool,
    last_is_tool: bool,
    prompt: &[u32],
    stable: &[u32],
) -> Result<Option<Checkpoint>> {
    if !crate::debug::investigation_env::gemma_full_input::selected().map_err(|e| anyhow!(e))? {
        return Ok(None);
    }
    Ok(plan(gemma, last_is_tool, prompt, stable))
}
fn plan(gemma: bool, last_is_tool: bool, prompt: &[u32], stable: &[u32]) -> Option<Checkpoint> {
    (gemma && last_is_tool && !prompt.is_empty() && prompt == stable).then(|| Checkpoint {
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
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gemma_full_input_identity_requires_exact_family_tool_and_full_render() {
        let p = [1, 2, 3];
        let tag = plan(true, true, &p, &p).unwrap();
        assert!(tag.validate(&p).is_ok());
        assert!(tag.validate(&[1, 9, 3]).is_err());
        assert!(tag.validate(&[1, 2]).is_err());
        assert!(plan(false, true, &p, &p).is_none());
        assert!(plan(true, false, &p, &p).is_none());
        assert!(plan(true, true, &p, &[1, 2]).is_none());
        assert!(plan(true, true, &[], &[]).is_none());
    }
}

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

pub(super) fn validate_startup(
    gemma: bool,
    slots: Option<u32>,
    hybrid: bool,
    bits: u32,
) -> Result<()> {
    anyhow::ensure!(
        gemma && matches!(slots, Some(1..=4)) && hybrid && bits == 8,
        "requires Gemma Hybrid/TQ8 SlotAware1–4; no fallback"
    );
    Ok(())
}

#[cfg(test)]
mod routing_controls {
    use super::super::{gemma4_cross_batch_candidate, Request, SamplingParams};
    use super::*;
    #[test]
    fn gemma_full_input_tag_cannot_enter_cold_or_empty_cue_batch() {
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
                gemma_full_input_checkpoint: plan(true, true, &p, &p),
                ..SamplingParams::default()
            },
            reply,
        };
        assert!(gemma4_cross_batch_candidate(&full).is_none());
    }
    #[test]
    fn gemma_full_input_startup_refuses_unsupported_substrate_without_fallback() {
        for n in 1..=4 {
            validate_startup(true, Some(n), true, 8).unwrap();
        }
        for (gemma, slots, hybrid, bits) in [
            (false, Some(4), true, 8),
            (true, None, true, 8),
            (true, Some(0), true, 8),
            (true, Some(5), true, 8),
            (true, Some(4), false, 8),
            (true, Some(4), true, 6),
        ] {
            assert!(validate_startup(gemma, slots, hybrid, bits).is_err());
        }
    }
}
