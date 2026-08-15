use super::*;

pub(super) const BYTE_EQUIV_E2E_ENV_GATE: &str = "HF2Q_BYTE_EQUIV_E2E";
pub(super) const BYTE_EQUIV_E2E_GGUF_ENV: &str = "HF2Q_BYTE_EQUIV_E2E_GGUF";

pub(super) fn skip_unless_gated(test_name: &str) -> bool {
    if std::env::var(BYTE_EQUIV_E2E_ENV_GATE).as_deref() == Ok("1") {
        return false;
    }
    eprintln!(
        "[skip] {test_name} — set {BYTE_EQUIV_E2E_ENV_GATE}=1 and \
         {BYTE_EQUIV_E2E_GGUF_ENV}=<path/to/gemma4.gguf> to run the real-model test"
    );
    true
}

pub(super) fn positive_test_env_usize(name: &str, default: usize, maximum: usize) -> usize {
    let value = match std::env::var(name) {
        Ok(raw) => raw
            .parse::<usize>()
            .unwrap_or_else(|error| panic!("{name} must be an integer: {error}")),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("{name} is not valid Unicode: {error}"),
    };
    assert!(
        (1..=maximum).contains(&value),
        "{name} must be in 1..={maximum}, got {value}"
    );
    value
}

pub(super) fn gemma4_test_decode_rows(
    g: &mut GemmaLoadedModel,
    kv: &mut [crate::inference::models::gemma4::kv_cache::MultiSeqHbKvBuffers],
    hybrid: Option<&mut [crate::inference::models::gemma4::kv_cache::MultiSeqHybridKvBuffers]>,
    feed_tokens: &mut [u32],
    positions: &mut [usize],
    width: usize,
    context: &str,
) {
    assert!(width > 0 && width <= feed_tokens.len());
    assert!(width <= positions.len());
    for layer in kv.iter_mut() {
        for (active_idx, &position) in positions[..width].iter().enumerate() {
            layer.seq_lens[active_idx] = position as u32;
        }
    }
    let mut regime = match hybrid {
        Some(hybrid) => {
            for layer in hybrid.iter_mut() {
                for (active_idx, &position) in positions[..width].iter().enumerate() {
                    layer.seq_lens[active_idx] = position as u32;
                }
            }
            crate::inference::models::gemma4::batched_body::BatchedKvRegime::Hybrid(hybrid)
        }
        None => crate::inference::models::gemma4::batched_body::BatchedKvRegime::FullTq(kv),
    };
    let slot_ids: Vec<SlotId> = (0..width).map(|index| SlotId(index as u32)).collect();
    let mut fused_head = None;
    let hidden = g
        .weights
        .forward_decode_body_batched(
            &feed_tokens[..width],
            &slot_ids,
            &positions[..width],
            &mut regime,
            &mut fused_head,
            &mut g.ctx,
        )
        .unwrap_or_else(|error| panic!("batched decode body failed ({context}): {error:#}"));
    let head = match fused_head {
        Some(head) => head,
        None => g
            .weights
            .lm_head_batched(&hidden, width, &mut g.ctx)
            .unwrap_or_else(|error| panic!("batched decode head failed ({context}): {error:#}")),
    };
    assert!(
        head.logits.iter().all(|value| value.is_finite()),
        "batched decode produced non-finite logits ({context})"
    );
    let vocab_size = g.weights.vocab_size;
    let hidden_size = g.weights.hidden_size;
    for active_idx in 0..width {
        let logits = &head.logits[active_idx * vocab_size..(active_idx + 1) * vocab_size];
        let normed = &head.normed[active_idx * hidden_size..(active_idx + 1) * hidden_size];
        let (top1_idx, top1_val) = logits.iter().copied().enumerate().fold(
            (0u32, f32::NEG_INFINITY),
            |best, (index, value)| {
                if value > best.1 {
                    (index as u32, value)
                } else {
                    best
                }
            },
        );
        feed_tokens[active_idx] = g
            .weights
            .finalize_token_from_logits(logits, normed, top1_idx, top1_val)
            .unwrap_or_else(|error| panic!("decode finalize failed ({context}): {error:#}"));
        positions[active_idx] += 1;
    }
}
