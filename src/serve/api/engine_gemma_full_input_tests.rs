//! Ignored REAL-GGUF small-input control, not a tiny-weight/model-free claim.
use super::*;
#[test]
#[ignore = "real Gemma GGUF prefill/seed/ring-restore control; requires reviewed model packet"]
fn gemma_full_input_real_prefill_preserves_sample_and_exact_retry() {
    // This test thread plays the SlotAware worker that owns the scaffold.
    crate::debug::investigation_env::gemma_full_input::set_worker_active(true);
    let path = PathBuf::from(std::env::var("HF2Q_GEMMA_FULL_INPUT_TEST_GGUF").unwrap());
    let opts = LoadOptions {
        model_path: path,
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
        glp_path: None,
        glp_alpha: None,
        kv_graft_path: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
    };
    let mut loaded = LoadedModel::load(&opts).unwrap();
    let LoadedModel::Gemma(g) = &mut loaded else {
        panic!("Gemma required")
    };
    assert!(g.config.sliding_window > 0 && g.config.sliding_window <= 2048);
    g.provision_multi_seq_kv_for_slot_aware(2).unwrap();
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    let prompt: Vec<u32> = (0..guard.model.config.sliding_window + 4)
        .map(|i| 10 + (i % 32) as u32)
        .collect();
    let params = SamplingParams {
        max_tokens: 2,
        repetition_penalty: 1.05,
        seed: Some(7),
        ..SamplingParams::default()
    };
    let prefill = |guard: &mut Gemma4KvGuard<'_>, slot, tokens: &[u32]| {
        guard
            .model
            .weights
            .forward_prefill_with_soft_tokens_slot_aware(
                tokens,
                &[],
                2,
                &mut guard.model.ctx,
                slot,
                &mut guard.kv,
                guard.hybrid.as_mut(),
                guard.dense.as_mut(),
                guard.mlx.as_mut(),
            )
            .unwrap()
    };
    let first = prefill(&mut guard, SlotId(0), &prompt);
    clear_gemma4_self_mounts(guard.model);
    install_gemma4_slot_cursor(&mut guard, SlotId(0), prompt.len()).unwrap();
    let before = guard.model.weights.logits_view().unwrap().to_vec();
    assert!(before.iter().all(|x| x.is_finite()));
    let state_before = Gemma4DecodeState::from_first_token(
        guard.model,
        &prompt,
        &params,
        None,
        SlotId(0),
        first,
        Some(&before),
        Duration::ZERO,
        0,
    )
    .unwrap();
    let tag = gemma_full_input::request_tag(true, true, &prompt, &prompt).unwrap();
    let anchor = gemma_full_input::capture(&guard, SlotId(0), &prompt, &tag).unwrap();
    let after = guard.model.weights.logits_view().unwrap().to_vec();
    assert_eq!(
        before.len(),
        after.len(),
        "first-logit length changed during capture"
    );
    let first_changed = before
        .iter()
        .zip(&after)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        first_changed.is_none(),
        "capture changed first-logit bits at index={first_changed:?}"
    );
    let state_after = Gemma4DecodeState::from_first_token(
        guard.model,
        &prompt,
        &params,
        None,
        SlotId(0),
        first,
        Some(&after),
        Duration::ZERO,
        0,
    )
    .unwrap();
    assert_eq!(state_before.next_token, state_after.next_token);
    assert_eq!(state_before.generated_tokens, state_after.generated_tokens);
    assert!(
        state_after.generated_tokens.len() <= 1,
        "no double first-token acceptance, including EOS"
    );
    assert_eq!(state_after.valid_tokens, prompt.len());
    assert!(
        gemma_full_input::capture(&guard, SlotId(0), &prompt[..prompt.len() - 1], &tag).is_err()
    );
    // Independent fresh reference uses the SAME last-token decode kernel.
    let last = prompt.len() - 1;
    prefill(&mut guard, SlotId(1), &prompt[..last]);
    clear_gemma4_self_mounts(guard.model);
    install_gemma4_slot_cursor(&mut guard, SlotId(1), last).unwrap();
    let decode = |guard: &mut Gemma4KvGuard<'_>, slot, token, position| {
        guard
            .model
            .weights
            .forward_decode_slot_aware(
                token,
                position,
                &mut guard.model.ctx,
                &mut None,
                slot,
                &mut guard.kv,
                guard.hybrid.as_mut(),
                guard.dense.as_mut(),
                guard.mlx.as_mut(),
            )
            .unwrap()
    };
    let cross_route_token = decode(&mut guard, SlotId(1), prompt[last], last);
    let cross_route_logits = guard.model.weights.logits_view().unwrap().to_vec();
    // This remains a cross-route DIAGNOSTIC, never the restore oracle: its
    // previous rows came from a separate P-1 batched prefill, not our P image.
    // Preserve the failed V2 evidence; do not call the difference roundoff.

    // Before allocating backup payload, account this test-only copy explicitly.
    // Production anchor remains immutable/private; no Clone or budget bypass
    // is introduced into production. At selected geometry this is158105600B.
    let layers = guard.hybrid.as_ref().unwrap();
    let backup_bytes: usize = layers
        .iter()
        .filter(|b| b.is_sliding)
        .map(|b| {
            let n = b.n_seqs as usize;
            b.k.byte_len() / n
                + b.v_packed.byte_len() / n
                + if b.v_norms.byte_len() == 4 {
                    0
                } else {
                    b.v_norms.byte_len() / n
                }
        })
        .sum();
    assert!(
        backup_bytes <= 256 * 1024 * 1024,
        "test-only CPU backup packet exceeded"
    );
    let mut original_sliding = Vec::new();
    for layer in layers.iter().filter(|b| b.is_sliding) {
        assert!(
            layer.bf16_xlen_k.is_none() && layer.bf16_xlen_v.is_none(),
            "fixed default packet only"
        );
        let mut copies = Vec::new();
        for buf in [&layer.k, &layer.v_packed, &layer.v_norms] {
            if buf.byte_len() == 4 {
                copies.push(Vec::new());
                continue;
            }
            let bytes = buf.as_slice::<u8>().unwrap();
            assert_eq!(bytes.len() % layer.n_seqs as usize, 0);
            let len = bytes.len() / layer.n_seqs as usize;
            let mut copy = Vec::new();
            copy.try_reserve_exact(len).unwrap();
            copy.extend_from_slice(&bytes[..len]);
            copies.push(copy);
        }
        original_sliding.push(copies);
    }
    let other_slot_fingerprint = |guard: &Gemma4KvGuard<'_>| {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        for layer in guard.hybrid.as_ref().unwrap() {
            for buf in [&layer.k, &layer.v_packed, &layer.v_norms] {
                if buf.byte_len() == 4 {
                    continue;
                }
                let bytes = buf.as_slice::<u8>().unwrap();
                let len = bytes.len() / layer.n_seqs as usize;
                let shape = buf.shape();
                assert_eq!(shape.len(), 4);
                let heads = shape[1];
                let cap = layer.capacity;
                let row_bytes = len / (heads * cap);
                let valid = if layer.is_sliding { cap } else { prompt.len() };
                assert!(valid <= cap);
                // Full layers are overwrite-allocated: never read unwritten
                // capacity beyond the independently completed other-slot P.
                for head in 0..heads {
                    let begin = len + head * cap * row_bytes;
                    hash.update(&bytes[begin..begin + valid * row_bytes]);
                }
            }
        }
        <[u8; 32]>::from(hash.finalize())
    };
    let other_before = other_slot_fingerprint(&guard);
    // SAME P-prefill cache image, SAME last-input scalar kernel, before any
    // destructive extra decode. Rewind cursor only; the CPU anchor is untouched.
    install_gemma4_slot_cursor(&mut guard, SlotId(0), last).unwrap();
    let reference = decode(&mut guard, SlotId(0), prompt[last], last);
    let reference_logits = guard.model.weights.logits_view().unwrap().to_vec();
    assert!(reference_logits
        .iter()
        .chain(&cross_route_logits)
        .all(|v| v.is_finite()));
    let cross_route_max_abs = reference_logits
        .iter()
        .zip(&cross_route_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("GEMMA_CROSS_ROUTE_DIAGNOSTIC accepted_as_restore_oracle=false cold_p_minus_one_token={cross_route_token} same_image_retry_token={reference} max_abs={cross_route_max_abs}");
    decode(&mut guard, SlotId(0), first, prompt.len());
    decode(&mut guard, SlotId(0), first, prompt.len() + 1);
    crate::inference::models::gemma4::kv_cache::restore_gemma_hybrid_slot_anchor(
        guard.hybrid.as_mut().unwrap(),
        SlotId(0),
        &anchor,
        last,
    )
    .unwrap();
    // Byte-for-byte selected sliding payload, before the replay can overwrite
    // it. This stronger storage oracle is independent of floating arithmetic.
    for (layer, copies) in guard
        .hybrid
        .as_ref()
        .unwrap()
        .iter()
        .filter(|b| b.is_sliding)
        .zip(&original_sliding)
    {
        for (buf, saved) in [&layer.k, &layer.v_packed, &layer.v_norms]
            .into_iter()
            .zip(copies)
        {
            if saved.is_empty() {
                continue;
            }
            let bytes = buf.as_slice::<u8>().unwrap();
            assert!(
                bytes[..saved.len()] == saved[..],
                "restored sliding payload differs byte-for-byte"
            );
        }
    }
    assert_eq!(
        other_slot_fingerprint(&guard),
        other_before,
        "baseline/overwrite/restore touched other slot"
    );
    drop(original_sliding);
    install_gemma4_slot_cursor(&mut guard, SlotId(0), last).unwrap();
    let restored = decode(&mut guard, SlotId(0), prompt[last], last);
    assert_eq!(
        reference, restored,
        "same-image exact retry after actual ring overwrite"
    );
    let restored_logits = guard.model.weights.logits_view().unwrap();
    assert!(restored_logits.iter().all(|x| x.is_finite()));
    assert_eq!(
        reference_logits.len(),
        restored_logits.len(),
        "restored vocabulary length changed"
    );
    let mismatch = reference_logits
        .iter()
        .zip(restored_logits)
        .enumerate()
        .find(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(index, (a, b))| (index, a.to_bits(), b.to_bits()));
    assert!(
        mismatch.is_none(),
        "same-image/same-kernel restored logits differ in exact bits: first={mismatch:?}, len={}",
        reference_logits.len()
    );
    // Exercise the ACTUAL resumable transaction completion, not only capture.
    // Rebuild a real prefix, then advance the same state Pending -> Ready.
    prefill(&mut guard, SlotId(1), &prompt[..last]);
    clear_gemma4_self_mounts(guard.model);
    install_gemma4_slot_cursor(&mut guard, SlotId(1), last).unwrap();
    let tagged = SamplingParams {
        gemma_full_input_checkpoint: Some(tag.clone()),
        ..params.clone()
    };
    let counts_before_pending = gemma_full_input::capture_counts();
    let pending = Gemma4PrefillState::new(
        SlotId(1),
        prompt.clone(),
        Vec::new(),
        tagged.clone(),
        0,
        Some(Gemma4PromptAnchor {
            prompt_tokens: vec![99], // distinguish the old rollback owner
            kv: anchor,
            vision_fingerprint: None,
        }),
        None,
    )
    .unwrap();
    let partial = pending.progress.plan(last).unwrap();
    assert!(!partial.ready && !partial.capture_anchor);
    let pending = match pending
        .finish_committed_transaction(&mut guard, None, partial, first, None, Duration::ZERO)
        .unwrap_or_else(|e| panic!("Pending failed: {}", e.error))
    {
        Gemma4PrefillAdvance::Pending { state, .. } => state,
        Gemma4PrefillAdvance::Ready { .. } => panic!("Pending published seed"),
    };
    assert_eq!(pending.pending_anchor.as_ref().unwrap().prompt_tokens, [99]);
    assert_eq!(pending.progress.cursor(), last);
    assert_eq!(gemma_full_input::capture_counts(), counts_before_pending);
    // The old owner survives Pending; no full-input checkpoint was constructed.
    let partial_payload = pending.pending_anchor.as_ref().unwrap().kv.total_bytes();
    assert!(partial_payload > 0);
    let final_first = decode(&mut guard, SlotId(1), prompt[last], last);
    install_gemma4_slot_cursor(&mut guard, SlotId(1), prompt.len()).unwrap();
    let ready_logits = guard.model.weights.logits_view().unwrap().to_vec();
    let expected_seed = Gemma4DecodeState::from_first_token(
        guard.model,
        &prompt,
        &tagged,
        None,
        SlotId(1),
        final_first,
        Some(&ready_logits),
        Duration::ZERO,
        0,
    )
    .unwrap();
    let final_plan = pending.progress.plan(1).unwrap();
    let (ready_state, ready_anchor) = match pending
        .finish_committed_transaction(
            &mut guard,
            None,
            final_plan,
            final_first,
            Some(&ready_logits),
            Duration::ZERO,
        )
        .unwrap_or_else(|e| panic!("Ready failed: {}", e.error))
    {
        Gemma4PrefillAdvance::Ready {
            state,
            prompt_anchor,
            ..
        } => (
            state,
            prompt_anchor.expect("Ready must carry checkpoint before seed publication"),
        ),
        Gemma4PrefillAdvance::Pending { .. } => panic!("final transaction stayed Pending"),
    };
    assert_eq!(
        gemma_full_input::capture_counts(),
        (counts_before_pending.0 + 1, counts_before_pending.1 + 1)
    );
    assert_eq!(ready_anchor.prompt_tokens, prompt);
    assert_eq!(ready_anchor.kv.prompt_len(), prompt.len());
    assert_eq!(ready_state.next_token, expected_seed.next_token);
    assert_eq!(ready_state.generated_tokens, expected_seed.generated_tokens);
    assert!(ready_state.generated_tokens.len() <= 1);
    let actual_ready_logits = guard.model.weights.logits_view().unwrap();
    assert_eq!(actual_ready_logits.len(), ready_logits.len());
    assert!(
        actual_ready_logits
            .iter()
            .zip(&ready_logits)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "Ready capture changed first-logit bits"
    );

    // Actual final transaction failure must return the latest rollback owner,
    // not a Ready state that a caller could publish. No fake GPU fault needed:
    // a different input with the same length violates the actual tag identity.
    let mut changed = prompt.clone();
    changed[0] ^= 1;
    let failing = Gemma4PrefillState::new(
        SlotId(1),
        changed,
        Vec::new(),
        tagged,
        last,
        Some(ready_anchor),
        None,
    )
    .unwrap();
    let failure_plan = failing.progress.plan(1).unwrap();
    let failure = match failing.finish_committed_transaction(
        &mut guard,
        None,
        failure_plan,
        final_first,
        Some(&ready_logits),
        Duration::ZERO,
    ) {
        Err(failure) => failure,
        Ok(_) => panic!("capture failure published an advance/seed"),
    };
    assert_eq!(
        gemma_full_input::capture_counts(),
        (counts_before_pending.0 + 2, counts_before_pending.1 + 1)
    );
    assert!(failure
        .error
        .to_string()
        .contains("capture Gemma full-input"));
    let rollback = failure.rollback_anchor.expect("latest rollback owner lost");
    assert_eq!(rollback.prompt_tokens, prompt);
    assert_eq!(rollback.kv.prompt_len(), prompt.len());
    println!("GEMMA_FULL_INPUT_REAL_COMPLETE sample_equal=true logits_unchanged=true exact_retry=true same_image=true sliding_bytes_equal=true actual_pending=true actual_ready=true failure_rollback=true model_free=false");
}

#[test]
fn gemma_full_input_actual_progress_requires_final_ready_before_publication() {
    let mut progress = Gemma4PrefillProgress::new(0, 9, None).unwrap();
    let first = progress.plan(8).unwrap();
    assert!(!first.capture_anchor && !first.ready);
    assert_eq!(
        progress.commit(first).unwrap(),
        Gemma4PrefillCommit::Pending { advanced_tokens: 8 }
    );
    let final_plan = progress.plan(1).unwrap();
    assert!(!final_plan.capture_anchor && final_plan.ready);
    assert_eq!(
        progress.commit(final_plan).unwrap(),
        Gemma4PrefillCommit::Ready { advanced_tokens: 1 }
    );
    assert!(
        progress.commit(final_plan).is_err(),
        "duplicate final commit must fail"
    );
}
