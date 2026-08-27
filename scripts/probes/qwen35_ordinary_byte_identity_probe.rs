use super::*;
use sha2::{Digest, Sha256};

fn update_u64(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn update_buffer(hasher: &mut Sha256, buffer: &MlxBuffer) {
    update_u64(hasher, buffer.shape().len());
    for &dimension in buffer.shape() {
        update_u64(hasher, dimension);
    }
    let bytes = buffer
        .as_slice::<u8>()
        .expect("ordinary identity probe buffer bytes");
    update_u64(hasher, bytes.len());
    hasher.update(bytes);
}

fn update_optional_buffer(hasher: &mut Sha256, buffer: Option<&MlxBuffer>) {
    hasher.update([u8::from(buffer.is_some())]);
    if let Some(buffer) = buffer {
        update_buffer(hasher, buffer);
    }
}

fn update_full_slot(hasher: &mut Sha256, slot: &super::super::kv_cache::FullAttnKvSlot) {
    update_u64(hasher, slot.current_len.len());
    for &cursor in &slot.current_len {
        hasher.update(cursor.to_le_bytes());
    }
    update_optional_buffer(hasher, slot.k.as_ref());
    update_optional_buffer(hasher, slot.v.as_ref());
    hasher.update([u8::from(slot.tq.is_some())]);
    if let Some(tq) = &slot.tq {
        hasher.update(tq.norms_per_pos.to_le_bytes());
        update_buffer(hasher, &tq.k_packed);
        update_buffer(hasher, &tq.k_norms);
        update_buffer(hasher, &tq.v_packed);
        update_buffer(hasher, &tq.v_norms);
    }
}

fn physical_cache_sha256(cache: &HybridKvCache) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"hf2q-qwen35-ordinary-physical-cache-v1\0");
    hasher.update(cache.max_seq_len.to_le_bytes());
    hasher.update(cache.n_seqs.to_le_bytes());
    hasher.update(cache.conv_channels.to_le_bytes());
    hasher.update([u8::from(cache.tq_kv_active)]);
    update_u64(&mut hasher, cache.full_attn.len());
    for slot in &cache.full_attn {
        update_full_slot(&mut hasher, slot);
    }
    hasher.update([u8::from(cache.mtp_slot.is_some())]);
    if let Some(slot) = &cache.mtp_slot {
        update_full_slot(&mut hasher, slot);
    }
    update_u64(&mut hasher, cache.linear_attn.len());
    for slot in &cache.linear_attn {
        update_u64(&mut hasher, slot.pp_flipped.len());
        for &flipped in &slot.pp_flipped {
            hasher.update([u8::from(flipped)]);
        }
        update_buffer(&mut hasher, &slot.conv_state);
        update_buffer(&mut hasher, &slot.conv_state_scratch);
        update_buffer(&mut hasher, &slot.recurrent);
        update_buffer(&mut hasher, &slot.recurrent_scratch);
        update_optional_buffer(&mut hasher, slot.capture_states.as_ref());
        update_optional_buffer(&mut hasher, slot.conv_capture_states.as_ref());
    }
    hex::encode(hasher.finalize())
}

fn f32_sha256(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"hf2q-f32-le-v1\0");
    update_u64(&mut hasher, values.len());
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

fn buffer_f32_sha256(buffer: &MlxBuffer) -> String {
    f32_sha256(
        buffer
            .as_slice::<f32>()
            .expect("ordinary identity probe f32 buffer"),
    )
}

fn text_positions(start: i32, len: usize) -> Vec<i32> {
    let mut flat = vec![0_i32; 4 * len];
    for axis in 0..4 {
        for index in 0..len {
            flat[axis * len + index] = start + index as i32;
        }
    }
    flat
}

fn process_mtp(
    model: &Qwen35Model,
    tokens: &[u32],
    pending_target_hidden: Option<&MlxBuffer>,
    target_hidden: &MlxBuffer,
    positions: &[i32],
    cache: &mut HybridKvCache,
) {
    let shared_embed = model
        .embed_tokens_gpu(tokens)
        .expect("ordinary identity probe shared embedding");
    let mtp = model.mtp.as_ref().expect("ordinary identity probe MTP");
    model
        .with_gpu_cache_mut(|device, registry| {
            mtp.process_target_batch(
                tokens,
                pending_target_hidden,
                target_hidden,
                &shared_embed,
                cache,
                SlotId(0),
                positions,
                device,
                registry,
                &model.cfg,
            )?;
            let mut drain = device.command_encoder()?;
            drain.commit_and_wait_labeled("test.qwen35.ordinary_identity_probe")?;
            Ok(())
        })
        .expect("ordinary identity probe MTP execution");
}

#[test]
#[ignore = "requires HF2Q_ORDINARY_IDENTITY_GGUF and an exclusive Apple-Silicon gate"]
fn ordinary_prefill_continuation_matches_main_byte_for_byte() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let path = std::path::PathBuf::from(
        std::env::var_os("HF2Q_ORDINARY_IDENTITY_GGUF")
            .expect("HF2Q_ORDINARY_IDENTITY_GGUF is required"),
    );
    let output = std::path::PathBuf::from(
        std::env::var_os("HF2Q_ORDINARY_IDENTITY_OUTPUT")
            .expect("HF2Q_ORDINARY_IDENTITY_OUTPUT is required"),
    );
    let gguf = mlx_native::gguf::GgufFile::open(&path).expect("open ordinary identity GGUF");
    let mut progress = crate::serve::header::LoadProgress::new(false, 1, 0);
    let model =
        Qwen35Model::load_from_gguf(&gguf, &mut progress).expect("load ordinary identity model");
    assert!(
        model.mtp.is_some(),
        "ordinary identity artifact requires MTP"
    );
    model
        .ensure_gpu_cache_primed()
        .expect("prime ordinary identity model");
    let mut cache = model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&model.cfg, device, 64, 1, true)
        })
        .expect("ordinary identity cache");

    let seed = [151_643_u32, 9707, 374, 279, 15];
    let prefix: Vec<u32> = (0..33).map(|index| seed[index % seed.len()]).collect();
    let continuation: Vec<u32> = (0..3)
        .map(|index| seed[(index + prefix.len()) % seed.len()])
        .collect();
    let prefix_positions = text_positions(0, prefix.len());
    let continuation_positions = text_positions(prefix.len() as i32, continuation.len());

    let (prefix_logits, prefix_hidden) = model
        .forward_gpu_last_logits_with_hidden(&prefix, &prefix_positions, &mut cache, SlotId(0))
        .expect("ordinary identity prefix target");
    process_mtp(
        &model,
        &prefix,
        None,
        &prefix_hidden,
        &prefix_positions,
        &mut cache,
    );
    let pending = super::super::spec_decode::last_hidden_row(&prefix_hidden, model.cfg.hidden_size)
        .expect("ordinary identity pending target hidden");
    let prefix_cache = physical_cache_sha256(&cache);

    let (continuation_logits, continuation_hidden) = model
        .forward_gpu_last_logits_with_hidden(
            &continuation,
            &continuation_positions,
            &mut cache,
            SlotId(0),
        )
        .expect("ordinary identity continuation target");
    process_mtp(
        &model,
        &continuation,
        Some(&pending),
        &continuation_hidden,
        &continuation_positions,
        &mut cache,
    );
    let continuation_cache = physical_cache_sha256(&cache);

    let receipt = format!(
        "schema\t1\nroute\tordinary-target-plus-mtp\nprefix_tokens\t33\ncontinuation_tokens\t3\nprefix_logits_sha256\t{}\nprefix_hidden_sha256\t{}\nprefix_cache_sha256\t{}\ncontinuation_logits_sha256\t{}\ncontinuation_hidden_sha256\t{}\ncontinuation_cache_sha256\t{}\n",
        f32_sha256(&prefix_logits),
        buffer_f32_sha256(&prefix_hidden),
        prefix_cache,
        f32_sha256(&continuation_logits),
        buffer_f32_sha256(&continuation_hidden),
        continuation_cache,
    );
    std::fs::write(&output, receipt).expect("write ordinary identity probe receipt");
}
