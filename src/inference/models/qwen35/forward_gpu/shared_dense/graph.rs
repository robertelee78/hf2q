use super::*;
fn map(maps: &mut Vec<OperatorMap>, layer: Option<usize>, operator: &'static str, rows: &[Row]) {
    maps.push(OperatorMap {
        layer,
        operator,
        rows: rows.to_vec(),
        physical_rows: rows.len(),
    });
}
/// Uses existing production adapters; attention includes singleton QKV/output.
/// The physical shared claim is confined to RMS/DenseQ and final full-row head.
pub(super) fn encode(
    model: &Qwen35Model,
    c: &mut ForwardGpuCache,
    cache: &mut HybridKvCache,
    rows: &[Row],
    roots: &mut Vec<MlxBuffer>,
) -> Result<Output> {
    let _actual_input = native_rows(rows);
    let cfg = &model.cfg;
    let h = cfg.hidden_size as usize;
    let n = rows.len();
    let dev = &c.device;
    let reg = &mut c.registry;
    let tokens = rows.iter().map(|r| r.token).collect::<Vec<_>>();
    let mut hidden = embed_tokens_gpu(
        &tokens,
        &model.token_embd,
        model.token_embd_native.as_ref(),
        cfg.vocab_size,
        cfg.hidden_size,
        dev,
        reg,
    )?;
    roots.push(hidden.clone());
    let mut maps = Vec::new();
    map(
        &mut maps,
        None,
        "packed_embedding_gather_not_weight_matmul",
        rows,
    );
    for (layer, weights) in c.layer_weights.iter().enumerate() {
        let attention = dev.alloc_buffer(n * h * 4, DType::F32, vec![n, h])?;
        roots.push(attention.clone());
        // Each helper receives one physical row and its actual independent
        // cursor/cache slice. Never pass these rows as a temporal sequence.
        for (i, row) in rows.iter().enumerate() {
            let row_root_start=roots.len();
            { // All row-local views/encoders die before the legacy reset.
            let _actual_row = native_rows(std::slice::from_ref(row));
            let x = hidden
                .slice_view((i * h * 4) as u64, h)
                .with_shape(vec![1, h])?;
            let mut pos = dev.alloc_buffer(16, DType::I32, vec![4])?;
            pos.as_mut_slice::<i32>()?
                .fill(i32::try_from(row.position)?);
            roots.push(pos.clone());
            let sid = row.handle.slot_id;
            let out = match weights {
                LayerWeightsGpu::FullAttn { attn, .. } => {
                    let rank = match cache.slot_index_for_layer(layer as u32) {
                        Some(super::super::super::kv_cache::LayerSlot::Full(v)) => v as usize,
                        _ => anyhow::bail!("full cache layout"),
                    };
                    let shape = FullAttnShape::from_config(cfg);
                    let max = cache.max_seq_len;
                    build_gated_attn_layer(
                        dev,
                        reg,
                        &x,
                        &pos,
                        attn,
                        Some(&mut cache.full_attn[rank]),
                        max,
                        1,
                        shape.hidden_size,
                        shape.n_head,
                        shape.n_kv,
                        shape.head_dim,
                        shape.rotary_dim,
                        shape.rope_theta,
                        shape.mrope_section,
                        shape.rms_norm_eps,
                        None,
                        None,
                        Some(roots),
                        None,
                        sid,
                    )?
                }
                LayerWeightsGpu::LinearAttn { attn, .. } => {
                    let rank = match cache.slot_index_for_layer(layer as u32) {
                        Some(super::super::super::kv_cache::LayerSlot::Linear(v)) => v as usize,
                        _ => anyhow::bail!("linear cache layout"),
                    };
                    let shape = DeltaNetLayerShape::from_config(cfg);
                    let slot = &cache.linear_attn[rank];
                    let (ci, co) = slot.conv_bufs_for_slot(sid);
                    let (ri, ro) = slot.recurrent_bufs_for_slot(sid);
                    let out = build_delta_net_layer(
                        dev,
                        reg,
                        &x,
                        attn,
                        ci,
                        co,
                        ri,
                        ro,
                        1,
                        shape.hidden_size,
                        shape.n_k_heads,
                        shape.n_v_heads,
                        shape.d_k,
                        shape.d_v,
                        shape.conv_kernel,
                        shape.rms_norm_eps,
                        None,
                        None,
                        sid,
                    )?;
                    cache.linear_attn[rank].swap_for_slot(sid);
                    out
                }
            };
            roots.push(out.clone());
            let mut enc = dev.command_encoder().map(crate::inference::native_observation::encoder)?;
            enc.blit_copy_bytes(&out, 0, &attention, (i * h * 4) as u64, (h * 4) as u64)?;
            // Drain before the next singleton helper / packed consumer. This
            // deliberate diagnostic boundary is not a performance assertion.
            enc.commit_and_wait_labeled("s3.singleton_attention_row_copy")?;
            #[cfg(test)]
            fixture_after_checked_attention_copy(layer,i)?;
            map(
                &mut maps,
                Some(layer),
                "singleton_entire_attention_including_QKV_output",
                std::slice::from_ref(row),
            );
            drop(enc);
            drop(out);
            } // x/positions and every helper-local wrapper are out of scope.
            // No future access: dedicated packed attention owns the checked
            // copied row. Only the row's helper-root suffix is released. On an
            // earlier Err this is never reached; caller retains quarantine.
            retire_row_scratch(roots,row_root_start,||{
                super::super::super::decode_pool::checked_reset_for_prefill_chunk().map_err(Into::into)
            })?;
        }
        let norm = match weights {
            LayerWeightsGpu::FullAttn { attn, .. } => &attn.post_attn_norm,
            LayerWeightsGpu::LinearAttn { attn, .. } => &attn.post_attn_norm,
        };
        let input = dev.alloc_buffer(n * h * 4, DType::F32, vec![n, h])?;
        let residual = dev.alloc_buffer(n * h * 4, DType::F32, vec![n, h])?;
        roots.extend([input.clone(), residual.clone()]);
        let mut enc = dev.command_encoder().map(crate::inference::native_observation::encoder)?;
        dispatch_fused_residual_norm_f32(
            &mut enc,
            reg,
            dev.metal_device(),
            &hidden,
            &attention,
            norm,
            &input,
            Some(&residual),
            n as u32,
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        enc.memory_barrier();
        let ffn = match weights {
            LayerWeightsGpu::FullAttn {
                ffn: FfnWeightsGpu::DenseQ(f),
                ..
            }
            | LayerWeightsGpu::LinearAttn {
                ffn: FfnWeightsGpu::DenseQ(f),
                ..
            } => f,
            _ => anyhow::bail!("preflight DenseQ drift"),
        };
        hidden =
            build_dense_ffn_layer_gpu_q_into(&mut enc, dev, reg, &input, ffn, Some(&residual))?;
        roots.push(hidden.clone());
        enc.commit_and_wait_labeled("s3.shared_postattn_RMS_dense_FFN")?;
        drop(enc);
        map(
            &mut maps,
            Some(layer),
            "packed_postattn_RMS_dense_FFN_residual",
            rows,
        );
        // Every preceding helper and packed encoder has healthy completion.
        super::super::super::decode_pool::checked_reset_for_prefill_chunk()?;
    }
    let (all, normed) = apply_output_head_gpu_with_nextn_hidden(
        dev,
        reg,
        &hidden,
        &c.output_head,
        n as u32,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.rms_norm_eps,
    )?;
    ensure!(
        all.len() == n * cfg.vocab_size as usize && all.iter().all(|v| v.is_finite()),
        "full finite logits"
    );
    for r in rows {
        cache.validate_sequence_len_for_slot(r.handle.slot_id, r.position as usize + 1)?;
    }
    map(
        &mut maps,
        None,
        "packed_output_RMS_full_vocabulary_head",
        rows,
    );
    ensure!(normed.dtype()==DType::F32&&normed.shape()==[n,h]&&normed.data_byte_len()==n*h*4,"durable actual post-RMS source extent");
    let nextn_hidden=(0..n).map(|row|super::super::super::spec_decode::nth_hidden_row(&normed,cfg.hidden_size,row as u64)).collect::<Result<Vec<_>>>()?;
    Ok(Output {
        nextn_hidden,
        logits: all
            .chunks_exact(cfg.vocab_size as usize)
            .map(<[f32]>::to_vec)
            .collect(),
        maps,
        whole_body_b4: false,
    })
}
