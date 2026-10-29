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
    #[cfg(test)]let three=super::target_timing_hooks::three_la_active();
    #[cfg(not(test))]let three=false;
    if three {require_three_la_pattern(&c.layer_weights)?;}
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
    let mut block_root_start=0;
    let mut block_first=0;
    for (layer, weights) in c.layer_weights.iter().enumerate() {
        if !three||layer%4==0||layer%4==3 {block_root_start=roots.len();block_first=layer;}
        let attention = dev.alloc_buffer(n * h * 4, DType::F32, vec![n, h])?;
        roots.push(attention.clone());

        retained::begin_layer(layer,matches!(weights,LayerWeightsGpu::LinearAttn{..}))?;
        // Each helper receives one physical row and its actual independent
        // cursor/cache slice. Never pass these rows as a temporal sequence.
        for (i, row) in rows.iter().enumerate() {
            retained::begin_row(i,row.handle.slot_id.0)?;
            #[cfg(test)]super::target_timing_hooks::context(layer,row.handle.slot_id.0)?;
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
            // SAME-QUEUE submit once; keep exact encoder + all issued pool epochs
            // until this layer's existing FFN checked terminal fence.
            retained::submit_copy(enc)?;
            map(
                &mut maps,
                Some(layer),
                "singleton_entire_attention_including_QKV_output",
                std::slice::from_ref(row),
            );
            drop(out);
            } // x/positions and every helper-local wrapper are out of scope.
            // No suffix truncation/reset here: Metal retain alone cannot stop
            // a pool reset from reissuing these CPU-refilled parameter roots.

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
        if retained::intermediate()? {
            // No reset, suffix retirement, or CPU parameter reissue between LA
            // layers. Own the exact intermediate FFN before its single submit.
            retained::submit_ffn(enc)?;
        } else {
            #[cfg(test)]let clock=super::target_timing_hooks::start()?;
            let anchor=enc.commit_and_wait_labeled("s3.shared_postattn_RMS_dense_FFN").map_err(anyhow::Error::from);
            #[cfg(test)]let anchor=anchor.and_then(|()|super::target_timing_hooks::checked("shared-FFN",&enc,clock,Some((layer,None))));
            retained::after_layer_fence(anchor)?;
            drop(enc);
            #[cfg(test)]for checked_layer in block_first..=layer {for (i,_) in rows.iter().enumerate(){fixture_after_checked_attention_copy(checked_layer,i)?;}}
            // Every block CB is healthy; final outputs are dedicated nonpool
            // parents. Keep their local owners through helper suffix retirement.
            retire_row_scratch(roots,block_root_start,||{
                super::super::super::decode_pool::checked_reset_for_prefill_chunk().map_err(Into::into)
            })?;
            roots.extend([input.clone(),residual.clone(),hidden.clone()]);
        }

        map(
            &mut maps,
            Some(layer),
            "packed_postattn_RMS_dense_FFN_residual",
            rows,
        );
        // Layer epoch was already retired/reset exactly once after all checks.
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

fn require_three_la_pattern(weights:&[LayerWeightsGpu])->Result<()> {
    ensure!(weights.len()==64,"three LA requires actual64 layers");
    for (layer,weight) in weights.iter().enumerate(){
        ensure!(matches!(weight,LayerWeightsGpu::LinearAttn{..})==(layer%4<3),"actual three LA / FA pattern drift");
    }Ok(())
}

#[cfg(test)]mod three_la_resource_controls {
    #[test]fn finite_three_la_cold_bucket_bound_includes_all_optional_casts(){
        let buckets:[(u64,u64);12]=[(4,3),(8,30),(16,6),(64,6),(256,24),(8192,18),(32768,30),(65536,12),(262144,9),(524288,12),(67108864,12),(134217728,6)];
        assert_eq!(buckets.iter().map(|x|x.1).sum::<u64>(),168);
        assert_eq!(buckets.iter().map(|x|x.0*x.1).sum::<u64>(),1_621_187_292);
        assert_eq!(3*204_832u64,614_496);
        assert!(1_621_187_292u64<24*1024*1024*1024);
    }
}
