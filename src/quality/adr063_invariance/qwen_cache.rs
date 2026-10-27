//! Complete active Qwen cache reads, no snapshots/new Metal roots. Reads are
//! allowed only after the caller's actual checked forward terminal. Native
//! packed codes and norms remain distinct tensors; residency aliases do not
//! add data. This first adapter refuses multi-slot layouts explicitly.
use super::{
    capture,
    receipt::{Coordinate, ExpectedTensor, Scalar, Stage},
    store::Store,
};
use crate::inference::models::qwen35::kv_cache::{FullAttnKvSlot, HybridKvCache, LayerSlot};
use crate::serve::multi_seq_kv::SlotId;
use anyhow::{ensure, Context, Result};
use mlx_native::MlxBuffer;
pub fn after_checked_terminal(
    store: &mut Store,
    cache: &HybridKvCache,
    layer_count: u32,
    agent: &str,
    boundary: usize,
) -> Result<()> {
    ensure!(cache.n_seqs==1,"Q1 cache adapter currently requires independently owned serial caches; no cross-slot guess");
    ensure!(
        cache.sequence_len_for_slot(SlotId(0))? as usize == boundary,
        "actual Qwen visible cache boundary mismatch"
    );
    for layer in 0..layer_count {
        match cache
            .slot_index_for_layer(layer)
            .ok_or_else(|| anyhow::anyhow!("actual Qwen layer mapping missing"))?
        {
            LayerSlot::Full(index) => full(
                store,
                &cache.full_attn[index as usize],
                agent,
                boundary,
                layer,
                boundary,
            )?,
            LayerSlot::Linear(index) => {
                let state = &cache.linear_attn[index as usize];
                for (stage, buffer) in [
                    (Stage::CacheConv, state.conv_bufs_for_slot(SlotId(0)).0),
                    (
                        Stage::CacheRecurrent,
                        state.recurrent_bufs_for_slot(SlotId(0)).0,
                    ),
                ] {
                    capture::after_checked_drain(
                        store,
                        at(agent, boundary, layer, stage, 0),
                        buffer,
                        buffer.shape(),
                        None,
                        "Qwen actual terminal + visible cursor + per-slot current ping-pong getter",
                    )
                    .map_err(anyhow::Error::msg)?;
                }
            }
        }
    }
    if let Some(ref mtp) = cache.mtp_slot {
        // MTP has its own cursor. It is not assumed advanced by target-only
        // ordinary forwards, and an unexecuted MTP capacity tail is not read.
        let valid = mtp.current_len[0] as usize;
        if valid > 0 {
            full(store, mtp, agent, valid, layer_count, boundary)?;
        }
    }
    Ok(())
}
fn full(
    store: &mut Store,
    slot: &FullAttnKvSlot,
    agent: &str,
    valid: usize,
    layer: u32,
    observation_boundary: usize,
) -> Result<()> {
    ensure!(
        slot.current_len.as_slice() == [valid as u32],
        "Qwen cache role cursor mismatch"
    );
    for (stage, buffer) in [
        (Stage::CacheKey, slot.k.as_ref()),
        (Stage::CacheValue, slot.v.as_ref()),
    ] {
        if let Some(buffer) = buffer {
            heads(
                store,
                buffer,
                agent,
                valid,
                layer,
                stage,
                observation_boundary,
            )?;
        }
    }
    if let Some(ref tq) = slot.tq {
        for (stage, buffer) in [
            (Stage::CacheKeyPacked, &tq.k_packed),
            (Stage::CacheValuePacked, &tq.v_packed),
            (Stage::CacheKeyNorm, &tq.k_norms),
            (Stage::CacheValueNorm, &tq.v_norms),
        ] {
            heads(
                store,
                buffer,
                agent,
                valid,
                layer,
                stage,
                observation_boundary,
            )?;
        }
    }
    Ok(())
}
fn heads(
    store: &mut Store,
    buffer: &MlxBuffer,
    agent: &str,
    valid: usize,
    layer: u32,
    stage: Stage,
    observation_boundary: usize,
) -> Result<()> {
    let shape = buffer.shape();
    ensure!(
        shape.len() == 4 && shape[0] == 1 && valid <= shape[2],
        "Qwen actual cache geometry mismatch"
    );
    for span in head_spans(
        shape,
        buffer.dtype().size_of(),
        buffer.data_byte_len(),
        valid,
    )? {
        let view = buffer
            .slice_view(span.offset as u64, span.elements)
            .with_shape(span.shape)?;
        // One COMPLETE active matrix per head. No token row is omitted;
        // the first differing flat element can be decoded with its shape.
        capture::after_checked_drain(store,at(agent,observation_boundary,layer,stage,span.head),&view,view.shape(),None,
            "Qwen actual checked terminal; complete visible head matrix, excludes inactive capacity")
            .map_err(anyhow::Error::msg)?;
    }
    Ok(())
}
fn at(agent: &str, boundary: usize, layer: u32, stage: Stage, segment: u32) -> Coordinate {
    Coordinate {
        logical_step: boundary as u64,
        layer,
        stage,
        agent: agent.into(),
        logical_row_start: 0,
        segment,
    }
}

/// Shared by actual buffer capture and its model-free packet manifest.
#[derive(Debug)]
pub(crate) struct HeadSpan {
    pub head: u32,
    pub offset: usize,
    pub elements: usize,
    pub shape: Vec<usize>,
}
pub(crate) fn head_spans(
    shape: &[usize],
    width: usize,
    data_bytes: usize,
    valid: usize,
) -> Result<Vec<HeadSpan>> {
    ensure!(
        shape.len() == 4
            && shape[0] == 1
            && shape[1] > 0
            && shape[3] > 0
            && valid > 0
            && valid <= shape[2],
        "Qwen head capture geometry"
    );
    let bytes = shape
        .iter()
        .try_fold(width, |n, &d| n.checked_mul(d))
        .context("Qwen actual root shape overflow")?;
    ensure!(
        bytes == data_bytes,
        "Qwen root logical extent differs from descriptor"
    );
    let stride = shape[2]
        .checked_mul(shape[3])
        .context("Qwen head stride overflow")?;
    let elements = valid
        .checked_mul(shape[3])
        .context("Qwen head extent overflow")?;
    (0..shape[1])
        .map(|head| {
            Ok(HeadSpan {
                head: u32::try_from(head)?,
                offset: head
                    .checked_mul(stride)
                    .and_then(|n| n.checked_mul(width))
                    .context("Qwen head offset overflow")?,
                elements,
                shape: vec![valid, shape[3]],
            })
        })
        .collect()
}
/// Source-bound selected TQ manifest, generated from actual allocator plans.
/// MTP valid length is explicit; it never inherits target validity by guess.
pub(crate) fn manifest(
    cfg: &crate::inference::models::qwen35::Qwen35Config,
    capacity: u32,
    agent: &str,
    boundary: usize,
    mtp_valid: usize,
) -> Result<Vec<ExpectedTensor>> {
    use crate::inference::models::qwen35::kv_cache::q1_capture_layout;
    ensure!(
        boundary > 0 && boundary <= capacity as usize && mtp_valid <= boundary,
        "Qwen manifest boundary"
    );
    let mut expected = Vec::new();
    for layout in q1_capture_layout::layouts(cfg, capacity)? {
        match layout.slot {
            LayerSlot::Linear(_) => {
                ensure!(
                    layout.roots.len() == 4,
                    "Qwen actual linear descriptor count"
                );
                for (index, stage) in [(0, Stage::CacheConv), (2, Stage::CacheRecurrent)] {
                    let (dtype, shape) = &layout.roots[index];
                    ensure!(*dtype == mlx_native::DType::F32, "Qwen state scalar");
                    expected.push(ExpectedTensor {
                        at: at(agent, boundary, layout.layer, stage, 0),
                        scalar: Scalar::F32,
                        shape: shape.clone(),
                    });
                }
            }
            LayerSlot::Full(_) => {
                let valid = if layout.mtp { mtp_valid } else { boundary };
                if valid == 0 {
                    continue;
                }
                ensure!(layout.roots.len() == 4, "Qwen selected TQ descriptor count");
                for ((dtype, shape), stage) in layout.roots.iter().zip([
                    Stage::CacheKeyPacked,
                    Stage::CacheKeyNorm,
                    Stage::CacheValuePacked,
                    Stage::CacheValueNorm,
                ]) {
                    let scalar = match dtype {
                        mlx_native::DType::U8 => Scalar::U8,
                        mlx_native::DType::F32 => Scalar::F32,
                        _ => anyhow::bail!("Qwen TQ scalar mismatch"),
                    };
                    let bytes = shape
                        .iter()
                        .try_fold(dtype.size_of(), |n, &d| n.checked_mul(d))
                        .context("Qwen descriptor bytes overflow")?;
                    for span in head_spans(shape, dtype.size_of(), bytes, valid)? {
                        expected.push(ExpectedTensor {
                            at: at(agent, boundary, layout.layer, stage, span.head),
                            scalar,
                            shape: span.shape,
                        });
                    }
                }
            }
        }
    }
    Ok(expected)
}

#[cfg(test)]
mod quota_controls {
    use super::*;
    use crate::inference::models::qwen35::{
        default_layer_types, Qwen35Config, Qwen35MoeConfig, Qwen35Variant,
    };
    pub(super) fn header_config(q38: bool) -> Qwen35Config {
        let (serialized_blocks, nextn, inner, kv, hidden, heads) = if q38 {
            (65, 1, 6144, 4, 5120, 24)
        } else {
            (41, 1, 4096, 2, 2048, 16)
        };
        let base = serialized_blocks - nextn;
        Qwen35Config {
            variant: if q38 {
                Qwen35Variant::Dense
            } else {
                Qwen35Variant::Moe
            },
            hidden_size: hidden,
            num_hidden_layers: base,
            num_attention_heads: heads,
            num_key_value_heads: kv,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_num_value_heads: inner / 128,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            full_attention_interval: 4,
            layer_types: default_layer_types(base, 4),
            partial_rotary_factor: 0.25,
            rope_theta: 1e7,
            rotary_dim: 64,
            mrope_section: [11, 11, 10, 0],
            mrope_interleaved: true,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 262144,
            vocab_size: 248320,
            attn_output_gate: true,
            mtp_num_hidden_layers: nextn,
            mtp_use_dedicated_embeddings: false,
            intermediate_size: if q38 { Some(17408) } else { None },
            moe: if q38 {
                None
            } else {
                Some(Qwen35MoeConfig {
                    moe_intermediate_size: 512,
                    num_experts: 256,
                    num_experts_per_tok: 8,
                    shared_expert_intermediate_size: 512,
                })
            },
        }
    }

    #[test]
    fn q1_full_active_cache_manifest_payload_and_packet_fit() {
        for (q38, cache_bytes, cache_records, total_bytes) in [
            (true, 1_293_617_152usize, 2944usize, 1_865_746_432usize),
            (false, 539_356_160usize, 1184usize, 1_111_485_440usize),
        ] {
            let cfg = header_config(q38);
            let mut records = Vec::new();
            for boundary in [128, 144] {
                for lane in 0..4 {
                    records.extend(
                        manifest(&cfg, 262144, &format!("agent-{lane}"), boundary, boundary)
                            .unwrap(),
                    );
                }
            }
            assert_eq!(records.len(), cache_records);
            let bytes: usize = records
                .iter()
                .map(|r| r.shape.iter().product::<usize>() * r.scalar.bytes())
                .sum();
            assert_eq!(bytes, cache_bytes);
            assert_eq!(bytes + 576 * 248320 * 4, total_bytes);
            assert!(total_bytes > 1024 * 1024 * 1024 && total_bytes < 2 * 1024 * 1024 * 1024);
            // The ACTUAL expected-cache packet field, not a bytes-per-record guess.
            let encoded = serde_json::to_vec(&records).unwrap();
            assert!(
                encoded.len() + 64 * 1024 < 1024 * 1024,
                "manifest plus bounded non-cache packet fields must fit reader"
            );
            let inactive = manifest(&cfg, 262144, "agent-0", 128, 0).unwrap();
            assert!(inactive.iter().all(|r| r.at.layer < cfg.num_hidden_layers));
            assert!(manifest(&cfg, 262144, "agent-0", 128, 129).is_err());
        }
    }
    #[test]
    fn q1_head_extent_uses_full_active_rows_and_byte_offsets() {
        let spans = head_spans(&[1, 4, 262144, 256], 1, 4 * 262144 * 256, 144).unwrap();
        assert_eq!(spans.len(), 4);
        assert_eq!(spans[3].offset, 3 * 262144 * 256);
        assert_eq!(spans[3].shape, [144, 256]);
        assert_eq!(
            spans.iter().map(|s| s.elements).sum::<usize>(),
            4 * 144 * 256
        );
        assert!(head_spans(&[1, 4, 10, 256], 1, 4 * 10 * 256 - 1, 10).is_err());
        assert!(head_spans(&[1, 4, 10, 256], 1, 4 * 10 * 256, 11).is_err());
    }
}

#[cfg(test)]
pub(crate) fn selected_header_fixture(q38: bool) -> crate::inference::models::qwen35::Qwen35Config {
    quota_controls::header_config(q38)
}
