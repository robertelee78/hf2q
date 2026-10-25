//! Shared actual caller geometry; no GPU allocation or selection work.
use super::super::attention::{
    append_prefill_window_indices, compressed_attention_index_plan, compressed_indices,
    window_indices,
};
use anyhow::{ensure, Context, Result};

pub(crate) struct AttentionLayout {
    pub(crate) index_storage: Vec<i32>,
    #[cfg(test)]
    pub(crate) preparation_numeric_capacity_bytes: usize,
    pub(crate) storage_stride: usize,
    pub(crate) attention_width: usize,
    pub(crate) indexer_output_offset: Option<usize>,
    pub(crate) raw_prefix_len: usize,
    pub(crate) compact_raw_kv_len: usize,
    pub(crate) compact_prefill_kv_len: usize,
    pub(crate) compressed_attention_offset: usize,
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_layout(
    ratio: u32,
    rows: usize,
    start_position: usize,
    graft: usize,
    window_capacity: usize,
    valid_compressed: usize,
    index_top_k: usize,
    prefill: bool,
) -> Result<AttentionLayout> {
    ensure!(
        rows > 0 && window_capacity > 0,
        "DeepSeek-V4 empty attention layout"
    );
    let end = start_position
        .checked_add(rows)
        .context("DeepSeek-V4 attention end overflow")?;
    u32::try_from(end).context("DeepSeek-V4 attention end exceeds u32")?;
    let raw_prefix_len = usize::from(rows > 1 && start_position > 0) * window_capacity;
    let compact_raw_kv_len = raw_prefix_len
        .checked_add(rows)
        .context("DeepSeek-V4 raw span overflow")?;
    let compact_prefill_kv_len = compact_raw_kv_len
        .checked_add(graft)
        .and_then(|v| v.checked_add(valid_compressed))
        .context("DeepSeek-V4 compact span overflow")?;
    let compressed_attention_offset = if rows > 1 {
        compact_raw_kv_len
    } else {
        window_capacity
    };
    let max_width = window_capacity
        .checked_add(graft)
        .and_then(|v| {
            v.checked_add(if ratio == 4 {
                index_top_k
            } else {
                graft.checked_add(valid_compressed)?
            })
        })
        .context("DeepSeek-V4 index width overflow")?;
    rows.checked_mul(max_width)
        .and_then(|v| v.checked_mul(4))
        .context("DeepSeek-V4 index storage overflow")?;
    i32::try_from(
        compressed_attention_offset
            .checked_add(graft)
            .context("DeepSeek-V4 attention offset overflow")?,
    )
    .context("DeepSeek-V4 attention offset exceeds i32")?;
    #[cfg(test)]
    let preparation_numeric_capacity_bytes: usize;
    let (index_storage, storage_stride, attention_width, indexer_output_offset) = if prefill {
        let window = if raw_prefix_len > 0 {
            append_prefill_window_indices(window_capacity, rows, start_position)?
        } else {
            window_indices(window_capacity, rows, start_position)?
        };
        // ADR-059 #308: fabricated graft rows index BETWEEN the
        // window and the compressed series (ratio 128) or the
        // ratio-4 indexer output region (the indexer kernel writes
        // its top-k slots at `indexer_output_offset`, past the
        // graft block, so the graft indices survive it).
        let tail_width = if ratio == 4 {
            index_top_k
        } else {
            graft + valid_compressed
        };
        let raw_index_width = window
            .first()
            .map(Vec::len)
            .context("DeepSeek-V4 append window index row missing")?;
        let width = raw_index_width + graft + tail_width;
        let mut storage = vec![-1_i32; rows * width];
        let compressed = (ratio == 128)
            .then(|| {
                compressed_indices(
                    ratio as usize,
                    rows,
                    start_position,
                    compressed_attention_offset,
                    graft,
                )
            })
            .transpose()?;
        #[cfg(test)]
        {
            preparation_numeric_capacity_bytes = storage.capacity() * 4
                + window.iter().map(|row| row.capacity() * 4).sum::<usize>()
                + compressed
                    .as_ref()
                    .map_or(0, |c| c.iter().map(|row| row.capacity() * 4).sum::<usize>());
        }
        for query in 0..rows {
            let row = &mut storage[query * width..(query + 1) * width];
            row[..raw_index_width].copy_from_slice(&window[query]);
            if ratio == 4 {
                // The ratio-4 tail is the indexer output region
                // (filled by the lightning indexer kernel): the
                // graft block is planted here, ahead of it.
                for slot in 0..graft {
                    row[raw_index_width + slot] = (compressed_attention_offset + slot) as i32;
                }
            }
            // ratio-128: `compressed_indices` already emits the
            // graft block then the shifted real completed groups.
            if let Some(compressed) = compressed.as_ref() {
                row[raw_index_width + graft..].copy_from_slice(&compressed[query]);
            }
        }
        (
            storage,
            width,
            width,
            (ratio == 4).then_some(raw_index_width + graft),
        )
    } else {
        let plan = compressed_attention_index_plan(
            ratio as usize,
            window_capacity,
            index_top_k,
            start_position,
            graft,
        )?;
        let storage_stride = plan.storage.len();
        #[cfg(test)]
        {
            preparation_numeric_capacity_bytes = plan.storage.capacity() * 4;
        }
        (
            plan.storage,
            storage_stride,
            plan.attention_width,
            plan.indexer_output_offset,
        )
    };
    Ok(AttentionLayout {
        #[cfg(test)]
        preparation_numeric_capacity_bytes,
        index_storage,
        storage_stride,
        attention_width,
        indexer_output_offset,
        raw_prefix_len,
        compact_raw_kv_len,
        compact_prefill_kv_len,
        compressed_attention_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn indexer_layout_actual_bias_early_late_graft() {
        for (start, valid, bias) in [(0, 64, 257), (3843, 1025, 385)] {
            for graft in [0, 3] {
                let p = prepare_layout(4, 257, start, graft, 128, valid, 512, true).unwrap();
                assert_eq!(p.compressed_attention_offset, bias);
                assert_eq!(
                    (p.storage_stride, p.attention_width, p.indexer_output_offset),
                    (640 + graft, 640 + graft, Some(128 + graft))
                );
                for row in p.index_storage.chunks_exact(p.storage_stride) {
                    for g in 0..graft {
                        assert_eq!(row[128 + g], (bias + g) as i32);
                    }
                    assert!(row[128 + graft..].iter().all(|x| *x == -1));
                }
            }
        }
    }
    #[test]
    fn indexer_layout_ratio128_graft_and_padding_preserved() {
        let p = prepare_layout(128, 257, 0, 3, 128, 2, 512, true).unwrap();
        assert_eq!(
            (p.storage_stride, p.attention_width, p.indexer_output_offset),
            (136, 136, None)
        );
        // Preserve the original independent graft prefix padding PLUS the
        // compressed_indices-produced graft block, rather than normalize it.
        for (query, row) in p.index_storage.chunks_exact(136).enumerate() {
            assert_eq!(&row[128..131], &[-1, -1, -1]);
            let valid = (query + 1) / 128;
            assert_eq!(&row[131..134], &[257, 258, 259]);
            assert_eq!(row[134], if valid > 0 { 260 } else { -1 });
            assert_eq!(row[135], if valid > 1 { 261 } else { -1 });
        }
    }
    #[test]
    fn indexer_layout_scalar_preserves_authoritative_plan() {
        for ratio in [4, 128] {
            for graft in [0, 3] {
                let actual = prepare_layout(ratio, 1, 3843, graft, 128, 30, 512, false).unwrap();
                let old =
                    compressed_attention_index_plan(ratio as usize, 128, 512, 3843, graft).unwrap();
                assert_eq!(actual.index_storage, old.storage);
                assert_eq!(actual.attention_width, old.attention_width);
                assert_eq!(actual.indexer_output_offset, old.indexer_output_offset);
            }
        }
    }
    #[test]
    fn indexer_layout_invalid_preflight_is_checked() {
        for (rows, start, graft) in [
            (0, 0, 0),
            (usize::MAX, 0, 0),
            (1, usize::MAX, 0),
            (1, 0, usize::MAX),
        ] {
            assert!(prepare_layout(4, rows, start, graft, 128, 1, 512, true).is_err());
        }
    }
}
