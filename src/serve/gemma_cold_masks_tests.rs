//! Capture-only (recorded session, never committed; no model): every reader
//! of a cold-prefill mask is separated from the mask fill by a barrier, and
//! the fill carries its write range for the recorded-graph reorder pass.
use super::*;
use mlx_native::ops::flash_attn_prefill_blk::alloc_blk_buffer;
use mlx_native::{CapturedNode, GraphExecutor, RecordedBinding};

const LEN: usize = 64;

fn device() -> Option<MlxDevice> {
    match MlxDevice::new() {
        Ok(device) => Some(device),
        Err(_) => {
            eprintln!("skip: no MlxDevice");
            None
        }
    }
}

fn registry() -> KernelRegistry {
    let mut registry = KernelRegistry::new();
    mlx_native::ops::flash_attn_prefill_mask::register(&mut registry);
    mlx_native::ops::flash_attn_prefill_blk::register(&mut registry);
    registry
}

fn blk(device: &MlxDevice, bq: u32, bk: u32) -> (BlkParams, MlxBuffer) {
    let params = BlkParams {
        seq_len_q: LEN as u32,
        seq_len_k: LEN as u32,
        bq,
        bk,
    };
    let buffer = alloc_blk_buffer(device, &params).unwrap();
    (params, buffer)
}

fn binds(node: &CapturedNode, buffer: &MlxBuffer) -> bool {
    let CapturedNode::Dispatch { bindings, .. } = node else {
        return false;
    };
    bindings.iter().any(|(_, binding)| {
        matches!(
            binding,
            RecordedBinding::Buffer { metal_buffer, .. }
                if std::ptr::eq(&**metal_buffer, &**buffer.metal_buffer())
        )
    })
}

fn covers(ranges: &[(usize, usize)], buffer: &MlxBuffer) -> bool {
    let begin = buffer.contents_ptr() as usize;
    let end = begin + buffer.byte_len();
    ranges
        .iter()
        .any(|&(start, stop)| start <= begin && stop >= end)
}

/// The fill of `mask` (its first binding dispatch) and the later dispatches
/// that bind it, each paired with whether a barrier separates it from the
/// fill.
fn readers(nodes: &[CapturedNode], mask: &MlxBuffer) -> (usize, Vec<bool>) {
    let fill = nodes
        .iter()
        .position(|node| binds(node, mask))
        .expect("mask fill dispatch captured");
    let readers = (fill + 1..nodes.len())
        .filter(|&index| binds(&nodes[index], mask))
        .map(|index| {
            nodes[fill + 1..index]
                .iter()
                .any(|node| matches!(node, CapturedNode::Barrier))
        })
        .collect();
    (fill, readers)
}

fn assert_fenced(
    nodes: &[CapturedNode],
    mask: &MlxBuffer,
    fill_reads: &[&MlxBuffer],
    expected_readers: usize,
    label: &str,
) {
    let (fill, fenced) = readers(nodes, mask);
    let CapturedNode::Dispatch { reads, writes, .. } = &nodes[fill] else {
        unreachable!()
    };
    assert!(
        covers(writes, mask),
        "{label}: fill write range not declared"
    );
    for read in fill_reads {
        assert!(covers(reads, read), "{label}: fill read range not declared");
    }
    assert_eq!(fenced.len(), expected_readers, "{label}: reader count");
    for (index, fenced) in fenced.iter().enumerate() {
        assert!(
            fenced,
            "{label}: reader {index} may run concurrently with the mask fill"
        );
    }
}

#[test]
fn cold_square_masks_fence_cast_and_tile_classifier() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Some(device) = device() else { return };
    let mut reg = registry();
    let (sw_params, blk_sw) = blk(&device, 32, 16);
    let (gl_params, blk_gl) = blk(&device, 8, 64);
    let commits_before = mlx_native::commit_count();
    let executor = GraphExecutor::new(device.clone());
    let mut s = executor.begin_recorded().unwrap();
    let sliding = encode_cold_mask(
        &mut s,
        &device,
        &mut reg,
        ColdMaskLayout::Square { len: LEN },
        Some(16),
        true,
        "sliding_mask",
    )
    .unwrap();
    let global = encode_cold_mask(
        &mut s,
        &device,
        &mut reg,
        ColdMaskLayout::Square { len: LEN },
        None,
        true,
        "global_mask",
    )
    .unwrap();
    let sliding_f16 = sliding.f16.as_ref().expect("sliding F16 mask");
    let global_f16 = global.f16.as_ref().expect("global F16 mask");
    assert_eq!(sliding_f16.shape(), &[LEN, LEN]);
    assert_eq!(global_f16.shape(), &[LEN, LEN]);
    encode_mask_tiles(
        &mut s,
        &device,
        &mut reg,
        &sliding.bf16,
        &blk_sw,
        &sw_params,
        "blk_sliding",
    )
    .unwrap();
    encode_mask_tiles(
        &mut s,
        &device,
        &mut reg,
        &global.bf16,
        &blk_gl,
        &gl_params,
        "blk_global",
    )
    .unwrap();
    let nodes = s.encoder_mut().take_capture().unwrap();
    drop(s);
    // Each BF16 mask is read by its F16 cast and its blk classifier.
    assert_fenced(&nodes, &sliding.bf16, &[], 2, "sliding");
    assert_fenced(&nodes, &global.bf16, &[], 2, "global");
    drop(nodes);
    assert_eq!(mlx_native::commit_count(), commits_before);
}

#[test]
fn cold_block_diagonal_masks_fence_tile_classifier() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Some(device) = device() else { return };
    let mut reg = registry();
    // Two sequences of 40 and 24 tokens: the classifier must see the -inf
    // off-diagonal blocks, or the tile-skip path attends across sequences.
    let mut seq_id = device.alloc_buffer(LEN * 4, DType::U32, vec![LEN]).unwrap();
    let mut local_pos = device.alloc_buffer(LEN * 4, DType::U32, vec![LEN]).unwrap();
    for (token, (sid, pos)) in seq_id
        .as_mut_slice::<u32>()
        .unwrap()
        .iter_mut()
        .zip(local_pos.as_mut_slice::<u32>().unwrap().iter_mut())
        .enumerate()
    {
        (*sid, *pos) = if token < 40 {
            (0, token as u32)
        } else {
            (1, token as u32 - 40)
        };
    }
    let (sw_params, blk_sw) = blk(&device, 32, 16);
    let (gl_params, blk_gl) = blk(&device, 8, 64);
    let commits_before = mlx_native::commit_count();
    let executor = GraphExecutor::new(device.clone());
    let mut s = executor.begin_recorded().unwrap();
    let layout = || ColdMaskLayout::BlockDiagonal {
        seq_id: &seq_id,
        local_pos: &local_pos,
        len: LEN,
    };
    let sliding = encode_cold_mask(
        &mut s,
        &device,
        &mut reg,
        layout(),
        Some(16),
        false,
        "sliding_mask",
    )
    .unwrap();
    let global = encode_cold_mask(
        &mut s,
        &device,
        &mut reg,
        layout(),
        None,
        false,
        "global_mask",
    )
    .unwrap();
    assert!(sliding.f16.is_none() && global.f16.is_none());
    encode_mask_tiles(
        &mut s,
        &device,
        &mut reg,
        &sliding.bf16,
        &blk_sw,
        &sw_params,
        "blk_sliding",
    )
    .unwrap();
    encode_mask_tiles(
        &mut s,
        &device,
        &mut reg,
        &global.bf16,
        &blk_gl,
        &gl_params,
        "blk_global",
    )
    .unwrap();
    let nodes = s.encoder_mut().take_capture().unwrap();
    drop(s);
    let fill_reads = [&seq_id, &local_pos];
    assert_fenced(&nodes, &sliding.bf16, &fill_reads, 1, "multi-seq sliding");
    assert_fenced(&nodes, &global.bf16, &fill_reads, 1, "multi-seq global");
    drop(nodes);
    assert_eq!(mlx_native::commit_count(), commits_before);
}

/// The pre-fix sequence: a builder encoded straight into the session
/// encoder is invisible to the tracker, so the classifier's
/// `barrier_between(&[mask], ..)` emits no barrier. Pins that the fence above
/// comes from the fill declaration, not from the reader's own declaration.
#[test]
fn undeclared_mask_fill_leaves_tile_classifier_unfenced() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Some(device) = device() else { return };
    let mut reg = registry();
    let (params, blk_buffer) = blk(&device, 32, 16);
    let executor = GraphExecutor::new(device.clone());
    let mut s = executor.begin_recorded().unwrap();
    let mask = build_sdpa_mask_bf16(
        &device,
        &mut reg,
        s.encoder_mut(),
        &SdpaMaskParams {
            seq_len_q: LEN as u32,
            seq_len_k: LEN as u32,
            window_size: Some(16),
            causal: true,
            q_abs_offset: 0,
        },
    )
    .unwrap();
    encode_mask_tiles(
        &mut s,
        &device,
        &mut reg,
        &mask,
        &blk_buffer,
        &params,
        "blk",
    )
    .unwrap();
    let nodes = s.encoder_mut().take_capture().unwrap();
    drop(s);
    let (_, fenced) = readers(&nodes, &mask);
    assert_eq!(fenced, vec![false]);
}
