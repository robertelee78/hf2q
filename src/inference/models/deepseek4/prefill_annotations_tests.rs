//! Capture-only: eight guarded roots/3468 requested bytes; no GPU submission.
use super::*;
use mlx_native::{CapturedNode, ComputeGraph, DType, GraphExecutor, RecordedBinding};

struct Root {
    owner: MlxBuffer,
    view: MlxBuffer,
}
impl Root {
    fn new(device: &MlxDevice, dtype: DType, shape: Vec<usize>) -> Self {
        let elements = shape.iter().product::<usize>();
        let bytes = elements * dtype.size_of();
        let mut owner = device
            .alloc_buffer(bytes + 128, dtype, vec![(bytes + 128) / dtype.size_of()])
            .unwrap();
        owner.as_mut_slice::<u8>().unwrap().fill(0x5a);
        let mut view = owner.slice_view(64, elements).with_shape(shape).unwrap();
        view.as_logical_mut_slice::<u8>().unwrap().fill(0);
        Self { owner, view }
    }
    fn guards(&self) {
        let bytes = self.owner.as_slice::<u8>().unwrap();
        assert!(bytes[..64].iter().all(|&v| v == 0x5a));
        assert!(bytes[bytes.len() - 64..].iter().all(|&v| v == 0x5a));
    }
}

// Bind the oracle to the actual captured Metal object/offset, not a second
// annotation list. Whole-root conservative ranges may contain this logical view.
fn role_covered(node: &CapturedNode, slot: u64, expected: &MlxBuffer, read: bool) -> bool {
    let CapturedNode::Dispatch {
        bindings,
        reads,
        writes,
        ..
    } = node
    else {
        panic!("expected actual dispatch");
    };
    let (
        _,
        RecordedBinding::Buffer {
            metal_buffer,
            offset,
        },
    ) = bindings.iter().find(|(index, _)| *index == slot).unwrap()
    else {
        panic!("expected buffer binding");
    };
    assert!(std::ptr::eq(&**metal_buffer, &**expected.metal_buffer()));
    assert_eq!(*offset, expected.byte_offset());
    let begin = metal_buffer.contents() as usize + *offset as usize;
    let end = begin + expected.data_byte_len();
    let ranges = if read { reads } else { writes };
    ranges
        .iter()
        .any(|&(start, stop)| start <= begin && stop >= end)
}
fn remove_actual_read(node: &CapturedNode, buffer: &MlxBuffer) -> CapturedNode {
    let mut changed = node.clone();
    let CapturedNode::Dispatch { reads, .. } = &mut changed else {
        unreachable!()
    };
    let begin = buffer.contents_ptr() as usize + buffer.byte_offset() as usize;
    let end = begin + buffer.data_byte_len();
    reads.retain(|&(start, stop)| !(start <= begin && stop >= end));
    assert!(
        !reads.is_empty(),
        "negative must evade empty-read detection"
    );
    changed
}

#[test]
#[ignore = "tiny Metal allocation and pipeline capture; no dispatch replay/submission"]
fn prefill_annotation_actual_capture_covers_parameters_and_rope_reads() {
    let device = MlxDevice::new().unwrap();
    let mut registry = KernelRegistry::new();
    let mut input = Root::new(&device, DType::F32, vec![1, 1, 1, 128]);
    input.view.as_logical_mut_slice::<f32>().unwrap().fill(1.0);
    let mut weight = Root::new(&device, DType::F32, vec![128]);
    weight.view.as_logical_mut_slice::<f32>().unwrap().fill(1.0);
    let scaled = Root::new(&device, DType::F32, vec![1, 128]);
    let unscaled = Root::new(&device, DType::F32, vec![1, 128]);
    let mut params = Root::new(&device, DType::F32, vec![2]);
    params
        .view
        .as_logical_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[1e-6, 128.0]);
    let positions = Root::new(&device, DType::U32, vec![1]);
    let mut frequencies = Root::new(&device, DType::F32, vec![32]);
    frequencies
        .view
        .as_logical_mut_slice::<f32>()
        .unwrap()
        .fill(1.0);
    let rope_output = Root::new(&device, DType::BF16, vec![1, 1, 1, 128]);
    let roots = [
        &input,
        &weight,
        &scaled,
        &unscaled,
        &params,
        &positions,
        &frequencies,
        &rope_output,
    ];
    assert_eq!(
        roots
            .iter()
            .map(|root| root.owner.data_byte_len())
            .sum::<usize>(),
        3468
    );
    let commits_before = mlx_native::commit_count();
    let executor = GraphExecutor::new(device.clone());
    let mut session = executor.begin_recorded().unwrap();
    rms_norm(
        &mut session,
        &mut registry,
        device.metal_device(),
        &input.view,
        &weight.view,
        &scaled.view,
        &params.view,
        1,
        128,
    )
    .unwrap();
    rms_norm_no_scale_f32(
        &mut session,
        &mut registry,
        device.metal_device(),
        &input.view,
        &unscaled.view,
        &params.view,
        1,
        128,
    )
    .unwrap();
    indexer_query_rope(
        &mut session,
        &mut registry,
        &device,
        &input.view,
        &positions.view,
        &frequencies.view,
        &rope_output.view,
        &DeepSeekTailRopeParams {
            batch: 1,
            seq_len: 1,
            heads: 1,
            head_dim: 128,
            rope_dim: 64,
            inverse: 0,
        },
    )
    .unwrap();
    let captured = session.encoder_mut().take_capture().unwrap();
    // Never finish/commit/replay. Captured owners and session close before roots.
    drop(session);
    let dispatches: Vec<_> = captured
        .iter()
        .filter(|node| matches!(node, CapturedNode::Dispatch { .. }))
        .collect();
    assert_eq!(dispatches.len(), 3);
    assert_eq!(
        ComputeGraph::from_nodes(captured.clone()).unannotated_dispatch_count(),
        0
    );
    for (node, slots) in [
        (
            dispatches[0],
            vec![
                (0, &input.view, true),
                (1, &weight.view, true),
                (2, &scaled.view, false),
                (3, &params.view, true),
            ],
        ),
        (
            dispatches[1],
            vec![
                (0, &input.view, true),
                (1, &unscaled.view, false),
                (2, &params.view, true),
            ],
        ),
        (
            dispatches[2],
            vec![
                (0, &input.view, true),
                (1, &positions.view, true),
                (2, &frequencies.view, true),
                (3, &rope_output.view, false),
            ],
        ),
    ] {
        for (slot, buffer, read) in slots {
            assert!(role_covered(node, slot, buffer, read));
        }
    }
    for (index, slot, buffer) in [
        (0, 3, &params.view),
        (1, 2, &params.view),
        (2, 1, &positions.view),
        (2, 2, &frequencies.view),
    ] {
        let negative = remove_actual_read(dispatches[index], buffer);
        assert_eq!(
            ComputeGraph::from_nodes(vec![negative.clone()]).unannotated_dispatch_count(),
            0
        );
        assert!(!role_covered(&negative, slot, buffer, true));
    }
    for root in roots {
        root.guards();
    }
    drop(captured);
    assert_eq!(mlx_native::commit_count(), commits_before);
    println!("prefill_annotation_capture_complete roots=8 requested_bytes=3468 dispatches=3 missing_read_negatives=4 gpu_submitted=false model_loaded=false");
}
