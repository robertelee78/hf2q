//! Ignored actual HF postprojection boundary; each case runs in a fresh process.
use super::plan::{workspace_buffers, IndexerBufferPlan};
use super::selection::*;
use half::bf16;
use mlx_native::{
    CapturedNode, ComputeGraph, DType, GraphExecutor, KernelRegistry, MlxBuffer, MlxDevice,
};
const Q: usize = 257;
const STRIDE: usize = 640;
const COLUMN: usize = 128;
struct Root {
    owner: MlxBuffer,
    data: MlxBuffer,
}
impl Root {
    fn new(d: &MlxDevice, dtype: DType, shape: Vec<usize>) -> Self {
        Self::allocate(d, dtype, shape, "HF indexer control")
    }
    fn allocate(d: &MlxDevice, dtype: DType, shape: Vec<usize>, label: &str) -> Self {
        let n = shape.iter().product::<usize>();
        let bytes = n * dtype.size_of();
        // Actual HF allocation route; fresh test thread remains Inactive.
        let mut owner = super::super::forward_support::alloc(
            d,
            dtype,
            vec![(bytes + 128) / dtype.size_of()],
            label,
        )
        .unwrap();
        assert_eq!(
            owner.byte_len(),
            bytes + 128,
            "must use exact inactive allocator"
        );
        owner.as_mut_slice::<u8>().unwrap().fill(0x5a);
        let mut data = owner.slice_view(64, n).with_shape(shape).unwrap();
        data.as_logical_mut_slice::<u8>().unwrap().fill(0);
        Self { owner, data }
    }
    fn descriptor(d: &MlxDevice, p: IndexerBufferPlan) -> Self {
        p.allocate_with(|dtype, shape, label| Ok(Self::allocate(d, dtype, shape, label)))
            .unwrap()
    }
    fn guards(&self) {
        let b = self.owner.as_slice::<u8>().unwrap();
        assert!(b[..64]
            .iter()
            .chain(b[b.len() - 64..].iter())
            .all(|x| *x == 0x5a));
    }
}
fn range(b: &MlxBuffer, offset: usize, bytes: usize) -> (usize, usize) {
    let begin = b.contents_ptr() as usize + b.byte_offset() as usize + offset;
    (begin, begin + bytes)
}
#[allow(clippy::too_many_arguments)]
fn capture(
    nodes: &[CapturedNode],
    k: usize,
    q: &MlxBuffer,
    kv: &MlxBuffer,
    w: &MlxBuffer,
    score: &MlxBuffer,
    top: &MlxBuffer,
    out: &MlxBuffer,
) {
    let dispatches: Vec<_> = nodes
        .iter()
        .filter_map(|n| match n {
            CapturedNode::Dispatch { reads, writes, .. } => Some((reads, writes)),
            _ => None,
        })
        .collect();
    let blocks = k.div_ceil(1024);
    let per_tile = if blocks == 1 { 3 } else { 4 };
    assert_eq!(dispatches.len(), 2 * per_tile);
    for (tile, (origin, rows)) in [(0, 256), (256, 1)].into_iter().enumerate() {
        let s = range(score, 0, rows * k * 4);
        let a = range(top, 0, rows * blocks * 512 * 4);
        let b = range(top, rows * blocks * 512 * 4, rows * blocks * 512 * 4);
        let mut expected = vec![
            (
                vec![
                    range(q, origin * 64 * 128 * 2, rows * 64 * 128 * 2),
                    range(kv, 0, k * 128 * 2),
                    range(w, origin * 64 * 4, rows * 64 * 4),
                ],
                vec![s],
            ),
            (vec![s], vec![a]),
        ];
        if blocks > 1 {
            expected.push((vec![s, a], vec![b]));
        }
        expected.push((
            vec![if blocks == 1 { a } else { b }],
            vec![range(out, origin * STRIDE * 4, rows * STRIDE * 4)],
        ));
        for (stage, (reads, writes)) in expected.iter().enumerate() {
            assert_eq!(dispatches[tile * per_tile + stage].0, reads);
            assert_eq!(dispatches[tile * per_tile + stage].1, writes);
        }
    }
}
fn oracle(
    out: &MlxBuffer,
    request: &SelectionRequest,
    q: &MlxBuffer,
    kv: &MlxBuffer,
    w: &MlxBuffer,
    alter_expected: bool,
) -> anyhow::Result<()> {
    let qb = q.as_logical_slice::<u16>().unwrap();
    let kb = kv.as_logical_slice::<u16>().unwrap();
    let wb = w.as_logical_slice::<f32>().unwrap();
    let values = out.as_logical_slice::<i32>().unwrap();
    assert!(qb
        .iter()
        .enumerate()
        .all(|(i, x)| i % (64 * 128) == 0 || *x == 0));
    assert!(kb.iter().enumerate().all(|(i, x)| i % 128 == 0 || *x == 0));
    assert!(wb.iter().enumerate().all(|(i, x)| i % 64 == 0 || *x == 0.0));
    let k = request.params.kv_len as usize;
    let mut row = Vec::<(f32, i32)>::with_capacity(k);
    assert!(row.capacity() * std::mem::size_of::<(f32, i32)>() <= 16384);
    for t in 0..Q {
        row.clear();
        let valid = k.min((request.params.start_pos as usize + t + 1) / 4);
        for j in 0..valid {
            let score = (bf16::from_bits(qb[t * 64 * 128]).to_f32()
                * bf16::from_bits(kb[j * 128]).to_f32())
            .max(0.0)
                * wb[t * 64];
            assert!(score.is_finite());
            row.push((score, j as i32));
        }
        row.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for rank in 0..512 {
            let expected = row.get(rank).map_or(-1, |x| x.1 + request.params.offset);
            let actual = values[t * STRIDE + COLUMN + rank];
            let expected = if alter_expected && expected >= 0 {
                expected + 1
            } else {
                expected
            };
            anyhow::ensure!(
                actual == expected,
                "row={t} rank={rank} actual={actual} expected={expected}"
            );
        }
        if t < 3 && request.params.start_pos == 0 {
            assert!(values[t * STRIDE + COLUMN..t * STRIDE + COLUMN + 512]
                .iter()
                .all(|x| *x == -1));
        }
    }
    Ok(())
}
fn run(case: &str, start: usize, k: usize, tail: f32) {
    let cfg = {
        let mut c = super::super::residency_tests::tiny_config();
        c.index_num_heads = 64;
        c.index_head_dim = 128;
        c.index_top_k = 512;
        c
    };
    let layout = super::layout::prepare_layout(4, Q, start, 0, 128, k, 512, true).unwrap();
    assert!(layout.preparation_numeric_capacity_bytes <= 900000);
    assert!(layout.index_storage.capacity() * 4 <= 700000);
    let request = SelectionRequest::new(
        &cfg,
        Q,
        start,
        k,
        layout.compressed_attention_offset,
        layout.storage_stride,
        layout.indexer_output_offset.unwrap(),
    )
    .unwrap();
    assert_eq!((request.stride, request.column), (STRIDE, COLUMN));
    assert_eq!(request.params.offset, if start == 0 { 257 } else { 385 });
    let d = MlxDevice::new().unwrap();
    let executor = GraphExecutor::new(d.clone());
    let mut registry = KernelRegistry::new();
    let mut q = Root::new(&d, DType::BF16, vec![1, Q, 64, 128]);
    let mut kv = Root::new(&d, DType::BF16, vec![1, k, 128]);
    let mut w = Root::new(&d, DType::F32, vec![1, Q, 64]);
    for t in 0..Q {
        q.data.as_logical_mut_slice::<u16>().unwrap()[t * 64 * 128] =
            bf16::from_f32(if t % 3 == 0 { 0.0 } else { 1.0 }).to_bits();
        w.data.as_logical_mut_slice::<f32>().unwrap()[t * 64] = if t % 3 == 0 { 0.0 } else { 1.0 };
    }
    for j in 0..k {
        kv.data.as_logical_mut_slice::<u16>().unwrap()[j * 128] =
            bf16::from_f32(if j == 1024 { tail } else { (j % 16) as f32 }).to_bits();
    }
    let full =
        mlx_native::ops::deepseek_indexer::deepseek_indexer_topk_scratch_elements(1, Q, k).unwrap();
    let score0 = Root::new(&d, DType::F32, vec![1, Q, k]);
    let top0 = Root::new(&d, DType::I32, vec![1, Q, full / Q]);
    let [sp, tp] = workspace_buffers(&request).unwrap();
    let score1 = Root::descriptor(&d, sp);
    let top1 = Root::descriptor(&d, tp);
    let mut out0 = Root::new(&d, DType::I32, vec![1, Q, STRIDE]);
    let mut out1 = Root::new(&d, DType::I32, vec![1, Q, STRIDE]);
    out0.data
        .as_logical_mut_slice::<i32>()
        .unwrap()
        .copy_from_slice(&layout.index_storage);
    out1.data
        .as_logical_mut_slice::<i32>()
        .unwrap()
        .copy_from_slice(&layout.index_storage);
    let bytes = [&q, &kv, &w, &score0, &top0, &score1, &top1, &out0, &out1]
        .iter()
        .map(|r| r.owner.data_byte_len())
        .sum::<usize>();
    assert_eq!(bytes, if k == 1025 { 12161668 } else { 7842432 });
    assert!(bytes + 1024 * 1024 < 16 * 1024 * 1024);
    let mut a = d.command_encoder().unwrap();
    mlx_native::ops::deepseek_indexer::dispatch_deepseek_indexer_into_mma(
        &mut a,
        &mut registry,
        &d,
        &q.data,
        &kv.data,
        &w.data,
        &score0.data,
        &top0.data,
        &out0.data,
        STRIDE,
        COLUMN,
        &request.params,
    )
    .unwrap();
    let mut session = executor.begin_recorded().unwrap();
    encode_selection(
        &mut session,
        &mut registry,
        &d,
        &request,
        &q.data,
        &kv.data,
        &w.data,
        &score1.data,
        &top1.data,
        &out1.data,
    )
    .unwrap();
    let nodes = session.encoder_mut().take_capture().unwrap();
    drop(session);
    capture(
        &nodes,
        k,
        &q.data,
        &kv.data,
        &w.data,
        &score1.data,
        &top1.data,
        &out1.data,
    );
    let mut graph = ComputeGraph::from_nodes(nodes);
    assert_eq!(graph.unannotated_dispatch_count(), 0);
    let moved = graph.reorder();
    let mut b = d.command_encoder().unwrap();
    let barriers = graph.encode_with_barriers(&mut b);
    a.commit();
    b.commit();
    let done_a = a.wait_until_completed();
    let done_b = b.wait_until_completed();
    // BOTH completion outcomes collected before assertion/readback/owner release.
    done_a.unwrap();
    done_b.unwrap();
    assert!(barriers > 0);
    assert_eq!(
        out0.data.as_logical_slice::<i32>().unwrap(),
        out1.data.as_logical_slice::<i32>().unwrap()
    );
    oracle(&out1.data, &request, &q.data, &kv.data, &w.data, false).unwrap();
    // Feed altered expected indices through the SAME actual-readback comparator.
    assert!(oracle(&out1.data, &request, &q.data, &kv.data, &w.data, true).is_err());
    let output = out1.data.as_logical_slice::<i32>().unwrap();
    for t in 0..Q {
        assert_eq!(
            &output[t * STRIDE..t * STRIDE + COLUMN],
            &layout.index_storage[t * STRIDE..t * STRIDE + COLUMN]
        );
    }
    if case == "winner" {
        assert_ne!(output[255 * STRIDE + COLUMN], output[256 * STRIDE + COLUMN]);
        assert_eq!(output[256 * STRIDE + COLUMN], 1409);
    }
    if case == "tie" {
        assert_eq!(output[256 * STRIDE + COLUMN + 63], 1408);
        assert_eq!(output[256 * STRIDE + COLUMN + 64], 1409);
    }
    for r in [&q, &kv, &w, &score0, &top0, &score1, &top1, &out0, &out1] {
        r.guards();
    }
    drop(graph);
    drop(a);
    drop(b);
    // All reused owners are fully drained. Rejections must add no dispatch.
    let short = score1
        .data
        .slice_view(0, score1.data.data_byte_len() / 4 - 1);
    let alias_output = top1
        .data
        .slice_view(0, Q * STRIDE)
        .with_shape(vec![1, Q, STRIDE])
        .unwrap();
    for (score, top, output) in [
        (&short, &top1.data, &out1.data),
        (&score1.data, &top1.data, &alias_output),
        (&score1.data, &score1.data, &out1.data),
    ] {
        let mut s = executor.begin_recorded().unwrap();
        assert!(encode_selection(
            &mut s,
            &mut registry,
            &d,
            &request,
            &q.data,
            &kv.data,
            &w.data,
            score,
            top,
            output
        )
        .is_err());
        assert!(!s
            .encoder_mut()
            .take_capture()
            .unwrap()
            .iter()
            .any(|n| matches!(n, CapturedNode::Dispatch { .. })));
    }
    // Explicit actual native pipeline provenance; never force origin/override policy.
    for kernel in [
        "deepseek_indexer_score_mma_bf16",
        "deepseek_indexer_topk_block_i32",
        "deepseek_indexer_topk_merge_i32",
        "deepseek_indexer_topk_finalize_i32",
    ] {
        let identity = registry.pipeline_identity(kernel).unwrap();
        println!("HF_INDEXER_PIPELINE case={case} identity={identity:?}");
    }
    println!("HF_INDEXER_TILED_COMPLETE case={case} roots=9 metal_bytes={bytes} host_numeric_bound=1048576 cpu=true untiled=true captured_replay=true moved={moved} barriers={barriers} zero_dispatch_negatives=3 runtime_admission=false");
}
#[test]
#[ignore = "fresh-process bounded HF selection helper GPU control"]
fn hf_indexer_tiles_causal_winner() {
    run("winner", 3843, 1025, 32.0);
}
#[test]
#[ignore = "fresh-process bounded HF selection helper GPU control"]
fn hf_indexer_tiles_cross_block_tie() {
    run("tie", 3843, 1025, 15.0);
}
#[test]
#[ignore = "fresh-process bounded HF selection helper GPU control"]
fn hf_indexer_tiles_early_sentinels() {
    run("early", 0, 64, 32.0);
}
