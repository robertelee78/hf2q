//! Model-free exactness of the prefill down call's schedule reuse (fix 2).
//!
//! Encodes the DeepSeek-V4 prefill FFN's routed sequence with hf2q's own
//! wrappers — paired IQ2_XXS gate/up (`expert_matmul_pair`, which builds the
//! `map0c` schedule in scratch 0), an unrelated dispatch in between, then the
//! slotted Q2_K down call — and requires the schedule-reusing down call to be
//! byte-identical to the self-scheduling one, in a plain session and in a
//! recorded session finished with the graph reorder pass (the multi-layer
//! prefill graph). A schedule built for other ids is refused. Two consecutive
//! MoE layers sharing one scratch in one recorded, reordered session (the
//! next layer's `map0c` overwrites the schedule the previous down call
//! reads) equal the same layers run in separate committed sessions with
//! self-scheduling down calls.

use mlx_native::graph::GraphSession;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, IdMmScratch, KernelRegistry, MlxBuffer, MlxDevice};

use super::ds4_bench_support::{bench_matvec, f32_buf, Rng};
use super::ds4_op_microbench_tests::block_weight;
use super::forward_support::{alloc, expert_matmul, expert_matmul_pair, ExpertMatmulRoute};
use super::residency::RawMatrixRef;
use crate::serve::gpu::GpuContext;

const EXPERTS: usize = 32;
const TOP_K: usize = 6;
const HIDDEN: usize = 512;
const INTER: usize = 256;

fn ids(device: &MlxDevice, rows: usize, seed: u64) -> MlxBuffer {
    let mut buffer = alloc(device, DType::U32, vec![rows, TOP_K], "ids").unwrap();
    let mut rng = Rng(seed);
    for row in buffer.as_mut_slice::<u32>().unwrap().chunks_mut(TOP_K) {
        let mut chosen: Vec<u32> = Vec::with_capacity(TOP_K);
        while chosen.len() < TOP_K {
            let expert = (rng.next() % EXPERTS as u64) as u32;
            if !chosen.contains(&expert) {
                chosen.push(expert);
            }
        }
        row.copy_from_slice(&chosen);
    }
    buffer
}

struct Fixture {
    gate: MlxBuffer,
    up: MlxBuffer,
    down: MlxBuffer,
    x: MlxBuffer,
    act: MlxBuffer,
    g: MlxBuffer,
    u: MlxBuffer,
    filler: MlxBuffer,
    rows: usize,
}

impl Fixture {
    fn new(device: &MlxDevice, rows: usize) -> Self {
        let zeroed = |elements: usize| {
            let mut buffer = alloc(device, DType::F32, vec![elements], "out").unwrap();
            buffer.as_mut_slice::<u8>().unwrap().fill(0);
            buffer
        };
        Self {
            gate: block_weight(device, GgmlType::IQ2_XXS, vec![EXPERTS, INTER, HIDDEN], 1),
            up: block_weight(device, GgmlType::IQ2_XXS, vec![EXPERTS, INTER, HIDDEN], 2),
            down: block_weight(device, GgmlType::Q2_K, vec![EXPERTS, HIDDEN, INTER], 3),
            x: f32_buf(device, vec![rows, HIDDEN], 4, 1.0),
            act: f32_buf(device, vec![rows * TOP_K, INTER], 5, 1.0),
            g: zeroed(rows * TOP_K * INTER),
            u: zeroed(rows * TOP_K * INTER),
            filler: f32_buf(device, vec![rows, HIDDEN], 6, 1.0),
            rows,
        }
    }

    /// Paired gate/up on `pair_ids` (builds the schedule in `scratch`), an
    /// unrelated dense dispatch, then the down call on `down_ids`.
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        session: &mut GraphSession<'_>,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        scratch: &mut IdMmScratch,
        pair_ids: &MlxBuffer,
        down_ids: &MlxBuffer,
        route: ExpertMatmulRoute,
        output: &MlxBuffer,
    ) -> anyhow::Result<()> {
        let gu_shape = [EXPERTS, INTER, HIDDEN];
        let d_shape = [EXPERTS, HIDDEN, INTER];
        let weight = |buffer, ggml_type, shape| RawMatrixRef {
            buffer,
            ggml_type,
            shape,
            matvec: bench_matvec(),
        };
        expert_matmul_pair(
            session,
            registry,
            device,
            &self.x,
            &weight(&self.gate, GgmlType::IQ2_XXS, &gu_shape),
            &weight(&self.up, GgmlType::IQ2_XXS, &gu_shape),
            pair_ids,
            &self.g,
            &self.u,
            self.rows,
            TOP_K,
            EXPERTS,
            INTER,
            HIDDEN,
            scratch,
            "test paired gate/up",
        )?;
        // Stands in for the shared expert and SwiGLU between gate/up and down.
        session.barrier_between(&[&self.x], &[&self.filler]);
        mlx_native::ops::copy::dispatch_copy_f32(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            &self.x,
            &self.filler,
            0,
            0,
            self.rows * HIDDEN,
        )?;
        expert_matmul(
            session,
            registry,
            device,
            &self.act,
            &weight(&self.down, GgmlType::Q2_K, &d_shape),
            down_ids,
            output,
            self.rows,
            TOP_K,
            EXPERTS,
            HIDDEN,
            INTER,
            route,
            Some(scratch),
            "test routed down",
        )
    }
}

fn output(device: &MlxDevice, rows: usize) -> MlxBuffer {
    let mut buffer = alloc(device, DType::F32, vec![rows * TOP_K * HIDDEN], "down").unwrap();
    buffer.as_mut_slice::<u8>().unwrap().fill(0);
    buffer
}

fn assert_reuse_matches(rows: usize, recorded: bool) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let fixture = Fixture::new(&device, rows);
    let id = ids(&device, rows, 11);
    let mut scratch = IdMmScratch::alloc(&device, EXPERTS as u32, rows as u32).unwrap();
    let run = |ctx: &mut GpuContext, scratch: &mut IdMmScratch, route, out: &MlxBuffer| {
        let (executor, registry) = ctx.split();
        let mut session = if recorded {
            executor.begin_recorded().unwrap()
        } else {
            executor.begin().unwrap()
        };
        fixture
            .encode(&mut session, registry, &device, scratch, &id, &id, route, out)
            .unwrap();
        if recorded {
            session.finish_with_reorder().unwrap();
        } else {
            session.finish().unwrap();
        }
    };
    let fresh = output(&device, rows);
    run(&mut ctx, &mut scratch, ExpertMatmulRoute::SlottedMm, &fresh);
    let reused = output(&device, rows);
    run(&mut ctx, &mut scratch, ExpertMatmulRoute::SlottedMmReusingSchedule, &reused);
    let fresh_bits = fresh.as_slice::<u32>().unwrap();
    assert!(fresh_bits.iter().all(|&bits| f32::from_bits(bits).is_finite()));
    assert!(fresh_bits.iter().any(|&bits| bits != 0), "down never ran");
    assert!(
        fresh_bits == reused.as_slice::<u32>().unwrap(),
        "rows {rows} recorded {recorded}: reused schedule differs from a fresh schedule"
    );
}

#[test]
fn down_schedule_reuse_is_bitwise_in_a_plain_session() {
    assert_reuse_matches(100, false);
    assert_reuse_matches(333, false);
}

#[test]
fn down_schedule_reuse_is_bitwise_in_a_reordered_recorded_session() {
    assert_reuse_matches(100, true);
    assert_reuse_matches(333, true);
}

#[test]
fn down_schedule_reuse_refuses_a_schedule_for_other_ids() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let rows = 100;
    let fixture = Fixture::new(&device, rows);
    let pair_ids = ids(&device, rows, 11);
    let down_ids = ids(&device, rows, 12);
    let mut scratch = IdMmScratch::alloc(&device, EXPERTS as u32, rows as u32).unwrap();
    let (executor, registry) = ctx.split();
    let mut session = executor.begin().unwrap();
    let refused = fixture.encode(
        &mut session,
        registry,
        &device,
        &mut scratch,
        &pair_ids,
        &down_ids,
        ExpertMatmulRoute::SlottedMmReusingSchedule,
        &output(&device, rows),
    );
    assert!(refused.is_err());
    session.finish().unwrap();
}

/// Cross-layer scratch hazards with reuse: layer A (ids A) and layer B (ids
/// B) run back to back in one recorded session finished with the reorder
/// pass, both reusing the paired schedule from the one shared scratch, with
/// no data dependency between the layers (the worst case for reordering:
/// only the declared scratch ranges keep B's `map0c` after A's down reads).
/// Reference: each layer alone in a committed plain session with the
/// self-scheduling down call.
#[test]
fn down_schedule_reuse_across_two_layers_in_one_recorded_session_is_bitwise() {
    const ROWS: usize = 333;
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let fixture = Fixture::new(&device, ROWS);
    let layer_ids = [ids(&device, ROWS, 11), ids(&device, ROWS, 23)];
    assert_ne!(
        layer_ids[0].as_slice::<u32>().unwrap(),
        layer_ids[1].as_slice::<u32>().unwrap(),
        "the two layers must route differently"
    );
    let mut scratch = IdMmScratch::alloc(&device, EXPERTS as u32, ROWS as u32).unwrap();

    let reference: Vec<MlxBuffer> = layer_ids
        .iter()
        .map(|id| {
            let out = output(&device, ROWS);
            let (executor, registry) = ctx.split();
            let mut session = executor.begin().unwrap();
            fixture
                .encode(&mut session, registry, &device, &mut scratch, id, id, ExpertMatmulRoute::SlottedMm, &out)
                .unwrap();
            session.finish().unwrap();
            out
        })
        .collect();

    let reused: Vec<MlxBuffer> = (0..layer_ids.len()).map(|_| output(&device, ROWS)).collect();
    {
        let (executor, registry) = ctx.split();
        let mut session = executor.begin_recorded().unwrap();
        for (id, out) in layer_ids.iter().zip(&reused) {
            fixture
                .encode(
                    &mut session,
                    registry,
                    &device,
                    &mut scratch,
                    id,
                    id,
                    ExpertMatmulRoute::SlottedMmReusingSchedule,
                    out,
                )
                .unwrap();
            session.barrier();
        }
        session.finish_with_reorder().unwrap();
    }
    for (layer, (expected, actual)) in reference.iter().zip(&reused).enumerate() {
        let expected = expected.as_slice::<u32>().unwrap();
        assert!(expected.iter().any(|&bits| bits != 0), "layer {layer} down never ran");
        assert!(
            expected == actual.as_slice::<u32>().unwrap(),
            "layer {layer}: reused schedule in the two-layer recorded session differs"
        );
    }
}
