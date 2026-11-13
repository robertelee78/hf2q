//! Model-free Metal gate: a row of the prompt-prefill MoE FFN
//! ([`super::gpu_ffn::build_moe_ffn_layer_gpu_q_into_with_arena`] on native
//! router / shared-expert storage, the served Qwen3.6 route) does not depend
//! on the other rows of its call, once the call has more than
//! [`super::prefill_joint::MOE_ROW_EXACT_MIN_ROWS`] rows.
//!
//! Above that row count every stage of the layer runs one kernel family
//! whatever the row count (tensor matmul router, quantized MM shared expert,
//! expert-grouped `mm_id`), and each computes an output row from that input
//! row alone. A joint prefill relies on it: it runs the layer once over the
//! rows of every lane whose own forward has more than that many rows, and
//! each lane must receive the bits of its own call
//! ([`super::prefill_joint`]). At or below the threshold the layer picks
//! other kernels (per-pair expert matvecs, the row-exact router matvec), so
//! such calls are not interchangeable with larger ones; the gate prints
//! what it finds there and asserts nothing.
//!
//! Layer shape and weights: the Qwen3.6-35B-A3B fixture of
//! [`super::moe_batched_tests`].

use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::gpu_ffn::{build_moe_ffn_layer_gpu_q_into_with_arena, MoeFfnWeightsGpuQ};
use super::moe_batched_tests::{packed, rows, shape, Layer, H};
use super::prefill_joint::MOE_ROW_EXACT_MIN_ROWS;

/// One arena-route layer over the rows `select` of `x` / `residual`, as one
/// call in its own command buffer; the output rows as bits.
fn layer_rows(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &MoeFfnWeightsGpuQ,
    x: &[Vec<f32>],
    residual: &[Vec<f32>],
    select: &[usize],
) -> Vec<Vec<u32>> {
    let shape = shape();
    let n = select.len();
    let x = packed(x, select, device);
    let residual = packed(residual, select, device);
    let mut arena = super::MoeFfnArena::new(
        device,
        n as u32,
        shape.hidden_size,
        shape.num_experts_per_tok,
        shape.moe_intermediate_size,
        shape.shared_intermediate_size,
        shape.num_experts,
    )
    .expect("MoeFfnArena");
    let mut out: MlxBuffer = device
        .alloc_buffer(n * H * 4, DType::F32, vec![n, H])
        .expect("output rows");
    let mut enc = device
        .command_encoder()
        .map(crate::inference::native_observation::encoder)
        .expect("encoder");
    build_moe_ffn_layer_gpu_q_into_with_arena(
        &mut enc,
        device,
        registry,
        &x,
        weights,
        shape,
        Some(&residual),
        &mut arena,
        &mut out,
        0,
    )
    .expect("prefill MoE layer");
    enc.commit_and_wait().expect("commit prefill MoE layer");
    out.as_slice::<u32>().expect("output bits")[..n * H]
        .chunks(H)
        .map(<[u32]>::to_vec)
        .collect()
}

/// Rows of `together` (a call over `parts` concatenated) that differ from
/// the same rows computed by each part's own call.
fn rows_that_depend_on_the_call(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &MoeFfnWeightsGpuQ,
    x: &[Vec<f32>],
    residual: &[Vec<f32>],
    parts: &[std::ops::Range<usize>],
) -> Vec<usize> {
    let all: Vec<usize> = parts.iter().flat_map(|part| part.clone()).collect();
    let together = layer_rows(device, registry, weights, x, residual, &all);
    let mut differing = Vec::new();
    let mut at = 0;
    for part in parts {
        let select: Vec<usize> = part.clone().collect();
        let alone = layer_rows(device, registry, weights, x, residual, &select);
        for (offset, row) in alone.iter().enumerate() {
            if *row != together[at + offset] {
                differing.push(at + offset);
            }
        }
        at += select.len();
    }
    differing
}

fn gate(down_t: GgmlType, seed: u64) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut registry = KernelRegistry::new();
    let layer = Layer::random(down_t, seed);
    let experts = layer.gpu_experts(&device);
    let weights = layer.gpu(&device, true, &experts);
    let (x, residual) = rows(1_600, seed ^ 0x5EED);
    let min = MOE_ROW_EXACT_MIN_ROWS as usize;
    assert_eq!(min, 32, "the cases below straddle 32 rows");

    // Calls above the threshold, cut at and off the 8-, 32- and 128-row tile
    // edges of the kernels, a part repeated at another offset, three and
    // four parts, and the first-turn shape (a shared prefix and four tails).
    let exact: [&[std::ops::Range<usize>]; 9] = [
        &[0..33, 33..66],
        &[0..40, 40..197],
        &[0..157, 157..253],
        &[0..64, 64..128],
        &[0..129, 129..257],
        &[100..133, 0..96, 300..457],
        &[0..162, 200..362, 400..562, 600..762],
        &[0..687, 700..981],
        &[0..406, 406..687, 700..981, 1000..1281, 1300..1581],
    ];
    for parts in exact {
        let differing =
            rows_that_depend_on_the_call(&device, &mut registry, &weights, &x, &residual, parts);
        assert!(
            differing.is_empty(),
            "down {down_t:?}: {} row(s) of a call over {parts:?} differ from their own call's \
             (first at {:?})",
            differing.len(),
            differing.first()
        );
    }
    // A row's output does not depend on its position in a call either.
    let forward = layer_rows(&device, &mut registry, &weights, &x, &residual, &(0..70).collect::<Vec<_>>());
    let reversed = layer_rows(&device, &mut registry, &weights, &x, &residual, &(0..70).rev().collect::<Vec<_>>());
    for row in 0..70 {
        assert!(
            forward[row] == reversed[69 - row],
            "down {down_t:?}: row {row} depends on its position in a 70-row call"
        );
    }
    // Repeatable on fresh scratch.
    let again = layer_rows(&device, &mut registry, &weights, &x, &residual, &(0..70).collect::<Vec<_>>());
    assert!(forward == again, "down {down_t:?}: a 70-row call is not repeatable");

    // At or below the threshold: reported, not asserted.
    let small: [&[std::ops::Range<usize>]; 5] = [
        &[0..5, 5..45],
        &[0..20, 20..60],
        &[0..31, 31..62],
        &[0..32, 32..72],
        &[0..8, 8..16],
    ];
    for parts in small {
        let differing =
            rows_that_depend_on_the_call(&device, &mut registry, &weights, &x, &residual, parts);
        eprintln!(
            "[MOE_PREFILL_ROWS] down={down_t:?} parts={parts:?}: {} of {} rows differ from their \
             own call's",
            differing.len(),
            parts.iter().map(std::ops::Range::len).sum::<usize>()
        );
    }
}

/// The layer as a joint prefill forward runs it over `segments` (row counts,
/// laid side by side): one call over every row, a barrier, then each
/// segment of at most [`MOE_ROW_EXACT_MIN_ROWS`] rows again on its own
/// scratch over its rows of that output, all in one command buffer. Returns
/// the output rows.
fn joint_forward_layer(
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    weights: &MoeFfnWeightsGpuQ,
    x: &[Vec<f32>],
    residual: &[Vec<f32>],
    segments: &[usize],
) -> Vec<Vec<u32>> {
    let shape = shape();
    let n: usize = segments.iter().sum();
    let all: Vec<usize> = (0..n).collect();
    let x = packed(x, &all, device);
    let residual = packed(residual, &all, device);
    let arena = |rows: usize| {
        super::MoeFfnArena::new(
            device,
            rows as u32,
            shape.hidden_size,
            shape.num_experts_per_tok,
            shape.moe_intermediate_size,
            shape.shared_intermediate_size,
            shape.num_experts,
        )
        .expect("MoeFfnArena")
    };
    let mut forward_arena = arena(n);
    let mut short_arenas: Vec<super::MoeFfnArena> = segments
        .iter()
        .filter(|&&rows| rows <= MOE_ROW_EXACT_MIN_ROWS as usize)
        .map(|&rows| arena(rows))
        .collect();
    let out: MlxBuffer = device
        .alloc_buffer(n * H * 4, DType::F32, vec![n, H])
        .expect("output rows");
    let mut enc = device
        .command_encoder()
        .map(crate::inference::native_observation::encoder)
        .expect("encoder");
    let mut all_rows = out.clone();
    build_moe_ffn_layer_gpu_q_into_with_arena(
        &mut enc,
        device,
        registry,
        &x,
        weights,
        shape,
        Some(&residual),
        &mut forward_arena,
        &mut all_rows,
        0,
    )
    .expect("forward-wide MoE layer");
    enc.memory_barrier();
    let mut first = 0usize;
    let mut short = short_arenas.iter_mut();
    for &rows in segments {
        if rows <= MOE_ROW_EXACT_MIN_ROWS as usize {
            let view = |buffer: &MlxBuffer| {
                super::prefill_joint::row_view(buffer, first as u32, rows as u32, H)
                    .with_shape(vec![rows, H])
                    .expect("segment rows")
            };
            let mut segment_out = view(&out);
            build_moe_ffn_layer_gpu_q_into_with_arena(
                &mut enc,
                device,
                registry,
                &view(&x),
                weights,
                shape,
                Some(&view(&residual)),
                short.next().expect("short segment scratch"),
                &mut segment_out,
                0,
            )
            .expect("short segment MoE layer");
        }
        first += rows;
    }
    enc.commit_and_wait().expect("commit joint MoE layer");
    out.as_slice::<u32>().expect("output bits")[..n * H]
        .chunks(H)
        .map(<[u32]>::to_vec)
        .collect()
}

/// The joint prefill forward's MoE FFN: every segment, long or short, holds
/// the rows of its own call, in forwards that mix both, that hold only long
/// ones and only short ones.
#[test]
fn a_joint_forward_gives_every_segment_the_rows_of_its_own_moe_call() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let mut registry = KernelRegistry::new();
    let layer = Layer::random(GgmlType::Q6_K, 0x51A5_0003);
    let experts = layer.gpu_experts(&device);
    let weights = layer.gpu(&device, true, &experts);
    let (x, residual) = rows(700, 0x5EED_0003);
    let layouts: [&[usize]; 5] = [
        // Folded cues behind resumed chunks, a short chunk between them.
        &[157, 5, 31, 200, 5],
        // A shared prefix and three tails with their cues.
        &[160, 96, 5, 96, 5, 33, 5],
        // Long segments only.
        &[162, 162, 96],
        // A long segment and a 32-row one (the last short row count).
        &[33, 32, 2],
        // One lane: a folded chunk.
        &[266, 5],
    ];
    for segments in layouts {
        let together =
            joint_forward_layer(&device, &mut registry, &weights, &x, &residual, segments);
        let mut first = 0usize;
        for &segment_rows in segments {
            let select: Vec<usize> = (first..first + segment_rows).collect();
            let alone = layer_rows(&device, &mut registry, &weights, &x, &residual, &select);
            assert!(
                alone[..] == together[first..first + segment_rows],
                "segment of {segment_rows} rows at {first} in a joint forward of {segments:?} \
                 differs from its own call"
            );
            first += segment_rows;
        }
    }
}

#[test]
fn a_prefill_moe_row_above_32_rows_does_not_depend_on_its_call_down_q6_k() {
    gate(GgmlType::Q6_K, 0x51A5_0001);
}

#[test]
fn a_prefill_moe_row_above_32_rows_does_not_depend_on_its_call_down_q4_k() {
    gate(GgmlType::Q4_K, 0x51A5_0002);
}
