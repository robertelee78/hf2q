use mlx_native::graph::GraphSession;
use mlx_native::ops::copy::dispatch_copy_f32;
use mlx_native::{DType, GraphExecutor, MlxBuffer};

use super::cache::Deepseek4Cache;
use super::forward_support::alloc;
use super::model::Deepseek4Model;
use super::real_artifact_tests::official_artifact;

const PREFIX_ROWS: usize = 148;
const PHYSICAL_TO_LOGICAL: [usize; 4] = [2, 0, 3, 1];

fn prefix_tokens(logical_lane: usize) -> Vec<u32> {
    (0..PREFIX_ROWS)
        .map(|row| ((row * 97 + logical_lane * 193 + 11) % 120_000) as u32)
        .collect()
}

fn supplied_tokens() -> [u32; 4] {
    std::array::from_fn(|logical_lane| ((logical_lane * 7_919 + 17) % 120_000) as u32)
}

fn read_f32(buffer: &MlxBuffer, label: &str) -> Vec<f32> {
    buffer
        .as_logical_slice::<f32>()
        .unwrap_or_else(|error| panic!("read {label}: {error}"))
        .to_vec()
}

fn first_mismatch(expected: &[f32], actual: &[f32]) -> Option<(usize, f32, f32)> {
    assert_eq!(expected.len(), actual.len());
    expected
        .iter()
        .zip(actual)
        .enumerate()
        .find(|(_, (expected, actual))| expected.to_bits() != actual.to_bits())
        .map(|(index, (&expected, &actual))| (index, expected, actual))
}

fn serial_layer0(
    model: &mut Deepseek4Model,
    tokens: [u32; 4],
    caches: &mut [Deepseek4Cache],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut attentions = Vec::with_capacity(4);
    let mut states = Vec::with_capacity(4);
    for logical_lane in 0..4 {
        let attention = model
            .forward_uncompressed_attention_one(
                None,
                tokens[logical_lane],
                0,
                &mut caches[logical_lane],
                false,
                None,
                None,
            )
            .expect("serial layer-0 attention");
        attentions.push(read_f32(&attention, "serial layer-0 attention"));
        let state = model
            .forward_ffn_one(&attention, tokens[logical_lane], 0, None, None)
            .expect("serial layer-0 FFN");
        states.push(read_f32(&state, "serial layer-0 state"));
    }
    (attentions, states)
}

fn cohort_layer0(
    model: &mut Deepseek4Model,
    tokens: [u32; 4],
    caches: &mut [Deepseek4Cache],
) -> (Vec<f32>, Vec<f32>) {
    let hc = model.cfg.hyper_connection_count as usize;
    let hidden = model.cfg.hidden_size as usize;
    let row_elements = hc * hidden;
    let device = model.ctx.device().clone();
    let combined_attention = alloc(
        &device,
        DType::F32,
        vec![4, hc, hidden],
        "diagnostic B=4 attention",
    )
    .expect("allocate diagnostic combined attention");
    let executor = GraphExecutor::new(device.clone());
    let mut session: GraphSession<'_> = executor.begin().expect("begin diagnostic B=4 layer 0");
    let mut attentions = Vec::with_capacity(4);
    for physical_lane in 0..4 {
        attentions.push(
            model
                .forward_uncompressed_attention_one(
                    None,
                    tokens[physical_lane],
                    0,
                    &mut caches[physical_lane],
                    false,
                    None,
                    Some(&mut session),
                )
                .expect("B=4 layer-0 attention"),
        );
    }
    session.barrier();
    for (physical_lane, attention) in attentions.iter().enumerate() {
        dispatch_copy_f32(
            session.encoder_mut(),
            &mut model.ctx.registry,
            device.metal_device(),
            attention,
            &combined_attention,
            0,
            physical_lane * row_elements,
            row_elements,
        )
        .expect("pack diagnostic B=4 attention");
    }
    session.barrier();
    let state = model
        .forward_ffn_rows(
            &combined_attention,
            &tokens,
            0,
            None,
            Some(&mut session),
            None,
            None,
        )
        .expect("B=4 layer-0 FFN");
    session.finish().expect("execute diagnostic B=4 layer 0");
    (
        read_f32(&combined_attention, "B=4 layer-0 attention"),
        read_f32(&state, "B=4 layer-0 state"),
    )
}

#[test]
#[ignore = "loads the exact artifact and localizes the first B=4 decode divergence"]
fn official_artifact_b4_decode_layer0_divergence_is_in_multirow_ffn() {
    let (path, gguf) = official_artifact();
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut model = Deepseek4Model::load_from_gguf(&gguf)
        .unwrap_or_else(|error| panic!("load official artifact {}: {error:#}", path.display()));
    let prefixes = (0..4).map(prefix_tokens).collect::<Vec<_>>();
    let mut serial_caches = (0..4)
        .map(|logical_lane| {
            let mut cache = model.allocate_cache(PREFIX_ROWS + 1).unwrap();
            model
                .forward_verifier_prefill(&prefixes[logical_lane], &mut cache)
                .unwrap();
            cache
        })
        .collect::<Vec<_>>();
    let mut cohort_caches = PHYSICAL_TO_LOGICAL
        .iter()
        .map(|&logical_lane| {
            let mut cache = model.allocate_cache(PREFIX_ROWS + 1).unwrap();
            model
                .forward_verifier_prefill(&prefixes[logical_lane], &mut cache)
                .unwrap();
            cache
        })
        .collect::<Vec<_>>();
    let logical_tokens = supplied_tokens();
    let physical_tokens = PHYSICAL_TO_LOGICAL.map(|logical_lane| logical_tokens[logical_lane]);
    let (serial_attention, serial_state) =
        serial_layer0(&mut model, logical_tokens, &mut serial_caches);
    let (cohort_attention, cohort_state) =
        cohort_layer0(&mut model, physical_tokens, &mut cohort_caches);
    let row_elements = 4 * 4_096;
    let mut first_ffn_mismatch = None;
    for (physical_lane, &logical_lane) in PHYSICAL_TO_LOGICAL.iter().enumerate() {
        let attention =
            &cohort_attention[physical_lane * row_elements..(physical_lane + 1) * row_elements];
        let state = &cohort_state[physical_lane * row_elements..(physical_lane + 1) * row_elements];
        assert!(
            first_mismatch(&serial_attention[logical_lane], attention).is_none(),
            "layer-0 attention first differs for logical lane {logical_lane}: {:?}",
            first_mismatch(&serial_attention[logical_lane], attention)
        );
        if let Some(mismatch) = first_mismatch(&serial_state[logical_lane], state) {
            first_ffn_mismatch.get_or_insert((logical_lane, mismatch));
        }
    }
    let (logical_lane, mismatch) = first_ffn_mismatch
        .expect("multi-row layer-0 FFN unexpectedly matched all four serial m=1 lanes bit-for-bit");
    eprintln!(
        "multi-row layer-0 FFN first mismatch: logical_lane={logical_lane} index={} serial={} cohort={}",
        mismatch.0, mismatch.1, mismatch.2
    );
}
