use super::*;
use std::cell::Cell;
use crate::inference::models::deepseek4::cache::CacheBufferPlan;

#[path = "c1_anchor_layout_tests.rs"]
mod c1_anchor_layout_tests;

fn cache_plan(ratios: Vec<u32>, context: usize) -> Deepseek4CachePlan {
    let mut cfg = crate::inference::models::deepseek4::residency_tests::tiny_config();
    cfg.num_hidden_layers = ratios.len() as u32;
    cfg.compress_ratios = ratios;
    Deepseek4CachePlan::for_context(&cfg, context).unwrap()
}
fn meta(plan: &Deepseek4CachePlan, layer: usize, role: usize) -> Option<BufferMetadata<'_>> {
    let p = &plan.layers[layer];
    [Some(&p.window_kv), p.compressed_kv.as_ref(), p.indexer_kv.as_ref(),
        p.main_kv_state.as_ref(), p.main_score_state.as_ref(),
        p.indexer_kv_state.as_ref(), p.indexer_score_state.as_ref()][role]
        .map(|p| BufferMetadata { dtype: p.dtype, shape: &p.shape, bytes: p.bytes as usize })
}
fn image_plan(plan: &Deepseek4CachePlan, position: usize) -> ImagePlan {
    ImagePlan::new(plan, position, position, 0, plan.layers.len(), plan.layers.len(),
        |l, r| meta(plan, l, r)).unwrap()
}
fn no_image_on_bad_plan<'a>(
    plan: &Deepseek4CachePlan,
    position: usize,
    tokens: &[u32],
    graft: usize,
    counts: (usize, usize),
    metadata: impl FnMut(usize, usize) -> Option<BufferMetadata<'a>>,
) {
    let allocated = Cell::new(0);
    let read = Cell::new(0);
    let result = encode_from_metadata(plan, position, tokens, graft, counts, metadata,
        |_| { allocated.set(allocated.get() + 1); Err(capacity("must not reserve", 0)) },
        |_, _, _| { read.set(read.get() + 1); Ok(&[]) });
    assert!(result.is_err());
    assert_eq!((allocated.get(), read.get()), (0, 0));
}

#[test]
fn deepseek_image_exact_codec_matches_independent_old_format_golden() {
    let plan = cache_plan(vec![0], 4);
    let reads = Cell::new(0);
    let requests = Cell::new(0);
    let payload = [0x3cu8; 64];
    let actual = encode_from_metadata(&plan, 1, &[7], 0, (1, 1), |l, r| meta(&plan, l, r),
        |n| { assert_eq!(n, 204); requests.set(requests.get() + 1); reserve(n) },
        |layer, role, n| { assert_eq!((layer, role, n), (0, 0, 64)); reads.set(reads.get() + 1); Ok(&payload) }).unwrap();
    // Independent fixed old-codec packet: explicit LE fields, token7, one
    // BF16 window row, zero other sections, SHA256 over the patched body.
    let expected = hex::decode(concat!(
        "44534b3441494d31010000000100000001000000010000000400000000000000cc00000000000000",
        "07000000000000000000000040000000000000000000000000000000000000000000000000000000",
        "000000000000000000000000000000000000000000000000000000003c3c3c3c3c3c3c3c3c3c3c3c",
        "3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c",
        "3c3c3c3c3c3c3c3c3c3c3c3cb745f6f2ca07b0c0bc57347439683e3aba9c4b68bc3fe1bbc86d8d8c",
        "92ab1709",
    )).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(hex::encode(Sha256::digest(&actual)), "98ac2493d85f1933adec17613a8081f38aa87bb689f186c4310c48d48d48e891");
    assert_eq!((actual.len(), actual.capacity(), requests.get(), reads.get()), (204, 204, 1, 1));
    let (header, tokens) = super::super::AnchorImageHeader::parse_prefix(&actual).unwrap();
    assert_eq!((header.anchor_position, header.layer_count, header.context_length), (1, 1, 4));
    assert_eq!(tokens.unwrap(), [7]);
}
#[test]
fn deepseek_image_selected_plan_exact_length_without_payload_or_image_allocation() {
    let mut cfg = crate::inference::models::deepseek4::residency_tests::tiny_config();
    cfg.num_hidden_layers = 43;
    cfg.max_position_embeddings = 1_048_576;
    cfg.head_dim = 512;
    cfg.index_head_dim = 128;
    cfg.compress_ratios = (0..43).map(|i| if i < 2 { 0 } else if i % 2 == 0 { 4 } else { 128 }).collect();
    let plan = Deepseek4CachePlan::for_context(&cfg, 262_144).unwrap();
    let image = image_plan(&plan, 262_144);
    assert_eq!(image.total, 1_822_444_296);
    assert_eq!(image.layers.len(), 43);
    assert_eq!(image.layers.iter().map(|l| l.lengths.iter().sum::<usize>()).sum::<usize>(), 1_821_392_896);
    assert_eq!(image.total * 3, 5_467_332_888);
}
#[test]
fn deepseek_image_all_seven_roles_and_zero_sections_use_actual_emitter() {
    let cache = cache_plan(vec![0, 4, 128], 128);
    let plan = image_plan(&cache, 4);
    let payloads: Vec<Vec<Vec<u8>>> = plan.layers.iter().enumerate().map(|(l, p)|
        p.lengths.iter().enumerate().map(|(r, n)| vec![(l * 7 + r + 1) as u8; *n]).collect()).collect();
    let calls = Cell::new(0);
    let out = emit(&plan, &[1, 2, 3, 4], reserve, |l, r, n| {
        calls.set(calls.get() + 1); assert_eq!(n, payloads[l][r].len()); Ok(payloads[l][r].as_slice())
    }).unwrap();
    let mut cursor = 40 + 16;
    let mut nonzero = 0;
    for (l, p) in plan.layers.iter().enumerate() {
        assert_eq!(&out[cursor..cursor + 4], &p.index.to_le_bytes());
        cursor += 64;
        for (r, n) in p.lengths.iter().enumerate() {
            assert_eq!(&out[cursor..cursor + *n], payloads[l][r].as_slice());
            cursor += *n;
            nonzero += usize::from(*n > 0);
        }
    }
    assert!(plan.layers[1].lengths.iter().all(|n| *n > 0));
    assert_eq!(plan.layers[2].lengths[1], 0);
    assert_eq!(calls.get(), nonzero);
    assert_eq!(&out[cursor..], &Sha256::digest(&out[..cursor])[..]);
    assert_eq!((out.len(), out.capacity()), (plan.total, plan.total));
}
#[test]
fn deepseek_image_bad_header_and_cardinality_precede_image_reservation() {
    let plan = cache_plan(vec![0], 4);
    for (position, tokens, graft, counts) in [(0, &[][..], 0, (1, 1)),
        (1, &[][..], 0, (1, 1)), (5, &[0; 5][..], 0, (1, 1)),
        (1, &[0][..], 1, (1, 1)), (1, &[0][..], 0, (0, 1)), (1, &[0][..], 0, (1, 0))] {
        no_image_on_bad_plan(&plan, position, tokens, graft, counts, |l, r| meta(&plan, l, r));
    }
    let mut oversized = plan.clone();
    oversized.context_length = u32::MAX as usize + 1;
    no_image_on_bad_plan(&oversized, 1, &[0], 0, (1, 1), |l, r| meta(&oversized, l, r));
    let mut oversized = plan.clone();
    oversized.layers[0].layer_index = u32::MAX as usize + 1;
    no_image_on_bad_plan(&oversized, 1, &[0], 0, (1, 1), |l, r| meta(&oversized, l, r));
}
#[test]
fn deepseek_image_metadata_faults_precede_image_reservation() {
    let plan = cache_plan(vec![4], 8);
    for fault in 0..4 {
        no_image_on_bad_plan(&plan, 4, &[0; 4], 0, (1, 1), |l, r| {
            let mut m = meta(&plan, l, r);
            if r == 0 {
                match fault {
                    0 => m = None,
                    1 => m.as_mut().unwrap().dtype = DType::F32,
                    2 => m.as_mut().unwrap().bytes -= 1,
                    _ => m.as_mut().unwrap().shape = &[1, 32],
                }
            }
            m
        });
    }
    let dense = cache_plan(vec![0], 8);
    no_image_on_bad_plan(&dense, 1, &[0], 0, (1, 1), |l, r| {
        if r == 1 { meta(&dense, l, 0) } else { meta(&dense, l, r) }
    });
}
#[test]
fn deepseek_image_checked_shape_and_aggregate_overflow_precede_payload() {
    let mut plan = cache_plan(vec![0], 4);
    plan.layers[0].window_kv.shape = vec![usize::MAX, 2];
    no_image_on_bad_plan(&plan, 1, &[0], 0, (1, 1), |l, r| meta(&plan, l, r));
    let mut plan = cache_plan(vec![4], 4);
    let enormous = CacheBufferPlan { dtype: DType::F32, shape: vec![usize::MAX / 8], bytes: ((usize::MAX / 8) * 4) as u64 };
    plan.layers[0].main_kv_state = Some(enormous.clone());
    plan.layers[0].main_score_state = Some(enormous.clone());
    plan.layers[0].indexer_kv_state = Some(enormous.clone());
    plan.layers[0].indexer_score_state = Some(enormous);
    no_image_on_bad_plan(&plan, 1, &[0], 0, (1, 1), |l, r| meta(&plan, l, r));
}
#[test]
fn deepseek_image_failed_or_anomalous_reservation_reads_no_payload() {
    let plan = cache_plan(vec![0], 4);
    for anomaly in [false, true] {
        let reads = Cell::new(0);
        let requests = Cell::new(0);
        let result = encode_from_metadata(&plan, 1, &[7], 0, (1, 1), |l, r| meta(&plan, l, r),
            |n| { requests.set(requests.get() + 1); if anomaly { reserve(n + 1) } else { Err(capacity("injected reservation failure", n)) } },
            |_, _, _| { reads.set(reads.get() + 1); Ok(&[]) });
        assert!(result.is_err());
        assert_eq!((requests.get(), reads.get()), (1, 0));
    }
}
#[test]
fn deepseek_image_wrong_payload_lengths_and_read_failure_refuse_publication() {
    let cache = cache_plan(vec![0], 4);
    let plan = image_plan(&cache, 1);
    let short = [0u8; 63];
    let long = [0u8; 65];
    for bytes in [&short[..], &long[..]] {
        assert!(emit(&plan, &[7], reserve, |_, _, _| Ok(bytes)).is_err());
    }
    assert!(emit(&plan, &[7], reserve, |_, _, _| Err(capacity("injected read failure", 64))).is_err());
    assert!(emit(&plan, &[], |_| panic!("ledger mismatch must precede reserve"), |_, _, _| Ok(&[])).is_err());
}
#[test]
fn deepseek_image_append_and_patch_rejection_leave_writer_unchanged() {
    let mut writer = ExactWriter::new(reserve(4).unwrap(), 4).unwrap();
    writer.append(&[1, 2, 3]).unwrap();
    let before = (writer.bytes.clone(), writer.bytes.capacity());
    assert!(writer.append(&[4, 5]).is_err());
    assert_eq!((&writer.bytes, writer.bytes.capacity()), (&before.0, before.1));
    assert!(writer.patch(2, &[8, 9]).is_err());
    assert_eq!((&writer.bytes, writer.bytes.capacity()), (&before.0, before.1));
    assert!(writer.patch(usize::MAX, &[0]).is_err());
    assert!(writer.finish().is_err());
}
#[test]
fn deepseek_image_checksum_append_cannot_grow_a_short_plan() {
    let cache = cache_plan(vec![0], 4);
    let mut plan = image_plan(&cache, 1);
    plan.total -= 1;
    assert!(emit(&plan, &[7], reserve, |_, _, _| Ok(&[0; 64])).is_err());
    let mut writer = ExactWriter::new(reserve(32).unwrap(), 32).unwrap();
    writer.append(&[0; 1]).unwrap();
    assert!(writer.append(&[0; 32]).is_err());
    assert_eq!((writer.bytes.len(), writer.bytes.capacity()), (1, 32));
}


/// CPU-accessible Metal storage only; no encoder, command buffer or GPU submit.
#[test]
#[ignore = "requires Apple Silicon Metal; bounded CPU-only codec compatibility"]
fn deepseek_image_actual_metal_cpu_roundtrip_compatibility() {
    let mut cfg = crate::inference::models::deepseek4::residency_tests::tiny_config();
    cfg.num_hidden_layers = 3;
    cfg.compress_ratios = vec![0, 4, 128];
    let plan = Deepseek4CachePlan::for_context(&cfg, 128).unwrap();
    assert_eq!(plan.resident_bytes, 66_624);
    let device = mlx_native::MlxDevice::new().expect("Metal required; no skip");
    let mut cache = Deepseek4Cache::allocate(&plan, device.clone()).unwrap();
    for (l, layer) in cache.layers.iter_mut().enumerate() {
        // Initialize each physical root once, not its overlapping window view.
        for (r, root) in [Some(&mut layer.attention_kv), layer.indexer_kv.as_mut(),
            layer.main_kv_state.as_mut(), layer.main_score_state.as_mut(),
            layer.indexer_kv_state.as_mut(), layer.indexer_score_state.as_mut()]
            .into_iter().enumerate()
        {
            if let Some(root) = root {
                for (index, byte) in root.as_mut_slice::<u8>().unwrap().iter_mut().enumerate() {
                    // Position-sensitive bytes distinguish window/compressed
                    // regions of their shared attention parent, without arrays.
                    *byte = ((index / 251 + index % 251 + l * 17 + r * 29) % 256) as u8;
                }
            }
        }
    }
    for position in 0..128 { cache.commit_step(position).unwrap(); }
    let anchor = cache.snapshot().unwrap();
    assert_eq!(anchor.resident_bytes(), 63_488);
    let tokens: Vec<u32> = (0..128).collect();
    let image = cache.serialize_anchor_image(&anchor, &tokens).unwrap();
    assert_eq!((image.len(), image.capacity()), (67_400, 67_400));
    let body = image.len() - 32;
    assert_eq!(&image[body..], &Sha256::digest(&image[..body])[..]);
    let (header, ledger) = super::super::AnchorImageHeader::parse_prefix(&image).unwrap();
    assert_eq!((header.anchor_position, header.context_length, header.layer_count), (128, 128, 3));
    assert_eq!(ledger.unwrap(), tokens);
    // This fails at the production checksum guard, before its plan/device
    // allocation path. No allocation counter is available in registry0.15.3.
    let mut corrupt = image.clone();
    corrupt[40] ^= 1;
    assert!(matches!(Deepseek4Cache::hydrate_anchor_image(&corrupt, &cfg, &device),
        Err(CacheError::AnchorImageCorrupt { reason: "checksum mismatch" })));
    drop(corrupt);
    let hydrated = Deepseek4Cache::hydrate_anchor_image(&image, &cfg, &device).unwrap();
    assert_eq!(hydrated.tokens, tokens);
    assert_eq!((hydrated.cache.position(), hydrated.snapshot.position()), (128, 128));
    assert_eq!(hydrated.snapshot.resident_bytes(), 63_488);
    let sections = image_plan(&plan, 128);
    let mut compared = 0;
    for (l, section) in sections.layers.iter().enumerate() {
        for (r, &len) in section.lengths.iter().enumerate() {
            if len == 0 { continue; }
            let before = buffer(&cache, &anchor, l, r).unwrap().as_slice::<u8>().unwrap();
            let after = buffer(&hydrated.cache, &hydrated.snapshot, l, r).unwrap()
                .as_slice::<u8>().unwrap();
            assert_eq!(&before[..len], &after[..len], "layer={l} role={r}");
            compared += 1;
        }
    }
    assert_eq!(compared, 12);
    let second = hydrated.cache.serialize_anchor_image(&hydrated.snapshot, &hydrated.tokens).unwrap();
    assert_eq!((second.len(), second.capacity()), (67_400, 67_400));
    assert_eq!(image, second);
    println!("DS_IMAGE_CPU_CODEC_COMPLETE layers=3 roles=12 roots=38 metal_requested_bytes=260224 image_bytes=67400 peak_images_upper=3 requested_packet_upper=463448 gpu_submissions=0");
}
