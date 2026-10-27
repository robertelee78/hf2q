//! Actual cache/image planners and emitter, without Metal allocation or hydration.
use super::*;

#[test]
fn anchor_c1_deepseek_window_compressed_indexer_roles_and_native_tail_coordinates() {
    let cache = cache_plan(vec![0, 4, 128], 256);
    for position in [1, 4, 127, 128, 129, 256] {
        let image = image_plan(&cache, position);
        for (layer, spec) in cache.layers.iter().enumerate() {
            let lengths = &image.layers[layer].lengths;
            assert_eq!(lengths[0], position.min(128) * spec.window_kv.shape[1] * 2);
            let (write, valid) =
                crate::inference::models::deepseek4::cache_buffers::completed_group_step(
                    position,
                    spec.compress_ratio,
                );
            if spec.compress_ratio == 0 {
                assert_eq!((write, valid), (None, 0));
                assert!(lengths[1..].iter().all(|n| *n == 0));
            } else {
                assert_eq!(valid, position / spec.compress_ratio as usize);
                assert_eq!(lengths[1], valid * spec.attention_kv.shape[1] * 2);
                assert!(lengths[3] > 0 && lengths[4] > 0);
                assert_eq!(
                    write.is_some(),
                    position % spec.compress_ratio as usize == 0
                );
                assert_eq!(spec.indexer_kv.is_some(), spec.compress_ratio == 4);
                assert_eq!(lengths[2] > 0, spec.compress_ratio == 4 && valid > 0);
                assert_eq!(lengths[5] > 0, spec.compress_ratio == 4);
                assert_eq!(lengths[6] > 0, spec.compress_ratio == 4);
            }
            assert_eq!(
                spec.attention_kv.bytes,
                spec.window_kv.bytes + spec.compressed_kv.as_ref().map_or(0, |p| p.bytes)
            );
        }
    }
    // The native eight-token recovery suffix is a separate engine integration
    // obligation, not proved by this cache/image layout fixture.
}

#[test]
fn anchor_c1_deepseek_late_sidecar_refuses_before_source_read_or_destination_reserve() {
    let plan = cache_plan(vec![0, 4, 128], 256);
    // Corrupt the final layer's score-state metadata after all earlier roles.
    no_image_on_bad_plan(&plan, 129, &vec![7; 129], 0, (3, 3), |l, r| {
        if l == 2 && r == 4 {
            None
        } else {
            meta(&plan, l, r)
        }
    });
    no_image_on_bad_plan(&plan, 129, &vec![7; 129], 1, (3, 3), |l, r| {
        meta(&plan, l, r)
    });
    let image = image_plan(&plan, 129);
    let bytes: Vec<Vec<Vec<u8>>> = image
        .layers
        .iter()
        .enumerate()
        .map(|(l, p)| {
            p.lengths
                .iter()
                .enumerate()
                .map(|(r, n)| vec![(l * 7 + r + 1) as u8; *n])
                .collect()
        })
        .collect();
    let tokens: Vec<u32> = (0..129).map(|n| n + 100).collect();
    let encoded = emit(&image, &tokens, reserve, |l, r, n| {
        assert_eq!(n, bytes[l][r].len());
        Ok(bytes[l][r].as_slice())
    })
    .unwrap();
    let (header, ledger) = super::super::super::AnchorImageHeader::parse_prefix(&encoded).unwrap();
    assert_eq!(
        (
            header.anchor_position,
            header.context_length,
            header.layer_count
        ),
        (129, 256, 3)
    );
    assert_eq!(ledger.unwrap(), tokens);
    assert!(encoded.len() < 1024 * 1024, "bounded actual codec fixture");
}
