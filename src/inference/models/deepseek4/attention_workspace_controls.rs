use super::*;
#[test]
fn selected_three_role_extent_and_savings_cover_dense_compressed_profiles() {
    for ratio in [0, 4, 128] {
        for rows in [1, 65, 512, 2048] {
            let p = Plan::new(rows, 4 * 4096, 64, 512).unwrap();
            assert_eq!(p.root_elements, rows * 32768);
            assert_eq!(
                p.separate_bytes().unwrap() - p.root_elements * 4,
                rows * 196608,
                "ratio {ratio}"
            );
        }
    }
    assert_eq!(
        Plan::new(2048, 16384, 64, 512)
            .unwrap()
            .separate_bytes()
            .unwrap(),
        640 * 1024 * 1024
    );
    assert_eq!(
        Plan::new(512, 16384, 64, 512).unwrap().root_elements * 4,
        64 * 1024 * 1024
    );
}
#[test]
fn workspace_rejects_empty_or_overflow_geometry_and_preserves_larger_norm() {
    for (r, hc, h, d) in [
        (0, 16, 4, 8),
        (1, 0, 4, 8),
        (1, 16, 0, 8),
        (1, 16, 4, 0),
        (usize::MAX, 16, 4, 8),
        (1, 16, usize::MAX, 8),
    ] {
        assert!(Plan::new(r, hc, h, d).is_err());
    }
    let p = Plan::new(3, 128, 2, 16).unwrap();
    assert_eq!(p.root_elements, 384);
    assert!(p.norm_elements <= p.root_elements && p.query_elements <= p.root_elements);
}
