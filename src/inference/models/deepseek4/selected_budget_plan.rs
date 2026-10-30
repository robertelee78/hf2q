//! SAME production catalog, storage validation, pure mapper and cache planner.
//! No Device/mmap/payload; whole requested roots, never file-page union credit.
use super::{
    cache::Deepseek4CachePlan,
    weights::{validate_tensor_catalog, TensorRole},
    Deepseek4Model,
};
use anyhow::{ensure, Context, Result};
use mlx_native::gguf::GgufFile;
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Minimum {
    pub weights: u64,
    pub four_caches_stub: u64,
    pub weight_roots: usize,
    pub cache_roots: usize,
}
pub(crate) fn derive(g: &GgufFile, page: usize, max: usize, file_bytes: u64) -> Result<Minimum> {
    ensure!(
        g.tensor_count() <= 4096,
        "Q5 catalog bound before plan allocation"
    );
    let cfg = Deepseek4Model::load_config_only(g)?;
    let specs = validate_tensor_catalog(g, &cfg)?;
    ensure!(specs.len() <= 4096, "Q5 catalog capacity");
    let plan = g.mapped_tensor_storage_plan_with_limits(page, max)?;
    ensure!(
        plan.segments().len() <= 4096 && plan.tensor_views().len() == specs.len(),
        "Q5 exact map/catalog coverage"
    );
    let mut weights = 0u64;
    let mut roots = plan.segments().len();
    for s in plan.segments() {
        ensure!(
            s.mapping_file_end() as u64 <= file_bytes,
            "Q5 mmap escapes SAMEFD"
        );
        weights = weights
            .checked_add(s.root_byte_len() as u64)
            .context("Q5 mapped sum overflow")?;
    }
    for spec in &specs {
        let info = g
            .tensor_info(&spec.name)
            .context("Q5 missing catalog tensor")?;
        super::residency::validate_storage_type(&spec.name, spec.role, info.ggml_type)?;
        if spec.role == TensorRole::ElementwiseF32 {
            let bytes = info
                .shape
                .iter()
                .try_fold(4u64, |n, &d| n.checked_mul(d as u64))
                .context("Q5 F32 extent overflow")?;
            weights = weights
                .checked_add(bytes)
                .context("Q5 weight sum overflow")?;
            roots = roots.checked_add(1).context("Q5 root count overflow")?;
        }
    }
    let full_plan = Deepseek4CachePlan::for_context(&cfg, 262144)?;
    let stub_plan = Deepseek4CachePlan::for_context(&cfg, 128)?;
    let root_count = |p: &Deepseek4CachePlan| -> usize {
        p.layers
            .iter()
            .map(|l| {
                1 + usize::from(l.indexer_kv.is_some())
                    + usize::from(l.main_kv_state.is_some())
                    + usize::from(l.main_score_state.is_some())
                    + usize::from(l.indexer_kv_state.is_some())
                    + usize::from(l.indexer_score_state.is_some())
            })
            .sum()
    };
    let cache_roots = root_count(&full_plan)
        .checked_mul(4)
        .and_then(|n| n.checked_add(root_count(&stub_plan)))
        .context("Q5 cache root count overflow")?;
    let full = full_plan.resident_bytes;
    let stub = stub_plan.resident_bytes;
    Ok(Minimum {
        weights,
        four_caches_stub: full
            .checked_mul(4)
            .and_then(|x| x.checked_add(stub))
            .context("Q5 cache sum overflow")?,
        weight_roots: roots,
        cache_roots,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_production_catalog_pure_mapper_and_cache_plan_are_used_without_device() {
        let mut cfg = super::super::residency_tests::tiny_config();
        cfg.max_position_embeddings = 262144;
        let (_dir, g) = super::super::residency_tests::open_fixture(&cfg, false, false);
        let bytes = std::fs::metadata(_dir.path().join("weights.gguf"))
            .unwrap()
            .len();
        let m = derive(&g, 16384, 86586540032, bytes).unwrap();
        let map = g
            .mapped_tensor_storage_plan_with_limits(16384, 86586540032)
            .unwrap();
        assert!(
            m.weights
                >= map
                    .segments()
                    .iter()
                    .map(|s| s.root_byte_len() as u64)
                    .sum::<u64>()
        );
        assert_eq!(
            m.four_caches_stub,
            Deepseek4CachePlan::for_context(&cfg, 262144)
                .unwrap()
                .resident_bytes
                * 4
                + Deepseek4CachePlan::for_context(&cfg, 128)
                    .unwrap()
                    .resident_bytes
        );
        assert!(derive(&g, 16384, 86586540032, 1).is_err());
    }
}
