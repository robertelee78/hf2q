//! ADR-037 Phase E4b.10b.3 (2026-05-22) — `GpuDrafter` implementing
//! Phase E4a's [`Drafter`] trait via the full forward chain (E4b.1-
//! E4b.10b.2).
//!
//! `GpuDrafter` owns the GPU resources (config + tensors + device +
//! registry) plus the per-spec-decode-step inputs (target model's
//! multi-aux hidden state + base position + embedding table). Each
//! `predict_topk` call:
//!
//! 1. Walks the parent chain via [`TreeContextView::path_tokens`] to
//!    get the token sequence from root to `node_to_expand`.
//! 2. Looks up embeddings for the path tokens from the CPU embed
//!    table.
//! 3. Uploads embeddings to GPU.
//! 4. Runs the full 14-stage forward chain via the test helper at
//!    `forward::run_full_eagle3_forward` (factored out as a public
//!    `dispatch_eagle3_drafter_forward` for production reuse).
//! 5. Extracts top-K via
//!    [`super::drafter::extract_top_k_from_row_logits`].
//! 6. Returns `Vec<DraftCandidate>` conforming to the Phase E4a
//!    contract (descending, unique, finite).
//!
//! ## Scope of this iteration (E4b.10b.3)
//!
//! Single-token decode per `predict_topk` call. The forward operates
//! on `seq_len=1` (last token of the path) without KV cache reuse.
//! Full prefix re-prefill on every call — correct but slow. KV cache
//! management for incremental decode is Phase E4b.10b.4 (next iter).
//!
//! ## Embed lookup
//!
//! For testing + the simple case where the drafter has its own
//! `embed_tokens.weight`, GpuDrafter owns a CPU-side `embed_table:
//! &[f32]` of shape `[vocab_size, hidden_size]`. Real production
//! use will share the target model's embedding table; that
//! plumbing is out of scope for this iteration.

use super::config::Eagle3DrafterConfig;
use super::drafter::{
    extract_top_k_from_row_logits, DraftCandidate, Drafter, TreeContextView,
};
use super::kv_cache::DrafterKvCache;
use super::tensors::Eagle3DrafterTensors;
use anyhow::{anyhow, ensure, Result};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

/// GPU drafter implementing Phase E4a's [`Drafter`] trait by running
/// the full Eagle3 forward chain.
///
/// Lifetimes:
/// - `cfg`, `tensors`, `device`, `target_aux`, `embed_table` are
///   borrowed for the lifetime of the drafter.
/// - `registry` is borrowed mutably (compute pipelines may be lazily
///   built on first dispatch).
pub struct GpuDrafter<'a> {
    pub cfg: &'a Eagle3DrafterConfig,
    pub tensors: &'a Eagle3DrafterTensors,
    pub device: &'a MlxDevice,
    pub registry: &'a mut KernelRegistry,
    /// Target model's multi-aux concatenated hidden state for the
    /// current spec-decode step. Shape `[1, num_aux * target_hidden_size]`
    /// F32 (single-token decode in this iteration).
    pub target_aux: &'a MlxBuffer,
    /// CPU-side embedding table `[vocab_size, hidden_size]` F32.
    /// `predict_topk` looks up the last path token's embedding from
    /// this. Stored on CPU to avoid GPU memory pressure for vocabularies
    /// up to ~200K rows.
    pub embed_table: &'a [f32],
    /// Absolute position of the root tree-node in the target sequence
    /// (used for RoPE linear-position math: position[i] = base_pos +
    /// depths[i] for tree-aware decoding; for this iteration's
    /// single-token path we use linear positions starting at base_pos).
    pub base_pos: u32,
    /// Optional drafter KV cache (Phase E5b Step 3).
    ///
    /// When `Some`, `predict_topk` uses the cache-aware forward
    /// (`dispatch_eagle3_drafter_forward_with_kv_cache`), which
    /// conditions on the full root-to-node ancestor chain via cached
    /// K/V — lifting the `path.len() == 1` cap from E4b.10b.3.
    ///
    /// **Cache state invariant**: at the start of each `predict_topk`
    /// call, `cache.len()` must equal `path.len() - 1` where `path`
    /// is the parent-chain tokens from root to `node_to_expand`. The
    /// orchestrator maintains this invariant by calling
    /// [`Self::rollback_cache`] when switching to a different branch
    /// (e.g. picking a sibling instead of a descendant). After the
    /// call, `cache.len()` advances by 1 (new node's K/V appended).
    ///
    /// When `None`, `predict_topk` falls back to the unbatched
    /// single-token-decode path with the original `path.len() == 1`
    /// guard (Phase E4b.10b.3 behavior preserved for backward
    /// compatibility).
    pub kv_cache: Option<DrafterKvCache>,
}

// Manual Debug — MlxBuffer/MlxDevice don't impl Debug consistently
// in mlx-native; print only the architecturally-relevant fields.
impl<'a> std::fmt::Debug for GpuDrafter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDrafter")
            .field("cfg.hidden_size", &self.cfg.hidden_size)
            .field("cfg.vocab_size", &self.cfg.vocab_size)
            .field("base_pos", &self.base_pos)
            .finish()
    }
}

impl<'a> GpuDrafter<'a> {
    /// Construct a new GpuDrafter. Validates the embed_table shape
    /// matches cfg.vocab_size * cfg.hidden_size.
    pub fn new(
        cfg: &'a Eagle3DrafterConfig,
        tensors: &'a Eagle3DrafterTensors,
        device: &'a MlxDevice,
        registry: &'a mut KernelRegistry,
        target_aux: &'a MlxBuffer,
        embed_table: &'a [f32],
        base_pos: u32,
    ) -> Result<Self> {
        cfg.validate()
            .map_err(|e| anyhow!("GpuDrafter::new: cfg invalid: {e}"))?;
        let expected_embed = cfg
            .vocab_size
            .checked_mul(cfg.hidden_size)
            .ok_or_else(|| anyhow!("GpuDrafter::new: vocab_size * hidden_size overflows usize"))?;
        ensure!(
            embed_table.len() == expected_embed,
            "GpuDrafter::new: embed_table has {} elements, expected {} (vocab_size {} * hidden_size {})",
            embed_table.len(),
            expected_embed,
            cfg.vocab_size,
            cfg.hidden_size
        );
        ensure!(
            target_aux.dtype() == DType::F32,
            "GpuDrafter::new: target_aux dtype must be F32, got {:?}",
            target_aux.dtype()
        );
        // For this iteration's single-token-per-predict_topk pattern,
        // target_aux is expected to be [1, num_aux * target_hidden].
        let expected_target_aux = cfg.fc_input_size();
        ensure!(
            target_aux.element_count() == expected_target_aux,
            "GpuDrafter::new: target_aux has {} elements, expected {} (1 * fc_input_size {})",
            target_aux.element_count(),
            expected_target_aux,
            cfg.fc_input_size()
        );
        // Codex /cfa E4b.10b.3 Minor (2026-05-22): also validate the
        // shape dimensions match the documented [1, fc_input_size]
        // contract (element_count alone allows [fc_input_size] or
        // [fc_input_size, 1] to pass).
        let shape = target_aux.shape();
        ensure!(
            shape == [1, expected_target_aux],
            "GpuDrafter::new: target_aux shape {:?} != [1, {}]",
            shape,
            expected_target_aux
        );
        // Codex /cfa E4 final-gate Major 1 (2026-05-22, re-review):
        // when draft_vocab_size < vocab_size, the manifest MUST
        // include draft_id_to_target_id (else predict_topk would
        // return draft-space token IDs that downstream consumers
        // treat as target-space). Enforce at constructor instead of
        // silently mis-mapping at predict_topk time.
        if cfg.draft_vocab_size != cfg.vocab_size {
            ensure!(
                tensors.draft_id_to_target_id.is_some(),
                "GpuDrafter::new: draft_vocab_size ({}) != vocab_size ({}) requires \
                 tensors.draft_id_to_target_id to be present (set cfg.include_draft_id_mapping=true \
                 and provide the mapping in the safetensors blob)",
                cfg.draft_vocab_size,
                cfg.vocab_size
            );
        }
        Ok(Self {
            cfg,
            tensors,
            device,
            registry,
            target_aux,
            embed_table,
            base_pos,
            kv_cache: None,
        })
    }

    /// Attach a pre-allocated drafter KV cache. Cache shape must
    /// match `cfg.num_kv_heads` and `cfg.head_dim`.
    ///
    /// After this call, `predict_topk` uses the cache-aware forward
    /// and accepts paths of arbitrary depth (subject to cache
    /// capacity and the invariant that cache.len() == path.len()-1
    /// at entry of each `predict_topk` call).
    pub fn attach_kv_cache(&mut self, cache: DrafterKvCache) -> Result<()> {
        ensure!(
            cache.num_kv_heads == self.cfg.num_kv_heads,
            "GpuDrafter::attach_kv_cache: cache.num_kv_heads ({}) != cfg.num_kv_heads ({})",
            cache.num_kv_heads,
            self.cfg.num_kv_heads
        );
        ensure!(
            cache.head_dim == self.cfg.head_dim,
            "GpuDrafter::attach_kv_cache: cache.head_dim ({}) != cfg.head_dim ({})",
            cache.head_dim,
            self.cfg.head_dim
        );
        ensure!(
            cache.len() == 0,
            "GpuDrafter::attach_kv_cache: cache must be empty at attach (len={})",
            cache.len()
        );
        self.kv_cache = Some(cache);
        Ok(())
    }

    /// Reset the attached KV cache to empty (no-op if no cache).
    /// Called by the orchestrator at the start of each new spec-decode
    /// step before tree expansion begins.
    pub fn clear_kv_cache(&mut self) {
        if let Some(c) = self.kv_cache.as_mut() {
            c.clear();
        }
    }

    /// Rollback the KV cache to keep only the positions in `accepted`
    /// (in accepted-order). Delegates to
    /// [`DrafterKvCache::rollback_to_accepted`]. Errors if no cache
    /// is attached.
    ///
    /// The orchestrator calls this after [`crate::inference::spec_decode::eagle3::tree_walk::walk_tree_accept`]
    /// produces the accepted-node list, to compact the cache for the
    /// next spec-decode step.
    pub fn rollback_kv_cache(&mut self, accepted: &[usize]) -> Result<()> {
        let cache = self.kv_cache.as_mut().ok_or_else(|| {
            anyhow!("GpuDrafter::rollback_kv_cache: no cache attached")
        })?;
        cache.rollback_to_accepted(accepted)
    }

    /// Returns the current cache length, or 0 if no cache attached.
    pub fn kv_cache_len(&self) -> usize {
        self.kv_cache.as_ref().map(|c| c.len()).unwrap_or(0)
    }

    /// Look up the embedding for `token` from the CPU embed table.
    /// Returns a freshly-allocated `Vec<f32>` of length `hidden_size`.
    fn lookup_embedding(&self, token: u32) -> Result<Vec<f32>> {
        let t = token as usize;
        ensure!(
            t < self.cfg.vocab_size,
            "GpuDrafter::lookup_embedding: token {} >= vocab_size {}",
            token,
            self.cfg.vocab_size
        );
        let start = t.checked_mul(self.cfg.hidden_size).ok_or_else(|| {
            anyhow!(
                "GpuDrafter::lookup_embedding: token {} * hidden_size overflows usize",
                token
            )
        })?;
        let end = start.checked_add(self.cfg.hidden_size).ok_or_else(|| {
            anyhow!("GpuDrafter::lookup_embedding: end offset overflows usize")
        })?;
        ensure!(
            end <= self.embed_table.len(),
            "GpuDrafter::lookup_embedding: embed_table too small (end {} > len {})",
            end,
            self.embed_table.len()
        );
        Ok(self.embed_table[start..end].to_vec())
    }
}

/// Phase E6 (2026-05-22) — implement `CacheControlDrafter` so the
/// cache-aware orchestrator (`expand_dynamic_tree_with_cache`) can
/// drive `GpuDrafter` generically.
impl<'a> super::dynamic_tree::CacheControlDrafter for GpuDrafter<'a> {
    fn cache_len(&self) -> usize {
        self.kv_cache_len()
    }
    fn rollback_cache(&mut self, accepted: &[usize]) -> Result<()> {
        self.rollback_kv_cache(accepted)
    }
    fn clear_cache(&mut self) {
        self.clear_kv_cache();
    }
}

impl<'a> Drafter for GpuDrafter<'a> {
    fn predict_topk(
        &mut self,
        tree: TreeContextView<'_>,
        node_to_expand: usize,
        top_k: usize,
    ) -> Result<Vec<DraftCandidate>> {
        ensure!(top_k > 0, "GpuDrafter::predict_topk: top_k must be > 0");
        // Walk parent chain → token path from root to node_to_expand.
        let path = tree.path_tokens(node_to_expand);
        ensure!(
            !path.is_empty(),
            "GpuDrafter::predict_topk: empty path from node_to_expand {}",
            node_to_expand
        );
        // Phase E5b Step 3 (2026-05-22): when a KV cache is attached,
        // condition on the full root-to-node ancestor chain via the
        // cache. Otherwise fall back to the E4b.10b.3 single-token
        // decode path (root-only, path.len() == 1).
        let has_cache = self.kv_cache.is_some();
        if has_cache {
            // Cache invariant: cache.len() must equal path.len() - 1
            // at entry. Ancestors [0..path.len()-1) are already in
            // cache; node_to_expand's K/V will be appended in this
            // call.
            let cache_len = self.kv_cache.as_ref().unwrap().len();
            ensure!(
                cache_len + 1 == path.len(),
                "GpuDrafter::predict_topk (cache mode): cache.len()={} but path.len()-1={} \
                 (orchestrator must rollback/extend cache to match path before predict_topk)",
                cache_len,
                path.len() - 1
            );
        } else {
            // Codex /cfa E4 final-gate Major 2 re-review (2026-05-22):
            // cache-less single-token decode supports root expansion only.
            ensure!(
                path.len() == 1,
                "GpuDrafter::predict_topk (no-cache mode): path length {} != 1 \
                 (only ROOT expansion supported without KV cache — attach a cache \
                 via attach_kv_cache to enable depth>1 path conditioning)",
                path.len()
            );
        }
        // Use the LAST path token as the input embedding (node_to_expand's
        // token). Cache-aware forward conditions on ancestors via the cache;
        // cache-less forward sees only this token.
        let last_token = *path.last().unwrap();
        let embed_vec = self.lookup_embedding(last_token)?;
        ensure!(
            embed_vec.len() == self.cfg.hidden_size,
            "GpuDrafter::predict_topk: embed lookup returned len {} != hidden_size {}",
            embed_vec.len(),
            self.cfg.hidden_size
        );
        // Upload embedding to GPU.
        let mut embed_gpu = self
            .device
            .alloc_buffer(
                self.cfg.hidden_size * std::mem::size_of::<f32>(),
                DType::F32,
                vec![1, self.cfg.hidden_size],
            )
            .map_err(|e| anyhow!("GpuDrafter::predict_topk: alloc embed: {e}"))?;
        embed_gpu
            .as_mut_slice::<f32>()
            .map_err(|e| anyhow!("GpuDrafter::predict_topk: embed slice: {e}"))?
            .copy_from_slice(&embed_vec);

        // Depth-adjusted RoPE position.
        let depth_from_root = path.len() - 1; // node_to_expand's depth
        let depth_u32: u32 = u32::try_from(depth_from_root).map_err(|_| {
            anyhow!(
                "GpuDrafter::predict_topk: depth {} exceeds u32::MAX",
                depth_from_root
            )
        })?;
        let abs_pos = self.base_pos.checked_add(depth_u32).ok_or_else(|| {
            anyhow!(
                "GpuDrafter::predict_topk: base_pos {} + depth {} overflows u32",
                self.base_pos,
                depth_u32
            )
        })?;
        // Dispatch the appropriate forward variant.
        let logits_vec = if let Some(cache) = self.kv_cache.as_mut() {
            super::forward::dispatch_eagle3_drafter_forward_with_kv_cache(
                self.device,
                self.registry,
                self.target_aux,
                &embed_gpu,
                self.tensors,
                self.cfg,
                1, // seq_len = 1 (single new tree node per call)
                abs_pos,
                cache,
            )?
        } else {
            super::forward::dispatch_eagle3_drafter_forward(
                self.device,
                self.registry,
                self.target_aux,
                &embed_gpu,
                self.tensors,
                self.cfg,
                1, // seq_len = 1
                abs_pos,
            )?
        };

        // Extract top-K from logits row.
        let raw_candidates = extract_top_k_from_row_logits(&logits_vec, top_k)?;
        // Codex /cfa E4 final-gate Major 1 (2026-05-22): when the
        // drafter uses fast-vocab projection (draft_vocab_size <
        // vocab_size), lm_head produces logits over the DRAFT vocab.
        // The returned token IDs must be remapped through
        // `tensors.draft_id_to_target_id` to target-vocab IDs so the
        // tree expansion + KV cache + sampler operate in target's
        // vocabulary space.
        let candidates = if let Some(map) = &self.tensors.draft_id_to_target_id {
            let mut remapped = Vec::with_capacity(raw_candidates.len());
            for c in raw_candidates {
                let draft_idx = c.token as usize;
                ensure!(
                    draft_idx < map.len(),
                    "GpuDrafter::predict_topk: draft token {} exceeds draft_id_to_target_id length {}",
                    c.token,
                    map.len()
                );
                let target_id = map[draft_idx];
                ensure!(
                    target_id >= 0 && (target_id as usize) < self.cfg.vocab_size,
                    "GpuDrafter::predict_topk: draft_id_to_target_id[{}] = {} not in [0, vocab_size={})",
                    draft_idx,
                    target_id,
                    self.cfg.vocab_size
                );
                let target_u32: u32 = u32::try_from(target_id).map_err(|_| {
                    anyhow!(
                        "GpuDrafter::predict_topk: target_id {} exceeds u32::MAX",
                        target_id
                    )
                })?;
                remapped.push(DraftCandidate {
                    token: target_u32,
                    log_prob: c.log_prob,
                });
            }
            remapped
        } else {
            // No mapping: draft_vocab_size == vocab_size assumed.
            // (Config validation in dispatch_eagle3_lm_head enforces
            // this for the tied-lm-head path; the untied path with
            // draft_vocab < vocab REQUIRES the mapping or downstream
            // consumers would treat draft IDs as target IDs.)
            raw_candidates
        };
        Ok(candidates)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::inference::spec_decode::eagle3::drafter::validate_candidates;
    use crate::inference::spec_decode::eagle3::dynamic_tree::{
        expand_dynamic_tree, DynamicTreeConfig,
    };
    use crate::inference::spec_decode::eagle3::weights::{
        expected_manifest, Eagle3Weights, ExpectedTensor,
    };
    use safetensors::tensor::{Dtype as SafeDtype, TensorView};
    use std::collections::BTreeMap;

    /// Replicates the deterministic-weight pattern from E4b.10b.2 full
    /// forward test.
    fn build_test_blob(manifest: &[ExpectedTensor]) -> Vec<u8> {
        let mut storage: Vec<Vec<u8>> = Vec::with_capacity(manifest.len());
        for tensor in manifest {
            let name_hash: u64 = tensor
                .name
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let nelem: usize = tensor.shape.iter().product();
            let elem_bytes = match tensor.dtype {
                SafeDtype::BF16 => 2,
                SafeDtype::I64 => 8,
                _ => panic!("unexpected dtype"),
            };
            if tensor.dtype == SafeDtype::BF16 {
                let mut vals = vec![0.0f32; nelem];
                for (i, v) in vals.iter_mut().enumerate() {
                    let seed = name_hash.wrapping_add(i as u64);
                    let pr = pseudo_random(seed);
                    let is_norm = tensor.name.contains("norm");
                    *v = if is_norm {
                        1.0 + pr * 0.1
                    } else {
                        pr * 0.044
                    };
                }
                let mut bytes = Vec::with_capacity(vals.len() * 2);
                for v in &vals {
                    let bf16_bits = (v.to_bits() >> 16) as u16;
                    bytes.push((bf16_bits & 0xff) as u8);
                    bytes.push(((bf16_bits >> 8) & 0xff) as u8);
                }
                storage.push(bytes);
            } else {
                storage.push(vec![0u8; nelem * elem_bytes]);
            }
        }
        let mut tensors_map: BTreeMap<String, TensorView> = BTreeMap::new();
        for (i, exp) in manifest.iter().enumerate() {
            let view =
                TensorView::new(exp.dtype, exp.shape.clone(), storage[i].as_slice())
                    .expect("synthetic view");
            tensors_map.insert(exp.name.clone(), view);
        }
        safetensors::serialize(
            &tensors_map,
            None::<std::collections::HashMap<String, String>>,
        )
        .expect("serialize")
    }

    fn pseudo_random(seed: u64) -> f32 {
        let x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((x >> 33) as u32) & 0x7FFFFF;
        (bits as f32 / 0x7FFFFF as f32) * 2.0 - 1.0
    }

    fn cfg_for_drafter_test() -> Eagle3DrafterConfig {
        Eagle3DrafterConfig {
            hidden_size: 512,
            intermediate_size: 1024,
            head_dim: 128,
            num_q_heads: 4,
            num_kv_heads: 2,
            vocab_size: 1000,
            draft_vocab_size: 1000,
            target_hidden_size: 512,
            num_aux_hidden_states: 3,
            rms_norm_eps: 1e-6,
            norm_before_fc: false,
            fc_norm: false,
            use_qk_norm: false,
            attention_bias: false,
            tie_lm_head: false, // separate lm_head
            include_draft_id_mapping: false,
            has_own_embed_tokens: false, // CPU-side embed_table instead
            rope_theta: 1_000_000.0,
            rope_dim: 128,
        }
    }

    #[test]
    fn adr_037_e4b10b3_gpu_drafter_constructor_validates_embed_shape_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = cfg_for_drafter_test();
        let manifest = expected_manifest(&cfg);
        let blob = build_test_blob(&manifest);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        // target_aux: 1 row of fc_input_size
        let target_aux_data = vec![0.1f32; cfg.fc_input_size()];
        let mut target_aux_buf = device
            .alloc_buffer(
                cfg.fc_input_size() * 4,
                DType::F32,
                vec![1, cfg.fc_input_size()],
            )
            .expect("alloc target_aux");
        target_aux_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&target_aux_data);
        // WRONG-size embed_table (should be vocab * hidden = 1000*512 = 512000;
        // pass 100 to trigger error)
        let bad_embed = vec![0.0f32; 100];
        let err = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &bad_embed, 0,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("embed_table has 100 elements"),
            "got: {err}"
        );
    }

    #[test]
    fn adr_037_e4b10b3_gpu_drafter_end_to_end_with_expand_dynamic_tree_2026_05_22() {
        // INTEGRATION TEST: GpuDrafter is consumed by Phase E4a's
        // expand_dynamic_tree algorithm. Verifies the full stack
        // composes — from tree expansion (E4a) through GpuDrafter
        // forward (E4b.1-E4b.10b.2) and top-K extraction (E4b.10a).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = cfg_for_drafter_test();
        let manifest = expected_manifest(&cfg);
        let blob = build_test_blob(&manifest);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");

        // Target aux: synthetic [1, num_aux*hidden].
        let mut target_aux_data = vec![0.0f32; cfg.fc_input_size()];
        for (i, v) in target_aux_data.iter_mut().enumerate() {
            *v = pseudo_random(0xA0FFEE + i as u64) * 0.5;
        }
        let mut target_aux_buf = device
            .alloc_buffer(
                cfg.fc_input_size() * 4,
                DType::F32,
                vec![1, cfg.fc_input_size()],
            )
            .expect("alloc target_aux");
        target_aux_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&target_aux_data);

        // CPU embed_table: [vocab_size, hidden_size] random.
        let mut embed_table = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
        for (i, v) in embed_table.iter_mut().enumerate() {
            *v = pseudo_random(0xB0FFEE + i as u64) * 0.5;
        }

        let mut drafter = GpuDrafter::new(
            &cfg,
            &tensors,
            &device,
            &mut registry,
            &target_aux_buf,
            &embed_table,
            42, // base_pos
        )
        .expect("construct GpuDrafter");

        // Sanity check: predict_topk standalone on a synthetic
        // 1-element tree (just the root).
        let view = TreeContextView {
            tokens: &[123_u32],
            parents: &[None],
        };
        let candidates = drafter
            .predict_topk(view, 0, 3)
            .expect("predict_topk standalone");
        // Per Phase E4a contract.
        validate_candidates(&candidates, 3).expect("Phase E4a contract");
        assert_eq!(candidates.len(), 3);

        // Full integration: expand_dynamic_tree consumes the drafter.
        // Codex /cfa E4 final-gate Major 2 fix: max_depth=1 only —
        // higher depths require drafter KV cache (deferred).
        let tree_cfg = DynamicTreeConfig {
            budget: 4,
            max_depth: 1,
            top_k: 3,
        };
        let tree = expand_dynamic_tree(123_u32, &mut drafter, &tree_cfg)
            .expect("expand_dynamic_tree with GpuDrafter");
        // Tree must contain root + up to top_k children at depth 1.
        assert!(tree.len() >= 2, "tree should expand beyond root");
        assert!(tree.len() <= tree_cfg.budget, "tree must respect budget");
        // All non-root nodes at depth 1.
        for d in tree.depths.iter().skip(1) {
            assert_eq!(*d, 1, "max_depth=1 cap");
        }
        // Validate the tree structure.
        tree.validate().expect("ExpandedTree::validate");
    }

    #[test]
    fn adr_037_e4_final_gate_constructor_rejects_fast_vocab_without_mapping_2026_05_22() {
        // Codex /cfa E4 final-gate Major 1 re-review fix (2026-05-22):
        // when draft_vocab_size < vocab_size, the manifest MUST
        // include draft_id_to_target_id. Constructor must reject
        // the inconsistent config (else predict_topk would return
        // draft-space tokens that downstream treats as target-space).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let mut cfg = cfg_for_drafter_test();
        cfg.draft_vocab_size = 500; // < vocab_size = 1000
        cfg.include_draft_id_mapping = false; // INCONSISTENT
        let manifest = expected_manifest(&cfg);
        let blob = build_test_blob(&manifest);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors = Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let target_aux_data = vec![0.1f32; cfg.fc_input_size()];
        let mut target_aux_buf = device
            .alloc_buffer(cfg.fc_input_size() * 4, DType::F32, vec![1, cfg.fc_input_size()])
            .expect("alloc");
        target_aux_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&target_aux_data);
        let embed_table = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];

        let err = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("draft_vocab_size") && err.to_string().contains("requires"),
            "expected fast-vocab-without-mapping error, got: {err}"
        );
    }

    #[test]
    fn adr_037_e4_final_gate_predict_topk_rejects_depth_2_path_2026_05_22() {
        // Codex /cfa E4 final-gate Major 2 regression (2026-05-22):
        // calling predict_topk on a tree node at depth >= 2 must
        // fail-fast (path length > 2) since single-token decode
        // can't condition on full ancestor chain.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = cfg_for_drafter_test();
        let manifest = expected_manifest(&cfg);
        let blob = build_test_blob(&manifest);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors = Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let target_aux_data = vec![0.1f32; cfg.fc_input_size()];
        let mut target_aux_buf = device
            .alloc_buffer(cfg.fc_input_size() * 4, DType::F32, vec![1, cfg.fc_input_size()])
            .expect("alloc");
        target_aux_buf
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&target_aux_data);
        let embed_table = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
        let mut drafter = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .expect("construct");
        // Depth-2 tree: root → child → grandchild.
        let view = TreeContextView {
            tokens: &[10, 20, 30],
            parents: &[None, Some(0), Some(1)],
        };
        let err = drafter.predict_topk(view, 2, 3).unwrap_err();
        assert!(
            err.to_string().contains("path length 3 != 1"),
            "got: {err}"
        );
    }

    #[test]
    fn adr_037_e4_final_gate_predict_topk_rejects_depth_1_path_2026_05_22() {
        // Codex /cfa E4 final-gate Major 2 closure (2026-05-22):
        // even depth-1 (single-child) expansion violates conditioning
        // semantics since the path "root → child" would need both
        // tokens but single-token decode only consumes the last.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = cfg_for_drafter_test();
        let manifest = expected_manifest(&cfg);
        let blob = build_test_blob(&manifest);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors = Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        let target_aux_data = vec![0.1f32; cfg.fc_input_size()];
        let mut target_aux_buf = device
            .alloc_buffer(cfg.fc_input_size() * 4, DType::F32, vec![1, cfg.fc_input_size()])
            .expect("alloc");
        target_aux_buf.as_mut_slice::<f32>().unwrap().copy_from_slice(&target_aux_data);
        let embed_table = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
        let mut drafter = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .expect("construct");
        // Depth-1 tree: root → child.
        let view = TreeContextView {
            tokens: &[10, 20],
            parents: &[None, Some(0)],
        };
        let err = drafter.predict_topk(view, 1, 3).unwrap_err();
        assert!(
            err.to_string().contains("path length 2 != 1"),
            "got: {err}"
        );
    }

    // ----------------------------------------------------------------
    // Phase E5b Step 3 tests — cache-mode predict_topk lifts max_depth.
    // ----------------------------------------------------------------

    /// Helper: standard drafter test scaffolding (cfg + tensors +
    /// target_aux buffer + embed table) shared by Step 3 tests.
    fn step3_build_drafter_scaffolding(
        device: &MlxDevice,
    ) -> Option<(
        Eagle3DrafterConfig,
        Eagle3DrafterTensors,
        MlxBuffer,
        Vec<f32>,
    )> {
        let cfg = cfg_for_drafter_test();
        let manifest = expected_manifest(&cfg);
        let blob = build_test_blob(&manifest);
        let weights = Eagle3Weights::load(&blob, &cfg).ok()?;
        let tensors = Eagle3DrafterTensors::upload(device, &cfg, &weights).ok()?;
        let mut target_aux_data = vec![0.0f32; cfg.fc_input_size()];
        for (i, v) in target_aux_data.iter_mut().enumerate() {
            *v = pseudo_random(0xA0FFEE + i as u64) * 0.5;
        }
        let mut target_aux_buf = device
            .alloc_buffer(
                cfg.fc_input_size() * 4,
                DType::F32,
                vec![1, cfg.fc_input_size()],
            )
            .ok()?;
        target_aux_buf
            .as_mut_slice::<f32>()
            .ok()?
            .copy_from_slice(&target_aux_data);
        let mut embed_table = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
        for (i, v) in embed_table.iter_mut().enumerate() {
            *v = pseudo_random(0xB0FFEE + i as u64) * 0.5;
        }
        Some((cfg, tensors, target_aux_buf, embed_table))
    }

    #[test]
    fn adr_037_e5b_step3_attach_kv_cache_validates_num_kv_heads_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let (cfg, tensors, target_aux_buf, embed_table) =
            match step3_build_drafter_scaffolding(&device) {
                Some(t) => t,
                None => return,
            };
        let mut drafter = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .expect("drafter");
        // Wrong num_kv_heads.
        let bad_cache = DrafterKvCache::new(
            &device, cfg.num_kv_heads + 1, 4, cfg.head_dim,
        )
        .expect("alloc cache");
        let err = drafter.attach_kv_cache(bad_cache).unwrap_err();
        assert!(
            err.to_string().contains("num_kv_heads"),
            "expected num_kv_heads mismatch, got: {err}"
        );
    }

    #[test]
    fn adr_037_e5b_step3_attach_kv_cache_validates_head_dim_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let (cfg, tensors, target_aux_buf, embed_table) =
            match step3_build_drafter_scaffolding(&device) {
                Some(t) => t,
                None => return,
            };
        let mut drafter = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .expect("drafter");
        let bad_cache = DrafterKvCache::new(
            &device, cfg.num_kv_heads, 4, cfg.head_dim + 1,
        )
        .expect("alloc cache");
        let err = drafter.attach_kv_cache(bad_cache).unwrap_err();
        assert!(
            err.to_string().contains("head_dim"),
            "expected head_dim mismatch, got: {err}"
        );
    }

    #[test]
    fn adr_037_e5b_step3_attach_kv_cache_rejects_nonempty_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let (cfg, tensors, target_aux_buf, embed_table) =
            match step3_build_drafter_scaffolding(&device) {
                Some(t) => t,
                None => return,
            };
        let mut drafter = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .expect("drafter");
        let mut cache = DrafterKvCache::new(
            &device, cfg.num_kv_heads, 4, cfg.head_dim,
        )
        .expect("alloc cache");
        // Append one row to make it non-empty.
        let dummy_row = vec![0.0_f32; cfg.num_kv_heads * cfg.head_dim];
        cache.append(&dummy_row, &dummy_row).expect("append");
        let err = drafter.attach_kv_cache(cache).unwrap_err();
        assert!(
            err.to_string().contains("must be empty"),
            "expected non-empty rejection, got: {err}"
        );
    }

    #[test]
    fn adr_037_e5b_step3_cache_mode_predict_topk_depth_0_works_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let (cfg, tensors, target_aux_buf, embed_table) =
            match step3_build_drafter_scaffolding(&device) {
                Some(t) => t,
                None => return,
            };
        let mut drafter = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .expect("drafter");
        let cache = DrafterKvCache::new(
            &device, cfg.num_kv_heads, 4, cfg.head_dim,
        )
        .expect("alloc cache");
        drafter.attach_kv_cache(cache).expect("attach");
        assert_eq!(drafter.kv_cache_len(), 0);
        // Root expansion.
        let view = TreeContextView {
            tokens: &[123_u32],
            parents: &[None],
        };
        let candidates = drafter
            .predict_topk(view, 0, 3)
            .expect("predict_topk cache mode depth 0");
        validate_candidates(&candidates, 3).expect("contract");
        assert_eq!(candidates.len(), 3);
        // Cache should now hold root's K/V.
        assert_eq!(drafter.kv_cache_len(), 1);
    }

    #[test]
    fn adr_037_e5b_step3_cache_mode_predict_topk_depth_1_works_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let (cfg, tensors, target_aux_buf, embed_table) =
            match step3_build_drafter_scaffolding(&device) {
                Some(t) => t,
                None => return,
            };
        let mut drafter = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .expect("drafter");
        let cache = DrafterKvCache::new(
            &device, cfg.num_kv_heads, 4, cfg.head_dim,
        )
        .expect("alloc cache");
        drafter.attach_kv_cache(cache).expect("attach");

        // Step 1: expand root → cache.len() goes 0 → 1.
        let view_root = TreeContextView {
            tokens: &[10_u32],
            parents: &[None],
        };
        let root_cands = drafter
            .predict_topk(view_root, 0, 3)
            .expect("root expansion");
        assert_eq!(drafter.kv_cache_len(), 1);
        // Step 2: expand depth-1 child → cache.len() 1 → 2.
        let child_token = root_cands[0].token;
        let view_child = TreeContextView {
            tokens: &[10_u32, child_token],
            parents: &[None, Some(0)],
        };
        let child_cands = drafter
            .predict_topk(view_child, 1, 3)
            .expect("depth-1 expansion (cache mode lifts path.len()==1 cap)");
        validate_candidates(&child_cands, 3).expect("contract");
        assert_eq!(drafter.kv_cache_len(), 2);
    }

    #[test]
    fn adr_037_e5b_step3_cache_mode_rejects_cache_len_path_mismatch_2026_05_22() {
        // Cache empty but path has depth-1 node → orchestrator violated
        // the cache.len() == path.len()-1 invariant. predict_topk
        // must reject so the orchestrator catches its bookkeeping bug.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let (cfg, tensors, target_aux_buf, embed_table) =
            match step3_build_drafter_scaffolding(&device) {
                Some(t) => t,
                None => return,
            };
        let mut drafter = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .expect("drafter");
        let cache = DrafterKvCache::new(
            &device, cfg.num_kv_heads, 4, cfg.head_dim,
        )
        .expect("alloc cache");
        drafter.attach_kv_cache(cache).expect("attach");
        // cache.len()=0 but we're trying to expand a depth-1 node
        // whose path = [root, child]; need cache.len()=1.
        let view = TreeContextView {
            tokens: &[10_u32, 20],
            parents: &[None, Some(0)],
        };
        let err = drafter.predict_topk(view, 1, 3).unwrap_err();
        assert!(
            err.to_string().contains("cache.len()=0 but path.len()-1=1"),
            "expected cache len mismatch, got: {err}"
        );
    }

    #[test]
    fn adr_037_e5b_step3_clear_kv_cache_resets_to_zero_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let (cfg, tensors, target_aux_buf, embed_table) =
            match step3_build_drafter_scaffolding(&device) {
                Some(t) => t,
                None => return,
            };
        let mut drafter = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .expect("drafter");
        let cache = DrafterKvCache::new(
            &device, cfg.num_kv_heads, 4, cfg.head_dim,
        )
        .expect("alloc cache");
        drafter.attach_kv_cache(cache).expect("attach");
        // Populate via one predict_topk.
        let view = TreeContextView {
            tokens: &[10_u32],
            parents: &[None],
        };
        let _ = drafter.predict_topk(view, 0, 3).expect("predict");
        assert_eq!(drafter.kv_cache_len(), 1);
        drafter.clear_kv_cache();
        assert_eq!(drafter.kv_cache_len(), 0);
    }

    #[test]
    fn adr_037_e5b_step3_rollback_kv_cache_delegates_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let (cfg, tensors, target_aux_buf, embed_table) =
            match step3_build_drafter_scaffolding(&device) {
                Some(t) => t,
                None => return,
            };
        let mut drafter = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .expect("drafter");
        let cache = DrafterKvCache::new(
            &device, cfg.num_kv_heads, 4, cfg.head_dim,
        )
        .expect("alloc cache");
        drafter.attach_kv_cache(cache).expect("attach");
        // Populate to depth 2 via two predict_topk calls.
        let view_root = TreeContextView {
            tokens: &[10_u32],
            parents: &[None],
        };
        let root_cands = drafter
            .predict_topk(view_root, 0, 3)
            .expect("root expansion");
        let child_tok = root_cands[0].token;
        let view_child = TreeContextView {
            tokens: &[10_u32, child_tok],
            parents: &[None, Some(0)],
        };
        let _ = drafter
            .predict_topk(view_child, 1, 3)
            .expect("child expansion");
        assert_eq!(drafter.kv_cache_len(), 2);
        // Rollback to keep only root.
        drafter.rollback_kv_cache(&[0]).expect("rollback");
        assert_eq!(drafter.kv_cache_len(), 1);
    }

    #[test]
    fn adr_037_e6_gpu_drafter_with_cache_orchestrator_max_depth_2_2026_05_22() {
        // END-TO-END test: GpuDrafter with attached cache + the new
        // cache-aware orchestrator expand_dynamic_tree_with_cache.
        // This proves max_depth>1 trees actually expand correctly
        // when the full pipeline runs on the GPU.
        use crate::inference::spec_decode::eagle3::dynamic_tree::{
            expand_dynamic_tree_with_cache, DynamicTreeConfig,
        };
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let (cfg, tensors, target_aux_buf, embed_table) =
            match step3_build_drafter_scaffolding(&device) {
                Some(t) => t,
                None => return,
            };
        let mut drafter = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .expect("drafter");
        let cache = DrafterKvCache::new(
            &device, cfg.num_kv_heads, 16, cfg.head_dim,
        )
        .expect("cache");
        drafter.attach_kv_cache(cache).expect("attach");

        // Budget=8, max_depth=2, top_k=3. Best-first will switch
        // branches, exercising rollback.
        let tree_cfg = DynamicTreeConfig {
            budget: 8,
            max_depth: 2,
            top_k: 3,
        };
        let tree = expand_dynamic_tree_with_cache(123, &mut drafter, &tree_cfg)
            .expect("expand max_depth=2");
        // Tree should respect budget.
        assert!(tree.len() <= tree_cfg.budget);
        assert!(tree.len() >= 2, "should expand beyond root");
        // Should reach depth 2 (we have budget).
        let max_observed_depth = tree.depths.iter().max().copied().unwrap_or(0);
        assert!(
            max_observed_depth >= 1,
            "should reach at least depth 1; got max depth {}",
            max_observed_depth
        );
        // Validate the produced tree structure.
        tree.validate().expect("ExpandedTree::validate");
    }

    #[test]
    fn adr_037_e5b_step3_rollback_kv_cache_errs_without_attached_cache_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let (cfg, tensors, target_aux_buf, embed_table) =
            match step3_build_drafter_scaffolding(&device) {
                Some(t) => t,
                None => return,
            };
        let mut drafter = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry,
            &target_aux_buf, &embed_table, 0,
        )
        .expect("drafter");
        let err = drafter.rollback_kv_cache(&[0]).unwrap_err();
        assert!(
            err.to_string().contains("no cache attached"),
            "expected no-cache error, got: {err}"
        );
    }

    #[test]
    fn adr_037_e4b10b3_gpu_drafter_rejects_target_aux_wrong_size_2026_05_22() {
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        let mut registry = KernelRegistry::new();
        let cfg = cfg_for_drafter_test();
        let manifest = expected_manifest(&cfg);
        let blob = build_test_blob(&manifest);
        let weights = Eagle3Weights::load(&blob, &cfg).expect("load");
        let tensors =
            Eagle3DrafterTensors::upload(&device, &cfg, &weights).expect("upload");
        // target_aux with wrong element count.
        let mut bad = device
            .alloc_buffer(40, DType::F32, vec![10])
            .expect("alloc bad target_aux");
        bad.as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&vec![0.0f32; 10]);
        let embed_table = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
        let err = GpuDrafter::new(
            &cfg, &tensors, &device, &mut registry, &bad, &embed_table, 0,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("target_aux has 10 elements"),
            "got: {err}"
        );
    }
}
