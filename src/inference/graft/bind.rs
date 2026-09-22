//! ADR-059 — a bound graft bank plus its activation-affecting identity.
//!
//! Mirrors `inference/glp/bind.rs`'s role for GLP: the loader produces a
//! `BoundGraft`, the family model carries it, and the cache-identity
//! machinery consumes [`graft_params_hash`] so a grafted engine's saved
//! KV can never be addressed by an ungrafted (or differently-grafted)
//! one. Device placement is NOT done here: unlike a GLP direction (which
//! must live on-GPU beside the activations it steers), a graft bank is
//! spliced into per-request cache bytes by
//! `HybridKvCache::splice_graft_for_slot` at slot admission — the bank
//! stays host-side until then.

use super::reader::{GraftBank, GraftKind, GraftMode};

/// A loaded, bind-validated graft bank with its identity hash.
#[derive(Debug, Clone)]
pub struct BoundGraft {
    pub bank: GraftBank,
    /// Stable identity over every activation-affecting bank field (see
    /// [`graft_params_hash`]). Raw (pre-splice) values are hashed so the
    /// identity tracks the artifact FILE, not any derived cache state.
    pub identity_hash: u64,
}

impl BoundGraft {
    /// Bind a conformance-checked bank. Pure: no device work, no
    /// mutation — the hash is derived once and travels with the bank.
    pub fn bind(bank: GraftBank) -> Self {
        let identity_hash = graft_params_hash(Some(&bank));
        Self { bank, identity_hash }
    }

    /// The splice region length; also the position offset the request
    /// path must apply for every write/read on a grafted slot.
    pub fn n_slots(&self) -> u32 {
        self.bank.n_slots
    }
}

/// Fingerprint of the activation-affecting graft configuration: splice
/// site, mode, derivation kind, bank length, RoPE identity, and the raw
/// per-layer K/V bytes in layer order. Mirrors
/// `engine_qwen35::glp_steering_params_hash` semantics: `None` hashes a
/// constant so ungrafted keys stay stable and distinct; any change to
/// the bank's content or splice semantics changes the hash.
pub fn graft_params_hash(graft: Option<&GraftBank>) -> u64 {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    match graft {
        None => h.update(b"graft=none"),
        Some(bank) => {
            h.update(b"graft=v1");
            h.update(bank.hook_point.as_str().as_bytes());
            h.update(graft_mode_str(bank.mode).as_bytes());
            h.update(graft_kind_str(bank.kind).as_bytes());
            h.update(&bank.n_slots.to_le_bytes());
            h.update(&bank.rope_theta.to_le_bytes());
            h.update(&bank.rotary_dim.to_le_bytes());
            h.update(&bank.position_base.to_le_bytes());
            h.update([bank.mrope_interleaved as u8]);
            for (layer, kv) in &bank.layers {
                h.update(&layer.to_le_bytes());
                for value in kv.k.iter().chain(kv.v.iter()) {
                    h.update(&value.to_le_bytes());
                }
            }
        }
    }
    let digest = h.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("sha256 prefix"))
}

fn graft_mode_str(mode: GraftMode) -> &'static str {
    match mode {
        GraftMode::SplicePrefix => "splice_prefix",
    }
}

fn graft_kind_str(kind: GraftKind) -> &'static str {
    match kind {
        GraftKind::PrefillKv => "prefill_kv",
        GraftKind::SoftPromptKv => "softprompt_kv",
        GraftKind::DirectKv => "direct_kv",
    }
}

/// Convenience constructor for tests and fixture code: a minimal
/// complete-coverage bank over the given layers.
#[cfg(test)]
pub(crate) fn test_bank(
    n_slots: u32,
    heads: usize,
    head_dim: usize,
    layers: &[u32],
) -> GraftBank {
    use super::reader::{GraftHookPoint, GraftLayerKv};
    use std::collections::BTreeMap;

    GraftBank {
        mode: GraftMode::SplicePrefix,
        kind: GraftKind::DirectKv,
        hook_point: GraftHookPoint::FullAttnKv,
        n_slots,
        layers: layers
            .iter()
            .map(|&l| {
                let n = n_slots as usize * heads * head_dim;
                (
                    l,
                    GraftLayerKv {
                        k: vec![0.25; n],
                        v: vec![-0.5; n],
                    },
                )
            })
            .collect::<BTreeMap<u32, GraftLayerKv>>(),
        n_kv_heads: heads,
        head_dim,
        rope_theta: 1e7,
        rotary_dim: 64,
        position_base: 0,
        mrope_interleaved: true,
        content_sha256: None,
        quant_lane: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S6-style identity discipline: same bank → same hash; any change
    /// to content, geometry, site, or RoPE identity → different hash;
    /// ungrafted → stable and distinct from every grafted config.
    #[test]
    fn graft_identity_separates_configurations() {
        let a = test_bank(3, 2, 8, &[3, 7]);
        let a2 = test_bank(3, 2, 8, &[3, 7]);
        assert_eq!(
            graft_params_hash(Some(&a)),
            graft_params_hash(Some(&a2)),
            "identical banks must share the identity key"
        );

        // Bank content change.
        let mut other = test_bank(3, 2, 8, &[3, 7]);
        other.layers.get_mut(&3).unwrap().k[0] = 0.5;
        assert_ne!(
            graft_params_hash(Some(&a)),
            graft_params_hash(Some(&other)),
            "content change must rebuild, not reuse"
        );

        // Bank length change.
        let longer = test_bank(4, 2, 8, &[3, 7]);
        assert_ne!(graft_params_hash(Some(&a)), graft_params_hash(Some(&longer)));

        // Layer-coverage change.
        let wider = test_bank(3, 2, 8, &[3, 7, 11]);
        assert_ne!(graft_params_hash(Some(&a)), graft_params_hash(Some(&wider)));

        // RoPE identity change.
        let mut rope = test_bank(3, 2, 8, &[3, 7]);
        rope.rope_theta = 1e6;
        assert_ne!(graft_params_hash(Some(&a)), graft_params_hash(Some(&rope)));

        // Ungrafted is stable and distinct.
        let none = graft_params_hash(None);
        assert_eq!(none, graft_params_hash(None));
        assert_ne!(none, graft_params_hash(Some(&a)));
    }

    #[test]
    fn bound_graft_carries_bank_and_hash() {
        let bank = test_bank(2, 2, 8, &[3, 7]);
        let bound = BoundGraft::bind(bank.clone());
        assert_eq!(bound.n_slots(), 2);
        assert_eq!(bound.identity_hash, graft_params_hash(Some(&bank)));
        assert_eq!(bound.bank.layers.len(), 2);
    }
}
