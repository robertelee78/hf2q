//! C2 §7.4: production store versus the C1 CPU oracle. CPU payload OWNERS
//! are allowed here; no inference or GPU state is mocked by this test.
use super::*;
use crate::inference::anchor_contract as c1;
use std::sync::Arc;
const CACHE_OWNER: u64 = 1;
const CACHE_BYTES: usize = 4096;

fn identity(generation: u64) -> c1::Identity {
    c1::Identity { artifact: [1; 32], family_schema: "Qwen35-C2".into(),
        tokenizer: [2; 32], template: [3; 32], configuration: [4; 32],
        conversation: 0, slot_generation: generation, lineage_generation: generation,
        graft_generation: generation, graft_digest: [5; 32], graft_len: 0 }
}
struct Pair {
    production: Qwen35AnchorStore,
    oracle: c1::Store,
    identity: c1::Identity,
    known: Vec<c1::Anchor>,
    next_owner: u64,
    generation: u64,
}
impl Pair {
    fn new(keep: usize, quota: usize) -> Self {
        let identity = identity(0);
        Self { production: Qwen35AnchorStore::new_with_quota(keep, Some(quota)).unwrap(),
            oracle: c1::Store::new(identity.clone(), keep, quota + CACHE_BYTES,
                vec![c1::Owner { id: CACHE_OWNER, extent: CACHE_BYTES }]).unwrap(),
            identity, known: Vec::new(), next_owner: 100, generation: 0 }
    }
    fn candidate(&mut self, request: u64, depth: usize, payload: Arc<[u8]>, owner_id: u64)
        -> (Qwen35Anchor, c1::Anchor)
    {
        let production = Qwen35Anchor { id: 0, request, prompt_tokens: vec![7; depth],
            kv: HybridKvSlotAnchor::test_with_payload_owner(depth, payload.clone()),
            prefill_logits: vec![request as f32; 11], vision_fingerprint: None, spec: None };
        let ancillary = production.owned_bytes() - payload.len();
        let oracle = c1::Anchor { id: self.production.next_id, request,
            identity: self.identity.clone(), logical_tokens: depth, prefix_digest: [depth as u8; 32],
            components: vec![c1::Component { role: "DeltaNet-image".into(), shape: vec![payload.len()],
                element_bytes: 1, representation: "CPU-owned-qualification".into(), cursor: depth,
                capacity: CACHE_BYTES, bytes: payload.to_vec() }],
            owners: vec![c1::Owner { id: owner_id, extent: payload.len() },
                c1::Owner { id: self.next_owner, extent: ancillary }],
            required_live_rows: vec![c1::LiveRows { owner: CACHE_OWNER, start: 0, end: depth }] };
        self.next_owner += 1;
        (production, oracle)
    }
    fn stage(&mut self, request: u64, depth: usize, payload: Arc<[u8]>, owner: u64) -> u64 {
        let (production, mut oracle) = self.candidate(request, depth, payload, owner);
        let prod = self.production.stage(request, production);
        let reference = self.oracle.stage(oracle.clone());
        assert_eq!(prod.is_ok(), reference.is_ok(), "capture/stage refusal");
        if let Ok(id) = prod { oracle.id = id; self.known.push(oracle); id } else { 0 }
    }
    fn commit(&mut self, request: u64) {
        let before: Vec<_> = self.production.committed().iter().map(|a| a.id).collect();
        let production = self.production.commit(request);
        let oracle = self.oracle.complete(request, true);
        // C1 reports duplicate completion as refusal; the serving API's specified
        // duplicate-completion result is an idempotent no-op. Publication is equal.
        if oracle.is_err() { assert!(production.is_err() || matches!(production, Ok(false)));
            assert_eq!(before, self.production.committed().iter().map(|a| a.id).collect::<Vec<_>>()); }
        else { assert_eq!(production.unwrap(), true); }
        self.check("completion");
    }
    fn check(&mut self, step: &str) {
        assert_eq!(self.production.owned_bytes(), self.oracle.reserved().unwrap() - CACHE_BYTES,
            "{step}: unique-owner conservation (nonzero images, alias roots and ancillary owners)");
        for anchor in &self.known {
            let production = self.production.ticket(anchor.id);
            let oracle = self.oracle.preflight(anchor.id, &self.identity, anchor.logical_tokens,
                &anchor.prefix_digest, &anchor.components);
            assert_eq!(production.is_ok(), oracle.is_ok(), "{step}: visibility/refusal id{}", anchor.id);
            if let Err(error) = &production {
                let expected_reason = if self.oracle.is_poisoned() { AnchorRestoreRefusal::Poisoned }
                    else if self.production.open_transaction().is_some() { AnchorRestoreRefusal::Transaction }
                    else { AnchorRestoreRefusal::Unavailable };
                assert_eq!(error.downcast_ref::<AnchorRestoreRefusal>(), Some(&expected_reason),
                    "{step}: typed refusal reason");
            }
            if let Ok(ticket) = production { self.production.abort_transaction(ticket).unwrap(); }
        }
        assert_eq!(self.production.is_poisoned(), self.oracle.is_poisoned(), "{step}: poison");
    }
    fn restore_and_prune(&mut self, id: u64) {
        let anchor = self.known.iter().find(|a| a.id == id).unwrap().clone();
        let production = self.production.ticket(id).unwrap();
        let reference = self.oracle.preflight(id, &self.identity, anchor.logical_tokens,
            &anchor.prefix_digest, &anchor.components).unwrap();
        let pending_request = self.production.pending().map(|pending| pending.request);
        let mut live = Vec::new();
        self.oracle.apply(reference, &mut live).unwrap();
        self.production.before_restore(production).unwrap();
        self.production.finish_restore(production, true).unwrap();
        self.oracle.before_writes(id, &[c1::LiveRows { owner: CACHE_OWNER,
            start: anchor.logical_tokens, end: CACHE_BYTES }]).unwrap();
        // The normative oracle invalidates overlapping live rows. C2's Phase B
        // additionally discards its request's pending publication even when
        // that payload is shallower and would survive the write-range test.
        if let Some(request) = pending_request { self.oracle.cancel(request, &[], true); }
        assert_eq!(live, anchor.components, "apply restores actual distinct CPU payload bytes");
        self.check("apply/beforewrites");
    }
    fn reset(&mut self, graft_identity_change: bool) {
        self.generation += 1;
        let next = identity(self.generation);
        if graft_identity_change { self.oracle.replace_identity(next.clone(), true).unwrap(); }
        else { self.oracle.reset(next.clone()).unwrap(); }
        self.production.reset();
        self.identity = next;
        self.check(if graft_identity_change { "identity replacement" } else { "reset" });
    }
}

#[test]
fn anchor_c1_production_store_transition_differential_nonzero_aliased_owners() {
    for keep in [1, 4, 8] {
        let mut pair = Pair::new(keep, 64 * 1024);
        let shared: Arc<[u8]> = (0..257).map(|i| (i % 251) as u8).collect::<Vec<_>>().into();
        for request in 1..=6 {
            let id = pair.stage(request, request as usize * 8, shared.clone(), 10);
            assert!(id > 0);
            pair.check("pending invisible");
            pair.commit(request);
            pair.commit(request);
        }
        let deepest = pair.production.deepest().unwrap().id;
        pair.restore_and_prune(deepest);
        if keep > 1 {
            let shallow = pair.production.committed().first().unwrap().id;
            pair.restore_and_prune(shallow);
        }
        let pending = pair.stage(20, 80, Arc::from(vec![91u8; 511]), 11);
        assert!(pending > 0);
        pair.production.discard_pending(20);
        pair.oracle.cancel(20, &[], true);
        pair.check("cancel aftercapture beforecommit");
        pair.production.discard_pending(21);
        pair.oracle.cancel(21, &[], true);
        pair.check("cancel beforecapture");
        pair.production.poison_and_clear();
        pair.oracle.partial_failure(true);
        pair.check("encoding partial failure/known drain");
        assert_eq!(pair.stage(22, 88, shared.clone(), 10), 0, "poison cannot publish");
        pair.reset(false);
        pair.stage(23, 96, shared.clone(), 10);
        pair.commit(23);
        pair.reset(true);
    }
}

#[test]
fn anchor_c1_production_capture_restore_transaction_and_stale_ticket_differential() {
    let mut pair = Pair::new(4, 64 * 1024);
    let payload: Arc<[u8]> = vec![0xadu8; 333].into();
    let (candidate, oracle) = pair.candidate(1, 8, payload.clone(), 10);
    let capture = pair.production.begin_capture().unwrap();
    pair.oracle.begin_capture(oracle.clone(), &[]).unwrap();
    let refusal = pair.production.ticket(1).err().expect("open capture must refuse restore");
    assert_eq!(refusal.downcast_ref::<AnchorRestoreRefusal>(), Some(&AnchorRestoreRefusal::Transaction));
    assert!(pair.oracle.preflight(1, &pair.identity, 8, &[8; 32], &oracle.components).is_err());
    pair.production.finish_capture(capture).unwrap();
    pair.production.stage(1, candidate).unwrap();
    pair.oracle.finish_capture(true).unwrap();
    pair.known.push(oracle);
    pair.commit(1);
    let ticket = pair.production.ticket(1).unwrap();
    let reference_anchor = pair.known[0].clone();
    let oracle_ticket = pair.oracle.preflight(1, &pair.identity, 8, &[8; 32],
        &reference_anchor.components).unwrap();
    pair.production.abort_transaction(ticket).unwrap();
    pair.stage(2, 16, payload, 10);
    let refusal = pair.production.validate_ticket(ticket).err().expect("stale ticket must refuse");
    assert_eq!(refusal.downcast_ref::<AnchorRestoreRefusal>(), Some(&AnchorRestoreRefusal::Stale));
    assert!(pair.oracle.apply(oracle_ticket, &mut Vec::new()).is_err());
    pair.production.discard_pending(2);
    pair.oracle.cancel(2, &[], true);
    pair.check("stale ticket after cancelled capture");
    pair.reset(false);
}

#[test]
fn anchor_c1_production_quota_refusal_is_atomic_for_nonzero_payload() {
    let mut pair = Pair::new(4, 256);
    let payload: Arc<[u8]> = vec![37u8; 1024].into();
    assert_eq!(pair.stage(1, 8, payload, 10), 0);
    assert!(pair.production.is_empty());
    pair.check("quota refused beforepublication");
}

#[test]
fn anchor_c1_completed_pending_is_not_an_open_capture_transaction() {
    let mut pair = Pair::new(4, 64 * 1024);
    let payload: Arc<[u8]> = vec![0x5bu8; 257].into();
    let committed = pair.stage(1, 32, payload.clone(), 10);
    pair.commit(1);
    let pending = pair.stage(2, 8, payload.clone(), 10);
    pair.check("completed capture: pending unavailable, committed preflightable");
    assert!(pair.production.anchor(pending).is_none());
    assert!(pair.production.open_transaction().is_none());
    pair.restore_and_prune(committed);
    assert!(pair.production.pending().is_none(),
        "C2 Phase B must discard even a shallower non-overlapping pending payload");
    pair.commit(2);

    let (candidate, constructing) = pair.candidate(3, 40, payload, 10);
    let capture = pair.production.begin_capture().unwrap();
    pair.oracle.begin_capture(constructing.clone(), &[]).unwrap();
    let error = pair.production.ticket(committed).unwrap_err();
    assert_eq!(error.downcast_ref::<AnchorRestoreRefusal>(), Some(&AnchorRestoreRefusal::Transaction));
    let original = pair.known.iter().find(|a| a.id == committed).unwrap();
    assert_eq!(pair.oracle.preflight(committed, &pair.identity, original.logical_tokens,
        &original.prefix_digest, &original.components).err(),
        Some("restore identity/lineage/transaction"));
    pair.production.finish_capture(capture).unwrap();
    pair.production.stage(3, candidate).unwrap();
    pair.oracle.finish_capture(true).unwrap();
    pair.known.push(constructing);
    pair.check("checked capture ended: prior committed anchors visible again");
    pair.production.discard_pending(3);
    pair.oracle.cancel(3, &[], true);
    pair.check("pending discarded without publication");
}
