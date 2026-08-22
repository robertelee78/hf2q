# ADR-049: Agentic state reuse (multi-anchor) and Mixed-phase cooperative prefill

- Status: Accepted; execution in progress (qwen35-family model-free Lane A
  proof complete at rev 4, real-artifact gates and cross-family phases open)
- Date: 2026-08-22
- Updated: 2026-08-22 (rev 4, qwen35-family model-free execution milestone)
  — implementation commit `95d618c8`, based on main `32181b61`: explicit
  per-slot AnchorStore,
  linear-lineage pruning, fail-atomic restore preflight, exact payload
  ownership/accounting, terminal publication, A.8 logs/metrics, idle audit,
  independent reference-model + 17-mutation battery, and Lane C corrections.
  Qwen3.6/Qwen3.8 hardware receipts are intentionally not claimed here; see
  the execution ledger below.
- Owners: hf2q serving engine (execution: the active qwen35/qwen38 serving-lane session; plan authored by the FreeToken research session)
- Code pins: planning review at hf2q `242882e8`; rev-4 execution based on
  merged main `32181b61`; mlx-native `0.11.2`. Anchors were authored at
  `815bd48d`; every correction-touched anchor was re-verified before editing.
- Provenance: full paper+code study of FreeToken (arXiv 2608.16157, "FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution") mapped onto hf2q/mlx-native by a nine-agent research swarm, then adversarially reviewed by two independent external models (Kimi K3 via opencode; gpt-5.6-sol via codex, 516k-token source-grounded review). Both reviews' MUST-FIX items are incorporated; the gpt-5.6-sol review found and this ADR closes a stale-KV lineage coherence bug in the original draft (§A.2).

## Context

FreeToken serves frontier MoE models on consumer discrete-GPU PCs by treating host RAM as the expert-weight source of truth and VRAM as a global (layer,expert) LRU cache, splitting decode misses between PCIe-fetch and CPU-execute by a measured closed-form ratio, streaming whole expert layers double-buffered during prefill, checkpointing hybrid-model recurrent state at chunk boundaries nearly free, and resizing pools elastically at idle safe points. Its headline agentic result: worst-case TTFT under 44 s where llama.cpp/Ollama/KTransformers show 232–946 s worst turns — earned mostly by recurrent-state checkpoint placement plus prefix reuse, not by raw kernel speed.

Most of FreeToken's bandwidth machinery does not transfer to Apple Silicon unified memory (§Rejections). What does transfer lands exactly on hf2q's two open sore spots:

1. **Same-slot context edits go cold.** Agent harnesses (opencode, Claude Code) rewrite context mid-conversation — strip thinking blocks, collapse tool output. hf2q keeps exactly one anchor per slot (`prompt_anchors[slot] = Some(...)`, install at src/serve/api/engine.rs:16925), so any divergence inside retained tokens that predates the single anchor recomputes the whole prefix. hf2q's *matching* is already ahead of FreeToken's (template-aware cue-less re-render, strict-token-prefix acceptance, src/serve/api/handlers.rs:1905-1938, vision-aware via `expand_stable_prompt_boundary`); its checkpoint *depth* is 1.
2. **DeepSeek4 Mixed-phase prefill never aggregates rows.** The landed cooperative prefill cohort (protected runs 1.2537×/1.2816×) is bypassed whenever a decode is runnable: `max_prefill_windows: has_runnable_decode.then_some(DEEPSEEK4_INTERACTIVE_PREFILL_WINDOWS /* = 2 */)` (src/serve/api/engine.rs:7857, 7885) and any `Some` cap skips cohort planning (engine.rs:9716-9731). The 35 s tool-result wave failures accrue in exactly this Mixed phase, where each serial 256-token slice re-pays the fixed cost F ≈ 1.05–1.4 s (one full expert-weight stream, ~96 GiB at ~90–98 GB/s effective mm_id rate; marginal cost c ≈ 1.17 ms/row).

Key enabling facts at HEAD:
- The per-slot anchor payload `HybridKvSlotAnchor` (src/inference/models/qwen35/kv_cache.rs:996-1002) is cursors + DeltaNet state only — **zero KV bytes copied** (full-attn K/V is append-only; the slot's own KV is the pin). Snapshot and restore are **already slot-indexed** (`snapshot_slot_anchor` kv_cache.rs:1532, `restore_slot_anchor` :1607) with the cursor proof `live_cursor >= saved_cursor` (:1651-1656) and per-slot ping-pong parity re-canonicalization (:1694).
- One anchor costs ≈ 62.8 MiB for Qwen3.6-35B-A3B (30 DeltaNet layers × (2 MiB recurrent [128×128×32 f32] + 96 KiB conv [8192×3 f32])); ≈ 149.6 MiB for Qwen3.8-27B (48 layers). Capture is a ~2–4 ms host memcpy at a point where the engine is already stopped on the boundary. Anchors are host-owned `Vec<u8>` — anonymous RAM, not Metal working set.
- The prefill chunker already clamps a transaction to end exactly at the stable boundary (`qwen35_next_prefill_end`, src/serve/api/engine_qwen35.rs:3635-3648; checkpoint emission gated on `stable_prompt_prefix_tokens == Some(end)`, :3966-3985). Since commit `35e42b28`, single-slot workers use a 4,096-token transaction ceiling while multi-slot stays 2,048 (`qwen35_slot_prefill_chunk_tokens`, engine.rs:16729-16749).
- DeltaNet spec-decode rollback (`rollback_la_to`) is per-slot in signature (kv_cache.rs:2296-2300) but structurally absent from the slot-aware worker call graph (pin H38, engine.rs:44835-44906) — prefill-time anchors face no speculative interleaving today.

## Decision

Three lanes, in this order. Lane A is the primary value; Lane B is gated on a coherence spike; Lane C ships with Lane A's first PR.

**Scope directive (Robert, 2026-08-22): every lever this ADR ships is a cross-family benefit — all supported models, all supported families share it. Single-model or single-family shipments are milestones, never the deliverable, and the ADR is not complete while any supported family lacks a shipped lever it can benefit from.** For Lane A: the first implementation lands on the qwen35-family engine (one engine serving both Qwen3.6-35B-A3B and Qwen3.8-27B); gemma4 and deepseek4 parity are REQUIRED phases, not optional follow-ons; per-family gates run on the artifact each lane actually serves. For Lane B: deepseek4 is the first implementation and §B.2 carries the required family-generalization evaluation. The directive also binds future families: any family gaining serve support later (e.g. Qwen3-VL if/when ADR-041's engine seam lands) must adopt these levers as part of its engine bring-up, not as a deferred extra.

**Not in committed scope, but an OPEN HYPOTHESIS with a deciding spike** (framing per Robert 2026-08-22: these are questions needing data, not parked scope): raising `MAX_COOPERATIVE_PREFILL_ROWS` (the "Lane 2b" of the draft). Hypothesis: a larger aggregate row budget still pays in pure-prefill waves without breaching the memory envelope. Data against so far: ADR-042 records a 4,096-row OOM (ADR-042:59, :503) and a later 4,096/cold-cooperative failure (:2069); projected gain ~1.15× confined to the pure-prefill wave; receipt verifiers/artifact tests encode the exact 2,048-row shapes. Deciding spike: Lane B's wave-phase profiling (does F-dominance survive in pure-prefill waves once Mixed cohorts land?) plus a fresh transient high-water measurement at 4,096 beside the 100 GiB artifact under the current single-layer-CB cooperative structure. Outcome: implement, or falsify and record.

### Lane A — Multi-anchor slot-local checkpoints (qwen35 first)

**Contract (per gpt-5.6-sol M3, smaller than the draft's):** one validated current-turn boundary per *successfully committed* request, accumulated over observed turns into a per-slot anchor store. No sorted-boundary-set machinery, no seeding of historical turns from a cold transcript (that would require handler/template work to render and validate multiple message prefixes — explicitly out of scope). The handler already computes exactly one boundary per request (handlers.rs:1905-1938 → `SamplingParams` → `Qwen35PrefillState`); the engine change is to *accumulate* instead of overwrite. Default-on: the workload is 100% agentic, capture is ~2–4 ms at a natural stop, and depth-4 at shipping `max_slots=4` costs ≈ 1.0 GiB (ceiling: 2.0 GiB at 8 slots) against ~25 GiB headroom in the 48 GiB KV grant.

**A.1 — AnchorStore.** Replace the single `Option<Qwen35PromptAnchor>` (struct at engine.rs:16865-16871) with an explicit per-slot store, not a raw Vec:

```
AnchorStore { committed: Vec<Anchor>, pending: Option<Anchor>, lineage_epoch: u64, owned_bytes: u64 }
```

Three-state publication machine (gpt-5.6-sol M2; generalizes DeepSeek4's pending→committed two-phase commit at engine_deepseek4.rs:921-946):
- *Committed* anchors: visible to affinity and cancellation rollback.
- *Request-local pending* capture: invisible to other admissions until the request reaches terminal cache+ledger success (the retained-token ledger publishes at engine.rs:19148 — pending merges atomically there, then eviction applies).
- On failure or cancellation: discard pending; the committed list survives unchanged. The existing install sites (engine.rs:18825, :18870) sit in the same match arms as cancellation recovery (:18829-18841, :18872-18881) — recovery must additionally prune every committed anchor whose cursor exceeds the post-recovery live cursor (Kimi M2).

**A.2 — Linear-lineage invariant (the coherence core; closes the draft's stale-KV bug).** Anchors index positions in ONE mutable per-slot KV log. Restoring anchor A and then writing a divergent suffix overwrites the physical rows that backed every deeper anchor — after which a deeper anchor's token match AND cursor check can both pass while the KV bytes are wrong. Therefore, fail-closed law:

> Before the first KV write after restoring anchor A, invalidate every anchor deeper than A (bump `lineage_epoch`; drop or tombstone the descendants). A cold reset, a slot poison, or any FAILED restore invalidates the ENTIRE store for that slot — mandatory for fail-closed recovery. Anchor selection must check epoch, never tokens+cursor alone.

Mandatory regression (must exist before any restore path merges): build lineage A→B→C, restore A, prefill divergent branch X, then send a request matching old C — the engine must go cold (or restore A), NEVER restore B or C. Byte-compare the divergent-branch output against a cold run.

**A.3 — Restore-on-divergence.** Extend slot affinity (`qwen35_slot_affinity`, engine.rs:16975-17047) from best-of-{live cursor, one anchor} to best-of-{live cursor, deepest *epoch-valid* committed anchor whose tokens are a prefix of the request}. Matching preserves **equality** (not strict prefix): the existing full-prompt-equality path replays stored `prefill_logits` and skips the forward entirely (engine.rs:16985, :17001) — that behavior generalizes to the deepest anchor equal to the full new prompt. On divergence inside retained tokens: `restore_slot_anchor` at the selected anchor, re-prefill only the suffix. Today's behavior in that case is cold; this is the lane's entire payoff.

**Restore contract (fail-atomicity — executor-audit finding, verified at `242882e8`):** `restore_slot_anchor` currently interleaves validation with mutation — full-attn cursors are rewound inside the same loop as the per-layer ensures, and the MTP cursor is rewound before the linear-state copies, which can still fail — so a mid-restore error leaves the slot partially rewound. Lane A must refactor it to **preflight ALL validations, then mutate**. On any restore error: hard-reset the slot and clear its entire anchor store — NEVER fall back to a shallower anchor after a partial restore (that is the A.2 bug class by another road).

**A.4 — Eviction & budget.**
- Eviction: positional keep-newest-K (K default 4; anchors form a nested prefix chain, so LRU-by-restore is actively wrong — a twice-edited turn would evict the deeper anchor about to be needed; both reviewers concurred). Descendant invalidation (A.2) runs before any eviction policy matters. One refinement permitted later, telemetry-first: reserve slot 0 of the list for the oldest/system boundary, K−1 for newest.
- **Payload ownership rule (executor-audit finding):** every element of an anchor's payload must be host-owned or a dedicated right-sized allocation — NEVER a view/clone retaining a larger transient allocation. Today `pending_target_hidden` is an `MlxBuffer` captured by cloning a view whose parent is the prefill residual allocation (engine_qwen35.rs:3406, capture sites :3972/:4759): the logical row is ~20 KiB, but the clone retains the ~40 MiB (2,048-row) / ~80 MiB (4,096-row) parent Metal allocation — one per anchor. Capture must copy the row into a dedicated `[1, H]` allocation or host memory. Required regression: after capture, assert no chunk-sized parent allocation remains retained by any anchor (allocation-accounting check).
- Budget: `HybridKvSlotAnchor::total_bytes()` (kv_cache.rs:1004) undercounts — it omits prompt tokens, the vocab-sized `prefill_logits`, and the spec hidden row owned by `Qwen35PromptAnchor` (engine.rs:16865-16871, spec boundary struct engine_qwen35.rs:3404-3407). Account **all owned payload** as a separate reclaimable `anchor_owned_bytes` line surfaced to admission — NOT added to scheduler high-water, which is deliberately monotonic because Metal pages are never reclaimed (src/serve/scheduler.rs:1047, :1218); host-owned evictable bytes charged there would never return. **K counts committed anchors only; the preflight charges K committed + 1 pending.** Preflight fail-closed: a capture that would exceed the anchor budget is skipped (documented scope gap), never partially taken.
- Per-model anchor cost (budget is byte-denominated; `K_effective = min(4, floor(slot_anchor_budget / anchor_bytes(model)))` — computed from allocation code, never doc comments):

  | Model | recurrent+conv state | ≈ total w/ logits+tokens+hidden | K=4 × 4 slots | K=4 × 8 slots |
  |---|---|---|---|---|
  | Qwen3.6-35B-A3B (30 DeltaNet layers) | 62.8 MiB | ≈63.5 MiB | ≈1.02 GiB | ≈2.03 GiB |
  | Qwen3.8-27B (48 layers) | 149.6 MiB | ≈150.3 MiB | ≈2.35 GiB | ≈4.70 GiB |

  Gemma4/DeepSeek4 rows are computed the same way when their parity phases open.
- Idle conservation invariant (strict equality, deliberately stricter than FreeToken's own `<=` on its GDN pool): at scheduler idle, per slot and in aggregate, `live cursor bytes + retained prefix accounting + anchor_owned_bytes + free = grant`. Run it in the same idle hook style FreeToken audits use; it is the audit that would have caught the repo's stale `n_v_heads=8` byte-math comments years early.

**A.5 — Speculation interplay (LIVE requirements — corrected after the executor audit; verified at `242882e8`).** Slot-aware speculative rollback is live on main via its own transactional functions — `rollback_slot_mtp_transaction` (engine_qwen35.rs:4142) and `rollback_slot_target_transaction` (:4173), with fail-closed slot reset if a rollback itself fails (:4197-4226). H38 (engine.rs:44835-44897) still pins `rollback_la_to` — and with it the per-token DeltaNet capture arena — out of the slot-aware worker. The rules below bind NOW and are co-designed with the speculation lane:
1. Anchor capture and restore must be sequenced strictly outside an open MTP/target rollback transaction — never observe mid-transaction cursors. Prefill-time anchors copy bytes OUT of live buffers and are safe once that sequencing holds.
2. After any restore, consumers must re-validate that target AND MTP cursors both equal the anchor's `token_count` — the coherence marker documented on `Qwen35SpecPrefixBoundary` itself (engine_qwen35.rs:3401-3402).
3. A decode-time anchor (not in this ADR's scope) may only snapshot after accept/rollback settles — never from optimistic post-verify state; any capture sharing the LA arena must extract its rows before `clear_la_capture`/re-arm. The arena's true size is ≈1.96 GiB at the 32-token SerialFifo window (30 layers × 67 MiB), 4× the stale in-code comment — Lane A does NOT use this arena (boundary capture reads live state) and MUST NOT replicate it per slot.

**A.6 — Family parity (REQUIRED phases per the scope directive, same invariants):** gemma4 (`Gemma4PromptAnchor` is structurally identical; retire serial-path `live_prefix_tokens` special-casing where subsumed), then deepseek4 (its two-anchor pending/committed pair generalizes to the store; wire `Deepseek4CacheSnapshot::resident_bytes()` — currently zero callers — into the same accounting). The ADR does not reach Implemented until every supported serving family carries the anchor store.

**A.7 — Open hypotheses: the spikes that decide them** (not "deferred" — each is a live question whose data collection is already scheduled or cheap):
- *Cross-slot / restart-surviving registry tier*. Hypothesis: foreign-slot landings and restart warm-up are frequent enough on the real workload to justify a shared CoW prefix store. Deciding data: A.8 telemetry — slot-affinity foreign-landing counts and restart-cold counts over production use. If confirmed → its own ADR (needs a slot-parameterized `restore_partial` — the current one is copy-owning and rewrites every sequence cursor, kv_cache.rs:3435, :3493 — plus an ownership answer to the `b44b92ed` tenant-isolation pin; FreeToken's donate-not-copy/CoW/dual-currency eviction is that ADR's design vocabulary). Meanwhile the SerialFifo `LcpRegistry` + disk hydrate remains the restart-hydrate tier.
- *Dense/stride state capture* (finer-than-boundary anchors). Hypothesis: harness edits land off semantic boundaries often enough that boundary anchors miss real reuse. Deciding data: A.8 divergence-position histograms (an off-boundary edit degrades to cold, never to wrong — so the data can be gathered safely in production). Decision ladder if confirmed: first another semantic oracle (verified tool-call opener, FreeToken's `--enable-special-token-ckpt` analog — cheaper, targeted), then a stride-mode capture kernel under its own ADR and byte budget. Note (corrected FreeToken characterization): FreeToken itself records only the deepest crossed 64-boundary per forward, not every boundary — dense-stride economics were never validated even there.
- *Decode-time anchors*. Hypothesis: divergence points inside generated spans (not just prompt boundaries) carry meaningful reuse. Deciding data: the same histograms, split by prompt-span vs generated-span divergence. If confirmed → A.5 rule 3 already defines the safe capture point.
- *SerialFifo recovery-arena reclaim* (~1.96 GiB). Hypothesis: once the anchor store lands, the 32-token recovery-capture path is redundant for boundary edits. Deciding data: post-Phase-2 recovery-capture hit rate vs anchor coverage. If confirmed → cap or retire the arena in a follow-on.

**A.8 — Telemetry before policy** (ships with A.1): per restore attempt — hit depth, divergence distance, tokens saved, descendant-prune count, eviction reason, capture ms, peak committed+pending bytes. This data decides every deferred refinement above.

**Lane A gates (all fail-closed):**
1. Byte-identity: anchor-restore-plus-suffix-prefill output vs cold full prefill, at every anchor depth, at both transaction widths {2,048 multi-slot, 4,096 single-slot}, boundary at slice edge and mid-slice-clamped.
2. The A.2 lineage regression (A→B→C / rewind / old-C-must-not-restore).
3. Cancellation: cancel mid-prefill after ≥2 anchors installed; committed list must equal the pre-request list.
4. `scripts/test_qwen35_slot_anchor_divergence.sh`: explicit
   `--scheduler inflight-batched --max-slots 4`, truly concurrent clients,
   equality hits, divergent rewrites, cancellation, failed prefill,
   speculative-state carry, and stale-descendant rejection. It exits nonzero
   on every miss and refuses to run when the listener process arguments do not
   prove the required scheduler shape. (`bench_lcp_resume_speedup.sh` is NOT a
   gate for this feature: it drives the stride registry, issues its "4-worker"
   load sequentially, does not select the slot-aware scheduler, and exits 0 on
   a failed speedup — bench_lcp_resume_speedup.sh:303, :442.
   `test_agentic_cache_lifecycle.sh` covers cancellation/isolation, not
   multi-depth lineage; keep it, extend nothing into it.)
5. Perf acceptance: on the divergent-edit scenario, TTFT strictly better than cold; on append-only scenarios, byte-stable and within noise of today. Quiet box, receipts.
6. `cargo test --bin hf2q` (never `--lib`), 40-module GPU lock intact.

### Lane B — Mixed-phase cooperative prefill (deepseek4)

**B.0 — Coherence spike, first, in its own branch (Kimi M1 — this gate decides the lane).** Mixed cooperative execution has never run: cohort commit/poison concurrent with live decode cursors on peer slots. Before any policy work: enumerate exactly which per-slot state the cooperative transaction touches (`Deepseek4CooperativePrefillPlan` path, all-or-poison commit via `publish_prefill_cohort_after_gate` — verifier_forward.rs:76, the cooperative-PREFILL publisher; `publish_verifier_cohort_after_gate` in decode_cohort.rs:12 is the DECODE publisher, which a prior draft misnamed here; direct session-cache borrow engine_deepseek4.rs:649-683). The spike must exercise BOTH the commit and poison paths of the prefill publisher, and their interplay with the decode publisher during a Mixed step, prove decode-lane KV append cursors and compressor accumulators are untouched by commit AND by poison, and land a byte-identity test shaped *cohort-prefill + concurrent decode step* (not cohort-prefill alone). **Abort criterion:** if the spike shows cohort commit touches decode-lane state, the `engine.rs:9716-9731` bypass is load-bearing for correctness, Lane B becomes a scheduler redesign, and it exits this ADR to its own.

**B.1 — Mixed cohort policy (only after B.0 passes).** Treat this as a new scheduler policy, not the removal of one bypass: cohort planning under a runnable decode must honor a per-lane row cap AND the aggregate cap while preserving FIFO-prefix compatibility, identical-plan/reply-class requirements, and recovery-tail behavior (planner engine.rs:9601-9687; lane clamp slots.rs:224-242; `MIN_MATRIX_APPEND_TOKENS = 33`, verifier_forward.rs:25; recovery tail engine_deepseek4.rs:48).
- Rows-per-lane parameterized ∈ {128, 256}; **default 4×128** (halves F payments per aggregate progress while keeping GPU occupancy nearest today's serial Mixed slice); promote to 4×256 only on hardware measurement with contract margin. Projected walls for the canonical four ~3,520-token warm-suffix workload: serial Mixed ≈ 75 s of aggregate prefill work; 4×256 ≈ 31 s; 4×128 ≈ 46 s with near-serial per-slice occupancy.
- Dual latency contract, both fail-closed: (i) scheduler decode-visit gap bound; (ii) client-visible **semantic SSE gap** bound per active decoder. Numbers to be fixed in the spike branch from measured baselines, recorded in this ADR at execution time.
- New required workload test: four prefills + active streaming decoders, measuring decoder starvation behind Mixed prefill — the existing B4 artifact proof is a pure 132-step decode comparison after prefixes are installed (real_artifact_decode_cohort_tests.rs:309) and does not cover this.

**B.2 — Family generalization (scope directive).** Row-aggregation economics are not DeepSeek-specific: every MoE family pays a per-slice fixed cost ≈ one full expert-weight read. DeepSeek4 already carries the aggregation machinery — it is Lane B's first implementation, not an omission from this list — while cross-slot aggregation is deepseek4-only machinery at HEAD (`src/serve/forward_prefill_batched.rs` batches within one sequence; its own doc records the slot-aware N/A at :400-426). Hypothesis: qwen35-family and gemma4 MoE slot-aware prefill show the same F-dominance and would benefit from cooperative suffix aggregation. Deciding spike: measure per-slice fixed cost vs rows on those families' slot-aware prefill using the ADR-042 receipt methodology. If confirmed, cooperative aggregation for those families becomes a REQUIRED phase under this ADR (new code — qwen35 is the in-repo per-slot-KV reference for concurrency-correct forwards); if refuted (e.g. smaller expert pools make F negligible there), record the falsification here with the measurements.

**Lane B gates:** B.0 byte-identity (cohort+concurrent-decode); cooperative receipt regime (≥5 alternating serial/cooperative pairs, sustained median faster, peak RSS recorded, independent receipt verification); thermal contract (Nominal start, continuous Fair-or-better, no gap >5 s, fail-closed); memory H3 (≤116 GiB peak beside the 100 GiB artifact); product ceilings unchanged (60 s cold / 15 s cached-automatic-SSE / 35 s tool-result — never widened); B4 decode-cohort gate re-pass; the two B.1 latency contracts.

### Lane C — Hygiene & methodology (ships with Lane A's first PR)

1. Correct the stale docs that actively misled this research: engine_qwen35.rs:169-170 ("capacity = 1" — live registry is byte-budgeted, capacity `usize::MAX` via `with_byte_budget`, :523-526); kv_cache.rs:127-128 (`n_v_heads=8`/"~60-90 MB" arena — real: n_v=32, ≈1.96 GiB); engine.rs:2300-2301 (qwen35 "501 short-circuit" — stale since 2026-05); lcp_registry.rs:781-783 (chunk_pos "in params_hash" — it is mangled into tenant_id); investigation_env.rs:553-556 ("~96 MB per 27B checkpoint" — real ≈149.6 MiB); load_info.rs:2151 fixture note (missing MTP layer → 6.25% admission undercount for Qwen3.8-27B).
2. Document (not fix, this ADR) the two budget hazards: two independent 5%-of-RAM LCP budgets (engine.rs:3595-3597, engine_qwen35.rs:523-525 — same `default_lcp_byte_budget()` instantiated twice); `HF2Q_KV_PERSIST` carries THREE meanings (path — serve/mod.rs:974; `"0"` disable — :4457-4485; `"1"`/`"on"` enable — kv_persist/families/gemma4_dense.rs:21, kv_persist/index.rs:9). State the registry end-state: the SerialFifo registry + disk hydrate is the *permanent restart-hydrate tier* until the A.7 follow-on ADR replaces it — documented scope, not a stub.
3. Import FreeToken's evaluation discipline into release evidence: report worst-case (tail) TTFT against client watchdog ceilings, not just means; an agentic-stability criterion (decode rate within a fixed % of single-turn under the N=4 workload); the A.4 strict-equality idle conservation audit; a reference-model invariant battery over the AnchorStore state machine (injected-mutation style — FreeToken's equivalent suite caught 17/17).

### Qwen execution ledger — rev 4

Model-free evidence at implementation commit `95d618c8`:

- `qwen35_anchor_store` has an independent reference state machine and eight
  focused tests. The mutation battery rejects 17/17 injected corruptions;
  A→B→C/rewind removes B and C before branch X can write; pending state is
  affinity-invisible; eviction is positional keep-newest-K; accounting charges
  four committed payloads plus one pending payload exactly.
- `slot_anchor_restore_preflights_every_payload_before_mutation` was added as
  a falsifier before the restore refactor and failed against the interleaved
  implementation: an invalid final recurrent payload left the cursor rewound
  from 14 to 9. After the two-phase preflight/mutate refactor, the same test
  proves all cursors, recurrent bytes, conv bytes, and parity remain unchanged
  on validation failure.
- `logical_buffer_copy_does_not_retain_chunk_sized_parent` proves the
  speculative hidden row owns a fresh logical-size allocation after the
  chunk-sized parent drops. Store accounting includes token/logit capacities,
  every nested recurrent/conv allocation, cursor tables, and the detached
  hidden row.
- The SlotAware worker now stages captures as pending, publishes only after
  the retained cache/ledger commit, selects the deepest epoch-valid match,
  prunes descendants before the first divergent write, and clears the full
  store before hard reset after failed restore, poison, or cold reset. A.8
  fields emit in structured logs and Prometheus counters.

Open proof work is concrete rather than implied by these unit results: run the
new concurrent SlotAware divergence gate plus every-depth cold byte comparison
on both Qwen3.6-35B-A3B and Qwen3.8-27B at 2,048- and 4,096-token transaction
widths; record cancellation, failed-prefill, speculative-state, tail-TTFT,
append-only-no-regression, and N=4 stability receipts. The release driver owns
that hardware window. Gemma4/deepseek4 parity and Lane B remain required by the
scope directive.

## Falsified findings and open hypotheses (studied, decided, documented)

Framing (Robert): items below are either FALSIFIED with evidence in hand, or OPEN HYPOTHESES whose deciding spike is named — never silently parked scope.

| FreeToken concept | Verdict | Reason / data so far | Deciding spike or condition |
|---|---|---|---|
| q\* CPU–GPU co-execution | FALSIFIED (for this hardware) | One memory controller: CPU matmul adds no aggregate bandwidth to a BW-bound decode and cannot be bit-identical to Metal kernels (coherence gate). FreeToken's own degenerate limit (B_H→B_P ⇒ q\*→m) is this hardware's whole regime. | None on unified memory; the *method* (measured contended-pair bench → closed-form policy) stays reusable for genuinely disjoint domains. |
| Global LRU expert cache + elastic expert↔KV pools | OPEN HYPOTHESIS | Target models fit wired in 128 GiB today; no capacity misses exist. SSD-as-PCIe analog is ~5–7 GB/s vs their 25–53 — a harsher regime that the hypothesis must survive. | Spike: serve an artifact (or synthetic constraint) exceeding RAM through the deepseek4 mmap/residency seam (residency.rs:75-120) and measure page-in behavior + decode floor. Runs when a >DRAM model is targeted. |
| Elastic pool rebuild control plane | OPEN HYPOTHESIS (honestly NOT covered in hf2q's idiom today — scheduler high-water is monotonic by design, scheduler.rs:1051) | Not justified by current single-grant serving. | Spike: telemetry on pool-pressure / hot-swap contention events. If a KV-grant resize is built, crib FreeToken's *ordering* (fit-check before destructive free; rollback on failure), not its mechanism. |
| bf16 chunk-pipeline state capture | FALSIFIED (for production capture) | The experimental FLA-style chunk path materializes per-chunk states like FreeToken's, but in bf16 — restore is not byte-identical (bf16 accumulation already failed the W-5b.3 walk-bar at pp4096). | An exact-F32 side channel is a legitimate new hypothesis, under its own ADR. |
| Dense 64-token stride anchors | OPEN HYPOTHESIS | See A.7. FreeToken itself only keeps the deepest crossed boundary per forward. | A.8 divergence-position histograms; decision ladder in A.7. |
| FTW weight format / graph capture | Not needed | hf2q already loads role-aware zero-copy GGUF into Metal buffers; mm_id grids are routing-independent so recorded-CB replay is routing-safe. | — |

## Sequencing & coordination (per AGENTS.md, which supersedes older session notes)

- All work in worktree branches under `/opt/hf2q-worktrees/`, PR to main; `/opt/hf2q` stays on main. Merges to main are **never blocked by gate runs** (identity checks assert main *ancestry*, not tip — `hf2q/release/protocol`); compile-quiet applies only during model-gate windows.
- Cross-platform coordination through Ruflo memory namespace `coordination`: check `hf2q/release/status` before heavy box work; `hf2q/release/driver` owns release dispatch; never two writers in one worktree.
- Collision map vs the active qwen38 universal-quant/multi-slot lane (which serves through qwen35 infra):
  - HIGH — the engine_qwen35.rs prefill-chunker region (`qwen35_next_prefill_end` :3635-3648, emission :3966-3985) and the engine.rs transaction-ceiling logic (16729-16749): `35e42b28` already widened single-slot to 4,096; the Lane A invariant *transaction end == stable boundary* must survive every further widening. Co-review any change to either.
  - MEDIUM-HIGH — anchor payload carries `spec`/`mtp_current_len`; `HF2Q_QWEN_SPECULATION` defaults to `auto` at HEAD (qwen35_speculation.rs:13-14, :55-57). Payload-shape changes co-review with the speculation lane.
  - MEDIUM (registry tier only) / LOW (anchor store) — KV-packing/quant changes: state-only anchors are packing-independent (cursors + F32 DeltaNet state).
  - Safe in parallel: kv_persist/*, handlers.rs boundary oracle, load_info.rs accounting; Lane A touches no forward/spec/MTP kernel code.
- Order: Lane C + Lane A.1–A.4 (one branch, spike-first per the kata) → Lane A gates → A.6 parity branches → Lane B.0 spike branch (may run concurrently with A.6) → B.1 only after B.0 passes. Release-scope window for the first merge: next open minor after 0.1.10 (confirm with `hf2q/release/status`).

## Acceptance (the ADR is DONE when)

- [ ] A.1–A.4 landed for the qwen35-family engine with all six Lane A gates green on the artifacts the lane actually serves — both Qwen3.6-35B-A3B and Qwen3.8-27B (receipts linked here).
- [ ] Family coverage complete per the scope directive: gemma4 and deepseek4 anchor stores landed under the same invariants and gates.
- [ ] B.2 family-generalization spike executed for qwen35-family and gemma4 MoE prefill, with either the required aggregation phases opened or the falsification recorded with measurements.
- [ ] Payload-ownership regression (no retained parent allocations) and the preflight-then-mutate `restore_slot_anchor` refactor landed.
- [ ] A.2 lineage regression and the new SlotAware divergence gate exist in `scripts/` and fail closed.
- [ ] Telemetry (A.8) emitting in production logs.
- [ ] Lane C doc corrections merged; budget hazards + registry end-state documented in operating-kv-cache.md; ADR-017-per-family-status row updated.
- [ ] B.0 spike verdict recorded here (pass → B.1 executed with contracts; fail → linked successor ADR).
- [ ] This ADR's Status flipped to Implemented (or Superseded-in-part with links), with Updated stamps at each landing.

## Consequences

### Positive
- Same-slot context edits stop going cold: restore from the deepest surviving anchor at ~63 MiB/anchor and ~2–4 ms capture, no kernel changes, no new GPU memory class.
- The Mixed-phase F-payment reduction attacks the measured 35 s tool-result failure mode at its actual location (projected 75 s → 31–46 s aggregate prefill work), instead of optimizing the already-working pure-prefill wave.
- The lineage/publication state machine, exact reclaimable accounting, and strict conservation audits raise the whole cache subsystem's verifiability — mutation-testable at AnchorStore scale before any shared-store ambitions.
- Falsified paths carry their evidence and open hypotheses carry their named deciding spikes, so future sessions extend the data instead of re-litigating from scratch.

### Negative
- Up to ~1 GiB (shipping config) / ~2 GiB (8-slot ceiling) of host RAM held in anchors; bounded and reclaimable, but real.
- Anchor lifecycle adds state-machine complexity to the slot worker (three-state publication, epoch checks, descendant pruning) — the price of multi-depth reuse on a mutable KV log.
- Lane B carries genuine schedule risk: if B.0 fails, the Mixed-phase win needs a scheduler redesign (separate ADR), and the tool-result failure mode keeps its current mitigation only.

### Neutral
- FreeToken's remaining machinery stays un-imported; this ADR records why, which is itself a decision.
- The SerialFifo registry tier remains active for SerialFifo restore/hydrate but unused by the SlotAware scheduler; documented as the restart-hydrate mechanism pending the A.7 follow-on.

## Links
- ADR-040 (continuous batching; slot-aware scheduler substrate) — Status 🟢 full-context three-family workload served.
- ADR-042 (DeepSeek-V4-Flash; slice economics, cooperative cohort, receipt/thermal regime, product ceilings) — Accepted; Lane B lands inside its contract regime and must update it in the same work.
- ADR-017-per-family-status (KV persist family hooks) — Lane C updates its row.
- ADR-044 (qwen38 native; speculation default) — collision-map counterpart.
- FreeToken: arXiv 2608.16157; reference checkout /opt/freetoken (read-only).
- Review artifacts: Kimi K3 and gpt-5.6-sol full reviews in the research session transcript (2026-08-22); both verdicts and all MUST-FIX items incorporated above.
