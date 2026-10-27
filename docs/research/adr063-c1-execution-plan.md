# C1 execution HLD/LLD V1 — exact anchor contract, not C2 implementation

Design approval: [recorded review](https://github.com/robertelee78/hf2q/issues/359#issuecomment-6451204642). Original acceptance, predecessors and delivery gates remain in force; approval does not mark the task complete. The authoring notes below describe preparation history.

Source design by f1-review-agent,2026-10-27. Read full issue[C1#359](https://github.com/robertelee78/hf2q/issues/359), ADR-063 §C1/C2 and ADR-049 current source audit. HFmainb0ecf434 and native85f3846 are parent-reported integrated baselines; final implementation manifest must bind exact clean source. F3#358 Done is a satisfied predecessor; Q5#371 remains open. Design can proceed now; final C1 delivery requires its predecessor and merged independently reviewed contract+boundary regression evidence. No implementation approval is implied, and no builds/tests/probes/metadata/model/HF/GitHub writes or repository edits were performed.

## HLD and deliberately small delivery

C1 specifies the common identity/lineage/publication/byte-accounting protocol and typed family validity. It does not install a multi-depth production store, change affinity, choose a new decoder, implement S3, change native kernels or deliver C2's Qwen speed/quality claims. Deliver a checked-in contract/runbook plus finite synthetic state-machine and actual family-layout regressions using existing source types/planners/helpers. Existing real-model evidence is reused only for its exact layout/operation/source; no mandatory additional full-model run merely to label this design. A regression requiring genuinely missing restore code becomes a named C2/family implementation obligation, not a fake C1 PASS for current production.

No generic cache framework or universal model payload type is required. One identity/transaction envelope may wrap different Qwen/Gemma/DS payload validators. Each family retains its existing serialization, graft restrictions, sequence layout and restore implementation until a separately reviewed implementation changes it.

## Every C1 acceptance item mapped

| Requirement | Existing source/evidence | Smallest missing contract/regression |
|---|---|---|
| Logical lengths vs physical cursors/graft; FULL/SWA/recurrent/compressed/indexer | ADR049 coordinate audit; Qwen HybridKvSlotAnchor/slot cursor state; Gemma Hybrid slot anchor; DS CachePlan/snapshot/anchor image | Exact typed component-validity table below; checked coordinate arithmetic/graft-generation mismatch and component omission negatives using real family dimensions/layout constructors |
| Committed/pending; preflight-all-then-mutate; descendants/reset/identity | DS pending_turn_anchor versus turn_anchor; Qwen/Gemma existing boundary checkpoints; registry/session identity; ADR049 lineage rule | Normative finite transition table, two-phase restore plan, preflight-no-mutation/partial-failure-reset tests. Current Qwen restore interleaves validation/mutation, so do NOT document it as already atomic |
| K committed+1pending owned bytes; right-sized hidden; memory/disk | Existing total_bytes/resident_bytes and Gemma CPU anchor quota; native shared-buffer Arc ownership; persistence families | Disjoint owner ledger with explicit aliases/full parent bounds, K excludes pending; metadata/token/logit/hidden copies; pending quota admission; idle conservation/last-owner release tests. Existing methods alone do not count every retained owner |
| A→B→C rewind/diverge/oldC; cancel; graft; isolation; idle | Existing family snapshot/restore/cancel tests and actual DS native8tail/Gemma fullinput/Qwen prefix controls | Named finite oracles below before C2 implementation, including old-C rejection after executed divergent write and different session generation |
| Never publish cancelled pending; prune executed writes; no resurrection | DS recovery selector/pending separation; current Qwen speculative rollback; known drain/poison semantics | Fault-injection transition tests before/after capture/submit/drain/ledger commit, preserve only physically valid ancestors; uncertain failure invalidates whole store. No shallower fallback after partial restore |

## LLD: identity, coordinates and validity

Immutable AnchorIdentity: model artifact+architecture/layout revision; tokenizer+template raw/effective hash; effective inference/cache/quantization configuration; session/conversation identity; slot handle generation; lineage generation; graft identity/content hash+physical length; logical token prefix digest+count; exact cache component schema. A path/model name is not identity. Sampling-only settings belong when saved logits/hidden/speculative state depend on them; include all graph-affecting defaults and representation parameters. Compatibility cannot be inferred from equal vector lengths.

Checked coordinates: logical prefix L excludes graft. Expected physical cursor is graftLen+L only for the family/component whose source defines that offset; validate actual separate component cursors rather than one generic value. Overflow/negative/capacity mismatch refuses before mutation. Graft replacement/replant increments graft/lineage generation and invalidates all incompatible anchors BEFORE any new graft write. Preserve DS explicit combined graft+persistence refusal until source supports graft-aware hydrate; no Qwen fallback.

| Family/component | Retained payload and validity proof | Unsafe shortcut forbidden |
|---|---|---|
| Q38/Q36 FULL attention | Current HybridKvSlotAnchor stores per-layer cursors, not duplicate FULLKV. Same slot generation and live rows still valid through savedcursor; physical writes since capture tracked by lineage/range. | livecursor>=savedcursor alone is not lineage proof after rewind/divergent overwrite; equal tokens alone cannot repair overwritten parents |
| Qwen recurrent DeltaNet/conv | Exact per-slot conv/recurrent bytes and ping-pong parity; source layout/dtype/slot extent. MTP/target cursor/bookkeeping snapshot only after actual accept/reject transaction settles. | Approximate recurrent replay, BF16 substitute or per-token huge LA capture arena replicated per anchor |
| Gemma FULL/SWA Hybrid/TQ | FULL append rows plus actual SWA ring rows/tags/cursor, TQ packed K/V/norm/xlen sidecars and representation identity; existing GemmaHybridSlotAnchor CPU copies/quota. Fullinput checkpoint tag includes token identity; retryP−1 semantics distinct from capture/full-growth credit. | Restoring ring cursor without bytes/tags; substituting fullattention semantics for sliding state; same-shape wrongcodec sidecar accepted |
| DS window/compression/indexer | Deepseek4CacheSnapshot exact window+compressor recurrent state; append-only compressed/indexer rows validity at snapshotposition; hash/indexer plan/ratio. Persisted anchor image additionally carries valid compressed/indexer rows and rendered ledger. Native8tail recovery and pending logits have their existing cursor contract. | Treating compressed cursor as rawwindow position, copying only mainKV while missing score/indexer state, using Qwen snapshot or claiming append-only rows survive divergent overwrite |

Current source anchors: qwen35/kv_cache.rs HybridKvSlotAnchor:1015, snapshot_slot_anchor:1553, restore_slot_anchor:1629; DS cache.rs snapshot:693/restore:766, cache/anchor_image.rs typed image validation/serialize tests; Gemma kv_cache.rs GemmaHybridSlotAnchor:1575, snapshot:1673/restore:1756; engine_deepseek4.rs pending/committed turn recovery and cancellation tests; engine_qwen35.rs MTP/target rollback transactions; engine.rs actual slot publication sites. Line numbers are reading aids, not immutable source attestation; exact hashes/call symbols required in implementation manifest.

## Normative state machine and two-phase restore

Store is slot-local: ordered committed[<=K], pending[<=1], lineage generation, accounting/quota and poisoned flag. K is committed count ONLY, default proposedK4 as ADR049 policy, not permission for arbitrary new production allocation. Pending has request/transaction ID and is invisible to affinity, equality and cancellation recovery. Reserve bounded OLDcommitted+constructing+pending owners before capture; no “K counts pending” budget shortcut.

1. Capture intent: validate identity/component schema and settled transaction, reserve full payload/token/logit/hidden/metadata capacity, construct independent snapshot. On construction failure release new owners after known drain, publish nothing.
2. Pending: capture complete+checkeddrain, record identity/cursors/lineage, still not visible. Replacing pending requires prior payload final-reader retirement; never keep an uncharged hidden old pending.
3. Request commit: only after successful cache/component/token ledger publication and terminal request success under existing family semantics, atomically transfer pending into committed, then apply bounded positional keep-newest-K policy. Failure/cancellation never makes pending visible. If existing live cache succeeded but response disconnects before this policy's publicationpoint, conservatively discard pending; don't invent successful request.
4. Restore phaseA preflight: validate ALL layer/component shapes/dtypes/sidecar presence/byte lengths/offsets/capacities, identity/graft/slot generation, retained physical rows/lineage, no open speculative transaction, all intended copies/CPU/GPU roots/temporary quota and readable known owners. Produce immutable validated restore plan; no cursor or byte mutation in this phase.
5. Restore phaseB mutate: execute exact planned writes/cursors with ordered copies, known healthy completion and ledger publication. No user-visible success before the commitpoint. Before the FIRST possible divergent KV/ring/recurrent write, tombstone every descendant whose rows can be invalidated. Conservative pruning may occur before submission even if the write later never executes; retaining an unsafe descendant is worse than losing a hit.
6. If preflight refuses, no mutation: ordinary cold/reset fallback allowed after explicit refusal. If any mutation/submit may have happened and restore fails, hard-reset/poison affected slot and clear ENTIRE store; never try a shallower anchor in partially mutated state. Uncertain drain quarantines owners and fails the worker/cohort per existing policy, not a fake free/reset over live commands.
7. Cancellation: discard pending; prune descendants of executed writes and every committed anchor whose required regions are no longer physically valid. Keep a committed ancestor only if each component's required bytes/rows still valid after known recovery; not “committedlist unchanged” unconditionally. Reset/poison/model/graft/session generation change clears all incompatible anchors.

**Current gap is explicit:** Qwen restore_slot_anchor rewinds FULL cursors inside validation loop, then checks MTP and copies linear states, with later fallible operations. Existing source is not preflight-all-then-mutate. C1 freezes this required protocol and a rejecting regression specification; C2/family integration implements/refactors the actual restore boundary with whole-store reset on failure. Do not silently land production restore changes under the C1 design deliverable or mark current path atomic. Gemma and DS restore bodies likewise need concrete all-component preflight audit; similar wrappers are not proof.

## Owner accounting and promotion policy

For each accepted source/profile compute unique retained owners: committed K +one pending +construction temporary/old pending coexistence +saved logits/hidden/token ledgers/metadata, not nominal tensor view length. Shared parent charged once with ownerID/extent and alias list; independent CPU/GPU copies separately. A borrowed live FULLKV pin is not a reclaimable snapshot copy: count cache capacity once and separately describe anchor's validity dependency. Eviction credit includes ONLY dedicated payload roots/CPUowners actually released at final observer/command owner, never the liveKV backing merely because the logicalanchor is removed.

Qwen/Gemma hidden/logit snapshots must be host-owned or dedicated right-sized allocations. A20KiB view retaining a40/80MiB residual parent is charged at fullparent and fails the right-sized payload contract; don't hide it as metadata. Native MlxBuffer shared Arc/root lifetime already exists; no new kernel/API is justified solely for this policy. Existing total_bytes may count only recurrent CPUvecs, not token/logit/owner metadata, so use a separate explicit contract ledger rather than relabel it comprehensive.

At idle after checked release and no command/persistence reader, verify conservation with disjoint quantities: grant = livecache/prefix capacity +uniqueanchor-owned payload +other reserved cells +free; count livecache/prefix aliases ONCE. Report per-slot and global totals, pendingconstructing0, and no leaked quota after eviction/reset. This is declared owner/quota conservation, not OSresident-page equality or final driver-release proof. Positive retirement/opaque allowance remains from Q5; stale allocator history is not refundauthority.

Disk tier is optional promotion, not another invisible committedanchor. Existing Qwen/Gemma/DS family persistence stays typed. Promote only immutable committed payload after publication, with versioned identity/checksum and atomic temp→rename; pending/cancelled/poisoned payload never hydrates. Async writer retains sourcepayload until completion; charge extra buffer/encoding/I/O/queuedwrite and boundqueue/cancel/cleanup. Disk promotion does not free in-memory credit before final reader retires. Hydrate preflights full family identity/sidecars before mutation and charges decoded/newCPU/GPU coexistence. C1 defines this contract; it does not enable a new backgroundtier or expand current unsupported combinations.

## Finite regression/oracle plan before C2

Implement small contract tests plus real-family layout regressions, not a second serving runtime. Proposed files: focused contract document `docs/research/adr063-anchor-state-contract.md`; small test-only transition fixtures near existing cache tests; no production AnchorStore/affinity change. Exact manifest/filters/source-dependent helpers to be frozen after independent review. A tiny typed test-state machine can check specification transitions, but alone is not proof production follows it; distinguish normative-controlPASS from actual-source integrationGap.

1. Identity/coordinates: exact boundary L0/L1/fullcapacity; graft0/nonzero/changedhash; checked overflow/onepast; same modelname/differentartifact or cachecodec/template/session generation refuses beforemutation.
2. Component preflight: omit last required sidecar or corrupt lastlayer shape after earlier layers validate; normative preflight must preserve every original cursor/byte. Actual currently non-atomic restore gets explicit failing/known-gap disposition until C2, not a deliberately failing test merged intoCI.
3. A→B→C lineage: restoreA, execute divergentX write, then oldCrequest; oldB/C rejected, validA allowed only if physical family component stillvalid. Compare all claimed retained components to cold fixture; use distinct bytes so wrongbranch cannot pass by equalzeros.
4. Pending lifecycle: cancel beforecapture/duringconstruction/aftercapture/beforecommit; no pending publication. Successful commit exactlyonce; repeatedcompletioncannotappendtwice. Failedprefill/copy/drain never resurrects a tombstoned descendant.
5. Rollback/recovery: before-write refusal preserves validcommitted; known partialwrite failure resetswhole store; unknown drain quarantines/fails closed. Cancellation preserving a valid ancestor versus window/recurrent overwrite destroying it are separate cases.
6. Graft replant/sessionisolation: same tokens under new graft/slotgeneration/conversation cannotreuse staleanchor. Four distinct conversations and permutedslotindices must retain independentlineage/rows.
7. Byte ownership: Kcommitted+onepending (K4 fixtures), constructingpeak, aliasparentdedup, right-sizedhiddennegative, no credit until lastowner, diskreader prolongsownership, eviction/reset idleconservation. Check both dropping aliases and independentcopies; no OSphysicalcredit inferred.
8. Family layout: Qwen dense andMoE real cache config/slot strides including recurrent+MTPpresence/parity and append-onlyFULL dependency; Gemma actual FULL/SWA/TQnorm/xlen/ringtag layouts and CPUquota fixture (existing tinyCPUcopy/restoreretry proof reusableonlyexactsource); DS actual CachePlan ratio0/4/128 window+compressor/indexer/score-state and anchor-image schema/validrowgeometry. Use the actual plan/type constructors and existing serialization/copy helpers with small legal geometry where supported. If productionhelper needsGPU, separate ignored bounded actual operation only when necessary and under rootpacket, not fabricated CPUreimplementation. No new fullGGUF/F32reload required to test layout arithmetic.

Every case records expected state/owner count/component-byte result and deliberate negative control. Missing actual familyvalidator cannot be called covered by the normative mock; maintain coverage table `normative tested / actuallayout tested / productionintegration pendingC2`. Real source callsites already have functional/cancel/prefix tests; reuse exact matched receipts rather than rerun models for a design label. C1 acceptance still needs its explicit boundary regressions actually executed and independently reviewed; this document itself has no such PASS.

## Evidence, dependencies and routine approval

Reusable: ADR049F3 corrected design/history; existing Qwen slotanchor and rollback source tests, DS cache snapshot/anchorimage serialize/recovery tests and actualnative8tail cancel/growth proofs, Gemma Hybrid/fullinput CPUcopy+realmodel checkedrestore evidence, published native buffer ownership/copy tests. Bind each exact source/shape/receipt; no aggregate unit count as a substitute. Current Q36 E0282, Q38worker/Gemma3600failures remain unrelated failedquality/stress evidence, not repaired by anchor-contract tests.

Remaining decisions before source implementation handoff: root/independent approval of normative publicationpoint peractualrequestfamily, exact layouttest manifest and K/bytequota testvalues, safe preflight-failure injection seam, explicit normative-vsproductiongap status, and optionaldiskpromotion scope(defaultno newpromotion). These are finite policies inside userauthorizedprogram, not a reason for genericnewuserpermission. Root may approve reviewed colleague work; no self-acceptance. Final C1 delivery waits Q5, independent boundary/layout proof and merged contract/evidence. It need not wait C2multi-anchor speedimplementation or all backendqualityrepairs. Native productionchange/publication is unnecessary unless a genuine missing bufferprimitive is identified separately.

Realistic closure: a focused contract+layout/state test component can be source-authored/reviewed while hardwarebusy; exact build/test time and any required tinydevicecontrol remain rootserialized. Do not promise a fourth daily task closed until all stated boundary/layout evidence and predecessorQ5 are complete. Do not hide missing atomicrestoreimplementation by closing C2 underC1's design-onlylabel.

## Read source hash inventory

`docs/adr/ADR-063-apple-silicon-four-agent-speed-coherence.md` SHA256 `e2f6c034c58f216a125f500dfca26af0a82c588263c53508405ed3b3a9f57b7b`

`src/inference/models/qwen35/kv_cache.rs` SHA256 `fbc78999342721bd64f9dbf79fa8c1b3bf2b4cea9cb884aebb5b8c0679818256`

`src/inference/models/deepseek4/cache.rs` SHA256 `3eddfb2cf49b48290e5fa7cf4d32b3f1a54489c7cc1c10b35ab21c199412eb42`

`src/inference/models/deepseek4/cache/anchor_image.rs` SHA256 `6ebeb5d7257eb834e7486542cc7bbcc4552d2e6e8e3251f7e0afd0ba30e8e0d4`

`src/inference/models/gemma4/kv_cache.rs` SHA256 `810f7558f9f055575006427bf272083f756dbdfdb4f83973b741c09755fac27f`

`src/inference/models/gemma4/anchor_budget.rs` SHA256 `f859798ab4655d77e94cd142b8d7373b06080e3dfc2b4567453bc03d2efcb307`

`src/serve/api/engine_deepseek4.rs` SHA256 `aa15a4ff96f7159827d18818c50016b739290b2008513ce5f50b4d64bfb1e4a5`

`src/serve/api/engine_qwen35.rs` SHA256 `39eef94e81b911bd7f8b00a1d50e45ca955bcc3bd06cbe2c7eaedd9dd8888a3a`

