# Gemma4 SlotAware cache scaffold checked plan

Updated2026-10-24. Extraction source basis HF017ac49d5c608a86ef448f6695dbb5a39e680d70, composed onto HF77db6a24bf80169620c28ad17b881b2c9b09768c with reviewed Gemma bytes preserved. Published native0.15.2 dependency is unchanged. Validation is below; no full-context allocation is authorized by this note.

The actual SlotAware constructor now consumes a checked phase/layer/buffer plan. Four standalone allocator signatures use the same slot-descriptor constructors and allocation traversal. Descriptors carry actual dtype, shape, requested bytes, initialization and allocation error labels. Engine and pure fault controls share the phase-publication traversal; this is not a parallel estimate-only formula. Original scalar-loader cache, activations, prefill, head, snapshots and retired/native owners are outside this plan.

Invocation modes are resolved once under the declared stable-process-environment precondition. Existing parsers/defaults/acknowledgment policy remain; standalone hybrid calls resolve their own direct FULL_F16_KV/xlen flags without triggering unrelated global investigation configuration. All enabled phases preflight before allocation/publication. Invalid later geometry/arithmetic now fails with zero allocations/assignments: this deliberate fail-closed change differs from legacy invalid-input partial publication. For a valid plan, allocation failure preserves prior completed phase publication; disabled phases do not assign or clear old fields, enabled empty phases publish Some(empty). Newly allocated hybrid xlen-off fields remain None.

Allocation order is ALWAYS HB scaffold, optional hybrid, optional dense, optional nibble. Within each phase layers retain actual weights-layer order and config geometry. HB Kpacked/Knorm/Vpacked/Vnorm, hybrid K/V/Vnorm followed by optional xlenK/V, dense K/V and nibble Kpacked/Knorm/Vpacked/Vnorm are unchanged. Norm count stays floor(D/256).max(1), nibble oddD remains rejected, F16V uses a shared4B dummy norm. Sliding rings use zeroed storage; full layers remain cursor-bound overwrite, except dummy norms ordinary zeroed storage. Logical full context262144 retains physical262145 guard capacity; saturating guard arithmetic is preserved.

Selected F1 Gemma artifact declared SHA bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53 has30layers:25 sliding8KV/D256/window1024 and5 global2KV/D512/context262144, four slots. Default HB and hybrid allocations coexist:

| Requested bytes | Sliding25 | Full5 | Total / roots |
| --- | ---: | ---: | ---: |
| HB always |425984000|10905232000|11331216000 /120 |
| Hybrid default |632422400|16190075200|16822497600 /90 |
| Combined |1058406400|27095307200|28153713600 /210 |

These are requested payload, not measured physical memory. Loader's original120nibble roots1438187520B remain outside this table and must be separately charged. One selected scaffold plan does not establish all-shape/whole-model/OS upper bounds. No resource refunds or live-byte credits follow from phase assignment, wrapper drop or completion.

Eleven model-free pure controls cover handwritten selected geometry/totals, actual descriptor traversal, phase publication including each descriptor fault across all four phases, later invalid-phase zero-callback preflight, skipped fields, enabled empty phases, fullF16/xlen order, floor norms/nibble shapes, dense dtypes, N1/N4, actual schedule, diagnostics and overflow, and original nibble allocation error labels through the actual fault traversal. No pure test allocates Metal, constructs a model or reads model files. Existing hardware/cache isolation gates remain separate; no fullB/C/R/G, numeric cap/fit/admission, Qwen profile, native growth-gate adoption or model-speed claim follows.

## Validation

Root validation passed eleven matching pure Gemma controls and ten existing
Qwen allocation-plan controls on the composed source. Locked all-target / all-feature
check passed (29.49s), and the locked release build passed (38.41s). No Metal,
model, tensor-payload or full-context execution was performed. Independent source
review accepted the frozen descriptor, allocator and publication behavior.

The initial compile stopped on the superseded private allocator helper being
unused under denied warnings; no tests or hardware ran in that attempt. Removing
that unreachable helper and preserving its safety note in the actual shared
allocator resolved the error. The corrected source was reviewed before these
passing checks. Existing real cache/isolation and model gates were not run.
