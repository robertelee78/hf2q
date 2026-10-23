# ADR-063 F3 source audit — 2026-10-23

Scope: reconcile six historical ADRs with current source and primary references.
This is a documentation audit, not a new model benchmark, design approval, or
release receipt. Historical measurements keep their original dates and scope.

Source identities:

- hf2q: `5b385d4ff580cc2c9cc70cdb425809b0c64dff63` (audit baseline).
- mlx-native: `f92eb020c4d3f821700e648fbce15d6cf75c2cc6`, version 0.15.1.
- hf2q `Cargo.toml` pins `=0.15.1`; `Cargo.lock` records registry source and
  checksum `76ce4c8d5773c72554a98020aadd330e566792dd27f843451ac5dd567bb6b5dd`.
  Native source inspection is not a substitute for clean registry-artifact gates.

## Claim-to-evidence table

Paths and symbols below refer to those exact source identities. Native paths refer to the owned mlx-native repository; hf2q paths are relative
to this repository.

| ADR / claim | Evidence inspected | Disposition |
|---|---|---|
| [010](../adr/ADR-010-exact-batched-kernel-parity.md): router-only exactness fixes the expert swap | Its own 2026-04-16 F64 reconciliation: router accurate on its own input, input already differs, expert swap remains under F64 | Withdraw router-only implementation target. Investigate the upstream attention/norm/residual chain. Input relative error cannot be compared directly with an absolute logit gap. Smoothing cannot count as exact model parity. Preserve failed experiments. |
| [011](../adr/ADR-011-flash-attn-prefill.md): tiled attention/block skipping remain new work | `src/serve/forward_prefill_batched.rs`: D256/D512 f16/bf16 `*_with_blk` and resume calls; native `src/ops/flash_attn_prefill.rs`, `flash_attn_prefill_d512.rs`, `flash_attn_prefill_blk.rs` and associated shaders | Already implemented surfaces; audit shape/path selection before a replacement spike. Block 0 skips KV tile, 2 skips mask addition, 1 runs masked. No blanket family dispatch or new speed claim. |
| [019](../adr/ADR-019-mlx-native-encoder-architecture.md): encoder architecture starts from scratch | Native `src/encoder_session.rs`: session commits/reset/fences; `src/device.rs`: `encoder_session`; hf2q `src/inference/models/qwen35/forward_gpu.rs`: `layer_session`, layer encoder borrowing | Existing phase/session machinery is the starting point (session opt-in via `HF2Q_ENCODER_SESSION=1`, multi-row only). Output-head fusion and arena DN stage-A consolidation already exist; non-arena paths differ. Remaining D3/D4 performance choice stays Proposed; scratch/residency/CPU-read lifetime boundaries require individual proofs. |
| 019: Objective-C lifetime patch awaits native publication | Native `src/encoder.rs`: `new_with_residency`, compute encoder creation, `reset_for_next_stage` command-buffer rotation use local autorelease pools and retained handles; hf2q dependency pin/lock | Correction is present in published 0.15.1. Publication-pending August wording is historical; no claim that all later performance gates passed. |
| [037](../adr/ADR-037-tree-decoding-design.md): EAGLE-2/HASS/Hydra are one interchangeable lineage | [EAGLE-2](https://arxiv.org/abs/2406.16858), [EAGLE-3](https://arxiv.org/abs/2503.01840); hf2q `src/inference/spec_decode/eagle3/dynamic_tree.rs`: pending `BinaryHeap`, global best-first loop | Correct attribution/chronology; local confidence-guided builder is not paper speed evidence. Preserve historical implementation/test receipts and distinguish pending qualification from already shipped wiring. |
| [039](../adr/ADR-039-hass-hydra-speculative-decoding.md): HASS extends EAGLE-3, supplies an ensemble, fits its schema and compounds with Hydra | [HASS primary paper](https://arxiv.org/html/2408.15766v3), §3; [Hydra primary paper](https://arxiv.org/html/2402.05109v2), §3/3.1 and Appendix A.2 | Withdraw false schema, topology, loss and compounded-gain assumptions. Retain research intent with artifact/graph/verification/paired-measurement deciding spikes. No implementation approval. |
| [049](../adr/ADR-049-agentic-state-reuse-and-mixed-cooperative-prefill.md): token-ledger length equals cache cursor under graft | `src/serve/api/engine_qwen35.rs`: `Qwen35PrefillState::graft_len`, capacity/pre-fill position offsets; `src/serve/api/engine.rs`: Gemma graft position arithmetic; `src/serve/api/engine_deepseek4.rs`: compressed graft binding | Require explicit logical-to-physical translation and graft identity in anchor matching, restore, invalidation and capacity. Token equality alone is insufficient. |
| 049: DeepSeek needs persistence built from a Qwen-like anchor | `src/serve/kv_persist/families/deepseek4_anchor.rs`: `Deepseek4DiskPersistor`; `src/serve/api/engine_deepseek4.rs`: disk persistor/session seams and rejection of combined graft+persistence; `src/inference/models/deepseek4/cache/anchor_image.rs`: anchor image contract | Typed recovery-anchor disk persistence already exists (ADR-062); multi-anchor policy remains separate. Preserve current unsupported-combination rejection. |
| 049: same wrapper permits family-neutral state restore | `src/inference/models/qwen35/kv_cache.rs`: `snapshot_slot_anchor`/`restore_slot_anchor`, attention/linear/MTP validation; `src/serve/api/engine.rs`: `Gemma4PromptAnchor`; DeepSeek anchor image implementation | Family payloads and validity predicates remain explicit. New multi-anchor preflight must preserve model/config/template/prefix/slot/lineage/graft and required sidecars. No claim that all these new checks are already implemented. |
| 049: Mixed cooperative prefill is already available / budget is 4096 | `src/serve/api/engine.rs`: `deepseek4_mixed_work_budget`, `advance_deepseek4_prefill_quantum` early serial branch for `Some` window cap; `src/inference/models/deepseek4/verifier_forward.rs`: `MAX_COOPERATIVE_PREFILL_ROWS = 2_048` | Mixed aggregation remains Proposed; 4096 is an experiment requiring new memory and latency proof. Historical timing estimates are not current receipts. |

## Independent-review correction

The independent reviewer identified an additional unsupported ADR-037 §1 claim:
EAGLE-3 v3 Table 5 does not establish an acceptance floor at 4K+ context.
[Its §4.4/Table 5](https://arxiv.org/html/2503.01840v3) concerns throughput across
batch sizes. The numerical acceptance citation and guaranteed closure/training
claims are withdrawn. The historical local 82%→60% acceptance failure remains;
long-context improvement is now explicitly a hypothesis with a matched
acceptance/cost/quality measurement that decides it.

## Validation and remaining limits

All six committed ADRs were inspected with the installed ADR inspector before
editing. ADR-049's unbolded metadata was not recognized with `--require-status`;
its declared Proposed status and dates are now in recognized Markdown metadata.
The working-tree inspector is required again before handoff. No builds or real
models are needed to establish these documentation corrections, and none were
run by this audit. There is no new correctness, quality or speed measurement.

The HASS/Hydra replacement design, artifact availability for the four target
models, benefit of deeper anchors, Mixed cohort economics and D3/D4 cutover
remain unproven. ADR-063 requires their deciding gates; this audit does not
silently approve them. The FreeToken provenance and August projections in
ADR-049 were outside this named-claim audit and remain historical attribution,
not independently reproduced evidence.

Tracker scope: read-only enumeration of all issue bodies found #358 as the
issue mentioning these six ADRs, sourced to ADR-063. Existing managed source
bindings must additionally be checked by the integration owner, and every
changed bound source extracted/reconciled at its immutable landing commit
before handoff. No issue/source record was hand-edited by this audit.
