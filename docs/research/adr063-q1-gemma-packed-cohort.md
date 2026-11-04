# Q1 Gemma T2 packed-cohort source adapter (prospective)

This extends the reviewed ordinary endpoint slice with one separate ignored
T2 filter. It is not T6 cancellation or a new serving/scheduling policy. No
model execution, numerical acceptance or F2 physical certificate is asserted.

The existing production `forward_prefill_batched_multi_seq` concatenates four
independent128-token inputs, installs their actual slot views and sequence
length/offset/start descriptor, and executes one512-row body. Its attention
uses the existing per-sequence isolation paths. The diagnostic head extension
now requires that exact descriptor to join a frozen four-sequence map: four
distinct slots, each start0/length128, contiguous offsets0/128/256/384. It maps
all512 actual hidden rows back to (agent,absolute token), rejects a lone512-row
sequence masquerading as four agents, and retains exactly the original four
last-row tokens/logit returns. Extra head rows are captured after their existing
checked finishes; they are not additional logical inputs or a performance run.

The test-only engine child reuses `Gemma4KvGuard`, the production prefill API,
and `commit_gemma4_slot_cursors`; cursor publication follows successful checked
prefill, not merely dimensions or submitted work. Decode uses the existing
production-default batched body with four actual token/slot/position arrays.
It respects `HF2Q_BATCHED_BODY=0` by refusing this axis rather than overriding
it. It preserves the existing fused-head eligibility or existing separate
`lm_head_batched` fallback. No flag is changed. It reuses the production cursor
validation/publication helpers, and checked body/head output is downloaded as
four actual vocabulary rows. Capture identifies these downloaded CPU Vec rows
as CPU parents, not invented GPU resource identities.

These are source-specific operator statements: prefill token-wise projections
and FFN consume the packed stream; attention preserves sequence boundaries;
normal prefill heads are per-sequence scalar heads; decode has an actual four
input row body and selected fused-or-batched head. This does not label every
inner kernel B4, infer native membership from a command-buffer union, or replace
the separately required F2 native witness/certification. Actual native kernel
selection and empirical execution remain unqualified until captured.

The same full144-per-agent logits and active Hybrid cache manifests at128/144
are required. Fixed input vectors, context262144, output8192, Hybrid dtype/
geometry, ownership and resource checks are unchanged. All four slots must be
fresh before prefill, every decode input position must match the real cursor,
and final cache capture requires144. The separate test filter refuses an
ordinary-axis packet and the ordinary filter refuses T2. T5/T6 remain missing
actual Gemma producers. All original endpoint/profile results remain unchanged.

New hosted controls exercise the actual packed-owner geometry parser and the
engine adapter's distinct-slot guard. Existing ordinary scope controls retain
multi-sequence mismatch and omitted-row refusals. Root must compile and execute
those controls before preparing a guarded actual Gemma T2 case. No CPU oracle
or new numeric tolerance is introduced; backend findings remain separate from
oracle completeness and independent task quality.

## Four-lane stage localizer (Q1-GEMMA-T2LOC, #652)

Two test filters in `gemma_actual` run the T0 and T2 bodies above unchanged,
with the same endpoint cache/logit capture and routing witness as the b3
stores, and add a nested stage store at `output/stages`:
`q1_gemma_actual_all128_stage_localizer` (`T0ColdSerial`, the All128
reference for agents 0–3) and `q1_gemma_actual_cohort_stage_localizer`
(`T2ActualCohort`). Both read `HF2Q_Q1_GEMMA_ENDPOINT_PACKET`, refuse an
ordinary endpoint packet, and the endpoint filters refuse theirs.

- **Packed prefill.** The per-layer prefill terminal receives the exact
  packed descriptor and maps every one of the 512 rows at `lane*128` through
  `gemma_endpoint::packed_owners` to `(agent, position)`.
- **Batched decode.** New `#[cfg(test)]` hooks in `batched_body.rs`
  (`gemma_stage::capture_batched`) capture the single-row decode roles (Norm,
  Query, Key, RopeQuery, RopeKey, Value, Attention, FfnNorm, router
  logits/ids/weights, LayerOutput) per actual `[N, stride]` row. Row `i`
  belongs to slot `i`, which the Q1 cohort binds to lane `i`.
- **All128 reference.** A serial-lane lease names the lane of each
  single-sequence prefill and single-row decode, so the existing single-row
  hooks label rows by lane. They serve the All128 route only.

Targets are owned-row templates. A row is stored only at the owner the source
names, and a fixed-lane path cannot capture them. One route-agnostic manifest
(`gemma_cohort_localizer::manifest`: layers × prefill roles × 4 lanes × rows
0–127, then decode positions 128–143) is frozen in both packets. The two stage
stores therefore compare coordinate for coordinate. Both routes share one
separate composition key (`gemma-four-lane-stage-localizer-native-routing1511-v1`).

The metadata prepare's optional `cohort_localizer` block (`layers`,
`prefill_stages`, `decode_stages`, the required `observer_effect_reference`
and `packed_observer_effect_reference`, e.g. `gemma-t0-b3` / `gemma-t2-b3`, or
`gemma-t0-b5g` for the #665 branch P reference; optional `mode` and
`packet_form`) writes `cohort-localizer-prepared.json` and two disabled
`multistore` packets (`all128`, `packed`). Each packet echoes its
`observer_effect_reference`, and the localizer's `execution.json` records it
with `route` (the recipe route label), `localizer_mode`,
`stage_extent_source` and `routing_witness_installed`. The supervisor's
`multistore` kind with check `gemma_loc4` validates both stores, all four
lanes and the four-lane policy (`lanes: 4`, `stage_payload_upper_bytes`). In
the supervisor's manifest form the bound manifest is the extent of record: the
packet carries no `expected_stores`, and any `lanes`/`host_payload_limit` it
keeps must equal the manifest's (`docs/research/adr063-q1-spec-design.md`
Part 3).

Packet forms: `Inline` (default, #652) freezes both store lists in the
packet's `expected_stores` (16 MiB read). `PlanOnly` writes a
`cohort-localizer-manifest-<route>.json` per route, exactly
`{lanes, host_payload_limit, expected_stores}` (≤ 64 MiB), and the packet
binds it as `stage_manifest` `{path, sha256}` instead (packet ≤ 1 MiB). In both
forms the stage list must equal the source `manifest()`.

`DriftAll128` mode (#665 §2.6, build-drift localization): All128 only,
witness-OFF (packet `routing_witness` null, witness never installed), prefill
only (`decode_stages: []`), `observer_effect_reference` null, own route
`gemma-all128-stage-drift-witness-off-v1` and so its own composition key. Only
this mode skips the published-witness native check; the identity native must
still be the build's own `Cargo.toml` pin.

Observer-effect control: `exact` of each localizer endpoint store against its
`observer_effect_reference` store must be `ExactEqual` (DriftAll128 instead
gates row 127 offline against the preparation's retained logits). The stage
localization record is `scripts/four_agent/q1_stage_localization.py`
(per-lane first divergence in execution order, router-id changes, and the rows
29–127 repetition with its magnitude). Packed versus All128 is Class B, so a
localized stage implies no fix.
