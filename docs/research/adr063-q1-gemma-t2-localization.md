# ADR-063 Q1-GEMMA-T2LOC (#652): Gemma packed T2 versus All128, stage localization

Date: 2026-11-04. Host: Apple M5 Max, 128 GiB. Evidence roots:
- `$C` = `~/.local/state/hf2q/campaigns/adr063-preflight-20261023/q1-closure-v1` (build, prep, packets, compositions, reviews, bindings);
- `$L` = `~/.local/state/hf2q/v0130/q1-gemma-t2loc` (outs, decisions, run log).

## Result

**The packed route first diverges from All128 at layer 0, prefill `Attention` (the SDPA output), in all four lanes and on all 128 prompt rows.**
- Everything before that point in execution order is byte-identical on both routes: layer-0 prefill Query, Key, Value, RopeQuery and RopeKey for every prompt row.
- Decode, after the prefill: decode layer 0 is exact at all 16 positions on both routes, so at layer 0 the batched decode body is bit-exact given equal inputs. Layer-1 decode Query, Key and Value are also exact. The first decode difference is layer-1 decode `Attention`, which reads the layer-1 KV cache that the packed prefill already wrote differently, so the decode body's own contribution at layer 1 cannot be separated. **Scope:** only sliding-window layers 0–1 (head dim 256) were observed in decode. The full-attention layers (5, 11, 17, 23, 29; head dim 512, a different SDPA path) were not observed in decode, and this record makes no claim about them.

Packed versus All128 is a Class B route pair, so this record implies no fix.

| Lane | First divergence (execution order) | Rows | max \|Δ\| | relative RMS |
|---|---|---|---|---|
| agent-0 | prefill, layer 0, `Attention` | 128 / 128 | 0.15625 | 0.00341 |
| agent-1 | prefill, layer 0, `Attention` | 128 / 128 | 0.15625 | 0.00341 |
| agent-2 | prefill, layer 0, `Attention` | 128 / 128 | 0.1875 | 0.00341 |
| agent-3 | prefill, layer 0, `Attention` | 128 / 128 | 0.15625 | 0.00343 |

In each lane, 5 stage groups (Query, Key, Value, RopeQuery, RopeKey at layer 0) are equal before the first difference. Relative RMS is computed over the differing rows (`q1_stage_localization.py`).

This agrees with the earlier endpoint bracket (Q1 #367): layer-0 K/V cache equal, layer-1 K different, logits different from row 0. The K/V cache is written before attention, so the layer-0 cache stays equal while the layer-0 attention output, and everything after it, differs.

### Stage table (all four lanes; prefill rows 0–127; decode positions 128–143)

| Phase | Layer | Equal | Differing (records) | max \|Δ\| | relative RMS (differing) |
|---|---|---|---|---|---|
| prefill | 0 | Query, Key, Value, RopeQuery, RopeKey (512 each); MoeOutput 512 (all zero on both routes, see below) | — | 0 | — |
| prefill | 0 | | Attention 512 | 0.1875 | 0.0034 |
| prefill | 0 | | Residual 512, FfnNorm 512, RouterLogits 512, RouterWeights 512 | 0.643 / 0.0113 / 0.0171 / 0.0034 | 0.0021 / 0.0022 / 0.0010 / 0.0013 |
| prefill | 0 | | RouterIds 7 | — (ids) | — |
| prefill | 0 | | DenseOutput 512, LayerOutput 512 | 47.7 / 4.09 | 0.0067 / 0.0063 |
| prefill | 1 | MoeOutput 512 (all zero on both routes) | every other role (512 each); RouterIds 37 | Key 2.17, Attention 0.469, DenseOutput 55.9 | 0.0037–0.0114 |
| decode | 0 | all 12 roles (64 each) | — | 0 | — |
| decode | 1 | Norm, Query, Key, RopeQuery, RopeKey, Value | Attention 64, FfnNorm 64, RouterLogits 64, RouterWeights 64, LayerOutput 64; RouterIds 1 | 0.0313 (Attention) | 0.0007–0.0020 |

Full table: `$L/decisions/stage-table.json`.

`MoeOutput` (`pf_moe_accum`) is equal on both routes only because it is all zeros in both: on the production path that buffer is not written by the per-layer terminal. It carries no information and plays no part in the localization. A future hook revision should drop it, or point it at the buffer the MoE path actually writes. The same applies to the single-lane localizer's `PREFILL_STAGES`.

### Repeated-token prefix (rows 29–127)

In all four lanes of workload842, prompt rows 29–127 hold one token (496) repeated 99 times; rows 0–28 are the real template prefix. The first divergence happens at the same stage in both segments:

| Lane | Rows 0–28: differing / max \|Δ\| / rRMS | Rows 29–127: differing / max \|Δ\| / rRMS |
|---|---|---|
| agent-0 | 29 / 0.15625 / 0.00399 | 99 / 0.125 / 0.00326 |
| agent-1 | 29 / 0.15625 / 0.00397 | 99 / 0.125 / 0.00326 |
| agent-2 | 29 / 0.1875 / 0.00399 | 99 / 0.125 / 0.00325 |
| agent-3 | 29 / 0.15625 / 0.00399 | 99 / 0.125 / 0.00327 |

So the onset does not come from the repeated tail; the magnitude in the tail is slightly lower.

### Routing

**Selection domain (routing witness, softmax before top-k).** `routing` of the localizer's all128 bank against its packed bank gives `RoutingChanged`:
- 8,914 / 17,280 rows changed (4,587 with a changed set, 4,327 order-only);
- `changed_gap_max` 0.0243;
- 17,216 rows with differing weight bits.

The decision file is byte-identical to the b3 `gemma-b3-routing-t0-t2.json` (sha256 `15f78365e58544b92a97c58d688edb6d3d3a1d695046068b969a6c5453b0cdad` for both). The witness banks reproduce the b3 routing comparison exactly.

**Stage store router-id changes** (prefill layers 0–1 plus decode). Margins are the reference's raw router-logit kth-to-(k+1)th gap. Per-row detail is in `localization-record.json`.

| Lane | id changes | set changed | layer 0 / layer 1 | smallest raw gap |
|---|---|---|---|---|
| agent-0 | 19 | 3 | 3 / 16 | 1.86e-3 |
| agent-1 | 13 | 4 | 2 / 11 | 2.69e-5 |
| agent-2 | 6 | 3 | 1 / 5 | 6.78e-4 |
| agent-3 | 7 | 4 | 1 / 6 | 1.39e-4 |

The 45 changes are 7 prefill layer 0, 37 prefill layer 1 and 1 decode layer 1, which matches the stage table.

## Observer-effect control

Every comparison was made with `q1_decision_policy.py exact`, sha256 `b152c92d…45e2`:

| Comparison | Classification | Common | Left-only / right-only / differing |
|---|---|---|---|
| loc4-all128 endpoint store vs same-build T0 (`gemma-t0-b0dac052`) | **ExactEqual** | 5,616 | 0 / 0 / 0 |
| loc4-all128 endpoint store vs `gemma-t0-b3` | **ExactEqual** | 5,616 | 0 / 0 / 0 |
| loc4-packed endpoint store vs `gemma-t2-b3` | **ExactEqual** | 5,616 | 0 / 0 / 0 |
| same-build T0 vs `gemma-t0-b3` (cross-build bridge) | **ExactEqual** | 5,616 | 0 / 0 / 0 |

So the stage hooks, the per-layer diagnostic drains, the batched-decode splits and the new build change none of the 4 × 144 logit rows and none of the active cache at 128/144. The stage stores therefore describe the same computation as the b3 T0/T2 endpoint stores.

Stage stores: `exact` all128 vs packed is `DivergedExactInvariant`, with 15,872 common records, 10,093 differing, and first differing `Attention / agent-0 / row 0 / layer 0`, as expected for Class B.

## Build of record and deviation

The ADR named b3 `91b0c382` as the build of record. Root decided on 2026-11-03 (operator: no b3 rebase) that #652 runs on its own build, the PR #675 merge commit, not shared with #665's b5g. The b3 stores are bridged by the observer-effect control above, and the re-diff of `91b0c382..b0dac052` was accepted in the loc4 acquisition review.

| Item | Value |
|---|---|
| Source | `b0dac0525373fb1171886dd937a9588d92bbf904` (PR #675 merge) |
| Test binary sha256 | `fb69efe657859b5ca2bb7bac196e20d6d4a735a08d14afc8a0ec0ed3506ed908` |
| Build receipt | `$C/build-b0dac052…/build-receipt.json`, `f0bd75063a6a07a0e2ca73cfc5b67446d430ba56ed2a449c18a725ef062924ac`, `controls_exit` 0 |
| Controls | adr063_invariance 115/0/19; q1_ 70/0/28; full non-ignored 5754/0/138; Python 26 + 21 + 4 OK |
| Native | mlx-native 0.15.16, checksum `baace413db8bc4897bdd0da698c7060dd972e56f9e9069c3149bc24ffc79562f` |
| Model | `gemma-4-26B-A4B-it-hf2q-q4_k_m.gguf`, retained sha256 `bf65b7e1…ef53` |
| Workload | workload842 (greedy144), token vectors `84276820…5034` |
| Supervisor source | `q1_endpoint_supervisor.py` `2f0efa5922e0fbd21217f3732dd9c4a3c058a7d50e482830953ebb59f1622973` (admitted, condition C1 applied: exact `completion_line`) |
| Compositions | t0 `0a115d5830b474d81328cff1c70cffa9de218bdcab243678c1d78b45a3f09f6a` (config `ed3798fa…`, = b3); loc4 `5e2f2e344ca235d4f7381fe6e72434ac22aa4fe71d32426b63660edad6f17646` (config `911fef75…`) |
| Policies | t0 v2 `28304e463b8053019de1932ae6beef1968b52681f06c61baff6d5bfa6d163f86`; loc4 v2 `66e649bf86034ac545299ce8606e2e31d0a784730eedde665b8061bff2f9d567` (`lanes` 4) |
| Final bindings / copies | gemma-t0 `a1058077fc3ee3fb90f3647418bf9c69cde0ef7dfa4c304cf698b3be91623fa7` / `3c9d3688dd942ef951367481037f77d3684c6bf3ec9f2cc30f7ec318b688047f`; gemma-loc4 `4aac50aaf44792461fbe7bfe916f8386880ad97de11ffc510010b9f2a4995b10` / `cd22130bb6a7887e2a78ec116b9ad86afb6eb7701ddd5469c8e0f4410aa54cf7` |
| Operating reviews | t0 `1c82590e6753b612892e5625cfbc283699b3ed6cc5fd04e10aabd34503806948`, loc4-all128 `99b766c284c1af3b1573e86f190813536813ae5ed4b1c9b5941628227a22cfa0`, loc4-packed `d1f9c22a240dfeb6c58a0c222bf4ab1f9fa5db9dcd1b81ab60dc867f56275619` |
| Bind-time memory | 2026-11-04T05:02:23Z, quiet host: Free + Cached 109,398.0 MiB, pressure normal (`$L/bind-reading-20261104T050223Z.json`, `78e5c8ebdd9dba9bc042b81f317fbf911bd7048b49627cc81edddfcd98707f3c`) |
| Plan | layers [0,1]; 14 prefill roles; 12 decode roles; 4 lanes; outs on internal APFS at `$L/outs` |

## Runs (HOST-LOCK held by T2Loc 05:06:23–05:14:00Z; run log `$L/run-20261104T050623Z.log`, `2f7eef1c2d9b2fff90e4607d039c63e3076b3123c9c9019f0eca3f9921900154`)

| Case | Filter | Elapsed | Records | Bytes | Peak footprint / RSS | Receipt sha256 |
|---|---|---|---|---|---|---|
| t0 | `gemma_actual::q1_gemma_actual_ordinary_endpoint_trace` | 49.4 s | 5,616 | 788,765,696 | 17.81 / 2.69 GiB | `d5038fc8cb6847ca1517485dc27ce7fbfe55a358363db12bc1bd8239632061a5` |
| loc4-all128 | `gemma_actual::q1_gemma_actual_all128_stage_localizer` | 117.6 s | 5,616 + 15,872 | 788,765,696 + 147,595,264 | 17.81 / 2.70 GiB | `674c11435296c2eecce4cba739e72a03f19705c0c048d3bed5cec1bd35b871e9` |
| loc4-packed | `gemma_actual::q1_gemma_actual_cohort_stage_localizer` | 115.0 s | 5,616 + 15,872 | 788,765,696 + 147,595,264 | 18.23 / 3.13 GiB | `9b3ce449bef7939baac2801b9d0eaae174aa01d6887f1d4ad22544905d62f876` |

- Every receipt has `case_complete: true`, and the full routing bank (4 × 144 × 30) was checked.
- Execution `stage_route`: `SerialAll128PrefillSingleRowDecode` for all128 and `PackedPrefill512OwnersBatchedDecodeRows` for packed.
- The warm kernel was `VecHybrid` throughout.
- Each case ran under the supervisor's own reviewed resource gates: preload and runtime floors, normal memory pressure, swap and compressor growth limits, and the quiet-host denylist.
- Q5 `CalibrationPass` is the verdict of the four-agent campaign runner. These Q1 diagnostic supervisor runs do not produce it, as was also true of the b3 Q1 runs.

Store shas (`trace.json`):
- t0 `44d3abb25be3b487c86c971995eb8d0b65959e18e884577c9977beafddb9f85a`;
- loc4-all128 endpoint `5596112b078e3b0cb5e68ff5a513a53edfef0953b2fc9a9cff27d878e86ae67c`, stages `16e25a8d2d8c88f72244c322d21d6601a648bd04720952501035971f262b2c19`;
- loc4-packed endpoint `4053b28d2e1c853a2bccbcf943f7316efd1204d9fcd95379d07cad9c4da8d42b`, stages `16e1d9a2050e7720837962f0f9a3983361553503f9a15eddfd514fdc507869a8`.

Decision files are in `$L/decisions/`.

Each file has a `hf2q.q1.closure-decision-record.v1` `<name>.record.json` beside it. The record holds the exact argv (`nice -n 19 python3 -I -B` running the tool by `runpy`), the tool sha, the input paths and shas, the output sha, the bindings, the expectation and the check rule.

The records were produced by `$L/sup/record_decisions.py` (`22d7a8c87cc8f098925698838b09fcae95014751c9855aa412d9d455225ad4fd`). It re-executes every decision against the internal `$L/outs` stores and refuses unless the result is byte-identical to the stored file; every result was. The decision tools are those of the build of record:
- `q1_decision_policy.py`: `b152c92d4c84080c33251aae1ba1afc9847405b737fca58af52c0d311c0945e2`;
- `q1_stage_localization.py`: `313f6155f6c499fdc4c8955805d09c931bf8885466f1049097c12ff85ff59572`;
- `stage-table.json` producer `$L/sup/stage_table.py`: `503145f814477f548759b41c38ec34ee722eb40532ec1d22a6c85681d741de76`. It imports the two tools above.

| File | sha256 | Record sha256 |
|---|---|---|
| `exact-loc4all128-t0new.json`, `exact-loc4all128-t0b3.json`, `exact-loc4packed-t2b3.json`, `exact-t0new-t0b3.json` | `07c1fd75c7dbee642c64297de65c322072ee9f5eb47951032b21a6e26055d7f2` (all four byte-identical) | `89843c0b1cb5fe327ac353de723bc143ff36e58dbe5d5c10b81a3a9cc02b12d8`, `1c56cab33743587ab0a60c2acb37ffb2eb4f3b523249bf0cdf6a74ea588a0b42`, `5fdb476e04e62977d23e9ad5237921b7be16e6778948652d1b1d0797af78aead`, `d03fdddb23b786325b436be3b60752e9984cf831657fe80cf4b5cabd61a93a71` |
| `exact-stages-all128-packed.json` | `11cfddc6f0abdb4925b5b5a2e2c76243e3f2af4d7c4adcb9ca24fd3b1c6cbb87` | `68651f06d940f5244f48de4ebe61d33f51e4bc33eec4f6bc94f08aaa94ad2224` |
| `routing-loc4all128-loc4packed.json` | `15f78365e58544b92a97c58d688edb6d3d3a1d695046068b969a6c5453b0cdad` | `53cd68fefcaf36f5ca57f03f7a2ecaff1ef34143dcfaa544975336dadf18beb1` |
| `localization-record.json` | `795f56434dffff3aa0bf384f462bd5d61b010359ce931140b1d4faff1bcf5a16` | `f1a60d3cde6d32dd0ec8f19361e1c2de34cd8fff9233f026fa3e060636b67720` |
| `stage-table.json` | `f834471ec2afa4ba7555e36ba9956b62cb1fb3237caebe5bd16ba6427e9ec3a0` | `3045183089fc92623c9ac11b8f3529c3edf3a1d973eceb9b0cecb8dda66044db` |

## Reviews

- **Done before the run (ReviewT2LocA):**
  - supervisor admission (ADMITTED with condition C1, applied);
  - both acquisition reviews (Accepted);
  - resource-policy values (accepted; placement moved to internal APFS by root decision);
  - packet derivations (attested);
  - three operating reviews.
- **Localization record and observer-effect control (story Validation): Review680, 2026-11-04, story #652 ACCEPT.** The reviewer re-derived every decision byte for byte and recomputed all 52 stage-table rows independently. Their PR findings are applied in this revision:
  - the corrected packed endpoint store sha;
  - the decode claim limited to the observed sliding layers 0–1;
  - the result ordering;
  - decision provenance records;
  - the MoeOutput row in the stage table.
