# ADR-063 Q1-GEMMA-SPEC (#665): Gemma n-gram T5 evidence on b6, with branch-N verifier-route localization

Design: `docs/research/adr063-q1-spec-design.md` Part 2 (§2.3–§2.7). Numeric policy: `docs/research/adr063-q1-numeric-policy.md`. Companion record for #653: `docs/research/adr063-q1-qwen36-spec-evidence.md`.
Paths: `$E` = `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`, `$C` = `$E/q1-closure-v1`, `$L` = `/Users/robert/.local/state/hf2q/v0130/q1-spec`.
Every number below is read from the cited file; sha256 values are of that file.

## Result

**Verdict: RouteSensitive; localized: on all four lanes the first divergent mapped stage group is verify-committed / layer 0 / `Query` / rows 128–130** (max_abs 0.353 / 0.354 / 0.381 / 0.232). It follows one equal group, and the Logits head rule does not fire.

| Decision | Class | Verdict | Evidence |
|---|---|---|---|
| G5-A1 pre-fork state | A | **ExactEqual**, 4204 records | `$L/decisions/G5-A1.json` |
| G5-A2 integer decisions and transaction | A | **Pass**: in-process checks + supervisor `gemma_t5_ordinary` / `gemma_t5_ngram` | receipts below |
| G5-B committed rows | B | **RouteSensitive**: 64 rows, top1 61, 3 flips, 22 rows over τ | `$L/decisions/G5-B.json` |
| cache@144 | B | ExplicitInconclusiveNumericPolicy (recorded, not gated) | `$L/decisions/G5-cache144-compare.{json,log}` |
| routing (mapped) | — | **RoutingChanged**, in verify-committed and continuation only | `$L/decisions/G5-routing-mapped.json` |
| observer-effect controls (pass 1, pass 2) | A | **All ExactEqual / ExactRouting** | `$L/decisions/OBS-*.json`, `OBS2-*.json` |
| verifier-route localization | — | **Localized** (pass 2): verify-committed / L0 / `Query` / rows 128–130, all lanes | `$L/decisions/locN-localization-record.json` |

**Scope of the verdict (design §2.4):**
- Class B: a localized stage implies no fix. It records where the two routes first part, and it is not a defect.
- No serving consequence: Gemma serving does not speculate. The n-gram path is reachable only through `hf2q generate` with `HF2Q_SPEC_NGRAM=1`, and the transaction drafts are an injected diagnostic fixture (`injected_origin` `InjectedDiagnosticFixtureNotProductionProposalQuality`).
- The ordinary reference is the global route, so this record makes no claim about the slot-aware serving body.
- **Disclosure (design §2.4, as in #652):** the greedy144 prefix repeats token 496 in rows 29–127. Both localization records report it on every lane: `repeated_prefix` rows 29–127, 1 distinct token, modal token 496, 99 times, longest run 99. Prefix inputs are identical in both arms, and only rows 128..143 are classified.
- `whole_q1_qualified` is false in every receipt, both localization records and the cache@144 report. This record makes **no performance or physical-admission claim**.

## Build of record b6 and supervisors

- Source main `38eea8393862b16068bee23789ded41ddf853328`; mlx-native `=0.15.18`, checksum `9d10f97e0e25a00c1b4706a0ea9a838b2524b3fc2b999d05f21cd2580c524bee`.
- Test binary `$C/build-38eea8393862b16068bee23789ded41ddf853328/test-binary` `46bc533f1e3512e3599fc04b4979496d0e69031919b5af6a5dee52bfc9cc805c` (re-hashed; equals the cache@144 report identity). Build receipt `f7727e1215a0858e73238d428f482ba6fde21b362b3180b630baf93a16f4de06`, `controls_exit` 0. Relation: `$L/b6-relation-20261105T051546Z.json` `c639949f524f642cbc3f50142242c7eca0412139afe0427400c39f927e5f7e0d` (Gemma has no ADR-065 ordering constraint, design §2.7).
- **Supervisor for gemma t0, t5ord and t5ngram:** `q1_endpoint_supervisor.py` `5fdedd43d4c8d0756c9140366bced59e7982074fde29a801e5e791425a5451ff`. Admission: `$C/reviews/supervisor-5fdedd43-admission-review.md` `8ecae9f32026146ceb4e29f99f70c39d96ff7f2be5865f8fad1f2f370b8e3d83` (ADMITTED).
- **Supervisor for the four localizer cases:** `dc7b5c56` (PR #709, merged as `9e27bcdf`). Admission: `$C/reviews/supervisor-dc7b5c56-admission-review.md` `3eaa76ad656ae2bf3db26bfb72035c7330ccd8689bab476c7738d9f6b166e3ce` (ADMITTED for the #665 localizer units). Every earlier b6 unit keeps `5fdedd43`, which its bindings pin.
- Reviews 1–4, the recomposition re-reviews, the localizer acquisition coverage, the policies, the key maps and all final binds: `$C/reviews/b6-spec-review-notes.md` `ca21cb62d491c22d089863dcd510ebbbdcfcf3ba9ca6719da8ab620db963ed36`.

## Compositions, reviews and bindings

| Unit | Composition (sha256) | Acquisition review | Bindings (sha256) | Policy (sha256) |
|---|---|---|---|---|
| gemma t0 | `$C/gemma/b6/composition-v2.json` `f32789c8fa101ae859d482993572b15b6da0ff4d3ea94fb337b1bd1f4b5a3385` | `t0-acquisition-review.json` `d39c3d24308f87d29e29d4837cf2131e87ee46022eefb8e1e1a8a1580c5f7a4a` | `gemma-t0-b6-bindings.json` `810620864e651fdb5135d22d33586a84b0cc1c33395040823c0d3746dca7f5c8` | `resource-policy-gemma-t0-b6-v1.json` `404aecc9043baffc2f811edac44a0192c93f330c538b45713757ad1ffaff5c93` |
| gemma t5 (t5ord, t5ngram) | `$C/gemma/b6-t5/composition-v2.json` `e569a44ec8ec92e5e2783cb38f3c1211e1b08bb7a6a762a256b3fc6e1ec9c218` | `t5-acquisition-review-composition-v2.json` `4f35312766e4b3608c7e42c86cdbf68c087124343e8b1bf00b2e9fdf31f71866` | `gemma-t5-b6-bindings.json` `f82df4fbcbbc0cc472eeb0d2b22e6d8910e9c3b46ddaec6959afa79e89ee0aec` | `resource-policy-gemma-t5-b6-v1.json` `cc21c258769c6b6eb5faa4e0b7b951abc31663b643ada94358d34e835087a0dc` |
| t5loc-ord-p1 | `$C/gemma/b6-t5/composition-v3.json` `80289f53d095be3542832437011abe549b91d01e96bd26267828f50912dbc631` | `t5-acquisition-review.json` `bc3f92920af6b3a7fe12c94fce0fcc6524a9745a7b2fd38011012318c007fcb5` | `gemma-t5loc-n1-ord-b6-bindings.json` `c60e2e3b2c27ec46e18079e6807b395b2b350e6d75659afd4778b2ca19388077` | `…-t5loc-n1-ord-b6-v2.json` `24ea6b713b0454b96c5f13eac61062e3f2e1ac7f9a68b3482e019317d474dd5c` |
| t5loc-ngram-p1 | v3 (same) | same | `gemma-t5loc-n1-ngram-b6-bindings.json` `58d798b91adbd0bde8796c581ede457a926196fd83e17fa1a58d23c228b401a2` | `…-t5loc-n1-ngram-b6-v2.json` `f3a6af38520ef45d13a04e27806e17b989d773e9fc3390b2034de7d4e799f43c` |
| t5loc-ord-p2 | v3 (same) | same | `gemma-t5loc-n2-ord-b6-bindings.json` `2d96f0527994d08b77167ad6fa6bb39cf1a0d7013466e6e6db98f831869e2c5b` | `…-t5loc-n2-ord-b6-v1.json` `3ada80eca844a549ab9caf6d2f62304d580cb1af19b14b9335e045f98cc92eb7` |
| t5loc-ngram-p2 | v3 (same) | same | `gemma-t5loc-n2-ngram-b6-bindings.json` `6f4a04240636bc8b5616dd1dcfd1099bbac559cc2c360c16bcd766632495237a` | `…-t5loc-n2-ngram-b6-v1.json` `261e0261ddb3d917a84ceca3e90da575fb8cde314b397e13917c4817f7b5e241` |

Reviews are under `$C/gemma/reviews/b6/`, bindings under `$C/sup/`, policies under `$C/inputs/`. Every bindings file above has `final_approval` true and binds the composition and policy shown.

**Reused review path.** The t5ord/t5ngram bindings `gemma-t5-b6-bindings.json` `f82df4fb…` bind `t5-acquisition-review.json` at sha `4f35312766e4b3608c7e42c86cdbf68c087124343e8b1bf00b2e9fdf31f71866`. That path now holds the v3 review `bc3f9292…`. The bound bytes survive unchanged as `t5-acquisition-review-composition-v2.json` (`4f353127…`). The copies in the two outs (`$C/gemma-{t5ord,t5ngram}-b6/acquisition-review.json`) were re-serialized and have sha `b53e66a2bb72643cb8025a53196191a7970af5ef8a9d16a206020b52701b08b7`.

**Recompositions (ctime only).** The compositions went v1 → v2 → v3:
- t0: v1 `e4bcc3ccd449e28e7bf6a366069685e549d3df88f340fe585ee464a5ad683061` → v2 `f32789c8…3385`;
- t5: v1 `aa6db81beef1cd18262059a7e7056cc06bba14ad8059db358efed3bb3d6da123` → v2 `e569a44e…c218` → v3 `80289f53…c631`.

Each step differs from its predecessor only in the model file's ctime / ctime_ns (lines 21–22). The ctime moved when hard links to the model were added and removed for Q5 runs. Inode, size, mtime and the full-content receipt are unchanged, so the comparison key is identical. Each version was re-reviewed: the v1 reviews are retained as `t0-acquisition-review-composition-v1.json` `1a28928ed3db3d927de105a75e130a6ae65688a348f58b7ad423724ea4187b79` and `t5-acquisition-review-composition-v1.json` `1f300fef1dbfde66153803c4cedbec744b58c8e36ba9518eac40d946655f4801`, and the v2 T5 review is retained beside the v3 one (`b6-spec-review-notes.md`, "Gemma re-review on composition v2", "Gemma t5 composition v3").

**The localizer shares the T5 comparison key (disclosed).** At b6, `runner.rs` routes both filters to `t5_run::run(localizer)`. `check_config` uses the same recipe whatever the mode, and `ComparisonKey` has no localizer field. The localizer key therefore equals the gemma-t5 key, and the t5 composition and its acquisition review cover the localizer cases (review notes, "#665 branch N pass 1", item 1). This is also what the observer-effect `exact` needs: localizer T5 stores vs T5 stores under one key. The localizer cases ran on v3, which has the same key as v2.

Quiet bind readings (`$L/`): `bind-reading-gemma-20261105T052840Z.json` `5e298986a093c4a47c875a0af2c338d91f06e01ff34ef452b6a9935744895322` (t0/t5), `bind-reading-gemma-locn1-20261105T164719Z.json` `d89fe094812ae69effc6a65fb2037257a97bc859f77097f5a6752133a9dab13e` (pass 1; the refused v1 reading is `bind-reading-gemma-locn1-20261105T164408Z.json` `bc2a22a0c5850dcf09ef23861e5c424fc84295f99b5908c2f7db596814fdee3f`), and `bind-reading-gemma-locn2-20261105T170853Z.json` `c80ac746c13748899cf2dc7c31b2620887e698f1960319cb36ab1234d811cfff` (pass 2).

## Cases (supervisor `receipt.json`)

All seven cases: `exit` 0, `case_complete` true, `model_launch_attempted` true, `runtime_pressure_admitted` `["1"]`, `whole_q1_qualified` false, `performance_qualified` false, no `failure`. No rerun.

| Case | Out | Kind | Elapsed (s) | receipt.json sha256 | execution.json sha256 |
|---|---|---|---:|---|---|
| t0 | `$C/gemma-t0-b6` | endpoint | 49.89 | `470956408305af817f97ddb20fb1552cdd8ce5f42f7e93f7a4f63693d8997ed6` | `b18d3aee6d992a1871f894a4116823f148c0c3657b7a4b250fb243c3d8ec3da3` |
| t5ord | `$C/gemma-t5ord-b6` | multistore | 58.59 | `ce800a8b04092823acf6753869575b10486de9ee754bf88754950af528b1ed4a` | `77536d6d24b9293c0fa47a02c725ed95891ce5c5a2defef53568e6735f089944` |
| t5ngram | `$C/gemma-t5ngram-b6` | multistore | 58.57 | `7ef5d58ad8d48bbb1eef0303f1e4ebdb883229dd7e8f232d0bac0595c8e67857` | `aea70f109ed28529b7b0882d88e58b7f2fd19b1f7f9a426742ff41aa9b31011c` |
| t5loc-ord-p1 | `$C/gemma-t5loc-ord-p1-b6` | multistore | 287.92 | `5f00e51b7e63de182c453e8fba956f9e76c1842ccb1889b709cd7fb7f8f91b66` | `0dda16d76e93266bfda5ebaa152cc65e7277e531e7a332720dc3c801c48cd69e` |
| t5loc-ngram-p1 | `$C/gemma-t5loc-ngram-p1-b6` | multistore | 292.21 | `cd2d899bd0acefede5456887da499332db3576f480cdc291ea54e91bbc2c046e` | `8267eda7750b1db855c1f93f8b9d1b928cb51f8233f3476b909fdf67b3d7559f` |
| t5loc-ord-p2 | `$C/gemma-t5loc-ord-p2-b6` | multistore | 94.47 | `9f4d3993ebb5b75d237c6d96ad183a1e149cb5289e977782abda3e6c42c3e85c` | `aa96d46485a78793b1693aa0161e82fa063d2460147fbf8c9915662ac7ae6ba0` |
| t5loc-ngram-p2 | `$C/gemma-t5loc-ngram-p2-b6` | multistore | 96.66 | `17ed39af9dded468cb703ec95f91c5e281eb76679960b5e6519f5a1f49560a1b` | `5c17b1984428c2637a97132f3a423ec4d295397993dc240f1173abc4e8fc28a1` |

Elapsed is `elapsed_seconds`, rounded to 0.01 s. Operating reviews (`$C/gemma/reviews/b6/`): t0 `6ef7fa81…`, t5ord `91374ad1…`, t5ngram `14362fa2…`, t5loc-ord-p1 `c22e4d28…`, t5loc-ngram-p1 `5d5c1ae3…`, t5loc-ord-p2 `3727ce06…`, t5loc-ngram-p2 `9ace6efe…`.

## T5 decisions

### G5-A1: pre-fork state — ExactEqual

`exact` t5ord vs t5ngram `prefork`: ExactEqual, common 4204, differing 0, left_only 0, right_only 0. `G5-A1.json` `c541b62cfb5ba4b390f3a24f329e27f6a820dec3ee77a1f2b5cce7153a874072`.

### G5-A2: integer decisions and the transaction — pass

Checked in process by the runner (every refusal is a test failure) and by the supervisor registry checks `gemma_t5_ordinary` and `gemma_t5_ngram` (design §3). Both cases completed. The helper `execution.json` files (shas in the case table) show:
- Both arms: status `CompleteScopedAcquisition`, `error` null, `class_b_refusal` null. Stores: t5ord {prefork 4204, committed 4264}; t5ngram {prefork 4204, committed 4264, raw-verifier 4}.
- **Initial argmax = `frozen[128]` on all four lanes in both arms:** 4861, 236787, 45518 and 236772 (`initial_equals_frozen` true). The cache@144 report agrees (`initial_argmax_equal` true).
- **N-gram transaction, every lane:** `accepted` 2, `rejected` 1, `rollback_cursor` 131, injected origin `InjectedDiagnosticFixtureNotProductionProposalQuality`. The fallbacks are 107 / 236787 / 236787 / 236800, and each equals the lane's decode input at position 131.

### G5-B: committed rows — RouteSensitive

`G5-B.json` `b5ddbb6abea04b284d42c61c3b0c4fe8e6ca651d7dfe4999f1cec19bb76b3500`, `envelope` t5ord (left) vs t5ngram (right) `committed`, policy `hf2q.q1.numeric-policy.v1`, τ 0.25:
- rows 64, `top1_agree` 61, `exact_rows` 0, flips 3, rows over τ 22, `rows_left_margin_below_tau` 5;
- D: max 9.45925098657608, p95 0.8005056381225586, p50 0.12999916076660156; KL (nats): max 4.987708601350244, p95 0.0275132276395714, p50 0.00035832429890351943.

**Split into verifier rows (128..130) vs continuation rows (131..143)**, computed from the per-row `flips` and `rows_over_tau` lists of `G5-B.json`:

| Group | Flips | Rows over τ | D max among listed rows |
|---|---:|---:|---:|
| verifier rows 128..130 | 0 | 4: agent-0 r130 (0.332), agent-1 r130 (0.504), agent-2 r129 (0.578), agent-3 r130 (0.338) | 0.5779542922973633 |
| continuation rows 131..143 | 3: agent-1 r133 (D 0.579, ordinary 236770 → n-gram 500), agent-2 r138 (D 0.382, 4685 → 236746), agent-3 r133 (D 9.459, 101 → 45518) | 18 (including the 3 flip rows) | 9.45925098657608 |

Row 128 has no flip and no row over τ. Every flip is in the continuation, after the rollback.

### cache@144 — ExplicitInconclusiveNumericPolicy

Rust Gemma T5 artifact compare. Packet `$L/inputs/gemma-t5-artifact-compare-packet.json` `9cc77fcfba4e8115954f9d9199cf9c1d5f75212f3abf55541ebd0db18333970f`. Log `G5-cache144-compare.log` `e4638f5ade28ce7759b6e943d679a6fe48e34645e9fa8b489542aba15e197cd8`, with `Q1_GEMMA_T5_ARTIFACT_COMPARE_COMPLETE lanes=4 committed_records=4264 float_policy=Unspecified whole_q1=false` and `test result: ok. 1 passed; 0 failed`. Report `G5-cache144-compare.json` `1fe529ee5558b6d6317a1b9e5dfa5fd57c3d1b425c4271a306bcba64fde4de13`:
- identity b6, config `b8c7b92f…`, tokens `84276820…5034`;
- `committed.disposition` ExplicitInconclusiveNumericPolicy over 424,459,264 B (146,665,216 elements): max_abs 15.828309059143066, rms 0.17994208323215888, relative_rms 0.03157689079177133, cosine 0.9995096031243895, 0 non-finite;
- first difference: Logits, agent-0, `logical_row_start` 128; cache boundary 144; records 4264;
- `route_class` "ClassB: OptionC 132-row re-prefill verifier vs M1 decode; Logits rows decided by envelope, not here".

Recorded, not gated.

### Routing (mapped) — RoutingChanged

`routing-mapped` t5ngram vs t5ord witnesses through the pass-1 key map (`G5-routing-mapped.json` `e30da21322d53ca45ed777886fc0da8dbf2dfd14cac0c1a896a0fa648cb87554`, `map_sha256` `2f49479d…`):

| Phase | Rows | Changed | Order only | Set changed | Weight bits differing | Changed gap max |
|---|---:|---:|---:|---:|---:|---:|
| verify-prefix | 15,360 | 0 | 0 | 0 | 0 | — |
| verify-committed | 360 | 74 | 62 | 12 | 360 | 0.013137051835656166 |
| continuation | 1,560 | 374 | 253 | 121 | 1,560 | 0.008336720988154411 |

15,480 candidate rows are unmapped by construction (namespaces `T5Initial` and `T5Verify` row 131; review notes, pass-1 map review).

## Branch N verifier-route localization

### Observer-effect controls — all exact

Each localizer case's T5 stores and routing witness, compared with the T5 case of the same arm:

| Control | Pass 1 (`OBS-*`) | Pass 2 (`OBS2-*`) |
|---|---|---|
| ord prefork | ExactEqual 4204 | ExactEqual 4204 |
| ord committed | ExactEqual 4264 | ExactEqual 4264 |
| ord routing | ExactRouting, 17,280 rows, 0 changed | ExactRouting, 17,280 rows, 0 changed |
| ngram prefork | ExactEqual 4204 | ExactEqual 4204 |
| ngram committed | ExactEqual 4264 | ExactEqual 4264 |
| ngram raw-verifier | ExactEqual 4 | ExactEqual 4 |
| ngram routing | ExactRouting, 32,760 rows, 0 changed | ExactRouting, 32,760 rows, 0 changed |

Every control has 0 differing, 0 left-only and 0 right-only. The pass-1 and pass-2 files of each control are byte-identical. sha256: prefork `c541b62cfb5ba4b390f3a24f329e27f6a820dec3ee77a1f2b5cce7153a874072`, committed `20427968e23d7e83c68abc2905280ba86ebe3b203ba44f8d89210641f4c27cf3`, raw `40ac4f9c08f72c20b1c91ef1b9cfc14c41dad60973044a8b25ab63e155322056`, ord routing `ddcb9defacbd582f05378871fa19a1a414d62b9a979a2f8bd49fe3c854d9be91`, ngram routing `12e94c3046369e67cfd3785e55f890335feee1723f2ecac0ecc818d0abee8fd3`. So the localizer did not perturb either T5 arm.

### Maps, reviews and pass bindings

| File (`$L/maps/`) | sha256 | Note |
|---|---|---|
| `pass1-map.json` | `2f49479d73aa651a15310c0d6012db4d36ceaf9c66a10536c09d88c205019f12` | `ngram-map` over the two T5 receipts and packet `packet-t5loc-ngram-p1.json`; layers 0..29 × {RouterLogits, RouterIds, LayerOutput} |
| `pass1-map-review.json` | `dde254d24de39f6d5b65f5d1aee44701bcff5cf476584f9de94d0353f4bff439` | Accepted (ReviewSpecB6) |
| `pass1-binding.json` | `43c39f3d2a0aace1abbf4a419a9bf19d2a3a87d4c2f354dd4277bd838aa44490` | map + four pass-1 traces |
| `pass2-map.json` | `4ebaa5d213b309270f0aedf5746f57ad282640363eaf26a372c8376ca336d2a4` | layers [0] (frozen rule: {0} when L* = 0); 14 prefill + 12 decode roles |
| `pass2-map-review.json` | `210e3be69e1476c9a492a7b4e2982b0a8b92733db497a00b25fab99d306f1efb` | Accepted (ReviewSpecB6) |
| `pass2-binding.json` | `edee151da03b8570e15852518df96770301423106e2194a1de0e9d355a6caff0` | map + four pass-2 traces |

### Pass 1 — L* = 0 on all lanes

`$L/decisions/locN-pass1-record.json` `b79a1f02dac8d0944d79df750d3464e6f5610f4313eb7667542a41ec085dbe13`, `mapped` over map `2f49479d…`, binding `43c39f3d…`, workload `$C/inputs/workload842.json` `4bb8863e367a7f4dde5d072ca1988305a2d5c9130a867b5321d4e373dc2ed424`. `accepted` [2,2,2,2]; 1350 groups per lane.
- Every lane has one equal group, then its first divergence at **verify-committed / layer 0 / `RouterLogits` / rows 128–130**. max_abs 0.012012392282485962 / 0.012101888656616211 / 0.014510571956634521 / 0.012163162231445312 for lanes 0–3.
- So L* = [0, 0, 0, 0], and pass 2 covers layer 0.
- Committed Logits differ on all 16 rows on every lane; max_abs 5.341 / 3.814 / 2.461 / 7.297. The head rule does not fire.
- `router_id_changes` over layers 0..29, per lane: 111 / 156 / 75 / 106, of which set changes 25 / 54 / 25 / 29.

### Pass 2 — localization record

`$L/decisions/locN-localization-record.json` `9675a8ac07312d5f039dd74ba1a02ca050e7ec4398ef2df9e3e39d5ff1e2cd6b`, `mapped` over map `4ebaa5d2…`, binding `edee151d…`, the same workload. `accepted` [2,2,2,2]; 181 groups per lane; `route_class` "ClassB: route-sensitive; a localized stage implies no fix".

| Lane | Equal groups before | First divergence | Differing rows | max_abs | relative_rms | Committed Logits max_abs |
|---|---:|---|---|---:|---:|---:|
| 0 | 1 | verify-committed / L0 / `Query` | 128, 129, 130 | 0.3529052734375 | 0.0009486917656017347 | 5.341278076171875 |
| 1 | 1 | verify-committed / L0 / `Query` | 128, 129, 130 | 0.35400390625 | 0.0007610382162511766 | 3.814427375793457 |
| 2 | 1 | verify-committed / L0 / `Query` | 128, 129, 130 | 0.3812255859375 | 0.0008512465599480964 | 2.461132049560547 |
| 3 | 1 | verify-committed / L0 / `Query` | 128, 129, 130 | 0.23175048828125 | 0.0010207942724838217 | 7.296700954437256 |

The head rule does not fire on any lane. Each lane lists 26 unmapped records; the map review gives the reason: candidate row 131 and the Residual/DenseOutput/MoeOutput roles at 128..130, and reference Norm at decode 128..130.

Query is the first comparable stage on this pair (the decode-only Norm has no verify twin), so the divergence is already present at layer 0's first mapped tensor; anything before the Q projection is not separately observable.

**`router_id_changes` (layer 0):**
- lane 0: row 130, order only (set unchanged). Reference `[46, 48, 22, 60, 0, 39, 1, 102]`, candidate `[46, 48, 22, 0, 60, 39, 1, 102]`, reference raw-logit gap 0.00778508186340332.
- lane 1: none.
- lane 2: none.
- lane 3: row 129, **set changed** (expert 90 → 78). Reference `[104, 100, 121, 33, 116, 68, 39, 90]`, candidate `[104, 100, 121, 33, 116, 68, 39, 78]`, reference raw-logit gap 0.003525972366333008.

## Deviations

1. **Split localizer bindings (finding 4.4(b)).** The design's localizer unit was one combined bindings file. The review found that the combined form would exceed the 1 s `max_sample_gap_ns` walking every listed out (finding 4.4). Each localizer case therefore has its own bindings file with the gemma-t0 gate, and the other localizer out is charged as unlisted. The combined pass-1 file was never launched; it is retained as `$C/sup/gemma-t5loc-n1-b6-SUPERSEDED.json` `61b0eecd93fc859d8532979665a7adc2a03e6e8322069ea0f3d3a612c8c325bf`.
2. **Supervisor change #709 (`dc7b5c56`, finding 4.4(a)).** For each sample the supervisor walks only the live out. The other listed outs are measured once before the baseline and re-checked unchanged after the child exits. It was re-admitted for the localizer units only; pass-1 receipts show the path ran (review notes: `other_listed_out_bytes` 1,102,272,338, the t0 gate).
3. **Cross-family Qwen charge dropped (root decision 2026-11-05, v2 policies).** The pass-1 v1 quiet bind was refused (105,778.6 MiB against 106,420 required, `bind-reading-gemma-locn1-20261105T164408Z.json`). The root took the reviewer's pre-accepted option and dropped the unlisted Qwen outs from the charge. Policies v2: ord `24ea6b71…` (floor 98,796,243,513) and ngram `f3a6af38…` (floor 100,710,635,345) supersede v1 `fa6fc170…` / `52fece2d…`. The pass-2 policies have no cross-family charge either.
4. **Gemma recompositions v2/v3.** Ctime-only, after the Q5 hard-link moves; same comparison key, each re-reviewed (section "Compositions" above).

## Decision provenance

`$L/sup/record_decisions.py` `e84540756a9cb2bb2a69d6ce0536f2737281993bf3eb1f15beb3e4ee58a6b3eb` (the #653 script, extended with every #665 decision) re-ran each Python decision with `python3 -B`. It used `q1_decision_policy.py` `01dd7fdf8d4d4374cb41a42405d6e575bc1ac368e2abe1c9eaaf7916083ce72f` or `q1_stage_localization.py` `9ff11f6185ba0b4added4ada14eb2d772533906f4388d9eb86baefa38b278123` (both at `/opt/hf2q-worktrees/build-q1-b6/scripts/four_agent`). Each output had to be byte-identical to the existing file. It wrote `hf2q.q1.closure-decision-record.v1` records with argv, tool sha, input shas, output sha, expectation, check rule and bindings. The cache@144 record binds the packet, log, build receipt, report and test-binary shas, without re-running. Existing records (the #653 ones) were kept, not rewritten.

| Record | sha256 |
|---|---|
| `decisions/G5-A1.record.json` | `f17f340558f751e1097ba12f28fcc530131e3944b52be96e9e1e4ac5776f7253` |
| `decisions/G5-B.record.json` | `3cbbde2b0ffe617ab41f411758fca10d2e344adb18a7f0c8bafb5724ad41e9a9` |
| `decisions/G5-routing-mapped.record.json` | `e94432b08ebf9e8f38d5e8901af9b4758e045033aa2735ad7886bda04f700276` |
| `decisions/G5-cache144-compare.record.json` | `07b1941f6cc5483bee7254e5554df7f8a60d96eb186355f98a01a2c0049b9baa` |
| `decisions/OBS-ord-prefork.record.json` | `aaa3cc882ac249e41d16447070bc751950030595d7836c2482cd8e816b235837` |
| `decisions/OBS-ord-committed.record.json` | `1baf53356d8da6f31666103f792d98801d0a17c5c4f1426d3233c42247d90d4a` |
| `decisions/OBS-ord-routing.record.json` | `02ac200c71d70f5604938edd22c0f769e1d25424e9cc4bea21896755288c4b47` |
| `decisions/OBS-ngram-prefork.record.json` | `9a74bbba8195681ce9730ab131273c646236ff3df800280cbd7c8bfef4eb949f` |
| `decisions/OBS-ngram-committed.record.json` | `c7b5be41eb8ed465b83fb5ba6cc6e7aae2505acb8511692b5ef5ac241242b5e9` |
| `decisions/OBS-ngram-raw.record.json` | `8a8b090ac8209b9c6069fe67dee9b861f1f7bcf96fcc02511be6529a0f28c2a3` |
| `decisions/OBS-ngram-routing.record.json` | `fbad6c6b538ec1032b740567ac6abcca2c396e42ff4b79f93853957cb550006a` |
| `decisions/OBS2-ord-prefork.record.json` | `07a432d124b64548bb9269e0753e4b7fedbad4abd9a98fe53e0ffd47b1f2681c` |
| `decisions/OBS2-ord-committed.record.json` | `dca4d6091d74d0a6cf3dd171aa18431f86c1ea9269f53d24367585a5c0de3e29` |
| `decisions/OBS2-ord-routing.record.json` | `f6cc87e45561aca00f4967dc2f01bd66e1d96b21b59659dad5102657b5a14a68` |
| `decisions/OBS2-ngram-prefork.record.json` | `561cc1cdc2920b5c6235c322354b9c70dd3a60e5dd6ca42074fb770ba300f77c` |
| `decisions/OBS2-ngram-committed.record.json` | `a37234afdb6d3811bfa10d1212a6b59b2778d53caeff6436939d40cd73735c20` |
| `decisions/OBS2-ngram-raw.record.json` | `d0c98fd3f790721f7a7b87e06120a96c8608851ec49abae1475c8f6222b6f34e` |
| `decisions/OBS2-ngram-routing.record.json` | `2291521208c4b235e37dc48c792fa8d05c09fb03971e1a215ccb8d7f581169dd` |
| `decisions/locN-pass1-record.record.json` | `39f10aabfe03f35b1413b96da28221cdf6777f7adc1b5b7985716c0e9d486b47` |
| `decisions/locN-localization-record.record.json` | `d12ae273125806d2e4c9c9dc4421c04065c347832424d6d378854bcfb0bf610e` |
| `maps/pass1-map.record.json` | `6ba2de821ef688fe337af9bfb747fcf4e16b9bee1b9f74614e5af80d72e87981` |
| `maps/pass1-binding.record.json` | `3dec69d0b215b9b9d558921367f548277c0047213b191ff801be7008387e234b` |
| `maps/pass2-map.record.json` | `d4181094191852edfc12900eed8d4f7dda8d435582f5b027013691c2b641763a` |
| `maps/pass2-binding.record.json` | `5dbabf25f74d9c0b5857b8d8ec2aa187692ab71f1f5c084743ea2413aec4efc2` |

## Unchanged in this PR

- `matrix.rs`: the Gemma T5 gap text ("not the slot-aware serving body") stays; it matches design §2.4's scope statement.
- Design text: no "localizer pending" wording remains, so the design is unchanged.
