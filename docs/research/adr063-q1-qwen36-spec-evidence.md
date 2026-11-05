# ADR-063 Q1-QWEN36-SPEC (#653): Qwen3.6 T5/T6 evidence on b6

Design: `docs/research/adr063-q1-spec-design.md` (b6 amendment, PR #685). Numeric policy: `docs/research/adr063-q1-numeric-policy.md`.
Paths: `$E` = `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`, `$C` = `$E/q1-closure-v1`, `$L` = `/Users/robert/.local/state/hf2q/v0130/q1-spec`.
Every number below is read from the cited file; sha256 values are of that file.

## Result

| Decision | Class | Verdict | Evidence |
|---|---|---|---|
| T6 (P6-1..P6-3) | A | **PASS**: ExactEqual 1440 / 560 / 560 | `$L/decisions/P6-{1,2,3}.json` |
| T6 diagnostics (P6-4, P6-5) | A diag | ExactOnCommon, 0 differing, right_only 816 / 1696 | `$L/decisions/P6-{4,5}.json` |
| T5-A1 integer decisions | A | Checked in process by `t5_artifact_compare` (passed) | `$L/decisions/T5-artifact-compare-q36.{json,log}` |
| T5-A2 pre-fork state | A | **ExactEqual**, 596 scoped records | `$L/decisions/T5-A2.json` |
| T5-A3 side effects | A | **Exact**: discard 148/148, MTP catch-up 4/4 in both arms | `$L/decisions/T5-A3-*.json` |
| T5-B committed rows | B | **WithinDecisionEnvelope** (64 rows, top1 64, 0 flips, 0 over τ) | `$L/decisions/T5-B.json` |
| caches at 131/144 | B | ExplicitInconclusiveNumericPolicy (recorded, not gated) | `$L/decisions/T5-artifact-compare-q36.json` |
| routing | — | **Missing-with-reason**: no Qwen3.6 MoE routing witness | design §1.5 |
| verifier_route_localization | — | **Missing**("no Qwen3.6 verifier-route localizer") | — |

No Class A defect. `whole_q1_qualified` is false in every receipt and in the T5 report. This record makes **no serving, ordinary AUTO-selection or performance claim** (`performance_qualified` false, `physical_admission` false, `ordinary_AUTO_selection_qualified` false in `T5-artifact-compare-q36.json`).

## Build of record b6 and its relation

- Source main `38eea8393862b16068bee23789ded41ddf853328`; mlx-native `=0.15.18`, checksum `9d10f97e0e25a00c1b4706a0ea9a838b2524b3fc2b999d05f21cd2580c524bee`.
- Test binary `$C/build-38eea8393862b16068bee23789ded41ddf853328/test-binary` `46bc533f1e3512e3599fc04b4979496d0e69031919b5af6a5dee52bfc9cc805c` (re-hashed; equals the build receipt and the T5 report identity).
- Build receipt `build-receipt.json` `f7727e1215a0858e73238d428f482ba6fde21b362b3180b630baf93a16f4de06`: `controls_exit` 0, all seven control exits 0, commit stamp 8 occurrences.
- Relation `$L/b6-relation-20261105T051546Z.json` `c639949f524f642cbc3f50142242c7eca0412139afe0427400c39f927e5f7e0d`, recorded immediately before the first b6 launch: #512 and #527 OPEN, J1 PR #676 OPEN (not merged), `ea6caf46`, the #684 merge and the #685 merge all ancestors of b6. So b6 precedes P4b and J1 as §1.2 requires.

## Supervisor admission

- Supervisor of record `q1_endpoint_supervisor.py` `5fdedd43d4c8d0756c9140366bced59e7982074fde29a801e5e791425a5451ff` (main `ff4af4bd`, unchanged at b6).
- `$C/reviews/supervisor-5fdedd43-admission-review.md` `8ecae9f32026146ceb4e29f99f70c39d96ff7f2be5865f8fad1f2f370b8e3d83`: **ADMITTED**, no source change and no binding condition; findings A1–A3 are P3 (A1 and A2 are fixed in the design text by this PR).
- `dc7b5c56` (PR #709, `$C/reviews/supervisor-dc7b5c56-admission-review.md` `3eaa76ad656ae2bf3db26bfb72035c7330ccd8689bab476c7738d9f6b166e3ce`) is admitted only for the #665 localizer units; every #653 unit's bindings pin `5fdedd43`.
- Reviews 1–4, the deviation ruling and the final binds: `$C/reviews/b6-spec-review-notes.md` `5f574f1b038c09ecda30d4dd1623b31d8ac77a64892bb11b1204090bd4555a5a`.

## Compositions and bindings

| Unit | Composition sha256 | Acquisition review sha256 | Bindings sha256 |
|---|---|---|---|
| endpoint (t0, t3) | `2807ccf9dc8c5e36944bc4cb8e81b105aa4bd5b403e742ce03e12274a4a3ddac` | `c41306e623659384a04656f546c63f584379b92aac05721bfcd0ff992f2dae96` | `b39029418e34efe7af3e4a89ebff8f74c990b7c3206aa17a5d79abd65bb75505` |
| continuation (t3cm, t6) | `4a00663291cabc8f215048e40b8eace7106199f5b3c21c656aa03d96ece9cdd6` | `0165d872dbaf11a53409be5cfb9cca7e78846605f2b4dd53c1546c31742524fe` | `180ca1065fe934efa2e1bb71a9200470d3970b102ba37d1101a79bd7b95be57e` |
| t5prep | `120fffd38f29b1055c568fa12f8430b91f9789f959392819b6f1990ffaaa9daa` | `1bd0e61ceaa62108aaeca8f7cf7cb77eeb0a9aa8afdd19732681212c68bef733` | `0e953bb28643e513819213d20af00ac040e1e0b1a91d4e08bef0a1ea3afedda6` |
| t5phases (t5base, t5forced) | `b563baa7d896b0699caa58eb72d57332bad700e13d9bff586a8d2a6e01b7a243` | `2b5096b6becae4f11c55e87608dac5f921e146a167745b45577392c14d66d62b` | `f015901c7f6d9a10af26299460ada17c57d8675f0e97c98ae511b3b2edc378c2` |
| t5ff (t5ffbase, t5ffforced) | same as t5phases (deviation D4 below) | same as t5phases | `fb6b37a60e71697b9e5b5c648e5fc74416cd795cefb3ae3bd2dff46f1259a955` |

Compositions under `$C/qwen36/b6*/composition-v1.json`, reviews under `$C/qwen36/reviews/b6/`, bindings under `$C/sup/qwen36-*-b6-bindings.json`. Tail128 derivation review `e61b85ef51a8cd47e9fc003cc943463118f3e84cdf8b420ca0f861581e23189e`; T5 frozen-file review `t5-frozen-review.json` `822b6ea9044472c493d362cb5e49220eec4e6820141a5bc705b8fb9dbdbd96f8` (frozen workload `492e917c9761c02a7bbbd8141fdf2f87a5f00160fb08347aebe6c3e0af4ca52d`).
Quiet bind readings: `$L/bind-reading-qwen-20261105T051628Z.json` `9d32eaa48a33b0fdc5e1be12af5cf834f53fc9672424876e6b11adf3e545ee4e` (endpoint, continuation, t5prep) and `$L/bind-reading-qwen-t5-20261105T052939Z.json` `f81c88c6ee5a87e71e9364854f661b52cfc80256f4f7f7ed29e41561235f35ec` (t5phases, t5ff).

### Deviation D4: t5ff shares the t5phases composition (review §5, finding 5.1)

The design §1.2 table lists the T5 first fork as its own composition row. `ComparisonKey` has no first-fork field (`acquisition.rs:73-84`; `runner.rs:258-267`, `:297-306`), so the t5ff key equals the t5phases key and two composition files would be byte-equal. The review (`b6-spec-review-notes.md` §5) ACCEPTED one shared composition and one acquisition review, under conditions D1–D4. D1–D3 are recorded there (both bindings bind the same composition and review; the review attests the t5ff packets; separate policies, bindings, copies and operating reviews). **D4 is this statement.** The design §1.2 table is amended by this PR.

## Cases (supervisor `receipt.json`)

All nine cases: `exit` 0, `case_complete` true, `model_launch_attempted` true, `runtime_pressure_admitted` `["1"]`, `whole_q1_qualified` false, `performance_qualified` false. No rerun.

| Case | Out | Kind / mode | Elapsed (s) | receipt.json sha256 |
|---|---|---|---:|---|
| t0 | `$C/qwen36-t0-b6` | endpoint T0ColdSerial | 34.19 | `d83283b60de7cdae8e3e74f758ac2cba53d698cb6eac83c0c0a14816718a8756` |
| t3 | `$C/qwen36-t3-b6` | endpoint T3Chunk64 | 39.33 | `0fb95dcf9bec383e9f2352ebf093e05bf542a0da7a590fbfc8b19ae9cc295b66` |
| t3cm | `$C/qwen36-t3cm-b6` | multistore | 36.91 | `74cf9403c0d788eed5c95edb31680cc052370b0b5667ad691b02f33272800439` |
| t6 | `$C/qwen36-t6-b6` | multistore | 57.46 | `0ad137f7d3ea375e5b65cf1702bb3cd56180681cc8175ff9e3183744824f8006` |
| t5prep | `$C/qwen36-t5prep-b6` | multistore | 11.74 | `c3b2eaf8baccd549af17525d0a4720e463a53df1ceaa1c837490c97eeceb14b7` |
| t5base | `$C/qwen36-t5base-b6` | multistore | 32.07 | `f33755be37904bfbe4e27322a2d73f64f6c1bacbdaf043977f553450f95bd153` |
| t5forced | `$C/qwen36-t5forced-b6` | multistore | 32.78 | `9360c3474a39595a9055e7898173748ca9ebbc9bdaf2a9b8b55eb79faec3a212` |
| t5ffbase | `$C/qwen36-t5ffbase-b6` | multistore | 12.68 | `71e6d47294ef96b768e14d98abe51d154b8f2ebc5c122fdbb46ff1982f871a3d` |
| t5ffforced | `$C/qwen36-t5ffforced-b6` | multistore | 14.37 | `1abb9c7bf69512a4b42254db507f223ed690b5a85318ba51fbf895a2359b8bb4` |

Elapsed is `elapsed_seconds`, rounded to 0.01 s. Helper `execution.json` statuses: t5prep `CompleteWorkloadPreparationNotComparable`; t5base, t5forced `CompleteScopedAcquisition` (1840 committed records; forced 20 raw-verifier records); t5ffbase, t5ffforced `CompleteFirstForkDiagnosticNotQ1Qualified` (299 committed records). t5ffforced's one transition: lane 0, entry 129, matched 1, rejected 1, committed 131 (`qwen36-t5ffforced-b6/output/execution.json` `897c4e0b0d8a72be920e7b32ff31e8e7a19b0c98312c0f4b9722f01db53a07ff`).

## T6: PASS

**Deviation (design §1.4 condition, review answer 1): the store of record is T3cm (`qwen36-t3cm-b6`), not the endpoint T3 that the issue literally names.** T3cm is the only same-route never-cancelled baseline; the endpoint T3 comparisons are the pre-registered diagnostics P6-4/P6-5.

| Decision | Left vs right | Classification | common | differing | left_only | right_only | sha256 |
|---|---|---|---:|---:|---:|---:|---|
| P6-1 | `t6/continuation` vs `t3cm/continuation` | ExactEqual | 1440 | 0 | 0 | 0 | `e4a9f5b46f21cbf6d4b7b9845467905c0da3eff3024acd181f64a7550e617960` |
| P6-2 | `t6/recovered` vs `t3cm/prefix64` | ExactEqual | 560 | 0 | 0 | 0 | `ed496175f230b6b521a0ad9917382cff45985f8d79490694c70b01c3b162ce17` |
| P6-3 | `t6/before` vs `t6/recovered` | ExactEqual | 560 | 0 | 0 | 0 | `ed496175f230b6b521a0ad9917382cff45985f8d79490694c70b01c3b162ce17` |
| P6-4 (diag) | `t6/continuation` vs `t3` | ExactOnCommon | 1440 | 0 | 0 | 816 | `a6df54f75d7b9f8a2c7cc49cb3e9d0746d67c1b0cc26effac3679e8815aaf4a1` |
| P6-5 (diag) | `t6/recovered` vs `t3` | ExactOnCommon | 560 | 0 | 0 | 1696 | `663dd5589be439f339994431d306c2fd659b689c95dd9d71eaa25483b3bb86f0` |

P6-2 and P6-3 have the same bytes, hence the same sha. In process (`qwen36-t6-b6/output/execution.json` `b2b0519fbea9531ab90db90eb4bf2d92e085808ed7d6bb736190a6d439029301`): `captured_rows_per_lane` 80, `cancelled_unpublished_rows_per_lane` 64, `trace_records` 1440, `recovery_proof.disposition` ExactEqual over `compared_bytes` 266,117,120 (max_abs 0, cosine 1.0). T3cm (`execution.json` `0df6827ad84397005a3940957445f084cab02b65b1010f91acd8c3db0c011ab2`): `cancelled_unpublished_rows_per_lane` 0, `trace_records` 1440.

## T5

### T5-A1: frozen lineage and transitions

Checked inside the Rust `t5_artifact_compare` (`t5_artifact_compare.rs:68-98`, `:101-145`) over packet `$L/inputs/t5-artifact-compare-q36-packet.json` `6ab11a5bff9f3b7f8da9aca6cbc0fd69df933df1ad0983256a3e513c7d86134e`. The run passed: log `$L/decisions/T5-artifact-compare-q36.log` `0723fa050f3ea2d8ba068d733fa38d84c3ba10dc2ef5b8d5c0af215c0704f240` shows `Q1_T5_ARTIFACT_COMPARE_COMPLETE lanes=4 committed_rows=16 raw_records=20 float_policy=Unspecified whole_q1=false` and `test result: ok. 1 passed; 0 failed`. Report `T5-artifact-compare-q36.json` `4fa5e7f5171b8f7d05b05624575a0e3e99f7a097b0fc40ba76b8b155801b5a3c`; its identity is b6 (source `38eea839…`, binary `46bc533f…`, native 0.15.18), config `1b3dab7e…`, tokens `06af2c45…`. Raw-verifier audit: 20 records, 4 verifier rows per lane, 16,023,680 B, all payload hashes and pages verified, all f32 finite.

### T5-A2: pre-fork state — ExactEqual

`exact --scope $C/qwen36-spec-extent-b6/t5-prefork-scope.json` (scope `bc87799308edd7182ea91cc5ea47e5fa120474258b37c9762afca0a07e4d6a80`), t5base vs t5forced `committed`: ExactEqual, `scoped_records` 596, common 596, differing 0, left_only 0, right_only 0. `T5-A2.json` `e78eab77dcd9f1002ea3225e2c93ef61d3a7a829caaa8c4c49ed9d7b0cb9e88c`.

### T5-A3: side effects — exact

| Decision | Left vs right | Classification | common | differing | left_only | right_only | sha256 |
|---|---|---|---:|---:|---:|---:|---|
| discard | `t5ffforced/committed` vs `t5ffforced/post-draft-discard129` | ExactOnCommon | 148 | 0 | 151 | 0 | `08b87f5b0ccdabe4c42fa30043c7678eac1380b8ed268ae8f2e44f9223ed365c` |
| catch-up, base arm | `t5ffbase/target-before-mtp` vs `target-after-mtp` | ExactEqual | 4 | 0 | 0 | 0 | `40ac4f9c08f72c20b1c91ef1b9cfc14c41dad60973044a8b25ab63e155322056` |
| catch-up, forced arm | `t5ffforced/target-before-mtp` vs `target-after-mtp` | ExactEqual | 4 | 0 | 0 | 0 | `40ac4f9c08f72c20b1c91ef1b9cfc14c41dad60973044a8b25ab63e155322056` |

All 148 post-draft-discard records (cache@129 after draft+discard) equal the committed state; the 151 left-only records are the rest of the committed store, which has no post-discard twin. Both catch-up files have the same bytes.

**Fork-arm diagnostic** (`T5-FF-fork129.json` `7c0bd82e2cffb0cb691789547a7d5efff21577cb1ae136125a5d7db1d632db8f`; t5ffbase vs t5ffforced `committed`): DivergedExactInvariant, common 299, differing 148, so 151 common records are equal and the 148 post-fork cache@131 records differ. First differing key `[CacheConv, agent-0, 0, 1, 0, 131]`. This is the expected route split after the fork, recorded, not gated.

### T5-B: committed rows — WithinDecisionEnvelope

`T5-B.json` `02c184abafb96f1410f49cd2e668f84a1387dba60452a02e628b175ed3e290e8`, `envelope` t5base vs t5forced `committed`, policy `hf2q.q1.numeric-policy.v1`, τ (`route_decision_error_nats`) 0.25:
- rows 64, `top1_agree` 64, `exact_rows` 4, flips 0, `rows_over_tau` 0, `rows_left_margin_below_tau` 6;
- decision error D: max 0.21340179443359375, p95 0.13153076171875, p50 0.007526397705078125;
- KL (nats): max 0.001882108948175161, p95 0.0002380261191757894, p50 1.1502984424842552e-06.

**Split into verifier rows (129, 130) vs post-rollback rows (131..143):** flips and rows over τ are 0 in both groups, since both lists are empty. A per-group D distribution (max, p50, p95) or per-group top1/exact counts are **not computable from the record**: `T5-B.json` keeps only whole-set aggregates and lists per-row data only for flips and rows over τ.

### Caches at 131/144 — ExplicitInconclusiveNumericPolicy

From `T5-artifact-compare-q36.json`: `committed.disposition` ExplicitInconclusiveNumericPolicy ("no approved policy or required native routing capture unavailable"), `float_policy` Unspecified, cache checkpoints [129, 131, 144], committed rows [128, 144]. Over `compared_bytes` 872,421,120 (231,757,248 elements): max_abs 0.9131593704223633, rms 0.009416968274988544, relative_rms 0.011574870461004997, cosine 0.9999330193587563, 0 non-finite. First difference: Logits, layer 40, agent-0, `logical_row_start` 129. `packed_cache_byte_invariants_retained` true. Recorded, not gated (design §1.5).

### Routing: Missing-with-reason

No Qwen3.6 MoE routing witness exists; the only witness, `gemma_routing_witness`, has Gemma-only namespaces (design §1.5).

### verifier_route_localization: Missing

Missing("no Qwen3.6 verifier-route localizer"). T5-B is WithinDecisionEnvelope, so the design's RouteSensitive localization scope is not triggered.

## Decision provenance

`$L/sup/record_decisions.py` `9cea5f429669a44654c45e86426eeee65cb296879f5026c538d7c5a7e37c23e6` (from the #652 precedent) re-ran every Python decision with `python3 -B` and decision tool `/opt/hf2q-worktrees/build-q1-b6/scripts/four_agent/q1_decision_policy.py` `01dd7fdf8d4d4374cb41a42405d6e575bc1ac368e2abe1c9eaaf7916083ce72f`. It required each output to be byte-identical to the existing file and wrote `hf2q.q1.closure-decision-record.v1` records (argv, tool sha, input shas, output sha, expectation, check rule, bindings). The Rust compare record binds the packet, log, build receipt and report shas and the test-binary sha, without re-running.

| Record (`$L/decisions/`) | sha256 |
|---|---|
| `P6-1.record.json` | `f8d1185e22d039b37ffff0d9cf4e21c690fb2c04caf840ebf918669bbc70d98f` |
| `P6-2.record.json` | `71e667a6ef6625be406b092e555c7b8cf927829f91f17d9f94e22a286280fd1e` |
| `P6-3.record.json` | `65b8a807125fcbe70cc476860da0f6b92d3e29948809701d50f49b00181d6754` |
| `P6-4.record.json` | `0aa291c997f180014d8d92dd6fa99f8792fbf701017f61ba3e835ce9312fc930` |
| `P6-5.record.json` | `389cbb7883182033d17f2a6d897602260016bf8e86476e6b28ebd166e34ead46` |
| `T5-A2.record.json` | `e8315750869c5db67515e9d4d3c73db7a34aad947546a4dd3d369800c85c3fcb` |
| `T5-B.record.json` | `3d0b8ec0401b2753acaf1fb763db3be7fa26be98eb1e952b569917d83eae82df` |
| `T5-A3-discard.record.json` | `306192719b5263b9dbb3a1b3eb59cd129444304c4c83696bb4fdc53188a17bf8` |
| `T5-A3-catchup-base.record.json` | `5713c83ec2d0922f8e43dac56c5b76df668f3da32cfdcf45de9aef90cddf1454` |
| `T5-A3-catchup-forced.record.json` | `aae547412914dd389e2409a297337a6632004bf60152fbea6313bec4fb9501c9` |
| `T5-FF-fork129.record.json` | `a7a88cfbb57b1cc4995b10857655eb885468dbb09cd1a4753ebef4335fced159` |
| `T5-artifact-compare-q36.record.json` | `21227d4edf725935da708431a72d9ff6e29147f4d7a6b5fb80fc842664abe731` |

## Design text and matrix changes in this PR

- Design §1.2: the t5ff row shares the t5phases composition and acquisition review (finding 5.1).
- Design §1.4: t3cm `cancelled_unpublished_rows_per_lane` is 0 (t6 64); the recovery proof binds `compared_bytes` (finding A1).
- Design §2.6: decode stage records use `logical_step` 0 with `logical_row_start` = position (finding A2).
- `src/quality/adr063_invariance/matrix.rs`: "Qwen36 actual execution pending" removed from the T5 and T6 gap texts (design §1.7(4)).

## #665 interim status (not a verdict)

Q1-GEMMA-SPEC (#665) stays open: the branch-N verifier-route localizer (pass 1) is pending, behind the PR #709 supervisor fix (`dc7b5c56`). Values recorded so far, from `$L/decisions/`:
- **G5-A1** (`G5-A1.json` `c541b62cfb5ba4b390f3a24f329e27f6a820dec3ee77a1f2b5cce7153a874072`): ExactEqual, 4204 common, 0 differing.
- **G5-B** (`G5-B.json` `b5ddbb6abea04b284d42c61c3b0c4fe8e6ca651d7dfe4999f1cec19bb76b3500`): RouteSensitive; 64 rows, top1 61, 3 flips (agent-1 row 133, agent-2 row 138, agent-3 row 133), 22 rows over τ, D max 9.45925098657608.
- **cache@144** (`G5-cache144-compare.json` `1fe529ee5558b6d6317a1b9e5dfa5fd57c3d1b425c4271a306bcba64fde4de13`): ExplicitInconclusiveNumericPolicy.
- **routing-mapped** (`G5-routing-mapped.json` `e30da21322d53ca45ed777886fc0da8dbf2dfd14cac0c1a896a0fa648cb87554`): RoutingChanged, over map `2f49479d73aa651a15310c0d6012db4d36ceaf9c66a10536c09d88c205019f12` (`$L/maps/pass1-map.json`, pass-1 map review `$L/maps/pass1-map-review.json` `dde254d24de39f6d5b65f5d1aee44701bcff5cf476584f9de94d0353f4bff439`).

These are not provenance-recorded by this PR; #665's evidence record does that once the localizer completes.
