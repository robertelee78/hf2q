# ADR-065 E3: shadow draft acceptance

Story: [#506](https://github.com/robertelee78/hf2q/issues/506), [ADR-065 E3](../adr/ADR-065-four-agent-joint-speculative-verify.md#e3-measure-counterfactual-draft-acceptance-on-joint-ready-states).

## Diagnostic contract

`HF2Q_ADR065_SHADOW_DRAFT=1` requires `HF2Q_QWEN_SPECULATION=off`. On each scheduler-confirmed joint group it runs the request-history proposer and, when MTP state is available, MTP K3. The draft cache and lookahead capture are discarded before ordinary target decode. Drafts never enter target verification or commit. Each proposal is compared with later ordinary committed tokens and emits one closed-schema `hf2q.adr065.shadow-draft.v1` record when it mismatches, completes, or is censored at request completion.

`scripts/four_agent/adr065_shadow_acceptance.py` rejects unknown or malformed fields, excludes censored proposals from acceptance histograms, and aggregates by proposer, slot, cold/warm turn and context bucket. It also reports proposal wall cost and the share for which grammar, repetition penalty, forced thinking and tool-trigger transforms are all absent.

## Bound identities

| Item | Value |
| --- | --- |
| Implementation commit | `e208b19fd6235ea7b390078c9af4fbbedb170593` |
| Candidate binary | `$E3/build-e208b19f/hf2q`, sha256 `708c20b4ea963955d31ee4b180e334ad903316a74a4cc9141238f8b0e23e9aa9` |
| Model | Qwen3.8 27B `$MODELS/qwen38-stock/Qwen3.8-27B-vanilla.gguf`, sha256 `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7` |
| Off case | `$E3/qwen38-e3-off.json`, sha256 `487105e08273794ed453fc1271de25843b3e902954d3483bb1fe9417b3c14833` |
| On case | `$E3/qwen38-e3-on.json`, sha256 `98a2cc690676908d46090ef39c69fb077bd5c5afec383f584c5b4c1ec07a00c0` |
| Off receipt | `$E3/off/receipt.json`, sha256 `06bcda5224e3c6a9f30c5898c0d1b38626de1958f35122953ac4f3d0456f2935` |
| On receipt | `$E3/on/receipt.json`, sha256 `4b6abf984bd344a31b3b42125461d9e4b8588f89036204f9c98b4cfeb0fa0d59` |

`$E3` is `/Users/robert/.local/state/hf2q/v0130/e3-506`; `$MODELS` is `/Users/robert/.local/share/hf2q/models`. Both cases use the same frozen binary, model, four-agent task, sampler settings, `HF2Q_QWEN_SPECULATION=off`, client approvals, sandbox proof and negative-control receipt. They differ only in `HF2Q_ADR065_SHADOW_DRAFT` (`0` versus `1`).

## Retained four-agent result

Both runs returned `CalibrationPass` and proved server cleanup. The transcript comparator covered 28 agent/turn positions and classified them `ExactTranscripts`: 28 same-request groups, zero different outputs. `$E3/transcripts.json` has sha256 `43c8fd2e86982d300f260ecb0f249b8ca7c92c80e5e143e66e291c3170cbee5d`.

The enabled run emitted 1,958 proposal records: 1,918 complete and 40 censored. Censored records are retained but excluded below. The four-agent run stayed in the `near_2k` context bucket.

| Proposer | Accepted 0 | Accepted 1 | Accepted 2 | Accepted 3 | Complete | Median proposal cost | p95 proposal cost | No-transform share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| History | 257 | 197 | 81 | 774 | 1,309 | 333 ns | 916 ns | 0% |
| MTP K3 | 27 | 49 | 50 | 483 | 609 | 8.569 ms | 8.918 ms | 0% |

History acceptance by slot:

| Slot | Accepted 0 | Accepted 1 | Accepted 2 | Accepted 3 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 64 | 46 | 18 | 252 |
| 1 | 60 | 49 | 21 | 173 |
| 2 | 74 | 55 | 22 | 179 |
| 3 | 59 | 47 | 20 | 170 |

MTP K3 acceptance by slot:

| Slot | Accepted 0 | Accepted 1 | Accepted 2 | Accepted 3 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 7 | 13 | 13 | 118 |
| 1 | 7 | 13 | 14 | 121 |
| 2 | 6 | 10 | 9 | 123 |
| 3 | 7 | 13 | 14 | 121 |

History produced 51 complete cold-turn records (`20/12/5/14` for accepted prefix `0/1/2/3`) and 1,258 warm-turn records (`237/185/76/760`). MTP state was available only on cold turns in this retained run, producing `27/49/50/483`; the absence of warm MTP records is reported, not imputed. Every complete record was in `near_2k`, so its per-context histogram equals the proposer total.

The no-transform share is 0% because this agentic workload equips every measured request with automatic tool-trigger grammar. This misses ADR-065 E8's E3 prerequisite of at least 50% untransformed verify rows; E8 cannot select the host-selection optimization on this evidence regardless of E2's readback attribution.

The reduced report is `$E3/acceptance.json`, sha256 `d5120aae936b469b8c86c2af213366b4ca28c9fa08f870de3e42fd74fd1e16dd`.

## Reproduction

From the repository root, with the prerequisite paths bound in each case:

```sh
python3 -B scripts/four_agent/q5_calibration.py "$E3/qwen38-e3-off.json" \
  --output "$E3/off" --negative-control-receipt "$NEGATIVE_CONTROL"
python3 -B scripts/four_agent/q5_calibration.py "$E3/qwen38-e3-on.json" \
  --output "$E3/on" --negative-control-receipt "$NEGATIVE_CONTROL"
python3 -B scripts/four_agent/adr065_shadow_acceptance.py \
  "$E3/on/server.stderr" --output "$E3/acceptance.json"
python3 -B scripts/four_agent/q1_decision_policy.py transcripts \
  "$E3/off" "$E3/on" > "$E3/transcripts.json"
```
