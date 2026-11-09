# ADR-065 E4: four-agent critical path and verify eligibility

Status: complete (2026-11-09). Issue: #507.

## Method

`HF2Q_ADR065_CRITICAL_PATH=1` emits closed-schema `CLOCK_MONOTONIC` intervals for every Qwen prefill slice and every joint-reference or serial-solo decode turn. Decode intervals carry all participating slots and one counterfactual verify status per lane:

- `eligible`: request transforms are clear and an MTP K3 or request-history proposal is available;
- `blocked_transform`: grammar, repetition penalty, forced thinking or the automatic tool trigger prevents a verify row;
- `no_draft`: transforms are clear but neither proposer can supply a row;
- `unknown`: fail-closed status reserved by the schema.

The switch is exact-value opt-in. The disabled joint and solo paths test the switch before evaluating lane eligibility, so they do not clone proposer state, allocate the lane record or read a timestamp. On the measured path, interval bounds exclude observer record construction and JSON emission: the begin timestamp follows eligibility collection, and the end timestamp precedes serialization and `stderr` output.

`scripts/four_agent/adr065_critical_path.py` joins server intervals to the retained Q5 coordinator summary. The summary supplies the baseline makespan and every request, terminal, task-completion and barrier timestamp on the same monotonic clock. Prompt-token and cached-token counts identify each initial prefill signature. Equal signatures remain an aggregate queue-depth cohort; the extractor does not invent an agent-to-slot assignment. Per-agent intervals before the first request, between a terminal response and the next request, after the last response, and after task completion are retained as idle or tool gaps.

The extractor partitions each baseline makespan with this precedence: engine operation (joint, solo or prefill), queue, tool gap, then idle/unattributed. Categories are therefore disjoint and sum exactly to makespan. A width-one padded joint-reference turn is `joint`, not `solo`; `solo` is only the serial one-row route.

The E4 estimand is time-weighted per-lane eligibility:

$$
 f_r = \frac{\sum_{j \in r} t_j\,e_j/w_j}{T_r},
$$

where $t_j$ is a critical-path joint-turn duration, $w_j$ its lane count, $e_j$ its eligible-lane count, and $T_r$ the run's baseline makespan. The reported uncertainty is a two-sided 95% Student-$t$ interval over independent run-level $f_r$ clusters. Unknown statuses are reported but never counted eligible.

## Frozen inputs

| Input | Identity |
| --- | --- |
| Final observer binary source | `ab9aa92cb3f6597aa1c4899f006b594e7d13d0f2` (embedded stamp verified) |
| Final observer binary | `96860900f47029f044fce139bfefa4d0295c14bc8d9b353e83ce006a90ccf2af` |
| Final extractor source | `6d4c2068793638fa66808960de8129c387d116f9` |
| Q5 case | `565440066294e1368f4c13f53f27d9ad8c285c8be30e84ca44da537622e37fc5` |
| Qwen3.8 vanilla GGUF | `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7` |
| Sandbox proof | `4b2bea7715c6244e7636d6d07b4e33bd41b7c525313488dcca0f107cc79d486e` |
| Toolchain | `stable-aarch64-apple-darwin`, rustc/cargo 1.99.1 |
| Serving controls | four slots, context 262144 per slot, 48 GiB shared KV budget, text-only, `HF2Q_QWEN_SPECULATION=off`, delivered scheduler `LOG_FILTER`, E4 observer on |

The source-stamped release build used locked dependencies. The Q5 supervisor validated the case, binary, artifact, client approval and sandbox proof before each load.

## Retained runs

All three accepted runs completed the canonical four-agent coding protocol, reached `CalibrationPass`, and proved owned-process cleanup with no remaining process or listener.

| Run | Makespan (s) | Joint | Prefill | Queue | Tool gap | Idle / unattributed | Engine intervals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `final-01` | 128.211 | 85.4580% | 14.1945% | 0.0211% | 0.0058% | 0.3206% | 1,726 |
| `final-02` | 128.779 | 85.4705% | 14.2025% | 0.0204% | 0.0054% | 0.3013% | 1,730 |
| `final-03` | 126.822 | 85.3029% | 14.3620% | 0.0206% | 0.0062% | 0.3083% | 1,731 |

No serial-solo turn occurred. The padded joint-reference route handled lone ordinary lanes, so the width distribution includes width one.

| Joint width | Median turns per run | Median critical-path time (s) |
| ---: | ---: | ---: |
| 1 | 51 | 3.089 |
| 2 | 80 | 4.972 |
| 3 | 77 | 4.822 |
| 4 | 1,469 | 96.750 |

Each run retained 28 request intervals and 35 per-agent idle/tool intervals. The aggregate request queue reached depth four. Median critical-path category times were 109.567 s joint, 18.214 s prefill, 0.026 s queue, 0.007 s tool gap and 0.391 s idle/unattributed.

## Eligibility result

Every observed lane was `blocked_transform`: 6,310 lane-turns in each run, representing 414.616 s, 416.151 s and 408.718 s of lane-time. There were zero `eligible`, `no_draft` or `unknown` lane-turns. This agrees with E3: the canonical agentic workload keeps the automatic tool-trigger transform active.

Therefore every run has $f_r=0$. Across three independent retained run clusters:

- mean $f = 0$;
- 95% run-clustered interval: $[0, 0]$;
- eligible critical-path joint-decode time: 0 ns.

For this workload, joint verify cannot change makespan until its transform contract changes. E8's Amdahl projection therefore has $S=1$ for any finite complete-round speed factor $s$; E2 and E7 still own their independent measurement acceptance.

## Reproduction

```sh
GIT_COMMIT_SHA=ab9aa92cb3f6597aa1c4899f006b594e7d13d0f2 \
RUSTUP_TOOLCHAIN=stable-aarch64-apple-darwin \
RUSTC=/Users/robert/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rustc \
  cargo build --release --locked

python3 scripts/four_agent/q5_calibration.py \
  "$E4/qwen38-e4-critical-path.json" \
  --negative-control-receipt "$NC/receipt.json" \
  --output "$E4/final-01"
# Repeat as final-02 and final-03, one model load at a time.

python3 scripts/four_agent/adr065_critical_path.py \
  "$E4/final-01" "$E4/final-02" "$E4/final-03" \
  --output "$E4/critical-path-report.json"
```

`$E4` is `/Users/robert/.local/state/hf2q/v0130/e4-507`. `$NC` is `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023/q5-negative-control-20261030T041437Z`; the final receipts bind its stop-latency evidence.

## Evidence hashes

| Evidence | SHA-256 |
| --- | --- |
| Critical-path report | `121fa2b4cdda00b34319792614007a094e14893de661c05f76a28c838c5f51ed` |
| `final-01` Q5 receipt | `759e3574198482db8b3732077d0be20e76a48aa48f39125abfdb39aab0f17b65` |
| `final-02` Q5 receipt | `5c9900302b3cab36947bdd5d3445e0c7ca958b1904dbf446924cc97ce7fd798d` |
| `final-03` Q5 receipt | `b88aa8e2133f790c2d305fce38cdb7de8b9f2ca0af7c3c1a2eda7b4dc070a675` |
| `final-01` server log | `24c4c9943cebba5a0fa7f6e17417d21b9ba5c0de94af8ce27cfdb186b1758e8c` |
| `final-02` server log | `bf30ded333abe602dda71f96ddd1396ee9112ba1f5522bcb75aab15ac330e180` |
| `final-03` server log | `ef28ec586c373530adba6f2cd923b802e54b181ec1bd6a6127c714c99f7bf90a` |
| `final-01` coordinator summary | `dcf24391f837b9a6e4a50fe42cfd154d55f5be57ddbbd75e59334b9edd905f8f` |
| `final-02` coordinator summary | `a7a67daec978eaf2842537313bdcdf280cc1e71bf9e210fdd42ecc75bf9d4340` |
| `final-03` coordinator summary | `d17cb6329fe434e0cc7c8dbfaa9dc87d60e29a9597ff0f656a86472be3833c04` |

One earlier `CalibrationPass`, `$E4/run-01`, is explicitly excluded. It used source `e0c66ec0a75b625791afc304dfcd527a636f1e9c`, binary `bfd5467315c84557dec73c40a420585993bc6ddde8c2a4af8d51388ad1ea5caa`, case `96423f95124666ea504789d5187212090300b026ac400e2b8ff42342169f4676`, and receipt `a299e35b3cf979c7c7da70af332342622e5898641e98e44d1d838242bc504807`. The source was superseded before analysis to ensure the disabled observer does not evaluate proposer eligibility. Its measurements were never pooled with the final three-run set.
