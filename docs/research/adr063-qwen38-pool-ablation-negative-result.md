# Qwen3.8 N4 bounded scratch-pool profile: negative acceptance result

The opt-in `adr063-24g-v1` allocation profile did **not** meet the predeclared
5% improvement requirement in this same-binary Qwen3.8 coding-task experiment.
Estimated client-time reduction was **1.794329%**, with nominal 95% interval
**[0.014756%, 3.542229%]**. Precision passed; the required benefit did not.
Retain this negative acceptance result. It is not a recommendation to enable the
profile by default, a stock-release/latest-peer comparison, an all-four-model
speed result, or full F2/Q5 qualification.

Campaign identity: `qwen38-pool-balanced-20261026T070338Z`, completed 2026-10-26 UTC.
The controller completed its fixed five blocks and 20 cohorts. Quality, exact
stable-prefix reuse and owned cleanup passed independent review for all 20.
Only retained source/evidence inspection and mathematics produced this report;
no new model, build, test or device execution was performed for the report.

## Frozen identities

| Component | Identity |
|---|---|
| hf2q source, both arms | `5a9d5e9f3e895151e1aab23ddbf2277a1e559253` |
| Same hf2q binary SHA256 | `a9f118ae7d2567ac006c335db8562a48060a1760b7dde73b9a7a8b6cd6f62e3d` |
| Published mlx-native | `0.15.4`, registry source; VCS `cffc3bb53d4823ab39879ddeb45ee16b9d902478` |
| mlx-native crate checksum | `359470da84f037916cd34534b56fc283fdd467c152a9813a85979c1d6e67f137` |
| Coding client source | `1f9fda16c6a8bad0c3e475a1cde596d16606a006` |
| Vanilla hf2q-produced Qwen3.8 27B Q4_K_M GGUF SHA256 | `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7` |
| GGUF bytes | `16810714976` |
| Canonical coding fixture SHA256 | `1643af68f5be6118ab2c83a386027564fcd9954ab284d1cc1fbcba544fe053ff` |
| Runtime tokenizer semantic SHA256 | `b6af43455bcb7c7fdedcc5a45e14359fc21567b6f87e01fa822229c68871b711` |
| Runtime template semantic SHA256 | `192710c8be89098fa3be92445b92d829c08b2e01186b15f2344e4042c1a0ba77` |
| Observer parser source | `8ed4863930583b1ef6a70fa0e7084fbcab53abab` |
| Offline prefix helper source SHA256 | `7f969ab076a68d34ea3f94b87d8a9ed9ee8414803dafe495d022740797b3172a` |
| Offline prefix binary SHA256 | `4c0e5234273b33547d93fb22f9e5d06d62cba6807a5c6e18cb14e48bacb12a4c` |
| Prospective packet generator SHA256 | `f88506d1492f927295d650780109d4548f16845aef1b3db110dbe338a0bc19ec` |
| Fixed-five controller SHA256 | `1fc17d8ad96e2adf24f3be18dac4bcf4ad0aeb5f5c66747ce6a376bfeede319b` |

The full client freeze also hashes its seven Python source/test members and
research note. Exact private archives retain requests, raw responses, patches,
actual Rust test output, source freezes, host observations and cleanup receipts.
These archives contain local/private data and are not committed here. Tokenizer
and template fingerprints use the runtime measurement's canonical semantic
JSON domains; they are not raw-file SHA256 substitutes.

Host: Apple M5 Max, 128 GiB unified memory, arm64, macOS 27.2 build 26B5112a.
No other model runtime was co-resident. The actual core auto-loaded the matching
projector; all task requests were image-free. Projector residency was retained,
not silently treated as absent because the launcher requested vision off.

## Arms, workload and supervision

Both arms used the same source, binary, GGUF, client, task, route settings and
resource policy. ON set `HF2Q_QWEN_POOL_GROWTH_PROFILE=adr063-24g-v1` and
`HF2Q_UNSAFE_EXPERIMENTS=1`. OFF removed **both** keys from the full environment;
literal `off` is not the supported profile parser contract. Selection is bound by
source and launch environment, not an observed native physical-pool counter.

Common settings: four slot-aware requests, context262144 **per slot**, shared
KV budget51539607552 bytes, AUTO speculation, TQ KV/codebook8, encoder session1,
Qwen launcher width routing mvN0/mv_ext1, repetition1.05, fresh/tool thinking
budgets2048/512, output8192 per request, overflow Reject, loopback service. Actual
wire omissions resolve greedily in this hf2q source; do not import peer defaults.

Each cohort had four independent conversations, a simultaneous barrier, seven
alternating unary/SSE requests and six real tools per conversation: read source,
read tests, write source, test ports, read source, test all. The task fixes
`parse_port` to return errors for zero, invalid, negative and overflowing/out-of-range
values while accepting1 and65535. The immutable Rust fixture and sandbox/toolchain
proofs were common to both arms. The final patch and raw positive test counts,
not merely process exit0, were independently reviewed.

Every engine exit was followed by at least60 seconds of cooldown. Each start
required Nominal thermal state0; runtime allowed0/1 and rejected Serious/Critical
or unknown observations. Same-invocation CSV/text accounting stayed strict.
Only fully corroborated negative-N-page counter overcount could trigger up to
three fresh observations inside the existing total250ms deadline; rejected raw
observations stayed retained and never became admission credit. Valid bad
memory/pressure readings were never retried to obtain a favorable observation.

Both arms used the same operator-approved dedicated-host discovery policy:
preload Free+Cached-files floor90GiB, runtime Free+Cached-files floor3GiB (2GiB reserve plus
1GiB reaction margin), swap-used growth limit512MiB, compressor growth8GiB,
process RSS/footprint abort100GiB, normal pressure,500ms sampling,250ms total
observation bound, maximum1s completed-sample gap and required process ledger
with250ms query/1s age bounds. These are recorded experiment settings, **not** a
newly certified memory-admission formula or recommended default. Cached-file
working estimates are not unique ownership proof; native physical high-water
and formal Q5 closure do not follow from a successful cohort.

## Fixed protocol and raw primary measurements

Five fixed blocks; no optional stopping or adaptive extension. Odd blocks were
OFF,ON,ON,OFF; even blocks ON,OFF,OFF,ON. An entire campaign stopped on any failed
arm, with no replacement or rescore. Independent task review remained required.

Primary metric: `client/summary.json` coordinator elapsed seconds, including
client setup/cleanup and excluding model loading. Supervisor startup/shutdown
wall time is a different metric and was not substituted.

| Block | Modes in position order | Client seconds in position order |
|---|---|---|
| 1 | OFF, ON, ON, OFF | 121.96552483295090, 119.86812120804098, 116.68482987501193, 117.01260549994186 |
| 2 | ON, OFF, OFF, ON | 120.12036841607187, 117.63934608304407, 122.12268274999224, 118.62941054208204 |
| 3 | OFF, ON, ON, OFF | 120.92701512493659, 114.93083933403250, 125.46108137490228, 121.62170954199973 |
| 4 | ON, OFF, OFF, ON | 114.47624674998224, 119.18291595799383, 121.32411250006407, 118.79460745805409 |
| 5 | OFF, ON, ON, OFF | 122.88691900006961, 118.93188695795834, 117.73718108294997, 122.60942641610745 |

Within each block compute `d_b=log(mean(OFF seconds)/mean(ON seconds))`, using
arithmetic means of its two cohorts per arm. Effects are0.010199965347137161,
0.004230831374834185,0.008932022259750325,0.030549070637337108,
0.03661923286778913. The statistical sample is **five block contrasts**, not20
cohorts or80 task conversations.

Mean log effect0.01810622449736958; sample SD0.014463379219069135. With n5,
df4 and `t_0.975,4=2.776445105`, form `mean ± t*SD/sqrt(5)`. Transform point and
endpoints by `100*(1-exp(-d))`. The nominal Student-t interval assumes independent,
approximately normal block contrasts; balanced ordering alone does not prove
those assumptions.

| Criterion | Result |
|---|---|
| Estimated reduction | 1.7943291663% |
| Nominal95% interval | [0.0147556342%,3.5422292051%] |
| Interval half-width | 1.7637367854 percentage points |
| Required half-width≤2.5pp | PASS |
| Required point reduction≥5% and lower bound>0 | **FAIL: point below5%** |
| Quality/prefix/cleanup | PASS for20/20 cohorts |

## Reproduction

The table above suffices to reproduce the statistical calculation without private
data. To independently recompute from an authorized export of the frozen
`cohorts.json` (containing these20rows), use:

```sh
python3 - "$COHORTS_JSON" <<'PY'
import json, math, statistics, sys
rows = json.load(open(sys.argv[1]))
assert len(rows) == 20
values = []
for block in range(1, 6):
    group = rows[4*(block-1):4*block]
    order = ['OFF','ON','ON','OFF'] if block % 2 else ['ON','OFF','OFF','ON']
    assert [r['mode'] for r in group] == order
    assert all(r['block'] == block and r['exit'] == 0 for r in group)
    off = statistics.mean(r['client_elapsed_seconds'] for r in group if r['mode']=='OFF')
    on = statistics.mean(r['client_elapsed_seconds'] for r in group if r['mode']=='ON')
    values.append(math.log(off/on))
mean = statistics.mean(values)
half = 2.776445105 * statistics.stdev(values) / math.sqrt(5)
f = lambda x: 100 * (1-math.exp(-x))
point, lo, hi = f(mean), f(mean-half), f(mean+half)
print({'point_percent':point, 'ci95_percent':[lo,hi],
       'halfwidth_pp':(hi-lo)/2, 'precision_pass':(hi-lo)/2 <= 2.5,
       'benefit_pass':point >= 5 and lo > 0})
PY
```

For a new execution, freeze an independently reviewed packet before starting,
using the pinned client source and workload above. The exact common server
payload, excluding model/run-specific paths, was:

```sh
# Supply an approved binary and the exact verified vanilla GGUF through these variables.
# ON: add profile=adr063-24g-v1 and unsafe acknowledgement=1 to this same environment.
env -u HF2Q_QWEN_POOL_GROWTH_PROFILE -u HF2Q_UNSAFE_EXPERIMENTS \
  HF2Q_ENCODER_SESSION=1 HF2Q_QWEN_SPECULATION=auto \
  HF2Q_TQ_KV=1 HF2Q_TQ_CODEBOOK_BITS=8 \
  HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 \
  "$HF2Q_BIN" -v serve --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --overflow-policy reject --scheduler inflight-batched --max-slots 4 \
  --kv-cache-budget 51539607552 --default-repetition-penalty 1.05 \
  --default-thinking-token-budget 2048 --default-tool-thinking-token-budget 512 \
  --ctx 262144
```

For ON, replace only the leading `env -u HF2Q_QWEN_POOL_GROWTH_PROFILE -u HF2Q_UNSAFE_EXPERIMENTS`
with `env HF2Q_QWEN_POOL_GROWTH_PROFILE=adr063-24g-v1 HF2Q_UNSAFE_EXPERIMENTS=1`;
retain every remaining assignment and argument.

This records server semantics, not a replacement for the approved supervisor's
memory/thermal/process/cleanup controls. The workload entrypoint is
`scripts/four_agent/qwen_n4_live_smoke.py` at the pinned client commit, with exact
approval/source hashes, sandbox proof, compiler/SDK identity, model alias, owned
server PID and isolated output. Follow its [diagnostic contract](adr063-qwen-n4-live-coding-diagnostic.md)
and the [exact-prefix replay contract](adr063-qwen38-prefix-post-oracle.md).
The archived controller/generator are identified above but contain private
machine-specific launch bindings; reproducing the raw campaign requires an
operator-authorized equivalent packet and retained artifacts, not invented
portable approvals. New observations form a new campaign; never append them to
these fixed-five results.

## Quality, exclusions and disposition

All80 conversations completed560 requests and480 actual tools. Every final patch
was independently accepted against the fixed task; raw tests had positive
assertion counts. All80 seven-wire post-oracle replays passed exact stable-prefix
and actual cached-token usage bindings; per-request raw responses and source
hashes were retained. All20 owned server/client cleanups and listener checks
passed. This is specific to this declared workload, not generalized model quality,
isolation proof for arbitrary schedules or physical batch-four GPU execution.

Two earlier campaigns, `qwen38-pool-balanced-20261026T060836Z` and
`qwen38-pool-balanced-20261026T062450Z`, stopped on strict observer-accounting
rejections. They remain failed and were neither rescored nor pooled. Independent
review found zero overlapping cohort run paths, and recomputed every reported
block effect from only the20newcohorts. Later observer generalization did not
retroactively make either failed campaign pass.

The independent review composed disjoint index ranges0–7,8–13,14–18 and19 with
no missing/duplicate run. The final analysis matched independent recalculation.
Keep the bounded profile opt-in; this report supplies no evidence to change
process-wide defaults. A small positive point estimate and narrow interval do
not satisfy the predeclared5% criterion. Other source-bound optimization
hypotheses remain separate work, as do latest-peer/stock-release comparisons,
DeepSeek/Qwen3.6/Gemma results and all-four-model acceptance.
