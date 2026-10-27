# Gemma checkpoint matched component timing

The fixed 20-cohort study measured a **39.87% reduction** in
streamed semantic time to first token with the full-input checkpoint enabled,
relative to the same hf2q binary with it disabled. The paired-block 95% bootstrap
interval is **39.52–40.34%**;
its half-width is 0.413 percentage points, below the frozen 2.5-point target.
This is a Gemma checkpoint component result. It does not satisfy the campaign's
four-model comparison against the latest valid peer, and does not promote the
experiment by default.

## Frozen comparison

- Server source: `18162bb702806ac6c083bf01ec80f4675207fda4`; clean release SHA256
  `ff655efd5013a11ad608330f35b3622c2e448f019dcc4e94c8c483eb03cd09e1`.
- Published dependency: mlx-native 0.15.5. Both arms use that identical retained
  clean binary; the later 0.15.6 adoption is not relabeled as this execution.
- Gemma 4 26B A4B It artifact: SHA256
  `bf65b7e163216885790442d0324839101ba4b5b49f7be4ec0cf3d7c5c538ef53`.
- Four slots, 262144 context, 8192 output cap, Hybrid/TQ8 cache, LCP8/KV48,
  repetition penalty 1.05; identical model, sampling, templates and server settings.
- OFF removes both checkpoint and unsafe-ack environment variables. ON sets both
  to `1`. Five fixed OFF/ON/ON/OFF blocks give ten cohorts per arm, without
  replacements, extensions or selecting a favorable subset. Every cohort gets
  the canonical BOS-prefill/one-decode warmup and a 30-second post-cleanup cooldown.
- Primary component metric: geometric mean of semantic first-token time for
  streamed turns 1, 3 and 5 across all four agents (12 observations per cohort).
  Unary elapsed time is not presented as semantic TTFT. Bootstrap: 10000 draws,
  block resampling, seed 6306, median ON divided by median OFF.

The primary ON/OFF ratio is 0.601285391. The secondary
coordinator-wall ratio is 0.772222376
(22.78% lower); it is descriptive component evidence,
not the campaign's matched-peer makespan/goodput acceptance result.

## Correctness and comparability

All 20 cohorts completed their fixed four-agent coding protocol: 560 HTTP200
turns, 480 actual tool calls, 400 successful verified compiler/test processes,
and 80 correct final patches. Tool normalization first executes real tools and
retains their raw outputs/process/artifact receipts, then exposes stable results
from actual source hashes, compiler artifacts and test counts to the model.
It never substitutes a successful response for a failed tool.

All paired production input vectors match exactly: ten OFF/ON pairs, four agents,
seven complete u32-token vectors each. There is no hash-only, edited-transcript
or selected-turn surrogate for equality. After all model runtimes unloaded,
20 neutral actual renderer exports and 40 ON family-specific seven-wire replays
passed. OFF retained its original strict anchor. ON covered the preceding
accepted anchor on every edge: the initial strict anchor on turn 1, then full
tool-input checkpoints on later turns, with no minus-one allowance. The generic
strict oracle is unchanged.

All 725 saved runtime/post thermal/power observations were Nominal0, low-power
mode off and the frozen AC source, within the 250ms observation budget. Every
server/client/tool cleanup gate passed. Exactly one hf2q runtime was used at a
time. The earlier passive-interface type failure is retained separately; a
narrow exact enum-string bridge repair passed a model-free capability check
before the first timing cohort.

Independent review rehashed all 26528 final study artifacts without a mismatch,
verified all 60 actual post-unload jobs, and recomputed the complete frozen
bootstrap result exactly. Its scoped component verdict passed; the calculation
itself deliberately remains non-self-qualifying.

## Reproduction and retained identities

Use the checked-in matched profile, neutral exporter and pure statistics module
with the source-bound private fixed20 controller/approval packet. The controller
refuses missing bindings, input drift, an existing once-marker or study output,
unknown process ownership, power/thermal changes and incomplete quality/proof
results. Its immutable plan admits no resume or replacement cohort.

The following model-free command recomputes the frozen numerical calculation
from retained cohort receipts and actual exported vectors. It does not rerun or
independently qualify the tools, model or cleanup evidence:

```sh
PYTHONPATH=scripts/four_agent python3 - /absolute/path/to/gemma-matched-fixed20-study-v1 <<'PY'
import json, sys
from pathlib import Path
from gemma_matched_stats import PLAN, analyze
root = Path(sys.argv[1])
runs, vectors = [], []
for plan in PLAN:
    index = plan["index"]
    raw = root / f"cohort-{index:02d}" / "raw"
    receipt = json.loads((raw / "receipt.json").read_text())
    client = json.loads((raw / "client/summary.json").read_text())
    runs.append({**plan,
        "quality_verified": receipt["protocol_complete"] and receipt["final_verified"],
        "cleanup_verified": receipt["cleanup_proved"],
        "primary_sse_geomean_seconds": client["benchmark_metrics"]["primary_sse_geomean_seconds"],
        "coordinator_wall_seconds": client["elapsed_seconds"]})
    proof = root / "post" / f"neutral-{index:02d}" / "proof.json"
    vectors.append(json.loads(proof.read_text())["vectors"])
print(json.dumps(analyze(runs, vectors), indent=2))
PY
```

Evidence packet: `gemma-matched-fixed20-packet-v1`; study:
`gemma-matched-fixed20-study-v1`, beneath the ADR063 campaign evidence root.
Raw machine paths, weights and generated data are not repository artifacts.

| Retained file | SHA256 |
| --- | --- |
| Calculation | `d6ce7291093ae59a6774fa548338d39d753bb84f20ebad66ed70efcfbfc8d13a` |
| Master source | `ced26a26d8b19f23be4f7650bef9deaabef114c74832a7f41de25682eaee702c` |
| Whole-study approval | `40bfa6a8cf7a4362b2c8c570818d93538b4511f7c9606830fd7b413685973c6d` |
| Final ledger | `d8dfbf776d15984abd2cc571b0ad2c1b076bf4a3b4a2772d12032107237fd53a` |
| Artifact manifest | `9196b79a72d29e6d7c5d49be13b54ff5c1d2964e80b1ef889ae6ce0b24a990ad` |

The metadata helper binary is
`23db37416ebc9a7f815dbf0cd13ef729a50c78eb376dfe54cb89c1aa8d2b0851`,
compiled from `5415f2f6abe1b54f9182f3ad5f9f994da1c52ea6` with documentation-only
uncommitted changes; six exporter controls passed. Its selected production
renderer/checkpoint units equal the executed server source. It is not a timing
runtime and executes only after final unload.

Replay this historical packet with the archived helper or its exact `5415f2f6`
source. Later integration onto main keeps the original source guards; a newly
compiled helper with different source units must not silently qualify this
recorded server. Current-integration pure checks and historical actual replay
proofs are distinct evidence.

No largest-context stress, cancellation acceptance, general physical-memory
admission, all-model win, final peer comparison or default promotion follows
from this component study. See the [study boundary](adr063-gemma-matched-study-boundary.md)
and the four-agent correctness proof on the checkpoint implementation branch.
