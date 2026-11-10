# Qwen3.8: why Splash decodes faster, and what hf2q must change

Date: 2026-11-09  
Scope: ADR-063 Q4-QWEN38 / ADR-065  
Host: Apple M5 Max, 128 GiB

## Problem

Splash's Qwen3.8 raw decode is materially faster than hf2q's ordinary joint decode. A feature checklist does not explain the gap. This record reduces the matched-artifact runs and reads the pinned Splash implementation at source level.

The result has two distinct parts:

1. Splash has no valid Q4 comparator yet. Every point in the frozen sweep failed the canonical task gate, so its raw makespan cannot establish a Q4 Win or Loss.
2. Its raw decode lead is real. It comes from producing about 5.3 target-owned tokens per speculative cycle without making that cycle materially longer than hf2q's one-token weight-streaming cycle.

## Frozen identities

| Item | Identity |
|---|---|
| Target | `Qwen3.8-27B-vanilla.gguf`, SHA-256 `cf0eb0103c1ffe7b3702670aa794a669bafbe328b95e892268305dd87c2c30d7` |
| Splash | `031a25a6125af2d7acf7096580afe56eb33d65bc` |
| DFlash2 | `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Splash point S0 | `kv=int8,state=fp32,ane=auto,verify=auto,decode_share=0.5,cache_disk=auto,max_memory=auto` |
| hf2q joint-MTP pilot | source `09961830` plus the right-sized speculative-anchor fix; `HF2Q_QWEN_JOINT_MTP=1`, `HF2Q_QWEN_JOINT_MMA=1` |

Primary receipts:

- Splash corrected sweep: `~/.local/state/hf2q/v0130/q4-qwen38/splash-sweep-strict-parity-v2/selection.json`
- Splash source-attribution S0: `~/.local/state/hf2q/v0130/q4-qwen38/splash-s0-final-51a3923f-v2/`
- hf2q previous default: `~/.local/state/hf2q/v0130/q4-qwen38/hf2q-default-09961830-v1/`
- hf2q joint-MTP pilot: `~/.local/state/hf2q/v0130/q4-qwen38/hf2q-joint-mtp-mma-09961830-v2/`
- hf2q fixed campaign: `~/.local/state/hf2q/campaigns/adr065-qwen38-joint-mtp-cdb180d5-v1/`
- hf2q command timing: `~/.local/state/hf2q/v0130/q4-trace-gpu-timing/`
- ADR-065 E7: `~/.local/state/hf2q/v0130/e7-510/smoke/report.json`

These are local evidence paths, not published release receipts.

## Check 1: Splash has no valid Q4 point

The corrected one-factor sweep contains eight configurations and two runs per
configuration. Its request normalization removes only Splash's semantically
false `strict` tool default and restores that default in the canonical inverse.
All 320 archived turns pass the hf2q byte post-oracle. Prompt rendering is
therefore no longer a validity blocker.

No configuration passes the quality gate:

| Configurations | Task pass fraction |
|---|---:|
| S0, KV BF16, ANE off, matched | 0.000 |
| state BF16, decode share 0.25 | 0.125 |
| cache disk off, decode share 1.0 | 0.250 |

The required floor is 0.75. Every one of the 16 runs lacks `CalibrationPass`
and a passing run audit. In every run at least one lane annotates `u16::parse`
with `std::fmt::Error`, fails compilation with E0271, and stops after the
fourth step rather than completing the required six-step workflow. The
supervisor then cancels unfinished lanes. A few lanes complete the task before
that stop, but the best two-run configuration reaches only two of eight tasks.

Failed clients do not execute Splash's normal closing render proof, so the
parity replay re-rendered their archived requests through the same pinned
model/template endpoint after the runs. The swept knobs affect KV/state
precision, ANE, verification, scheduling and cache placement, not chat
rendering. `splash_sweep.py parity-config` now accepts the authoritative
per-agent receipts embedded in a failed client summary instead of requiring
the success-only per-agent summary files.

The corrected selection record has SHA-256
`74568385b4cb7c913fbf4e2603b199c45b3a54a508398dd98f2e3f0fd1564064`.
The original non-parity sweep remains historical evidence only.

Per ADR-063, a failed or absent Splash arm cannot be turned into a Win against nothing. The raw measurements below are architecture evidence only.

## Check 2: the raw decode advantage

Across the 17 completed S0 requests before the client failure:

| Metric | Splash S0 |
|---|---:|
| Completion tokens | 3,613 |
| Drafted tokens | 4,746 |
| Accepted draft tokens | 2,918 |
| Speculative cycles | 678 |
| Draft acceptance | 61.48% |
| Accepted drafts plus target bonus per cycle | 5.304 |
| Completion tokens per cycle | 5.329 |
| Median per-request decode | 60.05 tok/s |
| Per-request decode range | 48.38–79.48 tok/s |

Splash normally drafts seven tokens per lane per cycle. The unary subset whose complete timing objects are directly readable has a median cycle wall time of 73.30 ms (range 72.28–80.64 ms). The server log reports 4.6–6.6 tok/cycle for the completed turns.

hf2q's neutral command-timing trace explains why this matters. At four lanes, an ordinary joint target step has median span 89.08 ms, 87.93 ms GPU busy, and only 0.53 ms of GPU bubbles. Sixteen uniform four-layer chains consume about 83 ms and the output head 3.62 ms. Width one still takes 83.66 ms. The path is weight-streaming-bound; removing host scheduler overhead cannot provide the missing factor.

Splash gets several target-owned tokens from a cycle in roughly the time hf2q streams the target weights for one token. That is the lead.

## Source-level differential

### DFlash2 proposes a coherent seven-token path

The pinned DFlash2 checkpoint declares:

- five sliding-attention layers at target layers 5, 19, 33, 47 and 61;
- hidden size 5,120, intermediate size 17,408, 32 query heads, eight KV heads, head dimension 128;
- block size eight and seven proposal positions;
- a two-tap dynamic convolution with group size 16;
- selector rank 256 and top-k 16;
- vocabulary 248,320 and sliding window 2,048.

The raw BF16 checkpoint is 3,848,817,896 bytes. Relative to hf2q's older DFlash implementation it adds the selector projection and predecessor/successor codebooks plus per-layer dynamic-convolution kernel bases and projections. The old hf2q parser expects the older config layout and 58-tensor manifest; it cannot parse or execute this 81-tensor DFlash2 checkpoint.

Splash's `DFlashDraft::addDecode` executes eight rows per lane through one graph. `DraftSelector` then:

1. takes top 16 candidates at proposal rows 1–7;
2. scores adjacent 16×16 transitions with the rank-256 predecessor/successor codebooks;
3. selects one coherent path across the block;
4. emits confidence for each proposal position.

This is not seven independent argmax calls. The path selector is the main proposal-quality difference from hf2q's one-layer recursive native MTP head.

### Verification is confidence- and cost-priced

`VerifyBudget` keeps separate constrained/unconstrained online tables by lane count. It learns:

- cycle cost as a function of total verify rows;
- proposal acceptance from confidence and proposal position;
- periodic exploration to keep the model identified.

For four lanes its seed relative costs are 0.40, 0.47, 0.67 and 1.0 at 1, 2, 4 and 8 rows per lane. The selector can choose 1–8 target rows per lane. Grammar-constrained cycles use their own table.

hf2q's pilot has a fixed total budget of eight rows. With four active lanes, four rows are anchors and only one draft per lane fits. Deeper windows appear only as lanes leave. That caps a full cohort near two output tokens per lane-cycle even when the MTP draft is almost always right.

### The cycle uses three overlapped command stages

Splash does not execute the whole cycle as one command. `VerifyTicket` advances through draft/selector/budget, target forward, and commit commands. After the draft command the host reads the proposals required by constrained lanes and requests CPU grammar simulation for the anchor plus all seven proposals. It submits target verification immediately, overlaps that GPU work with mask computation, uploads the masks when ready, then submits sampling, acceptance, target-state commit and draft-ring/context commit together. Unconstrained lanes skip the mask wait. Tools therefore do not disable deep speculation.

hf2q's ordinary timing already has almost no GPU bubble, so merely reducing command boundaries is not enough. Splash's material advantages are streaming target weights across a useful temporal block and hiding grammar-mask latency behind that target work.

### Scheduling protects semantic token latency

Splash's `decode_share=0.5` charges half of contended prefill wall time as decode debt. Equal-priority decode runs until it pays that debt. Prefill also shrinks toward a 500 ms slice under contention. This prevents a long prefill from starving semantic decode tokens.

This is relevant to TTFT and SSE cadence, but it does not explain the 5.3 tok/cycle decode factor.

## hf2q joint-MTP pilot

The first pilot exposed a real serving bug: a stable prompt anchor retained an MTP hidden-row view whose owner covered a larger transient multi-row allocation. The anchor store correctly refused it, so no stable checkpoint was published and prefix reuse failed. The fix materializes only the retained row into a right-sized owner. Its focused regression passed, and the repeated real Q5 passed with no checkpoint refusal or missing-checkpoint warning.

The four-point factorial pilot used one binary, artifact, case and host. All four runs reached `CalibrationPass`, completed 28 turns, reused the full eligible prompt prefixes and produced the same final source-hash multiset:

| Path | Client makespan | Completion tokens | Aggregate completion rate | Median request decode |
|---|---:|---:|---:|---:|
| Previous default | 137.458 s | 6,310 | 45.905 tok/s | 13.042 tok/s |
| Column-invariant MMA only | 124.568 s | 6,388 | 51.281 tok/s | 14.615 tok/s |
| Joint MTP only | 130.161 s | 6,310 | 48.478 tok/s | 13.674 tok/s |
| Joint MTP + column-invariant MMA | 94.188 s | 6,388 | 67.822 tok/s | 21.046 tok/s |

MMA alone is 9.38% sooner and joint MTP alone 5.31% sooner than the previous default. Together they are 31.48% sooner. Joint MTP is 24.39% sooner when added to MMA; MMA is 27.64% sooner when added to joint MTP. The log-scale interaction ratio is 0.799, so treating these as independent improvements would miss a further 20.15% reduction. This matches the mechanism: joint MTP creates wider temporal products, and the column-invariant few-row kernel prevents those products from paying the canonical width growth.

The combined pilot executed 857 global joint-MTP cycles and 3,172 lane-cycles. The fixed allocator issued an estimated 3,556 drafts. Subtracting one prefill seed per request and one target decision per lane-cycle from the 6,388 completions gives 3,188 accepted drafts: about 89.65% acceptance and 2.01 output tokens per lane-cycle. This estimate assumes the production seed/settle accounting; terminal rows can distort it slightly.

The fixed-count one-switch campaign subsequently confirmed that the practical
premise behind ADR-065 E8's `f=0` does not describe the production integration.
Across six fixed n4 blocks the combined candidate reduces makespan by 34.600%
(95% CI 33.738–35.452%). All 70 A/B runs pass Q5, task quality, thermal and
fairness gates; cancellation recovery passes with byte-exact survivors. The
official analysis is `Win`, SHA-256
`0f1a077bc5484cb190257ac5c46dd22a8d71f990dd6424dc157fd6841882f05c`.
The operator superseded E8 for this measured J4/J5 design and promoted it as
the dense-MTP default, with `HF2Q_QWEN_JOINT_MTP=0` as the kill switch.

It also does not match Splash's raw cycle productivity:

| Engine/path | Proposals at four lanes | Accepted/bonus output per lane-cycle | Cycle scale |
|---|---:|---:|---:|
| hf2q previous default | 0 | 1 | 89 ms traced median |
| hf2q joint-MTP pilot | 1 per lane | about 2.01 | same order; Q5 pilot |
| Splash DFlash2 S0 | 7 per lane | 5.304 | about 73–81 ms on readable completed requests |

## Decision

1. Do not claim a matched speed Win against Splash. The fully prompt-equivalent
   sweep has no valid Splash point, so policy provides no competitor speed
   denominator.
2. Ship joint MTP plus column-invariant MMA as one default candidate treatment
   on dense artifacts with an MTP head. The fixed campaign, not the factorial
   pilot, is the promotion evidence.
3. Keep every target token target-verified and every tool/grammar path enabled.
   No quality waiver converts Splash's raw speed into a Q4 result.
4. Treat Splash's 5.304 accepted/bonus outputs per lane-cycle as the
   architectural target for a successor, not as a v0.1.30 comparison result.

## Required successor to match raw Splash

Matching the raw Splash path requires all of the following, not one isolated optimization:

- a native DFlash2-compatible drafter or another measured proposer with comparable deep-prefix quality;
- at least seven proposal positions per lane and a coherent path selector;
- a target verifier that handles the resulting four-lane temporal block near the current 80–90 ms weight-streaming cycle, rather than scaling toward ADR-065 E7's 219.055 ms at 4×4;
- confidence/cost-priced verify rows, with separate grammar-constrained statistics;
- graph-resident sampling, acceptance and target/draft state commit;
- a scheduler debt rule that bounds semantic-token starvation during contended prefill;
- the existing Class A cohort, Class B decision-envelope, transaction, Q5, task-quality, fairness and cancellation gates.

ADR-065 currently forbids new drafter architectures and tree drafting. A DFlash2 port therefore requires an explicit successor ADR or an operator amendment; it must not be smuggled into J4. Any new Metal verifier/selector primitive belongs in mlx-native, published and exactly pinned before hf2q qualification.

## Next checks

- Promoted source `6adfbcde` passes Q5 with the switch unset in 92.792 s.
  Its startup record has `explicit=false`, `joint_mtp=true` and
  `column_invariant_mma=true`; cleanup is proven.
- Complete the remaining v0.1.30 family gates; do not rerun or select a Splash
  point from this frozen sweep.
- If raw Splash parity remains a product requirement after v0.1.30, approve
  the successor drafter/verifier scope explicitly and measure it against the
  5.304 tok/cycle target above.
