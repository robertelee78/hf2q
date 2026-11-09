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

- Splash S0: `~/.local/state/hf2q/v0130/q4-qwen38/splash-s0-final-51a3923f-v2/`
- Splash sweep: `~/.local/state/hf2q/v0130/q4-qwen38/splash-sweep-v1/selection.json`
- hf2q previous default: `~/.local/state/hf2q/v0130/q4-qwen38/hf2q-default-09961830-v1/`
- hf2q joint-MTP pilot: `~/.local/state/hf2q/v0130/q4-qwen38/hf2q-joint-mtp-mma-09961830-v2/`
- hf2q command timing: `~/.local/state/hf2q/v0130/q4-trace-gpu-timing/`
- ADR-065 E7: `~/.local/state/hf2q/v0130/e7-510/smoke/report.json`

These are local evidence paths, not published release receipts. The pilot is not a release verdict.

## Check 1: Splash has no valid Q4 point

The frozen one-factor sweep contains eight configurations and two runs per configuration. None is valid:

- every configuration has `valid: false`;
- seven configurations have task pass fraction `0.0`;
- the ANE-off configuration has task pass fraction `0.125`;
- every run lacks `CalibrationPass`, a passing run audit, and the required prompt-parity post-oracle.

S0 itself ended `ClientFailed`. All four agents produced the same invalid Rust edit, annotating `u16::parse` with `std::fmt::Error`; compilation failed with E0271. The supervisor then cancelled the remaining agents. This is a quality failure, not a speed win.

The frozen sweep is not prompt-equivalent. A parity-specific replay of S0 found its first mismatch at byte 532: Splash rendered `"strict": false` for each of the three tool functions while hf2q omitted that semantic default, adding 51 prompt bytes. The adapter now omits only this false default from Splash's wire request and restores it for the canonical hf2q request. A replay of every completed turn from a corrected Q5 run passed the hf2q byte post-oracle: 20 of 20 rendered prompts were identical.

That corrected Q5 still ended `ClientFailed`. Two agents emitted `std::num::ParseIntError` and passed both requested test targets; two retained the invalid `std::fmt::Error` annotation. The resulting two-of-four success is below the 0.75 task floor, and one agent stopped after the failed fourth tool step instead of completing the required six-step sequence. One corrected run cannot attribute the output change to the schema fix, but it rules out prompt rendering as the remaining failure in that run. Splash also cannot reproduce hf2q's 1.05/64 repetition-penalty semantics and therefore sends no penalty, as frozen by the Splash design. The original sweep remains invalid and is never relabelled.

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

This overturns the practical premise behind ADR-065 E8's `f=0`: the production integration can make joint-window eligibility nonzero on the canonical grammar/tool workload and delivers a large factorial-pilot effect. It does not by itself overturn the decision. ADR-065 requires a fixed-count one-switch A/B, fairness, quality and independent review before J4/J5 can be reopened or a default changed.

It also does not match Splash's raw cycle productivity:

| Engine/path | Proposals at four lanes | Accepted/bonus output per lane-cycle | Cycle scale |
|---|---:|---:|---:|
| hf2q previous default | 0 | 1 | 89 ms traced median |
| hf2q joint-MTP pilot | 1 per lane | about 2.01 | same order; Q5 pilot |
| Splash DFlash2 S0 | 7 per lane | 5.304 | about 73–81 ms on readable completed requests |

## Decision

1. Do not claim that hf2q has beaten Splash. Splash has no valid point, while hf2q has not matched its raw decode architecture.
2. Preserve joint MTP plus column-invariant MMA as one candidate treatment, not a default. The factorial pilot shows that neither component represents the combined effect.
3. Give the combined treatment one production kill switch, then run ADR-065's fixed-count A/B, fairness and quality gates. A single successful run is not qualification.
4. Keep every target token target-verified and every tool/grammar path enabled. No quality waiver converts raw speed into a Q4 result.

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

- Bind joint MTP plus column-invariant MMA to one candidate switch and run ADR-065's fixed-count ABBA campaign against the previous default.
- Run the current Q4 peer campaign on the same release candidate if the A/B passes.
- Run the Splash prompt-parity post-oracle. Until one Splash point passes Q5, audits and the 0.75 task floor, its Q4 comparison remains failed rather than a speed denominator.
- If raw Splash parity remains a product requirement after v0.1.30, approve the successor drafter/verifier scope explicitly and measure it against the 5.304 tok/cycle target above.
