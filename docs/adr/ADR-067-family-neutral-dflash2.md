# ADR-067: Family-neutral DFlash2 speculative decoding

- **Date:** 2026-11-10
- **Status:** Accepted; implementation authorized for the v0.1.30 program
- **Decider:** Robert E. Lee (product owner), 2026-11-10
- **Scope:** every enabled hf2q text-generation architecture; Qwen3.8 is the first qualified checkpoint
- **Updated:** 2026-11-12 — correct the family binding of the retained Splash arm; preserve ADR-063's existing mechanism, stock, peer, quality and serving gates.

## Problem

ADR-063 originally required a valid Qwen3.8 comparison against Splash for Q4. The corrected
Splash campaign cannot supply one: all 16 corrected Splash runs preserved prompt,
artifact, and output-token parity, but every run failed the coding-task floor and
therefore lacked `CalibrationPass`. A failed comparator cannot count as a Win.

The same campaign still exposes a real architecture gap. hf2q's promoted
joint-MTP route produces about 2.01 output tokens per lane-cycle. Splash's pinned
DFlash2 route produced 3,596 accepted-plus-bonus tokens over 678 lane-cycles,
or 5.304 tokens per lane-cycle. hf2q's ordinary dense target step is already
weight-streaming bound at about 89 ms, with only about 0.53 ms of GPU bubble.
Scheduling or another single-row kernel does not close a 2.64x temporal-product
gap.

The release contract and the product goal are separate:

1. v0.1.30 cannot claim a Splash Win without a valid Splash denominator or an
   explicit operator rescope.
2. Matching Splash's raw decode mechanism requires a new drafter and verifier
   architecture that ADR-065 deliberately excludes from its bounded J4 scope.

## Source-bound mechanism

The implementation reference is pinned Splash commit
`031a25a6cc18eed83a99a9883de91dcaa272f160`, licensed Apache-2.0. The checkpoint
is pinned as `z-lab/Qwen3.8-27B-DFlash2` revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, also Apache-2.0. The raw BF16
checkpoint is 3,848,817,896 bytes and has 81 tensors.

The checkpoint contract is:

- five sliding-attention draft layers, hidden size 5,120, 32 query heads, eight
  KV heads, head dimension 128, and a 2,048-token window;
- target hidden captures after layers 5, 19, 33, 47, and 61;
- an eight-row block: one anchor plus seven proposals;
- two-tap dynamic convolutions, group size 16, in both the attention and MLP
  paths of every draft layer;
- a rank-256 hidden projection and two `[248320, 256]` BF16 predecessor and
  successor codebooks;
- 16 candidate tokens per proposal position.

Splash executes all eight draft rows per lane in one graph. At each of the seven
proposal positions it takes the top 16 unary candidates. Its selector scores
every adjacent 16x16 edge by multiplying the projected hidden row with the
predecessor token code and successor candidate code, then walks one coherent
path. The selected path and per-position confidence feed a verify-row budget.
This is materially different from seven independent argmax decisions.

For the pinned dense Qwen3.8 target, Splash verifies every row. Its confidence
budget matters to MoE targets but does not reduce this dense target's work.
The immediate hf2q requirement is therefore a four-lane, eight-row verifier:
up to 32 target rows in one weight-streaming pass. The current production joint
verifier accepts at most eight rows; its diagnostic exact path accepts 16; the
column-invariant MMA path rejects more than eight.

Splash further overlaps grammar simulation with the target GPU forward and
commits accepted target and draft state on graph. Decode-debt scheduling protects
semantic decode under prefill contention. Those mechanisms affect TTFT and
cadence, but the coherent seven-token proposal and temporal target block explain
the throughput factor.

The DFlash2 model card independently reports acceptance lengths of 4.10 to 5.46
across MT-Bench, code, and math tasks on H200. The local pinned Splash run's 5.304
accepted-plus-bonus tokens per lane-cycle is consistent with that mechanism;
the H200 throughput numbers are not transferable to Apple Silicon.

## Family scope and checkpoint availability

The runtime architecture is family-neutral. `dflash2` is not a Qwen namespace
and must not encode Qwen target state in its core interfaces. Every enabled
text-generation architecture is in scope through an explicit adapter:

| hf2q generation architecture | DFlash2 target adapter | Published compatible checkpoint at decision time |
|---|---|---|
| `qwen35` dense, including Qwen3.8 | dense recurrent/attention capture and verify | Yes: `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| `qwen35moe`, including Qwen3.5/3.6 | MoE recurrent/attention capture, routing and confidence-priced verify | Yes for Qwen3.6: `incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b6923ad6c14cb1bb41c37a9041446496ab6`; no compatible Qwen3.5 checkpoint found |
| `gemma4` dense and MoE | Gemma hybrid-cache capture, verify and rollback | Yes for `google/gemma-4-E2B-it` and `google/gemma-4-31B-it`; no checkpoint matching the release-gate `gemma-4-26B-A4B-it` artifact found |
| `deepseek4` | DeepSeek recurrent/attention capture, MoE verify and rollback | No compatible DeepSeek-V4 checkpoint found |

Embedding-only architectures do not decode and are excluded. Conversion-only
architectures enter this table when their generation surface is enabled; they
must not be routed through an approximately compatible adapter.

The initial collection search understated availability. The operator harness
already pins the Apache-2.0 Qwen3.6 checkpoint above: its 1,052,513,456-byte
BF16 safetensors file has SHA256
`d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d`,
and its config SHA256 is
`86d2e8e57e7c70bb867240708d00c0535ae9ed3c828dbdcb21584471c7bbce59`.
It uses six draft layers and target captures after layers
`[1, 6, 11, 16, 22, 27, 32, 37]`. Qwen3.6 is therefore an available second
qualification path, not an unavailable-family placeholder.

Two Apache-2.0 Gemma checkpoints also establish that the shared format is not
Qwen-only. `rasyosef/gemma-4-E2B-it-dflash2@6eb3ade2e4ae841974d692e9edc177dcaa7e20ff`
publishes its training command, state and validation metrics; the latter reports
3.6595 accepted draft tokens and 4.4580 expected accepted length. Its
646,093,904-byte weights have SHA256
`842a33bc4f38cc054dc5df0784ee332b49b2a824956d60795d8e94f269024fee`.
`RedHatAI/gemma-4-31B-it-speculator.dflash2@b0cd01b4918aae6b6388272950c4f9851734674b`
publishes the same 81-tensor names as the Qwen3.8 checkpoint with Gemma-bound
dimensions; its 5,494,652,680-byte weights have SHA256
`3e82ec115f31020a81a69c6145aee727996ea9d83664f6285728e6bff0dd6b0f`.
Both use the nested `speculators` config envelope, which requires a separate
versioned config schema even though the tensor naming convention is shared.

The search found no DFlash2 checkpoint compatible with DeepSeek-V4 and none for
the release-gate Gemma 4 26B-A4B target. A family or target without compatible
trained weights can receive the shared runtime and explicit adapter, but hf2q
must report DFlash2 as unavailable until source-bound weights and their
training/export provenance exist. Synthetic or cross-family weights cannot
qualify a production path.

The implementation order follows available evidence, not family preference:
the shared checkpoint/runtime/selector/transaction layer first; Qwen3.8 and
Qwen3.6 end-to-end because their exact weights and target artifacts exist;
Gemma adapters against an exact supported target/checkpoint pair next; and
DeepSeek when compatible weights exist. This is one cross-family program, not a
Qwen implementation followed by uncommitted ports.

## Prior hf2q work: reusable and rejected

`src/inference/spec_decode/dflash/` implements the older 58-tensor DFlash format.
It provides useful ownership patterns for checkpoint parsing, draft KV rings,
target hidden capture, rejection sampling, and rollback. It does not parse or
execute this 81-tensor checkpoint and must not be silently extended behind the
old type names.

ADR-034's old Qwen DFlash experiment is not a production base. Its block-four
route peaked at 23.17 tok/s, 0.77x the MTP baseline; its proposals lost coherence;
and its older batched target verifier diverged and produced garbled long output.
The successor must reuse only independently requalified components.

ADR-065's promoted Qwen joint verifier is the target-side semantic base. It
already proves per-lane sequential DeltaNet state, full-attention single-row
routing, exact spec-off row decisions, prefix commit/replay, lane-local failure,
and cancellation. Its row budget and projection kernels must be widened rather
than reintroducing ADR-034's verifier semantics.

## Decision

Implement a new family-neutral `dflash2` subsystem with explicit core and
per-family types and a default-off feature switch until that family's gates pass.
Keep the old DFlash module unchanged until each corresponding adapter is
qualified; then delete only code made obsolete by that adapter's clean cutover.
Qwen3-VL remains outside the enabled generation surface and must stay fail-closed.

Production remains Rust plus `mlx-native`. It must not shell out to Python,
Splash, SGLang, vLLM, or another engine. External programs are allowed only in
the pinned parity or checkpoint-provenance harness.

### Ownership split

hf2q owns:

- exact checkpoint/config validation and model-family binding;
- a family-neutral checkpoint descriptor and explicit target-adapter trait;
- per-family hidden-capture, verifier-state and rollback implementations;
- draft and target cache lifetimes;
- lane scheduling, grammar state, sampling policy, acceptance, cancellation,
  prefix reuse, and state rollback;
- OpenAI request/SSE semantics and audit receipts.

`mlx-native` owns new Metal graph primitives:

- grouped two-tap dynamic convolution with residual finish;
- top-16 unary extraction, rank-256 edge scoring, coherent-path selection, and
  confidence output;
- row-invariant dense projections and target verify support through 32 packed
  rows;
- graph-resident accept/retained-row compaction where host readback would split
  the cycle.

Any required `mlx-native` change must be published, exactly pinned in
`Cargo.toml` and `Cargo.lock`, and validated from a clean source tree. An ignored
local patch cannot qualify the result.

### Runtime contract

One decode cycle performs the same transaction for every family. Block size,
target capture layers, hidden width, selector dimensions and verify limits come
from a validated family/checkpoint descriptor and use bounded preallocated
storage. Qwen3.8's descriptor is eight rows and seven proposals; a different
shape is never inferred from a nearby model.

1. Gather each lane's anchor, declared target hidden captures, draft KV ring,
   output budget, and sampling/grammar state through the family adapter.
2. Run one batched DFlash2 graph for the descriptor's rows per active lane.
3. Run the coherent-path selector and emit the descriptor's proposals plus
   confidence per lane.
4. Begin CPU grammar simulation for each lane's anchor plus proposals while the
   target graph starts.
5. Run one family target joint verify over the active lane windows, preserving
   each row's serial spec-off recurrent, attention, routing and logits state.
   Qwen3.8 requires up to 32 rows for four lanes.
6. Apply forced-thinking, greedy, sampled, logprob, and grammar decisions against
   the same predecessor state as spec-off; accept only the matching prefix and
   one canonical mismatch or bonus token.
7. Commit the retained target rows and accepted draft-ring state atomically.
   On lane-local error, restore that lane. On command-buffer or supervisor
   failure, fail-stop the worker.
8. Queue all accepted semantic tokens so SSE emits them in order without an
   empty-event substitute for TTFT.

No token is committed without target verification. Greedy output must be
bitwise spec-off. Sampling must preserve the target distribution under the same
seed and policy. Tool calls, tool-result continuation, grammar constraints,
thinking budgets, EOS, length limits, cancellation, and prefix reuse remain
first-class paths.

## Implementation gates

Each gate is a stop/go boundary. A failed spike remains evidence, not landed
production code.

### D0: checkpoint and source contract

- Validate every required config field and every tensor name, dtype, shape, and
  byte range against a versioned checkpoint schema. The Qwen3.8 schema has 81
  tensors; another family or DFlash2 revision gets its own explicit schema.
- Reject wrong target architecture, target checkpoint identity, target layer
  list, vocabulary, hidden size, block size, selector dimensions, convolution
  dimensions, or draft revision.
- Record draft checkpoint and provenance, target GGUF, reference source, hf2q
  source, family adapter and native dependency hashes in every campaign.

### D1: drafter parity

- Implement the five-layer draft forward and sliding KV ring.
- Compare layer outputs, dynamic-convolution outputs, logits, selector hidden,
  top-16 candidates, edge tables, selected paths, and confidences against a
  pinned reference fixture generated by the Apache-2.0 implementation.
- Use deterministic prompts spanning short context, a 2,048-token wrap, multiple
  lanes, and context restore. Define numeric tolerances per tensor before the
  implementation run; never widen them after observing a failure.
- Reject the path if proposal acceptance on the Q4 workload does not materially
  exceed the promoted one-token MTP route.

#### D1 reference-oracle correction and measured status

The first fixture spike used the checkpoint's end-to-end BF16 PyTorch graph as
the numeric oracle. It was rejected before qualification: hf2q's existing
`mlx-native` contract keeps draft activations in F32, packs attention Q/K/V and
the sliding KV ring in BF16, and executes the target GGUF vocabulary head with
its native quantized kernel. The BF16-only oracle therefore tested a different
precision contract. Adding F32-to-BF16 round trips after every operation made
both numeric and candidate parity worse and added avoidable per-layer kernels;
that failed spike was removed.

Fixture schema `hf2q.dflash2-reference-fixture.v2` instead runs the pinned
reference with F32 draft activations and BF16 packed attention/cache. The
isolated second-lane case is the numeric calibration case and is excluded from
the holdout claim. Its frozen limits are: input max-absolute `0.03`, layer
relative max-absolute `0.05`, final-hidden max-absolute `2.5`, logit
max-absolute `2.0`, selector-hidden max-absolute `1.1`, at least 10 common
members in each top-16 candidate set, and at least four matching positions in
each seven-token path. The short, 2,048-token wrap, two-lane and restore cases
are the holdouts. Their frozen fixture SHA256 values are, respectively,
`44ff640455fae488cde9fd70e8861296575326596eaee5148c61c595eb2e5002`,
`597020bfafc62647dbf0f653b554bf8c2aa004c3dbb51c0365632a4f6a1ab8d2`,
`00682c8dd2cbd9b3f1e28f5476993ad867812a60d2f34ccd2181bf9fd009a06c`,
and `9d041ae20e9f4a39368e6953ad54cf7dbdb5368a8524c1edb45ff4667f636595`.

The v2 holdouts pass those frozen limits. Short, wrap and restore match all
seven selected path positions; the two-lane case matches `7/7` and `4/7`.
Candidate overlap is at least `10/16` at every position. This proves bounded
operational-precision parity for the forward, ring and coherent selector. D1
remains open until aligned edge/confidence comparison and the Q4-workload
acceptance gate are complete; these measurements do not promote the route.

### D2: 32-row target verifier

- Extend the ADR-065 verifier to four independent windows of one to eight rows,
  32 rows total. DFlash2 always executes a fixed width-32 tile; inactive rows
  remain masked and cannot change a live lane's bytes.
- Freeze width-32 verification as a Q1 Class B route against serial spec-off
  before collecting parity data. For every target-logit row, require
  `max_abs_delta <= 0.25` and exact argmax equality. Record hidden-state
  max-absolute, mean-absolute and cosine metrics, but do not create an
  elementwise hidden gate that the governing Q1 policy does not define.
- Require bitwise cohort invariance for each live lane across solo, mixed-length,
  full four-lane and lane-permuted width-32 tiles. For MoE targets, also record
  selected expert IDs and router margins per row.
- Prove DeltaNet sequential state, full-attention KV positions, and mixed lane
  lengths, then prove exact retained-prefix cache state after accepting zero
  through seven proposals.
- Keep the promoted route available behind `HF2Q_DFLASH2=0` (the one kill
  switch; D6) until the new default passes Q5.
- Stop if a 32-row target forward scales approximately with rows instead of
  streaming the target weights once. The architecture only wins if temporal
  batching amortizes the target.

### D3: acceptance and state transactions

- Prove accept-prefix, mismatch bonus, EOS inside the window, output-budget
  truncation, forced thinking, grammar rejection, sampled decisions, logprobs,
  cancellation at every stage, and lane-local failure.
- Prove target and draft rollback after zero through seven accepted proposals.
- Prove repeated multi-turn tool calls and tool-result continuation preserve the
  unchanged prefix KV state and never leak state across conversations.

### D4: graph and scheduler integration

- Remove avoidable per-cycle host copies and synchronizations.
- Overlap grammar simulation with target verification; measure the overlap rather
  than inferring it from command count.
- Add decode-debt scheduling only as a separately attributed experiment after
  D1-D3 pass. It must not obscure drafter/verifier attribution.

#### D2-D4 implementation-candidate status

The first production-shaped candidate is deliberately default-off. It provisions
only when `HF2Q_DFLASH2=1` and both `HF2Q_DFLASH2_CONFIG` and
`HF2Q_DFLASH2_WEIGHTS` match a pinned descriptor and the loaded target exactly.
Target binding is the hf2q source-bundle receipt, not architecture dimensions or
the operator's filename. The pinned exact-source identities are
`73ded708c49c2d0a47c790ce1d6181e848ac7591dab741de83dbb57218cc6873`
for Qwen3.8 and
`851811f43867ae177636f0c536bb98c33731d992ce357831809a1eda5eb974ca`
for Qwen3.6. External GGUFs, a different source receipt and descriptors without a
qualified target receipt are rejected before draft weights upload. A startup
smoke with the release-gate `qwen36-abliterix-t63-APEX` GGUF exercised this
boundary and was rejected as intended; matching dimensions did not admit the
fine-tuned target. (Superseded by the operator decision of 2026-11-12: a
target with a pinned target's architecture and shapes now qualifies; see
"D6 policy as implemented".)

The Qwen SlotAware path captures cold-prefill target layers, keeps one persistent
draft ring per physical slot, drafts all provisioned slots in one graph, verifies
one to eight rows per live lane in the fixed width-32 target route, applies the
existing spec-off semantic decision function, commits only accepted target and
draft prefixes, and drains accepted output through the existing ordered SSE
settlement path.

Each resident draft ring is bound to the exact committed target-token prefix,
not just the target cursor length. Slot reassignment or a different same-length
conversation therefore clears the draft ring. Before the ring's first
sliding-window wrap, an exact earlier prefix may safely rewind the logical cursor
because all required rows remain resident; after the first wrap, an earlier
restore fails closed to the established baseline route. Qwen3.5 and any target
without a pinned identity match are likewise unavailable. Gemma has a schema
and target-weight adapter but no serving transaction yet; DeepSeek has no
compatible trained checkpoint and remains explicitly unavailable.

The shared runtime and capture descriptor are family-neutral: the runtime has
no production dependency on Qwen types, and `DFlash2TargetFamily` constructs
the bounded capture geometry consumed by an explicit family verifier. Qwen is
the first consumer; the same contract is available to the existing Gemma
adapter without routing Gemma through Qwen state. The opt-in ADR-065 stage
profiler also owns a complete DFlash2 cycle and labels the draft command
separately, so draft, target categories and host gap can be measured without
changing the default graph.

This is implementation evidence, not full gate closure. The candidate still
performs the cold-prefill capture conversion through host memory and does not
overlap grammar simulation with target verification. Real-target width-32
parity, exact-prefix continuation, inactive-lane handling, EOS suppression,
logprobs canonical decisions, cancellation recovery and the fixed performance
campaign now have evidence below. Grammar rejection, sampled-policy parity,
post-wrap rollback and the complete multi-turn serving matrix remain open.
D5's same-artifact peer threshold and D6 promotion also remain open.

One exact-target runtime smoke used an hf2q Q5_K_M conversion of
`Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`
(source receipt `851811f4…`), the pinned 1,052,513,456-byte Qwen3.6
DFlash2 checkpoint (`d1f16c0…`), one SlotAware lane, temperature 0 and cold
prompts. Provisioning succeeded. A 16-token unary request completed two DFlash2
cycles whose retained target-row counts were 8 and 7, at 123.55 output tok/s.
An SSE request emitted a valid forced `echo` tool call with incremental
`{"text":"hi"}` arguments and `finish_reason:"tool_calls"`; a following
tool-result request completed normally. The tool-call run exercised retained
counts from 1 through 8 and forced-thinking closure. The mismatched
`qwen36-abliterix-t63-APEX` startup refusal above was run against the same draft
artifact. These are end-to-end smoke observations, not D2-D6 qualification:
they are single-lane, unmatched to spec-off, and not the Q4 workload.

The first four-lane Qwen3.6 run exposed a separate target-shape boundary: its
native-storage batched MoE layer admitted only the ordinary joint-decode width
of eight rows, while DFlash2 always supplies 32 target rows. The route now
admits 32. Rows one through eight retain their bitwise single-row invariant;
the F32 router and shared-gate logits at width 32 use dense MM and are instead
qualified against the whole-layer F64 reference. Q4_K and Q6_K model-free
tests matched expert IDs outside declared near ties and stayed within the
`8 * (hidden + moe_i) * 2^-24 * S` error envelope. The exact checkpoint then
completed a four-lane repair-turn Q5 calibration in 35.619491 seconds with all
clients successful and cleanup proven. This closes the runtime shape defect;
it is not a matched D5 campaign result.

An exact-target Qwen3.8 Q4_K_M follow-up smoke on 2026-11-10 exercised that
prefix binding. The cold unary turn used 64 prompt tokens, produced 36 completion
tokens through DFlash2, and returned `ALPHA`. The next rendered prompt had 120
tokens; target admission reused exactly 100 cached tokens, prefilled only
positions 100 through 119, then ran eight DFlash2 verify windows with retained
counts `2,1,3,3,2,8,8,2` and returned `BETA`. A four-lane regression discovered
and fixed the native selector boundary for inactive physical lanes: inactive
lanes now carry the smallest valid one-row bound while remaining excluded from
verification and settlement. These observations prove exact-prefix warm
continuation and inactive-lane safety for the candidate, not the D3 cancellation,
tool/SSE and long-context rollback matrix or D6 qualification.

The first D4 stage trace found that acceptance was not the remaining wall-time
problem. On the exact Qwen3.8 target, the default mlx-native tensor-v2 route
spent a median 215.494 ms in the fixed 32-row target verify and 243.857 ms in
the physical transaction. Tensor-v2 uses a 128-row output tile, so this shape
left three quarters of its row tile empty. The same four-lane, 256-token smoke
with tensor-v1's matching 32-row tile reduced those medians to 123.537 ms and
149.179 ms and produced byte-identical assistant payloads. Disabling tensor MM
entirely was neutral to slower, which isolates the result to the tensor-v1
versus tensor-v2 shape route rather than tensor execution generally.

The scheduler retains this observer behind the existing DFlash progress DEBUG
target. Enabled traces emit draft, target verify, host planning, target commit,
draft-context commit and total wall time for each physical transaction; normal
INFO serving does not create `Instant` samples or emit a record.

mlx-native therefore routes exactly 32-row dense GGML MM through tensor-v1 and
retains every other shape's previous route. Its production route and resolved
trace tests cover the decision, while the existing Q5 tensor numerical gate
covers row/position invariance. A local exact-source integration smoke completed
the full four-agent coding protocol in 74.388081 seconds with all four clients
successful and cleanup proven, down from 103.141573 seconds before the route
change and below the accepted joint-MTP campaign's 87.704486–94.153340-second
range. The smoke is performance evidence, not D5 closure: it is one run, used a
local native checkout rather than the release-candidate published pin, and does
not supply a confidence interval.

The fixed six-block ABBA campaign on the published `mlx-native` 0.21.0 pin
(`8baa89fb…` package SHA256) compared the prior joint-MTP route with DFlash2.
All measured runs were quality eligible, thermally valid `CalibrationPass`
runs. DFlash2's makespan ratio was `0.894442` with 95% CI
`[0.882325, 0.906726]`, a 10.556% improvement with 95% CI
`[9.327%, 11.768%]`; TTFT and SSE-gap upper bounds also showed no regression.
The retained campaign artifact
`adr067-qwen38-dflash2-148e0e3f-v1-run/analysis.json` has SHA256
`a3fb5214…`. Trace reduction measured 3,710 accepted-plus-bonus
outputs over 699 lane-cycles, or `5.307582` outputs per lane-cycle, clearing
the preregistered absolute `5.304` mechanism floor.

The cancellation rerun taught the harness the DFlash2 physical-width record
instead of relabelling it as the older joint route. It observed widths one
through four during the cancellation pass, returned byte-exact survivor
payloads, kept the victim payload as a strict prefix of its control, preserved
server readiness, and proved owned cleanup. The retained cancellation artifact
`qwen38-cancel-148e0e3f-v2/receipt.json` has SHA256 `ac2541d2…`; the client
summary SHA256 is `2b3f4187…`. Scheduler
regressions additionally prove that a terminal EOS proposal is neither
streamed nor retained and that logprobs requests keep full-logit canonical
decisions.

This closes neither D5 nor D6. The historical valid same-artifact peer median
is about 79.7 seconds; the fixed DFlash2 campaign's median is about 80.47
seconds. Those unmatched medians provide no confidence interval for the amended
candidate/peer upper-ratio bound of 1.00. DFlash2 therefore remains explicit
opt-in and the global `HF2Q_DFLASH2=0` control continues to restore the prior
route.

#### Splash dense-kernel differential

The pinned Splash source explains why matching only the 32-row tile did not
close the remaining target cost. Splash repacks native GGUF blocks at load into
its `MDGG0001` code, optional high-bit and predecoded metadata planes. Its
32-row kernel then reads coalesced code/meta vectors, double-buffers half-weight
stages, overlaps the next payload and metadata loads with the current
`matmul2d`, and keeps activations in BF16. hf2q's production tensor-v1 kernel
still streams native GGUF block layouts and repeatedly decodes irregular
Q4_K/Q6_K metadata from those blocks. Splash's own source notes about an 8%
cost for dynamic Q4_K scale-byte indexing support this attribution.

Bounded experiments falsified the available shortcuts. Generic tensor-v2
64x32 routing raised target-verify medians to roughly 161 ms; restricting that
route to Q6_K still measured roughly 155 ms. A dedicated load-once Q4_K/Q6_K
code/meta repack passed model-free numerical parity and every real-model Q5
quality and cleanup gate, but none of its measured schedules beat the promoted
route end to end:

| candidate | four-task makespan (s) | target-verify median (ms) | transaction median (ms) |
|---|---:|---:|---:|
| golden-ratio K walk | 119.305 | 233.553 | 261.045 |
| sequential prepared planes | 83.802 | 152.115 | 178.083 |
| selected projection split-K | 82.591 | 152.900 | 178.249 |
| all transformer projections split-K | 82.747 | 152.591 | 178.367 |
| staged half activations | 83.167 | 150.212 | 175.863 |
| pipelined staged half activations | 84.316 | 150.355 | 175.719 |

The prepared route also retains a second model-sized weight representation.
Its small kernel-stage gain does not justify that memory cost without an
end-to-end win. A separate smallest-covering 4/8/16/32 verifier-width spike
regressed to 87.782 seconds because the narrower stock kernels cost more than
the fixed width-32 pass; the fixed dense width remains intentional. All seven
spikes are rejected, not production fallbacks. The remaining credible dense
kernel path requires a complete prepared-weight BF16/fusion execution model
that wins end to end, not another raw-block row tile, K-order, split-K, or
partial activation-staging change.

#### Splash MoE verification differential

The exact Qwen3.6 DFlash2 campaign falsified the assumption that the dense
width-32 route transfers unchanged to MoE. Before an external interruption
invalidated the campaign, 18 complete four-agent blocks measured the candidate
at 37.542 seconds median: 33.66% faster than v0.1.29, but 5.55% slower than the
valid peer and 2.316 times Splash's 16.420-second median. All 18 candidate
quality outcomes passed. The point estimate is too far from either comparator
to justify rerunning the unchanged route.

Pinned Splash source identifies the missing mechanism. MoE uses
`--verify-rows auto`, while dense targets verify every row. For each draft
position it maps selector confidence to acceptance probability with
`logit(p) = 0.7 * logit(confidence) + intercept[position]`; the seven prior
intercepts are `0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45`. It ranks each
lane's cumulative survival probabilities and chooses the prefix count that
maximizes expected accepted-plus-bonus tokens divided by cycle cost. The
initial cost shapes at one, two, four and eight rows per lane are
`[0.60, 0.69, 0.85, 1]`, `[0.50, 0.58, 0.74, 1]`,
`[0.45, 0.52, 0.70, 1]`, and `[0.40, 0.47, 0.67, 1]` for one through four
lanes respectively. Splash later refits acceptance from the latest 4,096
labels every 64 cycles and cycle cost with exponential decay `0.97`; every
32nd cycle verifies all rows for exploration.

The target-side saving is not a host-only truncation. Splash's MoE router marks
rows beyond each lane's selected prefix invalid, so they do not count toward
expert groups, trigger streamed expert loads, or execute expert tiles. hf2q's
prior Qwen3.6 route selected and executed all 32 rows. The bounded successor
ports Splash's online 4,096-label acceptance fit, 64-cycle refit, decayed cost
fit and 32-cycle exploration schedule for identity-matched Qwen MoE targets.
It routes only selected rows and uses the smallest covering 4/8/16/32 output
tile; dense Qwen3.8 keeps its qualified fixed width-32 route.

The successive four-agent calibration smokes all passed task quality and
cleanup. Online budgeting reduced makespan from 36.129 to 35.203 seconds;
excluding padding rows from expert work reduced it to 30.947 seconds. At that
point the four-lane median selected 18 of 32 rows and accepted 15, while target
verification still cost 46.488 ms of a 58.538 ms transaction. Generic
`mm_id` grouping does not close that residual gap: the allocation-owning
32-row route regressed to 36.162 seconds, and pooling one compact gate/up
schedule and reusing it for down still measured 33.781 seconds. A separate
five-route live-expert scalar tile reused each native Q4_K/Q6_K weight row
across grouped routes and passed bitwise operator parity plus full-layer
quality, but regressed the four-agent smoke to 44.440 seconds. Its diagnostic
four-lane step spent 94.366 ms in target verification at 19 selected rows,
versus 46.488 ms for the exact matvec route at 18. These grouped routes are
rejected on end-to-end performance.

Splash's remaining advantage is therefore kernel-specific, not an omitted
selector constant or grouping alone. A complete device-policy trace corrects
an earlier inference from the presence of its Table16 register kernel.
`ExecutionPlans::moeConfig` calls `moeGgufTile`; `MoE.hpp` selects the register
tile only for Apple9 GPUs with more than ten cores, while the release M5 Max is
Apple10 and therefore always selects `MoeGgufTile::Staged`. The valid Splash
comparator consequently does **not** run the Table16/register expert path. It
runs eight-row staged tiles over BF16 grouped activations and outputs, launches
only the indirect grid of live tiles, switches one- or two-row expert tiles to
a dedicated column-dot kernel, and executes gate and up together with one
simdgroup per projection and a fused BF16 SiLU epilogue. Shared-expert slices
are inserted into the same grouped representation.

The retained status from valid Splash run `n4-070-b17-splash-a0` confirms that
policy on the comparison host: `apple_gpu_family=10`, 40 GPU cores, 21.213 GB
of target weights and 480 MB of draft weights. Across 606 mixed-width decode
batches (183 B1, 95 B2, 145 B3 and 183 B4), target forward consumed
10,866.091 ms, or 17.93 ms per batch, while complete model GPU decode consumed
12,721.392 ms, or 20.99 ms per batch. Its final B1 sample reported 13.156 ms
target forward and 15.276 ms complete GPU decode. Against hf2q's steady B4
50.4-55.9 ms target verify, this localizes the remaining deficit inside the
target kernels and activation graph rather than the host transaction tail.

hf2q's generic grouped path instead uses 32-row F32 tiles, while the scalar
live-expert spike loses output-column occupancy and has too little route reuse
at the observed 10–19 selected rows. Both pay more scheduling, staging, or
weight-decode cost than the Q4_K route matvecs. The M5-class successor must
therefore implement the staged M8 design, including the one/two-row column
tile and one-pass gate/up, over every admitted GGUF expert format. A Table16
register copy targets Splash's Apple9 policy and cannot establish M5 parity.

The source-bound register route is more specific than an eight-row threshold.
`MoE.cpp:225-283` first applies the GPU verify-row counts while grouping routes,
writes a compact live-tile count plus indirect gate/up and down grids, and
launches no inactive expert tile. `moe_gather_table16` then transposes each live
eight-row tile from BF16 into a 64-input-span Table16 image and accumulates its
row sums; one 256-thread group handles one tile and 256 input columns.
`moe_expert_gguf_sg_*` launches one threadgroup per 64 output columns and live
expert tile, selects the expert's prepared GGUF planes at runtime, and performs
no K split. Gate and up are separate register passes with the up pass applying
`SiLU(gate) * up`; the intermediate remains BF16, is prepared into another
Table16 tile, and feeds down. Splash's own source reports the Table16 gather at
8 microseconds for one lane and 14 microseconds for four lanes on an M3 Max.

Therefore the next hf2q MoE candidate must be one coherent Apple10 route:
BF16 target/expert activations, verify-row-aware compact tile grouping,
prepared Q4_K/Q6_K planes, staged M8 with the dedicated one/two-row column
path, one-pass gate/up with a BF16 intermediate, indirect live-tile grids,
shared-expert slices in the grouped representation, and grouped reduction.
Testing one item while retaining F32 pair matrices or fixed-grid matvec
launches does not reproduce Splash's mechanism and cannot support promotion.

The converter-style prepared-plane M8 spike then passed Q4_K/Q6_K operator
parity and reduced the best exact four-agent smoke to 28.641453 seconds. A
dedicated one/two-route register column tile and a one-time F32-to-F16 input
gather also passed their kernel gates, but the gathered variant completed the
same smoke in 32.553126 seconds. Both remain rejected: even the best prepared
result is 1.744x Splash's 16.420344-second median, and the gathered route
regresses that prepared baseline.

A complete audit of pinned Splash found two orthogonal graph differences.
`DraftCheckpoint.cpp:28-41,58-93` repacks every draft projection from source
BF16 into `GGUF_FMT_AF4G64` planes; norms, convolution kernels and selector
codebooks alone remain BF16. It also concatenates Q/K/V rows into one projection
and executes gate/up through a fused epilogue. hf2q currently uploads every
draft projection as BF16, retaining a 3.85 GB drafter and issuing separate Q,
K, V, gate and up projections. AF4G64's nominal four-bit values use about
one-quarter of the projection payload before group metadata, making projection
bandwidth and dispatch shape the primary draft differential rather than only an
activation-precision difference.

The exact Qwen3.6 checkpoint is 1,052,513,456 bytes: 797,966,336 bytes
are projection tensors and 254,536,704 bytes are retained BF16 tensors.
At 36 bytes per 64 values, an equivalent packed projection payload is
224,428,032 bytes and the complete tensor payload is about 479 MB.

`Runtime.cpp:3037-3092` also encodes the draft forward, GPU proposal and
verify-row selection, target input embedding, target verify, acceptance, target
commit and draft commit into one Metal command graph and submits it once. hf2q
currently waits for the draft, reads proposals and row counts on the host,
submits target verify, plans acceptance on the host, commits target state, then
submits the draft-context commit. Splash keeps target activations and MoE
outputs in BF16; hf2q's Qwen target graph is predominantly F32. Representative
hf2q prepared-plane traces spend about 9.5 ms in the draft and 20.8-24.8 ms in
one-lane target verify, or 39-49 ms in four-lane target verify. The next spike
must therefore quantify the draft projection categories, then replace the BF16
draft projection path with a source-bound quantized fused-QKV/fused-gate-up
path before attempting the larger GPU-resident cycle transaction. Another
expert threshold, scalar tile or activation-gather change cannot qualify from
the measured evidence.

The opt-in stage profiler then passed the realistic Qwen3.6 four-agent Q5
smoke with cleanup proven
(`qwen36-dflash2-stage-profile-v1/receipt.json`). Across complete decode
rounds, the median draft GPU command was stable at 9.318-9.359 ms for widths
one through four. Median classified target GPU time was about 19.0 ms at width
one and 42.8 ms at width four; width-four MoE alone was 22.769 ms. The
profile-only category waits inflate median wall time to 67.845 ms at width one
and 97.876 ms at width four, so those wall values are attribution evidence,
not performance claims. The result confirms both independent work fronts:
quantized/fused draft projections must attack the fixed 9.3 ms command, while
prepared target kernels, BF16 target activations and the single transaction
must close the wider target and synchronization gap.

The first isolated packed-projection spike used hf2q's owned Q4_K quantizer
behind `HF2Q_DFLASH2_DRAFT_Q4_K=1`. Startup confirmed Q4_K activation, all
focused tests and six paired realistic Q5 runs passed, and cleanup was proven.
At width four, Q4_K reduced the profiled draft command from 9.318 ms to
8.118 ms (12.88%). Three Williams-ordered paired blocks produced BF16 elapsed
times of 38.285, 41.871 and 40.584 seconds and Q4_K times of 41.626, 39.659
and 34.414 seconds. The medians were 40.584 and 39.659 seconds respectively;
the paired ratio estimate was 0.95583 with a non-qualifying 95% CI of
[0.70149, 1.30238]. Generic Q4_K dispatch therefore remains an isolated failed
experiment: payload compression helps the draft command, but does not approach
Splash's valid 16.420344-second median. The next spike must combine packed
prepared planes with fused Q/K/V and gate/up execution; payload density alone
is insufficient.

The follow-on fused Q4_K spike concatenated Q/K/V and gate/up projection rows,
used one quantized matmul for each fused projection, split Q/K/V on GPU, and
viewed the K/V suffix without duplicating weight storage for context commits.
Twenty-five focused DFlash2 tests and the realistic profiled Q5 run passed
with cleanup proven. The width-four draft-command median was 8.133 ms versus
8.118 ms for the unfused Q4_K route: fusion through the generic GGML kernel
and follow-up split did not improve the draft command. It remains isolated.
The next draft projection route must use the source-bound AF4G64-style
group-64 packed affine format and fuse gate/up activation into the linear
epilogue rather than adding a split pass.

The source-bound AF4G64 successor then quantized each 64-weight group to
64 four-bit affine codes plus BF16 scale and bias, packed the codes into
projection-major planes, and reused one preallocated BF16 Table16/F32-sum
scratch pair across every draft projection. Q/K/V and gate/up weights remain
concatenated at upload, so each group is read once per fused projection. The
exact Qwen3.6 Q5 calibration passed and cleanup was proven first in
`qwen36-dflash2-af4g64-fused-profile-v10`, then from the published
`mlx-native 0.21.1` registry package in
`qwen36-dflash2-af4g64-registry-final-v1`.

| Isolated width-four draft route | Median draft GPU command (ms) | Decision |
|---|---:|---|
| BF16 production baseline | 9.318 | baseline |
| generic Q4_K | 8.118 | rejected; generic kernel |
| row-major AF4G64 | 15.868 | rejected |
| prepared AF4G64, direct BF16 dequantization | 10.984 | rejected |
| prepared AF4G64, exact F32 affine epilogue | 9.387 | rejected |
| prepared AF4G64, activation Table16 | 7.878 | accepted primitive |

The accepted primitive is 15.45% faster than the BF16 draft command and 2.96%
faster than the generic Q4_K spike. Widths one through three measured
7.908, 7.904 and 7.903 ms, so the gain is not a width-four-only schedule
artifact. This authorizes the exact packed projection primitive and fused
weight layout behind `HF2Q_DFLASH2_DRAFT_AF4_G64=1`; it does not authorize
release promotion. The profile waits distort topology, and the remaining
target, host-readback and multi-command costs still require the single
GPU-resident transaction and the binding end-to-end Splash gate.

The next isolated spike fused the AF4G64 gate and up dot products in one
register kernel and applied `SiLU(gate) * up` before the output write. It
removed the separate gate/up output, deinterleave and SiLU dispatches. The
partial-row Metal reference test passed, and the exact Qwen3.6 Q5 smoke passed
with cleanup proven in `qwen36-dflash2-af4g64-silu-profile-v11`. The complete
draft command nevertheless regressed at every observed width:

| Width | Prepared AF4G64 (ms) | Fused SiLU epilogue (ms) | Change |
|---:|---:|---:|---:|
| 1 | 7.908 | 7.957 | +0.62% |
| 2 | 7.904 | 7.933 | +0.37% |
| 3 | 7.903 | 8.006 | +1.31% |
| 4 | 7.878 | 7.920 | +0.53% |

The dual accumulator path's removed dispatches do not repay its larger
register working set. The fused epilogue remains on isolated experiment
branches `mlx-native/perf/af4g64-silu-epilogue@978699b` and
`hf2q/perf/dflash2-af4-silu@c72e2dd3`; it must not land. This falsifies that
specific fusion shape, not the source-backed AF4G64 store or the required
single GPU-resident decode transaction.

The prepared target-MoE spike repacked Q4_K/Q6_K expert rows into Metal
tensor planes and gathered target activations as BF16. Its production-shape
microbenchmark reduced one Qwen3.6 gate projection from 1.086 ms to 0.875 ms
(19.4%). The BF16 route passed a realistic four-agent Q5 run with cleanup
proven, but changed numerical outputs relative to the F32 authority by
0.17-0.22% relative RMSE in the focused kernel cases. Its standalone Q5
makespan was 37.231 seconds and the combined packed-draft run was 40.826
seconds; these unpaired, output-dependent samples do not establish an
end-to-end win. The route therefore remains isolated until a matched quality
and performance campaign can qualify it.

The Table16/register-matrix follow-up did not rescue that target route. The
spike consumed the same prepared Q4_K/Q6_K planes directly, gathered eight
BF16 rows into Splash's transposed 64-input table, retained its F32 seed/sum
correction, and passed shared/slotted F32, BF16 gate/up epilogue and
BF16-input down-projection parity at the existing 0.3% kernel envelope. A
first implementation exposed an hf2q-specific compact-schedule hazard:
per-expert route slices have one guard entry, not eight-row padding, so padded
Table16 writes overlapped the next expert. Reserving seven additional rows per
expert removed that race. Narrow per-lane payload loads then reduced the
register result from 1.207 to 1.187 ms per projection, but the matched
100-dispatch Qwen3.6 gate probe still measured 1.134 ms for the scalar route,
0.862 ms for the prepared Metal-tensor route and 1.187 ms for the register
route. The register route is 37.7% slower than the already insufficient
prepared route and remains a rejected experiment. Source policy now shows
that this result measures Splash's Apple9 arithmetic, not the Apple10 staged
M8 path used by the valid M5 Max comparator; it cannot reject or validate the
required staged successor.

The next staged-M8 spike implemented the Apple10 arithmetic shape: prepared
Q4_K/Q6_K expert planes, BF16 gathered activations and intermediate, a
dedicated one/two-row column path, and one gate/up dispatch with one simdgroup
per projection over the same 32 output columns. Gate remains in threadgroup
memory and the up side writes `BF16(up) * SiLU(BF16(gate))`; down consumes that
BF16 intermediate and returns F32. Focused Q4_K/Q6_K gate/up/down parity passed.

The first exact downstream run exposed a compact-map storage boundary that the
focused kernel case had not enforced: map0c inserts one guard entry before each
expert route list, so a gathered buffer sized for routed rows alone can be
overrun and stall the command buffer. mlx-native 0.21.5 now rejects that
undersized extent, and hf2q reserves `routed_rows + experts`; the boundary
regression passed. The registry-pinned release build then completed the exact
four-agent repair-turn Q5 with tool calls and cleanup proven.

The original microbenchmark conclusion was invalid because it used
`K=5120`, `N=1536`, 128 experts rather than Qwen3.6's actual `K=2048`,
`N=512`, 256 experts. The corrected 18-row probe measured 0.184 ms for one
prepared projection and 0.255 ms for the one-pass gate/up pair. More
importantly, matched exact-model traces compare equal selected-row bins rather
than output-dependent request makespans. Both the staged-M8 and native-control
four-agent Q5 runs passed all task/tool/cleanup gates (605 and 479
transactions). Across four-lane bins from 13 through 24 selected rows,
staged-M8 reduced target verification by 1.27% to 9.42%; at exactly 32 rows it
reduced 59.464 ms to 52.095 ms (12.39%), and the matched-bin weighted median
improvement was 4.52%. The route therefore lands opt-in as
`HF2Q_QWEN_MOE_STAGED_M8=1`, requiring `HF2Q_QWEN_MOE_BATCHED=1`. It is not a
Splash-parity promotion: its four-lane target median was 43.796 ms versus
Splash's 17.93 ms target-forward average. Shared-expert grouping, indirect
live grids and broader BF16 target activation flow remain open.

The original fused-BF16 arithmetic did not pass the repeatability requirement.
A later exact repair-turn Q5 run produced the same uncompilable Rust repair in
two lanes while its same-binary native-MoE control passed all four lanes. This
is a task-quality failure, regardless of the fused route's 44.212 ms
four-lane target median, so the BF16 intermediate is no longer offered by the
switch.

The quality successor keeps the prepared expert planes and compact grouping,
but runs gate and up as separate F32-output staged-M8 projections over one
reused schedule and gathered input. The established F32 SwiGLU feeds the
prepared down projection as F32. Two initial consecutive exact four-agent Q5
runs passed task, tool and cleanup gates.

A separately predeclared six-block balanced campaign at
`$E/qwen36-m8-d5-v1` compared that successor with same-binary native batched
MoE. All 12 exact task/tool/cleanup gates passed. The staged route reduced the
four-lane target-verification median by 9.740% (95% log-ratio CI
8.407–11.053%, half-width 1.323 points): its per-run median was 44.589 ms
versus 49.387 ms for the control. Client makespan reduced by 6.904%, but the
95% CI was -2.976–15.836% (half-width 9.406 points), so the end-to-end effect
is inconclusive. This replaces the fused-BF16 implementation behind
`HF2Q_QWEN_MOE_STAGED_M8=1`; the route remains opt-in pending an end-to-end
precision pass and still does not establish Splash parity.

Preparing the shared expert as a separate one-expert staged-M8 stack did not
reproduce Splash's grouping. The first exact run exposed another untracked
buffer hazard: replacing the compact schedule and gathered rows before the
prior projection finished corrupted all four responses. Explicit barriers
restored task and tool quality, and the exact repair-turn Q5 passed with
cleanup proven, but every populated matched four-lane bin regressed. Target
verification rose by 7.19% to 13.37% across 12 to 32 selected rows; its
run-wide median rose from 45.283 ms to 49.774 ms (9.92%). The experiment is
preserved at `6b5e3793` and rejected. Splash's shared-expert slices must join
the routed experts in one compact schedule and one gate/up/down dispatch
sequence; adding a second schedule, gathered-buffer overwrite and barriers is
the opposite mechanism.

The exact single-schedule variant also failed the performance gate. mlx-native
experiment `4d2dca2` prepared 256 routed experts plus the shared expert into
one 257-expert allocation without a model-sized concatenation, appended the
shared expert as route nine, built one compact schedule, ran one gate/up/down
dispatch sequence and reduced the final route with the sigmoid shared gate.
hf2q experiment `143feea4` applied the DFlash2 route mask to all nine routes
and removed the separate shared projections. Q4_K/Q6_K prepared-tail parity,
SIMD routing, shared-last reduction, the native all-target check and seven
focused hf2q MoE tests passed. Two exact four-agent repair-turn Q5 runs also
passed task, tool and cleanup gates.

Repeated timing did not justify promotion. The two registry/clean staged-M8
baselines contained 617 four-lane samples with a 45.295 ms aggregate target
median; the two combined-route runs contained 600 samples with a 45.3565 ms
median, a 0.136% regression. The median of run medians was effectively flat
(45.2895 versus 45.3005 ms), while equal-selected-row bins had a 1.41%
weighted-median regression. Rows 21, 22, 25, 26, 27 and 32 regressed by 2.62%
to 6.97%. The implementation is preserved on its experiment branches and
rejected. Splash's shared representation is not an isolated speed lever:
adding a ninth grouped route trades away the efficient dense shared projection
at the wider verifier rows. The next target slice must compact the fixed target
graph to live rows and reduce target weight/dequantization traffic rather than
rearranging the same shared-expert work.

An unprofiled four-request diagnostic on the exact Qwen3.6 Q4_K_M target then
separated transaction boundaries from target work without the stage profiler's
forced waits. With GPU proposal handoff and AF4G64 draft projections active,
steady four-lane cycles spent about 50.4-55.9 ms in target verification and
52.9-58.5 ms in the whole transaction. Host planning was normally below
0.5 ms, target settlement about 0.9-1.7 ms and draft-context settlement about
0.7-0.8 ms. The four requests each generated 160 tokens at 58.1-64.2 output
tokens/s and server cleanup was proven. The trace also showed the adaptive MoE
mask often accepting only 8-18 rows across four lanes while the fixed target
tile still admitted 32 rows; only routed-expert work was masked. A single
command graph remains required for Splash parity and state correctness, but
these measurements reject command boundaries as the dominant remaining
throughput gap. The next target slice must reduce the fixed target graph's
weight/dequantization traffic and execute only the selected MoE rows; deleting
the roughly 2-3 ms host/settlement tail alone cannot close a roughly 2x gap.

A separate checked asynchronous target-context submission spike retained the
in-flight command encoder, observed its error at the next same-queue draft
wait, and drained it before capture overwrite, lane clear, or runtime drop.
Twenty-five focused DFlash2 tests and realistic profiled and unprofiled Q5
runs passed with cleanup proven. At width four, however, the topology-distorted
profile median moved only from 97.876 ms to 97.460 ms, while median host gap
moved from 44.518 ms to 43.784 ms. A 0.43% profile-wall change is not a
qualifying end-to-end result. Asynchronous tail submission alone is rejected;
the accepted design still requires GPU-resident proposals, target verify,
acceptance, and both commits in one transaction.

The next accepted dense-target slice keeps draft proposals on the GPU through
target verification. A bounded Metal composition kernel packs each logical
lane's anchor plus selected proposal prefix into the fixed target tile. The
draft command submits without a host wait; the target's native embedding
gather consumes that tile on the same queue. Proposal readback occurs only
after the target output-head completion boundary, when host acceptance still
needs the token identities.

The matched Qwen3.8 Q5 candidate and control used the same binary, target,
checkpoint, prompts and AF4G64 draft projections. Both passed calibration and
cleanup. Every agent produced the same workflow and final-source hashes in
both arms.

| Qwen3.8 arm | Four-agent makespan (s) | Relative to staged control |
|---|---:|---:|
| Staged host proposal handoff | 86.397 | baseline |
| GPU proposal-to-target handoff | 80.758 | 6.53% faster |

This promotes the GPU handoff for the identity-matched dense Qwen3.8
descriptor, with `HF2Q_DFLASH2_GPU_TARGET_TOKENS=0` as its escape hatch. The
first Qwen3.6 MoE attempt failed task quality because host acceptance consumed
all eight target output rows after the GPU selected a shorter verify prefix.
Those unverified outputs could therefore be accepted and committed. The
corrected transaction reads the selected counts only after target completion,
excludes unselected outputs from acceptance, masks their routed expert IDs on
the GPU, and makes quantized expert kernels return zero before any sentinel
expert weight address or read.

The corrected exact Qwen3.6 candidate passed the four-agent Q5 workflow,
tool-result continuation, unary/SSE response and cleanup gates in
`qwen36-dflash2-moe-adaptive-v3`; its own-output makespan was 31.981 seconds
versus 48.534 seconds for the published-package staged control
`qwen36-dflash2-af4g64-registry-final-v1`, 34.11% lower. An override-free
default-on repetition then failed task quality in
`qwen36-dflash2-moe-default-v4`: one lane produced the same Rust
type-inference error in both its initial edit and allowed repair. The next
override-free repetition passed in `qwen36-dflash2-moe-default-v5` at 31.192
seconds, 35.73% below the staged control. A clean release build against the
published `mlx-native 0.21.3` crate then exercised the same GPU route, valid
unary/SSE responses, tool-result continuation and prefix reuse in
`qwen36-dflash2-moe-published-v6`, but one lane wrote an unterminated Rust
string and did not issue the required repair call, so that run correctly
failed task quality. All runs proved cleanup and resolved the GPU route at
startup.

The mixed quality result does not qualify a default. Qwen3.6 MoE therefore
supports the complete GPU adaptive-row route when
`HF2Q_DFLASH2_GPU_TARGET_TOKENS=1`, but remains staged by default pending a
repeatable fixed-count quality campaign. Gemma remains fail-closed because no
exact release-gate target/checkpoint pair exists. This is not the complete
Splash transaction: target output readback, host acceptance, target-state
settlement and draft-context commit remain separate boundaries. The previously
quoted 4.918 ratio compared Qwen3.8 with Qwen3.6's Splash median and is withdrawn.

The next bounded transaction slice adds mlx-native's
`dispatch_dflash2_accept_greedy` operation. On transform-free exact-greedy
requests, the target output-head command now computes retained and accepted
counts in physical-lane order and consumes the retained counts in the draft
context commit before submission. The canonical host acceptance walk still
runs after readback and must match both GPU counts exactly; any mismatch or
post-submit error fail-stops the worker. Requests with repetition penalty,
forced-thinking transitions, grammar state, lazy tool triggers, stop strings,
sampling, or logprobs remain on the prior host acceptance path because their
semantic token is not the raw target argmax.

The realistic Qwen3.8 four-agent coding calibration passed with unary/SSE tool
semantics, tool-result continuation, prefix reuse and cleanup proven; its
production transforms correctly kept `gpu_acceptance=false`. A separate
one-slot Qwen3.8 smoke disabled thinking and repetition transforms, returned the
requested token sequence, and recorded `gpu_acceptance=true`, four retained
rows, `draft_commit_us=0`, and a 1.398 ms target replay. The one-slot smoke also
found and fixed the GPU target tile allocation: target verification now always
receives the canonical 32-row tile, with unused logical lanes padded by lane
zero's anchor, even when the worker owns fewer than four physical lanes.

This still is not Splash's single transaction. Partial acceptance requires
`commit_joint_verify` to replay retained Qwen DeltaNet rows in a later target
command. Splash instead leaves verified GDN rows pending on a tape and folds
the retained prefix into the next graph. The next state-transaction slice must
therefore add deferred retained-row state or an equivalent dynamic GPU commit;
moving another host wait without removing that replay cannot close the gap.

Staged M8 and the GPU adaptive-row route were each qualified alone and had
never run together. With both switches set, every Qwen3.6 request failed
(`embedding_q4_k: token_ids[9]=4294967295 exceeds vocabulary`): on the
mlx-native 0.21.6 build with and without native profiling, and on the
`main`-equivalent 0.21.5 build with profiling on. The 0.21.6 build without
`HF2Q_QWEN_MOE_STAGED_M8` passed. The route mask rewrites
every slot of an unselected row to the id `num_experts`. The staged-M8 compact
schedule treated that as invalid metadata and poisoned itself, and the grouped
M8 kernels write nothing under a poisoned schedule, so one masked row dropped
the expert projections of every row in the step. Nothing refused the
combination at startup; the request failed only because a later host-side
token check happened to see the damage.

mlx-native 0.21.7 adds a masked compact schedule: an id equal to
`num_experts` is neither validated, counted nor listed, any other
out-of-range or duplicate id still poisons, masked rows are left unwritten,
and a masked weighted reduce skips zero-weight routes without reading them.
hf2q uses those entry points whenever a route mask is present. This is
Splash's rule that rows beyond a lane's selected prefix do not count toward
expert groups. On a development build, the exact Qwen3.6 four-agent Q5 with
both switches passed 10 of 11 runs; in the other, one agent did not follow
the six-step workflow, which is a task failure, not a server error. The same
build without staged M8 passed 11 of 11, and the build pinned to the
published mlx-native 0.21.7 passed 2 of 2 with both switches. These counts do
not separate the two rates, so the combination is not quality-qualified. A
fixed six-block
same-binary campaign (`$E/qwen36-staged-masked-dev-d5-v1`) against the masked
native expert matvecs passed all 12 gates but did not resolve a makespan
effect: 5.77% lower with a 95% CI from 8.88% higher to 18.45% lower. Qwen3.6
output length varied from 4,703 to 6,409 completion tokens between runs of
one arm, so six blocks of makespan cannot resolve an effect of this size.
Unprofiled stage traces, one run per arm, put the median four-lane cycle at
56.47 ms with staged M8 and 61.06 ms without, at 4.18 and 3.98 accepted
tokens per lane-cycle. The combination no longer fails; it carries no
performance or quality claim.

The same traces count what one cycle encodes. A four-lane Qwen3.6 cycle
issues 8,107 compute dispatches and 2,763 memory barriers in 25 command
buffers with three host waits; a one-lane cycle issues 3,204 dispatches. A
four-lane Qwen3.8 cycle issues 12,147 dispatches and 4,331 barriers; a
one-lane cycle 5,652. Dispatch count grows by about 1,630 (Qwen3.6) and
2,170 (Qwen3.8) per added lane, so a large share of the graph is encoded per
lane or per row rather than once over the 32-row tile.

The count itself is not what the cycle costs. A throwaway mlx-native probe
encoded the same numbers of 64-element dispatches with a barrier after every
third, split over the same number of command buffers, and waited for all of
them: 8,107 dispatches took 3.90 ms and 12,147 took 5.40 ms, about 0.5
microseconds each, with 2.4 and 3.7 ms of host encode. Metal's fixed
dispatch and barrier cost is therefore about 6% of the 61 ms Qwen3.6 cycle
and 5% of the 114 ms Qwen3.8 cycle. The probe reuses one pipeline and one
set of buffers, so it bounds Metal's overhead, not hf2q's own per-dispatch
host work.

That host work was then measured as the worker thread's CPU time per cycle:
a four-lane Qwen3.6 cycle uses 12.5–13.6 ms of host CPU (about 1.6
microseconds per dispatch) and a four-lane Qwen3.8 cycle 29.6 ms (2.4
microseconds per dispatch), 17–25% of the cycle. The host does not bound
the cycle today; the GPU does. It is a floor on the way down: Splash's whole
Qwen3.6 GPU decode averages 20.99 ms per batch.

A stage profile of the current routes (explicit profiling, one run per arm,
development build) gives the median GPU time per cycle by category at four
lanes on Qwen3.6:

| Category (ms) | GPU adaptive rows | + staged M8, masked routes |
|---|---:|---:|
| Expert FFN (MoE) | 27.24 | 20.69 |
| Full attention | 12.27 | 11.97 |
| Draft command | 8.08 | 8.11 |
| Dense projections | 7.75 | 7.74 |
| DeltaNet recurrence and convolution | 2.43 | 2.35 |
| Output head | 1.32 | 1.33 |
| Other | 1.05 | 1.04 |
| Target total, excluding the draft (median of per-cycle sums) | 53.01 | 45.52 |

Staged M8 with masked routes cut expert GPU time by 24% and target GPU time
by 14% in this pair of runs. Against Splash's 17.93 ms average target forward
and about 3 ms for everything else in its cycle, the largest remaining items
are the expert FFN, full attention (which grows with the lane count: 4.1,
7.9, 10.0 and 12.3 ms at one to four lanes) and the draft command, whose
8.1 ms does not vary with lanes. A two-lane cycle still pays 13.75 ms for
dense projections and 2.75 ms for the head, against 7.75 and 1.32 ms at four
lanes: the 16-row tile takes a slower kernel route than the 32-row tile.

The same profile on Qwen3.8 with the prepared BF16 route (one run, pinned
build) gives a four-lane target of 102.97 ms: stock-kernel projections for
the DeltaNet and attention layers 31.10, prepared FFN 30.09, full attention
28.71, DeltaNet recurrence and convolution 5.69, output head 4.49 and other
1.92, beside an 18.53 ms draft command. Full attention is 9.78, 16.47, 28.32
and 28.71 ms at one to four lanes. The DeltaNet and attention projections are
now the largest item and still run the stock kernel; the earlier rejection of
prepared DeltaNet projections was measured with profiling forced on and is
not a valid rejection.

#### Prepared BF16 target successor

The source-complete Splash comparison then identified a stronger target
differential than the rejected partial repacks above. Splash's prepared target
kernel consumes BF16 activations directly, applies a golden-ratio stagger to
the K walk on Apple10, chooses split-K with a wave/occupancy model, keeps
ordinary projection outputs in BF16, and selects one- versus two-pass gate/up
from that same model. Its Q4_K gate/up pair is intentionally two-pass on the
40-core M5 Max; “fused gate/up” is not an unconditional one-pass rule.

mlx-native's successor preparation retained the source GGUF allocation,
generated the same code/high-bit/metadata plane classes once per persistent
weight, and fed BF16 activations directly to MPP. On the exact Qwen3.8
`M=32,N=17408,K=5120` Q4_K gate shape, the isolated median moved from
0.186 ms for the stock kernel to 0.098 ms for the direct-BF16 prepared kernel,
a 1.895x speedup. The golden-ratio walk reduced the prepared-BF16 point from
0.118 ms to 0.098 ms in the paired development sequence. A one-pass
32-column gate/up kernel measured 0.235 ms, slower than two 0.098 ms prepared
passes, and was removed; this agrees with Splash's M5 Max policy rather than
contradicting it.

The exact Qwen3.8 four-agent diagnostics separate prepared weights from command
chain length. The stock kernel measured 149.218 ms at four lanes with one
64-layer command graph, effectively equal to the preceding 149.539 ms
four-layer-chain control. Prepared BF16-input dense FFNs measured 117.001 ms
over 227 four-lane samples with the legacy four-layer chains and 113.673 ms
over 227 samples with one 64-layer chain. Their output-dependent client walls
were 69.741 and 69.902 seconds respectively. The prepared route is therefore
accepted; forcing one target-layer command graph is not. The latter did not
improve the stock route or end-to-end wall and does not implement Splash's
larger draft-through-commit transaction.

The prepared dense-FFN change alone passes calibration, task/tool quality, and
cleanup and reduces the binding four-lane target stage by 21.76% against the
matched stock 149.539 ms control. This rejects the earlier conclusion that
prepared target weights were intrinsically noncompetitive; those spikes lacked
direct BF16 input. It does not establish parity with Splash.

A fixed six-block balanced campaign at `$E/qwen38-prepared-bf16-d5-v1`
compared the prepared route with the stock GGUF kernels; both arms used the
legacy four-layer command chains. All 12 exact task/tool/cleanup gates passed
and client makespan fell by 12.610% (95% paired-block log-ratio CI
11.243–13.955%; medians 70.898 and 80.979 seconds). That campaign, and every
Qwen3.8 diagnostic in this section, ran with mlx-native per-command-buffer
profiling unintentionally enabled in both arms: the Q5 case exported
`MLX_PROFILE_CB=""`, and mlx-native up to 0.21.6 treats any present value as
on, which makes every labeled commit synchronous. Its absolute times are
inflated and it is superseded by the campaign below. The per-stage numbers and
the accept/reject decisions above and below that rest on them were measured
under the same condition. Command-chain length has since been re-measured
(below); the BF16 gate/up epilogue, the prepared DeltaNet and output-head
projections and the last-arriver reduction have not.

After mlx-native 0.21.6 was published (crate SHA256 `a4287522…6de0`, source
`a02890be`), hf2q was pinned to `=0.21.6` with no local patch; the locked
all-target check, release build and full test suite passed. With that
registry-pinned release binary and identical output (16 requests, 5,361
completion tokens), the same case measured 72.072 and 69.694 seconds with the
empty variable exported and 61.145, 60.267 and 60.089 seconds without it.

The predeclared replacement campaign
`$E/qwen38-prepared-bf16-d5-unprofiled-v1` used that binary, the same fixed
six-block alternating order and the Q5 supervisor that no longer exports empty
case controls. All 12 runs returned `CalibrationPass` with cleanup proven and
no profile dump. Prepared BF16 target FFNs reduced client makespan by 13.925%
(95% paired-block log-ratio CI 12.527–15.300%); per-run medians were 59.944
seconds for the candidate and 69.758 seconds for the control. The arms do not
produce identical text: every candidate run generated 5,361 completion tokens
and every control run 5,255, so the candidate finished sooner while emitting
2.0% more tokens. This qualifies the prepared weight mechanism against hf2q's
stock target kernels.

It supports no gated Splash comparison. The 16.420344-second Splash median
quoted in this ADR is the 17-run Splash arm of the Qwen3.6 campaign
`q4-qwen36-dflash2-release-gate-v2`. Splash does load and serve the Qwen3.8
artifact: 33 runs are retained, and in every one a single agent failed with
"initial six-step workflow not followed", which aborts the run, so none has a
makespan. Earlier statements applying that Qwen3.6 median to Qwen3.8 are
attribution errors, not a valid cross-family performance comparison. D5 and the
comparator-validity risk below follow ADR-063's existing family-specific
amendment; no acceptance threshold is changed by this correction.

The aborted Splash runs still carry per-turn timing on the same workload,
artifact and host. Over the 16 runs of the strict-parity sweep (eight
configurations), Splash produced 41.4 completion tokens per lane-second
(38.1–49.2 by configuration) with a 4.79-second median first turn. The six
unprofiled prepared-route runs produced 28.9 and a 10.61-second first turn;
the six stock-kernel runs 24.4 and 10.90 seconds. A least-squares fit of turn
time on uncached prompt tokens, completion tokens and a constant gives
Splash 2.17 ms per uncached prompt token and 16.63 ms per completion token,
and the prepared route 8.71 ms and 23.00 ms. On the prepared route's own
token counts, prefill and decode each account for about half of the
difference. These are per-lane times under four-lane contention from
truncated Splash runs with different text, so they attribute the gap; they
are not a verdict. hf2q prefills the four lanes of the first turn one after
another (1.88, 1.61, 1.65 and 1.61 seconds for 720 tokens each) and spends
18.7 of about 60 seconds of a run in prefill.

Prefill was then attributed on the Qwen3.8 artifact (one diagnostic run with
per-section host timing, and a model-free probe of the stock GGML matmul at
the served shapes). Per run, 28 content chunks and 28 five-token generation-cue
chunks are prefilled:

| Chunk tokens | Chunks | Mean (ms) | ms per token |
|---:|---:|---:|---:|
| 5 (generation cue) | 28 | 76.5 | 15.30 |
| 31 | 4 | 225.7 | 7.28 |
| 72–118 | 8 | 306.8 | 3.41 |
| 157 | 4 | 439.1 | 2.80 |
| 197–201 | 4 | 505.6 | 2.53 |
| 263–267 | 4 | 750.3 | 2.83 |
| 720 | 4 | 1,682.5 | 2.34 |

Four causes are measured:

1. Every turn prefills twice: the content up to the stable prompt boundary,
   then the five-token cue, because the per-slot anchor needs the DeltaNet
   state at the boundary. The cue chunk is one more pass over the weights:
   28 of them cost about 2.1 seconds of a 60-second run.
2. The stock matmul is slow per row on mid-size chunks. The FFN gate
   projection (17,408 by 5,120, Q4_K) runs at 6.8 TFLOP/s at 32 rows, 12.2 at
   157, 37.7 at 720 and 45.2 at 2,048. The prepared BF16 kernel does the
   32-row case in 0.214 ms against 0.838 ms. Twenty-four of the 28 content
   chunks are 267 tokens or fewer and take about 10.1 of the 16.9 seconds of
   content prefill.
3. A full-attention layer costs 30–54 ms of a 720-token chunk against
   16.7 ms for a DeltaNet layer. The FFN is 10.2 ms in both. The attention
   projections sum to 3.9 ms and the flash-attention kernel to 1.1 ms in
   isolation, so about 20 ms per full-attention layer (about 0.3 seconds per
   720-token chunk) is not yet explained.
4. The four lanes of a turn prefill one after another and nothing is shared
   between them. Splash reports 309 of the 725 first-turn prompt tokens as
   cached on a first turn; hf2q reports none.

The kernel floor for this model is about 45.6 GFLOP per token, or about
1.14 ms per token at the 40 TFLOP/s the stock matmul reaches on large
chunks. The 720-token chunk is at 2.34 ms and the mid-size chunks at
2.5–7.3 ms.

Command-chain length was then re-measured without profiling on the same
binary and prepared route, each run emitting the identical 5,361 completion
tokens. Thirteen four-layer-chain runs span 58.074–61.660 seconds.
`HF2Q_PARTIAL_CHAIN_N` of 1, 2, 3, 6, 8, 16 and 64 measured 59.865, 59.413,
58.759, 59.322, 57.990, 58.603 and 61.351 seconds. Chain length therefore
has no measurable effect from 1 to 64 at single-run precision. A first pass
measured 1, 16 and 64 at about 63 seconds; those three runs directly
followed a ten-minute Qwen3.6 campaign, and a later Qwen3.8 run that
directly followed Qwen3.6 runs measured 73.162 seconds against 59.7–61.7 for
the same binary minutes afterwards. Runs that directly follow a different
model's load are not comparable and are excluded here.

The follow-up two-pass BF16 gate/up epilogue also passed model-free parity and
the exact task/tool gate, but its 113.745 ms four-lane target median was flat
against the prepared F32-output route, so it remains rejected. Preparing the
three large DeltaNet projections added about 3.55 GiB and regressed the
four-lane median from 117.001 to 120.029 ms; preparing the output head added
about 0.98 GiB and moved the same median only to 116.708 ms while worsening
the output-dependent client wall. Both were removed. Finally, replacing the
separate split reduction with a device-coherent last-arriver reduction
regressed the exact four-lane target median from 116.708 to 127.588 ms despite
an isolated microbenchmark gain, so the canonical separate reduction remains.

#### Splash per-stage targets and prepared projections

Splash's own developer tools were then built out of tree from the pinned
commit and run on this host under the host lock
(`$V/splash-kernel-bench/REPORT.md`: projection, MoE and attention
microbenchmarks, and `decode-profile`, which loads the real target and draft
without a server and attributes one DFlash cycle per dispatch). They replace
the estimates above with measurements. One four-lane verify cycle, GPU ms:

| Stage | Splash Qwen3.8 | hf2q Qwen3.8 | Splash Qwen3.6 | hf2q Qwen3.6 |
|---|---:|---:|---:|---:|
| Draft command | 5.98 | 16.2 | 3.36 | 8.1 |
| Target FFN (dense, or the whole MoE layer) | 24.0 | 26.7 | 9.7 | 20.7 |
| Target DeltaNet (projections and recurrence) | 12.8 | about 22 | 4.7 | about 9 |
| Target full attention (projections and kernels) | 4.7 | about 24 | 1.8 | about 12 |
| Target head | 1.95 | 4.3 | 0.86 | 1.3 |
| Norms and other | 1.2 | 1.8 | 0.4 | 1.0 |
| Whole cycle | 50.6 | 96 GPU, 110–118 wall | 20.8 | 54 GPU, 56–61 wall |
| Dispatches per cycle | 626 | 12,147 | 655 | 8,107 |

The hf2q columns are one explicit stage profile each (Qwen3.8 with prepared
FFN and the prepared projections below; Qwen3.6 with staged M8 and GPU
adaptive rows); "about" marks categories split by hand because hf2q's
attention category includes the Q/K/V projections and its projection category
includes the attention output projection. Splash prefills 512 Qwen3.8 rows in
478–532 ms and 2,048 rows in 1,875 ms. hf2q is therefore 2.2–2.3 times
Splash's cycle on Qwen3.8 and 2.7–2.9 times on Qwen3.6, and every stage has a
measured target. Splash issues one fused dispatch for a DeltaNet layer's
qkv, gate, alpha and beta projections and one for an attention layer's Q/gate,
K and V; its attention kernels cost 0.11–0.17 ms per layer against about
1.3 ms for hf2q's.

A serialized-dispatch probe explains the earlier projection result: the stock
32-row GGML matmul has a floor of about 0.1 ms per dispatch whatever its
size (0.133, 0.113 and 0.101 ms for 8,192 by 5,120, 4,096 by 5,120 and 5,120
by 4,096 Q4_K), while the prepared BF16 kernel takes 0.057, 0.031 and
0.027 ms. Numbers taken
with eight unbarriered dispatches per command overlap and hide that floor.

`HF2Q_REPACKED_GGUF_M32_PROJ=1` (with `HF2Q_REPACKED_GGUF_M32=1`) therefore
prepares the DeltaNet qkv, gate and output projections and the attention
fused Q/gate, K, V and output projections on first use, and reads one BF16
copy of each shared activation. A fixed six-block same-binary campaign
(`$E/qwen38-prepared-proj-d5-v1`, DEBUG stage timing on both arms, run while
other builds loaded the host) passed all 12 gates: the four-lane target
stage fell by 7.07% (95% CI 4.78–9.30%; medians 117.66 against 127.02 ms) and
client makespan by 4.32% (95% CI 0.39–8.08%; 66.02 against 68.76 seconds).
The two arms emit different text (5,385 against 5,361 completion tokens per
run), and the process footprint peaked at 69.95 GiB against 65.34 GiB. The
explicit stage profile moved projections from 31.1 to 18.6 ms and attention
from 28.7 to 22.8 ms. One exact Qwen3.6 run with the switch passed with a
49.19 ms four-lane target median against 51.92 ms for its control. This
withdraws the rejection of prepared DeltaNet projections recorded above. The
route stays opt-in pending the combined-candidate campaign.

Two follow-ups were then measured the same way, each against the preceding
binary with otherwise identical settings.

The alpha and beta logit projections (48 outputs each on Qwen3.8) were the
last stock products of a DeltaNet layer's input wave, so that wave still paid
the stock floor. They now run as one prepared product: alpha's rows, beta's
rows and zero rows up to 128 outputs form one weight, and its output is split
after the wave's barrier. This is Splash's fused alpha/beta segment. The
six-block campaign (`$E/qwen38-proj-ab-d5-v2`) passed all 12 gates: four-lane
target stage −5.29% (95% CI 0.51–9.84%; 102.93 against 107.25 ms), client
makespan −3.48% (95% CI 0.55–6.32%; 58.44 against 60.17 seconds). It is part
of `HF2Q_REPACKED_GGUF_M32_PROJ`.

`HF2Q_REPACKED_GGUF_M32_HEAD=1` runs the 32-row output head through the
prepared route. The six-block campaign (`$E/qwen38-proj-head-d5-v1`) passed
all 12 gates and shortened the four-lane target stage by 3.27% (95% CI
1.98–4.55%; 94.20 against 97.39 ms), but client makespan did not move (0.66%
higher, 95% CI from 2.81% higher to 1.45% lower). The head's logits differ
from the stock head's, so the text differs: this arm emitted 5,147 completion
tokens against 5,237, ran 377.5 cycles against 354 because its lanes finished
less evenly, and accepted 5.21 tokens per lane-cycle against 5.33. It adds
0.94 GiB of footprint. The earlier rejection of a prepared head is withdrawn
as a kernel result and stands as an end-to-end result on this workload; the
switch is kept off by default for the combined-candidate campaign to decide.

These campaigns repeat one deterministic conversation per arm, so their
confidence intervals cover timing noise only. Two arms that emit different
text are also running different workloads, and six repeats do not sample
that. Stage time per cycle is the comparable quantity between such arms;
makespan between them is one draw.

#### Staged draft command

A probe that replays every dispatch of one draft command as its own command
buffer (`dflash2::probe::dflash2_draft_command_dispatch_probe`, real target
and checkpoint, four lanes, 512 ring rows per lane, median of 20 rounds)
reproduces the profiled draft cost and splits it. GPU ms, Splash from its
`decode-profile` dispatch lists on the same host:

| Group | Splash Qwen3.8 | hf2q Qwen3.8 | staged | Splash Qwen3.6 | hf2q Qwen3.6 | staged |
|---|---:|---:|---:|---:|---:|---:|
| Projections | 2.89 | 8.23 | 2.55 | 1.46 | 2.36 | 1.14 |
| Attention | 0.34 | 3.30 | 0.30 | 0.39 | 3.95 | 0.37 |
| Norms, convolutions, input | 0.20 | 0.24 | 0.17 | 0.19 | 0.20 | 0.15 |
| Vocabulary head | 1.95 | 4.38 | 1.96 | 0.85 | 1.33 | 0.85 |
| Selection | 0.14 | 0.15 | 0.14 | 0.15 | 0.14 | 0.14 |
| Draft command | 5.51 | 16.30 | 5.12 | 3.04 | 7.98 | 2.64 |
| Context commit | 0.46 | 1.09 | 0.37 | 0.32 | 0.50 | 0.23 |

The three differences are kernels, not dispatch count (171 draft dispatches
against Splash's 90). hf2q multiplied AF4G64 planes on the register tile
where Splash's Apple10 plan stages them (`runtime/ops/LinearGguf.cpp:190-212,
272-275`) and runs gate and up in one pass
(`runtime/metal/kernels/shared/gguf_linear.metal:131-180`). hf2q copied every
lane's 2,047-row ring window into a packed K/V tile and filled a dense mask
before each layer's flash attention, whatever the context length; Splash
multiplies the ring in place over the live 128-slot tiles in four splits
(`runtime/metal/kernels/decode/draft.metal:151-398`). hf2q's draft head ran
the stock Q6_K kernel; Splash runs the target's prepared tile
(`runtime/model/DFlashDraft.cpp:334-341`).

`HF2Q_DFLASH2_DRAFT_STAGED=1` (requires `HF2Q_DFLASH2_DRAFT_AF4_G64=1`)
adopts all three on unchanged weights: mlx-native's
`dflash2_af4g64_staged` tile for projections of at least 960 outputs, with
the split count from Splash's wave law evaluated once for 40 cores and 32
rows so that it never depends on the row count; `dflash2_ring_attention`;
and, at four lanes, the prepared head. Projections and attention are Class B
against the default route. Each kernel has an F64 reference test with a
K-scaled bound, a determinism test and a row and lane independence test
(bitwise, across tile heights, positions and neighbours). Sparse probes
check that every 32-input step of every served projection shape, up to
K = 25,600, is summed exactly once; the served 32-over-8-head attention is
checked at every tile and window boundary, alone and in every cohort; and
every test's scratch starts as NaN. The ring split kernel holds 16,384 bytes
of threadgroup memory, which Metal shader validation doubles to the device
limit, and the kernel tests give the same bits under
`MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`. A synthetic checkpoint runs
both routes to within 5e-3 relative RMS in rings, hidden states and logits,
and the prepared head to within 5e-3 of the stock head. Against the pinned
Qwen3.6 reference fixture the first layer's relative maximum error is 0.1960
staged and 0.1962 on the register tile: four-bit quantization, not the tile,
sets the draft's distance from the BF16 oracle.

In situ on the exact four-agent Q5 workload, each arm on one binary, with the
draft command waited (`HF2Q_DFLASH2_GPU_TARGET_TOKENS=0`) so that `draft_us`
is its wall time, the four-lane median fell from 17.20 to 6.01 ms on Qwen3.8
(one run per arm) and from 8.25 and 8.19 ms to 3.07 ms on Qwen3.6 (two
passing off-runs, one passing on-run). `draft_commit_us` fell from 1.39–1.50
to 0.60–0.66 ms on Qwen3.8 and from 0.72–0.78 to 0.45–0.47 ms on Qwen3.6 in
every passing run. In the served GPU-token mode Qwen3.8 emits the same
completion tokens with the switch on or off (5,237 in each of seven runs of
the measured binary) and accepts 5.334 against 5.330 tokens per lane-cycle;
rebased onto the capture-wait fix (`eea0fa1a`), one run per arm accepts 5.366
against 5.370 for the same 5,237 tokens (1,184 against 1,183 lane-cycles).
On Qwen3.6, whose text varies between runs of one arm, four on-runs accepted
4.03–4.18 tokens per lane-cycle against 4.10 and 4.25 for two off-runs. All
six Qwen3.8 and all four Qwen3.6 on-runs in the GPU-token mode passed
calibration.

Root's six-block paired campaign on that binary
(`$E/qwen38-draft-staged-d5-v1`: Qwen3.8, GPU-token mode, the switch against
unset with every other setting equal, fixed alternating order, a loaded
host) passed all 12 runs with identical text in both arms (5,237 tokens).
The four-lane cycle fell by 7.41% (95% CI 6.70–8.12%; medians 126.96
against 136.56 ms), the four-lane target stage by 6.87% (6.17–7.56%; 121.43
against 130.03 ms) and client makespan by 5.89% (4.02–7.71%; 67.94 against
71.80 seconds). Peak footprint was 71.10 against 70.08 GiB. The owner's
seven unpaired runs had not shown this (on 109.7, 108.6, 95.7 and 98.9 ms
against off 100.9, 106.4 and 103.3 ms); root attributes the two slow
on-runs to host load. In the waited mode the four-lane cycle fell from
102.0 to 92.7 ms in one run per arm.

Qwen3.6 is not qualified. In the waited mode, which is the Qwen MoE default,
two of three on-runs failed the six-step workflow oracle on model text (a
tool call cut short after a reasoning-budget close, and an extra edit after
a compiler warning); two off-runs did not. Every emitted token is the
target's own decision, so a draft cannot emit an unverified token. In that
mode, however, the host selects verify rows from the draft's confidence and
the Qwen3.6 verify tile width follows the row total
(`dflash2_verify_width`), so a different draft changes the target kernels
that run and the target's logit bits. Dense Qwen3.8 verifies at a fixed
width, which is why its text is identical. Five runs do not separate the
arms (Fisher p about 0.4); root's waited-mode Qwen3.6 campaign must.

The head is prepared a second time beside `HF2Q_REPACKED_GGUF_M32_HEAD`
(0.98 GiB on Qwen3.8 and 0.39 GiB on Qwen3.6 by plane size) until the two
routes share one. The switch stays off by default.

#### Prepared routes at every joint width, shared head (wave 2)

`HF2Q_REPACKED_GGUF_SHARED_HEAD=1` makes the model own one prepared output
head (`Qwen35Model::prepared_output_head`, prepared in a command buffer of
its own when the first consumer that reads it asks); a staged draft command
of 32 draft rows takes it through
`DFlash2TargetFamily::shared_prepared_vocabulary_head` and the prepared
target head reads the same planes. With neither consumer nothing is
prepared. Logits are bitwise those of the separate
copies (`staged::tests::shared_prepared_head_gives_the_bits_of_the_drafts_own_copy`,
`prepared_width_tests`). On Qwen3.8 the prepared target head then costs no
footprint: peak 75.80 / 75.69 GB with the shared planes against 75.81 /
75.79 GB with the stock target head, and 76.75 GB (one run) with the
prepared target head and a second copy.

`HF2Q_REPACKED_GGUF_ANY_WIDTH=1` admits the prepared dense FFN, projections
and head at every tile width of a DFlash2 target verify step, 1 to 32 rows
(one rule, `gpu_ffn::repacked_rows_eligible`, behind
`prepared_projection::admits*`). The Qwen3.6 verify tile is 4, 8, 16 or 32
rows; before, only 32 took the prepared kernels, so one- and two-lane cycles
ran the stock ones and a row's bits depended on the tile width.

The admission is confined to DFlash2 verify transactions
(`JointPacking::dflash2_verify`, `joint_projection::dflash2_verify_scope`).
Plain joint decode and joint MTP verify steps keep their routes under every
switch, because in the exact mode their rows are held to the serial
one-token forward's bits and the serial forward never takes a prepared
kernel: the official-artifact gate
`qwen36_joint_decode_matches_serial_decode_bitwise` passes with every switch
on. The consequence is the one that already held at 32 rows: a token
verified by DFlash2 and the same token decoded serially or in a plain joint
step are computed by different routes (Class B), now at every tile width.
Without the switch a one-lane Qwen3.6 verify (8 rows, stock kernels) was
bitwise a plain joint or serial step; with it, it is not, so a lane that
falls out of DFlash2 changes route.

Gates. Model free (`prepared_width_tests`): a row of a prepared projection,
of the fused alpha/beta product and of the whole dense FFN layer has the
same bits at widths 1, 4, 8, 16, 24 and 32 in three cohorts each and beside
padding rows of NaN, infinity, a half-overflowing value and a denormal;
repeated runs agree; and each of the three is inside a derived bound of an
F64 reference over the dequantized weights (the dense FFN's bound is
dominated by the BF16 rounding of the hidden activation, 2^-8 relative).
Whole step, synthetic model, switches set through a test-only override
(`prepared_step_tests`): each family is wired to its switch; in the dense
layout (32-row tile, 8 to 32 live rows) and the MoE layout (tiles of 4, 8,
16 and 32 rows; quantized dense FFN layers stand in for the MoE layers) a
lane's rows, committed cache and captures are bitwise the same in every
cohort, packing order and padding, on the host-token and GPU-token paths
(padding rows holding other tokens), with the per-row and the tiled verify
attention, and equal the 32-row tile's rows without the switch; plain joint
and joint MTP steps are bitwise unchanged by the switches and equal the
serial forward in the exact mode. Real MoE layers:
`qwen36_dflash2_verify_rows_have_one_value_at_every_tile_width` (ignored,
official artifact). The model-free gate also passes under
`MTL_SHADER_VALIDATION=1` with and without `MTL_DEBUG_LAYER=1`. The
whole-step gate skips itself under shader validation: there a whole joint
step of the synthetic model did not always repeat its bits (three of four
validated runs failed, each in another comparison, stock steps included),
which was not explained; the official-artifact gate was run plain only.

Exact Q5 runs on the combined candidate (binary `455a172c`, every other
candidate switch on; "on" is `_M32_HEAD`, `_SHARED_HEAD` and `_ANY_WIDTH`
together), all `CalibrationPass`:

| Model, arm | 1 lane ms | 2 lanes ms | 3 lanes ms | 4 lanes ms | accepted per lane-cycle (4 lanes) | dispatches (4 lanes) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8 off (two runs) | 61.3 / 61.4 | 62.8 / 62.9 | 67.5 / 67.5 | 72.5 / 72.4 | 5.30 / 5.32 | 9,063 |
| Qwen3.8 on (two runs) | 59.8 / 59.2 | 60.5 / 60.1 | 64.9 / 65.2 | 71.0 / 70.7 | 5.22 / 5.22 | 9,208 |
| Qwen3.6 off (two runs) | 22.6 / 22.1 | 36.8 / 34.5 | 29.3 / 27.8 | 33.6 / 32.8 | 3.97 / 4.07 | 6,316 |
| Qwen3.6 on (three runs) | 17.1 / 17.4 / 18.6 | 22.0 / 23.0 / 23.0 | 28.5 / 28.0 / 28.0 | 32.3 / 32.9 / 32.6 | 3.89 / 4.00 / 3.96 | 6,317 |

Qwen3.8's gain is the prepared target head (the dense target always verifies
at 32 rows). The dispatch medians differ by the head's input cast; the
9,063 / 9,208 split is a bimodal per-cycle count present in both arms.
Diagnostic category profile on Qwen3.6 (one run each, GPU ms, widths of one
to four lanes): head 4.55 / 4.27 / 1.95 / 1.65 off against 0.86 / 1.10 /
0.94 / 1.02 on; projections 11.93 / 20.71 / 5.20 / 4.69 off against 4.45 /
5.56 / 5.23 / 5.42 on.

The review fixes (admission confined to DFlash2 verify steps, planes
published after their own command buffer, shared head prepared on demand)
were then run from a registry build (binary `a5aaae28`, crates.io mlx-native
0.21.8, no path override), every switch on, all `CalibrationPass`:

| Model (runs) | 1 lane ms | 2 lanes ms | 3 lanes ms | 4 lanes ms | accepted per lane-cycle (4 lanes) | dispatches (4 lanes) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6 (four) | 18.6 / 18.8 / 18.0 / 17.8 | 24.3 / 24.3 / 23.2 / 23.5 | 29.8 / 30.3 / 29.1 / 28.6 | 34.2 / 35.0 / 33.4 / 33.1 | 3.97 / 4.09 / 3.98 / 3.97 | 6,317 |
| Qwen3.8 (two) | 71.8 / 60.2 | 61.3 / 61.2 | 71.6 / 65.7 | 72.4 / 72.0 | 5.23 / 5.26 | 9,064 |

These runs were hours later on a busier host than the table above, so the
two tables are not a comparison of the binaries; the Qwen3.8 transcript and
the two Qwen3.6 transcripts are those of the earlier binary.

Run-to-run text on Qwen3.6. With the switches off every pair of runs
differs in every agent within its first five turns (three runs compared).
With them on, eight runs (seven timing runs of two binaries and one
profiled run, whose cycle mix and timing are very different) produced two
transcripts: six runs one, two runs the other, which differs from the
first at one token of one agent's last turn and nowhere else (three of the
four registry-build runs are identical; the fourth is the other
transcript). So the width-dependent projections and head were the main
source of the variation but not the only one. The independent review of
this change located a remaining source in the tiled verify attention: its
split length follows the window's start cursor, so the same token verified
at another window offset can differ once the history crosses a split
boundary (first at 1,536 tokens, then every 768), and the window
boundaries follow a verify budget fed by wall-clock time. That belongs to
the attention tile and the scheduler, not to this change, and was not
fixed here; mlx-native 0.21.9 removes it (see the attention-kernel
differential below). Eight runs do not measure quality; the
workflow-oracle pass rate needs root's long comparison.

Prepared planes are published only after the command buffer that writes
them has completed (`prepared_projection::ensure_planes`; one buffer per
projection weight, one per dense FFN layer for gate, up and down). Before,
a cell was set when the repack was encoded into the caller's command
buffer, so a forward abandoned between that encode and its commit left
zero planes cached for the life of the model (an independent probe read
640 of 640 outputs zero), and the prefill route shares those planes.
`prepared_width_tests` covers the projection, fused alpha/beta and FFN
cells: a failed preparation publishes nothing and the retry succeeds, and
the planes an abandoned encoder published are the weight's. The price is
one GPU wait per weight on first use, paid by the first prefill chunk
after load (the prefill route prepares the planes the verify step then
finds): 560-613 ms against 520-576 ms on Qwen3.6 (four runs against three)
and 2.43-2.48 s against 2.05-2.07 s (two runs against four) on Qwen3.8; the first verify cycle, which prepares
only the head, took 44.7-50.4 ms against 47.6-48.0 ms on Qwen3.6 and
171.5-196.9 ms against 166.9-177.7 ms on Qwen3.8.

A one-pass gate/up pair over the prepared planes with a BF16 SiLU epilogue,
a residual folded into the down projection's reduce pass and a row-batched
argmax were built and rejected: none was faster on the GPU, and the pair
was not bitwise the separate route (3 of 278,528 products). Model free, at
the GPU's top performance state, the existing prepared FFN layer takes
0.354-0.42 ms (Q4_K down) and 0.39-0.52 ms (Q6_K down), 23.9-30 ms per 64
layers against Splash's 24.0 ms. The same projection reads 0.118 ms when
the GPU queue never drains and drifts to 0.18 ms within seconds when each
sample leaves an idle gap, so the profiled FFN category (29.1 ms) measures
the cycle's duty cycle, not a kernel deficit
(mlx-native `benches/bench_qwen_dense_ffn.rs`).

#### Live-tile MoE expert route

The Qwen3.6 MoE stage was then attributed model-free at the served shape
(mlx-native `bench_qwen36_moe_layer`: hidden 2,048, 256 experts, top 8,
intermediate 512, one weight set, 16 layers per command, median of 20
commands). Its routes are the expert ids of real four-lane verify steps,
recorded from one exact Q5 run: 24.6 of 32 rows live, 80 live experts, 69% of
them with one or two routes. Splash's `decode-profile` figure of 9.7 ms per
cycle (0.24 ms per layer) does not describe that routing. Its four lanes
replay one synthetic prompt, so every lane routes to the same experts and no
expert holds fewer than four routes. Splash's own MoE operator, replayed on
the recorded routes through a copy of its `gguf-moe-benchmark`, takes 0.340 ms
per layer with Q4_K down experts and 0.383 ms with Q6_K; on those routes with
one lane repeated in all four it takes 0.212 and 0.250 ms. The staged-M8 route
takes 0.465 and 0.51–0.54 ms on the recorded routes. Serialized, its router
and shared-gate logit cost 0.068 and 0.063 ms (a 32-row tensor matmul of four
and one threadgroups), the shared expert 0.10 ms, gate and up 0.30 ms against
0.21 ms for Splash's one pass, and down 0.13 against 0.10 ms.

The earlier one-pass BF16 route was re-examined on fixed inputs
(mlx-native `moe_expert_tiles`, ignored test `fused_bf16`). Its kernel stores
exactly `BF16(SiLU(BF16(gate)) * BF16(up))` of the sums the served kernels
produce (65,285 of 65,536 elements bitwise; the rest are SwiGLU values that
underflow, within 4e-38 of the row's RMS), so it has no kernel defect. Against
an F64 reference its SwiGLU and down outputs carry 4.0e-3 and 4.3–4.6e-3
relative RMS error, the served route 3.3e-3 and 3.7e-3, and Splash's
arithmetic order 3.6e-3 and 3.5–3.6e-3 (3.9e-3 with its BF16 down store). The
withdrawn route was Splash's order except that it staged weights as BF16
where Splash stages half. One failed task run does not separate a 1.2 times
larger rounding error from the workload's sensitivity to any change of bits.

`HF2Q_QWEN_MOE_LIVE_TILES=1` (with `HF2Q_QWEN_MOE_STAGED_M8=1`) adopts
Splash's plan on the staged-M8 planes (`runtime/ops/MoE.cpp:102-166,225-283`,
`kernels/shared/moe_gguf.metal`, `kernels/common/gguf_staged_tile.h:83-130`,
`gguf_column_tile.h`): one route schedule; one gate/up pass whose SwiGLU stays
in the schedule's grouped rows; a down pass over those rows; both on the
indirect grid of live tiles; weights staged as half in two alternating stages
with the next payload loaded during the matmul; experts of one or two routes
in registers; the shared expert as tiles of the same passes. It differs from
Splash in three ways. The SwiGLU is rounded to BF16 once, from FP32 gate and
up (2.6e-3 and 2.6–2.8e-3 relative RMS error, the lowest of the four). The
register path adds a group's products in the order the device's `matmul2d`
adds them, found by probing and re-proved at model load, so it is bitwise the
matmul tile; Splash documents that its two tiles differ and that its MoE
results depend on the step's other rows (`moe_expert_slab.h:16-27`). The
router and shared-gate logits run the row-exact kernel per block of eight
rows. A row of a masked width-32 step is therefore bitwise the row decoded
alone, which the staged-M8 route does not guarantee. The equality excludes
expert products that underflow FP32: the register path and `matmul2d` may
round a subnormal product differently, so a result of that magnitude (an
independent review observed at most 4e-31) can depend on how many routes
share its expert. The kernel keeps 8-row tiles only: a kernel that also held
16- and 32-row matmuls ran every tile 15% slower. The three MoE switches are
resolved together before any weight upload, so an invalid value or
combination fails startup, and the worker logs the resolved route as
`moe_experts`.

Model-free, the layer takes 0.375 ms (Q4_K down) and 0.411 ms (Q6_K down) on
the recorded routes, 1.10 and 1.07 times Splash on the same routes. In situ,
ten of ten exact four-agent Q5 runs with the switch and six of six without it
were `CalibrationPass`, over three successive binaries of this branch (two,
four and four runs with the switch, two without on each; the first binary ran
the shared expert on its FP32 kernels in blocks of eight rows). On the last
two binaries the median four-lane target stage is 45.0 ms with the switch
(eight runs, 44.2–46.7) against 48.2 ms without (four runs, 47.8–48.4).
Explicit stage profiles (diagnostic, one pair per binary) put the MoE
category at 17.6 against 21.2 ms at four lanes on the final binary. What
remains against
Splash on the same routes is the gate/up pass (0.219 against 0.208 ms), the
schedule and its extra barriers, and, on routes where lanes share experts,
experts of more than eight routes, which this route runs as successive 8-row
tiles in one threadgroup where Splash launches one threadgroup per tile. The
route needs unpublished mlx-native kernels and is not quality-qualified.

Second pass (wave 2). Measured inside the served cycle without the category
profile (a diagnostic binary that gives every verify cycle one variant and
reads each command buffer's GPU timestamps), the stage cost 15.5 ms of GPU
time per four-lane cycle at the 20.3 rows the combined candidate verifies
(8.7 ms plus 0.335 ms per verified row), not the profile's 17.6 to 21.5 ms:
profiled cycles verify 24 to 25 rows because their waits flatten the verify
budget's cost line, the category also holds the post-attention norm and the
capture copy, and GPU time drifts inside a profiled process. The model-free
0.375 ms came from one set of expert weights re-read sixteen times per
command; with every layer on its own planes, as served, the layer took 0.323
ms (Q4_K down) and 0.358 ms (Q6_K down) on the candidate's routes against
Splash's 0.294 and 0.326 ms. The target of 13.6 ms was Splash on one weight
set at 25 live rows; on the served routes and memory pattern it is 12.3 ms
(medians of nine Q4_K-down and six Q6_K-down runs of 20 commands).

The route now encodes its expert stages with `moe_tiles_fused` (Splash:
`kernels/shared/moe.metal` grouping, `moe_expert_slab.h:63-100`): one
schedule that applies the verify-row mask, emits tiles of at most eight
routes (an expert of more routes is several threadgroups) and appends the
shared expert's tiles; one gather; one gate/up and one down pass for routed
and shared tiles together; and the router and shared-gate logits of all full
blocks of eight rows in one dispatch each of the row-exact kernel. A layer
is 8 dispatches and 6 barriers at 32 rows instead of 19 and 7. Every route
keeps the tile arithmetic and register-path rule, so the layer is bitwise
the stage-by-stage form (whole layers, masked and unmasked, widths 1 to 32,
experts of up to 32 routes), which remains as the fallback for layers whose
routed and shared planes differ in format or width. Model-free the layer
takes 0.296 ms (Q4_K down) and 0.331 ms (Q6_K down), 12.5 ms for 40 layers
against Splash's 12.3. In the served cycle, alternating the two forms per
verify cycle in one process (506 four-lane cycles of two runs), the fused
form uses 0.90 ± 0.13 ms less GPU time per cycle and its MoE stage takes
14.8 instead of 15.7 ms at 20.6 verified rows. Four exact Q5 runs per form,
interleaved on a host whose cycle time drifted between 34 and 43 ms, do not
resolve that difference (median four-lane cycle 39.1 against 38.2 ms, three
of four passes each); dispatches per cycle fall from 6,316 to 5,876. About
1.5 ms of the stage in the server is not reproduced model-free: a GPU left
idle before the command, as between cycles, accounts for 0.4 ms of it.

The lower accepted-token count per lane-cycle of the combined candidate
(4.00 against 4.17) follows the verify budget, not the tiles' arithmetic:
the candidate removes fixed cycle cost (40.3 to 23.8 ms) while a verified
row still costs 0.36 ms, so the budget verifies 4.29 instead of 4.73 draft
rows per lane-cycle, and accepted tokens per second rise from 321 to 473.
The first draft token matches equally often in every arm (0.815 to 0.823).
On identical inputs (128 independent verify windows after a common prefill,
teacher-forced), the staged-M8 route and the live tiles sit equally far from
the FP32-expert route (relative logit error per window 3.2e-2 ± 0.25e-2 and
3.1e-2 ± 0.18e-2, ten argmax differences in 1,024 rows each), and running
the shared expert or the router on the staged-M8 route's FP32 kernels
changes neither figure.

#### Splash attention-kernel differential

The full-attention kernels (K/V cache write, attention, output gate; the
Q/K/V and output projections are counted above) were then attributed model
free. A DFlash2 verify step repeats each lane's slot eight times, so
`tq_sdpa_rows_route` never admitted it and every full-attention layer ran the
serial chain once per row behind barriers: two TQ encodes, the Q rotation,
the split-K main and its reduce, the un-rotation and a row copy, 225
dispatches in about 160 serialized stages per layer at four lanes, each row
and query head scanning the slot's history on its own. mlx-native
`bench_qwen_verify_attn` (one layer, served shapes, GPU at serving clocks,
median of 20, `$V/splash-gap/attnverify/bench-qwen3{8,6}.log`) reads 1.03, 1.81
and 3.59 ms per Qwen3.8 layer at four lanes over 512, 2,048 and 8,192 tokens
of history (0.27, 0.46, 0.90 ms at one lane); Splash's three dispatches take
0.073, 0.130 and 0.341 ms. Running the same per-row kernels row-batched
behind two barriers (bitwise the serial chain) only reaches 0.36, 1.14 and
2.31 ms: past a few hundred tokens the per-row kernel's own work dominates,
because each of the 32 rows and 24 query heads dequantizes the history again.

Splash computes a lane's eight rows and a KV head's query heads together, 48
fused rows for Qwen3.8 and 64 for Qwen3.6, one threadgroup per KV head, lane
and history split, with Metal tensor matmuls over 32-token pages, and
combines the splits in a fixed order (`paged_attention_tile.h:279-387`,
`:414-457`; split count from the lane's own history only,
`PagedAttention.hpp:56-73`). `HF2Q_QWEN_VERIFY_ATTN_TILE=1` adopts that
structure over hf2q's unchanged TQ-HB cache (mlx-native
`tq_hb_verify_tile`): one batched K and one V encode for all rows, one query
rotation and pack, one split, one reduce and one fused un-rotation and gate,
six dispatches in four stages. The Lloyd-Max codebook is not linear, so a
page's codes are dequantized to half centroids in threadgroup memory instead
of being a matmul operand as Splash's INT8 pages are; that costs one more
barrier per page and no change to the cache format, the prefill writer, the
anchors or the spill path. Cache bytes and cursors are bitwise the serial
chain's. The attention output is Class B: half matmul operands and a
per-page online softmax. Its gates are an F64 reference with a K-scaled
bound at both served geometries (largest observed error 0.43 of the bound),
bitwise determinism, and bitwise independence of a row from the other lanes,
the window's later rows, its packed row and the padding. Against the serial
chain on hf2q's cache one layer's gated output moves by at most 1.0% of the
row RMS in a deliberately peaked fixture, a relative RMS of 6.2e-4 to 7.9e-4
per row, also across five steps with partial acceptance and cursor
truncation; the test limits are 2% and 2e-3.

An independent review measured the route's error against the exact F64
attention at 3e-4 on flat and about 1e-3 on peaked attention, 1.1 to 2.5
times the serial chain's on peaked attention; half the tile's error is the
half query the serial chain also uses, the rest the half centroids. The
probability operand of P·V is `p * norm / 16` in half: it saturates for value
norms above 1,048,064, and a token that shares a split with a dominant one
drops out once the operand is below 3e-8, which is invisible for norms from
1 up, 1.7e-3 in the worst head at 0.25 and 2% at 0.02. The stored norms were
therefore surveyed on the real caches: a diagnostic binary printed, every 50
verify commits of one exact four-agent run per model, the norms of every
committed token of the step's lanes (about 2,900 tokens per lane at the
end, `$V/splash-gap/attnverify/runs/norm-table.txt`).

| Stored value norm | min | 0.1% | 1% | 50% | 99% | 99.9% | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8, lowest layer (rank 1) | 2.02 | 2.16 | 2.58 | 7.03 | 11.90 | 13.91 | 15.40 |
| Qwen3.8, highest layer (rank 15) | 56.70 | 60.48 | 65.21 | 84.99 | 107.09 | 125.94 | 168.20 |
| Qwen3.8, all 16 layers, all snapshots | 1.99 | | | | | | 168.20 |
| Qwen3.6, lowest layer (rank 1) | 2.89 | 3.16 | 3.50 | 7.01 | 9.97 | 11.12 | 13.19 |
| Qwen3.6, highest layer (rank 9) | 16.72 | 17.36 | 17.90 | 26.27 | 33.20 | 34.65 | 36.33 |
| Qwen3.6, all 10 layers, all snapshots | 2.83 | | | | | | 36.33 |

Key norms are 16.2 to 29.1. No layer is below 0.5 or above 1e5, so the
operand needs no rescaling for these models; a model whose value norms left
1 to 1e6 would need a per-row power-of-two operand scale first.

The review also found that the split kernel's 26 to 29 KiB of static
threadgroup memory doubled under Metal shader validation and aborted under
the debug layer. That storage is now threadgroup memory bound by the host
(same arithmetic, equal output digests plain, under either layer, under
both, and from runtime-compiled source), and the route reports itself
unavailable, so the step keeps the per-row chain, when a kernel's threads or
threadgroup memory exceed the device's limits. A step the switch asked for
and the route refused logs its reason once.

A later review found that a token's bits depended on its verify window. As
released in mlx-native 0.21.8 the tile cut a lane's history into splits by
the pages of the whole window, so a window that crossed the page boundary
where the split length steps (every 768 tokens of history from 1,536 on) gave
its rows before the boundary another partition than the same tokens get in a
window that ends there: at cursor 1,525 none of four tokens kept its bits and
one logit moved by 0.85. Window starts follow the verify budget, which is fed
by wall-clock time, so the served text could change from run to run. From
mlx-native 0.21.9 a token's split length is a function of its own page count.
Rows that ask for the same length share a tile as before, because a page or
a split that holds none of a row's tokens leaves its bits alone, and a window
whose rows ask for two lengths runs as two tiles on the lane's scratch. A
token's output is now bitwise the same at every row index, window length,
lane count and width. mlx-native verifies every position up to 9,152 tokens
of history at every row index over every window start, and every window
start within 48 tokens of each of the 43 such boundaries up to 33,824 tokens,
at both geometries, plain, under Metal shader validation, under the debug
layer and under both; with the cut disabled the first test fails on 70
Qwen3.8 tokens. Its F64 reference, determinism, saturation, cohort and
causal-end tests each include cut windows.
`verify_attn_tile_window_tests` repeats the sweep on hf2q's cache: 24
positions around every such boundary below 9,000 tokens, around plain page
boundaries and at an empty slot, under eleven window plans including partial
acceptance, in steps of one to four lanes at widths 8, 16 and 32, with equal
cache bytes after every plan; and seven steps of partial acceptance and
rollback from cursors just below 1,536 tokens keep the serial chain's cache
bytes and cursors and ignore rejected tokens in cut windows. An independent
review confirmed the independence for every window start up to 33,824
tokens and that the only bits that differ from 0.21.8 are those of the rows
before the boundary in a cut window. Everywhere else the partition and the bits are
those of 0.21.8, and so is the cost: −2 to +1% at 512 to 32,768 tokens with
one and four lanes (two passes of the median of 20,
`$V/splash-gap/attnverify/align/batch1.log`). A cut window costs 0.142
against 0.099 ms per Qwen3.8 layer at four lanes and is 7 of every 768 window
starts. An absolute partition whose runs grow with their start page was
built first and rejected: it gives the same independence but 53 to 105
splits against 24 at 8,192 and 32,768 tokens, 3 to 36% slower there.

In situ, every combined-candidate switch on, exact four-agent Q5
(`$V/splash-gap/attnverify/align/runs`). With the any-width switches of #825
(hf2q `68d90b67` plus this change, mlx-native `807ac72`, binary `ec64fe57`):
seven Qwen3.6 runs, all `CalibrationPass`, all with the same transcripts for
all four agents (5,322 completion tokens), and two Qwen3.8 runs with
identical transcripts (5,198). The same switches on the 0.21.8 tile gave two
transcripts in eight Qwen3.6 runs (six and two, see above). At that rate
seven equal runs would happen by chance about 13% of the time, so this
supports, and does not prove, that the tile was the remaining source; the
proof is the model-free tests. Against the 0.21.8 runs every agent's text
first differs after its history has crossed 1,536 tokens, never before, as
the changed bits of tokens in cut windows allow. The four-lane cycle medians
are 31.6 to 33.9 ms for Qwen3.6 (33.1 to 35.0 on the 0.21.8 tile) and 68.9
and 69.9 ms for Qwen3.8 (72.0 and 72.4), runs of different sessions.

Without the any-width switches (hf2q `991abc2e`, mlx-native `b76b42b`, binary
`0a23d8a5`, before #825 merged): two Qwen3.8 runs with identical transcripts
(5,330 completion tokens). Of six Qwen3.6 runs five are `CalibrationPass`
and one is `ClientFailed` (an agent left the six-step workflow; two of the
six 0.21.8 candidate runs were `ClientFailed` too, one with this error); no
two of the five have the same transcripts. That variation is the
width-dependent route #825 removes, not the defect fixed here: it starts in
the first assistant message, at 687 prompt tokens and under 200 generated
ones, below the first page boundary where the split length steps, and the
0.21.8 control arm with the tile off shows it too (up to four variants of an
agent's first message in six runs). No same-session control with the 0.21.8
tile was run for either set.

Model free the route takes 0.051, 0.116 and 0.331 ms per Qwen3.8 layer at
four lanes (0.033, 0.048, 0.103 ms at one) and 0.046, 0.083 and 0.195 ms per
Qwen3.6 layer, including the K/V encodes and the gate: 0.70 to 0.97 times
Splash's store, split and reduce at four lanes and 1.2 to 1.5 times at one
lane, where six dispatches against three and the extra barrier decide.
Splash's numbers are from its own tool and warm-up; they were not
re-measured under this bench's filler protocol.

In situ, same binary (sha256 `a57c958d…`: this change before its rebase
onto `eea0fa1a` and before the review fixes above, which leave the
arithmetic unchanged; prepared FFN and projections on), one exact Q5 run per
cell, all `CalibrationPass` (`$V/splash-gap/attnverify/runs`):

| Qwen3.8, per cycle | 1 lane | 2 lanes | 3 lanes | 4 lanes |
|---|---:|---:|---:|---:|
| Attention category off, GPU ms | 9.45 | 16.12 | 23.27 | 25.95 |
| Attention plus KV write on, GPU ms | 1.14 | 1.24 | 2.10 | 2.02 |
| Cycle total off, ms | 106.88 | 104.39 | 120.11 | 130.08 |
| Cycle total on, ms (two runs) | 100.59, 89.52 | 102.55, 100.97 | 105.83, 103.35 | 109.87, 106.52 |
| Dispatches off / on | 6,084 / 4,884 | 8,201 / 6,233 | 10,462 / 7,726 | 12,579 / 9,219 |

| Qwen3.6, per cycle | 1 lane | 2 lanes | 3 lanes | 4 lanes |
|---|---:|---:|---:|---:|
| Attention category off, GPU ms | 4.42 | 8.27 | 10.54 | 11.48 |
| Attention plus KV write on, GPU ms | 0.63 | 0.75 | 0.91 | 0.81 |
| Cycle total off, ms | 36.74 | 58.75 | 51.63 | 58.23 |
| Cycle total on, ms (two runs) | 34.58, 34.51 | 53.03, 52.59 | 42.91, 42.18 | 45.98, 45.79 |
| Dispatches off / on | 3,324 / 2,814 | 4,751 / 3,681 | 7,079 / 5,369 | 8,407 / 6,217 |

The category rows are explicit stage profiles (one profiled run per arm);
the totals are unprofiled medians of the DEBUG stage timing. Splash's
attention kernels take 1.71 ms per Qwen3.8 cycle at a 512-token history and
2.68 ms at 2,048, and 0.72 ms per Qwen3.6 cycle. The Qwen3.8 arms emit
different text (5,206 completion tokens in both runs with the switch, 5,237
without; makespan 62.5 and 60.5 against 68.7 seconds), and the Qwen3.6 runs
differ from run to run in both arms, so makespan is one draw per run and the
per-cycle medians are the comparison. The switch is off by default pending
the combined-candidate quality campaign. It needs an mlx-native release that
contains `tq_hb_verify_tile`.

#### Combined candidate after the stage work

The staged draft command, the live-tile MoE route, the tiled verify attention
and the prepared prefill route were built into one binary (`d5f48657`, mlx-native
0.21.8 sources) and measured together, every switch on against the previous
best configuration (`HF2Q_REPACKED_GGUF_M32=1` with `_PROJ=1`) of the same
binary. `$E` is `/Users/robert/.local/state/hf2q/campaigns`.

Qwen3.8, six balanced blocks (`$E/qwen38-int-all-d5-v1`): 12 of 12
`CalibrationPass`. Client makespan −28.18% (95% CI 19.83–35.66%; 47.37
against 63.36 seconds) and the four-lane cycle −28.77% (95% CI 20.13–36.48%).
The host was loaded during this campaign; on a quiet host one run per arm read
41.45 against 55.65 seconds and an 80.3 against 100.5 ms four-lane cycle. The
text differs (5,462 against 5,237 completion tokens, each arm repeating its own
text exactly). Peak footprint 70.59 against 70.08 GiB.

Qwen3.6, twenty interleaved rounds (`$E/qwen36-int-isolate-v1`, arms `all`
and `control`): 19 of 20 candidate runs and 18 of 20 control runs passed; all
three failures are the workflow oracle's "initial six-step workflow not
followed". Over the 17 rounds where both arms passed:

| Metric | All switches | Previous best | Ratio (95% CI) |
|---|---:|---:|---|
| Four-lane cycle (ms) | 33.57 | 51.04 | 0.657 (0.652–0.661) |
| Client makespan (s) | 21.75 | 31.31 | 0.695 (0.659–0.732) |
| Total prefill per run (s) | 5.27 | 6.44 | 0.815 (0.776–0.856) |
| Accepted tokens per lane-cycle | 3.964 | 4.121 | 0.959 (0.950–0.968) |

A first six-block Qwen3.6 campaign (`$E/qwen36-int-all-d5-v1`) was invalid:
two of six candidate runs failed the same oracle. Five runs of each switch
alone then gave 5/5 (staged draft), 5/5 (attention tile), 4/5 (prepared
prefill) and 4/5 (live tiles), and the twenty-round comparison above shows the
control failing as often as the candidate. The failing transcripts are
coherent tool calls that take a wrong turn, not corrupted output. Qwen3.6 on
this workload therefore fails the oracle in roughly one run in twelve with or
without these routes, which a release campaign has to budget for.

Two things the combined measurement shows that the single-switch ones did
not:

- The gains are not additive. On Qwen3.6 the staged draft alone and the live
  tiles alone left the four-lane cycle at the control's 51–52 ms, and the
  attention tile alone brought it to 42 ms; together they reach 33.6 ms. A
  route must be judged on the combined candidate with one switch removed, not
  alone.
- Draft acceptance on Qwen3.6 falls by about 4% with every switch on (live
  tiles alone read 4.01 against 4.12 accepted tokens per lane-cycle in four
  runs). It is paid for many times over by the shorter cycle, but it is a
  cost of the changed target arithmetic and remains open.

One profiled run of each combined candidate gives the four-lane GPU time by
category, which is where the remaining distance to Splash (50.6 ms per
Qwen3.8 cycle, 20.8 ms per Qwen3.6 cycle on its own synthetic routing) sits:

| Category (GPU ms) | Qwen3.8 | Splash 3.8 | Qwen3.6 | Splash 3.6 |
|---|---:|---:|---:|---:|
| Draft | 5.5 | 5.98 | 3.6 | 3.36 |
| Attention and KV write | 2.4 | 4.7 | 1.2 | 1.8 |
| FFN or MoE | 29.1 | 24.0 | 21.5 | 9.7 |
| Projections and DeltaNet | 22.3 | 12.8 | 8.6 | 4.7 |
| Head | 4.5 | 1.95 | 1.9 | 0.86 |

The draft and attention stages now meet Splash's. The Qwen3.8 cycle's wall
time (80 ms) exceeds its GPU categories (66 ms) by about 14 ms of host and
synchronization time over 9,063 dispatches. Profiling serializes dispatches,
so the Qwen3.6 MoE figure overstates its unprofiled cost, and Splash's 9.7 ms
is for lanes that share experts; on the recorded routes of this workload its
own kernels take 13.6 ms.

#### Cycle scheduling: wall time against GPU time

A per-command-buffer timeline (`HF2Q_DFLASH2_CYCLE_TIMELINE`, mlx-native
`command_timeline`: host encode, commit and wait times beside Metal's
`kernelStartTime`/`GPUStartTime`/`GPUEndTime` on one clock, no commit turned
into a wait) replaces the estimate above with a measurement. One four-lane
Qwen3.8 cycle of the combined candidate: 73.6 ms wall, 64.3 ms of GPU
execution, 9.3 ms of GPU idle. The 16 layer commands (9,063 dispatches,
1,683 barriers) are encoded in 11.8 ms and all committed by 22.6 ms, 3 to
41 ms before the GPU reaches them: dispatch count costs worker CPU, not wall
time. The idle is 6.9 ms between the draft command and the target's first
command, where the worker allocates and zero-fills the verify step's
DeltaNet replay storage (188 MB per step; 91 MB and 2.6 ms on Qwen3.6), and
2.2 ms after the output head (readback, host plan, replay encode, an idle
GPU's 0.5 ms start latency), plus two waited commit commands.

Two Class A switches, off by default, remove most of it:
`HF2Q_QWEN_VERIFY_REPLAY_REUSE=1` keeps the replay storage between steps
(almost all of the gain; the retained storage is outside memory admission)
and `HF2Q_DFLASH2_CONTEXT_FIRST=1` submits the draft-context commit before
the replay is encoded, so the two commit commands overlap on the GPU (the
step still waits for both; 0.4–0.5 ms off the cycle's tail). With both on,
GPU idle per
four-lane cycle falls to 2.7 ms on Qwen3.8 and 1.7 ms on Qwen3.6. The same
timelines explain the non-additive single-switch readings: GPU time per
command class adds exactly (live tiles −3.3 ms, attention tile −11.3 ms,
both −14.7 ms of layer time); half of the staged draft's 5.0 ms was hidden
behind the replay-storage allocation; and the single-switch arms had been
measured in a block that ran about 3 ms slower than the control block. The
measured tables, commands and what remains (host acceptance between the
target and its commit commands, which Splash runs as one command) are in
`splash-gap/CycleHost.md` of the campaign state.

#### Splash mixer differential (projections and DeltaNet)

The combined candidate's profiled `projections` + `delta_conv` + `delta_gdn`
categories read 23.2 ms per four-lane Qwen3.8 cycle and 8.4 ms per Qwen3.6
cycle. They hold more than Splash's "DeltaNet" stage (12.8 and 4.7 ms): the
pre-mixer norm of every layer and the attention projections, head norms and
rotary encoding are in them too. Splash spends 16.5 and 6.0 ms on the same
work (its decode profile, `splash-kernel-bench/REPORT.md` section 4).

A per-dispatch-group profile and two model-free benches (mlx-native
`bench_qwen_delta_proj`, `bench_qwen_delta_core`; DRAM-cold weights and
state, samples behind a filler command buffer) place the difference:

- Product for product the prepared kernel is at or below Splash's (qkv
  0.072 against 0.083 ms, z gate 0.057 against 0.072, output 0.046 against
  0.062). But each prepared product with several K splits is a split-grid
  pass, a barrier and a reduce pass, so the three input products of a layer
  ran as four serialized stages: 0.177 ms per Qwen3.8 DeltaNet layer against
  0.129 for Splash's one fused dispatch
  (`runtime/metal/kernels/shared/gguf_linear.metal:197-232`, in-kernel split
  reduce `common/split_reduce.h`).
- Between the projections a DeltaNet layer issued 148 dispatches in five
  stages at four lanes (two convolution dispatches, a raw-row copy and the
  recurrence per window; Q norm, K norm, V copy and query scale per row):
  0.098 ms per Qwen3.8 layer against 0.073 for Splash's one
  `verify_gdn_fused` (`runtime/metal/kernels/decode/gdn.metal`), which hands
  its phases their operands in threadgroup memory.
- Two F32 to BF16 cast stages per layer, which Splash's BF16 residual stream
  does not have.

`HF2Q_QWEN_VERIFY_MIXER_FUSED=1` (default off) adopts the structure and keeps
hf2q's arithmetic. The input products of a layer are issued as one wave, one
kernel pass per product with no reduce pass or barrier
(`dispatch_repacked_gguf_mm_bf16_input_one_pass`, the bits of the default
entry): 0.130 ms per Qwen3.8 layer with Q4_K qkv (Splash 0.129), 0.153 with
Q6_K (0.148), 0.048 per Qwen3.6 layer (0.041). When every product of a layer
is prepared the pre-mixer norm writes their BF16 rows
(`dispatch_rms_norm_f32_to_bf16`). The DeltaNet core of every window is one
dispatch (`gdn_window_fused`) that writes the output projection's BF16 rows:
0.068 ms per Qwen3.8 layer and 0.054 per Qwen3.6 layer at four lanes. Every
kernel repeats the stock kernels' operations in their order, including the
128-thread tree of the norm reductions, and the route is Class A: rows,
captures, state and retained replay rows are bitwise the stock route's. That
equality is qualified by test on the Apple M5 Max, not by the source alone:
outside the pinned norm sums and scales the fused kernel's phases keep
Metal's fast-math flags, as the stock kernels do, so mlx-native
`tests/test_gdn_window_fused.rs` is the gate for any other GPU family. The
core serves head dimension 128, two to nine convolution taps and windows of
at most eight rows within the device's threadgroup limits; other layers keep
the stock chain, and the first use and first refusal are logged. A DeltaNet
layer is 7 dispatches in five stages instead of 159 in 14.

Exact four-agent runs, every other switch of the combined candidate on, same
binary with the switch on and off, all `CalibrationPass`
(`$V/splash-gap/deltaproj/runs`; two runs per arm; first the binary
without the BF16 norm on a quiet host, then the final binary on a host that
ran both arms about 22% slower):

| Four lanes | Qwen3.8 off | Qwen3.8 on | Qwen3.6 off | Qwen3.6 on |
|---|---:|---:|---:|---:|
| Cycle, first binary (ms) | 75.1 / 75.6 | 70.6 / 70.9 | 33.3 / 33.9 | 31.2 / 31.7 |
| Cycle, final binary (ms) | 92.6 / 92.3 | 86.0 / 88.0 | 36.6 / 36.4 | 35.6 / 34.7 |
| Dispatches per cycle | 9,063 | 1,703 | 6,316 | 1,716 |
| Barriers per cycle | 1,683 | 1,203 | 1,146 | 846 |
| Profiled projections + DeltaNet (GPU ms, final binary) | 23.8 | 17.0 | 8.2 | 6.9 |

Qwen3.8 emitted the same text in every run of both arms (5,462 completion
tokens, 5.3191 accepted rows per lane-cycle), as a bitwise route must. Qwen3.6
text varies from run to run in both arms; over six runs per arm its accepted
rows per lane-cycle read 3.87-4.34 with the switch and 4.04-4.21 without,
which these runs cannot separate. No paired campaign has been run.

Left of the like-for-like distance (0.5 ms on Qwen3.8, 0.9 ms on Qwen3.6,
profiled, with the profile's own command buffer per category boundary on the
hf2q side): the attention layers' Q/gate de-interleave, head norms and rotary
stages (0.3 against 0.15 ms), one cast stage per attention layer, and on
Qwen3.6 the stock projections of 8- and 16-row steps. (Those runs predate
`HF2Q_REPACKED_GGUF_ANY_WIDTH`. With it the prepared routes serve every tile
width of a DFlash2 verify step and the wave and the BF16 norm follow the
same width rule; tests hold the step bitwise equal to the stock chain at
widths 4, 8, 16 and 32 on the synthetic MoE layout and on the official
Qwen3.6 artifact. Its effect on the Qwen3.6 cycle has not been measured.)
Outside these
categories the DeltaNet replay command of a partially accepted window still
costs 2.1 ms GPU per Qwen3.8 cycle and 1.2 ms per Qwen3.6 cycle plus a host
wait; Splash folds the retained rows into the state at the lane's next step
instead (`gdn.metal:22-25`).

#### Second combined candidate and the end-to-end comparison with Splash

The second round of stage work (prepared routes at every verify width with a
shared head, fused MoE stages, replay-storage reuse and context-first commit,
the fused DeltaNet mixer, the alignment-independent attention tile, the cue
fold and lane-pass capture ingest; mlx-native 0.21.9) was built into one
binary (`ff578ce3`) and measured with every switch on against the first
combined candidate's switch set on the same binary. `$E` is
`/Users/robert/.local/state/hf2q/campaigns`. Each block's two arms ran back
to back with nothing else on the host.

Qwen3.8, six balanced blocks (`$E/qwen38-w2-all-d5-v1`), 12 of 12
`CalibrationPass`:

| Metric | All switches | First combined candidate | Reduction (95% CI) |
|---|---:|---:|---|
| Client makespan (s) | 32.75 | 37.91 | 13.94% (13.05–14.81%) |
| Four-lane cycle (ms) | 60.18 | 71.92 | 17.15% (16.18–18.10%) |
| Total prefill per run (s) | 12.03 | 12.69 | 5.47% (4.63–6.30%) |

Each arm repeats its own text exactly (5,198 and 5,330 completion tokens).
Peak footprint 71.51 against 70.49 GiB.

Qwen3.6, ten balanced blocks (`$E/qwen36-w2-all-d5-v1`): 10 of 10 candidate
runs and 9 of 10 control runs passed (the control failure is the workflow
oracle's "initial six-step workflow not followed"). Over the nine blocks
where both passed:

| Metric | All switches | First combined candidate | Ratio (95% CI) |
|---|---:|---:|---|
| Client makespan (s) | 18.85 | 22.82 | 0.870 (0.803–0.943) |
| Four-lane cycle (ms) | 27.27 | 32.66 | 0.853 (0.834–0.873) |
| Total prefill per run (s) | 4.99 | 5.28 | 0.997 (0.936–1.062) |
| Accepted tokens per lane-cycle | 3.914 | 3.987 | 0.978 (0.960–0.997) |

The Qwen3.6 candidate emitted the same 5,322 completion tokens in all ten
runs, and in seven further runs of the attention-tile owner; the control
emitted between 3,771 and 6,581. Qwen3.6 output used to vary from run to run
because a target row's bits depended on the verify tile width (stock kernels
at 8 and 16 rows, prepared at 32) and on the alignment of the verify window
(the attention tile's split length), and both follow a verify budget fed by
wall-clock time. With the prepared routes at every width and the tile's split
length a function of the token's own position, a row's bits no longer depend
on either. Rows decoded outside a DFlash2 verify step still use other
kernels.

A run's GPU speed follows the GPU clock. Over eleven Qwen3.8 runs with a
100 ms sampler of the GPU performance state beside them, the 16 layer
commands of a cycle took 50.3–53.2 ms in runs at 1,470–1,506 MHz and
60.9–63.1 ms in runs at 1,133–1,189 MHz (r = 0.97 across run means; r =
0.83–0.99 within a run). Builds and other work on the host lower the clock.
Two runs minutes apart can therefore differ by 13 ms per cycle with identical
commands, which is why earlier single-run comparisons disagreed and why the
campaigns above ran their arms back to back on an otherwise idle host. A
stage gain under about 5% needs the behaviour alternated inside one process
or the sampler beside every run.

The end-to-end comparison on the Qwen3.6 workload, from the retained
artifacts of Splash's valid arm (17 runs, median 16.42 s) and of the first
combined candidate (19 runs, median 21.98 s):

| Part | hf2q | Splash | Difference |
|---|---:|---:|---:|
| Prompt prefill, GPU wall per run (s) | 5.27 | 2.78 | +2.5 |
| Decode and host time while requests are in flight (s) | 16.66 | 13.58 | +3.1 |
| Prefill cost per row (ms) | 0.709 | 0.241 | |
| Tokens per lane-cycle | 4.00 | 4.23 | |

Splash prefills several lanes' rows in one batch (about 383 rows and 95 ms
per batch, 30 batches per run against hf2q's 56 chunks) and shares the prefix
the four agents' first prompts have in common. Its served cycles cost about
15, 20, 25 and 28 ms at one to four lanes, more than its synthetic profile's
20.8 ms at four lanes suggests for the smaller cohorts; the second combined
candidate's cycles are 16.5, 20.9, 25.5 and 27.0 ms. Splash did more work in
that arm (69% of its agents needed the repair turns against 22% of hf2q's),
so at equal work the gap was larger than the medians show. After this round
the remaining distance on Qwen3.6 (18.85 against 16.42 s) is prompt prefill.
On Qwen3.8, where no valid Splash end-to-end run exists, prefill is 12.0 s
of a 32.8 s run and the four-lane cycle is 60.2 ms against Splash's 50.6 ms
profile.

#### Cycle tail: draft prefetch and decision candidates

After the stage work the four-lane Qwen3.8 cycle (59.7–60.1 ms begin to
begin on the second combined candidate) spends 4.5–4.6 ms between the output
head's GPU end and the GPU start of the next cycle's draft; Qwen3.6 spends
2.9 ms of 28.3. Splash has 0.46 ms of GPU work there (acceptance and the
draft-context update are dispatches of its one command). In hf2q's tail the
host copies 32 rows of 248,320 logits out of the head's buffer (0.50 ms),
samples them (0.77 ms; 0.48 on Qwen3.6), pays a residency-set commit for the
step's released layer outputs (0.40 ms on Qwen3.8), and then runs the
draft-context commit and the DeltaNet replay in front of the next draft.
This workload always settles acceptance on the host: every request carries a
repetition penalty, a thinking budget and the lazy tool grammar.

Two Class A switches, off by default, remove the tail without changing a
decision:

- `HF2Q_DFLASH2_DRAFT_PREFETCH=1`. Once the plan exists the step knows every
  input of the next step's draft (anchors, positions, row limits, lane
  order). It submits the draft-context commit, which reads its own copy of
  the step's positions so the next positions can be written at once; commits
  the replay behind it; commits the next draft behind both; and only then
  waits for the replay and the context commit. The draft's ring reads are
  ordered behind the context commit's ring writes by Metal's synchronization
  of tracked resources (the rings are default-tracked shared buffers bound
  directly to both commands), not by the queue as such: the replay shares no
  resource with either and overlaps them. The next step's draft call
  adopts the committed command when its inputs are exactly the prefetched
  ones and nothing wrote the rings in between; a lane that joined or left, a
  prefill ingest or a cleared lane makes it wait for the command, drop its
  outputs and draft again. A step that runs while another slot is in prefill
  does not prefetch. The order of the three commits matters: a wait on a
  command buffer returns only when every command committed before it on the
  queue has completed, so a replay committed behind the draft is waited for
  until the draft ends (measured: the cycle then gains 1 ms instead of 4).
  The step's layer outputs are released after the commit settles.
- `HF2Q_DFLASH2_GPU_DECISIONS=1`. For a lane at temperature zero whose
  sampler is a repetition penalty followed by an argmax, the penalty changes
  at most 64 logits of a row. The output-head command gathers those logits
  (the lane's repetition window after the window's earlier drafts, assembled
  on the GPU from the lane's base window and the token tile), masks them and
  runs the stock argmax over the row; the host finishes each decision from at
  most 65 candidates with `apply_repetition_penalty` and `sample_greedy`
  themselves and hands it to the same decision path a greedy lane uses
  (forced thinking-budget tokens, grammar advance, tool trigger and terminal
  handling are unchanged host code). The GPU copies and compares logits; it
  performs no arithmetic on them. A row the candidates cannot decide (the
  tool grammar is constraining it, a logit is not finite, the history is not
  the one the kernel assembled) is sampled on the host from that row's own
  logits, restored bit for bit; a request with logit bias, top-k, top-p,
  min-p, a seed or logprobs keeps the host route for its whole window.
  `check` runs both routes for every window and compares the plans.

Rejected for this step: acceptance and the context commit inside the head's
command (it needs the penalty multiply on the GPU and a second context commit
for lanes the GPU cannot decide, for a further 0.2–0.35 ms), and a host-step
spin (start latency only; mlx-native has no shared-event split).

Gates. `draft_prefetch_tests`: on a synthetic checkpoint, both draft routes
and both entry points, 40 steps past a ring wrap leave every draft output and
the rings bitwise the stock order's with 39 of 40 prefetches adopted; five
kinds of mismatching next step, a cleared lane and a prefill ingest each drop
the command and redraft with the stock bits; a step that fails after its
submissions leaves no prefetched draft and clears its lanes.
`dflash2_submitted_commit_*`: one to four lanes at every accept count, stock
and tiled attention, bitwise the waited commit. `verify_decision` tests: the
kernels against `sampler_pure::sample_token` at vocabularies up to the served
248,320 under penalties 1.0, 1.05, 1.3 and 0.8, with history tokens on top,
exact ties on either side, signed zeros, duplicate history tokens and
non-finite logits; identical digests plain, under shader validation, with the
debug layer and from runtime source. `dflash2_scheduler::decision_tests`: 600
verify windows with penalties, thinking-budget closes forced inside the
window, the lazy grammar waiting, triggered inside the window and active,
planned by both routes with equal plans.

In situ, Qwen3.8 (the second combined candidate with and without the two
switches, one binary, eight exact four-agent runs interleaved inside one
measurement ticket, all passing with one transcript): the time from the
head's GPU end to the next draft's GPU start falls from 4.63 to 1.00 ms and
GPU idle per four-lane cycle from 2.7 to 0.5 ms; the four-lane cycle falls
from 60.1 to 58.2–58.4 ms (−1.85 ms, 3.1%) and four-agent makespan from 33.0
to 32.5 seconds (−1.5%); peak footprint is unchanged. Alternating the
switches in blocks of 16 cycles inside one run gives −2.75 ms for both
(se 0.35), −1.93 ms for the prefetch alone and −1.63 ms for the decision
candidates alone. Less than the tail's 3.6 ms reaches the cycle for two
reasons. The replay now runs beside the next draft and slows it by 0.8–1.0 ms
(a cycle with no partially accepted lane is 56.5–57.4 ms head to head, one
with four is 59.2–59.7 ms): the replay still costs about 1.3 ms per four-lane
cycle, which folding the retained rows into the next step's DeltaNet kernel,
as Splash does, would remove. And the GPU's performance state falls when its
idle goes: the layer commands take 0.5 ms longer in whole runs with the
switches, which the 16-cycle blocks do not show.

Qwen3.6, ten exact runs interleaved in one ticket (all passing, one
transcript): tail 2.84 to 0.88 ms, four-lane cycle 27.34 to 26.28 ms (−3.9%),
one-lane cycle 16.83 to 16.31 ms, makespan 18.79 to 18.36 seconds (−2.3%;
four runs per arm, every run with the switches below every run without),
peak footprint unchanged. With `check`, 5,646 served verify windows on the
two models were planned by both routes with equal plans.

Equal-clock comparison with Splash. Splash's `decode-profile` and the hf2q
server were run in the same tickets with a GPU performance-state sampler
beside each. Run after a pause, Splash's tool reads 51.59 ms per fused
four-lane cycle at 1617 MHz (its attributed stage table at 1620 MHz); the
hf2q server ran minutes later at 1394 MHz (1342 MHz with the switches). The
difference is not the engines. The GPU leaves its top state after a few
seconds of sustained load: Splash's tool with 160 fused cycles per width
instead of 8 holds 1620 MHz for 3.5 seconds and then runs at 1200–1300 MHz,
with nothing else on the host and the GPU never idle, and its cycles
lengthen with it (three lanes: 8.3 + 66,968 / MHz ms, 49.1 ms at 1620 MHz,
61.1 ms after ten seconds); run straight after an hf2q run it reads 55.17 ms
at 1483 MHz. hf2q's served runs slide the same way from their first cycle
to their last (1511 to 1277 MHz) and their command classes follow the
frequency (16 layer commands: 10.3 + 51,088 / MHz ms, r = 0.96). A 140 W
adapter in place of a 65 W one with the battery supplying did not raise the
served state. Compared at equal frequency, by interpolation on both sides
between 1306 and 1620 MHz, the hf2q cycle with the two switches is within
1.5 ms of Splash's measured points on either side (51.7 against 51.6 ms at
1620 MHz, 54.7 against 55.2 at 1483) and without them 1 to 4 ms above: the
nine milliseconds between hf2q's 60.1 ms and Splash's published 50.6 ms were
four of tail and five of comparing a served run in the settled state with a
tool's first seconds. By stage at equal clock (one category profile scaled
by the fit): hf2q is faster in the output projections (−1.6 ms) and the
DeltaNet core (−0.5 ms) and slower in the FFN block's surrounding stages
(+0.9 ms: casts, SiLU, reduce and residual as separate dispatches behind six
barriers where Splash has two dispatches with epilogues) and on the
attention side (+0.3 to +0.6 ms). Whether the envelope is thermal or
electrical was not measured.

### D5: performance qualification

Use the exact Q4 target GGUF, four-agent workload, prompts, sampling settings,
context, quality oracle, and hardware. Report medians and confidence intervals
for:

- accepted-plus-bonus tokens per lane-cycle;
- proposal acceptance by depth and grammar/non-grammar class;
- draft, selector, target, sampling/accept, commit, and synchronization wall
  time;
- total GPU busy time and bubble;
- cached-token counts, prefill/decode rates, TTFT, SSE cadence, completion rate,
  and exact quality verdicts.

Qwen3.8's mechanism gate is at least 5.304 accepted-plus-bonus tokens per
lane-cycle on the matched workload without a quality or task regression.
It must also finish at least 5% sooner than v0.1.29, with the candidate/stock
95% CI upper ratio at most 0.95, and reach statistical parity with every valid
same-artifact peer, with each candidate/peer upper ratio at most 1.00.
No retained Splash Qwen3.8 run passes the task gate, so it supplies no valid
end-to-end performance denominator; ADR-063's mechanism substitution remains
in force for that family.

The 17 completed Splash runs with median 16.420344 seconds belong to
`q4-qwen36-dflash2-release-gate-v2`, not Qwen3.8. They establish the required
Splash comparator for Qwen3.6: its candidate/Splash 95% CI upper ratio must
be at most 1.00. The Qwen3.8 mechanism gate cannot substitute for that valid
Qwen3.6 comparator or for any family's other valid same-artifact peer.

Every additional family preregisters its own mechanism and end-to-end gates
against spec-off, stock and valid peers before seeing candidate results. It
must establish a net gain; merely loading a drafter or accepting one token does
not qualify the adapter.

### D6: promotion

- Run the realistic multi-turn coding, tool-call, tool-result, unary/SSE, prefix
  reuse, cancellation, and clean-shutdown gate required by the serving contract.
- Run Q5 on the release-candidate commit with the family-neutral DFlash2 policy
  unset and prove the startup audit selects DFlash2 only for a qualified,
  identity-matched target/draft pair.
- Run the global kill-switch control with `HF2Q_DFLASH2=0` and prove each
  family's prior promoted route remains correct.
- Remove superseded family-specific DFlash code and temporary fixtures only
  after that family's cutover.

#### D6 policy as implemented (Qwen3.8 and Qwen3.6)

`HF2Q_DFLASH2` is a three-state policy resolved once per model load in
`serve::load_engine`, from the GGUF header, before any weight is read
(`src/serve/dflash2_policy.rs`).

- **Unset: auto.** On a SlotAware worker of one to four slots DFlash2 is
  selected for a target in either of two classes. A *pinned pair* carries
  hf2q provenance and the `target_source_sha256` of a pinned Qwen descriptor
  of its architecture. A target with the *same architecture and shapes* as a
  pinned descriptor's target has its architecture string, hidden size, layer
  count, vocabulary size, tokenizer identity, attention and DeltaNet head
  counts and dimensions, FFN width, MoE expert count, routed experts per
  token and expert widths, and embedding and output-head tensor shapes;
  hf2q provenance is not required. The descriptor is chosen from the target;
  the drafter is then located: both explicit path variables, else the
  Hugging Face cache snapshot of the descriptor's repository at its pinned
  revision, else a download at that revision into that cache, verified by
  the configuration digest, the weight length and the weight digest. The
  checkpoint is bound to the target by the adapter's own shape check
  (`DFlash2Config::validate_target`, what `DFlash2TargetFamily::validate`
  runs), on the header before the load and on the loaded model at
  provisioning. Every D0 checkpoint identity check still runs when the
  checkpoint is provisioned; the target-source identity check runs for a
  pinned pair. Auto never fails startup: a target in neither class, a
  drafter that cannot be obtained or bound, too little memory and a load
  that fails with DFlash2 selected all end on the prior route with the
  reason logged.
- **Static MoE boundary.** Auto qualifies a `qwen35moe` target only when its
  batched MoE verification route is admitted for the artifact's storage.
  Without it a 32-row verify step runs every row through the per-row expert
  path, which is not a qualified fast path. This is a qualification rule
  read from the GGUF header before the load. It is not the run-time
  drop-back the operator rejected: nothing monitors acceptance, output or
  timing, no route changes after the load, and the rule carries no
  threshold. Same-base fine-tunes whose storage admits the fast graph keep
  the automatic drafter. Admitted is what the loader does with the storage,
  not what the switch says: the loader keeps a layer whose router, gate
  logit or shared expert is not in the batched route's storage on the
  per-row route whoever turned `HF2Q_QWEN_MOE_BATCHED` on, so setting the
  switch by hand on such an artifact does not qualify it.
- **`0`: kill switch.** No drafter is located and no profile applies. Every
  switch of the route profile resolves from the environment alone, so the
  server runs what it ran before this policy, for every family.
- **`1`: required.** A target in neither class, or a drafter that cannot be
  obtained or bound, fails startup. It accepts both classes, and it serves
  the drafter on a MoE target without its batched route, with a WARN line
  that the route is unqualified and was measured slower. This is the only
  override of the boundary. The staged expert rows stay withheld there: they
  run inside the batched layer and would fail the load.

This supersedes the earlier rule that admitted the exact source identity
only (operator decision, 2026-11-12: fine-tunes of a qualified model get the
fast path by default, with no automatic drop-back; the off switch is
`HF2Q_DFLASH2=0`). The drafter proposes and the target decides every emitted
token, so a drafter that fits a fine-tune less well costs acceptance, never
correctness. Measured on the launcher-default models, exact four-agent Q5,
two runs per arm, prior route against the default:

| Launcher default | Class and routes under auto | Prior route | Drafter | Accepted per lane-cycle |
|---|---|---:|---:|---:|
| Qwen3.8 abliterated SFT (Q4_K_M, hf2q) | same shapes; full profile | 59.42 / 59.15 s | 29.56 / 27.10 s | 5.318 (vanilla pair 5.254) |
| Qwen3.6 APEX-Q5_K_M (external) | same shapes; batched, staged and live-tile MoE and prepared head withheld | 39.46 / 39.57 s | 48.58 / 49.13 s | 4.43 / 4.53 (vanilla pair 3.93) |

Every run is a `CalibrationPass`. The dense fine-tune halves its makespan.
The Q5_K MoE artifact is slower with the drafter: its shared expert is
Q5_K, the batched MoE route takes Q4_K and Q6_K, and a 32-row verify step
on the per-row expert path costs 132 ms per four-lane cycle. Hence the
static MoE boundary above: under auto that artifact keeps its prior route,
and its audit line says the batched route was not admitted.

Each pinned Qwen descriptor has one route profile, a row per switch in one
table (`src/inference/spec_decode/dflash2/route_profile.rs`): the draft
AF4G64 and staged command, GPU target tokens (Qwen3.6), the context-first
commit, the prepared FFN, projections, head, any-width and shared-head
routes, tiled verify attention, the fused verify mixer, the replay reuse,
prepared prefill with the cue fold and, for Qwen3.6, the batched, staged-M8
and live-tile MoE routes. A switch resolves to its environment value when one
is set, else to the profile's value when DFlash2 is selected for that model
and the row is not withheld, else to its own default. The profile is applied
switch by switch: a row is withheld when the switch it requires is off, when
its kernels would refuse the artifact's storage (for example Q5_K, where the
prepared and staged routes take Q4_K and Q6_K only), or when the memory rule
dropped it; the drafter serves with the rows that are left and the audit
line gives each reason. The resolved set belongs to the model: its load
thread and its worker thread bind it, so a second model in the same process,
or any path outside `hf2q serve`, reads the environment alone. The prepared
prefill and the MoE switches, which act on every artifact of a family, are
therefore on by default only for a selected target.

A row is also withheld when the host cannot run its kernels: the policy
asks `inference::route_host` (the Metal tensor API is absent on macOS before
26 and on GPUs outside the Metal 4 family) for every row before turning it
on. A switch set on by hand on such a host fails startup with the host's
text; a host that could not be asked at all keeps the prior route under
auto and fails a required DFlash2. Whether a value sets a switch on is
decided by the grammar of that switch's own reader, which the host table
records per row (exactly `1` for the prepared verify routes; `1`, `on` or
`true` for the prepared prefill and the MoE routes). The same test settles
the planes a route keeps and the row another row requires, so a value the
reader leaves off (`HF2Q_REPACKED_GGUF_M32=on`) selects no kernel, counts no
plane and never refuses the stock load it leaves.

A memory rule runs before selection: headroom is physical memory minus what
the engines already resident in the process have committed, the model file,
the configured KV budget and 8 GiB; the drafter costs its file size, the
prepared planes the native layout's bytes for the weights they cover (two
output heads when the shared head is off), and a route's retained scratch
its declared figure (none today). When the planes do not fit, the
plane-carrying rows are withheld and DFlash2 is kept if the drafter fits;
otherwise the prior route is kept. Every `load_engine` call decides against
a process ledger (`serve::resident_memory`) and reserves what it decided in
one transaction, before it allocates, so concurrent loads through one pool
or several are never decided against the same total. The reservation is
owned by whoever owns the model's buffers: the loading thread, then the
worker thread, to which `Engine::spawn*` hands it with the model. Only the
worker's teardown releases it (`WorkerTeardown`): the model is dropped, the
thread's GPU weight cache is dropped, the thread terminates (a second
thread joins it, so every thread-local it had is destroyed), the cache's
device is flushed (the staged residency-set removals are committed and the
commands committed on its queue are waited for), and then the reservation
is released. An evicted engine whose worker is still tearing down or is
wedged stays counted, and a teardown step that fails strands the
reservation for the life of the process; a load that fails after its worker
started waits for that teardown before the prior-route retry, which
reserves its own commitment. The planes of a
prepared route the operator set are the model's own and are committed with
or without a drafter, also beside `HF2Q_DFLASH2=0`; only the staged draft's
head needs the drafter. The hot-swap pool charges the same figure instead of
the GGUF file size.

Measured peak process footprint of the default profile on the exact
four-agent Q5: Qwen3.8 76.73 GB against 57.87 GB on the prior route (the rule
charges 19.94 GB), Qwen3.6 57.22 GB against 35.18 GB (the rule charges
22.42 GB). Qwen3.6 had measured 77.16 GB: the loading thread primed a GPU
weight cache of its own to upload the drafter and never released it, so the
20 GB of staged expert planes existed twice (58.2 GiB after load against
39.9 GiB now). The planes remain a second representation beside the GGUF
blocks; holding one copy of every weight (about 61.7 GB and 36.2 GB) needs
every consumer to read the planes and is not part of this change.

The decision is published by `/hf2q/v1/runtime` as
`measurement_snapshot.active_controls.dflash2` (policy, selection, qualifying
class, descriptor, target and drafter source, every switch with value,
source and withholding reason, memory commitment), so a Q5 receipt, which
records that document, binds the route that ran.

One INFO line per load, `DFlash2 policy resolved`, is the startup audit the
first two D6 items quote: `policy`, `selected`, `reason` (the qualifying
class), `descriptor`, `target_source`, `drafter_source`, `profile`,
`withheld`, the memory figures and every table switch as
`NAME=value(source)`.

Not implemented by this change: Gemma and DeepSeek have no DFlash2 serving
transaction, so the policy reports them as unselected; the configuration
with the planes dropped has no exact Q5 run.

## Risks

- **Memory:** the BF16 drafter is 3.85 GB; its two selector codebooks alone are
  about 254 MB. The release gate must include peak residency and admission under
  four lanes, not just steady-state throughput.
- **Verifier width:** current production MMA is capped at eight rows and the
  diagnostic exact path at 16. A correct but row-linear 32-row verifier loses
  the intended amortization.
- **State complexity:** five draft KV rings plus target DeltaNet and attention
  state create more rollback surfaces than MTP. Atomic retained-row commit is a
  correctness requirement, not an optimization.
- **Sampling and grammar:** the reference selector's sampled policy and host/GPU
  overlap cannot be approximated by greedy-only qualification.
- **Checkpoint availability:** compatible checkpoints exist for Qwen3.8,
  Qwen3.6, Gemma4 E2B and Gemma4 31B. No compatible checkpoint was found for
  Qwen3.5, the release-gate Gemma4 26B-A4B artifact, or DeepSeek-V4. Those
  target families must fail closed until identity-matched weights with
  reproducible provenance exist; support cannot be inferred from a nearby
  model.
- **Comparator validity:** an arm is bound to its own model artifact and
  workload. The retained valid Splash arm is Qwen3.6's, not Qwen3.8's.
  Qwen3.6 must establish candidate/Splash parity; Qwen3.8 retains ADR-063's
  absolute mechanism gate and every valid same-artifact peer comparison.

## Operator decision

The operator decided on 2026-11-10 to:

1. expand v0.1.30 with the DFlash2 successor and amend ADR-063/ADR-065 rather
   than defer the architecture;
2. replace Qwen3.8's unavailable valid-Splash denominator with the
   quality-preserving 5.304 accepted-plus-bonus token/lane-cycle mechanism gate,
   while retaining its valid same-artifact peer parity and every Q5, quality,
   fairness and serving gate;
3. make the subsystem family-neutral for every enabled text-generation
   architecture, not a Qwen-only path;
4. qualify Qwen3.8 first because a compatible source-bound checkpoint exists,
   and require explicit adapters, compatible trained weights and independent
   D0-D6 qualification before activation on Qwen MoE, Gemma or DeepSeek.

Later on 2026-11-10, the operator amended the cross-family Q4 speed contract:
each candidate must finish at least 5% sooner than v0.1.29 and reach statistical
parity with every valid peer. ADR-067 inherits the exact 95% CI bounds above;
the amendment changes no checkpoint, mechanism, quality, memory or serving
gate.

The implementation may begin under this amended contract. Missing family
weights are an explicit unavailable prerequisite, never permission for an
approximate adapter, fabricated checkpoint, quality waiver or external
production fallback.

On 2026-11-11 the operator approved adopting the source-backed Splash-class
execution strategy rather than continuing threshold tuning: a packed
four-bit-class draft projection store, fused Q/K/V, fused gate/up, BF16 target
activations where parity permits, and one GPU-resident draft-to-commit
transaction. The first isolated candidate uses hf2q's owned Q4_K quantizer
because its 4.5-bit payload density matches AF4G64 and its kernels already have
width-32 coverage. Each step remains default-off until its own exact checkpoint,
quality, acceptance, memory and end-to-end peer gates pass. Cross-family support
still requires an exact family adapter and compatible trained checkpoint; this
decision does not authorize routing another architecture through Qwen state.

The operator further clarified on 2026-11-11 that a DFlash2 candidate slower
than a valid same-artifact Splash arm must not be promoted. This is binding
for Qwen3.6, whose retained Splash arm is valid. It does not turn those
Qwen3.6 runs into a valid Qwen3.8 denominator or supersede Qwen3.8's existing
mechanism substitution; the absolute mechanism, stock and other peer gates
remain unchanged.
