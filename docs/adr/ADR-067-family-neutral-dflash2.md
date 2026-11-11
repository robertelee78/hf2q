# ADR-067: Family-neutral DFlash2 speculative decoding

Date: 2026-11-10
Status: Accepted; implementation authorized for the v0.1.30 program
Decider: Robert E. Lee (product owner), 2026-11-10
Scope: every enabled hf2q text-generation architecture; Qwen3.8 is the first qualified checkpoint

## Problem

ADR-063 requires a valid Qwen3.8 comparison against Splash for Q4. The corrected
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
- Keep the promoted route available behind `HF2Q_QWEN_DFLASH2=0` until the new
  default passes Q5.
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
fine-tuned target.

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
Splash transaction: Qwen3.8 remains 4.918 times Splash's 16.420-second median,
and target output readback, host acceptance, target-state settlement and
draft-context commit remain separate boundaries.

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
lane-cycle on the matched workload without a quality or task regression. The
17 completed Splash runs form an independently valid arm despite the separate
peer arm's failure: Splash's median is 16.420344 seconds, with no Splash-arm
failure. Promotion therefore requires candidate/Splash statistical parity,
with the candidate/Splash 95% CI upper ratio at most 1.00. Qwen3.8 must also
finish at least 5% sooner than v0.1.29, with the candidate/stock 95% CI upper
ratio at most 0.95, and reach statistical parity with every other valid
same-artifact peer, with each candidate/peer upper ratio at most 1.00. The
absolute mechanism gate remains necessary but cannot substitute for the valid
Splash arm.

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
- **Comparator validity:** the overall Qwen3.8 campaign is invalid because its
  separate peer arm failed, but the 17/17 completed Splash arm is independently
  valid and binding. Neither the absolute mechanism gate nor another peer can
  waive candidate/Splash parity.

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
than the valid Splash arm must not be promoted. This supersedes the temporary
unavailable-Splash substitution in item 2 above now that a valid Splash arm
exists; it does not relax the absolute mechanism, stock or other peer gates.
