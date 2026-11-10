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
| `qwen35` dense, including Qwen3.8 | dense recurrent/attention capture and verify | Yes for Qwen3.8-27B |
| `qwen35moe`, including Qwen3.5/3.6 | MoE recurrent/attention capture, routing and confidence-priced verify | No official DFlash2 checkpoint; official DFlash v1 checkpoints exist |
| `gemma4` dense and MoE | Gemma hybrid-cache capture, verify and rollback | No official DFlash2 checkpoint; official DFlash v1 checkpoints exist |
| `deepseek4` | DeepSeek recurrent/attention capture, MoE verify and rollback | No official DFlash2 checkpoint |

Embedding-only architectures do not decode and are excluded. Conversion-only
architectures enter this table when their generation surface is enabled; they
must not be routed through an approximately compatible adapter.

The official `z-lab/dflash-2` collection contains Qwen3.8-27B and
Muse-Glimmer-30B; Muse is not an hf2q generation architecture. The pinned
DFlash repository publishes no training/export contract, and its training-code
request issue #19 remains open. Therefore a family without compatible trained
weights can receive the shared runtime and explicit adapter, but hf2q must report
DFlash2 as unavailable for that target until source-bound weights and their
training/export provenance exist. Synthetic or cross-family weights cannot
qualify a production path.

The implementation order follows available evidence, not family preference:
the shared checkpoint/runtime/selector/transaction layer first; Qwen3.8
end-to-end because its weights exist; then Qwen MoE, Gemma and DeepSeek adapters
as compatible weights and family-specific parity fixtures become available.
This is one cross-family program, not a Qwen implementation followed by
uncommitted ports.

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

### D2: 32-row target verifier

- Extend the ADR-065 verifier to four independent windows of one to eight rows,
  32 rows total.
- Prove every row's hidden, logits, and argmax against serial spec-off, including
  DeltaNet sequential state, full-attention KV positions, and mixed lane
  lengths.
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
operator replaced its unavailable valid-Splash denominator with this
source-bound absolute gate on 2026-11-10; failed Splash output remains evidence,
not a Win. Qwen3.8 must still beat its valid same-artifact peer under Q4's 95%
CI rule.

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
- **Checkpoint availability:** Qwen3.8 is the only enabled generation family
  with an official DFlash2 checkpoint at this decision. Other families cannot
  advertise support until compatible weights and provenance exist.
- **Comparator validity:** the failed Qwen3.8 Splash campaign remains invalid.
  The operator-approved absolute mechanism gate does not relabel it.

## Operator decision

The operator decided on 2026-11-10 to:

1. expand v0.1.30 with the DFlash2 successor and amend ADR-063/ADR-065 rather
   than defer the architecture;
2. replace Qwen3.8's unavailable valid-Splash denominator with the
   quality-preserving 5.304 accepted-plus-bonus token/lane-cycle mechanism gate,
   while retaining its valid same-artifact peer Win and every Q5, quality,
   fairness and serving gate;
3. make the subsystem family-neutral for every enabled text-generation
   architecture, not a Qwen-only path;
4. qualify Qwen3.8 first because a compatible source-bound checkpoint exists,
   and require explicit adapters, compatible trained weights and independent
   D0-D6 qualification before activation on Qwen MoE, Gemma or DeepSeek.

The implementation may begin under this amended contract. Missing family
weights are an explicit unavailable prerequisite, never permission for an
approximate adapter, fabricated checkpoint, quality waiver or external
production fallback.
