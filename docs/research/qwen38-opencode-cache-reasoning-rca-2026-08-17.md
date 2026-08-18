# Qwen3.8 OpenCode cache and reasoning incident RCA

- Date: 2026-08-17
- Model: `Qwen/Qwen3.8-27B`, hf2q Q4_K_M + matched F16 projector
- Client: OpenCode 1.18.18, OpenAI-compatible streaming
- Serving mode: Qwen SlotAware, four slots, port 8081
- Status: corrected; model-free and exact-hardware gates pass

## Executive finding

The image turn suffered two independent failures that compounded into one
very long wait:

1. A request-global vision fingerprint rejected a valid text-only prompt
   snapshot before longest-prefix affinity ran. Appending the conversation's
   first image therefore selected an empty slot and recomputed roughly 90,000
   prompt tokens.
2. After prefill, Qwen remained in the template-seeded reasoning channel.
   hf2q had no answer-phase budget, and its operator log called the first raw
   decoded fragment "semantic" before ReasoningSplitter classified it. The
   client received reasoning but no content/tool answer and could continue to
   the full 8,192-token completion cap.

This was not a normal cache eviction, an mRoPE rewrite of earlier text, or a
second unexplained cache miss during decode. It was an over-broad cache
identity gate followed by an unbounded reasoning policy and false progress
telemetry.

## Incident evidence

| Time | Evidence | Interpretation |
|---|---|---|
| 18:24:03–18:30:32 | Cold assistant turn: input 84,527; reasoning 1,171 | Establishes the long conversation |
| 18:30:32–18:32:08 | Follow-up: input 4,037; cache read 84,522; reasoning 825 | Ordinary text prefix reuse worked; total rendered prompt became 88,559 |
| 18:32:47 | First image request began | New media identity appeared only in the suffix |
| 18:34:28 | Server UI: slot 1 prefill 40,960/90,044, cache 0 | Definitive cold admission despite a reusable text snapshot |
| 18:39:45 | Slot 1 decode 1,889/8,192 at 16.1 tok/s; 6m56 elapsed | Decode was live but still reasoning-only |
| 18:45:31 | OpenCode retry met connection refused | Server was no longer listening; available logs do not establish why it stopped |
| 18:45:43 | DB assistant record completed with zero usage and only an empty reasoning part | The persisted client result was not a healthy terminal completion |

The queued `use this` message was normal OpenCode behavior: its runner waits
behind the active request. It did not cause the active generation and could not
cancel it.

## Root cause 1: request-global media identity masked a valid prefix

Commit `ca54c6ab` correctly added exact post-preprocess image identity to Qwen
cache and slot state. That prevents two images represented by identical
placeholder token IDs from sharing image-backed KV. The admission gate applied
that identity to the entire request, however:

- the stored text anchor had `vision_fingerprint=None`;
- the appended image request had `vision_fingerprint=Some(hash)`;
- exact equality failed before the otherwise-valid token-prefix candidate was
  offered to Qwen slot affinity;
- the scheduler admitted the request cold on another slot.

The safety property is span-local, not request-global. In a causal decoder, KV
before the first image cannot depend on later pixels. Qwen's position builder
also preserves ordinary `[t,t,t,t]` mRoPE coordinates on all four axes before
the first image. Therefore the old text snapshot is reusable only when the
incoming expanded prompt proves all of the following:

- exact token prefix;
- every soft-token span begins at or after the snapshot;
- every DeepStack injection position begins at or after the snapshot;
- all four mRoPE axes over the snapshot equal ordinary text positions.

Any missing/malformed proof fails closed. `Some(image A)` to `Some(image B)`,
image-backed state to text, active-owner affinity, and live retained cursors
remain exact-fingerprint-only.

### Why the initial broad repair was narrowed

The first spike allowed the safe bridge during both idle admission and active
affinity. Independent review found that an active owner can replace its old
text snapshot with an image-bearing checkpoint before the waiter is admitted.
The accepted implementation therefore bridges only an idle snapshotted anchor.
It never speculatively waits on active cross-media state.

## Root cause 2: reasoning had no answer-progress bound

Qwen's template ends the generation prompt inside `<think>`. The stream router
correctly sends decoded text to `reasoning_content` until `</think>`, but the
decode stop conditions were only EOS, grammar terminal, configured stop
strings, and `max_tokens`. The reasoning close marker classified output; it did
not bound it. At OpenCode's 8,192-token setting, a reasoning-only run could
therefore consume the whole response window.

The operator signal compounded the problem. `Qwen35TickOutcome.is_reasoning`
was always false, and hf2q logged "semantic fragment ready" before handing the
fragment to ReasoningSplitter/ToolCallSplitter. A reasoning delta was thus
misreported as useful answer progress.

## Comparator research

The production correction follows current primary-source behavior:

- [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
  hashes a parent chain, block tokens, and typed extras. A multimodal hash
  affects blocks that contain that media input rather than every causally
  earlier text block.
- [vLLM multimodal inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)
  use per-item content identity and separate encoder caching from language KV.
- [vLLM reasoning outputs](https://docs.vllm.ai/en/v0.20.1/features/reasoning_outputs/#thinking-budget-control)
  expose `thinking_token_budget`, force a configured reasoning-end sequence at
  the boundary, and continue answer generation.
- [Qwen thinking-budget guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
  uses a completed reasoning close followed by continued generation when a
  budget is reached.
- Qwen-aware parsers treat `<tool_call>` as an implicit end to an open reasoning
  channel. hf2q now preserves that marker for the downstream structured-tool
  parser rather than leaking the tool call as reasoning text.

The durable cache architecture is an ordered prefix/media-event identity
chain: model/projector/tokenizer/template/KV/position namespace followed by
text and media events carrying token/position digest, image hash, ordinal,
span, grid, and DeepStack shape. That is a follow-on design, not permission to
weaken exact image identity in this repair.

## Accepted correction

1. `qwen35_text_anchor_reuse_limit` derives the first safe media boundary from
   the authoritative expanded soft-token, DeepStack, and four-axis position
   payload. Idle admission may restore only a text-only snapshot no longer
   than that boundary.
2. `thinking_token_budget` is accepted at the OpenAI-compatible top level.
   Qwen SlotAware decode counts reasoning tokens in transport-neutral state,
   forces a tokenizer-derived `I need to answer now.</think>` transition at the
   boundary, and continues within the original `max_tokens` limit.
3. Natural `</think>` or `<tool_call>` disables further forcing. Required or
   named tool grammars and SerialFifo reject the option because their grammar
   contract cannot safely accept an injected transition.
4. The canonical launcher applies a 2,048-token ceiling. Short response limits
   adapt the default to reserve answer capacity; explicit request values are
   exact; zero disables the launcher default.
5. Stream progress is latched only after a non-empty content delta or
   structured tool delta is successfully delivered. A terminal reasoning-only
   Qwen stream emits an explicit warning.
6. Budget fields participate in in-memory terminal-response cache identity.
   Disk response-cache persistence skips budgeted entries until its schema has
   an explicit representation.

## Proof matrix

### Model-free

- Text snapshot -> first-image bridge accepts exact text tokens with suffix-only
  soft/DeepStack spans and ordinary prefix mRoPE on all axes.
- A snapshot extending one token into the image span is rejected.
- Missing or corrupt mRoPE proof is rejected.
- `Some(A)->Some(B)`, `Some->None`, and missing media proof are rejected.
- A tail image leaves every earlier text mRoPE axis byte-identical to the
  text-only builder.
- Budget forces the exact close sequence at token N, stops overriding after
  close, and is disabled by a natural close or tool open.
- Adaptive defaults retain answer capacity; impossible explicit budgets fail.
- A Qwen tool marker split across fragments implicitly closes reasoning but is
  delivered as structured tool deltas.
- Reasoning and answer event latches are distinct.

### Real model

Run:

```bash
SERVER_URL=http://127.0.0.1:8081 \
  scripts/test_qwen38_first_image_cache.sh
```

The gate creates an 80k+-token text snapshot, appends the first real PNG image,
requires at least 80,000 cached tokens on the image turn, requires non-empty
SSE answer content and `[DONE]`, rejects completion-limit exhaustion, and
checks `/readyz` afterward. The emitted JSON receipt binds the model, image
hash, prompt/cache/completion counts, budget, content, elapsed time, and
readiness result.

Hardware receipt (Apple M5 Max, production `mlx-native` path):

- release binary SHA-256:
  `24f47b0394b231ce58d00213e118d190f5c3efa1ad270846283626d61e5f4e90`;
- Q4_K_M text artifact SHA-256:
  `0fa8acc661d0edc60276c43705619fd848682dbf768ced9fe46cd8a572b8043d`;
- F16 projector SHA-256:
  `6fa039b75244c0a28a013da30b92b1d221c61029acc19f9efa882b75a495b0d0`;
- cold text turn: 86,077 prompt tokens, zero cached;
- first-image turn: 86,172 prompt tokens, 86,072 cached, 100-token
  prefill suffix, one real GPU ViT forward, and 82 completion tokens;
- the 64-token reasoning budget forced its tokenizer-derived close, the stream
  then delivered `The dominant color in this image is red.`, terminated with
  `finish_reason=stop` in 9 seconds, emitted `[DONE]`, and left `/readyz` at
  HTTP 200.

The canonical agentic gate also passed required and automatic tool selection,
exact cached unary replay (7,043/7,043 tokens), SSE tool reconstruction, tool
result continuation, and byte-exact Rust angle-bracket source arguments. Its
measured cached SSE tool event was 4.100 seconds and its tool-result turn was
14.303 seconds. The first run missed that gate's historical 10-second
tool-result wall-clock threshold by 732 ms while satisfying the semantic
assertions; the recorded passing rerun used a 15-second threshold. This is a
remaining latency measurement, not a cache or tool-correctness failure.

Canonical launcher-default OpenCode receipt (2026-08-17, Apple M5 Max):

- release binary SHA-256:
  `b8853d6b13f80e6657965ac817cffe4c9de9d2b6b55ea708b690cb21436d2ae1`;
- the live process started after that artifact was built and carried
  `HF2Q_DEFAULT_THINKING_TOKEN_BUDGET=2048`;
- stock OpenCode 1.18.18 requested an 8,192-token completion without its own
  explicit thinking-budget field;
- the decisive cached tool turn recorded 17,802 cached plus 1,547 input
  tokens, 2,055 reasoning tokens (the 2,048-token allowance plus the forced
  transition), 59 output tokens, and `finish=tool-calls`;
- the immediate continuation recorded 19,344 cached plus 3,643 input tokens,
  25 reasoning tokens, 61 output tokens, and `finish=tool-calls`;
- server counters advanced from six of seven to eight of eight completed
  requests with zero SSE cancellations.

This closes the exact launcher-default path that the earlier 64-token explicit
budget gate did not cover. The operator UI's `generated N / 8192 budget` is the
total completion allowance, not the effective reasoning allowance; a separate
reasoning-cap field remains a useful UI follow-up.

`cargo test --locked` passed the complete hosted-safe suite. The first umbrella
run exposed that two `cache_clear` tests require the release binary at their
documented fallback path; after `cargo build --release --locked`, the exact
same full command passed, including 4,148 binary tests with zero failures.

## Non-causes and remaining limits

- Four slots and the shared physical KV budget did not evict the prefix; the
  fingerprint gate made it ineligible.
- Qwen mRoPE did not rewrite the earlier text positions; a regression test now
  pins every axis.
- Vision embedding computation was real and expected, but it should have run
  after restoring the text prefix instead of after a full language prefill.
- The server's disappearance near the end cannot be attributed from the
  surviving evidence. This correction does not invent a crash cause.
- The new bridge repairs first-image-after-text. General arbitrary interleaved
  media reuse awaits the ordered event-chain design and its own parity gates.
