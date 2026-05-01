# iter-219b — llama.cpp peer cross-check on Gemma 4 tool-call parsing

**Date:** 2026-05-01
**Source:** ADR-005 Phase 4 reopen iter-218 honest-scope follow-up
**Audit driver:** Agent D (researcher, worktree-isolated)
**Status:** Research finding, NOT a fix landing

## Summary

Three areas compared between `/opt/llama.cpp` and `/opt/hf2q` to triage the iter-218 LIVE-captured `function":{"name":"get_currentcall:get_current_weather"}` malformed name.

### A. Open-marker stripping — EQUIVALENT

- **llama.cpp** `common/chat.cpp:1183-1184` — `scan-to-toolcall = until("<|tool_call>")`; `content = until_one_of({"<|channel>", "<channel|>", "<|tool_call>"})`. Content before the marker is preserved verbatim into the assistant message; marker bytes are stripped.
- **hf2q** `src/serve/api/registry.rs:663-705` — `ToolCallSplitter::feed` emits `Content(t)` for everything before `<|tool_call>` and `ToolCallText(t)` between `<|tool_call>` and `<tool_call|>`. Marker bytes stripped.

Both reject the iter-218 doc-named hypothesis: a leading `get_current` content fragment will be routed to delta.content, NOT prepended to the body. iter-219 BASELINE unit reproducers confirm this on real-tokenizer round-trip bytes.

### B. Grammar shape — DIVERGES on lazy-gate

- **llama.cpp** `common/chat.cpp:1130-1186` — grammar always includes the leading `<|tool_call>call:` literal anchored with `peek(literal("{"))`. Repeat `1..(parallel?-1:1)`. The marker is part of the constrained grammar from byte 0; no awaiting_trigger gate.
- **hf2q** `src/serve/api/registry.rs:1349-1391` — three shapes: `SingleBody` / `OneOrMoreCalls` / `OneOrMoreCallsBodyOnly` (Wave 3.5 HIGH-1 default after iter-218). `OneOrMoreCallsBodyOnly { parallel: false }` emits `body close_marker space` — **the leading open marker is stripped from the grammar** because the awaiting_trigger no-op gate is supposed to consume it before grammar engages.

**Divergence consequence:** under hf2q's lazy gate, between the model's first `<|tool_call>` token (at which the splitter calls `runtime.trigger()`) and the next token sampled with the now-eager grammar mask, there is exactly one token boundary. Tokens emitted by the model BEFORE the splitter recognizes `<|tool_call>` (e.g. partial-marker token sequences) flow through unmasked. llama.cpp's grammar locks the open marker down inline, leaving no such gap.

### C. Stream-emitter contract — DIVERGES on incremental shape

- **llama.cpp** `common/chat-peg-parser.cpp:869-1010` — `gemma4_to_json` runs full-AST reparse per pass and emits ONE `arguments` delta as a string-prefix diff against the prior pass.
- **hf2q** `src/serve/api/engine.rs:3382-3700` — `ToolCallStreamEmitter::advance` scans body byte-by-byte for closed-kv boundaries and emits per-kv `arguments` deltas as they close.

Adopting llama.cpp's reparse-then-diff pattern would simplify hf2q's emitter at the cost of running the parse pass per chunk. Out of scope for iter-219b; flagged for ADR-005 Phase 5 candidate.

## Convergence verdict

For the byte streams the iter-219 BASELINE unit reproducers test (canonical `<|tool_call>call:get_current_weather{...}<tool_call|>`), llama.cpp and hf2q produce equivalent output. The iter-218 LIVE bug `get_currentcall:get_current_weather` is **upstream of both the splitter and the per-kv emitter** — the splitter+emitter unit surface is contract-correct against either reference.

## Hypothesis-narrowing for Agent A's LIVE capture

Focus the LIVE byte capture on:

1. The exact tokens emitted at and immediately before the splitter's first `<|tool_call>` recognition — does the model emit `<|tool_call>` as token id 48 (single special token), or as a multi-token sequence like `<|tool` + `_call>`? Multi-token emission would create a window where post-`<|tool` tokens flow unmasked into the body.
2. Whether `runtime.trigger()` actually fires synchronously with the splitter's `ToolCallOpen` event, or whether there's a token-boundary lag.
3. Whether the route_content path or the cache-replay path (`engine.rs:4163-4239`) was active during the LIVE capture — they have separate trigger paths and the cache-replay one drops grammar.

## Citations

| Claim | File:Line |
|---|---|
| llama.cpp open-marker scanner | `/opt/llama.cpp/common/chat.cpp:1183-1184` |
| llama.cpp tool grammar inline shape | `/opt/llama.cpp/common/chat.cpp:1172, 1177-1181` |
| llama.cpp gemma4 mapper | `/opt/llama.cpp/common/chat-peg-parser.cpp:869-1010` |
| llama.cpp delta diff | `/opt/llama.cpp/common/chat.cpp:186-209` |
| hf2q splitter feed | `/opt/hf2q/src/serve/api/registry.rs:663-705` |
| hf2q grammar shapes | `/opt/hf2q/src/serve/api/registry.rs:1280-1391` |
| hf2q name extraction | `/opt/hf2q/src/serve/api/engine.rs:3813` |
| hf2q `OneOrMoreCallsBodyOnly` default flip | `/opt/hf2q/src/serve/api/handlers.rs:3257-3294` (iter-218) |
