# iter-219b — Grammar-exhaust audit on `OneOrMoreCallsBodyOnly{parallel:false}`

**Date:** 2026-05-01
**Source:** ADR-005 Phase 4 reopen iter-218 honest-scope follow-up
**Audit driver:** Agent C (researcher, worktree-isolated, read-only)
**Status:** Research finding, NOT a fix landing
**Companion:** `iter219b-llama-cpp-cross-check-2026-05-01.md` (Agent D)

## Summary — iter-218's "natural grammar-exhaust" termination is structurally broken

iter-218's commit message claimed:

> single-call termination is delivered by the grammar shape exhausting (parallel_tool_calls=false default → shape `body close_marker space` exhausts after first close → is_dead → halt)

This audit traces the actual runtime state machine and finds:

- After `accept_bytes(b"<tool_call|>")` runs at `engine.rs:4996`, the runtime's stacks expand the `space` rule and yield THREE successor stacks: one **empty** (accepting), one starting on `Char(' ')`, one on `Char('\n')`.
- `is_accepted()` returns **true** (empty alt makes runtime accepting).
- `is_dead()` returns **false** (continuation stacks remain alive).
- The decode loop break at `engine.rs:5045-5059` is `is_accepted && bytes.is_empty()` for the just-sampled token. Token id 49 (`<tool_call|>`) decodes to **12 bytes (non-empty)**, so the break is suppressed.

**Result:** the loop continues past the close marker. The grammar mask DOES kill `<|tool_response>` (token id 50) via stack-walk in `mask_invalid_tokens` — verified the empty-stack contributes nothing and the `Char(' ')` / `Char('\n')` stacks reject `<`. But mask only protects when grammar is attached AND the path uses `sample_logits=true`. The `<|tool_response>` leak therefore most likely surfaces on:

1. **Greedy / non-grammar paths** — `sample_logits=false`, mask never fires, the unconditional `accept_bytes` at `engine.rs:4994-4998` advances the empty-alt stack on every byte but doesn't BLOCK the model from emitting non-whitespace.
2. **Non-streaming `generate_once` mirror** — same break logic applies at `engine.rs:2960-3010` with the popped-token wart (the wave-2.5 audit divergence cited in `engine.rs:1401, 1489, 1554, 2041, 2145, 2195`).

## Q&A receipts

### Q1 — Mask filtering token 50 post-close

**Yes**, mask correctly sets `logits[50] = -inf` after `<tool_call|>` is consumed. `mask_invalid_tokens` (`mask.rs:58-93`) clones the runtime, feeds `<|tool_response>`'s 17 bytes, walks each stack:
- Empty stack → no successor (sampler.rs:302-304 short-circuits).
- `Char(' ')` first stack → rejects `<`.
- `Char('\n')` first stack → rejects `<`.

All resulting stacks empty → `accept_bytes` returns false → mask kills token 50.

### Q2 — Empty-bytes break gate is the bug

`engine.rs:5049-5057` checks `is_accepted && next_token bytes empty`. Designed for grammars that exhaust on an EOS-like token with zero bytes. Tool-call grammars terminate on `<tool_call|>` (id 49) which decodes to 12 printable bytes — break never fires, loop continues until `max_tokens` or another stop-string hits.

Mirror at `engine.rs:2960-3010` (non-streaming `generate_once`) has the same logic with a popped-token wart.

### Q3 — Grammar doesn't exhaust on its own

Confirmed. `space ::= | " " | "\n"{1,2} [ \t]{0,20}` has THREE alternatives. Once expanded:
- Alt 1 (empty) → instantly accepting; consumes no bytes; remains accepting.
- Alt 2 (`Char(' ')`) → matches one space, then end.
- Alt 3 (`\n{1,2}[ \t]{0,20}`) → matches 1-2 newlines + 0-20 spaces/tabs.

Empty alt + `[ \t]{0,20}` (min 0) keeps `is_accepted=true` perpetually. `is_dead` never fires unless ALL stacks die (impossible while empty alt is in play). Engine's "halt on dead" path is structurally unreachable for `OneOrMoreCallsBodyOnly{parallel:false}`.

## Hypothesis — testable

**(c) Loop break condition has wrong empty-bytes guard.** Intended semantic: "halt when grammar is fully satisfied and only whitespace continuations remain." Implemented as `is_accepted && bytes.is_empty()` which fails for any close marker that decodes to printable bytes.

**(b) [complementary] is_dead never flips** because the empty alt of `space` keeps the runtime perpetually accepting-and-alive simultaneously.

The fix candidates (ranked):

1. **Replace `space` with a self-exhausting rule**. e.g. `space ::= | " " | "\n"{1,2}` (drop the unbounded `[ \t]{0,20}` tail) AND check `is_dead` after greedy-consume of remaining whitespace tokens. Cleaner; tightens grammar to actual chat-template emit.
2. **Add `is_dead_or_only_whitespace()` API to GrammarRuntime** and replace the `bytes.is_empty()` guard. More general but adds API surface.
3. **ToolCallClose-as-stop-signal**. After the splitter emits `ToolCallClose` AND `parallel_tool_calls=false`, set a `decode_should_stop` flag the loop checks. Aligns with llama.cpp's signal-based termination (per Agent D's cross-check).

iter-218's research log explicitly REJECTED Fix C (signal-based) as "inferior — it traded the loop bug for a downstream `<|tool_response>`-leak / max_tokens-burn surface that the natural grammar-exhaust path avoids cleanly." This audit shows iter-218's premise — that the grammar-exhaust path avoids the leak — is **false**. The grammar doesn't exhaust; the leak surface remains.

## Proposed fail-first test (sketch)

Lives in `src/serve/api/grammar/sampler.rs` tests. Uses synthetic grammar (no real tokenizer). Asserts the structural claim that `is_dead` never flips post-close for the canonical `body close_marker space` shape.

```rust
#[test]
fn space_rule_after_close_marker_does_not_reach_is_dead() {
    let src = "\
        root ::= body close space\n\
        body ::= \"B\"\n\
        close ::= \"<tool_call|>\"\n\
        space ::= | \" \" | \"\\n\"{1,2} [ \\t]{0,20}\n\
    ";
    let g = parse(src).expect("parse");
    let rid = g.rule_id("root").expect("root");
    let mut rt = GrammarRuntime::new(g, rid).expect("rt");
    assert!(rt.accept_bytes(b"B<tool_call|>"));
    assert!(rt.is_accepted(), "post-close: empty alt of space keeps accepted=true");
    // BUG: is_dead is FALSE here — so the engine's `is_dead` break path
    // is unreachable for OneOrMoreCallsBodyOnly{parallel:false}.
    // After fix: this assertion should be `assert!(rt.is_dead())`.
    assert!(
        !rt.is_dead(),
        "iter-219b ground-truth pin: HEAD's space rule keeps is_dead=false post-close. \
         Fix candidate is to replace `space` with a self-exhausting rule (drop the \
         unbounded [ \\t]{{0,20}} tail) so this assertion flips to `assert!(rt.is_dead())`."
    );
}
```

This test pins the CURRENT (buggy) behavior so the fix landing is auditable: the assertion flips from `!rt.is_dead()` to `rt.is_dead()` exactly when the structural fix lands.

## Citations

| Claim | File:Line |
|---|---|
| Streaming break gate | `/opt/hf2q/src/serve/api/engine.rs:5045-5059` |
| Non-streaming break gate (mirror) | `/opt/hf2q/src/serve/api/engine.rs:2960-3010` |
| `space` rule emission | `/opt/hf2q/src/serve/api/registry.rs:1396` |
| `OneOrMoreCallsBodyOnly` body-only single shape | `/opt/hf2q/src/serve/api/registry.rs:1389-1391` |
| `mask_invalid_tokens` impl | `/opt/hf2q/src/serve/api/grammar/mask.rs:58-93` |
| Empty-bytes mask short-circuit | `/opt/hf2q/src/serve/api/grammar/mask.rs:77-79` |
| `accept_bytes` impl | `/opt/hf2q/src/serve/api/grammar/sampler.rs:530-605` |
| `is_accepted` / `is_dead` | `/opt/hf2q/src/serve/api/grammar/sampler.rs:615-636` |
| Parser leading-`|` produces empty alt | `/opt/hf2q/src/serve/api/grammar/parser.rs:256-275, 508-510` |
| `parallel_tool_calls` default flip | `/opt/hf2q/src/serve/api/handlers.rs:2958` |
