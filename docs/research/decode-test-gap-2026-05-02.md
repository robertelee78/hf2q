# Decode-CLI test-gap audit (2026-05-02)

**Trigger**: 3-commit thrash on `SPECIAL_TOKEN_STOPS` substring policy in
`/opt/hf2q/src/serve/mod.rs` (commits `f8cdebc → d9b23e9 → b254723`) — three
opposing edits in one day, none of them backed by a unit test that pinned the
policy at the call site against a real decode trajectory. User feedback
(verbatim): "this is deeply problematic", "our template stuff is hacky hacky
bullshit", "we need to be a lot more methodical and handle the common
templates for the models we support".

**CFA session**: cfa-20260502-decode-test-gap (solo, 5 read-only researchers
in parallel). Full per-hypothesis reports in `/tmp/cfa-decode-gap/H[1-5]-*.md`.

## Top-level finding

The visible bug today (Phi-3-style `<|end|>` fragment on a Qwen3-arch GGUF
plus subsequent `<|im_start|>` echo loop) is **NOT** a stop-policy bug. It
is a **chat-template render bug**: `render_jinja_template` at
`src/serve/mod.rs:404-413` does not pass `enable_thinking` (or any other
per-request flag) into the Jinja context. The qwen3-chatml.jinja `else`
branch at `src/backends/chat_templates/qwen3-chatml.jinja:147-153` then
opens an unclosed `<think>` block, and the model improvises a close marker
based on whatever its post-finetune training learned (`<|end|>` here),
followed by degenerate output as it tries to start a new turn. The 3-commit
substring-policy thrash treated symptoms downstream of this root cause.

Three of the five hypotheses landed PROVEN at high confidence (H2, H3, H4).
H1 was PARTIAL with a corrected bug-class definition, and H5 was
PROVEN-WITH-NUANCE (the byte-prefix infrastructure exists but is bash-only).

## Per-hypothesis verdicts

### H1 — bin-only crate test-gap → PARTIAL (high confidence)

The hypothesis's first clause was wrong: hf2q already has a narrow `[lib]`
target at `src/lib.rs:1-65` (added by ADR-017 §A.2 for kv_persist). But the
conclusion was right for a different mechanical reason: today's regression
was untestable because `SPECIAL_TOKEN_STOPS` was declared **inside**
`cmd_generate_qwen35`'s 488-LOC body. Rust fn-body `const` is fn-local-scope
and unreachable from `mod tests`. The fix landed at `b254723` (extract to
module scope, test from `super::*`) is structurally correct and proves the
lib/bin distinction was not the load-bearing variable.

**Un-fixed twin**: `src/serve/api/handlers.rs:5763` declares
`["<bos>", "<|begin_of_text|>", "<s>", "<|im_start|>"]` inside a 275-LOC
async fn body — same regression class, one careless reorder away.

**Peer cite**: llama.cpp puts the EOG list in `src/llama-vocab.cpp:2354-2655`
inside libllama, reached by `tests/test-tokenizer-1-bpe.cpp` and
`tests/test-chat.cpp`. mlx-lm avoids the bug class entirely by using
vocab-id-based stops (`tokenizer.eos_token_ids` at `mlx_lm/generate.py:721`).

**Concrete fix**: 3 man-hr — extract handlers.rs:5763 probe to module scope,
add an `assert_cmd` end-to-end leakage test, add a `syn`-based structural
audit that fails on any new `const X: &[&str]` declared inside a serve fn
body. Defer the full hf2q-core+hf2q-bin workspace split (2-4 man-weeks; right
end-state but ROI-deferred).

Full report: `/tmp/cfa-decode-gap/H1-binonly-report.md`.

### H2 — chat-template render path untested → PROVEN (high confidence)

**Headline catalog ratio: 0 / 11 (model, variant) pairs claimed-supported by
hf2q have a passing rendered-output test.** Zero. The closest existing test
is `vendor_chat_template_lengths_match_fixtures` at
`src/backends/chat_templates.rs:~95` which only pins
`QWEN3_CHATML.len() == 7764` and never executes the template. The byte-assert
sites at `serve/mod.rs:3563-3591` + `engine.rs:6240-6360` use toy templates or
substring assertions on the hardcoded `FALLBACK_GEMMA4_*` constants.

**Root cause of today's bug** (the one we actually want fixed):
`render_jinja_template` at `src/serve/mod.rs:404-413` doesn't pass
`enable_thinking` into the Jinja context. The `else` branch at
`src/backends/chat_templates/qwen3-chatml.jinja:147-153` opens an unclosed
`<think>` block; model hallucinates close marker; degeneracy follows.

**Architecture gap (hf2q vs peers)**:

| Aspect | hf2q | llama.cpp | candle |
|---|---|---|---|
| Render strategy | `FALLBACK_GEMMA4_CHAT_TEMPLATE.replace("{{PROMPT}}", user_prompt)` (string replace, Gemma-4 fallback for non-Gemma) at `serve/mod.rs:347` | AST parse + `caps_get(prog)` capability walk + `common_chat_try_specialized_template` per-family dispatch at `common/chat.cpp:580-678` | Real `ChatTemplate` struct + per-format presets at `candle-examples/src/chat_template.rs:154-279` |
| Per-family render branches | None (one mash-up path) | 13+ families (Mistral 3, GPT-OSS, Functionary, Kimi K2, LFM2, GigaChat V3, DeepSeek V3.2, Gemma4, etc.) | Format-keyed presets |
| Vendor template fixtures | None pinned with rendered output | 50+ at `tests/test-chat-template.cpp:336-712`; 56 jinjas at `models/templates/` | Per-format unit tests |
| `enable_thinking` plumbed | NO (`render_jinja_template:404-413`) | YES (per-request) | YES |
| `tools[]` rendered | Partial (handlers.rs branch) | YES (per-family) | YES |

**Full peer architectural breakdown** in
`/tmp/cfa-decode-gap/H2-chat-template-report.md` §"Peer architecture
comparison & rewrite proposal".

### H3 — no fixture matrix → PROVEN (high confidence)

Today's 12 new unit tests for `find_special_token_stop` cover the helper in
**isolation** but miss (a) the EOS multi-id collapse at `serve/mod.rs:1270`,
(b) the cumulative-decode UTF-8 path at `:1527`, (c) the entire decode-loop
body, and (d) all four model axes (Qwen3-thinking, Phi-3, GPT-OSS, Qwen3-VL).

**Peer cite**: `/opt/llama.cpp/tools/server/tests/unit/test_template.py:27-39`
ships a per-family `(template_name, reasoning, expected_end)` matrix.
`test_tool_call.py:524-567` ships a per-family `(hf_repo, reasoning_format,
expect_reasoning_content_regex)` matrix. Plus 12 per-model schema snapshots
at `tests/snapshots/`. We have none of these.

**Structural unlock**: extract `decode_loop_step` from `cmd_generate_qwen35`
(`mod.rs:1543-1652`, ~110 LOC), replacing the model call with
`FnMut(&[u32]) -> u32`. This makes Layer A (synthetic-token-stream) tests
trivial. Honest ROI: catches 5 of 6 known bug classes including today's
French-Toast repetition + candy echo + EOS multi-id collapse.

Full report: `/tmp/cfa-decode-gap/H3-fixture-matrix-report.md`.

### H4 — special-token logic fragmented → PROVEN (high confidence)

**Six hard-coded special-token sites, zero shared registry**:

1. `src/serve/mod.rs:1178-1183` — `SPECIAL_TOKEN_STOPS` (Qwen CLI decode)
2. `src/serve/api/registry.rs:634-648` — `family_resync_markers` (in-call abort)
3. `src/serve/api/registry.rs:878-912` — `ALL_FAMILY_LEAK_MARKERS` + `scrub_special_tokens`
4. `src/serve/api/registry.rs:932-935` — `is_valid_tool_name` (4 callers at registry.rs:975, 1073, engine.rs:4280, 4299)
5. `src/serve/mod.rs:1604-1610` — inline filter (today's add)
6. `src/backends/gguf.rs:2705-2716` — Gemma atomic-tokenize hash

**Smoking gun**: sites 1 (Qwen `<|im_end|>`/`<|im_start|>`/`<|end|>`/
`<|endoftext|>`) and 3 (Gemma `<|channel>`/`<|tool_response>`/`<|turn>`/
`<think>`/etc.) **share zero literal bytes** despite both purporting to
enumerate "registered special tokens". The Qwen↔Gemma asymmetry is enforced
by neither types nor tests. `src/serve/mod.rs:668-671` documents in a
comment that Gemma's `cmd_generate` decode loop deliberately skips the
fragment scan because Qwen's marker set differs from Gemma's. That comment
IS the bug — it should be code consuming a per-family registry.

**Peer cite**: llama.cpp `src/llama-vocab.cpp:1669` —
`std::set<llama_token> special_eog_ids`, populated per-vocab from GGUF
metadata at load, consumed via single `is_eog(token)` query at line 2838.
One source of truth.

**Proposal**: `SpecialTokenRegistry` (~400 LOC additive, ~100 LOC deletion,
7 sites edited, low risk). Per-family static structs (QWEN3, PHI3, GEMMA4)
with five fragment categories. The 12 invariant + cross-site sibling tests
in the H4 report would have caught (a) the iter-218 tool-call leak BEFORE
iter-219c's 4-layer defense was even needed and (b) today's thrash on the
first commit not the third.

Full report: `/tmp/cfa-decode-gap/H4-special-token-fragmentation-report.md`.

### H5 — no golden-output regression backbone → PROVEN-WITH-NUANCE (high)

Standard `cargo test` has zero default-run tests that pin decoded text or
token-id sequences from a real GGUF. Every default decode test asserts
"non-empty + no leak markers + degenerate-pattern heuristics" only.

A real byte-prefix golden corpus **already exists** but the consumers are
bash scripts: `tests/evals/reference/` (sourdough/short_hello/sliding_wrap ×
hf2q+llama, MANIFEST.json floors 179/50/108 bytes), plus 12 llama-completion
captures at `tests/coherence_golden/`. Bash gate consumers:
`scripts/sourdough_gate.sh`, `release-check.sh`, `parity_check.sh`. The
Rust-side anchor is one stale shape test
(`phase_d_sourdough_constants_match_shell_gate`) pinning floor=3094 while
the live floor is 179.

**Smallest viable golden GGUF**: Qwen3-0.6B (~390 MB, operator-pulled). hf2q
lacks llama-arch support so the obvious choice (`stories260K.gguf`) is out.

**Honest scoring against 12 past decode-class bugs**: 8 caught directly,
2 partial (today's `SPECIAL_TOKEN_STOPS` thrash + iter-218 tool-call), 2
out-of-scope. Phi-3 `<|end|>` defense remains uncatchable until hf2q gains
llama/Phi-3 arch support.

Full report: `/tmp/cfa-decode-gap/H5-golden-output-report.md`.

## Cross-cutting findings

1. **The 3-commit thrash had four distinct underlying gaps**: H1 (fn-body
   const), H2 (chat-template misrender), H3 (no call-site fixture), H4
   (no central registry). Any one of them in place would have caught at
   least one of the three commits. All four together would have prevented
   the thrash entirely.

2. **The user's "hacky hacky bullshit" call on chat templates is correct**.
   We have one render path (string-replace on Gemma fallback) for 11+
   distinct (model, variant) pairs. llama.cpp ships per-family AST-parsed
   render branches with golden tests for 50+ vendor templates.

3. **We're not lacking infrastructure ambition; we're lacking the foundation**.
   The byte-prefix golden corpus exists. The `[lib]` target exists. The
   convert-to-GGUF synthetic harness exists. None of them are wired into
   the decode/template surface.

## Concrete plan — priority-ordered

| # | Item | Lines | Risk | Catches | Done by |
|---|---|---|---|---|---|
| 1 | **Plumb `enable_thinking` into `render_jinja_template`** + add a render-byte test that pins Qwen3-thinking output for both `enable_thinking=true` and `=false`. **Fixes today's actual user-reported bug.** | ~50 | low | today's bug | this session |
| 2 | **Extract handlers.rs:5763 BOS probe** to module scope + 4-test matrix (one per family, asserting probe order). | ~30 | low | H1 twin | this session |
| 3 | **`syn`-based structural audit test** that fails on any new `const X: &[&str]` declared inside a serve/api fn body. | ~80 | low | future fn-body-const regressions | this session |
| 4 | **Extract `decode_loop_step` helper** from `cmd_generate_qwen35:1543-1652` with `FnMut(&[u32]) -> u32` injection. Land 8-12 synthetic-stream tests covering Qwen3-thinking, Phi-3, Gemma4-thinking, GPT-OSS reasoning patterns. | ~250 | med | H3 Layer A — 5/6 known bug classes | next sub-iter |
| 5 | **`SpecialTokenRegistry` with QWEN3/PHI3/GEMMA4 per-family structs + 5 categories**. Migrate all 6 sites to consume it. 12 invariant + cross-site sibling tests. | ~400 add / ~100 del | med | iter-218 leak + today's thrash | next sub-iter |
| 6 | **Operator-gated Qwen3-0.6B golden harness** — Layer B in H3 + dual-track Layer 2 in H5. 6 byte-prefix tests via env-override + sentinel-skip. Reuses existing tests/evals/reference/ floor pattern. | ~300 | low | most decode regressions at PR-time on operator runs | within this week |
| 7 | **AST-based per-family chat-template renderer + 11+ (model, variant) byte-pin tests** — full peer parity. Mirror llama.cpp's `common/chat.cpp:580-678` architecture. | ~1500 + 11 jinjas | high | the entire chat-template hacky-bullshit class | multi-week, separate ADR |

Items 1-3 land in this CFA cycle (this turn or the next). Items 4-5 are the
next /loop iteration. Item 6 is operator-gated (needs Qwen3-0.6B pull). Item
7 is multi-week and warrants its own ADR.

## References

- `/tmp/cfa-decode-gap/H1-binonly-report.md` — bin-only crate (PARTIAL)
- `/tmp/cfa-decode-gap/H2-chat-template-report.md` — chat-template (PROVEN)
- `/tmp/cfa-decode-gap/H3-fixture-matrix-report.md` — fixture matrix (PROVEN)
- `/tmp/cfa-decode-gap/H4-special-token-fragmentation-report.md` — registry (PROVEN)
- `/tmp/cfa-decode-gap/H5-golden-output-report.md` — golden output (PROVEN-WITH-NUANCE)
- Today's incident commits: `f8cdebc` (added `<|im_start|>`), `d9b23e9`
  (removed `<|im_start|>`), `b254723` (restored + added `<|end|>` + 12-test matrix)
- Memory pin: `project_hf2q_no_lib_target_unit_test_friction` (now superseded
  in part — there IS a narrow `[lib]`, but bug class is fn-body-local consts)
