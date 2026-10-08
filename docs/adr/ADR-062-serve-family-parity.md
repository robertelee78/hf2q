# ADR-062: Serve family parity: one serving contract across model families

- **Status**: proposed
- **Date**: 2026-10-08
- **Deciders**: Robert E. Lee (owner)
- **Tags**: serve, operator-settings, api-contract, scheduler, kv-persist, reasoning, v0.1.24
- **Milestone**: v0.1.24
- **Work items**: #278 (umbrella), #256

The key words MUST, MUST NOT, SHOULD, and MAY are to be interpreted as
described in RFC 2119.

## Context

The owner's principle (2026-10-08): `hf2q serve` gives a user the same
settings and expectations whatever model family it serves (Qwen3.5/3.6/3.8,
Gemma 4, DeepSeek-V4), and its default is functional, never a broken mode.

A read-only audit of main @ be17e2fe (issue #278) and hands-on QE of
Gemma 4 found that the user experience depended on the family, on whether
`hf2q setup` had run, and on the scheduler:

- A serve with no flags and no `config.toml` ran `fifo-serial`, whose
  single-transaction prefill rejects prompts over 2048 tokens (Qwen) with
  HTTP 501. OpenCode's system prompt alone is 2,174 tokens, and OpenCode
  retries 5xx indefinitely. Setup recorded `inflight-batched` with 1 slot;
  the engine's own inflight default was 4.
- Setup wrote one Qwen-tuned profile (repetition penalty 1.05, thinking
  budget 2048, tool-thinking budget 512) for every family. DeepSeek-V4's
  launcher deliberately uses 1.0, and an unconfigured Qwen got no profile.
  Gemma's and Qwen's qualified behavior needed launcher-only `HF2Q_*`
  variables. The default port was 8080, 8081, or 8082 depending on the path.
- Client mistakes and unservable requests surfaced as 500/501 on some
  families and 400 on others (DeepSeek-V4 embeddings always 500; per-family
  prompt caps as 501; a JSON-schema response cut by `max_tokens` as 500
  (#256); a Qwen
  `tool_choice=required` under `fifo-serial` as a 400 about a field the client
  never sent).
- `--max-slots` above 8 aborted startup unless an environment variable lifted
  it, with an error blaming speculative decoding that was not running.
- `--kv-persist` was silently ignored on DeepSeek-V4 while `/v1/control`
  reported it enabled; Qwen's disk restart path runs only on `fifo-serial`.
- `reasoning_effort` worked only on DeepSeek-V4 (and `medium` returned 400
  there, which stock OpenCode sends); thinking budgets were Qwen-only.

Five design studies (D1–D5) traced each area through the code and propose the
decisions below. They are the evidence base for this ADR.

## Decision

### D0. Functional default scheduler (implemented on fix/serve-default-scheduler)
With no `--scheduler` and no `config.toml`, serve MUST use the scheduler and
slot count `hf2q setup` records: `inflight-batched`, 4 slots, from one shared
constant. An over-limit prompt under an explicitly chosen `fifo-serial` MUST be
HTTP 400 `prompt_exceeds_scheduler_limit` naming the fix.

### D1. One built-in serving profile per family
- Precedence MUST be: request field > CLI `--default-*` > the family's
  built-in value. Setup MUST NOT write behavior profile keys; the
  `[serve] repetition_penalty`, `thinking_token_budget`, and
  `tool_thinking_token_budget` keys are retired (parse warns and ignores).
- Family defaults are the values each canonical launcher already uses:
  Qwen 1.05 / 2048 / 512 with the encoder session on; Gemma 4 1.05 with
  cross-slot admission (25 ms coalesce) on; DeepSeek-V4 1.0 with a
  tool-thinking budget of 512 (owner decision 2026-10-08, see D5).
- Launchers MUST need no `HF2Q_*` variables for qualified behavior.
  `HF2Q_KV_LCP_LONG_RESUME` (inert on the default scheduler) is dropped.
- The default port MUST be 8081 on every path.
- `hf2q info` and `/hf2q/v1/runtime` MUST report each effective value and its
  origin.

### D2. One error contract
- Client mistakes and requests this server can never satisfy MUST be 4xx with
  a stable `code`; 5xx MUST be reserved for server faults. Engine errors MUST
  carry a typed classification instead of substring sentinels.
- `capability_unsupported` becomes 400. Context overflows use a code OpenCode
  recognizes for compaction.
- Running out of `max_tokens` before a grammar completes (#256, and required
  tool calls cut by `max_tokens`) MUST report `finish_reason: "length"`, never
  a 500 and never a partial `tool_calls` entry.
- DeepSeek-V4 embeddings MUST return 400 until supported, and the shipping
  contract MUST stop listing them.

### D3. Slot capacity is a validated operator setting
`--max-slots` MUST accept 1–8 for every family (`MAX_SUPPORTED_SLOTS`), checked
by one function at the CLI, config load, and the setup prompt. The
`HF2Q_MAX_BATCHED_SLOTS` / `HF2Q_SPEC_DECODE_ALLOW_OVERSIZED` gates are removed.
Raising the bound requires a hands-on qualification at the new width. Docs and
`--help` MUST state defaults and the range.

### D4. `--kv-persist` is honest everywhere and works on DeepSeek-V4
- `kv_persist_enabled` MUST report what each loaded engine does, not the
  config.
- DeepSeek-V4 persists its exact recovery anchor as one prefix image per
  conversation and hydrates it on a cold miss, on both schedulers. A missing,
  corrupt, or incompatible file MUST only cause a cold prefill.
- If that cannot land in the release, `--kv-persist` on DeepSeek-V4 MUST fail
  at startup with a clear message instead of being ignored.

### D5. Reasoning controls behave the same on every family
- `reasoning_effort`, its `reasoning.{effort,enabled,max_tokens}` aliases, and
  `thinking_token_budget` MUST be accepted on every family with one effort
  table (none/minimal off; low 512; medium 2048; high 8192; xhigh/max no
  ceiling; DeepSeek native tiers mapped, `medium` -> `high`).
- One shared budget enforcer MUST cover every family on `inflight-batched`,
  including Gemma 4. The tool-thinking budget has one meaning on every family.
- Server defaults MUST never cause a 4xx; on `fifo-serial` only an explicit
  client budget is rejected, identically on every family.

### Verification
Each decision is verified by hand on Qwen3.8, DeepSeek-V4, and Gemma 4 through
the HTTP API, `hf2q chat`, and OpenCode, and recorded in `docs/qe/`. No new
unit tests, CI jobs, or release gates are added (owner direction 2026-10-08);
existing tests are updated only where a change breaks them.

## Consequences

### Positive
- A plain `hf2q serve`, a setup-configured serve, `hf2q chat`, and the
  launchers behave identically for a given model.
- Harnesses stop retry-looping on errors that can never succeed.
- One code location defines each family's behavior; built-in values move with
  the binary when a family is re-qualified.
- DeepSeek-V4 sessions resume after a restart in seconds instead of minutes.

### Negative
- Operators lose persistent `config.toml` overrides for the three retired
  keys (CLI flags remain).
- Some previously accepted requests now get a 400 (contradictory reasoning
  controls, reasoning on non-thinking
  templates), matching OpenAI behavior.
- DeepSeek-V4's cold required-tool latency must be re-checked against
  ADR-042's 60-second ceiling with the 512-token budget.

### Neutral
- Qwen's required-tool turns use the tool-thinking ceiling instead of the
  base ceiling.
- `fifo-serial` remains available as an explicit compatibility mode.

## Amends
- ADR-050 (operator settings): precedence, retired keys, slot range,
  reasoning controls.
- ADR-045 (setup): five stable defaults restored; 4 slots; port 8081.
- ADR-040 (scheduler): default scheduler, slot range, Gemma cross-slot
  admission default, serve error contract section.
- ADR-042 (DeepSeek-V4): recovery-anchor persistence, reasoning tiers,
  retired 8-token tool budget.
- ADR-044 (Qwen speculation and budgets): shared enforcer, fifo-serial rule.
- ADR-052 §5: `length` finish for budget-exhausted grammars.
- ADR-027 and ADR-017: encoder session default, per-family persistence status.

## Links
- Issue #278 (consistency audit), #256, milestone v0.1.24
- ADR-005 (API contract, historical), ADR-017, ADR-027, ADR-040, ADR-042,
  ADR-044, ADR-045, ADR-050, ADR-052, ADR-061
