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
- `--kv-persist` was silently ignored on DeepSeek-V4 while `/hf2q/v1/runtime`
  reported it enabled (the endpoint reads the config flag, not engine state);
  Qwen's disk restart path runs only on `fifo-serial`.
- `reasoning_effort` worked only on DeepSeek-V4 (and `medium` returned 400
  there, which stock OpenCode sends); thinking budgets were Qwen-only.

Five design studies (D1–D5) traced each area through the code and propose the
decisions below. They are the evidence base for this ADR.

## Decision

### D0. Functional default scheduler (implemented on fix/serve-default-scheduler)
With no `--scheduler` and no `config.toml`, serve MUST use the scheduler and
slot count `hf2q setup` records: `inflight-batched`, 4 slots, from one shared
constant. An over-limit prompt under an explicitly chosen `fifo-serial` MUST be
HTTP 400 `prompt_exceeds_server_limit` naming the fix (the code the
implementation branch landed: the `serial_prompt_limit` sentinel). The
`serial_prompt_limit` substring sentinel is interim: D2's typed error
classification subsumes it when D2 lands.

### D1. One built-in serving profile per family
- Precedence MUST be: request field > CLI `--default-*` > the family's
  built-in value. Setup MUST NOT write behavior profile keys; the
  `[serve] repetition_penalty`, `thinking_token_budget`, and
  `tool_thinking_token_budget` keys are retired (parse warns and ignores).
- Family defaults are the values each canonical launcher already uses, and
  the built-in table states every key per family (an absent key is a
  decision, not an omission): Qwen 1.05 / 2048 / 512 with the encoder
  session on; Gemma 4 1.05 with cross-slot admission (25 ms coalesce) on
  and no thinking budgets; DeepSeek-V4 1.0 with a tool-thinking budget of
  512 and no thinking budget — D5's effort table governs its reasoning
  (owner decision 2026-10-08, see D5).
- Launchers MUST need no `HF2Q_*` variables for qualified behavior.
  `HF2Q_KV_LCP_LONG_RESUME` (inert on the default scheduler) is dropped.
- The default port MUST be 8081 on every path.
- `hf2q info` and `/hf2q/v1/runtime` MUST report each effective value and its
  origin.

### D2. One error contract
- Client mistakes and requests this server can never satisfy MUST be 4xx with
  a stable `code`; 5xx MUST be reserved for server faults. Engine errors MUST
  carry a typed classification instead of substring sentinels.
- `capability_unsupported` becomes 400. Context overflows MUST return the
  existing typed 400 `context_length_exceeded` — the code OpenCode maps to
  its `context_overflow`/compaction path (verified in the OpenCode 1.18
  client: `error.code === "context_length_exceeded"`, or HTTP 413) — never
  a 500/501 from an engine-level cap.
- The stable `code` catalog is the typed `ApiError` code table in
  `src/serve/api/schema.rs`: every new stable code lands there with its
  status and param, and no code is emitted outside it.
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
- On DeepSeek-V4 this works when no `--kv-graft` is bound: a bound graft
  still refuses `--kv-persist` at boot (ADR-059 gate 5's mutual refusal
  stands; the graft-aware disk codec is ADR-059's follow-up, not this
  ADR's).
- DeepSeek-V4 persists its exact recovery anchor as one prefix image per
  conversation and hydrates it on a cold miss, on both schedulers. A missing,
  corrupt, or incompatible file MUST only cause a cold prefill.
- If that cannot land in the release, `--kv-persist` on DeepSeek-V4 MUST fail
  at startup with a clear message instead of being ignored. (Resolved
  2026-10-10: the persistence landed, so this fallback is moot — it remains
  the rule for any future family where the hook cannot ship.)

### D5. Reasoning controls behave the same on every family
- `reasoning_effort`, its `reasoning.{effort,enabled,max_tokens}` aliases, and
  `thinking_token_budget` MUST be accepted on every family with one effort
  table (none/minimal off; low 512; medium 2048; high 8192; xhigh/max no
  ceiling; DeepSeek native tiers mapped, `medium` -> `high`).
- One shared budget enforcer MUST cover every family on `inflight-batched`,
  including Gemma 4. The tool-thinking budget has one meaning on every family.
- Server defaults MUST never cause a 4xx; on `fifo-serial` only an explicit
  client budget is rejected, identically on every family.
- A reasoning control on a family whose template has no thinking channel
  MUST be 400 (OpenAI behavior), identically on every family — never
  accepted-and-ignored.

### Verification
Each decision is verified by hand on Qwen3.8, DeepSeek-V4, and Gemma 4 through
the HTTP API, `hf2q chat`, and OpenCode, and recorded in `docs/qe/`. No new
unit tests, CI jobs, or release gates are added (owner direction 2026-10-08);
existing tests are updated only where a change breaks them.

## Execution state (2026-10-08)

- **D0** is implemented and pushed on `fix/serve-default-scheduler`
  (c556f6d4..baefef4b, cut from `main` @ be17e2fe, unmerged): a fresh
  serve runs `inflight-batched` with 4 slots from one shared constant, the
  over-limit `fifo-serial` rejection is HTTP 400
  `prompt_exceeds_server_limit`, and the four-slot default is recorded in
  the setup docs, README, and ADR-040/045.
- **D2's #256 piece** is implemented but uncommitted on
  `fix/grammar-length-finish` (worktree `len-finish`, cut from `main` @
  be17e2fe): a live grammar or tool call cut by `max_tokens` finishes
  `"length"` with the partial text, never a 500 and never a partial
  `tool_calls` entry; `cargo check --locked --all-targets --all-features`
  is clean. The rest of D2 (embeddings 400, `capability_unsupported` →
  400, typed engine error classification, the context-overflow compaction
  code) and D1/D3/D4/D5 are not started.
- **D1's built-in profiles half** (#288) is implemented on
  `fix/serve-builtin-profiles` (worktree `builtin-profiles`, cut from
  `main` @ d17e959b): the per-family built-in table
  (`src/serve/operator_settings.rs::family_serve_profile`) is applied per
  loaded engine at request time, the retired `[serve]` behavior keys warn
  and are ignored on parse, setup no longer writes them, and Gemma 4's
  cross-slot admission (25 ms coalesce) is the built-in default with the
  `HF2Q_*` names kept as overrides until the launcher cleanup (#289).
  The `hf2q info` / `/hf2q/v1/runtime` per-value origin reporting and the
  launcher simplification remain #289; hands-on verification per
  Verification above is pending.
- **D1's launchers/port/reporting half** (#289) is implemented on
  `fix/serve-launchers-reporting` (worktree `launchers-reporting`, cut
  from `main` @ 4002580a, rebased onto the #314 merge of the built-in
  profiles, uncommitted at handoff): the canonical launchers set only
  `HF2Q_*` variables that gate genuinely non-default behavior (Qwen's
  encoder session, Qwen3.8's K-quant width routing; `HF2Q_TQ_KV`,
  `HF2Q_FFN_TERMINAL_K_BATCH`, Gemma's LCP-resume, cross-slot-admit and
  coalesce defaults are dropped, and the inert `HF2Q_KV_LCP_LONG_RESUME`
  is gone); the built-in port is 8081 on every path (CLI/config fallback,
  `ServerConfig` default, Gemma launcher, README); `hf2q info` reports
  the effective scheduler/slot values and sampling defaults with
  per-value origins; `/hf2q/v1/runtime` reports the serve block
  (scheduler, slots, port, origins) and the measurement snapshot's
  `sampling_default_origins`. The DeepSeek launcher's tool-thinking flag
  moved to the D1 value (512). The origin reporting layers on #314's
  design: `family_serve_profile` stays the value authority and the
  former behavior keys keep no Config origin. The Gemma
  cross-slot/coalesce removals ride on #288's built-in profiles, which
  landed first as #314; hands-on verification per Verification above is
  pending.
- **D5** is implemented on `fix/serve-reasoning-controls` (worktree
  `reasoning-controls`, cut from `main` @ 64a2344f, uncommitted at handoff):
  one effort table (`src/serve/api/reasoning_controls.rs`) accepts
  `reasoning_effort`, the new `reasoning.{effort,enabled,max_tokens}`
  aliases, and `thinking_token_budget` on every family (none/minimal off;
  low 512; medium 2048; high 8192; xhigh/max no ceiling; DeepSeek-V4 tiers
  mapped with `medium` -> `high`, and `medium`/`xhigh` also accepted as
  native `chat_template_kwargs.reasoning_effort` aliases); one shared budget
  enforcer (`resolve_thinking_budget_policy`) covers every family on
  `inflight-batched`, including Gemma 4, whose no-thinking-channel family
  now rejects every reasoning control with one 400 instead of
  accepting-and-ignoring it; server defaults never cause a 4xx (under
  `fifo-serial` they are dropped with a warning; only an explicit client
  budget is a 400, identically on every family — the finding-3
  auto-injection defect); `reasoning_effort` `medium` maps to DeepSeek's
  native `high` tier (stock OpenCode works on every family), and `none` now
  turns thinking off instead of normalizing to DeepSeek's `low`.
  `cargo check --locked --all-targets --all-features` is clean and the
  focused suites pass; hands-on verification per Verification above (all
  three families through the HTTP API, `hf2q chat`, and OpenCode) is
  pending.
- **D4's DeepSeek-V4 half** (#293) is implemented on
  `fix/serve-ds-kv-persist` (worktree `ds-kv-persist`, cut from `main` @
  204b7cba, uncommitted at handoff): `Deepseek4LoadedModel` constructs a
  `Deepseek4DiskPersistor`
  (`src/serve/kv_persist/families/deepseek4_anchor.rs`) from the typed
  `--kv-persist` path at load — an unusable persistence directory fails
  startup with a clear message instead of silently serving unpersisted.
  Every promoted turn anchor (loaded surface AND each SlotAware agent
  session — the shared commit seam) writes ONE prefix image per
  conversation to
  `<kv-persist>/ds4-<fingerprint>/<key>.dsimg`: the anchor snapshot's
  circular-window rows and recurrent compressor pools plus the
  append-only compressed/indexer rows valid at the anchor position (the
  exact state the in-memory RecoveryAnchor resume path relies on the
  live cache holding), with the rendered token ledger and a SHA-256
  checksum (codec:
  `inference::models::deepseek4::cache::anchor_image`). A conversation's
  newer anchor supersedes its strictly shorter prefix images, and the
  typed byte budget evicts oldest-mtime images. On a cold miss
  (`prefill_suffix` and `begin_resumable_cold_prefill`, when the surface
  holds no reusable prefix) the longest image whose token ledger is an
  EXACT proper prefix of the rendered prompt is hydrated into the live
  surface (cache + ledger + recovery anchor), so the ordinary reuse
  machinery prefills only the suffix — SerialFifo directly and SlotAware
  via the cold→cached delegation (`seed_deepseek4_single` reports the
  hydrated plan's cached tokens). A missing, corrupt (checksum), or
  incompatible (shape/context drift, graft rows) file only warns and
  falls back to a cold prefill; corrupt files are best-effort deleted.
  A bound `--kv-graft` still refuses `--kv-persist` at boot (ADR-059
  gate 5, unchanged and re-verified), and the codec refuses graft rows
  in both directions as a backstop. `cmd_serve` registers
  `Deepseek4AnchorSpill` for a DeepSeek-V4 `--model` so #292's
  `family_persist_active` reports the family honestly (the block
  contract stays Skipped by design — the anchor image is
  conversation-shaped, not block-shaped). `cargo check --locked
  --all-targets --all-features` is clean; the focused suites pass
  (`kv_persist` 231, `deepseek4` 134, `persist` 249); the codec's
  serialize→hydrate round trip (window/compressed/indexer/states
  byte-exact, checksum corruption, schedule drift, graft-header
  refusal) was verified on the real Metal path with a throwaway test
  removed before handoff per the no-new-tests direction. Hands-on
  restart-resume verification per Verification above (both schedulers,
  cached-token counts, on the local 107 GiB artifact) is pending — the
  orchestrator runs those gates after this implementation.
- **Integration order:** merge D0 first (the shared scheduler constant and
  error path); commit the #256 piece on `fix/grammar-length-finish` and
  merge it second; D2's remainder re-cuts from `main` after both (it
  touches the same error-classification code); D1/D3/D5 may run in
  parallel after D0 merges; D4 is serialized against ADR-059's DeepSeek
  cache work — both mutate the same DeepSeek snapshot code, one at a time.
- The ADR flips to accepted when D0–D5 are verified per Verification above
  and merged.

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
