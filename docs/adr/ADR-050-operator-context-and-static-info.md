# ADR-050: Operator context, shared KV budget, and static GGUF serving preflight

- **Status:** Accepted; implementation and release proof tracked below
- **Date:** 2026-08-23
- **Supersedes:** the dead `serve --max-seq-len` surface, the
  `HF2Q_DEEPSEEK_MAX_SEQ_LEN` serving override, and the raw
  `--kv-cache-budget-bytes` / `HF2Q_KV_CACHE_BUDGET_BYTES` surface, plus the
  environment-only `HF2Q_KV_PERSIST_BUDGET_BYTES` disk ceiling and the
  family-only `HF2Q_DEEPSEEK4_REQUIRED_TOOL_THINKING_TOKEN_BUDGET` default
- **Amends:** ADR-040 full-context slots, ADR-042 DeepSeek serving, and the
  schema-v2 setup contract

## Problem and user job

An operator can read a GGUF that advertises a 1,048,576-token context yet had
no honest hf2q control for serving it at 262,144 tokens. The only context flag
shown by `--help` was explicitly reserved and unused. DeepSeek had a hidden,
family-specific environment override that silently clamped at the model
maximum. This made memory planning dependent on source archaeology and shell
state rather than the CLI and setup file.

The same operator needs to answer, before loading tensor data or initializing
Metal:

- what family and maximum context hf2q detects;
- what context, scheduler, slots, and KV budget the intended serve invocation
  will actually use;
- whether the text GGUF can consume vision and whether an explicitly supplied
  projector is compatible; and
- whether static serving preflight is ready or the exact reason hf2q will
  reject the artifacts.

## Hypothesis, spike, and reformulation

The initial hypothesis was that the existing `--max-seq-len` flag could be
wired through. Source tracing falsified that assumption: it had a 4,096-token
default, no production consumer, and contradicted the per-model GGUF default.
The live DeepSeek path instead read `HF2Q_DEEPSEEK_MAX_SEQ_LEN`, defaulted to a
smaller family value, and silently applied `min(requested, declared)`.

The allocation trace also confirmed that logical context and physical KV
residency are independent. Slot-aware caches retain a full logical capacity
for every conversation. A shared byte budget admits actual retained high-water
growth across those slots; dividing context by slot count would change the
model contract and violate ADR-040.

A second source spike found that setup's server behavior was bridged back into
process environment after the read-once investigation snapshot initialized.
The configured repetition penalty therefore did not reliably reach serving.
`generate --kv-bits` had the same late-environment-mutation defect.

The first typed `--kv-bits` spike fed only that investigation snapshot. A
follow-up production-reader audit falsified it: scalar prefill, batched
prefill, decode, slot provisioning, and persistence descriptors still read
`HF2Q_TQ_CODEBOOK_BITS` independently. The corrected design centralizes
typed-CLI-first resolution for all of those consumers and keeps the
read-once environment snapshot only as the development fallback when the real
flag is absent.

The reformulated design uses one typed, environment-free resolver for the
public serving plan, passes the unresolved context request through hot-swap
configuration so each GGUF resolves independently, and gives `info` the same
static planning input as `serve`.

A final Qwen loader trace checked the memory-pressure premise directly. Its
KV caches are provisioned lazily from `cfg.max_position_embeddings`; the
resolved cap is now applied inside the Qwen family loader before any such
cache can be created. The shared outer resolver still rejects an oversized
request from header metadata before tensor weights are loaded.

The first live structured-tool spike also exposed two contradictions in the
existing gate rather than in serving: its recovery result said “one corrected
call” while still requiring two todos, and its continuation claimed that the
user selected JSON even when the generated tool call offered other choices.
Repeated runs reproduced both failures with otherwise valid tool-call JSON.
The gate now states the original cardinality explicitly and returns the first
option the model actually offered. Its schema, recovery, exact-`ACK`, SSE, and
cache-reuse assertions remain strict.

## Decision

### Context

`hf2q serve` and `hf2q info` accept `--ctx <TOKENS>`. Setup schema v2 accepts
an optional `[serve] ctx = <TOKENS>` key but setup leaves it absent by default.
Resolution is:

1. explicit command-line `--ctx`;
2. `[serve] ctx` in the selected `config.toml`; then
3. the maximum declared by `{general.architecture}.context_length` in that
   model's GGUF.

Zero is invalid. An explicit value above the GGUF maximum is a hard error that
names both values and whether the request came from CLI or config. It is never
silently clamped. The check occurs after header parsing and before model tensor
loading or context-linear KV allocation.
Architecture-specific structural minima still fail explicitly; for example,
DeepSeek-V4 cannot serve below its native sliding window, and `info` reports
that same rejection before a runtime load is attempted.

The effective context is the logical prompt-plus-generation limit of **each**
conversation slot. `--max-slots 4 --ctx 262144` means up to four active
conversations, each able to address 262,144 tokens. It never means 65,536
tokens per slot. `max_slots` is named as a maximum because slots are admitted
on demand rather than promising that every slot is simultaneously full.

The live `/v1/models` entry reports the effective limit as `context_length`
and preserves the intrinsic GGUF capability as `max_context_length`.

### Physical KV residency

`--kv-cache-budget <SIZE>` replaces `--kv-cache-budget-bytes`. Setup accepts
the optional string `[serve] kv_cache_budget`. The parser accepts checked bare
bytes and SI/IEC suffixes such as `8GB` and `8GiB`. CLI overrides config. Zero
or omission means no explicit shared ceiling.

This budget is an aggregate physical admission limit across active slots. It
does not change or divide their logical context. If actual high-water demand
cannot be admitted, hf2q queues or rejects explicitly according to the proven
scheduler behavior; it never shortens a conversation invisibly.

### Persistent KV disk budget

`--kv-persist-budget <SIZE>` is the separate on-disk ceiling for data written
under `serve --kv-persist PATH`. Setup accepts the optional string
`[serve] kv_persist_budget`; interactive setup prompts for it and
non-interactive setup exposes `--serve-kv-persist-budget`. CLI overrides
config. The same checked SI/IEC parser is used, zero/omission is unlimited, and
malformed explicit input fails instead of silently becoming unlimited.
An explicit CLI budget requires `--kv-persist PATH` so it cannot silently do
nothing; a setup-level budget may remain dormant until an invocation enables a
persistent store.

The resolved value is threaded to both `DiskBlockStore` and
`Qwen35DiskPersistor`. The former production reader for
`HF2Q_KV_PERSIST_BUDGET_BYTES` is removed. This disk ceiling does not affect a
slot's logical context or the shared in-memory `--kv-cache-budget`.
Production serving also no longer lets `HF2Q_KV_PERSIST=0` silently defeat an
explicit `--kv-persist PATH`; omission of the typed path is the disable
operation. Development-only generation experiments retain their historical
environment activation until that separate surface is redesigned.

### Scheduling and server behavior

`--scheduler` and `--max-slots` remain the typed public controls, with CLI over
setup config. Scheduler spellings in Clap are `fifo-serial` and
`inflight-batched`; TOML retains `fifo_serial` and `inflight_batched`.
Supplying more than one max slot to FIFO is an error instead of an ignored
argument.

The server behavior defaults are now typed:

- `--default-repetition-penalty` / `[serve] repetition_penalty`;
- `--default-thinking-token-budget` / `[serve] thinking_token_budget`; and
- `--default-tool-thinking-token-budget` /
  `[serve] tool_thinking_token_budget`.

CLI overrides config, which overrides the built-in default. A request field
still overrides the server default. Zero preserves the documented disable
semantics for thinking defaults. The former `HF2Q_DEFAULT_*` bridge is removed.
The generic typed tool-thinking default also replaces the former DeepSeek-only
required-tool-thinking environment setting.

The old scheduler, slot, context, and shared-KV environment fallbacks are also
removed. `HF2Q_AUTH_TOKEN` remains an intentional secret-injection exception;
setup must not persist authentication material.

### `hf2q info`

The old Hugging Face source-directory/repository inspector is removed. The
new command is:

```bash
hf2q info --model model.gguf \
  [--mmproj mmproj.gguf] \
  [--ctx 262144] \
  [--scheduler inflight-batched] \
  [--max-slots 4] \
  [--kv-cache-budget 8GiB] \
  [--kv-persist /var/cache/hf2q/kv] \
  [--kv-persist-budget 32GiB]
```

It never searches for a sibling projector. `--mmproj` is explicit so the
report previews one exact serve invocation. Without a projector, a compatible
text model reports that vision is supported but image input requires
`--mmproj`. With one, the command checks the same profile, width, DeepStack,
source, and optional artifact-digest binding used by serving. A projector is
also checked against the same pair-generation lock, schema, and transaction
journal used by serving. It is fully hashed only when the text GGUF declares
an exact projector digest.

Inspection reads the GGUF metadata and tensor directory, validates the strict
family configuration, embedded tokenizer, chat-template contract, required
tensor names/types/shapes that can be proven from headers, and the optional
projector contract. It reports per-slot and worst-case KV estimates plus a
model-file + worst-case-KV host-memory warning; runtime scratch and allocator
overhead are explicitly outside that static estimate.
It does not decode tensor payloads, create an MLX device, upload weights, warm
up Metal, mutate the inspected model directory, or claim runtime execution
proof. A read-only text-inode lock coordinates with paired-artifact writers.
When the text GGUF declares an
exact projector digest, the explicitly supplied projector is streamed as
bytes solely to verify that checksum; this is the disclosed exception to
header/directory-only I/O.

The last line is either `Serve support: ready ...` or
`Serve support: rejected — <exact reason>`. Exit status is zero only for a
statically serve-ready invocation; the top-level error handler must not append
a duplicate diagnostic after that final verdict.

Human-readable output is the only format in this ADR. Structured JSON, TOML,
or CSV output can be added later when a concrete consumer needs it.

### Clap, help, and completion are one contract

Serve and info flatten the same `ServePlanningArgs` definition, so parsing and
help cannot drift. Server-behavior defaults remain serve-only. Decoder and
projector arguments use the existing semantic path completers. Both dynamic
completion and static Bash, Elvish, Fish, PowerShell, and Zsh output must expose
the new flags and omit the removed flags. Parser, rendered-help, static
completion, dynamic-protocol, and model/projector path-completion tests are
release-blocking evidence for this surface.

### Environment inventory boundary

The source-tree audit found 363 distinct lexical `HF2Q_*` names before this
change, including embedded tests. That count is not a 363-option product
surface. The disposition is:

- ordinary operator jobs get typed CLI/config state and no downstream
  environment rereads;
- secrets and shell/package integration may remain environment-driven;
- safe algorithm escape hatches may remain documented while their normal
  production choice becomes a qualified default;
- benchmarks, dumps, profiles, fault injection, unsafe acknowledgments,
  fixture paths, and unfinished experiments remain development controls; and
- canonical-launcher variables that are required for correct qualified
  behavior must become typed family policy/defaults, not permanent shell UX.

The reproducible full inventory and the remaining promotion queue live in
`docs/env-var-inventory.md`. This ADR completes the context, shared-KV,
persistent-KV disk budget, scheduler/slot, server-behavior, and
`generate --kv-bits` correctness slice. Development-generation persistence
activation/root, LCP memory policy, model-pool budgets, and launcher-owned
family policy remain separately bounded follow-up work; they must not be
misrepresented as ordinary supported UX in the meantime.

## Consequences

- A direct `hf2q serve --model model.gguf` uses the GGUF maximum unless the
  selected setup config intentionally caps context.
- A one-million-token model may create significant virtual KV capacity per
  slot. Operators who want lower pressure use `--ctx`; the shared budget is a
  separate protection against aggregate physical growth.
- Existing scripts using removed flags or environment aliases fail visibly and
  must migrate. No compatibility alias preserves misleading behavior.
- `info` can prove static compatibility only. Release acceptance for serving,
  caching, tools, SSE, prefix reuse, and memory/performance still requires the
  applicable real-model hardware gate.

## Validation record

Implementation and source refinement on 2026-08-23:

| Gate | Result |
|---|---|
| `git diff --check` | PASS |
| rustfmt check on every edited Rust file with child recursion disabled | PASS |
| `bash -n` on every edited shell script | PASS |
| `bash scripts/test_getting_started_guide.sh` | PASS |
| `bash scripts/test_shipping_contract.sh` | PASS |
| generated environment-inventory table equals the checked-in 356-name snapshot | PASS |
| `cargo metadata --locked --no-deps --format-version 1` | PASS |
| focused CLI, info, setup, completion, context, budget, persistence, dispatch, and model-list tests | PASS |
| `cargo check --locked --all-targets --all-features` | PASS |
| `cargo build --release --locked` | PASS |
| `cargo test --locked` | PASS: 4,673 main-binary tests passed, 55 ignored; all integration and doc-test targets passed |

Static real-artifact preflight used the release binary and performed no model
load:

| Invocation/artifact | Observed result |
|---|---|
| Gemma text model, no projector | PASS: 262,144-token GGUF maximum per slot; vision capability detected; text-only ready with explicit-projector guidance |
| Gemma text model plus its explicit projector | PASS: text+vision ready |
| Gemma text model plus the Qwen projector | expected exit 3: exact projector-profile mismatch |
| Nomic BERT GGUF | expected exit 3: unsupported architecture with exact reason |
| DeepSeek, no `ctx` in config or CLI | PASS: 1,048,576 effective tokens per slot from GGUF; 27.06 GiB worst-case KV and 127.12 GiB model-plus-KV warning |
| DeepSeek with config `ctx = 131072` | PASS: config selected 131,072 tokens per slot |
| DeepSeek with config plus `--ctx 262144` | PASS: CLI selected 262,144 tokens per slot; 6.91 GiB worst-case KV and 106.96 GiB model-plus-KV warning |
| DeepSeek with `--ctx 1048577` | expected exit 3 before tensor load: CLI request exceeds GGUF maximum 1,048,576 |
| DeepSeek with `--ctx 0` | expected Clap exit 2 |
| canonical DeepSeek launcher with `CHECK_ONLY=1` | PASS: explicit 262,144 context, four full-context slots, 8 GiB shared KV budget, no tensor load |

The DeepSeek artifact was
`/opt/hf2q/models/deepseek4/DeepSeek-V4-Flash-0731-agentic-q2.gguf`,
107,431,343,168 bytes, SHA-256
`936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d`,
on an arm64 host with 128 GiB physical memory. The live server used
`--ctx 262144 --scheduler inflight-batched --max-slots 4
--kv-cache-budget 8589934592 --overflow-policy reject
--default-repetition-penalty 1.0 --default-tool-thinking-token-budget 8`.
Startup reported a 262,144-token logical context for each of four slots, a
131,072-token family cache allocation, and the aggregate 8 GiB residency
budget. `/health` reported 262,144 and `/v1/models` reported
`context_length=262144`, `max_context_length=1048576`.

`BASE_URL=http://127.0.0.1:18081 REPEATS=2 MAX_TOKENS=512
scripts/test_deepseek4_structured_tools.sh` used temperature 0.55, top-p 0.95,
and maximum reasoning effort. It passed required and automatic nested tool
calls, repeated-null recovery, unary and SSE response semantics, exactly one
SSE terminator, terminal tool-result continuation, and prefix reuse. Repeated
question/todo prompts reused 542/487 tokens in both required and automatic
modes; continuation reused 542 tokens. Cached eight-token suffix prefill ran
at roughly 32–35 token/s and decode was roughly 24–30 token/s across the
observed requests; the semantic SSE TTFT was 2.00 seconds on its cold/reset
prefill. The server then completed its KV drain and worker join after SIGINT;
the process exited and port 18081 had no listener.

The final commit, exact-SHA CI, merge, and publication lineage are intentionally
not claimed by this worktree validation record; they remain the completion
phase below.
