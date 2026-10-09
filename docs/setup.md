# Configure hf2q

`hf2q setup` learns the selected Apple Silicon Mac and records a small set of
operator defaults that the existing `hf2q convert` and `hf2q serve` commands
consume. It does not choose or download a model, convert or quantize weights,
start a server, edit another application, or manage a session cache.

## Recommended setup

Run the interactive form:

```bash
hf2q setup
```

Setup reports the selected state root, Apple chip and Metal device, unified
memory, Metal recommended working-set size, macOS and core facts, configured
shell, `RLIMIT_NOFILE`, and containing-volume capacity. It then asks for:

- the default `hf2q convert --quant` selector;
- whether serving should favor long agent/tool prompts or short direct use;
- the maximum simultaneous active requests for inflight serving (accepted
  range 1-8 on every family);
- localhost-only or LAN binding; and
- the default API port; and
- an optional persistent-KV disk ceiling.

Enter retains the displayed recommendation or the current configured value on
a rerun. EOF or interrupted input cancels successfully without changing the
configuration.

The canonical Qwen3.8 guide defaults are Q4_K_M conversion, localhost port
8081, the inflight-batched scheduler with four active slots, and the qualified
agentic serving profile (see the config keys below). On a fresh
config, automation can record those values without prompting; on a rerun the
same flag accepts the current values. Operators using another family should
review its scheduler support or run interactive setup:

```bash
hf2q setup --accept-defaults
```

Or specify every value explicitly:

```bash
hf2q setup \
  --default-quant q4_k_m \
  --serve-host 127.0.0.1 \
  --serve-port 8081 \
  --serve-scheduler inflight-batched \
  --serve-max-slots 4 \
  --serve-kv-persist-budget 32GiB
```

A fresh non-interactive invocation requires `--accept-defaults` or a complete
set of explicit choices. A non-interactive rerun may change only selected
fields; omitted fields retain their current values.

## State root and config

The default state root is `$HOME/.hf2q`. `--state-root` is a global option and
may appear before or after the command:

```bash
hf2q --state-root /Volumes/Private/hf2q setup --accept-defaults
hf2q --state-root /Volumes/Private/hf2q convert MODEL_ID --output model.gguf
hf2q --state-root /Volumes/Private/hf2q serve --model model.gguf
```

The path must be absolute. Convert and serve must receive the same custom root
to consume its config.

Setup writes canonical, bounded UTF-8 TOML at `<state-root>/config.toml`:

```toml
kind = "hf2q.config"
schema_version = 2
package = "hf2q"

[convert]
quant = "q4_k_m"

[serve]
host = "127.0.0.1"
port = 8081
scheduler = "inflight_batched"
max_slots = 4
```

Setup writes no behavior profile keys (ADR-062 D1): the retired
`repetition_penalty`, `thinking_token_budget`, and
`tool_thinking_token_budget` keys are no longer carried, and a config that
still has them parses with a warning and is ignored. Serving behavior
defaults come from the per-family built-in profile (Qwen 1.05 / 2048 / 512,
Gemma 4 1.05 with no thinking budgets, DeepSeek-V4 1.0 with a 512-token
tool-thinking budget); the explicit `hf2q serve --default-*` flags override
the built-ins, and an explicit request value always wins.

Setup intentionally omits logical context and both KV budgets when accepting
defaults. That lets each model use the maximum context declared by its own
GGUF and avoids writing a
universal value that could exceed another model's capability. Interactive
setup prompts for the persistent-store disk ceiling, and non-interactive setup
accepts `--serve-kv-persist-budget <SIZE>`. An operator may also add the keys
deliberately:

```toml
[serve]
ctx = 262144
kv_cache_budget = "8GiB"
kv_persist_budget = "32GiB"
```

`ctx` applies to every conversation slot and is never divided by `max_slots`.
An explicit value above a model's GGUF maximum fails before tensor loading.
`kv_cache_budget` is instead an aggregate physical-residency ceiling shared by
the slots; a bare byte count or a checked SI/IEC value such as `8GB` or `8GiB`
is accepted.
`kv_persist_budget` caps bytes written beneath `serve --kv-persist PATH`; it is
independent of the active-slot residency budget. CLI
`--kv-persist-budget <SIZE>` overrides the config value and requires
`--kv-persist PATH` in that invocation, so a command-line ceiling can never be
silently inactive. A config value may remain dormant until an invocation
enables a store. Zero or omission is unlimited, while malformed explicit
values fail setup or serve.

Model identity, revision, output path, cache location, source retention, auth
tokens, hardware snapshots, and calibration state remain explicit or owned by
their existing commands.

The state root is mode `0700`; `config.toml`, the persistent transaction lock,
and any partial are owned, single-link, same-device regular files at mode
`0600`. Exact reruns preserve the config inode and mtime. Malformed, provisional
schema-1, or future config remains unchanged and fails with an actionable
error. Precommit interruption is retryable; an error after the atomic commit
is reported as durability-unknown and an exact retry revalidates it.

## How commands use the defaults

For conversion, `--quant` wins over config. If no config exists, conversion
continues to require an explicit `--quant`; hf2q does not invent a hidden
quantization default. Source, revision, and output remain explicit.

For serving host and port, precedence is CLI, config, then the pre-setup safe
built-ins. Scheduler, active slots, context, shared KV budget, repetition
penalty, and thinking-default precedence is:

1. explicit CLI arguments;
2. setup config; and
3. the pre-setup safe built-ins or, for context, the model GGUF maximum.

Active slots are family-independent: every family accepts 1-8, the default is
4, and a value above 8 fails the CLI, config load, and the setup prompt with
the accepted range because raising the bound requires a hands-on
qualification at the new width.

The public serving plan does not fall back to hidden `HF2Q_*` variables.
`HF2Q_AUTH_TOKEN` remains the intentional secret-injection exception and is
equivalent to `--auth-token` when that flag is absent.

An invalid config fails before source download, model load, or listener bind.
If setup records LAN binding (`0.0.0.0`), serve also requires `--auth-token` or
`HF2Q_AUTH_TOKEN`; setup never stores the token.

See [Getting started: hf2q + OpenCode + local web research](getting-started.md)
for the complete tested workflow using these defaults.
