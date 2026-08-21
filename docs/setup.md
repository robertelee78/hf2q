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
- the maximum simultaneous active requests for inflight serving;
- localhost-only or LAN binding; and
- the default API port.

Enter retains the displayed recommendation or the current configured value on
a rerun. EOF or interrupted input cancels successfully without changing the
configuration.

The canonical Qwen3.8 guide defaults are Q4_K_M conversion, localhost port
8081, and the inflight-batched scheduler with one active slot. On a fresh
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
  --serve-max-slots 1
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
max_slots = 1
```

Those are the only durable defaults. Model identity, revision, output path,
cache location, source retention, auth tokens, hardware snapshots, and
calibration state remain explicit or owned by their existing commands.

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
built-ins. Scheduler and active-slot precedence is:

1. explicit CLI arguments;
2. existing `HF2Q_SCHEDULER` and `HF2Q_MAX_SLOTS` environment overrides;
3. setup config; and
4. the pre-setup safe built-ins.

An invalid config fails before source download, model load, or listener bind.
If setup records LAN binding (`0.0.0.0`), serve also requires `--auth-token` or
`HF2Q_AUTH_TOKEN`; setup never stores the token.

See [Get started with hf2q and Qwen3.8](getting-started.md) for the complete
tested hf2q workflow using these defaults.
