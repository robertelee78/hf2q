# Diagnostic chat

`hf2q chat` is a deliberately small terminal client for troubleshooting the
model and hf2q's OpenAI-compatible server without another client harness in
the way. It keeps ordinary shell scrollback and holds conversation context
only until the process exits.

```bash
# Discover a live local hf2q server, or start one when none is available
hf2q chat

# Select or load a particular model
hf2q chat --model /path/to/model.gguf

# Use any explicitly named OpenAI-compatible endpoint
hf2q chat --url http://127.0.0.1:9123 --model model-id
```

On macOS, `hf2q serve` advertises its actual bound port through machine-local
DNS-SD. Chat verifies discovered candidates over loopback; it does not scan
ports or use a runtime registry file. One verified server is selected
automatically and multiple servers produce a numbered picker. A manually
started server is discoverable in the same way as one started by chat.

If chat starts a server, it gracefully stops that child on exit. A server that
was already running is always left alone. Use `--keep-serving` or `/detach` to
leave a chat-started server running too.

## Session controls

- `/new` clears the in-memory transcript.
- `/model` selects another advertised model and begins a new transcript.
- `/thinking auto|on|off` controls the existing hf2q thinking override.
- `/status` prints the current endpoint, model, session, and available pool
  status.
- `/detach` relinquishes ownership of a server started by this chat.
- `/quit` exits.

The client sends no implicit system prompt, tool definitions, sampling knobs,
or template arguments. Optional `--system`, `--temperature`, `--top-p`,
`--max-tokens`, `--seed`, and `--reasoning-effort` flags add exactly the named
request fields. `HF2Q_AUTH_TOKEN`, when set, is used as the bearer token but is
never published in discovery metadata. Because machine-local DNS-SD candidates
are not authenticated, automatic discovery is disabled while that variable is
set; use `--url` to explicitly name the endpoint that may receive the token.

When the endpoint advertises hf2q's diagnostic capability, chat adds the
capability-declared `x-hf2q-diagnostic-no-evict: 1` HTTP header. This does not
change the OpenAI JSON body; it makes a concurrent residency change fail with
409 instead of silently evicting a model.

Reasoning is visibly separated when the server emits it. Structured tool
calls are displayed but never executed. A footer reports server-provided
usage and timing—prompt/cached/output/reasoning tokens, TTFT, prefill/decode
rates, and pool information when available. Chat does not silently enable
logprobs or calculate perplexity because those change the diagnostic workload.

## Model switching

Diagnostic activation never silently evicts another resident model. If the
requested model fits, hf2q loads it through the existing server and pool. If
it would evict another model, chat shows the exact conflict and offers an
explicit switch. A confirmed switch keeps the same server process, stops new
work for the victim model, drains its active requests, spills configured KV
state, shuts down that model worker, and only then loads the requested model.

An explicit `--url` needs only OpenAI-compatible chat and model-list routes for
basic use. The switch operation appears only when that endpoint exposes the
hf2q lifecycle capability.

The hf2q-specific control routes are `GET /hf2q/v1/runtime` and
`POST /hf2q/v1/models/activate`. They use the same Bearer-auth middleware as
the OpenAI routes.

See [ADR-047](ADR-047-diagnostic-chat.md) for the discovery, ownership,
admission, and failure contracts.
