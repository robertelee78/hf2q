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

# A repository offers receipt-backed local GGUFs before hosted GGUFs.
# Use --quant or --artifact for non-interactive local-first selection.
hf2q chat --model owner/model
hf2q chat --model owner/model --quant Q6_K
hf2q chat --model owner/model --artifact gguf/model-q6_k.gguf

# Add a bounded server-local receipt root for a manually launched server
hf2q serve --port 9123 --model-dir /srv/hf2q-models

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
leave a chat-started server running too. Ctrl-C, terminal loss, and abnormal
parent exit also stop the chat-owned process group. Child server logs are
captured rather than inherited, so a stopped chat cannot leave a download
progress bar painting over the shell prompt.

For a Hugging Face repository, diagnostic chat first asks the hf2q server for
schema-v3 conversion receipts and canonical managed-cache entries. The server
automatically inventories its startup directory's `models/` tree; repeatable
`serve --model-dir DIR` flags add explicit roots. A matching `--quant` or
`--artifact` selects a unique local artifact without contacting the Hub. With
no selector, local rows appear first and `Browse hosted artifacts` makes Hub
metadata access explicit. Paths, receipt-recorded paths, and output digests
never enter the TUI.

If hosted browsing is selected, the current bridge selects Q3_K_M, Q4_K_M,
Q5_K_M, Q6_K, and Q8_0. BF16, split GGUFs, and mmproj companions are shown
with an unavailable reason. The catalog's
quant is a filename-derived hint, not a compatibility claim. After selection,
hf2q verifies local artifacts against their full receipt/cache SHA-256, or
verifies hosted downloads against pinned Hub commit, filename, byte size, LFS
SHA-256, and GGUF header, before pool publication. Source conversion remains
outside this diagnostic flow and is never its silent default.

A unique resident repository match connects without disk or Hub access.
Receipt-backed and canonical managed-cache candidates are next. Legacy cache
metadata without canonical emitted-artifact authority remains excluded.

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

The hf2q-specific control routes are `GET /hf2q/v1/runtime`, server-local
`GET /hf2q/v1/models/local-artifacts`, metadata-only
`GET /hf2q/v1/models/catalog`, and `POST /hf2q/v1/models/activate`. They use the
same Bearer-auth middleware as the OpenAI routes. Local SHA verification and
hosted payload transfer run behind request-cancellable hf2q process boundaries;
cancelling an external chat request stops preparation without stopping the
pre-existing server. Local verification is bounded to one active child;
metadata helpers and hosted transfers are independently bounded to two, and
opaque activation selections expire after ten minutes.

Embedding activation is explicit. Query either artifact route with
`kind=embedding`, then call the activation route with `kind: "embedding"`, the
returned `candidate_id`, and the exact `model` repository. `action: "switch"`
also requires the resident embedding generation as `expected_revision`.
Load/switch never accept a bare local path; `--embedding-model` is the
startup-only path surface. Omitting `kind` preserves generative activation and
never infers embedding behavior from a filename or architecture.

Shell completion files are generated snapshots of the clap grammar. After
upgrading from a build that predates `chat`, regenerate and re-source them with
`hf2q completions --shell zsh` (or `bash`/`fish`).

See [ADR-047](adr/ADR-047-diagnostic-chat.md) for the discovery, ownership,
admission, and failure contracts.
