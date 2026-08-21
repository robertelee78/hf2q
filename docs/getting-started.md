# Get started with hf2q and Qwen3.8

Zero to a local, uncensored Qwen3.8-27B served by hf2q and driving OpenCode —
three copy-paste blocks. The only long wait is the model download in block 2
(18.2 GiB: minutes on fast broadband, longer on slow links; it resumes if
interrupted).

The guide model is
[`jenerallee78/Qwen3.8-27B-Abliterated-SFT`](https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT),
an Apache-2.0 community checkpoint derived from `Qwen/Qwen3.8-27B`. You
download the model author's own pre-converted GGUF, pinned below to an exact
repository revision and SHA-256; hf2q validates externally produced GGUFs at
serve time. To produce the quantized artifact yourself from the pinned source
revision instead, see [Convert the model
yourself](#convert-the-model-yourself-optional).

## Before you start

- An Apple Silicon Mac.
- About 25 GiB of free disk.
- Node.js + npm (`brew install node`) — for OpenCode and Agentic Kit.
- `jq` (`brew install jq`) — used to *merge into*, never overwrite, your
  OpenCode config.
- Docker Desktop — only if you want the optional local search/scrape services
  at the end.

The hf2q side needs no Rust or compile step, and uses only `curl` and
`shasum`, both preinstalled on macOS.

Validated on an Apple M5 Max with 128 GiB unified memory; lower-memory hosts
are not yet characterized for this model, and 128 GiB is the validation host,
not a claimed formal minimum. Text-only: do not add `--mmproj` or advertise
image input — Qwen3.8 vision remains a separately gated candidate.

## 1. Install hf2q

```bash
curl -fsSL https://hf2q.us/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
hf2q setup --accept-defaults
hf2q doctor
```

The signed, notarized binary lands in `~/.local/bin` (the `export` is only
needed if that directory is not already on your PATH). Setup records the
guide defaults — Q4_K_M conversion, localhost port 8081, inflight-batched
serving with one slot — and downloads or loads nothing.

## 2. Download the model

One file is everything hf2q needs: the GGUF embeds the weights, the
tokenizer, and the chat template.

```bash
MODEL="$HOME/.local/share/hf2q/models/qwen3.8/qwen38-abliterated-sft-q5_k_m.gguf"
mkdir -p "$(dirname "$MODEL")"
curl -fL -C - -o "$MODEL" \
  "https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT/resolve/fe1ff12a900bcb7021872a901a920dc6713ac583/gguf/qwen38-abliterated-sft-q5_k_m.gguf"
echo "4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e  $MODEL" |
  shasum -a 256 -c -
```

The file is 18.2 GiB (19,535,701,568 bytes), pinned to repository revision
`fe1ff12a900bcb7021872a901a920dc6713ac583`; the last command proves the bytes
match the pinned SHA-256 (`OK`). `-C -` resumes an interrupted download —
rerun the same block after a drop.

## 3. Serve it and connect OpenCode

```bash
# start the model server in the background; first load takes about a minute
MODEL="$HOME/.local/share/hf2q/models/qwen3.8/qwen38-abliterated-sft-q5_k_m.gguf"
nohup hf2q serve --model "$MODEL" > /tmp/hf2q-serve.log 2>&1 &

# install OpenCode + Agentic Kit, then let the kit wire OpenCode
# (ruflo + ruvnet-brain MCP, skills, subagents) while the model loads
npm install -g opencode-ai @pacphi/agentic-kit@next
(cd "$HOME" && ak setup --yes && ak setup --opencode --yes)

# add the hf2q provider and the assistant agent — merges into your existing
# config, removes nothing, and leaves a timestamped backup
CONFIG="$HOME/.config/opencode/opencode.json"
mkdir -p "$(dirname "$CONFIG")"
[ -f "$CONFIG" ] || echo '{}' > "$CONFIG"
cp "$CONFIG" "$CONFIG.$(date +%Y%m%d%H%M%S).bak"
jq '
  .provider = ((.provider // {}) + {
    "hf2q": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local hf2q",
      "options": { "baseURL": "http://127.0.0.1:8081/v1", "apiKey": "local" },
      "models": {
        "Release Qwen38 E2": {
          "name": "Qwen3.8 27B Abliterated SFT via hf2q",
          "tool_call": true,
          "reasoning": true,
          "interleaved": "reasoning_content",
          "temperature": true,
          "limit": { "context": 262144, "output": 8192 }
        }
      }
    }
  })
  | .agent = ((.agent // {}) + {
    "assistant": {
      "description": "Minimal-frame Qwen3.8 agent (model-card measured configuration)",
      "mode": "primary",
      "model": "hf2q/Release Qwen38 E2",
      "prompt": "You are a helpful assistant.",
      "temperature": 0,
      "tools": { "*": false },
      "permission": "deny"
    }
  })
  | .default_agent = "assistant"
' "$CONFIG" > "$CONFIG.tmp" && mv "$CONFIG.tmp" "$CONFIG" || {
  rm -f "$CONFIG.tmp"
  echo "OpenCode config at $CONFIG was not valid JSON; left untouched" >&2
}

# wait for the model, then go
until curl -fsS http://127.0.0.1:8081/readyz > /dev/null 2>&1; do sleep 2; done
opencode
```

What just happened, and why:

- `hf2q serve` runs in the background on `127.0.0.1:8081`; logs at
  `/tmp/hf2q-serve.log`; stop it later with `pkill -f "hf2q serve"`. The model
  loads while npm and `ak setup` run, so nothing waits idle.
- `ak setup` is run from your home directory because inside a git repo it also
  runs *project* setup. `--yes` accepts the printed defaults so the block
  pastes cleanly; see [agentic-kit](https://github.com/pacphi/agentic-kit) for
  what each step does.
- The served model ID is `Release Qwen38 E2`, read from the GGUF's embedded
  name — no lookup needed. Serving a different file? Get its ID from
  `curl -fsS http://127.0.0.1:8081/v1/models`.
- The `jq` step adds `provider.hf2q` and an `assistant` agent and makes that
  agent the OpenCode default. Your existing providers, agents, MCP servers,
  and permissions are untouched, and the previous config sits next to it as
  `opencode.json.<timestamp>.bak`. Invalid JSON is left untouched, not
  replaced.
- **The default agent is deliberate, and measured.** The model card's
  [frame research](https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT/blob/main/docs/NOTE-base-prompt-sensitivity.md)
  shows that a stock coding-harness frame (multi-KB system prompt, dozens of
  tool schemas) silently moves this model back toward refusal on sensitive
  requests, and that counter-instructions do not fix it. `assistant`
  reproduces the card's measured configuration: minimal prompt, `temperature:
  0`, and no tool schemas in the request (`tools: {"*": false}` — permission
  gates execution, not context). For file-editing and shell work, switch to
  the `build` agent (Tab) — that is the correct lane for tool work.
- Run `opencode` from a clean directory: OpenCode injects the `AGENTS.md` /
  `CLAUDE.md` chain found above your cwd into the system frame, and stray ops
  docs are frame mass on every request.
- The embedded chat template defaults to thinking mode; the card's measured
  configuration is thinking-off. Through the raw API you can state that
  explicitly (below); OpenCode displays the reasoning stream either way.

Raw API proof, no OpenCode needed:

```bash
curl -fsS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Release Qwen38 E2","messages":[{"role":"user","content":"Write a Rust function that adds two i64 values."}],"temperature":0,"max_tokens":256,"hf2q_enable_thinking":false}'
```

For streaming, add `"stream":true` and consume the SSE stream to `[DONE]`.

## Optional: web search and scraping (SearXNG, Crawl4AI, Firecrawl)

These tools serve the tool-enabled agents (`build`); the default `assistant`
agent intentionally keeps its request free of tool schemas. Everything here is
merge-only, same as block 3.

Start the local services (Docker), and mint a token for Crawl4AI:

```bash
# SearXNG — private local search on 127.0.0.1:8888, JSON API enabled
mkdir -p "$HOME/.local/share/searxng"
cat > "$HOME/.local/share/searxng/settings.yml" <<EOF
use_default_settings: true
server:
  secret_key: "$(openssl rand -hex 32)"
search:
  formats: [html, json]
EOF
docker run -d --name searxng --restart unless-stopped \
  -p 127.0.0.1:8888:8080 \
  -v "$HOME/.local/share/searxng:/etc/searxng" searxng/searxng

# Crawl4AI — local scraping/crawling with its native MCP server on 11235
CRAWL4AI_API_TOKEN="$(openssl rand -hex 32)"
echo "$CRAWL4AI_API_TOKEN" > "$HOME/.config/hf2q-crawl4ai-token"  # reuse on restarts
docker run -d --name crawl4ai --restart unless-stopped --shm-size=1g \
  -p 127.0.0.1:11235:11235 \
  -e CRAWL4AI_API_TOKEN="$CRAWL4AI_API_TOKEN" unclecode/crawl4ai:latest
```

Wire all three into OpenCode (Firecrawl is hosted — nothing to install; its
keyless free tier covers search/scrape/parse, and you can add an
`Authorization: Bearer <key>` header later for the full toolset):

```bash
CONFIG="$HOME/.config/opencode/opencode.json"
cp "$CONFIG" "$CONFIG.$(date +%Y%m%d%H%M%S).bak"
CRAWL4AI_API_TOKEN="$(cat "$HOME/.config/hf2q-crawl4ai-token")"
jq --arg token "$CRAWL4AI_API_TOKEN" '
  .mcp = ((.mcp // {}) + {
    "searxng": {
      "type": "local",
      "command": ["npx", "-y", "mcp-searxng"],
      "enabled": true,
      "environment": { "SEARXNG_URL": "http://127.0.0.1:8888" }
    },
    "crawl4ai": {
      "type": "remote",
      "url": "http://127.0.0.1:11235/mcp/sse",
      "enabled": true,
      "headers": { "Authorization": "Bearer \($token)" },
      "oauth": false
    },
    "firecrawl": {
      "type": "remote",
      "url": "https://mcp.firecrawl.dev/v2/mcp",
      "enabled": true,
      "oauth": false
    }
  })
  | .permission = ((if (.permission | type) == "object" then .permission else {} end) + {
      "searxng_*": "allow", "crawl4ai_*": "allow", "firecrawl_*": "allow"
    })
' "$CONFIG" > "$CONFIG.tmp" && mv "$CONFIG.tmp" "$CONFIG" || {
  rm -f "$CONFIG.tmp"
  echo "OpenCode config at $CONFIG was not valid JSON; left untouched" >&2
}
```

Prove each piece before reopening OpenCode:

```bash
curl -fsS "http://127.0.0.1:8888/search?q=hf2q&format=json" | jq -r '.results[0].title'
curl -fsS http://127.0.0.1:11235/health
# Firecrawl is hosted, so there is nothing local to probe
```

Then restart `opencode` and, in the `build` agent, ask it to search the web —
the SearXNG and Firecrawl tools answer; Crawl4AI handles JS-heavy pages.

## Convert the model yourself (optional)

The download above is the model author's pre-converted artifact. hf2q's owned
pipeline can produce the artifact from the exact pinned source revision
`08c2f075b43bc06456382db6b918a3dcabdcf4dd` — 51.77 GiB of selected source
files. Because that source is multimodal, the same command automatically
produces a source-matched text GGUF and F16 projector:

```bash
hf2q convert jenerallee78/Qwen3.8-27B-Abliterated-SFT \
  --revision 08c2f075b43bc06456382db6b918a3dcabdcf4dd \
  --output "$HOME/.local/share/hf2q/models/qwen3.8/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf"
```

The omitted `--quant` comes from setup (Q4_K_M). An explicit `--quant` always
wins; without either setup config or the flag, convert fails and asks for a
choice. The measured native Q4_K_M output is 16,810,714,848 bytes
(15.66 GiB).

hf2q uses its in-process Hugging Face client; Python, `huggingface-cli`,
llama.cpp, and an external converter are not involved. The command writes the
text path shown above, derives
`Qwen3.8-27B-Abliterated-SFT-Q4_K_M-mmproj.gguf` beside it, and writes one
conversion receipt beside each artifact. The text GGUF embeds the exact
projector digest, so a missing or mismatched sidecar fails closed at serve
time. Keep both receipts with the pair: they bind both outputs to the same
resolved source and converter revisions while recording each output identity
and quantization (`q4_k_m` and `f16-mmproj`). Plan for 100 GiB free and a
substantially longer download-plus-conversion run than the direct download
above. Use `--text-only` only when omitting vision is intentional.

## What this guide does not do

- It does not compile anything; source builds live in the [README](../README.md#install).
- It does not enable Qwen3.8 vision or speculative MTP decoding.
- It does not remove or replace existing OpenCode configuration — every edit
  is a key-level merge with a timestamped backup.
- It does not add a second model-preparation, model-registry, or session-cache
  subsystem around the existing CLI.

## If something goes wrong

- `setup ... state root permissions must be 0700`: a pre-existing `~/.hf2q`
  directory has loose permissions — `chmod 700 ~/.hf2q` and rerun block 1.
- `hf2q: command not found` in a new terminal: add
  `export PATH="$HOME/.local/bin:$PATH"` to your shell profile.
- `ak setup` asks questions mid-paste: it was run without `--yes`; answer its
  prompts or rerun the block as written.
- Server never becomes ready: read `/tmp/hf2q-serve.log`; the model file is
  incomplete if the SHA-256 check in block 2 did not print `OK`.

See [Converting a model](converting-a-model.md) for the general conversion
reference and [the shipping contract](shipping-contract.md) for the exact
family and scheduler acceptance boundaries.
