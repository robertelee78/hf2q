# Get started: hf2q + Qwen3.8 + OpenCode

A local, uncensored Qwen3.8-27B — served by hf2q, driving OpenCode — in three
paste-able blocks. The only long wait is the 18.2 GiB download in block 2
(minutes on fast broadband; it resumes if interrupted).

Requires: an Apple Silicon Mac with ~25 GiB free, Node.js, and `jq`
(`brew install node jq`). Docker only if you want the optional web tools.

*Text-only (no `--mmproj`); validated on an Apple M5 Max with 128 GiB — the
validation host, not a claimed minimum.*

## 1. Install hf2q

```bash
curl -fsSL https://hf2q.us/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
hf2q setup --accept-defaults && hf2q doctor
```

Signed, notarized binary — no Rust, no compile.

## 2. Download the model

The model author's pre-converted GGUF — weights, tokenizer, and chat template
in one file — pinned by revision and SHA-256. hf2q validates externally
produced GGUFs at serve time.

```bash
MODEL="$HOME/.local/share/hf2q/models/qwen3.8/qwen38-abliterated-sft-q5_k_m.gguf"
mkdir -p "$(dirname "$MODEL")"
curl -fL -C - -o "$MODEL" \
  "https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT/resolve/fe1ff12a900bcb7021872a901a920dc6713ac583/gguf/qwen38-abliterated-sft-q5_k_m.gguf"
echo "4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e  $MODEL" |
  shasum -a 256 -c -
```

Download dies? Rerun the block — it resumes. Check prints `FAILED`?
`rm "$MODEL"` and rerun; never serve a file that failed verification.

## 3. Serve it and connect OpenCode

```bash
MODEL="$HOME/.local/share/hf2q/models/qwen3.8/qwen38-abliterated-sft-q5_k_m.gguf"
nohup hf2q serve --model "$MODEL" > /tmp/hf2q-serve.log 2>&1 &

npm install -g opencode-ai @pacphi/agentic-kit@next
(cd "$HOME" && ak setup --yes && ak setup --opencode --yes)

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
' "$CONFIG" > "$CONFIG.tmp" && mv "$CONFIG.tmp" "$CONFIG"

for _ in $(seq 1 90); do
  curl -fsS http://127.0.0.1:8081/readyz > /dev/null 2>&1 && break
  sleep 2
done
curl -fsS http://127.0.0.1:8081/readyz || echo "not ready yet — see /tmp/hf2q-serve.log" >&2
opencode
```

The model loads in the background while OpenCode and Agentic Kit install. The
`jq` step is a key-level **merge** — existing providers, agents, MCP servers,
and permissions stay, with a timestamped backup beside the file (and an
untouched file plus an error if the JSON was unreadable).

The default `assistant` agent is minimal on purpose: the model card's
[frame research](https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT/blob/main/docs/NOTE-base-prompt-sensitivity.md)
measured that a stock coding-harness frame (big system prompt + tool schemas)
moves this model back toward refusal. So: `"You are a helpful assistant."`,
temperature 0, no tool schemas. Switch to the `build` agent (Tab) for
file/shell work, and run `opencode` from a clean directory — `AGENTS.md` files
above your cwd are injected into the frame. (If your home directory is itself
a git repo, run the `ak setup` line from any other directory.)

When OpenCode opens you are on `hf2q/Release Qwen38 E2` via `assistant` — ask
it something; the answer is computed locally. Server logs:
`/tmp/hf2q-serve.log`; stop it with `pkill -f "hf2q serve"`.

Prove it without OpenCode — `hf2q chat` discovers the running server and its
model by itself (`/thinking off` matches the card's measured configuration;
`/status` shows the endpoint; `/quit` exits):

```bash
hf2q chat
```

Or the raw API (`hf2q_enable_thinking: false` does the same there; add
`"stream":true` for SSE):

```bash
curl -fsS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Release Qwen38 E2","messages":[{"role":"user","content":"Write a Rust function that adds two i64 values."}],"temperature":0,"max_tokens":256,"hf2q_enable_thinking":false}'
```

## Optional: web search and scraping (SearXNG, Crawl4AI, Firecrawl)

These serve the tool-enabled `build` agent; `assistant` intentionally sends no
tool schemas. Docker required for the two local services.

```bash
# SearXNG — private local search on 127.0.0.1:8888
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

# Crawl4AI — local scraping, native MCP server on 11235
CRAWL4AI_API_TOKEN="$(openssl rand -hex 32)"
echo "$CRAWL4AI_API_TOKEN" > "$HOME/.config/hf2q-crawl4ai-token"
chmod 600 "$HOME/.config/hf2q-crawl4ai-token"
docker run -d --name crawl4ai --restart unless-stopped --shm-size=1g \
  -p 127.0.0.1:11235:11235 \
  -e CRAWL4AI_API_TOKEN="$CRAWL4AI_API_TOKEN" unclecode/crawl4ai:latest
```

Firecrawl is hosted — nothing to run; its keyless free tier covers
search/scrape/parse. Wire all three into OpenCode (same merge-only pattern):

```bash
CONFIG="$HOME/.config/opencode/opencode.json"
cp "$CONFIG" "$CONFIG.$(date +%Y%m%d%H%M%S).bak"
jq --arg token "$(cat "$HOME/.config/hf2q-crawl4ai-token")" '
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
' "$CONFIG" > "$CONFIG.tmp" && mv "$CONFIG.tmp" "$CONFIG"
```

Prove it, then restart `opencode` and ask the `build` agent to search the web:

```bash
curl -fsS "http://127.0.0.1:8888/search?q=hf2q&format=json" | jq -r '.results[0].title'
curl -fsS http://127.0.0.1:11235/health
```

Firecrawl's full toolset: add an `Authorization: Bearer <key>` header to its
entry. Crawl4AI container restarts must pass the same token from
`~/.config/hf2q-crawl4ai-token`.

## Optional: convert the model yourself

The download above is the author's artifact. hf2q's owned pipeline reproduces
it from the exact pinned source revision and writes a provenance receipt
(`<output>.receipt.json`) — no Python, `huggingface-cli`, or llama.cpp
involved. Plan for 100 GiB free:

```bash
hf2q convert jenerallee78/Qwen3.8-27B-Abliterated-SFT \
  --revision 08c2f075b43bc06456382db6b918a3dcabdcf4dd \
  --output "$HOME/.local/share/hf2q/models/qwen3.8/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf"
```

`--quant` defaults to Q4_K_M from setup. Reference:
[Converting a model](converting-a-model.md).

## Problems?

- `state root permissions must be 0700` → `chmod 700 ~/.hf2q`, rerun block 1.
- `hf2q: command not found` in a new terminal → put
  `export PATH="$HOME/.local/bin:$PATH"` in your shell profile.
- Server never ready → read `/tmp/hf2q-serve.log`; if block 2's check did not
  print `OK`, the file is incomplete — delete it and refetch.
- OpenCode cannot reach the model → the server is not running (a reboot kills
  it); rerun the first two lines of block 3.

Acceptance boundaries live in [the shipping contract](shipping-contract.md).
