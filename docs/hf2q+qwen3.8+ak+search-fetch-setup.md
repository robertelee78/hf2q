# hf2q + Qwen3.8 + AK + search/fetch: complete setup

This guide installs a local abliterated Qwen3.8-27B, serves its verified text
and vision artifacts with hf2q, and connects it to a fully functional OpenCode
Build agent with Agentic Kit and web tools.

The download is 17.4 GiB and resumes if interrupted. You need an Apple Silicon
Mac, at least 20 GiB of free disk space, Node.js, `jq`, and `curl`. The complete
multimodal path was validated on an Apple M5 Max with 128 GiB unified memory;
that identifies the validation host, not a minimum-memory claim. The complete
local research stack also uses `uv` and Google Chrome; its installer checks and
explains either missing prerequisite.

## 1. Install hf2q

```bash
curl -fsSL https://hf2q.us/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
hf2q setup --accept-defaults
hf2q doctor
```

The standalone installer supplies the signed and notarized binary. It does not
require Rust or a local build.

## 2. Download the verified hf2q Q4_K_M pair

The repository also contains GGUFs made for the peer and L40 GPUs. Do not use
those files with this guide. In particular,
`qwen38-abliterated-sft-q5_k_m.gguf` cannot execute hf2q's direct GPU embedding
path, and `mmproj-qwen38-f16.gguf` is not paired with the hf2q Q4 text model.

Download the two hf2q artifacts and their checksums from the immutable artifact
commit:

```bash
MODEL_DIR="$HOME/.local/share/hf2q/models/qwen3.8"
HF_REV="40d771ee15d826017f297261f5bedcf2c32cf4c2"
HF_BASE="https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT/resolve/$HF_REV/gguf"
mkdir -p "$MODEL_DIR"

for FILE in \
  qwen38-abliterated-sft-hf2q-q4_k_m.gguf \
  qwen38-abliterated-sft-hf2q-q4_k_m-mmproj.gguf \
  hf2q-q4_k_m-SHA256SUMS.txt \
  hf2q-q4_k_m-manifest.json
do
  curl -fL -C - -o "$MODEL_DIR/$FILE" "$HF_BASE/$FILE"
done

(
  cd "$MODEL_DIR"
  shasum -a 256 -c hf2q-q4_k_m-SHA256SUMS.txt
)
```

Both lines must print `OK`. If either fails, delete only the named failed file
and rerun this block. The text GGUF embeds the exact projector digest, so hf2q
also rejects a mixed or damaged pair before serving.

The published hashes are:

- text: `1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a`
- projector: `463b264713f8e081f0fae753c80d8089308e01b1e2ac0948dd9966d0711d8f1b`

The model repository's [GGUF manifest](https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT/blob/main/GGUFs.md)
records sizes, conversion provenance, and the multimodal validation evidence.

## 3. Serve the model

Use the same explicit port that the client will use. The four-slot SlotAware
path below is the qualified 128 GiB profile for concurrent coding-harness
work; the smaller one-slot override is documented immediately after it.

```bash
MODEL_DIR="$HOME/.local/share/hf2q/models/qwen3.8"
MODEL="$MODEL_DIR/qwen38-abliterated-sft-hf2q-q4_k_m.gguf"
MMPROJ="$MODEL_DIR/qwen38-abliterated-sft-hf2q-q4_k_m-mmproj.gguf"
STATE_DIR="$HOME/.local/state/hf2q"
LOG="$STATE_DIR/qwen38-serve.log"
MAX_SLOTS="${MAX_SLOTS:-4}"
KV_CACHE_BUDGET_BYTES="${KV_CACHE_BUDGET_BYTES:-51539607552}"
mkdir -p "$STATE_DIR"

if lsof -nP -iTCP:8081 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port 8081 is already in use; stop or choose the intended server first" >&2
  lsof -nP -iTCP:8081 -sTCP:LISTEN >&2
  exit 1
fi

nohup env \
  HF2Q_DEFAULT_REPETITION_PENALTY=1.05 \
  HF2Q_DEFAULT_THINKING_TOKEN_BUDGET=2048 \
  HF2Q_DEFAULT_TOOL_THINKING_TOKEN_BUDGET=512 \
  HF2Q_TQ_KV=1 \
  HF2Q_ENCODER_SESSION=1 \
  HF2Q_FFN_TERMINAL_K_BATCH=8 \
  HF2Q_QWEN_SPECULATION=auto \
  HF2Q_DECODE_MVN=0 \
  HF2Q_DECODE_MV_EXT=1 \
  HF2Q_QWEN_GQA_Q2=auto \
  hf2q serve \
  --model "$MODEL" \
  --mmproj "$MMPROJ" \
  --host 127.0.0.1 \
  --port 8081 \
  --overflow-policy reject \
  --scheduler inflight-batched \
  --max-slots "$MAX_SLOTS" \
  --kv-cache-budget-bytes "$KV_CACHE_BUDGET_BYTES" \
  --operator-ui plain \
  > "$LOG" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$STATE_DIR/qwen38-serve.pid"

READY=0
for _ in $(seq 1 300); do
  if curl --connect-timeout 1 --max-time 2 -fsS \
    http://127.0.0.1:8081/readyz >/dev/null 2>&1
  then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "server exited during startup; inspect $LOG" >&2
    exit 1
  fi
  sleep 2
done
test "$READY" -eq 1 || {
  echo "server did not become ready; inspect $LOG" >&2
  exit 1
}
```

This is the canonical 128 GiB OpenCode profile: four independent agent slots
with a shared 48 GiB physical KV high-water. For a smaller-memory host, start
with `MAX_SLOTS=1 KV_CACHE_BUDGET_BYTES=12884901888` before running the block;
every slot still advertises the model's full logical context.

The environment settings are part of the qualified profile, not random tuning:
the repetition penalty prevents long coding sessions from collapsing into
loops; the two thinking budgets preserve an answer after reasoning and keep
tool continuations short; TQ KV and encoder sessions preserve exact prefix
reuse; and Qwen3.8 exact speculation plus the width-four K-quant path accelerates
eligible decoding without changing target-model output. OpenCode manages its
own compaction, so `--overflow-policy reject` prevents the server from silently
rewriting conversation history.

The PID file belongs to this exact launch. Do not use a broad `pkill` command;
it can stop unrelated hf2q work.

## 4. Prove text, streaming, and vision before OpenCode

`/readyz` proves that startup completed. It does not replace a real generation
test. The following block requires a nonempty unary completion, a valid SSE
completion ending in `[DONE]`, and a correct answer about an embedded red PNG.

```bash
API_ROOT="http://127.0.0.1:8081"
MODEL_ID="$(curl -fsS "$API_ROOT/v1/models" |
  jq -er '.data | map(select(.loaded == true)) | .[0].id')"

UNARY_REQUEST="$(jq -n --arg model "$MODEL_ID" '{
  model: $model,
  messages: [{role: "user", content: "Write a Rust function that adds two i64 values."}],
  temperature: 0,
  max_tokens: 256,
  stream: false,
  hf2q_enable_thinking: false
}')"
UNARY_RESPONSE="$(curl -fsS "$API_ROOT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$UNARY_REQUEST")"
printf '%s\n' "$UNARY_RESPONSE" |
  jq -e '.choices[0].message.content | strings | length > 0' >/dev/null

SSE_FILE="$(mktemp -t hf2q-sse.XXXXXX)"
SSE_REQUEST="$(jq -n --arg model "$MODEL_ID" '{
  model: $model,
  messages: [{role: "user", content: "Reply with exactly: stream works"}],
  temperature: 0,
  max_tokens: 32,
  stream: true,
  hf2q_enable_thinking: false
}')"
curl -NfsS "$API_ROOT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$SSE_REQUEST" > "$SSE_FILE"
grep -q '^data: \[DONE\]' "$SSE_FILE"
SSE_TEXT="$(sed -n 's/^data: //p' "$SSE_FILE" |
  grep -v '^\[DONE\]' |
  jq -rs '[.[] | .choices[0].delta.content // empty] | join("")')"
test -n "$SSE_TEXT"
rm "$SSE_FILE"

RED_PNG="iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAABEElEQVR4Ae3AA6AkWZbG8f937o3IzKdyS2Oubdu2bdu2bdu2bWmMnpZKr54yMyLu+Xa3anqmhztr1a8+5ZZb+F+M4H83gv/dCP53I/jfjeB/N4L/3Qj+dyP4343gfzeC/90I/ncj+N+N4H83gv/dCP53I/jfjeB/N4L/3Qj+dyP4343gfzeC/90I/ncj+N+N4H83gv/dCP53I/jfjeB/N4L/3Qj+dyP4343gfzeC/90I/ncj+N+N4H83gv/dCP53I/jfjeB/N4L/3Qj+dyP4343gfzeC/90I/ncj+N+N4H83gv/dCP53I/jfjeB/N4L/3Qj+dyP4343gfzeC/90I/ncj+N+N4H83gv/dCP534x8BmV0Bmx29tGQAAAAASUVORK5CYII="
VISION_REQUEST="$(jq -n \
  --arg model "$MODEL_ID" \
  --arg image "data:image/png;base64,$RED_PNG" '{
    model: $model,
    messages: [{role: "user", content: [
      {type: "text", text: "What is the dominant color? Answer with one color word."},
      {type: "image_url", image_url: {url: $image}}
    ]}],
    temperature: 0,
    max_tokens: 64,
    stream: false,
    hf2q_enable_thinking: false
  }')"
VISION_RESPONSE="$(curl -fsS "$API_ROOT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$VISION_REQUEST")"
printf '%s\n' "$VISION_RESPONSE" |
  jq -er '.choices[0].message.content' |
  grep -qi 'red'

echo "text, SSE, and vision checks passed for: $MODEL_ID"
```

Do not continue if any check fails. Read the server log and fix that failure
first; OpenCode retries otherwise obscure the original runtime error.

## 5. Install full Agentic Kit and OpenCode

Run this from the Git repository where you want Agentic Kit's project files.
Full project setup may replace existing agent configuration, so commit or back
up any existing project-level agent files first.

```bash
npm install -g opencode-ai @pacphi/agentic-kit@next
ak setup --yes
ak setup --opencode --yes
ak status
```

The first `ak setup` performs full machine and project setup. The second pass
adds the OpenCode host integration: MCP servers, skills, lifecycle plugin,
permissions, and converted subagents. `--minimal` is deliberately absent; it
skips project setup and is not the workflow promised by this guide.

## 6. Add only the hf2q provider to OpenCode

Agentic Kit owns its OpenCode integrations. This merge adds the local provider
and selects its model while preserving every existing agent, tool, permission,
plugin, instruction, and MCP setting.

```bash
API_ROOT="http://127.0.0.1:8081"
MODEL_ID="$(curl -fsS "$API_ROOT/v1/models" |
  jq -er '.data | map(select(.loaded == true)) | .[0].id')"
CONFIG="$HOME/.config/opencode/opencode.json"
mkdir -p "$(dirname "$CONFIG")"
[ -f "$CONFIG" ] || printf '{}\n' > "$CONFIG"
cp "$CONFIG" "$CONFIG.$(date +%Y%m%d%H%M%S).bak"
TMP_CONFIG="$(mktemp "$CONFIG.tmp.XXXXXX")"

jq --arg model_id "$MODEL_ID" '
  .provider = (.provider // {})
  | .provider.hf2q = ((.provider.hf2q // {}) + {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local hf2q"
    })
  | .provider.hf2q.options = ((.provider.hf2q.options // {}) + {
      "baseURL": "http://127.0.0.1:8081/v1",
      "apiKey": "local"
    })
  | .provider.hf2q.models = ((.provider.hf2q.models // {}) + {
      ($model_id): {
        "name": "Qwen3.8 27B Abliterated SFT via hf2q",
        "tool_call": true,
        "attachment": true,
        "modalities": {"input": ["text", "image"], "output": ["text"]},
        "reasoning": true,
        "interleaved": "reasoning_content",
        "temperature": true,
        "cost": {"input": 0, "output": 0},
        "limit": {"context": 262144, "output": 8192}
      }
    })
  | .model = ("hf2q/" + $model_id)
' "$CONFIG" > "$TMP_CONFIG" && mv "$TMP_CONFIG" "$CONFIG"

OC_MODEL="hf2q/$MODEL_ID"
opencode --model "$OC_MODEL" --agent build
```

The stock primary `build` agent retains Bash, read/write/edit, task, skill, and
MCP tools. This guide never creates or replaces `agent.assistant`, never sets a
tool-less default agent, and never writes a blanket permission denial.

The model card's [base-prompt sensitivity note](https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT/blob/main/docs/NOTE-base-prompt-sensitivity.md)
is a model-qualification concern, not a reason to disable the coding harness.
OpenCode's stock Build system prompt and tool schemas remain intact here. Any
future prompt reduction must be isolated, exhaustively tested for coding and
tool behavior, and offered only after it passes those gates.

Inside OpenCode, make the Build agent perform a harmless proof: list the current
directory with Bash, read a small tracked file, and report the result. A prose
claim that tools exist is not sufficient; the transcript must contain the tool
calls and successful tool results.

## 7. Install the complete local research stack

This is the durable setup from `search-fetch-setup.md`, packaged as one
installer. It preserves Bash, files, tasks, skills, MCP, and every Agentic Kit
capability. It removes only OpenCode's redundant fetch-only `webfetch` from the
model's tool catalog and replaces it with the stronger `web_fetch`; measured
Qwen3.8 runs otherwise guessed URLs instead of calling search.

- `web_search` searches a curated local SearXNG engine set, removes junk and
  duplicate URLs, then reads the best pages in parallel.
- `web_fetch` turns an exact URL into clean Markdown using a fast static path,
  a warm Chromium path for JavaScript, and a bounded Cloudflare-aware stealth
  fallback only when anti-bot evidence requires it.
- `web_crawl` performs a bounded multi-page crawl with domain and relevance
  filters.
- `web_extract` performs JSON-CSS or semantic extraction.
- `WebSearch`, `WebFetch`, `WebCrawl`, and `WebExtract` aliases keep Ruflo and
  Agentic Kit research workflows compatible.
- Two loopback-only LaunchAgents start the services at login and restart them
  after a crash. There are no API keys and no Docker dependency.

Install Google Chrome if it is absent, then run the installer. The following
convenience command uses Homebrew and fails with a clear manual-install message
when `brew` is unavailable:

```bash
if [ ! -d '/Applications/Google Chrome.app' ]; then
  command -v brew >/dev/null 2>&1 || {
    echo "Install Google Chrome from https://www.google.com/chrome/ and rerun this section." >&2
    exit 1
  }
  brew install --cask google-chrome
fi

curl -fsSL \
  https://raw.githubusercontent.com/robertelee78/hf2q/main/scripts/install_opencode_web_stack.sh \
  | bash
```

This narrow tool replacement is categorically different from the broken guide's
`tools: {"*": false}` and blanket permission denial: the Build agent remains a
fully functional coding agent and gains four research capabilities.

The installer pins the SearXNG revision and every direct Python package,
preserves changed files as timestamped backups, validates its Python and
JavaScript, runs the fetch-routing regression suite, checks both LaunchAgents,
and performs live search and fetch requests. Restart OpenCode once after
installation so it loads the global plugin.

Prove all four capabilities outside OpenCode:

```bash
curl -fsS http://127.0.0.1:11235/healthz |
  jq -e '.ok == true and .browser_warm == true'

curl -fsS --get http://127.0.0.1:8888/search \
  --data-urlencode 'q=hf2q local inference' \
  --data 'format=json' \
  | jq -e 'select((.results | type) == "array" and (.results | length) > 0) |
      {results: (.results | length), unresponsive_engines}'

curl -fsS -X POST http://127.0.0.1:11235/fetch \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/","mode":"auto","max_chars":2000}' \
  | jq -e 'select(.ok == true and (.markdown | length) > 0) |
      {ok, title, via, chars: (.markdown | length)}'

curl -fsS -X POST http://127.0.0.1:11235/crawl \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://docs.crawl4ai.com/","max_depth":2,"max_pages":6,"allowed_domains":["docs.crawl4ai.com"],"query":"extraction strategy"}' \
  | jq -e 'select(.ok == true and .crawled > 0) |
      {ok, crawled, pages: [.pages[] | {url, title, depth}]}'

curl -fsS -X POST http://127.0.0.1:11235/extract \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://docs.crawl4ai.com/core/quickstart/","strategy":"json_css","schema":{"baseSelector":"h1","fields":[{"name":"title","type":"text"}]}}' \
  | jq -e 'select(.ok == true and (.data | length) > 0)'
```

Then start a new OpenCode Build session and require actual calls to
`web_search`, `web_fetch`, `web_crawl`, and `web_extract`. Each transcript must
contain a successful tool result. Tool names in resolved configuration or a
healthy HTTP endpoint are necessary diagnostics, not end-to-end proof.

The services bind only to `127.0.0.1`; do not expose ports 8888 or 11235 to a
network. The stealth fallback does not provide credentials, bypass paywalls, or
guarantee access to every protected site. Use it only for content you are
authorized to access and respect site policies and rate limits.

## Optional: convert the pair yourself

The download above is the known-good hf2q artifact. To reproduce it from the
immutable Hugging Face source, plan for 100 GiB of free disk space:

```bash
hf2q convert jenerallee78/Qwen3.8-27B-Abliterated-SFT \
  --revision 08c2f075b43bc06456382db6b918a3dcabdcf4dd \
  --quant q4_k_m \
  --output "$HOME/.local/share/hf2q/models/qwen3.8/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf"
```

For this multimodal source, the command produces the text GGUF, the derived
`Qwen3.8-27B-Abliterated-SFT-Q4_K_M-mmproj.gguf`, and a provenance receipt for
each artifact. Keep the pair and both receipts together. Use `--text-only` only
when omitting vision is intentional. See [Converting a model](converting-a-model.md).

## Stop or troubleshoot

Temporarily turn off every search/fetch/crawl/extract tool and both background
services without removing data:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/robertelee78/hf2q/main/scripts/install_opencode_web_stack.sh \
  | bash -s -- --disable
```

Restart OpenCode after disabling so the current session unloads the plugin.
Turn the same stack back on with `--enable`, or inspect it with `--status`:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/robertelee78/hf2q/main/scripts/install_opencode_web_stack.sh \
  | bash -s -- --enable

curl -fsSL \
  https://raw.githubusercontent.com/robertelee78/hf2q/main/scripts/install_opencode_web_stack.sh \
  | bash -s -- --status
```

For full removal, `--uninstall` unloads only the two named LaunchAgents and
moves the plugin, services, environments, plists, and logs into a timestamped
folder under `~/.Trash`. Nothing is permanently deleted, and hf2q, OpenCode,
Agentic Kit, their shared config, and the downloaded model are untouched:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/robertelee78/hf2q/main/scripts/install_opencode_web_stack.sh \
  | bash -s -- --uninstall
```

Stop only the server started by this guide:

```bash
PID_FILE="$HOME/.local/state/hf2q/qwen38-serve.pid"
MODEL="$HOME/.local/share/hf2q/models/qwen3.8/qwen38-abliterated-sft-hf2q-q4_k_m.gguf"
[ -r "$PID_FILE" ] || {
  echo "no server PID file found at $PID_FILE" >&2
  exit 1
}
SERVER_PID="$(cat "$PID_FILE")"
SERVER_COMMAND="$(ps -p "$SERVER_PID" -o command= 2>/dev/null || true)"
case "$SERVER_COMMAND" in
  *"hf2q serve"*"--model $MODEL"*"--port 8081"*)
    kill "$SERVER_PID"
    rm "$PID_FILE"
    ;;
  "")
    echo "server is already stopped; removing stale PID file"
    rm "$PID_FILE"
    ;;
  *)
    echo "refusing to stop PID $SERVER_PID because it is not this guide's server:" >&2
    echo "$SERVER_COMMAND" >&2
    exit 1
    ;;
esac
```

Useful diagnostics:

- Server startup or generation failed: inspect
  `~/.local/state/hf2q/qwen38-serve.log`, then repeat section 4.
- Projector binding failed: download both hf2q-named files again from the same
  pinned artifact commit; do not substitute `mmproj-qwen38-f16.gguf`.
- OpenCode lost tools: run `opencode debug config` and look for an agent-level
  `permission: "deny"` or old `tools: {"*": false}` left by another config.
  Neither is written by this guide.
- Agentic Kit is incomplete: run `ak status`, then repeat both full setup
  commands from section 5 in the intended project.
- Research tools are missing: restart OpenCode, run `opencode debug config`,
  check both LaunchAgents with
  `launchctl list | grep 'com.opencode.\(searxng\|crawl4ai\)'`, and repeat the
  actual tool calls from section 7.

Acceptance boundaries live in [the shipping contract](shipping-contract.md).
