# Getting started: hf2q + OpenCode + local web research

This guide takes a fresh Mac from nothing to a working local coding agent:
hf2q serving a verified Qwen3.8-27B model, OpenCode connected to it, Agentic
Kit installed, and a local search/fetch research stack the agent can use.

You need an Apple Silicon Mac, about 20 GiB of free disk space, Node.js,
`jq`, and `curl`. Google Chrome is required later for the research stack's
browser fallback. The flow was validated on an Apple M5 Max with 128 GiB
unified memory; that names the validation host, not a minimum.

## 1. Install hf2q

```bash
curl -fsSL https://hf2q.us/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
hf2q setup
hf2q doctor
```

`hf2q setup` asks five questions and writes `~/.hf2q/config.toml`; every
answer can be changed later by rerunning it:

- **Quantization** — keep the default `q4_k_m`.
- **Optimize for long agent and tool-use prompts** — answer `Y` for agentic
  coding; it selects the batched scheduler and persists the qualified
  agentic serving profile (repetition penalty and bounded thinking budgets)
  into the configuration, so `hf2q serve` inherits it with no environment
  variables to set.
- **Maximum simultaneous active requests** — `4` is a good agentic-coding
  value on a 128 GiB host; `1` is the safe default elsewhere.
- **LAN access** — `N` keeps the server on loopback.
- **Port** — keep the default `8081`.

Non-interactive shells can run `hf2q setup --accept-defaults` instead.
`hf2q doctor` confirms the installation is healthy before you download 17 GiB.

## 2. Prepare and serve the model (terminal 1)

```bash
hf2q serve jenerallee78/Qwen3.8-27B-Abliterated-SFT:Q4_K_M
```

That is the whole model preparation command. hf2q first looks for an exact
hf2q-bound Q4_K_M or one uniquely compatible manually downloaded GGUF under
the managed model directory. Manual files are admitted from bounded GGUF
metadata, tokenizer, tensor layout, size, and runtime support while a retained
descriptor keeps the selected file stable; the multi-gigabyte tensor payload
is not hashed before serving. This includes a final file symlink such as a
GGUF linked from another local model library: hf2q retains the target and
revalidates both link and target identities without traversing linked
directories. If none exists, hf2q resolves the repository to one immutable
commit, checks disk space, and downloads and verifies the exact hosted GGUF in
the Hugging Face cache. It then publishes a final-leaf symlink below the
readable managed path
`$HOME/.local/share/hf2q/models/<owner>/<repo>/<commit>/<artifact>`.
The payload is not copied or moved. Legacy `v2-<hex>` directories remain
readable but are never created by a new download.

For this multimodal model, hf2q also reuses or downloads the one matching
`mmproj`, verifies the pair, and loads it automatically. If a valid projector
cannot be established, the server warns and remains available text-only.
Host, port, scheduler, and slot count come from your `hf2q setup` answers.
Leave this foreground terminal open and wait for the listening message.

Use `hf2q serve list` at any time to inspect local receipt-backed, managed,
cached, and loose GGUF options without contacting the Hub.

## 3. Chat with the model (terminal 2)

```bash
hf2q chat
```

Chat discovers the running server on this machine and connects. Ask it
something real—this is the first inference proof. `/status` shows the endpoint
and token statistics; `/quit` exits.

For a one-command standalone check, this performs the same preparation and
starts an owned loopback server automatically. A targeted chat deliberately
owns its exact server even when another local server is already advertised:

```bash
hf2q chat jenerallee78/Qwen3.8-27B-Abliterated-SFT:Q4_K_M
```

## 4. Confirm the managed artifact

```bash
hf2q chat list
```

The Q4_K_M row must name the canonical repository and immutable revision. A
second `serve` or model-targeted `chat` invocation reuses those local bytes
instead of transferring or hashing the full payload again.

## 5. Prove vision with one request

Still in terminal 2, after `/quit` returns you to the shell. The surrounding
parentheses run this check in a subshell, so a failed proof reports its error
without closing your terminal:

```bash
(
RED_PNG="iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAb0lEQVR4nO3PAQkAAAyEwO9feoshgnABdLep8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3IPanc8OLDQitxAAAAAElFTkSuQmCC"
MODEL_ID="$(curl -fsS http://127.0.0.1:8081/v1/models |
  jq -r '[.data[] | select(.loaded == true) | .id] | first // ""')"
if [ -z "$MODEL_ID" ]; then
  echo "no loaded model yet — wait for terminal 1 to report the model is ready" >&2
  exit 1
fi
RESPONSE="$(jq -n --arg model "$MODEL_ID" --arg image "data:image/png;base64,$RED_PNG" '{
  model: $model,
  messages: [{role: "user", content: [
    {type: "text", text: "What is the dominant color? One word."},
    {type: "image_url", image_url: {url: $image}}
  ]}],
  temperature: 0, max_tokens: 16, stream: false, hf2q_enable_thinking: false
}' | curl -sS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' -d @-)"
echo "$RESPONSE" | jq -er '.choices[0].message.content' | grep -i red || {
  echo "vision check failed; the server said:" >&2
  echo "$RESPONSE" | jq -r '.error.message // .' >&2
  exit 1
}
echo "vision check passed: $MODEL_ID saw red"
)
```

It must end with `vision check passed`. If it does, text, streaming, and
vision are all proven. Do not continue until both this and the previous
section pass; on failure the block prints the server's own error message —
read that and the server output in terminal 1 before retrying.

## 6. Install OpenCode

```bash
npm install -g opencode-ai
```

## 7. Point OpenCode at hf2q

This merge adds the local hf2q provider and selects its model.
It preserves every existing agent, tool, permission, plugin, instruction,
and MCP setting:

```bash
CONFIG="$HOME/.config/opencode/opencode.json"
MODEL_ID="$(curl -fsS http://127.0.0.1:8081/v1/models |
  jq -er '.data | map(select(.loaded == true)) | .[0].id')"
mkdir -p "$(dirname "$CONFIG")"
[ -f "$CONFIG" ] || printf '{}\n' > "$CONFIG"
cp "$CONFIG" "$CONFIG.$(date +%Y%m%d%H%M%S).bak"

jq --arg model_id "$MODEL_ID" '
  .provider = (.provider // {})
  | .provider.hf2q = ((.provider.hf2q // {}) + {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local hf2q",
      "options": {"baseURL": "http://127.0.0.1:8081/v1", "apiKey": "local"},
      "models": {
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
      }
    })
  | .model = ("hf2q/" + $model_id)
' "$CONFIG" > "$CONFIG.tmp" && mv "$CONFIG.tmp" "$CONFIG"
```

Try it: `opencode --model "hf2q/$MODEL_ID"`, then ask the agent to list the
current directory. The first launch downloads the provider package
(`@ai-sdk/openai-compatible`), which can take a minute on a fresh machine;
later launches reuse it. The transcript must show a real Bash tool call and
its result — a prose claim that tools exist is not proof.

## 8. Install Agentic Kit

Run this from the Git repository where you want Agentic Kit's project files.
Project setup may replace existing agent configuration, so commit or back up
first.

```bash
npm install -g @pacphi/agentic-kit@next
ak setup
ak setup --opencode
ak sync
ak status
```

The first `ak setup` performs machine and project setup. The second wires the
OpenCode host integration, and `ak sync` upgrades, heals, and verifies the
whole kit. `ak status` should report the memory backend, MCP integrations,
and OpenCode convergence as healthy.

## 9. Install the local research stack

This gives OpenCode a visible `/search QUERY` command plus `web_search`,
`web_fetch`, `web_crawl`, and `web_extract` (and capitalized aliases), backed
by local SearXNG and fetch services. Both
services bind to `127.0.0.1` only, restart automatically at login, and need
no API keys. Google Chrome is required for JavaScript page reads and the
bounded discovery cascade used when SearXNG produces no usable URLs:

```bash
if [ ! -d '/Applications/Google Chrome.app' ]; then
  brew install --cask google-chrome
fi

curl -fsSL \
  https://raw.githubusercontent.com/robertelee78/hf2q/main/scripts/install_opencode_web_stack.sh \
  | bash
```

If `brew` is missing, install Chrome from <https://www.google.com/chrome/>
first. The installer pins every dependency, backs up changed files, and
verifies a current fact (today's gold price), an obscure attribution (who
wrote Unicornscan), and company research (IOActive) before activating the
OpenCode assets. A result counts only when its title, URL, or excerpt contains
the query's identifying term; unrelated URLs fail the gate. If both SearXNG
and the bounded fallback cascade fail, the installer exits nonzero,
stops the managed services, and does not activate a new plugin or command.
A successful run is the installation proof. Restart OpenCode once afterward
so it loads the plugin and `/search` command.

The fallback is a best-effort, fixed-provider cascade, not a promise to solve
every search-engine CAPTCHA. It tries guarded static Brave results
(`brave-search-fallback`), guarded Bing RSS (`bing-rss-fallback`), and only
then bounded Bing browser/stealth transport (`bing-browser-fallback`). The
service accepts only query-relevant parsed organic results and validates every
result URL before reporting success. Automatic page reads are limited to
public HTTP(S) targets and revalidate every redirect; a blocked page read
leaves the search excerpt and source URL intact.

## 10. Use the full stack

```bash
opencode
```

Ask it to research something current, for example a library released this
month. The transcript must contain a real `web_search` tool call with fetched
page content, and ordinary Bash/file tools must work in the same session.
That transcript is the end-to-end acceptance proof.

You can also invoke the visible command directly inside OpenCode:

```text
/search OpenCode AI coding agent
```

The result must show `Search route:` and engine or fallback provenance. A
`WEB_SEARCH_FAILED` response is an infrastructure failure, not evidence that
the web has no matching pages.

To stop the model server, press Ctrl-C in terminal 1. To manage the research
services, rerun the installer with `--status`, `--disable`, `--enable`, or
`--uninstall` (removal moves everything into a timestamped Trash folder; the
model, hf2q, OpenCode, and Agentic Kit are untouched).

## Troubleshooting

- **Server fails to start or generate** — read the output in terminal 1, then
  repeat sections 4 and 5.
- **Port 8081 already in use** — something else is listening; inspect with
  `lsof -nP -iTCP:8081 -sTCP:LISTEN`, or rerun `hf2q setup` and pick another
  port (use the same port in section 7's `baseURL`).
- **Projector binding failed** — read the automatic-projector warning, then run
  `hf2q serve list`. Repeating the model-targeted serve command rechecks local
  authority and the exact repository revision; it never guesses a projector
  from an unrelated file.
- **OpenCode has no tools** — run `opencode debug config` and remove any
  agent-level `permission: "deny"` or `tools: {"*": false}` left by other
  configuration; this guide writes neither.
- **Agentic Kit unhealthy** — run `ak sync`, then `ak status`.
- **Research tools or `/search` missing** — restart OpenCode, then rerun the
  installer with `--status`. Status exits nonzero when neither primary nor
  fallback discovery returns a usable URL and prints the failing route.
