# Get started with hf2q and Qwen3.8

For the complete verified multimodal download, stock OpenCode, full Agentic
Kit, and local search/fetch/crawl/extract journey, use
**[hf2q + Qwen3.8 + AK + search/fetch: complete setup](hf2q+qwen3.8+ak+search-fetch-setup.md)**.
The source-first core CLI journey remains below.

This guide installs hf2q, configures it for one Mac, converts and quantizes an
exact Hugging Face source revision with hf2q, serves the resulting GGUF, and
opens a local terminal chat. Those are ordinary hf2q commands; there is no
second model-preparation or onboarding system.

The guide model is
[`jenerallee78/Qwen3.8-27B-Abliterated-SFT`](https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT)
at exact revision `08c2f075b43bc06456382db6b918a3dcabdcf4dd`.
It is an Apache-2.0 community checkpoint derived from `Qwen/Qwen3.8-27B`.
Pinning the revision keeps the source identity stable even if the repository's
default branch later changes.

## Before you start

You need an Apple Silicon Mac, `curl`, and about 100 GiB free on the volume
used for the Hugging Face cache and converted model. The selected source files
are about 51.77 GiB. The measured text Q4_K_M output is about 15.66 GiB, and
the conversion also produces a source-matched F16 vision projector.

This exact journey was validated on an Apple M5 Max with 128 GiB unified
memory. That describes the validation host; it is not a claimed minimum for
every Apple Silicon Mac. Conversion and the first model load can take a while.

The core journey does not require Homebrew, Node.js, npm, jq, Python,
`huggingface-cli`, llama.cpp, OpenCode, or Docker.

## 1. Install and configure hf2q

Install the signed and notarized Apple Silicon binary without `sudo`:

```bash
curl -fsSL https://hf2q.us/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
hf2q --version
```

This guide is validated against hf2q 0.1.8. If `hf2q --version` reports an
older release, run `hf2q update` before continuing.

Run setup and answer its five operator choices. Press Enter to accept a
recommendation; explicit flags on later commands still win.

```bash
hf2q setup
hf2q doctor
```

For an unattended setup that accepts all guide defaults, use
`hf2q setup --accept-defaults` instead. Those defaults are Q4_K_M conversion,
localhost port 8081, and inflight-batched serving with one active slot. Setup
only inspects the Mac and writes hf2q configuration; it does not download,
convert, load, or serve a model.

## 2. Convert and quantize the pinned source

Choose a durable destination, then let hf2q resolve, download, verify,
convert, and quantize the source in-process:

```bash
MODEL_DIR="$HOME/.local/share/hf2q/models/qwen3.8"
MODEL="$MODEL_DIR/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

hf2q convert jenerallee78/Qwen3.8-27B-Abliterated-SFT \
  --revision 08c2f075b43bc06456382db6b918a3dcabdcf4dd \
  --quant q4_k_m \
  --output "$MODEL"
```

The explicit quantization keeps this guide's artifact reproducible; outside
the guide, omitting `--quant` uses the default chosen during setup. hf2q does
not invoke an external converter or quantizer.

Because the pinned source is multimodal, the same command writes these
source-bound outputs and one conversion receipt beside each:

- `Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf`, the text model used below; and
- `Qwen3.8-27B-Abliterated-SFT-Q4_K_M-mmproj.gguf`, its F16 projector.

Keep each receipt with its artifact. This guide serves the text model only;
producing the matched projector does not by itself enable image requests.

## 3. Serve the model

In the same terminal, start the server in the foreground:

```bash
hf2q serve --model "$MODEL"
```

The setup defaults bind only to `127.0.0.1:8081` and select the proven
inflight-batched, one-slot configuration. Leave this terminal running. Stop
the server later with Ctrl-C; there is no background process or PID file to
clean up.

## 4. Chat from a second terminal

Wait until the model finishes loading, then run:

```bash
hf2q chat --url http://127.0.0.1:8081/v1
```

At the chat prompt:

```text
/thinking off
/status
Write a Rust function that adds two i64 values.
/quit
```

`/status` shows the endpoint and the model selected from `/v1/models`.
`/quit` exits chat but leaves the foreground server running in the first
terminal.

## 5. Check the OpenAI-compatible API directly

The native conversion emits `Qwen3.8 27B Abliterated SFT` as this artifact's
model ID. First confirm the live server advertises it:

```bash
curl -fsS http://127.0.0.1:8081/v1/models
```

Then send a normal OpenAI-compatible request:

```bash
curl -fsS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8 27B Abliterated SFT","messages":[{"role":"user","content":"Write a Rust function that adds two i64 values."}],"temperature":0,"max_tokens":256,"hf2q_enable_thinking":false}'
```

Add `"stream":true` to the request body for SSE. A valid stream ends with a
`[DONE]` event.

## 6. Connect an existing OpenCode installation (optional)

hf2q does not install or rewrite OpenCode. If OpenCode is already installed,
add an hf2q provider to your own `opencode.json` and use the exact model ID
confirmed above:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "hf2q/Qwen3.8 27B Abliterated SFT",
  "provider": {
    "hf2q": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local hf2q",
      "options": {
        "baseURL": "http://127.0.0.1:8081/v1",
        "apiKey": "local"
      },
      "models": {
        "Qwen3.8 27B Abliterated SFT": {
          "name": "Qwen3.8 27B Abliterated SFT via hf2q",
          "tool_call": true,
          "reasoning": true,
          "interleaved": "reasoning_content",
          "temperature": true,
          "limit": {
            "context": 262144,
            "output": 8192
          }
        }
      }
    }
  }
}
```

The `npm` property above is OpenCode's name for its provider adapter; it is
not an hf2q npm installation channel. Start OpenCode only after `/readyz`
succeeds. A realistic client check should complete a tool call, return the
tool result to the same conversation, and report cached input tokens on the
follow-up.

## Update, uninstall, and cleanup

Check for an hf2q update without changing anything:

```bash
hf2q update --check
```

`hf2q update` follows the channel that owns the current executable. Standalone
and Cargo installs update through their respective channel. Source-development
builds receive exact checkout instructions instead of having their repository
changed automatically.

Preview uninstall before confirming it:

```bash
hf2q uninstall
hf2q uninstall --yes
```

Normal uninstall preserves configuration, caches, downloaded source, and
converted models. `--purge-config` and `--purge-cache` are separate explicit
choices shown in the preview. Delete the model directory yourself only if you
also want to remove the large model artifacts.

For problems, run `hf2q doctor`, read [Setup](setup.md), and see
[Converting a model](converting-a-model.md). The exact supported-family and
release boundaries live in [the shipping contract](shipping-contract.md).
