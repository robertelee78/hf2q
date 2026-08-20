# Get started with hf2q and Qwen3.8

This guide takes one supported text model from its exact Hugging Face source
revision to a local OpenAI-compatible API and, optionally, OpenCode. Every
model command shown here is an existing `hf2q` command; there is no separate
onboarding workflow or model-preparation service.

The guide model is
[`jenerallee78/Qwen3.8-27B-Abliterated-SFT`](https://huggingface.co/jenerallee78/Qwen3.8-27B-Abliterated-SFT),
pinned to revision `08c2f075b43bc06456382db6b918a3dcabdcf4dd`.
It is an Apache-2.0 community checkpoint derived from
`Qwen/Qwen3.8-27B`; the pinned revision, not a moving `main`, is the guide's
acceptance artifact.

## Before you start

You need:

- an Apple Silicon Mac;
- Rust 1.88 or newer for the current source installation path;
- at least 100 GiB free on the volume used for the Hugging Face cache and
  converted model; and
- `curl` and `jq` for the API examples.

This exact journey was validated on an Apple M5 Max with 128 GiB unified
memory. Lower-memory Apple Silicon hosts have not yet been characterized for
this model; 128 GiB is the validation host, not a claimed formal minimum.

The pinned source revision contains 51.77 GiB of selected weight and metadata
files. The measured Q4_K_M output is 16,810,714,848 bytes (15.66 GiB). The
100 GiB requirement leaves room for both plus conversion and filesystem
headroom.

This first guide is text-only. Do not add `--mmproj` or advertise image input:
Qwen3.8 text conversion and serving are accepted, while its vision path remains
a separately gated candidate.

## 1. Install today's build

The standalone installer and package-manager channels are separate ADR-045
deliverables and are not published yet. Until they pass their installed-
artifact gates, build the current source explicitly:

```bash
git clone https://github.com/robertelee78/hf2q.git
cd hf2q
GIT_COMMIT_SHA="$(git rev-parse HEAD)" cargo build --release --locked
export PATH="$PWD/target/release:$PATH"

hf2q --version
hf2q doctor
```

The explicit `GIT_COMMIT_SHA` gives remote-conversion receipts the immutable
converter identity they require. A future published binary installer will
provide that identity as part of the release artifact instead.

## 2. Convert and quantize the model

Choose a durable destination outside the source checkout, then let hf2q
resolve, download, verify, convert, and quantize the exact model revision:

```bash
MODEL_DIR="$HOME/.local/share/hf2q/models/qwen3.8"
MODEL="$MODEL_DIR/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

hf2q convert jenerallee78/Qwen3.8-27B-Abliterated-SFT \
  --revision 08c2f075b43bc06456382db6b918a3dcabdcf4dd \
  --quant q4_k_m \
  --output "$MODEL"
```

hf2q uses its in-process Hugging Face client; Python, `huggingface-cli`,
llama.cpp, and an external converter are not involved. The command writes the
GGUF and a sibling conversion receipt at `$MODEL.receipt.json`.

Keep the receipt with the model. It records the resolved source revision,
converter revision, selected quantization, output identity, and bounded
conversion evidence.

## 3. Serve the GGUF

Start a localhost-only server:

```bash
hf2q serve \
  --model "$MODEL" \
  --host 127.0.0.1 \
  --port 8081 \
  --scheduler inflight-batched \
  --max-slots 1
```

The one-slot SlotAware scheduler is deliberate. Small direct requests also
fit the default SerialFifo path, but OpenCode's measured agent prompt is about
7,100 tokens and exceeds SerialFifo's 2,048-token bounded prefill transaction.
SlotAware keeps the model's full declared context and lets the same server
support both the direct examples and OpenCode.

Leave that terminal running. In a second terminal, verify readiness and ask
the server for its actual model ID:

```bash
curl -fsS http://127.0.0.1:8081/readyz

MODEL_ID="$(
  curl -fsS http://127.0.0.1:8081/v1/models |
    jq -r '.data[0].id'
)"
printf 'Model ID: %s\n' "$MODEL_ID"
```

Do not guess the ID from the filename; use the server response.

## 4. Send a chat request

The selected model card recommends non-thinking operation. State that choice
explicitly in requests because the embedded Qwen template otherwise defaults
to thinking mode:

```bash
curl -fsS http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg model "$MODEL_ID" '{
    model: $model,
    messages: [{
      role: "user",
      content: "Write a Rust function that adds two i64 values."
    }],
    temperature: 0,
    max_tokens: 256,
    hf2q_enable_thinking: false
  }')"
```

For a streaming response, add `stream: true` to the JSON body and consume the
SSE stream until the terminal `[DONE]` event.

## 5. Connect OpenCode (optional)

OpenCode can use hf2q as a custom OpenAI-compatible provider. Put the model ID
returned above into your OpenCode configuration rather than assuming a fixed
value:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "hf2q/<MODEL_ID_FROM_V1_MODELS>",
  "small_model": "hf2q/<MODEL_ID_FROM_V1_MODELS>",
  "provider": {
    "hf2q": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local hf2q",
      "options": {
        "baseURL": "http://127.0.0.1:8081/v1",
        "apiKey": "local"
      },
      "models": {
        "<MODEL_ID_FROM_V1_MODELS>": {
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

Start OpenCode only after `/readyz` succeeds. A normal acceptance run should
exercise one tool call, return the tool result to the model, continue the same
conversation, and confirm that the unchanged prompt prefix reports reused
cached tokens.

## What this guide does not do

- It does not use `hf2q setup` yet. The current setup schema is provisional;
  ADR-045's next setup slice will replace it with defaults that `convert` and
  `serve` actually consume.
- It does not install or configure OpenCode for you.
- It does not enable Qwen3.8 vision or speculative MTP decoding.
- It does not add a second model-preparation, model-registry, or session-cache
  subsystem around the existing CLI.

See [Converting a model](converting-a-model.md) for the general conversion
reference and [the shipping contract](shipping-contract.md) for the exact
family and scheduler acceptance boundaries.
