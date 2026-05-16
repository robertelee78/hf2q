#!/usr/bin/env python3
"""03-openai-python-client.py

Drive an `hf2q serve` instance from the stock OpenAI Python SDK.

`hf2q serve` implements `/v1/chat/completions`, `/v1/embeddings`,
`/v1/models`, and SSE streaming — the OpenAI SDK works as-is when
pointed at the local port.

Usage:
    pip install openai
    # In one terminal:
    hf2q serve --model <gguf-path> --port 8080
    # In another terminal:
    python examples/03-openai-python-client.py
"""

from openai import OpenAI

# hf2q serve doesn't require an API key but the SDK insists on one;
# any non-empty string works.
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")

# --- Non-streaming completion -----------------------------------------------
print("=== one-shot chat completion ===")
resp = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "List 3 sourdough bread tips."}],
    max_tokens=200,
    temperature=0,
)
print(resp.choices[0].message.content)

# --- Streaming completion (SSE) --------------------------------------------
print("\n=== streaming chat completion ===")
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a joke about quantization."}],
    max_tokens=120,
    temperature=0,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()
