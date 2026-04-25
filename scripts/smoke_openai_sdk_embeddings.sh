#!/usr/bin/env bash
# ADR-005 Phase 2a Task #12 — OpenAI SDK acceptance smoke for /v1/embeddings.
#
# Boots hf2q against a real BERT GGUF (bge-small-en-v1.5-f16 by default)
# and exercises the endpoint via the official `openai` Python client.
# Validates that hf2q's wire format is byte-for-byte compatible with
# what an OpenAI-SDK consumer expects:
#
#   - GET  /v1/models            lists the embedding model
#   - POST /v1/embeddings (str)  returns one embedding object
#   - POST /v1/embeddings (list) returns one embedding object per input
#   - Embedding dim matches the model's hidden_size
#   - Each embedding is L2-normalized (||y||₂ ≈ 1.0)
#   - usage.prompt_tokens + usage.total_tokens populated
#
# Usage:
#   bash scripts/smoke_openai_sdk_embeddings.sh
#
# Env overrides:
#   HF2Q_BIN     binary path (default: ./target/release/hf2q)
#   PORT         server port (default: 38443)
#   EMBED_GGUF   embedding model GGUF
#                (default: /opt/hf2q/models/bert-test/bge-small-en-v1.5-f16.gguf)
#
# Exits 0 on all-pass, 1 on first failure.

set -euo pipefail

HF2Q_BIN="${HF2Q_BIN:-./target/release/hf2q}"
PORT="${PORT:-38443}"
EMBED_GGUF="${EMBED_GGUF:-/opt/hf2q/models/bert-test/bge-small-en-v1.5-f16.gguf}"

if [[ ! -x "$HF2Q_BIN" ]]; then
    echo "[smoke] Binary not found: $HF2Q_BIN" >&2
    echo "[smoke] Build first: cargo build --release --bin hf2q" >&2
    exit 1
fi

if [[ ! -f "$EMBED_GGUF" ]]; then
    echo "[smoke] Embedding GGUF not found: $EMBED_GGUF" >&2
    echo "[smoke] Download via: curl -L https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-f16.gguf -o $EMBED_GGUF" >&2
    exit 1
fi

if ! python3 -c "import openai" 2>/dev/null; then
    echo "[smoke] openai Python client not installed. Run: pip install openai" >&2
    exit 1
fi

MODEL_ID="$(basename "$EMBED_GGUF" .gguf)"
HOST="127.0.0.1"
BASE="http://${HOST}:${PORT}"
LOG_DIR="$(mktemp -d -t hf2q-smoke-openai-XXXXXX)"
SERVER_LOG="$LOG_DIR/server.log"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$LOG_DIR"
}
trap cleanup EXIT INT TERM

echo "[smoke] Booting hf2q on ${BASE}"
echo "[smoke]   --embedding-model $EMBED_GGUF"
echo "[smoke]   model id: $MODEL_ID"

"$HF2Q_BIN" serve \
    --embedding-model "$EMBED_GGUF" \
    --port "$PORT" \
    --host "$HOST" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for /readyz to flip to 200. With no --model the engine is absent
# and the server is trivially ready, but we still wait for the bind.
DEADLINE=$(( $(date +%s) + 30 ))
while true; do
    if [[ $(date +%s) -gt $DEADLINE ]]; then
        echo "[smoke] FAIL: server didn't come up within 30s. Log:" >&2
        cat "$SERVER_LOG" >&2
        exit 1
    fi
    if curl -sSf -m 1 "${BASE}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 0.2
done

echo "[smoke] Server is up. Running OpenAI SDK assertions..."

python3 - "$BASE" "$MODEL_ID" <<'PY'
import sys, math
from openai import OpenAI

base, model_id = sys.argv[1], sys.argv[2]

client = OpenAI(
    base_url=f"{base}/v1",
    api_key="not-required-for-localhost",
    timeout=30.0,
)

# --- 1. models.list() ---
print(f"[smoke] GET /v1/models")
mlist = client.models.list()
ids = [m.id for m in mlist.data]
assert model_id in ids, f"Expected {model_id} in models.list, got {ids}"
print(f"  ✓ {model_id} in models.list ({len(ids)} models total)")

# --- 2. embeddings.create with single string input ---
print(f"[smoke] POST /v1/embeddings (single string)")
resp = client.embeddings.create(
    model=model_id,
    input="hello world",
)
assert resp.object == "list", f"Expected object='list', got {resp.object!r}"
assert resp.model == model_id, f"Model mismatch: {resp.model!r} != {model_id!r}"
assert len(resp.data) == 1, f"Expected 1 embedding, got {len(resp.data)}"
emb_obj = resp.data[0]
assert emb_obj.object == "embedding", f"Expected nested object='embedding', got {emb_obj.object!r}"
assert emb_obj.index == 0
emb = list(emb_obj.embedding)
assert len(emb) > 0, "Empty embedding vector"
norm = math.sqrt(sum(v * v for v in emb))
assert abs(norm - 1.0) < 1e-3, f"Embedding not unit-norm: ||y||₂ = {norm}"
assert resp.usage.prompt_tokens > 0, "Expected non-zero prompt_tokens"
assert resp.usage.total_tokens >= resp.usage.prompt_tokens, (
    f"total_tokens ({resp.usage.total_tokens}) < prompt_tokens ({resp.usage.prompt_tokens})"
)
hidden_size = len(emb)
print(f"  ✓ 1 embedding, dim={hidden_size}, ||y||₂={norm:.6f}, "
      f"prompt_tokens={resp.usage.prompt_tokens}")

# --- 3. embeddings.create with list of inputs ---
print(f"[smoke] POST /v1/embeddings (batch of 3)")
inputs = ["hello world", "the quick brown fox", "a longer sentence to test variable lengths in batch"]
resp = client.embeddings.create(model=model_id, input=inputs)
assert len(resp.data) == 3, f"Expected 3 embeddings, got {len(resp.data)}"
for i, eo in enumerate(resp.data):
    assert eo.index == i, f"data[{i}].index = {eo.index}"
    assert len(eo.embedding) == hidden_size, (
        f"data[{i}].embedding dim ({len(eo.embedding)}) != single-input dim ({hidden_size})"
    )
    n = math.sqrt(sum(v * v for v in eo.embedding))
    assert abs(n - 1.0) < 1e-3, f"data[{i}] not unit-norm: {n}"
print(f"  ✓ 3 embeddings, all dim={hidden_size}, all unit-norm, "
      f"total_tokens={resp.usage.total_tokens}")

# --- 4. embeddings.create with encoding_format='float' (explicit) ---
print(f"[smoke] POST /v1/embeddings (encoding_format='float')")
resp = client.embeddings.create(
    model=model_id,
    input="explicit float encoding",
    encoding_format="float",
)
assert len(resp.data) == 1
print(f"  ✓ encoding_format='float' accepted")

# --- 5. encoding_format='base64' (explicit — user decodes manually) ---
# When the SDK's caller explicitly passes encoding_format='base64', the
# SDK returns the base64 string raw — the caller is expected to decode
# it. (The SDK auto-decodes ONLY when the user did NOT pass
# encoding_format; case 2 above already exercised the auto-decode path
# since the SDK's default-omitted call sets base64 internally.)
import array, base64
print(f"[smoke] POST /v1/embeddings (encoding_format='base64' — manual decode)")
resp_b64 = client.embeddings.create(
    model=model_id,
    input="round-trip test",
    encoding_format="base64",
)
resp_f = client.embeddings.create(
    model=model_id,
    input="round-trip test",
    encoding_format="float",
)
b64_str = resp_b64.data[0].embedding
assert isinstance(b64_str, str), (
    f"With explicit encoding_format='base64' the SDK keeps the string raw; "
    f"got {type(b64_str).__name__}"
)
v_b64 = array.array("f", base64.b64decode(b64_str)).tolist()
v_f = list(resp_f.data[0].embedding)
assert len(v_b64) == hidden_size, f"base64 decoded dim {len(v_b64)} != {hidden_size}"
assert len(v_f) == hidden_size, f"float dim {len(v_f)} != {hidden_size}"
# Bit-exact: same forward pass, only the encoding step differs.
max_diff = max(abs(a - b) for a, b in zip(v_b64, v_f))
if max_diff > 1e-6:
    raise AssertionError(
        f"base64 vs float decode mismatch: max_diff = {max_diff}\n"
        f"  v_b64[:4] = {v_b64[:4]}\n"
        f"  v_f[:4]   = {v_f[:4]}"
    )
n_b64 = math.sqrt(sum(v * v for v in v_b64))
assert abs(n_b64 - 1.0) < 1e-3, f"base64 vector not unit-norm: {n_b64}"
print(f"  ✓ base64 manual-decode bit-exact to float (max_diff={max_diff:.2e}, ||y||₂={n_b64:.6f})")

# --- 6. embeddings.create with empty model id should 400/404 ---
print(f"[smoke] POST /v1/embeddings (wrong model id — must fail)")
try:
    client.embeddings.create(
        model="this-model-does-not-exist",
        input="hello",
    )
    raise AssertionError("Expected wrong model id to fail")
except Exception as e:
    msg = str(e)
    assert "model_not_loaded" in msg or "400" in msg or "404" in msg, (
        f"Expected model_not_loaded / 400 / 404 in error, got: {e}"
    )
    print(f"  ✓ wrong model id rejected (error: {type(e).__name__})")

print()
print(f"[smoke] ALL CHECKS PASSED ✓")
PY

echo "[smoke] PASS"
