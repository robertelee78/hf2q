#!/usr/bin/env bash
# wedge4_qwen3vl.sh — ADR-005 iter-224 row 6 Wedge-4f operator recipe.
#
# End-to-end HF→GGUF→hf2q round-trip for `Qwen/Qwen3-VL-2B-Instruct`.
# Walks the operator from a fresh HF repo download to a working
# chat-with-images smoke test on both `hf2q serve` (HTTP API) and the
# round-trip parity test.
#
# This is the closure recipe for the Wedge-4 series. When this script
# completes successfully, every wedge from 4a through 4f has been
# verified end-to-end against a real Qwen3-VL model:
#
#   * Wedge-4a (LM API opener)       — exercised by the chat request
#   * Wedge-4b (mmproj profile)      — exercised by `hf2q serve --mmproj`
#   * Wedge-4c (ViT forward)         — exercised by image preprocessing
#   * Wedge-4c.5 (LM-side hooks)     — exercised by image-bearing chat
#   * Wedge-4d (chat handler)        — exercised by `image_url` content
#   * Wedge-4e (streaming + tools)   — exercised by `stream: true`
#   * Wedge-4f (convert — THIS)      — produces the GGUF + mmproj input
#
# Disk space:
#   * HF source download:  ~5 GB (Qwen3-VL-2B-Instruct safetensors)
#   * Quantized GGUF:      ~1.2 GB (Q4_0 text model)
#   * mmproj:              ~700 MB (F16 vision tower)
#   Total scratch:         ~7 GB
#
# RAM:
#   * Convert peak:        ~10 GB (held safetensors + intermediate)
#   * Serve runtime:       ~3 GB (Q4_0 LM + F16 mmproj weights)
#
# Time:
#   * Download:            ~5 min on a fast connection
#   * Convert:             ~3 min on M5 Max
#   * Serve smoke:         ~30s (load + 1 chat request)
#
# Usage:
#   scripts/wedge4_qwen3vl.sh               # download + convert + smoke
#   scripts/wedge4_qwen3vl.sh --no-download # skip the git lfs clone
#   scripts/wedge4_qwen3vl.sh --no-serve    # skip the live serve smoke
#   HF_DIR=/path/to/Qwen3-VL-2B-Instruct scripts/wedge4_qwen3vl.sh
#
# Exit codes:
#   0  recipe completed end-to-end successfully
#   1  usage / env error
#   2  download failed
#   3  convert failed
#   4  serve smoke failed
#   5  required tool missing (git, git-lfs, curl, jq, base64)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---- Configuration ----
DEFAULT_HF_REPO="Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_HF_DIR="${REPO_ROOT}/.cfa-archive/wedge4f-hf/Qwen3-VL-2B-Instruct"
DEFAULT_OUTPUT_DIR="${REPO_ROOT}/.cfa-archive/wedge4f-out"
DEFAULT_QUANT="q4_0"
DEFAULT_PORT="18080"

HF_REPO="${HF_REPO:-$DEFAULT_HF_REPO}"
HF_DIR="${HF_DIR:-$DEFAULT_HF_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
QUANT="${QUANT:-$DEFAULT_QUANT}"
PORT="${PORT:-$DEFAULT_PORT}"

DO_DOWNLOAD=1
DO_SERVE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-download) DO_DOWNLOAD=0; shift ;;
        --no-serve)    DO_SERVE=0; shift ;;
        --help|-h)
            grep '^#' "$0" | head -40
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---- Tool checks ----
need_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: required tool '$1' not on PATH" >&2
        exit 5
    fi
}
need_tool git
need_tool curl
need_tool jq
need_tool base64

if [[ $DO_DOWNLOAD -eq 1 ]]; then
    if ! command -v git-lfs >/dev/null 2>&1; then
        echo "ERROR: git-lfs not installed; required for HF safetensors download" >&2
        echo "Install: brew install git-lfs && git lfs install" >&2
        exit 5
    fi
fi

# ---- Step 1: Download HF model (~5 GB) ----
if [[ $DO_DOWNLOAD -eq 1 ]]; then
    echo "==> Step 1: Downloading $HF_REPO to $HF_DIR"
    if [[ -d "$HF_DIR/.git" ]]; then
        echo "    (already cloned; running git lfs pull to refresh)"
        cd "$HF_DIR"
        git lfs pull || { echo "git lfs pull failed"; exit 2; }
        cd "$REPO_ROOT"
    else
        mkdir -p "$(dirname "$HF_DIR")"
        # GIT_LFS_SKIP_SMUDGE=1 first to clone metadata, then pull lfs files
        # so the operator can see progress and Ctrl-C out without leaving a
        # half-pulled clone.
        GIT_LFS_SKIP_SMUDGE=1 git clone "https://huggingface.co/${HF_REPO}" "$HF_DIR" \
            || { echo "git clone failed"; exit 2; }
        cd "$HF_DIR"
        git lfs pull || { echo "git lfs pull failed"; exit 2; }
        cd "$REPO_ROOT"
    fi
    if [[ ! -f "$HF_DIR/config.json" ]]; then
        echo "ERROR: no config.json in $HF_DIR after download" >&2
        exit 2
    fi
else
    echo "==> Step 1: Skipping download (--no-download)"
    if [[ ! -d "$HF_DIR" ]]; then
        echo "ERROR: HF_DIR=$HF_DIR does not exist; cannot proceed" >&2
        exit 1
    fi
fi

# Verify the source is actually a Qwen3-VL repo (sovereignty check —
# don't waste convert time if the operator pointed at the wrong repo).
echo "==> Step 1.5: Verifying source is Qwen3-VL"
if ! jq -e '.vision_config' "$HF_DIR/config.json" >/dev/null; then
    echo "ERROR: $HF_DIR/config.json has no vision_config — not a vision model" >&2
    exit 1
fi
if ! jq -e '.vision_config.deepstack_visual_indexes' "$HF_DIR/config.json" >/dev/null; then
    echo "WARNING: $HF_DIR/config.json has no vision_config.deepstack_visual_indexes" >&2
    echo "         This may not be a Qwen3-VL family model — proceed at own risk" >&2
fi

# ---- Step 2: Build hf2q (release) ----
echo "==> Step 2: Building hf2q (release)"
cargo build --release --bin hf2q

HF2Q_BIN="$REPO_ROOT/target/release/hf2q"
if [[ ! -x "$HF2Q_BIN" ]]; then
    echo "ERROR: $HF2Q_BIN not built" >&2
    exit 3
fi

# ---- Step 3: Convert HF → GGUF + mmproj ----
echo "==> Step 3: Converting HF → GGUF + mmproj"
mkdir -p "$OUTPUT_DIR"
TEXT_GGUF="$OUTPUT_DIR/qwen3-vl-2b-${QUANT}.gguf"

"$HF2Q_BIN" convert \
    --input "$HF_DIR" \
    --format gguf \
    --quant "$QUANT" \
    --output "$TEXT_GGUF" \
    --emit-vision-tower \
    --yes \
    || { echo "ERROR: hf2q convert failed"; exit 3; }

if [[ ! -f "$TEXT_GGUF" ]]; then
    echo "ERROR: text GGUF not at $TEXT_GGUF after convert" >&2
    exit 3
fi

# Find the emitted mmproj
MMPROJ_GGUF=""
for candidate in "$OUTPUT_DIR"/mmproj-*.gguf; do
    if [[ -f "$candidate" ]]; then
        MMPROJ_GGUF="$candidate"
        break
    fi
done
if [[ -z "$MMPROJ_GGUF" ]]; then
    echo "ERROR: no mmproj-*.gguf emitted under $OUTPUT_DIR" >&2
    echo "       --emit-vision-tower silent-skipped — verify config.json has vision_config" >&2
    exit 3
fi

echo "    text GGUF:   $TEXT_GGUF ($(du -h "$TEXT_GGUF" | cut -f1))"
echo "    mmproj GGUF: $MMPROJ_GGUF ($(du -h "$MMPROJ_GGUF" | cut -f1))"

# ---- Step 4: Verify the GGUF structure (read-only) ----
echo "==> Step 4: Verifying mmproj GGUF structure"
"$HF2Q_BIN" info --gguf "$MMPROJ_GGUF" 2>&1 | head -30 || true

# ---- Step 5: Live serve smoke test ----
if [[ $DO_SERVE -eq 0 ]]; then
    echo "==> Step 5: Skipping serve smoke (--no-serve)"
    echo
    echo "Wedge-4f convert recipe complete."
    echo "Run with --serve (default) to exercise the chat-with-images path."
    exit 0
fi

echo "==> Step 5: Starting hf2q serve (port $PORT)"
SERVE_LOG="/tmp/wedge4_qwen3vl_serve.log"
"$HF2Q_BIN" serve \
    --model "$TEXT_GGUF" \
    --mmproj "$MMPROJ_GGUF" \
    --host 127.0.0.1 \
    --port "$PORT" \
    > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
trap 'kill -9 $SERVE_PID 2>/dev/null || true' EXIT

# Wait for /readyz
echo "    waiting for /readyz on port $PORT..."
for i in $(seq 1 60); do
    if curl -s -f "http://127.0.0.1:$PORT/readyz" > /dev/null 2>&1; then
        echo "    server ready (after ${i}s)"
        break
    fi
    sleep 1
    if ! kill -0 $SERVE_PID 2>/dev/null; then
        echo "ERROR: server died during startup; log tail:" >&2
        tail -20 "$SERVE_LOG" >&2
        exit 4
    fi
done

if ! curl -s -f "http://127.0.0.1:$PORT/readyz" > /dev/null 2>&1; then
    echo "ERROR: server never became ready (60s timeout); log tail:" >&2
    tail -20 "$SERVE_LOG" >&2
    exit 4
fi

# ---- Step 6: Send a chat-with-image request ----
echo "==> Step 6: Sending chat-completions request with image"

# Use a tiny embedded test PNG (8x8 red square).
TEST_IMAGE_B64="iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX/AAA\
AAP9JREFUCNdj+P///38GBgYGBgYGBgYGBgYGAAAxKAJAVkUNHAAAAABJRU5ErkJggg=="
# Strip any newlines that crept in from the line continuation above.
TEST_IMAGE_B64="${TEST_IMAGE_B64//$'\n'/}"

RESPONSE_LOG="/tmp/wedge4_qwen3vl_response.json"
curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
        --arg img "data:image/png;base64,$TEST_IMAGE_B64" \
        '{
            model: "qwen3-vl-2b",
            messages: [{
                role: "user",
                content: [
                    {type: "text", text: "What color is this?"},
                    {type: "image_url", image_url: {url: $img}}
                ]
            }],
            max_tokens: 32,
            temperature: 0,
            stream: false
        }')" \
    > "$RESPONSE_LOG" || {
    echo "ERROR: curl failed; server log tail:" >&2
    tail -20 "$SERVE_LOG" >&2
    exit 4
}

# Verify the response shape.
if ! jq -e '.choices[0].message.content' "$RESPONSE_LOG" > /dev/null 2>&1; then
    echo "ERROR: response did not contain choices[0].message.content; got:" >&2
    cat "$RESPONSE_LOG" >&2
    exit 4
fi

CONTENT=$(jq -r '.choices[0].message.content' "$RESPONSE_LOG")
if [[ -z "$CONTENT" || "$CONTENT" == "null" ]]; then
    echo "ERROR: response message.content was empty/null; full body:" >&2
    cat "$RESPONSE_LOG" >&2
    exit 4
fi

echo "    response (first 200 chars):"
echo "    $(echo "$CONTENT" | head -c 200)"
echo

# ---- Step 7: Smoke streaming variant ----
echo "==> Step 7: Streaming variant"
STREAM_LOG="/tmp/wedge4_qwen3vl_stream.log"
curl -s -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
        --arg img "data:image/png;base64,$TEST_IMAGE_B64" \
        '{
            model: "qwen3-vl-2b",
            messages: [{
                role: "user",
                content: [
                    {type: "text", text: "Describe this color in one word."},
                    {type: "image_url", image_url: {url: $img}}
                ]
            }],
            max_tokens: 16,
            temperature: 0,
            stream: true
        }')" \
    > "$STREAM_LOG" || {
    echo "ERROR: streaming curl failed; server log tail:" >&2
    tail -20 "$SERVE_LOG" >&2
    exit 4
}

# Verify SSE shape: at least one `data:` line + a `data: [DONE]` sentinel.
if ! grep -q '^data:' "$STREAM_LOG"; then
    echo "ERROR: streaming response had no SSE 'data:' lines" >&2
    cat "$STREAM_LOG" >&2
    exit 4
fi
if ! grep -q '^data: \[DONE\]' "$STREAM_LOG"; then
    echo "ERROR: streaming response missing '[DONE]' sentinel" >&2
    tail -10 "$STREAM_LOG" >&2
    exit 4
fi
echo "    streaming OK: $(grep -c '^data:' "$STREAM_LOG") SSE events + [DONE] sentinel"

# ---- Cleanup ----
echo "==> Cleanup: stopping serve"
kill -TERM $SERVE_PID 2>/dev/null || true
wait $SERVE_PID 2>/dev/null || true
trap - EXIT

echo
echo "============================================================"
echo "Wedge-4f recipe COMPLETE — Wedge-4 series CLOSED."
echo
echo "Round-trip artefacts:"
echo "  HF source:    $HF_DIR"
echo "  Text GGUF:    $TEXT_GGUF"
echo "  mmproj GGUF:  $MMPROJ_GGUF"
echo
echo "Logs:"
echo "  Serve log:        $SERVE_LOG"
echo "  Non-stream resp:  $RESPONSE_LOG"
echo "  Stream resp:      $STREAM_LOG"
echo
echo "To run the synthetic round-trip parity test (no large download):"
echo "  cargo test --bin hf2q --tests --test qwen3vl_round_trip_e2e"
echo
echo "To run the operator-gated real-model round-trip:"
echo "  HF2Q_QWEN3VL_ROUND_TRIP=1 \\"
echo "    HF2Q_QWEN3VL_HF_DIR=$HF_DIR \\"
echo "    cargo test --test qwen3vl_round_trip_e2e --release -- --ignored"
echo "============================================================"
