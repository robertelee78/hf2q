#!/usr/bin/env bash
# sourdough_gate.sh — ADR-005 Phase 1b post-1bNEW.20.FIX correctness gate.
#
# Runs hf2q and llama-completion on the 22-token user sourdough prompt at
# T=0 greedy, max_tokens=1000, and asserts that the common byte prefix of
# their outputs is at least MIN_COMMON_PREFIX bytes (default: 3094). This
# enforces "hf2q's decode output is byte-identical to llama.cpp on the
# DWQ GGUF for the first ~830 decode tokens" as a mandatory pre-merge
# condition for every future speed item.
#
# Origin: 2026-04-11 investigation (ADR-005 "Sourdough たglitch" Walk
# Exception entry). The user reported a hiragana character appearing
# mid-decode; the investigation determined both hf2q and llama.cpp emit
# the same character at the same byte position on the same GGUF, making
# it a DWQ-quant weight artifact rather than an hf2q bug. The only real
# hf2q-vs-llama.cpp drift in 1000 decoded tokens is a single-letter case
# flip at decode token 840 (' On' vs ' ON' in "Phase 1 (Lid On/ON)"),
# which this gate's 3094-byte floor catches: if anything widens the drift
# earlier than byte 3094, the gate fails.
#
# Usage:
#   scripts/sourdough_gate.sh <gguf_path>
#   scripts/sourdough_gate.sh <gguf_path> --min-prefix 3094
#
# Exit codes:
#   0  common prefix >= MIN_COMMON_PREFIX (gate passed)
#   1  usage / env error
#   2  gate failed (common prefix < MIN_COMMON_PREFIX)
#   3  tool invocation failure (hf2q or llama-completion crashed)
#
# Example (post-1bNEW.20.FIX reference):
#   hf2q:  3656 bytes
#   llama: 3658 bytes
#   common prefix: 3095 bytes
#   -> PASS (>= 3094)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HF2Q_BIN="target/release/hf2q"
OUT_HF2Q="/tmp/sourdough_gate_hf2q.txt"; LOG_HF2Q="/tmp/sourdough_gate_hf2q.log"
OUT_LLAMA="/tmp/sourdough_gate_llama.txt"; LOG_LLAMA="/tmp/sourdough_gate_llama.log"
RENDERED_PROMPT="/tmp/sourdough_gate_rendered.txt"
RENDERED_PROMPT_LLAMA="/tmp/sourdough_gate_rendered_nobos.txt"

# The user's original prompt from the 2026-04-11 investigation. Tokenizes
# to exactly 22 tokens under the Gemma 4 tokenizer. DO NOT "fix" the typo
# ("Complrehensive") — it is load-bearing for the fixture: the typo is
# what the tokenizer sees, so the forward trajectory is locked to the
# specific token sequence that hit the byte-1204 たand byte-3095 On/ON
# positions we benchmarked against. A different typo would produce a
# different trajectory and a different drift profile.
USER_PROMPT="Complrehensive instructions for making sourdough bread."
MAX_TOKENS=1000
MIN_COMMON_PREFIX=3094

usage() {
  cat <<EOF
Usage: scripts/sourdough_gate.sh <gguf_path> [--min-prefix N] [--max-tokens N]
  <gguf_path>         Path to the Gemma 4 GGUF model (required)
  --min-prefix N      Common byte prefix floor (default: 3094)
  --max-tokens N      Max decode tokens per run (default: 1000)

Exit codes:
  0  gate passed
  1  usage / env error
  2  gate failed (drift earlier than min-prefix)
  3  tool invocation failure
EOF
}
err() { echo "error: $*" >&2; exit 1; }

GGUF_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --min-prefix)
      [[ $# -ge 2 ]] || err "--min-prefix requires an argument"
      MIN_COMMON_PREFIX="$2"; shift 2 ;;
    --max-tokens)
      [[ $# -ge 2 ]] || err "--max-tokens requires an argument"
      MAX_TOKENS="$2"; shift 2 ;;
    -*) usage >&2; err "unknown flag: $1" ;;
    *)  [[ -z "$GGUF_PATH" ]] || err "unexpected positional arg: $1"
        GGUF_PATH="$1"; shift ;;
  esac
done
[[ -n "$GGUF_PATH" ]] || { usage >&2; exit 1; }
[[ -f "$GGUF_PATH" ]] || err "GGUF file not found: $GGUF_PATH"
[[ -x "$HF2Q_BIN"  ]] || err "hf2q binary not found at $HF2Q_BIN (run: cargo build --release --features metal)"

if command -v llama-completion >/dev/null 2>&1; then
  LLAMA_BIN="$(command -v llama-completion)"
elif [[ -x "/opt/llama.cpp/build/bin/llama-completion" ]]; then
  LLAMA_BIN="/opt/llama.cpp/build/bin/llama-completion"
else
  err "llama-completion not found on PATH or at /opt/llama.cpp/build/bin/llama-completion"
fi

GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "=== Sourdough byte-prefix gate (ADR-005 Phase 1b post-1bNEW.20.FIX) ==="
echo "GGUF:           $GGUF_PATH"
echo "hf2q:           $HF2Q_BIN"
echo "llama-comp:     $LLAMA_BIN"
echo "git HEAD:       $GIT_HEAD"
echo "prompt:         $USER_PROMPT"
echo "max_tokens:     $MAX_TOKENS"
echo "min-prefix:     $MIN_COMMON_PREFIX bytes"
echo

# 1. Pre-render the chat template via hf2q, BOS-stripped for llama-completion.
#    Same discipline as crawl_verify.sh so both tools see byte-identical
#    token-level input. See crawl_verify.sh:96-183 for the long Chesterton
#    fence on why we strip the leading literal <bos>.
echo "--- Rendering chat template via hf2q ---"
if ! HF2Q_DUMP_RENDERED_PROMPT="$RENDERED_PROMPT" \
      "$HF2Q_BIN" generate --model "$GGUF_PATH" --prompt "$USER_PROMPT" \
        --max-tokens 1 --temperature 0 \
        >/dev/null 2>"$LOG_HF2Q"; then
  echo "hf2q render failed. See $LOG_HF2Q" >&2; exit 3
fi
[[ -s "$RENDERED_PROMPT" ]] || err "rendered prompt is empty"

python3 - "$RENDERED_PROMPT" "$RENDERED_PROMPT_LLAMA" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
data = open(src, "rb").read()
if not data.startswith(b"<bos>"):
    sys.stderr.write("error: rendered prompt does not start with literal '<bos>'\n")
    sys.exit(1)
open(dst, "wb").write(data[5:])
PY
[[ -s "$RENDERED_PROMPT_LLAMA" ]] || err "BOS-stripped prompt is empty"

# 2. Run llama-completion on the BOS-stripped rendered prompt.
echo "--- Running llama-completion (T=0 greedy, $MAX_TOKENS tokens) ---"
if ! "$LLAMA_BIN" --model "$GGUF_PATH" --file "$RENDERED_PROMPT_LLAMA" \
      --predict "$MAX_TOKENS" --temp 0 --seed 42 \
      --no-display-prompt -no-cnv \
      -st -ngl 999 \
      </dev/null >"$OUT_LLAMA" 2>"$LOG_LLAMA"; then
  echo "llama-completion failed. See $LOG_LLAMA" >&2; exit 3
fi
if grep -q "prompt also starts with a BOS token" "$LOG_LLAMA"; then
  echo "error: llama-completion reports double-BOS — BOS-strip broken." >&2; exit 1
fi

# 3. Run hf2q on the raw user prompt (it applies the chat template
#    internally and produces byte-identical input tokens to step 1).
echo "--- Running hf2q (T=0 greedy, $MAX_TOKENS tokens) ---"
if ! "$HF2Q_BIN" generate --model "$GGUF_PATH" --prompt "$USER_PROMPT" \
      --max-tokens "$MAX_TOKENS" --temperature 0 \
      >"$OUT_HF2Q" 2>"$LOG_HF2Q"; then
  echo "hf2q failed. See $LOG_HF2Q" >&2; exit 3
fi

# 4. Common-byte-prefix diff.
# hf2q prints a 4-line header on stdout (src/serve/header.rs, commit 172488b):
#   1: "hf2q · <chip> · <backend>"
#   2: "<model> · loaded in <s>s · <n> layers · <gb> GB"
#   3: "prefill: <n> tok in <ms>ms (<tok/s> tok/s)"
#   4: blank line
# Strip by skipping the first 4 newline-terminated lines before diffing.
#
# Write TSV to a temp file instead of process substitution — the python
# heredoc contains literal ) which bash's <( ... ) parser miscounts.
DIFF_TSV="/tmp/sourdough_gate_diff.tsv"
python3 - "$OUT_LLAMA" "$OUT_HF2Q" "$DIFF_TSV" <<'PY'
import sys, json
a = open(sys.argv[1], "rb").read()
b_raw = open(sys.argv[2], "rb").read()
out = sys.argv[3]
if b_raw.startswith(b"hf2q \xc2\xb7 "):
    pos = 0
    for _ in range(4):
        nl = b_raw.find(b"\n", pos)
        if nl < 0:
            break
        pos = nl + 1
    b = b_raw[pos:]
else:
    b = b_raw
n = min(len(a), len(b))
i = 0
while i < n and a[i] == b[i]:
    i += 1
def snip(buf, start, length=120):
    s = buf[start:start+length].decode("utf-8", errors="replace")
    if len(buf) - start > length:
        s += "..."
    return json.dumps(s)
with open(out, "w") as f:
    f.write(f"{len(a)}\t{len(b)}\t{i}\t{snip(a, i)}\t{snip(b, i)}\n")
PY
IFS=$'\t' read -r LLAMA_BYTES HF2Q_BYTES COMMON_BYTES DIVERGE_A DIVERGE_B < "$DIFF_TSV"

echo
echo "--- Comparison ---"
echo "llama output:  $LLAMA_BYTES bytes"
echo "hf2q  output:  $HF2Q_BYTES bytes"
echo "common prefix: $COMMON_BYTES bytes"
echo "min required:  $MIN_COMMON_PREFIX bytes"
echo

if [[ "$COMMON_BYTES" -lt "$MIN_COMMON_PREFIX" ]]; then
  echo "FAIL: drift widened vs the post-1bNEW.20.FIX baseline."
  echo
  echo "First 120 chars after byte $COMMON_BYTES:"
  echo "  llama: $DIVERGE_A"
  echo "  hf2q:  $DIVERGE_B"
  echo
  echo "This gate exists to catch any speed item that accidentally widens"
  echo "the hf2q-vs-llama.cpp correctness drift. Bisect the most recent"
  echo "src/ change that could have touched the forward pass (attention,"
  echo "MoE, norms, RoPE, lm_head, KV cache, SDPA) and verify against this"
  echo "gate before landing."
  exit 2
fi

echo "PASS: common prefix $COMMON_BYTES >= $MIN_COMMON_PREFIX."
if [[ "$COMMON_BYTES" -gt "$MIN_COMMON_PREFIX" ]]; then
  echo "      (drift is $((COMMON_BYTES - MIN_COMMON_PREFIX)) bytes tighter than the floor)"
fi
exit 0
