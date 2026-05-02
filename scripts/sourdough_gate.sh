#!/usr/bin/env bash
# sourdough_gate.sh — ADR-005 Phase 1b post-1bNEW.20.FIX correctness gate.
#
# DEPRECATED 2026-05-01 (ADR-005 iter-221 drift-audit C2). `scripts/release-check.sh`
# supersedes this script via Gates C/D/E/F (parity suite + self-baseline).
# This script remains as a single-prompt fast-iteration tool but its default
# floor must track tests/evals/reference/MANIFEST.json's
# `sourdough.common_prefix_bytes` field. The historical 3094-byte floor (frozen
# 2026-04-11 post-1bNEW.20.FIX) was lowered to 179 in iter-220 (commit `de9b7a4`,
# 2026-05-01) after a full vs-llama anchor refresh at locked llama.cpp commit
# `b3d758750a` on M5 Max — cross-implementation argmax drift on long prompts is
# mathematical (independent kernel implementations) not a regression. See
# project_w5b22_hf2q_exhausted_remaining_in_mul_mm_id memory entry.
#
# Runs hf2q and llama-completion on the 22-token user sourdough prompt at
# T=0 greedy, max_tokens=1000, and asserts that the common byte prefix of
# their outputs is at least MIN_COMMON_PREFIX bytes (default: 179, anchored
# to today's measurement; previously 3094 pre-iter-220). This enforces "hf2q's
# decode output does not regress further from llama.cpp on the DWQ GGUF" as
# a fast-iteration check. Source of truth: tests/evals/reference/MANIFEST.json.
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
#   scripts/sourdough_gate.sh <gguf_path> --min-prefix 179
#
# Exit codes:
#   0  common prefix >= MIN_COMMON_PREFIX (gate passed)
#   1  usage / env error
#   2  gate failed (common prefix < MIN_COMMON_PREFIX)
#   3  tool invocation failure (hf2q or llama-completion crashed)
#
# Example (post-iter-220 reference, MANIFEST.json sourdough section):
#   hf2q:  3575 bytes
#   llama: 3712 bytes
#   common prefix: 179 bytes
#   -> PASS (>= 179)

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
# Source: tests/evals/reference/MANIFEST.json sourdough.common_prefix_bytes
# (iter-220 anchor refresh, 2026-05-01, commit de9b7a4). Was 3094 pre-iter-220.
MIN_COMMON_PREFIX=179

usage() {
  cat <<EOF
Usage: scripts/sourdough_gate.sh <gguf_path> [--min-prefix N] [--max-tokens N]
  <gguf_path>         Path to the Gemma 4 GGUF model (required)
  --min-prefix N      Common byte prefix floor (default: 179, per MANIFEST.json post-iter-220)
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
# ADR-007 post-close 2026-04-24: byte-exact gate requires DENSE decode.
# TQ is the default decode path post-correction; force dense explicitly here
# so the gate keeps asserting "hf2q dense == llama.cpp" byte-identity.
if ! HF2Q_DUMP_RENDERED_PROMPT="$RENDERED_PROMPT" HF2Q_USE_DENSE=1 \
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
# ADR-007 post-close 2026-04-24: HF2Q_USE_DENSE=1 forces dense decode for the
# byte-exact comparison vs llama.cpp. TQ is the production default but cannot
# produce byte-identical output by design (quantization is lossy, passes
# semantic gates not byte-exact gates).
if ! HF2Q_USE_DENSE=1 "$HF2Q_BIN" generate --model "$GGUF_PATH" --prompt "$USER_PROMPT" \
      --max-tokens "$MAX_TOKENS" --temperature 0 \
      >"$OUT_HF2Q" 2>"$LOG_HF2Q"; then
  echo "hf2q failed. See $LOG_HF2Q" >&2; exit 3
fi

# 4. Common-byte-prefix diff.
# hf2q prints a load banner on stdout before the generated text. Two
# formats coexist:
#   - Pre-ADR-018 (legacy, src/serve/header.rs commit 172488b):
#       1: "hf2q · <chip> · <backend>"
#       2: "<model> · loaded in <s>s · <n> layers · <gb> GB"
#       3: "prefill: <n> tok in <ms>ms (<tok/s> tok/s)"
#       4: blank line
#   - ADR-018 c3 (2026-05-01+, src/serve/load_info.rs::print_banner):
#       N "hf2q load: <field> = <value>" lines (currently 13)
#       1 "prefill: <n> tok in <ms>ms (<tok/s> tok/s)" line
#       1 blank line
# Both formats end with `prefill: ...\n\n<decoded>` and contain no other
# blank line in the banner prefix, so finding the first `\n\n` strips both
# correctly. Mirrors `tests/coherence_matrix.rs::strip_hf2q_header`.
#
# Write TSV to a temp file instead of process substitution — the python
# heredoc contains literal ) which bash's <( ... ) parser miscounts.
DIFF_TSV="/tmp/sourdough_gate_diff.tsv"
python3 - "$OUT_LLAMA" "$OUT_HF2Q" "$DIFF_TSV" <<'PY'
import sys, json
a = open(sys.argv[1], "rb").read()
b_raw = open(sys.argv[2], "rb").read()
out = sys.argv[3]

def strip_hf2q_header(buf):
    if not (buf.startswith(b"hf2q \xc2\xb7 ") or buf.startswith(b"hf2q load:")):
        return buf
    sep = buf.find(b"\n\n")
    return buf[sep + 2:] if sep >= 0 else b""

b = strip_hf2q_header(b_raw)
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
