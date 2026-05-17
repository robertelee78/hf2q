#!/usr/bin/env bash
# EXP-4 — Content-blind hf2q-vs-llama cross-runtime comparison harness.
#
# When you (the operator) are testing an uncensored model with a prompt
# whose content you don't want to share with the assistant, this script
# runs the SAME prompt under hf2q and under llama-cli, then emits ONLY
# structural metrics — never the prompt, never the response content.
#
# Usage:
#   scripts/exp4-hf2q-vs-llama-content-blind.sh \
#       --model models/.../gemma4-ara-2pass-APEX-Q5_K_M.gguf \
#       --prompt-file /tmp/private_prompt.txt
#
# Optional env:
#   LLAMA_BIN  default /opt/llama.cpp/build/bin/llama-cli
#   HF2Q_BIN   default target/release/hf2q
#   MAX_TOKENS default 300
#
# Output (safe to share):
#   prompt_bytes  : N
#   hf2q_bytes    : N      (output bytes, content NOT shown)
#   llama_bytes   : N
#   common_prefix : N      (cp(hf2q_output, llama_output))
#   template_ratio_hf2q  : %    (heuristic: count of corporate-template-bigrams)
#   template_ratio_llama : %
#   first_diff_byte_offset : N  (where hf2q and llama outputs diverge)
#   hf2q_sha256 : <8 prefix>    (hashes prove byte-identity across runs)
#   llama_sha256 : <8 prefix>
#
# If template_ratio_hf2q ≈ template_ratio_llama (both high) → model is
# falling into template-fallback on this prompt class regardless of
# runtime → MODEL ISSUE (abliteration partial-refusal pathway).
#
# If template_ratio_hf2q >> template_ratio_llama → hf2q has a real bug
# on this prompt class → HF2Q ISSUE → bisect kernel paths.
#
# If both low → output is on-topic and the "failure mode" reported was
# transient (re-run nondeterminism).

set -euo pipefail

MODEL=""
PROMPT_FILE=""
MAX_TOKENS="${MAX_TOKENS:-300}"
HF2Q_BIN="${HF2Q_BIN:-target/release/hf2q}"
LLAMA_BIN="${LLAMA_BIN:-/opt/llama.cpp/build/bin/llama-cli}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --max-tokens)  MAX_TOKENS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$MODEL" ]]       || { echo "--model required" >&2; exit 1; }
[[ -n "$PROMPT_FILE" ]] || { echo "--prompt-file required" >&2; exit 1; }
[[ -f "$MODEL" ]]       || { echo "model not found: $MODEL" >&2; exit 1; }
[[ -f "$PROMPT_FILE" ]] || { echo "prompt file not found: $PROMPT_FILE" >&2; exit 1; }
[[ -x "$HF2Q_BIN" ]]    || { echo "hf2q binary not at $HF2Q_BIN (cargo build --release)" >&2; exit 1; }
[[ -x "$LLAMA_BIN" ]]   || { echo "llama-cli not at $LLAMA_BIN (set LLAMA_BIN)" >&2; exit 1; }

TS="$(date +%s)"
OUT_DIR="/tmp/exp4-content-blind-${TS}"
mkdir -p "$OUT_DIR"

PROMPT_BYTES="$(wc -c < "$PROMPT_FILE" | tr -d ' ')"

# Run hf2q.  Strip everything before the first decoded byte by skipping
# the load banner + prefill line.  Output saved to hf2q.txt.
env -u HF2Q_USE_DENSE -u HF2Q_LAYER_POLICY -u HF2Q_TQ_CODEBOOK_BITS \
    "$HF2Q_BIN" generate \
        --model "$MODEL" \
        --prompt-file "$PROMPT_FILE" \
        --max-tokens "$MAX_TOKENS" \
        --temperature 0 \
    2>"$OUT_DIR/hf2q.stderr" \
    | sed -n '/^prefill:/,$p' | tail -n +3 \
    > "$OUT_DIR/hf2q.txt"

# Run llama-cli.  -no-cnv disables conversation mode, --temp 0 deterministic.
"$LLAMA_BIN" \
    -m "$MODEL" \
    -f "$PROMPT_FILE" \
    -n "$MAX_TOKENS" \
    --temp 0 \
    --top-p 1.0 \
    --top-k 1 \
    --repeat-penalty 1.0 \
    -no-cnv \
    --no-warmup \
    --jinja \
    > "$OUT_DIR/llama.raw" 2>"$OUT_DIR/llama.stderr"

# llama-cli echoes the prompt; strip it (heuristic: drop everything up to
# and including the prompt bytes if it appears verbatim, else keep raw).
if grep -qF -f "$PROMPT_FILE" "$OUT_DIR/llama.raw"; then
    awk -v prompt_file="$PROMPT_FILE" '
        BEGIN {
            while ((getline line < prompt_file) > 0) prompt = prompt line "\n"
            close(prompt_file); sub(/\n$/, "", prompt)
            seen = 0
        }
        seen { print; next }
        index($0, prompt) > 0 {
            seen = 1
            tail = substr($0, index($0, prompt) + length(prompt))
            if (length(tail) > 0) print tail
        }
    ' "$OUT_DIR/llama.raw" > "$OUT_DIR/llama.txt"
else
    cp "$OUT_DIR/llama.raw" "$OUT_DIR/llama.txt"
fi

HF2Q_BYTES="$(wc -c < "$OUT_DIR/hf2q.txt" | tr -d ' ')"
LLAMA_BYTES="$(wc -c < "$OUT_DIR/llama.txt" | tr -d ' ')"
HF2Q_SHA="$(shasum -a 256 "$OUT_DIR/hf2q.txt" | cut -c1-8)"
LLAMA_SHA="$(shasum -a 256 "$OUT_DIR/llama.txt" | cut -c1-8)"

# Common prefix in bytes (Python — deterministic + handles binary).
COMMON_PREFIX="$(python3 -c '
import sys
a = open(sys.argv[1],"rb").read()
b = open(sys.argv[2],"rb").read()
n = min(len(a),len(b)); i = 0
while i < n and a[i] == b[i]: i += 1
print(i)
' "$OUT_DIR/hf2q.txt" "$OUT_DIR/llama.txt")"

# Template-fallback ratio heuristic — count corporate boilerplate bigrams /
# total words.  These are phrases that appear in soft-refusal corporate-
# report output.  Case-insensitive.
TEMPLATE_TOKENS="Integrated Comprehensive Robust Risk Management Compliance Framework Implementation Stakeholder Strategic Alignment Governance Methodology Optimization Implementation Effectiveness Monitoring Assessment Mitigation Operational Excellence Continuous Improvement"
template_ratio() {
    local f="$1"
    local words; words=$(wc -w < "$f" | tr -d ' ')
    [[ "$words" -gt 0 ]] || { echo "0.0"; return; }
    local hits=0
    for tok in $TEMPLATE_TOKENS; do
        local n; n=$(grep -i -o -w "$tok" "$f" 2>/dev/null | wc -l | tr -d ' ')
        hits=$((hits + n))
    done
    python3 -c "print(f'{$hits / $words * 100:.1f}')"
}

HF2Q_TPL="$(template_ratio "$OUT_DIR/hf2q.txt")"
LLAMA_TPL="$(template_ratio "$OUT_DIR/llama.txt")"

cat <<EOF
=== EXP-4 content-blind hf2q vs llama-cli comparison ===
out dir              : $OUT_DIR
model_sha256         : $(shasum -a 256 "$MODEL" | cut -c1-16)
prompt_bytes         : $PROMPT_BYTES
hf2q_bytes           : $HF2Q_BYTES
llama_bytes          : $LLAMA_BYTES
common_prefix        : $COMMON_PREFIX
first_diff_byte      : $COMMON_PREFIX
hf2q_sha256_8        : $HF2Q_SHA
llama_sha256_8       : $LLAMA_SHA
template_ratio_hf2q  : ${HF2Q_TPL}%
template_ratio_llama : ${LLAMA_TPL}%

DIAGNOSIS HEURISTIC:
  - both template_ratio ≥ 3.0%   → model is template-fallback regardless of runtime → MODEL ISSUE (abliteration partial-refusal)
  - hf2q_ratio ≥ 3.0%, llama_ratio < 1.0% → HF2Q ISSUE on this prompt class
  - both < 1.0%, common_prefix > 50 → output is on-topic; report may have been transient

Outputs saved under $OUT_DIR (you can inspect or share them).
EOF
