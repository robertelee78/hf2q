#!/usr/bin/env bash
# Extract generation content from each runtime's captured output, then
# diff. Does not print content — only positions, sizes, and hex windows.
set -euo pipefail

LLAMA_RAW="${1:-/tmp/llama-20k.out}"
HF2Q_RAW="${2:-/tmp/bug-gemma-run1/hf2q.stdout}"

if [[ ! -f "$LLAMA_RAW" ]]; then echo "missing: $LLAMA_RAW" >&2; exit 64; fi
if [[ ! -f "$HF2Q_RAW"  ]]; then echo "missing: $HF2Q_RAW"  >&2; exit 64; fi

# Extract generation block from llama-cli output captured via script(1).
# Strategy: take everything BETWEEN the last banner line and the first
# footer line. Banner lines are recognized by their characteristic
# library-prefixed format.
extract_llama() {
  awk '
    BEGIN {
      banner_re = "^(llama_|ggml_|print_info:|common_|main: |build: |build_info: |sampler |generate: |system_info|load_backend:|register_backend:|Metal |MTL |GGML_|init_|tokenize:|chat_template:|samplers:|sampling params|repeat_|top_|min_|temp |seed|mirostat|dynatemp)"
      footer_re = "^(llama_perf_|common_memory_breakdown|ggml_metal_free|Exiting\\.\\.\\.|\\[end of text\\])"
      in_gen = 0
    }
    {
      if (in_gen == 0) {
        if ($0 ~ banner_re || /^$/ || /^Available/ || /^\.\.\.$/) next
        in_gen = 1
      }
      if (in_gen == 1 && $0 ~ footer_re) { exit }
      print
    }
  ' "$1"
}

# Extract generation block from hf2q output. hf2q banners are well-defined.
extract_hf2q() {
  awk '
    BEGIN {
      banner_re = "^(hf2q load:|prefill:|\\[INFO\\]|\\[hf2q\\]|--- |Available)"
      footer_re = "^(\\[INFO\\] Generated|\\[hf2q\\] Gemma|--- mlx-native:)"
      in_gen = 0
    }
    {
      if (in_gen == 0) {
        if ($0 ~ banner_re || /^$/) next
        in_gen = 1
      }
      if (in_gen == 1 && $0 ~ footer_re) { exit }
      print
    }
  ' "$1"
}

TA="$(mktemp)"; TB="$(mktemp)"
trap 'rm -f "$TA" "$TB"' EXIT

extract_llama "$LLAMA_RAW" > "$TA"
extract_hf2q  "$HF2Q_RAW"  > "$TB"

la=$(wc -c <"$TA" | tr -d ' ')
lb=$(wc -c <"$TB" | tr -d ' ')

echo "=== extraction sizes ==="
echo "  llama.cpp generation block: $la bytes"
echo "  hf2q     generation block: $lb bytes"
echo

# Strip ANSI escape codes from llama-cli's captured pty stream
# (script(1) preserves them; they'd confuse cmp).
sed -E $'s/\\x1b\\[[0-9;]*[A-Za-z]//g' "$TA" > "$TA.clean" && mv "$TA.clean" "$TA"
sed -E $'s/\\x1b\\[[0-9;]*[A-Za-z]//g' "$TB" > "$TB.clean" && mv "$TB.clean" "$TB"

la2=$(wc -c <"$TA" | tr -d ' ')
lb2=$(wc -c <"$TB" | tr -d ' ')
echo "=== after ANSI strip ==="
echo "  llama.cpp: $la2 bytes"
echo "  hf2q     : $lb2 bytes"
echo

divline=$(cmp -l "$TA" "$TB" 2>/dev/null | head -1 || true)

if [[ -z "$divline" ]]; then
  if [[ "$la2" -eq "$lb2" ]]; then
    echo "IDENTICAL: $la2 bytes both"
  else
    short=$(( la2 < lb2 ? la2 : lb2 ))
    long=$(( la2 > lb2 ? la2 : lb2 ))
    echo "PREFIX_MATCH: first $short bytes identical; one continues to $long"
    if [[ "$la2" -lt "$lb2" ]]; then
      echo "  (hf2q is LONGER than llama.cpp — possibly still looping past llama.cpp's natural EOS)"
    else
      echo "  (llama.cpp is LONGER than hf2q — hf2q stopped early)"
    fi
  fi
  echo "  (no divergent byte found in common prefix)"
  exit 0
fi

pos=$(echo "$divline" | awk '{print $1}')
pos=$(( pos - 1 ))

if [[ "$pos" -eq 0 ]]; then
  prefix_sha="(no agreed prefix)"
else
  prefix_sha=$(head -c "$pos" "$TA" | shasum -a 256 | awk '{print $1}')
fi

# Try to count newlines in prefix — gives a rough "line N" reference
if [[ "$pos" -eq 0 ]]; then
  prefix_lines=0
else
  prefix_lines=$(head -c "$pos" "$TA" | tr -cd '\n' | wc -c | tr -d ' ')
fi

# Estimate token position: ~4 bytes per English token avg
est_tokens=$(( pos / 4 ))

echo "=== divergence ==="
echo "  first divergent byte: position $pos (0-indexed)"
echo "  approx token offset:  ~$est_tokens (assuming ~4 bytes/token)"
echo "  approx line offset:   line $prefix_lines"
echo "  agreed-prefix sha256: $prefix_sha"
echo
ctx_a=$(dd if="$TA" bs=1 skip="$pos" count=8 2>/dev/null | xxd -p)
ctx_b=$(dd if="$TB" bs=1 skip="$pos" count=8 2>/dev/null | xxd -p)
echo "  8-byte hex window FROM divergence:"
echo "    llama.cpp: $ctx_a"
echo "    hf2q     : $ctx_b"
echo
echo "  files preserved:"
echo "    $TA (llama clean generation)"
echo "    $TB (hf2q clean generation)"
# Disable trap so files persist
trap - EXIT
