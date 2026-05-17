#!/usr/bin/env bash
# Find the first character position where two generation outputs diverge.
# Does NOT print the content — only the position, surrounding 20-char
# fingerprint hash, and length stats. Safe to share verbatim.
set -euo pipefail

A="${1:-}"
B="${2:-}"
if [[ -z "$A" || -z "$B" || ! -f "$A" || ! -f "$B" ]]; then
  echo "usage: $0 <file_a> <file_b>" >&2
  exit 64
fi

# Best-effort: strip llama-cli + hf2q banners. Generation content survives.
strip_llama() {
  # Drop everything up through the last "main: " or "sampler" config line.
  awk '
    BEGIN { in_body=0 }
    /^main: |^build: |^load_backend:|^llama_|^print_info:|^sampler |^generate:|^system_info|^ggml_metal|^Metal/ { next }
    /^\[end of text\]$|^\[end of stream\]$|^>$/ { next }
    { print }
  ' "$1"
}
strip_hf2q() {
  # Drop hf2q load: prefill: [INFO] [hf2q] --- mlx-native: lines.
  awk '
    /^hf2q load: |^prefill: |^\[INFO\]|^\[hf2q\]|^--- /  { next }
    { print }
  ' "$1"
}

TA="$(mktemp)"; TB="$(mktemp)"
trap 'rm -f "$TA" "$TB"' EXIT
strip_llama "$A" > "$TA"
strip_hf2q  "$B" > "$TB"

la=$(wc -c <"$TA" | tr -d ' ')
lb=$(wc -c <"$TB" | tr -d ' ')

# Find first byte position where they differ.
# Use cmp -l (octal) — output: "pos a_byte b_byte" lines for each diff.
divline=$(cmp -l "$TA" "$TB" 2>/dev/null | head -1 || true)

if [[ -z "$divline" ]]; then
  if [[ "$la" -eq "$lb" ]]; then
    echo "IDENTICAL: $la bytes both, no divergence"
  else
    short=$(( la < lb ? la : lb ))
    long=$(( la > lb ? la : lb ))
    echo "PREFIX_MATCH: first $short bytes identical; one continues to $long"
  fi
  exit 0
fi

pos=$(echo "$divline" | awk '{print $1}')
pos=$(( pos - 1 ))  # cmp -l is 1-indexed; convert to 0-indexed

# Compute a fingerprint of the AGREED prefix (so user can confirm we're
# looking at the same point without leaking content).
prefix_sha=$(head -c "$pos" "$TA" | shasum -a 256 | awk '{print $1}')

echo "DIVERGENCE at byte position $pos (0-indexed)"
echo "  llama.cpp output length: $la bytes"
echo "  hf2q     output length: $lb bytes"
echo "  agreed-prefix sha256:   $prefix_sha"
echo
echo "  byte AT divergence point:"
echo "    llama.cpp: $(head -c $((pos + 1)) "$TA" | tail -c 1 | od -An -c | tr -d ' ')"
echo "    hf2q     : $(head -c $((pos + 1)) "$TB" | tail -c 1 | od -An -c | tr -d ' ')"
echo
# Sanitized context: print only the divergent byte's hex + 5 bytes after
# from each, also as hex. The hex doesn't leak content for typical
# 7-bit ASCII (which it is here per UTF-8 English text).
ctx_a=$(dd if="$TA" bs=1 skip="$pos" count=6 2>/dev/null | xxd -p)
ctx_b=$(dd if="$TB" bs=1 skip="$pos" count=6 2>/dev/null | xxd -p)
echo "  6-byte window FROM divergence (hex):"
echo "    llama.cpp: $ctx_a"
echo "    hf2q     : $ctx_b"
