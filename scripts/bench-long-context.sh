#!/usr/bin/env bash
# ADR-007 Path C F-4.2/F-4.3: Long-context memory + bandwidth bench.
#
# Runs hf2q generate at multiple prompt sizes and captures peak RSS,
# prefill/decode t/s, and any error signals. Output:
#   /tmp/f04-bench/{8k,32k,64k}/{stdout.log,stderr.log,bench.json}
#
# Memory measurement uses /usr/bin/time -l "maximum resident set size"
# which on macOS reports peak RSS in bytes. On Apple Silicon with
# unified memory, this includes GPU buffer allocations.
#
# Usage: scripts/bench-long-context.sh [tokens]
#   tokens: comma-separated prompt-target token counts. Default: 8192,32768
#
# Tokenizer is GPT-style ~4 chars/token, so we replicate a base text
# segment to produce N tokens. The tokenizer + model will normalize
# this into the actual prompt length, which we capture from stderr.

set -euo pipefail

MODEL="/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf"
HF2Q="/opt/hf2q/target/release/hf2q"
OUT_ROOT="/tmp/f04-bench"
TOKENS_DEFAULT="8192,32768"
TOKENS="${1:-$TOKENS_DEFAULT}"

# Base text replicated to fill the prompt.
BASE_TEXT="The quick brown fox jumps over the lazy dog. "

mkdir -p "$OUT_ROOT"

generate_prompt() {
    local target_tokens="$1"
    local target_chars=$(( target_tokens * 5 ))    # ~5 chars per token (tokenizer-conservative)
    local base_len=${#BASE_TEXT}
    local repeats=$(( target_chars / base_len + 1 ))
    yes "$BASE_TEXT" | head -n "$repeats" | tr -d '\n'
}

run_one() {
    local target="$1"
    local outdir="$OUT_ROOT/${target}tok"
    mkdir -p "$outdir"
    local prompt_file="$outdir/prompt.txt"
    generate_prompt "$target" > "$prompt_file"
    local prompt_chars=$(wc -c < "$prompt_file" | tr -d ' ')

    echo "=== F-4 bench at target $target tokens ($prompt_chars chars) ==="
    local time_log="$outdir/time.log"
    local stderr_log="$outdir/stderr.log"
    local stdout_log="$outdir/stdout.log"

    /usr/bin/time -l "$HF2Q" generate \
        --model "$MODEL" \
        --prompt-file "$prompt_file" \
        --max-tokens 5 \
        > "$stdout_log" 2> "$time_log" \
        || { echo "FAILED at target $target"; tail -10 "$time_log"; return 1; }

    # Split: hf2q logs to stderr; /usr/bin/time appends time stats.
    cp "$time_log" "$stderr_log"

    # Parse stats.
    local actual_tokens=$(grep -oE "prefill: [0-9]+" "$time_log" | head -1 | grep -oE "[0-9]+" || echo "0")
    local prefill_ms=$(grep -oE "prefill:.*[0-9]+ms" "$time_log" | head -1 | grep -oE "[0-9]+ms" | grep -oE "[0-9]+" || echo "0")
    local prefill_tps=$(grep -oE "prefill:.*\([0-9]+ tok/s\)" "$time_log" | head -1 | grep -oE "[0-9]+ tok/s" | grep -oE "[0-9]+" || echo "0")
    local decode_tps=$(grep -oE "[0-9]+\.[0-9]+ tok/s" "$time_log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "0")
    local peak_rss_bytes=$(grep -oE "[0-9]+ +maximum resident set size" "$time_log" | grep -oE "^[0-9]+" | head -1 || echo "0")
    local peak_rss_gb=$(awk "BEGIN { printf \"%.2f\", $peak_rss_bytes / (1024.0*1024.0*1024.0) }")
    local wall_sec=$(grep -oE "[0-9]+\.[0-9]+ real" "$time_log" | grep -oE "^[0-9]+\.[0-9]+" || echo "0")

    cat > "$outdir/bench.json" <<EOF
{
  "target_tokens": $target,
  "actual_prefill_tokens": $actual_tokens,
  "prefill_ms": $prefill_ms,
  "prefill_tok_per_s": $prefill_tps,
  "decode_tok_per_s": "$decode_tps",
  "peak_rss_bytes": $peak_rss_bytes,
  "peak_rss_gb": $peak_rss_gb,
  "wall_seconds": "$wall_sec"
}
EOF
    echo "  prefill: $actual_tokens tok in ${prefill_ms}ms ($prefill_tps tok/s)"
    echo "  decode:  $decode_tps tok/s"
    echo "  peak RSS: $peak_rss_gb GB"
    echo "  wall:    $wall_sec s"
    echo
}

IFS=',' read -ra TOKENS_ARR <<< "$TOKENS"
for t in "${TOKENS_ARR[@]}"; do
    run_one "$t" || echo "skipping $t"
done

# Aggregate.
echo "=== Aggregate ==="
echo "target prefill_tok decode_t/s peak_GB wall_s"
for t in "${TOKENS_ARR[@]}"; do
    f="$OUT_ROOT/${t}tok/bench.json"
    if [[ -f "$f" ]]; then
        python3 -c "
import json
d = json.load(open('$f'))
print(f'{d[\"target_tokens\"]:>6} {d[\"actual_prefill_tokens\"]:>10} {d[\"decode_tok_per_s\"]:>10} {d[\"peak_rss_gb\"]:>7} {d[\"wall_seconds\"]:>6}')
"
    fi
done
