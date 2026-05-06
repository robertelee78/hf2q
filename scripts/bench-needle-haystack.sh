#!/usr/bin/env bash
# ADR-007 Path C F-4.4: Needle-in-haystack retrieval at long context.
#
# Plants a unique fact ("the secret code is XXXXX") at a random position
# in a long-context prompt and asks the model to retrieve it. Reports
# pass/fail per (length, position) cell.
#
# Usage: scripts/bench-needle-haystack.sh [lengths] [trials_per_length]
#   lengths: comma-separated target context-token counts. Default: 4096,8192,16384
#   trials_per_length: int, default 3 (one per fractional position bucket)
#
# Position fractions: 0.1 (early), 0.5 (middle), 0.9 (late) — covers the
# typical needle-in-haystack failure modes (early vs late attention).
#
# A "PASS" requires the model to produce the exact 5-char secret code
# in its response. Anything else is "FAIL" (and we capture the response
# for debugging).

set -euo pipefail

MODEL="/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf"
HF2Q="/opt/hf2q/target/release/hf2q"
OUT_ROOT="/tmp/f04-needle"
LENGTHS="${1:-4096,8192,16384}"
TRIALS="${2:-3}"

mkdir -p "$OUT_ROOT"

# Filler text segments — varied so the haystack isn't pure repetition,
# which models can game. Each line ~12 tokens.
FILLER_LINES=(
"In the depths of the forest, leaves whispered secrets to each other under starlight."
"The blacksmith's hammer rang clear against the anvil as dawn broke over the village."
"Mathematicians have long debated whether infinity should be considered a number."
"Old maps show the river once ran through what is now the city center, before diversion."
"Honeybees communicate flower locations through a precise figure-eight waggle dance."
"The library's third floor housed manuscripts that nobody had touched in centuries."
"Coastal winds carry salt particles that gradually corrode metal structures inland."
"Astronomers detected a faint signal from the constellation that defied easy explanation."
"Ancient pottery fragments often reveal trade routes that historians had not suspected."
"The clock tower's mechanism had four hundred separate gears, each individually crafted."
)

generate_haystack() {
    local target_tokens="$1"
    local needle="$2"
    local position_frac="$3"   # 0.0 to 1.0

    local target_lines=$(( target_tokens / 12 ))
    local needle_line_idx=$(awk "BEGIN { printf \"%d\", $target_lines * $position_frac }")

    local i=0
    while [ "$i" -lt "$target_lines" ]; do
        if [ "$i" -eq "$needle_line_idx" ]; then
            echo "IMPORTANT FACT: The secret code is $needle. Remember this exactly."
        else
            local line_idx=$(( i % ${#FILLER_LINES[@]} ))
            echo "${FILLER_LINES[$line_idx]}"
        fi
        i=$(( i + 1 ))
    done
    echo
    echo "QUESTION: What is the secret code mentioned earlier? Answer with just the 5-character code."
}

run_one_trial() {
    local target="$1"
    local pos_frac="$2"
    local trial_idx="$3"

    # Generate a 5-char alphanumeric needle (deterministic from trial_idx + pos_frac for reproducibility).
    local seed_val="${trial_idx}_${pos_frac}_${target}"
    local needle=$(echo "$seed_val" | shasum -a 256 | head -c 5 | tr 'a-f' 'A-E' | tr '0-9' 'A-J')
    needle=$(echo "$needle" | tr 'a-z' 'A-Z')

    local outdir="$OUT_ROOT/${target}tok_pos${pos_frac}"
    mkdir -p "$outdir"
    local prompt_file="$outdir/prompt_${trial_idx}.txt"
    local out_file="$outdir/out_${trial_idx}.log"

    generate_haystack "$target" "$needle" "$pos_frac" > "$prompt_file"

    local actual_chars=$(wc -c < "$prompt_file" | tr -d ' ')

    timeout 1800 "$HF2Q" generate \
        --model "$MODEL" \
        --prompt-file "$prompt_file" \
        --max-tokens 20 \
        > "$out_file" 2>&1 || { echo "  trial $trial_idx FAILED (exit $?)"; return 1; }

    # Output + needle check.
    # hf2q emits some bracketed banner lines (e.g. [HF2Q_TQ_CODEBOOK_BITS])
    # to stderr that get interleaved with stdout via 2>&1. Strip those
    # before checking, and concatenate all whitespace so cross-line
    # split-token output still matches.
    local cleaned=$(grep -v '^\[HF2Q_' "$out_file" \
        | grep -v '^\[iter-' \
        | sed -e 's/\[HF2Q_TQ_CODEBOOK_BITS\][^[:print:]]*[^[]*$//' \
        | tr -d '\n ')
    local pass="FAIL"
    if echo "$cleaned" | grep -qF "$needle"; then
        pass="PASS"
    fi

    # Stats from any prefill line
    local prefill_tok=$(echo "$response" | grep -oE "prefill: [0-9]+ tok" | grep -oE "[0-9]+" | head -1 || echo "0")

    echo "  trial $trial_idx (needle=$needle, pos=$pos_frac, prefill=${prefill_tok} tok): $pass"

    cat > "$outdir/trial_${trial_idx}.json" <<EOF
{
  "target_tokens": $target,
  "actual_prefill_tokens": $prefill_tok,
  "position_fraction": $pos_frac,
  "needle": "$needle",
  "trial": $trial_idx,
  "verdict": "$pass",
  "prompt_chars": $actual_chars
}
EOF
}

run_length() {
    local target="$1"
    echo "=== F-4.4 needle-in-haystack at target $target tokens ==="
    local positions=("0.1" "0.5" "0.9")
    for pos in "${positions[@]}"; do
        run_one_trial "$target" "$pos" 0 || true
    done
}

IFS=',' read -ra LENGTHS_ARR <<< "$LENGTHS"
for l in "${LENGTHS_ARR[@]}"; do
    run_length "$l"
    echo
done

# Aggregate.
echo "=== F-4.4 Aggregate Pass Rate ==="
total=0
passes=0
for l in "${LENGTHS_ARR[@]}"; do
    for pos in "0.1" "0.5" "0.9"; do
        f="$OUT_ROOT/${l}tok_pos${pos}/trial_0.json"
        if [[ -f "$f" ]]; then
            verdict=$(grep verdict "$f" | grep -oE 'PASS|FAIL')
            total=$(( total + 1 ))
            if [[ "$verdict" == "PASS" ]]; then
                passes=$(( passes + 1 ))
            fi
            printf '  L%-5s pos=%-4s %s\n' "$l" "$pos" "$verdict"
        fi
    done
done
if [[ $total -gt 0 ]]; then
    rate=$(awk "BEGIN { printf \"%.1f\", $passes * 100.0 / $total }")
    echo "Overall: $passes/$total ($rate%)"
fi
