#!/usr/bin/env bash
# ADR-007 Path C F-4.4 / ADR-027 Phase B iter-38: Needle-in-haystack
# retrieval at long context.
#
# Plants a unique fact ("the secret code is XXXXX") at a random position
# in a long-context prompt and asks the model to retrieve it. Reports
# pass/fail per (length, position) cell.
#
# Usage: scripts/bench-needle-haystack.sh [lengths] [trials_per_length]
#   lengths: comma-separated target context-token counts. Default: 4096,8192,16384
#   trials_per_length: int, default 3 (one per fractional position bucket)
#
# Env (ADR-027 iter-38 generalization):
#   MODEL=...      path to GGUF (default: gemma-4-26B-A4B); set to a
#                  qwen35/qwen36 GGUF for ADR-027 needle-haystack runs.
#   OUT_ROOT=...   output root dir (default: /tmp/f04-needle)
#   HF2Q=...       hf2q binary (default: /opt/hf2q/target/release/hf2q)
#   HF2Q_TQ_KV=1   activate TQ-only KV mode (post-iter-34 production
#                  configuration; the iter-23 chain's 3.94× memory savings
#                  is LIVE under this flag for qwen35/qwen36 models).
#
# Position fractions: 0.1 (early), 0.5 (middle), 0.9 (late) — covers the
# typical needle-in-haystack failure modes (early vs late attention).
#
# A "PASS" requires the model to produce the exact 5-char secret code
# in its response. Anything else is "FAIL" (and we capture the response
# for debugging).

set -euo pipefail

MODEL="${MODEL:-/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf}"
HF2Q="${HF2Q:-/opt/hf2q/target/release/hf2q}"
OUT_ROOT="${OUT_ROOT:-/tmp/f04-needle}"
LENGTHS="${1:-4096,8192,16384}"
TRIALS="${2:-3}"
# iter-38: max-tokens budget. Default 20 preserves gemma harness shape
# (gemma typically answers directly with 5-char code + a few words).
# Qwen3-thinking variants emit `<think>...` reasoning before the answer,
# so MAX_TOKENS=256+ is required for the answer to appear in the
# truncated output. Operator overrides via env when targeting qwen35/qwen36.
MAX_TOKENS="${MAX_TOKENS:-20}"
# iter-40: optional --chat-template-file override. qwen36 APEX-Q5_K_M
# (and likely other long-context-fine-tuned thinking models) emits
# `<|im_end|>` as the first assistant token at ≥~4100 prefill tokens
# under the GGUF-embedded chat template — see memory file
# `project_iter40_qwen36_chat_template_long_context_eos_2026_05_09.md`.
# Set CHAT_TEMPLATE_FILE to a null/passthrough template to bypass chat
# templating for long-context needle-haystack runs. Empty string (default)
# uses the GGUF embedded template.
#
# **Recommended canonical value** (iter-52 / iter-53 consolidation):
#   CHAT_TEMPLATE_FILE=/opt/hf2q/scripts/qwen35_raw_passthrough.jinja
# That file ships with the repo (1-line `{{ messages[0].content }}`,
# ADR-005 Phase 1) and is also used by `scripts/qwen35_tokenizer_parity.sh`
# + `scripts/adr027-long-context-sweep.sh`. Reuse rather than rolling
# your own ad-hoc null template — single source of truth.
#
# Caveat: under null template the model treats the prompt as a text-
# completion task rather than instruction-following, so middle-position
# needles may PASS by chance while early/late positions FAIL with the
# model echoing more haystack filler. See iter-51 in
# `project_iter40_qwen36_chat_template_long_context_eos_2026_05_09.md`
# for the falsified MAX_TOKENS-budget hypothesis.
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-}"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: MODEL not found: $MODEL"
    echo "Hint: set MODEL=/path/to/model.gguf"
    exit 1
fi
if [ ! -x "$HF2Q" ]; then
    echo "ERROR: HF2Q binary not found or not executable: $HF2Q"
    echo "Hint: cargo build --release --bin hf2q"
    exit 1
fi

echo "=== bench-needle-haystack ==="
echo "  MODEL    = $MODEL"
echo "  HF2Q     = $HF2Q"
echo "  OUT_ROOT = $OUT_ROOT"
echo "  LENGTHS  = $LENGTHS"
echo "  TRIALS/length = $TRIALS"
echo "  HF2Q_TQ_KV   = ${HF2Q_TQ_KV:-unset (F32 KV cache)}"
echo "  HF2Q_KV_PERSIST = ${HF2Q_KV_PERSIST:-unset}"
echo "  CHAT_TEMPLATE_FILE = ${CHAT_TEMPLATE_FILE:-unset (use GGUF embedded)}"
echo

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

    # iter-40: optional --chat-template-file pass-through.
    local chat_template_arg=()
    if [ -n "$CHAT_TEMPLATE_FILE" ]; then
        chat_template_arg=(--chat-template-file "$CHAT_TEMPLATE_FILE")
    fi
    timeout 1800 "$HF2Q" generate \
        --model "$MODEL" \
        --prompt-file "$prompt_file" \
        --max-tokens "$MAX_TOKENS" \
        "${chat_template_arg[@]}" \
        > "$out_file" 2>&1 || { echo "  trial $trial_idx FAILED (exit $?)"; return 1; }

    # Output + needle check.
    # hf2q emits a [HF2Q_TQ_CODEBOOK_BITS] banner via stderr that gets
    # interleaved with stdout under `2>&1`. The banner can split a
    # token mid-output (e.g. model emits "E", banner injects, model
    # continues "AFDE"). Strip the banner anywhere on a line — not
    # just at line start — and concatenate whitespace so a needle
    # split across the banner injection still matches.
    local cleaned=$(python3 -c "
import re, sys
log = sys.stdin.read()
log = re.sub(r'\[HF2Q_TQ_CODEBOOK_BITS\][^\n]*', '', log)
log = re.sub(r'\[iter-21 Track B\][^\n]*', '', log)
idx = log.find('--- mlx-native:')
log = log[:idx] if idx >= 0 else log
pre = log.find('tok/s)\n')
log = log[pre + len('tok/s)\n'):] if pre >= 0 else log
print(log.replace('\n','').replace(' ',''))
" < "$out_file")
    local pass="FAIL"
    if echo "$cleaned" | grep -qF "$needle"; then
        pass="PASS"
    fi

    # Stats from any prefill line.
    # iter-38 fix: read from the actual output file ($out_file), not
    # the never-set $response variable (pre-existing bug — the script's
    # `set -u` would have caught it but the local declaration masks it).
    local prefill_tok=$(grep -oE "prefill: [0-9]+ tok" "$out_file" 2>/dev/null | grep -oE "[0-9]+" | head -1 || echo "0")
    [ -z "$prefill_tok" ] && prefill_tok=0

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
