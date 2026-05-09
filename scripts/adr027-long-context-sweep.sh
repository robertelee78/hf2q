#!/usr/bin/env bash
# ADR-027 Phase B long-context cross-axis sweep harness (iter-42).
#
# Sibling of `adr027-cross-axis-sweep.sh` — validates byte-identity
# between F32 baseline and TQ-on at PRODUCTION-REALISTIC PREFILL
# LENGTHS (4K, 8K by default). The default sweep runs at 31-token
# prefill (decode-dominant) — fast, byte-identical proof for routine
# regression. This sweep extends that proof to long-context chunked
# prefill where the iter-23 sub-iter chain's TQ-cache prefill SDPA
# wiring (iter-31..34) is actually exercised.
#
# # Why this exists (iter-41 finding)
#
# iter-41 empirically validated TQ-vs-F32 byte-identity at:
#   - 31-tok prefill (cross-axis sweep, all 4 cells)
#   - 5747-tok prefill (4K-target with null template)
#   - 11487-tok prefill (8K-target with null template)
# This script packages that validation as a reusable regression net.
#
# # Why null chat template (iter-40 finding)
#
# qwen36 APEX-Q5_K_M emits `<|im_end|>` as the FIRST assistant token
# at ≥~4100 prefill tokens under the GGUF-embedded chat template (this
# is the model's training-data characteristic at long-context chat
# prompts; reproduces under F32 baseline — NOT a TQ regression).
# Bypass with `--chat-template-file <null-template.j2>` where the
# template body is `{{ messages[0].content }}` — model engages the
# prompt and produces meaningful output that exercises the TQ-cache
# prefill SDPA path.
#
# # Usage
#
# bash scripts/adr027-long-context-sweep.sh [LENGTHS] [MAX_TOKENS]
#
# Defaults:
#   LENGTHS     = 4096,8192   (comma-separated needle-haystack target tokens)
#   MAX_TOKENS  = 64
#   MODEL       = (env override) /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf
#   HF2Q_BIN    = (env override) /opt/hf2q/target/release/hf2q
#
# # Per-length cells (only 2 — persist axis covered by the 31-tok sweep)
#
# - E_LEN: F32 baseline at LEN-target (HF2Q_TQ_KV unset)
# - F_LEN: TQ-on at LEN-target (HF2Q_TQ_KV=1)
#
# Persist axis is NOT exercised here — iter-21 4-cell sweep covers it
# at 31-tok and the persist=write path doesn't differ between short
# and long contexts (write-through is invariant in length).
#
# # Exit codes
#
# 0 = byte-identical at every length tested
# 1 = at least one length diverged (F32 vs TQ-on differ)
# 2 = generate command failed at some cell
#
# # Runtime budget
#
# At qwen36 35B-A3B-APEX-Q5_K_M: each cell takes ~3-7s prefill + ~0.6s
# decode = ~5-8s total. 2 lengths × 2 cells = 4 generation calls
# (~30s wall). Default lengths (4K + 8K) fit comfortably in a /loop
# iter window. Add more lengths by passing 4096,8192,16384.

set -euo pipefail

LENGTHS="${1:-4096,8192}"
MAX_TOKENS="${2:-64}"
MODEL="${MODEL:-/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex/APEX-Q5_K_M.gguf}"
HF2Q_BIN="${HF2Q_BIN:-/opt/hf2q/target/release/hf2q}"

if [[ ! -x "$HF2Q_BIN" ]]; then
    echo "ERROR: $HF2Q_BIN not found or not executable."
    echo "Build with: cargo build --bin hf2q --release"
    exit 2
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model not found: $MODEL"
    exit 2
fi

WORK_DIR="$(mktemp -d -t hf2q-adr027-long-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

# Null chat template — bypasses iter-40 chat-template-induced early-EOS.
NULL_TEMPLATE="$WORK_DIR/null-template.j2"
cat > "$NULL_TEMPLATE" <<'EOF'
{{ messages[0].content }}
EOF

# Filler text for haystack-style prompts (mirrors bench-needle-haystack.sh).
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

generate_long_prompt() {
    local target_tokens="$1"
    local out_file="$2"
    local target_lines=$(( target_tokens / 12 ))
    local i=0
    > "$out_file"
    while [ "$i" -lt "$target_lines" ]; do
        local line_idx=$(( i % ${#FILLER_LINES[@]} ))
        echo "${FILLER_LINES[$line_idx]}" >> "$out_file"
        i=$(( i + 1 ))
    done
    echo "" >> "$out_file"
    echo "Continue this passage with three more sentences in the same style." >> "$out_file"
}

echo "=== ADR-027 Phase B long-context byte-identity sweep ==="
echo "Model:        $MODEL"
echo "Lengths:      $LENGTHS"
echo "Max tokens:   $MAX_TOKENS"
echo "Null tmpl:    $NULL_TEMPLATE"
echo ""

# Strip dynamic banner lines so byte-equivalence holds across cells
# regardless of momentary system memory or wall-clock timing. Same
# filter as the 31-tok sweep + the explicit prefill banner that varies
# by (kv_active vs inactive) state.
extract_output() {
    grep -v '^hf2q load\|^prefill:\|^---\|hf2q lcp\] byte_budget=' || true
}

run_cell() {
    local label="$1"
    local env_pre="$2"
    local prompt_file="$3"
    local out
    if ! out=$(eval "$env_pre $HF2Q_BIN generate --model $MODEL --prompt-file $prompt_file --max-tokens $MAX_TOKENS --temperature 0 --chat-template-file $NULL_TEMPLATE" 2>&1); then
        echo "ERROR: cell $label failed to generate" >&2
        echo "$out" | tail -10 >&2
        exit 2
    fi
    local prefill_line
    local decode_line
    prefill_line=$(echo "$out" | grep '^prefill:' | head -1)
    decode_line=$(echo "$out" | grep 'tokens in' | tail -1)
    {
        printf "    [%s]\n" "$label"
        printf "      %s\n" "$prefill_line"
        printf "      %s\n" "$decode_line"
    } >&2
    echo "$out" | extract_output
}

FAILED=0
IFS=',' read -ra LENGTHS_ARR <<< "$LENGTHS"
for tgt in "${LENGTHS_ARR[@]}"; do
    echo "  === target=${tgt}tok ==="
    prompt_file="$WORK_DIR/prompt_${tgt}.txt"
    generate_long_prompt "$tgt" "$prompt_file"

    OUT_E=$(run_cell "E_${tgt}: F32 baseline" "" "$prompt_file")
    OUT_F=$(run_cell "F_${tgt}: TQ-on"        "HF2Q_TQ_KV=1" "$prompt_file")

    # Byte-equivalence check at this length.
    if diff <(echo "$OUT_E") <(echo "$OUT_F") > /dev/null; then
        printf "    target=%-6s  F32 vs TQ-on  BYTE-IDENTICAL ✓\n" "${tgt}tok"
    else
        printf "    target=%-6s  F32 vs TQ-on  DIFF FOUND ✗\n" "${tgt}tok"
        echo "      --- diff ---"
        diff <(echo "$OUT_E") <(echo "$OUT_F") | head -20 | sed 's/^/      /'
        FAILED=1
    fi
    echo ""
done

if [[ "$FAILED" -ne 0 ]]; then
    echo "FAILED: at least one length diverged between F32 and TQ-on." >&2
    echo "ADR-027 iter-41 long-context byte-identity claim has REGRESSED." >&2
    exit 1
fi

echo "PASSED: all lengths byte-identical between F32 and TQ-on."
echo "ADR-027 iter-41 long-context byte-identity validated at: $LENGTHS"
