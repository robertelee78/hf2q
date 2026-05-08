#!/usr/bin/env bash
# ADR-020 §8.3 AC#7 — boundary delta-NLL measurement at the serve path.
#
# AC#7 status before this script:
#   - DONE (per-Linear) at training time via iter-12f-1 + iter-12f-2 +
#     threshold gate (`hf2q dwq-train --bench` reports
#     mean_delta_kl_nats; threshold +0.05 nats).
#   - HARNESS READY (this script) at the serve boundary.  Closure
#     event: operator's run + commit of SUMMARY.txt artifact comparing
#     mean per-token NLL between two serve configs (vanilla GGUF vs
#     GGUF + --dwq-overlay).
#
# What this measures:
#   For each prompt in the eval corpus, the script:
#     1. Sends a `/v1/chat/completions` POST with `logprobs: true`,
#        `max_tokens: 1`, `temperature: 0.0` to the BASELINE serve
#        (no overlay).
#     2. Records the top-1 log-probability of the next token.
#     3. Same against the OVERLAY serve (--dwq-overlay).
#     4. Aggregates: `mean(NLL_overlay) - mean(NLL_baseline)`.
#
# AC#7 boundary acceptance gate:
#   `mean(NLL_baseline) - mean(NLL_overlay) > 0.05 nats`
# (positive ⇒ overlay's predictions are MORE confident on average,
# matching the per-Linear delta_kl_nats positive signal.)
#
# Note this is top-1-NLL, not full-corpus perplexity.  Full PPL would
# need teacher-forcing prefix eval (not in the OpenAI-compat surface).
# Top-1-NLL is sufficient for differential AC#7 closure provided the
# eval corpus is large enough to swamp per-prompt variance.
#
# Usage:
#   scripts/adr020_ac7_boundary_validate.sh \
#     <gguf-path> \
#     <dwq-overlay.safetensors> \
#     [eval-corpus.jsonl] \
#     [N=20]
#
# Eval corpus format (one prompt per line, JSONL):
#   {"prompt": "The capital of France is "}
# Defaults to /tmp/adr020_ac7_eval.jsonl built from a small inlined
# wikitext-style sample.
#
# Per-port pair: BASELINE on $BASELINE_PORT (8830), OVERLAY on
# $OVERLAY_PORT (8831).  Both serves are spawned, eval runs, both
# serves are killed on exit (TRAP).
#
# Exit codes:
#   0 — both serves succeeded; SUMMARY.txt written.  Verdict in last line.
#   1 — server start / eval failure (one-of).  Diagnostic in stderr.
#   2 — script-internal error (missing args / GGUF / overlay).

set -u
set -o pipefail

GGUF="${1:-}"
OVERLAY="${2:-}"
CORPUS="${3:-/tmp/adr020_ac7_eval.jsonl}"
N="${4:-20}"

if [[ -z "$GGUF" || -z "$OVERLAY" ]]; then
    sed -n '2,40p' "$0"
    exit 2
fi
[[ -f "$GGUF" ]] || { echo "missing GGUF: $GGUF" >&2; exit 2; }
[[ -f "$OVERLAY" ]] || { echo "missing overlay safetensors: $OVERLAY" >&2; exit 2; }

# ─────────────── eval corpus ───────────────
if [[ ! -f "$CORPUS" ]]; then
    echo "[ac7] writing default eval corpus to $CORPUS"
    cat > "$CORPUS" << 'EOF'
{"prompt": "The capital of France is "}
{"prompt": "The largest planet in our solar system is "}
{"prompt": "The author of 'Pride and Prejudice' is "}
{"prompt": "The chemical symbol for gold is "}
{"prompt": "The speed of light in a vacuum is approximately "}
{"prompt": "Mount Everest is located on the border between "}
{"prompt": "The Pacific Ocean is the largest of the world's "}
{"prompt": "The square root of 144 is "}
{"prompt": "Photosynthesis converts carbon dioxide and water into "}
{"prompt": "The Great Wall of China was built primarily during the "}
{"prompt": "The currency of Japan is the "}
{"prompt": "The Mona Lisa was painted by "}
{"prompt": "The longest river in the world is the "}
{"prompt": "The first president of the United States was "}
{"prompt": "DNA stands for "}
{"prompt": "The boiling point of water at sea level is "}
{"prompt": "The Eiffel Tower is located in "}
{"prompt": "Shakespeare wrote the play 'Hamlet' in approximately "}
{"prompt": "The element with atomic number 1 is "}
{"prompt": "The Pythagorean theorem relates the sides of a "}
EOF
fi

CORPUS_COUNT=$(wc -l < "$CORPUS" | tr -d ' ')
[[ "$CORPUS_COUNT" -ge "$N" ]] || { echo "corpus has $CORPUS_COUNT lines but N=$N requested" >&2; exit 2; }

OUT_DIR="/tmp/adr020_ac7"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/SUMMARY.txt"

# ─────────────── binary ───────────────
HF2Q="$(dirname "$0")/../target/release/hf2q"
[[ -x "$HF2Q" ]] || { echo "hf2q binary missing: $HF2Q" >&2; exit 2; }

BASELINE_PORT=8830
OVERLAY_PORT=8831

# ─────────────── trap cleanup ───────────────
BASELINE_PID=""
OVERLAY_PID=""
cleanup() {
    [[ -n "$BASELINE_PID" ]] && kill "$BASELINE_PID" 2>/dev/null
    [[ -n "$OVERLAY_PID"  ]] && kill "$OVERLAY_PID"  2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

# ─────────────── start serves ───────────────
echo "[ac7] starting baseline serve on :$BASELINE_PORT"
"$HF2Q" serve --model "$GGUF" --port "$BASELINE_PORT" --quiet \
    > "$OUT_DIR/baseline.log" 2>&1 &
BASELINE_PID=$!

echo "[ac7] starting overlay serve on :$OVERLAY_PORT"
"$HF2Q" serve --model "$GGUF" --dwq-overlay "$OVERLAY" --port "$OVERLAY_PORT" --quiet \
    > "$OUT_DIR/overlay.log" 2>&1 &
OVERLAY_PID=$!

# Wait for both servers to be listening (poll /v1/models).
wait_listening() {
    local port="$1"
    local label="$2"
    for _ in $(seq 1 60); do
        if curl -sm2 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
            echo "[ac7] $label serve up on :$port"
            return 0
        fi
        sleep 1
    done
    echo "[ac7] $label serve failed to start on :$port" >&2
    tail -20 "$OUT_DIR/$label.log" >&2
    return 1
}
wait_listening "$BASELINE_PORT" "baseline" || exit 1
wait_listening "$OVERLAY_PORT" "overlay" || exit 1

# Resolve loaded model id from /v1/models (single model expected per serve).
get_model_id() {
    curl -sm5 "http://127.0.0.1:$1/v1/models" | python3 -c \
        'import sys, json; d=json.load(sys.stdin); print(d["data"][0]["id"])'
}
BASELINE_MODEL=$(get_model_id "$BASELINE_PORT")
OVERLAY_MODEL=$(get_model_id "$OVERLAY_PORT")
echo "[ac7] baseline model=$BASELINE_MODEL"
echo "[ac7] overlay  model=$OVERLAY_MODEL"

# ─────────────── eval ───────────────
# For each prompt, request 1 token with logprobs and capture the top
# logprob.  Negative → larger NLL.  Aggregate across N prompts.
eval_one() {
    local port="$1"
    local model="$2"
    local prompt="$3"
    curl -sm60 "http://127.0.0.1:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c "import json,sys; print(json.dumps({'model':sys.argv[1],'messages':[{'role':'user','content':sys.argv[2]}],'max_tokens':1,'temperature':0.0,'logprobs':True}))" "$model" "$prompt")" \
    | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    lp = d['choices'][0].get('logprobs')
    if lp and 'content' in lp and lp['content']:
        print(lp['content'][0]['logprob'])
    else:
        print('NA')
except Exception as e:
    print('NA')
"
}

declare -a baseline_lls
declare -a overlay_lls
declare -a delta_lls

i=0
while IFS= read -r line && [[ "$i" -lt "$N" ]]; do
    prompt=$(printf '%s' "$line" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["prompt"])')
    b=$(eval_one "$BASELINE_PORT" "$BASELINE_MODEL" "$prompt")
    o=$(eval_one "$OVERLAY_PORT" "$OVERLAY_MODEL" "$prompt")
    if [[ "$b" == "NA" || "$o" == "NA" ]]; then
        echo "[ac7] eval[$i] NA for prompt: $prompt" >&2
        i=$((i+1))
        continue
    fi
    baseline_lls+=("$b")
    overlay_lls+=("$o")
    delta=$(python3 -c "print($b - $o)")
    delta_lls+=("$delta")
    printf "[ac7] eval[%2d] baseline=%9.4f overlay=%9.4f Δ=%+8.5f  prompt=%s\n" \
        "$i" "$b" "$o" "$delta" "$prompt"
    i=$((i+1))
done < "$CORPUS"

# Aggregate.
if [[ "${#delta_lls[@]}" -eq 0 ]]; then
    echo "[ac7] no successful eval samples — aborting"
    exit 1
fi

mean_baseline_ll=$(python3 -c "
ls = [${baseline_lls[*]/%/,}]
ls = [x for x in ls if x is not None]
print(sum(ls)/len(ls))" 2>/dev/null)
mean_overlay_ll=$(python3 -c "
ls = [${overlay_lls[*]/%/,}]
print(sum(ls)/len(ls))" 2>/dev/null)
mean_delta_nats=$(python3 -c "
ls = [${delta_lls[*]/%/,}]
print(sum(ls)/len(ls))" 2>/dev/null)

# AC#7 boundary acceptance: mean(NLL_baseline) - mean(NLL_overlay) > 0.05
# log-probabilities are negative-NLL-up-to-sign, so:
#   NLL_baseline = -mean_baseline_ll
#   NLL_overlay  = -mean_overlay_ll
#   delta_nats   = NLL_baseline - NLL_overlay = -mean_baseline_ll + mean_overlay_ll
# But our `delta` per sample = baseline_ll - overlay_ll, so mean_delta_nats
# is exactly (mean_baseline_ll - mean_overlay_ll), which is positive when
# baseline is MORE confident.  AC#7 wants OVERLAY more confident, so the
# right direction is `mean_overlay_ll > mean_baseline_ll` ⇒ `mean_delta_nats < -0.05`.
# Reframe: report `boundary_delta_nats = mean_overlay_ll - mean_baseline_ll`
# and accept when > +0.05.
boundary_delta_nats=$(python3 -c "print(-($mean_delta_nats))")

verdict="FAIL"
if python3 -c "import sys; sys.exit(0 if $boundary_delta_nats > 0.05 else 1)"; then
    verdict="PASS"
fi

{
    echo "ADR-020 §8.3 AC#7 — boundary delta-NLL measurement"
    echo "===================================================="
    printf "samples_used:        %d / %d (corpus=%s)\n" "${#delta_lls[@]}" "$N" "$CORPUS"
    printf "baseline_mean_ll:    %+10.6f\n" "$mean_baseline_ll"
    printf "overlay_mean_ll:     %+10.6f\n" "$mean_overlay_ll"
    printf "boundary_delta_nats: %+10.6f  (overlay - baseline log-prob mean)\n" "$boundary_delta_nats"
    printf "AC#7 threshold:      +0.05 nats (overlay more confident)\n"
    printf "AC#7 verdict:        %s\n" "$verdict"
    printf "gguf:                %s\n" "$GGUF"
    printf "overlay:             %s\n" "$OVERLAY"
} | tee "$SUMMARY"

if [[ "$verdict" == "PASS" ]]; then
    exit 0
else
    exit 1
fi
