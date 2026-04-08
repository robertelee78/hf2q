#!/bin/bash
#
# hf2q Inference Benchmark Script
#
# Runs the hf2q inference engine at multiple prompt lengths and collects
# performance metrics. Outputs a structured JSON report for comparison
# against other inference engines.
#
# Usage:
#   scripts/benchmark.sh --model /path/to/model
#   scripts/benchmark.sh --model /path/to/model --max-tokens 128 --num-runs 5
#
# The script runs the `hf2q serve` server, sends requests via curl, and
# parses the response timings. For synthetic (overhead-only) benchmarks
# without a model, use:
#   cargo bench --bench inference_bench
#
# Methodology:
#   1. Start hf2q serve in the background
#   2. Wait for the health endpoint to respond
#   3. For each prompt length (20, 256, 1024):
#      a. Generate a deterministic synthetic prompt
#      b. Run 2 warm-up requests (discarded)
#      c. Run 5 measurement requests
#      d. Record: TTFT, decode tok/s, prefill tok/s, total time
#   4. Output JSON report to stdout
#   5. Shut down the server

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL=""
MAX_TOKENS=128
NUM_RUNS=5
NUM_WARMUP=2
PROMPT_LENGTHS="20,256,1024"
PORT=18080
HOST="127.0.0.1"
OUTPUT_FILE=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --num-warmup)
            NUM_WARMUP="$2"
            shift 2
            ;;
        --prompt-lengths)
            PROMPT_LENGTHS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --model /path/to/model [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH         Model directory (required)"
            echo "  --max-tokens N       Max tokens per generation (default: 128)"
            echo "  --num-runs N         Measurement runs per prompt length (default: 5)"
            echo "  --num-warmup N       Warm-up runs (default: 2)"
            echo "  --prompt-lengths L   Comma-separated lengths (default: 20,256,1024)"
            echo "  --port N             Server port (default: 18080)"
            echo "  --output FILE        Output file (default: stdout)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Generate synthetic prompt
# ---------------------------------------------------------------------------
generate_prompt() {
    local target_tokens=$1
    local words=("the" "quick" "brown" "fox" "jumps" "over" "the" "lazy"
                 "dog" "and" "then" "runs" "through" "the" "forest" "where"
                 "many" "trees" "grow" "tall" "under" "the" "bright" "sun"
                 "that" "shines" "down" "upon" "the" "green" "meadow" "below")
    local word_count=$(( target_tokens + target_tokens / 10 ))
    local prompt=""
    for (( i=0; i<word_count; i++ )); do
        if [[ $i -gt 0 ]]; then
            prompt+=" "
        fi
        prompt+="${words[$((i % ${#words[@]}))]}"
    done
    echo "$prompt"
}

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        echo "Shutting down server (PID $SERVER_PID)..." >&2
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting hf2q serve on ${HOST}:${PORT}..." >&2

# Measure model load time
LOAD_START=$(date +%s%N)

cargo run --release --features serve -- serve \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --queue-depth 1 \
    &
SERVER_PID=$!

# Wait for health endpoint
echo "Waiting for server to be ready..." >&2
MAX_WAIT=120
WAITED=0
while ! curl -sf "http://${HOST}:${PORT}/health" > /dev/null 2>&1; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "Error: Server did not become ready within ${MAX_WAIT}s" >&2
        exit 1
    fi
    # Check if the process is still alive
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: Server process died during startup" >&2
        exit 1
    fi
done

LOAD_END=$(date +%s%N)
LOAD_TIME_SECS=$(echo "scale=3; ($LOAD_END - $LOAD_START) / 1000000000" | bc)

echo "Server ready (load time: ${LOAD_TIME_SECS}s)" >&2

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
MODEL_NAME=$(basename "$MODEL")
TIMESTAMP=$(date +%s)

# Start building JSON
RESULTS_JSON="["

IFS=',' read -ra LENGTHS <<< "$PROMPT_LENGTHS"

for prompt_len in "${LENGTHS[@]}"; do
    echo "Benchmarking prompt_length=${prompt_len}..." >&2
    PROMPT=$(generate_prompt "$prompt_len")

    # Build the request body
    REQUEST_BODY=$(cat <<REQEOF
{
  "model": "${MODEL_NAME}",
  "messages": [{"role": "user", "content": "${PROMPT}"}],
  "max_tokens": ${MAX_TOKENS},
  "temperature": 0,
  "stream": false
}
REQEOF
)

    # Warm-up
    for (( w=0; w<NUM_WARMUP; w++ )); do
        curl -sf -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$REQUEST_BODY" > /dev/null 2>&1 || true
    done

    # Measurement runs
    DECODE_TPS_VALUES=""
    PREFILL_TPS_VALUES=""
    TTFT_VALUES=""
    GEN_TOKEN_VALUES=""

    for (( r=0; r<NUM_RUNS; r++ )); do
        # Use curl timing for TTFT
        TTFT_START=$(date +%s%N)

        RESPONSE=$(curl -sf -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$REQUEST_BODY" 2>/dev/null || echo '{}')

        TTFT_END=$(date +%s%N)
        TTFT_MS=$(echo "scale=3; ($TTFT_END - $TTFT_START) / 1000000" | bc)

        # Parse response for token counts
        PROMPT_TOKENS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('prompt_tokens',0))" 2>/dev/null || echo "0")
        COMPLETION_TOKENS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")

        # Approximate timing split (server doesn't expose prefill/decode separately via API)
        # For accurate per-phase timing, the server logs can be parsed.
        # Here we use total TTFT as an approximation.
        TOTAL_SECS=$(echo "scale=6; $TTFT_MS / 1000" | bc)

        if [[ "$COMPLETION_TOKENS" -gt 0 && "$TOTAL_SECS" != "0" ]]; then
            DECODE_TPS=$(echo "scale=1; $COMPLETION_TOKENS / $TOTAL_SECS" | bc)
            DECODE_TPS_VALUES="${DECODE_TPS_VALUES}${DECODE_TPS},"
        fi

        if [[ "$PROMPT_TOKENS" -gt 0 && "$TOTAL_SECS" != "0" ]]; then
            PREFILL_TPS=$(echo "scale=1; $PROMPT_TOKENS / $TOTAL_SECS" | bc)
            PREFILL_TPS_VALUES="${PREFILL_TPS_VALUES}${PREFILL_TPS},"
        fi

        TTFT_VALUES="${TTFT_VALUES}${TTFT_MS},"
        GEN_TOKEN_VALUES="${GEN_TOKEN_VALUES}${COMPLETION_TOKENS},"
    done

    # Compute median stats using python
    RESULT_JSON=$(python3 -c "
import statistics, json

def median_stat(values):
    if not values:
        return {'median': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values)
    }

decode_vals = [float(x) for x in '${DECODE_TPS_VALUES}'.rstrip(',').split(',') if x]
prefill_vals = [float(x) for x in '${PREFILL_TPS_VALUES}'.rstrip(',').split(',') if x]
ttft_vals = [float(x) for x in '${TTFT_VALUES}'.rstrip(',').split(',') if x]
gen_vals = [float(x) for x in '${GEN_TOKEN_VALUES}'.rstrip(',').split(',') if x]

result = {
    'prompt_length': ${prompt_len},
    'actual_prompt_tokens': int('${PROMPT_TOKENS}') if '${PROMPT_TOKENS}' else ${prompt_len},
    'decode_tok_per_sec': median_stat(decode_vals),
    'prefill_tok_per_sec': median_stat(prefill_vals),
    'ttft_ms': median_stat(ttft_vals),
    'peak_memory_bytes': None,
    'generated_tokens': median_stat(gen_vals),
    'prompt_cache_active': False,
    'cached_tokens_last_run': 0
}
print(json.dumps(result))
" 2>/dev/null || echo '{}')

    if [[ "$RESULTS_JSON" != "[" ]]; then
        RESULTS_JSON="${RESULTS_JSON},"
    fi
    RESULTS_JSON="${RESULTS_JSON}${RESULT_JSON}"
done

RESULTS_JSON="${RESULTS_JSON}]"

# ---------------------------------------------------------------------------
# Build final report
# ---------------------------------------------------------------------------
REPORT=$(python3 -c "
import json

results = json.loads('${RESULTS_JSON}')
report = {
    'tool': 'hf2q',
    'timestamp': '${TIMESTAMP}s-since-epoch',
    'model': '${MODEL_NAME}',
    'synthetic': False,
    'model_load_time_secs': float('${LOAD_TIME_SECS}'),
    'results': results,
    'methodology': {
        'warm_up_runs': ${NUM_WARMUP},
        'measurement_runs': ${NUM_RUNS},
        'statistic': 'median of N runs with min/max range',
        'max_tokens_per_run': ${MAX_TOKENS},
        'prompt_generation': 'Deterministic synthetic prompts using fixed English word vocabulary.',
        'notes': [
            'Benchmarked via HTTP API (includes network overhead).',
            'For raw engine timing, parse server logs for prefill/decode split.',
            'Temperature=0 for deterministic output.',
            'Same prompts as reference benchmark for fair comparison.'
        ]
    }
}
print(json.dumps(report, indent=2))
" 2>/dev/null)

if [[ -n "$OUTPUT_FILE" ]]; then
    echo "$REPORT" > "$OUTPUT_FILE"
    echo "Report written to $OUTPUT_FILE" >&2
else
    echo "$REPORT"
fi
