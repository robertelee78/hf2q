#!/usr/bin/env bash
set -euo pipefail

# Matched production-route A/B for the Qwen stable-boundary compound prefill.
# Each engine gets a fresh process, the same GGUF, the same greedy requests,
# and the same four-slot scheduler. The candidate must emit the production
# compound-route receipt; output equality and nonzero timings are fail-closed.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)

MODEL=${MODEL:-/opt/hf2q/models/qwen3.8/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf}
QWEN38_FORMAT=${QWEN38_FORMAT:-Q4_K_M}
BASELINE_SOURCE_ROOT=${BASELINE_SOURCE_ROOT:-/opt/hf2q}
CANDIDATE_SOURCE_ROOT=${CANDIDATE_SOURCE_ROOT:-$root_dir}
BASELINE_BIN=${BASELINE_BIN:-/opt/hf2q/target/release/hf2q}
CANDIDATE_BIN=${CANDIDATE_BIN:-$root_dir/target/release/hf2q}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-52841}
readonly TRIALS=5
readonly PROMPT_LINES=80
readonly MAX_TOKENS=8
readonly MAX_SLOTS=4
readonly KV_CACHE_BUDGET_BYTES=51539607552
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-240}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-300}
readonly MIN_SINGLE_TTFT_RATIO=1.01
readonly MIN_SINGLE_WALL_RATIO=1.0
readonly MIN_FOUR_SLOT_WAVE_RATIO=1.0
OUT_DIR=${OUT_DIR:-$(mktemp -d /var/tmp/hf2q-qwen-compound-ab.XXXXXX)}
GATE_CARGO_HOME=${GATE_CARGO_HOME:-${TMPDIR:-/var/tmp}/hf2q-qwen-exact-cargo-home}

for command in awk cargo curl find git jq lsof mv perl ps rg rm sed shasum sort stat; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ "$BASELINE_SOURCE_ROOT" == /* && "$CANDIDATE_SOURCE_ROOT" == /* \
    && "$GATE_CARGO_HOME" == /* && "$OUT_DIR" == /* ]] || {
    echo "source roots, GATE_CARGO_HOME, and OUT_DIR must be absolute" >&2
    exit 2
}
BASELINE_SOURCE_ROOT=$(cd "$BASELINE_SOURCE_ROOT" && pwd -P)
CANDIDATE_SOURCE_ROOT=$(cd "$CANDIDATE_SOURCE_ROOT" && pwd -P)
mkdir -p "$GATE_CARGO_HOME"
GATE_CARGO_HOME=$(cd "$GATE_CARGO_HOME" && pwd -P)
case "$GATE_CARGO_HOME/" in
    "$BASELINE_SOURCE_ROOT"/*|"$CANDIDATE_SOURCE_ROOT"/*|"$OUT_DIR"/*)
        echo "GATE_CARGO_HOME must be outside source and evidence trees" >&2
        exit 2
        ;;
esac
case "$OUT_DIR/" in
    "$BASELINE_SOURCE_ROOT"/*|"$CANDIDATE_SOURCE_ROOT"/*)
        echo "OUT_DIR must be outside both source trees" >&2
        exit 2
        ;;
esac
[[ "$MODEL" == /* && -f "$MODEL" && -r "$MODEL" && ! -L "$MODEL" ]] || {
    echo "model artifact must be an absolute readable regular non-symlink: $MODEL" >&2
    exit 2
}
for source_root in "$BASELINE_SOURCE_ROOT" "$CANDIDATE_SOURCE_ROOT"; do
    [[ -d "$source_root/.git" || -f "$source_root/.git" ]] || {
        echo "source root is not a Git worktree: $source_root" >&2
        exit 2
    }
    [[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]] || {
        echo "matched compound A/B requires a clean source tree: $source_root" >&2
        exit 2
    }
done
baseline_commit=$(git -C "$BASELINE_SOURCE_ROOT" rev-parse HEAD)
candidate_commit=$(git -C "$CANDIDATE_SOURCE_ROOT" rev-parse HEAD)
[[ "$baseline_commit" =~ ^[0-9a-f]{40}$ && "$candidate_commit" =~ ^[0-9a-f]{40}$ \
    && "$baseline_commit" != "$candidate_commit" ]] || {
    echo "baseline and candidate require distinct exact source commits" >&2
    exit 2
}
[[ "$(git -C "$BASELINE_SOURCE_ROOT" symbolic-ref --quiet --short HEAD || true)" == main ]] || {
    echo "baseline source must be the main worktree" >&2
    exit 2
}
git -C "$CANDIDATE_SOURCE_ROOT" merge-base --is-ancestor \
    "$baseline_commit" "$candidate_commit" || {
    echo "candidate must descend from the exact main baseline" >&2
    exit 2
}

# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$script_dir/qwen38_artifact_contract.sh"
# shellcheck source=scripts/qwen38_physical_multislot_contract.sh
source "$script_dir/qwen38_physical_multislot_contract.sh"
# shellcheck source=scripts/qwen35_compound_wave_contract.sh
source "$script_dir/qwen35_compound_wave_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"
qwen35_compound_validate_policy "$MIN_SINGLE_TTFT_RATIO" \
    "$MIN_SINGLE_WALL_RATIO" "$MIN_FOUR_SLOT_WAVE_RATIO"
qwen35_compound_require_fresh_out_dir "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd -P)
qwen38_reject_cargo_configuration "$BASELINE_SOURCE_ROOT" "$GATE_CARGO_HOME"
qwen38_reject_cargo_configuration "$CANDIDATE_SOURCE_ROOT" "$GATE_CARGO_HOME"
IFS=$'\t' read -r _format _relative _expected_bytes _expected_sha256 expected_file_type \
    <<<"$(qwen38_artifact_record "$QWEN38_FORMAT")"
actual_bytes=$(stat -f '%z' "$MODEL" 2>/dev/null || stat -c '%s' "$MODEL")
actual_sha256=$(shasum -a 256 "$MODEL" | awk '{print $1}')
qwen38_validate_artifact_identity \
    "$QWEN38_FORMAT" "$actual_sha256" "$actual_bytes" "$expected_file_type"
model_snapshot=$(hf2q_release_model_snapshot "$MODEL")
[[ -n "$model_snapshot" ]] || {
    echo "could not capture the exact model file identity" >&2
    exit 2
}

assert_model_unchanged() {
    [[ -f "$MODEL" && ! -L "$MODEL" \
        && "$(hf2q_release_model_snapshot "$MODEL")" == "$model_snapshot" ]] || {
        echo "model artifact changed during matched compound A/B" >&2
        return 1
    }
}

[[ "$BASELINE_BIN" == "$BASELINE_SOURCE_ROOT/target/release/hf2q" \
    && "$CANDIDATE_BIN" == "$CANDIDATE_SOURCE_ROOT/target/release/hf2q" ]] || {
    echo "binaries must be the canonical release outputs of their exact source roots" >&2
    exit 2
}

baseline_dependency=$(qwen38_mlx_native_registry_identity "$BASELINE_SOURCE_ROOT")
candidate_dependency=$(qwen38_mlx_native_registry_identity "$CANDIDATE_SOURCE_ROOT")
CARGO_HOME="$GATE_CARGO_HOME" GIT_COMMIT_SHA="$baseline_commit" \
    cargo build --release --locked --bin hf2q \
    --manifest-path "$BASELINE_SOURCE_ROOT/Cargo.toml"
CARGO_HOME="$GATE_CARGO_HOME" GIT_COMMIT_SHA="$candidate_commit" \
    cargo build --release --locked --bin hf2q \
    --manifest-path "$CANDIDATE_SOURCE_ROOT/Cargo.toml"
for identity in "$baseline_commit:$BASELINE_BIN" "$candidate_commit:$CANDIDATE_BIN"; do
    commit=${identity%%:*}
    binary=${identity#*:}
    grep -aFq "$commit" "$binary" || {
        echo "binary does not embed exact source commit $commit: $binary" >&2
        exit 2
    }
done
baseline_binary_sha=$(shasum -a 256 "$BASELINE_BIN" | awk '{print $1}')
candidate_binary_sha=$(shasum -a 256 "$CANDIDATE_BIN" | awk '{print $1}')
[[ "$baseline_binary_sha" != "$candidate_binary_sha" ]] || {
    echo "baseline and candidate binary SHA-256 identities must differ" >&2
    exit 2
}
for file in "$MODEL" "$BASELINE_BIN" "$CANDIDATE_BIN"; do
    [[ -f "$file" ]] || { echo "required file not found: $file" >&2; exit 2; }
done
for binary in "$BASELINE_BIN" "$CANDIDATE_BIN"; do
    [[ -x "$binary" ]] || { echo "binary is not executable: $binary" >&2; exit 2; }
done
for setting in PORT TRIALS PROMPT_LINES MAX_TOKENS KV_CACHE_BUDGET_BYTES \
    READY_TIMEOUT_SECONDS REQUEST_TIMEOUT_SECONDS; do
    value=${!setting}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "$setting must be a positive integer (got: $value)" >&2
        exit 2
    }
done
for setting in MIN_SINGLE_TTFT_RATIO MIN_SINGLE_WALL_RATIO MIN_FOUR_SLOT_WAVE_RATIO; do
    value=${!setting}
    if ! [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || ! awk -v value="$value" 'BEGIN { exit !(value > 0) }'; then
        echo "$setting must be a positive finite number (got: $value)" >&2
        exit 2
    fi
done
(( PORT <= 65535 )) || { echo "PORT exceeds 65535" >&2; exit 2; }
(( PROMPT_LINES <= 100 )) || {
    echo "PROMPT_LINES must remain <= 100 so the stable boundary fits one 2K transaction" >&2
    exit 2
}
[[ "$BASELINE_BIN" != "$CANDIDATE_BIN" ]] || {
    echo "BASELINE_BIN and CANDIDATE_BIN must be distinct paths" >&2
    exit 2
}
[[ -z "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ]] || {
    echo "$HOST:$PORT is already in use" >&2
    exit 2
}

server_pid=""
receipt_tmp="$OUT_DIR/.receipt.json.tmp.$$"
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=""
    fi
    rm -f "$receipt_tmp"
}
trap cleanup EXIT INT TERM

median_file() {
    local file=$1 count
    count=$(wc -l <"$file" | tr -d ' ')
    sort -n "$file" | awk -v count="$count" '
        NR == int((count + 1) / 2) { left=$1 }
        NR == int((count + 2) / 2) { right=$1 }
        END { if (count % 2) print left; else print (left + right) / 2 }
    '
}

build_request() {
    local engine=$1 trial=$2 client=$3 output=$4 context
    context=$(awk -v lines="$PROMPT_LINES" -v trial="$trial" -v client="$client" 'BEGIN {
        for (i = 1; i <= lines; i++)
            printf "trial-%02d client-%d line-%03d: exact cache state, deterministic routing, and Apple Silicon inference.\n", trial, client, i
    }')
    jq -n --arg model "$engine" --arg context "$context" --arg trial "$trial" \
        --arg client "$client" --argjson max_tokens "$MAX_TOKENS" '{
          model:$model,
          messages:[
            {role:"system",content:("compound-boundary matched gate trial=" + $trial + " client=" + $client)},
            {role:"user",content:($context + "\nReply with exactly READY.")}
          ],
          temperature:0,seed:42,max_tokens:$max_tokens,repetition_penalty:1.0,
          stream:false,hf2q_enable_thinking:false,
          chat_template_kwargs:{enable_thinking:false}
        }' >"$output"
}

post_request() {
    local request=$1 response=$2 wall=$3
    curl --fail-with-body --silent --show-error \
        --connect-timeout 5 --max-time "$REQUEST_TIMEOUT_SECONDS" \
        -H 'Content-Type: application/json' --data-binary "@$request" \
        -o "$response" -w '%{time_total}\n' \
        "http://$HOST:$PORT/v1/chat/completions" >"$wall"
    jq -e '
      (.choices | length) == 1
      and (.choices[0].message.content | type) == "string"
      and (.usage.prompt_tokens | numbers) > 0
      and (.usage.completion_tokens | numbers) > 0
      and (.x_hf2q_timing.time_to_first_token_ms | numbers) > 0
    ' "$response" >/dev/null
}

run_engine() {
    local label=$1
    local binary=$2
    local source_commit=$3
    local dependency=$4
    local require_physical_instrumentation=$5
    local model_id trial client completed_clients compound_receipts
    local engine_dir="$OUT_DIR/$label"
    assert_model_unchanged
    mkdir -p "$engine_dir/requests" "$engine_dir/responses"
    env \
        HF2Q_QWEN_SPECULATION=off \
        HF2Q_TQ_KV=1 \
        HF2Q_ENCODER_SESSION=1 \
        HF2Q_FFN_TERMINAL_K_BATCH=8 \
        "$binary" -v serve \
        --model "$MODEL" --host "$HOST" --port "$PORT" \
        --overflow-policy reject --scheduler inflight-batched \
        --max-slots "$MAX_SLOTS" --kv-cache-budget "$KV_CACHE_BUDGET_BYTES" \
        --default-repetition-penalty 1.0 \
        --default-thinking-token-budget 0 \
        --default-tool-thinking-token-budget 0 \
        >"$engine_dir/server.stdout" 2>"$engine_dir/server.stderr" &
    server_pid=$!

    for ((trial = 1; trial <= READY_TIMEOUT_SECONDS; trial++)); do
        if curl --fail --silent "http://$HOST:$PORT/readyz" >/dev/null 2>&1; then
            break
        fi
        kill -0 "$server_pid" 2>/dev/null || {
            echo "$label server exited before readiness" >&2
            tail -n 80 "$engine_dir/server.stderr" >&2
            exit 1
        }
        sleep 1
    done
    curl --fail --silent "http://$HOST:$PORT/readyz" >/dev/null || {
        echo "$label server did not become ready" >&2
        exit 1
    }
    model_id=$(curl --fail --silent "http://$HOST:$PORT/v1/models" |
        jq -er '[.data[] | select(.loaded == true)] | if length == 1 then .[0].id else error("expected one loaded model") end')
    qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" \
        "$binary" "$MODEL" "$MAX_SLOTS"
    assert_model_unchanged

    : >"$engine_dir/single_ttft_ms"
    : >"$engine_dir/single_wall_seconds"
    : >"$engine_dir/wave_wall_seconds"
    mkdir -p "$engine_dir/physical-wave-metrics"
    for ((trial = 1; trial <= TRIALS; trial++)); do
        qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" \
            "$binary" "$MODEL" "$MAX_SLOTS"
        assert_model_unchanged
        request="$engine_dir/requests/single-$trial.json"
        response="$engine_dir/responses/single-$trial.json"
        wall="$engine_dir/responses/single-$trial.wall"
        build_request "$model_id" "$trial" 0 "$request"
        post_request "$request" "$response" "$wall"
        jq -er '.x_hf2q_timing.time_to_first_token_ms' "$response" >>"$engine_dir/single_ttft_ms"
        tr -d '[:space:]' <"$wall" >>"$engine_dir/single_wall_seconds"
        printf '\n' >>"$engine_dir/single_wall_seconds"

        metrics_before="$engine_dir/physical-wave-metrics/trial-$trial.before"
        metrics_after="$engine_dir/physical-wave-metrics/trial-$trial.after"
        curl --fail --silent --show-error --max-time 10 \
            "http://$HOST:$PORT/metrics" >"$metrics_before"
        qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" \
            "$binary" "$MODEL" "$MAX_SLOTS"
        wave_started=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
        wave_pids=()
        for ((client = 1; client <= MAX_SLOTS; client++)); do
            request="$engine_dir/requests/wave-$trial-$client.json"
            response="$engine_dir/responses/wave-$trial-$client.json"
            wall="$engine_dir/responses/wave-$trial-$client.wall"
            build_request "$model_id" "$trial" "$client" "$request"
            post_request "$request" "$response" "$wall" &
            wave_pids+=("$!")
        done
        for pid in "${wave_pids[@]}"; do
            wait "$pid"
        done
        qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" \
            "$binary" "$MODEL" "$MAX_SLOTS"
        assert_model_unchanged
        wave_finished=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
        awk -v start="$wave_started" -v finish="$wave_finished" 'BEGIN { print finish - start }' \
            >>"$engine_dir/wave_wall_seconds"
        curl --fail --silent --show-error --max-time 10 \
            "http://$HOST:$PORT/metrics" >"$metrics_after"
        completed_clients=0
        for ((client = 1; client <= MAX_SLOTS; client++)); do
            [[ -s "$engine_dir/responses/wave-$trial-$client.json" \
                && -s "$engine_dir/responses/wave-$trial-$client.wall" ]] || {
                echo "$label trial $trial lacks a completed client-$client response" >&2
                exit 1
            }
            completed_clients=$((completed_clients + 1))
        done
        prove_compound_wave "$label" "$trial" "$metrics_before" "$metrics_after" \
            "$engine_dir/physical-wave-metrics/trial-$trial.json" \
            "$require_physical_instrumentation" "$completed_clients"
    done

    kill -0 "$server_pid" 2>/dev/null || {
        echo "$label server exited before receipt creation" >&2
        exit 1
    }
    qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" \
        "$binary" "$MODEL" "$MAX_SLOTS"
    qwen36_reject_fatal_log "$engine_dir/server.stderr"
    assert_model_unchanged

    compound_receipts=$(rg -c 'Qwen stable-boundary compound prefill complete' \
        "$engine_dir/server.stderr" || true)
    compound_receipts=${compound_receipts:-0}
    jq -n \
        --arg label "$label" --arg binary "$binary" \
        --arg binary_sha256 "$(shasum -a 256 "$binary" | awk '{print $1}')" \
        --arg source_commit "$source_commit" --arg dependency "$dependency" \
        --arg model_id "$model_id" \
        --argjson compound_receipts "$compound_receipts" \
        --argjson single_median_ttft_ms "$(median_file "$engine_dir/single_ttft_ms")" \
        --argjson single_median_wall_seconds "$(median_file "$engine_dir/single_wall_seconds")" \
        --argjson four_slot_median_wave_seconds "$(median_file "$engine_dir/wave_wall_seconds")" \
        --argjson single_ttft_samples_ms "$(jq -Rsc 'split("\n") | map(select(length > 0) | tonumber)' "$engine_dir/single_ttft_ms")" \
        --argjson single_wall_samples_seconds "$(jq -Rsc 'split("\n") | map(select(length > 0) | tonumber)' "$engine_dir/single_wall_seconds")" \
        --argjson four_slot_wave_samples_seconds "$(jq -Rsc 'split("\n") | map(select(length > 0) | tonumber)' "$engine_dir/wave_wall_seconds")" \
        --argjson wave_execution_receipts "$(jq -s . "$engine_dir"/physical-wave-metrics/trial-*.json)" \
        '{label:$label,binary:$binary,binary_sha256:$binary_sha256,
          source_commit:$source_commit,dependency_identity:$dependency,model_id:$model_id,
          compound_receipts:$compound_receipts,
          single_median_ttft_ms:$single_median_ttft_ms,
          single_median_wall_seconds:$single_median_wall_seconds,
          four_slot_median_wave_seconds:$four_slot_median_wave_seconds,
          single_ttft_samples_ms:$single_ttft_samples_ms,
          single_wall_samples_seconds:$single_wall_samples_seconds,
          four_slot_wave_samples_seconds:$four_slot_wave_samples_seconds,
          wave_execution_receipts:$wave_execution_receipts}' \
        >"$engine_dir/summary.json"
    cleanup
}

# Fresh-process ABBA balances first-load, global Metal cache, and thermal order.
run_engine baseline-a "$BASELINE_BIN" "$baseline_commit" "$baseline_dependency" 0
assert_model_unchanged
run_engine candidate-a "$CANDIDATE_BIN" "$candidate_commit" "$candidate_dependency" 1
assert_model_unchanged
run_engine candidate-b "$CANDIDATE_BIN" "$candidate_commit" "$candidate_dependency" 1
assert_model_unchanged
run_engine baseline-b "$BASELINE_BIN" "$baseline_commit" "$baseline_dependency" 0
assert_model_unchanged

for summary in \
    "$OUT_DIR/baseline-a/summary.json" "$OUT_DIR/candidate-a/summary.json" \
    "$OUT_DIR/candidate-b/summary.json" "$OUT_DIR/baseline-b/summary.json"; do
    jq -e --argjson trials "$TRIALS" '
      (.wave_execution_receipts | length) == $trials
      and all(.wave_execution_receipts[];
        .client_wave_complete == true and .client_count == 4)
    ' "$summary" >/dev/null || {
        echo "matched A/B did not prove every four-client wave completed: $summary" >&2
        exit 1
    }
done
for summary in "$OUT_DIR/candidate-a/summary.json" "$OUT_DIR/candidate-b/summary.json"; do
    jq -e '
      all(.wave_execution_receipts[];
        .physical_instrumentation == "available"
        and (.physical_width_four_observed | type) == "boolean")
    ' "$summary" >/dev/null || {
        echo "candidate did not provide honest physical observations: $summary" >&2
        exit 1
    }
done

baseline_summary="$OUT_DIR/baseline-summary.json"
candidate_summary="$OUT_DIR/candidate-summary.json"
qwen35_compound_aggregate_arm baseline "$OUT_DIR/baseline-a/summary.json" \
    "$OUT_DIR/baseline-b/summary.json" "$baseline_summary"
qwen35_compound_aggregate_arm candidate "$OUT_DIR/candidate-a/summary.json" \
    "$OUT_DIR/candidate-b/summary.json" "$candidate_summary"

for source_root in "$BASELINE_SOURCE_ROOT" "$CANDIDATE_SOURCE_ROOT"; do
    [[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]] || {
        echo "source tree changed during matched A/B: $source_root" >&2
        exit 1
    }
done
[[ "$(git -C "$BASELINE_SOURCE_ROOT" rev-parse HEAD)" == "$baseline_commit" \
    && "$(git -C "$CANDIDATE_SOURCE_ROOT" rev-parse HEAD)" == "$candidate_commit" ]] || {
    echo "source commit changed during matched A/B" >&2
    exit 1
}

baseline_semantic_sha=$(jq -Ssc 'map({content:.choices[0].message.content,
    prompt_tokens:.usage.prompt_tokens,completion_tokens:.usage.completion_tokens})' \
    "$OUT_DIR"/baseline-*/responses/*.json | shasum -a 256 | awk '{print $1}')
candidate_semantic_sha=$(jq -Ssc 'map({content:.choices[0].message.content,
    prompt_tokens:.usage.prompt_tokens,completion_tokens:.usage.completion_tokens})' \
    "$OUT_DIR"/candidate-*/responses/*.json | shasum -a 256 | awk '{print $1}')
[[ "$baseline_semantic_sha" == "$candidate_semantic_sha" ]] || {
    echo "baseline/candidate semantic receipt mismatch" >&2
    exit 1
}

candidate_receipts=$(jq -er '.compound_receipts' "$candidate_summary")
baseline_receipts=$(jq -er '.compound_receipts' "$baseline_summary")
expected_receipts=$((2 * TRIALS * (MAX_SLOTS + 1)))
(( baseline_receipts == 0 )) || {
    echo "baseline unexpectedly emitted $baseline_receipts compound-route receipts" >&2
    exit 1
}
(( candidate_receipts == expected_receipts )) || {
    echo "candidate compound-route receipts=$candidate_receipts expected=$expected_receipts" >&2
    exit 1
}

single_ttft_ratio=$(jq -nr --slurpfile baseline "$baseline_summary" \
    --slurpfile candidate "$candidate_summary" \
    '$baseline[0].single_median_ttft_ms / $candidate[0].single_median_ttft_ms')
single_wall_ratio=$(jq -nr --slurpfile baseline "$baseline_summary" \
    --slurpfile candidate "$candidate_summary" \
    '$baseline[0].single_median_wall_seconds / $candidate[0].single_median_wall_seconds')
four_slot_ratio=$(jq -nr --slurpfile baseline "$baseline_summary" \
    --slurpfile candidate "$candidate_summary" \
    '$baseline[0].four_slot_median_wave_seconds / $candidate[0].four_slot_median_wave_seconds')
for gate in \
    "$single_ttft_ratio:$MIN_SINGLE_TTFT_RATIO:single TTFT" \
    "$single_wall_ratio:$MIN_SINGLE_WALL_RATIO:single wall" \
    "$four_slot_ratio:$MIN_FOUR_SLOT_WAVE_RATIO:four-slot wave"; do
    actual=${gate%%:*}
    rest=${gate#*:}
    minimum=${rest%%:*}
    label=${rest#*:}
    awk -v actual="$actual" -v minimum="$minimum" 'BEGIN { exit !(actual >= minimum) }' || {
        echo "$label ratio $actual is below required $minimum" >&2
        exit 1
    }
done

jq -n \
    --arg model "$MODEL" \
    --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
    --arg format "$QWEN38_FORMAT" --arg model_sha256 "$actual_sha256" \
    --arg model_snapshot "$model_snapshot" \
    --argjson model_bytes "$actual_bytes" --argjson model_file_type "$expected_file_type" \
    --arg baseline_commit "$baseline_commit" --arg candidate_commit "$candidate_commit" \
    --arg semantic_sha256 "$candidate_semantic_sha" \
    --argjson trials "$TRIALS" --argjson max_slots "$MAX_SLOTS" \
    --argjson prompt_lines "$PROMPT_LINES" --argjson max_tokens "$MAX_TOKENS" \
    --argjson kv_cache_budget_bytes "$KV_CACHE_BUDGET_BYTES" \
    --argjson min_single_ttft_ratio "$MIN_SINGLE_TTFT_RATIO" \
    --argjson min_single_wall_ratio "$MIN_SINGLE_WALL_RATIO" \
    --argjson min_four_slot_wave_ratio "$MIN_FOUR_SLOT_WAVE_RATIO" \
    --slurpfile baseline "$baseline_summary" \
    --slurpfile candidate "$candidate_summary" '
      {schema:3,verdict:"pass",gate:"qwen35-stable-boundary-compound-abba",
       model:$model,model_snapshot:$model_snapshot,
       repository:$repository,revision:$revision,
       format:$format,model_sha256:$model_sha256,
       model_bytes:$model_bytes,model_file_type:$model_file_type,
       baseline_commit:$baseline_commit,candidate_commit:$candidate_commit,
       trials_per_process:$trials,max_slots:$max_slots,
       policy:{
         process_order:["baseline-a","candidate-a","candidate-b","baseline-b"],
         processes_per_arm:2,
         trials_per_process:$trials,prompt_lines:$prompt_lines,
         max_tokens:$max_tokens,max_slots:$max_slots,
         kv_cache_budget_bytes:$kv_cache_budget_bytes,
         speculation:"off",temperature:0,seed:42,repetition_penalty:1.0,
         candidate_physical_instrumentation_required:true,
         physical_decode_width_claim:false,
         min_ratios:{single_ttft:$min_single_ttft_ratio,
           single_wall:$min_single_wall_ratio,
           four_slot_wave:$min_four_slot_wave_ratio}
       },
       semantic_sha256:$semantic_sha256,baseline:$baseline[0],candidate:$candidate[0],
       ratios:{
         single_ttft:($baseline[0].single_median_ttft_ms / $candidate[0].single_median_ttft_ms),
         single_wall:($baseline[0].single_median_wall_seconds / $candidate[0].single_median_wall_seconds),
         four_slot_wave:($baseline[0].four_slot_median_wave_seconds / $candidate[0].four_slot_median_wave_seconds)
       }}' >"$receipt_tmp"

qwen35_compound_publish_receipt "$receipt_tmp" "$OUT_DIR/receipt.json"
jq . "$OUT_DIR/receipt.json"
echo "receipt: $OUT_DIR/receipt.json" >&2
