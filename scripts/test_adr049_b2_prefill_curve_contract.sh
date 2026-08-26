#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bench="$script_dir/bench_adr049_b2_prefill_curve.sh"
verify="$script_dir/verify_adr049_b2_prefill_curve.py"
tmp_dir=$(mktemp -d -t hf2q-adr049-b2-contract.XXXXXX)
trap 'rm -rf "$tmp_dir"' EXIT

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

# Rev-1 failed for two independent evidence-integrity reasons: its Qwen
# payload landed outside the 128-row bin and script(1) buffered every trace.
# Keep both corrections executable so a later cleanup cannot make the gate
# vacuous again.
grep -Fq 'qwen35_moe) trace_kind=qwen35_chunk; payload_word_adjustment=36' "$bench"
grep -Fq 'preflight_live_trace' "$bench"
grep -Fq 'server log is not live/unbuffered after the preflight request' "$bench"
grep -Fq "s/\\e\\[[0-9;]*m//g" "$bench"

make_fixture() {
    local destination=$1 intercept=$2 slope=$3 family=${4:-qwen35_moe} trials=${5:-7}
    local trace_kind trace_label trace_prefix phase sweeps sweep position target
    local sample_id request response trace wall_file prefill secs ttft wall wall_secs
    local -a widths=(128 256 512 1024 1792) order
    case "$family" in
        qwen35_moe)
            trace_kind=qwen35_chunk
            trace_label='Qwen35 bounded prefill chunk complete'
            trace_prefix=chunk_tokens
            ;;
        gemma4_moe)
            trace_kind=gemma4_transaction
            trace_label='Gemma4 bounded prefill transaction complete'
            trace_prefix=advanced_tokens
            ;;
        *) return 2 ;;
    esac
    mkdir -p "$destination/samples"
    : >"$destination/samples.jsonl"
    : >"$destination/server-measurement.log"
    for phase in warmup measure; do
        if [[ "$phase" == warmup ]]; then sweeps=2; else sweeps=$trials; fi
        for ((sweep = 0; sweep < sweeps; sweep++)); do
            if ((sweep % 2 == 0)); then order=("${widths[@]}"); else order=(1792 1024 512 256 128); fi
            for ((position = 0; position < ${#order[@]}; position++)); do
                target=${order[$position]}
                sample_id=$(printf '%s-%02d-%02d-%04d' "$phase" "$sweep" "$position" "$target")
                request="$destination/samples/$sample_id.request.json"
                response="$destination/samples/$sample_id.response.json"
                trace="$destination/samples/$sample_id.trace.log"
                wall_file="$destination/samples/$sample_id.wall"
                prefill=$(awk -v f="$intercept" -v c="$slope" -v rows="$target" \
                    'BEGIN { printf "%.9f", f + c * rows }')
                secs=$(awk -v ms="$prefill" 'BEGIN { printf "%.12f", ms / 1000 }')
                ttft=$(awk -v ms="$prefill" 'BEGIN { printf "%.9f", ms + 1 }')
                wall=$(awk -v ms="$prefill" 'BEGIN { printf "%.9f", ms + 2 }')
                wall_secs=$(awk -v ms="$wall" 'BEGIN { printf "%.12f", ms / 1000 }')
                jq -n --arg content "adr049-b2-$sample_id measurement Reply with one word." '{
                  model:"fixture-model",messages:[{role:"user",content:$content}],
                  max_tokens:1,seed:42,temperature:0,repetition_penalty:1,stream:false,
                  hf2q_enable_thinking:false,chat_template_kwargs:{enable_thinking:false}
                }' >"$request"
                jq -n --argjson prompt "$target" --argjson secs "$secs" \
                    --argjson ttft "$ttft" '{
                      choices:[{message:{content:"OK"},finish_reason:"length"}],
                      usage:{prompt_tokens:$prompt,prompt_tokens_details:{cached_tokens:0}},
                      x_hf2q_timing:{prefill_time_secs:$secs,time_to_first_token_ms:$ttft}
                    }' >"$response"
                printf '%s=%s %s\n' "$trace_prefix" "$target" "$trace_label" >"$trace"
                printf '%s\n' "$wall_secs" >"$wall_file"
                printf '%s=%s %s\n' "$trace_prefix" "$target" "$trace_label" \
                    >>"$destination/server-measurement.log"
                jq -cn --arg sample "$sample_id" --arg phase "$phase" \
                    --argjson sweep "$sweep" --argjson position "$position" \
                    --argjson target "$target" --argjson prefill "$prefill" \
                    --argjson ttft "$ttft" --argjson wall "$wall" \
                    --arg request "samples/$sample_id.request.json" \
                    --arg request_sha "$(sha256_file "$request")" \
                    --arg response "samples/$sample_id.response.json" \
                    --arg response_sha "$(sha256_file "$response")" \
                    --arg wall_path "samples/$sample_id.wall" \
                    --arg wall_sha "$(sha256_file "$wall_file")" \
                    --arg trace "samples/$sample_id.trace.log" \
                    --arg trace_sha "$(sha256_file "$trace")" '{
                      schema_version:1,sample_id:$sample,phase:$phase,sweep:$sweep,
                      position:$position,target_rows:$target,prompt_tokens:$target,
                      cached_tokens:0,work_rows:$target,prefill_ms:$prefill,
                      ttft_ms:$ttft,wall_ms:$wall,trace_event_count:1,
                      trace_advanced_rows:$target,request_path:$request,
                      request_sha256:$request_sha,response_path:$response,
                      response_sha256:$response_sha,trace_path:$trace,
                      wall_path:$wall_path,wall_sha256:$wall_sha,
                      trace_sha256:$trace_sha
                    }' >>"$destination/samples.jsonl"
            done
        done
    done
    : >"$destination/thermal-settle.log"
    : >"$destination/contention-settle.log"
    for timestamp in 1000 1005 1010 1015 1020 1025 1030 1035 1040 1045 1050 1055 1060; do
        printf '%s\tnominal\tadr049-b2-settle\n' "$timestamp" >>"$destination/thermal-settle.log"
        printf '%s\tquiet\tadr049-b2-settle\t100\t-\n' "$timestamp" >>"$destination/contention-settle.log"
    done
    printf '2000\tnominal\tadr049-b2-measurement-start\n' >"$destination/thermal-measurement.log"
    printf '2002\tfair\tadr049-b2-measurement\n' >>"$destination/thermal-measurement.log"
    printf '2004\tfair\tadr049-b2-measurement-end\n' >>"$destination/thermal-measurement.log"
    printf '2000\tquiet\tadr049-b2-measurement-start\t100\t-\n' >"$destination/contention-measurement.log"
    printf '2002\tquiet\tadr049-b2-measurement\t100\t-\n' >>"$destination/contention-measurement.log"
    printf '2004\tquiet\tadr049-b2-measurement-end\t100\t-\n' >>"$destination/contention-measurement.log"
    jq -n '{data:[{id:"fixture-model"}]}' >"$destination/models.json"
    jq -n --arg family "$family" --arg trace_kind "$trace_kind" --argjson trials "$trials" \
        --arg samples "$(sha256_file "$destination/samples.jsonl")" \
        --arg server "$(sha256_file "$destination/server-measurement.log")" \
        --arg models "$(sha256_file "$destination/models.json")" \
        --arg settle "$(sha256_file "$destination/thermal-settle.log")" \
        --arg measurement "$(sha256_file "$destination/thermal-measurement.log")" \
        --arg contention_settle "$(sha256_file "$destination/contention-settle.log")" \
        --arg contention_measurement "$(sha256_file "$destination/contention-measurement.log")" '{
          schema_version:1,status:"measured",family:$family,trace_kind:$trace_kind,
          width_targets:[128,256,512,1024,1792],warmups:2,trials:$trials,
          order:"ascending-descending-alternating",max_slots:4,
          request_settings:{max_tokens:1,seed:42,repetition_penalty:1,stream:false,temperature:0,thinking:false},
          identity:{source_sha:("a"*40),source_dirty:false,binary_path:"/fixture/hf2q",
            binary_sha256:("b"*64),model_path:"/fixture/model.gguf",model_sha256:("c"*64),
            model_bytes:123456,server_pid:100,server_command:"fixture server",
            server_model_id:"fixture-model",base_url:"http://127.0.0.1:1"},
          files:{
            samples:{path:"samples.jsonl",sha256:$samples},
            models:{path:"models.json",sha256:$models},
            server_log:{path:"server-measurement.log",sha256:$server},
            thermal_settle:{path:"thermal-settle.log",sha256:$settle},
            thermal_measurement:{path:"thermal-measurement.log",sha256:$measurement},
            contention_settle:{path:"contention-settle.log",sha256:$contention_settle},
            contention_measurement:{path:"contention-measurement.log",sha256:$contention_measurement}
          }
        }' >"$destination/manifest.json"
}

expect_rejected() {
    local label=$1
    shift
    if python3 "$verify" "$@" >/dev/null 2>&1; then
        echo "ADR-049 B.2 verifier accepted invalid fixture: $label" >&2
        exit 1
    fi
}

confirmed="$tmp_dir/confirmed"
make_fixture "$confirmed" 20 0.05
python3 "$verify" "$confirmed" "$confirmed/summary.json"
jq -e '.status == "valid" and .analysis.decision == "confirmed" and .analysis.trials_per_width == 7' "$confirmed/summary.json" >/dev/null
expect_rejected summary-overwrite "$confirmed" "$confirmed/summary.json"

extended="$tmp_dir/extended"
make_fixture "$extended" 20 0.05 qwen35_moe 21
python3 "$verify" "$extended" \
    | jq -e '.analysis.decision == "confirmed" and .analysis.trials_per_width == 21' >/dev/null

bad_trials="$tmp_dir/bad-trials"
cp -R "$confirmed" "$bad_trials"
jq '.trials = 9' "$bad_trials/manifest.json" >"$bad_trials/manifest.new"
mv "$bad_trials/manifest.new" "$bad_trials/manifest.json"
expect_rejected trial-count "$bad_trials"

gemma="$tmp_dir/gemma"
make_fixture "$gemma" 20 0.05 gemma4_moe
python3 "$verify" "$gemma" | jq -e '.analysis.decision == "confirmed"' >/dev/null

falsified="$tmp_dir/falsified"
make_fixture "$falsified" 1 0.2
python3 "$verify" "$falsified" | jq -e '.status == "valid" and .analysis.decision == "falsified"' >/dev/null

inconclusive="$tmp_dir/inconclusive"
make_fixture "$inconclusive" 5 0.05
python3 "$verify" "$inconclusive" | jq -e '.status == "valid" and .analysis.decision == "inconclusive"' >/dev/null

invalid_fit="$tmp_dir/invalid-fit"
make_fixture "$invalid_fit" 20 0
expect_rejected invalid-fit "$invalid_fit"

bad_hash="$tmp_dir/bad-hash"
cp -R "$confirmed" "$bad_hash"
printf '\n' >>"$bad_hash/samples.jsonl"
expect_rejected raw-hash "$bad_hash"

bad_order="$tmp_dir/bad-order"
cp -R "$confirmed" "$bad_order"
jq -c 'if input_line_number == 1 then .target_rows = 256 else . end' \
    "$bad_order/samples.jsonl" >"$bad_order/samples.new"
mv "$bad_order/samples.new" "$bad_order/samples.jsonl"
jq --arg sha "$(sha256_file "$bad_order/samples.jsonl")" \
    '.files.samples.sha256 = $sha' "$bad_order/manifest.json" >"$bad_order/manifest.new"
mv "$bad_order/manifest.new" "$bad_order/manifest.json"
expect_rejected order-drift "$bad_order"

bad_response="$tmp_dir/bad-response"
cp -R "$confirmed" "$bad_response"
first_response=$(jq -r -s '.[0].response_path' "$bad_response/samples.jsonl")
jq '.x_hf2q_timing.prefill_time_secs *= 2' "$bad_response/$first_response" >"$bad_response/response.new"
mv "$bad_response/response.new" "$bad_response/$first_response"
first_sha=$(sha256_file "$bad_response/$first_response")
jq -c --arg sha "$first_sha" 'if input_line_number == 1 then .response_sha256 = $sha else . end' \
    "$bad_response/samples.jsonl" >"$bad_response/samples.new"
mv "$bad_response/samples.new" "$bad_response/samples.jsonl"
jq --arg sha "$(sha256_file "$bad_response/samples.jsonl")" \
    '.files.samples.sha256 = $sha' "$bad_response/manifest.json" >"$bad_response/manifest.new"
mv "$bad_response/manifest.new" "$bad_response/manifest.json"
expect_rejected response-mismatch "$bad_response"

bad_cached="$tmp_dir/bad-cached"
cp -R "$confirmed" "$bad_cached"
first_response=$(jq -r -s '.[0].response_path' "$bad_cached/samples.jsonl")
first_trace=$(jq -r -s '.[0].trace_path' "$bad_cached/samples.jsonl")
first_target=$(jq -r -s '.[0].prompt_tokens' "$bad_cached/samples.jsonl")
first_work=$((first_target - 1))
jq '.usage.prompt_tokens_details.cached_tokens = 1' \
    "$bad_cached/$first_response" >"$bad_cached/response.new"
mv "$bad_cached/response.new" "$bad_cached/$first_response"
first_sha=$(sha256_file "$bad_cached/$first_response")
printf 'chunk_tokens=%s Qwen35 bounded prefill chunk complete\n' "$first_work" \
    >"$bad_cached/$first_trace"
first_trace_sha=$(sha256_file "$bad_cached/$first_trace")
jq -c --arg sha "$first_sha" --arg trace_sha "$first_trace_sha" \
    --argjson work "$first_work" \
    'if input_line_number == 1 then
       .response_sha256 = $sha
       | .cached_tokens = 1
       | .work_rows = $work
       | .trace_advanced_rows = $work
       | .trace_sha256 = $trace_sha
     else . end' \
    "$bad_cached/samples.jsonl" >"$bad_cached/samples.new"
mv "$bad_cached/samples.new" "$bad_cached/samples.jsonl"
jq --arg sha "$(sha256_file "$bad_cached/samples.jsonl")" \
    '.files.samples.sha256 = $sha' "$bad_cached/manifest.json" \
    >"$bad_cached/manifest.new"
mv "$bad_cached/manifest.new" "$bad_cached/manifest.json"
expect_rejected cached-prefix "$bad_cached"

bad_thermal="$tmp_dir/bad-thermal"
cp -R "$confirmed" "$bad_thermal"
sed '1s/nominal/fair/' "$bad_thermal/thermal-measurement.log" >"$bad_thermal/thermal.new"
mv "$bad_thermal/thermal.new" "$bad_thermal/thermal-measurement.log"
jq --arg sha "$(sha256_file "$bad_thermal/thermal-measurement.log")" \
    '.files.thermal_measurement.sha256 = $sha' "$bad_thermal/manifest.json" >"$bad_thermal/manifest.new"
mv "$bad_thermal/manifest.new" "$bad_thermal/manifest.json"
expect_rejected thermal-start "$bad_thermal"

echo "ADR-049 B.2 prefill curve receipt contract passed"
