#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
verifier="$root_dir/scripts/verify_qwen38_long_decode_receipt.sh"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qwen38-contract.XXXXXX")
cleanup_contract() { rm -rf "$tmp"; }
trap cleanup_contract EXIT

source_sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
crate_sha=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
binary_sha=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
model_sha=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
thermal_probe_sha=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

build_fixture() {
  local destination=$1
  local include_seed=${2:-0}
  local slow_auto=${3:-0}
  local noisy_off=${4:-0}
  local benchmark_dir="$destination/benchmark"
  local trial_index mode decode_tps decode_seconds elapsed_ms prewarm
  local response_total_seconds wall_total_seconds
  local trial_start request_start request_end trial_end
  local trial_dir artifacts_json prompt_sha prompt_bytes request_sha semantic_sha
  mkdir -p "$benchmark_dir" "$destination/thermal"
  printf 'synthetic canonical prompt\n' >"$benchmark_dir/prompt.txt"
  jq -n --rawfile prompt "$benchmark_dir/prompt.txt" '
    {model:"Qwen3.8 27B",messages:[
      {role:"system",content:"synthetic"},{role:"user",content:$prompt}],
      temperature:0,max_tokens:512,stream:false,hf2q_enable_thinking:false,
      repetition_penalty:1.0}
  ' >"$benchmark_dir/request.json"
  if [[ "$include_seed" == 1 ]]; then
    jq '.seed = 1234' "$benchmark_dir/request.json" \
      >"$benchmark_dir/request.seed.json"
    mv "$benchmark_dir/request.seed.json" "$benchmark_dir/request.json"
  fi
  prompt_sha=$(sha256_file "$benchmark_dir/prompt.txt")
  prompt_bytes=$(wc -c <"$benchmark_dir/prompt.txt" | tr -d ' ')
  request_sha=$(sha256_file "$benchmark_dir/request.json")
  : >"$benchmark_dir/phase.log"

  trial_index=0
  for mode in off auto auto off; do
    trial_index=$((trial_index + 1))
    case "$trial_index" in
      1)
        if [[ "$noisy_off" == 1 ]]; then
          decode_tps=9.7; decode_seconds=52.7835051546; wall_total_seconds=53.9
        else
          decode_tps=10; decode_seconds=51.2; wall_total_seconds=52.3
        fi
        ;;
      2)
        if [[ "$slow_auto" == 1 ]]; then
          decode_tps=10.4; decode_seconds=49.2307692308; wall_total_seconds=50.4
        else
          decode_tps=12; decode_seconds=42.6666666667; wall_total_seconds=43.8
        fi
        ;;
      3)
        if [[ "$slow_auto" == 1 ]]; then
          decode_tps=10.5; decode_seconds=48.7619047619; wall_total_seconds=49.9
        else
          decode_tps=12.2; decode_seconds=41.9672131148; wall_total_seconds=43.1
        fi
        ;;
      4) decode_tps=10.2; decode_seconds=50.1960784314; wall_total_seconds=51.3 ;;
    esac
    response_total_seconds=$(awk -v seconds="$decode_seconds" \
      'BEGIN { printf "%.10f", seconds + 1 }')
    case "$trial_index" in
      1) trial_start=2000; request_start=2001; request_end=2053; trial_end=2054 ;;
      2) trial_start=2055; request_start=2056; request_end=2100; trial_end=2101 ;;
      3) trial_start=2102; request_start=2103; request_end=2146; trial_end=2147 ;;
      4) trial_start=2148; request_start=2149; request_end=2200; trial_end=2201 ;;
    esac
    {
      printf '%s\t%s\t%s\ttrial-start\n' "$trial_start" "$trial_index" "$mode"
      printf '%s\t%s\t%s\trequest-start\n' "$request_start" "$trial_index" "$mode"
      printf '%s\t%s\t%s\trequest-end\n' "$request_end" "$trial_index" "$mode"
      printf '%s\t%s\t%s\ttrial-end\n' "$trial_end" "$trial_index" "$mode"
    } >>"$benchmark_dir/phase.log"
    elapsed_ms=$(awk -v seconds="$decode_seconds" 'BEGIN { printf "%.6f", seconds * 1000 }')
    if [[ "$mode" == auto ]]; then prewarm=true; else prewarm=false; fi
    trial_dir="$benchmark_dir/trial-${trial_index}-${mode}"
    mkdir -p "$trial_dir"
    cp "$benchmark_dir/request.json" "$trial_dir/request.json"
    jq -n --argjson decode_seconds "$decode_seconds" --argjson decode_tps "$decode_tps" \
      --argjson response_total_seconds "$response_total_seconds" '
      {id:"synthetic",object:"chat.completion",created:1,
       model:"Qwen3.8 27B",choices:[{index:0,
         message:{role:"assistant",content:"same exact output"},
         finish_reason:"length"}],
       usage:{prompt_tokens:105100,completion_tokens:512,total_tokens:105612},
       x_hf2q_timing:{prefill_time_secs:1,decode_time_secs:$decode_seconds,
         total_time_secs:$response_total_seconds,time_to_first_token_ms:1000,
         prefill_tokens_per_sec:105100,decode_tokens_per_sec:$decode_tps,
         gpu_sync_count:512,gpu_dispatch_count:1024}}
    ' >"$trial_dir/response.json"
    jq -S '{model,choices,usage}' "$trial_dir/response.json" \
      >"$trial_dir/semantic.json"
    printf 'http_code=200\ntotal_seconds=%s\n' "$wall_total_seconds" \
      >"$trial_dir/curl.metrics"
    printf '{"ready":true,"detail":"ready"}\n' >"$trial_dir/readyz.json"
    printf '{"object":"list","data":[{"id":"Qwen3.8 27B","loaded":true}]}\n' \
      >"$trial_dir/models.json"
    for offset in 0 5 10 15 20 25 30; do
      printf '%s\tnominal\tqwen38-trial-%s-%s-settle\n' \
        "$((1000 + trial_index * 100 + offset))" "$trial_index" "$mode"
    done >"$trial_dir/settle.log"
    printf 'HF2Q_QWEN_GQA_Q2=%s\nHF2Q_PIPELINE_PREWARM_LOG=1\nQWEN38_VISION=off\n' \
      "$mode" >"$trial_dir/environment.txt"
    {
      printf '[prewarm] warmed 1 / 1 no-const kernels + 1 / 1 fa-prefill variants + gqa_q2=%s in 1.00ms\n' "$prewarm"
      if [[ "$mode" == auto ]]; then
        printf 'INFO kv_seq_len=105100 num_heads=32 num_kv_heads=8 Qwen TQ-HB decode selected GQA-cooperative Q2 attention\n'
      fi
      printf 'INFO mode=unary generated_tokens=512 elapsed_ms=%s tokens_per_second=%s Qwen35 decode complete\n' \
        "$elapsed_ms" "$decode_tps"
    } >"$trial_dir/server.log"
    semantic_sha=$(sha256_file "$trial_dir/semantic.json")
    artifacts_json=$(
      for name in request.json response.json semantic.json curl.metrics server.log readyz.json models.json environment.txt settle.log; do
        jq -n --arg name "$name" --arg sha256 "$(sha256_file "$trial_dir/$name")" \
          '{name:$name,sha256:$sha256}'
      done | jq -s .
    )
    jq -n --argjson index "$trial_index" --arg mode "$mode" \
      --arg binary_sha256 "$binary_sha" --arg model_sha256 "$model_sha" \
      --arg request_sha256 "$request_sha" --arg semantic_sha256 "$semantic_sha" \
      --argjson decode_seconds "$decode_seconds" --argjson decode_tps "$decode_tps" \
      --argjson response_total_seconds "$response_total_seconds" \
      --argjson wall_total_seconds "$wall_total_seconds" \
      --argjson artifacts "$artifacts_json" '
      {index:$index,mode:$mode,status:"pass",binary_sha256:$binary_sha256,
       binary_file_identity:"1:2",model_sha256:$model_sha256,
       model_file_identity:"3:4",request_sha256:$request_sha256,
       semantic_sha256:$semantic_sha256,prompt_tokens:105100,
       completion_tokens:512,finish_reason:"length",decode_seconds:$decode_seconds,
       decode_tokens_per_second:$decode_tps,
       response_total_seconds:$response_total_seconds,
       wall_total_seconds:$wall_total_seconds,artifacts:$artifacts}
    ' >"$trial_dir/trial.json"
  done

  if [[ "$noisy_off" == 1 ]]; then
    off_mean=9.95
    off_spread_percent=5.025126
    off_wall_mean=52.6
  else
    off_mean=10.1
    off_spread_percent=1.980198
    off_wall_mean=51.8
  fi
  if [[ "$slow_auto" == 1 ]]; then
    auto_mean=10.45
    auto_spread_percent=0.956938
    auto_wall_mean=50.15
  else
    auto_mean=12.1
    auto_spread_percent=1.652893
    auto_wall_mean=43.45
  fi
  improvement_percent=$(awk -v baseline="$off_mean" -v candidate="$auto_mean" \
    'BEGIN { printf "%.6f", ((candidate / baseline) - 1) * 100 }')
  semantic_sha=$(sha256_file "$benchmark_dir/trial-1-off/semantic.json")
  phase_sha=$(sha256_file "$benchmark_dir/phase.log")
  phase_bytes=$(wc -c <"$benchmark_dir/phase.log" | tr -d ' ')
  jq -n --arg source_sha "$source_sha" --arg crate_sha256 "$crate_sha" \
    --arg binary_sha256 "$binary_sha" --arg model_sha256 "$model_sha" \
    --arg prompt_sha256 "$prompt_sha" --argjson prompt_bytes "$prompt_bytes" \
    --arg phase_sha256 "$phase_sha" --argjson phase_bytes "$phase_bytes" \
    --arg request_sha256 "$request_sha" --arg semantic_sha256 "$semantic_sha" \
    --arg thermal_probe_sha256 "$thermal_probe_sha" \
    --argjson off_mean "$off_mean" --argjson auto_mean "$auto_mean" \
    --argjson off_spread_percent "$off_spread_percent" \
    --argjson auto_spread_percent "$auto_spread_percent" \
    --argjson off_wall_mean "$off_wall_mean" \
    --argjson auto_wall_mean "$auto_wall_mean" \
    --argjson improvement_percent "$improvement_percent" \
    --slurpfile trial1 "$benchmark_dir/trial-1-off/trial.json" \
    --slurpfile trial2 "$benchmark_dir/trial-2-auto/trial.json" \
    --slurpfile trial3 "$benchmark_dir/trial-3-auto/trial.json" \
    --slurpfile trial4 "$benchmark_dir/trial-4-off/trial.json" '
    {schema_version:1,status:"pass",benchmark:"qwen38-long-decode-gqa-q2",
     identity:{source_sha:$source_sha,crate_sha256:$crate_sha256,
       binary:{path:"/sealed/hf2q",sha256:$binary_sha256,file_identity:"1:2"},
       model:{id:"Qwen3.8 27B",path:"/models/qwen38.gguf",
         sha256:$model_sha256,file_identity:"3:4",bytes:123456},
       prompt:{path:"prompt.txt",sha256:$prompt_sha256,bytes:$prompt_bytes,
         padding_tokens:105000},
       phase_log:{path:"phase.log",sha256:$phase_sha256,bytes:$phase_bytes},
       request:{path:"request.json",sha256:$request_sha256},
       hardware:{model:"Mac16,1",chip:"Apple M5 Max",arch:"arm64",
         memory_bytes:137438953472,os_version:"26.0",
         thermal_probe:{path:"/usr/bin/swift",sha256:$thermal_probe_sha256}}},
     settings:{temperature:0,max_tokens:512,stream:false,thinking:false,
       repetition_penalty:1.0,min_prompt_tokens:100000,max_prompt_tokens:120000,
       trial_settle_seconds:30,maximum_within_arm_spread_percent:5,
       maximum_wall_timing_delta_seconds:2},
     trial_order:["off","auto","auto","off"],
     trials:[$trial1[0],$trial2[0],$trial3[0],$trial4[0]],
     aggregate:{off_mean_decode_tokens_per_second:$off_mean,
       auto_mean_decode_tokens_per_second:$auto_mean,
       off_within_arm_spread_percent:$off_spread_percent,
       auto_within_arm_spread_percent:$auto_spread_percent,
       off_mean_wall_seconds:$off_wall_mean,
       auto_mean_wall_seconds:$auto_wall_mean,
       improvement_percent:$improvement_percent,minimum_improvement_percent:15,
       exact_output_sha256:$semantic_sha256}}
  ' >"$benchmark_dir/summary.json"
  printf '%s  summary.json\n' "$(sha256_file "$benchmark_dir/summary.json")" \
    >"$benchmark_dir/summary.json.sha256"

  for offset in 0 5 10 15 20 25 30 35 40 45 50 55 60; do
    printf '%d\tnominal\tqwen38-long-decode-settle\n' "$((1000 + offset))"
  done >"$destination/thermal/settle.log"
  for epoch in $(seq 1998 2 2204); do
    if [[ "$epoch" == 1998 ]]; then
      printf '%s\tnominal\tqwen38-long-decode-measurement-start\n' "$epoch"
    elif [[ "$epoch" == 2204 ]]; then
      printf '%s\tfair\tqwen38-long-decode-measurement-end\n' "$epoch"
    else
      printf '%s\tfair\tqwen38-long-decode-measurement\n' "$epoch"
    fi
  done >"$destination/thermal/measurement.log"
  jq -n --arg benchmark_summary_sha256 "$(sha256_file "$benchmark_dir/summary.json")" \
    --arg settle_log_sha256 "$(sha256_file "$destination/thermal/settle.log")" \
    --arg measurement_log_sha256 "$(sha256_file "$destination/thermal/measurement.log")" '
    {status:"pass",phase:"qwen38-long-decode",required_start_state:"nominal",
     maximum_measurement_state:"fair",
     runtime_preflight:"pass",measurement_scope:"full-abba-benchmark",
     benchmark_summary_sha256:$benchmark_summary_sha256,settle_seconds:60,
     settle_duration_seconds:60,settle_samples:13,measurement_samples:104,
     measurement_duration_seconds:206,sample_interval_seconds:2,
     maximum_sample_gap_seconds:5,settle_sample_interval_seconds:5,
     maximum_settle_sample_gap_seconds:8,non_nominal_measurement_samples:103,
     fair_measurement_samples:103,over_limit_measurement_samples:0,
     settle_telemetry_gaps:0,telemetry_gaps:0,
     settle_log_sha256:$settle_log_sha256,
     measurement_log_sha256:$measurement_log_sha256}
  ' >"$destination/thermal/summary.json"
  jq -n --slurpfile benchmark "$benchmark_dir/summary.json" \
    --slurpfile thermal "$destination/thermal/summary.json" \
    '{schema_version:1,status:"pass",benchmark:$benchmark[0],thermal:$thermal[0]}' \
    >"$destination/receipt.json"
}

expect_rejected() {
  local fixture=$1
  local label=$2
  if bash "$verifier" release "$fixture" "$source_sha" "$crate_sha" \
    "$binary_sha" "$model_sha" >/dev/null 2>&1; then
    echo "Qwen3.8 verifier accepted invalid synthetic evidence: $label" >&2
    exit 1
  fi
}

expect_benchmark_rejected() {
  local fixture=$1
  local label=$2
  if bash "$verifier" benchmark "$fixture/benchmark" "$source_sha" "$crate_sha" \
    "$binary_sha" "$model_sha" >/dev/null 2>&1; then
    echo "Qwen3.8 verifier accepted invalid benchmark evidence: $label" >&2
    exit 1
  fi
}

rehash_trial_into_summary() {
  local fixture=$1
  local trial_index=$2
  local trial_dir=$3
  local trial_json="$trial_dir/trial.json"
  local summary="$fixture/benchmark/summary.json"
  local artifact
  for artifact in request.json response.json semantic.json curl.metrics server.log readyz.json models.json environment.txt settle.log; do
    jq --arg name "$artifact" --arg sha "$(sha256_file "$trial_dir/$artifact")" '
      .artifacts |= map(if .name == $name then .sha256 = $sha else . end)
    ' "$trial_json" >"$trial_json.tmp"
    mv "$trial_json.tmp" "$trial_json"
  done
  jq --argjson index "$trial_index" --slurpfile trial "$trial_json" '
    .trials[$index] = $trial[0]
  ' "$summary" >"$summary.tmp"
  mv "$summary.tmp" "$summary"
  printf '%s  summary.json\n' "$(sha256_file "$summary")" >"$summary.sha256"
}

valid="$tmp/valid"
build_fixture "$valid"
bash "$verifier" release "$valid" "$source_sha" "$crate_sha" \
  "$binary_sha" "$model_sha"

seeded="$tmp/seeded"
build_fixture "$seeded" 1
expect_rejected "$seeded" seeded-request

wrong_order="$tmp/wrong-order"
cp -R "$valid" "$wrong_order"
jq '.trial_order = ["off","auto","off","auto"]' \
  "$wrong_order/benchmark/summary.json" >"$wrong_order/benchmark/summary.tmp"
mv "$wrong_order/benchmark/summary.tmp" "$wrong_order/benchmark/summary.json"
printf '%s  summary.json\n' \
  "$(sha256_file "$wrong_order/benchmark/summary.json")" \
  >"$wrong_order/benchmark/summary.json.sha256"
if bash "$verifier" benchmark "$wrong_order/benchmark" "$source_sha" "$crate_sha" \
  "$binary_sha" "$model_sha" >/dev/null 2>&1; then
  echo "Qwen3.8 verifier accepted a non-ABBA trial order" >&2
  exit 1
fi

tampered="$tmp/tampered"
cp -R "$valid" "$tampered"
printf 'tamper\n' >>"$tampered/benchmark/trial-2-auto/server.log"
expect_rejected "$tampered" tampered-raw-artifact

wrong_loaded_model="$tmp/wrong-loaded-model"
cp -R "$valid" "$wrong_loaded_model"
printf '{"object":"list","data":[{"id":"downloaded-shadow","loaded":true}]}\n' \
  >"$wrong_loaded_model/benchmark/trial-2-auto/models.json"
rehash_trial_into_summary "$wrong_loaded_model" 1 \
  "$wrong_loaded_model/benchmark/trial-2-auto"
expect_benchmark_rejected "$wrong_loaded_model" wrong-loaded-model

download_fallback="$tmp/download-fallback"
cp -R "$valid" "$download_fallback"
printf 'INFO auto-pipeline: downloading from HF Hub repo="Qwen/Qwen3.8-27B"\n' \
  >>"$download_fallback/benchmark/trial-2-auto/server.log"
rehash_trial_into_summary "$download_fallback" 1 \
  "$download_fallback/benchmark/trial-2-auto"
expect_benchmark_rejected "$download_fallback" auto-pipeline-download

slow="$tmp/slow"
build_fixture "$slow" 0 1
expect_rejected "$slow" below-fifteen-percent

noisy="$tmp/noisy"
build_fixture "$noisy" 0 0 1
expect_benchmark_rejected "$noisy" excessive-within-arm-spread

wall_inconsistent="$tmp/wall-inconsistent"
cp -R "$valid" "$wall_inconsistent"
printf 'http_code=200\ntotal_seconds=60\n' \
  >"$wall_inconsistent/benchmark/trial-2-auto/curl.metrics"
jq '.wall_total_seconds = 60' \
  "$wall_inconsistent/benchmark/trial-2-auto/trial.json" \
  >"$wall_inconsistent/benchmark/trial-2-auto/trial.tmp"
mv "$wall_inconsistent/benchmark/trial-2-auto/trial.tmp" \
  "$wall_inconsistent/benchmark/trial-2-auto/trial.json"
rehash_trial_into_summary "$wall_inconsistent" 1 \
  "$wall_inconsistent/benchmark/trial-2-auto"
expect_benchmark_rejected "$wall_inconsistent" inconsistent-independent-wall-clock

stream_mode="$tmp/stream-mode"
cp -R "$valid" "$stream_mode"
sed 's/mode=unary/mode=stream/' \
  "$stream_mode/benchmark/trial-2-auto/server.log" \
  >"$stream_mode/benchmark/trial-2-auto/server.tmp"
mv "$stream_mode/benchmark/trial-2-auto/server.tmp" \
  "$stream_mode/benchmark/trial-2-auto/server.log"
rehash_trial_into_summary "$stream_mode" 1 \
  "$stream_mode/benchmark/trial-2-auto"
expect_benchmark_rejected "$stream_mode" wrong-decode-telemetry-mode

missing_decode="$tmp/missing-decode"
cp -R "$valid" "$missing_decode"
grep -v 'Qwen35 decode complete' \
  "$missing_decode/benchmark/trial-2-auto/server.log" \
  >"$missing_decode/benchmark/trial-2-auto/server.tmp"
mv "$missing_decode/benchmark/trial-2-auto/server.tmp" \
  "$missing_decode/benchmark/trial-2-auto/server.log"
rehash_trial_into_summary "$missing_decode" 1 \
  "$missing_decode/benchmark/trial-2-auto"
expect_benchmark_rejected "$missing_decode" missing-decode-telemetry

duplicate_decode="$tmp/duplicate-decode"
cp -R "$valid" "$duplicate_decode"
decode_line=$(grep 'Qwen35 decode complete' \
  "$duplicate_decode/benchmark/trial-2-auto/server.log")
printf '%s\n' "$decode_line" \
  >>"$duplicate_decode/benchmark/trial-2-auto/server.log"
rehash_trial_into_summary "$duplicate_decode" 1 \
  "$duplicate_decode/benchmark/trial-2-auto"
expect_benchmark_rejected "$duplicate_decode" duplicate-decode-telemetry

wrong_generated="$tmp/wrong-generated"
cp -R "$valid" "$wrong_generated"
sed 's/generated_tokens=512/generated_tokens=511/' \
  "$wrong_generated/benchmark/trial-2-auto/server.log" \
  >"$wrong_generated/benchmark/trial-2-auto/server.tmp"
mv "$wrong_generated/benchmark/trial-2-auto/server.tmp" \
  "$wrong_generated/benchmark/trial-2-auto/server.log"
rehash_trial_into_summary "$wrong_generated" 1 \
  "$wrong_generated/benchmark/trial-2-auto"
expect_benchmark_rejected "$wrong_generated" mismatched-generated-token-count

wrong_elapsed="$tmp/wrong-elapsed"
cp -R "$valid" "$wrong_elapsed"
sed 's/elapsed_ms=[^ ]*/elapsed_ms=42676.666667/' \
  "$wrong_elapsed/benchmark/trial-2-auto/server.log" \
  >"$wrong_elapsed/benchmark/trial-2-auto/server.tmp"
mv "$wrong_elapsed/benchmark/trial-2-auto/server.tmp" \
  "$wrong_elapsed/benchmark/trial-2-auto/server.log"
rehash_trial_into_summary "$wrong_elapsed" 1 \
  "$wrong_elapsed/benchmark/trial-2-auto"
expect_benchmark_rejected "$wrong_elapsed" mismatched-decode-elapsed

wrong_tps="$tmp/wrong-tps"
cp -R "$valid" "$wrong_tps"
sed 's/tokens_per_second=[^ ]*/tokens_per_second=13/' \
  "$wrong_tps/benchmark/trial-2-auto/server.log" \
  >"$wrong_tps/benchmark/trial-2-auto/server.tmp"
mv "$wrong_tps/benchmark/trial-2-auto/server.tmp" \
  "$wrong_tps/benchmark/trial-2-auto/server.log"
rehash_trial_into_summary "$wrong_tps" 1 \
  "$wrong_tps/benchmark/trial-2-auto"
expect_benchmark_rejected "$wrong_tps" mismatched-decode-throughput

wrong_semantic="$tmp/wrong-semantic"
cp -R "$valid" "$wrong_semantic"
jq '.choices[0].message.content = "different output"' \
  "$wrong_semantic/benchmark/trial-2-auto/response.json" \
  >"$wrong_semantic/benchmark/trial-2-auto/response.tmp"
mv "$wrong_semantic/benchmark/trial-2-auto/response.tmp" \
  "$wrong_semantic/benchmark/trial-2-auto/response.json"
jq -S '{model,choices,usage}' \
  "$wrong_semantic/benchmark/trial-2-auto/response.json" \
  >"$wrong_semantic/benchmark/trial-2-auto/semantic.json"
jq --arg sha "$(sha256_file "$wrong_semantic/benchmark/trial-2-auto/semantic.json")" \
  '.semantic_sha256 = $sha' \
  "$wrong_semantic/benchmark/trial-2-auto/trial.json" \
  >"$wrong_semantic/benchmark/trial-2-auto/trial.tmp"
mv "$wrong_semantic/benchmark/trial-2-auto/trial.tmp" \
  "$wrong_semantic/benchmark/trial-2-auto/trial.json"
rehash_trial_into_summary "$wrong_semantic" 1 \
  "$wrong_semantic/benchmark/trial-2-auto"
expect_benchmark_rejected "$wrong_semantic" cross-arm-semantic-mismatch

fatal_log="$tmp/fatal-log"
cp -R "$valid" "$fatal_log"
printf 'ERROR GPU Timeout during synthetic trial\n' \
  >>"$fatal_log/benchmark/trial-2-auto/server.log"
rehash_trial_into_summary "$fatal_log" 1 \
  "$fatal_log/benchmark/trial-2-auto"
expect_benchmark_rejected "$fatal_log" fatal-runtime-signature

non_nominal="$tmp/non-nominal"
cp -R "$valid" "$non_nominal"
sed '2s/fair/serious/' "$non_nominal/thermal/measurement.log" \
  >"$non_nominal/thermal/measurement.tmp"
mv "$non_nominal/thermal/measurement.tmp" "$non_nominal/thermal/measurement.log"
jq --arg sha "$(sha256_file "$non_nominal/thermal/measurement.log")" \
  '.measurement_log_sha256 = $sha
   | .fair_measurement_samples = 90
   | .over_limit_measurement_samples = 1' \
  "$non_nominal/thermal/summary.json" \
  >"$non_nominal/thermal/summary.tmp"
mv "$non_nominal/thermal/summary.tmp" "$non_nominal/thermal/summary.json"
jq -n --slurpfile benchmark "$non_nominal/benchmark/summary.json" \
  --slurpfile thermal "$non_nominal/thermal/summary.json" \
  '{schema_version:1,status:"pass",benchmark:$benchmark[0],thermal:$thermal[0]}' \
  >"$non_nominal/receipt.json"
expect_rejected "$non_nominal" non-nominal-thermal-state

mismatched_envelope="$tmp/mismatched-envelope"
cp -R "$valid" "$mismatched_envelope"
awk -F '\t' 'BEGIN { OFS="\t" }
  $1 >= 2000 && $1 <= 2054 { $2="nominal" }
  { print }
' "$mismatched_envelope/thermal/measurement.log" \
  >"$mismatched_envelope/thermal/measurement.tmp"
mv "$mismatched_envelope/thermal/measurement.tmp" \
  "$mismatched_envelope/thermal/measurement.log"
jq --arg sha "$(sha256_file "$mismatched_envelope/thermal/measurement.log")" '
  .measurement_log_sha256 = $sha
  | .non_nominal_measurement_samples = 63
  | .fair_measurement_samples = 63
' "$mismatched_envelope/thermal/summary.json" \
  >"$mismatched_envelope/thermal/summary.tmp"
mv "$mismatched_envelope/thermal/summary.tmp" \
  "$mismatched_envelope/thermal/summary.json"
jq -n --slurpfile benchmark "$mismatched_envelope/benchmark/summary.json" \
  --slurpfile thermal "$mismatched_envelope/thermal/summary.json" \
  '{schema_version:1,status:"pass",benchmark:$benchmark[0],thermal:$thermal[0]}' \
  >"$mismatched_envelope/receipt.json"
expect_rejected "$mismatched_envelope" mismatched-decode-thermal-envelope

echo "Qwen3.8 long-decode receipt contract tests passed" >&2
