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

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

build_fixture() {
  local destination=$1
  local include_seed=${2:-0}
  local slow_auto=${3:-0}
  local benchmark_dir="$destination/benchmark"
  local trial_index mode decode_tps decode_seconds elapsed_ms prewarm
  local trial_dir artifacts_json prompt_sha prompt_bytes request_sha semantic_sha
  mkdir -p "$benchmark_dir" "$destination/thermal"
  printf 'synthetic canonical prompt\n' >"$benchmark_dir/prompt.txt"
  jq -n --rawfile prompt "$benchmark_dir/prompt.txt" '
    {model:"Qwen/Qwen3.8-27B",messages:[
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

  trial_index=0
  for mode in off auto auto off; do
    trial_index=$((trial_index + 1))
    case "$trial_index" in
      1) decode_tps=10; decode_seconds=51.2 ;;
      2)
        if [[ "$slow_auto" == 1 ]]; then
          decode_tps=10.2; decode_seconds=50.1960784314
        else
          decode_tps=14; decode_seconds=36.5714285714
        fi
        ;;
      3)
        if [[ "$slow_auto" == 1 ]]; then
          decode_tps=10.4; decode_seconds=49.2307692308
        else
          decode_tps=16; decode_seconds=32
        fi
        ;;
      4) decode_tps=12; decode_seconds=42.6666666667 ;;
    esac
    elapsed_ms=$(awk -v seconds="$decode_seconds" 'BEGIN { printf "%.6f", seconds * 1000 }')
    if [[ "$mode" == auto ]]; then prewarm=true; else prewarm=false; fi
    trial_dir="$benchmark_dir/trial-${trial_index}-${mode}"
    mkdir -p "$trial_dir"
    cp "$benchmark_dir/request.json" "$trial_dir/request.json"
    jq -n --argjson decode_seconds "$decode_seconds" --argjson decode_tps "$decode_tps" '
      {id:"synthetic",object:"chat.completion",created:1,
       model:"Qwen/Qwen3.8-27B",choices:[{index:0,
         message:{role:"assistant",content:"same exact output"},
         finish_reason:"length"}],
       usage:{prompt_tokens:105100,completion_tokens:512,total_tokens:105612},
       x_hf2q_timing:{prefill_time_secs:1,decode_time_secs:$decode_seconds,
         total_time_secs:($decode_seconds + 1),time_to_first_token_ms:1000,
         prefill_tokens_per_sec:105100,decode_tokens_per_sec:$decode_tps,
         gpu_sync_count:512,gpu_dispatch_count:1024}}
    ' >"$trial_dir/response.json"
    jq -S '{model,choices,usage}' "$trial_dir/response.json" \
      >"$trial_dir/semantic.json"
    printf 'http_code=200\ntotal_seconds=60\n' >"$trial_dir/curl.metrics"
    printf '{"ready":true,"detail":"ready"}\n' >"$trial_dir/readyz.json"
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
      for name in request.json response.json semantic.json curl.metrics server.log readyz.json environment.txt; do
        jq -n --arg name "$name" --arg sha256 "$(sha256_file "$trial_dir/$name")" \
          '{name:$name,sha256:$sha256}'
      done | jq -s .
    )
    jq -n --argjson index "$trial_index" --arg mode "$mode" \
      --arg binary_sha256 "$binary_sha" --arg model_sha256 "$model_sha" \
      --arg request_sha256 "$request_sha" --arg semantic_sha256 "$semantic_sha" \
      --argjson decode_seconds "$decode_seconds" --argjson decode_tps "$decode_tps" \
      --argjson artifacts "$artifacts_json" '
      {index:$index,mode:$mode,status:"pass",binary_sha256:$binary_sha256,
       binary_file_identity:"1:2",model_sha256:$model_sha256,
       model_file_identity:"3:4",request_sha256:$request_sha256,
       semantic_sha256:$semantic_sha256,prompt_tokens:105100,
       completion_tokens:512,finish_reason:"length",decode_seconds:$decode_seconds,
       decode_tokens_per_second:$decode_tps,artifacts:$artifacts}
    ' >"$trial_dir/trial.json"
  done

  off_median=11
  if [[ "$slow_auto" == 1 ]]; then
    auto_median=10.3
  else
    auto_median=15
  fi
  improvement_percent=$(awk -v baseline="$off_median" -v candidate="$auto_median" \
    'BEGIN { printf "%.6f", ((candidate / baseline) - 1) * 100 }')
  semantic_sha=$(sha256_file "$benchmark_dir/trial-1-off/semantic.json")
  jq -n --arg source_sha "$source_sha" --arg crate_sha256 "$crate_sha" \
    --arg binary_sha256 "$binary_sha" --arg model_sha256 "$model_sha" \
    --arg prompt_sha256 "$prompt_sha" --argjson prompt_bytes "$prompt_bytes" \
    --arg request_sha256 "$request_sha" --arg semantic_sha256 "$semantic_sha" \
    --argjson off_median "$off_median" --argjson auto_median "$auto_median" \
    --argjson improvement_percent "$improvement_percent" \
    --slurpfile trial1 "$benchmark_dir/trial-1-off/trial.json" \
    --slurpfile trial2 "$benchmark_dir/trial-2-auto/trial.json" \
    --slurpfile trial3 "$benchmark_dir/trial-3-auto/trial.json" \
    --slurpfile trial4 "$benchmark_dir/trial-4-off/trial.json" '
    {schema_version:1,status:"pass",benchmark:"qwen38-long-decode-gqa-q2",
     identity:{source_sha:$source_sha,crate_sha256:$crate_sha256,
       binary:{path:"/sealed/hf2q",sha256:$binary_sha256,file_identity:"1:2"},
       model:{id:"Qwen/Qwen3.8-27B",path:"/models/qwen38.gguf",
         sha256:$model_sha256,file_identity:"3:4",bytes:123456},
       prompt:{path:"prompt.txt",sha256:$prompt_sha256,bytes:$prompt_bytes,
         padding_tokens:105000},
       request:{path:"request.json",sha256:$request_sha256},
       hardware:{model:"Mac16,1",chip:"Apple M5 Max",arch:"arm64",
         memory_bytes:137438953472,os_version:"26.0"}},
     settings:{temperature:0,max_tokens:512,stream:false,thinking:false,
       repetition_penalty:1.0,min_prompt_tokens:100000,max_prompt_tokens:120000},
     trial_order:["off","auto","auto","off"],
     trials:[$trial1[0],$trial2[0],$trial3[0],$trial4[0]],
     aggregate:{off_median_decode_tokens_per_second:$off_median,
       auto_median_decode_tokens_per_second:$auto_median,
       improvement_percent:$improvement_percent,minimum_improvement_percent:15,
       exact_output_sha256:$semantic_sha256}}
  ' >"$benchmark_dir/summary.json"
  printf '%s  summary.json\n' "$(sha256_file "$benchmark_dir/summary.json")" \
    >"$benchmark_dir/summary.json.sha256"

  for offset in 0 5 10 15 20 25 30 35 40 45 50 55 60; do
    printf '%d\tnominal\tqwen38-long-decode-settle\n' "$((1000 + offset))"
  done >"$destination/thermal/settle.log"
  {
    printf '2000\tnominal\tqwen38-long-decode-measurement-start\n'
    printf '2002\tnominal\tqwen38-long-decode-measurement\n'
    printf '2004\tnominal\tqwen38-long-decode-measurement-end\n'
  } >"$destination/thermal/measurement.log"
  jq -n --arg benchmark_summary_sha256 "$(sha256_file "$benchmark_dir/summary.json")" \
    --arg settle_log_sha256 "$(sha256_file "$destination/thermal/settle.log")" \
    --arg measurement_log_sha256 "$(sha256_file "$destination/thermal/measurement.log")" '
    {status:"pass",phase:"qwen38-long-decode",required_state:"nominal",
     runtime_preflight:"pass",measurement_scope:"full-abba-benchmark",
     benchmark_summary_sha256:$benchmark_summary_sha256,settle_seconds:60,
     settle_duration_seconds:60,settle_samples:13,measurement_samples:3,
     measurement_duration_seconds:4,sample_interval_seconds:2,
     maximum_sample_gap_seconds:5,settle_sample_interval_seconds:5,
     maximum_settle_sample_gap_seconds:8,non_nominal_measurement_samples:0,
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

slow="$tmp/slow"
build_fixture "$slow" 0 1
expect_rejected "$slow" below-fifteen-percent

non_nominal="$tmp/non-nominal"
cp -R "$valid" "$non_nominal"
sed '2s/nominal/fair/' "$non_nominal/thermal/measurement.log" \
  >"$non_nominal/thermal/measurement.tmp"
mv "$non_nominal/thermal/measurement.tmp" "$non_nominal/thermal/measurement.log"
jq --arg sha "$(sha256_file "$non_nominal/thermal/measurement.log")" \
  '.measurement_log_sha256 = $sha' "$non_nominal/thermal/summary.json" \
  >"$non_nominal/thermal/summary.tmp"
mv "$non_nominal/thermal/summary.tmp" "$non_nominal/thermal/summary.json"
jq -n --slurpfile benchmark "$non_nominal/benchmark/summary.json" \
  --slurpfile thermal "$non_nominal/thermal/summary.json" \
  '{schema_version:1,status:"pass",benchmark:$benchmark[0],thermal:$thermal[0]}' \
  >"$non_nominal/receipt.json"
expect_rejected "$non_nominal" non-nominal-thermal-state

echo "Qwen3.8 long-decode receipt contract tests passed" >&2
