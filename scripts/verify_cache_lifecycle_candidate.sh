#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 RUN_ID SOURCE_SHA VERSION CANDIDATE_RUN_ID CRATE_SHA256 BINARY_SHA256 OUTPUT_DIRECTORY" >&2
  exit 2
fi

run_id=$1
source_sha=$2
version=$3
candidate_run_id=$4
crate_sha256=$5
binary_sha256=$6
output_directory=$7
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

fail() {
  echo "cache lifecycle candidate verification: $*" >&2
  exit 1
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

for command in awk find gh grep jq shasum tr wc; do
  command -v "$command" >/dev/null || fail "missing required command: $command"
done
for id in "$run_id" "$candidate_run_id"; do
  [[ "$id" =~ ^[1-9][0-9]*$ ]] || fail "workflow run ID is not canonical"
done
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]] || fail "source SHA is not canonical"
[[ "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]] || \
  fail "version is not canonical stable SemVer"
for digest in "$crate_sha256" "$binary_sha256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || fail "artifact SHA-256 is not canonical"
done
[[ "$output_directory" == /* && ! -e "$output_directory" ]] || \
  fail "output directory must be a new absolute path"
: "${GH_TOKEN:?GH_TOKEN is required}"

run_json=$(gh run view "$run_id" --json conclusion,event,workflowName,url)
[[ $(jq -r .event <<<"$run_json") == workflow_dispatch ]] || \
  fail "qualification was not explicitly dispatched"
if [[ ${GITHUB_RUN_ID:-} == "$run_id" ]]; then
  [[ $(jq -r .workflowName <<<"$run_json") == Release ]] || \
    fail "in-progress qualification is not part of the current Release workflow"
else
  [[ $(jq -r .workflowName <<<"$run_json") == "Cache lifecycle" ]] || \
    fail "qualification is not a Cache lifecycle workflow"
  [[ $(jq -r .conclusion <<<"$run_json") == success ]] || \
    fail "qualification workflow did not succeed"
fi

mkdir -m 0700 "$output_directory"
gh run download "$run_id" --name "cache-lifecycle-$source_sha" \
  --dir "$output_directory"
unset GH_TOKEN GITHUB_TOKEN

[[ -z $(find "$output_directory" -type l -print -quit) ]] || \
  fail "qualification artifact contains a symbolic link"
manifest="$output_directory/manifest.json"
checksum="$output_directory/manifest.json.sha256"
[[ -s "$manifest" && -s "$checksum" && ! -L "$manifest" && ! -L "$checksum" ]] || \
  fail "qualification manifest or checksum is missing"
[[ $(wc -l <"$checksum" | tr -d ' ') == 1 ]] || fail "manifest checksum has extra records"
recorded_sha=$(awk 'NR == 1 {print $1}' "$checksum")
[[ "$recorded_sha" == "$(sha256_file "$manifest")" ]] || fail "manifest checksum differs"

jq -e --arg source_sha "$source_sha" --arg version "$version" \
  --arg candidate_run_id "$candidate_run_id" \
  --arg crate_sha256 "$crate_sha256" --arg binary_sha256 "$binary_sha256" '
    .kind == "hf2q.cache-lifecycle-release-manifest"
    and .schema_version == 2
    and .status == "pass"
    and .source_sha == $source_sha
    and .version == $version
    and .standalone_candidate_run_id == $candidate_run_id
    and .crate_sha256 == $crate_sha256
    and .binary_sha256 == $binary_sha256
    and .power_policy.guarded == true
    and .power_policy.cable_required == false
    and (.power_policy.initial_source == "ac"
      or .power_policy.initial_source == "battery")
    and (.power_policy.final_source == "ac"
      or .power_policy.final_source == "battery")
    and ((.power_policy.mode.name == "high"
        and .power_policy.mode.numeric_canary == 2)
      or (.power_policy.mode.name == "automatic"
        and .power_policy.mode.numeric_canary == 0
        and .power_policy.initial_source == "ac"
        and .power_policy.final_source == "ac"))
    and (.power_guarded_ac == (.power_policy.initial_source == "ac"
      and .power_policy.final_source == "ac"))
    and .models.deepseek.architecture == "deepseek4"
    and .models.gemma.architecture == "gemma4"
    and .models.qwen.architecture == "qwen35moe"
    and .models.qwen38.architecture == "qwen35"
    and all(.models.deepseek,.models.gemma,.models.qwen,.models.qwen38;
      (.sha256 | test("^[0-9a-f]{64}$"))
      and (.model_info_sha256 | test("^[0-9a-f]{64}$"))
      and (.bytes | type == "number" and . > 0))
    and all(.families.deepseek,.families.gemma,.families.qwen,.families.qwen38;
      .status == "pass" and .structured_outputs.status == "pass")
  ' "$manifest" >/dev/null || fail "qualification manifest identity is invalid"

for family in deepseek gemma qwen qwen38; do
  receipt="$output_directory/$family/r2c-structured"
  model_info="$output_directory/$family/model-info.txt"
  model_sha256=$(jq -er --arg family "$family" '.models[$family].sha256' "$manifest")
  expected_architecture=$(jq -er --arg family "$family" '.models[$family].architecture' "$manifest")
  expected_model_info_sha=$(jq -er --arg family "$family" \
    '.models[$family].model_info_sha256' "$manifest")
  [[ -s "$model_info" && ! -L "$model_info" ]] || fail "$family model info is missing"
  [[ "$(sha256_file "$model_info")" == "$expected_model_info_sha" ]] || \
    fail "$family model info hash differs"
  [[ $(grep -Ec '^Architecture: ' "$model_info" || true) == 1 ]] || \
    fail "$family model info architecture is ambiguous"
  actual_architecture=$(awk '/^Architecture: / {print $2}' "$model_info")
  [[ "$actual_architecture" == "$expected_architecture" ]] || \
    fail "$family model info architecture differs"
  "$script_dir/verify_r2c_structured_output_receipt.sh" "$receipt" "$family" \
    "$source_sha" "$version" "$crate_sha256" "$binary_sha256" "$model_sha256" \
    >/dev/null || fail "$family structured-output receipt is invalid"
  receipt_sha=$(sha256_file "$receipt/summary.json")
  jq -e --arg family "$family" --arg receipt_sha "$receipt_sha" \
    --slurpfile receipt "$receipt/summary.json" '
      .receipt_sha256[$family].structured_outputs == $receipt_sha
      and .families[$family].structured_outputs == $receipt[0]
    ' "$manifest" >/dev/null || fail "$family receipt is not bound into the manifest"
done

echo "cache lifecycle candidate verification: PASS"
