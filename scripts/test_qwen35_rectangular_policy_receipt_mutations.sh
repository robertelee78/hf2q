#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt=${1:?usage: test_qwen35_rectangular_policy_receipt_mutations.sh RECEIPT [SOURCE_ROOT]}
SOURCE_ROOT=${2:-}
[[ "$receipt" == /* && -f "$receipt" ]] || {
    echo "mutation test requires an absolute real receipt" >&2
    exit 2
}
evidence_root=$(cd "$(dirname "$receipt")" && pwd -P)
tmp_dir=$(mktemp -d "${TMPDIR:-/var/tmp}/qwen-rectangular-mutations.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

verify() {
    "$script_dir/verify_qwen35_rectangular_policy_receipt.sh" "$1" "$SOURCE_ROOT"
}

verify "$receipt" >/dev/null

mutate_json_and_reject() {
    local name=$1 filter=$2
    local copy="$tmp_dir/$name"
    cp -R "$evidence_root" "$copy"
    jq "$filter" "$copy/receipt.json" >"$copy/receipt.json.tmp"
    mv "$copy/receipt.json.tmp" "$copy/receipt.json"
    if verify "$copy/receipt.json" >/dev/null 2>&1; then
        echo "receipt verifier accepted injected mutation: $name" >&2
        exit 1
    fi
}

mutate_json_and_reject verdict-skip '.verdict = "skip"'
mutate_json_and_reject false-speedup '.result.wave_speedup = 99'
mutate_json_and_reject false-tail '.result.single_max_matched_overhead_ms = -999'
mutate_json_and_reject summary-hash \
    '.evidence.processes["on-a"].summary_sha256 = ("0" * 64)'
mutate_json_and_reject semantic-hash \
    '.equality.semantic_and_token_sha256 = ("f" * 64)'
mutate_json_and_reject contention-policy \
    '.environment.host_contention.policy = "process-group-cpu-v1"'
mutate_json_and_reject contention-threshold \
    '.environment.host_contention.maximum_foreign_cpu_percent = 999'
mutate_json_and_reject contention-owner \
    '.environment.host_contention.owner_pgid += 1'
mutate_json_and_reject contention-continuous \
    '.environment.host_contention.continuous = false'

owner_copy="$tmp_dir/rebound-contention-owner"
cp -R "$evidence_root" "$owner_copy"
awk -F '\t' 'BEGIN { OFS = "\t" } NR == 2 { $4 += 1 } { print }' \
    "$owner_copy/on-a/contention-measurement.tsv" \
    >"$owner_copy/on-a/contention-measurement.tsv.tmp"
mv "$owner_copy/on-a/contention-measurement.tsv.tmp" \
    "$owner_copy/on-a/contention-measurement.tsv"
owner_raw_sha=$(shasum -a 256 \
    "$owner_copy/on-a/contention-measurement.tsv" | awk '{print $1}')
REBOUND_RAW_SHA="$owner_raw_sha" perl -i -pe '
  if (/  contention-measurement[.]tsv$/) {
    s/^[0-9a-f]{64}/$ENV{REBOUND_RAW_SHA}/;
    $seen++;
  }
  END {die "contention measurement absent from manifest\n" unless $seen == 1}
' "$owner_copy/on-a/evidence.sha256"
owner_manifest_sha=$(shasum -a 256 "$owner_copy/on-a/evidence.sha256" \
    | awk '{print $1}')
jq --arg raw "$owner_raw_sha" --arg manifest "$owner_manifest_sha" '
  .environment.contention_measurement_sha256 = $raw
  | .evidence_manifest_sha256 = $manifest
' "$owner_copy/on-a/summary.json" >"$owner_copy/on-a/summary.json.tmp"
mv "$owner_copy/on-a/summary.json.tmp" "$owner_copy/on-a/summary.json"
owner_summary_sha=$(shasum -a 256 "$owner_copy/on-a/summary.json" \
    | awk '{print $1}')
jq --arg manifest "$owner_manifest_sha" --arg summary "$owner_summary_sha" '
  .evidence.processes["on-a"].manifest_sha256 = $manifest
  | .evidence.processes["on-a"].summary_sha256 = $summary
' "$owner_copy/receipt.json" >"$owner_copy/receipt.json.tmp"
mv "$owner_copy/receipt.json.tmp" "$owner_copy/receipt.json"
if verify "$owner_copy/receipt.json" >/dev/null 2>&1; then
    echo "receipt verifier trusted a rebound raw contention owner" >&2
    exit 1
fi

raw_copy="$tmp_dir/raw-response"
cp -R "$evidence_root" "$raw_copy"
printf '\n' >>"$raw_copy/on-b/responses/wave-3-2.json"
if verify "$raw_copy/receipt.json" >/dev/null 2>&1; then
    echo "receipt verifier accepted tampered raw response" >&2
    exit 1
fi

rebound_copy="$tmp_dir/rebound-raw-wave"
cp -R "$evidence_root" "$rebound_copy"
awk 'NR == 1 {$1 += 0.125} {printf "%.6f\n", $1}' \
    "$rebound_copy/on-a/wave-wall-seconds" \
    >"$rebound_copy/on-a/wave-wall-seconds.tmp"
mv "$rebound_copy/on-a/wave-wall-seconds.tmp" \
    "$rebound_copy/on-a/wave-wall-seconds"
rebound_raw_sha=$(shasum -a 256 "$rebound_copy/on-a/wave-wall-seconds" \
    | awk '{print $1}')
REBOUND_RAW_SHA="$rebound_raw_sha" perl -i -pe '
  if (/  wave-wall-seconds$/) {
    s/^[0-9a-f]{64}/$ENV{REBOUND_RAW_SHA}/;
    $seen++;
  }
  END {die "wave-wall-seconds absent from manifest\n" unless $seen == 1}
' "$rebound_copy/on-a/evidence.sha256"
rebound_manifest_sha=$(shasum -a 256 "$rebound_copy/on-a/evidence.sha256" \
    | awk '{print $1}')
jq --arg sha "$rebound_manifest_sha" '.evidence_manifest_sha256 = $sha' \
    "$rebound_copy/on-a/summary.json" >"$rebound_copy/on-a/summary.json.tmp"
mv "$rebound_copy/on-a/summary.json.tmp" "$rebound_copy/on-a/summary.json"
rebound_summary_sha=$(shasum -a 256 "$rebound_copy/on-a/summary.json" \
    | awk '{print $1}')
jq --arg manifest "$rebound_manifest_sha" --arg summary "$rebound_summary_sha" '
  .evidence.processes["on-a"].manifest_sha256 = $manifest
  | .evidence.processes["on-a"].summary_sha256 = $summary
' "$rebound_copy/receipt.json" >"$rebound_copy/receipt.json.tmp"
mv "$rebound_copy/receipt.json.tmp" "$rebound_copy/receipt.json"
if verify "$rebound_copy/receipt.json" >/dev/null 2>&1; then
    echo "receipt verifier trusted a rebound summary over changed raw timings" >&2
    exit 1
fi

metric_copy="$tmp_dir/rebound-raw-metric"
cp -R "$evidence_root" "$metric_copy"
awk '$1 == "hf2q_qwen_rectangular_prefill_cohorts_total" {
       $2 += 1; seen++
     }
     {print}
     END {if (seen != 1) exit 1}' \
    "$metric_copy/on-a/waves/3.metrics-after" \
    >"$metric_copy/on-a/waves/3.metrics-after.tmp"
mv "$metric_copy/on-a/waves/3.metrics-after.tmp" \
    "$metric_copy/on-a/waves/3.metrics-after"
metric_raw_sha=$(shasum -a 256 "$metric_copy/on-a/waves/3.metrics-after" \
    | awk '{print $1}')
METRIC_RAW_SHA="$metric_raw_sha" perl -i -pe '
  if (/  waves\/3[.]metrics-after$/) {
    s/^[0-9a-f]{64}/$ENV{METRIC_RAW_SHA}/;
    $seen++;
  }
  END {die "wave metric absent from manifest\n" unless $seen == 1}
' "$metric_copy/on-a/evidence.sha256"
metric_manifest_sha=$(shasum -a 256 "$metric_copy/on-a/evidence.sha256" \
    | awk '{print $1}')
jq --arg sha "$metric_manifest_sha" '.evidence_manifest_sha256 = $sha' \
    "$metric_copy/on-a/summary.json" >"$metric_copy/on-a/summary.json.tmp"
mv "$metric_copy/on-a/summary.json.tmp" "$metric_copy/on-a/summary.json"
metric_summary_sha=$(shasum -a 256 "$metric_copy/on-a/summary.json" \
    | awk '{print $1}')
jq --arg manifest "$metric_manifest_sha" --arg summary "$metric_summary_sha" '
  .evidence.processes["on-a"].manifest_sha256 = $manifest
  | .evidence.processes["on-a"].summary_sha256 = $summary
' "$metric_copy/receipt.json" >"$metric_copy/receipt.json.tmp"
mv "$metric_copy/receipt.json.tmp" "$metric_copy/receipt.json"
if verify "$metric_copy/receipt.json" >/dev/null 2>&1; then
    echo "receipt verifier trusted rebound wave JSON over changed raw metrics" >&2
    exit 1
fi

echo "Qwen rectangular policy receipt mutations: 13/13 REJECTED"
