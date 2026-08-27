#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt=${1:?usage: test_qwen35_mixed_rectangular_receipt_mutations.sh RECEIPT [SOURCE_ROOT]}
SOURCE_ROOT=${2:-}
[[ "$receipt" == /* && -f "$receipt" ]] || exit 2
root=$(cd "$(dirname "$receipt")" && pwd -P)
tmp=$(mktemp -d "${TMPDIR:-/var/tmp}/qwen-mixed-mutations.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
for command in awk cp jq mv perl shasum; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done

verify() {
  "$script_dir/verify_qwen35_mixed_rectangular_receipt.sh" "$1" "$SOURCE_ROOT"
}
verify "$receipt" >/dev/null

mutate_receipt_and_reject() {
  local name=$1 filter=$2
  local copy="$tmp/$name"
  cp -R "$root" "$copy"
  jq "$filter" "$copy/receipt.json" >"$copy/receipt.json.tmp"
  mv "$copy/receipt.json.tmp" "$copy/receipt.json"
  if verify "$copy/receipt.json" >/dev/null 2>&1; then
    echo "mixed verifier accepted receipt mutation: $name" >&2
    exit 1
  fi
}

rebind_process() {
  local copy=$1 label=$2 relative=$3 summary_filter=${4:-.}
  local process="$copy/$label" raw_sha manifest_sha summary_sha filter
  raw_sha=$(shasum -a 256 "$process/$relative" | awk '{print $1}')
  RAW_SHA="$raw_sha" RELATIVE="$relative" perl -i -pe '
    if (/  \Q$ENV{RELATIVE}\E$/) {s/^[0-9a-f]{64}/$ENV{RAW_SHA}/; $seen++}
    END {die "raw evidence row absent\n" unless $seen == 1}
  ' "$process/evidence.sha256"
  manifest_sha=$(shasum -a 256 "$process/evidence.sha256" | awk '{print $1}')
  filter="($summary_filter) | .evidence.manifest_sha256 = \$manifest"
  jq --arg manifest "$manifest_sha" --arg raw "$raw_sha" \
    "$filter" "$process/summary.json" \
    >"$process/summary.json.tmp"
  mv "$process/summary.json.tmp" "$process/summary.json"
  summary_sha=$(shasum -a 256 "$process/summary.json" | awk '{print $1}')
  jq --arg label "$label" --arg manifest "$manifest_sha" --arg summary "$summary_sha" '
    .evidence.processes[$label].manifest_sha256 = $manifest
    | .evidence.processes[$label].summary_sha256 = $summary
  ' "$copy/receipt.json" >"$copy/receipt.json.tmp"
  mv "$copy/receipt.json.tmp" "$copy/receipt.json"
}

mutate_receipt_and_reject verdict-skip '.verdict = "skip"'
mutate_receipt_and_reject false-speedup '.result.mixed_prefill_speedup = 99'
mutate_receipt_and_reject widened-gap '.thresholds.max_semantic_gap_ms = 999999'
mutate_receipt_and_reject false-canonical '.equality.canonical_sha256 = ("f" * 64)'
mutate_receipt_and_reject summary-hash \
  '.evidence.processes["on-a"].summary_sha256 = ("0" * 64)'
mutate_receipt_and_reject contention-policy \
  '.environment.host_contention.policy = "process-group-cpu-v1"'
mutate_receipt_and_reject contention-threshold \
  '.environment.host_contention.maximum_foreign_cpu_percent = 999'
mutate_receipt_and_reject contention-owner \
  '.environment.host_contention.owner_pgid += 1'
mutate_receipt_and_reject contention-continuous \
  '.environment.host_contention.continuous = false'

owner_copy="$tmp/rebound-contention-owner"
cp -R "$root" "$owner_copy"
awk -F '\t' 'BEGIN { OFS = "\t" } NR == 2 { $4 += 1 } { print }' \
  "$owner_copy/on-a/contention-measurement.tsv" \
  >"$owner_copy/on-a/contention-measurement.tsv.tmp"
mv "$owner_copy/on-a/contention-measurement.tsv.tmp" \
  "$owner_copy/on-a/contention-measurement.tsv"
# shellcheck disable=SC2016
rebind_process "$owner_copy" on-a contention-measurement.tsv \
  '.evidence.contention_measurement_sha256 = $raw'
if verify "$owner_copy/receipt.json" >/dev/null 2>&1; then
  echo "mixed verifier trusted a rebound raw contention owner" >&2
  exit 1
fi

raw_copy="$tmp/raw-sse"
cp -R "$root" "$raw_copy"
printf '\n' >>"$raw_copy/on-a/responses/decoder-3.sse"
if verify "$raw_copy/receipt.json" >/dev/null 2>&1; then
  echo "mixed verifier accepted a tampered raw SSE stream" >&2
  exit 1
fi

trace_copy="$tmp/rebound-semantic-trace"
cp -R "$root" "$trace_copy"
awk -F '\t' 'NR == 2 {$1 += 20} {print $1 "\t" $2}' \
  "$trace_copy/on-a/responses/decoder-2.frames.tsv" \
  >"$trace_copy/on-a/responses/decoder-2.frames.tsv.tmp"
mv "$trace_copy/on-a/responses/decoder-2.frames.tsv.tmp" \
  "$trace_copy/on-a/responses/decoder-2.frames.tsv"
rebind_process "$trace_copy" on-a responses/decoder-2.frames.tsv
if verify "$trace_copy/receipt.json" >/dev/null 2>&1; then
  echo "mixed verifier trusted rebound evidence over a corrupt semantic clock" >&2
  exit 1
fi

metric_copy="$tmp/rebound-off-metric"
cp -R "$root" "$metric_copy"
awk '$1 == "hf2q_qwen_rectangular_prefill_cohorts_total" {$2 += 1; seen++}
  {print} END {if (seen != 1) exit 1}' \
  "$metric_copy/off-a/waves/3.metrics-after" \
  >"$metric_copy/off-a/waves/3.metrics-after.tmp"
mv "$metric_copy/off-a/waves/3.metrics-after.tmp" \
  "$metric_copy/off-a/waves/3.metrics-after"
rebind_process "$metric_copy" off-a waves/3.metrics-after
if verify "$metric_copy/receipt.json" >/dev/null 2>&1; then
  echo "mixed verifier trusted rebound OFF metrics over a false cohort" >&2
  exit 1
fi

policy_copy="$tmp/rebound-policy"
cp -R "$root" "$policy_copy"
perl -i -pe '
  if (/cross_slot_admit=true/ && !$seen) {
    s/cross_slot_admit=true/cross_slot_admit=false/; $seen=1
  }
  END {die "policy event absent\n" unless $seen}
' \
  "$policy_copy/on-b/server.stderr"
rebind_process "$policy_copy" on-b server.stderr
if verify "$policy_copy/receipt.json" >/dev/null 2>&1; then
  echo "mixed verifier trusted rebound raw logs over policy drift" >&2
  exit 1
fi

power_copy="$tmp/rebound-power"
cp -R "$root" "$power_copy"
perl -i -pe 's/on-a-measurement-end/on-a-wrong-phase/' \
  "$power_copy/on-a/power.tsv"
power_sha=$(shasum -a 256 "$power_copy/on-a/power.tsv" | awk '{print $1}')
rebind_process "$power_copy" on-a power.tsv \
  ".evidence.power_sha256 = \"$power_sha\""
if verify "$power_copy/receipt.json" >/dev/null 2>&1; then
  echo "mixed verifier trusted rebound raw power evidence over a corrupt phase" >&2
  exit 1
fi

publication_copy="$tmp/rebound-publication"
cp -R "$root" "$publication_copy"
perl -i -pe 's/mtp_outcome=(Succeeded|NotRequested)/mtp_outcome=OrdinaryReplay/' \
  "$publication_copy/on-a/waves/1.log"
rebind_process "$publication_copy" on-a waves/1.log
if verify "$publication_copy/receipt.json" >/dev/null 2>&1; then
  echo "mixed verifier trusted rebound logs over the wrong MTP outcome" >&2
  exit 1
fi

echo "Qwen mixed rectangular receipt mutations: 16/16 REJECTED"
