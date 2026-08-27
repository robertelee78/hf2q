#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt=${1:?receipt required}
source_root=${2:?source root required}
[[ "$receipt" == /* && -f "$receipt" ]] || exit 2
root=$(cd "$(dirname "$receipt")" && pwd -P)
tmp_dir=$(mktemp -d "${TMPDIR:-/var/tmp}/qwen-lifecycle-mutation.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT
"$script_dir/verify_qwen35_agentic_lifecycle_cell.sh" \
    "$receipt" "$source_root" >/dev/null

copy="$tmp_dir/q5-policy-flipped"
cp -R "$root" "$copy"
jq '.runtime.routing.dense_q5k_canonical_q4x4 |= not' \
    "$copy/receipt.json" >"$copy/receipt.json.tmp"
mv "$copy/receipt.json.tmp" "$copy/receipt.json"
if "$script_dir/verify_qwen35_agentic_lifecycle_cell.sh" \
    "$copy/receipt.json" "$source_root" >/dev/null 2>&1; then
    echo "lifecycle verifier accepted a routing policy contradicted by the runtime log" >&2
    exit 1
fi

copy="$tmp_dir/q5-policy-numeric"
cp -R "$root" "$copy"
jq '.runtime.routing.dense_q5k_canonical_q4x4 = 1' \
    "$copy/receipt.json" >"$copy/receipt.json.tmp"
mv "$copy/receipt.json.tmp" "$copy/receipt.json"
if "$script_dir/verify_qwen35_agentic_lifecycle_cell.sh" \
    "$copy/receipt.json" "$source_root" >/dev/null 2>&1; then
    echo "lifecycle verifier accepted a numeric routing-policy surrogate" >&2
    exit 1
fi

copy="$tmp_dir/rebound-summary"
cp -R "$root" "$copy"
jq '.queued_exact_retry_cached_tokens += 1000000' \
    "$copy/lifecycle/summary.json" >"$copy/lifecycle/summary.json.tmp"
mv "$copy/lifecycle/summary.json.tmp" "$copy/lifecycle/summary.json"
summary_sha=$(shasum -a 256 "$copy/lifecycle/summary.json" | awk '{print $1}')
SUMMARY_SHA="$summary_sha" perl -i -pe '
  if (/  lifecycle\/summary[.]json$/) {
    s/^[0-9a-f]{64}/$ENV{SUMMARY_SHA}/;
    $seen++;
  }
  END {die "lifecycle summary absent from manifest\n" unless $seen == 1}
' "$copy/evidence.sha256"
manifest_sha=$(shasum -a 256 "$copy/evidence.sha256" | awk '{print $1}')
jq --arg summary "$summary_sha" --arg manifest "$manifest_sha" '
  .lifecycle.summary_sha256 = $summary
  | .evidence.manifest_sha256 = $manifest
' "$copy/receipt.json" >"$copy/receipt.json.tmp"
mv "$copy/receipt.json.tmp" "$copy/receipt.json"
if "$script_dir/verify_qwen35_agentic_lifecycle_cell.sh" \
    "$copy/receipt.json" "$source_root" >/dev/null 2>&1; then
    echo "lifecycle verifier trusted a fully rebound summary over raw responses" >&2
    exit 1
fi

expect_rebound_request_mutation_rejected() {
    local label=$1
    local phase=$2
    local filter=$3
    local mutated="$tmp_dir/$label"
    local relative="lifecycle/$phase.request.json"
    local request_sha manifest_sha
    cp -R "$root" "$mutated"
    jq "$filter" "$mutated/$relative" >"$mutated/$relative.tmp"
    mv "$mutated/$relative.tmp" "$mutated/$relative"
    request_sha=$(shasum -a 256 "$mutated/$relative" | awk '{print $1}')
    RELATIVE="$relative" REQUEST_SHA="$request_sha" perl -i -pe '
      if (/  \Q$ENV{RELATIVE}\E$/) {
        s/^[0-9a-f]{64}/$ENV{REQUEST_SHA}/;
        $seen++;
      }
      END {die "request absent from manifest\n" unless $seen == 1}
    ' "$mutated/evidence.sha256"
    manifest_sha=$(shasum -a 256 "$mutated/evidence.sha256" | awk '{print $1}')
    jq --arg manifest "$manifest_sha" '.evidence.manifest_sha256 = $manifest' \
        "$mutated/receipt.json" >"$mutated/receipt.json.tmp"
    mv "$mutated/receipt.json.tmp" "$mutated/receipt.json"
    if "$script_dir/verify_qwen35_agentic_lifecycle_cell.sh" \
        "$mutated/receipt.json" "$source_root" >/dev/null 2>&1; then
        echo "lifecycle verifier accepted rebound request mutation: $label" >&2
        exit 1
    fi
    rm -rf "$mutated"
}

budget_mutations=0
for phase in seed active sibling; do
    expect_rebound_request_mutation_rejected \
        "$phase-budget-deleted" "$phase" 'del(.thinking_token_budget)'
    expect_rebound_request_mutation_rejected \
        "$phase-budget-zero" "$phase" '.thinking_token_budget = 0'
    expect_rebound_request_mutation_rejected \
        "$phase-budget-altered" "$phase" '.thinking_token_budget = 17'
    budget_mutations=$((budget_mutations + 3))
done
[[ "$budget_mutations" == 9 ]]
expect_rebound_request_mutation_rejected \
    base-budget-added base '.thinking_token_budget = 16'
expect_rebound_request_mutation_rejected \
    isolation-disable-deleted isolation 'del(.hf2q_enable_thinking)'
expect_rebound_request_mutation_rejected \
    isolation-disable-flipped isolation '.hf2q_enable_thinking = true'
expect_rebound_request_mutation_rejected \
    isolation-budget-added isolation '.thinking_token_budget = 16'
expect_rebound_request_mutation_rejected \
    isolation-template-override isolation \
    '.chat_template_kwargs = {enable_thinking:true}'

expect_rebound_response_mutation_rejected() {
    local label=$1
    local filter=$2
    local mutated="$tmp_dir/$label"
    local relative=lifecycle/isolation.response.json
    local response_sha manifest_sha
    cp -R "$root" "$mutated"
    jq "$filter" "$mutated/$relative" >"$mutated/$relative.tmp"
    mv "$mutated/$relative.tmp" "$mutated/$relative"
    response_sha=$(shasum -a 256 "$mutated/$relative" | awk '{print $1}')
    RELATIVE="$relative" RESPONSE_SHA="$response_sha" perl -i -pe '
      if (/  \Q$ENV{RELATIVE}\E$/) {
        s/^[0-9a-f]{64}/$ENV{RESPONSE_SHA}/;
        $seen++;
      }
      END {die "response absent from manifest\n" unless $seen == 1}
    ' "$mutated/evidence.sha256"
    manifest_sha=$(shasum -a 256 "$mutated/evidence.sha256" | awk '{print $1}')
    jq --arg manifest "$manifest_sha" '.evidence.manifest_sha256 = $manifest' \
        "$mutated/receipt.json" >"$mutated/receipt.json.tmp"
    mv "$mutated/receipt.json.tmp" "$mutated/receipt.json"
    if "$script_dir/verify_qwen35_agentic_lifecycle_cell.sh" \
        "$mutated/receipt.json" "$source_root" >/dev/null 2>&1; then
        echo "lifecycle verifier accepted rebound response mutation: $label" >&2
        exit 1
    fi
    rm -rf "$mutated"
}

expect_rebound_response_mutation_rejected \
    isolation-reasoning-added \
    '.choices[0].message.reasoning_content = "private trace"
      | .usage.completion_tokens_details.reasoning_tokens = 1'
expect_rebound_response_mutation_rejected \
    isolation-length-finish '.choices[0].finish_reason = "length"'

echo "Qwen agentic lifecycle receipt mutations: summary + 18/18 mutations REJECTED"
