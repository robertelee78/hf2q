#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/qwen35_mixed_rectangular_contract.sh
source "$script_dir/qwen35_mixed_rectangular_contract.sh"
tmp=$(mktemp -d "${TMPDIR:-/var/tmp}/qwen-mixed-contract.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

"$script_dir/verify_qwen35_mixed_rectangular_receipt.sh" --self-test >/dev/null

cat >"$tmp/frames.tsv" <<'EOF'
10.000000000	{"id":"x","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}
10.100000000	{"id":"x","object":"chat.completion.chunk","choices":[{"delta":{"content":"alpha"},"finish_reason":null}]}
10.200000000	{"id":"x","object":"chat.completion.chunk","choices":[{"delta":{"reasoning_content":"beta"},"finish_reason":null}]}
10.300000000	{"id":"x","object":"chat.completion.chunk","choices":[{"delta":{"content":"omega"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":3,"total_tokens":11,"prompt_tokens_details":{"cached_tokens":0}}}
10.400000000	[DONE]
EOF
[[ "$(qwen35_mixed_semantic_frame_count "$tmp/frames.tsv")" == 3 ]]
qwen35_mixed_semantic_trace_json "$tmp/frames.tsv" 9.900 10.150 10.250 \
  >"$tmp/trace.json"
qwen35_mixed_validate_semantic_trace "$tmp/trace.json" 3 250 101

if qwen35_mixed_semantic_trace_json "$tmp/frames.tsv" 9.900 10.050 10.250 \
  >"$tmp/no-before.json" 2>/dev/null \
  && qwen35_mixed_validate_semantic_trace "$tmp/no-before.json" 3 250 101; then
  echo "mixed contract accepted no semantic event before the wave" >&2
  exit 1
fi
sed '$d' "$tmp/frames.tsv" >"$tmp/no-done.tsv"
if qwen35_mixed_semantic_trace_json "$tmp/no-done.tsv" 9.900 10.150 10.250 \
  >/dev/null 2>&1; then
  echo "mixed contract accepted a stream without [DONE]" >&2
  exit 1
fi
awk -F '\t' 'NR == 3 {$1=9.0} {printf "%.9f\t%s\n", $1, $2}' "$tmp/frames.tsv" \
  >"$tmp/nonmonotonic.tsv"
if qwen35_mixed_semantic_trace_json "$tmp/nonmonotonic.tsv" 8.900 9.500 10.250 \
  >/dev/null 2>&1; then
  echo "mixed contract accepted a non-monotonic semantic clock" >&2
  exit 1
fi
awk -F '\t' 'NR == 3 {$1="malformed"} {print $1 "\t" $2}' "$tmp/frames.tsv" \
  >"$tmp/malformed.tsv"
if qwen35_mixed_semantic_trace_json "$tmp/malformed.tsv" 9.900 10.150 10.250 \
  >/dev/null 2>&1; then
  echo "mixed contract silently dropped a malformed timestamp row" >&2
  exit 1
fi

cat >"$tmp/events.jsonl" <<'EOF'
{"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":null}]}
{"choices":[{"delta":{"content":"alpha"},"finish_reason":null}]}
{"choices":[{"delta":{"reasoning_content":"beta"},"finish_reason":null}]}
{"choices":[{"delta":{"content":"omega"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":3,"total_tokens":11,"prompt_tokens_details":{"cached_tokens":0}}}
EOF
qwen35_mixed_canonical_sse_json "$tmp/events.jsonl" >"$tmp/canonical.json"
jq -e '
  .message == {role:"assistant",content:"alphaomega",
    reasoning_content:"beta",tool_calls:[],refusal:null}
  and .finish_reason == "stop"
  and .usage == {prompt_tokens:8,completion_tokens:3,total_tokens:11,
    prompt_tokens_details:{cached_tokens:0}}
' "$tmp/canonical.json" >/dev/null

qwen35_mixed_validate_publication \
  'Qwen rectangular prefill published lanes=4 rows_per_lane=96 aggregate_rows=384 mtp_prefill=false checkpoint_at_end=true mtp_outcome=NotRequested' \
  not-requested
if qwen35_mixed_validate_publication \
  'Qwen rectangular prefill published lanes=4 rows_per_lane=96 aggregate_rows=384 mtp_prefill=false checkpoint_at_end=true mtp_outcome=NotRequested' \
  succeeded; then
  echo "mixed contract accepted the Qwen3.6 no-MTP outcome for Qwen3.8" >&2
  exit 1
fi
if qwen35_mixed_validate_publication \
  'Qwen rectangular prefill published lanes=4 rows_per_lane=96 aggregate_rows=384 mtp_prefill=false checkpoint_at_end=false mtp_outcome=NotRequested' \
  not-requested; then
  echo "mixed contract accepted an unpublished stable-boundary checkpoint" >&2
  exit 1
fi

runner="$script_dir/bench_qwen35_mixed_rectangular_cell.sh"
matrix="$script_dir/bench_qwen35_rectangular_policy_matrix.sh"
mutation="$script_dir/test_qwen35_mixed_rectangular_receipt_mutations.sh"
rg -q '^readonly MAX_SLOTS=8$' "$runner"
rg -q '^readonly PREFILL_LANES=4$' "$runner"
rg -q '^readonly MIN_MIXED_SPEEDUP=1[.]01$' "$runner"
rg -q '^readonly MAX_DECODER_TTFT_MS=15000$' "$runner"
rg -q '^readonly MAX_SEMANTIC_GAP_MS=15000$' "$runner"
rg -q '^readonly MAX_PREFILL_TAIL_MS=60000$' "$runner"
[[ "$(rg -c 'bench_qwen35_mixed_rectangular_cell[.]sh' "$matrix")" == 2 ]]
[[ "$(rg -c 'test_qwen35_mixed_rectangular_receipt_mutations[.]sh' "$matrix")" == 2 ]]
[[ "$(rg -c 'run_qwen35_agentic_lifecycle_cell[.]sh' "$matrix")" == 2 ]]
rg -q 'MODEL_SHAPE=qwen38-dense' "$matrix"
rg -q 'MODEL_SHAPE=qwen36-moe' "$matrix"
rg -q 'Qwen mixed rectangular receipt mutations: 16/16 REJECTED' "$mutation"
rg -q 'HF2Q_QWEN_MIXED_GATE_ISOLATED=1' "$runner"
rg -q 'host_contention_require_isolated_gate_owner' "$runner"
rg -q 'HOST_CONTENTION_GATE_OWNER_PID' "$runner"
rg -q 'owner_scope == "release-gate-process-group"' \
  "$script_dir/verify_qwen35_mixed_rectangular_receipt.sh"
rg -q 'contention_log owner binding' \
  "$script_dir/verify_qwen35_mixed_rectangular_receipt.sh"
rg -q 'observed_source=[$][(]resolve_live_power_source[)]' "$runner"
if rg -Fq 'pmset -g batt | rg -q' "$runner"; then
  echo "mixed Qwen runner retains the early-match AC probe" >&2
  exit 1
fi

if env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  HF2Q_QWEN_MIXED_GATE_ISOLATED=1 \
  bash "$runner" >"$tmp/inherited-group.log" 2>&1; then
  echo "mixed Qwen runner accepted a forced inherited process group" >&2
  exit 1
fi
rg -q 'calibrated leaf does not own an isolated process group' \
  "$tmp/inherited-group.log"
if rg -q 'MODEL_PATH is required' "$tmp/inherited-group.log"; then
  echo "mixed Qwen runner reached model admission before ownership" >&2
  exit 1
fi
if env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  bash "$runner" >"$tmp/missing-env.log" 2>&1; then
  echo "mixed Qwen runner accepted a missing exact model contract" >&2
  exit 1
fi
rg -q 'MODEL_PATH is required' "$tmp/missing-env.log"

echo "Qwen mixed rectangular model-free contract: PASS"
