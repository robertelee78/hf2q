#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
verifier="$root_dir/scripts/verify_deepseek4_mixed_policy_receipt.py"
runner="$root_dir/scripts/bench_deepseek4_mixed_policy_abba.sh"
mutations="$root_dir/scripts/test_deepseek4_mixed_policy_receipt_mutations.sh"
launcher="$root_dir/scripts/serve_deepseek4_opencode.sh"
engine="$root_dir/src/serve/api/engine.rs"
tmp_dir=$(mktemp -d "${TMPDIR:-/var/tmp}/deepseek-b1-contract.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

bash -n "$runner" "$mutations" "$0"

cat >"$tmp_dir/valid.timed-sse" <<'EOF'
1.000000000	data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}
2.000000000	data: {"choices":[{"delta":{"content":"READY"},"finish_reason":null}]}
3.000000000	data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}
4.000000000	data: [DONE]
EOF
python3 "$verifier" --canonicalize "$tmp_dir/valid.timed-sse" "$tmp_dir/valid.json"
jq -e '
  .schema == 1 and .role_events == 1 and .content == "READY"
  and .reasoning_content == "" and .tool_calls == []
  and .finish_reason == "stop" and .done_count == 1
  and .semantic_events == 1 and .semantic_max_gap_ms == 0
  and .usage == {prompt_tokens:2,completion_tokens:1,total_tokens:3}
' "$tmp_dir/valid.json" >/dev/null

# Execute the producer's exact request-id parser. A static source assertion did
# not catch a literal shell-continuation backslash inside the Perl program.
cat >"$tmp_dir/server.log" <<'EOF'
DeepSeek-V4 request started request_id=40 max_tokens=1
DeepSeek-V4 request started request_id=41 max_tokens=256
DeepSeek-V4 request started max_tokens=8 request_id=42
DeepSeek-V4 request started request_id=43 max_tokens=2560
DeepSeek-V4 request started request_id=45 max_tokens=10
unrelated request_id=44 max_tokens=256
EOF
sed -n '/^extract_request_ids() {/,/^}/p' "$runner" >"$tmp_dir/extract-request-ids.sh"
# shellcheck disable=SC1090,SC1091
source "$tmp_dir/extract-request-ids.sh"
decoder_ids=$(extract_request_ids "$tmp_dir/server.log" 256)
prefill_ids=$(extract_request_ids "$tmp_dir/server.log" 8)
prime_ids=$(extract_request_ids "$tmp_dir/server.log" 1)
[[ "$decoder_ids" == 41 && "$prefill_ids" == 42 && "$prime_ids" == 40 ]] || {
    echo "DeepSeek B.1 request-id parser drifted" >&2
    exit 1
}

expect_canonical_reject() {
    local name=$1
    if python3 "$verifier" --canonicalize "$tmp_dir/$name.timed-sse" \
        "$tmp_dir/$name.json" >/dev/null 2>&1; then
        echo "canonical SSE verifier accepted mutation: $name" >&2
        exit 1
    fi
}

head -n 3 "$tmp_dir/valid.timed-sse" >"$tmp_dir/missing-done.timed-sse"
expect_canonical_reject missing-done
perl -pe 'if ($. == 2) {s/^2[.]000000000/0.500000000/}' \
  "$tmp_dir/valid.timed-sse" >"$tmp_dir/backwards-time.timed-sse"
expect_canonical_reject backwards-time
awk 'NR==2 {$0="2.000000000\tdata: {not-json}"} {print}' \
  "$tmp_dir/valid.timed-sse" >"$tmp_dir/malformed-json.timed-sse"
expect_canonical_reject malformed-json
awk 'NR==2 {print "1.500000000\tdata: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}"} {print}' \
  "$tmp_dir/valid.timed-sse" >"$tmp_dir/duplicate-role.timed-sse"
expect_canonical_reject duplicate-role
awk 'NR==2 {$0="2.000000000\tdata: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0}]},\"finish_reason\":null}]}"} {print}' \
  "$tmp_dir/valid.timed-sse" >"$tmp_dir/tool-call.timed-sse"
expect_canonical_reject tool-call

: >"$tmp_dir/fake.gguf"
if HF2Q_DEEPSEEK_MIXED_COHORT=invalid MODEL="$tmp_dir/fake.gguf" HF2Q_BIN=/usr/bin/true \
    CHECK_ONLY=1 bash "$launcher" >"$tmp_dir/launcher.stdout" 2>"$tmp_dir/launcher.stderr"; then
    echo "canonical launcher accepted an invalid Mixed policy" >&2
    exit 1
fi
rg -q 'HF2Q_DEEPSEEK_MIXED_COHORT must be 0 or 1' "$tmp_dir/launcher.stderr"

rg -q 'const DEEPSEEK4_MIXED_COHORT_ENV: &str = "HF2Q_DEEPSEEK_MIXED_COHORT"' "$engine"
rg -q 'selection: "invalid-serial-fallback"' "$engine"
rg -q 'let mixed_cohort_policy = deepseek4_mixed_cohort_policy\(\);' "$engine"
rg -q 'mixed_cohort_selection = mixed_cohort_policy.selection' "$engine"
rg -Uq 'mixed_budget[.]max_cooperative_rows_per_lane,\n[[:space:]]+mixed_cohort,' "$engine"
rg -q 'export HF2Q_DEEPSEEK_MIXED_COHORT="\$MIXED_COHORT"' "$launcher"

rg -q 'readonly TRIALS=5' "$runner"
rg -q 'readonly MAX_SLOTS=8' "$runner"
rg -q 'readonly LIVE_DECODERS=4' "$runner"
rg -q 'readonly PREFILLERS=4' "$runner"
rg -q 'readonly MIXED_ROWS=128' "$runner"
rg -q 'readonly DECODER_PRIME_MAX_TOKENS=1' "$runner"
rg -q '^prime_decoder_sessions()' "$runner"
rg -Fq 'prime_decoder_sessions "$model" "$replica"' "$runner"
rg -Fq 'local replica=${label##*-}' "$runner"
rg -Fq 'run_wave "$arm" "$model" "$replica" "-$warmup"' "$runner"
rg -q 'bounded_mixed=true' "$runner"
[[ "$(rg -F -c 'DeepSeek-V4 cooperative prefill complete.*bounded_mixed=true' "$runner")" == 2 ]]
rg -q 'readonly MAX_PEAK_RSS_BYTES=124554051584' "$runner"
rg -Uq 'run_process off-a off\nrun_process on-a on\nrun_process on-b on\nrun_process off-b off' "$runner"
[[ "$(rg -F -c 'BEGIN{exit !(value>minimum)}' "$runner")" == 3 ]]
rg -q 'speedup > MIN_WAVE_SPEEDUP and all\(value > 1[.]0 for value in neighbors\)' "$verifier"
rg -q 'MAX_PEAK_RSS_BYTES = 116 \* 1024\*\*3' "$verifier"
rg -q '^def verify_decoder_prime' "$verifier"
rg -q 'decoder did not use its primed cache' "$verifier"
rg -Fq 'work_tokens == prompt_tokens - cached_tokens' "$verifier"
rg -Fq 'sorted(client_cached_tokens["decoder"]) == sorted(decoder_plan_cached_tokens)' "$verifier"
rg -q 'declared cold prefill reused cache state' "$verifier"
rg -Fq 'prompt_tokens <= work_tokens < 2 * prompt_tokens' "$verifier"
rg -Fq 'client_cached_tokens["prefill"] == [0] * PREFILLERS' "$verifier"
rg -q 'prefills were not admitted before the first live decoder completed' "$verifier"
rg -Fq 'row.get("bounded_mixed") in ("true", "false")' "$verifier"
rg -q 'observed_source=[$][(]matched_parse_live_power_source' "$runner"
rg -q 'bounded_mixed = max_cooperative_rows_per_lane.is_some()' "$engine"
rg -Fq 'host_contention:{policy:$contention_policy' "$runner"
rg -Fq 'HF2Q_DEEPSEEK_B1_GATE_ISOLATED=1' "$runner"
rg -Fq 'host_contention_require_isolated_gate_owner' "$runner"
rg -Fq 'owner_scope:"release-gate-process-group"' "$runner"
rg -Fq 'owner_pgid:$contention_owner_pgid,continuous:true' "$runner"
rg -Fq '"policy": "process-group-cpu-v2"' "$verifier"
rg -Fq 'row[3] == str(owner_pgid)' "$verifier"
rg -Fq 'rebound-contention-row-owner mutate_contention_raw_owner' "$mutations"
rg -Fq '23/23 (22 REJECTED, 1 ACCEPTED)' "$mutations"
if rg -Fq 'pmset -g batt | rg -q' "$runner"; then
    echo "DeepSeek B.1 runner retains the early-match AC probe" >&2
    exit 1
fi

inherited_group_log="$tmp_dir/inherited-group.log"
if env -i PATH=/usr/bin:/bin HF2Q_DEEPSEEK_B1_GATE_ISOLATED=1 \
    bash "$runner" >"$inherited_group_log" 2>&1; then
    echo "DeepSeek B.1 runner accepted a forced sentinel in an inherited process group" >&2
    exit 1
fi
rg -Fq 'calibrated leaf does not own an isolated process group' \
  "$inherited_group_log"

echo "DeepSeek-V4 Mixed policy model-free contract: PASS"
