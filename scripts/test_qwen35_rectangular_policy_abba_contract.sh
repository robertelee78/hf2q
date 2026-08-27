#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner="$script_dir/bench_qwen35_rectangular_policy_abba.sh"
matrix_runner="$script_dir/bench_qwen35_rectangular_policy_matrix.sh"
verifier="$script_dir/verify_qwen35_rectangular_policy_receipt.sh"
mutation_test="$script_dir/test_qwen35_rectangular_policy_receipt_mutations.sh"
lifecycle_runner="$script_dir/run_qwen35_agentic_lifecycle_cell.sh"
lifecycle_verifier="$script_dir/verify_qwen35_agentic_lifecycle_cell.sh"
lifecycle_mutations="$script_dir/test_qwen35_agentic_lifecycle_receipt_mutations.sh"
lifecycle_client="$script_dir/test_agentic_cache_lifecycle.sh"

fail() {
    echo "rectangular policy ABBA contract: $*" >&2
    exit 1
}

require_text() {
    local text=$1 message=$2
    grep -Fq -- "$text" "$runner" || fail "$message"
}

[[ -f "$runner" && -x "$runner" ]] || fail "runner is absent or not executable"
[[ -f "$matrix_runner" && -x "$matrix_runner" ]] \
    || fail "matrix runner is absent or not executable"
[[ -f "$verifier" && -x "$verifier" ]] \
    || fail "independent receipt verifier is absent or not executable"
[[ -f "$mutation_test" && -x "$mutation_test" ]] \
    || fail "receipt mutation test is absent or not executable"
[[ -f "$lifecycle_runner" && -x "$lifecycle_runner" ]] \
    || fail "agentic lifecycle cell runner is absent or not executable"
[[ -f "$lifecycle_verifier" && -x "$lifecycle_verifier" ]] \
    || fail "agentic lifecycle cell verifier is absent or not executable"
[[ -f "$lifecycle_mutations" && -x "$lifecycle_mutations" ]] \
    || fail "agentic lifecycle mutation test is absent or not executable"
[[ -f "$lifecycle_client" && -x "$lifecycle_client" ]] \
    || fail "agentic lifecycle client is absent or not executable"
bash -n "$runner"
bash -n "$matrix_runner"
bash -n "$verifier"
bash -n "$mutation_test"
bash -n "$lifecycle_runner"
bash -n "$lifecycle_verifier"
bash -n "$lifecycle_mutations"
bash -n "$lifecycle_client"
"$verifier" --self-test
if command -v shellcheck >/dev/null 2>&1; then
    shellcheck -x "$runner"
    shellcheck -x "$matrix_runner" "$verifier" "$mutation_test" \
        "$lifecycle_runner" "$lifecycle_verifier" "$lifecycle_mutations" \
        "$lifecycle_client"
fi

require_text 'readonly TRIALS=5' 'trial count is not immutable'
require_text 'readonly MAX_SLOTS=4' 'width-four policy is not immutable'
require_text 'readonly COALESCE_US=25000' 'coalescing window is not immutable'
require_text 'readonly THERMAL_SETTLE_SECONDS=60' 'thermal settle is not immutable'
require_text 'readonly THERMAL_SAMPLE_SECONDS=2' 'thermal sampling is not immutable'
require_text 'readonly POWER_PROBE_ATTEMPTS=3' \
    'power probe acquisition retry count is not immutable'
require_text 'readonly MIN_WAVE_SPEEDUP=1.01' 'speed threshold is not immutable'
require_text 'readonly MAX_SINGLE_OVERHEAD_MS=50' \
    'single-user ceiling is not immutable'
# shellcheck disable=SC2016
require_text 'HF2Q_CROSS_SLOT_ADMIT="$cross_slot"' \
    'server policy is not bound per fresh process'
# shellcheck disable=SC2016
require_text 'HF2Q_ADMIT_COALESCE_US="$coalesce"' \
    'server coalescing window is not bound per fresh process'
require_text 'HF2Q_QWEN_SPECULATION=auto' 'AUTO speculation is not exercised'
# shellcheck disable=SC2016
require_text 'env -i HOME="$RUNTIME_HOME"' \
    'measured servers do not start from a scrubbed environment'
require_text 'kv_persist_enabled=(true|false)' \
    'resolved persistence-free serve plan is not checked'
# shellcheck disable=SC2016
require_text '--cache-dir "$engine_dir/runtime-cache"' \
    'server cache directory is not evidence-local and explicit'
require_text 'HF2Q_FFN_TERMINAL_K_BATCH=8' \
    'canonical Qwen FFN batching policy is not explicit'
! grep -Fq 'HF2Q_KV_PERSIST' "$runner" \
    || fail 'runner contains the inert generate-only HF2Q_KV_PERSIST setting'
! grep -Fq -- '--no-vision' "$runner" "$lifecycle_runner" \
    || fail 'runner contains the removed serve --no-vision option'
require_text 'hf2q_qwen_rectangular_prefill_cohorts_total' \
    'rectangular publication metric is not checked'
require_text 'Qwen rectangular prefill published' \
    'rectangular production event is not checked'
require_text 'checkpoint_at_end=true' \
    'stable-boundary checkpoint publication is not required'
require_text 'mtp_prefill=true checkpoint_at_end=true mtp_outcome=Succeeded' \
    'Qwen3.8 MTP success is not required'
require_text 'mtp_prefill=false checkpoint_at_end=true mtp_outcome=NotRequested' \
    'Qwen3.6 no-MTP capability is not required'
require_text 'skew <= 0.100 && latest < earliest' \
    'launch skew and actual overlap are not fail-closed'
# shellcheck disable=SC2016
require_text 'latest_start:$latest_start' \
    'wave evidence omits the independently derived latest start'
# shellcheck disable=SC2016
require_text 'earliest_finish:$earliest_finish' \
    'wave evidence omits the independently derived earliest finish'
require_text 'cached_tokens == 0' 'cold measured requests are not required'
# shellcheck disable=SC2016
require_text '>>"$engine_dir/single-wall-ms"' \
    'single-user ceiling does not observe end-to-end client wall time'
# shellcheck disable=SC2016
require_text '.single_median_wall_ms - $off[0].single_median_wall_ms' \
    'single-user diagnostics still use an internal post-admission clock'
require_text 'single_max_matched_overhead_ms' \
    'single-user tail overhead is not fail-closed'
# shellcheck disable=SC2016
require_text 'cmp -s "$OUT_DIR/off-$replica/$relative"' \
    'OFF/ON request byte identity is not required'
require_text 'semantic_and_token_sha256' \
    'canonical response equality is absent from the receipt'
require_text 'thermal_wait_for_nominal' 'nominal thermal settle is absent'
require_text 'thermal_monitor_fair_or_better_while_pid' \
    'continuous thermal monitoring is absent'
require_text 'host_contention_validate_measurement_log' \
    'host contention measurement is not validated'
require_text 'HF2Q_QWEN_RECTANGULAR_POLICY_GATE_ISOLATED=1' \
    'runner does not self-reexec through the isolated gate supervisor'
require_text 'host_contention_require_isolated_gate_owner' \
    'runner does not prove dedicated process-group ownership'
require_text 'HOST_CONTENTION_GATE_OWNER_PID' \
    'runner does not use one stable contention owner'
require_text 'qwen36_bind_server_process' 'runtime PID/binary/model binding is absent'
require_text 'qwen36_reject_fatal_log' 'fatal server logs are not rejected'
# shellcheck disable=SC2016
require_text 'actual_arch=$(jq -er' 'exact /v1/models architecture is not checked'
require_text 'speculation_policy=(Auto|Off)' 'resolved speculation policy is not checked'
require_text 'record_power_contract' 'AC/power continuity is not checked per process'
# shellcheck disable=SC2016
require_text 'observed_mode=$(resolve_ac_energy_mode)' \
    'AC Energy Mode acquisition is not bounded and explicit'
# shellcheck disable=SC2016
require_text 'observed_code=$(resolve_live_power_mode_code)' \
    'live power-mode acquisition is not bounded and explicit'
# shellcheck disable=SC2016
require_text 'observed_source=$(resolve_live_power_source)' \
    'live AC-source acquisition is not bounded and explicit'
! grep -Fq 'pmset -g batt | rg -q' "$runner" \
    || fail 'runner still uses a pipefail-sensitive early-match AC probe'
require_text 'qwen36_start_power_guard "$$"' \
    'long gate has no owned caffeinate assertion'
require_text 'thermal_cleanup_probe' 'owned thermal probe is not cleaned'
require_text 'wave_prompt_tokens' \
    'single-user samples are not joined to the eligible cohort shape'
require_text 'shasum -a 256 -c evidence.sha256' \
    'raw evidence manifest is not reopened'
# shellcheck disable=SC2016
require_text 'summary_sha256:$off_a_summary_sha' \
    'top receipt does not bind process summaries'
# shellcheck disable=SC2016
require_text 'manifest_sha256:$off_a_manifest_sha' \
    'top receipt does not bind process manifests'

cleanup_line=$(grep -n '^    cleanup$' "$runner" | head -1 | cut -d: -f1)
# shellcheck disable=SC2016
seal_line=$(grep -n 'cd "$engine_dir"' "$runner" | tail -1 | cut -d: -f1)
[[ "$cleanup_line" =~ ^[0-9]+$ && "$seal_line" =~ ^[0-9]+$ \
    && "$cleanup_line" -lt "$seal_line" ]] \
    || fail "process evidence is sealed before the server finishes shutdown"

[[ "$(grep -Ec '^run_process (off-a off|on-a on|on-b on|off-b off)$' "$runner")" == 4 ]] \
    || fail "process order is not exact OFF-A/ON-A/ON-B/OFF-B"
! grep -Fq 'HF2Q_QWEN_SPECULATION=off' "$runner" \
    || fail "runner silently disables speculation"
! grep -Eq 'MIN_WAVE_SPEEDUP=\$\{|MAX_SINGLE_OVERHEAD_MS=\$\{' "$runner" \
    || fail "acceptance thresholds are environment-weakenable"
grep -Fq 'MODEL_SHAPE=qwen38-dense' "$matrix_runner" \
    || fail "matrix omits the dense/MTP Qwen3.8 cell"
grep -Fq 'MODEL_SHAPE=qwen36-moe' "$matrix_runner" \
    || fail "matrix omits the MoE/no-MTP Qwen3.6 cell"
[[ "$(grep -Fc 'run_qwen35_agentic_lifecycle_cell.sh' "$matrix_runner")" == 2 ]] \
    || fail "matrix does not join both exact-artifact lifecycle cells"
grep -Fq 'test_agentic_cache_lifecycle.sh' "$lifecycle_runner" \
    || fail "lifecycle cell does not execute the canonical agentic fixture"
# shellcheck disable=SC2016
grep -Fq 'HF2Q_Q5K_CANONICAL_Q4X4="$Q5K_CANONICAL_Q4X4"' "$lifecycle_runner" \
    || fail "lifecycle server does not receive the explicit Q5_K routing policy"
# shellcheck disable=SC2016
grep -Fq -- '--argjson q5k_canonical_q4x4 "$expected_q5k_policy"' \
    "$lifecycle_runner" \
    || fail "lifecycle receipt does not serialize the Q5_K route as a boolean"
grep -Fq 'dense_q5k_canonical_q4x4=(true|false)' "$lifecycle_verifier" \
    || fail "lifecycle verifier does not reopen the frozen Q5_K routing policy"
grep -Fq 'CONTINUATION_THINKING_TOKEN_BUDGET=16' "$lifecycle_runner" \
    || fail "Qwen lifecycle continuation reasoning is not explicitly bounded"
grep -Fq 'all(.[1:4][]; .thinking_token_budget == 16)' "$lifecycle_verifier" \
    || fail "lifecycle verifier does not reopen every budgeted continuation"
grep -Fq '.[4].hf2q_enable_thinking == false' "$lifecycle_verifier" \
    || fail "lifecycle verifier does not reopen non-thinking isolation"
grep -Fq '18/18 mutations REJECTED' "$lifecycle_mutations" \
    || fail "lifecycle budget mutation battery is incomplete"
grep -Fq 'for command in awk curl date grep jq sed' "$lifecycle_client" \
    || fail "scrubbed lifecycle client depends on a non-system search tool"
! grep -Fq 'for command in awk curl date jq rg sed' "$lifecycle_client" \
    || fail "scrubbed lifecycle client still requires Homebrew rg"
grep -Fq 'agentic_lifecycle_validate_summary' "$lifecycle_verifier" \
    || fail "lifecycle cell receipt is not reopened independently"
grep -Fq 'fully rebound summary' "$lifecycle_mutations" \
    || fail "lifecycle raw-vs-summary mutation is absent"
grep -Fq 'rectangular matrix cells do not share one exact clean binary' \
    "$matrix_runner" || fail "matrix does not require one exact binary"
grep -Fq 'verify_qwen35_rectangular_policy_receipt.sh' "$runner" \
    || fail "runner does not reopen its final receipt independently"
grep -Fq 'single_max_matched_overhead_ms' "$verifier" \
    || fail "verifier does not recompute the tail ceiling"
grep -Fq 'raw/summary derivation' "$verifier" \
    || fail "verifier trusts process timing summaries"
grep -Fq 'canonical semantic equality' "$verifier" \
    || fail "verifier does not recompute canonical equality"
grep -Fq '13/13 REJECTED' "$mutation_test" \
    || fail "mutation battery cardinality drifted"
grep -Fq 'owner_scope == "release-gate-process-group"' "$verifier" \
    || fail "verifier does not bind the release-gate owner scope"
grep -Fq 'contention_log owner binding' "$verifier" \
    || fail "verifier does not join raw contention rows to the owner"

contract_tmp=$(mktemp -d "${TMPDIR:-/var/tmp}/qwen-rectangular-contract.XXXXXX")
trap 'rm -rf "$contract_tmp"' EXIT
inherited_group_log="$contract_tmp/inherited-group.log"
if env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin \
    HF2Q_QWEN_RECTANGULAR_POLICY_GATE_ISOLATED=1 \
    bash "$runner" >"$inherited_group_log" 2>&1; then
    fail "forced isolation sentinel accepted an inherited process group"
fi
grep -Fq 'calibrated leaf does not own an isolated process group' \
    "$inherited_group_log" \
    || fail "forced isolation sentinel did not fail at process-group ownership"
! grep -Fq 'MODEL_PATH is required' "$inherited_group_log" \
    || fail "forced isolation sentinel reached model admission before ownership"

missing_env_log="$contract_tmp/missing-env.log"
if env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin \
    bash "$runner" >"$missing_env_log" 2>&1; then
    fail "runner accepted a missing exact model contract"
fi
grep -Fq 'MODEL_PATH is required' "$missing_env_log" \
    || fail "runner did not fail at the missing exact model boundary"

echo "Qwen rectangular policy ABBA contract: PASS"
