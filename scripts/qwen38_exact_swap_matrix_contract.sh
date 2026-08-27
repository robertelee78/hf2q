#!/usr/bin/env bash

# Model-free authority for the exact Qwen3.8 A -> B -> A artifact matrix.
# The runtime runner sources this after qwen38_artifact_contract.sh.

qwen38_validate_exact_swap_manifest() {
    local manifest=${1:?matrix manifest is required}
    local format expected_file expected_bytes expected_sha expected_type
    local actual_file actual_bytes actual_sha actual_type

    [[ -f "$manifest" && -r "$manifest" && ! -L "$manifest" ]] || return 1
    jq -e \
      --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
      --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" '
      .schema == "hf2q.qwen38-exact-swap-matrix.v1"
      and .repository == $repository and .revision == $revision
      and .architecture == "qwen35" and .arch_family == "qwen35"
      and (.artifacts | type == "array" and length == 5)
      and (.artifacts | map(.format))
        == ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"]
      and ([.artifacts[].file] | unique | length) == 5
      and ([.artifacts[].sha256] | unique | length) == 5
      and all(.artifacts[];
        (.file | test("^gguf/[a-z0-9._-]+\\.gguf$"))
        and (.bytes | type == "number" and . > 0 and floor == .)
        and (.sha256 | test("^[0-9a-f]{64}$"))
        and (.gguf_file_type | type == "number" and . >= 0 and floor == .))
      and (.pairs | type == "array" and length == 5)
      and (.pairs == [
        {id:"bf16--q4_k_m",a:"BF16",b:"Q4_K_M"},
        {id:"q4_k_m--q5_k_m",a:"Q4_K_M",b:"Q5_K_M"},
        {id:"q5_k_m--q6_k",a:"Q5_K_M",b:"Q6_K"},
        {id:"q6_k--q8_0",a:"Q6_K",b:"Q8_0"},
        {id:"q8_0--bf16",a:"Q8_0",b:"BF16"}
      ])
      and ([.pairs[].a] | sort)
        == ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"]
      and ([.pairs[].b] | sort)
        == ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"]
      and all(.pairs[]; .a != .b)
      and .proof == {
        sequence:["A1","B","A2"],
        one_resident_generation:true,
        fresh_generation_per_activation:true,
        execution_receipt_join:true,
        bounded_residency:true,
        no_double_residency:true,
        exact_a_replay:true
      }
    ' "$manifest" >/dev/null || return 1

    for format in $(qwen38_artifact_formats); do
        IFS=$'\t' read -r _format expected_file expected_bytes expected_sha expected_type \
            <<<"$(qwen38_artifact_record "$format")" || return 1
        IFS=$'\t' read -r actual_file actual_bytes actual_sha actual_type \
            < <(jq -er --arg format "$format" '
              .artifacts[] | select(.format == $format)
              | [.file,.bytes,.sha256,.gguf_file_type] | @tsv
            ' "$manifest") || return 1
        [[ "$actual_file" == "$expected_file" \
          && "$actual_bytes" == "$expected_bytes" \
          && "$actual_sha" == "$expected_sha" \
          && "$actual_type" == "$expected_type" ]] || {
            echo "exact swap manifest artifact mismatch for $format" >&2
            return 1
        }
    done
}

qwen38_validate_exact_swap_cell() {
    local cell=$1 pair_id=$2 format_a=$3 format_b=$4 source_commit=$5
    local binary_sha=$6 binary_git_commit=$7 dependency_version=$8
    local dependency_source=$9 dependency_checksum=${10}
    local a_file a_bytes a_sha a_type b_file b_bytes b_sha b_type

    IFS=$'\t' read -r _format a_file a_bytes a_sha a_type \
        <<<"$(qwen38_artifact_record "$format_a")" || return 1
    IFS=$'\t' read -r _format b_file b_bytes b_sha b_type \
        <<<"$(qwen38_artifact_record "$format_b")" || return 1
    jq -e \
      --arg pair_id "$pair_id" --arg format_a "$format_a" \
      --arg format_b "$format_b" --arg source_commit "$source_commit" \
      --arg binary_sha "$binary_sha" --arg binary_git_commit "$binary_git_commit" \
      --arg dependency_version "$dependency_version" \
      --arg dependency_source "$dependency_source" \
      --arg dependency_checksum "$dependency_checksum" \
      --arg a_file "$a_file" --argjson a_bytes "$a_bytes" --arg a_sha "$a_sha" \
      --argjson a_type "$a_type" --arg b_file "$b_file" \
      --argjson b_bytes "$b_bytes" --arg b_sha "$b_sha" \
      --argjson b_type "$b_type" '
      .schema == 1 and .verdict == "pass"
      and .pair == {id:$pair_id,a:$format_a,b:$format_b}
      and .binding.source_commit == $source_commit
      and .binding.binary_sha256 == $binary_sha
      and .binding.binary_git_commit == $binary_git_commit
      and .binding.binary_git_commit == .binding.source_commit
      and .binding.dependency == {name:"mlx-native",version:$dependency_version,
        source:$dependency_source,checksum:$dependency_checksum}
      and .artifacts.a == {format:$format_a,file:$a_file,bytes:$a_bytes,
        sha256:$a_sha,gguf_file_type:$a_type}
      and .artifacts.b == {format:$format_b,file:$b_file,bytes:$b_bytes,
        sha256:$b_sha,gguf_file_type:$b_type}
      and .proof == {one_resident_every_phase:true,
        fresh_generation_per_activation:true,execution_receipts_joined:true,
        bounded_residency:true,no_double_residency:true,
        evicted_artifact_absent:true,exact_a_replay:true}
      and .pool_budget_bytes == ([$a_bytes,$b_bytes] | max)
      and .load_budget_seconds == 10
      and (.phases | map(.phase)) == ["A1","B","A2"]
      and .phases[0].format == $format_a
      and .phases[1].format == $format_b
      and .phases[2].format == $format_a
      and .phases[0].resident.bytes == $a_bytes
      and .phases[1].resident.bytes == $b_bytes
      and .phases[2].resident.bytes == $a_bytes
      and all(.phases[];
        (.resident.pool_key_sha256 | test("^[0-9a-f]{64}$"))
        and (.resident.generation | type == "number" and . > 0 and floor == .)
        and (.resident.engine_config_sha256 | test("^[0-9a-f]{64}$"))
        and .execution.pool_key_sha256 == .resident.pool_key_sha256
        and .execution.generation == .resident.generation
        and (.result_sha256 | test("^[0-9a-f]{64}$"))
        and .semantic.role == "assistant"
        and .semantic.content == "HF2Q_SWAP_OK"
        and .semantic.finish_reason == "stop"
        and (.semantic.completion_tokens | type == "number" and . > 0 and floor == .)
        and .semantic.cached_tokens == 0)
      and .phases[0].execution.artifact_sha256 == $a_sha
      and .phases[1].execution.artifact_sha256 == $b_sha
      and .phases[2].execution.artifact_sha256 == $a_sha
      and all(.phases[];
        .execution.arch_family == "qwen35"
        and .execution.architecture == "qwen35")
      and .phases[0].resident.pool_key_sha256
        == .phases[2].resident.pool_key_sha256
      and .phases[0].resident.pool_key_sha256
        != .phases[1].resident.pool_key_sha256
      and ([.phases[].resident.generation] | unique | length) == 3
      and .phases[0].result_sha256 == .phases[2].result_sha256
      and .phases[0].resident.engine_config_sha256
        == .phases[2].resident.engine_config_sha256
      and .transitions.a_to_b.load_seconds > 0
      and .transitions.a_to_b.load_seconds < .load_budget_seconds
      and .transitions.b_to_a.load_seconds > 0
      and .transitions.b_to_a.load_seconds < .load_budget_seconds
      and (((([.memory.a1.rss_bytes,.memory.b.rss_bytes] | max) / 10 | floor)) as $memory_margin
        | ([$memory_margin,2147483648] | max) as $peak_margin
        | .transitions.a_to_b.rss_bound_bytes
          == (([.memory.a1.rss_bytes,.memory.b.rss_bytes] | max) + $peak_margin)
        and .transitions.a_to_b.host_wired_bound_bytes
          == (([.memory.a1.host_wired_bytes,.memory.b.host_wired_bytes] | min)
            + .artifacts.b.bytes + $peak_margin))
      and (([((.memory.a1.rss_bytes / 10) | floor),2147483648] | max) as $reload_margin
        | .transitions.b_to_a.rss_bound_bytes
          == (([.memory.a1.rss_bytes,.memory.b.rss_bytes] | max) + $reload_margin)
        and .transitions.b_to_a.host_wired_bound_bytes
          == (([.memory.b.host_wired_bytes,.memory.a2.host_wired_bytes] | min)
            + .artifacts.a.bytes + $reload_margin)
        and .replay_bounds.rss_bytes == (.memory.a1.rss_bytes + $reload_margin)
        and .replay_bounds.host_wired_bytes
          == (([.memory[].host_wired_bytes] | min) + .artifacts.a.bytes + $reload_margin))
      and (([((.memory.a1.physical_footprint_bytes / 10) | floor),2147483648] | max) as $footprint_margin
        | .replay_bounds.physical_footprint_bytes
          == (.memory.a1.physical_footprint_bytes + $footprint_margin))
      and (([((.memory.a1.wired_bytes / 10) | floor),2147483648] | max) as $wired_margin
        | .replay_bounds.wired_bytes == (.memory.a1.wired_bytes + $wired_margin))
      and all(.transitions[];
        .peak_rss_bytes > 0 and .peak_host_wired_bytes > 0
        and .peak_rss_bytes <= .rss_bound_bytes
        and .peak_host_wired_bytes <= .host_wired_bound_bytes)
      and all(.memory[];
        .rss_bytes > 0 and .physical_footprint_bytes > 0
        and .wired_bytes >= 0 and .host_wired_bytes > 0)
      and .memory.a2.rss_bytes <= .replay_bounds.rss_bytes
      and .memory.a2.physical_footprint_bytes
        <= .replay_bounds.physical_footprint_bytes
      and .memory.a2.wired_bytes <= .replay_bounds.wired_bytes
      and .memory.a2.host_wired_bytes <= .replay_bounds.host_wired_bytes
      and (.storage.b | IN("file_backed","anonymous_accounted"))
      and .storage.a1_file_backed == true
      and .storage.a2_file_backed == true
    ' "$cell" >/dev/null
}

qwen38_validate_exact_swap_chain() {
    local chain=$1 source_commit=$2 binary_sha=$3 binary_git_commit=$4
    local dependency_version=$5 dependency_source=$6 dependency_checksum=$7
    local format expected_file expected_bytes expected_sha expected_type
    local actual_file actual_bytes actual_sha actual_type
    local chain_json

    chain_json=$(jq -c . "$chain") || return 1
    jq -e \
      --arg source_commit "$source_commit" --arg binary_sha "$binary_sha" \
      --arg binary_git_commit "$binary_git_commit" \
      --arg dependency_version "$dependency_version" \
      --arg dependency_source "$dependency_source" \
      --arg dependency_checksum "$dependency_checksum" '
      . as $root
      | .schema == 1 and .verdict == "pass"
      and .gate == "qwen38-exact-five-format-two-cycle"
      and .binding.source_commit == $source_commit
      and .binding.binary_sha256 == $binary_sha
      and .binding.binary_git_commit == $binary_git_commit
      and .binding.binary_git_commit == .binding.source_commit
      and .binding.dependency == {name:"mlx-native",version:$dependency_version,
        source:$dependency_source,checksum:$dependency_checksum}
      and (.artifacts | type == "array" and length == 5)
      and (.artifacts | map(.format))
        == ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"]
      and .pool_budget_bytes == ([.artifacts[].bytes] | max)
      and .load_budget_seconds == 10
      and (.process.pid | type == "number" and floor == . and . > 0)
      and .sequence == ["BF16","Q4_K_M","BF16","Q5_K_M","BF16","Q6_K",
        "BF16","Q8_0","BF16","Q4_K_M","BF16","Q5_K_M","BF16","Q6_K",
        "BF16","Q8_0","BF16"]
      and .proof == {one_long_lived_process:true,two_complete_cycles:true,
        fresh_generation_every_activation:true,cold_generation_cache:true,
        execution_receipts_joined:true,bounded_every_transition:true,
        evicted_artifacts_absent:true,exact_bf16_replay:true}
      and (.phases | type == "array" and length == 17)
      and (.transitions | type == "array" and length == 16)
      and .cycle_replay_phase_indexes == [8,16]
      and ([.phases[].resident.generation] | unique | length) == 17
      and all(.phases[];
        (.index | type == "number" and floor == . and . >= 0 and . < 17)
        and .process_pid == $root.process.pid
        and .format == .artifact.format
        and .resident.bytes == .artifact.bytes
        and (.resident.pool_key_sha256 | test("^[0-9a-f]{64}$"))
        and (.resident.engine_config_sha256 | test("^[0-9a-f]{64}$"))
        and .execution.pool_key_sha256 == .resident.pool_key_sha256
        and .execution.generation == .resident.generation
        and .execution.artifact_sha256 == .artifact.sha256
        and .execution.arch_family == "qwen35"
        and .execution.architecture == "qwen35"
        and (.result_sha256 | test("^[0-9a-f]{64}$"))
        and .semantic.role == "assistant"
        and .semantic.content == "HF2Q_SWAP_OK"
        and .semantic.finish_reason == "stop"
        and (.semantic.completion_tokens | type == "number" and . > 0 and floor == .)
        and .semantic.cached_tokens == 0
        and .memory.rss_bytes > 0
        and .memory.physical_footprint_bytes > 0
        and .memory.wired_bytes >= 0
        and .memory.host_wired_bytes > 0
        and (.storage | IN("file_backed","anonymous_accounted")))
      and (. as $root | all(range(0;17); . as $i |
        $root.phases[$i].index == $i
        and $root.phases[$i].format == $root.sequence[$i]
        and $root.phases[$i].artifact
          == ($root.artifacts[] | select(.format == $root.sequence[$i]))))
      and .phases[0].result_sha256 == .phases[8].result_sha256
      and .phases[0].result_sha256 == .phases[16].result_sha256
      and (.phases | group_by(.format) | all(.[];
        ([.[].resident.pool_key_sha256] | unique | length) == 1
        and ([.[].resident.engine_config_sha256] | unique | length) == 1
        and ([.[].result_sha256] | unique | length) == 1))
      and ([.phases[].resident.pool_key_sha256] | unique | length) == 5
      and (. as $root | all(range(0;16); . as $i |
        $root.phases[$i].memory as $previous
        | $root.phases[$i + 1].memory as $current
        | [((([$previous.rss_bytes,$current.rss_bytes] | max) / 10) | floor),2147483648] | max as $margin
        | $root.transitions[$i].index == $i
        and $root.transitions[$i].from == $root.sequence[$i]
        and $root.transitions[$i].to == $root.sequence[$i + 1]
        and $root.transitions[$i].load_seconds > 0
        and $root.transitions[$i].load_seconds < $root.load_budget_seconds
        and $root.transitions[$i].rss_bound_bytes
          == (([$previous.rss_bytes,$current.rss_bytes] | max) + $margin)
        and $root.transitions[$i].host_wired_bound_bytes
          == (([$previous.host_wired_bytes,$current.host_wired_bytes] | min)
            + $root.phases[$i + 1].artifact.bytes + $margin)
        and $root.transitions[$i].peak_rss_bytes <= $root.transitions[$i].rss_bound_bytes
        and $root.transitions[$i].peak_host_wired_bytes
          <= $root.transitions[$i].host_wired_bound_bytes))
      and (([((.phases[0].memory.rss_bytes / 10) | floor),2147483648] | max) as $rss_margin
        | .replay_bounds.rss_bytes == (.phases[0].memory.rss_bytes + $rss_margin)
        and .replay_bounds.host_wired_bytes
          == (([.phases[].memory.host_wired_bytes] | min)
            + .phases[0].artifact.bytes + $rss_margin))
      and (([((.phases[0].memory.physical_footprint_bytes / 10) | floor),2147483648] | max) as $footprint_margin
        | .replay_bounds.physical_footprint_bytes
          == (.phases[0].memory.physical_footprint_bytes + $footprint_margin))
      and (([((.phases[0].memory.wired_bytes / 10) | floor),2147483648] | max) as $wired_margin
        | .replay_bounds.wired_bytes == (.phases[0].memory.wired_bytes + $wired_margin))
      and (. as $root | all($root.cycle_replay_phase_indexes[]; . as $i |
        $root.phases[$i].memory.rss_bytes <= $root.replay_bounds.rss_bytes
        and $root.phases[$i].memory.physical_footprint_bytes
          <= $root.replay_bounds.physical_footprint_bytes
        and $root.phases[$i].memory.wired_bytes <= $root.replay_bounds.wired_bytes
        and $root.phases[$i].memory.host_wired_bytes <= $root.replay_bounds.host_wired_bytes))
    ' <<<"$chain_json" >/dev/null || return 1

    for format in $(qwen38_artifact_formats); do
        IFS=$'\t' read -r _format expected_file expected_bytes expected_sha expected_type \
            <<<"$(qwen38_artifact_record "$format")" || return 1
        IFS=$'\t' read -r actual_file actual_bytes actual_sha actual_type \
            < <(jq -er --arg format "$format" '
              .artifacts[] | select(.format == $format)
              | [.file,.bytes,.sha256,.gguf_file_type] | @tsv
            ' <<<"$chain_json") || return 1
        [[ "$actual_file" == "$expected_file" \
          && "$actual_bytes" == "$expected_bytes" \
          && "$actual_sha" == "$expected_sha" \
          && "$actual_type" == "$expected_type" ]] || return 1
    done
}

qwen38_validate_exact_swap_receipt() {
    local receipt=$1 source_root=${2:-}
    local source_commit binary_sha binary_git_commit dependency_version dependency_source
    local dependency_checksum pair_id format_a format_b index=0
    local receipt_identity source_identity

    [[ -f "$receipt" && -r "$receipt" && ! -L "$receipt" ]] || return 1
    jq -e \
      --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
      --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" '
      .schema == 1 and .verdict == "pass"
      and .gate == "qwen38-exact-model-swap-matrix"
      and .repository == $repository and .revision == $revision
      and .architecture == "qwen35" and .arch_family == "qwen35"
      and .formats == ["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"]
      and (.source_commit | test("^[0-9a-f]{40}$"))
      and (.binary.sha256 | test("^[0-9a-f]{64}$"))
      and .binary.git_commit == .source_commit
      and .dependency.name == "mlx-native"
      and (.dependency.version | test("^[0-9]+\\.[0-9]+\\.[0-9]+$"))
      and .dependency.source ==
        "registry+https://github.com/rust-lang/crates.io-index"
      and (.dependency.checksum | test("^[0-9a-f]{64}$"))
      and (.evidence.runner_sha256 | test("^[0-9a-f]{64}$"))
      and (.evidence.matrix_contract_sha256 | test("^[0-9a-f]{64}$"))
      and (.evidence.artifact_contract_sha256 | test("^[0-9a-f]{64}$"))
      and (.evidence.manifest_sha256 | test("^[0-9a-f]{64}$"))
      and (.results | type == "array" and length == 5)
      and (.chain | type == "object")
    ' "$receipt" >/dev/null || return 1
    source_commit=$(jq -er .source_commit "$receipt") || return 1
    binary_sha=$(jq -er .binary.sha256 "$receipt") || return 1
    binary_git_commit=$(jq -er .binary.git_commit "$receipt") || return 1
    IFS=$'\t' read -r dependency_version dependency_source dependency_checksum \
        < <(jq -er '[.dependency.version,.dependency.source,.dependency.checksum] | @tsv' \
          "$receipt") || return 1
    while IFS=$'\t' read -r pair_id format_a format_b; do
        qwen38_validate_exact_swap_cell \
          <(jq -c --argjson index "$index" '.results[$index]' "$receipt") \
          "$pair_id" "$format_a" "$format_b" "$source_commit" "$binary_sha" \
          "$binary_git_commit" "$dependency_version" "$dependency_source" \
          "$dependency_checksum" \
          || return 1
        index=$((index + 1))
    done <<'PAIRS'
bf16--q4_k_m	BF16	Q4_K_M
q4_k_m--q5_k_m	Q4_K_M	Q5_K_M
q5_k_m--q6_k	Q5_K_M	Q6_K
q6_k--q8_0	Q6_K	Q8_0
q8_0--bf16	Q8_0	BF16
PAIRS
    qwen38_validate_exact_swap_chain \
      <(jq -c '.chain' "$receipt") "$source_commit" "$binary_sha" \
      "$binary_git_commit" "$dependency_version" "$dependency_source" \
      "$dependency_checksum" || return 1
    if [[ -n "$source_root" ]]; then
        [[ "$(git -C "$source_root" rev-parse HEAD)" == "$source_commit" \
          && -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]] \
          || return 1
        source_identity=$(qwen38_mlx_native_registry_identity "$source_root") \
          || return 1
        receipt_identity="$dependency_version"$'\t'"$dependency_source"$'\t'"$dependency_checksum"
        [[ "$receipt_identity" == "$source_identity" ]] || return 1
    fi
}

qwen38_validate_exact_swap_seal() {
    local receipt=$1 source_root=${2:-}
    local receipt_dir evidence result manifest_path expected_entries actual_entries
    local expected_evidence_paths actual_evidence_paths
    local matrix_sha evidence_sha pair_id cell log
    local preflight_dir expected_preflight_paths actual_preflight_paths
    local format relative_file expected_sha receipt_slug preflight_receipt

    [[ "$(basename "$receipt")" == matrix.json ]] || return 1
    receipt_dir=$(cd "$(dirname "$receipt")" && pwd) || return 1
    evidence="$receipt_dir/evidence.sha256"
    result="$receipt_dir/result.sha256"
    [[ -f "$evidence" && ! -L "$evidence" \
      && -f "$result" && ! -L "$result" ]] || return 1
    qwen38_validate_exact_swap_receipt "$receipt" "$source_root" || return 1
    qwen38_validate_evidence_manifest_paths "$evidence" || return 1
    expected_evidence_paths='bf16--q4_k_m.json
bf16--q4_k_m.log
preflight/bf16.json
preflight/q4_k_m.json
preflight/q5_k_m.json
preflight/q6_k.json
preflight/q8_0.json
q4_k_m--q5_k_m.json
q4_k_m--q5_k_m.log
q5_k_m--q6_k.json
q5_k_m--q6_k.log
q6_k--q8_0.json
q6_k--q8_0.log
q8_0--bf16.json
q8_0--bf16.log
two-cycle-chain.json
two-cycle-chain.log'
    actual_evidence_paths=$(awk '{ print substr($0, 67) }' "$evidence" | sort) \
      || return 1
    [[ "$actual_evidence_paths" == "$expected_evidence_paths" ]] || return 1

    manifest_path="$QWEN38_ARTIFACT_CONTRACT_ROOT_DIR/data/qwen38_exact_swap_matrix.v1.json"
    jq -e \
      --arg runner "$(shasum -a 256 "$QWEN38_ARTIFACT_CONTRACT_ROOT_DIR/scripts/run_qwen38_exact_swap_matrix.sh" | awk '{print $1}')" \
      --arg matrix_contract "$(shasum -a 256 "$QWEN38_ARTIFACT_CONTRACT_ROOT_DIR/scripts/qwen38_exact_swap_matrix_contract.sh" | awk '{print $1}')" \
      --arg artifact_contract "$(shasum -a 256 "$QWEN38_ARTIFACT_CONTRACT_ROOT_DIR/scripts/qwen38_artifact_contract.sh" | awk '{print $1}')" \
      --arg manifest "$(shasum -a 256 "$manifest_path" | awk '{print $1}')" '
      .evidence == {runner_sha256:$runner,
        matrix_contract_sha256:$matrix_contract,
        artifact_contract_sha256:$artifact_contract,
        manifest_sha256:$manifest}
    ' "$receipt" >/dev/null || return 1
    expected_entries='bf16--q4_k_m.json
bf16--q4_k_m.log
evidence.sha256
matrix.json
preflight
q4_k_m--q5_k_m.json
q4_k_m--q5_k_m.log
q5_k_m--q6_k.json
q5_k_m--q6_k.log
q6_k--q8_0.json
q6_k--q8_0.log
q8_0--bf16.json
q8_0--bf16.log
result.sha256
two-cycle-chain.json
two-cycle-chain.log'
    actual_entries=$(cd "$receipt_dir" \
      && find . -mindepth 1 -maxdepth 1 -print | sed 's#^./##' | sort) \
      || return 1
    [[ "$actual_entries" == "$expected_entries" ]] || return 1
    preflight_dir="$receipt_dir/preflight"
    [[ -d "$preflight_dir" && ! -L "$preflight_dir" ]] || return 1
    expected_preflight_paths='bf16.json
q4_k_m.json
q5_k_m.json
q6_k.json
q8_0.json'
    actual_preflight_paths=$(cd "$preflight_dir" \
      && find . -mindepth 1 -maxdepth 1 -print | sed 's#^./##' | sort) \
      || return 1
    [[ "$actual_preflight_paths" == "$expected_preflight_paths" ]] || return 1
    for format in $(qwen38_artifact_formats); do
        IFS=$'\t' read -r _format relative_file _bytes expected_sha _type \
            <<<"$(qwen38_artifact_record "$format")" || return 1
        receipt_slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
        preflight_receipt="$preflight_dir/$receipt_slug.json"
        [[ -f "$preflight_receipt" && ! -L "$preflight_receipt" ]] || return 1
        jq -e --arg file "$relative_file" --arg sha "$expected_sha" '
          .schema_version == 2
          and (.path | type == "string" and endswith("/" + $file))
          and .sha256 == $sha
          and (.file_snapshot | type == "string" and length > 0)
          and (.file_stamp | type == "object")
          and .content_hash_verified == true
          and (.run_verification
            | IN("content_hash","cached_unchanged_file","upgraded_legacy_receipt"))
        ' "$preflight_receipt" >/dev/null || return 1
    done
    for pair_id in bf16--q4_k_m q4_k_m--q5_k_m q5_k_m--q6_k q6_k--q8_0 q8_0--bf16; do
        cell="$receipt_dir/$pair_id.json"
        log="$receipt_dir/$pair_id.log"
        [[ -f "$cell" && ! -L "$cell" && -f "$log" && ! -L "$log" ]] \
          || return 1
        [[ "$(jq -Sc --arg pair "$pair_id" '.results[] | select(.pair.id == $pair)' "$receipt")" \
          == "$(jq -Sc . "$cell")" ]] || return 1
    done
    cell="$receipt_dir/two-cycle-chain.json"
    log="$receipt_dir/two-cycle-chain.log"
    [[ -f "$cell" && ! -L "$cell" && -f "$log" && ! -L "$log" ]] \
      || return 1
    [[ "$(jq -Sc '.chain' "$receipt")" == "$(jq -Sc . "$cell")" ]] \
      || return 1
    [[ "$(awk 'END { print NR }' "$result")" == 2 ]] || return 1
    matrix_sha=$(shasum -a 256 "$receipt" | awk '{print $1}') || return 1
    evidence_sha=$(shasum -a 256 "$evidence" | awk '{print $1}') || return 1
    [[ "$(sed -n '1p' "$result")" == "$matrix_sha  matrix.json" \
      && "$(sed -n '2p' "$result")" == "$evidence_sha  evidence.sha256" ]] \
      || return 1
    (cd "$receipt_dir" && shasum -a 256 -c evidence.sha256 >/dev/null \
      && shasum -a 256 -c result.sha256 >/dev/null)
}
