#!/usr/bin/env bash

# Model-free schema and receipt authority for the pool-resident generative
# cross-family swap gate. BERT and Nomic have a dedicated process-global
# lifecycle and are excluded.

HF2Q_GENERATIVE_SWAP_REGISTRY_SOURCE='registry+https://github.com/rust-lang/crates.io-index'
HF2Q_GENERATIVE_SWAP_GIB=$((1024 * 1024 * 1024))

hf2q_generative_swap_sha256_file() {
    shasum -a 256 "$1" | awk '{print $1}'
}

hf2q_validate_generative_swap_matrix() {
    local manifest=${1:?matrix manifest is required}
    [[ -f "$manifest" && -r "$manifest" && ! -L "$manifest" ]] || return 1
    jq -e '
      (keys | sort) == (["artifacts","excluded_subsystems","load_budget_seconds",
        "pairs","schema","sequence"] | sort)
      and .schema == "hf2q.generative-swap-matrix.v1"
      and .load_budget_seconds == 60
      and (.artifacts | type == "array" and length == 4)
      and (.pairs | type == "array" and length == 3)
      and .excluded_subsystems == ["bert", "nomic"]
      and ([.artifacts[].id] | sort
        == ["deepseek", "gemma", "qwen-dense", "qwen-moe"])
      and ([.artifacts[] | [.id, .architecture, .arch_family]] | sort
        == ([
          ["qwen-dense", "qwen35", "qwen35"],
          ["qwen-moe", "qwen35moe", "qwen35"],
          ["gemma", "gemma4", "gemma4"],
          ["deepseek", "deepseek4", "deepseek4"]
        ] | sort))
      and [.artifacts[] | [.id,.path_env,.file,.bytes,.sha256,.canary]] == [
        ["qwen-dense","HF2Q_SWAP_QWEN_DENSE_PATH",
          "Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf",
          16810714944,
          "1ee55c653644d6f645c6b2f39fc56a3ce28093620fd34dd43678875f348f2e1a",
          "HF2Q_SWAP_QWEN_DENSE_OK"],
        ["qwen-moe","HF2Q_SWAP_QWEN_MOE_PATH","APEX-Q5_K_M.gguf",25043007488,
          "f2c702182a4661d2cef573b388ff23336ce65aabb112762d1c1a24d4ba0cbc25",
          "HF2Q_SWAP_QWEN_MOE_OK"],
        ["gemma","HF2Q_SWAP_GEMMA_PATH","gemma4-ara-2pass-APEX-Q5_K_M.gguf",
          20576631488,
          "82beae39cdee643824dde5bc3fb1a3d6e2e4f8701572930163b0d703298bcf82",
          "HF2Q_SWAP_GEMMA_OK"],
        ["deepseek","HF2Q_SWAP_DEEPSEEK_PATH",
          "DeepSeek-V4-Flash-0731-agentic-q2.gguf",107431343168,
          "936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d",
          "HF2Q_SWAP_DEEPSEEK_OK"]
      ]
      and (all(.artifacts[];
        (keys | sort) == (["arch_family","architecture","bytes","canary","file",
          "id","path_env","sha256"] | sort)
        and (.path_env | type == "string" and test("^HF2Q_SWAP_[A-Z_]+_PATH$"))
        and (.file | type == "string" and length > 5
          and (contains("/") | not) and endswith(".gguf"))
        and (.bytes | type == "number" and floor == . and . > 0)
        and (.sha256 | type == "string" and test("^[0-9a-f]{64}$"))
        and (.canary | type == "string"
          and test("^HF2Q_SWAP_[A-Z_]+_OK$") and length <= 32)
        and (has("sha256_env") | not)))
      and ([.artifacts[].path_env] | unique | length == 4)
      and ([.artifacts[].file] | unique | length == 4)
      and ([.artifacts[].bytes] | unique | length == 4)
      and ([.artifacts[].sha256] | unique | length == 4)
      and ([.artifacts[].canary] | unique | length == 4)
      and ([.pairs[] | [.id, .a, .b]] | sort
        == ([
          ["qwen-dense--deepseek", "qwen-dense", "deepseek"],
          ["qwen-moe--deepseek", "qwen-moe", "deepseek"],
          ["gemma--deepseek", "gemma", "deepseek"]
        ] | sort))
      and (all(.pairs[]; .a != .b))
      and (all(.pairs[]; (keys | sort) == ["a","b","id"]))
      and ([.pairs[].a] | sort == ["gemma", "qwen-dense", "qwen-moe"])
      and ([.pairs[].b] | unique == ["deepseek"])
      and .sequence == [
        "qwen-dense", "deepseek", "qwen-moe", "deepseek", "gemma",
        "deepseek", "qwen-dense", "deepseek", "qwen-moe", "deepseek",
        "gemma", "deepseek", "qwen-dense"
      ]
      and (. as $matrix
        | ($matrix.artifacts | map(.bytes) | max) as $budget
        | all(range(0; ($matrix.sequence | length) - 1);
            . as $index
            | ($matrix.artifacts[]
                | select(.id == $matrix.sequence[$index]) | .bytes) as $before
            | ($matrix.artifacts[]
                | select(.id == $matrix.sequence[$index + 1]) | .bytes) as $after
            | ($before + $after) > $budget))
      and (all(.artifacts[]; (.id | test("bert|nomic"; "i")) | not))
    ' "$manifest" >/dev/null
}

# Cargo configuration may inject an ignored path patch or compiler wrapper
# after Git identity is established. Exact-source execution rejects every
# configuration file in source ancestry and Cargo home.
hf2q_generative_swap_reject_cargo_configuration() {
    local source_root=${1:?source root is required}
    local cargo_home=${2:-${CARGO_HOME:-${HOME:?HOME is required}/.cargo}}
    local current candidate

    source_root=$(cd "$source_root" && pwd -P) || return 1
    current=$source_root
    while :; do
        for candidate in "$current/.cargo/config" "$current/.cargo/config.toml"; do
            [[ ! -e "$candidate" && ! -L "$candidate" ]] || {
                echo "generative swap gate rejects Cargo configuration: $candidate" >&2
                return 1
            }
        done
        [[ "$current" == / ]] && break
        current=$(dirname "$current")
    done
    for candidate in "$cargo_home/config" "$cargo_home/config.toml"; do
        [[ ! -e "$candidate" && ! -L "$candidate" ]] || {
            echo "generative swap gate rejects Cargo configuration: $candidate" >&2
            return 1
        }
    done
}

# Print version<TAB>source<TAB>checksum for the one exact mlx-native registry
# dependency selected by both Cargo.toml and Cargo.lock.
hf2q_generative_swap_dependency_identity() {
    local source_root=${1:?source root is required}
    local manifest_versions manifest_version lock_records record_count
    local lock_version lock_source lock_checksum

    [[ -f "$source_root/Cargo.toml" && -f "$source_root/Cargo.lock" ]] || return 1
    manifest_versions=$(sed -nE \
      's/^[[:space:]]*mlx-native[[:space:]]*=[[:space:]]*"=([0-9]+\.[0-9]+\.[0-9]+)"[[:space:]]*$/\1/p' \
      "$source_root/Cargo.toml")
    [[ "$(printf '%s\n' "$manifest_versions" \
      | awk 'NF { count++ } END { print count + 0 }')" == 1 ]] || {
        echo "mlx-native must have one exact manifest requirement" >&2
        return 1
    }
    manifest_version=$(printf '%s\n' "$manifest_versions" | awk 'NF { print; exit }')
    lock_records=$(awk '
      function emit() {
        if (in_package && name == "mlx-native") {
          print version "\t" source "\t" checksum
        }
      }
      /^\[\[package\]\]$/ {
        emit(); in_package = 1; name = version = source = checksum = ""; next
      }
      in_package && /^name = "/ { v=$0; sub(/^name = "/,"",v); sub(/"$/,"",v); name=v; next }
      in_package && /^version = "/ { v=$0; sub(/^version = "/,"",v); sub(/"$/,"",v); version=v; next }
      in_package && /^source = "/ { v=$0; sub(/^source = "/,"",v); sub(/"$/,"",v); source=v; next }
      in_package && /^checksum = "/ { v=$0; sub(/^checksum = "/,"",v); sub(/"$/,"",v); checksum=v; next }
      END { emit() }
    ' "$source_root/Cargo.lock")
    record_count=$(printf '%s\n' "$lock_records" \
      | awk 'NF { count++ } END { print count + 0 }')
    [[ "$record_count" == 1 ]] || {
        echo "Cargo.lock must contain one mlx-native package" >&2
        return 1
    }
    IFS=$'\t' read -r lock_version lock_source lock_checksum <<<"$lock_records"
    [[ "$lock_version" == "$manifest_version" \
      && "$lock_source" == "$HF2Q_GENERATIVE_SWAP_REGISTRY_SOURCE" \
      && "$lock_checksum" =~ ^[0-9a-f]{64}$ ]] || {
        echo "mlx-native manifest/registry lock identity is not exact" >&2
        return 1
    }
    printf '%s\t%s\t%s\n' "$lock_version" "$lock_source" "$lock_checksum"
}

# Validate the execution receipt independently of the Rust producer. In
# particular, transition and replay memory ceilings are recomputed from raw
# phase samples instead of trusting producer-authored bound fields.
hf2q_validate_generative_swap_receipt() {
    local receipt=${1:?execution receipt is required}
    local manifest=${2:?matrix manifest is required}
    local source_commit=${3:?source commit is required}
    local binary_sha256=${4:?binary SHA-256 is required}
    local dependency_version=${5:?dependency version is required}
    local dependency_source=${6:?dependency source is required}
    local dependency_checksum=${7:?dependency checksum is required}
    local receipt_kind=${8:-runtime}

    hf2q_validate_generative_swap_matrix "$manifest" || return 1
    [[ -f "$receipt" && -r "$receipt" && ! -L "$receipt" \
      && "$source_commit" =~ ^[0-9a-f]{40}$ \
      && "$binary_sha256" =~ ^[0-9a-f]{64}$ \
      && "$dependency_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ \
      && "$dependency_source" == "$HF2Q_GENERATIVE_SWAP_REGISTRY_SOURCE" \
      && "$dependency_checksum" =~ ^[0-9a-f]{64}$ ]] || return 1

    jq -e \
      --arg source_commit "$source_commit" \
      --arg binary_sha256 "$binary_sha256" \
      --arg dependency_version "$dependency_version" \
      --arg dependency_source "$dependency_source" \
      --arg dependency_checksum "$dependency_checksum" \
      --arg receipt_kind "$receipt_kind" \
      --argjson gib "$HF2Q_GENERATIVE_SWAP_GIB" \
      --slurpfile manifest "$manifest" '
      def margin($bytes): ([($bytes / 10 | floor), (2 * $gib)] | max);
      . as $receipt | $manifest[0] as $matrix
      | (["artifacts","binding","cycle_replay_phase_indexes","gate",
          "load_budget_seconds","phases","pool_budget_bytes","process","proof",
          "replay_bounds","schema","sequence","transitions","verdict"] | sort) as $base_keys
      | (($receipt_kind == "runtime" and (keys | sort) == $base_keys)
          or ($receipt_kind == "matrix"
            and (keys | sort) == (($base_keys + ["evidence"]) | sort)))
      and .schema == 1 and .verdict == "pass"
      and .gate == "generative-cross-family-two-cycle"
      and .binding == {
        source_commit:$source_commit,
        binary_sha256:$binary_sha256,
        binary_git_commit:$source_commit,
        dependency:{name:"mlx-native",version:$dependency_version,
          source:$dependency_source,checksum:$dependency_checksum}
      }
      and .load_budget_seconds == $matrix.load_budget_seconds
      and .pool_budget_bytes == ($matrix.artifacts | map(.bytes) | max)
      and (.process | keys) == ["pid"]
      and (.process.pid | type == "number" and floor == . and . > 0)
      and .sequence == $matrix.sequence
      and .cycle_replay_phase_indexes == [6,12]
      and .proof == {
        one_long_lived_process:true,
        two_complete_cycles:true,
        every_required_family:true,
        unique_semantic_canary_per_family:true,
        fresh_generation_every_activation:true,
        cold_generation_cache:true,
        execution_receipts_joined:true,
        process_policy_preserved:true,
        q5_policy_preserved_and_observed:true,
        bounded_every_transition:true,
        evicted_artifacts_absent:true,
        exact_family_replay:true
      }
      and .artifacts == ($matrix.artifacts | map({
        id,architecture,arch_family,file,bytes,sha256,canary
      }))
      and (.phases | type == "array" and length == ($matrix.sequence | length))
      and (.phases | map(.index)) == [range(0; ($matrix.sequence | length))]
      and all(.phases[];
        . as $phase
        | ($matrix.artifacts[] | select(.id == $phase.artifact.id)) as $artifact
        | (keys | sort) == (["artifact","execution","format","index","memory",
            "phase","process_pid","process_policy","process_policy_sha256","q5_route","resident",
            "result_sha256","semantic","storage"] | sort)
        and $matrix.sequence[$phase.index] == $artifact.id
        and $phase.format == $artifact.id
        and $phase.phase == ("P"
          + (if $phase.index < 10 then "0" else "" end)
          + ($phase.index | tostring) + "-" + $artifact.id)
        and $phase.process_pid == $receipt.process.pid
        and $phase.artifact == ($artifact | {
          id,architecture,arch_family,file,bytes,sha256,canary
        })
        and ($phase.resident | keys | sort)
          == ["bytes","engine_config_sha256","generation","pool_key_sha256"]
        and ($phase.resident.pool_key_sha256 | test("^[0-9a-f]{64}$"))
        and ($phase.resident.generation | type == "number" and floor == . and . > 0)
        and $phase.resident.bytes == $artifact.bytes
        and ($phase.resident.engine_config_sha256 | test("^[0-9a-f]{64}$"))
        and ($phase.process_policy | keys | sort) == ([
          "ggml_routing","kv_cache_budget_bytes","kv_metrics_sink",
          "kv_persist_budget_bytes","kv_persist_enabled","queue_capacity",
          "requested_context_tokens","scheduler","schema_version",
          "warmup_synchronously"] | sort)
        and $phase.process_policy.schema_version == 1
        and ($phase.process_policy.scheduler | keys | sort) == ["mode"]
        and $phase.process_policy.scheduler.mode == "serial_fifo"
        and ($phase.process_policy.requested_context_tokens == null
          or ($phase.process_policy.requested_context_tokens | type == "number"))
        and ($phase.process_policy.kv_cache_budget_bytes == null
          or ($phase.process_policy.kv_cache_budget_bytes | type == "number"))
        and ($phase.process_policy.queue_capacity | type == "number" and floor == . and . > 0)
        and ($phase.process_policy.warmup_synchronously | type == "boolean")
        and ($phase.process_policy.kv_metrics_sink | type == "boolean")
        and ($phase.process_policy.kv_persist_enabled | type == "boolean")
        and ($phase.process_policy.kv_persist_budget_bytes
          | type == "number" and floor == . and . >= 0)
        and ($phase.process_policy.ggml_routing | keys | sort) == ([
          "allow_dense_large_tile_mm","dense_decode_mv_ext","dense_decode_mvn",
          "dense_q5k_canonical_q4x4","dense_q6k_mv_nr2","dense_q8_0_mv_nr2",
          "dense_tensor_mm","expert_mm_threshold","expert_q6k_mv_nr2",
          "expert_q8_0_mv_nr2","expert_tensor_mm"] | sort)
        and $phase.process_policy.ggml_routing.dense_q5k_canonical_q4x4 == true
        and ($phase.process_policy_sha256 | test("^[0-9a-f]{64}$"))
        and (if ($artifact.arch_family == "qwen35"
          and ($artifact.file | contains("Q5_K_M"))) then
          $phase.q5_route == {
            policy_enabled:true,
            route:"dense_q5k_canonical_q4x4",
            route_observed:true
          }
        else
          $phase.q5_route == {
            policy_enabled:true,
            route:"N/A",
            route_observed:false
          }
        end)
        and ($phase.execution | keys | sort)
          == ["arch_family","architecture","artifact_sha256","generation",
              "pool_key_sha256"]
        and $phase.execution.pool_key_sha256 == $phase.resident.pool_key_sha256
        and $phase.execution.generation == $phase.resident.generation
        and $phase.execution.artifact_sha256 == $artifact.sha256
        and $phase.execution.arch_family == $artifact.arch_family
        and $phase.execution.architecture == $artifact.architecture
        and ($phase.result_sha256 | test("^[0-9a-f]{64}$"))
        and ($phase.semantic | keys | sort)
          == ["cached_tokens","completion_tokens","content","finish_reason","role"]
        and $phase.semantic.role == "assistant"
        and $phase.semantic.content == $artifact.canary
        and $phase.semantic.finish_reason == "stop"
        and ($phase.semantic.completion_tokens | type == "number" and floor == . and . > 0)
        and $phase.semantic.cached_tokens == 0
        and ($phase.storage == "file_backed"
          or $phase.storage == "anonymous_accounted")
        and ($phase.memory | keys | sort) == ["host_wired_bytes",
          "physical_footprint_bytes","physical_footprint_peak_bytes","rss_bytes",
          "wired_bytes"]
        and all($phase.memory[]; type == "number" and floor == . and . > 0))
      and ([.phases[].resident.generation] | unique | length)
        == ($matrix.sequence | length)
      and ([.phases[].resident.pool_key_sha256] | unique | length) == 4
      and ([.phases[].process_policy_sha256] | unique | length) == 1
      and ([.phases[].process_policy] | unique | length) == 1
      and (.phases | group_by(.artifact.id) | all(.[];
        ([.[].resident.pool_key_sha256] | unique | length) == 1
        and ([.[].resident.engine_config_sha256] | unique | length) == 1
        and ([.[].result_sha256] | unique | length) == 1
        and ([.[].semantic.content] | unique | length) == 1))
      and (.transitions | type == "array"
        and length == (($matrix.sequence | length) - 1))
      and (.transitions | map(.index))
        == [range(0; (($matrix.sequence | length) - 1))]
      and all(.transitions[];
        . as $transition
        | $receipt.phases[$transition.index] as $before
        | $receipt.phases[$transition.index + 1] as $after
        | ([$before.memory.rss_bytes, $after.memory.rss_bytes] | max) as $rss
        | ([$before.memory.host_wired_bytes, $after.memory.host_wired_bytes] | min) as $host_wired
        | margin($rss) as $memory_margin
        | (keys | sort) == ["from","host_wired_bound_bytes","index","load_seconds",
            "peak_host_wired_bytes","peak_rss_bytes","rss_bound_bytes","to"]
        and $transition.from == $before.artifact.id
        and $transition.to == $after.artifact.id
        and ($transition.load_seconds | type == "number" and . >= 0
          and . < $matrix.load_budget_seconds)
        and $transition.rss_bound_bytes == ($rss + $memory_margin)
        and $transition.host_wired_bound_bytes
          == ($host_wired + $after.artifact.bytes + $memory_margin)
        and $transition.peak_rss_bytes <= ($rss + $memory_margin)
        and $transition.peak_host_wired_bytes
          <= ($host_wired + $after.artifact.bytes + $memory_margin))
      and .replay_bounds == {
        rss_bytes:($receipt.phases[0].memory.rss_bytes
          + margin($receipt.phases[0].memory.rss_bytes)),
        physical_footprint_bytes:($receipt.phases[0].memory.physical_footprint_bytes
          + margin($receipt.phases[0].memory.physical_footprint_bytes)),
        wired_bytes:($receipt.phases[0].memory.wired_bytes
          + margin($receipt.phases[0].memory.wired_bytes)),
        host_wired_bytes:(([$receipt.phases[].memory.host_wired_bytes] | min)
          + $receipt.phases[0].artifact.bytes
          + margin($receipt.phases[0].memory.rss_bytes))
      }
      and all(.cycle_replay_phase_indexes[];
        . as $index
        | $receipt.phases[$index].artifact.id == $receipt.phases[0].artifact.id
        and $receipt.phases[$index].result_sha256 == $receipt.phases[0].result_sha256
        and $receipt.phases[$index].memory.rss_bytes <= $receipt.replay_bounds.rss_bytes
        and $receipt.phases[$index].memory.physical_footprint_bytes
          <= $receipt.replay_bounds.physical_footprint_bytes
        and $receipt.phases[$index].memory.wired_bytes
          <= $receipt.replay_bounds.wired_bytes
        and $receipt.phases[$index].memory.host_wired_bytes
          <= $receipt.replay_bounds.host_wired_bytes)
    ' "$receipt" >/dev/null
}

hf2q_validate_generative_swap_matrix_receipt() {
    local matrix=${1:?matrix receipt is required}
    local manifest=${2:?matrix manifest is required}
    local source_commit=${3:?source commit is required}
    local binary_sha256=${4:?binary SHA-256 is required}
    local dependency_version=${5:?dependency version is required}
    local dependency_source=${6:?dependency source is required}
    local dependency_checksum=${7:?dependency checksum is required}
    local runner_sha=${8:?runner SHA-256 is required}
    local contract_sha=${9:?contract SHA-256 is required}
    local manifest_sha=${10:?manifest SHA-256 is required}
    local runtime_sha=${11:?runtime receipt SHA-256 is required}

    hf2q_validate_generative_swap_receipt "$matrix" "$manifest" \
      "$source_commit" "$binary_sha256" "$dependency_version" \
      "$dependency_source" "$dependency_checksum" matrix || return 1
    jq -e \
      --arg runner_sha "$runner_sha" --arg contract_sha "$contract_sha" \
      --arg manifest_sha "$manifest_sha" --arg runtime_sha "$runtime_sha" '
      .evidence == {
        runner_sha256:$runner_sha,
        contract_sha256:$contract_sha,
        manifest_sha256:$manifest_sha,
        runtime_receipt_sha256:$runtime_sha
      }
    ' "$matrix" >/dev/null
}

hf2q_validate_generative_swap_seal() {
    local matrix=${1:?matrix receipt is required}
    local manifest=${2:?matrix manifest is required}
    local source_root=${3:?exact source root is required}
    local directory runtime evidence result
    local source_commit binary_sha dependency_version dependency_source dependency_checksum
    local runtime_sha runner_sha contract_sha manifest_sha expected_entries

    [[ -f "$matrix" && -r "$matrix" && ! -L "$matrix" \
      && "$(basename "$matrix")" == matrix.json ]] || return 1
    directory=$(cd "$(dirname "$matrix")" && pwd -P) || return 1
    runtime="$directory/runtime.json"
    evidence="$directory/evidence.sha256"
    result="$directory/result.sha256"
    for path in "$runtime" "$evidence" "$result"; do
        [[ -f "$path" && -r "$path" && ! -L "$path" ]] || return 1
    done
    expected_entries=$'evidence.sha256\nmatrix.json\nresult.sha256\nruntime.json\nruntime.log'
    [[ "$(find "$directory" -mindepth 1 -maxdepth 1 -print \
      | sed 's#^.*/##' | sort)" == "$expected_entries" ]] || return 1

    source_commit=$(jq -er '.binding.source_commit' "$matrix") || return 1
    binary_sha=$(jq -er '.binding.binary_sha256' "$matrix") || return 1
    IFS=$'\t' read -r dependency_version dependency_source dependency_checksum \
      <<<"$(jq -er '[.binding.dependency.version,.binding.dependency.source,.binding.dependency.checksum] | @tsv' "$matrix")"
    runtime_sha=$(hf2q_generative_swap_sha256_file "$runtime") || return 1
    runner_sha=$(hf2q_generative_swap_sha256_file \
      "$source_root/scripts/run_generative_swap_matrix.sh") || return 1
    contract_sha=$(hf2q_generative_swap_sha256_file \
      "$source_root/scripts/generative_swap_matrix_contract.sh") || return 1
    manifest_sha=$(hf2q_generative_swap_sha256_file "$manifest") || return 1

    hf2q_validate_generative_swap_matrix_receipt "$matrix" "$manifest" \
      "$source_commit" "$binary_sha" "$dependency_version" "$dependency_source" \
      "$dependency_checksum" "$runner_sha" "$contract_sha" "$manifest_sha" \
      "$runtime_sha" || return 1
    hf2q_validate_generative_swap_receipt "$runtime" "$manifest" \
      "$source_commit" "$binary_sha" "$dependency_version" "$dependency_source" \
      "$dependency_checksum" || return 1
    [[ "$(jq -Sc 'del(.evidence)' "$matrix")" == "$(jq -Sc . "$runtime")" ]] \
      || return 1
    [[ "$(awk 'END { print NR }' "$evidence")" == 2 \
      && "$(sed -n '1p' "$evidence")" == \
        "$(hf2q_generative_swap_sha256_file "$runtime")  runtime.json" \
      && "$(sed -n '2p' "$evidence")" == \
        "$(hf2q_generative_swap_sha256_file "$directory/runtime.log")  runtime.log" \
      && "$(awk 'END { print NR }' "$result")" == 2 \
      && "$(sed -n '1p' "$result")" == \
        "$(hf2q_generative_swap_sha256_file "$matrix")  matrix.json" \
      && "$(sed -n '2p' "$result")" == \
        "$(hf2q_generative_swap_sha256_file "$evidence")  evidence.sha256" ]] \
      || return 1
    (cd "$directory" && shasum -a 256 -c evidence.sha256 >/dev/null \
      && shasum -a 256 -c result.sha256 >/dev/null)

    [[ "$(git -C "$source_root" rev-parse HEAD)" == "$source_commit" \
      && -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" \
      && "$(hf2q_generative_swap_dependency_identity "$source_root")" == \
        "$dependency_version"$'\t'"$dependency_source"$'\t'"$dependency_checksum" ]]
}
