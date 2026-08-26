#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
# shellcheck source=scripts/generative_swap_matrix_contract.sh
source "$ROOT_DIR/scripts/generative_swap_matrix_contract.sh"
manifest="$ROOT_DIR/data/generative_swap_matrix.v1.json"
hf2q_validate_generative_swap_matrix "$manifest"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Exact-source dependency identity must reject local/path-shaped locks and
# Cargo configuration injection before a model process can start.
mkdir -p "$tmp/dependency" "$tmp/cargo-home" "$tmp/config-source"
printf 'mlx-native = "=9.8.7"\n' >"$tmp/dependency/Cargo.toml"
printf '%s\n' \
  '[[package]]' \
  'name = "mlx-native"' \
  'version = "9.8.7"' \
  'source = "registry+https://github.com/rust-lang/crates.io-index"' \
  'checksum = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"' \
  >"$tmp/dependency/Cargo.lock"
[[ "$(hf2q_generative_swap_dependency_identity "$tmp/dependency")" == \
  $'9.8.7\tregistry+https://github.com/rust-lang/crates.io-index\tcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc' ]]
sed '/^source = /d;/^checksum = /d' "$tmp/dependency/Cargo.lock" \
  >"$tmp/dependency/Cargo.lock.path"
mv "$tmp/dependency/Cargo.lock.path" "$tmp/dependency/Cargo.lock"
if hf2q_generative_swap_dependency_identity "$tmp/dependency" >/dev/null 2>&1; then
    echo "path-shaped dependency lock unexpectedly passed" >&2
    exit 1
fi
hf2q_generative_swap_reject_cargo_configuration \
  "$tmp/config-source" "$tmp/cargo-home"
mkdir -p "$tmp/config-source/.cargo"
touch "$tmp/config-source/.cargo/config.toml"
if hf2q_generative_swap_reject_cargo_configuration \
  "$tmp/config-source" "$tmp/cargo-home" >/dev/null 2>&1; then
    echo "source-local Cargo configuration unexpectedly passed" >&2
    exit 1
fi

expect_manifest_rejected() {
    local label=$1 fixture=$2
    if hf2q_validate_generative_swap_matrix "$fixture"; then
        echo "manifest mutation unexpectedly passed: $label" >&2
        exit 1
    fi
}

jq 'del(.artifacts[0])' "$manifest" >"$tmp/missing-family.json"
expect_manifest_rejected missing-family "$tmp/missing-family.json"
jq '(.artifacts[] | select(.id == "qwen-moe")).architecture = "qwen35"' \
  "$manifest" >"$tmp/dense-moe-collapse.json"
expect_manifest_rejected dense-moe-collapse "$tmp/dense-moe-collapse.json"
jq '.artifacts[0].id = "bert"' "$manifest" >"$tmp/bert-leak.json"
expect_manifest_rejected bert-leak "$tmp/bert-leak.json"
jq 'del(.pairs[2])' "$manifest" >"$tmp/missing-pair.json"
expect_manifest_rejected missing-pair "$tmp/missing-pair.json"
jq '.sequence[4] = "qwen-moe"' "$manifest" >"$tmp/broken-cycle.json"
expect_manifest_rejected broken-cycle "$tmp/broken-cycle.json"
jq '.artifacts[1].sha256 = .artifacts[0].sha256' \
  "$manifest" >"$tmp/shared-artifact-digest.json"
expect_manifest_rejected shared-artifact-digest "$tmp/shared-artifact-digest.json"
jq '.artifacts[1].canary = .artifacts[0].canary' \
  "$manifest" >"$tmp/shared-canary.json"
expect_manifest_rejected shared-canary "$tmp/shared-canary.json"
jq '.artifacts[0].bytes += 1' "$manifest" >"$tmp/wrong-bytes.json"
expect_manifest_rejected wrong-bytes "$tmp/wrong-bytes.json"
jq '.artifacts[0].sha256_env = "HF2Q_SWAP_QWEN_DENSE_SHA256"' \
  "$manifest" >"$tmp/ambient-digest.json"
expect_manifest_rejected ambient-digest "$tmp/ambient-digest.json"
jq '.artifacts[0].path_env = "HF2Q_SWAP_ALTERNATE_PATH"' \
  "$manifest" >"$tmp/ambient-path.json"
expect_manifest_rejected ambient-path "$tmp/ambient-path.json"

source_commit=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
binary_sha=$(printf 'b%.0s' {1..64})
dependency_version=9.8.7
dependency_source=$HF2Q_GENERATIVE_SWAP_REGISTRY_SOURCE
dependency_checksum=$(printf 'c%.0s' {1..64})
gib=$HF2Q_GENERATIVE_SWAP_GIB

# Independent synthetic producer: its raw measurements are deliberately
# simple enough to make every recomputed bound auditable in this test.
jq -n \
  --arg source_commit "$source_commit" --arg binary_sha "$binary_sha" \
  --arg dependency_version "$dependency_version" \
  --arg dependency_source "$dependency_source" \
  --arg dependency_checksum "$dependency_checksum" \
  --argjson gib "$gib" --slurpfile manifest "$manifest" '
  def key($id):
    if $id == "qwen-dense" then ("1" * 64)
    elif $id == "qwen-moe" then ("2" * 64)
    elif $id == "gemma" then ("3" * 64)
    else ("4" * 64) end;
  def config($id):
    if $id == "qwen-dense" then ("5" * 64)
    elif $id == "qwen-moe" then ("6" * 64)
    elif $id == "gemma" then ("7" * 64)
    else ("8" * 64) end;
  def result($id):
    if $id == "qwen-dense" then ("9" * 64)
    elif $id == "qwen-moe" then ("a" * 64)
    elif $id == "gemma" then ("b" * 64)
    else ("c" * 64) end;
  def rss($id):
    if $id == "qwen-dense" then 20*$gib
    elif $id == "qwen-moe" then 30*$gib
    elif $id == "gemma" then 25*$gib
    else 100*$gib end;
  def memory($id): {
    rss_bytes:rss($id),
    physical_footprint_bytes:(rss($id) - $gib),
    physical_footprint_peak_bytes:rss($id),
    wired_bytes:(3*$gib),
    host_wired_bytes:(8*$gib)
  };
  def margin($bytes): ([($bytes/10|floor),2*$gib]|max);
  def process_policy: {
    schema_version:1,
    scheduler:{mode:"serial_fifo"},
    requested_context_tokens:null,
    kv_cache_budget_bytes:null,
    queue_capacity:32,
    warmup_synchronously:true,
    kv_metrics_sink:true,
    kv_persist_enabled:false,
    kv_persist_budget_bytes:0,
    ggml_routing:{
      dense_q5k_canonical_q4x4:true,
      dense_decode_mvn:true,
      dense_decode_mv_ext:false,
      dense_q6k_mv_nr2:true,
      dense_q8_0_mv_nr2:true,
      dense_tensor_mm:"auto_probe",
      allow_dense_large_tile_mm:true,
      expert_mm_threshold:8,
      expert_q6k_mv_nr2:true,
      expert_q8_0_mv_nr2:false,
      expert_tensor_mm:"auto_probe"
    }
  };
  $manifest[0] as $m
  | [range(0;($m.sequence|length)) as $index
      | $m.sequence[$index] as $id
      | ($m.artifacts[] | select(.id==$id)) as $artifact
      | {
          index:$index,
          phase:("P"+(if $index<10 then "0" else "" end)+($index|tostring)+"-"+$id),
          format:$id,process_pid:42,
          artifact:($artifact|{id,architecture,arch_family,file,bytes,sha256,canary}),
          resident:{pool_key_sha256:key($id),generation:($index+1),bytes:$artifact.bytes,
            engine_config_sha256:config($id)},
          process_policy:process_policy,
          process_policy_sha256:("f" * 64),
          q5_route:(if ($artifact.arch_family == "qwen35"
            and ($artifact.file | contains("Q5_K_M"))) then {
            policy_enabled:true,route:"dense_q5k_canonical_q4x4",route_observed:true
          } else {
            policy_enabled:true,route:"N/A",route_observed:false
          } end),
          execution:{pool_key_sha256:key($id),generation:($index+1),
            artifact_sha256:$artifact.sha256,arch_family:$artifact.arch_family,
            architecture:$artifact.architecture},
          result_sha256:result($id),
          semantic:{role:"assistant",content:$artifact.canary,finish_reason:"stop",
            completion_tokens:1,cached_tokens:0},
          memory:memory($id),storage:"file_backed"
        }
    ] as $phases
  | [range(0;(($m.sequence|length)-1)) as $index
      | $phases[$index] as $before | $phases[$index+1] as $after
      | ([$before.memory.rss_bytes,$after.memory.rss_bytes]|max) as $rss
      | ([$before.memory.host_wired_bytes,$after.memory.host_wired_bytes]|min) as $wired
      | margin($rss) as $margin
      | {index:$index,from:$before.artifact.id,to:$after.artifact.id,load_seconds:1,
          peak_rss_bytes:$rss,peak_host_wired_bytes:$wired,
          rss_bound_bytes:($rss+$margin),
          host_wired_bound_bytes:($wired+$after.artifact.bytes+$margin)}
    ] as $transitions
  | {
      schema:1,verdict:"pass",gate:"generative-cross-family-two-cycle",
      binding:{source_commit:$source_commit,binary_sha256:$binary_sha,
        binary_git_commit:$source_commit,
        dependency:{name:"mlx-native",version:$dependency_version,
          source:$dependency_source,checksum:$dependency_checksum}},
      artifacts:($m.artifacts|map({id,architecture,arch_family,file,bytes,sha256,canary})),
      pool_budget_bytes:($m.artifacts|map(.bytes)|max),
      load_budget_seconds:$m.load_budget_seconds,process:{pid:42},sequence:$m.sequence,
      proof:{one_long_lived_process:true,two_complete_cycles:true,
        every_required_family:true,unique_semantic_canary_per_family:true,
        fresh_generation_every_activation:true,cold_generation_cache:true,
        execution_receipts_joined:true,bounded_every_transition:true,
        process_policy_preserved:true,
        q5_policy_preserved_and_observed:true,
        evicted_artifacts_absent:true,exact_family_replay:true},
      phases:$phases,transitions:$transitions,cycle_replay_phase_indexes:[6,12],
      replay_bounds:{
        rss_bytes:($phases[0].memory.rss_bytes+margin($phases[0].memory.rss_bytes)),
        physical_footprint_bytes:($phases[0].memory.physical_footprint_bytes
          +margin($phases[0].memory.physical_footprint_bytes)),
        wired_bytes:($phases[0].memory.wired_bytes+margin($phases[0].memory.wired_bytes)),
        host_wired_bytes:(([$phases[].memory.host_wired_bytes]|min)
          +$phases[0].artifact.bytes+margin($phases[0].memory.rss_bytes))}
    }
' >"$tmp/runtime.json"

validate_receipt() {
    hf2q_validate_generative_swap_receipt "$1" "$manifest" \
      "$source_commit" "$binary_sha" "$dependency_version" \
      "$dependency_source" "$dependency_checksum"
}
validate_receipt "$tmp/runtime.json"

expect_receipt_rejected() {
    local label=$1 filter=$2
    jq "$filter" "$tmp/runtime.json" >"$tmp/mutated.json"
    if validate_receipt "$tmp/mutated.json"; then
        echo "receipt mutation unexpectedly passed: $label" >&2
        exit 1
    fi
}

expect_receipt_rejected source-commit '.binding.source_commit = ("d" * 40)'
expect_receipt_rejected extra-root-field '.unexpected = true'
expect_receipt_rejected binary-commit '.binding.binary_git_commit = ("d" * 40)'
expect_receipt_rejected dependency-checksum '.binding.dependency.checksum = ("d" * 64)'
expect_receipt_rejected artifact-digest '.artifacts[0].sha256 = ("d" * 64)'
expect_receipt_rejected stale-execution-artifact '.phases[1].execution.artifact_sha256 = .artifacts[0].sha256'
expect_receipt_rejected wrong-family '.phases[4].execution.arch_family = "qwen35"'
expect_receipt_rejected wrong-canary '.phases[3].semantic.content = "HF2Q_SWAP_GEMMA_OK"'
expect_receipt_rejected cached-kv '.phases[4].semantic.cached_tokens = 1'
expect_receipt_rejected reused-generation '.phases[1].resident.generation = .phases[0].resident.generation | .phases[1].execution.generation = .phases[0].execution.generation'
expect_receipt_rejected wrong-pid '.phases[5].process_pid = 43'
expect_receipt_rejected wrong-phase-label '.phases[5].phase = "P05-qwen-dense"'
expect_receipt_rejected process-policy-drift '.phases[5].process_policy_sha256 = ("d" * 64)'
expect_receipt_rejected typed-process-policy-drift '.phases[5].process_policy.queue_capacity = 99'
expect_receipt_rejected q5-route-not-observed '.phases[2].q5_route.route_observed = false'
expect_receipt_rejected non-q5-route-claimed '.phases[0].q5_route = {policy_enabled:true,route:"dense_q5k_canonical_q4x4",route_observed:true}'
expect_receipt_rejected q5-container-without-route-claimed '.phases[4].q5_route = {policy_enabled:true,route:"dense_q5k_canonical_q4x4",route_observed:true}'
expect_receipt_rejected slow-load '.transitions[0].load_seconds = 60'
expect_receipt_rejected coordinated-memory-inflation '.transitions[0].peak_rss_bytes += 999999999999 | .transitions[0].rss_bound_bytes += 999999999999'
expect_receipt_rejected inflated-replay-bound '.replay_bounds.rss_bytes += 999999999999'
expect_receipt_rejected divergent-replay '.phases[4].result_sha256 = ("d" * 64)'
expect_receipt_rejected sequence-substitution '.sequence[1] = "gemma"'
expect_receipt_rejected proof-substitution '.proof.every_required_family = false'
expect_receipt_rejected storage-substitution '.phases[0].storage = "unknown"'
expect_receipt_rejected pool-alias '.phases[1].resident.pool_key_sha256 = .phases[0].resident.pool_key_sha256 | .phases[1].execution.pool_key_sha256 = .phases[0].execution.pool_key_sha256'

runner="$ROOT_DIR/scripts/run_generative_swap_matrix.sh"
rust_gate="$ROOT_DIR/tests/multi_model_swap.rs"
runner_sha=$(hf2q_generative_swap_sha256_file "$runner")
contract_sha=$(hf2q_generative_swap_sha256_file \
  "$ROOT_DIR/scripts/generative_swap_matrix_contract.sh")
manifest_sha=$(hf2q_generative_swap_sha256_file "$manifest")
runtime_sha=$(hf2q_generative_swap_sha256_file "$tmp/runtime.json")
jq \
  --arg runner_sha "$runner_sha" --arg contract_sha "$contract_sha" \
  --arg manifest_sha "$manifest_sha" --arg runtime_sha "$runtime_sha" '
  . + {evidence:{runner_sha256:$runner_sha,contract_sha256:$contract_sha,
    manifest_sha256:$manifest_sha,runtime_receipt_sha256:$runtime_sha}}
' "$tmp/runtime.json" >"$tmp/matrix.json"
hf2q_validate_generative_swap_matrix_receipt "$tmp/matrix.json" "$manifest" \
  "$source_commit" "$binary_sha" "$dependency_version" "$dependency_source" \
  "$dependency_checksum" "$runner_sha" "$contract_sha" "$manifest_sha" \
  "$runtime_sha"
jq '.evidence.runner_sha256 = ("d" * 64)' "$tmp/matrix.json" \
  >"$tmp/mutated-matrix.json"
if hf2q_validate_generative_swap_matrix_receipt "$tmp/mutated-matrix.json" \
  "$manifest" "$source_commit" "$binary_sha" "$dependency_version" \
  "$dependency_source" "$dependency_checksum" "$runner_sha" "$contract_sha" \
  "$manifest_sha" "$runtime_sha"; then
    echo "matrix evidence mutation unexpectedly passed" >&2
    exit 1
fi

grep -Fq 'generative_cross_family_two_cycle_e2e' "$runner"
grep -Fq 'HF2Q_GENERATIVE_SWAP_CHAIN_RECEIPT' "$runner"
# shellcheck disable=SC2016 # source-canary literals, not expansions here
grep -Fq 'GIT_COMMIT_SHA="$source_commit" cargo build' "$runner"
# shellcheck disable=SC2016 # source-canary literals, not expansions here
grep -Fq '"$binary" __build-info' "$runner"
grep -Fq 'hf2q_validate_generative_swap_seal' "$runner"
grep -Fq 'GATE_PORT=52337' "$runner"
grep -Fq 'generative_cross_family_two_cycle_e2e' "$rust_gate"
grep -Fq 'post_inference_with_canary' "$rust_gate"
grep -Fq 'cached_tokens, 0' "$rust_gate"
grep -Fq 'assert_chain_transition_memory' "$rust_gate"
grep -Fq 'process serving policy changed across family replacement' "$rust_gate"
grep -Fq 'Q5_K_M inference encoded no canonical Q5 dispatch' "$rust_gate"
grep -Fq 'MLX_DISP_BUCKET=1' "$runner"
grep -Fq 'HF2Q_Q5K_CANONICAL_Q4X4=1' "$runner"
if grep -Fq 'actual_sha=' "$runner"; then
    echo "runner must consume the retained load-time artifact digest" >&2
    exit 1
fi
if grep -Fq 'HF2Q_HOT_SWAP_E2E_MAX_SECS' "$runner"; then
    echo "generative gate must not accept an ambient timing relaxation" >&2
    exit 1
fi
if grep -Fq 'HF2Q_GENERATIVE_SWAP_MATRIX' "$runner"; then
    echo "generative gate must use the exact source-owned manifest" >&2
    exit 1
fi
if grep -Eq 'sha256_env|HF2Q_SWAP_[A-Z_]+_SHA256' "$manifest"; then
    echo "artifact digests must be immutable manifest data, not ambient inputs" >&2
    exit 1
fi

echo "generative swap matrix contract and mutations passed"
