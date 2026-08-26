#!/usr/bin/env bash
# Literal source-contract needles intentionally retain shell expressions.
# shellcheck disable=SC2016
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest="$root_dir/data/qwen38_exact_swap_matrix.v1.json"
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$root_dir/scripts/qwen38_artifact_contract.sh"
# shellcheck source=scripts/qwen38_exact_swap_matrix_contract.sh
source "$root_dir/scripts/qwen38_exact_swap_matrix_contract.sh"

qwen38_validate_exact_swap_manifest "$manifest"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qwen38-swap-contract.XXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
mutation_dir="$tmp-mutations"
mkdir -p "$mutation_dir"
trap 'rm -rf -- "$tmp" "$mutation_dir"' EXIT

expect_rejected() {
    local label=$1
    shift
    if "$@" >/dev/null 2>&1; then
        echo "exact Qwen3.8 swap contract accepted mutation: $label" >&2
        exit 1
    fi
}

mutate_manifest() {
    local label=$1 filter=$2 changed
    changed="$mutation_dir/$label.json"
    jq "$filter" "$manifest" >"$changed"
    expect_rejected "$label" qwen38_validate_exact_swap_manifest "$changed"
}

mutate_manifest missing-artifact 'del(.artifacts[0])'
mutate_manifest wrong-artifact-hash '.artifacts[2].sha256 = ("0" * 64)'
mutate_manifest missing-pair 'del(.pairs[4])'
mutate_manifest self-swap '.pairs[0].b = .pairs[0].a'
mutate_manifest replay-not-required '.proof.exact_a_replay = false'

binary_sha=$(printf 'b%.0s' {1..64})
# Synthetic source-only fixture. The production runner derives these fields
# from Cargo.toml/Cargo.lock; a contract test must not invent release authority.
dependency_version=0.13.0
dependency_source='registry+https://github.com/rust-lang/crates.io-index'
dependency_checksum=$(printf 'c%.0s' {1..64})

# Exercise the final verifier against a clean exact Git tree without granting
# the source-only fixture a production bypass. The copied contracts are byte-
# identical to the reviewed source; the tiny manifest/lock pair supplies the
# synthetic registry identity used by this model-free test.
validation_root="$mutation_dir/validation-source"
mkdir -p "$validation_root/scripts" "$validation_root/data"
cp "$root_dir/scripts/qwen38_artifact_contract.sh" \
  "$root_dir/scripts/qwen38_exact_swap_matrix_contract.sh" \
  "$root_dir/scripts/run_qwen38_exact_swap_matrix.sh" \
  "$root_dir/scripts/verify_qwen38_exact_swap_release_receipt.sh" \
  "$validation_root/scripts/"
cp "$manifest" "$validation_root/data/"
printf 'mlx-native = "=%s"\n' "$dependency_version" \
  >"$validation_root/Cargo.toml"
printf 'version = 4\n\n[[package]]\nname = "mlx-native"\nversion = "%s"\nsource = "%s"\nchecksum = "%s"\n' \
  "$dependency_version" "$dependency_source" "$dependency_checksum" \
  >"$validation_root/Cargo.lock"
git -C "$validation_root" init -q
git -C "$validation_root" config user.email hf2q-contract@example.invalid
git -C "$validation_root" config user.name hf2q-contract
git -C "$validation_root" add .
git -C "$validation_root" -c commit.gpgsign=false commit -qm fixture
source_commit=$(git -C "$validation_root" rev-parse HEAD)
binary_git_commit=$source_commit

make_cell() {
    local pair_id=$1 format_a=$2 format_b=$3 ordinal=$4 destination=$5
    local file_a bytes_a sha_a type_a file_b bytes_b sha_b type_b
    local pool_a pool_b result_a result_b gen_a1 gen_b gen_a2
    IFS=$'\t' read -r _format file_a bytes_a sha_a type_a \
        <<<"$(qwen38_artifact_record "$format_a")"
    IFS=$'\t' read -r _format file_b bytes_b sha_b type_b \
        <<<"$(qwen38_artifact_record "$format_b")"
    pool_a=$(printf 'pool:%s' "$format_a" | shasum -a 256 | awk '{print $1}')
    pool_b=$(printf 'pool:%s' "$format_b" | shasum -a 256 | awk '{print $1}')
    result_a=$(printf 'result:%s' "$format_a" | shasum -a 256 | awk '{print $1}')
    result_b=$(printf 'result:%s' "$format_b" | shasum -a 256 | awk '{print $1}')
    gen_a1=$((ordinal * 3 + 1))
    gen_b=$((ordinal * 3 + 2))
    gen_a2=$((ordinal * 3 + 3))
    jq -n \
      --arg pair_id "$pair_id" --arg format_a "$format_a" \
      --arg format_b "$format_b" --arg file_a "$file_a" \
      --argjson bytes_a "$bytes_a" --arg sha_a "$sha_a" \
      --argjson type_a "$type_a" --arg file_b "$file_b" \
      --argjson bytes_b "$bytes_b" --arg sha_b "$sha_b" \
      --argjson type_b "$type_b" --arg source_commit "$source_commit" \
      --arg binary_sha "$binary_sha" --arg binary_git_commit "$binary_git_commit" \
      --arg dependency_version "$dependency_version" \
      --arg dependency_source "$dependency_source" \
      --arg dependency_checksum "$dependency_checksum" \
      --arg pool_a "$pool_a" --arg pool_b "$pool_b" \
      --arg result_a "$result_a" --arg result_b "$result_b" \
      --argjson gen_a1 "$gen_a1" --argjson gen_b "$gen_b" \
      --argjson gen_a2 "$gen_a2" '{
        schema:1,verdict:"pass",pair:{id:$pair_id,a:$format_a,b:$format_b},
        binding:{source_commit:$source_commit,binary_sha256:$binary_sha,
          binary_git_commit:$binary_git_commit,
          dependency:{name:"mlx-native",version:$dependency_version,
            source:$dependency_source,checksum:$dependency_checksum}},
        artifacts:{
          a:{format:$format_a,file:$file_a,bytes:$bytes_a,sha256:$sha_a,
            gguf_file_type:$type_a},
          b:{format:$format_b,file:$file_b,bytes:$bytes_b,sha256:$sha_b,
            gguf_file_type:$type_b}},
        pool_budget_bytes:([$bytes_a,$bytes_b] | max),
        load_budget_seconds:10,
        proof:{one_resident_every_phase:true,
          fresh_generation_per_activation:true,execution_receipts_joined:true,
          bounded_residency:true,no_double_residency:true,
          evicted_artifact_absent:true,exact_a_replay:true},
        phases:[
          {phase:"A1",format:$format_a,
            resident:{pool_key_sha256:$pool_a,generation:$gen_a1,
              bytes:$bytes_a,engine_config_sha256:("1" * 64)},
            execution:{pool_key_sha256:$pool_a,generation:$gen_a1,
              artifact_sha256:$sha_a,arch_family:"qwen35",architecture:"qwen35"},
            result_sha256:$result_a,
            semantic:{role:"assistant",content:"HF2Q_SWAP_OK",finish_reason:"stop",
              completion_tokens:1,cached_tokens:0}},
          {phase:"B",format:$format_b,
            resident:{pool_key_sha256:$pool_b,generation:$gen_b,
              bytes:$bytes_b,engine_config_sha256:("2" * 64)},
            execution:{pool_key_sha256:$pool_b,generation:$gen_b,
              artifact_sha256:$sha_b,arch_family:"qwen35",architecture:"qwen35"},
            result_sha256:$result_b,
            semantic:{role:"assistant",content:"HF2Q_SWAP_OK",finish_reason:"stop",
              completion_tokens:1,cached_tokens:0}},
          {phase:"A2",format:$format_a,
            resident:{pool_key_sha256:$pool_a,generation:$gen_a2,
              bytes:$bytes_a,engine_config_sha256:("1" * 64)},
            execution:{pool_key_sha256:$pool_a,generation:$gen_a2,
              artifact_sha256:$sha_a,arch_family:"qwen35",architecture:"qwen35"},
            result_sha256:$result_a,
            semantic:{role:"assistant",content:"HF2Q_SWAP_OK",finish_reason:"stop",
              completion_tokens:1,cached_tokens:0}}],
        transitions:{
          a_to_b:{load_seconds:1,peak_rss_bytes:11811160064,
            peak_host_wired_bytes:11811160064,
            rss_bound_bytes:12884901888,
            host_wired_bound_bytes:(12884901888 + $bytes_b)},
          b_to_a:{load_seconds:1,peak_rss_bytes:11811160064,
            peak_host_wired_bytes:11811160064,
            rss_bound_bytes:12884901888,
            host_wired_bound_bytes:(12884901888 + $bytes_a)}},
        memory:{
          a1:{rss_bytes:10737418240,physical_footprint_bytes:10737418240,
            physical_footprint_peak_bytes:10737418240,wired_bytes:1073741824,
            host_wired_bytes:10737418240},
          b:{rss_bytes:10737418240,physical_footprint_bytes:10737418240,
            physical_footprint_peak_bytes:10737418240,wired_bytes:1073741824,
            host_wired_bytes:10737418240},
          a2:{rss_bytes:10737418240,physical_footprint_bytes:10737418240,
            physical_footprint_peak_bytes:10737418240,wired_bytes:1073741824,
            host_wired_bytes:10737418240}},
        replay_bounds:{rss_bytes:12884901888,physical_footprint_bytes:12884901888,
          wired_bytes:3221225472,host_wired_bytes:(12884901888 + $bytes_a)},
        storage:{a1_file_backed:true,b:"file_backed",a2_file_backed:true}
      }' >"$destination"
}

cell_paths=()
ordinal=0
while IFS=$'\t' read -r pair_id format_a format_b; do
    cell="$tmp/$pair_id.json"
    make_cell "$pair_id" "$format_a" "$format_b" "$ordinal" "$cell"
    printf 'synthetic source-only execution log for %s\n' "$pair_id" \
      >"$tmp/$pair_id.log"
    qwen38_validate_exact_swap_cell "$cell" "$pair_id" "$format_a" "$format_b" \
      "$source_commit" "$binary_sha" "$binary_git_commit" "$dependency_version" \
      "$dependency_source" "$dependency_checksum"
    cell_paths+=("$cell")
    ordinal=$((ordinal + 1))
done < <(jq -r '.pairs[] | [.id,.a,.b] | @tsv' "$manifest")

chain="$tmp/two-cycle-chain.json"
jq -n \
  --slurpfile catalog "$manifest" \
  --arg source_commit "$source_commit" --arg binary_sha "$binary_sha" \
  --arg binary_git_commit "$binary_git_commit" \
  --arg dependency_version "$dependency_version" \
  --arg dependency_source "$dependency_source" \
  --arg dependency_checksum "$dependency_checksum" '
  ($catalog[0].artifacts) as $artifacts
  | ["BF16","Q4_K_M","BF16","Q5_K_M","BF16","Q6_K","BF16","Q8_0",
      "BF16","Q4_K_M","BF16","Q5_K_M","BF16","Q6_K","BF16","Q8_0",
      "BF16"] as $sequence
  | {
    schema:1,verdict:"pass",gate:"qwen38-exact-five-format-two-cycle",
    binding:{source_commit:$source_commit,binary_sha256:$binary_sha,
      binary_git_commit:$binary_git_commit,
      dependency:{name:"mlx-native",version:$dependency_version,
        source:$dependency_source,checksum:$dependency_checksum}},
    artifacts:$artifacts,
    pool_budget_bytes:([$artifacts[].bytes] | max),load_budget_seconds:10,
    process:{pid:4242},
    sequence:$sequence,
    proof:{one_long_lived_process:true,two_complete_cycles:true,
      fresh_generation_every_activation:true,cold_generation_cache:true,
      execution_receipts_joined:true,bounded_every_transition:true,
      evicted_artifacts_absent:true,exact_bf16_replay:true},
    phases:[range(0;17) as $i
      | $sequence[$i] as $format
      | ($artifacts[] | select(.format == $format)) as $artifact
      | {index:$i,process_pid:4242,phase:("P" + ($i|tostring)),
        format:$format,artifact:$artifact,
        resident:{pool_key_sha256:$artifact.sha256,generation:($i + 1),
          bytes:$artifact.bytes,engine_config_sha256:("e" * 64)},
        execution:{pool_key_sha256:$artifact.sha256,generation:($i + 1),
          artifact_sha256:$artifact.sha256,arch_family:"qwen35",architecture:"qwen35"},
        result_sha256:("d" * 64),
        semantic:{role:"assistant",content:"HF2Q_SWAP_OK",finish_reason:"stop",
          completion_tokens:1,cached_tokens:0},
        memory:{rss_bytes:10737418240,physical_footprint_bytes:10737418240,
          physical_footprint_peak_bytes:10737418240,wired_bytes:1073741824,
          host_wired_bytes:10737418240},storage:"file_backed"}],
    transitions:[range(0;16) as $i
      | ($artifacts[] | select(.format == $sequence[$i + 1])) as $after
      | {index:$i,from:$sequence[$i],to:$sequence[$i + 1],load_seconds:1,
        peak_rss_bytes:11811160064,peak_host_wired_bytes:11811160064,
        rss_bound_bytes:12884901888,
        host_wired_bound_bytes:(12884901888 + $after.bytes)}],
    cycle_replay_phase_indexes:[8,16],
    replay_bounds:{rss_bytes:12884901888,physical_footprint_bytes:12884901888,
      wired_bytes:3221225472,
      host_wired_bytes:(12884901888 + $artifacts[0].bytes)}
  }' >"$chain"
printf 'synthetic source-only long-lived chain log\n' >"$tmp/two-cycle-chain.log"
qwen38_validate_exact_swap_chain "$chain" "$source_commit" "$binary_sha" \
  "$binary_git_commit" "$dependency_version" "$dependency_source" \
  "$dependency_checksum"

mkdir -p "$tmp/preflight"
for format in $(qwen38_artifact_formats); do
    IFS=$'\t' read -r _format relative_file _bytes artifact_sha _type \
      <<<"$(qwen38_artifact_record "$format")"
    receipt_slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
    jq -n --arg path "/qualified/$relative_file" --arg sha "$artifact_sha" '{
      schema_version:2,path:$path,sha256:$sha,file_snapshot:"1:2:3:4:5",
      file_stamp:{device:1,inode:2,bytes:3,modified_seconds:4,
        modified_nanoseconds:0,changed_seconds:5,changed_nanoseconds:0},
      content_hash_verified:true,run_verification:"cached_unchanged_file"
    }' >"$tmp/preflight/$receipt_slug.json"
done

results=$(jq -s . "${cell_paths[@]}")
chain_result=$(jq -c . "$chain")
jq -n \
  --arg source_commit "$source_commit" --arg binary_sha "$binary_sha" \
  --arg binary_git_commit "$binary_git_commit" \
  --arg dependency_version "$dependency_version" \
  --arg dependency_source "$dependency_source" \
  --arg dependency_checksum "$dependency_checksum" \
  --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
  --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
  --arg runner_sha "$(shasum -a 256 "$root_dir/scripts/run_qwen38_exact_swap_matrix.sh" | awk '{print $1}')" \
  --arg matrix_contract_sha "$(shasum -a 256 "$root_dir/scripts/qwen38_exact_swap_matrix_contract.sh" | awk '{print $1}')" \
  --arg artifact_contract_sha "$(shasum -a 256 "$root_dir/scripts/qwen38_artifact_contract.sh" | awk '{print $1}')" \
  --arg manifest_sha "$(shasum -a 256 "$manifest" | awk '{print $1}')" \
  --argjson results "$results" --argjson chain "$chain_result" '{
    schema:1,verdict:"pass",gate:"qwen38-exact-model-swap-matrix",
    source_commit:$source_commit,
    binary:{sha256:$binary_sha,git_commit:$binary_git_commit},
    dependency:{name:"mlx-native",version:$dependency_version,
      source:$dependency_source,checksum:$dependency_checksum},
    repository:$repository,revision:$revision,
    architecture:"qwen35",arch_family:"qwen35",
    formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],
    evidence:{runner_sha256:$runner_sha,
      matrix_contract_sha256:$matrix_contract_sha,
      artifact_contract_sha256:$artifact_contract_sha,
      manifest_sha256:$manifest_sha},results:$results,chain:$chain
  }' >"$tmp/matrix.json"

: >"$tmp/evidence.sha256"
for path in "${cell_paths[@]}"; do
    pair_id=$(basename "$path" .json)
    for suffix in json log; do
        printf '%s  %s\n' \
          "$(shasum -a 256 "$tmp/$pair_id.$suffix" | awk '{print $1}')" \
          "$pair_id.$suffix" >>"$tmp/evidence.sha256"
    done
done
for file in two-cycle-chain.json two-cycle-chain.log; do
    printf '%s  %s\n' \
      "$(shasum -a 256 "$tmp/$file" | awk '{print $1}')" "$file" \
      >>"$tmp/evidence.sha256"
done
for receipt in "$tmp"/preflight/*.json; do
    printf '%s  %s\n' \
      "$(shasum -a 256 "$receipt" | awk '{print $1}')" \
      "preflight/$(basename "$receipt")" >>"$tmp/evidence.sha256"
done
sort -o "$tmp/evidence.sha256" "$tmp/evidence.sha256"
printf '%s  matrix.json\n%s  evidence.sha256\n' \
  "$(shasum -a 256 "$tmp/matrix.json" | awk '{print $1}')" \
  "$(shasum -a 256 "$tmp/evidence.sha256" | awk '{print $1}')" \
  >"$tmp/result.sha256"
qwen38_validate_exact_swap_seal "$tmp/matrix.json"

cp "$tmp/preflight/bf16.json" "$mutation_dir/preflight-bf16.json"
cp "$tmp/evidence.sha256" "$mutation_dir/preflight-evidence.sha256"
cp "$tmp/result.sha256" "$mutation_dir/preflight-result.sha256"
jq '.sha256 = ("0" * 64)' "$tmp/preflight/bf16.json" \
  >"$tmp/preflight/bf16.json.tmp"
mv "$tmp/preflight/bf16.json.tmp" "$tmp/preflight/bf16.json"
mutated_preflight_sha=$(shasum -a 256 "$tmp/preflight/bf16.json" | awk '{print $1}')
awk -v digest="$mutated_preflight_sha" '
  $2 == "preflight/bf16.json" { print digest "  " $2; next }
  { print }
' "$tmp/evidence.sha256" >"$tmp/evidence.sha256.tmp"
mv "$tmp/evidence.sha256.tmp" "$tmp/evidence.sha256"
printf '%s  matrix.json\n%s  evidence.sha256\n' \
  "$(shasum -a 256 "$tmp/matrix.json" | awk '{print $1}')" \
  "$(shasum -a 256 "$tmp/evidence.sha256" | awk '{print $1}')" \
  >"$tmp/result.sha256"
expect_rejected preflight-identity-drift \
  qwen38_validate_exact_swap_seal "$tmp/matrix.json"
cp "$mutation_dir/preflight-bf16.json" "$tmp/preflight/bf16.json"
cp "$mutation_dir/preflight-evidence.sha256" "$tmp/evidence.sha256"
cp "$mutation_dir/preflight-result.sha256" "$tmp/result.sha256"

mkdir -p "$mutation_dir/protected"
release_manifest="$mutation_dir/protected/release-manifest.json"
jq -n \
  --arg source_sha "$source_commit" --arg binary_sha "$binary_sha" \
  --arg matrix_sha "$(shasum -a 256 "$tmp/matrix.json" | awk '{print $1}')" \
  --arg dependency_version "$dependency_version" \
  --arg dependency_source "$dependency_source" \
  --arg dependency_checksum "$dependency_checksum" \
  --slurpfile matrix "$tmp/matrix.json" '{
    status:"pass",source_sha:$source_sha,binary_sha256:$binary_sha,
    dependency_provenance:{dependency:{name:"mlx-native",
      version:$dependency_version,source:$dependency_source,
      checksum:$dependency_checksum}},
    receipt_sha256:{qwen38:{exact_swap:$matrix_sha}},
    families:{qwen38:{status:"pass",exact_swap:$matrix[0]}}
  }' >"$release_manifest"
printf '%s  %s\n' \
  "$(shasum -a 256 "$release_manifest" | awk '{print $1}')" \
  "$(basename "$release_manifest")" >"$release_manifest.sha256"
bash "$root_dir/scripts/verify_qwen38_exact_swap_release_receipt.sh" \
  "$release_manifest" "$release_manifest.sha256" "$tmp/matrix.json" \
  "$source_commit" "$binary_sha" "$dependency_version" \
  "$dependency_source" "$dependency_checksum" "$validation_root"
printf 'dirty\n' >"$validation_root/untracked-proof-input"
expect_rejected dirty-protected-source \
  bash "$root_dir/scripts/verify_qwen38_exact_swap_release_receipt.sh" \
  "$release_manifest" "$release_manifest.sha256" "$tmp/matrix.json" \
  "$source_commit" "$binary_sha" "$dependency_version" \
  "$dependency_source" "$dependency_checksum" "$validation_root"
rm -f -- "$validation_root/untracked-proof-input"

mutate_receipt() {
    local label=$1 filter=$2 changed
    changed="$mutation_dir/$label.receipt.json"
    jq "$filter" "$tmp/matrix.json" >"$changed"
    expect_rejected "$label" qwen38_validate_exact_swap_receipt "$changed"
}

mutate_receipt wrong-source '.source_commit = ("0" * 40)'
mutate_receipt wrong-binary '.binary.sha256 = ("0" * 64)'
mutate_receipt wrong-embedded-commit '
  .binary.git_commit = ("0" * 40)
  | .results[].binding.binary_git_commit = ("0" * 40)
  | .chain.binding.binary_git_commit = ("0" * 40)'
mutate_receipt stale-generation '.results[1].phases[1].execution.generation += 1'
mutate_receipt missing-cell 'del(.results[4])'
mutate_receipt replay-diverged '.results[2].phases[2].result_sha256 = ("0" * 64)'
mutate_receipt double-residency-unproven '.results[0].proof.no_double_residency = false'
mutate_receipt peak-over-bound '.results[0].transitions.a_to_b.peak_rss_bytes
  = (.results[0].transitions.a_to_b.rss_bound_bytes + 1)'
mutate_receipt coordinated-pair-bound-inflation '
  .results[0].transitions.a_to_b.peak_rss_bytes += 4294967296
  | .results[0].transitions.a_to_b.rss_bound_bytes += 4294967296
  | .results[0].transitions.a_to_b.peak_host_wired_bytes += 4294967296
  | .results[0].transitions.a_to_b.host_wired_bound_bytes += 4294967296'
mutate_receipt wrong-artifact '.results[3].artifacts.b.sha256 = ("0" * 64)'
mutate_receipt mutable-load-budget '.results[0].load_budget_seconds = 600'
mutate_receipt load-over-fixed-budget '.results[0].transitions.a_to_b.load_seconds = 11'
mutate_receipt empty-sentinel '.results[0].phases[0].semantic.content = ""'
mutate_receipt wrong-role '.results[0].phases[0].semantic.role = "tool"'
mutate_receipt wrong-finish '.results[0].phases[0].semantic.finish_reason = "length"'
mutate_receipt zero-completion-tokens '.results[0].phases[0].semantic.completion_tokens = 0'
mutate_receipt warm-generation-cache '.results[0].phases[0].semantic.cached_tokens = 1'
mutate_receipt chain-not-long-lived '.chain.proof.one_long_lived_process = false'
mutate_receipt chain-process-changed '.chain.phases[7].process_pid = 4243'
mutate_receipt chain-missing-phase 'del(.chain.phases[10])'
mutate_receipt chain-reused-generation '.chain.phases[10].resident.generation
  = .chain.phases[9].resident.generation
  | .chain.phases[10].execution.generation = .chain.phases[9].execution.generation'
mutate_receipt coordinated-chain-bound-inflation '
  .chain.transitions[0].peak_rss_bytes += 4294967296
  | .chain.transitions[0].rss_bound_bytes += 4294967296
  | .chain.transitions[0].peak_host_wired_bytes += 4294967296
  | .chain.transitions[0].host_wired_bound_bytes += 4294967296'
mutate_receipt chain-warm-generation-cache '.chain.phases[6].semantic.cached_tokens = 1'

mutate_release_manifest() {
    local label=$1 filter=$2 changed
    changed="$mutation_dir/$label.release.json"
    jq "$filter" "$release_manifest" >"$changed"
    printf '%s  %s\n' \
      "$(shasum -a 256 "$changed" | awk '{print $1}')" \
      "$(basename "$changed")" >"$changed.sha256"
    expect_rejected "$label" \
      bash "$root_dir/scripts/verify_qwen38_exact_swap_release_receipt.sh" \
      "$changed" "$changed.sha256" "$tmp/matrix.json" \
      "$source_commit" "$binary_sha" "$dependency_version" \
      "$dependency_source" "$dependency_checksum" "$validation_root"
}

mutate_release_manifest missing-protected-swap 'del(.families.qwen38.exact_swap)'
mutate_release_manifest wrong-protected-swap-digest \
  '.receipt_sha256.qwen38.exact_swap = ("0" * 64)'
mutate_release_manifest embedded-protected-swap-drift \
  '.families.qwen38.exact_swap.chain.phases[0].semantic.content = ""'
mutate_release_manifest protected-dependency-drift \
  '.dependency_provenance.dependency.version = "0.0.0"'

cp "$tmp/evidence.sha256" "$mutation_dir/evidence.sha256"
cp "$tmp/result.sha256" "$mutation_dir/result.sha256"
sed -n '2,$p' "$tmp/evidence.sha256" >"$tmp/evidence.sha256.missing"
mv "$tmp/evidence.sha256.missing" "$tmp/evidence.sha256"
printf '%s  matrix.json\n%s  evidence.sha256\n' \
  "$(shasum -a 256 "$tmp/matrix.json" | awk '{print $1}')" \
  "$(shasum -a 256 "$tmp/evidence.sha256" | awk '{print $1}')" \
  >"$tmp/result.sha256"
expect_rejected missing-evidence-cell \
  qwen38_validate_exact_swap_seal "$tmp/matrix.json"
cp "$mutation_dir/evidence.sha256" "$tmp/evidence.sha256"
cp "$mutation_dir/result.sha256" "$tmp/result.sha256"

printf 'tamper\n' >>"$tmp/bf16--q4_k_m.log"
expect_rejected evidence-tamper \
  qwen38_validate_exact_swap_seal "$tmp/matrix.json"

# shellcheck disable=SC2016 # source-canary literal, not an expansion
grep -Fq 'HF2Q_HOT_SWAP_EXACT_RECEIPT="$cell"' \
  "$root_dir/scripts/run_qwen38_exact_swap_matrix.sh"
grep -Fq 'CARGO_HOME="$GATE_CARGO_HOME" GIT_COMMIT_SHA="$source_commit"' \
  "$root_dir/scripts/run_qwen38_exact_swap_matrix.sh"
grep -Fq 'qwen38_reject_cargo_configuration "$root_dir" "$GATE_CARGO_HOME"' \
  "$root_dir/scripts/run_qwen38_exact_swap_matrix.sh"
grep -Fq 'build_info=$("$binary" __build-info)' \
  "$root_dir/scripts/run_qwen38_exact_swap_matrix.sh"
grep -Fq 'sealed_binary_dir=$(mktemp -d' \
  "$root_dir/scripts/run_qwen38_exact_swap_matrix.sh"
grep -Fq 'cp "$binary" "$sealed_binary"' \
  "$root_dir/scripts/run_qwen38_exact_swap_matrix.sh"
grep -Fq 'binary=$sealed_binary' \
  "$root_dir/scripts/run_qwen38_exact_swap_matrix.sh"
grep -Fq 'HF2Q_HOT_SWAP_E2E_MAX_SECS=10' \
  "$root_dir/scripts/run_qwen38_exact_swap_matrix.sh"
grep -Fq 'qwen38_exact_five_format_two_cycle_e2e' \
  "$root_dir/scripts/run_qwen38_exact_swap_matrix.sh"
grep -Fq 'write_exact_receipt(std::path::Path::new(&receipt_path), &receipt)' \
  "$root_dir/tests/multi_model_swap.rs"
grep -Fq 'sha256_file(&executed_binary).expect("hash executed hf2q binary")' \
  "$root_dir/tests/multi_model_swap.rs"
grep -Fq 'const SWAP_SENTINEL: &str = "HF2Q_SWAP_OK";' \
  "$root_dir/tests/multi_model_swap.rs"
grep -Fq 'bash scripts/test_qwen38_exact_swap_matrix_contract.sh' \
  "$root_dir/.github/workflows/ci.yml"
validate_protected_source() {
    local release_gate=$1 workflow=$2
    grep -Fxq 'run_qwen38_exact_swap_release_gate' "$release_gate" \
      && grep -Fq -- '--arg qwen38_exact_swap_sha' "$release_gate" \
      && grep -Fq -- '--slurpfile qwen38_exact_swap' "$release_gate" \
      && grep -Fq 'exact_swap:$qwen38_exact_swap_sha' "$release_gate" \
      && grep -Fq 'exact_swap:$qwen38_exact_swap[0]' "$release_gate" \
      && grep -Fq 'verify_qwen38_exact_swap_release_receipt.sh' "$release_gate" \
      && grep -Fq 'HF2Q_SOURCE_ROOT="$GITHUB_WORKSPACE"' "$workflow"
}

release_gate="$root_dir/scripts/run_agentic_cache_release_gate.sh"
workflow="$root_dir/.github/workflows/cache-lifecycle.yml"
validate_protected_source "$release_gate" "$workflow"
sed '/^run_qwen38_exact_swap_release_gate$/d' "$release_gate" \
  >"$mutation_dir/protected-without-runtime-gate.sh"
expect_rejected protected-without-runtime-gate validate_protected_source \
  "$mutation_dir/protected-without-runtime-gate.sh" "$workflow"
sed '/--slurpfile qwen38_exact_swap/d' "$release_gate" \
  >"$mutation_dir/protected-without-embedded-swap.sh"
expect_rejected protected-without-embedded-swap validate_protected_source \
  "$mutation_dir/protected-without-embedded-swap.sh" "$workflow"
sed '/HF2Q_SOURCE_ROOT="\$GITHUB_WORKSPACE"/d' "$workflow" \
  >"$mutation_dir/protected-without-exact-source.yml"
expect_rejected protected-without-exact-source validate_protected_source \
  "$release_gate" "$mutation_dir/protected-without-exact-source.yml"

echo "exact Qwen3.8 swap matrix contract fixtures passed"
