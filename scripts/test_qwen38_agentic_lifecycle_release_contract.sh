#!/usr/bin/env bash
# Literal release-source needles below intentionally retain shell expressions.
# shellcheck disable=SC2016
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
release_gate="$root_dir/scripts/run_agentic_cache_release_gate.sh"
verifier="$root_dir/scripts/verify_qwen38_agentic_lifecycle_receipt.sh"
# shellcheck source=scripts/agentic_cache_lifecycle_contract.sh
source "$root_dir/scripts/agentic_cache_lifecycle_contract.sh"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-qwen38-agentic-lifecycle.XXXXXX")
trap 'rm -rf -- "$tmp_dir"' EXIT

source_sha=$(printf 'a%.0s' {1..40})
crate_sha=$(printf 'b%.0s' {1..64})
model_sha=$(printf 'd%.0s' {1..64})
dependency_sha=$(printf 'c%.0s' {1..64})
run_id="release-${source_sha:0:12}-qwen38"
model="$tmp_dir/qwen38.gguf"
binary="$tmp_dir/hf2q"
lifecycle="$tmp_dir/summary.json"
manifest="$tmp_dir/manifest.json"
dependency="$tmp_dir/dependency.json"
model_verification="$tmp_dir/model-verification.json"
printf 'model-fixture' >"$model"
printf '#!/usr/bin/env bash\nprintf "hf2q fixture\\n"\n' >"$binary"
chmod 0555 "$binary"
binary_sha=$(shasum -a 256 "$binary" | awk '{print $1}')
model_snapshot=$(stat -f '%d:%i:%z:%m:%c' "$model" 2>/dev/null \
  || stat -c '%d:%i:%s:%Y:%Z' "$model")
jq -n --arg path "$model" --arg sha256 "$model_sha" \
  --arg file_snapshot "$model_snapshot" \
  '{schema_version:1,path:$path,sha256:$sha256,
    file_snapshot:$file_snapshot,content_hash_verified:true,
    run_verification:"fixture"}' >"$model_verification"

write_seal() {
  local payload=$1
  printf '%s  %s\n' \
    "$(shasum -a 256 "$payload" | awk '{print $1}')" \
    "$(basename "$payload")" >"$payload.sha256"
}

write_headers() {
  local destination=$1
  local artifact=${2:-$model_sha}
  local family=${3:-qwen35}
  local architecture=${4:-qwen35}
  local generation=${5:-7}
  printf 'HTTP/1.1 200 OK\r\nx-hf2q-execution-pool-key-b64: cXdlbjM4LXBvb2w=\r\nx-hf2q-execution-generation: %s\r\nx-hf2q-execution-artifact-sha256: %s\r\nx-hf2q-execution-arch-family: %s\r\nx-hf2q-execution-architecture: %s\r\n\r\n' \
    "$generation" "$artifact" "$family" "$architecture" >"$destination"
}

receipts=()
for phase in base seed active_sse sibling isolation; do
  headers="$tmp_dir/$phase.headers"
  receipt="$tmp_dir/$phase.receipt.json"
  write_headers "$headers"
  stream=false
  [[ "$phase" == active_sse ]] && stream=true
  agentic_lifecycle_execution_receipt_json "$headers" \
    "$model_sha" qwen35 qwen35 \
    | jq --arg phase "$phase" --argjson stream "$stream" \
      '. + {phase:$phase,stream:$stream}' >"$receipt"
  receipts+=("$receipt")
done
execution_receipts=$(jq -s '.' "${receipts[@]}")

jq -n \
  --arg run_id "$run_id" \
  --argjson execution_receipts "$execution_receipts" '{
    schema_version:3,status:"pass",model:"qwen38-fixture",
    base_url:"http://127.0.0.1:18083",run_id:$run_id,context_lines:2800,
    continuation_thinking_token_budget:16,
    unrelated_conversation_thinking_enabled:false,
    base_prompt_tokens:12000,seed_cached_tokens:11950,
    active_stream_cancelled_without_done:true,
    queued_exact_retry_cached_tokens:11950,
    unrelated_conversation_cached_tokens:4,
    unrelated_conversation_content:"ISOLATION_OK",
    execution_receipts:$execution_receipts
  }' >"$lifecycle"
write_seal "$lifecycle"

jq -n --arg crate_sha "$crate_sha" --arg checksum "$dependency_sha" '{
  schema_version:1,status:"pass",
  package:{source:"packed-crate",crate_sha256:$crate_sha},
  build:{cargo_target_checkout_disjoint:true,
    rust_build_override_env_cleared:true},
  dependency:{name:"mlx-native",version:"9.8.7",requirement:"=9.8.7",
    source:"registry+https://github.com/rust-lang/crates.io-index",
    checksum:$checksum}
}' >"$dependency"
dependency_receipt_sha=$(shasum -a 256 "$dependency" | awk '{print $1}')
lifecycle_sha=$(shasum -a 256 "$lifecycle" | awk '{print $1}')
model_bytes=$(wc -c <"$model" | tr -d '[:space:]')

write_manifest() {
  local destination=$1
  jq -n \
    --arg source_sha "$source_sha" \
    --arg crate_sha "$crate_sha" \
    --arg binary_sha "$binary_sha" \
    --arg model "$model" \
    --arg model_sha "$model_sha" \
    --arg lifecycle_sha "$lifecycle_sha" \
    --arg dependency_sha "$dependency_receipt_sha" \
    --argjson model_bytes "$model_bytes" \
    --slurpfile dependency "$dependency" \
    --slurpfile lifecycle "$lifecycle" '{
      status:"pass",source_sha:$source_sha,crate_sha256:$crate_sha,
      binary_sha256:$binary_sha,power_guarded_ac:true,
      power_event_snapshots_sha256:("f" * 64),
      dependency_provenance:$dependency[0],
      models:{qwen38:{path:$model,bytes:$model_bytes,sha256:$model_sha}},
      receipt_sha256:{qwen38:{lifecycle:$lifecycle_sha,
        long_decode:("e" * 64)},provenance:{dependency:$dependency_sha}},
      families:{qwen38:{status:"pass",lifecycle:$lifecycle[0],
        long_decode:{status:"pass"}}}
    }' >"$destination"
  write_seal "$destination"
}

write_manifest "$manifest"
bash "$verifier" "$manifest" "$manifest.sha256" \
  "$lifecycle" "$lifecycle.sha256" "$dependency" "$dependency_receipt_sha" \
  "$model_verification" "$source_sha" "$crate_sha" \
  "$binary" "$binary_sha" "$model" "$model_sha"

expect_rejected() {
  local label=$1
  shift
  if "$@" >/dev/null 2>&1; then
    echo "Qwen3.8 lifecycle contract accepted mutation: $label" >&2
    exit 1
  fi
}

wrong_artifact_headers="$tmp_dir/wrong-artifact.headers"
write_headers "$wrong_artifact_headers" "$(printf 'f%.0s' {1..64})"
expect_rejected wrong-executed-artifact \
  agentic_lifecycle_execution_receipt_json "$wrong_artifact_headers" \
  "$model_sha" qwen35 qwen35

wrong_family_headers="$tmp_dir/wrong-family.headers"
write_headers "$wrong_family_headers" "$model_sha" gemma4
expect_rejected wrong-executed-family \
  agentic_lifecycle_execution_receipt_json "$wrong_family_headers" \
  "$model_sha" qwen35 qwen35

wrong_architecture_headers="$tmp_dir/wrong-architecture.headers"
write_headers "$wrong_architecture_headers" "$model_sha" qwen35 gemma4
expect_rejected wrong-executed-architecture \
  agentic_lifecycle_execution_receipt_json "$wrong_architecture_headers" \
  "$model_sha" qwen35 qwen35

missing_header="$tmp_dir/missing-header.headers"
sed '/execution-generation/d' "$tmp_dir/base.headers" >"$missing_header"
expect_rejected missing-generation-header \
  agentic_lifecycle_execution_receipt_json "$missing_header" \
  "$model_sha" qwen35 qwen35

duplicate_header="$tmp_dir/duplicate-header.headers"
sed '/execution-generation/p' "$tmp_dir/base.headers" >"$duplicate_header"
expect_rejected duplicate-generation-header \
  agentic_lifecycle_execution_receipt_json "$duplicate_header" \
  "$model_sha" qwen35 qwen35

jq '.execution_receipts[3].generation = 8' "$lifecycle" \
  >"$tmp_dir/mixed-generation.json"
expect_rejected mixed-execution-generation agentic_lifecycle_validate_summary \
  "$tmp_dir/mixed-generation.json" "$run_id" 2800 "$model_sha" qwen35 qwen35 16 false

jq 'del(.execution_receipts[2])' "$lifecycle" \
  >"$tmp_dir/missing-sse.json"
expect_rejected missing-sse-execution-receipt agentic_lifecycle_validate_summary \
  "$tmp_dir/missing-sse.json" "$run_id" 2800 "$model_sha" qwen35 qwen35 16 false

jq '.execution_receipts[4].pool_key_b64 = "c3RhbGUtcG9vbA=="' "$lifecycle" \
  >"$tmp_dir/mixed-pool.json"
expect_rejected mixed-execution-pool agentic_lifecycle_validate_summary \
  "$tmp_dir/mixed-pool.json" "$run_id" 2800 "$model_sha" qwen35 qwen35 16 false

jq '.schema_version = 2' "$lifecycle" >"$tmp_dir/old-lifecycle-schema.json"
expect_rejected old-lifecycle-schema agentic_lifecycle_validate_summary \
  "$tmp_dir/old-lifecycle-schema.json" "$run_id" 2800 \
  "$model_sha" qwen35 qwen35 16 false

jq 'del(.continuation_thinking_token_budget)' "$lifecycle" \
  >"$tmp_dir/missing-continuation-budget.json"
expect_rejected missing-continuation-budget agentic_lifecycle_validate_summary \
  "$tmp_dir/missing-continuation-budget.json" "$run_id" 2800 \
  "$model_sha" qwen35 qwen35 16 false

jq '.unrelated_conversation_thinking_enabled = true' "$lifecycle" \
  >"$tmp_dir/isolation-thinking-enabled.json"
expect_rejected isolation-thinking-enabled agentic_lifecycle_validate_summary \
  "$tmp_dir/isolation-thinking-enabled.json" "$run_id" 2800 \
  "$model_sha" qwen35 qwen35 16 false

cp "$lifecycle.sha256" "$tmp_dir/bad-summary-sidecar"
printf '\n' >>"$tmp_dir/bad-summary-sidecar"
expect_rejected lifecycle-seal bash "$verifier" \
  "$manifest" "$manifest.sha256" "$lifecycle" "$tmp_dir/bad-summary-sidecar" \
  "$dependency" "$dependency_receipt_sha" "$model_verification" \
  "$source_sha" "$crate_sha" \
  "$binary" "$binary_sha" "$model" "$model_sha"

mutate_manifest() {
  local label=$1
  local filter=$2
  local changed="$tmp_dir/$label.json"
  jq "$filter" "$manifest" >"$changed"
  write_seal "$changed"
  expect_rejected "$label" bash "$verifier" "$changed" "$changed.sha256" \
    "$lifecycle" "$lifecycle.sha256" "$dependency" \
    "$dependency_receipt_sha" \
    "$model_verification" "$source_sha" "$crate_sha" \
    "$binary" "$binary_sha" "$model" "$model_sha"
}

mutate_manifest missing-lifecycle 'del(.families.qwen38.lifecycle)'
mutate_manifest wrong-lifecycle-digest '.receipt_sha256.qwen38.lifecycle = ("0" * 64)'
mutate_manifest embedded-lifecycle-mismatch '.families.qwen38.lifecycle.run_id = "stale"'
mutate_manifest wrong-source '.source_sha = ("0" * 40)'
mutate_manifest wrong-crate '.crate_sha256 = ("0" * 64)'
mutate_manifest wrong-binary '.binary_sha256 = ("0" * 64)'
mutate_manifest wrong-model '.models.qwen38.sha256 = ("0" * 64)'
mutate_manifest wrong-dependency '.dependency_provenance.dependency.version = "9.8.6"'
mutate_manifest unguarded-power '.power_guarded_ac = false'
mutate_manifest missing-existing-long-decode 'del(.families.qwen38.long_decode)'

bad_dependency="$tmp_dir/bad-dependency.json"
jq '.dependency.version = "9.8.6" | .dependency.requirement = "=9.8.6"' \
  "$dependency" >"$bad_dependency"
bad_dependency_sha=$(shasum -a 256 "$bad_dependency" | awk '{print $1}')
jq --slurpfile dependency "$bad_dependency" \
  --arg dependency_sha "$bad_dependency_sha" \
  '.dependency_provenance = $dependency[0]
    | .receipt_sha256.provenance.dependency = $dependency_sha' \
  "$manifest" >"$tmp_dir/self-consistent-bad-dependency.json"
write_seal "$tmp_dir/self-consistent-bad-dependency.json"
expect_rejected self-consistent-wrong-dependency bash "$verifier" \
  "$tmp_dir/self-consistent-bad-dependency.json" \
  "$tmp_dir/self-consistent-bad-dependency.json.sha256" \
  "$lifecycle" "$lifecycle.sha256" "$bad_dependency" \
  "$dependency_receipt_sha" \
  "$model_verification" "$source_sha" "$crate_sha" \
  "$binary" "$binary_sha" "$model" "$model_sha"

cp "$manifest.sha256" "$tmp_dir/bad-manifest-sidecar"
printf '\n' >>"$tmp_dir/bad-manifest-sidecar"
expect_rejected manifest-seal bash "$verifier" \
  "$manifest" "$tmp_dir/bad-manifest-sidecar" \
  "$lifecycle" "$lifecycle.sha256" "$dependency" \
  "$dependency_receipt_sha" \
  "$model_verification" "$source_sha" "$crate_sha" \
  "$binary" "$binary_sha" "$model" "$model_sha"

tampered_binary="$tmp_dir/tampered-hf2q"
cp "$binary" "$tampered_binary"
chmod 0755 "$tampered_binary"
printf '\n' >>"$tampered_binary"
expect_rejected binary-seal bash "$verifier" \
  "$manifest" "$manifest.sha256" "$lifecycle" "$lifecycle.sha256" \
  "$dependency" "$dependency_receipt_sha" "$model_verification" \
  "$source_sha" "$crate_sha" "$tampered_binary" \
  "$binary_sha" "$model" "$model_sha"

# The production snapshot receipt is deliberately a no-second-hash contract.
# Cross a filesystem timestamp tick, then preserve byte length while changing
# content so this mutation is rejected by the recorded file snapshot rather
# than by the manifest's byte-count check.
sleep 1
printf 'MODEL-fixture' >"$model"
expect_rejected model-snapshot-mutation bash "$verifier" \
  "$manifest" "$manifest.sha256" "$lifecycle" "$lifecycle.sha256" \
  "$dependency" "$dependency_receipt_sha" "$model_verification" \
  "$source_sha" "$crate_sha" \
  "$binary" "$binary_sha" "$model" "$model_sha"

validate_release_source() {
  local candidate=$1
  grep -Fq '    qwen38)' "$candidate" \
    && grep -Fq 'launcher_env+=(QWEN38_VISION=off)' "$candidate" \
    && grep -Fq 'verify_model qwen38 "$QWEN38_MODEL" "$QWEN38_MODEL_SHA256"' \
      "$candidate" \
    && grep -Fq 'start_server qwen38 agentic-lifecycle scripts/serve_qwen38_opencode.sh' \
    "$candidate" \
    && grep -Fq 'run_lifecycle qwen38 2800 "$QWEN38_MODEL_SHA256" qwen35 qwen35' \
      "$candidate" \
    && grep -Fq 'run_qwen38_agentic_lifecycle_release_gate' "$candidate" \
    && grep -Fq -- '--arg qwen38_lifecycle_sha' "$candidate" \
    && grep -Fq -- '--slurpfile qwen38_lifecycle' "$candidate" \
    && grep -Fq 'qwen38:{lifecycle:$qwen38_lifecycle_sha' "$candidate" \
    && grep -Fq 'lifecycle:$qwen38_lifecycle[0]' "$candidate" \
    && grep -Fq 'verify_qwen38_agentic_lifecycle_receipt.sh' "$candidate" \
    && grep -Fq '"$dependency_provenance_receipt" "$dependency_provenance_receipt_sha"' \
      "$candidate"
}

validate_release_source "$release_gate"
sed '/run_lifecycle qwen38 2800/d' "$release_gate" >"$tmp_dir/no-fixture-route.sh"
expect_rejected source-without-lifecycle-fixture \
  validate_release_source "$tmp_dir/no-fixture-route.sh"
sed '/--slurpfile qwen38_lifecycle/d' "$release_gate" >"$tmp_dir/no-manifest-input.sh"
expect_rejected source-without-manifest-input \
  validate_release_source "$tmp_dir/no-manifest-input.sh"
sed '/launcher_env+=(QWEN38_VISION=off)/d' "$release_gate" \
  >"$tmp_dir/no-text-only-policy.sh"
expect_rejected source-without-explicit-text-only-policy \
  validate_release_source "$tmp_dir/no-text-only-policy.sh"

printf 'qwen38 agentic lifecycle release contract passed\n'
