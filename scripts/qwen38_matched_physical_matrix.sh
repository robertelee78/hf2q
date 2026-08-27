#!/usr/bin/env bash
set -euo pipefail

# Fail-closed five-artifact by five-width matched physical ABBA authority.
# Every child is independently sealed; this runner publishes only after the
# exact physical matrix, all 25 matched cells, and every identity join pass.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT_DIR="$ROOT_DIR/scripts"
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$SCRIPT_DIR/qwen38_artifact_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$SCRIPT_DIR/qwen36_watchdog_validate.sh"
# shellcheck source=scripts/macos_runtime_identity.sh
source "$SCRIPT_DIR/macos_runtime_identity.sh"
# shellcheck source=scripts/macos_thermal_guard.sh
source "$SCRIPT_DIR/macos_thermal_guard.sh"
# shellcheck source=scripts/qwen38_matched_reference_contract.sh
source "$SCRIPT_DIR/qwen38_matched_reference_contract.sh"
# shellcheck source=scripts/qwen38_matched_physical_contract.sh
source "$SCRIPT_DIR/qwen38_matched_physical_contract.sh"

HF2Q_BIN=${HF2Q_BIN:?HF2Q_BIN is required}
HF2Q_SOURCE_DIR=${HF2Q_SOURCE_DIR:?HF2Q_SOURCE_DIR is required}
HF2Q_COMMIT=${HF2Q_COMMIT:?HF2Q_COMMIT is required}
HF2Q_SHA256=${HF2Q_SHA256:?HF2Q_SHA256 is required}
REFERENCE_BIN=${REFERENCE_BIN:?REFERENCE_BIN is required}
REFERENCE_SOURCE_DIR=${REFERENCE_SOURCE_DIR:?REFERENCE_SOURCE_DIR is required}
REFERENCE_COMMIT=${REFERENCE_COMMIT:?REFERENCE_COMMIT is required}
REFERENCE_SHA256=${REFERENCE_SHA256:?REFERENCE_SHA256 is required}
REFERENCE_RUNTIME_MANIFEST_SHA256=${REFERENCE_RUNTIME_MANIFEST_SHA256:?REFERENCE_RUNTIME_MANIFEST_SHA256 is required}
MODEL_ROOT=${MODEL_ROOT:?MODEL_ROOT is required}
PHYSICAL_MATRIX_RECEIPT=${PHYSICAL_MATRIX_RECEIPT:?PHYSICAL_MATRIX_RECEIPT is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
PORT=${PORT:-18096}
MIN_HF2Q_RATIO=${MIN_HF2Q_RATIO:-1.0}
MAX_LAUNCH_SKEW_SECONDS=${MAX_LAUNCH_SKEW_SECONDS:-$QWEN38_MATCHED_MAX_LAUNCH_SKEW_SECONDS}
KV_CACHE_BUDGET_BYTES=${KV_CACHE_BUDGET_BYTES:-$QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES}

for command in awk cp find git grep jq otool ps realpath shasum sort stat tr; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ "$HF2Q_SOURCE_DIR" == /* && "$REFERENCE_SOURCE_DIR" == /* \
  && "$MODEL_ROOT" == /* && "$OUT_DIR" == /* ]] || {
    echo "source directories, MODEL_ROOT, and OUT_DIR must be absolute" >&2
    exit 2
}
[[ ! -e "$OUT_DIR" \
  || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "matched physical matrix output must be fresh: $OUT_DIR" >&2
    exit 2
}
[[ "$KV_CACHE_BUDGET_BYTES" == \
  "$QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES" ]] || {
    echo "matched physical matrix requires the canonical KV-cache budget" >&2
    exit 2
}
awk -v actual="$MAX_LAUNCH_SKEW_SECONDS" \
  -v required="$QWEN38_MATCHED_MAX_LAUNCH_SKEW_SECONDS" \
  'BEGIN { exit !(actual == required) }' || {
    echo "matched physical matrix requires the canonical launch-skew ceiling" >&2
    exit 2
}
case "$OUT_DIR" in
    "$ROOT_DIR"|"$ROOT_DIR"/*|"$HF2Q_SOURCE_DIR"|"$HF2Q_SOURCE_DIR"/*|\
    "$REFERENCE_SOURCE_DIR"|"$REFERENCE_SOURCE_DIR"/*)
        echo "evidence must live outside all source worktrees" >&2
        exit 2
        ;;
esac
[[ -x "$HF2Q_BIN" && -x "$REFERENCE_BIN" \
  && -x "$HF2Q_SOURCE_DIR/scripts/serve_qwen38_opencode.sh" \
  && -f "$PHYSICAL_MATRIX_RECEIPT" ]] || exit 2
for commit in "$HF2Q_COMMIT" "$REFERENCE_COMMIT"; do
    [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || {
        echo "source commits must be exact lowercase digests" >&2
        exit 2
    }
done
for digest in "$HF2Q_SHA256" "$REFERENCE_SHA256" \
  "$REFERENCE_RUNTIME_MANIFEST_SHA256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
        echo "artifact identities must be lowercase SHA-256 digests" >&2
        exit 2
    }
done
harness_commit=$(git -C "$ROOT_DIR" rev-parse HEAD)
matched_physical_require_clean_exact_source "$ROOT_DIR" "$harness_commit" harness
matched_physical_require_clean_exact_source \
  "$HF2Q_SOURCE_DIR" "$HF2Q_COMMIT" hf2q
matched_physical_require_clean_exact_source \
  "$REFERENCE_SOURCE_DIR" "$REFERENCE_COMMIT" reference
[[ "$(shasum -a 256 "$HF2Q_BIN" | awk '{print $1}')" == \
  "$HF2Q_SHA256" ]] || exit 2
grep -aFq "$HF2Q_COMMIT" "$HF2Q_BIN" || {
    echo "hf2q binary does not embed $HF2Q_COMMIT" >&2
    exit 2
}
reference_version=$("$REFERENCE_BIN" --version 2>&1)
[[ "$reference_version" == *"${REFERENCE_COMMIT:0:9}"* \
  && "$(shasum -a 256 "$REFERENCE_BIN" | awk '{print $1}')" == \
    "$REFERENCE_SHA256" ]] || {
    echo "reference binary identity mismatch" >&2
    exit 2
}
reference_runtime_manifest=$(hf2q_macos_runtime_manifest "$REFERENCE_BIN")
reference_runtime_manifest_sha=$(printf '%s\n' "$reference_runtime_manifest" \
  | shasum -a 256 | awk '{print $1}')
[[ "$reference_runtime_manifest_sha" == \
  "$REFERENCE_RUNTIME_MANIFEST_SHA256" ]] || {
    echo "reference runtime closure mismatch: expected=$REFERENCE_RUNTIME_MANIFEST_SHA256 actual=$reference_runtime_manifest_sha" >&2
    exit 2
}
export HF2Q_MODEL_VERIFICATION_BINARY="$HF2Q_BIN"
qwen38_validate_pinned_peer_commit "$REFERENCE_COMMIT"
qwen38_validate_physical_matrix_seal \
  "$PHYSICAL_MATRIX_RECEIPT" "$HF2Q_SOURCE_DIR" "$HF2Q_BIN"

mkdir -p "$OUT_DIR/artifacts" "$OUT_DIR/preflight"
qwen38_copy_physical_matrix_seal "$PHYSICAL_MATRIX_RECEIPT" \
  "$OUT_DIR/physical-proof"
physical_receipt="$OUT_DIR/physical-proof/matrix.json"
physical_matrix_sha=$(shasum -a 256 "$physical_receipt" | awk '{print $1}')
physical_binary_sha=$(jq -er '.results[0].binary.sha256' \
  "$physical_receipt")
physical_source_commit=$(jq -er '.binding.source_commit' \
  "$physical_receipt")
[[ "$physical_binary_sha" == "$HF2Q_SHA256" ]] || {
    echo "physical matrix and matched binary identities differ" >&2
    exit 2
}
[[ "$physical_source_commit" == "$HF2Q_COMMIT" ]] || {
    echo "physical matrix and matched source commits differ" >&2
    exit 2
}

child_pid=''
cleanup_child() {
    local original_rc=$?
    trap - EXIT
    matched_physical_terminate_owned_child "$child_pid"
    exit "$original_rc"
}
trap cleanup_child EXIT
trap 'exit 130' INT TERM

# Verify the whole catalog before the first model load, then reuse exactly one
# snapshot-bound verification receipt per artifact across its five widths.
for format in $(qwen38_artifact_formats); do
    IFS=$'\t' read -r _format relative_file _bytes sha256 file_type \
      <<<"$(qwen38_artifact_record "$format")"
    model_path="$MODEL_ROOT/$relative_file"
    receipt="$OUT_DIR/preflight/$(printf '%s' "$format" \
      | tr '[:upper:]' '[:lower:]').json"
    [[ -f "$model_path" && -r "$model_path" ]] || {
        echo "qualified $format artifact is missing: $model_path" >&2
        exit 2
    }
    hf2q_release_prepare_model_verification "$model_path" "$sha256" "$receipt" \
      "${HF2Q_MODEL_VERIFICATION_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}/hf2q/model-verification}"
    actual_bytes=$(stat -f '%z' "$model_path" 2>/dev/null \
      || stat -c '%s' "$model_path")
    qwen38_validate_artifact_identity "$format" "$sha256" "$actual_bytes" \
      "$file_type"
done

child_summaries=()
child_seals=()
for format in $(qwen38_artifact_formats); do
    IFS=$'\t' read -r _format relative_file bytes sha256 _file_type \
      <<<"$(qwen38_artifact_record "$format")"
    slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
    child_dir="$OUT_DIR/artifacts/$slug"
    physical_receipt="$OUT_DIR/physical-proof/matrix.json"
    preflight_receipt="$OUT_DIR/preflight/$slug.json"
    HF2Q_BIN="$HF2Q_BIN" \
    HF2Q_SOURCE_DIR="$HF2Q_SOURCE_DIR" \
    HF2Q_COMMIT="$HF2Q_COMMIT" \
    HF2Q_SHA256="$HF2Q_SHA256" \
    REFERENCE_BIN="$REFERENCE_BIN" \
    REFERENCE_SOURCE_DIR="$REFERENCE_SOURCE_DIR" \
    REFERENCE_COMMIT="$REFERENCE_COMMIT" \
    REFERENCE_SHA256="$REFERENCE_SHA256" \
    REFERENCE_RUNTIME_MANIFEST_SHA256="$REFERENCE_RUNTIME_MANIFEST_SHA256" \
    MODEL_PATH="$MODEL_ROOT/$relative_file" \
    MODEL_SHA256="$sha256" \
    MODEL_FORMAT="$format" \
    MODEL_BYTES="$bytes" \
    PHYSICAL_MATRIX_RECEIPT="$physical_receipt" \
    PHYSICAL_MATRIX_SHA256="$physical_matrix_sha" \
    HF2Q_MODEL_VERIFICATION_RECEIPT="$preflight_receipt" \
    OUT_DIR="$child_dir" \
    PORT="$PORT" \
    MIN_HF2Q_RATIO="$MIN_HF2Q_RATIO" \
    KV_CACHE_BUDGET_BYTES="$KV_CACHE_BUDGET_BYTES" \
    MAX_LAUNCH_SKEW_SECONDS="$MAX_LAUNCH_SKEW_SECONDS" \
      "$SCRIPT_DIR/qwen38_matched_physical_abba.sh" &
    child_pid=$!
    child_rc=0
    wait "$child_pid" || child_rc=$?
    child_pid=''
    if ((child_rc != 0)); then exit "$child_rc"; fi
    matched_physical_validate_reopened_child "$child_dir"
    matched_physical_validate_expected_reference_closure \
      "$child_dir/summary.json" "$REFERENCE_RUNTIME_MANIFEST_SHA256"
    child_summaries+=("$child_dir/summary.json")
    child_seals+=("$(jq -nc --arg format "$format" \
      --arg path "artifacts/$slug" \
      --arg summary "$(shasum -a 256 "$child_dir/summary.json" | awk '{print $1}')" \
      --arg evidence "$(shasum -a 256 "$child_dir/evidence.sha256" | awk '{print $1}')" \
      --arg result "$(shasum -a 256 "$child_dir/result.sha256" | awk '{print $1}')" \
      '{format:$format,path:$path,summary_sha256:$summary,
        evidence_manifest_sha256:$evidence,result_seal_sha256:$result}')")
done

matched_physical_require_clean_exact_source "$ROOT_DIR" "$harness_commit" harness
matched_physical_require_clean_exact_source \
  "$HF2Q_SOURCE_DIR" "$HF2Q_COMMIT" hf2q
matched_physical_require_clean_exact_source \
  "$REFERENCE_SOURCE_DIR" "$REFERENCE_COMMIT" reference
hf2q_macos_verify_runtime_manifest "$REFERENCE_BIN" \
  "$reference_runtime_manifest"
reference_runtime_manifest_final=$(hf2q_macos_runtime_manifest "$REFERENCE_BIN")
reference_runtime_manifest_final_sha=$(printf '%s\n' \
  "$reference_runtime_manifest_final" | shasum -a 256 | awk '{print $1}')
[[ "$reference_runtime_manifest_final_sha" == \
  "$REFERENCE_RUNTIME_MANIFEST_SHA256" ]] || {
    echo "reference runtime closure changed before matrix sealing" >&2
    exit 2
}

results=$(jq -s . "${child_summaries[@]}")
sealed_children=$(printf '%s\n' "${child_seals[@]}" | jq -s .)
script_sha=$(shasum -a 256 "$SCRIPT_DIR/qwen38_matched_physical_matrix.sh" \
  | awk '{print $1}')
contract_sha=$(shasum -a 256 "$SCRIPT_DIR/qwen38_matched_physical_contract.sh" \
  | awk '{print $1}')
artifact_contract_sha=$(shasum -a 256 "$SCRIPT_DIR/qwen38_artifact_contract.sh" \
  | awk '{print $1}')

jq -n --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
  --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
  --arg harness_commit "$harness_commit" \
  --arg reference_commit "$REFERENCE_COMMIT" \
  --arg reference_runtime_manifest_sha \
    "$REFERENCE_RUNTIME_MANIFEST_SHA256" \
  --arg physical_sha "$physical_matrix_sha" \
  --arg physical_binary_sha "$physical_binary_sha" \
  --arg physical_source_commit "$physical_source_commit" \
  --arg script_sha "$script_sha" --arg contract_sha "$contract_sha" \
  --arg artifact_contract_sha "$artifact_contract_sha" \
  --arg hf2q_speculation "$QWEN38_MATCHED_HF2Q_SPECULATION_POLICY" \
  --arg reference_speculation "$QWEN38_MATCHED_REFERENCE_SPECULATION_POLICY" \
  --arg hf2q_kv "$QWEN38_MATCHED_HF2Q_KV_CACHE" \
  --arg reference_k "$QWEN38_MATCHED_REFERENCE_KV_CACHE_K" \
  --arg reference_v "$QWEN38_MATCHED_REFERENCE_KV_CACHE_V" \
  --argjson context_tokens "$QWEN38_MATCHED_CONTEXT_TOKENS" \
  --argjson decode_mvn "$QWEN38_PHYSICAL_DECODE_MVN" \
  --argjson decode_mv_ext "$QWEN38_PHYSICAL_DECODE_MV_EXT" \
  --argjson q5k_canonical_q4x4 \
    "$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4" \
  --argjson kv_budget "$KV_CACHE_BUDGET_BYTES" \
  --argjson max_launch_skew "$MAX_LAUNCH_SKEW_SECONDS" \
  --argjson results "$results" --argjson sealed_children "$sealed_children" '{
    schema:2,verdict:"pass",gate:"qwen38-matched-physical-artifact-matrix",
    repository:$repository,revision:$revision,
    harness:{commit:$harness_commit,
      source_binding:"clean exact harness worktree"},
    pinned_reference_commit:$reference_commit,
    reference_runtime_manifest_sha256:$reference_runtime_manifest_sha,
    hf2q_effective_routing_policy:{dense_decode_mvn:$decode_mvn,
      dense_decode_mv_ext:$decode_mv_ext,
      dense_q5k_canonical_q4x4:$q5k_canonical_q4x4},
    formats:["BF16","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],
    widths:[1,2,4,8,16],
    workload:{speculation:{hf2q:$hf2q_speculation,reference:$reference_speculation},
      cache_settings:{
        hf2q:{format:$hf2q_kv,budget_bytes:$kv_budget,
          context_tokens_per_slot:$context_tokens},
        reference:{k_format:$reference_k,v_format:$reference_v,
          context_tokens_total:$context_tokens}}},
    acceptance:{maximum_launch_skew_seconds:$max_launch_skew},
    physical_matrix:{sha256:$physical_sha,seal_validated:true,
      self_contained_path:"physical-proof/matrix.json",
      gate:"qwen38-artifact-physical-width-matrix",
      source_commit:$physical_source_commit,
      binary_sha256:$physical_binary_sha},
    evidence:{script_sha256:$script_sha,contract_sha256:$contract_sha,
      artifact_contract_sha256:$artifact_contract_sha,
      child_results_sealed:true,children:$sealed_children},results:$results
  }' >"$OUT_DIR/summary.json.tmp"
qwen38_validate_matched_physical_matrix_receipt "$OUT_DIR/summary.json.tmp"
matched_physical_validate_matrix_reference_cohort \
  "$OUT_DIR/summary.json.tmp" "$REFERENCE_RUNTIME_MANIFEST_SHA256"

evidence_manifest="$OUT_DIR/evidence.sha256"
: >"$evidence_manifest.tmp"
while IFS= read -r path; do
    case "$path" in
        summary.json|summary.json.tmp|evidence.sha256|evidence.sha256.tmp|result.sha256)
            continue
            ;;
    esac
    printf '%s  %s\n' "$(shasum -a 256 "$OUT_DIR/$path" | awk '{print $1}')" \
      "$path" >>"$evidence_manifest.tmp"
done < <(cd "$OUT_DIR" && find . -type f -print | sed 's#^./##' | sort)
mv "$evidence_manifest.tmp" "$evidence_manifest"
(cd "$OUT_DIR" && shasum -a 256 -c evidence.sha256 >/dev/null)
matched_publish_result "$OUT_DIR/summary.json.tmp" "$OUT_DIR/summary.json" \
  "$evidence_manifest" "$OUT_DIR/result.sha256"
if ! matched_physical_validate_reopened_matrix "$OUT_DIR"; then
    mv "$OUT_DIR/summary.json" "$OUT_DIR/summary.json.unsealed"
    exit 1
fi
printf 'matched physical artifact matrix sealed at %s\n' "$OUT_DIR/summary.json"
