#!/usr/bin/env bash
set -euo pipefail

# Exact-artifact Qwen3.8 A -> B -> A matrix. This composes the production
# explicit-switch integration gate with the immutable five-format catalog.
# Pairwise diagnostics use isolated servers. Acceptance additionally requires
# two complete five-format cycles in one long-lived server process.

MODEL_ROOT=${MODEL_ROOT:?MODEL_ROOT is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
GATE_CARGO_HOME=${GATE_CARGO_HOME:-${TMPDIR:-/var/tmp}/hf2q-qwen-exact-cargo-home}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
manifest="$root_dir/data/qwen38_exact_swap_matrix.v1.json"
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$script_dir/qwen38_artifact_contract.sh"
# shellcheck source=scripts/qwen38_exact_swap_matrix_contract.sh
source "$script_dir/qwen38_exact_swap_matrix_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

for command in awk cargo chmod cp find git jq mktemp mv rm rmdir sed shasum sort stat tr; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done

assert_artifact_snapshots_unchanged() {
    local index=0 format relative_file model_path current
    for format in $(qwen38_artifact_formats); do
        IFS=$'\t' read -r _format relative_file _bytes _sha _type \
            <<<"$(qwen38_artifact_record "$format")"
        model_path="$MODEL_ROOT/$relative_file"
        current=$(stat -f '%d:%i:%z:%m:%c' "$model_path" 2>/dev/null \
          || stat -c '%d:%i:%s:%Y:%Z' "$model_path")
        [[ "$current" == "${artifact_snapshots[$index]}" ]] || {
            echo "qualified $format artifact changed during exact swap gate" >&2
            return 1
        }
        index=$((index + 1))
    done
}
[[ "$MODEL_ROOT" == /* && "$OUT_DIR" == /* && "$GATE_CARGO_HOME" == /* ]] || {
    echo "MODEL_ROOT, OUT_DIR, and GATE_CARGO_HOME must be absolute paths" >&2
    exit 2
}
mkdir -p "$GATE_CARGO_HOME"
GATE_CARGO_HOME=$(cd "$GATE_CARGO_HOME" && pwd -P)
case "$OUT_DIR/" in
    "$root_dir"/*)
        echo "OUT_DIR must be outside the exact source tree" >&2
        exit 2
        ;;
esac
case "$GATE_CARGO_HOME/" in
    "$root_dir"/*|"$OUT_DIR"/*)
        echo "GATE_CARGO_HOME must be outside the source and evidence trees" >&2
        exit 2
        ;;
esac
[[ ! -e "$OUT_DIR" \
  || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "exact swap output directory must be fresh: $OUT_DIR" >&2
    exit 2
}
qwen38_validate_exact_swap_manifest "$manifest" || {
    echo "invalid exact Qwen3.8 swap manifest" >&2
    exit 2
}

source_commit=$(git -C "$root_dir" rev-parse HEAD)
[[ "$source_commit" =~ ^[0-9a-f]{40}$ \
  && -z "$(git -C "$root_dir" status --porcelain --untracked-files=all)" ]] || {
    echo "exact swap matrix requires a clean exact source tree" >&2
    exit 2
}
qwen38_reject_cargo_configuration "$root_dir" "$GATE_CARGO_HOME"
IFS=$'\t' read -r dependency_version dependency_source dependency_checksum \
    <<<"$(qwen38_mlx_native_registry_identity "$root_dir")"

# Build once with the exact commit embedded, or consume the already-sealed
# release candidate supplied by the protected hardware workflow.
if [[ -n "${HF2Q_EXACT_SWAP_BINARY:-}" ]]; then
    binary=$HF2Q_EXACT_SWAP_BINARY
    [[ "$binary" == /* ]] || {
        echo "HF2Q_EXACT_SWAP_BINARY must be absolute" >&2
        exit 2
    }
else
    (cd "$root_dir" \
      && CARGO_HOME="$GATE_CARGO_HOME" GIT_COMMIT_SHA="$source_commit" \
        cargo build --release --locked --bin hf2q)
    target_dir=$(cd "$root_dir" \
      && CARGO_HOME="$GATE_CARGO_HOME" \
        cargo metadata --locked --no-deps --format-version 1 | jq -er .target_directory)
    binary="$target_dir/release/hf2q"
fi
[[ -f "$binary" && -x "$binary" && ! -L "$binary" ]] || {
    echo "release hf2q binary is missing or not a regular executable" >&2
    exit 1
}
# `cargo test` builds the integration-test binary targets and may replace
# target/release/hf2q even when the caller supplied that exact path. Execute
# and attest a private immutable copy so compilation cannot change the
# candidate after its digest is captured.
sealed_binary_dir=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-exact-swap-bin.XXXXXX")
sealed_binary="$sealed_binary_dir/hf2q"
cleanup_sealed_binary() {
    case "$sealed_binary_dir/" in
        "${TMPDIR:-/tmp}"/hf2q-exact-swap-bin.*/*)
            rm -f -- "$sealed_binary" || true
            rmdir -- "$sealed_binary_dir" || true
            ;;
        *)
            echo "refusing unsafe sealed-binary cleanup: $sealed_binary_dir" >&2
            ;;
    esac
}
trap cleanup_sealed_binary EXIT
cp "$binary" "$sealed_binary"
chmod u+x "$sealed_binary"
binary=$sealed_binary
binary_sha=$(shasum -a 256 "$binary" | awk '{print $1}')
[[ "$binary_sha" =~ ^[0-9a-f]{64}$ ]] || exit 1
build_info=$("$binary" __build-info)
binary_git_commit=$(jq -er '
  select(.schema == "hf2q.build-info.v1") | .git_commit
' <<<"$build_info")
[[ "$binary_git_commit" == "$source_commit" ]] || {
    echo "hf2q binary does not embed exact source commit $source_commit" >&2
    exit 1
}

# Resolve all five immutable inputs through the same schema-v2 single-hash
# authority consumed by the product server. A partial/stale catalog is a matrix
# failure, never a skipped cell or a fallback to per-swap hashing.
mkdir -p "$OUT_DIR/preflight"
export HF2Q_MODEL_VERIFICATION_BINARY="$binary"
artifact_snapshots=()
for format in $(qwen38_artifact_formats); do
    IFS=$'\t' read -r _format relative_file expected_bytes expected_sha expected_type \
        <<<"$(qwen38_artifact_record "$format")"
    model_path="$MODEL_ROOT/$relative_file"
    [[ -f "$model_path" && -r "$model_path" && ! -L "$model_path" ]] || {
        echo "qualified $format artifact is missing or symlinked: $model_path" >&2
        exit 2
    }
    receipt_slug=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')
    receipt="$OUT_DIR/preflight/$receipt_slug.json"
    hf2q_release_prepare_model_verification "$model_path" "$expected_sha" \
      "$receipt" \
      "${HF2Q_MODEL_VERIFICATION_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}/hf2q/model-verification}"
    actual_bytes=$(stat -f '%z' "$model_path" 2>/dev/null \
      || stat -c '%s' "$model_path")
    qwen38_validate_artifact_identity \
      "$format" "$expected_sha" "$actual_bytes" "$expected_type"
    artifact_snapshots+=("$(jq -er .file_snapshot "$receipt")")
done

cell_paths=()
artifact_index_for_format() {
    case $1 in
        BF16) printf '0\n' ;;
        Q4_K_M) printf '1\n' ;;
        Q5_K_M) printf '2\n' ;;
        Q6_K) printf '3\n' ;;
        Q8_0) printf '4\n' ;;
        *) return 1 ;;
    esac
}

while IFS=$'\t' read -r pair_id format_a format_b; do
    IFS=$'\t' read -r _format file_a _bytes_a sha_a type_a \
        <<<"$(qwen38_artifact_record "$format_a")"
    IFS=$'\t' read -r _format file_b _bytes_b sha_b type_b \
        <<<"$(qwen38_artifact_record "$format_b")"
    model_a="$MODEL_ROOT/$file_a"
    model_b="$MODEL_ROOT/$file_b"
    index_a=$(artifact_index_for_format "$format_a")
    index_b=$(artifact_index_for_format "$format_b")
    [[ "$(stat -f '%d:%i:%z:%m:%c' "$model_a" 2>/dev/null \
      || stat -c '%d:%i:%s:%Y:%Z' "$model_a")" == "${artifact_snapshots[$index_a]}" \
      && "$(stat -f '%d:%i:%z:%m:%c' "$model_b" 2>/dev/null \
      || stat -c '%d:%i:%s:%Y:%Z' "$model_b")" == "${artifact_snapshots[$index_b]}" ]] || {
        echo "$pair_id artifact changed after preflight" >&2
        exit 1
    }
    cell="$OUT_DIR/$pair_id.json"
    log="$OUT_DIR/$pair_id.log"
    echo "running exact Qwen3.8 swap pair: $pair_id" >&2
    (
      cd "$root_dir"
      HF2Q_HOT_SWAP_E2E=1 \
      HF2Q_HOT_SWAP_E2E_MODEL_A="$model_a" \
      HF2Q_HOT_SWAP_E2E_MODEL_B="$model_b" \
      HF2Q_HOT_SWAP_E2E_MODEL_A_SHA256="$sha_a" \
      HF2Q_HOT_SWAP_E2E_MODEL_B_SHA256="$sha_b" \
      HF2Q_HOT_SWAP_E2E_MODEL_A_ARCHITECTURE=qwen35 \
      HF2Q_HOT_SWAP_E2E_MODEL_B_ARCHITECTURE=qwen35 \
      HF2Q_HOT_SWAP_E2E_MODEL_A_ARCH_FAMILY=qwen35 \
      HF2Q_HOT_SWAP_E2E_MODEL_B_ARCH_FAMILY=qwen35 \
      HF2Q_HOT_SWAP_EXACT_RECEIPT="$cell" \
      HF2Q_HOT_SWAP_EXACT_PAIR_ID="$pair_id" \
      HF2Q_HOT_SWAP_EXACT_FORMAT_A="$format_a" \
      HF2Q_HOT_SWAP_EXACT_FORMAT_B="$format_b" \
      HF2Q_HOT_SWAP_EXACT_FILE_A="$file_a" \
      HF2Q_HOT_SWAP_EXACT_FILE_B="$file_b" \
      HF2Q_HOT_SWAP_EXACT_FILE_TYPE_A="$type_a" \
      HF2Q_HOT_SWAP_EXACT_FILE_TYPE_B="$type_b" \
      HF2Q_HOT_SWAP_EXACT_SOURCE_COMMIT="$source_commit" \
      HF2Q_HOT_SWAP_EXACT_BINARY_SHA256="$binary_sha" \
      HF2Q_HOT_SWAP_EXACT_BINARY_GIT_COMMIT="$binary_git_commit" \
      HF2Q_HOT_SWAP_EXACT_MLX_VERSION="$dependency_version" \
      HF2Q_HOT_SWAP_EXACT_MLX_SOURCE="$dependency_source" \
      HF2Q_HOT_SWAP_EXACT_MLX_CHECKSUM="$dependency_checksum" \
      HF2Q_HOT_SWAP_E2E_MAX_SECS=10 \
      HF2Q_HOT_SWAP_EXECUTABLE="$binary" \
      HF2Q_MODEL_VERIFICATION_RECEIPT_DIR="$OUT_DIR/preflight" \
      CARGO_HOME="$GATE_CARGO_HOME" \
      GIT_COMMIT_SHA="$source_commit" \
        cargo test --release --locked --test multi_model_swap \
          model_swap_a_b_a_reclaims_and_replays_e2e \
          -- --exact --nocapture --test-threads=1 >"$log" 2>&1
    )
    [[ "$(shasum -a 256 "$binary" | awk '{print $1}')" == "$binary_sha" ]] || {
        echo "hf2q binary changed during $pair_id" >&2
        exit 1
    }
    qwen38_validate_exact_swap_cell "$cell" "$pair_id" "$format_a" "$format_b" \
      "$source_commit" "$binary_sha" "$binary_git_commit" "$dependency_version" \
      "$dependency_source" "$dependency_checksum" || {
        echo "$pair_id produced an invalid execution receipt" >&2
        exit 1
    }
    cell_paths+=("$cell")
done < <(jq -r '.pairs[] | [.id,.a,.b] | @tsv' "$manifest")

chain="$OUT_DIR/two-cycle-chain.json"
chain_log="$OUT_DIR/two-cycle-chain.log"
assert_artifact_snapshots_unchanged
chain_spec=$(jq -c --arg model_root "$MODEL_ROOT" '{
  artifacts:[.artifacts[] | . + {path:($model_root + "/" + .file)}]
}' "$manifest")
echo "running exact Qwen3.8 long-lived two-cycle chain" >&2
(
  cd "$root_dir"
  HF2Q_HOT_SWAP_E2E=1 \
  HF2Q_HOT_SWAP_EXACT_CHAIN_RECEIPT="$chain" \
  HF2Q_HOT_SWAP_EXACT_CHAIN_SPEC="$chain_spec" \
  HF2Q_HOT_SWAP_EXACT_SOURCE_COMMIT="$source_commit" \
  HF2Q_HOT_SWAP_EXACT_BINARY_SHA256="$binary_sha" \
  HF2Q_HOT_SWAP_EXACT_BINARY_GIT_COMMIT="$binary_git_commit" \
  HF2Q_HOT_SWAP_EXACT_MLX_VERSION="$dependency_version" \
  HF2Q_HOT_SWAP_EXACT_MLX_SOURCE="$dependency_source" \
  HF2Q_HOT_SWAP_EXACT_MLX_CHECKSUM="$dependency_checksum" \
  HF2Q_HOT_SWAP_E2E_MAX_SECS=10 \
  HF2Q_HOT_SWAP_EXECUTABLE="$binary" \
  HF2Q_MODEL_VERIFICATION_RECEIPT_DIR="$OUT_DIR/preflight" \
  CARGO_HOME="$GATE_CARGO_HOME" \
  GIT_COMMIT_SHA="$source_commit" \
    cargo test --release --locked --test multi_model_swap \
      qwen38_exact_five_format_two_cycle_e2e \
      -- --exact --nocapture --test-threads=1 >"$chain_log" 2>&1
)
[[ "$(shasum -a 256 "$binary" | awk '{print $1}')" == "$binary_sha" ]] || {
    echo "hf2q binary changed during long-lived exact swap chain" >&2
    exit 1
}
qwen38_validate_exact_swap_chain "$chain" "$source_commit" "$binary_sha" \
  "$binary_git_commit" "$dependency_version" "$dependency_source" \
  "$dependency_checksum" || {
    echo "long-lived exact swap chain produced an invalid receipt" >&2
    exit 1
}
assert_artifact_snapshots_unchanged

[[ "$(git -C "$root_dir" rev-parse HEAD)" == "$source_commit" \
  && -z "$(git -C "$root_dir" status --porcelain --untracked-files=all)" \
  && "$(shasum -a 256 "$binary" | awk '{print $1}')" == "$binary_sha" ]] || {
    echo "source or binary identity changed during exact swap matrix" >&2
    exit 1
}
qwen38_reject_cargo_configuration "$root_dir" "$GATE_CARGO_HOME"
[[ "$(qwen38_mlx_native_registry_identity "$root_dir")" == \
  "$dependency_version"$'\t'"$dependency_source"$'\t'"$dependency_checksum" ]] || {
    echo "mlx-native registry identity changed during exact swap matrix" >&2
    exit 1
}

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
  --arg runner_sha "$(shasum -a 256 "$script_dir/run_qwen38_exact_swap_matrix.sh" | awk '{print $1}')" \
  --arg matrix_contract_sha "$(shasum -a 256 "$script_dir/qwen38_exact_swap_matrix_contract.sh" | awk '{print $1}')" \
  --arg artifact_contract_sha "$(shasum -a 256 "$script_dir/qwen38_artifact_contract.sh" | awk '{print $1}')" \
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
  }' >"$OUT_DIR/matrix.json.tmp"
qwen38_validate_exact_swap_receipt "$OUT_DIR/matrix.json.tmp"

evidence="$OUT_DIR/evidence.sha256"
: >"$evidence.tmp"
for path in "${cell_paths[@]}"; do
    pair_id=$(basename "$path" .json)
    for suffix in json log; do
        file="$pair_id.$suffix"
        printf '%s  %s\n' \
          "$(shasum -a 256 "$OUT_DIR/$file" | awk '{print $1}')" "$file" \
          >>"$evidence.tmp"
    done
done
for file in two-cycle-chain.json two-cycle-chain.log; do
    printf '%s  %s\n' \
      "$(shasum -a 256 "$OUT_DIR/$file" | awk '{print $1}')" "$file" \
      >>"$evidence.tmp"
done
for receipt in "$OUT_DIR"/preflight/*.json; do
    file="preflight/$(basename "$receipt")"
    printf '%s  %s\n' \
      "$(shasum -a 256 "$receipt" | awk '{print $1}')" "$file" \
      >>"$evidence.tmp"
done
sort -o "$evidence.tmp" "$evidence.tmp"
mv "$evidence.tmp" "$evidence"
matrix_sha=$(shasum -a 256 "$OUT_DIR/matrix.json.tmp" | awk '{print $1}')
evidence_sha=$(shasum -a 256 "$evidence" | awk '{print $1}')
printf '%s  matrix.json\n%s  evidence.sha256\n' "$matrix_sha" "$evidence_sha" \
  >"$OUT_DIR/result.sha256"
mv "$OUT_DIR/matrix.json.tmp" "$OUT_DIR/matrix.json"
qwen38_validate_exact_swap_seal "$OUT_DIR/matrix.json" "$root_dir" || {
    mv "$OUT_DIR/matrix.json" "$OUT_DIR/matrix.json.unsealed"
    exit 1
}
jq . "$OUT_DIR/matrix.json"
