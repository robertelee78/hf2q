#!/usr/bin/env bash
set -euo pipefail

# Exact complete-artifact proof for the shared Qwen3.5/Qwen3.6 dense and MoE
# MTP execution graph. Runs one model per process so every cell unloads before
# the next artifact is mapped.

MODEL_ROOT=${MODEL_ROOT:?MODEL_ROOT is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
GATE_CARGO_HOME=${GATE_CARGO_HOME:-${TMPDIR:-/var/tmp}/hf2q-qwen-exact-cargo-home}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$script_dir/qwen38_artifact_contract.sh"
# shellcheck source=scripts/qwen_family_mtp_artifact_contract.sh
source "$script_dir/qwen_family_mtp_artifact_contract.sh"

for command in awk cargo find git jq mv sed shasum sort stat; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ "$MODEL_ROOT" == /* && "$OUT_DIR" == /* && "$GATE_CARGO_HOME" == /* ]] || {
    echo "MODEL_ROOT, OUT_DIR, and GATE_CARGO_HOME must be absolute" >&2
    exit 2
}
MODEL_ROOT=$(cd "$MODEL_ROOT" && pwd -P)
mkdir -p "$GATE_CARGO_HOME"
GATE_CARGO_HOME=$(cd "$GATE_CARGO_HOME" && pwd -P)
case "$GATE_CARGO_HOME/" in
    "$root_dir"/*|"$OUT_DIR"/*|"$MODEL_ROOT"/*)
        echo "GATE_CARGO_HOME must be outside source, model, and evidence trees" >&2
        exit 2
        ;;
esac
[[ ! -e "$OUT_DIR" \
    || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "Qwen family MTP matrix output must be fresh: $OUT_DIR" >&2
    exit 2
}
source_commit=$(git -C "$root_dir" rev-parse HEAD)
[[ "$source_commit" =~ ^[0-9a-f]{40}$ \
    && -z "$(git -C "$root_dir" status --porcelain --untracked-files=all)" ]] || {
    echo "Qwen family MTP matrix requires a clean exact source tree" >&2
    exit 2
}
qwen38_reject_cargo_configuration "$root_dir" "$GATE_CARGO_HOME"
IFS=$'\t' read -r dependency_version dependency_source dependency_checksum \
    <<<"$(qwen38_mlx_native_registry_identity "$root_dir")"

artifact_snapshots=()
artifact_paths=()
for cell in $(qwen_family_mtp_cells); do
    IFS=$'\t' read -r _cell _repository _revision file bytes sha _arch \
        _file_type _target_layers _total_blocks _variant \
        <<<"$(qwen_family_mtp_record "$cell")"
    path="$MODEL_ROOT/$file"
    [[ -f "$path" && -r "$path" && ! -L "$path" ]] || {
        echo "qualified $cell artifact must be a regular readable file: $path" >&2
        exit 2
    }
    actual_bytes=$(stat -f '%z' "$path" 2>/dev/null || stat -c '%s' "$path")
    actual_sha=$(shasum -a 256 "$path" | awk '{print $1}')
    qwen_family_mtp_validate_artifact_identity "$cell" "$actual_bytes" "$actual_sha"
    artifact_paths+=("$path")
    artifact_snapshots+=("$(stat -f '%d:%i:%z:%m:%c' "$path" 2>/dev/null \
        || stat -c '%d:%i:%s:%Y:%Z' "$path")")
done

mkdir -p "$OUT_DIR"
test_name='inference::models::qwen35::forward_gpu::tests::block_segmented_native_qwen_family_mtp_artifact_matches_sequential_control'
cell_paths=()
index=0
for cell in $(qwen_family_mtp_cells); do
    IFS=$'\t' read -r _cell repository revision file bytes sha arch file_type \
        target_layers total_blocks variant <<<"$(qwen_family_mtp_record "$cell")"
    path=${artifact_paths[$index]}
    [[ "$(stat -f '%d:%i:%z:%m:%c' "$path" 2>/dev/null \
        || stat -c '%d:%i:%s:%Y:%Z' "$path")" == "${artifact_snapshots[$index]}" ]] || {
        echo "$cell artifact changed after preflight" >&2
        exit 1
    }
    log="$OUT_DIR/$cell.log"
    env -u HF2Q_TEST_QWEN38_GGUF -u HF2Q_TEST_QWEN38_FORMAT \
        HF2Q_TEST_QWEN_FAMILY_CELL="$cell" \
        HF2Q_TEST_QWEN_FAMILY_GGUF="$path" \
        MLX_DISP_BUCKET=1 HF2Q_DECODE_MVN=1 HF2Q_DECODE_MV_EXT=0 \
        CARGO_HOME="$GATE_CARGO_HOME" GIT_COMMIT_SHA="$source_commit" \
        cargo test --manifest-path "$root_dir/Cargo.toml" --locked --bin hf2q \
            "$test_name" -- --ignored --exact --nocapture >"$log" 2>&1
    [[ "$(grep -Ec 'test result: ok[.] 1 passed; 0 failed; 0 ignored;' "$log")" == 1 ]] || {
        echo "$cell did not execute exactly one passing full-model gate" >&2
        sed -n '1,260p' "$log" >&2
        exit 1
    }
    [[ "$(stat -f '%d:%i:%z:%m:%c' "$path" 2>/dev/null \
        || stat -c '%d:%i:%s:%Y:%Z' "$path")" == "${artifact_snapshots[$index]}" ]] || {
        echo "$cell artifact changed during execution" >&2
        exit 1
    }
    cell_path="$OUT_DIR/$cell.json"
    jq -n --arg cell "$cell" --arg repository "$repository" --arg revision "$revision" \
        --arg file "$file" --arg sha "$sha" --arg arch "$arch" --arg variant "$variant" \
        --argjson bytes "$bytes" --argjson file_type "$file_type" \
        --argjson target_layers "$target_layers" --argjson total_blocks "$total_blocks" \
        --arg log_path "$(basename "$log")" \
        --arg log_sha "$(shasum -a 256 "$log" | awk '{print $1}')" '{
          schema:1,verdict:"pass",cell:$cell,repository:$repository,revision:$revision,
          artifact:{file:$file,sha256:$sha,bytes:$bytes,file_type:$file_type,
            architecture:$arch,variant:$variant,target_layers:$target_layers,
            total_blocks:$total_blocks,mtp_layers:1},
          proof:{metadata_preflight_before_metal:true,
            native_storage_without_substitution:true,
            shared_embedding_and_exact_head_allocation:true,
            mtp_compound_matches_sequential:true,
            target_and_mtp_final_bytes_equal:true,
            quantized_dispatch_observed:true},
          log:{path:$log_path,sha256:$log_sha}}
        ' >"$cell_path.tmp"
    mv "$cell_path.tmp" "$cell_path"
    cell_paths+=("$cell_path")
    index=$((index + 1))
done

[[ "$(git -C "$root_dir" rev-parse HEAD)" == "$source_commit" \
    && -z "$(git -C "$root_dir" status --porcelain --untracked-files=all)" ]] || {
    echo "source tree changed during Qwen family MTP matrix" >&2
    exit 1
}
qwen38_reject_cargo_configuration "$root_dir" "$GATE_CARGO_HOME"
[[ "$(qwen38_mlx_native_registry_identity "$root_dir")" == \
  "$dependency_version"$'\t'"$dependency_source"$'\t'"$dependency_checksum" ]] || {
    echo "mlx-native dependency identity changed during Qwen family MTP matrix" >&2
    exit 1
}

results=$(jq -s . "${cell_paths[@]}")
jq -n --arg source_commit "$source_commit" \
    --arg dependency_version "$dependency_version" \
    --arg dependency_source "$dependency_source" \
    --arg dependency_checksum "$dependency_checksum" \
    --arg runner_sha "$(shasum -a 256 "$script_dir/qwen_family_mtp_artifact_matrix.sh" | awk '{print $1}')" \
    --arg contract_sha "$(shasum -a 256 "$script_dir/qwen_family_mtp_artifact_contract.sh" | awk '{print $1}')" \
    --argjson results "$results" '{
      schema:1,verdict:"pass",gate:"qwen-family-native-mtp-compound-matrix",
      source_commit:$source_commit,
      dependency:{name:"mlx-native",version:$dependency_version,
        source:$dependency_source,checksum:$dependency_checksum},
      evidence:{runner_sha256:$runner_sha,contract_sha256:$contract_sha},
      cells:["qwen35-dense","qwen36-dense","qwen36-moe"],results:$results}
    ' >"$OUT_DIR/matrix.json.tmp"
qwen_family_mtp_validate_matrix_receipt "$OUT_DIR/matrix.json.tmp"

: >"$OUT_DIR/evidence.sha256.tmp"
while IFS= read -r file; do
    case "$file" in
        matrix.json|matrix.json.tmp|evidence.sha256|evidence.sha256.tmp|result.sha256|result.sha256.tmp)
            continue
            ;;
    esac
    printf '%s  %s\n' "$(shasum -a 256 "$OUT_DIR/$file" | awk '{print $1}')" "$file" \
        >>"$OUT_DIR/evidence.sha256.tmp"
done < <(cd "$OUT_DIR" && find . -type f -print | sed 's#^./##' | sort)
mv "$OUT_DIR/evidence.sha256.tmp" "$OUT_DIR/evidence.sha256"
mv "$OUT_DIR/matrix.json.tmp" "$OUT_DIR/matrix.json"
printf '%s  matrix.json\n%s  evidence.sha256\n' \
    "$(shasum -a 256 "$OUT_DIR/matrix.json" | awk '{print $1}')" \
    "$(shasum -a 256 "$OUT_DIR/evidence.sha256" | awk '{print $1}')" \
    >"$OUT_DIR/result.sha256"
qwen_family_mtp_validate_matrix_seal "$OUT_DIR/matrix.json" "$root_dir"
jq . "$OUT_DIR/matrix.json"
