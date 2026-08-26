#!/usr/bin/env bash
set -euo pipefail

# Apply one identical cfg(test)-only probe to exact main and candidate commits,
# then require bit-identical ordinary prefill/continuation logits, verifier
# hidden rows, and physical target+MTP cache storage. The probe calls only the
# ordinary target and MTP entry points; the compound coordinator is unreachable.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
BASELINE_SOURCE_ROOT=${BASELINE_SOURCE_ROOT:-/opt/hf2q}
CANDIDATE_SOURCE_ROOT=${CANDIDATE_SOURCE_ROOT:-$root_dir}
MODEL=${MODEL:-/opt/hf2q/models/qwen3.8/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
GATE_CARGO_HOME=${GATE_CARGO_HOME:-${TMPDIR:-/var/tmp}/hf2q-qwen-exact-cargo-home}

for command in awk cargo cmp cp diff find git grep jq mkdir mktemp mv rm sed shasum sort stat; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ "$BASELINE_SOURCE_ROOT" == /* && "$CANDIDATE_SOURCE_ROOT" == /* \
    && "$MODEL" == /* && "$OUT_DIR" == /* && "$GATE_CARGO_HOME" == /* ]] || {
    echo "ordinary identity paths must be absolute" >&2
    exit 2
}
BASELINE_SOURCE_ROOT=$(cd "$BASELINE_SOURCE_ROOT" && pwd -P)
CANDIDATE_SOURCE_ROOT=$(cd "$CANDIDATE_SOURCE_ROOT" && pwd -P)
mkdir -p "$GATE_CARGO_HOME"
GATE_CARGO_HOME=$(cd "$GATE_CARGO_HOME" && pwd -P)
[[ -f "$MODEL" && -r "$MODEL" && ! -L "$MODEL" ]] || {
    echo "ordinary identity model must be a regular readable non-symlink: $MODEL" >&2
    exit 2
}
[[ ! -e "$OUT_DIR" \
    || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "ordinary identity output must be fresh: $OUT_DIR" >&2
    exit 2
}
for source_root in "$BASELINE_SOURCE_ROOT" "$CANDIDATE_SOURCE_ROOT"; do
    [[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]] || {
        echo "ordinary identity gate requires clean source: $source_root" >&2
        exit 2
    }
done
baseline_commit=$(git -C "$BASELINE_SOURCE_ROOT" rev-parse HEAD)
candidate_commit=$(git -C "$CANDIDATE_SOURCE_ROOT" rev-parse HEAD)
[[ "$(git -C "$BASELINE_SOURCE_ROOT" symbolic-ref --quiet --short HEAD || true)" == main \
    && "$baseline_commit" =~ ^[0-9a-f]{40}$ \
    && "$candidate_commit" =~ ^[0-9a-f]{40}$ \
    && "$baseline_commit" != "$candidate_commit" ]] || {
    echo "ordinary identity gate requires distinct exact main and candidate commits" >&2
    exit 2
}
git -C "$CANDIDATE_SOURCE_ROOT" merge-base --is-ancestor \
    "$baseline_commit" "$candidate_commit" || {
    echo "ordinary identity candidate must descend from exact main" >&2
    exit 2
}

# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$script_dir/qwen38_artifact_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"
# shellcheck source=scripts/qwen35_ordinary_identity_contract.sh
source "$script_dir/qwen35_ordinary_identity_contract.sh"
qwen38_reject_cargo_configuration "$BASELINE_SOURCE_ROOT" "$GATE_CARGO_HOME"
qwen38_reject_cargo_configuration "$CANDIDATE_SOURCE_ROOT" "$GATE_CARGO_HOME"
IFS=$'\t' read -r _format _file expected_bytes expected_sha expected_file_type \
    <<<"$(qwen38_artifact_record Q4_K_M)"
actual_bytes=$(stat -f '%z' "$MODEL" 2>/dev/null || stat -c '%s' "$MODEL")
actual_sha=$(shasum -a 256 "$MODEL" | awk '{print $1}')
qwen38_validate_artifact_identity Q4_K_M "$actual_sha" "$actual_bytes" "$expected_file_type"
model_snapshot=$(hf2q_release_model_snapshot "$MODEL")
[[ -n "$model_snapshot" ]] || { echo "model snapshot failed" >&2; exit 2; }
baseline_dependency=$(qwen38_mlx_native_registry_identity "$BASELINE_SOURCE_ROOT")
candidate_dependency=$(qwen38_mlx_native_registry_identity "$CANDIDATE_SOURCE_ROOT")

probe_module="$CANDIDATE_SOURCE_ROOT/scripts/probes/qwen35_ordinary_byte_identity_probe.rs"
probe_patch="$CANDIDATE_SOURCE_ROOT/scripts/probes/qwen35_ordinary_byte_identity_module.patch"
[[ -f "$probe_module" && ! -L "$probe_module" && -f "$probe_patch" && ! -L "$probe_patch" ]] \
    || { echo "ordinary identity probe sources are missing" >&2; exit 2; }

scratch=$(mktemp -d "${TMPDIR:-/var/tmp}/hf2q-ordinary-identity.XXXXXX")
baseline_worktree="$scratch/baseline-source"
candidate_worktree="$scratch/candidate-source"
cleanup() {
    for worktree in "$baseline_worktree" "$candidate_worktree"; do
        if [[ -d "$worktree" ]]; then
            git -C "$CANDIDATE_SOURCE_ROOT" worktree remove --force "$worktree" \
                >/dev/null 2>&1 || true
        fi
    done
    case "$scratch/" in
        "${TMPDIR:-/var/tmp}"/hf2q-ordinary-identity.*/*)
            rm -rf -- "$scratch"
            ;;
        *) echo "refusing unsafe ordinary identity cleanup: $scratch" >&2 ;;
    esac
}
trap cleanup EXIT INT TERM
mkdir -p "$OUT_DIR"
git -C "$CANDIDATE_SOURCE_ROOT" worktree add --detach "$baseline_worktree" "$baseline_commit" \
    >/dev/null
git -C "$CANDIDATE_SOURCE_ROOT" worktree add --detach "$candidate_worktree" "$candidate_commit" \
    >/dev/null

assert_model_unchanged() {
    [[ -f "$MODEL" && ! -L "$MODEL" \
        && "$(hf2q_release_model_snapshot "$MODEL")" == "$model_snapshot" ]] || {
        echo "ordinary identity model changed during gate" >&2
        return 1
    }
}

test_name='inference::models::qwen35::forward_gpu::ordinary_byte_identity_probe::ordinary_prefill_continuation_matches_main_byte_for_byte'
run_probe() {
    local label=$1 worktree=$2 commit=$3 output log status
    output="$OUT_DIR/$label.probe.tsv"
    log="$OUT_DIR/$label.log"
    git -C "$worktree" apply "$probe_patch"
    mkdir -p "$worktree/src/inference/models/qwen35/forward_gpu"
    cp "$probe_module" \
        "$worktree/src/inference/models/qwen35/forward_gpu/ordinary_byte_identity_probe.rs"
    status=$(git -C "$worktree" status --short --untracked-files=all)
    [[ "$status" == ' M src/inference/models/qwen35/forward_gpu.rs
?? src/inference/models/qwen35/forward_gpu/ordinary_byte_identity_probe.rs' ]] || {
        echo "$label probe modified an unexpected source surface:" >&2
        printf '%s\n' "$status" >&2
        return 1
    }
    [[ "$(shasum -a 256 "$worktree/src/inference/models/qwen35/forward_gpu/ordinary_byte_identity_probe.rs" | awk '{print $1}')" == \
        "$(shasum -a 256 "$probe_module" | awk '{print $1}')" ]] || return 1
    assert_model_unchanged
    env \
        HF2Q_ORDINARY_IDENTITY_GGUF="$MODEL" \
        HF2Q_ORDINARY_IDENTITY_OUTPUT="$output" \
        HF2Q_QWEN_SPECULATION=off HF2Q_TQ_KV=1 \
        MLX_DISP_BUCKET=1 HF2Q_DECODE_MVN=1 HF2Q_DECODE_MV_EXT=0 \
        CARGO_HOME="$GATE_CARGO_HOME" GIT_COMMIT_SHA="$commit" \
        cargo test --manifest-path "$worktree/Cargo.toml" \
            --target-dir "$scratch/$label-target" --locked --bin hf2q \
            "$test_name" -- --ignored --exact --nocapture >"$log" 2>&1
    [[ "$(grep -Ec 'test result: ok[.] 1 passed; 0 failed; 0 ignored;' "$log")" == 1 ]] \
        || { sed -n '1,260p' "$log" >&2; return 1; }
    qwen35_validate_ordinary_probe_tsv "$output"
    assert_model_unchanged
}

run_probe baseline "$baseline_worktree" "$baseline_commit"
run_probe candidate "$candidate_worktree" "$candidate_commit"
cmp -s "$OUT_DIR/baseline.probe.tsv" "$OUT_DIR/candidate.probe.tsv" || {
    echo "ordinary target/MTP bytes differ between main and candidate" >&2
    diff -u "$OUT_DIR/baseline.probe.tsv" "$OUT_DIR/candidate.probe.tsv" >&2 || true
    exit 1
}
assert_model_unchanged

probe_identity=$(jq -Rn '
  [inputs | split("\t") | select(.[0] | endswith("_sha256"))
    | {key:.[0],value:.[1]}] | from_entries
' <"$OUT_DIR/candidate.probe.tsv")
IFS=$'\t' read -r baseline_dependency_version baseline_dependency_source \
    baseline_dependency_checksum <<<"$baseline_dependency"
IFS=$'\t' read -r candidate_dependency_version candidate_dependency_source \
    candidate_dependency_checksum <<<"$candidate_dependency"
jq -n \
    --arg baseline_commit "$baseline_commit" --arg candidate_commit "$candidate_commit" \
    --arg baseline_dependency_version "$baseline_dependency_version" \
    --arg baseline_dependency_source "$baseline_dependency_source" \
    --arg baseline_dependency_checksum "$baseline_dependency_checksum" \
    --arg candidate_dependency_version "$candidate_dependency_version" \
    --arg candidate_dependency_source "$candidate_dependency_source" \
    --arg candidate_dependency_checksum "$candidate_dependency_checksum" \
    --arg model "$MODEL" --arg model_sha "$actual_sha" \
    --arg model_snapshot "$model_snapshot" --argjson model_bytes "$actual_bytes" \
    --arg runner_sha "$(shasum -a 256 "$script_dir/qwen35_ordinary_byte_identity.sh" | awk '{print $1}')" \
    --arg contract_sha "$(shasum -a 256 "$script_dir/qwen35_ordinary_identity_contract.sh" | awk '{print $1}')" \
    --arg module_sha "$(shasum -a 256 "$probe_module" | awk '{print $1}')" \
    --arg patch_sha "$(shasum -a 256 "$probe_patch" | awk '{print $1}')" \
    --argjson identity "$probe_identity" '{
      schema:1,verdict:"pass",gate:"qwen35-ordinary-main-byte-identity",
      baseline:{commit:$baseline_commit,dependency:{name:"mlx-native",
        version:$baseline_dependency_version,source:$baseline_dependency_source,
        checksum:$baseline_dependency_checksum}},
      candidate:{commit:$candidate_commit,dependency:{name:"mlx-native",
        version:$candidate_dependency_version,source:$candidate_dependency_source,
        checksum:$candidate_dependency_checksum}},
      model:{path:$model,format:"Q4_K_M",sha256:$model_sha,bytes:$model_bytes,
        file_type:15,file_snapshot:$model_snapshot},
      route:{speculation:"off",tq_kv:true,mlx_disp_bucket:1,mvn:false,mv_ext:true,
        prefix_tokens:33,continuation_tokens:3},
      proof:{same_probe_on_both_commits:true,ordinary_target_and_mtp_direct:true,
        exact_probe_receipts_equal:true,exact_f32_logits_equal:true,
        exact_target_hidden_equal:true,
        exact_physical_target_and_mtp_cache_bytes_equal:true},
      identity:$identity,
      evidence:{runner_sha256:$runner_sha,contract_sha256:$contract_sha,
        probe_module_sha256:$module_sha,probe_patch_sha256:$patch_sha}}
' >"$OUT_DIR/receipt.json.tmp"
qwen35_validate_ordinary_identity_receipt "$OUT_DIR/receipt.json.tmp"
mv "$OUT_DIR/receipt.json.tmp" "$OUT_DIR/receipt.json"
: >"$OUT_DIR/evidence.sha256"
for file in baseline.log baseline.probe.tsv candidate.log candidate.probe.tsv; do
    printf '%s  %s\n' "$(shasum -a 256 "$OUT_DIR/$file" | awk '{print $1}')" "$file" \
        >>"$OUT_DIR/evidence.sha256"
done
printf '%s  receipt.json\n%s  evidence.sha256\n' \
    "$(shasum -a 256 "$OUT_DIR/receipt.json" | awk '{print $1}')" \
    "$(shasum -a 256 "$OUT_DIR/evidence.sha256" | awk '{print $1}')" \
    >"$OUT_DIR/result.sha256"
qwen35_validate_ordinary_identity_seal "$OUT_DIR/receipt.json" \
    "$BASELINE_SOURCE_ROOT" "$CANDIDATE_SOURCE_ROOT"
jq . "$OUT_DIR/receipt.json"
