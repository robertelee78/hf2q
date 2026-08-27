#!/usr/bin/env bash
set -euo pipefail

# Exact cross-family generative swap authority. DeepSeek is the eviction hub:
# every adjacent pair exceeds the largest-artifact byte budget, so all twelve
# transitions must replace rather than co-reside. The three spokes run twice
# in one server PID. Fresh-server rows cannot expose cumulative leaks, stale
# family state, or generation reuse and are not acceptance proof.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
MANIFEST=$ROOT_DIR/data/generative_swap_matrix.v1.json
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
GATE_CARGO_HOME=${HF2Q_GENERATIVE_SWAP_CARGO_HOME:-${TMPDIR:-/var/tmp}/hf2q-generative-swap-cargo-home}
# shellcheck source=scripts/generative_swap_matrix_contract.sh
source "$ROOT_DIR/scripts/generative_swap_matrix_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$ROOT_DIR/scripts/qwen36_watchdog_validate.sh"

for command in awk cargo chmod cp find footprint git grep jq lsof mkdir mktemp mv ps rm rmdir sed shasum sort stat vm_stat vmmap; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
GATE_PORT=52337
if lsof -nP -iTCP:"$GATE_PORT" -sTCP:LISTEN 2>/dev/null | sed -n '2p' | grep -q .; then
    echo "generative swap gate port is already occupied: $GATE_PORT" >&2
    exit 2
fi
[[ "$OUT_DIR" == /* ]] || {
    echo "OUT_DIR must be absolute" >&2
    exit 2
}
[[ "$GATE_CARGO_HOME" == /* ]] || {
    echo "HF2Q_GENERATIVE_SWAP_CARGO_HOME must be absolute" >&2
    exit 2
}
mkdir -p "$GATE_CARGO_HOME"
GATE_CARGO_HOME=$(cd "$GATE_CARGO_HOME" && pwd -P)
[[ "$OUT_DIR" != / && -d "$(dirname "$OUT_DIR")" && ! -L "$OUT_DIR" ]] || {
    echo "OUT_DIR must have an existing parent and may not be a symlink" >&2
    exit 2
}
out_physical=$(cd "$(dirname "$OUT_DIR")" && pwd -P)/$(basename "$OUT_DIR")
case "$out_physical/" in
    "$ROOT_DIR"/*)
        echo "OUT_DIR must be outside the exact source tree" >&2
        exit 2
        ;;
esac
case "$GATE_CARGO_HOME/" in
    "$ROOT_DIR"/*|"$out_physical"/*)
        echo "gate Cargo home must be outside the source and evidence trees" >&2
        exit 2
        ;;
esac
[[ ! -e "$OUT_DIR" \
  || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "generative swap output directory must be fresh: $OUT_DIR" >&2
    exit 2
}
hf2q_validate_generative_swap_matrix "$MANIFEST" || {
    echo "invalid generative swap matrix: $MANIFEST" >&2
    exit 2
}

source_commit=$(git -C "$ROOT_DIR" rev-parse HEAD)
[[ "$source_commit" =~ ^[0-9a-f]{40}$ \
  && -z "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=all)" ]] || {
    echo "generative swap gate requires a clean exact source tree" >&2
    exit 2
}
hf2q_generative_swap_reject_cargo_configuration "$ROOT_DIR" "$GATE_CARGO_HOME"
IFS=$'\t' read -r dependency_version dependency_source dependency_checksum \
  <<<"$(hf2q_generative_swap_dependency_identity "$ROOT_DIR")"

# Resolve and measure every artifact before compilation or the first model
# load. The sealed binary later hashes each stable artifact once, before the
# server starts, and publishes schema-v2 receipts consumed by the server's
# exact identity registry. The snapshot below also proves each qualified input
# stayed stable throughout the complete run.
artifact_snapshots=()
while IFS=$'\t' read -r id path_env expected_file expected_bytes expected_sha; do
    path=${!path_env:-}
    [[ "$path" == /* && -f "$path" && -r "$path" && ! -L "$path" ]] || {
        echo "$id requires an absolute regular artifact via $path_env" >&2
        exit 2
    }
    canonical=$(cd "$(dirname "$path")" && pwd -P)/$(basename "$path")
    [[ "$(basename "$canonical")" == "$expected_file" ]] || {
        echo "$id artifact basename differs from the qualified manifest" >&2
        exit 2
    }
    actual_bytes=$(stat -f '%z' "$canonical" 2>/dev/null || stat -c '%s' "$canonical")
    [[ "$actual_bytes" == "$expected_bytes" ]] || {
        echo "$id artifact byte length differs from the qualified manifest" >&2
        exit 2
    }
    [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || exit 2
    printf -v "$path_env" '%s' "$canonical"
    export "${path_env?}"
    gguf_siblings=$(find "$(dirname "$canonical")" -maxdepth 1 -type f \
      -iname '*.gguf' -print | awk 'END { print NR + 0 }')
    [[ "$gguf_siblings" == 1 ]] || {
        echo "$id text-only swap artifact must be isolated from projector/other GGUF siblings" >&2
        exit 2
    }
    artifact_snapshots+=("$(stat -f '%d:%i:%z:%m:%c' "$canonical" 2>/dev/null \
      || stat -c '%d:%i:%s:%Y:%Z' "$canonical")")
done < <(jq -r '.artifacts[] | [.id,.path_env,.file,.bytes,.sha256] | @tsv' "$MANIFEST")

# Distinct paths are insufficient: reject hardlink aliases by physical
# device/inode identity before the model process starts.
physical_identities=()
while IFS= read -r path_env; do
    path=${!path_env}
    physical_identities+=("$(stat -f '%d:%i' "$path" 2>/dev/null \
      || stat -c '%d:%i' "$path")")
done < <(jq -r '.artifacts[].path_env' "$MANIFEST")
[[ "$(printf '%s\n' "${physical_identities[@]}" | sort -u | awk 'END { print NR }')" == 4 ]] || {
    echo "generative swap artifacts must be four distinct physical files" >&2
    exit 2
}

# Build one binary with the exact source commit embedded, or consume a sealed
# candidate supplied by the release workflow.
if [[ -n "${HF2Q_GENERATIVE_SWAP_BINARY:-}" ]]; then
    binary=$HF2Q_GENERATIVE_SWAP_BINARY
    [[ "$binary" == /* ]] || {
        echo "HF2Q_GENERATIVE_SWAP_BINARY must be absolute" >&2
        exit 2
    }
else
    (cd "$ROOT_DIR" \
      && CARGO_HOME="$GATE_CARGO_HOME" GIT_COMMIT_SHA="$source_commit" \
        cargo build --release --locked --bin hf2q)
    target_dir=$(cd "$ROOT_DIR" \
      && CARGO_HOME="$GATE_CARGO_HOME" \
        cargo metadata --locked --no-deps --format-version 1 | jq -er .target_directory)
    binary="$target_dir/release/hf2q"
fi
[[ -f "$binary" && -x "$binary" && ! -L "$binary" ]] || {
    echo "release hf2q binary is missing or not a regular executable" >&2
    exit 1
}
# `cargo test` may replace target/release/hf2q even when the caller supplied
# that exact pathname. Execute and attest a private copy so harness compilation
# cannot change the candidate after its digest is captured.
sealed_binary_dir=$(mktemp -d "${TMPDIR:-/tmp}/hf2q-generative-swap-bin.XXXXXX")
sealed_binary="$sealed_binary_dir/hf2q"
cleanup_sealed_binary() {
    case "$sealed_binary_dir/" in
        "${TMPDIR:-/tmp}"/hf2q-generative-swap-bin.*/*)
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
binary_sha=$(hf2q_generative_swap_sha256_file "$binary")
build_info=$("$binary" __build-info)
binary_git_commit=$(jq -er \
  'select(.schema == "hf2q.build-info.v1") | .git_commit' <<<"$build_info")
[[ "$binary_git_commit" == "$source_commit" ]] || {
    echo "hf2q binary does not embed exact source commit $source_commit" >&2
    exit 1
}

mkdir -p "$OUT_DIR"
# Resolve all four immutable artifacts through the same schema-v2 one-hash
# authority consumed by the production server. A stale or partial authority
# fails before any model becomes resident; a switch never pays this cost after
# the previous engine has been drained.
mkdir -p "$OUT_DIR/preflight"
export HF2Q_MODEL_VERIFICATION_BINARY="$binary"
while IFS=$'\t' read -r id path_env _expected_file _expected_bytes expected_sha; do
    path=${!path_env}
    hf2q_release_prepare_model_verification "$path" "$expected_sha" \
      "$OUT_DIR/preflight/$id.json" \
      "${HF2Q_MODEL_VERIFICATION_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME:?HOME is required}/.cache}/hf2q/model-verification}"
done < <(jq -r '.artifacts[] | [.id,.path_env,.file,.bytes,.sha256] | @tsv' "$MANIFEST")
hf2q_validate_generative_swap_preflight "$OUT_DIR/preflight" "$MANIFEST" || {
    echo "cross-family preverified identity authority is invalid" >&2
    exit 1
}

runtime_receipt="$OUT_DIR/runtime.json"
runtime_log="$OUT_DIR/runtime.log"
chain_spec=$(jq -c '
  {load_budget_seconds,sequence,
   artifacts:[.artifacts[] | . + {path:env[.path_env]} | del(.path_env)]}
' "$MANIFEST")
(
  cd "$ROOT_DIR"
  HF2Q_HOT_SWAP_E2E=1 \
  HF2Q_GENERATIVE_SWAP_CHAIN_RECEIPT="$runtime_receipt" \
  HF2Q_GENERATIVE_SWAP_CHAIN_SPEC="$chain_spec" \
  HF2Q_HOT_SWAP_EXACT_SOURCE_COMMIT="$source_commit" \
  HF2Q_HOT_SWAP_EXACT_BINARY_SHA256="$binary_sha" \
  HF2Q_HOT_SWAP_EXACT_BINARY_GIT_COMMIT="$binary_git_commit" \
  HF2Q_HOT_SWAP_EXACT_MLX_VERSION="$dependency_version" \
  HF2Q_HOT_SWAP_EXACT_MLX_SOURCE="$dependency_source" \
  HF2Q_HOT_SWAP_EXACT_MLX_CHECKSUM="$dependency_checksum" \
  HF2Q_HOT_SWAP_EXECUTABLE="$binary" \
  HF2Q_MODEL_VERIFICATION_RECEIPT_DIR="$OUT_DIR/preflight" \
  HF2Q_Q5K_CANONICAL_Q4X4=1 \
  MLX_DISP_BUCKET=1 \
  CARGO_HOME="$GATE_CARGO_HOME" \
  GIT_COMMIT_SHA="$source_commit" \
    cargo test --release --locked --test multi_model_swap \
      generative_cross_family_two_cycle_e2e \
      -- --exact --nocapture --test-threads=1 >"$runtime_log" 2>&1
)

[[ "$(hf2q_generative_swap_sha256_file "$binary")" == "$binary_sha" ]] || {
    echo "hf2q binary changed during generative swap gate" >&2
    exit 1
}
hf2q_validate_generative_swap_receipt "$runtime_receipt" "$MANIFEST" \
  "$source_commit" "$binary_sha" "$dependency_version" "$dependency_source" \
  "$dependency_checksum" || {
    echo "cross-family execution receipt failed independent validation" >&2
    exit 1
}

index=0
while IFS=$'\t' read -r id path_env; do
    path=${!path_env}
    current=$(stat -f '%d:%i:%z:%m:%c' "$path" 2>/dev/null \
      || stat -c '%d:%i:%s:%Y:%Z' "$path")
    [[ "$current" == "${artifact_snapshots[$index]}" ]] || {
        echo "qualified $id artifact changed during generative swap gate" >&2
        exit 1
    }
    index=$((index + 1))
done < <(jq -r '.artifacts[] | [.id,.path_env] | @tsv' "$MANIFEST")

runner_sha=$(hf2q_generative_swap_sha256_file "$ROOT_DIR/scripts/run_generative_swap_matrix.sh")
contract_sha=$(hf2q_generative_swap_sha256_file "$ROOT_DIR/scripts/generative_swap_matrix_contract.sh")
manifest_sha=$(hf2q_generative_swap_sha256_file "$MANIFEST")
runtime_sha=$(hf2q_generative_swap_sha256_file "$runtime_receipt")
jq \
  --arg runner_sha "$runner_sha" --arg contract_sha "$contract_sha" \
  --arg manifest_sha "$manifest_sha" --arg runtime_sha "$runtime_sha" '
  . + {evidence:{
    runner_sha256:$runner_sha,
    contract_sha256:$contract_sha,
    manifest_sha256:$manifest_sha,
    runtime_receipt_sha256:$runtime_sha
  }}
' "$runtime_receipt" >"$OUT_DIR/matrix.json.tmp"
mv "$OUT_DIR/matrix.json.tmp" "$OUT_DIR/matrix.json"

{
  printf '%s  runtime.json\n%s  runtime.log\n' \
    "$(hf2q_generative_swap_sha256_file "$runtime_receipt")" \
    "$(hf2q_generative_swap_sha256_file "$runtime_log")"
  while IFS= read -r id; do
      receipt="preflight/$id.json"
      printf '%s  %s\n' \
        "$(hf2q_generative_swap_sha256_file "$OUT_DIR/$receipt")" "$receipt"
  done < <(jq -r '.artifacts[].id' "$MANIFEST")
} | sort >"$OUT_DIR/evidence.sha256"
printf '%s  matrix.json\n%s  evidence.sha256\n' \
  "$(hf2q_generative_swap_sha256_file "$OUT_DIR/matrix.json")" \
  "$(hf2q_generative_swap_sha256_file "$OUT_DIR/evidence.sha256")" \
  >"$OUT_DIR/result.sha256"

hf2q_validate_generative_swap_seal "$OUT_DIR/matrix.json" "$MANIFEST" "$ROOT_DIR" || {
    echo "cross-family generative swap evidence seal is invalid" >&2
    exit 1
}
printf 'cross-family generative swap gate passed: %s\n' "$OUT_DIR/matrix.json"
