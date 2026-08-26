#!/usr/bin/env bash
set -euo pipefail

manifest=${1:?manifest is required}
manifest_sidecar=${2:?manifest sidecar is required}
matrix=${3:?exact swap matrix is required}
expected_source_commit=${4:?expected source commit is required}
expected_binary_sha256=${5:?expected binary SHA-256 is required}
expected_dependency_version=${6:?expected dependency version is required}
expected_dependency_source=${7:?expected dependency source is required}
expected_dependency_checksum=${8:?expected dependency checksum is required}
source_root=${9:?exact source root is required}

[[ "$source_root" == /* && "$manifest" == /* && "$matrix" == /* \
  && "$expected_source_commit" =~ ^[0-9a-f]{40}$ \
  && "$expected_binary_sha256" =~ ^[0-9a-f]{64}$ \
  && "$expected_dependency_checksum" =~ ^[0-9a-f]{64}$ \
  && -f "$source_root/scripts/qwen38_artifact_contract.sh" \
  && ! -L "$source_root/scripts/qwen38_artifact_contract.sh" \
  && -f "$source_root/scripts/qwen38_exact_swap_matrix_contract.sh" \
  && ! -L "$source_root/scripts/qwen38_exact_swap_matrix_contract.sh" \
  && "$(git -C "$source_root" rev-parse HEAD)" == "$expected_source_commit" \
  && -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" \
  && -f "$manifest" && -r "$manifest" && ! -L "$manifest" \
  && -f "$manifest_sidecar" && -r "$manifest_sidecar" \
  && ! -L "$manifest_sidecar" \
  && -f "$matrix" && -r "$matrix" && ! -L "$matrix" ]] || exit 1

# The validator and artifact catalog must come from the exact checkout whose
# commit is embedded in the candidate binary, not from an unrelated package.
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$source_root/scripts/qwen38_artifact_contract.sh"
# shellcheck source=scripts/qwen38_exact_swap_matrix_contract.sh
source "$source_root/scripts/qwen38_exact_swap_matrix_contract.sh"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

[[ "$manifest_sidecar" == "$manifest.sha256" \
  && "$(awk 'END { print NR }' "$manifest_sidecar")" == 1 \
  && "$(sed -n '1p' "$manifest_sidecar")" \
    == "$(sha256_file "$manifest")  $(basename "$manifest")" ]] || exit 1
(cd "$(dirname "$manifest")" \
  && shasum -a 256 -c "$(basename "$manifest_sidecar")" >/dev/null)

qwen38_validate_exact_swap_seal "$matrix" "$source_root"
matrix_sha=$(sha256_file "$matrix")
jq -e \
  --arg source_commit "$expected_source_commit" \
  --arg binary_sha "$expected_binary_sha256" \
  --arg dependency_version "$expected_dependency_version" \
  --arg dependency_source "$expected_dependency_source" \
  --arg dependency_checksum "$expected_dependency_checksum" \
  --arg matrix_sha "$matrix_sha" --slurpfile matrix "$matrix" '
  .status == "pass"
  and .source_sha == $source_commit
  and .binary_sha256 == $binary_sha
  and .receipt_sha256.qwen38.exact_swap == $matrix_sha
  and .families.qwen38.status == "pass"
  and .families.qwen38.exact_swap == $matrix[0]
  and .dependency_provenance.dependency
    == {name:"mlx-native",version:$dependency_version,
      source:$dependency_source,checksum:$dependency_checksum}
  and $matrix[0].source_commit == $source_commit
  and $matrix[0].binary.sha256 == $binary_sha
  and $matrix[0].binary.git_commit == $source_commit
  and $matrix[0].dependency == {name:"mlx-native",version:$dependency_version,
    source:$dependency_source,checksum:$dependency_checksum}
' "$manifest" >/dev/null
