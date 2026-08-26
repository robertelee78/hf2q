#!/usr/bin/env bash
set -euo pipefail

# Cross-shape ADR-049 authority for Qwen rectangular prefill. The dense/MTP
# and MoE/no-MTP cells use the same exact clean hf2q binary. Each architecture
# executes the pure rectangular ABBA gate, the agentic lifecycle/tool gate,
# and the missing eight-slot live-decoder-plus-four-prefill Mixed gate.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
SOURCE_ROOT=${SOURCE_ROOT:-$root_dir}
HF2Q_BIN=${HF2Q_BIN:-$SOURCE_ROOT/target/release/hf2q}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
QWEN38_MODEL_PATH=${QWEN38_MODEL_PATH:-/opt/hf2q/models/qwen3.8/hub/gguf/qwen38-abliterated-sft-q5_k_m.gguf}
QWEN36_MODEL_PATH=${QWEN36_MODEL_PATH:-/opt/hf2q/models/qwen3.6/APEX-Q5_K_M.gguf}
QWEN38_PORT=${QWEN38_PORT:-52849}
QWEN36_PORT=${QWEN36_PORT:-52850}
QWEN38_LIFECYCLE_PORT=${QWEN38_LIFECYCLE_PORT:-52851}
QWEN36_LIFECYCLE_PORT=${QWEN36_LIFECYCLE_PORT:-52852}
QWEN38_MIXED_PORT=${QWEN38_MIXED_PORT:-52853}
QWEN36_MIXED_PORT=${QWEN36_MIXED_PORT:-52854}
readonly QWEN38_MODEL_SHA256=4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e
readonly QWEN38_MODEL_BYTES=19535701568
readonly QWEN36_MODEL_SHA256=f2c702182a4661d2cef573b388ff23336ce65aabb112762d1c1a24d4ba0cbc25
readonly QWEN36_MODEL_BYTES=25043007488

for command in awk find git jq mkdir mv shasum; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ "$SOURCE_ROOT" == /* && "$HF2Q_BIN" == /* && "$OUT_DIR" == /* \
    && "$QWEN38_MODEL_PATH" == /* && "$QWEN36_MODEL_PATH" == /* ]] || {
    echo "all source, binary, model, and output paths must be absolute" >&2
    exit 2
}
[[ ! -e "$OUT_DIR" \
    || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "OUT_DIR must be fresh: $OUT_DIR" >&2
    exit 2
}
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd -P)

SOURCE_ROOT="$SOURCE_ROOT" HF2Q_BIN="$HF2Q_BIN" \
MODEL_PATH="$QWEN38_MODEL_PATH" MODEL_SHA256="$QWEN38_MODEL_SHA256" \
MODEL_BYTES="$QWEN38_MODEL_BYTES" MODEL_SHAPE=qwen38-dense \
OUT_DIR="$OUT_DIR/qwen38-dense" PORT="$QWEN38_PORT" \
    "$script_dir/bench_qwen35_rectangular_policy_abba.sh"

SOURCE_ROOT="$SOURCE_ROOT" HF2Q_BIN="$HF2Q_BIN" \
MODEL_PATH="$QWEN36_MODEL_PATH" MODEL_SHA256="$QWEN36_MODEL_SHA256" \
MODEL_BYTES="$QWEN36_MODEL_BYTES" MODEL_SHAPE=qwen36-moe \
OUT_DIR="$OUT_DIR/qwen36-moe" PORT="$QWEN36_PORT" \
    "$script_dir/bench_qwen35_rectangular_policy_abba.sh"

SOURCE_ROOT="$SOURCE_ROOT" HF2Q_BIN="$HF2Q_BIN" \
MODEL_PATH="$QWEN38_MODEL_PATH" MODEL_SHA256="$QWEN38_MODEL_SHA256" \
MODEL_BYTES="$QWEN38_MODEL_BYTES" MODEL_SHAPE=qwen38-dense \
OUT_DIR="$OUT_DIR/qwen38-lifecycle" PORT="$QWEN38_LIFECYCLE_PORT" \
    "$script_dir/run_qwen35_agentic_lifecycle_cell.sh"

SOURCE_ROOT="$SOURCE_ROOT" HF2Q_BIN="$HF2Q_BIN" \
MODEL_PATH="$QWEN36_MODEL_PATH" MODEL_SHA256="$QWEN36_MODEL_SHA256" \
MODEL_BYTES="$QWEN36_MODEL_BYTES" MODEL_SHAPE=qwen36-moe \
OUT_DIR="$OUT_DIR/qwen36-lifecycle" PORT="$QWEN36_LIFECYCLE_PORT" \
    "$script_dir/run_qwen35_agentic_lifecycle_cell.sh"

SOURCE_ROOT="$SOURCE_ROOT" HF2Q_BIN="$HF2Q_BIN" \
MODEL_PATH="$QWEN38_MODEL_PATH" MODEL_SHA256="$QWEN38_MODEL_SHA256" \
MODEL_BYTES="$QWEN38_MODEL_BYTES" MODEL_SHAPE=qwen38-dense \
OUT_DIR="$OUT_DIR/qwen38-mixed" PORT="$QWEN38_MIXED_PORT" \
    "$script_dir/bench_qwen35_mixed_rectangular_cell.sh"

SOURCE_ROOT="$SOURCE_ROOT" HF2Q_BIN="$HF2Q_BIN" \
MODEL_PATH="$QWEN36_MODEL_PATH" MODEL_SHA256="$QWEN36_MODEL_SHA256" \
MODEL_BYTES="$QWEN36_MODEL_BYTES" MODEL_SHAPE=qwen36-moe \
OUT_DIR="$OUT_DIR/qwen36-mixed" PORT="$QWEN36_MIXED_PORT" \
    "$script_dir/bench_qwen35_mixed_rectangular_cell.sh"

qwen38_receipt="$OUT_DIR/qwen38-dense/receipt.json"
qwen36_receipt="$OUT_DIR/qwen36-moe/receipt.json"
qwen38_lifecycle_receipt="$OUT_DIR/qwen38-lifecycle/receipt.json"
qwen36_lifecycle_receipt="$OUT_DIR/qwen36-lifecycle/receipt.json"
qwen38_mixed_receipt="$OUT_DIR/qwen38-mixed/receipt.json"
qwen36_mixed_receipt="$OUT_DIR/qwen36-mixed/receipt.json"
"$script_dir/test_qwen35_rectangular_policy_receipt_mutations.sh" \
    "$qwen38_receipt" "$SOURCE_ROOT" >"$OUT_DIR/qwen38-mutations.log"
"$script_dir/test_qwen35_rectangular_policy_receipt_mutations.sh" \
    "$qwen36_receipt" "$SOURCE_ROOT" >"$OUT_DIR/qwen36-mutations.log"
jq -e '
  .schema == 1 and .verdict == "pass"
  and .gate == "qwen35-rectangular-policy-abba"
  and .model.shape == "qwen38-dense"
  and .workload.same_binary == true
  and .workload.speculation == "auto"
  and .result.wave_speedup >= .thresholds.min_wave_speedup
  and .result.single_max_matched_overhead_ms <= .thresholds.max_single_overhead_ms
' "$qwen38_receipt" >/dev/null
jq -e '
  .schema == 1 and .verdict == "pass"
  and .gate == "qwen35-rectangular-policy-abba"
  and .model.shape == "qwen36-moe"
  and .workload.same_binary == true
  and .workload.speculation == "auto"
  and .result.wave_speedup >= .thresholds.min_wave_speedup
  and .result.single_max_matched_overhead_ms <= .thresholds.max_single_overhead_ms
' "$qwen36_receipt" >/dev/null
"$script_dir/verify_qwen35_agentic_lifecycle_cell.sh" \
    "$qwen38_lifecycle_receipt" "$SOURCE_ROOT"
"$script_dir/verify_qwen35_agentic_lifecycle_cell.sh" \
    "$qwen36_lifecycle_receipt" "$SOURCE_ROOT"
"$script_dir/test_qwen35_agentic_lifecycle_receipt_mutations.sh" \
    "$qwen38_lifecycle_receipt" "$SOURCE_ROOT" \
    >"$OUT_DIR/qwen38-lifecycle-mutation.log"
"$script_dir/test_qwen35_agentic_lifecycle_receipt_mutations.sh" \
    "$qwen36_lifecycle_receipt" "$SOURCE_ROOT" \
    >"$OUT_DIR/qwen36-lifecycle-mutation.log"
"$script_dir/verify_qwen35_mixed_rectangular_receipt.sh" \
    "$qwen38_mixed_receipt" "$SOURCE_ROOT"
"$script_dir/verify_qwen35_mixed_rectangular_receipt.sh" \
    "$qwen36_mixed_receipt" "$SOURCE_ROOT"
"$script_dir/test_qwen35_mixed_rectangular_receipt_mutations.sh" \
    "$qwen38_mixed_receipt" "$SOURCE_ROOT" \
    >"$OUT_DIR/qwen38-mixed-mutation.log"
"$script_dir/test_qwen35_mixed_rectangular_receipt_mutations.sh" \
    "$qwen36_mixed_receipt" "$SOURCE_ROOT" \
    >"$OUT_DIR/qwen36-mixed-mutation.log"

source_commit=$(jq -er '.source.commit' "$qwen38_receipt")
binary_sha256=$(jq -er '.source.sha256' "$qwen38_receipt")
[[ "$(jq -er '.source.commit' "$qwen36_receipt")" == "$source_commit" \
    && "$(jq -er '.source.sha256' "$qwen36_receipt")" == "$binary_sha256" \
    && "$(jq -er '.source.commit' "$qwen38_lifecycle_receipt")" == "$source_commit" \
    && "$(jq -er '.source.sha256' "$qwen38_lifecycle_receipt")" == "$binary_sha256" \
    && "$(jq -er '.source.commit' "$qwen36_lifecycle_receipt")" == "$source_commit" \
    && "$(jq -er '.source.sha256' "$qwen36_lifecycle_receipt")" == "$binary_sha256" \
    && "$(jq -er '.source.commit' "$qwen38_mixed_receipt")" == "$source_commit" \
    && "$(jq -er '.source.sha256' "$qwen38_mixed_receipt")" == "$binary_sha256" \
    && "$(jq -er '.source.commit' "$qwen36_mixed_receipt")" == "$source_commit" \
    && "$(jq -er '.source.sha256' "$qwen36_mixed_receipt")" == "$binary_sha256" \
    && "$(jq -er '.model.shape' "$qwen38_lifecycle_receipt")" == \
        "$(jq -er '.model.shape' "$qwen38_receipt")" \
    && "$(jq -er '.model.sha256' "$qwen38_lifecycle_receipt")" == \
        "$(jq -er '.model.sha256' "$qwen38_receipt")" \
    && "$(jq -er '.model.shape' "$qwen36_lifecycle_receipt")" == \
        "$(jq -er '.model.shape' "$qwen36_receipt")" \
    && "$(jq -er '.model.sha256' "$qwen36_lifecycle_receipt")" == \
        "$(jq -er '.model.sha256' "$qwen36_receipt")" \
    && "$(jq -er '.model.shape' "$qwen38_mixed_receipt")" == \
        "$(jq -er '.model.shape' "$qwen38_receipt")" \
    && "$(jq -er '.model.sha256' "$qwen38_mixed_receipt")" == \
        "$(jq -er '.model.sha256' "$qwen38_receipt")" \
    && "$(jq -er '.model.shape' "$qwen36_mixed_receipt")" == \
        "$(jq -er '.model.shape' "$qwen36_receipt")" \
    && "$(jq -er '.model.sha256' "$qwen36_mixed_receipt")" == \
        "$(jq -er '.model.sha256' "$qwen36_receipt")" \
    && "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$source_commit" \
    && -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" \
    && "$(shasum -a 256 "$HF2Q_BIN" | awk '{print $1}')" == "$binary_sha256" ]] || {
    echo "rectangular matrix cells do not share one exact clean binary" >&2
    exit 1
}

matrix_tmp="$OUT_DIR/.matrix.json.tmp.$$"
jq -n --arg source_commit "$source_commit" --arg binary "$HF2Q_BIN" \
    --arg binary_sha256 "$binary_sha256" \
    --arg runner_sha256 "$(shasum -a 256 "$script_dir/bench_qwen35_rectangular_policy_abba.sh" | awk '{print $1}')" \
    --arg matrix_runner_sha256 "$(shasum -a 256 "$script_dir/bench_qwen35_rectangular_policy_matrix.sh" | awk '{print $1}')" \
    --arg qwen38_receipt_sha256 "$(shasum -a 256 "$qwen38_receipt" | awk '{print $1}')" \
    --arg qwen36_receipt_sha256 "$(shasum -a 256 "$qwen36_receipt" | awk '{print $1}')" \
    --arg qwen38_lifecycle_sha256 "$(shasum -a 256 "$qwen38_lifecycle_receipt" | awk '{print $1}')" \
    --arg qwen36_lifecycle_sha256 "$(shasum -a 256 "$qwen36_lifecycle_receipt" | awk '{print $1}')" \
    --arg qwen38_lifecycle_mutation_sha256 "$(shasum -a 256 "$OUT_DIR/qwen38-lifecycle-mutation.log" | awk '{print $1}')" \
    --arg qwen36_lifecycle_mutation_sha256 "$(shasum -a 256 "$OUT_DIR/qwen36-lifecycle-mutation.log" | awk '{print $1}')" \
    --arg mixed_runner_sha256 "$(shasum -a 256 "$script_dir/bench_qwen35_mixed_rectangular_cell.sh" | awk '{print $1}')" \
    --arg mixed_verifier_sha256 "$(shasum -a 256 "$script_dir/verify_qwen35_mixed_rectangular_receipt.sh" | awk '{print $1}')" \
    --arg qwen38_mixed_sha256 "$(shasum -a 256 "$qwen38_mixed_receipt" | awk '{print $1}')" \
    --arg qwen36_mixed_sha256 "$(shasum -a 256 "$qwen36_mixed_receipt" | awk '{print $1}')" \
    --arg qwen38_mixed_mutation_sha256 "$(shasum -a 256 "$OUT_DIR/qwen38-mixed-mutation.log" | awk '{print $1}')" \
    --arg qwen36_mixed_mutation_sha256 "$(shasum -a 256 "$OUT_DIR/qwen36-mixed-mutation.log" | awk '{print $1}')" \
    --arg qwen38_mutations_sha256 "$(shasum -a 256 "$OUT_DIR/qwen38-mutations.log" | awk '{print $1}')" \
    --arg qwen36_mutations_sha256 "$(shasum -a 256 "$OUT_DIR/qwen36-mutations.log" | awk '{print $1}')" \
    --slurpfile qwen38 "$qwen38_receipt" --slurpfile qwen36 "$qwen36_receipt" \
    --slurpfile qwen38_mixed "$qwen38_mixed_receipt" \
    --slurpfile qwen36_mixed "$qwen36_mixed_receipt" '{
      schema:1,verdict:"pass",gate:"qwen35-rectangular-policy-matrix",
      source:{commit:$source_commit,binary:$binary,sha256:$binary_sha256},
      evidence:{runner_sha256:$runner_sha256,
        matrix_runner_sha256:$matrix_runner_sha256,
        cells:{qwen38_dense_receipt_sha256:$qwen38_receipt_sha256,
          qwen36_moe_receipt_sha256:$qwen36_receipt_sha256},
        lifecycle:{qwen38_dense_receipt_sha256:$qwen38_lifecycle_sha256,
          qwen36_moe_receipt_sha256:$qwen36_lifecycle_sha256,
          qwen38_mutation_sha256:$qwen38_lifecycle_mutation_sha256,
          qwen36_mutation_sha256:$qwen36_lifecycle_mutation_sha256},
        mixed:{runner_sha256:$mixed_runner_sha256,
          verifier_sha256:$mixed_verifier_sha256,
          qwen38_dense_receipt_sha256:$qwen38_mixed_sha256,
          qwen36_moe_receipt_sha256:$qwen36_mixed_sha256,
          qwen38_mutation_sha256:$qwen38_mixed_mutation_sha256,
          qwen36_mutation_sha256:$qwen36_mixed_mutation_sha256},
        mutation_battery:{qwen38_dense_sha256:$qwen38_mutations_sha256,
          qwen36_moe_sha256:$qwen36_mutations_sha256}},
      cells:[$qwen38[0],$qwen36[0]],
      mixed_cells:[$qwen38_mixed[0],$qwen36_mixed[0]]
    }' >"$matrix_tmp"
mv "$matrix_tmp" "$OUT_DIR/matrix.json"
jq . "$OUT_DIR/matrix.json"
echo "matrix receipt: $OUT_DIR/matrix.json" >&2
