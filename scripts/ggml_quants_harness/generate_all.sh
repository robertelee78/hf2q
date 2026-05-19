#!/usr/bin/env bash
# Generate ADR-033 P0 reference fixtures for all 11 v1 ggml_quants types.
#
# Output: tests/fixtures/ggml_quants/<type>_<n_per_row>_<variant>_{input,expected}.bin
# Variants: noim (no imatrix), im (imatrix on; required for IQ-family).
#
# Deterministic seeds: input_seed=1, imatrix_seed=2 (constants — change is an
# ADR amendment because the fixtures get regenerated).

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
GEN="$HERE/gen"
[ -x "$GEN" ] || { echo "harness not built; run build.sh first"; exit 1; }

REPO=$(cd "$HERE/../.." && pwd)
FIX="$REPO/tests/fixtures/ggml_quants"
mkdir -p "$FIX"

INPUT_SEED=1
IMATRIX_SEED=2

# (type, n_per_row, n_rows)
# n_per_row picks: legacy types use 64 (= 2 blocks of 32); K-family uses 512 (= 2 blocks of 256).
# n_rows = 2 to exercise the inner stride (start, nrows, n_per_row contract).
cases=(
    "q4_0    64  2"
    "q4_1    64  2"
    "q5_0    64  2"
    "q5_1    64  2"
    "q8_0    64  2"
    "iq4_nl  64  2"
    "q2_k    512 2"
    "q3_k    512 2"
    "q4_k    512 2"
    "q5_k    512 2"
    "q6_k    512 2"
)

for c in "${cases[@]}"; do
    read -r ttype npr nr <<<"$c"
    # noim variant (no imatrix). Skip if the type structurally requires imatrix.
    if "$GEN" "$ttype" "$npr" "$nr" "$INPUT_SEED" none \
        "$FIX/${ttype}_${npr}_noim_input.bin" \
        "$FIX/${ttype}_${npr}_noim_expected.bin" 2>/dev/null; then
        echo "  noim: $ttype n=$npr"
    else
        echo "  noim: $ttype n=$npr SKIPPED (requires imatrix)"
    fi

    # im variant (imatrix on)
    "$GEN" "$ttype" "$npr" "$nr" "$INPUT_SEED" "$IMATRIX_SEED" \
        "$FIX/${ttype}_${npr}_im_input.bin" \
        "$FIX/${ttype}_${npr}_im_expected.bin"
    echo "  im:   $ttype n=$npr"
done

echo ""
echo "Fixtures at $FIX:"
ls -la "$FIX" | grep -E '\.bin$' | awk '{print $5, $9}'
